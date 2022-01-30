#!/usr/bin/env python
# coding: utf-8

# ## Stack the images in fits files.
# ### Define input parameters

# In[1]:


import os, imageio, cv2, time, sys, matplotlib

import scipy.fft as fft
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time as astro_time
import astropy.units as u


# Working directory, all raw or fits files should be in this directory
working_dir = "/work/astro/fits"

# Name of the final image
final_file = "final.tiff"

# Working precision takes effect in FFT and matrix multiplication
working_precision = "float64"

# Define the input file extension. All files in the working directory with this extension will be used.
extension = "fit"

# page number of data in the fits file
page_num = 0

# tag for the obs. date and time string (for fits file)
date_tag = 'DATE-OBS'

# define the Bayer matrix format, only for the fits file
bayer_matrix_format = cv2.COLOR_BayerBG2RGB

# if true, work in console mode, will not process the Jupyter notebook code and will not produce online images.
console = True

# define the maximum number of processes to be used
nproc_max = 96

# If true, do not report the alignment result
less_report = True

# fraction of frames that will not be used
bad_fraction = 0.3

# dark_fac means the fraction of pixels that will be ignored as "too dark".
dark_frac = 5e-4

# bright_frac means the fraction of pixels that will be ignored as "too bright".
bright_frac = 5e-4

# The red, green and blue pixels are amplified by this factor for a custom white balance.
color_amp_fac = [1.05, 0.90, 1.05]

# final gamma
gamma = [0.5,0.5,0.5]

# save aligned binary files or not. Note that for multiprocessing, this must be True
save_aligned_binary = False

# save aligned images?
save_aligned_image = False

# number of ADC digit. The true maximum value should be 2**adc_digit
adc_digit_max = 16


# ### Change the width of notebook for better view and define subroutines.

# In[2]:


##############################################################
#
# read fits file and convert to bin so it can be used by multiprocessing returns: frame, time
#
def read_frame_fits(file):
    with fits.open(file) as hdu:
        frame = hdu[page_num].data
        raw_data_type = frame.dtype
        hdr = hdu[page_num].header
        date_str = hdr[date_tag]
        jd = astro_time(date_str).to_value(format="jd")
    return frame, jd, raw_data_type


#
# use the fits information to convert a frame to rgb image
#
def frame2rgb_fits(frame):
    rgb = cv2.cvtColor(frame.astype(raw_data_type), bayer_matrix_format)
    return rgb


# 
# align a frame to the reference frame
# 
def align_frames(i):
    tst = time.time()
    # read the raw data as an object, obtain the image and compute its fft
    frame = np.fromfile(file_swp[i], dtype=raw_data_type).reshape(n1, n2)
    frame_fft = fft.fft2(frame.astype(working_precision))
    
    # compute the frame offset
    cache = np.abs(fft.ifft2(ref_fft*frame_fft))
    index = np.unravel_index(np.argmax(cache, axis=None), cache.shape)
    s1, s2 = -index[0], -index[1]
    
    # make sure that the Bayer matrix will not be corrupted
    s1 = s1 - np.mod(s1, 2)
    s2 = s2 - np.mod(s2, 2)
    
    # fix the offset and save into the result array
    frame = np.roll(frame, (s1, s2), axis=(0,1))
    
    # save the aligned images and binaries if necessary
    frame.tofile(file_bin[i])
    
    if save_aligned_image is True:
        raw.raw_image[:,:] = frame
        rgb = raw.postprocess(use_camera_wb=True)
        imageio.imsave(file_tif[i], rgb)
    if not less_report:
        print("\nFrame %6i (%s) aligned in %8.2f sec, (sx, sy) = (%8i,%8i)." 
              %(i, file_lst[i], time.time()-tst, s1, s2))
    return i, s1, s2


def compute_weights(frames_working):
    # read the alignment results of multiple processes
    tst = time.time()
    for i in range(n_files):
        frame = np.fromfile(file_bin[i], dtype=raw_data_type)
        frames_working[i,:,:] = frame.reshape(n1, n2)
        if save_aligned_binary is False:
            os.remove(file_bin[i])
    print("Aligned frames read in. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()

    # remove the mean value from each frame
    tst = time.time()
    for i in range(0, n_files):
        frames_working[i,:,:] = frames_working[i,:,:] - np.mean(frames_working[i,:,:])
    print("Mean values of frames removed. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    
    # compute the covariance matrix
    frames_working = frames_working.reshape(n_files, n1*n2)
    cov = np.dot(frames_working, frames_working.transpose())

    # compute weights from the covariance matrix
    w = np.zeros(n_files)
    for i in range(n_files):
        w[i] = np.sum(cov[i,:])/cov[i,i] - 1

    return w


import heapq as hp
# subroutine for adjusting the colors 
def adjust_color(i, m1, m2, npix, bin_file, raw_data_type):
    # number of "too dark" pixels and threshold
    samp = np.fromfile(bin_file, dtype=raw_data_type).reshape(m1*m2)
    # samp = samp.reshape(m1*m2)
    ndark = int(npix*dark_frac)
    d1 = hp.nsmallest(ndark, samp)[ndark-1]*1.
    # number of "too bright" pixels and threshold
    nbright = int(npix*bright_frac)
    d2 = hp.nlargest(nbright, samp)[nbright-1]*1.
    # rescaling, note that this requires 16-bit to save weak signal from bright sky-light
    # note that val is expected to be in range [0,1]. Out-of-range values will be truncated.
    val = (samp-d1)/(d2-d1)
    val = np.where(val<0, 0, val)
    val = np.where(val>1, 1, val)
    samp = (val**gamma[i])*256*color_amp_fac[i]
    samp = np.where(samp<  0,   0, samp)
    samp = np.where(samp>255, 255, samp)
    samp.astype(raw_data_type).tofile(bin_file)


# ## Do the following:
# #### 1. Align the frames using the initial reference frame.
# #### 2. Compute the weights from covarinace matrix.
# #### 3. Set the reference frame to the one with highest weight.
# #### 4. Align the frames again using the new reference frame.
# #### 5. Re-compute the weights from covarinace matrix.
# #### 6. Stack with weights.
# #### 7. Adjust the color

# In[3]:


if console == False:
    # Improve the display effect of Jupyter notebook
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
else:
    # do not produce online images (but will still save pdf)
    matplotlib.use('Agg')


# make a list of woking files and determine the number of processes to be used
os.chdir(working_dir)
file_lst, file_bin, file_swp, file_tif = [], [], [], []
for file in os.listdir():
    if file.endswith(extension):
        file_lst.append(file)
file_lst.sort()
# first sort, and then build auxiliary file lists accordingly
for file in file_lst:
    file_swp.append(os.path.splitext(file)[0] + '.swp')
    file_bin.append(os.path.splitext(file)[0] + '_aligned.bin')
    file_tif.append(os.path.splitext(file)[0] + '_aligned.tif')
n_files = np.int64(len(file_lst))

if nproc_max > n_files:
    nproc = n_files
else:
    nproc = nproc_max


# prepare the reference frame in the Fourier domain
ref_file = file_lst[0]
ref_frame, ref_jd, raw_data_type = read_frame_fits(ref_file) 
ref_fft = np.conjugate(fft.fft2(ref_frame.astype(working_precision)))

n1 = np.int64(np.shape(ref_frame)[0])
n2 = np.int64(np.shape(ref_frame)[1])


# use multiprocessing to work on multiple files' alignment.
frames_working = np.zeros([n_files, n1, n2], dtype=working_precision)
jd = np.zeros(n_files)

if __name__ == '__main__':
    tst = time.time()
    for i in range(n_files):
        frame1, jd1, _ = read_frame_fits(file_lst[i])
        frame1.tofile(file_swp[i])
    print("Frames read and cached, time cost: %9.2f sec." %(time.time()-tst)); tst = time.time()

    tst = time.time()
    with mp.Pool(nproc) as pool:
        output = [pool.map(align_frames, range(n_files))]
    print("Initial alignment done. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    
    # identify the frame of maximum weight, and use it as the new reference frame.
    w = compute_weights(frames_working)
    wid = np.argmax(w)
    ref_file = file_lst[wid]
    ref_frame, ref_jd, _ = read_frame_fits(ref_file) 
    ref_fft = np.conjugate(fft.fft2(ref_frame.astype(working_precision)))
    print("Frame %i chosen as the new reference frame. All frames will be re-aligned." %(wid))
    print("The new reference file is: %s" %(file_lst[wid]))
    
    # work with multiprocessing to align the frames again, and remove the swp files
    tst = time.time()
    with mp.Pool(nproc) as pool:
        output = [pool.map(align_frames, range(n_files))]
    for file in file_swp:
        os.remove(file)
    print("Final alignment done. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    
    # parse and record the offsets
    output_arr = np.array(output)
    sx, sy = output_arr[0,:,1], output_arr[0,:,2]
    sx = np.where(sx >  n1/2, sx-n1, sx)
    sx = np.where(sx < -n1/2, sx+n1, sx)
    sy = np.where(sy >  n2/2, sy-n2, sy)
    sy = np.where(sy < -n2/2, sy+n2, sy)
    print("Alignment done. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    
    # recompute the weights
    tst = time.time()
    w = compute_weights(frames_working)
    # exclude the low quality frames
    n_bad = int(n_files*bad_fraction)
    thr = hp.nsmallest(n_bad, w)[n_bad-1]
    if thr<0: thr = 0
    w = np.where(w <= thr, 0, w)
    w = w / np.sum(w)
    print("Final weights computed. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    
    # plot the weights for test 
    plt.figure(figsize=(4,2),dpi=200)
    plt.title(r'Stacking weights ($w\times N_{frames}$)')
    plt.xlabel('Frame number',fontsize=9)
    plt.ylabel(r'$w\times N_{frames}$',fontsize=9)
    w1 = np.where(w==0, np.nan, w)
    w2 = np.where(w==0, np.median(w), np.nan)
    plt.plot(w1*n_files, marker="o", label='Valid')
    plt.plot(w2*n_files, marker="*", label='Invalid')
    plt.legend()
    plt.savefig('weights.pdf')
    
    # plot the XY-shifts
    plt.figure(figsize=(4,2),dpi=200)
    plt.title('XY shifts in pixel')
    plt.xlabel('Y shifts',fontsize=9)
    plt.ylabel('X shifts',fontsize=9)
    plt.scatter(sx, sy, s=50, alpha=.5)
    plt.savefig('xy-shifts.pdf')

    # stack the frames with weights.
    tst = time.time()
    frame_stacked = np.dot(w, frames_working.reshape(n_files, n1*n2)).reshape(n1, n2)
    # normalize the stacked result to 0-65535.
    fmin = np.amin(frame_stacked)
    fmax = np.amax(frame_stacked)
    cache = (frame_stacked-fmin)/(fmax-fmin)
    tmax = 2.**(adc_digit_max) - 1.
    frame_stacked = np.floor(cache*tmax)
    print("Stacked frame obtained from %i/%i best frames. Time cost: %9.2f" 
        %(n_files-n_bad, n_files, time.time()-tst)); tst = time.time()


    # adjust the color and make the final 8-bit image
    tst = time.time()
    rgb = frame2rgb_fits(frame_stacked)
    r_bin_file = os.path.splitext(final_file)[0] + '.r'; rgb[:,:,0].tofile(r_bin_file)
    g_bin_file = os.path.splitext(final_file)[0] + '.g'; rgb[:,:,1].tofile(g_bin_file)
    b_bin_file = os.path.splitext(final_file)[0] + '.b'; rgb[:,:,2].tofile(b_bin_file)
    
    # color correction in parallel
    m1, m2, npix = np.shape(rgb)[0], np.shape(rgb)[1], rgb.size
    tst = time.time()
    ic = [0, 1, 2]
    im1 = [m1, m1, m1]
    im2 = [m2, m2, m2]
    inp = [npix, npix, npix]
    ifn = [r_bin_file, g_bin_file, b_bin_file]
    dtp = [raw_data_type, raw_data_type, raw_data_type]
    with mp.Pool(3) as pool:
        output = [pool.starmap(adjust_color, zip(ic, im1, im2, inp, ifn, dtp))]

    # read the color correction result and save to 
    rgb[:,:,0] = np.fromfile(r_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(r_bin_file)
    rgb[:,:,1] = np.fromfile(g_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(g_bin_file)
    rgb[:,:,2] = np.fromfile(b_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(b_bin_file)
    print("Color adjusted. Time cost: %8.2f" %(time.time()-tst)); tst = time.time()
    

    # save the final figure
    imageio.imsave(final_file, np.uint8(rgb))

    # show the final figure
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb))

    print("Done!")





