#!/usr/bin/env python
# coding: utf-8

##############################################################
# Stack the images in fits files.
##############################################################


##############################################################
# Define input parameters
##############################################################
import os, cv2, time, sys, matplotlib

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import heapq as hp

from scipy import ndimage
from scipy.signal.windows import tukey
from datetime import datetime
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time as astro_time
from astropy.time import TimezoneInfo
import astropy.units as u

# longitude, latitude, height and time zone of the AHU observatory
# and define the location object of the AHU observatory
ahu_lon = 117.21  # needs to be updated
ahu_lat = 31.84   # needs to be updated
ahu_alt = 63.00   # needs to be updated
ahu_zone = +8
ahu_observatory = EarthLocation(lon=ahu_lon*u.deg, lat=ahu_lat*u.deg, height=ahu_alt*u.m)

# Define observation target, either in name or in ra-dec.
# in name, use (for example): target = SkyCoord.from_name("m31"); 
# in ra-dec, use (for example): target = SkyCoord(ra=45*u.deg, dec=45*u.deg)
target = SkyCoord(ra=83.82208333*u.deg, dec=-5.39111111*u.deg)
# target = SkyCoord.from_name("m42")

# If true, work in console mode, will not process the Jupyter notebook code
# and will not produce online images.
console = True

# Working directory, all raw or fits files should be in this directory. Will
# be overwritten by the command-line parameter.
working_dir = "./fits"

# Define the input file extension. All files in the working directory with
# this extension will be used. Will be overwritten by the command-line
# parameter.
extension = "fits"

# Define the maximum number of processes to be used. Will be overwritten by
# the command-line parameter.
nproc_max = int(mp.cpu_count()/2)

# Fraction of frames that will not be used
bad_fraction = 0.1

output_dir = "output"

# Specify whether or not to fix the field rotation (needs the observation
# time, target and site locations). For an Alt-az mount, this is necessary,
# but for an equatorial mount this is unnecessary.
do_fix_ratation = False

# Page number of data in the fits file
page_num = 0

# Tag for the obs. date and time string (for fits file)
# date_tag = 'DATE-OBS'
date_tag = 'DATE-OBS'

# 2D High-pass cut frequency as a fraction of the Nyquist frequency. Only for
# alignment to reduce background impacts. Will not affect the frames.
highpass_cut = 0.01

# Tukey window alpha parameter, used to improve the matched filtering for
# example, 0.04 means 2% at each edge (left, right, top, bottom) is suppressed
# Note that this should match the maximum shifts
tukey_alpha = 0.2

# sigma of Gaussian filtering (smoothing), only for alignment.
gauss_sigma = 8

# Save aligned binary files or not. Note that for multiprocessing, this must be True
save_aligned_binary = False

# Save aligned images?
save_aligned_image = False

# If true, do not report the alignment result
less_report = True

# Number of ADC digit. The true maximum value should be 2**adc_digit. This should usualLy be 16.
adc_digit_max = 16

vmax_global = 2**adc_digit_max - 1

# field rotation ang file
file_rot_ang = "field_rot_ang.bin"

# Name of the final image
final_file_tif = "final.tiff"

# Name of the final fits
final_file_fits = "frame_stacked.fits"

# Working precision takes effect in FFT and matrix multiplication
working_precision = "float32"











##############################################################
# Define subroutines, no computation here.
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
    return frame, date_str, raw_data_type


# make HDU with property cards
def data2hdu(data, cards=None, primary=False):
    # Make HDU
    if primary:
        hdu = fits.PrimaryHDU(data)
    else:
        hdu = fits.ImageHDU(data)
    # Set property cards.read_fits
    if cards != None:
        hdr = hdu.header
        for card in cards:
            hdr.set(card[0], card[1], card[2])
    return hdu


# write fits file from a list of HDU
def hdu2fits(list_of_hdu, file, overwrite=True):
    hdu_list = fits.HDUList(list_of_hdu)
    hdu_list.writeto(file, overwrite=overwrite)
    hdu_list.close()


# simple program to write array to fits file
def write_fits_simple(list_of_data, file, overwrite=True):
    list_of_hdu = [fits.PrimaryHDU(None)]
    for data in list_of_data:
        list_of_hdu.append(data2hdu(data))
    hdu2fits(list_of_hdu, file, overwrite=overwrite)


# compute 2d real FFT frequencies as a fraction of the Nyquist frequency
def rfft2_freq(n1, n2):
    n2r = n2//2 + 1
    f1 = np.repeat(np.fft.fftfreq(n1), n2r).reshape(n1,n2r)
    f2 = np.repeat(np.fft.rfftfreq(n2),n1).reshape(n2r,n1).transpose()
    f = np.sqrt(f1**2 + f2**2)
    return f


def pre_process(frame):
    return ndimage.gaussian_filter(frame, sigma=gauss_sigma).astype(working_precision)


# align a frame to the reference frame
def align_frames(i):
    tst = time.time()
    # read the raw data as an object, obtain the image and compute its fft
    frame = np.fromfile(file_swp[i], dtype=raw_data_type).reshape(n1, n2)
    frame_fft = np.fft.rfft2(pre_process(frame*win))
    
    # compute the frame offset
    cache = np.abs(np.fft.irfft2((ref_fft*frame_fft*mask_hp).astype("complex64")))
    index = np.unravel_index(np.argmax(cache, axis=None), cache.shape)
    s1, s2 = -index[0], -index[1]
    
    # make sure that the Bayer matrix will not be corrupted
    s1 = s1 - np.mod(s1, 2)
    s2 = s2 - np.mod(s2, 2)
    
    # fix the offset and save into the result array
    frame = np.roll(frame, (s1, s2), axis=(0,1))
    
    # save the aligned images and binaries if necessary
    frame.tofile(file_swp[i])
    
    if less_report==False:
        print("\nFrame %6i (%s) aligned in %8.2f sec, (sx, sy) = (%8i,%8i)." %(i, file_lst[i], time.time()-tst, s1, s2))
    return i, s1, s2


def compute_weights(frames_working):
    # read the alignment results of multiple processes
    tst = time.time()
    for i in range(n_files):
        frame = np.fromfile(file_swp[i], dtype=raw_data_type)
        frames_working[i,:,:] = frame.reshape(n1, n2)

    print("Aligned frames read in, time cost:                        %9.2f" %(time.time()-tst)); tst = time.time()

    # remove the mean value from each frame
    tst = time.time()
    for i in range(0, n_files):
        frames_working[i,:,:] = frames_working[i,:,:] - np.mean(frames_working[i,:,:])
    print("Mean values of frames removed, time cost:                 %9.2f" %(time.time()-tst)); tst = time.time()
    
    # compute the covariance matrix
    frames_working = frames_working.reshape(n_files, n1*n2)
    cov = np.dot(frames_working, frames_working.transpose())

    # compute weights from the covariance matrix
    w = np.zeros(n_files)
    for i in range(n_files):
        w[i] = np.sum(cov[i,:])/cov[i,i] - 1

    return w


# re-scale array (linear) to [vmin, vmax] 
def rescale_array(x_in, vmin, vmax):
    x = x_in.copy()
    xmin, xmax = np.amin(x), np.amax(x)
    # first to range [0, 1]
    x = np.float64(x-xmin) / np.float64(xmax-xmin)
    # then to range [vmin, vmax]
    x = x*(vmax-vmin) + vmin
    return x


# manually adjust the rgb range
def adjust_color_manual(rgb, rgb_min, rgb_max, rgb_gamma):
    global_max = vmax_global
    r = rescale_array(rgb[:,:,0], rgb_min[0], rgb_max[0])
    g = rescale_array(rgb[:,:,1], rgb_min[1], rgb_max[1])
    b = rescale_array(rgb[:,:,2], rgb_min[2], rgb_max[2])
    r = (r**rgb_gamma[0])*global_max
    g = (g**rgb_gamma[1])*global_max
    b = (b**rgb_gamma[2])*global_max
    rgb_adjusted = rgb*0
    rgb_adjusted[:,:,0] = r
    rgb_adjusted[:,:,1] = g
    rgb_adjusted[:,:,2] = b
    return rgb_adjusted


# convert elevation and azimuth to unit vectors
def elaz2vec(el, az):
    n = np.size(el)
    d2r = np.pi/180
    vec = np.zeros([n, 3])
    vec[:,0] = np.cos(el*d2r)*np.cos(az*d2r)
    vec[:,1] = np.cos(el*d2r)*np.sin(az*d2r)
    vec[:,2] = np.sin(el*d2r)
    return vec


def len_vec(vec):
    return np.sqrt(np.sum(vec*vec, axis=1))


def unit_vec(vec):
    amp = len_vec(vec)
    vec_nml = vec.copy()
    vec_nml[:,0] = vec_nml[:,0] / amp
    vec_nml[:,1] = vec_nml[:,1] / amp
    vec_nml[:,2] = vec_nml[:,2] / amp
    return vec_nml


# Convert round angle to equivalent linear angles. The idea is: when the
# difference between two neighbors is bigger than a threshold, all following
# angles are corrected by n*360 degree to minimize the difference.
def round2linear(ang_in, deg=True, threshold=350):
    n = np.size(ang_in)
    ang = ang_in.copy()
    fac = 180/np.pi
    if deg==False:
        ang = ang*fac
    for i in range(n-1):
        dif = ang[i]-ang[i+1]
        if np.abs(dif)>threshold:
            da = ang[i+1] - ang[i]
            cc = np.cos(da/fac)
            ss = np.sin(da/fac)
            da1 = np.arctan2(ss, cc)*fac
            ang[i+1:n] = ang[i+1:n] + da1 - da
    return ang


# compute the field rotation, return continuous angle, but can also return
# arccos values by debug=True (discontinuous, only for debug)
def compute_field_rot(target, hor_ref_frame, debug=False):
    north_pole = SkyCoord(ra=0*u.deg, dec=90*u.deg)
    north_pole_coord_hor = north_pole.transform_to(hor_ref_frame)
    target_coord_hor = target.transform_to(hor_ref_frame)
    # convert el-az to unit vectors
    el_target = target_coord_hor.alt.to_value()
    az_target = target_coord_hor.az.to_value()
    el_np = north_pole_coord_hor.alt.to_value()
    az_np = north_pole_coord_hor.az.to_value()
    vec_target = elaz2vec(el_target, az_target)
    vec_north  = elaz2vec(el_np, az_np)
    vec_z      = vec_north*0; vec_z[:,2] = 1
    # compute local east by cross product, with normalization
    east_cel = unit_vec(np.cross(vec_target, vec_north))
    east_hor = unit_vec(np.cross(vec_target, vec_z    ))
    # determine the hor-to-cel rotation direction (sign of rotation)
    vec_cel2hor = np.cross(east_cel, east_hor)
    flag = np.sum(vec_cel2hor*vec_target, axis=1)
    flag = np.where(flag>0, 1, -1)
    # compute field rotation angle by atan2
    val_cos = np.sum(east_cel*east_hor, axis=1)
    val_sin = len_vec(vec_cel2hor)
    rot_ang = np.arctan2(val_sin, val_cos)*180/np.pi
    # and determine the direction of rotation
    rot_ang = rot_ang * flag
    if debug==True:
        rot_ang = np.arccos(val_cos)*180/np.pi
    return rot_ang


# note: need to check if we should multiply -1 to angle. 
# note: the four Bayer blocks are rotated separately so as not to corrupt the Bayer matrix.
def fix_rotation(file, angle, raw_data_type, n1, n2):
    frame = np.fromfile(file, dtype=raw_data_type).reshape(int(n1/2), 2, int(n2/2), 2)
    frame00 = frame[:,0,:,0].reshape(int(n1/2), int(n2/2))
    frame01 = frame[:,0,:,1].reshape(int(n1/2), int(n2/2))
    frame10 = frame[:,1,:,0].reshape(int(n1/2), int(n2/2))
    frame11 = frame[:,1,:,1].reshape(int(n1/2), int(n2/2))
    frame00 = ndimage.rotate(frame00, angle, reshape=False)
    frame01 = ndimage.rotate(frame01, angle, reshape=False)
    frame10 = ndimage.rotate(frame10, angle, reshape=False)
    frame11 = ndimage.rotate(frame11, angle, reshape=False)
    frame00 = np.where(frame00==0, np.median(frame00), frame00)
    frame01 = np.where(frame01==0, np.median(frame01), frame01)
    frame10 = np.where(frame10==0, np.median(frame10), frame10)
    frame11 = np.where(frame11==0, np.median(frame11), frame11)
    frame[:,0,:,0] = frame00
    frame[:,0,:,1] = frame01
    frame[:,1,:,0] = frame10
    frame[:,1,:,1] = frame11
    frame.tofile(file)


# Re-scale the stacked frame values to avoid problem in type conversion to
# uint16. Do it separately for the four Bayer matrices.
def normalize_frame(frame, vmin, vmax):
    frame_norm = frame.copy()
    shape = np.shape(frame_norm)
    n1, n2 = shape[0], shape[1]
    frame_norm = frame_norm.reshape(int(n1/2), 2, int(n2/2), 2)
    frame_norm[:,0,:,0] = rescale_array(frame_norm[:,0,:,0], vmin, vmax)
    frame_norm[:,0,:,1] = rescale_array(frame_norm[:,0,:,1], vmin, vmax)
    frame_norm[:,1,:,0] = rescale_array(frame_norm[:,1,:,0], vmin, vmax)
    frame_norm[:,1,:,1] = rescale_array(frame_norm[:,1,:,1], vmin, vmax)
    return frame_norm.reshape(n1, n2)
    
    
    
    

    


##############################################################
# Do the following:
# 1. Align the frames using the initial reference frame.
# 2. Compute the weights from the covarinace matrix.
# 3. Set the reference frame to the one with highest weight.
# 4. Re-align the frames using the new reference frame.
# 5. Re-compute the weights from the covarinace matrix.
# 6. Stack the frames with weights.
##############################################################


# in console mode, do not produce online images (but will save pdf)
if console == True:
    matplotlib.use('Agg')
    if len(sys.argv)==4:
        working_dir = sys.argv[1]
        extension = sys.argv[2]
        nproc_max = int(sys.argv[3])
    else:
        print("Warning: command-line parameters less than 3, will use the default ones instead:")
        print("Working directory:         %s" %(working_dir))
        print("Input file extension:      %s" %(extension))
        print("Number of processes limit: %s" %(nproc_max))
else:
    # Improve the display effect of Jupyter notebook
    from IPython.core.display import display, HTML

# make a list of working files and determine the number of processes to be used
os.chdir(working_dir)
fullpath = os.path.abspath("./")    
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

file_lst, file_swp = [], []
for file in os.listdir():
    if file.endswith(extension):
        file_lst.append(file)

n_files = np.int64(len(file_lst))
if n_files<2:
    print("Too few files, maybe check the directory name or extension?")
    sys.exit()

if nproc_max > n_files:
    nproc = n_files
else:
    nproc = nproc_max
    
# sort the file list and then build auxiliary file lists accordingly
file_lst.sort()
for file in file_lst:
    file_swp.append(os.path.splitext(file)[0] + '.swp')

# use the first file as the initial reference file
ref_frame, _, raw_data_type = read_frame_fits(file_lst[0]) 
n1 = np.int64(np.shape(ref_frame)[0])
n2 = np.int64(np.shape(ref_frame)[1])

# make the 2D-tukey window
w1 = tukey(n1, alpha=tukey_alpha)
w2 = tukey(n2, alpha=tukey_alpha)
win = np.dot(w1.reshape(n1, 1), w2.reshape(1, n2))

# make a low-pass window in the Fourier domain
freq = rfft2_freq(n1, n2)
mask_hp = np.where(freq<highpass_cut, 0, 1)

ref_fft = np.conjugate(np.fft.rfft2(pre_process(ref_frame*win)))
ref_fft = ref_fft * mask_hp

# read the frames by main
if __name__ == '__main__':
    # prepare the working array
    frames_working = np.zeros([n_files, n1, n2], dtype=working_precision)
    
    # read frames, save the observation times, and compute the local
    # horizontal reference frames accordingly
    tst = time.time()
    datetime = []
    for i in range(n_files):
        frame1, time1, _ = read_frame_fits(file_lst[i])
        frame1.tofile(file_swp[i])
        datetime.append(time1)
        frames_working[i,:,:] = frame1
    print("Frames read and cached by the main proc, time cost:       %9.2f" %(time.time()-tst))

# fix rotation if necessary (multi-processes)
if __name__ == '__main__' and do_fix_ratation==True:
    tst = time.time()
    obstime_list = astro_time(datetime) - ahu_zone*u.hour
    hor_ref_frame = AltAz(obstime=obstime_list, location=ahu_observatory)

    # compute the reletive time in seconds, and subtract the median value to minimize the rotations
    rel_sec = (obstime_list - obstime_list[0]).to_value(format='sec')
    rel_sec = rel_sec - np.median(rel_sec)
    
    # compute the absolute field rotation angles as "rot_ang"
    rot_ang = compute_field_rot(target, hor_ref_frame)
    rot_ang = rot_ang - np.median(rot_ang)
    file = os.path.join(fullpath,outputdir,file_rot_ang)
    rot_ang.astype(working_precision).tofile(file)
    print("Rotation angles computed, time cost:                      %9.2f" %(time.time()-tst))

    # plot the field rotation angle for test 
    plt.figure(figsize=(4,2), dpi=200)
    plt.title('The field rotation angle')
    plt.xlabel('Time (sec)', fontsize=9)
    plt.ylabel('Angle', fontsize=9)
    plt.plot(rel_sec, rot_ang, marker="o")
    plt.savefig('field_rot_angle.pdf')

    # fix the field rotation (multi-processes)
    tst = time.time()
    p1, p2, p3, p4, p5 = [], [], [], [], []
    for i in range(n_files):
        p1.append(file_swp[i])
        p2.append(rot_ang[i])
        p3.append(raw_data_type)
        p4.append(n1)
        p5.append(n2)
    with mp.Pool(nproc) as pool:
        output = [pool.starmap(fix_rotation, zip(p1, p2, p3, p4, p5))]
    print("Field rotation fixed, time cost:                          %9.2f" %(time.time()-tst) )

# For all processes: if fix-rotation is required, then read rot_ang from file and 
# reset the reference frame to the one with least rotation (frame already fixed)
if do_fix_ratation==True:
    file = os.path.join(fullpath,output_dir,file_rot_ang)
    rot_ang = np.fromfile(file, dtype=working_precision)
    wid = np.argmin(np.abs(rot_ang))
    ref_frame = np.fromfile(file_swp[wid], dtype=raw_data_type).reshape(n1, n2)
    ref_fft = np.conjugate(np.fft.rfft2(pre_process(ref_frame*win)))
    ref_fft = ref_fft * mask_hp
    
if __name__ == '__main__':    
    tst = time.time()
    with mp.Pool(nproc) as pool:
        output = [pool.map(align_frames, range(n_files))]
    print("Initial alignment done, time cost:                        %9.2f" %(time.time()-tst))

    output_arr = np.array(output)
    sx1, sy1 = output_arr[0,:,1], output_arr[0,:,2]
    sx1 = np.where(sx1 >  n1/2, sx1-n1, sx1)
    sx1 = np.where(sx1 < -n1/2, sx1+n1, sx1)
    sy1 = np.where(sy1 >  n2/2, sy1-n2, sy1)
    sy1 = np.where(sy1 < -n2/2, sy1+n2, sy1)

    # identify the frame of maximum weight, and use it as the new reference frame.
    w = compute_weights(frames_working)
    wid = np.argmax(w)
    ref_frame = np.fromfile(file_swp[wid], dtype=raw_data_type).reshape(n1, n2)
    ref_fft = np.conjugate(np.fft.rfft2(pre_process(ref_frame*win)))
    ref_fft = ref_fft * mask_hp
    print("****************************************************")
    print("Frame %i is chosen as the new reference frame. All frames will be re-aligned." %(wid))
    print("The new reference file is: %s" %(file_lst[wid]))
    print("****************************************************")
    
    # work with multiprocessing to align the frames again, and remove the swp files
    tst = time.time()
    with mp.Pool(nproc) as pool:
        output = [pool.map(align_frames, range(n_files))]
    print("Final alignment done, time cost:                          %9.2f" %(time.time()-tst))
    
    # parse and record the offsets
    output_arr = np.array(output)
    sx2, sy2 = output_arr[0,:,1], output_arr[0,:,2]
    sx2 = np.where(sx2 >  n1/2, sx2-n1, sx2)
    sx2 = np.where(sx2 < -n1/2, sx2+n1, sx2)
    sy2 = np.where(sy2 >  n2/2, sy2-n2, sy2)
    sy2 = np.where(sy2 < -n2/2, sy2+n2, sy2)

    # plot the XY-shifts
    plt.figure(figsize=(5,4), dpi=200)
    plt.title('XY shifts in pixel')
    plt.xlabel('Y shifts', fontsize=9)
    plt.ylabel('X shifts', fontsize=9)
    TT = np.float64(np.arange(n_files))/n_files
    plt.scatter(sx1, sy1, s=50, c=TT, cmap='viridis', alpha=0.8, label='Round 1')
    plt.scatter(sx2, sy2, s=50, c='k', alpha=0.8, label='Round 2')
    plt.legend()
    plt.savefig(os.path.join(fullpath,output_dir,'xy-shifts.pdf'))
    
    # recompute the weights
    tst = time.time()
    w = compute_weights(frames_working)
    # exclude the low quality frames
    n_bad = int(n_files*bad_fraction)
    thr = hp.nsmallest(n_bad, w)[n_bad-1]
    if thr<0: thr = 0
    w = np.where(w <= thr, 0, w)
    w = w / np.sum(w)
    print("Final weights computed, time cost:                        %9.2f" %(time.time()-tst))
    
    # plot the weights for test 
    plt.figure(figsize=(4,2), dpi=200)
    plt.title(r'Stacking weights ($w\times N_{frames}$)')
    plt.xlabel('Frame number', fontsize=9)
    plt.ylabel(r'$w\times N_{frames}$', fontsize=9)
    w1 = np.where(w==0, np.nan, w)
    w2 = np.where(w==0, np.nanmean(w1), np.nan)
    plt.plot(w1*n_files*(1-bad_fraction), marker="o", label='Valid')
    plt.plot(w2*n_files*(1-bad_fraction), marker="*", label='Invalid')
    plt.legend()
    plt.savefig(os.path.join(fullpath,output_dir,'weights.pdf'))

    # stack the frames with weights.
    tst = time.time()
    frame_stacked = np.dot(w, frames_working.reshape(n_files, n1*n2)).reshape(n1, n2)

    # normalize the stacked frames to avoid negative values (due to subtraction of mean values).
    frame_stacked = normalize_frame( frame_stacked, 0, vmax_global )

    file_stacked = os.path.join(fullpath,output_dir,final_file_fits)
    write_fits_simple([frame_stacked.astype(raw_data_type)], file_stacked, overwrite=True)

    print("Stacked frame obtained from %i/%i best frames, time cost: %9.2f" 
        %(n_files-n_bad, n_files, time.time()-tst))

    # clear swap files
    for file in file_swp:
        os.remove(file)

    print("Done!")




    # tst = time.time()
    # # stacked frame to linear rgb value (handle the Bayer matrix)
    # rgb = frame2rgb_fits(frame_stacked)

    # rgb_modified, cdf = color_correction(rgb, rgb_gamma, rgb_max=rgb_max, bins=32000)

    # # save the 48-bit color image
    # imageio.imsave(os.path.join(working_dir,output_dir,final_file_tif), 
    #     rgb_modified.astype(raw_data_type))

    # # show the 8-bit color image as a quick preview (lower quality)
    # if console==False:
    #     plt.figure(figsize=(6,4),dpi=200)
    #     plt.xlabel('Y',fontsize=12)
    #     plt.ylabel('X',fontsize=12)
    #     plt.imshow(np.uint8(rgb_modified/256))
    # print("****************************************************")
    # print("Final image obtained with colors corrected by modified histogram equalization")
    # print("Current color correction parameters: (r, g, b) ranges: %7i, %7i, %7i " 
    #     %(rgb_max[0], rgb_max[1], rgb_max[2]))
    # print("Current color correction parameters: (r, g, b) gamma: %7.2f, %7.2f, %7.2f " 
    #     %(rgb_gamma[0], rgb_gamma[1], rgb_gamma[2]))
    
    # # adjust the color and make the final 8-bit image
    # tst = time.time()
    # rgb = frame2rgb_fits(frame_stacked)

    # r_bin_file = os.path.splitext(final_file_tif)[0] + '.r'; rgb[:,:,0].tofile(r_bin_file)
    # g_bin_file = os.path.splitext(final_file_tif)[0] + '.g'; rgb[:,:,1].tofile(g_bin_file)
    # b_bin_file = os.path.splitext(final_file_tif)[0] + '.b'; rgb[:,:,2].tofile(b_bin_file)
    
    # # color correction in parallel
    # m1, m2, npix = np.shape(rgb)[0], np.shape(rgb)[1], rgb.size
    # tst = time.time()
    # ic = [0, 1, 2]
    # im1 = [m1, m1, m1]
    # im2 = [m2, m2, m2]
    # ifn = [r_bin_file, g_bin_file, b_bin_file]
    # dtp = [raw_data_type, raw_data_type, raw_data_type]
    # with mp.Pool(3) as pool:
    #     output = [pool.starmap(adjust_color, zip(ic, im1, im2, ifn, dtp))]

    # # read the color correction result and save to 
    # rgb[:,:,0] = np.fromfile(r_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(r_bin_file)
    # rgb[:,:,1] = np.fromfile(g_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(g_bin_file)
    # rgb[:,:,2] = np.fromfile(b_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(b_bin_file)
    # print("Color adjusted, time cost:                                %9.2f" %(time.time()-tst)); tst = time.time()
    

    # # save the final figure
    # imageio.imsave(final_file_tif, rgb.astype(raw_data_type))

    # # show the final figure
    # plt.figure(figsize=(6,4),dpi=200)
    # plt.xlabel('Y',fontsize=12)
    # plt.ylabel('X',fontsize=12)
    # plt.imshow(np.uint8(rgb))


# import heapq as hp
# # subroutine for adjusting the colors 
# def adjust_color(i, m1, m2, bin_file, raw_data_type):
#     # number of "too dark" pixels and threshold
#     val_max = vmax_global
#     samp = np.fromfile(bin_file, dtype=raw_data_type).reshape(m1*m2)
#     npix = np.int64(m1)*np.int64(m2)
#     ndark = int(npix*dark_frac)
#     d1 = hp.nsmallest(ndark, samp)[ndark-1]*1.
#     # number of "too bright" pixels and threshold
#     nbright = int(npix*bright_frac)
#     d2 = hp.nlargest(nbright, samp)[nbright-1]*1.
#     # re-scaling, note that this requires 16-bit to save weak signal from bright sky-light
#     # note that val is expected to be in range [0,1]. Out-of-range values will be truncated.
#     val = np.float64(samp-d1)/np.float64(d2-d1)
#     val = np.where(val<=0, 1./val_max, val)
#     val = np.where(val >1, 1, val)
#     samp = (val**rgb_gamma[i])*val_max*rgb_fac[i]
#     samp = np.where(samp<      0,       0, samp)
#     samp = np.where(samp>val_max, val_max, samp)
#     samp.astype(raw_data_type).tofile(bin_file)