##############################################################
import os, rawpy, imageio, time, sys
import scipy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import multiprocessing as mp
from astropy.io import fits


##############################################################
# working directory, all raw files should be in this directory
working_dir = "/work/astro/fits"

# name of the final image
final_file = "final.tiff"

# working precision is for FFT and matrix multiplication
working_precision = "float64"

# define the input file extension, can be fit, fits (for fits files); or CR2,
# CR3, nef...(for raw files)
extension = "fit"

# the file of reference frames. If set to None, then use the first frame in the list.
reference_file = 'Light_ASIImg_0.1sec_Bin1_-19.0C_gain0_2022-01-28_200730_frame0001.fit'

# define the Bayer matrix format, only for the fits file
bayer_matrix_format = cv.COLOR_BayerBG2RGB

# the page number configuration, only for the fits file
page_num = 0

# if true, work in console mode, will not process the Jupyter notebook code
# and will not produce online images.
console = True

# define the maximum number of processes to be used
nproc_max = 96

# If true, do not report the alignment result
less_report = True

# fraction of frames that will not be used
bad_fraction = 0.4

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


##############################################################
if console == False:
    # Improve the display effect of Jupyter notebook
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
else:
    # do not produce online images (but will still save pdf)
    matplotlib.use('Agg')

if extension=='fit' or extension=='fits':
    input_format = 'fits'
else:
    input_format = 'raw'


##############################################################
#
# read fits or raw file and convert to bin so it can be used by
# multiprocessing
#
def read_frame_raw():
    tst = time.time()
    raw_data_type = None
    for i in range(n_files):
        # read the raw data as an object, obtain the image and compute its fft
        with rawpy.imread(file_lst[i]) as raw:
            if raw_data_type==None:
                raw_data_type = raw.raw_image.dtype
            raw.raw_image.tofile(file_swp[i])
    return raw_data_type

def read_frame_fits():
    tst = time.time()
    raw_data_type = None
    for i in range(n_files):
        # read the raw data as an object, obtain the image and compute its fft
        with fits.open(file_lst[i]) as hdu:
            frame = hdu[page_num].data
            if raw_data_type==None:
                raw_data_type = frame.dtype
            frame.tofile(file_swp[i])
    return raw_data_type


#
# use the raw or fits information to convert a frame to rgb image
#
def frame2rgb_raw(raw, frame):
    raw.raw_image[:,:] = frame
    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    return rgb

def frame2rgb_fits(frame):
    rgb = cv.cvtColor(frame.astype(raw_data_type), bayer_matrix_format)
    return rgb


# 
# align a frame to the reference frame
# 
def align_frames(i):
    tst = time.time()
    # read the raw data as an object, obtain the image and compute its fft
    frame = np.fromfile(file_swp[i], dtype=raw_data_type).reshape(n1, n2)
    os.remove(file_swp[i])
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



import heapq as hp
# subroutine for adjusting the colors 
def adjust_color(i, m1, m2, npix, bin_file):
    # number of "too dark" pixels and threshold
    samp = np.fromfile(bin_file, dtype="uint16").reshape(m1*m2)
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
    samp.astype("uint16").tofile(bin_file)



# make a list of woking files
os.chdir(working_dir)
file_lst, file_bin, file_swp, file_tif = [], [], [], []
for file in os.listdir():
    if file.endswith(extension):
        file_lst.append(file)
        file_swp.append(os.path.splitext(file)[0] + '.swp')
        file_bin.append(os.path.splitext(file)[0] + '_aligned.bin')
        file_tif.append(os.path.splitext(file)[0] + '_aligned.tif')
n_files = np.int64(len(file_lst))

if nproc_max > n_files:
    nproc = n_files
else:
    nproc = nproc_max



# prepare the reference frame in the Fourier domain
if reference_file==None:
    reference_file = file_lst[0]

if input_format=='raw':
    ref_raw = rawpy.imread(reference_file)
    ref_frame = ref_raw.raw_image
elif input_format=='fits':
    with fits.open(reference_file) as hdu:
        ref_frame = hdu[page_num].data
else:
    print("Error! Unknown input file format, should be fits or raw!")
    sys.exit()

n1 = np.int64(np.shape(ref_frame)[0])
n2 = np.int64(np.shape(ref_frame)[1])
ref_fft = np.conjugate(fft.fft2(ref_frame.astype(working_precision)))


# use multiprocessing to work on multiple files' alignment.
frames_aligned = np.zeros([n_files, n1, n2], dtype=working_precision)

if __name__ == '__main__':
    tst = time.time()

    # read all raw files
    if input_format=='raw':
        raw_data_type = read_frame_raw()
        print("Raw file read in. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    elif input_format=='fits':
        raw_data_type = read_frame_fits()
        print("Fits file read in. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    else:
        print("Error! Unknown input file format, should be fits or raw!")
        sys.exit()


    # work with multiprocessing
    with mp.Pool(nproc) as pool:
        output = [pool.map(align_frames, range(n_files))]

    # # work with sequential alignment
    # for i in range(n_files):
    #     align_frames(i)

    output_arr = np.array(output)
    sx, sy = output_arr[0,:,1], output_arr[0,:,2]
    sx = np.where(sx >  n1/2, sx-n1, sx)
    sx = np.where(sx < -n1/2, sx+n1, sx)
    sy = np.where(sy >  n2/2, sy-n2, sy)
    sy = np.where(sy < -n2/2, sy+n2, sy)
    print("Alignment done. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()


    # read the alignment results of multiple processes
    for i in range(n_files):
        frame = np.fromfile(file_bin[i],dtype=raw_data_type)
        frames_aligned[i,:,:] = frame.reshape(n1, n2)
        if save_aligned_binary is False:
            os.remove(file_bin[i])
    print("Aligned frames read in. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()


    # stack the frames with weights computed from the summed cross
    # covariance between frames (ignore auto covariance)
    tst = time.time()
    for i in range(0, n_files):
        frames_aligned[i,:,:] = frames_aligned[i,:,:] - np.mean(frames_aligned[i,:,:])
    print("Mean values of frames removed. Time cost: %9.2f" %(time.time()-tst)); tst = time.time()
    

    # compute the covariance matrix
    frames_aligned = frames_aligned.reshape(n_files, n1*n2)
    cov = np.dot(frames_aligned, frames_aligned.transpose())

    # compute weights from the covariance matrix
    w = np.zeros(n_files)
    for i in range(n_files):
        w[i] = np.sum(cov[i,:])/cov[i,i] - 1

    # exclude the low quality frames
    n_bad = int(n_files*bad_fraction)
    thr = hp.nsmallest(n_bad, w)[n_bad-1]
    if thr<0: thr = 0
    w = np.where(w <= thr, 0, w)
    w = w / np.sum(w)

    # stack the frames with weights.
    # note that we must normalize the stacked result.
    frame_stacked = np.dot(w, frames_aligned).reshape(n1, n2)
    fmin = np.amin(frame_stacked)
    fmax = np.amax(frame_stacked)
    cache = (frame_stacked-fmin)/(fmax-fmin)
    tmax = 2.**(adc_digit_max) - 1.
    frame_stacked = np.floor(cache*tmax)
    print("Stacked frame obtained from %i/%i best frames. Time cost: %9.2f" 
        %(n_files-n_bad, n_files, time.time()-tst)); tst = time.time()
        

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


    # adjust the color and prepare for 8-bit output, do it on linear rgb so
    # that we still have linear response, but don't have to worry about the
    # Bayer matrix. note that "gamma=(1,1), no_auto_bright=True" means to get
    # linear rgb rememer to use output_bps=16 for 16-bit output depth
    #
    # convert the stacked frame to raw and then to rgb, save the rgb files.
    # note that the image size is different after converting from raw to rgb
    # image
    if input_format=='raw':
        rgb = frame2rgb_raw(ref_raw, frame_stacked)
    elif input_format=='fits':
        rgb = frame2rgb_fits(frame_stacked)
    else:
        print("Error! Unknown input file format, should be fits or raw!")
        sys.exit()

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
    with mp.Pool(3) as pool:
        output = [pool.starmap(adjust_color, zip(ic, im1, im2, inp, ifn))]

    # read the color correction result and save to 
    rgb[:,:,0] = np.fromfile(r_bin_file, dtype="uint16").reshape(m1, m2); os.remove(r_bin_file)
    rgb[:,:,1] = np.fromfile(g_bin_file, dtype="uint16").reshape(m1, m2); os.remove(g_bin_file)
    rgb[:,:,2] = np.fromfile(b_bin_file, dtype="uint16").reshape(m1, m2); os.remove(b_bin_file)
    print("Color adjusted. Time cost: %8.2f" %(time.time()-tst))
    tst = time.time()
    

    # save the final figure
    imageio.imsave(final_file, np.uint8(rgb))

    # show the final figure
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb))

    print("Done!")
    
    

