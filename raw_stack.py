##############################################################
import os, rawpy, imageio, time, sys
import scipy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


##############################################################
# working directory, all raw files should be in this directory
working_dir = "/work/astro/temp"

# define the raw file extension
extension = "CR3"

# the file of reference frames. If set to None, then use the first frame in the list.
reference_file = 'IMG_0000.CR3'

# name of the final image
final_file = "final.tiff"

# working precision is for FFT and matrix multiplication
working_precision = "float64"

# if true, work in console mode, will not process the Jupyter notebook
# code and will not produce online images.
console = True

# define the number of processes to be used
nproc = 96

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
If console is False:
    # Improve the display effect of Jupyter notebook
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
else:
    # do not produce online images (but will still save pdf)
    matplotlib.use('Agg')


##############################################################
#
# read raw file and convert to bin so it can be used by multiprocessing
#
def read_raw():
    tst = time.time()
    raw_data_type = None
    for i in range(n_files):
        # read the raw data as an object, obtain the image and compute its fft
        with rawpy.imread(file_lst[i]) as raw:
            if raw_data_type==None:
                raw_data_type = raw.raw_image.dtype
            raw.raw_image.tofile(file_swp[i])
    return raw_data_type


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
    if save_aligned_binary is True:
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
def adjust_color(i, samp):
    # number of "too dark" pixels and threshold
    samp = samp.reshape(m1*m2)
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
    return samp.reshape(m1, m2)

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
if nproc > n_files:
    nproc = n_files



# prepare the reference frame in the Fourier domain
if reference_file==None:
    reference_file = file_lst[0]

ref_raw = rawpy.imread(reference_file)
ref_frame = ref_raw.raw_image
n1 = np.int64(np.shape(ref_frame)[0])
n2 = np.int64(np.shape(ref_frame)[1])
ref_fft = np.conjugate(fft.fft2(ref_frame.astype(working_precision)))



# use multiprocessing to work on multiple files' alignment.
frames_aligned = np.zeros([n_files, n1, n2], dtype=working_precision)

if __name__ == '__main__':
    tst = time.time()

    raw_data_type = read_raw()

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
    
    print("Alignment done. Time cost: %9.2f" %(time.time()-tst))

    # read the alignment results of multiple processes
    for i in range(n_files):
        frame = np.fromfile(file_bin[i],dtype=raw_data_type)
        frames_aligned[i,:,:] = frame.reshape(n1, n2)



    # stack the frames with weights computed from the summed cross
    # covariance between frames (ignore auto covariance)
    tst = time.time()
    for i in range(0, n_files):
        frames_aligned[i,:,:] = frames_aligned[i,:,:] - np.median(frames_aligned[i,:,:])
    
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
    # note that we must normalize the stacked result.
    frame_stacked = np.dot(w, frames_aligned).reshape(n1, n2)
    fmin = np.amin(frame_stacked)
    fmax = np.amax(frame_stacked)
    cache = (frame_stacked-fmin)/(fmax-fmin)
    tmax = 2.**(adc_digit_max) - 1.
    frame_stacked = np.floor(cache*tmax)
    print("Weights computed and %i/%i best frames used for stacking. Time cost: %6.2f" 
        %(n_files-n_bad, n_files, time.time()-tst))



    # adjust the color and prepare for 8-bit output, do it on linear rgb so that we still have 
    # linear response, but don't have to worry about the Bayer matrix.
    # note that "gamma=(1,1), no_auto_bright=True" means to get linear rgb
    # rememer to use output_bps=16 for 16-bit output depth
    print("Ready to adjust the colors:")
    ref_raw.raw_image[:,:] = frame_stacked
    rgb = ref_raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    # note that the image size is different after converting from raw to rgb image
    m1, m2, npix = np.shape(rgb)[0], np.shape(rgb)[1], rgb.size
    tst = time.time()
    for i in range(3):
        rgb[:,:,i] = adjust_color(i, rgb[:,:,i])
        print("Color adjusted for color channel %i, time cost: %8.2f" %(i, time.time()-tst))
        tst = time.time()

    # save the final figure
    imageio.imsave(final_file, np.uint8(rgb))

    # show the final figure
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb))

    print("Done!")
    
    
