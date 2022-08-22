import sys
import cv2, os, imageio, time
import numpy as np
import pandas as pd
import multiprocessing as mp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import resample
from scipy.stats import linregress
from multiprocessing.managers import SharedMemoryManager


# define the subroutines
def rfft2_freq(n1, n2):
    n2r = n2//2 + 1
    f1 = np.repeat(np.fft.fftfreq(n1), n2r).reshape(n1,n2r)
    f2 = np.repeat(np.fft.rfftfreq(n2),n1).reshape(n2r,n1).transpose()
    f = np.sqrt(f1**2 + f2**2)
    return f

def lowpass_2d(array, ratio):
    n1 = array.shape[0]
    n2 = array.shape[1]
    f2 = rfft2_freq(n1, n2)
    f2 = f2/np.amax(f2)
    mask = np.where(f2>ratio, 0, 1)
    array_hp = np.fft.irfft2(np.fft.rfft2(array)*mask)
    return array_hp

def highpass_2d(array, ratio):
    n1 = array.shape[0]
    n2 = array.shape[1]
    f2 = rfft2_freq(n1, n2)
    f2 = f2/np.amax(f2)
    mask = np.where(f2<ratio, 0, 1)
    array_hp = np.fft.irfft2(np.fft.rfft2(array)*mask)
    array_hp = array_hp - np.amin(array_hp)
    return array_hp

# simplified 2D wiener filter that assume the highest "fraction" of FFT spectrum is dominated by 
def wiener2D_simple(array, frac):
    shape = array.shape
    n1, n2 = shape[0], shape[1]
    df = np.fft.fft2(array)
    amp = np.abs(df)**2
    na = hq.nsmallest(int(n1*n2*frac), list(amp.flatten()))[-1]
    amp = np.where(amp<na, na, amp)
    fac = (amp-na)/amp
    data_filt = np.fft.ifft2(df*fac)
    data_filt = np.real(data_filt)
    return data_filt

def read_fits(file):
    with fits.open(file) as hdu:
        n = len(hdu)
        frame = hdu[n-1].data
    return frame

# read and average all frames in the folder
def ave_frame(folder):
    frame = None
    n = 0.
    for root, dirs, files in os.walk(folder):
        for file in files:
            if n==0:
                frame = read_fits(os.path.join(folder, file))
            else:
                frame = frame + read_fits(os.path.join(folder, file))
            n = n + 1.
        break
    return frame/n

# decorrelate x from y
def decorr(x, y):
    res = linregress(x.flatten(), y.flatten())
    return y - x*res.slope

# do histogram equalization and gamma correction for one channel
def hist_equal_gamma(i):
    global rgb, rgb_corrected
    # get image histogram
    array = rgb[:,:,i].flatten()
    hist, bins = np.histogram(array, rgb_nbins, density=True)
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    # use linear interpolation of cdf to find new pixel values
    array = np.interp(array, bins[:-1], cdf)
    if gamma != 1:
        x1 = np.linspace(0, 1, num=rgb_nbins)
        cdf1 = x1**(gamma)
        array = np.interp(array, x1, cdf1)
    buff = norm_arr(array.reshape(n1, n2), rgb_vmin, rgb_vmax)
    rgb_corrected[:,:,i] = buff
    # rgb_corrected[:,:,i] = rgb[:,:,i]

def norm_arr(array, vmin, vmax):
    x = array.copy()
    xmin = np.amin(x)
    xmax = np.amax(x)
    x = (x-xmin)/(xmax-xmin)*(vmax-vmin) + vmin
    return x

def read_params(excel_file):
    global working_dir, file_stacked, rgb_vmin, rgb_vmax, hori_inv, vert_inv, \
    down_samp_fac, rgb_nbins, gamma, gauss_sigma, fix_offset, fix_dark, \
    fix_flat, dir_offset, dir_dark, dir_flat, show_image, file_tif, \
    vertical_clip, horizontal_clip, raw_data_type, bayer_matrix_format, multi_sess, console_mode
    
    df = pd.read_excel(excel_file)

    working_dir  = df[2][1]
    file_stacked = df[2][2]

    rgb_vmin = int(df[2][4])
    rgb_vmax = int(df[2][5])

    bayer_string = df[2][6]

    hori_inv = df[2][8].lower() == "true"
    vert_inv = df[2][9].lower() == "true"

    vc0 = float(df[2][11])
    vc1 = float(df[2][12])
    hc0 = float(df[2][14])
    hc1 = float(df[2][15])

    down_samp_fac = float(df[2][17])
    rgb_nbins     = int(df[2][18])
    gamma         = float(df[2][19])
    gauss_sigma   = float(df[2][20])

    fix_offset = df[2][22].lower() == "true"
    fix_dark   = df[2][23].lower() == "true"
    fix_flat   = df[2][24].lower() == "true"

    dir_offset = df[2][26]
    dir_dark   = df[2][27]
    dir_flat   = df[2][28]
    
    multi_sess = df[2][30].lower() == "true"
    console_mode = df[2][31].lower() == "true"

    show_image      = True
    raw_data_type   = np.uint16
    
    if bayer_string.lower()=="rggb":
        bayer_matrix_format = cv2.COLOR_BayerRG2RGB
    elif bayer_string.lower()=="bggr":
        bayer_matrix_format = cv2.COLOR_BayerBG2RGB
    elif bayer_string.lower()=="grbg":
        bayer_matrix_format = cv2.COLOR_BayerGR2RGB
    else:
        bayer_matrix_format = cv2.COLOR_BayerGB2RGB


# Run image adjustment and corrections
# read parameters from excel
read_params("adjust_image_params.xlsx")

if len(sys.argv)>1:
    working_dir = sys.argv[1]
if len(sys.argv)>2:
    gamma = float(sys.argv[2])
if len(sys.argv)>3:
    vc0 = float(sys.argv[3])
if len(sys.argv)>4:
    vc1 = float(sys.argv[4])
if len(sys.argv)>5:
    hc0 = float(sys.argv[5])
if len(sys.argv)>6:
    hc1 = float(sys.argv[6])

vertical_clip   = [vc0, vc1]
horizontal_clip = [hc0, hc1]

file_tif  = os.path.join(working_dir, "final_gamma"+str(gamma).format("4.4i")+".tiff")

print("Working directory:         %s" %(working_dir))
print("Gamma:                     %s" %(gamma))

# read the stacked frame and make sure the vlaues are all positive
frame_stacked = read_fits( os.path.join(working_dir, file_stacked) )
frame_stacked = frame_stacked - np.amin(frame_stacked)


# record the frame shape
shape = frame_stacked.shape
n1, n2 = shape[0], shape[1]

# clip image when necessary
v1, v2 = int((n1*vertical_clip[0])//2*2), int((n1*vertical_clip[1])//2*2)
h1, h2 = int((n2*horizontal_clip[0])//2*2), int((n2*horizontal_clip[1])//2*2)
frame_stacked = frame_stacked[v1:v2, h1:h2]
n1, n2 = v2-v1, h2-h1
print("Original size:    (hori, vert) = (%6i, %6i)" %(n2, n1) )
print("Work with subframe: horizontal = (%6i, %6i)" %(h1, h2) )
print("                      vertical = (%6i, %6i)" %(v1, v2) )


# convert stacked frame to linear rgb values (only handle the Bayer matrix)
tst = time.time()
if multi_sess == True:
    # start the shared memory manager
    smm = SharedMemoryManager()
    smm.start()

    shm_rgb = smm.SharedMemory(frame_stacked.size*3*8)
    rgb = np.frombuffer(shm_rgb.buf, dtype=np.float64).reshape(n1, n2, 3)

    shm_rgb_corrected = smm.SharedMemory(rgb.size*rgb.itemsize)
    rgb_corrected = np.frombuffer(shm_rgb_corrected.buf, dtype=np.float64).reshape(n1, n2, 3)

    rgb[:,:,:] = cv2.cvtColor(frame_stacked.astype(raw_data_type), bayer_matrix_format)
    if __name__ == '__main__':
        with mp.Pool(3) as pool:
            output = pool.map(hist_equal_gamma, range(3))
else:
    rgb = cv2.cvtColor(frame_stacked.astype(raw_data_type), bayer_matrix_format)
    rgb_corrected = rgb*0
    for i in range(3):
        hist_equal_gamma(i)
print("Color correction done,                time cost: %9.2f" %(time.time()-tst) )
print("Current color correction parameters:  gamma  : %7.2f" %(gamma) )


# Gaussian smoothing
if gauss_sigma != 0:
    for i in range(3):
        rgb_corrected[:,:,i] = ndimage.gaussian_filter(rgb_corrected[:,:,i], sigma=gauss_sigma)

# Horizontal or vertial inverting
if hori_inv==True:
    rgb_corrected = np.flip(rgb_corrected, axis=1)
if vert_inv==True:
    rgb_corrected = np.flip(rgb_corrected, axis=0)


# Down-sampling
if down_samp_fac>1:
    n1s = int(n1/down_samp_fac)
    n2s = int(n2/down_samp_fac)
    rgb_final = np.zeros([n1s, n2s, 3])
    for i in range(3):
        buff = resample(rgb_corrected[:,:,i], n1s, axis=0)
        buff = resample(buff, n2s, axis=1)
        rgb_final[:,:,i] = norm_arr(buff, rgb_vmin, rgb_vmax)
else:
    rgb_final = rgb_corrected


# save the 48-bit color image
imageio.imsave(file_tif, rgb_final.astype(raw_data_type))


# show the image
if console_mode == False:
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb_final/256))
    plt.show()

if multi_sess == True:
    rgb, rgb_corrected = 0, 0
    smm.shutdown()