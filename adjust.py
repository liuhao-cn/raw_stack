import sys, cv2, os, imageio, time, re

import numpy as np
import pandas as pd
import multiprocessing as mp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import adjust_params as par

from scipy import ndimage, fft
from scipy.signal import resample
from scipy.stats import linregress
from multiprocessing.managers import SharedMemoryManager


 # parse the channel names
def read_channel_name(filename):
    chn_pat = par.chn_pattern
    chn_pat = chn_pat + '.'
    name_loc = len(chn_pat) - 1
    pattern = re.compile(chn_pat)
    res = pattern.search(filename)
    chn_name = res.group(0)[name_loc]
    return chn_name


def get_ilc(frames):
    frames = np.array(frames)
    shape = frames.shape
    frames = np.reshape(frames, [shape[0], shape[1]*shape[2]])
    cov = np.cov(frames)
    for i in range(shape[0]):
        cov[i,i] = cov[i,i] * par.ilc_diag_fac
    cov_inv = np.linalg.inv(cov)
    w = np.sum(cov_inv, axis=0) / np.sum(cov_inv)
    frame = np.dot(w.transpose(), frames).reshape([shape[1], shape[2]])
    return frame


def get_pos_r(array, r, nbins):
    buff = array.flatten()
    hist, bins = np.histogram(buff, nbins, density=True)
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    val = bins[np.argmin(np.abs(cdf-r))]
    return val


# as a naive method, the noise level is defined as the 0.1% level of the
# FFT-power by default.
def naive_wiener_filter(frame, noise_fac=0.001):
    pow_norm_fac = 0.1
    frame_fft = fft.fft2(frame)
    frame_fft_pow_sqrt = np.abs(frame_fft*np.conj(frame_fft))**pow_norm_fac
    noise_level = get_pos_r(frame_fft_pow_sqrt, noise_fac, par.rgb_nbins)
    pow_now = 1./pow_norm_fac 
    fac = (frame_fft_pow_sqrt**pow_now - noise_level**pow_now)/frame_fft_pow_sqrt**pow_now
    fac[fac<0] = 0
    frame_fft = frame_fft*fac
    frame1 = np.abs(fft.ifft2(frame_fft))
    return frame1



# read frames with the given channel and return the average.
def ave_by_channel(channel, ilc=False):
    i = 0
    if ilc==True:
        frames = []
    else:
        frame = 0
    for file in os.listdir(par.working_dir):
        if file.endswith(par.extension):
            file = os.path.join(par.working_dir, file)
            if channel != '':
                chn_name = read_channel_name(file)
                if chn_name.lower()==channel.lower():
                    if ilc==True:
                        frames.append(read_fits(file)*1.)
                    else:
                        frame = frame + read_fits(file)*1.
                    i = i+1
            else:
                if ilc==True:
                    frames.append(read_fits(file)*1.)
                else:
                    frame = frame + read_fits(file)*1.
                i = i+1
    print('Number of files for channel %6s: %5i' %(channel, i))

    if i==0:
        return np.zeros([1])
    else:
        if ilc==True:
            frame = get_ilc(frames)
        else:
            frame = frame/i
        return frame

def comb_channels(r, g, b, l=None):
    # check the frame shape, note that one channel could be zero
    if r.size>1:
        shape = r.shape
    if g.size>1:
        shape = g.shape
    if b.size>1:
        shape = b.shape

    n1, n2 = shape[0], shape[1]
    frame = np.zeros([n1, n2, 3])

    # give empty channels proper shapes
    if r.size==1:
        r = np.zeros([n1, n2])
    if g.size==1:
        g = np.zeros([n1, n2])
    if b.size==1:
        b = np.zeros([n1, n2])

    if len(np.shape(l)) != 0:
        # read the rgb relative weights
        a1, a2, a3 = par.rgb_fac[0], par.rgb_fac[1], par.rgb_fac[2]
        # make the L-channel by inverse-noise weighting
        l1  = l.copy()
        l2  = (r*a1 + g*a2 + b*a3)/(a1+a2+a3)*3
        if par.ave_with_ilc==True:
            l = get_ilc([l1, l2])
        else:
            l = (l1 + l2)/2
        # make the color information with smoothed RGBs
        r = ndimage.gaussian_filter(r*1., sigma=3)
        g = ndimage.gaussian_filter(g*1., sigma=3)
        b = ndimage.gaussian_filter(b*1., sigma=3)
        # make final RGB with the L-information
        frame[:,:,0] = r*a1/(r*a1 + g*a2 + b*a3) * l 
        frame[:,:,1] = g*a2/(r*a1 + g*a2 + b*a3) * l
        frame[:,:,2] = b*a3/(r*a1 + g*a2 + b*a3) * l
    else:
        frame[:,:,0] = r 
        frame[:,:,1] = g
        frame[:,:,2] = b
    return frame


def read_fits(file):
    with fits.open(file) as hdu:
        frame = hdu[len(hdu)-1].data
    return frame


# do histogram equalization and gamma correction for one channel
def hist_equal_gamma(i):
    global rgb, rgb_corrected
    # get image histogram
    array = rgb[:,:,i].flatten()
    array = array - np.amin(array)
    hist, bins = np.histogram(array, par.rgb_nbins, density=True)
    hist = hist[0:par.rgb_nbins-1]
    bins = bins[0:par.rgb_nbins-1]
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    # find the edge values
    vv0 = bins[np.argmin(np.abs(cdf-par.edge_cut0[i]))]
    vv1 = bins[np.argmin(np.abs(cdf-par.edge_cut1[i]))]
    bins[bins<vv0] = vv0
    # use linear interpolation of cdf to find new pixel values
    array = np.interp(array, bins, cdf)
    if par.gamma[i] != 1:
        x1 = np.linspace(0, 1, num=par.rgb_nbins)
        cdf1 = x1**(par.gamma[i])
        array = np.interp(array, x1, cdf1)
    buff = norm_arr(array.reshape(n1, n2), par.rgb_vmin, par.rgb_vmax)
    rgb_corrected[:,:,i] = buff.copy()


# cut the edge values, do scaling, and modify the gamma values.
def cut_and_gamma(i):
    global rgb, rgb_corrected
    array = rgb[:,:,i].flatten()
    # find the edge values
    vv0 = get_pos_r(array, par.edge_cut0[i], par.rgb_nbins)
    vv1 = get_pos_r(array, par.edge_cut1[i], par.rgb_nbins)
    # normalize the array to 0~1
    upper_lim = (vv1 - vv0)*1.
    array = (array - vv0)/upper_lim
    array[array<0] = 0
    array[array>1] = 1
    # gamma correction
    array = array**(par.gamma[i])*(par.rgb_vmax - par.rgb_vmin)
    # # further scaling and normalization
    # array[array<par.rgb_vmin] = par.rgb_vmin
    # array[array>par.rgb_vmax] = par.rgb_vmax
    # save the result
    rgb_corrected[:,:,i] = array.reshape(n1, n2)


def norm_arr(array, vmin, vmax):
    x = array.copy()
    xmin = np.amin(x)
    xmax = np.amax(x)
    if xmin != xmax:
        x = (x-xmin)/(xmax-xmin)*(vmax-vmin) + vmin
    return x


if len(sys.argv)>1:
    par.working_dir = sys.argv[1]
if len(sys.argv)>2:
    par.gamma[0] = float(sys.argv[2])
    par.gamma[1] = float(sys.argv[3])
    par.gamma[2] = float(sys.argv[4])
if len(sys.argv)>5:
    par.vc0 = float(sys.argv[5])
if len(sys.argv)>6:
    par.vc1 = float(sys.argv[6])
if len(sys.argv)>7:
    par.hc0 = float(sys.argv[7])
if len(sys.argv)>8:
    par.hc1 = float(sys.argv[8])

vertical_clip   = [par.vc0, par.vc1]
horizontal_clip = [par.hc0, par.hc1]

gamma_str = str(par.gamma[0]).format("4.4i")+'-'+str(par.gamma[1]).format("4.4i")+'-'+str(par.gamma[2]).format("4.4i")
file_tif  = os.path.join(par.working_dir, "final_gamma"+gamma_str+".tiff")

print("Working directory  = %s"      %(par.working_dir))
print("Gamma              = %s"      %(par.gamma))
print("Parallel           = %s"      %(par.parallel))
print("Hist-equalization  = %s"      %(par.hist_eq))
print("Stack color mode   = %s"      %(par.stack_mode))
print("Horizontal reverse = %s"      %(par.hori_inv))
print("Vertical reverse   = %s"      %(par.vert_inv))
print("Horizontal range   = %s - %s" %(par.hc0, par.hc1))
print("Vertical range     = %s - %s" %(par.vc0, par.vc1))

# read the stacked frame and make sure the values are all positive
if par.stack_mode.lower() == 'color':
    frame_stacked = ave_by_channel('')
else:
    chn_L = ave_by_channel('L', ilc=par.ave_with_ilc)
    chn_R = ave_by_channel('R', ilc=par.ave_with_ilc)
    chn_G = ave_by_channel('G', ilc=par.ave_with_ilc)
    chn_B = ave_by_channel('B', ilc=par.ave_with_ilc)
    chn_H = ave_by_channel('H', ilc=par.ave_with_ilc)
    chn_S = ave_by_channel('S', ilc=par.ave_with_ilc)
    chn_O = ave_by_channel('O', ilc=par.ave_with_ilc)
    if   par.stack_mode.upper() == 'LRGB':    
        frame_stacked = comb_channels(chn_R, chn_G, chn_B, l=chn_L)
    elif par.stack_mode.upper() == 'LHSO':
        frame_stacked = comb_channels(chn_H, chn_S, chn_O, l=chn_L)
    elif par.stack_mode.upper() == 'LSHO':
        frame_stacked = comb_channels(chn_S, chn_H, chn_O, l=chn_L)
    elif par.stack_mode.upper() == 'RGB':
        frame_stacked = comb_channels(chn_R, chn_G, chn_B)
    elif par.stack_mode.upper() == 'HSO':
        frame_stacked = comb_channels(chn_H, chn_S, chn_O)
    elif par.stack_mode.upper() == 'SHO':
        frame_stacked = comb_channels(chn_S, chn_H, chn_O)
    else:
        print('Unknown stack mode, should be one of: color, LRGB, RGB, LHSO or HSO!')
        sys.exit()

# record the frame shape
shape = frame_stacked.shape
n1, n2 = shape[0], shape[1]

# clip image when necessary
v1, v2 = int((n1*vertical_clip[0])//2*2), int((n1*vertical_clip[1])//2*2)
h1, h2 = int((n2*horizontal_clip[0])//2*2), int((n2*horizontal_clip[1])//2*2)
frame_stacked = frame_stacked[v1:v2, h1:h2]
n1, n2 = v2-v1, h2-h1
print("Original size:    (hori, vert) = (%14i, %14i)" %(n2, n1) )
print("Working subframe: (hori, vert) = (%6i ~%6i, %6i ~%6i)" %(h1, h2, v1, v2) )

# convert stacked frame to linear rgb values (only handle the Bayer matrix)
tst = time.time()
if par.parallel == True:
    # start the shared memory manager
    smm = SharedMemoryManager()
    smm.start()

    if par.stack_mode.lower() == 'color':
        shm_rgb = smm.SharedMemory(frame_stacked.size*3*8)
    else:
        shm_rgb = smm.SharedMemory(frame_stacked.size*8)
    rgb = np.frombuffer(shm_rgb.buf, dtype=np.float64).reshape(n1, n2, 3)

    shm_rgb_corrected = smm.SharedMemory(rgb.size*rgb.itemsize)
    rgb_corrected = np.frombuffer(shm_rgb_corrected.buf, dtype=np.float64).reshape(n1, n2, 3)

    if par.stack_mode.lower() == 'color':
        rgb[:,:,:] = cv2.cvtColor(frame_stacked.astype(par.raw_data_type), par.bayer_matrix_format)
    else:
        rgb[:,:,:] = frame_stacked

    if __name__ == '__main__':
        if par.hist_eq==True:
            with mp.Pool(3) as pool:
                output = pool.map(hist_equal_gamma, range(3))
        else:
            with mp.Pool(3) as pool:
                output = pool.map(cut_and_gamma, range(3))
else:
    if par.stack_mode.lower() == 'color':
        rgb = cv2.cvtColor(frame_stacked, par.bayer_matrix_format)
    else:
        rgb = frame_stacked

    if par.hist_eq==True:
        rgb_corrected = rgb*0
        for i in range(3):
            hist_equal_gamma(i)
    else:
        rgb_corrected = rgb*0
        for i in range(3):
            cut_and_gamma(i)

print("Color correction done,    time cost: %9.2f" %(time.time()-tst) )


# Gaussian smoothing
if par.gauss_sigma != 0:
    print("Running %s pixel Gaussian smoothing" %(par.gauss_sigma))
    for i in range(3):
        rgb_corrected[:,:,i] = ndimage.gaussian_filter(rgb_corrected[:,:,i], sigma=par.gauss_sigma)

# Horizontal or vertial inverting
if par.hori_inv==True:
    rgb_corrected = np.flip(rgb_corrected, axis=1)
if par.vert_inv==True:
    rgb_corrected = np.flip(rgb_corrected, axis=0)

if par.wiener_filter==True:
    for i in range(3):
        rgb_corrected[:,:,i] = naive_wiener_filter(rgb_corrected[:,:,i], noise_fac=par.wiener_noise_fac)

# Down-sampling
if par.down_samp_fac>1:
    n1s = int(n1/par.down_samp_fac)
    n2s = int(n2/par.down_samp_fac)
    rgb_final = np.zeros([n1s, n2s, 3])
    for i in range(3):
        buff = rgb_corrected[:,:,i].copy()
        buff = buff.reshape(n1s, int(par.down_samp_fac), n2s, int(par.down_samp_fac))
        buff = np.mean(buff, axis=1)
        buff = np.mean(buff, axis=2)
        rgb_final[:,:,i] = buff
else:
    rgb_final = rgb_corrected.copy()

# adjust RGB by the scaling factor
for i in range(3):
    rgb_final[:,:,i] = rgb_final[:,:,i]*par.scaling_fac[i]

# save the 48-bit color image
imageio.imsave(file_tif, rgb_final.astype(par.raw_data_type))
print("Final image saved as %s" %(file_tif))

# show the image
if par.console_mode == False:
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb_final/256))
    plt.show()

if par.parallel == True:
    rgb, rgb_corrected = 0, 0
    smm.shutdown()