import cv2, os, imageio, time
import numpy as np
import multiprocessing as mp
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy import ndimage

working_dir = "/work/astro/m42/output"

file_stacked = "frame_stacked.fits"

# decrease this value will amplify the corresponding color
rgb_vmax = [65535, 65535, 65535]

rgb_vmin = [0, 0, 0]

rgb_nbins = [16384, 16384, 16384]

# increase this value will increase the contrast
gamma = 4.0
rgb_gamma = [gamma, gamma, gamma]

# signam of the Gaussian filter kernel
gauss_sigma = 1

# Fractional horizontal and vertial clips, from the upper-left corner.
vertical_clip = [0., 1.]

horizontal_clip = [0., 1.]

show_image = True

file_tif = "final"+str(gamma).format("3.3i")+".tiff"


# The bayer matrix format
bayer_matrix_format = cv2.COLOR_BayerBG2RGB
# bayer_matrix_format = cv2.COLOR_BayerBG2GRAY

raw_data_type = np.uint16


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
    mask = np.where(f2>ratio, 0, 1)
    array_hp = np.fft.irfft2(np.fft.rfft2(array)*mask)
    return array_hp


# read frame from fits
def read_fits(file):
    with fits.open(file) as hdu_list:
        data_list = []
        hdr_list = []
        for hdu in hdu_list:
            data_list.append(hdu.data)
            hdr_list.append(hdu.header)
    return data_list, hdr_list


# do histogram equalization and gamma correction for one channel
def hist_equal_gamma(i):
    # get image histogram
    array = rgb[:,:,i].flatten()
    hist, bins = np.histogram(array, rgb_nbins[i], density=True)
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    # use linear interpolation of cdf to find new pixel values
    array = np.interp(array, bins[:-1], cdf)
    if rgb_gamma[i] != 1:
        x1 = np.linspace(0, 1, num=rgb_nbins[i])
        cdf1 = x1**(rgb_gamma[i])
        array = np.interp(array, x1, cdf1)
    buff = array.reshape(n1, n2)*(rgb_vmax[i]-rgb_vmin[i]) + rgb_vmin[i]
    rgb_corrected[:,:,i] = buff


os.chdir(working_dir)

frame_stacked, _ = read_fits(file_stacked)
frame_stacked = frame_stacked[1]
frame_stacked = frame_stacked - np.amin(frame_stacked)

shape = frame_stacked.shape
n1, n2 = shape[0], shape[1]
v1, v2 = int((n1*vertical_clip[0])//2*2), int((n1*vertical_clip[1])//2*2)
h1, h2 = int((n2*horizontal_clip[0])//2*2), int((n2*horizontal_clip[1])//2*2)
frame_stacked = frame_stacked[v1:v2, h1:h2]
n1, n2 = v2-v1, h2-h1

print("Original size:    (hori, vert) = (%6i, %6i)" %(n2, n1) )
print("Work with subframe: horizontal = (%6i, %6i)" %(h1, h2) )
print("                      vertical = (%6i, %6i)" %(v1, v2) )

# stacked frame to linear rgb values (only handle the Bayer matrix)
tst = time.time()
rgb = cv2.cvtColor(frame_stacked.astype(raw_data_type), bayer_matrix_format)
print("Stacked frame converted to RGB image, time cost: %9.2f" %(time.time()-tst))

# rgb = np.where(rgb>8000, 8000, rgb)

tst = time.time()
rgb_corrected = rgb*0
for i in range(3):
    hist_equal_gamma(i)
# if __name__ == '__main__':
#     with mp.Pool(4) as pool:
#         pool.map(hist_equal_gamma, range(3))

print("Color correction doen,                time cost: %9.2f" %(time.time()-tst))
print("Current color correction parameters: (r, g, b) maximum: %7i, %7i, %7i " 
    %(rgb_vmax[0], rgb_vmax[1], rgb_vmax[2]))
print("Current color correction parameters: (r, g, b) gamma  : %7.2f, %7.2f, %7.2f " 
    %(rgb_gamma[0], rgb_gamma[1], rgb_gamma[2]))

for i in range(3):
    rgb_corrected[:,:,i] = ndimage.gaussian_filter(rgb_corrected[:,:,i], sigma=gauss_sigma)

# save the 48-bit color image
imageio.imsave(file_tif, rgb_corrected.astype(raw_data_type))

if show_image==True:
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb_corrected/256))
