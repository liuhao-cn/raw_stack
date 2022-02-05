import cv2, os, imageio, time
import numpy as np
import multiprocessing as mp
import astropy.io.fits as fits
import matplotlib.pyplot as plt

# decrease this value will amplify the corresponding color
rgb_vmax = [65535, 65535, 65535]

rgb_vmin = [0, 0, 0]

rgb_nbins = [4096, 4096, 4096]

# Fractional horizontal and vertial clips, from the upper-left corner.
vertical_clip = [0.1, 0.7]

horizontal_clip = [0.05, 0.8]

show_image = True

# increase this value will increase the contrast
g = 2
rgb_gamma = [g, g, g]

# The bayer matrix format
bayer_matrix_format = cv2.COLOR_BayerBG2RGB
# bayer_matrix_format = cv2.COLOR_BayerBG2GRAY

working_dir = "E:/astro/temp"

file_stacked = "frame_stacked.fits"

file_tif = "final.tiff"

raw_data_type = np.uint16


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
    array = rgb[:,:,i]
    hist, bins = np.histogram(array.flatten(), rgb_nbins[i], density=True)
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    # use linear interpolation of cdf to find new pixel values
    array_equal = np.interp(array.flatten(), bins[:-1], cdf)
    if rgb_gamma[i] != 1:
        x1 = np.linspace(0, 1, num=rgb_nbins[i])
        cdf1 = x1**(rgb_gamma[i])
        array_equal = np.interp(array_equal, x1, cdf1)
    buff = array_equal.reshape(n1, n2)*(rgb_vmax[i]-rgb_vmin[i]) + rgb_vmin[i]
    rgb_corrected[:,:,i] = buff
#     rgb_corrected[:,:,i] = buff


os.chdir(working_dir)

frame_stacked, _ = read_fits(file_stacked)
frame_stacked = frame_stacked[1]
frame_stacked = frame_stacked - np.amin(frame_stacked)

shape = frame_stacked.shape
n1, n2 = shape[0], shape[1]
print(n1, n2)
v1, v2 = int((n1*vertical_clip[0])//2*2), int((n1*vertical_clip[1])//2*2)
h1, h2 = int((n2*horizontal_clip[0])//2*2), int((n2*horizontal_clip[1])//2*2)
frame_stacked = frame_stacked[v1:v2, h1:h2]
n1, n2 = v2-v1, h2-h1

print(v1, v2, h1, h2, n1, n2)

# stacked frame to linear rgb values (only handle the Bayer matrix)
tst = time.time()
rgb = cv2.cvtColor(frame_stacked.astype(raw_data_type), bayer_matrix_format)
print("Stacked frame converted to RGB image, time cost: %9.2f" %(time.time()-tst))

# rgb = np.where(rgb>8000, 8000, rgb)

tst = time.time()
rgb_corrected = rgb*0
for i in range(3):
    hist_equal_gamma(i)

print("Color correction doen,                time cost: %9.2f" %(time.time()-tst))
print("Current color correction parameters: (r, g, b) ranges: %7i, %7i, %7i " 
    %(rgb_vmax[0], rgb_vmax[1], rgb_vmax[2]))
print("Current color correction parameters: (r, g, b) gamma: %7.2f, %7.2f, %7.2f " 
    %(rgb_gamma[0], rgb_gamma[1], rgb_gamma[2]))

# save the 48-bit color image
imageio.imsave(file_tif, rgb_corrected.astype(raw_data_type))

if show_image==True:
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb_corrected/256))
