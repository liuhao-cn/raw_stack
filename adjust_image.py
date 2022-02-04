import cv2, os, imageio, time
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

# decrease this value will amplify the corresponding color
rgb_max = [65535,65535,65535]

rgb_min = [0, 0, 0]

rgb_nbins = [2048, 2048, 2048] # list(np.array(rgb_max) - np.array(rgb_min) + 1)

show_image = False

# increase this value will increase the contrast
rgb_gamma = [24, 24, 24]

# The bayer matrix format
bayer_matrix_format = cv2.COLOR_BayerBG2RGB
# bayer_matrix_format = cv2.COLOR_BayerBG2GRAY

working_dir = "/work/astro/fits1/output"

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
def hist_equal_gamma(array, vmin, vmax, nbins, gamma):
    # get image histogram
    array_op = array.flatten()
    hist, bins = np.histogram(array_op, nbins, density=True)
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    # use linear interpolation of cdf to find new pixel values
    array_op = np.interp(array_op, bins[:-1], cdf)
    if gamma != 1:
        x1 = np.linspace(0, 1, num=nbins)
        cdf1 = x1**(gamma)
        array_op = np.interp(array_op, x1, cdf1)
    return array_op.reshape(array.shape)*(vmax-vmin) + vmin

# do histogram equalization and gamma correction for a frame 
def color_correction(rgb, vmin, vmax, nbins, gamma):
    rgb_corrected = rgb*0
    for i in range(3):
        rgb_corrected[:,:,i] = hist_equal_gamma(rgb[:,:,i], vmin[i], vmax[i], nbins[i], gamma=gamma[i])
    return rgb_corrected.astype(raw_data_type)


os.chdir(working_dir)

frame_stacked, _ = read_fits(file_stacked)
frame_stacked = frame_stacked[1]

frame_stacked = frame_stacked - np.amin(frame_stacked)

# stacked frame to linear rgb values (only handle the Bayer matrix)
tst = time.time()
rgb = cv2.cvtColor(frame_stacked.astype(raw_data_type), bayer_matrix_format)
print("Stacked frame converted to RGB image, time cost: %9.2f" %(time.time()-tst))

# color correction
tst = time.time()
rgb_corrected = color_correction(rgb, rgb_min, rgb_max, rgb_nbins, rgb_gamma)
print("Color correction doen,                time cost: %9.2f" %(time.time()-tst))

print("Current color correction parameters: (r, g, b) ranges: %7i, %7i, %7i " 
    %(rgb_max[0], rgb_max[1], rgb_max[2]))
print("Current color correction parameters: (r, g, b) gamma: %7.2f, %7.2f, %7.2f " 
    %(rgb_gamma[0], rgb_gamma[1], rgb_gamma[2]))

# save the 48-bit color image
imageio.imsave(file_tif, rgb_corrected.astype(raw_data_type))

if show_image==True:
    plt.figure(figsize=(6,4),dpi=200)
    plt.xlabel('Y',fontsize=12)
    plt.ylabel('X',fontsize=12)
    plt.imshow(np.uint8(rgb_corrected/256))
