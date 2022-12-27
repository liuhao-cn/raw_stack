import cv2
import numpy as np

working_dir  = '/home/hao/astro/raw/2022-07-07_21_27_18Z/output/'
file_stacked = 'frame_stacked.fits'

rgb_vmin = 200
rgb_vmax = 64000

bayer_string = 'BGGR'

hori_inv = False
vert_inv = False

vc0 = 0.2
vc1 = 0.6
hc0 = 0.3
hc1 = 0.7

down_samp_fac = 1.0
rgb_nbins     = 16384
gamma         = [8, 64, 24]
gauss_sigma   = 0.0

multi_sess = False
console_mode = True

# the stack mode can be color, LHSO, HSO, LRGB, RGB
stack_mode = 'HSO'
hist_eq = True

chn_pattern = '_Bin2_'

show_image      = False
raw_data_type   = np.uint16

if bayer_string.lower()=="rggb":
    bayer_matrix_format = cv2.COLOR_BayerRG2RGB
elif bayer_string.lower()=="bggr":
    bayer_matrix_format = cv2.COLOR_BayerBG2RGB
elif bayer_string.lower()=="grbg":
    bayer_matrix_format = cv2.COLOR_BayerGR2RGB
else:
    bayer_matrix_format = cv2.COLOR_BayerGB2RGB
