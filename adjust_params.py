import cv2
import numpy as np

working_dir  = '/home/hao/astro/raw/2022-07-07_21_27_18Z/output/'
file_stacked = 'frame_stacked.fits'

rgb_vmin = 0
rgb_vmax = 65535

bayer_string = 'BGGR'

hori_inv = False
vert_inv = False

vc0 = 0.0
vc1 = 1.0
hc0 = 0.0
hc1 = 1.0

down_samp_fac = 1.0
rgb_nbins     = 16384
gamma         = 32
gauss_sigma   = 0.0

multi_sess = False
console_mode = True

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
