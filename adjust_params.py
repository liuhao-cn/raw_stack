import cv2
import numpy as np

working_dir  = '/home/hao/astro/raw/2022-07-07_21_27_18Z/output/aligned/'
file_stacked = 'frame_stacked.fits'
extension    = 'fit'

rgb_vmin     = 0
rgb_vmax     = 65535
bayer_string = 'BGGR'

hori_inv = False
vert_inv = False

vc0 = 0.0
vc1 = 1.0
hc0 = 0.0
hc1 = 1.0

down_samp_fac = 1.0
rgb_nbins     = 8192
gamma         = [64, 128, 128]
gauss_sigma   = 0.0

# cut very low and very high values at a given percentage, only for the
# case without histogram equalization.
edge_cut0     = [0.05, 0.05, 0.05]
edge_cut1     = [0.95, 0.95, 0.95]

# scale the color channels with given factors, only for the case without
# histogram equalization.
scaling_fac   = [20.00, 5.00, 5.00]

multi_sess   = False
console_mode = True

# the stack mode can be color, LHSO, HSO, LRGB, RGB
stack_mode = 'HSO'
hist_eq = False

# pattern (prefix) of the channel name
chn_pattern = '_Bin2_'

raw_data_type   = np.uint16

if bayer_string.lower()=="rggb":
    bayer_matrix_format = cv2.COLOR_BayerRG2RGB
elif bayer_string.lower()=="bggr":
    bayer_matrix_format = cv2.COLOR_BayerBG2RGB
elif bayer_string.lower()=="grbg":
    bayer_matrix_format = cv2.COLOR_BayerGR2RGB
else:
    bayer_matrix_format = cv2.COLOR_BayerGB2RGB
