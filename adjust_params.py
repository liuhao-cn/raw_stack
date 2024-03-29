import cv2
import numpy as np

working_dir  = '/home/hao/astro/raw/2022-07-07_21_27_18Z/output/aligned/'
extension    = 'fit'

rgb_vmin     = 0
rgb_vmax     = 65535

hori_inv = False
vert_inv = False

vc0 = 0.0
vc1 = 1.0
hc0 = 0.0
hc1 = 1.0

down_samp_fac = 2
rgb_nbins     = 8192
gamma         = [6, 6, 6]
gauss_sigma   = [0.5, 0.5, 0.5]

# cut very low and very high values at a given percentage, only for the
# case without histogram equalization.
edge_cut0     = [0.10, 0.10, 0.10]
edge_cut1     = [0.99, 0.99, 0.99]

# prior (physical) RGB combination factor, also applied to HSO
rgb_fac = [1.0, 1.0, 1.0]

# posterior (final, visual) scaling factors of the RGB color channels.
scaling_fac   = [1.00, 1.00, 0.95]

# average the frames with ILC?
# ilc_diag_fac means to amplify the diagonal factor for stabilization
ave_with_ilc = True
ilc_diag_fac = 1.1

parallel     = True
console_mode = True

# the stack mode can be color, LHSO, HSO, LSHO, SHO, LRGB, RGB
stack_mode = 'HSO'
hist_eq = True

# Use naive wiener filter to reduce the noise?
wiener_filter = False
wiener_noise_fac = 0.3

# pattern (prefix) of the channel name
chn_pattern = '_294MM_'

raw_data_type   = np.uint16

bayer_string = 'BGGR'
if bayer_string.lower()=="rggb":
    bayer_matrix_format = cv2.COLOR_BayerRG2RGB
elif bayer_string.lower()=="bggr":
    bayer_matrix_format = cv2.COLOR_BayerBG2RGB
elif bayer_string.lower()=="grbg":
    bayer_matrix_format = cv2.COLOR_BayerGR2RGB
else:
    bayer_matrix_format = cv2.COLOR_BayerGB2RGB
