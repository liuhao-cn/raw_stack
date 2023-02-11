import numpy as np

# longitude, latitude, altitude and time zone of the observatory
site_lon     = 117.181
site_lat     =  31.772
site_alt     =  43.000
tzone        =   8

# (ra, dec) of the observation target, only for rotation correction
ra           =  83.818662
dec          =  -5.389679

# Alignment parameters
#
# align_color_mode: Camera color type for alignment. "color" means RGB (in
# form of Bayer components) are aligned separately, which will automatically
# correct the color dispersion due to atmosphere refraction. "mono" means the
# entire frame is aligned as one, which is better for a low signal-to-noise
# ratio.
#
# align_hp_ratio: 2D High-pass cut frequency as a fraction of the Nyquist
# frequency. Only for alignment to reduce background impacts. Will not affect
# the actual frames.
#
# align_tukey_alpha: Tukey window alpha parameter to improve the matched
# filtering. For example, 0.04 means 2% at each edge (left, right, top,
# bottom) is suppressed. Note that this should match the maximum shifts
#
# align_gauss_sigma: sigma of Gaussian filtering (smoothing), only for
# alignment.
#
# align_rounds: rounds of alignment. In each round a new (better) reference
# frame is chosen. 4 is usually more than enough.
#
# align_fix_rotation: Specify whether or not to fix the field rotation
# (requires the observation time, target and site locations). For an Alt-az
# mount, this is necessary, but for an equatorial mount this is unnecessary.
#
# align_time_is_utc: Specify whether or not the original observation time is
# already in UTC. If this is False, then a time zone correction will be
# applied.
#
# align_report: If false, do not report the alignment result
#
# align_save: If true, the aligned frames will be saved. Nesessary for
# multi-channel photos.
#
align_color_mode    = 'mono'
align_hp_ratio      = 0.010
align_tukey_alpha   = 0.200
align_gauss_sigma   = 4
align_rounds        = 4
align_fix_rotation  = False
align_time_is_utc   = True
align_report        = False

# File and folder parameters
#
# working_dir: Working directory, all raw or fits files should be in this
# directory. Will be overwritten by the command-line parameter.
#
# extension: Define the input file extension. All files in the working
# directory with this extension will be used. If this is fits or fit, work in
# fits mode (usually for an astro-camera), otherwise work in raw mode (usually
# for a DSLR). Will be overwritten by the command-line parameter.
#
# bad_fraction: Fraction of frames that will not be used
#
# page_num: Page number of data in the fits file
#
# date_tag: Tag in fits file for the obs. date and time information
#
# output_dir: output sub-directory
#
working_dir  = '/home/hao/astro/raw/2022-07-07_21_27_18Z'
extension    = 'fit'
bad_fraction = 0.00
date_tag     = 'DATE-OBS'
output_dir   = 'aligned'

# working precision for real and complex numbers
working_precision         = np.dtype('float32')
working_precision_complex = np.dtype('complex64')

# bias, dark, and flat corrections
fix_bias    = True
fix_dark    = True
fix_flat    = True
flat_suffix = '-flat.fits'

# time for the light and dark frames. The scaling factor of the dark frame is
# frame time/ dark time
frame_time = 2000.
dark_time =  2400.

bias_file = '/home/hao/astro/bias-dark-flat/294mm-pro-bin2/bias-master.fits'
dark_file = '/home/hao/astro/bias-dark-flat/294mm-pro-bin2/dark-master.fits'
dir_flat  = '/home/hao/astro/2023-01-30/Flat'

flat_channels = 'HSOLRGB'
chn_pattern   = '_294MM_'

# number of processes to be used, 0 means auto detect.
nproc_setting = 0

# fix local extrema?
fix_local_extrema = False
fac_local_extrema = 0.00003

# other parameters. 
# 
# console: If true, work in console mode, no online figures.
#
# adc_digit_limit: upper limit for the ADC digit, should usually be 16.
#
# final_file_fits: the final output file.
#
console         = True
adc_digit_limit = 16
