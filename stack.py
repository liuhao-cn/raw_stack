#!/usr/bin/env python
# coding: utf-8

##############################################################
# Stack the images in fits or raw files.
##############################################################

# Note that the flat file should be in dir_flat and be names like
# 'L-flat.fit', 'H-flat.fit'...


##############################################################
# Imports
##############################################################
import os, time, rawpy, sys, matplotlib, re

import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import heapq as hp
import astropy.units as u

from scipy import ndimage
from scipy import fft
from scipy.stats import linregress
from scipy.signal.windows import tukey
from datetime import datetime
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time as astro_time
from astropy.time import TimezoneInfo
from multiprocessing.managers import SharedMemoryManager


# read in parameters from the excel form
df = pd.read_excel("stack_params.xlsx")

# longitude, latitude, altitude and time zone of the observatory
site_lon     = float(df[2][1])
site_lat     = float(df[2][2])
site_alt     = float(df[2][3])
tzone        =   int(df[2][4])

# (ra, dec) of the observation target
ra           = float(df[5][1])
dec          = float(df[5][2])

# Alignment parameters
#
# align_color_mode: Camera color type for alignment. "color" means RGB
# (in form of Bayer components) are aligned separately, which will
# automatically correct the color dispersion due to atmosphere
# refraction. "mono" means the entire frame is aligned as one, which is
# much better for a low signal-to-noise ratio.
#
# align_hp_ratio: 2D High-pass cut frequency as a fraction of the
# Nyquist frequency. Only for alignment to reduce background impacts.
# Will not affect the actual frames.
#
# align_tukey_alpha: Tukey window alpha parameter to improve the matched
# filtering. For example, 0.04 means 2% at each edge (left, right, top,
# bottom) is suppressed. Note that this should match the maximum shifts
#
# align_gauss_sigma: sigma of Gaussian filtering (smoothing), only for
# alignment.
#
# align_rounds: rounds of alignment. In each round a new (better)
# reference frame is chosen. 4 is usually more than enough.
#
# align_fix_ratation: Specify whether or not to fix the field rotation
# (requires the observation time, target and site locations). For an
# Alt-az mount, this is necessary, but for an equatorial mount this is
# unnecessary.
#
# align_time_is_utc: Specify whether or not the original observation
# time is already in UTC. If this is False, then a time zone correction
# will be applied.
#
# align_report: If true, do not report the alignment result
#
align_color_mode    =   str(df[5][ 6])
align_hp_ratio      = float(df[5][ 7])
align_tukey_alpha   = float(df[5][ 8])
align_gauss_sigma   = float(df[5][ 9])
align_rounds        =   int(df[5][10])
align_fix_ratation  = (df[5][11].lower() == "true")
align_time_is_utc   = (df[5][12].lower() == "true")
align_report        = (df[5][13].lower() == "true")
align_save          = (df[5][14].lower() == "true")

# File and folder parameters
#
# working_dir: Working directory, all raw or fits files should be in
# this directory. Will be overwritten by the command-line parameter.
#
# extension: Define the input file extension. All files in the working
# directory with this extension will be used. If this is fits or fit,
# work in fits mode (usually for an astro-camera), otherwise work in raw
# mode (usually for a DSLR). Will be overwritten by the command-line
# parameter.
#
# bad_fraction: Fraction of frames that will not be used
#
# page_num: Page number of data in the fits file
#
# date_tag: Tag in fits file for the obs. date and time information
#
# output_dir: output directory
#
working_dir  =   str(df[2][7])
extension    =   str(df[2][8])
bad_fraction = float(df[2][9])
page_num     =   int(df[2][10])
date_tag     =   str(df[2][11])
output_dir   =   str(df[2][12])

# working precision for real and complex numbers
working_precision         = np.dtype(df[5][17])
working_precision_complex = np.dtype(df[5][18])

# bias, dark, and flat corrections
fix_bias = df[2][19].lower() == "true"
fix_dark = df[2][20].lower() == "true"
fix_flat = df[2][21].lower() == "true"

bias_file = df[2][23]
dark_file = df[2][24]
dir_flat  = df[2][25]

flat_channels = df[5][23].upper()
chn_pattern = df[5][24]

nproc_setting = int(df[5][21])

# fix local extrema?
fix_local_extrema = df[2][27].lower() == "true"
fac_local_extrema = float(df[2][28])

# other parameters. 
# 
# console: If true, work in console mode, no online figures.
#
# adc_digit_limit: upper limit for the ADC digit, should usually be 16.
#
# final_file_fits: the final output file.
#
console         = (df[2][15].lower() == "true")
adc_digit_limit = int(df[2][16])
final_file_fits = str(df[2][17])


obs_site = EarthLocation(lon=site_lon*u.deg, lat=site_lat*u.deg, height=site_alt*u.m)
# or by name, like target = SkyCoord.from_name("m42")
target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
# The maximum number of processes, overwritten by the command-line input
if nproc_setting == 0:
    nproc_max = int(mp.cpu_count()/2)
else:
    nproc_max = nproc_setting
# raw data maximum value
vmax_global = 2**adc_digit_limit - 1



##############################################################
# Fits related routines:

def read_frame_simple(file):
    with fits.open(file) as hdu:
        n = len(hdu)
        frame = hdu[n-1].data
    return frame

# Get the flat frame automatically by the channel name. Note that the flat
# file should be in dir_flat and be names like 'L-flat.fit', 'H-flat.fit'...
def get_channel(filename, chn_pat=chn_pattern):
    chn_pat = chn_pat + '.'
    name_loc = len(chn_pat) - 1
    pattern = re.compile(chn_pat)
    res = pattern.search(filename)
    chn_name = res.group(0)[name_loc]
    return chn_name

# read the bias, dark and flat frames. Note that the flat file should be in
# dir_flat and be named like 'L-flat.fits', 'H-flat.fits'...
def get_bias_dark_flat():
    flat_suffix = '-flat.fits'
    
    if fix_bias==True:
        bias = read_frame_simple(bias_file)
    else:
        bias = 0.

    if fix_dark==True:
        dark = read_frame_simple(dark_file)
        if fix_bias==True:
            dark = dark - bias
    else:
        dark = 0.

    if fix_flat==True:
        flat = {'L':None}
        for chn in flat_channels:
            file_flat = os.path.join(dir_flat, chn) + flat_suffix
            frame = read_frame_simple(file_flat)
            if fix_bias==True:
                frame = frame - bias
            if fix_dark==True:
                frame = decorr(dark, frame)
            flat[chn] = frame.copy()
    else:
        flat = 1.
    
    return bias, dark, flat

# read fits or raw file and convert to bin so it can be used by
# multiprocessing returns: frame, time, date type
def read_frame_fits(file):
    with fits.open(file) as hdu:
        frame = hdu[page_num].data
        data_type = frame.dtype
        hdr = hdu[page_num].header
        date_str = hdr[date_tag]
        if fix_bias==True:
            frame = frame - bias
        if fix_dark==True:
            frame = decorr(dark, frame)
        if fix_flat==True:
            fac = flat[get_channel(file)]
            fac = fac / np.amax(fac)
            frame = frame / fac
    return frame, date_str, data_type.name

def read_frame_raw(file):
    with rawpy.imread(file) as raw:
        data_type = raw.raw_image.dtype
        frame = raw.raw_image.copy()
    return frame, None, data_type.name

# make HDU with property cards
def data2hdu(data, cards=None, primary=False):
    # Make HDU
    if primary:
        hdu = fits.PrimaryHDU(data)
    else:
        hdu = fits.ImageHDU(data)
    # Set property cards.read_fits
    if cards != None:
        hdr = hdu.header
        for card in cards:
            hdr.set(card[0], card[1], card[2])
    return hdu

# write fits file from a list of HDU
def hdu2fits(list_of_hdu, file, overwrite=True):
    hdu_list = fits.HDUList(list_of_hdu)
    hdu_list.writeto(file, overwrite=overwrite)
    hdu_list.close()

# simple program to write array to fits file
def write_fits_simple(list_of_data, file, overwrite=True):
    list_of_hdu = [fits.PrimaryHDU(None)]
    for data in list_of_data:
        list_of_hdu.append(data2hdu(data))
    hdu2fits(list_of_hdu, file, overwrite=overwrite)

def decorr(x, y):
    res = linregress(x.flatten(), y.flatten())
    return y - x*res.slope

# # read and average all frames in the folder
# def ave_frame(folder):
#     frame = None
#     n = 0.
#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if n==0:
#                 buff, _, _ = read_frame_fits(os.path.join(folder, file))
#                 frame = buff.copy()
#             else:
#                 buff, _, _ = read_frame_fits(os.path.join(folder, file))
#                 frame = frame + buff
#             n = n + 1.
#         break
#     return frame/n


##############################################################
# FFT routines:
##############################################################
# compute 2d real FFT frequencies as a fraction of the Nyquist frequency
def rfft2_freq(n1, n2):
    n2r = n2//2 + 1
    f1 = np.repeat(fft.fftfreq(n1), n2r).reshape(n1,n2r)
    f2 = np.repeat(fft.rfftfreq(n2),n1).reshape(n2r,n1).transpose()
    f = np.sqrt(f1**2 + f2**2)
    return f

# wrapper of the frame-to-fft process, including pre-processing:
#
# 1. A Tukey window to suppress the edge effect.
#
# 2. A pre-process before the matched filtering, including a Gaussian
#    filtering (smoothing) to suppress the abnormally high and non-Gaussian
#    CMOS noise peaks, which could corrupt the matched filtering.
#
# 3. Explicit control of the precision to save memory.
#
def frame2fft(frame):
    shape = frame.shape

    if align_color_mode=="color":
        # reshape to separate the Bayer components
        frame_fft = np.zeros([f1s*2, f2s*2], dtype=working_precision_complex)
        for jj in range(4):
            frame1 = get_Bayerframe(frame, jj)*win
            frame1 = ndimage.gaussian_filter(frame1, sigma=align_gauss_sigma).astype(working_precision)
            frame_fft = put_Bayerframe(frame_fft, fft.rfft2(frame1), jj)
            # frame_fft[:,jj,:,kk] = fft.rfft2(frame1)
    else:
        frame = frame.reshape(n1, n2)
        frame1 = (frame*win).astype(working_precision)
        frame1 = ndimage.gaussian_filter(frame1, sigma=align_gauss_sigma).astype(working_precision)
        frame_fft = fft.rfft2(frame1).astype(working_precision_complex)
    
    frame = frame.reshape(shape)
    return frame_fft



##############################################################
# Alignment routines:
##############################################################
# normalize periodical data to -period/2~period/2 with circular statistics
def periodical_norm(x, period):
    ang_rad = x/period*2*np.pi
    cx = np.cos(ang_rad)
    sx = np.sin(ang_rad)
    ang_rad = np.arctan2(sx, cx)
    x1 = ang_rad/2/np.pi*period
    return x1

# compute the mean of periodical data
def periodical_mean(x, period):
    ang_rad = x/period*2*np.pi
    cx = np.mean( np.cos(ang_rad) )
    sx = np.mean( np.sin(ang_rad) )
    ang_mean = np.arctan2(sx, cx)
    x_mean = ang_mean/2/np.pi*period
    return x_mean

# get or put a Bayer-frame
def get_Bayerframe(frame, index):
    kk = index % 2
    jj = int((index - kk)/2)
    n1, n2 = frame.shape[0], frame.shape[1]
    frame = frame.reshape(int(n1/2), 2, int(n2/2), 2)
    subframe = (frame[:,jj,:,kk]).reshape(int(n1/2), int(n2/2))
    frame = frame.reshape(n1, n2)
    return subframe

def put_Bayerframe(frame, subframe, index):
    kk = index % 2
    jj = int((index - kk)/2)
    n1, n2 = frame.shape[0], frame.shape[1]
    frame_copy = frame.reshape(int(n1/2), 2, int(n2/2), 2)
    frame_copy[:,jj,:,kk] = subframe.reshape(int(n1/2), int(n2/2))
    frame_copy = frame_copy.reshape(n1, n2)
    return frame_copy


def fix_extrema(i):
    global frames_working

    if fix_local_extrema is True:
        frame = frames_working[i,:,:]

        # reshape to separate the Bayer components
        for jj in range(4):
            frame1 = get_Bayerframe(frame, jj)

            frame1[frame1==0] = 1
            fsize = frame1.size
            n_bad = int(fsize*fac_local_extrema)
            buff1 = np.roll(frame1, ( 0, 1)).flatten()
            buff2 = np.roll(frame1, ( 0,-1)).flatten()
            buff3 = np.roll(frame1, ( 1, 0)).flatten()
            buff4 = np.roll(frame1, (-1, 0)).flatten()
            s1 = np.roll(frame1, ( 0, 4)).flatten()
            s2 = np.roll(frame1, ( 0,-4)).flatten()
            s3 = np.roll(frame1, ( 4, 0)).flatten()
            s4 = np.roll(frame1, (-4, 0)).flatten()
            frame1 = frame1.flatten()
            
            diff1 = np.abs(frame1.flatten() - buff1)**(0.25)
            diff2 = np.abs(frame1.flatten() - buff2)**(0.25)
            diff3 = np.abs(frame1.flatten() - buff3)**(0.25)
            diff4 = np.abs(frame1.flatten() - buff4)**(0.25)
            diff = diff1*diff2*diff3*diff4/np.abs(frame1)
            thr = hp.nlargest(n_bad, diff.flatten())[n_bad-1]
            
            id_nan = np.where( diff>thr )[0]
            frame1[id_nan] = (s1[id_nan] + s2[id_nan] + s3[id_nan] + s4[id_nan])/4

            frame = put_Bayerframe(frame, frame1, jj)

        frames_working[i,:,:] = frame.reshape(n1, n2)

# align a frame to the reference frame, do it for the four Bayer components
# separately so the color dispersion will be corrected at the same time.
def align_frames(i):
    global frames_working

    tst = time.time()

    frame = frames_working[i,:,:]
    frame_fft = frame2fft(frame)

    # For a color camera, the four Bayer components are aligned separately to
    # fix the color dispersion. However, for a mono-camera, the alignment is
    # done with the average offsets (computed with circular statistics).
    if align_color_mode=="color":
        s1 = np.zeros(4, dtype=np.int32)
        s2 = np.zeros(4, dtype=np.int32)

        # compute the frame offset for each Bayer component and save the offsets
        for jj in range(4):
            ff1 = get_Bayerframe(ref_fft, jj)
            ff2 = get_Bayerframe(frame_fft, jj)
            buff_local = np.abs( fft.irfft2(ff1*ff2*mask_hp) ).astype(working_precision_complex)
            index = np.unravel_index(np.argmax(buff_local, axis=None), buff_local.shape)
            s1[jj], s2[jj] = -index[0], -index[1]

        # apply the offset correction
        s1 = np.round(periodical_norm(s1, n1s)).astype(np.int32)
        s2 = np.round(periodical_norm(s2, n2s)).astype(np.int32)
        for jj in range(4):
            # fix the offset for one Bayer component and save into the result array
            frame1 = get_Bayerframe(frame, jj)
            frame1 = np.roll(frame1, (s1[jj], s2[jj]), axis=(0,1)) # - np.mean(frame1)
            frame = put_Bayerframe(frame, frame1, jj)
    else:
        # compute the frame offset
        fft_comb = (ref_fft*frame_fft*mask_hp).astype(working_precision_complex)
        buff_local = np.abs( fft.irfft2(fft_comb) )
        index = np.unravel_index(np.argmax(buff_local, axis=None), buff_local.shape)
        s1, s2 = -index[0], -index[1]

        # Avoid corruption of the Bayer matrix
        s1 = s1 - np.mod(s1, 2)
        s2 = s2 - np.mod(s2, 2)

        # apply the offset correction
        s1 = np.round(periodical_norm(s1, n1)).astype(np.int32)
        s2 = np.round(periodical_norm(s2, n2)).astype(np.int32)
        frame = np.roll( frame, (s1, s2), axis=(0,1) ) #- np.mean(frame)

    # save the aligned binaries (without mean)
    frames_working[i,:,:] = frame.reshape(n1, n2)

    if align_report==True:
        print("\nFrame %6i (%s) aligned in %8.2f sec, (sx, sy) = (%8i,%8i)." %(i, file_lst[i], time.time()-tst, s1, s2))
    
    if align_color_mode=="color":
        return i, s1.flatten(), s2.flatten()
    else:
        return i, s1, s2

# parse the offsets from the starmap() output list
def parse_offsets(out_list):
    sx, sy = [], []
    for i in range(n_files):
        sx.append(out_list[i][1])
        sy.append(out_list[i][2])
    sx = periodical_norm( np.array(sx), n1s )
    sy = periodical_norm( np.array(sy), n2s )
    return sx, sy



##############################################################
# Weights and covariance computation
##############################################################
def compute_weights(frames_working):
    # compute the covariance matrix (note that the mean value of the frame was
    # already removed)
    tst = time.time()
    buff = frames_working.reshape(n_files, n1*n2)
    cov = np.dot( buff, buff.transpose() )

    # compute weights from the covariance matrix
    w = np.zeros(n_files)
    for i in range(n_files):
        w[i] = np.sum(cov[i,:])/cov[i,i] - 1
    print("Computing weights... done, time cost:                     %9.2f" %(time.time()-tst)); tst = time.time()

    return w



##############################################################
# Field rotation routines
##############################################################
# convert elevation and azimuth to unit vectors
def elaz2vec(el, az):
    n = np.size(el)
    d2r = np.pi/180
    vec = np.zeros([n, 3])
    vec[:,0] = np.cos(el*d2r)*np.cos(az*d2r)
    vec[:,1] = np.cos(el*d2r)*np.sin(az*d2r)
    vec[:,2] = np.sin(el*d2r)
    return vec

def vec_norm(vec):
    return np.sqrt(np.sum(vec*vec, axis=1))

def unit_vec(vec):
    amp = vec_norm(vec)
    vec_nml = vec.copy()
    vec_nml[:,0] = vec_nml[:,0] / amp
    vec_nml[:,1] = vec_nml[:,1] / amp
    vec_nml[:,2] = vec_nml[:,2] / amp
    return vec_nml

# Convert a round angle series to an equivalent linear angle series. The idea
# is: when the difference between two neighbors is bigger than a threshold,
# all following angles are corrected by 360N degree to minimize the
# difference.
def periodical2linear(ang_in, deg=True, threshold=350):
    n = np.size(ang_in)
    ang = ang_in.copy()
    d2r = np.pi/180
    if deg==True:
        ang = ang*d2r

    for i in range(n-1):
        dif = ang[i+1] - ang[i]
        if np.abs(dif)>threshold*d2r:
            dif1 = np.arctan2( np.sin(dif), np.cos(dif) )
            ang[i+1:n] = ang[i+1:n] + dif1 - dif
    return ang/d2r

# compute the field rotation, return continuous angle, but can also return
# arccos values by debug=True (discontinuous, only for debug)
def compute_field_rot(target, hor_ref_frame, debug=False):
    north_pole = SkyCoord(ra=0*u.deg, dec=90*u.deg)
    north_pole_hor = north_pole.transform_to(hor_ref_frame)
    target_coord_hor = target.transform_to(hor_ref_frame)
    # convert el-az to unit vectors
    el_target = target_coord_hor.alt.to_value()
    az_target = target_coord_hor.az.to_value()
    el_np = north_pole_hor.alt.to_value()
    az_np = north_pole_hor.az.to_value()
    vec_target_hor      = elaz2vec(el_target, az_target)
    vec_north_pole_hor  = elaz2vec(el_np, az_np)
    vec_zenith          = vec_north_pole_hor*0; vec_zenith[:,2] = 1
    # compute local east by cross product, with normalization
    east_cel = unit_vec(np.cross(vec_target_hor, vec_north_pole_hor))
    east_hor = unit_vec(np.cross(vec_target_hor, vec_zenith        ))
    # determine the hor-to-cel rotation direction (sign of rotation)
    vec_cel2hor = np.cross(east_cel, east_hor)
    flag = np.sum(vec_cel2hor*vec_target_hor, axis=1)
    flag = np.where(flag>0, 1, -1)
    # compute field rotation angle by arctan2
    val_cos = np.sum(east_cel*east_hor, axis=1)
    val_sin = vec_norm(vec_cel2hor)
    rot_ang = np.arctan2(val_sin, val_cos)*180/np.pi
    # and determine the direction of rotation
    rot_ang = rot_ang * flag
    if debug==True:
        rot_ang = np.arccos(val_cos)*180/np.pi
    return rot_ang

# note: need to check if we should multiply -1 to angle. 
# note: the four Bayer blocks are rotated separately so as not to corrupt the Bayer matrix.
def fix_rotation(i):
    if align_color_mode=="color":
        frame = frames_working[i,:,:].reshape(n1s, 2, n2s, 2)
        for j in range(2):
            for k in range(2):
                frame1 = frame[:,j,:,k].reshape(n1s, n2s)
                frame1 = ndimage.rotate(frame1, -rot_ang[i], reshape=False)
                frame1 = np.where(frame1==0, np.median(frame1), frame1)
                frame[:,j,:,k] = frame1
    else:
        frame = frames_working[i,:,:].reshape(n1, n2)
        frame = ndimage.rotate(frame, -rot_ang[i], reshape=False)
        frame = np.where(frame==0, np.median(frame), frame)
    
    frames_working[i,:,:] = frame.reshape(n1, n2)



##############################################################
# Frame normalization
##############################################################
# re-scale array (linear) to [vmin, vmax] 
def rescale_array(x_in, vmin, vmax):
    x = x_in.copy()
    xmin, xmax = np.amin(x), np.amax(x)
    # first to range [0, 1]
    x = np.float64(x-xmin) / np.float64(xmax-xmin)
    # then to range [vmin, vmax]
    x = x*(vmax-vmin) + vmin
    return x

# Re-scale the stacked frame values to avoid problem in type conversion to
# uint16. Do it separately for the four Bayer matrices.
def normalize_frame(frame, vmin, vmax):
    frame_norm = frame.copy()
    shape = np.shape(frame_norm)
    n1, n2 = shape[0], shape[1]
    frame_norm = frame_norm.reshape(int(n1/2), 2, int(n2/2), 2)
    frame_norm[:,0,:,0] = rescale_array(frame_norm[:,0,:,0], vmin, vmax)
    frame_norm[:,0,:,1] = rescale_array(frame_norm[:,0,:,1], vmin, vmax)
    frame_norm[:,1,:,0] = rescale_array(frame_norm[:,1,:,0], vmin, vmax)
    frame_norm[:,1,:,1] = rescale_array(frame_norm[:,1,:,1], vmin, vmax)
    return frame_norm.reshape(n1, n2)
    



##############################################################
# Do the following:
# 1. Align the frames using the initial reference frame.
# 2. Compute the weights from the covariance matrix.
# 3. Set the reference frame to the one with highest weight and iterate.
# 4. Stack the frames with the final weights.
##############################################################



##############################################################
# Initialization
##############################################################
tst_tot = time.time()
if console == True:
    # In console mode, do not produce online images (but will save pdf
    # instead)
    matplotlib.use('Agg')
    
    if len(sys.argv)>1:
        working_dir = sys.argv[1]
    if len(sys.argv)>2:
        bad_fraction = float(sys.argv[2])
    if len(sys.argv)>3:
        extension = sys.argv[3]
    if len(sys.argv)>4:
        nproc_max = int(sys.argv[4])
    
    print("Working directory:         %s" %(working_dir))
    print("Input file extension:      %s" %(extension))
    print("Number of processes limit: %s" %(nproc_max))
else:
    # In non-console mode, improve the display effect of Jupyter notebook
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))

# check the color type
if align_color_mode!="color" and align_color_mode!="mono":
    print("Error! The color type has to be color or mono!")
    sys.exit()

# get the bias, dark and flat corrections
bias, dark, flat = get_bias_dark_flat()

# determine the working mode, currently only fits allows field rotation
# correction
if extension=="fits" or extension=='fit':
    mode = "fits"
else:
    mode = "raw"
    align_fix_ratation = False
    align_time_is_utc = False

# make a list of working files and determine the number of processes to be used
os.chdir(working_dir)
fullpath = os.path.abspath("./")    
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# start the shared memory manager
smm = SharedMemoryManager()
smm.start()

if __name__ == '__main__':
    # list input files, sort and build auxiliary file list accordingly
    list1 = []
    for file in os.listdir():
        if file.endswith(extension):
            list1.append(file)
    list1.sort()
    file_lst = smm.ShareableList(list1)

    if len(file_lst) < 2:
        print("Too few files, check the dir/file name or extension?")
        print("Usage: python stack.py dir_of_inputs file_extension number_of_cores")
        print("or (on cluster): sbatch file_of_sbash_script stack.py dir_of_inputs file_extension number_of_cores")
        sys.exit()

    list1 = []
    for file in file_lst:
        list1.append(os.path.splitext(file)[0] + '.swp')    
    file_swp = smm.ShareableList(list1)

n_files = len(file_lst)

if nproc_max > n_files:
    nproc = n_files
else:
    nproc = nproc_max

# use the first file as the initial reference file and get size, type
# information
if __name__ == '__main__':
    if mode=="fits":
        frame, _, cache = read_frame_fits(file_lst[0]) 
    else:
        frame, _, cache = read_frame_raw(file_lst[0]) 
    info_lst = smm.ShareableList([cache, np.shape(frame)[0], np.shape(frame)[1]])

raw_data_type, n1, n2 = info_lst[0], info_lst[1], info_lst[2]

if align_color_mode=='color':
    n1s = int(n1/2)
    n2s = int(n2/2)
else:
    n1s = n1
    n2s = n2

f1s, f2s = n1s, n2s//2 + 1

# make and share window function, highpass mask.
if __name__ == '__main__':
    shm_win = smm.SharedMemory(n1s*n2s*working_precision.itemsize)
    shm_mask_hp = smm.SharedMemory(f1s*f2s*working_precision.itemsize)

    # make the 2D-tukey window
    buff = np.frombuffer(shm_win.buf, dtype=working_precision).reshape(n1s, n2s)
    w1 = tukey(n1s, alpha=align_tukey_alpha).reshape(n1s, 1)
    w2 = tukey(n2s, alpha=align_tukey_alpha).reshape(1, n2s)
    buff[:,:] = np.dot(w1, w2)

    freq = rfft2_freq(n1s, n2s)
    buff = np.frombuffer(shm_mask_hp.buf, dtype=working_precision).reshape(f1s, f2s)
    buff[:,:] = np.where(freq<align_hp_ratio, 0, 1)

    if align_color_mode=="color":
        ref_fft_size = 4*f1s*f2s*working_precision_complex.itemsize
    else:
        ref_fft_size = f1s*f2s*working_precision_complex.itemsize
    shm_ref_fft = smm.SharedMemory(ref_fft_size)

    buff = None

win = np.frombuffer(shm_win.buf, dtype=working_precision).reshape(n1s, n2s)
mask_hp = np.frombuffer(shm_mask_hp.buf, dtype=working_precision).reshape(f1s, f2s)
if align_color_mode=="color":
    ref_fft = np.frombuffer(shm_ref_fft.buf, dtype=working_precision_complex).reshape(f1s*2, f2s*2)
else:
    ref_fft = np.frombuffer(shm_ref_fft.buf, dtype=working_precision_complex).reshape(f1s, f2s)

# read all frames, and make the shared array
if __name__ == '__main__':
    # read frames, save the observation times, and compute the local
    # horizontal reference frames accordingly
    tst = time.time()
    shm_frames = smm.SharedMemory(np.int64(n_files)*np.int64(n1)*np.int64(n2)*working_precision.itemsize)
    buff = np.frombuffer(shm_frames.buf, dtype=working_precision).reshape(n_files, n1, n2)
    list1 = []
    
    for i in range(n_files):
        if mode=="fits":
            frame, datet, _ = read_frame_fits(file_lst[i])
        elif mode=="raw":
            frame, datet, _ = read_frame_raw(file_lst[i])
        buff[i,:,:] = frame
        list1.append(datet)
    datetime = smm.ShareableList(list1)
    buff = None
    print("Frames read and cached by the main proc, time cost:       %9.2f" %(time.time()-tst))

frames_working = np.frombuffer(shm_frames.buf, dtype=working_precision).reshape(n_files, n1, n2)

# make and share arrays of field rotation
if __name__ == '__main__' and align_fix_ratation==True:
    shm_rot_ang = smm.SharedMemory(n_files*working_precision.itemsize)

if align_fix_ratation==True:
    rot_ang = np.frombuffer(shm_rot_ang.buf, dtype=working_precision).reshape(n_files)

print("Initialization done, time cost:                           %9.2f" %(time.time()-tst_tot))



##############################################################
# Fix field rotations if necessary (multi-processes)
##############################################################
if __name__ == '__main__' and align_fix_ratation==True:
    tst = time.time()
    if align_time_is_utc==False:
        time_shift = tzone*u.hour
    else:
        time_shift = 0*u.hour
    obstime_list = astro_time(datetime) - time_shift
    hor_ref_frame = AltAz(obstime=obstime_list, location=obs_site)

    # compute the relative time in seconds, and subtract the median
    # value to minimize the rotations
    rel_sec = (obstime_list - obstime_list[0]).to_value(format='sec')
    rel_sec = rel_sec - np.median(rel_sec)
    
    # compute the absolute field rotation angles as "rot_ang"
    rot_ang = compute_field_rot(target, hor_ref_frame)
    rot_ang = rot_ang - np.median(rot_ang)
    print("Rotation angles computed, time cost:                      %9.2f" %(time.time()-tst))

    # plot the field rotation angle for test 
    plt.figure(figsize=(4,2), dpi=200)
    plt.title('The field rotation angle')
    plt.xlabel('Time (sec)', fontsize=9)
    plt.ylabel('Angle', fontsize=9)
    plt.plot(rel_sec, rot_ang, marker="o")
    plt.savefig(os.path.join(fullpath,output_dir,'field_rot_ang.pdf'))

    # fix the field rotation (multi-processes)
    tst = time.time()
    with mp.Pool(nproc) as pool:
        output = [pool.map(fix_rotation, range(n_files))]
    print("Field rotation fixed, time cost:                          %9.2f" %(time.time()-tst) )

# For all processes: if fix-rotation is required, then read rot_ang from file
# and reset the reference frame to the one with least rotation (frame already
# fixed)
if align_fix_ratation==True:
    wid = np.argmin(np.abs(rot_ang))
else:
    wid = 0



##############################################################
# Alignment (multi-processes)
##############################################################
if __name__ == '__main__':
    sx, sy = [], []

    tst_tot = time.time()

    # fix extrema
    tst = time.time()
    with mp.Pool(nproc) as pool:
        output = pool.map(fix_extrema, range(n_files))
    print("Extrema fixed, time cost:                                %9.2f" %(time.time()-tst))


    for nar in range(align_rounds):
        print("****************************************************")
        print("Alignment round %4i: Frame %4i is the current reference frame." %(nar+1, wid))
        print("The current reference file is: %s" %(file_lst[wid]))

        # compute the reference frame fft by the main process
        tst = time.time()
        if align_color_mode=="color":
            ref_fft[:,:] = np.conjugate( frame2fft(frames_working[wid,:,:]) )
        else:
            ref_fft[:,:] = np.conjugate( frame2fft(frames_working[wid,:,:]) )
        
        with mp.Pool(nproc) as pool:
            output = pool.map(align_frames, range(n_files))
        print("Alignment round %4i done, time cost:                     %9.2f" %(nar+1, time.time()-tst))
        
        # parse the output and save the offsets
        sx_now, sy_now = parse_offsets(output)
        sx.append(sx_now)
        sy.append(sy_now)

        # identify the frame of maximum weight, and use it as the new reference frame.
        w = compute_weights(frames_working)
        wid1 = np.argmax(w)
        if wid1 == wid: 
            break
        else:
            wid = wid1

    print(np.mean(frames_working[0,:,:]))

    print("****************************************************")
    print("Alignment done in %4i rounds, time cost:                 %9.2f" %(nar+1, time.time()-tst_tot))
    print("Frame %4i is the final reference frame." %(wid))
    print("The final reference file is: %s" %(file_lst[wid]))
    print("****************************************************")

    # exclude the low quality frames
    n_bad = int(n_files*bad_fraction)
    thr = hp.nsmallest(n_bad, w)[n_bad-1]
    if thr<0: thr = 0
    w = np.where(w <= thr, 0, w)
    w = w / np.sum(w)

    if align_save==True:
        print('Will save the aligned frames...')
        align_dir = os.path.join(fullpath, output_dir, 'aligned')
        if not os.path.isdir(align_dir):
            os.mkdir(align_dir)
        for i in range(n_files):
            if w[i] > 0:
                file_aligned = os.path.join(align_dir, file_lst[i])
                write_fits_simple([frames_working[i,:,:].astype(raw_data_type)], file_aligned, overwrite=True)

    # stack the frames with weights.
    tst = time.time()
    frame_stacked = np.dot(w, frames_working.reshape(n_files, n1*n2)).reshape(n1, n2)

    # normalize the stacked frames to avoid negative values (due to subtraction of mean values).
    frame_stacked = normalize_frame( frame_stacked, 0, vmax_global )

    file_stacked = os.path.join(fullpath,output_dir,final_file_fits)
    write_fits_simple([frame_stacked.astype(raw_data_type)], file_stacked, overwrite=True)

    print("Stacked frame obtained from %4i/%4i frames, time cost.    %9.2f" 
        %(n_files-n_bad, n_files, time.time()-tst))

    # plot the XY-shifts
    w_mask = np.where(w==0, np.nan, 1)

    plt.figure(figsize=(6,4), dpi=200)
    plt.title('XY shifts in pixel')
    plt.xlabel('X shifts', fontsize=9)
    plt.ylabel('Y shifts', fontsize=9)
    
    if align_color_mode=="color":
        if nar>0:
            plt.scatter(sy[-2][:,0]*w_mask, sx[-2][:,0]*w_mask, s=50, c='r', alpha=0.5, label='Round 1, Bayer-00')
            plt.scatter(sy[-2][:,1]*w_mask, sx[-2][:,1]*w_mask, s=30, c='g', alpha=0.5, label='Round 1, Bayer-01')
            plt.scatter(sy[-2][:,2]*w_mask, sx[-2][:,2]*w_mask, s=30, c='g', alpha=0.5, label='Round 1, Bayer-10')
            plt.scatter(sy[-2][:,3]*w_mask, sx[-2][:,3]*w_mask, s=10, c='b', alpha=0.5, label='Round 1, Bayer-11')
        
        plt.scatter(sy[-1][:,0]*w_mask, sx[-1][:,0]*w_mask, s=10, c='k', alpha=0.5, label='Round 2, Bayer-00')
        plt.scatter(sy[-1][:,1]*w_mask, sx[-1][:,1]*w_mask, s=10, c='k', alpha=0.5, label='Round 2, Bayer-01')
        plt.scatter(sy[-1][:,2]*w_mask, sx[-1][:,2]*w_mask, s=10, c='k', alpha=0.5, label='Round 2, Bayer-10')
        plt.scatter(sy[-1][:,3]*w_mask, sx[-1][:,3]*w_mask, s=10, c='k', alpha=0.5, label='Round 2, Bayer-11')
    else:
        TT = np.arange(n_files)
        if nar>0:
            plt.scatter(sy[-2]*w_mask, sx[-2]*w_mask, s=50, c=TT, alpha=0.5, label='Round 1')
        plt.scatter(sy[-1]*w_mask, sx[-1]*w_mask, s=15, c='k', alpha=0.5, label='Round 2')

    plt.legend()
    plt.savefig(os.path.join(fullpath,output_dir,'xy-shifts.pdf'))


    # plot the weights for test 
    plt.figure(figsize=(6,4), dpi=200)
    plt.title(r'Stacking weights ($w\times N_{frames}$)')
    plt.xlabel('Frame number', fontsize=9)
    plt.ylabel(r'$w\times N_{frames}$', fontsize=9)
    w1 = np.where(w==0, np.nan, w)
    w2 = np.where(w==0, np.nanmean(w1), np.nan)
    plt.plot(w1*n_files*(1-bad_fraction), marker="o", label='Valid')
    plt.plot(w2*n_files*(1-bad_fraction), marker="*", label='Invalid')
    plt.legend()
    plt.savefig(os.path.join(fullpath,output_dir,'weights.pdf'))

    print("Done!")


# explicitly release all "np.frombuffer()" variables before releasing
# the shared memory, otherwise smm.shutdown() is unable to release them.
buff, win, mask_hp, ref_fft, frames_working, rot_ang = 0, 0, 0, 0, 0, 0

smm.shutdown()




















# # # manually adjust the rgb range
# # def adjust_color_manual(rgb, rgb_min, rgb_max, rgb_gamma):
# #     global_max = vmax_global
# #     r = rescale_array(rgb[:,:,0], rgb_min[0], rgb_max[0])
# #     g = rescale_array(rgb[:,:,1], rgb_min[1], rgb_max[1])
# #     b = rescale_array(rgb[:,:,2], rgb_min[2], rgb_max[2])
# #     r = (r**rgb_gamma[0])*global_max
# #     g = (g**rgb_gamma[1])*global_max
# #     b = (b**rgb_gamma[2])*global_max
# #     rgb_adjusted = rgb*0
# #     rgb_adjusted[:,:,0] = r
# #     rgb_adjusted[:,:,1] = g
# #     rgb_adjusted[:,:,2] = b
# #     return rgb_adjusted


#     # tst = time.time()
#     # # stacked frame to linear rgb value (handle the Bayer matrix)
#     # rgb = frame2rgb_fits(frame_stacked)

#     # rgb_modified, cdf = color_correction(rgb, rgb_gamma, rgb_max=rgb_max, bins=32000)

#     # # save the 48-bit color image
#     # imageio.imsave(os.path.join(working_dir,output_dir,final_file_tif), 
#     #     rgb_modified.astype(raw_data_type))

#     # # show the 8-bit color image as a quick preview (lower quality)
#     # if console==False:
#     #     plt.figure(figsize=(6,4),dpi=200)
#     #     plt.xlabel('Y',fontsize=12)
#     #     plt.ylabel('X',fontsize=12)
#     #     plt.imshow(np.uint8(rgb_modified/256))
#     # print("****************************************************")
#     # print("Final image obtained with colors corrected by modified histogram equalization")
#     # print("Current color correction parameters: (r, g, b) ranges: %7i, %7i, %7i " 
#     #     %(rgb_max[0], rgb_max[1], rgb_max[2]))
#     # print("Current color correction parameters: (r, g, b) gamma: %7.2f, %7.2f, %7.2f " 
#     #     %(rgb_gamma[0], rgb_gamma[1], rgb_gamma[2]))
    
#     # # adjust the color and make the final 8-bit image
#     # tst = time.time()
#     # rgb = frame2rgb_fits(frame_stacked)

#     # r_bin_file = os.path.splitext(final_file_tif)[0] + '.r'; rgb[:,:,0].tofile(r_bin_file)
#     # g_bin_file = os.path.splitext(final_file_tif)[0] + '.g'; rgb[:,:,1].tofile(g_bin_file)
#     # b_bin_file = os.path.splitext(final_file_tif)[0] + '.b'; rgb[:,:,2].tofile(b_bin_file)
    
#     # # color correction in parallel
#     # m1, m2, npix = np.shape(rgb)[0], np.shape(rgb)[1], rgb.size
#     # tst = time.time()
#     # ic = [0, 1, 2]
#     # im1 = [m1, m1, m1]
#     # im2 = [m2, m2, m2]
#     # ifn = [r_bin_file, g_bin_file, b_bin_file]
#     # dtp = [raw_data_type, raw_data_type, raw_data_type]
#     # with mp.Pool(3) as pool:
#     #     output = [pool.starmap(adjust_color, zip(ic, im1, im2, ifn, dtp))]

#     # # read the color correction result and save to 
#     # rgb[:,:,0] = np.fromfile(r_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(r_bin_file)
#     # rgb[:,:,1] = np.fromfile(g_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(g_bin_file)
#     # rgb[:,:,2] = np.fromfile(b_bin_file, dtype=raw_data_type).reshape(m1, m2); os.remove(b_bin_file)
#     # print("Color adjusted, time cost:                                %9.2f" %(time.time()-tst)); tst = time.time()
    

#     # # save the final figure
#     # imageio.imsave(final_file_tif, rgb.astype(raw_data_type))

#     # # show the final figure
#     # plt.figure(figsize=(6,4),dpi=200)
#     # plt.xlabel('Y',fontsize=12)
#     # plt.ylabel('X',fontsize=12)
#     # plt.imshow(np.uint8(rgb))


# # import heapq as hp
# # # subroutine for adjusting the colors 
# # def adjust_color(i, m1, m2, bin_file, raw_data_type):
# #     # number of "too dark" pixels and threshold
# #     val_max = vmax_global
# #     samp = np.fromfile(bin_file, dtype=raw_data_type).reshape(m1*m2)
# #     npix = np.int64(m1)*np.int64(m2)
# #     ndark = int(npix*dark_frac)
# #     d1 = hp.nsmallest(ndark, samp)[ndark-1]*1.
# #     # number of "too bright" pixels and threshold
# #     nbright = int(npix*bright_frac)
# #     d2 = hp.nlargest(nbright, samp)[nbright-1]*1.
# #     # re-scaling, note that this requires 16-bit to save weak signal from bright sky-light
# #     # note that val is expected to be in range [0,1]. Out-of-range values will be truncated.
# #     val = np.float64(samp-d1)/np.float64(d2-d1)
# #     val = np.where(val<=0, 1./val_max, val)
# #     val = np.where(val >1, 1, val)
# #     samp = (val**rgb_gamma[i])*val_max*rgb_fac[i]
# #     samp = np.where(samp<      0,       0, samp)
# #     samp = np.where(samp>val_max, val_max, samp)
# #     samp.astype(raw_data_type).tofile(bin_file)


# # Define observation target, either in name or in ra-dec.
# # in name, use (for example): target = SkyCoord.from_name("m31"); 
# # in ra-dec, use (for example): target = SkyCoord(ra=45*u.deg, dec=45*u.deg)
# # For example: 
# # # M42 by ra-dec: 

# # # M31 by ra-dec: 
# # target = SkyCoord(ra=10.684793*u.deg, dec=41.269065*u.deg)
# # # Barnard 33 by ra-dec: 
# # target = SkyCoord(ra=85.24583333*u.deg, dec=-2.45833333*u.deg)

# # # M42 by name (reqires internet connection)
# # target = SkyCoord.from_name("m42")
# # # M31 by name (reqires internet connection)
# # target = SkyCoord.from_name("m31")







# ##############################################################
# # Alignment parameters
# ##############################################################
# # Camera color type for alignment. "color" means RGB (in form of Bayer
# # components) are aligned separately, which will automatically correct the
# # color dispersion due to atmosphere refraction. "mono" means the entire frame
# # is aligned as one, which is much better for a low signal-to-noise ratio.
# align_color_mode = "color"

# # 2D High-pass cut frequency as a fraction of the Nyquist frequency. Only for
# # alignment to reduce background impacts. Will not affect the actual frames.
# align_hp_ratio = 0.01

# # Tukey window alpha parameter to improve the matched filtering. For example,
# # 0.04 means 2% at each edge (left, right, top, bottom) is suppressed. Note
# # that this should match the maximum shifts
# align_tukey_alpha = 0.2

# # sigma of Gaussian filtering (smoothing), only for alignment.
# align_gauss_sigma = 8

# # rounds of alignment. In each round a new (better) reference frame is
# # chosen. 4 is usually more than enough.
# align_rounds = 4

# # Specify whether or not to fix the field rotation (requires the observation
# # time, target and site locations). For an Alt-az mount, this is necessary,
# # but for an equatorial mount this is unnecessary.
# align_fix_ratation = True

# # Specify whether or not the original observation time is already in UTC. If
# # this is False, then a time zone correction will be applied.
# align_time_is_utc = True

# # If true, do not report the alignment result
# align_report = False


# ##############################################################
# # Precision settings
# ##############################################################
# # Working precision takes effect in FFT and matrix multiplication
# working_precision = np.dtype("float32")

# # Working precision takes effect in FFT and matrix multiplication
# working_precision_complex = np.dtype("complex64")


# ##############################################################
# # Other parameters
# ##############################################################
# # Number of ADC digit. The maximum value should be 2**adc_digit-1. This should
# # usualLy be 16, even when the actual digit is less, e.g., 12 or 14 bits.
# adc_digit_limit = 16

# vmax_global = 2**adc_digit_limit - 1

# # Name of the final fits
# final_file_fits = "frame_stacked.fits"
