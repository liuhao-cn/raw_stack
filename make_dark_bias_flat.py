#!/usr/bin/env python
# coding: utf-8

import os, re, time, sys, matplotlib
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.stats import linregress


extension = 'fit'
date_tag  = 'DATE-OBS'

flat_channels     = 'HSOLRGB'
chn_pattern       = '_294MM_'
working_precision = np.float32

# read fits or raw file and convert to bin so it can be used by
# multiprocessing returns: frame, time, date type
def read_frame_fits(file):
    with fits.open(file) as hdu:
        page_num = len(hdu)-1
        frame = hdu[page_num].data
        frame = frame.astype(working_precision)
    return frame

# simple program to write array to fits file
def write_frame_fits(frame, file, cards=None, overwrite=True):
    hdu_prime = fits.PrimaryHDU(None)
    hdu_frame = fits.ImageHDU(frame)
    if cards != None:
        hdr = hdu_frame.header
        for card in cards:
            hdr.set(card[0], card[1], card[2])
    hdu_list = fits.HDUList( [hdu_prime, hdu_frame] )
    hdu_list.writeto(file, overwrite=overwrite)
    hdu_list.close()

 # parse the channel names
def read_channel_name(filename):
    chn_pat = chn_pattern
    chn_pat = chn_pat + '.'
    name_loc = len(chn_pat) - 1
    pattern = re.compile(chn_pat)
    res = pattern.search(filename)
    chn_name = res.group(0)[name_loc]
    return chn_name

# read and average all frames in the folder
def ave_frame(folder, outfile, write_file=True):
    print('Making %s...' %(outfile))
    frame, n = 0., 0.
    for root, dirs, files in os.walk(folder):
        for file in files:
            buff = read_frame_fits(os.path.join(folder, file))
            frame = frame + buff*1.
            n = n + 1.
        break
    frame = frame / n
    if write_file:
        write_frame_fits(np.float32(frame), outfile)
        print("%s writen, min=%16.3f, max=%16.3f" %(outfile, np.amin(frame), np.amax(frame)) )
    return frame

# read frames with the given channel and return the average.
def ave_by_channel(working_dir, channel, outfile):
    i = 0
    frame = 0
    for file in os.listdir(working_dir):
        if file.endswith(extension):
            file = os.path.join(working_dir, file)
            if channel != '':
                chn_name = read_channel_name(file)
                if chn_name.lower()==channel.lower():
                    frame = frame + read_frame_fits(file)*1.
                    i = i+1
            else:
                frame = frame + read_frame_fits(file)*1.
                i = i+1
    print('Number of files for channel %6s: %5i' %(channel, i))
    write_frame_fits(frame, outfile)
    if i==0:
        return 0
    else:
        return frame/i

def rescale(frame, e0=0.05, e1=0.95):
    nbins = 8192
    # get image histogram
    array = frame.copy()
    hist, bins = np.histogram(array, nbins, density=True)
    cdf = hist.cumsum()
    cdf = cdf/np.amax(cdf)
    # find the edge values
    for ind in range(nbins):
        if cdf[ind]>e0:
            vv0 = bins[ind]
            break
    for ind in range(nbins):
        if cdf[ind]>e1:
            vv1 = bins[ind]
            break
    # normalize the array
    upper_lim = (vv1 - vv0)*1.
    array = (array - vv0)/upper_lim
    array[array<0] = 0
    array[array>1] = 1
    return array
    
def show_frame(frame, fig, gs, row_id0, row_id1, col_id0, col_id1, title, e0=0.05, e1=0.95, color=False):
    cache = rescale(frame, e0=e0, e1=e1)

    ax = fig.add_subplot(gs[row_id0:row_id1, col_id0:col_id1])

    ax.tick_params(labelbottom=False, labelleft=False)

    if color:
        ax.imshow(cache)
    else:
        ax.imshow(cache, cmap='viridis')
    
    ax.set_title(title)
    return cache


# dir_main = 'D:/astro/raw/bias-dark-flat/294mm-pro-bin2/'
# bias   = ave_frame(dir_main+'bias/', dir_main+'bias-master.fits')
# dark   = ave_frame(dir_main+'dark/', dir_main+'dark-master.fits')

# make flat frame
working_dir = 'D:/astro/raw/2023-01-30/'

L_flat = ave_by_channel(working_dir+'Flat/', 'L', working_dir+'L-flat.fits')
R_flat = ave_by_channel(working_dir+'Flat/', 'R', working_dir+'R-flat.fits')
G_flat = ave_by_channel(working_dir+'Flat/', 'G', working_dir+'G-flat.fits')
B_flat = ave_by_channel(working_dir+'Flat/', 'B', working_dir+'B-flat.fits')
H_flat = ave_by_channel(working_dir+'Flat/', 'H', working_dir+'H-flat.fits')
S_flat = ave_by_channel(working_dir+'Flat/', 'S', working_dir+'S-flat.fits')
O_flat = ave_by_channel(working_dir+'Flat/', 'O', working_dir+'O-flat.fits')

fig = plt.figure(layout="tight", figsize=(12,6))
fig.suptitle("Flat")

from matplotlib.gridspec import GridSpec
gs = GridSpec(7, 12, figure=fig)

show_frame(L_flat, fig, gs, 0, 2, 0, 2, 'L-flat')
show_frame(R_flat, fig, gs, 0, 2, 3, 5, 'R-flat')
show_frame(G_flat, fig, gs, 0, 2, 6, 8, 'G-flat')
show_frame(B_flat, fig, gs, 0, 2, 9,11, 'B-flat')
show_frame(H_flat, fig, gs, 3, 6, 0, 3, 'H-flat')
show_frame(S_flat, fig, gs, 3, 6, 4, 7, 'S-flat')
show_frame(O_flat, fig, gs, 3, 6, 8,11, 'O-flat')

plt.show()
