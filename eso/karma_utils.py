#!/usr/bin/env python
#
# This file include the several help functions to help the arm allocation with karma
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
#
# Usage: 
# History:
    # 2025.04.11: first release, v0.1
 

import os
import re
import glob
from datetime import datetime
import numpy as np
from astropy.table import Table, vstack, hstack, unique
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy import units as u

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

# for cmd tool
import argparse
import textwrap
import inspect
import logging, warnings, subprocess 

__version__ = '0.1'

# predefined field center
field_centers={'COS4':[150.126, 2.26800],
               'GS4':[53.11875, -27.8048],
               'U4':[34.4100, -5.1925]}

def mask_image(filename, vmax=0.02, vmin=-0.005):
    with fits.open(filename) as hdu:
        hdu.info()
        data = hdu[0].data
        data[data > vmax] = vmax
        data[data < vmin] = vmin
        hdu[0].data = data
        hdu.writeto(filename[:-5]+'_updated.fits', overwrite=True)

def plot_image(fitsimage, cat, prioriy_groups=([1,2,3], [21,22,23], [31,32,33]), 
               band=None, vmax=0.02, vmin=-0.005, figname=None, ax=None, figsize=None,
               markersize=16, markeredgewidth=2, ):
    # define the priority color and markers
    # colors = ['tomato', 'royalblue', 'darkkhaki']
    colors = ['red', 'blue', 'grey']
    n_sub_groups = 3
    markers = ['o', '^', 's']
    if band is None:
        band = cat.split('_')[-1].split('.')[0]
    # plot the image
    with fits.open(fitsimage) as hdu:
        data = hdu[0].data
        header = hdu[0].header
        wcs_image = WCS(header)
    if figsize is None:
        figsize = np.array(data.shape)//1000
    if ax is None:
        newfig = True
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        newfig = False
    ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap='Greys')
    # select the targets
    if not isinstance(cat, (list, tuple)):
        tab = Table.read(cat, format='ascii')
        tab_list = [tab, ]
    else:
        tab_list = []
        for cat_file in cat:
            tab_list.append(Table.read(cat_file, format='ascii'))
    for fi in range(len(tab_list)):
        tab = tab_list[fi]
        marker = markers[fi]
        for target in tab:
            ra, dec = target['RA'], target['DEC']
            pixel_coord = skycoord_to_pixel(SkyCoord(ra=ra, dec=dec, unit='deg'), 
                                            wcs_image)
            for i in range(n_sub_groups):
                prio_sub = prioriy_groups[i]
                if target['prio'] in prio_sub:
                    if i == 1:
                        if target[f'do_{band}'] < 0.8:
                            print("reducing priority!")
                            color = colors[i+1]
                    color = colors[i]
                    break
            ax.plot(pixel_coord[0], pixel_coord[1], marker=marker, color=color, 
                    markersize=markersize, markeredgewidth=markeredgewidth, 
                    markerfacecolor='none', alpha=0.8)

    ax.axis('off')
    if newfig:
        if figname is not None:
            fig.savefig(figname, bbox_inches='tight', dpi=200)
            plt.close(fig)
        else:
            plt.show()

def plot_image_bands(fitsimage, cat, prioriy_groups=([1,2,3], [21,22,23], [31,32,33]), 
                     band=None, vmax=0.02, vmin=-0.005, figname=None, ax=None, 
                     nrow=None, ncol=None, figsize=(12,12)):
    n_cat = len(cat)
    if (nrow is None) or (ncol is None):
        nrow, ncol = 1, n_cat
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
    for i in range(len(cat)):
        cat_single = cat[i]
        plot_image(fitsimage, cat_single, ax=ax[i])
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        plt.close(fig)
    else:
        plt.show()
        
def set_priority(prio, prioriy_groups=([1,2,3], [21,22,23], [31,32,33]),
                 do_band=None):
    if prio in prioriy_groups[0]:
        if do_band == 0.5: # lower the priority of top-priority with do_band=0.5 
            return 2 
        else:
            return 1
    elif prio in prioriy_groups[1]:
        if do_band == 0.5: # lower the priority of top-priority with do_band=0.5 
            return 3 
        return 2
    elif prio in prioriy_groups[2]:
        return 3
    else:
        return -1

def generate_cat(master_cat, star_cat, allocations=[], 
                 prioriy_groups=([1,2,3], [21,22,23], [31,32,33]),
                 center_on_source=False,
                 field_name=None, band=None, pointing_number=None, 
                 outfile=None, overwrite=False):
    """generate cat for each pointing"""
    # read the master catalogs
    master_tab = Table.read(master_cat, format='ascii')
    star_tab = Table.read(star_cat, format='ascii', header_start=4)

    # read the allocations from the ins files
    allocated_targets = []
    for ins_file in allocations:
        print(f'including {ins_file}')
        allocated_targets.append(read_ins(ins_file))
    if len(allocated_targets) > 0:
        allocated_tab = vstack(allocated_targets)
        allocated_targets_this_band = allocated_tab[allocated_tab['band']==band]
        allocated_targets_other_band = allocated_tab[allocated_tab['band']!=band]
    else:
        allocated_targets_this_band = None
        allocated_targets_other_band = None

    # select targets
    targets_selected = []
    for t in master_tab:
        # check whether it is qualified
        do_band = t['do_'+band]
        if do_band < 0.4:
            continue
        t_id = t['IDv4']
        # check whether the target has been observed in the same band
        if allocated_targets_this_band is not None:
            if t_id in allocated_targets_this_band['name']:
                print(f'Target {t_id} has been allocated!')
                continue
        # then, read all the useful info
        t_ra = t['RA'] 
        t_dec = t['DEC']
        t_prio = t['prio']
        t_magnitude = 23.0
        t_type = 'O'
        t_comments = ''
        # checking its priority
        priority = set_priority(t_prio, prioriy_groups, do_band)
        if priority < 0:
            continue
        if priority > 1:
            # checking whether it has been observed with other bands
            if allocated_targets_other_band is not None:
                if t_id in allocated_targets_other_band['name']:
                    if do_band > 0.5:
                        priority = 1 # increase the priority
                        print(f"Increase the priority of {t_id} from {t_prio} to top priority")
                    else:
                        print(f"Target {t_id} is observed, but in band-{band} is only for [OII], keep the original priority")
        t_comments += str(t_prio) # keep the original priority as comments
        # appending as [ID, RA, DEC, Type, Magnitude, Band, Priority, Comments]
        targets_selected.append([t_id, t_ra, t_dec, t_type, t_magnitude, band, 
                                 priority, t_comments])
    if center_on_source:
        # calculate the center with most crowded targets
        ra_list = [t[1] for t in targets_selected]
        dec_list = [t[2] for t in targets_selected]
        ra_center = np.median(ra_list)
        dec_center = np.median(dec_list)
    else:
        ra_center, dec_center = field_centers[field_name]

    #writting to the output file
    today=datetime.today().strftime('%Y-%m-%d')
    header=f"""#------------------------------------------------------------------------------
# FIELD: {field_name}
# POINTING: {pointing_number}
# BAND: {band}
# Date created: {today}
#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
# ID  RA(J2000)  Dec(J2000)  Type  mag  Band  Pri  Comment
#------------------------------------------------------------------------------
{field_name}_p{pointing_number}  {ra_center}  {dec_center}  C  *  *  *\n"""
    if outfile is None:
        outfile = f'kprime_v00_edit_{field}_{band}_p{pointing_number}.cat'
    if os.path.isfile(outfile):
        if not overwrite:
            print(f'Pointing catalog: {outfile} already exist! Delete the old one to create a new one!')
            return
    with open(outfile, 'w+') as fp:
        fp.write(header)
        for tgs in targets_selected:
            fp.write("{}  {}  {}  {}  {}  {}  {}  {}\n".format(*tgs))
        fp.write('#\n')
        # copy the stars at the end
        for star in star_tab:
            if star['Type'] == 'G':
                star['Pri'] = 3
            else:
                star['Pri'] = 3
            fp.write("{}  {}  {}  {}  {}  {}  {}  {}\n".format(*star))
    print(f"Found the catalog: {outfile}")

def read_ins(filename):
    "read the output ins file from Karma"
    with open(filename, 'r') as fp:
        karma_results = fp.readlines()
    arms = list(range(1,24+1))
    # dead_arms = [2,8,10,13]
    dead_arms = [6, 17]
    arm_info_list = []
    arm_info_tab = Table(names=('name', 'karma_prio', 'band', 'comment','ra','dec','pins'), 
                         dtype=('S20', 'i4', 'S10', 'S20','S20', 'S20','i4'))
    for arm in arms:
        arm_info = [] # record: [name, karma_prioity, band, comments]
        arm_ocs_science = f'OCS.ARM{arm}.SCI'
        for line in karma_results:
            if 'TEL.TARG.ALPHA' in line:
                ra = line.split()[1]
                ra_string = "{}:{}:{}".format(ra[:2], ra[2:4], ra[4:])
            if 'TEL.TARG.DELTA' in line:
                dec = line.split()[1]
                dec_string = "{}:{}:{}".format(dec[:3], dec[3:5], dec[5:])
        if arm in dead_arms:
            # check whether it has been disabled
            if any([arm_ocs_science in line for line in karma_results]):
                print(f'It seems that you have used an arm that should have been disabled... arm={arm}')
            continue
        for line in karma_results:
            if arm_ocs_science in line:
                # this arm has been used
                if 'NAME' in line:
                    arm_target = line.split()[1].strip('"')
                if 'PRIOR' in line:
                    arm_prior = line.split()[1]
                if 'BAND' in line:
                    arm_band = line.split()[1].strip('"')
                if 'COMMENT' in line:
                    arm_comment = line.split()[1].strip('"')
        arm_info_tab.add_row([arm_target, arm_prior, arm_band, arm_comment, ra_string, dec_string, -1])
    return arm_info_tab 

def collect_ins(filenames, master_cat=None, exclude_possible=True, debug=False):
    # read the allocations from the ins files
    # sort the filenames
    # filenames = np.sort(filenames)
    if isinstance(filenames, str):
        filenames = [filenames,]
    allocated_targets = []
    for ins_file in filenames:
        if exclude_possible:
            if 'possible' in ins_file:
                continue
        if debug:
            print(f'including {ins_file}')
        allocated_targets.append(read_ins(ins_file))
    if len(allocated_targets) > 0:
        allocated_tab = vstack(allocated_targets)
        # remove the stars
        stars_select = [True if 'str' in name else False for name in allocated_tab['name']]
        allocated_tab = allocated_tab[~np.array(stars_select)]
        if master_cat is not None:
            master_tab = Table.read(master_cat, format='ascii')
            # match all the other info based on the target name
            select_idx = []
            for name in allocated_tab['name']:
                where_res = np.where(master_tab['IDv4']==name)[0]
                if len(where_res) > 0:
                    select_idx.append(where_res[0])
            # print(select_idx)
            selected_master_tab = master_tab[select_idx]
            allocated_tab = hstack([selected_master_tab, allocated_tab])
    else:
        allocated_tab = []
    return allocated_tab

def merger_targets(allocated_tab):
    """merge the target observed with multiple bands
    """
    names_unique = allocated_tab['IDv4']

def statistics_ins(planfile, return_unique=True):
    plan = Table.read(planfile, format='ascii')
    fields = np.unique(plan['field'])
    bands = np.unique(plan['band'])
    plan_all = collect_ins(plan['insfile'])
    plan_select = collect_ins(plan['insfile'][plan['select']==1])
    all_tab_list = []
    select_tab_list = []
    for field in fields:
        for band in bands:
            field_band_plan = plan[(plan['field']==field) & (plan['band']==band)]
            master_cat = f'v01_selection/kprime_v01_sel_{field}_{band}.dat'
            field_band_tab_all = collect_ins(field_band_plan['insfile'], master_cat=master_cat)
            field_band_selection = field_band_plan['select'] == 1
            # print(field_band_selection)
            field_band_tab_selected = collect_ins(field_band_plan['insfile'][field_band_selection],
                                                  master_cat=master_cat)
            all_tab_list.append(field_band_tab_all)
            select_tab_list.append(field_band_tab_selected)
    # merger the table of different bands
    all_tab = vstack(all_tab_list)
    select_tab = vstack(select_tab_list)
    if return_unique:
        all_tab = unique(all_tab, keys='name')
        select_tab = unique(select_tab, keys='name')
    return all_tab, select_tab

def plot_statistics_sample(planfile,figname=None):
    all_tab, select_tab = statistics_ins(planfile)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    # first show the all the select source
    ax[0,1].set_title('LoI targets')
    ax[0,0].scatter(all_tab['lM'], all_tab['dMS'], s=10, c=all_tab['z'], marker='o', 
                    cmap='turbo')
    ax[0,1].scatter(all_tab['lM'], all_tab['dMRe'], s=10, c=all_tab['z'], marker='o', 
                    cmap='turbo')
    ax[0,1].set_ylim(-1.5, 1.5)
    ax[0,2].scatter(all_tab['z'], all_tab['lM'], s=10, c=all_tab['z'], marker='o', 
                    cmap='turbo')

    ax[1,1].set_title('Final targets')
    ax[1,0].scatter(select_tab['lM'], select_tab['dMS'], s=10, c=select_tab['z'], 
                    marker='o', cmap='turbo')
    ax[1,1].scatter(select_tab['lM'], select_tab['dMRe'], s=10, c=select_tab['z'], 
                    marker='o', cmap='turbo')
    ax[1,1].set_ylim(-1.5, 1.5)
    ax[1,2].scatter(select_tab['z'], select_tab['lM'], s=10, c=select_tab['z'], 
                    marker='o', cmap='turbo')
    # ax[0,0].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    ax[0,0].set_ylabel(r'$\Delta$MS')
    # ax[0,1].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    ax[0,1].set_ylabel(r'$\Delta$MRe')
    # ax[0,2].set_xlabel('z')
    ax[0,2].set_ylabel(r'log($\text{M}_*/\text{M}_\odot$)')
    ax[1,0].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    ax[1,0].set_ylabel(r'$\Delta$MS')
    ax[1,1].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    ax[1,1].set_ylabel(r'$\Delta$MRe')
    ax[1,2].set_xlabel('z')
    ax[1,2].set_ylabel(r'log($\text{M}_*/\text{M}_\odot$)')

    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
    else:
        plt.show()

def plot_statistics_sample2(planfile,figname=None, bins=10, fontsize=12):
    """add x y histogram compared with plot_statistics_sample
    """
    all_tab, select_tab = statistics_ins(planfile)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].scatter(select_tab['lM'], select_tab['dMS'], s=10, c=select_tab['z'], 
                  marker='o', cmap='turbo')
    # add the x and y axis histogram
    # first show the all the select source
    ax1_histx = ax[0].inset_axes([0,0,1,0.2], sharex=ax[0])
    ax1_histx.axis('off')
    ax1_histy = ax[0].inset_axes([0.8,0,0.2,1], sharey=ax[0])
    ax1_histy.axis('off')
    ax1_histx.hist(all_tab['lM'], histtype='step', color='C0', 
                     alpha=0.8, bins=bins, label='LoI targets')
    ax1_histx.hist(select_tab['lM'], histtype='step', color='C1', 
                     alpha=0.8, bins=bins, label='Final targets')
    ax1_histy.invert_xaxis()
    ax1_histy.hist(all_tab['dMS'], histtype='step', color='C0', 
                     alpha=0.8, bins=bins, orientation='horizontal')
    ax1_histy.hist(select_tab['dMS'], histtype='step', color='C1', 
                     alpha=0.8, bins=bins, orientation='horizontal')

    # second
    ax[1].scatter(select_tab['lM'], select_tab['dMRe'], s=10, c=select_tab['z'], 
                  marker='o', cmap='turbo')
    ax[1].set_ylim(-1.5, 1.5)
    # add the x and y axis histogram
    ax2_histx = ax[1].inset_axes([0,0,1,0.2], sharex=ax[1])
    ax2_histx.axis('off')
    ax2_histy = ax[1].inset_axes([0.8,0,0.2,1], sharey=ax[1])
    ax2_histy.axis('off')
    ax2_histx.hist(all_tab['lM'], histtype='step', color='C0', 
                     alpha=0.8, bins=bins, label='LoI targets')
    ax2_histx.hist(select_tab['lM'], histtype='step', color='C1', 
                     alpha=0.8, bins=bins, label='Final targets')
    ax2_histy.invert_xaxis()
    ax2_histy.hist(all_tab['dMRe'], histtype='step', color='C0', 
                   alpha=0.8, bins=bins, orientation='horizontal',
                   range=[-1.5,1.5])
    ax2_histy.hist(select_tab['dMRe'], histtype='step', color='C1', 
                   alpha=0.8, bins=bins, orientation='horizontal',
                   range=[-1.5,1.5])
    # third
    ax[2].scatter(select_tab['z'], select_tab['lM'], s=10, c=select_tab['z'], 
                  marker='o', cmap='turbo')
    
    # set label
    ax[0].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)', fontsize=fontsize)
    ax[1].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)', fontsize=fontsize)
    ax[2].set_xlabel('z', fontsize=fontsize)
    ax[0].set_ylabel(r'$\Delta$MS', fontsize=fontsize)
    ax[1].set_ylabel(r'$\Delta$MRe', fontsize=fontsize)
    ax[2].set_ylabel(r'log($\text{M}_*/\text{M}_\odot$)', fontsize=fontsize)
    
    # set legend
    legend_elements = [
            Line2D([0], [0], label='LoI targets', color='C0'),
            Line2D([0], [0], label='Final targets', color='C1'),
            ]
    ax[1].legend(handles=legend_elements, fontsize=fontsize)

    fig.subplots_adjust(wspace=0.3)
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
    else:
        plt.show()

def plot_statistics_sample3(planfile,figname=None, bins=10):
    """show a single scatter with x and y histogram
    """
    all_tab, select_tab = statistics_ins(planfile)
    fig, ax = plt.subplot_mosaic([['histx', '.'],
                                   ['scatter', 'histy']],
                                  figsize=(6, 6),
                                  width_ratios=(4, 1), height_ratios=(1, 4),
                                  layout='constrained')
    ax['scatter'].scatter(select_tab['lM'], select_tab['dMS'], s=10, c=select_tab['z'], 
                    marker='o', cmap='turbo')
    ax['scatter'].scatter(select_tab['lM'], select_tab['dMRe'], s=10, c=select_tab['z'], 
                    marker='o', cmap='turbo')
    ax['scatter'].set_ylim(-1.5, 1.5)
     
    ax['scatter'].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    ax['scatter'].set_ylabel(r'$\Delta$MS')
    # now determine nice limits by hand:
    ax['histx'].hist(all_tab['lM'], histtype='step', color='C0', 
                     alpha=0.8, bins=bins, label='LoI targets')
    ax['histx'].hist(select_tab['lM'], histtype='step', color='C1', 
                     alpha=0.8, bins=bins,label='Final targets')
    ax['histx'].legend()
    ax['histy'].hist(all_tab['dMS'], histtype='step', color='C0', 
                     alpha=0.8, bins=bins, orientation='horizontal',
                     range=[-1.5,1.5])
    ax['histy'].hist(select_tab['dMS'], histtype='step', color='C1', 
                     alpha=0.8, bins=bins, orientation='horizontal',
                     range=[-1.5,1.5])
   
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
    else:
        plt.show()


def plot_statistics_redshift(planfile,figname=None):
    all_tab, select_tab = statistics_ins(planfile)
    # plot the fraction 
    z_bins_reps = ["0.8", "1.5", "2.4"]
    z_bins = [(0.5,1.2), (1.2, 1.8), (1.8,2.7)]
    z_all_tab = all_tab['z']
    z_select_tab = select_tab['z']
    z_bins_all = []
    z_bins_select = []
    for z_bin in z_bins:
        z_bins_all.append(np.sum((all_tab['z']>z_bin[0]) & (all_tab['z']<z_bin[1])))
        z_bins_select.append(np.sum((select_tab['z']>z_bin[0]) & (select_tab['z']<z_bin[1])))
    fig, ax = plt.subplots()
    ax.bar(z_bins_reps, z_bins_all, width=0.4, alpha=0.5,
           label=f'LoI targets ({np.sum(z_bins_all)}+17)',)
    ax.bar(z_bins_reps, z_bins_select, width=0.4, alpha=0.5,
           label=f'Final taregts ({np.sum(z_bins_select)})')
    # add the number of galaxies at each bar 
    for i in range(3):
        ax.text(z_bins_reps[i], z_bins_all[i]-2, z_bins_all[i], 
                ha='center', va='top')
        ax.text(z_bins_reps[i], z_bins_select[i]-2, z_bins_select[i],
                ha='center', va='top')
 
    # ax.text(.8,.9, 'Unique target', horizontalalignment='center', transform=ax.transAxes, fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel(r'z', fontsize=14)
    ax.set_ylabel('# of galaxies', fontsize=14)
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
    else:
        plt.show()

def plot_environments_Chartab(planfile, hist=False, figname=None):
    all_tab, select_tab = statistics_ins(planfile)
    # define the coordinates
    select_cat = SkyCoord(ra=select_tab['RA'], dec=select_tab['DEC'], unit='deg')

    # read the environmental catalogue
    env_tab = Table.read('./apjab61fdt2_mrt.txt', format='ascii')
    env_cat = SkyCoord(ra=env_tab['RAdeg'], dec=env_tab['DEdeg'], unit='deg')
    
    # matching the catalogue
    idx, d2d, d3d = select_cat.match_to_catalog_sky(env_cat)

    # select only the valid matches, d2d<0.1arcsec
    d2_tab = Table([d2d.to(u.arcsec).value,], names=['sky_distance',])
    env_matched_tab = hstack([select_tab, env_tab[idx], d2_tab])
    
    valid_match = d2d < 0.1*u.arcsec
    if not np.any(valid_match):
        print("removing {np.sum(~valid_match)} targets without proper match!")
    valid_tab = env_matched_tab[valid_match]

    # fig, ax = plt.subplots(1,2, figsize=(12, 4))
    # ax[0].scatter(valid_tab['lM'], valid_tab['np'], s=10, c=valid_tab['z_1'], 
                    # marker='o', cmap='turbo')
    # ax[1].scatter(valid_tab['lM'], valid_tab['delta'], s=10, c=valid_tab['z_1'], 
                    # marker='o', cmap='turbo')

    # ax[0].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    # ax[1].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
    # ax[0].set_ylabel(r'Physical density (Mpc/h)$^{-3}$')
    # ax[1].set_ylabel(r'Density contract')
    if hist:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        # ax[0].hist(valid_tab['np'], bins=10, density=1)
        # ax[0].set_xlabel(r'h/Mpc$^3$')
        ax.hist(valid_tab['delta'], bins=15, color='steelblue')
        delta_mean = 0.45
        delta_sigma = 0.75
        ax.axvline(x=0.45, ymin=0, ymax=200, color='k', alpha=0.8)
        ax.axvline(delta_mean-delta_sigma, ymin=0, ymax=1000, linestyle='--', color='k', 
                 alpha=0.8,) 
        ax.axvline(delta_mean+delta_sigma, ymin=0, ymax=1000, linestyle='--', color='k', 
                 alpha=0.8,)
        ax.set_xlabel(r'Density contrast $\delta$')
        ax.set_ylim(0, 200)
    else:
        scolor = valid_tab['delta']
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].scatter(valid_tab['lM'], valid_tab['dMS'], s=10, c=scolor, marker='o', 
                        cmap='turbo')
        ax[1].scatter(valid_tab['lM'], valid_tab['dMRe'], s=10, c=scolor, marker='o', 
                        cmap='turbo')
        ax[1].set_ylim(-1.5, 1.5)
        im=ax[2].scatter(valid_tab['z_1'], valid_tab['lM'], s=10, c=scolor, marker='o', 
                        cmap='turbo')
        cbar = plt.colorbar(im, ax=ax[2])
        cbar.set_label(r'Overdensity $\delta$')
        ax[0].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
        ax[0].set_ylabel(r'$\Delta$MS')
        ax[1].set_xlabel(r'log($\text{M}_*/\text{M}_\odot$)')
        ax[1].set_ylabel(r'$\Delta$MRe')
        ax[2].set_xlabel('z')
        ax[2].set_ylabel(r'log($\text{M}_*/\text{M}_\odot$)')

    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
    else:
        plt.show()
    # return env_matched_tab




def print_statistics(field_observed, ngroup=None):
    named_fields = ['k3d', 'sins', 'kros', 'kges', 'kurv', 'klev', 'kash', 'muse', 'mosd', 'vd15']
    len_observed = len(field_observed)
    if ngroup is not None:
        for i in range(0, len(field_observed), ngroup):
            print(f'>>Pointing {i//17}:')
            print('field\ttotal\ttier1\ttier2\ttier3')
            if i+17<len_observed:
                pointing_observed = field_observed[i:i+17]
            else:
                pointing_observed = field_observed[i:-1]
            for field in named_fields:
                print("{}\t{}\t{}\t{}\t{}".format(field, np.sum(pointing_observed[field]==1), 
                        np.sum((pointing_observed[field]==1) & (pointing_observed['prio'] < 4)), 
                        np.sum((pointing_observed[field]==1) & (pointing_observed['prio'] < 30) & (pointing_observed['prio'] > 20)), 
                        np.sum((pointing_observed[field]==1) & (pointing_observed['prio'] > 30))))
            print("tier1_total: {}; tier2_total: {}; tier3_total: {}".format(
                  np.sum(pointing_observed['prio']<4),
                  np.sum((pointing_observed['prio']>20)&(pointing_observed['prio']<30)),                   
                  np.sum(pointing_observed['prio']>30)))
    print('Summary for all pointings:')
    print('field\ttotal\ttier1\ttier2\ttier3')
    for field in named_fields:
        print("{}\t{}\t{}\t{}\t{}".format(field, np.sum(field_observed[field]==1), 
                np.sum((field_observed[field]==1) & (field_observed['prio'] < 4)), 
                np.sum((field_observed[field]==1) & (field_observed['prio'] < 30) & (field_observed['prio'] > 20)), 
                np.sum((field_observed[field]==1) & (field_observed['prio'] > 30))))
    print("tier1_total: {}; tier2_total: {}; tier3_total: {}".format(
          np.sum(field_observed['prio']<4),
          np.sum((field_observed['prio']>20)&(field_observed['prio']<30)),                   
          np.sum(field_observed['prio']>30)))

def assign_pointings(field_observed, ngroup=17):
    len_observed = len(field_observed)
    for i in range(0, len(field_observed), ngroup):
        print(f'>>Pointing {i//17}:')
        if i+17<len_observed:
            pointing_observed = field_observed[i:i+17]
        else:
            pointing_observed = field_observed[i:]
        pointing_observed['pointing'] = i//17
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='karma_utils.py',
            description=f"Welcome to jhchen's karma utilities {__version__}",
            epilog='Reports bugs and problems to cjhastro@gmail.com')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--logfile', default=None, help='the filename of the log file')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')

    # add subparsers
    subparsers = parser.add_subparsers(title='Available task', dest='task', 
                                       metavar=textwrap.dedent(
        '''
            * gencat

          To get more details about each task:
          $ karma_utils.py task_name --help
        '''))
 
    ################################################
    # gencat
    subp_gencat = subparsers.add_parser('gencat',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            generate the listobs as a text files
            --------------------------------------------
            Examples:

              python karma_utils.py gencat --catalogue data/targets_v03/kprime_v03_sel_GS4_H.dat --stars data/stars/GS_stars.cat --allocations kprime_work_v03 --field GS4 --band H --pointing 0 --outfile kprime_work_v03/kprime_v03_karma_H_p0.cat

            '''))
    subp_gencat.add_argument('-c', '--catalogue', help='catalogue file')
    subp_gencat.add_argument('-s', '--stars', help='star catalogue file')
    subp_gencat.add_argument('-a', '--allocations', help='directory including all the existing allocations')
    subp_gencat.add_argument('-f', '--field', help='name of the field')
    subp_gencat.add_argument('-b', '--band', help='observing band')
    subp_gencat.add_argument('-o', '--outfile', help='output filename')
    subp_gencat.add_argument('--pointing_number', default='*', help='the number of pointing, just for naming')
    subp_gencat.add_argument('--prioriy_groups', help='the prioriy groups')

    args = parser.parse_args()
    logging.basicConfig(filename=args.logfile, encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Welcome to alma_utils.py {__version__}")
 
    if args.debug:
        logging.debug(args)
        func_args = list(inspect.signature(locals()[args.task]).parameters.keys())
        func_str = f"Executing:\n \t{args.task}("
        for ag in func_args:
            try: func_str += f"{ag}={args.__dict__[ag]},"
            except: func_str += f"{ag}=None, "
        func_str += ')\n'
        logging.debug(func_str)

    if args.task == 'gencat':
        targets_allocations = glob.glob(os.path.join(args.allocations,'*.ins'))
        generate_cat(args.catalogue, args.stars, 
                     prioriy_groups=([1,2,3], [21,22,23], [31,32,33]),
                     allocations=targets_allocations, 
                     field_name=args.field, band=args.band, 
                     pointing_number=args.pointing_number, 
                     outfile=args.outfile)
 
    if False:
        # generate the working catalogue
        working_dir = 'kprime_work_v03'
        targets_allocations = glob.glob(working_dir+'/*.ins')
        step = 1
        # first step, for H band
        if step == 1:
            # generate_cat('targets_v01/kprime_v01_sel_GS4_H.dat', 
                         # 'stars/GS_stars.cat', 
                         # allocations=targets_allocations, 
                         # field_name='GS4', band='H', pointing_number=1, 
                         # outfile='kprime_work/kprime_v01_karma_H_p0.cat')
            # generate_cat('targets_v01/kprime_v01_sel_GS4_H.dat', 
                         # 'stars/GS_stars.cat', 
                         # prioriy_groups=([1,2,3], [21,22,23], [31]),
                         # allocations=targets_allocations, 
                         # field_name='GS4', band='H', pointing_number=2, 
                         # outfile='kprime_work/kprime_v01_karma_H_p2.cat')
            generate_cat('data/targets_v03/kprime_v03_sel_GS4_H.dat', 
                         'stars/GS_stars.cat', 
                         prioriy_groups=([1,2,3], [21,22,23], [31,32,33]),
                         allocations=targets_allocations, 
                         field_name='GS4', band='H', pointing_number=3, 
                         outfile=working_dir+'/kprime_karma_H_p3.cat')
        
        # second step, for YJ band
        if step == 2:
            # targets_allocations = glob.glob('kprime_work/kprime_v01_karma_H*.ins')
            generate_cat('targets_v01/kprime_v01_sel_GS4_YJ.dat', 
                         'stars/GS_stars.cat', 
                         allocations=targets_allocations, 
                         field_name='GS4', band='YJ', pointing_number=1, 
                         outfile='kprime_work/kprime_v01_karma_YJ_p0.cat')

        # third step, for IZ band
        if step == 3:
            generate_cat('targets_v01/kprime_v01_sel_GS4_IZ.dat', 
                         'stars/GS_stars.cat', 
                         allocations=targets_allocations, 
                         field_name='GS4', band='IZ', pointing_number=1, 
                         outfile='kprime_work/kprime_v01_karma_IZ_p0.cat')

        # fourth step, for K band
        if step == 4:
            generate_cat('targets_v01/kprime_v01_sel_GS4_K.dat', 
                         'stars/GS_stars.cat', 
                         allocations=targets_allocations, 
                         field_name='GS4', band='K', pointing_number=1, 
                         outfile='kprime_work/kprime_v01_karma_K_p0.cat')


    if False:
        # generate table for general usage
        master_tab = Table.read('./targets_v01/kprime_v01_sel_all.dat', format='ascii')
        master_tab_GS4 = master_tab[master_tab['field'] == 'GS4']
        new_tab = Table(np.zeros((len(master_tab_GS4), 4)),
                        names=('sel_K', 'sel_H', 'sel_YJ', 'sel_IZ'), 
                        dtype=('f4', 'f4', 'f4', 'f4'))
        master_names = master_tab_GS4['IDv4'].tolist()
        observed_names = all_observed['IDv4'].tolist()
        for i in range(len(master_tab_GS4)):
            target_i = master_tab_GS4[i]
            for j in range(len(all_observed)):
                target_obs = all_observed[j]
                if master_names[i] == observed_names[j]:
                    if target_i['do_'+target_obs['band']] > 0:
                        # print(master_names[i], observed_names[j])
                        new_tab['sel_'+target_obs['band']][i] = 1
        # for allo in all_observed:
            # print(allo['IDv4'])
            # target_idx = np.where(str(master_tab['IDv4']) == str(allo['IDv4']))[0]
            # print(target_idx)
            # new_tab['sel_'+allo['band']][target_idx[0]] = 1
        new_master_tab = hstack([master_tab_GS4, new_tab])

