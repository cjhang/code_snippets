#!/usr/bin/env python
#
# This file include the several help functions to help the data analysis of ALMA data
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
#
# Usage: due the requirement for all the casa internal tasks/tools this file should 
#        be execfile interactively
#       
#        # as a python package
#        execfile(path_to_this_file)
#        recover_pipeline(root_folder)
#
#        # as cmd tools
#        $ alma_utils.py -h
#
# History:
    # 2019.09.27: first release, v0.1
    # 2022.08.07: add pipeline tools
    # 2024.02.01: add cmd interface

# import the packages shipped with casa
# for basic buildin packages
import os, io, sys, glob, re, datetime
# for logging and system calls
import logging, warnings, subprocess 
import xml.etree.ElementTree as ElementTree
import argparse
import textwrap
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

__version__ = '0.1.3'

const_c = 2.99792458e5 # km/s

############ search for installed packages
try: import astropy; has_astropy=True
except: has_astropy=False

try: import astroquery; has_astroquery=True
except: has_astroquery=False


if has_astropy:
    from astropy.units import Quantity
    from astropy.coordinates import SkyCoord
    import astropy.units as u

###########################################
# Data retrieval
####################
# It connect with ALMA tap service
import multiprocessing
from functools import partial

def rename_uid(uid, reverse=False):
    """rename the name format uid names

    renmae '://' to '___' and '/' to '_', or do the opposite is reverse=False
    """
    if reverse:
        uid1 = uid.replace('___', '://')
        uid2 = uid1.replace('_', '/')
    else:
        uid1 = uid.replace('://', '___')
        uid2 = uid1.replace('/', '_')
    return uid2

def unpack_almadata(mous_id, savedir='./', unpack=True,):
    """download and unpack the raw data
    """
    if not has_astroquery:
        print("Please install astroquery to use this function")
        return
    uid_savedir = os.path.join(savedir, rename_uid(mous_id))
    os.system('mkdir -p {}'.format(uid_savedir))
    datalink = pyvo.dal.adhoc.DatalinkResults.from_result_url(f"https://almascience.eso.org/datalink/sync?ID={mous_id}")
    access_datalinks = datalink['access_url'][datalink['readable']].tolist()
    myAlma = Alma() # for download only
    myAlma.download_files(access_datalinks, savedir=uid_savedir)
    if unpack:
        for ftar in glob.glob(os.path.join(uid_savedir, '*.tar')):
            os.system('tar -xf {} -C {}'.format(ftar, uid_savedir))
        os.system('rm -rf {}'.format(os.path.join(uid_savedir, '*.tar')))
    
def query_ALMACAL(objname, band_list, savedir='./data', download_files=False, nprocess=1):
    """query for ALMA calibrator observations from ALMA archive
    
    Args:
        objname (str): alma name for the calibrator, like: J0001-0001
        band_list (str): alma bands, like: '5,6,7'
        
    """
    if not has_astroquery:
        print("Please install astroquery to use this function")
        return
    if calibration_observation:
        pass
    query_string = "select * from ivoa.obscore WHERE target_name='{}' AND science_observation='F' AND band_list in ({})".format(objname, band_list)
    alma_q = Alma.query_tap(query_string)
    member_uids = np.unique(alma_q['member_ous_uid'])
    #print(member_uids)
    
    if download_files:
        _get_data = partial(get_almadata, savedir=savedir)
        with multiprocessing.Pool(nprocess) as p:
            p.map(_get_data, member_uids)
        # the old way
        #for mous_id in alma_q['member_ous_uid']:
        #    savedir = os.path.join(outdir, mous_id)
        #    get_almadata(mous_id, savedir=savedir)
    return member_uids

def query_ALMA(ra, dec, objname=None, radius=10, band=None):
    """a wrapper of alma query_region

    Args:
        ra
        dec
        objname
        radius: (float or Quantity), default units is arcsec
    """
    if not has_astropy:
        print("Please install astropy to use this function")
        return
    if not has_astroquery:
        print("Please install astroquery to use this function")
        return
    radius = radius*u.arcsec
    if isinstance(ra, Quantity):
        sk = SkyCoord(ra=ra, dec=dec)
    elif isinstance(ra, str):
        sk = SkyCoord(f'{ra} {dec}', unit=(u.hourangle, u.deg))
    else:
        sk = SkyCoord(ra=ra, dec=dec, unit=u.deg)
        #raise ValueError('Unsupported coordinate')
    if not isinstance(radius, Quantity):
        radius = radius * u.arcsec

    res = alma.query_region(sk, radius=radius)

def read_query_frequencies(fstring):
    #re.compile()
    spw_list_dict = {'spw_freq':[], 'spw_width':[], 'spw_sensitivity':[],
                     'spw_sensitivity_native':[], 'spw_pols':[]}
    spw_list = fstring.split(' U ')
    for spw_string in spw_list:
        spw_items = spw_string[1:-1].split(',')
        item_freq_range = np.array(spw_items[0][:-3].split('..')).astype(float)
        item_width = float(spw_items[1][:-3])/np.mean(item_freq_range)*const_c*1e-6
        item_sensitivity_10kms = spw_items[2]
        item_sensitivity_native = spw_items[3]
        item_pols = spw_items[4]
        spw_list_dict['spw_freq'].append(item_freq_range.tolist())
        spw_list_dict['spw_width'].append(item_width)
        spw_list_dict['spw_sensitivity'].append(item_sensitivity_10kms)
        spw_list_dict['spw_sensitivity_native'].append(item_sensitivity_native)
    return spw_list_dict

def plot_ALMA_query(tabdata, bands=['B3','B4','B5','B6','B7','B8'], z=0,
                    lines=[345.80,461.04,576.27], lines_names=['CO32','CO43','CO54'], savefile=None):
    """plot the alma from the query table
    """
    fig = plt.figure(figsize=(3*len(bands),5))
    plt.rcParams['hatch.linewidth'] = 0.5
    ax = fig.add_subplot(111)

    #ALMA band information, in GHz
    band_list = {'B3':[84, 116], 'B4':[125, 163], 'B5':[163, 211],
            'B6':[211, 275], 'B7':[275, 373], 'B8':[385, 500], \
            'B9':[602, 720], 'B10':[787, 950]}
    band_min = 1000
    band_max = 30
    for band in bands:
        if band_min > band_list[band][0]:
            band_min = np.min(band_list[band])
        if band_max < band_list[band][1]:
            band_max = np.max(band_list[band])
        ax.broken_barh([(band_list[band][0], np.diff(band_list[band])[0]),],
                       (0, 1), facecolor='lightblue', edgecolor='grey', \
                       linewidth=1, alpha=0.5)
        ax.text(np.mean(band_list[band]), 1.1, "Band"+band[1:],
                horizontalalignment='center', verticalalignment='center')
    ax.set_xlim(band_min-10, band_max+10)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(r'$t_{\rm on\,source}$ fraction')
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='x', which='minor', labelsize=6)
    ax.tick_params(axis='y', labelcolor='w', top='off', bottom='on', left='off', right='off', labelsize=2)

    # color --> resolution, hatch --> velocity resolution
    legend_elements = [Patch(facecolor='tomato', edgecolor='none', label='<1 arcsec', alpha=0.8),
                       Patch(facecolor='chocolate', edgecolor='none', label='0.6-1.0 arcsec', alpha=0.8),
                       Patch(facecolor='orange', edgecolor='none', label='0.4-0.6 arcsec', alpha=0.8),
                       Patch(facecolor='y', edgecolor='none', label='0.2-0.4 arcsec', alpha=0.8),
                       Patch(facecolor='limegreen', edgecolor='none', label='<0.2 arcsec', alpha=0.8),
                       # velocity resolution
                       Patch(facecolor='none', edgecolor='grey', label='>100km/s', alpha=0.8, hatch='...'),
                       Patch(facecolor='none', edgecolor='grey', label='40-100km/s', alpha=0.8, hatch='///'),
                       Patch(facecolor='none', edgecolor='grey', label='<40km/s', alpha=0.8, hatch='+++'),
                       ]
    # Create the figure
    ax.legend(handles=legend_elements, loc='lower left', ncols=8)

    # plot the spectral window
    band_total_time = dict(zip(bands, np.zeros(len(bands))))
    band_height = dict(zip(bands, np.zeros(len(bands))))
    print(band_total_time)
    # get the total time in each bands
    for obs in tabdata:
        obs_band = 'B'+obs['band_list']
        if obs_band not in bands:
            continue
        exptime = obs['t_exptime']
        #print(obs_band, 'exptime', exptime, type(exptime))
        band_total_time[obs_band] += exptime
    for band in bands:
        ax.text(np.mean(band_list[band]), 1.05, "t_total: {:.2f}h".format(band_total_time[band]/3600.),
                horizontalalignment='center', verticalalignment='center')
    print(band_total_time)
    # now plot each obs
    for obs in tabdata:
        obs_band = 'B'+obs['band_list']
        if obs_band not in bands:
            continue
        exptime = obs['t_exptime']
        obs_res = obs['s_resolution']
        if obs_res>1.0: bar_color='tomato'
        elif obs_res>0.6: bar_color='chocolate'
        elif obs_res>0.4: bar_color='orange'
        elif obs_res>=0.2: bar_color='y'
        elif obs_res<0.2: bar_color='limegreen'
        spw_list = read_query_frequencies(obs['frequency_support'])
        # calculate the height from the exptime
        h = band_height[obs_band]
        dh = exptime/band_total_time[obs_band]
        n_spw = len(spw_list['spw_freq'])
        for i in range(n_spw):
            freq_range = spw_list['spw_freq'][i]
            width = spw_list['spw_width'][i]
            if width > 100.0: hatch_style='...'
            elif width > 40: hatch_style='//'
            elif width < 40: hatch_style='+++'
            ax.broken_barh([(freq_range[0], np.diff(freq_range)[0]),],
                            (h, dh), facecolors=bar_color, edgecolors='none', \
                            alpha=0.8, hatch=hatch_style)
            #ax.hlines(y=h, xmin=band_list[band][0], xmax=band_list[band][1],
            #        color='r', linestyle='-', alpha=0.1, linewidth=1/n_obs)
        band_height[obs_band] += dh
    # plot the spectral lines
    if lines is not None:
        if isinstance(lines, (float,int)):
            lines = [lines,]
        if lines_names:
            if isinstance(lines_names, (str)):
                lines = [lines_names,]
        for idx, line in enumerate(lines):
            line_obs = line / (1.0 + z)
            ax.vlines(line_obs, 0, 1, alpha=0.6, ls='--',color='k', linewidth=0.5)
            if lines_names:
                ax.text(line_obs, -0.05, lines_names[idx], fontsize=8, alpha=0.6, horizontalalignment='center')
    
    if savefile:
        fig.savefig(savefile, bbox_inches='tight')
        plt.close(fig)

def info(files, casa=None):
    """generate the listobs file for each uid
    """
    if casa is None:
        casa = 'casa'
    if isinstance(files, (list, tuple)):
        filelist = files
    else:
        filelist = glob.glob(files)
    exec_string = ""
    for file in filelist:
        exec_string+= f"listobs('{file}', listfile='{file}.listobs.txt')\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        exec_file = os.path.join(tmpdir, '_tmp.py')
        with open(exec_file, 'w') as fp:
            fp.write(f'{exec_string}')
        os.system(f"casa --nologger -c {exec_file}") 


###########################################
# Table tools
####################

try:
    from casatools import table, msmetadata
    from casatasks import listobs, rmtables, tclean, exportfits
    from casatools import synthesisutils
    has_casa = True
except:
    has_casa = False
    print("Does not find casa libraries")

def _read_spw(vis):
    """read the spectral windows
    """
    if isinstance(vis, str):
        vis = [vis, ]

    if not isinstance(vis, list):
        raise ValueError("read_spw: Unsupported measurements files!")

    tb = table()
    spw_specrange = {}

    for v in vis:
        tb.open(v + '/SPECTRAL_WINDOW')
        col_names = tb.getvarcol('NAME')
        col_freq = tb.getvarcol('CHAN_FREQ')
        tb.close()

    return list(spw_specrange.values())

def read_spw(vis):
    """read the spectral windows

    Return: list of frequency range in GHz
    """
    if isinstance(vis, str):
        vis = [vis, ]

    if not isinstance(vis, list):
        raise ValueError("read_spw: Unsupported measurements files!")

    tb = table()
    spw_specrange = {}

    for v in vis:
        tb.open(v + '/SPECTRAL_WINDOW')
        col_names = tb.getvarcol('NAME')
        col_freq = tb.getvarcol('CHAN_FREQ')
        tb.close()

        for key in col_names.keys():
            freq_max = np.max(col_freq[key]) / 1e9
            freq_min = np.min(col_freq[key]) / 1e9
            spw_specrange[key] = [freq_min, freq_max]

    return list(spw_specrange.values())

def plot_spw(spws, show_alma_bands=False):
    """this function visualize the spw windows of multiple spws and observations
    """
    spws = np.array(spws)
    if spws.ndim == 2:
        spws = [spws,]
    nspw = len(spws)
    spw_height = 1./nspw
    fig = plt.figure(figsize=(8,5))
    # fig.suptitle(os.path.basename(objfolder))
    ax = fig.add_subplot(111)

    #ALMA band information, in GHz
    if show_alma_bands:
        band_list = {'B3':[84, 116], 'B4':[125, 163], 'B5':[163, 211], 
                     'B6':[211, 275], 'B7':[275, 373], 'B8':[385, 500], \
                     'B9':[602, 720], 'B10':[787, 950]}
    for i,spw in enumerate(spws):
        for freq in spw:
            ax.broken_barh(([freq[0], np.diff(freq)[0]],), 
                          np.array([0, 1])*spw_height+spw_height*i, 
                           facecolor='lightblue', edgecolor='grey', \
                          linewidth=1, alpha=0.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelcolor='w', top='off', bottom='on', left='off', right='off', labelsize=2)

def mask_to_spwstring(mask):
    """this function convert the 1D masked array to casa 
       spw selection string

    example: 
    $ mask = [True, True, False, False, True, False]
    $ mask_to_spwstring(mask)
    '2~3;5'

    """
    n_mask = len(mask)
    i_start = 0
    i_end = -1
    selection_pairs = []
    for i,cmask in enumerate(mask):
        if cmask:
            if i_end >= i_start:
                selection_pairs.append([i_start, i_end])
            if i+1>i_start:
                  i_start=i+1
        else:
            if i+1>i_end:
                i_end = i
            if i==n_mask-1:
                selection_pairs.append([i_start, i_end])
        # print('>>> i_star={} i_end={}'.format(i_start, i_end))
    selection_string = ''
    if len(selection_pairs) > 0:
        for pair in selection_pairs:
            if np.diff(pair) == 0:
                selection_string += '{};'.format(pair[0])
            else:
                selection_string += '{}~{};'.format(*pair)
    return selection_string.strip(';')

def mask_channel(vis, frequencies=[], width=0.4,):
    """mask the spectral channels to exclude spectral lines

    Args:
        vis: the visibility
        frequencies: the possible spetral lines
        width: the line width in GHz
    """
    tb = table()
    spw_specrange = {}

    tb.open(vis + '/SPECTRAL_WINDOW')
    col_names = tb.getvarcol('NAME')
    col_freq = tb.getvarcol('CHAN_FREQ') # in units Hz
    nwindows = len(col_freq)
    tb.close()

    selection_string = ''
    for spw_name in col_names.keys():
        spw_freq = col_freq[spw_name].flatten()/1e9 # converted to GHz
        freq_mask = np.full(len(spw_freq), fill_value=False)
        for freq in frequencies:
            spw_channel = np.arange(len(spw_freq))
            # get the masked channels
            freq_mask = freq_mask | ((spw_freq>freq-0.5*width) & (spw_freq<freq+0.5*width))
        spw_selection = mask_to_spwstring(freq_mask)
        if spw_selection != '':
            selection_string += "{}:{},".format(int(spw_name[1:])-1, spw_selection)
    return selection_string.strip(',')

def alma_spw_stat(vis=None, jsonfile=None, plot=False, savefile=None,
        bands=['B3','B4','B5', 'B6', 'B7', 'B8','B9','B10'], figname=None, showfig=False,
        time_select=False, start_time='2010-01-01T00:00:00', end_time='2050-01-01T00:00:00',
        z=0, lines=None, lines_names=None, exclude_aca=True, debug=False):
    """make the statistics about one calibrator

    Args:
        objfolder (str): the folder contains all the visibility
        vis: a single visibility or a list
        jsonfile: the json file to be read
        plot (bool): whehther plot the results interactively (default is False)
            - plotbands (list): the band to be plotted (default is ['B5', 'B6',
              'B7', 'B8'])
            - figname (bool): the figname of the saved figure, if None, no figures
              will be saved (default is None)
        savedata (bool): whether save the statistics into file, default to save as
            json file
            - filename (str): the filename of the saved data

        lines (list): the spectral lines to be added to the plots, in units of GHz,
                      like: [115.3, 230.5, 345.8]
        lines_names (list): the names of the lines, like: ['CO1-0, CO2-1, CO3-2']
        z (float): the redshift of the source
    """
    tb = table()
    spw_list = {}
    for band in bands:
        spw_list[band] = {'name':[], 'time':[], 'freq':[]}
    filelist = []

    if vis is not None:
        if debug:
            print('vis', vis)
        if isinstance(vis, str):
            filelist = [vis,]
        elif isinstance(vis, list):
            filelist = vis
    elif jsonfile:
        if debug:
            print('jsonfile', jsonfile)
        with open(jsonfile, 'r') as f:
            spw_list = json.load(f)
    else:
        raise ValueError("No valid files have been given!")

    band_match = re.compile(r"_(?P<band>B\d{1,2})")
    for obs in filelist:
        # try:
        if True:
            if band_match.search(obs):
                band = band_match.search(obs).groupdict()['band']
                if debug:
                    print("Band: ", band)
            else:
                if debug:
                    print("Error in band match.")
                continue
            if band not in bands:            
                continue

            if exclude_aca:
                try:
                    tb.open(obs + '/ANTENNA')
                    antenna_diameter = np.mean(tb.getcol('DISH_DIAMETER'))
                    tb.close()
                except:
                    continue
                if antenna_diameter < 12.0:
                    if debug:
                        print("Excuding data from {}".format(antenna_diameter))
                    continue

            if time_select:
                start_time = Time(start_time)
                end_time = Time(end_time)
                try:
                    tb.open(obs)
                    obs_time = Time(tb.getcol('TIME').max()/24/3600, format='mjd')
                    tb.close()
                except:
                    if debug:
                        print("Error in opening the visibility!")
                    continue
                if debug:
                    print('> obs_time', obs_time.iso)
                    print('> start_time', start_time.iso)
                    print('> end_time', end_time.iso)
                if obs_time < start_time or obs_time > end_time:
                    if debug:
                        print(">> Skip by wrong observation time: {}".format(obs))
                    continue

            time_on_source = au.timeOnSource(obs, verbose=False, debug=False)
            time_minutes = time_on_source[0]['minutes_on_source']
            if debug:
                print('time_on_source', time_on_source)
            if time_minutes < 1e-6:
                print('No valid on source time!')
                continue
            spw_list[band]['time'].append(time_minutes)
            spw_list[band]['name'].append(os.path.basename(obs))
            spw_specrange = _read_spw(obs)
            spw_list[band]['freq'].append(list(spw_specrange))
        # except:
            # print("Error: in", obs)
    if plot:
        fig = plt.figure(figsize=(3*len(bands),5))
        # fig.suptitle(os.path.basename(objfolder))
        ax = fig.add_subplot(111)

        #ALMA band information, in GHz
        band_list = {'B3':[84, 116], 'B4':[125, 163], 'B5':[163, 211],
                'B6':[211, 275], 'B7':[275, 373], 'B8':[385, 500], \
                'B9':[602, 720], 'B10':[787, 950]}

        band_min = 1000
        band_max = 10
        for band in bands:
            if band_min > band_list[band][0]:
                band_min = np.min(band_list[band])
            if band_max < band_list[band][1]:
                band_max = np.max(band_list[band])

            ax.broken_barh([(band_list[band][0], np.diff(band_list[band])[0]),],
                           (0, 1), facecolor='lightblue', edgecolor='grey', \
                           linewidth=1, alpha=0.5)
            ax.text(np.mean(band_list[band]), 1.1, "Band"+band[1:],
                    horizontalalignment='center', verticalalignment='center')
        ax.set_xlim(band_min-10, band_max+10)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel(r'$t_{\rm on\,source}$ fraction')
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='x', which='minor', labelsize=6)
        ax.tick_params(axis='y', labelcolor='w', top='off', bottom='on', left='off', right='off', labelsize=2)

        # plot the spectral window
        for band in bands:
            h = 0
            band_total_time = np.sum(spw_list[band]['time'])
            ax.text(np.mean(band_list[band]), -0.1,
                    r"$t_{{\rm total}}$ = {:.2f} min".format(band_total_time), \
                    horizontalalignment='center', verticalalignment='center')
            n_obs = len(spw_list[band]['time'])
            if n_obs < 1:
                continue
            for i in range(n_obs):
                dh = spw_list[band]['time'][i]/band_total_time
                for spw in spw_list[band]['freq'][i]:
                    ax.broken_barh([(spw[0], np.diff(spw)[0]),],
                                   (h, dh), facecolors='salmon', edgecolors='none', \
                                   alpha=0.5)
                    ax.hlines(y=h, xmin=band_list[band][0], xmax=band_list[band][1],
                            color='r', linestyle='-', alpha=0.1, linewidth=1/n_obs)
                h = h + dh
        # plot the spectral lines
        if lines:
            if isinstance(lines, (float,int)):
                lines = [lines,]
            if lines_names:
                if isinstance(lines_names, (str)):
                    lines = [lines_names,]
            for idx, line in enumerate(lines):
                line_obs = line / (1.0 + z)
                ax.vlines(line_obs, 0, 1, alpha=0.6, color='b', linewidth=1)
                if lines_names:
                    ax.text(line_obs, 0.8, lines_names[idx], fontsize=12, alpha=0.6, horizontalalignment='center')

        if showfig:
            plt.show()
        if figname:
            fig.savefig(figname, bbox_inches='tight')
            plt.close(fig)

    if savefile is not None:
        with open(savefile, 'w') as fp:
            json.dump(spw_list, fp)
    return spw_list

def read_baselines(vis, spw=None, field=None, cell=None, imsize=None,):
    tb = table()
    tb.open(vis)
    nrows = tb.nrows()
    select = np.ones(nrows, dtype=bool)
    # select the rows of the targeted spw
    data_desc_ids = tb.getcol('DATA_DESC_ID')
    if spw is not None:
        spw_list = spw.split(',')
        for spw_id in np.unique(data_desc_ids):
            if spw_id not in spw_ids:
                raise ValueError("Please check the SPWs of the data!")
        spw_select = np.zeros(nrows, dtype=bool)
        for spw_id in spw_list:
            spw_select[data_desc_ids==int(spw_id)] = True
        select = select & spw_select

    # select the rows of the targeted field
    if field is not None:
        field_ids = tb.getcol('FIELD_ID')
        field_select = (field_ids == field)
        select = select & field_select
    rows_select = np.where(select==True)[0]
    tb_select = tb.selectrows(rows_select);
    u, v, w = tb_select.getcol('UVW')
    uvdist = np.hypot(u, v)
    return uvdist[uvdist>1e-6] # return only valide baselines

###########################################
# imaging tools
####################
def quick_image(vis, basename=None, baseline_percent=80, 
                imsize=None, cell=None, field='', spw='',
                gridder='standard',
                imsize_scale=1.7,
                suffix='', imagename=None, outdir='images',
                dry_run=True, save_fits=True):
    """quickly get the image


    quick_image(vis, baname='test1')
    """
    tb = table()
    if basename is None:
        if isinstance(vis, (list, tuple)):
            basename = 'combined'
        else:
            basename = os.path.basename(vis)
    # read the reference frequency
    spw_specrange = read_spw(vis)
    freq_mean = np.mean(spw_specrange) # in GHz
 
    # read the baselines
    if isinstance(vis, list):
        baselines_list = []
        for v in vis:
            baselines_list.append(read_baselines(v))
        baselines_list = [item for sublist in baselines_list for item in sublist]
    else:
        baselines_list = read_baselines(vis)
    baseline_typical = np.percentile(baselines_list, baseline_percent)

    # read the antenna_diameter
    if isinstance(vis, list):
        antenna_diameter_list = []
        for v in vis:
            tb.open(v + '/ANTENNA')
            antenna_diameter_list.append(tb.getcol('DISH_DIAMETER'))
            tb.close()
        antenna_diameter_list = [item for sublist in antenna_diameter_list for item in sublist]
    if isinstance(vis, str):
        tb.open(vis + '/ANTENNA')
        antenna_diameter_list = tb.getcol('DISH_DIAMETER')
        tb.close()
    antenna_diameter = np.max(antenna_diameter_list) # in m

    wavelength = const_c / freq_mean * 1e-6 # in um
    fov = 1.02 * wavelength / antenna_diameter * 206265
    print('>fov', fov)

    # calcuate the cell size
    if cell is None:
        # if myuvtaper:
            # if isinstance(myuvtaper, list):
                # myuvtaper_value = u.Unit(myuvtaper[0]).to(u.arcsec)
            # else:
                # myuvtaper_value = u.Unit(myuvtaper).to(u.arcsec)
            # uvtaper = str(myuvtaper_value)+'arcsec'
            # cellsize = np.sqrt((206265 / (baseline_typical / wavelength).decompose())**2 
                               # + myuvtaper_value**2)/ 5.0
        # else:
        cellsize = 206265 / (baseline_typical / wavelength)/ 6.0
        cellsize = np.round(cellsize*100.)/100.
        cell = "{:.2f}arcsec".format(cellsize)
        print(">cell", cell)
    # calculate image size 
    if imsize is None:
        from casatools import synthesisutils
        su = synthesisutils()
        imsize = su.getOptimumSize(int(imsize_scale * fov / cellsize))
        print(">imsize", imsize)
    # calculate frequecy
    myrestfreq = str(freq_mean)+'GHz'

    if imagename is None:
        imagename = os.path.join(outdir, basename + suffix)
    if not dry_run:
        rmtables('{}.*'.format(imagename))
        os.system('rm -rf {}.fits'.format(imagename))
        tclean(vis=vis, 
               field=field, spw=spw,
               imagename=imagename,
               intent='OBSERVE_TARGET#ON_SOURCE',
               imsize=imsize, cell=cell, #datacolumn='data',
               specmode='mfs', niter=2000, nsigma=1.0, # roughly 2sigma RMS level
               usemask='auto-multithresh',
               gridder=gridder, # only for mosaic
               #phasecenter=phasecenter, # only for mosaic
               reffreq='',
               pbcor=True, interactive=False,
               )
    else:
        print("mean frequecy:", freq_mean)
        print("maximum baseline:", baseline_typical)
        print("cell size:", cell)
        print("image size:", imsize)

    # convert images into fits file
    if save_fits:
        for imagefile in glob.glob(imagename+'.image')+glob.glob(
                imagename+'*.image.pbcor'):
            if not os.path.isfile(imagefile+'.fits'):
                exportfits(imagename=imagefile, fitsimage=imagefile+'.fits')

def quick_cube():
    pass

###########################################
# Pipeline tools
####################
# global variables
# ALMA pipeline versions: https://almascience.nrao.edu/processing/science-pipeline
ALMA_PIPELINEs = (
    ['2023-09-30','2024-09-30',"6.5.4",'casa-6.5.4-9-pipeline-2023.1.0.124'],
    ['2022-09-26','2023-09-30',"6.4.1",'casa-6.4.1-12-pipeline-2022.2.0.68'],
    ['2021-05-01','2022-09-26',"6.2.1",'casa-6.2.1-7-pipeline-2021.2.0.128'],
    # ['2021-05-10','2021-10-01',"6.1.1",'casa-6.1.1-15-pipeline-2020.1.0.40'],
    ['2019-10-01','2021-05-10',"5.6.1",'casa-pipeline-release-5.6.1-8.el7'],
    ['2018-10-01','2019-10-01',"5.4.0",'casa-release-5.4.0-70.el7'],
    ['2017-10-01','2018-10-01',"5.1.1",'casa-release-5.1.1-5.el7'],
    ['2016-10-25','2017-10.01',"4.7.2",'casa-release-4.7.2-el7'],
    ['2016-02-10','2016-10-25',"4.5.3",'casa-release-4.5.3-el6'],
    ['2015-08-07','2016-02-10',"4.3.1",'casa-release-4.3.1-pipe-el6']
    )

def choose_alma_pipeline(asdm=None, vis=None, basedir=None):
    """check the required casa pipeline version

    This function currently only support pipeline versions for linux

    Args:
        mous_id: the member mous id of the observation 
        project_id: project id
        vis: the represent visibility or the raw file
    """
    if basedir is None:
        basedir = os.path.join(os.path.expanduser('~'),'.local/casa')
    start_date = datetime.date(1858,11,17)
    final_date = ''
    if asdm is not None:
        if isinstance(asdm, (list, tuple)):
            asdm_list = asdm
        else:
            asdm_list = [asdm,]
        for sdm in asdm_list:
            xml_tree = ElementTree.parse(os.path.join(sdm, 'Main.xml'))
            root = xml_tree.getroot()
            days = int(root[-1][0].text)/1e9/3600/24
            obs_date = datetime.timedelta(days=days) + start_date
            if obs_date.strftime("%Y-%m-%d") > final_date:
                final_date = obs_date.strftime("%Y-%m-%d")
    if vis is not None:
        if isinstance(vis, (list, tuple)):
            vis_list = vis
        else:
            vis_list = [vis,]
        for visi in vis_list:
            metadata = listobs(visi)
            obs_date = datetime.timedelta(days=metadata['BeginTime']
                                          +metadata['IntegrationTime']/24./3600.) + start_date
            if obs_date.strftime("%Y-%m-%d") > final_date:
                final_date = obs_date.strftime("%Y-%m-%d")
        # print(metadata['BeginTime'])
    pipeline_version = None
    for pipe in ALMA_PIPELINEs:
        date_range = pipe[0:2]
        if (final_date >= date_range[0]) and (final_date < date_range[1]):
            pipeline_version = pipe[2]
            pipeline_selected = pipe[3]
    if pipeline_version is None:
        return None,None,None
    return pipeline_version, pipeline_selected, os.path.join(basedir, pipeline_selected, 'bin/casa')

def auto_execute_file(member_dir, mous_id=None, casa_pipeline=None, overwrite=False):
    """automatic execute the "*.scriptForPI.py" in the given directory
    
    Args:
        basedir: the starting folder contains the untarred files
    """
    subfolders = os.listdir(member_dir)
    if "calibration" not in subfolders:
        raise ValueError("No calibration can be found!")
    if "script" not in subfolders:
        raise ValueError("No script can be found!")
    if "raw" not in subfolders:
        raise ValueError("No raw files can be found!")
    if "calibrated" in subfolders:
        if not overwrite:
            print("Found existing calibrations, set 'overwrite=True' to overwrite!")
            pass
        else:
            os.system('rm -rf {}'.format(os.path.join(member_dir, "calibrated")))
    print("memeber_dir", member_dir)
    os.chdir(os.path.join(member_dir, 'script'))
    scripts_PI = glob.glob('./*.scriptForPI.py')
    if len(scripts_PI) < 1:
        raise ValueError("No scripts for PI!")
    print('{} --pipeline -c *.scriptForPI.py'.format(casa_pipeline))
    os.system('{} --pipeline --nologger -c *.scriptForPI.py'.format(casa_pipeline))
    
    #for PI_script in scripts_PI:
    #    #exec(open(PI_script).read())
    #    execfile(PI_script)

def restore_pipeline(project, dry_run=False, overwrite=False, pipeline=None, 
                     pipeline_dir=None,
                     ):
    """recover the pipeline calibration for a whole project
    
    Args:
        basedir: the starting folder contains the projects, with subfolders named as the 
                 project number, such as: 2021.1.00001.S
    """
    if (pipeline is None) and (pipeline_dir is None):
        raise ValueError('Please either give a pipeline executive or the folder includes the pipelines')
    rootdir = os.getcwd()
    for science_goal in os.listdir(os.path.join(project)):
        for group in os.listdir(os.path.join(project, science_goal)):
            for member in os.listdir(os.path.join(project,science_goal, group)):
                is_calibrated = False
                calibrated_dir = os.path.join(project, science_goal, group, member, 'calibrated')
                script_dir = os.path.join(project, science_goal, group, member, 'script')
                raw_dir = os.path.join(project, science_goal, group, member, 'raw')
                if os.path.isdir(calibrated_dir):
                    if not overwrite:
                        is_calibrated = True
                        logging.info("{}/{}/{} is Done!".format(science_goal, group, member))
                    else:
                        os.system('rm -rf {}'.format(calibrated_dir))
                if not is_calibrated:
                    # search scriptForPI
                    scripts_PI = glob.glob(os.path.join(script_dir, '*scriptForPI.py'))
                    if len(scripts_PI) < 1:
                        logging.info("{}/{}/{} has no scriptForPI".format(science_goal, 
                                                                          group, member))
                        continue
                    logging.info("Working on: {}/{}/{}".format(science_goal, group, member))
                    if pipeline is None:
                        _,_,pipeline = choose_alma_pipeline(asdm=glob.glob(raw_dir+'/*.asdm.sdm'), basedir=pipeline_dir)
                    print('asdms: {}'.format(glob.glob(raw_dir+'/*.asdm.sdm')))
                    print('casa pipeline directory: {}'.format(pipeline_dir))
                    print('casa pipeline is: {}'.format(pipeline))
                    if pipeline is None:
                        logging.info("{}/{}/{} is not supported by pipeline! maybe too old?".format(science_goal, group, member))
                        continue
                    os.chdir(script_dir)
                    for script in scripts_PI:
                        logging.info("Exectuting : {}".format(script))
                        script_basefile = os.path.basename(script)
                        if not dry_run:
                            try:
                                print('{} --nogui --agg --pipeline -c {}'.format(pipeline, script_basefile))
                                os.system('{} --nogui --agg --pipeline -c {}'.format(pipeline, script_basefile))
                                logging.info("{}/{}/{} is Done!".format(science_goal, group, member))
                            except:
                                logging.info("{}/{}/{} is failed! Checking the logs in the script folder".format(science_goal, group, member))
                # get back to the starting folder after pipeline restoration
                    os.chdir(rootdir)


def run_pipeline(rawdata=None, workdir='./', pipe_template=None,
                 pipeline=None, pipeline_dir=None, dry_run=False):
    """run costumed pipeline on a given dataset

    1. it will link the raw data to the current data
    """
    if (pipeline is None) and (pipeline_dir is None):
        raise ValueError('Please either give a pipeline executive or the folder includes the pipelines')
    # prepare the folder structure
    subprocess.run(['mkdir', '-p', os.path.join(workdir, 'rawdata')])
    subprocess.run(['mkdir', '-p', os.path.join(workdir, 'working')])
    subprocess.run(['mkdir', '-p', os.path.join(workdir, 'products')])
    # linking the raw data
    rawdata_basename = os.path.basename(rawdata)
    rawdata_id = rawdata_basename.split('.')[0]
    subprocess.run(['ln', '-s', os.path.abspath(rawdata), 
                   os.path.join(workdir, 'working', rawdata_id)])
    # generate the pipeline script
    if pipe_template is None:
        script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                  'templates')
        pipe_template = os.path.join(script_dir, 'pipeline_cal_alma.py')
        with open(pipe_template, 'r') as f:
            # lines = f.read().splitlines()
            lines = f.readlines()
        with open(os.path.join(workdir, 'working/casa_pipescript.py'), 'w') as f_out:
            for line in lines:
                if '@1' in line:
                    f_out.write(line.replace('@1', rawdata_id))
                else:
                    f_out.write(line)
    if pipeline is None:
        _,_,pipeline = choose_alma_pipeline(asdm=rawdata, basedir=pipeline_dir)
    print('casa pipeline directory: {}'.format(pipeline_dir))
    print('casa pipeline is: {}'.format(pipeline))
    if pipeline is None:
        logging.info("{}/{}/{} is not supported by pipeline! maybe too old?".format(science_goal, group, member))
    rootdir = os.getcwd()
    script_dir = os.path.join(workdir, 'working')
    if not dry_run:
        os.chdir(script_dir)
        try:
            print('{} --nogui --agg --pipeline -c casa_pipescript.py'.format(pipeline))
            os.system('export FLUX_SERVICE_URL="https://almascience.eso.org/sc/flux"')
            os.system('{} --nogui --agg --pipeline -c casa_pipescript.py'.format(pipeline))
        except:
            logging.info("Pipeline is failed! Checking the logs in the working directory")
    os.chdir(rootdir)

##########################################
# simulation with ALMA
##########################################
def simulate_almaobs(targetfile, totaltime='3600s', antennalist=["alma.cycle8.3.cfg"], pwv=0.6, niter=1000):
    with fits.open(targetfile) as hdu:
        hdu.info()
    simalma(project=os.path.basename(targetfile)[:-5], 
            skymodel=targetfile, 
            indirection="",
            incell="", # pixel size
            inbright="", #Jy/pixel
            incenter="", # frequence center
            inwidth="", # frequence center
            totaltime=totaltime,
            pwv=pwv,
            )

"""
simalma(project='U4_20704_JWST_F277W', dry_run=False, skymodel='input.fits', 
        inbright='2.4mJy/pixel', # rescale the bright pixel
        incell='', # reset the pixel scale
        incenter='140GHz', # the central frequency
        inwidth='100MHz', # the channel width
        antennalist = ['alma.cycle8.6.cfg'],
        totaltime = ['10h'],
        image=True,
        imsize = 1000
        cell = '0.03arcsec',
        niter = 200,
        threshold = '1uJy',
        )

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='alma_pipeline.py',
            description=f"Welcome to jhchen's alma utilities {__version__}",
            epilog='Reports bugs and problems to cjhastro@gmail.com')
    parser.add_argument('--basedir', type=str, default='./',
                        help='the directory includes the projects')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    parser.add_argument('--logfile', default=None, help='the filename of the log file')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')

    # add subparsers
    subparsers = parser.add_subparsers(title='Available task', dest='task', 
                                       metavar=textwrap.dedent(
        '''
            * run_pipeline
            * restore_pipeline
            * info: generate the listobs file

          To get more details about each task:
          $ alma_pipeline.py task_name --help
        '''))
    
    ################################################
    # run_pipeline
    subp_run_pipeline = subparsers.add_parser('run_pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            run costumed pipeline
            --------------------------------------------------------
            Examples:

              alma_pipeline.py run_pipeline -d '../raw/uid***.asdm.sdm' 

            '''))
    subp_run_pipeline.add_argument('-d', '--data', help='Raw data')
    subp_run_pipeline.add_argument('--workdir', default='./', help='Working directory')
    subp_run_pipeline.add_argument('--pipeline', help='The specified casa pipeline')
    subp_run_pipeline.add_argument('--pipeline_dir', default='~/apps/casa', help='The folder including all the available casa pipeline')
    # subp_run_pipeline.add_argument('--overwrite', action='store_true', 
                                      # help='Overwrite the existing files')

    ################################################
    # restore_pipeline
    subp_restore_pipeline = subparsers.add_parser('restore_pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            restore the pipeline calibration from alma data delivery
            --------------------------------------------------------
            Examples:

              alma_pipeline.py restore_pipeline --project 2010.00001.S 

            '''))
    subp_restore_pipeline.add_argument('-p', '--project', help='Project directory')
    subp_restore_pipeline.add_argument('--pipeline', help='The specified casa pipeline')
    subp_restore_pipeline.add_argument('--pipeline_dir', default='~/apps/casa', help='The folder including all the available casa pipeline')
    subp_restore_pipeline.add_argument('--overwrite', action='store_true', 
                                      help='Overwrite the existing files')

    ################################################
    # info 
    subp_info = subparsers.add_parser('info',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            generate the listobs as a text files
            --------------------------------------------
            Examples:

              alma_pipeline.py info --files *.ms 

            '''))
    subp_info.add_argument('-f', '--files', nargs='+', help='files, support widecard')
    subp_info.add_argument('--casa', help='executable casa command')
 

    ################################################
    # Welcome page
    ################################################
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

    if args.task == 'run_pipeline':
        run_pipeline(args.data, workdir=args.workdir, 
                     pipeline=args.pipeline, pipeline_dir=args.pipeline_dir,
                     dry_run=args.dry_run, 
                     # overwrite=args.overwrite,
                     )
    
    if args.task == 'restore_pipeline':
        restore_pipeline(args.project, 
                         pipeline=args.pipeline, pipeline_dir=args.pipeline_dir,
                         dry_run=args.dry_run, overwrite=args.overwrite,)
    
    if args.task == 'info':
        info(files=args.files, casa=args.casa)
    logging.info('Finished')
