#!/usr/bin/env python

# This file include the several help functions to help the data analysis of ALMA data
# Copywrite @ Jianhang Chen
# Email: cjhastro@gmail.com

# History:
    # 2019.09.27: first release, v0.1
    # 2022.08.07: add pipeline tools


import os
import sys
import glob
import re
import numpy as np
from matplotlib import pyplot as plt

version = '0.0.1'

try:
    import astropy
    has_astropy = True
except:
    print("Warning: astropy is not found, many functions may not work")
    has_astropy = False

###########################################
# Data retrieval
####################
# It connect with ALMA tap service
import multiprocessing
from functools import partial

try:
    from astroquery.alma import Alma
    has_astroquery = True
except:
    print("Warning: astroquery can be found, ALMA archive query will not work")
    has_astroquery = False

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
    
if has_astroquery:
    def query_ALMACAL(objname, band_list, savedir='./data', download_files=False, nprocess=1):
        """query for ALMA calibrator observations from ALMA archive
        
        Args:
            objname (str): alma name for the calibrator, like: J0001-0001
            band_list (str): alma bands, like: '5,6,7'
            
        """
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


###########################################
# Table tools
####################

from casatools import table, msmetadata

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

        for key in col_names.keys():
            freq_max = np.max(col_freq[key]) / 1e9
            freq_min = np.min(col_freq[key]) / 1e9
            spw_specrange[key] = [freq_min, freq_max]

    return list(spw_specrange.values())

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

    band_match = re.compile('_(?P<band>B\d{1,2})')
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


###########################################
# Pipeline tools
####################
# global variables
# ALMA pipeline versions: https://almascience.nrao.edu/processing/science-pipeline
ALMA_PIPELINE_versions = {
    "6.2.1":['2021-05-10','2099-10-01','casa-6.2.1-7-pipeline-2021.2.0.128'],
    "5.6.1":['2019-10-01','2021-05-10','casa-pipeline-release-5.6.1-8.el7'],
    "5.4.0":['2018-10-01', '2019-10-01', 'casa-release-5.4.0-70.el7'],
    "5.1.1":['2017-10-01', '2018-10-01', 'casa-release-5.1.1-5.el7'],
    "4.7.2":['2016-10-25', '2017-10.01', 'casa-release-4.7.2-el7'],
    "4.5.3":['2016-02-10', '2016-10-25', 'casa-release-4.5.3-el6'],
    "4.3.1":['2015-08-07', '2016-02-10', 'casa-release-4.3.1-pipe-1-el6']
    }

def check_pipeline_version(mous_id=None, project_id=None, vis=None):
    """check the required casa pipeline version

    Args:
        mous_id: the member mous id of the observation 
        project_id: project id
        vis: the represent visibility or the raw file
    """
    pass

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

def recover_pipeline(basedir, dry_run=False, overwrite=False, casadir=None, **kwargs):
    """recover the pipeline calibration for a whole project
    
    Args:
        basedir: the starting folder contains the untarred files
    """
    project_match = re.compile('(?P<year>\d+)\.\d+\.\d+\.\w')
    for item in os.listdir(basedir):
        if project_match.match(item):
            project_year = project_match(item).groupdict()['year']
            project_cycle
    abs_basedir = os.path.abspath(basedir)
    member_list = []
    for root, subfolders, files in os.walk(abs_basedir):
        for subfolder in subfolders:
            if subfolder == 'script':
                scripts_PI = glob.glob(os.path.join(root, subfolder, './*.scriptForPI.py'))
                if len(scripts_PI) > 0:
                    member_list.append(root)
    if dry_run:
        return member_list
    for member in member_list:
        auto_execute_file(member, overwrite=overwrite, **kwargs)



