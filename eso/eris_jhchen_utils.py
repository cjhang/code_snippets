#!/usr/bin/env python

"""
Authors: Jianhang Chen

This program was initially written when I learnt how to analysis the ESO/ERIS data for the first
time. 

History:
    - 2023-11-22: first release, v0.1
    - 2023-11-28: bug fix for testing project 110.258S, v0.2
    - 2023-12-28: test with eris piepline 1.5.0, v0.3
"""
__version__ = '0.3'
import os 
import textwrap
import inspect
import shutil
import re
import datetime
import logging
import getpass
import glob
import warnings

import numpy as np
import astropy.table as table
import astropy.units as units
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from astropy import stats as astro_stats
from astroquery.eso import Eso
from astroquery.eso.core import NoResultsWarning
import requests 

# for fits combination
from reproject import mosaicking
from reproject import reproject_adaptive, reproject_exact


#####################################
######### DATA Retrieval ############

def download_file(url, filename=None, outdir='./', auth=None): 
    """download files automatically 

    Features:
    1. fast
    2. redownload failed files
    3. skip downloaded files

    Args:
        url (str): the full url of the file
        filename (str): the filename to be saved locally
        outdir (str): the output directory
        auth (str): the authentication if needed
    """
    is_downloaded = False
    if not os.path.isdir(outdir):
        os.system(f'mkdir -p {outdir}')
    with requests.get(url, auth=auth, stream=True) as r:
        if filename is None:
            # automatically define the filename
            try:
                filename_match = re.compile('filename=(?P<filename>[\w.\-\:]+)')
                filename = filename_match.search(r.headers['Content-Disposition']).groupdict()['filename']
            except:
                logging.warning(f"Failed to find the filename from headers, set to Undefined")
                filename = 'Undefined'
        filename_fullpath = os.path.join(outdir, filename)
        # check the local file if it exists
        if os.path.isfile(filename_fullpath):
            filesize = os.path.getsize(filename_fullpath)
            try:
                if str(filesize) == r.headers['Content-Length']:
                    logging.info(f"{filename} is already downloaded.")
                    is_downloaded = True
                else:
                    logging.warning('Find local inconsistent file, overwriting...')
            except:
                logging.warning(f'Overwriting {filename_fullpath}')
        if not is_downloaded:
            with open(filename_fullpath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

def download_eris(eris_query_tab, outdir='raw', metafile=None, username=None):
    """download the calib files of eris (wrapper of download_file)

    Args:
        eris_query_tab (astropy.table): the query table returned by astroquery.eso
        outdir (str): the directory to store the download files and saved meta table
        metafile (str): the filename of the saved tabe from eris_query_tab
        save_columns (list): the selected column names to be saved.
                             set the 'None' to save all the columns
    """
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    if username is not None:
        passwd = getpass.getpass(f'{username} enter your password:\n')
        auth = requests.auth.HTTPBasicAuth(username, passwd)
    else: auth = None
    for fileid in eris_query_tab['DP.ID']:
        file_url = root_calib_url+fileid
        download_file(file_url, outdir=outdir, auth=auth)
    if metafile is None:
        metafile = os.path.join(outdir, 'metadata.csv')
    if os.path.isfile(metafile):
        os.system(f'mv {metafile} {metafile}.bak')
    eris_query_tab.write(metafile, format='csv')

def eris_auto_quary(start_date, end_date=None, start_time=12, end_time=12, max_days=30, 
                    column_filters={}, **kwargs):
    """query ESO/ERIS raw data from the database

    Args:
        start_date (str): the starting date: lile 2023-04-08
        end_date (str): the same format as start_date, the default value is start_date + 1day
        column_filters: the parameters of the query form
                        such as: 
                        column_filters = {
                                'dp_cat': 'CALIB',
                                'dp_type': 'FLAT%',
                                'seq_arm': 'SPIFFIER',
                                'ins3_spgw_name': 'K_low',
                                'ins3_spxw_name': '100mas',}
        max_days: the maximum days to search for the availble calibration files
        **kwargs: the keyword arguments of the possible column filters
                  such as: dp_tech='IFU', see more options in: 
                  https://archive.eso.org/wdb/wdb/cas/eris/form
    """
    if column_filters is None:
        column_filters={}
    for key, value in kwargs.items():
        column_filters[key] = value
    eso = Eso()
    sdatetime = datetime.datetime.strptime(f'{start_date} {start_time:0>2d}', '%Y-%m-%d %H')
    if end_date is not None:
        edatetime = datetime.datetime.strptime(f'{end_date} {end_time:0>2d}', '%Y-%m-%d %H')
        sdatetime = (edatetime - sdatetime)/2 + sdatetime
    delta_time = datetime.timedelta(days=1)
    matched = 0
    for i in range(0, max_days):
        if matched == 0:
            t_start = (sdatetime - 0.5*datetime.timedelta(days=i))
            t_end = (sdatetime + 0.5*datetime.timedelta(days=i))
            column_filters['stime'] = t_start.strftime('%Y-%m-%d')
            column_filters['etime'] = t_end.strftime('%Y-%m-%d')
            column_filters['starttime'] = t_start.strftime('%H')
            column_filters['endtime'] = t_end.strftime('%H')
            warnings.simplefilter('ignore', category=NoResultsWarning)
            tab_eris = eso.query_instrument('eris', column_filters=column_filters)
            if tab_eris is not None:
                matched = 1
    if matched == 0:
        # print("Cannot find proper calib files, please check your input")
        logging.warning("eris_auto_quary: cannot find proper calib files!")
    else:
        return tab_eris

def request_eris_calib(start_date=None, band=None, resolution=None, 
                       exptime=None, outdir='raw',
                       end_date=None, dpcat='CALIB', arm='SPIFFIER', 
                       metafile=None,
                       steps=['dark','detlin','distortion','flat','wavecal'],
                       dry_run=False, **kwargs):
    """a general purpose to qeury calib files of ERIS/SPIFFIER observation
    
    Args:
        start_date (str): ISO date format, like: 2023-04-08
        end_date (str, None): same format as start_date
        band (str): grating configurations
        resolution (str): the spaxel size of the plate, [250mas, 100mas, 25mas]
        exptime (int,float): the exposure time, in seconds
        outdir (str): the output directory of the download files
        steps (list): a list of calibration steps, the connection with DRP types:
            
            --------------------------------------------------------------------------
            name         dp_type           function
            --------------------------------------------------------------------------
            dark         DARK              dark data reduction (daily)
            detlin       LINEARITY%        detector's non linear bad pixels (monthly)
            distortion   NS%               distortion correction
            flat         FLAT%             flat field data reduction
            wavecal      WAVE%             wavelength calibration
            stdstar      %STD%             std stars, including the flux stdstar
            psfstar      %PSF-CALIBRATOR   the psfstar
            --------------------------------------------------------------------------

        dpcat (str): default to be 'CALIB'
        arm (str): the instrument arm of eris: SPIFFIER or NIX (in develop)
        dry_run: return the list of files instead of downloading them

    """
    dptype_dict = {'dark':'DARK', 'detlin':'LINEARITY%', 'distortion':'NS%',
                   'flat':'FLAT%', 'wavecal':'WAVE%', 'stdstar':'%STD%', 
                   'psfstar': '%PSF-CALIBRATOR'}
    if exptime is None: exptime = ''
    query_tabs = []

    for step in steps:
        logging.info(f'Requesting {step} calibration files')
        column_filters = {'dp_cat': dpcat,
                          'seq_arm': arm,
                          'dp_type': dptype_dict[step]}
        if step == 'dark':
            # drop the requirement for band and resolution
            column_filters['exptime'] = exptime
        if step in ['distortion', 'flat', 'wavecal', 'stdstar', 'psfstar']:
            column_filters['ins3_spgw_name'] = band
            column_filters['ins3_spxw_name'] = resolution

        step_query = eris_auto_quary(start_date, end_date=end_date, column_filters=column_filters, **kwargs)
        # fix the data type issue of masked table columns
        if step_query is not None:
            for col in step_query.colnames:
                step_query[col] = step_query[col].astype(str)
            query_tabs.append(step_query)
        else:
            # raise ValueError('Failed in requesting the calib files! Consider release the `max_days`?')
            logging.warning(f"No calib file found for {step}, consider release the `max_days`?'")
        if len(query_tabs) > 0:
            all_tabs = table.vstack(query_tabs)
        else:
            all_tabs = []
    if dry_run:
        return all_tabs
    else:
        if len(all_tabs) > 0:
            download_eris(all_tabs, metafile=metafile, outdir=outdir)
        else:
            logging.warning("No files for downloading!")

def requests_eris_science(program_id='', username=None, metafile='metadata.csv',
                          outdir=None, target='', observation_id='', 
                          start_date='', end_date='', debug=False, **kwargs):
    """download the science data 

    To download the proprietory data, you need to provide your eso username
    and you will be asked to input your password.

    Args:
        program_id (str): program id
        username (str): the user name of your eso account
        metafile (str): the output file to store all the meta data
                        default: metadata.csv
        target (str): the target name
        outdir (str): the directory to save all the raw files
        observation_id (str, int): the id of the observation
        start_date (str): starting data, in the format of '2023-04-08'
        end_date (str): end date, same format as start_date
        **kwargs: other keyword filters
    """
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    if outdir is None:
        if program_id is not None: outdir = program_id+'_raw'
        else: outdir = 'raw'
    if os.path.isdir(outdir):
        os.system(f'mkdir -p {outdir}')
    logging.info(f'Requesting the data from project: {program_id}')
    eso = Eso()
    if end_date is None:
        sdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        edate = sdate + datetime.timedelta(days=1)
        end_date = edate.strftime('%Y-%m-%d')
    eris_query_tab = eso.query_instrument(
            'eris', column_filters={'ob_id': observation_id,
                                    'prog_id':program_id,
                                    'stime': start_date,
                                    'etime': end_date,
                                    'target': target,})
    if debug:
        return eris_query_tab
    else:
        download_eris(eris_query_tab, username=username, outdir=outdir, metafile=metafile)


def generate_metafile(dirname, metafile='metadata.csv', extname='PRIMARY', 
                      work_dir=None, clean_work_dir=False):
    """generate metafile from download files

    Args:
        dirname (str): the directory include the fits file
    """
    # colnames
    colnames = ['Release Date', 'Object', 'RA', 'DEC','Program ID', 'DP.ID', 'OB.ID', 
                'OBS.TARG.NAME', 'EXPTIME', 'DPR.CATG', 'DPR.TYPE', 'DPR.TECH', 'TPL.START', 
                'SEQ.ARM', 'DET.SEQ1.DIT', 'INS3.SPGW.NAME', 'INS3.SPXW.NAME']
    colnames_header = ['DATE', 'OBJECT', 'RA', 'DEC', 'HIERARCH ESO OBS PROG ID', 'ARCFILE', 
                       'HIERARCH ESO OBS ID', 'HIERARCH ESO OBS NAME', 'EXPTIME', 
                       'HIERARCH ESO DPR CATG', 'HIERARCH ESO DPR TYPE', 'HIERARCH ESO DPR TECH', 
                       'HIERARCH ESO TPL START', 'HIERARCH ESO SEQ ARM', 'HIERARCH ESO DET SEQ1 DIT',
                       'HIERARCH ESO INS3 SPFW NAME', 'HIERARCH ESO INS3 SPXW NAME']
    dirname = dirname.strip('/')
    fits_Zfiles = glob.glob(dirname+'/*.fits.Z')
    fits_files = glob.glob(dirname+'/*.fits')
    if work_dir is None:
        work_dir = '.tmp_generate_metafile'
        clean_work_dir = True

    dir_umcompressed = os.path.join(work_dir, 'umcompressed')
    dir_header = os.path.join(work_dir, 'headers')
    for d in [dir_umcompressed, dir_header]:
        if not os.path.isdir(d):
            os.system(f'mkdir -p {d}')
    # compress the fits file
    if len(fits_Zfiles) > 0:
        for ff in fits_Zfiles:
            os.system(f'cp {ff} {dir_umcompressed}/')
        os.system(f'uncompress {dir_umcompressed}/*.Z')
        fits_Zfiles = glob.glob(f'{dir_umcompressed}/*.fits')
    
    fits_files = fits_files + fits_Zfiles
    # extract info from the headers and save the info
    meta_tab = table.Table(names=colnames, dtype=['U32']*len(colnames))
    if len(fits_files) > 0:
        for ff in fits_files:
            with fits.open(ff) as hdu:
                header = hdu[extname].header
                # [header[cn] if cn in header.keys() else '' for cn in colnames_header]
                header_values = []
                for cn in colnames_header:
                    try: header_values.append(str(header[cn]).strip('.fits'))
                    except: header_values.append('')
                meta_tab.add_row(header_values)
    if os.path.isfile(metafile):
        os.system(f'mv {metafile} {metafile}.bak')
    meta_tab.write(metafile, format='csv')
    if clean_work_dir:
        os.system(f'rm -rf {work_dir}')


#####################################
######### DATA Calibration ##########

def search_static_calib(esorex):
    # use the default staticPool
    if '/' in esorex:
        try:
            binpath_match = re.compile('(?P<bindir>^[\/\w\s\_\.\-]*)/esorex')
            bindir = binpath_match.search(esorex).groupdict()['bindir']
        except:
            raise ValueError('Failed to locate the install direction of esorex!')
    else: 
        bindir = os.path.dirname(f'which {esorex}')
    install_dir = os.path.dirname(bindir)
    static_pool_list = glob.glob(os.path.join(install_dir, 'calib/eris-*'))
    static_pool = sorted(static_pool_list)[-1] # choose the latest one
    return static_pool

def generate_calib(metafile, raw_pool='./raw', work_dir=None, 
                   calib_pool='./calibPool', static_pool=None,
                   steps=['dark','detlin','distortion','flat','wavecal'],
                   dark_sof=None, detlin_sof=None, distortion_sof=None, flat_sof=None, 
                   wavecal_sof=None, drp_type_colname='DPR.TYPE', 
                   esorex=None, dry_run=False, archive=False, archive_name=None,
                   debug=False):
    """generate the science of frame of each calibration step

    Args:
        metafile (str): the metadata file where you can find the path and objects of each dataset
        raw_pool (str): the raw data pool
        static_pool (str): the static data pool
        calib_pool (str): the directory to keep all the output files from esorex
        steps (list): the steps to generate the corresponding calibration files
                      it could include: ['dark','detlin','distortion','flat','wavecal']
        drp_type_colname (str): the column name of the metadata table includes the DPR types. 
        esorex (str): the callable esorex command from the terminal
        dry_run (str): set it to True to just generate the sof files
        archive (bool): switch to archive mode, where the calibPool is organised in the folder
                        based structures, the subfolder name is specified by archive_name
        archive_name (str): the archive name. It is suggested to be the date of the observation
    """
    cwd = os.getcwd()
    calib_pool = calib_pool.rstrip('/')
    raw_pool = raw_pool.rstrip('/')
    if static_pool is None:
        # use the default staticPool
        static_pool = search_static_calib(esorex)
    else: static_pool = static_pool.rstrip('/')
    if archive:
        if archive_name is None:
            # use the date tag as the archive name
            # archive_name = datetime.date.today().strftime("%Y-%m-%d")
            raise ValueError('Please give the archive name!')
        calib_pool = os.path.join(calib_pool, archive_name)
        if not os.path.isdir(calib_pool):
            os.system(f'mkdir -p {calib_pool}')
        sof_name = f'{archive_name}.sof'
        work_dir = calib_pool
    else:    
        sof_name = 'esorex_ifu_eris.sof'
    # setup directories
    if work_dir is None:
        work_dir = '.'
    for dire in [work_dir, calib_pool]:
        if not os.path.isdir(dire):
            os.system(f'mkdir -p {dire}')

    if esorex is None: esorex_cmd = f'esorex --output-dir={calib_pool}'
    else: esorex_cmd = f'{esorex} --output-dir={calib_pool}'
    meta_tab = table.Table.read(metafile, format='csv')

    if 'dark' in steps:
        if dark_sof is None:
            # generate the sof for dark calibration
            dark_sof = os.path.join(work_dir, 'dark.sof')
            # dark_sof = os.path.join(calib_pool, sof_name)
            with open(dark_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z DARK\n")
                # read the timestamp and exptime
                timestamp = item['Release Date']
                exptime = item['DET.SEQ1.DIT']
                openf.write(f'# dark: date={timestamp} exptime={exptime}\n')
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_dark {dark_sof}")
            # if rename
                # # rename the files with keywords
                # dark_bpm_fits = f'{calib_pool}/eris_ifu_dark_bpm_{exptime:0.1f}s_{timestamp}.fits'
                # dark_master_fits = f'{calib_pool}/eris_ifu_dark_master_{exptime:0.1f}s_{timestamp}.fits'
            # else:
                # dark_bpm_fits = f'{calib_pool}/eris_ifu_dark_bpm.fits'
                # dark_master_fits = f'{calib_pool}/eris_ifu_dark_master'
            # os.system(f'mv {work_dir}/eris_ifu_dark_bpm.fits {dark_bpm_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_dark_bpm.fits {dark_master_fits}')
    if 'detlin' in steps:
        if detlin_sof is None:
            # generate the sof for detector's linarity
            detlin_sof = os.path.join(work_dir, 'detlin.sof')
            # detlin_sof = os.path.join(calib_pool, sof_name)
            with open(detlin_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'LINEARITY,DARK,DETCHAR']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z LINEARITY_LAMP\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'LINEARITY,LAMP,DETCHAR']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z LINEARITY_LAMP\n")
                # read the timestamp
                timestamp = item['Release Date']
                openf.write(f'# detlin: date={timestamp}\n')
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_detlin {detlin_sof}")
            # if rename:
                # detlin_bpm_filt_fits = f'{calib_pool}/eris_ifu_detlin_bpm_filt_{timestamp}.fits'
                # detlin_bpm_fits = f'{calib_pool}/eris_ifu_detlin_bpm_{timestamp}.fits'
                # detlin_gain_info_fits = f'{calib_pool}/eris_ifu_detlin_gain_info_{timestamp}.fits'
            # else:
                # detlin_bpm_filt_fits = f'{calib_pool}/eris_ifu_detlin_bpm_filt.fits'
                # detlin_bpm_fits = f'{calib_pool}/eris_ifu_detlin_bpm.fits'
                # detlin_gain_info_fits = f'{calib_pool}/eris_ifu_detlin_gain_info.fits'
            # os.system(f'mv {work_dir}/eris_ifu_detlin_bpm_filt.fits {detlin_bpm_filt_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_detlin_bpm.fits {detlin_bpm_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_detlin_gain_info.fits {detlin_gain_info_fits}')

    if 'distortion' in steps:
        if distortion_sof is None:
            # generate the sof for distortion
            distortion_sof = os.path.join(work_dir, 'distortion.sof')
            # distortion_sof = os.path.join(calib_pool, sof_name)
            with open(distortion_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z DARK_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,SLIT']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FIBRE_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,WAVE,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z WAVE_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,WAVE,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z WAVE_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,FLAT,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FLAT_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,FLAT,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FLAT_NS\n")
                openf.write(f"{static_pool}/eris_ifu_first_fit.fits FIRST_WAVE_FIT\n") 
                openf.write(f"{static_pool}/eris_ifu_ref_lines.fits REF_LINE_ARC\n")
                openf.write(f"{static_pool}/eris_ifu_wave_setup.fits WAVE_SETUP\n")
                # read the timestamp, band, and resolution
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                resolution = item['INS3.SPXW.NAME']
                openf.write(f'# distortion: date={timestamp} band={band} resolution={resolution}\n')
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_distortion {distortion_sof}")
            # if rename:
                # distortion_bpm_fits = f'{calib_pool}/eris_ifu_distortion_bpm_{band}_{resolution}_{timestamp}.fits'
                # distortion_distortion_fits = f'{calib_pool}/eris_ifu_distortion_distortion_{band}_{resolution}_{timestamp}.fits'
                # distortion_slitlet_pos_fits = f'{calib_pool}/eris_ifu_distortion_slitlet_pos_{band}_{resolution}_{timestamp}.fits'
            # else:
                # distortion_bpm_fits = f'{calib_pool}/eris_ifu_distortion_bpm.fits'
                # distortion_distortion_fits = f'{calib_pool}/eris_ifu_distortion_distortion.fits'
                # distortion_slitlet_pos_fits = f'{calib_pool}/eris_ifu_distortion_slitlet_pos.fits'
            # os.system(f'mv {work_dir}/eris_ifu_distortion_bpm.fits {distortion_bpm_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_distortion_distortion.fits {distortion_distortion_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_distortion_slitlet_pos.fits {distortion_slitlet_pos_fits}')
    if 'flat' in steps:
        if flat_sof is None:
            # generate the sof for flat
            flat_sof = os.path.join(work_dir, 'flat.sof')
            # flat_sof = os.path.join(calib_pool, sof_name)
            with open(flat_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'FLAT,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FLAT_LAMP\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'FLAT,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FLAT_LAMP\n")
                openf.write(f"{calib_pool}/eris_ifu_dark_bpm.fits BPM_DARK\n")
                openf.write(f"{calib_pool}/eris_ifu_detlin_bpm_filt.fits BPM_DETLIN\n")
                openf.write(f"{calib_pool}/eris_ifu_distortion_bpm.fits BPM_DIST\n")
                # read the timestamp, band, and resolution
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                resolution = item['INS3.SPXW.NAME']
                openf.write(f'# flat: date={timestamp} band={band} resolution={resolution}\n')
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_flat {flat_sof}")
            # if rename:
                # flat_bpm_fits = f'{calib_pool}/eris_ifu_flat_bpm_{band}_{resolution}_{timestamp}.fits'
                # flat_master_flat_fits = f'{calib_pool}/eris_ifu_flat_master_flat_{band}_{resolution}_{timestamp}.fits'
            # else:
                # flat_bpm_fits = f'{calib_pool}/eris_ifu_flat_bpm.fits'
                # flat_master_flat_fits = f'{calib_pool}/eris_ifu_flat_master_flat.fits'
            # os.system(f'mv {work_dir}/eris_ifu_flat_bpm.fits {flat_bpm_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_flat_bpm.fits {flat_master_flat_fits}')
    if 'wavecal' in steps:
        if wavecal_sof is None:
            # generate the sof for wavecal
            wavecal_sof = os.path.join(work_dir, 'wavecal.sof')
            # wavecal_sof = os.path.join(calib_pool, sof_name)
            with open(wavecal_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'WAVE,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z WAVE_LAMP\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'WAVE,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z WAVE_LAMP\n")
                openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
                openf.write(f"{static_pool}/eris_ifu_ref_lines.fits REF_LINE_ARC\n")
                openf.write(f"{static_pool}/eris_ifu_wave_setup.fits WAVE_SETUP\n") 
                openf.write(f"{static_pool}/eris_ifu_first_fit.fits FIRST_WAVE_FIT\n") 
                openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
                openf.write(f"{calib_pool}/eris_ifu_flat_bpm.fits BPM_FLAT\n")
                # read the timestamp, band, and resolution
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                resolution = item['INS3.SPXW.NAME']
                openf.write(f'# wavecal: date={timestamp} band={band} resolution={resolution}\n')
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_wavecal {wavecal_sof}")
            # if rename:
                # wave_map_fits = f'{calib_pool}/eris_ifu_wave_map_{band}_{resolution}_{timestamp}.fits'
                # wave_arcImg_resampled_fits = f'{calib_pool}/eris_ifu_wave_arcImag_resampled_{band}_{resolution}_{timestamp}.fits'
                # wave_arcImg_stacked_fits = f'{calib_pool}/eris_ifu_wave_arcImag_stacked_{band}_{resolution}_{timestamp}.fits'
            # else:
                # wave_map_fits = f'{calib_pool}/eris_ifu_wave_map.fits'
                # wave_arcImg_resampled_fits = f'{calib_pool}/eris_ifu_wave_arcImag_resampled.fits'
                # wave_arcImg_stacked_fits = f'{calib_pool}/eris_ifu_wave_arcImag_stacked.fits'
            # os.system(f'mv {work_dir}/eris_ifu_wave_map.fits {wave_map_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_wave_arcImag_resampled.fits {wave_arcImg_resampled_fits}')
            # os.system(f'mv {work_dir}/eris_ifu_wave_arcImag_stacked.fits {wave_arcImg_stacked_fits}')

def auto_gitter(metafile=None, raw_pool=None, outdir='./', calib_pool='calibPool', 
                static_pool=None, band='', esorex='', mode='stdstar',
                dry_run=False):
    """calibrate the science target or the standard stars
    """
    calib_pool = calib_pool.rstrip('/')
    meta_tab = table.Table.read(metafile, format='csv')
    
    if not os.path.isdir(outdir):
        os.system(f'mkdir -p {outdir}')
    
    if static_pool is None:
        # use the default staticPool
        static_pool = search_static_calib(esorex)
    else:
        static_pool = static_pool.rstrip('/')
    
    auto_jitter_sof = os.path.join(outdir, 'auto_jitter.sof')
    if esorex is None: esorex_cmd = f'esorex --output-dir={outdir}'
    else: esorex_cmd = f'{esorex} --output-dir={outdir}'

    with open(auto_jitter_sof, 'w+') as openf:
        # write OBJ
        if mode == 'gitter':
            for item in meta_tab[meta_tab['DPR.CATG'] == 'SCIENCE']:
                if item['DPR.TYPE'] == 'OBJECT':
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z OBJ\n")
                elif item['DPR.TYPE'] == 'SKY':
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z SKY_OBJ\n")
        elif mode == 'stdstar':
            stdstar_names = meta_tab[meta_tab[drp_type_colname] == 'STD']['OBS.TARG.NAME'].data.tolist()
            if len(stdstar_names)>1:
                logging.warning(f'Finding more than one stdstar: {stdstar_names}') 
                logging.warning(f'Choosing the first one: {stdstar_names[0]}')
            stdstar_name = stdstar_names[0]
            for item in meta_tab[meta_tab[drp_type_colname] == 'STD']:
                if item['OBS.TARG.NAME'] == stdstar_name:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z STD #{stdstar_name}\n")
                else: openf.write(f"#{raw_pool}/{item['DP.ID']}.fits.Z STD #{item['OBS.TARG.NAME']}\n")
            for item in meta_tab[meta_tab[drp_type_colname] == 'SKY,STD']:
                if item['OBS.TARG.NAME'] == stdstar_name:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z SKY_STD #{stdstar_name}\n")
                else: openf.write(f"#{raw_pool}/{item['DP.ID']}.fits.Z SKY_STD #{item['OBS.TARG.NAME']}\n")
            # for psf stars
            for item in meta_tab[meta_tab[drp_type_colname] == 'PSF,SKY,STD']: #TODO
                if item['OBS.TARG.NAME'] == stdstar_name:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z PSF_CALIBRATOR #{stdstar_name}\n")
                else: openf.write(f"#{raw_pool}/{item['DP.ID']}.fits.Z SKY_PSF_CALIBRATOR #{item['OBS.TARG.NAME']}\n")
        openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
        openf.write(f"{calib_pool}/eris_ifu_wave_map.fits WAVE_MAP\n")
        #openf.write(f"{calib_pool}/eris_ifu_distortion_slitlet_pos.fits SLITLET_POS\n")
        openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
        openf.write(f"{calib_pool}/eris_ifu_dark_master_dark.fits MASTER_DARK\n")
        #openf.write(f"{static_pool}/EXTCOEFF_TABLE.fits EXTCOEFF_TABLE\n")
        openf.write(f"{static_pool}/eris_oh_spec.fits OH_SPEC\n")
        if band in ['H_low', 'J_low', 'K_low']:
            openf.write(f"{static_pool}/RESPONSE_WINDOWS_{band}.fits RESPONSE\n")

    if not dry_run:
        if mode == 'stdstar':
            os.system(f"{esorex_cmd} eris_ifu_stdstar {auto_jitter_sof}")
        elif mode == 'gitter':
            os.system(f'{esorex_cmd} eris_ifu_jitter --product_depth=2 --sky_tweak=0 --dar-corr=true {auto_jitter_sof}')
            # os.system(f'{esorex} eris_ifu_jitter --derot_corr --product_depth=2 --tbsub=true --sky_tweak=1 --dar-corr=true --flux-calibrate=false {science_sof}')

def eris_pipeline(project, start_date, band, resolution, program_id, 
                  username=None, end_date=None, observation_id='',
                  target='',
                  outdir='./', static_pool=None, esorex='esorex',
                  **kwargs):
    """simple pipeline for eris data reduction
    
    Args:
        project (str): the project code, used to orgnise the folder
        outdir (str): the output director. By default, a project folder
                      will be created inside the output directory
        
    """
    project_dir = os.path.join(outdir, project)
    if os.path.isdir(os.path.isdir(project_dir)):
        logging.warning("project folder existing! Reusing all the possible data!")
    
    # preparing the all the necessary folders
    project_calib_raw = project_dir+'/calib_raw'
    project_science_raw = project_dir+'/science_raw'
    project_calib_pool = project_dir+'/calibPool'
    project_calibrated = project_dir+'/calibrated'
    working_dirs = [project_calib_raw, project_science_raw, project_calib_pool, 
                    project_calibrated]
    for wd in working_dirs:
        if not os.path.isdir(wd):
            os.system(f'mkdir -p {wd}')
   
    # download the calibration and science raw files
    request_eris_calib(start_date, band, resolution, exptime, outdir=project_calib_raw,
                       end_date=end_date, **kwargs)
    requests_eris_science(program_id=program_id, username=username, 
                          outdir=project_science_raw,
                          start_date=start_date, end_date=end_date, 
                          observation_id=observation_id, target=target,
                          **kwargs)

    # generate the calib files
    generate_calib(metafile=os.path.join(raw_pool, 'metadata.csv'), 
                   raw_pool=project_calib_raw, static_pool=static_pool, 
                   calib_pool=project_calib_pool, drp_type_colname='DPR.TYPE',
                   esorex=esorex)
    
    # run calibration
    auto_gitter(metafile=os.path.join(raw_pool, 'metadata.csv'),
                raw_pool=project_science_raw, calib_pool=project_calib_pool,
                outdir=project_calibrated, static_pool=static_pool,
                grating=grating, esorex=esorex)

#####################################
######### DATA Combination ##########

def fix_micron_unit_header(header):
    """this small program fix the unrecongnized unit "micron" by astropy.wcs

    *only tested with the 3D datacube from VLT/ERIS
    """
    if 'CUNIT3' in header:
        if header['CUNIT3'] == 'MICRON':
            header['CUNIT3'] = 'um'
    return header

def find_combined_wcs(image_list=None, wcs_list=None, header_ext='DATA', frame=None, resolution=None,
                      pixel_shifts=None):
    """compute the final coadded wcs

    It suports the combination of the 3D datacubes.
    It uses the first wcs to comput the coverage of all the images;
    Then, it shifts the reference point to the center.
    If resolution provided, it will convert the wcs to the new resolution

    Args:
        image_list (list, tuple, np.ndarray): a list fitsfile, astropy.io.fits.header, 
                                              or astropy.wcs.WCS <TODO>
        wcs_list (list, tuple, np.ndarray): a list of astropy.wcs.WCS, need to include
                                            the shape information
        header_ext (str): the extension name of the fits card
        frame (astropy.coordinate.Frame): The sky frame, by default it will use the 
                                          frame of the first image
        resolution (float): in arcsec, the final resolution of the combined image <TODO>
        pixel_shifts (list, tuple, np.ndarray): same length as image_list, with each 
                                                element includes the drift in each 
                                                dimension, in the order of [(drift_x(ra),
                                                drift_y(dec), drift_chan),]
    """
    # if the input is fits files, then first calculate their wcs
    if image_list is not None:
        wcs_list = []
        for i,fi in enumerate(image_list):
            with fits.open(fi) as hdu:
                header = fix_micron_unit_header(hdu[header_ext].header)
                image_wcs = WCS(hdu[header_ext].header)
                wcs_list.append(image_wcs)
    
    # check the shape of the shifts
    n_wcs = len(wcs_list)
    if pixel_shifts is not None:
        if len(pixel_shifts) != n_wcs:
            raise ValueError("Pixel_shift does not match the number of images or WCSs!")
        pixel_shifts = np.array(pixel_shifts)

    # get the wcs of the first image
    first_wcs = wcs_list[0] 
    first_shape = first_wcs.array_shape # [size_chan, size_y, size_x]
    naxis = first_wcs.wcs.naxis

    # then looping through all the images to get the skycoord of the corner pixels
    if naxis == 2: # need to reverse the order of the shape size
        # compute the two positions: [0, 0], [size_x, size_y]
        corner_pixel_coords = [(0,0), np.array(first_shape)[::-1]-1] # -1 because the index start at 0
    elif naxis == 3:
        # compute three positions: [0,0,0], [size_x, size_y, size_chan]
        corner_pixel_coords = [(0,0,0), np.array(first_shape)[::-1]-1]
    else: 
        raise ValueError("Unsupport datacube! Check the dimentions of the datasets!")
    image_wcs_list = []
    corners = []
    resolutions = []
    for i,fi in enumerate(wcs_list):
            image_wcs = wcs_list[i]
            if pixel_shifts is not None:
                image_wcs.wcs.crpix -= pixel_shifts[i]
            array_shape = image_wcs.array_shape
            # get the skycoord of corner pixels
            for pixel_coord in corner_pixel_coords:
                # pixel order: [x, y, chan]
                corner = wcs_utils.pixel_to_pixel(image_wcs, first_wcs, *pixel_coord)
                # up_corner = wcs_utils.pixel_to_pixel(first_wcs, image_wcs, *array_shape)
                corners.append(corner)
            resolutions.append(wcs_utils.proj_plane_pixel_scales(image_wcs))

    # calculate the reference point
    corners = np.array(corners)
    low_boundaries = np.min(corners, axis=0)
    up_boundaries = np.max(corners, axis=0)
    ranges = np.round(up_boundaries - low_boundaries + 1).astype(int) # [range_x, range_y, range_chan]
    chan0 = low_boundaries[0]
    x0, y0 = ranges[:2]*0.5 # only need the first two for x and y

    # get the skycoord of the reference point
    reference_skycoord = wcs_utils.pixel_to_skycoord(x0, y0, wcs=first_wcs)

    # assign the new reference to the new wcs
    wcs_combined = first_wcs.deepcopy()
    if naxis == 3:
        # shift the reference point to the center
        # reference channel point the first channel of the combined data
        try: dchan = first_wcs.wcs.cd[-1,-1]
        except: pass
        try: dchan = first_wcs.wcs.pc[-1,-1]
        except: 
            raise ValueError("Cannot read the step size of the spectral dimension!")

        reference_chan = first_wcs.wcs.crval[-1] + (first_wcs.wcs.crpix[-1]-chan0-1)*dchan
        wcs_combined.wcs.crval = np.array([reference_skycoord.ra.to(units.deg).value, 
                         reference_skycoord.dec.to(units.deg).value,
                         reference_chan])
        wcs_combined.wcs.crpix = np.array([x0, y0, 1])
        # wcs_combined.wcs.cdelt = wcs.wcs.cd.diagonal() # cdelt will be ignored when CD is present
    elif naxis == 2:
        wcs_combined.wcs.crval = np.array([reference_skycoord.ra.to(units.deg).value, 
                                           reference_skycoord.dec.to(units.deg).value])
    wcs_combined.array_shape = tuple(ranges[::-1]) # need to reverse again

    # by default, the resolution of the first image will be used
    # update the resolution if needed
    if resolution is not None:
        min_resolutions = np.min(np.array(resolutions), axis=1)
        scales = min_resolutions / first_resolutions
        wcs_new = wcs_combined.deepcopy()
        wcs_new.wcs.cd = wcs_new.wcs.cd * scales
        if (scales[-1] - 1) > 1e-6:
            nchan = int(wcs_combined.array_shape[0] / scales)
        x0_new, y0_new = wcs_utils.skycoord_to_pixel(reference_skycoord)
        wcs_new.crpix = np.array([x0_new.item(), y0_new.item(), 1]).astype(int)
        wcs_new.array_shape = tuple(np.round(np.array(wcs_combined.array_shape) / scales).astype(int))
        wcs_combined = wcs_new
    return wcs_combined

def find_combined_wcs_test(image_list, wcs_list=None, header_ext='DATA', frame=None, 
                           resolution=None, ):
    """this is just a wrapper of reproject.mosaicking.find_optimal_celestial_wcs
    
    Used to test the performance of `find_combined_wcs`
    """
    # define the default values
    image_wcs_list = []
    for img in image_list:
        # read the image part
        with fits.open(img) as hdu:
            header = hdu[header_ext].header
            image_shape = (header['NAXIS2'], header['NAXIS1'])
            nchan = header['NAXIS3']
            image_wcs = WCS(header).celestial #sub(['longitude','latitude'])
            if frame is None:
                frame = wcs_utils.wcs_to_celestial_frame(image_wcs)    
            image_wcs_list.append((image_shape, image_wcs))
    wcs_combined, shape_combined = mosaicking.find_optimal_celestial_wcs(
            tuple(image_wcs_list), frame=frame, resolution=resolution)
    return wcs_combined, shape_combined

def compute_weighting_eris(image_list, mode='exptime', header_ext='DATA'):
    """compute the weighting of each image

    1. computing the weighting based on the integration time
    2. based on the RMS level
    """
    if mode == 'exptime':
        total_time = 0
        time_list = []
        for img in image_list:
            with fits.open(img) as hdu:
                header = hdu[header_ext].header
                total_time += header['EXPTIME']
                time_list.append(header['EXPTIME'])
        return np.array(time_list)/total_time

def fill_mask(image, mask):
    """Using iterative median to filled the masked region
    In each cycle, the masked pixel is set to the median value of all the values in the 
    surrounding region (in cubic 3x3 region, total 8 pixels)
    Inspired by van Dokkum+2023 (PASP) and extended to support 3D datacube
    
    Args:
        image (ndarray): the input image
        mask (ndarray): the same shape as image, with masked pixels are 1 and rest are 0
    """
    ndim = image.ndim
    image_filled = image.copy().astype(float)
    image_filled[mask==1] = np.nan
    image_shape = np.array(image.shape)
    up_boundaries = np.repeat(image_shape, 2) - 1
    mask_idx = np.argwhere(mask > 0)
    while np.any(np.isnan(image_filled)):
        for idx in mask_idx:
            idx_range = np.array([[i-1,i+2] for i in idx])
            # check if reaches low boundaries, 0
            if np.any(idx < 1):  
                idx_range[idx_range < 0] = 0
            # check if reach the upper boundaries
            if np.any(image_shape - idx < 1):
                idx_range[idx_range>up_boundaries] = up_boundaries[idx_range>up_boundaries]
            ss = tuple(np.s_[idx_range[i][0]:idx_range[i][1]] for i in range(ndim))
            image_filled[tuple(idx)] = np.nanmedian(image_filled[ss])
    return image_filled

def construct_wcs(header, data_shape=None):
    """try to construct the wcs from a broken header 
    """
    try:
        # read some useful information
        ndim = header['NAXIS']
        crpix1, crpix2 = header['CRPIX1'], header['CRPIX2']
        ra, dec = header['CRVAL1'], header['CRVAL2']
        cdelt1, cdelt2 = header['CD1_1'], header['CD2_2']
        cunit1, cunit2 = header['CUNIT1'], header['CUNIT2']
        if ndim>2:
            crpix3 = header['CRPIX3']
            crvar3 = header['CRVAL3']
            cdelt3 = header['CD3_3']
            cunit3 = header['CUNIT3']
    except:
        ra, dec = header['RA'], heaer['DEC']
        crpix1, crpix2 = xsize/2., ysize/2.
        cdelt1, cdelt2 = resolution/3600., resolution/3600.
        cunit1, cunit2 = 'deg', 'deg'
        if ndim>2:
            # should be fine for given random units, as all the wcs 
            # shares the same units
            crpix3 = 1
            crvar3 = 1
            cdelt3 = 1 
            cunit1 = 'um'
    if True:
        data_shape = hdu[data_ext].data.shape
        ndim = len(data_shape)
        ysize, xsize = data_shape[-2:]
        print('Warning: making use of mock wcs!')
        wcs_mock = WCS(naxis=ndim)
        if ndim == 2:
            wcs_mock.wcs.crpix = crpix1, crpix2
            wcs_mock.wcs.cdelt = cdelt1, cdelt2
            wcs_mock.wcs.cunit = 'deg', 'deg'
            wcs_mock.wcs.ctype = 'RA', 'DEC'
            wcs_mock.array_shape = [ysize, xsize]
        elif ndim == 3:
            wcs_mock.wcs.crpix = crpix1, crpix2, crpix3
            wcs_mock.wcs.cdelt = cdelt1, cdelt2, cdelt3
            wcs_mock.wcs.cunit = 'deg', 'deg', 'um'
            wcs_mock.wcs.ctype = 'RA', 'DEC', 'Wavelength'
            wcs_mock.array_shape = [ndim, ysize, xsize]

def data_combine(image_list, data_ext='DATA', mask=None, mask_ext='DQI',  
                 pixel_shifts=None, ignore_wcs=False, 
                 sigma_clip=True, sigma=3.0, bgsub=True,
                 header_ext=None, weighting=None, frame=None, projection='TAN', 
                 resolution=None, savefile=None):
    """combine the multiple observation of the same target

    By default, the combined wcs uses the frame of the first image

    sigma_clip, apply if there are large number of frames to be combined
    background: global or chennel-per-channel and row-by-row
    Args:
        bgsub (bool): set to true to subtract global thermal background
        sigma_clip (bool): set to true to apply sigma_clip with the sigma controlled
                           by `sigma`
        sigma (bool): the deviation scale used to control the sigma_clip
    """
    # define the default variables
    nimages = len(image_list)
    if header_ext is None:
        header_ext = data_ext

    # check the input variables
    if pixel_shifts is not None:
        if len(pixel_shifts) != nimages:
            raise ValueError("Pixel_shift does not match the number of images!")
        pixel_shifts = np.array(pixel_shifts)

    if ignore_wcs:
        # ignore the wcs and mainly relies on the pixel_shifts to align the images
        # to make advantage of reproject, we still need a roughly correct wcs or 
        # a mock wcs
        # first try to extract basic information from the first image
        with fits.open(image_list[0]) as hdu:
            header = fix_micron_unit_header(hdu[header_ext].header)
            try:
                # if the header have a rougly correct wcs
                wcs_mock = WCS(header)
            except:
                wcs_mock = construct_wcs(header, data_shape=None)
        # shifting the mock wcs to generate a series of wcs
        wcs_list = []
        if pixel_shifts is not None:
            for i in range(nimages):
                wcs_tmp = wcs_mock.deepcopy()
                x_shift, y_shift = pixel_shifts[i]
                wcs_tmp.wcs.crpix += np.array([x_shift, y_shift, 0])
                wcs_list.append(wcs_tmp)
        else:
            wcs_list = [wcs_mock]*nimages
    else:
        wcs_list = []
        # looping through the image list to extract their wcs
        for i,fi in enumerate(image_list):
            with fits.open(fi) as hdu:
                # this is to fix the VLT micron header
                header = fix_micron_unit_header(hdu[header_ext].header)
                image_wcs = WCS(header)
                wcs_list.append(image_wcs)
    # compute the combined wcs 
    wcs_combined = find_combined_wcs(wcs_list=wcs_list, frame=frame, 
                                     resolution=resolution)
    shape_combined = wcs_combined.array_shape
    if len(shape_combined) == 3:
        nchan, size_y, size_x = shape_combined
    elif len(shape_combined) == 2:
        size_y, size_x = shape_combined

    # define the combined cube
    image_shape_combined = shape_combined[-2:]
    data_combined = np.full(shape_combined, fill_value=0.)
    coverage_combined = np.full(shape_combined, fill_value=1e-8)
    
    # handle the weighting
    if weighting is None:
        # treat each dataset equally
        weighting = np.full(nimages, fill_value=1./nimages)
    
    # reproject each image to the combined wcs
    for i in range(nimages):
        image_wcs = wcs_list[i].celestial
        data = fits.getdata(image_list[i], data_ext)
        if mask_ext is not None:
            mask = fits.getdata(image_list[i], mask_ext)
        else:
            mask = np.full(data.shape, fill_value=False)
        # reset the masked value to zero, to be removed from combination
        # <TODO>: find a better way to fix the masked pixels
        # now we just set it to zeros
        data_masked = np.ma.masked_array(data, mask=mask)

        if sigma_clip:
            data_masked = astro_stats.sigma_clip(
                    data_masked, sigma=sigma, maxiters=5, masked=True)
        if bgsub:
            data_masked = data_masked - np.ma.median(
                            data_masked, axis=(1,2))[:, np.newaxis, np.newaxis]
        data = data_masked.filled(0)
        mask = data_masked.mask
        # 
        data_reprojected, footprint = reproject_adaptive((data, image_wcs), 
                                                          wcs_combined.celestial, 
                                                          shape_out=shape_combined,
                                                          conserve_flux=True)
        mask_reprojected, footprint = reproject_adaptive((mask, image_wcs), 
                                                          wcs_combined.celestial, 
                                                          shape_out=shape_combined,
                                                          conserve_flux=False)
        data_combined += data_reprojected * weighting[i]
        footprint = footprint.astype(bool)
        coverage_combined += (1.-mask_reprojected)
        # error2_combined += error_reprojected**2 * weighting[i]
    # error_combined = np.sqrt(error2_combined)
    data_combined = data_combined / coverage_combined

    if savefile is not None:
        # save the combined data
        hdr = wcs_combined.to_header() 
        hdr['OBSERVER'] = 'MPE-IR'
        hdr['COMMENT'] = 'Combined by eris_jhchen_utils.py'
        # reset the cdelt
        header['CDELT1'] = header['PC1_1']
        header['CDELT2'] = header['PC2_2']
        header['CDELT3'] = header['PC3_3']
        primary_hdu = fits.PrimaryHDU(header=hdr)
        data_combined_hdu = fits.ImageHDU(data_combined, name="DATA", header=hdr)
        hdus = fits.HDUList([primary_hdu, data_combined_hdu])
        hdus.writeto(savefile, overwrite=True)
    else:
        return data_combined, error_combined

def data_combine_pixel(image_list, offsets=None, savefile=None):
    """deprecated, will be removed in the future

    this function combine the image/datacube in the pixel space

    image_list (list): the list of filenames or ndarray
    offset (list,ndarray): the offset (x, y) of each image
    """
    data_combined = 0.0
    coverage_combined = 1.0
    
    # quick test without offsets
    if offsets is None:
        for i,image in enumerate(image_list):
            header = fits.getheader(image, 'DATA')
            image_data = fits.getdata(image, 'DATA')
            image_mask = fits.getdata(image, 'DQI')
            image_data[(1.0-image_mask)<1e-6] = 0.0
            if i == 0:
                data_combined = np.zeros_like(image_data)
                coverage_combined = np.zeros_like(image_mask)
                header_combined = fits.getheader(image, 0)
            data_combined += image_data
            coverage_combined += (1 - image_mask)
    else:
        # calculate the combined image size
        # to make things simpler, here we still keep the x and y equal
        padding = np.max(offsets)
        for i,image in enumerate(image_list):
            header = fits.getheader(image, 'DATA')
            image_data = fits.getdata(image, 'DATA')
            image_mask = fits.getdata(image, 'DQI')
            if i==0:
                # skip the resampling
                ygrid, xgrid = np.mgrid(ny, nx) + padding
                pass
            else:
                # get the real pixel coordinate of the image
                ygrid, xgrid = np.mgrid(ny, nx) + padding + offsets[i]
                # get the nearest grid pixel
                ygrid2, xgrid2 = np.round(ygrid), np.round(xgrid)
                # resample the image to be aligned with the grid
                image_resampling(image, offset)
                scipy.interpolate.griddata(np.array(list(zip(ygrid, xgrid))), 
                                           image_data.ravel(), 
                                           np.array(list(zip(ygrid2, xgrid2))),
                                           method='linear',)

    data_combined = data_combined / coverage_combined

    if savefile is not None:
        # save the combined data
        hdr = header_combined 
        hdr['OBSERVER'] = ''
        hdr['COMMENT'] = ''
        primary_hdu = fits.PrimaryHDU(header=hdr)
        data_combined_hdu = fits.ImageHDU(data_combined, name="DATA", header=hdr)
        # error_combined_hdu = fits.ImageHDU(error_combined, name="ERROR", header=hdr)
        hdus = fits.HDUList([primary_hdu, data_combined_hdu])
        hdus.writeto(savefile, overwrite=True)
    else:
        return data_combined 

def read_eris_drifts(datfile, arcfilenames):
    """read eris drifting table
    """
    pixel_center = [32., 32.] # the expected center
    dat = table.read(datfile, format='csv')
    drifts = np.zeros(len(arcfilenames), 2) # drift in [x, y]
    for arcfile in arcfilenames:
        dat_img = dat[dat['ARCFILE'] == arcfile]
        model_center = [dat_img['x_model'], ]
        drifts[i] = dat_img['x_model']-pixel_center[0], dat_img['y_model']-pixel_center[1]
    return drifts

def compute_eris_offset(image_list, additional_drifts=None, header_ext='Primary',
                        header_ext_data='DATA',
                        ra_offset_header='HIERARCH ESO OCS CUMOFFS RA',
                        dec_offset_header='HIERARCH ESO OCS CUMOFFS DEC',
                        x_drift_colname='x_model', y_drift_colname='y_model',
                        coord_system='sky'):
    """compute the eris offset based on the telescope pointing

    This program will read the cumulative OCS offset from the header of each image,
    then it will compute the relative offset compare to the first image.
    The OCS offset is the expected offset from the telescope, but it may not always
    perform so accurately. If there is additional dirfts, it can also be accounted
    
    Args:
        image_list: the fits images, with the header include the offset information
        additional_drifts: (str, ndarray): the additional drift in pixels
    """
    # initialise the reference point
    nimage = len(image_list)
    array_offset = np.zeros((nimage, 2))
    arcfilenames = []
    # the followind code assuming the relative offset is small, so the sky offset has
    # been directly converted into pixel offset
    for i, img in enumerate(image_list):
        with fits.open(img) as hdu:
            header = hdu[header_ext].header
            data_header = hdu[header_ext_data].header
            arcfilenames.append(header['ARCFILE'])
            ra_offset = header[ra_offset_header]
            dec_offset = header[dec_offset_header]
            ra_diff = abs(data_header['CD1_1']*3600.)
            dec_diff = abs(data_header['CD2_2']*3600.)
            if i == 0: 
                ra_offset_0 = ra_offset
                dec_offset_0 = dec_offset
                # convert the skycoords to pixels use the first wcs
                if coord_system == 'sky':
                    image_wcs = WCS(header)
            array_offset[i][:] = (ra_offset-ra_offset_0)/ra_diff, (dec_offset-dec_offset_0)/dec_diff
    # consider additional offset
    if additional_drifts is not None:
        if isinstance(additional_drifts, str):
            additional_drifts = read_eris_drifts(additional_drifts, arcfilenames)
        for i in range(nimage):
            array_offset[i] += additional_drifts[i]
    return array_offset

def search_eris_files(dirname, pattern=''):
    matched_files = []
    # This will return absolute paths
    file_list = [f for f in glob.iglob(dirname.strip('/')+"/**", recursive=True) if os.path.isfile(f)]
    for ff in file_list:
        if re.search(pattern, os.path.basename(ff)):
            matched_files.append(ff)
    return matched_files

def combine_eris_ifu(image_list=None, dirname=None, pattern='', drifts_file=None, outfile=None, **kwargs):
    if dirname is not None:
        image_list = search_eris_files(dirname, pattern)
    weighting = compute_weighting_eris(image_list)
    drifts = compute_eris_offset(image_list, additional_drifts=drifts_file)
    data_combine(image_list, weighting=weighting, pixel_shifts=drifts, savefile=outfile, **kwargs)

#####################################
########## CMD wrapprt ##############


def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.info('Started')
    pass
    logging.info('Finished')

# add main here to run the script from cmd
import argparse

if __name__ == '__main__':


    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='eris_jhchen_utils.py',
            description="Welcome to jhchen's ERIS utilities",
            epilog='Reports bugs and problems to jhchen@mpe.mpg.de')
    parser.add_argument('--esorex', nargs='?', type=str, default='esorex',
                        help='specify the customed esorex')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')

    # add subparsers
    subparsers = parser.add_subparsers(title='Available task', dest='task', 
                                       metavar=textwrap.dedent(
        '''
          * request_calib: search and download the raw calibration files
          * request_science: download the science data
          * generate_calib: generate the calibration files
          * auto_gitter: run gitter recipe automatically
          * data_combine: combine the reduced data
          
          To get more details about each task:
          $ eris_jhchen_utils.py task_name --help
        '''))


    # request_eris_calib
    subp_request_calib = subparsers.add_parser('request_calib',
            description='Search and download the required calib files')
    subp_request_calib.add_argument('--start_date', type=str, help='The starting date of the observation, e.g. 2023-03-08')
    subp_request_calib.add_argument('--end_date', type=str, help='The finishing date of the observation, e.g. 2023-03-08')
    subp_request_calib.add_argument('--steps', type=str, nargs='+', 
        help="Calibration steps, can be combination of: 'dark','detlin','distortion','flat','wavecal'",
                                     default=['dark','detlin','distortion','flat','wavecal'])
    subp_request_calib.add_argument('--band', type=str, help='Observing band')
    subp_request_calib.add_argument('--exptime', type=int, help='Exposure time')
    subp_request_calib.add_argument('--resolution', type=str, help='Pixel resolution')
    subp_request_calib.add_argument('--outdir', type=str, help='Output directory',
                                    default='raw')
    subp_request_calib.add_argument('--metafile', type=str, help='Summary file')
    subp_request_calib.add_argument('--max_days', type=int, help='Maximum searching days before and after the observing day.', default=30)
    
    # request_science
    subp_request_science = subparsers.add_parser('request_science',
            description='Search and download the required calib files')
    subp_request_science.add_argument('--start_date', type=str, help='The starting date of the observation. Such as 2023-03-08')
    subp_request_science.add_argument('--band', type=str, help='Observing band')
    subp_request_science.add_argument('--resolution', type=str, help='Pixel resolution')
    subp_request_science.add_argument('--user', type=str, help='The user name in ESO User Eortal.')
    subp_request_science.add_argument('--outdir', type=str, help='Output directory')
    subp_request_science.add_argument('--program_id', type=str, help='Program ID')
    subp_request_science.add_argument('--observation_id', type=str, help='Observation ID')
    subp_request_science.add_argument('--metafile', type=str, help='Summary file')
    
    # generate_calib
    subp_generate_calib = subparsers.add_parser('generate_calib',
            description='Generate the required calibration files')
    subp_generate_calib.add_argument('metafile', type=str, help='The summary file')
    subp_generate_calib.add_argument('--raw_pool', type=str, help='The directory includes the raw files')
    subp_generate_calib.add_argument('--calib_pool', type=str, help='The output directory',
                                     default='./calibPool')
    subp_generate_calib.add_argument('--static_pool', type=str, help='The static pool')
    subp_generate_calib.add_argument('--steps', type=str, nargs='+', 
        help="Calibration steps, can be combination of: 'dark','detlin','distortion','flat','wavecal'",
                                     default=['dark','detlin','distortion','flat','wavecal'])
    subp_generate_calib.add_argument('--dark_sof', help='dark sof')
    subp_generate_calib.add_argument('--detlin_sof', help='detector linearity sof')
    subp_generate_calib.add_argument('--distortion_sof', help='distortion sof')
    subp_generate_calib.add_argument('--flat_sof', help='flat sof')
    subp_generate_calib.add_argument('--wavecal_sof', help='wavecal sof')
    subp_generate_calib.add_argument('--archive', action='store_true', help='Turn on archive mode')
    subp_generate_calib.add_argument('--archive_name', help='Archive name')

    # auto_gitter
    subp_auto_gitter = subparsers.add_parser('auto_gitter',
            description='Automatically run the gitter recipe')

    # combine data
    subp_data_combine = subparsers.add_parser('data_combine',
            description='Combine reduced datacubes')

    args = parser.parse_args()
    if args.debug:
        print(args)
        func_args = list(inspect.signature(request_eris_calib).parameters.keys())
        func_str = f"Executing: {args.task}("
        for ag in func_args:
            try: func_str += f"{ag}={args.__dict__[ag]},"
            except: func_str += f"{ag}=None, "
        func_str += ')'
        print(func_str)


    if args.task == 'request_calib':
        request_eris_calib(start_date=args.start_date, band=args.band, resolution=args.resolution, 
                           exptime=args.exptime, end_date=args.end_date, outdir=args.outdir, 
                           metafile=args.metafile, max_days=args.max_days)

    elif args.task == 'request_science':
        request_eris_science(program_id=args.program_id, 
                             observation_id=args.observation_id,
                             start_date=args.start_date, 
                             band=args.band, resolution=args.resolution, 
                             exptime=args.exptime, end_date=args.end_date, 
                             outdir=args.outdir, metafile=args.metafile)

    elif args.task == 'generate_calib':
        generate_calib(args.metafile, raw_pool=args.raw_pool, calib_pool=args.calib_pool,
                       static_pool=args.static_pool, steps=args.steps, 
                       dark_sof=args.dark_sof, detlin_sof=args.detlin_sof, distortion_sof=args.distortion_sof,
                       flat_sof=args.flat_sof, wavecal_sof=args.wavecal_sof, 
                       archive=args.archive, archive_name=args.archive_name,
                       esorex=args.esorex, dry_run=args.dry_run, debug=args.debug)
