#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

This program was initially written when I learnt how to analysis the 
ESO/ERIS data for the first time. 

History:
    - 2023-11-22: first release, v0.1
    - 2024-01-04: add cmd interface, v0.2
    - 2024-01-11: add quick tools, v0.3
    - 2024-01-30: add customized combining task, v0.4
    - 2024-02-26: support reducing PSF and standard stars, v0.5
    - 2024-04-07: add data quicklook tools, v0.6
"""
__version__ = '0.6.2'

# import the standard libraries
import os 
import tempfile
import textwrap
import inspect
import shutil
import re
import datetime
import logging
import getpass
import glob
import warnings
import subprocess
import argparse

# external but required libraries
import numpy as np
import scipy
from scipy import ndimage, optimize
import astropy.table as table
import astropy.units as units
import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from astropy import stats as astro_stats
from astropy.modeling import models
from astropy.convolution import convolve, Gaussian2DKernel
from astroquery.eso import Eso
from astroquery.eso.core import NoResultsWarning
from photutils.aperture import EllipticalAperture, RectangularAperture
import requests 
# for plot
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# filtering the warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

#####################################
######### DATA Retrieval ############

def download_file(url, filename=None, outdir='./', auth=None, debug=False): 
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
        subprocess.run(['mkdir', '-p', outdir])
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
                    if debug:
                        logging.info(f"{filename} is already downloaded.")
                    is_downloaded = True
                else:
                    logging.warning('Find local inconsistent file, overwriting...')
            except:
                logging.warning(f'Overwriting {filename_fullpath}')
        if not is_downloaded:
            with open(filename_fullpath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

def read_metadata(metadata):
    """United way to read metadata
    """
    if isinstance(metadata, str):
        meta_tab = table.Table.read(metadata, format='csv')
    elif isinstance(metadata, table.Table):
        meta_tab = metadata
    else:
        print(metadata)
        raise ValueError(f'Unsupported file type of metadata: {type(metadata)}')
    try: meta_tab.sort(['Release Date'])
    except: pass
    return meta_tab

def save_metadata(metadata, metafile='metadata.csv'):
    """United way to save metadata
    """
    try:
        if len(metadata) > 0:
            if os.path.isfile(metafile):
                subprocess.run(['mv', metafile, metafile+'.bak'])
            else:
                if '/' in metafile:
                    subprocess.run(['mkdir', '-p', os.path.dirname(metafile)])
            metadata.sort(['Release Date'])
            metadata.write(metafile, format='csv')
    except:
        raise ValueError('Unsupported metadata!')

def download_eris(eris_query_tab, outdir='raw', metafile=None, username=None, auth=None):
    """download the calib files of eris (wrapper of download_file)

    Args:
        eris_query_tab (astropy.table): the query table returned by astroquery.eso
        outdir (str): the directory to store the download files and saved meta table
        metafile (str): the filename of the saved tabe from eris_query_tab
        save_columns (list): the selected column names to be saved.
                             set the 'None' to save all the columns
        username (str): the username from ESO portal
        auth (str): the existing authentication
    """
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    if auth is None:
        if username is not None:
            passwd = getpass.getpass(f'{username} enter your password:\n')
            auth = requests.auth.HTTPBasicAuth(username, passwd)
    for fileid in eris_query_tab['DP.ID']:
        file_url = root_calib_url+fileid
        download_file(file_url, outdir=outdir, auth=auth)
    if metafile is not None:
        save_metadata(eris_query_tab, metafile=metafile)

def eris_auto_quary(start_date, end_date=None, start_time=12, end_time=12, max_days=40, 
                    column_filters={}, dry_run=False, debug=False, **kwargs):
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
    eso.ROW_LIMIT = -1 # remove the row limit of eso.query_instrument
    sdatetime = datetime.datetime.strptime(f'{start_date} {start_time:0>2d}', '%Y-%m-%d %H')
    if end_date is not None:
        edatetime = datetime.datetime.strptime(f'{end_date} {end_time:0>2d}', '%Y-%m-%d %H')
        sdatetime = (edatetime - sdatetime)/2 + sdatetime
    delta_oneday = datetime.timedelta(days=1)
    matched = 0
    for i in range(0, max_days):
        # t_start = (sdatetime - 0.5*datetime.timedelta(days=i))
        # t_end = (sdatetime + 0.5*datetime.timedelta(days=i))
        for j in [-1, 1]:
            if matched == 0:
                if j == -1:
                    t_start = (sdatetime - datetime.timedelta(days=i))
                    t_end = sdatetime + delta_oneday
                elif j == 1:
                    t_start = sdatetime
                    t_end = (sdatetime + datetime.timedelta(days=i))
                column_filters['stime'] = t_start.strftime('%Y-%m-%d')
                column_filters['etime'] = t_end.strftime('%Y-%m-%d')
                column_filters['starttime'] = t_start.strftime('%H')
                column_filters['endtime'] = t_end.strftime('%H')
                warnings.simplefilter('ignore', category=NoResultsWarning)
                tab_eris = eso.query_instrument('eris', column_filters=column_filters)
                if tab_eris is not None:
                    matched = 1
                    # for some addition validity check
                    dptype = column_filters['dp_type']
                    if 'FLAT' in dptype:
                        # make sure both dark and lamp are present
                        n_flat_dark = np.sum(tab_eris['DPR.TYPE'] == 'FLAT,DARK')
                        n_flat_lamp = np.sum(tab_eris['DPR.TYPE'] == 'FLAT,LAMP')
                        if (n_flat_dark<1) or (n_flat_dark!=n_flat_lamp):
                            matched = 0
                    if 'NS' in dptype:
                        n_ds_fiber = np.sum(tab_eris['DPR.TYPE'] == 'NS,SLIT')
                        n_ds_fiber_dark = np.sum(tab_eris['DPR.TYPE'] == 'NS,FLAT,DARK')
                        n_ds_fiber_lamp = np.sum(tab_eris['DPR.TYPE'] == 'NS,FLAT,LAMP')
                        n_ds_wave_dark = np.sum(tab_eris['DPR.TYPE'] == 'NS,WAVE,DARK')
                        n_ds_wave_lamp = np.sum(tab_eris['DPR.TYPE'] == 'NS,WAVE,LAMP')
                        # print("NS", n_ds_fiber, n_ds_fiber_dark, n_ds_fiber_lamp, n_ds_wave_dark, n_ds_wave_lamp)
                        if (n_ds_fiber!=1) or (n_ds_fiber_lamp!=1) or (n_ds_fiber_lamp!=n_ds_fiber_dark) or (n_ds_wave_lamp<1) or (n_ds_wave_dark<1):
                            matched = 0
                    if 'WAVE' in dptype:
                        n_wave_dark = np.sum(tab_eris['DPR.TYPE'] == 'WAVE,DARK')
                        n_wave_lamp = np.sum(tab_eris['DPR.TYPE'] == 'WAVE,LAMP')
                        # print("WAVE", n_wave_dark, n_wave_lamp)
                        if (n_wave_dark<1) or (n_wave_lamp<1) or ((n_wave_dark+n_wave_lamp)>6):
                            matched = 0

    if matched == 0:
        # print("Cannot find proper calib files, please check your input")
        logging.warning("eris_auto_quary: cannot find proper calib files!")
    else:
        if dry_run:
            print(column_filters)
        return tab_eris

def request_calib(start_date=None, band=None, spaxel=None, exptime=None, 
                  outdir='raw', end_date=None, dpcat='CALIB', arm='SPIFFIER', 
                  metafile=None, max_days=60,
                  steps=['dark','detlin','distortion','flat','wavecal'],
                  dry_run=False, debug=False, **kwargs):
    """a general purpose to qeury calib files of ERIS/SPIFFIER observation
    
    Args:
        start_date (str): ISO date format, like: 2023-04-08
        end_date (str, None): same format as start_date
        band (str): grating configurations
        spaxel (str): the spaxel size of the plate, [250mas, 100mas, 25mas]
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
            psfstar      %PSF-CALIBRATOR   the psfstar,usually packed with the science
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
    
    if debug:
        print("Input parameters:")
        print(f'steps: {steps}')
        print(f'spaxel: {spaxel}')
        print(f'band: {band}')
        print(f'exptime: {exptime}')
        print(f'max_days: {max_days}')
    for step in steps:
        logging.info(f'Requesting {step} calibration files')
        column_filters = {'dp_cat': dpcat,
                          'seq_arm': arm,
                          'dp_type': dptype_dict[step]}
        if step == 'dark':
            # drop the requirement for band and spaxel
            column_filters['exptime'] = exptime
        if step in ['distortion', 'flat', 'wavecal', 'stdstar']:
            column_filters['ins3_spgw_name'] = band
            column_filters['ins3_spxw_name'] = spaxel

        step_query = eris_auto_quary(start_date, end_date=end_date, column_filters=column_filters,
                                     dry_run=dry_run, debug=debug, **kwargs)
        # fix the data type issue of masked table columns
        if step_query is not None:
            for col in step_query.colnames:
                step_query[col] = step_query[col].astype(str)
            query_tabs.append(step_query)
        else:
            # raise ValueError('Failed in requesting the calib files! Consider release the `max_days`?')
            logging.warning(f"request_calib: no calib file found for '{step}', consider to relax the `max_days`?'")
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

def request_science(prog_id='', metafile='metadata.csv',
                    username=None, password=None, auth=None, 
                    outdir=None, target='', ob_id='', exptime='',
                    start_date='2022-04-01', end_date='', debug=False, 
                    dp_type='',
                    dry_run=False, archive=False, uncompress=False,
                    **kwargs):
    """download the science data 

    To download the proprietory data, you need to provide your eso username
    and you will be asked to input your password.

    If the requested data has been observed across multiple days, they will
    be organised within each folders names of the observing dates.

    Args:
        prog_id (str): program id
        username (str): the user name of your eso account
        metafile (str): the output file to store all the meta data
                        default: metadata.csv
        target (str): the target name
        outdir (str): the directory to save all the raw files
        ob_id (str, int): the id of the observation
        start_date (str): starting data, in the format of '2023-04-08'
        end_date (str): end date, same format as start_date
        dp_type (str): the type of the calib, set it to "%PSF-CALIBRATOR" 
                       along with the project id to download the PSF star only
        **kwargs: other keyword filters
    """
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    if outdir is None:
        if prog_id is not None: outdir = prog_id+'_raw'
        else: outdir = 'raw'
    if not os.path.isdir(outdir):
        if not dry_run:
            subprocess.run(['mkdir', '-p', outdir])
    logging.info(f'Requesting the data from project: {prog_id}')
    eso = Eso()
    eso.ROW_LIMIT = -1
    # if end_date is '' and start_date != '':
        # sdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # edate = sdate + datetime.timedelta(days=1)
        # end_date = edate.strftime('%Y-%m-%d')
    column_filters={'ob_id': ob_id,
                    'prog_id':prog_id,
                    'stime': start_date,
                    'exptime': exptime,
                    'etime': end_date,
                    'dp_type': dp_type, 
                    'obs_targ_name': target,}
    if debug:
        for key, item in column_filters.items():
            print(f"{key}: {item}")
    eris_query_tab = eso.query_instrument('eris', column_filters=column_filters)
    eris_query_tab = eris_query_tab[eris_query_tab['DPR.TYPE'] != 'PERSISTENCE']
    # remove the persistence observations
    n_item = len(eris_query_tab)
    if n_item < 1:
        logging.warning("No science data has been found!")
        return
    
    if dry_run:
        return eris_query_tab

    # to avaid type the password many time, we save the auth here
    if auth is None:
        if username is not None:
            if password is None:
                password = getpass.getpass(f'{username} enter your password:\n')
        auth = requests.auth.HTTPBasicAuth(username, password)

    if archive:
        # check the observing dates
        dates_list = np.full(n_item, fill_value='', dtype='U32')
        # datetime_matcher = re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})')
        tplstarts = eris_query_tab['TPL.START']
        for i in range(n_item):
            i_start = datetime.datetime.fromisoformat(tplstarts[i])
            if i_start.hour < 12:
                dates_list[i] = i_start.strftime('%Y-%m-%d')
            else:
                dates_list[i] = (i_start + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        unique_dates = np.unique(dates_list)
        logging.info(f"Find data in dates: {unique_dates}")
        for date in unique_dates:
            daily_selection = (dates_list == date)
            daily_outdir = os.path.join(outdir, date)
            daily_metafile = os.path.join(daily_outdir, 'metadata.csv')
            if dry_run:
                print(f"On {date}: {np.sum(daily_selection)} files will be download")
            else:
                logging.info(f'Downloading science data of {date}...')
                download_eris(eris_query_tab[daily_selection], auth=auth, 
                              outdir=daily_outdir, metafile=daily_metafile)
    else:
        download_eris(eris_query_tab, auth=auth, outdir=outdir, metafile=metafile)

    if uncompress:
        os.system(f'uncompress {outdir}/*.Z')

def generate_metadata(data_dir=None, header_dir=None, metafile='metadata.csv', 
                      extname='PRIMARY', work_dir=None, clean_work_dir=False,
                      dry_run=False, debug=False, overwrite=False):
    """generate metafile from download files

    Args:
        data_dir (str): the directory include the fits file
    """
    # colnames
    colnames = ['Release Date', 'Object', 'RA', 'DEC','Program ID', 'DP.ID', 'EXPTIME', 
                'OB.ID', 'OBS.TARG.NAME', 'DPR.CATG', 'DPR.TYPE', 'DPR.TECH', 'TPL.START', 
                'SEQ.ARM', 'DET.SEQ1.DIT', 'INS3.SPGW.NAME', 'INS3.SPXW.NAME','ARCFILE']
    colnames_header = ['DATE', 'OBJECT', 'RA', 'DEC', 'HIERARCH ESO OBS PROG ID', 
                       'ARCFILE', 'EXPTIME',
                       'HIERARCH ESO OBS ID', 'HIERARCH ESO OBS TARG NAME',  
                       'HIERARCH ESO DPR CATG', 'HIERARCH ESO DPR TYPE', 
                       'HIERARCH ESO DPR TECH', 'HIERARCH ESO TPL START', 
                       'HIERARCH ESO SEQ ARM', 'HIERARCH ESO DET SEQ1 DIT',
                       'HIERARCH ESO INS3 SPGW NAME', 'HIERARCH ESO INS3 SPXW NAME', 'ARCFILE']
    # check the exiting metafile
    if metafile is not None:
        if os.path.isfile(metafile):
            if not overwrite:
                if debug: print("Exiting file found, skip it...")
                return
        else:
            subprocess.run(['mkdir','-p', os.path.dirname(metafile)])
    if data_dir is not None:
        data_dir = data_dir.strip('/')
        fits_Zfiles = glob.glob(data_dir+'/*.fits.Z')
        fits_files = glob.glob(data_dir+'/*.fits')
        if work_dir is None:
            work_dir = '.tmp_generate_metadata'
            clean_work_dir = True

        dir_uncompressed = os.path.join(work_dir, 'uncompressed')
        dir_header = os.path.join(work_dir, 'headers')
        for d in [dir_uncompressed, dir_header]:
            if not os.path.isdir(d):
                subprocess.run(['mkdir', '-p', d])
        # compress the fits file
        if len(fits_Zfiles) > 0:
            for ff in fits_Zfiles:
                subprocess.run(['cp', ff, dir_uncompressed])
            # subprocess.run(['uncompress', dir_uncompressed+'/*.Z'])
            os.system(f'uncompress {dir_uncompressed}/*.Z')
            fits_Zfiles = glob.glob(f'{dir_uncompressed}/*.fits')
        
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
                        try: header_values.append(str(header[cn]).removesuffix('.fits'))
                        except: header_values.append('')
                    meta_tab.add_row(header_values)
    elif header_dir is not None:
        fits_headers = glob.glob(header_dir+'/*.hdr')
        meta_tab = table.Table(names=colnames, dtype=['U32']*len(colnames))
        if len(fits_headers) > 0:
            for fh in fits_headers:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyWarning)
                    header = fits.header.Header.fromtextfile(fh)
                    header_values = []
                    for cn in colnames_header:
                        try: header_values.append(str(header[cn]).removesuffix('.fits'))
                        except: header_values.append('')
                    meta_tab.add_row(header_values)
    # clean working directory
    if clean_work_dir:
        subprocess.run(['rm', '-rf', work_dir])
    # save the header metadata
    if dry_run:
        print(f"Read {len(meta_tab)} files, no problem found.")
    if debug:
        print('The metadata table:')
        print(meta_tab)
    if metafile is not None:
        save_metadata(meta_tab, metafile=metafile)
        return metafile
    else:
        return meta_tab


#####################################
######### DATA Calibration ##########

def search_static_calib(esorex=None, debug=False):
    """try to automatically find the static calibration files

    There are various locations this program will search for static files
    1. the current folder: ./
    2. the esorex installation directory: /path/to/esorex/calib/eris-*.*.*
    3. the manually installed eris-pipeline dierectory: 
       like esorex in /usr/local/bin/esorex, then it will search calib 
       in /usr/local/calib/ifs

    The priority deceases as the order
    """
    static_pool = None
    logging.info("Search for staticPool...")
    # first search local static calib named staticPool
    cwd = os.getcwd()
    if debug:
        logging.info(f'Current directory is {cwd}')
    if 'staticPool' in os.listdir(cwd):
        static_pool = os.path.join(cwd, 'staticPool')
    if 'STATIC_CALIB' in os.environ:
        static_pool = os.environ['STATIC_CALIB']
    if static_pool is None:
        if ' ' in esorex:
            esorex = esorex.split(' ')[0]
        if (esorex is None) or (esorex=='esorex'):
            esorex = shutil.which('esorex')
        if '/' in esorex:
            try:
                binpath_match = re.compile('(?P<install_dir>^[\/\w\s\_\.\-]*)/bin/esorex')
                install_dir = binpath_match.search(esorex).groupdict()['install_dir']
            except:
                logging.info(f'Failed to locate the install direction of esorex from {esorex}!')
        else:
            install_dir = '/usr/local'
        try:
            # search static calib names withed the pipeline version number
            static_pool_list = glob.glob(os.path.join(install_dir, 'calib/eris-*'))
            static_pool = sorted(static_pool_list)[-1] # choose the latest one
        except:
            if 'ifs' in os.listdir(install_dir+'/calib'):
                static_pool = os.path.join(install_dir, 'calib/ifs')
            else:
                static_pool = None
        if debug:
            logging.info(f'installing directory is {install_dir}')
    if static_pool is not None:
        logging.info(f'Found static_pool as {static_pool}')
        # check whether it is a valid folder
        static_calib_files = os.listdir(static_pool)
        for ff in ['eris_oh_spec.fits', 'eris_ifu_wave_setup.fits', 
                   'eris_ifu_first_fit.fits']:
            if ff not in static_calib_files:
                logging.warning('Found staticPool in {}, but it seems incomplete!')
    else:
        logging.info('Failed in locating any valid staticPool!')
    return static_pool

def generate_calib(metadata, raw_pool='./raw', outdir='./', static_pool=None,
                   steps=['dark','detlin','distortion','flat','wavecal'],
                   dark_sof=None, detlin_sof=None, distortion_sof=None, flat_sof=None, 
                   wavecal_sof=None, stdstar_sof=None, drp_type_colname='DPR.TYPE', 
                   fits_suffix='fits.Z', esorex=None, dry_run=False,
                   debug=False):
    """generate the science of frame of each calibration step

    Args:
        metafile (str): the metadata file where you can find the path and objects of each dataset
        raw_pool (str): the raw data pool
        static_pool (str): the static data pool
        outdir (str): the directory to keep all the output files from esorex
        steps (list): the steps to generate the corresponding calibration files
                      it could include: ['dark','detlin','distortion','flat','wavecal','stdstar']
        drp_type_colname (str): the column name of the metadata table includes the DPR types. 
        esorex (str): the callable esorex command from the terminal
        dry_run (str): set it to True to just generate the sof files
    """
    esorex_cmd_list = esorex.split(' ')
    cwd = os.getcwd()
    outdir = outdir.rstrip('/')
    raw_pool = raw_pool.rstrip('/')
    if static_pool is None:
        # use the default staticPool
        static_pool = search_static_calib(esorex)
        if static_pool is None:
            raise ValueError("Cannot find static calibration files!")
    if not os.path.isdir(outdir):
        subprocess.run(['mkdir', '-p', outdir])

    meta_tab = read_metadata(metadata)
    try: meta_tab.sort(['Release Date']); 
    except: pass
    
    # the outdir is also the calib_pool for the subsequent recipes
    calib_pool = outdir

    if 'dark' in steps:
        if dark_sof is None:
            # generate the sof for dark calibration
            dark_sof = os.path.join(calib_pool, 'dark.sof')
        if not os.path.isfile(dark_sof):
            with open(dark_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} DARK\n")
                # read the timestamp and exptime
                timestamp = item['Release Date']
                exptime = item['DET.SEQ1.DIT']
                openf.write(f'# dark: date={timestamp} exptime={exptime}\n')
        if not dry_run:
            subprocess.run([*esorex_cmd_list, f'--output-dir={calib_pool}', 
                            'eris_ifu_dark', dark_sof])

    if 'detlin' in steps:
        if detlin_sof is None:
            # generate the sof for detector's linarity
            detlin_sof = os.path.join(calib_pool, 'detlin.sof')
        if not os.path.isfile(detlin_sof):
            with open(detlin_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'LINEARITY,DARK,DETCHAR']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} LINEARITY_LAMP\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'LINEARITY,LAMP,DETCHAR']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} LINEARITY_LAMP\n")
                # read the timestamp
                timestamp = item['Release Date']
                openf.write(f'# detlin: date={timestamp}\n')
        if dry_run:
            print(f"{esorex_cmd} eris_ifu_detlin {detlin_sof}")
        else:
            subprocess.run([*esorex_cmd_list, f'--output-dir={calib_pool}', 
                            'eris_ifu_detlin', detlin_sof])

    if 'distortion' in steps:
        if distortion_sof is None:
            # generate the sof for distortion
            distortion_sof = os.path.join(calib_pool, 'distortion.sof')
        if not os.path.isfile(distortion_sof):
            with open(distortion_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} DARK_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,SLIT']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} FIBRE_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,WAVE,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} WAVE_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,WAVE,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} WAVE_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,FLAT,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} FLAT_NS\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'NS,FLAT,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} FLAT_NS\n")
                openf.write(f"{static_pool}/eris_ifu_first_fit.fits FIRST_WAVE_FIT\n") 
                openf.write(f"{static_pool}/eris_ifu_ref_lines.fits REF_LINE_ARC\n")
                openf.write(f"{static_pool}/eris_ifu_wave_setup.fits WAVE_SETUP\n")
                # read the timestamp, band, and spaxel
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                spaxel = item['INS3.SPXW.NAME']
                openf.write(f'# distortion: date={timestamp} band={band} spaxel={spaxel}\n')
        if not dry_run:
            subprocess.run([*esorex_cmd_list, f'--output-dir={calib_pool}', 
                            'eris_ifu_distortion', distortion_sof])

    if 'flat' in steps:
        if flat_sof is None:
            # generate the sof for flat
            flat_sof = os.path.join(calib_pool, 'flat.sof')
        if not os.path.isfile(flat_sof):
            with open(flat_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'FLAT,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} FLAT_LAMP\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'FLAT,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} FLAT_LAMP\n")
                openf.write(f"{calib_pool}/eris_ifu_dark_bpm.fits BPM_DARK\n")
                openf.write(f"{calib_pool}/eris_ifu_detlin_bpm_filt.fits BPM_DETLIN\n")
                openf.write(f"{calib_pool}/eris_ifu_distortion_bpm.fits BPM_DIST\n")
                # read the timestamp, band, and spaxel
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                spaxel = item['INS3.SPXW.NAME']
                openf.write(f'# flat: date={timestamp} band={band} spaxel={spaxel}\n')
        if not dry_run:
            subprocess.run([*esorex_cmd_list, f'--output-dir={calib_pool}', 
                            'eris_ifu_flat', flat_sof])

    if 'wavecal' in steps:
        if wavecal_sof is None:
            # generate the sof for wavecal
            wavecal_sof = os.path.join(calib_pool, 'wavecal.sof')
        if not os.path.isfile(wavecal_sof):
            with open(wavecal_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'WAVE,DARK']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} WAVE_LAMP\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'WAVE,LAMP']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} WAVE_LAMP\n")
                openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
                openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
                openf.write(f"{calib_pool}/eris_ifu_flat_bpm.fits BPM_FLAT\n")
                openf.write(f"{static_pool}/eris_ifu_ref_lines.fits REF_LINE_ARC\n")
                openf.write(f"{static_pool}/eris_ifu_wave_setup.fits WAVE_SETUP\n") 
                openf.write(f"{static_pool}/eris_ifu_first_fit.fits FIRST_WAVE_FIT\n") 
                # read the timestamp, band, and spaxel
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                spaxel = item['INS3.SPXW.NAME']
                openf.write(f'# wavecal: date={timestamp} band={band} spaxel={spaxel}\n')
        if not dry_run:
            subprocess.run([*esorex_cmd_list, f'--output-dir={calib_pool}', 
                            'eris_ifu_wavecal', wavecal_sof])

    if 'stdstar' in steps:
        if stdstar_sof is None:
            # generate the sof for wavecal
            stdstar_sof = os.path.join(calib_pool, 'stdstar.sof')
        if not os.path.isfile(stdstar_sof):
            with open(stdstar_sof, 'w+') as openf:
                for item in meta_tab[meta_tab[drp_type_colname] == 'STD']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} STD\n")
                for item in meta_tab[meta_tab[drp_type_colname] == 'SKY,STD']:
                    openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} SKY_STD\n")
                openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
                openf.write(f"{calib_pool}/eris_ifu_wave_map.fits WAVE_MAP\n")
                openf.write(f"{calib_pool}/eris_ifu_distortion_slitlet_pos.fits SLITLET_POS\n")
                openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
                openf.write(f"{calib_pool}/eris_ifu_dark_master_dark.fits MASTER_DARK\n")
                openf.write(f"{static_pool}/EXTCOEFF_TABLE.fits EXTCOEFF_TABLE\n")
                openf.write(f"{static_pool}/eris_oh_spec.fits OH_SPEC\n")
                # openf.write(f"{static_pool}/FLUX_STD_CATALOG.fits FLUX_STD_CATALOG\n")
                # openf.write(f"{static_pool}/TELL_MOD_CATALOG.fits TELL_MOD_CATALOG\n")
                # openf.write(f"{static_pool}/RESP_FIT_POINTS_CATALOG.fits RESP_FIT_POINTS_CATALOG\n")
                # if band in ['H_low', 'J_low', 'K_low']:
                    # openf.write(f"{static_pool}/RESPONSE_WINDOWS_{band}.fits RESPONSE\n")
                    # openf.write(f"{static_pool}/EFFICIENCY_WINDOWS_{band}.fits EFFICIENCY_WINDOWS\n")
                    # openf.write(f"{static_pool}/HIGH_ABS_REGIONS_{band}.fits HIGH_ABS_REGIONS\n")
                    # openf.write(f"{static_pool}/FIT_AREAS_{band}.fits FIT_AREAS\n")
                    # openf.write(f"{static_pool}/QUALITY_AREAS_{band}.fits QUALITY_AREAS\n")
                # read the timestamp, band, and spaxel
                timestamp = item['Release Date']
                band = item['INS3.SPGW.NAME']
                spaxel = item['INS3.SPXW.NAME']
                openf.write(f'# stdstar: date={timestamp} band={band} spaxel={spaxel}\n')
        if not dry_run:
            subprocess.run([*esorex_cmd_list, f'--output-dir={calib_pool}', 
                            'eris_ifu_stdstar', stdstar_sof])
 
def auto_jitter(metadata=None, raw_pool=None, outdir='./', calib_pool='calibPool', 
                sof=None, fits_suffix='fits.Z',
                static_pool=None, esorex='esorex', stdstar_type='PSF',
                objname=None, band=None, spaxel=None, exptime=None, 
                dpr_tech='IFU', dpr_catg='SCIENCE', prog_id=None, ob_id=None,
                sky_tweak=1, product_depth=2,
                dry_run=False, debug=False):
    """calibrate the science target or the standard stars


    dpr_catg: the category of the science exposures, it can be:
               "SCIENCE": reduce the science data
               "ACQUISITION": reduce the acquisition data
               "PSF-CALIBRATOR": reduce the psf star
    stdstar_type: the type of the standard star, it can be PSF or STD
    """
    calib_pool = calib_pool.rstrip('/')
    esorex_cmd_list = esorex.split(' ')
    jitter_sky_option = []

    if sof is not None:
        auto_jitter_sof = sof
    else:
        if metadata is None:
            if raw_pool is not None:
                metadata = generate_metadata(raw_pool)
            else:
                raise ValueError('Please give the metadata or the raw_pool!')
        meta_tab = read_metadata(metadata)
        
        if not os.path.isdir(outdir):
            subprocess.run(['mkdir','-p',outdir])
        
        if static_pool is None:
            # use the default staticPool
            static_pool = search_static_calib(esorex)
        else:
            static_pool = static_pool.rstrip('/')
        
        # apply the selections
        if ob_id is not None:
            meta_tab = meta_tab[meta_tab['OB.ID'].astype(type(ob_id)) == ob_id]
        if objname is not None:
            meta_tab = meta_tab[meta_tab['Object'] == objname]
        if band is not None:
            meta_tab = meta_tab[meta_tab['INS3.SPGW.NAME'] == band]
        if spaxel is not None:
            meta_tab = meta_tab[meta_tab['INS3.SPXW.NAME'] == spaxel]
        if exptime is not None:
            meta_tab = meta_tab[(meta_tab['DET.SEQ1.DIT']-exptime)<1e-6]
        if prog_id is not None:
            meta_tab = meta_tab[meta_tab['Program ID'] == prog_id]
        if dpr_tech is not None:
            dpr_tech_select = [True if dpr_tech in item['DPR.TECH'] else False for item in meta_tab]
            meta_tab = meta_tab[dpr_tech_select]
        if dpr_catg is not None:
            meta_tab = meta_tab[meta_tab['DPR.CATG'] == dpr_catg]

        if len(meta_tab) < 1:
            print(" >> skipped, no valid data found!")
            logging.warning(f"skipped {objname}({ob_id}) with {band}+{spaxel}+{exptime}, no valid data")
            return

        if dpr_catg == 'SCIENCE' or dpr_catg == 'ACQUISITION':
            auto_jitter_sof = os.path.join(outdir, f'{dpr_catg}_jitter.sof')
            with open(auto_jitter_sof, 'w+') as openf:
                # write OBJ
                n_obj = 0
                n_sky = 0
                for item in meta_tab[meta_tab['DPR.CATG'] == dpr_catg]:
                    if item['DPR.TYPE'] == 'OBJECT':
                        openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} OBJ\n")
                        n_obj += 1
                    elif item['DPR.TYPE'] == 'SKY':
                        openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} SKY_OBJ\n")
                        n_sky += 1
                if n_sky < 1: 
                    jitter_sky_option = ['--tbsub=false', '--sky_tweak=0', '--aj-method=7']
                else:
                    jitter_sky_option = ['--tbsub=true', '--sky_tweak=1']
                if n_obj < 1:
                    print(f" >> skipped, on find {n_obj} science frame.")
                    logging.warning(f"skipped {objname}({ob_id}) with {band}+{spaxel}+{exptime}, only find {n_obj} science frame and {n_sky} sky frame")
                    return

        elif dpr_catg == 'CALIB': 
            calib_meta_tab = meta_tab[meta_tab['DPR.CATG'] == 'CALIB']
            if stdstar_type == 'PSF':
                psf_calib = [True if 'PSF' in item['DPR.TYPE'] else False for item in calib_meta_tab]
                if np.sum(psf_calib) < 1:
                    logging.warning("No PSF star found!")
                    return
                auto_jitter_sof = os.path.join(outdir, f'{dpr_catg}_{stdstar_type}_jitter.sof')
                with open(auto_jitter_sof, 'w+') as openf:
                    for item in calib_meta_tab[psf_calib]:
                        if 'SKY' in item['DPR.TYPE']:
                            openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} SKY_PSF_CALIBRATOR\n")
                        else:
                            openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} PSF_CALIBRATOR\n")
            elif stdstar_type == 'STD':
                std_calib = [True if 'STD' in item['DPR.TYPE'] else False for item in calib_meta_tab]
                if np.sum(std_calib) < 1:
                    logging.warning("No STD star found!")
                    return
                auto_jitter_sof = os.path.join(outdir, f'{dpr_catg}_{stdstar_type}_jitter.sof')
                with open(auto_jitter_sof, 'w+') as openf:
                    for item in calib_meta_tab[std_calib]:
                        if 'SKY' in item['DPR.TYPE']:
                            openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} SKY_STD\n")
                        else:
                            openf.write(f"{raw_pool}/{item['DP.ID']}.{fits_suffix} STD\n")
        with open(auto_jitter_sof, 'a+') as openf:    
            openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
            openf.write(f"{calib_pool}/eris_ifu_wave_map.fits WAVE_MAP\n")
            # openf.write(f"{calib_pool}/eris_ifu_distortion_slitlet_pos.fits SLITLET_POS\n")
            openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
            openf.write(f"{calib_pool}/eris_ifu_dark_master_dark.fits MASTER_DARK\n")
            # openf.write(f"{static_pool}/EXTCOEFF_TABLE.fits EXTCOEFF_TABLE\n")
            openf.write(f"{static_pool}/eris_oh_spec.fits OH_SPEC\n")
            if (dpr_catg == 'CALIB') and (stdstar_type == 'STD'):
                if band in ['H_low', 'J_low', 'K_low']:
                    openf.write(f"{static_pool}/RESPONSE_WINDOWS_{band}.fits RESPONSE\n")
    if not dry_run:
        if dpr_catg == 'SCIENCE':
            subprocess.run([*esorex_cmd_list, f'--output-dir={outdir}', 
                            # run the pipeline with corrupted or missing files
                            '--check-sof-exist=false', 
                            'eris_ifu_jitter',
                            # '--tbsub=true', '--sky_tweak=1', # moved to jitter_sky_option 
                            *jitter_sky_option,
                            # '--aj-method=0', 
                            '--product_depth={}'.format(product_depth), 
                            '--dar-corr=true', '--cube.combine=false', 
                            auto_jitter_sof])
        elif dpr_catg == 'ACQUISITION':
            subprocess.run([*esorex_cmd_list, f'--output-dir={outdir}', 
                            '--check-sof-exist=false', 
                            'eris_ifu_stdstar', 
                            # '--tbsub=true', '--sky_tweak=1', # moved to jitter_sky_option 
                            *jitter_sky_option,
                            '--extract-source=false',
                            '--product_depth={}'.format(product_depth), 
                            '--dar-corr=true', '--cube.combine=false',
                            auto_jitter_sof])
        elif dpr_catg == 'CALIB':
            subprocess.run([*esorex_cmd_list, f'--output-dir={outdir}', 
                            'eris_ifu_stdstar', *jitter_sky_option,
                            '--product_depth={}'.format(product_depth), 
                            auto_jitter_sof])


#####################################
######### Flux calibration ##########

def gaussian_2d(params, x=None, y=None):
    """a simple 2D gaussian function
    
    Args:
        params: all the free parameters
                [amplitude, x_center, y_center, x_sigma, y_sigma, beta]
        x: x grid
        y: y grid
    """
    amp, x0, y0, xsigma, ysigma, beta = params
    return amp*np.exp(-0.5*(x-x0)**2/xsigma**2 - beta*(x-x0)*(y-y0)/(xsigma*ysigma)- 0.5*(y-y0)**2/ysigma**2)

def fit_star(starfits, x0=None, y0=None, pixel_size=1, plot=False,
             extract_spectrum=True, star_type=None, T_star=None,
             outfile=None):
    """two dimentional Gaussian fit

    """
    # read the fits file
    with fits.open(starfits) as hdu:
        header = hdu[0].header
        cube_header = hdu[1].header
        cube = hdu[1].data
        wavelength = ((((np.arange(cube_header['NAXIS3'])+1.0) - cube_header['CRPIX3']) * cube_header['CD3_3'] 
                       + cube_header['CRVAL3']) * u.Unit(cube_header['CUNIT3'])).to(u.um)
        exptime = header['EXPTIME']
        arcfile = header['ARCFILE']

    nchan, yshape, xshape = cube.shape
    rms = 1

    # collapse the cube to get the image of the star
    # this is used to find the star and get its shape (by 2d-Gaussian fitting)
    image = np.ma.median(np.ma.masked_invalid(cube), axis=0)

    # star the gaussian fitting for the 2D-image
    # generate the grid
    sigma2FWHM = np.sqrt(8*np.log(2))
    ygrid, xgrid = np.mgrid[0:yshape,0:xshape]
    # ygrid, xgrid = np.meshgrid((np.arange(0, yshape)+0.5)*pixel_size,
                               # (np.arange(0, xshape)+0.5)*pixel_size)
    # automatically estimate the initial parameters
    if x0 is None:
        # amp = np.percentile(image, 0.8)
        amp = np.max(image)
        yidx, xidx = np.where(image>0.9*amp)
        x0, y0 = np.median(xidx), np.median(yidx)
        xsigma, ysigma = pixel_size, pixel_size 
        p_init = [amp, x0, y0, xsigma, ysigma, 0]

    def _cost(params, xgrid, ygrid):
        return np.sum((image - gaussian_2d(params, xgrid, ygrid))**2/rms**2)
    res_minimize = optimize.minimize(_cost, p_init, args=(xgrid, ygrid), method='BFGS')

    amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = res_minimize.x
    xfwhm_fit, yfwhm_fit = xsigma_fit*sigma2FWHM, ysigma_fit*sigma2FWHM
    # print(xfwhm_fit, yfwhm_fit)
   
    if extract_spectrum:
        # get all the best fit value
        aperture_size = np.mean([xfwhm_fit, yfwhm_fit])
        aperture_correction = 1.0667
        aperture = EllipticalAperture([x0_fit, y0_fit], aperture_size, aperture_size, theta=0)
        aper_mask = aperture.to_mask().to_image([xshape, yshape]).astype(bool)
        aper_mask_3D = np.repeat(aper_mask[None,:,:], nchan, axis=0)
        data_selected = cube[aper_mask_3D]
        spectrum = aperture_correction*np.sum(data_selected.reshape((nchan, np.sum(aper_mask))), axis=1)
        spectrum = spectrum / exptime # units in adu/second
        output = spectrum
        output_unit = 'adu/s'

    if T_star is not None:
        blackbody_star = models.BlackBody(temperature=T_star*u.K, 
                                          scale=1.0*u.erg/u.s/u.cm**2/u.um/u.sr)

        transmission = spectrum/blackbody_star(wavelength)
        dwavelength = wavelength[1] - wavelength[0]
        # TODO: mask the possible spectral obsorption lines
        norm = np.nansum(transmission * dwavelength) / (dwavelength * np.nansum(np.ones_like(transmission)))
        transmission_norm = transmission/norm
        output = transmission_norm.value
        output_unit = transmission_norm.unit.to_string()

    if plot:
        fit_image = gaussian_2d(res_minimize.x, xgrid, ygrid)
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(image, origin='lower')
        ax1.plot([x0], [y0], 'x', color='red')
        ax2 = fig.add_subplot(gs[0,1])
        ax2.imshow(fit_image, origin='lower')
        ax3 = fig.add_subplot(gs[0,2])
        ax3.imshow(image - fit_image, origin='lower')
        ax4 = fig.add_subplot(gs[1,:])
        ax4.plot(wavelength, output)
        if T_star is not None:
            pass
            #ax4.plot(wavelength, blackbody_star(wavelength))
    if outfile is not None:
        np.savetxt(outfile, np.vstack([wavelength.value, output]).T, delimiter=',',
                   header=wavelength.unit.to_string()+','+output_unit)
    else:
        return wavelength.value, output

def apply_filter(spectrum, filter_profile, filter_name=None):
    wave_spec, data = spectrum
    wave_filter, transmission = filter_profile
    dwave = np.min([wave_spec[1]-wave_spec[0], wave_filter[1]-wave_filter[0]])
    # check if the spectrum covered the whole filter
    max_wave_filter = np.max(wave_filter)
    min_wave_filter = np.min(wave_filter)
    if (np.min(wave_spec) > min_wave_filter) or (np.max(wave_spec) < max_wave_filter):
        print(np.min(wave_spec), np.max(wave_spec))
        print(min_wave_filter, max_wave_filter)
        # raise ValueError("The spectrum cannot be covered by the given filter!")
        print('Warning: the spectrum cannot be covered by the given filter!')
    wave_ref = np.arange(min_wave_filter, max_wave_filter, dwave)
    spec_resampled = np.interp(wave_ref, wave_spec, data)
    filter_resampled = np.interp(wave_ref, wave_filter, transmission)
    return np.nansum(spec_resampled*filter_resampled*dwave)/np.sum(filter_resampled*dwave)

def magnitude2flux_2mass(magnitude, band):
    # transform Ks magnitude in total flux, integrated over the filter band pass
    # https://iopscience.iop.org/article/10.1086/376474/pdf
    if band == 'Ks':
        flux0 = 4.283e-7 # erg/s/cm2/micron #
    elif band == 'J':
        flux0 = 3.129e-7
    elif band == 'H':
        flux0 = 2.843e-7
    f = magnitude / (-2.5) + np.log10(flux0)
    return 10**f * u.erg / (u.cm * u.cm * u.s * u.um)

def get_zero_point(spec, filter_profile, magnitude, band, transmission):
    """get the zero point of from the STD star
    """
    wave_star, spec_star = spec
    wave_filter, transmission_filter = filter_profile
    star_raw = spec_star / transmission[1] # adu/s
    flux = magnitude2flux_2mass(magnitude, band) 
    zp = flux / apply_filter([wave_star, spec_star], filter_profile)
    return zp

def correct_cube(fitscube, transmission=None, zp=None, suffix='.corrected'):
    """correct the flux the cube"""
    with fits.open(fitscube) as hdu:
        header = hdu[0].header
        cube = hdu[1].data
        cube_header = hdu[1].header

    correction = 1.0
    if transmission is not None:
        transmission_corr = np.loadtxt(transmission, delimiter=',')
        correction = correction/transmission_corr[:,1]
    if zp is not None:
        cube_header['BUNIT'] = 'erg/s/cm^2/um'
        correction = correction * zp
    primary_hdu = fits.PrimaryHDU(header=header)
    modified_hdu = fits.ImageHDU(cube*correction[:,np.newaxis, np.newaxis], name="DATA", header=cube_header)
    # error_combined_hdu = fits.ImageHDU(error_combined, name="ERROR", header=hdr)
    hdus = fits.HDUList([primary_hdu, modified_hdu])
    hdus.writeto(fitscube[:-5]+f'{suffix}.fits', overwrite=True)


#####################################
######### DATA Combination ##########

def fill_mask(data, mask=None, step=1, debug=False):
    """Using iterative median to filled the masked region
    In each cycle, the masked pixel is set to the median value of all the values in the 
    surrounding region (in cubic 3x3 region, total 8 pixels)
    Inspired by van Dokkum+2023 (PASP) and extended to support 3D datacube
   
    This implementation are pure python code, so it is super
    slow if the number of masked pixels are large
    <TODO>: rewrite this extension with c

    Args:
        data (ndarray): the input data
        mask (ndarray): the same shape as data, with masked pixels are 1 (True)
                        and the rest are 0 (False)
    """
    if isinstance(data, np.ma.MaskedArray):
        mask = data.mask
        data = data.data
    elif mask is None:
        data = np.ma.masked_invalid(data)
        mask = data.mask
        data = data.data
    # skip the filling if there are too little data
    if debug:
        print("data and mask:",data.size, np.sum(mask))
        print(f"mask ratio: {1.*np.sum(mask)/data.size}")
    if 1.*np.sum(mask)/data.size > 0.2:
        logging.warning(f"skip median filling, too inefficient...")
        data[mask] = np.median(data[~mask])
        return data
    ndim = data.ndim
    data_filled = data.copy().astype(float)
    data_filled[mask==1] = np.nan
    data_shape = np.array(data.shape)
    up_boundaries = np.repeat(data_shape,2).reshape(len(data_shape),2)-1
    mask_idx = np.argwhere(mask > 0)
    while np.any(np.isnan(data_filled)):
        for idx in mask_idx:
            idx_range = np.array([[i-step,i+1+step] for i in idx])
            # check if reaches low boundaries, 0
            if np.any(idx_range < 1):  
                idx_range[idx_range < 0] = 0
            # check if reach the upper boundaries
            if np.any(idx > up_boundaries):
                idx_range[idx_range>up_boundaries] = up_boundaries[idx_range>up_boundaries]
            ss = tuple(np.s_[idx_range[i][0]:idx_range[i][1]] for i in range(ndim))
            data_filled[tuple(idx)] = np.nanmedian(data_filled[ss])
    return data_filled

def construct_wcs(header, data_shape=None):
    """try to construct the wcs from a broken header 

    TODO: not tested
    """
    return
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
        cdelt1, cdelt2 = spaxel/3600., spaxel/3600.
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

def mask_spectral_lines(wavelength, redshift=0, line_width=1000,):
    """mask spectral lines

    Args:
        wavelength (ndarray): wavelength in um
        redshift (float): redshift
        line_width (float): velocity width in km/s
    """
    line_dict = {
         'Halpha': 6564.61 * units.AA, 
         'NII6548': 6549.86 * units.AA, 
         'NII6583': 6585.27 * units.AA, 
         'SII6717': 6718.29 * units.AA, 
         'SII6731': 6732.67 * units.AA, 
         'Hbeta': 4862.68 * units.AA, 
         'OIII4959': 4960.295 * units.AA, 
         'OIII5007': 5008.240 * units.AA,
         }
    wave_masked = np.full_like(wavelength, fill_value=False)
    for line_name in line_dict:
        line_wave = (line_dict[line_name] * (1.0 + redshift)).to(units.um).value
        #line_width = 800.0 # km/s
        line_mask = np.logical_and(
            (wavelength >= (line_wave - (line_width/2.0/3e5)*line_wave)),
            (wavelength <= (line_wave + (line_width/2.0/3e5)*line_wave))
        )
        wave_masked = np.logical_or(wave_masked, line_mask)
    return wave_masked

def clean_cube(datacube, mask=None, signal_mask=None, sigma_clip=True, median_subtract=True,
               cont_subtract=False, median_filter=False, median_filter_size=(5,3,3),
               channel_chunk=None, sigma=5.0):
    """clean the datacubes

    It supports:
      - sigma clipping
      - background subtraction
      - continue subtraction
    """
    if mask is not None:
        if datacube.shape != mask.shape:
            raise ValueError("Mask does not match with data!")
    nchan, ny, nx = datacube.shape
    mask = mask | np.ma.masked_invalid(datacube).mask

    if median_filter:
        # apply the median filter to filter out the outliers caused by sky lines
        # choose the filter size to preserve weak science data
        datacube = ndimage.median_filter(datacube, size=median_filter_size)

    # prepare the masked data
    datacube_masked = np.ma.array(datacube, mask=mask)

    if z is not None:
        # mask the possible spectral line


    if sigma_clip:
        # it is only safe to apply the sigma_clip if the signal is weak,
        # and the single exposure is dominated by noise and sky line (residuals)
        datacube_masked = astro_stats.sigma_clip(datacube_masked, sigma=sigma, 
                                                 maxiters=2, masked=True) 
        if channel_chunk is not None:
            # apply chunk based sigma clip, it could be useful if different 
            # spectral window show different noise level
            # chose the chunk size so that the signal is weaker than noise
            cube_chunk_mask = np.full_like(datacube, fill_value=False)
            chunk_steps = np.hstack((np.arange(0, nchan, channel_chunk)[:-1], 
                                   (np.arange(nchan, 0, -channel_chunk)-channel_chunk)[:-1]))
            for chan in chunk_steps:
                chunk_masked = astro_stats.sigma_clip(datacube_masked[chan:chan+channel_chunk], 
                                                      maxiters=2, sigma=sigma, masked=True)
                cube_chunk_mask[chan:chan+channel_chunk] = chunk_masked.mask
            datacube_masked = np.ma.array(datacube_masked, mask=cube_chunk_mask)

        # apply channel-wise sigma_clip (deprecated, too dangerous)
        # datacube_masked = astro_stats.sigma_clip(datacube_masked, maxiters=2, sigma=sigma,
                                                 # axis=(1,2), masked=True)
        
        # apply sigma_clip along the spectral axis
        # make sure to mask signals (through signal_mask) before applying this
        # datacube_masked = astro_stats.sigma_clip(datacube_masked, maxiters=2, sigma=sigma,
                                                 # axis=0, masked=True)

    if median_subtract:
        #
        datacube_signal_masked = np.ma.array(datacube_masked, mask=signal_mask)

        # apply a global median subtraction
        datacube_masked -= np.ma.median(datacube_signal_masked, axis=0).data[np.newaxis,:,:]

        # row and column based median subtraction
        # by y-axis
        datacube_masked -= np.ma.median(datacube_signal_masked, axis=1).data[:,np.newaxis,:]
        # by x-axis
        datacube_masked -= np.ma.median(datacube_signal_masked, axis=2).data[:,:,np.newaxis]
        
        datacube_signal_masked = np.ma.array(datacube_masked, mask=signal_mask)
        # median subtraction image by image
        spec_median =  np.ma.median(datacube_signal_masked, axis=(1,2))
        spec_median_filled = fill_mask(spec_median, step=5)
        datacube_masked -= spec_median_filled[:,np.newaxis, np.newaxis]

    if cont_subtract: # this will take a very long time
        # apply a very large median filter to cature large scale continue variation
        cont_datacube = ndimage.median_filter(datacube_signal_masked, size=(200, 10, 10))
        # set the science masked region as the median value of the final median subtracted median
        datacube_masked = datacube_masked - cont_datacube
        
    return datacube_masked

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

def combine_eris_cube(cube_list, pixel_shifts=None, savefile=None, z=None,
                      sigma_clip=True, sigma=5.0, median_subtract=True,
                      mask_ext=None, median_filter=False, median_filter_size=(5,3,3),
                      cont_subtract=False, weighting=None, overwrite=False, debug=False):
    """combining data directly in pixel space
    this function combine the image/datacube in the pixel space

    image_list (list): the list of filenames or ndarray
    offset (list,ndarray): the offset (x, y) of each image
    """
    # check existing files
    if savefile is not None:
        if os.path.isfile(savefile):
            if overwrite is False:
                logging.info(f"Find existing: {savefile}, set `overwrite=True` to overwrite existing file!")
                return 0
    ncubes = len(cube_list)
    # check the input variables
    if pixel_shifts is not None:
        if len(pixel_shifts) != ncubes:
            raise ValueError("Pixel_shift does not match the number of images!")
        pixel_shifts = np.array(pixel_shifts)
    
    # quick combine without offsets
    if pixel_shifts is None:
        print('Direct combining cubes...')
        for i in range(ncubes):
            cube = cube_list[i]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                header = fits.getheader(cube, 'DATA')
                cube_data = fits.getdata(cube, 'DATA')
                if i == 0:
                    first_header = header
                    first_wcs = WCS(header)
            if mask_ext is not None:
                cube_mask = fits.getdata(cube, mask_ext)
            else:
                cube_mask = np.full(cube_data.shape, fill_value=1e-8)
            cube_data[np.abs(1.0-cube_mask)<1e-6] = 0.0
            if i == 0:
                data_combined = np.zeros_like(cube_data, dtype=float)
                coverage_combined = np.zeros_like(cube_mask, dtype=float)
                header_combined = fits.getheader(cube, 0)
            data_combined += cube_data
            coverage_combined += (1 - cube_mask)
    else:
        print('Combining cubes with drifts correction...')
        # find the combined wcs
        # calculate the combined image size
        # to make things simpler, here we still keep the x and y equal

        # read the first image to get the image size
        # <TODO>: read all the header and compute the best size and wavelength
        # such as: wcs_combined = find_combined_wcs(image_list, offsets=offsets)
        first_header = fits.getheader(cube_list[0], 'DATA')
        first_wcs = WCS(first_header)
        pad = np.round(np.abs(pixel_shifts).max()).astype(int)
        padded_cube_size = (first_header['NAXIS3'], 
                             first_header['NAXIS2'] + 2*pad,
                             first_header['NAXIS1'] + 2*pad)

        # define the final product
        wavelength = get_wavelength(first_header)
        len_wavelength = len(wavelength)
        data_combined = np.full(padded_cube_size, fill_value=0.)
        coverage_combined = np.full(padded_cube_size, fill_value=0.)

        # to avoid numerical issue, scaling is needed to extrapolate 3D datacube
        # we will normalize the wavelength to be roughly the same as values of
        # the pixel coordinates
        wave_scale = np.mean(padded_cube_size[1:]) # mean size of the x and y shape
        wave_min, wave_max = np.min(wavelength), np.max(wavelength)
        wavelength_norm = (wavelength - wave_min)/wave_max*wave_scale
          
        # (deprecated), the weighting is now taking care by the coverage
        # # handle the weighting
        # if weighting is None:
            # # treat each dataset equally
            # try:
                # # get the weighting by the exposure time
                # weighting = compute_weighting_eris(cube_list)
            # except:
                # # otherwise treat every observation equally
                # weighting = np.full(ncubes, fill_value=1./ncubes)
 
        for i in range(ncubes):
            # ge the offset, we need to multiply -1 to inverse the shifts
            # as the calculation of pixel shifts is the (x0-xi, y0-yi) 
            # later we can them directly use "+"
            offset = -1 * pixel_shifts[i]

            # read data from each observation
            cube = cube_list[i]
            logging.info(f'{i+1}/{ncubes} working on: {cube}')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                header_primary = fits.getheader(cube, 0)
                cube_exptime = float(header_primary['EXPTIME'])
                header = fits.getheader(cube, 'DATA')
                cube_data = fits.getdata(cube, 'DATA')
            if mask_ext is not None:
                cube_mask = fits.getdata(cube, mask_ext)
            else:
                cube_mask = np.full(cube_data.shape, fill_value=False)
            cube_wavelength = get_wavelength(header)
            cube_wavelength_norm = (cube_wavelength-wave_min)/wave_max*wave_scale
            cube_nchan, cube_ny, cube_nx = cube_data.shape
        
            if z is not None:
                # with redshift, we can mask the signals, here are the H-alpha and [N II]
                wave_mask = mask_spectral_lines(cube_wavelength, redshift=z)
                cube_wave_mask = np.repeat(wave_mask, cube_ny*cube_nx).reshape(
                                           len(cube_wavelength), cube_ny, cube_nx)
            else:
                cube_wave_mask = None
            
            # mask the outliers and subtract median noise, see clean_cube for details
            # needs to be cautious in preveting removing possible signal 
            cube_data_masked = clean_cube(cube_data, mask=cube_mask, signal_mask=cube_wave_mask, 
                                          sigma_clip=sigma_clip, median_subtract=median_subtract,
                                          median_filter=median_filter, cont_subtract=cont_subtract)
            # simple alternative is:
            # cube_data_masked = astro_stats.sigma_clip(cube_data, sigma=5)

            cube_data = cube_data_masked.filled(0)
            cube_mask = cube_data_masked.mask

            # follow two steps
            # step 1: get the pixel level shifts
            offset_pixel = np.round(offset).astype(int) 
            offset_residual = np.array(offset) - offset_pixel
            # step 2: interpolate the sub-pixel shift 
            #         to get the data on the closest grid
            xidx = np.arange(0., cube_nx)
            yidx = np.arange(0., cube_ny)
            xidx_origin = xidx + offset_residual[0]
            yidx_origin = yidx + offset_residual[1]

            # define the final grid
            wavegrid, ygrid, xgrid = np.meshgrid(wavelength_norm, yidx, xidx, indexing='ij')
            # if np.sum(np.abs(offset_residual)) < 1e-2:
                # print('skip interpolation')
                # # skip interpolation
                # cube_data_new = cube_data
                # cube_mask_new = cube_mask
            # else:
                # pass
            if True:
                if debug:
                    print(f'Current cube shape: {cube_data.shape}')
                    print(f'Extrapolated cube shape: {len_wavelength, cube_ny, cube_nx}')
                    print(f'the first two channel: {wavelength[0]} vs {cube_wavelength[0]}')
                    print(f'the last two channel: {wavelength[-1]} vs {cube_wavelength[-1]}')
                # the commented one is too slow for 3D datacube, so switch to the second method
                # cube_data_new = scipy.interpolate.griddata((wavegrid_origin, ygrid_origin, xgrid_origin), 
                                                        # cube_data.ravel(), (wavegrid, ygrid, xgrid), 
                                                        # method='linear', fill_value=0)
                cube_data_new = scipy.interpolate.interpn(
                        (cube_wavelength_norm, yidx_origin, xidx_origin), cube_data,
                        np.vstack([wavegrid.ravel(), ygrid.ravel(), xgrid.ravel()]).T, 
                        method='linear', bounds_error=False, fill_value=0).reshape(
                                (len_wavelength, cube_ny, cube_nx))

                # for the mask, we use the nearest interpolate
                # cube_mask_new = scipy.interpolate.griddata((wavegrid_origin, ygrid_origin, xgrid_origin), 
                                                            # cube_mask.ravel(), (wavegrid_final, ygrid, xgrid), 
                                                            # method='nearest', fill_value=0)
                cube_mask_new = scipy.interpolate.interpn(
                        (cube_wavelength_norm, yidx_origin, xidx_origin), cube_mask.astype(float), 
                        np.vstack([wavegrid.ravel(), ygrid.ravel(), xgrid.ravel()]).T, 
                        method='nearest', bounds_error=False, fill_value=True).reshape(
                                len_wavelength,cube_ny,cube_nx)
            if debug:
                print(f'slicing in y: {offset_pixel[1]+pad} {offset_pixel[1]+cube_ny+pad}')
                print(f'slicing in x: {offset_pixel[0]+pad} {offset_pixel[0]+cube_nx+pad}')
            data_combined[:, (offset_pixel[1]+pad):(offset_pixel[1]+cube_ny+pad), 
                          (offset_pixel[0]+pad):(offset_pixel[0]+cube_nx+pad)] += cube_data_new
            coverage_combined[:, (offset_pixel[1]+pad):(offset_pixel[1]+cube_ny+pad), 
                              (offset_pixel[0]+pad):(offset_pixel[0]+cube_nx+pad)] += cube_exptime*(1.-cube_mask_new)
    # the divide rescale the unit from asu to asu/second
    # (*+1e-8) is to avid numerical issue
    with np.errstate(divide='ignore', invalid='ignore'):
        data_combined = data_combined / (coverage_combined) 

    if savefile is not None:
        # save the combined data
        hdr = first_wcs.to_header()
        # for key in ['NAXIS3', 'CTYPE3', 'CUNIT3', 'CRPIX3', 'CRVAL3', 'CDELT3', 
                    # 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
                    # 'CD3_3', 'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2', 
                    # 'PC3_3', 'PC1_3', 'PC2_3', 'PC3_1', 'PC3_2']:
            # try:
                # hdr[key] = first_header[key]
            # except:
                # pass
        if pixel_shifts is not None:
            hdr['CRPIX1'] = first_header['CRPIX1'] + pad + pixel_shifts[0][0]
            hdr['CRPIX2'] = first_header['CRPIX2'] + pad + pixel_shifts[0][1]
        hdr['BUNIT'] = first_header['BUNIT'].strip() + '/s'
        hdr['OBSERVER'] = 'galphys'
        hdr['COMMENT'] = 'MPE/IR'
        hdr['COMMENT'] = 'Report problems to jhchen@mpe.mpg.de'
        primary_hdu = fits.PrimaryHDU()
        data_combined_hdu = fits.ImageHDU(data_combined, name="DATA", header=hdr)
        # error_combined_hdu = fits.ImageHDU(error_combined, name="ERROR", header=hdr)
        hdus = fits.HDUList([primary_hdu, data_combined_hdu])
        hdus.writeto(savefile, overwrite=overwrite)
    else:
        return data_combined 

def read_eris_drifts(datfile, arcfilenames, xcolname='Xref', ycolname='Yref'):
    """read eris drifting table
    """
    pixel_center = [32., 32.] # the expected center
    dat = table.Table.read(datfile, format='csv')
    drifts = np.zeros((len(arcfilenames), 2)) # drift in [x, y]
    if xcolname not in dat.colnames:
        print(f"Failed read the drift from the colname {xcolname} in {datfile}")
        return drifts
    for i in range(len(arcfilenames)):
        arcfile = arcfilenames[i]
        dat_img = dat[dat['ARCFILE'] == arcfile]
        if len(dat_img) == 1:
            drifts[i] = [dat_img[xcolname][0]-pixel_center[0], dat_img[ycolname][0]-pixel_center[1]]
        else:
            print("Drifts not found!")
            drifts[i] = [0.,0.]
    return drifts

def compute_eris_offset(image_list, additional_drifts=None, header_ext='Primary',
                        header_ext_data='DATA',
                        ra_offset_header='HIERARCH ESO OCS CUMOFFS RA',
                        dec_offset_header='HIERARCH ESO OCS CUMOFFS DEC',
                        x_drift_colname='x_model', y_drift_colname='y_model',
                        coord_system='sky', debug=False, outfile=None):
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
            ra_diff = data_header['CD1_1']*3600.
            dec_diff = data_header['CD2_2']*3600.
            if i == 0: 
                ra_offset_0 = ra_offset
                dec_offset_0 = dec_offset
                # <TODO>: add support for sky level offset
                # convert the skycoords to pixels use the first wcs
                # if coord_system == 'sky':
                    # image_wcs = WCS(header)
            array_offset[i][:] = (ra_offset_0-ra_offset)/ra_diff, (dec_offset_0-dec_offset)/dec_diff
    # consider additional offset
    if debug:
        print(">>>>>>>>\nOCS offset:")
        print(array_offset)
    if additional_drifts is not None:
        if isinstance(additional_drifts, str):
            additional_drifts = read_eris_drifts(additional_drifts, arcfilenames)
            if debug:
                print('++++++++\n additional difts:')
                print(additional_drifts)
        for i in range(nimage):
            array_offset[i] += additional_drifts[i]
    if outfile is not None:
        with open(outfile, 'w+') as fp:
            for i in array_offset:
                fp.write(f"{i[0]:.2f} {i[1]:.2f} \n")
    return array_offset

def search_eris_files(dirname, pattern=''):
    matched_files = []
    # This will return absolute paths
    file_list = [f for f in glob.iglob(dirname.strip('/')+"/**", recursive=True) if os.path.isfile(f)]
    for ff in file_list:
        if re.search(pattern, os.path.basename(ff)):
            matched_files.append(ff)
    return matched_files

def search_archive(datadir, target=None, band=None, spaxel=None, excludes=None, outfile=None, 
                   tag='',):
    target_matcher = re.compile("(?P<target>[\w\s\-\.+]+)_(?P<id>\d{7})_(?P<band>[JKH]_[\w]{3,6}?)_(?P<spaxel>\d{2,3}mas)_(?P<exptime>\d+)s")
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')

    dates = os.listdir(datadir)
    image_list = []
    image_exp_list = []
    for date in dates:
        if not date_matcher.match(date):
            continue
        for obs in os.listdir(os.path.join(datadir, date)):
            try:
                obs_match = target_matcher.search(obs).groupdict()
            except:
                obs_match = None
                continue
            if obs_match is not None:
                obs_dir = os.path.join(datadir, date, obs)
                ob_target, ob_id= obs_match['target'], obs_match['id']
                ob_band, ob_spaxel = obs_match['band'], obs_match['spaxel']
                ob_exptime = obs_match['exptime']

            if ob_target != target:
                continue
            if (ob_band==band) and (ob_spaxel==spaxel):
                # combine the exposures within each OB
                exp_list = glob.glob(obs_dir+'/eris_ifu_jitter_dar_cube_[0-9]*.fits')
                # check the arcfile name not in the excludes file list
                for fi in exp_list:
                    with fits.open(fi) as hdu:
                        arcfile = hdu['PRIMARY'].header['ARCFILE']
                        if excludes is not None:
                            if arcfile in excludes:
                                continue
                        image_list.append(fi)
                        image_exp_list.append(ob_exptime)

    if outfile is not None:
        with open(outfile, 'w+') as fp:
            for img in image_list:
                fp.write(f"{img} {tag}\n")
    return image_list, image_exp_list


#####################################
######### DATA Quicklook ############

def pv_diagram(datacube, velocity=None, z=None, pixel_center=None,
               vmin=-400, vmax=400,
               length=1, width=1, theta=0, debug=False, plot=True, pixel_size=1):
    """generate the PV diagram of the cube

    Args:
        datacube: the 3D data [velocity, y, x]
        pixel_center: the center of aperture
        lengthth: the length of the aperture, in x axis when theta=0
        width: the width of the aperture, in y axis when theta=0
        theta: the angle in degree, from positive x to positive y axis
    """
    # nchan, ny, nx = datacube.shape
    if isinstance(datacube, str):
        if z is not None:
            velocity, datacube, header = read_eris_cube(datacube, z=z)
        else:
            wavelength, datacube, header = read_eris_cube(datacube)
    # extract data within the velocity range 
    if velocity is not None:
        vel_selection = (velocity > vmin) & (velocity < vmax)
        datacube = datacube[vel_selection]
        velocity = velocity[vel_selection]
    cubeshape = datacube.shape
    aper = RectangularAperture(pixel_center, length, width, theta=theta/180*np.pi)
    s1,s2 = aper.to_mask().get_overlap_slices(cubeshape[1:])
    # cutout the image
    sliced_cube = datacube[:,s1[0],s1[1]]
    # rotate the cube to make the aperture with theta=0
    sliced_cube_rotated = ndimage.rotate(sliced_cube, aper.theta/np.pi*180, axes=[1,2], reshape=True, prefilter=False, order=0, cval=np.nan)
    # sum up the central plain (x axis) within the height
    nchan, nynew, nxnew = sliced_cube_rotated.shape
    # define the new aperture on the rotated sub-cube
    aper_rotated = RectangularAperture([0.5*(nxnew-1), 0.5*(nynew-1)], length, width, theta=0)

    hi_start = np.round(nynew/2.-width/2.).astype(int)
    width_slice = np.s_[hi_start:hi_start+width]
    sliced_pvmap = np.nansum(sliced_cube_rotated[:,width_slice,:], axis=1)
    if debug: # debug plot 
        fig = plt.figure(figsize=(12,5))
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        ax1.imshow(np.sum(datacube,axis=0),origin='lower')
        aper.plot(ax=ax1)
        ax2.imshow(np.sum(sliced_cube_rotated, axis=0), origin='lower')
        aper_rotated.plot(ax=ax2)
        ax3.imshow(sliced_pvmap.T, origin='lower')
    if plot:
        fig = plt.figure(figsize=(12,5))
        # ax = fig.subplots(1,2)
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        vmin, vmax = -1*np.nanstd(sliced_cube), 4*np.nanmax(sliced_cube)
        ax1.imshow(np.sum(datacube, axis=0), origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
        aper.plot(ax=ax1, color='white', linewidth=4)
        # show the pv-diagram
        # ax[1].imshow(sliced_pvmap, origin='lower', cmap='magma', extent=extent)
        positions = np.linspace(-0.5*length, 0.5*length, nxnew)
        if velocity is None:
            velocity = np.linspace(-1,1,nchan)
        vmesh, pmesh = np.meshgrid(velocity,positions)
        ax2.pcolormesh(vmesh, pmesh, sliced_pvmap.T, cmap='magma')
        gauss_kernel = Gaussian2DKernel(1)
        smoothed_pvmap = convolve(sliced_pvmap, gauss_kernel)
        ax3.pcolormesh(vmesh, pmesh, smoothed_pvmap.T, cmap='magma')
        ax2.set_ylabel('Position')
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Velocity')
    return 

def gaussian_1d(params, velocity=None):
    """a simple 1D gaussian function
    Args:
        params: all the free parameters
                [amplitude, center, sigma, cont]
        v: velocity
    """
    n_param = len(params)
    amp, v0, sigma = params[:3]
    if n_param == 4:
        cont = params[3]
    else:
        cont = 0
    # return amp / ((2*np.pi)**0.5 * sigma) * np.exp(-0.5*(velocity-v0)**2/sigma**2) + cont
    return amp * np.exp(-0.5*(velocity-v0)**2/sigma**2) + cont

def calc_gaussian1d_chi2(params, velocity=None, spec=None, std=None):
    fit = gaussian_1d(params, velocity=velocity)
    chi2 = np.sum((spec-fit)**2/std**2)
    return chi2

def fit_spec(vel, spec, std, plot=False, ax=None,
             sigma_guess=30, velocity_bounds=[-400,500], 
             intensity_bounds=[0, np.inf], sigma_bouds=[10,1000], 
             fit_cont=True, cont_bounds=[0, np.inf], ):
    """a simple spectrum fitter
    """
    # normalize the spectrum to aviod numerical issue
    norm_scale = np.ma.max(spec)
    spec_norm = spec/norm_scale
    # guess the initial parameters
    spec_selection = spec_norm > np.percentile(spec_norm, 90)
    I0 = 2*np.mean(spec_norm[spec_selection])
    vel0 = np.median(vel[spec_selection])
    vsig0 = sigma_guess
    bounds = [intensity_bounds, velocity_bounds, sigma_bouds]
    if fit_cont:
        cont0 = np.median(spec_norm[~spec_selection])
        initial_guess = [I0, vel0, vsig0, cont0]
        bounds.append(cont_bounds)
    else:
        cont0 = 0
        initial_guess = [I0, vel0, vsig0]
    # make profile of initial guess.  this is only used for information/debugging purposes.
    guess = gaussian_1d(initial_guess, velocity=vel)
    # do the fit
    fit_result = optimize.minimize(calc_gaussian1d_chi2, initial_guess, 
                                   args=(vel, spec_norm, std*spec_norm), bounds=bounds)

    # make profile of the best fit 
    bestfit = gaussian_1d(fit_result.x, velocity=vel)
    if plot:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax.step(vel, spec_norm, color='black', where='mid', label='data')
        ax.plot(vel, guess, linestyle='dashed', color='blue', label='guess')
        # ax.plot(vel, spec*0+cont0+std, linestyle='dashed', color='blue', alpha=0.5)
        # ax.plot(vel, spec*0+cont0-std, linestyle='dashed', color='blue', alpha=0.5)
        ax.plot(vel, bestfit, color='red')
        plt.show()
    bestfit_params = fit_result.x
    bestfit_params[0] = bestfit_params[0]*norm_scale
    if fit_cont:
        bestfit_params[-1] = bestfit_params[-1]*norm_scale
    return bestfit_params

def fit_eris_cube(cube, velocity=None, SNR_limit=2, plot=False, fit_cont=True,
                  z=None, rest_wave=0.656279*u.um, 
                  minaper=1, maxaper=4, box=None, 
                  vel_low=-400, vel_up=400, savefile=None):
    """fit gaussian line through the cube

    Args:
        datacube: it can be the fitsfile or the datacube
        box: the selection pixel of a box 'z1,z2,y1,y2,x1,x2', 
             (x1, y1): the bottom left coord
             (x2, y2): upper right coord
    """
    if isinstance(cube, str):
        vel, cube, header = read_eris_cube(cube, z=z, rest_wave=rest_wave)
    if box is not None:
        box_idx = box.split(',')
        box_idx = np.array(box.split(',')).astype(int)
        cube = cube[box_idx[0]:box_idx[1], box_idx[2]:box_idx[3], box_idx[4]:box_idx[5]]
        vel = vel[box_idx[0]:box_idx[1]]
    cube_shape = cube.shape
    fitcube = np.zeros_like(cube)
    imagesize = cube_shape[-2:]
    nspec = cube_shape[-3]
    # mean, median, std = astro_stats.sigma_clipped_stats(cube, sigma=10)
    # mask = np.ma.masked_invalid(cube).mask
    # vmax = 1.5*np.percentile(cube[~mask], 90)

    # A cube to save all the best-fit values (maps)
    # [amp, velocity, sigma, SNR]
    fitmaps = np.full((5, cube_shape[-2], cube_shape[-1]), fill_value=np.nan)
    
    if plot:
        fig = plt.figure()
    # loop over all pixels in the cube and fit 1d spectra
    for y in range(0, cube_shape[-2]):
        for x in range(0, cube_shape[-1]):
            is_fitted = False
            # loop over the range of adaptive binning vakues
            for aper in range(minaper,maxaper+1):
                if is_fitted:
                    break
                if True:
                    # deal with the edges of the cube
                    sz = cube_shape
                    xlo = x - aper
                    xhi = x + aper
                    ylo = y - aper
                    yhi = y + aper
                    if xlo <= 0: xlo = 0
                    if xhi > sz[2]: xhi = sz[2]
                    if ylo <= 0: ylo = 0
                    if yhi >= sz[1]: yhi = sz[1]

                    # vector to hold the spectra
                    spec = np.zeros(sz[0])
                    # loop over x/y and integrate the cube to make a 1D spectrum
                    for m in range(xlo,xhi):
                        for n in range(ylo,yhi):
                            tmp = cube[:,n,m]
                            tmp = tmp-np.median(tmp)
                            spec = spec + tmp
                    spec = spec / ((yhi-ylo) * (xhi-xlo))

                    # spec = cube[:, y, x]
                    # only do a fit if there are values in the array
                    if np.nansum(spec) != 0:
                        # measure the std
                        cont_window = np.where((vel<=vel_low) | (vel>=vel_up))
                        std = np.std(spec[cont_window[0]])                            
                        med = np.median(spec[cont_window[0]])
                        # get chi^2 of straight line fit
                        chi2_sline = np.sum((spec-med)**2 / std**2)
                        # do a Gaussian profile fit of the line
                        px = fit_spec(vel, spec, std, fit_cont=fit_cont)
                        bestfit = gaussian_1d(px, velocity=vel)
                        # calculate the chi^2 of the Gaussian profile fit
                        chi2_gauss = np.sum((spec-bestfit)**2 / std**2)
                        # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                        SNR = (chi2_sline - chi2_gauss)**0.5

                        # store the fit parameters with S/N>SNR_limit
                        if SNR >= SNR_limit:
                            is_fitted = True
                            print(f'fit found at {(x,y)} with S/N={SNR}')
                            fitmaps[0:3, y, x] = px[:3] # amp, vcenter, vsigma
                            if fit_cont:
                                fitmaps[3, y, x] = px[3]
                            fitmaps[4, y, x] = SNR
                            fitcube[:,y,x] = bestfit

                            # plot data and fit if S/N is above threshold
                            if plot:
                                ax = fig.add_subplot(111)
                                ax.step(vel, spec, color='black', where='mid', label='data')
                                ax.plot(vel, bestfit, color='red', alpha=0.8)
                                plt.show(block=False)
                                plt.pause(0.02)
                                plt.clf()
                        elif SNR<SNR_limit:
                            print(f'no fit found at {(x,y)} with S/N={SNR}')
    if savefile is None:
        return fitcube, fitmaps
    else:
        # keep files into fits file
        pass


#####################################
########### Quick Tools #############

def get_daily_calib(date, band, spaxel, exptime, outdir='./', esorex='esorex', 
                    calib_raw=None, overwrite=False, static_pool=None, 
                    steps=None, max_days=60, 
                    rename=False, debug=False, dry_run=False):
    """A wrapper to get daily calibration file quickly

    """
    if steps is None:
        steps=['dark','detlin','distortion','flat','wavecal']
    logging.info(f'Requesting calibration files on {date}:{band}+{spaxel}+{exptime}s...')
    archive_name = f'{date}_{band}_{spaxel}_{exptime}s'
    archive_outdir = os.path.join(outdir, archive_name)
    is_reduced = True
    # check if existing calibration files are exists
    if 'dark' in steps:
        if not os.path.isfile(archive_outdir+'/eris_ifu_dark_master_dark.fits'):
            is_reduced = False
    if 'detlin' in steps:
        if not os.path.isfile(archive_outdir+'/eris_ifu_detlin_bpm_filt.fits'):
            is_reduced = False
    if 'distortion' in steps:
        if not os.path.isfile(archive_outdir+'/eris_ifu_distortion_distortion.fits'):
            is_reduced = False
    if 'flat' in steps:
        if not os.path.isfile(archive_outdir+'/eris_ifu_flat_master_flat.fits'):
            is_reduced = False
    if 'wavecal' in steps:
        if not os.path.isfile(archive_outdir+'/eris_ifu_wave_map.fits'):
            is_reduced = False
    if 'stdstar' in steps:
        if not os.path.isfile(archive_outdir+'/eris_ifu_stdstar_std_cube_coadd.fits'):
            is_reduced = False
    if is_reduced:
        if not overwrite:
            logging.info(f"> re-use existing calibPool in {outdir}")
            return archive_outdir 

    # The date here is not the starting date. Normally, we can safely assume the
    # starting date is date - 1day, if the search is started at 12pm.
    # which is the default of eris_auto_quary
    start_date = (datetime.date.fromisoformat(date) 
                  - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    with tempfile.TemporaryDirectory() as tmpdir:
        if calib_raw is None:
            calib_daily_raw = tmpdir
        else:
            calib_daily_raw = os.path.join(calib_raw, date)
        metafile = os.path.join(calib_daily_raw, f'{date}_{band}_{spaxel}_{exptime}s.csv')
        request_calib(start_date=start_date, band=band, spaxel=spaxel, exptime=exptime, 
                      outdir=calib_daily_raw, metafile=metafile, steps=steps,
                      max_days=max_days, debug=debug, dry_run=dry_run)
        generate_calib(metafile, raw_pool=calib_daily_raw, outdir=archive_outdir, 
                       static_pool=static_pool, steps=steps, esorex=esorex,
                       debug=debug, dry_run=dry_run)
    if rename:
        # rename the files name to avoid conflicts
        for ff in glob.glob(os.path.join(archive_outdir, '*.fits')):
            ff_basename = os.path.basename(ff)
            if 'dark' in ff_basename:
                subprocess.run(['mv', ff, os.path.join(outdir, ff_basename[:-5]+f'_{exptime}s_{date}'+'.fits')])
            else:
                subprocess.run(['mv', ff, os.path.join(outdir, ff_basename[:-5]+f'_{band}_{spaxel}_{date}'+'.fits')])
        subprocess.run(['rm','-rf', archive_outdir])
        return outdir
    else:
        return archive_outdir

def run_eris_pipeline(datadir=None, outdir=None, esorex='esorex', overwrite=False, 
                      calib_pool='calibPool', calib_raw='calib_raw',
                      selected_categories=['SCIENCE','CALIB'], #ACQUISITION
                      debug=False, dry_run=False):
    """A quick pipeline for archived eris data

    To run this pipeline, the input datadir is organised by the dates of the 
    observations. Within each date, all the relevant science data have been
    download. 

    """
    # match all the dates
    datadir = datadir.strip('/')
    if outdir is not None:
        outdir = outdir.strip('/')
    else:
        outdir = '.'
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')
    date_list = []
    for subfolder in os.listdir(datadir):
        if date_matcher.match(subfolder):
            date_list.append(subfolder)
    # sort the dates
    date_list = sorted(date_list)

    for date in date_list:
        ## Step-1
        # generate all the summary files
        logging.info(f"> generating the metadata from {datadir}/{date}")
        # metadata = generate_metadata(os.path.join(datadir, date))
        date_metafile = os.path.join(datadir, date, 'metadata.csv')
        if os.path.isfile(date_metafile):
            # TODO: check the modified date of the files and metafile
            logging.info(f"> finding existing metadata:{date_metafile}")
        else:
            logging.info(f"> generating the metadata of {date}")
            date_folder = os.path.join(datadir, date)
            # if os.path.isdir(date_folder + '/headers'):
                # metadata = generate_metadata(header_dir=date_folder+'/headers', 
                                             # metafile=date_metafile)
            # else:
                # with tempfile.TemporaryDirectory() as tmpdir:
                    # metadata = generate_metadata(data_dir=date_folder, work_dir=tmpdir, 
                                                 # metafile=date_metafile)
            with tempfile.TemporaryDirectory() as tmpdir:
                metadata = generate_metadata(data_dir=date_folder, work_dir=tmpdir, 
                                             metafile=date_metafile)

        ## Step-2
        # read available data 
        logging.info(f"> reducing the science data in {datadir}/{date}")
    
        metadata = read_metadata(date_metafile)

        dpr_tech_select = [True if 'IFU' in item['DPR.TECH'] else False for item in metadata]
        metadata = metadata[dpr_tech_select]
        if len(metadata) < 1:
            logging.info(f"> No ERIS/SPIFFIER data found on {date}")
            continue
        # if valid data is found, then set up the directories
        daily_calib_pool = os.path.join(calib_pool, date)
        daily_datadir = os.path.join(datadir, date)

        # split all the data based on their OBs
        unique_OBs = np.unique(metadata['OB.ID'])

        for ob_id in unique_OBs:
            # each OB share the same band, spaxel
            # but we still need to get the dpr catagories with different exptime
            ob_metadata = metadata[metadata['OB.ID'] == ob_id]
            # get the name of the science target, here we assume each ob has only one target
            target = None
            try:
                target = ob_metadata[ob_metadata['DPR.CATG']=='SCIENCE']['OBS.TARG.NAME'][0]
            except:
                try:
                    target = ob_metadata[ob_metadata['DPR.CATG']=='CALIB']['OBS.TARG.NAME'][0]
                except:
                    logging.warning(f'Skip OB:{ob_id}, cannot determine the target name!')
                    continue
            # get all the DPR catagories: SCIENCE, ACQUISITION, CALIB (PSF_CALIBRATOR + STD)
            ob_cats_unique = np.unique(ob_metadata['DPR.CATG'])
            logging.info(f'Found DPR catagories: {",".join(ob_cats_unique)}')
            # read all the data with different catagories
            for cat in ob_cats_unique:
                if cat not in selected_categories:
                    continue
                cat_metadata = ob_metadata[ob_metadata['DPR.CATG']==cat]
                cat_exptime_list = np.unique(cat_metadata['DET.SEQ1.DIT']).astype(int).tolist()
                logging.info(f'{cat} recieve exptime with: {cat_exptime_list}')
                for exptime in cat_exptime_list:
                    stdstar_type = ''
                    first_meta = cat_metadata[(cat_metadata['DET.SEQ1.DIT']-exptime)<1e-6][0]
                    band  = first_meta['INS3.SPGW.NAME']
                    spaxel = first_meta['INS3.SPXW.NAME']
                    if cat == 'CALIB':
                        # determine it is STD or PSF_CALIBRATOR
                        dpr_type = first_meta['DPR.TYPE']
                        if 'PSF' in dpr_type:
                            stdstar_type = 'PSF'
                        elif 'STD' in dpr_type:
                            stdstar_type = 'STD'

                    # check whether the OB has been reduced
                    is_reduced = False
                    daily_ob_cat_outdir = os.path.join(outdir, date, 
                                             f'{target}_{cat}{stdstar_type}_{ob_id}_{band}_{spaxel}_{exptime}s')
                    expected_output = ['eris_ifu_jitter_dar_cube_coadd.fits', 
                                       'eris_ifu_jitter_obj_cube_coadd.fits',
                                       'eris_ifu_stdstar_obj_cube_000.fits',
                                       'eris_ifu_stdstar_sky_cube_000.fits',
                                       'eris_ifu_stdstar_psf_cube_000.fits',
                                       ]
                    for eo in expected_output:
                        if os.path.isfile(os.path.join(daily_ob_cat_outdir, eo)):
                            if not overwrite:
                                logging.info(f"> Done: {date}:{target}(OB.ID={ob_id}) with {cat}+{band}+{spaxel}+{exptime}s")
                                is_reduced = True
                    if is_reduced:
                        continue
                    # check whether the cat of the OB has been tried but failed
                    if os.path.isfile(os.path.join(daily_ob_cat_outdir, 'failed')):
                        if not overwrite:
                            logging.info(f"> Skip failure: {date}:{target}(OB.ID={ob_id}) with {cat}+{band}+{spaxel}+{exptime}s")
                            continue
                    
                    ## Step-3
                    # generate the daily calibPool
                    logging.info(f"> generating calibPool for {date} with {cat}+{band}+{spaxel}+{exptime}s")
                    daily_calib_pool_id = get_daily_calib(date,  band, spaxel, exptime, esorex=esorex,
                                                          outdir=daily_calib_pool, calib_raw=calib_raw,
                                                          overwrite=overwrite)
                    # except:
                        # logging.warning(f"> Error found in geting the calibPool of {date}: {target}(OB.ID={ob_id}) with {band}+{spaxel}+{exptime}s")

                    ## Step-4
                    # run eris_ifu_gitter on science target and acquisition/psf stars
                    logging.info(f"> working on {date}: {target}(OB.ID={ob_id}) with {cat}+{band}+{spaxel}+{exptime}s")

                    try:
                        auto_jitter(metadata=ob_metadata, raw_pool=daily_datadir, 
                                    outdir=daily_ob_cat_outdir, calib_pool=daily_calib_pool_id, 
                                    ob_id=ob_id, esorex=esorex, dpr_catg=cat, 
                                    stdstar_type=stdstar_type, product_depth=2,
                                    dry_run=dry_run)
                    except:
                        # subprocess.run(['rm','-rf', daily_ob_outdir])
                        # subprocess.run(['touch', daily_ob_outdir+'/failed'])
                        logging.warning(f"> Error found in runing {date}: {target}(OB.ID={ob_id}) with {band}+{spaxel}+{exptime}s")

def quick_combine_legacy(datadir=None, target=None, offsets=None, excludes=None, band=None,
                  spaxel=None, drifts=None, outdir='./', esorex='esorex', z=None, 
                  savefile=None, suffix='combined', overwrite=False):
    """A wrapper of combine_data

    This quick tool search the all the available and valid observations
    and combine them with the combine_data.

    This tool take the outdir from `run_eris_pipeline` as input, it will search 
    all the available observations, and combined all the available data
    """
    target_matcher = re.compile("(?P<target>[\w\s\-\.+]+)_(?P<id>\d{7})_(?P<band>[JKH]_[\w]{3,6}?)_(?P<spaxel>\d{2,3}mas)_(?P<exptime>\d+)s")
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')

    if not os.path.isdir(outdir):
        subprocess.run(['mkdir','-p', outdir])
    dates = os.listdir(datadir)
    image_list = []
    image_exp_list = []
    for date in dates:
        if not date_matcher.match(date):
            continue

        for obs in os.listdir(os.path.join(datadir, date)):
            try:
                obs_match = target_matcher.search(obs).groupdict()
            except:
                obs_match = None
                continue
            if obs_match is not None:
                obs_dir = os.path.join(datadir, date, obs)
                ob_target, ob_id= obs_match['target'], obs_match['id']
                ob_band, ob_spaxel = obs_match['band'], obs_match['spaxel']
                ob_exptime = obs_match['exptime']
            if ob_target != target:
                continue
            if (ob_band==band) and (ob_spaxel==spaxel):
                # combine the exposures within each OB
                exp_list = glob.glob(obs_dir+'/eris_ifu_jitter_dar_cube_[0-9]*.fits')
                # check the arcfile name not in the excludes file list
                exp_list_valid = []
                exp_list_arcfilenames = []
                for fi in exp_list:
                    with fits.open(fi) as hdu:
                        arcfile = hdu['PRIMARY'].header['ARCFILE']
                        if excludes is not None:
                            if arcfile in excludes:
                                continue
                        exp_list_valid.append(fi)
                        exp_list_arcfilenames.append(arcfile)

                if len(exp_list_valid) < 1:
                    logging.warning("no valid data for {target}({ob_id}) with {ob_band},{ob_spaxel},{ob_total_exptime}s")
                    continue
                # within each ob, the exposure time are equal, so we just ignore the
                # weighting
                ob_total_exptime = int(ob_exptime) * len(exp_list)
                obs_combined_filename = os.path.join(obs_dir, 
                    f"{target}_{ob_id}_{ob_band}_{ob_spaxel}_{ob_total_exptime}s_{suffix}.fits")

                if (not os.path.isfile(obs_combined_filename)) or overwrite:
                    obs_offset = compute_eris_offset(exp_list, additional_drifts=drifts)
                    combine_eris_cube(cube_list=exp_list, pixel_shifts=obs_offset, 
                                      z=z, overwrite=overwrite,
                                      savefile=obs_combined_filename)

                image_list.append(obs_combined_filename)
                image_exp_list.append(ob_total_exptime)
    # then, combine the data from different OB
    
    if len(image_list) < 1:
        logging.warning(f"no valid data for {target} with {band},{spaxel}")
        return
    # <TODO>: how to align different OBs
    total_exp = np.sum(image_exp_list)
    weighting = np.array(image_exp_list) / total_exp 
    logging.info("combining images from:")
    with open(os.path.join(outdir,
              f'{target}_{band}_{spaxel}_{total_exp/3600:.1f}h_{suffix}_list.txt'), 'w+') as fp:
        n_combine = len(image_list)
        for i in range(n_combine):
            im = image_list[i]
            im_exptime = image_exp_list[i]
            fp.write(f"{im} {im_exptime}\n")
            # with fits.open(im) as hdu:
                # print(hdu.info())
    if savefile is None:
        savefile = os.path.join(outdir, 
                    f'{target}_{band}_{spaxel}_{total_exp/3600:.1f}h_{suffix}.fits')
    combine_eris_cube(cube_list=image_list, weighting=weighting, z=z,
                      sigma_clip=False, median_subtract=False, overwrite=overwrite,
                      mask_ext=None, savefile=savefile)

def quick_combine(datadir=None, target=None, offsets=None, excludes=None, band=None,
                  spaxel=None, drifts=None, outdir='./', esorex='esorex', z=None, 
                  savefile=None, recipe='combine_eris_cube', suffix='combined', overwrite=False):
    """A wrapper of combine_data

    This quick tool search the all the available and valid observations
    and combine them with the combine_data.

    This tool take the outdir from `run_eris_pipeline` as input, it will search 
    all the available observations, and combined all the available data
    """
    target_matcher = re.compile("(?P<target>[\w\s\-\.+]+)_()(?P<id>\d{7})_(?P<band>[JKH]_[\w]{3,6}?)_(?P<spaxel>\d{2,3}mas)_(?P<exptime>\d+)s")
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')
    esorex_cmd_list = esorex.split(' ')

    if not os.path.isdir(outdir):
        subprocess.run(['mkdir','-p', outdir])
    dates = os.listdir(datadir)
    image_list = []
    image_exp_list = []
    image_arcname_list = []
    for date in dates:
        if not date_matcher.match(date):
            continue
        for obs in os.listdir(os.path.join(datadir, date)):
            try:
                obs_match = target_matcher.search(obs).groupdict()
            except:
                obs_match = None
                continue
            if obs_match is not None:
                obs_dir = os.path.join(datadir, date, obs)
                ob_target, ob_id= obs_match['target'], obs_match['id']
                ob_band, ob_spaxel = obs_match['band'], obs_match['spaxel']
                ob_exptime = obs_match['exptime']
            if ob_target != target:
                continue
            if (ob_band==band) and (ob_spaxel==spaxel):
                # combine the exposures within each OB
                exp_list = glob.glob(obs_dir+'/eris_ifu_jitter_dar_cube_[0-9]*.fits')
                # check the arcfile name not in the excludes file list
                for fi in exp_list:
                    with fits.open(fi) as hdu:
                        arcfile = hdu['PRIMARY'].header['ARCFILE']
                        if excludes is not None:
                            if arcfile in excludes:
                                continue
                        image_list.append(fi)
                        image_arcname_list.append(arcfile)
                        image_exp_list.append(float(ob_exptime))

    if len(image_list) < 1:
        logging.warning(f"no valid data for {target} with {band},{spaxel}")
        return
    # <TODO>: how to align different OBs
    total_exp = np.sum(image_exp_list)
    logging.info("combining images from:")
    with open(os.path.join(outdir,
              f'{target}_{band}_{spaxel}_{total_exp/3600:.1f}h_{suffix}_list.txt'), 'w+') as fp:
        n_combine = len(image_list)
        for i in range(n_combine):
            im = image_list[i]
            im_exptime = image_exp_list[i]
            fp.write(f"{im} {im_exptime}\n")
    if savefile is None:
        savefile = os.path.join(outdir, 
                    f'{target}_{band}_{spaxel}_{total_exp/3600:.1f}h_{suffix}.fits')
    obs_offset = compute_eris_offset(image_list, additional_drifts=drifts)
    if recipe == 'combine_eris_cube':
        combine_eris_cube(cube_list=image_list, pixel_shifts=obs_offset, 
                          z=z, overwrite=overwrite,
                          savefile=savefile)
    elif recipe == 'eris_ifu_combine':
        # create the combine.sof
        combine_sof = os.path.join(outdir, 
                    f'{target}_{band}_{spaxel}_{total_exp/3600:.1f}h_{suffix}_combine.sof')
        if not os.path.isfile(combine_sof):
            with open(combine_sof, 'w+') as fp:
                for img in image_list:
                    fp.write(f"{img} OBJECT_CUBE\n")
        # save the offset into text file
        combine_offset_file = os.path.join(outdir, 
                    f'{target}_{band}_{spaxel}_{total_exp/3600:.1f}h_{suffix}_offset.list')
        if not os.path.isfile(combine_offset_file):
            with open(combine_offset_file, 'w+') as fp:
                for ofs in obs_offset:
                    fp.write(f"{ofs[0]} {ofs[1]}\n")
        subprocess.run([*esorex_cmd_list, 'eris_ifu_combine',
                        '--ks_clip=TRUE', '--kappa=3.0',
                        '--offset_mode=FALSE', f'--name_i={combine_offset_file}',
                        f'--name_o={savefile}', combine_sof])
    logging.info(f'Found the combined cube: {savefile}')


#####################################
######## helper functions ###########

def read_eris_cube(cubefile, z=None, rest_wave=0.656279*u.um, header_ext='DATA',):
    """read eris datacube and convert the wavelength to velocity relative to Ha
    """
    with fits.open(cubefile) as hdu:
        header = hdu[header_ext].header
        cube = hdu[header_ext].data
    header = fix_micron_unit_header(header)
    wavelength = get_wavelength(header, output_unit='um')
    if z is not None:
        refwave = (rest_wave * (1+z)).to(u.um).value
        velocity = 299792.458 * (wavelength-refwave)/refwave
        return velocity, cube, header
    return wavelength, cube, header

def fix_micron_unit_header(header):
    """this small program fix the unrecongnized unit "micron" by astropy.wcs

    *only tested with the 3D datacube from VLT/ERIS
    """
    if 'CUNIT3' in header:
        if header['CUNIT3'] == 'MICRON':
            header['CUNIT3'] = 'um'
    return header

def get_wavelength(header=None, wcs=None, output_unit='um'):
    if header is None:
        try:
            header = wcs.to_header()
            header['NAXIS3'],header['NAXIS2'],header['NAXIS1'] = wcs.array_shape
        except:
            raise ValueError("Please provide valid header or wcs!")
    if 'PC3_3' in header.keys():
        cdelt3 = header['PC3_3']
    elif 'CD3_3' in header.keys():
        cdelt3 = header['CD3_3']
    else: 
        cdelt3 = header['CDELT3']
    # because wcs.slice with change the reference channel, the generated ndarray should 
    # start with 1 
    chandata = (header['CRVAL3'] + (np.arange(1, header['NAXIS3']+1)-header['CRPIX3']) * cdelt3)
    if 'CUNIT3' in header.keys():
        chandata = chandata*units.Unit(header['CUNIT3'])
        wavelength = chandata.to(units.Unit(output_unit)).value
    else:
        wavelength = chandata
    return wavelength

def start_logger():
    logging.basicConfig(filename='myapp.log', encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.INFO)
    logging.info('Started')
    pass
    logging.info('Finished')


#####################################
########## CMD wrapper ##############

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='eris_jhchen_utils.py',
            description="Welcome to jhchen's ERIS utilities v{}".format(__version__),
            epilog='Reports bugs and problems to jhchen@mpe.mpg.de')
    parser.add_argument('--esorex', type=str, default='esorex',
                        help='specify the customed esorex')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    parser.add_argument('--logfile', help='the logging output file')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')

    # add subparsers
    subparsers = parser.add_subparsers(title='Available task', dest='task', 
                                       metavar=textwrap.dedent(
        '''
          * request_calib: search and download the raw calibration files
          * request_science: download the science data
          * generate_metadata: generate metadata from downloaded data
          * generate_calib: generate the calibration files
          * auto_jitter: run jitter recipe automatically
          * combine_data: combine the reduced data

          Quick tools:

          * get_daily_calib: quick way to get dalily calibration files
          * run_eris_pipeline: quickly reduce science data with raw files
          * quick_combine: quick combine reduced science data

          To get more details about each task:
          $ eris_jhchen_utils.py task_name --help
        '''))

    ################################################
    # request_calib
    subp_request_calib = subparsers.add_parser('request_calib',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            Search and download the required calib files
            --------------------------------------------
            Examples:

              # request all the calibration data
              eris_jhchen_utils request_calib --start_date 2023-04-09 --band K_low --spaxel 100mas --exptime 600 --outdir ./raw --metafile raw/2023-04-09.metadata.csv

              # requst the calibration data for dark and detlin
              eris_jhchen_utils request_calib --steps dark detlin --start_date 2023-04-09 --band K_low --spaxel 100mas --exptime 600 --outdir ./raw --metafile raw/2023-04-09.metadata.csv

            '''))
    subp_request_calib.add_argument('--start_date', type=str, help='The starting date of the observation, e.g. 2023-03-08')
    subp_request_calib.add_argument('--end_date', type=str, help='The finishing date of the observation, e.g. 2023-03-08')
    subp_request_calib.add_argument('--steps', type=str, nargs='+', 
        help="Calibration steps, can be combination of: 'dark','detlin','distortion','flat','wavecal'",
                                     default=['dark','detlin','distortion','flat','wavecal'])
    subp_request_calib.add_argument('--band', type=str, help='Observing band')
    subp_request_calib.add_argument('--exptime', type=int, help='Exposure time')
    subp_request_calib.add_argument('--spaxel', type=str, help='Spatia pixel size')
    subp_request_calib.add_argument('--outdir', type=str, help='Output directory',
                                    default='raw')
    subp_request_calib.add_argument('--metafile', type=str, help='Summary file')
    subp_request_calib.add_argument('--max_days', type=int, help='Maximum searching days before and after the observing day.', default=40)
    subp_request_calib.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    subp_request_calib.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    
    
    ################################################
    # request_science
    subp_request_science = subparsers.add_parser('request_science',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            search and download the science data
            ------------------------------------
            example:
            
            eris_jhchen_utils request_science --user username --prog_id 111.255U.002 --outdir science_raw --metafile science_raw/sience_metadata.csv

                                        '''))
    subp_request_science.add_argument('--start_date', type=str, help='The starting date of the observation. Such as 2023-03-08', default='2022-04-01')
    subp_request_science.add_argument('--band', type=str, help='Observing band', default='')
    subp_request_science.add_argument('--spaxel', type=str, help='Spatial pixel resolution', default='')
    subp_request_science.add_argument('--target', type=str, help='The name of the target', default='')
    subp_request_science.add_argument('--exptime', type=str, help='Integration time', default='')
    subp_request_science.add_argument('--username', type=str, help='The user name in ESO User portal.', default='')
    subp_request_science.add_argument('--password', type=str, help='The password in ESO User Eortal.')
    subp_request_science.add_argument('--outdir', type=str, help='Output directory')
    subp_request_science.add_argument('--prog_id', type=str, help='Program ID', default='')
    subp_request_science.add_argument('--ob_id', type=str, help='Observation ID', default='')
    subp_request_science.add_argument('--metafile', type=str, help='Summary file',default='metadata.csv')
    subp_request_science.add_argument('--end_date', type=str, help='The finishing date of the observation. Such as 2023-03-08', default='')
    subp_request_science.add_argument('--dp_type', type=str, help='The type of the observation, set to "%PSF-CALIBRATOR" to download only the PSF star', default='')
    subp_request_science.add_argument('--archive', action='store_true', 
                                      help='Organise the date based on the observing date')
    subp_request_science.add_argument('--uncompress', action='store_true', 
                                      help='Uncompress the fits.Z files')


    ################################################
    # search_static_calib
    subp_search_static_calib = subparsers.add_parser('search_static_calib',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            search for the static calibration files
            -----------------------------------------------
            example:

                eris_jhchen_utils search_static_calib --esorex esorex
                                        '''))

    subp_search_static_calib.add_argument('--esorex', type=str, default='esorex',
                                          help='The callable esorex command')
 

    ################################################
    # generate_metadata
    subp_generate_metadata = subparsers.add_parser('generate_metadata',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            generate the metadata file from downloaded data
            -----------------------------------------------
            example:

                eris_jhchen_utils generate_metadata --header_dir science/2023-12-06/headers --metafile metadata/2023-12-06/metadata.csv
                
                eris_jhchen_utils generate_metadata --data_dir science/2023-12-06 --extname DATA --metafile metadata/2023-12-06/metadata.csv
                                        '''))

    subp_generate_metadata.add_argument('--data_dir', type=str, help='The directory with all the downloaded files, including all the *.fits.Z or *.fits')
    subp_generate_metadata.add_argument('--extname', type=str, help='The extension or card name of the targeted data in fits file', default='Primary')
    subp_generate_metadata.add_argument('--header_dir', type=str, help='The directory with all the processed headers, header files end with *.hdr')
    subp_generate_metadata.add_argument('--metafile', type=str, help='The output file with all the extracted informations from the fits headers')
    subp_generate_metadata.add_argument('--overwrite', action='store_true', help='Overwrite exiting metafile if present')

    
    ################################################
    # generate_calib
    subp_generate_calib = subparsers.add_parser('generate_calib',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            generate the required calibration files
            ---------------------------------------
            example:
            
            esorex=~/esorex/bin/esorex
            
            # generate all the calibration files
            eris_jhchen_utils generate_calib --metadata raw/2023-04-09.metadata.csv --raw_pool raw --outdir calibPool
            
            # only the specified step, eg: dark + detlin
            eris_jhchen_utils generate_calib --metadata raw/2023-04-09.metadata.csv --raw_pool raw --outdir calibPool --steps dark detlin

                                        '''))
    subp_generate_calib.add_argument('--metadata', type=str, help='The summary file')
    subp_generate_calib.add_argument('--raw_pool', type=str, help='The directory includes the raw files')
    subp_generate_calib.add_argument('--outdir', type=str, help='The output directory',
                                     default='./')
    subp_generate_calib.add_argument('--esorex', nargs='?', type=str, default='esorex',
                                     help='specify the customed esorex')
    subp_generate_calib.add_argument('--static_pool', type=str, help='The static pool')
    subp_generate_calib.add_argument('--steps', type=str, nargs='+', 
        help="Calibration steps, can be combination of: 'dark','detlin','distortion','flat','wavecal'",
                                     default=['dark','detlin','distortion','flat','wavecal'])
    subp_generate_calib.add_argument('--dark_sof', help='dark sof')
    subp_generate_calib.add_argument('--detlin_sof', help='detector linearity sof')
    subp_generate_calib.add_argument('--distortion_sof', help='distortion sof')
    subp_generate_calib.add_argument('--flat_sof', help='flat sof')
    subp_generate_calib.add_argument('--wavecal_sof', help='wavecal sof')


    ################################################
    # auto_jitter
    subp_auto_jitter = subparsers.add_parser('auto_jitter',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            automatically run the jitter recipe
            -----------------------------------
            Examples:

            eris_jhchen_utils auto_jitter --metadata science_raw/metadata.csv --raw_pool science_raw --calib_pool calibPool/2023-04-09_K_low_100mas_600s --ob_id 3589012 --outdir science_output 

            eris_jhchen_utils auto_jitter --sof manual.sof --raw_pool science_raw --calib_pool calibPool/2023-04-09_K_low_100mas_600s --outdir science_output

                                        '''))
    subp_auto_jitter.add_argument('--metadata', help='The summary file')
    subp_auto_jitter.add_argument('--sof', help='Specify the exisint sof')
    subp_auto_jitter.add_argument('--raw_pool', help='The folder name with all the raw files')
    subp_auto_jitter.add_argument('--outdir', help='The output directory')
    subp_auto_jitter.add_argument('--calib_pool', help='The folder with all the calibration files')
    subp_auto_jitter.add_argument('--static_pool', help='The folder with all the static calibration files')
    subp_auto_jitter.add_argument('--mode', help='The mode of the recipe, can be jitter and stdstar', default='jitter')
    subp_auto_jitter.add_argument('--objname', help='Select only the data with Object=objname')
    subp_auto_jitter.add_argument('--band', help='Select only the data with INS3.SPGW.NAME=band')
    subp_auto_jitter.add_argument('--spaxel', help='Select only the data with INS3.SPXW.NAME=spaxel')
    subp_auto_jitter.add_argument('--exptime', help='Select only the data with DET.SEQ1.DIT=exptime')
    subp_auto_jitter.add_argument('--dpr_tech', help='Select only the data with DPR.TECH=dpr_tech', default='IFU')
    subp_auto_jitter.add_argument('--prog_id', help='Select only the data with Program ID=prog_id')
    subp_auto_jitter.add_argument('--ob_id', help='Select only the data with OB.ID=prog_id')

    ################################################
    # get_daily_calib
    subp_get_daily_calib = subparsers.add_parser('get_daily_calib',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            quickly get the daily calibration files
            ---------------------------------------
            example:
            
              eris_jhchen_utils get_daily_calib -d 2023-04-09 -b K_low -s 100mas -e 600 
                                        '''))
    subp_get_daily_calib.add_argument('-d','--date', help='Observing date')
    subp_get_daily_calib.add_argument('-b','--band', help='Observation band')
    subp_get_daily_calib.add_argument('-s','--spaxel', help='Pixel size')
    subp_get_daily_calib.add_argument('-e','--exptime', help='Exposure time')
    subp_get_daily_calib.add_argument('--outdir', default='./', 
                                      help='Output directory, default ./')
    subp_get_daily_calib.add_argument('--static_pool', default=None, 
                                      help='Static calibration pool directory')
    subp_get_daily_calib.add_argument('--steps', nargs='+', default=None, 
                                      help='Calibration steps to be proceeded')
    subp_get_daily_calib.add_argument('--calib_raw', default=None, 
                                      help='Directory for raw calibration files, default calib_raw')
    subp_get_daily_calib.add_argument('--max_days', default=60, 
                                      help='The maxium days to search for calibration raw files')
    subp_get_daily_calib.add_argument('--overwrite', action='store_true', 
                                      help='Overwrite the existing files')
    subp_get_daily_calib.add_argument('--rename', action='store_true', 
                                      help='Rename output files with detailed configurations')
    


    ################################################
    # run_eris_pipeline
    subp_run_eris_pipeline = subparsers.add_parser('run_eris_pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            quickly reduce the science data
            -------------------------------
            example:

              eris_jhchen_utils run_eris_pipeline -d science_raw
                                        '''))
    subp_run_eris_pipeline.add_argument('-d', '--datadir', 
                                          help='The folder with downloaded science data')
    subp_run_eris_pipeline.add_argument('--outdir', default='./', help='The output folder')
    subp_run_eris_pipeline.add_argument('--calib_pool', default='calibPool',
                                        help='The calibration pool')
    subp_run_eris_pipeline.add_argument('--calib_raw', default='calib_raw', 
                                        help='The raw pool of calibration files')
    subp_run_eris_pipeline.add_argument('--overwrite', action='store_true', 
                                        help='Overwrite the existing files')
    subp_run_eris_pipeline.add_argument('--selected_categories', type=str, nargs='+', 
        help="Selected targets types, can be combination of: SCIENCE, CALIB, ACQUISITION",
                                     default=['SCIENCE','CALIB'])

    
    ################################################
    # quick combine
    subp_quick_combine = subparsers.add_parser('quick_combine',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            quickly reduce the science data
            -------------------------------
            example:

              eris_jhchen_utils quick_combine --datadir science_reduced --target bx482 --band K_middle --spaxel 25mas --drifts drifts_file --suffix test1 --outdir combined
                                        '''))
    subp_quick_combine.add_argument('--datadir', help='The data dierectory')
    subp_quick_combine.add_argument('--target', help='The target name')
    subp_quick_combine.add_argument('--offsets', help='The txt offsets file')
    subp_quick_combine.add_argument('--excludes', help='The excluded files')
    subp_quick_combine.add_argument('--band', help='Observing band')
    subp_quick_combine.add_argument('--spaxel', help='Observing spaxel scale')
    subp_quick_combine.add_argument('--z', type=float, help='The redshift of the target')
    subp_quick_combine.add_argument('--drifts', help='Additional drifts')
    subp_quick_combine.add_argument('--outdir', help='Output dierectory')
    subp_quick_combine.add_argument('--suffix', default='combined', 
                                    help='The suffix of the output files')
    subp_quick_combine.add_argument('--recipe', default='combine_eris_cube', 
                                    help='The combine recipe to be used, set to "eris_ifu_combine" to use the pipeline\'s default combine recipe')
    subp_quick_combine.add_argument('--overwrite', action='store_true', help='Overwrite exiting fits file')


    ################################################
    # start the parser
    ################################################
    args = parser.parse_args()
    ret = None # return status
    
    # set up the logging options
    logging.basicConfig(filename=args.logfile, encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Welcome to eris_jhchen_utils.py v{__version__}")

    if args.logfile is not None:
        esorex = f"{args.esorex} --log-file={args.logfile} --log-level=info"
    else:
        esorex = args.esorex

    if args.debug:
        logging.debug(args)
        func_args = list(inspect.signature(locals()[args.task]).parameters.keys())
        func_str = f"Executing:\n \t{args.task}("
        for ag in func_args:
            try: func_str += f"{ag}={args.__dict__[ag]},"
            except: func_str += f"{ag}=None, "
        func_str += ')\n'
        logging.info(func_str)
        logging.info(f"Using esorex from {args.esorex}")
        logging.info(f"Using static files from {search_static_calib(args.esorex)}")
    if args.task == 'request_calib':
        request_calib(start_date=args.start_date, band=args.band, steps=args.steps,
                      end_date=args.end_date, outdir=args.outdir, exptime=args.exptime, 
                      spaxel=args.spaxel, metafile=args.metafile, 
                      max_days=args.max_days, dry_run=args.dry_run, debug=args.debug)
    elif args.task == 'request_science':
        ret = request_science(prog_id=args.prog_id, ob_id=args.ob_id, 
                              start_date=args.start_date, 
                              target=args.target,
                              username=args.username, password=args.password,
                              band=args.band, spaxel=args.spaxel, 
                              exptime=args.exptime, end_date=args.end_date, 
                              outdir=args.outdir, metafile=args.metafile,
                              dp_type=args.dp_type,
                              archive=args.archive, uncompress=args.uncompress,
                              dry_run=args.dry_run, debug=args.debug)
    elif args.task == 'search_static_calib':
        search_static_calib(args.esorex, debug=args.debug)
    elif args.task == 'generate_metadata':
        generate_metadata(data_dir=args.data_dir, extname=args.extname,
                          header_dir=args.header_dir, metafile=args.metafile,
                          dry_run=args.dry_run, debug=args.debug, 
                          overwrite=args.overwrite)
    elif args.task == 'generate_calib':
        generate_calib(args.metadata, raw_pool=args.raw_pool, 
                       outdir=args.outdir, static_pool=args.static_pool, 
                       steps=args.steps, dark_sof=args.dark_sof, 
                       detlin_sof=args.detlin_sof, distortion_sof=args.distortion_sof,
                       flat_sof=args.flat_sof, wavecal_sof=args.wavecal_sof, 
                       esorex=esorex, dry_run=args.dry_run, debug=args.debug)
    elif args.task == 'auto_jitter':
        ret = auto_jitter(metadata=args.metadata, raw_pool=args.raw_pool, 
                          sof=args.sof,
                          outdir=args.outdir, calib_pool=args.calib_pool, 
                          mode=args.mode, objname=args.objname, band=args.band, 
                          spaxel=args.spaxel, exptime=args.exptime, 
                          dpr_tech=args.dpr_tech, esorex=esorex, 
                          prog_id=args.prog_id, ob_id=args.ob_id, 
                          dry_run=args.dry_run, debug=args.debug)
    # the quick tools
    elif args.task == 'get_daily_calib':
        get_daily_calib(args.date, args.band, args.spaxel, args.exptime, 
                        outdir=args.outdir, steps=args.steps, esorex=esorex, 
                        static_pool=args.static_pool, rename=args.rename,
                        calib_raw=args.calib_raw,
                        max_days=args.max_days, overwrite=args.overwrite, 
                        debug=args.debug, dry_run=args.dry_run)
    elif args.task == 'run_eris_pipeline':
        run_eris_pipeline(args.datadir, outdir=args.outdir, calib_pool=args.calib_pool, 
                          calib_raw=args.calib_raw, esorex=esorex, overwrite=args.overwrite, 
                          selected_categories=args.selected_categories,
                          debug=args.debug, dry_run=args.dry_run)
    elif args.task == 'quick_combine':
        quick_combine(datadir=args.datadir, target=args.target, offsets=args.offsets,
                      excludes=args.excludes, band=args.band, spaxel=args.spaxel,
                      z=args.z, overwrite=args.overwrite, recipe=args.recipe,
                      outdir=args.outdir, drifts=args.drifts, suffix=args.suffix)
    else:
        pass
    if args.debug:
        if ret is not None:
            logging.info(ret)
    logging.info('Finished')
