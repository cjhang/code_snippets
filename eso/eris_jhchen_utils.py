#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

This program was initially written when I learnt how to reduce ESO/ERIS 
data for the first time. 

History:
    - 2023-11-22: first release, v0.1
    - 2024-01-04: add cmd interface, v0.2
    - 2024-01-11: add quick tools, v0.3
    - 2024-01-30: add customized combining task, v0.4
    - 2024-02-26: support reducing PSF and standard stars, v0.5
    - 2024-04-07: add data quicklook tools, v0.6
    - 2024-08-15: add support for drifts correction, v0.7
    - 2024-09-05: add support for flux calibration, v0.8
"""
__version__ = '0.8.11'

# import the standard libraries
import os 
import pathlib
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
from scipy import ndimage, optimize, stats, interpolate
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
from matplotlib.backends.backend_pdf import PdfPages

# filtering the warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
from astropy.utils.exceptions import AstropyWarning

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
                filename_match = re.compile('filename="(?P<filename>.+.fits.Z)"')
                filename = filename_match.search(r.headers['Content-Disposition']).groupdict()['filename']
                if debug:
                    print("Content-Disposition is: {}".format(filename))
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
    # try to sort the metadata based on their DP.ID
    try: metadata.sort(['DP.ID'])
    except: pass
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

def download_eris(eris_query_tab, outdir='raw', metafile=None, username=None, auth=None,
                  debug=False):
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
        download_file(file_url, outdir=outdir, auth=auth, debug=debug)
    if metafile is not None:
        save_metadata(eris_query_tab, metafile=metafile)

def eris_quary(ob_id='', prog_id='',
               start_date='2022-04-01', end_date='2094-10-06', 
               exptime='', dp_type='', band='', spaxel='', target='',
               debug=False):
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    eso = Eso()
    eso.ROW_LIMIT = -1
    eso.clear_cache()
    # cache_dir = pathlib.Path(eso.cache_location)
    # for cache_file in pp.iterdir():
        # pathlib.Path.unlink(cache_file)
    # for cache_file in os.listdir(cache_dir):
        # os.remove(os.path.join(cache_dir, cache_file))
    column_filters={'ob_id': ob_id,
                    'prog_id':prog_id,
                    'stime': start_date,
                    'exptime': exptime,
                    'etime': end_date,
                    'dp_type': dp_type, 
                    'ins3_spgw_name': band,
                    'ins3_spxw_name': spaxel,
                    'obs_targ_name': target,}
    if debug:
        for key, item in column_filters.items():
            print(f"{key}: {item}")
    eris_query_tab = eso.query_instrument('eris', column_filters=column_filters)
    return eris_query_tab

def eris_auto_quary(start_date, end_date=None, start_time=9, end_time=9, max_days=60, 
                    column_filters={}, dry_run=False, debug=False, **kwargs):
    """query ESO/ERIS raw data from the database

    Args:
        start_date (str): the starting date: like 2023-04-08
        end_date (str): the same format as start_date, the default value is start_date + 1day
        start_time (int): the starting hour, it is set to 9am (UT), which is 12:00 in Chile
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
    eso.clear_cache()
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
                    if 'DARK' in dptype:
                        n_dark = np.sum(tab_eris['DPR.TYPE'] == 'DARK')
                        if n_dark < 3:
                            matched = 0
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
                        if (n_ds_fiber!=1) or (n_ds_fiber_lamp!=n_ds_fiber_dark) or (n_ds_wave_lamp<1) or (n_ds_wave_dark<1):
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
                  outdir='calib_raw', end_date=None, dpcat='CALIB', arm='SPIFFIER', 
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
        if debug:
            print(step_query)
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
            download_eris(all_tabs, metafile=metafile, outdir=outdir, debug=debug)
        else:
            logging.warning("No files for downloading!")

def request_science(prog_id='', metafile='metadata.csv',
                    username=None, password=None, auth=None, 
                    outdir=None, target='', ob_id='', exptime='',
                    start_date='2022-04-01', end_date='2094-10-06', debug=False, 
                    dp_type='', band='', spaxel='',
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
        band (str): the observing band, like 'H_middle', 'K_low'
        spaxel (str): the spaxel resolution, like '250mas', '100mas', '25mas'
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
    eso.clear_cache()
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
                    'ins3_spgw_name': band,
                    'ins3_spxw_name': spaxel,
                    'obs_targ_name': target,}
    if debug:
        for key, item in column_filters.items():
            print(f"{key}: {item}")
    eris_query_tab = eso.query_instrument('eris', column_filters=column_filters)
    if len(eris_query_tab)>1:
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
    """generate metafile from download files to the same format as the online queries

    Args:
        data_dir (str): the directory include the fits file
    """
    # colnames
    colnames = ['Release Date', 'Object', 'RA', 'DEC','Program ID', 'DP.ID', 'EXPTIME', 
                'OB ID', 'OBS TARG NAME', 'DPR.CATG', 'DPR.TYPE', 'DPR.TECH', 'TPL.START', 
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
                binpath_match = re.compile('(?P<install_dir>^.*)/bin/esorex')
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
                tpl_start=None,
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
            meta_tab = meta_tab[meta_tab['OB ID'].astype(type(ob_id)) == ob_id]
        if objname is not None:
            meta_tab = meta_tab[meta_tab['Object'] == objname]
        if band is not None:
            meta_tab = meta_tab[meta_tab['INS3.SPGW.NAME'] == band]
        if spaxel is not None:
            meta_tab = meta_tab[meta_tab['INS3.SPXW.NAME'] == spaxel]
        if exptime is not None:
            meta_tab = meta_tab[abs(meta_tab['DET.SEQ1.DIT']-exptime)<1e-6]
        if prog_id is not None:
            meta_tab = meta_tab[meta_tab['Program ID'] == prog_id]
        if dpr_tech is not None:
            dpr_tech_select = [True if dpr_tech in item['DPR.TECH'] else False for item in meta_tab]
            meta_tab = meta_tab[dpr_tech_select]
        if dpr_catg is not None:
            meta_tab = meta_tab[meta_tab['DPR.CATG'] == dpr_catg]
        if tpl_start is not None:
            meta_tab = meta_tab[meta_tab['TPL.START'] == tpl_start]

        if len(meta_tab) < 1:
            print(" >> skipped, no valid data found!")
            logging.warning(f"skipped {objname}({ob_id}+{tpl_start}) with {band}+{spaxel}+{exptime}, no valid data")
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
            openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
            openf.write(f"{calib_pool}/eris_ifu_dark_master_dark.fits MASTER_DARK\n")
            openf.write(f"{static_pool}/eris_oh_spec.fits OH_SPEC\n")
            # openf.write(f"{calib_pool}/eris_ifu_distortion_slitlet_pos.fits SLITLET_POS\n")
            # openf.write(f"{static_pool}/EXTCOEFF_TABLE.fits EXTCOEFF_TABLE\n")
            #if (dpr_catg == 'CALIB') and (stdstar_type == 'STD'):
            #    if band in ['H_low', 'J_low', 'K_low']:
            #        openf.write(f"{static_pool}/RESPONSE_WINDOWS_{band}.fits RESPONSE\n")
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

def reduce_eris(metafile=None, datadir=None, outdir=None,
                calib_pool='calibPool', calib_raw=None, static_pool=None,
                categories=['SCIENCE','CALIB'], #ACQUISITION
                esorex='esorex', overwrite=False, 
                debug=False, dry_run=False):
    """ reduce eris data within a folder

    Args:
        metafile: the metafile include all the raw files in the datadir
        datadir: the directory include all the fits or fits.Z files
        outdir: the output directory
        calib_pool: the output directory of the intermediate calibration files
        calib_raw: the directory to store the possible calibration raw files
        catagories: the dpr_catg, possible with any combination of
                    ['SCIENCE','CALIB', 'ACQUISITION']
        esorex: the callable esorex command
        overwrite: overwrite the existing files
        debug: print debuging messages

    """
    datadir = datadir.strip('/')
    if outdir is not None:
        outdir = outdir.strip('/')
    else:
        outdir = '.'

    ## Step-1
    # generate all the summary file if there is none
    if os.path.isfile(metafile):
        logging.info(f"> using existing metadata:{metafile}")
        metadata = read_metadata(metafile)
    else:
        logging.info(f"> generating the metadata of {datadir}")
        # if os.path.isdir(date_folder + '/headers'):
            # metadata = generate_metadata(header_dir=date_folder+'/headers', 
                                         # metafile=date_metafile)
        # else:
            # with tempfile.TemporaryDirectory() as tmpdir:
                # metadata = generate_metadata(data_dir=date_folder, work_dir=tmpdir, 
                                             # metafile=date_metafile)
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = generate_metadata(data_dir=datadir, work_dir=tmpdir, 
                                         metafile=metafile)
    ## Step-2
    # read available data 
    logging.info(f"> reducing the science data in {datadir}")

    dpr_tech_select = [True if 'IFU' in item['DPR.TECH'] else False for item in metadata]
    metadata = metadata[dpr_tech_select]
    if len(metadata) < 1:
        logging.info(f"> No ERIS/SPIFFIER data found on {date}")
        return

    # group the data based on their TPL.START
    unique_tpl_starts = np.unique(metadata['TPL.START'])
    
    for tpl_time in unique_tpl_starts:
        date = tpl_time[:10]
        tpl_metadata = metadata[metadata['TPL.START'] == tpl_time]
        # get the ob id
        if 'OB.ID' in tpl_metadata.colnames:
            ob_id = tpl_metadata['OB.ID'][0]
        elif 'OB ID' in tpl_metadata.colnames:
            ob_id = tpl_metadata['OB ID'][0]

        # get the name of the science target, here we assume each ob has only one target
        target = None
        try:
            target = tpl_metadata[tpl_metadata['DPR.CATG']=='SCIENCE']['OBS TARG NAME'][0]
        except:
            try:
                target = tpl_metadata[tpl_metadata['DPR.CATG']=='CALIB']['OBS TARG NAME'][0]
            except:
                logging.warning(f'> Skip OB:{ob_id} on {tpl_time}')
                logging.warning(f'>> faild in finding the target name!')
                continue
        # get all the DPR catagories: SCIENCE, ACQUISITION, CALIB (PSF_CALIBRATOR + STD)
        tpl_cats_unique = np.unique(tpl_metadata['DPR.CATG'])
        logging.info(f'Found DPR catagories: {",".join(tpl_cats_unique)}')
        # read all the data with different catagories
        for cat in tpl_cats_unique:
            if cat not in categories:
                continue
            cat_metadata = tpl_metadata[tpl_metadata['DPR.CATG']==cat]
            cat_exptime_list = np.unique(
                    cat_metadata['DET.SEQ1.DIT']).astype(int).tolist()
            logging.info(f'{cat} recieve exptime with: {cat_exptime_list}s')
            for exptime in cat_exptime_list:
                stdstar_type = ''
                exp_metadata = cat_metadata[
                        abs((cat_metadata['DET.SEQ1.DIT']-exptime))<1e-6]
                if len(exp_metadata) < 1:
                    logging.warning(f'{cat} recieve no real exptime with {exptime}s! Skip...')
                    continue
                first_meta = exp_metadata[0]
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
                tpl_cat_outdir = os.path.join(outdir, f'{target}_{cat}{stdstar_type}_{ob_id}_{tpl_time}_{band}_{spaxel}_{exptime}s')
                # count the OBJ and SKY frames <TODO> to get the expected_output
                dpr_is_sky = np.array([True if 'SKY' in item else False for item in exp_metadata['DPR.TYPE']])
                n_sky = np.sum(dpr_is_sky)
                n_obj = np.sum(~dpr_is_sky)
                n_total = n_sky + n_obj
                expected_output = [f'eris_ifu_jitter_obj_cube_{n_total-1:0>3d}.fits', 
                                   f'eris_ifu_jitter_sky_cube_{n_total-1:0>3d}.fits',
                                   f'eris_ifu_stdstar_obj_cube_{n_total-1:0>3d}.fits',
                                   f'eris_ifu_stdstar_sky_cube_{n_total-1:0>3d}.fits',
                                   f'eris_ifu_stdstar_std_cube_{n_total-1:0>3d}.fits',
                                   f'eris_ifu_stdstar_psf_cube_{n_total-1:0>3d}.fits',
                                   ]
                for eo in expected_output:
                    if os.path.isfile(os.path.join(tpl_cat_outdir, eo)):
                        if not overwrite:
                            logging.info(f"> Done: {date}:{target} on {tpl_time})")
                            logging.info(f">>      with {ob_id}+{cat}+{band}+{spaxel}+{exptime}s")
                            is_reduced = True
                if is_reduced:
                    continue
                # check whether the cat of the OB has been tried but failed
                if os.path.isfile(os.path.join(tpl_cat_outdir, 'failed')):
                    if not overwrite:
                        logging.info(f"> Skip failure: {target} with TPL.START={tpl_time}")
                        logging.info(f">> with {ob_id}+{cat}+{band}+{spaxel}+{exptime}s")
                        continue
                
                ## Step-3
                # generate the calibPool
                logging.info(f"> generating calibPool for {date} with {cat}+{band}+{spaxel}+{exptime}s")
                if not dry_run:
                    calib_pool_tpl = get_daily_calib(
                            date, band, spaxel, exptime, esorex=esorex, 
                            outdir=calib_pool, calib_raw=calib_raw,
                            overwrite=overwrite)
                # except:
                    # logging.warning(f"> Error found in geting the calibPool of {date}: {target}(OB.ID={ob_id}) with {band}+{spaxel}+{exptime}s")

                ## Step-4
                # run eris_ifu_gitter on science target and acquisition/psf stars
                logging.info(f">> with {ob_id}+{cat}+{band}+{spaxel}+{exptime}s")
                if dry_run:
                    continue
                try:
                    auto_jitter(metadata=exp_metadata, 
                                raw_pool=datadir, 
                                outdir=tpl_cat_outdir, 
                                calib_pool=calib_pool_tpl, 
                                static_pool=static_pool,
                                tpl_start=tpl_time, ob_id=ob_id, 
                                esorex=esorex, dpr_catg=cat, 
                                stdstar_type=stdstar_type, product_depth=2,
                                dry_run=dry_run)
                except:
                    # subprocess.run(['rm','-rf', daily_ob_outdir])
                    # subprocess.run(['touch', daily_ob_outdir+'/failed'])
                    logging.warning(f"> Error: found in runing {date}: {target} on {tpl_time}")
                    logging.warning(f">>       with {ob_id}+{band}+{spaxel}+{exptime}s")


#####################################
######### Flux calibration ##########

def query_star_VizieR(name, band=None):
    """get the star info from VizieR database

    """
    # return Tstar, magnetude, type
    pass

def gaussian_2d_legacy(params, x=None, y=None):
    """a simple 2D gaussian function
    
    Args:
        params: all the free parameters
                [amplitude, x_center, y_center, x_sigma, y_sigma, beta]
        x: x grid
        y: y grid
    """
    amp, x0, y0, xsigma, ysigma, beta = params
    return amp*np.exp(-0.5*(x-x0)**2/xsigma**2 - beta*(x-x0)*(y-y0)/(xsigma*ysigma)- 0.5*(y-y0)**2/ysigma**2)

def gaussian_2d(params, x=None, y=None):
    """a simple 2D gaussian function
    
    Args:
        params: all the free parameters
                [amplitude, x_center, y_center, x_sigma, y_sigma, theta]
        x: x grid
        y: y grid
    """
    amp, x0, y0, xsigma, ysigma, theta = params
    a = np.cos(theta)**2/(2*xsigma**2) + np.sin(theta)**2/(2*ysigma**2)
    b = np.sin(2*theta)/(2*xsigma**2) - np.sin(2*theta)/(2*ysigma**2)
    c = np.sin(theta)**2/(2*xsigma**2) + np.cos(theta)**2/(2*ysigma**2)
    return amp*np.exp(-a*(x-x0)**2 - b*(x-x0)*(y-y0) - c*(y-y0)**2)

def moffat_2d(params, x=None, y=None):
    """a simple 2d moffat function, used for std star fitting
    
    Args:
        params: [amplitude, x_center, y_center, gamma, alpha]
        x: x grid
        y: y grid
    """
    amp, x0, y0, gamma, alpha = params
    return amp*(1+((x-x0)**2+(y-y0)**2)/gamma**2)**(-alpha)

def airy_2d(params, x=None, y=None):
    """a simple 2D airy function, used for star fitting
    Args:
        params: [amplitude, x_center, y_center, radius]
        x: x grid
        y: y grid
    """
    from scipy.special import j1, jn_zeros
    amplitude, x_center, y_center, radius = params
    rz = jn_zeros(1, 1)[0] / np.pi
    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2) / (radius / rz)
    z = np.ones(r.shape)
    rt = np.pi * r[r > 0]
    z[r > 0] = (2.0 * j1(rt) / rt) ** 2
    z *= amplitude
    return z

def fit_star(starfits, x0=None, y0=None, pixel_size=1, 
             plot=False, plotfile=None, model='gaussian_2d',
             extract_spectrum=False, interactive=False, 
             outfile=None, negative_signal=False):
    """two dimentional Moffat/Gaussian fit

    """
    # read the fits file
    with fits.open(starfits) as hdu:
        header = hdu['PRIMARY'].header
        data_header = hdu['DATA'].header
        data = hdu['DATA'].data
        ndim = data.ndim
        exptime = header['EXPTIME']
        arcfile = header['ARCFILE']
    if ndim == 2:
        # it is already a map
        image = data
    elif data.ndim == 3:
        wavelength = get_wavelength(header=data_header)
        nchan = len(data)
        # collapse the cube to make map
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            data = astro_stats.sigma_clip(data, sigma=5, axis=0).filled(0)
        # collapse the cube to make map
        data_median_filtered = ndimage.median_filter(data, size=20, axes=0)
        image = np.nansum(data_median_filtered, axis=0)

        # collapse the cube to get the image of the star
        # this is used to find the star and get its shape (by 2d-Gaussian fitting)
        # data = clean_cube(data, median_filter=False, median_subtract=False, sigma=10)
        # image = np.ma.sum(data, axis=0)
    if negative_signal:
        data *= -1.
        image *= -1.
    yshape, xshape = image.shape
    margin_padding = 10
    norm_scale = 2*np.percentile(image[margin_padding:-margin_padding,
                                         margin_padding:-margin_padding], 98)
    image_normed = image / np.abs(norm_scale)
    rms = 1

    # start the gaussian fitting for the 2D-image
    # generate the grid
    sigma2FWHM = np.sqrt(8*np.log(2))
    center_ref = np.array([xshape*0.5, yshape*0.5])
    xgrid, ygrid = np.meshgrid((np.arange(0, xshape) - center_ref[0])*pixel_size,
                               (np.arange(0, yshape) - center_ref[1])*pixel_size)
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    vmax = np.nanpercentile(image_normed, 99)
    vmin = np.nanpercentile(image_normed, 1)

    if model == 'moffat_2d':
        model_func = moffat_2d
        gamma0 = pixel_size
        alpha0 = 1
    elif model == 'gaussian_2d':
        model_func = gaussian_2d
        xsigma, ysigma = pixel_size, pixel_size
    elif model == 'airy_2d':
        model_func = 'airy_2d'

    if interactive:
        # get the initial guess from the user
        plt.figure()
        plt.imshow(image_normed, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.title('Please click on the star')
        plt.colorbar()
        plt.show(block=False)
        xclick, yclick = plt.ginput(1)[0]#, timeout=5
        plt.close()
        if model == 'gaussian_2d':
            p_init = [image_normed[int(yclick), int(xclick)], 
                      xclick-center_ref[0], yclick-center_ref[1], xsigma, ysigma, 0]
        elif mode == 'moffat_2d':
            p_init = [image_normed[int(yclick), int(xclick)], xclick-center_ref[0], 
                      yclick-center_ref[1], gamma0, alpha0]
        elif mode == 'airy_2d':
            p_init = [image_normed[int(yclick), int(xclick)], xclick-center_ref[0], 
                      yclick-center_ref[1], gamma0, alpha0]
        
    # automatically estimate the initial parameters
    else:
        amp = 1.2*vmax #np.nanmax(image_normed[10:-10,10:-10])
        #amp = 1
        # yidx, xidx = np.where(image>0.9*amp)
        # x0, y0 = np.median(xidx), np.median(yidx)
        if x0 is not None:
            x0 = x0
        else: x0 = 0
        if y0 is not None:
            y0 = y0
        else: y0 = 0
        # y0, x0 = 0., 0.#center_ref[0], center_ref[1]
        if model == 'gaussian_2d':
            p_init = [amp, x0, y0, xsigma, ysigma, 0]
        elif model == 'moffat_2d':
            p_init = [amp, x0, y0, gamma0, alpha0]
    # print(f"p_init: {p_init}")
    # plt.figure()
    # plt.imshow(rgrid)
    
    if model == 'gaussian_2d':
        bounds = [[0, 100], [-30,30], [-30,30], [0.1,5], [0.1,5],
                  [-np.pi*0.5,np.pi*0.5]]
    elif model == 'moffat_2d':
        bounds=[[0, 10], [-30,30], [-30,30], [0.001,100], [0.01,100]]
    

    def _cost(params, xgrid, ygrid):
            # return np.sum((image_normed - gaussian_2d(params, xgrid, ygrid))**2/rms**2)
        return np.sum((image_normed - model_func(params, xgrid, ygrid))**2/(rgrid**2 + (15+rms)**2))
       # return np.sum((image_normed - gaussian_2d(params, xgrid, ygrid))**2/(rgrid**2 + rms**2))
    res_minimize = optimize.minimize(_cost, p_init, args=(xgrid, ygrid), method='L-BFGS-B',
                        bounds=bounds)

    if model == 'moffat_2d':
        amp_fit, x0_fit, y0_fit, gamma_fit, alpha_fit = res_minimize.x
    elif model == 'gaussian_2d':
        amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = res_minimize.x
        xfwhm_fit, yfwhm_fit = xsigma_fit*sigma2FWHM, ysigma_fit*sigma2FWHM
    bestfit_params = res_minimize.x
    # print(f"p_fit: {bestfit_params}")
   
    if extract_spectrum and (ndim == 3):
        # get all the best fit value
        aperture_size = 1.5*np.mean([xfwhm_fit, yfwhm_fit])
        #aperture_correction = 1.000006565#1.0374 for 1*fwhm
        aperture_correction = 1.0533
        aperture_correction = 1.12
        aperture = EllipticalAperture([x0_fit+center_ref[0], y0_fit+center_ref[1]], 
                                      aperture_size, aperture_size, theta=0)
        aper_mask = aperture.to_mask().to_image([xshape, yshape]).astype(bool)
        aper_mask_3D = np.repeat(aper_mask[None,:,:], nchan, axis=0)
        data_selected = data[aper_mask_3D]
        spectrum = aperture_correction*np.sum(data_selected.reshape((nchan, np.sum(aper_mask))), axis=1)
        spectrum = spectrum / exptime # units in adu/second
        output = spectrum
        output_unit = 'adu/s'
        # output_unit = 'adu'
    else: aperture = None
    if plot:
        fit_image = model_func(res_minimize.x, xgrid, ygrid)
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0,0])
        im = ax1.imshow(image_normed, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax1)
        ax1.plot([x0]+center_ref[0], [y0]+center_ref[1], 'x', color='red')
        ax2 = fig.add_subplot(gs[0,1])
        im = ax2.imshow(fit_image, origin='lower',)# vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax2)
        ax3 = fig.add_subplot(gs[0,2])
        im = ax3.imshow(image_normed - fit_image, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax3)
        if aperture is not None:
            aperture.plot(ax=ax1, color='cyan')
            aperture.plot(ax=ax2, color='cyan')
            aperture.plot(ax=ax3, color='cyan')
        if extract_spectrum:
            ax4 = fig.add_subplot(gs[1,:])
            ax4.plot(wavelength, output)
            ax4.set_xlabel('wavelength [um]')
            ax4.set_ylabel(output_unit)
        if plotfile is not None:
            plotfile.savefig(fig, bbox_inches='tight')
            plt.close()
        if plot:
            plt.show()
        else:
            plt.close()

    if extract_spectrum:
        if outfile is not None:
            np.savetxt(outfile, np.vstack([wavelength, output]).T, delimiter=',')
                       #header=wavelength.unit.to_string()+','+output_unit)
        return wavelength, spectrum
    else:
        return bestfit_params

def scale_blackbody(blackbody_func, band, magnitude, filter_file):
    """scale the blackbody function to the correct flux values
    """
    wave_filter, transmission_filter = read_filter(filter_file)
    dwave = np.zeros_like(wave_filter)
    dwave[:-1] = np.diff(wave_filter)
    dwave[-1] = dwave[-2]
    # normalised the blackbody
    blackbody_filter = blackbody_func(wave_filter)
    flux_bb = np.sum(blackbody_filter*transmission_filter*dwave)/np.sum(
                     transmission_filter*dwave)
    flux_2mass = magnitude2flux_2mass(magnitude, band)
    flux_2mass = flux_2mass.to(u.W/u.m**2/u.um)
    bb_zero_point = flux_2mass / flux_bb # in unit of 1/sr
    return bb_zero_point

def read_filter(filter_file):
    filter_data = np.loadtxt(filter_file)
    wave, transmission = filter_data[:,0], filter_data[:,1]
    return wave, transmission

def filter_photometry(spectrum, filter_profile, filter_name=None):
    wave_spec, data = spectrum
    wave_filter, transmission_filter = filter_profile
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
    if band == 'Ks' or band == 'K':
        flux0 = 4.283e-7 # erg/s/cm2/micron #
    elif band == 'J':
        flux0 = 3.129e-7
    elif band == 'H':
        flux0 = 2.843e-7
    f = magnitude / (-2.5) + np.log10(flux0)
    return 10**f * u.erg / (u.cm * u.cm * u.s * u.um)

def get_corrections(wavelength, spectrum, T_star, band_2mass=None, magnitude_2mass=None,
                   outfile=None, plot=False, plotfile=None, static_datadir=None):
    """this is the combined function for get_transmission and get_zero_point 
    Args:
        wavelength: the wavelength of the stdstar
        spectrum: the extracted spectrum of the stdstar
        T_star: the star temperature
        band: the observed band, 2mass band
        magnitude: the magnetude of the obsered band

    check `get_transmission` and `get_zero_point` for detailed explanation
    """
    spec_corrected = np.copy(spectrum)
    dwave = np.zeros_like(wavelength)
    dwave[:-1] = np.diff(wavelength)
    dwave[-1] = dwave[-2]
    # remove the spectrum feature of the stars
    from astropy.modeling import fitting, models
    obsorption_lines = [1.2820, 2.166,] #[Paschen_Beta, Brackett_gamma, ]
    line_width = 0.015

    if plot:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
    for line in obsorption_lines:
        if (wavelength[0] < line) & (wavelength[-1]>line):
            line_mask = (wavelength > line - line_width) & (wavelength < line + line_width)
            spec_select = spec_corrected[line_mask]
            cont_line = np.median(spec_select)
            amp_absorb = np.min(spec_select) - cont_line
            line_model = models.Lorentz1D(amplitude=amp_absorb, x_0=line, fwhm=dwave[0]) + models.Linear1D(slope=0, intercept=cont_line)
            line_fit = fitting.LevMarLSQFitter()
            line_fitted = line_fit(line_model, wavelength[line_mask], spec_select)
            if plot:
                ax[0].plot(wavelength, spectrum)
                ax[0].plot(wavelength[line_mask], line_fitted(wavelength[line_mask]), label='fitted stellar obsorption line')
                ax[0].legend()
            absorb_line = models.Lorentz1D(amplitude=line_fitted.amplitude_0, x_0=line_fitted.x_0_0, 
                                           fwhm=line_fitted.fwhm_0)
            spec_corrected[line_mask] = spec_corrected[line_mask] - absorb_line(wavelength[line_mask])
            if plot:
                ax[0].plot(wavelength[line_mask], absorb_line(wavelength[line_mask]), 'r')

    blackbody_star = models.BlackBody(temperature=T_star*u.K, 
                                      scale=1.0*u.W/u.m**2/u.um/u.sr)
    transmission = spec_corrected/blackbody_star(wavelength*u.um) #* bb_zero_point)
    transmission_norm = np.sum(transmission*dwave)/np.sum(dwave)
    transmission_normed = transmission/transmission_norm
    filter_file = os.path.join(static_datadir, f'2MASS_2MASS.{band_2mass}.dat')
    bb_zero_point = scale_blackbody(blackbody_star, band_2mass, magnitude_2mass, filter_file)
    zero_point = bb_zero_point / transmission_norm
    with np.errstate(divide='ignore', invalid='ignore'):
        correction = zero_point.value / transmission_normed.value
    # extrapolation the correction to the wavelength of the data cube
    correction = np.ma.masked_invalid(correction)
    # mask the abnormal corrections with sigma_clip
    correction = astro_stats.sigma_clip(correction, sigma=10).filled(np.ma.median(correction))
    if plot:
        ax[1].plot(wavelength, transmission_normed, label='normalised transmission')
        ax[1].plot(wavelength, zero_point/correction, 'r', alpha=0.5,)
        ax[1].text(.8,.05, f'zp={zero_point.value:.3e}', horizontalalignment='center', 
                   transform=ax[1].transAxes, fontsize=10)
        ax[1].legend(loc=1)

    if plot:
        if plotfile is not None:
            plotfile.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show(fig)

    if outfile is not None:
        # save the transmission and zero_point
        corr_tab = table.Table([wavelength, transmission_normed, correction], 
                               names=('wavelength','transmission','correction'),
                               meta={'comments':['wavelength in unit of um, correction in units of W/m^2/um/Adu*s', 'Need to be multiplied by the data with units of Adu/s']})
        corr_tab.write(outfile, format='ascii', overwrite=True)
    return zero_point, transmission_normed, correction

def get_transmission(wavelength, spectrum, T_star, **kwargs):
    # get the star spectrum from blackbody
    blackbody_star = models.BlackBody(temperature=T_star*u.K, 
                                      scale=1.0*u.W/u.m**2/u.um/u.sr)
    dwave = np.zeros_like(wavelength)
    dwave[:-1] = np.diff(wavelength)
    dwave[-1] = dwave[-2]
    # spectrum in unit of adu/s, blackbody_star in unit of u.W/u.m2/u.um/u.sr
    transmission = spectrum/blackbody_star(wavelength*u.um) #* bb_zero_point)
    transmission_norm = np.sum(transmission*dwave)/np.sum(dwave)
    # the final transmission should in unit of adu/u.s/(u.W/u.m2/u.um)
    transmission_normed = transmission/transmission_norm
    return transmission_normed

def get_zero_point(wavelength, spectrum, T_star, band_2mass=None, magnetude_2mass=None):
    # get the star spectrum from blackbody
    blackbody_star = models.BlackBody(temperature=T_star*u.K, 
                                      scale=1.0*u.W/u.m**2/u.um/u.sr)
    dwave = np.zeros_like(wavelength)
    dwave[:-1] = np.diff(wavelength)
    dwave[-1] = dwave[-2]
    # spectrum in unit of adu/s, blackbody_star in unit of u.W/u.m2/u.um/u.sr
    transmission = spectrum/blackbody_star(wavelength*u.um) #* bb_zero_point)
    # the final transmission normalisation should in unit of adu/u.s/(u.W/u.m2/u.um)
    transmission_norm = np.sum(transmission*dwave)/np.sum(dwave)
    # from magnitude we have the flux in unit of u.W/u.m2/u.um
    # so the zero point for blackbody (bb_zero_point) should in unit of 1/u.sr
    bb_zero_point = scale_blackbody(blackbody_star, band_2mass, magnetude_2mass,)
    # calculate the zero point for the observed spectrum
    # should in unit equivalent to u.W/u.m^2/u.um
    zero_point = bb_zero_point / transmission_norm
    return zero_point

def get_telluric_calibration(star_list=None, star_catalogue=None,
                             datadir='science_reduced', outdir='spectral_corrections', 
                             target_types=['CALIBSTD'],# also 'CALIBPSF'
                             static_datadir=None,
                             plot=True,
                             esorex='esorex', overwrite=False, 
                             debug=False, dry_run=False):
    """derive the flux and transmission corrections from the standard telluric stars
    """
    if outdir is not None:
        if not os.path.isdir(outdir):
            os.system(f'mkdir {outdir}')
    else:
        print("Please provide the output directory")
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')
    if star_list is not None:
        if isinstance(star_list, str):
            star_list = np.loadtxt(star_list, dtype=str, delimiter=',')
    if star_catalogue is not None:
        star_catalogue = table.Table.read(star_catalogue, format='csv')
    if star_list is None:
        if datadir is None:
            raise ValueError("Please either provide the list of star fits files or the datadir!")
        datadir = datadir.strip('/')
        star_list,_,_ = search_archive(datadir, target_type='CALIBSTD')
    if False:
        date_list = []
        if datadir is None:
            raise ValueError("Please either provide the list of star fits files or the datadir!")
        for date_folder in os.listdir(datadir):
            if date_matcher.match(date_folder):
                obs_date = date_folder
                for obs in os.listdir(os.path.join(datadir, date_folder)):
                    obs_dir = os.path.join(datadir, date_folder, obs)
                    obs_match = match_obs_folder(obs)
                    if obs_match is None:
                        continue
                    target_star = obs_match['target']
                    exptime = obs_match['exptime']
                    target_type = obs_match['target_type']
                    band = obs_match['band']
                    spaxel = obs_match['spaxel']
                    if target_type not in target_types:
                        continue
                    if star_list is not None:
                        if target_star not in star_list:
                            logging.warning(f'Skip star {target_star}')
                            continue
                    if star_catalogue is not None:
                        if target_star not in star_catalogue['name']:
                            logging.warning(f'Skip star {target_star}, missed in the catalogue.')
                            continue
                        star_info = star_catalogue[star_catalogue['name']==target_star]
                    logging.info(f'Working on {target_star} on {obs_date} with {band}+{spaxel}')
                            
                    if target_type == 'CALIBPSF':
                        ob_exp_list = glob.glob(obs_dir+'/eris_ifu_stdstar_psf_cube_[0-9]*.fits')
                    elif target_type == 'CALIBSTD':
                        ob_exp_list = glob.glob(obs_dir+'/eris_ifu_stdstar_std_cube_[0-9]*.fits')
                    if len(ob_exp_list) > 2:
                        print('Find more than one star datacube. Please check!')
                        print(ob_exp_list)
                    for starfile in ob_exp_list:
                        star_list.append(starfile)
    for starfile in star_list:
        # read the star fits to get the airmass
        print("Working on: {}".format(starfile))
        with fits.open(starfile) as hdu:
            star_header = hdu[0].header
            target_star = star_header['HIERARCH ESO OBS TARG NAME']
            exptime = star_header['EXPTIME']
            band = star_header['HIERARCH ESO INS3 SPGW NAME']
            spaxel = star_header['HIERARCH ESO INS3 SPXW NAME']
            tpl_start = star_header['HIERARCH ESO TPL START'] 
            airmass_start = star_header['HIERARCH ESO TEL AIRM START']
            airmass_end = star_header['HIERARCH ESO TEL AIRM END']
            star_airmass = np.mean([airmass_start, airmass_end])
            star_order = starfile[-8:-5]

        star_info = star_catalogue[star_catalogue['name']==target_star]
        if len(star_info) < 1:
            logging.warning(f'No info found for star {target_star}')
            continue
        star_basenname = f'{target_star}_{tpl_start}_{band}_{spaxel}_airmass{star_airmass:.2f}'
        star_plotfile = os.path.join(outdir, star_basenname+'_qa.pdf')
        star_pdf_plotfile = PdfPages(star_plotfile)
        wave_extract, spec_extract = fit_star(starfile, extract_spectrum=True, 
                                              plot=True, plotfile=star_pdf_plotfile) 
        if 'K' in band: band_2mass = 'Ks'
        elif 'J' in band: band_2mass = 'J'
        elif 'H' in band: band_2mass = 'H'
        try:
            zero_point, transmission_normed, correction = get_corrections(
                    wave_extract, spec_extract, T_star=star_info['T'], band_2mass=band_2mass, 
                    magnitude_2mass=star_info[band_2mass][0], static_datadir=static_datadir,
                    plot=True, plotfile=star_pdf_plotfile)
        except:
            print(f"Errors in getting the correction for star {target_star}")
            continue
        # save the transmission and zero_point as fits file
        if True:
            star_corr_file = os.path.join(outdir, star_basenname+f'_zp{zero_point.value*1e16:.2f}.fits')
            header = fits.Header()
            header['ZP'] = zero_point.value
            header['ZP_UNITS'] = zero_point.unit.to_string()
            header['UNITS'] = (zero_point.unit*(u.s/u.adu)).to_string()
            header['AIRM'] = star_airmass
            col1 = fits.Column(name='wavelength', format='E', array=wave_extract)
            col2 = fits.Column(name='spectrum', format='E', array=spec_extract)
            col3 = fits.Column(name='transmission', format='E', array=transmission_normed)
            col4 = fits.Column(name='correction', format='E', array=correction)
            primary_hdu = fits.PrimaryHDU(header=header)
            bintable_hdu = fits.BinTableHDU.from_columns(
                    [col1, col2, col3, col4], header=header, name='DATA')
            hdul = fits.HDUList([primary_hdu, bintable_hdu])
            hdul.writeto(star_corr_file, overwrite=True)
        star_pdf_plotfile.close()

def match_telluric_corrections(name):
    """match the corrections for each observation
    """
    # target_matcher = re.compile("(?P<target>[\w\s\-\.+]+)_(?P<target_type>(SCIENCE|CALIBPSF|CALIBSTD))_(?P<id>\d{7})_(?P<tpl_start>[\d\-\:T]+)_(?P<band>[JKH]_[\w]{3,6}?)_(?P<spaxel>\d{2,3}mas)_(?P<exptime>\d+)s")
    name_matcher = re.compile("(?P<target>.+)_(?P<tpl_start>[0-9-:T]+)_(?P<band>[JKH]_[a-z]{3,6}?)_(?P<spaxel>[0-9]{2,3}mas)_airmass(?P<airmass>\d+\.\d+)_zp(?P<zp>\d+\.\d+)")
    try:
        corr_match = name_matcher.search(name).groupdict()
    except:
        corr_match = None
    return corr_match

def search_telluric_corrections(datadir, target_name=None, date=None, delta_day=15, band=None,
                                spaxel=None, airmass=None, delta_airmass=0.2, zp_range=[0.5, 3.0],
                                mode=None, ):
    """
    Args:
        datadir: the directory with all the corrections
        mode:
            None: return all the matched correction files
            'closest': return the closest correction in airmass (priority) and observing dates
            'average': average the correction
            'median': use the median value of the correction
            'extrapolation': extrapolation the correction based on the airmass
    """
    correct_files = glob.glob(datadir+'/*.fits')
    matched_corrections = []
    for corrfile in correct_files:
        obs_delta_days = 0
        obs_delta_airmass = 0
        corr_matched = match_telluric_corrections(os.path.basename(corrfile))
        if corr_matched is None:
            continue
        if target_name is not None:
            if corr_matched['target'] != target_name:
                continue
        if band is not None:
            if band != corr_matched['band']:
                continue
        if spaxel is not None:
            if spaxel != corr_matched['spaxel']:
                continue
        if (float(corr_matched['zp']) < zp_range[0]) or (float(corr_matched['zp']) > zp_range[1]):
            continue
        if airmass is not None:
            obs_delta_airmass = np.abs(airmass - float(corr_matched['airmass']))
            if  obs_delta_airmass > delta_airmass:
                continue
        if date is not None:
            obs_delta_seconds = np.abs((np.array(date, dtype='datetime64[s]') 
                              - np.array(corr_matched['tpl_start'], dtype='datetime64[s]')))
            obs_delta_days = obs_delta_seconds.astype(int) / (3600*12)
            if obs_delta_days > delta_day: 
                continue
        matched_corrections.append([corrfile, obs_delta_airmass, obs_delta_days])
    matched_corrections = np.array(matched_corrections)
    matched_corrections = table.Table(matched_corrections, 
                                      names=('filename', 'delta_airmass', 'delta_time'))
    if mode is None:
        return matched_corrections
    if mode == 'closest':
        matched_corrections.sort(['delta_airmass','delta_time'])
        return matched_corrections['filename'][0]
    if mode == 'average':
        pass
    if mode == 'interpolation':
        pass

def auto_spec_correction(fitslist, correction_dir, outdir=None):
    """automatically apply the transimission and zero point correction"""
    fitslist_corrected = []
    with fitsfile in fitslist:
        obs_info = read_eris_header(fitsfile)
        obs_airmass = 0.5*(obs_info['airm_start'] + obs_info['airmass_end'])
        fits_corr = search_telluric_corrections(correction_dir, date=obs_info['tpl_start'], 
                                    band=obs_info['band'], spaxel=obs_info['spaxel'],
                                    airmass=obs_airmass, mode='closest')
        correct_cube = correct_cube(fitsfile, correction=fits_corr, outdir=None)
        fitslist_corrected.append(correct_cube)

def correct_cube(fitscube, wavelength=None, transmission=None, zp=None, correction=None, 
                 suffix='corrected', exptime=None, outdir=None):
    """correct the flux the cube

    fitscube: the datacube to be corrected
    wavelength: the wavelength of the transmission
    transmission: the transmission
    zp: zero point
    correction: the correction file or combined correction with zero_point/transmission
    suffix: the suffix of the saved cube
    outdir: the output directory
    """
    with fits.open(fitscube) as hdu:
        header = hdu['PRIMARY'].header
        cube_header = hdu['DATA'].header
        cube_unit = units.Unit(cube_header['BUNIT'])
        cube_wavelength = get_wavelength(cube_header)
        cube = hdu['DATA'].data
        if not cube_unit.is_equivalent('adu/s'):
            if cube_unit.is_equivalent('adu'):
                if exptime is None:
                    try: exptime = header['EXPTIME']
                    except: pass
                if exptime is None:
                    raise ValueError('Please provide the exptime!')
            else:
                raise ValueError('Please make sure the data unit is either adu or adu/s')
        
        try:
            arcfile = header['ARCFILE']
        except:
            pass
    if correction is not None:
        if isinstance(correction, str):
            if 'fits' in correction:
                with fits.open(correction) as hdu:
                    corr_data = hdu['DATA'].data
                    wavelength, correction = corr_data['wavelength'], corr_data['correction']
            elif 'dat' in correction:
                corr_tab = table.Table.read(correction, format='ascii')
                wavelength = corr_tab['wavelength'].value
                correction = corr_tab['correction'].value
    else:
        correction = 1.0
        if transmission is not None:
            if isinstance(transmission, str):
                transmission_corr = np.loadtxt(transmission, delimiter=',')
                transmission = transmission[:,1]
            correction = correction/transmission
        if zp is not None:
            correction = correction * zp
    # extrapolation the correction to the wavelength of the data cube
    # correction = np.ma.masked_invalid(correction)
    # mask the abnormal corrections with sigma_clip
    # correction = astro_stats.sigma_clip(correction, sigma=5).filled(np.ma.median(correction))
    cspl_interp = interpolate.CubicSpline(wavelength, correction) # cubic interpolation
    cube_correction = cspl_interp(cube_wavelength)
    cube_header['BUNIT'] = 'W / (m^2 um)'

    if cube_unit.is_equivalent('adu'):
        modified_hdu = fits.ImageHDU(cube/exptime*cube_correction[:,np.newaxis, np.newaxis], 
                                     name="DATA", header=cube_header)
    elif cube_unit.is_equivalent('adu/s'):
        modified_hdu = fits.ImageHDU(cube*cube_correction[:,np.newaxis, np.newaxis], 
                                     name="DATA", header=cube_header)
    else:
        logging.warning('Incorrect units in the header, please verify!')
        return
    # error_combined_hdu = fits.ImageHDU(error_combined, name="ERROR", header=hdr)
    primary_hdu = fits.PrimaryHDU(header=header)
    hdus = fits.HDUList([primary_hdu, modified_hdu])
    if outdir is not None:
        if not os.path.isdir(outdir):
            os.system(f'mkdir -p {outdir}')
        outfile = os.path.join(outdir, os.path.basename(fitscube)[:-4]+f'{suffix}.fits')
        hdus.writeto(outfile, overwrite=True)
        return outfile
    return hdus

#####################################
######### DATA Combination ##########

def fill_mask(data, mask=None, step=1, debug=False):
    """Using iterative median to filled the masked region
    In each cycle, the masked pixel is set to the median value of all the values in the 
    surrounding region (in cubic 3x3 region, total 8 pixels)
    Inspired by van Dokkum+2023 (PASP) and extended to support 3D datacube
   
    This implementation are pure python code, so it is relatively
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
            if np.any(idx_range > up_boundaries):
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

def clean_cube(datacube, mask=None, signal_mask=None, 
               sigma_clip=True, sigma=5.0,
               median_filter=False, median_filter_size=(5,3,3),
               median_subtract=True, median_subtract_row=False, median_subtract_col=False,
               channel_chunk=None):
    """clean the datacubes

    Args:
    datacube: the 3D datacube
    mask: the mask of the 3D datacube
    signal_mask: the mask of possible signals

    It supports:
      - sigma clipping
      - median filter, by default is False as it may remove the signals if it is very spiky
      - background subtraction
      - continue subtraction
    """
    # with warnings.catch_warnings():
        # warnings.filterwarnings('ignore', category=AstropyUserWarning)
    datacube_masked = np.ma.masked_invalid(datacube)
    if mask is not None:
        if datacube.shape != mask.shape:
            raise ValueError("Mask does not match with data!")
        datacube_masked.mask = mask | np.ma.masked_invalid(datacube).mask
    
    nchan, ny, nx = datacube.shape

    if median_filter:
        # apply the median filter to filter out the outliers caused by sky lines
        # choose the filter size to preserve weak science data
        datacube_filtered = ndimage.median_filter(datacube_masked, size=median_filter_size)
        # note that ndimage.median_filter will ignore the mask
        datacube_masked = np.ma.array(datacube_filtered, mask=mask)
    
    # prepare the masked data
    datacube_masked = np.ma.array(datacube, mask=mask)
    datacube_signal_masked = np.ma.array(datacube_masked, mask=signal_mask)

    if sigma_clip:
        # apply sigma_clip along the spectral axis
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            # first apply a global masking, masking outliers with S/N of 10*sigma
            # with a single exposure, it is very unlikely that the signal will have
            # such large S/N
            datacube_masked = astro_stats.sigma_clip(datacube_masked, sigma=10*sigma, 
                                                     maxiters=5, masked=True)
            # then apply a strick clipping on the channels without signals.
            sigma_clip_masked = astro_stats.sigma_clip(datacube_signal_masked, sigma=sigma, 
                                                       maxiters=5, axis=0, masked=True)
        if channel_chunk is not None:
            # apply chunk based sigma clip, it could be useful if different 
            # spectral window show different noise level
            # chose the chunk size so that the signal is weaker than noise
            cube_chunk_mask = np.full_like(datacube, fill_value=False)
            chunk_steps = np.hstack((np.arange(0, nchan, channel_chunk)[:-1], 
                                   (np.arange(nchan, 0, -channel_chunk)-channel_chunk)[:-1]))
            for chan in chunk_steps:
                chunk_masked = astro_stats.sigma_clip(datacube_signal_masked[chan:chan+channel_chunk], 
                                                      maxiters=2, sigma=sigma, masked=True)
                cube_chunk_mask[chan:chan+channel_chunk] = np.logical_xor(chunk_masked.mask, signal_mask[chan:chan+channel_chunk]) | datacube_masked.mask[chan:chan+channel_chunk]
            sigma_clip_masked = np.ma.array(datacube_masked, mask=cube_chunk_mask)

        # apply channel-wise sigma_clip (deprecated, too dangerous)
        # datacube_masked = astro_stats.sigma_clip(datacube_masked, maxiters=2, sigma=sigma,
                                                 # axis=(1,2), masked=True)
        datacube_signal_masked.mask = sigma_clip_masked.mask
        if signal_mask is not None:
            datacube_masked.mask = (np.logical_xor(sigma_clip_masked.mask, signal_mask) | datacube_masked.mask)
        else:
            datacube_masked.mask = (sigma_clip_masked.mask | datacube_masked.mask)

    if median_subtract:
        # print('median subtraction')

        # median subtraction along the spectral axis
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            spec_median = np.ma.median(datacube_signal_masked, axis=0)
        datacube_masked -= spec_median[None,:,:]

        if median_subtract_row or median_subtract_col:
            # prepre a cube for col and row subtraction
            spec_median_cube = np.repeat(spec_median[None,:,:], nchan, axis=0)
            spec_median_cube[~signal_mask] = datacube_masked.data[~signal_mask]
            datacube_signal_replaced = np.ma.array(spec_median_cube, 
                                                   mask=datacube_masked.mask)
        
            # row and column based median subtraction
            # datacube_masked -= np.ma.median(datacube_signal_replaced, axis=1).data[:,np.newaxis,:]
            # by x-axis
            datacube_masked -= np.ma.median(datacube_signal_replaced, axis=2).data[:,:,np.newaxis]

        # median subtraction image by image
        # spec_median =  np.ma.median(datacube_signal_masked, axis=(1,2))
        # spec_median_filled = fill_mask(spec_median, step=5)
        # datacube_masked -= spec_median_filled[:,np.newaxis, np.newaxis]

    # apply global median subtraction by default
    datacube_masked -= np.median(stats.sigmaclip(datacube_signal_masked.compressed(), low=sigma, high=sigma)[0])
      
    return datacube_masked

def get_sky_lines_mask(wavelength, esorex='esorex', sky_mask_min=1.0):
    """apply sigma-clip on the sky lines
    """
    sky_spec = np.zeros_like(wavelength)
    static_pool = search_static_calib(esorex)
    eris_oh_spec = os.path.join(static_pool, 'eris_oh_spec.fits')
    with fits.open(eris_oh_spec) as hdu:
        #hdu.info()
        H_header = hdu[1].header
        H_wavelength = (H_header['CRVAL1'] + (np.arange(1, H_header['NAXIS1']+1)-H_header['CRPIX1']) * H_header['CDELT1'])
        H_data = hdu[1].data
        J_header = hdu[2].header
        J_wavelength = (J_header['CRVAL1'] + (np.arange(1, J_header['NAXIS1']+1)-J_header['CRPIX1']) * J_header['CDELT1'])
        J_data = hdu[2].data
        K_header = hdu[3].header
        K_wavelength = (K_header['CRVAL1'] + (np.arange(1, K_header['NAXIS1']+1)-K_header['CRPIX1']) * K_header['CDELT1'])
        K_data = hdu[3].data
    #check whether the wavelength fill within any of the band
    if (K_wavelength[0] <= wavelength[-1]) and (K_wavelength[-1] >= wavelength[-1]):
        logging.info('Masking sky lines in K band')
        skyline_data = K_data
        skyline_wavelength = K_wavelength
        cspl_interp = interpolate.CubicSpline(skyline_wavelength, skyline_data) # cubic interpolation
        sky_spec += np.ma.masked_invalid(cspl_interp(wavelength, extrapolate=False)).filled(0)

    if (wavelength[0] >= H_wavelength[0]) and (wavelength[-1] <= H_wavelength[0]):
        logging.info('Masking sky lines in H band')
        skyline_data = H_data
        skyline_wavelength = H_wavelength
        cspl_interp = interpolate.CubicSpline(skyline_wavelength, skyline_data) # cubic interpolation
        sky_spec += np.ma.masked_invalid(cspl_interp(wavelength, extrapolate=False)).filled(0)

    if (wavelength[0] >= J_wavelength[0]) and (wavelength[-1] <= J_wavelength[0]):
        logging.info('Masking sky lines in K band')
        skyline_data = J_data
        skyline_wavelength = J_wavelength
        cspl_interp = interpolate.CubicSpline(skyline_wavelength, skyline_data) # cubic interpolation
        sky_spec += np.ma.masked_invalid(cspl_interp(wavelength, extrapolate=False)).filled(0)
    return sky_spec >  sky_mask_min

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

def combine_with_drifts(cube_list, pixel_shifts=None, debug=False):
    """this is the test function to test the combine data with drifts
    """
    ncubes = len(cube_list)
    # check the input variables
    if pixel_shifts is not None:
        if len(pixel_shifts) != ncubes:
            raise ValueError("Pixel_shift does not match the number of images!")
        pixel_shifts = np.array(pixel_shifts)
    if True:
        nchan, ysize, xsize = cube_list[0].shape
        pad = np.round(np.abs(pixel_shifts).max()).astype(int)
        padded_cube_size = (nchan, ysize+2*pad, xsize+2*pad)
        # define the final product
        data_combined = np.full(padded_cube_size, fill_value=0.)
          
        for i in range(ncubes):
            # get the offset, we need to multiply -1 to inverse the shifts
            # as the calculation of pixel shifts is the (x0-xi, y0-yi) 
            # later we can them directly use "+"
            offset = -1 * pixel_shifts[i]

            # read data from each observation
            cube = cube_list[i]
            cube_exptime = 600
            cube_data = cube_list[i]
            cube_nchan, cube_ny, cube_nx = cube_data.shape
        
            # follow two steps
            # step 1: get the pixel level shifts
            offset_pixel = np.round(offset).astype(int) 
            offset_residual = np.array(offset) - offset_pixel
            # step 2: interpolate the sub-pixel shift 
            #         to get the data on the closest grid
            chan_idx = np.arange(0, cube_nchan)
            xidx = np.arange(0., cube_nx)
            yidx = np.arange(0., cube_ny)
            xidx_origin = xidx + offset_residual[0]
            yidx_origin = yidx + offset_residual[1]

            # define the final grid
            changrid, ygrid, xgrid = np.meshgrid(chan_idx, yidx, xidx, indexing='ij')
            if True:
                if debug:
                    print(f'Current cube shape: {cube_data.shape}')
                    print(f'Extrapolated cube shape: {cube_nchan, cube_ny, cube_nx}')
                cube_data_new = scipy.interpolate.interpn(
                        (chan_idx, yidx_origin, xidx_origin), cube_data,
                        np.vstack([changrid.ravel(), ygrid.ravel(), xgrid.ravel()]).T, 
                        method='linear', bounds_error=False, fill_value=0).reshape(
                                (cube_nchan, cube_ny, cube_nx))

            data_combined[:, (offset_pixel[1]+pad):(offset_pixel[1]+cube_ny+pad), 
                          (offset_pixel[0]+pad):(offset_pixel[0]+cube_nx+pad)] += cube_data_new
    print('Final datacube shape:', data_combined.shape)
    return data_combined 

def combine_eris_cube(cube_list, pixel_shifts=None, savefile=None, 
                      z=None, line_width=1000, wave_range=None,
                      sigma_clip=True, sigma=5.0, median_subtract=False,
                      mask_ext=None, median_filter=False, median_filter_size=(5,3,3),
                      weighting=None, overwrite=False, debug=False):
    """combining data directly in pixel space
    this function combine the image/datacube in the pixel space

    image_list (list): the list of filenames or ndarray
    offset (list,ndarray): the offset (x, y) of each image
    wave_range (list, ndarray): the wavelength range to combine, in um
    savefile: the filename to keep the combined cube
    z (float): the redshift of the target, use to protect signal during sigma clipping
    sigma_clip (bool): by default it is on
    sigma (float): the sigma value of the sigma_clip
    median_subtract (bool): by default the median subtraction is off
    weighting (deprecated): not been used any more
    """
    if sigma_clip is True:
        logging.info(f'sigma_clip is {sigma_clip} with sigma={sigma}, use this with cautions if the signal is bright!')
    if median_subtract is True:
        logging.info(f'median_subtract is {median_subtract}, use this with cautions if the signal is extended!')
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
            cube_data[abs(1.0-cube_mask)<1e-6] = 0.0
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
        # here we just use the header from the first cube
        first_header = fits.getheader(cube_list[0], 'DATA')
        first_wcs = WCS(first_header)
        pad = np.round(np.abs(pixel_shifts).max()).astype(int)

        # get the wavelength of the cube
        # all the following cube have different wavelength will be interpolated
        # to this wavelength
        wavelength = get_wavelength(first_header)

        # cut the datacube if the wavelength range is defined
        if wave_range is not None:
            wave_unit = u.Unit(first_header['CUNIT3'])
            wave_convert_scale = wave_unit.to(u.um)
            print('wave_convert_scale', wave_convert_scale)
            wave_range = np.array(wave_range) / wave_convert_scale
            wave_select = (wavelength > wave_range[0]) & (wavelength < wave_range[1])
            print('channel selected', np.sum(wave_select))
        else:
            wave_select = np.full_like(wavelength, fill_value=True, dtype=bool)

        wavelength = wavelength[wave_select]
        len_wavelength = len(wavelength)
        padded_cube_size = (len_wavelength, 
                            first_header['NAXIS2'] + 2*pad,
                            first_header['NAXIS1'] + 2*pad)
        # define the final product
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
            # get the offset, we need to multiply -1 to inverse the shifts
            # as the calculation of pixel shifts is the (x0-xi, y0-yi) 
            # later we can them directly use "+"
            offset = -1 * pixel_shifts[i]

            # read data from each observation
            cube = cube_list[i]
            logging.info(f'{i+1}/{ncubes} working on: {cube}')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                cube_header_primary = fits.getheader(cube, 0)
                cube_exptime = float(cube_header_primary['EXPTIME'])
                cube_header = fits.getheader(cube, 'DATA')
                cube_data = fits.getdata(cube, 'DATA')
            if mask_ext is not None:
                cube_mask = fits.getdata(cube, mask_ext)
            else:
                cube_mask = np.full(cube_data.shape, fill_value=False)
            cube_wavelength = get_wavelength(cube_header)
            if wave_range is not None:
                cube_wave_unit = u.Unit(cube_header['CUNIT3'])
                cube_wave_convert_scale = wave_unit.to(u.um)
                cube_wave_range = np.array(wave_range) / wave_convert_scale
                cube_wave_select = (cube_wavelength > cube_wave_range[0]) & (
                                    cube_wavelength < cube_wave_range[1])
            else:
                cube_wave_select = np.full_like(cube_wavelength, fill_value=True, dtype=bool)

            # if False: #if (len(cube_wavelength) != len(wavelength)) or ((
                #np.diff([wavelength, cube_wavelength], axis=0) > 1e-4).any())
                # the interpolation is needed for the spectral axis

            # # cut the datacube if needed, temperary solution
            # if len(cube_wavelength) > len(wave_select):
                # n_diff = len(cube_wavelength) - len(wave_select)
                # print('Inconsistent wavelength length, datacube has {} larger channel(s)'.format(n_diff))
                # cube_data = cube_data[:-n_diff]
                # cube_mask = cube_mask[:-n_diff]
                # cube_wavelength = cube_wavelength[:-n_diff]
            # if len(cube_wavelength) < len(wave_select):
                # print('Inconsistent wavelength length, bool selection has a larger size')
                # n_diff = len(wave_select) - len(cube_wavelength)
                # cube_data2 = np.zeros(np.array(cube_data.shape)+np.array([n_diff,0,0]))
                # cube_data2[:-n_diff] = cube_data
                # cube_data = cube_data2
                # cube_wavelength2 = np.zeros(len(cube_wavelength) + n_diff)
                # cube_wavelength2[:-n_diff] = cube_wavelength
                # cube_wavelength = cube_wavelength2
            cube_wavelength = cube_wavelength[cube_wave_select]
            cube_data = cube_data[cube_wave_select]
            cube_mask = cube_mask[cube_wave_select]

            cube_wavelength_norm = (cube_wavelength-wave_min)/wave_max*wave_scale
            cube_nchan, cube_ny, cube_nx = cube_data.shape
        
            if z is not None:
                # with redshift, we can mask the signals, here are the H-alpha and [N II]
                wave_mask = mask_spectral_lines(cube_wavelength, redshift=z, line_width=line_width)
                cube_wave_mask = np.repeat(wave_mask, cube_ny*cube_nx).reshape(
                                           len(cube_wavelength), cube_ny, cube_nx)
            else:
                cube_wave_mask = None
            
            # mask the outliers and subtract median noise, see clean_cube for details
            # needs to be cautious in preveting removing possible signal 
            cube_data_masked = clean_cube(cube_data, mask=cube_mask, signal_mask=cube_wave_mask, 
                                          sigma_clip=sigma_clip, sigma=sigma, median_subtract=False)
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

    if z is not None:
        wave_mask = mask_spectral_lines(wavelength, redshift=z, line_width=line_width)
        wave_mask = np.repeat(wave_mask, padded_cube_size[1]*padded_cube_size[2]).reshape(
                              *padded_cube_size)
        print('final wave_mask shape',wave_mask.shape)
    else:
        wave_mask = None
    sky_lines_mask = get_sky_lines_mask(wavelength)
    if sky_lines_mask is not None:
        # apply stronger mask on the sky lines
        # get the statistics with skyline masked
        sky_line_cube_mask = np.zeros_like(data_combined).astype(bool)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            mean, median, std = astro_stats.sigma_clipped_stats(data_combined[~sky_lines_mask],
                                                                sigma=3, maxiters=5,)
        sky_chan_data = data_combined[sky_lines_mask].copy()
        sky_chan_data[(sky_chan_data < -2.*std) | (sky_chan_data > 2*std)] = np.nan
        data_combined[sky_lines_mask] = sky_chan_data
        #sky_line_cube_mask[sky_lines_mask] = np.nan#(sky_chan_data)
        #data_combined[sky_line_cube_mask] = np.nan

    data_combined = clean_cube(data_combined,
                               signal_mask=wave_mask, 
                               sigma_clip=sigma_clip, sigma=sigma,
                               median_subtract=median_subtract)
    data_combined = data_combined.filled(np.nan)
    if savefile is not None:
        # save the combined data
        hdr = first_wcs.to_header()
        # copy the useful header infos
        # for key in ['NAXIS3', 'CTYPE3', 'CUNIT3', 'CRPIX3', 'CRVAL3', 'CDELT3', 
                    # 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2',
                    # 'CD3_3', 'CD1_3', 'CD2_3', 'CD3_1', 'CD3_2', 
                    # 'PC3_3', 'PC1_3', 'PC2_3', 'PC3_1', 'PC3_2']:
            # try:
                # hdr[key] = first_header[key]
            # except:
                # pass
        # fixed the header with pixel shifts
        if pixel_shifts is not None:
            hdr['CRPIX1'] = first_header['CRPIX1'] + pad + pixel_shifts[0][0]
            hdr['CRPIX2'] = first_header['CRPIX2'] + pad + pixel_shifts[0][1]
        # fixed the header if cutout is performed
        hdr['CRVAL3'] = wave_min * u.Unit(first_header['CUNIT3']).to(u.Unit(hdr['CUNIT3']))
        # change the wavelength unit from m to um
        if hdr['CUNIT3'].strip() == 'm':
            hdr['PC3_3'] = 1e6 * hdr['PC3_3']
            hdr['CRVAL3'] = (1e6 * hdr['CRVAL3'], '[um] Coordinate value at reference point') 
            hdr['CDELT3'] = (1.0, '[um] Coordinate increment at reference point') 
            hdr['CUNIT3'] = 'um'

        # copy the PC records to CD to support QFitsView
        hdr['CD1_1'] = hdr['PC1_1']
        hdr['CD2_2'] = hdr['PC2_2']
        hdr['CD3_3'] = hdr['PC3_3']
        hdr['BUNIT'] = first_header['BUNIT'].strip() + '/s'
        hdr['OBSERVER'] = 'VLT/ERIS'
        hdr['COMMENT'] = 'Created by the eris_jhchen_utils.py'
        hdr['COMMENT'] = 'Report problems to jhchen@mpe.mpg.de'
        primary_hdu = fits.PrimaryHDU()
        data_combined_hdu = fits.ImageHDU(data_combined, name="DATA", header=hdr)
        # error_combined_hdu = fits.ImageHDU(error_combined, name="ERROR", header=hdr)
        hdus = fits.HDUList([primary_hdu, data_combined_hdu])
        hdus.writeto(savefile, overwrite=overwrite)
    else:
        return data_combined 

def drift_jhchen_model(delta_alt, delta_az):
    """thr drift model derived by jhchen

    delta_alt: the altitude difference between the beginning PSF star and the science target
    delta_az: the azimuth angle difference
    """
    a1, b1, c1 = [0.03672065, 0.30166069, 0.02538038]
    a2, b2, c2 = [0.05556658, 0.38972143, -0.06249007]
    delta_x = a1*delta_alt**2 + b1*delta_alt + c1*delta_az
    delta_y = a2*delta_alt**2 + b2*delta_alt + c2*delta_az
    return delta_x, delta_y

def read_eris_drifts(datfile, arcfilenames, xcolname='Xref', ycolname='Yref'):
    """read eris drifting table
    """
    pixel_center = [32., 32.] # the expected center
    dat = table.Table.read(datfile, format='csv')
    drifts = np.zeros((len(arcfilenames), 2)) # drift in [x, y]
    if xcolname not in dat.colnames:
        logging.warning(f"Failed read the drift from the colname {xcolname} in {datfile}")
        return drifts
    for i in range(len(arcfilenames)):
        arcfile = arcfilenames[i]
        try:
            dat_img = dat[dat['ARCFILE'] == arcfile]
        except:
            dat_img = dat[dat['arcfile'] == arcfile]

        if len(dat_img) == 1:
            drifts[i] = [dat_img[xcolname][0]-pixel_center[0], dat_img[ycolname][0]-pixel_center[1]]
        else:
            logging.warning(f"Drifts not found for {arcfile}")
            drifts[i] = [0.,0.]
    return drifts

def compute_eris_offset(image_list, additional_drifts=None, header_ext='Primary',
                        header_ext_data='DATA',
                        ra_offset_header='HIERARCH ESO OCS CUMOFFS RA',
                        dec_offset_header='HIERARCH ESO OCS CUMOFFS DEC',
                        coord_system='sky', debug=False):
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
    return array_offset

def match_obs_folder(obsname):
    """match the foldername where stores the reduced data for every single OB (tpl_start)
    """
    # target_matcher = re.compile("(?P<target>[\w\s\-\.+]+)_(?P<target_type>(SCIENCE|CALIBPSF|CALIBSTD))_(?P<id>\d{7})_(?P<tpl_start>[\d\-\:T]+)_(?P<band>[JKH]_[\w]{3,6}?)_(?P<spaxel>\d{2,3}mas)_(?P<exptime>\d+)s")
    target_matcher = re.compile("(?P<target>.+)_(?P<target_type>(SCIENCE|CALIBPSF|CALIBSTD|CALIB))_(?P<id>[0-9]{7})_(?P<tpl_start>[0-9-:T]+)_(?P<band>[JKH]_[a-z]{3,6}?)_(?P<spaxel>[0-9]{2,3}mas)_(?P<exptime>[0-9]+)s")
    try:
        obs_match = target_matcher.search(obsname).groupdict()
    except:
        obs_match = None
    return obs_match

def search_archive(datadir, target=None, target_type=None, band=None, spaxel=None, 
                   tpl_start=None, exptime=None,
                   ob_list=None, exclude_ob=False, 
                   arcfile_list=None, exclude_arcfile=True,
                   outfile=None, outdir=None, sof_file=None, tag='',):
    """search file in the archive -- datadir

    """
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')
    if ob_list is not None:
        if isinstance(ob_list, str):
            ob_list = np.loadtxt(ob_list, dtype=str)
    if arcfile_list is not None:
        if isinstance(arcfile_list, str):
            arcfile_list = np.loadtxt(arcfile_list, dtype=str)
    if target is not None:
        if isinstance(target, str):
            target_list = [target,]
        elif isinstance(target, (list,tuple)):
            target_list = target
        else:
            logging.warning(f"target: {target} is not a valid name(s)")
            target_list = []
    if target_type is not None:
        if isinstance(target_type, str):
            target_type_list = [target_type,]
        elif isinstance(target_type, (list,tuple)):
            target_type_list = target_type
        else:
            logging.warning(f"target_type: {target_type} is not valid")
            target_type_list = []
    else:
        target_type_list = ['SCIENCE','CALIBPSF','CALIBSTD', 'CALIB']
    dates = os.listdir(datadir)
    image_list = []
    image_exp_list = []
    image_arcname_list = []
    for date in dates:
        if not date_matcher.match(date):
            continue
        for obs in os.listdir(os.path.join(datadir, date)):
            obs_match = match_obs_folder(obs)
            if obs_match is not None:
                obs_dir = os.path.join(datadir, date, obs)
                ob_target, ob_id = obs_match['target'], obs_match['id']
                ob_target_type = obs_match['target_type']
                if ob_list is not None:
                    if exclude_ob:
                        if ob_id in ob_list:
                            continue
                    else:
                        if ob_id not in ob_list:
                            continue
                ob_band, ob_spaxel = obs_match['band'], obs_match['spaxel']
                ob_exptime = obs_match['exptime']
                ob_tpl_start = obs_match['tpl_start']
            if target is not None:
                if ob_target not in target_list:
                    continue
            if tpl_start is not None:
                if tpl_start != ob_tpl_start:
                    continue
            if target_type is not None:
                if ob_target_type not in target_type_list:
                    continue
            if band is not None:
                if ob_band != band:
                    continue
            if spaxel is not None:
                if ob_spaxel != spaxel:
                    continue
            if exptime is not None:
                if ob_exptime != str(exptime):
                    continue
            if ob_target_type in target_type_list:
                # combine the exposures within each OB
                if ob_target_type == 'SCIENCE':
                    exp_list = glob.glob(obs_dir+'/eris_ifu_jitter_dar_cube_[0-9]*.fits')
                elif ob_target_type == 'CALIBPSF':
                    exp_list = glob.glob(obs_dir+'/eris_ifu_stdstar_psf_cube_[0-9]*.fits')
                elif ob_target_type == 'CALIBSTD':
                    exp_list = glob.glob(obs_dir+'/eris_ifu_stdstar_std_cube_[0-9]*.fits')
                # check the arcfile name not in the excludes file list
                for fi in exp_list:
                    with fits.open(fi) as hdu:
                        arcfile = hdu['PRIMARY'].header['ARCFILE']
                        if arcfile_list is not None:
                            if exclude_arcfile:
                                if arcfile in arcfile_list:
                                    continue
                            else:
                                if arcfile not in arcfile_list:
                                    continue
                        image_list.append(fi)
                        image_arcname_list.append(arcfile)
                        image_exp_list.append(float(ob_exptime))

    if outfile is not None:
        with open(outfile, 'w+') as fp:
            for img, arcname, exp in zip(image_list, image_arcname_list, image_exp_list):
                fp.write(f"{img} {arcname} {exp}\n")

    if sof_file is not None:
        with open(sof_file, 'w+') as fp:
            for img in image_list:
                fp.write(f"{img} {tag}\n")
    if outdir is not None:
        # copy all the select files to the outdir
        for img in image_list:
            subprocess.run(['cp', '{img}', outdir])
    return image_list, image_arcname_list, image_exp_list

def summarise_eso_files(filelist, outfile=None):
    """generate summary information of the filelist (fitsfiles)

    This function is designed to read all the possible useful information from the header
    to calculate the drift
    """
    colnames_header = ['ARCFILE', 'DATE-OBS',  'HIERARCH ESO OBS ID', 'HIERARCH ESO TPL START', 
                       'HIERARCH ESO DET SEQ1 DIT',
                       'HIERARCH ESO PRO REC1 RAW1 CATG', 
                       'HIERARCH ESO TEL ALT', 'HIERARCH ESO TEL AZ',
                       'HIERARCH ESO TEL PARANG START', 'HIERARCH ESO TEL PARANG END',
                       'HIERARCH ESO ADA ABSROT START', 'HIERARCH ESO ADA ABSROT END', 
                       ]
    colnames = ['filename', 'arcfile', 'date_obs', 'ob_id', 'tpl_start', 'exptime', 'catg','tel_alt', 'tel_az', 'tel_parang_start', 'tel_parang_end', 'ada_absrot_start', 'ada_absrot_end']
    summary_tab = table.Table(names=colnames, dtype=['U32']*len(colnames))
    for ff in filelist:
        with fits.open(ff) as hdu:
            header_primary = hdu['PRIMARY'].header
        header_values = [ff,]
        for col in colnames_header:
            header_values.append(str(header_primary[col]))
        summary_tab.add_row(header_values)
    summary_tab.sort('tpl_start')
    if outfile is not None:
        save_metadata(summary_tab, metafile=outfile)
    else:
        return summary_tab

def fit_star_position(starfits, x0=None, y0=None, pixel_size=1, plot=False,
                      interactive=False, basename=None, outfile=None, plotfile=None,
                      ):
    """two dimentional Gaussian fit

    """
    if basename is None:
        basename = starfits
    # image_func = lambda data: np.sum(data, axis=0)
    def image_func(data, channel_range=[0,-1], **kwargs):
        chan_low, chan_up = channel_range
        image = np.sum(data[int(chan_low):int(chan_up)], axis=0)
        # row by row, median subtraction
        image = image - np.nanmedian(image, axis=1)[:,None]
        return image 
    # read the fits file
    if basename is None:
        basename = starfits
    print(f'Reading stars from {starfits}')
    with fits.open(starfits) as hdu:
        header = hdu['PRIMARY'].header
        data_header = hdu['DATA'].header
        data = hdu['DATA'].data
        ndim = data.ndim
        exptime = header['EXPTIME']
        arcfile = header['ARCFILE']
    if True:
        wavelength = get_wavelength(header=data_header)
        # collapse the cube to make map
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            data = astro_stats.sigma_clip(data, sigma=5, axis=0).filled(0)
        # data = ndimage.median_filter(data, size=20, axes=0)
        nchan, ny, nx = data.shape

        # get the initial image
        image = image_func(data)
        vmax = np.nanpercentile(image[20:-20, 20:-20], 99)
        vmin = np.nanpercentile(image[20:-20, 20:-20], 1)
        # if abs(vmax) < abs(vmin):
            # vmax, vmin = -1.*vmin, vmax
            # data = -1.*data
            # image = -1.*image

        fit_image = fit_gaussian_2d(image, return_fitimage=True)
        residual_image = image - fit_image
    if True: # make a interative fit
        # prepare the ajustable parameters for the interactive window
        from matplotlib.widgets import Slider, RangeSlider, Button
        image_kwargs = {}
        slider_height=0.1
        slider_kwargs = {
                'channel_range':{'default':[0,nchan], 'range':[0, nchan]},
                'color_scale':{'default':[1,99], 'range':[0, 100]},
                'x': {'default':32, 'range':[0, 64]},
                'y': {'default':32, 'range':[0, 64]}
                }

        fig = plt.figure(figsize=(12, 7))
        
        # set the height of the sliders
        if slider_height == 'auto':
            nparam = len(slider_args)
            slider_height = np.min([0.5/nparam, 0.1])
        bottom_pad = 0.35
        
        # generate the slider the for the keywords args of input function
        args_default = {}
        args_slider = {}
        for i,item in enumerate(slider_kwargs.items()):
            key, value = item
            ax_param = fig.add_axes([0.65, bottom_pad-i*0.5*slider_height, 0.25, 0.5*slider_height])
            arg_default = value['default']
            arg_range = value['range']
            if isinstance(arg_default, (list,tuple)):
                ax_slider = RangeSlider(ax=ax_param, label=key.replace('_',' ')+' ', valmin=arg_range[0], 
                                        valmax=arg_range[1], valinit=arg_default, 
                                        orientation='horizontal')
            else:
                ax_slider = Slider(ax=ax_param, label=key.replace('_',' ')+' ', valmin=arg_range[0], 
                                   valmax=arg_range[1], valinit=arg_default, 
                                   orientation='horizontal')
            args_default[key] = arg_default
            args_slider[key] = ax_slider
       
        # plot the main figure
        ax_main = fig.add_axes([0.08, 0.1, 0.45, 0.85])
        line0, = ax_main.plot(32, 32, 'x', color='grey', alpha=0.6)
        line1, = ax_main.plot(args_default['x'], args_default['y'], 'x', color='red')
        image_ax = ax_main.imshow(image, vmax=vmax, vmin=vmin, origin='lower',
                                  **image_kwargs, )
        ax_main.set_xlabel('x')
        ax_main.set_ylabel('y')

        # plot the tow subimages for the fitting
        ax_sub1 = fig.add_axes([0.56, 0.5, 0.2, 0.4])
        ax_sub2 = fig.add_axes([0.78, 0.5, 0.2, 0.4])
        ax_sub1.set_title('Model')
        ax_sub2.set_title('Residual')
        ax_sub2.axes.set_yticklabels([])
        subimage_ax1 = ax_sub1.imshow(fit_image, origin='lower')
        subimage_ax2 = ax_sub2.imshow(residual_image, vmin=vmin, vmax=vmax, origin='lower')

        # instantly update the canvas
        def update(val):
            kwargs = {}
            for a in slider_kwargs.keys():
                kwargs[a] = args_slider[a].val
            image = image_func(data, **kwargs)
            vmin = np.nanpercentile(image[20:-20, 20:-20], kwargs['color_scale'][0])
            vmax = np.nanpercentile(image[20:-20, 20:-20], kwargs['color_scale'][1])
            image_ax.set(data=image, clim=(vmin, vmax))
            subimage_ax1.set(clim=(vmin, vmax))
            subimage_ax2.set(clim=(vmin, vmax))
            line1.set(data=[[kwargs['x']],[kwargs['y']]])
            fig.canvas.draw_idle()

        for slid in args_slider.values():
            slid.on_changed(update)

        # add the fit botton
        fit_botton = fig.add_axes([0.9, 0.1, 0.05, 0.04])
        button_fit = Button(fit_botton, 'Fit', hovercolor='0.975')
        # accept_botton = fig.add_axes([0.9, 0.025, 0.05, 0.04])
        # button_accept = Button(accept_botton, 'Accept', hovercolor='0.975')
        def fit_position(event):
            kwargs = {}
            for a in slider_kwargs.keys():
                kwargs[a] = args_slider[a].val
            image = image_func(data, **kwargs)
            fit_image = fit_gaussian_2d(image, x0=kwargs['x'], y0=kwargs['y'],
                                                    return_fitimage=True)
            residual_image = image - fit_image
            subimage_ax1.set(data=fit_image)
            ax_sub1.imshow(fit_image, origin='lower')
            subimage_ax2.set(data=residual_image)
            fig.canvas.draw_idle()
        button_fit.on_clicked(fit_position)
        # set block=True to hold the plot
        plt.show(block=True)
        # print the final slider keyword values
        final_kwargs = {}
        for a in slider_kwargs.keys():
            final_kwargs[a] = args_slider[a].val
        image = image_func(data, **final_kwargs)
        best_fit = fit_gaussian_2d(image, x0=final_kwargs['x'], y0=final_kwargs['y'], 
                                   basename=basename, plot=plot, plotfile=plotfile)
        print("best_fit", best_fit)
        return best_fit

def fit_gaussian_2d(image, amp=None, x0=None, y0=None, xsigma=None, ysigma=None,
                    theta=None, bounds=None, return_fitimage=False,
                    basename='', plot=False, plotfile=None):
    """costumized 2d gaussian fitting
    
    Args:
        image: 2d image
        p_init: the initial guess of the gaussian: [amp, x0, y0, xsigma, ysigma, theta]
                x0, y0 xsigma, ysigma: in the unit of pixel
                theta in unit of radian
        bounds: the boundaries of all the parameters
    """
    yshape, xshape = image.shape
    margin_padding = int(0.05*(yshape+xshape))
    norm_scale = 2*np.percentile(image[margin_padding:-margin_padding,
                                   margin_padding:-margin_padding], 98)
    sigma2FWHM = np.sqrt(8*np.log(2))
    image_median = np.nanmedian(image)
    image_normed = (image - image_median) / norm_scale 
    center_ref = np.array([xshape*0.5, yshape*0.5])
    # make initial guess
    if amp is None: amp = 1.
    if x0 is None: x0 = center_ref[0]
    if y0 is None: y0 = center_ref[1]
    if xsigma is None: xsigma = 1
    if ysigma is None: ysigma = 1
    if theta is None: theta = 0
    # prepare for fitting
    p_init = [amp/norm_scale, x0-center_ref[0], y0-center_ref[1], xsigma, ysigma, theta]
    if bounds == None:
        bounds = [[-10, 10], [-0.5*xshape,0.5*xshape], [-0.5*yshape,0.5*yshape], 
                  [1/sigma2FWHM,0.5*xshape/sigma2FWHM], 
                  [1/sigma2FWHM,0.5*yshape/sigma2FWHM],
                  [-np.pi*0.5,np.pi*0.5]]
    # ygrid, xgrid = (np.mgrid[0:yshape,0:xshape] - center_ref[:,None,None])
    xgrid, ygrid = np.meshgrid((np.arange(0, xshape) - center_ref[0]),
                               (np.arange(0, yshape) - center_ref[1]))
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    rms = 1

    def _cost(params, xgrid, ygrid):
        # return np.sum((image_normed - gaussian_2d(params, xgrid, ygrid))**2/rms**2)
        return np.sum((image_normed - gaussian_2d(params, xgrid, ygrid))**2/(rgrid**2+(5+rms)**2))
    res_minimize = optimize.minimize(_cost, p_init, args=(xgrid, ygrid), method='L-BFGS-B',
                                     bounds=bounds)
    amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = res_minimize.x
    best_fit = [amp_fit*norm_scale, x0_fit+center_ref[0], y0_fit+center_ref[1], 
                xsigma_fit, ysigma_fit, beta_fit]
    xfwhm_fit, yfwhm_fit = xsigma_fit*sigma2FWHM, ysigma_fit*sigma2FWHM
   
    if plot:
        vmax = np.nanpercentile(image_normed[20:-20, 20:-20], 99)
        vmin = np.nanpercentile(image_normed[20:-20, 20:-20], 1)
        fit_image = gaussian_2d(res_minimize.x, xgrid, ygrid)
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0,0])
        im = ax1.imshow(image_normed, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax1)
        ax1.plot(x0_fit+32, y0_fit+32, 'x', color='red')
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_title(basename, fontsize=8)
        im = ax2.imshow(fit_image, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax2)
        ax3 = fig.add_subplot(gs[0,2])
        im = ax3.imshow(image_normed - fit_image, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax3)
        if plotfile is not None:
            plotfile.savefig(fig, bbox_inches='tight')
            plt.close()
        if plot:
            plt.show()
        else:
            plt.close()
    if return_fitimage:
        return gaussian_2d(res_minimize.x, xgrid, ygrid) * norm_scale
    else:
        return best_fit
 
def fit_star_position_legacy(starfits, x0=None, y0=None, pixel_size=1, plot=False,
                      interactive=False, basename=None, outfile=None, plotfile=None,
                      ):
    """two dimentional Gaussian fit

    """
    # read the fits file
    if basename is None:
        basename = starfits
    with fits.open(starfits) as hdu:
        header = hdu['PRIMARY'].header
        data_header = hdu['DATA'].header
        data = hdu['DATA'].data
        ndim = data.ndim
        exptime = header['EXPTIME']
        arcfile = header['ARCFILE']
    if ndim == 2:
        # it is already a map
        image = data
    elif data.ndim == 3:
        wavelength = get_wavelength(header=data_header)
        # collapse the cube to make map
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            data = astro_stats.sigma_clip(data, sigma=5, axis=0).filled(0)
        data = ndimage.median_filter(data, size=40, axes=0)
        image = np.nansum(data, axis=0)

        # collapse the cube to get the image of the star
        # this is used to find the star and get its shape (by 2d-Gaussian fitting)
        # data = clean_cube(data, median_filter=False, median_subtract=False, sigma=10)
        # image = np.ma.sum(data, axis=0)
    yshape, xshape = image.shape
    margin_padding = 10
    norm_scale = 2*np.percentile(image[margin_padding:-margin_padding,
                                       margin_padding:-margin_padding], 98)

    image_median = np.nanmedian(image)
    image_normed = (image - image_median) / norm_scale 
    rms = 1

    # start the gaussian fitting for the 2D-image
    # generate the grid
    sigma2FWHM = np.sqrt(8*np.log(2))
    center_ref = np.array([yshape*0.5, xshape*0.5])
    # ygrid, xgrid = (np.mgrid[0:yshape,0:xshape] - center_ref[:,None,None])
    xsigma, ysigma = pixel_size, pixel_size
    ygrid, xgrid = np.meshgrid((np.arange(0, yshape) - center_ref[0])*pixel_size,
                               (np.arange(0, xshape) - center_ref[1])*pixel_size)
    rgrid = np.sqrt(xgrid**2 + ygrid**2)
    # vmin = np.nanpercentile(image_normed[margin_padding:-margin_padding, 
                                         # margin_padding:-margin_padding], 1)
    # vmax = np.nanpercentile(image_normed[margin_padding:-margin_padding, 
                                         # margin_padding:-margin_padding], 99)
    vmax = np.nanpercentile(image_normed[20:-20, 20:-20], 99)
    vmin = np.nanpercentile(image_normed[20:-20, 20:-20], 1)
    if abs(vmax) < abs(vmin):
        vmax, vmin = vmin, vmax
        image_normed = -1.*image_normed
    # vmax = 6
    vmin = -0.5*vmax
    if interactive:
        # get the initial guess from the user
        plt.figure()
        plt.imshow(image_normed, origin='lower', vmin=vmin, vmax=vmax)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.title('Please click on the star')
        plt.colorbar()
        plt.show(block=False)
        xclick, yclick = plt.ginput(1)[0]#, timeout=5
        plt.close()
        p_init = [image_normed[int(yclick), int(xclick)], 
                  xclick-center_ref[1], yclick-center_ref[0], xsigma, ysigma, 0]
        
    else:
        # automatically estimate the initial parameters
        amp = 1.2*vmax #np.nanmax(image_normed[10:-10,10:-10])
        amp_select = image_normed > vmax
        yidx, xidx = ygrid[amp_select], xgrid[amp_select]
        # x0, y0 = np.median(xidx), np.median(yidx)
        # if abs(x0)>20 or abs(y0)>20:
            # y0, x0 = 0, 0
        y0, x0 = 0, 0
        p_init = [amp, x0, y0, xsigma, ysigma, 0]
    print(f"p_init: {p_init}")

    def _cost(params, xgrid, ygrid):
        # return np.sum((image_normed - gaussian_2d(params, xgrid, ygrid))**2/rms**2)
        return np.sum((image_normed - gaussian_2d(params, xgrid, ygrid))**2/(rgrid**2 + (15+rms)**2))
    res_minimize = optimize.minimize(_cost, p_init, args=(xgrid, ygrid), method='L-BFGS-B',
                        bounds = [[0, 100], [-30,30], [-30,30], [0.1,5], [0.1,5],
                                  [-np.pi*0.5,np.pi*0.5]])

    amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = res_minimize.x
    xfwhm_fit, yfwhm_fit = xsigma_fit*sigma2FWHM, ysigma_fit*sigma2FWHM
    # print(xfwhm_fit, yfwhm_fit)
   
    if True:
        fit_image = gaussian_2d(res_minimize.x, xgrid, ygrid)
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[0,0])
        im = ax1.imshow(image_normed, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax1)
        ax2 = fig.add_subplot(gs[0,1])
        ax2.set_title(basename, fontsize=8)
        im = ax2.imshow(fit_image, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax2)
        ax3 = fig.add_subplot(gs[0,2])
        im = ax3.imshow(image_normed - fit_image, origin='lower', vmax=vmax, vmin=vmin)
        cbar = plt.colorbar(im, ax=ax3)
        if plotfile is not None:
            plotfile.savefig(fig, bbox_inches='tight')
            plt.close()
        if plot:
            plt.show()
        else:
            plt.close()

    # print([amp_fit, x0_fit, y0_fit, xfwhm_fit, yfwhm_fit, beta_fit])
    return [amp_fit, x0_fit, y0_fit, xfwhm_fit, yfwhm_fit, beta_fit]

def construct_drift_file(tpl_start_list, datadir, plot=False, driftfile=None, 
                         interactive=False, debug=False, overwrite=False):
    """ construct the drift file of the science exposures

    1. the program take the tpl_start as the unique identity of the science observation,
    2. then, it searches back the OB ID to check the closest PSF stars
    3. it uses the position of the two closest PSF stars to derive the drifts whenerver possible
    """
    tpl_start_list = np.unique(tpl_start_list).tolist()
    tpl_start_list.sort()
    summary_all = []
    group_id = 0
    table_start = None
    tpl_start_list_existing = []
    if driftfile is not None:
        if os.path.isfile(driftfile) and not overwrite:
            print(f'Found existing driftfile: {driftfile}')
            table_start = table.Table.read(driftfile, format='csv')
            tpl_start_list_existing = np.unique(table_start['tpl_start'])
            group_id = np.max(table_start['group_id']) + 1
    for tpl_start in tpl_start_list:
        if tpl_start in tpl_start_list_existing:
            print(f'skip tpl_start: {tpl_start}')
            continue
        # get all the observations within each tpl_start_list
        fits_objs, arcname_objs, exptime_objs = search_archive(datadir, tpl_start=tpl_start)
        # get the summaries of the all the observations
        summary_tab_objs = summarise_eso_files(fits_objs)
        summary_tab_objs.sort(['date_obs'])
        time_array_objs = np.array(summary_tab_objs['date_obs'], dtype='datetime64[s]')
        # get the PSF stars with the same OB ID
        ob_id = summary_tab_objs['ob_id'][0]
        ob_exptime = float(summary_tab_objs['exptime'][0])
        fits_stars, arcname_stars, exptime_stars = search_archive(datadir, ob_list=[ob_id,], target_type='CALIBPSF')
        summary_tab_stars = summarise_eso_files(fits_stars)
        if len(summary_tab_stars) < 1:
            logging.warning(f'{tpl_start}:{ob_id}: No valid PSF stars!')
            summary_tab = summary_tab_objs
        elif len(summary_tab_stars) == 1:
            logging.info(f'{tpl_start}:{ob_id}: Only found one valid PSF stars!')
            print(f'{tpl_start}:{ob_id}: Only found one valid PSF stars!')
            summary_tab = table.vstack([summary_tab_objs, summary_tab_stars])
        elif len(summary_tab_stars) > 1:
            logging.info(f'{tpl_start}:{ob_id}: Found {len(summary_tab_stars)} PSF stars!')
            print(f'{tpl_start}:{ob_id}: Found {len(summary_tab_stars)} PSF stars!')
            time_array_stars = np.array(summary_tab_stars['date_obs'], dtype='datetime64[s]') 
            time_array_stars_delta1 = np.abs(time_array_stars-time_array_objs[0]).astype(float)
            time_array_stars_delta2 = np.abs(time_array_stars-time_array_objs[-1]).astype(float)
            print("time difference:", time_array_stars_delta1, time_array_stars_delta2)
            summary_tab_stars_valid = summary_tab_stars[(time_array_stars_delta1<300) | (time_array_stars_delta2<ob_exptime*2+300)]
            print(summary_tab_stars_valid)
            n_valid_star = len(summary_tab_stars_valid)
            print(f"{n_valid_star} is found!")
            if n_valid_star > 1:
                summary_tab_stars.sort(['date_obs'])
                # psf_star_start = summary_tab_stars[np.argmin(np.abs(time_array_stars-time_array_objs[0]))]
                # psf_star_end = summary_tab_stars[np.argmin(np.abs(time_array_stars-time_array_objs[-1]))]
                psf_star_start = summary_tab_stars_valid[0]
                psf_star_end = summary_tab_stars_valid[-1]
                summary_tab = table.vstack([psf_star_start, summary_tab_objs, psf_star_end])
            elif n_valid_star == 1:
                time_array_stars_valid = np.array(summary_tab_stars_valid['date_obs'], 
                                                  dtype='datetime64[s]') 
                if (time_array_stars_valid - time_array_objs[0]).astype(float) < 0:
                    summary_tab = table.vstack([summary_tab_stars_valid, summary_tab_objs])
                else:
                    summary_tab = table.vstack([summary_tab_objs, summary_tab_stars_valid])
            else:
                logging.warning(f'{tpl_start}:{ob_id}: all {len(summary_tab_stars)} stars are not valid!')
                summary_tab = summary_tab_objs
        # print(summary_tab[-4:]['tpl_start', 'ob_id','date_obs'])

        # create columns to store the center of the offsets
        n_summary_tab = len(summary_tab)
        #['Xref','Yref','is_star','fitted','group_id']
        added_info = np.zeros((n_summary_tab, 6)) 
        for i,item in enumerate(summary_tab):
            if item['catg'] == 'PSF_CALIBRATOR':
                # is_star, group_id, fitted, interpolated, Xref, Yref
                added_info[i] = [1, group_id, 0, 0, 32, 32]
            else:
                added_info[i] = [0, group_id, 0, 0, 32, 32]
        added_info_tab = table.Table(added_info, 
                                     names=['is_star','group_id','fitted','interpolated','Xref','Yref',],
                                     dtype=['i4','i8','i4','i4','f8','f8'])
        summary_all.append(table.hstack([summary_tab, added_info_tab]))
        tmp_tab = table.hstack([summary_tab, added_info_tab])
                               
        group_id += 1
    if len(summary_all) > 0:
        table_all = table.vstack(summary_all)
    else:
        table_all = []
    if table_start is not None:
        if len(table_all) > 0:
            # change the dtype of to the same, here we use string type
            table_start = table.Table(table_start, dtype=['U1024']*len(table_start.columns))
            table_all = table.Table(table_all, dtype=['U1024']*len(table_all.columns))
            table_all = table.vstack([table_start, table_all])
        else:
            table_all = table_start
    # merge the continuous OBs without two PSFs stars 
    if True:
        table_all.sort(['date_obs'])
        gid_list = np.unique(table_all['group_id'])
        for i in range(1, len(gid_list)):
            obs_group_ip = table_all[table_all['group_id'] == gid_list[i-1]]
            if np.sum(obs_group_ip['catg'] == 'PSF_CALIBRATOR') >= 2:
                continue
            # compare the time difference with previous group
            obs_ip = obs_group_ip[-1]
            obs_i = table_all[table_all['group_id'] == gid_list[i]][0]
            t_ip = np.array(obs_ip['date_obs'], dtype='datetime64[s]')
            t_i = np.array(obs_i['date_obs'], dtype='datetime64[s]')
            if abs((t_i - t_ip).astype(float)) < 600: # within 10 minutes
                # check whether the second observation has recieved acquisition
                eris_quary_tab = eris_quary(ob_id=obs_i['ob_id'])
                # print("checking acquisition for", obs_i['ob_id'])
                if 'ACQUISITION' not in eris_quary_tab['DPR.CATG']:
                    # print('No acquisition, changing the group_id')
                    # print(table_all['group_id'])
                    table_all['group_id'][table_all['group_id'] == obs_i['group_id']] = obs_ip['group_id']
                    gid_list[i] = obs_ip['group_id'] # change the group_id list

    if len(summary_all) > 0: 
        if driftfile is not None:
            table_all.write(driftfile, format='csv', overwrite=True)
        else:
            return table_all
    else:
        print("Nothing to return")

def fit_eris_psf_star(summary_table, plotfile=None, interactive=False, overwrite=True, 
                      category='PSF_CALIBRATOR'):
    # fit the PSF stars in the summary table
    if isinstance(summary_table, str):
        writefile = summary_table
        summary_table = table.Table.read(summary_table, format='csv')
    else:
        writefile = None

    if plotfile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_file = PdfPages(plotfile)
    else:
        pdf_file = None

    for i,item in enumerate(summary_table):
        if item['catg'] == category:
            print(f'Working on psf star: {item["ob_id"]}+{item["tpl_start"]}...')
            if not item['fitted']:
                # logging.info('Fitting star {}'.format(item[]))
                if interactive:
                    _amp, xpix, ypix, _xsigma, _ysigma, _theta = \
                            fit_star_position(item['filename'],
                                              plot=True, plotfile=pdf_file)
                else:
                    _amp, xpix, ypix, _xsigma, _ysigma, _theta = \
                            fit_star_position_legacy(item['filename'], interactive=False,
                                              plot=True, plotfile=pdf_file)
                item['Xref'] = xpix
                item['Yref'] = ypix
                item['fitted'] = True
    if plotfile is not None:
        pdf_file.close()

    if writefile is not None:
        if overwrite:
            summary_table.write(writefile, format='csv', overwrite=True)
        else:
            summary_table.write(writefile[:-4]+'.updated.csv', format='csv')
    else:
        return summary_table

def interpolate_drifts(summary_table, extrapolate=True, overwrite=True):
    """ derive the drift of the science exposures 

    """
    # read the summary table
    if isinstance(summary_table, str):
        writefile = summary_table
        summary_table = table.Table.read(summary_table, format='csv')
    else:
        writefile = None
    # loop throught the observation share the same OB and group number
    # a same group meanes the observation were continuous without break
    group_id_list = np.unique(summary_table['group_id'])
    for group_id in group_id_list:
        # extrapolate the science scans with the two PSF stars
        group_select = summary_table['group_id'] == group_id
        group_obs = summary_table[group_select]
        group_is_star = group_obs['is_star'].astype(bool)
        # check if the group has already been interpolated
        group_is_interpolated = group_obs['interpolated'].astype(int)
        if (group_is_interpolated > 0).all():
            print("Skip group={}".format(group_id))
            continue
        if np.sum(group_is_star) >= 2:
            print('Extrapolating offsets for group={}'.format(group_id))
            obs_time_array = np.array(summary_table[group_select]['date_obs'], dtype='datetime64[s]')
            obs_delta_time = (obs_time_array - obs_time_array[0]).astype(float) # in units of second
            if extrapolate:
                # new implementation to support extrapolate
                spl_X = interpolate.CubicSpline(obs_delta_time[group_is_star], 
                                                     group_obs['Xref'][group_is_star])
                spl_Y = interpolate.CubicSpline(obs_delta_time[group_is_star], 
                                                     group_obs['Yref'][group_is_star])
                summary_table['Xref'][group_select] = spl_X(obs_delta_time) 
                summary_table['Yref'][group_select] = spl_Y(obs_delta_time) 
                summary_table['interpolated'][group_select] = '1'
            else:
                # the simplest linear interpolate, without extrapolate
                summary_table['Xref'][group_select] = np.interp(obs_delta_time, 
                                                            obs_delta_time[group_is_star], 
                                                            group_obs['Xref'][group_is_star],
                                                            right=32)
                summary_table['Yref'][group_select] = np.interp(obs_delta_time, 
                                                            obs_delta_time[group_is_star], 
                                                            group_obs['Yref'][group_is_star],
                                                            right=32)
        elif np.sum(group_is_star) == 1:
            print('Extrapolating offsets from existing model for group={}'.format(group_id))
            # print(group_obs[group_is_star]['Xref','Yref'])
            ref0_tab = group_obs[group_is_star]['Xref', 'Yref']
            Xref0, Yref0 = ref0_tab['Xref'].value[0], ref0_tab['Yref'].value[0]
            delta_alt = group_obs['tel_alt'] - group_obs['tel_alt'][0]
            delta_az = group_obs['tel_az'] - group_obs['tel_az'][0]
            Xref, Yref = drift_jhchen_model(delta_alt, delta_az) + np.array([Xref0, Yref0])[:,None]
            summary_table['Xref'][group_select] = Xref
            summary_table['Yref'][group_select] = Yref
            summary_table['interpolated'][group_select] = '2'
        else:
            print('No stars for position extrapolation for group={}, skip'.format(group_id))
    if writefile is not None:
        if overwrite:
            summary_table.write(writefile, format='csv', overwrite=True)
        else:
            summary_table.write(writefile[:-4]+'.updated.csv', format='csv')
    else:
        return summary_table

#####################################
######### DATA Quicklook ############

def pv_diagram(datacube, velocity=None, z=None, pixel_center=None,
               vmin=-1000, vmax=1000,
               length=1, width=1, theta=0, debug=False, plot=True, pixel_size=1):
    """generate the PV diagram of the cube

    Args:
        datacube: the 3D data [velocity, y, x]
        pixel_center: the center of aperture
        lengthth: the length of the aperture, in x axis when theta=0
        width: the width of the aperture, in y axis when theta=0
        theta: the angle in radian, from positive x to positive y axis
    """
    # nchan, ny, nx = datacube.shape
    width = np.round(width).astype(int)
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
    aper = RectangularAperture(pixel_center, length, width, theta=theta)
    s1,s2 = aper.to_mask().get_overlap_slices(cubeshape[1:])
    # cutout the image
    try:
        sliced_cube = datacube[:,s1[0],s1[1]]
    except:
        sliced_cube = None
        sliced_pvmap = None

    if sliced_cube is not None:
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
        ax1.imshow(np.nansum(datacube,axis=0),origin='lower')
        aper.plot(ax=ax1)
        if sliced_cube is not None:
            ax2.imshow(np.nansum(sliced_cube_rotated, axis=0), origin='lower')
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
        if sliced_cube is not None:
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
    return sliced_pvmap

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
                  z=None, rest_wave=0.65646*u.um, smooth_width=None, 
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
        if len(box_idx) == 6:
            cube = cube[box_idx[0]:box_idx[1], box_idx[2]:box_idx[3], box_idx[4]:box_idx[5]]
            vel = vel[box_idx[0]:box_idx[1]]
        elif len(box_idx) == 4:
            cube = cube[:, box_idx[0]:box_idx[1], box_idx[2]:box_idx[3]]
    if smooth_width is not None:
        gauss_kernel = Gaussian2DKernel(smooth_width)
        cube = convolve(cube, gauss_kernel)

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

def plot_fitcube(fitmaps, vmin=-300, vmax=300):
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(131)
    ax1.imshow(fitmaps[0], origin='lower')
    ax1.set_title("Intensity")

    ax2 = fig.add_subplot(132)
    ax2.imshow(fitmaps[1], vmin=vmin, vmax=vmax, origin='lower', cmap='RdBu_r')
    ax2.set_title("Velocity")
    
    ax3 = fig.add_subplot(133)
    ax3.imshow(fitmaps[2], origin='lower')
    ax3.set_title("Velocity")
    plt.show()


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
    exptime = int(exptime)
    logging.info(f'Requesting calibration files on {date}:{band}+{spaxel}+{exptime}s...')
    archive_name = f'{date}_{band}_{spaxel}_{exptime}s'
    archive_outdir = os.path.join(outdir, archive_name)
    # for backward compatibility
    if not os.path.isdir(archive_outdir):
        archive_name_old = f'{date}_{band}_{spaxel}_{exptime:.1f}s'
        archive_outdir_old = os.path.join(outdir, archive_name_old)
        if os.path.isdir(archive_outdir_old):
            print(f'moving the old calibPool {archive_outdir_old} to {archive_outdir}')
            os.system(f'mv {archive_outdir_old} {archive_outdir}')
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
            logging.info(f"> re-use existing calibPool in {archive_outdir}")
            return archive_outdir 

    # The date here is not the starting date. Normally, we can safely assume the
    # starting date is date - 1day, if the search is started at 12pm.
    # which is the default of eris_auto_quary
    start_date = (datetime.date.fromisoformat(date) 
                  - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    with tempfile.TemporaryDirectory() as tmpdir:
        if calib_raw is None:
            calib_raw = tmpdir
        metafile = os.path.join(calib_raw, f'{date}_{band}_{spaxel}_{exptime}s.csv')
        request_calib(start_date=start_date, band=band, spaxel=spaxel, exptime=exptime, 
                      outdir=calib_raw, metafile=metafile, steps=steps,
                      max_days=max_days, debug=debug, dry_run=dry_run)
        generate_calib(metafile, raw_pool=calib_raw, outdir=archive_outdir, 
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

def run_eris_pipeline(datadir='science_raw', outdir='science_reduced', 
                      categories=['SCIENCE','CALIB'], #ACQUISITION
                      calib_pool='calibPool', calib_raw='calib_raw',
                      dates=None, 
                      esorex='esorex', overwrite=False, 
                      debug=False, dry_run=False):
    """A wrapper for reduce_eris, designed for automatic pipeline for data
    reduction.

    To run this pipeline, the input datadir is organised by the dates of the 
    observations. Within each date, all the relevant science data have been
    download. 

    """
    # match all the dates
    datadir = datadir.strip('/')
    if outdir is not None:
        outdir = outdir.strip('/')
    else:
        outdir = 'science_reduced'
    date_matcher = re.compile(r'(\d{4}-\d{2}-\d{2})')
    date_list = []
    for subfolder in os.listdir(datadir):
        if date_matcher.match(subfolder):
            date_list.append(subfolder)
    # sort the dates in the datadir
    date_list = sorted(date_list)
    if dates is not None:
        date_selected = dates
    else:
        date_selected = date_list

    for date in date_list:
        if date not in date_selected:
            continue
        ## Step-1
        # generate all the summary files
        logging.info(f"> generating the metadata from {datadir}/{date}")
        # metadata = generate_metadata(os.path.join(datadir, date))
        date_metafile = os.path.join(datadir, date, 'metadata.csv')
        daily_datadir = os.path.join(datadir, date)
        daily_calib_pool = os.path.join(calib_pool, date)
        daily_calib_raw = os.path.join(calib_raw, date)
        daily_outdir = os.path.join(outdir, date)

        reduce_eris(metafile=date_metafile, datadir=daily_datadir, 
                    outdir=daily_outdir, calib_pool=daily_calib_pool, 
                    calib_raw=daily_calib_raw, categories=categories,
                    esorex=esorex, overwrite=overwrite, debug=debug, 
                    dry_run=dry_run)
        logging.info(f"<")

def quick_combine_legacy(datadir=None, target=None, offsets=None, excludes=None, band=None,
                  spaxel=None, drifts=None, outdir='./', esorex='esorex', z=None, 
                  savefile=None, suffix='combined', overwrite=False):
    """A wrapper of combine_data

    This quick tool search the all the available and valid observations
    and combine them with the combine_data.

    This tool take the outdir from `run_eris_pipeline` as input, it will search 
    all the available observations, and combined all the available data
    """
    # target_matcher = re.compile("(?P<target>[\w\s\-\.+]+)_(?P<id>\d{7})_(?P<band>[JKH]_[\w]{3,6}?)_(?P<spaxel>\d{2,3}mas)_(?P<exptime>\d+)s")
    target_matcher = re.compile("(?P<target>.+)_(?P<target_type>(SCIENCE|CALIBPSF|CALIBSTD))_(?P<id>[0-9]{7})_(?P<tpl_start>[0-9-:T]+)_(?P<band>[JKH]_[a-z]{3,6}?)_(?P<spaxel>[0-9]{2,3}mas)_(?P<exptime>[0-9]+)s")
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

def quick_combine(datadir=None, target=None, target_type='SCIENCE', offsets=None, 
                  band=None, exptime=None, filelist=None, 
                  sigma_clip=True, sigma=5.0, weak_signal=False, 
                  ob_list=None, exclude_ob=False, 
                  arcfile_list=None, exclude_arcfile=False,
                  spaxel=None, drifts=None, outdir='./', esorex='esorex', 
                  z=None, line_width=None, wave_range=None, median_subtract=None,
                  savefile=None, recipe='combine_eris_cube', suffix='combined', 
                  overwrite=False):
    """A wrapper of combine_data

    This quick tool search the all the available and valid observations
    and combine them with the combine_data.

    Args:
        datadir: the directory with all the reduced data orgnised by: 
                 <datadir>/<date>/<target>_<target_type>_<ob_id>_<band>_<spaxel>_<exptime>
        target: the name of the target
        target_type: the type of the target, can be "SCIENCE", "CALIBPSF", "CALIBSTD"

    This tool take the outdir from `run_eris_pipeline` as input, it will search 
    all the available observations, and combined all the available data
    """
    esorex_cmd_list = esorex.split(' ')
    if not os.path.isdir(outdir):
        subprocess.run(['mkdir','-p', outdir])
    
    if filelist is not None:
        if isinstance(filelist, str):
            filelist = np.loadtxt(filelist, dtype=str)
        image_list, image_arcname_list, image_exp_list = filelist.T
        image_exp_list = image_exp_list.astype(float)
    else:
        image_list, image_arcname_list, image_exp_list = search_archive(
                datadir=datadir, target=target, target_type=target_type, 
                band=band, spaxel=spaxel, exptime=exptime,
                ob_list=ob_list, exclude_ob=exclude_ob, 
                arcfile_list=arcfile_list, exclude_arcfile=exclude_arcfile)
    
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
        if median_subtract is None:
            if wave_range is not None: median_subtract = True
            else: median_subtract = False
        if weak_signal:
            sigma_clip = True
            sigma = 3.
            median_subtract = True
        combine_eris_cube(cube_list=image_list, pixel_shifts=obs_offset, 
                          z=z, line_width=line_width, wave_range=wave_range, 
                          sigma_clip=sigma_clip, sigma=sigma,
                          median_subtract=median_subtract,
                          overwrite=overwrite, savefile=savefile)
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

def quick_pv_diagram(datacube, z=None, mode='horizontal', cmap='magma',
                     step=5, nslits=16, length=80, signal_vmin=-500, signal_vmax=500, 
                     vmin=-2000, vmax=2000, smooth=None, name=None,
                     xcenter=None, ycenter=None, sigma_clip=True, sigma=3.0,
                     median_subtract=True,
                     comments=None, additional_theta=0, 
                     savefig=None, savefile=None, overwrite=True, debug=False):
    """a wrapper of pv_diagram to generate a overview of the slits based pv_diagrams
    """
    # nchan, ny, nx = datacube.shape
    if name is None: 
        name = os.path.basename(datacube)
    if isinstance(datacube, str):
        if z is not None:
            velocity, datacube, header = read_eris_cube(datacube, z=z)
        else:
            wavelength, datacube, header = read_eris_cube(datacube)
    if header is not None:
        if 'CD1_1' in header: pixel_size = header['CD1_1']
        elif 'CDELT1' in header: pixel_size = header['CDELT1']
        else: pixel_size = None
    # extract data within the velocity range 
    if velocity is not None:
        vel_selection = (velocity > vmin) & (velocity < vmax)
        datacube = datacube[vel_selection]
        velocity = velocity[vel_selection]
    if debug:
        print('shape of datacube:', datacube.shape)
        print('velocity:', velocity.shape, velocity)
    nz, ny, nx = datacube.shape
    if xcenter is None: xcenter = nx*0.5
    if ycenter is None: ycenter = ny*0.5
    
    # get some statistics of the map
    signal_vmask = (velocity < signal_vmax) & (velocity > signal_vmin)
    datacube = clean_cube(datacube, signal_mask=np.tile(signal_vmask, (nx, ny,1)).T, 
                          sigma_clip=sigma_clip, median_subtract=median_subtract,
                          median_subtract_row=False, median_subtract_col=False)
    std = np.nanstd(datacube[:,int(ycenter)-30:int(ycenter)+30,int(xcenter)-30:int(xcenter)+30])
    # _mean, _median, std = astro_stats.sigma_clipped_stats(datacube[:,10:ny-10,10:nx-10], sigma=5, maxiters=3)

    # caculate the pixel_centers along the slit
    if mode == 'horizontal':
        y_centers = np.arange(ycenter-step*nslits*0.5, ycenter+step*nslits*0.5, step) + 0.5*step
        x_centers = np.full_like(y_centers, fill_value=nx*0.5)
        theta = 0
    if mode == 'vertical':
        x_centers = np.arange(xcenter-step*nslits*0.5, xcenter+step*nslits*0.5, step) + 0.5*step
        y_centers = np.full_like(x_centers, fill_value=ny*0.5)
        theta = np.pi/2
    apertures = RectangularAperture(list(zip(x_centers, y_centers)), length, step, 
                                    theta=theta+additional_theta)
    signal_collapsed = np.nansum(datacube[signal_vmask], axis=0)
    if True:
        try: hdr = WCS(header).sub(['longitude','latitude']).to_header()
        except: hdr = fits.Header()
        hdr['OBSERVER'] = name
        hdr['COMMENT'] = "PV diagrams generated by eris_jhchen_utils.py"
        primary_hdu = fits.PrimaryHDU(header=header)
        image_hdu = fits.ImageHDU(signal_collapsed.filled(np.nan), name="image")
        hdu_list = [primary_hdu, image_hdu,]
    if True:
        fig = plt.figure(figsize=(16,6))
        nrow = np.ceil(nslits / 4).astype(int)
        ax1 = plt.subplot2grid((nrow, 6), (0, 0), rowspan=3, colspan=2)
        ax1.set_title(name, fontsize=10)
        ax1.imshow(signal_collapsed, origin='lower', vmin=-10*std, vmax=40*std, cmap=cmap)
        info_string = 'z={}'.format(z)
        if pixel_size is not None: 
            info_string += ',  pixel={:.3f}"'.format(abs(pixel_size)*3600.)
        if smooth is not None:
            info_string += ',  smooth={:.2f} pixels'.format(smooth)
        ax1.text(0.0, -0.1,'INFO: {}'.format(info_string), transform=ax1.transAxes, 
                 ha='left', va='center')
        if comments is not None:
            ax1.text(0.0, -0.18, 'Comments: {}'.format(comments), transform=ax1.transAxes, 
                     ha='left', va='center')
        # add a fake axis for the labels
        ax2 = plt.subplot2grid((nrow, 6), (0, 2), rowspan=nrow, colspan=4)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])
        ax2.set(frame_on=False)
        ax2.set_xlabel('Velocity: [{}~{} km/s]'.format(vmin, vmax), fontsize=12)
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Positions: width={} pixels'.format(step), fontsize=12)
        for i,aper in enumerate(apertures):
            aper.plot(ax=ax1, color='grey', linewidth=1, alpha=0.5)
            i_row = i//4 
            i_col = i%4
            sliced_pvmap = pv_diagram(datacube, velocity, pixel_center=aper.positions, 
                                      length=aper.w, width=aper.h, theta=aper.theta,
                                      vmin=vmin, vmax=vmax,
                                      plot=False)
            if sliced_pvmap is None:
                continue
            nynew, nxnew = sliced_pvmap.shape

            # save each pvmap
            pvmap_header = fits.Header()
            pvmap_header['CD1_1'] = pixel_size
            pvmap_header['CD2_2'] = velocity[1]-velocity[0]
            pvmap_header['CRPIX1'] = aper.w*0.5; pvmap_header['CRVAL1'] = aper.positions[0]
            pvmap_header['CRPIX2'] = 0; pvmap_header['CRVAL2'] = vmin
            pvmap_hdu = fits.ImageHDU(sliced_pvmap.T, name="slit-{}".format(i), header=pvmap_header)
            hdu_list.append(pvmap_hdu)
            
            # make the plot
            if mode == 'horizontal':
                ax1.text(0, aper.positions[1], i, horizontalalignment='left', 
                         verticalalignment='center', fontsize=6,
                         bbox=dict(facecolor='white', alpha=0.8,  edgecolor='none', boxstyle="circle"))
            if mode == 'vertical':
                ax1.text(aper.positions[0], 0, i, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=6,
                         bbox=dict(facecolor='white', alpha=0.8,  edgecolor='none', boxstyle="circle"))


            ax = plt.subplot2grid((nrow,6), (i_row,i_col+2))
            positions = np.linspace(-0.5*aper.w, 0.5*aper.h, nxnew)
            if velocity is None:
                velocity = np.linspace(-1,1,nchan)
            vmesh, pmesh = np.meshgrid(velocity, positions)
            if smooth is not None:
                gauss_kernel = Gaussian2DKernel(smooth)
                smoothed_pvmap = convolve(sliced_pvmap, gauss_kernel)
                ax.pcolormesh(vmesh, pmesh, smoothed_pvmap.T, cmap=cmap, vmin=-1*std, vmax=5*std)
            else:
                ax.pcolormesh(vmesh, pmesh, sliced_pvmap.T, cmap=cmap, vmin=-1*std, vmax=5*std)
            ax.text(0.03,0.9, i, horizontalalignment='center', transform=ax.transAxes, fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.8,  edgecolor='none', boxstyle="circle"))
            ax.axis('off')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        # fig.tight_layout()
    if savefile is not None:
        hdus = fits.HDUList(hdu_list)
        hdus.writeto(savefile, overwrite=overwrite)
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight', dpi=200)
    else:
        plt.show()
     

#####################################
######## helper functions ###########

def read_eris_header(fitsheader):
    """convert the header info to the info friendly to human
    """
    colnames = ['release_date', 'object', 'ra', 'dec','prog_id', 'dp_id', 'arcfile', 'exptime', 
                'ob_id', 'target_name', 'dpr_catg', 'dpr_type', 'dpr_tech', 'tpl_start', 
                'seq_arm', 'dit', 'band', 'spaxel', 'airm_start', 'airm_end']
    colnames_header = ['DATE', 'OBJECT', 'RA', 'DEC', 'HIERARCH ESO OBS PROG ID', 
                       'ARCFILE', 'ARCFILE','EXPTIME',
                       'HIERARCH ESO OBS ID', 'HIERARCH ESO OBS TARG NAME',  
                       'HIERARCH ESO DPR CATG', 'HIERARCH ESO DPR TYPE', 
                       'HIERARCH ESO DPR TECH', 'HIERARCH ESO TPL START', 
                       'HIERARCH ESO SEQ ARM', 'HIERARCH ESO DET SEQ1 DIT',
                       'HIERARCH ESO INS3 SPGW NAME', 'HIERARCH ESO INS3 SPXW NAME', 
                       'HIERARCH ESO TEL AIRM START', 'HIERARCH ESO TEL AIRM END']
    if isinstance(fitsheader, fits.Header):
        header = fitsheader
    elif isinstance(fitsheader, str) and ('.fits' in fitsheader):
        with fits.open(fitsheader) as hdu:
            header = hdu['PRIMARY'].header
    header_dict = {}
    for col, col_header in zip(colnames, colnames_header):
        try:
            header_dict[col] = header[col_header]
        except:
            pass
    return header_dict

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

def quick_fix_nan(cubefile, box=None, frac=0.1, header_ext='DATA', step=1,
                  outfile=None, overwrite=True):
    """quickly fix the nan values in the cube
    """
    hdu = fits.open(cubefile)
    cube = hdu[header_ext].data
    
    if len(box) == 4:
        ylow, yup, xlow, xup = box
        chanlow,chanup = 0,-1
    if len(box) == 6:
        chanlow, chanup, ylow, yup, xlow, xup = box
    cube_small = cube[chanlow:chanup,ylow:yup,xlow:xup]    
    nchan, ny, nx = cube_small.shape
    ## implementation of nearest interpolater
    mask = np.isnan(cube_small)
    x, y, z = np.mgrid[0:nchan, 0:ny, 0:nx]
    nearest_interp = interpolate.NearestNDInterpolator(list(zip(x[~mask].flatten(),y[~mask].flatten(),z[~mask].flatten())), cube_small[~mask].flatten())
    cube_small = nearest_interp(x, y, z)
    ## TODO: add support for interpolation along the spectral axis only, better for removing skylines
    #s3d = np.array([[[0,0,0],[0,1,0],[0,0,0]],
    #                        [[0,1,0],[1,1,1],[0,1,0]], 
    #                        [[0,0,0],[0,1,0],[0,0,0]]])
    #mask_global = ndimage.binary_closing(mask, iterations=5, structure=s3d)
    #cube_small[mask_global] = np.nan
    
    ## old method, using the fill_mask, deprecated now
    #npix = ny*nx
    #if len(box) == 4:
    #    ylow, yup, xlow, xup = box
    #    chanlow,chanup = 0,-1
    #if len(box) == 6:
    #    chanlow, chanup, ylow, yup, xlow, xup = box
    #cube[chanlow:chanup,ylow:yup,xlow:xup] = fill_mask(cube[chanlow:chanup,ylow:yup,xlow:xup], 
    #                                                   step=step)
    #for i in range(nchan):
    #    if np.sum(np.isnan(cube[i,ylow:yup,xlow:xup]))/npix < frac:
    #        cube[i,ylow:yup,xlow:xup] = fill_mask(cube[i,ylow:yup,xlow:xup])
    hdu[header_ext].data[chanlow:chanup,ylow:yup,xlow:xup] = cube_small
    if outfile is not None:
        hdu.writeto(outfile, overwrite=overwrite)
    else:
        hdu.writeto(cubefile, overwrite=overwrite)


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
    try:
        static_datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static_data')
        print('Default static data directory:', static_datadir)
    except:
        static_datadir = None
        print('Faild in determing the static_data directory!')

    # add subparsers
    subparsers = parser.add_subparsers(title='Available task', dest='task', 
                                       metavar=textwrap.dedent(
        '''
          * request_calib: search and download the raw calibration files
          * request_science: download the science data
          * generate_metadata: generate metadata from downloaded data
          * generate_calib: generate the calibration files
          * auto_jitter: run jitter recipe automatically
          * search_archive: search archive files
          * get_telluric_calibration: derive the transmission curve and zero_point

          Quick tools:

          * get_daily_calib: quick way to get dalily calibration files
          * run_eris_pipeline: quickly reduce science data with raw files
          * quick_combine: quickly combine reduced science data
          * quick_pv_diagram: quickly get the pv diagram of the datacube

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
    subp_request_science.add_argument('--username', type=str, help='The user name in ESO User portal.')
    subp_request_science.add_argument('--password', type=str, help='The password in ESO User Eortal.')
    subp_request_science.add_argument('--outdir', type=str, help='Output directory')
    subp_request_science.add_argument('--prog_id', type=str, help='Program ID', default='')
    subp_request_science.add_argument('--ob_id', type=str, help='Observation ID', default='')
    subp_request_science.add_argument('--metafile', type=str, help='Summary file',default='metadata.csv')
    subp_request_science.add_argument('--end_date', type=str, help='The finishing date of the observation. Such as 2023-03-08', default='')
    subp_request_science.add_argument('--dp_type', type=str, default='', 
                                      help='The type of the observation, set to "%%PSF-CALIBRATOR" to download only the PSF star')
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
    # reduce_eris
    subp_reduce_eris = subparsers.add_parser('reduce_eris',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            reduce eris data
            ----------------
            Examples:

            eris_jhchen_utils reduce_eris --metadata raw/metadata.csv --datadir raw --outdir outdir 
                                        '''))
    subp_reduce_eris.add_argument('--metafile', help='The summary file')
    subp_reduce_eris.add_argument('--datadir', help='The folder name with all the raw files')
    subp_reduce_eris.add_argument('--outdir', help='The output directory')
    subp_reduce_eris.add_argument('--calib_pool', help='The folder with all the calibration files')
    subp_reduce_eris.add_argument('--calib_raw', help='The folder to store the raw calibration files',
                                  default=None)
    subp_reduce_eris.add_argument('--static_pool', help='The folder with all the static calibration files')
    subp_reduce_eris.add_argument('--overwrite', action='store_true', 
                                  help='Overwrite the existing files')
    subp_reduce_eris.add_argument('--categories', type=str, nargs='+', 
        help="Selected targets types, can be combination of: SCIENCE, CALIB, ACQUISITION",
                                     default=['SCIENCE','CALIB'])

    ################################################
    # search_archive
    subp_search_archive = subparsers.add_parser('search_archive',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            search the archive files
            ------------------------
            Examples:

            eris_jhchen_utils search_archive --data science_reduced --target BX482 --band K_low
                                        '''))
    subp_search_archive.add_argument('--datadir', help='The archive directory')
    subp_search_archive.add_argument('--target', help='The name of the target')
    subp_search_archive.add_argument('--target_type', help='The type of the target observation: SCIENCE,CALIBPSF,CALIBSTD')
    subp_search_archive.add_argument('--band', help='The band of the observation')
    subp_search_archive.add_argument('--spaxel', help='The size of the spatial pixels')
    subp_search_archive.add_argument('--exptime', help='The exposure time')
    subp_search_archive.add_argument('--tpl_start', help='The tpl_start of the observation')
    subp_search_archive.add_argument('--ob_list', type=str, nargs='+', help='The ob names of the observation')
    subp_search_archive.add_argument('--exclude_ob', action='store_true', help='Use along with --ob_list, control the program to include or exclude the given list of OBs')
    subp_search_archive.add_argument('--arcfile_list', type=str, nargs='+', 
                                     help='The arcname list of the observation')
    subp_search_archive.add_argument('--exclude_arcfile', action='store_true', help='Use along with --arcfile_list, control the program to include or exclude the given list of filenames(arcname)')
    subp_search_archive.add_argument('--outfile', help='The output file to save the searching results')
    subp_search_archive.add_argument('--outdir', help='Copy the files into output directory')
    subp_search_archive.add_argument('--sof_file', help='Write the output as a sof file')
    subp_search_archive.add_argument('--tag', help='Use along with --sof_file, append the tag at each line')
    

    ################################################
    # get_telluric_calibration
    subp_get_flux_calibration = subparsers.add_parser('get_telluric_calibration',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            derive the transmission and zero point from the standard stars
            ---------------------------------------
            example:
            
              # get calibration files from the folder
              eris_jhchen_utils get_telluric_calibration --datadir science_reduced --outdir cube_corrections --star_catalogue star_catalogue.csv

              # get calibration files from a list of stars
              eris_jhchen_utils get_telluric_calibration --star_list file1 file2 --outdir cube_corrections --star_catalogue star_catalogue.csv

                                        '''))
    subp_get_flux_calibration.add_argument('--star_list', type=str, nargs='+', 
                                           help='a list of star fitsfile')
    subp_get_flux_calibration.add_argument('--star_list_file', type=str,
                                           help='a file includes a list of star fitsfile')
    subp_get_flux_calibration.add_argument('--datadir', help='reduced star files')
    subp_get_flux_calibration.add_argument('--outdir', help='output directory')
    subp_get_flux_calibration.add_argument('--star_catalogue', help='the catalogue file of stars')
    subp_get_flux_calibration.add_argument('--plotfile', help='the plotfile for quick check')
    subp_get_flux_calibration.add_argument('--static_datadir', default=static_datadir,
                                           help='the directory for some static data (such as filters)')

    ################################################
    # get_daily_calib
    subp_get_daily_calib = subparsers.add_parser('get_daily_calib',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            quickly get the daily calibration files
            ---------------------------------------
            example:
            
              # get calibration files
              eris_jhchen_utils get_daily_calib -d 2023-04-09 -b K_low -s 100mas -e 600 

              # prepare the daily calibration files for an on-going observation, which can
              # be later re-used by the `run_eris_pipeline`
              obsdate="2024-08-25"
              eris_jhchen_utils get_daily_calib -d "$obsdate" -b K_low -s 100mas -e 600 --outdir calibPool/"$obsdate" --calib_raw calib_raw/"$obsdate"


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
    subp_get_daily_calib.add_argument('--debug', action='store_true', 
                                      help='turn on the debug mode')
    
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
    subp_run_eris_pipeline.add_argument('-d', '--datadir', default='science_raw',
                                        help='The folder with downloaded science data')
    subp_run_eris_pipeline.add_argument('--outdir', default='science_reduced', 
                                        help='The output folder')
    subp_run_eris_pipeline.add_argument('--calib_pool', default='calibPool',
                                        help='The calibration pool')
    subp_run_eris_pipeline.add_argument('--calib_raw', default='calib_raw', 
                                        help='The raw pool of calibration files')
    subp_run_eris_pipeline.add_argument('--overwrite', action='store_true', 
                                        help='Overwrite the existing files')
    subp_run_eris_pipeline.add_argument('--categories', type=str, nargs='+', 
        help="Selected targets types, can be combination of: SCIENCE, CALIB, ACQUISITION",
                                     default=['SCIENCE','CALIB'])
    subp_run_eris_pipeline.add_argument('--dates', type=str, nargs='+', default=None,
                                        help='the selected dates')

    
    ################################################
    # quick combine
    subp_quick_combine = subparsers.add_parser('quick_combine',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            quickly reduce the science data
            -------------------------------
            example:

              eris_jhchen_utils quick_combine --datadir science_reduced --target bx482 --band K_middle --spaxel 25mas --drifts drifts_file --suffix test1 --outdir combined --z 2.2571 --weak_signal
                                        '''))
    subp_quick_combine.add_argument('--datadir', help='The data dierectory')
    subp_quick_combine.add_argument('--target', help='The target name')
    subp_quick_combine.add_argument('--offsets', help='The txt offsets file')
    subp_quick_combine.add_argument('--filelist', help='The files to be included')
    subp_quick_combine.add_argument('--sigma_clip', type=float, default=5, 
                                    help='The sigma for sigma_clip')
    subp_quick_combine.add_argument('--weak_signal', action='store_true',
                                    help='add it to combined the cubes with weak signals')
    subp_quick_combine.add_argument('--ob_list', help='The file includes ob ids')
    subp_quick_combine.add_argument('--exclude_ob', action='store_true', 
                                    help='add it to exclude obs in ob_list, otherwise it will use only the obs in ob_list')
    subp_quick_combine.add_argument('--arcfile_list', help='The file includes the arcfile list')
    subp_quick_combine.add_argument('--exclude_arcfile', action='store_true', 
                                    help='add it to exclude file in arcfile_list, otherwise it will use only the arcfiles in arcfile_list')
    subp_quick_combine.add_argument('--band', help='Observing band')
    subp_quick_combine.add_argument('--spaxel', help='Observing spaxel scale')
    subp_quick_combine.add_argument('--exptime', help='Exposure time')
    subp_quick_combine.add_argument('--z', type=float, help='The redshift of the target')
    subp_quick_combine.add_argument('--line_width', type=float, default=1000, 
                                    help='line width of the signal')
    subp_quick_combine.add_argument('--wave_range', nargs='+', type=float, help='The wavelength range in um')
    subp_quick_combine.add_argument('--drifts', help='Additional drifts')
    subp_quick_combine.add_argument('--outdir', help='Output dierectory')
    subp_quick_combine.add_argument('--suffix', default='combined', 
                                    help='The suffix of the output files')
    subp_quick_combine.add_argument('--recipe', default='combine_eris_cube', 
                                    help='The combine recipe to be used, set to "eris_ifu_combine" to use the pipeline\'s default combine recipe')
    subp_quick_combine.add_argument('--overwrite', action='store_true', help='Overwrite exiting fits file')


    ################################################
    # quick pv diagram
    subp_quick_pv_diagram = subparsers.add_parser('quick_pv_diagram',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            quickly get the pv diagram 
            -------------------------------
            example:

                eris_jhchen_utils quick_pv_diagram -d bx482nmfs_k025_hr_21h05.fits -z 2.2571 --smooth 1 --mode vertical --savefig bx482_pv.png --savefile bx482_pv.fits --comments "this is a test"

                                        '''))
    subp_quick_pv_diagram.add_argument('-d', '--datacube', help='The fits datafile')
    subp_quick_pv_diagram.add_argument('-z', '--redshift', type=float, help='The redshift')
    subp_quick_pv_diagram.add_argument('--mode', default='horizontal', 
                                       help='The mode: vertical or horizontal')
    subp_quick_pv_diagram.add_argument('--step', type=float, default=5, 
                                       help='The step or width of the slit in pixel, default=5')
    subp_quick_pv_diagram.add_argument('--nslits', type=int, default=16, 
                                       help='The number of slits, default=16')
    subp_quick_pv_diagram.add_argument('--length', type=float, default=80, 
                                       help='The length of each slit, default=64')
    subp_quick_pv_diagram.add_argument('--cmap', default='magma', help='The color map of the plot')
    subp_quick_pv_diagram.add_argument('-f', '--savefig', help='The filename to save the plot')
    subp_quick_pv_diagram.add_argument('--savefile', help='The filename to save the pv diagrams in each slit')
    subp_quick_pv_diagram.add_argument('--xcenter', type=float, help='The x coordinate center of slits')
    subp_quick_pv_diagram.add_argument('--ycenter', type=float, help='The y coordinate center of slits')
    subp_quick_pv_diagram.add_argument('--vmin', type=float, default=-2000, 
                                       help='The minimal velocty, default=-2000 km/s')
    subp_quick_pv_diagram.add_argument('--vmax', type=float, default=2000, 
                                       help='The maximal velocty, default=2000 km/s')
    subp_quick_pv_diagram.add_argument('--signal_vmin', type=float, default=-500, 
                                       help='The minimal velocty, default=-500 km/s')
    subp_quick_pv_diagram.add_argument('--signal_vmax', type=float, default=500, 
                                       help='The maximal velocty default=500 km/s')
    subp_quick_pv_diagram.add_argument('--smooth', type=float, default=None, 
                                       help='The kernel size for the smooth')
    subp_quick_pv_diagram.add_argument('--sigma_clip', type=float, default=3.0, 
                                       help='the sigma limit for sigma_clip')
    subp_quick_pv_diagram.add_argument('--median_subtract', default=True, 
                                       type=lambda x: (str(x).lower() == 'true'),
                                       help='skip median subtraction to the datacube')
    subp_quick_pv_diagram.add_argument('--name', help='The name of the plot')
    subp_quick_pv_diagram.add_argument('--comments', help="comments to be added to the plot") 

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
    elif args.task == 'reduce_eris':
        reduce_eris(metafile=args.metadata, datadir=args.datadir, 
                    outdir=args.outdir, calib_pool=args.calib_pool, 
                    static_pool=args.static_pool, calib_raw=args.calib_raw,
                    catagories=args.categories, overwrite=args.overwrite,
                    esorex=esorex, debug=args.debug)
    elif args.task == 'search_archive':
        image_list, _,_ = search_archive(
                datadir=args.datadir, target=args.target, target_type=args.target_type,
                band=args.band, spaxel=args.spaxel, exptime=args.exptime, 
                tpl_start=args.tpl_start,
                ob_list=args.ob_list, exclude_ob=args.exclude_ob, 
                arcfile_list=args.arcfile_list, exclude_arcfile=args.exclude_arcfile,
                outfile=args.outfile, outdir=args.outdir, sof_file=args.sof_file,
                tag=args.tag,)
        print(image_list)

    elif args.task == 'get_telluric_calibration':
        if args.star_list_file is not None:
            star_list = args.star_list_file
        else:
            star_list = args.star_list
        get_telluric_calibration(star_list=star_list, 
                                 datadir=args.datadir,
                                 star_catalogue=args.star_catalogue, 
                                 outdir=args.outdir, static_datadir=static_datadir)

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
                          dates=args.dates,
                          calib_raw=args.calib_raw, esorex=esorex, overwrite=args.overwrite, 
                          categories=args.categories,
                          debug=args.debug, dry_run=args.dry_run)
    elif args.task == 'quick_combine':
        if args.sigma_clip <= 0:
            sigma_clip = False
        else:
            sigma_clip = True
        quick_combine(datadir=args.datadir, target=args.target, offsets=args.offsets,
                      filelist=args.filelist, sigma_clip=sigma_clip, sigma=args.sigma_clip, 
                      weak_signal=args.weak_signal,
                      ob_list=args.ob_list, exclude_ob=args.exclude_ob,
                      arcfile_list=args.arcfile_list, exclude_arcfile=args.exclude_arcfile,
                      band=args.band, spaxel=args.spaxel, exptime=args.exptime,
                      z=args.z, line_width=args.line_width, wave_range=args.wave_range,
                      overwrite=args.overwrite, recipe=args.recipe,
                      outdir=args.outdir, drifts=args.drifts, suffix=args.suffix,
                      )
    elif args.task == 'quick_pv_diagram':
        if args.sigma_clip <= 0:
            sigma_clip = False
        else:
            sigma_clip = True
        quick_pv_diagram(datacube=args.datacube, z=args.redshift, mode=args.mode, 
                         step=args.step, nslits=args.nslits, length=args.length, cmap=args.cmap,
                         signal_vmin=args.signal_vmin, signal_vmax=args.signal_vmax,
                         xcenter=args.xcenter, ycenter=args.ycenter,
                         savefig=args.savefig, savefile=args.savefile,
                         name=args.name, comments=args.comments, debug=args.debug,
                         sigma_clip=sigma_clip, sigma=args.sigma_clip, 
                         median_subtract=args.median_subtract,
                         vmin=args.vmin, vmax=args.vmax, smooth=args.smooth,
                         )
    else:
        pass
    if args.debug:
        if ret is not None:
            logging.info(ret)
    logging.info('Finished')
