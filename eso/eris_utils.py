#!/usr/bin/env python

import os, 
import shutil
import re
import datetime

import astropy.table as table
from astroquery.eso import Eso
import requests 

def eris_auto_quary(start_date, end_date=None, column_filters={}, max_days=30):
    """query ESO/ERIS raw data from the database

    start_date (str): the starting date: lile 2023-04-08
    end_date (str): the same format as start_date, the default value is start_date + 1day
    column_filters: the parameters of the query form
                    such as: 
                    column_filters = {
                            'stime': '2023-04-08',
                            'etime': '2023-04-09',
                            'dp_cat': 'CALIB',
                            'dp_type': 'FLAT%',
                            'seq_arm': 'SPIFFIER',
                            'ins3_spgw_name': 'K_low',
                            'ins3_spxw_name': '100mas',}
    max_days: the maximum days to search for the availble calibration files
    """
    eso = Eso()
    sdate = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if end_date is None:
        edate = sdate + datetime.timedelta(days=1)
    delta_time = datetime.timedelta(days=1)
    matched = 0
    for i in range(0, max_days):
        if matched == 0:
            column_filters['stime'] = (sdate - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            column_filters['etime'] = (edate + datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            tab_eris = eso.query_instrument('eris', column_filters=column_filters)
            if tab_eris is not None:
                matched = 1
    if matched == 0:
        print("Cannot find proper calib files, please check your input")
    else:
        return tab_eris

def download_file(url, filename=None, outdir='./'):
    """download files automatically 

    Features:
    1. fast
    2. redownload failed files
    3. skip downloaded files

    Args:
        url (str): the full url of the file
        filename (str): the filename to be saved locally
        outdir (str): the output directory
    """
    is_downloaded = False
    if not os.path.isdir(outdir):
        os.system(f'mkdir -p {outdir}')
    with requests.get(url, stream=True) as r:
        if filename is None:
            # automatically define the filename
            try:
                filename_match = re.compile('filename=(?P<filename>[\w.\-\:]+)')
                filename = filename_match.search(r.headers['Content-Disposition']).groupdict()['filename']
            except:
                print(f"Errors in handling the filename from: {r.headers['Content-Disposition']}")
                filename = 'Undefined'
        filename_fullpath = os.path.join(outdir, filename)
        # check the local file if it exists
        if os.path.isfile(filename_fullpath):
            filesize = os.path.getsize(filename_fullpath)
            if str(filesize) == r.headers['Content-Length']:
                print(f"{filename} is already downloaded.")
                is_downloaded = True
            else:
                print('Find local inconsistent file, overwriting...')
        if not is_downloaded:      
            with open(filename_fullpath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)


def download_eris(eris_query_tab, outdir='raw', metafile=None, 
                  save_columns=['DP.ID','Object','EXPTIME','INS3.SPGW.NAME','INS3.SPXW.NAME']):
    """download the calib files of eris

    Args:
        eris_query_tab (astropy.table): the query table returned by astroquery.eso
        outdir (str): the directory to store the download files and saved meta table
        metafile (str): the filename of the saved tabe from eris_query_tab
        save_columns (list): the selected column names to be saved.
                             set the 'None' to save all the columns
    """
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    for fileid in eris_query_tab['DP.ID']:
        print(f"Handling {fileid}")
        file_url = root_calib_url+fileid
        download_file(file_url, outdir=outdir)
    if meta_file is not None:
        if '/' not in metafile:
            meta_file = os.path.join(outdir, metafile)
        if os.path.isfile(metafile):
            os.system(f'mv {meta_file} {meta_file}.bak')
        if save_columns is not None:
            saved_tab = eris_query_tab[save_columns]
        else:
            saved_tab = eris_query_tab
        saved_tab.write(metafile, format='csv')


def request_eris_calib(project, start_date, end_date=None, outdir=None,
                     steps=['dark','distortion','flat','wavecal','stdstar'],
                     columns_names=['DP.ID','Object','EXPTIME','INS3.SPGW.NAME','INS3.SPXW.NAME'],
                     dpcat='CALIB', arm='SPIFFIER', grating='K_low', spaxel='100mas', **kwargs):
    """a general purpose to qeury calib files of ERIS/SPIFFIER observation
    
    Args:
        project (str): the project code
        start_date (str): ISO date format, like: 2023-04-08
        end_date (str, None): same format as start_date
        outdir (str): the output directory of the download files
        steps (list): a list of calibration steps, the connection with DRP types:
            
            --------------------------------------------------------------------------
            name         dp_type           function
            --------------------------------------------------------------------------
            dark         DARK              dark data reduction
           *detlin       LINEARITY%        detector's linearity & non linear bad pixels
            distortion   NS%               distortion correction
            flat         FLAT%             flat field data reduction
            wavecal      WAVE%             wavelength calibration
            stdstar      %STD              reconstruct data cubes from std stars
            --------------------------------------------------------------------------
            (* not fully tested)

        dpcat (str): default to be 'CALIB'
        arm (str): the instrument arm of eris: SPIFFIER or NIX (in develop)
        grating (str): grating configurations
        spaxel (str): the spaxel size of the plate

    """
    dptype_dict = {'dark':'DARK', 'detlin':'LINEARITY*', 'distortion':'NS%',
                   'flat':'FLAT%', 'wavecal':'WAVE%', 'stdstar':'%STD'}
    query_tabs = []

    if project is not None:
            if outdir is None:
                outdir = project+'_raw'
            else: outdir = 'raw'

    for step in steps:
        column_filters = {'dp_type': dptype_dict[step],
                          'dp_cat': dpcat,
                          'seq_arm': arm,
                          'ins3_spgw_name': grating,
                          'ins3_spxw_name': spaxel, }
        step_query = eris_auto_quary(start_date, end_date=end_date, column_filters=column_filters, **kwargs)
        query_tabs.append(step_query[columns_names])

    all_tabs = table.vstack(query_tabs)

    download_eris(all_tabs, meta_file=project+'_metadata.csv')


def generate_sof(metafile, steps=[], static_calib=''):
    """generate the science of frame of each calibration step

    """
    pass


def generate_calib():
    """generate calibration files

    """
    pass

def auto_gitter():
    pass



# add main here to run the script from cmd
