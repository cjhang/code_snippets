#!/usr/bin/env python

"""
Authors: Jianhang Chen
Version: 0.1
History:
    - 2023-11-22: first release, v0.1
"""

import os 
import shutil
import re
import datetime
import logging
import getpass

import astropy.table as table
from astroquery.eso import Eso
import requests 

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
    if metafile is not None:
        if '/' not in metafile:
            metafile = os.path.join(outdir, metafile)
        if os.path.isfile(metafile):
            os.system(f'mv {metafile} {metafile}.bak')
        eris_query_tab.write(metafile, format='csv')

def eris_auto_quary(start_date, end_date=None, column_filters={}, max_days=30):
    """query ESO/ERIS raw data from the database

    Args:
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

def request_eris_calib(project, start_date, band, resolution, exptime,
                       end_date=None, outdir=None, dpcat='CALIB', arm='SPIFFIER', 
                       steps=['dark','detlin','distortion','flat','wavecal','stdstar'],
                       columns_names=['Release Date','DP.ID','Object','DPR.TYPE','DPR.CATG','EXPTIME',
                                      'INS3.SPGW.NAME','INS3.SPXW.NAME','SEQ.ARM'],
                       **kwargs):
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
            dark         DARK              dark data reduction (daily)
            detlin       LINEARITY%        detector's non linear bad pixels (monthly)
            distortion   NS%               distortion correction
            flat         FLAT%             flat field data reduction
            wavecal      WAVE%             wavelength calibration
            stdstar      %STD              reconstruct data cubes from std stars
            --------------------------------------------------------------------------

        dpcat (str): default to be 'CALIB'
        arm (str): the instrument arm of eris: SPIFFIER or NIX (in develop)
        band (str): grating configurations
        resolution (str): the spaxel size of the plate, [250mas, 100mas, 25mas]

    """
    dptype_dict = {'dark':'DARK', 'detlin':'LINEARITY%', 'distortion':'NS%',
                   'flat':'FLAT%', 'wavecal':'WAVE%', 'stdstar':'%STD'}
    query_tabs = []

    if outdir is None:
        if project is not None:
            outdir = project+'_raw'
        else: outdir = 'raw'

    for step in steps:
        column_filters = {'dp_cat': dpcat,
                          'seq_arm': arm,
                          'dp_type': dptype_dict[step]}
        if step == 'dark':
            # drop the requirement for band and resolution
            column_filters['exptime'] = exptime
        if step in ['distortion', 'flat', 'wavecal', 'stdstar']:
            column_filters['ins3_spgw_name'] = band
            column_filters['ins3_spxw_name'] = resolution

        step_query = eris_auto_quary(start_date, end_date=end_date, column_filters=column_filters, **kwargs)
        query_tabs.append(step_query[columns_names])

    all_tabs = table.vstack(query_tabs)
    download_eris(all_tabs, metafile=project+'_metadata.csv', outdir=outdir)

def requests_eris_science(program_id='', username=None, metafile=None,
                          outdir='science_raw', target='', observation_id='', 
                          start_date='', end_date=''):
    """download the science data 

    To download the proprietory data, you need to provide your eso username
    and you will be asked to input your password.

    Args:
        program_id (str): program id
        username (str): the user name of your eso account
        metafile (str): the output file to store all the meta data
                        default: program_id
        target (str): the target name
        outdir (str): the directory to save all the raw files
        observation_id (str, int): the id of the observation
        start_date (str): starting data, in the format of '2023-04-08'
        end_date (str): end date, same format as start_date
    """
    root_calib_url = 'https://dataportal.eso.org/dataportal_new/file/'
    if metafile is None:
        metafile = program_id+'.csv'
    if os.path.isdir(outdir):
        os.system(f'mkdir -p {outdir}')
    logging.info(f'Requesting the data from project: {program_id}')
    eso = Eso()
    eris_query_tab = eso.query_instrument(
            'eris', column_filters={'ob_id': observation_id,
                                    'prog_id':program_id,
                                    'stime': start_date,
                                    'etime': end_date,
                                    'target': target,})
    download_eris(eris_query_tab, username=username, outdir=outdir, metafile=metafile)

def generate_calib(project, metafile, raw_pool='./raw', static_pool='./staicPool', 
                   calib_pool='./calibPool', 
                   steps=['dark','detlin','distortion','flat','wavecal','stdstar'],
                   drp_type_colname='Object', esorex=None, dry_run=False):
    """generate the science of frame of each calibration step

    Args:
        project (str): the project name, used to organise the folders
        metafile (str): the metadata file where you can find the path and objects of each dataset
        raw_pool (str): the raw data pool
        static_pool (str): the static data pool
        calib_pool (str): the directory to keep all the output files from esorex
        steps (list): the steps to generate the corresponding calibration files
                      it could include: ['dark','detlin','distortion','flat','wavecal','stdstar']
        drp_type_colname (str): the column name of the metadata table includes the DPR types. 
        esorex (str): the callable esorex command from the terminal
        dry_run (str): set it to True to just generate the sof files
    """
    static_pool = static_pool.rstrip('/')
    calib_pool = calib_pool.rstrip('/')
    raw_pool = raw_pool.rstrip('/')
    if not os.path.isdir(calib_pool):
        os.system(f'mkdir -p {calib_pool}')
    if project is None:
        try:
            project_match = re.compile('(?P<project>[\w.\-]+)_metadata.csv')
            project = project_match.search(metafile).groupdict()['project']
        except:
            project = 'default001'
    if esorex is None:
        esorex_cmd = f'esorex --output-dir {calib_pool} --log-file {project}.log'
    else: esorex_cmd = esorex
    meta_tab = table.Table.read(metafile, format='csv')

    if 'dark' in steps:
        # generate the sof for dark calibration
        dark_sof = os.path.join(calib_pool, project+'_dark.sof')
        with open(dark_sof, 'w+') as openf:
            for item in meta_tab[meta_tab[drp_type_colname] == 'DARK']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z DARK\n")
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_dark {dark_sof}")
    if 'detlin' in steps:
        detlin_sof = os.path.join(calib_pool, project+'_detlin.sof')
        with open(detlin_sof, 'w+') as openf:
            for item in meta_tab[meta_tab[drp_type_colname] == 'LINEARITY,DARK,DETCHAR']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z LINEARITY_LAMP\n")
            for item in meta_tab[meta_tab[drp_type_colname] == 'LINEARITY,LAMP,DETCHAR']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z LINEARITY_LAMP\n")
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_detlin {detlin_sof}")

    if 'distortion' in steps:
        # generate the sof for distortion
        distortion_sof = os.path.join(calib_pool, project+'_distortion.sof')
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
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_distortion {distortion_sof}")
    if 'flat' in steps:
        # generate the sof for flat
        flat_sof = os.path.join(calib_pool, project+'_flat.sof')
        with open(flat_sof, 'w+') as openf:
            for item in meta_tab[meta_tab[drp_type_colname] == 'FLAT,DARK']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FLAT_LAMP\n")
            for item in meta_tab[meta_tab[drp_type_colname] == 'FLAT,LAMP']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z FLAT_LAMP\n")
            openf.write(f"{calib_pool}/eris_ifu_dark_bpm.fits BPM_DARK\n")
            openf.write(f"{calib_pool}/eris_ifu_detlin_bpm_filt.fits BPM_DETLIN\n")
            openf.write(f"{calib_pool}/eris_ifu_distortion_bpm.fits BPM_DIST\n")
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_flat {flat_sof}")
    if 'wavecal' in steps:
        # generate the sof for wavecal
        wavecal_sof = os.path.join(calib_pool, project+'_wavecal.sof')
        with open(wavecal_sof, 'w+') as openf:
            for item in meta_tab[meta_tab[drp_type_colname] == 'WAVE,DARK']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z WAVE_LAMP\n")
            for item in meta_tab[meta_tab[drp_type_colname] == 'WAVE,LAMP']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z WAVE_LAMP\n")
            openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
            openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
            openf.write(f"{static_pool}/eris_ifu_first_fit.fits FIRST_WAVE_FIT\n") 
            openf.write(f"{static_pool}/eris_ifu_ref_lines.fits REF_LINE_ARC\n")
            openf.write(f"{static_pool}/eris_ifu_wave_setup.fits WAVE_SETUP\n") 
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_wavecal {wavecal_sof}")

    if 'stdstar' in steps:
        # generate the stdstar
        stdstar_sof = os.path.join(calib_pool, project+'_stdstar.sof')
        with open(stdstar_sof, 'w+') as openf:
            for item in meta_tab[meta_tab[drp_type_colname] == 'STD']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z STD\n")
            for item in meta_tab[meta_tab[drp_type_colname] == 'SKY,STD']:
                openf.write(f"{raw_pool}/{item['DP.ID']}.fits.Z SKY_STD\n")
            openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
            openf.write(f"{calib_pool}/eris_ifu_distortion_slitlet_pos.fits SLITLET_POS\n")
            openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
            openf.write(f"{calib_pool}/eris_ifu_flat_bpm.fits BPM_FLAT\n")
            openf.write(f"{calib_pool}/eris_ifu_wave_map.fits WAVE_MAP\n")
            openf.write(f"{static_pool}/EXTCOEFF_TABLE.fits EXTCOEFF_TABLE\n")
            openf.write(f"{static_pool}/eris_oh_spec.fits OH_SPEC\n")
            openf.write(f"{static_pool}/RESPONSE_WINDOWS_K_low.fits RESPONSE\n")
        if not dry_run:
            os.system(f"{esorex_cmd} eris_ifu_stdstar {stdstar_sof}")

def auto_gitter(rawdir, metafile, outdir='./', calib_pool='calibPool', static_pool='staticPool', 
                grating='', esorex=''):
    """calibrate the science target or the standard stars
    """
    static_pool = static_pool.rstrip('/')
    calib_pool = calib_pool.rstrip('/')
    meta_tab = table.Table.read(metafile, format='csv')
    if not os.path.isdir(outdir):
        os.system(f'mkdir -p {outdir}')
    science_sof = os.path.join(outdir, 'eris_ifu_jitter.sof')
    with open(science_sof, 'w+') as openf:
        # write OBJ
        for item in meta_tab[meta_tab['DPR.CATG'] == 'SCIENCE']:
            if item['DPR.TYPE'] == 'OBJECT':
                openf.write(f"{rawdir}/{item['DP.ID']}.fits.Z OBJ\n")
            elif item['DPR.TYPE'] == 'SKY':
                openf.write(f"{rawdir}/{item['DP.ID']}.fits.Z SKY_OBJ\n")
        openf.write(f"{calib_pool}/eris_ifu_distortion_distortion.fits DISTORTION\n")
        openf.write(f"{calib_pool}/eris_ifu_distortion_slitlet_pos.fits SLITLET_POS\n")
        openf.write(f"{calib_pool}/eris_ifu_flat_master_flat.fits MASTER_FLAT\n")
        openf.write(f"{calib_pool}/eris_ifu_flat_bpm.fits BPM_FLAT\n")
        openf.write(f"{calib_pool}/eris_ifu_wave_map.fits WAVE_MAP\n")
        openf.write(f"{static_pool}/EXTCOEFF_TABLE.fits EXTCOEFF_TABLE\n")
        openf.write(f"{static_pool}/eris_oh_spec.fits OH_SPEC\n")
        if grating in ['H_low', 'J_low', 'K_low']:
            openf.write(f"{static_pool}/RESPONSE_WINDOWS_{grating}.fits RESPONSE\n")
    os.system(f'{esorex} eris_ifu_jitter --product_depth=2 --tbsub=true --sky_tweak=1 --dar-corr=true --flux-calibrate=false {science_sof}')
        
def data_combine():
    """combine the multiple observation of the same target

    sigma_clip, apply if there are large number of frames to be combined
    background: global or chennel-per-channel and row-by-row
    """

def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.info('Started')
    pass
    logging.info('Finished')

# add main here to run the script from cmd
if __name__ == '__main__':
    pass
