# a collection of utilities to handle measurements and asdm files and casa tables
# it only works alongside with casa

# by Jianhang Chen
# cjhastro@gmail.com 
# History: 
#   2020.03.21, first release

import os
import re
import glob
import numpy as np

# import tools from casa
## for casa 6
try:
    from casatools import table as tbtool
    from casatasks import listobs, importasdm
except:
    pass

## for casa 5
try:
    from importasdm import importasdm
    from listobs_cli import listobs_cli as listobs
except:
    pass

def import_rawdata(rawdir='../raw', outdir='./', overwrite=False, **kwargs):
    """This function is used to importasdm and generate the file descriptions
    """
    # import the asdm data into outdir
    for asdm in glob.glob(os.path.join(rawdir, '*.asdm.sdm')):
        basename = os.path.basename(asdm)[:-9]
        msfile = basename + '.ms'
        msfile_fullpath = os.path.join(outdir, msfile)
        if os.path.isdir(msfile_fullpath):
            if not overwrite:
                print("Reused existing data, set overwrite=True to overwrite.")
                continue
            else:
                print("Overwriting existing files...")
        importasdm(asdm=asdm, vis=os.path.join(outdir, basename+'.ms'), overwrite=overwrite, **kwargs)
    # generate the data disscription
    for obs in glob.glob(os.path.join(outdir, '*.ms')):
        obs_listobs = obs + '.listobst.txt'
        if os.path.isfile(obs_listobs):
            print("Reused existing listobs")
            continue
        else:
            listobs(vis=obs, listfile=obs_listobs, verbose=True, overwrite=overwrite)


def vis_jackknif(vis, copy=False, outdir=None):
    """make jackknifed image with only noise"""
    if copy:
        if outdir is None:
            raise ValueError('outdir must be given to do the copy!')
        if not os.path.isdir(outdir):
            os.system('mkdir {}'.format(outdir))
        vis_copied = os.path.join(outdir, os.path.basename(vis) + '.copy')
        os.system('cp -r {} {}'.format(vis, vis_copied))
        vis = vis_copied
    tb.open(vis, nomodify=False)

    for row in range(0, tb.nrows(), 2):
        key = 'r'+str(row+1)
        row_data = tb.getvarcol('DATA', row, 1)[key]
        row_data_inverse = (-1.+0j) * row_data
        #tb.removecorowkey)
        tb.putvarcol('DATA', {key:row_data_inverse}, row, 1)
        tb.flush()
    tb.close()
    return vis

def read_spw(vis):
    """read the spectral windows
    """
    tb = tbtool()
    if isinstance(vis, str):
        vis = [vis, ]

    if not isinstance(vis, list):
        raise ValueError("read_spw: Unsupported measurements files!")

    spw_specrange = {}

    for v in vis:
        tb.open(v + '/SPECTRAL_WINDOW')
        col_names = tb.getvarcol('NAME')
        col_freq = tb.getvarcol('CHAN_FREQ')
        tb.close()

        for key in col_names.keys():
            freq_max = np.max(col_freq[key]) / 1e9
            freq_min = np.min(col_freq[key]) / 1e9
            freq_interv = np.abs(np.mean(np.diff(col_freq[key], axis=0))) / 1e9
            freq_nchan = len(col_freq[key])
            spw_specrange[key] = [freq_min, freq_max, freq_interv, freq_nchan]

    return list(spw_specrange.values())
