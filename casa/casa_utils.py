# a collection of utilities to handle measurements and asdm files and casa tables
# it only works alongside with casa

# by Jianhang Chen
# cjhastro@gmail.com 
# History: 
    # 2020.03.21, first release
    # 2022.02.06, add ms tools

import os
import re
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



###########################################
# asdm tools
####################

def import_rawdata(rawdir='../raw', outdir='./', overwrite=False, **kwargs):
    """This function is used to importasdm and generate the file descriptions

    Paramters
    ---------
    rawdir : str
        the directory of raw files, ended with .asdm
    outdir : str
        the output directory
    overwrite : bool
        overwriting exisiting  folder with the same names
    kwargs:
        the kwargs passed to importasdm
    
    Examples
    --------
    In the project directory

        os.system('mkdir ./data')
        import_rawdata(rawdir='../raw', outdir='./data', overwrite=False)
    
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


###########################################
# ms tools
####################

def search_fields(msfile):
    try:
        from casatools import msmetadata
        msmd_local = msmetadata()
    except:
        print("Failed: cannot import msmetadata from casatools!")
        return 0
    fields = {}
    msmd_local.open(msfile)
    lookup = msmd.fieldnames()
    fields['gcal'] = lookup[msmd.fieldsforintent('*PHASE*')[0]]
    fields['bcal'] = lookup[msmd.fieldsforintent('*BANDPASS*')[0]]
    fields['pcal'] = lookup[msmd.fieldsforintent('*POLARIZATION*')[0]]
    fields['fcal'] = lookup[msmd.fieldsforintent('*FLUX*')[0]]
    msmd_local.close()
    return fields




###########################################
# Table tools
####################

from casatools import table, msmetadata

def vis_jackknif(vis, copy=False, outdir=None):
    """make jackknifed image with only noise"""
    tb = table()
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

def read_spwinfo(vis):
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

def read_spw(vis, norm=1e9, spw_nchans=None, spw_desc='FULL_RES'):
    """read the spw details

    Params:
        norm: the normalization factor, default is 1e9 converting the default Hz to GHz
    """
    tb = table()
    tb.open(os.path.join(vis + '/SPECTRAL_WINDOW'))
    col_names = tb.getvarcol('NAME')
    col_freq = tb.getvarcol('CHAN_FREQ')
    col_spwIDs = tb.getvarcol('ASSOC_SPW_ID')
    tb.close()
    
    spw_dict = {}
    for key in col_names.keys():
        selected = True
        if spw_nchans is not None:
            selected = selected & (len(col_freq[key]) == spw_nchans)
        if spw_desc is not None:
            selected = selected & (spw_desc in col_names[key][0])
        if selected:
            spw_id = col_spwIDs[key].flatten()[0] -1
            spw_freq = col_freq[key].flatten()/norm
            spw_dict[spw_id] = spw_freq
    return spw_dict

def search_spw(spw_dict, freqs=[], mode='range'):
    """generate the masked spw without the selected frequencies
    
    Params:
        freqs: the selected frequencies
        mode: the format of freqs. 
            - 'range': means freq=[[freq_min, freq_max], ...]
            - 'center': means freq=[[freq_center, width], ...]
    """
    if mode == 'center': # convert to the range format
        freqs_range = []
        for freq in freqs:
            freqs_range.append([freq[0]-freq[1], freq[0]+freq[1]])
        freqs = freqs_range
        mode = 'range'
    if mode == 'range':
        spw_str = ''
        for freq in freqs:
            for spw_id in spw_dict.keys():
                # spw_range = np.array([np.min(spw_dict[spw_id]), np.max(spw_dict[spw_id])])
                # if spw_range[0]<=freqs[0] and spw_range[1]>=freqs[1]:
                spw_chan = np.arange(len(spw_dict[spw_id]))
                spw_sign = np.sign(np.diff(spw_dict[spw_id][0:2]))
                if spw_sign > 0:
                    lower_selection = spw_dict[spw_id] < freq[0]
                    upper_selection = spw_dict[spw_id] >= freq[1]
                    if np.sum(lower_selection) > 0:
                        chan_low = np.max(spw_chan[lower_selection])
                    else:
                        chan_low = 0
                    if np.sum(upper_selection) > 0:
                        chan_up = np.min(spw_chan[upper_selection])
                    else: 
                        chan_up = spw_chan[-1]
                else:
                    lower_selection = spw_dict[spw_id] < freq[1]
                    upper_selection = spw_dict[spw_id] >= freq[0]
                    if np.sum(lower_selection) > 0:
                        chan_low = np.min(spw_chan[lower_selection])
                    else: chan_low = spw_chan[-1]
                    if np.sum(upper_selection) > 0:
                        chan_up = np.max(spw_chan[upper_selection])
                    else: chan_up = 0
                if chan_low < chan_up:
                    spw_str = spw_str+'{}:{}~{},'.format(spw_id, chan_low, chan_up)
    return spw_str[:-1]

def read_intent(vis, unique=True):
    """read the intents of the observations

    This function is initially used for ALMACAL project where each file only has one field
    
    """
    tb = table()
    tb.open(vis)
    intent_idx = np.unique(tb.getcol('STATE_ID'))
    tb.close()
    tb.open(vis+'/STATE')
    intents = tb.getcol('OBS_MODE')
    intents_list = []
    for idx in intent_idx:
        intents_list.append(intents[idx])

    intents_valid = []
    for item in intents_list:
        # print('item', item)
        if "BANDPASS" in item:
            intents_valid.append('BANDPASS')
        if "PHASE" in item:
            intents_valid.append('PHASE')
        if "FLUX" in item:
            intents_valid.append('FLUX')
    if unique:
        intents_valid = np.unique(intents_valid).tolist()
    return intents_valid


def search_fields(msfile):
    fields = {}
    msmd.open(msfile)
    lookup = msmd.fieldnames()
    fields['gcal'] = lookup[msmd.fieldsforintent('*PHASE*')[0]]
    fields['bcal'] = lookup[msmd.fieldsforintent('*BANDPASS*')[0]]
    fields['pcal'] = lookup[msmd.fieldsforintent('*POLARIZATION*')[0]]
    fields['fcal'] = lookup[msmd.fieldsforintent('*FLUX*')[0]]
    fields['science'] = lookup[msmd.fieldsforintent('*TARGET*')[0]]
    msmd.close()
    return fields

def flagchannels(vis):
    """this function used to generate the channels to be flagged
    Useful to flag the edge channels 

    """
    #TODO
    pass


if has_astropy:
    def read_refdir(vis, return_coord=False):
        """read the reference direction and return the standard direction string
        """
        tb = table()
        tb.open(vis+'/FIELD')
        reference_dir = tb.getcol('REFERENCE_DIR').flatten()
        tb.close()
        
        rad2deg = 180./np.pi
        ref_coord = SkyCoord(reference_dir[0]*rad2deg, reference_dir[1]*rad2deg, 
                                          unit="deg")
        direction = "J2000 " + ref_coord.to_string('hmsdms')
        if return_coord:
            return ref_coord

        return direction



