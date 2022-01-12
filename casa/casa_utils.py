"""utilities with measurements set, casa table

"""
import os
import re
import json
import numpy as np
from astropy.coordinates import SkyCoord

try:
    # for casa6
    from casatool import table as tbtool
except:
    from taskinit import tbtool


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
            spw_specrange[key] = [freq_min, freq_max]

    return spw_specrange.values()

def read_refdir(vis):
    """read the reference direction and return the standard direction string
    """
    tb = tbtool()
    tb.open(vis+'/FIELD')
    reference_dir = tb.getcol('REFERENCE_DIR').flatten()
    tb.close()
    
    rad2deg = 180./np.pi
    direction = "J2000 " + SkyCoord(reference_dir[0]*rad2deg, reference_dir[1]*rad2deg, 
                                      unit="deg").to_string('hmsdms')
    return direction

