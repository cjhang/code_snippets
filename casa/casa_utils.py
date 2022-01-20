"""utilities with measurements set, casa table

"""
import os
import re
import numpy as np

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
