#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""A minimalist tool to handle visibility

Author: Jianhang Chen, cjhastro@gmail.com

Requirement:

History:
    2024-10-25: first release, basic functions, v0.1
    2025-03-10: 

"""

__version__ = '0.1.0'

import time
import logging
import numpy as np
import os, gzip, sys, json, shutil


import matplotlib.pyplot as plt
# import astropy.units as u
# import astropy.constants as const

from scipy.fft import fft2, fftshift
from numpy import fft
from scipy import interpolate, special, optimize, linalg
from astropy.io import fits
import emcee
from multiprocessing import Pool

try:
    import pynufft
except:
    pass
try:
    from mpfit import mpfit
    from capfit import capfit
except:
    pass

const_c = 299792458.0 # m/s
arcsec2rad = (np.pi/(180.*3600.))

#######################################################
# visibility data handling
#######################################################

class UVdata:
    def __init__(self, u=None, v=None, w=None, real=None, imag=None,
                 weight=None, sigma=None, mask=None, frequency=None,
                 polarization=None, PBfwhm=None, phase_center=None):
        self.u = u
        self.v = v
        self.real = real
        self.imag = imag
        self.mask = mask
        self.weight = weight
        self.frequency = weight
        self.polarization = polarization
        self.PBfwhm = PBfwhm
        self.phase_center = phase_center
    def __add__(self, uvdata):
        uvdict_combined = combine_uvdict(self.to_dictionary(), uvdata.to_dictionary())
        return self.from_dictionary(uvdict_combined)
    @property
    def visi(self):
        return self.real + self.imag*1j
    @visi.setter
    def visi(self, val):
        self.real = val.real
        self.imag = val.imag
    @property
    def amplitude(self):
        return np.ma.absolute(self.visi)
    @property
    def angle(self):
        return np.rad2deg(np.angle(self.visi))
    @property
    def phase_center(self):
        return self.phase_center
    @property
    def PBfwhm(self):
        return self.PBfwhm
    def uvdist(self, unit=None):
        if unit is None:
            return np.hypot(self.u, self.v)
        if unit == 'm':
            return np.hypot(self.u*reffreq/const_c, 
                            self.v*reffreq/const_c)
    def shift_phase(self, delta_ra, delta_dec):
        """shift the phase of the source

        Args:
            delta_ra: positive value shifts towards the east
            delta_dec: positive value shifts towards the north
        """
        if delta_ra != 0 and delta_dec!=0:
            phi = self.u * 2*np.pi * delta_ra + self.v * n*np.pi * delta_dec
            visi = (self.real + 1j*self.imag) * (np.cos(phi) * 1j*np.sin(phi))
        self.real = visi.real
        self.imag = visi.imag
        #TODO: change also the phase_center
    def rotate(self, theta):
        """rotate the uv coordinate counter-clockwise with an radian: theta
        """
        if theta != 0:
            u_new = self.u * np.cos(theta) - self.v * np.sin(theta)
            v_new = self.u * np.sin(theta) + self.v * np.cos(theta)
        self.u = u_new
        self.v = v_new
    def deproject(self, inc):
        """deproject the visibility with inclination angel: inc"""
        self.u = self.u * np.cos(inc)

    def from_dictionary(self, uvdata):
        """load the uvdata from the dictionary
        """
        # check the dictionary have all the required info
        required_colnames = ['u', 'v', 'real', 'imag', 'weight', 'mask', 
                             'frequency', 'polarization']
        dict_names = list(uvdata.keys())
        for colname in required_colnames:
            if colname not in dict_names:
                raise ValueError(f"{colname} is not found in the dictionary")
        self.u, self.v = uvdata['u'], uvdata['v']
        self.real = uvdata['real']
        self.imag = uvdata['imag']
        self.weight = uvdata['weight']
        self.mask = uvdata['mask']
        if 'frequency' in dict_names:
            self.frequency = uvdata['frequency']
        if 'polarization' in dict_names:
            self.polarization = uvdata['polarization']
        if 'PBfwhm' in dict_names:
            self.PBfwhm = uvdata['PBfwhm']
        if 'phase_center' in dict_names:
            self.phase_center = uvdata['phase_center']
    def to_dictionary(self):
        uvdata = {}
        uvdata['u'] = self.u
        uvdata['v'] = self.v
        uvdata['real'] = self.real
        uvdata['imag'] = self.imag
        uvdata['weight'] = self.weight
        uvdata['mask'] = self.mask
        uvdata['frequency'] = self.frequency
        uvdata['polarization'] = self.polarization
        uvdata['PBfwhm'] = self.PBfwhm
        uvdata['phase_center'] = self.phase_center
        return uvdata
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,7))
        ax.scatter(self.u, self.v)
    def write(self, filename):
        save_uvdata(self.to_dictionary(), filename)
    def read(self, filename):
        self.from_dictionary(read_uvdata(filename))

def combine_uvdict(uvdict1, uvdict2, check_phase_center=True):
    """combine two uvdict
    """
    if check_phase_center:
        if (uvdict1['phase_center'] is not None) \
                and (uvdict2['phase_center'] is not None):
            if np.any(uvdict1['phase_center'] != uvdict2['phase_center']):
                print(uvdict1['phase_center'], uvdict2['phase_center'])
                raise ValueError('Merging two uvdata with different phase center!')
            else:
                phase_center = uvdict1['phase_center']
        else: # choose one of the not None phase_center
            phase_center = choose_not_none([uvdict1['phase_center'], 
                                            uvdict2['phase_center']])
    else:
        phase_center = None
    if (uvdict1['PBfwhm'] is not None) and (uvdict2['PBfwhm'] is not None):
        if uvdict1['PBfwhm'] != uvdict2['PBfwhm']:
            PBfwhm = np.min([uvdict1['PBfwhm'], uvdict2['PBfwhm']])
            logging.warning("Inconsistent primary beam, adopted the smaller one!")
    else:
        PBfwhm = choose_not_none([uvdict1['PBfwhm'], uvdict2['PBfwhm']])
    return {'u':np.hstack([uvdict1['u'], uvdict2['u']]), 
            'v':np.hstack([uvdict1['v'], uvdict2['v']]),
            'real':np.hstack([uvdict1['real'], uvdict2['real']]), 
            'imag':np.hstack([uvdict1['imag'], uvdict2['imag']]),
            'weight':np.hstack([uvdict1['weight'], uvdict2['weight']]), 
            'mask':np.hstack([uvdict1['mask'], uvdict2['mask']]),
            'frequency':np.hstack([uvdict1['frequency'], uvdict2['frequency']]),
            'polarization':np.hstack([uvdict1['polarization'], uvdict2['polarization']]),
            'PBfwhm': np.mean([uvdict1['PBfwhm'], uvdict2['PBfwhm']]),
            'phase_center':phase_center}

def save_uvdata(uvdata, filename, filetype=None):
    """Unified way to save and load uvdata

    filename: without suffix
    """
    if filetype is None:
        if ('.txt' in filename) or ('.dat' in filename) or ('.gz' in filename):
            filetype = 'text'
        elif ('.np' in filename) or ('.npz' in filename):
            filetype = 'binary'
    if filetype == 'text':
        header_meta = f'PBfwhm={uvdata["PBfwhm"]},phase_center={uvdata["phase_center"]}\n'
        header_colnames = 'u\tv\treal\timag\tweight\tfrequency'
        data_combined = np.column_stack([uvdata['u'], uvdata['v'], 
                                    uvdata['real'], uvdata['imag'],
                                    uvdata['weight'], uvdata['frequency'],
                                    uvdata['polarization']])
        np.savetxt(filename, data_combined,
                   fmt=6*'%10.8e\t'+'%2d', 
                   header=header_meta+header_colnames)
    elif filetype == 'binary':
        np.savez(filename, **uvdata)

def read_uvdata(filename, filetype=None):
    """Unified way to read the uvdata from either text file or binary files

    Return:
        dictionary
    """
    if filetype is None:
        if ('.txt' in filename) or ('.dat' in filename) or ('.gz' in filename):
            filetype = 'text'
        elif ('.np' in filename) or ('.npz' in filename):
            filetype = 'binary'
    if filetype == 'text':
        dtype={'names': ('u','v','real','imag','weight','frequency','polarization'),
               'formats': ('f8','f8','f8','f8','f8','f8', '<i4')}
        uvload = np.loadtxt(filename, dtype=dtype)
        # read the meta data from comments
        if '.gz' in filename:
            with gzip.open(filename,'rt') as fp:
                meta = fp.readline().split(',')
                names = fp.readline().split()
        else:
            with open(filename,'r') as fp:
                meta = fp.readline().split(',')
                names = fp.readline().split()
        phase_center = meta[1].split('=')[-1].strip()
        PBfwhm = float(meta[0].split('=')[-1].strip())
       
    elif filetype == 'binary':
        uvload = np.load(filename)
        phase_center = uvload['phase_center']
        PBfwhm = uvload['PBfwhm']
    # convet it into dictionary
    uvdata = {}
    uvdata['u'], uvdata['v'] = uvload['u'], uvload['v'], 
    uvdata['real'], uvdata['imag'] = uvload['real'], uvload['imag']
    uvdata['weight'], uvdata['frequency'] = uvload['weight'], uvload['frequency']
    uvdata['polarization'] = uvload['polarization']
    uvdata['phase_center'] = phase_center
    uvdata['PBfwhm'] = PBfwhm
    return uvdata

def read_uv_ALMA(vis, field=None, spw=None, chunk_size=None, average_time=None, 
                 average_channel=True, average_polarization=True, 
                 outfile=None, uv_unit='wavelength'):
    """read uv data from ALMA observations


    The weighting is crucial for uv data modelling, 
    For ALMA or VLA data, it is necessary to know that ALMA changes the 
    weighting scheme overtime and with/without channel averaging, check:
    https://casaguides.nrao.edu/index.php/DataWeightsAndCombination

    For simplicity, it is recommended to run statwt before concat or combine
    your science data
    """
    try:
        from casatools import table as _tbtool
        from casatools import msmetadata as _msmetadata
    except:
        print('Install "casatools" or execute the code within the casa shell!')
        return 0

    # define some global variables
    const_c = 299792458.0 # m/s
    tb = _tbtool()
    if spw is not None:
        spw = int(spw)
        # if isinstance(spw, str): 
            # spw_list = spw.split(',')
        # else:
            # spw_list = [str(spw),]
    else:
        # select all the spectral window
        tb.open(vis+'/DATA_DESCRIPTION')
        spw_ids = tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()
        spw_list = spw_ids.astype(str).tolist()
        if len(spw_list) > 1:
            raise ValueError('More than one spectral window found in {vis}, please specify the spw')
        else:
            spw = int(spw_list[0])
    
    # read the field info
    tb.open(vis+'/FIELD')
    fields_names = tb.getcol('NAME')
    fields_ids = tb.getcol('SOURCE_ID')
    fields_phase_center = np.squeeze(tb.getcol('PHASE_DIR'))
    # fields_phase_center = np.squeeze(tb.getcol('REFERENCE_DIR'))
    tb.close()
    if field is None:
        if len(fields_ids) > 1:
            raise ValueError('More than one field found, please specify the field name!')
        field = fields_ids[0]

    if field is not None:
        # select the data only for field
        if isinstance(field, str):
            # change the string field to index
            field = field_ids[fields_names==field]
        if len(fields_ids)>1:
            phase_center = np.rad2deg(fields_phase_center[:,field])
        else:
            phase_center = np.rad2deg(fields_phase_center)
    else:
        logging.warning('No valid phase center can be found!')
        phase_center = ''
    
    # Get the frequencies of each spectral window 
    tb.open(vis+'/SPECTRAL_WINDOW')
    chanfreqs = tb.getvarcol('CHAN_FREQ')
    tb.close()
    if len(chanfreqs) > 0:
        spw_freqs = chanfreqs['r'+str(spw+1)].flatten()
        reffreq = np.mean(spw_freqs)
        # if spw_list is not None:
            # for spw_single in spw_list:
                # spw_freq = chanfreqs['r'+str(int(spw_single)+1)].flatten()
                # spw_freqs[spw_single]=spw_freq
    else:
        raise ValueError('Cannot find the frequencies of the spectral window!')
      
    # Get the primary beam size
    tb.open(vis+'/ANTENNA')
    diam = np.median(tb.getcol('DISH_DIAMETER')) # be careful if mixing arrays 
    tb.close()
    PBfwhm = compute_PBfwhm(reffreq, diam) # in arcsec
 
    # read some critical info from the whole ms, but avoid reading the whole data
    tb.open(vis)
    nrows = tb.nrows()
    colnames = tb.colnames()
    if 'CORRECTED_DATA' in colnames:
        data_colname = 'CORRECTED_DATA'
    else:
        data_colname = 'DATA'
    data_desc_ids = tb.getcol('DATA_DESC_ID')
    unique_spws = np.unique(data_desc_ids)
    # check whether the spw id in the unique data_desc_ids
    if spw not in unique_spws:
        raise ValueError(f"Canoot find SPW{spw} in {vis}")
    # spw_check = [True if spw_i in unique_spws else False for spw_i in spw_list]
    # if not np.all(spw_check):
        # spw_list_valid = []
        # for spw_id in spw_list:
            # if int(spw_id) not in unique_spws:
                # logging.warning(f'SPW{spw_id} not found in the data!')
                # continue
            # spw_list_valid.append(spw_id)
        # if len(spw_list_valid) < 1:
            # raise ValueError('No valid spw exists!')
        # spw_list = spw_list_valid
    # define the global selection array
    select = np.ones(nrows, dtype=bool)
    # select the rows of the targeted field
    if field is not None:
        field_ids = tb.getcol('FIELD_ID')
        field_select = (field_ids == field)
        if np.sum(field_select) < 1:
            raise ValueError('Field selection failed!')
        select = select & field_select
    # select the rows of selected spw
    spw_select = np.zeros(nrows, dtype=bool)
    spw_select[data_desc_ids==int(spw)] = True
    if np.sum(spw_select) < 1:
        raise ValueError('SPW selection failed!')
    select = select & spw_select
    # print(np.sum(select), np.sum(spw_select), np.sum(field_select))
    # if len(spw_list) > 0:
        # # define the selection array
        # spw_select = np.zeros(nrows, dtype=bool)
        # for spw_id in spw_list: 
            # spw_select[data_desc_ids==int(spw_id)] = True
        # select = select & spw_select
    
    rows_select = np.where(select)[0]
    data_desc_ids_select = data_desc_ids[rows_select]
    tb_select = tb.selectrows(rows_select);
    data_orig = tb_select.getcol(data_colname)
    data_shape = data_orig.shape
    data_mask = tb_select.getcol('FLAG')
    data_u, data_v, data_w = tb_select.getcol('UVW') # with origin unit of "m"
    # data_time = tb_select.getcol('TIME')
    data_weight = tb_select.getcol('WEIGHT')
    # data_weight = np.expand_dims(data_weight, axis=1) # add the dimention for channels
    # by default, first add uniform weighting along the spectral axis
    data_weight = np.repeat(data_weight[:,None,:], data_shape[1], axis=1)

    if average_polarization:
        # average the two polarisations and change the shape to [1, nchan, nrow]
        data_real = np.ma.array(data_orig.real, mask=data_mask)
        data_imag = np.ma.array(data_orig.imag, mask=data_mask)
        # data_real = np.ma.sum(data_real * data_weight[:,None,:], axis=0) / ( 
                              # np.sum(data_weight, axis=0))
        data_real = np.ma.average(data_real, weights=data_weight, axis=0)
        data_imag = np.ma.average(data_imag, weights=data_weight, axis=0)
        data_orig = np.expand_dims(data_real.data+data_imag.data*1j, axis=0)
        data_weight = np.expand_dims(np.sum(data_weight, axis=0), axis=0)
        data_mask = np.expand_dims(data_real.mask, axis=0)
        data_pol = np.full(data_orig.shape, 11) # only one pol left
    else:
        data_shape = data_orig.shape
        if data_shape[0] == 4:
            # add pol XX, XY, YX, YY = 11, 12, 21, 22
            data_pol = np.repeat([11,12,21,22], data_shape[1]*data_shape[2]).reshape(data_shape)
        elif data_shape[0] == 2:
# add pol XX, YY = 11, 22
            data_pol = np.repeat([11,22], data_shape[1]*data_shape[2]).reshape(data_shape)
        # data_weight = np.expand_dims(data_weight, axis=1)
        # data_sigma = np.expand_dims(data_weight, axis=1)
        # data_real = np.ma.sum(data_real * data_weight, axis=0) / ( 
                              # np.sum(data_weight, axis=0))
        # data_imag = np.ma.sum(data_imag * data_weight, axis=0) / ( 
                              # np.sum(data_weight, axis=0))
        # uvdata['data'] = np.sum(uvdata['data'] * polarization_weights, axis=0) / ( 
                         # np.sum(polarization_weights, axis=0))
        # data_sigma = np.sum(data_sigma**-2, axis=0)**-0.5
    if average_channel:
        data_real = np.ma.array(data_orig.real, mask=data_mask)
        data_imag = np.ma.array(data_orig.imag, mask=data_mask)
        if "WEIGHT_SPECTRUM" in colnames:
            chan_weight = tb_select.getcol('WEIGHT_SPECTRUM')
        else:
            if data_real.shape[1] > 1:
                logging.warning('No info for weight_spectrum in the header, assuming uniform weights')
            chan_weight = np.ones(data_real.shape[1])/data_real.shape[1]
        data_real = np.ma.average(data_real, axis=1, weights=chan_weight) 
        data_imag = np.ma.average(data_imag, axis=1, weights=chan_weight) 
        data_orig = np.expand_dims(data_real.data+data_imag.data*1j, axis=1)
        data_weight = np.expand_dims(np.sum(data_weight, axis=-2), axis=1)
        data_mask = np.expand_dims(data_real.mask, axis=1)
        # spw_freqs[spw_id] = np.average(spw_freqs[spw_id], weights=chan_weight)
        # print(spw_freqs)
        # print(len(spw_freqs), chan_weight.shape)
        # reffreq = np.average(spw_freqs, weights=chan_weight)
        spw_freqs = np.array([reffreq])
        data_freq = np.full_like(data_real.data, reffreq)
        data_pol = np.average(data_pol, axis=1)
    else:
        data_shape = data_orig.shape
        data_freq = np.tail(spw_freqs, 
                            [data_shape[0], data_shape[-1]]).reshape(data_shape)

        # data_real = np.ma.average(data_real * data_weight, axis=-2) / np.sum(data_weight, axis=-2)
        # data_imag = np.ma.average(data_imag * data_weight, axis=-2) / np.sum(data_weight, axis=-2)
        # data_sigma = np.sum(data_sigma**-2, axis=-2)**-0.5 # just to drop the channel axis

    # channel weigting needs to be careful, as the uv coordinates changes with
    # frequency
    
    
    # convert uvw coords from m to wavelengths
    # follow data_uvw *= reffreq/const_c
    # the following code deal with the different shapes due to the averaging
    # it takes care of the frequency difference (k\lambda difference) for channels
    if uv_unit == 'm':
        pass # the default
    if uv_unit == 'wavelength':
        data_shape = data_orig.shape
        data_u = np.tile(data_u, [data_shape[0], data_shape[1], 1])/const_c * spw_freqs[None,:,None]
        data_v = np.tile(data_v, [data_shape[0], data_shape[1], 1])/const_c * spw_freqs[None,:,None]

    uvdata = {'u': data_u.flatten(),
              'v': data_v.flatten(),
              # 'time': data_time,
              # 'antenna1': tb_select.getcol('ANTENNA1'),
              # 'antenna2': tb_select.getcol('ANTENNA2'), 
              'real': data_real.data.flatten(),
              'imag': data_imag.data.flatten(),
              'mask': data_real.mask.flatten(),
              'weight': data_weight.flatten(),
              'frequency': data_freq.flatten(),
              'polarization': data_pol.flatten(),
              'phase_center': phase_center,
              'PBfwhm': PBfwhm,
              'uv_unit': uv_unit}
    if outfile is not None:
        save_uvdata(uvdata, outfile)
    else:
        return uvdata

def read_uvfits(uvfits, average_channel=False, uv_unit='wavelength'):
    """read uv data from uvfits file

    Arga:
        uv_unit: the units of the uv coordinates
                 options: 'm' or 'wavelength'
    Note:
    * uvfits keep the masked data as negative weights
    """
    with fits.open(uvfits) as hdu:
        header = hdu[0].header
        fitsdata = hdu[0].data
    chanfreq = (header['CRVAL4'] + (np.arange(1, header['NAXIS4']+1)-header['CRPIX4']) * header['CDELT4'])
    if len(chanfreq) == 1:
        chanfreq = chanfreq[0]
    ndim = fitsdata['DATA'].ndim
    nstokes, ncol = fitsdata['DATA'].shape[-2:]

    if ndim == 6:
        # one data channel 
        if nstokes == 1:
            _data_real = fitsdata['DATA'][:,0,0,0,0,0]
            _data_imag = fitsdata['DATA'][:,0,0,0,0,1]
            data_weight = fitsdata['DATA'][:,0,0,0,0,2]
            data_mask = data_weight < 0
            data_real = np.ma.array(_data_real, mask=data_mask)
            data_imag = np.ma.array(_data_imag, mask=data_mask)
        elif nstokes == 2:
            _data_real = fitsdata['DATA'][:,0,0,0,:,0]
            _data_imag = fitsdata['DATA'][:,0,0,0,:,1]
            data_weight = fitsdata['DATA'][:,0,0,0,:,2]
            data_mask = data_weight < 0
            data_real = np.ma.array(_data_real, mask=data_mask)
            data_imag = np.ma.array(_data_imag, mask=data_mask)
            # average two polarisation
            print(f"Averaging the {nstokes} polarizations")
            data_real = np.ma.average(data_real, axis=1, weights=data_weight)
            data_imag = np.ma.average(data_imag, axis=1, weights=data_weight)
            data_weight = np.ma.sum(np.ma.array(data_weight, mask=data_mask),
                                    axis=1).filled(0.)
        else:
            raise ValueError("Unsupported data, possibly due to the {} stokes dimension?")
    if ndim == 7:
        # multi-channel data
        if nstokes == 1:
            _data_real = fitsdata['DATA'][:,0,0,0,:,0,0]
            _data_imag = fitsdata['DATA'][:,0,0,0,:,0,1]
            data_weight = fitsdata['DATA'][:,0,0,0,:,0,2]
            data_mask = data_weight < 0
            data_real = np.ma.array(_data_real, mask=data_mask)
            data_imag = np.ma.array(_data_imag, mask=data_mask)
        elif nstokes == 2:
            _data_real = fitsdata['DATA'][:,0,0,0,:,:,0]
            _data_imag = fitsdata['DATA'][:,0,0,0,:,:,1]
            data_weight = fitsdata['DATA'][:,0,0,0,:,:,2]
            data_mask = data_weight < 0
            print(f"Averaging the {nstokes} polarizations")
            data_real = np.ma.array(_data_real, mask=data_mask)
            data_imag = np.ma.array(_data_imag, mask=data_mask)
            data_real = np.ma.average(data_real, axis=2, weights=data_weight)
            data_imag = np.ma.average(data_imag, axis=2, weights=data_weight)
            data_weight = np.ma.sum(np.ma.array(data_weight, mask=data_mask),
                                    axis=2).filled(0)
        if average_channel:
            print("Averaging along the channels")
            data_real = np.ma.average(data_real, axis=1, weights=data_weight)
            data_imag = np.ma.average(data_imag, axis=1, weights=data_weight)
            data_weight = np.ma.sum(np.ma.array(data_weight, mask=data_weight<1e-8),
                                    axis=1).filled(0.)
            chanfreq = np.mean(chanfreq)
    if uv_unit == 'm':
        # fitsdata['UU'] and fitsdata['VV'] is in units of arcseconds
        data_u = const_c * fitsdata['UU'] # in unit of m
        data_v = const_c * fitsdata['VV']
    elif uv_unit == 'wavelength':
        # data_shape = data_real.shape
        data_u = fitsdata['UU'] * chanfreq
        data_v = fitsdata['VV'] * chanfreq
    elif uv_unit == None:
        data_u = fitsdata['UU']
        data_v = fitsdata['VV']
    uvmask = data_real.mask
    uvdata = {
            'u': data_u,
            'v': data_v,
            'real': data_real.data,
            'imag': data_imag.data,
            'mask': uvmask,
            'weight': data_weight,
            'frequency': chanfreq,
            'uv_unit': uv_unit,
            }
    return uvdata


#######################################################
# Fitter Wrappers
def gaussian_log_prior(x, mean, sigma):
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(x-mean)**2/sigma**2

class Parameter:
    """the base class of model parameter (copy from model_utils.py)
    """
    def __init__(self, value, limits=None, fprior=None, prior=None,
                 name=''):
        if isinstance(value, Parameter):
            self.value = value.value
            self.limits = value.limits
            self.fprior = value.fprior
            self.prior = value.prior
            self.name = value.name
        else:
            self.value = value
            self.limits = limits
            self.fprior = fprior
            self.prior = prior
            self.name = name
        if self.fprior is None:
            self.fprior = gaussian_log_prior #lambda x: gaussian_log_prior(x, *prior)
    @property
    def is_fixed(self):
        if self.limits is None:
            return True
        else:
            return False
    def set_prior(self, prior, fprior=None):
        self.prior = prior
        if self.fprior is None:
            self.fprior = gaussian_log_prior
    def get_log_prior(self, value):
        if self.limits is None:
            return 0
        limits = self.limits
        if limits[0] <= value <= limits[1]:
            if self.prior is None:
                return 0
            else:
                return self.fprior(value, *self.prior)#self.fprior(value)
        else:
            return -np.inf

class Model:
    def __init__(self, name=None):
        self._name = name
    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name
    @name.setter
    def name(self, name):
        self._name = name
    @property
    def parameters(self):
        params_list = {}
        for key,value in self.__dict__.items():
            if not isinstance(value, Parameter):
                continue
            if not value.is_fixed:
                params_list[self.name+'.'+key] = value.value
        return params_list

class LineModel(Model):
    def __init__(self, m=None, b=None, name=None):
        super().__init__(name=name)
        self.m = Parameter(m)
        self.b = Parameter(b)
    def evaluate(self, x):
        return self.m.value * x + self.b.value

class Sersic2D(Model):
    def __init__(self, amplitude=None, reff=None, n=None, x0=None, y0=None, 
                 ellip=None, theta=None, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude)
        self.reff = Parameter(reff)
        self.n = Parameter(n)
        self.x0 = Parameter(x0)
        self.y0 = Parameter(y0)
        self.ellip = Parameter(ellip)
        self.theta = Parameter(theta)
        self.name = name
    def evaluate(self, grid):
        x, y = grid.create_meshgrid()
        bn = special.gammaincinv(2.0 * self.n.value, 0.5)
        cos_theta = np.cos(self.theta.value)
        sin_theta = np.sin(self.theta.value)
        x_maj = np.abs((x - self.x0.value) * cos_theta + (y - self.y0.value) * sin_theta)
        x_min = np.abs(-(x - self.x0.value) * sin_theta + (y - self.y0.value) * cos_theta)

        b = (1 - self.ellip.value) * self.reff.value
        expon = 2.0
        inv_expon = 1.0 / expon
        z = ((x_maj / self.reff.value) ** expon + (x_min / b) ** expon) ** inv_expon
        return self.amplitude.value * np.exp(-bn * (z ** (1 / self.n.value) - 1.0))

class ExpSersic3D(Model):
    """Exponential disk with Sersic2D flux density
    """
    def __init__(self, vmax=None, vsigma=None, inc=None, pa=None, 
                 amplitude=None, reff=None, n=None, x0=None, y0=None, 
                 ellip=None, theta=None, name=None):
        super().__init__(name=name)
        self.vmax = Parameter(vmax)
        self.vsigma = Parameter(vsigma)
        self.inc = Parameter(inc)
        self.pa = Parameter(pa)
        self.amplitude = Parameter(amplitude)
        self.reff = Parameter(reff)
        self.n = Parameter(n)
        self.x0 = Parameter(x0)
        self.y0 = Parameter(y0)
        #self.ellip = Parameter(ellip) # defined by inc
        #self.theta = Parameter(theta) # defined by pa
        self.name = name
    def evaluate(self, grid):
        chandata = grid.chandata
        x, y = grid.create_meshgrid()

        rdisk = np.sqrt(x**2 + y**2)
        theta_from_pa = 0.5*np.pi - self.pa# 
        x_new = x*np.cos(theta_from_pa) - y*np.sin(theta_from_pa)
        y_new = x*np.sin(theta_from_pa) + y*np.cos(theta_from_pa)
        vel_disk = exponential(rdisk, vmax=vmax)
        pa_disk = np.arctan2(y_new, x_new)
        vel_map = vel_disk*np.sin(inc)*np.cos(pa_disk)

        bn = special.gammaincinv(2.0 * self.n.value, 0.5)
        # cos_theta = np.cos(self.theta.value)
        # sin_theta = np.sin(self.theta.value)
        cos_theta = np.cos(theta_from_pa)
        sin_theta = np.sin(theta_from_pa)
        x_maj = np.abs((x - self.x0.value) * cos_theta + (y - self.y0.value) * sin_theta)
        x_min = np.abs(-(x - self.x0.value) * sin_theta + (y - self.y0.value) * cos_theta)

        b = (1 - self.ellip.value) * self.reff.value
        expon = 2.0
        inv_expon = 1.0 / expon
        z = ((x_maj / self.reff.value) ** expon + (x_min / b) ** expon) ** inv_expon
        flux = self.amplitude.value * np.exp(-bn * (z ** (1 / self.n.value) - 1.0))

        cube_data = np.zeros((nchan, ny, nx))
        vel_chan = np.linspace(-2000,2000,nchan)
        for yi in range(ny):
            for xi in range(nx):
                flux_i = flux_map[yi, xi]
                vel_i = vel_map[yi, xi]
                cube_data[:, yi, xi] = gaussian_1d([
                    flux_i, vel_i, vel_dispersion], vel_chan) + np.random.randn(nchan)
        return cube_data

class ScaleUncertainty(Model):
    def __init__(self, factor=None, name=None):
        super().__init__(name=name)
        self.factor = factor #Parameter(factor)
    def evaluate(self, x):
        return 0

class Models:
    """
    If a model compose of several same Models, they should have different name
    """
    def __init__(self, models):
        self.models = models
        # assigned new names for these who share the same name
        model_names = []
        model_dict = {}
        name_idx = 1
        for model in models:
            if model.name not in model_names:
                model_names.append(model.name)
            else:
                model.name = model.name + f'_{name_idx}'
                name_idx += 1
            setattr(self, model.name, model)
            model_dict[model.name] = model
        self.model_names = model_names
        self.model_dict = model_dict
    def __getitem__(self,s):
        if isinstance(s, (slice, int)):
            return self.models[s]
        elif isinstance(s, str):
            return self.model_dict[s]
    def get_parameters(self):
        params_list = {}
        for model in self.models:
            for key,value in model.__dict__.items():
                if not isinstance(value, Parameter):
                    continue
                if not value.is_fixed:
                    params_list[model.name+'.'+key] = value.value
        return params_list
    def get_limits(self):
        limits_list = {}
        for model in self.models:
            for key,value in model.__dict__.items():
                if not isinstance(value, Parameter):
                    continue
                if not value.is_fixed:
                    limits_list[model.name+'.'+key] = value.limits
        return limits_list
 
    def update_parameters(self, params_dict):
        for model in self.models:
            for key in params_dict.keys():
                model_name, model_param_name = key.split('.')
                if model_param_name in model.__dict__.keys():
                    model.__dict__[model_param_name].value = params_dict[key]

    def set_priors(self, params_dict):
        for model in self.models:
            for key in params_dict.keys():
                model_name, model_param_name = key.split('.')
                if model_name != model.name:
                    continue
                if model_param_name in model.__dict__.keys():
                    model.__dict__[model_param_name].set_prior(params_dict[key])

    def get_priors(self, params_dict):
        log_priors = 0
        for model in self.models:
            for key in params_dict.keys():
                model_name, model_param_name = key.split('.')
                if model_name != model.name:
                    continue
                if model_param_name in model.__dict__.keys():
                    # print(model_param_name, model.__dict__[model_param_name].get_log_prior(params_dict[key]))
                    log_priors += model.__dict__[model_param_name].get_log_prior(params_dict[key])
        return log_priors
    def model_on_grid(self, grid):
        # parameter_names = self.get_parameters()
        # self.update_parameters(dict(zip(parameter_names, theta)))
        model_value = 0.
        for model in self.models:
            model_value += model.evaluate(grid)
        return model_value


class Grid2D_test:
    def __init__(self, imsize, pixelsize=1):
        if not isinstance(imsize, (list,tuple,np.ndarray)):
            self._imsize = np.array([imsize,])
        else:
            self._imsize = np.array(imsize)
        if isinstance(pixelsize, (int, float)):
            self._pixelsize = np.array([pixelsize,])
        elif isinstance(pixelsize, (tuple, list, np.ndarray)):
            self._pixelsize = np.array(pixelsize)
        self._uvgrid = None
        # sort the pixelsize and imsize, so we always start with
        # the smallest grid
        self.index_sort = np.argsort(self._pixelsize)

    @property
    def imsize(self):
        return self._imsize[self.index_sort]
    @property
    def pixelsize(self):
        return self._pixelsize[self.index_sort]
    @property
    def uvgrid(self):
        if self._uvgrid is not None:
            return self._uvgrid
        imsize = self.create_meshgrid()[0].shape[0]
        kmax = 0.5/(np.min(self.pixelsize)*np.pi/180/3600) # arcsec to radian
        self._uvgrid = np.linspace(-kmax, kmax, imsize)
        return self._uvgrid
    def create_meshgrid(self):
        n_depth = len(self.imsize)
        grid_coords = []
        grid_min = 0
        grid_max = 0
        # loop through the pixelsizes, from smallest to largest
        # add calculate the pixel coordinates
        for i in range(n_depth):
            imsize = self.imsize[i]
            pixelsize = self.pixelsize[i]
            pixel_coords_i = (np.arange(imsize) - imsize*0.5 + 0.5) * pixelsize
            if grid_coords == []:
                # if there is no grid coordinates, just add the new one
                grid_coords.append(pixel_coords_i)
            else:
                # if there is already smaller grid, add the surrounding courser grid
                grid_coords.append(pixel_coords_i[(pixel_coords_i < grid_min) | (pixel_coords_i > grid_max)])
            if grid_min > pixel_coords_i.min():
                grid_min = pixel_coords_i.min()
            if grid_max < pixel_coords_i.max():
                grid_max = pixel_coords_i.max()
        # sort the pixel coordinates
        grid_coords = np.sort([x for gs in grid_coords for x in gs])
        return np.meshgrid(grid_coords, grid_coords)


class Grid2D:
    def __init__(self, imsize=None, pixelsize=None,
                 oversample=1):
        """the class to hand the gridding of the models
        """
        self.imsize = imsize
        self.pixelsize = pixelsize
        self._uvgrid = None
    @property
    def uvgrid(self):
        if self._uvgrid is not None:
            return self._uvgrid
        kmax = 0.5/(self.pixelsize*np.pi/180/3600) # arcsec to radian
        self._uvgrid = np.linspace(-kmax, kmax, self.imsize)
        return self._uvgrid
    def create_meshgrid(self):
        pixel_coords = (np.arange(self.imsize) - self.imsize*0.5 + 0.5) * self.pixelsize
        return np.meshgrid(pixel_coords, pixel_coords)

class Grid3D(Grid2D):
    def __init__(self):
        super().__init__(self,)
        self.chandata = None
        pass

class Fitter():
    def __init__(self, models, grid, data, name=None):
        self.models = models
        self.grid = grid
        self.data = data
        model_params = models.get_parameters()
        self.parameters = list(model_params.keys())
        self.initial_guess = list(model_params.values())
        self.n_params = len(self.parameters)
        self.limits = models.get_limits()
        self.name = name

class EmceeFitter(Fitter):
    def __init__(self, models, grid, data, datamodel=None, name=None,
                 nwalkers=None, outfile=None):
        super().__init__(models=models, grid=grid, data=data, name=name)
        self.models = models
        self.grid = grid
        self.data = data
        if datamodel is None:
            self.datamodel = self.data.copy()
        else:
            self.datamodel = datamodel
        self.outfile = outfile
        self.nwalkers = nwalkers
        if self.nwalkers is None:
            self.nwalkers = 6*self.n_params
        limits = models.get_limits()
        limits_ranges = np.array(list(limits.values()))
        limits_diff = np.diff(limits_ranges).flatten()
        # calculate the difference between the boundary and the initial guess
        initial_diff = np.min(np.abs(limits_ranges \
                                     - np.array(self.initial_guess)[:,None]), axis=1)
        self.pos = np.array(self.initial_guess) \
                + 1e-2*initial_diff[None,:]*np.random.randn(self.nwalkers, 
                                                            self.n_params)
        self.keys = list(models.get_parameters().keys())
    def run(self, nsteps=5000, ncores=1, progress=True):
        if self.outfile is not None:
            backend = emcee.backends.HDFBackend(self.outfile)
            backend.reset(self.nwalkers, self.n_params)
        else:
            backend = None
        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers, self.n_params, uv_log_probability, 
                args=(self.models, self.grid, self.data, self.datamodel), 
                backend=backend,
                pool=pool)
            sampler.run_mcmc(self.pos, nsteps, progress=progress)
        self.sampler = sampler

    def auto_run(self, max_nsteps=10000, check_nsteps=100, progress=True):
        check_nsteps = int(check_nsteps)
        index = 0
        old_tau = np.inf
        autocorr = np.empty(max_nsteps)
        for sample in self.sampler.sample(self.pos, iterations=max_nsteps, 
                                          progress=progress):
            if sampler.iteration % check_nsteps:
                continue
            # compute the autocorrelation
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            # check convergence
            converged = np.all(tau*check_nsteps < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

    def samples(self, discard=None, thin=None, flat=True):
        tau = self.sampler.get_autocorr_time()
        if discard is None:
            discard = int(2*np.max(tau))
        if thin is None:
            thin = int(0.5*np.min(tau))
        samples = self.sampler.get_chain(discard=discard, flat=flat, thin=thin)
        return samples 
    
    def plot(self, ax=None):
        samples = self.samples(flat=True)
        if ax is None:
            fig = plt.figure(figsize=(16, 4))
        for i in range(self.n_params):
            if ax is None:
                ax_i = plt.subplot2grid((self.n_params,2), (i,0), colspan=2)
            else:
                ax_i = ax[i]
            ax_i.plot(samples[:, i], "k", alpha=0.2)
            ax_i.set_xlim(0, len(samples))
            if i<self.n_params-1: ax_i.axes.get_xaxis().set_ticks([])
            ax_i.text(0, 0.8, self.keys[i], ha='left', transform=ax_i.transAxes, fontsize=10)
        ax_i.set_xlabel('steps')
        if ax is None:
            plt.show()
 
def uv_cost(theta, models, grid, uvdata, uvmodel, nufftobj):
    # return -1.0*uv_log_probability(*args)
    return np.sum(uv_vector_cost(theta, models, grid, uvdata, uvmodel, nufftobj))

def uv_vector_cost(theta, models, grid, uvdata, uvmodel, nufftobj):
    """
    Args:
        theta: the list of the free parameters
        model: Models
        uvdata: uvdata
        kwargs: the kwargs for models.create_model
    calculate the loglikelihood of the uv model and data
    """
    model_parameters = models.get_parameters()
    dict_theta = dict(zip(model_parameters.keys(), theta))
    models.update_parameters(dict_theta)
    #params_priors = models.get_priors(dict_theta)
    # if not np.isfinite(params_priors):
        # return -np.inf 
    # if np.any(np.isnan(model)):
        # return -np.inf
    if uvmodel is None:
        inplace = False
    else:
        inplace = True
    # fft_interpolate(uvdata, models, grid)
    if nufftobj is None:
        model_fft(uvmodel, models, grid, inplace=inplace)
    else:
        model_nufft(uvmodel, models, grid, nufftobj, inplace=inplace)

    mask = ~uvdata['mask']
    sigma2 = 1.0/(uvdata['weight'][mask]+1e-8)
    model_vector_cost = ((uvdata['real'][mask]-uvmodel['real'][mask])**2+(uvdata['imag'][mask]-uvmodel['imag'][mask])**2)/sigma2 #+ params_priors
    return model_vector_cost 

def uv_vector_cost_mpfit(theta, models, grid, uvdata, uvmodel=None, fjac=None):
    # mpfit needs to return status
    return 0, uv_vector_cost(theta, models, grid, uvdata, uvmodel)

class MinimizeFitter(Fitter):
    def __init__(self, models, grid, data, datamodel=None, name=None,
                 fftmode='fft'):
        """
        Args:
            pfunc: callable, function to calculate the log probability
        """
        super().__init__(models=models, grid=grid, data=data, name=name)
        limits = models.get_limits()
        limits_ranges = np.array(list(limits.values()))
        self.bounds = limits_ranges
        if datamodel is None:
            self.datamodel = self.data.copy()
        else:
            self.datamodel = datamodel
        self.fftmode = fftmode
        if self.fftmode == 'nufft':
            self.nufftobj = pynufft.NUFFT()
            om = np.array([data['u'], data['v']]).T
            om_scale = np.pi/np.max(om)
            Nd = (grid.imsize, grid.imsize)
            Kd = (512,512)
            Jd = (6,6)
            self.nufftobj = pynufft.NUFFT()
            self.nufftobj.plan(om*om_scale, Nd, Kd, Jd)
        elif self.fftmode == 'fft':
            self.nufftobj = None
        else:
            raise ValueError("unsupported fftmode")

    def run(self, initial_guess=None, debug=False):
        if initial_guess is None:
            initial_guess = self.initial_guess

        first_result = optimize.minimize(
                uv_cost, initial_guess, 
                args=(self.models,self.grid,self.data,self.datamodel,self.nufftobj), 
                bounds=self.bounds)
        # compute the chi2 of the first fit
        data_n = len(self.data['u'])
        first_cost = uv_cost(first_result.x, self.models, 
                                self.grid, self.data, self.datamodel, self.nufftobj)
        first_chi2 = first_cost/(data_n - len(initial_guess))
        if debug:
            print("first best_fit:", first_result.x)
            print(f"data_n: {data_n}; first_cost {first_cost};")
            print(f"Reduced chi2 of first minimize: {first_chi2}")
        # second time with least_squre fit, derive the errorbar
        res = optimize.least_squares(
                uv_vector_cost, first_result.x, jac='3-point', 
                bounds=self.bounds.T, method='trf', loss='soft_l1',
                args=(self.models, self.grid, self.data, self.datamodel,self.nufftobj))
        # adopted from scipy.curve_fit
        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)

        infodict = dict(nfev=res.nfev, fvec=res.fun)
        ier = res.status
        errmsg = res.message

        data_size = len(res.fun)
        cost = 2 * res.cost  # res.cost is half sum of squares!
        popt = res.x

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = linalg.svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)


        # fit_result = optimize.leastsq(
                # uv_vector_cost, first_result.x, full_output=True,
                # args=(self.models, self.grid, self.data))
        # popt, pcov, infodict, errmsg, ier = fit_result
        # data_size = len(infodict['fvec'])
        # cost = np.sum(infodict['fvec'] ** 2)
        
        # second fit with capfit
        # arameters_limits = self.models.get_limits()
        # bounds = np.array(list(parameters_limits.values())).T
        # res = capfit(uv_vector_cost, first_result.x, bounds=bounds,
                     # args=(self.models, self.grid, self.data))
        # print("second best_fit:", res.x)
        # print("second chi2", res.chi2, res.chi2/(data_n-len(initial_guess)))

        if data_size > len(popt):
            reduce_chi2 = cost / (data_size - len(popt))
            fit_cost = np.sum(uv_vector_cost(popt, self.models, self.grid, 
                              self.data, self.datamodel,self.nufftobj))
            reduce_chi2_mine = fit_cost/(data_n-len(initial_guess))
            pcov_scaled = pcov * reduce_chi2
            best_fit_error = np.sqrt(np.diag(pcov_scaled))
            best_fit_error2 = np.sqrt(np.diag(pcov))
            if debug:
                print(f"data_size: {data_size}; cost: {cost}")
                print(f"Reduced chi2 of leastsq: {reduce_chi2}")
                print(f"second cost: {fit_cost}")
                print(f"Reduced chi2 of first minimize (mine): {reduce_chi2_mine}")
                print(f"Error scaled: {best_fit_error}")
                print(f"Error pcov: {best_fit_error2}")
        else: 
            best_fit_error = None
        self.best_fit0 = dict(zip(self.parameters, first_result.x))
        self.best_fit = dict(zip(self.parameters, popt))
        self.best_fit_error = dict(zip(self.parameters, best_fit_error))

        # self.best_fit = dict(zip(self.parameters, res.x))
        # self.best_fit_error = dict(zip(self.parameters, res.x_err))

    def save_bestfit(self, outfile=None, overwrite=False):
        final_dict = {}
        final_dict['best_fit'] = self.best_fit
        final_dict['best_fit_error'] = self.best_fit_error
        try:
            final_dict['best_fit0'] = self.best_fit0
        except:
            pass
        if os.path.isfile(outfile):
            if not overwrite:
                print("Found existing file, set overwrite=True to overwrite!")
                return
        with open(outfile, 'w') as fp:
            json.dump(final_dict, fp)

    def mpfit_run(self, initial_guess=None, mpmaxstep=100):
        data_n = len(self.data['u'])
        if initial_guess is None:
            initial_guess = self.initial_guess
        parinfo = []

        parameters_values = self.models.get_parameters()
        parameters_limits = self.models.get_limits()
        for param in self.parameters:
            value = parameters_values[param]
            limits = parameters_limits[param]
            parinfo.append({'value':value, 'fixed':0.,
                            'limited':[1,1], 'limits':limits})
        m = mpfit(uv_vector_cost_mpfit, initial_guess, parinfo=parinfo,
                  functkw={'models':self.models, 'grid':self.grid, 
                           'uvdata':self.data})
        if (m.status <= 0):
            print("Error: {m.errmsg}")
        print(m.params)
        print(m)
        fit_cost = np.sum(uv_vector_cost(m.params, self.models, self.grid, 
                          self.data))
        reduce_chi2_mine = fit_cost/(data_n-len(initial_guess))
        print(f"data_n: {data_n}; fit_cost {fit_cost};")
        self.best_fit = m.params

    def capfit_run(self, initial_guess=None,):
        data_n = len(self.data['u'])
        if initial_guess is None:
            initial_guess = self.initial_guess
        parameters_values = self.models.get_parameters()
        parameters_limits = self.models.get_limits()
        bounds = np.array(list(parameters_limits.values())).T
        print("bounds", bounds)
        res = capfit(uv_vector_cost, initial_guess, bounds=bounds,
                     args=(self.models, self.grid, self.data, self.datamodel))
        self.best_fit = res.x
        print("best-fit:", res.x)
        print(res)
        
 
        
# fitting functions

def mcmc_log_probability(theta, models, var, data, data_err):
    """
    args: [var, data, data_err]
    """
    # var, data, data_err = args
    model_parameters = models.get_parameters()
    # for key, value in model_parameters.items():
        # print(key, value.value)
    dict_theta = dict(zip(model_parameters.keys(), theta))
    models.update_parameters(dict_theta)
    params_priors = models.get_priors(dict_theta)
    if not np.isfinite(params_priors):
        return -np.inf 
    model = models.create_model(var)
    for m in models.models:
        if isinstance(m, ScaleUncertainty):
            log_f = m.factor.value
            err2_addition = model**2*np.exp(2*log_f)
        else:
            err2_addition = 0
    sigma2 = data_err**2 + err2_addition
    model_log_likelihood = -0.5*np.sum((data-model)**2)/data_err**2 + np.log(data_err**2)
    #model_log_likelihood = -0.5*np.sum((data-model)**2/sigma2 + np.log(sigma2))
    return model_log_likelihood + params_priors

def uv_match(uv):
    """
    interpolate the uv data to a given grid
    """
    pass

def uv_log_probability(theta, models, grid, uvdata, uvmodel):
    """
    Args:
        theta: the list of the free parameters
        model: Models
        uvdata: uvdata
        kwargs: the kwargs for models.create_model
    calculate the loglikelihood of the uv model and data
    """
    model_parameters = models.get_parameters()
    dict_theta = dict(zip(model_parameters.keys(), theta))
    models.update_parameters(dict_theta)
    params_priors = models.get_priors(dict_theta)
    model = models.model_on_grid(grid)
    # print(dict_theta)
    # print(params_priors)
    if not np.isfinite(params_priors):
        return -np.inf 
    if np.any(np.isnan(model)):
        return -np.inf
    fft_interpolate(uvmodel, models, grid)
    # if 'msfile' in fit_kwargs.keys():
        # msfile = fit_kwargs['msfile']
        # print("Hack msfile {} and read it again".format(msfile))
        # hack_ms(uvmodel, fit_kwargs['msfile'])
        # uvmodel = read_uv_ALMA(fit_kwargs['msfile'])
    # for m in models.models:
    #     if isinstance(m, ScaleUncertainty):
    #         log_f = m.factor.value
    #         err2_addition = model**2*np.exp(2*log_f)
    #     else:
    #         err2_addition = 0
    mask = ~uvdata['mask']
    #sigma2 = (uvdata['sigma'][mask])**2 #+ err2_addition
    sigma2 = 1.0/(uvdata['weight'][mask]+1e-8)
    # model_log_likelihood = -0.5*np.sum((data-model)**2)/data_err**2 + np.log(data_err**2)
    model_log_likelihood = -0.5*np.sum(((uvdata['real'][mask]-uvmodel['real'][mask])**2+(uvdata['imag'][mask]-uvmodel['imag'][mask])**2)/sigma2)# + np.log(sigma2))
    global_likelihood = model_log_likelihood + params_priors
    if np.isnan(global_likelihood):
        return -np.inf
    return global_likelihood

def uv_log_probability_test(uvdata1, uvdata2):
    mask = ~uvdata1['mask']
    sigma2 = (uvdata1['sigma'][mask])**2 
    log_likelihood = -0.5*np.sum(((uvdata1['real'][mask]-uvdata2['real'][mask])**2+(uvdata1['imag'][mask]-uvdata2['imag'][mask])**2)/sigma2)# + np.log(sigma2))
    return log_likelihood

def model_to_uvdict(models, grid, ):
    image = models.model_on_grid(grid)
    # create uv coordinates from give maps
    kmax = 0.5/(grid.pixelsize*np.pi/180/3600) # arcsec to radian
    uvgrid_size = grid.uvgrid.size**2
    u, v = grid.create_meshgrid()
    uvdict = {'u':u.flatten(), 'v':v.flatten()}
    uvdict['real'] = np.zeros(uvgrid_size)
    uvdict['imag'] = np.zeros(uvgrid_size)
    if True:
        image_flipped = image[::-1,:] # match the pixel position to the sky coords
        imfft = fftshift(fft2(fftshift(image_flipped)))
        uvdict['real'] = imfft.real.flatten()
        uvdict['imag'] = imfft.imag.flatten()
    uvdict['weight'] = np.ones(uvgrid_size)
    return uvdict

def model_fft(uvdata, models, grid, inplace=True):
    imsize = grid.imsize
    model_image = models.model_on_grid(grid)
    # correct for primary beam attenuation
    try:
        PBfwhm = uvdata['PBfwhm']
    except:
        PBfwhm = None
    if PBfwhm is not None: 
        ymap, xmap = grid.create_meshgrid()
        PBsigma = uvdata['PBfwhm'] / (2.*np.sqrt(2.*np.log(2)))
        model_image *= np.exp(-(xmap**2./(2.*PBsigma**2.)) - (ymap**2./(2.*PBsigma**2.)))
 
    uvgrid = grid.uvgrid
    u, v = uvdata['u'], uvdata['v']
    image_flipped = model_image[::-1,:] # match the pixel position to the sky coords
    imfft = fftshift(fft2(fftshift(image_flipped)))
    real_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.real, kx=1, ky=1)
    imag_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.imag, kx=1, ky=1)
    fft_real, fft_imag = real_spline.ev(v, u), imag_spline.ev(v, u)

    if inplace:
        uvdata['real'] = fft_real 
        uvdata['imag'] = fft_imag
    else:
        uvdata_interp = uvdata.copy()
        uvdata_interp['real'] = fft_real
        uvdata_interp['imag'] = fft_imag
        return uvdata_interp

def model_nufft(uvdata, models, grid, nufftobj=None, inplace=True):
    model_image = models.model_on_grid(grid)
    # correct for primary beam attenuation
    try:
        PBfwhm = uvdata['PBfwhm']
    except:
        PBfwhm = None
    if PBfwhm is not None: 
        ymap, xmap = grid.create_meshgrid()
        PBsigma = uvdata['PBfwhm'] / (2.*np.sqrt(2.*np.log(2)))
        model_image *= np.exp(-(xmap**2./(2.*PBsigma**2.)) - (ymap**2./(2.*PBsigma**2.)))
    # define the spatial frequencies
    if nufftobj is None:
        om = np.array([uvdata['u'], uvdata['v']]).T
        # om_scale = np.pi/np.max(om)
        Nd = model_image.shape
        Kd = (512,512)
        Jd = (6,6)
        nufftobj = pynufft.NUFFT()
        nufftobj.plan(om, Nd, Kd, Jd)

    model_image_uv = nufftobj.forward(model_image)
    if inplace:
        uvdata['real'] = model_image_uv.real
        uvdata['imag'] = model_image_uv.imag
    else:
        uvdata_interp = uvdata.copy()
        uvdata_interp['real'] = model_image_uv.real
        uvdata_interp['imag'] = model_image_uv.imag
        return uvdata_interp

def fft_interpolate(uvdata, models, grid, phase_shift=[0,0], 
                    amplitude_scale=1, inplace=True,
                    debug=False, msfile=None, **kwargs):
    """
    To improve efficiency, the uvdata will be overwrite to avoid copy the file each time
    phase_shift: [arcsec, arcsec]
    pixel_scale: [arcsec, arcsec] in yaxis and xaxis
    """
    # the current implementation requires an equal size in x and y
    imsize = grid.imsize
    image_model = models.model_on_grid(grid)
    image_model = image_model[::-1,:] # match the pixel position to the sky coords  
    # correct for primary beam attenuation
    try:
        PBfwhm = uvdata['PBfwhm']
    except:
        PBfwhm = None
    if PBfwhm is not None: 
        ymap, xmap = grid.create_meshgrid()
        PBsigma = uvdata['PBfwhm'] / (2.*np.sqrt(2.*np.log(2)))
        image_model *= np.exp(-(xmap**2./(2.*PBsigma**2.)) - (ymap**2./(2.*PBsigma**2.)))
    
    imfft = fftshift(fft2(fftshift(image_model)))
    
    # Calculate the uv points we need, if we don't already have them
    uvgrid = grid.uvgrid
    # if uvgrid is None:
        # calculate the largest frequency scale from the smallest spatial scale
        # kmax = 0.5/(pixel_scale*np.pi/180/3600) # arcsec to radian
        # uvgrid = np.linspace(-kmax, kmax, nx)
    # interpolate the FFT'd image onto the data's uv points
    # RectBivariateSpline could be faster in the rectangular gridded data?
    u,v = uvdata['u'], uvdata['v'] 
    real_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.real, kx=1, ky=1)
    imag_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.imag, kx=1, ky=1)
    # RectBivariateSpline operate on [xgrid, ygrid] data
    # so we need to shift the position of u and v

    # uvdata_interp['real'] = real_interp
    if inplace:
        uvdata['real'] = real_spline.ev(v, u) 
        uvdata['imag'] = imag_spline.ev(v, u)
    else:
        uvdata_interp = uvdata.copy()
        uvdata_interp['real'] = real_spline.ev(v, u) 
        uvdata_interp['imag'] = imag_spline.ev(v, u)
        return uvdata_interp

    # amp_interp = np.sqrt(real_interp**2. + imag_interp**2.)
    # phase_interp = np.arctan2(imag_interp, real_interp)
    # # apply scaling, phase shifts; wrap phases to +/- pi.
    # amp_interp *= amplitude_scale
    # # phase shifts
    # # print(imfft.real.shape, u.shape, uvgrid.shape, real_interp.shape, phase_interp.shape)
    # phase_interp += 2.*np.pi*arcsec2rad*(phase_shift[0]*u + phase_shift[1]*v)
    # phase_interp = (phase_interp + np.pi) % (2*np.pi) - np.pi
    # real_interp = amp_interp * np.cos(phase_interp)
    # imag_interp = amp_interp * np.sin(phase_interp)
    
    # uvdata_interp = uvdata.copy()
    # uvdata_interp['real'] = real_interp
    # uvdata_interp['imag'] = imag_interp

    # return uvdata_interp

def fft_interpolate_cube(uvdata, data, xmap=None, ymap=None, pixel_scale=1,
                    uvgrid=None, scaleamp=1., phase_shift=[0.,0.], 
                    amplitude_scale=1, mode='loop'):
    """
    phase_shift: [arcsec, arcsec]
    pixel_scale: [arcsec, arcsec] in yaxis and xaxis
    """
    # the current implementation requires an equal size in x and y
    arcsec2rad = (np.pi/(180.*3600.))
    ndim = data.ndim
    if ndim == 3:
        nchan, ny, nx = data.shape
    elif ndim == 2:
        ny, nx = data.shape
        nchan = 0
    if not isinstance(pixel_scale, (int, float)):
        raise ValueError('Only support square pixels!')
    yp = (np.arange(ny) - ny*0.5 + 0.5) * pixel_scale
    xp = (np.arange(nx) - nx*0.5 + 0.5) * pixel_scale
    ymap, xmap = np.meshgrid(yp, xp)

    # match the pixel position to the sky coords  
    data = np.flip(data, axis=-2)
    # correct for primary beam attenuation
    PBfwhm = uvdata['PBfwhm']
    if PBfwhm is not None: 
        PBsigma = uvdata['PBfwhm'] / (2.*np.sqrt(2.*np.log(2)))
        pbcor = np.exp(-(xmap**2./(2.*PBsigma**2.)) 
                       - (ymap**2./(2.*PBsigma**2.)))
        if ndim == 3:
            data *= pbcor[None,:,:]
        else:
            data *= pbcor
        
    if ndim == 3:
        datafft = fft.fftshift(fft.fftn(fft.fftshift(data, axes=[1,2]), axes=[1,2]), 
                               axes=[1,2])
    elif ndim == 2:
        datafft = fft.fftshift(fft.fft2(fft.fftshift(data)))
    
    # Calculate the uv points we need, if we don't already have them
    if uvgrid is None:
        # calculate the largest frequency scale from the smallest spatial scale
        kmax = 0.5/(pixel_scale*np.pi/180/3600) # arcsec to radian
        uvgrid = np.linspace(-kmax, kmax, nx)
    u,v,w = uvdata['uvw'] 
    if ndim == 2:
        # interpolate the FFT'd image onto the data's uv points
        # RectBivariateSpline could be faster in the rectangular gridded data?
        real_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.real, kx=1, ky=1)
        imag_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.imag, kx=1, ky=1)
        # RectBivariateSpline operate on [xgrid, ygrid] data
        # so we need to shift the position of u and v
        real_interp = real_spline.ev(v, u) 
        imag_interp = imag_spline.ev(v, u) 
        # visi_interp = real_interp + imag_interp*1j
    elif (ndim == 3) & (mode=='loop'):
        start = time.time()
        data_interp = np.zeros((nchan, len(u)), dtype=complex)
        print("start loop interpolation")
        for i in range(nchan):
            real_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, datafft[i].real, kx=1, ky=1)
            imag_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, datafft[i].imag, kx=1, ky=1)
            real_interp = real_spline.ev(v, u) 
            imag_interp = imag_spline.ev(v, u) 
            data_interp[i] = real_interp + imag_interp*1j
        end = time.time()
        print(f"used {end-start}s")
    elif (ndim == 3) & (mode=='3d'):
        start = time.time()
        # 3D linear interpolation
        u,v,w = uvdata['uvw'] 
        specgrid = np.arange(0, nchan)
        specmeshgrid, uvmeshgrid, uvmeshgrid = np.meshgrid(specgrid, uvgrid, uvgrid, 
                                                         indexing='ij')
        print("interpolation")
        data_interp = interpolate.interpn(
                        (specgrid, uvgrid, uvgrid), datafft,
                        np.concatenate([np.tile(specgrid, [1,len(u),1]).T, 
                                        np.tile(np.array(np.array([v,u]).T), 
                                                [len(specgrid),1,1])], 
                                        axis=2), 
                        method='linear', bounds_error=False, fill_value=0).reshape(
                        (nchan, len(u)))
        # imag_interp = interpolate.interpn(
                        # (specgrid, ygrid, xgrid), data.imag,
                        # np.vstack([specgrid.ravel(), u.ravel(), v.ravel()]).T, 
                        # method='linear', bounds_error=False, fill_value=0).reshape(
                        # (len_wavelength, cube_ny, cube_nx))
        end = time.time()
        print(f"used {end-start}s")

    real_interp = data_interp.real
    imag_interp = data_interp.imag

    amp_interp = np.sqrt(real_interp**2. + imag_interp**2.)
    phase_interp = np.arctan2(imag_interp, real_interp)
    # apply scaling, phase shifts; wrap phases to +/- pi.
    amp_interp *= amplitude_scale
    # phase shifts
    print(amp_interp.shape, phase_interp.shape, u.shape)
    phase_interp += 2.*np.pi*arcsec2rad*(phase_shift[0]*u + phase_shift[1]*v)
    phase_interp = (phase_interp + np.pi) % (2*np.pi) - np.pi
    real_interp = amp_interp * np.cos(phase_interp)
    imag_interp = amp_interp * np.sin(phase_interp)
    
    uvdata_interp = uvdata.copy()
    uvdata_interp['real'] = real_interp
    uvdata_interp['imag'] = imag_interp

    return uvdata_interp


#######################################################
# Helper functions
#######################################################

def choose_not_none(inplist):
    item_is_not_none = [True if item is not None else False for item in inplist]
    if sum(item_is_not_none) > 0:
        item_choose = inplist[np.where(item_is_not_none)[0][0]]
    else:
        item_choose = None
    return item_choose

def read_uv_ALMA_legacy(vis, field=None, spw=None, chunk_size=None, average_time=None, 
                 average_channel=True, average_polarization=True, 
                 outfile=None):
    """read uv data from ALMA observations


    The weighting is crucial for uv data modelling, 
    For ALMA or VLA data, it is necessary to know that ALMA changes the 
    weighting scheme overtime and with/without channel averaging, check:
    https://casaguides.nrao.edu/index.php/DataWeightsAndCombination

    For simplicity, it is recommended to run statwt before concat or combine
    your science data
    """
    try:
        from casatools import table as _tbtool
        from casatools import msmetadata as _msmetadata
    except:
        print('Install "casatools" or execute the code within the casa shell!')
        return 0

    const_c = 299792458.0 # m/s
    tb = _tbtool()
    if spw is not None:
        if isinstance(spw, str): 
            spw_list = spw.split(',')
        else:
            spw_list = [str(spw),]
    else:
        tb.open(vis+'/DATA_DESCRIPTION')
        spw_ids = tb.getcol('SPECTRAL_WINDOW_ID')
        tb.close()
        spw_list = spw_ids.astype(str).tolist()
    
    # read the field info
    tb.open(vis+'/FIELD')
    fields_names = tb.getcol('NAME')
    fields_ids = tb.getcol('SOURCE_ID')
    fields_phase_center = np.squeeze(tb.getcol('PHASE_DIR'))
    # fields_phase_center = np.squeeze(tb.getcol('REFERENCE_DIR'))
    tb.close()
    if field is not None:
        # select the data only for field
        if isinstance(field, str):
            # change the string field to index
            field = field_ids[fields_names==field]
        if len(fields_ids)>1:
            phase_center = np.rad2deg(fields_phase_center[:,field])
        else:
            phase_center = np.rad2deg(fields_phase_center)
    else:
        phase_center = np.rad2deg(fields_phase_center)
    
    # Get the frequency of this channel
    tb.open(vis+'/SPECTRAL_WINDOW')
    chanfreqs = tb.getvarcol('CHAN_FREQ')
    if len(chanfreqs) > 0:
        if spw_list is not None:
            if len(spw_list) > 1:
                print("Warning: average the frequency of the all the spectral windows")
            spw_reffreqs = []
            for spw_single in spw_list:
                spw_reffreq = chanfreqs['r'+str(int(spw_single)+1)].flatten()
                if len(spw_reffreq) > 1:
                    print("Warning, average the frequency within the spectral window to get the reference frequency.")
                    spw_reffreqs.append(np.mean(spw_reffreq))
        reffreq = np.mean(spw_reffreqs)
    tb.close()
      
    # Get the primary beam size
    tb.open(vis+'/ANTENNA')
    diam = np.median(tb.getcol('DISH_DIAMETER')) # be careful if mixing arrays 
    tb.close()
    PBfwhm = 1.2*(const_c/reffreq)/diam * (3600*180/np.pi) # in arcsec
    # print('PB FWHM:', PBfwhm)
 
    # read some critical info from the whole ms, but avoid reading the whole data
    tb.open(vis)
    nrows = tb.nrows()
    colnames = tb.colnames()
    if 'CORRECTED_DATA' in colnames:
        data_colname = 'CORRECTED_DATA'
    else:
        data_colname = 'DATA'
    # define the selection array
    select = np.ones(nrows, dtype=bool)
    # select the rows of the targeted spw
    data_desc_ids = tb.getcol('DATA_DESC_ID')
    if spw_list is not None:
        unique_spws = np.unique(data_desc_ids)
        for spw_id in unique_spws:
            if str(spw_id) not in spw_list:
                continue
                # raise ValueError("No valid SPW can be found, please check the SPWs of the data!")
        spw_select = np.zeros(nrows, dtype=bool)
        for spw_id in spw_list:
            spw_select[data_desc_ids==int(spw_id)] = True
        select = select & spw_select
    # select the rows of the targeted field
    if field is not None:
        field_ids = tb.getcol('FIELD_ID')
        field_select = (field_ids == field)
        select = select & field_select
    rows_select = np.where(select==True)[0]
    data_desc_ids_select = data_desc_ids[rows_select]
    tb_select = tb.selectrows(rows_select);

    data_orig = tb_select.getcol(data_colname)
    data_mask = tb_select.getcol('FLAG')
    data_real = np.ma.array(data_orig.real, mask=data_mask)
    data_imag = np.ma.array(data_orig.imag, mask=data_mask)
    data_uvw = tb_select.getcol('UVW') #* reffreq/const_c # m to wavelength
    data_time = tb_select.getcol('TIME')
    data_weight = tb_select.getcol('WEIGHT')
    data_weight = np.expand_dims(data_weight, axis=1)

    if average_polarization:
        # average the two polarisations
        # uvdata['data'] = np.average(uvdata['data'], weights=(uvdata['sigma']**-2), 
                                    # axis=0)
        # data_real = np.ma.average(data_real, weights=data_weight, axis=0)
        # data_imag = np.ma.average(data_real, weights=data_weight, axis=0)
        # data_weight = np.expand_dims(data_weight, axis=1)
        # data_sigma = np.expand_dims(data_weight, axis=1)
        data_real = np.ma.sum(data_real * data_weight, axis=0) / ( 
                              np.sum(data_weight, axis=0))
        data_imag = np.ma.sum(data_imag * data_weight, axis=0) / ( 
                              np.sum(data_weight, axis=0))
        # uvdata['data'] = np.sum(uvdata['data'] * polarization_weights, axis=0) / ( 
                         # np.sum(polarization_weights, axis=0))
        # data_sigma = np.sum(data_sigma**-2, axis=0)**-0.5
        data_weight = np.sum(data_weight, axis=0)
    if average_channel:
        # channel weigting needs to be careful, as the uv coordinates changes with
        # frequency
        if "WEIGHT_SPECTRUM" in colnames:
            chan_weight = tb_select.getcol('WEIGHT_SPECTRUM')
        else:
            chan_weight = np.ones(data_real.shape[-2])
        # data_real = np.ma.average(data_real, axis=-2, weights=data_weight) 
        # data_imag = np.ma.average(data_imag, axis=-2, weights=data_weight) 
        data_real = np.ma.average(data_real * data_weight, axis=-2) / np.sum(data_weight, axis=-2)
        data_imag = np.ma.average(data_imag * data_weight, axis=-2) / np.sum(data_weight, axis=-2)
        # data_sigma = np.sum(data_sigma**-2, axis=-2)**-0.5 # just to drop the channel axis
        data_weight = np.sum(data_weight, axis=-2)
    # uvdata['weight'] = 1/uvdata['sigma']**2
    
    # # convert uvw coords from m to wavelengths
    # if spw_list is None:
        # spw_list = np.unique(data_desc_ids)
    # for spw_single in spw_list:
        # spw_reffreqs = chanfreqs['r'+str(int(spw_single)+1)].flatten()
        # if len(spw_reffreqs) > 0:
            # print("Warning, average the frequency to convert the uv coordinates")
        # spw_reffreq = np.mean(spw_reffreqs)
            # uvdata['uvw'][data_desc_ids_select==spw_single] *= spw_reffreq/const_c

    data_uvw *= reffreq/const_c

    uvdata = {'uvw': data_uvw, 
              'time': data_time,
              'antenna1': tb_select.getcol('ANTENNA1'),
              'antenna2': tb_select.getcol('ANTENNA2'), 
              'real': data_real.data,
              'imag': data_imag.data,
              'mask': data_real.mask,
              'weight': data_weight,
              'phase_center': phase_center,
              'reffreq': reffreq,
              'PBfwhm': PBfwhm}
    return uvdata

def hack_ms(uvdict, ms, mode='add', outdir='./', keep_original=True):
    """hack the uv data

    It can be used to replace or add data to existing measurements
    """
    from casatools import table as _tbtool
    tb = _tbtool()
    if keep_original:
        ms_hacked = os.path.join(outdir, os.path.basename(ms)+'.hacked')
        os.system(f'cp -r {ms} {ms_hacked}')
    else:
        ms_hacked = ms
    tb.open(ms_hacked, nomodify=False)
    data_orig = tb.getcol('DATA')
    data_orig_shape = data_orig.shape
    newdata = uvdict['real'] + uvdict['imag']*1j
    newdata_shape = newdata.shape
    if len(newdata_shape) == len(data_orig_shape):
        print('Same data shape, pass...')
        assert newdata_shape == data_orig_shape
    elif len(newdata_shape) == len(data_orig_shape) - 1:
        print('averaged polarisation data')
        assert newdata_shape == data_orig_shape[1:]
        newdata = np.tile(newdata, [data_orig_shape[0], 1,1])
    elif len(newdata_shape) == len(data_orig_shape) - 2:
        print('averaged polarisation and spectral data')
        newdata = np.tile(newdata, [data_orig_shape[0], data_orig_shape[1], 1])
    print('mode:', mode)
    if mode == 'overwrite':
        tb.putcol('DATA', newdata)
    elif mode == 'add':
        tb.putcol('DATA', newdata+data_orig)
    elif mode == 'minus':
        tb.putcol('DATA', data_orig-newdata)
    else:
        raise ValueError('Unsupported mode, should be: overwrite, add, minus')
    tb.flush()
    tb.close()

def hack_uvfits(uvfits, uvdict, outname=None, mode='overwrite', overwrite=False):
    with fits.open(uvfits) as hdu:
        fitsdata = hdu[0].data
    ndim = fitsdata['DATA'].ndim
    if ndim == 6:
        print(uvdict['real'].shape)
        print(fitsdata['DATA'][:,0,0,0,0,0].shape)
        if mode == 'overwrite':
            fitsdata['DATA'][:,0,0,0,0,0] = uvdict['real']
            fitsdata['DATA'][:,0,0,0,0,1] = uvdict['imag']
            # fitsdata['DATA'][:,0,0,0,0,2]
        elif mode == 'minus':
            fitsdata['DATA'][:,0,0,0,0,0] -= uvdict['real']
            fitsdata['DATA'][:,0,0,0,0,1] -= uvdict['imag']
        elif mode == 'add':
            fitsdata['DATA'][:,0,0,0,0,0] += uvdict['real']
            fitsdata['DATA'][:,0,0,0,0,1] += uvdict['imag']

    if ndim == 7:
        # multi-channel data
        nrow, nchan = (fitsdata['DATA'][:,0,0,0,:,0,0]).shape
        if nchan > 1:
            raise ValueError("Unsupported yet! Be patient or hack the source code!")
        if mode == 'overwrite':
            fitsdata['DATA'][:,0,0,0,0,0,0] = uvdict['real']
            fitsdata['DATA'][:,0,0,0,0,0,1] = uvdict['imag']
        elif mode == 'minus':
            fitsdata['DATA'][:,0,0,0,0,0,0] -= uvdict['real']
            fitsdata['DATA'][:,0,0,0,0,0,1] -= uvdict['imag']
        elif mode == 'add':
            fitsdata['DATA'][:,0,0,0,0,0,0] += uvdict['real']
            fitsdata['DATA'][:,0,0,0,0,0,1] += uvdict['imag']
        # old untested code
        # fitsdata['DATA'][:,0,0,0,:,0,0] = np.repeat(uvdict['real'] / nchan, chan).reshape(nrow, nchan)
        # fitsdata['DATA'][:,0,0,0,:,0,1] = np.repeat(uvdict['imag'] / nchan, chan).reshape(nrow, nchan)
        # data_weight = fitsdata['DATA'][:,0,0,0,:,0,2] = 1
    if outname is None:
        outname = uvfits[:-4]+'.hacked.fits'
    if os.path.isfile(outname):
        if not overwrite:
            print("Found existing file, skiping...")
            return
    with fits.open(uvfits) as hdu:
        hdu[0].data = fitsdata
        hdu.writeto(outname, overwrite=overwrite)

def compute_PBfwhm(reffreq, diameter):
    """compute the FWHM of the primary beam

    Args:
        reffreq: refference frequency [Hz]
        diameter: diameter of the antennas [m]

    Return:
        float in unit of arcsec
    """
    return 1.2*(const_c/reffreq)/diameter * (3600*180/np.pi) # in arcsec

def uvbin(uvdict, step=None, max_dist=None):
    if max_dist is not None:
        max_dist = np.min([np.max(uvdist), max_dist])
    else:
        max_dist = np.max(uvdist)
    nbins = np.ceil(max_dist/step).astype(int)
    uvdist_bins = np.zeros(nbins)
    real_bins = np.zeros((nbins, 2)) # data and std
    imag_bins = np.zeros((nbins, 2)) # data and std
    for i in range(nbins):
        bin_select = (uvdist > i*step) & (uvdist < (i+1)*step)
        uvdist_bins[i] = np.mean(uvdist[bin_select])
        if np.sum(bin_select) > 0:
            real_bins[i,0] = np.ma.average(real[bin_select], 
                                           weights=weight[bin_select])
            real_bins[i,1] = 1/np.sqrt(np.sum(weight[bin_select]))
            # real_bins[i,1] = np.std(real[bin_select])
            imag_bins[i,0] = np.ma.average(imag[bin_select], 
                                           weights=weight[bin_select])
            real_bins[i,1] = 1/np.sqrt(np.sum(weight[bin_select]))
    visi_bins = real_bins[:,0] + imag_bins[:,0]*1j
    amp_bins = np.ma.absolute(visi_bins)
    amp_bins_manual = np.ma.hypot(real_bins[:,0], imag_bins[:,0])
    amp_err_bins = np.sqrt((real_bins[:,0]*real_bins[:,1]/amp_bins)**2 + 
                           (imag_bins[:,0]*imag_bins[:,1]/amp_bins)**2)
    phase_bins_manual = np.rad2deg(np.ma.arctan2(imag_bins[:,0], real_bins[:,0]))
    phase_bins = np.rad2deg(np.ma.angle(visi_bins))
    phase_err_bins = np.sqrt((imag_bins[:,0]*real_bins[:,1]/amp_bins**2)**2 + 
                             (-real_bins[:,0]*imag_bins[:,1]/amp_bins**2)**2)
    return uvdist_bins, real_bins, imag_bins

def plot_uv_model(image_model, grid,):
    """
    """
    # the current implementation requires an equal size in x and y
    imsize = grid.imsize
    ymap, xmap = grid.create_meshgrid()
    
    image_model = image_model[::-1,:] # match the pixel position to the sky coords  
   
    imfft = fftshift(fft2(fftshift(image_model)))
    uvgrid = grid.uvgrid

    real_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.real, kx=1, ky=1)
    imag_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.imag, kx=1, ky=1)
    amp_interp = np.sqrt(real_interp**2. + imag_interp**2.)
    phase_interp = np.arctan2(imag_interp, real_interp)
    
    uvdata_model = {}
    uvdata_model['real'] = real_interp
    uvdata_model['imag'] = imag_interp

def plot_uv(uvdict, yaxis='amp,phase', step=None, max_dist=None, 
            plot=True, show_error=True):
    """
    Args:
        uvdict
        step: in unit same as u and v
    """
    u, v = uvdict['u'], uvdict['v']
    if 'mask' in uvdict.keys():
        mask = uvdict['mask']
    else:
        mask = None
    weight = uvdict['weight']
    # sigma = np.ma.array(uvdict['sigma'], mask=mask)
    real = np.ma.array(uvdict['real'], mask=mask)
    imag = np.ma.array(uvdict['imag'], mask=mask)
    uvdist = np.ma.array(np.hypot(u, v), mask=mask)
    try:
        uv_unit = uvdict['uv_unit']
    except:
        uv_unit = ''

    if step is None:
        # calculate the complext directly
        visi = real + imag*1j
        phase = np.ma.angle(visi) # radian
        phase_angle = np.rad2deg(np.ma.angle(visi)) # degree
        amp = np.ma.absolute(visi)
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].scatter(uvdist, amp) 
        ax[0].set_xlabel(f'uvdist ({uv_unit})')
        ax[0].set_ylabel('amplitude')
        ax[1].scatter(uvdist, phase_angle)
        ax[1].set_xlabel(f'uvdist ({uv_unit})')
        ax[1].set_ylabel('phase')
    else:
        # bin the data before calculate the amp and phase
        if max_dist is not None:
            max_dist = np.min([np.max(uvdist), max_dist])
        else:
            max_dist = np.max(uvdist)
        nbins = np.ceil(max_dist/step).astype(int)
        uvdist_bins = np.zeros(nbins)
        real_bins = np.zeros((nbins, 2)) # data and std
        imag_bins = np.zeros((nbins, 2)) # data and std
        for i in range(nbins):
            bin_select = (uvdist > i*step) & (uvdist < (i+1)*step)
            uvdist_bins[i] = np.mean(uvdist[bin_select])
            if np.sum(bin_select) > 0:
                if show_error:
                    real_bins[i,0] = np.ma.average(real[bin_select], 
                                                   weights=weight[bin_select])
                    real_bins[i,1] = 1/np.sqrt(np.sum(weight[bin_select]))
                    # real_bins[i,1] = np.std(real[bin_select])
                    imag_bins[i,0] = np.ma.average(imag[bin_select], 
                                                   weights=weight[bin_select])
                    real_bins[i,1] = 1/np.sqrt(np.sum(weight[bin_select]))
                else:
                    real_bins[i,0] = np.mean(real[bin_select])
                    imag_bins[i,0] = np.mean(imag[bin_select])
        visi_bins = real_bins[:,0] + imag_bins[:,0]*1j
        amp_bins = np.ma.absolute(visi_bins)
        amp_bins_manual = np.ma.hypot(real_bins[:,0], imag_bins[:,0])
        amp_err_bins = np.sqrt((real_bins[:,0]*real_bins[:,1]/amp_bins)**2 + 
                               (imag_bins[:,0]*imag_bins[:,1]/amp_bins)**2)
        phase_bins_manual = np.rad2deg(np.ma.arctan2(imag_bins[:,0], real_bins[:,0]))
        phase_bins = np.rad2deg(np.ma.angle(visi_bins))
        if show_error:
            phase_err_bins = np.sqrt((imag_bins[:,0]*real_bins[:,1]/amp_bins**2)**2 + 
                                 (-real_bins[:,0]*imag_bins[:,1]/amp_bins**2)**2)
        if plot:
            fig, ax = plt.subplots(1,2, figsize=(12,5))
            # print(uvdist_bins)
            # print(amp_bins)
            # print(phase_bins)
            if yaxis=='amp,phase':
                if show_error:
                    errobar_style = {'linestyle':'none', 'marker':'o', 'capsize':2}
                    ax[0].errorbar(uvdist_bins, amp_bins, yerr=amp_err_bins, **errobar_style)
                    ax[1].errorbar(uvdist_bins, phase_bins, yerr=phase_err_bins, **errobar_style)
                else:
                    ax[0].plot(uvdist_bins, amp_bins)
                    ax[1].plot(uvdist_bins, phase_bins)

                ax[0].set_xlabel('uvdist')
                ax[0].set_ylabel('amplitude')
                ax[1].set_xlabel('uvdist')
                ax[1].set_ylabel('phase')
            elif yaxis == 'real,imag':
                if show_error:
                    errobar_style = {'linestyle':'none', 'marker':'o', 'capsize':2}
                    ax[0].errorbar(uvdist_bins, real_bins[:,0], yerr=real_bins[:,1], **errobar_style)
                    ax[1].errorbar(uvdist_bins, imag_bins[:,0], yerr=imag_bins[:,0], **errobar_style)
                else:
                    ax[0].plot(uvdist_bins, real_bins)
                    ax[1].plot(uvdist_bins, imag_bins)
                ax[0].set_xlabel('uvdist')
                ax[0].set_ylabel('real')
                ax[1].set_xlabel('uvdist')
                ax[1].set_ylabel('imag')
        # if yaxis=='amp,phase':
                # return uvdist_bins, amp_bins, amp_err_bins, phase_bins, phase_err_bins
        # elif yaxis == 'real,imag':
            # return uvdist_bins, real_bins[:,0], real_bins[:,1], imag_bins[:,0], imag_bins[:,1]
    plt.show()

# def shift_phase(u, v, visi, dRA=0, dDec=0):
    # dRA *= 2. * np.pi
    # dDec *= 2. * np.pi

    # phi = u * dRA + v * dDec
    # visi_shifted = visi * (np.cos(phi) + 1j * np.sin(phi))
    # return visi

def plot_mcmc(samples, models, discard=1000, thin=5, plot_chain=True, plot_corner=True):
    if isinstance(samples, str):
        import emcee
        reader = emcee.backends.HDFBackend(samples)
        samples = reader.get_chain(discard=discard, thin=thin, flat=False)
    try:
        import corner
    except:
        raise ValueError('Install corner for the corner plot')
    labels = list(models.get_parameters().keys())
    n_labels = len(labels)
    initial_guess = list(models.get_parameters().values())
    # first plot the chain
    if plot_chain:
        fig, axes = plt.subplots(n_labels, figsize=(10, 1*n_labels), sharex=True)
        for i in range(n_labels):
            ax = axes[i]
            ax.plot(samples[:,:, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("step number");
    # cornor plot
    samples_shape = samples.shape
    samples_flat = samples.reshape((samples_shape[0]*samples_shape[1], samples_shape[2]))
    if plot_corner:
        fig = corner.corner(samples_flat, labels=labels, truths=initial_guess, 
                quantiles=[0.16, 0.5, 0.84])
    bestfits = np.percentile(samples_flat, 50, axis=0)
    bestfits_low = np.percentile(samples_flat, 16, axis=0)
    bestfits_up = np.percentile(samples_flat, 84, axis=0)
    for i in range(n_labels):
        print(f"{labels[i]} = {bestfits[i]:.2e} -{bestfits_low[i]:.2e}+{bestfits_up[i]:.2e}")


#######################################################
# Tests functions
#######################################################

def test_adptive_grid():
    sersic2d = Sersic2D(amplitude=1, reff=1, x0=0, y0=0,
                              n=1.5, ellip=0.6, theta=np.pi/4)
    model = Models([sersic2d])
    grid2d = Grid2D(imsize=10, pixelsize=1)
    x,y = grid2d.create_meshgrid()
    grid2d_adaptive = Grid2D_test(imsize=[5,20], pixelsize=[2,0.2])
    x_a,y_a = grid2d_adaptive.create_meshgrid()
    img = model.model_on_grid(grid2d)
    img_adaptive = model.model_on_grid(grid2d_adaptive)
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].pcolormesh(x, y, img,)
    ax[1].pcolormesh(x_a, y_a, img_adaptive,antialiased=True)

# test
"""some testing code
uv_image = read_uv_ALMA('HATLAS1000.ms', average_channel=True)
im2 = make_gaussian_image((200, 200), fwhm=(20,8), sigma=0, area=1, offset=[0,0])
uvimage_interp = fft_interpolate(uv_image, im2, pixel_scale=0.1)


cube2 = np.tile(im2, [128,1,1]) * np.exp(-np.abs((np.arange(128)-64))/10)[:,None,None]

uvcube_interp = fft_interpolate_cube(uv4, cube2, pixel_scale=0.2, phase_shift=[0,0])
hack_ms(uvcube_interp, 'HATLAS1000.ms.hacked2')
make_test_cube('HATLAS1000.ms.hacked2', outdir='cubes', imagename='test4')

"""

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import emcee
    from multiprocessing import Pool

    from uv_utils import Sersic2D, Models, Parameter, uv_log_probability, read_uv_ALMA, fft_interpolate, hack_ms, plot_uv, Grid2D
    # A simple test with sersic2d
    grid2d = Grid2D(imsize=512, pixelsize=0.04)
    
    # the true input
    sersic2d_truth = Sersic2D(amplitude=2.5e-7, reff=1.2, x0=-0.2, y0=0.25,
                              n=1.5, ellip=0.6, theta=np.pi/4)
    model_truth = Models([sersic2d_truth])
    img_truth = model_truth.model_on_grid(grid2d)
    # mocked observation with ALMA, here we use the antenna setups from
    # one existing observation
    mock_obs_basefile = 'J0920_alma_co32_collapsed_subtracted.ms'
    uv_image = read_uv_ALMA(mock_obs_basefile)
    uv_truth_mock = fft_interpolate(uv_image, img_truth, grid2d)
    os.system('rm -rf mock_image.ms')
    os.system(f'cp -r {mock_obs_basefile} mock_image.ms')
    hack_ms(uv_truth_mock, 'mock_image.ms', mode='add')
    vis = 'mock_image.ms'
    tclean(vis, imagename='images/mock_clean_v1', specmode='mfs', imsize=1024, cell='0.08arcsec', niter=1000, usemask='auto-multithresh', threshold='10uJy')

    if True: # plot the truth image
        fig, ax = plt.subplots()
        log_img_truth = np.log10(img_truth)
        im = ax.imshow(log_img_truth, origin='lower')
        cbar = fig.colorbar(im, ax=ax)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


    # start the mcmc run here
    # setup the model with slightly wrong guests
    grid2d = Grid2D(imsize=512, pixelsize=0.04)
    sersic2d = Sersic2D(amplitude=Parameter(1e-5, limits=[0, 1e-2]), 
                        reff=Parameter(1, limits=[0, 10]), 
                        x0=Parameter(0, limits=[-10,10]), 
                        y0=Parameter(0, limits=[-10,10]), 
                        n=Parameter(2, limits=[0.2,6]), 
                        ellip=Parameter(0.5, limits=[0,1]), 
                        theta=Parameter(0, limits=[-0.5*np.pi, 0.5*np.pi]))
    models = Models([sersic2d,])
    parameters = models.get_parameters()
    print(parameters)
    limits = models.get_limits()
    limits_ranges = np.array(list(limits.values()))
    limits_diff = np.diff(limits_ranges).flatten()
    # # check whether there is magnitude difference
    initial_guess = np.array(list(models.get_parameters().values()))


    # setup the random walkers
    walker_multiplicity = 6*len(parameters)
    # pos = np.array(list(models.get_parameters().values())) + 1e-3 * np.random.randn(
            # walker_multiplicity*len(parameters), len(parameters))
    pos = initial_guess + 1e-2*limits_diff[None,:]*np.random.rand(walker_multiplicity, len(parameters))
    nwalkers, ndim = pos.shape
    nsteps = 10000
    uvdata = read_uv_ALMA('mock_image.ms')

    
    ##################
    # for single core
    # sampler = emcee.EnsembleSampler(
            # nwalkers, ndim, uv_log_probability, args=(models, uvdata, grid2d)
    # )
    # sampler.run_mcmc(pos, nsteps, progress=True)

    ##################
    # fit with multiprocessing
    filename = f"test_mcmcfit_{nsteps}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with Pool(18) as pool:
        sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, uv_log_probability, 
                        args=(models, uvdata, grid2d),
                        pool=pool, backend=backend)
        sampler.run_mcmc(pos, nsteps, progress=True)

    labels = list(parameters.keys())
    n_labels = len(labels)
    fig, axes = plt.subplots(n_labels, figsize=(10, 1*n_labels), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.show()

