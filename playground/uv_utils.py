#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""A minimalist tool to handle visibility

Author: Jianhang Chen, cjhastro@gmail.com

Requirement:

History:

"""

__version__ = '0.0.1'

import time
import logging
import numpy as np
import os

import matplotlib.pyplot as plt
# import astropy.units as u
# import astropy.constants as const

from scipy.fft import fft2, fftshift
from numpy import fft
from scipy import interpolate, special

try:
    from casatools import table as _tbtool
    from casatools import msmetadata as _msmetadata
except:
    print('Install casatools to import ALMA data.')

const_c = 299792458.0 # m/s

#######################################################
# visibility data handling
#######################################################

class UVdata:
    def __init__(self, u=None, v=None, w=None, real=None, imag=None,
                 weight=None, sigma=None, mask=None, reffreq=None):
        self._u = u
        self._v = v
        self._w = w
        self._real = real
        self._imag = imag
        self.mask = mask
        self._weight = weight
        self._sigma = sigma
        self.reffreq = reffreq
    def __add__(self, uvdata):
        if self.reffreq != uvdata.reffreq:
            logging.warning("unmatched reference frequency! Averaging the refrequencies!")
            reffreq = np.mean([self.reffreq, uvdata.reffreq])
        else:
            reffreq = self.reffreq
        return UVdata(u=np.vstack([self.u, uvdata.u]), v=np.vstack([self.v, uvdata.v]),
                      real=np.vstack([self.real, uvdata.real]), 
                      imag=np.vstack([self.imag, uvdata.imag]),
                      weight=np.vstack([self.weight, uvdata.weight]), 
                      mask=np.vstack([self.mask, uvdata.mask]),
                      reffreq=reffreq)
    @property
    def u(self):
        return self._u
    @u.setter
    def u(self, val):
        self._u = val
    @property
    def v(self):
        return self._v
    @v.setter
    def v(self, val):
        self._v = val
    @property
    def visi(self):
        return self._real + 1j*self._imag
    @visi.setter
    def visi(self, val):
        self._real = val.real
        self._imag = val.imag
    @property
    def real(self):
        return self._real
    @real.setter
    def real(self, val):
        self._real = val
    @property
    def imag(self):
        return self._imag
    @imag.setter
    def imag(self, val):
        self._imag = val
    @property
    def uvdist(self, unit=None):
        if unit is None:
            return np.hypot(self.u, self.v)
        if unit == 'm':
            return np.hypot(self.u*reffreq/const_c, 
                            self.v*reffreq/const_c)
    @property
    def visi(self):
        return self.real + self.imag*1j
    @property
    def angle(self):
        return np.rad2deg(np.angle(self.visi))
    def deprojection(self):
        pass
    def shift_phase(self, delta_ra, delta_dec):
        """shift the phase of the source
        delta_ra: positive value shifts towards the east
        delta_dec: positive value shifts towards the north
        """
        if delta_ra != 0 and delta_dec!=0:
            phi = self.u * 2*np.pi * delta_ra + self.v * n*np.pi * delta_dec
            visi = (self.real + 1j*self.imag) * (np.cos(phi) * 1j*np.sin(phi))
        self.real = visi.real
        self.imag = visi.imag
    def rotate(self, theta):
        """rotate the uv coordinate counter-clockwise with an radian: theta
        """
        if theta != 0:
            u_new = self.u * np.cos(theta) - self.v * np.sin(theta)
            v_new = self.u * np.sin(theta) + self.v * np.cos(theta)
        self.u = u_new
        self.v = v_new
    def deproject(self, inc):
        self.u = self.u * np.cos(inc)

    # all the read and write functions
    def from_dictionary(self, uvdata):
        # load the uvdata from the dictionary
        # check the dictionary have all the required info
        required_colnames = ['uvw', 'real', 'imag', 'sigma', 'mask']
        dict_names = list(uvdata.keys())
        for colname in required_colnames:
            if colname not in dict_names:
                raise ValueError(f"{colname} is not found in the dictionary")
        self.u, self.v, self.w = uvdata['uvw']
        self.real = uvdata['real']
        self.imag = uvdata['imag']
        self._weight = uvdata['weight']
        # self._sigma = uvdata['sigma']
        self.mask = uvdata['mask']
        self.reffreq = uvdata['reffreq']
    def to_dictionary(self):
        uvdata = {}
        uvdata['uvw'] = np.vstack([self.u, self.v, self.w])
        uvdata['real'] = self.real
        uvdata['imag'] = self.imag
        uvdata['weight'] = self._weight
        uvdata['sigma'] = self._sigma
        uvdata['mask'] = self.mask
        uvdata['reffreq'] = self.reffreq
        return uvdata
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,7))
        ax.scatter(self.u, self.v)
    def write(self, filename):
        # write uvdata into file
        pass
    def read(self, filename):
        pass

def read_uv_ALMA(vis, field=None, spw=None, chunk_size=None, average_time=None, 
                 average_channel=True, average_polarization=True):
    """read uv data from ALMA observations


    The weighting is crucial for uv data modelling, 
    For ALMA or VLA data, it is necessary to know that ALMA changes the 
    weighting scheme overtime and with/without channel averaging, check:
    https://casaguides.nrao.edu/index.php/DataWeightsAndCombination

    For simplicity, it is recommended to run statwt before concat or combine
    your science data
    """
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


def plot_uv(uvdata, yaxis='amp,phase', step=None, max_dist=None):
    """
    Args:
        step: in unit of 'm'
    """
    u, v, w = uvdata['uvw']
    mask = uvdata['mask']
    weight = uvdata['weight']
    # sigma = np.ma.array(uvdata['sigma'], mask=mask)
    real = np.ma.array(uvdata['real'], mask=mask)
    imag = np.ma.array(uvdata['imag'], mask=mask)
    uvdist = np.ma.array(np.hypot(u, v), mask=mask)

    if step is None:
        # calculate the complext directly
        visi = real + imag*1j
        phase = np.ma.angle(visi) # radian
        phase_angle = np.rad2deg(np.ma.angle(visi)) # degree
        amp = np.ma.absolute(visi)
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].scatter(uvdist, amp) 
        ax[0].set_xlabel('uvdist [m]')
        ax[0].set_ylabel('amplitude')
        ax[1].scatter(uvdist, phase_angle)
        ax[1].set_xlabel('uvdist [m]')
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
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        # print(uvdist_bins)
        # print(amp_bins)
        # print(phase_bins)
        if yaxis=='amp,phase':
            errobar_style = {'linestyle':'none', 'marker':'o', 'capsize':2}
            ax[0].errorbar(uvdist_bins, amp_bins, yerr=amp_err_bins, **errobar_style)
            ax[1].errorbar(uvdist_bins, phase_bins, yerr=phase_err_bins, **errobar_style)
            ax[0].set_xlabel('uvdist [m]')
            ax[0].set_ylabel('amplitude')
            ax[1].set_xlabel('uvdist [m]')
            ax[1].set_ylabel('phase')
            return uvdist_bins, amp_bins, amp_err_bins, phase_bins, phase_err_bins
        elif yaxis == 'real,imag':
            errobar_style = {'linestyle':'none', 'marker':'o', 'capsize':2}
            ax[0].errorbar(uvdist_bins, real_bins[:,0], yerr=real_bins[:,1], **errobar_style)
            ax[1].errorbar(uvdist_bins, imag_bins[:,0], yerr=imag_bins[:,0], **errobar_style)
            ax[0].set_xlabel('uvdist [m]')
            ax[0].set_ylabel('real')
            ax[1].set_xlabel('uvdist [m]')
            ax[1].set_ylabel('imag')
            return uvdist_bins, real_bins[:,0], real_bins[:,1], imag_bins[:,0], imag_bins[:,1]

# def shift_phase(u, v, visi, dRA=0, dDec=0):
    # dRA *= 2. * np.pi
    # dDec *= 2. * np.pi

    # phi = u * dRA + v * dDec
    # visi_shifted = visi * (np.cos(phi) + 1j * np.sin(phi))
    # return visi


def hack_ms(uvdata, ms, mode='add'):
    """hack the uv data

    It can be used to replace or add data to existing measurements
    """
    tb = _tbtool()
    tb.open(ms, nomodify=False)
    data_orig = tb.getcol('DATA')
    data_orig_shape = data_orig.shape
    newdata = uvdata['real'] + uvdata['imag']*1j
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

def make_test_cube(vis, imagename='test1', outdir='cubes'):
    tclean(vis=vis, imagename=os.path.join(outdir, imagename), 
           spw='0', imsize=200, cell='0.2arcsec', 
           specmode='cube', nchan=110, start=5,
           niter=2000, threshold='0mJy',
           usemask='auto-multithresh', )


#######################################################
# Basic fitting wrappers
#######################################################

class Parameter():
    def __init__(self, value, limits=None, name=''):
        if isinstance(value, Parameter):
            self.value = value.value
            self.limits = value.limits
            self.name = value.name
        else:
            self.value = value
            self.limits = limits
            self.name = name
    @property
    def is_fixed(self):
        if self.limits is None:
            return True
        else:
            return False
    def get_log_prior(self, value):
        if self.limits is None:
            return 0
        limits = self.limits
        if limits[0] < value < limits[1]:
            return 0
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
    def evaluate(self, x=None, y=None, **kwargs):
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
    def get_parameters(self):
        params_list = {}
        for model in self.models:
            for key,value in model.__dict__.items():
                if not isinstance(value, Parameter):
                    continue
                if not value.is_fixed:
                    params_list[model.name+'.'+key] = value.value
        return params_list
    def update_parameters(self, params_dict):
        for model in self.models:
            for key in params_dict.keys():
                model_name, model_param_name = key.split('.')
                if model_param_name in model.__dict__.keys():
                    model.__dict__[model_param_name].value = params_dict[key]
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
    def create_model(self, kwargs):
        # parameter_names = self.get_parameters()
        # self.update_parameters(dict(zip(parameter_names, theta)))
        model_value = 0.
        for model in self.models:
            model_value += model.evaluate(**kwargs)
        return model_value

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

def uv_log_probability(theta, models, uvdata, models_kwargs, fit_kwargs):
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
    # print(dict_theta)
    # print(params_priors)
    if not np.isfinite(params_priors):
        return -np.inf 
    model = models.create_model(models_kwargs)
    uvmodel = fft_interpolate(uvdata, model, **fit_kwargs)
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
    sigma2 = 1.0/uvdata['weight'][mask]
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

def fft_interpolate(uvdata, image, xmap=None, ymap=None, pixel_scale=1,
                    uvgrid=None, scaleamp=1., phase_shift=[0.,0.], 
                    amplitude_scale=1, debug=False, msfile=None, **kwargs):
    """
    phase_shift: [arcsec, arcsec]
    pixel_scale: [arcsec, arcsec] in yaxis and xaxis
    """
    # the current implementation requires an equal size in x and y
    arcsec2rad = (np.pi/(180.*3600.))
    ny, nx = image.shape
    if not isinstance(pixel_scale, (int, float)):
        raise ValueError('Only support square pixels!')
    if (xmap is None) or (ymap is None):
        yp = (np.arange(ny) - ny*0.5 + 0.5) * pixel_scale
        xp = (np.arange(nx) - nx*0.5 + 0.5) * pixel_scale
        ymap, xmap = np.meshgrid(yp, xp) # coordinate of at the center of pixels

    # correct for primary beam attenuation
    image = image[::-1,:] # match the pixel position to the sky coords  
    PBfwhm = uvdata['PBfwhm']
    if PBfwhm is not None: 
        PBsigma = uvdata['PBfwhm'] / (2.*np.sqrt(2.*np.log(2)))
        image *= np.exp(-(xmap**2./(2.*PBsigma**2.)) - (ymap**2./(2.*PBsigma**2.)))
    
    imfft = fftshift(fft2(fftshift(image)))
    
    # Calculate the uv points we need, if we don't already have them
    if uvgrid is None:
        # calculate the largest frequency scale from the smallest spatial scale
        kmax = 0.5/(pixel_scale*np.pi/180/3600) # arcsec to radian
        uvgrid = np.linspace(-kmax, kmax, nx)
    # interpolate the FFT'd image onto the data's uv points
    # RectBivariateSpline could be faster in the rectangular gridded data?
    u,v,w = uvdata['uvw'] 
    real_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.real, kx=1, ky=1)
    imag_spline = interpolate.RectBivariateSpline(uvgrid, uvgrid, imfft.imag, kx=1, ky=1)
    # RectBivariateSpline operate on [xgrid, ygrid] data
    # so we need to shift the position of u and v
    real_interp = real_spline.ev(v, u) 
    imag_interp = imag_spline.ev(v, u) 
    # visi_interp = real_interp + imag_interp*1j

    amp_interp = np.sqrt(real_interp**2. + imag_interp**2.)
    phase_interp = np.arctan2(imag_interp, real_interp)
    # apply scaling, phase shifts; wrap phases to +/- pi.
    amp_interp *= amplitude_scale
    # phase shifts
    # print(imfft.real.shape, u.shape, uvgrid.shape, real_interp.shape, phase_interp.shape)
    phase_interp += 2.*np.pi*arcsec2rad*(phase_shift[0]*u + phase_shift[1]*v)
    phase_interp = (phase_interp + np.pi) % (2*np.pi) - np.pi
    real_interp = amp_interp * np.cos(phase_interp)
    imag_interp = amp_interp * np.sin(phase_interp)
    
    uvdata_interp = uvdata.copy()
    uvdata_interp['real'] = real_interp
    uvdata_interp['imag'] = imag_interp

    if debug:
        print('pixel_scale', pixel_scale)
        if msfile is None:
            print('Provide msfile to test the output')
        else:
            hack_ms(uvdata_interp, msfile)
    return uvdata_interp

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
    x, y = np.meshgrid(np.arange(1024), np.arange(1024))
    x = x - 512
    y = y - 512

    sersic2d = Sersic2D(amplitude=Parameter(1e-3, limits=[0, 1e-2]), 
                        reff=Parameter(1, limits=[0, 10]), 
                        x0=Parameter(0, limits=[-10,10]), 
                        y0=Parameter(0, limits=[-10,10]), 
                        n=Parameter(2, limits=[0,6]), 
                        ellip=Parameter(0.8, limits=[0,1]), 
                        theta=Parameter(0, limits=[0, np.pi]))
    img = sersic2d.evaluate(x, y)
    fig, ax = plt.subplots()
    log_img = np.log10(img)
    im = ax.imshow(log_img, origin='lower', interpolation='nearest',
                )#vmin=-1, vmax=2)
    cbar = fig.colorbar(im, ax=ax)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    models = Models([sersic2d,])
    # models = Models([line_model,])
    parameters = models.get_parameters()

    import emcee
    pos = np.array(list(models.get_parameters().values())) + 1e-4 * np.random.randn(32, 7)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
            nwalkers, ndim, uv_log_probability, args=(models, uvdata, {'x':x, 'y':y})
    )
    nsteps = 5000
    sampler.run_mcmc(pos, nsteps, progress=True)

    # fit with multiprocessing
    from multiprocessing import Pool
    pos = np.array(list(models.get_parameters().values())) + 1e-4 * np.random.randn(32, 7)
    nwalkers, ndim = pos.shape
    nsteps = 1000
    with Pool(6) as pool:
        sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, uv_log_probability, 
                        args=(models, uvdata, {'x':x, 'y':y}),
                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)

    labels = list(paramters.keys())
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

