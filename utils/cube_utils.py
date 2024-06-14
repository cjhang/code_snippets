#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""A minimalist utilities dealing with fits datacube

Author: Jianhang Chen, cjhastro@gmail.com

Example:
        from cube_utils import read_ALMA_cube
        dc = read_ALMA_cube('test_data/alma_datacube_AR2623.image.fits')
        dc.convert_chandata('km/s', reffreq=98.5*u.GHz)
        dc.collapse_cube('10~20')

Requirement:
    numpy
    matplotlib
    astropy >= 5.0

Todo:
    * For module TODOs

History:
    2022-02-16: first release to handle the datacube from ALMA


"""

__version__ = '0.0.1'


import os
import sys
import re
import argparse
import textwrap
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt

import astropy
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as cosm
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy import coordinates
from astropy.stats import sigma_clipped_stats

# Filtering warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

# ndimage
from scipy import ndimage
from scipy import optimize

from photutils.aperture import aperture_photometry, EllipticalAperture, RectangularAperture, SkyEllipticalAperture, PixelAperture, SkyAperture
from photutils.detection import find_peaks
from astropy.modeling import models, fitting

# try to import other utils
from spec_utils import convert_spec, Spectrum

##############################
######### Cube ###############
##############################

class BaseCube(object):
    """basic cube class
    """
    def __init__(self):
        pass

class Cube(BaseCube):
    """The base data strcuture to handle 3D astronomy data
    """
    def __init__(self, data=None, mask=None, name=None,
                 header=None, wcs=None, beam=None, unit=None,
                 chandata=None,  reference=None):
        """initialize the cube

        Args:
            data: the 3D data, numpy ndaddray 
            header: the header can be identified by WCS
            wcs: it can be an alternative to the header
            specdata: the data in the third dimension, it can be velocity, frequency, wavelength. 
                It should have units
            beams: the beam shapes the units of [arcsec, arcsec, deg]. 
                It can be one dimension or two dimension (for each channel)
        """
        self.data = data
        self.mask = mask
        self.name = name
        self.reference = reference
        # hide the orinal header and wcs, due to their interchangeability
        self._header = header
        self._wcs = wcs
        # keep the original beam as sometimes it can be access from header
        self._beam = beam
        # other optional info 
        self._chandata = chandata
        self._unit = unit
    def __getitem__(self, i):
            # return self.data[i]
            return self.subcube(i)
    @property
    def info(self):
        # the shortcut to print basic info
        print(f"The shape of the data: {self.data.shape}")
        print(f"The `beam` is: {self.beam} in [arcsec, arcsec, deg]")
        print(f"The `pixel_sizes` is: {self.pixel_sizes} in [arcsec, arcsec]")
        print(self.wcs)
    @property
    def header(self):
        if self._header is not None:
            return self._header
        elif self.wcs is not None:
            return self.wcs.to_header()
        else:
            return None
    @header.setter
    def header(self, header):
        self._header = header
    @property
    def wcs(self):
        if self._wcs is not None:
            fullwcs = self._wcs
        elif self._header is not None:
            fullwcs = WCS(self._header)
        else:
            fullwcs = None
        if (fullwcs is not None) and (fullwcs.naxis == 4):
            # drop the stokes axis
            # return fullwcs.dropaxis(0)
            # return fullwcs.sub(['longitude','latitude','spectral'])
            # return fullwcs.sub(['longitude','latitude','cubeface'])
            return fullwcs.sub([1,2,3])
        else:
            raise ValueError('Unsupprted wcs!')
        return fullwcs
    @wcs.setter
    def wcs(self, wcs):
        self._wcs = wcs
    @property
    def unit(self):
        if self._unit is not None:
            return self.unit
        elif self.header is not None:
            try:
                return u.Unit(self._header['BUNIT'])
            except:
                pass
        return u.Unit('')
    @unit.setter
    def unit(self, unit):
        self._unit = u.Unit(unit)
        self._header.update({'BUNIT': self._unit.to_string()})
    @property
    def cube(self):
        return self.data*self.unit
    @property
    def chandata(self):
        if self._chandata is not None:
            return self._chandata
        else:
            # read the chandata from the header
            # the fits standard start with index 1
            if 'CD3_3' in self.header.keys():
                cdelt3 = self.header['CD3_3']
            else: 
                cdelt3 = self.header['CDELT3']
            # because wcs.slice with change the reference channel, the generated ndarray should 
            # start with 1 
            chandata = (self.header['CRVAL3'] + (np.arange(1, self.header['NAXIS3']+1)-self.header['CRPIX3']) * cdelt3)
            if 'CUNIT3' in self.header.keys():
                chandata = chandata*u.Unit(self.header['CUNIT3'])
            self._chandata = chandata
            return chandata
    @chandata.setter
    def chandata(self, chandata):
        if not isinstance(chandata, u.Quantity):
            raise ValueError("chandata needs to be Quantify")
        unit_out = chandata.unit
        if unit_out.is_equivalent('m/s'):
            ctype3 = 'VELOCITY'
        elif unit_out.is_equivalent('m'):
            ctype3 = 'WAVELENGTH'
        elif unit_out.is_equivalent('Hz'):
            ctype3 = 'FREQUENCY'
        else:
            ctype3 = 'Unknown'
        cunit3 = unit_out.to_string()
        # update the header
        self._header.update(
                {'CUNIT3':cunit3, 'CTYPE3':ctype3,
                 'CRVAL3':chandata[1].value, 'CRPIX3':1,
                 'CDELT3':(chandata[1]-chandata[0]).value})
        self.wcs = WCS(self._header)
        self._chandata = chandata
    @property
    def beam(self):
        # return beam shape in [bmaj, bmin, PA] in [arcsec, arcsec, degree]
        if self._beam is not None:
            cube_beam = self._beam
            if np.ndim(cube_beam) == 2:
                return np.median(cube_beam, axis=0)
            elif np.ndim(cube_beam) == 1:
                return cube_beam
        if self.header is not None:
            try:
                header = self.header
                header_beam = np.array([header['BMAJ'], header['BMIN'], header['BPA']]).T
                cube_beam = header_beam * np.array([3600.,3600.,1]) # convert deg to arcsec
                return cube_beam
            except: pass
        return None
    @beam.setter
    def beam(self, beam):
        """the beams could 2D array or 1D array like [bmaj, bmin, PA]
        the units should be [arcsec, arcsec, degree]
        """
        self._beam = beam
        header_beams = self.beam * np.array(1/3600., 1/3600, 1) # convert arcsec to deg
        self._header.update({'BMAJ':header_beams[0], 'BMIN':header_beams[1], 
                             'PA':header_beams[2]})
    @property
    def pixel_sizes(self):
        if self.header is not None:
            # Return the pixel size encoded in the header
            # In casa, it is in the units of deg, thus the returned value is pixel to deg
            pixel2arcsec_ra = (abs(self.header['CDELT1'])*u.Unit(self.header['CUNIT1'])).to(u.arcsec).value
            pixel2arcsec_dec = (abs(self.header['CDELT2'])*u.Unit(self.header['CUNIT2'])).to(u.arcsec).value
        else:
            pixel2arcsec_ra, pixel2arcsec_dec = 1, 1
        return np.array([pixel2arcsec_ra, pixel2arcsec_dec])
    @property
    def shape(self):
        return self.data.shape
    @property
    def imagesize(self):
        return self.shape[-2:]
    @property
    def ndim(self):
        try: return self.data.ndim
        except: return None
    @property
    def nchan(self):
        try: return self.data.shape[-3]
        except: return None
    @property
    def dchan(self):
        try: return calculate_diff(self.chandata)
        except: return None
    def get_wavelength(self, unit='mm', reference=None):
        return convert_spec(self.chandata, unit)
    def get_frequency(self, unit='GHz', reference=None):
        return convert_spec(self.chandata, unit)
    def get_velocity(self, unit='km/s', reference=None):
        return convert_spec(self.chandata, unit, reference=reference)
    def pixelcoods2sky(self, pixel_coords):
        #covert from pixel to skycoords
        pixel_coords = np.array(pixel_coords).T
        return pixel_to_skycoord(*pixel_coords, self.wcs)
    def skycoords2pixels(self, skycoords):
        #covert from pixel to skycoords
        if skycoords.size == 1:
            return np.array(skycoord_to_pixel(skycoords, self.wcs))
        elif skycoords.size > 1:
            return np.array(list(zip(*skycoord_to_pixel(skycoords, self.wcs))))
        else:
            raise ValueError('Unsupported sky coordinates!')
    def convert_chandata(self, unit_out, reference=None):
        """convert the units of the specdata 
        """
        self.reference = reference
        chandata = convert_spec(self.chandata, unit_out, reference=reference)
        self.chandata = chandata
    def extract_spectrum(self, pixel_coord=None, sky_coord=None, pixel_aperture=None, 
                         sky_aperture=None, plot=False, ax=None, **kwargs):
        """extract 1D spectrum from the datacube
    
        Args:
            pixel_coord (list): the pixel coordinate [x, y]
        """
        datashape = self.shape
        # make cutout image and calculate aperture
        if pixel_coord is None:
            if sky_coord is not None:
                if self.wcs is None:
                    raise ValueError("Cannot convert the skycoords into pixels! No wcs info can be found!")
                pixel_coord = np.round(wcs_utils.skycoord_to_pixel(sky_coord, wcs=self.wcs)).astype(int)
            else:
                # collapse the whole datacube into one spectrum
                spectrum = np.sum(np.sum(self.cube, axis=1), axis=1) * self.unit

        if pixel_coord is not None:
            if pixel_aperture is not None:
                aperture = EllipticalAperture(pixel_coord, *pixel_aperture)
                aper_mask = aperture.to_mask().to_image(self.imagesize).astype(bool)
                aper_mask_3D = np.repeat(aper_mask[None,:,:], self.nchan, axis=0)
                data_selected = self.cube[aper_mask_3D]
                spectrum = np.sum(data_selected.reshape((self.nchan, np.sum(aper_mask))), axis=1)*self.unit
            else:
                spectrum = self.cube[:,pixel_coord[1], pixel_coord[0]]*self.unit
            
        if plot:
            if ax is None:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111)
            ax.step(self.chandata, spectrum, where='mid')
            ax.set_xlabel(self.chandata.unit)
            ax.set_ylabel('Flux [Jy]')
            ax_top = ax.twiny()
            ax_top.step(np.arange(self.nchan), spectrum, where='mid', alpha=0)
        return self.chandata, spectrum*self.unit
    def collapse_cube(self, chans='', chanunit=None, return_header=False, 
                      reference=None, moment=0, mask=None):
        """collapse the cube to make one channel image

        Args:
            chans (str): follow the format of casa, like "20~30;45~90"

        Return:
            class::Image
        TODO: return the averaged beam shape
        """
        # decoding the chans
        selected_indices = []
        data = self.data
        if chanunit is not None:
            self.convert_chandata(chanunit, reference=reference)
        if chans == '':
            selected_indices = list(range(self.chandata.size))
        else:
            chan_par = re.compile(r"(\d+)~(\d+)")
            chan_ranges = np.array(chan_par.findall(chans)).astype(int)
            for chan_range in chan_ranges:
                # casa includes the last channel with the chan selection rules
                selected_indices.append(list(range(chan_range[0], chan_range[1]+1))) 
            selected_indices = [item for sublist in selected_indices for item in sublist]
        chandata_selected = self.chandata[selected_indices]
        dchan_selected = calculate_diff(self.chandata)[selected_indices]
        chan_selected = self.chandata[selected_indices]
        data_selected = data[selected_indices]
        collapsed_image = make_moments(data_selected, chandata_selected.value, moment=moment, mask=mask)
        # M0 = np.sum(data_selected * dchan_selected[:,None,None], axis=0)
        # if moment == 0:
            # collapsed_image = M0
        # if moment > 0:
            # M1 = np.sum(data_selected*chan_selected[:,None,None]*dchan_selected[:,None,None], axis=0)/M0
            # if moment == 1:
                # collapsed_image = M1
            # if moment > 1:
                # collapsed_image = np.sum(data_selected*(chan_selected[:,None,None]-M1[None,:,:])**moment*dchan_selected[:,None,None], axis=0)/M0
        collapsed_beam = self.beam
        collapsed_header = self.wcs.sub(['longitude','latitude']).to_header()
        return [collapsed_image, collapsed_header, collapsed_beam]
    def subcube(self, s_):
        """extract the subsection of the datacube

        Args:
            s_ (object:slice): the slice object, should be three dimension

        Return:
            Cube
        """
        # if not isinstance(s_, slice):
            # raise ValueError("Please input a numpy slice.")
        cube_sliced = self.data[s_].copy() # force to use new memory
        self_wcs = self.wcs
        if self_wcs is not None:
            wcs_sliced = self_wcs[s_]
            shape_sliced = cube_sliced.shape
            header_sliced = wcs_sliced.to_header()
        else:
            header_sliced = fits.Header()
        header_sliced.set('NAXIS',len(shape_sliced))
        header_sliced.set('NAXIS1',shape_sliced[2])
        header_sliced.set('NAXIS2',shape_sliced[1])
        header_sliced.set('NAXIS3',shape_sliced[0])
        return Cube(data=cube_sliced, header=header_sliced, beam=self.beam)
    def find_3Dstructure(self, **kwargs):
        if isinstance(self.cube, u.Quantity):
            return find_3Dstructure(self.cube.value, **kwargs)
        else:
            return find_3Dstructure(self.cube, **kwargs)
    def writefits(self, filename, overwrite=False):
        """write datacube into fitsfile

        This function also shifts the reference pixel of the image to the image center
        and the reference pixel of the spectrum to the 1 (the standard starting)
        """
        # remove the History and comments of the header
        header = self.header
        try:
            header.remove('HISTORY', remove_all=True)
            header.remove('COMMENT', remove_all=True)
        except:
            pass
        ysize, xsize = self.imagesize
        # try: # shift the image reference pixel, tolerence is 4 pixels
            # if header['CRPIX1'] < (xsize//2-4) or header['CRPIX1'] > (xsize//+4):
                # header['CRVAL1'] += (1 - header['CRPIX1']) * header['CDELT1']
                # header['CRPIX1'] = xsize//2
                # print('Warning: automatically shifted the x axis reference.')
            # if header['CRPIX2'] < (ysize//-4) or header['CRPIX2'] > (ysize//+4):
                # header['CRVAL2'] += (1 - header['CRPIX2']) * header['CDELT2']
                # header['CRPIX2'] = ysize//2
                # print('Warning: automatically shifted the y axis reference.')
        # except: pass
        # try: # shift the channel reference to the starting channel
            # if header['CRPIX3'] != 1:
                # header['CRVAL3'] += (1 - header['CRPIX3']) * header['CDELT3']
                # header['CRPIX3'] = 1
                # print('Warning: automatically shifted the channel reference.')
        # except: pass
        # header.set('BUNIT', self.unit.to_string())
        header.update({'history':'created by cube_utils.Cube',})
        hdu_list = []
        # write the beams
        beam = self.beam
        header.update({'BMAJ':beam[0]/3600., 'BMIN':beam[1]/3600., 
                       'BPA':self.beam[2]})
        # if ndim_beam == 2:
            # beams_T = self._beams.T
            # c1 = fits.Column(name='BMAJ', array=beams_T[0], format='D', unit='arcsec') # double float
            # c2 = fits.Column(name='BMIN', array=beams_T[1], format='D', unit='arcsec') # doubel float
            # c3 = fits.Column(name='BPA', array=beams_T[2], format='D', unit='deg') # double float
            # hdu_list.append(fits.BinTableHDU.from_columns([c1,c2,c3], name='BEAMS'))
        hdu_list.append(fits.PrimaryHDU(data=self.data, header=header))
        hdus = fits.HDUList(hdu_list)
        hdus.writeto(filename, overwrite=overwrite)
    def readfits(self, fitscube, extname='Primary', debug=False):
        """read general fits file
        """
        with fits.open(fitscube) as cube_hdu:
            if debug:
                print(cube_hdu.info())
            cube_header = cube_hdu[extname].header
            cube_data = cube_hdu[extname].data 
            # if 'BUNIT' in cube_header.keys():
                # cube_unit = u.Unit(cube_header['BUNIT'])
                # cube_data = cube_data * cube_unit
            try:
                beam_data = cube_hdu['beams'].data
                cube_beam = np.array([beam_data['BMAJ'], beam_data['BMIN'], beam_data['BPA']]).T
            except: cube_beam = None
        self.data=cube_data
        self.header=cube_header
        self._beam=cube_beam
    def read_ALMA(self, fitscube, debug=False, stokes=0, name=None):
        """read the fits file
        """
        _cube = read_ALMA_cube(fitscube, debug=debug, stokes=stokes, name=name)
        self.data = _cube.data
        self._header = _cube._header
        self._beam = _cube._beam # necessary as header does not have beam info
        self.name = _cube.name

#################################
###   stand alone functions   ###
#################################

def read_ALMA_cube(fitscube, debug=False, stokes=0, name=None):
    """read the fits file
    """
    with fits.open(fitscube) as cube_hdu:
        if debug:
            print(cube_hdu.info())
        cube_header = cube_hdu['primary'].header
        cube_data = cube_hdu['primary'].data #* u.Unit(cube_header['BUNIT'])
        try:
            beams_data = cube_hdu['beams'].data
            cube_beam = np.array([beams_data['BMAJ'], beams_data['BMIN'], beams_data['BPA']]).T
        except: cube_beam = None
        if False: # convert the units
            if '/beam' in cube_header['BUNIT']: # check whether it is beam^-1
                pixel2arcsec_ra = abs(cube_header['CDELT1']*u.Unit(cube_header['CUNIT1']).to(u.arcsec)) 
                pixel2arcsec_dec = abs(cube_header['CDELT2']*u.Unit(cube_header['CUNIT2']).to(u.arcsec))       
                pixel_area = pixel2arcsec_ra * pixel2arcsec_dec # in arcsec2
                beamsize = calculate_beamsize(cube_beams, debug=debug) / pixel_area 
                cube_data = cube_data / beamsize[None,:,None,None] * u.beam
    return Cube(data=cube_data[stokes], header=cube_header, beam=cube_beam, name=name)

def calculate_beamsize(beams, debug=False):
    """quick function to calculate the beams size
    
    Args:
        beams (list, ndarray): it can be one dimension [bmaj, bmin, bpa] or two dimension, like [[bmaj,bmin,bpa],[bmaj,bmin,bpa],]
    """
    ndim = np.ndim(beams)
    if debug:
        print(f"func::calculate_beamsize --> dimension of beams is {ndim} {beams.shape}")
    if ndim == 1:
        return 1/(np.log(2)*4.0) * np.pi * beams[0] * beams[1]
    if ndim == 2:
        return 1/(np.log(2)*4.0) * np.pi * beams[:,0] * beams[:,1]

def solve_cubepb(pbfile=None, datafile=None, pbcorfile=None, header=None):
    """A helper function to derive the primary beam correction
    """
    if pbfile is not None:
        with fits.open(pbfile) as hdu:
            pbdata = hdu[0].data
            if header is None:
                header = hdu[0].header
    elif (datafile is not None) and (pbcorfile is not None):
        with fits.open(datafile) as hdu:
            data = hdu[0].data
            if header is None:
                header = hdu[0].header
        with fits.open(pbcorfile) as hdu:
            pbcordata = hdu[0].data
            pbdata = data / pbcordata
    else:
        raise ValueError("No valid primary beam information has been provided!")
    return Cube(data=pbdata, header=header, name='pb')

def find_3Dstructure(data, sigma=2.0, std=None, minsize=9, opening_iters=1, 
                     dilation_iters=1, mask=None, label_coords=None,
                     plot=False, ax=None, debug=False):
    """this function use scipy.ndimage.label to search for continuous structures in the datacube

    To use this function, the image should not be oversampled! If so, increase the minsize is recommended

    #TODO: ad support for bianry_opening and binary_dilation
    """
    s3d = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[1,1,1],[0,1,0]], 
                            [[0,0,0],[0,1,0],[0,0,0]]])
    if std is None:
        mean, median, std = sigma_clipped_stats(data, sigma=sigma, maxiters=1, mask=mask)
    sigma_struct = data > (sigma * std)

    if opening_iters > 0:
        sigma_struct = ndimage.binary_opening(sigma_struct, iterations=opening_iters, 
                                              structure=s3d)
    if dilation_iters > 0:
        sigma_struct = ndimage.binary_dilation(sigma_struct, iterations=dilation_iters,
                                               structure=s3d)
    labels, nb = ndimage.label(sigma_struct, structure=s3d)
    if debug:
        print('nb=', nb)
    # label_new = 1
    if label_coords is not None:
        labels_selected = []
        for coord in label_coords:
            _z,_y,_x = coord
            labels_selected.append(labels[_z,_y,_x])
            print('labels_selected', labels_selected)
    else: labels_selected = None
    if labels_selected is not None:
        if debug:
            print("constructing structure based on selected labels")
        labeled_struct = np.zeros(labels.shape).astype(bool)
        for i in labels_selected:
            labeled_struct[labels==i] = True
    else:
        # cleaning the small labels
        for i in range(nb):
            if debug:
                print(f"working on label={i}")
            struct_select = ndimage.find_objects(labels==i)
            if debug:
                print("structure size: {}".format(sigma_struct[struct_select[0]].size))
            if sigma_struct[struct_select[0]].size < minsize:
                sigma_struct[struct_select[0]] = False
            labeled_struct = sigma_struct
    if plot:
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(labeled_struct)
        ax.set(xlabel='channel', ylabel='y', zlabel='x')
    return labeled_struct

def make_moments(data, chandata, moment=0, mask=None):
    data = np.ma.array(data, mask=mask)
    dchan = calculate_diff(chandata)
    M0 = np.sum(data * dchan[:,None,None], axis=0)
    if moment == 0:
        moment_image = M0
    if moment > 0:
        M1 = np.sum(data*chandata[:,None,None]*dchan[:,None,None], axis=0)/M0
        if moment == 1:
            moment_image = M1
        if moment > 1:
            moment_image = np.sum(data*(chandata[:,None,None]-M1[None,:,:])**moment*dchan[:,None,None], axis=0)/M0
    return moment_image

def gaussian1D(x, params):
    """ 1-D gaussian function with/withou continuum

    Args:
        x: the independent variable
        params: the gaussian parameters in the order of:
                [amp, mean, sigma, (continuum)]
    """
    if len(params) == 3:
        amp, mean, sigma = params
        cont = 0
    elif len(params) == 4: # fit the continuum
        amp, mean, sigma, cont = params
    return amp / ((2*np.pi)**0.5*sigma) * np.exp(-0.5*(x-mean)**2/sigma**2) + cont

def chi2_cost(params, vel, spec, std):
    """calculate the chi square

    Args:
        params: the required parameters of the function

    """
    return np.sum((spec-gaussian1D(vel, params))**2/std**2)

def fit_gaussian1D(vel, spec, std, vsig0=100, amp_bounds=[0, np.inf], vel_bounds=[-1000,1000], 
            sigma_bouds=[20,2000], cont_bounds=[0, np.inf], debug=False, fitcont=False,):
    """fit one spectrum in the velocity convension

    Args:
        vel: velocity
        spec: spectrum data
        std: the error or standard deviation of the spectrum
        vsig0: initial guess for the velocity dispersion
    """
    # guess the initial parameters
    ## amplitude
    spec_selection = spec > np.percentile(spec, 85)
    ## central velocity
    vel0 = np.median(vel[spec_selection])
    ## amplitude
    amp0 = np.mean(spec[spec_selection])
    if fitcont: ## continuum
        cont = np.median(spec[~spec_selection])
        initial_guess = [amp0, vel0, vsig0, cont]
        bounds = [amp_bounds, vel_bounds, sigma_bouds, cont_bounds]
    else:
        initial_guess = [amp0, vel0, vsig0]
        bounds = [amp_bounds, vel_bounds, sigma_bouds]
    args = vel, spec, std
    result = optimize.minimize(chi2_cost, initial_guess, args=args,
                bounds=bounds)
    # make profile of the initial guess and best-fit.
    guess = gaussian1D(vel, initial_guess)
    bestfit = gaussian1D(vel, result.x)
    if debug:
        plt.clf()
        plt.step(vel, spec, color='black', where='mid')
        plt.plot(vel, guess, linestyle='dashed',color='blue')
        plt.plot(vel, bestfit, color='red')
        plt.show(block=False)
        plt.pause(0.01)
    # calculate chi2
    chi2 = chi2_cost(result.x, *args)
    return result.x, bestfit, chi2
    # fitobj = Fitspectrum()
    # fitobj.fitparams = result.x
    # if fitcont:
        # fitobj.fitnames = ('amplitude', 'mean', 'sigma', 'continuum')
    # else:
        # fitobj.fitnames = ('amplitude', 'mean', 'sigma')
    # fitobj.bestfit = bestfit
    # fitobj.fitfunc = lambda a: gaussian1D(a, result.x)
    # fitobj.specchan = vel
    # fitobj.specdata = spec
    # fitobj.chi2 = chi2
    # return fitobj

def fitcube(vel, cube, snr_limit=5, std=1, minaper=1, maxaper=4, debug=False,
          vlow=-1000, vup=1000, return_fitcube=False, fitcont=False):
    """run gaussian fitting for all the pixels

    Args:
    """
    cube_shape = cube.shape
    # set up 3D array to hold the best fit parameters
    # including: amp, v_mean, v_sigma, cont, snr, aper,  
    fitmaps = np.full((6, cube_shape[-2], cube_shape[-1]), fill_value=np.nan)
    if return_fitcube:
        fitcube = np.zeros_like(cube)
    
    # loop over all pixels in the cube and fit 1d spectra
    for y in range(0, cube_shape[-2]):
        for x in range(0, cube_shape[-1]):
            # foundfit is 0 until a fit is found
            foundfit = False
            # loop over the range of adaptive binning vakues
            for aper in range(minaper, maxaper+1):
                # if foundfit is still zero, then attempt a fit.
                if not foundfit:
                    # deal with the edges of the cube
                    sz = cube_shape
                    xlo = x - aper
                    xhi = x + aper
                    ylo = y - aper
                    yhi = y + aper
                    # deal with the edges
                    if xlo <= 0: xlo = 0
                    if xhi > sz[2]: xhi = sz[2]
                    if ylo <= 0: ylo = 0
                    if yhi >= sz[1]: yhi = sz[1]
                    
                    # average the spectra within the aperture
                    spec = np.mean(cube[:,ylo:yhi,xlo:xhi], axis=(1,2))

                    if not np.isnan(np.sum(spec)):
                        vel_wing = (vel < vlow) | (vel > vup)
                        cont = np.median(spec[vel_wing])
                        if std is None:
                            std = np.std(spec[vel_wing])                            
                        
                        # get chi^2 of straight line fit
                        chi2_sline = np.sum((spec-cont)**2 / std**2)

                        # do a 1D Gaussian profile fit of the line
                        paramsfit, bestfit, chi2 = fit_gaussian1D(vel, spec, std, debug=debug, fitcont=fitcont)

                        # calculate the chi^2 of the Gaussian profile fit
                        chi2_gaussian = np.sum((spec-bestfit)**2 / std**2)

                        # calculate the S/N of the fit by comparing the chi^2 values.  sqrt(delta_chi^2)=S/N
                        snr = (chi2_sline - chi2_gaussian)**0.5

                        # if the S/N is above threshold, store the fit parameters and set foundfit=1
                        if snr >= snr_limit:
                            if debug:
                                print(f'fit found at {(x,y)} with aper={aper} and S/N={snr}')
                            foundfit = 1
                            if len(paramsfit) == 3:
                                fitmaps[0:3, y, x] = paramsfit # v, I, sigma
                            elif len(paramsfit) == 4:
                                fitmaps[0:4, y, x] = paramsfit # v, I, sigma, continuum
                            fitmaps[4, y, x] = snr
                            fitmaps[5, y, x] = aper
                            if return_fitcube:
                                fitcube[:,y,x] = bestfit
                        else:
                            if debug:
                                print(f'no fit found at {(x,y)} with aperture size {aper} and S/N={snr}')
    if return_fitcube:
        return fitcube, fitmaps
    return fitmaps

def calculate_diff(v):
    """approximate the differential v but keep the same shape
    """
    nchan = len(v)
    dv_cropped = np.diff(v)
    if isinstance(v, u.quantity.Quantity):
        v_expanded = np.zeros(nchan+1) * v.unit
    else:
        v_expanded = np.zeros(nchan+1)
    v_expanded[:-2] = v[:-1]-0.5*dv_cropped
    v_expanded[2:] = v_expanded[2:] + v[1:]+0.5*dv_cropped
    v_expanded[2:-2] = 0.5*v_expanded[2:-2]
    return np.abs(np.diff(v_expanded))

#################################
###      quick functions      ###
#################################

def subcube(fitscube, outfile=None, sky_center=None, sky_radius=None, 
            pixel_center=None, pixel_radius=None, bltr=None, chans=None,
            overwrite=True):
    """split a subcube from a big cube

    Args:
     fitscube: the datacube fits file
     sky_center: the sky coordinate of the center
    """
    cube = Cube.read(fitscube)
    # convert the sky coordinate to pixel coordinate
    if sky_center is not None:
        if not isinstance(sky_center, coordinates.SkyCoord):
            if ':' in sky_center or 'm' in sky_center:
                sky_center = SkyCoord(sky_center, unit=(u.hourangle, u.deg))
            else:
                ra, dec = sky_center.split(" ")
                sky_center = SkyCoord(float(ra), float(dec), unit='deg')
            pixel_center = cube.skycoords2pixels(sky_center)
    if sky_radius is not None:
        if isinstance(sky_radius, str):
            sky_radius = u.Quantity(sky_radius)
        if isinstance(sky_radius, u.Quantity):
            sky_radius = sky_radius.to(u.arcsec)
        pixel2arcsec_ra, pixel2arcsec_dec = cube.pixel_sizes
        # here we only use the pixel scale in ra
        pixel_radius = sky_radius/pixel2arcsec_ra
    if isinstance(chans, str):
        chans = np.array(chans.split('~')).astype(int)
    # convert the pixel coordinates to bltr
    if pixel_center is not None:
        bottom_left = pixel_center - pixel_radius
        top_right = pixel_center + pixel_radius
        bltr = [*bottom_left, *top_right]

    # convert bltr to slice
    xlow, xup = bltr[0], bltr[-2]
    ylow, yup = bltr[1], bltr[-1]
    if chans is None:
        subcube = cube.subcube(np.s_[:,ylow:yup,xlow:xup])
    else:
        subcube = cube.subcube(np.s_[chans[0]:chans[1],ylow:yup,xlow:xup])
    subcube.writefits(outfile, overwrite=overwrite)


#################################
###          testing          ###
#################################

def test_cube():
    """a small test suit for `cube_utils`
    """
    pass
    # create a mock cube
    # cut out a subcube
    # write to a fits file
    # read again the fits file
    # check all the properties
    # make a spectrum fitting
    # make moments maps


#################################
###        CMD wrapper        ###
#################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='cube_utils.py',
            description=f"Welcome to Jianhang's cube utilities {__version__}",
            epilog='Reports bugs and problems to cjhastro@gmail.com')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    parser.add_argument('--logfile', default=None, help='the filename of the log file')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')

    # add subparsers
    subparsers = parser.add_subparsers(
            title='Available task', dest='task', 
            metavar=textwrap.dedent(
        '''
          * subcube: get a subcube from a big cube

          To get more details about each task:
          $ cube_utils.py task_name --help
        '''))

    # define the tasks
    subp_subcube = subparsers.add_parser('subcube',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            cut a cube from a big cube
            --------------------------------------------
            Examples:
    
              cube_utils subcube --

            '''))
    subp_subcube.add_argument('--fitscube', type=str, help='The input fits datacube')
    subp_subcube.add_argument('--outfile', type=str, help='The output file name')
    subp_subcube.add_argument('--bltr', type=int, nargs='+', help='The pixel coordinate of [bottom left, top_right], e.g. 10 10 80 80')
    subp_subcube.add_argument('--sky_center', type=str, help='The sky coordinate of the center,  e.g. "1:12:43.2 +31:12:43" or "1h12m43.2s +1d12m43s" or "23.5 -34.2"')
    subp_subcube.add_argument('--sky_radius', type=str, help='The angular distance between the center and the boundaries,  e.g. "3.5*u.arcsec"')
    subp_subcube.add_argument('--pixel_center', type=int, nargs='+', help='The pixel coordinate of the center,  e.g. "80 80"')
    subp_subcube.add_argument('--pixel_radius', type=int, help='The pixel distance between the center and the boundaries,  e.g. "10"')
    subp_subcube.add_argument('--chans', type=str, help='The pixel distance between the center and the boundaries,  e.g. "10~20"')

    args = parser.parse_args()
    logging.basicConfig(filename=args.logfile, encoding='utf-8', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Welcome to cube_utils.py {__version__}")
 
    if args.debug:
        logging.debug(args)
        func_args = list(inspect.signature(locals()[args.task]).parameters.keys())
        func_str = f"Executing:\n \t{args.task}("
        for ag in func_args:
            try: func_str += f"{ag}={args.__dict__[ag]},"
            except: func_str += f"{ag}=None, "
        func_str += ')\n'
        logging.debug(func_str)

    if args.task == 'subcube':
        subcube(args.fitscube, outfile=args.outfile, 
                sky_center=args.sky_center, sky_radius=args.sky_radius,
                pixel_center=args.pixel_center, pixel_radius=args.pixel_radius,
                bltr=args.bltr,
                chans=args.chans,
                )

    logging.info('Finished')
