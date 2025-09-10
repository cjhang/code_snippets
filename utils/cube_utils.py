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
    2022-02-16: first release to handle the datacube from ALMA, v0.1


"""

__version__ = '0.1.2'


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
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve, Gaussian2DKernel

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
# from spec_utils import convert_spec, Spectrum

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
        if isinstance(data, u.Quantity):
            self.data = data.data
            unit = data.unit
        else:
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
        elif (fullwcs is not None) and (fullwcs.naxis == 3):
            return fullwcs
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
        if self.unit != u.Unit(''):
            print('unit converting factor"', self.unit.to(unit))
            self.data = self.data * self.unit.to(unit)
            self._unit = u.Unit(unit)
            self._header.update({'BUNIT': self._unit.to_string()})
        else:
            self._unit = u.Unit(unit)
    @property
    def cube(self):
        return self.data*self.unit
    @property
    def chandata(self):
        if self._chandata is not None:
            return self._chandata
        else:
            chandata = get_chandata(header=self.header)
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
        # convert arcsec to deg
        header_beams = self.beam * np.array([1/3600., 1/3600, 1]) 
        self._header.update({'BMAJ':header_beams[0], 'BMIN':header_beams[1], 
                             'BPA':header_beams[2]})
    @property
    def pixel_sizes(self):
        if self.header is not None:
            # Return the pixel size encoded in the header
            # In casa, it is in the units of deg, thus the returned value is pixel to deg
            try:
                delt_unit1 = u.Unit(self.header['CUNIT1'])
            except:
                delt_unit1 = u.deg

            if 'CDELT1' in self.header:
                pixel2arcsec_ra = (abs(self.header['CDELT1'])*delt_unit1).to(u.arcsec)
            elif 'CD1_1' in self.header:
                pixel2arcsec_ra = (abs(self.header['CD1_1'])*delt_unit1).to(u.arcsec)
            else:
                print("Don't find valid pixel size in ra")
                pixel2arcsec_ra = 1
            try:
                delt_unit2 = u.Unit(self.header['CUNIT2'])
            except:
                delt_unit2 = u.deg
            if 'CDELT2' in self.header:
                pixel2arcsec_dec = (abs(self.header['CDELT2'])*delt_unit2).to(u.arcsec)
            elif 'CD2_2' in self.header:
                pixel2arcsec_dec = (abs(self.header['CD2_2'])*delt_unit2).to(u.arcsec)
            else:
                print("Don't find valid pixel size in dec")
                pixel2arcsec_dec = 1

        else:
            pixel2arcsec_ra, pixel2arcsec_dec = 1, 1
        # return np.array([pixel2arcsec_ra, pixel2arcsec_dec])
        return u.Quantity([pixel2arcsec_ra, pixel2arcsec_dec])
    @property
    def pixel_beam(self):
        if self.beam is None:
            warnings.warn('No beam infomation has been found!')
            return None
        try:
            bmaj, bmin, bpa = self.beam
            pixel_sizes = self.pixel_sizes
            x_scale = 1/pixel_sizes[0].to(u.arcsec).value
            y_scale = 1/pixel_sizes[1].to(u.arcsec).value
            bmaj_pixel = np.sqrt((bmaj*np.cos(bpa/180*np.pi)*x_scale)**2 
                                 + (bmaj*np.sin(bpa/180*np.pi)*y_scale)**2)
            bmin_pixel = np.sqrt((bmin*np.sin(bpa/180*np.pi)*x_scale)**2 
                                 + (bmin*np.cos(bpa/180*np.pi)*y_scale)**2)
            pixel_beam = [bmaj_pixel, bmin_pixel, bpa]
        except:
            pixel_beam = None
        return pixel_beam

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
        if reference is None:
            reference = self.reference
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
            pixel_aperture (list): the pixel aperture coded as [maj, min, theta]
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
        if sky_aperture is not None:
            if not isinstance(sky_aperture, SkyEllipticalAperture):
                if isinstance(sky_aperture, (list, tuple)):
                    sky_aperture = SkyEllipticalAperture(sky_coord, *sky_aperture)
                elif isinstance(sky_aperture, u.Quantity):
                    sky_aperture = SkyEllipticalAperture(sky_coord, sky_aperture, 
                                                         sky_aperture, 0*u.deg)
                else:
                    raise ValueError("Unsupported sky_aperture!")
            pixel_aperture = sky_aperture.to_pixel(self.wcs.celestial)
        if pixel_aperture is not None:
            if not isinstance(pixel_aperture, EllipticalAperture):
                if isinstance(pixel_aperture, [list, tuple]):
                    pixel_aperture = EllipticalAperture(pixel_coord, *pixel_aperture)
                elif isinstance(pixel_aperture, [float, int]):
                    pixel_aperture = EllipticalAperture(pixel_coord, *pixel_aperture)
                else:
                    raise ValueError("Unsupported pixel_aperture!")

            aper_mask = pixel_aperture.to_mask().to_image(self.imagesize
                                                          ).astype(bool)
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
            try:
                ax.set_ylabel(spectrum.unit.to_string())
            except:
                pass
            ax_top = ax.twiny()
            ax_top.step(np.arange(self.nchan), spectrum, where='mid', alpha=0)
        return self.chandata, spectrum*self.unit
    def extract_slit_spectrum(self, center=None, length=None, width=None, step=1, theta=0, 
                              plot=False):
        return extract_slit_spectra(self.data, specdata=self.specdata, pixel_center=center, 
                                    length=length, width=wdith, 
                                    step=step, theta=theta, plot=plot)
    def collapse_cube(self, chans='', frequency_range=None, velocity_range=None, 
                      chanunit=None, return_header=False, 
                      reference=None, moment=0, mask=None):
        """collapse the cube to make one channel image

        Args:
            chans (str): follow the format of casa, like "20~30;45~90"
            frequency_range: [100,200] or np.array([100,200])*u.GHz,
                             if there is no unit, is will be assumed to 
                             be the same as chandata

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
        if frequency_range is not None:
            if not self.chandata.unit.is_equivalent('Hz'):
                raise ValueError('channel data is not equivalent to frequency')
            if isinstance(frequency_range, u.Quantity):
                frequency_range = frequency_range.to(self.chandata.unit).value
            if isinstance(self.chandata, u.Quantity):
                freq_value = self.chandata.value
            else:
                freq_value = self.chandata
            freq_selection = (freq_value > frequency_range[0]) & (freq_value < frequency_range[1])
            selected_indices = np.where(freq_selection)[0].tolist()
        if velocity_range is not None:
            if not self.chandata.unit.is_equivalent('m/s'):
                raise ValueError('channel data is not equivalent to velocity')
            if isinstance(velocity_range, u.Quantity):
                velocity_range = velocity_range.to(self.chandata.unit).value
            if isinstance(self.chandata, u.Quantity):
                vel_value = self.chandata.value
            else:
                vel_value = self.chandata
            vel_selection = (vel_value > velocity_range[0]) & (vel_value < velocity_range[1])
            selected_indices = np.where(vel_selection)[0].tolist()

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
        try:
            header_sliced.set('CUNIT3', self.header_sliced['CUNIT3'])
        except:
            pass
        for key, value in self.header.items():
            if key not in header_sliced:
                header_sliced.set(key, value)
        return Cube(data=cube_sliced, header=header_sliced, beam=self.beam)
    def find_3Dstructure(self, **kwargs):
        if isinstance(self.cube, u.Quantity):
            return find_3Dstructure(self.cube.value, **kwargs)
        else:
            return find_3Dstructure(self.cube, **kwargs)
    def extract_pv_diagram(self, reference=None, pixel_center=None, 
                           length=20, width=10, theta=0, **kwargs):
        if self.chandata.unit.is_equivalent('km/s'):
            velocity = self.get_velocity().to(u.km/u.s).value
        else:
            try: velocity = self.get_velocity(reference=reference)
            except: raise ValueError("Not reference to get the velocity!")
        return extract_pv_diagram(self.data, velocity.to(u.km/u.s).value, 
                                  pixel_center=pixel_center, length=length, width=width,
                                  theta=theta, **kwargs)
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
    def readfits(self, fitscube, extname='Primary', stokes=0, debug=False):
        """read general fits file
        """
        with fits.open(fitscube) as cube_hdu:
            if debug:
                print(cube_hdu.info())
            cube_header = cube_hdu[extname].header
            naxis = cube_header['NAXIS'] 
            if naxis == 4:
                cube_data = cube_hdu[extname].data[stokes]
            elif naxis == 3:
                cube_data = cube_hdu[extname].data
            else:
                raise ValueError('Fits is not datacube!')
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

def read_ALMA_cube(fitscube, debug=False, stokes=0, name=None, corret_beam=False):
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
        if corret_beam: # convert the units
            if '/beam' in cube_header['BUNIT']: # check whether it is beam^-1
                pixel2arcsec_ra = abs(cube_header['CDELT1']*u.Unit(cube_header['CUNIT1']).to(u.arcsec)) 
                pixel2arcsec_dec = abs(cube_header['CDELT2']*u.Unit(cube_header['CUNIT2']).to(u.arcsec))       
                pixel_area = pixel2arcsec_ra * pixel2arcsec_dec # in arcsec2
                beamsize = calculate_beamsize(cube_beams, debug=debug) / pixel_area 
                cube_data = cube_data / beamsize[None,:,None,None] * u.beam
    return Cube(data=cube_data[stokes], header=cube_header, beam=cube_beam, name=name)

def get_chandata(header=None, wcs=None, output_unit=None):
    """a general way to read the spectral axis from the header
    """
    if header is None:
        try:
            header = wcs.to_header()
            header['NAXIS3'],header['NAXIS2'],header['NAXIS1'] = wcs.array_shape
        except:
            raise ValueError("Please provide valid header or wcs!")
    header_keys = list(header.keys())
    if 'CDELT3' in header_keys:
        cdelt3 = header['CDELT3']
    elif 'CD3_3' in header_keys:
        cdelt3 = header['CD3_3']
    elif 'PC3_3' in header_keys:
        cdelt3 = header['PC3_3']
    else:
        print("No valid CDELT for the spectral axis, assuming CDELT3=1!")
        cdelt3 = 1
    # because wcs.slice with change the reference channel, the generated ndarray should 
    # start with 1 
    chandata = (header['CRVAL3'] + (np.arange(1, header['NAXIS3']+1)-header['CRPIX3']) * cdelt3)
    if 'CUNIT3' in header_keys:
        chandata = chandata*u.Unit(header['CUNIT3'])
        if output_unit is not None:
            chandata = chandata.to(u.Unit(output_unit)).value
    else:
        print("CUNIT3 is not fount, skip...")
    return chandata

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

def convert_spec(spec_in, unit_out, reference=None, mode='radio'):
    """convert the different spectral axis

    Args:
        spec_in (astropy.Quantity): any valid spectral data with units
        unit_out (str or astropy.Unit): valid units for output spectral axis
        reference: the reference frequency or wavelength, with units

    Return:
    """
    unit_in = spec_in.unit
    if not isinstance(unit_out, u.Unit):
        unit_out = u.Unit(unit_out)
    if spec_in.unit.is_equivalent(unit_out):
        return spec_in.to(unit_out)
    if reference is not None:
        if not isinstance(reference, u.Quantity):
            reference = reference * unit_in
        if reference.unit.is_equivalent('m'):
            refwave = reference
            reffreq = (const.c/refwave).to(u.GHz)
        elif reference.unit.is_equivalent('Hz'):
            reffreq = reference
            refwave = (const.c/reffreq).to(u.um)
    else:
        reffreq, refwave = None, None
    if spec_in.unit.is_equivalent('m'): # this is velocity
        if unit_out.is_equivalent('Hz'):
            return (const.c/spec_in).to(unit_out)
        elif unit_out.is_equivalent('km/s'):
            if mode=='radio':
                return ((spec_in-refwave)/spec_in*const.c).to(unit_out)
            elif mode=='optical':
                return ((spec_in-refwave)/refwave*const.c).to(unit_out)
    if spec_in.unit.is_equivalent('Hz'):
        if unit_out.is_equivalent('m'):
            return (const.c/spec_in).to(unit_out)
        elif unit_out.is_equivalent('km/s'):
            if mode=='radio':
                return ((reffreq-spec_in)/reffreq*const.c).to(unit_out)
            elif mode=='optical':
                return ((reffreq/spec_in-1)*const.c).to(unit_out)
    if spec_in.unit.is_equivalent('km/s'): # this is velocity
        if unit_out.is_equivalent('m'):
            if mode=='radio':
                return (refwave/(1-spec_in/const.c)).to(unit_out)
            elif mode=='optical':
                return (spec_in/const.c*refwave+refwave).to(unit_out)
        elif unit_out.is_equivalent('Hz'):
            if mode=='radio':
                return (reffreq*(1-spec_in/const.c)).to(unit_out)
            elif mode=='optical':
                return (reffreq/(spec_in/const.c+1)).to(unit_out)

def extract_pv_diagram(datacube, velocity=None, pixel_center=None,
               vmin=-1000, vmax=1000,
               length=10, width=2, theta=0, debug=False, plot=False, pixel_size=1):
    """generate the PV diagram of the cube

    Args:
        datacube: the 3D data [velocity, y, x]
        pixel_center: the center of aperture
        lengthth: the length of the aperture, in x axis when theta=0
        width: the width of the aperture, in y axis when theta=0
        theta: the angle in degree, from positive x to positive y axis
    """
    # nchan, ny, nx = datacube.shape
    # extract data within the velocity range 
    if velocity is not None:
        vel_selection = (velocity > vmin) & (velocity < vmax)
        datacube = datacube[vel_selection]
        velocity = velocity[vel_selection]
    cubeshape = datacube.shape
    aper = RectangularAperture(pixel_center, length, width, theta=theta/180*np.pi)
    s1,s2 = aper.to_mask().get_overlap_slices(cubeshape[1:])
    # cutout the image
    sliced_cube = datacube[:,s1[0],s1[1]]
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
        ax1.imshow(np.sum(datacube,axis=0),origin='lower')
        aper.plot(ax=ax1)
        ax2.imshow(np.sum(sliced_cube_rotated, axis=0), origin='lower')
        aper_rotated.plot(ax=ax2)
        ax3.imshow(sliced_pvmap.T, origin='lower')
    if plot:
        fig = plt.figure(figsize=(12,6))
        # ax = fig.subplots(1,2)
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)
        vmin, vmax = -1*np.nanstd(sliced_cube), 4*np.nanmax(sliced_cube)
        ax1.imshow(np.sum(datacube, axis=0), origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
        aper.plot(ax=ax1, color='white', linewidth=4)
        # show the pv-diagram
        # ax[1].imshow(sliced_pvmap, origin='lower', cmap='magma', extent=extent)
        positions = np.linspace(-0.5*length, 0.5*length, nxnew)
        if velocity is None:
            velocity = np.linspace(-1,1,nchan)
        vmesh, pmesh = np.meshgrid(velocity,positions)
        ax2.pcolormesh(vmesh, pmesh, sliced_pvmap.T, cmap='magma')
        ax2.set_xlabel('Velocity [km/s]')
    return sliced_pvmap.T

def extract_slit_spectra(datacube, specdata=None, pixel_center=None, 
                         length=10, width=2, step=2, theta=0, plot=False):
    """extract the spectrum along the slit

    Args:
        datacube: the 3D data [velocity, y, x]
        pixel_center: the center of aperture
        lengthth: the length of the aperture, in x axis when theta=0
        width: the width of the aperture, in y axis when theta=0
        theta: the angle in degree, from positive x to positive y axis
    """
    nchan, ny, nx = datacube.shape
    # caculate the pixel_centers along the slit
    nstep = int(0.5*length/step) # steps in both sides
    theta_radian = np.radians(theta)
    positions_positive = []
    positions_negative = []
    for i in range(nstep):
        xi = pixel_center[0] + i*step*np.cos(theta_radian)
        yi = pixel_center[1] + i*step*np.sin(theta_radian)
        positions_positive.append([xi, yi])
        # the opposite direction
        xi = pixel_center[0] - i*step*np.cos(theta_radian)
        yi = pixel_center[1] - i*step*np.sin(theta_radian)
        positions_negative.append([xi, yi])
    positions = positions_positive + positions_negative[::-1]
    apertures = RectangularAperture(positions, step, width, theta=theta_radian)
    spectra = np.zeros([len(apertures), nchan])
    for i,aper in enumerate(apertures):
        aper_mask = aper.to_mask().to_image([ny, nx]).astype(bool)
        aper_mask_3D = np.repeat(aper_mask[None,:,:], nchan, axis=0)
        data_selected = datacube[aper_mask_3D]
        spectrum = np.sum(data_selected.reshape((nchan, np.sum(aper_mask))), axis=1)
        spectra[i,:] = spectrum
    if plot:
        fig = plt.figure(figsize=(12,3))
        ax1 = plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=2)
        ax1.imshow(np.sum(datacube, axis=0), origin='lower', cmap='magma')
        for aper in apertures:
            aper.plot(ax=ax1, color='white', linewidth=1)
        for spec in spectra:
            if specdata is not None:
                ax2.step(specdata, spec, where='mid', alpha=0.3)
            else:
                ax2.plot(spec, alpha=0.3)
    return spectra
        
def cubechan_stat(datacube, n_boostrap=100, aperture=None, aperture_shape=None):
    cubesize = datacube.shape
    nchan = cubesize[-3]
    imagesize = cubesize[-2:]
    std_ref = np.zeros(nchan)
    # create ramdon aperture
    if aperture_shape is not None:
        pixel_x = np.random.random(n_boostrap) * imagesize[1] # 1 for x axis
        pixel_y = np.random.random(n_boostrap) * imagesize[0] # 0 for y axis
        pixel_coords_boostrap = np.vstack([pixel_x, pixel_y]).T
        apertures_boostrap = EllipticalAperture(pixel_coords_boostrap, *aperture_shape)
        for chan in range(nchan):
            chan_map = np.ma.masked_invalid(datacube[0, chan])
            noise_boostrap = aperture_photometry(chan_map, apertures_boostrap, 
                                                 mask=chan_map.mask)
            std_ref[chan] = np.std(noise_boostrap['aperture_sum'])
    else: # create pixel level std reference
        for chan in range(nchan):
            chan_map = np.ma.masked_invalid(datacube[0, chan])
            std_ref[chan] = np.ma.std(chan_map[~chan_map.mask])
        
    return std_ref

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

def fill_spectral_mask(cube, spectral_mask, padding=2, sigma=5., debug=False):
    """fill the data cube along with the spectral axis. It is a faster approach
    than the fill_mask in the 3D datacube, specified in the spectral axis

    Args:
        cube (ndarray): the 3D datacube
        spectral_mask (ndarray): the 1D spectral array show the masked channels
        padding (int): the padding region around the mask to calculate the median
                       value
        sigma (int, float): the sigma value for clipping the padding region

    Return:
        ndarray, same as input cube
    """
    nchan, ny, nx = cube.shape
    cube_filled = cube.copy()
    idx_masked = np.where(spectral_mask)[0]
    # find all the consecutive masked channels and combine they
    mask_idx_group = []
    idx_low = 0
    idx_up = 0
    for idx in idx_masked:
        if idx < idx_up+1:
            continue
        idx_low = idx
        idx_up = idx+1
        while idx_up in idx_masked:
            idx_up += 1
        mask_idx_group.append([idx_low, idx_up])
    if debug:
        print(f"Masking indices groups: {mask_idx_group}")
    # refilled the masked region with the median of nearby regions
    for idx_group in mask_idx_group:
        idx_near_low = np.max([idx_group[0]-padding, 0])
        idx_near_up = np.min([idx_group[1]+padding, nchan])
        if debug:
            print(f"Masking sub-idx group {idx_group}")
            print(f"Replacing the value from {idx_near_low}:{idx_group[0]} and {idx_group[1]}:{idx_near_up}")
        # cube_near = np.hstack([cube[idx_group[0]-padding:idx_group[0]]
                    # cube[idx_group[1]:idx_group[1]+padding]])
        cube_near = cube[list(range(idx_near_low, idx_group[0]))+
                         list(range(idx_group[1], idx_near_up))]
        near_background = sigma_clip(cube_near, sigma=sigma, axis=0)
        median_near_cube = np.repeat(np.nanmedian(near_background, axis=0)[None,:,:], 
                                     idx_group[1]-idx_group[0], axis=0)
        cube_filled[idx_group[0]:idx_group[1],:,:] = median_near_cube
    return cube_filled


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
     sky_radius: the sky radius, default with unit of arcsec
     pixel_center: the pixel coordinate of the center
                   default: the image center
    """
    cube = Cube()
    cube.readfits(fitscube)
    # cube.read_ALMA(fitscube)
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
        elif isinstance(sky_radius, u.Quantity):
            sky_radius = sky_radius.to(u.arcsec)
        else:
            sky_radius = sky_radius * u.arcsec
        pixel2arcsec_ra, pixel2arcsec_dec = cube.pixel_sizes.value
        # here we only use the pixel scale in ra
        pixel_radius = np.round(sky_radius.to(u.arcsec).value/pixel2arcsec_ra).astype(int)
    if isinstance(chans, str):
        chans = np.array(chans.split('~')).astype(int)
    # convert the pixel coordinates to bltr
    if pixel_radius is not None:
        if pixel_center is None:
            pixel_center = [cube.header['CRPIX1'], cube.header['CRPIX2']]
        bottom_left = np.round(pixel_center).astype(int) - pixel_radius
        top_right = np.round(pixel_center).astype(int) + pixel_radius
        bltr = [*bottom_left, *top_right]

    # convert bltr to slice
    xlow, xup = bltr[0], bltr[-2]
    ylow, yup = bltr[1], bltr[-1]
    if chans is None:
        subcube = cube.subcube(np.s_[:,ylow:yup,xlow:xup])
    else:
        subcube = cube.subcube(np.s_[chans[0]:chans[1]+1,ylow:yup+1,xlow:xup+1])
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
