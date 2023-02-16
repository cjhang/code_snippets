# -*- coding: utf-8 -*-
"""Utilities dealing with fits datacube

Author: Jianhang Chen, cjhastro@gmail.com
History:
    2022-08-11: first release to handle spectrum of ALMACAL data

Example:

        from cube_tools import read_fitscube_ALMA
        dc = read_fitscube_ALMA('test_data/alma_datacube_AR2623.image.fits')
        dc.chandata_in('km/s', reffreq=98.5*u.GHz)
        dc.collapse_cube('10~20')

Requirement:
    numpy
    matplotlib
    astropy >= 5.0

Todo:
    * For module TODOs

"""


import os
import sys
import re
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
import astropy.coordinates as coordinates

from photutils import aperture_photometry, find_peaks, EllipticalAperture, RectangularAperture, SkyEllipticalAperture
from astropy.modeling import models, fitting

class Cube(object):
    """The base data strcuture to handle 3D astronomy data
    """
    def __init__(self, data=None, header=None, wcs=None, chandata=None, beams=None):
        """initialize the cube

        Params:
            data: the 3D data with units (equivalent to Jy)
            header: the header can be identified by WCS
            wcs: it can be an alternative to the header
            specdata: the data in the third dimension, it can be velocity, frequency, wavelength. 
                It should have units
            beams: the beam shapes the units of [arcsec, arcsec, deg]. 
                It can be one dimension or two dimension
        """
        self.data = data
        self.header = header
        self.wcs = wcs
        self.chandata = chandata
        self.beams = beams
        if self.header is not None:
            if self.wcs is None:
                self.update_wcs()
            if self.chandata is None:
                self.update_chandata(self.header)
        if self.header is None:
            if self.wcs is not None:
                self.header = self.wcs.to_header()
                if self.chandata is None:
                    self.update_chandata()
    def __getitem__(self, i):
            return self.data[i]
    @property
    def ndim(self):
        return np.ndim(self.data)
    @property
    def imagesize(self):
        # the imagesize is in [ysize, xsize]
        try: return self.data.shape[-2:]
        except: return None
    @property
    def nchan(self):
        try: return self.data.shape[-3]
        except: return None
    @property
    def nstokes(self):
        try: return self.data.shape[-4]
        except: return None
    def update(self):
        self.update_wcs()
        self.update_chandata()
    def update_wcs(self, wcs=None):
        if wcs is not None:
            self.wcs = wcs
        else: 
            self.wcs = WCS(self.header)
    def update_chandata(self, header):
        # the fits standard start with index 1
        # because wcs.slice with change the reference channel, the generated ndarray should start with 1 
        self.chandata = (self.header['CRVAL3'] + (np.arange(1, self.header['NAXIS3']+1)-self.header['CRPIX3']) * self.header['CDELT3']
                         )*u.Unit(self.header['CUNIT3'])

    def convert_chandata(self, unit_out, refwave=None, reffreq=None):
        """convert the units of the specdata 
        """
        if refwave is not None:
            self.refwave = refwave
        if reffreq is not None:
            self.reffreq = reffreq
        self.chandata = convert_spec(self.chandata, unit_out, refwave=refwave, reffreq=reffreq)

    def velocity(self, reffreq=None, refwave=None):
        return convert_spec(self.chandata, 'km/s', refwave=refwave, reffreq=reffreq)

    def extract_spectrum(self, pixel_coord=None, sky_coord=None, pixel_aperture=None, 
                         sky_aperture=None, plot=False, ax=None):
        """extract 1D spectrum from the datacube
    
        """
        datashape = self.data.shape
        # make cutout image and calculate aperture
        if pixel_coord is None:
            if sky_coord is not None:
                if self.wcs is None:
                    raise ValueError("Cannot convert the skycoords into pixels! No wcs info can be found!")
                pixel_coord = np.round(wcs_utils.skycoord_to_pixel(sky_coord, wcs=self.wcs)).astype(int)
            else:
                # collapse the whole datacube into one spectrum
                spectrum = np.sum(np.sum(self.data, axis=1), axis=1)

        if pixel_coord is not None:
            if pixel_aperture is not None:
                aperture = EllipticalAperture(pixel_coord, *pixel_aperture)
                aper_mask = aperture.to_mask().to_image(self.imagesize).astype(bool)
                aper_mask_3D = np.repeat(aper_mask[None,:,:], self.nchan, axis=0)
                data_selected = self.data[0][aper_mask_3D]
                spectrum = np.sum(data_selected.reshape((self.nchan, np.sum(aper_mask))), axis=1)
            else:
                spectrum = self.data[:,:,pixel_coord[1], pixel_coord[0]]
            
        if plot:
            if ax is None:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111)
            ax.step(self.chandata, spectrum, where='mid')
            ax.set_xlabel(self.chandata.unit)
            ax.set_ylabel('Flux [Jy]')
            ax_top = ax.twiny()
            ax_top.step(np.arange(self.nchan), spectrum, where='mid', alpha=0)
        return Spectrum(specchan=self.chandata, specdata=spectrum)

    def collapse_cube(self, chans='', return_header=False):
        """collapse the cube to make one channel image

        Params:
            chans (str): follow the format of casa, like "20~30;45~90"

        Return:
            class::Image
        TODO: return the averaged beam shape
        """
        # decoding the chans
        selected_indices = []
        if self.ndim == 4:
            data = self.data[0]
        else:
            data = self.data
        chan_par = re.compile(r"(\d+)~(\d+)")
        chan_ranges = np.array(chan_par.findall(chans)).astype(int)
        for chan_range in chan_ranges:
            # casa includes the last channel with the chan selection rules
            selected_indices.append(list(range(chan_range[0], chan_range[1]+1))) 
        selected_indices = [item for sublist in selected_indices for item in sublist]
        dchan_selected = calculate_diff(self.chandata)[selected_indices]
        collapsed_image = np.sum(data[selected_indices] * dchan_selected[:,None,None], axis=0)
        collapsed_beam = np.median(self.beams[selected_indices], axis=0)
        collapsed_header = self.wcs.sub(['longitude','latitude']).to_header()
        return [collapsed_image, collapsed_header, collapsed_beam]
    

    def writefits(self, filename, overwrite=False):
        """write datacube into fitsfile

        This function also shifts the reference pixel of the image to the image center
        and the reference pixel of the spectrum to the 1 (the standard starting)
        """
        header = self.header
        if header is None:
            if self.wcs is not None:
                header = wcs.to_header()
            else:
                header = fits.Header()
        ysize, xsize = self.imagesize
        try: # shift the image reference pixel, tolerence is 4 pixels
            if header['CRPIX1'] < (xsize//2-4) or header['CRPIX1'] > (xsize//+4):
                header['CRVAL1'] += (1 - header['CRPIX1']) * header['CDELT1']
                header['CRPIX1'] = xsize//2
                print('Warning: automatically shifted the x axis reference.')
            if header['CRPIX2'] < (ysize//-4) or header['CRPIX2'] > (ysize//+4):
                header['CRVAL2'] += (1 - header['CRPIX2']) * header['CDELT2']
                header['CRPIX2'] = ysize//2
                print('Warning: automatically shifted the y axis reference.')
        except: pass
        try: # shift the channel reference to the starting channel
            if header['CRPIX3'] != 1:
                header['CRVAL3'] += (1 - header['CRPIX3']) * header['CDELT3']
                header['CRPIX3'] = 1
                print('Warning: automatically shifted the channel reference.')
        except: pass
        header.set('BUNIT', self.data.unit.to_string())
        header.update({'history':'cleaned by cube_tools.Cube',})
        hdu_list = []
        hdu_list.append(fits.PrimaryHDU(data=self.data.value, header=header))
        if self.beams is not None:
            beams_T = self.beams.T
            c1 = fits.Column(name='BMAJ', array=beams_T[0], format='D', unit='arcsec') # double float
            c2 = fits.Column(name='BMIN', array=beams_T[1], format='D', unit='arcsec') # doubel float
            c3 = fits.Column(name='BPA', array=beams_T[2], format='D', unit='deg') # double float
            hdu_list.append(fits.BinTableHDU.from_columns([c1,c2,c3], name='BEAMS'))
        hdus = fits.HDUList(hdu_list)
        hdus.writeto(filename, overwrite=overwrite)

    def subcube(self, s_):
        """extract the subsection of the datacube

        Params:
            s (object:slice): the slice object, should be four dimension

        ultimate goal is to seperate different source
        should return another Cube with valid wcs, header and beams
        """
        data_sliced = self.data[s_]
        wcs_sliced = self.wcs[s_]
        shape_sliced = data_sliced.shape
        header_sliced = wcs_sliced.to_header()
        header_sliced.set('NAXIS',len(shape_sliced))
        header_sliced.set('NAXIS1',shape_sliced[3])
        header_sliced.set('NAXIS2',shape_sliced[2])
        header_sliced.set('NAXIS3',shape_sliced[1])
        header_sliced.set('NAXIS4',shape_sliced[0])
        if self.beams is not None:
            beams_sliced = self.beams[s_[1]]
        else: 
            beams_sliced = None
        return Cube(data=data_sliced, header=header_sliced, beams=beams_sliced)

    def auto_measure(self):
        """designed for subcube

        With proper cutout, it should be used to extract the spectrum and measure the intensity
        automatically
        """
        pass

    @staticmethod
    def read(fitscube, debug=False):
        """read the fits file
        """
        with fits.open(fitscube) as cube_hdu:
            if debug:
                print(cube_hdu.info())
            cube_header = cube_hdu['primary'].header
            cube_data = cube_hdu['primary'].data
            try: cube_beams = cube_hdu['beams'].data
            except: cube_beams = None
        return Cube(data=cube_data, header=cube_header, beams=cube_beams)

class Spectrum(object):
    """the data structure to handle spectrum
    """
    def __init__(self, specchan=None, specdata=None):
        """ initialize the spectrum
        """
        self.specchan = specchan
        self.specdata = specdata

    def velocity(self, reffreq=None, refwave=None):
        return convert_spec(self.specchan, 'km/s', refwave=refwave, reffreq=reffreq)

    def to_restframe(self, z=0, restfreq=None):
        self.restfreq = convert_spec(self.specchan, 'GHz', reffreq=restfreq)
        self.restwave = convert_spec(self.restfreq, 'um')

    def fit_gaussian(self, reffreq=None, refwave=None, plot=False, ax=None):
        """fitting the single gaussian to the spectrum
        """
        fit_p = fitting.LevMarLSQFitter()
        if (reffreq is not None) or (refwave is not None):
            specvel = self.velocity(reffreq=reffreq, refwave=refwave)
            p_init = models.Gaussian1D(amplitude=np.max(self.specdata), mean=0*u.km/u.s, 
                                       stddev=50*u.km/u.s)
                    # bounds={"mean":np.array([-1000,1000])*u.km/u.s, 
                            # "stddev":np.array([0., 1000])*u.km/u.s,})
            p = fit_p(p_init, specvel, self.specdata)
            specfit = p(specvel)
        else:
            specvel = None
            p_init = models.Gaussian1D(amplitude=np.max(self.specdata), mean=np.mean(self.specchan), 
                                       stddev=5*np.median(np.diff(self.specchan)))
            p = fit_p(p_init, self.specchan, self.specdata)
            specfit = p(self.specchan)
        if plot:
            if ax is None:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111)
            if specvel is not None:
                ax.step(specvel, self.specdata, label='data')
                ax.plot(specvel, specfit, label='mode')
            else:
                ax.step(self.specchan, self.specdata, label='data')
                ax.plot(self.specchan, specfit, label='mode')
        return specfit


    def integral(self):
        return integral_spectrum(self.specvel, self.specdata)

#################################
###   stand alone functions   ###

def read_fitscube_ALMA(fitscube, pbcor=False, debug=False,):
    """read the datacube from the files and return with the Cube
    """
    with fits.open(fitscube) as cube_hdu:
        if debug:
            print(cube_hdu.info())
        cube_header = cube_hdu['primary'].header
        cube_data = cube_hdu['primary'].data
        cube_beams = cube_hdu['beams'].data
    cube_shape = cube_data.shape
    imagesize = cube_shape[-2:]
    nchan = cube_shape[-3]
    cube_wcs = WCS(cube_header)
    # read from the header
    pixel2arcsec_ra = abs(cube_header['CDELT1']*u.Unit(cube_header['CUNIT1']).to(u.arcsec)) 
    pixel2arcsec_dec = abs(cube_header['CDELT2']*u.Unit(cube_header['CUNIT1']).to(u.arcsec))       

    # handle the beams
    cube_beams = np.array([cube_beams['BMAJ'], cube_beams['BMIN'], cube_beams['BPA']]).T
    # In CASA/ALMA, the beams are stored in the units of [arcsec, arcsec, deg]
    pixel_area = pixel2arcsec_ra * pixel2arcsec_dec # in arcsec2
    # convert the beamsize into pixel area
    beamsize = calculate_beamsize(cube_beams, debug=debug) / pixel_area 

    # convert the units
    cube_data = cube_data / beamsize[:,None,None] * u.Jy

    return Cube(data=cube_data, header=cube_header, beams=cube_beams)


def calculate_beamsize(beams, debug=False):
    """quick function to calculate the beams size
    
    Params:
        beams (list, ndarray): it can be one dimension [bmaj, bmin, bpa] or two dimension, like [[bmaj,bmin,bpa],[bmaj,bmin,bpa],]
    """
    ndim = np.ndim(beams)
    if debug:
        print(f"func::calculate_beamsize --> dimension of beams is {ndim} {beams.shape}")
    if ndim == 1:
        return 1/(np.log(2)*4.0) * np.pi * beams[0] * beams[1]
    if ndim == 2:
        return 1/(np.log(2)*4.0) * np.pi * beams[:,0] * beams[:,1]


def convert_spec(spec_in, unit_out, reffreq=None, refwave=None, mode='radio'):
    """convert the different spectral axis

    Params:
        spec_in (astropy.Quantity): any valid spectral data with units
        unit_out (str or astropy.Unit): valid units for output spectral axis
        reffreq: reference frequency, with units
        refwave: reference wavelength, with units
    """
    if not isinstance(unit_out, u.Unit):
        unit_out = u.Unit(unit_out)
    if spec_in.unit.is_equivalent(unit_out):
        return spec_in.to(unit_out)
    if reffreq is not None:
        refwave = (const.c/reffreq).to(u.um)
    elif refwave is not None:
        reffreq = (const.c/refwave).to(u.GHz)
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

def integrate_spectrum(x, y):
    """calculate the area covered by the spectrum

    Params:
        x (ndarray): the x axis of the curve, with units
        y (ndarray): the y axis of the curve, with units
    """
    return np.sum(calculate_diff(x)*y)


