# -*- coding: utf-8 -*-
"""Utilities dealing with fits datacube

Author: Jianhang Chen, cjhastro@gmail.com
History:
    2022-02-16: first release to handle the datacube from ALMA

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
import warnings
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

from photutils import aperture_photometry, find_peaks, EllipticalAperture, RectangularAperture, SkyEllipticalAperture
from photutils import PixelAperture, SkyAperture
from astropy.modeling import models, fitting



##############################
######### Cube ###############
##############################
class Cube(object):
    """The base data strcuture to handle 3D astronomy data
    """
    def __init__(self, data=None, header=None, wcs=None, chandata=None, beams=None, name=None):
        """initialize the cube

        Args:
            data: the 3D data with/without units (equivalent to Jy)
            header: the header can be identified by WCS
            wcs: it can be an alternative to the header
            specdata: the data in the third dimension, it can be velocity, frequency, wavelength. 
                It should have units
            beams: the beam shapes the units of [arcsec, arcsec, deg]. 
                It can be one dimension or two dimension (for each channel)
        """
        self.data = data
        self.header = header
        self.wcs = wcs
        self.chandata = chandata
        self.beams = beams
        self.name = name
        if self.header is not None:
            if self.wcs is None:
                self.update_wcs()
            if self.chandata is None:
                self.update_chandata()
        if self.header is None:
            if self.wcs is not None:
                self.header = self.wcs.to_header()
                if self.chandata is None:
                    self.update_chandata()
        if self.wcs.naxis > 3:
            self.wcs = self.wcs.sub(['longitude','latitude','spectral'])

    def __getitem__(self, i):
            # return self.data[i]
            return self.subcube(i)
    @property
    def info(self):
        # the shortcut to print basic info
        print(f"The shape of the data: {self.cube.shape}")
        if self.beams is not None:
            print(f"The median beam [arcsec, arcsec, deg]: {np.median(self.beams, axis=0)}")
        print(f"Pixel size: {self.pixel_sizes}")
        print(f"Channel width: {np.median(self.dchan)}")
    @property
    def pixel_sizes(self):
        if self.header is not None:
            # Return the pixel size encoded in the header
            # In casa, it is in the units of deg, thus the returned value is pixel to deg
            pixel2arcsec_ra = (abs(self.header['CDELT1'])*u.Unit(self.header['CUNIT1'])).to(u.arcsec)
            pixel2arcsec_dec = (abs(self.header['CDELT2'])*u.Unit(self.header['CUNIT2'])).to(u.arcsec)
        else:
            pixel2arcsec_ra, pixel2arcsec_dec = 1, 1
        return [pixel2arcsec_ra, pixel2arcsec_dec]
    @property
    def shape(self):
        return self.cube.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def imagesize(self):
        # the imagesize is in [ysize, xsize], [nrow, ncol]
        try: 
            return self.data.shape[-2:]
        except: 
            return None
    @property
    def nchan(self):
        try: return self.data.shape[-3]
        except: return None
    @property
    def median_beam(self):
        if self.beam is not None:
            return np.median(self.beams, axis=0)
        else:
            print("Warning: no beam has been found!")
    @property
    def dchan(self):
        return calculate_diff(self.chandata)
    @property
    def cube(self):
        # return the 3D cube, with stokes dimension
        if self.ndim == 4:
            return self.data[0]
        else:
            return self.data
    @property
    def wavelength(self):
        return self.get_wavelength()
    @property
    def frequency(self):
        return self.get_frequency()

    def get_wavelength(self, unit=u.mm):
        return convert_spec(self.chandata, unit)

    def get_frequency(self, unit=u.GHz):
        return convert_spec(self.chandata, unit)

    def get_velocity(self, reffreq=None, refwave=None):
        return convert_spec(self.chandata, 'km/s', refwave=refwave, reffreq=reffreq)

    def update_wcs(self, wcs=None):
        if wcs is not None:
            self.wcs = wcs
        else: 
            self.wcs = WCS(self.header)

    def update_chandata(self):
        # the fits standard start with index 1
        # because wcs.slice with change the reference channel, the generated ndarray should start with 1 
        self.chandata = (self.header['CRVAL3'] + (np.arange(1, self.header['NAXIS3']+1)-self.header['CRPIX3']) * self.header['CDELT3'])*u.Unit(self.header['CUNIT3'])

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

    def convert_chandata(self, unit_out, refwave=None, reffreq=None):
        """convert the units of the specdata 
        """
        if refwave is not None:
            self.refwave = refwave
        if reffreq is not None:
            self.reffreq = reffreq
        self.chandata = convert_spec(self.chandata, unit_out, refwave=refwave, reffreq=reffreq)

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
                spectrum = np.sum(np.sum(self.cube, axis=1), axis=1)

        if pixel_coord is not None:
            if pixel_aperture is not None:
                aperture = EllipticalAperture(pixel_coord, *pixel_aperture)
                aper_mask = aperture.to_mask().to_image(self.imagesize).astype(bool)
                aper_mask_3D = np.repeat(aper_mask[None,:,:], self.nchan, axis=0)
                data_selected = self.cube[aper_mask_3D]
                spectrum = np.sum(data_selected.reshape((self.nchan, np.sum(aper_mask))), axis=1)
            else:
                spectrum = self.cube[:,pixel_coord[1], pixel_coord[0]]
            
        if plot:
            if ax is None:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111)
            ax.step(self.chandata, spectrum, where='mid')
            ax.set_xlabel(self.chandata.unit)
            ax.set_ylabel('Flux [Jy]')
            ax_top = ax.twiny()
            ax_top.step(np.arange(self.nchan), spectrum, where='mid', alpha=0)
        return Spectrum(specchan=self.chandata, specdata=spectrum, **kwargs)

    def collapse_cube(self, chans='', chanunit=None, return_header=False, 
                      reffreq=None, refwave=None, moment=0):
        """collapse the cube to make one channel image

        Args:
            chans (str): follow the format of casa, like "20~30;45~90"

        Return:
            class::Image
        TODO: return the averaged beam shape
        """
        # decoding the chans
        selected_indices = []
        data = self.cube
        if chanunit is not None:
            self.convert_chandata(chanunit, reffreq=reffreq, refwave=refwave)
        if chans == '':
            selected_indices = list(range(self.chandata.size))
        else:
            chan_par = re.compile(r"(\d+)~(\d+)")
            chan_ranges = np.array(chan_par.findall(chans)).astype(int)
            for chan_range in chan_ranges:
                # casa includes the last channel with the chan selection rules
                selected_indices.append(list(range(chan_range[0], chan_range[1]+1))) 
            selected_indices = [item for sublist in selected_indices for item in sublist]
        dchan_selected = calculate_diff(self.chandata)[selected_indices]
        chan_selected = self.chandata[selected_indices]
        data_selected = data[selected_indices]
        M0 = np.sum(data_selected * dchan_selected[:,None,None], axis=0)
        if moment == 0:
            collapsed_image = M0
        if moment > 0:
            M1 = np.sum(data_selected*chan_selected[:,None,None]*dchan_selected[:,None,None], axis=0)/M0
            if moment == 1:
                collapsed_image = M1
            if moment > 1:
                collapsed_image = np.sum(data_selected*(chan_selected[:,None,None]-M1[None,:,:])**moment*dchan_selected[:,None,None], axis=0)/M0
        if self.beams is not None:
            collapsed_beam = np.median(self.beams[selected_indices], axis=0)
        else: collapsed_beam = None
        collapsed_header = self.wcs.sub(['longitude','latitude']).to_header()
        return [collapsed_image, collapsed_header, collapsed_beam]

    def subcube(self, s_):
        """extract the subsection of the datacube

        Args:
            s_ (object:slice): the slice object, should be three dimension

        Return:
            Cube
        """
        cube_sliced = self.cube[s_].copy() # force to use new memory
        wcs_sliced = self.wcs[s_]
        shape_sliced = cube_sliced.shape
        header_sliced = wcs_sliced.to_header()
        header_sliced.set('NAXIS',len(shape_sliced))
        header_sliced.set('NAXIS1',shape_sliced[2])
        header_sliced.set('NAXIS2',shape_sliced[1])
        header_sliced.set('NAXIS3',shape_sliced[0])
        if self.beams is not None:
            beams_sliced = self.beams[s_[0]]
        else: 
            beams_sliced = None
        return Cube(data=cube_sliced, header=header_sliced, beams=beams_sliced)

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
        header = self.header
        if header is None:
            if self.wcs is not None:
                header = wcs.to_header()
            else:
                header = fits.Header()
        # remove the History and comments of the header
        header.remove('HISTORY', remove_all=True)
        header.remove('COMMENT', remove_all=True)
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
        header.update({'history':'created by cube_tools.Cube',})
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

    def auto_measure(self):
        """designed for subcube

        With proper cutout, it should be used to extract the spectrum and measure the intensity
        automatically
        """
        pass
    @staticmethod
    def read(fitscube, debug=False):
        """read general fits file
        """
        with fits.open(fitscube) as cube_hdu:
            if debug:
                print(cube_hdu.info())
            cube_header = cube_hdu['primary'].header
            cube_data = cube_hdu['primary'].data * u.Unit(cube_header['BUNIT'])
            try:
                beams_data = cube_hdu['beams'].data
                cube_beams = np.array([beams_data['BMAJ'], beams_data['BMIN'], beams_data['BPA']]).T
            except: cube_beams = None
        return Cube(data=cube_data, header=cube_header, beams=cube_beams)
    @staticmethod
    def read_ALMA(fitscube, debug=False, stokes=0, name=None):
        """read the fits file
        """
        with fits.open(fitscube) as cube_hdu:
            if debug:
                print(cube_hdu.info())
            cube_header = cube_hdu['primary'].header
            cube_data = cube_hdu['primary'].data * u.Unit(cube_header['BUNIT'])
            try:
                beams_data = cube_hdu['beams'].data
                cube_beams = np.array([beams_data['BMAJ'], beams_data['BMIN'], beams_data['BPA']]).T
            except: cube_beams = None
            if False: # convert the units
                if '/beam' in cube_header['BUNIT']: # check whether it is beam^-1
                    pixel2arcsec_ra = abs(cube_header['CDELT1']*u.Unit(cube_header['CUNIT1']).to(u.arcsec)) 
                    pixel2arcsec_dec = abs(cube_header['CDELT2']*u.Unit(cube_header['CUNIT2']).to(u.arcsec))       
                    pixel_area = pixel2arcsec_ra * pixel2arcsec_dec # in arcsec2
                    beamsize = calculate_beamsize(cube_beams, debug=debug) / pixel_area 
                    cube_data = cube_data / beamsize[None,:,None,None] * u.beam
        return Cube(data=cube_data[stokes], header=cube_header, beams=cube_beams, name=name)

##############################
######### Spectrum ###########
##############################
class Spectrum(object):
    """the data structure to handle spectrum
    """
    def __init__(self, specchan=None, specdata=None, reffreq=None, refwave=None, z=None):
        """ initialize the spectrum
        """
        self.specchan = specchan
        self.specdata = specdata
        self.reffreq = reffreq
        self.refwave = refwave
        if self.reffreq is None:
            if self.refwave is not None:
                self.reffreq = (const.c/self.refwave).to(u.GHz)
        if self.refwave is None:
            if self.reffreq is not None:
                self.refwave = (const.c/self.reffreq).to(u.um)
        if self.reffreq is not None:
            self.specvel = self.velocity(reffreq=reffreq)
        else:
            self.specvel = None
        if z is not None:
            self.to_restframe(z)
    @property
    def channels(self):
        return list(range(len(self.specchan)))
    @property
    def unit(self):
        return self.specchan.unit

    def wavelength(self, units=u.um):
        return convert_spec(self.specchan, units)

    def frequency(self, units=u.GHz):
        return convert_spec(self.specchan, units)

    def to_restframe(self, z):
        self.restfreq = convert_spec(self.specchan, 'GHz')*(z+1)
        self.restwave = convert_spec(self.specchan, 'um')/(z+1)

    def velocity(self, reffreq=None, refwave=None):
        if reffreq is None:
            reffreq = self.reffreq
        if refwave is None:
            refwave = self.refwave
        return convert_spec(self.specchan, 'km/s', refwave=refwave, reffreq=reffreq)

    def convert_specchan(self, unit_out, refwave=None, reffreq=None):
        """convert the units of the specdata 
        """
        if refwave is not None:
            self.refwave = refwave
        if reffreq is not None:
            self.reffreq = reffreq
        self.specchan = convert_spec(self.specchan, unit_out, refwave=refwave, reffreq=reffreq)

    def to_channel(self, values):
        #convert the spectral axis
        if values.unit.is_equivalent('km/s'): # velocity to channel
            return array_mapping(values, self.velocity(), self.channels)
        elif values.unit.is_equivalent('um'): # velocity to channel
            return array_mapping(values, self.wavelength(values.unit), self.channels)
        elif values.unit.is_equivalent('GHz'): # velocity to channel
            return array_mapping(values, self.frequency(values.unit), self.channels)

    def integral(self):
        """integrate the spectrum
        """
        return integral_spectrum(self.specvel, self.specdata)

    def plot(self, ax=None, **kwargs):
        plot_spectra(self.specchan, self.specdata, ax=ax, **kwargs)

    def fit_gaussian(self, plot=False, ax=None, **kwargs):
        """fitting the single gaussian to the spectrum
        """
        fit_p = fitting.LevMarLSQFitter()
        spec_selection = self.specdata > np.percentile(self.specdata, 85)
        amp0 = np.mean(self.specdata[spec_selection])
        if self.specvel is not None:
            # fit a gaussian at the vel=0
            ## central velocity
            vel0 = np.median(self.specvel[spec_selection])
            p_init = models.Gaussian1D(amplitude=amp0, mean=vel0, 
                                       stddev=50*u.km/u.s)
                    # bounds={"mean":np.array([-1000,1000])*u.km/u.s, 
                            # "stddev":np.array([0., 1000])*u.km/u.s,})
            specchan = self.specvel
        else:
            # will try to fit a gaussian in the centre of the spectrum
            mean0 = np.median(self.specchan[spec_selection])
            p_init = models.Gaussian1D(amplitude=amp0, mean=mean0, 
                                       stddev=5*np.median(np.diff(self.specchan)))
            specchan = self.specchan
        p = fit_p(p_init, specchan, self.specdata)
        specfit = p(self.specchan)
        if plot:
            if ax is None:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111)
                ax.step(specchan, self.specdata, label='data')
                ax.plot(specchan, specfit, label='mode')
        fitobj = Fitspectrum()
        fitobj.fitparams = p.param_sets.flatten()
        fitobj.fitnames = p.param_names
        fitobj.bestfit = specfit
        fitobj.specchan = specchan
        fitobj.specdata = self.specdata
        fitobj.fitfunc = p
        fitobj.chi2 = None #TODO #np.sum((bestfit - self.specdata)**2/std**2)
        return fitobj


class Fitspectrum():
    """the data structure to store the fitted spectrum
    """
    def __init__(self):
        """
        Args:
            fitparams: the fitted parameters
            bestfit: the best fit
            specchan: the spectrum axis
            specdata: the data axis
            chi2: the fitted chi2
        """
        self.fitparams = None
        self.fitnames = None
        self.bestfit = None
        self.fitfunc = None
        self.specchan = None
        self.specdata = None
        self.chi2 = None
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.step(self.specchan, self.specdata, where='mid', color='k')
        ax.step(self.specchan, self.bestfit, where='mid', color='red', alpha=0.8)

#################################
###   stand alone functions   ###
#################################
def plot_spectra(specchan, specdata, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.step(specchan, specdata, where='mid', **kwargs)

def read_fitscube_ALMA(fitscube, pbcor=False, debug=False,):
    """read the datacube from the files and return with the Cube
    """
    with fits.open(fitscube) as cube_hdu:
        if debug:
            print(cube_hdu.info())
        cube_header = cube_hdu['primary'].header
        cube_data = cube_hdu['primary'].data * u.Unit(cube_header['BUNIT'])
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
    cube_data = cube_data / beamsize[None,:,None,None] * u.beam

    return Cube(data=cube_data, header=cube_header, beams=cube_beams)

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

def convert_spec(spec_in, unit_out, reffreq=None, refwave=None, mode='radio'):
    """convert the different spectral axis

    Args:
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

    Args:
        x (ndarray): the x axis of the curve, with units
        y (ndarray): the y axis of the curve, with units
    """
    return np.sum(calculate_diff(x)*y)

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

def find_3Dstructure(data, sigma=2.0, std=None, minsize=9, opening_iters=1, dilation_iters=1, mask=None, plot=False, ax=None, debug=False):
    """this function use scipy.ndimage.label to search for continuous structures in the datacube

    To use this function, the image should not be oversampled! If so, increase the minsize is recommended

    #TODO: ad support for bianry_opening and binary_dilation
    """
    s3d = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[1,1,1],[0,1,0]], 
                            [[0,0,0],[0,1,0],[0,0,0]]])
    if std is None:
        mean, median, std = sigma_clipped_stats(data, sigma=sigma, maxiters=1, mask=mask)
    sigma_struct = data > sigma *  std

    if opening_iters > 0:
        sigma_struct = ndimage.binary_opening(sigma_struct, iterations=opening_iters, 
                                              structure=s3d)
    if dilation_iters > 0:
        sigma_struct = ndimage.binary_dilation(sigma_struct, iterations=dilation_iters,
                                               structure=s3d)
    labels, nb = ndimage.label(sigma_struct, structure=s3d)
    if debug:
        print('nb=', nb)
    for i in range(nb):
        struct_select = ndimage.find_objects(labels==i)
        if debug:
            print("structure size: {}".format(sigma_struct[struct_select[0]].size))
        if sigma_struct[struct_select[0]].size < minsize:
            sigma_struct[struct_select[0]] = False
    if plot:
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(sigma_struct)
        ax.set(xlabel='channel', ylabel='y', zlabel='x')
    return sigma_struct

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

def array_mapping(vals1, array1, array2):
    """mapping the values from array2 at the position of vals1 relative to array1
    """
    return (vals1 - array1[0])/(array1[-1]- array1[0]) * (array2[-1]-array2[0])
