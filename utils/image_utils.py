#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""A minimalist tool to deal with fits images

Author: Jianhang Chen, cjhastro@gmail.com

Notes:
    1. Throughout the code, the aperture are follow the `photoutils`
    2. The beam follow the rule of ALMA: [bmaj, bmin, theta], theta is in 
       degree, from north (top) to east (left).

Requirement:
    numpy
    matplotlib
    astropy >= 5.0
    photutils >= 1.0
    sep (optional, used for source_finder)

History:
    2022-01-11: first release to handle fits images from CASA, v0.1, Garching
    2024-02-16: divide into BasicImage and Image class, v0.2, IRAM30, Granada
    2024-06-14: add test, v0.3, MPE, Garching

"""

__version__ = '0.3.1'

import os
import sys
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from scipy import optimize
import warnings
from astropy.io import fits
from astropy.table import Table, vstack, hstack
from astropy import stats
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants as const
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.modeling import models, fitting
from astropy import convolution 

# Filtering warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

# optional library
from photutils.aperture import (aperture_photometry, EllipticalAperture, 
                                RectangularAperture, SkyEllipticalAperture, 
                                EllipticalAnnulus, PixelAperture)
from photutils.detection import find_peaks
from skimage.segmentation import watershed, random_walker

try: import petrofit; has_petrofit=True
except: has_petrofit = False
try: import reproject; has_repoject=True
except: has_repoject = False

try: import vorbin; has_vorbin=True
except: has_vorbin = False

##############################
######### Image ##############
##############################

class BaseImage(object):
    """The data strcuture to handle the 2D astronomical image

    """
    def __init__(self, data=None, name=None, mask=None, unit=None, 
                 header=None, beam=None, wcs=None,
                ):
        """initialize the image

        Args:
            data: the image data, with or without units
            name: the name of the image, used for plot and file handling
            mask: the mask of the data, True value to be exclude
            header: the header can be identified by astropy.wcs.WCS
            beam: the beam shape in the units of [arcsec, arcsec, deg]
            wcs: another way to set celestial reference
        """
        if isinstance(data, u.Quantity):
            self.data = data.value
            unit = data.unit
        if isinstance(data, np.ma.MaskedArray):
            self.data = data.data
            mask = data.mask
        else:
            self.data = data
        self.mask = mask
        if mask is not None:
            if mask.shape != self.data.shape:
                #raise ValueError("unmatched mask and data!")
                print("Warning: unmatched mask and data! Drop mask.")
                self.mask = None
            else:
                self.mask = mask
        self.name = name 
        self._unit = unit
        # hide the orinal header and wcs, due to their interchangeability
        self._header = header
        self._wcs = wcs
        # keep the original beam
        self._beam = beam
        # create a header if not exists
        if self.header is None:
            self._header = fits.Header()
        if self._header is None:
            self._header = self.header
    # def __getitem__(self, i):
            # return self.data[i]
    @property
    def info(self):
        # shortcut for print all the basic info
        print(f"Shape: {self.shape}")
        print(f"Units: {self.image.unit}")
        print(f"Pixel size {self.pixel_sizes}")
        print(f"Beam {self.beam}")
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
        if (fullwcs is not None) and (fullwcs.naxis > 3):
            # drop the stokes axis
            return fullwcs.sub(['longitude','latitude'])
        return fullwcs
    @wcs.setter
    def wcs(self, wcs):
        self._wcs = wcs
    @property
    def unit(self):
        if self._unit is not None:
            return u.Unit(self._unit)
        elif self.header is not None:
            try:
                return u.Unit(self._header['BUNIT'])
            except:
                pass
        return u.Unit('')
    @unit.setter
    def unit(self, unit):
        if self.unit != u.Unit(''):
            # print('unit converting factor"', self.unit.to(unit))
            self.data = self.data * self.unit.to(unit)
            self._unit = u.Unit(unit)
            self._header.update({'BUNIT': self._unit.to_string()})
        else:
            self._unit = u.Unit(unit)
    @property
    def image(self):
        """always return the 2D image with units

        Using self.data to access the original data
        """
        return self.data*self.unit
    @property
    def beam(self):
        # return beam shape in [bmaj, bmin, PA] in [arcsec, arcsec, degree]
        if self._beam is not None:
            return self._beam
        if self.header is not None:
            try:
                header = self.header
                header_beam = np.array([header['BMAJ'], header['BMIN'], header['BPA']]).T
                image_beam = header_beam * np.array([3600.,3600.,1]) # convert deg to arcsec
                return image_beam
            except: pass
        return None
    @beam.setter
    def beam(self, beam):
        """the beam should be a 1D array, like [bmaj, bmin, PA]
        the units should be [arcsec, arcsec, degree]
        """
        self._beam = beam
        header_beams = np.array(beam) * np.array([1/3600., 1/3600, 1]) # convert arcsec to deg
        self._header.update({'BMAJ':header_beams[0], 'BMIN':header_beams[1], 
                             'BPA':header_beams[2]})
    @property
    def pixel_sizes(self):
        if self.header is not None:
            # Return the pixel size encoded in the header. 
            # In casa, it is in the units of deg, thus the returned value 
            # is pixel to deg
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
        return u.Quantity([pixel2arcsec_ra, pixel2arcsec_dec])
    @pixel_sizes.setter
    def pixel_sizes(self, size):
        """ the size can be a list with two values for ra and dec or a single
        value for both. The default units should be [arcsec, arcsec]
        """
        if isinstance(size, u.Quantity):
            size = size.value
            sunit = size.unit
        else:
            sunit = u.Unit('arcsec')
        if not isinstance(size, [tuple, list, np.ndarray]):
            size = [size, size]
        self._header.update({'CDELT1':size[0], 'CDELT2':size[1], 
                             'CUNIT1':sunit.to_string(), 'CUNIT2':sunit.to_string()})
    @property
    def shape(self):
        return self.data.shape
    @property
    def beamarea(self):
        # calculate the beamszie in number of pixels
        # the final unit is pixels/beam, convert the Jy/beam to Jy/pixel need to divive the beam
        pixel_sizes = self.pixel_sizes
        beam = self.beam
        pixel_area = pixel_sizes[0].to('arcsec').value * pixel_sizes[1].to('arcsec').value
        return 1/(np.log(2)*4.0) * np.pi * beam[0] * beam[1] / pixel_area 
        # return calculate_beamarea(self.beam, scale=1/pixel_area)
    @property
    def pixel_beam(self):
        # convert the beam size into pixel sizes
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

    def pixel2skycoords(self, pixel_coords):
        """covert from pixel to skycoords

        Args:
            pixel_coords: pixel coordinates, in the format of 
                          [[x1,y1], [x2,y2],...]
        """
        pixel_coords = np.array(pixel_coords).T
        return pixel_to_skycoord(*pixel_coords, self.wcs)

    def skycoords2pixels(self, skycoords):
        if skycoords.size == 1:
            return np.array(skycoord_to_pixel(skycoords, self.wcs))
        else:
            return np.array(list(zip(*skycoord_to_pixel(skycoords, self.wcs))))

    def subimage(self, s_, shift_reference=False, compatible_mode=False):
        """extract subimage from the orginal image

        Args:
            s_ (:obj:`slice`): data slice, same shape as image

        Return:
            :obj:`Image`
        """
        image_sliced = self.data[s_].copy()
        shape_sliced = image_sliced.shape
        wcs_sliced = self.wcs[s_].deepcopy()
        if shift_reference:
            # TODO: temperary solution, more general solution needs to consider the 
            # shift the reference center, 
            wcs_sliced.wcs.crpix[0] = (s_[-1].start + s_[-1].stop)//2
            wcs_sliced.wcs.crpix[1] = (s_[-2].start + s_[-2].stop)//2
            try:
                wcs_sliced.wcs.crval = (self.wcs.wcs.crval + self.wcs.wcs.pc.dot(wcs_sliced.wcs.crpix - self.wcs.wcs.crpix))# * self.wcs.wcs.cdelt)
            except:
                pass
            try:
                wcs_sliced.wcs.crval = (self.wcs.wcs.crval + self.wcs.wcs.cd.dot(wcs_sliced.wcs.crpix - self.wcs.wcs.crpix))# * self.wcs.wcs.cdelt))
            except:
                raise ValueError("Invalid coordinate shifts system (defined either by CD and PC)")
        # wcs_sliced.wcs.crpix[0] -= s_[-1].start 
        # wcs_sliced.wcs.crpix[1] -= s_[-2].start
        header_sliced = wcs_sliced.to_header()
        naxis = len(shape_sliced)
        header_sliced.set('NAXIS',naxis)
        header_sliced.set('NAXIS1',shape_sliced[-1])
        header_sliced.set('NAXIS2',shape_sliced[-2])
        if compatible_mode:
            # make it compatible for older fits imager 
            header_keys = list(header_sliced.keys())
            if 'PC1_1' in header_keys:
                header_sliced.set('CD1_1', header_sliced['PC1_1'])
                header_sliced.set('CD2_2', header_sliced['PC2_2'])
            # header_sliced.set('CDELT1', header_sliced['CD1_1'])
            # header_sliced.set('CDELT2', header_sliced['CD2_2'])
        try:
            header_sliced['BUNIT'] = self.header['BUNIT']
        except:
            pass
        return BaseImage(data=image_sliced, header=header_sliced, beam=self.beam)

    def writefits(self, filename, overwrite=False, shift_reference=False):
        """write to fits file
        
        This function also shifts the reference pixel of the image to the image center
        """
        ysize, xsize = self.shape
        header = self.header
        if self._beam is not None:
            self.beam = self._beam
        if shift_reference:
            try: # shift the image reference pixel, tolerence is 4 pixels
                if header['CRPIX1'] < (xsize//2-4) or header['CRPIX1'] > (xsize//2+4):
                    header['CRVAL1'] += (1 - header['CRPIX1']) * header['CDELT1']
                    header['CRPIX1'] = xsize//2
                    print('Warning: automatically shifted the x axis reference.')
                if header['CRPIX2'] < (ysize//-4) or header['CRPIX2'] > (ysize//2+4):
                    header['CRVAL2'] += (1 - header['CRPIX2']) * header['CDELT2']
                    header['CRPIX2'] = ysize//2
                    print('Warning: automatically shifted the y axis reference.')
            except: pass
        imagehdu = fits.PrimaryHDU(data=self.data, header=header)
        header.update({'history':'created by image_utils.Image',})
        imagehdu.writeto(filename, overwrite=overwrite)

    def readfits(self, fitsimage, extname='primary', name=None, debug=False, 
                 correct_beam=False, spec_idx=0, stokes_idx=0):
        """read the fits file

        Parameters
        ----------
        fitsimage : str
            the filename of the fits file.
        name : str, optinal
            the name of the image.
        debug : bool 
            set to true to print the details of the fits file
        correct_beam : bool 
            set to True to correct the beams if the beam 
            information is available
        spec_idx : int
            the index of the select channel along the spectral dimension, 
            default to be the median value
        stokes_idx : int
            the index of the select stocks dimension, default to be the first axis
            
        """
        with fits.open(fitsimage) as image_hdu:
            if debug:
                print(image_hdu.info())
            image_header = image_hdu[extname].header
            ndim = image_header['NAXIS']
            if ndim > 4 or ndim < 2:
                raise ValueError("Unsupported data dimension! Only support 2D, 3D and 4D data!")
            image_data = image_hdu[extname].data
            if ndim > 3:
                image_data = image_data[stokes_idx]
            if ndim > 2:
                image_data = image_data[spec_idx]
            image_unit = ''
            if 'BUNIT' in image_header.keys():
                try:
                    image_unit = u.Unit(image_header['BUNIT'])
                except:
                    print(f"Warning: failed in interpreting the unit {image_header['BUNIT']}")
            # try to read the beam from the headers
            if 'BMAJ' in image_header.keys():
                image_beam = [image_header['BMAJ']*3600., image_header['BMIN']*3600., 
                              image_header['BPA']]
            else:
                try: full_beam = image_hdu['BEAMS'].data
                except: full_beam = None
                if full_beam is not None:
                    if ndim > 3:
                        full_beam = full_beam[stokes_idx]
                    if ndim > 2:
                        # image_beam = np.median(full_beam, axis=0)
                        image_beam = full_beam[spec_idx]
                else: image_beam = None
        if name is None:
            name = os.path.basename(fitsimage)
        self.data = image_data
        self.unit = image_unit
        self._header = image_header
        self._beam = image_beam
        self.name = name

class Image(BaseImage):
    """The image used for various handy analysis
    """
    def __init__(self, data=None, name=None, mask=None, unit=None,
                 header=None, beam=None, wcs=None, 
                 ):
        """initialize the image
        """
        if data is not None:
            if isinstance(data, (BaseImage, Image)):
                # copy the Image
                super().__init__(data=data.data, name=data.name, mask=data.mask, 
                                 header=data.header, beam=data.beam, unit=data.unit,
                                 wcs=data.wcs)
            else:
                super().__init__(data=data, header=header, beam=beam, wcs=wcs, unit=unit, 
                                 mask=mask, name=name)
        else:
            super().__init__()

    def __getitem__(self, s):
        return Image(self.subimage(s))

    def subimage(self, s):
        return Image(super().subimage(s))

    def imstats(self, sigma=5.0, maxiters=2, sigma_clip=False):
        """this function first mask the 4-sigma signal and expand the mask with
        scipy.ndimage.binary_dilation
        """
        if sigma_clip:
            return stats.sigma_clipped_stats(self.image, sigma=sigma, maxiters=maxiters)
        else:
            signal_mask = self.find_structure(sigma=4.0, dilation_iters=3)
            return (np.nanmean(self.image[~signal_mask]), 
                    np.nanmedian(self.image[~signal_mask]),
                    np.nanstd(self.image[~signal_mask]))

    def update_mask(self, mask=None, mask_invalid=True):
        newmask = np.zeros(self.imagesize, dtype=bool)
        if mask_invalid:
            image_masked = np.ma.masked_invalid(self.data)
            invalid_mask = image_masked.mask.reshape(self.imagesize)
            newmask = newmask | invalid_mask
        if mask is not None:
            mask = newmask | mask
        self.mask = mask

    def plot(self, mode='default', color_scales=[-1,10], fig=None, ax=None,
             figsize=(8,6), fontsize=14, figname=None, vmax=None, vmin=None, 
             **kwargs):
        """
        Args:
            mode: the format of the plot, can be ['default', 'pixel', 'sky']
                  "default": show the difference in arcsec relative to the center 
                  "pixel": pixel coordinate
                  "sky": with full ra and dec as the x and y axis
        """
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        mean, median, std = self.imstats()
        if vmin is None:
           vmin = color_scales[0]*std.value 
        if vmax is None:
           vmax = color_scales[1]*std.value

        if mode == 'default':
            plot_image(self.image.value, beam=self.beam, ax=ax,
                       pixel_size=self.pixel_sizes.value,
                       vmin=vmin, vmax=vmax,
                       fontsize=fontsize, **kwargs,
                       )

        if mode == 'pixel':
            plot_pixel_image(self.image.value, beam=self.pixel_beam, ax=ax,
                             vmin=vmin, vmax=vmax,
                             fontsize=fontsize, **kwargs,
                             )
        if mode == 'sky':
            ax.remove()
            ax = plot_sky_image(self.image.value, header=self.header, beam=self.beam,
                                fig=fig, vmin=vmin, vmax=vmax,
                                fontsize=fontsize, **kwargs,
                                )
        if figname is not None:
            fig.savefig(figname, bbox_inches='tight', dpi=200)
            plt.close(fig)
        else:
            return fig, ax

    def reproject(self, wcs_out=None, header_out=None, shape_out=None, **kwargs):
        """reproject the data into another wcs system
        
        """
        try: import reproject
        except: raise ValueError("Package `reproject` must be installed to handle the reprojection!")
        if header_out is not None:
            data_new, footprint = reproject.reproject_adaptive(
                                    (self.image.value, self.wcs.celestial), 
                                    header_out, return_footprint=True,
                                    conserve_flux=True, **kwargs)
        elif wcs_out is not None:
            if shape_out is None:
                try:
                    shape_out = wcs_out.array_shape
                except:
                    raise ValueError('the `shape_out` must be provide when use projection with WCS!')
            data_new, footprint = reproject.reproject_adaptive(
                                    (self.image.value, self.wcs.celestial), wcs_out, 
                                    shape_out=shape_out, return_footprint=True, 
                                    conserve_flux=True, **kwargs)
        else:
            raise ValueError("Please specify the output referece system, either the header_out or the wcs_out")
        return Image(data=data_new, header=header_out, wcs=wcs_out)
 
    def beam_stats(self, beam=None, mask=None, nsample=100):
        if beam is None:
            beam = self.pixel_beam
        if mask is None:
            mask = self.mask
        aperture = beam2aperture(beam)
        return aperture_stats(self.image, aperture=aperture, 
                              mask=mask, nsample=nsample)

    def beam2aperture(self, scale=1):
        """shortcut to convert beam to apertures

        The major difference is angle. In beam, it is degree; while in aperture it is radian.
        """
        return beam2aperture(self.pixel_beam, scale=scale)

    def correct_beam(self, inverse=False):
        """correct the flux per beam to per pixel
        """
        if self.beam is not None:
            if inverse:
                if 'beam' not in self.unit.to_string():
                    self.data = self.data*self.beamarea
                    self.unit = u.Unit(self.unit.to_string()+'/beam')
                else:
                    pass
            else:
                if 'beam' in self.unit.to_string():
                    self.data = self.data/self.beamarea
                    self._unit = self.unit*u.beam

    def correct_reponse(self, corr):
        """apply correction for the whole map
        """
        self.data = self.data * corr

    def find_structure(self, sigma=3.0, iterations=1, opening_iters=None, 
                       dilation_iters=None, plot=False, **kwargs):
        """find structures above a centain sigma

        Firstly, this function will find the 3sigma large structure;
        after that, it applys the binary opening to remove small structures
        then, binary dilation is used to exclude all the all the faint 
        extended emission
        This function can be used for find extended emission regions and generate
        additional masking

        Args:
            sigma: the sigma level to start the structure searching
            iterations: the opening and dilation iterations
            opening_iters: specifically set the opening iterations
            dilation_iters: specifically set the dilation iterations
            plot: plot the marked regions.

        Return:
            image: binary image with masked region with 1 and the others 0
        """
        sigma_structure = find_structure(self.image.value, sigma=sigma, mask=self.mask)
        return sigma_structure

    def source_finder(self, **kwargs):
        """class wrapper for source_finder
        """
        mean, median, std = self.imstats()
        return source_finder(self.image.value, wcs=self.wcs, std=std.value, name=self.name, 
                             **kwargs)

    def measure_flux(self, dets=None, coords=None, apertures=None, 
                     aperture_scale=4.0, segment_scale=3.0,
                     method='adaptive-aperture', minimal_aperture=None, **kwargs):
        """shortcut for flux measurement of existing detections

        Args:
            dets (`astropy.table`): the table includes all the detections
                required informations are:
                name  x   y   a   b  theta
                unit pix pix pix pix radian
            coords: the pixel coordinates of the detections
            apertures: the aperture used to measure the flux, follow the syntax of photoutils
            aperture_scale: the scale factor of the shape parameters (a and b in dets)

        Example:
            
            Image.measure_flux(dets=dets, minimal_aperture=im1.beam2aperture(scale=2), 
                               method='single-aperture')

        TODO: add support for aperture support
        """
        if dets is not None:
            if ('x' in dets.colnames) and ('y' in dets.colnames):
                coords=list(zip(dets['x'], dets['y']))
            if ('a' in dets.colnames) and ('b' in dets.colnames):
                apertures=list(zip(aperture_scale*dets['a'], 
                                   aperture_scale*dets['b'], 
                                   dets['theta'])) 
        corr = 1.0
        if method == 'single-aperture':
            if minimal_aperture is not None:
                for i in range(len(apertures)):
                    aper = apertures[i]
                    if aper[0]*aper[1] < minimal_aperture[0]*minimal_aperture[1]:
                        apertures[i] = minimal_aperture
            apercorr = aperture_correction(fwhm=self.pixel_beam, aperture=apertures)
            corr = np.array(apercorr) * corr
        flux_table = measure_flux(self.image.value, wcs=self.wcs, coords=coords, 
                                  apertures=apertures, method=method, 
                                  segment_size=segment_scale*np.max(self.pixel_beam[0]),
                                  **kwargs) 
        flux_table['flux'] = flux_table['flux']*self.unit*corr
        flux_table['flux_err'] = flux_table['flux_err']*self.unit*corr
        return flux_table

########################################
###### stand alone functions ###########
########################################

def read_ALMA_image(fitsimage, extname='primary', name=None, debug=False, 
                  correct_beam=False, spec_idx=0, stokes_idx=0):
    """read the fits file

    Args:
        fitsimage: the filename of the fits file.
        extname: the extention name of the table to be read.
        name (optinal): the name of the image.
        debug: set to true to print the details of the fits file
        correct_beam: set to True to correct the beams if the beam 
                      information is available
        select_data: to select the data if the fitsfile contains 
                     three dimensional or four dimensional data
        spec_idx: the index of the third dimension
        stokes_idx: the selecting idx of the stokes dimension
    """
    with fits.open(fitsimage) as image_hdu:
        if debug:
            print(image_hdu.info())
        image_header = image_hdu[extname].header
        image_data = image_hdu[extname].data[stokes_idx,spec_idx] 
        image_unit = u.Unit(image_header['BUNIT'])
        if 'BMAJ' in image_header.keys():
            image_beam = [image_header['BMAJ']*3600., image_header['BMIN']*3600., 
                          image_header['BPA']]
        else:
            try: full_beam = image_hdu['BEAMS'].data
            except: image_beam = None
            image_beam = list(full_beam[stokes_idx][:3])
    if correct_beam:
        # convert Jy/beam to Jy
        if image_beam is not None:
            if '/beam' in image_header['BUNIT']:
                pixel2arcsec_ra = abs(image_header['CDELT1']*u.Unit(image_header['CUNIT1']).to(u.arcsec)) 
                pixel2arcsec_dec = abs(image_header['CDELT2']*u.Unit(image_header['CUNIT2']).to(u.arcsec))       
                pixel_area = pixel2arcsec_ra * pixel2arcsec_dec
                beamsize = 1/(np.log(2)*4.0) * np.pi * image_beam[0] * image_beam[1] / pixel_area
                image_data = image_data / beamsize * u.beam
                image_header['BUNIT'] = image_data.unit.to_string()
    if name is None:
        name = os.path.basename(fitsimage)
    return Image(data=image_data, header=image_header, beam=image_beam, name=name, 
                 unit=image_unit)

def calculate_rms(data, mask=None, masked_invalid=True, sigma_clip=True, sigma=3.0, maxiters=5,
                  mask_structures=False):
    if masked_invalid:
        data = np.ma.masked_invalid(data, copy=True)
        if mask is not None:
            mask = mask & data.mask
        else:
            mask = data.mask
    if mask_structures:
        signal_mask = find_structure(data, sigma=4.0, dilation_iters=3)
        if mask is not None:
            mask = signal_mask | mask
        else:
            mask = signal_mask
    data = np.ma.masked_array(data, mask=mask)
    if sigma_clip:
        data = stats.sigma_clip(data, sigma=sigma, maxiters=maxiters)
        
    return np.sqrt(np.ma.sum(data**2)/(data.size-np.sum(data.mask)))

def beam2aperture(beam, scale=1):
    """cover the beam in interferometric image to aperture in `photutils`

    The major difference is the reference axis:
    beam: the position angle is relative to north
    aperture: the angle is from positive x-axis to positive y-axis
    """
    scale_array = np.array([scale, scale, 1])
    return (scale_array*np.array([beam[1], beam[0], beam[-1]]) * (0.5, 0.5, np.pi/180)).tolist()

def calculate_beamarea(beam, scale=1):
    """calculate the beamszie

    Example: calculate the beamsize from physical beam to pixel area

        calculate_beamsize(beam, scale=1/pixel_area)
        
    """
    return 1/(np.log(2)*4.0) * np.pi * beam[0] * beam[1] * scale

def fluxconvert(image, beam, scale=1):
    """convert the flux in 1/beam to 1/pixel
    """
    beamsize = calculate_beamsize(beam, scale=scale)
    return image/beamsize

def aperture_stats(image, aperture=(1.4,1.4,0), mask=None, nsample=100):
    """make statistics with aperture shape
        
    This program is designed for oversampled images, which the pixel is not independent.
    To calculate the mean, median, std, rms, we need to use the beam as the minimal sampler

    Args:
        data: (ndarray) the input data
        beam: the shape of the beam in pixel coordinate, (bmaj, bmin, bpa) in units (pix, pix, deg), bmaj (bmin) is the semimajor (semiminor) axis in pixels.
        axis: the axis to evaluate the statistics
        sigma: the sigma for sigma clipping
        niter: the iteration of the sigma clipping

    Return: [mean, median, std, rms]
    """
    imagesize = image.shape
    # automatically mask the invalid values
    imagemasked = np.ma.masked_invalid(image)
    if mask is not None:
        mask = imagemasked.mask | mask
    else:
        mask = imagemasked.mask
    pixel_x = np.random.random(nsample) * imagesize[1] # 1 for x axis
    pixel_y = np.random.random(nsample) * imagesize[0] # 0 for y axis
    pixel_coords_boostrap = np.vstack([pixel_x, pixel_y]).T
    if isinstance(aperture, EllipticalAperture):
        aperture = [aperture.a, aperture.b, aperture.theta]
    apertures_boostrap = EllipticalAperture(pixel_coords_boostrap, *aperture)
    noise_boostrap = aperture_photometry(image, apertures_boostrap, 
                                         mask=mask)
    aperture_sum = noise_boostrap['aperture_sum']
    mean = np.mean(aperture_sum)
    median = np.median(aperture_sum) 
    std = np.std(aperture_sum) 
    rms = np.sqrt(np.sum(aperture_sum**2)/len(aperture_sum))
    return mean, median, std, rms

def calculate_noise_fwhm(kernal):
    return np.sqrt(2*np.log(2)/np.pi)*np.sum(kernal)/np.sqrt(np.sum(kernal**2))

def gaussian_2d(params, x=None, y=None):
    amp, x0, y0, xsigma, ysigma, beta = params
    return amp*np.exp(-0.5*(x-x0)**2/xsigma**2 - beta*(x-x0)*(y-y0)/(xsigma*ysigma)- 0.5*(y-y0)**2/ysigma**2)

def curve_growth(image, aperture, center=None, step=0.5, mask=None, 
                 max_aperture=None, min_aperture=None):
    """a simple curve of growth program 
    """
    ny, nx = image.shape
    peak_value = np.ma.array(image, mask=mask).max()
    a, b, theta = aperture
    b2a = b/a
    if max_aperture is None:
        max_aperture = nx
    if min_aperture is None:
        min_aperture = np.min([a,b])
    radii = np.arange(step, max_aperture, step)
    if center is None:
        center = [0.5*nx-0.5, 0.5*ny-0.5]
    # build all the required apertures
    phot_table = Table(names=['r', 'area', 'flux', 'flux_err'],
                       dtype=('f8', 'f8', 'f8', 'f8'))
    for r in radii:
        aper = EllipticalAperture(center, r, r*b2a, theta)
        aper_sum, aper_sum_err = aper.do_photometry(image, mask=mask)
        # print([r, aper.area, aper_sum[0], 0])
        phot_table.add_row([r, aper.area, aper_sum[0], 0])
    return phot_table

def adaptive_aperture_photometry(image, aperture, error=None, step=0.5, mask=None, 
                                 center=None, offset=None,  
                                 tolerence=1e-4, plot=False, 
                                 max_aperture=None, min_aperture=None,
                                 return_table=False, flux_ratio=0.5):
    """automatic adaptive aperture photometry

    Args:
        image: the measured value of the image, no units
        aperture: the aperture to be used, in the format of [size_maj, size_min, theta]
        error: the reference error, same shape as the image
        step: the radius step
        mask: the mask to the image
        center: the position of the source
        offset: the offset relative to the center
        tolerence: the slope increasement to be reguarded as no increase
        plot: the visualised the measurement
        max_aperture: the maximal aperture size to be used
        min_aperture: the minimal size of the aperture
        return_table: return the measurements in a astropy table format
        flux_ratio: the flux ratio to define the size of measurement, 0.5 for Re
    """
    ny, nx = image.shape
    peak_value = np.ma.array(image, mask=mask).max()
    a, b, theta = aperture
    b2a = b/a
    if max_aperture is None:
        max_aperture = nx
    if min_aperture is None:
        min_aperture = np.min([a,b])
    radii = np.arange(step, max_aperture, step)
    if center is None:
        center = [0.5*nx-0.5, 0.5*ny-0.5]
    if offset is not None:
        center = np.array(center) + np.array(offset)
    # build all the required apertures
    phot_table = Table(names=['r', 'area', 'flux', 'flux_err'],
                       dtype=('f8', 'f8', 'f8', 'f8'))
    for r in radii:
        aper = EllipticalAperture(center, r, r*b2a, theta)
        aper_sum, aper_sum_err = aper.do_photometry(image, mask=mask, error=error)
        # print([r, aper.area, aper_sum, aper_sum_err])
        if len(aper_sum_err) > 0:
            phot_table.add_row([r, aper.area, aper_sum[0], aper_sum_err[0]])
        else:
            phot_table.add_row([r, aper.area, aper_sum[0], 0])
    # calculate the flattening radius and total flux
    slope_selection = (np.diff(phot_table['flux']) < step*tolerence) & (radii[:-1]>min_aperture)
    if np.any(slope_selection):
        max_idx = np.argmax(slope_selection)
    else:
        max_idx = len(slope_selection)

    # flux_table = Table(names=['r', 'area', 'flux', 'flux_err'],
                       # dtype=('f8', 'f8', 'f8', 'f8'))
    flux_max = phot_table['flux'][max_idx]
    flux_max_err = np.max([phot_table['flux_err'][max_idx], tolerence*flux_max])
    flux_max_r = radii[max_idx]
    # calculate the radius at the give flux ratio
    flux_r_select = ((phot_table['flux'] >= flux_ratio*(flux_max-flux_max_err)) & 
                     (phot_table['flux'] <= flux_ratio*(flux_max+flux_max_err)))
    flux_r_min_index = np.where(
            phot_table['flux'] >= flux_ratio*(flux_max-flux_max_err))[0][0]
    flux_r_max_index = np.where(
            phot_table['flux'] >= flux_ratio*(flux_max-flux_max_err))[0][0]
    # ratio_flux_r = np.nanmean(radii[flux_r_select])
    # ratio_flux_r_err = np.nanstd(radii[flux_r_select])
    ratio_flux_r = np.mean([radii[flux_r_min_index], radii[flux_r_min_index]])
    ratio_flux_r_err = np.max(
            [np.diff([radii[flux_r_min_index], radii[flux_r_min_index]])[0], step])
    # print('flux_max, flux_max_err', flux_max, flux_max_err)
    # print('ratio_flux_r, ratio_flux_r_err', ratio_flux_r, ratio_flux_r_err)
    if plot:
        fig, ax = plt.subplots(1,2,figsize=(12,5))
        ax[0].imshow(np.ma.array(image,mask=mask), origin='lower')
        for r in radii:
            aper = EllipticalAperture(center, r, r*b2a, theta)
            aper.plot(ax=ax[0], color='k', alpha=0.3)

        aper = EllipticalAperture(center, ratio_flux_r, ratio_flux_r*b2a, theta)
        aper.plot(color='red', ax=ax[0], alpha=0.8)
        aper = EllipticalAperture(center, flux_max_r, flux_max_r*b2a, theta)
        aper.plot(color='black', ax=ax[0], alpha=0.8)
        ax[1].errorbar(radii, phot_table['flux'], yerr=phot_table['flux_err'])
        ax[1].vlines(x=ratio_flux_r, ymin=0, ymax=flux_ratio*flux_max, color='red')
        ax[1].text(ratio_flux_r, flux_ratio*flux_max, f'{flux_ratio}', color='red')
        ax[1].vlines(x=flux_max_r, ymin=0, ymax=flux_max, color='black')
        plt.show()
    if return_table:
        return phot_table, [ratio_flux_r, ratio_flux_r_err, flux_max, flux_max_err]
    else:
        return ratio_flux_r, ratio_flux_r_err, flux_max, flux_max_err

def petrosian_photometry(image, aperture, center=None,  step=0.5, mask=None, 
                         plot=False):
    """photometry with petrosian method: Petrosian (1976)
    
    Args:
        image: 2d image array
        aperture: [a, b, theta]
    """
    if not has_petrofit:
        raise ImportError("Cannot find petrofit!")
    ny, nx = image.shape
    a, b, theta = aperture
    b2a = b/a
    radii = np.arange(step, 0.5*nx, step)
    if center is None:
        center = [0.5*nx-0.5, 0.5*ny-0.5]
    # build all the required apertures
    phot_table = Table(names=['r', 'area', 'flux', 'flux_err'],
                       dtype=('f8', 'f8', 'f8', 'f8'))
    for r in radii:
        a = r
        b = r * a / b
        aper = EllipticalAperture(center, r, r*b2a, theta)
        aper_sum, aper_sum_err = aper.do_photometry(image, mask=mask)
        phot_table.add_row((r, aper.area, aper_sum[0], 0))
    petro_profile = petrofit.Petrosian(phot_table['r'], phot_table['area'],
                                       phot_table['flux'], phot_table['flux_err'])
    pp = petro_profile
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.imshow(image)
        for r in radii:
            aper = EllipticalAperture(center, r, r*b2a, theta)
            aper.plot()

        aper = EllipticalAperture(center, pp.r_half_light, pp.r_half_light*b2a, theta)
        aper.plot(color='red')
        plt.show()
    # return
    return pp.r_half_light, pp.r_half_light_err, pp.total_flux, pp.total_flux_err

def gaussian_fit2d(image, x0=None, rms=1, pixel_size=1, plot=False, noise_fwhm=None):
    """two dimentional Gaussian fit follows the formula provided by Condon+1997
    http://adsabs.harvard.edu/abs/1997PASP..109..166C
    see also: 
    https://casadocs.readthedocs.io/en/latest/api/tt/casatasks.analysis.imfit.html
    
    The advantage is that it provides a formula to calculate the uncertainties
    But it also comes with limitations:
        1. The assumed gaussian model should close to the true profile
        2. The final S/N should at least larger than 3, ideally should large than 5
        4. It assume the noise is uniform across the the whole image

    """
    # generate the grid
    sigma2FWHM = np.sqrt(8*np.log(2))
    ysize, xsize = image.shape
    # print('ysize,xsize', ysize, xsize)
    ygrid, xgrid = np.meshgrid((np.arange(0, ysize)+0.5)*pixel_size,
                               (np.arange(0, xsize)+0.5)*pixel_size,
                               indexing='ij')
    # automatically estimate the initial parameters
    if x0 is None:
        amp = np.percentile(image, 0.9)*1.1
        x0, y0 = xsize*0.5, ysize*0.5
        xsigma, ysigma = pixel_size*4, pixel_size*4 
        x0 = [amp, x0, y0, xsigma, ysigma, 0]

    def _cost(params, xgrid, ygrid):
        return np.sum((image - gaussian_2d(params, xgrid, ygrid))**2/rms**2)
    res_minimize = optimize.minimize(_cost, x0, args=(xgrid, ygrid), method='BFGS')

    if plot:
        fit_image = gaussian_2d(res_minimize.x, xgrid, ygrid)
        fig, ax = plt.subplots(1,3, figsize=(12, 3))
        ax[0].imshow(image)
        ax[1].imshow(fit_image)
        ax[2].imshow(image - fit_image)
    amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = res_minimize.x
    xfwhm_fit, yfwhm_fit = xsigma_fit*sigma2FWHM, ysigma_fit*sigma2FWHM
    # print(xfwhm_fit, yfwhm_fit)
    flux = 2.0*np.pi*amp_fit*xsigma_fit*ysigma_fit
    is_uncorrelated_noise = True
    if noise_fwhm is not None:
        # for corrected noise map
        # calculate the maximal smooth limits
        def _f_rho(a_M, a_m):
            return (amp_fit/rms)*np.sqrt(xfwhm_fit*yfwhm_fit)/(2*noise_fwhm)*(1+(noise_fwhm/xfwhm_fit)**2)**(0.5*a_M)*(1+(noise_fwhm/yfwhm_fit)**2)**(0.5*a_m)
        amp_fiterr = amp_fit*np.sqrt(2)/_f_rho(3/2., 3/2.)
        xfwhm_fiterr = xfwhm_fit*np.sqrt(2)/_f_rho(2.5, 0.5)
        x0_fiterr = x0_fit*np.sqrt(2)/_f_rho(2.5, 0.5)
        yfwhm_fiterr = yfwhm_fit*np.sqrt(2)/_f_rho(0.5, 2.5)
        y0_fiterr = y0_fit*np.sqrt(2)/_f_rho(0.5, 2.5)
        beta_fiterr = beta_fit*np.sqrt(2)/_f_rho(0.5, 2.5)
        flux_err_maximal = flux*np.sqrt((amp_fiterr/amp_fit)**2+noise_fwhm**2/(xfwhm_fit*yfwhm_fit)*((xfwhm_fiterr/xfwhm_fit)**2+(yfwhm_fiterr/yfwhm_fit)**2))
        # calculate the minimal smooth limits
        rho = np.sqrt(xfwhm_fit*yfwhm_fit)/(2.*noise_fwhm)*amp_fit/rms
        flux_err_mimimal = np.sqrt(2.0)*flux/rho
        # print(flux_err_maximal, flux_err_mimimal)
        # determine to use which one
        if noise_fwhm**2 > 0.8*xfwhm_fit*yfwhm_fit:
            # if the corrected noise is big compared to the source size
            flux_err = flux_err_maximal
        elif noise_fwhm**2 < 0.01*xfwhm_fit*yfwhm_fit:
            # if the corrected noise is small compared to the source size
            flux_err = flux_err_mimimal
        else:
            # if in the middle
            factor = noise_fwhm/np.sqrt(xfwhm_fit*yfwhm_fit)
            flux_err = factor*flux_err_maximal + (1-factor)*flux_err_mimimal
    else:
        # uncorrelated noise
        rho = np.sqrt(np.pi*xsigma_fit*ysigma_fit)*amp_fit/rms
        flux_err = np.sqrt(2.0)*flux/rho
    return flux, flux_err, res_minimize.x

def gaussian_2Dfitting(image, x_mean=0., y_mean=0., x_stddev=1, y_stddev=1, theta=0, debug=False,
                       xbounds=None, ybounds=None, center_bounds_scale=0.125, plot=False, ax=None):
    """Apply simple two dimentional Gaussian fitting
    
    Args:
        center_bounds = ((xlow, xup), (ylow, yup))
        center_bounds_scale: the bounds are determined by the fraction of the image size
    """

    ysize, xsize = image.shape
    y_center, x_center = ysize/2., xsize/2.
    flux_list = []
    flux_error_list = []

    yidx, xidx = np.indices((ysize, xsize))
    yrad, xrad = yidx-ysize/2., xidx-xsize/2.

    image_scale = np.max(image)
    image_norm = image / image_scale
    if xbounds is None:
        xbounds = (-xsize*center_bounds_scale,xsize*center_bounds_scale)
    if ybounds is None:
        ybounds = (-ysize*center_bounds_scale,ysize*center_bounds_scale)
    p_init = models.Gaussian2D(amplitude=1, x_mean=x_mean, y_mean=y_mean, 
            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta, 
            bounds={"x_mean": xbounds, "y_mean": ybounds, 
                    "x_stddev":(0., xsize/2.), "y_stddev":(0., ysize/2.)})
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, xrad, yrad, image_norm, 
              weights=1/(yrad**2+xrad**2+(0.25*x_stddev)**2+(0.25*y_stddev)**2))
    flux_fitted = 2*image_scale*np.pi*p.x_stddev.value*p.y_stddev.value*p.amplitude.value
    dict_return = dict(zip(p.param_names, p.param_sets.flatten().tolist()))
    dict_return['amplitude'] *= image_scale
    dict_return['flux'] = flux_fitted

    if debug:
        print("Initial guess:", p_init)
        print("Fitting:", p)
        print("Flux:", flux_fitted)
        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, 3, figsize=(8, 3.5))
                im0 = ax[0].imshow(image_norm, origin='lower', interpolation='none', 
                                   extent=(-xsize/2., xsize/2., -ysize/2., ysize/2.))
                plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
                gaussian_init = patches.Ellipse((p_init.x_mean, p_init.y_mean), 
                                                height=p_init.y_stddev.value, 
                        width=p_init.x_stddev.value, angle=p_init.theta.value/np.pi*180,
                        linewidth=1, facecolor=None, fill=None, edgecolor='orange', alpha=0.8)
                ax[0].add_patch(gaussian_init)
                ax[0].set_title("Data")
                im1 = ax[1].imshow(p(xrad, yrad), origin='lower', interpolation='none',
                                   extent=(-xsize/2., xsize/2., -ysize/2., ysize/2.))
                plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
                ax[1].set_title("Model")
                im2 = ax[2].imshow(image_norm - p(xrad, yrad), origin='lower', interpolation='none',
                                   extent=(-xsize/2., xsize/2., -ysize/2., ysize/2.))
                plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
                ax[2].set_title("Residual")
            else:
                ax.set_title("Model")
                im1 = ax.imshow(p(xrad, yrad), origin='lower', interpolation='none',
                                   extent=(-xsize/2., xsize/2., -ysize/2., ysize/2.))
                plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    return dict_return

def find_structure(image, mask=None, rms=None, sigma=3.0, iterations=1, opening_iters=None, 
                   dilation_iters=None, kernel=None, plot=False):
        """find structures above a centain sigma

        Firstly, this function will find the 3sigma large structure;
        after that, it applys the binary opening to remove small structures
        then, binary dilation is used to exclude all the all the faint 
        extended emission
        This function can be used for find extended emission regions and generate
        additional masking

        Args:
            sigma: the sigma level to start the structure searching
            iterations: the opening and dilation iterations
            opening_iters: specifically set the opening iterations
            dilation_iters: specifically set the dilation iterations
            kernel: the convolving kernel, used to further extend the
                    existing structures
            plot: plot the marked regions.

        Return:
            image: binary image with masked region with 1 and the others 0
        """
        if rms is None:
            # std = calculate_rms(data, mask=mask, sigma=sigma, maxiters=maxiters)
            _, _, std = stats.sigma_clipped_stats(image, sigma=sigma, maxiters=iterations, mask=mask)
        struct_above_sigma = image > sigma*std
        if opening_iters is None:
            opening_iters = iterations
        if dilation_iters is None:
            dilation_iters = iterations
        sigma_structure = struct_above_sigma
        if opening_iters > 0:
            sigma_structure = ndimage.binary_opening(sigma_structure, iterations=opening_iters)
        if dilation_iters > 0:
            sigma_structure = ndimage.binary_dilation(sigma_structure, iterations=dilation_iters)
        # if kernel is not None:
            # kernel_image = np.ones()
            # sigma_structure = scipy.signal.fftconvolve(sigma_structure, kernel)
        if plot:
            plot_pixel_image(sigma_structure)
        return sigma_structure

def source_finder(image=None, wcs=None, std=None, mask=None, 
                  beam=None, aperture_scale=6.0, 
                  detection_threshold=3.0, 
                  plot=False, name=None, show_flux=False, 
                  method='auto', filter_kernel=None, # parameters for sep
                  find_peaks_params={}, DAOStarFinder_params={},
                  ax=None, savefile=None):
    """a source finder and flux measurement wrapper of SEP
    It is designed to handle the interferometric data
    
    Params:
        aperture_scale : aperture size in units of FWHM, it should be used along with the beam factor
        detection_threshold: the minimal SNR when searching for detections.

    Notes: 'sep' also return the size of the source [a, b, theta]. `a` and `b` is the pixel size 
        of the major and minor axis. The `theta` is the position-angle (in radian) from 
        positive x to positive y.

    """
    if method == 'auto':
        try:
            import sep
            method = 'sep'
        except:
            method = 'find_peaks'
    if beam is None:
        beam = [1,1] # set to 1x1 pixel size
    if method == 'sep':
        try:
            import sep
        except:
            raise ValueError("SEP cannot be found! Use method='find_peak' or 'DAOStarFinder_params'.")
        try: sources_found = Table(sep.extract(image, detection_threshold,
                                               err=std, mask=mask, filter_kernel=filter_kernel))
        except:
            # convert the byte order from big-endian (astropy.fits default) on little-endian machine
            image_little_endian = image.byteswap().newbyteorder()
            sources_found = Table(sep.extract(image_little_endian, detection_threshold, 
                                              err=std, mask=mask, filter_kernel=filter_kernel))
        if sources_found is not None:
            sources_found.rename_column('peak', 'peak_value')
            sources_found.add_column(sources_found['peak_value']/std, name='peak_snr')
        
    if method == 'find_peaks':
        try:
            from photutils import find_peaks
        except:
            raise ValueError("photutils is NOT installed! Try to use method='sep'")
        sources_found = find_peaks(image, threshold=detection_threshold*std, 
                                   mask=mask, **find_peaks_params) 
        if sources_found is not None:
            sources_found.rename_column('x_peak', 'x')
            sources_found.rename_column('y_peak', 'y')
            sources_found.add_column(sources_found['peak_value']/std, name='peak_snr')

    if method == 'DAOStarFinder':
        #TODO: not fully tested, the convolutional step is tricky for interferometric images
        print("Warning: DAOStarFinder is not fully tested!")
        try:
            from photutils import DAOStarFinder
        except:
            raise ValueError("photutils is NOT installed!")
        daofind = DAOStarFinder(threshold=detection_threshold*std, 
                                fwhm=beam[0], ratio=beam[1]/beam[0], theta=beam[-1]+90,
                                sigma_radius=1.5, 
                                sharphi=2.0, sharplo=0.2,) 
        sources_found = daofind(image, mask=mask)
        if sources_found is not None:
            sources_found.rename_column('xcentroid', 'x')
            sources_found.rename_column('ycentroid', 'y')
            sources_found.rename_column('peak', 'peak_value')
            sources_found['peak_snr'] = sources_found['peak_value']/std

    # add extra info
    if sources_found is not None:
        n_found = len(sources_found)
        table_new_name = Table(np.array(['', '', '']*n_found).reshape(n_found, 3), 
                               names={'name', 'code', 'comments'},
                               dtype=['U64', 'U64', 'U80'])
        table_objs = hstack([table_new_name, sources_found])
        for i in range(n_found):
            obj = table_objs[i]
            obj['code'] = 0
            if name is not None:
                obj['name'] = name + '_' + str(i)
            else:
                obj['name'] = str(i)
    else:
        table_objs = None

    if wcs is not None:
        n_dets = len(table_objs)
        table_new_data = Table(np.array([0., 0.] * n_dets).reshape(n_dets, 2),
                           names={'ra', 'dec'})
        table_objs = hstack([table_new_data, table_objs])
        # convert the pixel coordinates to sky coordinates
        for i in range(n_dets):
            obj = table_objs[i]
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest',
                       vmin=-2.*std, vmax=5.0*std, origin='lower')
        ax.set_title(name)
        n_found = len(table_objs)
        for i in range(n_found):
            obj = table_objs[i]
            ax.text(obj['x'], obj['y'], '{}'.format(i+1), alpha=0.5, color='red',
                    horizontalalignment='center', verticalalignment='center')
            try:
                ellipse = patches.Ellipse((obj['x'], obj['y']), width=6*obj['a'], 
                                          height=6*obj['b'], angle=obj['theta']*180/np.pi, 
                                          facecolor='none', edgecolor='black', alpha=0.5)
                ax.add_patch(ellipse)
            except:
                pass
                # ax.plot(obj['x'], obj['y'], 'x', color='tomato')
            if show_flux:
                if 'peak_value' in obj.colnames:
                    ax.text(obj['x'], 0.94*obj['y'], "{:.2e}".format(
                            obj['peak_value']*1e3), color='white', 
                            horizontalalignment='center', verticalalignment='top',)
    if savefile:
        save_array(table_objs, savefile)
    else:
        return table_objs

def make_apertures(coords, shapes=(1.4, 1.4, 0)):
    """create elliptical apertures

    Args:
        coords: the pixel coordinates of the apertures
        shapes: the shapes of apertures, follow the format of `photoutils.EllipticalAperture`
            [a,b,theta] or [[a1,b1,theta1],[a2,b2,theta2],], in the units of (pixel, pixel, radian)
            a is the length of semimajor axis, b is the length of semiminor axis
            default is the aperture of a single pixel 
    """
    ndim = np.ndim(coords)
    apertures = []
    ndim_shapes = np.ndim(shapes)
    if ndim == 1:
        if ndim_shapes == 1:
            apertures = EllipticalAperture(coords, *shapes)
        elif ndim_shapes == 2:
            apertures = []
            for i in range(len(shapes)):
                apertures.append(EllipticalAperture(coords, *shapes[i]))
        else:
            print('coords', coords)
            print('shapes', shapes)
            raise ValueError("``image_tools.create_aperture``: unmatched aperture with coordinates")
    elif ndim == 2:
        apertures = []
        if ndim_shapes == 1:
            for coord in coords:
                apertures.append(EllipticalAperture(coord, *shapes))
        elif ndim_shapes == 2:
            assert len(shapes) == len(coords)
            for i,coord in enumerate(coords):
                apertures.append(EllipticalAperture(coord, *shapes[i]))
    return apertures

def mask_coordinates(image=None, coords=None, shape=None, apertures=None):
    """remove all the detections, return image with only backgrouds

    All the coordinates and aperture are in pixels

    Args:
        image: the 2D image data
        apertures: the list of ``photoutils.EllipticalAperture``

    Return:
        2D mask array
    """
    image_mask = np.zeros_like(image, dtype=bool)
    image_shape = image.shape
    if isinstance(image_mask, u.Quantity):
        image_mask = image_mask.value
    if apertures is None:
        apertures = make_apertures(coords, shape=shape)
    if isinstance(apertures, EllipticalAperture):
        apertures = [apertures,]
    for aper in apertures:
        mask = aper.to_mask().to_image(image_shape) > 0
        image_mask[mask] = True
    return image_mask

def source_deblend(image, pixel_coords, mask=None, plot=False, algorithm='watershed'):
    """deblending two targets with overlapped regions

    """
    markers = np.zeros_like(image, dtype=bool)
    print('pixel_coords', pixel_coords)
    for coord in pixel_coords:
        markers[int(coord[1]), int(coord[0])] = True
    # markers = ndimage.binary_dilation(markers, iterations=2)
    marker_labels = ndimage.label(markers)

    if algorithm == 0 or algorithm=='watershed':
        labels = watershed(-image, markers=marker_labels[0], compactness=1, connectivity=1,
                       mask=mask)
    elif algorithm == 'random_walker':
        labels = random_walker(image, labels=marker_labels[0], mask=mask)
    if plot:
        fig, ax = plt.subplots(1,3, figsize=(12,4))
        ax[0].imshow(image)
        ax[0].imshow(labels, alpha=0.4)
        for coord in pixel_coords:
            ax[0].plot(int(coord[0]), int(coord[1]), 'x', color='red')
            ax[0].plot(coord[0], coord[1], 'o', color='white')
        ax[1].imshow(marker_labels[0])
        ax[2].imshow(image)
        plt.show()
    return labels

def measure_flux(image, wcs=None, detections=None,
                 coords=None, apertures=None, aperture_scale=1,
                 method='adaptive-aperture', max_aperture=None, tolerence=1e-4,
                 mask=None, n_boostrap=100,
                 segment_size=30.0, noise_fwhm=None, rms=None,
                 plot=False, ax=None, color='white', debug=False):
    """Two-dimension flux measurement in the pixel coordinates
    
    It supports three ways to provide the coordinates of the detection. The ``detections`` allows
    table-like input, while the pixel_coords and sky_coords allow direct coordinates.
    The aperture are provided by the aperture and aperture_scale. ``aperture`` provides the shape
    while aperture_scale constraints the size.

    Args:
        image: the data with or without units
        wcs: the world coordinate system of the image, optional, if provide, the final table
             will also include the ra and dec information
        detections: the detection table from sounce_finder, useful informations are 'x,y,a,b,theta', used
            to define the apetures
        coords: the pixel coordinates of the detections, [[x1,y1], [x2,y2]]
        aperture: the fixed size of the aperture, in pixels
        aperture_scale: scale the size of the aperture in units of aperture
        method: accepted methds including: single-aperture, adaptive-aperture, gaussian
        mask: the 2d mask array
        n_boostrap: used to boostrap the flux errors by randomly put the aperture across the image
        segment_size: segment size to cut out the image. which is used for gaussian fitting.LevMarLSQFitter
                      noly support one value, square cutout.

    """
    imagesize = image.shape
    segment_size = np.min([imagesize[0], segment_size])

    if detections is not None:
        coords = []
        coords = list(zip(detections['x'], detections['y']))
        # for det in detections:
            # coords.append((det['x'], det['y']))
        if apertures is None:
            apertures = []
            for det in detections:
                apertures.append([det['a'], det['b'], det['theta']])
    if (coords is not None) and (apertures is not None):
        aperture_list = []
        if np.ndim(coords) == 1:
            coords = [coords,]
        if np.array(apertures).ndim == 1:
            # use the same aperture for all the detection
            for coord in coords:
                aperture_list.append([*coord, *apertures])
        else:
            assert len(apertures) == len(coords)
            for i in range(len(coords)):
                aperture_list.append([*coords[i], *apertures[i]])
    n_sources = len(aperture_list)
    if debug:
        print("aperture_list", aperture_list)
        print(f"Found {n_sources} detections.")
        print(f"segment_size is {segment_size}")
    
    if False:
        # start the global masking with sigma clipping 
        if isinstance(apertures, EllipticalAperture):
            detections_apers = apertures
        else: # make new apertures with the shape parameters
            detections_apers = make_apertures(coords, shapes=apertures)
        # convert single item into list
        if isinstance(detections_apers, EllipticalAperture):
            detections_apers = [detections_apers,]
        detections_mask = mask_coordinates(image, apertures=detections_apers)
        n_sources = len(detections_apers)
    
    # mask all the strong sources
    detections_mask = find_structure(image, sigma=4, dilation_iters=2)
    if rms is None:
        rms = calculate_rms(image, sigma=4, mask=detections_mask)
    
    # check if segmentation is needed
    if n_sources > 0:
        segments = RectangularAperture(coords, segment_size, segment_size, theta=0)
        segments_mask = segments.to_mask(method='center')
    else:
        segments_mask = None
    coords = np.array(coords)
    if debug: 
        print(f"the coords is: {coords}")
        print(f'the aperture_list is: {aperture_list}')
    # create the table to record the results
    table_flux = Table(names=('ID','x','y','aper_maj','aper_min','theta','flux','flux_err'), 
                         dtype=('i4','f8','f8','f8','f8','f8','f8','f8'))
    if method == 'single-aperture':
        # measuring flux density
        for i in range(n_sources):
            x, y, a, b, theta = aperture_list[i]
            if segments_mask is not None:
                large_slice, _ = segments_mask[i].get_overlap_slices(imagesize)
                yslice, xslice = large_slice
                image_cutout = segments_mask[i].cutout(image)
                center = [x-xslice.start, y-yslice.start]
            else:
                image_cutout = image
                center = [x, y]
            aper = EllipticalAperture(center, a*aperture_scale, b*aperture_scale, theta)
            phot_table = aperture_photometry(image_cutout, aper, mask=mask)
            # aperture_correction
            flux = phot_table['aperture_sum'].value
            _, _, flux_err, _ = aperture_stats(image, aper, 
                                               )#mask=detections_mask)
            table_flux.add_row((i, x, y, a, b, theta, flux, flux_err))

    if method == 'adaptive-aperture':
        for i in range(n_sources):
            x, y, a, b, theta = aperture_list[i]
            # check if there is any overlapped sources in the segment
            if segments_mask is not None:
                # print(segments_mask[i].get_overlap_slices(imagesize))
                large_slice, _ = segments_mask[i].get_overlap_slices(imagesize)
                yslice, xslice = large_slice
                image_cutout = segments_mask[i].cutout(image)
                if mask is not None:
                    mask_cutout = segments_mask[i].cutout(mask)
                else:
                    mask_cutout = None
                center = [x-xslice.start, y-yslice.start]
                # 0.7 = np.sqrt(2)*0.5
                n_within = np.sum(np.sqrt((coords[:,0]-x)**2 + (coords[:,1]-y)**2) < 0.7*segment_size)-1
                if n_within > 0:
                    # mask the all the other targets
                    print("Found overlapped targets!")
                    coords_within = []
                    for coord in coords:
                        if ((coord[0]-x)**2 + (coord[1]-y)**2) < (0.7*segment_size)**2:
                            coords_within.append([coord[0]-xslice.start,coord[1]-yslice.start])
                    labels = source_deblend(image_cutout, pixel_coords=coords_within, 
                                            mask=image_cutout>2*rms)
                    # get the mask of the nearby source
                    primary_label = labels[int(center[1]), int(center[0])]
                    label_mask = (labels < 0.5) | (labels==primary_label)
                    if mask_cutout is not None:
                        mask_cutout = (~label_mask) | mask_cutout
                    else:
                        mask_cutout = ~label_mask
            else:
                image_cutout = image
                mask_cutout = mask
                center = [x,y]
            Rmax, Rmax_err, flux, flux_err = adaptive_aperture_photometry(
                    image_cutout, center=center, aperture=(a, b, theta), plot=debug, 
                    mask=mask_cutout,
                    tolerence=tolerence, max_aperture=max_aperture,
                    flux_ratio=1.0)
            a_Rmax = Rmax
            b_Rmax = Rmax * b/a
            theta = theta
            _, _, flux_err, _ = aperture_stats(image, (a_Rmax, b_Rmax, theta), )#mask=detections_mask)
            table_flux.add_row((i, x, y, a_Rmax, b_Rmax, theta, flux, flux_err))
   
    if method == 'gaussian':
        for i,s in enumerate(segments_mask):
            x,y = coords[i]
            if segments_mask is not None:
                image_cutout = segments_mask[i].cutout(image)
            else:
                image_cutout = image
            # gaussian_fitting = gaussian_2Dfitting(image_cutout, debug=debug, plot=plot)
            flux, flux_err, params_fit = gaussian_fit2d(image_cutout, rms=rms, noise_fwhm=noise_fwhm, 
                                                        plot=debug)
            a_fitted_aper = 3 * params_fit[3] 
            b_fitted_aper = 3 * params_fit[4]
            theta_fitted = params_fit[5]/180*np.pi
            
            table_flux.add_row((i, x, y, a_fitted_aper, b_fitted_aper, theta_fitted, 
                                  flux, flux_err))

    if method == 'gaussian_old':

        segments = RectangularAperture(coords, segment_size, segment_size, theta=0)
        segments_mask = segments.to_mask(method='center')
        for i,s in enumerate(segments_mask):
            x,y = coords[i]
            image_cutout = s.cutout(image)
            gaussian_fitting = gaussian_2Dfitting(image_cutout, debug=debug, plot=plot)
            
            flux = gaussian_fitting['flux'] 
            # caulcate the error: 
            
            # method1: boostrap for noise measurement
            a_fitted_aper = 3 * gaussian_fitting['x_stddev'] # 2xFWHM of gaussian
            b_fitted_aper = 3 * gaussian_fitting['y_stddev']
            theta_fitted = gaussian_fitting['theta']/180*np.pi
            _, _, flux_err, _ = aperture_stats(image, (a_fitted_aper, b_fitted_aper,
                                                                theta_fitted),
                                                        mask=detections_mask)
            table_flux.add_row((i, x, y, a_fitted_aper, b_fitted_aper, theta_fitted, 
                                  flux, flux_err))

    if wcs is not None:
        n_dets = len(table_flux)
        table_new_data = Table(np.array([0., 0., 0., 0.] * n_dets).reshape(n_dets, 4),
                               names={'ra', 'dec', 'aper_min_sky','aper_maj_sky'})
        table_flux = hstack([table_flux,table_new_data])
        # convert the pixel coordinates to sky coordinates
        # assuming equal pixel size in y and x
        for i in range(n_dets):
            obj = table_flux[i]
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
        pixel2arcsec_mean = np.mean(np.abs(wcs.wcs.cdelt)*3600)
        table_flux['aper_min_sky'] = table_flux['aper_min'] * pixel2arcsec_mean
        table_flux['aper_maj_sky'] = table_flux['aper_maj'] * pixel2arcsec_mean 

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest', origin='lower')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        for i in range(n_sources):
            obj = table_flux[i]
            # ax.text(obj['x'], obj['y'], i, color='red', horizontalalignment='center', 
                    # verticalalignment='center')
            ellipse = patches.Ellipse((obj['x'], obj['y']), 
                                      width=2*obj['aper_maj'], height=2*obj['aper_min'], 
                                      angle=obj['theta']*180/np.pi, 
                                      facecolor='none', edgecolor='black', alpha=0.5)
            vert_dist = 12+0.4*(obj['aper_maj']*np.cos(obj['theta']) + obj['aper_min']*np.cos(obj['theta']))
            ax.add_patch(ellipse)
            ax.text(obj['x'], obj['y']-abs(vert_dist), "{:.2f}".format(obj['flux']), 
                    color=color, horizontalalignment='center', verticalalignment='center',)
        plt.show()
        # # only for test
        # for ap in apertures_boostrap:
            # im = ap.plot(color='gray', lw=2, alpha=0.2)
    return table_flux


        #table_return = Table(np.array(['', 0., 0., 0., 0.] * n_found).reshape(n_sources, 5),
        #                   names={'name', 'ra', 'dec', 'flux', 'fluxerr'})
        #for i in range(n_sources):
        #    tab_item = table_return[i]
        #    tab_item['name'] = name + '-' + str(i)
            # tab_item['ra'] = detections['ra'][i]
            # tab_item['dec'] = detections['dec'][i]
            # tab_item['flux'] = name
            # tab_item['fluxerr'] = name

def detections_convert_wcs(detections, wcs_in, wcs_out):
    """convert the detections in different wcs
    """
    detections_new = detections.copy()
    ndets = len(detections)
    cdelt_in = wcs_in.wcs.cdelt*3600.
    cdelt_out = wcs_out.wcs.cdelt*3600.
    cdelt_scale = cdelt_in / cdelt_out
    for i in range(ndets):
        det = detections_new[i]
        sk = SkyCoord(ra=det['ra'], dec=det['dec'], unit='deg')
        det['x'], det['y'] = skycoord_to_pixel(sk, wcs_out)
        # non-equal pixelsize will cause problem here
        try:
            det['aper_min'] = det['aper_min'] * cdelt_scale[0]
            det['aper_maj'] = det['aper_maj'] * cdelt_scale[1]
        except:
            pass
        try:
            det['b'] = det['b'] * cdelt_scale[0]
            det['a'] = det['a'] * cdelt_scale[1]
        except:
            pass
    return detections_new

def make_rmap(image, pixel_scale=1.0):
    """make radial map relative to the image center
    """
    ny, nx = image.shape
    x_index = (np.arange(0, nx) - nx/2.0) * pixel_scale
    y_index = (np.arange(0, ny) - ny/2.0) * pixel_scale
    x_map, y_map = np.meshgrid(x_index, y_index)
    rmap = np.sqrt(x_map**2 + y_map**2)
    return rmap

def make_gaussian_kernel(shape, fwhm=None, sigma=None):
    if fwhm is not None:
        fwhm2sigma = 2.35482
        sigma = (np.array(fwhm) / fwhm2sigma).tolist()
    kernel = convolution.Gaussian2DKernel(*sigma)
    return kernel

def make_gaussian_image(imsize=None, fwhm=None, sigma=None, area=1., offset=(0,0), theta=0, 
                        normalize=False):
    """make a gaussian image

    Args:
        shape: the shape of the gaussian image
        fwhm: the FWHM shape of the beam, [bmaj, bmin, theta_in_degree]
        theta: in rad, rotating the gaussian counterclock wise (from positive x to positive y)

    Notes:
        1. The image center of Gaussian image is (x_shape-1)/2. This is because image coordinates is 
           defined as the pixel center.
    """
    if fwhm is not None:
        fwhm2sigma = 2.35482
        if isinstance(fwhm, (int, float)):
            sigma = fwhm / fwhm2sigma
        else:
            if len(fwhm) == 2:
                sigma = (np.array(fwhm) / fwhm2sigma).tolist()
            elif len(fwhm) == 3:
                bmaj, bmin, theta_degree = fwhm
                sigma = (np.array([bmaj, bmin]) / fwhm2sigma).tolist()
                theta = np.radians(theta_degree-90)
    if isinstance(sigma, (int, float)):
        sigma = [sigma, sigma]
    ysigma, xsigma = sigma
    if imsize is None:
        #auto_size = np.max([np.round(np.max(sigma) * 10).astype(int), 21])
        auto_size = (np.round(np.max(sigma) * 10).astype(int))//2*2-1
        imsize = [auto_size, auto_size]
    elif isinstance(imsize, (int, float)):
        imsize = [imsize, imsize]
    image = np.zeros(imsize, dtype=float)
    yidx, xidx = np.indices(imsize) + 0.5
    yrad, xrad = yidx-0.5*imsize[0], xidx-0.5*imsize[1]
    y = xrad*np.cos(theta) + yrad*np.sin(theta)
    x = yrad*np.cos(theta) - xrad*np.sin(theta)
    if isinstance(offset, (list, tuple)):
        x_offset, y_offset = offset
    flux = area * np.exp(-(x-x_offset)**2/2./xsigma**2 - (y-y_offset)**2/2./ysigma**2) / (2*np.pi*xsigma*ysigma)
    if normalize:
        flux = flux / np.sum(flux)
    return flux 

def aperture_correction(aperture=None, psf='Gaussian', fwhm=[1.4,1.4,0],  imsize=None, 
                        sourcesize=0, debug=False):
    """This function generate the aperture correction for assumed 3-D gaussian sources

    Args:
        imsize (int, list, tuple): the size of the generated image
        aperture (list, tuple): the beam size of the image --> [bmaj, bmin, theta]
                                follow the rules of `photoutils.EllipticalAperture`
        source size (float): the size of the source, only support round source
        fwhm: the beam shape of gaussian, in [pixel, pixel, degree]
    """
    #TODO: generaize this function to support input PSF
    # set up the aperture
    if psf == 'Gaussian':
        # set the FWHM of gaussian 
        if isinstance(fwhm, (int,float)):
            bmaj, bmin = fwhm, fwhm
            theta = 0
        if len(fwhm) == 2:
            bmaj, bmin = fwhm
            theta = 0
        elif len(fwhm) == 3:
            bmaj, bmin, theta = fwhm
        fwhm_convolved = np.sqrt(np.array([bmaj,bmin])**2+np.array([sourcesize, sourcesize])**2).tolist()
        image = make_gaussian_image(fwhm=[*fwhm_convolved, theta], area=1.0)
    else:
        image = psf
    imagesize = image.shape
    if isinstance(aperture, EllipticalAperture):
        apers = aperture
    else:
        if isinstance(aperture, (int, float)):
            aperture = [aperture, aperture, 0]
        apers = make_apertures(coords=[(imagesize[0]-1)/2,(imagesize[1]-1)/2], shapes=aperture)
        if isinstance(apers, EllipticalAperture):
            aper_phot = aperture_photometry(image, apers)
            aper_corr = 1.0/aper_phot['aperture_sum'].value[0]
        else:
            aper_corr = []
            for aper in apers:
                aper_phot = aperture_photometry(image, aper)
                aper_corr.append(1.0/aper_phot['aperture_sum'].value[0])
    if debug:
        plt.imshow(image, origin='lower')
        if isinstance(apers, EllipticalAperture):
            ap_patches = apers.plot(color='red', lw=2)
        else:
            for aper in apers:
                ap_patches = aper.plot(color='red', lw=2)
    return aper_corr

def save_array(array, savefile=None, overwrite=False, debug=False):
    if savefile:
        if overwrite or not os.path.isfile(savefile):
            Table(array).write(savefile, format='ascii', overwrite=overwrite)
        else: # it should append the new data at the bottom of the file
            if debug:
                print("File exsiting, appending to {}".format(savefile))
            new_Table = Table(array)
            # read the column from the existin savefile
            with open(savefile) as fp:
                first_line = fp.readline().strip()
                colnames = first_line.split(' ')
            with open(savefile, 'a+') as fp2: 
                for row in new_Table:
                    fp2.write(' '.join(map(str, row[colnames])) + '\n')
        
def read_array(datafile):
    """read txt files into structure array
    """
    return Table.read(datafile, format='ascii')
                
def solve_impb(datafile=None, data=None, pbfile=None, pbdata=None, 
               pbcorfile=None, pbcordata=None, header=None, 
               pixel_coords=None, sky_coords=None, debug=False):
    """This function derive the primary correction

    #TODO: add aperture support
    """
    if pbdata is not None:
        pass
    elif pbfile is not None:
        with fits.open(pbfile) as hdu:
            pbdata = hdu[0].data[0,0]
            if header is None:
                header = hdu[0].header
    elif (data is not None) and (pbcordata is not None):
        pbdata = data / pbcordata
    elif (datafile is not None) and (pbcorfile is not None):
        with fits.open(datafile) as hdu:
            data = hdu[0].data[0,0]
            if header is None:
                header = hdu[0].header
        with fits.open(pbcorfile) as hdu:
            pbcordata = hdu[0].data[0,0]
            pbdata = data / pbcordata
    else:
        raise ValueError("No valid primary beam information has been provided!")
    if sky_coords is not None:
        if sky_coords.size == 1:
            pixel_coords = np.array(skycoord_to_pixel(sky_coords, WCS(header)))
        else:
            pixel_coords = np.array(list(zip(*skycoord_to_pixel(sky_coords, WCS(header)))))
    if pixel_coords is None:
        return Image(data=pbdata, header=header, name='pb')
    else:
        if debug:
            print("pbdata.shape", pbdata.shape)
            print("pixel_coords", pixel_coords)
        # apply interpolation
        ys, xs = pbdata.shape
        yy, xx = np.arange(0, ys)+0.5, np.arange(0, xs)+0.5
        pb_interp = interpolate.RegularGridInterpolator((xx, yy), pbdata.T)
        pbcor_values = pb_interp(pixel_coords)
        # pbcor_values = []
        # for coord in pixel_coords:
            # coord_int = np.round(coord).astype(int)
            # pbcor_values.append(pbdata[coord_int[1], coord_int[0]])
        return pbcor_values

def beam2psf(beam, savefile=None, normalize=False, overwrite=False):
    """generate the psf from the beam

    """
    psf_image = make_gaussian_image(fwhm=beam, normalize=normalize)
    if savefile is not None:
        hdr = fits.Header()
        hdr['COMMENT'] = "PSF generated from the image beam."
        # hdr['CRVAL1'] = self.pixel_sizes[0].to('arcsec').value
        # hdr['CRVAL2'] = self.pixel_sizes[1].to('arcsec').value
        primary_hdu = fits.PrimaryHDU(header=hdr, data=psf_image)
        primary_hdu.writeto(savefile, overwrite=overwrite)
    else:
        return psf_image

def image2noise(image, shape=None, header=None, wcs=None, savefile=None, sigma=5.0, 
                mode='std', overwrite=False,):
    """
    """
    mean, median, std = stats.sigma_clipped_stats(image, sigma=sigma, maxiters=3)
    if shape is None:
        shape = image.shape
    if mode == 'median':
        noise_image = np.full(shape, fill_value=np.abs(median))
    if mode == 'std':
        noise_image = np.full(shape, fill_value=np.abs(std))
    elif mode == 'random_choice':
        image_masked = np.ma.masked_array(image, mask=image>sigma*std)
        image_1d = image_masked.filled(median+std).flatten()
        noise_image = np.abs(np.random.choice(image_1d, image.shape))
    if savefile is not None:
        hdr = fits.Header()
        hdr['COMMENT'] = "Noise image from the original ({}).".format(mode)
        # hdr['CRVAL1'] = self.pixel_sizes[0].to('arcsec').value
        # hdr['CRVAL2'] = self.pixel_sizes[1].to('arcsec').value
        primary_hdu = fits.PrimaryHDU(header=hdr, data=noise_image)
        primary_hdu.writeto(savefile, overwrite=overwrite)
    else:
        return noise_image

def rotate_map(x, y, image, pa):
    """ rotate the image according to the PA 
    in the direction of counterclock wise
    
    Args:
        pa: the angel counter-clockwise to the north

    """
    # change the PA to the radian angle relative to the positive x axis
    angle_rad = np.radians(90.-pa)
    rx = x*np.cos(angle_rad) - y*np.sin(angle_rad)
    ry = x*np.sin(angle_rad) + y*np.cos(angle_rad)
    return rx, ry

def vorbin(image, noise=None, snr=5., pixelsize=0.1, snr_limit=3.0, 
           npixels_limit=None):
    if not has_vorbin:
        raise ValueError("Install vorbin to use vorbin!")
    from vorbin.voronoi_2d_binning import voronoi_2d_binning
    if image.ndim != 2:
        raise ValueError("Only 2d image will be accepted!")
    ny,nx = image.shape
    mx_image = np.max(image.shape)
    x = ((np.arange(0, nx)) + 0.5) * pixelsize
    y = ((np.arange(0, ny)) + 0.5) * pixelsize
    xx, yy = np.meshgrid(x, y)
    xbin, ybin, signal = map(np.ravel, (xx, yy, image))
    if isinstance(noise, (int, float)):
        noise_flat = np.full(signal.size, fill_value=noise)
    elif noise.shape == image.shape:
        noise_flat = np.ravel(noise)
    print(xbin.shape, ybin.shape, signal.shape, noise_flat.shape)

    binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        xbin, ybin, signal, noise_flat, snr, plot=0, quiet=1, wvt=False, 
        pixelsize=pixelsize)
    
    bin_image = binNum.reshape(ny, nx)
    if snr_limit is not None:
        pass
    return bin_image

def fit_pa(image, xbin=None, ybin=None, mom=1, err=1):
    """fit the major axis of the data

    Inspired by the kinmetry and pa_fit

    mom = 0: returns the PA of the intensity
    mom = 1: returns the kinematic PA
    mom = 2: returns the dispersion PA
    """
    if image.ndim == 1:
        if (xbin is None) or (ybin is None):
            raise ValueError('Please provide the coordinates')
        if image.size != xbin.size != ybin.size:
            raise ValueError("data and coordinate should share the same size!")
    elif (image.ndim == 2) and (xbin is None):
        ny,nx = image.shape
        mx_img = np.max(img.shape)
        x = (np.arange(0, nx)) + 0.5
        y = (np.arange(0, ny)) + 0.5
        xx, yy = np.meshgrid(x, y)
        xbin, ybin = map(np.ravel, (xx, yy))
    else:
        raise ValueError("Please provide an image or xbin, ybin, data!")

    xbin, ybin, image = map(np.ravel, [xbin, ybin, image])
    pa_list = np.arange(-90, 90, 0.5)
    chi2 = np.zeros_like(pa_list)
    for i,pa in enumerate(pa_list):
        # create symmetric velocity map
        x, y = rotate_map(xbin, ybin, image, pa)
        x_sym = np.hstack([x,-x, x,-x])
        y_sym = np.hstack([y, y,-y,-y])
        image_sym = interpolate.griddata((x, y), image, (x_sym, y_sym))
        image_sym = image_sym.reshape(4, xbin.size)
        image_sym[0, :] = image
        if mom == 1:
            image_sym[[1, 3], :] *= -1.
        image_sym = np.nanmean(image_sym, axis=0)
        chi2[i] = np.sum(((image-image_sym)/err)**2)
    k = np.argmin(chi2)
    pa_best = pa_list[k]

    # calculate 1 sigma 
    f = chi2 - chi2[k] <= 1 + np.sqrt(2*xbin.size)
    pa_min_err = max(0.5, (pa_list[1] - pa_list[0])/2.0)
    if f.sum() > 1:
        pa_err = (np.max(pa_list[f]) - np.min(pa_list[f]))/2.0
        if pa_err >= 45:
            good = np.degrees(np.arctan(np.tan(np.radians(pa_liat[f]))))
            pa_err = (np.max(good) - np.min(good))/2.0
        pa_err = max(angErr, minErr)
    else:
        pa_err = pa_min_err

    return pa_best, pa_err 

def transfer_detections(dets, wcs_target, wcs_origin=None, aperture=None, beam=None):
    """transfer detections from one wcs into another

    Aperture: pixel size aperture
    beam: pixel size beam
    """
    dets_new = dets.copy()
    dets_new.rename_column('aper_maj', 'a')
    dets_new.rename_column('aper_min', 'b')
    dets_colnames = dets.colnames
    if ('ra' in dets_colnames) and ('dec' in dets_colnames):
        skcoords = SkyCoord(ra=dets['ra'], dec=dets['dec'], unit='deg')
    else:
        if wcs_origin is not None:
            skycoord = pixel_to_skycoord(dets['x'],dets['y'], wcs_origin)
        else:
            raise ValueError("No sky information available!")
    xp, yp = skycoord_to_pixel(skcoords, wcs_target)
    dets_new['x'] = xp
    dets_new['y'] = yp
    if ('aper_maj_sky' in dets_colnames) and ('aper_min_sky' in dets_colnames):
        pass
    elif wcs_origin is not None:
        # assuming equal pixel size in y and x
        pixel2arcsec_origin = np.mean(np.abs(wcs.wcs.cdelt))
        pixel2arcsec_target = np.mean(np.abs(wcs_target.wcs.cdelt))
        dets_new['a'] = (dets['aper_min']*pixel2arcsec_origin/pixel2arcsec_target)
        dets_new['b'] = (dets['aper_maj']*pixel2arcsec_origin/pixel2arcsec_target)
    else:
        if beam is not None:
            aperture = beam2aperture(beam)
        if aperture is not None:
            dets_new['a'] = aperture[0]
            dets_new['b'] = aperture[1]
            dets_new['theta'] = aperture[2]
    return dets_new

def extract_radial_profile(data, aperture=None, step=1., rmin=0., rmax=None,
                           mask=None,
                           center=None, offset=None, aperture_guess=None,
                           debug=False):
    """extract radial profile with ellipticalAnnulus

    The aperture and aperture_guess follows the format of photutils.aperture 
    Args:
        data: the 2D image data
        aperture: the aperture shape [a, b, theta]
        center: the pixel center of the image, valid for specified aperture
        offset: the offset to the central pixel, valid for specified aperture
        aperture_guess: the initial guess for the 2d Gaussian fitting, [a, b, theta]
        debug: to plot the fitting and extract the radial profile
    """
    ny, nx = data.shape
    if center is not None:
        x0, y0 = center
    else:
        x0, y0 = nx*0.5, ny*0.5
    if offset is None:
        offset = [0,0]
      
    if aperture is None:
        # run a first global gaussian fitting to determine the annulus aperture
        if aperture_guess is not None:
            bmaj, bmin, theta0 = aperture_guess
            x_stddev = bmaj/2.355
            y_stddev = bmin/2.355
            # theta0 = (bpa + 90) * np.pi/180
        else:
            x_stddev=1
            y_stddev=1 
            theta0=0
        yidx, xidx = np.indices((ny, nx))
        yrad, xrad = yidx-ny/2., xidx-nx/2.
        # 
        # rescale the image to avoid numerical issue
        image_scale = np.max(data) / (0.5*(ny+nx))
        image_norm = data / image_scale
        xbounds = (-nx, nx)
        ybounds = (-ny, ny)
        x_mean = offset[0]
        y_mean = offset[1]
        p_init = models.Gaussian2D(amplitude=1, x_mean=x_mean, y_mean=y_mean, 
                x_stddev=x_stddev, y_stddev=y_stddev, theta=theta0, 
                bounds={"x_mean": xbounds, "y_mean": ybounds, 
                        "x_stddev":(0., nx/2.355), "y_stddev":(0., ny/2.355)})
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, xrad, yrad, image_norm,) 
                  # weights=1/(yrad**2+xrad**2+(0.25*x_stddev)**2+(0.25*y_stddev)**2))
        model_image = p(xrad, yrad)
        residual = image_norm - p(xrad, yrad)
        # establish the aperture based on the fitting
        b2a = p.y_fwhm/p.x_fwhm
        theta = p.theta.value
        offset = [p.x_mean, p.y_mean]
    else:
        a, b, theta = aperture
        b2a = b/a
        model_image = None
        residual = None
    # update the center with offsets
    x0 = x0 + offset[0]
    y0 = y0 + offset[1]
    if rmax is None:
        rmax = 0.5*(ny+nx)
    if debug:
        print(f'debug: rmin={rmin}, rmax={rmax}, step={step}')
    radii = np.arange(rmin, rmax, step)
    # aperture photometry
    flux_list = []
    flux_err_list = []
    aper_list = []
    radii_list = 0.5*(radii[1:]+radii[:-1])
    # fix a complain about zeros from EllipticalAnnulus
    if radii[0] < 1e-6:
        radii[0] = 1e-6
    for i in range(radii.size-1):
        aper = EllipticalAnnulus([x0, y0], radii[i], radii[i+1], b2a*radii[i+1], 
                                 theta=theta)
        aper_list.append(aper)
        flux_i, _ = aper.do_photometry(data,)
        area_i = aper.area_overlap(data)
        flux_list.append(flux_i[0]/area_i)
        aper_mask = aper.to_mask().to_image(data.shape).astype(int)
        flux_err_list.append(np.std(data[aper_mask]))
    
    if debug:
        print(f'x0={x0}, y0={y0}, b2a={b2a}, theta={theta}')
        fig, ax = plt.subplots(1,4, figsize=(12,3))
        ax[0].imshow(data, origin='lower')
        for aper in aper_list:
            aper.plot(ax=ax[0], color='red', alpha=0.2)
        if model_image is not None:
            ax[1].imshow(model_image, origin='lower')
        if residual is not None:
            ax[2].imshow(residual, origin='lower')
        if rmax is not None:
            for a in ax[:3]:
                a.set_xlim(x0-rmax, x0+rmax)
                a.set_ylim(y0-rmax, y0+rmax)
        ax[3].errorbar(radii_list, flux_list, yerr=flux_err_list)
    return [radii_list, flux_list, flux_err_list]

def image_stats(image, sigma=5.0, maxiters=2, sigma_clip=False):
    """this function first mask the 4-sigma signal and expand the mask with
       scipy.ndimage.binary_dilation
    """
    if sigma_clip:
        return stats.sigma_clipped_stats(image, sigma=sigma, maxiters=maxiters)
    else:
        signal_mask = find_structure(image, sigma=4.0, dilation_iters=3)
        return (np.nanmean(image[~signal_mask]), 
                np.nanmedian(image[~signal_mask]),
                np.nanstd(image[~signal_mask]))

def make_rgb(image_b, image_g, image_r, box=None, remove_signal=True,
             gamma=0.3, vmin=0, vmax=1, alpha=None):
    """
    Args:
        box: the selection box: [xlow, xup, ylow, yup]
    """
    if isinstance(gamma, (list,tuple)):
        if len(gamma) != 3:
            raise ValueError('gamma should include three numbers!')
        b_gamma, g_gamma, r_gamma = gamma
    else:
        b_gamma = g_gamma = r_gamma = gamma

    if isinstance(vmin, (int, float)):
        vmin = [vmin, vmin, vmin]
    if isinstance(vmax, (int, float)):
        vmax = [vmax, vmax, vmax]
    if box is not None:
        xlow, xup, ylow, yup = box
        box_select = np.s_[ylow:yup, xlow:xup]
    else:
        box_select = np.s_[:]
    r_tmp = image_r[box_select]
    g_tmp = image_g[box_select]
    b_tmp = image_b[box_select]
    
    if remove_signal:
        r_mean, r_median, r_std = image_stats(r_tmp)
        g_mean, g_median, g_std = image_stats(r_tmp)
        b_mean, b_median, b_std = image_stats(r_tmp)
    else: r_median=r_mean=g_median=g_mean=b_mean=b_median=0
    r_max, g_max, b_max = np.max(r_tmp), np.max(g_tmp), np.max(b_tmp)

    layer_r = (image_r - r_median) / (r_max-r_mean)
    layer_g = (image_g - g_median) / (g_max-g_mean)
    layer_b = (image_b - b_median) / (b_max-b_mean)

    layers = [layer_r, layer_g, layer_b]
    for i in range(3):
        layer = layers[i]
        layer[layer < vmin[i]] = vmin[i]
        layer[layer > vmax[i]] = vmax[i]
    if alpha is not None:
        if isinstance(alpha, (int, float)):
            layer_alpha = np.full_like(layer_r, fill_value=alpha)
        else:
            layer_alpha = alpha
        return np.array([layer_r.T**r_gamma, layer_g.T**g_gamma, 
                         layer_b.T**b_gamma, layer_alpha.T]).T
    return np.array([layer_r.T**r_gamma, layer_g.T**g_gamma, layer_b.T**b_gamma]).T

########################################
########### plot functions #############
########################################

def plot_pixel_image(image, beam=None, zoom_in=None,
                     ax=None, figname=None, fontsize=14, 
                     show_colorbar=True, show_axis=True,  
                     **kwargs):
    """plot image in the pixel coordinates
    Args:
        image: the 2D array
        beam: [bmaj, bmin, theta] # theta should be the angle (in degree) start with north and go counterclock-wise if the angle is positive.
        zoom_in: [xmin, xmax, ymin, ymax] in pixels

    It is useful for any manipulation needs the pixel coordinates 
    """
    imagesize = image.shape
    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
    im = ax.imshow(image, origin='lower', interpolation='none', **kwargs)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if beam is not None:
        ellipse = patches.Ellipse((0.1,0.1), transform=ax.transAxes,
                      width=beam[1]/imagesize[0], height=beam[0]/imagesize[1],
                      angle=beam[2], 
                      facecolor='orange', edgecolor='white', alpha=0.8)
        ax.add_patch(ellipse)
    if zoom_in is not None:
        ax.set_xlim((zoom_in[0], zoom_in[1]))
        ax.set_ylim((zoom_in[2], zoom_in[3]))
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        try: plt.close(fig)
        except: pass

def plot_image(image, beam=None, ax=None, name=None, pixel_size=1, unit=None, 
               contour=None, contour_levels=None, 
               vmin=None, vmax=None, zoom=1.0,
               show_colorbar=True, colorbar_powerlimits=None,
               show_axis=True, show_ruler=True, ruler_size=1,
               show_fwhm=True, show_labels=True,
               fontsize=12, figname=None, color_scales=None,
                **kwargs):
    """plot image in real space scale

    Args:
        image: 2D array
        beam: the size of the beam, in [arcsec, arcsec, degree]
    """
    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
    if isinstance(image, u.Quantity):
        unit = image.unit
        image = image.value
    if isinstance(pixel_size, (int, float)):
        pixel_size = [pixel_size, pixel_size]
    if name is not None:
        ax.set_title(name)
    ny, nx = image.shape
    # the central coodinate of each pixel, not the edge of the pixel
    x_index = (np.arange(0, nx) - (nx-1)/2.0) * pixel_size[0] # to arcsec
    y_index = (np.arange(0, ny) - (ny-1)/2.0) * pixel_size[1] # to arcsec
    #x_map, y_map = np.meshgrid(x_index, y_index)
    #ax.pcolormesh(x_map, y_map, data_masked)
    # extent = [np.max(x_index), np.min(x_index), np.min(y_index), np.max(y_index)]
    # the X-axis is negative, because we are inside the sky sphere
    extent = [0.5*nx*pixel_size[0], -0.5*nx*pixel_size[0], 
              -0.5*ny*pixel_size[1], 0.5*ny*pixel_size[1]]
    im = ax.imshow(image, origin='lower', extent=extent, interpolation='nearest', 
                   vmin=vmin, vmax=vmax, **kwargs)
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_powerlimits is None:
            colorbar_powerlimits = (0,0)
        cbar.formatter.set_powerlimits(colorbar_powerlimits)
    else:
        cbar = None
    if contour is not None:
        ax.contour(contour, colors='white', levels=contour_levels, extent=extent, 
                   linewidths=0.5, ) 
        # here need to remove origin='lower' due to different coordinate system in contour
    if show_fwhm:
        if beam is not None:
            # fits image angle start with north, go counterclock-wise if the angle is positive
            # pyplot ellipse, angle is from positive direction of x to positive direction of y
            # for astronomical image, the positive x-direction is left, see extent
            # that's why the angle is multiplied by -1
            ellipse = patches.Ellipse((0.8*np.max(x_index)/zoom, 0.8*np.min(y_index)/zoom), 
                                  width=beam[1], height=beam[0], 
                                  angle=-beam[-1],
                                  facecolor='grey', edgecolor='white', alpha=0.8)
            ax.add_patch(ellipse)
            # another way to show the beam size with photutils EllipticalAperture
            # same orientation as pyplot ellipse, but using radians
            #aper = EllipticalAperture((0.8*np.max(x_index)/zoom, 0.8*np.min(y_index)/zoom), 
            #                          0.5*beam[1], 0.5*beam[0], 
            #                          theta=-beam[-1]/180*np.pi)
            #aper.plot(color='white', lw=1)
    if show_labels:
        ax.set_xlabel('arcsec', fontsize=fontsize)
        ax.set_ylabel('arcsec', fontsize=fontsize)
 
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        try: plt.close(fig)
        except: pass
    
    # zoom the image
    ax.set_xlim(np.max(x_index)/zoom, np.min(x_index)/zoom)
    ax.set_ylim(np.min(y_index)/zoom, np.max(y_index)/zoom)
    
    if not show_axis:
        ax.axis('off')
        # show the ruler
        if show_ruler:
            ax.plot([0.82*np.min(x_index)/zoom, 0.82*np.min(x_index)/zoom], 
                    [-0.5*ruler_size, 0.5*ruler_size], linestyle='-', lw=1, color='grey')
            ax.text(0.92*np.min(x_index)/zoom, 0, f'{ruler_size}arcsec', ha='center', va='center', 
                    rotation=90, fontsize=fontsize*0.6, color='grey')

    return ax, cbar

def plot_sky_image(image, header, beam=None, vmin=None, vmax=None, 
                   fig=None, ax_string=111, fontsize=14,
                   show_colorbar=True, contour=None, contour_levels=None,
                   **kwargs):
    """plot the image with sky coordinate system
    """
    imagesize = image.shape
    image_wcs = WCS(header)
    pixel2arcsec = np.abs(header['CDELT1'])*3600
    if fig == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection=image_wcs, slices=('x', 'y', 0, 0))
    else:
        ax = fig.add_subplot(ax_string, projection=image_wcs, slices=('x', 'y', 0, 0))
    im = ax.imshow(image, interpolation='none', 
                   vmin=vmin, vmax=vmax, **kwargs)
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))
    else:
        cbar = None
    if contour is not None:
        ax.contour(contour, transform=ax.get_transform(image_wcs.celestial), 
                   colors='white', levels=contour_levels, linewidths=0.5)
    if beam is not None:
        ellipse = patches.Ellipse((0.1,0.1), transform=ax.transAxes,
                                  width=beam[1]/pixel2arcsec/imagesize[0], 
                                  height=beam[0]/pixel2arcsec/imagesize[0],
                                  angle=beam[2], 
                                  facecolor='orange', edgecolor='white', alpha=0.8)
        ax.add_patch(ellipse)

    if True:
        ax.set_xlabel('R.A.', fontsize=fontsize)
        ax.set_ylabel('Dec.', fontsize=fontsize)
    return ax, cbar

def plot_detections(image, detections, wcs=None, beam=None, pixel_size=None,
                    ax=None, figname=None, aperture_scale=1, 
                    contour=None, contour_levels=None, 
                    show_flux=False, 
                    show_detections_color='white',
                    **kwargs):
    """plot the detected sources on the image
    
    if wcs is available, it will use the skycoordinate, [ra, dec] in detections
    if wcs is none, it will take the pixelcoordinate, [x, y] in detections
    """
    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
    ax, cbar = plot_image(image, beam=beam, pixel_size=pixel_size, 
                          ax=ax, figname=figname,
                          **kwargs)
    # plot the detections
    ny, nx = image.shape
    pixel_center = np.array([(nx-1)/2.0, (ny-1)/2.0])
    ndets = len(detections)
    if wcs is not None:
        pass
    else:
        for i in range(ndets):
            det = detections[i]
            ydet = (det['y'] - pixel_center[1])*pixel_size
            xdet = -1.0*(det['x'] - pixel_center[0])*pixel_size
            aper_width = det['aper_maj']*aperture_scale #* pixel_rescale
            aper_height = det['aper_min']*aperture_scale #* pixel_rescale
            ellip_det = patches.Ellipse([xdet, ydet],
                    width=aper_width, height=aper_height, 
                    angle=180-det['theta']*180/np.pi, # positive x is the left 
                    facecolor=None, edgecolor=show_detections_color, alpha=0.8, 
                    fill=False)
            ax.add_patch(ellip_det)
            if show_flux:
                ax.text(xdet, ydet+yoffset, "{:.2f}mJy".format(det['flux']*1e3), 
                        color='white', horizontalalignment='center', 
                        verticalalignment='top', fontsize=fontsize)
    if True:
        ax.set_xlabel('arcsec', fontsize=16)
        ax.set_ylabel('arcsec', fontsize=16)
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        try: plt.close(fig)
        except: pass
    else:
        return ax, cbar
 
def plot_detections_old(image, pixel_sizes=[1,1], 
                    rms=None, color_scales=[-1,10], 
                    beam=None, show_fwhm=True,
                    xlim=None, ylim=None,
                    contour=None, show_colorbar=False, show_axis=True,
                    fig=None, ax=None, pixel_center=None,
                    figsize=(8,6), fontsize=14, figname=None, vmax=None, vmin=None,
                    detections=None, show_detections=True,  
                    aperture_scale=1, aperture_size=None,
                    show_detections_color='white', show_flux=False,
                    **kwargs):
    "modified plot function to show the detections"
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if rms is None:
        rms = calculate_rms(image)
    if vmin is None:
       vmin = color_scales[0] * rms 
    if vmax is None:
       vmax = color_scales[1] * rms

    ny, nx = image.shape
    center = [nx/2.-0.5, ny/2.0-0.5]
    if pixel_center is not None:
        diff_center = pixel_center - center
    else:
        diff_center = [0,0]
 
    x_index = (np.arange(0, nx) - (nx-1)/2.0+diff_center[0]) * pixel_sizes[0] # to arcsec
    y_index = (np.arange(0, ny) - (ny-1)/2.0+diff_center[1]) * pixel_sizes[1] # to arcsec

    if xlim is not None:
        xselection = (x_index >= xlim[0]) & (x_index <= xlim[-1])
        x_index = x_index[xselection]
        image = image[:,xselection]
    if ylim is not None:
        yselection = (y_index >= ylim[0]) & (y_index <= ylim[-1])
        y_index = y_index[yselection]
        image = image[yselection,:]
 
   
    extent = [np.max(x_index), np.min(x_index), np.min(y_index), np.max(y_index)]

    im = ax.imshow(image, origin='lower', extent=extent, 
                   vmax=vmax, vmin=vmin, **kwargs)

    if contour is not None:
        ax.contour(contour, levels=contour_levels, extent=extent, **contour_kwargs)
    #ax.pcolormesh(x_map, y_map, data_masked)
    # ax.invert_xaxis()
    if show_colorbar:
        cbar=plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter.set_powerlimits((0, 0))
    else:
        cbar = None
    if not show_axis:
        ax.axis('off')
    if show_fwhm:
        if beam is not None:
            # fits image angle start with north, go counterclock-wise if the angle is positive
            # pyplot ellipse, angle is from positive direction of x to positive direction of y
            # for astronomical image, the positive x-direction is left, see extent
            # that's why the angle is multiplied by -1
            ellipse = patches.Ellipse((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                  width=beam[1], height=beam[0], 
                                  angle=-beam[-1],
                                  facecolor='tomato', edgecolor='none', alpha=0.8)
            ax.add_patch(ellipse)
            # another way to show the beam size with photutils EllipticalAperture
            # aper = EllipticalAperture((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                     # 0.5*beam[1], 0.5*beam[0], 
                                     # theta=-beam[-1]/180*np.pi)
            # aper.plot(color='red', lw=1)
            # however, the beam relise on the original transAxes, then the angle
            # should be keep the same

    if show_detections and (detections is not None):
        for idx,det in enumerate(detections):
            xdet = (-det['x']+center[0]-diff_center[0]) * pixel_sizes[0]
            ydet = (det['y']-center[1]+diff_center[1]) * pixel_sizes[1]
            try: 
                #TODO: improve to support non-regular x, y pixels
                aper_width = det['aper_maj']*2.*pixel_sizes[0]
                aper_height = det['aper_min']*2.*pixel_sizes[1]
            except:
                if aperture_scale is not None:
                    aper_width = det['a']*2.*aperture_scale #* pixel_rescale
                    aper_height = det['b']*2.*aperture_scale #* pixel_rescale
                if aperture_size is not None:
                    if isinstance(aperture_size, (list, tuple, np.ndarray)):
                        aper_width, aper_height = aperture_size
                    else:
                        aper_width = aperture_size
                        aper_height = aperture_size
            ellipse_det = patches.Ellipse((xdet, ydet), 
                    width=aper_width, height=aper_height, 
                    angle=180-det['theta']*180/np.pi, # positive x is the left 
                    facecolor=None, edgecolor=show_detections_color, alpha=0.8, 
                    fill=False)
            ax.add_patch(ellipse_det)
            # ellipse_det = EllipticalAperture([xdet, ydet], det['a']*aperture_scale, 
                                             # det['b']*aperture_scale, det['theta'])
            # ellipse_det.plot(color='white', lw=1, alpha=0.8)
            # if 'x' in show_detections_yoffset:
                # yoffset = float(show_detections_yoffset[:-1])
            # else:
                # yoffset = show_detections_yoffset
            if show_flux:
                ax.text(xdet, ydet+yoffset, "{:.2f}mJy".format(det['flux']*1e3), color='white',
                        horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
    
    if True:
        ax.set_xlabel('arcsec', fontsize=16)
        ax.set_ylabel('arcsec', fontsize=16)
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        try: plt.close(fig)
        except: pass
    else:
        return ax, cbar
 

########################################
################ tests #################
########################################

