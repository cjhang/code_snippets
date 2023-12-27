#!/usr/bin/env python3
"""A minimalist tool to deal with fits images

Author: Jianhang Chen, cjhastro@gmail.com
History:
    2022-01-11: first release to handle fits images from CASA


Notes:
    1. Throughout the code, the aperture are follow the `photoutils`
    2. The beam follow the rule of ALMA: [bmaj, bmin, theta], theta is in degree, 
       from north (top) to east (left).

Requirement:
    numpy
    matplotlib
    astropy >= 5.0
    photutils >= 1.0
    sep (optional, used for source_finder)
"""

__version__ = '1.0.8'

import os
import sys
import numpy as np
import scipy.ndimage as ndimage
from scipy import optimize
import warnings
from astropy.io import fits
from astropy.table import Table, vstack, hstack
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants as const
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.modeling import models, fitting
from astropy import convolution 

from photutils import aperture_photometry, find_peaks, EllipticalAperture, RectangularAperture, SkyEllipticalAperture

# Filtering warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

##############################
######### Image ##############
##############################

class BaseImage(object):
    """Basic image class
    """
    def __init__(self):
        pass

class Image(BaseImage):
    """The data strcuture to handle the 2D astronomical images from CASA
    """
    def __init__(self, data=None, header=None, beam=None, wcs=None, mask=None,
                 name=None):
        """initialize the image

        Args:
            data: the image data, with or without units
            header: the header can be identified by astropy.wcs.WCS
            beam: the beam shape in the units of [arcsec, arcsec, deg]
            wcs (optional): another way to set celestial reference
            mask (optional): same shape as the data
        """
        self.data = data
        self.header = header
        self.wcs = wcs
        self.beam = beam
        self.mask = mask
        self.name = name 
        if self.header is not None:
            if self.wcs is None:
                self.update_wcs()
        if self.header is None:
            if self.wcs is not None:
                self.header = self.wcs.to_header()
        self.pixel2arcsec = None
        self.pixel_beam = None
        self.set_pixel_beam()
    def __getitem__(self, i):
            return self.image.value[i]
    @property
    def info(self):
        # shortcut for print all the basic info
        print(f"Shape: {self.shape}")
        print(f"Units: {self.image.unit}")
        print(f"Pixel size {self.pixel_sizes}")
        print(f"Beam {self.beam}")
        print(f"Pixel beam: {self.pixel_beam}")
    @property
    def image(self):
        if np.ndim(self.data) > 2:
            return self.data.reshape(self.data.shape[:2])
        else:
            return self.data
    @property
    def unit(self):
        if isinstance(self.image, u.Quantity):
            return self.image.unit
        else: return None
    @property
    def shape(self):
        return self.image.shape
    @property
    def pixel_sizes(self):
        if self.header is not None:
            # Return the pixel size encoded in the header
            # In casa, it is in the units of deg, thus the returned value is pixel to deg
            pixel2arcsec_ra = (abs(self.header['CDELT1'])*u.Unit(self.header['CUNIT1'])).to(u.arcsec)
            pixel2arcsec_dec = (abs(self.header['CDELT2'])*u.Unit(self.header['CUNIT2'])).to(u.arcsec)
        elif self.pixel2arcsec is not None:
            pixel2arcsec_ra, pixel2arcsec_dec = self.pixel2arcsec, self.pixel2arcsec
        else:
            pixel2arcsec_ra, pixel2arcsec_dec = 1, 1
        return u.Quantity([pixel2arcsec_ra, pixel2arcsec_dec])
    @property
    def beamsize(self):
        # calculate the beamszie in number of pixels
        # the final unit is pixels/beam, convert the Jy/beam to Jy/pixel need to divive the beam
        pixel_area = self.pixel_sizes[0].to('arcsec').value * self.pixel_sizes[1].to('arcsec').value
        return calculate_beamsize(self.beam, scale=1/pixel_area)
        # return 1/(np.log(2)*4.0) * np.pi * self.beam[0] * self.beam[1] / pixel_area 
    def set_pixel_beam(self):
        # convert the beam size into pixel sizes
        if self.beam is None:
            warnings.warn('No beam infomation has been found!')
            # raise ValueError("No valid beams can be found!")
        else:
            try:
                bmaj, bmin, bpa = self.beam
                x_scale = 1/self.pixel_sizes[0].to(u.arcsec).value
                y_scale = 1/self.pixel_sizes[1].to(u.arcsec).value
                bmaj_pixel = np.sqrt((bmaj*np.cos(bpa/180*np.pi)*x_scale)**2 
                                     + (bmaj*np.sin(bpa/180*np.pi)*y_scale)**2)
                bmin_pixel = np.sqrt((bmin*np.sin(bpa/180*np.pi)*x_scale)**2 
                                     + (bmin*np.cos(bpa/180*np.pi)*y_scale)**2)
                self.pixel_beam = [bmaj_pixel, bmin_pixel, bpa]
            except:
                pass
    @property
    def imstats(self):
        return sigma_clipped_stats(self.image, sigma=5.0, maxiters=2)
    def beam_stats(self, beam=None, mask=None, nsample=100):
        if beam is None:
            beam = self.pixel_beam
        if mask is None:
            mask = self.mask
        aperture = beam2aperture(beam)
        return aperture_stats(self.image, aperture=aperture, 
                              mask=mask, nsample=nsample)
    def update_wcs(self, wcs=None, header=None):
        if wcs is not None:
            self.wcs = wcs
        elif header is not None:
            self.wcs = WCS(header).sub(['longitude','latitude'])
        else:
            self.wcs = WCS(self.header).sub(['longitude','latitude'])

    def correct_beam(self, inverse=False):
        """correct the flux per beam to per pixel
        """
        if self.beam is not None:
            if inverse:
                if 'beam' not in self.unit.to_string():
                    self.data = self.data*self.beamsize/u.beam
                    self.unit = u.Unit(self.unit.to_string()+'/beam')
                else:
                    pass
            else:
                if 'beam' in self.unit.to_string():
                    self.data = self.data/self.beamsize*u.beam

    def correct_reponse(self, corr):
        """apply correction for the whole map
        """
        self.data = self.data * corr

    def subimage(self, s_):
        """extract subimage from the orginal image

        Args:
            s_ (:obj:`slice`): data slice, same shape as image

        Return:
            :obj:`Image`
        """
        image_sliced = self.image[s_].copy()
        shape_sliced = image_sliced.shape
        wcs_sliced = self.wcs[s_].deepcopy()
        # TODO: temperary solution, more general solution needs to consider the 
        # shift the reference center, 
        wcs_sliced.wcs.crpix[0] = (s_[-1].start + s_[-1].stop)//2
        wcs_sliced.wcs.crpix[1] = (s_[-2].start + s_[-2].stop)//2
        wcs_sliced.wcs.crval = self.wcs.wcs.crval + self.wcs.wcs.pc.dot(wcs_sliced.wcs.crpix - self.wcs.wcs.crpix) * self.wcs.wcs.cdelt
        wcs_sliced.wcs.crpix[0] -= s_[-1].start 
        wcs_sliced.wcs.crpix[1] -= s_[-2].start
        header_sliced = wcs_sliced.to_header()
        naxis = len(shape_sliced)
        header_sliced.set('NAXIS',naxis)
        header_sliced.set('NAXIS1',shape_sliced[-1])
        header_sliced.set('NAXIS2',shape_sliced[-2])
        return Image(image_sliced, header_sliced, self.beam)

    def pixel2skycoords(self, pixel_coords):
        """covert from pixel to skycoords

        Args:
            pixel_coords: pixel coordinates, in the format of [[x1,y1], [x2,y2],...]
        """
        pixel_coords = np.array(pixel_coords).T
        return pixel_to_skycoord(*pixel_coords, self.wcs)

    def skycoords2pixels(self, skycoords):
        if skycoords.size == 1:
            return np.array(skycoord_to_pixel(skycoords, self.wcs))
        else:
            return np.array(list(zip(*skycoord_to_pixel(skycoords, self.wcs))))
    
    def update_mask(self, mask=None, mask_invalid=True):
        newmask = np.zeros(self.imagesize, dtype=bool)
        if mask_invalid:
            image_masked = np.ma.masked_invalid(self.data)
            invalid_mask = image_masked.mask.reshape(self.imagesize)
            newmask = newmask | invalid_mask
        if mask is not None:
            mask = newmask | mask
        self.mask = mask

    def plot(self, image=None, name=None, ax=None, figsize=(8,6), 
             contour=None, contour_levels=None, contour_kwargs={'colors':'white'},
             sky_center=None, pixel_center=None, fov=0, 
             vmax=None, vmin=None, vmax_scale=10, vmin_scale=-3,
             show_center=False, show_axis=True, show_fwhm=True, show_fov=False,
             show_rms=False, show_flux=False, show_sky_sources=[], show_pixel_sources=[],
             show_detections=False, detections=None, aperture_scale=1.0, aperture_size=None, 
             show_detections_yoffset='-1x', show_detections_color='white', fontsize=12, 
             figname=None, show_colorbar=True, xlim=None, ylim=None, **kwargs):
        """general plot function build on FitsImage

        Parameters
        ----------
            center : list
                [xcenter, ycenter]
            aperture_scale: in the units of semi-major axis of the detection if applicable
            aperture_size: the aperture_size in units of arcsec
        """
        # build-in image visualization function
        if ax == None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        if name is None:
            name = self.name
        if image is None:
            if isinstance(self.image, u.Quantity):
                image = self.image.value
            else: image = self.image
        if name is not None:
            ax.set_title(name)
        ny, nx = self.shape
        ny, nx = image.shape
        if sky_center is not None:
            center = self.skycoords2pixels(sky_center)
        elif pixel_center is not None:
            center = pixel_center
        else:
            center = [nx/2., ny/2.0]
        x_index = (np.arange(0, nx) - center[0]) * self.pixel_sizes[0].value # to arcsec
        y_index = (np.arange(0, ny) - center[1]) * self.pixel_sizes[1].value # to arcsec
        if xlim is not None:
            xselection = (x_index >= xlim[0]) & (x_index <= xlim[-1])
            x_index = x_index[xselection]
            image = image[:,xselection]
        if ylim is not None:
            yselection = (y_index >= ylim[0]) & (y_index <= ylim[-1])
            y_index = y_index[yselection]
            image = image[yselection,:]
        extent = [np.max(x_index), np.min(x_index), np.min(y_index), np.max(y_index)]
        mean, median, std = self.imstats
        if vmax is None:
            if std is not None:
                if self.unit is not None:
                    vmax = vmax_scale * std.value
                else:
                    vmax = vmax_scale * std
        if vmin is None:
            if std is not None:
                if self.unit is not None:
                    vmin = vmin_scale * std.value
                else:
                    vmin = vmin_scale * std
        im = ax.imshow(image, origin='lower', extent=extent, 
                       vmax=vmax, vmin=vmin, **kwargs)
        if contour is not None:
            ax.contour(contour, levels=contour_levels, extent=extent, **contour_kwargs)
        #ax.pcolormesh(x_map, y_map, data_masked)
        # ax.invert_xaxis()
        if show_colorbar:
            cbar=plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            cbar = None
        if show_center:
            ax.text(0, 0, '+', color='r', fontsize=24, fontweight=100, horizontalalignment='center',
                    verticalalignment='center')
        if not show_axis:
            ax.axis('off')
        if show_fwhm:
            if self.beam is not None:
                # fits image angle start with north, go counterclock-wise if the angle is positive
                # pyplot ellipse, angle is from positive direction of x to positive direction of y
                # for astronomical image, the positive x-direction is left, see extent
                ellipse = patches.Ellipse((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                      width=self.beam[1], height=self.beam[0], 
                                      angle=-self.beam[-1],
                                      facecolor='orange', edgecolor=None, alpha=0.8)
                ax.add_patch(ellipse)
                # another way to show the beam size with photutils EllipticalAperture
                #aper = EllipticalAperture((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                #                          0.5*self.beam[1], 0.5*self.beam[0], 
                #                          theta=-self.beam[-1]/180*np.pi)
                #aper.plot(color='white', lw=2)

        if show_fov:    
            ellipse_fov = patches.Ellipse((0, 0), width=fov, height=fov, 
                                    angle=0, fill=False, facecolor=None, edgecolor='gray', 
                                    alpha=0.8)
            ax.add_patch(ellipse_fov)
        if show_rms:
            ax.text(0.6*np.max(x_index), 0.8*np.min(y_index), 'rms={:.2f}'.format(self.std.to(u.uJy/u.beam)), 
                    fontsize=0.8*fontsize, horizontalalignment='center', verticalalignment='top', color='white')
        if show_detections:
            for idx,det in enumerate(detections):
                xdet = (-det['x']+center[0])*self.pixel_sizes[0].value
                ydet = (det['y']-center[1])*self.pixel_sizes[1].value
                try: 
                    #TODO: improve to support non-regular x, y pixels
                    aper_width = det['aper_maj']*2.*self.pixel_sizes[0].value
                    aper_height = det['aper_min']*2.*self.pixel_sizes[1].value
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
                        facecolor=None, edgecolor=show_detections_color, alpha=0.8, fill=False)
                ax.add_patch(ellipse_det)
                # ellipse_det = EllipticalAperture([xdet, ydet], det['a']*aperture_scale, 
                                                 # det['b']*aperture_scale, det['theta'])
                # ellipse_det.plot(color='white', lw=1, alpha=0.8)
                if 'x' in show_detections_yoffset:
                    yoffset = float(show_detections_yoffset[:-1])
                else:
                    yoffset = show_detections_yoffset
                if show_flux:
                    ax.text(xdet, ydet+yoffset, "{:.2f}mJy".format(det['flux']*1e3), color='white',
                            horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
        if show_sky_sources != []:
            try:
                n_sky_sources = len(show_sky_sources)
            except:
                n_sky_sources = 1
            if n_sky_sources == 1:
                show_sky_sources = [show_sky_sources,]
            for s in show_sky_sources:
                pixel_sources = skycoord_to_pixel(s, self.wcs)
                xs = (-pixel_sources[0]+nx/2.0)*self.pixel2deg_ra*3600
                ys = (pixel_sources[1]-ny/2.0)*self.pixel2deg_dec*3600
                if (abs(xs)>1.5*image_boundary) or (abs(ys)>1.5*image_boundary):
                    print("Warning: skycoords outside the images!")
                    continue
                ax.text(xs, ys, 'x', fontsize=24, fontweight=100, horizontalalignment='center',
                    verticalalignment='center', color='white', alpha=0.5)
        if show_pixel_sources != []:
            n_pixel_sources = len(show_pixel_sources)
            if n_pixel_sources == 1:
                show_pixel_sources = [show_pixel_sources,]
            for s in show_pixel_sources:
                xp = (-s[0]+nx/2.0)*self.pixel_sizes[0].value
                yp = (s[1]-ny/2.0)**self.pixel_sizes[0].value
                if (abs(xp)>1.5*image_boundary) or (abs(yp)>1.5*image_boundary):
                    print("Warning: pixel outside the images!")
                    continue
                ax.text(xp, yp, 'x', fontsize=24, fontweight=100, horizontalalignment='center',
                    verticalalignment='center', color='white', alpha=0.5)
        if figname is not None:
            fig.savefig(figname, bbox_inches='tight', dpi=200)
            plt.close(fig)
        else:
            return im,cbar
    
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
        mean, median, std = sigma_clipped_stats(self.image, sigma=sigma,
                maxiters=1, mask=self.mask)
        struct_above_sigma = self.image > sigma*std
        if opening_iters is None:
            opening_iters = iterations
        if dilation_iters is None:
            dilation_iters = iterations
        sigma_structure = struct_above_sigma
        if opening_iters > 0:
            sigma_structure = ndimage.binary_opening(sigma_structure, iterations=opening_iters)
        if dilation_iters > 0:
            sigma_structure = ndimage.binary_dilation(sigma_structure, iterations=dilation_iters)
        if plot:
            self.plot(image=sigma_structure, name=self.name+'_{}_structures'.format(sigma), 
                      **kwargs)
        return sigma_structure

    def beam2aperture(self, scale=1):
        """shortcut to convert beam to apertures

        The major difference is angle. In beam, it is degree; while in aperture it is radian.
        """
        return beam2aperture(self.pixel_beam, scale=scale)

    def source_finder(self, **kwargs):
        """class wrapper for source_finder
        """
        mean, median, std = self.imstats
        return source_finder(self.image.value, wcs=self.wcs, std=std.value, name=self.name, 
                                      beam=self.pixel_beam, **kwargs)

    def measure_flux(self, dets=None, coords=None, apertures=None, 
                     aperture_scale=4.0, segment_scale=3.0,
                     method='gaussian', minimal_aperture=None, **kwargs):
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
        flux_table = measure_flux(self.image.value, wcs=self.wcs,
                                  coords=coords, apertures=apertures, method=method, 
                                   segment_size=segment_scale*np.max(self.pixel_beam[0]),
                                     **kwargs) 
        flux_table['flux'] = flux_table['flux']*self.unit*corr
        flux_table['flux_err'] = flux_table['flux_err']*self.unit*corr
        return flux_table

    def fit_2Dgaussian(self, **kwargs):
        return gaussian_2Dfitting(self.image, **kwargs)

    def writefits(self, filename, overwrite=False, shift_reference=False):
        """write to fits file
        
        This function also shifts the reference pixel of the image to the image center
        """
        ysize, xsize = self.shape
        header = self.header
        if header is None:
            if self.wcs is not None:
                header = wcs.to_header()
            else: # create a new
                header = fits.Header()
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
        if self.beam is not None:
            # follow the same rule as CASA, which use the units of deg
            header['BMAJ'] = self.beam[0] / 3600.
            header['BMIN'] = self.beam[1] / 3600.
            header['BPA'] = self.beam[2]
        if self.unit is not None:
            header.set('BUNIT', self.unit.to_string())
            imagehdu = fits.PrimaryHDU(data=self.image.value, header=header)
        else:
            imagehdu = fits.PrimaryHDU(data=self.image, header=header)
        header.update({'history':'created by image_tools.Image',})
        imagehdu.writeto(filename, overwrite=overwrite)

    def correct_pb(self, coords, pbfile=None, pbcorfile=None):
        """read the primary beam correction from the primary beam profile
        """
        pbimage = solve_impb(pbfile)

    def reproject(self, wcs_out=None, header_out=None, shape_out=None, **kwargs):
        """reproject the data into another wcs system
        
        """
        try: import reproject
        except: raise ValueError("Package `reproject` must be installed to handle the reprojection!")
        if header_out is not None:
            data_new, footprint = reproject.reproject_interp((self.data.value, self.wcs), header_out, 
                                        return_footprint=True,**kwargs)
        if wcs_out is not None:
            if shape_out is None:
                raise ValueError('the `shape_out` must be provide when use projection with WCS!')
            data_new, footprint = reproject.reproject_interp((self.data.value, self.wcs), wcs_out, 
                                        shape_out=shape_out, return_footprint=True, **kwargs)
        return Image(data=data_new, header=header_out, wcs=wcs_out)
    
    @staticmethod
    def read(fitsimage, extname='primary', name=None, debug=False, correct_beam=False):
        """read the fits file

        Parameters
        ----------
        fitsimage : string
            the filename of the fits file.
        name (optinal): 
            the name of the image.
        debug : 
            set to true to print the details of the fits file
        correct_beam: 
            set to True to correct the beams if the beam 
            information is available
        """
        with fits.open(fitsimage) as image_hdu:
            if debug:
                print(image_hdu.info())
            image_header = image_hdu[extname].header
            image_data = image_hdu[extname].data * u.Unit(image_header['BUNIT'])
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
        if name is None:
            name = os.path.basename(fitsimage)
        return Image(data=image_data, header=image_header, beam=image_beam, name=name)

    @staticmethod
    def read_ALMA(fitsimage, extname='primary', name=None, debug=False, 
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
            image_data = image_hdu[extname].data[stokes_idx,spec_idx] * u.Unit(image_header['BUNIT'])
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
        if name is None:
            name = os.path.basename(fitsimage)
        return Image(data=image_data, header=image_header, beam=image_beam, name=name)

########################################
###### stand alone functions ###########
########################################

def calculate_rms(data, mask):
    data = np.ma.masked_array(data, mask=mask)
    return np.sqrt(np.ma.sum(data**2)/(data.size-np.sum(mask)))

def beam2aperture(beam, scale=1):
    """cover the beam in interferometric image to aperture in `photutils`

    The major difference is the reference axis:
    beam: the position angle is relative to north
    aperture: the angle is from positive x-axis to positive y-axis
    """
    scale_array = np.array([scale, scale, 1])
    return (scale_array*np.array([beam[1], beam[0], beam[-1]]) * (0.5, 0.5, np.pi/180)).tolist()

def calculate_beamsize(beam, scale=1):
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

    Return: [mean, median, std]
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
    else:
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
    yshape, xshape = image.shape
    ygrid, xgrid = np.meshgrid((np.arange(0, yshape)+0.5)*pixel_size,
                               (np.arange(0, xshape)+0.5)*pixel_size)
    # automatically estimate the initial parameters
    if x0 is None:
        amp = np.percentile(image, 0.9)*1.1
        x0, y0 = xshape*0.5, yshape*0.5
        xsigma, ysigma = pixel_size, pixel_size 
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
    return flux, flux_err 

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
    flux_fitted = 2*np.max(image)*np.pi*p.x_stddev.value*p.y_stddev.value*p.amplitude.value
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

def find_structure(image, mask=None, sigma=3.0, iterations=1, opening_iters=None, 
                   dilation_iters=None, plot=False):
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
        mean, median, std, rms = beam_stats(image, mask=mask)
        struct_above_sigma = self.image > sigma*std
        if opening_iters is None:
            opening_iters = iterations
        if dilation_iters is None:
            dilation_iters = iterations
        sigma_structure = struct_above_sigma
        if opening_iters > 0:
            sigma_structure = ndimage.binary_opening(sigma_structure, iterations=opening_iters)
        if dilation_iters > 0:
            sigma_structure = ndimage.binary_dilation(sigma_structure, iterations=dilation_iters)
        # if plot:
            # self.plot(image=sigma_structure, name=self.name+'_{}_structures'.format(sigma), 
                      # **kwargs)
        return sigma_structure

def source_finder(image=None, wcs=None, std=None, mask=None, 
                  beam=None, aperture_scale=6.0, 
                  detection_threshold=3.0, 
                  plot=False, name=None, show_flux=False, 
                  method='sep', filter_kernel=None, # parameters for sep
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
        print(sources_found)

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
            ellipse = patches.Ellipse((obj['x'], obj['y']), width=6*obj['a'], height=6*obj['b'], 
                                      angle=obj['theta']*180/np.pi, facecolor='none', edgecolor='black', 
                                      alpha=0.5)
            ax.add_patch(ellipse)
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

def measure_flux(image, coords=None, wcs=None,
                 method='single-aperture', apertures=None,
                 mask=None, n_boostrap=100,
                 segment_size=21.0, noise_fwhm=None, rms=None,
                 plot=False, ax=None, color='white', debug=False):
    """Two-dimension flux measurement in the pixel coordinates
    
    It supports three ways to provide the coordinates of the detection. The ``detections`` allows
    table-like input, while the pixel_coords and sky_coords allow direct coordinates.
    The aperture are provided by the aperture and aperture_scale. ``aperture`` provides the shape
    while aperture_scale constraints the size.

    Args:
        image: the data with or without units
        coords: the pixel coordinates of the detections, [[x1,y1], [x2,y2]]
        aperture: the fixed size of the aperture, in pixels
        segment_size: segment size to cut out the image. which is used for gaussian fitting.LevMarLSQFitter
                      noly support one value, square cutout.

    """
    imagesize = image.shape
    if isinstance(apertures, EllipticalAperture):
        detections_apers = apertures
    else: # make new apertures with the shape parameters
        detections_apers = make_apertures(coords, shapes=apertures)
    # convert single item into list
    if isinstance(detections_apers, EllipticalAperture):
        detections_apers = [detections_apers,]
    if np.ndim(coords) == 1:
        coords = [coords,]
    detections_mask = mask_coordinates(image, apertures=detections_apers)
    n_sources = len(detections_apers)
    
    # create the table to record the results
    table_flux = Table(names=('ID','x','y','aper_maj','aper_min','theta','flux','flux_err'), 
                         dtype=('i4','f8','f8','f8','f8','f8','f8','f8'))
    if method == 'single-aperture':
        # measuring flux density
        for i,aper in enumerate(detections_apers):
            x,y = coords[i]
            phot_table = aperture_photometry(image, aper, mask=mask)
            # aperture_correction
            flux = phot_table['aperture_sum'].value
            _, _, flux_err, _ = aperture_stats(image, (aper.a, aper.b, aper.theta), 
                                            mask=detections_mask)
            table_flux.add_row((i, x, y, aper.a, aper.b, aper.theta, flux, flux_err))
   
    if method == 'gaussian':

        segments = RectangularAperture(coords, segment_size, segment_size, theta=0)
        segments_mask = segments.to_mask(method='center')
        for i,s in enumerate(segments_mask):
            x,y = coords[i]
            image_cutout = s.cutout(image)
            gaussian_fitting = gaussian_2Dfitting(image_cutout, debug=debug, plot=plot)
            flux = gaussian_fitting['flux'] 
            if rms is None:
                rms = calculate_rms(image, mask=detections_mask)
            # caulcate the error: 
            
            # method1: boostrap for noise measurement
            a_fitted_aper = 3 * gaussian_fitting['x_stddev'] # 2xFWHM of gaussian
            b_fitted_aper = 3 * gaussian_fitting['y_stddev']
            theta_fitted = gaussian_fitting['theta']/180*np.pi
            # _, _, flux_err,_ = aperture_stats(image, (a_fitted_aper, b_fitted_aper,
                                                                # theta_fitted),
                                                        # mask=detections_mask)
            # method 2: condon+1997
            gf = gaussian_fitting
            sigma2FWHM = np.sqrt(8*np.log(2))
            amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = (
                    gf['amplitude'],gf['x_mean'],gf['y_mean'],gf['x_stddev'],gf['y_stddev'],gf['theta'])
            xfwhm_fit, yfwhm_fit = xsigma_fit*sigma2FWHM, ysigma_fit*sigma2FWHM
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


            table_flux.add_row((i, x, y, a_fitted_aper, b_fitted_aper, theta_fitted, 
                                  flux, flux_err))
    if wcs is not None:
        n_dets = len(table_flux)
        table_new_data = Table(np.array([0., 0.] * n_dets).reshape(n_dets, 2),
                           names={'ra', 'dec'})
        table_flux = hstack([table_new_data, table_flux])
        # convert the pixel coordinates to sky coordinates
        for i in range(n_dets):
            obj = table_flux[i]
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest', origin='lower')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        for i in range(n_sources):
            obj = table_flux[i]
            ax.text(obj['x'], obj['y'], i, color='red', horizontalalignment='center', 
                    verticalalignment='center')
            ellipse = patches.Ellipse((obj['x'], obj['y']), 
                                      width=2*obj['aper_maj'], height=2*obj['aper_min'], 
                                      angle=obj['theta']*180/np.pi, 
                                      facecolor='none', edgecolor='black', alpha=0.5)
            vert_dist = 0.8*(obj['aper_maj']*np.cos(obj['theta']) + obj['aper_min']*np.cos(obj['theta']))
            ax.add_patch(ellipse)
            ax.text(obj['x'], obj['y']-abs(vert_dist), "{:.2f}".format(obj['flux']), 
                    color=color, horizontalalignment='center', verticalalignment='center',)
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
        y_offset, x_offset = offset
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
               pixel_coords=None, sky_coords=None,):
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
        pbdata = data/pbcordata
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
        return Image(image=pbdata, header=header, name='pb')
    pbcor_values = []
    for coord in pixel_coords:
        coord_int = np.round(coord).astype(int)
        pbcor_values.append(pbdata[coord_int[1], coord_int[0]])
    # return Image(image=pbdata, header=header, name='pb')
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
    mean, median, std = sigma_clipped_stats(image, sigma=sigma, maxiters=3)
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


########################################
########### plot functions #############
########################################

def plot_pixel_image(image, beam=None, ax=None, figname=None, 
                     show_colorbar=True, show_axis=True, show_fwhm=True, show_center=True,
                     **kwargs):
    imagesize = image.shape
    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
    im = ax.imshow(image, origin='lower', interpolation='none', **kwargs)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if show_center:
        ax.text(imagesize[0]/2, imagesize[1]/2, '+', color='r', fontsize=24, fontweight=100, 
                horizontalalignment='center', verticalalignment='center')
    if not show_axis:
        ax.axis('off')
    if show_fwhm:
        if beam is not None:
            # fits image angle start with north, go counterclock-wise if the angle is positive
            # pyplot ellipse, angle is from positive direction of x to positive direction of y
            # for astronomical image, the positive x-direction is left, see extent
            ellipse = patches.Ellipse((0.1*imagesize[0], 0.1*imagesize[1]), 
                                  width=beam[1], height=beam[0], 
                                  angle=beam[-1],
                                  facecolor='orange', edgecolor=None, alpha=0.8)
            ax.add_patch(ellipse)
            # another way to show the beam size with photutils EllipticalAperture
            aper = EllipticalAperture((0.1*imagesize[0], 0.1*imagesize[1]), 
                                      0.5*beam[1], 0.5*beam[0], 
                                      theta=beam[-1]/180*np.pi)
            aper.plot(color='white', lw=2)
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        plt.close(fig)

def plot_image(image, ax=None, name=None, pixel_size=1, 
                show_colorbar=True, show_center=False, show_axis=True, show_fwhm=True,
                **kwargs):
    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
    if isinstance(self.image, u.Quantity):
        image = self.image.value
    if isinstance(pixel_size, (int, float)):
        pixel_size = [pixel_size, pixel_size]
    ax.set_title(name)
    ny, nx = image.shape
    x_index = (np.arange(0, nx) - nx/2.0) * pixel_sizes[0] # to arcsec
    y_index = (np.arange(0, ny) - ny/2.0) * pixel_sizes[1] # to arcsec
    #x_map, y_map = np.meshgrid(x_index, y_index)
    #ax.pcolormesh(x_map, y_map, data_masked)
    extent = [np.max(x_index), np.min(x_index), np.min(y_index), np.max(y_index)]
    im = ax.imshow(image, origin='lower', extent=extent, interpolation='none', 
                   **kwargs)
    # ax.invert_xaxis()
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if show_center:
        ax.text(0, 0, '+', color='r', fontsize=24, fontweight=100, horizontalalignment='center',
                verticalalignment='center')
    if not show_axis:
        ax.axis('off')
    if show_fwhm:
        if self.beam is not None:
            # fits image angle start with north, go counterclock-wise if the angle is positive
            # pyplot ellipse, angle is from positive direction of x to positive direction of y
            # for astronomical image, the positive x-direction is left, see extent
            ellipse = patches.Ellipse((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                  width=self.beam[1], height=self.beam[0], 
                                  angle=-self.beam[-1],
                                  facecolor='orange', edgecolor=None, alpha=0.8)
            ax.add_patch(ellipse)
        # another way to show the beam size with photutils EllipticalAperture
        aper = EllipticalAperture((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                  0.5*self.beam[1], 0.5*self.beam[0], 
                                  theta=-self.beam[-1]/180*np.pi)
        aper.plot(color='white', lw=2)
 
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight', dpi=200)
        plt.close(fig)
  
