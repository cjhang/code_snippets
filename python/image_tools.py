#!/usr/bin/env python3
"""A minimalist tool to deal with fits images

Author: Jianhang Chen, cjhastro@gmail.com
History:
    2022-01-11: first release to handle fits images from CASA

Requirement:
    numpy
    matplotlib
    astropy >= 5.0
    photutils >= 1.0
    sep (optional, used for source_finder)
"""

import os
import sys
import numpy as np
import scipy.ndimage as ndimage
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

class Image(object):
    """The base data strcuture to handle the 2D astronomical images
    """
    def __init__(self, data=None, header=None, beam=None, wcs=None, mask=None,
                 name=None):
        """initialize the image

        Args:
            data: the image data, with or without units
            header: the header can be identified by astropy.wcs.WCS
            beams: the beam shape in the units of [arcsec, arcsec, deg]
            wcs (optional): another way to set celestial reference
            mask (optional): same shape as the data
        """
        self.data = data
        self.shape = self.data.shape
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
        try:
            self.mean, self.median, self.std = sigma_clipped_stats(self.image, sigma=5.0, mask=self.mask)
        except:
            print("Cannot derive the statistics of the image!")
            pass

    def __getitem__(self, i):
            return self.data[i]
    @property
    def info(self):
        # shortcut for print all the basic info
        print(f"The shape of the data {self.data.shape}")
    @property
    def unit(self):
        if isinstance(self.data, u.Quantity):
            return self.data.unit
        else: return None
    @property
    def imagesize(self):
        # the imagesize is in [ysize, xsize]
        try: return self.shape[-2:]
        except: return None
    @property
    def nchan():
        try: return self.shape[-3]
        except: return None
    @property
    def npol():
        try: return self.shape[-4]
        except: return None
    @property
    def image(self,):
        if self.imagesize is not None:
            return self.data.reshape(self.imagesize)
        else:
            raise ValueError('No valid image data!')
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
    def pixel_beam(self):
        # convert the beam size into pixel sizes
        if self.beam is None:
            raise ValueError("No valid beams can be found!")
        bmaj, bmin, bpa = self.beam
        x_scale = 1/self.pixel_sizes[0].to(u.arcsec).value
        y_scale = 1/self.pixel_sizes[1].to(u.arcsec).value
        bmaj_pixel = np.sqrt((bmaj*np.cos(bpa/180*np.pi)*x_scale)**2 
                             + (bmaj*np.sin(bpa/180*np.pi)*y_scale)**2)
        bmin_pixel = np.sqrt((bmin*np.sin(bpa/180*np.pi)*x_scale)**2 
                             + (bmin*np.cos(bpa/180*np.pi)*y_scale)**2)
        return [bmaj_pixel, bmin_pixel, bpa]

    def update_wcs(self, wcs=None):
        if wcs is not None:
            self.wcs = wcs
        else: 
            self.wcs = WCS(self.header)

    def subimage(self, s_):
        """extract subimage from the orginal image

        Args:
            s_ (:obj:`slice`): data slice, same shape as data

        Return:
            :obj:`Image`
        """
        data_sliced = self.data[s_].copy()
        shape_sliced = data_sliced.shape
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
        if naxis >= 3:
            header_sliced.set('NAXIS3',shape_sliced[-3])
        if naxis >= 4:
            header_sliced.set('NAXIS4',shape_sliced[-4])
        return Image(data_sliced, header_sliced, self.beam)

    def imstat(self):
        # sigma clipping to remove any sources before calculating the RMS
        return sigma_clipped_stats(self.image, sigma=5.0, mask=self.mask)

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

    def plot(self, image=None, name=None, ax=None, figsize=(8,6), fov=0, vmax=10, vmin=-3,
             show_center=True, show_axis=True, show_fwhm=True, show_fov=False,
             show_rms=False, show_flux=True, show_sky_sources=[], show_pixel_sources=[],
             show_detections=False, detections=None, aperture_scale=6.0, aperture_size=None, 
             show_detections_yoffset='-1x', show_detections_color='white', fontsize=12, 
             figname=None, show_colorbar=True, **kwargs):
        """general plot function build on FitsImage

        Args:
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
        ax.set_title(name)
        ny, nx = self.imagesize
        x_index = (np.arange(0, nx) - nx/2.0) * self.pixel_sizes[0].value # to arcsec
        # x_index = (np.arange(nx-1, -1, -1) - nx/2.0) * self.pixel2deg_ra * 3600 # to arcsec
        y_index = (np.arange(0, ny) - ny/2.0) * self.pixel_sizes[1].value # to arcsec
        #x_map, y_map = np.meshgrid(x_index, y_index)
        #ax.pcolormesh(x_map, y_map, data_masked)
        extent = [np.max(x_index), np.min(x_index), np.min(y_index), np.max(y_index)]
        if self.std is not None:
            if self.unit is not None:
                vmax = vmax * self.std.value
                vmin = vmin * self.std.value
            else:
                vmax = vmax * self.std
                vmin = vmin * self.std
        im = ax.imshow(image, origin='lower', extent=extent, interpolation='none', 
                       vmax=vmax, vmin=vmin, **kwargs)
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
                # fits image angle start with north, go anti-clock wise if the angle is positive
                # pyplot ellipse, angle is start with long axis and goes to the negative direction of x
                ellipse = patches.Ellipse((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                      width=self.beam[1], height=self.beam[0], # put bmaj axis in north
                                      angle=-self.beam[-1], #angle=self.bpa, 
                                      facecolor='orange', edgecolor=None, alpha=0.8)
                ax.add_patch(ellipse)
        if False: # another way to show the beam size with photutils EllipticalAperture
            aper = EllipticalAperture((0.8*np.max(x_index), 0.8*np.min(y_index)), 
                                      0.5*self.bmin*3600, 0.5*self.bmaj*3600, # put bmaj axis in north
                                      theta=-self.bpa/180*np.pi)
            aper.plot(color='red', lw=2)

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
                xdet = (-det['x']+nx/2.0)*self.pixel_sizes[0].value
                ydet = (det['y']-ny/2.0)*self.pixel_sizes[1].value
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
                        angle=det['theta']*180/np.pi, 
                        facecolor=None, edgecolor=show_detections_color, alpha=0.8, fill=False)
                ax.add_patch(ellipse_det)
                # ellipse_det = EllipticalAperture([xdet, ydet], det['a']*aperture_scale, 
                                                 # det['b']*aperture_scale, det['theta'])
                # ellipse_det.plot(color='white', lw=1, alpha=0.8)
                if 'x' in show_detections_yoffset:
                    yoffset = float(show_detections_yoffset[:-1]) * (det['a'] + det['b'])
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

    def source_finder(self, **kwargs):
        """class wrapper for source_finder
        """
        sources_found = source_finder(self.image.value, std=self.std.value, name=self.name, 
                                      beam=self.pixel_beam, **kwargs)
        n_dets = len(sources_found)
        table_new_data = Table(np.array([0., 0.] * n_dets).reshape(n_dets, 2),
                           names={'ra', 'dec'})
        table_objs = hstack([table_new_data, sources_found])
        # convert the pixel coordinates to sky coordinates
        for i in range(n_dets):
            obj = table_objs[i]
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], self.wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
        return table_objs 
        # TODO
        # measure the flux the detections
        if n_dets > 0:
            table_objs['a'] = table_objs['a']*pixel2arcsec
            table_objs['b'] = table_objs['b']*pixel2arcsec
            flux, fluxerr = measure_flux(fitsimage, detections=table_objs, 
                                         aperture_size=[fitsimage.bmaj*3600, fitsimage.bmin*3600]
                                         )
            table_objs['flux'] = flux
            table_objs['flux_err'] = fluxerr
        return table_objs

    def measure_flux(self, dets=None, coords=None, apertures=None, **kwargs):
        """shortcut for flux measurement of existing detections

        Args:
            dets (`astropy.table`): the table includes all the detections
                required informations are:
                name  x   y   a   b  theta
                unit pix pix pix pix radian

        TODO: add support for aperture support
        """
        if dets is not None:
            flux, fluxerr = measure_flux(self.image.value, coords=list(zip(dets['x'], dets['y'])), 
                                         apertures=list(zip(4*dets['a'], 4*dets['b'], dets['theta'])), **kwargs) 
        elif (apertures is not None) and (coords is not None):
            flux, fluxerr = measure_flux(self.image.value, coords=coords, apertures=apertures, **kwargs) 
        return flux*self.unit, fluxerr*self.unit

    def fit_2Dgaussian(self, **kwargs):
        return gaussian_2Dfitting(self.image, **kwargs)

    def writefits(self, filename, overwrite=False):
        """write to fits file
        
        This function also shifts the reference pixel of the image to the image center
        """
        ysize, xsize = self.imagesize
        header = self.header
        if header is None:
            if self.wcs is not None:
                header = wcs.to_header()
            else: # create a new
                header = fits.Header()
        # try: # shift the image reference pixel, tolerence is 4 pixels
            # if header['CRPIX1'] < (xsize//2-4) or header['CRPIX1'] > (xsize//2+4):
                # header['CRVAL1'] += (1 - header['CRPIX1']) * header['CDELT1']
                # header['CRPIX1'] = xsize//2
                # print('Warning: automatically shifted the x axis reference.')
            # if header['CRPIX2'] < (ysize//-4) or header['CRPIX2'] > (ysize//2+4):
                # header['CRVAL2'] += (1 - header['CRPIX2']) * header['CDELT2']
                # header['CRPIX2'] = ysize//2
                # print('Warning: automatically shifted the y axis reference.')
        # except: pass
        if self.beam is not None:
            # follow the same rule as CASA, which use the units of deg
            header['BMAJ'] = self.beam[0] / 3600.
            header['BMIN'] = self.beam[1] / 3600.
            header['BPA'] = self.beam[2]
        header.set('BUNIT', self.data.unit.to_string())
        header.update({'history':'created by image_tools.Image',})
        imagehdu = fits.PrimaryHDU(data=self.data.value, header=header)
        imagehdu.writeto(filename, overwrite=overwrite)

    def correct_pb(self, pbfile, pbcorfile):
        pass

    @staticmethod
    def read(fitsimage, extname='primary', name=None, debug=False):
        """read the fits file
        """
        with fits.open(fitsimage) as image_hdu:
            if debug:
                print(image_hdu.info())
            image_header = image_hdu[extname].header
            image_data = image_hdu[extname].data * u.Unit(image_header['BUNIT'])
            try: 
                image_beam = [image_header['BMAJ']*3600., image_header['BMIN']*3600., 
                              image_header['BPA']]
            except: 
                image_beam = None
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
              weights=1/(yrad**2+xrad**2+(x_stddev)**2+(0.25*y_stddev)**2))
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
            gaussian_init = patches.Ellipse((p_init.x_mean, p_init.y_mean), height=p_init.y_stddev.value, 
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

def source_finder(image=None, std=None, mask=None, 
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
        of the major and minor axis. The `theta` is the position-angle of the a axis relative 
        to the first image axis. It is counted positive in the direction of the second axis.

    """
    if method == 'sep':
        try:
            import sep
        except:
            raise ValueError("SEP cannot be found!")
        sources_found = Table(sep.extract(image, detection_threshold, err=std, mask=mask, filter_kernel=filter_kernel))
        try: sources_found = Table(sep.extract(image, detection_threshold, err=std, mask=mask, filter_kernel=filter_kernel))
        except:
            # convert the byte order from big-endian (astropy.fits default) on little-endian machine
            image_little_endian = image.byteswap().newbyteorder()
            sources_found = sep.extract(image_little_endian, detection_threshold, err=std, mask=mask, 
                                  filter_kernel=filter_kernel)
        if sources_found is not None:
            sources_found.rename_column('peak', 'peak_value')
            sources_found.add_column(sources_found['peak_value']/std, name='peak_snr')
        
    if method == 'find_peaks':
        try:
            from photutils import find_peaks
        except:
            raise ValueError("photutils is NOT installed!")
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
                                fwhm=beam[0], ratio=beam[1]/beam[0], theta=beam[2]+90,
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
            if show_flux:
                if 'peak_value' in obj.colnames:
                    ax.text(obj['x'], 0.94*obj['y'], "{:.2e}".format(
                            obj['peak_value']*1e3), color='white', 
                            horizontalalignment='center', verticalalignment='top',)
    if savefile:
        save_array(table_objs, savefile)
    else:
        return table_objs

def create_apertures(coords, shapes=None, apermaj=None, apermin=None, theta=0):
    """create elliptical apertures

    Args:
        coords: the coordinates of the apertures
        shapes: the shapes of apertures, [a,b,theta] or [[a1,b1,theta1],[a2,b2,theta2],]
            in the units of (pixel, pixel, radian)
        apermaj: the pixel size of the major axis
        apermin (optional): the pixel size of the minor axis
        theta: the orientation of the aperture, degree,
        duplication: number of apertures with the same aperture
    """
    ndim = np.ndim(coords)
    apertures = []
    if shapes is not None:
        ndim_shapes = np.ndim(shapes)
        if ndim == 1:
            if ndim_shapes == 1:
                return EllipticalAperture(coords, *shapes)
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
    else:
        if apermin is None:
            apermin = apermaj
        ndim_aper = np.ndim(apermaj)
        if ndim == 1: # single coordinate
            if ndim_aper != 1:
                raise ValueError("``image_tools.create_aperture``: unmatched aperture with coordinates")
            else:
                return EllipticalAperture(coords, apermaj, apermin, theta)
        elif ndim == 2: # multiple coordinates
            apertures = []
            if ndim_aper == 1: # uniform aperture
                for coord in coords:
                    apertures.append(EllipticalAperture(coord, apermaj, apermin, theta))
            elif ndim_aper == 2: # different aperture for different sources
                if np.ndim(theta) == 1:
                    theta = np.full_like(apermaj, theta)
                if np.ndim(theta) != 2:
                    raise ValueError("``image_tools.create_aperture``: unmatched aperture between sizes and angles")
                for i,coord in enumerate(coords):
                    apertures.append(EllipticalAperture(coord, apermaj[i], apermin[i], theta[i]))
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
        apertures = create_apertures(coords, shape=shape)
    if isinstance(apertures, EllipticalAperture):
        apertures = [apertures,]
    for aper in apertures:
        mask = aper.to_mask().to_image(image_shape) > 0
        image_mask[mask] = True
    return image_mask

def measure_flux(image, coords=None,
                 method='single-aperture', apertures=None, 
                 mask=None, n_boostrap=100,
                 gaussian_segment_scale=4.0,
                 plot=False, ax=None, color='white', debug=False):
    """Two-dimension flux measurement in the pixel coordinates
    
    It supports three ways to provide the coordinates of the detection. The ``detections`` allows
    table-like input, while the pixel_coords and sky_coords allow direct coordinates.
    The aperture are provided by the aperture and aperture_scale. ``aperture`` provides the shape
    while aperture_scale constraints the size.

    Args:
        image: the data with or without units
        coords: the pixel coordinates of the detections, [[x1,y1], [x2,y2]]
        aperture: the fixed size of the aperture, in arcsec
    """
    imagesize = image.shape
    if isinstance(apertures, EllipticalAperture):
        detections_apers = apertures
    else: # make new apertures with the shape parameters
        detections_apers = create_apertures(coords, shapes=apertures)
    # convert single item into list
    if isinstance(detections_apers, EllipticalAperture):
        detections_apers = [detections_apers,]
    if np.ndim(coords) == 1:
        coords = [coords,]
    detections_mask = mask_coordinates(image, apertures=detections_apers)
    n_sources = len(detections_apers)
    flux = np.zeros(n_sources)
    fluxerr = np.zeros(n_sources)
    if method == 'single-aperture':
        # measuring flux density
        for i,aper in enumerate(detections_apers):
            x,y = coords[i]
            phot_table = aperture_photometry(image, aper, mask=mask)
            # aperture_correction
            flux[i] = phot_table['aperture_sum'].value
            
            # measuring the error of flux density with bootstraping
            pixel_x = np.random.random(n_boostrap) * imagesize[1] # 1 for x axis
            pixel_y = np.random.random(n_boostrap) * imagesize[0] # 0 for y axis
            pixel_coords_boostrap = np.vstack([pixel_x, pixel_y]).T
            apertures_boostrap = EllipticalAperture(pixel_coords_boostrap, aper.a, 
                                                    aper.b, aper.theta)
            noise_boostrap = aperture_photometry(image, apertures_boostrap, 
                                                 mask=detections_mask)
            fluxerr[i] = np.std(np.ma.masked_invalid(noise_boostrap['aperture_sum']))
    
    if method == 'gaussian':
        seg_size = gaussian_segment_scale*np.int(fitsimage.bmaj_pixel)
        segments = RectangularAperture(pixel_coords, seg_size, seg_size, theta=0)
        segments_mask = segments.to_mask(method='center')
        for i,s in enumerate(segments_mask):
            x,y = pixel_coords[i]
            pixel_fluxscale = 1/(fitsimage.beamsize)
            image_cutout = s.cutout(fitsimage.image)
            gaussian_fitting = gaussian_2Dfitting(image_cutout, debug=debug)
            flux[i] = gaussian_fitting['flux'] * pixel_fluxscale 
            # boostrap for noise measurement
            a_fitted_aper = 1.0 * 2.355 * gaussian_fitting['x_stddev'] # 2xFWHM of gaussian
            b_fitted_aper = 1.0 * 2.355 * gaussian_fitting['y_stddev']
            theta_fitted = gaussian_fitting['theta']
            detections_apers.append(EllipticalAperture([x, y], a_fitted_aper, b_fitted_aper, 
                                                       theta_fitted))
            pixel_x = np.random.random(n_boostrap) * fitsimage.imagesize[1] # 1 for x axis
            pixel_y = np.random.random(n_boostrap) * fitsimage.imagesize[0] # 0 for y axis
            pixel_coords_boostrap = np.vstack([pixel_x, pixel_y]).T
            apertures_boostrap = EllipticalAperture(pixel_coords_boostrap, a_fitted_aper, 
                                                    b_fitted_aper, theta_fitted)
            noise_boostrap = aperture_photometry(fitsimage.image, apertures_boostrap, 
                                                 mask=detections_mask)
            fluxerr[i] = np.std(np.ma.masked_invalid(noise_boostrap['aperture_sum'])) * pixel_fluxscale

            if fitsimage.has_pbcor:
                pixel_pbcor = 1./fitsimage.image_pb[int(np.round(x)), int(np.round(y))]
                flux[i] = flux[i] * pixel_pbcor 
                fluxerr[i] = fluxerr[i] * pixel_pbcor 

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest', origin='lower')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        for i in range(n_sources):
            obj = coords[i]
            im = detections_apers[i].plot(color=color, lw=1, alpha=0.8)
            ax.text(obj[0], obj[1], "{:.2f}mJy".format(flux[i]*1e3), 
                    color=color, horizontalalignment='center', verticalalignment='top',)
        # # only for test
        # for ap in apertures_boostrap:
            # im = ap.plot(color='gray', lw=2, alpha=0.2)
        
    return flux, fluxerr


        #table_return = Table(np.array(['', 0., 0., 0., 0.] * n_found).reshape(n_sources, 5),
        #                   names={'name', 'ra', 'dec', 'flux', 'fluxerr'})
        #for i in range(n_sources):
        #    tab_item = table_return[i]
        #    tab_item['name'] = name + '-' + str(i)
            # tab_item['ra'] = detections['ra'][i]
            # tab_item['dec'] = detections['dec'][i]
            # tab_item['flux'] = name
            # tab_item['fluxerr'] = name

def make_rmap(image, scale=1.0):
    """make radial map relative to the image center
    """
    ny, nx = image.shape
    if not isinstance(pixel_size, (list, np.ndarray)):
        pixel_size = [pixel_size, pixel_size]
    ra_scale, dec_dec = scale
    x_index = (np.arange(0, nx) - nx/2.0) * ra_scale
    y_index = (np.arange(0, ny) - ny/2.0) * dec_scale
    x_map, y_map = np.meshgrid(x_index, y_index)
    rmap = np.sqrt(x_map**2 + y_map**2)
    return rmap

def make_gaussian_kernel(shape, fwhm=None, sigma=None):
    if fwhm is not None:
        fwhm2sigma = 2.35482
        sigma = (np.array(fwhm) / fwhm2sigma).tolist()
    kernel = convolution.Gaussian2DKernel(*sigma)
    return kernel

def make_gaussian_image(shape, fwhm=None, sigma=None, area=1., offset=(0,0), theta=0, normalize=False):
    """make a gaussian image

    theta: in rad, rotating the gaussian counterclock wise
    """
    image = np.zeros(shape, dtype=float)
    yidx, xidx = np.indices(shape)
    yrad, xrad = yidx-shape[0]/2., xidx-shape[1]/2.
    y = xrad*np.cos(theta) + yrad*np.sin(theta)
    x = yrad*np.cos(theta) - xrad*np.sin(theta)
    if fwhm is not None:
        fwhm2sigma = 2.35482
        sigma = (np.array(fwhm) / fwhm2sigma).tolist()
    if isinstance(sigma, (list, tuple)):
        ysigma, xsigma = sigma
    elif isinstance(sigma, (int, float)):
        ysigma = xsigma = sigma
    if isinstance(offset, (list, tuple)):
        y_offset, x_offset = offset
    elif isinstance(offset, (int, float)):
        y_offset = x_offset = offset
    flux = area * np.exp(-(x-x_offset)**2/2./xsigma**2 - (y-y_offset)**2/2./ysigma**2) / (2*np.pi*xsigma*ysigma)
    if normalize:
        flux = flux / np.sum(flux)
    return flux

def gaussian_aperture_correction(imsize, fwhm, aperture, sourcesize=0,  debug=False):
    """This function generate the aperture correction for assumed 3-D gaussian sources

    Args:
        imsize (int, list, tuple): the size of the generated image
        beam (list, tuple): the beam size of the image --> [bmaj, bmin]
        source size (float): the source size of the image
        width (float): the width of the Elliptical Aperture
        height (float): the height of the Elliptical Aperture

    """
    if isinstance(aperture, (int,float)):
        aperture = [aperture, aperture]
    width, height = aperture
    if isinstance(imsize, (int, float)):
        imsize = [imsize, imsize]
    fwhm_convolved = np.sqrt(np.array(fwhm)**2+sourcesize**2).tolist()
    image_gaussian = make_gaussian_image(imsize, fwhm_convolved)
    aper = EllipticalAperture(0.5*np.array(imsize), width, height)
    aper_phot = aperture_photometry(image_gaussian, aper)
    if debug:
        plt.imshow(image_gaussian, origin='lower')
        ap_patches = aper.plot(color='red', lw=2)
    return 1.0/aper_phot['aperture_sum'].value[0]

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
                
def solve_impb(datafile, pbfile=None, pbcorfile=None, pixel_coords=None, sky_coords=None,):
    """This function derive the primary correction
    """
    with fits.open(datafile) as hdu:
        data = hdu[0].data
        header = hdu[0].header
    if pbfile is not None:
        with fits.open(pbfile) as hdu:
            pbdata = hdu[0].data
    elif pbcorfile is not None:
        with fits.open(pbcorfile) as hdu:
            pbcordata = hdu[0].data
            pbdata = data / pbcordata
    else:
        raise ValueError("No valid primary beam information has been provided!")
    return Image(data=pbdata, header=header, name='pb')

