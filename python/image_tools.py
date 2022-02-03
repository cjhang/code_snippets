"""Utilities dealing with fits images

Author: Jianhang Chen, cjhastro@gmail.com
History:
    2022-01-11: first release to handle fits images from CASA

Requirement:
    numpy
    matplotlib
    astropy >= 5.0
    sep
"""


import os
import sys
import numpy as np
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
from matplotlib.patches import Ellipse
from astropy.modeling import models, fitting

from photutils import aperture_photometry, find_peaks, EllipticalAperture, RectangularAperture

# Filtering warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

class FitsImage(object):
    """The datastructure to handle fits image files
    """
    def __init__(self, image_file, pbcor_file=None, pb_file=None, name=None, 
                 changebyteorder=True):
        # all the pre-defined internal values
        self.image_file = image_file
        self.pbcor_file = pbcor_file
        self.pb_file = pb_file
        self.data = None
        self.data_pbcor = None
        self.data_pb = None
        self.has_pbcor = False
        with fits.open(image_file) as hdu:
            # change the endianness of data, needed for SEP,
            # For x86-64 systems, it is using little-endian
            # same for all the data read by astropy.io.fits
            if changebyteorder:
                self.data = hdu[0].data.byteswap(inplace=True).newbyteorder()
            else: 
                self.data = hdu[0].data
            self.header = hdu[0].header
            self.wcs = WCS(self.header)
        
        # name of the image
        if name is None:
            if '.' in image_file:
                self.name = ".".join(os.path.basename(image_file).split('.')[:-1])
            else:
                self.name = os.path.basename(image_file)
        else:
            self.name = name
        
        if pbcor_file:
            with fits.open(pbcor_file) as hdu_pbcor:
                if changebyteorder:
                    self.data_pbcor = hdu_pbcor[0].data.byteswap(inplace=True).newbyteorder()
                else:
                    self.data_pbcor = hdu_pbcor[0].data
        if pb_file:
            with fits.open(pb_file) as hdu_pb:
                if changebyteorder:
                    self.data_pb = hdu_pb[0].data.byteswap(inplace=True).newbyteorder()
                else: 
                    self.data_pb = hdu_pb[0].data
        if self.data_pbcor is None:
            if self.data_pb is not None:
                self.data_pbcor = self.data / self.data_pb
        if self.data_pb is None:
            if self.data_pbcor is not None:
                self.data_pb = self.data / self.data_pbcor
        if self.data_pbcor is not None:
            self.has_pbcor = True

        # calculations from the header
        deg2pixel_ra, deg2pixel_dec = abs(1./self.header['CDELT1']), abs(1./self.header['CDELT2'])
        
        # read the reference system
        try:
            refer = 'J'+str(int(self.header['EQUINOX']))
        except:
            refer = 'J2000'
        self.direction = refer +' '+ SkyCoord(self.header['CRVAL1'], self.header['CRVAL2'], 
                                          unit="deg").to_string('hmsdms')
        self.reffreq = self.header['CRVAL3'] * u.Hz
        self.reflam = (const.c / self.reffreq).to(u.um)

        # read sythesized beam, units in deg, deg, deg 
        self.bmaj, self.bmin, self.bpa = self.header['BMAJ'], self.header['BMIN'], self.header['BPA'] 
        self.bmaj_pixel = self.header['BMAJ'] * deg2pixel_ra
        self.bmin_pixel = self.header['BMIN'] * deg2pixel_dec 
        self.beamsize = 1/(np.log(2)*4.0) * np.pi * self.bmaj * self.bmin * (deg2pixel_ra * deg2pixel_dec)
        # assign the image data
        ndim = self.data.ndim
        if ndim >= 2:
            self.imagesize = self.data.shape[-2:]
        elif ndim < 2:
            raise ValueError('Unsupport image files, at least 2 dimentional is needed!')
        self.imstat()

        # shorts cuts to access the main image
    @property
    def image(self):
        return self.data.reshape(self.imagesize)
    @property
    def masked_image(self):
        return np.ma.array(self.data.reshape(self.imagesize), mask=self.imagemask)
    @property
    def image_pbcor(self):
        if self.has_pbcor:
            return self.data_pbcor.reshape(self.imagesize)
        else:
            return None
    @property
    def image_pb(self):
        if self.has_pbcor:
            return self.data_pb.reshape(self.imagesize)
        else:
            return None
    @property
    def imagemask(self):
        return self.invalid_mask.reshape(self.imagesize)

    def get_fov(self, D=12.):
        """get the field of view

        Parameters:
        D : float
            the diameter of the telesope
        """
        return 1.02 * (self.reflam / (12*u.m)).decompose().value * 206264.806

    def imstat(self):
        # image based calculations
        image_masked = np.ma.masked_invalid(self.data)
        self.invalid_mask = image_masked.mask
        # sigma clipping to remove any sources before calculating the RMS
        self.mean, self.median, self.std = sigma_clipped_stats(image_masked.data, sigma=5.0, mask=self.invalid_mask)

    def plot(self):
        pass

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
                
def source_finder(fitsimage=None, plot=False, 
                  name=None, method='sep',
                  aperture_scale=3.0, detection_threshold=5.0, 
                  ax=None, savefile=None):
    """a source finder and flux measurement wrapper of SEP
    It is designed to handle the interferometric data
    
    Params:
        aperture_scale : aperture size in units of FWHM, it is used for aperture photometry;
                         and also for peak detections
        detection_threshold: the minimal SNR when searching for detections.

    """
    # initialize the image
    if isinstance(fitsimage, dict):
        fitsimage = FitsImage(fitsimage['image_file'], pbcor_file=fitsimage['pbcor_file'], 
                              pb_file=fitsimage['pb_file'], name=name)
    if name is None:
        name = fitsimage.name
    image = fitsimage.image
    mask = fitsimage.imagemask
    image_pbcor = fitsimage.image_pbcor
    image_pb = fitsimage.image_pb
    bmaj, bmin, theta = fitsimage.bmaj_pixel, fitsimage.bmin_pixel, fitsimage.bpa
    beamsize = fitsimage.beamsize
    std = fitsimage.std

    if method == 'sep':
        try:
            import sep
        except:
            raise ValueError("SEP not installed!")
        objects = sep.extract(image, detection_threshold, err=std)
        n_objects = len(objects)
       
        table_new_name = Table(np.array(['', '', '']*n_objects).reshape(n_objects, 3), 
                               names={'name', 'code', 'comments'},dtype=['U32', 'U32', 'U80'])
        table_new_data = Table(np.array([0., 0., 0., 0] * n_objects).reshape(n_objects, 4),
                           names={'ra', 'dec', 'fluxerr', 'fluxflag'})
        table_objs = hstack([table_new_name, table_new_data, Table(objects)])
        # aperture photometry based on the shape of the sources
        if fitsimage.has_pbcor:
            flux, fluxerr, fluxflag = sep.sum_ellipse(image_pbcor, objects['x'], objects['y'],
                                        aperture_scale*objects['a'], aperture_scale*objects['b'],\
                                        objects['theta'], err=std, gain=1.0)
        else:
            flux, fluxerr, fluxflag = sep.sum_ellipse(image, objects['x'], objects['y'],
                                        aperture_scale*objects['a'], aperture_scale*objects['b'],\
                                        objects['theta'], err=std, gain=1.0)
        table_objs['flux'] = flux / beamsize
        table_objs['fluxflag'] = fluxflag
        for i in range(n_objects):
            obj = table_objs[i]
            if fitsimage.has_pbcor:
                obj['fluxerr'] = np.sqrt(np.pi*aperture_scale**2*obj['a']*obj['b']) * (std / beamsize 
                        / image_pb[int(obj['y']), int(obj['x'])])
            else:
                obj['fluxerr'] = np.sqrt(np.pi*aperture_scale**2*obj['a']*obj['b']) * std / beamsize
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], fitsimage.wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
            obj['name'] = name + '_' + str(i)

    if method == 'find_peak':
        try:
            from photutils import find_peaks
        except:
            raise ValueError("photutils is NOT installed!")
        sources_found = find_peaks(image, threshold=detection_threshold*std, 
                                   box_size=aperture_scale, mask=mask) 
        sources_found.rename_column('x_peak', 'x')
        sources_found.rename_column('y_peak', 'y')
        sources_found.add_column(sources_found['peak_value'], name='flux') # / beamsize
        n_found = len(sources_found)

        table_new_name = Table(np.array(['', '', '']*n_found).reshape(n_found, 3), 
                               names={'name', 'code', 'comments'},dtype=['U32', 'U32', 'U80'])
        table_new_data = Table(np.array([0., 0., 0., 0., 0.] * n_found).reshape(n_found, 5),
                           names={'ra', 'dec', 'a', 'b', 'theta'})
        table_objs = hstack([table_new_name, table_new_data, sources_found])
        for i in range(n_found):
            obj = table_objs[i]
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], fitsimage.wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
            obj['name'] = name + '_' + str(i)
        table_objs['a'] = bmaj * 0.5
        table_objs['b'] = bmin * 0.5
        table_objs['theta'] = theta

    if method == 'DAOStarFinder':
        try:
            from photutils import DAOStarFinder
        except:
            raise ValueError("photutils is NOT installed!")
        daofind = DAOStarFinder(fwhm=bmaj, threshold=detection_threshold*std, ratio=bmin/bmaj, 
                                theta=theta+90, sigma_radius=1.5, sharphi=1.0, sharplo=0.2,)  
        sources_found = daofind(fitsimage.image, mask=mask)
        sources_found.rename_column('xcentroid', 'x')
        sources_found.rename_column('ycentroid', 'y')
        sources_found['flux'] = sources_found['peak'] #/ beamsize
        n_found = len(sources_found)

        table_new_name = Table(np.array(['', '', '']*n_found).reshape(n_found, 3), 
                               names={'name', 'code', 'comments'},dtype=['U32', 'U32', 'U80'])
        table_new_data = Table(np.array([0., 0., 0., 0., 0.] * n_found).reshape(n_found, 5),
                           names={'ra', 'dec', 'a', 'b', 'theta'})
        table_objs = hstack([table_new_name, table_new_data, sources_found])
        table_objs['a'] = bmaj * 0.5
        table_objs['b'] = bmin * 0.5
        table_objs['theta'] = theta

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest',
                       vmin=-0.2*std, vmax=10.0*std, origin='lower')
        ax.set_title(name)
        n_found = len(table_objs)
        for i in range(n_found):
            obj = table_objs[i]
            e = Ellipse((obj['x'], obj['y']),
                        width=2*aperture_scale*obj['a'],
                        height=2*aperture_scale*obj['b'],
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('white')
            ax.add_artist(e)
            ax.text(obj['x'], 0.94*obj['y'], "{:.2f}mJy".format(
                    obj['flux']*1e3), color='white', 
                    horizontalalignment='center', verticalalignment='top',)
    if savefile:
        save_array(table_objs, savefile)
    else:
        return table_objs

def gaussian_2Dfitting(image, x_mean=0., y_mean=0., x_stddev=1, y_stddev=1, theta=0, debug=False):
    """Apply two dimentional Gaussian fitting
    """
    ysize, xsize = image.shape
    y_center, x_center = ysize/2., xsize/2.
    flux_list = []
    flux_error_list = []

    yidx, xidx = np.indices((ysize, xsize))
    yrad, xrad = yidx-ysize/2., xidx-xsize/2.

    image_scale = np.max(image)
    image_norm = image / image_scale
    p_init = models.Gaussian2D(amplitude=1, x_mean=x_mean, y_mean=y_mean, 
            x_stddev=x_stddev, y_stddev=y_stddev, theta=theta, 
            bounds={"x_mean":(-xsize*0.2,xsize*0.2), "y_mean":(-ysize*0.2,ysize*0.2), 
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
        fig, ax = plt.subplots(1, 3, figsize=(8, 3.5))
        im0 = ax[0].imshow(image_norm, origin='lower', interpolation='none', 
                           extent=(-xsize/2., xsize/2., -ysize/2., ysize/2.))
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        gaussian_init = Ellipse((p_init.x_mean, p_init.y_mean), height=p_init.y_stddev.value, 
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
    return dict_return

def flux_measure(fitsimage, detections=None, pixel_coords=None, skycoords=None, 
        method='single-aperture', a=None, b=None, theta=None, n_boostrap=100,
        apertures_scale=2.0, gaussian_segment_scale=4.0,
        plot=False, ax=None, color='white'):
    """Accurate flux measure
    """
    if detections:
        if len(detections) < 1:
            return None,None
        ra = np.array([detections['ra']])
        dec = np.array([detections['dec']])
        skycoords = SkyCoord(ra.flatten(), dec.flatten(), unit='deg')
        if a is None:
            if 'a' in detections.colnames:
                a = detections['a']
            else:
                a = fitsimage.bmaj_pixel
        if b is None:
            if 'b' in detections.colnames:
                b = detections['b']
            else:
                b = fitsimage.bmin_pixel
        if theta is None:
            if 'theta' in detections.colnames:
                theta = detections['theta']
            else:
                theta = fitsimage.bpa # in deg
        if pixel_coords is None:
            pixel_coords = np.array(list(zip(*skycoord_to_pixel(skycoords, fitsimage.wcs))))
    if pixel_coords is None:
        print("Nothing to do...")
        return None, None
    n_sources = len(pixel_coords)
    
    # define aperture for all the detections
    if isinstance(a, (list, np.ndarray)):
        a_aper = apertures_scale*a
    else:
        a_aper = np.full(n_sources, apertures_scale*a)
    if isinstance(b, (list, np.ndarray)):
        b_aper = apertures_scale*b
    else:
        b_aper = np.full(n_sources, apertures_scale*b)
    if not isinstance(theta, (list, np.ndarray)):
        theta = np.full(n_sources, theta)
    apertures = []
    for i,coord in enumerate(pixel_coords):
        apertures.append(EllipticalAperture(coord, a_aper[i], b_aper[i], theta[i]))
    detections_mask = np.zeros(fitsimage.imagesize, dtype=bool)
    for mask in apertures:
        detections_mask = detections_mask + mask.to_mask().to_image(shape=fitsimage.imagesize)
    detections_mask = (detections_mask > 0) | fitsimage.imagemask


    detections_apers = []
    flux = np.zeros(n_sources)
    fluxerr = np.zeros(n_sources)
    if method == 'single-aperture':
        # measuring flux density
        for i,aper in enumerate(apertures):
            x,y = pixel_coords[i]
            pixel_fluxscale = 1./(fitsimage.beamsize)
            detections_apers.append(EllipticalAperture([x, y], a_aper[i], b_aper[i], theta[i]))
            if fitsimage.has_pbcor:
                phot_table = aperture_photometry(fitsimage.image_pbcor, aper, mask=fitsimage.imagemask)
            else:
                phot_table = aperture_photometry(fitsimage.image, apertures, mask=fitsimage.imagemask)
            flux[i] = phot_table['aperture_sum'].value * pixel_fluxscale
            # measuring the error of flux density
            # run the boostrap for random aperture with the image
            pixel_x = np.random.random(n_boostrap) * fitsimage.imagesize[1] # 1 for x axis
            pixel_y = np.random.random(n_boostrap) * fitsimage.imagesize[0] # 0 for y axis
            # points_select = (pixel_x**2 + pixel_y**2) < (np.min(fitsimage.imagesize)-np.max([a,b]))**2
            # points_select = (pixel_x-**2 + pixel_y**2) > (np.max([a,b]))**2
            # pixel_coords_boostrap = np.vstack([pixel_x[points_select], pixel_y[points_select]]).T
            pixel_coords_boostrap = np.vstack([pixel_x, pixel_y]).T
            apertures_boostrap = EllipticalAperture(pixel_coords_boostrap, a_aper[i], 
                                                    b_aper[i], theta[i])
            noise_boostrap = aperture_photometry(fitsimage.image, apertures_boostrap, 
                                                 mask=detections_mask)
            fluxerr[i] = np.std(np.ma.masked_invalid(noise_boostrap['aperture_sum'])) * pixel_fluxscale
            # fluxerr[i] = np.std(noise_boostrap['aperture_sum']) * pixel_fluxscale
            if fitsimage.has_pbcor:
                pixel_pbcor = 1./fitsimage.image_pb[int(np.round(x)), int(np.round(y))]
                fluxerr[i] = fluxerr[i] * pixel_pbcor 
    if method == 'gaussian':
        seg_size = gaussian_segment_scale*np.int(fitsimage.bmaj_pixel)
        segments = RectangularAperture(pixel_coords, seg_size, seg_size, theta=0)
        segments_mask = segments.to_mask(method='center')
        for i,s in enumerate(segments_mask):
            x,y = pixel_coords[i]
            pixel_fluxscale = 1/(fitsimage.beamsize)
            image_cutout = s.cutout(fitsimage.image)
            gaussian_fitting = gaussian_2Dfitting(image_cutout)
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
        im = ax.imshow(fitsimage.image, interpolation='nearest',
                       vmin=-0.2*fitsimage.std, vmax=10.0*fitsimage.std, origin='lower')
        plt.colorbar(im, fraction=0.046, pad=0.04)

        for i in range(n_sources):
            obj = pixel_coords[i]
            im = detections_apers[i].plot(color=color, lw=1, alpha=0.8)
            ax.text(obj[0], (1.0-2.0*detections_apers[i].a/fitsimage.imagesize[0])*obj[1], 
                    "{:.2f}mJy".format(flux[i]*1e3), 
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

def make_blank_image(fitsimage, mask_aperture=2, **kwargs):
    # remove all the detections, return blank image
    image = fitsimage.image.copy()
    # try:
    detections = source_finder(fitsimage, **kwargs)
    ra = np.array([detections['ra']])
    dec = np.array([detections['dec']])
    skycoords = SkyCoord(ra.flatten(), dec.flatten(), unit='deg')
    pixel_coords = np.array(list(zip(*skycoord_to_pixel(skycoords, fitsimage.wcs))))
    for i,coord in enumerate(pixel_coords):
        aper = EllipticalAperture(coord, mask_aperture, mask_aperture, 0)
        mask = aper.to_mask().to_image(shape=fitsimage.imagesize) > 0
        image[mask] = fitsimage.std
    # except:
        # return False

    return image


