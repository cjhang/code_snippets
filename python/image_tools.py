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


from photutils import aperture_photometry, find_peaks, EllipticalAperture, RectangularAperture, SkyEllipticalAperture

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
            try:
                self.wcs = WCS(self.header)
            except:
                self.wcs = None

        # calculations the from header
        ## image dimensions
        self.ndim = self.header['NAXIS']
        if self.ndim >= 2:
            self.imagesize = (self.header['NAXIS2'], self.header['NAXIS1']) # higher dimension first
        else:
            raise ValueError('Unsupport image files, at least 2 dimentional is needed!')
        if self.ndim >= 3:
            self.nfreq = self.header['NAXIS3']
            try:
                self.reffreq = self.header['CRVAL3'] * u.Hz
                self.restfreq = self.header['RESTFRQ'] * u.Hz
                self.reflam = (const.c / self.reffreq).to(u.um)
                self.restlam = (const.c / self.restfreq).to(u.um)
            except:
                pass
        if self.ndim >= 4:
            self.nstocks = self.header['NAXIS4']
        ## read the reference system
        try:
            refer = 'J'+str(int(self.header['EQUINOX']))
        except:
            refer = 'J2000'
        self.direction = refer +' '+ SkyCoord(self.header['CRVAL1'], self.header['CRVAL2'], 
                                          unit="deg").to_string('hmsdms')
        self.pixel_center = [np.abs(self.header['CRPIX1']), np.abs(self.header['CRPIX1'])]
        self.sky_center = [self.header['CRVAL1'], self.header['CRVAL2']]
        self.pixel2deg_ra, self.pixel2deg_dec = abs(self.header['CDELT1']), abs(self.header['CDELT2'])
        # read sythesized beam, units in deg, deg, deg 
        self.bmaj, self.bmin, self.bpa = self.header['BMAJ'], self.header['BMIN'], self.header['BPA'] 
        self.bmaj_pixel = self.header['BMAJ'] / self.pixel2deg_ra
        self.bmin_pixel = self.header['BMIN'] / self.pixel2deg_dec 
        self.beamsize = 1/(np.log(2)*4.0) * np.pi * self.bmaj * self.bmin \
                        / (self.pixel2deg_ra * self.pixel2deg_dec)
 
        
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

        self.imagemask = self.mask_image(mask_invalid=True)
        self.imstat()

        # shorts cuts to access the main image
    @property
    def image(self):
        return self.data.reshape(self.imagesize)
    @property
    def image_pbcor(self):
        return self.data_pbcor.reshape(self.imagesize)
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

    def mask_image(self, mask=None, mask_invalid=True):
        imagemask = np.zeros_like(self.image, dtype=bool)
        if mask_invalid:
            image_masked = np.ma.masked_invalid(self.data)
            invalid_mask = image_masked.mask.reshape(self.imagesize)
            imagemask = imagemask | invalid_mask
        if mask is not None:
            imagemask = imagemask | mask
        return imagemask

    def mask_detections(self, detections=None, aperture_scale=6, aperture_size=None,
                        sky_coords=None, pixel_coords=None, a=None, b=None, theta=None,
                        minimal_aperture_size=None):
        # generate mask around all the detections

        if detections is not None:
            if len(detections) < 1:
                print('No source founded...')
                return None,None
            dets_colnames = detections.colnames
            if ('x' in dets_colnames) and ('y' in dets_colnames):
                pixel_coords = np.array(list(zip(detections['x'], detections['y'])))
            elif ('ra' in dets_colnames) and ('dec' in dets_colnames):
                ra = np.array([detections['ra']])
                dec = np.array([detections['dec']])
                skycoords = SkyCoord(ra.flatten(), dec.flatten(), unit='deg')
                pixel_coords = np.array(list(zip(*skycoord_to_pixel(skycoords, fitsimage.wcs))))
            if aperture_scale is not None:
                if a is None:
                    if 'a' in detections.colnames:
                        a = detections['a']
                    else:
                        a = self.bmaj_pixel
                if b is None:
                    if 'b' in detections.colnames:
                        b = detections['b']
                    else:
                        b = self.bmin_pixel
                if theta is None:
                    if 'theta' in detections.colnames:
                        theta = detections['theta']
                    else:
                        theta = self.bpa # in deg
        elif sky_coords is not None:
            pixel_coords = np.array(list(zip(*skycoord_to_pixel(sky_coords, self.wcs))))
            if a is None:
                a = self.bmaj_pixel
            if b is None:
                b = self.bmin_pixel
            if theta is None:
                theta = self.bpa # in deg
        elif pixel_coords is not None:
            if a is None:
                a = self.bmaj_pixel
            if b is None:
                b = self.bmin_pixel
            if theta is None:
                theta = self.bpa # in deg
        else: 
            print("Nothing to do...")
            return None, None
        n_sources = len(pixel_coords)
        
        # define aperture for all the detections
        if aperture_scale is not None:
            if isinstance(a, (tuple, list, np.ndarray)):
                a_aper = aperture_scale*a
            else:
                a_aper = np.full(n_sources, aperture_scale*a)
            if isinstance(b, (tuple, list, np.ndarray)):
                b_aper = aperture_scale*b
            else:
                b_aper = np.full(n_sources, aperture_scale*b)
            if minimal_aperture_size is not None:
                minimal_aperture_size_in_pixel = minimal_aperture_size / (self.pixel2deg_ra*3600)
                a_aper[a_aper < minimal_aperture_size_in_pixel] = minimal_aperture_size_in_pixel
                b_aper[b_aper < minimal_aperture_size_in_pixel] = minimal_aperture_size_in_pixel
            if not isinstance(theta, (tuple, list, np.ndarray)):
                theta = np.full(n_sources, theta)
        if aperture_size is not None:
            aperture_size_pixel = aperture_size / (3600*self.pixel2deg_ra)
            a_aper = np.full(n_sources, aperture_size_pixel)
            b_aper = np.full(n_sources, aperture_size_pixel)
            theta = np.full(n_sources, 0)
        apertures = []
        for i,coord in enumerate(pixel_coords):
            apertures.append(EllipticalAperture(coord, a_aper[i], b_aper[i], theta[i]))
        detections_mask = np.zeros(self.imagesize, dtype=bool)
        for mask in apertures:
            image_aper_mask = mask.to_mask().to_image(shape=self.imagesize)
            if image_aper_mask is not None:
                detections_mask = detections_mask + image_aper_mask
            else:
                continue
        return detections_mask > 0


    def get_fov(self, D=12.):
        """get the field of view

        Parameters:
        D : float
            the diameter of the telesope
        """
        return 1.02 * (self.reflam / (12*u.m)).decompose().value * 206264.806

    def imstat(self):
        # image based calculations
        mask = self.imagemask
        # sigma clipping to remove any sources before calculating the RMS
        self.mean, self.median, self.std = sigma_clipped_stats(self.image, sigma=5.0, mask=mask)

    def find_structure(self, sigma=3.0, iterations=1, opening_iters=None, 
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
        mean, median, std = sigma_clipped_stats(self.image, sigma=sigma,
                maxiters=1, mask=self.imagemask)
        struct_above_sigma = self.image > sigma*std
        if opening_iters is None:
            opening_iters = iterations
        if dilation_iters is None:
            dilation_iters = iterations
        struct_opening = ndimage.binary_opening(struct_above_sigma, iterations=opening_iters)
        struct_dilation = ndimage.binary_dilation(struct_opening, iterations=dilation_iters)
        if plot:
            self.plot(data=struct_dilation, name=self.name+'_{}_structures'.format(sigma))

        return struct_dilation

    def plot(self, data=None, name=None, ax=None, figsize=(8,6), fov=0, vmax=10, vmin=-3,
             show_pbcor=False, show_center=True, show_axis=True, show_fwhm=True, show_fov=False,
             show_rms=False, show_sky_sources=[], show_pixel_sources=[],
             show_detections=False, detections=None, aperture_scale=6.0, aperture_size=None, 
             show_detections_yoffset='-1x', fontsize=12, **kwargs):
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
        if data is None:
            if show_pbcor:
                data = self.image_pbcor
            else:
                data = self.image
        ax.set_title(name)
        ny, nx = self.imagesize
        pixel_rescale = self.pixel2deg_ra * 3600
        x_index = (np.arange(0, nx) - nx/2.0) * self.pixel2deg_ra * 3600 # to arcsec
        y_index = (np.arange(0, ny) - ny/2.0) * self.pixel2deg_dec * 3600 # to arcsec
        #x_map, y_map = np.meshgrid(x_index, y_index)
        #ax.pcolormesh(x_map, y_map, data_masked)
        extent = [np.min(x_index), np.max(x_index), np.min(y_index), np.max(y_index)]
        ax.imshow(data, origin='lower', extent=extent, interpolation='none', 
                  vmax=vmax*self.std, vmin=vmin*self.std, **kwargs)
        if show_center:
            ax.text(0, 0, '+', color='r', fontsize=24, fontweight=100, horizontalalignment='center',
                    verticalalignment='center')
        if not show_axis:
            ax.axis('off')
        if show_fwhm:
            ellipse = patches.Ellipse((0.8*np.min(x_index), 0.8*np.min(y_index)), 
                                  width=self.bmin*3600, height=self.bmaj*3600, 
                                  angle=self.bpa, facecolor='orange', edgecolor=None, alpha=0.8)
            ax.add_patch(ellipse)
        if show_fov:    
            ellipse_fov = patches.Ellipse((0, 0), width=fov, height=fov, 
                                    angle=0, fill=False, facecolor=None, edgecolor='gray', 
                                    alpha=0.8)
            ax.add_patch(ellipse_fov)
        if show_rms:
            ax.text(0.6*np.max(x_index), 0.8*np.min(y_index), 'rms={:.0f}uJy'.format(self.std*1e6), 
                    fontsize=0.8*fontsize, horizontalalignment='center', verticalalignment='top', color='white')
        if show_detections:
            for idx,det in enumerate(detections):
                xdet = (det['x']-nx/2.0)*self.pixel2deg_ra*3600
                ydet = (det['y']-ny/2.0)*self.pixel2deg_dec*3600
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
                        facecolor=None, edgecolor='white', alpha=0.8, fill=False)
                ax.add_patch(ellipse_det)
                # ellipse_det = EllipticalAperture([xdet, ydet], det['a']*aperture_scale, 
                                                 # det['b']*aperture_scale, det['theta'])
                # ellipse_det.plot(color='white', lw=1, alpha=0.8)
                if 'x' in show_detections_yoffset:
                    yoffset = float(show_detections_yoffset[:-1]) * 0.5 * (det['a'] + det['b'])
                else:
                    yoffset = show_detections_yoffset

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
                xs = (pixel_sources[0]-nx/2.0)*self.pixel2deg_ra*3600
                ys = (pixel_sources[1]-ny/2.0)*self.pixel2deg_dec*3600
                ax.text(xs, ys, 'x', fontsize=24, fontweight=100, horizontalalignment='center',
                    verticalalignment='center', color='white')
        if show_pixel_sources != []:
            n_pixel_sources = len(show_pixel_sources)
            if n_pixel_sources == 1:
                show_pixel_sources = [show_pixel_sources,]
            for s in show_pixel_sources:
                xp = (s[0]-nx/2.0)*self.pixel2deg_ra*3600
                yp = (s[1]-ny/2.0)*self.pixel2deg_dec*3600
                ax.text(xp, yp, 'x', fontsize=24, fontweight=100, horizontalalignment='center',
                    verticalalignment='center', color='white')

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
                
def source_finder(fitsimage=None, plot=False, mask=None, 
                  name=None, method='sep', show_flux=True,
                  aperture_scale=6.0, 
                  detection_threshold=3.0, peak_snr_threshold=None,
                  ax=None, savefile=None):
    """a source finder and flux measurement wrapper of SEP
    It is designed to handle the interferometric data
    
    Params:
        aperture_scale : aperture size in units of FWHM, it is used for aperture photometry;
                         and also for peak detections
        detection_threshold: the minimal SNR when searching for detections.

    """
    # initialize the image
    if not isinstance(fitsimage, FitsImage):
        if isinstance(fitsimage, dict):
            fitsimage = FitsImage(fitsimage['image_file'], pbcor_file=fitsimage['pbcor_file'], 
                                  pb_file=fitsimage['pb_file'], name=name)
    if name is None:
        name = fitsimage.name
    image = fitsimage.image
    # mask = fitsimage.imagemask
    image_pbcor = fitsimage.image_pbcor
    image_pb = fitsimage.image_pb
    bmaj, bmin, theta = fitsimage.bmaj*3600, fitsimage.bmin*3600, fitsimage.bpa
    beamsize = fitsimage.beamsize
    std = fitsimage.std
    pixel2arcsec = fitsimage.pixel2deg_ra*3600
    arcsec2pixel = 1/pixel2arcsec
    sky_center = fitsimage.sky_center

    if method == 'sep':
        try:
            import sep
        except:
            raise ValueError("SEP not installed!")
        objects = sep.extract(image, detection_threshold, err=std, mask=mask, filter_kernel=None)
        n_objects = len(objects)
       
        table_new_name = Table(np.array(['', '', '', '']*n_objects).reshape(n_objects, 4), 
                               names={'pname', 'name', 'code', 'comments'},
                               dtype=['U64', 'U64', 'U64', 'U80'])
        table_new_data = Table(np.array([0., 0., 0., 0.] * n_objects).reshape(n_objects, 4),
                           names={'ra', 'dec', 'fluxerr', 'radial_distance'})
        table_objs = hstack([table_new_name, table_new_data, Table(objects)])
        table_objs.rename_column('peak', 'peak_flux')
        table_objs.add_column(table_objs['peak_flux'].data/std, name='peak_snr')

        if peak_snr_threshold is not None:
            peak_selection = table_objs['peak_snr'] >= peak_snr_threshold
            table_objs = table_objs[peak_selection]

        # aperture photometry based on the shape of the sources
        # if fitsimage.has_pbcor:
            # flux, fluxerr, fluxflag = sep.sum_ellipse(image_pbcor, objects['x'], objects['y'],
                                        # aperture_scale*objects['a'], aperture_scale*objects['b'],\
                                        # objects['theta'], err=std, gain=1.0)
        # else:
            # flux, fluxerr, fluxflag = sep.sum_ellipse(image, objects['x'], objects['y'],
                                        # aperture_scale*objects['a'], aperture_scale*objects['b'],\
                                        # objects['theta'], err=std, gain=1.0)
        n_dets = len(table_objs)
        for i in range(n_dets):
            obj = table_objs[i]
            # if fitsimage.has_pbcor:
                # obj['fluxerr'] = np.sqrt(np.pi*aperture_scale**2*obj['a']*obj['b']) * (std / beamsize 
                        # / image_pb[int(obj['y']), int(obj['x'])])
            # else:
                # obj['fluxerr'] = np.sqrt(np.pi*aperture_scale**2*obj['a']*obj['b']) * std / beamsize
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], fitsimage.wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
            # calculate the projected radial distance
            obj['radial_distance'] = obj_skycoord.separation(SkyCoord(*fitsimage.sky_center, 
                                                                      unit='deg')).to(u.arcsec).value
            obj['pname'] = name
            obj['name'] = name + '_' + str(i)

        if n_dets > 0:
            table_objs['a'] = table_objs['a']*pixel2arcsec
            table_objs['b'] = table_objs['b']*pixel2arcsec
            flux, fluxerr = measure_flux(fitsimage, detections=table_objs, 
                                         aperture_scale=aperture_scale)
            table_objs['flux'] = flux
            table_objs['flux_err'] = fluxerr

    if method == 'find_peak':
        try:
            from photutils import find_peaks
        except:
            raise ValueError("photutils is NOT installed!")
        sources_found = find_peaks(image, threshold=detection_threshold*std, 
                                   box_size=aperture_scale, mask=mask) 
        sources_found.rename_column('x_peak', 'x')
        sources_found.rename_column('y_peak', 'y')
        sources_found.add_column(sources_found['peak_value'], name='peak_flux') # / beamsize
        sources_found.add_column(sources_found['peak_value']/std, name='peak_snr') # / beamsize
        n_found = len(sources_found)

        table_new_name = Table(np.array(['', '', '', '']*n_found).reshape(n_found, 4), 
                               names={'pname', 'name', 'code', 'comments'},
                               dtype=['U64', 'U64', 'U8', 'U80'])
        table_new_data = Table(np.array([0., 0., 0., 0., 0.,0.] * n_found).reshape(n_found, 6),
                           names={'ra', 'dec', 'a', 'b', 'theta', 'radial_distance'})
        table_objs = hstack([table_new_name, table_new_data, sources_found])
        for i in range(n_found):
            obj = table_objs[i]
            obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], fitsimage.wcs)
            obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
            obj['radial_distance'] = np.sqrt((obj['ra']-sky_center[0])**2 
                                             + (obj['dec']-sky_center[1])**2) * 3600 # to arcsec
            obj['pname'] = name
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
        sources_found['peak_flux'] = sources_found['peak'] #/ beamsize
        sources_found['peak_snr'] = sources_found['peak']/std #/ beamsize
        n_found = len(sources_found)

        table_new_name = Table(np.array(['', '', '', '']*n_objects).reshape(n_objects, 4), 
                               names={'pname', 'name', 'code', 'comments'},
                               dtype=['U64', 'U64', 'U8', 'U80'])
        table_new_data = Table(np.array([0., 0., 0., 0., 0.] * n_found).reshape(n_found, 5),
                           names={'ra', 'dec', 'a', 'b', 'theta'})
        table_objs = hstack([table_new_name, table_new_data, sources_found])
        table_objs['a'] = bmaj * 0.5
        table_objs['b'] = bmin * 0.5
        table_objs['theta'] = theta

    # if peak_snr_threshold is not None:
        # peak_selection = table_objs['peak_snr'] >= peak_snr_threshold
        # table_objs = table_objs[peak_selection]

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest',
                       vmin=-2.*std, vmax=5.0*std, origin='lower')
        ax.set_title(name)
        n_found = len(table_objs)
        for i in range(n_found):
            obj = table_objs[i]
            e = patches.Ellipse((obj['x'], obj['y']),
                        width=2*aperture_scale*obj['a']*arcsec2pixel,
                        height=2*aperture_scale*obj['b']*arcsec2pixel,
                        angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('white')
            ax.add_artist(e)
            if show_flux:
                if 'flux' in obj.colnames:
                    ax.text(obj['x'], 0.94*obj['y'], "{:.2f}mJy".format(
                            obj['flux']*1e3), color='white', 
                            horizontalalignment='center', verticalalignment='top',)
                elif 'peak_flux' in obj.colnames:
                    ax.text(obj['x'], 0.94*obj['y'], "{:.2f}mJy/beam".format(
                            obj['peak_flux']*1e3), color='white', 
                            horizontalalignment='center', verticalalignment='top',)
    if savefile:
        save_array(table_objs, savefile)
    else:
        return table_objs

def gaussian_2Dfitting(image, x_mean=0., y_mean=0., x_stddev=1, y_stddev=1, theta=0, debug=False,
                       xbounds=None, ybounds=None, center_bounds_scale=0.125):
    """Apply two dimentional Gaussian fitting
    
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
    return dict_return

def measure_flux(fitsimage, detections=None, pixel_coords=None, skycoords=None, 
        method='single-aperture', a=None, b=None, theta=None, n_boostrap=100,
        minimal_aperture_size=1, 
        aperture_size=None, aperture_scale=6.0, gaussian_segment_scale=4.0,
        plot=False, ax=None, color='white', debug=False):
    """Accurate flux measure

    Args:
        fitsimage: the FitsImage class
        detections: astropy.table, including all source position and shapes
        pixel_coords: the pixel coordinates of the detections
        skycoords: the sky coordinates of the detections.
        aperture_size: the fixed size of the aperture, in arcsec
        a,b,theta: the size of the source, in arcsec and deg
        minimal_aperture_size: if the aperture_size is None, this can control the
            minial aperture_size for the fain source, where the adaptive aperture 
            could not be securely measured
        aperture_scale: the source shape determined aperture, lower priority than
                        aperture_size
    Note:
        When several coordinates parameters are provided, detections has the
        higher priority
    """
    pixel2arcsec = fitsimage.pixel2deg_ra*3600
    arcsec2pixel = 1/pixel2arcsec
    if detections is not None:
        if len(detections) < 1:
            print('No source founded...')
            return None,None
        dets_colnames = detections.colnames
        # if ('x' in dets_colnames) and ('y' in dets_colnames):
            # pixel_coords = np.array(list(zip(detections['x'], detections['y'])))
        if ('ra' in dets_colnames) and ('dec' in dets_colnames):
            ra = np.array([detections['ra']])
            dec = np.array([detections['dec']])
            skycoords = SkyCoord(ra.flatten(), dec.flatten(), unit='deg')
            pixel_coords = np.array(list(zip(*skycoord_to_pixel(skycoords, fitsimage.wcs))))
        if aperture_scale is not None:
            if a is None: # in arcsec
                if 'a' in detections.colnames:
                    a = detections['a']
                else:
                    a = fitsimage.bmaj*0.5*3600
            if b is None:
                if 'b' in detections.colnames:
                    b = detections['b']
                else:
                    b = fitsimage.bmin*0.5*3600
            if theta is None: # in deg
                if 'theta' in detections.colnames:
                    theta = detections['theta']
                else:
                    theta = fitsimage.bpa
    elif skycoords is not None:
        pixel_coords = np.array(list(zip(*skycoord_to_pixel(skycoords, fitsimage.wcs))))
        if a is None:
            a = fitsimage.bmaj*0.5*3600
        if b is None:
            b = fitsimage.bmin*0.5*3600
        if theta is None:
            theta = fitsimage.bpa # in deg
    elif pixel_coords is not None:
        if a is None:
            a = fitsimage.bmaj*0.5*3600
        if b is None:
            b = fitsimage.bmin*0.5*3600
        if theta is None:
            theta = fitsimage.bpa # in deg
    else: 
        print("Nothing to do...")
        return None, None
    n_sources = len(pixel_coords)
    
    # define aperture for all the detections
    if aperture_scale is not None:
        if isinstance(a, (tuple, list, np.ndarray)):
            a_aper = aperture_scale*a*arcsec2pixel
            # a_aper = aperture_scale*a*u.arcsec
        else:
            a_aper = np.full(n_sources, aperture_scale*a*arcsec2pixel)
        if isinstance(b, (tuple, list, np.ndarray)):
            b_aper = aperture_scale*b*arcsec2pixel
            # b_aper = aperture_scale*b*u.arcsec
        else:
            b_aper = np.full(n_sources, aperture_scale*b*arcsec2pixel)
        if minimal_aperture_size is not None:
            minimal_aperture_size_in_pixel = minimal_aperture_size*arcsec2pixel
            a_aper[a_aper < minimal_aperture_size_in_pixel] = minimal_aperture_size_in_pixel
            b_aper[b_aper < minimal_aperture_size_in_pixel] = minimal_aperture_size_in_pixel
        if not isinstance(theta, (tuple, list, np.ndarray)):
            theta = np.full(n_sources, theta)
    if aperture_size is not None:
        aperture_size_pixel = aperture_size*arcsec2pixel
        a_aper = np.full(n_sources, aperture_size_pixel)
        b_aper = np.full(n_sources, aperture_size_pixel)
        theta = np.full(n_sources, 0)
    apertures = []
    for i,coord in enumerate(pixel_coords):
        apertures.append(EllipticalAperture(coord, a_aper[i], b_aper[i], theta[i]))
        # apertures.append(SkyEllipticalAperture(skycoords, a_aper[i], b_aper[i], theta[i]))
    detections_mask = np.zeros(fitsimage.imagesize, dtype=bool)
    for mask in apertures:
        image_aper_mask = mask.to_mask().to_image(shape=fitsimage.imagesize)
        if image_aper_mask is not None:
            detections_mask = detections_mask + image_aper_mask
        else:
            continue
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
                phot_table = aperture_photometry(fitsimage.image, aper, mask=fitsimage.imagemask)
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

def make_blank_image(fitsimage, mask_detection=True, detections=None, mask_aperture=2, **kwargs):
    # remove all the detections, return blank image
    image = fitsimage.image.copy()
    if mask_detection:
        if detections is None:
            detections = source_finder(fitsimage, **kwargs)
        ra = np.array([detections['ra']])
        dec = np.array([detections['dec']])
        skycoords = SkyCoord(ra.flatten(), dec.flatten(), unit='deg')
        pixel_coords = np.array(list(zip(*skycoord_to_pixel(skycoords, fitsimage.wcs))))
        for i,coord in enumerate(pixel_coords):
            aper = EllipticalAperture(coord, mask_aperture, mask_aperture, 0)
            mask = aper.to_mask().to_image(shape=fitsimage.imagesize) > 0
            image[mask] = fitsimage.std
    return image


