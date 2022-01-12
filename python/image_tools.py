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
import sep
from astropy.io import fits
from astropy.table import Table, vstack, hstack
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class FitsImage(object):
    """The datastructure to handle fits image files
    """
    def __init__(self, image_file, pbcor_file=None, pb_file=None, name=None):
        self.data = None
        self.data_pbcor = None
        self.data_pb = None
        with fits.open(image_file) as hdu:
            # change byte order for SEP
            self.data = hdu[0].data.byteswap(inplace=True).newbyteorder()
            self.header = hdu[0].header
            self.wcs = WCS(self.header)
        # calculations from the header
        deg2pixel_ra, deg2pixel_dec = abs(1./self.header['CDELT1']), abs(1./self.header['CDELT2'])
        # read sythesized beam, units in deg, deg, deg 
        self.bmaj, self.bmin, self.bpa = self.header['BMAJ'], self.header['BMIN'], self.header['BPA'] 
        self.beamsize = 1/(np.log(2)*4.0) * np.pi * self.bmaj * self.bmin * (deg2pixel_ra * deg2pixel_dec)

        if pbcor_file:
            with fits.open(pbcor_file) as hdu_pbcor:
                self.data_pbcor = hdu_pbcor[0].data.byteswap(inplace=True).newbyteorder()
        if pb_file:
            with fits.open(pb_file) as hdu_pb:
                self.data_pb = hdu_pb[0].data.byteswap(inplace=True).newbyteorder()
        if self.data_pbcor is None:
            if self.data_pb is not None:
                self.data_pbcor = self.data / self.data_pb
        if self.data_pb is None:
            if self.data_pbcor is not None:
                self.data_pb = self.data / self.data_pbcor
        if name is None:
            if '.' in image_file:
                self.name = ".".join(os.path.basename(image_file).split('.')[:-1])
            else:
                self.name = os.path.basename(image_file)
        else:
            self.name = name

def save_array(array, savefile=None, overwrite=False):
    if savefile:
        if overwrite or not os.path.isfile(savefile):
            Table(array).write(savefile, format='ascii', overwrite=overwrite)
        else: # it should append the new data at the bottom of the file
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
                
def source_finder(fitsimage=None, image_file=None, pbcor_file=None, pb_file=None, plot=False, 
                  name=None, aperture_scale=3.0, ax=None, savefile=None):
    """a source finder and flux measurement wrapper of SEP
    It is designed to handle the interferometric data
    
    Params:
        aperture_scale : aperture size in units of FWHM
    """
    # initialize the image
    if fitsimage is None:
        fitsimage = FitsImage(image_file, pbcor_file=pbcor_file, pb_file=pb_file, name=name)
    if name is None:
        name = fitsimage.name
    image = fitsimage.data[0,0]
    image_pbcor = fitsimage.data_pbcor[0,0]
    image_pb = fitsimage.data_pb[0,0]
    beamsize = fitsimage.beamsize
    
    
    image_masked = np.ma.masked_invalid(image)
    # sigma clipping to remove any sources before calculating the RMS
    mean, median, std = sigma_clipped_stats(image, sigma=5.0, mask=image_masked.mask)
    objects = sep.extract(image, 3.0, err=std)
    n_objects = len(objects)
   
    table_newcols = Table(np.array(['', 0., 0., 0., 0] * n_objects).reshape(n_objects, 5),
                       names={'name', 'ra', 'dec', 'fluxerr', 'fluxflag'})
    table_objs = hstack([table_newcols, Table(objects)])
    # aperture photometry based on the shape of the sources
    flux, fluxerr, fluxflag = sep.sum_ellipse(image_pbcor, objects['x'], objects['y'],
                                    aperture_scale*objects['a'], aperture_scale*objects['b'],\
                                    objects['theta'], err=std, gain=1.0)
    table_objs['flux'] = flux / beamsize
    table_objs['fluxflag'] = fluxflag
    for i in range(n_objects):
        obj = table_objs[i]
        obj['fluxerr'] = np.sqrt(np.pi*aperture_scale**2*obj['a']*obj['b']) * (std / beamsize 
                / image_pb[int(obj['x']), int(obj['y'])])
        obj_skycoord = pixel_to_skycoord(obj['x'], obj['y'], fitsimage.wcs)
        obj['ra'], obj['dec'] = obj_skycoord.ra.to(u.deg).value, obj_skycoord.dec.to(u.deg).value
        obj['name'] = name + '-' + str(i)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(image, interpolation='nearest',
                       vmin=-0.2*std, vmax=10.0*std, origin='lower')

        # plot an elliptical aperture for each object
        for i in range(n_objects):
            e = Ellipse((objects['x'][i], objects['y'][i]),
                        width=2*aperture_scale*objects['a'][i],
                        height=2*aperture_scale*objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('white')
            ax.add_artist(e)
            # show the flux in the image
            #print(flux[i]*1e3)
            ax.text(objects['x'][i], 0.94*objects['y'][i], "{:.2f}mJy".format(
                    table_objs[i]['flux']*1e3), color='white', 
                    horizontalalignment='center', verticalalignment='top',)
    if savefile:
        save_array(table_objs, savefile)
    else:
        return table_objs

def flux_measure(image, ):
    pass
