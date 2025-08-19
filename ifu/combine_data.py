#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2023-11-22: initial release, v0.1
"""
__version__ = '0.1'

import warnings
import argparse
import textwrap
import glob
import logging
import inspect

import numpy as np
from astropy.io import fits
from astropy import units
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import optimize

from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

from reproject import mosaicking
from reproject import reproject_adaptive, reproject_exact, reproject_interp

def get_wavelength(image, output_unit='um'):
    if isinstance(image, WCS):
        header = image.to_header()
        header['NAXIS3'],header['NAXIS2'],header['NAXIS1'] = image.array_shape
    else:
        #if isinstance(image, fits.Header):
        header = image
    if 'PC3_3' in header.keys():
        cdelt3 = header['PC3_3']
    elif 'CD3_3' in header.keys():
        cdelt3 = header['CD3_3']
    else: 
        cdelt3 = header['CDELT3']
    # because wcs.slice with change the reference channel, the generated ndarray should 
    # start with 1 
    chandata = (header['CRVAL3'] + (np.arange(1, header['NAXIS3']+1)-header['CRPIX3']) * cdelt3)
    if 'CUNIT3' in header.keys():
        chandata = chandata*units.Unit(header['CUNIT3'])
        wavelength = chandata.to(units.Unit(output_unit)).value
    else:
        wavelength = chandata
    return wavelength

def clean_data(data, mask=None, signal_mask=None, sigma_clip=True, median_subtract=True,
               cont_subtract=False, median_filter=True, median_filter_size=(5,3,3),
               channel_chunk=None, sigma=5.0):
    """clean the datacubes

    It supports:
      - sigma clipping
      - background subtraction
      - continue subtraction
    """
    if mask is not None:
        if data.shape != mask.shape:
            raise ValueError("Mask does not match with data!")
    nchan, ny, nx = data.shape
    mask = mask | np.ma.masked_invalid(data).mask

    if median_filter:
        # apply the median filter to filter out the outliers caused by sky lines
        # choose the filter size to preserve weak science data
        datacube = ndimage.median_filter(data, size=median_filter_size)

    # prepare the masked data
    data_masked = np.ma.array(data, mask=mask)

    if sigma_clip:
        # it is only safe to apply the sigma_clip if the signal is weak,
        # and the single exposure is dominated by noise and sky line (residuals)
        data_masked = astro_stats.sigma_clip(data_masked, sigma=sigma, 
                                                 maxiters=2, masked=True) 
        if channel_chunk is not None:
            # apply chunk based sigma clip, it could be useful if different 
            # spectral window show different noise level
            # chose the chunk size so that the signal is weaker than noise
            cube_chunk_mask = np.full_like(data, fill_value=False)
            chunk_steps = np.hstack((np.arange(0, nchan, channel_chunk)[:-1], 
                                   (np.arange(nchan, 0, -channel_chunk)-channel_chunk)[:-1]))
            for chan in chunk_steps:
                chunk_masked = astro_stats.sigma_clip(data_masked[chan:chan+channel_chunk], 
                                                      maxiters=2, sigma=sigma, masked=True)
                cube_chunk_mask[chan:chan+channel_chunk] = chunk_masked.mask
            data_masked = np.ma.array(data_masked, mask=cube_chunk_mask)

        # apply channel-wise sigma_clip (deprecated, too dangerous)
        # datacube_masked = astro_stats.sigma_clip(datacube_masked, maxiters=2, sigma=sigma,
                                                 # axis=(1,2), masked=True)
        
        # apply sigma_clip along the spectral axis
        # make sure to mask signals (through signal_mask) before applying this
        # datacube_masked = astro_stats.sigma_clip(datacube_masked, maxiters=2, sigma=sigma,
                                                 # axis=0, masked=True)

    if median_subtract:
        #
        data_signal_masked = np.ma.array(data_masked, mask=signal_mask)

        # apply a global median subtraction
        data_masked -= np.ma.median(data_signal_masked, axis=0).data[np.newaxis,:,:]

        # row and column based median subtraction
        # by y-axis
        data_masked -= np.ma.median(data_signal_masked, axis=1).data[:,np.newaxis,:]
        # by x-axis
        data_masked -= np.ma.median(data_signal_masked, axis=2).data[:,:,np.newaxis]
        
        data_signal_masked = np.ma.array(data_masked, mask=signal_mask)
        # median subtraction image by image
        spec_median =  np.ma.median(data_signal_masked, axis=(1,2))
        spec_median_filled = fill_mask(spec_median, step=5)
        data_masked -= spec_median_filled[:,np.newaxis, np.newaxis]

    if cont_subtract: # this will take a very long time
        # apply a very large median filter to cature large scale continue variation
        cont_data = ndimage.median_filter(data_signal_masked, size=(200, 10, 10))
        # set the science masked region as the median value of the final median subtracted median
        data_masked = data_masked - cont_data
        
    return data_masked

def find_combined_wcs(image_list=None, wcs_list=None, header_ext='DATA', frame=None, 
                      pixel_size=None, pixel_shifts=None):
    """compute the final coadded wcs

    It suports the combination of the 3D datacubes.
    It uses the first wcs to comput the coverage of all the images;
    Then, it shifts the reference point to the center.
    If spaxel provided, it will convert the wcs to the new spatial pixel size

    Args:
        image_list (list, tuple, np.ndarray): a list fitsfile, astropy.io.fits.header, 
                                              or astropy.wcs.WCS <TODO>
        wcs_list (list, tuple, np.ndarray): a list of astropy.wcs.WCS, need to include
                                            the shape information
        header_ext (str): the extension name of the fits card
        frame (astropy.coordinate.Frame): The sky frame, by default it will use the 
                                          frame of the first image
        pixel_size (float): in arcsec, the final pixel resolution of the combined image <TODO>
        pixel_shifts (list, tuple, np.ndarray): same length as image_list, with each 
                                                element includes the drift in each 
                                                dimension, in the order of [(drift_x(ra),
                                                drift_y(dec), drift_chan),]
    """
    # if the input is fits files, then first calculate their wcs
    if image_list is not None:
        wcs_list = []
        for i,fi in enumerate(image_list):
            with fits.open(fi) as hdu:
                # header = fix_micron_unit_header(hdu[header_ext].header)
                image_wcs = WCS(hdu[header_ext].header)
                wcs_list.append(image_wcs)
    
    # check the shape of the shifts
    n_wcs = len(wcs_list)
    if pixel_shifts is not None:
        if len(pixel_shifts) != n_wcs:
            raise ValueError("Pixel_shift does not match the number of images or WCSs!")
        pixel_shifts = np.array(pixel_shifts)

    # get the wcs of the first image
    first_wcs = wcs_list[0] 
    first_shape = first_wcs.array_shape # [size_chan, size_y, size_x]
    naxis = first_wcs.wcs.naxis

    # then looping through all the images to get the skycoord of the corner pixels
    if naxis == 2: # need to reverse the order of the shape size
        # compute the two positions: [0, 0], [size_x, size_y]
        corner_pixel_coords = [[0,0], np.array(first_shape)[::-1]-1] # -1 because the index start at 0
    elif naxis == 3:
        # compute three positions: [0,0,0], [size_x, size_y, size_chan]
        corner_pixel_coords = [[0,0,0], np.array(first_shape)[::-1]-1]
    else: 
        raise ValueError("Unsupport datacube! Check the dimentions of the datasets!")
    image_wcs_list = []
    corners = []
    resolutions = []
    for i,fi in enumerate(wcs_list):
            image_wcs = wcs_list[i]
            if pixel_shifts is not None:
                image_wcs.wcs.crpix -= pixel_shifts[i]
            array_shape = image_wcs.array_shape
            # get the skycoord of corner pixels
            for pixel_coord in corner_pixel_coords:
                # pixel order: [x, y, chan]
                corner = wcs_utils.pixel_to_pixel(image_wcs, first_wcs, *pixel_coord)
                corners.append(corner)
            resolutions.append(wcs_utils.proj_plane_pixel_scales(image_wcs))

    # calculate the reference point
    corners = np.array(corners)
    low_boundaries = np.min(corners, axis=0)
    up_boundaries = np.max(corners, axis=0)
    ranges = np.round(up_boundaries - low_boundaries + 1).astype(int) # [range_x, range_y, range_chan]
    chan0 = low_boundaries[0]
    x0, y0 = ranges[:2]*0.5 # only need the first two for x and y

    # get the skycoord of the reference point
    reference_skycoord = wcs_utils.pixel_to_skycoord(x0, y0, wcs=first_wcs)

    # assign the new reference to the new wcs
    wcs_combined = first_wcs.deepcopy()
    if naxis == 3:
        # shift the reference point to the center
        # reference channel point the first channel of the combined data
        try: dchan = first_wcs.wcs.cd[-1,-1]
        except:
            try: dchan = first_wcs.wcs.pc[-1,-1]
            except:  
                raise ValueError("Cannot read the step size of the spectral dimension!")

        reference_chan = first_wcs.wcs.crval[-1] + (first_wcs.wcs.crpix[-1]-chan0-1)*dchan
        wcs_combined.wcs.crval = np.array([reference_skycoord.ra.to(units.deg).value, 
                         reference_skycoord.dec.to(units.deg).value,
                         reference_chan])
        wcs_combined.wcs.crpix = np.array([x0, y0, 1])
        # wcs_combined.wcs.cdelt = wcs.wcs.cd.diagonal() # cdelt will be ignored when CD is present
    elif naxis == 2:
        wcs_combined.wcs.crval = np.array([reference_skycoord.ra.to(units.deg).value, 
                                           reference_skycoord.dec.to(units.deg).value])
    wcs_combined.array_shape = tuple(ranges[::-1]) # need to reverse again

    # by default, the pixel size of the first image will be used
    # update the pixel size if needed
    # if pixel_size is not None: #<TODO>: untested
        # min_resolutions = np.min(np.array(resolutions), axis=1)
        # scales = min_resolutions / first_resolutions
        # wcs_new = wcs_combined.deepcopy()
        # wcs_new.wcs.cd = wcs_new.wcs.cd * scales
        # if (scales[-1] - 1) > 1e-6:
            # nchan = int(wcs_combined.array_shape[0] / scales)
        # x0_new, y0_new = wcs_utils.skycoord_to_pixel(reference_skycoord)
        # wcs_new.crpix = np.array([x0_new.item(), y0_new.item(), 1]).astype(int)
        # wcs_new.array_shape = tuple(np.round(np.array(wcs_combined.array_shape) / scales).astype(int))
        # wcs_combined = wcs_new
    return wcs_combined

def find_combined_wcs_test(image_list, wcs_list=None, header_ext='DATA', frame=None, 
                           pixel_size=None, ):
    """this is just a wrapper of reproject.mosaicking.find_optimal_celestial_wcs
    
    Used to test the performance of `find_combined_wcs`
    """
    # define the default values
    image_wcs_list = []
    for img in image_list:
        # read the image part
        with fits.open(img) as hdu:
            header = hdu[header_ext].header
            image_shape = (header['NAXIS2'], header['NAXIS1'])
            nchan = header['NAXIS3']
            image_wcs = WCS(header).celestial #sub(['longitude','latitude'])
            if frame is None:
                frame = wcs_utils.wcs_to_celestial_frame(image_wcs)    
            image_wcs_list.append((image_shape, image_wcs))
    wcs_combined, shape_combined = mosaicking.find_optimal_celestial_wcs(
            tuple(image_wcs_list), frame=frame, resolution=pixel_size)
    return wcs_combined, shape_combined

def gaussian_2d(params, x=None, y=None):
    """a simple 2D gaussian function
    
    Args:
        params: all the free parameters
                [amplitude, x_center, y_center, x_sigma, y_sigma, beta]
        x: x grid
        y: y grid
    """
    amp, x0, y0, xsigma, ysigma, beta = params
    return amp*np.exp(-0.5*(x-x0)**2/xsigma**2 - beta*(x-x0)*(y-y0)/(xsigma*ysigma)- 0.5*(y-y0)**2/ysigma**2)

def find_offset(filelist, header_ext='DATA', reference_coords=None, crop_radius=20, fit=True,
                outfile=None):
    if fit:
        def _cost(params, image, xgrid, ygrid, rms=1):
            # rmap = (xgrid-params[1])**2 + (ygrid-params[2])**2
            return np.sum((image - gaussian_2d(params, xgrid, ygrid))**2/(rms**2))

    reference_coords = []

    fig = plt.figure(figsize=(8, 8))

    for filename in filelist:
        with fits.open(filename) as hdu:
            header = hdu[header_ext].header
            data = hdu[header_ext].data
            image_wcs = WCS(header)
        if data.ndim > 2:
            # for cubes, collapse the cube to create the image
            image = np.median(np.ma.masked_invalid(data).filled(0), axis=0)

        elif data.ndim == 2:
            image = data
        else:
            raise ValueError("Unsupport data dimensions!")

        vmin = np.nanpercentile(image, 1)
        vmax = np.nanpercentile(image, 99)
        plt.title('Click the point source in the image', fontsize=14)
        plt.imshow(image, origin='lower', vmin=vmin, vmax=vmax)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.draw()
        # plt.colorbar()
        coords = plt.ginput(1, timeout=10)[0]#, timeout=5
        plt.show(block=False)
        xp, yp = coords
        xp_int, yp_int = int(xp), int(yp)
        print(xp_int, yp_int)
        
        if crop_radius is not None:
            crop_radius = np.min([xp_int, yp_int, crop_radius])
            # crop a subimage to make the gaussian fitting
            image_croped = image[yp_int-crop_radius:yp_int+crop_radius,
                                 xp_int-crop_radius:xp_int+crop_radius]
        else:
            image_croped = image
        yshape, xshape = image_croped.shape
        ygrid, xgrid = np.mgrid[0:yshape,0:xshape]
        image_normed = image_croped/np.percentile(image_croped, 98)
        p_init = [1.2, crop_radius, crop_radius, 0.5, 0.5, 0]
        res_minimize = optimize.minimize(_cost, p_init, args=(image_normed, xgrid, ygrid), method='BFGS')
        # print(res_minimize)
        # plot the fit
        # fit_image = gaussian_2d(res_minimize.x, xgrid, ygrid)
        # fig = plt.figure(figsize=(12, 3))
        # gs = gridspec.GridSpec(1, 3)
        # ax1 = fig.add_subplot(gs[0])
        # ax1.imshow(image_normed, origin='lower')
        # ax1.plot([crop_radius], [crop_radius], 'x', color='red')
        # ax2 = fig.add_subplot(gs[1])
        # ax2.imshow(fit_image, origin='lower')
        # ax3 = fig.add_subplot(gs[2])
        # ax3.imshow(image_normed - fit_image, origin='lower')
        # plt.pause(10)
        # plt.close(fig)

        amp_fit, x0_fit, y0_fit, xsigma_fit, ysigma_fit, beta_fit = res_minimize.x

        reference_coords.append([x0_fit-crop_radius+xp_int, y0_fit-crop_radius+yp_int])
    if outfile is not None:
        reference_coords = np.array(reference_coords)
        np.savetxt(reference_coords - reference_coords[0])

    return np.array(reference_coords)

def combine_data(filelist, data_ext='DATA', mask_ext=None,  
                 pixel_shifts=None, ignore_wcs=False, z=None,
                 header_ext=None, weighting=None, frame=None, projection='TAN', 
                 sigma_clip=False, median_subtract=False, cont_subtract=False,
                 reproject_mode='interp',
                 pixel_size=None, savefile=None, overwrite=False):
    """combine the multiple observation of the same target

    By default, the combined wcs uses the frame of the first image

    sigma_clip, apply if there are large number of frames to be combined
    background: global or chennel-per-channel and row-by-row
    Args:
        bgsub (bool): set to true to subtract global thermal background
        sigma_clip (bool): set to true to apply sigma_clip with the sigma controlled
                           by `sigma`
        sigma (bool): the deviation scale used to control the sigma_clip
        reproject_mode: default 'interp', supported 'interp', 'adaptive', 'exact'
    """
    # define the default variables
    nimages = len(filelist)
    if header_ext is None:
        header_ext = data_ext

    # check the input variables
    if pixel_shifts is not None:
        if len(pixel_shifts) != nimages:
            raise ValueError("Pixel_shift does not match the number of images!")
        pixel_shifts = np.array(pixel_shifts)

    if ignore_wcs:
        print("Combine data with the input offsets")
        # ignore the wcs and mainly relies on the pixel_shifts to align the images
        # to make advantage of reproject, we still need a roughly correct wcs or 
        # a mock wcs
        # first try to extract basic information from the first image
        with fits.open(filelist[0]) as hdu:
            # header = fix_micron_unit_header(hdu[header_ext].header)
            header = hdu[header_ext].header
            try:
                # if the header have a rougly correct wcs
                wcs_mock = WCS(header)
            except:
                wcs_mock = construct_wcs(header, data_shape=None)
        # shifting the mock wcs to generate a series of wcs
        wcs_list = []
        if pixel_shifts is not None:
            for i in range(nimages):
                wcs_tmp = wcs_mock.deepcopy()
                x_shift, y_shift = pixel_shifts[i]
                wcs_tmp.wcs.crpix += np.array([x_shift, y_shift, 0])
                wcs_list.append(wcs_tmp)
        else:
            wcs_list = [wcs_mock]*nimages
    else:
        print("Combine data with the wcs")
        wcs_list = []
        # looping through the image list to extract their wcs
        for i,fi in enumerate(filelist):
            with fits.open(fi) as hdu:
                # this is to fix the VLT micron header
                # header = fix_micron_unit_header(hdu[header_ext].header)
                header = hdu[header_ext].header
                image_wcs = WCS(header)
                wcs_list.append(image_wcs)
    # compute the combined wcs 
    wcs_combined = find_combined_wcs(wcs_list=wcs_list, frame=frame, 
                                     pixel_size=pixel_size)
    shape_combined = wcs_combined.array_shape
    if len(shape_combined) == 3:
        nchan, size_y, size_x = shape_combined
    elif len(shape_combined) == 2:
        size_y, size_x = shape_combined

    # define the combined cube
    image_shape_combined = shape_combined[-2:]
    data_combined = np.full(shape_combined, fill_value=0.)
    coverage_combined = np.full(shape_combined, fill_value=1e-8)
    

    # handle the weighting
    if weighting is None:
        # treat each dataset equally
        weighting = np.full(nimages, fill_value=1./nimages)
    
    # reproject each image to the combined wcs
    for i in range(nimages):
        logging.info(f"Working on filelist[i]")
        image_wcs = wcs_list[i].celestial
        data = fits.getdata(filelist[i], data_ext)
        img_ny, img_nx = data.shape[-2:]
        
        if mask_ext is not None:
            img_mask = fits.getdata(filelist[i], mask_ext)
        else:
            img_mask = np.full(data.shape, fill_value=False)

        # check if the channel length consistent with the combined wcs
        if data.ndim == 3: #<TODO>: find a better way to do
            if len(data) != shape_combined[0]:
                logging.warning("Combining data with different channels!")
            if len(data) >= shape_combined[0]:
                data = data[:shape_combined[0]]
                img_mask = img_mask[:shape_combined[0]]
            else:
                data_shape = data.shape
                data_shape_extend = [shape_combined[0], data_shape[1], data_shape[2]]
                data_extend = np.full(data_shape_extend, fill_value=0.)
                mask_extend = np.full(data_shape_extend, fill_value=False)
                data_extend[:data_shape[0]] = data[:]
                mask_extend[:data_shape[0]] = img_mask[:]
                data = data_extend
                img_mask = mask_extend
        
            # get the wavelength
            if z is not None:
                wavelength = get_wavelength(wcs_combined)
                wave_mask = mask_spectral_lines(wavelength, redshift=z)
                cube_wave_mask = np.repeat(wave_mask, img_ny*img_nx).reshape(len(wavelength), 
                                                                             img_ny, img_nx)
            else:
                cube_wave_mask = None

        data_masked = clean_data(data, mask=img_mask, signal_mask=cube_wave_mask, 
                                 sigma_clip=sigma_clip, median_subtract=median_subtract,
                                 cont_subtract=cont_subtract)

        data = data_masked.filled(0)
        mask = data_masked.mask
        # 
        data_reprojected, footprint = reproject_interp((data, image_wcs), 
                                                          wcs_combined.celestial, 
                                                          shape_out=shape_combined,)
                                                          # conserve_flux=True)
        mask_reprojected, footprint = reproject_interp((mask, image_wcs), 
                                                          wcs_combined.celestial, 
                                                          shape_out=shape_combined,)
                                                          # conserve_flux=False)
        data_combined += data_reprojected * weighting[i]
        footprint = footprint.astype(bool)
        coverage_combined += (1.-mask_reprojected)
        # error2_combined += error_reprojected**2 * weighting[i]
    # error_combined = np.sqrt(error2_combined)
    data_combined = data_combined / coverage_combined

    if savefile is not None:
        # save the combined data
        hdr = wcs_combined.to_header() 
        hdr['OBSERVER'] = 'combine_data.py'
        hdr['COMMENT'] = 'Combined by combine_data.py'
        # reset the cdelt
        if 'CD1_1' in header.keys():
            header['CDELT1'] = header['CD1_1']
            header['CDELT2'] = header['CD2_2']
            header['CDELT3'] = header['CD3_3']
        elif 'PC1_1' in header.keys():
            header['CDELT1'] = header['PC1_1']
            header['CDELT2'] = header['PC2_2']
            header['CDELT3'] = header['PC3_3']
        primary_hdu = fits.PrimaryHDU(header=hdr)
        data_combined_hdu = fits.ImageHDU(data_combined, name="DATA", header=hdr)
        hdus = fits.HDUList([primary_hdu, data_combined_hdu])
        hdus.writeto(savefile, overwrite=overwrite)
    else:
        return data_combined

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='eris_jhchen_utils.py',
            description="Welcome to jhchen's ERIS utilities v{}".format(__version__),
            epilog='Reports bugs and problems to jhchen@mpe.mpg.de')
    parser.add_argument('--esorex', type=str, default='esorex',
                        help='specify the customed esorex')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    parser.add_argument('--logfile', help='the logging output file')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')


    parser = argparse.ArgumentParser(
            usage='%(prog)s [options]',
            prog='combine_data.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''\
            Combine reduced datacubes
            -------------------------
            Examples:

            $ combine_data --savefile test2.fits
                                        '''),
            epilog=f'Reports bugs and problems to cjhastro@gmail.com, v{__version__}')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    parser.add_argument('--dry_run', action='store_true',
                        help='print the commands but does not execute them')
    parser.add_argument('-v','--version', action='version', version=f'v{__version__}')
    
    parser.add_argument('--filelist', nargs='+', 
                        help='The files or the wildcards of the files to be combined')
    parser.add_argument('--pixel_size', help='Pixel size of the combined file')
    parser.add_argument('--weighting', help='The weighting of the each combined file')
    parser.add_argument('--data_ext', default='DATA', help='The header extension name of the data')
    parser.add_argument('--mask_ext', help='The header extension name of the mask, optional')
    parser.add_argument('--z', help='The redshift of the target')
    parser.add_argument('--savefile', help='The output name of the saved file')
    parser.add_argument('--pixel_shifts', help='Pixel shifts of the files, coded in a text file')
    parser.add_argument('--ignore_wcs', action='store_true', help='To ignore the wcs')
    parser.add_argument('--crop_radius', type=int, help='the radius of the pixel cropping size')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file')

    ################################################
    # combine data
    # subp_combine_data = subparsers.add_parser('combine_data',
            # formatter_class=argparse.RawDescriptionHelpFormatter,
            # description=textwrap.dedent('''\
            # Combine reduced datacubes
            # -------------------------
            # Examples:

              # eris_jhchen_utils run_eris_pipeline -d science_raw -o science_output -c calibPool
                                        # '''))
    args = parser.parse_args()
    
    if args.debug:
        logging.debug(args)
        func_args = list(inspect.signature(locals()['combine_data']).parameters.keys())
        func_str = f"Executing:\n \t combine_data("
        for ag in func_args:
            try: func_str += f"{ag}={args.__dict__[ag]},"
            except: func_str += f"{ag}=None, "
        func_str += ')\n'
        print(func_str)
        logging.info(func_str)

    if len(args.filelist) == 1:
        filelist = glob.glob(args.filelist[0])
        print(filelist)
        if len(filelist) < 1:
            raise ValueError(f'No files have been matched by {filelist}')
    else:
        filelist = args.filelist

    if args.pixel_shifts == 'interactive':
        pixel_coords = find_offset(filelist, crop_radius=args.crop_radius)
        offsets = pixel_coords - pixel_coords[0]
        print(offsets)
    else:
        offsets = None

    combine_data(filelist, data_ext=args.data_ext, mask_ext=args.mask_ext, 
                 pixel_shifts=offsets, ignore_wcs=args.ignore_wcs, 
                 z=args.z, 
                 pixel_size=args.pixel_size, savefile=args.savefile,
                 overwrite=args.overwrite,
                 )

