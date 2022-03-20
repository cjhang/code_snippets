"""including the functions dealing with simulation with images
"""
import os
import sys
import numpy as np
from scipy import signal, ndimage 
from scipy.interpolate import CubicSpline
from scipy import interpolate
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.convolution import Gaussian2DKernel, convolve

from image_tools import FitsImage, source_finder, measure_flux

def gkern(bmaj=1., bmin=None, theta=0, size=21,):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    if bmin is None:
        bmin = bmaj
    size = np.max([size, 2*int(bmaj)+1])
    FWHM2sigma = 2.*np.sqrt(2*np.log(2))
    gkernx = signal.gaussian(size, std=bmin/FWHM2sigma).reshape(size, 1)
    gkerny = signal.gaussian(size, std=bmaj/FWHM2sigma).reshape(size, 1)

    kernel = np.outer(gkerny, gkernx)
    kernel_rot = ndimage.rotate(kernel, -1.*theta, reshape=False)
    
    return kernel_rot / np.sum(kernel_rot)

def make_random_source(direction, reffreq=None, n=None, radius=5, 
        prune=True, prune_threshold=3., debug=False, savefile=None, clname=None,
        fluxrange=[0, 1], fluxunit='Jy', known_sources=None,
        sampler=np.random.uniform, sampler_params={}, budget=None):
    """This function used to add random source around a given direction
        
    This is a proximate solution, which treat the field of view as a plane.
    Then calculate the longtitude and latitue using sin and cos functions.

    Args:
        direction: the pointing direction
        n: the number of points
        radius: the largest radius that the points can be put, in arcsec
                it can be a list whose format is [r_min, r_max] both in arcsec
        prune: remove close pairs, make it easier for pointsource testing
        savefile: save the coordination files
        clname: the component list filename
        budget: The total flux density, conflict with flux
        
    Example:
        direction = 'J2000 13h03m49.215900s -55d40m31.60870s'
        direction = '13h03m49.215900s -55d40m31.60870s'

        direction = [195.955, -55.675] # default in deg
        make_random_source(direction, n=20, radius=10)


   """
    # generate a random radius
    # print(direction)
    if isinstance(direction, str):
        if 'J2000' in direction:
            skycoord = SkyCoord(direction[5:])
        else:
            skycoord = SkyCoord(direction)

    if isinstance(direction, (tuple, list, np.ndarray)):
        skycoord = SkyCoord(*direction, unit='deg')
    if debug:
        print('fluxrange', fluxrange)
        print('radius', radius)
    if n:
        theta = 2*np.pi*np.random.uniform(0, 1, n)
        if isinstance(radius, (float, int)):
            rho = radius * np.sqrt(np.random.uniform(0, 1, n))
        elif isinstance(radius, (list, np.ndarray)):
           rho = np.diff(radius)*np.sqrt(np.random.uniform(0, 1, n)) + radius[0]
        flux_sampling = np.array([sampler(**sampler_params) for i in range(n)])
        flux_input = flux_sampling * np.diff(fluxrange) + fluxrange[0]
        if debug:
            print('flux range {} [{}]'.format(fluxrange, fluxunit))

    delta_ra = np.array(rho) * np.cos(theta)/(np.cos(skycoord.dec).value)
    delta_dec = np.array(rho) * np.sin(theta)

    if prune:
        select_idx = []
        n_sources = len(flux_input)
        for i in range(0, n_sources-1):
            if np.sum((delta_ra[i+1:]-delta_ra[i])**2 + (delta_dec[i+1:]-delta_dec[i])**2\
                       < prune_threshold**2) > 0:
                continue
            select_idx.append(i)
        delta_ra = delta_ra[select_idx]
        delta_dec = delta_dec[select_idx]
        flux_input = flux_input[select_idx]
        if debug:
            print("Pruned {} sources.".format(n - len(select_idx)))

    ra_random = (delta_ra*u.arcsec + skycoord.ra).to(u.deg).value
    dec_random = (delta_dec*u.arcsec+ skycoord.dec).to(u.deg).value

    if (known_sources is not None) and (len(known_sources) > 0):
        if isinstance(known_sources, Table):
            ra_known = known_sources['ra'].data
            dec_known = known_sources['dec'].data
        elif isinstance(known_sources, (list, tuple, np.ndarray)):
            ra_known = []
            dec_known = []
            for source in known_sources:
                ra_known.append(source[0])
                dec_known.append(source[1])
        select_idx = []
        n_sources = len(flux_input)
        for i in range(n_sources):
            if np.sum((ra_random[i] - ra_known)**2 + (dec_random[i]-dec_known)**2\
                       < (prune_threshold/3600)**2) > 0:
                continue
            select_idx.append(i)
        ra_random = ra_random[select_idx]
        dec_random = dec_random[select_idx]
        flux_input = flux_input[select_idx]
        if debug:
            print("Remove {} close-by sources.".format(n - len(select_idx)))

    if savefile:
        with open(savefile, 'w+') as sfile:
            sfile.write('# ra[deg]  dec[deg]  flux[{}]\n'.format(fluxunit))
            for ra, dec, flux in zip(ra_random, dec_random, flux_input):
                sfile.write('{:.6f} {:.6f} {:.9f}\n'.format(ra, dec, flux))
        # np.savetxt(savefile, [[ra_random], [dec_random], [flux_random]], 
                # header='#ra[arcsec]  dec[arcsec]  flux[{}]'.format(fluxunit))

    if clname:
        # generate component list
        skycoord_list = SkyCoord(ra=ra_random, dec=dec_random, unit='deg')
        f = lambda x: ('J2000 '+x.to_string('hmsdms')).encode('utf-8')
        direction_list = list(map(f, skycoord_list))
        os.system('rm -rf {}'.format(clname))
        cl.done()
        for d,f in zip(direction_list, flux_input):
            cl.addcomponent(dir=d, flux=f, fluxunit=fluxunit, 
                            freq=reffreq, shape='point',index=0)
        cl.rename(clname)
        cl.done()
        return clname
    else:
        return np.vstack([ra_random, dec_random, flux_input]).T

def add_random_sources(vis=None, fitsimage=None, mycomplist=None, outdir='./', 
        source_shape=None, outname=None, debug=False, **kwargs):
    """
    The sources will be injected to the original file, so only pass in the copied data!
    Args:
        radius (float): in arcsec
        budget: can be a single value, or a list, [mean, low_boundary, high_boundary]
        flux: units in Jy
        source_shape (list,tuple): [bmaj, bmin, theta] for the gaussian shape 

    Notes:
        1. To make the reading and writing more efficiently, the model will be added directly
           into original measurement. It is suggested to make a copy before calling this func
    """
    
    if not os.path.isdir(outdir):
        os.system('mkdir -p {}'.format(outdir))
   
    if vis:
        ft(vis=vis, complist=mycomplist)
        uvsub(vis=vis, reverse=True)
        delmod(vis=vis)
        clearcal(vis=vis)
    if fitsimage:
        # add random sources directly into the image
        hdu = fits.open(fitsimage)
        header = hdu[0].header
        wcs = WCS(header)
        data = hdu[0].data
        ny, nx = data.shape[-2:]
        data_masked = np.ma.masked_invalid(data.reshape(ny, nx))
        # mean, median, std = sigma_clipped_stats(data_masked, sigma=10.0)  

        # for DAOStarFinder, in pixel space
        pixel_scale = 1/np.abs(header['CDELT1'])
        fwhm = header['BMAJ']*3600*u.arcsec
        fwhm_pixel = header['BMAJ']*pixel_scale
        a, b = header['BMAJ']*pixel_scale, header['BMIN']*pixel_scale
        ratio = header['BMIN'] / header['BMAJ']
        theta = header['BPA']
        beamsize = np.pi*a*b/(4*np.log(2))

        # kernel = Gaussian2DKernel(stddev=0.25*(a+b))
        image_kernel = gkern(bmaj=a, bmin=b, theta=theta)
        # print('mycomplist', mycomplist)
        # hdr = wcs.to_header()
        # hdr['OBSERVER'] = 'Your name'
        # hdr['COMMENT'] = "Here's some comments about this FITS file."
        mycomplist_pixels = []
        blank_image = np.zeros((ny, nx))
        for ra, dec, flux in mycomplist:
            ra_pix, dec_pix = map(np.around, skycoord_to_pixel(SkyCoord(ra, dec, unit='deg'), wcs))
            # print(ra_pix, dec_pix, flux)
            mycomplist_pixels.append([int(ra_pix), int(dec_pix), flux])
            blank_image[int(dec_pix), int(ra_pix)] = flux
        if source_shape:
            source_kernel = gkern(*source_shape)
            blank_image = convolve(blank_image, source_kernel)
        fake_image = convolve(blank_image, image_kernel)*beamsize + data_masked # in units of Jy/beam
        fake_image = fake_image.filled(np.nan)
        hdu = fits.PrimaryHDU(header=header, data=fake_image)
        hdu.writeto(os.path.join(outdir , outname+'.fits'), overwrite=True)
        return 

def gen_sim_images(mode='image', vis=None, imagefile=None, outdir='./', basename=None,
                fluxrange = None, snr=[1, 20], fov_scale=1.5,
                n=20, start=0, repeat=1, debug=False, **kwargs):
    """generate the fake images with man-made sources

    Parameters:
     mode: 'image' or 'uv' 
        `image` mode needs the imagefile parameter to be set
        `uv` mode needs vis to be set
    n : int
        The number sources to be generated every time
    repeat : int
        The repeat cycles to be done
    start : int
        The start number of the cycle, default is 0
    outdir : str
        The output directory of the simulated images
    basename : str
        The basename of the output images, baseneme+repeated_number
                        
    """
    if not os.path.isdir(outdir):
        os.system('mkdir -p {}'.format(outdir))
    if basename is None:
        basename = os.path.basename(imagefile)
    if mode == 'image':
        if not imagefile:
            raise ValueError("The image file must be given!")
        fitsimage = FitsImage(imagefile, name=basename)
        try:
            refer = 'J'+str(int(header['EQUINOX']))
        except:
            refer = 'J2000'
        mydirection = fitsimage.direction
        myfreq = "{:.2f}GHz".format(fitsimage.reffreq.to(u.GHz).value)

        fluxrange = np.array(snr) * fitsimage.std
        known_sources = source_finder(fitsimage, detection_threshold=2.5)
        for i in range(repeat):
            i = i + start
            if debug:
                print('run {}'.format(i))
            basename_repeat = basename + '.run{}'.format(i)
            complist_file = os.path.join(outdir, basename_repeat+'.txt')
            mycomplist = make_random_source(mydirection, reffreq=myfreq, 
                    # add 0.9 to aviod puting source to the edge 
                    radius=fov_scale*0.5*0.9*fitsimage.get_fov(), 
                    debug=debug, fluxrange=fluxrange, savefile=complist_file, n=n, 
                    sampler=np.random.uniform, sampler_params={},
                    known_sources=known_sources) 
            add_random_sources(fitsimage=imagefile, mycomplist=mycomplist,
                    outdir=outdir, outname=basename_repeat, debug=debug, **kwargs)
    # adding source in uv has not been test for new scheme
    if mode == 'uv':
        if basename is None:
            basename = os.path.basename(vis)
        if not vis:
            raise ValueError("The visibility must be given!")
        if debug:
            print('basename', basename)
        # vistmp = os.path.join(outdir, basename+'.tmp')
        # split(vis=vis, outputvis=vistmp, datacolumn='data')
        # # read information from vis 
        md = msmdtool()
        if not md.open(vis):
            raise ValueError("Failed to open {}".format(vis))
        phasecenter = md.phasecenter()
        mydirection = phasecenter['refer'] +' '+ SkyCoord(phasecenter['m0']['value'], 
                        phasecenter['m1']['value'], unit="rad").to_string('hmsdms')
        freq_mean = np.mean(read_spw(vis))
        myfreq = "{:.2f}GHz".format(freq_mean)
        if debug:
            print(mydirection)
            print(myfreq)
        tb.open(vis + '/ANTENNA')
        antenna_diameter_list = tb.getcol('DISH_DIAMETER')
        tb.close()
        antenna_diameter = np.max(antenna_diameter_list) * u.m
        wavelength = const.c / (freq_mean * u.GHz) # in um
        fov = (fov_scale * 1.02 * wavelength / antenna_diameter * 206265).decompose().value
        if debug:
            print('fov', fov)
            print('radius', 0.45*fov)
        if imagefile is None:
            imagefile = os.path.join(outdir, basename+'.image.fits')
            if os.path.isfile(imagefile):
                print("Finding the default imagefile: {}".format(imagefile))
            else:
                if debug:
                    print('imagefile', imagefile)
                    print("No image file founded, trying to produce image instead!")
                make_cont_img(vis, outdir=outdir, clean=True, niter=1000, suffix='',
                              only_fits=True, uvtaper_scale=uvtaper_scale, pblimit=-0.01,
                              fov_scale=fov_scale, basename=basename)

        with fits.open(imagefile) as hdu:
            header = hdu[0].header
            # wcs = WCS(header)
            data = hdu[0].data
        ny, nx = data.shape[-2:]
        data_masked = np.ma.masked_invalid(data.reshape(ny, nx))
        mean, median, std = sigma_clipped_stats(data_masked, sigma=5.0, iters=5)  
        sensitivity = std * 1000 # convert to mJy/pixel
     

        if debug:
            print('sensitivity', sensitivity)
            # print('gaussian_norm', gaussian_norm)
            # print('beamsize', beamsize)

        fluxrange = np.array(snr) * sensitivity
        known_sources = source_finder(imagefile, fov_scale=fov_scale)
        clname_fullpath = os.path.join(outdir, basename+'.cl')
        for i in range(repeat):
            i = i + start
            print('run {}'.format(i))
            basename_repeat = basename + '.run{}'.format(i)
            complist_file = os.path.join(outdir, basename_repeat+'.txt')
            if debug:
                print('basename_repeat', basename_repeat)
                print('complist_file', complist_file)
            mycomplist = make_random_source(mydirection, freq=myfreq, 
                    radius=0.5*0.9*fov, # 0.9 is used to aviod add source to the edge 
                    debug=debug, fluxrange=fluxrange, savefile=complist_file, n=n, 
                    sampler=np.random.uniform, sampler_params={}, clname=clname_fullpath,
                    known_sources=known_sources, budget=budget, **kwargs) 
            add_random_sources(vis=vis, mycomplist=mycomplist, fov_scale=fov_scale,
                    outdir=outdir, outname=basename_repeat, debug=debug, **kwargs)
            rmtables(clname_fullpath)
    # if budget:
    # #if snr is None:
    # #TODO: better physical motivated
        # # Limitation from extragalactic background
        # EBL = 14 # 14-18Jy/deg2
        # radius = 0.5*fov
        # budget_mean = (np.pi*(radius*u.arcsec)**2 * 20*u.Jy/u.deg**2).to(u.mJy).value
        # print('budget_mean', budget_mean)
        # budget_sigma = 0.5
        # fluxrange = [1.0, 0.0001]
        # for i in range(repeat):
            # # generate the budget for each run
            # budget = budget_sigma * np.random.randn() + budget_mean

            # basename_new = basename+'.run{}'.format(i)
            # if mode == 'uv':
                # add_random_sources(vis=vistmp, n=None, budget=budget, radius=0.5*fov, outdir=outdir,
                        # uvtaper_scale=uvtaper_scale, basename=basename_new, known_file=known_file, 
                        # inverse_image=inverse_image, fluxrange=fluxrange, debug=debug, **kwargs)
            # elif mode == 'image':
                # add_random_sources(fitsimage=fitsimage, n=None, budget=budget, radius=0.5*fov, outdir=outdir,
                        # uvtaper_scale=uvtaper_scale, basename=basename_new, known_file=known_file, 
                        # inverse_image=inverse_image, fluxrange=fluxrange, debug=debug, **kwargs)

def calculate_sim_images(simfolder, vis=None, baseimage=None, repeat=10, 
        basename=None, savefile=None, fov_scale=1.5, second_check=False,
        detection_threshold=2.5, apertures_scale=5.0,
        plot=False, snr_mode='peak', debug=False,
        snr=[1,20], **kwargs):
    """simulation the completeness of source finding algorithm

    mode:
        peak: snr is the peak value
        integrated: snr is the integrated value

    The second_check is set to false to get the completeness across the whole SNR
    """
    baseimage = FitsImage(image_file=baseimage)

    # known_sources = source_finder(baseimage, fov_scale=fov_scale)
    if basename is None:
        basename = os.path.basename(baseimage)

    # define the statistical variables
    snr_array = np.linspace(*snr, 39)
    # comp_table = Table([snr_array, np.zeros_like(snr_array), np.zeros_like(snr_array)], 
                       # names=['snr_peak', 'n_missing', 'n_input']) 
    # fake_table = Table([snr_array, np.zeros_like(snr_array), np.zeros_like(snr_array)], 
                       # names=['snr_peak', 'n_fake', 'n_found']) 
    boosting_table = Table(names=['snr_peak','flux_input','flux_aperture','flux_gaussian'],
                     dtype=['f8', 'f8', 'f8', 'f8'])
    
    for run in np.arange(repeat):
        if debug:
            print("calculating run: {}".format(run))
        simimage_imagefile = "{basename}.run{run}.fits".format(basename=basename, run=run)
        simimage_sourcefile = "{basename}.run{run}.txt".format(basename=basename, run=run)
        simimage_fullpath = os.path.join(simfolder, simimage_imagefile)
        simimage_sourcefile_fullpath = os.path.join(simfolder, simimage_sourcefile)
        # print("simulated image:", simimage_fullpath)
        # print("simulated sources:", simimage_sourcefile_fullpath)
        
        sources_input = Table.read(simimage_sourcefile_fullpath, format='ascii')
        sources_input_coords = SkyCoord(ra=sources_input['ra[deg]']*u.deg, 
                                        dec=sources_input['dec[deg]']*u.deg)

        simimage = FitsImage(simimage_fullpath)
        sources_found = source_finder(simimage, detection_threshold=detection_threshold, 
                                      method='sep')
        sources_found_coords = SkyCoord(ra=sources_found['ra']*u.deg, 
                                        dec=sources_found['dec']*u.deg)
        
        seplimit = 1*u.arcsec # in arcsec
        if len(sources_input) > 0:
            if len(sources_found) > 0:
                idx_input, idx_found, d2d, d3d = search_around_sky(sources_input_coords, 
                                                                   sources_found_coords,
                                                                   seplimit)
                idx_input_comp = np.array(list(set(range(len(sources_input))) - set(idx_input)), dtype=int)
                idx_found_comp = np.array(list(set(range(len(sources_found))) - set(idx_found)), dtype=int)
                
                snr_array = sources_found['peak_flux'][idx_found] / baseimage.std
                flux_aperture, flux_aperture_err = measure_flux(simimage, 
                                                                detections=sources_found[idx_found], 
                                                                apertures_scale=apertures_scale,
                                                                method='single-aperture')
                flux_gaussian, flux_gaussian_err = measure_flux(simimage, 
                                                                detections=sources_found[idx_found], 
                                                                method='gaussian')
                flux_input = sources_input['flux[Jy]'][idx_input]
                boosting_single = Table([snr_array, flux_input, flux_aperture, flux_gaussian],
                                    names=['snr_peak','flux_input','flux_aperture','flux_gaussian'])
                boosting_table = vstack([boosting_table, boosting_single])
    if savefile:
        boosting_table.write(savefile, format='ascii')
                
    return boosting_table

