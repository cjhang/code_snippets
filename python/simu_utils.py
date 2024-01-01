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
        wcs=None, mask=None, prune_threshold_pixel=5.0,
        sampler=np.random.uniform, sampler_params={}, budget=None):
    """Generate random source around a given direction
        
    This is a proximate solution, which treat the field of view as a plane.
    Then calculate the longtitude and latitue using sin and cos functions.

    Args:
        direction: the pointing direction
        n: the number of points
        radius: the largest radius that the points can be put, in arcsec
                it can be a list whose format is [r_min, r_max] both in arcsec
        prune: remove close pairs, make it easier for pointsource testing
        prune_threshold: in arcsec for known_sources, but pixel for masks
        prune_threshold_pixel: pixel distance for masks
        fluxrange: the flux range in list [flux_min, flow_max], 
                   set flux_min=flux_max to inject source with single flux
        savefile: save the coordination files
        clname: the component list filename
        budget: The total flux density, conflict with flux
        wcs: the wcs to covert the sky coordinates into pixel coordinates
        mask: the pixel space coordinates based mask, which used to exclude 
            the random sources within the mask
        
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
    theta = 2*np.pi*np.random.uniform(0, 1, n)
    if isinstance(radius, (float, int)):
        rho = radius * np.sqrt(np.random.uniform(0, 1, n))
    elif isinstance(radius, (list, np.ndarray)):
       rho = np.diff(radius)*np.sqrt(np.random.uniform(0, 1, n)) + radius[0]
    flux_sampling = np.array([sampler(**sampler_params) for i in range(n)])
    flux_input = flux_sampling * np.diff(fluxrange) + fluxrange[0]
    if debug:
        print('flux range {} [{}]'.format(fluxrange, fluxunit))

    # delta_ra = np.array(rho) * np.cos(theta)/(np.cos(skycoord.dec).value)
    # delta_dec = np.array(rho) * np.sin(theta)
    sky_sources = []
    for i in range(n):
        sky_sources.append(skycoord.directional_offset_by(theta[i], rho[i]*u.arcsec))
    sky_sources = SkyCoord(sky_sources)
    # TODO: maybe a more efficienty ways is injections in pixel plane and the convert they
    # by pixel_to_skycoord

    if prune:
        select_idx = [0]
        for i in range(1, n):
            # if np.sum((delta_ra[i+1:]-delta_ra[i])**2 + (delta_dec[i+1:]-delta_dec[i])**2\
                       # < prune_threshold**2) > 0:
                # continue
            if np.sum(sky_sources[i].separation(sky_sources[:i]) < prune_threshold*u.arcsec) > 0:
                continue
            select_idx.append(i)
        sky_sources = sky_sources[select_idx]
        flux_input = flux_input[select_idx]
        if debug:
            print("Pruned {} sources.".format(n - len(select_idx)))

    ra_random = sky_sources.ra.to(u.deg).value
    dec_random = sky_sources.dec.to(u.deg).value

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
            print("Removed {} close to known sources.".format(n_sources - len(select_idx)))
    if mask is not None:
        if wcs is not None:
            select_idx = []
            xpix, ypix = skycoord_to_pixel(SkyCoord(ra=ra_random, dec=dec_random, unit='deg'), 
                                             wcs=wcs)
            n_sources = len(flux_input)
            mask_pix = np.where(mask)
            for i in range(n_sources):
                if np.sum((xpix[i]-mask_pix[1])**2 + (ypix[i]-mask_pix[0])**2 < 
                           prune_threshold_pixel**2) > 0: # first return of np.where is y axis
                    continue
                select_idx.append(i)
            ra_random = ra_random[select_idx]
            dec_random = dec_random[select_idx]
            flux_input = flux_input[select_idx]
            if debug:
                print("Remove {} close to the mask.".format(n_sources - len(select_idx)))
        else:
            raise ValueError("Mask should be provide with wcs")

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
        source_shape=None, outname=None, debug=False, overwrite=True, **kwargs):
    """Inject the sources into images 

    The sources will be injected to the original file, so only pass in the copied data!
    
    Args:
        vis: the visibility, it is needed if sources are injected into visibility
        fitsimage: the image, it is needed if sources are injected into image
        mycomplist: the flux densities of injected sources
        radius (float): in arcsec
        budget: can be a single value, or a list, [mean, low_boundary, high_boundary]
        flux: units in Jy
        source_shape (list,tuple): [bmaj, bmin, theta] for the gaussian shape
                                   the corresponding peak value will be 
                                   flux_peak = flux/(2*np.pi*bmaj*bmin)

    Notes:
        1. To make the reading and writing more efficiently, the model will be added directly
           into original measurement. It is suggested to make a copy before calling this 
           function
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
        if isinstance(fitsimage, FitsImage):
            wcs = fitsimage.wcs
            header = fitsimage.header
            a = fitsimage.bmaj_pixel
            b = fitsimage.bmin_pixel
            theta = fitsimage.bpa
            ny, nx = fitsimage.imagesize
            beamsize = fitsimage.beamsize
            data_masked = np.ma.masked_invalid(fitsimage.image)
            if outname is None:
                outname = fitsimage.name
        else:
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
            theta = header['BPA']
            beamsize = np.pi*a*b/(4*np.log(2))
            if outname is None:
                outname = os.path.basename(fitsimage)[:-4]

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
        temp_image = convolve(blank_image, image_kernel)
        fake_image = fake_image.filled(np.nan)
        hdu = fits.PrimaryHDU(header=header, data=fake_image)
        hdu.writeto(os.path.join(outdir , outname+'.fits'), overwrite=overwrite)
        return 

def gen_sim_images(mode='image', vis=None, fitsimage=None, outdir='./', basename=None,
                fluxrange = None, snr=[1, 20], fov_scale=1.5, prune_threshold=3.0,
                n=20, start=0, repeat=1, debug=False, mask=None, **kwargs):
    """generate the fake images with man-made sources
    Args:
        mode: 'image' or 'uv' 
            `image` mode needs the imagefile parameter to be set
            `uv` mode needs vis to be set
        n (int): The number sources to be generated every time
    repeat (int): The repeat cycles to be done
    start (int): The start number of the cycle, default is 0
    outdir (str): The output directory of the simulated images
    basename : str
        The basename of the output images, baseneme+repeated_number
    """
    if not os.path.isdir(outdir):
        os.system('mkdir -p {}'.format(outdir))
    if mode == 'image':
        if not isinstance(fitsimage, FitsImage):
            if not os.path.isfile(fitsimage):
                raise ValueError("The image file must be given!")
            fitsimage = FitsImage(fitsimage, name=basename)
        if basename is None:
            basename = fitsimage.name
        mydirection = fitsimage.direction
        myfreq = "{:.2f}GHz".format(fitsimage.reffreq.to(u.GHz).value)

        if fluxrange is None:
            fluxrange = np.array(snr) * fitsimage.std
        for i in range(repeat):
            i = i + start
            if debug:
                print('run {}'.format(i))
            basename_repeat = basename + '.run{}'.format(i)
            complist_file = os.path.join(outdir, basename_repeat+'.txt')
            mycomplist = make_random_source(mydirection, reffreq=myfreq, 
                    # add 0.9 to aviod puting source to the edge 
                    radius=fov_scale*0.5*0.9*fitsimage.get_fov(), 
                    wcs=fitsimage.wcs, mask=mask,
                    debug=debug, fluxrange=fluxrange, savefile=complist_file, n=n, 
                    prune_threshold=prune_threshold,
                    sampler=np.random.uniform, sampler_params={},
                    ) 
            add_random_sources(fitsimage=fitsimage, mycomplist=mycomplist,
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

def calculate_sim_images(simfolder, vis=None, baseimage=None, baseimage_file=None, repeat=10, 
        basename=None, savefile=None, fov_scale=1.5, second_check=False,
        detection_threshold=2.5, aperture_scale=6.0,
        plot=False, snr_mode='peak', debug=False, overwrite=True,
        method='sep', aperture_correction=1.0, 
        sourcesize=0.2, filter_kernel=None,
        snr=[1,20], seplimit_scale=0.5, seplimit_arcsec=None, mask=None, **kwargs):
    """simulation the completeness of source finding algorithm

    mode:
        peak: snr is the peak value
        integrated: snr is the integrated value
        seplimit: in arcsec

    The second_check is set to false to get the completeness across the whole SNR
    """
    if not isinstance(baseimage, FitsImage):
        if os.path.isfile(baseimage_file):
            baseimage = FitsImage(baseimage_file)
        else:
            raise ValueError("No valid baseimage or file has been provided!")
    # recalibrate the std based one the masked image
    baseimage.imagemask = baseimage.mask_image(mask=baseimage.find_structure(sigma=3.0))
    baseimage.imstat()
    # known_sources = source_finder(baseimage, fov_scale=fov_scale)
    if basename is None:
        basename = baseimage.name

    # define the statistical variables
    boosting_table = Table(names=['snr_peak_input', 'snr_peak_found', 'flux_input','flux_aperture','flux_gaussian'],
                     dtype=['f8', 'f8', 'f8', 'f8', 'f8'])
    comp_table = Table(names=['snr_peak', 'is_recovered'], dtype=['f8', 'int']) 
    fake_table = Table(names=['snr_peak', 'is_fake'], dtype=['f8','int']) 
    
    for run in np.arange(repeat):
        if debug:
            print("calculating run: {}".format(run))
        try:
        # if True:
            simimage_imagefile = "{basename}.run{run}.fits".format(basename=basename, run=run)
            simimage_sourcefile = "{basename}.run{run}.txt".format(basename=basename, run=run)
            simimage_fullpath = os.path.join(simfolder, simimage_imagefile)
            simimage_sourcefile_fullpath = os.path.join(simfolder, simimage_sourcefile)
            # print("simulated image:", simimage_fullpath)
            # print("simulated sources:", simimage_sourcefile_fullpath)
            
            ##### for new simulations
            sources_input = Table.read(simimage_sourcefile_fullpath, format='ascii')
            sources_input_coords = SkyCoord(ra=sources_input['ra[deg]']*u.deg, 
                                            dec=sources_input['dec[deg]']*u.deg)
            sources_input_flux = sources_input['flux[Jy]'] #convert to Jy
            ##### for old simulations
            # sources_input = Table.read(simimage_sourcefile_fullpath, format='ascii')
            # sources_input_coords = SkyCoord(ra=sources_input['ra[arcsec]']*u.arcsec, 
                                            # dec=sources_input['dec[arcsec]']*u.arcsec)
            # sources_input_flux = sources_input['flux[mJy]']/1000 #convert to Jy

            simimage = FitsImage(simimage_fullpath)
            sources_found = source_finder(simimage, detection_threshold=detection_threshold, 
                                          method=method, mask=mask, filter_kernel=filter_kernel)
            sources_found_coords = SkyCoord(ra=sources_found['ra']*u.deg, 
                                            dec=sources_found['dec']*u.deg)
            if seplimit_scale is not None:
                seplimit = seplimit_scale * simimage.bmaj*3600 * u.arcsec
            if seplimit_arcsec is not None:
                seplimit = seplimit_arcsec * u.arcsec
            if len(sources_input) > 0:
                if len(sources_found) > 0:
                    idx_input, idx_found, d2d, d3d = search_around_sky(sources_input_coords, 
                                                                       sources_found_coords,
                                                                       seplimit)
                    idx_input_comp = np.array(list(set(range(len(sources_input))) - set(idx_input)), 
                                              dtype=int)
                    idx_found_comp = np.array(list(set(range(len(sources_found))) - set(idx_found)), 
                                              dtype=int)
                    
                    # calculate the flux boosting
                    snr_found_boosting = sources_found['peak_flux'][idx_found] / baseimage.std
                    flux_aperture, flux_aperture_err = measure_flux(simimage, 
                            detections=sources_found[idx_found], 
                            aperture_scale=aperture_scale,
                            aperture_size=[simimage.bmaj*3600, simimage.bmin*3600],
                            sourcesize= sourcesize,# arcsec
                            method='single-aperture')
                    flux_gaussian, flux_gaussian_err = measure_flux(simimage, 
                                                                    detections=sources_found[idx_found], 
                                                                    method='gaussian')
                    flux_input = sources_input_flux[idx_input]
                    snr_input_boosting = sources_input_flux[idx_input] / baseimage.std
                    boosting_single = Table([snr_found_boosting, snr_input_boosting, snr_found_boosting, 
                        flux_input, flux_aperture, flux_gaussian],
                        names=['snr_peak','snr_peak_input','snr_peak_found','flux_input',
                               'flux_aperture','flux_gaussian'])
                    boosting_table = vstack([boosting_table, boosting_single])
                    
                    # calculate the completeness
                    snr_input_completeness = sources_input_flux/baseimage.std
                    is_recovered = np.zeros_like(snr_input_completeness).astype(int)
                    is_recovered[idx_input] = 1
                    is_recovered[idx_input_comp] = 0
                    comp_single = Table([snr_input_completeness, is_recovered], 
                                        names=['snr_peak', 'is_recovered'])
                    comp_table = vstack([comp_table, comp_single])


                    # calculate the false detection
                    snr_found_fakeness = sources_found['peak_flux']/baseimage.std
                    is_fake = np.zeros_like(snr_found_fakeness).astype(int)
                    is_fake[idx_found] = 0
                    is_fake[idx_found_comp] = 1
                    fake_single = Table([snr_found_fakeness, is_fake], names=['snr_peak', 'is_fake'])
                    fake_table = vstack([fake_table, fake_single])
        except:
            print('Failed in run {}'.format(run))
            continue
    if savefile:
        boosting_table.write(savefile+'_boosting.dat', format='ascii', overwrite=overwrite)
        comp_table.write(savefile+'_completeness.dat', format='ascii', overwrite=overwrite)
        fake_table.write(savefile+'_fake.dat', format='ascii', overwrite=overwrite)
                
    return boosting_table


# the end
