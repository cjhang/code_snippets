#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    - 2024-10-01
"""

__version__ = '0.0.1'

import os, sys, glob, re
import numpy as np
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.table as table
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage, optimize, stats, interpolate, signal


#####################################################################################
# Kinematic models

def exponential(radius, vmax):
    vel = vmax * (1.0 - np.exp(-radius))
    return vel
def arctan(radius, vmax):
    vel = np.arctan(radius) * vmax * 2. / np.pi
    return vel
def tanh(radius, vmax):
    vel = vmax * np.tanh(radius)
    return vel

#####################################################################################
# Profiles
def sersic_2d(params, x, y):
    """
    Args:
    params: [amplitude, reff, n, x0, y0, ellip, theta], 
            theta is the angle relative to the positive x-axis
    x: 
    y:
    """
    amplitude, reff, n, x0, y0, ellip, theta = params
    bn = scipy.special.gammaincinv(2.0 * n, 0.5)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_maj = np.abs((x - x0) * cos_theta + (y - y0) * sin_theta)
    x_min = np.abs(-(x - x0) * sin_theta + (y - y0) * cos_theta)

    b = (1 - ellip) * reff
    expon = 2.0
    inv_expon = 1.0 / expon
    z = ((x_maj / reff) ** expon + (x_min / b) ** expon) ** inv_expon
    return amplitude * np.exp(-bn * (z ** (1 / n) - 1.0))


#####################################################################################
# Galaxy models

def create_thin_disk(nx, ny, vmax=400, inc=0, pa=0, pixel_size=1):
    xcoord = (np.arange(nx)-0.5*nx+0.5)*pixel_size
    ycoord = (np.arange(ny)-0.5*nx+0.5)*pixel_size
    xdisk, ydisk = np.meshgrid(xcoord, ycoord)
    rdisk = np.sqrt(xdisk**2 + ydisk**2)
    # rotating the major axis to PA, 0 towards north and counterclock-wise as positive
    # we need first convert the PA (relative to North) angle to theta (relative to
    # the positive x-axis)
    theta_from_pa = 0.5*np.pi - pa# 
    xdisk_new = xdisk*np.cos(theta_from_pa) - ydisk*np.sin(theta_from_pa)
    ydisk_new = xdisk*np.sin(theta_from_pa) + ydisk*np.cos(theta_from_pa)
    vel_disk = exponential(rdisk, vmax=vmax)
    pa_disk = np.arctan2(ydisk_new, xdisk_new)
    vel_map = vel_disk*np.sin(inc)*np.cos(pa_disk)
    return vel_map

def create_sersic_disk(nx, ny, params, pixel_size=1):
    xcoord = (np.arange(nx)-0.5*nx+0.5)*pixel_size
    ycoord = (np.arange(ny)-0.5*nx+0.5)*pixel_size
    xmap, ymap = np.meshgrid(xcoord, ycoord)
    flux_map = sersic_2d(params, xmap, ymap)
    return flux_map
    

#####################################################################################
# helper functions

def get_wavelength(header=None, wcs=None, output_unit=None):
    """
    copied from cube_utils.py
    """
    if header is None:
        try:
            header = wcs.to_header()
            header['NAXIS3'],header['NAXIS2'],header['NAXIS1'] = wcs.array_shape
        except:
            raise ValueError("Please provide valid header or wcs!")
    # if 'PC3_3' in header.keys():
        # cdelt3 = header['PC3_3']
    # elif 'CD3_3' in header.keys():
        # cdelt3 = header['CD3_3']
    # else: 
    cdelt3 = header['CDELT3']
    # because wcs.slice with change the reference channel, the generated ndarray should 
    # start with 1 
    chandata = (header['CRVAL3'] + (np.arange(1, header['NAXIS3']+1)-header['CRPIX3']) * cdelt3)
    if 'CUNIT3' in header.keys():
        chandata = chandata*units.Unit(header['CUNIT3'])
        if output_unit is not None:
            chandata = chandata.to(units.Unit(output_unit)).value
    return chandata

def read_cube(cubefile, z=None, rest_wave=None, header_ext='DATA',):
    """read eris datacube and convert the wavelength to velocity relative to Ha
    """
    with fits.open(cubefile) as hdu:
        header = hdu[header_ext].header
        cube = hdu[header_ext].data
    header = fix_micron_unit_header(header)
    wavelength = get_wavelength(header, output_unit='um')
    if z is not None:
        refwave = (rest_wave * (1+z)).to(u.um).value
        velocity = 299792.458 * (wavelength-refwave)/refwave
        return velocity, cube, header
    return wavelength, cube, header

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

def gaussian_1d(params, velocity):
    """a simple 1D gaussian function
    Args:
        params: all the free parameters
                [amplitude, center, sigma, cont]
        v: velocity
    """
    n_param = len(params)
    amp, v0, sigma = params[:3]
    if n_param == 4:
        cont = params[3]
    else:
        cont = 0
    # return amp / ((2*np.pi)**0.5 * sigma) * np.exp(-0.5*(velocity-v0)**2/sigma**2) + cont
    return amp * np.exp(-0.5*(velocity-v0)**2/sigma**2) + cont

def calc_gaussian1d_chi2(params, velocity=None, spec=None, std=None):
    fit = gaussian_1d(params, velocity=velocity)
    chi2 = np.sum((spec-fit)**2/std**2)
    return chi2


#####################################################################################
# Fitters
def fit_spec(vel, spec, std, mode='minimize', plot=False, ax=None,
             p0=None,
             sigma_guess=30, velocity_bounds=[-500,500], 
             amplitude_bounds=None, sigma_bouds=[10,1000], 
             fit_cont=True, cont_bounds=None, 
             nstep=2000, discard=None):
    """a simple spectrum fitter
    """
    # normalize the spectrum to aviod numerical issue
    amp_scale = np.nanmax(spec)
    vel_scale = abs(vel[-1]-vel[0])
    spec_norm = spec/amp_scale
    std_norm = std/amp_scale
    if amplitude_bounds is None:
        amplitude_bounds_scaled = [np.min(spec_norm), np.max(spec_norm)]
    else:
        amplitude_bounds_scaled = [amplitude_bounds[0]/amp_scale, amplitude_bounds[1]/amp_scale]
    if fit_cont:
        if cont_bounds is None:
            cont_bounds_scaled = [np.min(spec_norm), np.max(spec_norm)]
        else:
            cont_bounds_scaled = [cont_bounds[0]/amp_scale, cont_bounds[1]/amp_scale]
    vel_norm = vel/vel_scale
    velocity_bounds_scaled = [velocity_bounds[0]/vel_scale, velocity_bounds[1]/vel_scale]
    # guess the initial parameters
    if p0 is not None:
        p0 = np.array(p0)
        I0, vel0, sig0 = p0[:3]/np.array([amp_scale, vel_scale, vel_scale])
    else:
        I0 = np.max(spec_norm)
        vel0 = 0 #np.median(vel[spec_selection])
        sig0 = sigma_guess/vel_scale
    if fit_cont:
        if (p0 is not None) and (len(p0)>3):
            cont0 = p0[3]
        else:
            spec_selection = (vel>velocity_bounds[0]) & (vel<velocity_bounds[1])
            cont0 = np.median(spec_norm[~spec_selection])
    sigma_bouds_scaled = [sigma_bouds[0]/vel_scale, sigma_bouds[1]/vel_scale]

    if mode == 'minimize':
        bounds = [amplitude_bounds_scaled, velocity_bounds_scaled, sigma_bouds_scaled]
        if fit_cont:
            initial_guess = [I0, vel0, sig0, cont0]
            bounds.append(cont_bounds_scaled)
        else:
            initial_guess = [I0, vel0, sig0]
        # make profile of initial guess.  this is only used for information/debugging purposes.
        guess = gaussian_1d(initial_guess, velocity=vel_norm)
        # do the fit
        fit_result = optimize.minimize(calc_gaussian1d_chi2, initial_guess, 
                                       args=(vel_norm, spec_norm, std_norm), bounds=bounds)
        # make profile of the best fit 
        bestfit = gaussian_1d(fit_result.x, velocity=vel_norm)
        if plot:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            ax.step(vel_norm, spec_norm, color='black', where='mid', label='data')
            ax.plot(vel_norm, guess, linestyle='dashed', color='blue', label='guess')
            # ax.plot(vel, spec*0+cont0+std, linestyle='dashed', color='blue', alpha=0.5)
            # ax.plot(vel, spec*0+cont0-std, linestyle='dashed', color='blue', alpha=0.5)
            ax.plot(vel_norm, bestfit, color='red')
            plt.show()
        bestfit_params = fit_result.x
    if mode == 'mcmc':
        try:
            from model_utils import Gaussian1D, Models, Parameter, EmceeFitter
        except:
            raise ImportError("Install/copy model_utils to the PYTHONPATH!")
        if fit_cont:
            gaussian_model = Gaussian1D(amp=Parameter(I0, limits=amplitude_bounds_scaled),
                                        mean=Parameter(vel0, limits=velocity_bounds_scaled),
                                        sigma=Parameter(sig0, limits=sigma_bouds_scaled),
                                        cont=Parameter(cont0, limits=cont_bounds_scaled),
                                        name='gaussian1d')
        else:
            gaussian_model = Gaussian1D(amp=Parameter(I0, limits=amplitude_bounds_scaled),
                                        mean=Parameter(vel0, limits=velocity_bounds_scaled),
                                        sigma=Parameter(sig0, limits=sigma_bouds_scaled),
                                        cont=0, name='gaussian1d')
        models = Models([gaussian_model])
        parameters = models.get_parameters()
        fitter = EmceeFitter(models, vel_norm, spec_norm, std_norm)
        fitter.run(progress=False, nstep=nstep)
        if discard is None:
            discard = int(0.1*nstep)
        samples = fitter.sampler.get_chain(discard=discard, flat=True,)
        bestfit_values = np.percentile(samples, 50, axis=0)
        bestfit_dict = {}
        for key, value in zip(parameters.keys(), bestfit_values):
            bestfit_dict[key] = value
        if fit_cont:
            bestfit_params = [bestfit_dict['gaussian1d.amp'], bestfit_dict['gaussian1d.mean'], 
                              bestfit_dict['gaussian1d.sigma'], bestfit_dict['gaussian1d.cont']]
        else:
            bestfit_params = [bestfit_dict['gaussian1d.amp'], bestfit_dict['gaussian1d.mean'], 
                              bestfit_dict['gaussian1d.sigma']]
    if fit_cont:
        bestfit_params = bestfit_params*np.array([amp_scale, vel_scale, vel_scale, amp_scale])
    else:
        bestfit_params = bestfit_params*np.array([amp_scale, vel_scale, vel_scale])
    return bestfit_params

def fit_spec_mcmc(vel, spec, std, sigma_guess=30, velocity_bounds=[-500,500], 
                  amplitude_bounds=[0, np.inf], sigma_bouds=[10,1000], 
                  fit_cont=True, cont_bounds=[0, np.inf], 
                  nstep=2000, debug=False,):
    from model_utils import Gaussian1D, Models, Parameter, EmceeFitter
    # normalize the spectrum to aviod numerical issue
    norm_scale = 1 #np.ma.max(spec)
    amplitude_bounds = [np.min(spec), np.max(spec)]
    cont_bounds = amplitude_bounds 
    spec_norm = spec/norm_scale
    # guess the initial parameters
    #spec_selection = spec_norm > np.percentile(spec_norm, 90)
    I0 = 0.8 #2*np.mean(spec_norm[spec_selection])
    vel0 = 0 #np.median(vel[spec_selection])
    vsig0 = sigma_guess
    if fit_cont:
        cont0 = np.median(spec_norm[~spec_selection])
        gaussian_model = Gaussian1D(amp=Parameter(I0, limits=amplitude_bounds),
                                    mean=Parameter(vel0, limits=velocity_bounds),
                                    sigma=Parameter(vsig0, limits=sigma_bouds),
                                    cont=Parameter(cont0, limits=cont_bounds),
                                    name='gaussian1d')
    else:
        gaussian_model = Gaussian1D(amp=Parameter(I0, limits=amplitude_bounds),
                                    mean=Parameter(vel0, limits=velocity_bounds),
                                    sigma=Parameter(vsig0, limits=sigma_bouds),
                                    cont=0, name='gaussian1d')
    models = Models([gaussian_model])
    parameters = models.get_parameters()
    fitter = EmceeFitter(models, vel, spec_norm, std)
    fitter.run(progress=debug, nstep=nstep)
    samples = fitter.sampler.get_chain(discard=200, flat=True,)
    bestfit_values = np.percentile(samples, 50, axis=0)
    bestfit_params = {}
    for key, value in zip(parameters.keys(), bestfit_values):
        if ('amp' in key) or ('cont' in key):
            bestfit_params[key] = value * norm_scale
        else:
            bestfit_params[key] = value
    if fit_cont:
        return bestfit_params['gaussian1d.amp'], bestfit_params['gaussian1d.mean'], bestfit_params['gaussian1d.sigma'], bestfit_params['gaussian1d.cont']
    else:
        return bestfit_params['gaussian1d.amp'], bestfit_params['gaussian1d.mean'], bestfit_params['gaussian1d.sigma']


def pixel_fit_cube(cube, velocity=None, mode='minimize', SNR_limit=3, 
                   plot=False, fit_cont=True,
                   smooth_width=None, sigma_guess=30, 
                   minaper=0, maxaper=4, 
                   vel_low=-300, vel_up=300, savefile=None, debug=False):
    """fit gaussian line through the cube

    Args:
        datacube: it can be the fitsfile or the datacube
        box: the selection pixel of a box 'z1,z2,y1,y2,x1,x2', 
             (x1, y1): the bottom left coord
             (x2, y2): upper right coord
    """
    if isinstance(cube, str):
        velocity, cube, header = read_cube(cube, z=z, rest_wave=rest_wave)
    if smooth_width is not None:
        gauss_kernel = gkern(smooth_width) 
        cube = signal.convolve(cube, gauss_kernel[None,:,:], mode='same')

    cube_shape = cube.shape
    fitcube = np.zeros_like(cube)
    imagesize = cube_shape[-2:]
    nspec = cube_shape[-3]
    # mean, median, std = astro_stats.sigma_clipped_stats(cube, sigma=10)
    # mask = np.ma.masked_invalid(cube).mask
    # vmax = 1.5*np.percentile(cube[~mask], 90)

    # A cube to save all the best-fit values (maps)
    # [amp, velocity, sigma, SNR]
    fitmaps = np.full((5, cube_shape[-2], cube_shape[-1]), fill_value=0.0)
    weight_map = np.zeros(imagesize)
    
    if plot:
        fig = plt.figure()
    # loop over all pixels in the cube and fit 1d spectra
    for y in range(0, cube_shape[-2]):
        for x in range(0, cube_shape[-1]):
            is_fitted = False
            # loop over the range of adaptive binning vakues
            for aper in range(minaper, maxaper+1):
                if is_fitted:
                    break
                if True:
                    # deal with the edges of the cube
                    sz = cube_shape
                    xlow = x - aper
                    xup = x + aper + 1
                    ylow = y - aper
                    yup = y + aper + 1
                    if xlow <= 0: xlow = 0
                    if xup > sz[2]: xup = sz[2]
                    if ylow <= 0: ylow = 0
                    if yup >= sz[1]: yup = sz[1]

                    # vector to hold the spectra
                    spec = np.zeros(sz[0])
                    # loop over x/y and integrate the cube to make a 1D spectrum
                    # npix = 0
                    if xlow == xup:
                        xidx_list = [xlow]
                    else:
                        xidx_list = list(range(xlow, xup))
                    if ylow == yup:
                        yidx_list = [ylow]
                    else:
                        yidx_list = list(range(ylow, yup))
                    # print(xidx_list, yidx_list)
                    spec = np.sum(cube[:, ylow:yup, xlow:xup], axis=(1,2))
                    # print('selection shape:', weight_map[ylow:yup, xlow:xup].shape)
                    # for m in range(xlow,xup+1):
                        # for n in range(ylow,yup+1):
                    # for n in yidx_list:
                        # for m in xidx_list:
                            # tmp = cube[:,n,m]
                            # tmp = tmp-np.median(tmp)
                            # spec = spec + tmp
                            # npix += 1
                    #print('npix', npix)
                    # spec = spec #/ npix #((1+yup-ylow) * (1+xup-xlow))

                    # spec = cube[:, y, x]
                    # only do a fit if there are values in the array
                    if np.nansum(spec) != 0:
                        # measure the std
                        cont_window = (velocity<=vel_low) | (velocity>=vel_up)
                        std = np.std(spec[cont_window])                            
                        med = np.median(spec[cont_window])
                        # get chi^2 of straight line fit
                        chi2_sline = np.sum((spec-med)**2 / std**2)
                        # do a Gaussian profile fit of the line
                        px = fit_spec(velocity, spec, std, fit_cont=fit_cont, 
                                      sigma_guess=sigma_guess, mode=mode)
                        bestfit = gaussian_1d(px, velocity=velocity)
                        # calculate the chi^2 of the Gaussian profile fit
                        chi2_gauss = np.sum((spec-bestfit)**2 / std**2)
                        # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                        SNR = (chi2_sline - chi2_gauss)**0.5

                        # store the fit parameters with S/N>SNR_limit
                        if SNR >= SNR_limit:
                            is_fitted = True
                            if debug:
                                print(f'fit found at {(x,y)} with S/N={SNR}')
                            # print('npix=', npix)
                            npix = weight_map[ylow:yup, xlow:xup].size
                            weight_fit = 1/npix
                            weight_map[ylow:yup, xlow:xup] += weight_fit
                            # print(weight_map[ylow:yup, xlow:xup])
                            # fitmaps[0, ylow:yup, xlow:xup] += px[0] * weight_fit
                            fitmaps[0, y, x] = px[0]/npix
                            fitmaps[1:3, y, x] = px[1:3] # vcenter, vsigma
                            if fit_cont:
                                fitmaps[3, y, x] = px[3]
                            fitmaps[4, y, x] = SNR
                            fitcube[:,y,x] = bestfit

                            # plot data and fit if S/N is above threshold
                            if plot:
                                ax = fig.add_subplot(111)
                                ax.step(vel, spec, color='black', where='mid', label='data')
                                ax.plot(vel, bestfit, color='red', alpha=0.8)
                                plt.show(block=False)
                                plt.pause(0.02)
                                plt.clf()
                        elif SNR<SNR_limit:
                            if debug:
                                print(f'no fit found at {(x,y)} with S/N={SNR}')
    # fitmaps[0] = fitmaps[0]/weight_map
    if savefile is None:
        return fitcube, fitmaps
    else:
        # keep files into fits file
        pass

def spiral_index(X, Y):
    idx_list = [] #np.zeros((X*Y,2))
    half_X = int(np.ceil(0.5*X))
    half_Y = int(np.ceil(0.5*Y))
    x = y = 0
    dx = 0
    dy = -1
    for i in range(max(X, Y)**2):
        if (-half_X < x <= half_X) and (-half_Y < y <= half_Y):
            idx_list.append([y, x])
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x+dx, y+dy
    return np.array(idx_list) + np.array([half_X-1, half_Y-1])

def voronori_cubefit(cube, velocity=None, mode='minimize', SNR_limit=3,
                     minaper=1, maxaper=5, fit_cont=False,
                     vel_low=-300, vel_up=300, sigma_guess=30,):
    nchan, ny, nx = cube.shape
    # define the cube to save fit results [amp, vcenter, vsigma, cont, snr, idx]
    fitcube = np.zeros_like(cube)
    fitmaps = np.full((5, ny, nx), fill_value=np.nan) 
    idxmap = np.zeros((ny, nx), dtype=int)
    spiral_idx = spiral_index(ny, nx)
    # print(spiral_idx[:5])
    # a container of the previous bestfit, useful for the following fit
    last_fit = None
    for i in range(len(spiral_idx)):
        is_fitted = False
        yi,xi = spiral_idx[i]
        # print('i=', i, [yi, xi])
        if idxmap[yi,xi] > 0:
            continue
        if i == 0:
            dy, dx = 0, 0
        else:
            dy, dx = spiral_idx[i] - spiral_idx[i-1]
        # print('spiral_idx:', spiral_idx[i], spiral_idx[i-1])
        if dx == 0:
            if xi < 0.5*nx: dx = -1
            else: dx = 1
        if dy == 0:
            if yi < 0.5*ny: dy = -1
            else: dy = 1
        # print('dy, dx', dy, dx)
        # dy, dx = [1 if d > -1 else -1 for d in (spiral_idx[i] - spiral_idx[i-1])]
        # print('dy, dx', dy, dx)
        for aper in range(minaper, maxaper+1):
            # print('aper=', aper)
            if is_fitted:
                # idxmap[yi:yi+aper*dy,xi:xi+aper*dx] = i+1
                break
            if dy > 0: ylow, yup = yi, yi+aper*dy
            else: ylow, yup = 1+yi+aper*dy, yi+1
            if dx > 0: xlow, xup = xi, xi+aper*dx
            else: xlow, xup = 1+xi+aper*dx, xi+1
            # if dy > 0: ylow, yup = yi+(1-aper)*dy, yi+aper*dy
            # else: ylow, yup = 1+yi+aper*dy, yi+1+(1-aper)*dy
            # if dx > 0: xlow, xup = xi+(1-aper)*dx, xi+aper*dx
            # else: xlow, xup = 1+xi+aper*dx, xi+1+(1-aper)*dx
            # if dy > 0: ys_ = np.s_[yi:yi+aper*dy]
            # else: ys_ = np.s_[1+yi+aper*dy:yi+1]
            # if dx > 0: xs_ = np.s_[xi:xi+aper*dx]
            # else: xs_ = np.s_[1+xi+aper*dx:xi+1]
            # check the boundary and already used pixels
            if ylow <= 0: ylow = 0
            if yup > ny: yup = ny
            if xlow <= 0: xlow = 0
            if xup > nx: xup = nx
            bin_select = ~(idxmap[ylow:yup,xlow:xup] > 0)
            # print(xlow, xup, ylow, yup)
            # print(bin_select)
            bin_select_cube = np.repeat(bin_select[None,:,:], nchan, axis=0)
            spec = np.sum(cube[:, ylow:yup, xlow:xup][bin_select_cube].reshape(
                    nchan, np.sum(bin_select)), axis=1)
            if np.nansum(spec) != 0:
                cont_window = (velocity<=vel_low) | (velocity>=vel_up)
                std = np.std(spec[cont_window])                            
                med = np.median(spec[cont_window])
                # get chi^2 of straight line fit
                chi2_sline = np.sum((spec-med)**2 / std**2)
                px = fit_spec(velocity, spec, std, p0=last_fit, fit_cont=fit_cont, 
                              sigma_guess=sigma_guess, mode=mode)
                bestfit = gaussian_1d(px, velocity=velocity)
                # calculate the chi^2 of the Gaussian profile fit
                chi2_gauss = np.sum((spec-bestfit)**2 / std**2)
                # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                SNR = (chi2_sline - chi2_gauss)**0.5
                # print('SNR=', SNR)
                # print(xlow, xup, ylow, yup)
                # print(bin_select)
                if SNR >= SNR_limit:
                    is_fitted = True
                    last_fit = None#px
                    idxmap[ylow:yup, xlow:xup][bin_select] = i+1
                    # fitmaps[0:3, yi, xi] = px[:3]/np.array([aper**2,1,1]) # amp, vcenter, vsigma
                    fitmaps[0, ylow:yup, xlow:xup][bin_select] = px[0]/np.sum(bin_select)
                    fitmaps[1, ylow:yup, xlow:xup][bin_select] = px[1]
                    fitmaps[2, ylow:yup, xlow:xup][bin_select] = px[2]
                    if fit_cont:
                        fitmaps[3, yi, xi] = px[3]
                    fitmaps[4, yi, xi] = SNR
                    fitcube[:,yi,xi] = bestfit
    print(idxmap)
    return fitcube, fitmaps

def voronori_cubefit2(cube, velocity=None, mode='minimize', SNR_limit=3,
                     minaper=0, maxaper=5, fit_cont=False,
                     vel_low=-300, vel_up=300, sigma_guess=30,):
    nchan, ny, nx = cube.shape
    cube_shape = cube.shape
    # construct the priority map, first fit the higher priority pixels
    max_value = np.max(cube)
    m0_map = np.sum(cube**2, axis=0) 
    m0_map = m0_map / np.max(m0_map)
    # get the distance to the center of the signal
    x_idx, y_idx = np.where(m0_map > np.percentile(m0_map, 95))
    yx_center = np.mean([y_idx, x_idx], axis=1) + np.array([0.5,0.5])
    yidx, xidx = np.mgrid[0:ny,0:nx]
    dist_map = 1/((yidx-yx_center[0])**2 + (xidx-yx_center[1])**2 + 0.01)
    # print(dist_map)
    cost_map = 1 / (m0_map * dist_map)
    
    if False:
        fig, ax = plt.subplots(1,3)
        im = ax[0].imshow(m0_map, origin='lower')
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(dist_map, origin='lower')
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(cost_map, origin='lower')
        plt.colorbar(im, ax=ax[2])
        plt.show()
        return
    
    # define the cube to save fit results [amp, vcenter, vsigma, cont, snr, idx]
    fitcube = np.zeros_like(cube)
    fitmaps = np.full((5, ny, nx), fill_value=np.nan) 
    idxmap = np.zeros((ny, nx), dtype=int)
    map_size = ny*nx
    sy, sx = np.unravel_index(np.argsort(cost_map, axis=None), (ny, nx))
    # print(spiral_idx[:5])
    # a container of the previous bestfit, useful for the following fit
    last_fit = None

    for i in range(map_size):
        is_fitted = False
        yi,xi = sy[i], sx[i]
        # print('i=', i, [yi, xi])
        if idxmap[yi,xi] > 0:
            continue

        for aper in range(minaper, maxaper+1):
            # print('aper=', aper)
            if is_fitted:
                # idxmap[yi:yi+aper*dy,xi:xi+aper*dx] = i+1
                break

            # deal with the edges of the cube
            sz = cube_shape
            xlow = xi - aper
            xup = xi + aper + 1
            ylow = yi - aper
            yup = yi + aper + 1
            if xlow <= 0: xlow = 0
            if xup > sz[2]: xup = sz[2]
            if ylow <= 0: ylow = 0
            if yup >= sz[1]: yup = sz[1]

            bin_select = ~(idxmap[ylow:yup,xlow:xup] > 0)
            bin_select_cube = np.repeat(bin_select[None,:,:], nchan, axis=0)
            spec = np.sum(cube[:, ylow:yup, xlow:xup][bin_select_cube].reshape(
                    nchan, np.sum(bin_select)), axis=1)
 
            if np.nansum(spec) != 0:
                cont_window = (velocity<=vel_low) | (velocity>=vel_up)
                std = np.std(spec[cont_window])                            
                med = np.median(spec[cont_window])
                # get chi^2 of straight line fit
                chi2_sline = np.sum((spec-med)**2 / std**2)
                px = fit_spec(velocity, spec, std, p0=last_fit, fit_cont=fit_cont, 
                              sigma_guess=sigma_guess, mode=mode)
                bestfit = gaussian_1d(px, velocity=velocity)
                # calculate the chi^2 of the Gaussian profile fit
                chi2_gauss = np.sum((spec-bestfit)**2 / std**2)
                # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                SNR = (chi2_sline - chi2_gauss)**0.5
                # print('SNR=', SNR)
                # print(xlow, xup, ylow, yup)
                # print(bin_select)
                if SNR >= SNR_limit:
                    is_fitted = True
                    last_fit = None#px
                    idxmap[ylow:yup, xlow:xup][bin_select] = i+1
                    # fitmaps[0:3, yi, xi] = px[:3]/np.array([aper**2,1,1]) # amp, vcenter, vsigma
                    fitmaps[0, ylow:yup, xlow:xup][bin_select] = px[0]/np.sum(bin_select)
                    fitmaps[1, ylow:yup, xlow:xup][bin_select] = px[1]
                    fitmaps[2, ylow:yup, xlow:xup][bin_select] = px[2]
                    if fit_cont:
                        fitmaps[3, yi, xi] = px[3]
                    fitmaps[4, yi, xi] = SNR
                    fitcube[:,yi,xi] = bestfit
    print(idxmap)
    return fitcube, fitmaps


def fit_kinms(cubefile, reffreq, smooth_width=None):
    cube_freq, cube_data = read_alma_simple(cubefile)
    cube_freq = cube_freq / 1e9 # change to GHz
    velocity = ((reffreq-cube_freq)/reffreq*const_c)

    fit_cube, fit_maps = pixel_fit_cube(cube_data, velocity, 
                                        SNR_limit=3, smooth_width=smooth_width, 
                                        fit_cont=False, plot=False, vel_low=-250, 
                                        vel_up=250,
                                        sigma_guess=20, )
    plot_fitcube(fit_maps, vel_min=-150, vel_max=150, flux_max=10*np.std(cube_data))


def plot_fitcube(fitmaps, flux_max=1, vel_min=-300, vel_max=300, sigma_max=100, ax=None):
    if ax is None:
        show_image = True
        fig, ax = plt.subplots(1,3, figsize=(12, 3))
    else:
        show_image = False
    ax1 = ax[0]
    ax1.set_title("Intensity")
    im = ax1.imshow(fitmaps[0], origin='lower', vmax=flux_max, vmin=-0.1*flux_max)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((0, 0))

    ax2 = ax[1]
    ax2.set_title("Velocity")
    im = ax2.imshow(fitmaps[1], vmin=vel_min, vmax=vel_max, origin='lower', 
                    cmap='RdBu_r')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((0, 500))
    
    ax3 = ax[2]
    ax3.set_title("Velocity dispersion")
    im = ax3.imshow(fitmaps[2], origin='lower', vmin=0, vmax=sigma_max)
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.formatter.set_powerlimits((0, 100))
    if show_image:
        plt.show()



###############################################
# test functions
def test_fit_spectrum():
    # create a random spectrum
    gaussian_truth = [2, -150, 200]
    vel = np.linspace(-2000, 2000, 500)
    line = gaussian_1d(gaussian_truth, velocity=vel)
    # add std=1 normal distribution noise
    spec = line + np.random.randn(500) 

    bestfit1 = fit_spec(vel, spec, std=1, fit_cont=False, mode='minimize')
    bestfit1_model = gaussian_1d(bestfit1, velocity=vel)
    print('truth:', gaussian_truth)
    print('best fit2:', bestfit1)
    
    bestfit2 = fit_spec(vel, spec, std=1, fit_cont=False, mode='mcmc')
    bestfit2_model = gaussian_1d(bestfit2, velocity=vel)
    print('best fit2:', bestfit2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(vel, spec, label='data')
    ax.step(vel, bestfit1_model, label='minimize')
    ax.step(vel, bestfit2_model, label='mcmc')
    ax.legend()
    plt.show()

def test_fit_cube():
    nx, ny, nchan= 10, 10, 200
    vel_dispersion = 100
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [10,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-1000,1000,nchan)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d([flux_i, vel_i, vel_dispersion], vel_chan) \
                    + np.random.randn(nchan)

    dchan = vel_chan[1]-vel_chan[0]
    M0 = np.sum(cube_data*dchan, axis=0)
    M1 = np.sum(cube_data*dchan*vel_chan[:,None,None], axis=0)/M0
    
    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    ax[0,0].set_title('Model Flux')
    im = ax[0,0].imshow(flux_map, origin='lower')
    plt.colorbar(im, ax=ax[0,0], fraction=0.046, pad=0.04)
    ax[0,1].set_title('Moment 0')
    im = ax[0,1].imshow(M0, origin='lower')
    plt.colorbar(im, ax=ax[0,1], fraction=0.046, pad=0.04)
    ax[0,2].set_title('Kinematics')
    im = ax[0,2].imshow(vel_map, origin='lower', cmap='RdBu_r', vmin=-400, vmax=400)
    plt.colorbar(im, ax=ax[0,2], fraction=0.046, pad=0.04)
    # appy the fitting
    fit_cube, fit_maps = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3, mode='minimize')
    fit_cube2, fit_maps2 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3, mode='mcmc')
    plot_fitcube(fit_maps, vel_min=-400, vel_max=400, flux_max=10, sigma_max=300, ax=ax[1])
    plot_fitcube(fit_maps2, vel_min=-400, vel_max=400, flux_max=10, sigma_max=300, ax=ax[2])
    plt.show()

def test_fit_cube2(mode='minimize'):
    nx, ny, nchan= 15, 15, 500
    vel_dispersion = 200
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [10,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-2000,2000,nchan)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d([flux_i, vel_i, vel_dispersion], vel_chan) \
                    + np.random.randn(nchan)

    dchan = vel_chan[1]-vel_chan[0]
    M0 = np.sum(cube_data*dchan, axis=0)
    M1 = np.sum(cube_data*dchan*vel_chan[:,None,None], axis=0)/M0
    
    fig, ax = plt.subplots(4, 3, figsize=(10, 10))
    ax[0,0].set_title('Model Flux')
    im = ax[0,0].imshow(flux_map, origin='lower')
    plt.colorbar(im, ax=ax[0,0], fraction=0.046, pad=0.04)
    # ax[0,1].set_title('Moment 0')
    # im = ax[0,1].imshow(M0, origin='lower')
    # plt.colorbar(im, ax=ax[0,1], fraction=0.046, pad=0.04)
    ax[0,1].set_title('Kinematics')
    im = ax[0,1].imshow(vel_map, origin='lower', cmap='RdBu_r', vmin=-400, vmax=400)
    plt.colorbar(im, ax=ax[0,1], fraction=0.046, pad=0.04)
    ax[0,2].set_title('Dispersion')
    im = ax[0,2].imshow(np.full_like(vel_map, fill_value=vel_dispersion), origin='lower')
    plt.colorbar(im, ax=ax[0,2], fraction=0.046, pad=0.04)
    # appy the fitting
    fit_cube1, fit_maps1 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3,minaper=0,maxaper=0,mode=mode,)
    fit_cube2, fit_maps2 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3,minaper=0,maxaper=4,mode=mode,)
    fit_cube3, fit_maps3 = voronori_cubefit2(cube_data, vel_chan, SNR_limit=3,maxaper=4,mode=mode,)
    # fit_cube2, fit_maps2 = voronori_cubefit(cube_data, vel_chan, SNR_limit=3,maxaper=6,mode=mode,)
    # fit_cube2, fit_maps2 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3, mode='mcmc')
    # fit_cube2, fit_maps2 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3, mode='mcmc')
    plot_fitcube(fit_maps1, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[1])
    plot_fitcube(fit_maps2, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[2])
    plot_fitcube(fit_maps3, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[3])
    plt.show()


def test_fit_cube3():
    nx, ny, nchan= 10, 10, 500
    vel_dispersion = 200
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [10,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-2000,2000,nchan)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d([flux_i, vel_i, vel_dispersion], vel_chan) \
                    + np.random.randn(nchan)
    voronori_cubefit2(cube_data)

