#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2024-10-01: first release, v0.1
    2025-05-06: add multiprocessing support for fitters, v0.2
"""

__version__ = '0.2.2'

import os, sys, glob, re, time
import warnings
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.io import fits
import astropy.table as table
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import ndimage, optimize, stats, interpolate, signal
from photutils.aperture import EllipticalAperture
from photutils.aperture import EllipticalAperture, RectangularAperture
from multiprocessing import Pool, Process, Queue

from astropy.modeling import fitting, models


#####################################################################################
# Kinematic models
#####################################################################################

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
#####################################################################################

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
#####################################################################################

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
#####################################################################################

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
    x = np.arange(size) - (size-1)/2.0
    sigma_x = bmin/FWHM2sigma
    sigma_y = bmaj/FWHM2sigma
    gkernx = np.exp(-0.5*(x/sigma_x)**2)
    gkerny = np.exp(-0.5*(x/sigma_y)**2)
    # gkernx = signal.gaussian(size, std=bmin/FWHM2sigma).reshape(size, 1)
    # gkerny = signal.gaussian(size, std=bmaj/FWHM2sigma).reshape(size, 1)

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
    # return (spec-fit)**2/std**2

def fit_spec(vel, spec, std=1, p0=None, mode='minimize', 
             plot=False, ax=None,
             velocity_bounds=[-500,500], amplitude_bounds=None, 
             sigma_bounds=[10, 1000], 
             fit_cont=False, cont_bounds=None, 
             steps=5000, discard=None):
    """a basic spectrum fitter, and it is also the base of other fancier fitters

    Args:
        vel: the velocity
        spec: the spectral data
        std: the standard diviation of the spec
        p0: the intial guess for the gaussian fitting
        mode: supports "minimize" and "mcmc".
        plot: to visualise the fitting
        ax: the plotting axes
        amplitude_bounds: the amplitude boundaries
        velocity_bounds: the velocity boundaries
        sigma_bounds: the velocity dispersion boundaries
        fit_cont: set to true to fit the continuum along with the line
        cont_bounds: the magnitude boundaries of continuum
        steps: the sampling steps for mcmc fitting
        discard: the number discarding steps when reading the mcmc samplings,
                 default: 0.1*steps
    """
    #
    # initialize some missing constraints
    #
    if amplitude_bounds is None:
        amplitude_bounds = [0, #np.min([0.8*amp_min,1.2*amp_min]
                                   1.2*np.abs(np.max(spec))]
    if fit_cont:
        if cont_bounds is None:
            cont_bounds = [np.min(spec), np.max(spec)]
    if sigma_bounds is not None:
        sigma_bounds = [sigma_bounds[0], sigma_bounds[1]]
    
    #
    # guess the initial parameters
    #
    if p0 is not None:
        p0 = np.array(p0)
        I0, vel0, sig0 = p0[:3]
    else:
        I0 = vel0 = sig0 = None
    if I0 is None:
        I0 = np.max(spec)
    if vel0 is None:
        vel0 = 0 #np.median(vel[spec_selection])
    if sig0 is None:
        if sigma_bounds is not None:
            sig0 = 1.2*np.max([np.median(np.diff(vel)), 1.1*sigma_bounds[0]])
        else:
            sig0 = 2*np.median(np.diff(vel))
    if fit_cont:
        if (p0 is not None) and (len(p0)>3):
            cont0 = p0[3]
        else:
            spec_selection = (vel>velocity_bounds[0]) & (vel<velocity_bounds[1])
            cont0 = np.median(spec[~spec_selection])

    if mode == 'minimize':
        bounds = [amplitude_bounds, velocity_bounds, sigma_bounds]
        if fit_cont:
            initial_guess = [I0, vel0, sig0, cont0]
            bounds.append(cont_bounds)
        else:
            initial_guess = [I0, vel0, sig0]
        # make profile of initial guess.  this is only used for information/debugging purposes.
        guess_spec = gaussian_1d(initial_guess, velocity=vel)
        # do the fit
        if False:# np.all(bounds) is None:
            fit_result = optimize.least_squares(calc_gaussian1d_chi2, initial_guess, 
                                       args=(vel, spec, std), 
                                       bounds=(-np.inf,np.inf), method='lm')
        if True:
            fit_result = optimize.minimize(calc_gaussian1d_chi2, initial_guess, 
                                       args=(vel, spec, std), 
                                       bounds=bounds,)
        # make profile of the best fit 
        bestfit_spec = gaussian_1d(fit_result.x, velocity=vel)
        bestfit_params = fit_result.x
    if mode == 'mcmc':
        try:
            from model_utils import Gaussian1D, Models, Parameter, EmceeFitter, Grid1D
        except:
            raise ImportError("Install/copy model_utils to the PYTHONPATH!")
        if fit_cont:
            gaussian_model = Gaussian1D(
                    amplitude=Parameter(I0, limits=amplitude_bounds),
                    mean=Parameter(vel0, limits=velocity_bounds),
                    sigma=Parameter(sig0, limits=sigma_bounds),
                    cont=Parameter(cont0, limits=cont_bounds),
                    name='gaussian1d')
        else:
            gaussian_model = Gaussian1D(
                    amplitude=Parameter(I0, limits=amplitude_bounds),
                    mean=Parameter(vel0, limits=velocity_bounds),
                    sigma=Parameter(sig0, limits=sigma_bounds),
                    cont=0, name='gaussian1d')
        models = Models([gaussian_model])
        grid1d = Grid1D()
        grid1d.x = vel
        parameters = models.get_parameters()
        guess_spec = models.create_model(grid1d)
        data = {'var':vel, 'data':spec, 'data_err':std}
        fitter = EmceeFitter(models, grid1d, data)
        # fitter.run(progress=False, steps=steps)
        # if discard is None:
            # discard = int(0.1*steps)
        # samples = fitter.sampler.get_chain(discard=discard, flat=True,)
        fitter.auto_run(progress=False, max_steps=steps)
        samples = fitter.samples(discard=discard, flat=True,)
        bestfit_values = np.percentile(samples, 50, axis=0)
        bestfit_dict = {}
        for key, value in zip(parameters.keys(), bestfit_values):
            bestfit_dict[key] = value
        models.update_parameters(bestfit_dict)
        bestfit_spec = models.create_model(grid1d)
        if fit_cont:
            bestfit_params = [bestfit_dict['gaussian1d.amplitude'], 
                              bestfit_dict['gaussian1d.mean'], 
                              bestfit_dict['gaussian1d.sigma'], 
                              bestfit_dict['gaussian1d.cont']]
        else:
            bestfit_params = [bestfit_dict['gaussian1d.amplitude'], 
                              bestfit_dict['gaussian1d.mean'], 
                              bestfit_dict['gaussian1d.sigma']]
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].step(vel, spec, color='black', where='mid', label='data')
        ax[0].plot(vel, bestfit_spec, linestyle='dashed', color='blue', label='bestfit')
        ax[0].legend()
        # ax.plot(vel, spec*0+cont0+std, linestyle='dashed', color='blue', alpha=0.5)
        # ax.plot(vel, spec*0+cont0-std, linestyle='dashed', color='blue', alpha=0.5)
        ax[1].step(vel, spec, color='black', where='mid', label='data')
        ax[1].step(vel, guess_spec, color='green', where='mid', label='guess')
        ax[1].plot(vel, bestfit_spec, linestyle='dashed', color='blue', label='bestfit')
        ax[1].legend()
        plt.show()
    return bestfit_params, bestfit_spec

def fit_spec_norm(vel, spec, std=1, p0=None, mode='minimize', 
                  I0=None, vel0=None, sigma0=None, plot=False, ax=None,
                  velocity_bounds=[-500,500], amplitude_bounds=None, 
                  sigma_bounds=[10,1000], 
                  fit_cont=False, cont_bounds=None, 
                  steps=5000, discard=None):
    """a basic spectrum fitter, similar to fit_spec, but it normalize the
    spec and vel
        
    Args:
        vel: the velocity
        spec: the spectral data
        std: the standard diviation of the spec
        p0: the intial guess for the gaussian fitting
        mode: supports "minimize" and "mcmc".
        plot: to visualise the fitting
        ax: the plotting axes
        amplitude_bounds: the amplitude boundaries
        velocity_bounds: the velocity boundaries
        sigma_bounds: the velocity dispersion boundaries
        fit_cont: set to true to fit the continuum along with the line
        cont_bounds: the magnitude boundaries of continuum
        steps: the sampling steps for mcmc fitting
        discard: the number discarding steps when reading the mcmc samplings,
                 default: 0.1*steps
    """
    
    #
    # normalize the spectrum (amplitude+velocity) to aviod numerical issues
    #
    amp_scale = np.nanmax(spec)
    # print('amp_scale:', amp_scale)
    # print(spec)
    if velocity_bounds is not None:
        vel_scale = abs(np.diff(velocity_bounds)[0])
    else:
        vel_scale = abs(vel[-1]-vel[0])
    spec_norm = spec/amp_scale
    std_norm = std/amp_scale
    if amplitude_bounds is None:
        amp_min = np.min(spec_norm)
        amplitude_bounds_scaled = [0, #np.min([0.8*amp_min,1.2*amp_min]
                                   1.2*np.abs(np.max(spec_norm))]
    else:
        amplitude_bounds_scaled = [amplitude_bounds[0]/amp_scale, 
                                   amplitude_bounds[1]/amp_scale]
    if fit_cont:
        if cont_bounds is None:
            cont_bounds_scaled = [np.min(spec_norm), np.max(spec_norm)]
        else:
            cont_bounds_scaled = [cont_bounds[0]/amp_scale, cont_bounds[1]/amp_scale]
    vel_norm = vel/vel_scale
    velocity_bounds_scaled = [velocity_bounds[0]/vel_scale, 
                              velocity_bounds[1]/vel_scale]
    if sigma_bounds is not None:
        sigma_bounds_scaled = [sigma_bounds[0]/vel_scale, sigma_bounds[1]/vel_scale]
    
    #
    # guess the initial parameters
    #
    if p0 is not None:
        I0, vel0, sigma0 = p0[:3]
    else:
        I0 = vel0 = sig0 = None
    if I0 is not None:
        I0 = I0 / amp_scale 
    if vel0 is not None:
        vel0 = vel0/vel_scale 
    if vel0 is not None:
        sig0 = vel0/vel_scale

    if I0 is None:
        I0 = np.max(spec_norm)
    if vel0 is None:
        vel0 = 0 #np.median(vel[spec_selection])
    if sig0 is None:
        if sigma_bounds is not None:
            sig0 = 1.2*np.max([np.median(np.diff(vel_norm)), sigma_bounds_scaled[0]])
        else:
            sig0 = 2*np.median(np.diff(vel_norm))
    sigma_bounds_scaled = [sigma_bounds[0]/vel_scale, sigma_bounds[1]/vel_scale]
    if fit_cont:
        if (p0 is not None) and (len(p0)>3):
            cont0 = p0[3]
        else:
            spec_selection = (vel>velocity_bounds[0]) & (vel<velocity_bounds[1])
            cont0 = np.median(spec_norm[~spec_selection])

    if mode == 'minimize':
        bounds = [amplitude_bounds_scaled, velocity_bounds_scaled, sigma_bounds_scaled]
        if fit_cont:
            initial_guess = [I0, vel0, sig0, cont0]
            bounds.append(cont_bounds_scaled)
        else:
            initial_guess = [I0, vel0, sig0]
        # make profile of initial guess.  this is only used for information/debugging purposes.
        guess_spec_norm = gaussian_1d(initial_guess, velocity=vel_norm)
        # do the fit
        fit_result = optimize.minimize(calc_gaussian1d_chi2, initial_guess, 
                                       args=(vel_norm, spec_norm, std_norm), 
                                       bounds=bounds)
        # make profile of the best fit 
        bestfit_spec_norm = gaussian_1d(fit_result.x, velocity=vel_norm)
        bestfit_params = fit_result.x
    if mode == 'mcmc':
        try:
            from model_utils import Gaussian1D, Models, Parameter, EmceeFitter, Grid1D
        except:
            raise ImportError("Install/copy model_utils to the PYTHONPATH!")
        if fit_cont:
            gaussian_model = Gaussian1D(
                    amplitude=Parameter(I0, limits=amplitude_bounds_scaled),
                    mean=Parameter(vel0, limits=velocity_bounds_scaled),
                    sigma=Parameter(sig0, limits=sigma_bounds_scaled),
                    cont=Parameter(cont0, limits=cont_bounds_scaled),
                    name='gaussian1d')
        else:
            gaussian_model = Gaussian1D(
                    amplitude=Parameter(I0, limits=amplitude_bounds_scaled),
                    mean=Parameter(vel0, limits=velocity_bounds_scaled),
                    sigma=Parameter(sig0, limits=sigma_bounds_scaled),
                    cont=0, name='gaussian1d')
        models = Models([gaussian_model])
        grid1d_norm = Grid1D()
        grid1d_norm.x = vel_norm
        # len(vel_norm)+2, pixelsize=vel_norm[1]-vel_norm[0])
        parameters = models.get_parameters()
        guess_spec_norm = models.create_model(grid1d_norm)
        data_norm = {'var':vel_norm, 'data':spec_norm, 'data_err':std_norm}
        fitter = EmceeFitter(models, grid1d_norm, data_norm)
        # fitter.run(progress=False, steps=steps)
        # if discard is None:
            # discard = int(0.1*steps)
        # samples = fitter.sampler.get_chain(discard=discard, flat=True,)
        fitter.auto_run(progress=False, max_steps=steps)
        samples = fitter.samples(discard=discard, flat=True,)
        bestfit_values = np.percentile(samples, 50, axis=0)
        bestfit_dict = {}
        for key, value in zip(parameters.keys(), bestfit_values):
            bestfit_dict[key] = value
        models.update_parameters(bestfit_dict)
        bestfit_spec_norm = models.create_model(grid1d_norm)
        if fit_cont:
            bestfit_params = [bestfit_dict['gaussian1d.amplitude'], 
                              bestfit_dict['gaussian1d.mean'], 
                              bestfit_dict['gaussian1d.sigma'], 
                              bestfit_dict['gaussian1d.cont']]
        else:
            bestfit_params = [bestfit_dict['gaussian1d.amplitude'], 
                              bestfit_dict['gaussian1d.mean'], 
                              bestfit_dict['gaussian1d.sigma']]
    if fit_cont:
        bestfit_params = bestfit_params*np.array(
                [amp_scale, vel_scale, vel_scale, amp_scale])
    else:
        bestfit_params = bestfit_params*np.array(
                [amp_scale, vel_scale, vel_scale])
    bestfit_spec = bestfit_spec_norm * amp_scale
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].step(vel, spec, color='black', where='mid', label='data')
        ax[0].plot(vel, bestfit_spec, linestyle='dashed', color='blue', label='bestfit')
        ax[0].legend()
        # ax.plot(vel, spec*0+cont0+std, linestyle='dashed', color='blue', alpha=0.5)
        # ax.plot(vel, spec*0+cont0-std, linestyle='dashed', color='blue', alpha=0.5)
        ax[1].step(vel_norm, spec_norm, color='black', where='mid', label='data')
        ax[1].step(vel_norm, guess_spec_norm, color='green', where='mid', label='guess')
        ax[1].plot(vel_norm, bestfit_spec_norm, linestyle='dashed', color='blue', label='bestfit')
        ax[1].legend()
        plt.show()
    return bestfit_params, bestfit_spec

def fit_spec_astropy(specchan, specdata, guess=None, bounds=None, 
              snr_limit=None, plot=False, ax=None, debug=False):
    """fit a single gaussian, similar to spec_autofit, but based on astropy.modelling

    Args:
        specchan: channel data, can be wavelength, frequency, velocity
        specdata: data
        guess: the initial guess for [amplitude, mean, sigma]
        bounds: the boundary for [(amp_min, amp_max), (mean_min, mean_max), 
                                  (sigma_min, sigma_max)]
    """
    specchan = np.array(specchan)
    specdata = np.array(specdata)
    if guess is not None:
        amp0, mean0, sigma0 = guess
    else:
        amp0 = mean0 = sigma0 = None
    spec_selection = specdata > np.percentile(specdata, 85)
    if amp0 is None:
        amp0 = 1.5*np.mean(specdata[spec_selection])
    if mean0 is None:
        mean0 = np.median(specchan[spec_selection])
    if sigma0 is None:
        sigma0 = np.abs(np.median(np.diff(specchan)))
    if bounds is None:
        # fit_p = fitting.LevMarLSQFitter()
        fit_p = fitting.LMLSQFitter()
        fit_bounds = {}
    else:
        fit_p = fitting.TRFLSQFitter()
        fit_bounds = {'amplitude':bounds[0], 'mean':bounds[1], 'stddev':bounds[2]}
    if debug:
        print(f"Initial guess: {amp0, mean0, sigma0}")
        print(f"Bounds: {bounds}")
    p_init = models.Gaussian1D(amplitude=amp0, mean=mean0, 
                               stddev=sigma0,
                               bounds=fit_bounds)
    p = fit_p(p_init, specchan, specdata)
    specfit = p(specchan)
    bestfit = p.param_sets.flatten()

    if snr_limit is not None:
        std = np.std(specdata - specfit)
        median = np.median(specdata - specfit)
        chi2_sline = np.nansum((specdata-median)**2/std**2)
        chi2_fit = np.nansum((specdata-specfit)**2 / std**2)
        snr = (chi2_sline - chi2_fit)**0.5
        # print(chi2_sline, chi2_fit, snr, bestfit)
        if snr > snr_limit:
            return bestfit, specfit
        else:
            return [0,0,0], None
    else:
        return bestfit, specfit
 

def pv_diagram(datacube, velocity=None, z=None, pixel_center=None,
               vmin=-1000, vmax=1000,
               length=1, width=1, theta=0, debug=False, plot=True, pixel_size=1):
    """generate the PV diagram of the cube

    Args:
        datacube: the 3D data [velocity, y, x]
        pixel_center: the center of aperture
        lengthth: the length of the aperture, in x axis when theta=0
        width: the width of the aperture, in y axis when theta=0
        theta: the angle in radian, from positive x to positive y axis
    """
    # nchan, ny, nx = datacube.shape
    width = np.round(width).astype(int)
    if isinstance(datacube, str):
        if z is not None:
            velocity, datacube, header = read_eris_cube(datacube, z=z)
        else:
            wavelength, datacube, header = read_eris_cube(datacube)
    # extract data within the velocity range 
    if velocity is not None:
        vel_selection = (velocity > vmin) & (velocity < vmax)
        datacube = datacube[vel_selection]
        velocity = velocity[vel_selection]
    cubeshape = datacube.shape
    aper = RectangularAperture(pixel_center, length, width, theta=theta)
    s1,s2 = aper.to_mask().get_overlap_slices(cubeshape[1:])
    # cutout the image
    try:
        sliced_cube = datacube[:,s1[0],s1[1]]
    except:
        sliced_cube = None
        sliced_pvmap = None

    if sliced_cube is not None:
        # rotate the cube to make the aperture with theta=0
        sliced_cube_rotated = ndimage.rotate(sliced_cube, aper.theta/np.pi*180, axes=[1,2], reshape=True, prefilter=False, order=0, cval=np.nan)
        # sum up the central plain (x axis) within the height
        nchan, nynew, nxnew = sliced_cube_rotated.shape
        # define the new aperture on the rotated sub-cube
        aper_rotated = RectangularAperture([0.5*(nxnew-1), 0.5*(nynew-1)], length, width, theta=0)

        hi_start = np.round(nynew/2.-width/2.).astype(int)
        width_slice = np.s_[hi_start:hi_start+width]
        sliced_pvmap = np.nansum(sliced_cube_rotated[:,width_slice,:], axis=1)
    if debug: # debug plot 
        fig = plt.figure(figsize=(12,5))
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        ax1.imshow(np.nansum(datacube,axis=0),origin='lower')
        aper.plot(ax=ax1)
        if sliced_cube is not None:
            ax2.imshow(np.nansum(sliced_cube_rotated, axis=0), origin='lower')
            aper_rotated.plot(ax=ax2)
            ax3.imshow(sliced_pvmap.T, origin='lower')
    if plot:
        fig = plt.figure(figsize=(12,5))
        # ax = fig.subplots(1,2)
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        vmin, vmax = -1*np.nanstd(sliced_cube), 4*np.nanmax(sliced_cube)
        ax1.imshow(np.sum(datacube, axis=0), origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
        aper.plot(ax=ax1, color='white', linewidth=4)
        if sliced_cube is not None:
            # show the pv-diagram
            # ax[1].imshow(sliced_pvmap, origin='lower', cmap='magma', extent=extent)
            positions = np.linspace(-0.5*length, 0.5*length, nxnew)
            if velocity is None:
                velocity = np.linspace(-1,1,nchan)
            vmesh, pmesh = np.meshgrid(velocity,positions)
            ax2.pcolormesh(vmesh, pmesh, sliced_pvmap.T, cmap='magma')
            gauss_kernel = Gaussian2DKernel(1)
            smoothed_pvmap = convolve(sliced_pvmap, gauss_kernel)
            ax3.pcolormesh(vmesh, pmesh, smoothed_pvmap.T, cmap='magma')
            ax2.set_ylabel('Position')
            ax3.set_ylabel('Position')
            ax3.set_xlabel('Velocity')
    return sliced_pvmap

def cube_viewer():
    """a simplified cube viewer
    """
    pass

#####################################################################################
# Fitter classes
#####################################################################################
try:
    from model_utils import Models, Parameter, Grid1D, EmceeFitter, MinimizeFitter
except:
    raise ImportError("Install/copy model_utils to the PYTHONPATH!")

class SpecFitter:
    def __init__(self, models, grid=None):
        self.models = models
        self.grid = grid
    def create_model(self, params_dict, grid=None):
        if grid is None:
            grid = self.grid
        self.models.update_parameters(params_dict)
        return self.models.create_model(grid)
    def fit(self, vel, spec, std, initial_guess=None, 
            mode='mcmc', discard=None, steps=2000):
        """ fit pixel by pixel

        Args:
            initial_guess: the initial guess for the fittable parameters, dictionary
        
        Note: different from the function version, the model is from user 
              input so we cannot normalize the data here, it is recommended 
              to normalize the data before calling this function
        """
        models = self.models 
        if self.grid is None:
            grid1d = Grid1D()
            grid1d.x = vel
            self.grid = grid1d
        else:
            grid1d = self.grid
        parameters = models.get_parameters()
        data = {'var':vel, 'data':spec, 'data_err':std}

        # get the initial guess
        # if initial_guess is None:
        
        if mode == 'minimize':
            fitter = MinimizeFitter(self.models, grid1d, data)
            fitter.run()
            bestfit_values = fitter.best_fit 

        if mode == 'mcmc':
            fitter = EmceeFitter(self.models, grid1d, data)
            fitter.run(progress=False, steps=steps)
            if discard is None:
                discard = int(0.1*steps)
            samples = fitter.sampler.get_chain(discard=discard, flat=True,)
            bestfit_values = np.percentile(samples, 50, axis=0)
        bestfit_dict = {}
        for key, value in zip(parameters.keys(), bestfit_values):
            bestfit_dict[key] = value
        return bestfit_dict

    def fit_alternative(self, vel, spec, std):
        """
        """
        pass

class CubeFitter:
    def __init__(self, cube, velocity=None, header=None):
        self.cube = cube
        self.velocity = velocity
        self.nchan, self.ny, self.nx = cube.shape

    def smooth(self, kernel=None):
        gauss_kernel = gkern(smooth_width) 
        self.cube = signal.convolve(self.cube, gauss_kernel[None,:,:], mode='same')

    def fit_cube(self, mode='minimize', SNR_limit=3,
                 vel_low=-300, vel_up=300, savefile=None, debug=None):
        fitcube, fitmaps = pixel_fit_cube(
                self.cube, velocity=self.velocity, mode=mode, SNR_limit=SNR_limit,
                vel_low=vel_low, vel_up=vel_up, 
                savefile=None, debug=debug)
        return fitcube, fitmaps

    def read_fits(self, fitsfile, z=None, rest_wave=None, header_ext='DATA'):
        velocity, cube, header = read_cube(cube, z=z, rest_wave=rest_wave, 
                                           header_ext=header_ext)
        self.velocity = velocity
        self.cube = cube
        self.header = header
        
class CubeMPFitter():
    def __init__(self, cube, velocity=None, mask=None,
                 header=None, mode='minimize', 
                 snr_limit=5, velocity_bounds=[-500,500], 
                 amplitude_bounds=None, sigma_bounds=[10,1000], 
                 steps=5000):
        # super().__init__(cube=cube, velocity=velocity, header=header)
        self.cube = cube
        self.velocity = velocity
        self.nchan, self.ny, self.nx = cube.shape
        if mask is None:
            self.mask = np.zeros((self.ny, self.nx)).astype(bool)
        self.mode = mode
        self.snr_limit = snr_limit
        self.velocity_bounds = velocity_bounds
        self.amplitude_bounds = amplitude_bounds
        self.sigma_bounds = sigma_bounds
        self.steps = steps
        self.input_queue = Queue()
        self.output_queue = Queue()
        self._processes = [] 
    def __call__(self, vel, spec):
        return spec_autofit(vel, spec, mode=self.mode, snr_limit=self.snr_limit,
                            amplitude_bounds=self.amplitude_bounds,
                            velocity_bounds=self.velocity_bounds,
                            sigma_bounds=self.sigma_bounds, steps=self.steps)
    def _cleanup(self):
        for p in self._processes:
            p.join()
            p.close()
    def worker(self):
        for i,j,vel,spec in iter(self.input_queue.get, 'STOP'):
            fit_value = self(vel, spec)
            # fit_value = spec_autofit(vel, spec)
            self.output_queue.put([i, j, fit_value])

    def fit(self, ncore=8):
        map_fit = np.zeros((3,self.ny, self.nx))
        # provide the queue for multiprocessing
        print("Queue-up")
        n_queue = 0
        for i in range(self.ny):
            for j in range(self.nx):
                if self.mask[i,j]:
                    continue
                self.input_queue.put([i,j, self.velocity, self.cube[:,i,j]])
                n_queue += 1
        # start the fit
        print("Start the fitting")
        processes = [Process(target=self.worker) for i in range(ncore)]
        for proc in processes:
            proc.start()
            # self._processes.append(proc)
        # read the fit
        print("Read the fitting")
        for i in range(n_queue):
            i, j, fit_value = self.output_queue.get()
            if fit_value is not None:
                map_fit[:,i,j] = fit_value
        # stop all the queue
        for i in range(ncore):
            self.input_queue.put('STOP')
        # for proc in processes:
            # proc.terminate()
            # proc.join()
            # proc.close()
        return map_fit

    def adaptive_fit(self, minaper=0, maxaper=4, ncore=8, debug=False):
        self.ncore = ncore
        map_is_fitted = np.zeros((self.ny, self.nx)).astype(bool)
        map_fit = np.full((3,self.ny, self.nx), fill_value=np.nan)
        processes = []
        for aper in range(minaper, maxaper):
            if aper == 0:
                aper_area = 1
            else:
                aper_area = (2*aper)**2
            if debug:
                print(f"Run workers with aper={aper}")
                print(f"Queue-up with aper={aper}")
            n_queue = 0
            for i in range(self.ny):
                for j in range(self.nx):
                    if self.mask[i,j]:
                        continue
                    if map_is_fitted[i,j]:
                        continue
                    # print("origin",self.cube[:,i,j][:10])
                    # print("summed",(self.cube[:,i-aper:i+aper,j-aper:j+aper]
                           # ).sum(axis=2).sum(axis=1)[:10])
                    if aper == 0:
                        self.input_queue.put([i,j, self.velocity, self.cube[:,i,j]])
                    else:
                        i_low, i_up = i-aper, i+aper
                        j_low, j_up = j-aper, j+aper
                        if i_low<0: i_low = 0
                        if j_low<0: j_low = 0
                        if i_up>self.ny-1: i_up = self.ny-1
                        if j_up>self.nx-1: j_up = self.nx-1
                        spec_coadded = np.zeros(self.nchan)
                        for i_aper in range(i_low, i_up):
                            for j_aper in range(j_low, j_up):
                                if self.mask[i_aper,j_aper]:
                                    continue
                                spec_coadded += self.cube[:,i_aper,j_aper]
                        self.input_queue.put([i,j, self.velocity, spec_coadded])
                    n_queue += 1
            if debug:
                print(f"Fit with aper={aper}, n_queue={n_queue}")
            if processes == []:
                processes = [Process(target=self.worker) for i in range(ncore)]
                for proc in processes:
                    proc.start()
            else:
                if debug:
                    print('re-using existing process')
            if debug:
                print(f"Read fitting with aper={aper}")
            for i in range(n_queue):
                i, j, fit_value = self.output_queue.get()
                if fit_value is not None:
                    map_fit[:,i,j] = fit_value / np.array([aper_area, 1,1])
                    map_is_fitted[i,j] = True
            # for proc in processes:
                # p.terminate()
                # proc.join()
                # proc.close()
        # stop all the queue
        for i in range(ncore):
            self.input_queue.put('STOP')
            self.output_queue.put('STOP')
        if debug:
            print("Done!")
        return map_fit 
 
    def voronoi_fit(self, vorbin, ncore=8):
        pass

class CubeMPFitter_test():
    """
    The goal was to use pool.map instead of Queue

    It is supposed to be a simpler version of CubeMPFitter, 
    but it is not easy to implement the adaptive_fit part, 
    where a pre-prepare of the data is required.
    tried with callback functionanity, but not work yet. 
    Now only leave the simple part here for future reference
    """
    def __init__(self, cube, velocity=None, header=None, mode='minimize', 
                 snr_limit=5, velocity_bounds=[-500,500], 
                 amplitude_bounds=None, sigma_bounds=[10,1000]):
        # super().__init__(cube=cube, velocity=velocity, header=header)
        self.cube = cube
        self.velocity = velocity
        self.nchan, self.ny, self.nx = cube.shape
        self.mode = mode
        self.snr_limit = snr_limit
        self.velocity_bounds = velocity_bounds
        self.amplitude_bounds = amplitude_bounds
        self.sigma_bounds = sigma_bounds
        self.map_fit = np.zeros((3,self.ny, self.nx))
    def evaluate(self, coord):
        i,j = coord
        spec = self.cube[:,i,j]
        fit_result = spec_autofit(self.velocity, spec, mode=self.mode, 
                                  snr_limit=self.snr_limit,
                                  amplitude_bounds=self.amplitude_bounds,
                                  velocity_bounds=self.velocity_bounds,
                                  sigma_bounds=self.sigma_bounds)
        return [i, j, fit_result]
    def fit(self, ncore=8):
        map_fit = np.zeros((3,self.ny, self.nx))
        xx, yy = np.meshgrid(range(self.ny), range(self.nx))
        coords = list(zip(xx.flatten(), yy.flatten()))
        with Pool(ncore) as pool:
            results = pool.map(self.evaluate, coords)
        for r in results:
            i, j, fit_result = r
            map_fit[:,i,j] = fit_result
        return map_fit


#####################################################################################
# Fitter functions
#####################################################################################

def spec_autofit(vel, spec, std=None, p0=None, 
                 mode='minimize', bootstrap=0,
                 snr_limit=5,
                 iteration=1,
                 velocity_search=None,
                 plot=False, ax=None,
                 velocity_bounds=[-500,500], amplitude_bounds=None, 
                 sigma_bounds=[30,1000], 
                 fit_cont=False, cont_bounds=None, 
                 steps=10000, discard=None, **kwargs):
    """an automatic spectra fitter, build upon fit_spec
    
    Args:
        vel: velocity
        spec: spectral data
        snr_limit: the accepted snr for each fitting
        iteration: do the fitting iteratively by removing 5sigma outliers each time
        bootstrap: bootstrap the flux to derive the fitting errors when mode=minimize

    Features:
    1. automatically determine the std based on the spectral, which needs the spectrum
       to be wide enough.
    2. iterative fitting to remove outliers after fitting
    3. robust error handling for all the fitted parameters
    """
    # determine the boundaries

    if std is None:
        vel_selection = (vel<velocity_bounds[0]) | (vel>velocity_bounds[1])
        std = np.std(spec[vel_selection]) 
    # run a first fit
    if amplitude_bounds is None:
        # amplitude_bounds = [-3**np.abs(np.max(spec)), 3*np.abs(np.max(spec))]
        amplitude_bounds = [0, 2*np.abs(np.max(spec))]
    if sigma_bounds is None:
        sigma_bounds = [0, np.abs(np.diff(velocity_bounds))]
    bounds_dict = {'velocity_bounds':velocity_bounds,
                   'amplitude_bounds':amplitude_bounds,
                   'sigma_bounds':sigma_bounds,
                   'cont_bounds':cont_bounds}
    if np.isnan(np.nansum(spec)):
        return None
    try:
        p_fit, spec_fit = fit_spec_norm(vel, spec, std=std, p0=p0, mode=mode,
                                   steps=steps, discard=discard,
                                   fit_cont=fit_cont, **bounds_dict, 
                                   **kwargs)
    except:
        # print("fit_spec: stoped due to an internal error!")
        return None
    # check the fitted SNR
    std_afterfit = np.std(spec - spec_fit)
    median = np.median(spec - spec_fit)
    chi2_sline = np.nansum((spec-median)**2/std_afterfit**2)
    chi2_fit = np.nansum((spec-spec_fit)**2 / std_afterfit**2)
    # print(p_fit, spec_fit)
    snr = (abs(chi2_sline - chi2_fit))**0.5
    
    # bootstrap
    if bootstrap > 0:
        # check std is provided
        noise_specs = np.random.randn(bootstrap, spec.size)*std
        p_bootstrap = np.zeros([bootstrap, 3])
        for i in range(bootstrap):
            p_fit_i, _ = fit_spec_norm(vel, spec+noise_specs[i], 
                                  std=std, p0=p_fit, mode=mode,
                                  steps=steps, discard=discard,
                                  fit_cont=fit_cont, 
                                  **kwargs)
            p_bootstrap[i] = p_fit_i
        p_fit_error = np.std(p_bootstrap, axis=0)
        print("Parameter errors:", p_fit_error)
    if plot:
        pass
    if snr > snr_limit:
        return p_fit
    else:
        return None 
 
def pixel_fit_cube(cube, velocity=None, mode='minimize', SNR_limit=3, 
                   plot=False, fit_cont=False,
                   smooth_width=None,
                   vel_low=-300, vel_up=300, savefile=None, debug=False, **kwargs):
    """fit gaussian line through the cube pixel by pixel

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
    fitmaps = np.full((5, cube_shape[-2], cube_shape[-1]), fill_value=np.nan)
    weight_map = np.zeros(imagesize)
    
   
    for y in range(0, cube_shape[-2]):
        for x in range(0, cube_shape[-1]):
            spec = cube[:,y,x]
            if np.nansum(spec) != 0:
                # measure the std
                cont_window = (velocity<=vel_low) | (velocity>=vel_up)
                std = np.nanstd(spec[cont_window])                            
                med = np.nanmedian(spec[cont_window])
                # get chi^2 of straight line fit
                chi2_sline = np.nansum((spec-med)**2 / std**2)
                # do a Gaussian profile fit of the line
                px, spec_fit = fit_spec_norm(velocity, spec, std, fit_cont=fit_cont, 
                              mode=mode)
                bestfit = gaussian_1d(px, velocity=velocity)
                # calculate the chi^2 of the Gaussian profile fit
                chi2_gauss = np.nansum((spec-bestfit)**2 / std**2)
                # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                SNR = (chi2_sline - chi2_gauss)**0.5
                # store the fit parameters with S/N>SNR_limit
                if SNR >= SNR_limit:
                    is_fitted = True
                    if debug:
                        print(f'fit found at {(y,x)} with S/N={SNR}')
                        print('chi2_sline, chi2_gauss', chi2_sline, chi2_gauss)
                        print('std, median', std, med)
                        print('mean,std,velocity', np.mean(spec), np.std(spec), np.mean(velocity))
                    fitmaps[0, y, x] = px[0]
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
    if plot:
        fig = plt.figure()
    # loop over all pixels in the cube and fit 1d spectra

    # fitmaps[0] = fitmaps[0]/weight_map
    if savefile is None:
        return fitcube, fitmaps
    else:
        # keep files into fits file
        pass

def pixel_fit_cube2(cube, velocity=None, mode='minimize', SNR_limit=3, 
                   plot=False, fit_cont=False,
                   smooth_width=None, 
                   minaper=0, maxaper=4, 
                   vel_low=-500, vel_up=500, savefile=None, debug=False, **kwargs):
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
    # fitcube = np.zeros_like(cube)
    imagesize = cube_shape[-2:]
    nspec = cube_shape[-3]
    # mean, median, std = astro_stats.sigma_clipped_stats(cube, sigma=10)
    # mask = np.ma.masked_invalid(cube).mask
    # vmax = 1.5*np.percentile(cube[~mask], 90)

    # A cube to save all the best-fit values (maps)
    # [amp, velocity, sigma, SNR]
    fitmaps = np.full((5, cube_shape[-2], cube_shape[-1]), fill_value=np.nan)
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
                        std = np.nanstd(spec[cont_window])                            
                        med = np.nanmedian(spec[cont_window])
                        # get chi^2 of straight line fit
                        chi2_sline = np.nansum((spec-med)**2 / std**2)
                        # do a Gaussian profile fit of the line
                        px,bestfit_spec = fit_spec_norm(velocity, spec, std, fit_cont=fit_cont, 
                                      mode=mode)
                        # calculate the chi^2 of the Gaussian profile fit
                        chi2_gauss = np.nansum((spec-bestfit_spec)**2 / std**2)
                        # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                        SNR = (chi2_sline - chi2_gauss)**0.5

                        # store the fit parameters with S/N>SNR_limit
                        if SNR >= SNR_limit:
                            is_fitted = True
                            if debug:
                                print('chi2_sline, chi2_gauss', chi2_sline, chi2_gauss)
                                print(f'fit found at {(y,x)} with S/N={SNR}')
                                print('ylow,yup,xlow,xup', ylow,yup,xlow,xup)
                                print('std, median', std, med)
                                print('mean,std,velocity', np.mean(spec), np.std(spec), np.mean(velocity))
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
                            # fitcube[:,y,x] = bestfit_spec

                            # plot data and fit if S/N is above threshold
                            if plot:
                                ax = fig.add_subplot(111)
                                ax.step(vel, spec, color='black', where='mid', label='data')
                                ax.plot(vel, bestfit_spec, color='red', alpha=0.8)
                                plt.show(block=False)
                                plt.pause(0.02)
                                plt.clf()
                        elif SNR<SNR_limit:
                            if debug:
                                print(f'no fit found at {(x,y)} with S/N={SNR}')
    # fitmaps[0] = fitmaps[0]/weight_map
    if savefile is None:
        return None, fitmaps
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
                     vel_low=-300, vel_up=300, sigma_guess=30, **kwargs):
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
                px,bestfit = fit_spec(velocity, spec, std, p0=last_fit, fit_cont=fit_cont, 
                                      mode=mode)
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
                     vel_low=-300, vel_up=300, sigma_guess=30, **kwargs):
    nchan, ny, nx = cube.shape
    cube_shape = cube.shape

    cont_window = (velocity<=1.5*vel_low) | (velocity>=1.5*vel_up)
    line_window = (velocity>vel_low) & (velocity<vel_up)
    # construct the priority map, first fit the higher priority pixels
    max_value = np.max(cube)
    m0_map = np.sum(cube[line_window]**2, axis=0) 
    m0_map = m0_map / np.max(m0_map)
    # get the distance to the center of the signal
    x_idx, y_idx = np.where(m0_map > np.percentile(m0_map, 95))
    yx_center = np.mean([y_idx, x_idx], axis=1) + np.array([0.5,0.5])
    yidx, xidx = np.mgrid[0:ny,0:nx]
    dist_map = 1/((yidx-yx_center[0])**2 + (xidx-yx_center[1])**2 + 1)
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
                std = np.std(spec[cont_window])                            
                med = np.median(spec[cont_window])
                # get chi^2 of straight line fit
                chi2_sline = np.sum((spec-med)**2 / std**2)
                px,bestfit = fit_spec(velocity, spec, std, p0=last_fit, fit_cont=fit_cont, 
                                      mode=mode)
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

def voronori_cubefit3(cube, velocity=None, mode='minimize', SNR_limit=3,
                      minaper=1, maxaper=5, vel_low=-300, vel_up=300, 
                      fit_cont=False, priority_map=None,
                      velocity_tolerance=1, 
                      debug=False, **kwargs):
    # v3: use apper based binning, instead of pixel squares
    nchan, ny, nx = cube.shape
    cube_shape = cube.shape

    cont_window = np.logical_or(velocity<=velocity_tolerance*vel_low, 
                                velocity>=velocity_tolerance*vel_up)
    line_window = np.logical_and(velocity>vel_low, velocity<vel_up)
    # construct the priority map, first fit the higher priority pixels
    if priority_map is None:
        # max_value = np.max(cube)
        m0_map = np.sum(cube[line_window]**2, axis=0) 
        m0_map = m0_map / np.max(m0_map)
        cost_map = -1 * m0_map 
    else:
        cost_map = -1*priority_map
    yidx, xidx = np.mgrid[0:ny,0:nx]
    # # get the distance to the center of the signal
    # x_idx, y_idx = np.where(m0_map > np.percentile(m0_map, 95))
    # yx_center = np.mean([y_idx, x_idx], axis=1) + np.array([0.5,0.5])
    # dist_map = 1/((yidx-yx_center[0])**2 + (xidx-yx_center[1])**2 + 1)
    # print(dist_map)
    # cost_map = 1 / (m0_map * dist_map)
    
    if debug:
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
    # fitcube = np.zeros_like(cube)
    fitmaps = np.full((5, ny, nx), fill_value=np.nan) 
    idxmap = np.zeros((ny, nx), dtype=int)
    sy, sx = np.unravel_index(np.argsort(cost_map, axis=None), (ny, nx))
    # print(spiral_idx[:5])
    # a container of the previous bestfit, useful for the following fit
    last_fit = None

    map_size = ny*nx
    for i in range(map_size):
        is_fitted = False
        yi,xi = sy[i], sx[i]
        # print('i=', i, [yi, xi])
        # check whether the pixel has been fitted
        if idxmap[yi,xi] > 0:
            continue
        # put the initial center of the aper as the pixel center
        yi_center, xi_center = yi, xi

        for aper in range(minaper, maxaper+1):
            if is_fitted:
                # idxmap[yi:yi+aper*dy,xi:xi+aper*dx] = i+1
                break
            # aper_j = EllipticalAperture((yi_center,xi_center), 0.5*aper, 0.5*aper, 0)
            # aper_mask = aper_j.to_mask().to_image((ny, nx)) > 0.5
            aper_j = EllipticalAperture((xi_center,yi_center), 0.5*aper, 0.5*aper, 0)
            aper_mask = aper_j.to_mask().to_image((ny, nx)) > 0.5
 
            bin_select = (~(idxmap > 0)) & (aper_mask)

            bin_select_cube = np.repeat(bin_select[None,:,:], nchan, axis=0)
            spec = np.sum(cube[bin_select_cube].reshape(
                    nchan, np.sum(bin_select)), axis=1)
 
            if np.nansum(spec) != 0:
                std = np.nanstd(spec[cont_window])                            
                if fit_cont:
                    chi2_sline = np.nansum(spec**2 / std**2)
                else:
                    med = np.nanmedian(spec[cont_window])
                    # get chi^2 of straight line fit
                    chi2_sline = np.nansum((spec-med)**2 / std**2)
                px,bestfit = fit_spec(velocity, spec, std, fit_cont=fit_cont, 
                                      mode=mode)
                # calculate the chi^2 of the Gaussian profile fit
                chi2_gauss = np.nansum((spec-bestfit)**2 / std**2)
                # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                if debug:
                    print('chi2_sline, chi2_gauss', chi2_sline, chi2_gauss)
                    print('std=', std, 'med', med)
                    print('mean,std,velocity', np.mean(spec), np.std(spec), np.mean(velocity))
                SNR = (chi2_sline - chi2_gauss)**0.5
                # print('SNR=', SNR)
                # print(xlow, xup, ylow, yup)
                # print(bin_select)
                if SNR >= SNR_limit:
                    is_fitted = True
                    last_fit = None#px
                    idxmap[bin_select] = i+1
                    # fitmaps[0:3, yi, xi] = px[:3]/np.array([aper**2,1,1]) # amp, vcenter, vsigma
                    fitmaps[0][bin_select] = px[0]/np.sum(bin_select) # amp
                    fitmaps[1][bin_select] = px[1] # vcenter
                    fitmaps[2][bin_select] = px[2] # vsigma
                    if fit_cont:
                        fitmaps[3][bin_select] = px[3]
                    fitmaps[4][bin_select] = SNR
                    # fitcube[:,yi,xi] = bestfit
                else:
                    # yi_center = np.max(yidx[bin_select] - yi)*0.5+yi
                    # xi_center = np.max(xidx[bin_select] - xi)*0.5+xi
                    yi_center = np.mean(yidx[bin_select])
                    xi_center = np.mean(xidx[bin_select])
                    # yi_center = np.unique(yidx[bin_select]).mean()
                    # xi_center = np.unique(xidx[bin_select]).mean()
    # print(idxmap)
    return None, fitmaps

def voronoi_binning(image, noise, priority_map=None, minaper=1, maxaper=5,
                    target_snr=5):
    ny, nx = image.shape
    if not isinstance(noise, np.ndarray):
        noise = np.full_like(image, fill_value=noise)

    if priority_map is None:
        priority_map = image / np.max(image)    
    cost_map = -1.0 * priority_map
    yidx, xidx = np.mgrid[0:ny,0:nx]

    idxmap = np.zeros((ny, nx), dtype=int)
    sy, sx = np.unravel_index(np.argsort(cost_map, axis=None), (ny, nx))

    map_size = ny*nx
    for i in range(map_size):
        yi,xi = sy[i], sx[i]
        # check whether the pixel has been fitted
        if idxmap[yi,xi] > 0:
            continue
        # put the initial center of the aper as the pixel center
        yi_center, xi_center = yi, xi
        is_fitted = False

        for aper in range(minaper, maxaper+1):
            if is_fitted:
                break
            aper_j = EllipticalAperture((xi_center,yi_center), 0.5*aper, 0.5*aper, 0)
            aper_mask = aper_j.to_mask().to_image((ny, nx)) > 0.5

            bin_select = (~(idxmap > 0)) & (aper_mask)
            snr = np.sum(image[bin_select]) / np.sqrt(np.sum(noise[bin_select]**2))
            if snr >= target_snr:
                is_fitted = True
                idxmap[bin_select] = i+1
    return idxmap
 

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

def plot_fitcube(fitmaps, flux_max=None, vel_min=-300, vel_max=300, 
                 sigma_max=None, ax=None, plotfile=None,
                 beam=None, **kwargs):
    if flux_max is None:
        flux_max = np.nanmax(fitmaps[0])*0.6
    if sigma_max is None:
        sigma_max = np.min([np.nanmax(fitmaps[-1]), 400])
    if ax is None:
        show_image = True
        fig, ax = plt.subplots(1,3, figsize=(12, 3))
    else:
        show_image = False
        fig = None
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

    if beam is not None:
        imagesize = fitmaps[0].shape
        ellipse = patches.Ellipse((0.1,0.1), transform=ax[0].transAxes,
                      width=beam[1]/imagesize[0], height=beam[0]/imagesize[1],
                      angle=beam[2], 
                      facecolor='orange', edgecolor='white', alpha=0.8)
        ax[0].add_patch(ellipse)


    for ai in ax:
        # ai.axis('off')
        ai.set_xticklabels([])
        ai.set_yticklabels([])
        ai.axes.get_xaxis().set_ticks([])
        ai.axes.get_yaxis().set_ticks([])
    if show_image:
        plt.show()
    if (plotfile is not None) and (fig is not None):
        fig.savefig(plotfile, bbox_inches='tight', dpi=200)



###############################################
# test functions
###############################################
def test_fit_spec(plot=False):
    """test the performance of fit_spec and fit_spec_norm
    """
    gaussian_truth = [4, -250, 200]
    vel = np.linspace(-2000, 2000, 500)
    line = gaussian_1d(gaussian_truth, velocity=vel)
    np.random.seed(1994)
    spec = line + np.random.randn(500) 

    print('truth:', gaussian_truth)
    bestfit1, bestfit1_spec = fit_spec(vel, spec, std=1, fit_cont=False, mode='minimize')
    print('fit_spec + minimize:', bestfit1)
    bestfit2, bestfit2_spec = fit_spec(vel, spec, std=1, fit_cont=False, mode='mcmc')
    print('fit_spec + mcmc:', bestfit2)
    
    bestfit3, bestfit3_spec = fit_spec_norm(vel, spec, std=1, fit_cont=False, mode='minimize',)
    print('fit_spec_norm + minimize:', bestfit3)
    bestfit4, bestfit4_spec = fit_spec_norm(vel, spec, std=1, fit_cont=False, mode='mcmc',)
    print('fit_spec_norm + mcmc:', bestfit4)
    if plot:
        fig, ax = plt.subplots(1, 4, figsize=(16,4))
        for i in range(4):
            ax[i].step(vel, spec, label='data')
            ax[i].step(vel, line, label='truth')
        ax[0].step(vel, bestfit1_spec, label='fit_spec+minimize')
        ax[1].step(vel, bestfit2_spec, label='fit_spec+mcmc')
        ax[2].step(vel, bestfit3_spec, label='fit_spec_norm+minimize')
        ax[3].step(vel, bestfit4_spec, label='fit_spec_norm+mcmc')
        for i in range(4):
            ax[i].legend()
        plt.show()

def test_fit_spectrum(plot=False):
    """test fit_spec with mcmc and minimize
    """
    # create a random spectrum
    gaussian_truth = [4, -150, 200]
    vel = np.linspace(-2000, 2000, 500)
    line = gaussian_1d(gaussian_truth, velocity=vel)
    # add std=1 normal distribution noise
    spec = line + np.random.randn(500) 

    bestfit1, bestfit1_spec = fit_spec(vel, spec, std=1, fit_cont=False, mode='minimize')
    print('truth:', gaussian_truth)
    print('best fit2:', bestfit1)
    
    bestfit2, bestfit2_spec = fit_spec(vel, spec, std=1, fit_cont=False, mode='mcmc',)
    print('best fit2:', bestfit2)

    assert np.std(bestfit1_spec - line) < 2 # 
    assert np.std(bestfit2_spec - line) < 2 # 

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(vel, spec, label='data')
        ax.step(vel, line, label='truth')
        ax.step(vel, bestfit1_spec, label='minimize')
        ax.step(vel, bestfit2_spec, label='mcmc')
        ax.legend()
        plt.show()

def test_SpecFitter(plot=False):
    try:
        from model_utils import Gaussian1D, Models
    except:
        raise ImportError('Cannot find model_utils!')
    # create a random spectrum
    grid1d_origin = Grid1D(size=200, pixelsize=1)
    grid1d_fit = Grid1D(size=200, pixelsize=1)
    gaussian_truth = [3, -10, 10, 0]
    amp0, mean0, sigma0, cont0 =gaussian_truth 
    gaussian_model1 = Gaussian1D(amplitude=Parameter(amp0, limits=[0,2]), 
                                 mean=Parameter(mean0, limits=[-2,2]), 
                                 sigma=Parameter(sigma0, limits=[0,5]),
                                 cont=0, #Parameter(cont0, limits=[-1,1]), 
                                 name='gaussian1')
    models = Models([gaussian_model1])
    model_data = models.create_model(grid1d_origin)
    std = 1.0
    spec = model_data + std*(np.random.randn(*model_data.shape))
    # data = {'var':grid1d.x, 'data':spec, 'data_err':y_data_std}
 
    # vel = np.linspace(-2000, 2000, 500)
    vel = grid1d_origin.x 
    
    fitter  = SpecFitter(models, grid1d_fit)

    bestfit1 = fitter.fit(vel, spec, std, mode='minimize')
    bestfit1_model = fitter.create_model(bestfit1, grid1d_fit)
    bestfit2 = fitter.fit(vel, spec, std, mode='mcmc')
    bestfit2_model = fitter.create_model(bestfit2, grid1d_fit)

    print('truth:', gaussian_truth)
    print('best fit2:', bestfit1)
    print('best fit2:', bestfit2)

    print(np.std(model_data - bestfit1_model), np.std(model_data - bestfit2_model))
    assert np.std(model_data - bestfit1_model) < 2*std
    assert np.std(model_data - bestfit2_model) < 2*std
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.step(vel, spec, label='data')
        ax.step(vel, model_data, label='truth')
        ax.step(grid1d_fit.x, bestfit1_model, label='minimize')
        ax.step(grid1d_fit.x, bestfit2_model, label='mcmc')
        ax.legend()
        plt.show()

def test_fit_cube():
    # test the minimize and mcmc algorithm
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
    """test three cube fitter (function) implementations and compare the performance
    """
    print('version:', __version__)
    nx, ny, nchan= 15, 15, 500
    vel_dispersion = 100
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [50,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-2000,2000,nchan)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d(
                    [flux_i, vel_i, vel_dispersion], vel_chan) \
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
    fit_cube1, fit_maps1 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=5,mode=mode,debug=False)
    fit_cube2, fit_maps2 = pixel_fit_cube2(cube_data, vel_chan, SNR_limit=5,minaper=0,maxaper=3,mode=mode,debug=False)
    fit_cube3, fit_maps3 = voronori_cubefit3(cube_data, vel_chan, SNR_limit=5,maxaper=7,mode=mode,debug=False,)
    # fit_cube2, fit_maps2 = voronori_cubefit(cube_data, vel_chan, SNR_limit=3,maxaper=6,mode=mode,)
    # fit_cube2, fit_maps2 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3, mode='mcmc')
    # fit_cube2, fit_maps2 = pixel_fit_cube(cube_data, vel_chan, SNR_limit=3, mode='mcmc')
    plot_fitcube(fit_maps1, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[1])
    plot_fitcube(fit_maps2, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[2])
    plot_fitcube(fit_maps3, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[3])
    plt.show()

def test_beam_smearing(debug=True):
    nx, ny, nchan= 20, 20, 500
    vel_dispersion = 100
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
   
    # smooth the cube, add beam smearing    
    smooth_width = 5
    gauss_kernel = gkern(smooth_width) 
    cube_data_smoothed = signal.convolve(cube_data, gauss_kernel[None,:,:], mode='same')

    hdr = fits.Header()
    hdr['OBSERVER'] = 'Your name'
    hdr['COMMENT'] = "Here's some comments about this FITS file."
    primary_hdu = fits.PrimaryHDU(header=hdr, data=cube_data)
    primary_hdu.writeto('cube_raw.fits', overwrite=True)

    hdr = fits.Header()
    hdr['OBSERVER'] = 'Your name'
    hdr['COMMENT'] = "Here's some comments about this FITS file."
    primary_hdu = fits.PrimaryHDU(header=hdr, data=cube_data_smoothed)
    primary_hdu.writeto('cube_smoothed.fits', overwrite=True)

def test_CubeFitter(mode='minimize'):
    """test three cube fitter (function) implementations and compare the performance
    """
    try:
        from model_utils import Gaussian1D, Models
    except:
        raise ImportError('Cannot find model_utils!')
    print('version:', __version__)
    nx, ny, nchan= 15, 15, 500
    vel_dispersion = 100
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [100,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-2000,2000,nchan)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d(
                    [flux_i, vel_i, vel_dispersion], vel_chan) \
                            + np.random.randn(nchan)

    dchan = vel_chan[1]-vel_chan[0]
    M0 = np.sum(cube_data*dchan, axis=0)
    M1 = np.sum(cube_data*dchan*vel_chan[:,None,None], axis=0)/M0

    # define the model
    gaussian_model1 = Gaussian1D(amplitude=Parameter(500, limits=[0,1000]), 
                                 mean=Parameter(0, limits=[-300,300]), 
                                 sigma=Parameter(100, limits=[0,300]),
                                 cont=0, #Parameter(cont0, limits=[-1,1]), 
                                 name='gaussian1')
    models = Models([gaussian_model1])
 
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
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
    cubefitter = CubeFitter(cube_data, velocity=vel_chan)
    fit_cube1, fit_maps1 = cubefitter.fit_cube(mode='minimize', SNR_limit=5, debug=False)
    plot_fitcube(fit_maps1, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[1])
    plt.show()

def test_CubeMPFitter(mode='minimize', snr_limit=5):
    """test CubeMPFitter 
    """
    try:
        from model_utils import Gaussian1D, Models
    except:
        raise ImportError('Cannot find model_utils!')
    print('version:', __version__)
    nx, ny, nchan= 10, 10, 200
    vel_dispersion = 100
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [10,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-1000,1000,nchan)
    np.random.seed(1994)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d(
                    [flux_i, vel_i, vel_dispersion], vel_chan) \
                            + np.random.randn(nchan)

    dchan = vel_chan[1]-vel_chan[0]
    M0 = np.sum(cube_data*dchan, axis=0)
    M1 = np.sum(cube_data*dchan*vel_chan[:,None,None], axis=0)/M0
    # plot the model
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
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
    cubempfitter = CubeMPFitter(cube_data, velocity=vel_chan, snr_limit=snr_limit, mode=mode, steps=5000)
    fit_maps1 = cubempfitter.fit(ncore=8)
    plot_fitcube(fit_maps1, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[1])
    fit_maps2 = cubempfitter.adaptive_fit(ncore=8)
    plot_fitcube(fit_maps2, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[2])
    plt.show()

def test_CubeMPFitter_test(mode='minimize', snr_limit=5):
    """test CubeMPFitter2, compare CubeMPFitter and CubeMPFitter2
    """
    try:
        from model_utils import Gaussian1D, Models
    except:
        raise ImportError('Cannot find model_utils!')
    print('version:', __version__)
    nx, ny, nchan= 10, 10, 200
    vel_dispersion = 100
    vel_map = create_thin_disk(nx, ny, inc=np.pi/3, pa=-np.pi/10)
    flux_map = create_sersic_disk(nx, ny, [10,2,1,0,0,0.5,np.pi/2-np.pi/10])
    # flux_map = 0.1*np.random.randn(nx,ny) + flux_map
    # create the datacube
    cube_data = np.zeros((nchan, ny, nx))
    vel_chan = np.linspace(-1000,1000,nchan)
    np.random.seed(1994)
    for yi in range(ny):
        for xi in range(nx):
            flux_i = flux_map[yi, xi]
            vel_i = vel_map[yi, xi]
            cube_data[:, yi, xi] = gaussian_1d(
                    [flux_i, vel_i, vel_dispersion], vel_chan) \
                            + np.random.randn(nchan)

    dchan = vel_chan[1]-vel_chan[0]
    M0 = np.sum(cube_data*dchan, axis=0)
    M1 = np.sum(cube_data*dchan*vel_chan[:,None,None], axis=0)/M0
    # plot the model
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
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
    start_time = time.time()
    cubempfitter1 = CubeMPFitter(cube_data, velocity=vel_chan, snr_limit=snr_limit, mode=mode)
    fit_maps1 = cubempfitter1.fit(ncore=8)
    print("Method1:", time.time() - start_time)
    start_time = time.time()
    cubempfitter2 = CubeMPFitter2(cube_data, velocity=vel_chan, snr_limit=snr_limit, mode=mode)
    fit_maps2 = cubempfitter2.fit(ncore=8)
    print("Method2:", time.time() - start_time)
    plot_fitcube(fit_maps1, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[1])
    plot_fitcube(fit_maps2, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[2])
    # fit_maps2 = cubempfitter.adaptive_fit(ncore=8)
    # plot_fitcube(fit_maps2, vel_min=-400, vel_max=400, flux_max=20, sigma_max=300, ax=ax[2])
    plt.show()

def test_all():
    test_fit_spec()
    test_fit_spectrum()
