#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2026-07-09: first release, v0.1
"""


from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial


C_KMS: float = 299792.45 #: Speed of light, km/s.
__version__ = '0.1'


# ---------------------------------------------------------------------------
# Single-Gaussian (optionally instrument-convolved) line fit
# ---------------------------------------------------------------------------

def make_gaussian_kernel(
    velocity_offsets: np.ndarray, 
    fwhm_kms: float
    ) -> np.ndarray:
    """Build a normalized Gaussian convolution kernel.

    Parameters
    ----------
    velocity_offsets
        Velocity offsets (km/s) from the kernel centre, sampled on the
        same grid spacing as the data to be convolved.
    fwhm_kms
        Full width at half maximum of the kernel, in km/s. Values
        below ~1e-3 effectively yield a delta function (no smoothing).

    Returns
    -------
    Kernel array, same shape as `velocity_offsets`, normalized to unit
    sum.
    """
    sigma = fwhm_kms / 2.3548200450309493
    if sigma <= 1.0e-3:
        kernel = np.zeros_like(velocity_offsets)
        kernel[int(np.argmin(np.abs(velocity_offsets)))] = 1.0
        return kernel
    kernel = np.exp(-0.5 * (velocity_offsets / sigma) ** 2)
    return kernel / kernel.sum()

def gaussian_conv_model(
    velocity: np.ndarray,
    amplitude: float,
    sigma_kms: float,
    velocity_offset: float,
    convolve_kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a Gaussian line profile, optionally convolved with a kernel.

    Parameters
    ----------
    velocity
        Velocity axis (km/s) on which to evaluate the model.
    amplitude
        Integrated line flux: the profile is normalized to sum to
        `amplitude` before any convolution.
    sigma_kms
        Gaussian dispersion, in km/s (absolute value is used, clipped
        to a small positive floor to avoid degenerate profiles).
    velocity_offset
        Centroid velocity of the Gaussian, in km/s.
    convolve_kernel
        Optional fixed convolution kernel (e.g. an instrumental
        profile), sampled on the same grid spacing as `velocity` and
        with the same array length.

    Returns
    -------
    Model flux array, same shape as `velocity`.

    Raises
    ------
    ValueError
        If `template_kernel` is given but its shape does not match
        `velocity`.
    """
    sigma_kms = max(abs(sigma_kms), 1.0e-3)
    profile = np.exp(-0.5 * ((velocity - velocity_offset) / sigma_kms) ** 2)
    norm = profile.sum()
    profile = amplitude * profile / norm if norm > 0 else np.zeros_like(profile)

    if convolve_kernel is not None:
        if convolve_kernel.shape != velocity.shape:
            raise ValueError("template_kernel must match the shape of `velocity`")
        profile = ndimage.convolve1d(profile, convolve_kernel, mode="nearest")
    return profile

def _chi2(
    params: np.ndarray,
    velocity: np.ndarray,
    data: np.ndarray,
    weight: np.ndarray,
    convolve_kernel: np.ndarray | None,
    mask: np.ndarray | None,
) -> float:
    """Chi-square cost function used by :func:`fit_gaussian_line`.

    Parameters
    ----------
    params
        ``[amplitude, sigma_kms, velocity_offset]``.
    velocity
        Velocity axis (km/s).
    data
        Observed (continuum-subtracted) flux.
    weight
        Per-pixel weight, normally the inversed noise^2
    template_kernel
        Optional fixed convolution kernel; see
        :func:`gaussian_conv_model`.
    mask
        Optional boolean mask selecting which pixels contribute to the
        chi-square sum (used for outlier rejection).

    Returns
    -------
    Chi-square value (sum of squared, noise-weighted residuals).
    """
    model = gaussian_conv_model(velocity, params[0], params[1], params[2], convolve_kernel)
    resid = (model - data) * weight
    if mask is not None:
        resid = resid[mask]
    return float(np.sum(resid ** 2))

def _mc_error(samples: np.ndarray, nominal: float, n_mc: int) -> float:
    """Symmetrized 68%-confidence-interval error from Monte-Carlo samples.

    Parameters
    ----------
    samples
        Monte-Carlo realizations of the quantity of interest (``NaN``
        entries are dropped).
    nominal
        Best-fit (noiseless) value of the quantity.
    n_mc
        Number of Monte-Carlo trials originally requested (used to
        locate the 16th/84th percentile indices).

    Returns
    -------
    Mean absolute distance from `nominal` to the 16th/84th percentile
    of `samples`, or ``NaN`` if too few finite samples remain.
    """
    arr = samples[np.isfinite(samples)]
    if arr.size < 2:
        return np.nan
    s = np.sort(arr)
    lo_i = min(round(n_mc * (1 - 0.6827) / 2), s.size - 1)
    hi_i = min(lo_i + round(n_mc * 0.6827), s.size - 1)
    return float(np.mean([abs(nominal - s[lo_i]), abs(nominal - s[hi_i])]))

def fit_gaussian_line(
    velocity: np.ndarray,
    flux: np.ndarray,
    noise: np.ndarray | None = None,
    convolve_kernel: np.ndarray | None = None,
    weight_type: str = "uniform",
    clip_sigma: float = 2.5,
    n_mc: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    """Fit a single Gaussian (optionally instrument-convolved) to a line.

    Uses a two-pass Nelder-Mead simplex fit: an initial fit, followed by
    a sigma-clip of the residuals and a refit restricted to the
    surviving pixels.

    Parameters
    ----------
    velocity
        Velocity axis (km/s) of the spectrum.
    flux
        Continuum-subtracted flux of the spectrum.
    noise
        Per-pixel noise of the spectrum, or ``None``.
    convolve_kernel
        Optional fixed convolution kernel representing the
        instrumental profile; see :func:`gaussian_conv_model`.
    weight_type
        ``"uniform"``, ``"poisson"`` or ``"gauss"``.
    clip_sigma
        Sigma-clipping threshold applied to the first-pass residuals.
    n_mc
        Number of Monte-Carlo noise realizations for error estimation.
        If zero, error entries are ``NaN``.
    rng
        Random number generator for the Monte-Carlo loop.

    Returns
    -------
    Dictionary with keys ``flux``, ``sigma``, ``velocity``,
    ``flux_err``, ``sigma_err``, ``velocity_err`` and ``model`` (the
    best-fit model array, same shape as `velocity`).
    """
    nan_result: dict[str, float | np.ndarray] = dict(
        flux=np.nan, sigma=np.nan, velocity=np.nan,
        flux_err=np.nan, sigma_err=np.nan, velocity_err=np.nan,
        model=np.full_like(flux, np.nan),
    )
    if flux.size < 3:
        return nan_result

    use_noise = noise is not None and np.all(np.isfinite(noise))
    if use_noise:
        if weight_type == "gauss":
            inv_sigma = 1.0 / np.clip(noise, 1e-10, None)
        elif weight_type == "poisson":
            inv_sigma = 1.0 / np.sqrt(np.clip(noise, 1e-10, None))
        else:
            inv_sigma = np.ones_like(flux)
    else:
        inv_sigma = np.ones_like(flux)
    #v0 = np.nansum(flux*velocity)/np.nansum(flux)
    p0 = np.array([float(np.sum(flux)), 50, 0])
    opts = dict(xatol=1e-4, fatol=1e-4, maxiter=2000, maxfev=2000)
    res1 = minimize(_chi2, p0, args=(velocity, flux, inv_sigma, convolve_kernel, None),
                     method="Nelder-Mead", options=opts)
    if not np.all(np.isfinite(res1.x)):
        return nan_result

    model1 = gaussian_conv_model(velocity, *res1.x, convolve_kernel)
    resid = flux - model1
    sigma_resid = float(np.std(resid, ddof=0))
    mask = np.abs(resid - np.mean(resid)) <= clip_sigma * max(sigma_resid, 1e-12)

    opts2 = dict(xatol=1e-5, fatol=1e-5, maxiter=2000, maxfev=2000)
    res2 = minimize(_chi2, res1.x, args=(velocity, flux, inv_sigma, convolve_kernel, mask),
                     method="Nelder-Mead", options=opts2)
    p_best = res2.x if np.all(np.isfinite(res2.x)) else res1.x

    model_best = gaussian_conv_model(velocity, *p_best, convolve_kernel)
    result: dict[str, float | np.ndarray] = dict(
        flux=float(p_best[0]), sigma=float(abs(p_best[1])), velocity=float(p_best[2]),
        flux_err=np.nan, sigma_err=np.nan, velocity_err=np.nan, model=model_best,
    )

    if n_mc > 0:
        if rng is None:
            rng = np.random.default_rng()
        noise_for_mc = noise if use_noise else np.full_like(flux, sigma_resid)
        flux_mc = np.full(n_mc, np.nan)
        sigma_mc = np.full(n_mc, np.nan)
        vel_mc = np.full(n_mc, np.nan)
        for i in range(n_mc):
            perturbed = flux + rng.normal(size=flux.size) * noise_for_mc
            r = minimize(_chi2, p_best,
                         args=(velocity, perturbed, inv_sigma, convolve_kernel, mask),
                         method="Nelder-Mead", options=opts2)
            if np.all(np.isfinite(r.x)):
                flux_mc[i], sigma_mc[i], vel_mc[i] = r.x[0], abs(r.x[1]), r.x[2]
        result["flux_err"] = _mc_error(flux_mc, result["flux"], n_mc)
        result["sigma_err"] = _mc_error(sigma_mc, result["sigma"], n_mc)
        result["velocity_err"] = _mc_error(vel_mc, result["velocity"], n_mc)

    return result

def fit_continuum(
    velocity: np.ndarray,
    flux: np.ndarray,
    noise: np.ndarray | None,
    cont_windows: Sequence[tuple[float, float]],
    poly_order: int = 1,
    eval_wave: np.ndarray | None = None,
    n_clip_iter: int = 5,
    clip_sigma: float = 4.0,
) -> dict[str, float | np.ndarray]:
    """Fit the continuum in one or more velocity windows.

    Parameters
    ----------
    velocity
        Full velocity axis of the spectrum.
    flux
        Full flux array, same shape as `velocity`.
    noise
        Full per-pixel noise array, same shape as `velocity`, or ``None``.
        Used only to weight the polynomial fit.
    cont_windows
        Sequence of ``(lo, hi)`` velocity pairs defining the
        continuum-fitting regions.
    poly_order
        Degree of the polynomial continuum fit (ignored if
        `sdss_cont` is set).
    eval_wave
        Velocities at which the returned continuum *level* is
        evaluated (typically the line window). Defaults to the
        continuum-window velocities.
    n_clip_iter
        Maximum number of sigma-clipping iterations for the polynomial
        fit.
    clip_sigma
        Clipping threshold, in units of the residual standard
        deviation.

    Returns
    -------
    Dictionary with keys: ``order``, ``continuum``, ``level``,
    ``level_err``, ``coeffs``.

    Raises
    ------
    ValueError
        If no finite data points are found in the continuum windows.
    """
    mask = np.zeros_like(velocity, dtype=bool)
    for lo, hi in cont_windows:
        mask |= (velocity >= lo) & (velocity <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("no data points found in the requested continuum windows")

    w = velocity[idx]
    f = flux[idx]
    n = noise[idx] if noise is not None else None
    finite = np.isfinite(f)

    keep = finite.copy()
    poly = Polynomial(np.zeros(poly_order + 1))
    for _ in range(n_clip_iter):
        if int(keep.sum()) < poly_order + 1:
            break
        weights = None
        if n is not None and np.all(np.isfinite(n[keep])) and np.all(n[keep] > 0):
            weights = 1.0 / n[keep]
        poly = Polynomial.fit(w[keep], f[keep], poly_order, w=weights)
        resid_full = f - poly(w)
        sigma = float(np.std(resid_full[keep], ddof=0))
        new_keep = finite & (np.abs(resid_full) <= clip_sigma * max(sigma, 1e-12))
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    resid_final = f[keep] - poly(w[keep])
    sigma = float(np.std(resid_final, ddof=0))
    level_err = sigma / np.sqrt(max(int(keep.sum()), 1))

    model = poly(velocity)
    eval_at = eval_wave if eval_wave is not None else w
    level = float(np.mean(poly(eval_at)))

    return {
        "order": poly_order,
        "continuum": model,
        "level": level,
        "level_err": level_err,
        "coeffs": poly.convert().coef,
    }

def fit_cube_gaussian(
        cube, 
        velocity=None, 
        plot=False, 
        smooth_width=None,
        vel_low=-300, 
        vel_up=300, 
        snr_limit=2.0,
        debug=False, 
        **kwargs):
    """fit cube pixel by pixel using curve_fit with 1d gaussian model

    Args:
        cube: it can be the fitsfile or the datacube
        velocity: the velocity of the cube
    """
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
                fit_result = fit_gaussian_line(velocity, spec, std,)
                spec_fit = fit_result['model']
                chi2_gauss = np.nansum((spec-spec_fit)**2 / std**2)
                # calculate the S/N of the fit: sqrt(delta_chi^2)=S/N
                SNR = (chi2_sline - chi2_gauss)**0.5

                if (snr_limit is not None) and (SNR < snr_limit):
                    continue
                # store the fit parameters
                fitmaps[0, y, x] = fit_result['flux']
                fitmaps[1, y, x] = fit_result['velocity']
                fitmaps[2, y, x] = fit_result['sigma']
                fitmaps[3, y, x] = 0 # reserved for continuum
                fitmaps[4, y, x] = SNR
                fitcube[:,y,x] = spec_fit

    return fitcube, fitmaps

# def line_reference_stacking(vel, ref_spec, target_spec, vel_target=None, mode='median'):
#     """stacking one weak line based on the velocity of a strong line
#
#     This is duplication of line_stacking, but can be used for two lines simutaneously
#     """
#     if vel_target is None:
#         vel_target = vel
#     velocity_list = []
#     stacked_ref_spec = np.zeros([len(ref_spec), len(vel)]) 
#     for i, spec in enumerate(ref_spec):
#         # best_fit_params, best_fit = fit_spec_astropy(vel, spec)
#         best_fit_params, best_fit = fit_gaussian_line(vel, spec, sigma_bounds=[40, 400])
#         velocity_list.append(best_fit_params[1])
#         # extrapolate the new ref spectra
#         grid_interper = interpolate.RegularGridInterpolator(
#             [vel-best_fit_params[1], ], spec, method='cubic', 
#             fill_value=np.nan, bounds_error=False)
#         stacked_ref_spec[i] = grid_interper(vel)
#     # stacking the target specrtra with velocity offset
#     stacked_target_spec = np.zeros([len(target_spec), len(vel_target)]) 
#     for i,spec in enumerate(target_spec):
#         vel_off = velocity_list[i]
#         vel_new = vel_target - vel_off
#         # extrapolate the new target spectra
#         grid_interper = interpolate.RegularGridInterpolator(
#             [vel_new, ], spec, method='cubic', 
#             fill_value=np.nan, bounds_error=False)
#         stacked_target_spec[i] = grid_interper(vel_target)
#     if mode == 'median':
#         return (np.nanmedian(stacked_ref_spec, axis=0), 
#                 np.nanmedian(stacked_target_spec, axis=0))
#     elif mode == 'mean':
#         return (np.nanmean(stacked_ref_spec, axis=0), 
#                 np.nanmean(stacked_target_spec, axis=0))
#

# ---------------------------------------------------------------------------
# testing function for each main function 
# ---------------------------------------------------------------------------

def test_fit_gaussian_line(plot: bool = False) -> None:
    """test the performance of fit_spec and fit_spec_norm

    Parameters
    ----------
    plot
        plot the fitting results, option for manual check

    """
    model_truth = [100, 200, -250] # flux, sigma, vel_offset
    vel = np.linspace(-2000, 2000, 500)
    line = gaussian_conv_model(vel, *model_truth, convolve_kernel=None)
    np.random.seed(1994)
    spec = line + np.random.randn(500) 

    print('truth:', model_truth)
    result = fit_gaussian_line(vel, spec, n_mc=100)
    print(f"flux={result['flux']} +- {result['flux_err']}")
    print(f"sigma={result['sigma']} +- {result['sigma_err']}")
    print(f"velocity_offset={result['velocity']} +- {result['velocity_err']}")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8,3))
        ax.step(vel, line, label='truth')
        ax.step(vel, spec, label='data')
        ax.step(vel, result['model'], label='best fit')
        ax.legend()
        plt.show()
    assert abs(result['flux'] - model_truth[0]) < result['flux_err']
    assert abs(result['sigma'] - model_truth[1]) < result['sigma_err']
    assert abs(result['velocity'] - model_truth[2]) < result['velocity_err']

def test_fit_continuum(plot=False):
    """Outliers should be clipped and not bias the fit significantly."""
    velocity = np.linspace(-500, 500, 300)
    true_slope, true_intercept = 0.01, 5.0
    flux = true_slope * velocity + true_intercept
    flux_origin = flux.copy()

    # inject large outliers
    outlier_idx = np.array([10, 50, 150, 250])
    flux[outlier_idx] += 20.0

    cont_windows = [(-500, -300), (300, 500)]
    result = fit_continuum(
        velocity, flux, noise=None, cont_windows=cont_windows,
        poly_order=1, n_clip_iter=5, clip_sigma=3.0
    )
    
    #print(result)
    #print([true_intercept, true_slope])
    #print(result['coeffs'])

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8,3))
        ax.step(velocity, flux_origin, label='truth')
        ax.step(velocity, flux, label='true+outliers')
        ax.step(velocity, result['continuum'], label='model')
        ax.legend()
        plt.show()
    # With clipping, fitted level should be close to true_level despite outliers
    #assert abs(result["level"] - true_level) < 1.0
    assert isinstance(result, dict)
    assert set(result.keys()) == {"order", "continuum", "level", "level_err", "coeffs"}
    assert result["order"] == 1
    assert result["continuum"].shape == velocity.shape
    np.testing.assert_allclose(result["coeffs"], [true_intercept, true_slope], atol=1e-4)
 

