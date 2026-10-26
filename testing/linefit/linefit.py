#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2026-07-09: first release, v0.1
    2026-10-25: add multiprocessing support, v0.2
"""


from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial


C_KMS: float = 299792.45 #: Speed of light, km/s.
__version__ = '0.2'


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

# ---------------------------------------------------------------------------
# Cube fitting: shared helpers (serial and multiprocessing)
# ---------------------------------------------------------------------------

def _prepare_cube(
    cube: str | os.PathLike | np.ndarray,
    velocity: Sequence[float] | np.ndarray | None,
    smooth_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load/sanitize a datacube and apply optional spatial smoothing.

    Parameters
    ----------
    cube
        Datacube array with shape ``(nspec, ny, nx)``, or the path of a
        FITS file whose primary HDU holds such an array.
    velocity
        Velocity axis (km/s) matching ``cube.shape[0]``.
    smooth_width
        If given, Gaussian sigma (in pixels) used to smooth the cube
        spatially (both image axes) before fitting.

    Returns
    -------
    Tuple ``(cube, velocity)`` of float arrays.

    Raises
    ------
    ValueError
        If the cube is not 3D, if `velocity` is missing, or if the
        velocity axis length does not match the spectral axis.
    """
    if isinstance(cube, (str, os.PathLike)):
        with fits.open(cube) as hdul:
            data = np.squeeze(hdul[0].data)
        data = np.asarray(data, dtype=float)
    else:
        data = np.asarray(cube, dtype=float)

    if data.ndim != 3:
        raise ValueError(
            f"cube must be 3D (nspec, ny, nx), got shape {data.shape}"
        )
    if velocity is None:
        raise ValueError("velocity axis is required for cube fitting")
    velocity = np.asarray(velocity, dtype=float)
    if velocity.size != data.shape[0]:
        raise ValueError(
            f"velocity axis has {velocity.size} channels but cube has "
            f"{data.shape[0]}"
        )

    if smooth_width is not None:
        # Spatial (image-plane) smoothing only; the spectral axis is untouched.
        data = ndimage.gaussian_filter(
            data, sigma=(0.0, smooth_width, smooth_width), mode="nearest"
        )
    return data, velocity

def _fit_one_spectrum(
    spec: np.ndarray,
    velocity: np.ndarray,
    vel_low: float,
    vel_up: float,
    snr_limit: float | None,
    fit_kwargs: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit one spectrum of a cube and derive its integrated S/N.

    The S/N is ``sqrt(chi2_line - chi2_gauss)``, i.e. the improvement in
    chi-square of the Gaussian fit over a constant continuum.

    Parameters
    ----------
    spec
        Spectrum to fit, extracted along the spectral axis.
    velocity
        Velocity axis (km/s), same length as `spec`.
    vel_low, vel_up
        Velocity limits of the line window; pixels outside
        ``[vel_low, vel_up]`` are used to estimate the continuum level
        and noise.
    snr_limit
        Minimum S/N for a fit to be accepted; ``None`` accepts all.
    fit_kwargs
        Extra keyword arguments forwarded to
        :func:`fit_gaussian_line`.
    rng
        Random generator forwarded to the Monte-Carlo error loop.

    Returns
    -------
    ``(params, model)`` with ``params = [flux, velocity, sigma, 0, snr]``
    (index 3 reserved for the continuum), or ``None`` if the pixel is
    blank, unusable, or rejected by `snr_limit`.
    """
    # Blank pixels (all zeros / no finite samples) carry no signal to fit
    if np.nansum(spec) == 0:
        return None

    cont_window = (velocity <= vel_low) | (velocity >= vel_up)
    if not np.any(cont_window):
        # Degenerate window: fall back to the full spectrum
        cont_window = np.ones_like(velocity, dtype=bool)
    std = np.nanstd(spec[cont_window])
    med = np.nanmedian(spec[cont_window])
    if not np.isfinite(std) or std <= 0:
        # No usable noise estimate (e.g. constant spectrum)
        return None

    # chi^2 of a constant (straight-line) model, used as the null model
    chi2_sline = np.nansum((spec - med) ** 2 / std ** 2)
    # Gaussian profile fit of the line
    if rng is not None:
        fit_kwargs = dict(fit_kwargs, rng=rng)
    fit_result = fit_gaussian_line(velocity, spec, std, **fit_kwargs)
    spec_fit = fit_result["model"]
    chi2_gauss = np.nansum((spec - spec_fit) ** 2 / std ** 2)
    # sqrt(delta chi^2) = S/N of the detection; noise-only pixels can give
    # a negative delta chi^2, which propagates as NaN (same as before).
    with np.errstate(invalid="ignore"):
        snr = (chi2_sline - chi2_gauss) ** 0.5

    if (snr_limit is not None) and (snr < snr_limit):
        return None

    params = np.array(
        [fit_result["flux"], fit_result["velocity"], fit_result["sigma"],
         0.0, snr],
        dtype=float,
    )
    return params, np.asarray(spec_fit, dtype=float)

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
    """Fit a Gaussian to every spectrum of a datacube (serial version).

    Parameters
    ----------
    cube
        Datacube with shape ``(nspec, ny, nx)``, or a FITS file holding
        one.
    velocity
        Velocity axis (km/s) of the cube.
    plot
        Unused; kept for API compatibility.
    smooth_width
        Optional Gaussian sigma (pixels) for spatial smoothing applied
        before fitting.
    vel_low, vel_up
        Velocity limits of the line window; pixels outside are used for
        the continuum level and noise.
    snr_limit
        Minimum integrated S/N; pixels below it are left blank.
    debug
        Print a short timing summary.
    **kwargs
        Forwarded to :func:`fit_gaussian_line` (e.g. ``n_mc``,
        ``weight_type``).

    Returns
    -------
    ``(fitcube, fitmaps)`` where `fitcube` holds the model cube and
    `fitmaps` the maps ``[flux, velocity, sigma, continuum, snr]``.
    """
    cube, velocity = _prepare_cube(cube, velocity, smooth_width)
    nspec, ny, nx = cube.shape

    # Model cube, and a cube of best-fit values [amp, vel, sigma, cont, snr]
    fitcube = np.zeros_like(cube)
    fitmaps = np.full((5, ny, nx), fill_value=np.nan)

    t_start = time.time()
    if debug:
        print(f"[linefit] serial fit of {ny * nx} spectra, {nspec} channels")

    for y in range(ny):
        for x in range(nx):
            out = _fit_one_spectrum(
                cube[:, y, x], velocity, vel_low, vel_up, snr_limit, kwargs
            )
            if out is None:
                continue
            params, spec_fit = out
            fitmaps[:, y, x] = params
            fitcube[:, y, x] = spec_fit

    if debug:
        print(f"[linefit] serial fit done in {time.time() - t_start:.1f} s")

    return fitcube, fitmaps

# ---------------------------------------------------------------------------
# Parallel (multiprocessing) cube fitting
# ---------------------------------------------------------------------------

# Per-process state, populated once by the Pool initializer so that large
# read-only inputs (velocity axis, configuration) are not pickled per task.
_WORKER_CTX: dict[str, Any] = {}

def _init_worker(
    velocity: np.ndarray,
    vel_low: float,
    vel_up: float,
    snr_limit: float | None,
    fit_kwargs: dict[str, Any],
    random_seed: int | None,
    debug: bool,
) -> None:
    """Store shared fitting configuration in the current worker process.

    Parameters
    ----------
    velocity
        Velocity axis (km/s) shared by all spectra.
    vel_low, vel_up
        Line-window limits used for the continuum estimate.
    snr_limit
        Minimum integrated S/N.
    fit_kwargs
        Keyword arguments forwarded to :func:`fit_gaussian_line`.
    random_seed
        Base seed for the per-pixel Monte-Carlo generators (used only
        when ``fit_kwargs['n_mc'] > 0`` and no ``rng`` was supplied).
    debug
        Whether to report per-block failures. The value is also needed
        in workers, so it is passed explicitly.
    """
    _WORKER_CTX.update(
        velocity=velocity, vel_low=vel_low, vel_up=vel_up,
        snr_limit=snr_limit, fit_kwargs=fit_kwargs,
        random_seed=random_seed, debug=debug,
    )

def _fit_pixel_block(
    task: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a block of spectra (one pool task).

    Parameters
    ----------
    task
        ``(flat_indices, spectra)`` where `flat_indices` are the
        row-major pixel indices in the image plane and `spectra` has
        shape ``(nblock, nspec)``.

    Returns
    -------
    ``(flat_indices, params, models)`` with `params` shaped
    ``(nblock, 5)`` and `models` shaped ``(nblock, nspec)``. Unfitted or
    rejected pixels keep ``NaN`` parameters and a zero model.
    """
    flat_idx, spectra = task
    ctx = _WORKER_CTX
    velocity = ctx["velocity"]
    fit_kwargs = ctx["fit_kwargs"]
    base_seed = ctx["random_seed"]
    n_mc = int(fit_kwargs.get("n_mc", 0) or 0)

    nblock = spectra.shape[0]
    params = np.full((nblock, 5), np.nan)
    models = np.zeros_like(spectra)

    for i in range(nblock):
        # Give every pixel an independent, reproducible MC noise stream
        rng = None
        if n_mc > 0 and "rng" not in fit_kwargs and base_seed is not None:
            rng = np.random.default_rng(base_seed + int(flat_idx[i]))
        try:
            out = _fit_one_spectrum(
                spectra[i], velocity, ctx["vel_low"], ctx["vel_up"],
                ctx["snr_limit"], fit_kwargs, rng=rng,
            )
        except Exception:  # keep one bad pixel from killing the whole pool
            if ctx["debug"]:
                print(f"[linefit] fit failed for pixel {flat_idx[i]}:",
                      file=sys.stderr)
                traceback.print_exc()
            continue
        if out is None:
            continue
        params[i], models[i] = out

    return flat_idx, params, models

def fit_cube_gaussian_parallel(
        cube,
        velocity=None,
        plot=False,
        smooth_width=None,
        vel_low=-300,
        vel_up=300,
        snr_limit=2.0,
        debug=False,
        n_workers=None,
        chunk_size=50,
        random_seed=None,
        **kwargs):
    """Fit a Gaussian to every spectrum of a datacube in parallel.

    Thin multiprocessing wrapper around :func:`fit_gaussian_line`. Blank
    pixels (zero summed flux) are filtered out before the work is split
    into blocks of `chunk_size` spectra, and each block is fitted in a
    worker process. Results are identical to :func:`fit_cube_gaussian`
    (up to floating-point ordering when ``n_mc`` is used) because both
    call the same per-spectrum helper.

    Parameters
    ----------
    cube
        Datacube with shape ``(nspec, ny, nx)``, or a FITS file holding
        one.
    velocity
        Velocity axis (km/s) of the cube.
    plot
        Unused; kept for API compatibility.
    smooth_width
        Optional Gaussian sigma (pixels) for spatial smoothing applied
        before fitting.
    vel_low, vel_up
        Velocity limits of the line window.
    snr_limit
        Minimum integrated S/N; fainter pixels are left blank.
    debug
        Print worker/pixel diagnostics and a timing summary.
    n_workers
        Number of worker processes. Defaults to ``os.cpu_count()``,
        capped by the number of pixel blocks.
    chunk_size
        Number of spectra handed to a worker per task. Larger values
        amortize pickling overhead; smaller values improve load
        balancing.
    random_seed
        Base seed for the per-pixel Monte-Carlo generators when
        ``n_mc > 0``. Each pixel uses ``random_seed + pixel_index``, so
        results are reproducible but independent across workers.
    **kwargs
        Forwarded to :func:`fit_gaussian_line` (e.g. ``n_mc``,
        ``weight_type``, ``convolve_kernel``).

    Returns
    -------
    ``(fitcube, fitmaps)`` where `fitcube` holds the model cube and
    `fitmaps` the maps ``[flux, velocity, sigma, continuum, snr]``.
    """
    cube, velocity = _prepare_cube(cube, velocity, smooth_width)
    nspec, ny, nx = cube.shape

    fitcube = np.zeros_like(cube)
    fitmaps = np.full((5, ny, nx), fill_value=np.nan)

    # Flatten the image plane: one column per spectrum (zero-copy views)
    flat_cube = cube.reshape(nspec, -1)
    flat_maps = fitmaps.reshape(5, -1)
    flat_models = fitcube.reshape(nspec, -1)

    # Only pixels holding signal need a fit
    valid = np.flatnonzero(np.nansum(flat_cube, axis=0) != 0)
    if valid.size == 0:
        if debug:
            print("[linefit] no non-blank pixels found, nothing to fit")
        return fitcube, fitmaps

    if chunk_size is None or chunk_size < 1:
        chunk_size = 1
    chunk_size = int(chunk_size)
    n_blocks = int(np.ceil(valid.size / chunk_size))
    cpu_count = os.cpu_count() or 1
    n_workers = cpu_count if n_workers is None else int(n_workers)
    n_workers = max(1, min(n_workers, n_blocks))

    t_start = time.time()
    if debug:
        print(
            f"[linefit] parallel fit of {valid.size} pixels "
            f"({ny * nx} total) with {n_workers} workers, "
            f"{n_blocks} blocks of <= {chunk_size}"
        )

    def _block_tasks():
        """Yield (flat_indices, spectra) blocks lazily to bound memory."""
        for start in range(0, valid.size, chunk_size):
            idx = valid[start:start + chunk_size]
            yield idx, np.ascontiguousarray(flat_cube[:, idx].T)

    if n_workers == 1:
        # Nothing to gain from a pool; stay in-process to keep it simple
        _init_worker(velocity, vel_low, vel_up, snr_limit, kwargs,
                     random_seed, debug)
        for idx, spectra in _block_tasks():
            idx, params, models = _fit_pixel_block((idx, spectra))
            flat_maps[:, idx] = params.T
            flat_models[:, idx] = models.T
    else:
        # 'spawn' is the default on macOS/Windows: the initializer payload
        # must therefore be picklable (plain arrays, scalars, dicts).
        ctx = mp.get_context()
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(velocity, vel_low, vel_up, snr_limit, kwargs,
                      random_seed, debug),
        ) as pool:
            # chunksize=1: our tasks are already coarse-grained blocks
            for idx, params, models in pool.imap_unordered(
                    _fit_pixel_block, _block_tasks(), chunksize=1):
                flat_maps[:, idx] = params.T
                flat_models[:, idx] = models.T

    if debug:
        n_fitted = int(np.isfinite(fitmaps[0]).sum())
        print(
            f"[linefit] parallel fit done in {time.time() - t_start:.1f} s "
            f"({n_fitted} pixels accepted at S/N >= {snr_limit})"
        )

    return fitcube, fitmaps

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

def test_fit_cube_gaussian_parallel(n_workers=2, chunk_size=3, plot=False):
    """Parallel and serial cube fits must agree pixel by pixel.

    Parameters
    ----------
    n_workers
        Number of worker processes used in the parallel run.
    chunk_size
        Spectra per pool task.
    plot
        Show the recovered flux/velocity/sigma maps.
    """
    np.random.seed(1994)
    nspec, ny, nx = 120, 6, 5
    velocity = np.linspace(-1000, 1000, nspec)

    # Mock cube: one Gaussian line per pixel, plus a blank (no-data) pixel
    amp_map = np.full((ny, nx), 10.0)
    amp_map[0, 0] = 40.0
    vel_map = np.linspace(-200, 200, ny * nx).reshape(ny, nx)
    sig_map = np.full((ny, nx), 120.0)
    cube = np.zeros((nspec, ny, nx))
    for y in range(ny):
        for x in range(nx):
            cube[:, y, x] = gaussian_conv_model(
                velocity, amp_map[y, x], sig_map[y, x], vel_map[y, x]
            )
    cube += np.random.randn(nspec, ny, nx) * 0.1
    cube[:, -1, -1] = 0.0  # blank pixel, must stay unfitted in both runs

    cube_serial, maps_serial = fit_cube_gaussian(
        cube, velocity, snr_limit=2.0
    )
    cube_par, maps_par = fit_cube_gaussian_parallel(
        cube, velocity, snr_limit=2.0, n_workers=n_workers,
        chunk_size=chunk_size,
    )

    assert cube_par.shape == cube.shape
    assert maps_par.shape == (5, ny, nx)
    np.testing.assert_allclose(maps_par, maps_serial, equal_nan=True, rtol=1e-6)
    np.testing.assert_allclose(cube_par, cube_serial, equal_nan=True, rtol=1e-6)

    # Blank pixel untouched, bright pixel recovered
    assert np.all(np.isnan(maps_par[:, -1, -1]))
    assert not np.any(cube_par[:, -1, -1])
    assert abs(maps_par[0, 0, 0] - amp_map[0, 0]) < 5.0
    assert abs(maps_par[1, 0, 0] - vel_map[0, 0]) < 30.0

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, i, label in zip(axes, (0, 1, 2), ('flux', 'velocity', 'sigma')):
            im = ax.imshow(maps_par[i], origin='lower')
            ax.set_title(label)
            fig.colorbar(im, ax=ax)
        plt.show()

