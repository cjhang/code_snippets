#!/usr/bin/env python3
"""
linefit.py
==========

Minimal emission-line fitting pipeline for IFU/longslit spectra, ported
(and deliberately simplified) from the IDL package ``linefit.pro`` /
``kinfit.pro``.

For each spaxel of a 3-D datacube (or a single 1-D spectrum) the
following steps are performed:

1. **Continuum fit** in one or more wavelength windows, either as an
   iteratively sigma-clipped polynomial of arbitrary order, or as the
   SDSS-style mean flux between the 40th and 60th percentile of the
   continuum-window flux distribution.
2. **Moments** (0th, 1st, 2nd) of the continuum-subtracted line,
   computed on a velocity grid centred on the line rest wavelength,
   with optional ``uniform`` / ``poisson`` / ``gauss`` weighting and
   optional Monte-Carlo error propagation.
3. **Single-Gaussian fit** of the continuum-subtracted line, optionally
   convolved with a fixed instrumental Gaussian kernel, fitted with a
   two-pass Nelder-Mead simplex (outlier pixels are sigma-clipped
   between the two passes) and optional Monte-Carlo error propagation.

Not implemented here (see the full translation for these):
NII multi-Gaussian fitting, Voronoi binning, diagnostic plotting,
instrument-specific template/resolution selection.

Third-party requirements
------------------------
``numpy``, ``scipy``, ``astropy``.
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

#: Speed of light, km/s.
C_KMS: float = 299792.458

WeightType = Literal["uniform", "poisson", "gauss"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def cut_isolated_outliers(indices: np.ndarray) -> np.ndarray:
    """Trim isolated single/double index outliers from both ends of a run.

    Parameters
    ----------
    indices
        Sorted array of integer indices (e.g. pixels passing a
        signal threshold).

    Returns
    -------
    Trimmed index array, or ``array([-1])`` if fewer than three
    contiguous indices remain (an "empty selection" sentinel).
    """
    idx = np.asarray(indices)
    if idx.size <= 2:
        return np.array([-1])

    idx = idx.copy()
    changed = True
    while changed and idx.size > 2:
        changed = False
        if idx[1] - 1 != idx[0]:
            idx = idx[1:]
            changed = True
        elif idx.size > 2 and idx[2] - 2 != idx[0]:
            idx = idx[2:]
            changed = True

    changed = True
    while changed and idx.size > 2:
        changed = False
        n = idx.size
        if idx[n - 1] - 1 != idx[n - 2]:
            idx = idx[:-1]
            changed = True
        elif n > 2 and idx[n - 1] - 2 != idx[n - 3]:
            idx = idx[:-2]
            changed = True

    if idx.size < 3:
        return np.array([-1])
    return idx


def _is_empty_selection(idx: np.ndarray) -> bool:
    """Return whether an index array is the sentinel "empty" selection.

    Parameters
    ----------
    idx
        Index array, typically the output of
        :func:`cut_isolated_outliers`.

    Returns
    -------
    ``True`` if `idx` represents an empty selection.
    """
    return idx.size == 0 or (idx.size == 1 and idx[0] == -1)


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


def convert_flux_density_to_flux(data: np.ndarray, wave_step_um: float) -> np.ndarray:
    """Convert flux density (W/m^2/um) into flux (erg/s/cm^2).

    Parameters
    ----------
    data
        Input array in W/m^2/um.
    wave_step_um
        Spectral pixel scale (``CDELT``), in microns.

    Returns
    -------
    Array of the same shape as `data`, in erg/s/cm^2.
    """
    return data * wave_step_um * 1.0e3


# ---------------------------------------------------------------------------
# Continuum model / fit
# ---------------------------------------------------------------------------

@dataclass
class ContinuumModel:
    """Callable continuum model (polynomial or constant).

    Attributes
    ----------
    kind
        ``"poly"`` (evaluated with :func:`numpy.polyval`) or
        ``"const"`` (a single constant level).
    coeffs
        Polynomial coefficients (highest degree first) if
        ``kind == "poly"``, or a one-element array holding the
        constant level if ``kind == "const"``.
    """

    kind: Literal["poly", "const"]
    coeffs: np.ndarray

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        """Evaluate the continuum model at the given wavelengths.

        Parameters
        ----------
        wave
            Wavelengths at which to evaluate the continuum.

        Returns
        -------
        Continuum flux evaluated at `wave`.
        """
        if self.kind == "poly":
            return np.polyval(self.coeffs, wave)
        return np.full_like(np.asarray(wave, dtype=float), float(self.coeffs[0]))


def fit_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    noise: np.ndarray | None,
    cont_windows: Sequence[tuple[float, float]],
    poly_order: int = 1,
    sdss_cont: bool = False,
    eval_wave: np.ndarray | None = None,
    n_clip_iter: int = 5,
    clip_sigma: float = 4.0,
) -> tuple[ContinuumModel, float, float]:
    """Fit the continuum in one or more wavelength windows.

    Parameters
    ----------
    wave
        Full wavelength axis of the spectrum.
    flux
        Full flux array, same shape as `wave`.
    noise
        Full per-pixel noise array, same shape as `wave`, or ``None``.
        Used only to weight the polynomial fit.
    cont_windows
        Sequence of ``(lo, hi)`` wavelength pairs defining the
        continuum-fitting regions.
    poly_order
        Degree of the polynomial continuum fit (ignored if
        `sdss_cont` is set).
    sdss_cont
        If set, use the SDSS-style 40th-60th percentile mean continuum
        instead of a polynomial fit.
    eval_wave
        Wavelengths at which the returned continuum *level* is
        evaluated (typically the line window). Defaults to the
        continuum-window wavelengths.
    n_clip_iter
        Maximum number of sigma-clipping iterations for the polynomial
        fit.
    clip_sigma
        Clipping threshold, in units of the residual standard
        deviation.

    Returns
    -------
    A 3-tuple ``(model, level, level_err)``: `model` is a callable
    :class:`ContinuumModel`, `level` is the continuum evaluated at
    `eval_wave` (mean value), and `level_err` is its estimated 1-sigma
    uncertainty.

    Raises
    ------
    ValueError
        If no finite data points are found in the continuum windows.
    """
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in cont_windows:
        mask |= (wave >= lo) & (wave <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("no data points found in the requested continuum windows")

    w = wave[idx]
    f = flux[idx]
    n = noise[idx] if noise is not None else None
    finite = np.isfinite(f)

    if sdss_cont:
        good = finite & (f != 0)
        if good.sum() < 2:
            raise ValueError("not enough finite continuum points for the SDSS-style fit")
        lo_pct, hi_pct = np.percentile(f[good], [40, 60])
        sel = good & (f >= lo_pct) & (f <= hi_pct)
        level0 = float(np.mean(f[sel]))
        resid = f[sel] - level0
        sigma = float(np.std(resid, ddof=0))
        level_err = sigma / np.sqrt(max(int(sel.sum()), 1))
        model = ContinuumModel("const", np.array([level0]))
        eval_at = eval_wave if eval_wave is not None else w
        return model, float(np.mean(model(eval_at))), level_err

    keep = finite.copy()
    coeffs = np.zeros(poly_order + 1)
    for _ in range(n_clip_iter):
        if int(keep.sum()) < poly_order + 1:
            break
        weights = None
        if n is not None and np.all(np.isfinite(n[keep])) and np.all(n[keep] > 0):
            weights = 1.0 / n[keep]
        coeffs = np.polyfit(w[keep], f[keep], poly_order, w=weights)
        resid_full = f - np.polyval(coeffs, w)
        sigma = float(np.std(resid_full[keep], ddof=0))
        new_keep = finite & (np.abs(resid_full) <= clip_sigma * max(sigma, 1e-12))
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    resid_final = f[keep] - np.polyval(coeffs, w[keep])
    sigma = float(np.std(resid_final, ddof=0))
    level_err = sigma / np.sqrt(max(int(keep.sum()), 1))

    model = ContinuumModel("poly", coeffs)
    eval_at = eval_wave if eval_wave is not None else w
    level = float(np.mean(model(eval_at)))
    return model, level, level_err


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def compute_moments(
    velocity: np.ndarray,
    flux_line: np.ndarray,
    noise_line: np.ndarray | None,
    threshold_sigma: float,
    cont_level: float,
    cont_noise_est: float,
    wave_step_um: float,
    weight_type: WeightType = "uniform",
    n_mc: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute flux/velocity/dispersion moments of a continuum-subtracted line.

    Parameters
    ----------
    velocity
        Velocity axis (km/s), relative to the assumed line centre.
    flux_line
        Continuum-subtracted flux in the line window.
    noise_line
        Per-pixel noise in the line window, or ``None``.
    threshold_sigma
        Minimum signal (in units of `cont_noise_est`) for a pixel to be
        included in the moment sums.
    cont_level
        Mean continuum level under the line (used for the equivalent
        width).
    cont_noise_est
        Noise level estimated from the continuum-fit residuals.
    wave_step_um
        Spectral pixel scale, in microns (equivalent-width unit
        conversion to Angstrom).
    weight_type
        ``"uniform"``, ``"poisson"`` or ``"gauss"``. Poisson/Gauss
        weighting requires `noise_line` to be finite.
    n_mc
        Number of Monte-Carlo noise realizations for error estimation.
        If zero, error entries are ``NaN``.
    rng
        Random number generator for the Monte-Carlo loop. A new
        default generator is created if not supplied.

    Returns
    -------
    Dictionary with keys ``flux``, ``velocity``, ``dispersion``,
    ``eqw``, ``flux_err``, ``velocity_err``, ``dispersion_err``.
    """
    result = dict(
        flux=np.nan, velocity=np.nan, dispersion=np.nan, eqw=np.nan,
        flux_err=np.nan, velocity_err=np.nan, dispersion_err=np.nan,
    )

    use_noise = noise_line is not None and np.all(np.isfinite(noise_line))
    if use_noise:
        if weight_type == "gauss":
            weights = 1.0 / np.clip(noise_line, 1e-10, None) ** 2
        elif weight_type == "poisson":
            weights = 1.0 / np.clip(noise_line, 1e-10, None)
        else:
            weights = np.ones_like(flux_line)
    else:
        weights = np.ones_like(flux_line)

    def _select(data: np.ndarray) -> np.ndarray:
        good = np.where(data > threshold_sigma * cont_noise_est)[0]
        return cut_isolated_outliers(good)

    def _compute(data: np.ndarray, good: np.ndarray) -> tuple[float, float, float]:
        fg, wg, xg = data[good], weights[good], velocity[good]
        sw = fg * wg
        flux = float(sw.sum() / wg.sum() * good.size) if use_noise else float(fg.sum())
        vel = float((sw * xg).sum() / sw.sum())
        disp = float(np.sqrt((sw * (xg - vel) ** 2).sum() / sw.sum()))
        return flux, vel, disp

    good = _select(flux_line)
    if _is_empty_selection(good):
        return result

    flux, vel, disp = _compute(flux_line, good)
    eqw = flux / cont_level * wave_step_um * 1.0e4 if cont_level else np.nan
    result.update(flux=flux, velocity=vel, dispersion=disp, eqw=eqw)

    if n_mc > 0:
        if rng is None:
            rng = np.random.default_rng()
        noise_for_mc = noise_line if use_noise else np.full_like(flux_line, cont_noise_est)
        flux_mc = np.full(n_mc, np.nan)
        vel_mc = np.full(n_mc, np.nan)
        disp_mc = np.full(n_mc, np.nan)
        for i in range(n_mc):
            perturbed = flux_line + rng.normal(size=flux_line.size) * noise_for_mc
            g = _select(perturbed)
            if _is_empty_selection(g):
                continue
            flux_mc[i], vel_mc[i], disp_mc[i] = _compute(perturbed, g)

        result["flux_err"] = _mc_error(flux_mc, flux, n_mc)
        result["velocity_err"] = _mc_error(vel_mc, vel, n_mc)
        result["dispersion_err"] = _mc_error(disp_mc, disp, n_mc)

    return result


# ---------------------------------------------------------------------------
# Single-Gaussian (optionally instrument-convolved) line fit
# ---------------------------------------------------------------------------

def make_gaussian_kernel(velocity_offsets: np.ndarray, fwhm_kms: float) -> np.ndarray:
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
    template_kernel: np.ndarray | None = None,
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
    template_kernel
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

    if template_kernel is not None:
        if template_kernel.shape != velocity.shape:
            raise ValueError("template_kernel must match the shape of `velocity`")
        profile = ndimage.convolve1d(profile, template_kernel, mode="nearest")
    return profile


def _chi2(
    params: np.ndarray,
    velocity: np.ndarray,
    data: np.ndarray,
    inv_sigma: np.ndarray,
    template_kernel: np.ndarray | None,
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
    inv_sigma
        Per-pixel inverse noise weight.
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
    model = gaussian_conv_model(velocity, params[0], params[1], params[2], template_kernel)
    resid = (model - data) * inv_sigma
    if mask is not None:
        resid = resid[mask]
    return float(np.sum(resid ** 2))


def fit_gaussian_line(
    velocity: np.ndarray,
    flux_line: np.ndarray,
    noise_line: np.ndarray | None,
    template_kernel: np.ndarray | None = None,
    weight_type: WeightType = "uniform",
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
        Velocity axis (km/s) of the line window.
    flux_line
        Continuum-subtracted flux in the line window.
    noise_line
        Per-pixel noise in the line window, or ``None``.
    template_kernel
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
        model=np.full_like(flux_line, np.nan),
    )
    if flux_line.size < 3:
        return nan_result

    use_noise = noise_line is not None and np.all(np.isfinite(noise_line))
    if use_noise:
        if weight_type == "gauss":
            inv_sigma = 1.0 / np.clip(noise_line, 1e-10, None)
        elif weight_type == "poisson":
            inv_sigma = 1.0 / np.sqrt(np.clip(noise_line, 1e-10, None))
        else:
            inv_sigma = np.ones_like(flux_line)
    else:
        inv_sigma = np.ones_like(flux_line)

    p0 = np.array([float(np.sum(flux_line)), 50.0, 0.0])
    opts = dict(xatol=1e-4, fatol=1e-4, maxiter=2000, maxfev=2000)
    res1 = minimize(_chi2, p0, args=(velocity, flux_line, inv_sigma, template_kernel, None),
                     method="Nelder-Mead", options=opts)
    if not np.all(np.isfinite(res1.x)):
        return nan_result

    model1 = gaussian_conv_model(velocity, *res1.x, template_kernel)
    resid = flux_line - model1
    sigma_resid = float(np.std(resid, ddof=0))
    mask = np.abs(resid - np.mean(resid)) <= clip_sigma * max(sigma_resid, 1e-12)

    opts2 = dict(xatol=1e-5, fatol=1e-5, maxiter=2000, maxfev=2000)
    res2 = minimize(_chi2, res1.x, args=(velocity, flux_line, inv_sigma, template_kernel, mask),
                     method="Nelder-Mead", options=opts2)
    p_best = res2.x if np.all(np.isfinite(res2.x)) else res1.x

    model_best = gaussian_conv_model(velocity, *p_best, template_kernel)
    result: dict[str, float | np.ndarray] = dict(
        flux=float(p_best[0]), sigma=float(abs(p_best[1])), velocity=float(p_best[2]),
        flux_err=np.nan, sigma_err=np.nan, velocity_err=np.nan, model=model_best,
    )

    if n_mc > 0:
        if rng is None:
            rng = np.random.default_rng()
        noise_for_mc = noise_line if use_noise else np.full_like(flux_line, sigma_resid)
        flux_mc = np.full(n_mc, np.nan)
        sigma_mc = np.full(n_mc, np.nan)
        vel_mc = np.full(n_mc, np.nan)
        for i in range(n_mc):
            perturbed = flux_line + rng.normal(size=flux_line.size) * noise_for_mc
            r = minimize(_chi2, p_best,
                         args=(velocity, perturbed, inv_sigma, template_kernel, mask),
                         method="Nelder-Mead", options=opts2)
            if np.all(np.isfinite(r.x)):
                flux_mc[i], sigma_mc[i], vel_mc[i] = r.x[0], abs(r.x[1]), r.x[2]
        result["flux_err"] = _mc_error(flux_mc, result["flux"], n_mc)
        result["sigma_err"] = _mc_error(sigma_mc, result["sigma"], n_mc)
        result["velocity_err"] = _mc_error(vel_mc, result["velocity"], n_mc)

    return result


# ---------------------------------------------------------------------------
# Single-spectrum and full-cube driver
# ---------------------------------------------------------------------------

_SPECTRUM_KEYS = (
    "cont", "cont_err",
    "flux_mom", "flux_mom_err", "vel_mom", "vel_mom_err",
    "disp_mom", "disp_mom_err", "eqw_mom",
    "flux_gau", "flux_gau_err", "vel_gau", "vel_gau_err",
    "disp_gau", "disp_gau_err",
)


def _nan_spectrum_result(return_model: bool = False) -> dict[str, float | np.ndarray]:
    """Return an all-``NaN`` result dict with the standard spectrum keys.

    Parameters
    ----------
    return_model
        If set, also include ``model_velocity`` / ``model_flux`` keys
        (both set to empty arrays).

    Returns
    -------
    Dictionary with all values set to ``NaN`` (or empty arrays for the
    optional model keys).
    """
    result = {k: np.nan for k in _SPECTRUM_KEYS}
    if return_model:
        result["model_velocity"] = np.array([])
        result["model_flux"] = np.array([])
    return result


def fit_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    noise: np.ndarray | None,
    line_window: tuple[float, float],
    cont_windows: Sequence[tuple[float, float]],
    line_center: float,
    *,
    poly_order: int = 1,
    sdss_cont: bool = False,
    weight_type: WeightType = "uniform",
    mom_threshold: float = 0.5,
    instrument_fwhm_kms: float = 0.0,
    n_mc: int = 0,
    rng: np.random.Generator | None = None,
    return_model: bool = False,
) -> dict[str, float | np.ndarray]:
    """Fit continuum, moments and a single Gaussian for one spectrum.

    Parameters
    ----------
    wave
        Full wavelength axis of the spectrum (microns).
    flux
        Full flux array, same shape as `wave`.
    noise
        Full per-pixel noise array, same shape as `wave`, or ``None``.
    line_window
        ``(lo, hi)`` wavelength window used for the line moments/fit.
    cont_windows
        Sequence of ``(lo, hi)`` wavelength pairs defining the
        continuum-fitting regions.
    line_center
        Rest (or expected) wavelength of the line, used as the
        velocity-axis zero point.
    poly_order
        Degree of the polynomial continuum fit (ignored if
        `sdss_cont` is set).
    sdss_cont
        Use the SDSS-style 40th-60th percentile mean continuum instead
        of a polynomial fit.
    weight_type
        ``"uniform"``, ``"poisson"`` or ``"gauss"``.
    mom_threshold
        Minimum signal (in units of the continuum noise) for a pixel
        to be included in the moment sums.
    instrument_fwhm_kms
        If greater than zero, the fitted Gaussian is convolved with a
        fixed instrumental Gaussian kernel of this FWHM (km/s) before
        being compared to the data.
    n_mc
        Number of Monte-Carlo noise realizations for error estimation.
        If zero, error entries are ``NaN``.
    rng
        Random number generator for the Monte-Carlo loops.
    return_model
        If set, additionally return ``model_velocity`` and
        ``model_flux`` (the best-fit Gaussian model, evaluated on the
        line-window velocity grid).

    Returns
    -------
    Dictionary with keys ``cont``, ``cont_err``, ``flux_mom``,
    ``flux_mom_err``, ``vel_mom``, ``vel_mom_err``, ``disp_mom``,
    ``disp_mom_err``, ``eqw_mom``, ``flux_gau``, ``flux_gau_err``,
    ``vel_gau``, ``vel_gau_err``, ``disp_gau``, ``disp_gau_err`` (plus
    optionally ``model_velocity`` / ``model_flux``).
    """
    line_mask = (wave >= line_window[0]) & (wave <= line_window[1])
    if line_mask.sum() < 3:
        return _nan_spectrum_result(return_model)

    wave_line = wave[line_mask]

    try:
        cont_model, cont_level, cont_level_err = fit_continuum(
            wave, flux, noise, cont_windows,
            poly_order=poly_order, sdss_cont=sdss_cont, eval_wave=wave_line,
        )
    except ValueError:
        return _nan_spectrum_result(return_model)

    flux_line = flux[line_mask] - cont_model(wave_line)
    noise_line = noise[line_mask] if noise is not None else None
    velocity = (wave_line - line_center) / line_center * C_KMS
    wave_step_um = float(np.median(np.diff(wave)))

    cont_noise_est = cont_level_err if np.isfinite(cont_level_err) and cont_level_err > 0 \
        else float(np.nanstd(flux_line))

    mom = compute_moments(
        velocity, flux_line, noise_line,
        threshold_sigma=mom_threshold, cont_level=cont_level,
        cont_noise_est=cont_noise_est, wave_step_um=wave_step_um,
        weight_type=weight_type, n_mc=n_mc, rng=rng,
    )

    kernel = None
    if instrument_fwhm_kms > 0:
        offsets = velocity - velocity[len(velocity) // 2]
        kernel = make_gaussian_kernel(offsets, instrument_fwhm_kms)

    gau = fit_gaussian_line(
        velocity, flux_line, noise_line, template_kernel=kernel,
        weight_type=weight_type, n_mc=n_mc, rng=rng,
    )

    result: dict[str, float | np.ndarray] = dict(
        cont=cont_level, cont_err=cont_level_err,
        flux_mom=mom["flux"], flux_mom_err=mom["flux_err"],
        vel_mom=mom["velocity"], vel_mom_err=mom["velocity_err"],
        disp_mom=mom["dispersion"], disp_mom_err=mom["dispersion_err"],
        eqw_mom=mom["eqw"],
        flux_gau=gau["flux"], flux_gau_err=gau["flux_err"],
        vel_gau=gau["velocity"], vel_gau_err=gau["velocity_err"],
        disp_gau=gau["sigma"], disp_gau_err=gau["sigma_err"],
    )
    if return_model:
        result["model_velocity"] = velocity
        result["model_flux"] = gau["model"]
    return result


def fit_cube(
    wave: np.ndarray,
    cube: np.ndarray,
    noise_cube: np.ndarray | None,
    line_window: tuple[float, float],
    cont_windows: Sequence[tuple[float, float]],
    line_center: float,
    *,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    **spectrum_kwargs,
) -> dict[str, np.ndarray]:
    """Fit continuum, moments and a single Gaussian for every spaxel.

    Parameters
    ----------
    wave
        Wavelength axis (microns), shared by all spaxels.
    cube
        Data cube of shape ``(nx, ny, n_wave)``.
    noise_cube
        Noise cube of the same shape as `cube`, or ``None``.
    line_window
        ``(lo, hi)`` wavelength window used for the line moments/fit.
    cont_windows
        Sequence of ``(lo, hi)`` wavelength pairs for the continuum.
    line_center
        Rest (or expected) wavelength of the line.
    rng
        Random number generator shared across all spaxels' Monte-Carlo
        loops. A new default generator is created if not supplied.
    verbose
        If set, print progress every 10 rows.
    **spectrum_kwargs
        Additional keyword arguments forwarded to :func:`fit_spectrum`
        (e.g. ``poly_order``, ``sdss_cont``, ``weight_type``,
        ``mom_threshold``, ``instrument_fwhm_kms``, ``n_mc``).

    Returns
    -------
    Dictionary mapping each of the :func:`fit_spectrum` output keys to
    a 2-D array of shape ``(nx, ny)``.
    """
    nx, ny, _ = cube.shape
    maps = {k: np.full((nx, ny), np.nan) for k in _SPECTRUM_KEYS}
    if rng is None:
        rng = np.random.default_rng()

    for ix in range(nx):
        if verbose and ix % 10 == 0:
            print(f"fitting row {ix + 1}/{nx}")
        for iy in range(ny):
            spec = cube[ix, iy, :]
            noise_spec = noise_cube[ix, iy, :] if noise_cube is not None else None
            res = fit_spectrum(
                wave, spec, noise_spec, line_window, cont_windows,
                line_center, rng=rng, **spectrum_kwargs,
            )
            for k in _SPECTRUM_KEYS:
                maps[k][ix, iy] = res[k]
    return maps


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def load_cube(
    path: str,
    data_ext: int = 0,
    noise_ext: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, fits.Header]:
    """Load a 3-D IFU datacube (and optional noise cube) from a FITS file.

    Parameters
    ----------
    path
        Path to the FITS file.
    data_ext
        Extension index holding the science data.
    noise_ext
        Extension index holding the noise/stddev cube, or ``None`` if
        not available.

    Returns
    -------
    A 4-tuple ``(wave, cube, noise_cube, header)``: `wave` is the
    wavelength axis (microns), `cube` has shape ``(nx, ny, n_wave)``,
    `noise_cube` is either ``None`` or has the same shape as `cube`,
    and `header` is the data-extension FITS header.

    Raises
    ------
    ValueError
        If the data extension does not contain a 3-D array.
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[data_ext].data, dtype=float)
        header = hdul[data_ext].header.copy()
        noise = np.asarray(hdul[noise_ext].data, dtype=float) if noise_ext is not None else None

    if data.ndim != 3:
        raise ValueError(f"expected a 3-D cube, got shape {data.shape}")

    # FITS/numpy convention: data.shape == (NAXIS3, NAXIS2, NAXIS1).
    cube = np.transpose(data, (2, 1, 0))
    noise_cube = np.transpose(noise, (2, 1, 0)) if noise is not None else None

    naxis3 = int(header.get("NAXIS3", cube.shape[2]))
    crval3 = float(header.get("CRVAL3", 0.0))
    cdelt3 = float(header.get("CDELT3", 1.0))
    crpix3 = float(header.get("CRPIX3", 1.0))
    wave = (np.arange(naxis3) + 1 - crpix3) * cdelt3 + crval3

    return wave, cube, noise_cube, header


def save_maps(
    outdir: str,
    maps: dict[str, np.ndarray],
    header: fits.Header | None = None,
    suffix: str = "",
) -> None:
    """Write each 2-D map to its own FITS file.

    Parameters
    ----------
    outdir
        Output directory (created if it does not exist).
    maps
        Dictionary of 2-D arrays, as returned by :func:`fit_cube`.
    header
        Optional reference header from which spatial WCS keywords are
        copied into each output file, if present.
    suffix
        Suffix appended to each output filename, before ``.fits``.
    """
    os.makedirs(outdir, exist_ok=True)
    spatial_keys = ("CRVAL1", "CDELT1", "CRPIX1", "CUNIT1",
                     "CRVAL2", "CDELT2", "CRPIX2", "CUNIT2")
    for name, arr in maps.items():
        hdu = fits.PrimaryHDU(np.asarray(arr, dtype=np.float32).T)
        if header is not None:
            for key in spatial_keys:
                if key in header:
                    hdu.header[key] = header[key]
        hdu.writeto(os.path.join(outdir, f"{name}{suffix}.fits"), overwrite=True)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="Fit an emission line (continuum + moments + single "
                     "Gaussian) in every spaxel of a 3-D IFU datacube.",
    )
    parser.add_argument("cube", help="Path to the input FITS datacube.")
    parser.add_argument("--data-ext", type=int, default=0,
                         help="FITS extension index holding the science data.")
    parser.add_argument("--noise-ext", type=int, default=None,
                         help="FITS extension index holding the noise cube.")
    parser.add_argument("--line-center", type=float, required=True,
                         help="Rest/expected wavelength of the line, in microns.")
    parser.add_argument("--line-window", type=float, nargs=2, metavar=("LO", "HI"),
                         default=None,
                         help="Line-fitting wavelength window. Defaults to "
                              "line_center +/- 0.008 microns.")
    parser.add_argument("--cont", type=float, nargs="+", required=True,
                         help="Continuum windows as (lo, hi) pairs, e.g. "
                              "'--cont 2.10 2.14 2.19 2.20' defines two windows.")
    parser.add_argument("--poly-order", type=int, default=1,
                         help="Polynomial order for the continuum fit.")
    parser.add_argument("--sdss-cont", action="store_true",
                         help="Use the SDSS-style 40-60%% mean continuum.")
    parser.add_argument("--weight-type", choices=["uniform", "poisson", "gauss"],
                         default="uniform", help="Weighting scheme for moments/fit.")
    parser.add_argument("--mom-threshold", type=float, default=0.5,
                         help="Moment inclusion threshold, in units of continuum noise.")
    parser.add_argument("--instrument-fwhm", type=float, default=0.0,
                         help="Instrumental FWHM, km/s, convolved into the fitted "
                              "Gaussian model.")
    parser.add_argument("--n-mc", type=int, default=0,
                         help="Number of Monte-Carlo error iterations (0 disables).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for the Monte-Carlo loops.")
    parser.add_argument("--outdir", default="linefit_out",
                         help="Output directory for the FITS maps.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point.

    Parameters
    ----------
    argv
        Argument list to parse (defaults to ``sys.argv[1:]``).
    """
    args = build_arg_parser().parse_args(argv)

    if len(args.cont) % 2 != 0:
        raise SystemExit("--cont must be given as (lo, hi) pairs")
    cont_windows = [(args.cont[i], args.cont[i + 1]) for i in range(0, len(args.cont), 2)]

    line_window = tuple(args.line_window) if args.line_window else \
        (args.line_center - 0.008, args.line_center + 0.008)

    rng = np.random.default_rng(args.seed)

    wave, cube, noise_cube, header = load_cube(
        args.cube, data_ext=args.data_ext, noise_ext=args.noise_ext,
    )
    print(f"Loaded cube of shape {cube.shape} (nx, ny, n_wave)")

    maps = fit_cube(
        wave, cube, noise_cube, line_window, cont_windows, args.line_center,
        poly_order=args.poly_order, sdss_cont=args.sdss_cont,
        weight_type=args.weight_type, mom_threshold=args.mom_threshold,
        instrument_fwhm_kms=args.instrument_fwhm, n_mc=args.n_mc,
        rng=rng, verbose=True,
    )

    save_maps(args.outdir, maps, header=header)
    print(f"Wrote {len(maps)} FITS maps to '{args.outdir}'")


if __name__ == "__main__":
    main()
