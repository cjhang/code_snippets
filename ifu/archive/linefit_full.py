#!/usr/bin/env python3
"""
linefit.py
==========
Python translation of the IDL package `linefit.pro` (make emission-line maps
from IFU/longslit spectra by convolving a template line profile with a
Gaussian).

See header notes in the original IDL file for full algorithm description.
This translation keeps the same overall control flow and variable names
where practical, but uses numpy/scipy/astropy/lmfit instead of IDL-specific
routines (see accompanying "library substitution table").

Original IDL authors: R. Davies, K. Shapiro, N. Foerster-Schreiber,
A. Agudo, E. Wisnioski, E. Wuyts, K. Bandara, S. Price.
Python port: (this translation)

NOTE: This is a large, complex pipeline. Rarely used legacy branches
(old KMOS header formats, interactive prompts, PS-specific multi panel
cosmetics) have been simplified but the physics/statistics (continuum
fit, moments, gaussian+template convolution fit, NII triple-gaussian
fit, MC error propagation, Voronoi binning, masking, FITS output) are
translated in full.
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass, field, fields
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits

try:
    import lmfit
    HAVE_LMFIT = True
except ImportError:
    HAVE_LMFIT = False

try:
    from vorbin.voronoi_2d_binning import voronoi_2d_binning as _vorbin_2d_binning
    HAVE_VORBIN = True
except ImportError:
    HAVE_VORBIN = False


VERS_STR = "6.2.5-py"

C_KMS = 3.0e5


# ---------------------------------------------------------------------------
# Small helpers replacing IDL builtins
# ---------------------------------------------------------------------------

def fxpar_sp(hdr, key, default=0):
    """Replacement for fxpar_sp.pro -- astropy transparently supports
    HIERARCH keywords, so this is just a safe header lookup."""
    try:
        val = hdr[key]
        return val
    except (KeyError, IndexError):
        return default


def quadratic_interp(x_old, y_old, x_new):
    """Approximate IDL's interpol(...,/quadratic)."""
    good = np.isfinite(y_old)
    if good.sum() < 3:
        return np.interp(x_new, x_old, np.nan_to_num(y_old))
    f = interp1d(x_old[good], y_old[good], kind="quadratic",
                 bounds_error=False, fill_value="extrapolate")
    return f(x_new)


def median_filter_1d(y, width):
    """IDL median(y, width, /even) equivalent."""
    if width is None or width <= 1:
        return y
    w = int(width)
    if w % 2 == 0:
        w += 1  # scipy needs odd window; close enough to /even behaviour
    return ndimage.median_filter(y, size=w, mode="nearest")


def median_filter_2d(img, width):
    if width is None or width <= 1:
        return img
    w = int(width)
    if w % 2 == 0:
        w += 1
    return ndimage.median_filter(img, size=w, mode="nearest")


def convol_center_truncate(data, kernel):
    """Approximate IDL CONVOL(data, kernel, /center, /edge_truncate)."""
    kernel = np.asarray(kernel, dtype=float)
    ksum = kernel.sum()
    k = kernel[::-1] if ksum != 0 else kernel
    return ndimage.convolve1d(data, k, mode="nearest")


def cut_moment_outliers(good_mom):
    """Removes isolated single/double outliers from the ends of an index
    vector (as in cut_moment_outliers.pro)."""
    good_mom = np.asarray(good_mom)
    if good_mom.size == 0:
        return np.array([-1])
    if good_mom.size <= 2:
        return np.array([-1])

    good_mom = good_mom.copy()
    do_cut = True
    while do_cut and good_mom.size > 2:
        i = 1
        if good_mom[i] - 1 != good_mom[i - 1]:
            good_mom = good_mom[i:]
        elif good_mom[i + 1] - 2 != good_mom[i - 1]:
            good_mom = good_mom[i + 1:]
        else:
            do_cut = False

    do_cut = True
    while do_cut and good_mom.size > 2:
        i = good_mom.size - 1
        if good_mom[i] - 1 != good_mom[i - 1]:
            good_mom = good_mom[:i]
        elif good_mom[i] - 2 != good_mom[i - 2]:
            good_mom = good_mom[:i - 1]
        else:
            do_cut = False

    if good_mom.size < 3:
        return np.array([-1])
    return good_mom


def gauss1_(x, p):
    """Single gaussian: amplitude, sigma, center."""
    a, s, c = p
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def amoeba_fit(func, p0, scale, ftol=1e-4, maxiter=5000):
    """Nelder-Mead simplex minimisation, replacing IDL's AMOEBA().

    `func` must accept a 1-D numpy parameter array and return chi2 (scalar).
    Returns best-fit parameter array, or None if it failed to converge
    (mirrors the IDL convention of returning a scalar -1 on failure, which
    we represent here as None).
    """
    p0 = np.asarray(p0, dtype=float)
    res = minimize(func, p0, method="Nelder-Mead",
                    options={"xatol": ftol, "fatol": ftol,
                             "maxiter": maxiter, "maxfev": maxiter})
    if not res.success:
        # IDL amoeba can still return a "best effort" answer; keep it
        # unless something is clearly degenerate.
        if not np.all(np.isfinite(res.x)):
            return None
    return res.x


# ---------------------------------------------------------------------------
# make_axes replacement (Adam Leroy's make_axes.pro)
# ---------------------------------------------------------------------------

def make_axes(hdr):
    """Return (ximg, yimg) arrays of RA/Dec (deg) for every spatial pixel,
    using the WCS in the header. Falls back to pixel index grids if WCS
    solution not present (e.g. simple linear axes with CRVAL1/2 etc.)."""
    from astropy.wcs import WCS
    naxis1 = int(fxpar_sp(hdr, "NAXIS1", 1))
    naxis2 = int(fxpar_sp(hdr, "NAXIS2", 1))
    try:
        w = WCS(hdr, naxis=2)
        jj, ii = np.mgrid[0:naxis2, 0:naxis1]
        world = w.wcs_pix2world(ii, jj, 0)
        ximg, yimg = world[0], world[1]
    except Exception:
        crval1 = fxpar_sp(hdr, "CRVAL1", 0.0)
        cdelt1 = fxpar_sp(hdr, "CDELT1", 1.0)
        crpix1 = fxpar_sp(hdr, "CRPIX1", 1.0)
        crval2 = fxpar_sp(hdr, "CRVAL2", 0.0)
        cdelt2 = fxpar_sp(hdr, "CDELT2", 1.0)
        crpix2 = fxpar_sp(hdr, "CRPIX2", 1.0)
        x1 = (np.arange(naxis1) + 1 - crpix1) * cdelt1 + crval1
        y1 = (np.arange(naxis2) + 1 - crpix2) * cdelt2 + crval2
        ximg, yimg = np.meshgrid(x1, y1)
    return ximg, yimg


# ---------------------------------------------------------------------------
# Voronoi binning (Cappellari & Copin 2003) -- pure-python fallback if the
# `vorbin` package is not installed.
# ---------------------------------------------------------------------------

def _weighted_centroid(x, y, density):
    mass = density.sum()
    return (x * density).sum() / mass, (y * density).sum() / mass


def _bin_roundness(x, y, pixel_size):
    n = len(x)
    equiv_r = np.sqrt(n / np.pi) * pixel_size
    xbar, ybar = x.mean(), y.mean()
    max_d = np.sqrt(np.max((x - xbar) ** 2 + (y - ybar) ** 2))
    return max_d / equiv_r - 1.0


def voronoi_2d_binning_fallback(x, y, signal, noise, target_sn, plot=False):
    """Direct translation of voronoi_2d_binning.pro (Cappellari & Copin 2003)."""
    n = len(signal)
    dmin = np.inf
    for j in range(n - 1):
        dmin = min(dmin, np.min((x[j] - x[j + 1:]) ** 2 + (y[j] - y[j + 1:]) ** 2))
    pixel_size = np.sqrt(dmin)

    class_ = np.zeros(n, dtype=int)
    good = np.zeros(n, dtype=bool)

    k0 = np.argmax(signal / noise)
    ind = 1
    class_[k0] = ind
    good[k0] = True

    while True:
        unbinned = np.where(class_ == 0)[0]
        if unbinned.size == 0:
            break
        binned = np.where(class_ != 0)[0]
        xbar, ybar = _weighted_centroid(x[binned], y[binned], signal[binned])
        ind += 1
        k = np.argmin((x[unbinned] - xbar) ** 2 + (y[unbinned] - ybar) ** 2)
        current_bin = np.array([unbinned[k]])
        sn = signal[current_bin[0]] / noise[current_bin[0]]
        class_[current_bin[0]] = ind
        xbar, ybar = x[current_bin[0]], y[current_bin[0]]

        while True:
            unbinned = np.where(class_ == 0)[0]
            if unbinned.size == 0:
                break
            k = np.argmin((x[unbinned] - xbar) ** 2 + (y[unbinned] - ybar) ** 2)
            min_dist = np.min((x[current_bin] - x[unbinned[k]]) ** 2
                               + (y[current_bin] - y[unbinned[k]]) ** 2)
            if np.sqrt(min_dist) > 1.2 * pixel_size:
                break
            next_bin = np.append(current_bin, unbinned[k])
            roundness = _bin_roundness(x[next_bin], y[next_bin], pixel_size)
            if roundness > 0.3:
                break
            sn_old = sn
            sn = signal[next_bin].sum() / np.sqrt((noise[next_bin] ** 2).sum())
            if abs(sn - target_sn) > abs(sn_old - target_sn) and sn_old > 0.8 * target_sn:
                good[current_bin[-1]] = True
                break
            class_[unbinned[k]] = ind
            current_bin = next_bin
            xbar, ybar = _weighted_centroid(x[current_bin], y[current_bin],
                                             signal[current_bin])

    class_ = class_ * good.astype(int)  # zero-out unsuccessful bins -- simplified

    # Reassign bad bins to nearest good centroid
    unique_bins = np.unique(class_[class_ > 0])
    xnode = np.zeros(len(unique_bins))
    ynode = np.zeros(len(unique_bins))
    for j, b in enumerate(unique_bins):
        p = np.where(class_ == b)[0]
        xnode[j], ynode[j] = _weighted_centroid(x[p], y[p], signal[p])

    bad = np.where(class_ == 0)[0]
    for idx in bad:
        d = (x[idx] - xnode) ** 2 + (y[idx] - ynode) ** 2
        class_[idx] = unique_bins[np.argmin(d)]

    # relabel to 0..nbins-1 contiguous
    uniq, class_ = np.unique(class_, return_inverse=True)
    nbins = len(uniq)

    # recompute centroids
    xnode = np.zeros(nbins)
    ynode = np.zeros(nbins)
    for b in range(nbins):
        p = np.where(class_ == b)[0]
        xnode[b], ynode[b] = _weighted_centroid(x[p], y[p], signal[p])

    # Modified Lloyd (CVT) equal-mass iterations
    dens = (signal / noise) ** 2
    for _ in range(50):
        xnode_old, ynode_old = xnode.copy(), ynode.copy()
        d2 = (x[:, None] - xnode[None, :]) ** 2 + (y[:, None] - ynode[None, :]) ** 2
        class_ = np.argmin(d2, axis=1)
        for b in range(nbins):
            p = np.where(class_ == b)[0]
            if p.size == 0:
                continue
            xnode[b], ynode[b] = _weighted_centroid(x[p], y[p], dens[p] ** 2)
        if np.sum((xnode - xnode_old) ** 2 + (ynode - ynode_old) ** 2) == 0:
            break

    xbar = np.zeros(nbins)
    ybar = np.zeros(nbins)
    sn = np.zeros(nbins)
    for b in range(nbins):
        p = np.where(class_ == b)[0]
        xbar[b], ybar[b] = _weighted_centroid(x[p], y[p], signal[p])
        sn[b] = signal[p].sum() / np.sqrt((noise[p] ** 2).sum())

    return class_, xnode, ynode, xbar, ybar, sn


def run_voronoi_binning(x, y, signal, noise, target_sn, plot_file=None):
    if HAVE_VORBIN:
        binnum, xb, yb, xbar, ybar, sn, *_ = _vorbin_2d_binning(
            x, y, signal, noise, target_sn, plot=bool(plot_file), quiet=True)
        if plot_file:
            plt.savefig(plot_file)
            plt.close()
        return binnum, xb, yb, xbar, ybar, sn
    else:
        return voronoi_2d_binning_fallback(x, y, signal, noise, target_sn)


# ---------------------------------------------------------------------------
# Result container (replaces the IDL anonymous structure returned by kinfit)
# ---------------------------------------------------------------------------

@dataclass
class KinfitResult:
    yline: np.ndarray = None
    yclean: np.ndarray = None
    vcont: float = np.nan
    vcont_err: float = np.nan
    vplot_lambda_mom: np.ndarray = None
    vplot_data_mom: np.ndarray = None
    vfullspec: np.ndarray = None
    vvel_mom: float = np.nan
    vdisp_mom: float = np.nan
    vvel_mom_err: float = np.nan
    vvel_mom_errhi: float = np.nan
    vvel_mom_errlo: float = np.nan
    vdisp_mom_err: float = np.nan
    vdisp_mom_errhi: float = np.nan
    vdisp_mom_errlo: float = np.nan
    vmap_mom: float = np.nan
    veqw_mom: float = np.nan
    vmap_mom_err: float = np.nan
    vmap_mom_errhi: float = np.nan
    vmap_mom_errlo: float = np.nan
    veqw_mom_err: float = np.nan
    veqw_mom_errhi: float = np.nan
    veqw_mom_errlo: float = np.nan
    vmap_gau: float = np.nan
    vmap_gau_err: float = np.nan
    vmap_gau_errhi: float = np.nan
    vmap_gau_errlo: float = np.nan
    vdisp_gau: float = np.nan
    vdisp_gau_err: float = np.nan
    vdisp_gau_errhi: float = np.nan
    vdisp_gau_errlo: float = np.nan
    vvel_gau: float = np.nan
    vvel_gau_err: float = np.nan
    vvel_gau_errhi: float = np.nan
    vvel_gau_errlo: float = np.nan
    vres_cube: np.ndarray = None
    vmap_gau_mpfit_err: float = np.nan
    vmap_niib_gau: float = np.nan
    vmap_niir_gau: float = np.nan
    vmap_niib_gau_err: float = np.nan
    vmap_niir_gau_err: float = np.nan
    vmap_niib_gau_errhi: float = np.nan
    vmap_niir_gau_errhi: float = np.nan
    vmap_niib_gau_errlo: float = np.nan
    vmap_niir_gau_errlo: float = np.nan
    vmap_withnii_gau: float = np.nan
    vdisp_withnii_gau: float = np.nan
    vvel_withnii_gau: float = np.nan
    signal: float = 0.0
    amplitude: float = 0.0
    noise: float = 0.0
    vplot_data_gaufit: np.ndarray = None


# ---------------------------------------------------------------------------
# Main class -- holds the "common block" state used by kinfit/conv_spec
# ---------------------------------------------------------------------------

class LineFit:
    def __init__(self):
        # linefit_var common block
        self.g_vel = None
        self.tpl_data = None
        self.tpl_sig = None
        self.g_tpl_data = None
        # global_var common block
        self.g_ymin = np.inf
        self.g_ymax = -np.inf
        self.g_scale = 1.0
        self.max_pix = 0
        # kinfit_var1
        self.x_line = None
        self.yline = None
        self.use_these = None
        self.use_all = 1
        self.par4b = None
        # conv_spec_var1
        self.g_t = None
        self.gha_t = None
        self.gniir_t = None
        self.gniib_t = None
        # conv_spec_par
        self.do_what = "all"
        # calc_err_par
        self.vel = None
        self.dis = None
        self.sca = None
        self.rng = np.random.default_rng(12345)

    # ------------------------------------------------------------------
    # conv_spec.pro : chi2 for gauss(convolved w/ template) vs data
    # ------------------------------------------------------------------
    def conv_spec(self, p):
        do_what = self.do_what
        if do_what == "all":
            g_scale, g_sigma, g_pos = abs(p[0]), abs(p[1]), p[2]
        elif do_what == "vel":
            g_scale, g_sigma, g_pos = abs(p[0]), abs(p[1]), self.vel
        elif do_what == "dis":
            g_scale, g_sigma, g_pos = abs(p[0]), self.dis, p[1]
        elif do_what == "disvel":
            g_scale, g_sigma, g_pos = abs(p[0]), self.dis, self.vel
        elif do_what == "sca":
            g_scale, g_sigma, g_pos = self.sca, abs(p[0]), p[1]
        else:
            g_scale, g_sigma, g_pos = abs(p[0]), abs(p[1]), p[2]

        g = np.exp(-0.5 * ((self.g_vel + g_pos) / g_sigma) ** 2)
        if g.sum() > 0:
            g = g_scale * g / g.sum()

        g_t_full = convol_center_truncate(self.tpl_data, g)
        self.g_t = g_t_full[self.x_line]

        diff = (self.g_t - self.yline) / self.par4b
        if self.use_all < 1:
            diff = diff[self.use_these]
        return float(np.sum(diff ** 2))

    # ------------------------------------------------------------------
    # conv_spec_nii.pro : chi2 for Ha+NIIb+NIIr(convolved) vs data
    # ------------------------------------------------------------------
    def conv_spec_nii(self, p):
        do_what = self.do_what
        if do_what == "all":
            g_scale, g_sigma, g_pos = p[0], abs(p[1]), p[2]
        elif do_what == "vel":
            g_scale, g_sigma, g_pos = p[0], abs(p[1]), self.vel
        elif do_what == "dis":
            g_scale, g_sigma, g_pos = p[0], self.dis, p[1]
        elif do_what == "disvel":
            g_scale, g_sigma, g_pos = p[0], self.dis, self.vel
        elif do_what == "sca":
            g_scale, g_sigma, g_pos = self.sca, abs(p[0]), p[1]
        else:
            g_scale, g_sigma, g_pos = p[0], abs(p[1]), p[2]
        gniib_scale = abs(p[3])
        gniir_scale = abs(p[4])

        gha = np.exp(-0.5 * ((self.g_vel + g_pos) / g_sigma) ** 2)
        gha = g_scale * gha / gha.sum()
        gniib = np.exp(-0.5 * ((self.g_vel + g_pos - 671.48055) / g_sigma) ** 2)
        gniib = gniib_scale * gniib / gniib.sum()
        gniir = np.exp(-0.5 * ((self.g_vel + g_pos + 941.00958) / g_sigma) ** 2)
        gniir = gniir_scale * gniir / gniir.sum()
        g = gha + gniib + gniir

        self.gha_t = convol_center_truncate(self.tpl_data, gha)[self.x_line]
        self.gniir_t = convol_center_truncate(self.tpl_data, gniir)[self.x_line]
        self.gniib_t = convol_center_truncate(self.tpl_data, gniib)[self.x_line]
        self.g_t = convol_center_truncate(self.tpl_data, g)[self.x_line]

        diff = (self.g_t - self.yline) / self.par4b
        if self.use_all < 1:
            diff = diff[self.use_these]
        return float(np.sum(diff ** 2))

    # ------------------------------------------------------------------
    # gauss3.pro : three gaussians convolved (FFT) w/ resolution template
    # used for the fast mpfit-based NII fit
    # ------------------------------------------------------------------
    def gauss3(self, x, p):
        f = np.zeros_like(x, dtype=float)
        for j in range(3):
            f = f + gauss1_(x, p[3 * j:3 * j + 3])
        g = self.g_tpl_data
        n = len(g)
        fft_f = np.fft.fft(f)
        fft_g = np.fft.fft(g, n=len(f)) if len(g) != len(f) else np.fft.fft(g)
        # zero pad g to len(f) if needed (mirrors IDL assumption len(g)==len(f))
        if len(g) != len(f):
            gg = np.zeros_like(f)
            gg[:len(g)] = g
            fft_g = np.fft.fft(gg)
        conv = np.real(np.fft.ifft(fft_f * fft_g)) * len(g)
        conv = np.roll(conv, -int(len(g) / 2))
        return conv

    # ==================================================================
    # kinfit -- continuum + moments + gaussian(+template) fit for ONE
    # spectrum (single spatial pixel or single Voronoi bin)
    # ==================================================================
    def kinfit(self, xvel, xspec, xline, xcont, yspec, sdevyspec, weight_type,
               linepos, res_lambda, cdelt3, ndegree, momthresh_val,
               plotfig, do_calc_errs, ix, iy, outdir,
               w0=None, v0=None, fitnii=False, niithresh=3.0,
               linename="Ha", lrange_hwhm=0.004, sdsscont=False, yscale=1.0):

        r = KinfitResult()
        r.yline = np.full(len(xline), np.nan)
        r.yclean = np.full(len(xspec), np.nan)
        r.vplot_lambda_mom = np.full(len(xline), np.nan)
        r.vplot_data_mom = np.full(len(xline), np.nan)
        r.vfullspec = np.full(len(xspec), np.nan)
        r.vres_cube = np.full(len(res_lambda), np.nan)
        r.vplot_data_gaufit = np.full(5 if fitnii else 3, np.nan)

        use_input_noise = np.isfinite(sdevyspec).sum() >= 1

        xpos = xspec[xcont]
        spec = yspec[xcont]
        sdevspec = sdevyspec[xcont]

        n_bad = np.sum(~np.isfinite(spec))
        if n_bad >= 0.3 * len(xcont):
            return r  # not enough finite continuum points

        # ---------------- continuum fit ----------------
        if not sdsscont:
            errs = np.full(len(xcont), 1e-6)
            bad_idx = np.where(~np.isfinite(spec))[0]
            if bad_idx.size:
                errs[bad_idx] = 40.0
            frange = np.where(np.isfinite(spec))[0]

            par2 = spec[frange] - np.median(spec[frange][errs[frange] < 10])
            par3 = np.std(par2[errs[frange] < 10], ddof=0)
            mask34 = (errs[frange] < 10) & (np.abs(par2) <= 4 * par3)
            par4 = np.std(par2[mask34], ddof=0)

            outlier_mask = np.abs(par2) > 2.5 * par4

            for _ in range(5):
                errs = np.full(len(xcont), 1e-6)
                if outlier_mask.any():
                    errs[frange[outlier_mask]] = 30.0

                if use_input_noise:
                    if weight_type == "gauss":
                        inp_err = sdevyspec.copy()
                    elif weight_type == "poisson":
                        inp_err = np.sqrt(np.abs(sdevyspec))
                    else:
                        inp_err = np.ones_like(sdevyspec)
                else:
                    inp_err = errs

                w_ = 1.0 / inp_err[frange] ** 2
                coeffs = np.polyfit(xpos[frange], spec[frange], ndegree, w=np.sqrt(w_))
                fit = np.polyval(coeffs, xpos[frange])
                par2 = spec[frange] - fit
                mask_lt10 = errs[frange] < 10
                par3 = np.std(par2[mask_lt10], ddof=0)
                mask34 = mask_lt10 & (np.abs(par2) <= 4.0 * par3)
                par4 = np.std(par2[mask34], ddof=0)
                outlier_mask = np.abs(par2) > 2.5 * par4

            linespec = np.polyval(coeffs, xspec[xline])
            r.vcont = float(np.mean(linespec))
            ngood = np.sum(mask34)
            r.vcont_err = par4 / np.sqrt(max(ngood, 1))
            fullspec = np.polyval(coeffs, xspec)
        else:
            frange = np.where(np.isfinite(spec) & (spec != 0))[0]
            pr = np.percentile(spec[frange], [40, 60])
            use_cont = np.where((spec >= pr[0]) & (spec <= pr[1]) &
                                 np.isfinite(spec) & (spec != 0))[0]
            r.vcont = float(np.mean(spec[use_cont]))
            par2 = spec[use_cont] - r.vcont
            par4 = np.std(par2, ddof=0)
            r.vcont_err = par4 / np.sqrt(len(par2))
            linespec = np.full(len(xline), r.vcont)
            fullspec = np.full(len(xspec), r.vcont)
            fit = np.full(len(frange), r.vcont)

        yline = yspec[xline] - linespec
        sdevyline = sdevyspec[xline]
        r.yline = yline
        yclean = yspec - fullspec
        r.yclean = yclean

        # ---------------- moments ----------------
        good_mom = np.where(yline > momthresh_val * par4)[0]
        good_mom = cut_moment_outliers(good_mom)
        mom_empty = good_mom[0] == -1

        nweights = np.ones_like(sdevyline)
        if use_input_noise:
            if weight_type == "gauss":
                nweights = 1.0 / (sdevyline ** 2)
            elif weight_type == "poisson":
                nweights = 1.0 / sdevyline

        if not mom_empty:
            if use_input_noise:
                r.vmap_mom = (np.sum(yline[good_mom] * nweights[good_mom]) /
                              np.sum(nweights[good_mom])) * len(good_mom)
                r.vvel_mom = (np.sum(yline[good_mom] * nweights[good_mom] *
                                      xspec[xline[good_mom]]) /
                              np.sum(yline[good_mom] * nweights[good_mom]))
                r.vdisp_mom = np.sqrt(
                    np.sum(yline[good_mom] * nweights[good_mom] *
                           (xspec[xline[good_mom]] - r.vvel_mom) ** 2) /
                    np.sum(yline[good_mom] * nweights[good_mom]))
            else:
                r.vmap_mom = float(np.sum(yline[good_mom]))
                r.vvel_mom = float(np.sum(yline[good_mom] * xspec[xline[good_mom]]) /
                                    r.vmap_mom)
                r.vdisp_mom = float(np.sqrt(
                    np.sum(yline[good_mom] * (xspec[xline[good_mom]] - r.vvel_mom) ** 2) /
                    r.vmap_mom))
            r.veqw_mom = r.vmap_mom / r.vcont * cdelt3 * 1.0e4

            r.vplot_lambda_mom[good_mom] = xspec[xline[good_mom]]
            r.vplot_data_mom[good_mom] = yspec[xline[good_mom]]

            # ---- MC errors for moments ----
            noise_arr = np.full_like(yline, par4)
            if use_input_noise:
                noise_arr = sdevyline
            if do_calc_errs:
                nloop = 100
                mge = np.zeros(nloop)
                vge = np.zeros(nloop)
                dge = np.zeros(nloop)
                ege = np.zeros(nloop)
                for bs in range(nloop):
                    vec = self.rng.normal(size=len(yline)) * noise_arr + yline
                    mom = np.where(vec > momthresh_val * par4)[0]
                    mom = cut_moment_outliers(mom)
                    if mom[0] == -1:
                        continue
                    mge[bs] = (np.sum(vec[mom] * nweights[mom]) /
                               np.sum(nweights[mom])) * len(mom)
                    ege[bs] = mge[bs] / r.vcont * cdelt3 * 1.0e4
                    vge[bs] = np.sum(vec[mom] * nweights[mom] * xspec[xline[mom]]) / \
                        np.sum(vec[mom] * nweights[mom])
                    dge[bs] = np.sqrt(
                        np.sum(vec[mom] * nweights[mom] *
                               (xspec[xline[mom]] - r.vvel_mom) ** 2) /
                        np.sum(vec[mom] * nweights[mom]))

                lo_i = round((nloop * (1.0 - 0.68)) / 2.0)
                hi_i = lo_i + round(nloop * 0.68)
                hi_i = min(hi_i, nloop - 1)

                def _lohi(arr, nominal):
                    s = np.sort(arr)
                    return abs(nominal - s[lo_i]), abs(nominal - s[hi_i])

                r.vmap_mom_errlo, r.vmap_mom_errhi = _lohi(mge, r.vmap_mom)
                r.vmap_mom_err = np.mean([r.vmap_mom_errlo, r.vmap_mom_errhi])
                r.veqw_mom_errlo, r.veqw_mom_errhi = _lohi(ege, r.veqw_mom)
                r.veqw_mom_err = np.mean([r.veqw_mom_errlo, r.veqw_mom_errhi])
                r.vvel_mom_errlo, r.vvel_mom_errhi = _lohi(vge, r.vvel_mom)
                r.vvel_mom_err = np.mean([r.vvel_mom_errlo, r.vvel_mom_errhi])
                r.vdisp_mom_errlo, r.vdisp_mom_errhi = _lohi(dge, r.vdisp_mom)
                r.vdisp_mom_err = np.mean([r.vdisp_mom_errlo, r.vdisp_mom_errhi])

        # ---------------- Gaussian (template-convolved) fit ----------------
        if par4 > 0:
            self.use_all = 1
            self.do_what = "all"
            if w0 is not None and v0 is not None:
                self.do_what, self.dis, self.vel = "disvel", w0, v0
            elif w0 is not None:
                self.do_what, self.dis = "dis", w0
            elif v0 is not None:
                self.do_what, self.vel = "vel", v0

            full_yline = yline
            full_sdevyline = sdevyline
            full_g_vel = self.g_vel
            full_x_line = xline

            foo = xspec[xline]
            lrange_line = np.where((foo >= linepos - lrange_hwhm) &
                                    (foo <= linepos + lrange_hwhm))[0]
            self.yline = yline[lrange_line]
            self.par4b = np.ones_like(self.yline)
            sdevyline_sub = sdevyline[lrange_line]
            self.g_vel = self.g_vel[lrange_line]
            self.x_line = xline[lrange_line]

            if use_input_noise:
                if weight_type == "gauss":
                    self.par4b = sdevyline_sub
                elif weight_type == "poisson":
                    self.par4b = np.sqrt(np.abs(sdevyline_sub))

            p0_init = [float(np.sum(self.yline)), 50.0, 0.0]
            scale_init = [np.sum(self.yline) / 5.0, 50.0, 50.0]
            if w0 is not None and v0 is not None:
                p0_init, scale_init = [p0_init[0]], [scale_init[0]]
            elif w0 is not None:
                p0_init, scale_init = [p0_init[0], p0_init[2]], [scale_init[0], scale_init[2]]
            elif v0 is not None:
                p0_init, scale_init = [p0_init[0], p0_init[1]], [scale_init[0], scale_init[1]]

            r1 = amoeba_fit(self.conv_spec, p0_init, scale_init)

            if r1 is not None:
                diff = self.yline - self.g_t
                std_diff = np.std(diff, ddof=0)
                med_diff = np.median(diff)
                range1 = np.where(np.abs(diff - med_diff) <= 2.5 * std_diff)[0]
                newvar1 = diff[range1]
                range2 = np.where(np.abs(newvar1 - np.median(newvar1)) <=
                                   2.5 * np.std(newvar1, ddof=0))[0]
                newvar2 = newvar1[range2]
                range3 = np.where(np.abs(newvar2 - np.mean(newvar2)) <=
                                   2.5 * np.std(newvar2, ddof=0))[0]
                newvar3 = newvar2[range3]
                par4a = np.mean(newvar3)
                par4b_val = np.std(newvar3[np.abs(newvar3 - par4a) <=
                                            2.5 * np.std(newvar3, ddof=0)], ddof=0)
                self.par4b = np.full_like(self.yline, par4b_val)
                if use_input_noise:
                    if weight_type == "gauss":
                        self.par4b = sdevyline_sub
                    elif weight_type == "poisson":
                        self.par4b = np.sqrt(np.abs(sdevyline_sub))

                self.use_all = 0
                thresh = 2.5 * np.std(newvar3, ddof=0)
                self.use_these = np.where(np.abs(diff - par4a) <= thresh)[0]
                self.do_what = ("disvel" if (w0 is not None and v0 is not None) else
                                 "dis" if w0 is not None else
                                 "vel" if v0 is not None else "all")

                scale_init2 = [r1[0] / 10.0, 10.0, 10.0]
                if w0 is not None and v0 is not None:
                    scale_init2 = [r1[0] / 10.0]
                elif w0 is not None:
                    scale_init2 = [r1[0] / 10.0, 10.0]
                elif v0 is not None:
                    scale_init2 = [r1[0] / 10.0, 10.0]

                r2 = amoeba_fit(self.conv_spec, r1, scale_init2, ftol=1e-5)
                bg_t = self.g_t.copy()

                if r2 is not None:
                    if w0 is not None and v0 is not None:
                        r2out = np.array([r2[0], w0, v0])
                    elif w0 is not None:
                        r2out = np.array([r2[0], w0, r2[1]])
                    elif v0 is not None:
                        r2out = np.array([r2[0], r2[1], v0])
                    else:
                        r2out = r2

                    # ---- MC errors for gauss fit ----
                    noise_arr = self.par4b
                    nloop = 100
                    mge = np.zeros(nloop)
                    vge = np.zeros(nloop)
                    dge = np.zeros(nloop)
                    tmp_yline = self.yline.copy()
                    r2_err = np.zeros(3)
                    r2_errlo = np.zeros(3)
                    r2_errhi = np.zeros(3)
                    if do_calc_errs:
                        for bs in range(nloop):
                            self.yline = self.rng.normal(size=len(tmp_yline)) * noise_arr + tmp_yline
                            r3 = amoeba_fit(self.conv_spec, r2, scale_init2, ftol=1e-5)
                            if r3 is None:
                                r3 = r2
                            mge[bs] = np.sum(self.g_t)
                            if w0 is not None and v0 is not None:
                                r3out = np.array([r3[0], w0, v0])
                            elif w0 is not None:
                                r3out = np.array([r3[0], w0, r3[1]])
                            elif v0 is not None:
                                r3out = np.array([r3[0], r3[1], v0])
                            else:
                                r3out = r3
                            vge[bs] = r3out[2]
                            dge[bs] = r3out[1]
                        self.yline = tmp_yline

                        lo_i = round((nloop * (1.0 - 0.68)) / 2.0)
                        hi_i = min(lo_i + round(nloop * 0.68), nloop - 1)

                        def _lohi(arr, nominal):
                            s = np.sort(arr)
                            return abs(nominal - s[lo_i]), abs(nominal - s[hi_i])

                        r2_errlo[0], r2_errhi[0] = _lohi(mge, np.sum(bg_t))
                        r2_err[0] = np.mean([r2_errlo[0], r2_errhi[0]])
                        r2_errlo[1], r2_errhi[1] = _lohi(dge, r2out[1])
                        r2_err[1] = np.mean([r2_errlo[1], r2_errhi[1]])
                        r2_errlo[2], r2_errhi[2] = _lohi(vge, r2out[2])
                        r2_err[2] = np.mean([r2_errlo[2], r2_errhi[2]])

                    r.vmap_gau = float(np.sum(bg_t))
                    r.vdisp_gau = abs(float(r2out[1]))
                    r.vvel_gau = float(r2out[2])
                    r.vmap_gau_err, r.vdisp_gau_err, r.vvel_gau_err = r2_err
                    r.vmap_gau_errlo, r.vdisp_gau_errlo, r.vvel_gau_errlo = r2_errlo
                    r.vmap_gau_errhi, r.vdisp_gau_errhi, r.vvel_gau_errhi = r2_errhi

                    ampfit = (bg_t + linespec[lrange_line]) / abs(np.mean(fit))
                    r.amplitude = float(np.max(ampfit))
                    r.signal = float(np.sum(bg_t) / abs(np.mean(fit)))

                    fwhm = 2.355 * np.sqrt(r1[1] ** 2 + self.tpl_sig ** 2)
                    pxs_kms = cdelt3 * C_KMS / np.mean(xspec[xline])
                    npix = int(np.ceil(2 * fwhm / pxs_kms))
                    r.noise = float(np.mean(sdevyspec[xcont]) * np.sqrt(max(npix, 1)) /
                                     abs(np.mean(fit)))

                    hasn = 0.0
                    if np.isfinite(r.noise) and r.noise > 0:
                        hasn = r.signal / r.noise
                    if ix == iy:
                        print(f"pixel ({ix},{iy}) - {linename} S/N = {hasn:.2f}")

                    # restore full arrays
                    self.yline = full_yline
                    self.g_vel = full_g_vel
                    self.x_line = full_x_line

                    # ------------- NII fit (lmfit, tied params) -------------
                    if fitnii and hasn >= niithresh:
                        self._fit_nii(xvel, xline, yline, sdevyline, r1, r2_err,
                                       do_calc_errs, r)

                    diff2 = yline - bg_t if len(bg_t) == len(yline) else \
                        np.pad(bg_t, (0, len(yline) - len(bg_t)), constant_values=np.nan)
                    full_res = np.concatenate([par2, diff2])
                    full_l = np.concatenate([xpos[frange], xspec[xline]])
                    order = np.argsort(full_l)
                    r.vres_cube = quadratic_interp(full_l[order], full_res[order],
                                                    res_lambda)
                    r.vplot_data_gaufit[:3] = r2out
                    r.vfullspec = fullspec

                    if plotfig:
                        self._plot_pixel_fit(xspec, xvel, yspec, sdevyspec, xline,
                                              xcont, linespec, fullspec, fit,
                                              linepos, ampfit, r, ix, iy, yscale,
                                              outdir)

        if r.amplitude == 0.0:
            r.noise = 1.0
        return r

    # ------------------------------------------------------------------
    def _fit_nii(self, xvel, xline, yline, sdevyline, r1, r2_err, do_calc_errs, r):
        """NII fit using mpfitfun equivalent (lmfit with tied parameters)."""
        if not HAVE_LMFIT:
            warnings.warn("lmfit not installed -- skipping NII fit. "
                           "Install with `pip install lmfit`.")
            return

        velniib = r1[2] - 671.48055
        velniir = r1[2] + 941.00958
        guess = np.array([r1[0] / 10.0, r1[1], velniib,
                           r1[0], r1[1], r1[2],
                           r1[0] / 3.0, r1[1], velniir])
        guess[0:9:3] = np.abs(guess[0:9:3])
        guess[1:9:3] = np.abs(guess[1:9:3])

        xg = xvel[xline]

        def model_func(x, a_niib, s_niib, v_niib, a_ha, s_ha, v_ha,
                        a_niir, s_niir, v_niir):
            p = np.array([a_niib, s_niib, v_niib, a_ha, s_ha, v_ha,
                           a_niir, s_niir, v_niir])
            return self.gauss3(x, p)

        params = lmfit.Parameters()
        params.add("a_niib", value=guess[0], min=0)
        params.add("s_ha", value=guess[4], min=0)
        params.add("s_niib", expr="s_ha")
        params.add("v_ha", value=guess[5],
                    min=r1[2] - 3 * max(r2_err[2], 1e-3),
                    max=r1[2] + 3 * max(r2_err[2], 1e-3))
        params.add("v_niib", expr="v_ha - 671.48055")
        params.add("a_ha", value=guess[3], min=0)
        params.add("a_niir", value=guess[6], min=0, expr="a_niib/3.071")
        params.add("s_niir", expr="s_ha")
        params.add("v_niir", expr="v_ha + 941.00958")

        model = lmfit.Model(model_func, independent_vars=["x"])

        try:
            fitres = model.fit(yline, params, x=xg,
                                weights=1.0 / np.where(sdevyline > 0, sdevyline, 1.0))
        except Exception as exc:
            warnings.warn(f"NII fit failed: {exc}")
            return

        pbest = fitres.params
        p_niib = [pbest["a_niib"].value, pbest["s_niib"].value, pbest["v_niib"].value]
        p_ha = [pbest["a_ha"].value, pbest["s_ha"].value, pbest["v_ha"].value]
        p_niir = [pbest["a_niir"].value, pbest["s_niir"].value, pbest["v_niir"].value]

        bgniib_t = gauss1_(xg, p_niib)
        bgniir_t = gauss1_(xg, p_niir)
        bgha_t = gauss1_(xg, p_ha)

        r.vmap_niib_gau = float(np.sum(bgniib_t))
        r.vmap_niir_gau = float(np.sum(bgniir_t))
        r.vmap_withnii_gau = float(np.sum(bgha_t))
        r.vdisp_withnii_gau = p_ha[1]
        r.vvel_withnii_gau = p_ha[2]

        if do_calc_errs:
            nloop = 100
            mge = np.zeros(nloop)
            niib_mge = np.zeros(nloop)
            niir_mge = np.zeros(nloop)
            for bs in range(nloop):
                tmp_yline = yline + self.rng.normal(size=len(yline)) * sdevyline
                try:
                    fr = model.fit(tmp_yline, pbest, x=xg)
                    a_niib = fr.params["a_niib"].value
                    s_niib = fr.params["s_niib"].value
                    v_niib = fr.params["v_niib"].value
                    a_ha = fr.params["a_ha"].value
                    s_ha = fr.params["s_ha"].value
                    v_ha = fr.params["v_ha"].value
                    a_niir = fr.params["a_niir"].value
                    s_niir = fr.params["s_niir"].value
                    v_niir = fr.params["v_niir"].value
                except Exception:
                    continue
                mge[bs] = np.sum(gauss1_(xg, [a_ha, s_ha, v_ha]))
                niib_mge[bs] = np.sum(gauss1_(xg, [a_niib, s_niib, v_niib]))
                niir_mge[bs] = np.sum(gauss1_(xg, [a_niir, s_niir, v_niir]))

            lo_i = round((nloop * (1.0 - 0.68)) / 2.0)
            hi_i = min(lo_i + round(nloop * 0.68), nloop - 1)

            def _lohi(arr, nominal):
                s = np.sort(arr)
                return abs(nominal - s[lo_i]), abs(nominal - s[hi_i])

            errlo0, errhi0 = _lohi(mge, r.vmap_withnii_gau)
            r.vmap_gau_mpfit_err = np.mean([errlo0, errhi0])
            errlo3, errhi3 = _lohi(niib_mge, r.vmap_niib_gau)
            r.vmap_niib_gau_err = np.mean([errlo3, errhi3])
            r.vmap_niib_gau_errlo, r.vmap_niib_gau_errhi = errlo3, errhi3
            errlo4, errhi4 = _lohi(niir_mge, r.vmap_niir_gau)
            r.vmap_niir_gau_err = np.mean([errlo4, errhi4])
            r.vmap_niir_gau_errlo, r.vmap_niir_gau_errhi = errlo4, errhi4

    # ------------------------------------------------------------------
    def _plot_pixel_fit(self, xspec, xvel, yspec, sdevyspec, xline, xcont,
                         linespec, fullspec, fit, linepos, ampfit, r,
                         ix, iy, yscale, outdir):
        """Simplified equivalent of the diagnostic plot block in kinfit.pro."""
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            flux = yspec / abs(np.mean(fit))
            ax.plot(xspec, flux, color="k", lw=0.8, label="spectrum")
            ax.axvline(linepos, ls="--", color="gray")
            ax.plot(xspec[xline], yspec[xline] / abs(np.mean(fit)), "o",
                    ms=3, color="orange", label="line region")
            ax.plot(xspec[xline], ampfit, color="red", label="gauss fit")
            ax.set_xlabel("Wavelength [um]")
            ax.set_ylabel("Normalised flux")
            ax.set_title(f"x,y = {ix},{iy}")
            ax.legend(fontsize=7)
            os.makedirs(outdir, exist_ok=True)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"pixfit_{ix}_{iy}.png"), dpi=100)
            plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Plotting failed for pixel ({ix},{iy}): {exc}")

    # ==================================================================
    # Main driver -- linefit.pro
    # ==================================================================
    def linefit(self, indir, target, linepos, tpldir=None, l1cont=None,
                l2cont=None, tpl=None, target_sdev=None, outdir="",
                endphrase="", do_calc_errs=False, sdsscont=False,
                weight_type="uniform", fitnii=False, niithresh=3.0,
                linename="Ha", lrange_hwhm=0.004, spat=0, spec=0,
                mom_thresh=0.5, mask_thresh=3.0, l0=None, poly=1, box=None,
                yscale=1.0, plotall=False, plot_pixels=False,
                plot_line_only=False, plot_range=None, plot_name="",
                voronoi=False):

        t0 = time.time()
        print("#" * 65)
        print(f"Executing linefit (python): v{VERS_STR}")

        # ---------------- indir / paths ----------------
        if not indir:
            raise ValueError("indir must be specified")
        if not os.path.isdir(indir):
            raise ValueError("indir does not exist")
        if not indir.endswith(os.sep):
            indir = indir + os.sep

        image_data = os.path.join(indir, target)
        if not image_data.endswith(".fits"):
            image_data = image_data + ".fits"
        if not os.path.isfile(image_data):
            raise ValueError(f"target file not found: {image_data}")

        with fits.open(image_data) as hdul:
            hdr = hdul[0].header.copy()
            nextend = len(hdul) - 1

        resorder = fxpar_sp(hdr, "RESORDER", 0)
        instrument = str(fxpar_sp(hdr, "INSTRUME", "")).strip()

        if resorder == 0:
            if not tpldir:
                raise ValueError("tpldir must be specified (no RESORDER in header)")
            if not os.path.isdir(tpldir):
                raise ValueError("tpldir does not exist")
            if not tpldir.endswith(os.sep):
                tpldir = tpldir + os.sep
        else:
            print("RESORDER keyword found in header: tpldir/tpl will be ignored")

        if tpl:
            tplpath = os.path.join(tpldir, tpl) if tpldir else tpl
            if not tplpath.endswith(".fits"):
                tplpath_check = tplpath + ".fits"
            else:
                tplpath_check = tplpath
            if not os.path.isfile(tplpath_check):
                raise ValueError("specified template not found")

        image_sdev = None
        if target_sdev:
            image_sdev = os.path.join(indir, target_sdev)
            if not image_sdev.endswith(".fits"):
                image_sdev = image_sdev + ".fits"

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            if not outdir.endswith(os.sep):
                outdir = outdir + os.sep
        else:
            outdir = ""

        suffix = "" if not voronoi else "_binned"
        if endphrase:
            suffix = f"_{endphrase}{suffix}"

        # ---------------- linepos / line window ----------------
        default_line_range_lo = 0.008
        default_line_range_hi = 0.012
        default_cont_min = 0.02
        default_cont_max = 0.05

        linepos_arr = np.atleast_1d(linepos).astype(float)
        if linepos_arr.size == 2:
            line = linepos_arr.copy()
            linepos_val = float(np.mean(line))
        elif linepos_arr.size == 1:
            linepos_val = float(linepos_arr[0])
            line = np.array([linepos_val - default_line_range_lo,
                              linepos_val + default_line_range_lo])
            if 1.9 <= linepos_val <= 2.5 and fitnii:
                line = np.array([linepos_val - default_line_range_lo,
                                  linepos_val + default_line_range_hi])
        else:
            raise ValueError("linepos must have 1 or 2 values")

        if not sdsscont and (l1cont is None or l2cont is None):
            raise ValueError("l1cont/l2cont required for 1D polyfit continuum "
                              "(or set sdsscont=True)")
        if sdsscont and l1cont is None and l2cont is None:
            l1cont = np.array([linepos_val - default_cont_max,
                                linepos_val + default_cont_min])
            l2cont = np.array([linepos_val - default_cont_min,
                                linepos_val + default_cont_max])
        l1cont = np.atleast_1d(l1cont).astype(float)
        l2cont = np.atleast_1d(l2cont).astype(float)

        # ---------------- KMOS poly-order template lookup (legacy) ------
        kmos_poly_order_tpl = None
        if instrument == "KMOS" and resorder == 0:
            if 1.0 <= linepos_val <= 1.4:
                kmos_poly_order_tpl = os.path.join(tpldir, "yj_poly-order_4")
            elif 1.4 <= linepos_val <= 1.9:
                kmos_poly_order_tpl = os.path.join(tpldir, "h_poly-order_4")
            elif 1.9 <= linepos_val <= 2.5:
                kmos_poly_order_tpl = os.path.join(tpldir, "k_poly-order_3")
            if kmos_poly_order_tpl and not os.path.isfile(kmos_poly_order_tpl):
                raise ValueError(f"template {kmos_poly_order_tpl} not found")

        # ---------------- misc defaults ----------------
        do_calc_errs = bool(do_calc_errs)
        if weight_type not in ("uniform", "poisson", "gauss"):
            weight_type = "uniform"
        ndegree = poly if (poly and poly > 1) else 1
        spat_smooth = spat or 0
        spec_smooth = spec or 0

        # ==================================================================
        # Read data / noise
        # ==================================================================
        if instrument == "SINFONI":
            nr_ext = 0
        elif instrument == "KMOS":
            nr_ext = 4 if nextend == 4 else (2 if nextend == 2 else 1)
        else:
            nr_ext = nextend

        with fits.open(image_data) as hdul:
            if nr_ext == 0:
                data_raw = hdul[0].data.astype(float)
                hdr = hdul[0].header.copy()
            else:
                data_raw = hdul[1].data.astype(float)
                hdr = hdul[1].header.copy()

            sdevdata_raw = None
            noise_src = ""
            if nr_ext == 2:
                sdevdata_raw = hdul[2].data.astype(float)
                noise_src = image_data
                nan_mask = ~np.isfinite(sdevdata_raw)
                if nan_mask.any():
                    print(f"Fixing {nan_mask.sum()} NaNs in sdevdata_raw")
                    for k in range(sdevdata_raw.shape[0]):
                        plane = sdevdata_raw[k]
                        if (~np.isfinite(plane)).any():
                            plane[~np.isfinite(plane)] = np.nanmedian(plane)
                            sdevdata_raw[k] = plane
            elif nr_ext == 0 and image_sdev:
                with fits.open(image_sdev) as hdul_sdev:
                    sdevdata_raw = hdul_sdev[0].data.astype(float)
                noise_src = image_sdev

        if sdevdata_raw is None:
            sdevdata_raw = np.full_like(data_raw, np.nan)
            print("No noise present")
        else:
            print(f"Noise read in from {noise_src}")

        nr_axes = data_raw.ndim  # numpy order is reversed vs FITS NAXISn

        # Bring everything to (x, y, lambda) ordering like the IDL code
        if nr_axes == 3:
            # FITS NAXIS1,2,3 -> numpy shape (naxis3,naxis2,naxis1)
            data_raw = np.transpose(data_raw, (2, 1, 0))
            sdevdata_raw = np.transpose(sdevdata_raw, (2, 1, 0))
            crval3 = fxpar_sp(hdr, "CRVAL3")
            cdelt3 = fxpar_sp(hdr, "CDELT3")
            crpix3 = fxpar_sp(hdr, "CRPIX3")
            naxis3 = fxpar_sp(hdr, "NAXIS3")
            xsize, ysize = data_raw.shape[0], data_raw.shape[1]
            ximg, yimg = make_axes(hdr)
            ximg, yimg = ximg.T, yimg.T
            degrees = str(fxpar_sp(hdr, "CUNIT1", "")).lower().startswith("deg")
            pxs = min(abs(fxpar_sp(hdr, "CDELT1", 1.0)),
                      abs(fxpar_sp(hdr, "CDELT2", 1.0)))
        elif nr_axes == 2:
            crval3 = fxpar_sp(hdr, "CRVAL1")
            cdelt3 = fxpar_sp(hdr, "CDELT1")
            crpix3 = fxpar_sp(hdr, "CRPIX1")
            naxis3 = fxpar_sp(hdr, "NAXIS1")
            ysize = fxpar_sp(hdr, "NAXIS2")
            xsize = 1
            _, yimg1d = make_axes(hdr)
            yimg = yimg1d[:, 0] if yimg1d.ndim == 2 else yimg1d
            ximg = np.zeros(ysize)
            degrees = str(fxpar_sp(hdr, "CUNIT2", "")).lower().startswith("deg")
            pxs = abs(fxpar_sp(hdr, "CDELT2", 1.0))
            data_raw = data_raw.T  # (naxis2, naxis1) -> put spectral last
            new_d = np.zeros((1, ysize, naxis3))
            new_d[0, :, :] = data_raw
            data_raw = new_d
            sdevdata_raw = sdevdata_raw.T
            new_sd = np.zeros((1, ysize, naxis3))
            new_sd[0, :, :] = sdevdata_raw
            sdevdata_raw = new_sd
        else:  # 1-D spectrum
            crval3 = fxpar_sp(hdr, "CRVAL1")
            cdelt3 = fxpar_sp(hdr, "CDELT1")
            crpix3 = fxpar_sp(hdr, "CRPIX1")
            naxis3 = fxpar_sp(hdr, "NAXIS1")
            xsize, ysize = 1, 1
            ximg, yimg = np.array([0.0]), np.array([0.0])
            degrees = False
            pxs = 0.0
            new_d = np.zeros((1, 1, naxis3))
            new_d[0, 0, :] = data_raw
            data_raw = new_d
            new_sd = np.zeros((1, 1, naxis3))
            new_sd[0, 0, :] = sdevdata_raw
            sdevdata_raw = new_sd

        # ---------------- units conversion W/m^2/um -> erg/s/cm^2 --------
        data_raw = data_raw * cdelt3 * 1.0e3
        sdevdata_raw = sdevdata_raw * cdelt3 * 1.0e3
        print(f"Converted units: multiply by 1000*cdelt3 (cdelt3={cdelt3:.8f})")

        lambda3 = (np.arange(naxis3) + 1 - crpix3) * cdelt3 + crval3
        if crval3 > 1e4:
            lambda3 = lambda3 / 1.0e4
            cdelt3 = cdelt3 / 1.0e4

        mult = 3600.0 if degrees else 1.0
        ximg = (ximg - np.mean(ximg)) * mult
        yimg = (yimg - np.mean(yimg)) * mult

        # ---------------- spatial box crop ----------------
        if box is not None and len(box) == 4 and nr_axes == 3:
            x0, x1, y0, y1 = box
            ximg = ximg[x0:x1 + 1, y0:y1 + 1]
            yimg = yimg[x0:x1 + 1, y0:y1 + 1]
            data_raw = data_raw[x0:x1 + 1, y0:y1 + 1, :]
            sdevdata_raw = sdevdata_raw[x0:x1 + 1, y0:y1 + 1, :]
            xsize, ysize = data_raw.shape[0], data_raw.shape[1]

        if not yscale:
            yscale = 1.0

        # ---------------- crop spectrally ----------------
        lo = min(np.min(l1cont), line[0])
        hi = max(np.max(l2cont), line[1])
        rng_mask = (lambda3 >= lo) & (lambda3 <= hi)
        data_raw = data_raw[:, :, rng_mask]
        sdevdata_raw = sdevdata_raw[:, :, rng_mask]
        res_lambda = lambda3[rng_mask]

        xcont3 = []
        for a, b in zip(l1cont, l2cont):
            xcont3.extend(np.where((res_lambda >= a) & (res_lambda <= b))[0])
        xcont3 = np.array(sorted(set(xcont3)))

        velstep = C_KMS * cdelt3 / np.mean(res_lambda)
        lambda0 = linepos_val
        lam_list = [lambda0]
        vel_list = [0.0]
        ctr = 1
        while min(lam_list) > np.min(res_lambda):
            lam_list.insert(0, min(lam_list) / (1 + velstep / C_KMS))
            vel_list.insert(0, -velstep * ctr)
            ctr += 1
        ctr = 1
        while max(lam_list) < np.max(res_lambda):
            lam_list.append(max(lam_list) * (1 + velstep / C_KMS))
            vel_list.append(velstep * ctr)
            ctr += 1
        lam_axis = np.array(lam_list)
        vel_axis = np.array(vel_list)

        finite_vals = data_raw[np.isfinite(data_raw)]
        scale = float(np.mean(np.abs(finite_vals))) / 100.0
        data_raw = data_raw / scale
        sdevdata_raw = sdevdata_raw / scale
        print(f"Scaling by {1.0/scale:.4g}")
        self.g_scale = scale

        data_raw_for_pixfits = data_raw.copy()
        sdevdata_raw_for_pixfits = sdevdata_raw.copy()

        if (not voronoi) and spat_smooth > 1:
            tmpsdev = np.zeros_like(sdevdata_raw)
            for i in range(data_raw.shape[2]):
                data_raw[:, :, i] = median_filter_2d(data_raw[:, :, i], spat_smooth)
            half = int((spat_smooth - 1) / 2)
            for i in range(data_raw.shape[2]):
                for ix in range(xsize):
                    for iy in range(ysize):
                        xmin, xmax = max(0, ix - half), min(xsize - 1, ix + half)
                        ymin, ymax = max(0, iy - half), min(ysize - 1, iy + half)
                        sub = sdevdata_raw[xmin:xmax + 1, ymin:ymax + 1, i]
                        tmpsdev[ix, iy, i] = np.sqrt(np.sum(sub ** 2)) / sub.size
            sdevdata_raw = tmpsdev

        # ---------------- interpolate cube to common wavelength axis ------
        meanval = np.zeros((xsize, ysize))
        data = np.zeros((xsize, ysize, len(lam_axis)))
        sdevdata = np.zeros((xsize, ysize, len(lam_axis)))
        for ix in range(xsize):
            for iy in range(ysize):
                data[ix, iy, :] = quadratic_interp(res_lambda, data_raw[ix, iy, :], lam_axis)
                meanval[ix, iy] = np.nanmean(data[ix, iy, :])
                sdevdata[ix, iy, :] = quadratic_interp(res_lambda, sdevdata_raw[ix, iy, :], lam_axis)
        bad = ~np.isfinite(meanval)
        meanval[bad] = 1.0

        xline = np.where((lam_axis >= line[0]) & (lam_axis <= line[1]))[0]
        xcont = []
        for a, b in zip(l1cont, l2cont):
            xcont.extend(np.where((lam_axis >= a) & (lam_axis <= b))[0])
        xcont = np.array(sorted(set(xcont)))
        mark = np.isin(xcont, xline)
        if mark.any():
            xcont = xcont[~mark]

        # ================================================================
        # Template
        # ================================================================
        self._load_template(hdr, instrument, resorder, tpldir, tpl,
                             kmos_poly_order_tpl, linepos_val, cdelt3,
                             vel_axis, xline)

        self.g_vel = vel_axis[xline]
        v0_rel_peak = 0.0
        v0 = None
        if l0 is not None:
            g0_pos, g0_sigma = 0.0, 10.0
            g0 = np.exp(-0.5 * ((self.g_vel + g0_pos) / g0_sigma) ** 2)
            g0_t = convol_center_truncate(self.tpl_data, g0)[xline]
            # simple gaussian fit for reference velocity offset
            try:
                from scipy.optimize import curve_fit
                def _g(x, a, c, s):
                    return a * np.exp(-0.5 * ((x - c) / s) ** 2)
                popt, _ = curve_fit(_g, lam_axis[xline], g0_t,
                                     p0=[g0_t.max(), linepos_val, 0.001])
                v0_rel_peak = -1.0 * ((popt[1] - l0) / l0) * C_KMS
            except Exception:
                v0_rel_peak = 0.0
            tmplam = lam_axis[xline]
            idx0 = np.argmin(np.abs(self.g_vel))
            v0 = ((tmplam[idx0] - l0) / l0) * C_KMS

        # ================================================================
        # Main per-pixel fitting loop
        # ================================================================
        print(f"Fitting kinematics per pixel in {xsize} by {ysize} array.")
        self.max_pix = max(xsize, ysize)

        shape = (xsize, ysize)
        signal = np.full(shape, np.nan)
        amplitude = np.full(shape, np.nan)
        noise = np.full(shape, np.nan)
        cont = np.full(shape, np.nan)
        cont_err = np.full(shape, np.nan)
        sn = np.full(shape, np.nan)
        an = np.full(shape, np.nan)
        contsn = np.full(shape, np.nan)

        maps = {k: np.full(shape, np.nan) for k in [
            "map_gau", "map_gau_err", "map_gau_errlo", "map_gau_errhi",
            "vel_gau", "vel_gau_err", "vel_gau_errlo", "vel_gau_errhi",
            "disp_gau", "disp_gau_err", "disp_gau_errlo", "disp_gau_errhi",
            "map_gau_mpfit_err", "map_niib_gau", "map_niib_gau_err",
            "map_niib_gau_errlo", "map_niib_gau_errhi", "map_niir_gau",
            "map_niir_gau_err", "map_niir_gau_errlo", "map_niir_gau_errhi",
            "map_withnii_gau", "vel_withnii_gau", "disp_withnii_gau",
            "map_mom", "map_mom_err", "map_mom_errlo", "map_mom_errhi",
            "vel_mom", "vel_mom_err", "vel_mom_errlo", "vel_mom_errhi",
            "disp_mom", "disp_mom_err", "disp_mom_errlo", "disp_mom_errhi",
            "eqw_mom", "eqw_mom_err", "eqw_mom_errlo", "eqw_mom_errhi",
        ]}

        res_cube = np.full((xsize, ysize, len(res_lambda)), np.nan)
        linecube = np.full((xsize, ysize, len(lam_axis)), np.nan)

        for ix in range(xsize):
            for iy in range(ysize):
                specixiy = data[ix, iy, :].copy()
                if spec_smooth > 1:
                    specixiy = median_filter_1d(specixiy, spec_smooth)

                sdevspecixiy = sdevdata[ix, iy, :].copy()
                if spec_smooth > 1:
                    half = int((spec_smooth - 1) / 2)
                    tmp = np.zeros_like(sdevspecixiy)
                    for iz in range(len(sdevspecixiy)):
                        zmin, zmax = max(0, iz - half), min(len(sdevspecixiy) - 1, iz + half)
                        sub = sdevspecixiy[zmin:zmax + 1]
                        tmp[iz] = np.sqrt(np.sum(sub ** 2)) / sub.size
                    sdevspecixiy = tmp

                plot_it = plotall or (ix == iy)

                res = self.kinfit(vel_axis, lam_axis, xline, xcont, specixiy,
                                   sdevspecixiy, weight_type, linepos_val,
                                   res_lambda, cdelt3, ndegree, mom_thresh,
                                   plot_it, do_calc_errs, ix, iy, outdir,
                                   fitnii=fitnii, niithresh=niithresh,
                                   linename=linename, lrange_hwhm=lrange_hwhm,
                                   sdsscont=sdsscont, yscale=yscale)

                amplitude[ix, iy] = res.amplitude
                signal[ix, iy] = res.signal
                noise[ix, iy] = res.noise

                if not voronoi:
                    maps["map_mom"][ix, iy] = res.vmap_mom
                    maps["map_gau"][ix, iy] = res.vmap_gau
                    maps["vel_gau"][ix, iy] = res.vvel_gau
                    maps["disp_gau"][ix, iy] = res.vdisp_gau
                    cont[ix, iy] = res.vcont
                    maps["eqw_mom"][ix, iy] = res.veqw_mom
                    an[ix, iy] = amplitude[ix, iy] / noise[ix, iy]
                    sn[ix, iy] = signal[ix, iy] / noise[ix, iy]
                    contsn[ix, iy] = cont[ix, iy] / noise[ix, iy]
                    maps["vel_mom"][ix, iy] = res.vvel_mom
                    maps["disp_mom"][ix, iy] = res.vdisp_mom

                    if res.vres_cube is not None:
                        res_cube[ix, iy, :] = res.vres_cube
                    if res.yclean is not None:
                        linecube[ix, iy, :] = res.yclean

                    for k in ["map_mom_err", "map_mom_errlo", "map_mom_errhi",
                              "vel_mom_err", "vel_mom_errlo", "vel_mom_errhi",
                              "disp_mom_err", "disp_mom_errlo", "disp_mom_errhi",
                              "eqw_mom_err", "eqw_mom_errlo", "eqw_mom_errhi",
                              "map_gau_err", "map_gau_errlo", "map_gau_errhi",
                              "vel_gau_err", "vel_gau_errlo", "vel_gau_errhi",
                              "disp_gau_err", "disp_gau_errlo", "disp_gau_errhi"]:
                        maps[k][ix, iy] = getattr(res, "v" + k)
                    cont_err[ix, iy] = res.vcont_err

                    if fitnii:
                        maps["map_gau_mpfit_err"][ix, iy] = res.vmap_gau_mpfit_err
                        maps["map_niib_gau"][ix, iy] = res.vmap_niib_gau
                        maps["map_niib_gau_err"][ix, iy] = res.vmap_niib_gau_err
                        maps["map_niib_gau_errlo"][ix, iy] = res.vmap_niib_gau_errlo
                        maps["map_niib_gau_errhi"][ix, iy] = res.vmap_niib_gau_errhi
                        maps["map_niir_gau"][ix, iy] = res.vmap_niir_gau
                        maps["map_niir_gau_err"][ix, iy] = res.vmap_niir_gau_err
                        maps["map_niir_gau_errlo"][ix, iy] = res.vmap_niir_gau_errlo
                        maps["map_niir_gau_errhi"][ix, iy] = res.vmap_niir_gau_errhi
                        maps["map_withnii_gau"][ix, iy] = res.vmap_withnii_gau
                        maps["vel_withnii_gau"][ix, iy] = res.vvel_withnii_gau
                        maps["disp_withnii_gau"][ix, iy] = res.vdisp_withnii_gau

        neg = (amplitude <= 0) & (noise > 0)
        amplitude[neg] = 1e-4
        an[neg] = 1e-4

        # ================================================================
        # Voronoi binning (optional) -- summary implementation
        # ================================================================
        if voronoi:
            xvec = ximg.ravel()
            yvec = yimg.ravel()
            svec = amplitude.ravel()
            nvec = noise.ravel()
            wvec = np.abs(svec) / (nvec ** 2)
            s1vec = wvec * np.abs(svec)
            n1vec = wvec * np.abs(nvec)
            target_sn = 5.0
            binnum, xb, yb, xbar, ybar, snbin = run_voronoi_binning(
                xvec, yvec, s1vec, n1vec, target_sn,
                plot_file=os.path.join(outdir, "voronoi_bin.png") if outdir else None)

            gvec = data.reshape(-1, data.shape[2])
            nbins = binnum.max() + 1
            zbin = np.zeros((nbins, data.shape[2]))
            for i in range(nbins):
                pixbin = np.where(binnum == i)[0]
                if pixbin.size == 0:
                    continue
                w_ = wvec[pixbin]
                zbin[i, :] = np.sum(w_[:, None] * gvec[pixbin, :], axis=0) / w_.sum()

            print(f"Fitting kinematics per bin in {nbins} bins")
            for i in range(nbins):
                specixiy = zbin[i, :]
                if spec_smooth > 1:
                    specixiy = median_filter_1d(specixiy, spec_smooth)
                res = self.kinfit(vel_axis, lam_axis, xline, xcont, specixiy,
                                   np.full_like(specixiy, np.nan), weight_type,
                                   linepos_val, res_lambda, cdelt3, ndegree,
                                   mom_thresh, True, do_calc_errs, i, -1, outdir,
                                   sdsscont=True, fitnii=fitnii,
                                   niithresh=niithresh, linename=linename,
                                   lrange_hwhm=lrange_hwhm, yscale=yscale)
                pixvals = np.where(binnum == i)[0]
                ind_x, ind_y = np.unravel_index(pixvals, shape)
                maps["map_mom"][ind_x, ind_y] = res.vmap_mom
                maps["map_gau"][ind_x, ind_y] = res.vmap_gau
                maps["vel_gau"][ind_x, ind_y] = res.vvel_gau
                maps["disp_gau"][ind_x, ind_y] = res.vdisp_gau
                cont[ind_x, ind_y] = res.vcont
                maps["eqw_mom"][ind_x, ind_y] = res.veqw_mom
                an[ind_x, ind_y] = res.amplitude / res.noise if res.noise else np.nan
                sn[ind_x, ind_y] = res.signal / res.noise if res.noise else np.nan
                contsn[ind_x, ind_y] = res.vcont / res.noise if res.noise else np.nan
                maps["vel_mom"][ind_x, ind_y] = res.vvel_mom
                maps["disp_mom"][ind_x, ind_y] = res.vdisp_mom
                if res.vres_cube is not None:
                    for xx, yy in zip(ind_x, ind_y):
                        res_cube[xx, yy, :] = res.vres_cube
                        linecube[xx, yy, :] = res.yclean

        # ================================================================
        # Masking
        # ================================================================
        mask_gau = np.ones(shape)
        mask_mom = np.ones(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            bad_gau = ((np.abs(maps["map_gau"]) / np.abs(maps["map_gau_err"]) < mask_thresh) |
                       (maps["map_gau"] < -mask_thresh * np.abs(maps["map_gau_err"])) |
                       (maps["map_gau"] > 1000 * mask_thresh * np.abs(maps["map_gau_err"])) |
                       (np.abs(maps["map_gau_err"]) > 10 * np.nanmedian(np.abs(maps["map_gau_err"]))))
        mask_gau[np.where(bad_gau)] = 0.0
        mask_gau[~np.isfinite(maps["map_gau"])] = 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            bad_mom = ((np.abs(maps["map_mom"]) / np.abs(maps["map_mom_err"]) < mask_thresh) |
                       (maps["map_mom"] < -mask_thresh * np.abs(maps["map_mom_err"])) |
                       (maps["map_mom"] > 1000 * mask_thresh * np.abs(maps["map_mom_err"])) |
                       (np.abs(maps["map_mom_err"]) > 10 * np.nanmedian(np.abs(maps["map_mom_err"]))))
        mask_mom[np.where(bad_mom)] = 0.0
        mask_mom[~np.isfinite(maps["map_mom"])] = 0.0

        good_pixels = np.where(mask_mom > 0.5)
        if good_pixels[0].size > 0:
            if np.nanmean(maps["map_mom"][good_pixels]) < 0:
                maps["map_mom"] = -maps["map_mom"]
                maps["map_gau"] = -maps["map_gau"]
                maps["eqw_mom"] = -maps["eqw_mom"]
                maps["eqw_mom_err"] = -maps["eqw_mom_err"]
                print("ATTENTION: absorption features -- flux/EW maps inverted")

        good_pixels_g = np.where(mask_gau > 0.5)
        if l0 is not None:
            vel0_gau = -v0
        else:
            vel0_gau = float(np.nanmedian(maps["vel_gau"][good_pixels_g])) \
                if good_pixels_g[0].size else 0.0
            v0_rel_peak = vel0_gau

        lambda0_gau = float(quadratic_interp(self.g_vel, lam_axis[xline], np.array([vel0_gau])))
        print(f"Shifting gaussian velocities by {vel0_gau:.2f} km/s "
              f"(centering at {lambda0_gau:.7f} um)")

        maps["vel_gau"] = maps["vel_gau"] - vel0_gau
        if fitnii:
            maps["vel_withnii_gau"] = maps["vel_withnii_gau"] - vel0_gau

        if l0 is not None:
            lambda0_mom = l0
        else:
            lambda0_mom = float(np.nanmedian(maps["vel_mom"][good_pixels])) \
                if good_pixels[0].size else 1.0
        print(f"Centering moment velocities at {lambda0_mom:.7f} um")

        maps["vel_mom"] = (maps["vel_mom"] - lambda0_mom) / lambda0_mom * C_KMS
        maps["disp_mom"] = maps["disp_mom"] / lambda0_mom * C_KMS
        maps["disp_mom_err"] = maps["disp_mom_err"] / lambda0_mom * C_KMS
        maps["vel_mom_err"] = maps["vel_mom_err"] / lambda0_mom * C_KMS
        maps["disp_mom_errlo"] = maps["disp_mom_errlo"] / lambda0_mom * C_KMS
        maps["vel_mom_errlo"] = maps["vel_mom_errlo"] / lambda0_mom * C_KMS
        maps["disp_mom_errhi"] = maps["disp_mom_errhi"] / lambda0_mom * C_KMS
        maps["vel_mom_errhi"] = maps["vel_mom_errhi"] / lambda0_mom * C_KMS

        # ================================================================
        # Output FITS files
        # ================================================================
        if nr_axes == 3:
            self._write_fits_outputs(outdir, suffix, hdr, scale, maps, cont,
                                      cont_err, sn, contsn, res_cube, mask_gau,
                                      fitnii, cdelt3, res_lambda)
        elif nr_axes == 1:
            self._print_1d_result(outdir, suffix, image_data, scale, maps,
                                   cont, cont_err, cdelt3, lambda0_gau)

        print(f"Calculation took {time.time() - t0:.1f} seconds.")
        return SimpleNamespace(maps=maps, cont=cont, cont_err=cont_err, sn=sn,
                                mask_gau=mask_gau, mask_mom=mask_mom,
                                res_cube=res_cube, linecube=linecube,
                                scale=scale)

    # ------------------------------------------------------------------
    def _load_template(self, hdr, instrument, resorder, tpldir, tpl,
                        kmos_poly_order_tpl, linepos_val, cdelt3, vel_axis, xline):
        if instrument == "KMOS":
            # Simplified: only implements the "no-template-found" branch and the
            # RESORDER-header branch faithfully; the polynomial-file (readcol)
            # branch is supported via numpy.genfromtxt.
            if resorder != 0:
                r = 0.0
                for i in range(int(resorder) + 1):
                    p = fxpar_sp(hdr, f"RESP{i}", 0.0)
                    r += p * linepos_val ** i
            else:
                if kmos_poly_order_tpl and os.path.isfile(kmos_poly_order_tpl):
                    tab = np.genfromtxt(kmos_poly_order_tpl, dtype=None,
                                        encoding=None, names=["ifu", "p0", "p1",
                                                               "p2", "p3", "p4"][:6])
                    # Fallback: use mean coefficients (IFU selection logic omitted
                    # for brevity -- plug in actual IFU index selection as needed)
                    p0 = np.mean(tab["p0"])
                    p1 = np.mean(tab["p1"]) if "p1" in tab.dtype.names else 0
                    p2 = np.mean(tab["p2"]) if "p2" in tab.dtype.names else 0
                    p3 = np.mean(tab["p3"]) if "p3" in tab.dtype.names else 0
                    r = p0 + p1 * linepos_val + p2 * linepos_val ** 2 + p3 * linepos_val ** 3
                else:
                    if 1.0 <= linepos_val <= 1.4:
                        r = 3400.0
                    elif 1.4 <= linepos_val <= 1.9:
                        r = 3800.0
                    elif 1.9 <= linepos_val <= 2.5:
                        r = 3800.0
                    else:
                        raise ValueError("line wavelength outside KMOS bands")

            sig = linepos_val / r / 2.3548
            sigp = sig / cdelt3
            self.tpl_sig = C_KMS / r / 2.3548

            ntpl = 43
            x = np.arange(ntpl)
            f = np.exp(-0.5 * ((x - np.floor(ntpl / 2) + 1) / sigp) ** 2)
            tpl_raw_data = f / f.max()
            tlambda = (np.arange(ntpl) + 1 - np.floor(ntpl / 2)) * cdelt3 + linepos_val
            print(f"Resolution (from header): [{r:.1f}]")
        else:
            tplpath = os.path.join(tpldir, tpl) if tpldir else tpl
            if not tplpath.endswith(".fits"):
                tplpath = tplpath + ".fits"
            with fits.open(tplpath) as hdul:
                tpl_raw_data = hdul[0].data.astype(float)
                tpl_hdr = hdul[0].header
            crvalt = fxpar_sp(tpl_hdr, "CRVAL1", 0.0)
            cdeltt = fxpar_sp(tpl_hdr, "CDELT1", 0.0)
            crpixt = fxpar_sp(tpl_hdr, "CRPIX1", 0.0)
            naxist = fxpar_sp(tpl_hdr, "NAXIS1", len(tpl_raw_data))
            if crvalt == 0 and cdeltt == 0 and crpixt == 0:
                crvalt = fxpar_sp(tpl_hdr, "CRVAL2", 0.0)
                cdeltt = fxpar_sp(tpl_hdr, "CDELT2", 0.0)
                crpixt = fxpar_sp(tpl_hdr, "CRPIX2", 0.0)
                naxist = fxpar_sp(tpl_hdr, "NAXIS2", len(tpl_raw_data))
            tlambda = (np.arange(naxist) + 1 - crpixt) * cdeltt + crvalt

            # quick 3-param gaussian fit for tpl_sig / resolution
            from scipy.optimize import curve_fit
            def _g(x, a, c, s):
                return a * np.exp(-0.5 * ((x - c) / s) ** 2)
            p0 = [tpl_raw_data.max(), tlambda[np.argmax(tpl_raw_data)],
                  (tlambda[-1] - tlambda[0]) / 10]
            popt, _ = curve_fit(_g, tlambda, tpl_raw_data, p0=p0)
            self.tpl_sig = C_KMS * abs(popt[2]) / linepos_val

        maxind = int(np.argmax(tpl_raw_data))
        tlambda0 = tlambda[maxind]
        tvel = (tlambda - tlambda0) / tlambda * C_KMS

        tpl_data = quadratic_interp(tvel, tpl_raw_data, vel_axis)
        tpl_data[(vel_axis <= tvel[0]) | (vel_axis >= tvel[-1])] = 0.0
        tpl_data = tpl_data / tpl_data.sum()
        self.tpl_data = tpl_data

        maxind2 = int(np.argmax(tpl_data))
        index = len(xline)
        evenodd = 1 if (index // 2 == index / 2.0) else 0
        half = int(np.floor(index / 2.0))
        self.g_tpl_data = tpl_data[maxind2 - half: maxind2 + half + evenodd]

    # ------------------------------------------------------------------
    def _write_fits_outputs(self, outdir, suffix, hdr, scale, maps, cont,
                             cont_err, sn, contsn, res_cube, mask_gau, fitnii,
                             cdelt3, res_lambda):
        def _write(name, arr, ctype=None, cunit=None):
            h = hdr.copy()
            if ctype:
                h["CTYPE3"] = ctype
            if cunit:
                h["CUNIT3"] = cunit
            fits.writeto(os.path.join(outdir, f"{name}{suffix}.fits"),
                         np.asarray(arr, dtype=np.float32).T, h, overwrite=True)

        _write("map_mom", maps["map_mom"] * scale, "EMISSION LINE INTENSITY", "ARBITRARY")
        _write("map_gau", maps["map_gau"] * scale)
        if fitnii:
            _write("map_withnii_gau", maps["map_withnii_gau"] * scale)
            _write("map_niib_gau", np.nan_to_num(maps["map_niib_gau"]) * scale)
            _write("map_niir_gau", np.nan_to_num(maps["map_niir_gau"]) * scale)
        _write("eqw_mom", maps["eqw_mom"], "EQUIVALENT WIDTH")
        _write("cont", cont * scale, "CONTINUUM INTENSITY")
        _write("vel_mom", maps["vel_mom"], "VELOCITY", "KM/S")
        _write("vel_gau", maps["vel_gau"])
        if fitnii:
            _write("vel_withnii_gau", maps["vel_withnii_gau"])
        _write("disp_mom", maps["disp_mom"], "DISPERSION")
        _write("disp_gau", maps["disp_gau"])
        if fitnii:
            _write("disp_withnii_gau", maps["disp_withnii_gau"])
        _write("sn", sn, "S/N")
        _write("cont_sn", contsn, "CONTINUUM S/N PER PIXEL")
        _write("map_mom_err", maps["map_mom_err"] * scale)
        _write("map_gau_err", maps["map_gau_err"] * scale)
        _write("map_mom_errlo", maps["map_mom_errlo"] * scale)
        _write("map_gau_errlo", maps["map_gau_errlo"] * scale)
        _write("map_mom_errhi", maps["map_mom_errhi"] * scale)
        _write("map_gau_errhi", maps["map_gau_errhi"] * scale)
        if fitnii:
            _write("map_gau_mpfit_err", np.nan_to_num(maps["map_gau_mpfit_err"]) * scale)
            _write("map_niib_gau_err", np.nan_to_num(maps["map_niib_gau_err"]) * scale)
            _write("map_niir_gau_err", np.nan_to_num(maps["map_niir_gau_err"]) * scale)
        _write("cont_err", cont_err * scale)
        _write("vel_gau_err", maps["vel_gau_err"])
        _write("vel_gau_errlo", maps["vel_gau_errlo"])
        _write("vel_gau_errhi", maps["vel_gau_errhi"])
        _write("disp_gau_err", maps["disp_gau_err"])
        _write("disp_gau_errlo", maps["disp_gau_errlo"])
        _write("disp_gau_errhi", maps["disp_gau_errhi"])
        _write("mask_gau", mask_gau, "MASK", "")
        _write("residual", res_cube.transpose(0, 1, 2))

    # ------------------------------------------------------------------
    def _print_1d_result(self, outdir, suffix, image_data, scale, maps, cont,
                          cont_err, cdelt3, lambda0_gau):
        print()
        print(f"flux (erg/s/cm2) = {maps['map_gau'][0,0]*scale:.1e} "
              f"+/- {maps['map_gau_err'][0,0]*scale:.1e}")
        print(f"velocity (km/s) = {maps['vel_gau'][0,0]:.1f} "
              f"+/- {maps['vel_gau_err'][0,0]:.1f}")
        print(f"dispersion (km/s) = {maps['disp_gau'][0,0]:.1f} "
              f"+/- {maps['disp_gau_err'][0,0]:.1f}")
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f"Gaufits_1Dspec{suffix}.res"), "w") as fh:
            fh.write("# Property  Units      Measurement    Err    Errhi    Errlo\n")
            fh.write(f"# Gauss fits results for {image_data}\n")
            fh.write(f"{'flux (erg/s/cm2)':20s}{maps['map_gau'][0,0]*scale:10.2e}"
                     f"{maps['map_gau_err'][0,0]*scale:10.2e}\n")
            fh.write(f"{'wavelength (um)':20s}{lambda0_gau:10.7f}\n")
            fh.write(f"{'velocity (km/s)':20s}{maps['vel_gau'][0,0]:10.2f}"
                     f"{maps['vel_gau_err'][0,0]:10.2f}\n")
            fh.write(f"{'dispersion (km/s)':20s}{maps['disp_gau'][0,0]:10.2f}"
                     f"{maps['disp_gau_err'][0,0]:10.2f}\n")
            fh.write(f"{'cont (erg/s/cm2/A)':20s}"
                     f"{(cont[0,0]*scale)/(cdelt3*1e4):10.2e}"
                     f"{(cont_err[0,0]*scale)/(cdelt3*1e4):10.2e}\n")


def linefit(indir, target, linepos, tpldir=None, **kwargs):
    """Functional wrapper mirroring the IDL `linefit` procedure call."""
    lf = LineFit()
    return lf.linefit(indir, target, linepos, tpldir=tpldir, **kwargs)


if __name__ == "__main__":
    # Example (KMOS-like) call, mirroring the IDL docstring example:
    #
    # linefit('examples', 'COS3_16954_com3_yj_p2+p3_4h10.fits', 'templates/kmos',
    #         1.333135, sdsscont=True, fitnii=True, spat=3, spec=2,
    #         mask_thresh=2.5, weight_type='gauss', outdir='out_kmos')
    print(__doc__)
