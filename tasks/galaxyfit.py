#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

Dependence:
    modeling_utils.py, version >= 0.2

History:
    2026-07-10: first release, v0.1

"""

from __future__ import annotations
__version__ = '0.1'

import argparse
import json
import pickle
import os
import warnings
from dataclasses import dataclass, field
from typing import Literal, Sequence, Optional, Union

import numpy as np
from numpy import linalg
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import minimize
from scipy import special
from scipy import optimize
from matplotlib import pyplot as plt
from astropy.convolution import convolve

from astropy.utils.exceptions import AstropyWarning

from modeling_utils import Parameter, Model, MinimizeFitter, EmceeFitter, Fitter, Grid2D, Sersic2D, Fitter, Line1D, Grid1D, DerivedParameter
from utils import plot_slider, image_slider

class Model2(Model):
    def interact(self, grid: "Grid", slider_ranges: Optional[dict] = None,
                 apply: bool = True, **kwargs) -> dict:
        """
        Launch an interactive slider GUI to explore this model's output
        as its parameters vary. Dispatches to `plot_slider` for 1D grids
        or `image_slider` for 2D grids.

        Parameters
        ----------
        grid : Grid
            Grid1D or Grid2D on which to evaluate the model.
        slider_ranges : dict, optional
            Override the auto-generated (min, max) range for specific
            parameters, e.g. {'bar.theta': (0, np.pi)}. Parameters not
            listed here get a range inferred from `self.limits`, with a
            sensible fallback if a bound is +/-inf.
        apply : bool, default True
            If True, push the final slider values back into the model's
            Parameters once the window is closed (so you can continue
            fitting/evaluating from the hand-tuned starting point).
        **kwargs :
            Forwarded to `plot_slider`/`image_slider` (e.g. xlabel,
            ylabel, cmap, origin, color, ...).

        Returns
        -------
        final_values : dict
            Parameter values at the moment the window was closed.

        Examples
        --------
        >>> galaxy = bulge + disk + bar
        >>> galaxy.interact(grid2d, cmap='inferno', origin='lower')
        """
        slider_ranges = slider_ranges or {}
        params = self.parameters
        limits = self.bounds

        slider_kwargs = {}
        for key, value in params.items():
            lo, hi = slider_ranges.get(key, limits.get(key, (-np.inf, np.inf)))
            lo, hi = self._resolve_slider_range(key, value, (lo, hi), grid)
            slider_kwargs[key] = {'default': value, 'range': (lo, hi)}

        if grid.ndim == 1:
            def func(x, **p):
                self.update_parameters(p)
                return self.evaluate(x)
            final = plot_slider(func, grid.x, slider_kwargs, **kwargs)

        elif grid.ndim == 2:
            def func(y, x, **p):
                self.update_parameters(p)
                return self.evaluate(y, x)
            y, x = grid.meshgrid
            final = image_slider(func, [y, x], slider_kwargs, **kwargs)

        else:
            raise NotImplementedError(
                f"interact() supports 1D/2D grids only (got ndim={grid.ndim})."
            )

        if apply:
            self.update_parameters(final)
        else:
            # restore original values, since evaluate()/update_parameters()
            # were called repeatedly during interaction
            self.update_parameters(params)

        return final

    @staticmethod
    def _resolve_slider_range(key, value, bounds, grid=None):
        """Infer a usable (lo, hi) slider range, falling back sensibly
        when a bound is +/-inf (common for x0, y0, theta, etc.)."""
        lo, hi = bounds
        attr = key.split('.')[-1]   # strip 'model_name.' prefix if present

        # Position parameters: prefer the grid's own coordinate extent
        if grid is not None and not (np.isfinite(lo) and np.isfinite(hi)):
            try:
                if attr == 'x0':
                    coords = grid.x
                    return float(np.min(coords)), float(np.max(coords))
                if attr == 'y0' and hasattr(grid, 'y'):
                    coords = grid.y
                    return float(np.min(coords)), float(np.max(coords))
            except Exception:
                pass

        # Generic fallback: scale around the current value
        span = abs(value) if value != 0 else 1.0
        lo = value - 5 * span if not np.isfinite(lo) else lo
        hi = value + 5 * span if not np.isfinite(hi) else hi
        if lo == hi:
            lo, hi = value - 1, value + 1
        return lo, hi

# class ModelList(Model):
#     def __init__(self, models: list[Model]):
#         self.models = models
#     def __getitem__(self, i: int) -> Model:
#         return self.models[i]
#     def _iter_params(self):
#         for model in self.models:
#             for m in model.submodels:
#                 yield from m._iter_params()
#     @property
#     def submodels(self) -> list[Model]:
#         _submodels = []
#         for model in self.models:
#             for submodel in model.submodels:
#                 _submodels.append(submodel)
#         return _submodels


class ModelList(Model):
    """
    Wraps one Model per filter/band for joint multi-band fitting.
    Unlike CompoundModel (+/-/*), models here stay logically separate
    -- each is evaluated against its own band's data -- but their free
    parameters are exposed as one union (with tying support), so a
    Fitter can treat them as a single parameter vector.
    """

    def __init__(self, models: list, filters: Optional[list] = None, name=None):
        super().__init__(name=name)
        self.models = models
        self.filters = filters or [f"band{i}" for i in range(len(models))]
        if len(self.filters) != len(self.models):
            raise ValueError("`filters` must match `models` in length.")
        self._disambiguate_names()

    def _disambiguate_names(self):
        """Ensure every leaf submodel has a name unique across ALL
        filters -- prevents the amplitude-freezing bug described above."""
        seen = {}
        for model, filt in zip(self.models, self.filters):
            for sub in model.submodels:
                seen.setdefault(sub.name, []).append((sub, filt))
        for key, entries in seen.items():
            if len(entries) > 1:
                for sub, filt in entries:
                    sub.name = f"{key}_{filt}"

    def __getitem__(self, i):
        return self.models[i]

    def __len__(self):
        return len(self.models)

    def _iter_params(self):
        for model in self.models:
            for m in model.submodels:
                yield from m._iter_params()

    @property
    def submodels(self):
        subs = []
        for model in self.models:
            subs.extend(model.submodels)
        return subs

    def evaluate(self, *args):
        return [m.evaluate(*args) for m in self.models]

    @classmethod
    def from_template(cls, template_builder, filters, tie_geometry=True,
                       geometry_params=('reff', 'n', 'x0', 'y0', 'ellip', 'theta')):
        """
        Build one independent model per filter from a shared template,
        then automatically tie geometry parameters across all filters.

        template_builder : callable
            Zero-arg function returning a fresh Model/CompoundModel,
            e.g. lambda: Sersic2D(...) + Sersic2D(...)  [bulge+disk]
        """
        models = [template_builder() for _ in filters]
        mlist = cls(models, filters=filters)

        if tie_geometry:
            n_leaves = len(models[0].submodels)
            for leaf_idx in range(n_leaves):
                for attr in geometry_params:
                    if not hasattr(models[0].submodels[leaf_idx], attr):
                        continue
                    names = [f"{m.submodels[leaf_idx].name}.{attr}" for m in models]
                    mlist.tie(*names)
        return mlist


class ExponentialDustScreen(Model):
    """Attenuation map: T(x,y) = exp(-tau0 * exp(-r/h_dust)), wavelength-dependent."""
    def __init__(self, tau0=0.5, h_dust=15.0, x0=0, y0=0, name=None):
        super().__init__(name=name)
        self.tau0 = Parameter(tau0, bounds=(0, 5))
        self.h_dust = Parameter(h_dust, bounds=(1, 100))
        self.x0, self.y0 = Parameter(x0), Parameter(y0)

    def evaluate(self, y, x):
        r = np.sqrt((x - self.x0.value)**2 + (y - self.y0.value)**2)
        tau = self.tau0.value * np.exp(-r / self.h_dust.value)
        return np.exp(-tau)   # transmission fraction, in [0,1]


class FixedDustScreen(Model):
    """
    Wraps a known, fixed dust surface density map as a wavelength-dependent
    attenuation screen:

        tau(x,y) = kappa * scale * Sigma_dust(x,y)
        T(x,y)   = exp(-tau)                      [foreground screen]
                   or (1 - exp(-tau)) / tau        [mixed slab]

    Intended usage -- multiply with a light-emitting model using the `*`
    operator already defined on Model:

        observed_model = intrinsic_light_model * dust_screen

    Parameters
    ----------
    dust_map : ndarray
        2D dust surface mass density map (e.g. Msun/pc^2), assumed known
        (e.g. from independent energy-balance / FIR SED fitting) and
        therefore NOT a fit parameter.
    dust_grid : Grid2D, optional
        The grid on which `dust_map` was originally computed/sampled.
        If it differs from the grid used when this Model is evaluated
        (e.g. different pixel scale or footprint), the map is
        automatically resampled via linear interpolation. If None,
        `dust_map` is assumed to already match the evaluation grid
        shape exactly (no resampling).
    kappa : float
        Mass attenuation coefficient at THIS filter's wavelength (units
        consistent with dust_map, e.g. pc^2/Msun). This is the primary
        wavelength-dependent free/derived parameter -- attach one
        FixedDustScreen per filter and tie their `kappa` values via a
        power-law dust law (see `attach_power_law_kappa` below).
    scale : float
        Overall calibration/normalization factor multiplying the dust
        map before computing tau (default 1.0, free to fit). Useful to
        absorb systematic uncertainty in the absolute mass calibration
        without needing to touch the input map itself.
    mixed_slab : bool, default False
        If True, use the mixed-slab attenuation law (dust genuinely
        interspersed with stars, e.g. appropriate for very young stellar
        populations per Charlot & Fall 2000). If False, use a simple
        foreground screen (dust entirely in front of the stars).
    name : str, optional
    """

    def __init__(self, dust_map, kappa=1.0, scale=1.0,
                 mixed_slab=False, name=None):
        super().__init__(name=name)

        dust_map = np.asarray(dust_map, dtype=float)
        if np.any(dust_map < 0):
            warnings.warn(
                f"[{name or 'FixedDustScreen'}] dust_map contains negative "
                f"values; these will produce transmission > 1 (net "
                f"brightening), which is usually unphysical."
            )
        self._dust_map = dust_map
        self.mixed_slab = mixed_slab

        self.kappa = Parameter(kappa, bounds=(0.0, np.inf))
        self.scale = Parameter(scale, bounds=(0.0, np.inf))

    # ------------------------------------------------------------
    def evaluate(self, y, x):
        sigma_dust = self._resample_to(y, x)
        tau = self.kappa.value * self.scale.value * sigma_dust

        if self.mixed_slab:
            with np.errstate(divide='ignore', invalid='ignore'):
                transmission = np.where(
                    tau > 1e-6,
                    (1.0 - np.exp(-tau)) / np.where(tau == 0, 1.0, tau),
                    1.0,
                )
        else:
            transmission = np.exp(-tau)

        return transmission

    # ------------------------------------------------------------
    # Convenience: tie kappa across filters via a standard power-law
    # dust attenuation curve, kappa(lambda) = kappa0 * (lambda/lambda0)^-beta
    # ------------------------------------------------------------
    @staticmethod
    def attach_power_law_kappa(screens: dict, wavelengths: dict,
                                lambda0: float = 5500.0, beta0: float = 1.3):
        """
        Tie the `kappa` parameters of multiple FixedDustScreen instances
        (one per filter) to a single shared power-law dust curve:

            kappa(lambda) = kappa0 * (lambda / lambda0) ** (-beta)

        Parameters
        ----------
        screens : dict {filter_name: FixedDustScreen}
        wavelengths : dict {filter_name: pivot_wavelength_in_angstrom}
        lambda0 : float
            Reference wavelength (e.g. 5500 A, V-band) for kappa0.
        beta0 : float
            Initial guess for the power-law slope (~1.3 for Calzetti-like,
            ~0.7-1.0 for a Milky-Way-like extinction curve).

        Returns
        -------
        kappa0, beta : Parameter, Parameter
            The two shared free parameters -- fit these instead of each
            filter's individual `kappa`.
        """
        kappa0 = Parameter(1.0, bounds=(0.0, np.inf))
        beta = Parameter(beta0, bounds=(0.0, 3.0))

        for filt, screen in screens.items():
            lam = wavelengths[filt]

            def predict(k0, b, lam=lam, lambda0=lambda0):
                return k0 * (lam / lambda0) ** (-b)

            screen.kappa = DerivedParameter(predict, kappa0, beta)

        return kappa0, beta

## Bar profile
class Ferrers2D(Model2):
    """
    2D Modified Ferrers Profile for modeling galactic bars.
    
    The classic Ferrers (1877) density profile, adapted for 2D surface
    brightness modeling of bars:
    
        I(r) = I0 * [1 - (r/a_bar)^2]^(n + 0.5),  for r <= a_bar
        I(r) = 0,                                  for r > a_bar
    
    where r is a generalized elliptical radius that can include boxiness:
    
        r = [ |x_maj/a_bar|^(c+2) + |x_min/b_bar|^(c+2) ]^(1/(c+2))
    
    Parameters
    ----------
    amplitude : float
        Central intensity I0 (peak surface brightness at r=0).
    a_bar : float
        Bar semi-major axis length (truncation radius along major axis).
    n : float
        Ferrers profile index. Controls how "flat-topped" vs "peaked"
        the bar is near the center. Typical values: n = 1-3.
        (Effective power-law exponent applied is n + 0.5 for smoothness
        at r = a_bar, following Ferrers' original convention.)
    x0, y0 : float
        Center coordinates of the bar.
    ellip : float
        Ellipticity, defined as (1 - b_bar/a_bar), where b_bar is the
        bar's semi-minor axis. Typical bar values: 0.6-0.8 (i.e., b/a ~ 0.2-0.4).
    theta : float
        Position angle of the bar major axis, in radians,
        measured counter-clockwise from the +x axis.
    c : float, optional (default=0.0)
        Boxiness/diskiness parameter for generalized ellipse isophotes.
            c > 0 : boxy isophotes (typical for strong bars)
            c = 0 : pure elliptical isophotes
            c < 0 : disky isophotes
    name : str, optional
        Model name.
    """
    
    def __init__(self, amplitude=1, a_bar=10, n=2, x0=0, y0=0,
                 ellip=0.3, theta=0, c=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0, np.inf))
        self.a_bar = Parameter(a_bar, bounds=(0, np.inf))
        self.n = Parameter(n, bounds=(0.5,4))
        self.x0 = Parameter(x0, bounds=(-np.inf, np.inf))
        self.y0 = Parameter(y0, bounds=(-np.inf, np.inf))
        self.ellip = Parameter(ellip, bounds=(0.001, 0.999))
        self.theta = Parameter(theta, bounds=(-np.pi/2, np.pi/2))
        self.c = Parameter(c, bounds=(-2,8))
        self.name = name

    def evaluate(self, y, x):
        # y, x = grid.meshgrid

        cos_theta = np.cos(self.theta.value)
        sin_theta = np.sin(self.theta.value)

        # Rotate coordinates into the bar's major/minor axis frame
        x_maj = (x - self.x0.value) * cos_theta + (y - self.y0.value) * sin_theta
        x_min = -(x - self.x0.value) * sin_theta + (y - self.y0.value) * cos_theta

        a_bar = self.a_bar.value
        b_bar = (1.0 - self.ellip.value) * a_bar

        # Generalized elliptical radius (supports boxy/disky isophotes)
        expon = self.c.value + 2.0
        inv_expon = 1.0 / expon
        r = (np.abs(x_maj / a_bar) ** expon +
             np.abs(x_min / b_bar) ** expon) ** inv_expon

        # Modified Ferrers profile: nonzero only inside r <= 1 (i.e., within a_bar)
        z = np.zeros_like(r)
        inside = r < 1.0

        # Avoid negative base issues by clipping at machine precision
        base = np.clip(1.0 - r[inside] ** 2, 0.0, None)
        z[inside] = base ** (self.n.value + 0.5)

        return self.amplitude.value * z

class Ferrers2DExtended(Model2):
    """
    Extended Modified Ferrers profile with independent inner and outer
    exponents (GALFIT-style):
    
        I(r) = I0 * [1 - (r/a_bar)^(2 - n_inner)]^n_out,  for r <= a_bar
        I(r) = 0,                                          for r > a_bar
    """
    
    def __init__(self, amplitude=1, a_bar=10, n_inner=2, n_out=1, 
                 x0=0, y0=0, ellip=0.3, theta=0, c=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0, np.inf))
        self.a_bar = Parameter(a_bar, bounds=(0, np.inf))
        self.n_inner = Parameter(n_inner, bounds=(0.5,4))
        self.n_out = Parameter(n_out, bounds=(0.5,4))
        self.x0 = Parameter(x0, bounds=(-np.inf, np.inf))
        self.y0 = Parameter(y0, bounds=(-np.inf, np.inf))
        self.ellip = Parameter(ellip, bounds=(0.001, 0.999))
        self.theta = Parameter(theta, bounds=(-np.pi/2, np.pi/2))
        self.c = Parameter(c, bounds=(-2,8))
        self.name = name

    def evaluate(self, y, x):
        cos_theta = np.cos(self.theta.value)
        sin_theta = np.sin(self.theta.value)

        x_maj = (x - self.x0.value) * cos_theta + (y - self.y0.value) * sin_theta
        x_min = -(x - self.x0.value) * sin_theta + (y - self.y0.value) * cos_theta

        a_bar = self.a_bar.value
        b_bar = (1.0 - self.ellip.value) * a_bar

        expon = self.c.value + 2.0
        inv_expon = 1.0 / expon
        r = (np.abs(x_maj / a_bar) ** expon +
             np.abs(x_min / b_bar) ** expon) ** inv_expon

        z = np.zeros_like(r)
        inside = r < 1.0

        power = 2.0 - self.n_inner.value
        base = np.clip(1.0 - r[inside] ** power, 0.0, None)
        z[inside] = base ** self.n_out.value

        return self.amplitude.value * z

@dataclass
class FittedModel:
    """
    Container for the result of a model fit, produced by either
    MinimizeFitter or EmceeFitter.

    This is a *snapshot*: it stores best-fit values, uncertainties, and
    optional posterior samples, but does NOT hold a live reference to
    the Fitter/sampler -- making it lightweight to save, pass around,
    and reload independent of the original fitting session.

    Attributes
    ----------
    model : Model
        The fitted Model (or CompoundModel). NOT automatically updated
        to best-fit values -- call `.apply()` to push best_fit into it.
    keys : list[str]
        Parameter names, in the same order as best_fit/errors/covariance.
    best_fit : np.ndarray
        Best-fit parameter values.
    best_fit_error : np.ndarray, optional
        Symmetric 1-sigma uncertainties (from covariance diagonal, or
        MCMC percentile-based if errors are roughly symmetric).
    best_fit_error_lower, best_fit_error_upper : np.ndarray, optional
        Asymmetric uncertainties (e.g., 16th/84th percentile distances
        from the median, for MCMC results). If set, these take
        precedence over `best_fit_error` for reporting.
    covariance : np.ndarray, optional
        Full parameter covariance matrix (least_squares fits only).
    samples : np.ndarray, optional
        Flattened posterior samples, shape (n_samples, n_params).
        Only present for MCMC-based fits.
    chi2, reduced_chi2, dof : float / int, optional
        Goodness-of-fit statistics (data residuals only, priors excluded).
    method : str
        Which fitting method produced this result, e.g.
        'minimize', 'least_squares', 'emcee'.
    converged : bool, optional
        Convergence flag (meaningful mainly for MCMC / iterative fits).
    grid : Grid, optional
        The evaluation grid used during fitting (stored for convenience,
        so `.predict()` works without needing to pass it again).
    data : dict, optional
        The original data dict used for fitting (stored for residual
        computation / plotting).
    mask : np.ndarray, optional
    metadata : dict
        Free-form extra info (e.g., runtime, backend file path, notes).
    """

    model: "Model"
    keys: list
    best_fit: np.ndarray
    best_fit_error: Optional[np.ndarray] = None
    best_fit_error_lower: Optional[np.ndarray] = None
    best_fit_error_upper: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
    chi2: Optional[float] = None
    reduced_chi2: Optional[float] = None
    dof: Optional[int] = None
    method: str = "unknown"
    converged: Optional[bool] = None
    grid: Optional["Grid"] = None
    data: Optional[dict] = None
    mask: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------
    @classmethod
    def from_minimize_fitter(cls, fitter: "MinimizeFitter", method: str = "least_squares") -> "FittedModel":
        """Build a FittedModel from a MinimizeFitter after auto_run()/run()."""
        best_fit = np.asarray(fitter.best_fit)
        keys = list(fitter.keys)

        chi2 = reduced_chi2 = dof = None
        if hasattr(fitter, "reduced_chi2"):
            reduced_chi2 = fitter.reduced_chi2
            fitter.models.update_parameters(dict(zip(keys, best_fit)))
            chi2 = float(np.sum(fitter._calculate_diff()))
            n_data = fitter._data_size()
            dof = n_data - len(best_fit)

        return cls(
            model=fitter.models,
            keys=keys,
            best_fit=best_fit,
            best_fit_error=getattr(fitter, "best_fit_error", None),
            covariance=getattr(fitter, "pcov", None),
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            dof=dof,
            method=method,
            grid=fitter.grid,
            data=fitter.data,
            mask=fitter.mask,
        )

    @classmethod
    def from_emcee_fitter(
        cls,
        fitter: "EmceeFitter",
        discard: Optional[int] = None,
        thin: Optional[int] = None,
        percentiles: tuple = (16, 50, 84),
    ) -> "FittedModel":
        """Build a FittedModel from an EmceeFitter after auto_run()/run()."""
        samples = fitter.samples(discard=discard, thin=thin, flat=True)
        keys = list(fitter.keys)

        lo, med, hi = np.percentile(samples, percentiles, axis=0)
        err_lower = med - lo
        err_upper = hi - med
        err_symmetric = 0.5 * (err_lower + err_upper)

        fitter.models.update_parameters(dict(zip(keys, med)))
        chi2 = float(np.sum(fitter._calculate_diff()))
        n_data = fitter._data_size()
        dof = n_data - len(med)
        reduced_chi2 = chi2 / dof if dof > 0 else np.nan

        return cls(
            model=fitter.models,
            keys=keys,
            best_fit=med,
            best_fit_error=err_symmetric,
            best_fit_error_lower=err_lower,
            best_fit_error_upper=err_upper,
            samples=samples,
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            dof=dof,
            method="emcee",
            converged=getattr(fitter, "converged", None),
            grid=fitter.grid,
            data=fitter.data,
            mask=fitter.mask,
            metadata={"autocorr_time": getattr(fitter, "autocorr", None)},
        )

    @classmethod
    def from_fitter(cls, fitter, **kwargs) -> "FittedModel":
        """Auto-dispatch based on fitter type (duck-typed on presence of `.sampler`)."""
        if hasattr(fitter, "sampler"):
            return cls.from_emcee_fitter(fitter, **kwargs)
        return cls.from_minimize_fitter(fitter, **kwargs)

    # ------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------
    @property
    def parameters(self) -> dict:
        """Best-fit values as a name -> value dict."""
        return dict(zip(self.keys, self.best_fit))

    @property
    def errors(self) -> dict:
        """Best-fit 1-sigma errors as a name -> value dict (symmetric)."""
        if self.best_fit_error is None:
            return {k: np.nan for k in self.keys}
        return dict(zip(self.keys, self.best_fit_error))

    @property
    def is_mcmc(self) -> bool:
        return self.samples is not None

    def apply(self) -> "Model":
        """Push best-fit values into the stored model (mutates self.model)."""
        self.model.update_parameters(self.parameters)
        return self.model

    def predict(self, grid: Optional["Grid"] = None) -> np.ndarray:
        """Evaluate the model at best-fit parameters on a grid
        (defaults to the grid used during fitting)."""
        self.apply()
        grid = grid if grid is not None else self.grid
        if grid is None:
            raise ValueError("No grid available -- pass one explicitly.")
        return self.model(grid)

    def residual(self, grid: Optional["Grid"] = None, data: Optional[np.ndarray] = None) -> np.ndarray:
        """obs - model, evaluated at best-fit parameters."""
        model_img = self.predict(grid)
        obs = data if data is not None else self.data.get("data")
        if obs is None:
            raise ValueError("No observed data available -- pass `data` explicitly.")
        return obs - model_img

    # ------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable parameter table + goodness-of-fit summary."""
        lines = [f"FittedModel (method={self.method}, converged={self.converged})"]
        lines.append("-" * 60)
        name_w = max(len(k) for k in self.keys) + 2

        for i, key in enumerate(self.keys):
            val = self.best_fit[i]
            if self.best_fit_error_lower is not None:
                lo, hi = self.best_fit_error_lower[i], self.best_fit_error_upper[i]
                lines.append(f"{key:<{name_w}} = {val: .5g}  (+{hi:.3g} / -{lo:.3g})")
            elif self.best_fit_error is not None:
                err = self.best_fit_error[i]
                lines.append(f"{key:<{name_w}} = {val: .5g}  +/- {err:.3g}")
            else:
                lines.append(f"{key:<{name_w}} = {val: .5g}")

        lines.append("-" * 60)
        if self.chi2 is not None:
            lines.append(f"chi2 = {self.chi2:.4g}   dof = {self.dof}   "
                          f"reduced_chi2 = {self.reduced_chi2:.4g}")
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:
        conv = f", converged={self.converged}" if self.converged is not None else ""
        rchi2 = f", reduced_chi2={self.reduced_chi2:.3g}" if self.reduced_chi2 is not None else ""
        return f"<FittedModel method={self.method!r}{conv}{rchi2} n_params={len(self.keys)}>"

    # ------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------
    def plot_corner(self, truths: Optional[np.ndarray] = None, **kwargs):
        """Corner plot of posterior samples (MCMC results only)."""
        if not self.is_mcmc:
            warnings.warn("No samples stored -- corner_plot only available for MCMC results.")
            return None
        try:
            import corner
        except ImportError:
            print("Install `corner` for plot_corner().")
            return None
        return corner.corner(self.samples, labels=self.keys, truths=truths, **kwargs)

    def plot_fit(self, grid: Optional[Grid] = None, figsize=(12, 4)):
        """Quick 3-panel plot: data, model, residual (2D data only)."""
        import matplotlib.pyplot as plt

        grid = grid if grid is not None else self.grid
        model_img = self.predict(grid)
        obs = self.data.get("data")
        resid = obs - model_img

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, img, title in zip(axes, [obs, model_img, resid], ["Data", "Model", "Residual"]):
            im = ax.imshow(img, origin="lower")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.show()
        return fig

    # ------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------
    def to_dict(self, include_samples: bool = False) -> dict:
        """Lightweight, JSON-friendly summary (excludes the Model object
        and Grid/data references, which may not be JSON-serializable)."""
        d = {
            "method": self.method,
            "converged": self.converged,
            "keys": self.keys,
            "best_fit": np.asarray(self.best_fit).tolist(),
            "best_fit_error": (
                np.asarray(self.best_fit_error).tolist()
                if self.best_fit_error is not None else None
            ),
            "best_fit_error_lower": (
                np.asarray(self.best_fit_error_lower).tolist()
                if self.best_fit_error_lower is not None else None
            ),
            "best_fit_error_upper": (
                np.asarray(self.best_fit_error_upper).tolist()
                if self.best_fit_error_upper is not None else None
            ),
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "dof": self.dof,
            "metadata": self.metadata,
        }
        if include_samples and self.samples is not None:
            d["samples"] = np.asarray(self.samples).tolist()
        return d

    def to_json(self, path: str, **kwargs) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(**kwargs), f, indent=2)

    def save(self, path: str) -> None:
        """Full-fidelity save (includes the Model, Grid, data, samples)
        via pickle. Requires Model/Grid to be picklable."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FittedModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a FittedModel instance.")
        return obj

    # ------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------
    @staticmethod
    def compare(*results: "FittedModel", names: Optional[list] = None,
                sort_by: str = "aic", verbose: bool = True):
        """
        Compare multiple fitted models via chi^2, AIC, BIC, and
        parameter-by-parameter differences -- useful for model
        selection (e.g., bulge-only vs. bulge+disk vs. bulge+disk+bar).

        Parameters
        ----------
        *results : FittedModel
            Two or more fit results to compare.
        names : list of str, optional
            Labels for each model (defaults to "model_0", "model_1", ...
            or each result's `.metadata['name']` if present).
        sort_by : str
            Column to sort the summary table by: 'aic', 'bic', 'chi2',
            or 'reduced_chi2'. Lower is better for all of these.
        verbose : bool
            If True, print the summary table and a short recommendation.

        Returns
        -------
        summary : pandas.DataFrame (or list[dict] if pandas unavailable)
            One row per model, with columns:
            name, n_params, dof, chi2, reduced_chi2, aic, bic,
            delta_aic, delta_bic, akaike_weight.
        param_table : pandas.DataFrame (or list[dict])
            Parameter-by-parameter comparison across models (rows =
            parameter names, columns = each model's best-fit +/- error).
        """
        if len(results) < 2:
            raise ValueError("compare() requires at least two FittedModel instances.")

        if names is None:
            names = [
                r.metadata.get("name", f"model_{i}")
                for i, r in enumerate(results)
            ]
        if len(names) != len(results):
            raise ValueError("`names` must have the same length as `results`.")

        # ---------- per-model goodness-of-fit summary ----------
        rows = []
        for name, r in zip(names, results):
            if r.chi2 is None or r.dof is None:
                warnings.warn(f"Model '{name}' is missing chi2/dof -- skipping "
                               f"AIC/BIC computation for it.")
                k = len(r.keys)
                n_data = np.nan
                aic = bic = np.nan
            else:
                k = len(r.keys)
                n_data = r.dof + k
                aic = r.chi2 + 2 * k
                bic = r.chi2 + k * np.log(n_data)

            rows.append({
                "name": name,
                "n_params": k,
                "n_data": n_data,
                "dof": r.dof,
                "chi2": r.chi2,
                "reduced_chi2": r.reduced_chi2,
                "aic": aic,
                "bic": bic,
            })

        # ---------- delta AIC/BIC + Akaike weights ----------
        aics = np.array([row["aic"] for row in rows], dtype=float)
        bics = np.array([row["bic"] for row in rows], dtype=float)

        valid = np.isfinite(aics)
        min_aic = np.nanmin(aics) if valid.any() else np.nan
        min_bic = np.nanmin(bics) if np.isfinite(bics).any() else np.nan

        delta_aic = aics - min_aic
        delta_bic = bics - min_bic

        # Akaike weights: exp(-0.5*delta_aic) / sum(...)
        rel_likelihood = np.exp(-0.5 * delta_aic)
        akaike_weights = rel_likelihood / np.nansum(rel_likelihood)

        for i, row in enumerate(rows):
            row["delta_aic"] = delta_aic[i]
            row["delta_bic"] = delta_bic[i]
            row["akaike_weight"] = akaike_weights[i]

        sort_key = {"aic": "aic", "bic": "bic", "chi2": "chi2",
                    "reduced_chi2": "reduced_chi2"}.get(sort_by, "aic")
        rows = sorted(rows, key=lambda r: (np.inf if not np.isfinite(r[sort_key]) else r[sort_key]))

        # ---------- parameter-by-parameter comparison table ----------
        all_keys = []
        for r in results:
            for k in r.keys:
                if k not in all_keys:
                    all_keys.append(k)

        param_rows = []
        for key in all_keys:
            row = {"parameter": key}
            for name, r in zip(names, results):
                if key in r.keys:
                    idx = r.keys.index(key)
                    val = r.best_fit[idx]
                    if r.best_fit_error_lower is not None:
                        lo = r.best_fit_error_lower[idx]
                        hi = r.best_fit_error_upper[idx]
                        row[name] = f"{val:.4g} (+{hi:.2g}/-{lo:.2g})"
                    elif r.best_fit_error is not None:
                        err = r.best_fit_error[idx]
                        row[name] = f"{val:.4g} +/- {err:.2g}"
                    else:
                        row[name] = f"{val:.4g}"
                else:
                    row[name] = "—"
            param_rows.append(row)

        # ---------- try pandas, fall back to plain text ----------
        try:
            import pandas as pd
            summary_df = pd.DataFrame(rows)[
                ["name", "n_params", "n_data", "dof", "chi2", "reduced_chi2",
                 "aic", "bic", "delta_aic", "delta_bic", "akaike_weight"]
            ]
            param_df = pd.DataFrame(param_rows).set_index("parameter")
            summary_out, param_out = summary_df, param_df
        except ImportError:
            summary_out, param_out = rows, param_rows

        if verbose:
            FittedModel._print_comparison(rows, param_rows, names)

        return summary_out, param_out

    @staticmethod
    def _print_comparison(rows, param_rows, names):
        """Plain-text pretty-printer, used when pandas is unavailable
        or simply for quick terminal output."""
        print("=" * 78)
        print("MODEL COMPARISON (sorted by AIC, lower is better)")
        print("=" * 78)
        header = f"{'name':<20}{'k':>4}{'chi2':>10}{'red.chi2':>10}{'AIC':>10}{'dAIC':>8}{'weight':>9}"
        print(header)
        print("-" * 78)
        for row in rows:
            print(
                f"{row['name']:<20}"
                f"{row['n_params']:>4d}"
                f"{row['chi2']:>10.4g}"
                f"{row['reduced_chi2']:>10.4g}"
                f"{row['aic']:>10.4g}"
                f"{row['delta_aic']:>8.2f}"
                f"{row['akaike_weight']:>9.3f}"
            )
        print("=" * 78)

        best = rows[0]
        if best["akaike_weight"] > 0.9:
            verdict = f"'{best['name']}' is strongly preferred (Akaike weight={best['akaike_weight']:.3f})."
        elif best["delta_aic"] == 0 and len(rows) > 1 and rows[1]["delta_aic"] < 2:
            verdict = (f"'{best['name']}' and '{rows[1]['name']}' are statistically "
                       f"indistinguishable (delta_AIC={rows[1]['delta_aic']:.2f} < 2).")
        else:
            verdict = f"'{best['name']}' is the best-supported model (delta_AIC to next-best = {rows[1]['delta_aic']:.2f})." if len(rows) > 1 else ""
        print(verdict)
        print()

        print("PARAMETER COMPARISON")
        print("-" * 78)
        name_w = max(len(p["parameter"]) for p in param_rows) + 2
        col_w = max(max(len(str(p[n])) for p in param_rows) for n in names) + 2
        header = f"{'parameter':<{name_w}}" + "".join(f"{n:<{col_w}}" for n in names)
        print(header)
        print("-" * 78)
        for p in param_rows:
            print(f"{p['parameter']:<{name_w}}" + "".join(f"{str(p[n]):<{col_w}}" for n in names))
        print("=" * 78)


class MinimizeFitter2(MinimizeFitter):
    # add auto_run with two cycles of minimization
    def auto_run(self, initial_guess=None, debug=False, maxiter=None):
        """
        Two-step fit:
          1) `minimize` for a robust starting point (handles bounds,
             flat/hard priors well, less sensitive to bad initial guesses).
          2) `least_squares` (Trust Region Reflective + soft_l1 loss) from
             that starting point, to additionally obtain a covariance-based
             parameter uncertainty estimate via the Jacobian at the optimum.
        """
        # ---- Step 1: robust scalar minimization ----

        first_result = optimize.minimize(self.cost, self.initial_guess, bounds=self.bounds)

        data_size = self._data_size()
        n_params = self.n_params

        if debug:
            self.models.update_parameters(first_result.x)
            first_chi2 = np.sum(self._calculate_diff())
            dof0 = max(data_size - n_params, 1)
            print("Parameters:", self.keys)
            print("First best-fit:", first_result.x)
            print(f"data_size={data_size}; chi2={first_chi2:.4g}; "
                  f"reduced_chi2={first_chi2 / dof0:.4g}")

        # ---- Step 2: least_squares for covariance/error estimation ----
        bounds_arr = np.array(self.bounds).T
        res = optimize.least_squares(
            self.vector_cost, first_result.x, jac='3-point',
            bounds=(bounds_arr[0], bounds_arr[1]), method='trf', loss='soft_l1',
            max_nfev=maxiter, verbose=int(debug),
        )

        if not res.success:
            warnings.warn(f"Optimal parameters not found: {res.message}")

        popt = res.x
        n_resid_total = len(res.fun)          # includes prior pseudo-residuals
        n_prior_resid = self._n_prior_residuals()
        n_data_resid = n_resid_total - n_prior_resid

        # ---- Covariance from Jacobian via SVD (curve_fit-style) ----
        _, s, VT = linalg.svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)

        # ---- Reduced chi^2 computed from DATA residuals only ----
        # (prior pseudo-residuals act as regularization, but shouldn't
        #  inflate/deflate the reported goodness-of-fit statistic)
        self.models.update_parameters(popt)
        chi2_data = np.sum(self._calculate_diff())
        dof = n_data_resid - n_params

        best_fit_error = None
        reduced_chi2 = None
        if dof > 0:
            reduced_chi2 = chi2_data / dof
            pcov_scaled = pcov * reduced_chi2
            best_fit_error = np.sqrt(np.diag(pcov_scaled))
            if debug:
                print(f"n_data_resid={n_data_resid}; n_prior_resid={n_prior_resid}")
                print(f"chi2_data={chi2_data:.4g}; dof={dof}; "
                      f"reduced_chi2={reduced_chi2:.4g}")
                print(f"Parameter errors: {best_fit_error}")
        else:
            warnings.warn(
                f"Degrees of freedom <= 0 (n_data_resid={n_data_resid}, "
                f"n_params={n_params}); cannot estimate parameter errors."
            )

        self.best_fit0 = first_result.x
        self.best_fit = popt
        self.best_fit_error = best_fit_error
        self.reduced_chi2 = reduced_chi2
        self.pcov = pcov


class MinimizePSFFitter(MinimizeFitter2):
    """extend the MinimizeFitter by support a additional convolution step,
    which can be handy for PSF modelling in galaxy observations

    """
    def __init__(self, models, grid, data, psf, psf_fwhm=None, name='None'):
        super().__init__(models=models, grid=grid, data=data, name='name')
        limits = models.get_limits()
        limits_ranges = np.array(list(limits.values()))
        #self.bounds = limits_ranges
        self.psf = psf
        self.psf_fwhm = psf_fwhm
    def vector_cost(self, theta):
        data = self.data
        model_parameters = self.models.get_parameters()
        dict_theta = dict(zip(model_parameters.keys(), theta))
        self.models.update_parameters(dict_theta)
        params_priors = self.models.get_priors(dict_theta)
        model = self.models.create_model(self.grid)
        var = data['var']
        try: var_err = data['var_err']
        except: var_err = 0
        obs = data['data']
        try: obs_err = data['data_err']
        except: obs_err = 0
        yrad, xrad = self.grid.meshgrid

        sigma2 = var_err**2 + obs_err**2 #+ 5**2 + xrad**2 + yrad**2 #+ err2_addition
        if np.sum(sigma2) < 1e-8:
            sigma2 = 1
        
        if self.psf_fwhm is not None:
            rmap = np.sqrt(yrad**2 + xrad**2)
            rmap[rmap < self.psf_fwhm] = 0.5*self.psf_fwhm
            sigma2 = rmap**2
        
        # check if interpolate is needed
        pass
        # interpolate the model grid to the observed coordinates
        if obs.ndim == 1:
            model_interp = interpolate.interpn(self.grid.grid(), model, var)
        if obs.ndim == 2:
            model_interp = interpolate.interpn(
                    self.grid.grid, model, np.array([var[0].flatten(),var[1].flatten()]).T).reshape(obs.shape)
        if obs.ndim == 3:
            model_interp = interpolate.interpn(
                    self.grid.grid, model, np.array([var[0].flatten(),var[1].flatten(), var[2].flatten()
                                                  ]).T).reshape(obs.shape)
        model_interp_convolved = convolve(model_interp, self.psf)
        diff2 = (obs-model_interp_convolved)**2/sigma2
        return (diff2 - params_priors).ravel()
    def cost(self, theta):
        return np.sum(self.vector_cost(theta))

class MinimizeGalaxyFitter(MinimizeFitter2):
    def __init__(self, models, grid, data, psf, mask=None, name='None'):
        super().__init__(models=models, grid=grid, data=data, name=name)
        limits = models.bounds
        limits_ranges = np.array(list(limits.values()))
        #self.bounds = limits_ranges
        self.psf = psf
        self.mask = mask
        #if self.mask is not None: 
        #    print(f"Masked pixels is {np.sum(self.mask)}")
    def _residuals(self):
        """Signed, sigma-normalized, mask-aware residuals."""
        obs = self.data['data']
        var_err = self.data.get('var_err', 0)
        obs_err = self.data.get('data_err', 0)
        sigma2 = np.asarray(var_err) ** 2 + np.asarray(obs_err) ** 2
        sigma2 = np.where(sigma2 <= 0, 1.0, sigma2)

        model_interp = self._model_at_data()
        model_interp_convolved = convolve(model_interp, self.psf)
        resid = (obs - model_interp_convolved) / np.sqrt(sigma2)

        if self.mask is not None:
            resid = resid[~self.mask]
        return resid

class MinimizeGalaxyMultipleFitter(MinimizeFitter2):
    """Model galaxy morphology with multiple images at different wavelength
    """
    def __init__(self, models, grid, data, filters, mask=None, name='None'):
        """
        Compared to MinimizeGalaxyFitter, the input data now include multiband
        observations

        data = {'F444W':data1, 'F444W_err': err1, 'F444W_psf': psf1,
                'F277W':data2, 'F277W_err': err2, 'F277W_psf': psf2,
        }
        filters = ['F444W', 'F277W']
        models = [model_f444w, model_f277w]

        The data keys should be the same name as filters

        All the data should be sampled on the same grid

        """
        super().__init__(models=models, grid=grid, data=data, name=name)
        self.filters = filters
        #self.bounds = limits_ranges
        self.mask = mask
        #if self.mask is not None: 
        #    print(f"Masked pixels is {np.sum(self.mask)}")
 

    def _residuals(self):
        resid_list = []
        for fname in self.filters:
            obs = self.data[fname]
            err = self.data.get(fname + '_err', None)
            psf = self.data.get(fname + '_psf', None)

            sigma2 = np.asarray(err) ** 2 if err is not None else np.ones_like(obs)
            sigma2 = np.where(sigma2 <= 0, 1.0, sigma2)

            model_img = self.models[self.filters.index(fname)](self.grid)
            if psf is not None:
                model_img = convolve(model_img, psf / np.sum(psf))  # normalize PSF flux
            r = (obs - model_img) / np.sqrt(sigma2)

            if self.mask is not None:
                r = r[~self.mask]
            resid_list.append(r.ravel())

        return np.concatenate(resid_list)


def fit_sersic2d_simple(data, error=None, reff0=10., n0=1., ellip0=0.3, 
                        theta0=0., grid2d=None, 
                        x0=0, y0=0,
                        beam=None, debug=False,
                        fitter='minimizer', pixelsize=1,
                        cmap='bone', mask=None,
                        plot=True, return_fit_images=False,):
    """fig 2d Gaussian to get the global geometry
    """
    ny, nx = data.shape
    if error is None:
        error = 1 # all pixel share same weights

    image_scale = 0.01*np.max(data)
    image_norm = data / image_scale

    # set up the grid
    if grid2d is None:
        grid2d = Grid2D(size=nx, pixelsize=pixelsize)
    sersic_2d_model = Sersic2D(
        amplitude=10, reff=reff0, n=n0,
        x0=x0, y0=y0, ellip=ellip0, theta=theta0,
    )
    sersic_2d_model.set_bounds({
        'amplitude': (0, 100), 'reff': (pixelsize, 400*pixelsize), 'n': (0.1, 8),
        'x0': (-10, 10), 'y0': (-10, 10),
        'ellip': (0, 1), 'theta': (-0.5 * np.pi, 0.5 * np.pi),
    })

    models = sersic_2d_model
    if debug:
        print('Initial parameters', models.parameters)
        print('Initial limits', models.bounds)
    model_data = models(grid2d)
    initial_guess = models.parameters
    # set up the fitter
    if fitter == 'minimizer':
        fitter = MinimizeFitter(models, grid2d,
                                {'var':grid2d.meshgrid, 'data':image_norm}, 
                                mask=mask)
        #fitter.auto_run(debug=debug)
        fitter.run(debug=debug)
        best_fit = fitter.best_fit
        best_fit_error = fitter.best_fit_error
        if best_fit_error is None:
            best_fit_error = np.zeros_like(best_fit)
    elif fitter == 'mcmc':
        fitter = EmceeFitter(models, grid2d, {'var':grid2d.meshgrid, 'data':image_norm})
        fitter.run(progress=True, steps=1000)
        if plot:
            fitter.plot()
        samples = fitter.samples(discard=200, flat=True)
        best_fit = np.percentile(samples, 50, axis=0)
        best_fit_mcmc_low = best_fit_mcmc - np.percentile(samples, 16, axis=0)
        best_fit_mcmc_up = np.percentile(samples, 84, axis=0) - best_fit_mcmc
        best_fit_error = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))

    models.update_parameters(dict(zip(initial_guess.keys(), best_fit)))
    model_image = models(grid2d)
    residual = image_norm - model_image

    if debug:
        fig, ax = plt.subplots(1,3, figsize=(12,3))
        ax[0].imshow(image_norm, origin='lower')
        if model_image is not None:
            ax[1].imshow(model_image, origin='lower')
        if residual is not None:
            ax[2].imshow(residual, origin='lower')

    if fitter == 'mcmc':
        fitres = dict(zip(initial_guess.keys(), zip(best_fit, best_fit_error)))
    else:
        fitres = dict(zip(initial_guess.keys(), zip(best_fit, best_fit_error)))

    if not return_fit_images:
        return fitres
    else:
        return fitres, model_image*image_scale, residual*image_scale

def fit_galaxy_center(image, radius=8, mask=None,
                      debug=False, plot=True):
    """fit the center of galaxy based on a simple 2D-Sersic fitting
    """
    yshape, xshape = image.shape
    xgrid, ygrid = np.meshgrid((np.arange(0, xshape)),
                              (np.arange(0, yshape)))
    xref, yref = 0.5*xshape-0.5, 0.5*yshape-0.5
    center_mask = (((xgrid - xref)**2 + (ygrid - yref)**2) > radius**2)

    if mask is not None:
        disk_mask = center_mask | mask
    else:
        disk_mask = center_mask
    # fit the center
    _bulge_fit = fit_sersic2d_simple(image, mask=disk_mask)
    if debug:
        print(_bulge_fit)
    fitted_center_params = {'x0': _bulge_fit['Sersic2D.x0'][0],
                    'y0': _bulge_fit['Sersic2D.y0'][0]}
    center = [xref+fitted_center_params['x0'], yref+fitted_center_params['y0']]
    if debug:
        print(fitted_center_params)
    if plot:
        fig, ax = plt.subplots(1,1, figsize=(6,5))
        ax.imshow(image, origin='lower')
        ax.imshow(disk_mask, origin='lower', alpha=0.4)

        axins = ax.inset_axes(
                    [0, 0, 0.3, 0.3],
                    xlim=(center[0]-0.02*xshape, center[0]+0.02*xshape), 
                    ylim=(center[1]-0.02*yshape, center[1]+0.02*yshape), 
                    xticklabels=[], yticklabels=[])
        axins.imshow(image, origin="lower")
        axins.plot(center[0], center[1], 'x', color='red')
    return fitted_center_params

def fit_galaxy_with_psf(models, data, error=None, psf=None, mask=None,
                        fitter='minimizer', mode='auto',
                        return_fit_images=False, fitsfile=None, pixelsize=1, 
                        debug=False, plot=True, figfile=None):
    """generic function to fit galaxy with composite models
    """
    ny, nx = data.shape
    if error is None: error = 1
    
    grid2d = Grid2D(size=(ny, nx), pixelsize=pixelsize)
    # set up the fitter
    if debug:
        print('Initial guess:', models.parameters)
    initial_guess = models.parameters
    if fitter == 'minimizer':
        data_fitter = MinimizeGalaxyFitter(
                models, grid2d,
                {'var':grid2d.meshgrid, 'data':data, 'data_err':error}, 
                psf, mask)
        if mode == 'auto':
            data_fitter.auto_run(debug=debug)
        elif mode == 'de':
            data_fitter.run_de(debug=debug)
        else:
            data_fitter.run(debug=debug)
        best_fit = data_fitter.best_fit
        best_fit_error = data_fitter.best_fit_error
        if best_fit_error is None:
            best_fit_error = np.zeros_like(best_fit)
    elif fitter == 'mcmc':
        raise ValueError('Unimplemented!')

    models.update_parameters(dict(zip(initial_guess.keys(), best_fit)))
    model_image = models(grid2d)
    model_image_convolve = convolve(model_image, psf)
    residual = data - model_image_convolve

    if debug:
        fig, ax = plt.subplots(1,4, figsize=(16,3))
        ax[0].imshow(data, origin='lower')
        ax[0].text(.5,.9, 'Data', horizontalalignment='center', 
                   transform=ax[0].transAxes, fontsize=10, color='white')
        if model_image is not None:
            ax[1].imshow(model_image, origin='lower')
            ax[1].text(.5,.9, 'Model', horizontalalignment='center', 
                       transform=ax[1].transAxes, fontsize=10, color='white')
            ax[2].imshow(model_image_convolve, origin='lower')
            ax[2].text(.5,.9, 'Model convolved', horizontalalignment='center', 
                       transform=ax[2].transAxes, fontsize=10, color='white')
        if residual is not None:
            ax[3].imshow(residual, origin='lower')
            ax[3].text(.5,.9, 'Residual', horizontalalignment='center', 
                       transform=ax[3].transAxes, fontsize=10, color='white')
        for ax_i in ax:
            ax_i.axis('off')
        fig.subplots_adjust(wspace=0.)
        if figfile is not None:
            fig.savefig(figfile, bbox_inches='tight', dpi=200)

    if fitter == 'mcmc':
        fitdict = dict(zip(initial_guess.keys(), zip(best_fit, best_fit_error)))
    else:
        fitdict = dict(zip(initial_guess.keys(), zip(best_fit, best_fit_error)))
    fitres = FittedModel.from_minimize_fitter(data_fitter)
    # save the fit into fits file
    if fitsfile is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            hdr = fits.Header()
            for item, value in fitdict.items():
                name = item.replace('.','_')
                hdr[name] = value[0]
                hdr[name+'_err'] = value[1]
            hdr['COMMENT'] = "Best fit from Galaxy Model"
            primary_hdu = fits.PrimaryHDU(header=hdr, data=data)
            model_hdu = fits.ImageHDU(model_image, name="Model")
            model_convolved_hdu = fits.ImageHDU(model_image_convolve, 
                                                name="Model_convolved")
            residual_hdu = fits.ImageHDU(residual, name="Residual")
            hdus = fits.HDUList([primary_hdu, model_hdu, model_convolved_hdu, residual_hdu])
            hdus.writeto(fitsfile, overwrite=True)
    
    if not return_fit_images:
        return fitres
    else:
        return fitres, model_image, model_image_convolve, residual

def fit_galaxy_multibands(models, grid2d, data, filters, mask=None,
                          fitter='minimizer', mode='auto',
                          return_fit_images=False, fitsfile=None, pixelsize=1, 
                          debug=False, plot=True, figfile=None):
    """generic function to fit galaxy with composite models


    Parameters
    ----------
    models : ``ModelList``
        The composed model from `ModelList`
    data : list
        The data from a list of filters, such as:
        data = {'F444W':data1, 'F444W_err': err1, 'F444W_psf': psf1,
                'F277W':data2, 'F277W_err': err2, 'F277W_psf': psf2,
        }
    filters : list
        A list of available filters
        filters = ['F444W', 'F277W']
    grid2d: ``Grid``
        The 2d grid from model_tils.Grid2D
    mask : ndarray
        The 2d mask apply to all the images

    """
    # set up the fitter
    nfilters = len(filters)
    if debug:
        initial_guess = models.parameters
        print('Initial guess:', models.parameters)
    if fitter == 'minimizer':
        data_fitter = MinimizeGalaxyMultipleFitter(
                models, grid2d, data, filters,
                mask)
        if mode == 'auto':
            data_fitter.auto_run(debug=debug)
        elif mode == 'de':
            data_fitter.run_de(debug=debug)
        else:
            data_fitter.run(debug=debug)
        best_fit = data_fitter.best_fit
        best_fit_error = data_fitter.best_fit_error
        if best_fit_error is None:
            best_fit_error = np.zeros_like(best_fit)
    elif fitter == 'mcmc':
        raise ValueError('Unimplemented!')

    models.update_parameters(best_fit)
    model_image = models(grid2d)
    model_images = []
    model_images_convolved = []
    model_residuals = []
    for i in range(nfilters):
        fname = filters[i]
        model_image = models[i](grid2d)
        model_image_convolve = convolve(model_image, data[fname+'_psf'])
        residual = data[fname] - model_image_convolve
        model_images.append(model_image)
        model_images_convolved.append(model_image_convolve)
        model_residuals.append(residual)

    if debug:
        fig, ax = plt.subplots(nfilters,4, figsize=(16,3*nfilters))
        for i in range(nfilters):
            ax[i,0].imshow(data, origin='lower')
            ax[i,0].text(.5,.9, 'Data', horizontalalignment='center', 
                   transform=ax[i,0].transAxes, fontsize=10, color='white')
            ax[i,1].imshow(model_image, origin='lower')
            ax[i,1].text(.5,.9, 'Model', horizontalalignment='center', 
                       transform=ax[i,1].transAxes, fontsize=10, color='white')
            ax[i,2].imshow(model_image_convolve, origin='lower')
            ax[i,2].text(.5,.9, 'Model convolved', horizontalalignment='center', 
                       transform=ax[i,2].transAxes, fontsize=10, color='white')
            ax[i,3].imshow(residual, origin='lower')
            ax[i,3].text(.5,.9, 'Residual', horizontalalignment='center', 
                       transform=ax[i,3].transAxes, fontsize=10, color='white')
            for ax_i in ax[i,:]:
                ax_i.axis('off')
        fig.subplots_adjust(wspace=0., hspace=0.1)
        if figfile is not None:
            fig.savefig(figfile, bbox_inches='tight', dpi=200)

    fitdict = dict(zip(models.parameters.keys(), zip(best_fit, best_fit_error)))
    # save the fit into fits file
    if fitsfile is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            hdr = fits.Header()
            for item, value in models.items():
                name = item.replace('.','_')
                hdr[name] = value[0]
                hdr[name+'_err'] = value[1]
            hdr['COMMENT'] = "Best fit from Galaxy Model"
            primary_hdu = fits.PrimaryHDU(header=hdr,)
            hdu_list = [primary_hdu,]
            for i in range(nfilters):
                fname = filters[i]
                model_hdu = fits.ImageHDU(model_image, name=fname+"_model")
                model_convolved_hdu = fits.ImageHDU(model_image_convolve, 
                                                name=fname+"_model_convolved")
                residual_hdu = fits.ImageHDU(residual, name=fname+"_residual")
                hdu_list.append(model_hdu, model_convolved_hdu, residual_hdu)
            hdus = fits.HDUList(hdu_list)
            hdus.writeto(fitsfile, overwrite=True)

    if not return_fit_images:
        return fitdict
    else:
        return fitdist, model_images, model_image_convolves, model_residuals


