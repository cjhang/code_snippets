#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2024-11-08: modeling become a separate module, v0.1
    2026-07-10: redesigned with CompoundModel, Prior hierarchy, cleaner Parameter, v0.2
"""

from __future__ import annotations
__version__ = '0.2.0'

import operator
import time
import warnings
from functools import reduce, cached_property
from multiprocessing import Pool
from typing import Callable, Optional

import emcee
import numpy as np
from numpy import linalg
from scipy import interpolate, optimize, special, stats
from matplotlib import pyplot as plt
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as cosm



###############################################
# Parameter & Prior
###############################################

class Parameter:
    """A single fittable value with metadata.

    Parameters
    ----------
    value : value of the parameter.
    fixed : whether the parameter is frozen during fitting.
    bounds : (lower, upper) hard bounds.
    prior : Prior distribution attached to this parameter.
    """
    # __slots__ = ("value", "fixed", "bounds", "prior")

    def __new__(
        cls, 
        value: float | int | Parameter = None,  
        *args, **kwargs
    ) -> Parameter:
        """
        Parameters
        ----------
        value : If a Parameter instance is passed, it is reused as-is (pass-through),
            enabling parameter tying at construction time. In that case, all
            other arguments (fixed, bounds, prior) are IGNORED -- configure
            the shared Parameter object directly instead.
        """
        if isinstance(value, Parameter):
            return value
        return super().__new__(cls)

    def __init__(self, value, fixed=False, bounds=(-np.inf, np.inf), prior=None):
        if isinstance(value, Parameter):
            return
        self.value = value
        self.fixed = fixed
        self.bounds = bounds
        self.prior = prior

    def log_prior(self, v):
        lo, hi = self.bounds
        if not (lo <= v <= hi):
            return -np.inf
        return 0.0 if self.prior is None else self.prior.logpdf(v)

    def __repr__(self):
        return f"Parameter(value={self.value}, bounds={self.bounds}, fixed={self.fixed})"

class DerivedParameter(Parameter):
    """
    A read-only Parameter computed from one or more source Parameters
    via a user-supplied function, instead of being independently fit.

    Always fixed=True -- automatically excluded from the free-parameter
    list, since its value is fully determined by its source(s).
    """

    def __new__(cls, func=None, *sources):
        # bypass Parameter.__new__'s "pass-through if given a Parameter"
        # logic -- here the first positional arg is a function, not a value.
        return object.__new__(cls)

    def __init__(self, func, *sources):
        self._func = func
        self._sources = sources
        self.fixed = True
        self.bounds = None
        self.prior = None
    
    def log_prior(self, v):
        # Derived parameters are fixed; they don't contribute to priors.
        return 0.0

    @property
    def value(self):
        return self._func(*(s.value for s in self._sources))

    @value.setter
    def value(self, v):
        raise AttributeError(
            "Cannot set .value directly on a DerivedParameter; it is "
            "computed automatically from its source parameter(s)."
        )

    def __repr__(self):
        return f"DerivedParameter(value={self.value!r})"

class Prior:
    """Base class for prior distributions.

    Subclasses must implement ``logpdf`` and ``ppf``.
    """
    def logpdf(self, v: float) -> float:
        """Log of the probability density function."""
        raise NotImplementedError

    def ppf(self, u: float) -> float:
        """Percent-point function (inverse CDF) for nested samplers."""
        raise NotImplementedError

class UniformPrior(Prior):
    """Uniform prior on [low, high]."""
    def __init__(self, low: float, high: float):
        self.low, self.high = low, high
        self._dist = stats.uniform(loc=low, scale=high - low)

    def logpdf(self, v: float) -> float:
        return self._dist.logpdf(v)  # type: ignore[no-any-return]

    def ppf(self, u: float) -> float:
        return self._dist.ppf(u)  # type: ignore[no-any-return]

class GaussianPrior(Prior):
    """Gaussian (normal) prior with mean *mu* and standard deviation *sigma*."""
    def __init__(self, mu: float, sigma: float):
        self.mu, self.sigma = mu, sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    def logpdf(self, v: float) -> float:
        return self._dist.logpdf(v)  # type: ignore[no-any-return]

    def ppf(self, u: float) -> float:
        return self._dist.ppf(u)  # type: ignore[no-any-return]


###############################################
# Axis and Grid
###############################################

class Axis:
    """
    A single coordinate axis: either explicit data, or lazily generated
    from (size, pixelsize, center) via the standard pixel-grid convention:

        x_i = (i - size/2 + 0.5) * pixelsize + center
    """

    def __init__(self, data=None, size=None, pixelsize=1.0, center=0.0):
        self._data = None
        if data is not None:
            self.data = data          # setter infers size/pixelsize/center
        else:
            if size is None:
                raise ValueError("Must provide either `data` or `size`.")
            self.size = size
            self.pixelsize = pixelsize
            self.center = center

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = (
                (np.arange(self.size) - self.size * 0.5 + 0.5) * self.pixelsize
                + self.center
            )
        return self._data

    @data.setter
    def data(self, arr):
        arr = np.asarray(arr)
        self._data = arr
        self.size = len(arr)
        self.pixelsize = float(arr[1] - arr[0]) if len(arr) > 1 else 1.0
        self.center = float(np.mean(arr))

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"Axis(size={self.size}, pixelsize={self.pixelsize}, center={self.center})"

class Grid:
    """
    N-dimensional coordinate grid, composed of an ordered list of Axis
    objects (fastest-varying last, matching array/FITS convention:
    (x,) for 1D, (y, x) for 2D, (z, y, x) for 3D).
    """

    def __init__(self, *axes: Axis):
        if not axes:
            raise ValueError("Grid requires at least one Axis.")
        self.axes = list(axes)
        self.ndim = len(axes)

    def _invalidate_cache(self):
        self.__dict__.pop('meshgrid', None)

    @property
    def shape(self):
        return tuple(ax.size for ax in self.axes)

    @property
    def grid(self):
        """Tuple of 1D coordinate arrays, ordered to match self.ndim."""
        return tuple(ax.data for ax in self.axes)

    @cached_property
    def meshgrid(self):
        return tuple(np.meshgrid(*self.grid, indexing='ij'))

    @property
    def pixelsize(self):
        values = {ax.pixelsize for ax in self.axes}
        if len(values) != 1:
            raise ValueError(
                "Axes have non-uniform pixel size; inspect grid.axes individually."
            )
        return values.pop()

    def __len__(self):
        return self.ndim

    def __getitem__(self, i):
        return self.axes[i].data

    def __repr__(self):
        return f"{type(self).__name__}(shape={self.shape})"

def _normalize_size(size, ndim):
    if size is None:
        return (None,) * ndim
    if isinstance(size, int):
        return (size,) * ndim
    if len(size) != ndim:
        raise ValueError(f"size must have length {ndim}")
    return tuple(size)

class Grid1D(Grid):
    def __init__(self, data=None, size=None, pixelsize=1.0, center=0.0):
        super().__init__(Axis(data=data, size=size, pixelsize=pixelsize, center=center),
                        )

class Grid2D(Grid):
    def __init__(self, size=None, pixelsize=1.0, center=(0., 0.)):
        ysize, xsize = _normalize_size(size, 2)
        ycenter, xcenter = center
        super().__init__(
            Axis(size=ysize, pixelsize=pixelsize, center=ycenter),
            Axis(size=xsize, pixelsize=pixelsize, center=xcenter),
        )

    # ---- radio-interferometry-specific Fourier grid, scoped to 2D ----
    @cached_property
    def uvgrid(self) -> np.ndarray:
        kmax = 0.5 / (self.axes[1].pixelsize * np.pi / 180 / 3600)  # arcsec -> rad
        return np.linspace(-kmax, kmax, self.axes[1].size)

class Grid3D(Grid):
    def __init__(self, size=None, pixelsize=1.0, center=(0., 0., 0.)):
        zsize, ysize, xsize = _normalize_size(size, 3)
        zc, yc, xc = center
        super().__init__(
            Axis(size=zsize, pixelsize=pixelsize, center=zc),
            Axis(size=ysize, pixelsize=pixelsize, center=yc),
            Axis(size=xsize, pixelsize=pixelsize, center=xc),
        )


###############################################
# Model base
###############################################

# ---------------- operator overloading, generated once ----------------
def _make_binary_op(sym: str):
    def op(self: Model, other) -> CompoundModel:
        other = other if isinstance(other, Model) else _Constant(other)
        return CompoundModel(self, other, sym)
    def rop(self: Model, other) -> CompoundModel:
        other = other if isinstance(other, Model) else _Constant(other)
        return CompoundModel(other, self, sym)
    return op, rop

class Model:
    """Base class for all models.

    Subclasses assign ``Parameter`` instances as attributes in ``__init__``
    and implement ``evaluate(*coords)``.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    def _iter_params(self):
        """Yield ``(owner_model, attr_name, Parameter)`` for each free parameter."""
        for attr, val in vars(self).items():
            if isinstance(val, Parameter) and not val.fixed:
                yield self, attr, val

    @property
    def submodels(self) -> list[Model]:
        return [self]

    def _params_map(self) -> dict[int, tuple[str, Parameter]]:
        """Return ``id(Parameter) -> (display_key, Parameter)``.

        Parameters that share the same Python object (via :meth:`tie`) are
        collapsed into a single entry.
        """
        out: dict[int, tuple[str, Parameter]] = {}
        seen_keys: dict[str, int] = {}
        for owner, attr, param in self._iter_params():
            pid = id(param)
            if pid in out:
                continue
            key = f"{owner.name}.{attr}"
            if key in seen_keys and seen_keys[key] != pid:
                warnings.warn(f"Different parameters share name '{key}'!")
            seen_keys[key] = pid
            out[pid] = (key, param)
        return out

    @property
    def parameters(self) -> dict[str, float]:
        """Return ``{name: value}`` for every free parameter."""
        return {k: p.value for k, p in self._params_map().values()}

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Return ``{name: (low, high)}`` for every free parameter."""
        return {k: p.bounds for k, p in self._params_map().values()}

    def update_parameters(self, params: dict[str, float] | np.ndarray | list) -> None:
        """Update parameter values from a dict or an array (in ``.parameters`` key order)."""
        pm = {k: p for k, p in self._params_map().values()}
        if not isinstance(params, dict):
            params = dict(zip(pm.keys(), params))
        for k, v in params.items():
            if k in pm:
                pm[k].value = v

    def get_priors(self, params: Optional[dict[str, float]] = None) -> float:
        """Return the sum of log-priors for the given (or current) parameter values."""
        pm = {k: p for k, p in self._params_map().values()}
        params = params if params is not None else {k: p.value for k, p in pm.items()}
        return sum(p.log_prior(params[k]) for k, p in pm.items() if k in params)

    # ---------- evaluation ----------
    def __call__(self, *args):
        if len(args) == 1 and hasattr(args[0], "ndim") and hasattr(args[0], "meshgrid"):
            grid = args[0]
            coords = (grid[0],) if grid.ndim == 1 else grid.meshgrid
            return self.evaluate(*coords)
        return self.evaluate(*args)

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        """Evaluate the model on coordinate arrays.  Override in subclasses."""
        raise NotImplementedError

    def __getitem__(self, i: int) -> Model:
        return self.submodels[i]

    # ---------- name resolution ----------
    def _resolve(self, path: str) -> tuple[Model, str]:
        """Resolve ``'bulge.x0'`` or ``'n'`` to ``(model, attr_name)``."""
        if '.' in path:
            model_name, attr = path.split('.', 1)
            for m in self.submodels:
                if m.name == model_name and isinstance(getattr(m, attr, None), Parameter):
                    return m, attr
            raise KeyError(f"Cannot resolve parameter path '{path}'")

        candidates = [
            (m, path) for m in self.submodels
            if isinstance(getattr(m, path, None), Parameter)
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise KeyError(f"No parameter named '{path}' found in this model.")
        names = [m.name for m, _ in candidates]
        raise ValueError(
            f"Ambiguous parameter name '{path}' found in {names}. "
            f"Use 'model_name.{path}' to disambiguate."
        )

    # ---------- fix / unfix ----------
    def fix(self, *names: str, **name_value_pairs: float) -> Model:
        """Freeze parameters.

        Examples
        --------
        model.fix('bulge.n', 'disk.n')               # freeze at current value
        model.fix(**{'bulge.n': 4.0, 'disk.reff': 10})  # set value AND freeze
        """
        for name in names:
            owner, attr = self._resolve(name)
            getattr(owner, attr).fixed = True
        for name, value in name_value_pairs.items():
            owner, attr = self._resolve(name)
            p = getattr(owner, attr)
            p.value = value
            p.fixed = True
        return self

    def unfix(self, *names: str) -> Model:
        """Unfreeze parameters."""
        for name in names:
            owner, attr = self._resolve(name)
            getattr(owner, attr).fixed = False
        return self

    def tie(self, *names: str) -> Model:
        """Tie two or more parameters so they share a single underlying value.

        Example
        -------
        model.tie('bulge.x0', 'disk.x0', 'bar.x0')
        """
        if len(names) < 2:
            raise ValueError("tie() requires at least two parameter names")
        owner0, attr0 = self._resolve(names[0])
        shared = getattr(owner0, attr0)
        for name in names[1:]:
            owner, attr = self._resolve(name)
            setattr(owner, attr, shared)
        return self

    # ---------- tie ----------
    def link(self, target, source, func=None):
        """
        Tie two parameters together.

        link target to source based on the given func
        the target name will become a DerivedParameter computed as
        func(source_value), and is excluded from fitting.

        Examples
        --------
        # Identity tie (unchanged behavior)
        model.link('bulge.x0', 'disk.x0') # same as tie

        # Functional tie: bar PA is always 90 deg offset from disk PA
        model.link('disk.theta', 'bar.theta', func=lambda t: t + np.pi / 2)
        
        Note
        ----
        lambda function is not pickable. When multiprocessing is used, use a dedicated
        function.
        """
        source_master = getattr(*self._resolve(source))
        target_owner, target_attr = self._resolve(target)
        setattr(target_owner, target_attr, DerivedParameter(func, source_master))
        return self

    def link_func(self, target, *sources, func):
        """
        General N-source functional tie: `target` becomes a
        DerivedParameter computed as func(*source_values), where each
        of `sources` remains an independent, free parameter.

        Example
        -------
        # bar_r amplitude = bar_g amplitude * color_ratio (both free)
        model.tie_func('bar_r.amplitude', 'bar_g.amplitude', 'color.ratio',
                        func=lambda a, r: a * r)

        Note
        ----
        lambda function is not pickable. When multiprocessing is used, use a dedicated
        function.
        """
        target_owner, target_attr = self._resolve(target)
        source_params = [getattr(*self._resolve(s)) for s in sources]
        setattr(target_owner, target_attr, DerivedParameter(func, *source_params))
        return self

    def set_bounds(self, name_bounds_pairs: dict[name, list]) -> Model:
        """Set hard bounds for one or more parameters.

        Example
        -------
        model.set_bounds({'bar.a_bar': (2, 50), 'bar.ellip': (0.0, 0.9)})
        """
        for name, bounds in name_bounds_pairs.items():
            owner, attr = self._resolve(name)
            getattr(owner, attr).bounds = tuple(bounds)
        return self

    def set_prior(self, name_prior_pairs: dict[name, Prior]) -> Model:
        """Attach Prior distributions to parameters.

        Example
        -------
        model.set_prior({'bulge.n': GaussianPrior(mu=4.0, sigma=0.5)})
        """
        for name, prior in name_prior_pairs.items():
            owner, attr = self._resolve(name)
            getattr(owner, attr).prior = prior
        return self

    def get_bounds(self) -> list[tuple[float, float]]:
        """Return an ordered list of bounds matching ``.parameters`` key order."""
        return list(self.bounds.values())

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Unit-cube → physical-value transform for nested samplers."""
        result = []
        for (_key, p), ui in zip(self._params_map().values(), u):
            if p.prior is not None:
                result.append(p.prior.ppf(ui))
            else:
                lo, hi = p.bounds
                result.append(lo + ui * (hi - lo))
        return np.array(result)

Model.__add__,      Model.__radd__      = _make_binary_op('+')
Model.__sub__,      Model.__rsub__      = _make_binary_op('-')
Model.__mul__,      Model.__rmul__      = _make_binary_op('*')
Model.__truediv__,  Model.__rtruediv__  = _make_binary_op('/')


class _Constant(Model):
    """Internal: wraps a scalar constant for ``model + 5.0`` style expressions."""
    def __init__(self, value: float) -> None:
        super().__init__(name="const")
        self.value = value

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        return np.full_like(args[0], self.value) if args else np.array(self.value)

    def _iter_params(self):
        return iter(())

class CompoundModel(Model):
    """Model formed by combining two models with an arithmetic operator.

    Created automatically via ``model1 + model2``, ``model1 * model2``, etc.
    """

    _ops = {
        '+': operator.add, '-': operator.sub,
        '*': operator.mul, '/': operator.truediv,
    }

    def __init__(
        self,
        left: Model,
        right: Model,
        op: str,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.left, self.right, self.op = left, right, op
        self._disambiguate_names()

    def _disambiguate_names(self) -> None:
        seen: dict[str, int] = {}
        for m in self.submodels:
            seen[m.name] = seen.get(m.name, 0) + 1
            if seen[m.name] > 1:
                m.name = f"{m.name}_{seen[m.name] - 1}"

    @property
    def submodels(self) -> list[Model]:
        return self.left.submodels + self.right.submodels

    def _iter_params(self):
        for m in self.submodels:
            yield from m._iter_params()

    def evaluate(self, *args: np.ndarray) -> np.ndarray:
        return self._ops[self.op](self.left(*args), self.right(*args))

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"

    def summary(self) -> None:
        """Print a table of all parameters with values, fixed status, and ties."""
        seen: dict[int, str] = {}
        lines: list[str] = []
        for m in self.submodels:
            for attr, p in vars(m).items():
                if not isinstance(p, Parameter):
                    continue
                pid = id(p)
                tied_note = ""
                if pid in seen:
                    tied_note = f"  (tied to {seen[pid]})"
                else:
                    seen[pid] = f"{m.name}.{attr}"
                status = "FIXED" if p.fixed else "free "
                lines.append(
                    f"{m.name}.{attr:<12} = {p.value!r:>10}  [{status}]{tied_note}"
                )
        print("\n".join(lines))

def join_models(models: list[Model]) -> Model:
    """Combine a list of models via addition (replaces the old ``Models([...])`` API)."""
    return reduce(operator.add, models)


###############################################
# Concrete model classes
###############################################
def gaussian_2d(
    grid: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    y_sigma: float,
    x_sigma: float,
    y0: float,
    x0: float,
    theta: float,
) -> np.ndarray:
    """Evaluate a 2D Gaussian on a coordinate grid.

    Parameters
    ----------
    grid : (y, x) meshgrid tuple.
    amplitude : peak value.
    y_sigma, x_sigma : standard deviations along major / minor axes.
    y0, x0 : centre coordinates.
    theta : position angle in radians.

    Returns
    -------
    2D array of the Gaussian evaluated at each pixel.
    """
    y, x = grid
    a = np.cos(theta)**2 / (2 * x_sigma**2) + np.sin(theta)**2 / (2 * y_sigma**2)
    b = np.sin(2 * theta) / (2 * x_sigma**2) - np.sin(2 * theta) / (2 * y_sigma**2)
    c = np.sin(theta)**2 / (2 * x_sigma**2) + np.cos(theta)**2 / (2 * y_sigma**2)
    return amplitude * np.exp(-a * (x - x0)**2 - b * (x - x0) * (y - y0) - c * (y - y0)**2)

def blackbody_nu(nu: np.ndarray, temperature: float) -> np.ndarray:
    """Blackbody spectral radiance per unit frequency.

    Parameters
    ----------
    nu : frequency in Hz.
    temperature : temperature in Kelvin.

    Returns
    -------
    B_nu in erg / s / sr / cm^2 / Hz.
    """
    const_h = const.h.cgs.value
    const_c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    return (
        2 * const_h * nu**3 / const_c**2
        * (np.exp(const_h * nu / (k_B * temperature)) - 1)**-1
    )

def blackbody_lambda(lam: np.ndarray, temperature: float) -> np.ndarray:
    """Blackbody spectral radiance per unit wavelength.

    Parameters
    ----------
    lam : wavelength in cm.
    temperature : temperature in Kelvin.

    Returns
    -------
    B_lambda in erg / s / sr / cm^2 / cm.
    """
    const_h = const.h.cgs.value
    const_c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    return (
        2 * const_h * const_c**2 / lam**5
        / (np.exp(const_h * const_c / (lam * k_B * temperature)) - 1)
    )

def modified_blackbody_nu(
    nu: np.ndarray,
    temperature: Optional[float] = None,
    z: float = 0,
    beta: float = 2.0,
    dustmass: Optional[float] = None,
    area: Optional[float] = None,
) -> np.ndarray:
    """Flux density of a modified blackbody.

    Parameters
    ----------
    nu : rest-frame frequency in Hz.
    temperature : dust temperature in K.
    z : redshift.
    beta : emissivity index.
    dustmass : dust mass in g.
    area : emitting area in cm^2.

    Returns
    -------
    Flux density in cgs units.
    """
    kappa0 = 0.45  # cm^2 / g
    nu0 = 250 * u.GHz.to(u.Hz)
    Tcmb = 2.73
    D_L = cosm.luminosity_distance(z).to(u.cm).value
    Td = temperature
    tau_nu = dustmass / area * kappa0 * (nu / nu0)**beta
    B_Td = blackbody_nu(nu, Td)
    B_Tcmb = blackbody_nu(nu, Tcmb)
    return (1 + z) * area / D_L**2 * (1 - np.exp(-tau_nu)) * (B_Td - B_Tcmb)

def check_regular_grid(arr: np.ndarray) -> bool:
    """Return True if *arr* is regularly spaced."""
    arr_diff = np.diff(arr)
    return bool(np.all(np.abs(arr_diff - arr_diff[0]) <= np.finfo(arr_diff.dtype).eps))

class Line1D(Model):
    """Straight line: ``y = m * x + b``.

    Parameters
    ----------
    m : float
        Slope.
    b : float
        Intercept.
    """

    def __init__(self, m=1.0, b=0.0, name=None):
        super().__init__(name=name)
        self.m = Parameter(m, bounds=(-np.inf, np.inf))
        self.b = Parameter(b, bounds=(-np.inf, np.inf))

    def evaluate(self, x):
        """Evaluate the line on coordinate array *x*."""
        return self.m.value * x + self.b.value

class ScaleUncertainty(Model):
    """Pseudo-model for extra variance scaling.
    ``evaluate`` always returns 0.

    The *factor* parameter is used by the likelihood to inflate uncertainties:

        sigma^2 += model^2 * exp(2 * log_f)

    Parameters
    ----------
    factor : float
        Log of the uncertainty scaling factor.
    """

    def __init__(self, factor=0.0, name=None):
        super().__init__(name=name)
        self.factor = Parameter(factor, bounds=(-np.inf, np.inf))

    def evaluate(self, *args):
        return np.zeros_like(args[0])

class Gaussian1D(Model):
    """One-dimensional Gaussian plus an optional continuum.

        f(x) = amplitude * exp(-0.5 * ((x - mean) / sigma)^2) + cont

    Parameters
    ----------
    amplitude : float
        Peak amplitude above the continuum.
    mean : float
        Central position.
    sigma : float
        Standard deviation (width).
    cont : float or Parameter
        Continuum offset. If a plain float is given it is stored as a
        fixed parameter.
    """

    def __init__(self, amplitude=1.0, mean=0.0, sigma=1.0, cont=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0.0, np.inf))
        self.mean      = Parameter(mean,      bounds=(-np.inf, np.inf))
        self.sigma     = Parameter(sigma,     bounds=(1e-3, np.inf))
        if isinstance(cont, Parameter):
            self.cont = cont
        else:
            self.cont = Parameter(cont, fixed=True)

    def evaluate(self, x):
        """Evaluate the Gaussian on coordinate array *x*."""
        amp   = self.amplitude.value
        mu    = self.mean.value
        sig   = self.sigma.value
        return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2) + self.cont.value

class Sersic1D(Model):
    """One-dimensional Sérsic profile.

        I(x) = amplitude * exp{ -b_n [(|x - x0| / reff)^(1/n) - 1] }

    Parameters
    ----------
    amplitude : float
        Surface brightness at the effective radius.
    reff : float
        Effective (half-light) radius.
    n : float
        Sérsic index.
    x0 : float
        Profile centre.
    """

    def __init__(self, amplitude=1.0, reff=1.0, n=2.0, x0=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0.0, np.inf))
        self.reff      = Parameter(reff,      bounds=(1e-3, np.inf))
        self.n         = Parameter(n,         bounds=(0.1, 10.0))
        self.x0        = Parameter(x0,        bounds=(-np.inf, np.inf))

    def evaluate(self, x):
        bn = special.gammaincinv(2.0 * self.n.value, 0.5)
        r = np.abs(x - self.x0.value) / self.reff.value
        return self.amplitude.value * np.exp(-bn * (r ** (1.0 / self.n.value) - 1.0))

class Gaussian2D(Model):
    """Two-dimensional Gaussian.

        f(x, y) = amplitude * exp(-a*(x-x0)^2 - b*(x-x0)*(y-y0) - c*(y-y0)^2)

    where *a*, *b*, *c* are computed from *x_sigma*, *y_sigma*, and *theta*.

    Parameters
    ----------
    amplitude : float
        Peak value.
    x0, y0 : float
        Centre coordinates.
    x_sigma, y_sigma : float
        Standard deviations along the principal axes before rotation.
    theta : float
        Position angle in radians, CCW from +x axis.
    """

    def __init__(self, amplitude=1.0, x0=0.0, y0=0.0, x_sigma=1.0, y_sigma=1.0,
                 theta=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0.0, np.inf))
        self.x0        = Parameter(x0,        bounds=(-np.inf, np.inf))
        self.y0        = Parameter(y0,        bounds=(-np.inf, np.inf))
        self.x_sigma   = Parameter(x_sigma,   bounds=(1e-3, np.inf))
        self.y_sigma   = Parameter(y_sigma,   bounds=(1e-3, np.inf))
        self.theta     = Parameter(theta,     bounds=(-np.pi, np.pi))

    def evaluate(self, y, x):
        """Evaluate the 2D Gaussian on coordinate arrays *y*, *x*."""
        amp   = self.amplitude.value
        y0    = self.y0.value
        x0    = self.x0.value
        y_sig = self.y_sigma.value
        x_sig = self.x_sigma.value
        theta = self.theta.value

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        a = cos_t**2 / (2 * x_sig**2) + sin_t**2 / (2 * y_sig**2)
        b = sin_t * cos_t / x_sig**2 - sin_t * cos_t / y_sig**2
        c = sin_t**2 / (2 * x_sig**2) + cos_t**2 / (2 * y_sig**2)

        dx, dy = x - x0, y - y0
        return amp * np.exp(-a * dx**2 - b * dx * dy - c * dy**2)

class Sersic2D(Model):
    """Two-dimensional Sérsic profile:

        I(r) = amplitude * exp{ -b_n [ (r / reff)^(1/n) - 1 ] }

    where *r* is the elliptical radius computed from the rotated,
    ellipticity-scaled coordinate system.

    Parameters
    ----------
    amplitude : float
        Surface brightness at r = reff.
    reff : float
        Effective (half-light) radius.
    n : float
        Sérsic index (n=4 -> de Vaucouleurs, n=1 -> exponential disk).
    x0, y0 : float
        Centre coordinates.
    ellip : float
        Ellipticity, 1 - b/a.
    theta : float
        Position angle in radians, CCW from +x axis.
    """

    def __init__(self, amplitude=1.0, reff=10.0, n=2.0, x0=0.0, y0=0.0,
                 ellip=0.3, theta=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0.0, np.inf))
        self.reff      = Parameter(reff,      bounds=(0.0, np.inf))
        self.n         = Parameter(n,         bounds=(0.1, 10.0))
        self.x0        = Parameter(x0,        bounds=(-np.inf, np.inf))
        self.y0        = Parameter(y0,        bounds=(-np.inf, np.inf))
        self.ellip     = Parameter(ellip,     bounds=(0.001, 0.999))
        self.theta     = Parameter(theta,     bounds=(-np.pi/2, np.pi/2))

    def evaluate(self, y, x):
        """Evaluate the Sérsic profile on coordinate arrays *y*, *x*."""
        amplitude = self.amplitude.value
        reff      = self.reff.value
        n         = self.n.value
        x0        = self.x0.value
        y0        = self.y0.value
        ellip     = self.ellip.value
        theta     = self.theta.value

        bn = special.gammaincinv(2.0 * n, 0.5)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        dx, dy = x - x0, y - y0
        x_maj =  dx * cos_theta + dy * sin_theta
        x_min = -dx * sin_theta + dy * cos_theta

        b = (1.0 - ellip) * reff
        r = np.sqrt((x_maj / reff) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-bn * (r ** (1.0 / n) - 1.0))

class Blackbody(Model):
    """Blackbody spectrum as a function of frequency.

        B_nu = 2 h nu^3 / c^2  *  (exp(h nu / k_B T) - 1)^-1

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    """

    def __init__(self, temperature=10.0, name=None):
        super().__init__(name=name)
        self.temperature = Parameter(temperature, bounds=(1e-3, np.inf))

    def evaluate(self, *args):
        """*args* is nu in GHz (or a Grid dispatched by base ``__call__``).

        Returns
        -------
        B_nu in erg / s / sr / cm^2 / Hz.
        """
        nu = args[0] if len(args) == 1 else args[1]
        return blackbody_nu(nu * 1e9, self.temperature.value)

class MBlackbody(Model):
    """Modified blackbody (grey-body) model.

    Parameters
    ----------
    temperature : float
        Dust temperature in K.
    z : float
        Redshift.
    beta : float
        Emissivity spectral index (1–2 typical).
    dustmass : float
        Dust mass in solar masses.
    area : float
        Emitting area in kpc^2.
    """

    def __init__(self, temperature=40.0, z=0.0, beta=2.0,
                 dustmass=1e10, area=1.0, name=None):
        super().__init__(name=name)
        self.temperature = Parameter(temperature, bounds=(1e-3, np.inf))
        self.z           = Parameter(z,           fixed=True)
        self.beta        = Parameter(beta,        fixed=True)
        self.dustmass    = Parameter(dustmass,    bounds=(0.0, np.inf))
        self.area        = Parameter(area,        fixed=True)

    def evaluate(self, *args):
        """*args* is nu in GHz (or a Grid dispatched by base ``__call__``).

        Returns
        -------
        Flux density in mJy.
        """
        nu = args[0] if len(args) == 1 else args[1]
        msun_to_g = 1.98841e33
        kpc2_to_cm2 = 9.521406e42
        mbb = modified_blackbody_nu(
            nu * 1e9,
            temperature=self.temperature.value,
            z=self.z.value,
            beta=self.beta.value,
            dustmass=self.dustmass.value * msun_to_g,
            area=self.area.value * kpc2_to_cm2,
        )
        return 1e26 * mbb

class MBlackbody_log(Model):
    """Modified blackbody with *log10(dustmass)* as the free parameter.

    This is identical to :class:`MBlackbody` except that *dustmass* is
    the base‑10 logarithm of the dust mass in solar masses, which helps
    when scanning over many orders of magnitude.

    Parameters
    ----------
    temperature : float
        Dust temperature in K.
    z : float
        Redshift.
    beta : float
        Emissivity spectral index (1–2 typical).
    dustmass : float
        log10(dust mass / Msun).
    area : float
        Emitting area in kpc^2.
    """

    def __init__(self, temperature=40.0, z=0.0, beta=2.0,
                 dustmass=10.0, area=1.0, name=None):
        super().__init__(name=name)
        self.temperature = Parameter(temperature, bounds=(1e-3, np.inf))
        self.z           = Parameter(z,           fixed=True)
        self.beta        = Parameter(beta,        fixed=True)
        self.dustmass    = Parameter(dustmass,    bounds=(0.0, np.inf))
        self.area        = Parameter(area,        fixed=True)

    def evaluate(self, *args):
        """*args* is nu in GHz (or a Grid dispatched by base ``__call__``).

        Returns
        -------
        Flux density in mJy.
        """
        nu = args[0] if len(args) == 1 else args[1]
        msun_to_g = 1.98841e33
        kpc2_to_cm2 = 9.521406e42
        mbb = modified_blackbody_nu(
            nu * 1e9,
            temperature=self.temperature.value,
            z=self.z.value,
            beta=self.beta.value,
            dustmass=10**self.dustmass.value * msun_to_g,
            area=self.area.value * kpc2_to_cm2,
        )
        return 1e26 * mbb


###############################################
# Log-probability / cost functions
###############################################

def emcee_log_probability(
    theta: np.ndarray,
    models: Model,
    grid: Grid,
    data: dict,
) -> float:
    """Log-probability function for ``emcee``.

    Parameters
    ----------
    theta : parameter vector in ``models.parameters`` key order.
    models : the combined model.
    grid : evaluation grid.
    data : dict with keys ``'var'``, ``'data'``, and optionally
           ``'var_err'``, ``'data_err'``.

    Returns
    -------
    Log-posterior (log-likelihood + log-prior), or -inf if out of bounds.
    """
    models.update_parameters(theta)
    params_priors = models.get_priors()
    if not np.isfinite(params_priors):
        return -np.inf

    model = models(grid)
    var = data['var']
    var_err = data.get('var_err', 0)
    obs = data['data']
    obs_err = data.get('data_err', 0)

    # ScaleUncertainty extra variance
    err2_addition = 0.
    for m in models.submodels:
        if isinstance(m, ScaleUncertainty):
            log_f = m.factor.value
            err2_addition = model**2 * np.exp(2 * log_f)

    sigma2 = var_err**2 + obs_err**2 + err2_addition

    # interpolate model grid → observed coordinates
    if obs.ndim == 1:
        if np.sum(np.abs(grid.grid[0] - var)) < 1e-6 * len(var):
            model_interp = model
        else:
            model_interp = interpolate.interpn(grid.grid, model, var)
    elif obs.ndim == 2:
        model_interp = interpolate.interpn(
            grid.grid, model,
            np.array([var[0].flatten(), var[1].flatten()]).T,
        ).reshape(obs.shape)
    elif obs.ndim == 3:
        model_interp = interpolate.interpn(
            grid.grid, model,
            np.array([var[0].flatten(), var[1].flatten(), var[2].flatten()]).T,
        ).reshape(obs.shape)
    else:
        model_interp = model

    diff2 = (obs - model_interp)**2 / sigma2
    return float(-0.5 * np.sum(diff2) + params_priors)


###############################################
# Fitters
###############################################

class Fitter:
    def __init__(self, models, grid, data, mask=None, name=None):
        self.models = models
        self.grid = grid
        self.data = data
        self.mask = mask
        self.name = name

        # cache whether data sits exactly on the grid (perf fast-path)
        self._var_is_grid = self._check_var_matches_grid()

    def _check_var_matches_grid(self):
        try:
            return all(
                np.allclose(v, g) for v, g in zip(
                    np.atleast_1d(self.data['var']), self.grid.grid
                )
            )
        except Exception:
            return False

    def _data_size(self):
        try:
            return np.asarray(self.data['data']).size
        except Exception:
            return 1

    @property
    def parameters(self):
        return self.models.parameters

    @property
    def bounds(self):
        return list(self.models.bounds.values())

    @property
    def keys(self):
        return list(self.parameters.keys())

    @property
    def n_params(self):
        return len(self.parameters)

    @property
    def initial_guess(self):
        return list(self.parameters.values())

    # ---- unified residual/likelihood machinery ----
    def _model_at_data(self):
        model = self.models(self.grid)
        if self._var_is_grid:
            # no interpolate needed between model grid and data grid
            return model
        else:
            # apply the linear interpolation
            var, obs = self.data['var'], self.data['data']
            if obs.ndim == 1:
                return interpolate.interpn(self.grid.grid, model, var)
            points = np.stack([v.ravel() for v in var], axis=-1)
            return interpolate.interpn(self.grid.grid, model, points).reshape(obs.shape)

    def _residuals(self):
        """Signed, sigma-normalized, mask-aware residuals."""
        obs = self.data['data']
        var_err = self.data.get('var_err', 0)
        obs_err = self.data.get('data_err', 0)
        sigma2 = np.asarray(var_err) ** 2 + np.asarray(obs_err) ** 2
        sigma2 = np.where(sigma2 <= 0, 1.0, sigma2)

        model_interp = self._model_at_data()
        resid = (obs - model_interp) / np.sqrt(sigma2)

        if self.mask is not None:
            resid = resid[~self.mask]
        return resid

    def _prior_residuals(self):
        """
        Pseudo-residuals for any Parameter with a Gaussian-type prior,
        expressed as (value - mu) / sigma. Squaring and summing these
        reproduces -2*log_prior (up to an additive constant), so they
        can be concatenated with data residuals for least_squares.

        Uniform/flat priors and hard bounds do NOT contribute residuals
        here -- bounds are instead enforced directly via the `bounds=`
        argument of the optimizer.
        """
        out = []
        for owner, attr, param in self.models._iter_params():
            prior = getattr(param, 'prior', None)
            if isinstance(prior, (GaussianPrior,)):
                mu = getattr(prior, 'mu', None)
                sigma = getattr(prior, 'sigma', None)
                if mu is not None and sigma:
                    out.append((param.value - mu) / sigma)
        return np.array(out)

    def _n_prior_residuals(self):
        return len(self._prior_residuals())

    def _calculate_diff(self):
        return self._residuals() ** 2

    def log_likelihood(self, theta=None):
        if theta is not None:
            self.models.update_parameters(theta)
        return -0.5 * np.sum(self._calculate_diff())

    def log_posterior(self, theta):
        self.models.update_parameters(theta)
        lp = self.models.get_priors()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood()

class MinimizeFitter(Fitter):
    def cost(self, theta: np.ndarray) -> float:
        """Scalar cost for `minimize`/`differential_evolution`: -log_posterior."""
        val = self.log_posterior(theta)
        return np.inf if not np.isfinite(val) else -val

    def vector_cost(self, theta: np.ndarray) -> np.ndarray:
        """
        Signed residual vector for `least_squares`.

        Concatenates:
          - data residuals: (obs - model) / sigma      [masked appropriately]
          - prior residuals: (value - mu) / sigma        [Gaussian priors only]

        Hard bounds are enforced separately via the optimizer's `bounds=`
        argument, NOT via residuals here.
        """
        self.models.update_parameters(theta)
        lp = self.models.get_priors()
        if not np.isfinite(lp):
            # Out-of-bounds: return a large but finite residual so the
            # optimizer can still compute a gradient direction away from it.
            n = self._data_size() + self._n_prior_residuals()
            return np.full(n, 1e5)
        data_resid = self._residuals().flatten()
        prior_resid = self._prior_residuals()
        return np.concatenate([data_resid, prior_resid])

    def run(self, debug=False):
        if debug:
            print(f'initial_guess: {self.initial_guess}; bounds: {self.bounds}')
        fit_result = optimize.minimize(self.cost, self.initial_guess, bounds=self.bounds)
        self.best_fit = fit_result.x
        self.best_fit_error = None
        if debug:
            print(f'fit result: {self.best_fit}')

    def run_de(self, debug=False, workers=-1):
        fit_result = optimize.differential_evolution(
            self.cost, bounds=self.bounds, x0=self.initial_guess, 
            workers=workers)
        self.best_fit = fit_result.x
        self.best_fit_error = None
        if debug:
            print(f'DE fit result: {self.best_fit}')

class EmceeFitter(Fitter):
    """MCMC fitter using ``emcee``.

    Parameters
    ----------
    models, grid, data, mask, name : see :class:`Fitter`.
    nwalkers : number of walkers (defaults to 4 * n_params).
    backend : path to an HDF5 backend file.
    backend_name : group name within the HDF5 file.
    reset : if True, reset any existing backend.
    """

    def __init__(
        self,
        models: Model,
        grid: Grid,
        data: dict,
        mask: Optional[np.ndarray] = None,
        nwalkers: Optional[int] = None,
        name: Optional[str] = None,
        backend: Optional[str] = None,
        backend_name: str = 'emceefitter',
        reset: bool = False,
    ) -> None:
        super().__init__(models=models, grid=grid, data=data, mask=mask, name=name)
        self.nwalkers = nwalkers or 4 * self.n_params

        self.iterations = 0
        self._hdf_backend: Optional[emcee.backends.HDFBackend] = None
        if backend is not None:
            self._hdf_backend = emcee.backends.HDFBackend(backend, name=backend_name)
            if reset:
                self._hdf_backend.reset(self.nwalkers, self.n_params)
            try:
                self.iterations = self._hdf_backend.iteration
            except AttributeError:
                self.iterations = 0

        if self.iterations > 0:
            print(f"Found existing steps={self.iterations}, continue")
            self.pos = self._hdf_backend.get_last_sample().coords
        else:
            self.pos = self._initial_walker_positions()

    def _initial_walker_positions(self) -> np.ndarray:
        """Scatter walkers around the initial guess, scaled by parameter
        range, with a safe fallback for unbounded (+/-inf) parameters."""
        bounds_arr = np.array(list(self.bounds), dtype=float)
        initial_guess = np.array(self.initial_guess, dtype=float)

        limits_diff = np.diff(bounds_arr, axis=1).flatten()
        # guard against +/-inf bounds -- fall back to a fraction of |x0|
        fallback = np.abs(initial_guess) * 0.1
        fallback = np.where(fallback > 0, fallback, 1e-3)
        limits_diff = np.where(np.isfinite(limits_diff), limits_diff, fallback)

        pos = (
            initial_guess
            + 1e-2 * limits_diff[None, :] * np.random.randn(self.nwalkers, self.n_params)
        )
        # clip to bounds so no walker starts outside a hard-bounded region
        lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
        return np.clip(pos, lo, hi)

    def _make_sampler(self, pool=None) -> emcee.EnsembleSampler:
        """Construct the EnsembleSampler using the unified log_posterior,
        which already incorporates mask, priors, and bounds correctly."""
        return emcee.EnsembleSampler(
            self.nwalkers, self.n_params, self.log_posterior,
            backend=self._hdf_backend, pool=pool,
        )

    # ------------------------------------------------------------
    def run(
        self,
        steps: int = 5000,
        progress: bool = True,
        force: bool = False,
    ) -> None:
        """Run MCMC for a fixed number of steps."""
        self.sampler = self._make_sampler()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='emcee')
            self.sampler.run_mcmc(
                self.pos, steps, progress=progress,
                skip_initial_state_check=force,
            )

    # ------------------------------------------------------------
    def auto_run(
        self,
        max_steps: int = 10000,
        progress: bool = True,
        check_steps: int = 1000,
        autocorr_multiplier: int = 50,
        debug: bool = False,
        rtol: float = 0.1
    ) -> bool:
        """
        Run MCMC with automatic convergence checking via integrated
        autocorrelation time, following the standard emcee recipe:

        Every `check_steps` iterations, estimate tau (autocorrelation
        time per parameter). Declare convergence once:
          (a) the chain is at least `check_steps` times longer than tau
              for every parameter, AND
          (b) tau itself has changed by < 1% since the last check.

        Parameters
        ----------
        check_steps : int
            How often (in steps) to *check* for convergence. Purely about
            checking frequency -- does NOT affect the convergence threshold.
        autocorr_multiplier : int
            Chain must be at least this many multiples of tau long to be
            considered converged (standard emcee recipe uses 50-100).
        rtol : float
            Maximum fractional change in tau between checks to be considered
            stable/converged.

        Returns
        -------
        converged : bool
            Whether the convergence criterion was met before max_steps.
        """

        if max_steps < check_steps:
            raise ValueError(f"max_steps ({max_steps}) must be >= check_steps ({check_steps})")

        self.sampler = self._make_sampler()

        index = 0
        old_tau = np.inf
        autocorr = np.empty(max_steps // check_steps + 1)
        converged = False

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='emcee')
            for _sample in self.sampler.sample(self.pos, iterations=max_steps, progress=progress):
                if self.sampler.iteration % check_steps:
                    continue

                # tol=0: return a best-effort estimate, never raise AutocorrError.
                # Reliability of the estimate is judged by our own criterion below.
                tau = self.sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                converged = np.all(tau * autocorr_multiplier < self.sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < rtol)

                if debug:
                    reduced_chi2 = self._reduced_chi2_at_current_state()
                    acc_frac = np.mean(self.sampler.acceptance_fraction)
                    print(
                        f"[iter {self.sampler.iteration}] "
                        f"mean_tau={np.mean(tau):.1f}  "
                        f"individual tau={dict(zip(self.keys, tau))} "
                        f"acc_frac={acc_frac:.3f}  "
                        f"converged={converged}  "
                        f"reduced_chi2(best-so-far)={reduced_chi2:.4g}"
                    )
                    if acc_frac < 0.05:
                        warnings.warn(
                            f"Very low acceptance fraction ({acc_frac:.3f}) at "
                            f"iteration {self.sampler.iteration} -- check walker "
                            f"initialization, bounds, or priors."
                        )

                if converged:
                    break
                old_tau = tau

        self.autocorr = autocorr[:index]
        self.converged = converged

        if not converged:
            warnings.warn(
                f"Chain did not converge within {max_steps} steps "
                f"(final mean tau ~ {np.mean(old_tau):.1f})."
            )

        return converged

    def _reduced_chi2_at_current_state(self) -> float:
        """Quick diagnostic: reduced chi^2 at the current best (max
        log-posterior) sample in the chain so far."""
        try:
            chain = self.sampler.get_chain(flat=True)
            log_prob = self.sampler.get_log_prob(flat=True)
            best_theta = chain[np.argmax(log_prob)]
            self.models.update_parameters(best_theta)
            chi2 = np.sum(self._calculate_diff())
            dof = max(self._data_size() - self.n_params, 1)
            return chi2 / dof
        except Exception as e:
            warnings.warn(f"Could not compute diagnostic reduced chi^2: {e}")
            return np.nan

    # ------------------------------------------------------------
    def multi_run(
        self,
        steps: int = 5000,
        ncores: Optional[int] = None,
        progress: bool = True,
    ) -> None:
        """Run MCMC with a multiprocessing pool."""
        with Pool(ncores) as pool:
            self.sampler = self._make_sampler(pool=pool)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='emcee')
                self.sampler.run_mcmc(self.pos, steps, progress=progress)

    # ------------------------------------------------------------
    def samples(
        self,
        discard: Optional[int] = None,
        thin: Optional[int] = None,
        flat: bool = True,
    ) -> np.ndarray:
        """Return the MCMC chain after burn-in and thinning."""
        tau = self.sampler.get_autocorr_time(tol=0)
        if discard is None:
            discard = int(2 * np.max(tau))
        if thin is None:
            thin = max(int(0.5 * np.min(tau)), 1)
        return self.sampler.get_chain(discard=discard, flat=flat, thin=thin)

    def best_fit_from_samples(self, **kwargs) -> dict:
        """
        Summarize posterior as median +/- 16th/84th percentile errors,
        analogous in spirit to MinimizeFitter.best_fit/best_fit_error.
        """
        samples = self.samples(**kwargs)
        lo, med, hi = np.percentile(samples, [16, 50, 84], axis=0)
        self.best_fit = med
        self.best_fit_error = np.vstack([med - lo, hi - med])  # asymmetric errors
        return dict(zip(self.keys, zip(med, med - lo, hi - med)))

    # ------------------------------------------------------------
    def plot(self, ax: Optional[list] = None) -> None:
        """Plot the MCMC chains for each parameter."""
        samples = self.sampler.get_chain()
        if ax is None:
            fig = plt.figure(figsize=(16, 4))
        for i in range(self.n_params):
            if ax is None:
                ax_i = plt.subplot2grid((self.n_params, 2), (i, 0), colspan=2)
            else:
                ax_i = ax[i]
            ax_i.plot(samples[:, :, i], "k", alpha=0.2)
            ax_i.set_xlim(0, len(samples))
            if i < self.n_params - 1:
                ax_i.axes.get_xaxis().set_ticks([])
            ax_i.text(0, 0.8, self.keys[i], ha='left', transform=ax_i.transAxes, fontsize=10)
        ax_i.set_xlabel('steps')
        if ax is None:
            plt.show()

    def corner_plot(self, truths: Optional[np.ndarray] = None) -> None:
        """Corner plot of the posterior samples."""
        try:
            import corner
        except ImportError:
            print("Install corner for corner_plot")
            return
        samples = self.samples()
        fig = corner.corner(samples, labels=self.keys, truths=truths)
        plt.show()

###############################################
# Test functions
###############################################

def test_line1d_fitting(plot: bool = False) -> None:
    m_true = -1.
    b_true = 4.
    N = 50
    x = np.sort(10 * np.random.rand(N))
    grid1d = Grid1D(data=x)
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true
    y += yerr * np.random.randn(N)

    line_model = Line1D(m=-0.8, b=2.0)
    line_model.set_bounds(**{'m': (-5., 0.5), 'b': (0., 10.)})
    models = join_models([line_model])
    data = {'var': grid1d[0], 'data': y, 'data_err': yerr}

    fitter_mcmc = EmceeFitter(models, grid1d, data)
    max_steps = 20000
    fitter_mcmc.auto_run(progress=True, max_steps=max_steps)

    samples = fitter_mcmc.samples(flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = best_fit_mcmc - np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0) - best_fit_mcmc
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless Truth:", [m_true, b_true])
    print("Initial guess:", list(models.parameters.values()))
    print('Best fit mcmc:', best_fit_mcmc)
    print('Best fit mcmc (error):', best_fit_mcmc_err)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_mcmc)))
        fit_data_mcmc = models(grid1d)
        ax.errorbar(x, y, yerr=yerr, linestyle='none', marker='o', capsize=4)
        ax.plot(x, fit_data_mcmc, label='bestfit')
        plt.show()


def test_gaussian1d_fitting(plot: bool = False) -> None:
    grid1d = Grid1D(size=100, pixelsize=0.5)
    grid_orig = Grid1D(size=50, pixelsize=1)
    amp0, mean0, sigma0 = 1., 0., 2.

    gaussian_model1 = Gaussian1D(
        amplitude=amp0, mean=mean0, sigma=sigma0, cont=0., name='gaussian1',
    )
    gaussian_model1.set_bounds(**{'amplitude': (0, 2), 'mean': (-2, 2), 'sigma': (0, 5)})
    models = join_models([gaussian_model1])
    model_data = models(grid_orig)
    np.random.seed(1994)
    y_err = 0.2 * (np.random.random(model_data.shape) - 0.5)
    y_data = model_data + y_err
    y_data_err = np.abs(y_err)
    data = {'var': grid_orig[0], 'data': y_data, 'data_err': y_data_err}

    start = time.time()
    fitter_minimize = MinimizeFitter(models, grid1d, data)
    fitter_minimize.run()
    print(f"MinimizeFitter used {time.time() - start}s")

    start = time.time()
    fitter_mcmc = EmceeFitter(models, grid1d, data)
    fitter_mcmc.auto_run(progress=True, max_steps=10000)
    print(f"EmceeFitter used {time.time() - start}s")

    initial_guess = np.array([amp0, mean0, sigma0])
    best_fit_minimize = fitter_minimize.best_fit
    best_fit_minimize_error = fitter_minimize.best_fit_error
    samples = fitter_mcmc.samples(flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = best_fit_mcmc - np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0) - best_fit_mcmc
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless Truth:", initial_guess)
    print('Best fit minimize:', best_fit_minimize)
    print('Best fit minimize (error):', best_fit_minimize_error)
    print('Best fit mcmc:', best_fit_mcmc)
    print('Best fit mcmc (error):', best_fit_mcmc_err)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_minimize)))
        fit_data_minimize = models(grid1d)
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_mcmc)))
        fit_data_mcmc = models(grid1d)
        for ax_i in ax:
            ax_i.errorbar(grid_orig[0], y_data, yerr=y_data_err, linestyle='none',
                          marker='o', markersize=8, capsize=4)
        ax[0].text(.05, .9, 'minimize fitter', transform=ax[0].transAxes, fontsize=12)
        ax[0].plot(grid1d[0], fit_data_minimize)
        ax[1].text(.05, .9, 'mcmc fitter', transform=ax[1].transAxes, fontsize=12)
        ax[1].plot(grid1d[0], fit_data_mcmc)
        plt.show()

    assert (np.abs(initial_guess - best_fit_minimize) < 0.2).all()
    assert (np.abs(initial_guess - best_fit_mcmc) < 0.2).all()


def test_gaussian1d_fitting_minimize() -> None:
    grid1d = Grid1D(size=100, pixelsize=1)
    grid_orig = Grid1D(size=40, pixelsize=1)
    amp0, mean0, sigma0 = 1., 0., 2.

    gaussian_model1 = Gaussian1D(
        amplitude=amp0, mean=mean0, sigma=sigma0, cont=0., name='gaussian1',
    )
    gaussian_model1.set_bounds(**{'amplitude': (0, 2), 'mean': (-2, 2), 'sigma': (0, 5)})
    models = join_models([gaussian_model1])
    model_data = models(grid_orig)
    np.random.seed(1994)
    y_data = model_data + 0.01 * (np.random.random(model_data.shape) - 0.5)
    y_data_err = 0.01
    data = {'var': grid_orig[0], 'data': y_data, 'data_err': y_data_err}

    fitter = MinimizeFitter(models, grid1d, data)
    fitter.run()
    initial_guess = np.array([amp0, mean0, sigma0])
    best_fit = fitter.best_fit
    best_fit_error = fitter.best_fit_error
    print("Noiseless Truth:", initial_guess)
    print('Best fit:', best_fit)
    print('Best fit error', best_fit_error)
    assert (np.abs(initial_guess - best_fit) < np.array([1, 0.1, 0.2])).all()


def test_double_gaussian_fitting() -> None:
    grid1d = Grid1D(size=100, pixelsize=1)
    grid_orig = Grid1D(size=40, pixelsize=1)
    amp1, mean1, sigma1 = 1., 0., 2.
    amp2, mean2, sigma2 = 0.5, 1., 2.

    sigma_joint = Parameter(sigma1, bounds=(-4, 4))
    gaussian_model1 = Gaussian1D(
        amplitude=amp1, mean=mean1, sigma=sigma_joint, cont=0., name='gaussian1',
    )
    gaussian_model1.set_bounds(**{'amplitude': (0, 2), 'mean': (-2, 2)})
    gaussian_model2 = Gaussian1D(
        amplitude=amp2, mean=mean2, sigma=sigma_joint, cont=0., name='gaussian2',
    )
    gaussian_model2.set_bounds(**{'amplitude': (0, 2), 'mean': (-2, 2)})

    models = join_models([gaussian_model1, gaussian_model2])
    model_data = models(grid_orig)
    np.random.seed(1994)
    y_data = model_data + 0.01 * (np.random.random(model_data.shape) - 0.5)
    y_data_err = 0.01
    data = {'var': grid_orig[0], 'data': y_data, 'data_err': y_data_err}

    fitter = MinimizeFitter(models, grid1d, data)
    fitter.run()
    initial_guess = np.array([amp1, mean1, sigma1])
    best_fit = fitter.best_fit
    best_fit_error = fitter.best_fit_error
    print("Noiseless truths:", initial_guess)
    print('Best fit:', best_fit)
    print('Best fit error', best_fit_error)
    assert (np.abs(initial_guess - best_fit) < np.array([1, 0.1, 0.2])).all()


def test_gaussian2d_curve_fit(plot: bool = False) -> None:
    from scipy.optimize import curve_fit

    def func(grid, amplitude, y_sigma, x_sigma, y0, x0, theta):
        return gaussian_2d(grid, amplitude, y_sigma, x_sigma, y0, x0, theta).ravel()

    grid2d = Grid2D(size=20, pixelsize=1)
    amp0, x0, y0, xsig0, ysig0, theta0 = 1., 0., 2., 2., 3., 0.

    gaussian_2d_model = Gaussian2D(
        amplitude=amp0, x0=x0, y0=y0,
        x_sigma=xsig0, y_sigma=ysig0, theta=theta0,
    )
    gaussian_2d_model.set_bounds(**{
        'amplitude': (0, 2), 'x0': (-5, 5), 'y0': (-5, 5),
        'x_sigma': (0, 5), 'y_sigma': (0, 5),
        'theta': (-0.5 * np.pi, 0.5 * np.pi),
    })
    models = join_models([gaussian_2d_model])
    model_data = models(grid2d)
    np.random.seed(1994)
    obs_data = model_data + 0.1 * (np.random.random(model_data.shape) - 0.5)
    obs_data_err = np.random.random(obs_data.shape)
    data = {'var': grid2d.meshgrid, 'data': obs_data, 'data_err': obs_data_err}

    truth_values = np.array([amp0, ysig0, xsig0, y0, x0, theta0])
    print(models.parameters)
    print(truth_values)

    popt, pcov = curve_fit(func, grid2d.meshgrid, obs_data.ravel())
    print("Best fit", popt)
    print("Best fit error:", np.sqrt(np.diag(pcov)))

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        data_shape = obs_data.shape
        ax[0].imshow(obs_data)
        ax[1].imshow(func(grid2d.meshgrid, *popt).reshape(data_shape))
        ax[2].imshow(obs_data - func(grid2d.meshgrid, *popt).reshape(data_shape))
        plt.show()


def test_gaussian2d_fitting(plot: bool = False, debug: bool = False) -> None:
    grid2d = Grid2D(size=20, pixelsize=1)
    amp0, x0, y0, xsig0, ysig0, theta0 = 1., 0., 2., 2., 3., 0.

    gaussian_2d_model = Gaussian2D(
        amplitude=amp0, x0=x0, y0=y0,
        x_sigma=xsig0, y_sigma=ysig0, theta=theta0,
    )
    gaussian_2d_model.set_bounds(**{
        'amplitude': (0, 2), 'x0': (-5, 5), 'y0': (-5, 5),
        'x_sigma': (0, 5), 'y_sigma': (0, 5),
        'theta': (-0.5 * np.pi, 0.5 * np.pi),
    })
    models = join_models([gaussian_2d_model])
    model_data = models(grid2d)
    np.random.seed(1994)
    obs_data = model_data + 0.1 * (np.random.random(model_data.shape) - 0.5)
    obs_data_err = np.random.random(obs_data.shape)
    data = {'var': grid2d.meshgrid, 'data': obs_data, 'data_err': obs_data_err}

    truths = list(models.parameters.values())
    initial_guess = np.array(truths) * (1 + 0.1 * np.random.randn(len(truths)))

    models.update_parameters(initial_guess)
    print("Noiseless truths:", truths)
    print("Initial guess:", initial_guess)

    fitter = EmceeFitter(models, grid2d, data)
    fitter.auto_run(progress=True, max_steps=10000)
    if debug:
        fitter.plot()
        fitter.corner_plot(truths=truths)

    best_fit = np.percentile(fitter.samples(discard=None, flat=True), 50, axis=0)
    print('Best fit:', best_fit)

    if plot:
        fig = plt.figure(figsize=(10, 3))
        ax_mcmc = [
            plt.subplot2grid((fitter.n_params, 5), (i, 1), colspan=4)
            for i in range(fitter.n_params)
        ]
        fitter.plot(ax=ax_mcmc)
        rowspan = fitter.n_params // 3
        ax1 = plt.subplot2grid((fitter.n_params, 5), (0, 0), rowspan=rowspan)
        ax2 = plt.subplot2grid((fitter.n_params, 5), (rowspan, 0), rowspan=rowspan)
        ax3 = plt.subplot2grid((fitter.n_params, 5), (2 * rowspan, 0), rowspan=rowspan)
        ax1.imshow(obs_data, origin='lower')
        best_fit_dict = dict(zip(models.parameters.keys(), best_fit))
        models.update_parameters(best_fit_dict)
        best_fit_model = models(grid2d)
        ax2.imshow(best_fit_model, origin='lower')
        ax3.imshow(obs_data - best_fit_model, origin='lower')
        plt.show()

    assert (np.abs(np.array(truths) - best_fit) < 0.5).all()


def test_sersic2d_fitting(plot: bool = False, debug: bool = False) -> None:
    grid2d = Grid2D(size=20, pixelsize=1)
    amplitude0, reff0, n0, x0, y0, ellip0, theta0 = (
        0.4, 4., 1., 0.2, -0.2, 0.4, np.pi / 3)

    sersic_2d_model = Sersic2D(
        amplitude=amplitude0, reff=reff0, n=n0,
        x0=x0, y0=y0, ellip=ellip0, theta=theta0,
    )
    sersic_2d_model.set_bounds(**{
        'amplitude': (0, 20), 'reff': (0.1, 30), 'n': (0.1, 8),
        'x0': (-10, 10), 'y0': (-10, 10),
        'ellip': (0, 1), 'theta': (-0.5 * np.pi, 0.5 * np.pi),
    })
    # sersic_2d_model.fix(**{'ellip':ellip0})

    models = join_models([sersic_2d_model])
    model_data = models(grid2d)
    np.random.seed(1994)
    obs_data = model_data + 0.01 * (np.random.random(model_data.shape) - 0.5)
    # obs_data_err = np.random.random(obs_data.shape)
    obs_data_err = np.sqrt(np.abs(obs_data) * 0.01 + 0.02**2)
    data = {'var': grid2d.meshgrid, 'data': obs_data, 'data_err': obs_data_err}

    truths = np.array(list(models.parameters.values()))
    #initial_guess = np.array(truths) * (1 + 0.1 * np.random.randn(len(truths)))
    initial_guess = np.array([1., 5., 2., 0., 0., 0.5, 0.])
    print("Noiseless truths:", truths)
    print("Initial guess:", initial_guess)
    models.update_parameters(initial_guess)

    # testing the automatic 
    fitter = EmceeFitter(models, grid2d, data, nwalkers=100)
    fitter.multi_run(progress=True, steps=1000, ncores=6)
    #fitter.auto_run(max_steps=10000, debug=debug)
    if debug:
        fitter.plot()
        fitter.corner_plot(truths=truths)

    best_fit = np.percentile(fitter.samples(discard=None, flat=True), 50, axis=0)
    print('Best fit:', best_fit)

    if plot:
        fig = plt.figure(figsize=(12, 4))
        ax_mcmc = [
            plt.subplot2grid((fitter.n_params, 5), (i, 1), colspan=4)
            for i in range(fitter.n_params)
        ]
        fitter.plot(ax=ax_mcmc)
        rowspan = fitter.n_params // 3
        ax1 = plt.subplot2grid((fitter.n_params, 5), (0, 0), rowspan=rowspan)
        ax2 = plt.subplot2grid((fitter.n_params, 5), (rowspan, 0), rowspan=rowspan)
        ax3 = plt.subplot2grid((fitter.n_params, 5), (2 * rowspan, 0), rowspan=rowspan)
        ax1.imshow(obs_data, origin='lower')
        best_fit_dict = dict(zip(models.parameters.keys(), best_fit))
        models.update_parameters(best_fit_dict)
        best_fit_model = models(grid2d)
        ax2.imshow(best_fit_model, origin='lower')
        ax3.imshow(obs_data - best_fit_model, origin='lower')
        plt.show()
    print(np.abs(truths - best_fit))
    assert (np.abs(truths - best_fit) < 0.2).all()


def test_blackbody(plot: bool = True, debug: bool = False) -> None:
    freq = np.array([100, 200, 400, 600, 800, 1000, 2000, 4000, 5000])
    grid1d = Grid1D(freq)
    temperature = 40.
    dustmass = 1e10

    mbb_model = MBlackbody(
        temperature=temperature, z=2., beta=2.0,
        dustmass=dustmass, area=1., name='test_mbb',
    )
    mbb_model.set_bounds(**{'temperature': (10, 100), 'dustmass': (1e9, 1e11)})
    models = join_models([mbb_model])
    model_data = models(grid1d)
    np.random.seed(1994)
    y_err = np.array([0.01, -0.03, -0.2, 0.5, 0.8, -1., 3, -4.5, 4.5])
    y_data = model_data + y_err
    y_data_err = np.abs(y_err)

    data = {'var': grid1d[0], 'data': y_data, 'data_err': y_data_err}

    fitter_minimize = MinimizeFitter(models, grid1d, data)
    fitter_minimize.run()
    #fitter_minimize.auto_run()
    best_fit_minimize = fitter_minimize.best_fit
    best_fit_minimize_error = fitter_minimize.best_fit_error

    params_keyorder = list(models.parameters.keys())
    models.update_parameters(dict(zip(params_keyorder, [50., 9.2])))
    fitter_mcmc = EmceeFitter(models, grid1d, data)
    fitter_mcmc.auto_run(progress=True, max_steps=5000)
    if debug:
        fitter_mcmc.plot()
        fitter_mcmc.corner_plot()

    samples = fitter_mcmc.samples(discard=1000, thin=1, flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0)
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless truths:", [temperature, dustmass])
    print('Best fit minimize:', best_fit_minimize)
    print('Best fit mcmc:', best_fit_mcmc)

    if plot:
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_minimize)))
        fit_data_minimize = models(grid1d)
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_mcmc)))
        fit_data_mcmc = models(grid1d)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(freq, y_data, yerr=y_data_err, linestyle='none', marker='o', label='data')
        ax.plot(freq, model_data, label='model')
        ax.plot(freq, fit_data_minimize, label='minimize fit')
        ax.plot(freq, fit_data_mcmc, label='mcmc fit')
        ax.invert_xaxis()
        ax.legend()
        plt.show()


def test_blackbody_log(plot: bool = True, debug: bool = False) -> None:
    grid1d = Grid1D()
    freq = np.array([100, 200, 400, 600, 800, 1000, 2000, 4000, 5000])
    grid1d[0] = freq
    temperature = 40.
    dustmass = 10.

    mbb_model = MBlackbody_log(
        temperature=temperature, z=2., beta=2.0,
        dustmass=dustmass, area=1., name='test_mbb',
    )
    mbb_model.set_bounds(**{'temperature': (10, 100), 'dustmass': (9, 11)})
    models = join_models([mbb_model])
    model_data = models(grid1d)
    np.random.seed(1994)
    y_err = np.array([0.01, -0.03, -0.2, 0.5, 0.8, -1., 3, -4.5, 4.5])
    y_data = model_data + y_err
    y_data_err = np.abs(y_err)

    data = {'var': grid1d[0], 'data': y_data, 'data_err': y_data_err}

    fitter_minimize = MinimizeFitter(models, grid1d, data)
    fitter_minimize.run()
    best_fit_minimize = fitter_minimize.best_fit
    best_fit_minimize_error = fitter_minimize.best_fit_error

    params_keyorder = list(models.parameters.keys())
    models.update_parameters(dict(zip(params_keyorder, [50., 9.2])))
    fitter_mcmc = EmceeFitter(models, grid1d, data)
    fitter_mcmc.auto_run(progress=True, max_steps=5000)
    if debug:
        fitter_mcmc.plot()
        fitter_mcmc.corner_plot()

    samples = fitter_mcmc.samples(discard=1000, thin=1, flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0)
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless truths:", [temperature, dustmass])
    print('Best fit minimize:', best_fit_minimize)
    print('Best fit mcmc:', best_fit_mcmc)

    if plot:
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_minimize)))
        fit_data_minimize = models(grid1d)
        models.update_parameters(dict(zip(models.parameters.keys(), best_fit_mcmc)))
        fit_data_mcmc = models(grid1d)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(freq, y_data, yerr=y_data_err, linestyle='none', marker='o', label='data')
        ax.plot(freq, model_data, label='model')
        ax.plot(freq, fit_data_minimize, label='minimize fit')
        ax.plot(freq, fit_data_mcmc, label='mcmc fit')
        ax.invert_xaxis()
        ax.legend()
        plt.show()
