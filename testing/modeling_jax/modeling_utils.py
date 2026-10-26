#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2024-11-08: modeling become a separate module, v0.1
    2026-07-10: redesigned with CompoundModel, Prior hierarchy, cleaner Parameter, v0.2
    2026-10-22: reconstructed with jax + numpyro acceleration, following the design of
                pysersic (https://github.com/pysersic/pysersic), v0.3

Design notes
------------
Following pysersic, the whole fitting pipeline is expressed as *pure jax
functions* of a flat parameter vector:

    theta (flat vector, `self.keys` order)
      -> models.update_parameters(theta)     # parameter injection (trace-time)
      -> render model image(s) on the grid   # jnp, differentiable
      -> (optional) resample to data coords  # jax map_coordinates
      -> (optional) PSF convolution          # pre-FFTed kernel, jnp.fft
      -> loss (Gaussian / Student-t) + priors

The rendered log-posterior, its gradient and its Hessian are `jax.jit`-compiled
once per fitter; inference then uses either

    * `find_map`          -- gradient-based MAP optimisation (scipy L-BFGS-B on
                             the compiled value_and_grad) + Laplace covariance
                             from the compiled Hessian,
    * `find_map_svi`      -- MAP via numpyro AutoDelta SVI (pysersic-style),
    * `estimate_posterior`-- 'laplace' / 'svi-mvn' / 'svi-flow' (numpyro SVI),
    * `sample`            -- NUTS (numpyro MCMC).

Because jax traces (rather than executes) the log-posterior, avoid Python-level
branching on parameter values inside `Model.evaluate`; all concrete models in
this module are written with jnp and `jnp.where` only.  Note that jax compiles
the fitter against the *current* fixed-parameter values, so construct the fitter
after fixing/tying parameters.
"""

from __future__ import annotations
__version__ = '0.3.0'

import operator
import time
import warnings
from functools import cached_property
from functools import reduce
from typing import Callable, Optional

import numpy as np

import jax
# 64-bit precision is essential for astronomical quantities (fluxes, positions)
# and improves the stability of gradients through the profile functions.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro import deterministic, infer, optim

from scipy import optimize, stats
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
    def __new__(
        cls,
        value: float | int | Parameter = None,
        *args, **kwargs
    ) -> Parameter:
        """
        If a Parameter instance is passed, it is reused as-is (pass-through),
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
        """Log-prior at *v*.  Written with jnp primitives so it is safe to
        call with jax tracers inside a compiled log-posterior.

        Outside the hard bounds a steep, finite linear barrier is applied
        instead of -inf: scipy's L-BFGS-B line search may probe points
        slightly beyond the declared bounds, where an -inf log-posterior
        would abort the optimisation (ABNORMAL_TERMINATION_IN_LNSRCH).
        """
        lo, hi = self.bounds
        lp = 0.0 if self.prior is None else self.prior.logpdf(v)
        if lo is not None and hi is not None:
            violation = jnp.clip(lo - v, 0.0, None) + jnp.clip(v - hi, 0.0, None)
            lp = lp - 1e10 * violation
        return lp

    def numpyro_prior(self) -> dist.Distribution:
        """Numpyro distribution for this parameter, used by the SVI/NUTS
        inference paths.  Falls back to a wide Uniform when no prior is
        attached and the bounds are unbounded."""
        if self.prior is not None:
            return self.prior.to_numpyro()
        lo, hi = self.bounds
        if not (np.isfinite(lo) and np.isfinite(hi)):
            warnings.warn(
                f"Parameter with bounds {self.bounds} has no prior; using a "
                f"wide Uniform({max(lo, -1e6)}, {min(hi, 1e6)}) for numpyro. "
                f"Attach a Prior for meaningful inference."
            )
            lo, hi = max(lo, -1e6), min(hi, 1e6)
        return dist.Uniform(lo, hi)

    def __repr__(self):
        return f"Parameter(value={self.value}, bounds={self.bounds}, fixed={self.fixed})"

class DerivedParameter(Parameter):
    """
    A read-only Parameter computed from one or more source Parameters
    via a user-supplied function, instead of being independently fit.

    Always fixed=True -- automatically excluded from the free-parameter
    list, since its value is fully determined by its source(s).

    Note: the linking function must be jnp-compatible (plain arithmetic,
    jnp functions) so that derived values remain differentiable under jit.
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

    Subclasses must implement ``logpdf`` (jnp-compatible), ``ppf`` and
    ``to_numpyro``.
    """
    def logpdf(self, v: float) -> float:
        """Log of the probability density function."""
        raise NotImplementedError

    def ppf(self, u: float) -> float:
        """Percent-point function (inverse CDF)."""
        raise NotImplementedError

    def to_numpyro(self) -> dist.Distribution:
        """Equivalent numpyro distribution for the SVI/NUTS paths."""
        raise NotImplementedError

class UniformPrior(Prior):
    """Uniform prior on [low, high]."""
    def __init__(self, low: float, high: float):
        self.low, self.high = low, high

    def logpdf(self, v: float) -> float:
        inside = (v >= self.low) & (v <= self.high)
        return jnp.where(inside, -jnp.log(self.high - self.low), -jnp.inf)

    def ppf(self, u: float) -> float:
        return self.low + u * (self.high - self.low)

    def to_numpyro(self) -> dist.Distribution:
        return dist.Uniform(self.low, self.high)

class GaussianPrior(Prior):
    """Gaussian (normal) prior with mean *mu* and standard deviation *sigma*."""
    def __init__(self, mu: float, sigma: float):
        self.mu, self.sigma = mu, sigma

    def logpdf(self, v: float) -> float:
        z = (v - self.mu) / self.sigma
        return -0.5 * z**2 - jnp.log(self.sigma * jnp.sqrt(2.0 * jnp.pi))

    def ppf(self, u: float) -> float:
        return self.mu + self.sigma * stats.norm.ppf(u)

    def to_numpyro(self) -> dist.Distribution:
        return dist.Normal(self.mu, self.sigma)


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

class Model:
    """Base class for all models.

    Subclasses assign ``Parameter`` instances as attributes in ``__init__``
    and implement ``evaluate(*coords)``.

    `evaluate` must be jnp-compatible (no data-dependent Python control
    flow, no in-place boolean-mask assignment) so the whole model tree can
    be traced by jax inside the compiled log-posterior.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    def __add__(self, other: Model):
        return CompoundModel(self, other if isinstance(other, Model) else _Constant(other), '+')
    def __sub__(self, other: Model):
        return CompoundModel(self, other if isinstance(other, Model) else _Constant(other), '-')
    def __mul__(self, other: Model):
        return CompoundModel(self, other if isinstance(other, Model) else _Constant(other), '*')
    def __truediv__(self, other: Model):
        return CompoundModel(self, other if isinstance(other, Model) else _Constant(other), '/')

    def _iter_params(self):
        """Yield ``(owner_model, attr_name, Parameter)`` for each free parameter."""
        for attr, val in vars(self).items():
            if isinstance(val, Parameter) and not val.fixed:
                yield self, attr, val

    @property
    def submodels(self) -> list:
        return [self]

    def _params_map(self) -> dict:
        """Return ``id(Parameter) -> (display_key, Parameter)``.

        Parameters that share the same Python object (via :meth:`tie`) are
        collapsed into a single entry.
        """
        out: dict = {}
        seen_keys: dict = {}
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
    def parameters(self) -> dict:
        """Return ``{name: value}`` for every free parameter."""
        return {k: p.value for k, p in self._params_map().values()}

    @property
    def bounds(self) -> dict:
        """Return ``{name: (low, high)}`` for every free parameter."""
        return {k: p.bounds for k, p in self._params_map().values()}

    def update_parameters(self, params) -> None:
        """Update parameter values from a dict or an array (in ``.parameters`` key order)."""
        pm = {k: p for k, p in self._params_map().values()}
        if not isinstance(params, dict):
            params = dict(zip(pm.keys(), params))
        for k, v in params.items():
            if k in pm:
                pm[k].value = v

    def get_priors(self, params: dict = None) -> float:
        """Return the sum of log-priors for the given (or current) parameter values."""
        pm = {k: p for k, p in self._params_map().values()}
        params = params if params is not None else {k: p.value for k, p in pm.items()}
        total = 0.0
        for k, p in pm.items():
            if k in params:
                total = total + p.log_prior(params[k])
        return total

    # ---------- evaluation ----------
    def __call__(self, *args):
        if len(args) == 1 and hasattr(args[0], "ndim") and hasattr(args[0], "meshgrid"):
            grid = args[0]
            coords = (grid[0],) if grid.ndim == 1 else grid.meshgrid
            return self.evaluate(*coords)
        return self.evaluate(*args)

    def evaluate(self, *args) -> np.ndarray:
        """Evaluate the model on coordinate arrays.  Override in subclasses."""
        raise NotImplementedError

    def __getitem__(self, i: int) -> Model:
        return self.submodels[i]

    # ---------- name resolution ----------
    def _resolve(self, path: str) -> tuple:
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
        The linking function must be jnp-compatible so the derived value
        stays differentiable inside the compiled log-posterior.
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
        model.link_func('bar_r.amplitude', 'bar_g.amplitude', 'color.ratio',
                        func=lambda a, r: a * r)
        """
        target_owner, target_attr = self._resolve(target)
        source_params = [getattr(*self._resolve(s)) for s in sources]
        setattr(target_owner, target_attr, DerivedParameter(func, *source_params))
        return self

    def set_bounds(self, name_bounds_pairs: dict) -> Model:
        """Set hard bounds for one or more parameters.

        Example
        -------
        model.set_bounds({'bar.a_bar': (2, 50), 'bar.ellip': (0.0, 0.9)})
        """
        for name, bounds in name_bounds_pairs.items():
            owner, attr = self._resolve(name)
            getattr(owner, attr).bounds = tuple(bounds)
        return self

    def set_prior(self, name_prior_pairs: dict) -> Model:
        """Attach Prior distributions to parameters.

        Example
        -------
        model.set_prior({'bulge.n': GaussianPrior(mu=4.0, sigma=0.5)})
        """
        for name, prior in name_prior_pairs.items():
            owner, attr = self._resolve(name)
            getattr(owner, attr).prior = prior
        return self

    def get_bounds(self) -> list:
        """Return an ordered list of bounds matching ``.parameters`` key order."""
        return list(self.bounds.values())

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Unit-cube -> physical-value transform."""
        result = []
        for (_key, p), ui in zip(self._params_map().values(), u):
            if p.prior is not None:
                result.append(p.prior.ppf(ui))
            else:
                lo, hi = p.bounds
                result.append(lo + ui * (hi - lo))
        return np.array(result)

    def summary(self) -> None:
        """Print a table of all parameters with values, fixed status, and ties."""
        seen: dict = {}
        lines: list = []
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
                if isinstance(p, DerivedParameter):
                    status = 'derived'
                else:
                    status = "FIXED" if p.fixed else "free "
                lines.append(
                    f"{m.name}.{attr:<12} = {p.value!r:>10}  [{status}]{tied_note}"
                )
        print("\n".join(lines))


class _Constant(Model):
    """Internal: wraps a scalar constant for ``model + 5.0`` style expressions."""
    def __init__(self, value: float) -> None:
        super().__init__(name="const")
        self.value = value

    def evaluate(self, *args) -> np.ndarray:
        return jnp.full_like(args[0], self.value) if args else jnp.array(self.value)

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
        seen: dict = {}
        for m in self.submodels:
            seen[m.name] = seen.get(m.name, 0) + 1
            if seen[m.name] > 1:
                m.name = f"{m.name}_{seen[m.name] - 1}"

    @property
    def submodels(self) -> list:
        return self.left.submodels + self.right.submodels

    def _iter_params(self):
        for m in self.submodels:
            yield from m._iter_params()

    def evaluate(self, *args) -> np.ndarray:
        return self._ops[self.op](self.left(*args), self.right(*args))

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"

    def summary(self) -> None:
        """Print a table of all parameters with values, fixed status, and ties."""
        seen: dict = {}
        lines: list = []
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




class ModelList(Model):
    """
    Wraps one Model per key for joint multi-model fitting.
    Unlike CompoundModel (+/-/*), models here stay logically separate
    -- each is evaluated against its own data -- but their free
    parameters are exposed as one union (with tying support), so a
    Fitter can treat them as a single parameter vector.

    link_models: holding parameters for model fitting, but not directly
                 contributing to pixel values.
    """

    def __init__(
            self, models: list,
            link_models: Optional[list] = None,
            name=None
        ):
        super().__init__(name=name)
        self.models = models
        self.link_models = link_models or []
        self._disambiguate_names()

    def _disambiguate_names(self):
        """Ensure every leaf submodel has a name unique."""
        seen = {}
        for m in self.submodels:
            seen[m.name] = seen.get(m.name, 0) + 1
            if seen[m.name] > 1:
                m.name = f"{m.name}_{seen[m.name] - 1}"

    def __getitem__(self, i):
        return self.models[i]

    def __len__(self):
        return len(self.models)

    def _iter_params(self):
        for model in self.models:
            for m in model.submodels:
                yield from m._iter_params()
        for lmodel in self.link_models:
            for lm in lmodel.submodels:
                yield from lm._iter_params()

    @property
    def submodels(self) -> list:
        subs = []
        for model in self.models:
            subs.extend(model.submodels)
        for lmodel in self.link_models:
            subs.extend(lmodel)
        return subs

    @property
    def link_paramters(self) -> dict:
        # (kept for backward compatibility; note the historical typo)
        subs = {}
        for link_model in self.link_models:
            subs.update(link_model.parameters)
        return subs

    def evaluate(self, *args) -> list:
        return [m.evaluate(*args) for m in self.models]


###############################################
# Concrete model classes
###############################################

def sersic_bn(n):
    """Differentiable approximation of the Sersic b_n factor:

        bn = gammaincinv(2n, 0.5)

    Uses the high-order analytic expansion adopted by pysersic, accurate to
    ~1e-4 for 0.5 < n < 10 and fully differentiable with jax (unlike
    scipy.special.gammaincinv).

    Parameters
    ----------
    n : Sersic index (scalar or jnp array).
    """
    # numerical guard: keep n away from zero so the expansion (and its
    # gradient) stays finite when an optimiser's line search probes n -> 0
    n = jnp.maximum(jnp.asarray(n), 1e-10)
    bn = (2 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)) + (
        ((0.00018073182821237496 / (n + jnp.exp(n))) + 3.7026973571904255e-5)
        / (-0.09149183119702775 - n) ** 2
    ) * (
        (
            jnp.log(n)
            + (
                ((n * 2.6248718397705195) + -0.9727511612512357)
                / (n ** (-4) + 94.78011643586419)
            )
        )
        / (n + -0.006044236674273689)
    )
    return bn

def gaussian_2d(
    grid: tuple,
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
    """
    y, x = grid
    a = jnp.cos(theta)**2 / (2 * x_sigma**2) + jnp.sin(theta)**2 / (2 * y_sigma**2)
    b = jnp.sin(2 * theta) / (2 * x_sigma**2) - jnp.sin(2 * theta) / (2 * y_sigma**2)
    c = jnp.sin(theta)**2 / (2 * x_sigma**2) + jnp.cos(theta)**2 / (2 * y_sigma**2)
    return amplitude * jnp.exp(-a * (x - x0)**2 - b * (x - x0) * (y - y0) - c * (y - y0)**2)

def blackbody_nu(nu, temperature):
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
        * (jnp.exp(const_h * nu / (k_B * temperature)) - 1)**-1
    )

def blackbody_lambda(lam, temperature):
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
        / (jnp.exp(const_h * const_c / (lam * k_B * temperature)) - 1)
    )

def modified_blackbody_nu(
    nu,
    temperature: float = None,
    z: float = 0,
    beta: float = 2.0,
    dustmass: float = None,
    area: float = None,
):
    """Flux density of a modified blackbody.

    Parameters
    ----------
    nu : rest-frame frequency in Hz.
    temperature : dust temperature in K.
    z : redshift (must be a fixed/known value -- astropy cosmology is not
        differentiable/jittable).
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
    return (1 + z) * area / D_L**2 * (1 - jnp.exp(-tau_nu)) * (B_Td - B_Tcmb)

def check_regular_grid(arr: np.ndarray) -> bool:
    """Return True if *arr* is regularly spaced."""
    arr_diff = np.diff(arr)
    return bool(np.all(np.abs(arr_diff - arr_diff[0]) <= np.finfo(arr_diff.dtype).eps))

class PSFConvolver:
    """FFT-based 'same'-mode convolution with a fixed PSF kernel (jax).

    Produces the same result as ``scipy.signal.fftconvolve(img, psf,
    mode='same')`` but is a pure jnp function, so it composes with
    ``jax.jit``/``jax.grad`` inside the compiled log-posterior.  The kernel's
    FFT is computed once in the constructor and reused as a constant.

    The caller is responsible for any kernel normalisation -- pass
    ``normalize=True`` to divide the PSF by its sum (recommended).

    Parameters
    ----------
    psf : ndarray
        The (2D) convolution kernel.
    shape : tuple[int, int]
        The ``(ny, nx)`` shape of the images that will be convolved.
    normalize : bool
        If True, normalise the kernel to unit sum before FFT-ing.
    """

    def __init__(self, psf, shape: tuple, normalize: bool = False):
        psf = jnp.asarray(psf, dtype=jnp.float64)
        if psf.ndim != 2:
            raise ValueError(f"PSFConvolver expects a 2D kernel, got ndim={psf.ndim}")
        if normalize:
            psf = psf / psf.sum()
        self._psf = psf
        self._shape = tuple(int(s) for s in shape)
        self._fsize = (self._shape[0] + psf.shape[0] - 1,
                       self._shape[1] + psf.shape[1] - 1)
        self._psf_fft = jnp.fft.rfft2(psf, s=self._fsize)

    def __call__(self, image) -> np.ndarray:
        """Convolve *image* (shape ``self._shape``) with the fixed PSF."""
        img = jnp.asarray(image)
        f = jnp.fft.rfft2(img, s=self._fsize) * self._psf_fft
        full = jnp.fft.irfft2(f, s=self._fsize)
        # central crop matching scipy/astropy 'same' mode
        oy = (self._fsize[0] - self._shape[0]) // 2
        ox = (self._fsize[1] - self._shape[1]) // 2
        return full[oy:oy + self._shape[0], ox:ox + self._shape[1]]

    @property
    def psf(self) -> np.ndarray:
        return self._psf

class Line1D(Model):
    """Straight line: ``y = m * x + b``."""

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
    """

    def __init__(self, factor=0.0, name=None):
        super().__init__(name=name)
        self.factor = Parameter(factor, bounds=(-np.inf, np.inf))

    def evaluate(self, *args):
        return jnp.zeros_like(args[0])

class Gaussian1D(Model):
    """One-dimensional Gaussian plus an optional continuum.

        f(x) = amplitude * exp(-0.5 * ((x - mean) / sigma)^2) + cont
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
        amp = self.amplitude.value
        mu = self.mean.value
        sig = self.sigma.value
        return amp * jnp.exp(-0.5 * ((x - mu) / sig) ** 2) + self.cont.value

class Sersic1D(Model):
    """One-dimensional Sérsic profile.

        I(x) = amplitude * exp{ -b_n [(|x - x0| / reff)^(1/n) - 1] }

    Uses the differentiable :func:`sersic_bn` approximation instead of
    ``scipy.special.gammaincinv``.
    """

    def __init__(self, amplitude=1.0, reff=1.0, n=2.0, x0=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0.0, np.inf))
        self.reff      = Parameter(reff,      bounds=(1e-3, np.inf))
        self.n         = Parameter(n,         bounds=(0.5, 10.0))
        self.x0        = Parameter(x0,        bounds=(-np.inf, np.inf))

    def evaluate(self, x):
        bn = sersic_bn(self.n.value)
        # numerical guard as in Sersic2D.evaluate
        r = jnp.abs(x - self.x0.value) / jnp.maximum(self.reff.value, 1e-10)
        return self.amplitude.value * jnp.exp(-bn * (r ** (1.0 / self.n.value) - 1.0))

class Gaussian2D(Model):
    """Two-dimensional Gaussian.

        f(x, y) = amplitude * exp(-a*(x-x0)^2 - b*(x-x0)*(y-y0) - c*(y-y0)^2)

    where *a*, *b*, *c* are computed from *x_sigma*, *y_sigma*, and *theta*.
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

        # numerical guards: keep the sigmas away from zero so the profile
        # (and its gradient) stays finite when a bounded optimiser's line
        # search probes the sigma -> 0 boundary
        x_sig = jnp.maximum(x_sig, 1e-10)
        y_sig = jnp.maximum(y_sig, 1e-10)

        cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
        a = cos_t**2 / (2 * x_sig**2) + sin_t**2 / (2 * y_sig**2)
        b = sin_t * cos_t / x_sig**2 - sin_t * cos_t / y_sig**2
        c = sin_t**2 / (2 * x_sig**2) + cos_t**2 / (2 * y_sig**2)

        dx, dy = x - x0, y - y0
        return amp * jnp.exp(-a * dx**2 - b * dx * dy - c * dy**2)

class Sersic2D(Model):
    """Two-dimensional Sérsic profile:

        I(r) = amplitude * exp{ -b_n [ (r / reff)^(1/n) - 1 ] }

    where *r* is the elliptical radius computed from the rotated,
    ellipticity-scaled coordinate system.  Uses the differentiable
    :func:`sersic_bn` approximation (accurate for 0.5 < n < 10).
    """

    def __init__(self, amplitude=1.0, reff=10.0, n=2.0, x0=0.0, y0=0.0,
                 ellip=0.3, theta=0.0, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude, bounds=(0.0, np.inf))
        self.reff      = Parameter(reff,      bounds=(0.0, np.inf))
        self.n         = Parameter(n,         bounds=(0.5, 10.0))
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

        bn = sersic_bn(n)
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        dx, dy = x - x0, y - y0
        x_maj =  dx * cos_theta + dy * sin_theta
        x_min = -dx * sin_theta + dy * cos_theta

        # numerical guards: keep the scale lengths away from zero so the
        # profile (and its gradient) stays finite when a bounded optimiser's
        # line search probes the ellip -> 1 or reff -> 0 boundary
        reff = jnp.maximum(reff, 1e-10)
        b = jnp.maximum((1.0 - ellip) * reff, 1e-10)
        r = jnp.sqrt((x_maj / reff) ** 2 + (x_min / b) ** 2)

        return amplitude * jnp.exp(-bn * (r ** (1.0 / n) - 1.0))

class Blackbody(Model):
    """Blackbody spectrum as a function of frequency.

        B_nu = 2 h nu^3 / c^2  *  (exp(h nu / k_B T) - 1)^-1
    """

    def __init__(self, temperature=10.0, name=None):
        super().__init__(name=name)
        self.temperature = Parameter(temperature, bounds=(1e-3, np.inf))

    def evaluate(self, *args):
        """*args* is nu in GHz (or a Grid dispatched by base ``__call__``)."""
        nu = args[0] if len(args) == 1 else args[1]
        return blackbody_nu(nu * 1e9, self.temperature.value)

class MBlackbody(Model):
    """Modified blackbody (grey-body) model.

    Note: `z` must remain fixed -- the astropy cosmology luminosity distance
    is not jax-differentiable.
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
        """*args* is nu in GHz (or a Grid dispatched by base ``__call__``)."""
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
    """Modified blackbody with *log10(dustmass)* as the free parameter."""

    def __init__(self, temperature=40.0, z=0.0, beta=2.0,
                 dustmass=10.0, area=1.0, name=None):
        super().__init__(name=name)
        self.temperature = Parameter(temperature, bounds=(1e-3, np.inf))
        self.z           = Parameter(z,           fixed=True)
        self.beta        = Parameter(beta,        fixed=True)
        self.dustmass    = Parameter(dustmass,    bounds=(0.0, np.inf))
        self.area        = Parameter(area,        fixed=True)

    def evaluate(self, *args):
        """*args* is nu in GHz (or a Grid dispatched by base ``__call__``)."""
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
# Loss functions
###############################################
# Each loss returns the *chi2-like* quantity -2*log(likelihood) up to an
# additive constant, summed over unmasked pixels.  They are pure jnp
# functions so they can be used inside the compiled log-posterior.

def gaussian_loss(mod, obs, sigma2, keep=None):
    """Gaussian chi2: sum((obs - mod)^2 / sigma^2) over unmasked pixels.

    Parameters
    ----------
    mod : model image.
    obs : observed image.
    sigma2 : per-pixel variance.
    keep : boolean array, True for pixels included in the loss.
    """
    chi2 = (obs - mod) ** 2 / sigma2
    if keep is None:
        return jnp.sum(chi2)
    return jnp.sum(jnp.where(keep, chi2, 0.0))

def student_t_loss(mod, obs, sigma2, keep=None, nu: float = 5.0):
    """Student-t loss (fat tails, robust to outliers).

    Follows pysersic: scale = sqrt((nu - 2) / 2) * rms.  Returns the
    chi2-like -2*log(L) up to an additive constant.
    """
    scale2 = (nu - 2.0) / 2.0 * sigma2
    z2 = (obs - mod) ** 2 / scale2
    chi2 = (nu + 1.0) * jnp.log(1.0 + z2 / nu)
    if keep is None:
        return jnp.sum(chi2)
    return jnp.sum(jnp.where(keep, chi2, 0.0))

LOSS_FUNCS = {'gaussian': gaussian_loss, 'student-t': student_t_loss}


def join_models(models: list) -> CompoundModel:
    """Combine a list of Models into a single additive CompoundModel."""
    return reduce(operator.add, models)


###############################################
# numpyro SVI training helper
###############################################

def train_numpyro_svi_early_stop(
    model: Callable,
    guide: Callable,
    num_round: int = 3,
    max_train: int = 5000,
    lr_init: float = 0.01,
    frac_lr_decrease: float = 0.1,
    patience: int = 100,
    ELBO_loss: Callable = None,
    rkey=None,
    verbose: bool = False,
):
    """Train an SVI model over multiple rounds with a decreasing learning
    rate and early stopping (adapted from pysersic).

    Returns
    -------
    (params, best_loss)
        Converged guide parameters and the best ELBO loss.
    """
    if rkey is None:
        rkey = jax.random.PRNGKey(0)
    if ELBO_loss is None:
        ELBO_loss = infer.Trace_ELBO()

    best_params, best_loss = None, np.inf
    keys = jax.random.split(rkey, num_round + 1)

    for r in range(num_round):
        lr = lr_init * frac_lr_decrease**r
        # fresh SVI per round so the new optimizer is picked up by the jit
        svi = infer.SVI(model, guide, optim.Adam(lr), loss=ELBO_loss)
        state = svi.init(keys[r])
        update = jax.jit(svi.update)

        best_state_r, best_loss_r = state, np.inf
        wait = 0
        for _ in range(max_train):
            state, loss = update(state)
            if float(loss) < best_loss_r:
                best_loss_r, best_state_r, wait = float(loss), state, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if verbose:
            print(f"  SVI round {r}: lr={lr:.1e}, loss={best_loss_r:.4e}")
        if best_loss_r < best_loss:
            best_loss = best_loss_r
            best_params = svi.get_params(best_state_r)

    return best_params, best_loss


###############################################
# Jax-accelerated Fitters
###############################################

class JaxFitter:
    """Base jax/numpyro fitter.

    The model tree, grids, data, uncertainties, masks and PSF kernels are
    frozen at construction; the log-likelihood, log-posterior, gradient,
    Hessian and model rendering are compiled once as pure functions of the
    flat parameter vector ``theta`` (ordered as ``self.keys``).

    Inference methods
    -----------------
    find_map()           -- gradient-based MAP (scipy L-BFGS-B on the
                            compiled value_and_grad) + Laplace covariance.
    find_map_svi()       -- MAP via numpyro AutoDelta SVI (pysersic-style).
    estimate_posterior() -- 'laplace' / 'svi-mvn' / 'svi-flow' SVI.
    sample()             -- NUTS via numpyro MCMC.

    Parameters
    ----------
    models : Model | CompoundModel | ModelList
        The model (tree) to fit.  Free parameters of all submodels (and any
        ``link_models`` of a ModelList) form the flat parameter vector.
    grid : Grid
        The evaluation grid (shared by all bands).
    bands : list of dict
        One dict per band with keys: ``model`` (Model to render), ``obs``,
        ``err`` (1-sigma uncertainties or None), ``mask`` (True = masked, or
        None), ``var`` (data coordinates; None = on-grid), ``psf`` (2D kernel
        or None).
    loss : str
        'gaussian' or 'student-t'.
    name : str, optional
    """

    def __init__(self, models, grid, bands: list, loss: str = 'gaussian',
                 name: str = None):
        if isinstance(models, (list, tuple)):
            models = ModelList(list(models))
        self.models = models
        self.grid = grid
        self.name = name or type(self).__name__
        if loss not in LOSS_FUNCS:
            raise ValueError(f"loss must be one of {list(LOSS_FUNCS)}")
        self.loss = loss

        # ---- free parameters (deterministic key order) ----
        self._free = list(models._params_map().values())  # [(key, Parameter)]
        self.keys = [k for k, _ in self._free]

        # ---- coordinate arrays (shared by all bands) ----
        coords = (grid[0],) if grid.ndim == 1 else grid.meshgrid
        self._coords = tuple(jnp.asarray(c, dtype=jnp.float64) for c in coords)

        # ---- build the static per-band data structures ----
        self._bands = [self._build_band(**b) for b in bands]
        self._n_data = sum(int(b['obs'].size) for b in self._bands)

        # ---- ScaleUncertainty pseudo-models per band (extra variance) ----
        self._scale_uncertainty = []
        for b in self._bands:
            sus = [m for m in b['model'].submodels if isinstance(m, ScaleUncertainty)]
            self._scale_uncertainty.append(sus[0] if sus else None)

        # ---- compile the core jax functions once ----
        self._render_jit = jax.jit(self._model_images)
        self._loglike_jit = jax.jit(self._log_likelihood)
        self._logpost_jit = jax.jit(self._log_posterior)
        self._logpost_valgrad_jit = jax.jit(jax.value_and_grad(self._log_posterior))
        self._logpost_hess_jit = jax.jit(jax.hessian(self._log_posterior))

        # results placeholders
        self.best_fit = None
        self.best_fit_error = None
        self.best_fit_error_lower = None
        self.best_fit_error_upper = None
        self.chi2 = None
        self.reduced_chi2 = None
        self.dof = None
        self.mcmc = None
        self.svi_result = None
        self.converged = None

    # ------------------------------------------------------------ builders
    def _build_band(self, model, obs, err=None, mask=None, var=None, psf=None):
        """Freeze one band's static data (obs, sigma2, keep-mask, PSF FFT,
        interpolation coordinates) as jnp arrays."""
        model = self._unwrap_model(model)
        obs = jnp.asarray(obs, dtype=jnp.float64)
        if err is None:
            sigma2 = jnp.ones_like(obs)
        else:
            sigma2 = jnp.asarray(err, dtype=jnp.float64) ** 2
            # guard against 0 / negative errors
            sigma2 = jnp.where(sigma2 > 0, sigma2, 1.0)
        if mask is None:
            keep = jnp.ones_like(obs, dtype=bool)
        else:
            # convention: mask True = bad pixel (excluded), as in the legacy code
            keep = ~jnp.asarray(mask, dtype=bool)

        conv = None
        if psf is not None:
            conv = PSFConvolver(psf, obs.shape, normalize=True)

        interp_coords = None
        if var is not None and not self._var_on_grid(var):
            interp_coords = self._make_interp_coords(var, obs.shape)

        return dict(model=model, obs=obs, sigma2=sigma2, keep=keep,
                    conv=conv, interp_coords=interp_coords)

    def _unwrap_model(self, model):
        """Accept a Model, CompoundModel, ModelList or an integer index into a
        ModelList; return the renderable Model."""
        if isinstance(model, ModelList):
            if len(model.models) != 1:
                raise ValueError(
                    "A ModelList band must contain exactly one image model; "
                    "use the multiband fitters for several images."
                )
            model = model.models[0]
        return model

    def _var_on_grid(self, var) -> bool:
        """Return True if *var* gives data coordinates matching the fitter's
        grid: a 1D coordinate array for 1D grids, or a stack of per-axis
        meshgrids of shape (ndim, *grid.shape) otherwise."""
        try:
            var_arr = np.asarray(var, dtype=float)
            ndim = self.grid.ndim
            if ndim == 1:
                if var_arr.ndim != 1:
                    return False
                return np.allclose(var_arr, self.grid.grid[0])
            if var_arr.ndim != ndim + 1:
                return False
            return all(np.allclose(v, g) for v, g in zip(var_arr, self.grid.grid))
        except Exception:
            return False

    def _make_interp_coords(self, var, obs_shape):
        """Pixel-space coordinates for ``jax.scipy.ndimage.map_coordinates``
        (linear interpolation of the model grid at the data coordinates).

        Returns ``(points, obs_shape)`` where *points* has shape
        ``(ndim, n_points)``.
        """
        var_arr = np.asarray(var, dtype=float)
        ndim = self.grid.ndim
        if ndim == 1:
            axes_arrays = [var_arr]
        else:
            axes_arrays = [var_arr[i] for i in range(ndim)]
        coords = []
        for v, g in zip(axes_arrays, self.grid.grid):
            step = g[1] - g[0] if len(g) > 1 else 1.0
            coords.append((v - g[0]) / step)
        points = np.stack([c.ravel() for c in coords], axis=0)
        return jnp.asarray(points), obs_shape

    # ------------------------------------------------------------ core jax functions
    def _model_images(self, theta):
        """Render all bands at *theta* (flat vector in ``self.keys`` order).

        Parameter injection happens by writing the traced values into the
        model's Parameters at trace time; the compiled graph then depends on
        ``theta`` through the traced operations, preserving autodiff.
        """
        self.models.update_parameters(theta)
        images = []
        for b in self._bands:
            img = b['model'].evaluate(*self._coords)
            if b['interp_coords'] is not None:
                points, obs_shape = b['interp_coords']
                img = jax.scipy.ndimage.map_coordinates(
                    img, points, order=1, mode='nearest').reshape(obs_shape)
            if b['conv'] is not None:
                img = b['conv'](img)
            images.append(img)
        return images

    def _log_likelihood(self, theta):
        """Data log-likelihood (priors excluded) at *theta*."""
        images = self._model_images(theta)
        total = 0.0
        loss_func = LOSS_FUNCS[self.loss]
        for b, img, su in zip(self._bands, images, self._scale_uncertainty):
            sigma2 = b['sigma2']
            if su is not None:
                # extra variance: sigma^2 += model^2 * exp(2 * log_f)
                sigma2 = sigma2 + img**2 * jnp.exp(2.0 * su.factor.value)
            total = total + loss_func(img, b['obs'], sigma2, b['keep'])
        return -0.5 * total

    def _log_posterior(self, theta):
        """Log-posterior (log-likelihood + log-priors) at *theta*."""
        return self._log_likelihood(theta) + self.models.get_priors()

    # ------------------------------------------------------------ parameter utilities
    @property
    def parameters(self) -> dict:
        return self.models.parameters

    @property
    def bounds(self) -> list:
        """Ordered bounds matching ``self.keys`` (for scipy optimizers)."""
        return list(self.models.bounds.values())

    @property
    def n_params(self) -> int:
        return len(self.keys)

    @property
    def initial_guess(self) -> np.ndarray:
        return np.array([float(p.value) for _, p in self._free])

    def _clipped_initial_guess(self) -> np.ndarray:
        """Initial guess clipped into the hard bounds."""
        guess, bounds = self.initial_guess, np.array(self.bounds, dtype=float)
        lo = np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf)
        hi = np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf)
        return np.clip(guess, lo, hi)

    def _call_jit(self, func, *args):
        """Call a compiled function, restoring the parameter values that jax
        tracing overwrites in the model's Parameters (tracers leak into
        ``Parameter.value`` during the trace and would break later eager
        evaluation if left in place)."""
        saved = [p.value for _, p in self._free]
        try:
            return func(*args)
        finally:
            for (_, p), v in zip(self._free, saved):
                p.value = v

    def _sync_parameters(self, theta) -> None:
        """Write concrete (non-tracer) values back into the model's Parameters
        after a compiled call, so ``models(grid)`` etc. keep working."""
        self.models.update_parameters(np.asarray(theta, dtype=float))

    def update_parameters(self, params) -> None:
        self.models.update_parameters(params)

    # ------------------------------------------------------------ numpy evaluation helpers
    def log_likelihood(self, theta=None) -> float:
        """Data log-likelihood at *theta* (or at the current parameter values)."""
        if theta is not None:
            self.models.update_parameters(theta)
            theta = self.initial_guess
        else:
            theta = self.initial_guess
        ll = float(self._call_jit(self._loglike_jit, jnp.asarray(theta)))
        self._sync_parameters(theta)
        return ll

    def log_posterior(self, theta) -> float:
        """Log-posterior at *theta* (concrete numpy in/out)."""
        lp = float(self._call_jit(self._logpost_jit, jnp.asarray(theta, dtype=float)))
        self._sync_parameters(theta)
        return lp

    def render(self, theta=None):
        """Model image(s) at *theta* (defaults to the current parameters).

        Returns a list of per-band images, or a single array if there is one
        band.
        """
        if theta is not None:
            self.models.update_parameters(theta)
        theta = self.initial_guess
        images = [np.asarray(im) for im in
                  self._call_jit(self._render_jit, jnp.asarray(theta))]
        self._sync_parameters(theta)
        return images[0] if len(images) == 1 else images

    def chi2_at(self, theta) -> float:
        """Data chi2 (priors excluded) at *theta*."""
        return -2.0 * float(
            self._call_jit(self._loglike_jit, jnp.asarray(theta, dtype=float)))

    def _calculate_diff(self) -> np.ndarray:
        """Masked, sigma-normalized squared residuals at the current parameter
        values (compatibility helper for FittedModel)."""
        images = self._model_images(jnp.asarray(self.initial_guess))
        diffs = []
        for b, img in zip(self._bands, images):
            d = (b['obs'] - img) ** 2 / b['sigma2']
            diffs.append(np.asarray(d)[np.asarray(b['keep'])])
        return np.concatenate(diffs)

    def _data_size(self) -> int:
        return self._n_data

    # ------------------------------------------------------------ MAP inference
    def find_map(self, method: str = 'L-BFGS-B', maxiter: int = None,
                 use_hessian: bool = True, debug: bool = False):
        """Gradient-based MAP optimisation.

        Minimises -log(posterior) with scipy (L-BFGS-B by default) using the
        jitted value-and-gradient.  If ``use_hessian``, the parameter
        covariance is estimated from the jitted Hessian at the optimum
        (Laplace approximation).

        Returns
        -------
        scipy OptimizeResult
        """
        def nlp_and_grad(theta):
            v, g = self._call_jit(self._logpost_valgrad_jit, jnp.asarray(theta))
            return -float(v), -np.asarray(g)

        options = {'maxiter': maxiter} if maxiter is not None else {}
        res = optimize.minimize(
            nlp_and_grad, self._clipped_initial_guess(), jac=True,
            bounds=self.bounds, method=method, options=options,
        )
        if not res.success:
            warnings.warn(f"MAP optimisation did not converge: {res.message}")

        self.best_fit = np.asarray(res.x)
        self._sync_parameters(self.best_fit)
        self._compute_chi2(self.best_fit)

        self.best_fit_error = self._laplace_errors(self.best_fit) if use_hessian else None

        if debug:
            print(f"Parameters: {self.keys}")
            print(f"MAP best-fit: {self.best_fit}")
            print(f"chi2={self.chi2:.4g}, reduced_chi2={self.reduced_chi2:.4g}")
            if self.best_fit_error is not None:
                print(f"Parameter errors: {self.best_fit_error}")
        return res

    def _laplace_errors(self, theta) -> np.ndarray:
        """1-sigma errors from the inverse Hessian (Laplace approximation)."""
        H = np.asarray(
            self._call_jit(self._logpost_hess_jit, jnp.asarray(theta, dtype=float)))
        try:
            pcov = np.linalg.inv(-H)
        except np.linalg.LinAlgError:
            warnings.warn("Hessian is singular; using pseudo-inverse for errors.")
            pcov = np.linalg.pinv(-H)
        return np.sqrt(np.abs(np.diag(pcov)))

    def _compute_chi2(self, theta) -> None:
        """Store chi2 / dof / reduced chi2 (data residuals only)."""
        self.chi2 = self.chi2_at(theta)
        self.dof = max(self._n_data - self.n_params, 1)
        self.reduced_chi2 = self.chi2 / self.dof

    def run(self, debug: bool = False):
        """Backward-compatible alias for :meth:`find_map`."""
        return self.find_map(debug=debug)

    def auto_run(self, debug: bool = False, **kwargs):
        """Backward-compatible alias for :meth:`find_map` (the gradient-based
        MAP with Hessian errors replaces the old two-step minimize +
        least_squares recipe)."""
        return self.find_map(debug=debug, **kwargs)

    def run_de(self, debug: bool = False, **kwargs):
        """Deprecated: differential evolution is superseded by the
        gradient-based MAP.  Kept for API compatibility."""
        warnings.warn("run_de is deprecated; using find_map instead.")
        return self.find_map(debug=debug)

    # ------------------------------------------------------------ SVI / NUTS inference
    def numpyro_model(self, return_model: bool = False) -> Callable:
        """Build the numpyro model (a callable for NUTS/SVI).

        Free parameters are sampled from their numpyro priors
        (:meth:`Parameter.numpyro_prior`); the (compiled) data log-likelihood
        is attached with ``numpyro.factor``.  Priors are contributed by the
        sample sites themselves, so the factor contains the likelihood only.
        """

        def model():
            theta = []
            for key, param in self._free:
                theta.append(numpyro.sample(key, param.numpyro_prior()))
            theta = jnp.stack(theta)
            # _call_jit keeps the Parameter values concrete outside the trace
            numpyro.factor("log_likelihood", self._call_jit(self._loglike_jit, theta))
            if return_model:
                images = self._call_jit(self._render_jit, theta)
                for i, img in enumerate(images):
                    suffix = f"_{i}" if len(images) > 1 else ""
                    deterministic(f"model{suffix}", img)

        return model

    def _init_values_dict(self) -> dict:
        """Current parameter values, clipped into their supports, for use as
        the numpyro initialization strategy.

        Values lying exactly on a finite bound are nudged slightly inward,
        since e.g. Uniform.log_prob is -inf at the support boundary and
        NUTS/SVI would reject the initial point.
        """
        guess = self._clipped_initial_guess()
        bounds = np.array(self.bounds, dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        span = np.where(np.isfinite(lo) & np.isfinite(hi), hi - lo, 1.0)
        eps = 1e-6 * span
        guess = np.where(guess <= lo + eps, lo + eps, guess)
        guess = np.where(guess >= hi - eps, hi - eps, guess)
        return {k: float(v) for k, v in zip(self.keys, guess)}

    def find_map_svi(self, seed: int = 0, max_train: int = 10000,
                     num_round: int = 3, patience: int = 250, debug: bool = False):
        """MAP via numpyro AutoDelta SVI (pysersic-style), trained with early
        stopping and a decreasing learning rate.  Often more robust than the
        scipy optimiser for badly scaled problems."""
        model = self.numpyro_model(return_model=False)
        guide = infer.autoguide.AutoDelta(
            model, init_loc_fn=infer.init_to_value(values=self._init_values_dict()))
        params, _ = train_numpyro_svi_early_stop(
            model, guide, num_round=num_round, max_train=max_train,
            patience=patience, lr_init=0.05, rkey=jax.random.PRNGKey(seed))

        med = guide.median(params)
        self.best_fit = np.array([float(med[k]) for k in self.keys])
        self._sync_parameters(self.best_fit)
        self._compute_chi2(self.best_fit)
        self.svi_result = params
        if debug:
            print(f"SVI-MAP best-fit: {self.best_fit}")
            print(f"chi2={self.chi2:.4g}, reduced_chi2={self.reduced_chi2:.4g}")
        return self.best_fit

    def estimate_posterior(
        self,
        method: str = 'laplace',
        num_sample: int = 1000,
        seed: int = 1,
        return_model: bool = False,
    ):
        """Fast posterior estimation via SVI (pysersic-style).  Generally
        much faster than NUTS, at some accuracy cost.

        Parameters
        ----------
        method : str
            - 'laplace'   : MAP + Gaussian approximation from the Hessian.
            - 'svi-mvn'   : low-rank multivariate-normal variational fit.
            - 'svi-flow'  : normalizing-flow (BNAF) variational fit; most
                            flexible but slowest to train.
        num_sample : int
            Number of posterior samples to draw from the fitted guide.
        """
        assert method in ('laplace', 'svi-mvn', 'svi-flow')
        model = self.numpyro_model(return_model=return_model)
        init = infer.init_to_value(values=self._init_values_dict())

        if method == 'laplace':
            guide = infer.autoguide.AutoLaplaceApproximation(model, init_loc_fn=init)
            elbo = infer.Trace_ELBO()
            lr, rounds, patience, max_train = 0.05, 3, 250, 20000
        elif method == 'svi-mvn':
            guide = infer.autoguide.AutoLowRankMultivariateNormal(
                model, init_scale=5e-3, init_loc_fn=init)
            elbo = infer.TraceMeanField_ELBO()
            lr, rounds, patience, max_train = 0.05, 3, 200, 5000
        else:  # svi-flow
            guide = infer.autoguide.AutoBNAFNormal(
                model, num_flows=1, hidden_factors=[16, 8], init_loc_fn=init)
            elbo = infer.Trace_ELBO(num_particles=8)
            lr, rounds, patience, max_train = 0.05, 3, 500, 20000

        params, _ = train_numpyro_svi_early_stop(
            model, guide, num_round=rounds, max_train=max_train,
            patience=patience, lr_init=lr, ELBO_loss=elbo,
            rkey=jax.random.PRNGKey(seed))
        self.svi_result = params
        self._guide = guide

        # posterior samples from the trained guide
        predictive = infer.Predictive(model=guide, params=params,
                                      num_samples=num_sample)
        raw = predictive(jax.random.PRNGKey(seed + 1))
        self.samples_arr = np.stack(
            [np.asarray(raw[k]) for k in self.keys], axis=-1)

        # MAP-like point estimate + uncertainties from the samples
        self.best_fit_from_samples(self.samples_arr)
        return self.samples_arr

    def sample(
        self,
        num_samples: int = 1000,
        num_warmup: int = 1000,
        num_chains: int = 1,
        seed: int = 0,
        return_model: bool = False,
        sampler_kwargs: dict = None,
        mcmc_kwargs: dict = None,
        progress_bar: bool = True,
    ):
        """Posterior sampling with the numpyro NUTS sampler.

        Parameters
        ----------
        num_samples, num_warmup, num_chains : standard MCMC settings.  For
            ``num_chains > 1`` the host device count is raised automatically
            (call ``numpyro.set_host_device_count`` before any jax array is
            created if this fails).
        sampler_kwargs, mcmc_kwargs : forwarded to ``infer.NUTS`` /
            ``infer.MCMC``.

        Returns
        -------
        numpyro MCMC object (also stored as ``self.mcmc``).
        """
        sampler_kwargs = sampler_kwargs or {}
        mcmc_kwargs = dict(mcmc_kwargs or {})
        if num_chains > 1:
            try:
                numpyro.set_host_device_count(num_chains)
            except Exception as e:
                warnings.warn(f"Could not set host device count ({e}); "
                              f"falling back to num_chains=1.")
                num_chains = 1

        model = self.numpyro_model(return_model=return_model)
        init = infer.init_to_value(values=self._init_values_dict())
        kernel = infer.NUTS(model, init_strategy=init, **sampler_kwargs)
        mcmc = infer.MCMC(
            kernel, num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=progress_bar, **mcmc_kwargs,
        )
        mcmc.run(jax.random.PRNGKey(seed))
        self.mcmc = mcmc

        self.best_fit_from_samples()
        return mcmc

    # ------------------------------------------------------------ posterior access
    def get_samples(self, flat: bool = True, group_by_chain: bool = False) -> np.ndarray:
        """Posterior samples as an array of shape (n_samples, n_params)
        (or (n_chains, n_steps, n_params) when ``group_by_chain``)."""
        if self.mcmc is None:
            raise RuntimeError("No MCMC run found; call sample() first.")
        raw = self.mcmc.get_samples(
            group_by_chain=group_by_chain or not flat)
        return np.stack([np.asarray(raw[k]) for k in self.keys], axis=-1)

    def best_fit_from_samples(self, samples: np.ndarray = None,
                              percentiles: tuple = (16, 50, 84)) -> dict:
        """Summarize the posterior as median +/- 16th/84th percentile errors."""
        if samples is None:
            samples = self.get_samples(flat=True)
        lo, med, hi = np.percentile(samples, percentiles, axis=0)
        self.best_fit = med
        self.best_fit_error_lower = med - lo
        self.best_fit_error_upper = hi - med
        self.best_fit_error = 0.5 * (self.best_fit_error_lower
                                     + self.best_fit_error_upper)
        self._sync_parameters(med)
        self._compute_chi2(med)
        return dict(zip(self.keys, zip(med, self.best_fit_error_lower,
                                       self.best_fit_error_upper)))

    def print_summary(self) -> None:
        """Print the numpyro MCMC summary (incl. divergences)."""
        if self.mcmc is None:
            raise RuntimeError("No MCMC run found; call sample() first.")
        self.mcmc.print_summary(exclude_deterministic=False)

    # ------------------------------------------------------------ plotting
    def plot(self, ax: list = None) -> None:
        """Plot the MCMC chains for each parameter."""
        samples = self.get_samples(flat=False)
        n_params = self.n_params
        if ax is None:
            plt.figure(figsize=(16, 4))
        for i in range(n_params):
            if ax is None:
                ax_i = plt.subplot2grid((n_params, 2), (i, 0), colspan=2)
            else:
                ax_i = ax[i]
            ax_i.plot(samples[:, :, i], "k", alpha=0.2)
            ax_i.set_xlim(0, samples.shape[1])
            if i < n_params - 1:
                ax_i.axes.get_xaxis().set_ticks([])
            ax_i.text(0, 0.8, self.keys[i], ha='left',
                      transform=ax_i.transAxes, fontsize=10)
        ax_i.set_xlabel('steps')
        if ax is None:
            plt.show()

    def corner_plot(self, truths: np.ndarray = None) -> None:
        """Corner plot of the posterior samples."""
        try:
            import corner
        except ImportError:
            print("Install corner for corner_plot")
            return
        corner.corner(self.get_samples(flat=True), labels=self.keys,
                      truths=truths)
        plt.show()

    def summary(self) -> None:
        """Print a fit summary table."""
        lines = [f"{self.name}: {len(self.keys)} free parameters",
                 "-" * 60]
        key_width = max(map(len, self.keys)) + 2
        for i, key in enumerate(self.keys):
            val = self.best_fit[i] if self.best_fit is not None \
                else self.initial_guess[i]
            if self.best_fit_error is not None:
                lines.append(f"{key:<{key_width}} = {val: .5g}  "
                             f"+/- {self.best_fit_error[i]:.3g}")
            else:
                lines.append(f"{key:<{key_width}} = {val: .5g}")
        if self.chi2 is not None:
            lines.append("-" * 60)
            lines.append(f"chi2 = {self.chi2:.4g}   dof = {self.dof}   "
                         f"reduced_chi2 = {self.reduced_chi2:.4g}")
        print("\n".join(lines))


###############################################
# Concrete fitters
###############################################

class MinimizeFitter(JaxFitter):
    """Generic single-band fitter (1D spectra, SEDs, images without PSF).

    Parameters
    ----------
    models, grid, data, mask, name : as in the legacy MinimizeFitter.
        ``data`` is a dict with keys ``'var'``, ``'data'`` and optionally
        ``'data_err'`` / ``'mask'``.
    """

    def __init__(self, models, grid, data: dict, mask=None, name=None,
                 loss: str = 'gaussian'):
        band = dict(
            model=models,
            obs=data['data'],
            err=data.get('data_err'),
            mask=mask if mask is not None else data.get('mask'),
            var=data.get('var'),
            psf=None,
        )
        super().__init__(models, grid, [band], loss=loss, name=name)

class GalaxyFitter(JaxFitter):
    """Single-band galaxy image fitter with PSF convolution.

    Replaces the legacy ``MinimizeGalaxyFitter``: the PSF convolution is
    part of the compiled log-posterior, so MAP gradients and NUTS both
    account for the PSF exactly.

    Parameters
    ----------
    models : Model | CompoundModel | ModelList
        The (composite) model to fit.
    grid : Grid2D
    data : dict with keys ``'var'`` (optional), ``'data'``, ``'data_err'``.
    psf : 2D ndarray, optional
        Pixelized PSF (normalised to unit sum internally).
    mask : 2D bool ndarray, optional (True = masked).
    """

    def __init__(self, models, grid, data: dict, psf=None, mask=None,
                 name=None, loss: str = 'gaussian'):
        band = dict(
            model=models,
            obs=data['data'],
            err=data.get('data_err'),
            mask=mask,
            var=data.get('var'),
            psf=psf,
        )
        super().__init__(models, grid, [band], loss=loss, name=name)

class MultiBandGalaxyFitter(JaxFitter):
    """Joint multiband galaxy fitter (one image per filter, each with its
    own PSF), sharing one parameter vector across bands via a ModelList.

    Parameters
    ----------
    models : ModelList (or list of Models)
        One renderable model per filter, in ``filters`` order.
    grid : Grid2D
        Shared grid; all bands must be sampled on this grid.
    data : dict
        ``{fname: image, fname+'_err': error, fname+'_psf': psf}`` per filter.
    filters : list of str
        Filter names, matching the models order.
    """

    def __init__(self, models, grid, data: dict, filters: list, mask=None,
                 name=None, loss: str = 'gaussian'):
        if isinstance(models, (list, tuple)):
            models = ModelList(list(models))
        bands = []
        for i, fname in enumerate(filters):
            bands.append(dict(
                model=models[i],
                obs=data[fname],
                err=data.get(fname + '_err'),
                mask=mask,
                var=None,   # multiband images are assumed to share the grid
                psf=data.get(fname + '_psf'),
            ))
        super().__init__(models, grid, bands, loss=loss, name=name)

class MultiBandGalaxyFitterDustScreen(JaxFitter):
    """Joint multiband fitter with list-based data (e.g. for dust-screen
    attenuation laws shared across wavelengths via ``link_models``).

    Parameters
    ----------
    models : ModelList (or list of Models)
        One renderable model per band.
    data : dict with list values:
        ``{'wavelength': [...], 'data': [...], 'err': [...], 'psf': [...]}``.
    """

    def __init__(self, models, grid, data: dict, mask=None, name=None,
                 loss: str = 'gaussian'):
        if isinstance(models, (list, tuple)):
            models = ModelList(list(models))
        bands = []
        for i in range(len(data['data'])):
            bands.append(dict(
                model=models[i],
                obs=data['data'][i],
                err=data['err'][i] if 'err' in data else None,
                mask=mask,
                var=None,
                psf=data['psf'][i] if 'psf' in data else None,
            ))
        super().__init__(models, grid, bands, loss=loss, name=name)


# ---- backward-compatibility aliases ----
MinimizeGalaxyFitter = GalaxyFitter
MinimizeGalaxyMultipleFitter = MultiBandGalaxyFitter
MinimizeGalaxyMultipleFitterDustScreen = MultiBandGalaxyFitterDustScreen


###############################################
# Test functions
###############################################

def test_line1d_fitting(plot: bool = False) -> None:
    m_true, b_true = -1., 4.
    N = 50
    x = np.sort(10 * np.random.rand(N))
    grid1d = Grid1D(data=x)
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true + yerr * np.random.randn(N)

    line_model = Line1D(m=-0.8, b=2.0)
    line_model.set_bounds({'m': (-5., 0.5), 'b': (0., 10.)})
    data = {'var': grid1d[0], 'data': y, 'data_err': yerr}

    fitter = MinimizeFitter(line_model, grid1d, data)
    fitter.find_map(debug=True)

    fitter_mcmc = MinimizeFitter(line_model, grid1d, data)
    fitter_mcmc.sample(num_samples=500, num_warmup=500, progress_bar=False)
    fitter_mcmc.best_fit_from_samples()
    print("Truth:", [m_true, b_true])
    print("Best fit (NUTS):", fitter_mcmc.best_fit)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.errorbar(x, y, yerr=yerr, linestyle='none', marker='o', capsize=4)
        ax.plot(x, line_model(grid1d), label='bestfit')
        ax.legend()
        plt.show()

    assert (np.abs(np.array([m_true, b_true]) - fitter.best_fit) < 0.5).all()
    assert (np.abs(np.array([m_true, b_true]) - fitter_mcmc.best_fit) < 0.5).all()

def test_gaussian1d_fitting(plot: bool = False) -> None:
    grid_fit = Grid1D(size=100, pixelsize=0.5)
    grid_data = Grid1D(size=50, pixelsize=1.0)
    amp0, mean0, sigma0 = 1., 0., 2.

    gmodel = Gaussian1D(amplitude=amp0, mean=mean0, sigma=sigma0, cont=0.,
                        name='gaussian1')
    gmodel.set_bounds({'amplitude': (0, 2), 'mean': (-2, 2), 'sigma': (0, 5)})
    y_data = gmodel(grid_data)
    np.random.seed(1994)
    y_err = 0.2 * (np.random.random(y_data.shape) - 0.5)
    y_data = y_data + y_err
    data = {'var': grid_data[0], 'data': y_data, 'data_err': np.abs(y_err)}

    start = time.time()
    fitter = MinimizeFitter(gmodel, grid_fit, data)
    fitter.find_map(debug=True)
    print(f"MAP used {time.time() - start:.2f}s")

    start = time.time()
    fitter_nuts = MinimizeFitter(gmodel, grid_fit, data)
    fitter_nuts.sample(num_samples=500, num_warmup=500, progress_bar=False)
    print(f"NUTS used {time.time() - start:.2f}s")
    fitter_nuts.best_fit_from_samples()

    print("Truth:", [amp0, mean0, sigma0])
    print('Best fit MAP:', fitter.best_fit, '+/-', fitter.best_fit_error)
    print('Best fit NUTS:', fitter_nuts.best_fit)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for ax_i in ax:
            ax_i.errorbar(grid_data[0], y_data, yerr=np.abs(y_err),
                          linestyle='none', marker='o', capsize=4)
        ax[0].plot(grid_fit[0], gmodel(grid_fit))
        ax[0].text(.05, .9, 'MAP', transform=ax[0].transAxes)
        ax[1].plot(grid_fit[0], gmodel(grid_fit))
        ax[1].text(.05, .9, 'NUTS', transform=ax[1].transAxes)
        plt.show()

    assert (np.abs(np.array([amp0, mean0, sigma0]) - fitter.best_fit) < 0.2).all()

def test_gaussian2d_fitting(plot: bool = False, debug: bool = False) -> None:
    grid2d = Grid2D(size=20, pixelsize=1)
    amp0, x0, y0, xsig0, ysig0, theta0 = 1., 0., 2., 2., 3., 0.

    gmodel = Gaussian2D(amplitude=amp0, x0=x0, y0=y0,
                        x_sigma=xsig0, y_sigma=ysig0, theta=theta0)
    gmodel.set_bounds({
        'amplitude': (0, 2), 'x0': (-5, 5), 'y0': (-5, 5),
        'x_sigma': (0, 5), 'y_sigma': (0, 5),
        'theta': (-0.5 * np.pi, 0.5 * np.pi),
    })
    obs_data = gmodel(grid2d)
    np.random.seed(1994)
    obs_data = obs_data + 0.1 * (np.random.random(obs_data.shape) - 0.5)
    data = {'var': grid2d.meshgrid, 'data': obs_data,
            'data_err': 0.1 * np.ones_like(obs_data)}

    truths = np.array(list(gmodel.parameters.values()))
    initial_guess = truths * (1 + 0.1 * np.random.randn(len(truths)))
    gmodel.update_parameters(initial_guess)
    print("Truths:", truths)
    print("Initial guess:", initial_guess)

    fitter = MinimizeFitter(gmodel, grid2d, data)
    fitter.find_map(debug=debug)
    print('Best fit MAP:', fitter.best_fit, '+/-', fitter.best_fit_error)

    fitter_nuts = MinimizeFitter(gmodel, grid2d, data)
    fitter_nuts.sample(num_samples=500, num_warmup=500, progress_bar=False)
    fitter_nuts.best_fit_from_samples()
    print('Best fit NUTS:', fitter_nuts.best_fit)

    if plot:
        plt.figure(figsize=(10, 3))
        best_fit_model = gmodel(grid2d)
        for i, img in enumerate([obs_data, best_fit_model, obs_data - best_fit_model]):
            ax_i = plt.subplot(1, 3, i + 1)
            ax_i.imshow(img, origin='lower')
        plt.show()

    assert (np.abs(truths - fitter.best_fit) < 0.5).all()

def test_sersic2d_fitting(debug: bool = False) -> None:
    grid2d = Grid2D(size=20, pixelsize=1)
    amplitude0, reff0, n0, x0, y0, ellip0, theta0 = (
        0.4, 4., 1., 0.2, -0.2, 0.4, np.pi / 3)

    smodel = Sersic2D(amplitude=amplitude0, reff=reff0, n=n0,
                      x0=x0, y0=y0, ellip=ellip0, theta=theta0)
    smodel.set_bounds({
        'amplitude': (0, 20), 'reff': (0.1, 30), 'n': (0.5, 8),
        'x0': (-10, 10), 'y0': (-10, 10),
        'ellip': (0, 1), 'theta': (-0.5 * np.pi, 0.5 * np.pi),
    })
    obs_data = smodel(grid2d)
    np.random.seed(1994)
    obs_data = obs_data + 0.01 * (np.random.random(obs_data.shape) - 0.5)
    data = {'var': grid2d.meshgrid, 'data': obs_data,
            'data_err': np.sqrt(np.abs(obs_data) * 0.01 + 0.02**2)}

    truths = np.array(list(smodel.parameters.values()))
    initial_guess = np.array([1., 5., 2., 0., 0., 0.5, 0.])
    print("Truths:", truths)
    print("Initial guess:", initial_guess)
    smodel.update_parameters(initial_guess)

    fitter = MinimizeFitter(smodel, grid2d, data)
    fitter.find_map(debug=debug)
    print('Best fit MAP:', fitter.best_fit, '+/-', fitter.best_fit_error)
    assert (np.abs(truths - fitter.best_fit) < 0.2).all()

def test_galaxy_with_psf() -> None:
    """Bulge+disk with PSF convolution: MAP + NUTS + multiband."""
    grid2d = Grid2D(size=40, pixelsize=1)
    psf_arr = np.zeros((11, 11))
    psf_arr[5, 5] = 0.6
    psf_arr[4:7, 5] = 0.1
    psf_arr[5, 4:7] = 0.1

    bulge = Sersic2D(amplitude=0.5, reff=2., n=2., x0=0.3, y0=-0.2,
                     ellip=0.2, theta=0.3, name='bulge')
    disk = Sersic2D(amplitude=0.1, reff=8., n=1., x0=0.3, y0=-0.2,
                    ellip=0.3, theta=0.3, name='disk')
    galaxy = bulge + disk
    galaxy.tie('bulge.x0', 'disk.x0')
    galaxy.tie('bulge.y0', 'disk.y0')
    galaxy.set_bounds({'bulge.reff': (0.1, 10), 'disk.reff': (2, 30),
                       'bulge.n': (0.5, 8), 'disk.n': (0.5, 8)})

    # mock two-band data on the same grid, each convolved with its own PSF
    data = {'g': galaxy(grid2d) + 0.01 * np.random.randn(*grid2d.shape),
            'g_err': 0.02 * np.ones(grid2d.shape),
            'g_psf': psf_arr,
            'r': galaxy(grid2d) + 0.01 * np.random.randn(*grid2d.shape),
            'r_err': 0.02 * np.ones(grid2d.shape),
            'r_psf': psf_arr}
    models = ModelList([bulge, disk])
    fitter = MultiBandGalaxyFitter(models, grid2d, data, filters=['g', 'r'])
    fitter.find_map(debug=True)
    print("Best fit:", dict(zip(fitter.keys, fitter.best_fit)))
    assert fitter.reduced_chi2 < 10.

    fitter_nuts = MultiBandGalaxyFitter(models, grid2d, data, filters=['g', 'r'])
    fitter_nuts.sample(num_samples=300, num_warmup=300, progress_bar=False)
    fitter_nuts.best_fit_from_samples()
    print("Best fit NUTS:", dict(zip(fitter_nuts.keys, fitter_nuts.best_fit)))
    assert fitter_nuts.reduced_chi2 < 10.

def test_psf_convolver() -> None:
    from scipy.signal import fftconvolve
    grid2d = Grid2D(size=20, pixelsize=1)
    psf_arr = np.zeros((5, 5))
    psf_arr[2, 2] = 0.6
    psf_arr[1:4, 2] = 0.2
    img = np.random.random(grid2d.shape)
    conv = PSFConvolver(psf_arr, grid2d.shape, normalize=True)
    assert np.allclose(conv(img), fftconvolve(img, psf_arr / psf_arr.sum(),
                                              mode='same'), atol=1e-12)
    print("PSFConvolver matches scipy.signal.fftconvolve(mode='same')")


if __name__ == "__main__":
    print(f"galaxyfit_jax.modeling_utils v{__version__}")
    test_psf_convolver()
    test_line1d_fitting()
    test_gaussian1d_fitting()
    test_gaussian2d_fitting()
    test_sersic2d_fitting()
    test_galaxy_with_psf()
    print("All modeling_utils self-tests passed.")
