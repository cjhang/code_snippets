"""
################################################################################

Copyright (C) 2017-2022, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your research,
I would appreciate an acknowledgement to the use of the
"CapFit constrained least-squares optimization program, which combines
the Sequential Quadratic Programming and the Levenberg-Marquardt methods
and is included in the pPXF Python package of Cappellari (2017)".

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.
In particular, redistribution of the code is not allowed.

###############################################################################

Changelog
---------

V2.5.1: MC, Oxford, 22 May 2023
-------------------------------

- ``capfit``: Relaxed tolerance when checking initial guess feasibility.

V2.5.0: MC, Oxford, 16 August 2022
----------------------------------

- Uses ``scipy.optimize.linprog`` to find feasible starting point in ``lsq_lin``.
- Set default ``linear_method='lsq_lin'`` in ``capfit``. This eliminates the
  need to install ``cvxopt`` when using general linear constraints.

V2.4.0: MC, Oxford, 04 March 2022
---------------------------------

- Remove the non-free variables before the optimization.
  This reduces the degeneracy of the Jacobian.

V2.3.0: MC, Oxford, 20 December 2020
------------------------------------

- New ``linear_method`` keyword to select ``cvxopt`` or ``lsq_lin``,
  for cases where the latter stops, when using linear constraints.
  Thanks to Kyle Westfall (UCO Lick) for a detailed bug report.

V2.2.1: MC, Oxford, 11 September 2020
------------------------------------

- Fixed possible infinite loop in ``lsq_box`` and ``lsq_lin``.
  Thanks to Shravan Shetty (pku.edu.cn) for the detailed report.
- Use Numpy rather than Scipy version of ``linalg.lstsq`` to avoid
  a Scipy bug in the default criterion for rank deficiency.
- Pass ``rcond`` keyword to ``cov_err`` for consistency.

V2.2.0: MC, Oxford, 20 August 2020
----------------------------------

- New function ``lsq_lin`` implementing a robust linear least-squares
  linearly-constrained algorithm which works with a rank-deficient matrix
  and allows for a starting guess. ``lsq_lin`` supersedes the former ``lsqlin``.
- Renamed ``lsqbox`` to ``lsq_box`` and revised its interface.

V2.1.0: MC, Oxford, 09 July 2020
--------------------------------

- New function ``lsqbox`` implementing a fast linear least-squares
  box-constrained (bounds) algorithm which allows for a starting guess.

V2.0.2: MC, Oxford, 20 June 2020
--------------------------------

- ``capfit``: new keyword ``cond`` (passed to ``lsqlin``).
- ``capfit``: Use ``bvls`` to solve quadratic subproblem with only ``bounds``.

V2.0.1: MC, Oxford, 24 January 2020
-----------------------------------

- New keyword ``cond`` for ``lsqlin``.
- Relaxed assertion for inconsistent inequalities in ``lsqlin`` to avoid false
  positives. Thanks to Kyle Westfall (UCO Lick) for a detailed bug report.

V2.0.0: MC, Oxford, 10 January 2020
-----------------------------------

- Use the new general linear least-squares optimization
  function``lsqlin`` to solve the quadratic sub-problem.
- Allow for linear inequality/equality constraints
  ``A_ineq``, ``b_ineq`` and  ``A_eq``, ``b_eq``

V1.0.7: MC, Oxford, 10 October 2019
-----------------------------------

- Included complete documentation.
- Improved print formatting.
- Return ``.message`` attribute.
- Improved ``xtol`` convergence test.
- Only accept final move if ``chi2`` decreased.
- Strictly satisfy bounds during Jacobian computation.

V1.0.6: MC, Oxford, 11 June 2019
++++++++++++++++++++++++++++++++

- Use only free parameters for small-step convergence test.
- Explain in words convergence status when verbose != 0
- Fixed program stop when abs_step is undefined.
- Fixed capfit ignoring optional max_nfev.

V1.0.5: MC, Oxford, 28 March 2019
+++++++++++++++++++++++++++++++++

- Raise an error if the user function returns non-finite values.

V1.0.4: MC, Oxford, 30 November 2018
++++++++++++++++++++++++++++++++++++

- Allow for a scalar ``abs_step``.

V1.0.3: MC, Oxford, 20 September 2018
+++++++++++++++++++++++++++++++++++++

- Raise an error if one tries to tie parameters to themselves.
  Thanks to Kyle Westfall (Univ. Santa Cruz) for the feedback.
- Use Python 3.6 f-strings.

V1.0.2: MC, Oxford, 10 May 2018
+++++++++++++++++++++++++++++++

- Dropped legacy Python 2.7 support.

V1.0.1: MC, Oxford, 13 February 2018
++++++++++++++++++++++++++++++++++++

- Make output errors of non-free variable exactly zero.

V1.0.0: MC, Oxford, 15 June 2017
++++++++++++++++++++++++++++++++

- Written by Michele Cappellari

"""
import numpy as np
from scipy import linalg, optimize

try:
    from cvxopt import matrix, solvers
    cvxopt_installed = True
except ImportError:
    cvxopt_installed = False

################################################################################

class lsq_lin_cvxopt:
    """
    Linear least-squares with linear equality and/or inequality constraints.

    Given a matrix A and vector b, find x which solves::

          Minimize      || A @ x - b ||
          Subject to    A_ineq @ x <= b_ineq
          and           A_eq @ x == b_eq
          and           bounds[0] <= x <= bounds[1]

    example:

        sol = lsq_lin_cvxopt(A_lsq, b_lsq, A_in, b_in, A_eq, b_eq)
        print("solution:", sol.x)

    """
    def __init__(self, A, b, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None,
                 bounds=None, initvals=None, options=None):

        assert cvxopt_installed, "The cvxopt package must be installed"
        opt = {'show_progress': 0}  # default
        if options is not None:
            opt.update(options)         # |= in Python 3.9

        # Convert bounds to linear inequality constraints
        if (bounds is not None) and np.any(np.isfinite(bounds)):
            m, n = A.shape
            bounds = np.array([np.resize(b, n) for b in bounds])
            w = np.isfinite(bounds).ravel()
            A_bnd = np.vstack([-np.identity(n), np.identity(n)])[w]
            b_bnd = np.append(-bounds[0], bounds[1])[w]

            if A_ineq is None:
                A_ineq, b_ineq = A_bnd, b_bnd
            else:
                A_ineq = np.vstack([A_ineq, A_bnd])
                b_ineq = np.append(b_ineq, b_bnd)

        if A_ineq is not None:
            A_ineq, b_ineq = matrix(A_ineq, tc='d'), matrix(b_ineq, tc='d')

        if A_eq is not None:
            A_eq, b_eq = matrix(A_eq, tc='d'), matrix(b_eq, tc='d')

        P = matrix(A.T @ A)
        q = -matrix(A.T @ b)
        sol = solvers.coneqp(P, q, A_ineq, b_ineq, None, A_eq, b_eq,
                             options=opt, initvals=initvals)

        self.initvals = sol
        self.x = np.squeeze(sol['x'])
        self.cost = chi2(A @ self.x - b)/2

###############################################################################

def fprint(x):
    return (" {:#.4g}"*len(x)).format(*x)

################################################################################

def chi2(x):
    return x @ x

################################################################################

def cov_err(jac, rcond=None):
    """
    Covariance and 1sigma formal errors (i.e. assuming diagonal covariance).
    See e.g. Press et al. 2007, Numerical Recipes, 3rd ed., Section 15.4.2

    """
    U, s, Vh = linalg.svd(jac, full_matrices=False)
    tol = np.spacing(s[0])*max(jac.shape) if rcond is None else rcond*s[0]
    w = s > tol
    cov = (Vh[w].T/s[w]**2) @ Vh[w]
    perr = np.sqrt(np.diag(cov))

    return cov, perr

###############################################################################

def lsq_eq(A, b, A_eq, b_eq, rcond=None):
    """
    Linear least-squares problem with linear equality constraints::

          Minimize      || A @ x - b ||
          Subject to    A_eq @ x == b_eq

    Uses algorithm 6.2.2 of Golub & van Loan, 2013, "Matrix Computations 4th ed".
    Unlike linalg.lapack.dgglse this function allows for a rank-deficient A.

    """
    if (A_eq is None) or (len(A_eq) == 0):
        x = np.linalg.lstsq(A, b, rcond=rcond)[0]
    else:
        p = b_eq.size
        Q, R = linalg.qr(A_eq.T)
        y = linalg.solve_triangular(R[:p, :p], b_eq, trans='T')
        AQ = A @ Q
        z = np.linalg.lstsq(AQ[:, p:], b - AQ[:, :p] @ y, rcond=rcond)[0]
        x = Q[:, :p] @ y + Q[:, p:] @ z

    return x

###############################################################################

class lsq_lin:
    """
    Linear least-squares with linear equality and/or inequality constraints.

    Given a matrix A and vector b, find x which solves::

          Minimize      || A @ x - b ||
          Subject to    A_ineq @ x <= b_ineq
          and           A_eq @ x == b_eq
          and           bounds[0] <= x <= bounds[1]

    The main loop of this algorithm is a generalization of the active-set NNLS
    Algorithm (23.10) in Lawson & Hanson, 1995, "Solving Lesat Squares Problems".
    http://doi.org/10.1137/1.9781611971217
    The generalization to linear constraints follows the ideas presented
    in Sec. 16.5 of Nocedal & Wright, 2006 "Numerical Optimization, 2nd ed".
    https://doi.org/10.1007/978-0-387-40065-5

    rcond:
        Cutoff for small singular values used to determine
        the effective rank of A. Singular values smaller than
        rcond*largest_singular_value are considered zero.
    ftol:
        Safety criterion to stop iterating. The minimum is generally
        computed to machine accuracy regardless of this value.

    """
    def __init__(self, A, b, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None,
                 bounds=None, x=None, verbose=False, ftol=1e-10, rcond=None):

        A, b = map(np.asarray, (A, b))
        m, n = A.shape
        assert m == b.size, "A/b size mismatch"

        if A_ineq is not None:
            A_ineq, b_ineq = map(np.asarray, (A_ineq, b_ineq))
            q, ni = A_ineq.shape
            assert q == b_ineq.size, "A_ineq/b_ineq size mismatch"
            assert ni == n, "A_ineq/A size mismatch"

        if A_eq is None:
            p = 0
        else:
            A_eq, b_eq = map(np.asarray, (A_eq, b_eq))
            p, ne = A_eq.shape
            assert p == b_eq.size, "A_eq/b_eq size mismatch"
            assert ne == n, "A_eq/A size mismatch"

        # Convert bounds to linear inequality constraints
        if (bounds is not None) and np.any(np.isfinite(np.append(*bounds))):
            bounds = np.array([np.resize(b, n) for b in bounds])
            w = np.isfinite(bounds).ravel()
            A_bnd = np.vstack([-np.eye(n), np.eye(n)])[w]
            b_bnd = np.append(-bounds[0], bounds[1])[w]

            if A_ineq is None:
                A_ineq, b_ineq = A_bnd, b_bnd
            else:
                A_ineq = np.vstack([A_ineq, A_bnd])
                b_ineq = np.append(b_ineq, b_bnd)

        if x is None or A_ineq is None:
            x = lsq_eq(A, b, A_eq, b_eq, rcond=rcond)
            if A_ineq is None or np.all(A_ineq @ x <= b_ineq):
                self.x = x
                self.cost = chi2(b - A @ x)/2
                if verbose:
                    print("The unconstrained solution is optimal")
                return

        eps = ftol*linalg.norm(A_ineq, axis=1)

        def solve_subproblem(free):
            if A_eq is None:
                Ae = A_ineq[~free]
                be = b_ineq[~free]
            else:
                Ae = np.vstack([A_eq, A_ineq[~free]])
                be = np.append(b_eq, b_ineq[~free])
            x = lsq_eq(A, b, Ae, be, rcond=rcond)
            out = A_ineq[free] @ x > b_ineq[free] + eps[free]
            return x, out

        if verbose:
            ita = 0
            print("  Iteration     Cost        Cost reduction")
            print(f"  {0:6d} {chi2(b - A @ x)/2:15.4e}")

        if np.all(A_ineq @ x <= b_ineq):
            free = A_ineq @ x < b_ineq - eps
        else:
            # Find feasible initial point
            q, ni = A_ineq.shape
            c = np.append(np.zeros(n), np.ones(p + q))
            A_ub = np.hstack([A_ineq, np.eye(q), np.zeros((q, p))])
            bnd = [(None, None)]*n + [(0, None)]*(p + q)
            if A_eq is not None:
                Ae, be = np.hstack([-A_eq, np.zeros((p, q)), np.diag(np.sign(b_eq))]), -b_eq
            else:
                Ae = be = None
            opt = {'disp': True} if verbose else None
            res = optimize.linprog(c, A_ub, b_ineq, Ae, be, bnd, method='highs', options=opt)
            x, free = res.x[:n], res.slack > eps

        # This is the main loop, starting from a feasible point
        chi2new, chi2old = chi2(b - A @ x), 1e300
        while True:   # loop A
            while True:    # loop B
                z, out = solve_subproblem(free)
                if np.any(out):
                    alpha = (b_ineq[free][out] - A_ineq[free][out] @ x)/(A_ineq[free][out] @ (z - x))
                    k = np.argmin(alpha)
                    x += alpha[k]*(z - x)
                    free[np.flatnonzero(free)[out][k]] = False
                    if verbose:
                        print(f"b*{ita:6d}** {chi2(b - A @ x)/2:13.4e}   alpha: {alpha[k]:.4f}")
                else:
                    x = z
                    break
            chi2new, chi2old = chi2(b - A @ x)/2, chi2new
            if verbose:
                ita += 1
                print(f"a {ita:6d} {chi2new:15.4e} {chi2old - chi2new:15.4e}")
            if np.all(free):
                if verbose:
                    print("No more constraints to free")
                break
            if abs(chi2old - chi2new) < ftol*chi2new:
                if verbose:
                    print(f"Fractional decrease less than {ftol:.2g}")
                break
            Ae = A_ineq[~free] if A_eq is None else np.vstack([A_eq, A_ineq[~free]])
            lam = np.linalg.lstsq(Ae.T, A.T @ (b - A @ x), rcond=rcond)[0][p:]
            j = np.argmin(lam)
            if lam[j] < 0:
                free[np.flatnonzero(~free)[j]] = True
            else:
                if verbose:
                    print("All Lagrange multipliers are non-negative")
                break

        self.x = x
        self.active_mask = ~free
        self.cost = chi2new/2

###############################################################################

class lsq_box:
    """
    Fast linear least-squares with box (bounds) constraints.

    Given a matrix A and vector b, and an optional starting guess for x,
    find a vector x which solves::

          Minimize      || A @ x - b ||
          Subject to    bounds[0] <= x <= bounds[1]

    This function implements an improved and faster version of the
    BVLS algorithm by Lawson & Hanson (1995) and Stark & Parker (1995),
    which both generalize the L&H active-set NNLS algorithm (23.10).
    This implementation allows for an initial value for x, which can
    dramatically speed up convergence when a good guess is available.
    I also included an initialization phase, which is an analogue to that
    suggested by Bro & de Jong (1997) to speed up the NNLS algorithm.

    rcond:
        Cutoff for small singular values used to determine
        the effective rank of A. Singular values smaller than
        rcond*largest_singular_value are considered zero.
    ftol:
        Safety criterion to stop iterating. The minimum is generally
        computed to machine accuracy regardless of thie value.

    """
    def __init__(self, A, b, bounds, x=None, rcond=None, verbose=False, ftol=1e-10):

        m, n = A.shape
        assert m == b.size, "A/b size mismatch"
        bounds = np.asarray([np.resize(b, n) for b in bounds])
        lb, ub = bounds
        assert np.all(lb <= ub), "must be lower bound <= upper bound"

        if x is None:
            x = np.linalg.lstsq(A, b, rcond=rcond)[0]
            if np.all((lb <= x) & (x <= ub)):
                self.x = x
                self.active_mask = np.zeros(n, dtype=bool)
                self.cost = chi2(b - A @ x)/2
                if verbose:
                    print("The unconstrained solution is optimal")
                return

        if verbose:
            ita = 0
            print("  Iteration     Cost        Cost reduction")
            print(f"  {ita:6d} {chi2(b - A @ x)/2:15.4e}")

        out = True
        while np.any(out):  # Initialization loop
            x = x.clip(*bounds)
            free = (lb < x) & (x < ub)
            x[free] = np.linalg.lstsq(A[:, free], b - A[:, ~free] @ x[~free], rcond=rcond)[0]
            out = (x < lb) | (ub < x)
            if verbose:
                ita += 1
                print(f"a*{ita:6d} {chi2(b - A @ x)/2:15.4e}")

        chi2new, chi2old = chi2(b - A @ x), 1e300
        while np.any(~free) and (chi2old - chi2new > ftol*chi2new):    # Loop A
            d = np.select([x <= lb, ub <= x], [1, -1])
            wd = A.T @ (b - A @ x)*d
            j = np.argmax(wd)
            if wd[j] > 0:
                free[j] = out = True
                while np.any(out):    # Loop B
                    z = x.clip(*bounds)
                    z[free] = np.linalg.lstsq(A[:, free], b - A[:, ~free] @ x[~free], rcond=rcond)[0]
                    out = (z < lb) | (ub < z)
                    if np.any(out):
                        bn = np.select([z < lb, ub < z], bounds)
                        alpha = (bn[out] - x[out])/(z[out] - x[out])
                        k = np.argmin(alpha)
                        x[free] += alpha[k]*(z[free] - x[free])
                        p = np.flatnonzero(out)[k]
                        x[p] = bn[p]  # Place blocking constraint exactly on bound
                        free[p] = False
                        if verbose:
                            print(f"b*{ita:6d}** {chi2(b - A @ x)/2:13.4e}   alpha: {alpha[k]:.4f}")
                    else:
                        x = z
                chi2old, chi2new = chi2new, chi2(b - A @ x)
            else:
                if verbose:
                    print("KKT condition satisfied")
                break
            if verbose:
                ita += 1
                print(f"a {ita:6d} {chi2new/2:15.4e} {(chi2old - chi2new)/2:15.4e}")

        self.x = x
        self.active_mask = ~free
        self.cost = chi2new/2

###############################################################################

class capfit:
    """
    CapFit
    ------

    ``CapFit`` solves linearly-constrained least-squares optimization problems.
    Linear inequality/equality constraints and bound constraints are supported.
    Moreover one can easily tie or fix some parameters without having to
    modify the fitting function.

    ``CapFit`` combines two successful ideas:

        (i) The Sequential Quadratic Programming (SQP) approach,
            specialized for the case of linear constraints;
        (ii) The Levenberg-Marquardt (LM) method.

    It was designed for the common situations where the user function is not
    a simple analytic function but is the result of some complex calculations
    and is more expensive to compute than the small quadratic subproblem.

    I found ``CapFit`` performance generally better, in terms of robustness
    and number of functions evaluations, than the best uncostrained or
    bound-constrained least-squares algorithms currently available, but
    in addition ``CapFit`` allows for more general constraints.

    Given a function of ``n`` model parameters ``x_k`` returning the ``m``
    model residuals ``f_j(x)``, ``CapFit`` finds a local minimum of the cost
    function::

        G(x) = sum[f_j(x)^2]

    subject to::

        A_ineq @ x <= b_ineq            # Linear Inequality Constraints
        A_eq @ x == b_eq                # Linear Equality Constraints
        bounds[0] <= x <= bounds[1]     # Bounds
        x_k = f(x)                      # Tied Parameters
        x_k = a_k                       # Fixed Parameters

    Attribution
    -----------

    If you use this software for your research, please cite the Python package
    ``ppxf`` by `Cappellari (2017)
    <http://adsabs.harvard.edu/abs/2017MNRAS.466..798C>`_, where this
    software was introduced. The BibTeX entry for the paper is::

        @ARTICLE{Cappellari2017,
            author = {{Cappellari}, M.},
            title = "{Improving the full spectrum fitting method:
                accurate convolution with Gauss-Hermite functions}",
            journal = {MNRAS},
            eprint = {1607.08538},
            year = 2017,
            volume = 466,
            pages = {798-811},
            doi = {10.1093/mnras/stw3020}
        }

    Calling Sequence
    ----------------

    .. code-block:: python

        res = capfit(func, p1, A_eq=None, A_ineq=None, abs_step=None, b_eq=None,
                     b_ineq=None, bounds=(-np.inf, np.inf), diff_step=1e-4,
                     fixed=None, ftol=1e-4, linear_method='cvxopt',
                     max_nfev=None, rcond=None, tied=None, verbose=0,
                     x_scale='jac', xtol=1e-4, args=(), kwargs={})

        print(f"solution: {res.x}")

    Usage Examples
    --------------

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt

        from ppxf.capfit import capfit

        def model(p, x, a):
            return p[0]*np.exp(-0.5*(x - p[1]/a)**2/p[2]**2)

        def resid(p, x=None, y=None, yerr=None, a=None):
            ymod = model(p, x, a)
            return (y - ymod)/yerr

        a = 1.0
        x = np.linspace(-3, 3, 100)
        ptrue = np.array([2., -1., 0.5])
        y = model(ptrue, x, a)
        yerr = np.full_like(y, 0.1)
        y += np.random.normal(0, yerr, x.size)
        p0 = np.array([1., 1., 1.])
        kwargs = {'x': x, 'y': y, 'yerr': yerr, 'a': a}

        print("#### Unconstrained case ####")
        res = capfit(resid, p0, kwargs=kwargs, verbose=1)

        print("#### Bounds on parameters ####")
        res = capfit(resid, p0, kwargs=kwargs, verbose=1,
                     bounds=[(-np.inf, -0.95, 0.55), np.inf])

        print("#### Tied parameters ####")
        res = capfit(resid, p0, kwargs=kwargs, tied=['', '-p[0]/2', ''], verbose=1)

        print("#### Fixed parameters ####")
        res = capfit(resid, [1, 1, 0.5], kwargs=kwargs, fixed=[0, 0, 1], verbose=1)

        plt.plot(x, y, 'o')
        plt.plot(x, model(res.x, x, a))

    Input Parameters
    ----------------

    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an 1-d darray of shape (n,).
        The function must return a 1-d array of shape (m,).
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. For guaranteed convergence, the
        initial guess must be feasible (satsfies the constraints) and for this
        reason an error is returned if it is not the case.


    Optional Keywords
    -----------------

    abs_step : None, scalar or array_like, optional
        Determines the absolute step size for the finite difference
        approximation of the Jacobian. If ``abs_step`` is given, then the value
        of ``diff_step`` is ignored and the finite differences use this fixed
        absolute step.
    A_eq: array_like with shape (q, n), optional
        Defines the linear equality constraints on the fitted parameters::

            A_eq @ x == b_eq

        The same result can be achieved using the ``tied`` keyword,
        which also allows for non-linear equality constraints.
    A_ineq: array_like with shape (p, n), optional
        Defines the linear inequality constraints on the fitted parameters::

            A_ineq @ x <= b_ineq

    b_ineq: array_like with shape (p), optional
        See description of ``A_ineq``.
    b_eq: array_like with shape (q), optional
        See description of ``A_eq``.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds (lb, ub) on independent variables.
        Defaults to no bounds. Each array must match the size of `x0` or be a
        scalar, in the latter case a bound will be the same for all variables.
        Use ``np.inf`` with an appropriate sign to disable bounds on all or
        some variables.
    diff_step : None, scalar or array_like, optional
        Determines the relative step size for the finite difference
        approximation of the Jacobian. The actual step is computed as
        ``diff_step*maximum(1, abs(x))`` (default ``diff_step=1e-4``)
    fixed:
        Boolean vector set to ``True`` where a given parameter has to be held
        fixed with the value given in ``x0``.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function (default
        is 1e-4). The optimization process is stopped when both
        ``prered < ftol`` and ``abs(actred) < ftol`` and additionally
        ``actred <= 2*prered``, where ``actred`` and ``prered`` are the actual
        and predicted relative changes respectively
        (as described in More' et al. 1980).
    linear_method: str, optional
        Method used for the solution of the linear least-squares sub-problem.
        When using linear constraints, this can be either 'lsq_lin' or 'cvxopt'
        (default). In the latter case, the cvxopt package must be installed.
        With only bounds, this keyword is ignored as ``capfit`` uses 'lsq_box'.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination
        (default is 100*n).
    rcond : float, optional
        Cutoff for small singular values used to determine the effective rank of
        the Jacobian. Singular values smaller than rcond*largest_singular_value
        are considered zero.
    tied : array_like with shape (n,), optional
        A list of string expressions. Each expression "ties" the parameter to
        other free or fixed parameters.  Any expression involving constants and
        the parameter array ``p[j]`` are permitted. Since they are totally
        constrained, tied parameters are considered to be fixed; no errors are
        computed for them.

        This is a vector with the same dimensions as ``x0``. In practice,
        for every element of ``x0`` one needs to specify either an empty string
        ``''`` implying that the parameter is free, or a string expression
        involving some of the variables ``p[j]``, where ``j`` represents the
        index of the vector of parameters. See usage example.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting `x_scale` is equivalent
        to reformulating the problem in scaled variables ``xs = x/x_scale``.
        An alternative view is that the initial size/2 of the box trust region
        along j-th dimension is given by ``x_scale[j]`` and the box ratains its
        shape during the optimization. Improved convergence is achieved by
        setting `x_scale` such that a step of a given size along any of the
        scaled variables has a similar effect on the cost function.  If set to
        'jac', the scale is iteratively updated using the inverse norms of the
        columns of the Jacobian matrix (as described in More' 1978).
    xtol : float or None, optional
        Tolerance for termination by the change ``dx`` of the independent
        variables (default is 1e-4). The condition is
        ``norm(dx) < xtol*(xtol + norm(xs))`` where ``xs`` is the value of ``x``
        scaled according to `x_scale` parameter (see below).
        If None, the termination by this condition is disabled.
    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun`, empty by default.
        The calling signature is ``fun(x, *args, **kwargs)``.

    Returns
    -------

    The following attributes of the ``capfit`` class are defined in output:

    .x: ndarray, shape (n,)
        Solution found.
    .x_err: ndarray, shape (n,)
        Formal uncertainties of the solution estimated from diagonal elements of
        the covariance matrix (i.e. ignoring covariance). This is only meaningful
        if the ``fun`` values represent residuals divided by their 1sigma errors.
        Uncertainties of the non-free parameters are returned as zero.
    .cost: float
        Value of the ``cost = 0.5*(fun @ fun)`` function at the solution.
    .cov: ndarray, shape (nfree, nfree)
        Covariance matrix for the *free* (not tied or fixed) parameters.
    .fun: ndarray, shape (m,)
        Vector of residuals at the solution.
    .jac: ndarray, shape (m, nfree)
        Modified Jacobian matrix at the solution, in the sense that J.T @ J
        is a Gauss-Newton approximation of the Hessian of the cost function.
        NB: This is the Jacobian of the *free* (not tied or fixed) parameters.
    .grad: ndarray, shape (m,)
        Gradient of the cost function at the solution.
    .nfev: int
        Number of function evaluations done.
    .njev: int
        Number of Jacobian evaluations done.
    .status: int
        The reason for algorithm termination:
            * -1 : improper input parameters status
            *  0 : the maximum number of function evaluations is exceeded.
            *  2 : `ftol` termination condition is satisfied.
            *  3 : `xtol` termination condition is satisfied.
            *  4 : Both `ftol` and `xtol` termination conditions are satisfied.
    .message: str
        Verbal description of the termination reason.
    .success: bool
        True if one of the convergence criteria is satisfied (`status` > 0).

    Notes
    -----

    An early SQP method specialized for linear constraints was described in
    `Fletcher (1972) <https://doi.org/10.1007/BF01584540>`_

    A general textbook description of the *uncostrained* LM algorithm is in:

    - Chapter 5.2 of `Fletcher R., 1987, Practical Methods of Optimization, 2nd ed., Wiley
      <http://doi.org/10.1002/9781118723203>`_
    - Chapter 10.3 of `Nocedal J. & Wright S.J., 2006, Numerical Optimization, 2nd ed., Springer
      <http://doi.org./10.1007/978-0-387-40065-5>`_

    The original papers introducing the *uncostrained* LM method are:

    - `Levenberg K., 1944, Quart. Appl. Math., 164, 2
      <https://doi.org/10.1090/qam/10666>`_
    - `Marquardt D.W., 1963, J. Soc. Indust. Appl. Math, 11, 431
      <https://doi.org/10.1137/0111030>`_

    The Jacobian scaling and convergence tests follow
    `More', J.J., Garbow, B.S. & Hillstrom, K.E. 1980, User Guide for MINPACK-1,
    Argonne National Laboratory Report ANL-80-74 <http://cds.cern.ch/record/126569>`_

    """
    def __init__(self, func, p0, A_eq=None, A_ineq=None, abs_step=None,
                 b_eq=None, b_ineq=None, bounds=(-np.inf, np.inf),
                 diff_step=1e-4, fixed=None, ftol=1e-4, linear_method='lsq_lin',
                 max_nfev=None, rcond=None, tied=None, verbose=0, x_scale='jac',
                 xtol=1e-4, args=(), kwargs={}):

        p0 = np.array(p0, dtype=float)  # Make copy to leave input unchanged
        bounds = np.asarray([np.resize(b, p0.size) for b in bounds])
        assert np.all(np.less(*bounds)), "Must be lower bound < upper bound"
        p0 = p0.clip(*bounds)   # Force initial guess within bounds

        self.bounds_only = (A_ineq is None) and (A_eq is None)
        if not self.bounds_only:
            assert linear_method in ['lsq_lin', 'cvxopt'], "`linear_method` must be 'lsq_lin' or 'cvxopt'"
            if linear_method == 'cvxopt':
                assert cvxopt_installed, "To use `linear_method`='cvxopt' the cvxopt package must be installed"
            if A_ineq is not None:
                A_ineq, b_ineq = np.asarray(A_ineq, dtype=float), np.asarray(b_ineq, dtype=float)
                assert A_ineq.shape == (b_ineq.size, p0.size), "A_ineq/b_ineq size mismatch"
                eps = 1e-10*linalg.norm(A_ineq, axis=1)
                assert np.all(A_ineq @ p0 <= b_ineq + eps), \
                    "Initial guess is unfeasible for inequality (after clipping to bounds)"
            if A_eq is not None:
                A_eq, b_eq = np.asarray(A_eq, dtype=float), np.asarray(b_eq, dtype=float)
                assert A_eq.shape == (b_eq.size, p0.size), "A_eq/b_eq size mismatch"
                assert np.allclose(A_eq @ p0, b_eq), "Initial guess is unfeasible for equality"

        fixed = np.full(p0.size, False) if fixed is None else np.asarray(fixed)
        if tied is None:
            tied = np.full(p0.size, '')
        else:
            assert np.all([f'p[{j}]' not in td for j, td in enumerate(tied)]), \
                "Parameters cannot be tied to themselves"
            tied = np.asarray([a.strip() for a in tied])
        assert len(p0) == len(fixed) == len(tied), \
            "`x0`, `fixed` and `tied` must have the same size"

        free = (np.asarray(fixed) == 0) & (tied == '')
        self.nfev = self.njev = 0
        self.diff_step = diff_step if np.size(diff_step) == 1 else np.asarray(diff_step)[free]
        self.abs_step = abs_step if np.size(abs_step) == 1 else np.asarray(abs_step)[free]
        self.max_nfev = 100*free.sum() if max_nfev is None else max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.verbose = verbose
        self.x_scale = x_scale
        self.rcond = rcond
        self.linear_method = linear_method

        # Only include free variables in optimization
        if A_ineq is not None:
            A_ineq =  A_ineq[:, free]
        if A_eq is not None:
            A_eq =  A_eq[:, free]

        jtied = np.flatnonzero(tied != '')
        def tie(pfree):
            p = p0.copy()
            p[free] = pfree
            for j in jtied:  # loop can be empty
                p[j] = eval(tied[j])
            return p

        def call(pfree):
            self.nfev += 1
            p = tie(pfree)
            resid = func(p, *args, **kwargs)
            assert np.all(np.isfinite(resid)), \
                "The fitting function returned infinite residuals"
            return resid

        p1, p1_err = self.optimize(call, p0[free], A_eq, b_eq, A_ineq, b_ineq, bounds[:, free])

        self.x = tie(p1)
        self.x_err = np.zeros_like(self.x)
        self.x_err[free] = p1_err

################################################################################

    def optimize(self, call, p1, A_eq, b_eq, A_ineq, b_ineq, bounds):
        """
        Optimization of free variables only

        """
        f1 = call(p1)
        assert f1.ndim == 1, "The fitting function must return a vector of residuals"
        J1 = self.fdjac(call, p1, f1, bounds)
        dd = linalg.norm(J1, axis=0)
        mx = np.max(dd)
        eps = np.finfo(float).eps
        dd[dd < eps*max(1, mx)] = 1  # As More'+80
        lam = 0.01*mx**2  # 0.01*max(diag(J1.T @ J1))

        if self.verbose == 2:
            print(f"\nStart lambda: {lam:#.4g}  chi2: {chi2(f1):#.4g}\nStart p_free:" + fprint(p1))

        while True:

            if self.x_scale == 'jac':
                dd = np.maximum(dd, linalg.norm(J1, axis=0))
            else:
                dd = np.ones_like(p1)/self.x_scale

            # Solve the constrained quadratic sub-problem
            dn = dd/np.max(dd)
            A = np.vstack([J1, np.diag(np.sqrt(lam)*dn)])
            b = np.append(-f1, np.zeros_like(p1))

            if self.bounds_only:
                h = lsq_box(A, b, bounds=bounds - p1, rcond=self.rcond).x
            else:
                b_ineq_p = None if A_ineq is None else b_ineq - A_ineq @ p1
                b_eq_p = None if A_eq is None else b_eq - A_eq @ p1
                if self.linear_method == 'lsq_lin':
                    h = lsq_lin(A, b, A_ineq, b_ineq_p, A_eq, b_eq_p, bounds=bounds - p1, rcond=self.rcond).x
                else:
                    h = lsq_lin_cvxopt(A, b, A_ineq, b_ineq_p, A_eq, b_eq_p, bounds=bounds - p1).x

            p2 = p1 + h
            f2 = call(p2)

            # Actual versus predicted chi2 reduction
            actred = 1 - chi2(f2)/chi2(f1)
            prered = 1 - chi2(f1 + J1 @ h)/chi2(f1)
            ratio = actred/prered

            status = self.check_conv(lam, f2, p2, h, dd, actred, prered)

            if status != -1:
                if actred > 0:
                    p1, f1 = p2, f2
                if self.verbose:
                    print(f"\n{self.message}\nFinal iter: {self.njev}  "
                          f"Func calls: {self.nfev}  chi2: {chi2(f1):#.4g}  "
                          f"Status: {status}\nFinal p:" + fprint(p1) + "\n")
                break

            # Algorithm (5.2.7) of Fletcher (1987)
            # Algorithm 4.1 in Nocedal & Wright (2006)
            if ratio < 0.25:
                lam *= 4
            elif ratio > 0.75:
                lam /= 2

            if ratio > 0.01:  # Successful step: move on
                J1 = self.fdjac(call, p2, f2, bounds)
                p1, f1 = p2, f2

        self.cost = 0.5*chi2(f1)  # as in least_squares()
        self.fun = f1
        self.jac = J1
        self.grad = J1.T @ f1
        self.status = status
        self.success = status > 0
        self.cov, p1_err = cov_err(J1, rcond=self.rcond)

        return p1, p1_err

################################################################################

    def fdjac(self, call, pars, f, bounds):

        self.njev += 1

        if self.abs_step is None:
            h = self.diff_step*np.maximum(1.0, np.abs(pars))  # as in least_squares()
        else:
            h = self.abs_step*np.ones_like(pars)

        x = pars + h
        hits = (x < bounds[0]) | (x > bounds[1])

        # Respect bounds (but not inequalities) in finite differences
        if np.any(hits):
            dist = np.abs(bounds - pars)
            fits = np.abs(h) <= np.maximum(*dist)
            h[hits & fits] *= -1
            forward = (dist[1] >= dist[0]) & ~fits
            backward = (dist[1] < dist[0]) & ~fits
            h[forward] = dist[1, forward]
            h[backward] = -dist[0, backward]

        # Compute finite-differences derivatives
        jac = np.zeros([f.size, pars.size])
        for j, hj in enumerate(h):
            pars1 = pars.copy()
            pars1[j] += hj
            f1 = call(pars1)
            jac[:, j] = (f1 - f)/hj

        return jac

################################################################################

    def check_conv(self, lam, f, p, h, dn, actred, prered):

        status = -1
        if self.nfev > self.max_nfev:
            self.message = "Terminating on function evaluations count"
            status = 0

        if prered < self.ftol and abs(actred) < self.ftol and actred <= 2*prered:  # (More'+80)
            self.message = "Terminating on small function variation (ftol)"
            status = 2

        if linalg.norm(dn*h) < self.xtol*(self.xtol + linalg.norm(dn*p)):
            if status == 2:
                self.message = "Both ftol and xtol convergence test are satisfied"
                status = 4
            else:
                self.message = "Terminating on small step (xtol)"
                status = 3

        if self.verbose == 2:
            print(f"\niter: {self.njev}  lambda: {lam:#.4g}  chi2: {chi2(f):#.4g}"
                  f"  ratio: {actred/prered:#.4g}\np_free:" + fprint(p) + "\nh:" + fprint(h))

        return status

################################################################################
