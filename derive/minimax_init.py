#!/usr/bin/env python3
"""Curve-space minimax init of the single-term GPS: (a*, b*) = (0.221, 0).

This is the initialization of every single-term training arm in the paper
(`pdfv4` with `activation_kwargs={'a': 0.221, 'b': 0.0}` in run_bench2.py) and
the value stated in the paper's Optimization subsection.

It is a Chebyshev (min-max) centre: the single-term curve that minimises its
worst squared error to the three fitted curves of Table `tab:approx`,

    (a*, b*) = argmin_{a,b}  max_{j in {GELU, Swish, Mish}}  D_j(a,b),
    D_j(a,b) = mean_x ( f(x | a,b) - j(x) )^2 .

Two notes on the metric, both matching the approximation fit
(`fitting/approximation.ipynb`) exactly:

  * the error is the plain MEAN SQUARED ERROR on the grid
    x = linspace(-8, 8, 20000) - the same metric and grid the approximation
    uses (the paper writes an integral over the whole line; the computation
    used this finite-grid squared error);
  * j runs over the three *fitted* curves f_j = f(x | a_j, b_j) with
    (a_j, b_j) the GELU / Swish / Mish fits of `tab:approx` - the curves the
    paper calls GELU, Swish and Mish in this construction.  Against the raw
    functions instead, the minimax drifts to (0.220, 0.015); the paper's
    b* = 0.000 is the fits version.

The minimax equalises GELU and Swish and leaves Mish marginally closer.  The
answer is grid-insensitive: x in [-6, 6] or [-8, 8], 2 000 or 20 000 points all
give (0.2206, 0.0000) to four decimals.

    python minimax_init.py          # needs numpy + scipy
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

LN2 = np.log(2.0)
# Table tab:approx: (a, b) of the GPS fit to each static activation
FITS = {
    'GELU':  (0.35028508, 0.0),
    'Swish': (0.14575758, 0.0),
    'Mish':  (0.21961491, 0.16713308),
}


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def gps(x, a, b):
    return softplus(x) - LN2 * np.exp(np.clip(-a * x**2 - b * x, -80.0, 80.0))


def minimax(targets, x):
    """argmin_{a,b} max_j mean((gps - target_j)^2) on the grid x."""
    def mses(z):
        f = gps(x, z[0], z[1])
        return [float(np.mean((f - t)**2)) for t in targets]

    def obj(z):
        return max(mses(z))

    best = differential_evolution(obj, [(0.05, 0.5), (-0.5, 0.5)], seed=0,
                                  tol=1e-12, maxiter=400, polish=True)
    for a0 in (0.15, 0.22, 0.30):
        for b0 in (-0.10, 0.0, 0.10):
            r = minimize(obj, [a0, b0], method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-16, 'maxiter': 9000})
            if r.fun < best.fun:
                best = r
    return best.x, mses(best.x)


def main():
    x = np.linspace(-8.0, 8.0, 20000)          # the approximation's grid
    names = list(FITS)

    # (1) minimax against the three FITTED curves - the paper's construction
    ab, ms = minimax([gps(x, *FITS[n]) for n in names], x)
    print('minimax vs the tab:approx fits (the paper):')
    print('  (a*, b*) = (%.4f, %.4f)' % tuple(ab))
    print('  per-target MSE  ' + '   '.join(
        '%s %.3e' % (n, m) for n, m in zip(names, ms)))
    # the report's quoted distances (2.7501 / 2.7501 / 2.6871) are ||.||_2 on
    # a 2 000-point grid over [-6, 6]: sqrt(2000 * MSE) there
    x6 = np.linspace(-6.0, 6.0, 2000)
    f = gps(x6, *ab)
    d6 = [float(np.linalg.norm(f - gps(x6, *FITS[n]))) for n in names]
    print('  ||.||_2 on [-6,6], 2 000 pts: ' + '   '.join(
        '%s %.4f' % (n, d) for n, d in zip(names, d6))
        + '   (report: 2.7501 / 2.7501 / 2.6871)')
    print('  paper rounds this to (0.221, 0.000)\n')

    # (2) cross-check against the raw static functions
    def gelu(x):
        from scipy.special import erf
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    def swish(x):
        return x / (1.0 + np.exp(-x))

    def mish(x):
        return x * np.tanh(softplus(x))

    ab2, ms2 = minimax([gelu(x), swish(x), mish(x)], x)
    print('cross-check, raw GELU/Swish/Mish as targets:')
    print('  (a*, b*) = (%.4f, %.4f)   [b drifts; not the paper value]' % tuple(ab2))
    print('  per-target MSE  ' + '   '.join(
        '%s %.3e' % (n, m) for n, m in zip(names, ms2)))


if __name__ == '__main__':
    main()
