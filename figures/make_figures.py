"""Regenerate the paper's activation-fit figure (Fig. 1, `activation_fits.pdf`).

Panels (a)-(c): target activation (dashed) vs its GPS fit (solid) for GELU, Swish
and Mish.  Panels (d)-(f): the corresponding residuals.  Fits are the (a, b) pairs
of Table 2 in the paper, so the figure and the table cannot drift apart.

    python paper/make_figures.py [--check]

`--check` only prints the fit MSEs so they can be compared against the table.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LN2 = math.log(2.0)
# Table 2 of the paper: (a, b, reported MSE)
FITS = {
    'GELU':  (0.3503, 0.0000, 1.36e-4),
    'Swish': (0.1458, 0.0000, 2.52e-4),
    'Mish':  (0.2196, 0.1671, 5.42e-4),
}


def softplus(x):
    return np.logaddexp(0.0, x)


def gps(x, a, b):
    return softplus(x) - LN2 * np.exp(-a * x * x - b * x)


def target(name, x):
    if name == 'GELU':                      # x * Phi(x)
        return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    if name == 'Swish':                     # x * sigma(x)
        return x / (1.0 + np.exp(-x))
    if name == 'Mish':                      # x * tanh(softplus(x))
        return x * np.tanh(softplus(x))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='print MSEs only')
    ap.add_argument('--points', type=int, default=20_000,
                    help='sample count on [-8, 8] (paper uses 20000)')
    # NOTE: activation_fits.pdf in this directory is the authors' canonical
    # figure. This script writes to preview/ so it can never overwrite it; use
    # --check to verify the table's MSEs without producing a figure at all.
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'preview', 'activation_fits_preview'))
    a = ap.parse_args()

    x = np.linspace(-8.0, 8.0, a.points)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.2))
    for col, (name, (pa, pb, reported)) in enumerate(FITS.items()):
        y = target(name, x)
        yhat = gps(x, pa, pb)
        mse = float(np.mean((y - yhat) ** 2))
        mae = float(np.mean(np.abs(y - yhat)))
        print(f'{name:6s} a={pa:.4f} b={pb:.4f}  MSE={mse:.3e} (table {reported:.2e})  '
              f'MAE={mae:.3e}')
        if not a.check:
            ax = axes[0, col]
            ax.plot(x, y, '--', color='0.35', lw=1.6, label=f'{name} (target)')
            ax.plot(x, yhat, '-', color='C0', lw=1.4, label='GPS fit')
            ax.set_title(f'({chr(97 + col)}) {name}', fontsize=10)
            ax.set_xlabel('$x$', fontsize=9)
            ax.set_ylabel('$f(x)$', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(alpha=0.25, lw=0.5)

            ax = axes[1, col]
            ax.axhline(0.0, color='0.6', lw=0.8)
            ax.plot(x, y - yhat, '-', color='C3', lw=1.1)
            ax.set_title(f'({chr(100 + col)}) {name} residual', fontsize=10)
            ax.set_xlabel('$x$', fontsize=9)
            ax.set_ylabel('error', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25, lw=0.5)

    if not a.check:
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        for ext in ('pdf', 'png'):
            path = f'{a.out}.{ext}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print('wrote', path)


if __name__ == '__main__':
    main()
