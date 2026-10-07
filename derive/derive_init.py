"""Derivation of the mixture initialisation w* = (0.361, 0.307, 0.332).

The three-term blend family  f_w(x) = softplus(x) - ln2 * sum_i w_i exp(-a_i x^2 - b_i x),
w in the 2-simplex, is a 3-parameter family.  Its single-term (2-parameter) regression

    Phi(w) = argmin_{a,b} || g_{a,b} - f_w ||^2 ,      g_{a,b}(x) = softplus(x) - ln2 exp(-a x^2 - b x)

maps the simplex into the (a,b) plane.  Phi is not affine: the image is a curved
patch, not a triangle, and it is not uniformly filled (density varies ~20x).

The initialisation is the preimage of the *centre of mass* of that patch:

    c  = E_{w ~ Uniform(simplex)} [ Phi(w) ]          (centre of mass of the folded space)
    w* = Phi^{-1}(c)                                  (the blend whose fold is that centre)

so it is neutral in the reparameterised 2-parameter solution space rather than in
weight space.  Run:  python derive_init.py
"""
import numpy as np
from scipy.optimize import least_squares

LN2 = np.log(2.0)
FITS = np.array([(0.35028508, 0.0),          # GELU fit
                 (0.14575758, 0.0),          # Swish fit
                 (0.21961491, 0.16713308)])  # Mish fit
XS = np.linspace(-8.0, 8.0, 4001)
SP = np.maximum(XS, 0.0) + np.log1p(np.exp(-np.abs(XS)))
B = np.array([np.exp(np.clip(-a * XS**2 - b * XS, -80, 80)) for a, b in FITS])


def blend(w):
    return SP - LN2 * (w @ B)


def single(a, b):
    return SP - LN2 * np.exp(np.clip(-a * XS**2 - b * XS, -80, 80))


def fold(w, x0=(0.22, 0.06)):
    """Phi(w): best single-term GPS for the blend curve w (2-parameter regression)."""
    r = least_squares(lambda p: single(*p) - blend(w), x0, method='lm', xtol=1e-14, ftol=1e-14)
    return r.x


def simplex_grid(step=0.01):
    W = [(w1, w2, 1.0 - w1 - w2)
         for w1 in np.arange(0.0, 1.0 + 1e-9, step)
         for w2 in np.arange(0.0, 1.0 - w1 + 1e-9, step)]
    return np.array(W)


def centre_of_mass(W):
    P = np.array([fold(w) for w in W])
    return P, P.mean(axis=0)


def preimage(target, W):
    """Phi^{-1}(target) by least squares over the two free simplex coordinates."""
    def resid(z):
        w = np.array([z[0], z[1], 1.0 - z[0] - z[1]])
        if np.any(w < 0):
            return np.array([9.0, 9.0])
        return (fold(w) - target) * 1e4
    best = None
    for w1 in (0.2, 0.35, 0.5):
        for w2 in (0.2, 0.35, 0.5):
            r = least_squares(resid, [w1, w2], method='lm')
            if best is None or np.linalg.norm(r.fun) < np.linalg.norm(best.fun):
                best = r
    w = np.array([best.x[0], best.x[1], 1.0 - best.x[0] - best.x[1]])
    return w


def main():
    W = simplex_grid(0.02)
    P, c = centre_of_mass(W)
    print('folded simplex: %d samples, a in [%.4f, %.4f], b in [%.4f, %.4f]'
          % (len(W), P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()))
    print('centre of mass  c = (%.5f, %.5f)' % (c[0], c[1]))
    print('  b_Mish/3 = %.5f   (b is linear in the Mish weight: slope %.4f, R^2 %.4f)'
          % (FITS[2, 1] / 3, np.polyfit(W[:, 2], P[:, 1], 1)[0], np.corrcoef(W[:, 2], P[:, 1])[0, 1]**2))
    w = preimage(c, W)
    print('preimage  Phi^-1(c) = (%.4f, %.4f, %.4f)' % tuple(w))
    shipped = np.array([0.361, 0.307, 0.332])
    print('shipped   w*        = (0.3610, 0.3070, 0.3320)   |dw| = %.4f, |fold(w*) - c| = %.5f'
          % (np.linalg.norm(w - shipped), np.linalg.norm(fold(shipped) - c)))
    print('uniform    (1/3,1/3,1/3) folds to (%.4f, %.4f), |fold - c| = %.5f'
          % (*fold(np.ones(3) / 3), np.linalg.norm(fold(np.ones(3) / 3) - c)))
    print('single-term min-max centre: (0.2206, 0.0000)')

    # why the answer is close to uniform: the fold map is nearly affine, and the
    # linear part sends the simplex centroid onto the centre of mass
    A = np.column_stack([np.ones(len(W)), W[:, 0], W[:, 1]])
    ca = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
    cb = np.linalg.lstsq(A, P[:, 1], rcond=None)[0]
    print('\nlinearised fold: a = %.4f + %.4f wG + %.4f wS   (R^2 %.4f)'
          % (*ca, 1 - ((P[:, 0] - A @ ca)**2).sum() / ((P[:, 0] - P[:, 0].mean())**2).sum()))
    print('                 b = %.4f + %.4f wG + %.4f wS   (R^2 %.4f)'
          % (*cb, 1 - ((P[:, 1] - A @ cb)**2).sum() / ((P[:, 1] - P[:, 1].mean())**2).sum()))
    M = np.array([[ca[1], ca[2]], [cb[1], cb[2]]])
    wlin = np.linalg.solve(M, np.array([c[0] - ca[0], c[1] - cb[0]]))
    print('linearised preimage of c = (%.4f, %.4f, %.4f)  <-- i.e. uniform weights'
          % (wlin[0], wlin[1], 1 - wlin[0] - wlin[1]))
    print('curvature shift exact - linear = (%+.4f, %+.4f, %+.4f)'
          % (w[0] - wlin[0], w[1] - wlin[1], w[2] - (1 - wlin[0] - wlin[1])))

    h = 1e-4
    d1 = (fold(np.array([w[0] + h, w[1], w[2] - h])) - fold(w)) / h
    d2 = (fold(np.array([w[0], w[1] + h, w[2] - h])) - fold(w)) / h
    sv = np.linalg.svd(np.column_stack([d1, d2]), compute_uv=False)
    print('|dPhi| at the preimage: singular values %.4f, %.4f -> 0.01 in w moves the fold '
          'by %.4f-%.4f in (a,b)' % (sv[0], sv[1], sv[1] * 0.01, sv[0] * 0.01))


if __name__ == '__main__':
    main()
