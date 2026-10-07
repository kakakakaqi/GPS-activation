"""Trainable-activation baselines for the GPS comparison.

Registers five trainable activations into gps_lab's registry, so they run through
exactly the same Trainer, optimizer groups, sharing machinery and recipe as GPS:

    beta_swish   x * sigmoid(beta*x)          beta trained, init 1    (== Swish at init)
    beta_gelu    x * Phi(beta*x)              beta trained, init 1    (== GELU  at init)
    eswish       beta * x * sigmoid(x)        beta trained, init 1    (== Swish at init)
    prelu_layer  max(0,x) + a * min(0,x)      a trained,    init 0.25 (1 parameter per layer)
    pau          P(x)/Q(x), degrees (3,2)     least-squares init to Swish (7 parameters)

Sources: beta-Swish is the family searched in Ramachandran et al. 2017
(arXiv:1710.05941); E-Swish is Alcaide 2018 (arXiv:1801.07145); PReLU is
He et al. 2015 (arXiv:1502.01852); PAU is Molina et al., ICLR 2020.

Design notes
------------
* Every class keeps its trainable values in a single `.params` tensor.  That is
  the convention gps_lab relies on: `share_activation_params` shares that tensor
  across sites for `share='global'`, and `split_param_groups` routes it to the
  activation learning-rate group.
* `.warmup()` exists because `share_activation_params` skips modules that do not
  have it.
* `prelu_layer` is deliberately the per-layer form (one parameter for the whole
  model when shared).  Canonical PReLU uses one parameter per channel, which is
  thousands of parameters on ResNet-18; see BASELINES.md for the fairness note.
* No custom kernels: these are plain autograd ops, so they need no Triton and are
  slower than GPS's kernel, which only makes the comparison conservative for GPS.
"""
import math

import numpy as np
import torch
import torch.nn as nn

SQRT2 = math.sqrt(2.0)


class _TrainableBase(nn.Module):
    """Shared plumbing: a flat `.params` tensor and the warmup() hook."""

    def __init__(self, values):
        super().__init__()
        self.params = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    @torch.no_grad()
    def warmup(self, n=256):
        """Materialise the parameter on its device (no kernels to autotune)."""
        if not self.params.is_cuda:
            return
        self.forward(torch.randn(n, device=self.params.device))


class BetaSwish(_TrainableBase):
    """x * sigmoid(beta * x); beta=1 is exactly Swish."""

    def __init__(self, beta=1.0, init=None):
        super().__init__([beta])

    def forward(self, x):
        return x * torch.sigmoid(self.params[0] * x)

    def extra_repr(self):
        return f'beta={self.params[0].item():.4f}'


class BetaGELU(_TrainableBase):
    """x * Phi(beta * x); beta=1 is exactly GELU."""

    def __init__(self, beta=1.0, init=None):
        super().__init__([beta])

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(self.params[0] * x / SQRT2))

    def extra_repr(self):
        return f'beta={self.params[0].item():.4f}'


class ESwish(_TrainableBase):
    """beta * x * sigmoid(x); beta=1 is exactly Swish."""

    def __init__(self, beta=1.0, init=None):
        super().__init__([beta])

    def forward(self, x):
        return self.params[0] * x * torch.sigmoid(x)

    def extra_repr(self):
        return f'beta={self.params[0].item():.4f}'


class PReLULayer(_TrainableBase):
    """max(0, x) + a * min(0, x), one parameter per layer (shared or per-site)."""

    def __init__(self, a=0.25, init=None):
        super().__init__([a])

    def forward(self, x):
        a = self.params[0]
        return torch.maximum(x, torch.zeros_like(x)) + a * torch.minimum(x, torch.zeros_like(x))

    def extra_repr(self):
        return f'a={self.params[0].item():.4f}'


def _fit_rational(target, m=3, n=2, points=2000, steps=3000, lr=0.02):
    """Least-squares P/Q of degrees (m, n) to `target` on [-8, 8], in numpy.

    P(x) = sum_{i<=m} a_i x^i,  Q(x) = |sum_{j<=n} b_j x^j| + 1e-3   (Q > 0).
    P is normalised so that a_m = b_n (right asymptote y -> x).
    """
    x = np.linspace(-8.0, 8.0, points)
    y = target(x)
    # start from a mildly curved rational that is already Swish-shaped
    a = np.array([0.0, 1.0, 0.0, 0.05][:m + 1], dtype=float)
    b = np.array([0.2, 0.0, 0.05][:n + 1], dtype=float)
    a[-1] = b[-1]

    def forward(a, b):
        p = sum(a[i] * x ** i for i in range(m + 1))
        q = np.abs(sum(b[j] * x ** j for j in range(n + 1))) + 1e-3
        return p / q

    for _ in range(steps):
        p = sum(a[i] * x ** i for i in range(m + 1))
        qs = sum(b[j] * x ** j for j in range(n + 1))
        q = np.abs(qs) + 1e-3
        r = p / q - y
        gp = 2.0 * r / q
        gq = -2.0 * r * p / (q * q) * np.sign(qs)
        a -= lr * np.array([np.mean(gp * x ** i) for i in range(m + 1)])
        b -= lr * np.array([np.mean(gq * x ** j) for j in range(n + 1)])
        a[-1] = b[-1]                      # keep the linear right asymptote
    return a, b


class PAU(_TrainableBase):
    """Rational activation P(x)/Q(x) with |Q| kept positive (Molina et al., ICLR 2020)."""

    M, N = 3, 2

    def __init__(self, init=None, values=None):
        if values is None:
            a, b = _fit_rational(_swish_np, self.M, self.N)
            values = np.concatenate([a, b])
        super().__init__(values)

    def forward(self, x):
        a = self.params[:self.M + 1]
        b = self.params[self.M + 1:]
        p = sum(a[i] * x ** i for i in range(self.M + 1))
        q = sum(b[j] * x ** j for j in range(self.N + 1)).abs() + 1e-3
        return p / q

    def extra_repr(self):
        return 'deg=(%d,%d), %d parameters' % (self.M, self.N, self.params.numel())


def _swish_np(x):
    return x / (1.0 + np.exp(-x))


def register():
    """Add the baselines to gps_lab's registry and to its trainable class list."""
    import gps_lab
    from gps_lab import ActivationRegistry, _trainable_factory
    ActivationRegistry.FACTORIES.update({
        'beta_swish':  _trainable_factory(BetaSwish),
        'beta_gelu':   _trainable_factory(BetaGELU),
        'eswish':      _trainable_factory(ESwish),
        'prelu_layer': _trainable_factory(PReLULayer),
        'pau':         _trainable_factory(PAU),
    })
    # gps_lab defines this as a bare class (no trailing comma), so accept both
    known = gps_lab.TRAINABLE_ACT_CLASSES
    known = known if isinstance(known, tuple) else (known,)
    gps_lab.TRAINABLE_ACT_CLASSES = known + tuple(
        c for c in (BetaSwish, BetaGELU, ESwish, PReLULayer, PAU) if c not in known)


register()

ARMS = ['beta_swish', 'beta_gelu', 'eswish', 'prelu_layer', 'pau']


if __name__ == '__main__':
    # self-test: shapes, gradients, parameter counts, and init == static
    from gps_lab import ActivationRegistry, ArchitectureRegistry
    import trainable_acts  # noqa: F401  (registers)
    x = torch.linspace(-8, 8, 9, requires_grad=True)
    ref = {'beta_swish': nn.SiLU()(x.detach()), 'beta_gelu': nn.GELU()(x.detach()),
           'eswish': nn.SiLU()(x.detach())}
    for name in ARMS:
        act = ActivationRegistry.get(name)
        y = act(x)
        y.sum().backward(retain_graph=True)
        n = sum(p.numel() for p in act.parameters())
        extra = ''
        if name in ref:
            extra = f'  max|f - static| = {(y.detach() - ref[name]).abs().max():.2e}'
        print(f'{name:12s} params={n}  out shape={tuple(y.shape)}{extra}')
    m = ArchitectureRegistry.get('resnet18', num_classes=10, activation='beta_swish',
                                 in_channels=3)
    from gps_lab import share_activation_params, split_param_groups
    share_activation_params(m, 'global')
    act_p, other_p = split_param_groups(m)
    print(f'resnet18/beta_swish: activation params {sum(p.numel() for p in act_p)}, '
          f'others {sum(p.numel() for p in other_p):,}')
