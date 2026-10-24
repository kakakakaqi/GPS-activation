"""GPS — Gaussian Probability Softplus, both optimization forms in one module.

The activation of the paper,

    f(x) = softplus(x) - ln2 * exp(-a x^2 - b x)                    (2-parameter)

and its convolutional training scheme,

    f(x) = softplus(x) - ln2 * sum_i w_i exp(-a_i x^2 - b_i x)      (mixture)
           w = softmax(logits),  sum_i w_i = 1

where the three ``(a_i, b_i)`` are the fits of GELU, Swish and Mish by the
2-parameter form and are held fixed; only the three logits are trained.  Both
forms keep ``softplus(x)`` as the backbone, so for any parameters

    f(x) -> 0  as x -> -inf,    f(x) -> x  as x -> +inf,    f(0) = 0,

and the wave term is a Gaussian (single form) or a convex combination of the
three fitted Gaussians (mixture form).

The module also carries the paper's only hyperparameter rule, the activation
learning-rate calibration: one epoch at unit multiplier measures how far the
activation parameters travel, and the multiplier that spends a fixed budget
``D`` over the remaining epochs follows in closed form.

Typical use::

    import gps

    model = ...                                      # any network
    gps.swap_activations(model)                      # GELU/SiLU/Mish -> GPS
    gps.share_parameters(gps.find(model), 'global')  # one (a, b) for the net

    weight_lr = 1e-3
    opt = torch.optim.AdamW([
        {'params': weights(model), 'lr': weight_lr},
        {'params': gps.find(model)[0].params,
         'lr': weight_lr * gps.PROBE_MULTIPLIER},    # 1.0: epoch 1 is the probe
    ])

    for epoch in range(200):
        if epoch == 1:                               # one epoch has been trained
            gps.calibrate(model, opt.param_groups[1], epochs=200)
        train_one_epoch()

The first epoch runs at ``PROBE_MULTIPLIER`` (1.0) so that its displacement
measures the unit-rate the rule divides by; ``calibrate`` then replaces the
activation lr with ``weight_lr * lambda``.  A run that is not probed keeps the
fallback ``MULTIPLIER_DEFAULT`` (0.5).  For the convolutional scheme use
``form='mixture'``, and ``fold(model)`` to return to the two-parameter form.

Pure PyTorch: the reference implementation below is the readable version of the
fused Triton kernel, which only matters for throughput.  ``gps_lab`` (the
experiment runtime) exposes the same calibration as ``probe_multiplier``.
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LN2', 'MAX_EXP', 'BASIS', 'BASIS_ORDER', 'INIT_AB', 'INIT_W',
    'GPS', 'gps_2param', 'gps_mixture', 'fold_weights',
    'PROBE_MULTIPLIER', 'MULTIPLIER_DEFAULT', 'displacement', 'probe_multiplier', 'calibrate',
    'calibrate_lr_multiplier', 'multiplier', 'multipliers', 'set_multiplier',
    'find', 'share_parameters', 'swap_activations', 'fold', 'warmup',
]

LN2 = math.log(2.0)

#: The exponent is clamped here before ``exp`` (the forward pass of the Triton
#: kernel does the same).  ``a < 0`` makes the exponent grow without bound; the
#: clamp keeps the forward finite and pushes the failure into the gradient,
#: which is where the paper's trust-region fallback intervenes.
MAX_EXP = 80.0

#: Table 2 of the paper: each static activation fitted by the 2-parameter form.
BASIS = {
    'gelu':  (0.35028508, 0.0),
    'swish': (0.14575758, 0.0),
    'mish':  (0.21961491, 0.16713308),
}
BASIS_ORDER = ('gelu', 'swish', 'mish')

#: Chebyshev (min-max) centre of the three fits above in the (a, b) plane; the
#: initialization of every 2-parameter run.
INIT_AB = (0.221, 0.0)

#: The weight vector whose least-squares image is that centre; the
#: initialization of every mixture run.
INIT_W = (0.361, 0.307, 0.332)

_INIT_W_TENSOR = torch.tensor(INIT_W, dtype=torch.float32)

#: Activation learning rate as a multiple of the weight learning rate, used for
#: the first epoch.  That epoch *is* the probe: it must run at unit multiplier
#: (activation lr == weight lr) for the measured displacement to be the rate the
#: rule divides by.
PROBE_MULTIPLIER = 1.0

#: The activation learning rate multiplier when no calibrated value is used --
#: the fallback for a run that is not probed, and what ``multiplier()`` returns
#: before :func:`calibrate` has run.  0.5 is also the value the mixture scheme
#: always uses.
MULTIPLIER_DEFAULT = 0.5


# --------------------------------------------------------------------------- #
# functional forms
# --------------------------------------------------------------------------- #
def gps_2param(x: torch.Tensor, a, b) -> torch.Tensor:
    """``softplus(x) - ln2 * exp(-a x^2 - b x)``.

    ``a`` and ``b`` may be python floats or tensors broadcastable to ``x``.
    """
    exponent = torch.clamp(-(a * x * x + b * x), max=MAX_EXP)
    return F.softplus(x) - LN2 * torch.exp(exponent)


def gps_mixture(x: torch.Tensor, w: torch.Tensor,
                basis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``softplus(x) - ln2 * sum_i w_i exp(-a_i x^2 - b_i x)``.

    Elementwise, like the single form, so ``x`` may have any shape; it is
    flattened for the three-term sum and reshaped afterwards.
    """
    if basis is None:
        basis = _BASIS_TENSOR.to(device=x.device, dtype=x.dtype)
    shape = x.shape
    xf = x.reshape(-1)
    w = w.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
    a = basis[:, 0].to(device=x.device, dtype=x.dtype).reshape(-1, 1)
    b = basis[:, 1].to(device=x.device, dtype=x.dtype).reshape(-1, 1)
    exponent = torch.clamp(-(a * xf * xf + b * xf), max=MAX_EXP)
    out = F.softplus(xf) - LN2 * (w * torch.exp(exponent)).sum(0)
    return out.reshape(shape)


_BASIS_TENSOR = torch.tensor(
    [BASIS[name] for name in BASIS_ORDER], dtype=torch.float32)

#: The fused Triton kernels, resolved on first use.  The reference functions
#: above are numerically equivalent (they agree to float32 round-off); the
#: kernels are what make GPS as fast as GELU, and they are the path the paper's
#: runs used.  Imported lazily so that this module needs nothing but torch when
#: they are absent or the tensors are on the CPU.
_KERNELS: Optional[tuple] = None


def _triton_kernels() -> Optional[tuple]:
    """``(single_form, mixture_form)`` Triton functions, or ``None``."""
    global _KERNELS
    if _KERNELS is None:
        try:
            try:
                from .gps_choice import _MixWave
                from .gps_lab import GaussianDecaySoftplusTriton
            except ImportError:
                from gps_choice import _MixWave
                from gps_lab import GaussianDecaySoftplusTriton
            _KERNELS = (GaussianDecaySoftplusTriton, _MixWave)
        except Exception:
            _KERNELS = ()
    return _KERNELS or None


# --------------------------------------------------------------------------- #
# the module
# --------------------------------------------------------------------------- #
class GPS(nn.Module):
    """The GPS activation.

    Parameters
    ----------
    form : ``'single'`` or ``'mixture'``
        Which optimization form is being trained.
    init : ``'minimax'`` (default), a pair ``(a, b)``, a 3-vector of weights, or
        ``None`` for the paper's defaults -- (0.221, 0) for the 2-parameter form
        and ``w* = (0.361, 0.307, 0.332)`` for the mixture.
    basis : optional ``(3, 2)`` tensor of fixed ``(a_i, b_i)``, overriding Table 2.
    multiplier : float, optional
        Activation lr as a multiple of the weight lr, used as the fallback and
        as the result of calibration.  Defaults to :data:`MULTIPLIER_DEFAULT`
        (0.5) and is replaced in place by :func:`calibrate` after the first
        epoch.  The probe epoch itself runs at :data:`PROBE_MULTIPLIER` (1.0),
        which is what makes the measurement interpretable.
    backend : ``'auto'`` (default), ``'triton'`` or ``'torch'``
        ``'auto'`` uses the fused Triton kernel for CUDA float32 tensors and
        falls back to the PyTorch reference otherwise.  ``'triton'`` raises
        instead of falling back, ``'torch'`` never uses the kernel.  The two
        paths agree to float32 round-off; the kernel is why GPS costs ~0.4%
        more per epoch than GELU instead of several times more.

    Attributes
    ----------
    ``params``
        The single trainable tensor, exposed under the name the experiment
        runtime keys on: shape ``(2,)`` for ``'single'``, shape ``(3,)`` of
        logits for ``'mixture'``.  Route it to its own optimizer group and set
        that group's learning rate to ``weight_lr * multiplier``.
    ``multiplier``
        The number above.  Read it with :func:`multiplier` (network level) and
        update the optimizer group whenever it changes.
    ``init_params``
        Buffer holding the value ``params`` started from; the calibration
        measures displacement against it.
    """

    def __init__(self, form: str = 'single', init: Optional[object] = 'minimax',
                 basis: Optional[torch.Tensor] = None,
                 multiplier: Optional[float] = None,
                 backend: str = 'auto'):
        super().__init__()
        if form not in ('single', 'mixture'):
            raise ValueError(f"form must be 'single' or 'mixture', got {form!r}")
        self.form = form
        if backend not in ('auto', 'triton', 'torch'):
            raise ValueError(f"backend must be 'auto', 'triton' or 'torch', got {backend!r}")
        #: Where the arithmetic runs: the fused Triton kernel ('triton'), the
        #: plain-PyTorch reference ('torch'), or whichever is available ('auto').
        self.backend = backend
        #: Activation lr as a multiple of the weight lr, as a fallback and as
        #: the result of calibration.  Starts at :data:`MULTIPLIER_DEFAULT`
        #: (0.5) and is overwritten in place by :func:`calibrate`.  It is *not*
        #: the value the probe epoch runs at -- that is
        #: :data:`PROBE_MULTIPLIER` (1.0).
        self.multiplier = (MULTIPLIER_DEFAULT if multiplier is None
                           else float(multiplier))
        if basis is None:
            basis = _BASIS_TENSOR.clone()
        self.register_buffer('basis', basis.to(torch.float32))

        if form == 'single':
            if init is None or init == 'minimax':
                ab = INIT_AB
            elif isinstance(init, str):
                ab = BASIS[init]                      # start at a static's fit
            else:
                ab = tuple(float(v) for v in init)
                if len(ab) != 2:
                    raise ValueError(f'single form wants 2 parameters, got {len(ab)}')
            self.params = nn.Parameter(torch.tensor(ab, dtype=torch.float32))
        else:
            if init is None or init == 'minimax':
                w = INIT_W
            elif isinstance(init, str):
                w = [0.0, 0.0, 0.0]                   # a simplex vertex
                w[BASIS_ORDER.index(init)] = 1.0
            else:
                w = tuple(float(v) for v in init)
                if len(w) != 3:
                    raise ValueError(f'mixture form wants 3 weights, got {len(w)}')
                if any(v <= 0.0 for v in w):
                    raise ValueError('mixture weights must be positive')
                s = sum(w)
                w = [v / s for v in w]
            # train the logits, not the weights, so sum_i w_i == 1 holds for
            # every parameter value
            self.params = nn.Parameter(torch.log(torch.tensor(w, dtype=torch.float32)))

        #: The value ``params`` started from.  The learning-rate calibration
        #: compares the current parameters against this, so a probe needs to
        #: know only where the run began -- not the paper's constants.
        self.register_buffer('init_params', self.params.detach().clone())

    # -- evaluation ---------------------------------------------------------
    def weights(self) -> torch.Tensor:
        """Current mixture weights (``softmax`` of the logits)."""
        return torch.softmax(self.params, dim=0)

    def _kernels(self, x: torch.Tensor) -> Optional[tuple]:
        """The Triton functions to use for ``x``, or ``None`` for the reference."""
        if self.backend == 'torch':
            return None
        kernels = _triton_kernels()
        usable = (kernels is not None and x.is_cuda and self.params.is_cuda
                  and x.dtype == torch.float32 and self.params.dtype == torch.float32)
        if not usable:
            if self.backend == 'triton':
                raise RuntimeError(
                    'backend="triton" needs the Triton kernels importable and '
                    'CUDA float32 tensors for both x and the parameters')
            return None
        return kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernels = self._kernels(x)
        if kernels is not None:
            single, mixture = kernels
            if self.form == 'single':
                return single.apply(x, self.params)
            return mixture.apply(x, self.weights(), self.basis)
        if self.form == 'single':
            return gps_2param(x, self.params[0], self.params[1])
        return gps_mixture(x, self.weights(), self.basis)

    # -- runtime contract ---------------------------------------------------
    def warmup(self, n: int = 1024) -> None:
        """Compile the forward and backward paths on a throwaway tensor.

        The fused kernel compiles and autotunes on first use, in both
        directions; calling this after the model is on the device and before
        the first optimizer step keeps that cost out of the first real step.
        Any gradient produced here is discarded -- call it before training.
        """
        x = torch.randn(n, device=self.params.device, dtype=self.params.dtype)
        out = self.forward(x)
        if out.requires_grad:
            out.sum().backward()
        self.zero_grad(set_to_none=True)

    # -- the two forms are interchangeable ----------------------------------
    def fold(self, steps: int = 400, lr: float = 0.02, warn: bool = True
             ) -> Tuple[float, float, float]:
        """The 2-parameter GPS that best matches this mixture.

        Solves ``(a, b) = argmin MSE(f_mixture, f_2param)`` on ``x in [-8, 8]``
        -- the paper's map ``phi(w)``.  Returns ``(a, b, mse)``; for the single
        form this is the identity.  This only *reports* the fold.  To actually
        convert a trained model to the two-parameter form, use the model-level
        :func:`fold`.

        .. warning::
           Folding changes the curve at the activation, but **not** the batch
           normalisation statistics that were collected while the network was
           running the mixture.  Recalibrate batch norm (a forward pass over the
           training data in ``eval()`` mode) before measuring or deploying the
           folded model -- the paper does this, and skips it at your peril.
           That recalibration is not part of this method, hence the warning;
           silence it with ``warn=False`` when folding in a loop.
        """
        if self.form == 'single':
            a, b = self.params.detach().cpu().tolist()
            return float(a), float(b), 0.0
        if warn:
            warnings.warn(
                'fold() maps only the activation curve: the batch-norm statistics '
                'still belong to the mixture. Recalibrate batch norm (forward pass '
                'over the training data in eval() mode) before evaluating or '
                'deploying the folded model.',
                RuntimeWarning, stacklevel=2)
        return fold_weights(self.weights().detach().cpu(), self.basis.detach().cpu(),
                            steps=steps, lr=lr)

    def to_form(self, form: str, **kwargs) -> 'GPS':
        """A fresh module of the other form, starting at this one's fold.

        Converting a mixture to the single form warns: see :meth:`fold`.
        """
        if form == self.form:
            return self
        if form == 'single':
            a, b, _ = self.fold(**{k: v for k, v in kwargs.items()
                                   if k in ('steps', 'lr', 'warn')})
            return GPS('single', init=(a, b), basis=self.basis.clone())
        a, b = (self.params.detach().cpu().tolist() if self.form == 'single'
                else self.fold(warn=False)[:2])
        # invert phi at (a, b) by a short fit on the simplex
        w = _invert_fold(a, b, self.basis.detach().cpu(),
                         **{k: v for k, v in kwargs.items() if k in ('steps', 'lr')})
        return GPS('mixture', init=w, basis=self.basis.clone())

    def extra_repr(self) -> str:
        if self.form == 'single':
            a, b = self.params.detach().tolist()
            body = f"form='single', a={a:.4f}, b={b:.4f}"
        else:
            w = torch.softmax(self.params, 0).detach().tolist()
            body = "form='mixture', w=[" + ', '.join(f'{v:.3f}' for v in w) + ']'
        return f'{body}, multiplier={self.multiplier:.3f}'


# --------------------------------------------------------------------------- #
# folding: mixture weights -> the two-parameter form
# --------------------------------------------------------------------------- #
def fold_weights(w: Sequence[float], basis: Optional[torch.Tensor] = None,
                 steps: int = 400, lr: float = 0.02,
                 init: Tuple[float, float] = INIT_AB,
                 bounds: Sequence[Tuple[float, float]] = ((0.01, 1.5), (-1.5, 1.5)),
                 n: int = 2001, lo: float = -8.0, hi: float = 8.0
                 ) -> Tuple[float, float, float]:
    """Least-squares projection of a mixture onto the 2-parameter form.

    ``(a, b) = argmin MSE( gps_mixture(.|w), gps_2param(.|a,b) )`` on a uniform
    grid over ``[lo, hi]``, solved by projected Adam from ``init``.  ``a`` is
    constrained positive so the result is a valid GPS curve (the paper's
    degenerate branch ``a <= 0`` is excluded).  Returns ``(a, b, mse)``.
    """
    if basis is None:
        basis = _BASIS_TENSOR
    x = torch.linspace(lo, hi, n, dtype=torch.float64)
    w_t = torch.as_tensor(list(w), dtype=torch.float64)
    if w_t.numel() != 3:
        raise ValueError(f'expected 3 weights, got {w_t.numel()}')
    if float(w_t.sum()) <= 0:
        raise ValueError('weights must sum to a positive value')
    w_t = w_t / w_t.sum()
    target = gps_mixture(x, w_t, basis.to(torch.float64)).detach()

    a = torch.tensor(float(init[0]), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(init[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(gps_2param(x, a, b), target)
        loss.backward()
        opt.step()
        with torch.no_grad():                          # keep the curve valid
            a.clamp_(min=bounds[0][0], max=bounds[0][1])
            b.clamp_(min=bounds[1][0], max=bounds[1][1])
    with torch.no_grad():
        mse = F.mse_loss(gps_2param(x, a.detach(), b.detach()), target).item()
        a_v, b_v = float(a.detach()), float(b.detach())
    return a_v, b_v, float(mse)


def _invert_fold(a: float, b: float, basis: torch.Tensor,
                 steps: int = 400, lr: float = 0.05) -> Tuple[float, float, float]:
    """Weights whose fold lands near ``(a, b)``, i.e. a numerical ``phi^-1``."""
    x = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float64)
    target = gps_2param(x, a, b).detach()
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(gps_mixture(x, torch.softmax(logits, 0), basis.to(torch.float64)),
                          target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = torch.softmax(logits, 0).tolist()
    return tuple(w)


# --------------------------------------------------------------------------- #
# learning-rate calibration
# --------------------------------------------------------------------------- #
def displacement(init: Sequence[torch.Tensor],
                 after: Sequence[torch.Tensor]) -> float:
    """Mean per-site L2 distance travelled by the activation parameters.

    With one shared parameter tensor (``share='global'``) this is that tensor's
    distance.  With one tensor per activation site the mean over sites is used
    rather than the sum, which keeps the number in a single site's units and
    independent of network depth.
    """
    if len(init) != len(after) or not init:
        raise ValueError('init and after must be non-empty and of equal length')
    total = 0.0
    for p0, p1 in zip(init, after):
        p0 = torch.as_tensor(p0).detach().to(torch.float64).reshape(-1)
        p1 = torch.as_tensor(p1).detach().to(torch.float64).reshape(-1)
        if p0.shape != p1.shape:
            raise ValueError(f'parameter shape changed: {tuple(p0.shape)} -> {tuple(p1.shape)}')
        total += float(torch.linalg.vector_norm(p1 - p0))
    return total / len(init)


def probe_multiplier(init: Sequence[torch.Tensor],
                     after: Sequence[torch.Tensor],
                     epochs: int,
                     remaining: Optional[int] = None,
                     target: float = 0.7,
                     probe_at: float = 1.0,
                     clip: Tuple[float, float] = (0.01, 10.0)) -> float:
    """Activation learning-rate multiplier, as a multiple of the weight lr.

    The activation group descends about ``rate`` distance units per epoch, and
    ``rate`` is proportional to the multiplier, so the multiplier that spends a
    budget ``target`` over the remaining epochs is

        mult = target / (d1 * remaining),   d1 = displacement at unit multiplier

    ``init`` is the activation state before training and ``after`` the state
    after one epoch.  That epoch need not have run at unit multiplier: pass the
    multiplier it *did* run at as ``probe_at`` and the displacement is rescaled
    by it, ``d1 = displacement / probe_at``, which is exact to the accuracy of
    the linearity in the multiplier.  With the default ``probe_at=1.0`` the
    probe epoch ran at ``activation lr == weight lr``.

    ``remaining`` defaults to ``epochs - 1`` -- the epochs still to be trained
    when the multiplier is set after the first epoch.  Pass ``epochs`` when
    training is restarted from scratch with the result, so that every epoch of
    the new run is governed by it.  Zero displacement means the probe saw no
    signal, so the multiplier pins to the top of ``clip``.
    """
    if epochs < 2:
        raise ValueError(f'epochs must be >= 2, got {epochs}')
    if remaining is None:
        remaining = epochs - 1
    if remaining < 1:
        raise ValueError(f'remaining must be >= 1, got {remaining}')
    if probe_at <= 0.0:
        raise ValueError(f'probe_at must be positive, got {probe_at}')
    lo, hi = clip
    if not 0.0 < lo <= hi:
        raise ValueError(f'clip must satisfy 0 < lo <= hi, got {clip}')
    d1 = displacement(init, after) / probe_at
    if d1 <= 0.0:
        return float(hi)
    return float(min(hi, max(lo, target / (d1 * remaining))))


def _probe_state(modules: Sequence[GPS],
                 init: Optional[Sequence[torch.Tensor]] = None
                 ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """``(starting state, current state)`` of each distinct activation tensor.

    Under ``share='global'`` every site holds the *same* parameter object, so
    the tensors are deduplicated by identity and each one is measured once.
    """
    if init is not None:
        init = [torch.as_tensor(v) for v in init]
        if len(init) != len(modules):
            raise ValueError(f'init has {len(init)} entries for {len(modules)} modules')
    seen, starts, current = set(), [], []
    for i, m in enumerate(modules):
        if id(m.params) in seen:
            continue
        seen.add(id(m.params))
        starts.append(init[i] if init is not None else m.init_params)
        current.append(m.params.detach())
    return starts, current


def calibrate_lr_multiplier(model: nn.Module,
                            run_epoch: Optional[Callable[[], None]] = None,
                            *,
                            epochs: int,
                            remaining: Optional[int] = None,
                            target: float = 0.7,
                            clip: Tuple[float, float] = (0.01, 10.0),
                            modules: Optional[Iterable[GPS]] = None,
                            init: Optional[Sequence[torch.Tensor]] = None,
                            probe_at: Optional[float] = None) -> float:
    """Return the activation learning-rate multiplier from a one-epoch probe.

    Two modes share this signature:

    **Probe mode** (`run_epoch` given) -- train one epoch, then return the
    multiplier for the rest of training::

        mult = calibrate_lr_multiplier(model, run_epoch, epochs=200)
        # activation group: g['lr'] = weight_lr * mult

    The probe epoch is the first epoch of the run, at :data:`PROBE_MULTIPLIER`
    (1.0): the activation learning rate **equal to the weight learning rate**.
    That is the default; if your loop used another value, pass it as
    ``probe_at`` and the displacement is rescaled to the unit rate.

    The probe epoch is thrown away -- the paper restarts training with the
    calibrated multiplier -- which costs one epoch.  Continuing from the probe
    instead is equivalent to within the epoch's share of the budget, because
    the first epoch falls inside the learning-rate warmup.

    **Already-probed mode** (`run_epoch=None`, the default) -- assume the model
    has *already* trained exactly one epoch from its initialisation, and
    recompute the multiplier from where it now stands::

        mult = calibrate_lr_multiplier(model, epochs=200)   # no epoch is run

    The starting point is each module's ``init_params`` (recorded when it was
    built), so this works on a model trained elsewhere, on a checkpoint, or on
    the epoch that the probe mode would otherwise have to rerun.  Pass ``init``
    to override it, e.g. with the constants a checkpoint was built from.

    ``remaining`` is the number of epochs the multiplier governs; it defaults to
    ``epochs - 1``.  Pass ``remaining=epochs`` when training is restarted with
    the multiplier, so that every epoch of the new run is governed by it.

    For the train-loop form that also *stores* the result on the modules, use
    :func:`calibrate`.
    """
    modules = list(modules if modules is not None else find(model))
    if not modules:
        raise ValueError('no GPS modules found in the model')

    if run_epoch is None:
        starts, current = _probe_state(modules, init)
    else:
        starts, _ = _probe_state(modules, init)
        starts = [torch.as_tensor(s).detach().clone() for s in starts]
        run_epoch()
        _, current = _probe_state(modules)
        current = [c.detach().clone() for c in current]

    if probe_at is None:
        # both modes assume the standard loop: the first epoch ran at the probe
        # multiplier (activation lr == weight lr), not at the fallback value
        probe_at = PROBE_MULTIPLIER

    return probe_multiplier(starts, current, epochs, remaining=remaining,
                            target=target, clip=clip, probe_at=probe_at)


def _common_multiplier(modules: Sequence[GPS]) -> Optional[float]:
    """The one multiplier shared by ``modules``, or ``None`` if they differ."""
    values = {m.multiplier for m in modules}
    return values.pop() if len(values) == 1 else None


def multipliers(model: nn.Module) -> List[float]:
    """Per-site activation lr multipliers, in module order."""
    modules = find(model)
    if not modules:
        raise ValueError('no GPS modules found; use swap_activations() first')
    return [m.multiplier for m in modules]


def multiplier(model: nn.Module) -> float:
    """The activation lr multiplier, as a multiple of the weight lr.

    This is the number to put on the activation optimizer group::

        {'params': gps_params(model), 'lr': weight_lr * gps.multiplier(model)}

    :data:`MULTIPLIER_DEFAULT` (0.5) until :func:`calibrate` has run, then
    whatever it measured.  Note that this is the *fallback* value: the probe
    epoch should run at :data:`PROBE_MULTIPLIER` (1.0), not at this.  Raises if
    the sites disagree (``share='none'`` and per-site calibration) -- read
    :func:`multipliers` instead.
    """
    modules = find(model)
    if not modules:
        raise ValueError('no GPS modules found; use swap_activations() first')
    value = _common_multiplier(modules)
    if value is None:
        raise ValueError('activation sites have different multipliers; '
                         'use multipliers(model) for the per-site list')
    return value


def set_multiplier(model: nn.Module, value: float) -> None:
    """Set the activation lr multiplier on every GPS module."""
    modules = find(model)
    if not modules:
        raise ValueError('no GPS modules found; use swap_activations() first')
    for m in modules:
        m.multiplier = float(value)


def calibrate(model: nn.Module,
              param_group: Optional[dict] = None,
              *,
              epochs: int,
              remaining: Optional[int] = None,
              target: float = 0.7,
              clip: Tuple[float, float] = (0.01, 10.0),
              init: Optional[Sequence[torch.Tensor]] = None,
              probe_at: Optional[float] = None,
              weight_lr: Optional[float] = None,
              verbose: bool = False) -> float:
    """Measure the activation lr multiplier and apply it.

    Call this once, after the first epoch.  It measures how far the activation
    parameters travelled, stores the new multiplier on the modules, and -- when
    the activation optimizer group is passed -- sets that group's learning rate
    for the remaining epochs::

        model = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 2))
        gps.swap_activations(model)
        gps.share_parameters(gps.find(model), 'global')

        weight_lr = 1e-3
        opt = torch.optim.AdamW([
            {'params': weights(model), 'lr': weight_lr},
            {'params': gps.find(model)[0].params,
             'lr': weight_lr * gps.PROBE_MULTIPLIER},    # 1.0: the probe epoch
        ])
        for epoch in range(200):
            if epoch == 1:                   # one epoch has been trained
                gps.calibrate(model, opt.param_groups[1], epochs=200)
            train_one_epoch()

    ``param_group`` must be the group that holds the activation parameters, with
    its learning rate currently set to ``weight_lr * gps.multiplier(model)`` --
    which is what the loop above does, and what the first epoch trained at.  The
    new rate is ``weight_lr * mult``, so the group is rescaled by
    ``new_mult / old_mult``.  If the group's rate is not of that form (e.g. it
    is already the bare weight rate), pass ``weight_lr=`` and it is used
    directly instead of being inferred.

    The first epoch must run at :data:`PROBE_MULTIPLIER` (1.0) -- activation lr
    equal to the weight lr -- because that epoch is the probe.  Pass
    ``probe_at=`` if your loop used a different value and the displacement is
    rescaled accordingly.  If the probe measures no travel at all, the fallback
    :data:`MULTIPLIER_DEFAULT` (0.5) is stored instead.  ``epochs`` is the
    total length of the run, and is required: it is what turns the measured
    displacement into a multiplier, so it belongs to the calibration, not to the
    model surgery.

    Returns the multiplier stored (a float when the sites agree, which is the
    normal case with ``share='global'``; otherwise a list in module order).
    Nothing is trained here -- the model is assumed to be one epoch past its
    initialisation.
    """
    modules = find(model)
    if not modules:
        raise ValueError('no GPS modules found; use swap_activations() first')

    # The probed epoch ran at the probe multiplier by definition; a caller
    # whose loop used something else says so explicitly.
    probed_at = PROBE_MULTIPLIER if probe_at is None else float(probe_at)
    if probed_at <= 0.0:
        raise ValueError(f'probe_at must be positive, got {probed_at}')

    starts, current = _probe_state(modules, init)
    seen, measured = set(), []                    # one entry per distinct tensor
    for m in modules:
        if id(m.params) in seen:
            continue
        seen.add(id(m.params))
        measured.append(m)

    values = []
    for i in range(len(measured)):
        if displacement([starts[i]], [current[i]]) <= 0.0:
            # the probe saw no travel at all: fall back rather than divide by zero
            values.append(MULTIPLIER_DEFAULT)
            continue
        values.append(probe_multiplier([starts[i]], [current[i]], epochs,
                                       remaining=remaining, target=target,
                                       clip=clip, probe_at=probed_at))

    for m, value in zip(measured, values):        # every site sharing this tensor
        for other in modules:
            if other.params is m.params:
                other.multiplier = value

    result = values[0] if len(values) == 1 else values
    if param_group is not None:
        if len(values) != 1:
            raise ValueError(
                'a single param_group needs one multiplier, but the activation '
                f'sites calibrated to {values}; apply them yourself via '
                'multipliers(model)')
        if weight_lr is None:
            if probed_at is None or probed_at <= 0.0:
                raise ValueError('cannot infer the weight lr from the param group; '
                                 'pass weight_lr=')
            weight_lr = float(param_group['lr']) / probed_at
        param_group['lr'] = float(weight_lr) * result
        if verbose:
            print(f'calibrated activation lr: {param_group["lr"]:.3e} '
                  f'(weight lr {float(weight_lr):.3e} x {result:.3f})')
    elif verbose:
        print(f'calibrated activation multiplier: {values}')

    return result


# --------------------------------------------------------------------------- #
# model surgery
# --------------------------------------------------------------------------- #
_DEFAULT_TARGETS = tuple(t for t in (nn.GELU, getattr(nn, 'SiLU', None),
                                     getattr(nn, 'Mish', None)) if t is not None)


def find(model: nn.Module) -> List[GPS]:
    """Every GPS module in ``model``, in definition order."""
    return [m for m in model.modules() if isinstance(m, GPS)]


def warmup(model: nn.Module, n: int = 1024) -> None:
    """Warm every GPS module up: compile the Triton kernel, discard the grads.

    Call after the model is on the GPU and before the first optimizer step;
    otherwise the first step pays the compilation cost.  Cheap on the PyTorch
    path.
    """
    for m in find(model):
        m.warmup(n)


def swap_activations(model: nn.Module, form: str = 'single',
                     init: Optional[object] = 'minimax',
                     targets: Optional[Sequence[type]] = None,
                     multiplier: Optional[float] = None,
                     backend: str = 'auto') -> List[GPS]:
    """Replace every static activation in ``model`` with GPS, in place.

    By default GELU, SiLU (Swish) and Mish become GPS -- the three functions the
    paper approximates.  Returns the new modules, so a caller can then share or
    group their parameters.  ReLU is deliberately not included: GPS is not
    designed as a ReLU replacement.

    ``multiplier`` is forwarded to every new module (it is the fallback value;
    the length of the run belongs to :func:`calibrate`, not here).
    """
    targets = tuple(targets) if targets is not None else _DEFAULT_TARGETS

    def walk(parent: nn.Module) -> List[GPS]:
        out: List[GPS] = []
        for name, child in list(parent.named_children()):
            if isinstance(child, targets):
                new = GPS(form=form, init=init, multiplier=multiplier,
                          backend=backend)
                setattr(parent, name, new)
                out.append(new)
            else:
                out.extend(walk(child))
        return out

    replaced = walk(model)
    if not replaced:
        raise ValueError('no GELU/SiLU/Mish submodules found; pass targets=...')
    # Follow the model's device, so swapping is safe after model.cuda() as well
    # as before.  The parameters stay float32: the fused kernel requires it, and
    # under AMP the activation parameters are normally kept in float32 anyway.
    ref = next((p for p in model.parameters()), None)
    if ref is None:
        ref = next((b for b in model.buffers()), None)
    if ref is not None and ref.device.type != 'cpu':
        for m in replaced:
            m.to(device=ref.device)
    return replaced


def fold(model: nn.Module, *, steps: int = 400, lr: float = 0.02,
         warn: bool = True) -> List[Tuple[float, float, float]]:
    """Reduce every mixture-form GPS in ``model`` to the two-parameter form.

    This is the paper's deployment step.  A trained mixture is projected onto the
    two-parameter curve by :func:`fold_weights` and each module is converted in
    place, so the network carries on running -- now with two trainable numbers
    per site instead of three::

        gps.swap_activations(model, form='mixture')
        train(model)                       # the CNN scheme, lambda = 0.5
        gps.fold(model)                    # every mixture site -> its fold
        recalibrate_batchnorm(model)       # the fold does not move BN statistics

    Returns ``(a, b, mse)`` per folded site, in definition order.  Sites that
    already use the single form are left alone.

    .. warning::
       The fold changes the curve at each activation but **not** the batch-norm
       statistics, which were collected while the network was running the
       mixture.  Recalibrate batch norm (a forward pass over the training data in
       ``eval()`` mode) before measuring or deploying the folded model -- the
       paper does this.  Silence this with ``warn=False``.
    """
    modules = [m for m in find(model) if m.form == 'mixture']
    if not modules:
        raise ValueError('no mixture-form GPS modules to fold; build them with '
                         "swap_activations(model, form='mixture')")
    results = []
    for m in modules:
        a, b, mse = fold_weights(m.weights().detach().cpu(),
                                 m.basis.detach().cpu(), steps=steps, lr=lr)
        # convert the module in place: same object, same device, same multiplier,
        # so anything holding a reference to it (including a shared parameter)
        # keeps working
        device, dtype = m.params.device, m.params.dtype
        m.form = 'single'
        m.params = nn.Parameter(torch.tensor([a, b], device=device, dtype=dtype))
        m.init_params = m.params.detach().clone()   # the buffer, re-seeded
        results.append((a, b, mse))
    if warn:
        warnings.warn(
            f'fold() converted {len(modules)} mixture site(s) to the two-parameter '
            'form but left the batch-norm statistics alone. Recalibrate batch norm '
            '(forward pass over the training data in eval() mode) before '
            'evaluating or deploying the model.',
            RuntimeWarning, stacklevel=2)
    return results


def share_parameters(modules: Iterable[GPS], mode: str = 'global') -> None:
    """Share the trainable tensor across activation sites.

    ``'global'``  one parameter set for the whole network: 2 trainable numbers
                  for a single-form GPS, 3 for a mixture, regardless of depth.
    ``'none'``    one set per site (no change).

    For a mixture, sharing the three logits means every layer uses the same
    blend of the three Gaussians; ``'none'`` lets each layer choose.
    """
    modules = list(modules)
    if mode == 'none':
        return
    if mode != 'global':
        raise ValueError(f"mode must be 'global' or 'none', got {mode!r}")
    if len(modules) < 2:
        return
    first = modules[0].params
    if any(m.params.shape != first.shape for m in modules):
        raise ValueError('cannot share: activation sites have different parameter shapes '
                         '(are you mixing forms?)')
    for m in modules:
        m.params = first                          # same tensor, tied


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    torch.manual_seed(0)

    EPOCHS = 20
    WEIGHT_LR = 1e-3

    net = nn.Sequential(nn.Linear(16, 32), nn.GELU(),
                        nn.Linear(32, 32), nn.GELU(),
                        nn.Linear(32, 4))
    mods = swap_activations(net)
    share_parameters(mods, 'global')
    print(f'2-parameter form: {len(mods)} sites, '
          f'{sum(p.numel() for p in net.parameters())} params total, '
          f'{mods[0].params.numel()} trainable in the activation')

    x = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    weight_params = [p for n, p in net.named_parameters() if 'params' not in n]
    opt = torch.optim.AdamW(
        [{'params': weight_params, 'lr': WEIGHT_LR},
         {'params': [mods[0].params], 'lr': WEIGHT_LR * PROBE_MULTIPLIER}],
        weight_decay=0.1)

    def train_one_epoch(steps: int = 4):
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(net(x), y).backward()
            opt.step()

    print(f'activation lr = weight lr x {PROBE_MULTIPLIER:.1f} for the probe epoch '
          f'(fallback {MULTIPLIER_DEFAULT})')
    for epoch in range(EPOCHS):
        if epoch == 1:                        # one epoch has been trained
            calibrate(net, opt.param_groups[1], epochs=EPOCHS)
            print(f'  epoch {epoch}: multiplier {multiplier(net):.3f}, '
                  f'act lr {opt.param_groups[1]["lr"]:.2e}')
        train_one_epoch()
    d = displacement([torch.tensor(INIT_AB)], [mods[0].params.detach()])
    print(f'after {EPOCHS} epochs the curve sits at {mods[0]} '
          f'(travelled {d:.4f})')

    print('-' * 62)
    net_mix = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))
    swap_activations(net_mix, form='mixture')
    print(f'mixture form: {find(net_mix)[0]}  '
          f'(weights sum to {float(find(net_mix)[0].weights().detach().sum()):.3f})')
    folds = fold(net_mix)                         # warns about batch norm
    print(f'after fold(model): forms {[m.form for m in find(net_mix)]}, '
          f'params per site {[m.params.numel() for m in find(net_mix)]}')
    print(f'  per-site fold (a, b): {[(round(a, 4), round(b, 4)) for a, b, _ in folds]}')


    for f in ('gelu', 'swish', 'mish'):
        w = [0.0, 0.0, 0.0]
        w[BASIS_ORDER.index(f)] = 1.0
        a, b, mse = fold_weights(w)
        print(f'  fold({f:5s}) -> a={a:.4f}, b={b:.4f} (mse {mse:.2e}); '
              f'fit={BASIS[f]}')
