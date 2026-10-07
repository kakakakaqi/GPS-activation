"""GPS-native choice/blend activations (structure-preserving).

The paper's GPS:  f(x) = softplus(x) - ln2 * exp(-a x^2 - b x)
                  = softplus(x) - ln2 * g(x),  g = Gaussian decay (the wave).

Blending the three fitted forms INSIDE the wave term (not by averaging static
outputs):

    f(x) = softplus(x) - ln2 * sum_i w_i * exp(-a_i x^2 - b_i x),  sum w_i = 1

The result is a single activation of the paper's class (S-shaped + wave-like),
whose wave is a Gaussian mixture — strictly richer than one Gaussian. Structural
guarantees hold by construction for ANY weights: softmax gives sum w = 1 so
f(0) = 0; convexity keeps each effective a > 0; left limit 0 and right
asymptote y=x are preserved. The three fitted forms are the simplex vertices.

Two modules:
  MixChoiceGPS  - the mixture-wave activation (softmax or straight-through
                  Gumbel weights; `params` = the logits).
  ParamBlendGPS - the 2-parameter subfamily: (a, b) = sum w_i (a_i, b_i),
                  evaluated with the original Triton kernel.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import gps_lab
from gps_lab import GaussianDecaySoftplusTriton

LOG2E = 1.4426950408889634
FORMS = torch.tensor([[0.35028508, 0.0],        # GELU fit
                      [0.14575758, 0.0],        # Swish fit
                      [0.21961491, 0.16713308]])  # Mish fit
FORM_INDEX = {'gelu': 0, 'silu': 1, 'swish': 1, 'mish': 2}


class _NoiseSampler:
    """Batched per-forward shape-noise sampler.

    One batched draw of `n` i.i.d. weight vectors per model forward; site i
    consumes row i (call order = construction order, deterministic). Same
    i.i.d. semantics as per-call drawing, but the expensive
    implicit-reparameterization backward runs ONCE per forward instead of once
    per site.  `n` is set to the number of sharing sites (see
    share_activation_params); default 1 = redraw every call.
    """

    def __init__(self):
        self.n = 1
        self.rows = None
        self.idx = 0

    def get(self, draw_fn):
        if self.rows is None or self.idx >= self.n:
            self.rows = draw_fn(self.n)
            self.idx = 0
        w = self.rows[self.idx]
        self.idx += 1
        return w


def _dirichlet_kl(alpha):
    """KL(Dir(alpha) || Dir(1)) — entropy-style regularizer for collapse."""
    a0 = alpha.sum()
    return (torch.lgamma(alpha).sum() - torch.lgamma(a0)
            + (a0 - 3) * torch.digamma(a0)
            - ((alpha - 1) * torch.digamma(alpha)).sum())


class _MixWave(torch.autograd.Function):
    """f(x) = softplus(x) - ln2 * sum_i w_i exp(-a_i x^2 - b_i x), sum w = 1.

    Fused Triton forward/backward (forms fixed, only w trains): memory
    footprint of one activation buffer per site, same convention as the
    paper's kernel (clamp treated as constant). PyTorch reference kept in
    _mixwave_ref for verification.
    """

    @staticmethod
    def forward(ctx, x, w, forms):
        if not x.is_contiguous():
            x = x.contiguous()
        ctx.save_for_backward(x, w, forms)
        out = torch.empty_like(x)
        n = x.numel()
        flat_forms = forms.contiguous()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _mix_fwd_kernel[grid](x, w, flat_forms, out, n,
                              math.log(2.0), LOG2E, 80.0, 3)
        return out

    @staticmethod
    def backward(ctx, go):
        x, w, forms = ctx.saved_tensors
        go = go.contiguous()
        n = x.numel()
        gx = torch.empty_like(x)
        # per-program partials + explicit reduction (deterministic, and safe
        # under autotune's benchmark executions — no atomic side effects)
        rows = triton.cdiv(n, 1024)
        partial = torch.zeros(rows, 9, device=x.device)  # gw(3), ga(3), gb(3)
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        _mix_bwd_kernel[grid](x, w, forms.contiguous(), go, gx, partial, n,
                              math.log(2.0), LOG2E, 80.0, 3)
        gw = partial[:, :3].sum(0)
        gforms = torch.stack([partial[:, 3:6].sum(0), partial[:, 6:9].sum(0)], dim=1)
        return gx, gw, gforms


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ], key=['n_elements'])
@triton.jit
def _mix_fwd_kernel(x_ptr, w_ptr, forms_ptr, out_ptr, n_elements,
                    ln2: tl.constexpr, log2e: tl.constexpr,
                    MAX_EXP: tl.constexpr, N: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bump = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(N):
        a = tl.load(forms_ptr + 2 * i)
        b = tl.load(forms_ptr + 2 * i + 1)
        w = tl.load(w_ptr + i)
        expo = tl.minimum(-a * x * x - b * x, MAX_EXP)
        bump += w * tl.exp2(expo * log2e)
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp2(-tl.abs(x) * log2e))
    tl.store(out_ptr + offsets, softplus - ln2 * bump, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ], key=['n_elements'])
@triton.jit
def _mix_bwd_kernel(x_ptr, w_ptr, forms_ptr, go_ptr, gx_ptr, partial_ptr,
                    n_elements,
                    ln2: tl.constexpr, log2e: tl.constexpr,
                    MAX_EXP: tl.constexpr, N: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    go = tl.load(go_ptr + offsets, mask=mask, other=0.0)
    gsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(N):
        a = tl.load(forms_ptr + 2 * i)
        b = tl.load(forms_ptr + 2 * i + 1)
        w = tl.load(w_ptr + i)
        expo = tl.minimum(-a * x * x - b * x, MAX_EXP)
        g = tl.exp2(expo * log2e)
        t = ln2 * w * g
        gsum += t * (2.0 * a * x + b)
        gw_i = -ln2 * tl.sum(go * g, axis=0)
        tl.store(partial_ptr + pid * 9 + i, gw_i)
        # d/da_i = sum go * t * x^2 ; d/db_i = sum go * t * x
        tl.store(partial_ptr + pid * 9 + 3 + i, tl.sum(go * t * x * x, axis=0))
        tl.store(partial_ptr + pid * 9 + 6 + i, tl.sum(go * t * x, axis=0))
    gx = go * (tl.sigmoid(x) + gsum)
    tl.store(gx_ptr + offsets, gx, mask=mask)


class MixChoiceGPS(nn.Module):
    def __init__(self, mode='blend', tau=1.0, init=None, init_bias=2.0,
                 tau_min=0.03, tau_calls=20000):
        super().__init__()
        self.mode = mode          # 'blend' | 'gumbel' | 'anneal'
        self.tau = tau
        self.tau_min = tau_min
        self.tau_calls = tau_calls
        self._calls = 0
        logits = torch.zeros(3)
        if isinstance(init, (list, tuple)):
            logits = torch.log(torch.tensor(init, dtype=torch.float32))
        elif init == 'minimax':
            logits = torch.log(torch.tensor([0.361, 0.307, 0.332]))
        elif init is not None:
            logits[FORM_INDEX[init]] = init_bias
        self.params = nn.Parameter(logits)   # lab machinery keys on .params
        self.register_buffer('forms', FORMS.clone())
        self._hardened = False

    def weights(self, hard: bool = None):
        logits = self.params
        if self.mode == 'anneal':
            t = min(1.0, self._calls / max(1, self.tau_calls))
            if self.training:
                self._calls += 1
            tau = self.tau * (self.tau_min / self.tau) ** t
            return torch.softmax(logits / tau, dim=0)
        if self.mode == 'gumbel' and self.training and not self._hardened:
            y_soft = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            return y_soft
        w = torch.softmax(logits, dim=0)
        if hard or self._hardened:
            return F.one_hot(logits.argmax(), 3).to(w.dtype)
        return w

    def harden(self):
        """Freeze to the argmax form (call at the end of the choice window)."""
        self._hardened = True

    def forward(self, x):
        w = self.weights()
        return _MixWave.apply(x, w, self.forms)

    def extra_repr(self):
        return f"mode={self.mode}, w={torch.softmax(self.params, 0).detach().tolist()}"


class MixDistGPS(MixChoiceGPS):
    """Distributional weights on the form simplex: sample w during training,
    collapse to the distribution's mean at eval / after `harden()`.

    modes:
      'dirichlet'  w ~ Dir(alpha),  alpha = softplus(params) + eps
                   (reparameterized rsample; every sample keeps sum w = 1,
                    so f(0)=0 holds sample-by-sample)
      'lognorm'    w = softmax(mu + sigma*eps),  sigma = softplus(scale)+eps
                   (logistic-normal: normal noise mapped to the simplex)
    'sigma_scale' multiplies the noise/concentration schedule; set
    self.sigma_scale = 0 externally to anneal the distribution to a point.
    """

    def __init__(self, mode='dirichlet', init=None, init_bias=2.0,
                 sigma_scale=1.0, concentration=4.0):
        super().__init__(mode='blend', init=init, init_bias=init_bias)
        self.mode = mode
        self.sigma_scale = sigma_scale
        self._sampler = _NoiseSampler()
        if mode == 'dirichlet':
            # params are unconstrained; alpha = softplus(params) + eps
            base = torch.full((3,), float(concentration))
            if init is not None:
                base[FORM_INDEX[init]] = concentration * 2
            with torch.no_grad():
                self.params.copy_(torch.log(torch.expm1(base.clamp(min=1e-3))))
        self.scale = nn.Parameter(torch.tensor(0.0))   # log-noise-scale (lognorm)

    def weights(self, hard: bool = None):
        if self._hardened or not self.training or self.sigma_scale == 0.0:
            return self.mean_w()
        if self.mode == 'dirichlet':
            alpha = (F.softplus(self.params) + 1e-4) * max(self.sigma_scale, 1e-3)
            return self._sampler.get(
                lambda k: torch.distributions.Dirichlet(alpha).rsample((k,)))
        # lognorm
        mu = self.params
        sig = F.softplus(self.scale) + 1e-4

        def draw(k):
            return torch.softmax(mu + sig * self.sigma_scale * torch.randn(k, 3, device=mu.device), dim=-1)
        return self._sampler.get(draw)

    def mean_w(self):
        if self.mode == 'dirichlet':
            alpha = F.softplus(self.params) + 1e-4
            return alpha / alpha.sum()
        return torch.softmax(self.params, dim=0)

    def forward(self, x):
        return _MixWave.apply(x, self.weights(), self.forms)


class MixFreeGPS(nn.Module):
    """Free mixture-wave GPS: the three (a_i, b_i) pairs AND the ratios w_i are
    trained jointly.  f(x) = softplus(x) - ln2 * sum w_i exp(-a_i x^2 - b_i x),
    sum w = 1 (=> f(0)=0 for ANY (a_i,b_i)); a_i > 0 enforced by softplus.
    Single 9-vector parameter [raw_a(3), raw_b(3), logits(3)] so the lab's
    share/freeze/snapshot machinery treats it as one shape parameter set.
    """

    def __init__(self, init=None, init_bias=2.0):
        super().__init__()
        raw_a = torch.log(torch.expm1(FORMS[:, 0].clamp(min=1e-3)))
        raw_b = FORMS[:, 1].clone()
        logits = torch.zeros(3)
        if isinstance(init, (list, tuple)):
            logits = torch.log(torch.tensor(init, dtype=torch.float32))
        elif init == 'minimax':
            logits = torch.log(torch.tensor([0.361, 0.307, 0.332]))
        elif init is not None:
            logits[FORM_INDEX[init]] = init_bias
        self.params = nn.Parameter(torch.cat([raw_a, raw_b, logits]))
        self.register_buffer('forms0', FORMS.clone())

    def split(self):
        p = self.params
        a = F.softplus(p[:3]) + 1e-3
        forms = torch.stack([a, p[3:6]], dim=1)
        w = torch.softmax(p[6:9], dim=0)
        return w, forms

    def forward(self, x):
        w, forms = self.split()
        return _MixWave.apply(x, w, forms)

    def extra_repr(self):
        w, forms = self.split()
        return f"w={w.detach().tolist()}, (a,b)={forms.detach().tolist()}"


class ParamBlendGPS(nn.Module):
    """(a,b) = sum w_i (a_i,b_i): stays in the original 2-param GPS family."""

    def __init__(self, mode='blend', tau=1.0, init=None, init_bias=2.0):
        super().__init__()
        self.mode = mode
        self.tau = tau
        logits = torch.zeros(3)
        if isinstance(init, (list, tuple)):
            logits = torch.log(torch.tensor(init, dtype=torch.float32))
        elif init == 'minimax':
            logits = torch.log(torch.tensor([0.361, 0.307, 0.332]))
        elif init is not None:
            logits[FORM_INDEX[init]] = init_bias
        self.params = nn.Parameter(logits)
        self.register_buffer('forms', FORMS.clone())

    def weights(self):
        if self.mode == 'gumbel' and self.training:
            return F.gumbel_softmax(self.params, tau=self.tau, hard=True)
        return torch.softmax(self.params, dim=0)

    def effective_ab(self):
        w = self.weights()
        return (w[:, None] * self.forms).sum(0)

    def forward(self, x):
        ab = self.effective_ab()
        return GaussianDecaySoftplusTriton.apply(x, ab)

    def extra_repr(self):
        ab = self.effective_ab().detach()
        return f"mode={self.mode}, (a,b)=({ab[0]:.4f},{ab[1]:.4f})"


# ---- register into the lab -------------------------------------------------
def _factory(cls):
    def make(init=None, **kwargs):
        return cls(init=init, **kwargs)
    return make


gps_lab.ActivationRegistry.FACTORIES['mixchoice'] = _factory(MixChoiceGPS)
gps_lab.ActivationRegistry.FACTORIES['mixdist'] = _factory(MixDistGPS)
gps_lab.ActivationRegistry.FACTORIES['mixfree'] = _factory(MixFreeGPS)
gps_lab.ActivationRegistry.FACTORIES['paramblend'] = _factory(ParamBlendGPS)
gps_lab.TRAINABLE_ACT_CLASSES = (gps_lab.Gaussian_pdf_softplus_v4,
                                 MixChoiceGPS, ParamBlendGPS, MixFreeGPS)
