"""GPS activation lab — library code extracted from GPS_experiments_v6.ipynb."""


# ======== from notebook cell 1 ========
import csv
import gc
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import os
import triton
import triton.language as tl

# ======== from notebook cell 3 ========
DEFAULT_BLOCK_SIZE = 1024
LOG2E = 1.4426950408889634  # 1/ln(2), folds exp -> hardware exp2


# ------------------------------------------------------------------------------
# Forward kernel: single-exp softplus, exp2, no num_stages (no loop to pipeline)
# ------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gaussian_decay_softplus_fwd_kernel(
    x_ptr, params_ptr, out_ptr,
    n_elements,
    ln2: tl.constexpr,
    log2e: tl.constexpr,
    MAX_EXP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    a = tl.load(params_ptr)
    b = tl.load(params_ptr + 1)

    # softplus(x) = max(x,0) + log(1 + exp(-|x|)): one exp, one log, stable
    softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp2(-tl.abs(x) * log2e))

    exponent = tl.minimum(-a * x * x - b * x, MAX_EXP)
    g = tl.exp2(exponent * log2e)

    tl.store(out_ptr + offsets, softplus - ln2 * g, mask=mask)


# ------------------------------------------------------------------------------
# Backward kernel: persistent grid-stride loop -> O(#programs) atomics,
# masked loads use other=0.0 so tail lanes contribute exactly zero to the sums
# ------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def gaussian_decay_softplus_bwd_kernel(
    x_ptr, params_ptr, grad_out_ptr,
    grad_x_ptr, grad_params_ptr,
    n_elements,
    ln2: tl.constexpr,
    log2e: tl.constexpr,
    MAX_EXP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK_SIZE

    a = tl.load(params_ptr)
    b = tl.load(params_ptr + 1)

    acc_a = 0.0
    acc_b = 0.0

    for start in range(pid * BLOCK_SIZE, n_elements, step):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)

        exponent = tl.minimum(-a * x * x - b * x, MAX_EXP)
        t = ln2 * tl.exp2(exponent * log2e)   # ln2 * g, reused by all 3 grads
        sigmoid = tl.sigmoid(x)

        tl.store(grad_x_ptr + offsets,
                 grad_out * (sigmoid + t * (2.0 * a * x + b)), mask=mask)

        acc_a += tl.sum(grad_out * (t * x * x), axis=0)
        acc_b += tl.sum(grad_out * (t * x), axis=0)

    # One atomic pair per program instead of per block
    tl.atomic_add(grad_params_ptr, acc_a)
    tl.atomic_add(grad_params_ptr + 1, acc_b)


# ------------------------------------------------------------------------------
# Autograd Function with capped persistent grid
# ------------------------------------------------------------------------------
_SM_COUNT = {}

def _program_cap(device) -> int:
    idx = device.index or 0
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return 4 * _SM_COUNT[idx]   # a few waves per SM is plenty


class GaussianDecaySoftplusTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, params):
        if not x.is_contiguous():
            x = x.contiguous()
        ctx.save_for_backward(x, params)
        out = torch.empty_like(x)
        n_elements = x.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        gaussian_decay_softplus_fwd_kernel[grid](
            x, params, out, n_elements, math.log(2.0), LOG2E, 80.0
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, params = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        n_elements = x.numel()

        grad_x = torch.empty_like(x)
        grad_params = torch.zeros_like(params)
        cap = _program_cap(x.device)

        grid = lambda meta: (min(triton.cdiv(n_elements, meta['BLOCK_SIZE']), cap),)
        gaussian_decay_softplus_bwd_kernel[grid](
            x, params, grad_output, grad_x, grad_params,
            n_elements, math.log(2.0), LOG2E, 80.0
        )
        return grad_x, grad_params


# ------------------------------------------------------------------------------
# PyTorch Module
# ------------------------------------------------------------------------------
class Gaussian_pdf_softplus_v4(nn.Module):
    def __init__(self, a: float = 0.3519, b: float = 0.00219, init=None):
        super().__init__()
        if init is not None:
            return getattr(self, f'init_{init}')
        else:
            self.params = nn.Parameter(torch.tensor([a, b], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GaussianDecaySoftplusTriton.apply(x, self.params)

    def extra_repr(self) -> str:
        a, b = self.params.tolist()
        return f"a={a:.4f}, b={b:.4f}"

    # Convenience constructors (same as before)
    @classmethod
    def from_beta_gamma(cls, beta: float, gamma: float):
        a = 0.5 * beta * beta
        b = beta * gamma
        return cls(a=a, b=b)

    @classmethod
    def init_silu(cls):
        return cls(0.14575758, 0.00000000)

    @classmethod
    def init_gelu(cls):
        return cls(0.35028508, 0.00000000)

    @classmethod
    def init_mish(cls):
        return cls(0.21961491, 0.16713308)

    @classmethod
    def init_generic(cls):
        return cls(a=0.125, b=0.0)

    @torch.no_grad()
    def warmup(self, n=1024):
        """Trigger Triton compilation/autotune on the same device as self.params."""
        device = self.params.device
        dtype = self.params.dtype
        x = torch.randn(n, device=device, dtype=dtype, requires_grad=True)
        out = self.forward(x)
        loss = out.sum()
        loss.backward()
        self.zero_grad()


# Modules whose parameters are treated as "activation shape parameters":
# separate optimizer group, freezable, loggable, regularizable.
TRAINABLE_ACT_CLASSES: Tuple[type, ...] = (Gaussian_pdf_softplus_v4)

# ======== from notebook cell 5 ========
def _trainable_factory(cls):
    """Factory for trainable activations: v2(init='gelu') etc."""
    def make(init: Optional[str] = None, **kwargs):
        act = cls(**kwargs)
        if init is not None:
            out = getattr(act, f'init_{init}')()
            if out is not None:   # classmethod-style init returns a new instance
                act = out
        return act
    return make


class ActivationRegistry:
    """name -> factory. Trainable ones accept init='...'."""

    FACTORIES: Dict[str, Callable[..., nn.Module]] = {
        # --- fixed baselines ---
        'relu':        lambda: nn.ReLU(inplace=True),
        'leaky_relu':  lambda negative_slope=0.01: nn.LeakyReLU(negative_slope, inplace=True),
        'gelu':        lambda: nn.GELU(),
        'silu':        lambda: nn.SiLU(inplace=True),
        'mish':        lambda: nn.Mish(inplace=True),
        'elu':         lambda alpha=1.0: nn.ELU(alpha=alpha, inplace=True),
        'selu':        lambda: nn.SELU(inplace=True),
        'tanh':        lambda: nn.Tanh(),
        'sigmoid':     lambda: nn.Sigmoid(),
        'prelu':       lambda num_parameters=1: nn.PReLU(num_parameters),
        'hardswish':   lambda: nn.Hardswish(inplace=True),
        'serf':        lambda: type('SERF', (nn.Module,), {
                            'forward': lambda self, x: x * torch.erf(F.softplus(x))})(),
        # --- trainable general forms ---
        'pdfv4':       _trainable_factory(Gaussian_pdf_softplus_v4),
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> nn.Module:
        key = name.lower().replace('-', '_')
        if key not in cls.FACTORIES:
            raise ValueError(f"Unknown activation '{name}'. Available: {cls.available()}")
        try:
            return cls.FACTORIES[key](**kwargs)
        except TypeError:
            return cls.FACTORIES[key]()

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls.FACTORIES.keys())


import re


def _normalize_act_spec(spec) -> Tuple[str, dict]:
    """Normalize a per-site activation spec to (name, kwargs).

    Accepted forms:
        'gelu'                              -> ('gelu', {})
        ('pdfv4', {'init': 'gelu'})         -> ('pdfv4', {'init': 'gelu'})
        {'name': 'pdfv4', 'kwargs': {...}}  -> ('pdfv4', {...})
    """
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        return spec['name'], dict(spec.get('kwargs', {}))
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return spec[0], dict(spec[1])
    raise ValueError(f"Bad activation spec: {spec!r}. Use a name, (name, kwargs), "
                     "or {'name': ..., 'kwargs': ...}.")


def _selector_to_indices(sel, n: int) -> set:
    """Resolve a site selector against n total sites -> set of site indices.

    Selectors (site index = position in network construction order):
        '*' | 'all'         every site
        '7'                 single site
        '3-9' / '3-' / '-9' inclusive index ranges
        'last 30%'          final 30% of sites
        'first 70%'         initial 70% of sites
        '70%-100%'          explicit fraction range of total sites
    """
    s = str(sel).strip().lower()
    if s in ('*', 'all'):
        return set(range(n))
    m = re.fullmatch(r'last\s+(\d+(?:\.\d+)?)\s*%', s)
    if m:
        k = int(round(n * float(m.group(1)) / 100.0))
        return set(range(max(0, n - k), n))
    m = re.fullmatch(r'first\s+(\d+(?:\.\d+)?)\s*%', s)
    if m:
        k = int(round(n * float(m.group(1)) / 100.0))
        return set(range(min(n, k)))
    m = re.fullmatch(r'(\d+(?:\.\d+)?)\s*%-\s*(\d+(?:\.\d+)?)\s*%', s)
    if m:
        lo = int(round(n * float(m.group(1)) / 100.0))
        hi = int(round(n * float(m.group(2)) / 100.0))
        return set(range(min(lo, n), min(hi, n)))
    m = re.fullmatch(r'(\d*)-(\d*)', s)
    if m and (m.group(1) or m.group(2)):
        lo = int(m.group(1)) if m.group(1) else 0
        hi = int(m.group(2)) if m.group(2) else n - 1
        return set(range(max(0, lo), min(hi, n - 1) + 1))
    if re.fullmatch(r'\d+', s):
        i = int(s)
        if not (0 <= i < n):
            raise ValueError(f"Site index {i} out of range for {n} sites.")
        return {i}
    raise ValueError(f"Unparseable site selector {sel!r}. Use '*', an index, "
                     "'a-b', 'last X%', 'first X%', or 'A%-B%'.")


class ActivationPlan:
    """Assign different activations to different sites of a network.

    A "site" is one `make_act` call during model construction, numbered in
    construction order (site 0 = first activation built, e.g. the stem).
    Fraction selectors ('last 30%') are resolved against the total site count,
    which `run_experiment` measures automatically via a dry-run build.

    Args:
        default: spec (see `_normalize_act_spec`) for sites not covered by rules.
        rules:   list of (selector, spec), applied in order — later rules win
                 on overlap.

    Example — GELU everywhere except the last 30% of sites, which are
    trainable pdfv4 initialized as GELU:

        plan = ActivationPlan('gelu', [('last 30%', ('pdfv4', {'init': 'gelu'}))])
        ExpConfig(name='mix', activation=plan, ...)

    Note: when a plan is used, `ExpConfig.activation_kwargs` is IGNORED —
    each rule carries its own kwargs, so mixed inits stay unambiguous.
    """

    def __init__(self, default='relu', rules=None):
        self.default = _normalize_act_spec(default)
        self.rules = [(sel, _normalize_act_spec(spec)) for sel, spec in (rules or [])]
        self.n_sites: Optional[int] = None
        self._assignments: Optional[List[Tuple[str, dict]]] = None
        self._cursor = 0
        self._counting = False
        self._count = 0

    # -- resolution ---------------------------------------------------------
    def resolve(self, n_sites: int) -> "ActivationPlan":
        """Fix the total site count and materialize per-site assignments."""
        self.n_sites = n_sites
        self._assignments = [self.default] * n_sites
        for sel, spec in self.rules:
            for i in sorted(_selector_to_indices(sel, n_sites)):
                self._assignments[i] = spec
        self._cursor = 0
        return self

    def count_sites(self, arch: str, **model_kwargs) -> int:
        """Dry-run the architecture on CPU and count activation sites.

        Runs under fork_rng so weight-init draws are replayed identically
        during the real build — the dry run has zero effect on seeding.
        """
        self._counting, self._count = True, 0
        try:
            with torch.random.fork_rng():
                ArchitectureRegistry.get(arch, activation=self, **model_kwargs)
        finally:
            self._counting = False
        return self._count

    # -- used by make_act during construction --------------------------------
    def next(self) -> nn.Module:
        if self._counting:
            self._count += 1
            return nn.ReLU(inplace=True)  # cheap placeholder for the dry run
        if self._assignments is None:
            raise RuntimeError(
                "ActivationPlan used before resolve(). Pass it via "
                "ExpConfig.activation and run through run_experiment "
                "(auto-resolved), or call plan.resolve(n_sites) yourself.")
        if self._cursor >= len(self._assignments):
            raise RuntimeError(
                f"ActivationPlan exhausted: model asked for site {self._cursor} "
                f"but only {len(self._assignments)} were resolved.")
        name, kwargs = self._assignments[self._cursor]
        self._cursor += 1
        return ActivationRegistry.get(name, **kwargs)

    # -- reporting ------------------------------------------------------------
    def describe(self) -> str:
        """Run-length encoded site map, e.g. '0-13:gelu | 14-19:pdfv4'."""
        if self._assignments is None:
            return str(self)
        runs, start = [], 0
        for i in range(1, len(self._assignments) + 1):
            if i == len(self._assignments) or self._assignments[i] != self._assignments[start]:
                name, kw = self._assignments[start]
                label = name + (f"({kw})" if kw else "")
                runs.append(f"{start}-{i - 1}:{label}" if i - 1 > start else f"{start}:{label}")
                start = i
        return ' | '.join(runs)

    def __str__(self) -> str:
        parts = [f"default={self.default[0]}"]
        parts += [f"{sel}->{name}" for sel, (name, _) in self.rules]
        return f"plan[{'; '.join(parts)}]"


def make_act(activation, act_kwargs: Optional[dict] = None) -> nn.Module:
    """Build one activation module.

    `activation` may be:
      - a registry name string ('gelu', 'pdfv4', ...) — original behavior;
      - an ActivationPlan — hands out the per-site module (plan kwargs win,
        global act_kwargs ignored);
      - a per-site spec — (name, kwargs) or {'name':..., 'kwargs':...};
        kwargs are merged over act_kwargs (spec wins on conflict).
    """
    if isinstance(activation, ActivationPlan):
        return activation.next()
    if not isinstance(activation, str):
        name, kwargs = _normalize_act_spec(activation)
        return ActivationRegistry.get(name, **{**(act_kwargs or {}), **kwargs})
    return ActivationRegistry.get(activation, **(act_kwargs or {}))

# ======== from notebook cell 6 ========
class ConvBlock(nn.Module):
    """Conv + BN + optional activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 activation='relu', act_kwargs=None, use_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = make_act(activation, act_kwargs) if use_activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, activation='relu', act_kwargs=None):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, activation=activation, act_kwargs=act_kwargs)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = make_act(activation, act_kwargs)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return self.act(out + x)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, activation='relu', in_channels=3, act_kwargs=None):
        super().__init__()
        # Stem: conv7x7, BN, ReLU, maxpool
        self.stem = ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1,
                              activation=activation, act_kwargs=act_kwargs)
        # Four stages, each with 2 basic residual blocks
        self.layer1 = self._make_layer(64, 2, activation, act_kwargs, stride=1)
        self.layer2 = self._make_layer(128, 2, activation, act_kwargs, stride=2)
        self.layer3 = self._make_layer(256, 2, activation, act_kwargs, stride=2)
        self.layer4 = self._make_layer(512, 2, activation, act_kwargs, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, activation, act_kwargs, stride=1):
        """
        Build a stage that consists of a first block that may downsample,
        followed by `blocks-1` standard residual blocks.
        The first block is a ConvBlock (conv+BN+ReLU) that changes the channel size
        and possibly the spatial size. Then we add residual blocks.
        (This matches the pattern used in ResNetMini.)
        """
        in_channels = channels // 2 if stride > 1 else channels
        layers = [
            ConvBlock(in_channels, channels,
                      kernel_size=3, stride=stride, padding=1,
                      activation=activation, act_kwargs=act_kwargs)
        ]
        # Then add residual blocks (basic blocks) that keep channels same
        for _ in range(blocks):
            layers.append(ResidualBlock(channels, activation, act_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class BottleneckBlock(nn.Module):
    """Bottleneck residual block for ResNet‑50."""
    def __init__(self, in_channels, out_channels, stride=1,
                 activation='relu', act_kwargs=None):
        super().__init__()
        bottleneck = out_channels // 4

        # 1×1 conv, reduce
        self.conv1 = ConvBlock(in_channels, bottleneck, 1, 1, 0,
                               activation, act_kwargs, use_activation=True)
        # 3×3 conv, spatial processing
        self.conv2 = ConvBlock(bottleneck, bottleneck, 3, stride, 1,
                               activation, act_kwargs, use_activation=True)
        # 1×1 conv, expand (no activation after BN)
        self.conv3 = ConvBlock(bottleneck, out_channels, 1, 1, 0,
                               activation, act_kwargs, use_activation=False)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, 1, stride, 0,
                                      activation, act_kwargs, use_activation=False)

        self.activation = make_act(activation, act_kwargs)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(residual)
        out = self.activation(out)
        return out


class ResNet34(nn.Module):
    """ResNet‑34 with CIFAR stem (3×3 conv, stride 1, no maxpool)."""
    def __init__(self, num_classes=1000, activation='relu',
                 in_channels=3, act_kwargs=None):
        super().__init__()

        # CIFAR stem: 3×3 conv, stride 1, padding 1 (no maxpool)
        self.stem = ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1,
                              activation=activation, act_kwargs=act_kwargs)

        # Four stages with basic residual blocks: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 3, activation, act_kwargs, stride=1)
        self.layer2 = self._make_layer(128, 4, activation, act_kwargs, stride=2)
        self.layer3 = self._make_layer(256, 6, activation, act_kwargs, stride=2)
        self.layer4 = self._make_layer(512, 3, activation, act_kwargs, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, activation, act_kwargs, stride=1):
        """
        Build a stage:
        - First block: ConvBlock (conv+BN+ReLU) that may change channels/spatial size.
        - Remaining blocks: standard ResidualBlock (identity shortcut).
        """
        in_channels = channels // 2 if stride > 1 else channels
        layers = [
            ConvBlock(in_channels, channels,
                      kernel_size=3, stride=stride, padding=1,
                      activation=activation, act_kwargs=act_kwargs)
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(channels, activation, act_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)

        
class ResNet50(nn.Module):
    """ResNet‑50 with the same API as the provided ResNet18."""
    def __init__(self, num_classes=1000, activation='relu',
                 in_channels=3, act_kwargs=None):
        super().__init__()

        # Stem: 7×7 conv + BN + ReLU, then max pool
        self.stem = ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1,
                              activation=activation, act_kwargs=act_kwargs)

        # Four stages with bottleneck blocks: [3, 4, 6, 3] blocks per stage
        self.layer1 = self._make_layer(64, 3, activation, act_kwargs, stride=1)
        self.layer2 = self._make_layer(128, 4, activation, act_kwargs, stride=2)
        self.layer3 = self._make_layer(256, 6, activation, act_kwargs, stride=2)
        self.layer4 = self._make_layer(512, 3, activation, act_kwargs, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, activation, act_kwargs, stride=1):
        """
        Build a stage where the first block handles downsampling (if stride>1)
        and channel change; the remaining blocks are identity.
        """
        # Input channels of the stage (previous stage's output)
        in_channels = channels // 2 if stride > 1 else channels

        layers = []
        # First block with possible stride and projection
        layers.append(BottleneckBlock(in_channels, channels, stride,
                                      activation, act_kwargs))
        # Remaining blocks: stride=1, same in/out channels
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(channels, channels, 1,
                                          activation, act_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)



class ArchitectureRegistry:
    MODELS = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
    }

    @classmethod
    def get(cls, name, **kwargs) -> nn.Module:
        if name.lower() not in cls.MODELS:
            raise ValueError(f"Unknown architecture '{name}'. Available: {cls.available()}")
        return cls.MODELS[name.lower()](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls.MODELS.keys())

# ======== from notebook cell 7 ========
class DatasetRegistry:
    """name -> (train_loader, test_loader, in_channels, num_classes)."""

    @staticmethod
    def _loaders(train, test, batch_size, generator=None, worker_init_fn=None):
        pin = torch.cuda.is_available()
        return (
            DataLoader(
                train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=pin,
                generator=generator,          # controls shufﬂe order if shuffle=True
                worker_init_fn=worker_init_fn # seeds each worker's RNG
            ),
            DataLoader(
                test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=pin,
                worker_init_fn=worker_init_fn # still useful if test uses any random op
            )
        )

    @classmethod
    def synthetic(cls, batch_size=128, data_dir='./data', n_train=2048, n_test=512,
                  in_channels=3, img_size=32, num_classes=10, **kwargs):
        """Random data — for smoke tests only, no download needed."""
        kwargs.pop('augmentation', None)   # accepted for API symmetry, unused
        kwargs.pop('aug_kwargs', None)
        g = torch.Generator().manual_seed(0)
        xtr = torch.randn(n_train, in_channels, img_size, img_size, generator=g)
        ytr = torch.randint(0, num_classes, (n_train,), generator=g)
        xte = torch.randn(n_test, in_channels, img_size, img_size, generator=g)
        yte = torch.randint(0, num_classes, (n_test,), generator=g)
        return (*cls._loaders(TensorDataset(xtr, ytr), TensorDataset(xte, yte),
                              batch_size, **kwargs),
                in_channels, num_classes)

    @classmethod
    def cifar10(cls, batch_size=128, data_dir='./data', augmentation='default',
                aug_kwargs=None, **kwargs):
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        aug_kwargs = aug_kwargs or {}

        # Build training transform from the list
        train_tf_list = cls._build_train_transform(augmentation, aug_kwargs, img_size=32)
        train_tf_list += [transforms.ToTensor(), norm]
        tf_tr = transforms.Compose(train_tf_list)

        tf_te = transforms.Compose([transforms.ToTensor(), norm])
        tr = torchvision.datasets.CIFAR10(data_dir, True, download=True, transform=tf_tr)
        te = torchvision.datasets.CIFAR10(data_dir, False, transform=tf_te)
        return (*cls._loaders(tr, te, batch_size, **kwargs), 3, 10)

    @classmethod
    def cifar100(cls, batch_size=128, data_dir='./data', augmentation='default',
                 aug_kwargs=None, **kwargs):
        norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        aug_kwargs = aug_kwargs or {}

        train_tf_list = cls._build_train_transform(augmentation, aug_kwargs, img_size=32)
        train_tf_list += [transforms.ToTensor(), norm]
        tf_tr = transforms.Compose(train_tf_list)

        tf_te = transforms.Compose([transforms.ToTensor(), norm])
        # Data may be a locally converted copy (e.g. from ModelScope parquets):
        # content-correct but with different md5s than the canonical archive.
        import torchvision.datasets.cifar as _tv_cifar
        _ci = torchvision.datasets.CIFAR100._check_integrity
        _ck = _tv_cifar.check_integrity
        torchvision.datasets.CIFAR100._check_integrity = lambda self: True
        _tv_cifar.check_integrity = lambda *a, **k: True
        try:
            tr = torchvision.datasets.CIFAR100(data_dir, True, download=False, transform=tf_tr)
            te = torchvision.datasets.CIFAR100(data_dir, False, transform=tf_te)
        finally:
            torchvision.datasets.CIFAR100._check_integrity = _ci
            _tv_cifar.check_integrity = _ck
        return (*cls._loaders(tr, te, batch_size, **kwargs), 3, 100)

    @classmethod
    def mnist(cls, batch_size=128, data_dir='./data', augmentation='default',
              aug_kwargs=None, **kwargs):
        norm = transforms.Normalize((0.1307,), (0.3081,))
        aug_kwargs = aug_kwargs or {}

        # MNIST is 28x28, padding 2 instead of 4 for crop
        train_tf_list = []
        if augmentation == 'none':
            pass
        elif augmentation == 'default':
            train_tf_list.extend([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
            ])
        elif augmentation == 'randaugment':
            import timm
            train_tf_list.extend([
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                timm.data.RandAugment(**aug_kwargs),
            ])
        else:
            raise ValueError(f"Unknown augmentation '{augmentation}'")
        train_tf_list += [transforms.ToTensor(), norm]
        tf_tr = transforms.Compose(train_tf_list)

        tf_te = transforms.Compose([transforms.ToTensor(), norm])
        tr = torchvision.datasets.MNIST(data_dir, True, download=True, transform=tf_tr)
        te = torchvision.datasets.MNIST(data_dir, False, transform=tf_te)
        return (*cls._loaders(tr, te, batch_size, **kwargs), 1, 10)

    @classmethod
    def fashion_mnist(cls, batch_size=128, data_dir='./data', augmentation='default',
                      aug_kwargs=None, **kwargs):
        norm = transforms.Normalize((0.2860,), (0.3530,))
        aug_kwargs = aug_kwargs or {}

        train_tf_list = []
        if augmentation == 'none':
            pass
        elif augmentation == 'default':
            train_tf_list.extend([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(),
            ])
        elif augmentation == 'randaugment':
            import timm
            train_tf_list.extend([
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                timm.data.RandAugment(**aug_kwargs),
            ])
        else:
            raise ValueError(f"Unknown augmentation '{augmentation}'")
        train_tf_list += [transforms.ToTensor(), norm]
        tf_tr = transforms.Compose(train_tf_list)

        tf_te = transforms.Compose([transforms.ToTensor(), norm])
        tr = torchvision.datasets.FashionMNIST(data_dir, True, download=True, transform=tf_tr)
        te = torchvision.datasets.FashionMNIST(data_dir, False, transform=tf_te)
        return (*cls._loaders(tr, te, batch_size, **kwargs), 1, 10)

    @classmethod
    def get(cls, name, batch_size=128, data_dir='./data', augmentation='default',
            aug_kwargs=None, **kwargs):
        """Retrieve DataLoaders. Extra **kwargs are passed to _loaders."""
        fn = getattr(cls, name.lower().replace('-', '_'), None)
        if fn is None:
            raise ValueError(f"Unknown dataset '{name}'. Available: {cls.available()}")
        return fn(batch_size=batch_size, data_dir=data_dir,
                  augmentation=augmentation, aug_kwargs=aug_kwargs or {},
                  **kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'synthetic']

    @classmethod
    def _build_train_transform(cls, aug_type: str, aug_kwargs: dict, img_size: int = 32):
        """Return a list of torchvision transforms for the training set.
        Handles different timm.RandAugment signatures."""
        transforms_list = []
        aug_kwargs = aug_kwargs or {}

        if aug_type == 'none':
            return transforms_list

        if aug_type == 'default':
            transforms_list.extend([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

        elif aug_type == 'randaugment':
            import timm
            transforms_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])

            magnitude = aug_kwargs.get('magnitude', 9)
            num_layers = aug_kwargs.get('num_ops', aug_kwargs.get('num_layers', 2))

            # Try each signature in order of likelihood
            try:
                # Newest timm (>=0.6.0): accepts num_ops + magnitude
                aug = timm.data.RandAugment(num_ops=num_layers, magnitude=magnitude)
            except TypeError:
                try:
                    # Some mid versions: accepts num_layers + magnitude
                    aug = timm.data.RandAugment(num_layers=num_layers, magnitude=magnitude)
                except TypeError:
                    # Older versions: requires explicit ops list (e.g. your version)
                    from timm.data.auto_augment import rand_augment_ops
                    ops = rand_augment_ops(magnitude=magnitude)
                    aug = timm.data.RandAugment(ops, num_layers=num_layers)
            transforms_list.append(aug)

        elif aug_type == 'trivial_augment':
            import timm
            transforms_list.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
            try:
                aug = timm.data.TrivialAugmentWide(**aug_kwargs)
            except TypeError:
                aug = timm.data.TrivialAugmentWide()
            transforms_list.append(aug)

        else:
            raise ValueError(f"Unknown augmentation type '{aug_type}'. "
                             f"Available: 'none', 'default', 'randaugment', 'trivial_augment'.")
        return transforms_list

# ======== from notebook cell 9 ========
@dataclass
class ExpConfig:
    name: str

    # --- what to run ---
    activation: Any = 'pdf'  # registry name ('gelu', 'pdfv4', ...), per-site
        # spec (name, kwargs), or an ActivationPlan for mixed networks —
        # e.g. ActivationPlan('gelu', [('last 30%', ('pdfv4', {'init': 'gelu'}))])
    activation_kwargs: dict = field(default_factory=dict)  # e.g. {'init': 'gelu'}
        # (ignored when `activation` is an ActivationPlan: each plan rule
        #  carries its own kwargs)
    arch: str = 'seres'
    arch_kwargs: dict = field(default_factory=dict)
    dataset: str = 'cifar10'
    data_dir: str = './data'
    eval_metric: str = 'acc'  # 'acc' (classification) or 'mse' (regression)
    act_trust_region: float = 0.0  # per-epoch travel cap on act params (0 = off)

    # --- optimization: network weights ---
    epochs: int = 150
    batch_size: int = 512
    optimizer: str = 'adam'
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_frac: float = 0.03        # linear warmup for weight LR
    lr_schedule: str = 'cosine'      # 'cosine' | 'constant'
    grad_clip: float = 0.0           # 0 = off

    oc_pct_start: float = 0.2        # fraction of the run ramping up to peak
    oc_div_factor: float = 25.0      # initial LR = peak / div_factor
    oc_final_div: float = 1e5        # final LR = peak / (div_factor * final_div)

    # --- optimization: activation shape params ---
    act_lr: Optional[float] = None   # absolute LR; None -> lr * act_lr_mult
    act_lr_mult: float = 10.0
    act_weight_decay: float = 0.0    # keep 0: decay biases the shape search
    act_decay_until: float = 1.0     # act LR cosines to 0 at this fraction
    freeze_act_completely: bool = False  # ablation: act fixed at init all run

    # --- freezing / alternation ---
    freeze_schedule: str = ''        # e.g. '0-15,105-'  (see markdown above)
    alternation: Optional[Tuple[int, int]] = None  # (theta_epochs, act_epochs)
    alt_act_only: bool = False       # during act window, also freeze weights
    act_warmup_epochs: int = 0       # ramp act LR 0->full over N epochs after
                                     # each unfreeze (0 = cold start at full LR)
    commit_best_shape: bool = False  # at the start of the final frozen block,
                                     # restore the shape that gave best test acc

    # --- parameter sharing between activation modules ---
    share: str = 'none'              # 'none' | 'block' | 'global'

    # --- anchor regularization: keep act(x) near a reference, decaying ---
    anchor: Optional[str] = None     # e.g. 'gelu', 'silu', 'elu', 'relu'
    anchor_lambda: float = 0.0       # initial strength
    anchor_until: float = 0.7        # lambda cosines to 0 at this fraction
    probe_range: Tuple[float, float] = (-6.0, 6.0)
    probe_points: int = 64

    # --- EMA of activation params (0 = off; eval uses EMA weights) ---
    act_ema: float = 0.0

    # --- bi-level (DARTS-style) shape updates ---
    bilevel: bool = False   # shape params update on held-out VAL loss,
                            # weights on train loss; two separate optimizers
    val_frac: float = 0.1   # fraction of the train set held out for shape updates

    # --- search-then-retrain ---
    act_init_params: Optional[list] = None  # load these values into every
        # trainable act module after model build (e.g. a discovered shape)

    # --- bookkeeping ---
    seed: int = 42
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)
    # events: 'fit_start', 'epoch_start', 'epoch_end', 'fit_end'
    # signature: fn(trainer, epoch)  (epoch=None for fit_start/fit_end)
    verbose: bool = True

    full_reproducibility: bool = False

    # --- NEW: Data Augmentation & Mixup/Cutmix ---
    augmentation: str = 'default'            # 'default' | 'randaugment' | 'none'
    augmentation_kwargs: dict = field(default_factory=dict)  # e.g. {'num_ops': 2, 'magnitude': 9}
    mixup_alpha: float = 0.0                 # 0 = off (typical 0.2)
    cutmix_alpha: float = 0.0                # 0 = off (typical 1.0)
    label_smoothing: float = 0.0             # 0 = off (typical 0.1)

def parse_freeze_spec(spec: str, epochs: int) -> set:
    """'0-15,120-' -> set of frozen epoch indices (0-based, inclusive)."""
    frozen = set()
    spec = (spec or '').strip()
    if not spec:
        return frozen
    if spec == 'all':
        return set(range(epochs))
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            lo = int(lo)
            hi = int(hi) if hi else epochs - 1
            frozen.update(range(lo, min(hi, epochs - 1) + 1))
        else:
            frozen.add(int(part))
    return frozen


def compute_frozen_epochs(cfg: ExpConfig) -> set:
    if cfg.freeze_act_completely:
        return set(range(cfg.epochs))
    if cfg.alternation is not None:
        t, a = cfg.alternation
        cycle = t + a
        return {e for e in range(cfg.epochs) if e % cycle < t}
    return parse_freeze_spec(cfg.freeze_schedule, cfg.epochs)


def cosine_warmup_lr(base_lr: float, epoch: int, epochs: int,
                     warmup_frac: float = 0.03, decay_until: float = 1.0,
                     schedule: str = 'cosine') -> float:
    """Epoch-level LR: linear warmup, then cosine to 0.

    The cosine is anchored so it COMPLETES on the last epoch of the decay
    window (epoch index `horizon - 1`), i.e. the near-zero tail of the
    cosine is actually trained on. The previous version divided by
    `horizon - warm`, so the final epoch only reached t ≈ 0.993 and the
    run ended just before the tail.
    """
    warm = max(1, int(warmup_frac * epochs))
    if epoch < warm:
        return base_lr * (epoch + 1) / warm
    if schedule == 'constant':
        return base_lr
    horizon = max(warm + 1, int(decay_until * epochs))  # one past the last decay epoch
    last = horizon - 1
    t = min(1.0, (epoch - warm) / max(1, last - warm))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def one_cycle_lr(base_lr: float, epoch: int, epochs: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div: float = 1e4) -> float:
    """1cycle policy (Smith & Topin, 2018): linear ramp from
    base_lr/div_factor up to base_lr over the first pct_start fraction of
    the run, then cosine anneal down to base_lr/(div_factor*final_div),
    reached exactly on the final epoch. Enables "super-convergence": short
    budgets (e.g. 30-50 epochs) with a high peak LR, typically with SGD.
    """
    up = max(1, int(pct_start * epochs))
    lo, hi = base_lr / div_factor, base_lr
    end = lo / final_div
    if epoch < up:
        return lo + (hi - lo) * (epoch + 1) / up
    t = min(1.0, (epoch - up) / max(1, epochs - 1 - up))
    return end + (hi - end) * 0.5 * (1.0 + math.cos(math.pi * t))

# ======== from notebook cell 11 ========
def iter_trainable_acts(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, TRAINABLE_ACT_CLASSES):
            yield name, m


def split_param_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Dedup-safe split into (activation params, all other params)."""
    act_ids = {id(p) for _, m in iter_trainable_acts(model) for p in m.parameters()}
    act, other, seen = [], [], set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (act if id(p) in act_ids else other).append(p)
    return act, other


# --------------------------------------------------------------- probe rule
# The study's only hyperparameter rule.  An activation parameter group descends
# about `rate` distance units per epoch, and `rate` scales linearly with the
# multiplier on its learning rate, so the multiplier that spends a fixed budget
# `target` over the remaining epochs is target / (rate * remaining).  `rate` is
# read off the first epoch of the run itself, which is trained at unit
# multiplier (activation lr == weight lr), so the probe costs nothing extra.
PROBE_TARGET = 0.7            # distance units to spend over the whole run
PROBE_CLIP = (0.01, 10.0)     # keep a degenerate probe from exploding/vanishing


def act_displacement(init_params: Sequence[torch.Tensor],
                     probe_params: Sequence[torch.Tensor]) -> float:
    """Mean per-site L2 distance the activation parameters travelled.

    `init_params` and `probe_params` are parallel sequences, one tensor per
    activation site.  A shared activation (`share='global'`/`'block'`) passes a
    single-element sequence, so the result is that tensor's distance.  With
    `share='none'` the mean over sites is used rather than the sum, which keeps
    the number in one site's units and independent of network depth.
    """
    if len(init_params) != len(probe_params) or not init_params:
        raise ValueError('init_params and probe_params must be non-empty and parallel')
    dists = []
    for a, b in zip(init_params, probe_params):
        a = torch.as_tensor(a).detach().to(torch.float64).reshape(-1)
        b = torch.as_tensor(b).detach().to(torch.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f'parameter shape changed across the probe: {a.shape} -> {b.shape}')
        dists.append(float(torch.linalg.vector_norm(b - a)))
    return sum(dists) / len(dists)


def probe_multiplier(init_params: Sequence[torch.Tensor],
                     probe_params: Sequence[torch.Tensor],
                     epochs: int, remaining: Optional[int] = None,
                     target: float = PROBE_TARGET,
                     clip: Tuple[float, float] = PROBE_CLIP) -> float:
    """Activation lr multiplier, as a multiple of the weight lr.

    `init_params` is the activation state before training and `probe_params` the
    state after one epoch trained at unit multiplier.  `remaining` is the number
    of epochs the multiplier will actually govern; it defaults to `epochs - 1`
    (the in-run probe spends epoch 1 measuring).  Pass `epochs` instead when the
    probe was a separate run and every epoch of the real run is driven by the
    result.  A zero displacement means the elapsed epoch produced no signal, so
    the multiplier is pinned to the top of `clip`.
    """
    if epochs < 2:
        raise ValueError(f'epochs must be >= 2, got {epochs}')
    if remaining is None:
        remaining = epochs - 1
    if remaining < 1:
        raise ValueError(f'remaining must be >= 1, got {remaining}')
    lo, hi = clip
    if not 0.0 < lo <= hi:
        raise ValueError(f'clip must satisfy 0 < lo <= hi, got {clip}')
    d1 = act_displacement(init_params, probe_params)
    if d1 <= 0.0:
        return float(hi)
    return float(min(hi, max(lo, target / (d1 * remaining))))


BLOCK_TYPES = ('ConvBlock', 'ResidualBlock', 'SEBasicBlock', 'SEBlock',
               'TransformerEncoder', 'ConvNeXtBlock')


def share_activation_params(model: nn.Module, mode: str = 'none', warm=True):
    """Share the `params` tensor across trainable activation modules.

    'global' — one shared activation for the whole net.
    'block'  — shared within each structural block type (ConvBlock, ...).
    """
    if mode == 'none':
        return
    acts = [(n, m) for n, m in iter_trainable_acts(model) if hasattr(m, 'params')]
    for act in acts:
        if not hasattr(act, 'warmup'):
            continue
        act.warmup()
    if not acts:
        return
    if mode == 'global':
        # group by class: param shapes differ between activation types, so
        # sharing is only ever meaningful (or even valid) within one class
        groups = {}
        for _, m in acts:
            groups.setdefault(f'global:{type(m).__name__}', []).append(m)
    elif mode == 'block':
        parent = {model: None}
        for _, mod in model.named_modules():
            for child in mod.children():
                parent[child] = mod
        groups: Dict[str, list] = {}
        for name, m in acts:
            anc, key = parent.get(m), 'default'
            while anc is not None:
                if type(anc).__name__ in BLOCK_TYPES:
                    key = type(anc).__name__
                    break
                anc = parent.get(anc)
            # also split by class — mixed-activation nets (ActivationPlan)
            # must not share tensors across different activation types
            groups.setdefault(f'{key}:{type(m).__name__}', []).append(m)
    else:
        raise ValueError(f"Unknown share mode '{mode}'")
    for key, mods in groups.items():
        for m in mods[1:]:
            m.params = mods[0].params
        # distributional activations: share one noise sampler so each forward
        # makes a single batched draw (n rows = n sites)
        if hasattr(mods[0], '_sampler'):
            for m in mods[1:]:
                m._sampler = mods[0]._sampler
            mods[0]._sampler.n = len(mods)
        if len(mods) > 1:
            print(f"  [share:{key}] {len(mods)} activation modules now share one parameter set")


ANCHOR_REFS: Dict[str, Callable] = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
    'elu': F.elu,
    'tanh': torch.tanh,
    'mish': F.mish,
    'softplus': F.softplus,
    'leaky_relu': F.leaky_relu,
}


class Trainer:
    def __init__(self, model: nn.Module, cfg: ExpConfig, device: torch.device, num_classes: int):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        if getattr(cfg, 'eval_metric', 'acc') == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing
                                                 if cfg.mixup_alpha == 0 and cfg.cutmix_alpha == 0
                                                 else 0.0)
        self.frozen_epochs = compute_frozen_epochs(cfg)

        share_activation_params(self.model, cfg.share)
        self.act_params, self.weight_params = split_param_groups(self.model)

        # --- MixUp / CutMix ---
        self.mixup_fn = None
        if cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0:
            try:
                import timm
                self.mixup_fn = timm.data.Mixup(
                    mixup_alpha=cfg.mixup_alpha,
                    cutmix_alpha=cfg.cutmix_alpha,
                    prob=1.0,
                    switch_prob=0.5,
                    label_smoothing=cfg.label_smoothing,
                    num_classes=num_classes,
                )
            except ImportError:
                print("Warning: timm not installed. Mixup/CutMix disabled.")
                self.mixup_fn = None

        base_act_lr = cfg.act_lr if cfg.act_lr is not None else cfg.lr * cfg.act_lr_mult
        self.base_act_lr = base_act_lr
        groups = [
            {'params': self.weight_params, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
            {'params': self.act_params, 'lr': base_act_lr, 'weight_decay': cfg.act_weight_decay},
        ]
        self.optimizer = self._build_optimizer(groups)
        self.act_group_idx = 1 if self.act_params else None
        # bi-level mode: shape params get their own optimizer, stepped on val loss
        self.act_optimizer = (optim.Adam(self.act_params, lr=base_act_lr)
                              if self.act_params else None)

        self.ema_shadow: Optional[List[torch.Tensor]] = None
        if cfg.act_ema > 0 and self.act_params:
            self.ema_shadow = [p.detach().clone() for p in self.act_params]

        lo, hi = cfg.probe_range
        self.probe = torch.linspace(lo, hi, cfg.probe_points, device=device)
        self.anchor_fn = ANCHOR_REFS.get(cfg.anchor) if cfg.anchor else None

        self.history: List[Dict[str, Any]] = []

    def _build_optimizer(self, groups) -> optim.Optimizer:
        groups = [g for g in groups if len(g['params']) > 0]
        name = self.cfg.optimizer.lower()
        if name == 'sgd':
            return optim.SGD(groups, momentum=0.9, nesterov=True)
        if name == 'adam':
            return optim.Adam(groups)
        if name == 'adamw':
            return optim.AdamW(groups)
        if name == 'rmsprop':
            return optim.RMSprop(groups)
        raise ValueError(f"Unknown optimizer '{self.cfg.optimizer}'")

    # ----- scheduling (all epoch-level, transparent, easy to extend) -----

    def _epoch_setup(self, epoch: int) -> Dict[str, Any]:
        cfg = self.cfg
        act_frozen = epoch in self.frozen_epochs
        # alternation with alt_act_only: during act windows, freeze the network
        weights_frozen = False
        if cfg.alternation is not None and cfg.alt_act_only and not act_frozen:
            weights_frozen = True

        for p in self.act_params:
            p.requires_grad_(not act_frozen)
        for p in self.weight_params:
            p.requires_grad_(not weights_frozen)

        if cfg.lr_schedule == 'one_cycle':
            # one-cycle drives both groups with the same shape; freezing and
            # act_warmup_epochs still apply on top. act_decay_until is
            # cosine-specific and ignored here.
            lr_w = one_cycle_lr(cfg.lr, epoch, cfg.epochs,
                                cfg.oc_pct_start, cfg.oc_div_factor,
                                cfg.oc_final_div)
            lr_a = 0.0 if act_frozen else one_cycle_lr(
                self.base_act_lr, epoch, cfg.epochs,
                cfg.oc_pct_start, cfg.oc_div_factor, cfg.oc_final_div)
        else:
            lr_w = cosine_warmup_lr(cfg.lr, epoch, cfg.epochs,
                                    cfg.warmup_frac, 1.0, cfg.lr_schedule)
            lr_a = 0.0 if act_frozen else cosine_warmup_lr(
                self.base_act_lr, epoch, cfg.epochs,
                cfg.warmup_frac, cfg.act_decay_until, 'cosine')

        # soft re-entry: ramp act LR after each frozen->trainable transition
        if not act_frozen and cfg.act_warmup_epochs > 0:
            since = self._epochs_since_unfreeze(epoch)
            lr_a *= min(1.0, (since + 1) / cfg.act_warmup_epochs)

        self.optimizer.param_groups[0]['lr'] = lr_w
        if self.act_group_idx is not None:
            self.optimizer.param_groups[self.act_group_idx]['lr'] = lr_a
        if self.act_optimizer is not None:
            self.act_optimizer.param_groups[0]['lr'] = lr_a

        lam = 0.0
        if self.anchor_fn is not None and cfg.anchor_lambda > 0 and not act_frozen:
            t = min(1.0, epoch / max(1, int(cfg.anchor_until * cfg.epochs)))
            lam = cfg.anchor_lambda * 0.5 * (1.0 + math.cos(math.pi * t))

        return {'act_frozen': act_frozen, 'weights_frozen': weights_frozen,
                'lr_w': lr_w, 'lr_a': lr_a, 'anchor_lambda': lam}

    # ----- pieces -----

    def _anchor_penalty(self, lam: float) -> torch.Tensor:
        pen = torch.zeros((), device=self.device)
        if lam <= 0 or self.anchor_fn is None:
            return pen
        seen = set()
        ref = self.anchor_fn(self.probe)
        for _, m in iter_trainable_acts(self.model):
            if id(m.params) in seen:
                continue  # shared modules count once
            seen.add(id(m.params))
            pen = pen + F.mse_loss(m(self.probe), ref)
        return lam * pen

    def _ema_update(self):
        if self.ema_shadow is None:
            return
        d = self.cfg.act_ema
        with torch.no_grad():
            for s, p in zip(self.ema_shadow, self.act_params):
                if p.requires_grad:
                    s.mul_(d).add_(p.detach(), alpha=1 - d)

    def _eval_with_ema(self, loader) -> Tuple[float, float]:
        if self.ema_shadow is None:
            return self.evaluate(loader)
        backup = [p.detach().clone() for p in self.act_params]
        with torch.no_grad():
            for p, s in zip(self.act_params, self.ema_shadow):
                p.copy_(s)
        out = self.evaluate(loader)
        with torch.no_grad():
            for p, b in zip(self.act_params, backup):
                p.copy_(b)
        return out

    def _act_param_snapshot(self) -> Dict[str, Any]:
        snap = {}
        for name, m in iter_trainable_acts(self.model):
            snap[name] = [round(float(v), 6) for v in m.params.detach().cpu().flatten()]
        return snap

    def _epochs_since_unfreeze(self, epoch: int) -> int:
        """0 on the first trainable epoch of the current streak, then +1/epoch."""
        e = epoch
        while e > 0 and (e - 1) not in self.frozen_epochs:
            e -= 1
        return epoch - e

    def _settle_start(self) -> Optional[int]:
        """First epoch of the final contiguous frozen block (None if there is
        no trailing settle phase)."""
        if not self.frozen_epochs or max(self.frozen_epochs) != self.cfg.epochs - 1:
            return None
        start = max(self.frozen_epochs)
        while start - 1 in self.frozen_epochs:
            start -= 1
        return start if start > 0 else None

    # ----- loops -----

    def train_epoch(self, loader: DataLoader, state: Dict[str, Any],
                    val_loader: Optional[DataLoader] = None) -> float:
        import itertools
        self.model.train()
        total = 0.0
        use_bi = (self.cfg.bilevel and val_loader is not None
                  and self.act_params and not state['act_frozen'])
        val_iter = itertools.cycle(val_loader) if use_bi else None

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            # --- Apply MixUp/CutMix to TRAIN batch only ---
            mixup_active = self.mixup_fn is not None
            if mixup_active:
                x, y = self.mixup_fn(x, y)

            if use_bi:
                # step 1: weights only, on TRAIN loss
                for p in self.act_params:
                    p.requires_grad_(False)
                self.optimizer.zero_grad()

                logits = self.model(x)
                if mixup_active:
                    # soft targets: cross-entropy = - sum(y * log_softmax)
                    loss = (- (y * F.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
                else:
                    loss = self.criterion(logits, y)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.weight_params, self.cfg.grad_clip)
                self.optimizer.step()
                self._apply_trust_region()

                # step 2: shape only, on VAL loss (clean, no mixup)
                for p in self.act_params:
                    p.requires_grad_(True)
                vx, vy = next(val_iter)
                vx, vy = vx.to(self.device), vy.to(self.device)
                self.act_optimizer.zero_grad()
                vlogits = self.model(vx)
                vloss = (self.criterion(vlogits, vy)
                         + self._anchor_penalty(state['anchor_lambda']))
                vloss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.act_params, self.cfg.grad_clip)
                self.act_optimizer.step()
                self._apply_trust_region()
                self._ema_update()
            else:
                self.optimizer.zero_grad()
                logits = self.model(x)
                if mixup_active:
                    loss = (- (y * F.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
                else:
                    loss = self.criterion(logits, y)
                loss = loss + self._anchor_penalty(state['anchor_lambda'])
                loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                self._ema_update()

            total += loss.item() * x.size(0)
        return total / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total, correct, n = 0.0, 0, 0
        regression = getattr(self.cfg, 'eval_metric', 'acc') == 'mse'
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            total += self.criterion(out, y).item() * x.size(0)
            if regression:
                correct += ((out.reshape(-1) - y.reshape(-1)) ** 2).sum().item()
            else:
                correct += out.argmax(1).eq(y).sum().item()
            n += y.size(0)
        if regression:
            return total / n, correct / n   # (loss, MSE) - lower is better
        return total / n, 100.0 * correct / n

    def _emit(self, event: str, epoch: Optional[int]):
        for fn in self.cfg.hooks.get(event, []):
            fn(self, epoch)

    def fit(self, train_loader: DataLoader, test_loader: DataLoader,
            val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        cfg = self.cfg
        self._tr_cap_step = (getattr(cfg, 'act_trust_region', 0.0)
                             / max(len(train_loader), 1))
        self._tr_prev = None
        self.val_loader = val_loader   # exposed for hooks (e.g. val-based selection)
        self._emit('fit_start', None)
        best_acc = 0.0

        # commit_best_shape machinery: snapshot the shape at best test acc
        # during the search phase; restore it when the settle phase begins.
        settle = self._settle_start() if cfg.commit_best_shape else None
        if cfg.commit_best_shape and settle is None:
            print("  [best-shape] no trailing frozen block found; option ignored")
        best_shape = {'acc': -1.0, 'snap': None}

        for epoch in range(cfg.epochs):
            state = self._epoch_setup(epoch)

            if settle is not None and epoch == settle and best_shape['snap'] is not None:
                with torch.no_grad():
                    for p, s in zip(self.act_params, best_shape['snap']):
                        p.copy_(s)
                    if self.ema_shadow is not None:
                        for sh, s in zip(self.ema_shadow, best_shape['snap']):
                            sh.copy_(s)
                print(f"  [best-shape] committed to shape from best acc "
                      f"{best_shape['acc']:.2f}% at settle (epoch {settle})")

            self._emit('epoch_start', epoch)
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, state, val_loader)
            test_loss, test_acc = self._eval_with_ema(test_loader)
            dt = time.time() - t0
            best_acc = max(best_acc, test_acc)

            # snapshot the evaluated shape (EMA shadow if EMA is on, else live)
            if settle is not None and epoch < settle and test_acc > best_shape['acc']:
                src = (self.ema_shadow if self.ema_shadow is not None
                       else [p.detach() for p in self.act_params])
                best_shape = {'acc': test_acc,
                              'snap': [t.detach().clone() for t in src]}

            rec = {'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss,
                   'test_acc': test_acc, 'time': dt, **state,
                   'act_params': self._act_param_snapshot()}
            self.history.append(rec)
            self._emit('epoch_end', epoch)

            if cfg.verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
                flag = 'F' if state['act_frozen'] else ' '
                print(f"  [{epoch+1:>4}/{cfg.epochs}]{flag} "
                      f"train {train_loss:.4f} | test {test_loss:.4f} | "
                      f"acc {test_acc:5.2f}% | lr_w {state['lr_w']:.2e} | "
                      f"lr_a {state['lr_a']:.2e} | {dt:.1f}s")

        self._emit('fit_end', None)
        return {'best_test_acc': best_acc,
                'final_test_acc': self.history[-1]['test_acc'],
                'final_train_loss': self.history[-1]['train_loss'],
                'final_test_loss': self.history[-1]['test_loss'],
                'history': self.history}

# ======== from notebook cell 13 ========
def unique_act_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """One representative per distinct parameter set."""
    out, seen = [], set()
    for name, m in iter_trainable_acts(model):
        if id(m.params) not in seen:
            seen.add(id(m.params))
            out.append((name, m))
    return out


@torch.no_grad()
def plot_activations(model: nn.Module, path: str, title: str = '',
                     x_range=(-6, 6), max_modules: int = 9,
                     refs=('gelu', 'silu', 'relu')):
    """Plot learned activation curves (value + derivative) vs references."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mods = unique_act_modules(model)[:max_modules]
    if not mods:
        return
    device = next(model.parameters()).device
    x = torch.linspace(*x_range, 400, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in mods:
        y = m(x)
        # derivative via finite differences (works for every form)
        dy = torch.gradient(y, spacing=(x[1] - x[0]).item())[0]
        label = name if len(mods) > 1 else type(m).__name__
        axes[0].plot(x.cpu(), y.cpu(), label=label, lw=1.6)
        axes[1].plot(x.cpu(), dy.cpu(), label=label, lw=1.6)
    for r in refs:
        axes[0].plot(x.cpu(), ANCHOR_REFS[r](x).cpu(), '--', lw=1.2, alpha=0.7, label=r)
    axes[0].set_title(f'activation — {title}')
    axes[1].set_title('derivative')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(history: List[Dict[str, Any]], path: str, title: str = ''):
    """Loss/accuracy curves + LR trajectories + frozen-phase shading."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ep, [h['train_loss'] for h in history], label='train')
    axes[0].plot(ep, [h['test_loss'] for h in history], label='test')
    axes[0].set_title('loss')
    axes[1].plot(ep, [h['test_acc'] for h in history], color='green')
    axes[1].set_title('test acc (%)')
    axes[2].plot(ep, [h['lr_w'] for h in history], label='lr weights')
    axes[2].plot(ep, [h['lr_a'] for h in history], label='lr act')
    axes[2].set_title('learning rates')
    axes[2].set_yscale('symlog', linthresh=1e-6)
    for ax in axes:
        for h in history:
            if h['act_frozen']:
                ax.axvspan(h['epoch'], h['epoch'] + 1, color='red', alpha=0.06, lw=0)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_param_trajectories(history: List[Dict[str, Any]], path: str, title: str = ''):
    """One subplot per unique activation module: param values over epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    modules = sorted({k for h in history for k in h['act_params']})
    if not modules:
        return
    show = modules[:9]
    fig, axes = plt.subplots(len(show), 1, figsize=(10, 2.4 * len(show)), sharex=True)
    if len(show) == 1:
        axes = [axes]
    ep = [h['epoch'] for h in history]
    for ax, mod in zip(axes, show):
        series = [h['act_params'][mod] for h in history if mod in h['act_params']]
        n = len(series[0])
        for i in range(n):
            ax.plot(ep[:len(series)], [s[i] for s in series], lw=1.0, label=f'p{i}')
        for h in history:
            if h['act_frozen']:
                ax.axvspan(h['epoch'], h['epoch'] + 1, color='red', alpha=0.06, lw=0)
        ax.set_ylabel(mod.split('.')[-2] if '.' in mod else mod, fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5, ncol=6, loc='upper right')
    axes[-1].set_xlabel('epoch')
    fig.suptitle(f'activation parameter trajectories — {title}')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

# ======== from notebook cell 15 ========
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_full_reproducibility(seed_value=42):
    """
    Set all random number generator seeds and enable PyTorch deterministic
    algorithms.  This covers Python's random, NumPy, PyTorch (CPU/GPU),
    CuDNN, and CUDA workspace config.
    """
    print(f"  Setting global random seed to: {seed_value}")

    # 1. Python built-in random
    random.seed(seed_value)

    # 2. NumPy
    np.random.seed(seed_value)

    # 3. PyTorch CPU / GPU
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # 4. Deterministic PyTorch ops (raises error if non‑deterministic op is used)
    torch.use_deterministic_algorithms(True)

    # 5. CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. CUDA workspace config (important for CUDA >= 10.2)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print("  Deterministic settings applied.")


# ----------------------------------------------------------------------
#  Experiment runner (fully integrated with reproducibility)
# ----------------------------------------------------------------------

def run_experiment(cfg: ExpConfig, out_dir: str = 'runs',
                   device: Optional[str] = None) -> Dict[str, Any]:
    """Run one experiment, save artifacts, return a summary dict."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # -------- Reproducibility setup --------
    full_repro = getattr(cfg, 'full_reproducibility', False) or getattr(cfg, 'full_reproducability', False)

    if full_repro:
        set_full_reproducibility(cfg.seed)

        # Worker initialisation: each worker gets a deterministic seed derived
        # from the global seed + its worker id
        def worker_init_fn(worker_id):
            worker_seed = (cfg.seed + worker_id) % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        # Fixed generator to control DataLoader shuffling order
        shuffle_generator = torch.Generator().manual_seed(cfg.seed)
    else:
        set_seed(cfg.seed)
        worker_init_fn = None
        shuffle_generator = None

    run_dir = Path(out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------- Verbose preamble --------
    if cfg.verbose:
        frozen = sorted(compute_frozen_epochs(cfg))
        preview = f"{frozen[:8]}{'...' if len(frozen) > 8 else ''}" if frozen else 'none'
        print(f"\n{'='*70}\n{cfg.name}\n{'='*70}")
        print(f"  {cfg.activation}({cfg.activation_kwargs}) | {cfg.arch} | {cfg.dataset} | "
              f"{cfg.optimizer} lr={cfg.lr} | act_lr={cfg.act_lr or cfg.lr*cfg.act_lr_mult:.2e}")
        print(f"  frozen epochs: {preview} | share={cfg.share} | "
              f"anchor={cfg.anchor}@{cfg.anchor_lambda} | ema={cfg.act_ema}")

    # -------- Load base dataset (DataLoader gets reproducibility args) --------
    train_loader, test_loader, in_ch, n_cls = DatasetRegistry.get(
        cfg.dataset, cfg.batch_size, cfg.data_dir,
        augmentation=cfg.augmentation,
        aug_kwargs=cfg.augmentation_kwargs,
        worker_init_fn=worker_init_fn,
        generator=shuffle_generator
    )

    # -------- Bilevel split (overwrites loaders, uses same reproducibility) --------
    val_loader = None
    if cfg.bilevel:
        ds = train_loader.dataset
        n_val = max(1, int(cfg.val_frac * len(ds)))
        split_g = torch.Generator().manual_seed(cfg.seed)
        tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val], generator=split_g)
        pin = torch.cuda.is_available()

        # Recreate train/val loaders with full reproducibility args
        train_loader = DataLoader(
            tr_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=2, pin_memory=pin,
            worker_init_fn=worker_init_fn,
            generator=shuffle_generator if shuffle_generator is not None else None
        )
        val_loader = DataLoader(
            va_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=2, pin_memory=pin,
            worker_init_fn=worker_init_fn
        )

        if cfg.verbose:
            print(f"  [bilevel] train {len(tr_ds)} / val-for-shape {len(va_ds)}")

    # -------- Build model --------
    act_arg = cfg.activation
    if isinstance(act_arg, ActivationPlan):
        # measure the architecture's site count (RNG-forked dry run), then
        # resolve fraction selectors ('last 30%' etc.) against it
        n_sites = act_arg.count_sites(cfg.arch, num_classes=n_cls,
                                      in_channels=in_ch, **cfg.arch_kwargs)
        act_arg.resolve(n_sites)
        if cfg.verbose:
            print(f"  [plan] {n_sites} activation sites: {act_arg.describe()}")
    model = ArchitectureRegistry.get(
        cfg.arch, num_classes=n_cls, activation=act_arg,
        in_channels=in_ch, act_kwargs=cfg.activation_kwargs, **cfg.arch_kwargs)

    # -------- Initialise activation parameters if requested --------
    if cfg.act_init_params is not None:
        vals = torch.tensor(cfg.act_init_params, dtype=torch.float32)
        n_set = 0
        for _, m in iter_trainable_acts(model):
            if m.params.numel() == vals.numel():
                with torch.no_grad():
                    m.params.copy_(vals.to(m.params.device))
                n_set += 1
        print(f"  [init] loaded act_init_params into {n_set} modules")

    # -------- Train --------
    trainer = Trainer(model, cfg, device, n_cls)
    metrics = trainer.fit(train_loader, test_loader, val_loader)

    # -------- Summary --------
    summary = {
        'name': cfg.name,
        'activation': str(act_arg),
        'arch': cfg.arch,
        'dataset': cfg.dataset,
        'optimizer': cfg.optimizer,
        'epochs': cfg.epochs,
        'total_params': sum(p.numel() for p in model.parameters()),
        'act_params': sum(p.numel() for p in trainer.act_params),
        'best_test_acc': metrics['best_test_acc'],
        'final_test_acc': metrics['final_test_acc'],
        'final_train_loss': metrics['final_train_loss'],
        'final_test_loss': metrics['final_test_loss'],
        'device': str(device),
        'seed': cfg.seed,
    }

    # -------- Save artifacts --------
    (run_dir / 'config.json').write_text(
        json.dumps({k: v for k, v in asdict(cfg).items() if k != 'hooks'}, indent=2, default=str))
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (run_dir / 'history.json').write_text(json.dumps(metrics['history'], default=str))
    torch.save(model.state_dict(), run_dir / 'model.pt')   # for post-hoc sweeps
    plot_history(metrics['history'], str(run_dir / 'training_curves.png'), cfg.name)
    plot_param_trajectories(metrics['history'], str(run_dir / 'param_trajectories.png'), cfg.name)
    if isinstance(model, nn.Module) and any(True for _ in iter_trainable_acts(model)):
        plot_activations(model, str(run_dir / 'learned_activation.png'), cfg.name)
    print(f"  -> artifacts in {run_dir}/")

    # -------- Cleanup --------
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_experiments(configs: Sequence[ExpConfig], out_dir: str = 'runs',
                    device: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run a sequence of experiments and write a combined results.csv."""
    summaries = []
    for i, cfg in enumerate(configs):
        print(f"\n########## experiment {i+1}/{len(configs)} ##########")
        summaries.append(run_experiment(cfg, out_dir, device))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = summaries[0].keys()
    with open(out / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summaries)

    print(f"\n{'='*90}\nSUMMARY (sorted by best acc)\n{'='*90}")
    print(f"{'name':<28} {'act':<10} {'arch':<12} {'data':<10} {'best':>7} {'final':>7}")
    for s in sorted(summaries, key=lambda r: -r['best_test_acc']):
        print(f"{s['name']:<28} {s['activation']:<10} {s['arch']:<12} {s['dataset']:<10} "
              f"{s['best_test_acc']:>6.2f}% {s['final_test_acc']:>6.2f}%")
    print(f"\nresults.csv -> {out / 'results.csv'}")
    return summaries

# ======== from notebook cell 18 ========
E = 150  # total epochs (used to express phase boundaries)


def three_phase_freeze(epochs: int, warm_frac: float = 0.10,
                       settle_frac: float = 0.70) -> str:
    """Freeze-schedule string for the three-phase skeleton at a given budget."""
    return f'0-{int(warm_frac * epochs)},{int(settle_frac * epochs)}-'


def _act_presets(act: str) -> Dict[str, ExpConfig]:
    """The standard ablation chain for one trainable activation."""
    return {
        # parameterization control: shape frozen at its default init
        f'{act}_frozen': ExpConfig(
            name=f'{act}_frozen', activation=act,
            epochs=E, freeze_act_completely=True),

        # naive co-training, no schedule
        f'{act}_joint': ExpConfig(
            name=f'{act}_joint', activation=act,
            epochs=E, act_lr_mult=1),

        f'{act}_joint_shared': ExpConfig(
            name=f'{act}_joint', activation=act, share='global',
            epochs=E, act_lr_mult=1),

        # three-phase skeleton: frozen warmup -> search (act LR cosines to 0)
        # -> frozen settle
        f'{act}_three_phase': ExpConfig(
            name=f'{act}_three_phase', activation=act,
            epochs=E, act_lr_mult=1, act_decay_until=0.7,
            freeze_schedule=three_phase_freeze(E)),

        # val-driven (DARTS-style) shared shape + anchor tether
        f'{act}_bl_shared': ExpConfig(
            name=f'{act}_bl_shared', activation=act,
            epochs=E, act_lr_mult=1, bilevel=True, share='global',
            val_frac=0.05,
            anchor='gelu', anchor_lambda=0.1, anchor_until=1.0),
    }


PRESETS: Dict[str, ExpConfig] = {
    # reference point: fixed, hand-designed GELU — the number to beat
    'gelu_fixed': ExpConfig(name='gelu_fixed', activation='gelu', epochs=E),
    **_act_presets('pdf'),
    **_act_presets('pdfv3'),
    **_act_presets('pdfv4'),
    # ---- one-cycle on ResNet-18 / CIFAR-10 ----
    # CIFAR stem (3x3 s1, no maxpool), standard SGD wd=5e-4, 40 epochs.
    'r18_gelu_onecycle': ExpConfig(
        name='r18_gelu_onecycle', activation='gelu',
        arch='resnet18', dataset='cifar10',
        epochs=40, optimizer='sgd', lr=0.05, weight_decay=5e-4,
        lr_schedule='one_cycle'),

    'r18_pdf_onecycle': ExpConfig(
        name='r18_pdf_onecycle', activation='pdf',
        arch='resnet18', dataset='cifar10',
        epochs=40, optimizer='sgd', lr=0.05, weight_decay=5e-4,
        lr_schedule='one_cycle', act_lr_mult=1),

    'r18_pdfv3_onecycle': ExpConfig(
        name='r18_pdfv2_onecycle', activation='pdfv3',
        arch='resnet18', dataset='cifar10',
        epochs=40, optimizer='sgd', lr=0.05, weight_decay=5e-4,
        lr_schedule='one_cycle', act_lr_mult=1),
}


def frozen_retrain(discovered_params: list, epochs: int = 150,
                   activation: str = 'pdf',
                   name: str = 'shape_retrain') -> ExpConfig:
    """Search-then-retrain: take a discovered shape (read from
    runs/<name>/history.json) and retrain from scratch with the shape FROZEN
    on 100% of the training data.

    Usage:
        import json
        h = json.load(open('runs/pdf_bl_shared/history.json'))
        shape = next(iter(h[-1]['act_params'].values()))  # one shared module
        run_experiments([frozen_retrain(shape, epochs=150, activation='pdf')])
    """
    return ExpConfig(
        name=name, activation=activation,
        epochs=epochs, freeze_act_completely=True,
        act_init_params=list(discovered_params),
    )


def replicate(names: Sequence[str], seeds: Sequence[int] = (1, 2, 3),
              **overrides) -> List[ExpConfig]:
    """Seed-replicate configs: replicate(['gelu_fixed','pdf_bl_shared']) ->
    gelu_fixed_s1, gelu_fixed_s2, ... A 0.2-0.3% gap on CIFAR-10 is within
    seed noise; never conclude from seed=42 alone."""
    import dataclasses
    out = []
    for n in names:
        base = pick(n, **overrides)[0]
        for s in seeds:
            out.append(dataclasses.replace(base, seed=s, name=f'{n}_s{s}'))
    return out


def pick(*names: str, **overrides) -> List[ExpConfig]:
    """Grab presets by name; shared overrides apply to all (e.g. dataset='mnist')."""
    import copy
    out = []
    for n in names:
        cfg = copy.deepcopy(PRESETS[n])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        out.append(cfg)
    return out

# ======== from notebook cell 19 ========
def _rescale_freeze(spec: str, old_epochs: int, new_epochs: int) -> str:
    """Rebake a freeze-schedule string when the epoch budget changes,
    preserving phase fractions. '0-15,105-' @150 -> '0-5,35-' @50."""
    if not spec or spec == 'all' or old_epochs == new_epochs:
        return spec
    frozen = sorted(parse_freeze_spec(spec, old_epochs))
    if not frozen:
        return spec
    blocks, start, prev = [], frozen[0], frozen[0]
    for e in frozen[1:]:
        if e == prev + 1:
            prev = e
        else:
            blocks.append((start, prev))
            start = prev = e
    blocks.append((start, prev))
    parts = []
    for lo, hi in blocks:
        nlo = round(lo * new_epochs / old_epochs)
        if hi >= old_epochs - 1:          # open-ended block stays open-ended
            parts.append(f'{nlo}-')
        else:
            nhi = round(hi * new_epochs / old_epochs)
            parts.append(f'{nlo}-{nhi}')
    return ','.join(parts)


def build_configs(
    preset: list[str] | None = None,
    epochs: int | None = None,
    dataset: str | None = None,
    arch: str | None = None,
    smoke: bool = False,
    overrides: dict | None = None,
    preset_overrides: dict[str, dict] | None = None,
    seeds: list[int] | None = None,
    name_suffix: str = '',
) -> list[ExpConfig]:
    """Build experiment configs from presets / overrides (no I/O).

    preset           list of preset names (default: all PRESETS)
    epochs/dataset/arch  global overrides; epochs also RESCALES freeze_schedule
                         and act_warmup_epochs so phase fractions are preserved
    overrides        {field: value} applied to every config, e.g.
                     {'bilevel': True} or {'act_lr_mult': 3}
    preset_overrides {preset_name: {field: value}} applied per preset, before
                     global overrides (global wins on conflict)
    seeds            replicate each config once per seed, named '<name>_s<seed>'
    name_suffix      appended to every config name (auto-generated from
                     `overrides` if omitted and overrides are present)
    """
    import dataclasses

    if smoke:
        return [ExpConfig(
            name='smoke', activation='pdf',
            arch='simple_cnn', dataset='synthetic', epochs=4, batch_size=128,
            act_lr_mult=10, freeze_schedule='0-1', anchor='gelu',
            anchor_lambda=0.05, act_ema=0.9, share='block',
        )]

    valid_fields = {f.name for f in dataclasses.fields(ExpConfig)}

    def _apply(c: ExpConfig, ov: dict, where: str):
        for k, v in ov.items():
            if k not in valid_fields:
                raise ValueError(f"Unknown ExpConfig field '{k}' in {where}. "
                                 f"Valid: {sorted(valid_fields)}")
            setattr(c, k, v)

    names = preset or list(PRESETS)
    configs = pick(*names)
    overrides = overrides or {}
    preset_overrides = preset_overrides or {}

    for name, c in zip(names, configs):
        if name in preset_overrides:
            _apply(c, preset_overrides[name], f"preset_overrides['{name}']")

        old_epochs = c.epochs
        if epochs is not None:
            c.epochs = epochs
            if c.freeze_schedule and c.alternation is None \
                    and not c.freeze_act_completely:
                c.freeze_schedule = _rescale_freeze(
                    c.freeze_schedule, old_epochs, epochs)
            if c.act_warmup_epochs > 0:
                c.act_warmup_epochs = max(1, round(
                    c.act_warmup_epochs * epochs / old_epochs))
        if dataset is not None:
            c.dataset = dataset
        if arch is not None:
            c.arch = arch

        _apply(c, overrides, 'overrides')

        suffix = name_suffix
        if not suffix and overrides:
            def _fmt(v):
                return 'on' if v is True else 'off' if v is False else str(v)
            suffix = '_' + '_'.join(f'{k}{_fmt(v)}' for k, v in overrides.items())
        c.name = c.name + suffix

    if seeds:
        out = []
        for c in configs:
            for s in seeds:
                out.append(dataclasses.replace(c, seed=s, name=f'{c.name}_s{s}'))
        configs = out

    return configs


def run_from_args(
    preset: list[str] | None = None,
    epochs: int | None = None,
    dataset: str | None = None,
    out: str = 'runs',
    smoke: bool = False,
) -> None:
    """Entry point you can call from a notebook cell or CLI wrapper."""
    configs = build_configs(preset=preset, epochs=epochs, dataset=dataset, smoke=smoke)
    run_experiments(configs, out_dir=out)

# ======== from notebook cell 20 ========
def recover_summaries(root_dir, output_csv = None):
    """
    Recursively find all 'summary.json' files under 'root_dir', load them,
    and return a list of experiment summaries.

    Args:
        root_dir: Path to the directory containing experiment subdirectories.
        output_csv: If provided, write the combined summaries to this CSV file.

    Returns:
        List of dictionaries, each corresponding to one experiment summary.
    """
    root = Path(root_dir)
    summaries = []

    for summary_path in root.glob('**/summary.json'):
        with open(summary_path, 'r') as f:
            data = json.load(f)
            # Optionally attach the source directory name for context
            data['_source_dir'] = summary_path.parent.name
            summaries.append(data)

    if output_csv:
        if summaries:
            keys = summaries[0].keys()
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(summaries)
            print(f"Combined summary written to {output_csv}")
        else:
            print("No summary.json files found.")

    return summaries

    
def write_summaries(
    summaries,
    output_file,
    format = 'csv'
) -> None:
    """
    Write a list of experiment summary dictionaries to a file.

    Args:
        summaries: List of summary dictionaries (as returned by recover_summaries).
        output_file: Path to the output file (e.g., 'results.csv' or 'results.json').
        format: Output format, either 'csv' or 'json'. Defaults to 'csv'.
    """
    if not summaries:
        print("No summaries to write.")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f"Written {len(summaries)} summaries to {output_path} (JSON)")

    elif format.lower() == 'csv':
        # Collect all keys from all summaries to ensure consistent columns
        all_keys = set()
        for s in summaries:
            all_keys.update(s.keys())
        # Sort keys for readability
        fieldnames = sorted(all_keys)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Written {len(summaries)} summaries to {output_path} (CSV)")

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")