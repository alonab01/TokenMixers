"""Observer classes — collect tensor statistics, compute scale/zero_point, fake-quantize.

Each observer is an nn.Module with three modes:
  DISABLED     — passthrough (no quantization applied)
  CALIBRATING  — update running stats in forward, return x unchanged
  FROZEN       — apply fake_quantize using stored scale/zero_point

Weights: call .observe(weight_tensor) then .freeze() once at QuantConv2d.__init__.
Activations: set mode=CALIBRATING, run N batches, set mode=FROZEN.

Phase 2 extensibility: subclass BaseObserver and register via @register_observer("name").
No pipeline code needs to change.
"""
from __future__ import annotations

from typing import Dict, Type

import torch
from torch import Tensor, nn

from quantization.fake_quant import (
    compute_scale_symmetric,
    compute_scale_zp_asymmetric,
    fake_quantize_asymmetric,
    fake_quantize_symmetric,
    qrange_asymmetric,
    qrange_symmetric,
)

DISABLED = 0
CALIBRATING = 1
FROZEN = 2

SYMMETRIC = "symmetric"
ASYMMETRIC = "asymmetric"


OBSERVER_REGISTRY: Dict[str, Type["BaseObserver"]] = {}


def register_observer(name: str):
    def deco(cls):
        if name in OBSERVER_REGISTRY:
            raise ValueError(f"Observer '{name}' already registered")
        OBSERVER_REGISTRY[name] = cls
        return cls
    return deco


def build_observer(name: str, bits: int, scheme: str) -> "BaseObserver":
    if name not in OBSERVER_REGISTRY:
        raise KeyError(
            f"Unknown observer '{name}'. Available: {list(OBSERVER_REGISTRY)}"
        )
    return OBSERVER_REGISTRY[name](bits=bits, scheme=scheme)


class BaseObserver(nn.Module):
    """Shared state machine + Q/DQ dispatch. Subclasses implement observe() and freeze()."""

    def __init__(self, bits: int, scheme: str):
        super().__init__()
        if scheme not in (SYMMETRIC, ASYMMETRIC):
            raise ValueError(f"scheme must be 'symmetric' or 'asymmetric', got '{scheme}'")
        self.bits = bits
        self.scheme = scheme

        if scheme == SYMMETRIC:
            qmin, qmax = qrange_symmetric(bits)
        else:
            qmin, qmax = qrange_asymmetric(bits)
        self.qmin = qmin
        self.qmax = qmax

        # scalar buffers — override shape in per-channel subclasses later
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("mode", torch.tensor(DISABLED, dtype=torch.int64))

    # ---- state transitions ---- #

    def set_mode(self, mode: int) -> None:
        assert mode in (DISABLED, CALIBRATING, FROZEN)
        if mode == CALIBRATING:
            self.reset()
        self.mode.fill_(mode)

    def reset(self) -> None:
        """Subclass: clear accumulated statistics."""
        raise NotImplementedError

    def observe(self, x: Tensor) -> None:
        """Subclass: update running statistics from x."""
        raise NotImplementedError

    def freeze(self) -> None:
        """Subclass: compute self.scale / self.zero_point from stats, then mode = FROZEN."""
        raise NotImplementedError

    # ---- forward ---- #

    def fake_quantize(self, x: Tensor) -> Tensor:
        if self.scheme == SYMMETRIC:
            return fake_quantize_symmetric(x, self.scale, self.qmin, self.qmax)
        return fake_quantize_asymmetric(x, self.scale, self.zero_point, self.qmin, self.qmax)

    def forward(self, x: Tensor) -> Tensor:
        m = int(self.mode.item())
        if m == DISABLED:
            return x
        if m == CALIBRATING:
            self.observe(x)
            return x
        return self.fake_quantize(x)


@register_observer("percentile")
class PercentileObserver(BaseObserver):
    """Uses low/high quantiles (default 0.001 / 0.999) instead of raw min/max.

    Rationale: a single 10-sigma outlier activation inflates the min/max range so much
    that the per-tensor 8-bit grid wastes most of its resolution on rarely-seen values,
    collapsing typical activations into 1-2 bins. Percentile clipping drops the top/bottom
    0.1% → scale reflects the range of "normal" values.

    Memory-efficient: keeps per-batch quantiles and averages them across batches.
    Not as accurate as a true global histogram but drastically better than min/max for
    activations with heavy tails.
    """

    def __init__(
        self,
        bits: int,
        scheme: str,
        low_percentile: float = 0.001,
        high_percentile: float = 0.999,
    ):
        super().__init__(bits=bits, scheme=scheme)
        if not (0.0 <= low_percentile < high_percentile <= 1.0):
            raise ValueError(
                f"Require 0 <= low < high <= 1, got ({low_percentile}, {high_percentile})"
            )
        self.low_p = low_percentile
        self.high_p = high_percentile
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("n_batches", torch.tensor(0, dtype=torch.int64))

    def reset(self) -> None:
        self.min_val.fill_(0.0)
        self.max_val.fill_(0.0)
        self.n_batches.zero_()

    def observe(self, x: Tensor) -> None:
        x = x.detach().flatten()
        # quantile on large tensors is expensive; subsample if > 1M elements
        if x.numel() > 1_000_000:
            idx = torch.randint(0, x.numel(), (1_000_000,), device=x.device)
            x = x[idx]
        low = torch.quantile(x, self.low_p)
        high = torch.quantile(x, self.high_p)
        n = self.n_batches.item()
        # running mean update
        self.min_val.copy_((self.min_val * n + low) / (n + 1))
        self.max_val.copy_((self.max_val * n + high) / (n + 1))
        self.n_batches.add_(1)

    def freeze(self) -> None:
        if self.n_batches.item() == 0:
            raise RuntimeError(
                "Cannot freeze observer: no data observed (n_batches=0)."
            )
        if self.scheme == SYMMETRIC:
            max_abs = torch.maximum(self.min_val.abs(), self.max_val.abs())
            self.scale.copy_(compute_scale_symmetric(max_abs, self.qmax))
            self.zero_point.zero_()
        else:
            scale, zp = compute_scale_zp_asymmetric(
                self.min_val, self.max_val, self.qmin, self.qmax
            )
            self.scale.copy_(scale)
            self.zero_point.copy_(zp)
        self.mode.fill_(FROZEN)


@register_observer("min_max")
class MinMaxObserver(BaseObserver):
    """Running min/max across all observed tensors (per-tensor granularity)."""

    def __init__(self, bits: int, scheme: str):
        super().__init__(bits=bits, scheme=scheme)
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def reset(self) -> None:
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))

    def observe(self, x: Tensor) -> None:
        x = x.detach()
        cur_min = x.min()
        cur_max = x.max()
        self.min_val.copy_(torch.minimum(self.min_val, cur_min))
        self.max_val.copy_(torch.maximum(self.max_val, cur_max))

    def freeze(self) -> None:
        if not torch.isfinite(self.min_val) or not torch.isfinite(self.max_val):
            raise RuntimeError(
                "Cannot freeze observer: no data observed (min/max still at inf)."
            )
        if self.scheme == SYMMETRIC:
            max_abs = torch.maximum(self.min_val.abs(), self.max_val.abs())
            self.scale.copy_(compute_scale_symmetric(max_abs, self.qmax))
            self.zero_point.zero_()
        else:
            scale, zp = compute_scale_zp_asymmetric(
                self.min_val, self.max_val, self.qmin, self.qmax
            )
            self.scale.copy_(scale)
            self.zero_point.copy_(zp)
        self.mode.fill_(FROZEN)


@register_observer("per_channel_min_max")
class PerChannelMinMaxObserver(BaseObserver):
    """Per-output-channel min/max for weight tensors.

    Shape contract: observed tensor is a Conv2d weight of shape (out_c, in_c, kH, kW).
    scale / zero_point / min_val / max_val all have shape (out_c,).
    During fake_quantize, we reshape to (out_c, 1, 1, 1) for broadcasting.

    Fixes the depthwise-conv cliff: per-tensor max(|W|) is dominated by a few
    channels with large magnitudes, crushing the small-magnitude channels.
    Per-channel gives each output channel its own scale.

    Designed for WEIGHTS only — activations have a different axis convention
    (channel is axis=1) and would need a separate axis parameter.
    """

    def __init__(self, bits: int, scheme: str, axis: int = 0):
        super().__init__(bits=bits, scheme=scheme)
        self.axis = axis
        # Buffers are lazily shaped on first observe(); start as scalar placeholders
        # to support state_dict load-time reshape.
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def _reshape_for_broadcast(self, per_channel: Tensor, ref: Tensor) -> Tensor:
        shape = [1] * ref.dim()
        shape[self.axis] = -1
        return per_channel.view(shape)

    def _collapse_to_channel(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Reduce all dims except `axis` to get per-channel min/max.
        dims = [d for d in range(x.dim()) if d != self.axis]
        if not dims:
            return x, x
        cur_min = x.amin(dim=dims)
        cur_max = x.amax(dim=dims)
        return cur_min, cur_max

    def reset(self) -> None:
        # Will be reshaped on first observe
        self.min_val = torch.tensor(float("inf"), device=self.min_val.device)
        self.max_val = torch.tensor(float("-inf"), device=self.max_val.device)

    def observe(self, x: Tensor) -> None:
        x = x.detach()
        n_channels = x.shape[self.axis]
        # lazy (re)allocate per-channel buffers on first call
        if self.min_val.numel() != n_channels:
            self.min_val = torch.full((n_channels,), float("inf"), device=x.device, dtype=x.dtype)
            self.max_val = torch.full((n_channels,), float("-inf"), device=x.device, dtype=x.dtype)
        cur_min, cur_max = self._collapse_to_channel(x)
        self.min_val.copy_(torch.minimum(self.min_val, cur_min))
        self.max_val.copy_(torch.maximum(self.max_val, cur_max))

    def freeze(self) -> None:
        if self.min_val.numel() == 0 or not torch.isfinite(self.min_val).all():
            raise RuntimeError(
                "Cannot freeze observer: no data observed (min/max still at inf)."
            )
        n = self.min_val.numel()
        if self.scheme == SYMMETRIC:
            max_abs = torch.maximum(self.min_val.abs(), self.max_val.abs())
            self.scale = compute_scale_symmetric(max_abs, self.qmax)
            self.zero_point = torch.zeros(n, dtype=torch.int64, device=self.min_val.device)
        else:
            scale, zp = compute_scale_zp_asymmetric(
                self.min_val, self.max_val, self.qmin, self.qmax
            )
            self.scale = scale
            self.zero_point = zp
        self.mode.fill_(FROZEN)

    def fake_quantize(self, x: Tensor) -> Tensor:
        scale_b = self._reshape_for_broadcast(self.scale, x)
        if self.scheme == SYMMETRIC:
            return fake_quantize_symmetric(x, scale_b, self.qmin, self.qmax)
        zp_b = self._reshape_for_broadcast(self.zero_point, x)
        return fake_quantize_asymmetric(x, scale_b, zp_b, self.qmin, self.qmax)
