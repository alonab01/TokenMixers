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


def build_observer(
    name: str, bits: int, scheme: str, axis: int | None = None
) -> "BaseObserver":
    """Construct an observer by registry name.

    `axis` is forwarded to per-channel observers (those whose __init__ accepts an
    `axis` kwarg). For per-tensor observers the kwarg is silently ignored.
    Use axis=0 for weight tensors, axis=1 for Conv2d activations (NCHW), and
    axis=-1 for Linear activations (last feature dim).
    """
    if name not in OBSERVER_REGISTRY:
        raise KeyError(
            f"Unknown observer '{name}'. Available: {list(OBSERVER_REGISTRY)}"
        )
    cls = OBSERVER_REGISTRY[name]
    import inspect
    if axis is not None and "axis" in inspect.signature(cls.__init__).parameters:
        return cls(bits=bits, scheme=scheme, axis=axis)
    return cls(bits=bits, scheme=scheme)


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
        # First batch: seed directly. min_val/max_val start at +/-inf so the
        # running-mean update would produce NaN (inf * 0). This matters when
        # PercentileObserver is used as a WEIGHT observer (single observe call,
        # no preceding reset()).
        if n == 0:
            self.min_val.copy_(low)
            self.max_val.copy_(high)
        else:
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

    def _norm_axis(self, x: Tensor) -> int:
        # Normalize negative axis (e.g. axis=-1 for Linear activations of any rank).
        return self.axis % x.dim()

    def _reshape_for_broadcast(self, per_channel: Tensor, ref: Tensor) -> Tensor:
        shape = [1] * ref.dim()
        shape[self._norm_axis(ref)] = -1
        return per_channel.view(shape)

    def _collapse_to_channel(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Reduce all dims except `axis` to get per-channel min/max.
        a = self._norm_axis(x)
        dims = [d for d in range(x.dim()) if d != a]
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
        n_channels = x.shape[self._norm_axis(x)]
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


# tiny epsilon for per-channel division (matches fake_quant.EPS)
EPS_PC = 1e-8


# ----------------------------------------------------------------------------
# Phase 6 / Phase 9: MSE-based scale search.
#
# Both MSE observers below are now histogram-backed. This makes them work for:
#   - weights (single observe call: a 1-tensor histogram is built and frozen),
#   - activations (multi-batch calibration: streaming histogram accumulates),
# and for both symmetric and asymmetric schemes.
#
# At freeze time we sweep candidate clip ranges and pick the one minimizing
# the count-weighted MSE between bin centers and their fake-quantized values.
# Forwarding declarations (HistogramObserver / PerChannelHistogramObserver)
# are below this block; the @register_observer decorator runs at class-body
# time so we just put the MSE classes after them.
# ----------------------------------------------------------------------------


def _hist_bin_centers(hist_min: float, hist_max: float, n_bins: int, *,
                      device, dtype=torch.float64) -> Tensor:
    edges = torch.linspace(hist_min, hist_max, n_bins + 1, device=device, dtype=dtype)
    return 0.5 * (edges[:-1] + edges[1:])


def _hist_bin_centers_per_channel(hist_min: Tensor, hist_max: Tensor,
                                  n_bins: int) -> Tensor:
    """Returns (C, n_bins) bin centers where each channel has its own [lo, hi]."""
    device = hist_min.device
    idx = torch.arange(n_bins, device=device, dtype=torch.float64)
    lo = hist_min.to(torch.float64).unsqueeze(1)            # (C, 1)
    hi = hist_max.to(torch.float64).unsqueeze(1)            # (C, 1)
    width = (hi - lo) / n_bins                              # (C, 1)
    return lo + (idx + 0.5).unsqueeze(0) * width            # (C, n_bins)


def _mse_search_symmetric_hist(
    bin_centers_abs: Tensor, counts: Tensor, max_abs: float,
    qmin: int, qmax: int, n_steps: int, p_min: float, p_max: float,
) -> Tensor:
    """Search alpha that minimizes MSE for symmetric quant.

    bin_centers_abs: (n_bins,) non-negative bin centers (|x| histogram).
    counts: (n_bins,) bin counts.
    Returns scalar scale (float64).
    """
    device = bin_centers_abs.device
    if max_abs <= 0 or not torch.isfinite(torch.tensor(max_abs)):
        return torch.tensor(EPS_PC, device=device, dtype=torch.float64)
    init = max_abs / qmax
    alphas = torch.linspace(p_min, p_max, n_steps, device=device, dtype=torch.float64)
    best_scale = torch.tensor(init, device=device, dtype=torch.float64)
    best_err = torch.tensor(float("inf"), device=device, dtype=torch.float64)
    total = counts.sum().clamp(min=1.0)
    for a in alphas:
        s = (init * a).clamp(min=EPS_PC)
        x_q = torch.round(bin_centers_abs / s).clamp(qmin, qmax) * s
        err = (counts * (bin_centers_abs - x_q).pow(2)).sum() / total
        if err < best_err:
            best_err = err
            best_scale = torch.tensor(s.item() if hasattr(s, "item") else s,
                                      device=device, dtype=torch.float64)
    return best_scale


def _mse_search_asymmetric_hist(
    bin_centers: Tensor, counts: Tensor, hist_min: float, hist_max: float,
    qmin: int, qmax: int, n_steps: int, p_min: float, p_max: float,
) -> tuple[Tensor, Tensor]:
    """Search 1-D shrinkage alpha ∈ [p_min, p_max] for asymmetric quant.

    For each alpha, the clip range shrinks symmetrically around the midpoint:
      mid = (hist_min + hist_max) / 2
      half = alpha * (hist_max - hist_min) / 2
      cmin = mid - half;  cmax = mid + half
    Then scale = (cmax - cmin) / (qmax - qmin); zp from cmin.
    Returns (scale, zp) as float64 / int64 scalars.
    """
    device = bin_centers.device
    mid = 0.5 * (hist_min + hist_max)
    halfw = 0.5 * (hist_max - hist_min)
    if halfw <= 0:
        return (torch.tensor(EPS_PC, device=device, dtype=torch.float64),
                torch.tensor(0, device=device, dtype=torch.int64))
    alphas = torch.linspace(p_min, p_max, n_steps, device=device, dtype=torch.float64)
    best_scale = torch.tensor(EPS_PC, device=device, dtype=torch.float64)
    best_zp = torch.tensor(0, device=device, dtype=torch.int64)
    best_err = torch.tensor(float("inf"), device=device, dtype=torch.float64)
    total = counts.sum().clamp(min=1.0)
    for a in alphas:
        cmin_t = torch.tensor(mid - a.item() * halfw, device=device, dtype=torch.float64)
        cmax_t = torch.tensor(mid + a.item() * halfw, device=device, dtype=torch.float64)
        scale, zp = compute_scale_zp_asymmetric(cmin_t, cmax_t, qmin, qmax)
        zp_f = zp.to(torch.float64)
        x_q = (torch.round(bin_centers / scale) + zp_f).clamp(qmin, qmax)
        x_q = (x_q - zp_f) * scale
        err = (counts * (bin_centers - x_q).pow(2)).sum() / total
        if err < best_err:
            best_err = err
            best_scale = scale.to(torch.float64)
            best_zp = zp.to(torch.int64)
    return best_scale, best_zp


def _mse_search_symmetric_hist_per_channel(
    bin_centers_abs: Tensor, counts: Tensor, max_abs: Tensor,
    qmin: int, qmax: int, n_steps: int, p_min: float, p_max: float,
) -> Tensor:
    """Vectorized per-channel symmetric MSE search.

    bin_centers_abs: (C, n_bins). counts: (C, n_bins). max_abs: (C,).
    Returns (C,) scales (float64).
    """
    device = bin_centers_abs.device
    init = (max_abs.to(torch.float64) / qmax).clamp(min=EPS_PC)        # (C,)
    alphas = torch.linspace(p_min, p_max, n_steps, device=device, dtype=torch.float64)
    total = counts.sum(dim=1).clamp(min=1.0)                            # (C,)
    best_scale = init.clone()
    best_err = torch.full_like(init, float("inf"))
    for a in alphas:
        s = (init * a).clamp(min=EPS_PC)                                # (C,)
        s_b = s.unsqueeze(1)                                            # (C, 1)
        x_q = torch.round(bin_centers_abs / s_b).clamp(qmin, qmax) * s_b
        err = (counts * (bin_centers_abs - x_q).pow(2)).sum(dim=1) / total
        better = err < best_err
        best_scale = torch.where(better, s, best_scale)
        best_err = torch.where(better, err, best_err)
    return best_scale


def _mse_search_asymmetric_hist_per_channel(
    bin_centers: Tensor, counts: Tensor, hist_min: Tensor, hist_max: Tensor,
    qmin: int, qmax: int, n_steps: int, p_min: float, p_max: float,
) -> tuple[Tensor, Tensor]:
    """Vectorized per-channel asymmetric MSE search."""
    device = bin_centers.device
    hi = hist_max.to(torch.float64)
    lo = hist_min.to(torch.float64)
    mid = 0.5 * (lo + hi)                                               # (C,)
    halfw = 0.5 * (hi - lo)                                             # (C,)
    alphas = torch.linspace(p_min, p_max, n_steps, device=device, dtype=torch.float64)
    total = counts.sum(dim=1).clamp(min=1.0)                            # (C,)
    best_scale = torch.full_like(mid, EPS_PC).to(torch.float64)
    best_zp = torch.zeros_like(mid, dtype=torch.int64)
    best_err = torch.full_like(mid, float("inf"))
    for a in alphas:
        cmin = mid - a * halfw
        cmax = mid + a * halfw
        scale, zp = compute_scale_zp_asymmetric(cmin, cmax, qmin, qmax)  # (C,), (C,)
        scale64 = scale.to(torch.float64).clamp(min=EPS_PC).unsqueeze(1)  # (C, 1)
        zp_f = zp.to(torch.float64).unsqueeze(1)                          # (C, 1)
        x_q = (torch.round(bin_centers / scale64) + zp_f).clamp(qmin, qmax)
        x_q = (x_q - zp_f) * scale64
        err = (counts * (bin_centers - x_q).pow(2)).sum(dim=1) / total
        better = err < best_err
        best_scale = torch.where(better, scale.to(torch.float64), best_scale)
        # zp is int64; use index assignment via masked_scatter for clarity
        best_zp = torch.where(better, zp.to(torch.int64), best_zp)
        best_err = torch.where(better, err, best_err)
    return best_scale, best_zp

    def fake_quantize(self, x: Tensor) -> Tensor:
        scale_b = self._reshape_for_broadcast(self.scale, x)
        return fake_quantize_symmetric(x, scale_b, self.qmin, self.qmax)


# ----------------------------------------------------------------------------
# Phase 6b: Histogram + KL-divergence observer (TensorRT-style)
# ----------------------------------------------------------------------------


@register_observer("histogram")
class HistogramObserver(BaseObserver):
    """KL-divergence calibration for activations.

    Streaming histogram over calibration batches; at freeze, search over candidate
    truncation thresholds. The threshold minimizing KL(reference || candidate)
    determines the clip range; scale follows.

    Activation-only (asymmetric scheme by default; symmetric also supported by using
    histogram of |x|). Works alongside other PTQ tools — drop-in replacement for
    `min_max` or `percentile` via `--quant.act-observer histogram`.

    Memory: O(n_bins) per observer (~16KB at default 2048 bins).
    """

    def __init__(
        self,
        bits: int,
        scheme: str,
        n_bins: int = 2048,
        symmetric_uses_abs: bool = True,
    ):
        super().__init__(bits=bits, scheme=scheme)
        self.n_bins = n_bins
        self.symmetric_uses_abs = symmetric_uses_abs and (scheme == SYMMETRIC)
        self.register_buffer("hist", torch.zeros(n_bins, dtype=torch.float64))
        self.register_buffer("hist_min", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("hist_max", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("min_val", torch.tensor(0.0))  # diagnostic-only
        self.register_buffer("max_val", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.int64))

    def reset(self) -> None:
        self.hist.zero_()
        self.hist_min.zero_()
        self.hist_max.zero_()
        self.min_val.zero_()
        self.max_val.zero_()
        self.initialized.zero_()

    def observe(self, x: Tensor) -> None:
        x = x.detach().flatten()
        if self.symmetric_uses_abs:
            x = x.abs()

        # Lazy device migration: histogram lives wherever the input lives.
        if self.hist.device != x.device:
            self.hist = self.hist.to(x.device)
            self.hist_min = self.hist_min.to(x.device)
            self.hist_max = self.hist_max.to(x.device)

        # Update diagnostic min/max
        cur_min = x.min().to(torch.float64)
        cur_max = x.max().to(torch.float64)
        self.min_val.copy_(cur_min.float())
        self.max_val.copy_(cur_max.float())

        if self.initialized.item() == 0:
            # First batch: set range from this batch (with small padding)
            lo = cur_min if not self.symmetric_uses_abs else torch.zeros_like(cur_min)
            hi = cur_max
            if hi == lo:
                hi = lo + 1.0  # avoid zero-width
            self.hist_min.copy_(lo)
            self.hist_max.copy_(hi)
            self.initialized.fill_(1)
            h = torch.histc(
                x.to(torch.float64), bins=self.n_bins,
                min=self.hist_min.item(), max=self.hist_max.item(),
            )
            self.hist.copy_(h)
            return

        # Subsequent batches: if range expanded, redistribute existing histogram
        if cur_max > self.hist_max:
            self._expand_range(new_max=cur_max.item(),
                               new_min=self.hist_min.item())
        if not self.symmetric_uses_abs and cur_min < self.hist_min:
            self._expand_range(new_max=self.hist_max.item(),
                               new_min=cur_min.item())

        h = torch.histc(
            x.to(torch.float64), bins=self.n_bins,
            min=self.hist_min.item(), max=self.hist_max.item(),
        )
        self.hist.add_(h)

    def _expand_range(self, new_max: float, new_min: float) -> None:
        """Rebin existing histogram into a wider range. Conservative: every old bin maps
        to the nearest new bin via linear interpolation of bin centers."""
        old_min = self.hist_min.item()
        old_max = self.hist_max.item()
        if new_min >= old_min and new_max <= old_max:
            return
        new_hist = torch.zeros(self.n_bins, dtype=torch.float64, device=self.hist.device)
        old_centers = old_min + (torch.arange(self.n_bins, dtype=torch.float64,
                                              device=self.hist.device) + 0.5) * (old_max - old_min) / self.n_bins
        # Map each old center to its new bin
        new_idx = ((old_centers - new_min) / (new_max - new_min) * self.n_bins).long().clamp(0, self.n_bins - 1)
        new_hist.scatter_add_(0, new_idx, self.hist)
        self.hist.copy_(new_hist)
        self.hist_min.fill_(new_min)
        self.hist_max.fill_(new_max)

    def freeze(self) -> None:
        if self.initialized.item() == 0:
            raise RuntimeError("Cannot freeze HistogramObserver: no data observed.")
        # Run KL search on the accumulated histogram.
        target_levels = self.qmax - self.qmin + 1   # number of representable levels
        threshold_value = _kl_threshold_search(
            self.hist, self.hist_min.item(), self.hist_max.item(),
            target_levels=target_levels,
        )
        if self.scheme == SYMMETRIC:
            # threshold_value is the |x| clip range
            max_abs = torch.tensor(threshold_value, dtype=torch.float32, device=self.scale.device)
            self.scale.copy_(compute_scale_symmetric(max_abs, self.qmax))
            self.zero_point.zero_()
        else:
            # asymmetric: KL search returns the upper clip; lower clip is hist_min.
            # For activations like ReLU outputs, hist_min ~ 0; for general signed activations
            # we'd want a 2-sided KL search. For Phase 6b we ship the simple one-sided path.
            min_val = torch.tensor(float(self.hist_min.item()), dtype=torch.float32,
                                   device=self.scale.device)
            max_val = torch.tensor(float(threshold_value), dtype=torch.float32,
                                   device=self.scale.device)
            scale, zp = compute_scale_zp_asymmetric(min_val, max_val, self.qmin, self.qmax)
            self.scale.copy_(scale)
            self.zero_point.copy_(zp)
        self.mode.fill_(FROZEN)


# ----------------------------------------------------------------------------
# Phase 7: Per-channel variants of percentile + histogram observers
# ----------------------------------------------------------------------------


@register_observer("per_channel_percentile")
class PerChannelPercentileObserver(BaseObserver):
    """Per-channel quantile-based observer (mirror of PercentileObserver).

    For each channel along ``axis``, computes (low, high) quantiles over all other
    dims; running mean across batches. Reduces the per-channel grid waste caused by
    per-channel outliers — the main motivation for trying this on 4-bit weights, where
    per_channel_min_max stretches 15 levels across the full max range.

    Supports both symmetric and asymmetric schemes. Designed for weights (axis=0); the
    per-channel granularity composes cleanly with int conv on the output dim.
    """

    def __init__(
        self,
        bits: int,
        scheme: str,
        axis: int = 0,
        low_percentile: float = 0.001,
        high_percentile: float = 0.999,
    ):
        super().__init__(bits=bits, scheme=scheme)
        if not (0.0 <= low_percentile < high_percentile <= 1.0):
            raise ValueError(
                f"Require 0 <= low < high <= 1, got ({low_percentile}, {high_percentile})"
            )
        self.axis = axis
        self.low_p = low_percentile
        self.high_p = high_percentile
        # Lazy (re)shape on first observe, like PerChannelMinMaxObserver.
        self.register_buffer("min_val", torch.tensor(0.0))
        self.register_buffer("max_val", torch.tensor(0.0))
        self.register_buffer("n_batches", torch.tensor(0, dtype=torch.int64))

    def _norm_axis(self, x: Tensor) -> int:
        return self.axis % x.dim()

    def _reshape_for_broadcast(self, per_channel: Tensor, ref: Tensor) -> Tensor:
        shape = [1] * ref.dim()
        shape[self._norm_axis(ref)] = -1
        return per_channel.view(shape)

    def reset(self) -> None:
        self.min_val = torch.tensor(0.0, device=self.min_val.device)
        self.max_val = torch.tensor(0.0, device=self.max_val.device)
        self.n_batches.zero_()

    def observe(self, x: Tensor) -> None:
        x = x.detach()
        a = self._norm_axis(x)
        n_channels = x.shape[a]
        if self.min_val.numel() != n_channels:
            self.min_val = torch.zeros(n_channels, device=x.device, dtype=x.dtype)
            self.max_val = torch.zeros(n_channels, device=x.device, dtype=x.dtype)

        perm = [a] + [d for d in range(x.dim()) if d != a]
        x_perm = x.permute(perm).contiguous().view(n_channels, -1)
        # subsample very large per-channel tensors (rare for weights)
        if x_perm.shape[1] > 1_000_000:
            idx = torch.randint(0, x_perm.shape[1], (1_000_000,), device=x.device)
            x_perm = x_perm[:, idx]

        low = torch.quantile(x_perm, self.low_p, dim=1)
        high = torch.quantile(x_perm, self.high_p, dim=1)
        n = self.n_batches.item()
        self.min_val.copy_((self.min_val * n + low) / (n + 1))
        self.max_val.copy_((self.max_val * n + high) / (n + 1))
        self.n_batches.add_(1)

    def freeze(self) -> None:
        if self.n_batches.item() == 0:
            raise RuntimeError(
                "Cannot freeze PerChannelPercentileObserver: no data observed."
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


@register_observer("per_channel_histogram")
class PerChannelHistogramObserver(BaseObserver):
    """Per-channel histogram + KL-divergence calibration (mirror of HistogramObserver).

    Each channel along ``axis`` gets its own (n_bins,) streaming histogram and its own
    KL-optimal threshold. The per-j outer loop is shared across channels (vectorized
    via batched scatter_add), so freeze cost stays in seconds even with 1024 channels.

    Memory: O(n_channels * n_bins). For 1024 channels x 2048 bins x 8 bytes ~ 16 MB
    per observer. Designed for weights (axis=0); activation use is possible but has
    the usual per-channel composition issue.
    """

    def __init__(
        self,
        bits: int,
        scheme: str,
        axis: int = 0,
        n_bins: int = 2048,
        symmetric_uses_abs: bool = True,
    ):
        super().__init__(bits=bits, scheme=scheme)
        self.axis = axis
        self.n_bins = n_bins
        self.symmetric_uses_abs = symmetric_uses_abs and (scheme == SYMMETRIC)
        # Lazy 2D allocation on first observe — start as scalar/1D placeholders.
        self.register_buffer("hist", torch.zeros(n_bins, dtype=torch.float64))
        self.register_buffer("hist_min", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("hist_max", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("min_val", torch.tensor(0.0))   # diagnostic-only
        self.register_buffer("max_val", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.int64))

    def _norm_axis(self, x: Tensor) -> int:
        return self.axis % x.dim()

    def _reshape_for_broadcast(self, per_channel: Tensor, ref: Tensor) -> Tensor:
        shape = [1] * ref.dim()
        shape[self._norm_axis(ref)] = -1
        return per_channel.view(shape)

    def reset(self) -> None:
        self.initialized.zero_()
        self.hist = torch.zeros(self.n_bins, dtype=torch.float64, device=self.hist.device)
        self.hist_min = torch.tensor(0.0, dtype=torch.float64, device=self.hist.device)
        self.hist_max = torch.tensor(0.0, dtype=torch.float64, device=self.hist.device)

    def observe(self, x: Tensor) -> None:
        x = x.detach()
        a = self._norm_axis(x)
        n_channels = x.shape[a]
        perm = [a] + [d for d in range(x.dim()) if d != a]
        x_perm = x.permute(perm).contiguous().view(n_channels, -1)
        x_for_hist = x_perm.abs() if self.symmetric_uses_abs else x_perm

        cur_min = x_for_hist.amin(dim=1).to(torch.float64)
        cur_max = x_for_hist.amax(dim=1).to(torch.float64)

        # Diagnostic min/max (signed)
        self.min_val = x_perm.amin(dim=1).to(torch.float32)
        self.max_val = x_perm.amax(dim=1).to(torch.float32)

        if self.initialized.item() == 0:
            self.hist = torch.zeros(
                (n_channels, self.n_bins), dtype=torch.float64, device=x.device
            )
            self.hist_min = (
                torch.zeros_like(cur_min) if self.symmetric_uses_abs else cur_min.clone()
            )
            self.hist_max = cur_max.clone()
            zero_width = self.hist_max == self.hist_min
            self.hist_max = torch.where(zero_width, self.hist_min + 1.0, self.hist_max)
            self.initialized.fill_(1)
        else:
            if self.hist.device != x.device:
                self.hist = self.hist.to(x.device)
                self.hist_min = self.hist_min.to(x.device)
                self.hist_max = self.hist_max.to(x.device)
            need_max = cur_max > self.hist_max
            need_min = (
                (cur_min < self.hist_min)
                if not self.symmetric_uses_abs
                else torch.zeros_like(need_max)
            )
            if need_max.any() or need_min.any():
                new_max = torch.maximum(self.hist_max, cur_max)
                new_min = (
                    torch.minimum(self.hist_min, cur_min)
                    if not self.symmetric_uses_abs
                    else self.hist_min
                )
                self._expand_range_per_channel(new_min, new_max)

        # torch.histc has no batched form — loop. Cheap relative to KL search.
        for c in range(n_channels):
            h = torch.histc(
                x_for_hist[c].to(torch.float64),
                bins=self.n_bins,
                min=self.hist_min[c].item(),
                max=self.hist_max[c].item(),
            )
            self.hist[c].add_(h)

    def _expand_range_per_channel(self, new_min: Tensor, new_max: Tensor) -> None:
        n_channels = self.hist.shape[0]
        new_hist = torch.zeros_like(self.hist)
        n_bins_t = torch.arange(self.n_bins, dtype=torch.float64, device=self.hist.device)
        for c in range(n_channels):
            old_lo = self.hist_min[c].item()
            old_hi = self.hist_max[c].item()
            nlo = new_min[c].item()
            nhi = new_max[c].item()
            if nlo >= old_lo and nhi <= old_hi:
                new_hist[c] = self.hist[c]
                continue
            old_centers = old_lo + (n_bins_t + 0.5) * (old_hi - old_lo) / self.n_bins
            denom = max(nhi - nlo, 1e-12)
            new_idx = (((old_centers - nlo) / denom) * self.n_bins).long().clamp(0, self.n_bins - 1)
            new_hist[c].scatter_add_(0, new_idx, self.hist[c])
        self.hist = new_hist
        self.hist_min = new_min
        self.hist_max = new_max

    def freeze(self) -> None:
        if self.initialized.item() == 0:
            raise RuntimeError(
                "Cannot freeze PerChannelHistogramObserver: no data observed."
            )
        n_channels = self.hist.shape[0]
        target_levels = self.qmax - self.qmin + 1
        thresholds = _kl_threshold_search_per_channel(
            self.hist, self.hist_min, self.hist_max, target_levels
        ).to(torch.float32)

        if self.scheme == SYMMETRIC:
            self.scale = compute_scale_symmetric(thresholds, self.qmax)
            self.zero_point = torch.zeros(
                n_channels, dtype=torch.int64, device=self.hist.device
            )
        else:
            min_val = self.hist_min.to(torch.float32)
            max_val = thresholds
            scale, zp = compute_scale_zp_asymmetric(min_val, max_val, self.qmin, self.qmax)
            self.scale = scale
            self.zero_point = zp
        self.mode.fill_(FROZEN)

    def fake_quantize(self, x: Tensor) -> Tensor:
        scale_b = self._reshape_for_broadcast(self.scale, x)
        if self.scheme == SYMMETRIC:
            return fake_quantize_symmetric(x, scale_b, self.qmin, self.qmax)
        zp_b = self._reshape_for_broadcast(self.zero_point, x)
        return fake_quantize_asymmetric(x, scale_b, zp_b, self.qmin, self.qmax)


def _kl_threshold_search_per_channel(
    hist: Tensor,        # (C, n_bins)
    hist_min: Tensor,    # (C,)
    hist_max: Tensor,    # (C,)
    target_levels: int,
) -> Tensor:
    """Vectorized per-channel KL search. Loops over j only; channel dim is batched.

    For each channel, finds j in [target_levels, n_bins] minimizing KL(P || Q) using
    the same TRT-style fold-in convention as ``_kl_threshold_search``. Returns the
    per-channel upper-edge threshold of shape (C,).
    """
    n_channels, n_bins = hist.shape
    if target_levels >= n_bins:
        return hist_max.clone()
    bin_widths = (hist_max - hist_min) / n_bins                # (C,)
    h = hist.to(torch.float64)
    eps = 1e-12

    tail_sum = h.flip(1).cumsum(1).flip(1)                     # (C, n_bins)

    best_kl = torch.full(
        (n_channels,), float("inf"), dtype=torch.float64, device=h.device
    )
    best_j = torch.full(
        (n_channels,), target_levels, dtype=torch.long, device=h.device
    )

    for j in range(target_levels, n_bins + 1):
        sliced = h[:, :j]                                      # (C, j)
        P = sliced.clone()
        if j < n_bins:
            P[:, -1] = P[:, -1] + tail_sum[:, j]
        P_sum = P.sum(dim=1)                                   # (C,)

        idx = (torch.arange(j, device=h.device, dtype=torch.long) * target_levels) // j
        idx_b = idx.unsqueeze(0).expand(n_channels, j)         # (C, j)
        mask = (sliced > 0).to(torch.float64)
        group_sum = torch.zeros(
            (n_channels, target_levels), dtype=torch.float64, device=h.device
        )
        group_count = torch.zeros_like(group_sum)
        group_sum.scatter_add_(1, idx_b, sliced)
        group_count.scatter_add_(1, idx_b, mask)
        per_group_avg = group_sum / group_count.clamp(min=1.0)
        Q = mask * per_group_avg.gather(1, idx_b)
        Q_sum = Q.sum(dim=1)                                   # (C,)

        valid = (P_sum > 0) & (Q_sum > 0)
        P_norm = P / P_sum.clamp(min=eps).unsqueeze(1)
        Q_norm = Q / Q_sum.clamp(min=eps).unsqueeze(1)
        mask_P = (P > 0).to(torch.float64)
        log_ratio = torch.log((P_norm + eps) / (Q_norm + eps))
        kl = (P_norm * log_ratio * mask_P).sum(dim=1)          # (C,)

        better = valid & (kl < best_kl)
        best_kl = torch.where(better, kl, best_kl)
        j_t = torch.tensor(j, device=h.device, dtype=torch.long)
        best_j = torch.where(better, j_t, best_j)

    thresholds = hist_min + best_j.to(hist_min.dtype) * bin_widths
    return thresholds


def _kl_threshold_search(
    hist: Tensor,
    hist_min: float,
    hist_max: float,
    target_levels: int,
) -> float:
    """TensorRT-style KL-divergence search (vectorized inner Q construction).

    Outer loop: candidate truncation index j in [target_levels, n_bins].
    Inner Q construction is done with scatter_add, eliminating Python iteration over
    target_levels. With 2048 bins x 256 target_levels, the freeze cost drops from
    ~200s to a few seconds on GPU.

    For each j:
      P = hist[:j] with the tail [j:] folded into bin j-1.
      Group bins in [0, j) into target_levels groups by `group_idx = i * target_levels // j`.
      Q[i] = (P>0)*(group_sum / group_active_count)[group_idx[i]].
      KL(P || Q) over P>0 mass.
    Pick the j minimizing KL; return the corresponding upper edge value.
    """
    n_bins = hist.numel()
    if target_levels >= n_bins:
        return hist_max
    bin_width = (hist_max - hist_min) / n_bins
    h = hist.to(torch.float64)
    eps = 1e-12

    # Reverse cumsum: tail_sum[j] = sum(h[j:]). Used for the fold-in step.
    tail_sum = h.flip(0).cumsum(0).flip(0)

    best_kl = float("inf")
    best_j = target_levels
    for j in range(target_levels, n_bins + 1):
        # SLICE = h[:j] (the un-folded histogram — used to build Q).
        sliced = h[:j]
        # P = SLICE with the right tail folded into bin j-1 (used as the reference).
        P = sliced.clone()
        if j < n_bins:
            P[-1] = P[-1] + tail_sum[j]
        P_sum = P.sum()
        if P_sum.item() <= 0:
            continue

        # Q is built from SLICE (no fold-in). Group bins into target_levels bins,
        # average within each group over bins where SLICE>0, expand back to length j.
        # Critically: this means when SLICE has a thin tail and P has a fat fold-in
        # spike, Q at index j-1 is small but P at index j-1 is large -> KL > 0.
        idx = (torch.arange(j, device=h.device, dtype=torch.long) * target_levels) // j
        mask = (sliced > 0).to(torch.float64)
        group_sum = torch.zeros(target_levels, dtype=torch.float64, device=h.device)
        group_count = torch.zeros(target_levels, dtype=torch.float64, device=h.device)
        group_sum.scatter_add_(0, idx, sliced)
        group_count.scatter_add_(0, idx, mask)
        per_group_avg = group_sum / group_count.clamp(min=1.0)
        Q = mask * per_group_avg[idx]
        Q_sum = Q.sum()
        if Q_sum.item() <= 0:
            continue

        P_norm = P / P_sum
        Q_norm = Q / Q_sum
        # KL(P || Q) — use a P>0 mask (fold-in spike makes P[-1] non-zero even if SLICE[-1]==0)
        mask_P = (P > 0).to(torch.float64)
        log_ratio = torch.log((P_norm + eps) / (Q_norm + eps))
        kl = (P_norm * log_ratio * mask_P).sum().item()

        if kl < best_kl:
            best_kl = kl
            best_j = j

    return hist_min + best_j * bin_width


# ----------------------------------------------------------------------------
# Phase 9 (refactor): MSE observers — histogram-backed.
#
# Inherit accumulation from the histogram observers (streaming hist over batches),
# then override freeze() to do an MSE search instead of a KL search. This makes
# MSE work for both weights (single observe call) and activations (multi-batch),
# in both symmetric and asymmetric schemes.
# ----------------------------------------------------------------------------


@register_observer("mse")
class MSEObserver(HistogramObserver):
    """Per-tensor MSE-optimal scale, histogram-backed.

    Search: alpha ∈ [p_min, p_max] candidates; symmetric sweeps the |x| clip
    threshold, asymmetric shrinks [hist_min, hist_max] symmetrically around
    the midpoint. Pick the candidate minimizing count-weighted MSE between
    bin centers and their fake-quantized values.
    """

    def __init__(
        self,
        bits: int,
        scheme: str,
        n_bins: int = 2048,
        n_steps: int = 80,
        p_min: float = 0.5,
        p_max: float = 1.2,
    ):
        super().__init__(bits=bits, scheme=scheme, n_bins=n_bins)
        self.n_steps = n_steps
        self.p_min = p_min
        self.p_max = p_max

    def freeze(self) -> None:
        if self.initialized.item() == 0:
            raise RuntimeError("Cannot freeze MSEObserver: no data observed.")
        device = self.scale.device
        bin_centers = _hist_bin_centers(
            self.hist_min.item(), self.hist_max.item(), self.n_bins,
            device=self.hist.device,
        )
        if self.scheme == SYMMETRIC:
            # symmetric_uses_abs is True, so bin_centers are |x| and hist_max
            # is the max |x| observed.
            scale = _mse_search_symmetric_hist(
                bin_centers, self.hist, self.hist_max.item(),
                self.qmin, self.qmax,
                self.n_steps, self.p_min, self.p_max,
            )
            self.scale.copy_(scale.to(self.scale.dtype).to(device))
            self.zero_point.zero_()
        else:
            scale, zp = _mse_search_asymmetric_hist(
                bin_centers, self.hist,
                self.hist_min.item(), self.hist_max.item(),
                self.qmin, self.qmax,
                self.n_steps, self.p_min, self.p_max,
            )
            self.scale.copy_(scale.to(self.scale.dtype).to(device))
            self.zero_point.copy_(zp.to(device))
        self.mode.fill_(FROZEN)


@register_observer("per_channel_mse")
class PerChannelMSEObserver(PerChannelHistogramObserver):
    """Per-channel MSE-optimal scale, histogram-backed (mirror of MSEObserver)."""

    def __init__(
        self,
        bits: int,
        scheme: str,
        axis: int = 0,
        n_bins: int = 2048,
        n_steps: int = 80,
        p_min: float = 0.5,
        p_max: float = 1.2,
    ):
        super().__init__(bits=bits, scheme=scheme, axis=axis, n_bins=n_bins)
        self.n_steps = n_steps
        self.p_min = p_min
        self.p_max = p_max

    def freeze(self) -> None:
        if self.initialized.item() == 0:
            raise RuntimeError("Cannot freeze PerChannelMSEObserver: no data observed.")
        n_channels = self.hist.shape[0]
        bin_centers = _hist_bin_centers_per_channel(
            self.hist_min, self.hist_max, self.n_bins
        )  # (C, n_bins)
        if self.scheme == SYMMETRIC:
            # symmetric_uses_abs True: hist_max is per-channel max |x|.
            scale = _mse_search_symmetric_hist_per_channel(
                bin_centers, self.hist, self.hist_max,
                self.qmin, self.qmax,
                self.n_steps, self.p_min, self.p_max,
            )
            self.scale = scale.to(torch.float32)
            self.zero_point = torch.zeros(
                n_channels, dtype=torch.int64, device=self.hist.device
            )
        else:
            scale, zp = _mse_search_asymmetric_hist_per_channel(
                bin_centers, self.hist, self.hist_min, self.hist_max,
                self.qmin, self.qmax,
                self.n_steps, self.p_min, self.p_max,
            )
            self.scale = scale.to(torch.float32)
            self.zero_point = zp.to(torch.int64)
        self.mode.fill_(FROZEN)
