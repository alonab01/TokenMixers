"""Pure Q->DQ (fake-quantization) math for PTQ simulation.

Symmetric signed (weights): zero_point = 0, qmin = -(2^(b-1)-1), qmax = 2^(b-1)-1.
    We intentionally drop -2^(b-1) so the range is symmetric around zero.
Asymmetric unsigned (activations): qmin = 0, qmax = 2^b - 1.

No autograd hooks — PTQ does not need STE. Forward-only.
"""
from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-8


def qrange_symmetric(bits: int) -> tuple[int, int]:
    assert bits >= 2, f"bits must be >= 2, got {bits}"
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    return qmin, qmax


def qrange_asymmetric(bits: int) -> tuple[int, int]:
    assert bits >= 2, f"bits must be >= 2, got {bits}"
    return 0, (1 << bits) - 1


def compute_scale_symmetric(max_abs: Tensor, qmax: int) -> Tensor:
    scale = max_abs / qmax
    return torch.clamp(scale, min=EPS)


def compute_scale_zp_asymmetric(
    min_val: Tensor, max_val: Tensor, qmin: int, qmax: int
) -> tuple[Tensor, Tensor]:
    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))
    scale = (max_val - min_val) / (qmax - qmin)
    scale = torch.clamp(scale, min=EPS)
    zp = torch.round(qmin - min_val / scale)
    zp = torch.clamp(zp, qmin, qmax).to(torch.int64)
    return scale, zp


def fake_quantize_symmetric(
    x: Tensor, scale: Tensor, qmin: int, qmax: int
) -> Tensor:
    """x -> quantize to integer -> dequantize, zero_point = 0."""
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, qmin, qmax)
    return x_q * scale


def fake_quantize_asymmetric(
    x: Tensor, scale: Tensor, zero_point: Tensor, qmin: int, qmax: int
) -> Tensor:
    zp = zero_point.to(x.dtype)
    x_q = torch.round(x / scale) + zp
    x_q = torch.clamp(x_q, qmin, qmax)
    return (x_q - zp) * scale
