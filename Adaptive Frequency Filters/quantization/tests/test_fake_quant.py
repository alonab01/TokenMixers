"""Tests for quantization/fake_quant.py."""

import math

import pytest
import torch

from quantization.fake_quant import (
    compute_scale_symmetric,
    compute_scale_zp_asymmetric,
    fake_quantize_asymmetric,
    fake_quantize_symmetric,
    qrange_asymmetric,
    qrange_symmetric,
)


torch.manual_seed(0)


# -------- qrange -------- #


def test_qrange_symmetric_is_centered():
    for b in [2, 4, 8, 16]:
        qmin, qmax = qrange_symmetric(b)
        assert qmin == -qmax
        assert qmax == 2 ** (b - 1) - 1


def test_qrange_asymmetric_is_unsigned():
    for b in [2, 4, 8, 16]:
        qmin, qmax = qrange_asymmetric(b)
        assert qmin == 0
        assert qmax == 2**b - 1


# -------- symmetric Q/DQ -------- #


def test_symmetric_bits16_near_identity():
    x = torch.randn(1024) * 5
    qmin, qmax = qrange_symmetric(16)
    scale = compute_scale_symmetric(x.abs().max(), qmax)
    x_dq = fake_quantize_symmetric(x, scale, qmin, qmax)
    max_err = (x - x_dq).abs().max().item()
    # ~range/qmax/2 is the resolution; err should be much smaller than FP32 meaningful precision
    assert max_err < 1e-3, f"16-bit symmetric should be near-identity, got max_err={max_err}"


def test_symmetric_4bit_has_exactly_15_levels():
    x = torch.linspace(-2, 2, steps=2000)
    qmin, qmax = qrange_symmetric(4)  # qmin=-7, qmax=7 -> 15 levels
    scale = compute_scale_symmetric(x.abs().max(), qmax)
    x_dq = fake_quantize_symmetric(x, scale, qmin, qmax)
    unique = torch.unique(x_dq)
    assert unique.numel() == 15, f"expected 15 distinct values at 4-bit symmetric, got {unique.numel()}"


def test_symmetric_clamps_out_of_range():
    # scale=1 means integer grid; inputs beyond [-127,127] must clamp
    x = torch.tensor([-200.0, 200.0])
    qmin, qmax = qrange_symmetric(8)
    scale = torch.tensor(1.0)
    x_dq = fake_quantize_symmetric(x, scale, qmin, qmax)
    assert x_dq[0].item() == float(qmin)
    assert x_dq[1].item() == float(qmax)


def test_symmetric_zero_stays_zero():
    x = torch.tensor([0.0, 0.0, 0.0])
    qmin, qmax = qrange_symmetric(4)
    scale = compute_scale_symmetric(torch.tensor(1.0), qmax)
    x_dq = fake_quantize_symmetric(x, scale, qmin, qmax)
    assert torch.all(x_dq == 0.0)


def test_symmetric_scale_handles_all_zero_tensor():
    x = torch.zeros(10)
    qmin, qmax = qrange_symmetric(8)
    scale = compute_scale_symmetric(x.abs().max(), qmax)
    assert scale.item() > 0, "scale must be positive to avoid div-by-zero"
    x_dq = fake_quantize_symmetric(x, scale, qmin, qmax)
    assert torch.all(x_dq == 0.0)


# -------- asymmetric Q/DQ -------- #


def test_asymmetric_bits16_near_identity():
    x = torch.rand(1024) * 10 + 2  # one-sided, positive
    qmin, qmax = qrange_asymmetric(16)
    scale, zp = compute_scale_zp_asymmetric(x.min(), x.max(), qmin, qmax)
    x_dq = fake_quantize_asymmetric(x, scale, zp, qmin, qmax)
    max_err = (x - x_dq).abs().max().item()
    assert max_err < 1e-3, f"16-bit asymmetric should be near-identity, got max_err={max_err}"


def test_asymmetric_4bit_has_exactly_16_levels():
    x = torch.linspace(-2, 5, steps=2000)
    qmin, qmax = qrange_asymmetric(4)  # 0..15 -> 16 levels
    scale, zp = compute_scale_zp_asymmetric(x.min(), x.max(), qmin, qmax)
    x_dq = fake_quantize_asymmetric(x, scale, zp, qmin, qmax)
    unique = torch.unique(x_dq)
    assert unique.numel() == 16, f"expected 16 distinct values at 4-bit asymmetric, got {unique.numel()}"


def test_asymmetric_zero_is_exactly_representable():
    """Key property of asymmetric quant: the zero_point makes x=0 round-trip losslessly."""
    x = torch.tensor([-1.0, 0.0, 3.0])
    qmin, qmax = qrange_asymmetric(8)
    scale, zp = compute_scale_zp_asymmetric(x.min(), x.max(), qmin, qmax)
    x_dq = fake_quantize_asymmetric(x, scale, zp, qmin, qmax)
    assert abs(x_dq[1].item()) < 1e-6, f"zero must be exactly representable, got {x_dq[1].item()}"


def test_asymmetric_clamps_out_of_range():
    x = torch.tensor([-10.0, 10.0])
    qmin, qmax = qrange_asymmetric(4)
    # calibrate on a narrow range, feed out-of-range values
    cal_min, cal_max = torch.tensor(-0.5), torch.tensor(0.5)
    scale, zp = compute_scale_zp_asymmetric(cal_min, cal_max, qmin, qmax)
    x_dq = fake_quantize_asymmetric(x, scale, zp, qmin, qmax)
    # Values beyond calibrated range must clamp
    assert x_dq[0].item() > x[0].item()
    assert x_dq[1].item() < x[1].item()


def test_asymmetric_handles_degenerate_range():
    """min == max (constant tensor) must not blow up."""
    x = torch.full((10,), 3.0)
    qmin, qmax = qrange_asymmetric(8)
    scale, zp = compute_scale_zp_asymmetric(x.min(), x.max(), qmin, qmax)
    assert scale.item() > 0
    x_dq = fake_quantize_asymmetric(x, scale, zp, qmin, qmax)
    assert torch.all(torch.isfinite(x_dq))


# -------- dtype / device preservation -------- #


def test_preserves_dtype_float32():
    x = torch.randn(64, dtype=torch.float32)
    qmin, qmax = qrange_symmetric(8)
    scale = compute_scale_symmetric(x.abs().max(), qmax)
    x_dq = fake_quantize_symmetric(x, scale, qmin, qmax)
    assert x_dq.dtype == torch.float32
