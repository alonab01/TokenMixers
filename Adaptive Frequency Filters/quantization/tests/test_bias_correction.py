"""Tests for quantization/bias_correction.py."""

import pytest
import torch
from torch import nn

from quantization.bias_correction import (
    _InputMeanHook,
    _compute_conv_delta_bias,
    _compute_linear_delta_bias,
    apply_bias_correction,
)
from quantization.convert import convert_model
from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear


torch.manual_seed(0)


def _make_simple_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )


def test_input_mean_hook_4d():
    h = _InputMeanHook()

    class FakeMod(nn.Module):
        def forward(self, x):
            return x  # passthrough so output IS x

    m = FakeMod()
    handle = m.register_forward_hook(h)
    x = torch.randn(4, 8, 16, 16)
    m(x)
    expected = x.mean(dim=(0, 2, 3))
    assert torch.allclose(h.running_mean, expected, atol=1e-5)
    handle.remove()


def test_input_mean_hook_running_average():
    h = _InputMeanHook()

    class FakeMod(nn.Module):
        def forward(self, x):
            return x

    m = FakeMod()
    handle = m.register_forward_hook(h)
    xs = [torch.randn(2, 4, 8, 8) for _ in range(5)]
    for x in xs:
        m(x)
    expected = torch.stack([x.mean(dim=(0, 2, 3)) for x in xs]).mean(dim=0)
    assert torch.allclose(h.running_mean, expected, atol=1e-5)
    handle.remove()


def test_compute_conv_delta_bias_shape():
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    qc = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    input_mean = torch.randn(3)
    delta = _compute_conv_delta_bias(qc, input_mean)
    assert delta.shape == (8,)


def test_compute_conv_delta_bias_zero_when_no_quant_error():
    """If weight quantization error is zero (16-bit), delta_bias should be near-zero."""
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    qc = QuantConv2d.from_conv2d(conv, weight_bits=16, act_bits=16)
    input_mean = torch.randn(3)
    delta = _compute_conv_delta_bias(qc, input_mean)
    assert delta.abs().max().item() < 1e-3, f"16-bit conv delta_bias too large: {delta}"


def test_compute_conv_delta_bias_grouped():
    """Depthwise conv (groups == in == out) should still produce per-channel delta."""
    conv = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
    qc = QuantConv2d.from_conv2d(conv, weight_bits=4, act_bits=8)
    input_mean = torch.randn(8)
    delta = _compute_conv_delta_bias(qc, input_mean)
    assert delta.shape == (8,)


def test_compute_linear_delta_bias():
    lin = nn.Linear(16, 10)
    ql = QuantLinear.from_linear(lin, weight_bits=4, act_bits=8)
    input_mean = torch.randn(16)
    delta = _compute_linear_delta_bias(ql, input_mean)
    assert delta.shape == (10,)


def test_apply_bias_correction_changes_bias():
    """End-to-end: model with quantized weights gets bias-corrected on simple data."""
    torch.manual_seed(42)
    model = _make_simple_model()
    convert_model(model, weight_bits=4, act_bits=8, weight_observer="per_channel_min_max",
                  act_observer="percentile", quantize_linear=True)

    # Calibrate quickly so act_observers are FROZEN (else they're DISABLED, output != intended)
    from quantization.calibrate import calibrate
    cal_data = [torch.randn(4, 3, 16, 16) for _ in range(3)]
    calibrate(model, cal_data, n_batches=3)

    # Snapshot biases
    bias_before = []
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            bias_before.append(None if m.bias is None else m.bias.detach().clone())

    bc_data = [torch.randn(4, 3, 16, 16) for _ in range(3)]
    n_corrected = apply_bias_correction(model, bc_data, n_batches=3)
    assert n_corrected == 3  # 2 convs + 1 linear

    # At least one bias should differ
    bias_after = []
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            bias_after.append(None if m.bias is None else m.bias.detach().clone())

    any_changed = any(
        a is not None and b is not None and not torch.equal(a, b)
        for a, b in zip(bias_before, bias_after)
    )
    assert any_changed, "Bias correction did not change any bias"


def test_apply_bias_correction_no_quant_modules_returns_zero():
    """If model has no QuantConv2d / QuantLinear, return 0 (no-op)."""
    model = _make_simple_model()  # plain FP32, not converted
    cal_data = [torch.randn(4, 3, 16, 16)]
    n = apply_bias_correction(model, cal_data, n_batches=1)
    assert n == 0


def test_apply_bias_correction_creates_bias_for_conv_without_one():
    """A Conv2d that originally had bias=False should get a bias added by correction."""
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    qc = QuantConv2d.from_conv2d(conv, weight_bits=4, act_bits=8,
                                  weight_observer="per_channel_min_max",
                                  act_observer="min_max")
    assert qc.bias is None

    # Wrap so apply_bias_correction sees it
    model = nn.Sequential(qc)
    from quantization.calibrate import calibrate
    cal_data = [torch.randn(2, 3, 8, 8) for _ in range(2)]
    calibrate(model, cal_data, n_batches=2)

    apply_bias_correction(model, cal_data, n_batches=2)
    assert qc.bias is not None
    assert qc.bias.shape == (8,)
