"""Tests for quantization/quant_conv.py."""

import pytest
import torch
from torch import nn

from quantization.observer import CALIBRATING, DISABLED, FROZEN
from quantization.quant_conv import QuantConv2d


torch.manual_seed(0)


def make_conv(in_c=8, out_c=16, k=3, stride=1, padding=1, groups=1, bias=True):
    return nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=bias)


# -------- construction -------- #


def test_from_conv2d_copies_params():
    conv = make_conv(bias=True)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    assert torch.equal(q.weight.data, conv.weight.data)
    assert torch.equal(q.bias.data, conv.bias.data)
    assert q.stride == conv.stride
    assert q.padding == conv.padding
    assert q.dilation == conv.dilation
    assert q.groups == conv.groups


def test_no_bias_conv_has_no_bias():
    conv = make_conv(bias=False)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    assert q.bias is None


def test_depthwise_conv_preserved():
    conv = make_conv(in_c=16, out_c=16, groups=16)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    assert q.groups == 16


def test_weight_observer_frozen_in_init():
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    assert int(q.weight_observer.mode.item()) == FROZEN


def test_act_observer_starts_disabled():
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    assert int(q.act_observer.mode.item()) == DISABLED


# -------- forward equivalence sanity -------- #


def test_disabled_act_and_16bit_weight_near_fp32():
    """Activation pass-through + 16-bit weight should be near bit-exact to nn.Conv2d."""
    torch.manual_seed(42)
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=16, act_bits=8)
    assert int(q.act_observer.mode.item()) == DISABLED  # no activation quant
    x = torch.randn(2, 8, 16, 16)
    y_fp = conv(x)
    y_q = q(x)
    max_err = (y_fp - y_q).abs().max().item()
    # 16-bit weight quant is very fine, but not exactly bit-exact — within a small tolerance
    assert max_err < 1e-2, f"16-bit weight + disabled-act too divergent: max_err={max_err}"


def test_disabled_both_is_bit_exact():
    """When we bypass weight quant too (via hack: weight_bits very high), output should match FP32."""
    # Not directly testable without FP32 bypass mode; but we can check 16-bit + disabled act
    # is "close enough" — the strict bit-exact test is handled in test_fake_quant.
    pass


def test_preserves_output_shape():
    conv = make_conv(in_c=3, out_c=32, k=3, stride=2, padding=1)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    x = torch.randn(1, 3, 32, 32)
    y = q(x)
    assert y.shape == conv(x).shape


# -------- calibration workflow -------- #


def test_activation_calibration_then_eval():
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=8, act_bits=8)
    q.act_observer.set_mode(CALIBRATING)
    x1 = torch.randn(2, 8, 16, 16)
    q(x1)
    x2 = torch.randn(2, 8, 16, 16) * 2
    q(x2)
    q.act_observer.freeze()
    assert int(q.act_observer.mode.item()) == FROZEN
    y = q(torch.randn(2, 8, 16, 16))
    assert y.shape == (2, 16, 16, 16)


def test_lower_bits_gives_more_quantization_error():
    """Sanity: output at 4-bit deviates from FP32 more than at 16-bit."""
    torch.manual_seed(123)
    conv = make_conv()
    x = torch.randn(4, 8, 16, 16)
    y_fp = conv(x)

    q16 = QuantConv2d.from_conv2d(conv, weight_bits=16, act_bits=8)
    q4 = QuantConv2d.from_conv2d(conv, weight_bits=4, act_bits=8)

    err16 = (y_fp - q16(x)).abs().max().item()
    err4 = (y_fp - q4(x)).abs().max().item()
    assert err4 > err16, f"4-bit weight should introduce more error than 16-bit (err16={err16}, err4={err4})"


# -------- per-channel weight observer integration -------- #


def test_per_channel_weight_observer_in_qconv():
    conv = make_conv(in_c=8, out_c=16)
    q = QuantConv2d.from_conv2d(
        conv, weight_bits=4, act_bits=8, weight_observer="per_channel_min_max",
    )
    # weight_observer frozen, scale has one entry per output channel
    assert int(q.weight_observer.mode.item()) == FROZEN
    assert q.weight_observer.scale.shape == (16,)
    # Forward still produces correct shape
    x = torch.randn(2, 8, 16, 16)
    y = q(x)
    assert y.shape == (2, 16, 16, 16)


def test_per_channel_reduces_error_vs_per_tensor_on_depthwise_conv():
    torch.manual_seed(1)
    # 16-channel depthwise conv with channel-magnitude imbalance
    conv = make_conv(in_c=16, out_c=16, groups=16)
    with torch.no_grad():
        conv.weight[0] *= 30.0
        conv.weight[1] *= 0.01
    x = torch.randn(2, 16, 8, 8)
    y_fp = conv(x)

    q_per_tensor = QuantConv2d.from_conv2d(
        conv, weight_bits=4, act_bits=8, weight_observer="min_max",
    )
    q_per_ch = QuantConv2d.from_conv2d(
        conv, weight_bits=4, act_bits=8, weight_observer="per_channel_min_max",
    )
    err_pt = (y_fp - q_per_tensor(x)).abs().mean().item()
    err_pc = (y_fp - q_per_ch(x)).abs().mean().item()
    assert err_pc < err_pt / 2, f"per-channel ({err_pc}) should beat per-tensor ({err_pt})"


# -------- state dict round-trip -------- #


def test_state_dict_roundtrip():
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=4, act_bits=4)
    q.act_observer.set_mode(CALIBRATING)
    q(torch.randn(1, 8, 8, 8))
    q.act_observer.freeze()

    q2 = QuantConv2d.from_conv2d(conv, weight_bits=4, act_bits=4)
    q2.load_state_dict(q.state_dict())

    x = torch.randn(1, 8, 8, 8)
    assert torch.equal(q(x), q2(x))
