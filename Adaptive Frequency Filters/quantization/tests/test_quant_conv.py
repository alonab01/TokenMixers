"""Tests for quantization/quant_conv.py — weight-only fake-quantization."""

import torch
from torch import nn

from quantization.observer import FROZEN
from quantization.quant_conv import QuantConv2d


torch.manual_seed(0)


def make_conv(in_c=8, out_c=16, k=3, stride=1, padding=1, groups=1, bias=True):
    return nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=bias)


# -------- construction -------- #


def test_from_conv2d_copies_params():
    conv = make_conv(bias=True)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8)
    assert torch.equal(q.weight.data, conv.weight.data)
    assert torch.equal(q.bias.data, conv.bias.data)
    assert q.stride == conv.stride
    assert q.padding == conv.padding
    assert q.dilation == conv.dilation
    assert q.groups == conv.groups


def test_no_bias_conv_has_no_bias():
    conv = make_conv(bias=False)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8)
    assert q.bias is None


def test_depthwise_conv_preserved():
    conv = make_conv(in_c=16, out_c=16, groups=16)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8)
    assert q.groups == 16


def test_weight_observer_frozen_in_init():
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=8)
    assert int(q.weight_observer.mode.item()) == FROZEN


def test_no_act_observer():
    """QuantConv2d must not have an act_observer — activation Q lives in stubs."""
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=8)
    assert not hasattr(q, "act_observer")


# -------- forward -------- #


def test_16bit_weight_near_fp32():
    """16-bit weight fake-quant should be nearly bit-exact to nn.Conv2d."""
    torch.manual_seed(42)
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=16)
    x = torch.randn(2, 8, 16, 16)
    max_err = (conv(x) - q(x)).abs().max().item()
    assert max_err < 1e-2, f"16-bit weight too divergent: {max_err}"


def test_preserves_output_shape():
    conv = make_conv(in_c=3, out_c=32, k=3, stride=2, padding=1)
    q = QuantConv2d.from_conv2d(conv, weight_bits=8)
    x = torch.randn(1, 3, 32, 32)
    assert q(x).shape == conv(x).shape


def test_lower_bits_gives_more_quantization_error():
    torch.manual_seed(123)
    conv = make_conv()
    x = torch.randn(4, 8, 16, 16)
    y_fp = conv(x)
    q16 = QuantConv2d.from_conv2d(conv, weight_bits=16)
    q4 = QuantConv2d.from_conv2d(conv, weight_bits=4)
    err16 = (y_fp - q16(x)).abs().max().item()
    err4 = (y_fp - q4(x)).abs().max().item()
    assert err4 > err16, f"4-bit ({err4}) should exceed 16-bit ({err16}) error"


# -------- per-channel weight observer -------- #


def test_per_channel_weight_observer_in_qconv():
    conv = make_conv(in_c=8, out_c=16)
    q = QuantConv2d.from_conv2d(conv, weight_bits=4, weight_observer="per_channel_min_max")
    assert int(q.weight_observer.mode.item()) == FROZEN
    assert q.weight_observer.scale.shape == (16,)
    assert q(torch.randn(2, 8, 16, 16)).shape == (2, 16, 16, 16)


def test_per_channel_reduces_error_vs_per_tensor_on_depthwise_conv():
    torch.manual_seed(1)
    conv = make_conv(in_c=16, out_c=16, groups=16)
    with torch.no_grad():
        conv.weight[0] *= 30.0
        conv.weight[1] *= 0.01
    x = torch.randn(2, 16, 8, 8)
    y_fp = conv(x)
    q_pt = QuantConv2d.from_conv2d(conv, weight_bits=4, weight_observer="min_max")
    q_pc = QuantConv2d.from_conv2d(conv, weight_bits=4, weight_observer="per_channel_min_max")
    err_pt = (y_fp - q_pt(x)).abs().mean().item()
    err_pc = (y_fp - q_pc(x)).abs().mean().item()
    assert err_pc < err_pt / 2, f"per-channel ({err_pc}) should beat per-tensor ({err_pt})"


# -------- state dict round-trip -------- #


def test_state_dict_roundtrip():
    conv = make_conv()
    q = QuantConv2d.from_conv2d(conv, weight_bits=4)
    q2 = QuantConv2d.from_conv2d(conv, weight_bits=4)
    q2.load_state_dict(q.state_dict())
    x = torch.randn(1, 8, 8, 8)
    assert torch.equal(q(x), q2(x))
