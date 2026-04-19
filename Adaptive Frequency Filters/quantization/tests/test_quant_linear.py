"""Tests for quantization/quant_linear.py — mirrors test_quant_conv.py."""

import pytest
import torch
from torch import nn

from quantization.observer import CALIBRATING, DISABLED, FROZEN
from quantization.quant_linear import QuantLinear


torch.manual_seed(0)


def make_linear(in_f=32, out_f=10, bias=True):
    return nn.Linear(in_f, out_f, bias=bias)


# -------- construction -------- #


def test_from_linear_copies_params():
    lin = make_linear()
    q = QuantLinear.from_linear(lin, weight_bits=8, act_bits=8)
    assert torch.equal(q.weight.data, lin.weight.data)
    assert torch.equal(q.bias.data, lin.bias.data)
    assert q.in_features == lin.in_features
    assert q.out_features == lin.out_features


def test_no_bias_linear_has_no_bias():
    lin = make_linear(bias=False)
    q = QuantLinear.from_linear(lin, weight_bits=8, act_bits=8)
    assert q.bias is None


def test_weight_observer_frozen_in_init():
    q = QuantLinear.from_linear(make_linear(), weight_bits=8, act_bits=8)
    assert int(q.weight_observer.mode.item()) == FROZEN


def test_act_observer_starts_disabled():
    q = QuantLinear.from_linear(make_linear(), weight_bits=8, act_bits=8)
    assert int(q.act_observer.mode.item()) == DISABLED


# -------- forward equivalence sanity -------- #


def test_preserves_output_shape():
    q = QuantLinear.from_linear(make_linear(32, 10), weight_bits=8, act_bits=8)
    x = torch.randn(4, 32)
    y = q(x)
    assert y.shape == (4, 10)


def test_disabled_act_and_16bit_weight_near_fp32():
    lin = make_linear(32, 10)
    q = QuantLinear.from_linear(lin, weight_bits=16, act_bits=8)
    x = torch.randn(2, 32)
    err = (lin(x) - q(x)).abs().max().item()
    assert err < 1e-2, f"16-bit weight + disabled-act too divergent: {err}"


def test_lower_bits_gives_more_error():
    torch.manual_seed(1)
    lin = make_linear(32, 10)
    x = torch.randn(4, 32)
    y_fp = lin(x)
    q16 = QuantLinear.from_linear(lin, weight_bits=16, act_bits=8)
    q4 = QuantLinear.from_linear(lin, weight_bits=4, act_bits=8)
    e16 = (y_fp - q16(x)).abs().max().item()
    e4 = (y_fp - q4(x)).abs().max().item()
    assert e4 > e16


# -------- per-channel integration -------- #


def test_per_channel_weight_observer_gives_per_out_feature_scale():
    q = QuantLinear.from_linear(
        make_linear(32, 10), weight_bits=4, act_bits=8,
        weight_observer="per_channel_min_max",
    )
    assert q.weight_observer.scale.shape == (10,)
    x = torch.randn(2, 32)
    assert q(x).shape == (2, 10)


# -------- calibration -------- #


def test_activation_calibration_then_eval():
    q = QuantLinear.from_linear(make_linear(16, 8), weight_bits=8, act_bits=8)
    q.act_observer.set_mode(CALIBRATING)
    q(torch.randn(4, 16))
    q(torch.randn(4, 16) * 2)
    q.act_observer.freeze()
    assert int(q.act_observer.mode.item()) == FROZEN
    assert q(torch.randn(4, 16)).shape == (4, 8)


# -------- rejects unsupported variants -------- #


def test_rejects_group_linear():
    """GroupLinear has a 3D weight (n_groups, in, out) — not supported."""
    from affnet.layers.linear_layer import GroupLinear

    gl = GroupLinear(in_features=16, out_features=8, n_groups=2, bias=True)
    with pytest.raises(TypeError, match="GroupLinear"):
        QuantLinear.from_linear(gl, weight_bits=8, act_bits=8)


def test_rejects_channel_first_linear_layer():
    """LinearLayer(channel_first=True) routes through F.conv2d — must be refused."""
    from affnet.layers.linear_layer import LinearLayer

    ll = LinearLayer(in_features=16, out_features=8, bias=True, channel_first=True)
    with pytest.raises(TypeError, match="channel_first"):
        QuantLinear.from_linear(ll, weight_bits=8, act_bits=8)


def test_accepts_plain_linear_layer():
    from affnet.layers.linear_layer import LinearLayer

    ll = LinearLayer(in_features=16, out_features=8, bias=True, channel_first=False)
    q = QuantLinear.from_linear(ll, weight_bits=8, act_bits=8)
    assert q.in_features == 16
    assert q.out_features == 8
    x = torch.randn(2, 16)
    assert q(x).shape == (2, 8)
