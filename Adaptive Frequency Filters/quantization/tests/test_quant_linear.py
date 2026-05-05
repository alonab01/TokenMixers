"""Tests for quantization/quant_linear.py — weight + activation fake-quantization."""

import pytest
import torch
from torch import nn

from quantization.observer import BaseObserver, CALIBRATING, DISABLED, FROZEN
from quantization.quant_linear import QuantLinear


torch.manual_seed(0)


def make_linear(in_f=32, out_f=10, bias=True):
    return nn.Linear(in_f, out_f, bias=bias)


# -------- construction -------- #


def test_from_linear_copies_params():
    lin = make_linear()
    q = QuantLinear.from_linear(lin, weight_bits=8)
    assert torch.equal(q.weight.data, lin.weight.data)
    assert torch.equal(q.bias.data, lin.bias.data)
    assert q.in_features == lin.in_features
    assert q.out_features == lin.out_features


def test_no_bias_linear_has_no_bias():
    lin = make_linear(bias=False)
    q = QuantLinear.from_linear(lin, weight_bits=8)
    assert q.bias is None


def test_weight_observer_frozen_in_init():
    q = QuantLinear.from_linear(make_linear(), weight_bits=8)
    assert int(q.weight_observer.mode.item()) == FROZEN


def test_has_act_observer():
    """QuantLinear owns an act_observer (BaseObserver subclass)."""
    q = QuantLinear.from_linear(make_linear(), weight_bits=8)
    assert isinstance(q.act_observer, BaseObserver)


def test_act_observer_starts_disabled():
    q = QuantLinear.from_linear(make_linear(), weight_bits=8)
    assert int(q.act_observer.mode.item()) == DISABLED


def test_act_observer_calibrate_freeze_quantizes_input():
    """After calibrate + freeze, act_observer Q's the input."""
    torch.manual_seed(7)
    lin = make_linear(in_f=4, out_f=4, bias=False)
    with torch.no_grad():
        lin.weight.zero_()
        for c in range(4):
            lin.weight[c, c] = 1.0
    q = QuantLinear.from_linear(
        lin, weight_bits=16, act_bits=4, act_observer="min_max", act_scheme="asymmetric",
    )
    q.act_observer.set_mode(CALIBRATING)
    for _ in range(3):
        q(torch.randn(8, 4))
    q.act_observer.freeze()
    assert int(q.act_observer.mode.item()) == FROZEN

    y = q(torch.randn(8, 4))
    # 16-bit identity weights are bit-exact; 4-bit asymmetric input -> at most 16 distinct values
    assert torch.unique(y).numel() <= 16


# -------- forward -------- #


def test_preserves_output_shape():
    q = QuantLinear.from_linear(make_linear(32, 10), weight_bits=8)
    assert q(torch.randn(4, 32)).shape == (4, 10)


def test_16bit_weight_near_fp32():
    lin = make_linear(32, 10)
    q = QuantLinear.from_linear(lin, weight_bits=16)
    x = torch.randn(2, 32)
    err = (lin(x) - q(x)).abs().max().item()
    assert err < 1e-2, f"16-bit weight too divergent: {err}"


def test_lower_bits_gives_more_error():
    torch.manual_seed(1)
    lin = make_linear(32, 10)
    x = torch.randn(4, 32)
    y_fp = lin(x)
    e16 = (y_fp - QuantLinear.from_linear(lin, weight_bits=16)(x)).abs().max().item()
    e4 = (y_fp - QuantLinear.from_linear(lin, weight_bits=4)(x)).abs().max().item()
    assert e4 > e16


# -------- per-channel weight observer -------- #


def test_per_channel_weight_observer_gives_per_out_feature_scale():
    q = QuantLinear.from_linear(
        make_linear(32, 10), weight_bits=4, weight_observer="per_channel_min_max",
    )
    assert q.weight_observer.scale.shape == (10,)
    assert q(torch.randn(2, 32)).shape == (2, 10)


# -------- rejects unsupported variants -------- #


def test_rejects_group_linear():
    from affnet.layers.linear_layer import GroupLinear
    gl = GroupLinear(in_features=16, out_features=8, n_groups=2, bias=True)
    with pytest.raises(TypeError, match="GroupLinear"):
        QuantLinear.from_linear(gl, weight_bits=8)


def test_rejects_channel_first_linear_layer():
    from affnet.layers.linear_layer import LinearLayer
    ll = LinearLayer(in_features=16, out_features=8, bias=True, channel_first=True)
    with pytest.raises(TypeError, match="channel_first"):
        QuantLinear.from_linear(ll, weight_bits=8)


def test_accepts_plain_linear_layer():
    from affnet.layers.linear_layer import LinearLayer
    ll = LinearLayer(in_features=16, out_features=8, bias=True, channel_first=False)
    q = QuantLinear.from_linear(ll, weight_bits=8)
    assert q.in_features == 16
    assert q.out_features == 8
    assert q(torch.randn(2, 16)).shape == (2, 8)
