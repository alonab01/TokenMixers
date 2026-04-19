"""Tests for quantization/quant_stub.py."""

import pytest
import torch
from torch import nn

from quantization.observer import CALIBRATING, DISABLED, FROZEN
from quantization.quant_stub import QuantStub, StubbedModule


torch.manual_seed(0)


# -------- QuantStub state machine -------- #


def test_stub_starts_disabled():
    s = QuantStub(act_bits=8)
    assert int(s.act_observer.mode.item()) == DISABLED


def test_stub_disabled_is_passthrough():
    s = QuantStub(act_bits=4)
    x = torch.randn(10)
    y = s(x)
    assert torch.equal(x, y)


def test_stub_calibrates_then_freezes():
    s = QuantStub(act_bits=8, act_observer="percentile", act_scheme="asymmetric")
    s.act_observer.set_mode(CALIBRATING)
    for _ in range(3):
        s(torch.randn(100))
    s.act_observer.freeze()
    assert int(s.act_observer.mode.item()) == FROZEN
    y = s(torch.randn(100))
    assert y.shape == (100,)
    # Frozen 8-bit asymmetric → at most 256 distinct values
    assert torch.unique(y).numel() <= 256


def test_stub_preserves_shape():
    s = QuantStub(act_bits=8)
    s.act_observer.set_mode(CALIBRATING)
    x = torch.randn(2, 16, 8, 8)
    y = s(x)
    assert y.shape == x.shape


def test_stub_default_observer_is_percentile():
    s = QuantStub(act_bits=8)
    from quantization.observer import PercentileObserver
    assert isinstance(s.act_observer, PercentileObserver)


def test_stub_custom_observer():
    s = QuantStub(act_bits=8, act_observer="min_max")
    from quantization.observer import MinMaxObserver
    assert isinstance(s.act_observer, MinMaxObserver)


# -------- StubbedModule wrapper -------- #


def test_stubbed_module_wraps_inner_forward():
    inner = nn.Conv2d(3, 4, 3, padding=1)
    stub = QuantStub(act_bits=8)
    wrapper = StubbedModule(inner=inner, stub=stub)
    x = torch.randn(1, 3, 8, 8)
    y = wrapper(x)
    assert y.shape == (1, 4, 8, 8)


def test_stubbed_module_disabled_is_bit_exact_to_inner():
    """When stub is DISABLED, wrapper output must equal inner output exactly."""
    inner = nn.Conv2d(3, 4, 3, padding=1, bias=True)
    stub = QuantStub(act_bits=8)
    wrapper = StubbedModule(inner=inner, stub=stub)
    x = torch.randn(2, 3, 8, 8)
    assert torch.equal(wrapper(x), inner(x))


def test_stubbed_module_frozen_changes_output():
    inner = nn.Conv2d(3, 4, 3, padding=1)
    stub = QuantStub(act_bits=4, act_observer="min_max", act_scheme="asymmetric")
    wrapper = StubbedModule(inner=inner, stub=stub)
    x = torch.randn(2, 3, 8, 8)

    stub.act_observer.set_mode(CALIBRATING)
    for _ in range(3):
        wrapper(torch.randn(2, 3, 8, 8))
    stub.act_observer.freeze()

    y_frozen = wrapper(x)
    y_fp = inner(x)
    # 4-bit quantization should introduce visible error
    assert not torch.equal(y_frozen, y_fp)
    # Still numerically close
    assert (y_frozen - y_fp).abs().max().item() < 5.0


def test_stubbed_module_is_registered_as_child():
    """Both inner and stub should appear under named_modules (important for calibrate walker)."""
    inner = nn.Conv2d(3, 4, 3)
    stub = QuantStub(act_bits=8)
    wrapper = StubbedModule(inner=inner, stub=stub)
    names = dict(wrapper.named_modules())
    assert "inner" in names
    assert "stub" in names
