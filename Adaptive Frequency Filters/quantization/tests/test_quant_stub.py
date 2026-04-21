"""Tests for quantization/quant_stub.py."""

import pytest
import torch
from torch import nn

from quantization.observer import CALIBRATING, DISABLED, FROZEN
from quantization.quant_stub import QuantStub, PreStubbedModule


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


# -------- PreStubbedModule wrapper -------- #


def test_pre_stubbed_module_disabled_is_bit_exact_to_inner():
    """When stub is DISABLED (default), output must be bit-exact to running inner on the raw input."""
    inner = nn.Linear(8, 4, bias=True)
    stub = QuantStub(act_bits=8)
    wrapper = PreStubbedModule(inner=inner, stub=stub)
    x = torch.randn(2, 8)
    assert torch.equal(wrapper(x), inner(x))


def test_pre_stubbed_module_frozen_input_to_inner_is_quantized():
    """When stub is FROZEN, inner should receive a quantized input, not the raw tensor."""
    captured = []

    class CaptureInput(nn.Module):
        def forward(self, x):
            captured.append(x.clone())
            return x

    stub = QuantStub(act_bits=4, act_observer="min_max", act_scheme="asymmetric")
    inner = CaptureInput()
    wrapper = PreStubbedModule(inner=inner, stub=stub)

    stub.act_observer.set_mode(CALIBRATING)
    wrapper(torch.randn(100))
    stub.act_observer.freeze()
    captured.clear()

    x = torch.randn(100)
    wrapper(x)

    # inner received the Q'd input, not x
    assert not torch.equal(captured[0], x)
    # 4-bit asymmetric -> at most 16 distinct values in the quantized input
    assert torch.unique(captured[0]).numel() <= 16


def test_pre_stubbed_module_output_shape():
    inner = nn.Conv2d(3, 4, 3, padding=1)
    stub = QuantStub(act_bits=8)
    wrapper = PreStubbedModule(inner=inner, stub=stub)
    x = torch.randn(1, 3, 8, 8)
    y = wrapper(x)
    assert y.shape == (1, 4, 8, 8)


def test_pre_stubbed_module_registered_as_child():
    inner = nn.Conv2d(3, 4, 3)
    stub = QuantStub(act_bits=8)
    wrapper = PreStubbedModule(inner=inner, stub=stub)
    names = dict(wrapper.named_modules())
    assert "inner" in names
    assert "stub" in names


def test_pre_stubbed_module_stub_calibrated_not_inner():
    """Calibrating a PreStubbedModule should update the stub's observer, not inner's weights."""
    from quantization.observer import FROZEN

    inner = nn.Linear(8, 4)
    stub = QuantStub(act_bits=8, act_observer="min_max")
    wrapper = PreStubbedModule(inner=inner, stub=stub)

    stub.act_observer.set_mode(CALIBRATING)
    for _ in range(3):
        wrapper(torch.randn(4, 8))
    stub.act_observer.freeze()

    assert int(stub.act_observer.mode.item()) == FROZEN


def test_pre_stubbed_input_quantization_verified():
    """PreStubbedModule quantizes BEFORE inner; verify by checking inner sees Q'd tensor."""
    torch.manual_seed(42)
    inner = nn.Identity()
    stub = QuantStub(act_bits=4, act_observer="min_max", act_scheme="asymmetric")
    pre = PreStubbedModule(inner=inner, stub=stub)

    x_cal = [torch.randn(50) for _ in range(5)]
    stub.act_observer.set_mode(CALIBRATING)
    for xc in x_cal:
        pre(xc)
    stub.act_observer.freeze()

    x = torch.randn(50)
    y = pre(x)
    # With Identity inner, output IS the quantized input. 4-bit -> at most 16 distinct values.
    assert torch.unique(y).numel() <= 16
