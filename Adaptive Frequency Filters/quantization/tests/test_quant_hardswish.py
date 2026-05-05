"""Tests for QuantHardswish: standalone forward, and the _swap_acts convert path."""

import torch
import torch.nn.functional as F
from torch import nn

from quantization.calibrate import calibrate
from quantization.convert import (
    collect_quant_convs,
    collect_quant_hardswish,
    collect_quant_modules,
    convert_model,
)
from quantization.observer import BaseObserver, DISABLED, FROZEN
from quantization.quant_conv import QuantConv2d
from quantization.quant_hardswish import QuantHardswish


# -------- standalone QuantHardswish -------- #


def test_quant_hardswish_disabled_passthrough():
    """In DISABLED mode the observer is a no-op, so output matches plain hardswish."""
    qh = QuantHardswish(act_bits=8, act_observer="min_max", act_scheme="asymmetric")
    assert int(qh.act_observer.mode.item()) == DISABLED
    x = torch.randn(2, 4, 8, 8)
    assert torch.allclose(qh(x), F.hardswish(x))


def test_quant_hardswish_carries_observer():
    qh = QuantHardswish(act_bits=8)
    assert isinstance(qh.act_observer, BaseObserver)
    assert qh.act_bits == 8


# -------- convert + calibrate on a toy model -------- #


class ToyNetWithSwish(nn.Module):
    """Conv -> Swish -> Conv -> HardSwish -> GAP -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.act2 = nn.Hardswish()
        self.classifier = nn.Linear(8, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return self.classifier(x.mean(dim=(2, 3)))


def test_swap_acts_replaces_silu_and_hardswish():
    model = ToyNetWithSwish()
    convert_model(model, weight_bits=8, act_bits=8, quantize_acts=True)
    # Both activation sites are now QuantHardswish.
    assert isinstance(model.act1, QuantHardswish)
    assert isinstance(model.act2, QuantHardswish)
    # Nothing left at nn.SiLU / nn.Hardswish.
    assert not any(
        isinstance(m, (nn.SiLU, nn.Hardswish)) and not isinstance(m, QuantHardswish)
        for m in model.modules()
    )
    assert len(collect_quant_hardswish(model)) == 2


def test_swap_acts_off_by_default():
    """Default convert_model leaves activations untouched (Phase 1-7 behavior)."""
    model = ToyNetWithSwish()
    convert_model(model, weight_bits=8, act_bits=8)
    assert isinstance(model.act1, nn.SiLU)
    assert isinstance(model.act2, nn.Hardswish)
    assert len(collect_quant_hardswish(model)) == 0


def test_swap_acts_respects_skip_list():
    model = ToyNetWithSwish()
    convert_model(
        model, weight_bits=8, act_bits=8, quantize_acts=True, skip_acts=["act1"]
    )
    assert isinstance(model.act1, nn.SiLU)
    assert isinstance(model.act2, QuantHardswish)


def test_calibrate_picks_up_quant_hardswish():
    """collect_quant_modules is duck-typed on .act_observer — QuantHardswish must qualify."""
    model = ToyNetWithSwish()
    convert_model(model, weight_bits=8, act_bits=8, quantize_acts=True)
    # 2 convs + 2 hardswish = 4 act observers (linear is not quantized here).
    assert len(collect_quant_modules(model)) == 4

    n = calibrate(
        model,
        [torch.randn(4, 3, 16, 16) for _ in range(3)],
        n_batches=3,
    )
    assert n == 4
    for qh in collect_quant_hardswish(model):
        assert int(qh.act_observer.mode.item()) == FROZEN
        # scale should be a finite positive value after seeing real data.
        assert torch.isfinite(qh.act_observer.scale).all()
        assert (qh.act_observer.scale > 0).all()


def test_forward_still_works_after_swap_acts():
    model = ToyNetWithSwish()
    convert_model(
        model, weight_bits=8, act_bits=8, quantize_linear=True, quantize_acts=True
    )
    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 10)
    # And one full calibrate-then-eval round-trip:
    calibrate(model, [torch.randn(4, 3, 16, 16) for _ in range(2)], n_batches=2)
    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_swap_acts_idempotent():
    """Running _swap_acts twice doesn't double-wrap."""
    model = ToyNetWithSwish()
    convert_model(model, weight_bits=8, act_bits=8, quantize_acts=True)
    n_first = len(collect_quant_hardswish(model))
    convert_model(model, weight_bits=8, act_bits=8, quantize_acts=True)
    assert len(collect_quant_hardswish(model)) == n_first
