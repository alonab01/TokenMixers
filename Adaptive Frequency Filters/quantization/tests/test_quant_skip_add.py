"""Tests for QuantSkipAdd: standalone forward, the _swap_skip_adds convert path,
and that the two per-branch QuantStub children are picked up by calibration."""

import torch
from torch import nn

from quantization.calibrate import calibrate
from quantization.convert import (
    collect_quant_modules,
    collect_quant_skip_adds,
    collect_quant_stubs,
    convert_model,
)
from quantization.observer import BaseObserver, DISABLED, FROZEN
from quantization.quant_skip_add import QuantSkipAdd, SkipAdd
from quantization.quant_stub import QuantStub


# -------- standalone QuantSkipAdd -------- #


def test_quant_skip_add_disabled_passthrough():
    """Both stubs in DISABLED mode are no-ops, so output equals plain `a + b`."""
    qsa = QuantSkipAdd.from_skip_add(
        SkipAdd(), bits=8, observer="min_max", scheme="asymmetric"
    )
    assert int(qsa.stub_main.act_observer.mode.item()) == DISABLED
    assert int(qsa.stub_skip.act_observer.mode.item()) == DISABLED
    a = torch.randn(2, 4, 8, 8)
    b = torch.randn(2, 4, 8, 8)
    assert torch.allclose(qsa(a, b), a + b)


def test_quant_skip_add_carries_two_observers():
    qsa = QuantSkipAdd.from_skip_add(
        SkipAdd(), bits=8, observer="min_max", scheme="asymmetric"
    )
    assert isinstance(qsa.stub_main, QuantStub)
    assert isinstance(qsa.stub_skip, QuantStub)
    assert isinstance(qsa.stub_main.act_observer, BaseObserver)
    assert isinstance(qsa.stub_skip.act_observer, BaseObserver)
    # Independence: distinct module instances (NOT shared)
    assert qsa.stub_main is not qsa.stub_skip
    assert qsa.stub_main.act_observer is not qsa.stub_skip.act_observer


def test_quant_skip_add_high_bits_recovers_fp_add():
    """At very high bit-width and frozen observers, QuantSkipAdd ≈ plain add (math sanity)."""
    qsa = QuantSkipAdd.from_skip_add(
        SkipAdd(), bits=16, observer="min_max", scheme="asymmetric"
    )
    # Calibrate by toggling stubs to CALIBRATING, running data through, then freezing.
    from quantization.observer import CALIBRATING
    qsa.stub_main.act_observer.set_mode(CALIBRATING)
    qsa.stub_skip.act_observer.set_mode(CALIBRATING)
    a = torch.randn(8, 4, 16, 16)
    b = torch.randn(8, 4, 16, 16)
    qsa(a, b)
    qsa.stub_main.act_observer.freeze()
    qsa.stub_skip.act_observer.freeze()
    # On unseen data, output should match plain add to within 16-bit grid noise.
    a2 = torch.randn(2, 4, 16, 16)
    b2 = torch.randn(2, 4, 16, 16)
    assert torch.allclose(qsa(a2, b2), a2 + b2, atol=1e-3, rtol=1e-3)


# -------- convert + calibrate on a toy model -------- #


class ToyResidualNet(nn.Module):
    """Conv -> SkipAdd(branch, x) -> Conv -> classifier. Two SkipAdd sites."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv1b = nn.Conv2d(8, 8, 3, padding=1)
        self.skip_add1 = SkipAdd()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2b = nn.Conv2d(8, 8, 3, padding=1)
        self.skip_add2 = SkipAdd()
        self.classifier = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.skip_add1(self.conv1b(x), x)
        x = self.conv2(x)
        x = self.skip_add2(self.conv2b(x), x)
        return self.classifier(x.mean(dim=(2, 3)))


def test_swap_skip_adds_replaces_passthroughs():
    model = ToyResidualNet()
    convert_model(model, weight_bits=8, act_bits=8, quantize_residuals=True)
    assert isinstance(model.skip_add1, QuantSkipAdd)
    assert isinstance(model.skip_add2, QuantSkipAdd)
    # Subclass relationship preserved (QuantSkipAdd extends SkipAdd) — the
    # post-swap instances are still SkipAdd in isinstance terms.
    assert isinstance(model.skip_add1, SkipAdd)
    # No bare SkipAdd (non-quantized) left.
    assert not any(
        isinstance(m, SkipAdd) and not isinstance(m, QuantSkipAdd)
        for m in model.modules()
    )
    assert len(collect_quant_skip_adds(model)) == 2


def test_swap_skip_adds_off_by_default():
    """Default convert_model leaves SkipAdd untouched (Phase 1-8c behavior)."""
    model = ToyResidualNet()
    convert_model(model, weight_bits=8, act_bits=8)
    assert type(model.skip_add1) is SkipAdd
    assert type(model.skip_add2) is SkipAdd
    assert len(collect_quant_skip_adds(model)) == 0


def test_swap_skip_adds_respects_skip_list():
    model = ToyResidualNet()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_residuals=True, skip_residuals=["skip_add1"],
    )
    assert type(model.skip_add1) is SkipAdd
    assert isinstance(model.skip_add2, QuantSkipAdd)


def test_calibrate_picks_up_quant_skip_add_stubs():
    """Each QuantSkipAdd's two child QuantStubs are walked by collect_quant_modules
    via the duck-type on .act_observer, no special-case required."""
    model = ToyResidualNet()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_linear=True, quantize_residuals=True,
    )
    # 4 convs + 1 linear + 2 QuantSkipAdds × 2 stubs each = 9 act observers.
    assert len(collect_quant_modules(model)) == 9
    # The 4 stub instances live as direct children of the QuantSkipAdds.
    assert len(collect_quant_stubs(model)) == 4

    n = calibrate(
        model,
        [torch.randn(4, 3, 16, 16) for _ in range(3)],
        n_batches=3,
    )
    assert n == 9
    for qsa in collect_quant_skip_adds(model):
        for stub in (qsa.stub_main, qsa.stub_skip):
            assert int(stub.act_observer.mode.item()) == FROZEN
            assert torch.isfinite(stub.act_observer.scale).all()
            assert (stub.act_observer.scale > 0).all()
        # Independence sanity: per-branch scales should generally differ on real data.
        # (This is a soft check — they could coincide, but very unlikely with random data.)
        assert qsa.stub_main.act_observer is not qsa.stub_skip.act_observer


def test_forward_still_works_after_swap():
    model = ToyResidualNet()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_linear=True, quantize_residuals=True,
    )
    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 10)
    # And one full calibrate-then-eval round-trip.
    calibrate(model, [torch.randn(4, 3, 16, 16) for _ in range(2)], n_batches=2)
    out = model(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_swap_skip_adds_idempotent():
    """Running the convert pass twice doesn't double-wrap (QuantSkipAdd is itself a SkipAdd)."""
    model = ToyResidualNet()
    convert_model(model, weight_bits=8, act_bits=8, quantize_residuals=True)
    n_first = len(collect_quant_skip_adds(model))
    convert_model(model, weight_bits=8, act_bits=8, quantize_residuals=True)
    assert len(collect_quant_skip_adds(model)) == n_first
