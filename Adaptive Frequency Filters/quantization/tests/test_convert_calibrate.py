"""Integration tests for convert.py + calibrate.py on small synthetic models."""

import pytest
import torch
from torch import nn

from quantization.calibrate import calibrate
from quantization.convert import (
    _is_skipped,
    collect_quant_convs,
    collect_quant_linears,
    collect_quant_modules,
    collect_quant_stubs,
    convert_model,
)
from quantization.observer import CALIBRATING, FROZEN
from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear
from quantization.quant_stub import QuantStub, StubbedModule


# -------- skip-list matching -------- #


def test_skip_list_exact_match():
    assert _is_skipped("conv_1", ["conv_1"])


def test_skip_list_prefix_match():
    assert _is_skipped("conv_1.block.conv", ["conv_1"])


def test_skip_list_does_not_match_similar_prefix():
    """'conv_1' must not match 'conv_1x1_exp' — the guard is the trailing dot."""
    assert not _is_skipped("conv_1x1_exp.block.conv", ["conv_1"])


def test_empty_skip_list():
    assert not _is_skipped("any.path", [])


def test_multiple_skip_entries():
    skip = ["conv_1", "classifier"]
    assert _is_skipped("conv_1.block.conv", skip)
    assert _is_skipped("classifier.linear", skip)
    assert not _is_skipped("layer_2.conv", skip)


# -------- convert_model on a toy model -------- #


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential()
        self.conv_1.add_module("conv", nn.Conv2d(3, 8, 3, padding=1))
        self.conv_1.add_module("bn", nn.BatchNorm2d(8))
        self.layer_1 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),  # depthwise
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)


def test_convert_swaps_all_convs_by_default():
    model = ToyNet()
    _, swapped = convert_model(model, weight_bits=8, act_bits=8)
    # ToyNet has 3 nn.Conv2d
    assert len(swapped) == 3
    # All Conv2d should now be QuantConv2d
    n_plain_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and not isinstance(m, QuantConv2d))
    assert n_plain_conv == 0
    n_qconv = len(collect_quant_convs(model))
    assert n_qconv == 3


def test_convert_respects_skip_list():
    model = ToyNet()
    _, swapped = convert_model(model, weight_bits=8, act_bits=8, skip_modules=["conv_1"])
    # conv_1.block.conv stayed as nn.Conv2d
    assert "conv_1.conv" not in swapped
    # The other two got swapped
    assert len(swapped) == 2
    # Verify conv_1.conv is still a plain nn.Conv2d
    assert isinstance(model.conv_1.conv, nn.Conv2d)
    assert not isinstance(model.conv_1.conv, QuantConv2d)


def test_convert_handles_sequential_numeric_indices():
    """layer_1 is nn.Sequential — children are accessed by string index '0', '2'."""
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    assert isinstance(model.layer_1[0], QuantConv2d)
    assert isinstance(model.layer_1[2], QuantConv2d)


def test_forward_still_works_after_convert():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    assert y.shape == (2, 10)


# -------- calibrate ---------- #


def test_calibrate_drives_observers_to_frozen():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    qcs = collect_quant_convs(model)

    def gen():
        for _ in range(10):
            yield torch.randn(2, 3, 16, 16)

    n = calibrate(model, gen(), n_batches=4)
    assert n == len(qcs)
    for qc in qcs:
        assert int(qc.act_observer.mode.item()) == FROZEN
        # stats should have been collected
        assert qc.act_observer.min_val.item() < float("inf")
        assert qc.act_observer.max_val.item() > float("-inf")


def test_calibrate_without_conversion_raises():
    model = ToyNet()
    with pytest.raises(RuntimeError, match="No quantized modules"):
        calibrate(model, iter([torch.randn(2, 3, 16, 16)]), n_batches=1)


def test_calibrate_respects_n_batches():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    count = [0]

    def gen():
        while True:
            count[0] += 1
            yield torch.randn(1, 3, 8, 8)

    calibrate(model, gen(), n_batches=3)
    assert count[0] == 3


def test_calibrate_input_fn_extracts_from_dict():
    """Real AFFNet loader yields {'samples': ..., 'targets': ...} dicts."""
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    batches = [{"samples": torch.randn(1, 3, 8, 8), "targets": torch.tensor([0])} for _ in range(3)]
    calibrate(model, batches, n_batches=3, input_fn=lambda b: b["samples"])
    for qc in collect_quant_convs(model):
        assert int(qc.act_observer.mode.item()) == FROZEN


def test_calibrate_restores_training_mode():
    model = ToyNet()
    model.train()
    convert_model(model, weight_bits=8, act_bits=8)
    batches = [torch.randn(1, 3, 8, 8) for _ in range(2)]
    calibrate(model, batches, n_batches=2)
    assert model.training is True


# -------- end-to-end 16-bit sanity on toy model -------- #


def test_16bit_quant_near_fp32_output():
    """16-bit weights + 16-bit activations should be nearly FP32-exact on a toy net."""
    torch.manual_seed(7)
    model = ToyNet().eval()
    x_cal = [torch.randn(4, 3, 16, 16) for _ in range(5)]
    x_test = torch.randn(2, 3, 16, 16)

    y_fp = model(x_test)

    convert_model(model, weight_bits=16, act_bits=16)
    calibrate(model, x_cal, n_batches=5)

    y_q = model(x_test)
    max_err = (y_fp - y_q).abs().max().item()
    # toy net has 3 quantized convs; compounded 16-bit error still very small
    assert max_err < 1e-2, f"16/16 on toy net too divergent: max_err={max_err}"


# -------- Phase 3: quantize_linear flag -------- #


def test_quantize_linear_swaps_nn_linear():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8, quantize_linear=True)
    assert isinstance(model.classifier, QuantLinear)
    assert len(collect_quant_linears(model)) == 1


def test_quantize_linear_default_off_keeps_fp32():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    assert isinstance(model.classifier, nn.Linear)
    assert not isinstance(model.classifier, QuantLinear)
    assert len(collect_quant_linears(model)) == 0


def test_quantize_linear_respects_skip_list():
    model = ToyNet()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_linear=True, skip_linears=["classifier"],
    )
    assert isinstance(model.classifier, nn.Linear)
    assert len(collect_quant_linears(model)) == 0


# -------- Phase 3: insert_stubs flag (with real AFFNet module types) -------- #


class ToyIRNet(nn.Module):
    """Small net using AFFNet's InvertedResidual so the default stub walker finds targets."""

    def __init__(self, opts):
        super().__init__()
        from affnet.modules.mobilenetv2 import InvertedResidual

        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.ir1 = InvertedResidual(
            opts=opts, in_channels=8, out_channels=8, stride=1, expand_ratio=2,
        )
        self.ir2 = InvertedResidual(
            opts=opts, in_channels=8, out_channels=16, stride=2, expand_ratio=2,
        )
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.ir1(x)
        x = self.ir2(x)
        return self.head(x.mean(dim=(2, 3)))


@pytest.fixture(scope="module")
def ir_opts():
    """Minimal opts namespace needed to build an AFFNet InvertedResidual."""
    import argparse
    ns = argparse.Namespace()
    # Only the flags the InvertedResidual stack reads — kept minimal.
    setattr(ns, "model.normalization.name", "batch_norm")
    setattr(ns, "model.normalization.groups", 1)
    setattr(ns, "model.normalization.momentum", 0.1)
    setattr(ns, "model.activation.name", "relu")
    setattr(ns, "model.activation.inplace", False)
    setattr(ns, "model.activation.neg_slope", 0.1)
    setattr(ns, "model.layer.global_pool", "mean")
    setattr(ns, "model.layer.conv_init", "kaiming_normal")
    setattr(ns, "model.layer.linear_init", "xavier_uniform")
    setattr(ns, "model.layer.conv_init_std_dev", None)
    setattr(ns, "model.layer.linear_init_std_dev", 0.01)
    return ns


def test_insert_stubs_wraps_inverted_residuals(ir_opts):
    model = ToyIRNet(opts=ir_opts)
    convert_model(model, weight_bits=8, act_bits=8, insert_stubs=True)
    # ir1 and ir2 both wrapped in StubbedModule
    assert isinstance(model.ir1, StubbedModule)
    assert isinstance(model.ir2, StubbedModule)
    stubs = collect_quant_stubs(model)
    assert len(stubs) >= 2


def test_insert_stubs_respects_skip_list(ir_opts):
    model = ToyIRNet(opts=ir_opts)
    convert_model(
        model, weight_bits=8, act_bits=8,
        insert_stubs=True, skip_stubs=["ir1"],
    )
    # ir1 NOT wrapped; ir2 is wrapped
    assert not isinstance(model.ir1, StubbedModule)
    assert isinstance(model.ir2, StubbedModule)


def test_insert_stubs_off_by_default(ir_opts):
    model = ToyIRNet(opts=ir_opts)
    convert_model(model, weight_bits=8, act_bits=8)
    assert not isinstance(model.ir1, StubbedModule)
    assert not isinstance(model.ir2, StubbedModule)
    assert len(collect_quant_stubs(model)) == 0


def test_forward_still_works_with_stubs(ir_opts):
    torch.manual_seed(3)
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(
        model, weight_bits=16, act_bits=16,
        quantize_linear=True, insert_stubs=True,
    )
    x = torch.randn(2, 3, 16, 16)
    y = model(x)
    assert y.shape == (2, 4)


def test_calibrate_covers_stubs_and_linears(ir_opts):
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_linear=True, insert_stubs=True,
    )
    qms = collect_quant_modules(model)
    # Covers QuantConv2d (stem) + QuantLinear (head) + every QuantStub
    assert len(qms) >= 3

    batches = [torch.randn(2, 3, 16, 16) for _ in range(3)]
    n = calibrate(model, batches, n_batches=3)
    assert n == len(qms)
    for qm in qms:
        assert int(qm.act_observer.mode.item()) == FROZEN


# -------- Phase 3: 16/16 math-sanity with stubs + linear ON -------- #


def test_16bit_with_all_phase3_near_fp32(ir_opts):
    """Stubs + QuantLinear at 16/16 should still be near FP32 on toy net."""
    torch.manual_seed(11)
    model = ToyIRNet(opts=ir_opts).eval()
    x_cal = [torch.randn(4, 3, 16, 16) for _ in range(5)]
    x_test = torch.randn(2, 3, 16, 16)
    y_fp = model(x_test)

    convert_model(
        model, weight_bits=16, act_bits=16,
        quantize_linear=True, insert_stubs=True,
        stub_observer="min_max", stub_bits=16,
        weight_observer="min_max", act_observer="min_max",
    )
    calibrate(model, x_cal, n_batches=5)

    y_q = model(x_test)
    max_err = (y_fp - y_q).abs().max().item()
    # 16-bit everywhere including stubs — should still be tight
    assert max_err < 5e-2, f"16/16 with Phase 3 too divergent: max_err={max_err}"
