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
from quantization.observer import FROZEN
from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear
from quantization.quant_stub import QuantStub, PreStubbedModule


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
    assert len(swapped) == 3
    n_plain = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d) and not isinstance(m, QuantConv2d))
    assert n_plain == 0
    assert len(collect_quant_convs(model)) == 3


def test_convert_respects_skip_list():
    model = ToyNet()
    _, swapped = convert_model(model, weight_bits=8, act_bits=8, skip_modules=["conv_1"])
    assert "conv_1.conv" not in swapped
    assert len(swapped) == 2
    assert isinstance(model.conv_1.conv, nn.Conv2d)
    assert not isinstance(model.conv_1.conv, QuantConv2d)


def test_convert_handles_sequential_numeric_indices():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    assert isinstance(model.layer_1[0], QuantConv2d)
    assert isinstance(model.layer_1[2], QuantConv2d)


def test_forward_still_works_after_convert():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 10)


def test_converted_convs_have_no_act_observer():
    """QuantConv2d must NOT have act_observer — activation Q lives in stubs."""
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    for qc in collect_quant_convs(model):
        assert not hasattr(qc, "act_observer")


# -------- calibrate — requires stubs -------- #


def test_calibrate_without_stubs_raises():
    """Without stubs there are no act_observers, so calibrate must raise."""
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    with pytest.raises(RuntimeError, match="No quantized modules"):
        calibrate(model, iter([torch.randn(2, 3, 16, 16)]), n_batches=1)


# -------- weight-only 16-bit sanity (no calibration needed) -------- #


def test_weight_only_16bit_near_fp32_output():
    """16-bit weight fake-quant (no stubs) should be near FP32 — no calibrate needed."""
    torch.manual_seed(7)
    model = ToyNet().eval()
    x = torch.randn(2, 3, 16, 16)
    y_fp = model(x)
    convert_model(model, weight_bits=16, act_bits=16)
    y_q = model(x)
    assert (y_fp - y_q).abs().max().item() < 1e-2


# -------- quantize_linear flag -------- #


def test_quantize_linear_swaps_nn_linear():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8, quantize_linear=True)
    assert isinstance(model.classifier, QuantLinear)
    assert len(collect_quant_linears(model)) == 1


def test_quantize_linear_default_off_keeps_fp32():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8)
    assert isinstance(model.classifier, nn.Linear)
    assert len(collect_quant_linears(model)) == 0


def test_quantize_linear_respects_skip_list():
    model = ToyNet()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_linear=True, skip_linears=["classifier"],
    )
    assert isinstance(model.classifier, nn.Linear)


def test_quantize_linear_has_no_act_observer():
    model = ToyNet()
    convert_model(model, weight_bits=8, act_bits=8, quantize_linear=True)
    assert not hasattr(model.classifier, "act_observer")


# -------- insert_stubs with real AFFNet module types -------- #


class ToyIRNet(nn.Module):
    """Small net using AFFNet's InvertedResidual so the default stub walker finds targets."""

    def __init__(self, opts):
        super().__init__()
        from affnet.modules.mobilenetv2 import InvertedResidual
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.ir1 = InvertedResidual(opts=opts, in_channels=8, out_channels=8, stride=1, expand_ratio=2)
        self.ir2 = InvertedResidual(opts=opts, in_channels=8, out_channels=16, stride=2, expand_ratio=2)
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.ir1(x)
        x = self.ir2(x)
        return self.head(x.mean(dim=(2, 3)))


@pytest.fixture(scope="module")
def ir_opts():
    import argparse
    ns = argparse.Namespace()
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
    assert isinstance(model.ir1, PreStubbedModule)
    assert isinstance(model.ir2, PreStubbedModule)
    assert len(collect_quant_stubs(model)) >= 2


def test_insert_stubs_respects_skip_list(ir_opts):
    model = ToyIRNet(opts=ir_opts)
    convert_model(model, weight_bits=8, act_bits=8, insert_stubs=True, skip_stubs=["ir1"])
    assert not isinstance(model.ir1, PreStubbedModule)
    assert isinstance(model.ir2, PreStubbedModule)


def test_insert_stubs_off_by_default(ir_opts):
    model = ToyIRNet(opts=ir_opts)
    convert_model(model, weight_bits=8, act_bits=8)
    assert not isinstance(model.ir1, PreStubbedModule)
    assert not isinstance(model.ir2, PreStubbedModule)
    assert len(collect_quant_stubs(model)) == 0


def test_forward_still_works_with_stubs(ir_opts):
    torch.manual_seed(3)
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(model, weight_bits=16, act_bits=16, quantize_linear=True, insert_stubs=True)
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 4)


# -------- calibrate with stubs -------- #


def test_calibrate_drives_stub_observers_to_frozen(ir_opts):
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(model, weight_bits=8, act_bits=8, insert_stubs=True)
    stubs = collect_quant_stubs(model)
    assert len(stubs) == 2

    qms = collect_quant_modules(model)
    assert len(qms) == 2  # only the 2 stubs — no conv act_observers

    batches = [torch.randn(2, 3, 16, 16) for _ in range(4)]
    n = calibrate(model, batches, n_batches=4)
    assert n == 2
    for s in stubs:
        assert int(s.act_observer.mode.item()) == FROZEN


def test_calibrate_respects_n_batches(ir_opts):
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(model, weight_bits=8, act_bits=8, insert_stubs=True)
    count = [0]

    def gen():
        while True:
            count[0] += 1
            yield torch.randn(1, 3, 8, 8)

    calibrate(model, gen(), n_batches=3)
    assert count[0] == 3


def test_calibrate_input_fn_extracts_from_dict(ir_opts):
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(model, weight_bits=8, act_bits=8, insert_stubs=True)
    batches = [{"samples": torch.randn(1, 3, 8, 8), "targets": torch.tensor([0])} for _ in range(3)]
    calibrate(model, batches, n_batches=3, input_fn=lambda b: b["samples"])
    for s in collect_quant_stubs(model):
        assert int(s.act_observer.mode.item()) == FROZEN


def test_calibrate_restores_training_mode(ir_opts):
    model = ToyIRNet(opts=ir_opts)
    model.train()
    convert_model(model, weight_bits=8, act_bits=8, insert_stubs=True)
    calibrate(model, [torch.randn(1, 3, 8, 8) for _ in range(2)], n_batches=2)
    assert model.training is True


def test_collect_quant_modules_returns_only_stubs(ir_opts):
    """With stubs-only activation Q, collect_quant_modules must return only QuantStub."""
    model = ToyIRNet(opts=ir_opts).eval()
    convert_model(
        model, weight_bits=8, act_bits=8,
        quantize_linear=True, insert_stubs=True,
    )
    qms = collect_quant_modules(model)
    assert all(isinstance(m, QuantStub) for m in qms)
    assert len(qms) == 2  # ir1.stub + ir2.stub (no QuantLinear act_observer)


# -------- 16-bit math-sanity with stubs -------- #


def test_16bit_with_stubs_near_fp32(ir_opts):
    """PreStubbedModule + QuantLinear at 16-bit should be near FP32."""
    torch.manual_seed(11)
    model = ToyIRNet(opts=ir_opts).eval()
    x_cal = [torch.randn(4, 3, 16, 16) for _ in range(5)]
    x_test = torch.randn(2, 3, 16, 16)
    y_fp = model(x_test)

    convert_model(
        model, weight_bits=16, act_bits=16,
        quantize_linear=True, insert_stubs=True,
        stub_observer="min_max", stub_bits=16,
        weight_observer="min_max",
    )
    calibrate(model, x_cal, n_batches=5)

    max_err = (y_fp - model(x_test)).abs().max().item()
    assert max_err < 5e-2, f"16/16 with stubs too divergent: max_err={max_err}"
