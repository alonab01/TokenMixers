"""QuantStub — insert re-quantization points at chosen FP32 boundaries.

A QuantStub re-quantizes a single tensor; its state machine is the usual
DISABLED / CALIBRATING / FROZEN from BaseObserver. Stubs are a flexible tool
for any FP32 boundary the user wants on the Q grid (LayerNorm input,
BatchNorm input, activation function input, AFNO2D output, GlobalPool, ...).
Conv/Linear inputs are NOT a stub job — those self-quantize via QuantConv2d /
QuantLinear's own act_observer.

The companion `StubConfig` dataclass is the per-type config consumed by
convert.py::_insert_stubs to dispatch (bits, observer, scheme) per nn.Module
type.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from quantization.observer import BaseObserver, build_observer


@dataclass(frozen=True)
class StubConfig:
    """Per-type stub configuration consumed by _insert_stubs."""
    bits: int
    observer: str = "percentile"
    scheme: str = "asymmetric"


class QuantStub(nn.Module):
    """Single-tensor re-quantization point using an activation observer."""

    def __init__(
        self,
        act_bits: int,
        act_observer: str = "percentile",
        act_scheme: str = "asymmetric",
    ):
        super().__init__()
        self.act_bits = act_bits
        # Stubs in this codebase only wrap modules that consume NCHW activations
        # (LayerNorm2D, AFNO2D, Block, AFFBlock, Swish, GlobalPool — all work on
        # 4-D channel-first tensors), so axis=1 is the right per-channel axis.
        # Per-tensor observers ignore the axis kwarg.
        self.act_observer: BaseObserver = build_observer(
            act_observer, bits=act_bits, scheme=act_scheme, axis=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act_observer(x)

    def extra_repr(self) -> str:
        return f"a_bits={self.act_bits}, observer={type(self.act_observer).__name__}"


class PreStubbedModule(nn.Module):
    """Wraps an inner module and quantizes its INPUT before passing in.

    Models the int8 memory read at the start of a hardware fused kernel.
    The inner module runs on a quantized input; its internal activations stay
    FP32 (no intermediate memory writes inside a fused kernel).
    """

    def __init__(self, inner: nn.Module, stub: QuantStub):
        super().__init__()
        self.inner = inner
        self.stub = stub

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x_q = self.stub(x)
        return self.inner(x_q, *args, **kwargs)

    def extra_repr(self) -> str:
        return f"pre_stubbed={type(self.inner).__name__}"
