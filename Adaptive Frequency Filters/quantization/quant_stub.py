"""QuantStub — insert re-quantization points at block boundaries.

Phase 3: simulate real hardware data flow. In deployed INT hardware, a tensor
leaving one block stays on the quantization grid when entering the next block;
it does not transiently live as unconstrained FP32. A QuantStub is placed at
every such boundary to re-quantize the tensor after any FP32 operation
(residual-add, activation, BN, AFNO2D output, GlobalPool, ...).

The stub wraps a standard activation observer. Its state machine is the usual
DISABLED / CALIBRATING / FROZEN from BaseObserver — no new math.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from quantization.observer import BaseObserver, build_observer


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
        self.act_observer: BaseObserver = build_observer(
            act_observer, bits=act_bits, scheme=act_scheme
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.act_observer(x)

    def extra_repr(self) -> str:
        return f"a_bits={self.act_bits}, observer={type(self.act_observer).__name__}"


class StubbedModule(nn.Module):
    """Wraps an inner module and applies a QuantStub to its output.

    Used by convert.py to post-quantize block outputs without editing AFFNet source.
    The wrapping is opaque: same input/output tensor signature as `inner`.
    """

    def __init__(self, inner: nn.Module, stub: QuantStub):
        super().__init__()
        self.inner = inner
        self.stub = stub

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y = self.inner(x, *args, **kwargs)
        return self.stub(y)

    def extra_repr(self) -> str:
        return f"stubbed={type(self.inner).__name__}"
