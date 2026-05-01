"""QuantHardswish — drop-in replacement for any Swish/HardSwish site with input quantization.

Mirrors QuantConv2d / QuantLinear: owns an `act_observer` on its input. Forward path
is observer (passthrough / calibrate / fake-quantize) -> F.hardswish. No weight quant
because HardSwish has no learnable parameters.

The convert pass installs this in place of every Swish (nn.SiLU subclass) and
HardSwish (nn.Hardswish subclass) in the model — the function-swap and the
activation-input quantization are bundled into one op, matching real INT8 hardware
where the Conv output is requantized before the nonlinearity reads it.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from quantization.observer import BaseObserver, build_observer


class QuantHardswish(nn.Module):
    def __init__(
        self,
        act_bits: int = 8,
        act_observer: str = "min_max",
        act_scheme: str = "asymmetric",
    ):
        super().__init__()
        self.act_bits = act_bits
        self.act_observer: BaseObserver = build_observer(
            act_observer, bits=act_bits, scheme=act_scheme
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.act_observer(x)
        return F.hardswish(x)

    def extra_repr(self) -> str:
        return f"a_bits={self.act_bits}, observer={type(self.act_observer).__name__}"
