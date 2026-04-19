"""QuantConv2d — wraps nn.Conv2d with weight + input-activation fake-quantization.

Weights: observed + frozen once in __init__ (static, no data needed).
Activations: observer starts DISABLED; convert.py + calibrate.py drive it to CALIBRATING then FROZEN.
Bias: stays FP32 (standard fake-quant convention).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quantization.observer import DISABLED, BaseObserver, build_observer


class QuantConv2d(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        stride,
        padding,
        dilation,
        groups: int,
        padding_mode: str,
        weight_bits: int,
        act_bits: int,
        weight_observer: str = "min_max",
        act_observer: str = "min_max",
        weight_scheme: str = "symmetric",
        act_scheme: str = "asymmetric",
    ):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight_bits = weight_bits
        self.act_bits = act_bits

        # Weight observer: observe-then-freeze immediately (weights are static).
        self.weight_observer: BaseObserver = build_observer(
            weight_observer, bits=weight_bits, scheme=weight_scheme
        )
        self.weight_observer.observe(self.weight)
        self.weight_observer.freeze()

        # Activation observer: starts DISABLED. calibrate.py drives it.
        self.act_observer: BaseObserver = build_observer(
            act_observer, bits=act_bits, scheme=act_scheme
        )

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, **quant_cfg) -> "QuantConv2d":
        """Factory: build a QuantConv2d that matches an existing nn.Conv2d."""
        return cls(
            weight=conv.weight.data,
            bias=conv.bias.data if conv.bias is not None else None,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            padding_mode=conv.padding_mode,
            **quant_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_q = self.act_observer(x)
        w_q = self.weight_observer.fake_quantize(self.weight)
        if self.padding_mode != "zeros":
            x_q = F.pad(
                x_q,
                self._reversed_padding_repeated_twice(),
                mode=self.padding_mode,
            )
            return F.conv2d(
                x_q, w_q, self.bias,
                self.stride, (0, 0), self.dilation, self.groups,
            )
        return F.conv2d(
            x_q, w_q, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

    def _reversed_padding_repeated_twice(self):
        p = self.padding
        if isinstance(p, int):
            p = (p, p)
        return (p[1], p[1], p[0], p[0])

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1] * self.groups}, out={self.weight.shape[0]}, "
            f"k={tuple(self.weight.shape[-2:])}, stride={self.stride}, "
            f"groups={self.groups}, w_bits={self.weight_bits}, a_bits={self.act_bits}"
        )
