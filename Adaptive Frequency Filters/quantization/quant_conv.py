"""QuantConv2d — wraps nn.Conv2d with weight-only fake-quantization.

Weights: observed + frozen once in __init__ (static, no data needed).
Activations: NOT quantized here. Activation quantization is handled exclusively
by PreStubbedModule stubs placed at block inputs (hardware memory-read model).
Bias stays FP32 (standard fake-quant convention).
"""
from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from quantization.observer import BaseObserver, build_observer


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
        weight_observer: str = "min_max",
        weight_scheme: str = "symmetric",
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

        self.weight_observer: BaseObserver = build_observer(
            weight_observer, bits=weight_bits, scheme=weight_scheme
        )
        self.weight_observer.observe(self.weight)
        self.weight_observer.freeze()

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
        w_q = self.weight_observer.fake_quantize(self.weight)
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice(), mode=self.padding_mode)
            return F.conv2d(
                x, w_q, self.bias,
                self.stride, (0, 0), self.dilation, self.groups,
            )
        return F.conv2d(
            x, w_q, self.bias,
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
            f"groups={self.groups}, w_bits={self.weight_bits}"
        )
