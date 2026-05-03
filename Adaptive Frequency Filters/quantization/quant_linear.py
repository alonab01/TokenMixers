"""QuantLinear — wraps nn.Linear / AFFNet LinearLayer with weight + activation fake-quantization.

Mirrors QuantConv2d: weight observer runs observe+freeze once in __init__, and an
own act_observer quantizes the input in forward. Bias stays FP32.

Weight shape is (out_features, in_features); per_channel_min_max with axis=0 matches
exactly (each output feature gets its own scale), so no extra reshape logic is needed.
"""
from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn

from quantization.observer import BaseObserver, build_observer


class QuantLinear(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        weight_bits: int,
        weight_observer: str = "min_max",
        weight_scheme: str = "symmetric",
        act_bits: int = 8,
        act_observer: str = "min_max",
        act_scheme: str = "asymmetric",
    ):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.out_features, self.in_features = self.weight.shape
        self.weight_bits = weight_bits
        self.act_bits = act_bits

        self.weight_observer: BaseObserver = build_observer(
            weight_observer, bits=weight_bits, scheme=weight_scheme, axis=0,
        )
        self.weight_observer.observe(self.weight)
        self.weight_observer.freeze()

        # Linear input: channel/feature axis is the last dim, regardless of rank.
        self.act_observer: BaseObserver = build_observer(
            act_observer, bits=act_bits, scheme=act_scheme, axis=-1,
        )

    @classmethod
    def from_linear(cls, linear: nn.Module, **quant_cfg) -> "QuantLinear":
        """Factory accepting nn.Linear OR AFFNet's LinearLayer (both own .weight / .bias)."""
        from affnet.layers.linear_layer import GroupLinear, LinearLayer

        if isinstance(linear, GroupLinear):
            raise TypeError("QuantLinear does not support GroupLinear (3D weight tensor).")
        if isinstance(linear, LinearLayer) and getattr(linear, "channel_first", False):
            raise TypeError(
                "QuantLinear does not support LinearLayer(channel_first=True); "
                "that path routes through F.conv2d and should be quantized as a Conv2d."
            )
        w = linear.weight.data
        b = linear.bias.data if linear.bias is not None else None
        return cls(weight=w, bias=b, **quant_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act_observer(x)
        w_q = self.weight_observer.fake_quantize(self.weight)
        return F.linear(x, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"w_bits={self.weight_bits}, a_bits={self.act_bits}"
        )
