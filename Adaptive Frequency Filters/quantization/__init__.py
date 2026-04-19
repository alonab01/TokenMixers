"""Fake-quantization PTQ infrastructure for AFFNet.

Phase 1: Conv2d weights + input activations.
Phase 2: per-channel weight observer + percentile activation observer.
Phase 3: QuantLinear + inter-block QuantStubs (hardware-faithful data flow).

See CLAUDE.md Quantization section.
"""
from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear
from quantization.quant_stub import QuantStub, StubbedModule

__all__ = [
    "QuantConv2d",
    "QuantLinear",
    "QuantStub",
    "StubbedModule",
]
