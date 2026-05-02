"""QuantSkipAdd — quantize the two input branches of a residual addition.

The residual `+` in `Block.forward` and `InvertedResidual.forward` is a Python
operator, not a Module, so it cannot be matched by `isinstance` in a convert
pass. We expose it as `SkipAdd` (a passthrough nn.Module that just returns
`main + skip`), which `Block.__init__` and `InvertedResidual.__init__` install
under `self.skip_add`. The convert pass swaps each `SkipAdd` for a
`QuantSkipAdd` carrying two independent `QuantStub` children — one per branch.

Each branch is fake-quantized with its own (s, z); the sum is FP32 (= INT32
accumulator on real hw). No output observer here — at every site we wrap the
post-add tensor's next consumer is either a QuantConv2d (its act_observer
requantizes) or a LayerNorm2D adjacent to the add (whose input is on the Q grid
when `--quant.stub-targets ln2d` is enabled), so an output stub on the sum
would be a strictly redundant fake-quant.
"""
from __future__ import annotations

from torch import Tensor, nn

from quantization.quant_stub import QuantStub


class SkipAdd(nn.Module):
    """Passthrough residual-add. Default identity for FP32 / no-quant runs."""

    def forward(self, main: Tensor, skip: Tensor) -> Tensor:
        return main + skip


class QuantSkipAdd(SkipAdd):
    """Residual-add with independent input-side fake-quant on each branch."""

    def __init__(self, *, stub_main: QuantStub, stub_skip: QuantStub):
        super().__init__()
        self.stub_main = stub_main
        self.stub_skip = stub_skip

    def forward(self, main: Tensor, skip: Tensor) -> Tensor:
        return self.stub_main(main) + self.stub_skip(skip)

    def extra_repr(self) -> str:
        return (
            f"main={type(self.stub_main.act_observer).__name__}, "
            f"skip={type(self.stub_skip.act_observer).__name__}"
        )

    @classmethod
    def from_skip_add(
        cls,
        _skip_add: SkipAdd,
        *,
        bits: int,
        observer: str,
        scheme: str,
    ) -> "QuantSkipAdd":
        """Build a QuantSkipAdd carrying two independent QuantStub instances.

        Each stub gets its own observer object — they accumulate and freeze
        independently, so `s_main` and `s_skip` are NOT shared.
        """
        stub_main = QuantStub(act_bits=bits, act_observer=observer, act_scheme=scheme)
        stub_skip = QuantStub(act_bits=bits, act_observer=observer, act_scheme=scheme)
        return cls(stub_main=stub_main, stub_skip=stub_skip)
