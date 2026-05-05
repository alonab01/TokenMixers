"""Conv+BN folding for PTQ.

Fuses an `nn.Conv2d` followed by `nn.BatchNorm2d`/`nn.SyncBatchNorm` (in eval mode,
using running stats) into the Conv weights+bias, in-place. The BN is then deleted
from its parent `nn.Sequential` — `Sequential.forward` iterates `_modules.values()`,
so a removed entry is simply skipped while the remaining children retain their
insertion order.

Math (per output channel c):
    alpha[c]   = gamma[c] / sqrt(running_var[c] + eps)
    W'[c]      = alpha[c] * W[c]              (broadcast over Cin/groups, kH, kW)
    b'[c]      = alpha[c] * b_conv[c]         (b_conv = 0 if Conv had no bias)
                  + beta[c] - running_mean[c] * alpha[c]

Result: BN becomes the identity, Conv carries the BN affine transform. The fused
Conv has bias even if the original did not — folding always introduces a bias term
because BN's beta - mu*alpha is nonzero in general.

Caller must put the model in `.eval()` mode first so BN uses running stats. Folding
is destructive (modifies Conv weights and bias in-place).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

# nn.SyncBatchNorm inherits from _BatchNorm but NOT from BatchNorm2d, so list it
# explicitly. Plain BatchNorm2d covers the common single-GPU case.
_BN_TYPES: Tuple[type, ...] = (nn.BatchNorm2d, nn.SyncBatchNorm)


def _fold_pair(conv: nn.Conv2d, bn: nn.modules.batchnorm._BatchNorm) -> None:
    """In-place fold of `bn`'s affine + running stats into `conv`'s weights/bias."""
    if not bn.track_running_stats:
        raise ValueError(
            "fold_conv_bn_: BatchNorm must track running stats to be foldable."
        )
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("fold_conv_bn_: BatchNorm running stats are None.")

    eps = bn.eps
    running_mean = bn.running_mean.detach()
    running_var = bn.running_var.detach()
    gamma = (
        bn.weight.detach()
        if bn.weight is not None
        else torch.ones_like(running_var)
    )
    beta = (
        bn.bias.detach()
        if bn.bias is not None
        else torch.zeros_like(running_var)
    )

    alpha = gamma / torch.sqrt(running_var + eps)        # (Cout,)

    # Fold weights — alpha multiplies each output channel uniformly across Cin/kH/kW.
    W = conv.weight.data                                  # (Cout, Cin/groups, kH, kW)
    W.mul_(alpha.view(-1, 1, 1, 1))

    # Fold bias — Conv may not have one; create it if needed.
    b_conv = (
        conv.bias.data
        if conv.bias is not None
        else torch.zeros_like(running_mean)
    )
    b_fused = alpha * b_conv + (beta - running_mean * alpha)
    if conv.bias is None:
        conv.bias = nn.Parameter(b_fused.clone())
    else:
        conv.bias.data.copy_(b_fused)


def fold_conv_bn_(model: nn.Module) -> int:
    """Walk `model`, fuse every (Conv2d, BN) adjacent pair found inside an
    `nn.Sequential` container. Deletes the BN child after folding and returns
    the number of fused pairs.

    Targets `nn.BatchNorm2d` and `nn.SyncBatchNorm`. GroupNorm/LayerNorm/InstanceNorm
    are NOT foldable (their stats depend on input, not training-time running stats).

    The model should be in `.eval()` mode so BN uses running stats — otherwise
    folding bakes the wrong stats. We do not enforce this to avoid surprising the
    caller; the contract is documented.
    """
    n_fused = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for i in range(len(children) - 1):
            name_a, mod_a = children[i]
            name_b, mod_b = children[i + 1]
            if isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, _BN_TYPES):
                _fold_pair(mod_a, mod_b)
                del module._modules[name_b]
                n_fused += 1
    return n_fused
