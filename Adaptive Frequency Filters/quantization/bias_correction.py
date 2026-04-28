"""Bias correction — compensate the systematic output mean shift from weight quantization.

For a Conv2d / Linear layer with FP32 weight W, quantized weight W_q, bias b:
    y     = layer(x, W)   + b
    y_q   = layer(x, W_q) + b
    E[y - y_q] = E[ layer(x, W - W_q) ] approx layer(E[x], W - W_q)

We add this expected output shift to the bias:
    b' = b + layer(E[x], W - W_q)

`E[x]` is measured over a calibration pass with weights AND activations already quantized
(so we capture the actual distribution the deployed network sees).

Standard PTQ technique. Cheap, zero training, often +1-2 pp at low bit-widths.
"""
from __future__ import annotations

from itertools import islice
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear


class _InputMeanHook:
    """Forward hook on a module's act_observer, capturing per-input-channel running mean
    of its OUTPUT (which is the post-act-quant tensor fed into the parent conv/linear)."""

    def __init__(self):
        self.running_mean: Optional[Tensor] = None
        self.n_batches: int = 0

    def __call__(self, module, args, output):
        x = output.detach()
        if x.dim() == 4:
            x_mean = x.mean(dim=(0, 2, 3))
        elif x.dim() == 3:
            x_mean = x.mean(dim=(0, 1))
        elif x.dim() == 2:
            x_mean = x.mean(dim=0)
        else:
            raise RuntimeError(
                f"_InputMeanHook: unexpected input dim {x.dim()} for shape {tuple(x.shape)}"
            )
        if self.running_mean is None:
            self.running_mean = x_mean.clone()
        else:
            n = self.n_batches
            self.running_mean.mul_(n / (n + 1)).add_(x_mean / (n + 1))
        self.n_batches += 1


def _compute_conv_delta_bias(qc: QuantConv2d, input_mean: Tensor) -> Tensor:
    """delta_b[c_out] = sum_{c_in, kh, kw} (W - W_q)[c_out, c_in, kh, kw] * E[x[c_in]]
    Implemented as a constant-input convolution to handle grouped convs naturally."""
    W = qc.weight.detach()
    W_q = qc.weight_observer.fake_quantize(W).detach()
    W_diff = W - W_q  # (Cout, Cin/groups, kH, kW)
    Cin_full = W_diff.shape[1] * qc.groups
    kH, kW = W_diff.shape[-2:]
    if input_mean.numel() != Cin_full:
        raise RuntimeError(
            f"input_mean size {input_mean.numel()} != expected Cin {Cin_full} "
            f"for Conv with groups={qc.groups}"
        )
    # constant input of channel means, shape (1, Cin, kH, kW)
    x_const = input_mean.view(1, Cin_full, 1, 1).expand(1, Cin_full, kH, kW).contiguous()
    out = F.conv2d(x_const, W_diff, bias=None, stride=1, padding=0,
                   dilation=1, groups=qc.groups)
    return out.view(-1)  # (Cout,)


def _compute_linear_delta_bias(ql: QuantLinear, input_mean: Tensor) -> Tensor:
    W = ql.weight.detach()
    W_q = ql.weight_observer.fake_quantize(W).detach()
    W_diff = W - W_q
    if input_mean.numel() != W_diff.shape[1]:
        raise RuntimeError(
            f"input_mean size {input_mean.numel()} != in_features {W_diff.shape[1]}"
        )
    return W_diff @ input_mean


def apply_bias_correction(
    model: nn.Module,
    batches: Iterable,
    n_batches: int,
    input_fn: Callable[[object], Tensor] = lambda b: b,
    device: Optional[torch.device] = None,
) -> int:
    """Run a forward pass over calibration data with both weights and activations
    already quantized, capture per-Conv / per-Linear input means, then compensate
    the bias of each Conv/Linear by the expected output shift due to weight quant.

    Returns the number of modules whose bias was adjusted.
    """
    targets: List[Tuple[nn.Module, _InputMeanHook]] = []
    handles = []
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            hook = _InputMeanHook()
            h = m.act_observer.register_forward_hook(hook)
            targets.append((m, hook))
            handles.append(h)

    if not targets:
        return 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in islice(batches, n_batches):
                x = input_fn(batch)
                if device is not None:
                    x = x.to(device)
                model(x)
    finally:
        if was_training:
            model.train()
        for h in handles:
            h.remove()

    n_corrected = 0
    for m, hook in targets:
        if hook.running_mean is None:
            continue
        if isinstance(m, QuantConv2d):
            delta_b = _compute_conv_delta_bias(m, hook.running_mean)
            target_shape = (m.weight.shape[0],)
        else:  # QuantLinear
            delta_b = _compute_linear_delta_bias(m, hook.running_mean)
            target_shape = (m.weight.shape[0],)
        if m.bias is not None:
            m.bias.data.add_(delta_b.to(m.bias.dtype).to(m.bias.device))
        else:
            m.bias = nn.Parameter(delta_b.to(m.weight.dtype), requires_grad=False)
        n_corrected += 1
    return n_corrected
