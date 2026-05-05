"""Run a calibration pass to fill activation-observer statistics, then freeze them."""
from __future__ import annotations

from itertools import islice
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor, nn

from quantization.convert import collect_quant_modules
from quantization.observer import CALIBRATING


def calibrate(
    model: nn.Module,
    batches: Iterable,
    n_batches: int,
    input_fn: Callable[[object], Tensor] = lambda b: b,
    device: Optional[torch.device] = None,
) -> int:
    """Run up to n_batches forward passes with activation observers in CALIBRATING mode, then freeze.

    Covers every module carrying an `act_observer` — QuantConv2d (Phase 1), QuantLinear
    and QuantStub (Phase 3), and anything future that fits the same duck type.

    Args:
        model: model already converted via convert_model().
        batches: iterable yielding batches (whatever shape the loader uses).
        n_batches: how many batches to process for calibration.
        input_fn: maps a batch to the input tensor fed to the model. Default assumes the batch IS the tensor.
        device: if provided, inputs are moved here before forward.

    Returns:
        number of activation observers frozen.
    """
    quant_modules = collect_quant_modules(model)
    if not quant_modules:
        raise RuntimeError(
            "No quantized modules found. Did you forget to run convert_model?"
        )

    for qm in quant_modules:
        qm.act_observer.set_mode(CALIBRATING)

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

    for qm in quant_modules:
        qm.act_observer.freeze()

    return len(quant_modules)
