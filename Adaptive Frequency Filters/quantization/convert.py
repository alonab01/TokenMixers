"""Walk a model and install quantization modules.

Design:
  - QuantConv2d / QuantLinear: weight-only fake-quantization. No act_observer.
  - PreStubbedModule: wraps a block and quantizes its INPUT before it runs.
    This models the int8 memory read at the start of each hardware fused kernel.
    Activation quantization lives exclusively in these stubs.

Phase 1/2: nn.Conv2d -> QuantConv2d (weight-only, respects skip-list).
Phase 3 additions (opt-in):
  (a) Linear -> QuantLinear (weight-only).
  (b) Block-level pre-stubs via PreStubbedModule: one stub per hardware boundary.
      Default targets: InvertedResidual, Block, GlobalPool.
      AFFBlock is excluded — its forward passes directly to Block[0] with no
      intervening transform, so wrapping both would double-quant the same tensor.
  (c) Standalone ConvLayer pre-stubs (Phase 2 of _insert_stubs): wraps every
      ConvLayer that is NOT already inside a wrapped block. Catches conv_1x1_exp
      and each AFFBlock.conv_proj, giving a complete set of hardware boundaries.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type

from torch import nn

from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear
from quantization.quant_stub import QuantStub, PreStubbedModule


def _is_skipped(path: str, skip_list: Sequence[str]) -> bool:
    for s in skip_list:
        if not s:
            continue
        if path == s or path.startswith(s + "."):
            return True
    return False


def _get_parent_and_attr(model: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def _set_submodule(parent: nn.Module, attr: str, new: nn.Module) -> None:
    if attr.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attr)] = new
    else:
        setattr(parent, attr, new)


# ---------- Conv2d swap (Phase 1/2) ---------- #


def convert_model(
    model: nn.Module,
    weight_bits: int,
    act_bits: int,
    weight_observer: str = "min_max",
    act_observer: str = "min_max",
    weight_scheme: str = "symmetric",
    act_scheme: str = "asymmetric",
    skip_modules: Sequence[str] = (),
    # Phase 3 additions (all opt-in) #
    quantize_linear: bool = False,
    skip_linears: Sequence[str] = (),
    insert_stubs: bool = False,
    skip_stubs: Sequence[str] = (),
    stub_bits: Optional[int] = None,
    stub_observer: str = "percentile",
    stub_scheme: str = "asymmetric",
    stub_targets: Optional[Sequence[Type[nn.Module]]] = None,
) -> Tuple[nn.Module, List[str]]:
    """In-place swap of nn.Conv2d -> QuantConv2d (weight-only) and optionally
    Linear -> QuantLinear (weight-only) and PreStubbedModule wrapping at block
    boundaries for hardware-faithful activation quantization.

    act_bits / act_observer / act_scheme are passed to stubs (not to Conv/Linear).

    Returns (model, list_of_swapped_conv_paths).
    """
    conv_cfg = dict(
        weight_bits=weight_bits,
        weight_observer=weight_observer,
        weight_scheme=weight_scheme,
    )

    swapped_convs = _swap_conv2d(model, conv_cfg, skip_modules)

    if quantize_linear:
        _swap_linears(model, conv_cfg, skip_linears)

    if insert_stubs:
        _insert_stubs(
            model,
            stub_bits=stub_bits if stub_bits is not None else act_bits,
            stub_observer=stub_observer,
            stub_scheme=stub_scheme,
            skip_stubs=skip_stubs,
            targets=stub_targets,
        )

    return model, swapped_convs


def _swap_conv2d(
    model: nn.Module,
    cfg: dict,
    skip_modules: Sequence[str],
) -> List[str]:
    targets: List[Tuple[str, nn.Conv2d]] = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if isinstance(module, QuantConv2d):
            continue
        if _is_skipped(path, skip_modules):
            continue
        targets.append((path, module))

    swapped: List[str] = []
    for path, conv in targets:
        new = QuantConv2d.from_conv2d(conv, **cfg)
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, new)
        swapped.append(path)
    return swapped


# ---------- Linear swap (Phase 3) ---------- #


def _swap_linears(
    model: nn.Module,
    cfg: dict,
    skip_linears: Sequence[str],
) -> List[str]:
    """Swap nn.Linear and AFFNet LinearLayer -> QuantLinear (weight-only).
    Skips channel_first LinearLayers (those route through F.conv2d)."""
    from affnet.layers.linear_layer import GroupLinear, LinearLayer

    targets: List[Tuple[str, nn.Module]] = []
    for path, module in model.named_modules():
        if isinstance(module, QuantLinear):
            continue
        if isinstance(module, GroupLinear):
            continue
        if isinstance(module, LinearLayer):
            if getattr(module, "channel_first", False):
                continue
        elif not isinstance(module, nn.Linear):
            continue
        if _is_skipped(path, skip_linears):
            continue
        targets.append((path, module))

    swapped: List[str] = []
    for path, lin in targets:
        new = QuantLinear.from_linear(lin, **cfg)
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, new)
        swapped.append(path)
    return swapped


# ---------- Stub insertion (Phase 3) ---------- #


def _default_stub_targets() -> Tuple[Type[nn.Module], ...]:
    """Module types whose INPUTS are pre-quantized (hardware INT8 memory read).

    InvertedResidual (MBConv): one fused kernel — expand + DW + project.
    Block: the post-AFNO2D residual boundary. LayerNorm runs before Block.mlp,
           so Block + IR stubs quantize two DIFFERENT tensors (no double-quant).
    GlobalPool: the final spatial reduction before the classifier head.

    AFFBlock is intentionally excluded: its forward passes the input directly to
    Block[0] without any transform, so wrapping both AFFBlock and Block would
    double-quant the same tensor.

    Standalone ConvLayers not inside these targets (e.g., conv_1x1_exp,
    AFFBlock.conv_proj) are handled by the Phase 2 ConvLayer pass in _insert_stubs.
    """
    from affnet.layers.global_pool import GlobalPool
    from affnet.modules.aff_block import Block
    from affnet.modules.mobilenetv2 import InvertedResidual, InvertedResidualSE

    return (InvertedResidual, InvertedResidualSE, Block, GlobalPool)


def _insert_stubs(
    model: nn.Module,
    stub_bits: int,
    stub_observer: str,
    stub_scheme: str,
    skip_stubs: Sequence[str],
    targets: Optional[Sequence[Type[nn.Module]]] = None,
) -> List[str]:
    if targets is None:
        targets = _default_stub_targets()
    target_types = tuple(targets)

    # ---- Phase 1: wrap block-level targets (deepest path first) ----
    # Collect first; mutating named_modules mid-iteration is unsafe.
    hits: List[Tuple[str, nn.Module]] = []
    for path, module in model.named_modules():
        if path == "":
            continue
        if isinstance(module, (PreStubbedModule, QuantStub, QuantConv2d, QuantLinear)):
            continue
        if not isinstance(module, target_types):
            continue
        if _is_skipped(path, skip_stubs):
            continue
        hits.append((path, module))

    # Deepest first: once a module is wrapped, StubbedModule doesn't proxy
    # arbitrary attributes, so ancestor paths become untraversable.
    hits.sort(key=lambda pm: pm[0].count("."), reverse=True)

    wrapped: List[str] = []
    for path, mod in hits:
        stub = QuantStub(act_bits=stub_bits, act_observer=stub_observer, act_scheme=stub_scheme)
        wrapper = PreStubbedModule(inner=mod, stub=stub)
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, wrapper)
        wrapped.append(path)

    # ---- Phase 2: wrap standalone ConvLayers not inside wrapped blocks ----
    # Catches conv_1x1_exp, AFFBlock.conv_proj, and any other top-level ConvLayer
    # that is not already inside a PreStubbedModule's inner.
    try:
        from affnet.layers.conv_layer import ConvLayer
    except ImportError:
        return wrapped  # non-AFFNet model, skip Phase 2

    # Build excluded prefixes: paths that are inside any already-wrapped module's .inner
    excluded_prefixes = {
        path + ".inner"
        for path, module in model.named_modules()
        if isinstance(module, PreStubbedModule)
    }

    def _is_inside_wrapped(p: str) -> bool:
        return any(p == ep or p.startswith(ep + ".") for ep in excluded_prefixes)

    conv_hits: List[Tuple[str, nn.Module]] = []
    for path, module in model.named_modules():
        if path == "":
            continue
        if isinstance(module, (PreStubbedModule, QuantStub)):
            continue
        if not isinstance(module, ConvLayer):
            continue
        if _is_skipped(path, skip_stubs):
            continue
        if _is_inside_wrapped(path):
            continue
        conv_hits.append((path, module))

    conv_hits.sort(key=lambda pm: pm[0].count("."), reverse=True)

    for path, mod in conv_hits:
        stub = QuantStub(act_bits=stub_bits, act_observer=stub_observer, act_scheme=stub_scheme)
        wrapper = PreStubbedModule(inner=mod, stub=stub)
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, wrapper)
        wrapped.append(path)

    return wrapped


# ---------- Inventories ---------- #


def collect_quant_convs(model: nn.Module) -> List[QuantConv2d]:
    return [m for m in model.modules() if isinstance(m, QuantConv2d)]


def collect_quant_modules(model: nn.Module) -> List[nn.Module]:
    """All modules whose activation observer participates in calibration.

    With the current architecture — stubs own all activation Q — this returns
    only QuantStub instances. Duck-typed on .act_observer for future extensibility.
    """
    from quantization.observer import BaseObserver
    return [
        m for m in model.modules()
        if isinstance(getattr(m, "act_observer", None), BaseObserver)
    ]


def collect_quant_linears(model: nn.Module) -> List[QuantLinear]:
    return [m for m in model.modules() if isinstance(m, QuantLinear)]


def collect_quant_stubs(model: nn.Module) -> List[QuantStub]:
    return [m for m in model.modules() if isinstance(m, QuantStub)]
