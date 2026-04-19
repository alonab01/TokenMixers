"""Walk a model and install quantization modules.

Phase 1/2: nn.Conv2d -> QuantConv2d (respects skip-list).
Phase 3: also (a) optionally swap Linear/LinearLayer -> QuantLinear,
         and (b) optionally wrap block-boundary modules with a QuantStub so the
         output stays on the Q grid between blocks (hardware-faithful simulation).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Type

from torch import nn

from quantization.quant_conv import QuantConv2d
from quantization.quant_linear import QuantLinear
from quantization.quant_stub import QuantStub, StubbedModule


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
    """In-place swap of nn.Conv2d → QuantConv2d and (optionally) Linear → QuantLinear
    and StubbedModule wrapping at block boundaries.

    Returns (model, list_of_swapped_conv_paths) for backward compat; the full audit
    is available via collect_quant_modules(model).
    """
    conv_cfg = dict(
        weight_bits=weight_bits,
        act_bits=act_bits,
        weight_observer=weight_observer,
        act_observer=act_observer,
        weight_scheme=weight_scheme,
        act_scheme=act_scheme,
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
    """Swap nn.Linear and AFFNet LinearLayer → QuantLinear. Skips channel_first LinearLayers
    (those route through F.conv2d and should be handled as convs)."""
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
    """Module types whose outputs should be re-quantized by default.

    Chosen to catch every block-output boundary on AFFNet-ET:
      - InvertedResidual:  residual-add + bare sequential convs in layers 1/2
                           and inside every Block.mlp
      - Block:             after AFNO2D filter + residual-add in layers 3-5
      - AFFBlock:          stage output before feeding the next layer
      - GlobalPool:        pooled vector before classifier (belt-and-suspenders)
    """
    from affnet.layers.global_pool import GlobalPool
    from affnet.modules.aff_block import AFFBlock, Block
    from affnet.modules.mobilenetv2 import InvertedResidual, InvertedResidualSE

    return (InvertedResidual, InvertedResidualSE, Block, AFFBlock, GlobalPool)


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

    # Collect first; mutating named_modules mid-iteration is unsafe.
    hits: List[Tuple[str, nn.Module]] = []
    for path, module in model.named_modules():
        if path == "":
            continue
        if isinstance(module, (StubbedModule, QuantStub, QuantConv2d, QuantLinear)):
            continue
        if not isinstance(module, target_types):
            continue
        if _is_skipped(path, skip_stubs):
            continue
        hits.append((path, module))

    # Wrap deepest paths first so an ancestor stays traversable while we drill into its children.
    # (A StubbedModule doesn't proxy arbitrary attributes of its `inner`, so once we wrap
    # e.g. an AFFBlock, we can no longer reach `affblock.global_rep.0` by attribute walk.)
    hits.sort(key=lambda pm: pm[0].count("."), reverse=True)

    wrapped: List[str] = []
    for path, mod in hits:
        stub = QuantStub(act_bits=stub_bits, act_observer=stub_observer, act_scheme=stub_scheme)
        wrapper = StubbedModule(inner=mod, stub=stub)
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, wrapper)
        wrapped.append(path)
    return wrapped


# ---------- Inventories ---------- #


def collect_quant_convs(model: nn.Module) -> List[QuantConv2d]:
    return [m for m in model.modules() if isinstance(m, QuantConv2d)]


def collect_quant_modules(model: nn.Module) -> List[nn.Module]:
    """All modules whose activation observer participates in the calibration pass.

    Phase 1: QuantConv2d.  Phase 3: + QuantLinear + QuantStub.
    Generalized by duck-typing on `.act_observer` to stay resilient as we add new kinds.
    """
    out: List[nn.Module] = []
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantStub)):
            out.append(m)
            continue
        # Future-proofing: accept anything with an act_observer attr that is a BaseObserver.
        if hasattr(m, "act_observer"):
            from quantization.observer import BaseObserver
            if isinstance(getattr(m, "act_observer"), BaseObserver):
                out.append(m)
    return out


def collect_quant_linears(model: nn.Module) -> List[QuantLinear]:
    return [m for m in model.modules() if isinstance(m, QuantLinear)]


def collect_quant_stubs(model: nn.Module) -> List[QuantStub]:
    return [m for m in model.modules() if isinstance(m, QuantStub)]
