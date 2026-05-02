"""Walk a model and install quantization modules.

Design:
  - QuantConv2d / QuantLinear: own weight + activation observers. Each Conv/Linear
    quantizes its own input — that is the dominant activation-Q path.
  - QuantStub (via PreStubbedModule): a flexible tool for additional FP32
    boundaries the user wants on the Q grid (LayerNorm, BatchNorm, activation
    fns, AFNO2D output, GlobalPool, ...). Per-type config via StubConfig.

`insert_stubs=True` with no targets is a no-op (warning). Default empty:
stubs are explicitly opt-in — Conv/Linear self-quantization is the default
activation-Q surface.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

from torch import nn

from quantization.quant_conv import QuantConv2d
from quantization.quant_hardswish import QuantHardswish
from quantization.quant_linear import QuantLinear
from quantization.quant_skip_add import QuantSkipAdd, SkipAdd
from quantization.quant_stub import PreStubbedModule, QuantStub, StubConfig
from utils import logger


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


# ---------- Conv2d / Linear swap ---------- #


def convert_model(
    model: nn.Module,
    weight_bits: int,
    act_bits: int,
    weight_observer: str = "min_max",
    act_observer: str = "min_max",
    weight_scheme: str = "symmetric",
    act_scheme: str = "asymmetric",
    skip_modules: Sequence[str] = (),
    quantize_linear: bool = False,
    skip_linears: Sequence[str] = (),
    quantize_acts: bool = False,
    skip_acts: Sequence[str] = (),
    quantize_residuals: bool = False,
    skip_residuals: Sequence[str] = (),
    residual_observer: str = "min_max",
    residual_scheme: str = "asymmetric",
    residual_bits: Optional[int] = None,
    insert_stubs: bool = False,
    skip_stubs: Sequence[str] = (),
    stub_target_configs: Optional[Dict[Type[nn.Module], StubConfig]] = None,
    # Back-compat scalar fallbacks: only consulted when stub_target_configs is None
    # and stub_targets is provided. They build a uniform-config dict.
    stub_bits: Optional[int] = None,
    stub_observer: str = "percentile",
    stub_scheme: str = "asymmetric",
    stub_targets: Optional[Sequence[Type[nn.Module]]] = None,
) -> Tuple[nn.Module, List[str]]:
    """In-place swap of nn.Conv2d -> QuantConv2d (weight + input act) and optionally
    nn.Linear / LinearLayer -> QuantLinear (weight + input act). Optional per-type
    PreStubbedModule wrapping for non-Conv/Linear FP32 boundaries.

    Returns (model, list_of_swapped_conv_paths).
    """
    quant_cfg = dict(
        weight_bits=weight_bits,
        weight_observer=weight_observer,
        weight_scheme=weight_scheme,
        act_bits=act_bits,
        act_observer=act_observer,
        act_scheme=act_scheme,
    )

    swapped_convs = _swap_conv2d(model, quant_cfg, skip_modules)

    if quantize_linear:
        _swap_linears(model, quant_cfg, skip_linears)

    if quantize_acts:
        _swap_acts(model, quant_cfg, skip_acts)

    if quantize_residuals:
        _swap_skip_adds(
            model,
            bits=residual_bits if residual_bits is not None else act_bits,
            observer=residual_observer,
            scheme=residual_scheme,
            skip_residuals=skip_residuals,
        )

    if insert_stubs:
        cfgs = stub_target_configs
        if cfgs is None and stub_targets is not None:
            fallback = StubConfig(
                bits=stub_bits if stub_bits is not None else act_bits,
                observer=stub_observer,
                scheme=stub_scheme,
            )
            cfgs = {t: fallback for t in stub_targets}
        if cfgs:
            _insert_stubs(model, target_configs=cfgs, skip_stubs=skip_stubs)
        else:
            logger.warning(
                "[quant] insert_stubs=True but no stub_target_configs / stub_targets "
                "provided — no stubs inserted."
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


def _swap_linears(
    model: nn.Module,
    cfg: dict,
    skip_linears: Sequence[str],
) -> List[str]:
    """Swap nn.Linear and AFFNet LinearLayer -> QuantLinear (weight + input act).
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


# ---------- Activation swap (Swish/HardSwish -> QuantHardswish) ---------- #


def _swap_acts(
    model: nn.Module,
    cfg: dict,
    skip_acts: Sequence[str],
) -> List[str]:
    """Replace every nn.SiLU / nn.Hardswish (and their AFFNet wrappers Swish /
    Hardswish, which subclass them) with QuantHardswish — a function-swap +
    activation-input quantization in one op. Models the int8 deployment story
    where the Conv output is requantized before the nonlinearity reads it,
    and the nonlinearity itself is the INT8-friendly HardSwish.
    """
    targets: List[Tuple[str, nn.Module]] = []
    for path, module in model.named_modules():
        if isinstance(module, QuantHardswish):
            continue
        if not isinstance(module, (nn.SiLU, nn.Hardswish)):
            continue
        if _is_skipped(path, skip_acts):
            continue
        targets.append((path, module))

    swapped: List[str] = []
    for path, _act in targets:
        new = QuantHardswish(
            act_bits=cfg["act_bits"],
            act_observer=cfg["act_observer"],
            act_scheme=cfg["act_scheme"],
        )
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, new)
        swapped.append(path)
    return swapped


# ---------- Residual-add swap (SkipAdd -> QuantSkipAdd) ---------- #


def _swap_skip_adds(
    model: nn.Module,
    *,
    bits: int,
    observer: str,
    scheme: str,
    skip_residuals: Sequence[str],
) -> List[str]:
    """Replace every `SkipAdd` (the passthrough residual-add wrapper installed by
    `Block.__init__` / `InvertedResidual.__init__`) with `QuantSkipAdd`, which
    fake-quantizes each input branch with its own (s, z). The two branches are
    independent — `stub_main` and `stub_skip` are distinct QuantStub instances.
    Already-swapped sites (`QuantSkipAdd` is a SkipAdd subclass) are skipped.
    """
    targets: List[Tuple[str, SkipAdd]] = []
    for path, module in model.named_modules():
        if isinstance(module, QuantSkipAdd):
            continue
        if not isinstance(module, SkipAdd):
            continue
        if _is_skipped(path, skip_residuals):
            continue
        targets.append((path, module))

    swapped: List[str] = []
    for path, sa in targets:
        new = QuantSkipAdd.from_skip_add(
            sa, bits=bits, observer=observer, scheme=scheme
        )
        parent, attr = _get_parent_and_attr(model, path)
        _set_submodule(parent, attr, new)
        swapped.append(path)
    return swapped


# ---------- Stub insertion (per-type) ---------- #


def _match_target(module: nn.Module, target_configs: Dict[Type[nn.Module], StubConfig]) -> Optional[Type[nn.Module]]:
    """Find the first key in target_configs that `module` is an instance of.

    Insertion order of the dict determines priority when a module would match
    multiple keys (e.g. a class and one of its bases).
    """
    for cls in target_configs:
        if isinstance(module, cls):
            return cls
    return None


def _insert_stubs(
    model: nn.Module,
    target_configs: Dict[Type[nn.Module], StubConfig],
    skip_stubs: Sequence[str] = (),
) -> List[str]:
    """Wrap every module matching a key in target_configs with PreStubbedModule.

    Walk the model once; for each module find the first matching type in
    target_configs (isinstance, so subclasses match), build a QuantStub from the
    matched type's StubConfig, and replace the module with PreStubbedModule(inner=mod, stub=stub).

    Skips any module already wrapped by a stub or that IS itself a quant module
    (PreStubbedModule, QuantStub, QuantConv2d, QuantLinear) — those have their
    own activation-Q story.
    """
    if not target_configs:
        return []

    hits: List[Tuple[str, nn.Module, Type[nn.Module]]] = []
    for path, module in model.named_modules():
        if path == "":
            continue
        if isinstance(module, (PreStubbedModule, QuantStub, QuantConv2d, QuantLinear, SkipAdd)):
            continue
        if _is_skipped(path, skip_stubs):
            continue
        matched = _match_target(module, target_configs)
        if matched is None:
            continue
        hits.append((path, module, matched))

    # Deepest first: once a module is wrapped, PreStubbedModule doesn't proxy
    # arbitrary attributes, so ancestor paths become untraversable.
    hits.sort(key=lambda pmt: pmt[0].count("."), reverse=True)

    wrapped: List[str] = []
    for path, mod, matched_type in hits:
        cfg = target_configs[matched_type]
        stub = QuantStub(act_bits=cfg.bits, act_observer=cfg.observer, act_scheme=cfg.scheme)
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

    Duck-typed on .act_observer: covers QuantConv2d, QuantLinear, QuantStub, and
    any future module that wears a BaseObserver under that name.
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


def collect_quant_hardswish(model: nn.Module) -> List[QuantHardswish]:
    return [m for m in model.modules() if isinstance(m, QuantHardswish)]


def collect_quant_skip_adds(model: nn.Module) -> List[QuantSkipAdd]:
    return [m for m in model.modules() if isinstance(m, QuantSkipAdd)]
