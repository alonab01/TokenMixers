"""Entry point for Phase 1 PTQ: convert Conv2d → QuantConv2d, calibrate, evaluate.

Mirrors main_eval.py but inserts the quant pipeline between model load and Evaluator.
Appends one row per run to --quant.results-csv.

Example:
    python main_quant.py \
        --common.config-file resource/config/imagenet_et/config.yaml \
        --common.results-loc results/ \
        --model.classification.pretrained resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt \
        --quant.enabled --quant.weight-bits 8 --quant.activation-bits 8
"""
from __future__ import annotations

import csv
import multiprocessing
import os
import time

import torch

from torch.utils.data import DataLoader, Dataset, RandomSampler

from affnet import get_model
from data import create_eval_loader
from data.datasets import evaluation_datasets
from engine import Evaluator
from options.opts import get_eval_arguments
from utils import logger
from utils.common_utils import create_directories, device_setup, move_to_device
from utils.ddp_utils import distributed_init, is_master
from utils.tensor_utils import image_size_from_opts

from quantization.bias_correction import apply_bias_correction
from quantization.calibrate import calibrate
from quantization.convert import (
    collect_quant_convs,
    collect_quant_hardswish,
    collect_quant_linears,
    collect_quant_modules,
    collect_quant_skip_adds,
    collect_quant_stubs,
    convert_model,
)
from quantization.fuse import fold_conv_bn_
from quantization.quant_stub import StubConfig

from typing import Dict, Type
import importlib

# Phase 5: registry of stub target classes addressable by short name.
# String form "module:Class" is resolved lazily so we don't import affnet
# modules at script import time (avoids circular-ish risks).
_STUB_TARGET_REGISTRY: Dict[str, str] = {
    "ln2d":       "affnet.layers.normalization.layer_norm:LayerNorm2D_NCHW",
    "afno2d":     "affnet.modules.aff_block:AFNO2D_channelfirst",
    "block":      "affnet.modules.aff_block:Block",
    "affblock":   "affnet.modules.aff_block:AFFBlock",
    # Phase 8c: leaf-level targets (Swish/SiLU and the model's GlobalPool head).
    "swish":      "affnet.layers.activation.swish:Swish",
    "globalpool": "affnet.layers.global_pool:GlobalPool",
}


def _resolve_target(name: str) -> Type:
    spec = _STUB_TARGET_REGISTRY[name]
    mod_name, cls_name = spec.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def _build_stub_targets(target_csv: str, cfg: StubConfig) -> Dict[Type, StubConfig]:
    """Build a {nn.Module subclass: StubConfig} dict from a comma-separated short-name list.

    All targets share the same `cfg`. For per-target overrides, use
    `_build_stub_targets_from_config`.
    """
    if not target_csv.strip():
        return {}
    out: Dict[Type, StubConfig] = {}
    for name in [t.strip() for t in target_csv.split(",") if t.strip()]:
        if name not in _STUB_TARGET_REGISTRY:
            raise ValueError(
                f"unknown --quant.stub-targets entry '{name}'; "
                f"valid: {sorted(_STUB_TARGET_REGISTRY)}"
            )
        out[_resolve_target(name)] = cfg
    return out


def _build_stub_targets_from_config(
    config_csv: str,
    default_cfg: StubConfig,
) -> Dict[Type, StubConfig]:
    """Parse `--quant.stub-config` per-target override format.

    Format: 'name=observer[:bits[:scheme]],...'
    Examples:
        'ln2d=min_max'                                       (observer override only)
        'ln2d=min_max,afno2d=percentile'
        'ln2d=min_max:8:asymmetric,afno2d=histogram:8:asymmetric'

    Bits/scheme default to `default_cfg`'s values when omitted. Empty input -> {}.
    """
    if not config_csv.strip():
        return {}
    out: Dict[Type, StubConfig] = {}
    for entry in [t.strip() for t in config_csv.split(",") if t.strip()]:
        if "=" not in entry:
            raise ValueError(
                f"--quant.stub-config entry '{entry}' missing '='; "
                f"format: 'name=observer[:bits[:scheme]]'"
            )
        name, spec = entry.split("=", 1)
        name = name.strip()
        if name not in _STUB_TARGET_REGISTRY:
            raise ValueError(
                f"unknown stub-config target '{name}'; "
                f"valid: {sorted(_STUB_TARGET_REGISTRY)}"
            )
        parts = spec.split(":")
        observer = parts[0].strip() or default_cfg.observer
        bits = int(parts[1]) if len(parts) > 1 and parts[1].strip() else default_cfg.bits
        scheme = parts[2].strip() if len(parts) > 2 and parts[2].strip() else default_cfg.scheme
        out[_resolve_target(name)] = StubConfig(bits=bits, observer=observer, scheme=scheme)
    return out


class _TupleIndexAdapter(Dataset):
    """Wraps AFFNet datasets whose __getitem__ expects (crop_h, crop_w, int_idx)."""

    def __init__(self, inner: Dataset, crop_h: int, crop_w: int):
        self.inner = inner
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int):
        return self.inner[(self.crop_h, self.crop_w, idx)]


def _build_calib_loader(opts, batch_size: int, seed: int) -> DataLoader:
    """Shuffled calibration loader — covers classes uniformly (ImageNet val is class-sorted)."""
    eval_ds = evaluation_datasets(opts)
    crop_h, crop_w = image_size_from_opts(opts)
    adapter = _TupleIndexAdapter(eval_ds, crop_h=crop_h, crop_w=crop_w)

    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(adapter, generator=g)

    n_workers = max(0, getattr(opts, "dataset.workers", 0))
    return DataLoader(
        adapter,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=False,
    )


class _RecordingEvaluator(Evaluator):
    """Evaluator subclass that captures final top-1/top-5 after run(). Minimally invasive."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_metrics = {}

    def eval_fn_image(self, model):
        from metrics import Statistics, metric_monitor
        from common import DEFAULT_LOG_FREQ
        from engine.utils import get_batch_size

        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)
        stats = Statistics(metric_names=self.metric_names, is_master_node=self.is_master_node)
        model.eval()
        with torch.no_grad():
            start = time.time()
            total = len(self.eval_loader)
            processed = 0
            for batch_id, batch in enumerate(self.eval_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)
                samples, targets = batch["samples"], batch["targets"]
                bs = get_batch_size(samples)
                pred = model(samples)
                processed += bs
                m = metric_monitor(
                    self.opts, pred_label=pred, target_label=targets,
                    loss=torch.tensor(0.0, device=self.device),
                    use_distributed=self.use_distributed, metric_names=self.metric_names,
                )
                stats.update(metric_vals=m, batch_time=0.0, n=bs)
                if batch_id % log_freq == 0 and self.is_master_node:
                    stats.iter_summary(
                        epoch=-1, n_processed_samples=processed,
                        total_samples=total, elapsed_time=start, learning_rate=0.0,
                    )
        stats.epoch_summary(epoch=-1, stage=self.stage_name)
        # Capture final averages for CSV logging
        try:
            self.final_metrics = {
                name: float(stats.avg_statistics(metric_name=name))
                for name in self.metric_names
            }
        except Exception as e:
            logger.warning(f"Could not extract final metrics: {e}")


def _log_csv_row(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def main(opts, **kwargs):
    device = getattr(opts, "dev.device", torch.device("cpu"))
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last", False)
        else torch.contiguous_format
    )
    is_master_node = is_master(opts)

    # ---- data ---- #
    val_loader = create_eval_loader(opts)

    # ---- model (FP32) ---- #
    model = get_model(opts)
    model = model.to(device=device, memory_format=memory_format)
    model.eval()

    # ---- (optional) fold Conv+BN ---- #
    # Done before the --quant.enabled gate so plain FP32 eval can be A/B tested
    # against itself (with vs. without folding).
    fold_bn = bool(getattr(opts, "quant.fold_bn", False))
    n_bn_fused = 0
    if fold_bn:
        n_bn_fused = fold_conv_bn_(model)
        if is_master_node:
            logger.log(f"[quant] Folded {n_bn_fused} Conv+BN pairs into Conv weights/bias")

    if not getattr(opts, "quant.enabled", False):
        if not fold_bn:
            logger.warning(
                "--quant.enabled was not set. Running plain FP32 eval (equivalent to main_eval.py)."
            )
        else:
            logger.log(
                "--quant.enabled was not set; running plain FP32 eval on the BN-folded model."
            )
        Evaluator(opts=opts, model=model, eval_loader=val_loader).run()
        return

    # ---- convert ---- #
    w_bits = getattr(opts, "quant.weight_bits", 8)
    a_bits = getattr(opts, "quant.activation_bits", 8)
    skip_str = getattr(opts, "quant.skip_modules", "")
    skip_list = [s.strip() for s in skip_str.split(",") if s.strip()]

    quantize_linear = bool(getattr(opts, "quant.quantize_linear", False))
    insert_stubs = bool(getattr(opts, "quant.insert_stubs", False))
    quantize_acts = bool(getattr(opts, "quant.quantize_acts", False))
    quantize_residuals = bool(getattr(opts, "quant.quantize_residuals", False))
    skip_linears_str = getattr(opts, "quant.skip_linears", "")
    skip_linears = [s.strip() for s in skip_linears_str.split(",") if s.strip()]
    skip_acts_str = getattr(opts, "quant.skip_acts", "")
    skip_acts = [s.strip() for s in skip_acts_str.split(",") if s.strip()]
    skip_residuals_str = getattr(opts, "quant.skip_residuals", "")
    skip_residuals = [s.strip() for s in skip_residuals_str.split(",") if s.strip()]
    residual_observer = getattr(opts, "quant.residual_observer", "min_max")
    residual_scheme = getattr(opts, "quant.residual_scheme", "asymmetric")
    residual_bits_arg = int(getattr(opts, "quant.residual_bits", -1))
    residual_bits = residual_bits_arg if residual_bits_arg > 0 else a_bits
    skip_stubs_str = getattr(opts, "quant.skip_stubs", "")
    skip_stubs = [s.strip() for s in skip_stubs_str.split(",") if s.strip()]
    stub_observer = getattr(opts, "quant.stub_observer", "percentile")
    stub_scheme = getattr(opts, "quant.stub_scheme", "asymmetric")
    stub_bits_arg = int(getattr(opts, "quant.stub_bits", -1))
    stub_bits = stub_bits_arg if stub_bits_arg > 0 else a_bits
    stub_targets_csv = getattr(opts, "quant.stub_targets", "")
    stub_config_csv = getattr(opts, "quant.stub_config", "")
    default_stub_cfg = StubConfig(bits=stub_bits, observer=stub_observer, scheme=stub_scheme)
    if insert_stubs:
        if stub_config_csv.strip():
            # Per-target overrides take precedence; --quant.stub-targets is ignored
            # to avoid the ambiguity of "what does the homogeneous one mean too?".
            stub_target_configs = _build_stub_targets_from_config(
                stub_config_csv, default_stub_cfg
            )
            if stub_targets_csv.strip():
                logger.warning(
                    "[quant] --quant.stub-config provided; --quant.stub-targets is ignored."
                )
        else:
            stub_target_configs = _build_stub_targets(stub_targets_csv, default_stub_cfg)
    else:
        stub_target_configs = {}

    if is_master_node:
        logger.log(
            "[quant] Converting: Conv2d->QuantConv2d "
            "(w={}b, a={}b, skip={}) linear={} acts={} residuals={} stubs={} stub_bits={} stub_obs={}".format(
                w_bits, a_bits, skip_list, quantize_linear, quantize_acts,
                quantize_residuals, insert_stubs, stub_bits, stub_observer,
            )
        )
        if quantize_residuals:
            logger.log(
                f"[quant] residual stubs: bits={residual_bits} "
                f"observer={residual_observer} scheme={residual_scheme}"
            )
        if stub_config_csv.strip():
            logger.log(
                f"[quant] stub-config = '{stub_config_csv}' "
                f"({len(stub_target_configs)} type(s) resolved, per-target observers)"
            )
        else:
            logger.log(
                f"[quant] stub targets = '{stub_targets_csv}' "
                f"({len(stub_target_configs)} type(s) resolved)"
            )
    convert_model(
        model,
        weight_bits=w_bits,
        act_bits=a_bits,
        weight_observer=getattr(opts, "quant.weight_observer", "min_max"),
        act_observer=getattr(opts, "quant.act_observer", "min_max"),
        weight_scheme=getattr(opts, "quant.weight_scheme", "symmetric"),
        act_scheme=getattr(opts, "quant.act_scheme", "asymmetric"),
        skip_modules=skip_list,
        quantize_linear=quantize_linear,
        skip_linears=skip_linears,
        quantize_acts=quantize_acts,
        skip_acts=skip_acts,
        quantize_residuals=quantize_residuals,
        skip_residuals=skip_residuals,
        residual_observer=residual_observer,
        residual_scheme=residual_scheme,
        residual_bits=residual_bits,
        insert_stubs=insert_stubs,
        skip_stubs=skip_stubs,
        stub_target_configs=stub_target_configs if stub_target_configs else None,
        stub_bits=stub_bits,
        stub_observer=stub_observer,
        stub_scheme=stub_scheme,
    )
    model = model.to(device=device)  # move freshly-created quant buffers to device
    n_qconvs = len(collect_quant_convs(model))
    n_qlinears = len(collect_quant_linears(model))
    n_qstubs = len(collect_quant_stubs(model))
    n_qacts = len(collect_quant_hardswish(model))
    n_qskip_adds = len(collect_quant_skip_adds(model))
    if is_master_node:
        logger.log(
            f"[quant] Installed: {n_qconvs} QuantConv2d, "
            f"{n_qlinears} QuantLinear, {n_qacts} QuantHardswish, "
            f"{n_qskip_adds} QuantSkipAdd, {n_qstubs} QuantStub"
        )

    # ---- calibrate ---- #
    calib_size = getattr(opts, "quant.calib_size", 512)
    calib_bs = getattr(opts, "quant.calib_batch_size", 32)
    calib_seed = getattr(opts, "quant.calib_seed", 0)
    n_cal_batches = max(1, (calib_size + calib_bs - 1) // calib_bs)

    calib_loader = _build_calib_loader(opts=opts, batch_size=calib_bs, seed=calib_seed)

    if is_master_node:
        logger.log(
            "[quant] Calibrating on {} images ({} batches of {}, seed={})".format(
                n_cal_batches * calib_bs, n_cal_batches, calib_bs, calib_seed
            )
        )

    def _extract(batch):
        return move_to_device(opts=opts, x=batch, device=device)["samples"]

    t0 = time.time()
    calibrate(
        model=model,
        batches=calib_loader,
        n_batches=n_cal_batches,
        input_fn=_extract,
    )
    if is_master_node:
        logger.log(f"[quant] Calibration done in {time.time() - t0:.1f}s")

    # ---- (optional) bias correction ---- #
    bias_correction = bool(getattr(opts, "quant.bias_correction", False))
    if bias_correction:
        # Use a fresh loader so we get a deterministic pass with the same calibration set.
        bc_loader = _build_calib_loader(opts=opts, batch_size=calib_bs, seed=calib_seed)
        t0 = time.time()
        n_corrected = apply_bias_correction(
            model=model,
            batches=bc_loader,
            n_batches=n_cal_batches,
            input_fn=_extract,
        )
        if is_master_node:
            logger.log(
                f"[quant] Bias correction applied to {n_corrected} modules "
                f"in {time.time() - t0:.1f}s"
            )

    # ---- diagnostic: dump observer stats ---- #
    if is_master_node:
        qcs_all = collect_quant_convs(model)
        qls_all = collect_quant_linears(model)
        qhs_all = collect_quant_hardswish(model)
        qsas_all = collect_quant_skip_adds(model)
        qss_all = collect_quant_stubs(model)
        logger.log(
            f"[quant-diag] {len(qcs_all)} QuantConv2d, {len(qls_all)} QuantLinear, "
            f"{len(qhs_all)} QuantHardswish, {len(qsas_all)} QuantSkipAdd, "
            f"{len(qss_all)} QuantStub"
        )

        def _scalar_range(t):
            if t is None:
                return float("nan"), float("nan")
            return t.min().item(), t.max().item()

        def _scalar(t):
            # Per-channel observers can have Long zero_point of length > 1; cast to
            # float before mean() to avoid "could not infer output dtype" on Long.
            if t.numel() > 1:
                return t.to(torch.float32).mean().item()
            return t.item()

        for i, qc in enumerate(qcs_all):
            w_ob, a_ob = qc.weight_observer, qc.act_observer
            w_min, w_max = _scalar_range(getattr(w_ob, "min_val", None))
            a_min, a_max = _scalar_range(getattr(a_ob, "min_val", None))
            logger.log(
                "  conv[{:2d}] W:[{:+.3f},{:+.3f}] wscale_mean={:.2e}  "
                "A:[{:+.3f},{:+.3f}] ascale={:.2e} zp={:3d}".format(
                    i, w_min, w_max, _scalar(w_ob.scale),
                    a_min, a_max,
                    _scalar(a_ob.scale), int(_scalar(a_ob.zero_point)),
                )
            )
        for i, ql in enumerate(qls_all):
            w_ob, a_ob = ql.weight_observer, ql.act_observer
            w_min, w_max = _scalar_range(getattr(w_ob, "min_val", None))
            a_min, a_max = _scalar_range(getattr(a_ob, "min_val", None))
            logger.log(
                "  lin[{:2d}] W:[{:+.3f},{:+.3f}] wscale_mean={:.2e}  "
                "A:[{:+.3f},{:+.3f}] ascale={:.2e} zp={:3d}".format(
                    i, w_min, w_max, _scalar(w_ob.scale),
                    a_min, a_max,
                    _scalar(a_ob.scale), int(_scalar(a_ob.zero_point)),
                )
            )
        for i, qh in enumerate(qhs_all):
            a_ob = qh.act_observer
            a_min, a_max = _scalar_range(getattr(a_ob, "min_val", None))
            logger.log(
                "  hswish[{:2d}] A:[{:+.3f},{:+.3f}] ascale={:.2e} zp={:3d}".format(
                    i, a_min, a_max,
                    _scalar(a_ob.scale), int(_scalar(a_ob.zero_point)),
                )
            )
        for i, qsa in enumerate(qsas_all):
            sm, ss = qsa.stub_main.act_observer, qsa.stub_skip.act_observer
            sm_min, sm_max = _scalar_range(getattr(sm, "min_val", None))
            ss_min, ss_max = _scalar_range(getattr(ss, "min_val", None))
            logger.log(
                "  qskipadd[{:2d}] main:[{:+.3f},{:+.3f}] mscale={:.2e} mzp={:3d}  "
                "skip:[{:+.3f},{:+.3f}] sscale={:.2e} szp={:3d}".format(
                    i, sm_min, sm_max,
                    _scalar(sm.scale), int(_scalar(sm.zero_point)),
                    ss_min, ss_max,
                    _scalar(ss.scale), int(_scalar(ss.zero_point)),
                )
            )
        for i, qs in enumerate(qss_all):
            a_ob = qs.act_observer
            a_min, a_max = _scalar_range(getattr(a_ob, "min_val", None))
            logger.log(
                "  stub[{:2d}] A:[{:+.3f},{:+.3f}] ascale={:.2e} zp={:3d}".format(
                    i, a_min, a_max,
                    _scalar(a_ob.scale), int(_scalar(a_ob.zero_point)),
                )
            )

    # ---- evaluate (with recording) ---- #
    evaluator = _RecordingEvaluator(opts=opts, model=model, eval_loader=val_loader)
    evaluator.run()

    # ---- log CSV ---- #
    if is_master_node:
        # Phase 3 features write to v2 by default so Phase 1/2 sweep history isn't mutated.
        user_csv = getattr(opts, "quant.results_csv", "results/quant_sweep.csv")
        if (quantize_linear or insert_stubs or quantize_acts or quantize_residuals) and user_csv == "results/quant_sweep.csv":
            csv_path = "results/quant_sweep_v2.csv"
        else:
            csv_path = user_csv
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "w_bits": w_bits,
            "a_bits": a_bits,
            "stub_bits": stub_bits if insert_stubs else -1,
            "calib_size": n_cal_batches * calib_bs,
            "w_observer": getattr(opts, "quant.weight_observer", "min_max"),
            "a_observer": getattr(opts, "quant.act_observer", "min_max"),
            "stub_observer": stub_observer if insert_stubs else "",
            "w_scheme": getattr(opts, "quant.weight_scheme", "symmetric"),
            "a_scheme": getattr(opts, "quant.act_scheme", "asymmetric"),
            "quantize_linear": quantize_linear,
            "quantize_acts": quantize_acts,
            "quantize_residuals": quantize_residuals,
            "residual_observer": residual_observer if quantize_residuals else "",
            "residual_bits": residual_bits if quantize_residuals else -1,
            "insert_stubs": insert_stubs,
            "stub_targets": stub_targets_csv if insert_stubs else "",
            "stub_config": stub_config_csv if insert_stubs else "",
            "bias_correction": bias_correction,
            "fold_bn": fold_bn,
            "n_bn_fused": n_bn_fused,
            "n_quant_convs": n_qconvs,
            "n_quant_linears": n_qlinears,
            "n_quant_hardswish": n_qacts,
            "n_quant_skip_adds": n_qskip_adds,
            "n_quant_stubs": n_qstubs,
            "skip_modules": skip_str,
            "model_ckpt": os.path.basename(
                getattr(opts, "model.classification.pretrained", "") or ""
            ),
        }
        for name, val in (evaluator.final_metrics or {}).items():
            row[name] = round(val, 4)
        _log_csv_row(csv_path, row)
        logger.log(f"[quant] Appended result to {csv_path}")


def distributed_worker(i, main_fn, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))
    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:
        ddp_rank = kwargs.get("start_rank", 0) + i
        setattr(opts, "ddp.rank", ddp_rank)
    distributed_init(opts)
    main_fn(opts, **kwargs)


def main_worker(**kwargs):
    opts = get_eval_arguments()
    opts = device_setup(opts)

    is_master_node = is_master(opts)
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = f"{save_dir}/{run_label}"
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    use_distributed = getattr(opts, "ddp.enable", False) and num_gpus > 1
    setattr(opts, "ddp.use_distributed", use_distributed)

    n_cpus = multiprocessing.cpu_count()
    if getattr(opts, "dataset.workers", -1) in (-1, None):
        setattr(opts, "dataset.workers", n_cpus)

    train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
    val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
    setattr(opts, "dataset.train_batch_size0", train_bsize)
    setattr(opts, "dataset.val_batch_size0", val_bsize)
    setattr(opts, "dev.device_id", None)

    main(opts=opts, **kwargs)


if __name__ == "__main__":
    main_worker()
