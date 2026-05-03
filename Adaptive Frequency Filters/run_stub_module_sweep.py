"""Step 2: per-module stub-observer sweep.

Take the top-1 (w_obs, a_obs) from results/quant_observer_sweep_bc.csv.
With BC + fold-BN + 4 stubs always on (ln2d, afno2d, swish, globalpool):

  Phase A — for each of the 4 stub types, sweep its observer through all 8
            options while the OTHER 3 stubs stay at min_max. 4*8 = 32 runs.

  Phase B — pick the per-module best observer from phase A. Run one combined
            config: stub-config = {ln2d=best_ln2d, afno2d=best_afno2d, ...}.

CSV: results/quant_stub_module_sweep.csv. Subprocess per run, idempotent skip
on already-present (stub_config) rows.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\alona\miniconda3\envs\AFFNet\python.exe"

BC_CSV = PROJECT / "results" / "quant_observer_sweep_bc.csv"
CSV_PATH = "results/quant_stub_module_sweep.csv"
LOG_DIR = PROJECT / "results" / "sweep_stub_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STUB_TYPES = ["ln2d", "afno2d", "swish", "globalpool"]
OBSERVERS = [
    "min_max", "percentile", "mse", "histogram",
    "per_channel_min_max", "per_channel_percentile",
    "per_channel_mse", "per_channel_histogram",
]
DEFAULT_STUB_OBS = "min_max"   # Phase 5 baseline; held constant for non-swept stubs


def pick_winner_from_bc() -> tuple[str, str]:
    if not BC_CSV.exists():
        sys.exit(f"BC CSV not found: {BC_CSV}. Run run_bc_top5.py first.")
    rows = list(csv.DictReader(BC_CSV.open()))
    if not rows:
        sys.exit("BC CSV empty.")
    best = max(rows, key=lambda r: float(r.get("top1") or "0"))
    return best["w_observer"], best["a_observer"]


def already_done() -> set:
    p = PROJECT / CSV_PATH
    if not p.exists():
        return set()
    with p.open() as f:
        return {r.get("stub_config", "") for r in csv.DictReader(f)}


def make_stub_config(per_type: dict) -> str:
    """per_type: {stub_type_name: observer_name} -> '--quant.stub-config' arg."""
    return ",".join(f"{t}={per_type[t]}" for t in STUB_TYPES)


def cmd(w_obs: str, a_obs: str, stub_config: str) -> list:
    return [
        PYTHON, "main_quant.py",
        "--common.config-file", "resource/config/imagenet_et/config.yaml",
        "--common.results-loc", "results/",
        "--model.classification.pretrained",
        "resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt",
        "--quant.enabled",
        "--quant.weight-bits", "8",
        "--quant.activation-bits", "8",
        "--quant.weight-observer", w_obs,
        "--quant.act-observer", a_obs,
        "--quant.weight-scheme", "symmetric",
        "--quant.act-scheme", "asymmetric",
        "--quant.quantize-linear",
        "--quant.bias-correction",
        "--quant.fold-bn",
        "--quant.insert-stubs",
        "--quant.stub-config", stub_config,
        "--quant.stub-bits", "8",
        "--quant.stub-scheme", "asymmetric",
        # stub-config takes precedence; stub-targets is ignored when set.
        "--quant.results-csv", CSV_PATH,
    ]


def run_one(label: str, w_obs: str, a_obs: str, stub_config: str,
            done: set, t0: float, idx: int, total: int) -> str:
    if stub_config in done:
        print(f"[stub] [{idx:02d}/{total}] {label}  SKIP (already done)", flush=True)
        return "SKIP"
    log_file = LOG_DIR / f"{idx:02d}_{label.replace('/', '_').replace(' ', '_')}.log"
    elapsed = (time.time() - t0) / 60
    avg = elapsed / max(1, idx - 1) if idx > 1 else 0
    eta = avg * (total - idx + 1)
    print(f"[stub] [{idx:02d}/{total}] {label}  elapsed={elapsed:.1f}min  ETA={eta:.0f}min",
          flush=True)
    print(f"        stub-config = {stub_config}", flush=True)
    with open(log_file, "w") as lf:
        ret = subprocess.run(cmd(w_obs, a_obs, stub_config),
                             cwd=str(PROJECT), stdout=lf, stderr=subprocess.STDOUT)
    if ret.returncode != 0:
        print(f"[stub]   FAILED rc={ret.returncode}; see {log_file}", flush=True)
        return "FAIL"
    print(f"[stub]   OK; log={log_file.name}", flush=True)
    return "OK"


def best_observer_per_type() -> dict:
    """After phase A, pick the observer with highest top-1 per stub type."""
    p = PROJECT / CSV_PATH
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))
    best = {}
    for t in STUB_TYPES:
        # rows where this type is the swept axis: only this type differs from default.
        candidates = []
        for r in rows:
            sc = r.get("stub_config", "")
            if not sc:
                continue
            kv = dict(part.split("=") for part in sc.split(","))
            # this row sweeps `t` if every other type is at DEFAULT_STUB_OBS
            if all(kv.get(other, DEFAULT_STUB_OBS) == DEFAULT_STUB_OBS for other in STUB_TYPES if other != t):
                candidates.append((kv.get(t, DEFAULT_STUB_OBS), float(r.get("top1") or "0")))
        if candidates:
            obs, _ = max(candidates, key=lambda x: x[1])
            best[t] = obs
        else:
            best[t] = DEFAULT_STUB_OBS
    return best


def main():
    w_obs, a_obs = pick_winner_from_bc()
    print(f"[stub] BC winner: w={w_obs} a={a_obs}", flush=True)

    done = already_done()
    total = len(STUB_TYPES) * len(OBSERVERS) + 1   # +1 combined
    t0 = time.time()
    idx = 0

    # ---- Phase A: 4 modules x 8 observers, vary one at a time ---- #
    for stub_type in STUB_TYPES:
        for obs in OBSERVERS:
            idx += 1
            per_type = {t: DEFAULT_STUB_OBS for t in STUB_TYPES}
            per_type[stub_type] = obs
            sc = make_stub_config(per_type)
            label = f"sweep_{stub_type}={obs}"
            run_one(label, w_obs, a_obs, sc, done, t0, idx, total)
            done.add(sc)   # in case we generated the same stub_config (e.g., default) twice

    # ---- Phase B: combined best ---- #
    best = best_observer_per_type()
    print(f"[stub] best per type from phase A: {best}", flush=True)
    sc_combined = make_stub_config(best)
    idx += 1
    label = "combined_best"
    run_one(label, w_obs, a_obs, sc_combined, done, t0, idx, total)

    print(f"[stub] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
