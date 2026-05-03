"""8x8 observer sweep at 8/8 PTQ on AFFNet-ET.

For each (weight_observer, act_observer) combination:
  - quantize all Conv2d + the final Linear (no stubs, no act fns, BC off)
  - calibrate on 512 ImageNet val images
  - evaluate on full ImageNet val (50k)
  - append a row to results/quant_observer_sweep.csv

Each combo is a separate `main_quant.py` subprocess for fault isolation —
one crash doesn't kill the rest of the sweep.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\alona\miniconda3\envs\AFFNet\python.exe"

WEIGHT_OBS = [
    "min_max",
    "percentile",
    "mse",
    "histogram",
    "per_channel_min_max",
    "per_channel_percentile",
    "per_channel_mse",
    "per_channel_histogram",
]
ACT_OBS = list(WEIGHT_OBS)

CSV_PATH = "results/quant_observer_sweep.csv"
LOG_DIR = PROJECT / "results" / "sweep_observer_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_cmd(w_obs: str, a_obs: str) -> list:
    return [
        PYTHON,
        "main_quant.py",
        "--common.config-file", "resource/config/imagenet_et/config.yaml",
        "--common.results-loc", "results/",
        "--model.classification.pretrained",
        "resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt",
        "--quant.enabled",
        "--quant.weight-bits", "8",
        "--quant.activation-bits", "8",
        "--quant.weight-observer", w_obs,
        "--quant.act-observer", a_obs,
        # weights default symmetric, activations default asymmetric — kept explicit
        "--quant.weight-scheme", "symmetric",
        "--quant.act-scheme", "asymmetric",
        "--quant.quantize-linear",
        "--quant.results-csv", CSV_PATH,
    ]


def already_done() -> set:
    """Return the set of (w_obs, a_obs) tuples already present in the CSV."""
    p = PROJECT / CSV_PATH
    if not p.exists():
        return set()
    import csv as _csv
    with p.open() as f:
        return {(r.get("w_observer", ""), r.get("a_observer", "")) for r in _csv.DictReader(f)}


def main():
    combos = list(itertools.product(WEIGHT_OBS, ACT_OBS))
    done = already_done()
    n = len(combos)
    n_skip = sum(1 for c in combos if c in done)
    print(f"[sweep] {n} combinations total; {n_skip} already in CSV (skipping); "
          f"CSV -> {CSV_PATH}", flush=True)
    t0 = time.time()
    for i, (w_obs, a_obs) in enumerate(combos, 1):
        if (w_obs, a_obs) in done:
            print(f"[sweep] [{i:02d}/{n}] w={w_obs} a={a_obs}  SKIP (already done)",
                  flush=True)
            continue
        log_file = LOG_DIR / f"{i:02d}_w-{w_obs}__a-{a_obs}.log"
        cmd = build_cmd(w_obs, a_obs)
        elapsed = time.time() - t0
        avg = elapsed / max(1, i - 1) if i > 1 else 0
        eta_s = avg * (n - i + 1) if avg > 0 else 0
        print(
            f"[sweep] [{i:02d}/{n}] w={w_obs} a={a_obs}  "
            f"elapsed={elapsed/60:.1f}min  ETA={eta_s/60:.0f}min",
            flush=True,
        )
        with open(log_file, "w") as lf:
            ret = subprocess.run(cmd, cwd=str(PROJECT), stdout=lf, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            print(f"[sweep]   FAILED rc={ret.returncode}; see {log_file}", flush=True)
        else:
            print(f"[sweep]   OK; log={log_file.name}", flush=True)
    print(f"[sweep] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
