"""Step 1: bias correction + fold-BN on the top-5 (w_obs, a_obs) pairs from
the 8x8 sweep. Conv2d + Linear quantized at 8/8, no stubs.
"""
from __future__ import annotations

import csv
import itertools
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\alona\miniconda3\envs\AFFNet\python.exe"

# Top 5 by top-1 from results/quant_observer_sweep.csv
TOP5 = [
    ("per_channel_min_max",  "per_channel_mse"),         # 70.87
    ("per_channel_mse",      "per_channel_mse"),         # 70.80
    ("per_channel_mse",      "per_channel_min_max"),     # 70.79
    ("per_channel_min_max",  "per_channel_min_max"),     # 70.78
    ("per_channel_min_max",  "per_channel_percentile"),  # 68.39
]

CSV_PATH = "results/quant_observer_sweep_bc.csv"
LOG_DIR = PROJECT / "results" / "sweep_bc_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def already_done() -> set:
    p = PROJECT / CSV_PATH
    if not p.exists():
        return set()
    with p.open() as f:
        return {(r.get("w_observer", ""), r.get("a_observer", "")) for r in csv.DictReader(f)}


def cmd(w_obs: str, a_obs: str) -> list:
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
        "--quant.results-csv", CSV_PATH,
    ]


def main():
    done = already_done()
    n = len(TOP5)
    print(f"[bc] {n} configs (BC + fold-BN); CSV -> {CSV_PATH}", flush=True)
    t0 = time.time()
    for i, (w, a) in enumerate(TOP5, 1):
        if (w, a) in done:
            print(f"[bc] [{i:02d}/{n}] w={w} a={a}  SKIP", flush=True)
            continue
        log_file = LOG_DIR / f"{i:02d}_w-{w}__a-{a}.log"
        elapsed = (time.time() - t0) / 60
        avg = elapsed / max(1, i - 1) if i > 1 else 0
        eta = avg * (n - i + 1)
        print(f"[bc] [{i:02d}/{n}] w={w} a={a}  elapsed={elapsed:.1f}min  ETA={eta:.0f}min",
              flush=True)
        with open(log_file, "w") as lf:
            ret = subprocess.run(cmd(w, a), cwd=str(PROJECT), stdout=lf, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            print(f"[bc]   FAILED rc={ret.returncode}; see {log_file}", flush=True)
        else:
            print(f"[bc]   OK; log={log_file.name}", flush=True)
    print(f"[bc] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
