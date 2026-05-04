"""Step 3: residual-add observer 8x8 grid (main x skip separate).

Base: BC + fold-BN, step 2 combined-best stub config (read from
results/quant_stub_module_sweep.csv at startup).

Sweeps every (residual_main_observer, residual_skip_observer) pair through
8x8 = 64 combos. CSV: results/quant_residual_sweep.csv.
"""
from __future__ import annotations

import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PYTHON = r"C:\Users\alona\miniconda3\envs\AFFNet\python.exe"

STUB_CSV = PROJECT / "results" / "quant_stub_module_sweep.csv"
CSV_PATH = "results/quant_residual_sweep.csv"
LOG_DIR = PROJECT / "results" / "sweep_residual_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

OBSERVERS = [
    "min_max", "percentile", "mse", "histogram",
    "per_channel_min_max", "per_channel_percentile",
    "per_channel_mse", "per_channel_histogram",
]
STUB_TYPES = ["ln2d", "afno2d", "swish", "globalpool"]


def pick_stub_config_from_step2() -> tuple[str, str, str]:
    """Determine the base (w_obs, a_obs, stub_config) from step 2 CSV.
    Picks the per-type best observer per stub type (single-axis sweep), then
    constructs the combined config string.
    """
    if not STUB_CSV.exists():
        sys.exit(f"step 2 CSV not found: {STUB_CSV}. Run step 2 first.")
    rows = list(csv.DictReader(STUB_CSV.open()))
    if not rows:
        sys.exit("step 2 CSV empty.")

    # All step 2 runs share the same w_obs / a_obs (the BC winner).
    w_obs = rows[0]["w_observer"]
    a_obs = rows[0]["a_observer"]

    # Per-type best from single-axis sweep rows (others-at-min_max).
    parsed = []
    for r in rows:
        sc = r.get("stub_config", "")
        if not sc:
            continue
        kv = dict(p.split("=") for p in sc.split(","))
        parsed.append((kv, float(r.get("top1") or "0")))

    best = {}
    for t in STUB_TYPES:
        cands = []
        for kv, t1 in parsed:
            if all(kv.get(o, "min_max") == "min_max" for o in STUB_TYPES if o != t):
                cands.append((kv.get(t, "min_max"), t1))
        if cands:
            obs, _ = max(cands, key=lambda x: x[1])
            best[t] = obs
        else:
            best[t] = "min_max"
    stub_config = ",".join(f"{t}={best[t]}" for t in STUB_TYPES)
    return w_obs, a_obs, stub_config


def already_done() -> set:
    p = PROJECT / CSV_PATH
    if not p.exists():
        return set()
    with p.open() as f:
        return {(r.get("residual_main_observer", ""),
                 r.get("residual_skip_observer", "")) for r in csv.DictReader(f)}


def cmd(w_obs: str, a_obs: str, stub_config: str, m_obs: str, s_obs: str) -> list:
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
        "--quant.quantize-residuals",
        "--quant.residual-bits", "8",
        "--quant.residual-scheme", "asymmetric",
        "--quant.residual-main-observer", m_obs,
        "--quant.residual-skip-observer", s_obs,
        "--quant.results-csv", CSV_PATH,
    ]


def main():
    w_obs, a_obs, stub_config = pick_stub_config_from_step2()
    print(f"[res] base: w={w_obs} a={a_obs}", flush=True)
    print(f"[res] stub-config = {stub_config}", flush=True)

    done = already_done()
    combos = list(itertools.product(OBSERVERS, OBSERVERS))
    n = len(combos)
    print(f"[res] {n} (main, skip) combos; CSV -> {CSV_PATH}", flush=True)
    t0 = time.time()
    for i, (m, s) in enumerate(combos, 1):
        if (m, s) in done:
            print(f"[res] [{i:02d}/{n}] main={m} skip={s}  SKIP", flush=True)
            continue
        log_file = LOG_DIR / f"{i:02d}_main-{m}__skip-{s}.log"
        elapsed = (time.time() - t0) / 60
        avg = elapsed / max(1, i - 1) if i > 1 else 0
        eta = avg * (n - i + 1)
        print(f"[res] [{i:02d}/{n}] main={m} skip={s}  elapsed={elapsed:.1f}min  ETA={eta:.0f}min",
              flush=True)
        with open(log_file, "w") as lf:
            ret = subprocess.run(cmd(w_obs, a_obs, stub_config, m, s),
                                 cwd=str(PROJECT), stdout=lf, stderr=subprocess.STDOUT)
        if ret.returncode != 0:
            print(f"[res]   FAILED rc={ret.returncode}; see {log_file}", flush=True)
        else:
            print(f"[res]   OK; log={log_file.name}", flush=True)
    print(f"[res] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
