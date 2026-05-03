"""Read results/quant_observer_sweep.csv and print an 8x8 top-1 grid.

Rows = weight observer; cols = activation observer. Cell = top-1 (top-5).
Missing cells are shown as '-' (combo not yet completed).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

CSV_PATH = Path("results/quant_observer_sweep.csv")

WEIGHT_OBS = [
    "min_max", "percentile", "mse", "histogram",
    "per_channel_min_max", "per_channel_percentile",
    "per_channel_mse", "per_channel_histogram",
]
ACT_OBS = list(WEIGHT_OBS)

SHORT = {
    "min_max":                    "min_max",
    "percentile":                 "pctl",
    "mse":                        "mse",
    "histogram":                  "hist",
    "per_channel_min_max":        "pc_min_max",
    "per_channel_percentile":     "pc_pctl",
    "per_channel_mse":            "pc_mse",
    "per_channel_histogram":      "pc_hist",
}


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(CSV_PATH.open()))
    print(f"# rows in CSV: {len(rows)}")

    # Latest result wins per (w_obs, a_obs).
    grid = {}
    for r in rows:
        key = (r.get("w_observer", ""), r.get("a_observer", ""))
        grid[key] = r  # later rows overwrite

    # Header
    col_w = 14
    name_w = max(len(SHORT[w]) for w in WEIGHT_OBS) + 2
    print()
    print("Top-1 (top-5) accuracy on ImageNet-1K val (50k), 8/8 PTQ, BC off")
    print("Rows = weight observer  |  Cols = activation observer\n")
    header = " " * name_w + "".join(f"{SHORT[a]:>{col_w}}" for a in ACT_OBS)
    print(header)
    print("-" * len(header))
    for w in WEIGHT_OBS:
        line = f"{SHORT[w]:<{name_w}}"
        for a in ACT_OBS:
            r = grid.get((w, a))
            if r is None:
                cell = "-"
            else:
                t1 = r.get("top1") or r.get("top-1") or ""
                t5 = r.get("top5") or r.get("top-5") or ""
                if t1:
                    cell = f"{float(t1):.2f}"
                    if t5:
                        cell += f"({float(t5):.1f})"
                else:
                    cell = "?"
            line += f"{cell:>{col_w}}"
        print(line)

    # Best
    best = None
    for k, r in grid.items():
        t1 = r.get("top1") or r.get("top-1")
        if not t1:
            continue
        if best is None or float(t1) > best[1]:
            best = (k, float(t1))
    if best:
        (w, a), t1 = best
        print(f"\nBest: w={w}  a={a}  -> top1={t1:.2f}")


if __name__ == "__main__":
    main()
