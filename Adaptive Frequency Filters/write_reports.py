"""Write per-section text reports from the various sweep CSVs.

Sections:
  step1_bc_top5         <- results/quant_observer_sweep_bc.csv
  step2_stub_module     <- results/quant_stub_module_sweep.csv
  bc_no_fold_bn         <- results/quant_observer_sweep_bc_no_fold_bn.csv
                            (with side-by-side vs step 1's BC+fold-BN run)
  residual_sweep        <- results/quant_residual_sweep.csv

Each section is a self-contained .txt file in results/. Idempotent — overwrites.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

PROJECT = Path(__file__).resolve().parent
RES = PROJECT / "results"

SHORT = {
    "min_max":                "min_max",
    "percentile":             "pctl",
    "mse":                    "mse",
    "histogram":              "hist",
    "per_channel_min_max":    "pc_min_max",
    "per_channel_percentile": "pc_pctl",
    "per_channel_mse":        "pc_mse",
    "per_channel_histogram":  "pc_hist",
}

OBSERVERS = list(SHORT.keys())


def _read_csv(path: Path) -> list:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def write_step1_report(out: Path = RES / "report_step1_bc_top5.txt") -> None:
    rows = _read_csv(RES / "quant_observer_sweep_bc.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r.get("top1") or "0"), reverse=True)
    lines = [
        "Step 1 — Bias Correction + fold-BN on the top-5 (w_obs, a_obs) configs",
        "=" * 80,
        "Setup: 8/8 PTQ, Conv2d + Linear quantized, no stubs, no act-Q.",
        "Calibration: 512 ImageNet val images.",
        "",
        f"{'rank':<5}{'weight observer':<28}{'act observer':<28}{'top1':>8}{'top5':>8}",
        "-" * 80,
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:<5}{r['w_observer']:<28}{r['a_observer']:<28}"
            f"{float(r['top1']):>8.2f}{float(r['top5']):>8.2f}"
        )
    best = rows[0]
    lines += [
        "",
        f"BEST: w={best['w_observer']}  a={best['a_observer']}  "
        f"top1={float(best['top1']):.2f}  top5={float(best['top5']):.2f}",
        f"FP32 reference: 73.02 top-1.  Gap: {73.02 - float(best['top1']):.2f}pp",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def write_step2_report(out: Path = RES / "report_step2_stub_module.txt") -> None:
    rows = _read_csv(RES / "quant_stub_module_sweep.csv")
    if not rows:
        return
    # parse stub_config into per-type dict
    stub_types = ["ln2d", "afno2d", "swish", "globalpool"]
    parsed = []
    for r in rows:
        sc = r.get("stub_config", "")
        if not sc:
            continue
        kv = dict(p.split("=") for p in sc.split(","))
        parsed.append((r, kv, float(r.get("top1") or "0"), float(r.get("top5") or "0")))

    # Build the 4x8 grid: rows = stub type, cols = observer choice.
    # Cell = top1 of the run where THAT type uses observer X and others use min_max.
    grid = {}   # (stub_type, observer) -> (top1, top5)
    for r, kv, t1, t5 in parsed:
        for t in stub_types:
            others_default = all(kv.get(o, "min_max") == "min_max" for o in stub_types if o != t)
            if others_default:
                grid[(t, kv[t])] = (t1, t5)

    lines = [
        "Step 2 — Per-module stub observer sweep (always-on stubs, vary one type at a time)",
        "=" * 90,
        "Base: BC + fold-BN, w=per_channel_min_max, a=per_channel_mse",
        "(step 1 winner; baseline w/o stubs = 72.76% top-1).",
        "",
        "All 4 stubs always installed (ln2d=21, afno2d=9, swish=55, globalpool=1).",
        "When sweeping one stub type's observer, the other 3 stay at min_max.",
        "",
        f"{'stub type':<14}" + "".join(f"{SHORT[o]:>14}" for o in OBSERVERS),
        "-" * 90,
    ]
    for t in stub_types:
        line = f"{t:<14}"
        for o in OBSERVERS:
            cell = grid.get((t, o))
            if cell is None:
                line += f"{'-':>14}"
            else:
                line += f"{cell[0]:>14.2f}"
        lines.append(line)

    # per-type best
    lines += ["", "Per-type best observer (highest top1 in row):"]
    best_per_type = {}
    for t in stub_types:
        cands = [(o, grid[(t, o)]) for o in OBSERVERS if (t, o) in grid]
        if cands:
            o_best, (t1_best, t5_best) = max(cands, key=lambda x: x[1][0])
            best_per_type[t] = o_best
            lines.append(f"  {t:<14} -> {o_best:<28} top1={t1_best:6.2f}  top5={t5_best:6.2f}")

    # combined-best run row, if present
    combined_sc = ",".join(f"{t}={best_per_type.get(t, 'min_max')}" for t in stub_types)
    combined = next((p for p in parsed if p[1] == dict(s.split("=") for s in combined_sc.split(","))), None)
    if combined is not None:
        _, _, c_t1, c_t5 = combined
        lines += [
            "",
            f"Combined (all per-type bests at once): {combined_sc}",
            f"  top1={c_t1:.2f}  top5={c_t5:.2f}",
        ]

    # global best row
    best_row = max(parsed, key=lambda x: x[2])
    _, kv_b, t1_b, t5_b = best_row
    lines += [
        "",
        f"GLOBAL BEST in step 2: {','.join(f'{k}={v}' for k, v in kv_b.items())}",
        f"  top1={t1_b:.2f}  top5={t5_b:.2f}",
        f"  vs step 1 winner (no stubs, 72.76): {t1_b - 72.76:+.2f}pp",
    ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def write_bc_no_fold_bn_report(out: Path = RES / "report_bc_no_fold_bn.txt") -> None:
    bc = _read_csv(RES / "quant_observer_sweep_bc.csv")            # BC + fold-BN
    bc_nf = _read_csv(RES / "quant_observer_sweep_bc_no_fold_bn.csv")  # BC only
    if not bc_nf:
        return

    bc_lookup = {(r["w_observer"], r["a_observer"]): r for r in bc}
    bc_nf_sorted = sorted(bc_nf, key=lambda r: float(r.get("top1") or "0"), reverse=True)

    lines = [
        "Step 1-extra — BC only (no fold-BN) on the top-5 configs, vs BC + fold-BN",
        "=" * 100,
        "Setup: 8/8 PTQ, Conv2d + Linear quantized, no stubs, no act-Q.",
        "",
        f"{'rank':<5}{'weight observer':<28}{'act observer':<28}"
        f"{'BC top1':>10}{'BC+BN top1':>14}{'Δ (BN gain)':>14}",
        "-" * 100,
    ]
    for i, r in enumerate(bc_nf_sorted, 1):
        key = (r["w_observer"], r["a_observer"])
        bc_t1 = float(r["top1"])
        bc_bn = bc_lookup.get(key)
        bc_bn_t1 = float(bc_bn["top1"]) if bc_bn else float("nan")
        delta = bc_bn_t1 - bc_t1 if bc_bn else float("nan")
        lines.append(
            f"{i:<5}{r['w_observer']:<28}{r['a_observer']:<28}"
            f"{bc_t1:>10.2f}{bc_bn_t1:>14.2f}{delta:>14.2f}"
        )

    bc_nf_best = bc_nf_sorted[0]
    lines += [
        "",
        f"BC-only best: w={bc_nf_best['w_observer']}  a={bc_nf_best['a_observer']}  "
        f"top1={float(bc_nf_best['top1']):.2f}  top5={float(bc_nf_best['top5']):.2f}",
        f"FP32: 73.02 top-1.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def write_residual_report(out: Path = RES / "report_residual_sweep.txt") -> None:
    rows = _read_csv(RES / "quant_residual_sweep.csv")
    if not rows:
        return
    # rows have residual_observer (single col) but we encoded the pair via CSV-ext;
    # actually we'll use a custom CLI patch that writes residual_main_observer +
    # residual_skip_observer. The reporter parses those directly.
    grid = {}
    for r in rows:
        m = r.get("residual_main_observer") or r.get("residual_observer", "")
        s = r.get("residual_skip_observer") or m
        t1 = float(r.get("top1") or "0")
        t5 = float(r.get("top5") or "0")
        grid[(m, s)] = (t1, t5)

    lines = [
        "Step 3 — Residual-add observer sweep (8x8 grid, main x skip)",
        "=" * 100,
        "Base: BC + fold-BN, step 2 combined-best stub config.",
        "Each cell = (residual_main_observer, residual_skip_observer); top-1 (top-5).",
        "",
        f"{'main / skip':<14}" + "".join(f"{SHORT[o]:>14}" for o in OBSERVERS),
        "-" * 100,
    ]
    for m in OBSERVERS:
        line = f"{SHORT[m]:<14}"
        for s in OBSERVERS:
            cell = grid.get((m, s))
            if cell is None:
                line += f"{'-':>14}"
            else:
                line += f"{cell[0]:>14.2f}"
        lines.append(line)

    # global best
    if grid:
        (m_b, s_b), (t1_b, t5_b) = max(grid.items(), key=lambda x: x[1][0])
        lines += [
            "",
            f"BEST: residual main={m_b}  skip={s_b}  top1={t1_b:.2f}  top5={t5_b:.2f}",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main():
    sect = sys.argv[1] if len(sys.argv) > 1 else "all"
    if sect in ("step1", "all"):
        write_step1_report()
    if sect in ("step2", "all"):
        write_step2_report()
    if sect in ("bc_nf", "all"):
        write_bc_no_fold_bn_report()
    if sect in ("residual", "all"):
        write_residual_report()


if __name__ == "__main__":
    main()
