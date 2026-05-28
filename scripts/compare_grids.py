"""
Compare several grid_eval runs side by side (V3.5 analysis).

Reads logs/grid/<tag>/{summary.json,cells.csv,ensemble_cells.csv} for each tag
and prints:
  1. a side-by-side summary (single + ensemble: median, std, %pos, robustness,
     beats-buy&hold, median CI),
  2. a per-fold ENSEMBLE return table — the key view for "did regime close the
     fold-3 (strong-trend) gap?",
  3. a per-fold single-seed mean return table.

Missing grids (e.g. a run still in progress) are skipped gracefully, so this is
safe to run before every grid has finished.

Usage:
    python scripts/compare_grids.py --grids m5 excess excess_regime
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import Config

# Friendly labels for known tags.
LABELS = {
    "m5": "cnn ABSOLUTE (baseline)",
    "cnnlstm": "cnn_lstm (hybrid)",
    "volsizing": "cnn + vol-sizing",
    "excess": "cnn EXCESS (beat-BH)",
    "excess_regime": "cnn EXCESS + REGIME",
    "h1": "cnn H1",
}


def _load(grid_dir: Path):
    summ = grid_dir / "summary.json"
    if not summ.exists():
        return None
    d = json.loads(summ.read_text())
    cells = pd.read_csv(grid_dir / "cells.csv") if (grid_dir / "cells.csv").exists() else None
    ens = (pd.read_csv(grid_dir / "ensemble_cells.csv")
           if (grid_dir / "ensemble_cells.csv").exists() else None)
    return {"summary": d, "cells": cells, "ensemble": ens}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare grid_eval runs.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--grids", nargs="+", default=["m5", "excess", "excess_regime"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    base = cfg.paths.logs / "grid"

    loaded, missing = {}, []
    for tag in args.grids:
        r = _load(base / tag)
        (loaded.__setitem__(tag, r) if r else missing.append(tag))
    if missing:
        print(f"(skipped - not found / not finished: {', '.join(missing)})\n")
    if not loaded:
        print("No finished grids to compare yet.")
        return

    # --- 1. Summary table ---------------------------------------------- #
    print("=" * 104)
    print("SUMMARY  (single-seed distribution / SEED-ENSEMBLE distribution; vs buy-and-hold)")
    print("=" * 104)
    hdr = f"{'config':28} | {'s.med':>6} {'s.std':>6} {'s%pos':>5} {'s.rob':>7} | {'e.med':>6} {'e.std':>6} {'e%pos':>5} {'e.rob':>7} | {'BH%':>6} | {'med 95% CI':>16}"
    print(hdr); print("-" * 104)
    for tag in args.grids:
        if tag not in loaded:
            continue
        d = loaded[tag]["summary"]; s = d["single"]; e = d.get("ensemble") or {}
        ci = d.get("median_ci", [float("nan"), float("nan")])
        print(f"{LABELS.get(tag, tag):28} | {s['median']:+6.1f} {s['std']:6.1f} {s['pct_positive']:5.0f} "
              f"{s['robustness_score']:+7.1f} | {e.get('median', float('nan')):+6.1f} {e.get('std', float('nan')):6.1f} "
              f"{e.get('pct_positive', float('nan')):5.0f} {e.get('robustness_score', float('nan')):+7.1f} | "
              f"{d['baseline_buy_hold_pct']:+6.1f} | [{ci[0]:+6.1f},{ci[1]:+6.1f}]")

    # --- 2. Per-fold ENSEMBLE return (the fold-3 question) ------------- #
    print("\n" + "=" * 60)
    print("PER-FOLD ENSEMBLE return %  (watch fold 3 = strong trend)")
    print("=" * 60)
    ens_tbl = {}
    for tag in args.grids:
        if tag in loaded and loaded[tag]["ensemble"] is not None:
            ens_tbl[LABELS.get(tag, tag)] = loaded[tag]["ensemble"].set_index("fold")["return_pct"]
    if ens_tbl:
        print(pd.DataFrame(ens_tbl).round(1).to_string())

    # --- 3. Per-fold single-seed MEAN return --------------------------- #
    print("\n" + "=" * 60)
    print("PER-FOLD single-seed MEAN return %")
    print("=" * 60)
    single_tbl = {}
    for tag in args.grids:
        if tag in loaded and loaded[tag]["cells"] is not None:
            single_tbl[LABELS.get(tag, tag)] = loaded[tag]["cells"].groupby("fold")["return_pct"].mean()
    if single_tbl:
        print(pd.DataFrame(single_tbl).round(1).to_string())


if __name__ == "__main__":
    main()
