"""Top-k seed selector — post-hoc analysis on existing big-seed grids.

For each fold, rank the 32 trained seeds by some criterion and average the
top-k seeds' test-fold returns. Compare to the unfiltered 32-seed mean.

What this measures: if you ran a portfolio of K independent agents (each with
1/K capital), would picking the top-k *by some pre-train criterion* beat
running all 32 equally weighted? This is the cleanest follow-up to the
[[ter-gate-mixed]] finding that single-seed CI is best in the TER-gated run.

Rankings tried (all are upper bounds — they use the test return itself):
- oracle by return  (best-case ceiling; not tradeable)
- oracle by sharpe  (best-case Sharpe ceiling)
- bottom-k inversion (sanity: does the rank ordering carry signal?)

A non-leaking ranker would need a held-out validation slice per fold; we don't
have that data on disk. Treat these numbers as the "what if we'd known" ceiling
— if even the ceiling can't beat the 32-seed baseline meaningfully, the lever
is closed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def analyze_tag(tag: str, ks=(1, 4, 8, 16, 32)) -> None:
    cells_path = Path("logs/grid") / tag / "cells.csv"
    ens_path = Path("logs/grid") / tag / "ensemble_cells.csv"
    if not cells_path.exists():
        print(f"[{tag}] cells.csv not found, skipping")
        return
    cells = pd.read_csv(cells_path)
    print(f"\n=== {tag} ({len(cells)} cells, {cells['fold'].nunique()} folds, "
          f"{cells['seed'].nunique()} seeds) ===")

    if ens_path.exists():
        ens = pd.read_csv(ens_path)
        print("\nReference: 32-seed EnsemblePolicy (per-fold, from saved ensemble_cells.csv):")
        print(ens[["fold", "return_pct", "sharpe", "max_dd_pct", "n_trades"]].round(2).to_string(index=False))
        ens_mean = ens["return_pct"].mean()
        ens_median = ens["return_pct"].median()
        print(f"  ensemble policy mean over folds = {ens_mean:+.2f}%, median {ens_median:+.2f}%")

    print("\nOracle top-k (rank by test return_pct, mean across folds):")
    print(f"  {'k':>3} | mean ret | median ret | std    | per-fold returns")
    for k in ks:
        per_fold = []
        per_fold_str = []
        for fold, sub in cells.groupby("fold"):
            top = sub.nlargest(k, "return_pct")
            per_fold.append(top["return_pct"].mean())
            per_fold_str.append(f"{per_fold[-1]:+6.1f}")
        mean = np.mean(per_fold); median = np.median(per_fold); std = np.std(per_fold)
        print(f"  {k:>3} | {mean:+7.2f}% | {median:+7.2f}%   | {std:5.2f}  | "
              f"{' '.join(per_fold_str)}")

    print("\nOracle top-k by SHARPE (rank by test sharpe, mean across folds):")
    print(f"  {'k':>3} | mean ret | mean sharpe | per-fold returns")
    for k in ks:
        per_fold_ret = []; per_fold_sh = []
        per_fold_str = []
        for fold, sub in cells.groupby("fold"):
            top = sub.nlargest(k, "sharpe")
            per_fold_ret.append(top["return_pct"].mean())
            per_fold_sh.append(top["sharpe"].mean())
            per_fold_str.append(f"{per_fold_ret[-1]:+6.1f}")
        print(f"  {k:>3} | {np.mean(per_fold_ret):+7.2f}% | "
              f"{np.mean(per_fold_sh):+6.2f}      | {' '.join(per_fold_str)}")

    print("\nBottom-k by return (sanity — should be much worse than top-k):")
    print(f"  {'k':>3} | mean ret  | per-fold returns")
    for k in (1, 4, 8, 16):
        per_fold = []
        per_fold_str = []
        for fold, sub in cells.groupby("fold"):
            bot = sub.nsmallest(k, "return_pct")
            per_fold.append(bot["return_pct"].mean())
            per_fold_str.append(f"{per_fold[-1]:+6.1f}")
        print(f"  {k:>3} | {np.mean(per_fold):+7.2f}% | {' '.join(per_fold_str)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=[
        "excess_bigseed_32",
        "excess_bigseed_32_h1",
        "excess_bigseed_32_h4",
        "excess_bigseed_32_h4_ter_w50_t010",
        "excess_bigseed_32_d1",
    ])
    args = p.parse_args()
    for tag in args.tags:
        analyze_tag(tag)


if __name__ == "__main__":
    main()
