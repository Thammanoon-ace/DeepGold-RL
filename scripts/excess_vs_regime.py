"""
Focused head-to-head: EXCESS reward  vs  EXCESS + REGIME features (V3.5 / 5B).

Answers one question with an automated verdict: **does adding 5B regime signals
to the observation improve the excess-reward agent** — specifically on
(a) ensemble Robustness Score, (b) the fold-3 strong-trend gap, (c) how often it
beats buy-and-hold, and (d) variance? Reports per-metric ✓/✗ and an overall call.

Reads only saved grid outputs (logs/grid/{excess,excess_regime}); no training.
Safe to run before the regime grid finishes (prints what's available).

Usage:
    python scripts/excess_vs_regime.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import Config


def _load(d: Path):
    if not (d / "summary.json").exists():
        return None
    out = {"summary": json.loads((d / "summary.json").read_text())}
    out["ensemble"] = (pd.read_csv(d / "ensemble_cells.csv")
                       if (d / "ensemble_cells.csv").exists() else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="EXCESS vs EXCESS+REGIME head-to-head.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--base", default="excess")
    ap.add_argument("--variant", default="excess_regime")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    g = cfg.paths.logs / "grid"
    A = _load(g / args.base)
    B = _load(g / args.variant)

    if A is None:
        print(f"Base grid '{args.base}' not found — run it first."); return
    if B is None:
        print(f"Variant '{args.variant}' not finished yet. Showing base only:\n")
        s = A["summary"]; e = s.get("ensemble") or {}
        print(f"  {args.base}: ensemble median {e.get('median'):+.1f}%  robust {e.get('robustness_score'):+.1f}"
              f"  | buy-hold {s['baseline_buy_hold_pct']:+.1f}%")
        print("\nRe-run this once logs/grid/%s/summary.json exists." % args.variant)
        return

    sa, sb = A["summary"], B["summary"]
    ea, eb = sa["ensemble"], sb["ensemble"]
    bh = sa["baseline_buy_hold_pct"]

    def row(name, a, b, better="higher", fmt="{:+.1f}"):
        if better == "higher":
            mark = "OK regime helps" if b > a + 1e-9 else "-- no gain"
        else:  # lower is better
            mark = "OK regime helps" if b < a - 1e-9 else "-- no gain"
        print(f"  {name:34} {fmt.format(a):>9}  ->  {fmt.format(b):>9}   {mark}")

    print("=" * 78)
    print(f"EXCESS ({args.base})  vs  EXCESS+REGIME ({args.variant})   | buy-hold {bh:+.1f}%")
    print("=" * 78)
    print("ENSEMBLE distribution:")
    row("median return %", ea["median"], eb["median"], "higher")
    row("robustness score", ea["robustness_score"], eb["robustness_score"], "higher")
    row("% profitable cells", ea["pct_positive"], eb["pct_positive"], "higher", "{:.0f}")
    row("std (variance, lower better)", ea["std"], eb["std"], "lower")
    print("SINGLE-seed distribution:")
    row("median return %", sa["single"]["median"], sb["single"]["median"], "higher")
    row("robustness score", sa["single"]["robustness_score"], sb["single"]["robustness_score"], "higher")
    row("beats buy-hold (cells)", sa["single"]["vs_baseline_winrate"],
        sb["single"]["vs_baseline_winrate"], "higher", "{:.0f}")

    # Fold-3 gap (the strong-trend weakness) on the ENSEMBLE.
    if A["ensemble"] is not None and B["ensemble"] is not None:
        fa = A["ensemble"].set_index("fold")["return_pct"]
        fb = B["ensemble"].set_index("fold")["return_pct"]
        print("\nPER-FOLD ENSEMBLE return % (fold 3 = strong trend, the gap):")
        tbl = pd.DataFrame({args.base: fa, args.variant: fb})
        tbl["delta"] = tbl[args.variant] - tbl[args.base]
        print(tbl.round(1).to_string())

    # Overall call.
    wins = sum([
        eb["median"] > ea["median"],
        eb["robustness_score"] > ea["robustness_score"],
        eb["pct_positive"] > ea["pct_positive"],
        eb["std"] < ea["std"],
    ])
    print("\n" + "=" * 78)
    print(f"VERDICT: regime features improved {wins}/4 ensemble metrics.")
    if wins >= 3:
        print("=> REGIME HELPED. Worth keeping; consider regime-conditioned sizing next.")
    elif wins == 2:
        print("=> MIXED. No clear win; the noise band may dominate — needs more seeds/folds.")
    else:
        print("=> REGIME DID NOT HELP. Drop it; the bottleneck is elsewhere (reward/market).")
    print("=" * 78)


if __name__ == "__main__":
    main()
