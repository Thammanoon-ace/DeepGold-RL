"""
Risk-adjusted comparison: does the agent beat BUY-AND-HOLD on a risk-adjusted
basis (Sharpe / Calmar / max-drawdown), even if it loses on raw return?

In a strong bull market, raw return is near-impossible to beat passively holding.
The fair question is whether the agent delivers better *risk-adjusted* performance
(higher Sharpe, lower drawdown). This reads the grid's already-saved per-cell
metrics and computes buy-and-hold's metrics on the identical fold test windows —
no retraining.

Usage:
    python scripts/risk_vs_benchmark.py --grid excess
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtest.metrics import compute_report
from config.config import BARS_PER_YEAR, Config
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer
from validation.splitters import TimeSeriesSplitter


def main() -> None:
    ap = argparse.ArgumentParser(description="Risk-adjusted agent-vs-buy&hold comparison.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--grid", default="excess", help="logs/grid/<tag> to analyse.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--timeframe", default=None, choices=["M5", "M15", "H1", "H4"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    loader = HistoricalDataLoader(cfg.data)
    raw = loader.load(args.data)
    if args.timeframe and args.timeframe != cfg.data.timeframe:
        raw = loader.resample(raw, args.timeframe)
    featured = FeatureEngineer(cfg.features).transform(raw)
    bpy = BARS_PER_YEAR.get(args.timeframe or cfg.data.timeframe, cfg.backtest.bars_per_year)

    folds = TimeSeriesSplitter(n_splits=args.folds, mode="expanding",
                               gap=cfg.env.window_size).split(len(featured))
    init = cfg.env.initial_balance

    # --- Buy-and-hold metrics per fold (same test windows) ---
    bh = []
    for f in folds:
        close = featured.iloc[f.test_start:f.test_end]["close"].to_numpy()
        eq = init * close / close[0]
        r = compute_report(eq, [], init, bars_per_year=bpy)
        bh.append({"fold": f.index, "bh_return": r.total_return_pct,
                   "bh_sharpe": r.sharpe_ratio, "bh_maxdd": r.max_drawdown_pct,
                   "bh_calmar": r.calmar_ratio})
    bh = pd.DataFrame(bh)

    gdir = cfg.paths.logs / "grid" / args.grid
    single = pd.read_csv(gdir / "cells.csv")
    ens = pd.read_csv(gdir / "ensemble_cells.csv") if (gdir / "ensemble_cells.csv").exists() else None

    # Agent per-fold = mean across seeds (single) and the ensemble cell.
    agent_single = single.groupby("fold").agg(
        a_return=("return_pct", "mean"), a_sharpe=("sharpe", "mean"),
        a_maxdd=("max_dd_pct", "mean")).reset_index()
    m = agent_single.merge(bh, on="fold")
    if ens is not None:
        m = m.merge(ens[["fold", "return_pct", "sharpe", "max_dd_pct"]].rename(
            columns={"return_pct": "e_return", "sharpe": "e_sharpe", "max_dd_pct": "e_maxdd"}),
            on="fold", how="left")

    print("=" * 90)
    print(f"RISK-ADJUSTED: agent ({args.grid}) vs BUY-AND-HOLD, per fold "
          f"(tf={args.timeframe or cfg.data.timeframe})")
    print("=" * 90)
    cols = ["fold", "a_return", "bh_return", "a_sharpe", "bh_sharpe", "a_maxdd", "bh_maxdd"]
    if ens is not None:
        cols = ["fold", "e_return", "bh_return", "e_sharpe", "bh_sharpe", "e_maxdd", "bh_maxdd"]
    print(m[cols].round(2).to_string(index=False))
    print("-" * 90)

    # Use ensemble if present else single-mean for the verdict.
    ar = m["e_return" if ens is not None else "a_return"]
    ash = m["e_sharpe" if ens is not None else "a_sharpe"]
    add = m["e_maxdd" if ens is not None else "a_maxdd"]
    print("AGENT (ensemble) vs BUY-AND-HOLD across folds:")
    print(f"  Return : agent {ar.mean():+6.1f}%  vs  BH {m['bh_return'].mean():+6.1f}%   "
          f"-> agent wins {100*(ar.values > m['bh_return'].values).mean():.0f}% of folds")
    print(f"  Sharpe : agent {ash.mean():+6.2f}   vs  BH {m['bh_sharpe'].mean():+6.2f}    "
          f"-> agent wins {100*(ash.values > m['bh_sharpe'].values).mean():.0f}% of folds")
    print(f"  Max DD : agent {add.mean():6.1f}%  vs  BH {m['bh_maxdd'].mean():6.1f}%   "
          f"-> agent lower (better) {100*(add.values < m['bh_maxdd'].values).mean():.0f}% of folds")
    print("=" * 90)


if __name__ == "__main__":
    main()
