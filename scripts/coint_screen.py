"""Cointegration screen — find a tradeable pair before backtesting (group 3-B).

Pairs trading requires cointegration: the spread between two legs must be
stationary (mean-reverting). XAU/XAG failed this (ADF p≈0.47, see
[[kalman-pairs-rejected]]). This script screens every available metals pair
with the proper Engle-Granger two-step test, on the TRAIN slice only, so we
never backtest a pair that has no mean-reversion to capture.

Decision rule: only pairs with Engle-Granger p < 0.05 on the train slice are
worth a Kalman backtest. Everything else: stop, don't waste time.

Loads whatever <SYM>_<TF>.csv files exist under data/ (M5 or H1), resamples
all to a common timeframe (default D1), aligns on common timestamps, and runs:
  - Engle-Granger cointegration (statsmodels.tsa.stattools.coint) both directions
  - correlation of log-prices and of log-returns (context)

Usage:
    python scripts/coint_screen.py --timeframe D1
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import Config
from utils.data_loader import HistoricalDataLoader

# Symbols to screen if their CSV exists (any timeframe).
SYMBOLS = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "COPPER"]


def find_csv(sym: str, data_dir: Path) -> Path | None:
    for tf in ("M5", "H1", "M15", "H4", "D1"):
        p = data_dir / f"{sym}_{tf}.csv"
        if p.exists():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="D1", choices=["H1", "H4", "D1"])
    p.add_argument("--test-start", default="2025-01-01")
    args = p.parse_args()

    try:
        from statsmodels.tsa.stattools import coint
    except Exception as exc:
        raise SystemExit(f"statsmodels required: {exc}")

    cfg = Config.from_yaml("config/config.yaml") if Path("config/config.yaml").exists() else Config()
    loader = HistoricalDataLoader(cfg.data)
    data_dir = cfg.paths.data

    # Load + resample each available symbol to the target timeframe.
    series = {}
    for sym in SYMBOLS:
        csv = find_csv(sym, data_dir)
        if csv is None:
            print(f"  {sym}: no CSV found, skipping")
            continue
        df = loader.load(str(csv))
        if args.timeframe != "M5":
            df = loader.resample(df, args.timeframe)
        series[sym] = df["close"]
        print(f"  {sym}: {len(df)} bars from {csv.name}")

    if len(series) < 2:
        raise SystemExit("Need at least 2 symbols with data.")

    test_start = pd.Timestamp(args.test_start)
    print(f"\n=== Engle-Granger cointegration screen @ {args.timeframe} "
          f"(train < {args.test_start}) ===")
    print(f"  {'pair':<16} {'EG p (y~x)':>11} {'EG p (x~y)':>11} "
          f"{'min p':>7} {'logP corr':>10} {'ret corr':>9}  verdict")
    print("  " + "-" * 80)

    results = []
    for a, b in itertools.combinations(series, 2):
        df = pd.DataFrame({"a": series[a], "b": series[b]}).dropna()
        train = df[df.index < test_start]
        if len(train) < 200:
            continue
        la, lb = np.log(train["a"].to_numpy()), np.log(train["b"].to_numpy())
        # Engle-Granger both directions (test is not symmetric).
        p_ab = coint(la, lb)[1]
        p_ba = coint(lb, la)[1]
        min_p = min(p_ab, p_ba)
        logp_corr = np.corrcoef(la, lb)[0, 1]
        ret_corr = np.corrcoef(np.diff(la), np.diff(lb))[0, 1]
        verdict = "COINTEGRATED" if min_p < 0.05 else ("borderline" if min_p < 0.10 else "no")
        print(f"  {a+'/'+b:<16} {p_ab:>11.4f} {p_ba:>11.4f} {min_p:>7.4f} "
              f"{logp_corr:>10.3f} {ret_corr:>9.3f}  {verdict}")
        results.append((f"{a}/{b}", min_p, logp_corr, ret_corr))

    print()
    coint_pairs = [r for r in results if r[1] < 0.05]
    if coint_pairs:
        print("COINTEGRATED pairs (worth a Kalman backtest):")
        for name, mp, _, _ in sorted(coint_pairs, key=lambda r: r[1]):
            print(f"  {name}  (EG p = {mp:.4f})")
    else:
        print("No cointegrated pair (all EG p >= 0.05). Pairs trading not viable "
              "on these metals over this period. Do NOT backtest.")


if __name__ == "__main__":
    main()
