"""
Inspect a market-data CSV and confirm the pipeline ingests it correctly.

Use this immediately after dropping a real XAUUSD export into ``data/`` to
verify the loader parses your broker's format, see the date range, detect gaps,
and confirm the walk-forward train/test split is non-empty.

Usage:
    python scripts/inspect_data.py --data data/XAUUSD_M5.csv
    python scripts/inspect_data.py                 # uses config's csv_filename
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import Config
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer


def _expected_step(timeframe: str) -> pd.Timedelta:
    return {"M5": pd.Timedelta(minutes=5),
            "M15": pd.Timedelta(minutes=15),
            "H1": pd.Timedelta(hours=1)}[timeframe]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an OHLCV CSV for the pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="Path to CSV (defaults to config).")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    loader = HistoricalDataLoader(cfg.data)

    print("=" * 70)
    print(f"Loading: {args.data or (cfg.paths.data / cfg.data.csv_filename)}")
    df = loader.load(args.data)

    print("\n--- Schema ---")
    print("Columns      :", list(df.columns))
    print("Rows         :", f"{len(df):,}")
    print("Index dtype  :", df.index.dtype)
    print("Date range   :", df.index.min(), "->", df.index.max())
    print("Duplicated ts:", int(df.index.duplicated().sum()))

    print("\n--- OHLCV sanity ---")
    print(df[["open", "high", "low", "close", "tick_volume"]].describe().T.to_string())
    bad_hl = int((df["high"] < df["low"]).sum())
    bad_range = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
                     (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    print(f"high<low rows           : {bad_hl}")
    print(f"OHLC out-of-range rows  : {bad_range}")

    print("\n--- Time gaps ---")
    deltas = df.index.to_series().diff().dropna()
    step = _expected_step(cfg.data.timeframe)
    gaps = deltas[deltas > step * 1.5]
    print(f"Expected step ({cfg.data.timeframe}): {step}")
    print(f"Bars with a larger-than-expected gap: {len(gaps):,} "
          f"(weekends/holidays are normal for FX)")
    if len(gaps):
        print(f"Largest gap: {gaps.max()} at {gaps.idxmax()}")

    print("\n--- Walk-forward split (config) ---")
    featured = FeatureEngineer(cfg.features).transform(df)
    train, test = loader.train_test_split(featured)
    print(f"test_start = {cfg.data.test_start}")
    print(f"Train bars : {len(train):,}  ({train.index.min()} -> {train.index.max()})")
    if len(test):
        print(f"Test  bars : {len(test):,}  ({test.index.min()} -> {test.index.max()})")
    else:
        print("Test  bars : 0  --> WARNING: no data on/after test_start. "
              "Your CSV may not include 2025; adjust data.test_start.")
    print("=" * 70)
    print("Ingestion OK." if len(train) else "Ingestion produced an empty training set!")


if __name__ == "__main__":
    main()
