"""4-run risk-adjusted analysis — Sharpe / Max-DD distribution across the
four H4 + cosine + SWA runs (512 cells total), vs buy-and-hold per fold.

Question: does the agent **risk-adjusted** beat buy-and-hold even though it
ties / trails on raw return ([[four-run-honest-verdict]])? If yes, the
"tradeable for Sharpe / DD users" framing from [[cosine-swa-unblock]] is
rescuable. If no, the project is closer to its original CLAUDE.md
negative-result wrap-up.

Reports:
- Per-cell Sharpe distribution (4 × 128 = 512 cells).
- Per-cell Max DD distribution.
- BH per-fold Sharpe + Max DD (computed deterministically from the test
  equity curve of a buy-and-hold position on the same fold's test slice).
- Cross-run comparison: median / P25 / P75 Sharpe per fold for agent vs BH.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtest.baselines import buy_and_hold_return
from backtest.metrics import max_drawdown, sharpe_ratio
from config.config import BARS_PER_YEAR, Config
from env.env_builder import TradingDataPipeline
from utils.normalization import FeatureNormalizer
from validation.splitters import TimeSeriesSplitter


def compute_bh_per_fold(cfg: Config) -> dict[int, dict]:
    """Run the same M5->H4 resample + fold split as grid_eval, then compute
    deterministic BH Sharpe / DD per fold on the test slice."""
    cfg.data.timeframe = "H4"
    pipe = TradingDataPipeline(cfg)
    raw = pipe.loader.load()
    raw = pipe.loader.resample(raw, "H4")
    featured = pipe.engineer.transform(raw)
    splitter = TimeSeriesSplitter(n_splits=5, mode="expanding",
                                  gap=cfg.env.window_size)
    folds = splitter.split(len(featured))
    bars_per_year = BARS_PER_YEAR.get("H4", cfg.backtest.bars_per_year)

    out = {}
    for fold in folds:
        test_raw = featured.iloc[fold.test_start:fold.test_end]
        close = test_raw["close"].to_numpy(dtype=float)
        init = cfg.env.initial_balance
        # Buy-and-hold equity curve: bh_units * close[t], where bh_units sized
        # at start so initial equity = init.
        bh_units = init / close[0]
        equity = bh_units * close
        rets = np.diff(equity) / equity[:-1]
        rets = rets[np.isfinite(rets)]
        bh_sharpe = sharpe_ratio(equity, periods_per_year=bars_per_year)
        bh_dd_pct = max_drawdown(equity) * 100.0
        bh_return_pct = (equity[-1] / init - 1.0) * 100.0
        out[fold.index] = {"bh_return_pct": bh_return_pct,
                           "bh_sharpe": bh_sharpe,
                           "bh_dd_pct": bh_dd_pct,
                           "n_bars": len(close)}
    return out


def load_all_cells() -> pd.DataFrame:
    tags = [
        "excess_bigseed_32_h4_cosine_swa",
        "excess_bigseed_32_h4_cosine_swa_s32",
        "excess_bigseed_32_h4_cosine_swa_s64",
        "excess_bigseed_32_h4_cosine_swa_s96",
    ]
    frames = []
    for tag in tags:
        path = Path("logs/grid") / tag / "cells.csv"
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        df = pd.read_csv(path)
        df["run"] = tag.split("_s")[-1] if "_s" in tag else "0"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.env.reward_mode = "excess"

    print("=== Buy-and-hold per fold (H4 test slices) ===")
    bh = compute_bh_per_fold(cfg)
    for f, m in bh.items():
        print(f"  fold {f}: BH return {m['bh_return_pct']:+7.2f}%  "
              f"Sharpe {m['bh_sharpe']:6.3f}  DD {m['bh_dd_pct']:5.2f}%  "
              f"bars {m['n_bars']}")
    print()

    cells = load_all_cells()
    if cells.empty:
        print("No cells.csv files found.")
        return
    print(f"Loaded {len(cells)} cells from {cells['run'].nunique()} runs")
    print()

    print("=== Agent per-fold distribution (median / IQR across 4 runs) ===")
    print(f"  {'fold':>4} {'BH_sh':>7} {'BH_dd':>6} | "
          f"{'med_sh':>7} {'p25_sh':>7} {'p75_sh':>7} {'beats BH on Sharpe %':>22} | "
          f"{'med_dd':>6} {'beats BH on DD %':>18}")
    for f in sorted(cells["fold"].unique()):
        sub = cells[cells["fold"] == f]
        bh_sh = bh[f]["bh_sharpe"]
        bh_dd = bh[f]["bh_dd_pct"]
        med_sh = sub["sharpe"].median()
        p25_sh = sub["sharpe"].quantile(0.25)
        p75_sh = sub["sharpe"].quantile(0.75)
        beats_bh_sharpe = (sub["sharpe"] > bh_sh).mean() * 100.0
        med_dd = sub["max_dd_pct"].median()
        beats_bh_dd = (sub["max_dd_pct"] < bh_dd).mean() * 100.0
        print(f"  {f:>4} {bh_sh:>7.3f} {bh_dd:>5.2f}% | "
              f"{med_sh:>7.3f} {p25_sh:>7.3f} {p75_sh:>7.3f} {beats_bh_sharpe:>21.1f}% | "
              f"{med_dd:>5.2f}% {beats_bh_dd:>17.1f}%")
    print()

    print("=== Overall (across all 512 cells) ===")
    overall_sharpe = cells["sharpe"]
    overall_dd = cells["max_dd_pct"]
    overall_ret = cells["return_pct"]
    bh_sharpe_avg = np.mean([m["bh_sharpe"] for m in bh.values()])
    bh_dd_avg = np.mean([m["bh_dd_pct"] for m in bh.values()])
    bh_ret_avg = np.mean([m["bh_return_pct"] for m in bh.values()])
    print(f"  Agent Sharpe : median {overall_sharpe.median():.3f}  "
          f"mean {overall_sharpe.mean():.3f}  std {overall_sharpe.std():.3f}")
    print(f"  Agent Max DD : median {overall_dd.median():.2f}%  "
          f"mean {overall_dd.mean():.2f}%")
    print(f"  Agent Return : median {overall_ret.median():.2f}%  "
          f"mean {overall_ret.mean():.2f}%")
    print(f"  BH (avg)     : Sharpe {bh_sharpe_avg:.3f}  "
          f"DD {bh_dd_avg:.2f}%  Return {bh_ret_avg:.2f}%")
    print()

    # Per-fold deltas
    deltas_sharpe = []
    deltas_dd = []
    deltas_ret = []
    for f in sorted(cells["fold"].unique()):
        sub = cells[cells["fold"] == f]
        deltas_sharpe.append(sub["sharpe"].median() - bh[f]["bh_sharpe"])
        deltas_dd.append(bh[f]["bh_dd_pct"] - sub["max_dd_pct"].median())  # DD smaller is better
        deltas_ret.append(sub["return_pct"].median() - bh[f]["bh_return_pct"])
    print("=== Median agent − BH per fold ===")
    print(f"  fold 0/1/2/3 Sharpe delta : "
          f"{deltas_sharpe[0]:+.2f} / {deltas_sharpe[1]:+.2f} / "
          f"{deltas_sharpe[2]:+.2f} / {deltas_sharpe[3]:+.2f}")
    print(f"  fold 0/1/2/3 DD reduction : "
          f"{deltas_dd[0]:+.2f} / {deltas_dd[1]:+.2f} / "
          f"{deltas_dd[2]:+.2f} / {deltas_dd[3]:+.2f}  (positive = agent better)")
    print(f"  fold 0/1/2/3 Return delta : "
          f"{deltas_ret[0]:+.2f} / {deltas_ret[1]:+.2f} / "
          f"{deltas_ret[2]:+.2f} / {deltas_ret[3]:+.2f}")
    print()

    print("=== Verdict ===")
    avg_sh_delta = np.mean(deltas_sharpe)
    avg_dd_delta = np.mean(deltas_dd)
    avg_ret_delta = np.mean(deltas_ret)
    print(f"  Avg Sharpe delta (median agent − BH) : {avg_sh_delta:+.3f}")
    print(f"  Avg DD reduction (BH − agent)        : {avg_dd_delta:+.2f}%")
    print(f"  Avg Return delta (median agent − BH) : {avg_ret_delta:+.2f}%")


if __name__ == "__main__":
    main()
