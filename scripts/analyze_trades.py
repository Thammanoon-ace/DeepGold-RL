"""
Trade-distribution analysis for a trained model on the 2025 test set.

Answers the key "is this edge real or luck?" question by inspecting *where* the
profit comes from: concentration in a few outlier trades, exit-reason mix,
win/loss streaks, and how the result survives removing the best trades.

Usage:
    python scripts/analyze_trades.py --name cmp_lstm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backtest.backtester import Backtester
from backtest.metrics import compute_trade_distribution
from config.config import Config
from env.env_builder import TradingDataPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a model's 2025 trade distribution.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--name", required=True, help="Model base name (e.g. cmp_lstm).")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--on", default="test", choices=["test", "train"])
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.training.algo = args.algo
    cfg.backtest.save_plots = False

    pipeline = TradingDataPipeline(cfg)
    pipeline.prepare(csv_path=args.data)
    bt = Backtester(cfg, pipeline=pipeline)
    report = bt.run(model_name=args.name, on=args.on)

    trades = bt.history["trades"]
    dist = compute_trade_distribution(trades)
    pnls = np.array([t["pnl"] for t in trades], dtype=float)

    print("\n" + "=" * 60)
    print(f"Trade distribution — {args.name} ({args.on})")
    print("=" * 60)
    print(report.pretty())

    if len(pnls) == 0:
        print("No trades.")
        return

    total = pnls.sum()
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    order = np.argsort(pnls)[::-1]           # best -> worst

    print("Distribution detail")
    print("-" * 60)
    print(f"  Trades              : {len(pnls)}")
    print(f"  Net PnL (sum)       : {total:+,.2f}")
    print(f"  Best / worst trade  : {pnls.max():+,.2f} / {pnls.min():+,.2f}")
    print(f"  PnL p05/median/p95  : {dist['pnl_p05']:+,.2f} / "
          f"{dist['pnl_median']:+,.2f} / {dist['pnl_p95']:+,.2f}")
    print(f"  Max consec W / L    : {dist['max_consecutive_wins']} / "
          f"{dist['max_consecutive_losses']}")
    print(f"  Avg bars held       : {dist['avg_bars_held']:.1f}")
    print(f"  Exit reasons        : {dist['exit_reasons']}")

    # --- Concentration: how much of net profit comes from the top trades? --- #
    print("\nProfit concentration (is the edge broad or a few lucky trades?)")
    print("-" * 60)
    for k in (1, 5, 10):
        if k <= len(pnls):
            topk = pnls[order[:k]].sum()
            share = (topk / total * 100.0) if total != 0 else float("nan")
            print(f"  Top {k:2d} trades contribute : {topk:+,.2f}  ({share:.0f}% of net PnL)")
    # Net PnL if the single best trade is removed (robustness check).
    if len(pnls) > 1:
        without_best = total - pnls.max()
        print(f"  Net PnL excl. best trade : {without_best:+,.2f}")
        without_top5 = total - pnls[order[:5]].sum()
        print(f"  Net PnL excl. top-5      : {without_top5:+,.2f}")
    print(f"  Winners / losers     : {len(wins)} / {len(losses)}")
    print("=" * 60)
    print("Heuristic: if a large share of net PnL comes from 1-5 trades, the "
          "result is fragile — confirm with walk-forward across many folds.")


if __name__ == "__main__":
    main()
