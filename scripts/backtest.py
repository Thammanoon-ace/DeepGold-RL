"""
Example backtest script (requirement #11).

Evaluates a trained PPO agent on the held-out 2025 data, prints the performance
report and saves metrics + charts under ``logs/backtest/``.

Examples
--------
    python scripts/backtest.py
    python scripts/backtest.py --name ppo_gold_smoke
    python scripts/backtest.py --on train      # in-sample sanity check
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.backtester import Backtester  # noqa: E402
from config.config import Config  # noqa: E402
from env.env_builder import TradingDataPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a trained PPO agent on 2025 data.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--name", default=None, help="Model base name (defaults to config).")
    parser.add_argument("--model", default=None, help="Explicit path to model .zip.")
    parser.add_argument("--data", default=None, help="Path to OHLCV CSV (defaults to config).")
    parser.add_argument("--on", default="test", choices=["test", "train"], help="Which split.")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation.")
    parser.add_argument("--algo", default=None, choices=["ppo", "dqn"],
                        help="Algorithm of the saved model (defaults to config).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.name:
        cfg.training.model_name = args.name
    if args.algo:
        cfg.training.algo = args.algo
    if args.no_plots:
        cfg.backtest.save_plots = False
    cfg.paths.ensure()

    pipeline = TradingDataPipeline(cfg)
    pipeline.prepare(csv_path=args.data)

    backtester = Backtester(cfg, pipeline=pipeline)
    report = backtester.run(model_path=args.model, model_name=args.name, on=args.on)
    out_dir = backtester.save_results()

    print("\n" + report.pretty())
    print(f"Results + charts saved to: {out_dir}")


if __name__ == "__main__":
    main()
