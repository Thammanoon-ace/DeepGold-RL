"""
Architecture comparison experiment (V3 / Phase 4 — "does deep help?").

Trains the same PPO setup with different policy/feature-extractor architectures
(MLP, LSTM, Transformer, CNN) on identical data, then evaluates each on the
held-out 2025 set and tabulates the out-of-sample metrics side by side.

This is the honest test of V3: a heavier model is only worth it if it
generalizes *better out-of-sample*, not just fits training data. The data
pipeline (and thus the normalizer) is shared across architectures so the only
thing that varies is the network.

Examples
--------
    python scripts/compare_archs.py --timesteps 200000
    python scripts/compare_archs.py --archs mlp lstm --timesteps 100000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from backtest.backtester import Backtester
from config.config import Config
from env.env_builder import TradingDataPipeline
from training.train_ppo import PPOTrainer, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy architectures out-of-sample.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="OHLCV CSV (defaults to config).")
    parser.add_argument("--archs", nargs="+", default=["mlp", "lstm", "transformer", "cnn"],
                        choices=["mlp", "lstm", "transformer", "cnn"])
    parser.add_argument("--timesteps", type=int, default=200_000, help="PPO steps per architecture.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="Force a device; otherwise MLP->cpu, others->cuda-if-available.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.paths.ensure()

    # Shared, leakage-safe pipeline (prepared once, reused for every arch).
    pipeline = TradingDataPipeline(cfg)
    pipeline.prepare(csv_path=args.data)
    cuda = resolve_device("auto") == "cuda"

    rows = []
    for arch in args.archs:
        cfg.training.policy_arch = arch
        cfg.training.model_name = f"cmp_{arch}"
        # MLP is bottlenecked by env stepping -> CPU; sequence models like GPU.
        cfg.training.device = args.device or ("cpu" if arch == "mlp" else ("cuda" if cuda else "cpu"))

        logging.info("=== Training arch=%s on device=%s for %d steps ===",
                     arch, cfg.training.device, args.timesteps)
        t0 = time.time()
        trainer = PPOTrainer(cfg, pipeline=pipeline)
        trainer._build_vec_env()
        trainer.train(total_timesteps=args.timesteps)
        trainer.save()
        train_secs = time.time() - t0

        # Out-of-sample evaluation on 2025.
        bt = Backtester(cfg, pipeline=pipeline)
        rep = bt.run(model_name=f"cmp_{arch}", on="test")
        rows.append({
            "arch": arch,
            "device": cfg.training.device,
            "train_sec": round(train_secs, 1),
            "return_pct": round(rep.total_return_pct, 2),
            "sharpe": round(rep.sharpe_ratio, 2),
            "calmar": round(rep.calmar_ratio, 2),
            "max_dd_pct": round(rep.max_drawdown_pct, 2),
            "win_rate_pct": round(rep.win_rate_pct, 1),
            "profit_factor": round(rep.profit_factor, 2),
            "n_trades": rep.n_trades,
        })

    df = pd.DataFrame(rows)
    out = cfg.paths.logs / "arch_comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)

    print("\n" + "=" * 78)
    print(f"Architecture comparison — {args.timesteps:,} steps each, OOS on 2025")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\nSaved to: {out / 'comparison.csv'}")
    print("Reminder: more timesteps are needed for conclusive results; this is a "
          "framework demonstration, not a tuned benchmark.")


if __name__ == "__main__":
    main()
