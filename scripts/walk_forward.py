"""
Walk-forward validation script (V2 / Phase 3).

Trains and evaluates the agent across multiple rolling/expanding out-of-sample
folds, then reports cross-fold stability and saves a stitched OOS equity curve.

Examples
--------
    python scripts/walk_forward.py                              # 5 expanding folds
    python scripts/walk_forward.py --folds 6 --mode rolling --timesteps 30000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from utils import visualization as viz  # noqa: E402
from validation.splitters import TimeSeriesSplitter  # noqa: E402
from validation.walk_forward import WalkForwardValidator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="OHLCV CSV (defaults to config).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--test-size", type=int, default=None, help="Bars per test fold.")
    parser.add_argument("--gap", type=int, default=None, help="Embargo bars (default=window_size).")
    parser.add_argument("--timesteps", type=int, default=50_000, help="PPO steps per fold.")
    parser.add_argument("--policy-arch", default=None,
                        choices=["mlp", "lstm", "transformer", "cnn", "cnn_lstm"],
                        help="Architecture to validate (defaults to config).")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--feature-groups", nargs="*", default=None,
                        choices=["trend", "momentum", "volatility", "candle", "structure", "volume"],
                        help="Phase 4B feature groups to enable (overrides config).")
    parser.add_argument("--correlation", type=float, default=None,
                        help="Correlation threshold for feature pruning (>0 enables).")
    parser.add_argument("--out-tag", default=None,
                        help="Suffix for the output subdir (avoids overwriting prior runs).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.policy_arch:
        cfg.training.policy_arch = args.policy_arch
    if args.device:
        cfg.training.device = args.device
    if args.feature_groups is not None:
        cfg.features.feature_groups = args.feature_groups
    if args.correlation is not None:
        cfg.features.correlation_threshold = args.correlation
    cfg.paths.ensure()

    splitter = TimeSeriesSplitter(
        n_splits=args.folds,
        test_size=args.test_size,
        mode=args.mode,
        gap=args.gap if args.gap is not None else cfg.env.window_size,
    )
    validator = WalkForwardValidator(cfg, timesteps_per_fold=args.timesteps, splitter=splitter)
    result = validator.run(csv_path=args.data)

    # Per-architecture subdirectory so multiple runs don't overwrite each other.
    subdir = cfg.training.policy_arch + (f"_{args.out_tag}" if args.out_tag else "")
    out_dir = cfg.paths.logs / "walk_forward" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_frame().to_csv(out_dir / "folds.csv", index=False)
    (out_dir / "aggregate.json").write_text(json.dumps(result.aggregate(), indent=2))
    viz.plot_equity_curve(
        result.stitched_equity, timestamps=result.stitched_timestamps,
        initial_balance=cfg.env.initial_balance,
        title="Walk-Forward Stitched Out-of-Sample Equity",
        save_path=out_dir / "stitched_equity.png",
    )

    print("\n" + result.summary())
    print(result.to_frame().to_string(index=False))
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
