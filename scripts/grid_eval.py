"""
V3.5 grid evaluation — the rigorous experimental protocol in one command.

Runs a seed x fold grid (vectorized-env training, scalar-env evaluation),
compares single-seed vs seed-ensemble vs buy-and-hold, and reports a
distribution + Robustness Score + bootstrap median CI. No single-run claims.

Examples
--------
    # Quick read
    python scripts/grid_eval.py --seeds 5 --folds 5 --timesteps 50000 --policy-arch cnn
    # Higher TF (Phase 5F) — set data.timeframe / supply an H1 CSV in config
    python scripts/grid_eval.py --policy-arch cnn --feature-groups trend volatility --correlation 0.95
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from validation.grid import GridEvaluator  # noqa: E402
from validation.splitters import TimeSeriesSplitter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Seed x fold grid evaluation (V3.5 protocol).")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--policy-arch", default=None, choices=["mlp", "lstm", "transformer", "cnn", "cnn_lstm"])
    p.add_argument("--seeds", type=int, default=5, help="Number of training seeds.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=50_000, help="PPO steps per (seed,fold) cell.")
    p.add_argument("--num-envs", type=int, default=32, help="Vectorized training lanes.")
    p.add_argument("--engine", default="sb3", choices=["sb3", "gpu"],
                   help="sb3 = CPU numpy VecEnv + SB3 PPO; gpu = GPU-resident "
                        "TorchVecGoldEnv + CleanRL PPO (cnn/mlp only). Eval is identical.")
    p.add_argument("--n-steps", type=int, default=128,
                   help="Rollout length per PPO update (gpu engine).")
    p.add_argument("--timeframe", default=None, choices=["M5", "M15", "H1", "H4", "D1"],
                   help="Resample the native CSV to this timeframe (Phase 5F). "
                        "Default = use the CSV as-is.")
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument("--vol-sizing", action="store_true",
                   help="Enable volatility-targeted position sizing (V3.5 reward redesign).")
    p.add_argument("--reward-mode", default=None, choices=["absolute", "excess", "dsr"],
                   help="excess = reward beating buy-and-hold; dsr = differential "
                        "Sharpe of excess (maximise information ratio).")
    p.add_argument("--regime", action="store_true",
                   help="Add 5B regime signals (regime_trend, regime_vol) to the observation.")
    p.add_argument("--feature-groups", nargs="*", default=None,
                   choices=["trend", "momentum", "volatility", "candle", "structure", "volume"])
    p.add_argument("--correlation", type=float, default=None)
    p.add_argument("--tag", default="grid", help="Output subdir tag.")
    p.add_argument("--min-trade-atr-pct", type=float, default=None,
                   help="Regime gate: block new entries when ATR/close < this. "
                        "Causal. 0.0 disables. Suggested H4 starting point: 0.004 "
                        "(blocks ~25% lowest-volatility bars).")
    p.add_argument("--ter-gate-window", type=int, default=None,
                   help="Eval-time Trend-Efficiency gate window (bars). 0 = disabled.")
    p.add_argument("--ter-gate-threshold", type=float, default=None,
                   help="Eval-time TER threshold; override new entries with HOLD "
                        "when TER < threshold. 0.0 = disabled.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.policy_arch:
        cfg.training.policy_arch = args.policy_arch
    if args.feature_groups is not None:
        cfg.features.feature_groups = args.feature_groups
    if args.correlation is not None:
        cfg.features.correlation_threshold = args.correlation
    if args.vol_sizing:
        cfg.env.volatility_target_sizing = True
    if args.reward_mode:
        cfg.env.reward_mode = args.reward_mode
    if args.regime:
        cfg.features.use_regime_features = True
    if args.min_trade_atr_pct is not None:
        cfg.env.min_trade_atr_pct = args.min_trade_atr_pct
    if args.ter_gate_window is not None:
        cfg.backtest.ter_gate_window = args.ter_gate_window
    if args.ter_gate_threshold is not None:
        cfg.backtest.ter_gate_threshold = args.ter_gate_threshold
    cfg.paths.ensure()

    splitter = TimeSeriesSplitter(n_splits=args.folds, mode="expanding", gap=cfg.env.window_size)
    evaluator = GridEvaluator(
        cfg, seeds=list(range(args.seeds)), splitter=splitter,
        timesteps_per_fold=args.timesteps, num_envs=args.num_envs,
        evaluate_ensemble=not args.no_ensemble, resample_to=args.timeframe,
        run_tag=args.tag, engine=args.engine, gpu_n_steps=args.n_steps,
    )
    result = evaluator.run(csv_path=args.data)

    out = cfg.paths.logs / "grid" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    result.cells.to_csv(out / "cells.csv", index=False)
    if not result.ensemble_cells.empty:
        result.ensemble_cells.to_csv(out / "ensemble_cells.csv", index=False)
    summary = {
        "policy_arch": cfg.training.policy_arch,
        "feature_groups": cfg.features.feature_groups,
        "baseline_buy_hold_pct": result.baseline_buy_hold_pct,
        "single": result.single.to_dict(),
        "ensemble": result.ensemble.to_dict() if result.ensemble else None,
        "median_ci": result.median_ci,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n" + result.summary())
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
