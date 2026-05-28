"""
Optuna hyperparameter tuning script (V2 / Phase 3).

Searches PPO hyperparameters against a VALIDATION slice carved from the training
data — the 2025 test set is never used for tuning. Writes the best params to
``models/<name>_best_params.json`` so you can apply them to a full training run.

Examples
--------
    python scripts/tune.py --trials 20 --timesteps 30000 --metric sharpe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from optimization.optuna_tuning import PPOTuner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters with Optuna.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="OHLCV CSV (defaults to config).")
    parser.add_argument("--name", default=None, help="Model base name for output file.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=30_000, help="PPO steps per trial.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--metric", choices=["sharpe", "return", "calmar"], default="sharpe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.paths.ensure()
    name = args.name or cfg.training.model_name

    tuner = PPOTuner(
        cfg, n_trials=args.trials, timesteps_per_trial=args.timesteps,
        val_fraction=args.val_fraction, metric=args.metric,
    )
    study = tuner.optimize(csv_path=args.data)

    out = cfg.paths.models / f"{name}_best_params.json"
    payload = {
        "metric": args.metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.trials,
    }
    out.write_text(json.dumps(payload, indent=2))

    print("\nBest validation %s: %.4f" % (args.metric, study.best_value))
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nSaved best params to: {out}")
    print("Apply them by editing config/config.yaml [training] and running scripts/train.py.")


if __name__ == "__main__":
    main()
