"""
Example training script (requirement #11).

Trains a PPO agent on pre-2025 XAUUSD data and saves the model, the
VecNormalize statistics and the feature normalizer under ``models/``.

Examples
--------
Fresh training run (uses config/config.yaml)::

    python scripts/train.py

Shorter smoke run + custom name::

    python scripts/train.py --timesteps 20000 --name ppo_gold_smoke

Resume from a checkpoint::

    python scripts/train.py --resume models/checkpoints/ppo_gold_100000_steps.zip
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from env.env_builder import TradingDataPipeline  # noqa: E402
from training.train_ppo import PPOTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO gold-trading agent.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--name", default=None, help="Override saved model name.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .zip to resume from.")
    parser.add_argument("--data", default=None, help="Path to OHLCV CSV (defaults to config).")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv (multiprocess).")
    parser.add_argument("--policy-arch", default=None,
                        choices=["mlp", "lstm", "transformer", "cnn", "cnn_lstm"],
                        help="Override the policy/feature-extractor architecture (V3).")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="Override compute device (MLP is often faster on CPU).")
    parser.add_argument("--algo", default=None, choices=["ppo", "dqn"],
                        help="RL algorithm: ppo (default) or dqn (off-policy).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.name:
        cfg.training.model_name = args.name
    if args.policy_arch:
        cfg.training.policy_arch = args.policy_arch
    if args.device:
        cfg.training.device = args.device
    if args.algo:
        cfg.training.algo = args.algo
    cfg.paths.ensure()

    # 1. Build the leakage-safe data pipeline (fit normalizer on train only).
    pipeline = TradingDataPipeline(cfg)
    pipeline.prepare(csv_path=args.data)

    # 2. Train.
    trainer = PPOTrainer(cfg, pipeline=pipeline)
    trainer._build_vec_env(use_subproc=args.subproc)
    trainer.train(total_timesteps=args.timesteps, resume_from=args.resume)

    # 3. Persist everything needed for backtesting / live trading.
    model_path = trainer.save()
    print(f"\nTraining complete. Model saved to: {model_path}")
    print(f"TensorBoard logs: {cfg.paths.logs / cfg.training.tensorboard_subdir}")
    print("Launch TensorBoard with:")
    print(f"    tensorboard --logdir \"{cfg.paths.logs / cfg.training.tensorboard_subdir}\"")


if __name__ == "__main__":
    main()
