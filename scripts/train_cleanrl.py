"""
GPU-native training entry point — CleanRL-style PPO on the torch-resident env.

This is the GPU-first counterpart to ``scripts/train_ppo.py`` / ``grid_eval.py``
(which train on the CPU numpy VecEnv with SB3). Here the rollout, GAE and PPO
update all run in torch on the chosen device, fed by :class:`TorchVecGoldEnv`,
so with ``--device cuda`` the GPU is the primary compute engine.

Data handling is identical to the rest of the project (leakage-safe): features
are engineered causally, the normalizer is fit on the TRAIN split only, and the
held-out test split is evaluated deterministically with the **scalar env +
run_episode** — the same single-path evaluation every other experiment uses. The
trained ActorCritic exposes an SB3-compatible ``predict`` so this works
unchanged.

Examples
--------
    # GPU run on the real M5 data
    python scripts/train_cleanrl.py --timesteps 300000 --num-envs 1024 --arch cnn
    # Quick CPU smoke test (no GPU contention)
    python scripts/train_cleanrl.py --device cpu --num-envs 8 --timesteps 4096 --n-steps 64
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from backtest.backtester import run_episode  # noqa: E402
from backtest.baselines import buy_and_hold_return  # noqa: E402
from backtest.metrics import compute_report  # noqa: E402
from config.config import BARS_PER_YEAR, Config  # noqa: E402
from env.env_builder import TradingDataPipeline  # noqa: E402
from env.torch_vec_env import TorchVecGoldEnv  # noqa: E402
from training.cleanrl_ppo import PPOConfig, train_cleanrl_ppo  # noqa: E402

logger = logging.getLogger("train_cleanrl")


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return name


def main() -> None:
    p = argparse.ArgumentParser(description="GPU-native CleanRL-style PPO training.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--arch", default="cnn", choices=["cnn", "mlp"])
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--num-envs", type=int, default=1024, help="GPU-resident lanes.")
    p.add_argument("--n-steps", type=int, default=256, help="Rollout length per update.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--reward-mode", default=None, choices=["absolute", "excess", "dsr"])
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--log-every", type=int, default=10, help="Updates between log lines (0=off).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.reward_mode:
        cfg.env.reward_mode = args.reward_mode
    cfg.paths.ensure()
    device = resolve_device(args.device)

    # ---- Leakage-safe data prep (fit normalizer on TRAIN only) ---------- #
    pipeline = TradingDataPipeline(cfg)
    train_df, test_df = pipeline.prepare(args.data)
    feat_cols = pipeline.feature_columns
    features = train_df[feat_cols].to_numpy(dtype=np.float32)
    prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    bars_per_year = BARS_PER_YEAR.get(cfg.data.timeframe, cfg.backtest.bars_per_year)

    logger.info("Device=%s | arch=%s | lanes=%d | n_steps=%d | reward=%s | steps=%d",
                device, args.arch, args.num_envs, args.n_steps, cfg.env.reward_mode, args.timesteps)
    logger.info("Train bars=%d (%d features) | test bars=%d",
                len(train_df), len(feat_cols), len(test_df))

    # ---- GPU-resident env + GPU-native PPO ------------------------------ #
    env = TorchVecGoldEnv(features, prices, cfg.env, num_envs=args.num_envs,
                          random_start=True, device=device, dtype=torch.float32, seed=args.seed)
    ppo = PPOConfig(total_timesteps=args.timesteps, n_steps=args.n_steps, learning_rate=args.lr)

    # Actual env interactions = full updates x batch (the loop floors
    # total_timesteps to whole updates of num_envs*n_steps).
    batch = args.num_envs * args.n_steps
    actual_updates = max(args.timesteps // batch, 1)
    actual_steps = actual_updates * batch
    if actual_steps != args.timesteps:
        logger.info("Rounded to %d updates x %d = %d env-steps (batch = lanes x n_steps).",
                    actual_updates, batch, actual_steps)

    t0 = time.perf_counter()
    ac, ep_returns = train_cleanrl_ppo(env, arch=args.arch, ppo=ppo, seed=args.seed,
                                       log_every=args.log_every)
    elapsed = time.perf_counter() - t0
    sps = actual_steps / max(elapsed, 1e-9)
    logger.info("Training done in %.1fs  (%.0f env-steps/s over %d steps, %d episodes seen)",
                elapsed, sps, actual_steps, len(ep_returns))

    # ---- Deterministic OOS evaluation (scalar env, single path) --------- #
    if test_df is None or test_df.empty:
        logger.warning("No test split configured (data.test_start); skipping OOS eval.")
        return
    eval_env = pipeline.make_env(which="test", random_start=False)
    hist = run_episode(ac, eval_env, deterministic=True)
    rep = compute_report(hist["equity_curve"], hist["trades"], hist["initial_balance"],
                         bars_per_year=bars_per_year)
    bh = buy_and_hold_return(test_df["close"].to_numpy())["total_return_pct"]

    print("\n" + "=" * 56)
    print("OUT-OF-SAMPLE EVALUATION (scalar env, deterministic)")
    print("=" * 56)
    print(f"  Return        : {rep.total_return_pct:+.2f}%")
    print(f"  Buy-and-hold  : {bh:+.2f}%   (excess {rep.total_return_pct - bh:+.2f}%)")
    print(f"  Sharpe        : {rep.sharpe_ratio:.3f}")
    print(f"  Max drawdown  : {rep.max_drawdown_pct:.2f}%")
    print(f"  Trades        : {rep.n_trades}")
    if ep_returns:
        print(f"  Train ep ret  : last50 mean {np.mean(ep_returns[-50:]):+.2f}%  "
              f"(n={len(ep_returns)})")
    print("\nNote: a single OOS path is not a verdict. For honest claims run the\n"
          "      seed x fold grid (scripts/grid_eval.py) — distribution + CI.")


if __name__ == "__main__":
    main()
