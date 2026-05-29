"""CPU-B+C bench: SB3 engine (CPU numpy VecEnv) at --num-envs 256 vs gpu engine.

Old box: SB3 grid was forced to small num-envs (RAM-bound, page-file thrash).
New box: 20 cores Arrow Lake + 32 GB RAM — bench shows whether the SB3 path,
which uses VectorizedGoldTradingEnv (numpy single-process batched) at larger
lane counts, can rival or beat the gpu engine for grid throughput.

Runs one (seed, fold) cell of identical timesteps for each engine and reports
wall-clock + env-steps/s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from config.config import Config
from env.env_builder import TradingDataPipeline
from env.torch_vec_env import TorchVecGoldEnv
from env.vectorized_env import VectorizedGoldTradingEnv
from policies.factory import build_policy_kwargs
from training.cleanrl_ppo import PPOConfig, train_cleanrl_ppo

TIMESTEPS = 1_310_720  # matches Tier 1.3 per-cell budget


def bench_sb3(train_df, feat_cols, cfg: Config, num_envs: int) -> float:
    tcfg = cfg.training
    features = train_df[feat_cols].to_numpy(dtype=np.float32)
    prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    vec = VectorizedGoldTradingEnv(features, prices, cfg.env, num_envs=num_envs,
                                   random_start=True, seed=0)
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=tcfg.gamma)
    model = PPO(
        policy=tcfg.policy, env=vec,
        learning_rate=tcfg.learning_rate, n_steps=tcfg.n_steps,
        batch_size=tcfg.batch_size, n_epochs=tcfg.n_epochs,
        gamma=tcfg.gamma, gae_lambda=tcfg.gae_lambda, clip_range=tcfg.clip_range,
        ent_coef=tcfg.ent_coef, vf_coef=tcfg.vf_coef, max_grad_norm=tcfg.max_grad_norm,
        policy_kwargs=build_policy_kwargs(tcfg, cfg.env.window_size),
        device="cpu", seed=0, verbose=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=TIMESTEPS)
    return time.perf_counter() - t0


def bench_gpu(train_df, feat_cols, cfg: Config, num_envs: int) -> float:
    features = train_df[feat_cols].to_numpy(dtype=np.float32)
    prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    env = TorchVecGoldEnv(features, prices, cfg.env, num_envs=num_envs,
                          random_start=True, device="cuda",
                          dtype=torch.float32, seed=0)
    ppo = PPOConfig(total_timesteps=TIMESTEPS, n_steps=128)
    t0 = time.perf_counter()
    train_cleanrl_ppo(env, arch="cnn", ppo=ppo, seed=0)
    return time.perf_counter() - t0


def main() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.training.policy_arch = "cnn"
    cfg.env.reward_mode = "excess"
    pipe = TradingDataPipeline(cfg)
    train_df, _ = pipe.prepare()
    feat_cols = pipe.feature_columns
    print(f"train bars={len(train_df)} | features={len(feat_cols)} | timesteps={TIMESTEPS:,}")

    # SB3 engine: numpy VectorizedGoldTradingEnv at 256 lanes (old box OOM'd ~64).
    print("\n== SB3 engine (CPU numpy VecEnv, 256 lanes) ==")
    el = bench_sb3(train_df, feat_cols, cfg, num_envs=256)
    sps_sb3 = TIMESTEPS / max(el, 1e-9)
    print(f"  elapsed {el:.1f} s  ->  {sps_sb3:,.0f} env-steps/s")

    # GPU engine: TorchVecGoldEnv at 2048 lanes (Tier 1.3 cell config).
    print("\n== GPU engine (TorchVecGoldEnv, 2048 lanes) ==")
    el = bench_gpu(train_df, feat_cols, cfg, num_envs=2048)
    sps_gpu = TIMESTEPS / max(el, 1e-9)
    print(f"  elapsed {el:.1f} s  ->  {sps_gpu:,.0f} env-steps/s")

    print(f"\nGPU / SB3 ratio: {sps_gpu / sps_sb3:.2f}x")


if __name__ == "__main__":
    main()
