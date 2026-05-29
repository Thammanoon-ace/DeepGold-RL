"""Tier 2.1 speedup benchmark — scalar run_episode vs run_episode_torch_vec.

Trains one quick CleanRL ActorCritic, then evaluates the same model both ways
on the real XAUUSD test split. Reports wall-clock + final-equity drift so the
caller can verify the GPU-vec path is faster *and* still equivalent on real
data. Also benchmarks an EnsemblePolicy of K models, which is the case Tier 2.1
was actually written for (the big-seed grid's ensemble eval was the bottleneck).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from backtest.backtester import run_episode, run_episode_torch_vec
from config.config import Config
from env.env_builder import TradingDataPipeline, make_env_from_frame
from env.torch_vec_env import TorchVecGoldEnv
from policies.ensemble import EnsemblePolicy
from training.cleanrl_ppo import PPOConfig, train_cleanrl_ppo


def train_one(features, prices, env_cfg, seed: int) -> "ActorCritic":
    env = TorchVecGoldEnv(features, prices, env_cfg, num_envs=512,
                          random_start=True, device="cuda",
                          dtype=torch.float32, seed=seed)
    ppo = PPOConfig(total_timesteps=64_000, n_steps=64)
    ac, _ = train_cleanrl_ppo(env, arch="cnn", ppo=ppo, seed=seed)
    return ac


def time_eval(label: str, fn) -> dict:
    t0 = time.perf_counter()
    hist = fn()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<22} {elapsed:>7.2f} s   final_equity = {hist['final_equity']:,.2f}   "
          f"n_trades = {len(hist['trades'])}")
    return {"elapsed": elapsed, "hist": hist}


def main() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.env.reward_mode = "excess"

    pipe = TradingDataPipeline(cfg)
    train_df, test_df = pipe.prepare()
    feat_cols = pipe.feature_columns
    train_features = train_df[feat_cols].to_numpy(dtype=np.float32)
    train_prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    test_features = test_df[feat_cols].to_numpy(dtype=np.float32)
    test_prices = test_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    times = list(test_df.index)
    print(f"train bars={len(train_df)} | test bars={len(test_df)}")

    # ---- Train K small models (the bottleneck Tier 2.1 targets) ------- #
    K = 8
    print(f"\nTraining {K} CleanRL models (64k steps each)...")
    t0 = time.perf_counter()
    models = [train_one(train_features, train_prices, cfg.env, seed=s) for s in range(K)]
    print(f"  trained in {time.perf_counter() - t0:.1f}s")

    # ---- Single-seed eval ------------------------------------------- #
    print(f"\nSingle-seed eval (model 0, test bars={len(test_df)}):")
    scalar_env = make_env_from_frame(test_df, feat_cols, cfg.env, random_start=False)
    a = time_eval("scalar run_episode",
                  lambda: run_episode(models[0], scalar_env, deterministic=True))
    b = time_eval("torch_vec run_episode",
                  lambda: run_episode_torch_vec(
                      models[0], test_features, test_prices, cfg.env,
                      times=times, device="cuda"))
    drift = abs(a["hist"]["final_equity"] - b["hist"]["final_equity"]) \
        / a["hist"]["initial_balance"] * 100.0
    print(f"  speedup: {a['elapsed'] / b['elapsed']:.1f}x  | final-equity drift = {drift:.4f}%")

    # ---- Ensemble eval (the actual Tier 2.1 target) -------------------- #
    print(f"\nEnsemble eval (K={K} models, test bars={len(test_df)}):")
    ens = EnsemblePolicy(models)
    scalar_env = make_env_from_frame(test_df, feat_cols, cfg.env, random_start=False)
    a = time_eval("scalar run_episode",
                  lambda: run_episode(ens, scalar_env, deterministic=True))
    b = time_eval("torch_vec run_episode",
                  lambda: run_episode_torch_vec(
                      ens, test_features, test_prices, cfg.env,
                      times=times, device="cuda"))
    drift = abs(a["hist"]["final_equity"] - b["hist"]["final_equity"]) \
        / a["hist"]["initial_balance"] * 100.0
    print(f"  speedup: {a['elapsed'] / b['elapsed']:.1f}x  | final-equity drift = {drift:.4f}%")


if __name__ == "__main__":
    main()
