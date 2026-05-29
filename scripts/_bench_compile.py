"""Tier 2.4 speedup benchmark — torch.compile(ActorCritic) on/off.

Runs the same workload twice (no-compile, then compile) and reports env-steps/s
to isolate the compile speedup from initialization noise. The first compile
update pays a one-off graph build (5–30 s on this codebase); the bench prints
both the first-update time and the steady-state throughput so the speedup is
not inflated by amortising the compile cost.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config.config import Config
from env.env_builder import TradingDataPipeline
from env.torch_vec_env import TorchVecGoldEnv
from training.cleanrl_ppo import PPOConfig, train_cleanrl_ppo


def run(label: str, ppo: PPOConfig, features, prices, env_cfg, seed: int = 0) -> float:
    env = TorchVecGoldEnv(features, prices, env_cfg,
                          num_envs=16_384, random_start=True,
                          device="cuda", dtype=torch.float32, seed=seed)
    t0 = time.perf_counter()
    ac, _ = train_cleanrl_ppo(env, arch="cnn", ppo=ppo, seed=seed)
    elapsed = time.perf_counter() - t0
    steps = (ppo.total_timesteps // (env.num_envs * ppo.n_steps)) * env.num_envs * ppo.n_steps
    sps = steps / max(elapsed, 1e-9)
    print(f"  {label:<20} {elapsed:>7.1f} s  ->  {sps:>10,.0f} steps/s   ({steps:,} steps)")
    del env, ac
    torch.cuda.empty_cache()
    return sps


def main() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.env.reward_mode = "excess"
    pipe = TradingDataPipeline(cfg)
    train_df, _ = pipe.prepare()
    features = train_df[pipe.feature_columns].to_numpy(dtype=np.float32)
    prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)

    # 16k lanes × 64 n_steps × 5 updates = 5.24M env-steps. Long enough that
    # the compile graph-build (one-off) is amortised over multiple updates.
    base = PPOConfig(total_timesteps=5_242_880, n_steps=64, compile=False)
    comp = PPOConfig(total_timesteps=5_242_880, n_steps=64, compile=True)

    print("== Baseline (no compile) ==")
    sps_base = run("no compile", base, features, prices, cfg.env)
    print("\n== torch.compile (reduce-overhead) ==")
    sps_comp = run("compile", comp, features, prices, cfg.env)

    print(f"\nSpeedup: {sps_comp / sps_base:.2f}x  ({sps_comp:,.0f} / {sps_base:,.0f} steps/s)")


if __name__ == "__main__":
    main()
