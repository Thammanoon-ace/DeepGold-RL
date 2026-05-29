"""CPU-A bench: SubprocVecEnv vs DummyVecEnv for the SB3 path (Tier 1.1 retry).

Old-machine project notes flagged SubprocVecEnv as a memory/IPC loss on the
1650 Ti box (each worker spawned a torch interpreter, page-file thrashing,
and the env stepping was too cheap for IPC to amortise). On 32 GB DDR5 + 20
cores we have room — bench shows whether the verdict flips.

This script bypasses ``scripts/train.py`` so we can A/B the same trainer with
``n_envs=8`` and the same PPO config, only changing the vec class.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from config.config import Config
from env.env_builder import TradingDataPipeline, make_env_from_frame
from policies.factory import build_policy_kwargs


def _make_env_fn(train_df, feat_cols, env_cfg):
    def _make():
        return Monitor(make_env_from_frame(train_df, feat_cols, env_cfg, random_start=True))
    return _make


def run(label: str, vec_cls, n_envs: int, train_df, feat_cols, cfg: Config,
        timesteps: int = 20_000) -> float:
    env_fns = [_make_env_fn(train_df, feat_cols, cfg.env) for _ in range(n_envs)]
    if vec_cls is SubprocVecEnv:
        vec = vec_cls(env_fns, start_method="spawn")
    else:
        vec = vec_cls(env_fns)
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=cfg.training.gamma)
    tcfg = cfg.training
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
    model.learn(total_timesteps=timesteps)
    elapsed = time.perf_counter() - t0
    sps = timesteps / max(elapsed, 1e-9)
    print(f"  {label:<20} (n_envs={n_envs}) {elapsed:>6.1f} s  ->  {sps:>8,.0f} env-steps/s")
    vec.close()
    return sps


def main() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.training.policy_arch = "cnn"
    cfg.env.reward_mode = "excess"
    pipe = TradingDataPipeline(cfg)
    train_df, _ = pipe.prepare()
    feat_cols = pipe.feature_columns
    print(f"train bars={len(train_df)} | features={len(feat_cols)}")

    # Use 8 workers — RAM 32 GB makes this trivial on the new box (page-file
    # thrashing was the old-rig issue).
    N = 8
    TS = 20_000

    print(f"\n== DummyVecEnv ({N} envs, {TS:,} steps, CPU PPO) ==")
    sps_d = run("DummyVecEnv", DummyVecEnv, N, train_df, feat_cols, cfg, timesteps=TS)

    print(f"\n== SubprocVecEnv ({N} workers) ==")
    sps_s = run("SubprocVecEnv", SubprocVecEnv, N, train_df, feat_cols, cfg, timesteps=TS)

    print(f"\nSpeedup (subproc / dummy): {sps_s / sps_d:.2f}x")


if __name__ == "__main__":
    main()
