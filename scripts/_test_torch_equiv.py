"""Equivalence test: TorchVecGoldEnv must match the (already scalar-verified)
numpy VectorizedGoldTradingEnv step-for-step on an identical action sequence.

Run in float64 on CPU so the only differences are floating-point order, not
device/precision. The numpy vec env is proven byte-equivalent to the scalar env
(scripts/_test_vec_equiv.py), so matching it transitively validates the torch
env against the canonical scalar simulator. Guards scientific validity: training
on the GPU env must simulate exactly the same market the backtester does.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config.config import EnvConfig
from env.torch_vec_env import TorchVecGoldEnv
from env.vectorized_env import VectorizedGoldTradingEnv

rng = np.random.default_rng(123)
N_BARS, F, W, N_ENVS, STEPS = 3000, 4, 16, 4, 2500

# Random-walk price (so SL/TP and sizing actually trigger) + random features.
close = 2000.0 + np.cumsum(rng.normal(0, 4, N_BARS))
high = close + np.abs(rng.normal(0, 3, N_BARS))
low = close - np.abs(rng.normal(0, 3, N_BARS))
feats = rng.normal(0, 1, (N_BARS, F)).astype(np.float64)
prices = np.stack([high, low, close], axis=1)


def run_mode(reward_mode: str) -> None:
    cfg = EnvConfig(window_size=W, reward_mode=reward_mode)

    npy = VectorizedGoldTradingEnv(feats, prices, cfg, num_envs=N_ENVS, random_start=False)
    tch = TorchVecGoldEnv(feats, prices, cfg, num_envs=N_ENVS, random_start=False,
                          device="cpu", dtype=torch.float64)

    o_n = npy.reset()
    o_t = tch.reset().numpy()
    obs_err = float(np.abs(o_n - o_t).max())

    act_rng = np.random.default_rng(7)
    max_obs = max_rew = max_eq = obs_err
    for _ in range(STEPS):
        a = act_rng.integers(0, 4, size=N_ENVS)
        o_n, r_n, d_n, _ = npy.step(a)
        o_t, r_t, d_t = tch.step(torch.as_tensor(a))
        o_t, r_t, d_t = o_t.numpy(), r_t.numpy(), d_t.numpy().astype(bool)

        max_rew = max(max_rew, float(np.abs(r_n - r_t).max()))
        # Equity compared on lanes that did NOT reset this step (post-reset
        # equity is initial_balance in both, so it would always match anyway).
        live = ~d_n
        if live.any():
            max_eq = max(max_eq, float(np.abs(npy.equity[live] - tch.equity.numpy()[live]).max()))
            max_obs = max(max_obs, float(np.abs(o_n[live] - o_t[live]).max()))
        assert np.array_equal(d_n, d_t), f"[{reward_mode}] done-mask mismatch"

    print(f"[{reward_mode:8s}] steps={STEPS}  obs_err={max_obs:.3e}  "
          f"rew_err={max_rew:.3e}  eq_err={max_eq:.3e}")
    assert max_rew < 1e-5, f"[{reward_mode}] reward mismatch {max_rew}"
    assert max_eq < 1e-6, f"[{reward_mode}] equity mismatch {max_eq}"
    assert max_obs < 1e-4, f"[{reward_mode}] observation mismatch {max_obs}"


if __name__ == "__main__":
    for mode in ("absolute", "excess", "dsr"):
        run_mode(mode)
    print("TORCH-EQUIV-OK")
