"""Equivalence test: run_episode_torch_vec must match the scalar run_episode
on the same data and policy, step for step (Tier 2.1).

Why: Tier 2.1 substitutes the GPU-vec eval path for the scalar env in
``validation/grid.py`` when the GPU engine is in use. Compatibility is only
acceptable if the resulting equity curve, trade list and performance numbers
are identical to the scalar path on the same inputs — otherwise the GPU-vec
"speed-up" would silently change what every published grid number means.

Setup mirrors ``_test_torch_equiv.py``: synthetic OHLC + features, no random
start, CPU float64 so we eliminate device/precision differences. A random
deterministic ``ActorCritic`` (same seed -> same weights -> same actions on
identical observations) is used as the policy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from backtest.backtester import run_episode, run_episode_torch_vec
from config.config import EnvConfig
from env.env_builder import make_env_from_frame
from env.gold_trading_env import N_ACCOUNT_FEATURES
from training.cleanrl_ppo import ActorCritic

rng = np.random.default_rng(42)
N_BARS, F, W = 1000, 4, 16
close = 2000.0 + np.cumsum(rng.normal(0, 4, N_BARS))
high = close + np.abs(rng.normal(0, 3, N_BARS))
low = close - np.abs(rng.normal(0, 3, N_BARS))
feats = rng.normal(0, 1, (N_BARS, F)).astype(np.float32)
prices = np.stack([high, low, close], axis=1)
times = pd.date_range("2024-01-01", periods=N_BARS, freq="5min")

# DataFrame view used by the scalar path (make_env_from_frame).
feat_cols = [f"f{i}" for i in range(F)]
df = pd.DataFrame(feats, columns=feat_cols, index=times)
df["high"] = prices[:, 0]; df["low"] = prices[:, 1]; df["close"] = prices[:, 2]
df["open"] = prices[:, 2]  # placeholder OHLC; scalar env uses high/low/close only


def build_policy(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    obs_dim = W * F + N_ACCOUNT_FEATURES
    ac = ActorCritic(obs_dim=obs_dim, window=W, n_features=F, arch="cnn").to("cpu").double()
    ac.eval()
    return ac


def run_mode(reward_mode: str) -> None:
    cfg = EnvConfig(window_size=W, reward_mode=reward_mode)
    ac = build_policy(seed=0)

    # --- Scalar path -------------------------------------------------- #
    scalar_env = make_env_from_frame(df, feat_cols, cfg, random_start=False)
    scalar_hist = run_episode(ac, scalar_env, deterministic=True)

    # --- Torch-vec path (CPU float64 to match scalar precision) ------- #
    torch_hist = run_episode_torch_vec(
        ac, feats, prices, cfg, times=list(times), device="cpu", dtype=torch.float64,
    )

    n_scalar = len(scalar_hist["equity_curve"])
    n_torch = len(torch_hist["equity_curve"])
    eq_diff = float(np.abs(np.asarray(scalar_hist["equity_curve"])
                           - np.asarray(torch_hist["equity_curve"])).max())
    fin_diff = abs(scalar_hist["final_equity"] - torch_hist["final_equity"])
    trades_scalar = len(scalar_hist["trades"])
    trades_torch = len(torch_hist["trades"])

    print(f"[{reward_mode:>8}] curve_len scalar={n_scalar} torch={n_torch} | "
          f"max eq err = {eq_diff:.3e} | final eq err = {fin_diff:.3e} | "
          f"trades scalar={trades_scalar} torch={trades_torch}")

    assert n_scalar == n_torch, "Equity curve lengths differ"
    assert trades_scalar == trades_torch, "Trade counts differ"
    assert eq_diff < 1e-6, f"Equity curve diverges ({eq_diff:.3e})"
    assert fin_diff < 1e-6, f"Final equity diverges ({fin_diff:.3e})"


def run_mode_float32_cuda(reward_mode: str) -> None:
    """Production-precision check: float32 on CUDA. Drift is expected to be
    tiny (< 0.5% on final equity) and trade count should still match exactly."""
    if not torch.cuda.is_available():
        print(f"[{reward_mode:>8}] (skip CUDA test — no GPU)")
        return
    cfg = EnvConfig(window_size=W, reward_mode=reward_mode)
    ac = build_policy(seed=0)
    scalar_env = make_env_from_frame(df, feat_cols, cfg, random_start=False)
    scalar_hist = run_episode(ac, scalar_env, deterministic=True)

    ac_gpu = build_policy(seed=0).to("cuda").float()
    torch_hist = run_episode_torch_vec(
        ac_gpu, feats, prices, cfg, times=list(times),
        device="cuda", dtype=torch.float32,
    )

    fin_diff_pct = abs(scalar_hist["final_equity"] - torch_hist["final_equity"]) \
        / scalar_hist["initial_balance"] * 100.0
    n_scalar, n_torch = len(scalar_hist["trades"]), len(torch_hist["trades"])
    print(f"[{reward_mode:>8}] (cuda fp32) final eq drift = {fin_diff_pct:.4f}% | "
          f"trades scalar={n_scalar} torch={n_torch}")
    assert fin_diff_pct < 0.5, f"float32 drift too large ({fin_diff_pct:.3f}%)"


def main() -> None:
    print("== CPU float64 (must be exact) ==")
    for mode in ("absolute", "excess", "dsr"):
        run_mode(mode)
    print("\n== CUDA float32 (small drift OK) ==")
    for mode in ("absolute", "excess", "dsr"):
        run_mode_float32_cuda(mode)
    print("\nOK: run_episode_torch_vec == run_episode across all reward modes.")


if __name__ == "__main__":
    main()
