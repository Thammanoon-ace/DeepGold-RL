"""Equivalence test: VectorizedGoldTradingEnv(N=1) must match GoldTradingEnv
step-for-step on an identical action sequence. Guards scientific validity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import EnvConfig
from env.gold_trading_env import GoldTradingEnv
from env.vectorized_env import VectorizedGoldTradingEnv

rng = np.random.default_rng(123)
N_BARS, F, W = 2500, 3, 16

# Random-walk price (so SL/TP and sizing actually trigger) + random features.
close = 2000.0 + np.cumsum(rng.normal(0, 4, N_BARS))
high = close + np.abs(rng.normal(0, 3, N_BARS))
low = close - np.abs(rng.normal(0, 3, N_BARS))
feats = rng.normal(0, 1, (N_BARS, F)).astype(np.float32)
idx = pd.date_range("2024-01-01", periods=N_BARS, freq="5min")
prices_df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

cfg = EnvConfig(window_size=W)

scalar = GoldTradingEnv(feats, prices_df, cfg, random_start=False)
vec = VectorizedGoldTradingEnv(feats, prices_df[["high", "low", "close"]].to_numpy(),
                               cfg, num_envs=1, random_start=False)

o_s, _ = scalar.reset()
o_v = vec.reset()[0]
assert np.allclose(o_s, o_v, atol=1e-4), f"reset obs mismatch: {np.abs(o_s - o_v).max()}"

act_rng = np.random.default_rng(7)
max_obs_err = max_rew_err = max_eq_err = 0.0
steps = 0
done = False
while not done and steps < N_BARS:
    a = int(act_rng.integers(0, 4))
    o_s, r_s, term, trunc, info_s = scalar.step(a)
    o_v_all, r_v, dones_v, infos_v = vec.step(np.array([a]))
    r_v = float(r_v[0]); done_v = bool(dones_v[0])
    eq_v = vec.equity[0] if not done_v else infos_v[0]["equity"]

    max_rew_err = max(max_rew_err, abs(r_s - r_v))
    max_eq_err = max(max_eq_err, abs(scalar.equity - eq_v))
    if not done_v:
        max_obs_err = max(max_obs_err, float(np.abs(o_s - o_v_all[0]).max()))
    else:
        term_obs = infos_v[0]["terminal_observation"]
        max_obs_err = max(max_obs_err, float(np.abs(o_s - term_obs).max()))
        assert (term or trunc), "scalar not done but vec done"
    done = term or trunc
    steps += 1

print(f"steps compared      : {steps}")
print(f"max reward error    : {max_rew_err:.3e}")
print(f"max equity error    : {max_eq_err:.3e}")
print(f"max observation err : {max_obs_err:.3e}")
print(f"scalar trades       : {scalar._n_trades} | vec trades: {int(vec.n_trades[0]) if steps and not done else 'n/a'}")
assert max_rew_err < 1e-6, "reward mismatch!"
assert max_eq_err < 1e-6, "equity mismatch!"
assert max_obs_err < 1e-4, "observation mismatch!"
print("VEC-EQUIV-OK")
