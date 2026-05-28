"""
Benchmark baselines (V3.5 evaluation rigor).

Any RL result must be judged *relative to* trivial baselines, not in absolute
terms. The two that matter most for XAUUSD:

* **Buy-and-hold** — what you'd get just holding gold over the test window. A
  long-only RL agent in a bull regime can look brilliant while merely tracking
  this; it must be beaten to claim skill.
* **Random / always-flat** — shows the cost drag. If a random (cost-aware)
  policy and "do nothing" are hard to beat, there is no edge.

All baselines are computed with the same costs/sizing as the agent (via the
environment) so the comparison is apples-to-apples.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from backtest.backtester import run_episode
from backtest.metrics import compute_report
from env.gold_trading_env import ACTION_BUY, ACTION_HOLD, GoldTradingEnv


def buy_and_hold_return(close: np.ndarray) -> Dict[str, float]:
    """Pure price buy-and-hold over the series (no costs, no leverage).

    The natural 'just hold the asset' benchmark. Returns total and the per-bar
    Sharpe of holding.
    """
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        return {"total_return_pct": 0.0, "sharpe": 0.0}
    total = (close[-1] / close[0] - 1.0) * 100.0
    rets = np.diff(close) / close[:-1]
    sharpe = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0
    return {"total_return_pct": total, "sharpe": sharpe}


def _scripted_episode(env: GoldTradingEnv, action_fn) -> Dict:
    """Run a scripted (non-RL) policy through the env and report it."""
    obs, _ = env.reset()
    done = False
    step = 0
    while not done:
        obs, _r, term, trunc, _info = env.step(action_fn(step))
        done = term or trunc
        step += 1
    hist = env.get_episode_history()
    return compute_report(hist["equity_curve"], hist["trades"], hist["initial_balance"]).to_dict()


def env_buy_and_hold(env: GoldTradingEnv) -> Dict:
    """Open one long at the start (env sizing/costs) and hold to the end."""
    return _scripted_episode(env, lambda step: ACTION_BUY if step == 0 else ACTION_HOLD)


def env_always_flat(env: GoldTradingEnv) -> Dict:
    """Never trade — the zero-skill, zero-cost floor."""
    return _scripted_episode(env, lambda step: ACTION_HOLD)


def env_random(env: GoldTradingEnv, seed: int = 0) -> Dict:
    """Cost-aware random policy — exposes the transaction-cost drag."""
    rng = np.random.default_rng(seed)
    return _scripted_episode(env, lambda step: int(rng.integers(0, 4)))
