"""
V4 — Optimal-execution Gymnasium environment for XAUUSD (sibling of
:class:`env.gold_trading_env.GoldTradingEnv`).

Episode = one execution task: trade ``target_lots`` lots of XAUUSD over
``deadline_bars`` bars, minimizing implementation shortfall vs the **arrival
price** at episode start. A shadow TWAP slicer runs in parallel so each
terminal info dict carries ``bps_savings_vs_twap`` — the headline metric the
seed × fold grid will optimize.

See [docs/V4_OPTIMAL_EXECUTION.md](../docs/V4_OPTIMAL_EXECUTION.md) for the
design rationale (why this pivots from "does RL beat buy-and-hold?" to
"does RL beat uniform slicing?").

Causality
---------
At step ``i`` the agent observes the feature window ending at bar ``i`` and
acts at ``close[i]``. The slice fills at ``close[i] ± (half-spread + slippage
+ impact·slice_lots)``, the env advances to bar ``i+1``, the next observation
is built. No future information enters the agent's view.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config.config import EnvConfig

# Five execution-state scalars appended to the flat feature window. Same slot
# count as V0–V3.5's ``N_ACCOUNT_FEATURES`` so the existing networks (CNN,
# CNN-LSTM, etc.) plug in unchanged — only the *meaning* of those slots changes.
N_EXEC_FEATURES = 5

# Execution-rate multipliers (× TWAP rate this bar): pause / slow / mean / fast
ACTION_RATE_MULT: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
N_ACTIONS = len(ACTION_RATE_MULT)


class ExecutionGoldEnv(gym.Env):
    """Single-asset optimal-execution env over XAUUSD bars.

    Parameters
    ----------
    features:
        ``(n_bars, n_features)`` causal indicator array (same as V0–V3.5).
    prices:
        Either a DataFrame with ``high``/``low``/``close`` columns or an
        ``(n_bars, 3)`` array of those three columns.
    config:
        Shared :class:`EnvConfig` — only ``window_size``, ``spread`` and
        ``slippage`` are used. (Position sizing/SL/TP/drawdown are NOT part of
        the execution task and are ignored.)
    deadline_range / target_lots_range:
        Sampled per episode to give the agent variety of horizons and order
        sizes. Defaults give 32–128 bar deadlines and 0.5–2.0 lot orders.
    fixed_cost_bps:
        Fixed cost per **lot** (not per trade) in basis points. Models the
        bid-ask spread + slippage you pay to cross the market, amortised over
        the order size so it doesn't penalise slicing per-se. Default ``1.0``
        bps = 0.01 % of price per lot. Independent of slicing strategy in
        expectation.
    impact_bps_per_lot:
        Price impact in basis points per lot **scaled by slice size** —
        ``impact_charge_per_lot = impact_bps_per_lot × slice_lots``. This is
        the dominant tension: a single full-lot slice pays this in full;
        spreading the same lot over 64 TWAP slices pays only ~``1/64`` of it.
        Default ``10.0`` bps gives a measurable ~10 bps spread between
        front-load and TWAP, plus drift effects on top.
    random_start / seed:
        Same role as in the directional env. Walk-forward folds set the slice
        of ``features``/``prices`` they pass in; ``random_start`` then samples
        episode starts inside that fold.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices,                        # pd.DataFrame or np.ndarray (n,3)
        config: EnvConfig,
        deadline_range: Tuple[int, int] = (32, 128),
        target_lots_range: Tuple[float, float] = (0.5, 2.0),
        fixed_cost_bps: float = 1.0,
        impact_bps_per_lot: float = 10.0,
        random_start: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.features = np.asarray(features, dtype=np.float32)
        if isinstance(prices, pd.DataFrame):
            self._close = prices["close"].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(prices, dtype=np.float64)
            self._close = arr[:, 2] if arr.ndim == 2 and arr.shape[1] >= 3 else arr.reshape(-1)

        self.window = config.window_size
        self.n_features = self.features.shape[1]
        self.n_bars = self.features.shape[0]

        self.deadline_range = deadline_range
        self.target_lots_range = target_lots_range
        self.fixed_cost_bps = fixed_cost_bps
        self.impact_bps_per_lot = impact_bps_per_lot
        self.random_start = random_start
        self.np_random = np.random.default_rng(seed)
        # Note: V0–V3.5's config.spread / config.slippage are directional-trading
        # frictions; the execution env uses its own bps-scale cost model above so
        # the dollar magnitudes don't depend on the EnvConfig defaults.

        obs_dim = self.window * self.n_features + N_EXEC_FEATURES
        self.observation_space = spaces.Box(-10.0, 10.0, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state (populated in reset).
        self.ptr: int = 0
        self.arrival_price: float = 0.0
        self.side: int = 0
        self.target_lots: float = 0.0
        self.deadline_bars: int = 0
        self.remaining_lots: float = 0.0
        self.remaining_bars: int = 0
        # Agent running totals.
        self.total_filled: float = 0.0
        self.cum_fill_value: float = 0.0
        # TWAP shadow running totals (constant slice each bar of the original deadline).
        self._twap_slice: float = 0.0
        self.twap_filled: float = 0.0
        self.twap_cum_value: float = 0.0

    # ------------------------------------------------------------------ #
    def _fill_price(self, price: float, slice_lots: float) -> float:
        """Adverse fill in bps of price.

        ``effective_bps = fixed_cost_bps + impact_bps_per_lot × slice_lots``,
        applied as ``price × (1 + side × effective_bps/1e4)``. The fixed term
        is amortised per lot (TWAP and front-load pay the same total fixed
        cost ≈ target_lots × fixed_cost_bps); the impact term is convex in
        slice size, so spreading the order reduces it ~linearly.
        """
        effective_bps = self.fixed_cost_bps + self.impact_bps_per_lot * slice_lots
        return price * (1.0 + self.side * effective_bps / 1e4)

    def _force_finish(self) -> None:
        """At terminal, if lots remain (agent OR TWAP shadow), force them in at
        the current bar with full impact — a hard cost the agent should avoid
        by finishing on time."""
        price = float(self._close[min(self.ptr, self.n_bars - 1)])
        if self.remaining_lots > 1e-9:
            f = self.remaining_lots
            self.cum_fill_value += self._fill_price(price, f) * f
            self.total_filled += f
            self.remaining_lots = 0.0
        twap_left = self.target_lots - self.twap_filled
        if twap_left > 1e-9:
            self.twap_cum_value += self._fill_price(price, twap_left) * twap_left
            self.twap_filled += twap_left

    def _obs(self) -> np.ndarray:
        # Causal window ending at the current bar; clipped at the lower edge
        # in case the episode somehow starts before warm-up bars (caller
        # responsibility, but we don't want a crash).
        offsets = np.arange(-self.window + 1, 1)
        idx = np.clip(self.ptr + offsets, 0, self.n_bars - 1)
        win = self.features[idx].reshape(-1)

        eps = 1e-9
        run_avg = (self.cum_fill_value / self.total_filled
                   if self.total_filled > eps else self.arrival_price)
        cur_close = float(self._close[min(self.ptr, self.n_bars - 1)])
        run_dev = (run_avg - self.arrival_price) / max(abs(self.arrival_price), eps)
        cur_dev = (cur_close - self.arrival_price) / max(abs(self.arrival_price), eps)
        exec_scalars = np.array([
            float(self.side),
            float(self.remaining_lots / max(self.target_lots, eps)),
            float(self.remaining_bars / max(self.deadline_bars, 1)),
            # Two raw price-deviation signals — the network can combine with
            # `side` to form a directional view. Scaled ×100 so the natural
            # magnitudes are O(1) (a 1 % price drift becomes 1.0).
            float(np.clip(run_dev * 100.0, -10.0, 10.0)),
            float(np.clip(cur_dev * 100.0, -10.0, 10.0)),
        ], dtype=np.float32)

        obs = np.concatenate([win, exec_scalars]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.side = int(self.np_random.choice([-1, 1]))
        self.target_lots = float(self.np_random.uniform(*self.target_lots_range))
        self.deadline_bars = int(self.np_random.integers(self.deadline_range[0],
                                                        self.deadline_range[1] + 1))

        lo = self.window - 1
        hi = self.n_bars - self.deadline_bars - 2
        if hi <= lo:
            self.ptr = lo
        elif self.random_start:
            self.ptr = int(self.np_random.integers(lo, hi))
        else:
            self.ptr = lo

        self.arrival_price = float(self._close[self.ptr])
        self.remaining_lots = self.target_lots
        self.remaining_bars = self.deadline_bars
        self.total_filled = 0.0
        self.cum_fill_value = 0.0
        self._twap_slice = self.target_lots / max(self.deadline_bars, 1)
        self.twap_filled = 0.0
        self.twap_cum_value = 0.0
        return self._obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        a = int(action)
        if a < 0 or a >= N_ACTIONS:
            raise ValueError(f"action {a} out of range [0, {N_ACTIONS - 1}]")

        # 1. Agent's slice this bar.
        twap_rate = self.remaining_lots / max(self.remaining_bars, 1)
        slice_lots = min(ACTION_RATE_MULT[a] * twap_rate, self.remaining_lots)
        price = float(self._close[self.ptr])
        if slice_lots > 1e-9:
            fill = self._fill_price(price, slice_lots)
            self.cum_fill_value += fill * slice_lots
            self.total_filled += slice_lots
            self.remaining_lots -= slice_lots

        # 2. TWAP shadow slice (constant rate per original deadline bar).
        twap_left = self.target_lots - self.twap_filled
        if twap_left > 1e-9:
            twap_slc = min(self._twap_slice, twap_left)
            self.twap_cum_value += self._fill_price(price, twap_slc) * twap_slc
            self.twap_filled += twap_slc

        # 3. Advance time.
        self.remaining_bars -= 1
        self.ptr = min(self.ptr + 1, self.n_bars - 1)

        terminated = self.remaining_lots <= 1e-9
        truncated = (self.remaining_bars <= 0 and not terminated) \
                    or self.ptr >= self.n_bars - 1

        reward = 0.0
        info: Dict[str, Any] = {}
        if terminated or truncated:
            self._force_finish()
            avg_fill = self.cum_fill_value / max(self.total_filled, 1e-9)
            twap_avg = self.twap_cum_value / max(self.twap_filled, 1e-9)
            arrival = self.arrival_price
            # Shortfall: positive = paid worse than arrival.
            sf_bps = (avg_fill - arrival) / arrival * 1e4 * self.side
            twap_bps = (twap_avg - arrival) / arrival * 1e4 * self.side
            # Reward: positive = bps SAVED vs arrival. Bps_savings_vs_twap is
            # the headline metric (TWAP-relative is the V4 grid's "robustness").
            reward = -float(sf_bps)
            info = {
                "avg_fill": float(avg_fill),
                "twap_avg_fill": float(twap_avg),
                "arrival_price": float(arrival),
                "shortfall_bps": float(sf_bps),
                "twap_shortfall_bps": float(twap_bps),
                "bps_savings_vs_twap": float(twap_bps - sf_bps),
                "side": int(self.side),
                "target_lots": float(self.target_lots),
                "deadline_bars": int(self.deadline_bars),
            }

        return self._obs(), float(reward), bool(terminated), bool(truncated), info
