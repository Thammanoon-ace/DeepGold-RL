"""
High-throughput vectorized trading environment (V3.5 / Phase 5E).

``VectorizedGoldTradingEnv`` simulates ``num_envs`` independent trading lanes in
a single process using pure NumPy array operations — no per-env Python loop and
no inter-process communication. It implements the Stable-Baselines3 ``VecEnv``
interface, so PPO/DQN can use it as a drop-in replacement for
``DummyVecEnv([... GoldTradingEnv ...])``, but steps all lanes in one vectorized
pass.

Why: the scalar :class:`GoldTradingEnv` stepped one bar at a time in Python;
``DummyVecEnv`` ran lanes sequentially (one CPU core) and ``SubprocVecEnv``
exhausted memory (a torch interpreter per worker). The variance-reduction work
(many seeds × folds × long training) is infeasible at that throughput. This env
removes the per-step Python overhead so training is markedly faster and the GPU
is fed larger batches.

Correctness: the per-lane logic is identical to :class:`GoldTradingEnv`
(fills, risk-based sizing, margin cap, SL/TP with stop-loss-first tie-break,
commission, reward shaping, max-drawdown termination, episode-end force-close).
``scripts/_test_vec_equiv.py`` asserts step-by-step equivalence to the scalar
env. This env is for TRAINING only; deterministic evaluation/backtests keep
using the scalar env + ``run_episode`` (single path, full trade records).
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Type

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from config.config import EnvConfig
from env.gold_trading_env import N_ACCOUNT_FEATURES


class VectorizedGoldTradingEnv(VecEnv):
    """NumPy-batched VecEnv equivalent of :class:`GoldTradingEnv`."""

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,            # (M, 3) columns = high, low, close
        config: EnvConfig,
        num_envs: int = 8,
        random_start: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        prices = np.asarray(prices, dtype=np.float64)
        self._high = prices[:, 0]
        self._low = prices[:, 1]
        self._close = prices[:, 2]
        from env.gold_trading_env import compute_atr_array
        self._atr = compute_atr_array(self._high, self._low, self._close, config.atr_period)
        self.cfg = config
        self.random_start = random_start
        self.window = config.window_size
        self.n_features = self.features.shape[1]
        self.n_bars = len(self.features)
        self._rng = np.random.default_rng(seed)

        self._start_step = self.window - 1
        self._end_step = self.n_bars - 2
        obs_dim = self.window * self.n_features + N_ACCOUNT_FEATURES
        observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Discrete(4)
        super().__init__(num_envs, observation_space, action_space)

        # Per-lane state vectors (shape (num_envs,)).
        n = num_envs
        self.ptr = np.zeros(n, dtype=np.int64)
        self.balance = np.full(n, config.initial_balance, dtype=np.float64)
        self.equity = np.full(n, config.initial_balance, dtype=np.float64)
        self.peak = np.full(n, config.initial_balance, dtype=np.float64)
        self.bh_units = np.zeros(n, dtype=np.float64)   # buy-hold ref (excess reward)
        self._dsr_a = np.zeros(n, dtype=np.float64)     # Differential Sharpe EMA state
        self._dsr_b = np.zeros(n, dtype=np.float64)
        self.pos = np.zeros(n, dtype=np.int64)          # -1/0/+1
        self.lots = np.zeros(n, dtype=np.float64)
        self.entry = np.zeros(n, dtype=np.float64)
        self.sl = np.zeros(n, dtype=np.float64)
        self.tp = np.zeros(n, dtype=np.float64)
        self.entry_step = np.zeros(n, dtype=np.int64)
        self.last_trade_step = np.full(n, -10**9, dtype=np.int64)
        self.n_trades = np.zeros(n, dtype=np.int64)

        self._actions: Optional[np.ndarray] = None
        self._adverse = config.spread / 2.0 + config.slippage
        self._floor = config.initial_balance * (1.0 - config.max_drawdown_pct)

    # ------------------------------------------------------------------ #
    # Lane (re)initialization
    # ------------------------------------------------------------------ #
    def _reset_lanes(self, mask: np.ndarray) -> None:
        k = int(mask.sum())
        if k == 0:
            return
        cfg = self.cfg
        if self.random_start and self._end_step - self._start_step > 256:
            high = self._end_step - 128
            self.ptr[mask] = self._rng.integers(self._start_step, high, size=k)
        else:
            self.ptr[mask] = self._start_step
        self.balance[mask] = cfg.initial_balance
        self.equity[mask] = cfg.initial_balance
        self.peak[mask] = cfg.initial_balance
        self.bh_units[mask] = cfg.initial_balance / self._close[self.ptr[mask]]
        self._dsr_a[mask] = 0.0
        self._dsr_b[mask] = 0.0
        self.pos[mask] = 0
        self.lots[mask] = 0.0
        self.entry[mask] = 0.0
        self.sl[mask] = 0.0
        self.tp[mask] = 0.0
        self.entry_step[mask] = 0
        self.last_trade_step[mask] = -10**9
        self.n_trades[mask] = 0

    # ------------------------------------------------------------------ #
    # Vectorized helpers (mirror GoldTradingEnv exactly)
    # ------------------------------------------------------------------ #
    def _lot_size(self, price: np.ndarray, balance: np.ndarray,
                  stop_distance: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        with np.errstate(divide="ignore", invalid="ignore"):
            lots = (balance * cfg.risk_fraction) / (stop_distance * cfg.contract_size)
        lots = np.minimum(lots, cfg.max_position_lots)
        margin_per_lot = (price * cfg.contract_size) / cfg.leverage
        max_affordable = np.where(margin_per_lot > 0, balance / margin_per_lot, lots)
        lots = np.minimum(lots, max_affordable)
        lots = np.floor(lots / 0.01) * 0.01
        return np.maximum(lots, 0.0)

    def _unrealized(self, price: np.ndarray) -> np.ndarray:
        return (price - self.entry) * self.pos * self.lots * self.cfg.contract_size

    def _build_obs(self) -> np.ndarray:
        # Gather the (num_envs, window, n_features) feature windows.
        offsets = np.arange(-self.window + 1, 1)
        idx = self.ptr[:, None] + offsets[None, :]          # (N, W)
        window = self.features[idx]                          # (N, W, F)
        flat = window.reshape(self.num_envs, -1)

        price = self._close[self.ptr]
        init = self.cfg.initial_balance
        unrealized = self._unrealized(price)
        bars_in_trade = np.where(self.pos != 0, self.ptr - self.entry_step, 0)
        account = np.stack(
            [
                self.pos.astype(np.float32),
                np.clip(unrealized / init, -1.0, 1.0),
                self.lots / max(self.cfg.max_position_lots, 1e-9),
                np.clip(self.equity / init - 1.0, -1.0, 1.0),
                np.clip(bars_in_trade / 100.0, 0.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32)
        obs = np.concatenate([flat, account], axis=1).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ------------------------------------------------------------------ #
    # VecEnv API
    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        self._reset_lanes(np.ones(self.num_envs, dtype=bool))
        return self._build_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(-1).astype(np.int64)

    def step_wait(self):
        cfg = self.cfg
        a = self._actions
        price = self._close[self.ptr]
        equity_before = self.equity.copy()
        opened = np.zeros(self.num_envs, dtype=bool)

        # ---- 1. Apply actions at current bar close --------------------- #
        flat = self.pos == 0
        cooldown_ok = (self.ptr - self.last_trade_step) >= cfg.min_bars_between_trades
        # Stop distance: ATR-based (vol-targeted) or % of price (default).
        if cfg.volatility_target_sizing:
            stop_dist = cfg.vol_target_risk_atr * self._atr[self.ptr]
        else:
            stop_dist = price * cfg.stop_loss_pct
        lots_all = self._lot_size(price, self.balance, stop_dist)

        open_dir = np.where(a == 1, 1, np.where(a == 2, -1, 0))
        open_mask = ((a == 1) | (a == 2)) & flat & cooldown_ok & (lots_all > 0)
        if cfg.max_trades_per_episode:  # hard cap on entries per episode (5C)
            open_mask &= self.n_trades < cfg.max_trades_per_episode
        if open_mask.any():
            d = open_dir[open_mask]
            fill = price[open_mask] + d * self._adverse
            lt = lots_all[open_mask]
            self.balance[open_mask] -= 0.5 * cfg.commission_per_lot * lt
            self.pos[open_mask] = d
            self.lots[open_mask] = lt
            self.entry[open_mask] = fill
            if cfg.volatility_target_sizing:
                sd = (cfg.vol_target_risk_atr * self._atr[self.ptr])[open_mask]
                rr = cfg.take_profit_pct / cfg.stop_loss_pct if cfg.stop_loss_pct else 2.0
                self.sl[open_mask] = fill - d * sd
                self.tp[open_mask] = fill + d * rr * sd
            else:
                self.sl[open_mask] = fill * (1.0 - d * cfg.stop_loss_pct)
                self.tp[open_mask] = fill * (1.0 + d * cfg.take_profit_pct)
            self.entry_step[open_mask] = self.ptr[open_mask]
            self.last_trade_step[open_mask] = self.ptr[open_mask]
            self.n_trades[open_mask] += 1
            opened[open_mask] = True

        close_mask = ((a == 3) & (self.pos != 0)
                      & ((self.ptr - self.entry_step) >= cfg.min_hold_bars))  # 5C min hold
        if close_mask.any():
            self._realize(close_mask, price[close_mask] - self.pos[close_mask] * self._adverse)

        # ---- 2. Advance one bar; resolve SL/TP ------------------------- #
        self.ptr += 1
        hi = self._high[self.ptr]
        lo = self._low[self.ptr]
        is_open = self.pos != 0
        long = self.pos > 0
        hit_sl = np.where(long, lo <= self.sl, hi >= self.sl) & is_open
        hit_tp = np.where(long, hi >= self.tp, lo <= self.tp) & is_open
        sl_mask = hit_sl
        tp_mask = hit_tp & ~hit_sl
        if sl_mask.any():
            self._realize(sl_mask, self.sl[sl_mask] - self.pos[sl_mask] * self._adverse)
        if tp_mask.any():
            self._realize(tp_mask, self.tp[tp_mask] - self.pos[tp_mask] * self._adverse)

        # ---- 3. Mark to market ----------------------------------------- #
        next_close = self._close[self.ptr]
        self.equity = self.balance + self._unrealized(next_close)
        self.peak = np.maximum(self.peak, self.equity)

        # ---- 4. Reward ------------------------------------------------- #
        # 'excess'/'dsr' modes subtract a buy-and-hold benchmark's change.
        if cfg.reward_mode in ("excess", "dsr"):
            bh_change = self.bh_units * (next_close - price)
        else:
            bh_change = 0.0

        if cfg.reward_mode == "dsr":
            # Vectorized Differential Sharpe Ratio of the excess return per lane.
            r = (self.equity - equity_before - bh_change) / max(cfg.initial_balance, 1e-9)
            d_a, d_b = r - self._dsr_a, r * r - self._dsr_b
            denom = self._dsr_b - self._dsr_a ** 2
            safe = np.where(denom > 1e-10, denom, 1.0)   # avoid 0**1.5 divide warning
            dsr = np.where(denom > 1e-10,
                           (self._dsr_b * d_a - 0.5 * self._dsr_a * d_b) / np.power(safe, 1.5),
                           0.0)
            self._dsr_a = self._dsr_a + cfg.dsr_eta * d_a
            self._dsr_b = self._dsr_b + cfg.dsr_eta * d_b
            reward = np.clip(dsr, -10.0, 10.0)
        else:
            net_profit = (self.equity - equity_before - bh_change) * cfg.reward_scaling
            drawdown = (self.peak - self.equity) / np.maximum(self.peak, 1e-9)
            overtrade_pen = cfg.overtrading_penalty * (1.0 + cfg.trade_penalty_growth * self.n_trades)
            reward = (net_profit
                      - cfg.drawdown_penalty_weight * drawdown
                      - np.where(opened, overtrade_pen, 0.0)
                      - np.where(self.pos != 0, cfg.holding_penalty, 0.0))
            reward = np.clip(reward, -10.0, 10.0)

        # ---- 5. Termination / truncation + episode-end force close ----- #
        terminated = self.equity <= self._floor
        truncated = self.ptr >= self._end_step
        dones = terminated | truncated
        force = dones & (self.pos != 0)
        if force.any():
            self._realize(force, next_close[force] - self.pos[force] * self._adverse)
            self.equity[force] = self.balance[force]

        # ---- 6. Observation + auto-reset of done lanes ----------------- #
        obs = self._build_obs()
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        if dones.any():
            done_idx = np.nonzero(dones)[0]
            terminal_obs = obs[done_idx].copy()
            for j, i in enumerate(done_idx):
                infos[i]["terminal_observation"] = terminal_obs[j]
                infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
                infos[i]["equity"] = float(self.equity[i])
                infos[i]["n_trades"] = int(self.n_trades[i])
            self._reset_lanes(dones)
            obs = self._build_obs()  # done lanes now show their fresh start

        return obs, reward.astype(np.float32), dones, infos

    def _realize(self, mask: np.ndarray, exit_fill: np.ndarray) -> None:
        """Close positions in ``mask`` at ``exit_fill`` (vectorized)."""
        cfg = self.cfg
        gross = (exit_fill - self.entry[mask]) * self.pos[mask] * self.lots[mask] * cfg.contract_size
        net = gross - 0.5 * cfg.commission_per_lot * self.lots[mask]
        self.balance[mask] += net
        self.pos[mask] = 0
        self.lots[mask] = 0.0

    # ------------------------------------------------------------------ #
    # VecEnv abstract methods (minimal valid implementations)
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        idx = self._idx(indices)
        return [getattr(self, attr_name, None) for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        idx = self._idx(indices)
        return [None for _ in idx]

    def env_is_wrapped(self, wrapper_class: Type, indices=None) -> List[bool]:
        return [False for _ in self._idx(indices)]

    def _idx(self, indices) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        return [seed] * self.num_envs
