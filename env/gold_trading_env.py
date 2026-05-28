"""
Custom Gymnasium trading environment for XAUUSD (requirements #3/#4/#13).

``GoldTradingEnv`` simulates a single-position spot-gold trading account with
realistic frictions — spread, slippage, commission, leverage/margin — and hard
risk controls (max position size, max drawdown, anti-overtrading).  The reward
is an equity-change signal shaped to discourage the usual reward-hacking
failure modes (see :meth:`_compute_reward`).

Causality / no look-ahead (requirement #14)
-------------------------------------------
At step ``i`` the agent observes the feature window ending at bar ``i`` and acts
at ``close[i]``.  The environment then advances to bar ``i+1`` and the PnL /
SL / TP outcome is evaluated against bar ``i+1`` — data the agent could not see
when it decided.  No future information ever enters the observation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config.config import EnvConfig

logger = logging.getLogger(__name__)


# Discrete action identifiers.
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_CLOSE = 3
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}

# Number of account-state scalars appended to the flattened feature window in
# every observation. Sequence feature extractors (policies/) use this to split
# the flat observation back into (window x features) + account state.
N_ACCOUNT_FEATURES = 5


def compute_atr_array(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """Causal Wilder ATR as a NumPy array aligned to the bars.

    Uses only current/past bars (true range + EWM smoothing), so it is safe to
    use for volatility-targeted position sizing without look-ahead.
    """
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = np.empty_like(tr)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):  # Wilder EWM (one pass, done once at construction)
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


@dataclass
class Position:
    """State of the (single) open position. ``direction`` 0 means flat."""

    direction: int = 0          # +1 long, -1 short, 0 flat
    lots: float = 0.0
    entry_price: float = 0.0    # cost-adjusted fill price
    entry_step: int = 0
    sl_price: float = 0.0
    tp_price: float = 0.0
    entry_time: Any = None

    @property
    def is_open(self) -> bool:
        return self.direction != 0


@dataclass
class TradeRecord:
    """Closed-trade record used by the backtester and plots."""

    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    direction: int
    lots: float
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str


class GoldTradingEnv(gym.Env):
    """A Gymnasium environment for reinforcement-learning gold trading.

    Parameters
    ----------
    features:
        2-D array ``(n_bars, n_features)`` of **already-normalized** features.
    prices:
        DataFrame aligned 1:1 with ``features`` containing raw (un-normalized)
        ``open/high/low/close`` columns, indexed by timestamp.  Raw prices are
        required for honest PnL, spread and SL/TP accounting.
    config:
        An :class:`~config.config.EnvConfig`.
    random_start:
        If True, each episode begins at a random offset (useful for training so
        the agent does not always see the same opening bars).  Set False for
        deterministic backtesting.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: pd.DataFrame,
        config: EnvConfig,
        random_start: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if len(features) != len(prices):
            raise ValueError("features and prices must have equal length.")
        for col in ("open", "high", "low", "close"):
            if col not in prices.columns:
                raise ValueError(f"prices is missing required column {col!r}.")

        self.features = np.asarray(features, dtype=np.float32)
        self.prices = prices.reset_index(drop=False)
        self._time_col = prices.index.name or "index"
        self._high = self.prices["high"].to_numpy(dtype=np.float64)
        self._low = self.prices["low"].to_numpy(dtype=np.float64)
        self._close = self.prices["close"].to_numpy(dtype=np.float64)
        self.times = self.prices[self._time_col].to_numpy()
        # Causal ATR for volatility-targeted sizing (computed once from own OHLC).
        self._atr = compute_atr_array(self._high, self._low, self._close, config.atr_period)

        self.config = config
        self.random_start = random_start
        self.render_mode = render_mode

        self.window_size = config.window_size
        self.n_features = self.features.shape[1]
        self.n_bars = len(self.features)
        if self.n_bars <= self.window_size + 2:
            raise ValueError(
                f"Not enough bars ({self.n_bars}) for window_size "
                f"{self.window_size}."
            )

        # ---- Spaces ---------------------------------------------------- #
        # Discrete: Hold / Buy / Sell / Close.
        self.action_space = spaces.Discrete(4)
        # Observation: flattened feature window + account-state features.
        self._n_account_features = N_ACCOUNT_FEATURES
        obs_dim = self.window_size * self.n_features + self._n_account_features
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # ---- Episode state (initialized in reset) ---------------------- #
        self.current_step: int = 0
        self.balance: float = config.initial_balance
        self.equity: float = config.initial_balance
        self.peak_equity: float = config.initial_balance
        self.position = Position()
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        self.time_history: List[Any] = []
        self._last_trade_step: int = -10**9
        self._n_trades: int = 0

    # ================================================================== #
    # Gymnasium API
    # ================================================================== #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self.config

        # First valid step needs a full lookback window behind it; we stop one
        # bar before the end so we can always evaluate the *next* bar.
        self._start_step = self.window_size - 1
        self._end_step = self.n_bars - 2
        if self.random_start and self._end_step - self._start_step > 256:
            # Random offset, leaving room for a meaningful episode.
            high = self._end_step - 128
            self.current_step = int(
                self.np_random.integers(self._start_step, high)
            )
        else:
            self.current_step = self._start_step

        self.balance = cfg.initial_balance
        self.equity = cfg.initial_balance
        self.peak_equity = cfg.initial_balance
        # Buy-and-hold reference fixed at the episode's start price (for the
        # 'excess' reward mode): units of gold worth `initial_balance` at start.
        self._bh_units = cfg.initial_balance / float(self._close[self.current_step])
        self._dsr_a = 0.0   # Differential Sharpe Ratio EMA state (reward_mode='dsr')
        self._dsr_b = 0.0
        self.position = Position()
        self.trades = []
        self.equity_curve = [self.equity]
        self.reward_history = []
        self.action_history = []
        self.time_history = [self.times[self.current_step]]
        self._last_trade_step = -10**9
        self._n_trades = 0

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        cfg = self.config
        price = float(self._close[self.current_step])
        equity_before = self.equity

        opened_trade = False
        # ---- 1. Apply the agent's action at the current bar's close ---- #
        if action == ACTION_BUY and not self.position.is_open:
            opened_trade = self._open_position(+1, price)
        elif action == ACTION_SELL and not self.position.is_open:
            opened_trade = self._open_position(-1, price)
        elif (action == ACTION_CLOSE and self.position.is_open
              and (self.current_step - self.position.entry_step) >= cfg.min_hold_bars):
            # Signal-close only after the minimum holding time (SL/TP still fire).
            self._close_position(self._fill_price(-self.position.direction, price),
                                  reason="signal")
        # Buy/Sell while already in a position, or Close while flat, are no-ops
        # (single-position model): this naturally limits churn.

        # ---- 2. Advance one bar and resolve SL/TP on the new bar ------- #
        self.current_step += 1
        next_high = float(self._high[self.current_step])
        next_low = float(self._low[self.current_step])
        next_close = float(self._close[self.current_step])
        if self.position.is_open:
            self._check_stop_levels(next_high, next_low)

        # ---- 3. Mark to market & update equity ------------------------- #
        self.equity = self.balance + self._unrealized_pnl(next_close)
        self.peak_equity = max(self.peak_equity, self.equity)

        # ---- 4. Reward ------------------------------------------------- #
        # Buy-and-hold benchmark change over this step (excess / dsr modes).
        bh_change = (self._bh_units * (next_close - price)
                     if cfg.reward_mode in ("excess", "dsr") else 0.0)
        reward = self._compute_reward(equity_before, opened_trade, bh_change)

        # ---- 5. Termination / truncation ------------------------------- #
        terminated = self._is_bankrupt()
        truncated = self.current_step >= self._end_step
        if (terminated or truncated) and self.position.is_open:
            # Force-close any open position at the final bar for clean accounting.
            self._close_position(
                self._fill_price(-self.position.direction, next_close),
                reason="episode_end",
            )
            self.equity = self.balance

        # ---- 6. Bookkeeping -------------------------------------------- #
        self.equity_curve.append(self.equity)
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.time_history.append(self.times[self.current_step])

        if self.render_mode == "human":
            self.render()

        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._get_info(),
        )

    # ================================================================== #
    # Trading mechanics
    # ================================================================== #
    def _fill_price(self, direction: int, mid_price: float) -> float:
        """Apply spread + slippage to obtain a realistic fill.

        A buy (direction +1) pays the ask = mid + half-spread + slippage.
        A sell (direction -1) receives the bid = mid - half-spread - slippage.
        """
        cfg = self.config
        half_spread = cfg.spread / 2.0
        adverse = half_spread + cfg.slippage
        return mid_price + direction * adverse

    def _stop_distance(self, mid_price: float) -> float:
        """Stop-loss distance in price units: ATR-based (vol-targeted) or % of price."""
        cfg = self.config
        if cfg.volatility_target_sizing:
            return cfg.vol_target_risk_atr * float(self._atr[self.current_step])
        return mid_price * cfg.stop_loss_pct

    def _compute_lot_size(self, mid_price: float) -> float:
        """Risk-based position sizing, capped for safety (requirement #13).

        Lots are chosen so that hitting the stop loss costs roughly
        ``risk_fraction`` of the current balance, then clamped to
        ``max_position_lots`` and to whatever free margin allows.  This prevents
        the agent from over-leveraging.
        """
        cfg = self.config
        stop_distance = self._stop_distance(mid_price)   # price units to the stop
        if stop_distance <= 0:
            return 0.0
        risk_capital = self.balance * cfg.risk_fraction
        lots = risk_capital / (stop_distance * cfg.contract_size)

        # Hard cap on absolute size.
        lots = min(lots, cfg.max_position_lots)

        # Margin cap: required margin must not exceed available balance.
        margin_per_lot = (mid_price * cfg.contract_size) / cfg.leverage
        if margin_per_lot > 0:
            max_affordable = self.balance / margin_per_lot
            lots = min(lots, max_affordable)

        # Round to a broker-like minimum lot step and floor tiny sizes to 0.
        lots = float(np.floor(lots / 0.01) * 0.01)
        return max(lots, 0.0)

    def _open_position(self, direction: int, mid_price: float) -> bool:
        """Open a new position. Returns True if a trade was actually opened."""
        cfg = self.config
        # Anti-overtrading: enforce a cooldown between entries.
        if self.current_step - self._last_trade_step < cfg.min_bars_between_trades:
            return False
        # Hard cap on number of entries per episode (0 = unlimited).
        if cfg.max_trades_per_episode and self._n_trades >= cfg.max_trades_per_episode:
            return False

        lots = self._compute_lot_size(mid_price)
        if lots <= 0:
            return False

        fill = self._fill_price(direction, mid_price)
        # Half the round-turn commission is charged on entry.
        commission = 0.5 * cfg.commission_per_lot * lots
        self.balance -= commission

        if cfg.volatility_target_sizing:
            # ATR-based SL/TP, same reward:risk ratio as the % defaults.
            sd = cfg.vol_target_risk_atr * float(self._atr[self.current_step])
            rr = cfg.take_profit_pct / cfg.stop_loss_pct if cfg.stop_loss_pct else 2.0
            sl = fill - direction * sd
            tp = fill + direction * rr * sd
        elif direction > 0:  # long: SL below, TP above (% of price)
            sl = fill * (1.0 - cfg.stop_loss_pct)
            tp = fill * (1.0 + cfg.take_profit_pct)
        else:              # short: SL above, TP below
            sl = fill * (1.0 + cfg.stop_loss_pct)
            tp = fill * (1.0 - cfg.take_profit_pct)

        self.position = Position(
            direction=direction,
            lots=lots,
            entry_price=fill,
            entry_step=self.current_step,
            sl_price=sl,
            tp_price=tp,
            entry_time=self.times[self.current_step],
        )
        self._last_trade_step = self.current_step
        self._n_trades += 1
        return True

    def _unrealized_pnl(self, mid_price: float) -> float:
        """Mark-to-market PnL of the open position at ``mid_price``."""
        pos = self.position
        if not pos.is_open:
            return 0.0
        return (mid_price - pos.entry_price) * pos.direction * pos.lots * self.config.contract_size

    def _close_position(self, exit_fill: float, reason: str) -> None:
        """Realize PnL, charge exit commission and flatten the position."""
        cfg = self.config
        pos = self.position
        if not pos.is_open:
            return

        gross = (exit_fill - pos.entry_price) * pos.direction * pos.lots * cfg.contract_size
        commission = 0.5 * cfg.commission_per_lot * pos.lots
        net = gross - commission
        self.balance += net

        notional = pos.entry_price * pos.lots * cfg.contract_size
        self.trades.append(
            TradeRecord(
                entry_time=pos.entry_time,
                exit_time=self.times[self.current_step],
                entry_price=pos.entry_price,
                exit_price=exit_fill,
                direction=pos.direction,
                lots=pos.lots,
                pnl=net,
                return_pct=(net / notional) if notional else 0.0,
                bars_held=self.current_step - pos.entry_step,
                exit_reason=reason,
            )
        )
        self.position = Position()  # flat

    def _check_stop_levels(self, bar_high: float, bar_low: float) -> None:
        """Resolve stop-loss / take-profit against the current bar's range.

        If both levels fall inside the bar we conservatively assume the stop
        loss triggered first (worst case for the trader).
        """
        pos = self.position
        if not pos.is_open:
            return

        if pos.direction > 0:  # long
            hit_sl = bar_low <= pos.sl_price
            hit_tp = bar_high >= pos.tp_price
        else:                  # short
            hit_sl = bar_high >= pos.sl_price
            hit_tp = bar_low <= pos.tp_price

        if hit_sl:
            # Exit at the stop with extra slippage in the adverse direction.
            exit_fill = self._fill_price(-pos.direction, pos.sl_price)
            self._close_position(exit_fill, reason="stop_loss")
        elif hit_tp:
            exit_fill = self._fill_price(-pos.direction, pos.tp_price)
            self._close_position(exit_fill, reason="take_profit")

    def _is_bankrupt(self) -> bool:
        """Episode-ending risk guard: equity below the max-drawdown floor."""
        floor = self.config.initial_balance * (1.0 - self.config.max_drawdown_pct)
        return self.equity <= floor

    # ================================================================== #
    # Reward (requirement #4) — designed to resist reward hacking (#14)
    # ================================================================== #
    def _compute_reward(self, equity_before: float, opened_trade: bool,
                        bh_change: float = 0.0) -> float:
        """Compute the shaped reward for the step.

        ``reward = net_profit - drawdown_penalty - overtrading_penalty - holding_penalty``

        * ``net_profit`` is the *change in equity*, scaled.  Because equity is
          marked to market, unrealized losses hurt immediately — the agent
          cannot dodge a penalty by refusing to realize a losing trade.  All
          transaction costs (spread, slippage, commission) are already baked
          into the equity change, so they are not double-counted here.
        * ``drawdown_penalty`` punishes new equity lows, discouraging the
          high-variance "all-in" behaviour that maximises raw PnL on a lucky
          backtest.
        * ``overtrading_penalty`` charges a small fixed cost per executed
          entry, so the agent only trades when it expects edge beyond costs.
        * ``holding_penalty`` (optional, default 0) gently discourages parking
          capital in idle positions.
        """
        cfg = self.config

        # 'dsr' mode: online Differential Sharpe Ratio of the excess (vs
        # buy-and-hold) return — maximises the information ratio. Self-contained
        # (encodes risk-adjustment), so no extra shaping penalties are added.
        if cfg.reward_mode == "dsr":
            r = (self.equity - equity_before - bh_change) / max(cfg.initial_balance, 1e-9)
            a, b = self._dsr_a, self._dsr_b
            d_a, d_b = r - a, r * r - b
            denom = b - a * a
            dsr = (b * d_a - 0.5 * a * d_b) / (denom ** 1.5) if denom > 1e-10 else 0.0
            self._dsr_a = a + cfg.dsr_eta * d_a
            self._dsr_b = b + cfg.dsr_eta * d_b
            return float(np.clip(dsr, -10.0, 10.0))

        # 'excess' mode subtracts the buy-and-hold benchmark change, so reward
        # accrues only for beating passive holding (bh_change is 0 in 'absolute').
        net_profit = (self.equity - equity_before - bh_change) * cfg.reward_scaling

        # Drawdown penalty: proportional to how far below the running peak we
        # are, expressed as a fraction of the peak.
        drawdown = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9)
        drawdown_penalty = cfg.drawdown_penalty_weight * drawdown

        # Overtrading penalty optionally grows with the episode's trade count.
        overtrading_penalty = (
            cfg.overtrading_penalty * (1.0 + cfg.trade_penalty_growth * self._n_trades)
            if opened_trade else 0.0
        )
        holding_penalty = cfg.holding_penalty if self.position.is_open else 0.0

        reward = net_profit - drawdown_penalty - overtrading_penalty - holding_penalty
        # Keep the signal bounded for stable PPO updates.
        return float(np.clip(reward, -10.0, 10.0))

    # ================================================================== #
    # Observation & info
    # ================================================================== #
    def _get_observation(self) -> np.ndarray:
        """Assemble the flattened feature window + normalized account state."""
        i = self.current_step
        window = self.features[i - self.window_size + 1 : i + 1]
        flat = window.flatten()

        price = float(self._close[i])
        unrealized = self._unrealized_pnl(price)
        init_bal = self.config.initial_balance
        bars_in_trade = (
            (i - self.position.entry_step) if self.position.is_open else 0
        )

        account_state = np.array(
            [
                float(self.position.direction),                      # -1/0/+1
                np.clip(unrealized / init_bal, -1.0, 1.0),           # PnL ratio
                self.position.lots / max(self.config.max_position_lots, 1e-9),
                np.clip(self.equity / init_bal - 1.0, -1.0, 1.0),    # equity ret
                np.clip(bars_in_trade / 100.0, 0.0, 1.0),            # holding age
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([flat, account_state]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _get_info(self) -> Dict[str, Any]:
        """Diagnostic dictionary returned alongside every step."""
        return {
            "step": self.current_step,
            "time": self.times[self.current_step],
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position.direction,
            "position_lots": self.position.lots,
            "n_trades": self._n_trades,
            "drawdown": (self.peak_equity - self.equity) / max(self.peak_equity, 1e-9),
        }

    # ================================================================== #
    # Reporting helpers (used by the backtester / notebooks)
    # ================================================================== #
    def get_episode_history(self) -> Dict[str, Any]:
        """Return equity curve, trades, rewards and timestamps for analysis."""
        return {
            "equity_curve": np.asarray(self.equity_curve, dtype=float),
            "timestamps": list(self.time_history),
            "rewards": np.asarray(self.reward_history, dtype=float),
            "actions": list(self.action_history),
            "trades": [t.__dict__ for t in self.trades],
            "final_balance": self.balance,
            "final_equity": self.equity,
            "initial_balance": self.config.initial_balance,
        }

    def render(self) -> None:
        """Minimal text render for ``render_mode='human'``."""
        pos = ACTION_NAMES.get(self.position.direction + 0, "")
        side = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.position.direction]
        print(
            f"step={self.current_step} time={self.times[self.current_step]} "
            f"equity={self.equity:,.2f} balance={self.balance:,.2f} "
            f"pos={side} lots={self.position.lots:.2f} trades={self._n_trades}"
        )

    def close(self) -> None:  # noqa: D401 - Gym API
        """No external resources to release."""
        return None
