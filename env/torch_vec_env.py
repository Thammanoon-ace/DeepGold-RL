"""
GPU-resident vectorized trading environment (V3.5+ — high-throughput RL).

``TorchVecGoldEnv`` simulates ``num_envs`` lanes entirely with **torch tensors on
a chosen device (CUDA)**, so the whole RL loop — env stepping, policy inference,
PPO update — runs on the GPU with no per-step CPU<->GPU transfer. With thousands
of lanes the GPU becomes the primary compute engine (the Isaac-Gym / Brax style),
unlike the numpy ``VectorizedGoldTradingEnv`` whose stepping is CPU-bound.

Scope: it implements the **default** trading path (risk-based %-of-price sizing,
%-based SL/TP, single position, spread/slippage/commission, max-drawdown
termination, episode-end force-close) and reward modes ``absolute`` / ``excess``
/ ``dsr`` — i.e. everything the main V3.5 experiments use. The optional
vol-targeted sizing and 5C trade-frequency knobs are intentionally omitted here
(they are off in the experiments); use the scalar/numpy env for those.

Correctness: ``scripts/_test_torch_equiv.py`` checks it matches the numpy env
(run in float64 on CPU). Training uses float32 on CUDA for speed; deterministic
backtests still use the scalar env.

This env is intended for a CleanRL-style PPO loop (``training/cleanrl_ppo.py``),
not SB3 — it returns torch tensors, not numpy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from config.config import EnvConfig
from env.gold_trading_env import N_ACCOUNT_FEATURES, compute_atr_array


class TorchVecGoldEnv:
    """Batched, GPU-resident gold-trading env (torch tensors).

    Eval-mode extensions (Tier 2.1, 2026-05-28):

    * ``auto_reset=False`` — for deterministic single-path evaluation, suppress
      the automatic ``_reset_mask(done)`` in ``step()`` so the episode terminates
      and the caller can read out the trajectory.
    * ``track_history=True`` (requires ``num_envs == 1``) — record per-step
      equity / reward / action and per-trade entry/exit/PnL on the host, so
      ``get_episode_history()`` returns a dict compatible with the scalar env's
      ``GoldTradingEnv.get_episode_history()``. Lets the backtester run a full
      OOS pass on the GPU with no per-step CPU↔GPU round trip — replacing the
      ensemble-eval bottleneck (~16 min/fold scalar → ~3 min/fold target).
    * ``times`` — optional sequence of timestamps aligned to ``prices`` rows;
      used to populate ``time_history`` and ``trades[].entry_time/exit_time``.
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,            # (M, 3) = high, low, close
        config: EnvConfig,
        num_envs: int = 1024,
        random_start: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        *,
        auto_reset: bool = True,
        track_history: bool = False,
        times: Optional[Sequence[Any]] = None,
    ) -> None:
        self.cfg = config
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.dtype = dtype
        self.num_envs = num_envs
        self.random_start = random_start
        self.window = config.window_size
        self.n_features = int(features.shape[1])
        self.n_bars = int(features.shape[0])
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

        prices = np.asarray(prices, dtype=np.float64)
        atr = compute_atr_array(prices[:, 0], prices[:, 1], prices[:, 2], config.atr_period)
        dev, dt = self.device, dtype
        self.features = torch.tensor(np.asarray(features), dtype=dt, device=dev)
        self._high = torch.tensor(prices[:, 0], dtype=dt, device=dev)
        self._low = torch.tensor(prices[:, 1], dtype=dt, device=dev)
        self._close = torch.tensor(prices[:, 2], dtype=dt, device=dev)
        self._atr = torch.tensor(atr, dtype=dt, device=dev)

        self.obs_dim = self.window * self.n_features + N_ACCOUNT_FEATURES
        self._start = self.window - 1
        self._end = self.n_bars - 2
        self._adverse = config.spread / 2.0 + config.slippage
        self._floor = config.initial_balance * (1.0 - config.max_drawdown_pct)
        self._offsets = torch.arange(-self.window + 1, 1, device=dev)

        z = lambda: torch.zeros(num_envs, dtype=dt, device=dev)         # noqa: E731
        zl = lambda: torch.zeros(num_envs, dtype=torch.long, device=dev)  # noqa: E731
        self.ptr, self.pos = zl(), zl()
        self.entry_step, self.last_trade_step, self.n_trades = zl(), zl(), zl()
        self.balance, self.equity, self.peak = z(), z(), z()
        self.lots, self.entry, self.sl, self.tp = z(), z(), z(), z()
        self.bh_units, self._dsr_a, self._dsr_b = z(), z(), z()
        self.last_ep_returns = torch.zeros(0, device=dev)  # for logging

        # ---- Eval-mode tracking (Tier 2.1) ------------------------------ #
        self.auto_reset = bool(auto_reset)
        self.track_history = bool(track_history)
        if self.track_history and num_envs != 1:
            raise ValueError("track_history requires num_envs == 1 (single eval path).")
        self._times: Optional[List[Any]] = list(times) if times is not None else None
        if self._times is not None and len(self._times) != self.n_bars:
            raise ValueError(
                f"times length {len(self._times)} != n_bars {self.n_bars}; pass the "
                f"same row index as the prices array.")
        # Per-step trajectory (filled in step()). Equity curve starts after reset().
        self.equity_curve: List[float] = []
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        self.time_history: List[Any] = []
        self.trades: List[Dict[str, Any]] = []
        # Per-lane open-trade record (eval-only, num_envs==1 so single dict is fine).
        self._open_trade: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    def _reset_mask(self, mask: torch.Tensor) -> None:
        k = int(mask.sum())
        if k == 0:
            return
        cfg = self.cfg
        if self.random_start and self._end - self._start > 256:
            hi = self._end - 128
            self.ptr[mask] = torch.randint(self._start, hi, (k,), generator=self.gen,
                                           device=self.device, dtype=torch.long)
        else:
            self.ptr[mask] = self._start
        for t, v in ((self.balance, cfg.initial_balance), (self.equity, cfg.initial_balance),
                     (self.peak, cfg.initial_balance)):
            t[mask] = v
        for t in (self.pos, self.entry_step, self.n_trades):
            t[mask] = 0
        self.last_trade_step[mask] = -10**9
        for t in (self.lots, self.entry, self.sl, self.tp, self._dsr_a, self._dsr_b):
            t[mask] = 0.0
        self.bh_units[mask] = cfg.initial_balance / self._close[self.ptr[mask]]

    def reset(self) -> torch.Tensor:
        self._reset_mask(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        if self.track_history:
            # Trajectory restart. Seed equity_curve with the post-reset equity
            # so the curve has len = n_steps + 1, matching the scalar env.
            self.equity_curve = [float(self.equity[0].item())]
            self.reward_history = []
            self.action_history = []
            self.time_history = [self._times[int(self.ptr[0].item())]] if self._times else []
            self.trades = []
            self._open_trade = None
        return self._obs()

    # ------------------------------------------------------------------ #
    def _unrealized(self, price: torch.Tensor) -> torch.Tensor:
        return (price - self.entry) * self.pos.to(self.dtype) * self.lots * self.cfg.contract_size

    def _obs(self) -> torch.Tensor:
        idx = self.ptr.unsqueeze(1) + self._offsets.unsqueeze(0)     # (N, W)
        window = self.features[idx].reshape(self.num_envs, -1)        # (N, W*F)
        price = self._close[self.ptr]
        init = self.cfg.initial_balance
        unreal = self._unrealized(price)
        bars = torch.where(self.pos != 0, self.ptr - self.entry_step,
                           torch.zeros_like(self.ptr)).to(self.dtype)
        acct = torch.stack([
            self.pos.to(self.dtype),
            torch.clamp(unreal / init, -1.0, 1.0),
            self.lots / max(self.cfg.max_position_lots, 1e-9),
            torch.clamp(self.equity / init - 1.0, -1.0, 1.0),
            torch.clamp(bars / 100.0, 0.0, 1.0),
        ], dim=1)
        return torch.clamp(torch.cat([window, acct], dim=1), -10.0, 10.0)

    def _realize(self, mask: torch.Tensor, exit_fill: torch.Tensor,
                 exit_reason: str = "unknown") -> None:
        cfg = self.cfg
        gross = (exit_fill - self.entry[mask]) * self.pos[mask].to(self.dtype) * self.lots[mask] * cfg.contract_size
        # Emit trade record (eval-only; single lane). Capture PnL/return BEFORE
        # we zero out pos/lots so the dict mirrors GoldTradingEnv's TradeRecord.
        if self.track_history and bool(mask[0].item()) and self._open_trade is not None:
            pnl = float(gross[0].item()) - 0.5 * cfg.commission_per_lot * float(self.lots[0].item())
            init = cfg.initial_balance
            exit_step = int(self.ptr[0].item())
            entry_step = int(self._open_trade["entry_step"])
            self.trades.append({
                "entry_time": self._open_trade["entry_time"],
                "exit_time": self._times[exit_step] if self._times else exit_step,
                "entry_price": float(self._open_trade["entry_price"]),
                "exit_price": float(exit_fill[0].item()),
                "direction": int(self._open_trade["direction"]),
                "lots": float(self._open_trade["lots"]),
                "pnl": pnl,
                "return_pct": pnl / max(init, 1e-9),
                "bars_held": exit_step - entry_step,
                "exit_reason": exit_reason,
            })
            self._open_trade = None
        self.balance[mask] += gross - 0.5 * cfg.commission_per_lot * self.lots[mask]
        self.pos[mask] = 0
        self.lots[mask] = 0.0

    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        a = actions.to(self.device).long().reshape(-1)
        price = self._close[self.ptr]
        equity_before = self.equity.clone()
        opened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 1. Apply actions at current close.
        flat = self.pos == 0
        cooldown_ok = (self.ptr - self.last_trade_step) >= cfg.min_bars_between_trades
        stop_dist = price * cfg.stop_loss_pct
        risk_cap = self.balance * cfg.risk_fraction
        lots = risk_cap / (stop_dist * cfg.contract_size)
        lots = torch.clamp(lots, max=cfg.max_position_lots)
        margin_per_lot = (price * cfg.contract_size) / cfg.leverage
        lots = torch.minimum(lots, self.balance / margin_per_lot)
        lots = torch.floor(lots / 0.01) * 0.01
        lots = torch.clamp(lots, min=0.0)

        open_dir = torch.where(a == 1, 1, torch.where(a == 2, -1, 0)).to(self.dtype)
        open_mask = ((a == 1) | (a == 2)) & flat & cooldown_ok & (lots > 0)
        # Regime gate (2026-05-28): suppress new entries when ATR/close < threshold.
        # Causal — uses ATR at the current bar (already computed at construction
        # from past OHLC). Exits / SL / TP are untouched.
        if cfg.min_trade_atr_pct > 0.0:
            atr_now = self._atr[self.ptr]
            atr_pct = atr_now / torch.clamp(price, min=1e-9)
            open_mask = open_mask & (atr_pct >= cfg.min_trade_atr_pct)
        if open_mask.any():
            d = open_dir[open_mask]
            fill = price[open_mask] + d * self._adverse
            lt = lots[open_mask]
            self.balance[open_mask] -= 0.5 * cfg.commission_per_lot * lt
            self.pos[open_mask] = d.long()
            self.lots[open_mask] = lt
            self.entry[open_mask] = fill
            self.sl[open_mask] = fill * (1.0 - d * cfg.stop_loss_pct)
            self.tp[open_mask] = fill * (1.0 + d * cfg.take_profit_pct)
            self.entry_step[open_mask] = self.ptr[open_mask]
            self.last_trade_step[open_mask] = self.ptr[open_mask]
            self.n_trades[open_mask] += 1
            opened[open_mask] = True
            # Eval bookkeeping: capture the trade's entry context for the trade
            # dict that will be emitted when _realize closes the position.
            if self.track_history and bool(open_mask[0].item()):
                entry_step = int(self.ptr[0].item())
                self._open_trade = {
                    "entry_step": entry_step,
                    "entry_time": self._times[entry_step] if self._times else entry_step,
                    "entry_price": float(fill[0].item()),
                    "direction": int(d[0].item()),
                    "lots": float(lt[0].item()),
                }

        close_mask = (a == 3) & (self.pos != 0)
        if close_mask.any():
            self._realize(
                close_mask,
                price[close_mask] - self.pos[close_mask].to(self.dtype) * self._adverse,
                exit_reason="signal",
            )

        # 2. Advance + resolve SL/TP.
        self.ptr += 1
        hi, lo = self._high[self.ptr], self._low[self.ptr]
        is_open = self.pos != 0
        long = self.pos > 0
        hit_sl = torch.where(long, lo <= self.sl, hi >= self.sl) & is_open
        hit_tp = torch.where(long, hi >= self.tp, lo <= self.tp) & is_open
        sl_mask = hit_sl
        tp_mask = hit_tp & ~hit_sl
        if sl_mask.any():
            self._realize(
                sl_mask,
                self.sl[sl_mask] - self.pos[sl_mask].to(self.dtype) * self._adverse,
                exit_reason="stop_loss",
            )
        if tp_mask.any():
            self._realize(
                tp_mask,
                self.tp[tp_mask] - self.pos[tp_mask].to(self.dtype) * self._adverse,
                exit_reason="take_profit",
            )

        # 3. Mark to market.
        next_close = self._close[self.ptr]
        self.equity = self.balance + self._unrealized(next_close)
        self.peak = torch.maximum(self.peak, self.equity)

        # 4. Reward.
        if cfg.reward_mode in ("excess", "dsr"):
            bh_change = self.bh_units * (next_close - price)
        else:
            bh_change = torch.zeros_like(self.equity)
        if cfg.reward_mode == "dsr":
            r = (self.equity - equity_before - bh_change) / max(cfg.initial_balance, 1e-9)
            d_a, d_b = r - self._dsr_a, r * r - self._dsr_b
            denom = self._dsr_b - self._dsr_a ** 2
            safe = torch.where(denom > 1e-10, denom, torch.ones_like(denom))
            reward = torch.where(denom > 1e-10,
                                 (self._dsr_b * d_a - 0.5 * self._dsr_a * d_b) / safe.pow(1.5),
                                 torch.zeros_like(denom))
            self._dsr_a = self._dsr_a + cfg.dsr_eta * d_a
            self._dsr_b = self._dsr_b + cfg.dsr_eta * d_b
            reward = torch.clamp(reward, -10.0, 10.0)
        else:
            net = (self.equity - equity_before - bh_change) * cfg.reward_scaling
            dd = (self.peak - self.equity) / torch.clamp(self.peak, min=1e-9)
            reward = (net - cfg.drawdown_penalty_weight * dd
                      - torch.where(opened, torch.tensor(cfg.overtrading_penalty, device=self.device), 0.0)
                      - torch.where(self.pos != 0, torch.tensor(cfg.holding_penalty, device=self.device), 0.0))
            reward = torch.clamp(reward, -10.0, 10.0)

        # 5. Termination + episode-end force-close + auto-reset.
        terminated = self.equity <= self._floor
        truncated = self.ptr >= self._end
        done = terminated | truncated
        force = done & (self.pos != 0)
        if force.any():
            self._realize(
                force,
                next_close[force] - self.pos[force].to(self.dtype) * self._adverse,
                exit_reason="force_close",
            )
            self.equity[force] = self.balance[force]
        if done.any():
            # Episode return (%) for logging.
            self.last_ep_returns = (self.equity[done] / cfg.initial_balance - 1.0) * 100.0
            if self.auto_reset:
                self._reset_mask(done)

        # Eval-mode trajectory capture (single lane, after force-close + equity
        # update so the stored sample is the same one compute_report consumes).
        if self.track_history:
            self.equity_curve.append(float(self.equity[0].item()))
            self.reward_history.append(float(reward[0].item()))
            self.action_history.append(int(a[0].item()))
            if self._times:
                self.time_history.append(self._times[int(self.ptr[0].item())])

        return self._obs(), reward, done

    # ------------------------------------------------------------------ #
    def get_episode_history(self) -> Dict[str, Any]:
        """Return the eval-mode trajectory as a dict matching the scalar env.

        Only meaningful after a deterministic single-path run (track_history=True,
        auto_reset=False, num_envs=1). The dict feeds ``compute_report`` and the
        ensemble eval path identically to ``GoldTradingEnv.get_episode_history``.
        """
        return {
            "equity_curve": np.asarray(self.equity_curve, dtype=float),
            "timestamps": list(self.time_history),
            "rewards": np.asarray(self.reward_history, dtype=float),
            "actions": list(self.action_history),
            "trades": list(self.trades),
            "final_balance": float(self.balance[0].item()),
            "final_equity": float(self.equity[0].item()),
            "initial_balance": float(self.cfg.initial_balance),
        }
