"""
Custom Stable-Baselines3 callbacks for richer TensorBoard logging.

``TradingMetricsCallback`` reads the ``info`` dicts emitted by
:class:`GoldTradingEnv` and logs trading-specific quantities (equity, number of
trades, drawdown) to TensorBoard, on top of SB3's default reward/length stats.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """Log per-rollout trading metrics to TensorBoard.

    Parameters
    ----------
    log_freq:
        How often (in calls to ``_on_step``) to push aggregated metrics.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # ``infos`` is a list (one per vectorized env) for the current step.
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        equities = [i.get("equity") for i in infos if "equity" in i]
        drawdowns = [i.get("drawdown") for i in infos if "drawdown" in i]
        n_trades = [i.get("n_trades") for i in infos if "n_trades" in i]

        if equities:
            self.logger.record("trading/equity_mean", float(np.mean(equities)))
        if drawdowns:
            self.logger.record("trading/drawdown_mean", float(np.mean(drawdowns)))
        if n_trades:
            self.logger.record("trading/trades_mean", float(np.mean(n_trades)))
        return True


class ProgressLogCallback(BaseCallback):
    """Lightweight stdout progress logger (handy in notebooks)."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.verbose and self.num_timesteps % self.log_freq < self.training_env.num_envs:
            logger.info("Timesteps: %d", self.num_timesteps)
        return True
