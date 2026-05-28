"""Backtesting package: evaluation engine, metrics and baselines."""
from .backtester import Backtester, run_episode
from . import baselines
from .metrics import (
    PerformanceReport,
    compute_report,
    compute_trade_distribution,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
)

__all__ = [
    "Backtester",
    "run_episode",
    "PerformanceReport",
    "compute_report",
    "compute_trade_distribution",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
]
