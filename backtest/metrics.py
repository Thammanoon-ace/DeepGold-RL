"""
Performance metrics for backtesting (requirement #6).

Standard, honestly-computed trading statistics derived from an equity curve and
a list of closed trades.  Metrics are intentionally conservative and based on
mark-to-market equity (not cherry-picked realized PnL) so they cannot be gamed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class PerformanceReport:
    """Container for all computed metrics (easy to print / serialize)."""

    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    expectancy: float            # expected PnL per trade, in account currency
    payoff_ratio: float          # avg win / avg loss (absolute)
    n_trades: int
    avg_trade_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_bars_held: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    final_equity: float
    initial_balance: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def pretty(self) -> str:
        """Return a human-readable multi-line summary."""
        return (
            "Backtest Performance\n"
            "--------------------\n"
            f"  Initial balance : {self.initial_balance:,.2f}\n"
            f"  Final equity    : {self.final_equity:,.2f}\n"
            f"  Total return    : {self.total_return_pct:+.2f}%\n"
            f"  CAGR            : {self.cagr_pct:+.2f}%\n"
            f"  Sharpe ratio    : {self.sharpe_ratio:.2f}\n"
            f"  Sortino ratio   : {self.sortino_ratio:.2f}\n"
            f"  Calmar ratio    : {self.calmar_ratio:.2f}\n"
            f"  Max drawdown    : {self.max_drawdown_pct:.2f}%\n"
            f"  Win rate        : {self.win_rate_pct:.2f}%\n"
            f"  Profit factor   : {self.profit_factor:.2f}\n"
            f"  Expectancy/trade: {self.expectancy:+,.2f}\n"
            f"  Payoff ratio    : {self.payoff_ratio:.2f}\n"
            f"  Trades          : {self.n_trades}\n"
            f"  Avg trade       : {self.avg_trade_pct:+.3f}%\n"
            f"  Avg win / loss  : {self.avg_win_pct:+.3f}% / {self.avg_loss_pct:+.3f}%\n"
            f"  Avg bars held   : {self.avg_bars_held:.1f}\n"
            f"  Max consec W/L  : {self.max_consecutive_wins} / {self.max_consecutive_losses}\n"
        )


def max_drawdown(equity: Sequence[float]) -> float:
    """Return the maximum drawdown as a positive fraction (0..1)."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    return float(np.max(drawdowns))


def sharpe_ratio(
    equity: Sequence[float], risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio computed from per-bar equity returns."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 3:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino_ratio(
    equity: Sequence[float], risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized Sortino ratio (downside-deviation denominator)."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 3:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    excess = returns - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / dd)


def calmar_ratio(cagr_pct: float, max_drawdown_pct: float) -> float:
    """Calmar ratio = annualized return / max drawdown (both in %)."""
    if max_drawdown_pct <= 0:
        return 0.0
    return float(cagr_pct / max_drawdown_pct)


def _max_consecutive(flags: np.ndarray) -> int:
    """Longest run of True values in a boolean array."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return int(best)


def compute_trade_distribution(trades: List[dict]) -> Dict[str, float]:
    """Summarize the distribution of closed-trade outcomes.

    Returns percentile PnL, win/loss streaks, holding time and a breakdown of
    exit reasons (stop_loss / take_profit / signal / episode_end). Useful for
    spotting pathologies like "all profit from one lucky trade" or "exits are
    almost entirely stop-losses".
    """
    if not trades:
        return {"n_trades": 0}

    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    bars = np.array([t.get("bars_held", 0) for t in trades], dtype=float)
    wins = pnls > 0

    reasons: Dict[str, int] = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "n_trades": int(len(pnls)),
        "pnl_p05": float(np.percentile(pnls, 5)),
        "pnl_median": float(np.median(pnls)),
        "pnl_p95": float(np.percentile(pnls, 95)),
        "pnl_std": float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0,
        "best_trade": float(pnls.max()),
        "worst_trade": float(pnls.min()),
        "max_consecutive_wins": _max_consecutive(wins),
        "max_consecutive_losses": _max_consecutive(~wins),
        "avg_bars_held": float(bars.mean()),
        "exit_reasons": reasons,
        # Share of total gross profit from the single best trade — a high value
        # warns that results hinge on one outlier.
        "top_trade_profit_share": float(
            pnls.max() / pnls[pnls > 0].sum()
        ) if np.any(pnls > 0) else 0.0,
    }


def compute_report(
    equity_curve: Sequence[float],
    trades: List[dict],
    initial_balance: float,
    risk_free_rate: float = 0.0,
    bars_per_year: int = 252,
) -> PerformanceReport:
    """Aggregate all metrics into a :class:`PerformanceReport`.

    Parameters
    ----------
    equity_curve:
        Per-bar mark-to-market equity (length ``T``).
    trades:
        List of closed-trade dicts with ``pnl`` and ``return_pct`` keys.
    initial_balance:
        Starting account balance.
    bars_per_year:
        Used to annualize Sharpe/Sortino and the CAGR exponent.
    """
    equity = np.asarray(equity_curve, dtype=float)
    final_equity = float(equity[-1]) if equity.size else initial_balance

    total_return = (final_equity / initial_balance - 1.0) * 100.0

    n_bars = max(equity.size - 1, 1)
    years = n_bars / bars_per_year
    if years > 0 and final_equity > 0:
        cagr = ((final_equity / initial_balance) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr = 0.0

    # Trade-level stats.
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    rets = np.array([t.get("return_pct", 0.0) for t in trades], dtype=float) * 100.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = (len(wins) / len(pnls) * 100.0) if len(pnls) else 0.0
    gross_profit = wins.sum() if wins.size else 0.0
    gross_loss = -losses.sum() if losses.size else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Expectancy (expected PnL per trade) and payoff ratio (avg win / avg loss).
    expectancy = float(pnls.mean()) if pnls.size else 0.0
    avg_win_abs = float(wins.mean()) if wins.size else 0.0
    avg_loss_abs = float(-losses.mean()) if losses.size else 0.0
    payoff_ratio = (avg_win_abs / avg_loss_abs) if avg_loss_abs > 0 else 0.0

    max_dd_pct = max_drawdown(equity) * 100.0
    dist = compute_trade_distribution(trades)

    return PerformanceReport(
        total_return_pct=total_return,
        cagr_pct=cagr,
        sharpe_ratio=sharpe_ratio(equity, risk_free_rate, bars_per_year),
        sortino_ratio=sortino_ratio(equity, risk_free_rate, bars_per_year),
        calmar_ratio=calmar_ratio(cagr, max_dd_pct),
        max_drawdown_pct=max_dd_pct,
        win_rate_pct=win_rate,
        profit_factor=float(profit_factor),
        expectancy=expectancy,
        payoff_ratio=payoff_ratio,
        n_trades=int(len(pnls)),
        avg_trade_pct=float(rets.mean()) if rets.size else 0.0,
        avg_win_pct=float(rets[rets > 0].mean()) if np.any(rets > 0) else 0.0,
        avg_loss_pct=float(rets[rets < 0].mean()) if np.any(rets < 0) else 0.0,
        avg_bars_held=float(dist.get("avg_bars_held", 0.0)),
        max_consecutive_wins=int(dist.get("max_consecutive_wins", 0)),
        max_consecutive_losses=int(dist.get("max_consecutive_losses", 0)),
        final_equity=final_equity,
        initial_balance=initial_balance,
    )
