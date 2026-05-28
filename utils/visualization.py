"""
Matplotlib visualizations (requirement #7).

Pure plotting helpers used by the backtester and notebooks:

* equity curve (with drawdown shading),
* price chart annotated with trade entries/exits,
* per-episode reward curve.

All functions accept an optional ``save_path`` and return the Matplotlib
``Figure`` so they compose nicely inside Jupyter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

# Use a non-interactive backend when running head-less (scripts / CI). Jupyter
# overrides this automatically with its inline backend.
if matplotlib.get_backend().lower() == "agg":
    pass
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _maybe_save(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")


def plot_equity_curve(
    equity: Sequence[float],
    timestamps: Optional[Sequence] = None,
    initial_balance: Optional[float] = None,
    title: str = "Equity Curve",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the account equity over time with drawdown shading."""
    equity = np.asarray(equity, dtype=float)
    x = pd.to_datetime(timestamps) if timestamps is not None else np.arange(len(equity))

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(x, equity, color="#1f77b4", lw=1.3, label="Equity")
    if initial_balance is not None:
        ax1.axhline(initial_balance, color="grey", ls="--", lw=1, label="Initial balance")
    ax1.set_title(title)
    ax1.set_ylabel("Equity (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, drawdown * 100, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    if timestamps is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_trades(
    price: pd.Series,
    trades: Iterable[dict],
    title: str = "Trades on Price",
    save_path: Optional[str | Path] = None,
    max_points: int = 20_000,
) -> plt.Figure:
    """Plot the close price with entry/exit markers.

    Parameters
    ----------
    price:
        Time-indexed close-price series.
    trades:
        Iterable of trade dicts with at least ``entry_time``, ``exit_time``,
        ``entry_price``, ``exit_price`` and ``direction`` (+1 long / -1 short).
    """
    # Down-sample very long series so plotting stays responsive.
    if len(price) > max_points:
        step = len(price) // max_points
        price = price.iloc[::step]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(price.index, price.values, color="#333333", lw=0.8, label="Close")

    for tr in trades:
        long = tr.get("direction", 1) > 0
        entry_color = "#2ca02c" if long else "#d62728"
        ax.scatter(
            tr["entry_time"], tr["entry_price"],
            marker="^" if long else "v",
            color=entry_color, s=60, zorder=5,
            edgecolors="black", linewidths=0.4,
        )
        ax.scatter(
            tr["exit_time"], tr["exit_price"],
            marker="o", color="black", s=30, zorder=5,
        )
        ax.plot(
            [tr["entry_time"], tr["exit_time"]],
            [tr["entry_price"], tr["exit_price"]],
            color=entry_color, lw=0.8, alpha=0.5,
        )

    ax.set_title(title)
    ax.set_ylabel("Price (USD/oz)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_reward_curve(
    rewards: Sequence[float],
    title: str = "Cumulative Reward",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot step rewards and their cumulative sum."""
    rewards = np.asarray(rewards, dtype=float)
    cumulative = np.cumsum(rewards)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(rewards, color="#9467bd", lw=0.6)
    ax1.set_ylabel("Step reward")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    ax2.plot(cumulative, color="#ff7f0e", lw=1.2)
    ax2.set_ylabel("Cumulative reward")
    ax2.set_xlabel("Step")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_training_curves(
    episode_rewards: Sequence[float],
    title: str = "Training Episode Rewards",
    save_path: Optional[str | Path] = None,
    smooth: int = 20,
) -> plt.Figure:
    """Plot raw and smoothed episode-reward learning curve."""
    rewards = np.asarray(episode_rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rewards, color="#cccccc", lw=0.8, label="Episode reward")
    if len(rewards) >= smooth:
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            np.arange(smooth - 1, len(rewards)),
            smoothed, color="#1f77b4", lw=1.8, label=f"Moving avg ({smooth})",
        )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
