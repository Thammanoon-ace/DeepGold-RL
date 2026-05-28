"""
Extended technical indicators, organized into feature GROUPS (V3 / Phase 4B).

The roadmap asks for an "Indicator Expansion System" governed by four rules:

1. *incremental feature addition*  → features live in named, opt-in groups
2. *avoid feature explosion*       → groups are enabled explicitly via config
3. *avoid correlated indicators*   → see :mod:`utils.feature_selection`
4. *validate every indicator*      → see ``scripts/feature_ab.py`` (walk-forward A/B)

Every indicator here is **causal**: it uses only the current and past bars
(rolling windows, EMAs, shifts), so enabling any group cannot introduce
look-ahead. Each group function takes the OHLCV frame plus the
:class:`~config.config.FeatureConfig` and returns a DataFrame of named feature
columns aligned to the input index.

The base/core features (RSI, MACD, EMA, ATR, candle return, rolling vol, regime,
multi-timeframe) remain in :mod:`utils.feature_engineering`; the groups below
are *additional* and never duplicate the core set.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from config.config import FeatureConfig


# --------------------------------------------------------------------------- #
# Small causal helpers
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).clip(0.0, 100.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Causal rolling OLS slope of ``series`` vs. time, vectorized.

    Slope of the best-fit line over each trailing ``window`` of values. Positive
    => rising trend. Computed with the closed-form OLS coefficient.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    y = series.to_numpy(dtype=float)
    if len(y) < window:
        return pd.Series(np.nan, index=series.index)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = (x_centered**2).sum()

    win = sliding_window_view(y, window)                     # (N-window+1, window)
    y_centered = win - win.mean(axis=1, keepdims=True)
    slope = (x_centered * y_centered).sum(axis=1) / denom
    out = np.full(len(y), np.nan)
    out[window - 1:] = slope
    return pd.Series(out, index=series.index)


# --------------------------------------------------------------------------- #
# Feature groups (each returns a DataFrame of causal, scale-free features)
# --------------------------------------------------------------------------- #
def group_trend(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """SMA distance, rolling VWAP distance, fast/slow trend spread."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    sma = close.rolling(cfg.sma_period).mean()
    out["sma_dist"] = (close - sma) / close

    # Rolling VWAP over a window (causal): sum(price*vol)/sum(vol).
    vol = df.get("tick_volume", pd.Series(0.0, index=df.index)).clip(lower=0.0)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * vol).rolling(cfg.vwap_window).sum()
    vv = vol.rolling(cfg.vwap_window).sum().replace(0.0, np.nan)
    vwap = pv / vv
    out["vwap_dist"] = (close - vwap) / close

    # Spread between fast and slow EMAs, normalized by price.
    ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean()
    out["trend_spread"] = (ema_fast - ema_slow) / close
    return out


def group_momentum(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Stochastic RSI and rate-of-change."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    rsi = _rsi(close, cfg.rsi_period)
    lo = rsi.rolling(cfg.stoch_rsi_period).min()
    hi = rsi.rolling(cfg.stoch_rsi_period).max()
    out["stoch_rsi"] = ((rsi - lo) / (hi - lo).replace(0.0, np.nan)) - 0.5  # centered

    out["roc"] = close.pct_change(cfg.roc_period)
    return out


def group_volatility(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Bollinger %B / bandwidth and a historical-volatility estimate."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    mid = close.rolling(cfg.bb_period).mean()
    std = close.rolling(cfg.bb_period).std()
    upper = mid + cfg.bb_std * std
    lower = mid - cfg.bb_std * std
    width = (upper - lower).replace(0.0, np.nan)
    out["bb_pctb"] = ((close - lower) / width) - 0.5          # 0.5 -> centered at mid
    out["bb_bandwidth"] = (upper - lower) / mid

    log_ret = np.log(close / close.shift(1))
    out["hist_vol"] = log_ret.rolling(cfg.hist_vol_window).std()
    return out


def group_candle(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Single-bar candle structure (body, wicks, pressure, momentum)."""
    out = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0.0, np.nan)

    out["candle_body"] = (c - o) / rng                       # signed body fraction
    out["upper_wick"] = (h - np.maximum(o, c)) / rng
    out["lower_wick"] = (np.minimum(o, c) - l) / rng
    # Where the close sits within the bar's range: 1 = top (bull), 0 = bottom.
    out["close_pressure"] = (c - l) / rng - 0.5
    # Body size relative to recent volatility (momentum of the candle).
    atr = _atr(h, l, c, cfg.atr_period)
    out["candle_momentum"] = (c - o) / atr.replace(0.0, np.nan)
    return out


def group_structure(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Market-structure features: swing distance, breakout, slope, S/R, breaks.

    All use *prior* rolling extrema (shifted by 1) so the current bar's own
    high/low cannot define the level it is being compared against.
    """
    out = pd.DataFrame(index=df.index)
    h, l, c = df["high"], df["low"], df["close"]
    w = cfg.structure_window
    atr = _atr(h, l, c, cfg.atr_period).replace(0.0, np.nan)

    prior_high = h.rolling(w).max().shift(1)
    prior_low = l.rolling(w).min().shift(1)

    # Distance to recent resistance / support, normalized by price.
    out["resistance_dist"] = (prior_high - c) / c
    out["support_dist"] = (c - prior_low) / c

    # Breakout strength: how far beyond the prior range, in ATR units.
    out["breakout_up"] = ((c - prior_high) / atr).clip(lower=0.0)
    out["breakout_dn"] = ((prior_low - c) / atr).clip(lower=0.0)
    # Continuous structure-break signal (+ above prior high, - below prior low).
    out["structure_break"] = out["breakout_up"] - out["breakout_dn"]

    # Trend slope over a window, normalized by price.
    out["trend_slope"] = _rolling_slope(c, cfg.slope_window) / c
    return out


def group_volume(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Relative volume, volume spike z-score, liquidity-sweep depth."""
    out = pd.DataFrame(index=df.index)
    vol = df.get("tick_volume", pd.Series(0.0, index=df.index)).clip(lower=0.0)
    w = cfg.volume_window

    mean_v = vol.rolling(w).mean().replace(0.0, np.nan)
    std_v = vol.rolling(w).std().replace(0.0, np.nan)
    out["rel_volume"] = vol / mean_v
    out["volume_spike"] = (vol - mean_v) / std_v

    # Liquidity sweep: bar pierces the prior low/high then closes back inside —
    # depth of the sweep in ATR units (causal: prior extrema are shifted).
    h, l, c = df["high"], df["low"], df["close"]
    atr = _atr(h, l, c, cfg.atr_period).replace(0.0, np.nan)
    prior_low = l.rolling(w).min().shift(1)
    prior_high = h.rolling(w).max().shift(1)
    swept_low = ((prior_low - l) / atr).clip(lower=0.0) * (c > prior_low).astype(float)
    swept_high = ((h - prior_high) / atr).clip(lower=0.0) * (c < prior_high).astype(float)
    out["liquidity_sweep"] = swept_low - swept_high
    return out


# Registry: group name -> builder. Enable via FeatureConfig.feature_groups.
FEATURE_GROUPS: Dict[str, Callable[[pd.DataFrame, FeatureConfig], pd.DataFrame]] = {
    "trend": group_trend,
    "momentum": group_momentum,
    "volatility": group_volatility,
    "candle": group_candle,
    "structure": group_structure,
    "volume": group_volume,
}


def available_groups() -> List[str]:
    """Names of all registered feature groups."""
    return list(FEATURE_GROUPS)


def build_groups(df: pd.DataFrame, cfg: FeatureConfig, groups: List[str]) -> pd.DataFrame:
    """Build and concatenate the requested feature groups (causal)."""
    frames = []
    for name in groups:
        if name not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group {name!r}; available: {available_groups()}")
        frames.append(FEATURE_GROUPS[name](df, cfg))
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)
