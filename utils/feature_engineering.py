"""
Technical-indicator feature engineering (requirement #2).

Indicators are implemented in pure pandas/NumPy so the framework has **no
native dependency** (no TA-Lib C library), which keeps the virtual-environment
install trivial on Windows + CUDA GPU setups.

Leakage safety (requirement #14)
--------------------------------
Every indicator here is *causal*: it uses only the current bar and past bars
(rolling windows, exponential moving averages, differences).  None of them
reference future rows, so features computed on the test set cannot leak
information backwards in time.  The warm-up rows that contain NaNs (because an
indicator needs ``period`` bars of history) are dropped.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from config.config import FeatureConfig

logger = logging.getLogger(__name__)

# Pandas offset aliases for higher timeframes that can be merged onto a base
# (intraday) series. Used by multi-timeframe feature engineering.
_HTF_OFFSET = {"M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}


class FeatureEngineer:
    """Compute a causal technical-indicator feature set from OHLCV data.

    Parameters
    ----------
    config:
        A :class:`~config.config.FeatureConfig` with indicator periods.  After
        :meth:`transform`, ``config.feature_columns`` is populated with the
        names of the engineered columns so downstream modules know exactly
        which columns to feed the agent.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Individual indicators (static, reusable, fully causal)
    # ------------------------------------------------------------------ #
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothing."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        # Wilder smoothing == EMA with alpha = 1/period.
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # When there are no losses RSI is 100; when no gains it is 0.
        rsi = rsi.fillna(100.0).where(avg_loss != 0, 100.0)
        return rsi.clip(0.0, 100.0)

    @staticmethod
    def macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence (line, signal, histogram)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
        )

    @staticmethod
    def ema(close: pd.Series, span: int) -> pd.Series:
        """Exponential moving average."""
        return close.ewm(span=span, adjust=False).mean()

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range using Wilder's smoothing."""
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # ------------------------------------------------------------------ #
    # Full feature pipeline
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` augmented with engineered features.

        The original OHLCV columns are preserved (the environment needs raw
        ``close`` for PnL accounting); engineered feature names are recorded in
        ``self.config.feature_columns``.
        """
        cfg = self.config
        out = df.copy()

        # --- Trend / momentum -------------------------------------------- #
        out["rsi"] = self.rsi(out["close"], cfg.rsi_period)
        macd_df = self.macd(out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        out = out.join(macd_df)
        out["ema_fast"] = self.ema(out["close"], cfg.ema_fast)
        out["ema_slow"] = self.ema(out["close"], cfg.ema_slow)
        # Distance of price from each EMA, expressed as a fraction (scale-free).
        out["ema_fast_dist"] = (out["close"] - out["ema_fast"]) / out["close"]
        out["ema_slow_dist"] = (out["close"] - out["ema_slow"]) / out["close"]

        # --- Volatility -------------------------------------------------- #
        out["atr"] = self.atr(out["high"], out["low"], out["close"], cfg.atr_period)
        out["atr_pct"] = out["atr"] / out["close"]

        # --- Returns ----------------------------------------------------- #
        # Candle return % uses the *previous* close, so it is strictly causal.
        out["candle_return"] = out["close"].pct_change()
        out["rolling_volatility"] = (
            out["candle_return"].rolling(cfg.volatility_window).std()
        )

        # The agent receives scale-free / bounded features only (raw price is
        # excluded to avoid the policy keying off absolute price levels, which
        # is a classic source of overfitting and look-ahead-style artefacts).
        feature_columns: List[str] = [
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "ema_fast_dist",
            "ema_slow_dist",
            "atr_pct",
            "candle_return",
            "rolling_volatility",
        ]

        # --- Volatility regime (V3) -------------------------------------- #
        # Current short-term volatility relative to its longer-run median: >1
        # means a high-volatility regime, <1 a calm one. Causal (past windows).
        if cfg.regime_window and cfg.regime_window > 0:
            baseline = out["rolling_volatility"].rolling(cfg.regime_window).median()
            out["vol_regime"] = out["rolling_volatility"] / baseline.replace(0.0, np.nan)
            feature_columns.append("vol_regime")

        # --- Multi-timeframe features (V3) ------------------------------- #
        # Merge higher-timeframe indicators onto the base bars, causally.
        for htf in cfg.multi_timeframe:
            htf_feats = self._higher_tf_features(df, htf)
            out = out.join(htf_feats)
            feature_columns.extend(list(htf_feats.columns))

        # --- Indicator Expansion groups (V3 / Phase 4B) ------------------ #
        # Opt-in extra feature groups, added incrementally via config.
        if cfg.feature_groups:
            from utils.indicators import build_groups

            group_feats = build_groups(out, cfg, cfg.feature_groups)
            out = out.join(group_feats)
            feature_columns.extend(list(group_feats.columns))

        # --- Regime signals (V3.5 / 5B) ---------------------------------- #
        # Causal trend-strength + vol-level so the agent can condition on regime.
        if getattr(cfg, "use_regime_features", False):
            from utils.regime import RegimeDetector

            rf = RegimeDetector(atr_period=cfg.atr_period).add_features(out)
            for col in ("regime_trend", "regime_vol"):
                out[col] = rf[col]
                feature_columns.append(col)

        cfg.feature_columns = feature_columns

        # Cleanup that preserves time continuity (critical for the env, which
        # steps through *contiguous* bars):
        #   1. inf -> NaN (from divisions on degenerate windows),
        #   2. drop only the leading warm-up region (until every feature is
        #      valid), so no interior bars are removed,
        #   3. fill any remaining sporadic interior NaN with a neutral 0 — some
        #      Phase 4B features (e.g. volume spike on a zero-volume window) can
        #      be NaN mid-series, and dropping those rows would tear the series.
        before = len(out)
        out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
        all_valid = out[feature_columns].notna().all(axis=1)
        if not all_valid.any():
            raise ValueError("No fully-valid feature rows after engineering.")
        start = all_valid.idxmax()  # first index where every feature is valid
        out = out.loc[start:].copy()
        out[feature_columns] = out[feature_columns].fillna(0.0)
        logger.info(
            "Engineered %d features; trimmed %d warm-up rows (%d -> %d).",
            len(feature_columns), before - len(out), before, len(out),
        )
        return out

    def _higher_tf_features(self, df: pd.DataFrame, htf: str) -> pd.DataFrame:
        """Compute higher-timeframe indicators and align them causally to base bars.

        The base OHLC is resampled to ``htf`` with right-labelled, right-closed
        bars (so each HTF bar is timestamped at its *close*). The HTF indicators
        are then re-indexed onto the base index with forward-fill: a base bar at
        time ``t`` therefore only ever sees the most recent HTF bar that has
        already closed at or before ``t`` — no future information leaks down.
        """
        if htf not in _HTF_OFFSET:
            raise ValueError(
                f"Unsupported higher timeframe {htf!r}; choose from {list(_HTF_OFFSET)}."
            )
        cfg = self.config
        offset = _HTF_OFFSET[htf]
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        bars = df.resample(offset, label="right", closed="right").agg(agg).dropna()

        prefix = htf.lower()
        feats = pd.DataFrame(index=bars.index)
        feats[f"{prefix}_rsi"] = self.rsi(bars["close"], cfg.rsi_period)
        macd_df = self.macd(bars["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        feats[f"{prefix}_macd_hist"] = macd_df["macd_hist"]
        ema_fast = self.ema(bars["close"], cfg.ema_fast)
        ema_slow = self.ema(bars["close"], cfg.ema_slow)
        feats[f"{prefix}_ema_fast_dist"] = (bars["close"] - ema_fast) / bars["close"]
        feats[f"{prefix}_ema_slow_dist"] = (bars["close"] - ema_slow) / bars["close"]
        atr = self.atr(bars["high"], bars["low"], bars["close"], cfg.atr_period)
        feats[f"{prefix}_atr_pct"] = atr / bars["close"]

        # Causal down-merge: forward-fill the last closed HTF bar onto base bars.
        return feats.reindex(df.index, method="ffill")

    @property
    def feature_columns(self) -> List[str]:
        """Names of the engineered feature columns (after :meth:`transform`)."""
        return self.config.feature_columns
