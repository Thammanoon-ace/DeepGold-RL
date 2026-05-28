"""
Causal market-regime detection (V3.5 / Phase 5B).

Classifies each bar into one of four regimes along two axes:

* **trend axis** — trending vs ranging (absolute rolling trend slope, in ATR units)
* **volatility axis** — high vs low (realised volatility vs its long-run level)

=> {range_lowvol, range_highvol, trend_lowvol, trend_highvol}

Leakage controls (mirrors the normalizer's discipline):
* every indicator is causal (rolling slope / ATR / rolling std — past+present only);
* the trend/vol **thresholds are fit on TRAINING data only** (``fit``) and then
  merely *applied* to validation/test (``label``). No future information is used.

Two uses:
* ``add_features`` appends the continuous regime signals (trend strength, vol
  level) to the agent's observation features, letting the policy *condition* on
  regime (Phase 6 groundwork);
* ``label`` produces discrete regime labels for **regime-stratified evaluation**
  (does the edge hold per regime, or only in bull trends?).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.indicators import _atr, _rolling_slope

REGIME_NAMES: Dict[int, str] = {
    0: "range_lowvol",
    1: "range_highvol",
    2: "trend_lowvol",
    3: "trend_highvol",
}


class RegimeDetector:
    """Causal four-regime classifier with train-fit thresholds.

    Parameters
    ----------
    trend_window: lookback for the rolling trend slope.
    vol_window: lookback for realised volatility.
    atr_period: ATR period used to normalise the slope (scale-free).
    trend_quantile / vol_quantile: training quantiles that split trending-vs-
        ranging and high-vs-low volatility (0.5 = median split).
    """

    def __init__(
        self,
        trend_window: int = 50,
        vol_window: int = 20,
        atr_period: int = 14,
        trend_quantile: float = 0.5,
        vol_quantile: float = 0.5,
    ) -> None:
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.atr_period = atr_period
        self.trend_quantile = trend_quantile
        self.vol_quantile = vol_quantile
        self.trend_threshold: Optional[float] = None
        self.vol_threshold: Optional[float] = None

    # ------------------------------------------------------------------ #
    def _indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return causal trend-strength and vol-level signals (+ trend sign)."""
        close = df["close"]
        atr = _atr(df["high"], df["low"], close, self.atr_period).replace(0.0, np.nan)
        slope = _rolling_slope(close, self.trend_window)
        out = pd.DataFrame(index=df.index)
        # Trend strength = |slope per bar| expressed in ATR units (scale-free).
        out["trend_strength"] = (slope.abs() / atr)
        out["trend_dir"] = np.sign(slope)
        # Volatility level = realised return volatility over the window.
        out["vol_level"] = close.pct_change().rolling(self.vol_window).std()
        return out

    def fit(self, train_df: pd.DataFrame) -> "RegimeDetector":
        """Fit the trend/vol split thresholds on TRAINING data only."""
        ind = self._indicators(train_df).replace([np.inf, -np.inf], np.nan).dropna()
        self.trend_threshold = float(ind["trend_strength"].quantile(self.trend_quantile))
        self.vol_threshold = float(ind["vol_level"].quantile(self.vol_quantile))
        return self

    def label(self, df: pd.DataFrame) -> pd.Series:
        """Return integer regime labels (0-3) for each bar (causal)."""
        if self.trend_threshold is None:
            raise RuntimeError("RegimeDetector must be fit() before label().")
        ind = self._indicators(df)
        is_trend = (ind["trend_strength"] > self.trend_threshold).astype(int)
        is_highvol = (ind["vol_level"] > self.vol_threshold).astype(int)
        regime = 2 * is_trend + is_highvol
        return regime.rename("regime")

    def label_names(self, df: pd.DataFrame) -> pd.Series:
        """Regime labels as human-readable names."""
        return self.label(df).map(REGIME_NAMES)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append continuous regime signals to ``df`` for the agent to condition on.

        Adds ``regime_trend`` (trend strength) and ``regime_vol`` (vol level),
        both causal and scale-free-ish. Call after ``fit`` is not required here
        (these are raw signals; normalization happens downstream).
        """
        ind = self._indicators(df).replace([np.inf, -np.inf], 0.0)
        out = df.copy()
        out["regime_trend"] = ind["trend_strength"].fillna(0.0)
        out["regime_vol"] = ind["vol_level"].fillna(0.0)
        return out

    @property
    def feature_names(self) -> List[str]:
        return ["regime_trend", "regime_vol"]
