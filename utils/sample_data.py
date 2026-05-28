"""
Synthetic XAUUSD data generator (for pipeline testing only).

IMPORTANT
---------
This module fabricates a *plausible-looking* gold price series using a
geometric-Brownian-motion + regime/seasonality model.  It exists purely so the
end-to-end pipeline (load -> features -> train -> backtest) can run out of the
box without a broker account.

It is **not** real market data and any "profit" obtained on it is meaningless.
For genuine research, replace ``data/XAUUSD_M5.csv`` with a real MT5 export.
This directly supports requirement #14: do not build a fake overfitted profit
generator — synthetic data is clearly labelled as a test fixture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Bars per trading day for each supported timeframe (24h FX-style sessions).
_BARS_PER_DAY = {"M5": 288, "M15": 96, "H1": 24}
_FREQ = {"M5": "5min", "M15": "15min", "H1": "1h"}


def generate_synthetic_ohlcv(
    start: str = "2019-01-01",
    end: str = "2025-12-31",
    timeframe: str = "M5",
    start_price: float = 1_300.0,
    annual_drift: float = 0.06,
    annual_vol: float = 0.16,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame indexed by timestamp.

    The series uses GBM for the close, adds intrabar high/low noise scaled by
    local volatility, and injects mild volatility regimes so technical
    indicators have something non-trivial to react to.
    """
    if timeframe not in _FREQ:
        raise ValueError(f"Unsupported timeframe {timeframe!r}.")

    rng = np.random.default_rng(seed)
    # Restrict to weekdays to imitate an FX-style trading week.
    index = pd.date_range(start=start, end=end, freq=_FREQ[timeframe])
    index = index[index.weekday < 5]
    n = len(index)

    bars_per_year = _BARS_PER_DAY[timeframe] * 252
    dt = 1.0 / bars_per_year

    # Slowly varying volatility regime (Ornstein-Uhlenbeck-like).
    vol_regime = np.ones(n)
    shock = rng.normal(0, 1, n)
    v = 1.0
    for i in range(n):
        v += 0.001 * (1.0 - v) + 0.05 * shock[i] * np.sqrt(dt)
        vol_regime[i] = max(0.3, v)

    mu = annual_drift
    sigma = annual_vol * vol_regime
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(0, 1, n)
    log_price = np.log(start_price) + np.cumsum(increments)
    close = np.exp(log_price)

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    bar_range = close * sigma * np.sqrt(dt) * rng.uniform(0.5, 2.0, n)
    high = np.maximum(open_, close) + bar_range * rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - bar_range * rng.uniform(0, 1, n)
    tick_volume = rng.integers(50, 500, n).astype(float) * vol_regime

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume,
        },
        index=index,
    )
    df.index.name = "time"
    return df


def write_sample_csv(
    path: str | Path,
    timeframe: str = "M5",
    rows_limit: Optional[int] = None,
    **kwargs,
) -> Path:
    """Generate synthetic data and write it to ``path`` as CSV.

    Returns the path written.
    """
    df = generate_synthetic_ohlcv(timeframe=timeframe, **kwargs)
    if rows_limit is not None:
        df = df.iloc[:rows_limit]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="time")
    return path
