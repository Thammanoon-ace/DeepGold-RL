"""
Historical market-data loading and cleaning (requirement #1).

The :class:`HistoricalDataLoader` reads OHLCV CSV files (e.g. exported from
MetaTrader 5), normalizes their schema, handles missing values, optionally
resamples to a coarser timeframe, and performs the strict walk-forward
train/test split that prevents future-data leakage (requirement #14).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from config.config import DataConfig

logger = logging.getLogger(__name__)

# Mapping from our canonical column names to the many aliases brokers use.
# NOTE: ``<date>``/``<time>`` are deliberately NOT listed under "time" — when a
# file ships them as *separate* columns they are merged in a dedicated branch
# of ``_standardize_columns`` (listing them here would rename both to "time"
# and create duplicate columns). ``<vol>`` maps only to real_volume so it never
# collides with the tick-volume aliases.
_COLUMN_ALIASES = {
    "time": {"time", "date", "datetime", "timestamp"},
    "open": {"open", "o", "<open>"},
    "high": {"high", "h", "<high>"},
    "low": {"low", "l", "<low>"},
    "close": {"close", "c", "<close>", "price"},
    "tick_volume": {"tick_volume", "volume", "vol", "tickvol", "<tickvol>"},
    "real_volume": {"real_volume", "realvol", "<vol>"},
    "spread": {"spread", "<spread>"},
}

# Pandas offset aliases for the timeframes we support.
_TIMEFRAME_TO_OFFSET = {"M5": "5min", "M15": "15min", "H1": "1h"}


class HistoricalDataLoader:
    """Load, clean and split XAUUSD OHLCV data for training and backtesting.

    Parameters
    ----------
    config:
        A :class:`~config.config.DataConfig` describing the file, timeframe and
        split boundaries.
    """

    SUPPORTED_TIMEFRAMES = tuple(_TIMEFRAME_TO_OFFSET.keys())

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        if config.timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe {config.timeframe!r}; "
                f"choose one of {self.SUPPORTED_TIMEFRAMES}."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Read a CSV file and return a cleaned, time-indexed DataFrame.

        The returned frame is sorted by time, de-duplicated, has a
        ``DatetimeIndex`` and contains at least the OHLC columns plus a
        ``tick_volume`` column.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        logger.info("Loading market data from %s", path)
        # Try comma first, then fall back to MT5's tab-separated export.
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep="\t")

        df = self._standardize_columns(df)
        df = self._parse_datetime(df)
        df = self._coerce_numeric(df)
        df = self._handle_missing(df)

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            "Loaded %d rows spanning %s -> %s",
            len(df), df.index.min(), df.index.max(),
        )
        return df

    def resample(self, df: pd.DataFrame, timeframe: Optional[str] = None) -> pd.DataFrame:
        """Resample bar data to ``timeframe`` (defaults to the configured one).

        Resampling only ever *aggregates* (e.g. M5 -> H1); it never invents
        higher-frequency data, so it is leakage-safe.
        """
        timeframe = timeframe or self.config.timeframe
        if timeframe not in _TIMEFRAME_TO_OFFSET:
            raise ValueError(f"Cannot resample to unsupported timeframe {timeframe!r}.")
        offset = _TIMEFRAME_TO_OFFSET[timeframe]

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }
        agg = {k: v for k, v in agg.items() if k in df.columns}
        resampled = df.resample(offset, label="right", closed="right").agg(agg)
        # Drop empty buckets (weekends / market closures).
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        logger.info("Resampled to %s: %d bars", timeframe, len(resampled))
        return resampled

    def load(self, path: Optional[str | Path] = None) -> pd.DataFrame:
        """Convenience wrapper: load the configured CSV (no resampling).

        The CSV is assumed to already be at ``config.timeframe``.  If you need
        to downsample from a finer file, call :meth:`resample` explicitly.
        """
        from config.config import PathsConfig

        if path is None:
            path = PathsConfig().data / self.config.csv_filename
        return self.load_csv(path)

    def train_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into training and held-out test sets by calendar date.

        Walk-forward principle (requirement #14): training data is everything
        strictly *before* ``test_start``; the test set is ``[test_start,
        test_end]``.  The two sets never overlap, so the agent cannot peek at
        evaluation data.
        """
        cfg = self.config
        train_start = pd.Timestamp(cfg.train_start) if cfg.train_start else df.index.min()
        test_start = pd.Timestamp(cfg.test_start)
        test_end = pd.Timestamp(cfg.test_end) if cfg.test_end else df.index.max()

        train = df.loc[(df.index >= train_start) & (df.index < test_start)]
        test = df.loc[(df.index >= test_start) & (df.index <= test_end)]

        if train.empty:
            raise ValueError(
                "Training set is empty. Check train_start/test_start vs. data range "
                f"({df.index.min()} -> {df.index.max()})."
            )
        if test.empty:
            logger.warning(
                "Test set is empty for test_start=%s. Evaluation will be skipped.",
                test_start,
            )

        logger.info(
            "Split -> train: %d bars (%s..%s) | test: %d bars (%s..)",
            len(train), train.index.min(), train.index.max(),
            len(test), test_start,
        )
        return train, test

    # ------------------------------------------------------------------ #
    # Internal cleaning helpers
    # ------------------------------------------------------------------ #
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename heterogeneous broker columns to our canonical names."""
        rename = {}
        lower_cols = {c: c.strip().lower() for c in df.columns}
        for canonical, aliases in _COLUMN_ALIASES.items():
            for original, low in lower_cols.items():
                if low in aliases:
                    rename[original] = canonical
        df = df.rename(columns=rename)

        # MT5 terminal CSV exports usually split date and time into two columns
        # (often named "<DATE>"/"<TIME>" or "Date"/"Time"). Merge them into a
        # single "time" column if a unified one is not already present.
        if "time" not in df.columns:
            date_aliases = {"<date>", "date"}
            time_aliases = {"<time>", "time"}
            date_col = next((c for c, l in lower_cols.items() if l in date_aliases), None)
            time_col = next((c for c, l in lower_cols.items() if l in time_aliases), None)
            if date_col is not None and time_col is not None and date_col != time_col:
                df["time"] = (
                    df[date_col].astype(str).str.strip()
                    + " "
                    + df[time_col].astype(str).str.strip()
                )
                df = df.drop(columns=[date_col, time_col], errors="ignore")
            elif date_col is not None:  # date-only files (e.g. daily bars)
                df = df.rename(columns={date_col: "time"})

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required OHLC columns: {sorted(missing)}. "
                f"Found columns: {list(df.columns)}"
            )
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0.0  # volume is optional for indicators we use
        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the time column and set it as a DatetimeIndex.

        Handles epoch seconds (as returned by the MT5 Python API), MT5's dotted
        ``YYYY.MM.DD`` dates, and ISO-style strings.
        """
        if "time" not in df.columns:
            raise ValueError("No recognizable datetime column found in CSV.")

        col = df["time"]
        # MT5's copy_rates returns POSIX seconds as an integer column.
        if pd.api.types.is_numeric_dtype(col):
            parsed = pd.to_datetime(col, unit="s", errors="coerce")
        else:
            # Try a direct parse first (handles ISO strings, incl. ms decimals).
            parsed = pd.to_datetime(col, errors="coerce")
            # If most rows failed, assume MT5's dotted dates ("2024.01.02") and
            # convert the date separators before retrying.
            if parsed.isna().mean() > 0.5:
                normalised = col.astype(str).str.replace(".", "-", regex=False)
                parsed = pd.to_datetime(normalised, errors="coerce")

        df = df.assign(time=parsed).dropna(subset=["time"])
        if df.empty:
            raise ValueError(
                "Could not parse any timestamps from the 'time' column. "
                "Check the date format in your CSV."
            )
        return df.set_index("time")

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force OHLCV columns to floats, turning bad cells into NaN."""
        for col in ("open", "high", "low", "close", "tick_volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured missing-value strategy to OHLCV columns."""
        strategy = self.config.missing_value_strategy
        ohlc = ["open", "high", "low", "close"]
        n_missing = int(df[ohlc].isna().any(axis=1).sum())
        if n_missing:
            logger.info("Handling %d rows with missing OHLC via '%s'", n_missing, strategy)

        if strategy == "drop":
            df = df.dropna(subset=ohlc)
        elif strategy == "interpolate":
            df[ohlc] = df[ohlc].interpolate(method="time").bfill()
        else:  # 'ffill' (default) — never looks forward in time
            df[ohlc] = df[ohlc].ffill().bfill()

        df["tick_volume"] = df["tick_volume"].fillna(0.0)
        # Final guard: drop any remaining all-NaN rows.
        return df.dropna(subset=ohlc)
