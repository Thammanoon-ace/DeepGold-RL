"""
Factory that turns raw CSV data into leakage-safe train / test environments.

This is the single place where the full data pipeline is assembled:

    load CSV -> engineer features -> walk-forward split
             -> fit normalizer on TRAIN only -> build GoldTradingEnv(s)

Centralizing it guarantees that training and backtesting use *exactly* the same
feature definitions and the *same* fitted scaler, and that the scaler is never
fit on test data (requirement #14).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import Config, EnvConfig
from env.gold_trading_env import GoldTradingEnv
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


def make_env_from_frame(
    df: pd.DataFrame,
    feature_columns: List[str],
    env_config: EnvConfig,
    random_start: bool = False,
    render_mode: Optional[str] = None,
) -> GoldTradingEnv:
    """Build a :class:`GoldTradingEnv` directly from a prepared DataFrame.

    The DataFrame must already contain the (normalized) ``feature_columns`` plus
    raw ``open/high/low/close`` columns. This standalone factory is shared by the
    pipeline, the walk-forward validator and the multi-dataset evaluator so they
    all construct environments identically.
    """
    features = df[feature_columns].to_numpy(dtype=np.float32)
    prices = df[["open", "high", "low", "close"]].copy()
    return GoldTradingEnv(
        features=features,
        prices=prices,
        config=env_config,
        random_start=random_start,
        render_mode=render_mode,
    )


class TradingDataPipeline:
    """Builds normalized train/test frames and environment factories.

    Parameters
    ----------
    config:
        The full :class:`~config.config.Config`.
    normalizer_method:
        Scaler type passed to :class:`FeatureNormalizer` ("robust"/"standard").
    """

    def __init__(self, config: Config, normalizer_method: str = "robust") -> None:
        self.config = config
        self.loader = HistoricalDataLoader(config.data)
        self.engineer = FeatureEngineer(config.features)
        self.normalizer_method = normalizer_method

        self.normalizer: Optional[FeatureNormalizer] = None
        self.feature_columns: List[str] = []
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        # Raw (featured but NOT normalized) splits, kept so a different
        # normalizer (e.g. the one saved at training time) can be applied later
        # without double-normalizing an already-scaled frame.
        self.train_raw: Optional[pd.DataFrame] = None
        self.test_raw: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def prepare(
        self, csv_path: Optional[str | Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full pipeline and return normalized (train_df, test_df).

        Steps, in order, are all causal and leakage-free:

        1. Load + clean the CSV.
        2. Engineer features (causal indicators only).
        3. Walk-forward split by date.
        4. Fit the normalizer **on training data only**, transform both sets.
        """
        raw = self.loader.load(csv_path)
        featured = self.engineer.transform(raw)
        self.feature_columns = self.engineer.feature_columns

        train_raw, test_raw = self.loader.train_test_split(featured)
        self.train_raw, self.test_raw = train_raw, test_raw

        # Phase 4B: drop near-constant / highly-correlated features (fit on the
        # training split only, then applied everywhere via the kept-column list).
        threshold = self.config.features.correlation_threshold
        if threshold and threshold > 0:
            from utils.feature_selection import FeatureSelector

            selector = FeatureSelector(correlation_threshold=threshold)
            selector.fit(train_raw, self.feature_columns)
            self.feature_columns = selector.kept_columns
            self.config.features.feature_columns = self.feature_columns

        self.normalizer = FeatureNormalizer(
            self.feature_columns, method=self.normalizer_method
        )
        self.train_df = self.normalizer.fit_transform(train_raw)
        self.test_df = (
            self.normalizer.transform(test_raw) if not test_raw.empty else test_raw
        )
        return self.train_df, self.test_df

    def apply_normalizer(self, normalizer: FeatureNormalizer) -> None:
        """Re-normalize the splits from RAW using an externally supplied scaler.

        Used by the backtester to apply the *saved training-time* normalizer
        without double-normalizing the already-scaled frames produced by
        :meth:`prepare`.
        """
        if self.train_raw is None:
            raise RuntimeError("Call prepare() before apply_normalizer().")
        self.normalizer = normalizer
        self.feature_columns = normalizer.feature_columns
        self.train_df = normalizer.transform(self.train_raw)
        self.test_df = (
            normalizer.transform(self.test_raw)
            if self.test_raw is not None and not self.test_raw.empty
            else self.test_raw
        )

    # ------------------------------------------------------------------ #
    # Environment construction
    # ------------------------------------------------------------------ #
    def make_env(
        self, which: str = "train", random_start: Optional[bool] = None,
        render_mode: Optional[str] = None,
    ) -> GoldTradingEnv:
        """Build a single :class:`GoldTradingEnv` for ``'train'`` or ``'test'``.

        Training environments default to random episode starts (more diverse
        rollouts); test environments are deterministic (full walk-through).
        """
        if self.train_df is None:
            raise RuntimeError("Call prepare() before make_env().")
        df = self.train_df if which == "train" else self.test_df
        if df is None or df.empty:
            raise ValueError(f"No data available for split {which!r}.")
        if random_start is None:
            random_start = which == "train"

        return make_env_from_frame(
            df, self.feature_columns, self.config.env,
            random_start=random_start, render_mode=render_mode,
        )

    def make_vectorized_env(
        self, which: str = "train", num_envs: int = 32,
        random_start: Optional[bool] = None, seed: Optional[int] = None,
    ):
        """Build a high-throughput :class:`VectorizedGoldTradingEnv` (V3.5/5E).

        A NumPy-batched VecEnv that steps ``num_envs`` lanes in one process —
        far faster than ``DummyVecEnv`` of scalar envs for training. For
        deterministic evaluation/backtests keep using ``make_env`` + the scalar
        env (single path, full trade records).
        """
        from env.vectorized_env import VectorizedGoldTradingEnv

        if self.train_df is None:
            raise RuntimeError("Call prepare() before make_vectorized_env().")
        df = self.train_df if which == "train" else self.test_df
        if df is None or df.empty:
            raise ValueError(f"No data available for split {which!r}.")
        if random_start is None:
            random_start = which == "train"
        features = df[self.feature_columns].to_numpy(dtype=np.float32)
        prices = df[["high", "low", "close"]].to_numpy(dtype=np.float64)
        return VectorizedGoldTradingEnv(
            features, prices, self.config.env,
            num_envs=num_envs, random_start=random_start, seed=seed,
        )

    def save_normalizer(self, path: Optional[str | Path] = None) -> Path:
        """Persist the fitted normalizer next to the model (requirement #8)."""
        if self.normalizer is None:
            raise RuntimeError("Nothing to save; call prepare() first.")
        if path is None:
            path = self.config.paths.models / "normalizer.joblib"
        self.normalizer.save(path)
        return Path(path)
