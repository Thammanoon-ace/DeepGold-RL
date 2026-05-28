"""
Feature normalization with leakage-safe fit/transform (requirements #1/#8).

The :class:`FeatureNormalizer` wraps a scikit-learn scaler.  Critically, it is
**fit on the training set only** and then *applied* to the test set.  Fitting on
the combined data — or on the test data — would leak the distribution of the
future into the past, inflating backtest results (requirement #14).

The fitted scaler is persisted alongside the trained model so that live trading
applies the exact same transformation the agent was trained on.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """Fit/transform feature columns and persist the scaler + config.

    Parameters
    ----------
    feature_columns:
        Columns to normalize.  Other columns (raw OHLCV) pass through untouched.
    method:
        ``"standard"`` (zero mean, unit variance) or ``"robust"`` (median / IQR,
        more resistant to the fat tails common in financial returns).
    """

    def __init__(
        self, feature_columns: List[str], method: str = "robust"
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.method = method
        self._scaler = self._make_scaler(method)
        self._fitted = False

    @staticmethod
    def _make_scaler(method: str):
        if method == "standard":
            return StandardScaler()
        if method == "robust":
            return RobustScaler()
        raise ValueError(f"Unknown normalization method {method!r}.")

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame) -> "FeatureNormalizer":
        """Fit the scaler on TRAINING data only (no leakage)."""
        self._scaler.fit(train_df[self.feature_columns].values)
        self._fitted = True
        logger.info(
            "Fitted %s scaler on %d training rows / %d features.",
            self.method, len(train_df), len(self.feature_columns),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with feature columns normalized."""
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer must be fit before transform().")
        out = df.copy()
        scaled = self._scaler.transform(out[self.feature_columns].values)
        # Clip to a sane range to keep the policy network's inputs bounded even
        # when the test set contains larger outliers than training.
        scaled = np.clip(scaled, -10.0, 10.0)
        out[self.feature_columns] = scaled.astype(np.float32)
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the training set and return its normalized copy."""
        return self.fit(train_df).transform(train_df)

    # ------------------------------------------------------------------ #
    # Persistence (requirement #8)
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Persist the fitted scaler (joblib) plus a JSON side-car of config."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, path)
        meta = {
            "method": self.method,
            "feature_columns": self.feature_columns,
            "fitted": self._fitted,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved normalizer to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """Reconstruct a fitted normalizer previously written by :meth:`save`."""
        import joblib

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        obj = cls(meta["feature_columns"], method=meta["method"])
        obj._scaler = joblib.load(path)
        obj._fitted = bool(meta.get("fitted", True))
        logger.info("Loaded normalizer from %s", path)
        return obj
