"""
Feature selection: drop near-constant and highly-correlated features (Phase 4B).

Implements the roadmap's "avoid feature explosion / avoid correlated indicators"
rules. Selection is **fit on training data only** (the kept-column list is then
applied unchanged to validation/test/live), so it introduces no leakage. The
kept list rides along with the normalizer's stored ``feature_columns``, keeping
train/serve consistent.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def variance_filter(df: pd.DataFrame, feature_cols: List[str], min_std: float = 1e-8) -> List[str]:
    """Return columns whose training standard deviation exceeds ``min_std``."""
    stds = df[feature_cols].std(ddof=0)
    kept = [c for c in feature_cols if stds.get(c, 0.0) > min_std]
    dropped = [c for c in feature_cols if c not in kept]
    if dropped:
        logger.info("Variance filter dropped %d near-constant feature(s): %s", len(dropped), dropped)
    return kept


def correlation_filter(
    df: pd.DataFrame, feature_cols: List[str], threshold: float = 0.95
) -> List[str]:
    """Greedily drop features that are highly correlated with a kept feature.

    Iterates features in their given order, keeping a feature only if its
    absolute Pearson correlation with every already-kept feature is below
    ``threshold``. Order-dependent by design — list higher-priority features
    first if you care which of a correlated pair survives.
    """
    if threshold <= 0 or threshold >= 1 or len(feature_cols) < 2:
        return list(feature_cols)

    corr = df[feature_cols].corr().abs()
    kept: List[str] = []
    dropped: List[str] = []
    for col in feature_cols:
        if any(corr.loc[col, k] >= threshold for k in kept):
            dropped.append(col)
        else:
            kept.append(col)
    if dropped:
        logger.info(
            "Correlation filter (>=%.2f) dropped %d feature(s): %s",
            threshold, len(dropped), dropped,
        )
    return kept


def mutual_information_ranking(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int = 1,
    target: str = "abs_return",
    seed: int = 0,
) -> pd.Series:
    """Rank features by mutual information with the forward return (Phase 5D).

    Mutual information captures non-linear dependence a correlation filter misses.
    The target is the *forward* return over ``horizon`` bars — used **only on
    training data to rank/select features** (the agent never sees it), so this is
    a feature-analysis step, not a leak. ``target='abs_return'`` ranks by
    relevance to volatility/magnitude; ``'return'`` by directional relevance.

    Returns a Series of MI scores indexed by feature, sorted descending.
    """
    from sklearn.feature_selection import mutual_info_regression

    fwd = df["close"].pct_change(horizon).shift(-horizon)
    y = fwd.abs() if target == "abs_return" else fwd
    data = df[feature_cols].join(y.rename("_target")).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(data) < 100:
        return pd.Series(0.0, index=feature_cols)
    mi = mutual_info_regression(
        data[feature_cols].to_numpy(), data["_target"].to_numpy(), random_state=seed
    )
    return pd.Series(mi, index=feature_cols).sort_values(ascending=False)


def select_top_k_by_mi(
    df: pd.DataFrame, feature_cols: List[str], k: int, **kwargs
) -> List[str]:
    """Keep the ``k`` features with the highest mutual information (train-fit)."""
    ranking = mutual_information_ranking(df, feature_cols, **kwargs)
    kept = list(ranking.head(k).index)
    logger.info("MI selection kept top %d/%d features: %s", len(kept), len(feature_cols), kept)
    return kept


class FeatureSelector:
    """Fit-on-train feature selector combining variance + correlation filters.

    Parameters
    ----------
    correlation_threshold:
        Drop a feature correlated above this with an already-kept feature
        (<=0 disables correlation filtering).
    min_std:
        Drop features whose training std is below this (near-constant).
    """

    def __init__(self, correlation_threshold: float = 0.95, min_std: float = 1e-8) -> None:
        self.correlation_threshold = correlation_threshold
        self.min_std = min_std
        self.kept_columns: Optional[List[str]] = None

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "FeatureSelector":
        cols = variance_filter(train_df, feature_cols, self.min_std)
        cols = correlation_filter(train_df, cols, self.correlation_threshold)
        self.kept_columns = cols
        logger.info("FeatureSelector kept %d / %d features.", len(cols), len(feature_cols))
        return self
