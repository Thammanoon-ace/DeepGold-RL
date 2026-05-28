"""
Multi-dataset evaluation (V2 / Phase 3).

Runs a *single already-trained* agent across several datasets — different
symbols, instruments, or time periods — to test how well its learned behaviour
generalizes beyond the data it was trained on. Robustness across diverse,
genuinely unseen markets is a strong anti-overfitting signal.

Leakage / skew control: every dataset is transformed with the **saved
training-time normalizer** (never re-fit), so the agent receives inputs scaled
exactly as during training.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from stable_baselines3 import PPO

from backtest.backtester import run_episode
from backtest.metrics import PerformanceReport, compute_report
from config.config import Config
from env.env_builder import make_env_from_frame
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


class MultiDatasetEvaluator:
    """Evaluate one trained model on many datasets and tabulate the results.

    Parameters
    ----------
    config:
        Full :class:`~config.config.Config`.
    model_name:
        Base name used to locate ``models/<name>.zip`` and the saved normalizer.
    """

    def __init__(self, config: Config, model_name: Optional[str] = None) -> None:
        self.config = config
        self.model_name = model_name or config.training.model_name
        self.model: Optional[PPO] = None
        self.normalizer: Optional[FeatureNormalizer] = None
        self.engineer = FeatureEngineer(config.features)

    def load(self, device: str = "auto") -> None:
        """Load the trained model and the training-time feature normalizer."""
        models = self.config.paths.models
        self.model = PPO.load(models / f"{self.model_name}.zip", device=device)
        self.normalizer = FeatureNormalizer.load(
            models / f"{self.model_name}_normalizer.joblib"
        )
        logger.info("Loaded model + normalizer for '%s'.", self.model_name)

    def evaluate_dataset(
        self,
        csv_path: str | Path,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> PerformanceReport:
        """Evaluate the model on one CSV, optionally restricted to a date range."""
        if self.model is None or self.normalizer is None:
            raise RuntimeError("Call load() before evaluate_dataset().")

        loader = HistoricalDataLoader(self.config.data)
        raw = loader.load_csv(csv_path)
        featured = self.engineer.transform(raw)

        if date_range is not None:
            start, end = date_range
            mask = (featured.index >= pd.Timestamp(start)) & (
                featured.index <= pd.Timestamp(end)
            )
            featured = featured.loc[mask]

        feat_cols = self.normalizer.feature_columns
        normalized = self.normalizer.transform(featured)
        env = make_env_from_frame(normalized, feat_cols, self.config.env, random_start=False)
        hist = run_episode(self.model, env, deterministic=True)
        return compute_report(
            hist["equity_curve"], hist["trades"], hist["initial_balance"],
            risk_free_rate=self.config.backtest.risk_free_rate,
            bars_per_year=self.config.backtest.bars_per_year,
        )

    def run(
        self,
        datasets: Sequence[Tuple[str, str | Path]],
        date_range: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """Evaluate across ``[(name, csv_path), ...]`` and return a comparison table."""
        if self.model is None:
            self.load()

        rows: List[dict] = []
        for name, path in datasets:
            try:
                rep = self.evaluate_dataset(path, date_range=date_range)
            except Exception as exc:  # one bad file should not abort the batch
                logger.exception("Failed to evaluate dataset %s (%s): %s", name, path, exc)
                continue
            rows.append({
                "dataset": name,
                "return_pct": rep.total_return_pct,
                "sharpe": rep.sharpe_ratio,
                "calmar": rep.calmar_ratio,
                "max_dd_pct": rep.max_drawdown_pct,
                "win_rate_pct": rep.win_rate_pct,
                "profit_factor": rep.profit_factor,
                "n_trades": rep.n_trades,
            })
            logger.info("[%s] return %+.2f%% | sharpe %.2f | trades %d",
                        name, rep.total_return_pct, rep.sharpe_ratio, rep.n_trades)

        df = pd.DataFrame(rows)
        if not df.empty:
            summary = {
                "dataset": "MEAN", "return_pct": df["return_pct"].mean(),
                "sharpe": df["sharpe"].mean(), "calmar": df["calmar"].mean(),
                "max_dd_pct": df["max_dd_pct"].mean(),
                "win_rate_pct": df["win_rate_pct"].mean(),
                "profit_factor": df["profit_factor"].replace([float("inf")], pd.NA).mean(),
                "n_trades": df["n_trades"].mean(),
            }
            df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        return df
