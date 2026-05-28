"""
Optuna hyperparameter tuning (V2 / Phase 3).

Searches PPO (and selected reward/env) hyperparameters to maximize an
*out-of-sample validation* objective.

Critical anti-leakage rule
--------------------------
Tuning is performed on a **validation slice carved out of the training data**
(the last ``val_fraction`` of pre-2025 bars) — NEVER on the 2025 test set.
Optimizing hyperparameters against the final test set would leak it into the
model-selection process and inflate reported performance. The 2025 set stays
pristine for the single, final backtest after tuning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from backtest.backtester import run_episode
from backtest.metrics import compute_report
from config.config import Config
from env.env_builder import make_env_from_frame
from policies.factory import build_policy_kwargs
from training.train_ppo import resolve_device
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


@dataclass
class TuningSplit:
    """Normalized train/validation frames used during tuning."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    feature_columns: list


class PPOTuner:
    """Bayesian hyperparameter search for the PPO gold agent.

    Parameters
    ----------
    config:
        Full :class:`~config.config.Config` (used as the search baseline).
    n_trials:
        Number of Optuna trials.
    timesteps_per_trial:
        PPO timesteps trained per trial — keep small; this runs ``n_trials`` times.
    val_fraction:
        Fraction of the (pre-2025) training data reserved as the validation slice.
    metric:
        Validation objective to maximize: ``'sharpe'`` (default, robust),
        ``'return'`` or ``'calmar'``.
    """

    def __init__(
        self,
        config: Config,
        n_trials: int = 20,
        timesteps_per_trial: int = 30_000,
        val_fraction: float = 0.2,
        metric: str = "sharpe",
    ) -> None:
        self.config = config
        self.n_trials = n_trials
        self.timesteps_per_trial = timesteps_per_trial
        self.val_fraction = val_fraction
        self.metric = metric
        self.device = resolve_device(config.training.device)
        self._split: Optional[TuningSplit] = None

    # ------------------------------------------------------------------ #
    def _prepare_split(self, csv_path=None) -> TuningSplit:
        """Build the train/validation split, normalizing on the train slice only."""
        loader = HistoricalDataLoader(self.config.data)
        engineer = FeatureEngineer(self.config.features)
        featured = engineer.transform(loader.load(csv_path))
        feat_cols = engineer.feature_columns

        # Restrict to the pre-2025 TRAINING period; the 2025 test set is never
        # touched during tuning.
        train_full, _test = loader.train_test_split(featured)
        n = len(train_full)
        cut = int(n * (1.0 - self.val_fraction))
        train_raw = train_full.iloc[:cut]
        val_raw = train_full.iloc[cut:]

        normalizer = FeatureNormalizer(feat_cols, method="robust")
        train_df = normalizer.fit_transform(train_raw)   # fit on train slice only
        val_df = normalizer.transform(val_raw)
        logger.info("Tuning split -> train %d bars | validation %d bars", len(train_df), len(val_df))
        return TuningSplit(train_df, val_df, feat_cols)

    def _sample_params(self, trial) -> dict:
        """Sample a PPO hyperparameter configuration for a trial."""
        net = trial.suggest_categorical("net_arch", ["64_64", "128_128", "256_256"])
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "n_epochs": trial.suggest_int("n_epochs", 4, 15),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
            "net_arch": [int(x) for x in net.split("_")],
        }

    def _objective(self, trial) -> float:
        params = self._sample_params(trial)
        split = self._split
        tcfg = self.config.training

        def _make():
            return Monitor(make_env_from_frame(
                split.train_df, split.feature_columns, self.config.env, random_start=True
            ))

        vec = DummyVecEnv([_make])
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=params["gamma"])
        # For non-MLP architectures the feature extractor is fixed by config;
        # the sampled net_arch only applies to the MLP policy heads.
        if (tcfg.policy_arch or "mlp").lower() == "mlp":
            policy_kwargs = dict(net_arch=params["net_arch"])
        else:
            policy_kwargs = build_policy_kwargs(tcfg, self.config.env.window_size)
        model = PPO(
            policy=tcfg.policy, env=vec,
            learning_rate=params["learning_rate"], n_steps=params["n_steps"],
            batch_size=params["batch_size"], n_epochs=params["n_epochs"],
            gamma=params["gamma"], gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"], ent_coef=params["ent_coef"],
            policy_kwargs=policy_kwargs,
            device=self.device, seed=tcfg.seed, verbose=0,
        )
        model.learn(total_timesteps=self.timesteps_per_trial, progress_bar=False)

        # Evaluate on the held-out VALIDATION slice (not the 2025 test set).
        val_env = make_env_from_frame(
            split.val_df, split.feature_columns, self.config.env, random_start=False
        )
        hist = run_episode(model, val_env, deterministic=True)
        report = compute_report(
            hist["equity_curve"], hist["trades"], hist["initial_balance"],
            risk_free_rate=self.config.backtest.risk_free_rate,
            bars_per_year=self.config.backtest.bars_per_year,
        )
        value = {
            "sharpe": report.sharpe_ratio,
            "return": report.total_return_pct,
            "calmar": report.calmar_ratio,
        }[self.metric]
        # Guard against NaN/inf objectives.
        if not np.isfinite(value):
            value = -1e9
        trial.set_user_attr("val_return_pct", report.total_return_pct)
        trial.set_user_attr("val_sharpe", report.sharpe_ratio)
        trial.set_user_attr("n_trades", report.n_trades)
        return float(value)

    def optimize(self, csv_path=None):
        """Run the Optuna study and return it (study.best_params holds the winner)."""
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Optuna is required for tuning. Install with `pip install optuna`."
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._split = self._prepare_split(csv_path)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.training.seed),
        )
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info("Best %s = %.4f", self.metric, study.best_value)
        logger.info("Best params: %s", study.best_params)
        return study
