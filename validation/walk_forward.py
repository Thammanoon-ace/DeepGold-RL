"""
Walk-forward validation (V2 / Phase 3) — the headline robustness tool.

Instead of a single train/2025-test split, walk-forward repeatedly trains the
agent on one window and evaluates it on the *immediately following* unseen
window, marching forward through history. Stable performance across many such
out-of-sample folds is far stronger evidence of genuine edge than one lucky
backtest, and it directly attacks overfitting (the core goal of V2).

Leakage controls (consistent with the rest of the framework):
* Features are engineered once on the full series, but every indicator is
  causal, so slicing a fold out is identical to computing it on that slice.
* The :class:`FeatureNormalizer` is re-fit on **each fold's training slice
  only** and then applied to that fold's test slice.
* An optional embargo ``gap`` separates train and test windows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from backtest.backtester import run_episode
from backtest.metrics import PerformanceReport, compute_report
from config.config import Config
from env.env_builder import make_env_from_frame
from policies.factory import build_policy_kwargs
from training.train_ppo import _progress_bar_available, resolve_device
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.normalization import FeatureNormalizer
from validation.splitters import TimeSeriesSplitter

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Aggregated outcome of a walk-forward run."""

    fold_reports: List[PerformanceReport]
    fold_meta: List[dict]
    stitched_equity: np.ndarray
    stitched_timestamps: list = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Per-fold metrics as a tidy DataFrame."""
        rows = []
        for meta, rep in zip(self.fold_meta, self.fold_reports):
            row = {
                "fold": meta["fold"],
                "train_start": meta["train_start"],
                "train_end": meta["train_end"],
                "test_start": meta["test_start"],
                "test_end": meta["test_end"],
                "return_pct": rep.total_return_pct,
                "sharpe": rep.sharpe_ratio,
                "calmar": rep.calmar_ratio,
                "max_dd_pct": rep.max_drawdown_pct,
                "win_rate_pct": rep.win_rate_pct,
                "profit_factor": rep.profit_factor,
                "n_trades": rep.n_trades,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def aggregate(self) -> dict:
        """Cross-fold summary statistics (the headline numbers)."""
        df = self.to_frame()
        rets = df["return_pct"].to_numpy()
        # Compounded return chaining each fold's OOS result.
        compounded = float(np.prod(1.0 + rets / 100.0) - 1.0) * 100.0
        return {
            "n_folds": len(df),
            "mean_return_pct": float(df["return_pct"].mean()),
            "std_return_pct": float(df["return_pct"].std(ddof=0)),
            "compounded_return_pct": compounded,
            "pct_profitable_folds": float((df["return_pct"] > 0).mean() * 100.0),
            "mean_sharpe": float(df["sharpe"].mean()),
            "mean_max_dd_pct": float(df["max_dd_pct"].mean()),
            "worst_fold_return_pct": float(df["return_pct"].min()),
            "mean_win_rate_pct": float(df["win_rate_pct"].mean()),
            "mean_trades": float(df["n_trades"].mean()),
        }

    def summary(self) -> str:
        agg = self.aggregate()
        return (
            "Walk-Forward Summary\n"
            "--------------------\n"
            f"  Folds                 : {agg['n_folds']}\n"
            f"  Mean OOS return / fold: {agg['mean_return_pct']:+.2f}%  "
            f"(std {agg['std_return_pct']:.2f})\n"
            f"  Compounded OOS return : {agg['compounded_return_pct']:+.2f}%\n"
            f"  Profitable folds      : {agg['pct_profitable_folds']:.0f}%\n"
            f"  Worst fold return     : {agg['worst_fold_return_pct']:+.2f}%\n"
            f"  Mean Sharpe           : {agg['mean_sharpe']:.2f}\n"
            f"  Mean max drawdown     : {agg['mean_max_dd_pct']:.2f}%\n"
            f"  Mean win rate         : {agg['mean_win_rate_pct']:.1f}%\n"
            f"  Mean trades / fold    : {agg['mean_trades']:.0f}\n"
        )


class WalkForwardValidator:
    """Run walk-forward training+evaluation over a dataset.

    Parameters
    ----------
    config:
        Full :class:`~config.config.Config`.
    timesteps_per_fold:
        PPO training timesteps for each fold. Keep modest — total cost scales
        linearly with the number of folds.
    splitter:
        A :class:`TimeSeriesSplitter`; if omitted, a sensible expanding-window
        splitter is created.
    normalizer_method:
        Scaler type for per-fold normalization.
    """

    def __init__(
        self,
        config: Config,
        timesteps_per_fold: int = 50_000,
        splitter: Optional[TimeSeriesSplitter] = None,
        normalizer_method: str = "robust",
        use_subproc: bool = False,
    ) -> None:
        self.config = config
        self.timesteps_per_fold = timesteps_per_fold
        self.normalizer_method = normalizer_method
        # SubprocVecEnv spawns a full torch interpreter per worker; only enable
        # on high-RAM hosts with a genuinely heavy env (default off).
        self.use_subproc = use_subproc
        self.device = resolve_device(config.training.device)
        self.splitter = splitter or TimeSeriesSplitter(
            n_splits=5, mode="expanding", gap=config.env.window_size,
        )
        self.loader = HistoricalDataLoader(config.data)
        self.engineer = FeatureEngineer(config.features)
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------ #
    def prepare_features(self, csv_path=None) -> pd.DataFrame:
        """Load and feature-engineer the full series once (causal -> sliceable)."""
        raw = self.loader.load(csv_path)
        featured = self.engineer.transform(raw)
        self.feature_columns = self.engineer.feature_columns
        return featured

    def _train_fold(self, train_df: pd.DataFrame, seed: int,
                    feature_columns: Optional[List[str]] = None) -> PPO:
        """Train a fresh PPO model on one fold's (normalized) training slice."""
        tcfg = self.config.training
        feat_cols = feature_columns or self.feature_columns

        def _make():
            env = make_env_from_frame(
                train_df, feat_cols, self.config.env, random_start=True
            )
            return Monitor(env)

        # Use DummyVecEnv (single process, multiple envs stepped in-process):
        # it gives the GPU-batching benefit of many envs without the memory cost
        # of SubprocVecEnv, which spawns a full torch/SB3 interpreter per worker
        # (~1-2 GB each) and exhausts RAM/page file on modest machines. Our env
        # is also too lightweight for subprocess IPC to pay off. SubprocVecEnv is
        # available via ``use_subproc`` for heavy envs / high-RAM hosts.
        n_envs = max(int(tcfg.n_envs), 1)
        env_fns = [_make for _ in range(n_envs)]
        if self.use_subproc and n_envs > 1:
            try:
                vec = SubprocVecEnv(env_fns, start_method="spawn")
            except Exception:
                logger.warning("SubprocVecEnv failed; using DummyVecEnv.", exc_info=True)
                vec = DummyVecEnv(env_fns)
        else:
            vec = DummyVecEnv(env_fns)
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=tcfg.gamma)
        model = PPO(
            policy=tcfg.policy, env=vec,
            learning_rate=tcfg.learning_rate, n_steps=tcfg.n_steps,
            batch_size=tcfg.batch_size, n_epochs=tcfg.n_epochs,
            gamma=tcfg.gamma, gae_lambda=tcfg.gae_lambda, clip_range=tcfg.clip_range,
            ent_coef=tcfg.ent_coef, vf_coef=tcfg.vf_coef, max_grad_norm=tcfg.max_grad_norm,
            policy_kwargs=build_policy_kwargs(tcfg, self.config.env.window_size),
            device=self.device, seed=seed, verbose=0,
        )
        model.learn(total_timesteps=self.timesteps_per_fold,
                    progress_bar=_progress_bar_available())
        return model

    def run(self, csv_path=None) -> WalkForwardResult:
        """Execute the full walk-forward procedure and return aggregated results."""
        featured = self.prepare_features(csv_path)
        folds = self.splitter.split(len(featured))
        logger.info("Walk-forward: %d folds over %d bars (%s window, gap=%d).",
                    len(folds), len(featured), self.splitter.mode, self.splitter.gap)

        reports: List[PerformanceReport] = []
        meta: List[dict] = []
        stitched_equity: List[float] = []
        stitched_times: List = []
        running_capital = self.config.env.initial_balance

        for fold in folds:
            train_raw = featured.iloc[fold.train_start:fold.train_end]
            test_raw = featured.iloc[fold.test_start:fold.test_end]
            ts_train = (featured.index[fold.train_start], featured.index[fold.train_end - 1])
            ts_test = (featured.index[fold.test_start], featured.index[fold.test_end - 1])
            logger.info(
                "Fold %d | train %s..%s (%d) | test %s..%s (%d)",
                fold.index, ts_train[0], ts_train[1], fold.train_len,
                ts_test[0], ts_test[1], fold.test_len,
            )

            # Per-fold feature selection (Phase 4B) — fit on this fold's TRAIN
            # slice only, so correlation/variance pruning never sees test data.
            fold_cols = self.feature_columns
            threshold = self.config.features.correlation_threshold
            if threshold and threshold > 0:
                from utils.feature_selection import FeatureSelector

                fold_cols = FeatureSelector(correlation_threshold=threshold).fit(
                    train_raw, self.feature_columns
                ).kept_columns

            # Per-fold normalizer fit on TRAIN ONLY (no leakage).
            normalizer = FeatureNormalizer(fold_cols, self.normalizer_method)
            train_df = normalizer.fit_transform(train_raw)
            test_df = normalizer.transform(test_raw)

            model = self._train_fold(train_df, seed=self.config.training.seed + fold.index,
                                     feature_columns=fold_cols)

            test_env = make_env_from_frame(
                test_df, fold_cols, self.config.env, random_start=False
            )
            hist = run_episode(model, test_env, deterministic=True)
            report = compute_report(
                hist["equity_curve"], hist["trades"], hist["initial_balance"],
                risk_free_rate=self.config.backtest.risk_free_rate,
                bars_per_year=self.config.backtest.bars_per_year,
            )
            reports.append(report)
            meta.append({
                "fold": fold.index,
                "train_start": str(ts_train[0]), "train_end": str(ts_train[1]),
                "test_start": str(ts_test[0]), "test_end": str(ts_test[1]),
            })

            # Stitch this fold's OOS equity onto the running compounded curve.
            eq = np.asarray(hist["equity_curve"], dtype=float)
            scaled = running_capital * (eq / eq[0])
            stitched_equity.extend(scaled.tolist())
            stitched_times.extend(hist["timestamps"])
            running_capital = float(scaled[-1])
            logger.info("Fold %d OOS return: %+.2f%%", fold.index, report.total_return_pct)

        result = WalkForwardResult(
            fold_reports=reports,
            fold_meta=meta,
            stitched_equity=np.asarray(stitched_equity, dtype=float),
            stitched_timestamps=stitched_times,
        )
        logger.info("\n%s", result.summary())
        return result
