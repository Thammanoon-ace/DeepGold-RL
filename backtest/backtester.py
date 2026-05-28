"""
Backtesting / evaluation engine (requirement #6).

``Backtester`` loads a trained PPO model plus its saved normalization artefacts
and runs the agent deterministically over the **held-out 2025** test set,
collecting the equity curve, trades and rewards, computing a
:class:`PerformanceReport`, and (optionally) rendering the standard charts.

Because the test set was never seen during training and the normalizer was fit
on training data only, this is a genuine out-of-sample, walk-forward evaluation
(requirement #14) — not an in-sample fit.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from backtest.metrics import PerformanceReport, compute_report
from config.config import Config
from env.env_builder import TradingDataPipeline
from env.gold_trading_env import GoldTradingEnv
from utils import visualization as viz

logger = logging.getLogger(__name__)


def run_episode(model, env: GoldTradingEnv, deterministic: bool = True) -> Dict[str, Any]:
    """Run a policy deterministically over one full pass of ``env``.

    Shared by :class:`Backtester`, the walk-forward validator and the
    multi-dataset evaluator so every evaluation executes the agent identically.
    Returns the environment's episode history dict.
    """
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        done = terminated or truncated
    return env.get_episode_history()


class Backtester:
    """Run a trained agent over unseen data and report performance.

    Parameters
    ----------
    config:
        The full :class:`~config.config.Config`.
    pipeline:
        Optional pre-built :class:`TradingDataPipeline`.  If provided already
        prepared, its fitted normalizer is reused; otherwise the backtester
        prepares data and (preferably) loads the saved normalizer to guarantee
        identical scaling to training.
    """

    def __init__(self, config: Config, pipeline: Optional[TradingDataPipeline] = None) -> None:
        self.config = config
        self.pipeline = pipeline or TradingDataPipeline(config)
        self.model: Optional[PPO] = None
        self.history: Optional[Dict[str, Any]] = None
        self.report: Optional[PerformanceReport] = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load_model(self, model_path: str | Path, device: str = "auto"):
        """Load a trained model (.zip) for inference, honouring the algo in config."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        algo = (self.config.training.algo or "ppo").lower()
        if algo == "dqn":
            from stable_baselines3 import DQN

            self.model = DQN.load(model_path, device=device)
        else:
            self.model = PPO.load(model_path, device=device)
        logger.info("Loaded %s model from %s", algo.upper(), model_path)
        return self.model

    def _ensure_data(self, model_name: Optional[str]) -> None:
        """Prepare data and reuse the *saved* normalizer when available.

        Loading the persisted normalizer (rather than re-fitting) ensures the
        exact training-time scaling is applied to the test set.
        """
        if self.pipeline.train_df is None:
            self.pipeline.prepare()

        if model_name:
            norm_path = self.config.paths.models / f"{model_name}_normalizer.joblib"
            if norm_path.exists():
                from utils.normalization import FeatureNormalizer

                saved = FeatureNormalizer.load(norm_path)
                # Re-normalize from the RAW featured splits with the saved
                # scaler — applying it to the already-normalized frame would
                # double-normalize and corrupt the agent's inputs.
                self.pipeline.apply_normalizer(saved)
                logger.info("Applied saved normalizer from %s", norm_path)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(
        self,
        model_path: Optional[str | Path] = None,
        model_name: Optional[str] = None,
        on: str = "test",
    ) -> PerformanceReport:
        """Execute the agent deterministically over the chosen split.

        Parameters
        ----------
        model_path:
            Path to the saved model; if omitted, uses ``models/<model_name>.zip``.
        model_name:
            Base name used to locate the saved normalizer/model.
        on:
            ``'test'`` (default, held-out 2025) or ``'train'`` (sanity check).
        """
        bcfg = self.config.backtest
        model_name = model_name or self.config.training.model_name
        if self.model is None:
            model_path = model_path or (self.config.paths.models / f"{model_name}.zip")
            self.load_model(model_path)

        self._ensure_data(model_name)

        env = self.pipeline.make_env(on, random_start=False)
        self.history = run_episode(self.model, env, deterministic=bcfg.deterministic)
        self.report = compute_report(
            equity_curve=self.history["equity_curve"],
            trades=self.history["trades"],
            initial_balance=self.history["initial_balance"],
            risk_free_rate=bcfg.risk_free_rate,
            bars_per_year=bcfg.bars_per_year,
        )
        logger.info("\n%s", self.report.pretty())
        return self.report

    # ------------------------------------------------------------------ #
    # Reporting / visualization (requirement #7)
    # ------------------------------------------------------------------ #
    def save_results(self, out_dir: Optional[str | Path] = None) -> Path:
        """Persist the metrics JSON, equity CSV and charts under ``logs/``."""
        if self.history is None or self.report is None:
            raise RuntimeError("Call run() before save_results().")
        out_dir = Path(out_dir or (self.config.paths.logs / "backtest"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Metrics JSON.
        (out_dir / "metrics.json").write_text(json.dumps(self.report.to_dict(), indent=2))

        # Equity curve CSV.
        eq_df = pd.DataFrame(
            {
                "time": self.history["timestamps"],
                "equity": self.history["equity_curve"],
            }
        )
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)

        # Trades CSV.
        if self.history["trades"]:
            pd.DataFrame(self.history["trades"]).to_csv(out_dir / "trades.csv", index=False)

        if self.config.backtest.save_plots:
            self._plot(out_dir)

        logger.info("Saved backtest results to %s", out_dir)
        return out_dir

    def _plot(self, out_dir: Path) -> None:
        """Generate equity-curve, trade and reward charts."""
        hist = self.history
        viz.plot_equity_curve(
            hist["equity_curve"],
            timestamps=hist["timestamps"],
            initial_balance=hist["initial_balance"],
            title="Out-of-sample Equity Curve (2025)",
            save_path=out_dir / "equity_curve.png",
        )
        viz.plot_reward_curve(
            hist["rewards"],
            title="Reward over Test Episode",
            save_path=out_dir / "reward_curve.png",
        )
        if hist["trades"]:
            price = pd.Series(
                self.pipeline.test_df["close"].values,
                index=pd.to_datetime(self.pipeline.test_df.index),
            )
            viz.plot_trades(
                price, hist["trades"],
                title="Trade Entries / Exits (2025)",
                save_path=out_dir / "trades.png",
            )
