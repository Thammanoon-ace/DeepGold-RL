"""
V4 — Seed × fold grid evaluator for optimal execution.

Mirrors :class:`validation.grid.GridEvaluator` but the env is
:class:`env.execution_env.ExecutionGoldEnv` and the headline metric is
``bps_savings_vs_twap`` (median over a batch of test-set episodes per cell),
not ``return_pct``. The robustness + bootstrap-median-CI machinery is reused
unchanged — the V0–V3.5 epistemology (distribution, not point) carries over.

The TWAP baseline is computed by the env itself as a shadow during every
episode, so there's no separate "baseline run" — each cell's
``bps_savings_vs_twap`` is already the agent-minus-TWAP delta.

Headline judgment for V4: an experiment is "real" if both
* the **single-seed distribution's median CI excludes 0** (each cell beats
  TWAP on average more often than not), AND
* the **per-fold ensemble's median is positive** with the bulk of folds
  beating TWAP — i.e. the variance-reduced policy is also a winning policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config.config import Config
from env.execution_env import ExecutionGoldEnv
from policies.ensemble import EnsemblePolicy
from policies.factory import build_policy_kwargs
from training.train_ppo import _progress_bar_available, resolve_device
from utils.data_loader import HistoricalDataLoader
from utils.feature_engineering import FeatureEngineer
from utils.normalization import FeatureNormalizer
from validation.robustness import RobustnessReport, bootstrap_median_ci, compute_robustness
from validation.splitters import TimeSeriesSplitter

logger = logging.getLogger(__name__)


@dataclass
class ExecutionGridResult:
    cells: pd.DataFrame                       # one row per (seed, fold)
    single: RobustnessReport                  # distribution over single-cell median bps savings
    ensemble: Optional[RobustnessReport]      # distribution over per-fold ensemble median bps savings
    median_ci: tuple
    ensemble_cells: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lo, hi = self.median_ci
        out = [
            "V4 EXECUTION GRID — bps_savings_vs_twap distribution",
            "=" * 60,
            f"Cells (seed x fold): {len(self.cells)}",
            "",
            "Single-seed distribution (median bps savings per cell):",
            self.single.pretty(),
            f"  Median 95% CI    : [{lo:+.2f}, {hi:+.2f}] bps  "
            f"({'straddles 0 -> RL not significantly beating TWAP' if lo < 0 < hi else 'excludes 0'})",
        ]
        if self.ensemble is not None:
            out += ["", "Ensemble (per fold) distribution:", self.ensemble.pretty()]
        return "\n".join(out)


class ExecutionGridEvaluator:
    """Walk-forward grid runner specialised for the execution task.

    Each cell trains one SB3 PPO on the fold's training slice (with random
    episode starts inside it) and evaluates ``n_eval_episodes`` deterministic
    episodes on the fold's test slice. Per-cell metrics are the medians /
    averages of ``bps_savings_vs_twap`` across those evaluation episodes.
    """

    def __init__(
        self,
        config: Config,
        seeds: Sequence[int] = (0, 1, 2),
        splitter: Optional[TimeSeriesSplitter] = None,
        timesteps_per_fold: int = 80_000,
        num_envs: int = 4,
        n_eval_episodes: int = 200,
        normalizer_method: str = "robust",
        deadline_range: tuple = (32, 128),
        target_lots_range: tuple = (0.5, 2.0),
        fixed_cost_bps: float = 1.0,
        impact_bps_per_lot: float = 10.0,
        evaluate_ensemble: bool = True,
        run_tag: str = "exec_grid",
    ) -> None:
        self.config = config
        self.seeds = list(seeds)
        self.splitter = splitter or TimeSeriesSplitter(
            n_splits=5, mode="expanding", gap=config.env.window_size)
        self.timesteps_per_fold = timesteps_per_fold
        self.num_envs = num_envs
        self.n_eval_episodes = n_eval_episodes
        self.normalizer_method = normalizer_method
        self.deadline_range = deadline_range
        self.target_lots_range = target_lots_range
        self.fixed_cost_bps = fixed_cost_bps
        self.impact_bps_per_lot = impact_bps_per_lot
        self.evaluate_ensemble = evaluate_ensemble
        self.run_tag = run_tag
        self.device = resolve_device(config.training.device)
        self.loader = HistoricalDataLoader(config.data)
        self.engineer = FeatureEngineer(config.features)
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------ #
    def _make_train_env(self, features: np.ndarray, prices: np.ndarray, seed: int):
        return ExecutionGoldEnv(
            features, prices, self.config.env,
            deadline_range=self.deadline_range,
            target_lots_range=self.target_lots_range,
            fixed_cost_bps=self.fixed_cost_bps,
            impact_bps_per_lot=self.impact_bps_per_lot,
            random_start=True, seed=seed,
        )

    def _train(self, train_df: pd.DataFrame, feat_cols: List[str], seed: int) -> PPO:
        tcfg = self.config.training
        features = train_df[feat_cols].to_numpy(dtype=np.float32)
        prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)

        # N independent copies of the env, each with a different RNG offset so
        # rollouts don't lockstep into the same episodes.
        def make_factory(env_idx: int):
            return lambda: self._make_train_env(features, prices, seed * 1000 + env_idx)
        vec = VecMonitor(DummyVecEnv([make_factory(i) for i in range(self.num_envs)]))

        model = PPO(
            policy="MlpPolicy", env=vec,
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

    # ------------------------------------------------------------------ #
    def _evaluate(self, model_or_ens, test_df: pd.DataFrame,
                  feat_cols: List[str], eval_seed: int) -> dict:
        features = test_df[feat_cols].to_numpy(dtype=np.float32)
        prices = test_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
        env = ExecutionGoldEnv(
            features, prices, self.config.env,
            deadline_range=self.deadline_range,
            target_lots_range=self.target_lots_range,
            fixed_cost_bps=self.fixed_cost_bps,
            impact_bps_per_lot=self.impact_bps_per_lot,
            random_start=True, seed=eval_seed,
        )
        savings, sf, twap_sf = [], [], []
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=eval_seed + ep)
            done = False
            info: dict = {}
            while not done:
                action, _ = model_or_ens.predict(obs, deterministic=True)
                obs, _r, term, trunc, info = env.step(int(action))
                done = term or trunc
            if "bps_savings_vs_twap" not in info:
                continue
            savings.append(info["bps_savings_vs_twap"])
            sf.append(info["shortfall_bps"])
            twap_sf.append(info["twap_shortfall_bps"])
        savings = np.asarray(savings)
        sf = np.asarray(sf)
        twap_sf = np.asarray(twap_sf)
        return {
            "median_savings_bps": float(np.median(savings)),
            "mean_savings_bps": float(np.mean(savings)),
            "win_rate_vs_twap": float(np.mean(savings > 0)),
            "agent_sf_median_bps": float(np.median(sf)),
            "twap_sf_median_bps": float(np.median(twap_sf)),
            "n_episodes": int(len(savings)),
        }

    # ------------------------------------------------------------------ #
    def run(self, csv_path=None) -> ExecutionGridResult:
        raw = self.loader.load(csv_path)
        featured = self.engineer.transform(raw)
        self.feature_columns = self.engineer.feature_columns
        folds = self.splitter.split(len(featured))
        logger.info("V4 Grid: %d folds x %d seeds = %d cells | %d steps/cell "
                    "| %d train envs | %d eval episodes/cell",
                    len(folds), len(self.seeds), len(folds) * len(self.seeds),
                    self.timesteps_per_fold, self.num_envs, self.n_eval_episodes)

        cells: List[dict] = []
        ens_cells: List[dict] = []
        for fold in folds:
            train_raw = featured.iloc[fold.train_start:fold.train_end]
            test_raw = featured.iloc[fold.test_start:fold.test_end]
            normalizer = FeatureNormalizer(self.feature_columns, self.normalizer_method)
            train_df = normalizer.fit_transform(train_raw)
            test_df = normalizer.transform(test_raw)

            seed_models: List[PPO] = []
            for seed in self.seeds:
                model = self._train(train_df, self.feature_columns, seed)
                m = self._evaluate(model, test_df, self.feature_columns,
                                   eval_seed=10_000 * (fold.index + 1) + seed)
                cells.append({"fold": fold.index, "seed": seed, **m})
                seed_models.append(model)
                logger.info("  fold %d seed %d -> median %+0.2f bps  win_rate %.1f%%",
                            fold.index, seed, m["median_savings_bps"],
                            m["win_rate_vs_twap"] * 100)

            if self.evaluate_ensemble and len(seed_models) > 1:
                ens = EnsemblePolicy(seed_models)
                em = self._evaluate(ens, test_df, self.feature_columns,
                                    eval_seed=20_000 * (fold.index + 1))
                ens_cells.append({"fold": fold.index, **em})
                logger.info("  fold %d ENSEMBLE -> median %+0.2f bps  win_rate %.1f%%",
                            fold.index, em["median_savings_bps"],
                            em["win_rate_vs_twap"] * 100)

        cells_df = pd.DataFrame(cells)
        ens_df = pd.DataFrame(ens_cells)
        single_savings = cells_df["median_savings_bps"].to_numpy()
        # Baseline here is 0 bps (TWAP shadow built-in); "beats baseline" = beats TWAP.
        single = compute_robustness(single_savings, baseline=0.0)
        ci = bootstrap_median_ci(single_savings)
        ensemble = (compute_robustness(ens_df["median_savings_bps"].to_numpy(),
                                       baseline=0.0)
                    if not ens_df.empty else None)
        result = ExecutionGridResult(cells_df, single, ensemble, ci, ens_df)
        logger.info("\n%s", result.summary())
        return result
