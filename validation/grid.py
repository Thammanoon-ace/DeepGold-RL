"""
Seed x Fold grid evaluator — the V3.5 experimental protocol, operationalized.

This is the keystone that ties the V3.5 foundations together:

* trains with the fast **vectorized env** (Phase 5E),
* evaluates each cell deterministically with the **scalar env + run_episode**
  (single OOS path, honest),
* fits the normalizer (and optional correlation filter) **per fold on train**,
* compares the **single-seed distribution** against a **seed-ensemble**
  (Phase 5A) and a **buy-and-hold baseline**,
* summarizes everything as a **distribution + Robustness Score + bootstrap CI**
  (never a point metric).

This is how every architecture/feature/timeframe claim must now be judged: a
config is only "better" if it improves the *distribution over the grid* relative
to baselines — not if one lucky run looked good.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from backtest.backtester import run_episode, run_episode_ter_gated
from backtest.baselines import buy_and_hold_return
from backtest.metrics import compute_report
from config.config import Config
from env.env_builder import make_env_from_frame
from env.vectorized_env import VectorizedGoldTradingEnv
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
class GridResult:
    """Outcome of a seed x fold grid run."""

    cells: pd.DataFrame                      # one row per (seed, fold)
    single: RobustnessReport                 # distribution over all cells
    ensemble: Optional[RobustnessReport]     # distribution over per-fold ensembles
    baseline_buy_hold_pct: float             # mean buy-and-hold over test folds
    median_ci: tuple                         # bootstrap CI of single-cell median
    ensemble_cells: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lo, hi = self.median_ci
        out = [
            "GRID EVALUATION SUMMARY",
            "=" * 60,
            f"Cells (seed x fold): {len(self.cells)}",
            f"Buy-and-hold (mean over folds): {self.baseline_buy_hold_pct:+.2f}%",
            "",
            "Single-seed distribution:",
            self.single.pretty(),
            f"  Median 95% CI    : [{lo:+.1f}, {hi:+.1f}]  "
            f"({'straddles 0 -> not significant' if lo < 0 < hi else 'excludes 0'})",
        ]
        if self.ensemble is not None:
            out += ["", "Ensemble (per fold) distribution:", self.ensemble.pretty(),
                    f"  Variance cut: single std {self.single.std:.1f} -> "
                    f"ensemble std {self.ensemble.std:.1f}"]
        return "\n".join(out)


class GridEvaluator:
    """Run a seed x fold grid with vectorized-env training.

    Parameters
    ----------
    config: full Config.
    seeds: list of training seeds (>=8 recommended for real runs).
    splitter: TimeSeriesSplitter (>=5 folds recommended).
    timesteps_per_fold: PPO steps per (seed, fold) cell.
    num_envs: vectorized lanes used for training (32-64 is efficient).
    evaluate_ensemble: also evaluate the per-fold seed-ensemble.
    """

    def __init__(
        self,
        config: Config,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        splitter: Optional[TimeSeriesSplitter] = None,
        timesteps_per_fold: int = 50_000,
        num_envs: int = 32,
        normalizer_method: str = "robust",
        evaluate_ensemble: bool = True,
        resample_to: Optional[str] = None,
        run_tag: str = "grid",
        tensorboard: bool = True,
        engine: str = "sb3",
        gpu_n_steps: int = 128,
        val_frac: float = 0.0,
    ) -> None:
        self.config = config
        self.seeds = list(seeds)
        self.run_tag = run_tag
        # Training engine: "sb3" (CPU numpy VecEnv + SB3 PPO) or "gpu"
        # (GPU-resident TorchVecGoldEnv + CleanRL-style PPO). Both evaluate
        # identically (scalar env + run_episode) so results stay comparable.
        self.engine = engine
        self.gpu_n_steps = gpu_n_steps
        # TensorBoard: one run per (fold, seed) cell -> live training curves.
        self._tb_log = str(config.paths.logs / "tensorboard" / "grid") if tensorboard else None
        self.splitter = splitter or TimeSeriesSplitter(
            n_splits=5, mode="expanding", gap=config.env.window_size)
        self.timesteps_per_fold = timesteps_per_fold
        self.num_envs = num_envs
        self.normalizer_method = normalizer_method
        self.evaluate_ensemble = evaluate_ensemble
        # Held-out validation slice for non-leaking seed ranking (top-k experiment).
        # When > 0, the last ``val_frac`` of each fold's training data is held out:
        # the agent trains on the first ``1 - val_frac`` only, the normalizer is
        # fit on the training slice only (no val leakage), and a deterministic
        # eval on the validation slice produces a ``val_sharpe`` column in
        # cells.csv. Rank seeds by val_sharpe and ensemble top-k. 0.0 = disabled.
        self.val_frac = float(val_frac)
        # Phase 5F: resample the native (M5) CSV to a higher timeframe (e.g. H1)
        # before feature engineering — to test the "M5 is too noisy" hypothesis.
        self.resample_to = resample_to
        from config.config import BARS_PER_YEAR
        effective_tf = resample_to or config.data.timeframe
        self._bars_per_year = BARS_PER_YEAR.get(effective_tf, config.backtest.bars_per_year)
        self.device = resolve_device(config.training.device)
        self.loader = HistoricalDataLoader(config.data)
        self.engineer = FeatureEngineer(config.features)
        self.feature_columns: List[str] = []

    # ------------------------------------------------------------------ #
    def _train(self, train_df: pd.DataFrame, feat_cols: List[str], seed: int,
               tb_name: Optional[str] = None) -> PPO:
        tcfg = self.config.training
        features = train_df[feat_cols].to_numpy(dtype=np.float32)
        prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
        vec = VectorizedGoldTradingEnv(features, prices, self.config.env,
                                       num_envs=self.num_envs, random_start=True, seed=seed)
        vec = VecMonitor(vec)  # tracks episode rewards -> rollout/ep_rew_mean in TB
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                           gamma=tcfg.gamma)
        model = PPO(
            policy=tcfg.policy, env=vec,
            learning_rate=tcfg.learning_rate, n_steps=tcfg.n_steps,
            batch_size=tcfg.batch_size, n_epochs=tcfg.n_epochs,
            gamma=tcfg.gamma, gae_lambda=tcfg.gae_lambda, clip_range=tcfg.clip_range,
            ent_coef=tcfg.ent_coef, vf_coef=tcfg.vf_coef, max_grad_norm=tcfg.max_grad_norm,
            policy_kwargs=build_policy_kwargs(tcfg, self.config.env.window_size),
            device=self.device, seed=seed, verbose=0,
            tensorboard_log=self._tb_log,
        )
        model.learn(total_timesteps=self.timesteps_per_fold,
                    tb_log_name=tb_name or "cell",
                    progress_bar=_progress_bar_available())
        return model

    def _train_gpu(self, train_df: pd.DataFrame, feat_cols: List[str], seed: int):
        """Train one cell with the GPU-resident env + CleanRL-style PPO.

        Drop-in alternative to :meth:`_train`: returns an ``ActorCritic`` whose
        SB3-compatible ``predict`` makes :meth:`_evaluate` (scalar env) work
        unchanged, so GPU-engine results are directly comparable to SB3 ones.
        Only ``cnn``/``mlp`` archs are supported here (the CleanRL ActorCritic).
        """
        import torch
        from env.torch_vec_env import TorchVecGoldEnv
        from training.cleanrl_ppo import PPOConfig, train_cleanrl_ppo

        tcfg = self.config.training
        arch = tcfg.policy_arch if tcfg.policy_arch in ("cnn", "mlp") else "cnn"
        features = train_df[feat_cols].to_numpy(dtype=np.float32)
        prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
        env = TorchVecGoldEnv(features, prices, self.config.env, num_envs=self.num_envs,
                              random_start=True, device=self.device,
                              dtype=torch.float32, seed=seed)
        ppo = PPOConfig(total_timesteps=self.timesteps_per_fold, n_steps=self.gpu_n_steps,
                        learning_rate=tcfg.learning_rate, gamma=tcfg.gamma,
                        gae_lambda=tcfg.gae_lambda, clip_range=tcfg.clip_range,
                        ent_coef=tcfg.ent_coef, vf_coef=tcfg.vf_coef,
                        max_grad_norm=tcfg.max_grad_norm)
        ac, _ = train_cleanrl_ppo(env, arch=arch, ppo=ppo, seed=seed)
        return ac

    def _evaluate(self, model, test_df: pd.DataFrame, feat_cols: List[str]) -> dict:
        # Note (2026-05-28): the Tier 2.1 GPU-vec eval path
        # (``backtest.run_episode_torch_vec``) is **correctness-equivalent** to
        # the scalar path (drift < 0.001% on the real M5 test split, exact at
        # CPU fp64; see ``scripts/_test_torch_eval_equiv.py``) but turned out
        # to be **slower** in practice — single-seed eval ran 27.8s scalar vs
        # 120.4s torch_vec; the ensemble of 8 was 195s vs 256s
        # (``scripts/_bench_torch_vec_eval.py`` for the rerun script). Reason:
        # TorchVecGoldEnv is built for batched lanes (num_envs >> 1); with
        # num_envs=1 each step still pays the batched-tensor + kernel-launch
        # overhead that the scalar GoldTradingEnv (pure NumPy float ops)
        # skips. The grid keeps the scalar eval as the default path.
        env = make_env_from_frame(test_df, feat_cols, self.config.env, random_start=False)
        bcfg = self.config.backtest
        if bcfg.ter_gate_window > 0 and bcfg.ter_gate_threshold > 0:
            hist = run_episode_ter_gated(
                model, env,
                ter_window=bcfg.ter_gate_window,
                ter_threshold=bcfg.ter_gate_threshold,
                deterministic=True,
            )
        else:
            hist = run_episode(model, env, deterministic=True)
        rep = compute_report(hist["equity_curve"], hist["trades"], hist["initial_balance"],
                             bars_per_year=self._bars_per_year)
        return {"return_pct": rep.total_return_pct, "sharpe": rep.sharpe_ratio,
                "max_dd_pct": rep.max_drawdown_pct, "n_trades": rep.n_trades}

    # ------------------------------------------------------------------ #
    def run(self, csv_path=None) -> GridResult:
        raw = self.loader.load(csv_path)
        if self.resample_to and self.resample_to != self.config.data.timeframe:
            n0 = len(raw)
            raw = self.loader.resample(raw, self.resample_to)
            logger.info("Resampled %s -> %s: %d -> %d bars",
                        self.config.data.timeframe, self.resample_to, n0, len(raw))
        featured = self.engineer.transform(raw)
        self.feature_columns = self.engineer.feature_columns
        folds = self.splitter.split(len(featured))
        logger.info("Grid[%s]: tf=%s | %d folds x %d seeds = %d cells | %d steps/cell | %d lanes",
                    self.engine, self.resample_to or self.config.data.timeframe,
                    len(folds), len(self.seeds), len(folds) * len(self.seeds),
                    self.timesteps_per_fold, self.num_envs)

        cells: List[dict] = []
        ens_cells: List[dict] = []
        baseline_bh: List[float] = []
        thr = self.config.features.correlation_threshold

        for fold in folds:
            train_raw_full = featured.iloc[fold.train_start:fold.train_end]
            test_raw = featured.iloc[fold.test_start:fold.test_end]

            # Held-out validation split: hold out the LAST val_frac of each
            # fold's training data and fit the normalizer on the earlier slice
            # only. The agent never sees the validation slice during training;
            # its deterministic return there is a leakage-free predictor of
            # test performance (for the [[topk-ranker]] experiment).
            if self.val_frac > 0:
                split_pt = max(1, int((1 - self.val_frac) * len(train_raw_full)))
                train_raw = train_raw_full.iloc[:split_pt]
                val_raw = train_raw_full.iloc[split_pt:]
            else:
                train_raw = train_raw_full
                val_raw = None

            fold_cols = self.feature_columns
            if thr and thr > 0:
                from utils.feature_selection import FeatureSelector
                fold_cols = FeatureSelector(correlation_threshold=thr).fit(
                    train_raw, self.feature_columns).kept_columns

            normalizer = FeatureNormalizer(fold_cols, self.normalizer_method)
            train_df = normalizer.fit_transform(train_raw)
            test_df = normalizer.transform(test_raw)
            val_df = normalizer.transform(val_raw) if val_raw is not None else None
            baseline_bh.append(buy_and_hold_return(test_raw["close"].to_numpy())["total_return_pct"])

            seed_models = []
            tf = self.resample_to or self.config.data.timeframe
            for seed in self.seeds:
                if self.engine == "gpu":
                    model = self._train_gpu(train_df, fold_cols, seed)
                else:
                    tb_name = f"{self.run_tag}_{tf}_{self.config.training.policy_arch}_f{fold.index}_s{seed}"
                    model = self._train(train_df, fold_cols, seed, tb_name=tb_name)
                m = self._evaluate(model, test_df, fold_cols)
                # Surface training-time Sharpe when available (gpu engine only)
                # so cells.csv can rank seeds *without test leakage*.
                train_sharpe = float(getattr(model, "_train_sharpe", float("nan")))
                # Held-out validation Sharpe: deterministic eval on a slice the
                # agent never trained on. Non-leaking predictor of test perf
                # ([[held-out-val-ranker]] experiment, follow-up to
                # [[train-sharpe-ranker-failed]]).
                if val_df is not None:
                    vm = self._evaluate(model, val_df, fold_cols)
                    val_sharpe = float(vm["sharpe"])
                    val_return = float(vm["return_pct"])
                else:
                    val_sharpe = float("nan")
                    val_return = float("nan")
                cells.append({"fold": fold.index, "seed": seed,
                              "train_sharpe": train_sharpe,
                              "val_sharpe": val_sharpe,
                              "val_return_pct": val_return, **m})
                seed_models.append(model)
                logger.info("  fold %d seed %d -> %+.2f%%", fold.index, seed, m["return_pct"])

            if self.evaluate_ensemble and len(seed_models) > 1:
                ens = EnsemblePolicy(seed_models)
                em = self._evaluate(ens, test_df, fold_cols)
                ens_cells.append({"fold": fold.index, **em})
                logger.info("  fold %d ENSEMBLE -> %+.2f%%", fold.index, em["return_pct"])

        cells_df = pd.DataFrame(cells)
        ens_df = pd.DataFrame(ens_cells)
        single_returns = cells_df["return_pct"].to_numpy()
        baseline_mean = float(np.mean(baseline_bh)) if baseline_bh else 0.0

        single = compute_robustness(single_returns, baseline=baseline_mean)
        ci = bootstrap_median_ci(single_returns)
        ensemble = (compute_robustness(ens_df["return_pct"].to_numpy(), baseline=baseline_mean)
                    if not ens_df.empty else None)

        result = GridResult(cells_df, single, ensemble, baseline_mean, ci, ens_df)
        logger.info("\n%s", result.summary())
        return result
