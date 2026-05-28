"""
PPO training pipeline (requirements #5/#12).

``PPOTrainer`` wires the data pipeline, vectorized environments, GPU selection,
checkpointing, TensorBoard logging and resume support into a single object.

Memory / GPU notes
------------------
* ``device='auto'`` selects CUDA when available; the MLP policy used here is
  tiny, so even a 4 GB card (e.g. GTX 1650 Ti) has ample headroom, and a larger
  card (RTX 3070) is comfortably oversized.
* Vectorized environments (``DummyVecEnv``/``SubprocVecEnv``) parallelize
  rollout collection; the observation is a modest flat float32 vector, keeping
  the replay footprint small.
* ``n_steps`` * ``n_envs`` controls the rollout buffer size — the main memory
  knob.  Defaults are conservative.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from config.config import Config
from env.env_builder import TradingDataPipeline
from training.callbacks import TradingMetricsCallback

logger = logging.getLogger(__name__)


def _progress_bar_available() -> bool:
    """SB3's progress bar needs both ``tqdm`` and ``rich``; degrade gracefully."""
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("tqdm", "rich"))


def resolve_device(requested: str = "auto") -> str:
    """Resolve the torch device string, preferring CUDA when available."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


class PPOTrainer:
    """Train, checkpoint and persist a PPO agent on the gold environment.

    Parameters
    ----------
    config:
        The full :class:`~config.config.Config`.
    pipeline:
        An optional pre-built :class:`TradingDataPipeline`.  If omitted, one is
        created and ``prepare()`` is called lazily.
    """

    def __init__(self, config: Config, pipeline: Optional[TradingDataPipeline] = None) -> None:
        self.config = config
        self.pipeline = pipeline or TradingDataPipeline(config)
        self.model: Optional[PPO] = None
        self.vec_env: Optional[VecNormalize] = None
        self.device = resolve_device(config.training.device)

        config.paths.ensure()
        self._models_dir = config.paths.models
        self._tb_dir = config.paths.logs / config.training.tensorboard_subdir
        self._ckpt_dir = config.paths.models / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Environment construction
    # ------------------------------------------------------------------ #
    def _build_vec_env(self, use_subproc: bool = False) -> VecNormalize:
        """Create the vectorized, reward-normalized training environment.

        We wrap the (already feature-normalized) env in ``VecNormalize`` with
        ``norm_obs=False`` (features are pre-scaled) and ``norm_reward=True``
        to stabilize PPO's value-function targets.
        """
        tcfg = self.config.training
        if self.pipeline.train_df is None:
            self.pipeline.prepare()

        def _make(rank: int):
            def _init():
                env = self.pipeline.make_env("train", random_start=True)
                # Monitor records episode reward/length for SB3 + callbacks.
                return Monitor(env)
            return _init

        env_fns = [_make(i) for i in range(tcfg.n_envs)]
        vec_cls = SubprocVecEnv if (use_subproc and tcfg.n_envs > 1) else DummyVecEnv
        vec = vec_cls(env_fns)
        # Reward normalization helps PPO's value targets, but stale running stats
        # interact badly with an off-policy replay buffer, so it is off for DQN.
        norm_reward = tcfg.algo.lower() == "ppo"
        vec = VecNormalize(
            vec, norm_obs=False, norm_reward=norm_reward, clip_reward=10.0, gamma=tcfg.gamma
        )
        self.vec_env = vec
        return vec

    def _build_eval_env(self) -> Optional[VecNormalize]:
        """Build a deterministic, single-env evaluation wrapper (test data)."""
        if self.pipeline.test_df is None or self.pipeline.test_df.empty:
            return None
        eval_vec = DummyVecEnv(
            [lambda: Monitor(self.pipeline.make_env("test", random_start=False))]
        )
        # Share reward-normalization statistics; do not update them at eval time.
        eval_vec = VecNormalize(
            eval_vec, norm_obs=False, norm_reward=False, training=False
        )
        return eval_vec

    # ------------------------------------------------------------------ #
    # Model construction / training
    # ------------------------------------------------------------------ #
    def build_model(self, resume_from: Optional[str | Path] = None) -> PPO:
        """Create a fresh PPO model or resume from a saved checkpoint.

        Resume support (requirement #5): pass ``resume_from`` pointing at a
        ``.zip`` produced by SB3; training continues with its weights and
        optimizer state.
        """
        tcfg = self.config.training
        vec = self.vec_env or self._build_vec_env()

        algo_cls = DQN if tcfg.algo.lower() == "dqn" else PPO

        if resume_from is not None:
            resume_from = Path(resume_from)
            logger.info("Resuming %s training from %s", tcfg.algo.upper(), resume_from)
            self.model = algo_cls.load(
                resume_from, env=vec, device=self.device,
                tensorboard_log=str(self._tb_dir),
            )
            # Reload matching VecNormalize stats if present.
            stats = resume_from.with_name(resume_from.stem + "_vecnormalize.pkl")
            if stats.exists():
                self.vec_env = VecNormalize.load(str(stats), vec.venv)
                self.model.set_env(self.vec_env)
            return self.model

        from policies.factory import build_policy_kwargs

        policy_kwargs = build_policy_kwargs(tcfg, self.config.env.window_size)
        logger.info("Algo=%s | policy architecture=%s", tcfg.algo.upper(), tcfg.policy_arch)

        if algo_cls is DQN:
            self.model = DQN(
                policy=tcfg.policy,
                env=vec,
                learning_rate=tcfg.learning_rate,
                buffer_size=tcfg.dqn_buffer_size,
                learning_starts=tcfg.dqn_learning_starts,
                batch_size=tcfg.batch_size,
                gamma=tcfg.gamma,
                train_freq=tcfg.dqn_train_freq,
                target_update_interval=tcfg.dqn_target_update_interval,
                exploration_fraction=tcfg.dqn_exploration_fraction,
                exploration_final_eps=tcfg.dqn_exploration_final_eps,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(self._tb_dir),
                device=self.device,
                seed=tcfg.seed,
                verbose=1,
            )
        else:
            self.model = PPO(
                policy=tcfg.policy,
                env=vec,
                learning_rate=tcfg.learning_rate,
                n_steps=tcfg.n_steps,
                batch_size=tcfg.batch_size,
                n_epochs=tcfg.n_epochs,
                gamma=tcfg.gamma,
                gae_lambda=tcfg.gae_lambda,
                clip_range=tcfg.clip_range,
                ent_coef=tcfg.ent_coef,
                vf_coef=tcfg.vf_coef,
                max_grad_norm=tcfg.max_grad_norm,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(self._tb_dir),
                device=self.device,
                seed=tcfg.seed,
                verbose=1,
            )
        logger.info("Built %s model on device=%s", tcfg.algo.upper(), self.device)
        return self.model

    def train(
        self,
        total_timesteps: Optional[int] = None,
        resume_from: Optional[str | Path] = None,
    ) -> PPO:
        """Run the full training loop with checkpoints and evaluation."""
        tcfg = self.config.training
        total_timesteps = total_timesteps or tcfg.total_timesteps

        if self.model is None:
            self.build_model(resume_from=resume_from)

        # ---- Callbacks ------------------------------------------------- #
        # CheckpointCallback frequency is counted in *per-env* steps.
        ckpt_cb = CheckpointCallback(
            save_freq=max(tcfg.checkpoint_freq // max(tcfg.n_envs, 1), 1),
            save_path=str(self._ckpt_dir),
            name_prefix=tcfg.model_name,
            save_vecnormalize=True,
            save_replay_buffer=False,
        )
        callbacks = [ckpt_cb, TradingMetricsCallback(log_freq=1000)]

        eval_env = self._build_eval_env()
        if eval_env is not None:
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=str(self._models_dir / "best_model"),
                log_path=str(self.config.paths.logs / "eval"),
                eval_freq=max(tcfg.eval_freq // max(tcfg.n_envs, 1), 1),
                n_eval_episodes=1,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_cb)

        logger.info(
            "Starting training: %d timesteps, %d env(s), device=%s",
            total_timesteps, tcfg.n_envs, self.device,
        )
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=tcfg.model_name,
            reset_num_timesteps=resume_from is None,
            progress_bar=_progress_bar_available(),
        )
        return self.model

    # ------------------------------------------------------------------ #
    # Persistence (requirement #8)
    # ------------------------------------------------------------------ #
    def save(self, name: Optional[str] = None) -> Path:
        """Save the model, VecNormalize stats and the feature normalizer.

        Everything needed to reproduce inference (the agent, reward-norm stats
        and the feature scaler) is written under ``models/``.
        """
        if self.model is None:
            raise RuntimeError("No model to save; train() first.")
        name = name or self.config.training.model_name
        model_path = self._models_dir / f"{name}.zip"
        self.model.save(model_path)

        if self.vec_env is not None:
            self.vec_env.save(str(self._models_dir / f"{name}_vecnormalize.pkl"))
        # Persist the feature scaler used to build observations.
        self.pipeline.save_normalizer(self._models_dir / f"{name}_normalizer.joblib")

        logger.info("Saved model + artefacts to %s", self._models_dir)
        return model_path
