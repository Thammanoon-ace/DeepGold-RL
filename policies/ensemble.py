"""
Ensemble policy for variance reduction (V3.5 / Phase 5A).

The decisive V3.5 finding was that a single PPO run's result is dominated by the
random seed (compounded −62%…+204%). The cheapest, highest-leverage mitigation
is to train K policies with different seeds and **average their action
probability distributions** at inference. If per-seed errors are partly
independent, the ensemble's decision variance falls roughly ~1/K, and behavior
is far more stable than any single member.

``EnsemblePolicy`` also implements a **confidence threshold** (Phase 5C): if the
ensemble's top mean action-probability does not exceed ``tau``, it returns Hold
— filtering low-conviction noise trades. With a single member it acts as a
confidence-gated single policy.

It exposes a ``predict(obs, deterministic=...)`` matching the Stable-Baselines3
interface, so it drops straight into ``run_episode`` / the backtester.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from env.gold_trading_env import ACTION_HOLD


class EnsemblePolicy:
    """Average the action distributions of several trained SB3 agents.

    Parameters
    ----------
    models:
        List of loaded SB3 algorithms (PPO/DQN) sharing the same observation and
        (Discrete) action space.
    confidence_threshold:
        If > 0, act only when the mean top action-probability exceeds this;
        otherwise Hold. 0 disables the gate (pure averaging).
    """

    def __init__(self, models: Sequence, confidence_threshold: float = 0.0) -> None:
        if not models:
            raise ValueError("EnsemblePolicy needs at least one model.")
        self.models = list(models)
        self.tau = float(confidence_threshold)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        paths: Sequence[str | Path],
        algo: str = "ppo",
        device: str = "auto",
        confidence_threshold: float = 0.0,
    ) -> "EnsemblePolicy":
        """Load K saved models (e.g. K seeds of the same architecture)."""
        from stable_baselines3 import DQN, PPO

        cls_ = DQN if algo.lower() == "dqn" else PPO
        models = [cls_.load(Path(p), device=device) for p in paths]
        return cls(models, confidence_threshold=confidence_threshold)

    # ------------------------------------------------------------------ #
    def _mean_probs(self, obs: np.ndarray) -> np.ndarray:
        """Mean categorical action-probabilities across members, shape (B, A).

        Works for both SB3 agents (via ``policy.get_distribution``) and
        GPU-trained ``ActorCritic`` agents (via their ``action_probs``), so the
        SB3 and CleanRL grid engines ensemble identically.
        """
        probs = None
        for m in self.models:
            if hasattr(m, "action_probs"):          # CleanRL ActorCritic
                p = m.action_probs(obs)
            else:                                   # SB3 PPO/DQN
                obs_t, _ = m.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    dist = m.policy.get_distribution(obs_t)
                    p = dist.distribution.probs.detach().cpu().numpy()  # (B, A)
            probs = p if probs is None else probs + p
        return probs / len(self.models)

    def predict(self, observation: np.ndarray, deterministic: bool = True, **kwargs):
        """SB3-compatible predict. Returns ``(action, None)``.

        ``deterministic`` is accepted for interface parity; the ensemble always
        acts on the mean distribution (argmax, with optional confidence gate),
        which is the stable choice for evaluation.
        """
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]

        mean_p = self._mean_probs(obs)             # (B, A)
        actions = mean_p.argmax(axis=1)
        if self.tau > 0:
            top = mean_p.max(axis=1)
            actions = np.where(top >= self.tau, actions, ACTION_HOLD)
        actions = actions.astype(np.int64)
        return (int(actions[0]) if single else actions), None

    # ------------------------------------------------------------------ #
    def agreement(self, observation: np.ndarray) -> float:
        """Fraction of members agreeing with the ensemble action (diagnostic).

        Low agreement => members disagree => the ensemble's value is highest
        there (it averages out the disagreement). Useful for analysing where
        seed variance concentrates.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        ens_action, _ = self.predict(obs[0])
        votes = []
        for m in self.models:
            a, _ = m.predict(obs[0], deterministic=True)
            votes.append(int(np.asarray(a).reshape(-1)[0]))
        return float(np.mean([v == ens_action for v in votes]))
