"""
Live-trading orchestration skeleton (requirement #9).

``LiveTrader`` closes the loop:

    fetch latest bars  ->  engineer + normalize features (training-identical)
                       ->  agent.predict()  ->  risk checks  ->  MT5Bridge order

It deliberately reuses the *exact* :class:`FeatureEngineer` and the *saved*
:class:`FeatureNormalizer` from training, so the observation the live agent sees
is built the same way as in training/backtesting — eliminating train/serve skew.

This is an architecture template, not a turnkey money machine.  The master
switch ``LiveConfig.enabled`` and ``LiveConfig.dry_run`` default to safe values,
and independent risk guards cap position count and size (requirement #13).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from config.config import Config
from env.gold_trading_env import ACTION_BUY, ACTION_CLOSE, ACTION_NAMES, ACTION_SELL
from live_trading.mt5_bridge import MT5Bridge
from training.cleanrl_ppo import ActorCritic
from utils.feature_engineering import FeatureEngineer
from utils.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)


class LiveTrader:
    """Drive a trained PPO agent against a live MT5 account.

    Parameters
    ----------
    config:
        The full :class:`~config.config.Config`.
    model_name:
        Base name used to locate ``<model_name>.zip`` and the normalizer.
    """

    def __init__(self, config: Config, model_name: Optional[str] = None) -> None:
        self.config = config
        self.model_name = model_name or config.training.model_name
        self.bridge = MT5Bridge(config.live)
        self.engineer = FeatureEngineer(config.features)
        self.model: Optional[Any] = None  # PPO | ActorCritic — both expose predict()
        self.normalizer: Optional[FeatureNormalizer] = None
        self.meta: Optional[dict] = None  # populated when loading a `.pt` (CleanRL)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def load_artifacts(self) -> None:
        """Load the trained model and the training-time feature normalizer.

        Detects the model format by extension:

        * ``<name>.zip`` — Stable-Baselines3 PPO (the original path).
        * ``<name>.pt`` — GPU CleanRL ``ActorCritic`` checkpoint produced by
          :mod:`training.cleanrl_ppo` (the H4 + cosine + SWA stack). Requires a
          sibling ``<name>_meta.json`` that records ``obs_dim``, ``window``,
          ``n_features`` and ``arch`` so we can reconstruct the network before
          loading the ``state_dict``. ``ActorCritic.predict(obs, deterministic=
          True)`` is signature-compatible with SB3 PPO, so the rest of the live
          loop is unchanged.

        ``.pt`` takes precedence when both files exist.
        """
        models = self.config.paths.models
        pt_path = models / f"{self.model_name}.pt"
        zip_path = models / f"{self.model_name}.zip"
        norm_path = models / f"{self.model_name}_normalizer.joblib"
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalizer not found: {norm_path}")

        if pt_path.exists():
            # CleanRL ActorCritic (H4 + cosine + SWA stack; see scripts/v4_smoke.py).
            meta_path = models / f"{self.model_name}_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Found {pt_path.name} but no sibling {meta_path.name}; "
                    "rerun scripts/v4_smoke.py or equivalent to regenerate metadata.")
            self.meta = json.loads(meta_path.read_text())
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
            ac = ActorCritic(ckpt["obs_dim"], ckpt["window"], ckpt["n_features"],
                             arch=ckpt.get("arch", "cnn")).to(device)
            ac.load_state_dict(ckpt["state_dict"])
            ac.eval()
            self.model = ac
            self.normalizer = FeatureNormalizer.load(norm_path)
            logger.info("Loaded CleanRL ActorCritic ('%s.pt') with %d params.",
                        self.model_name,
                        sum(p.numel() for p in ac.parameters()))
        elif zip_path.exists():
            # Legacy Stable-Baselines3 PPO path.
            self.model = PPO.load(zip_path, device="auto")
            self.normalizer = FeatureNormalizer.load(norm_path)
            logger.info("Loaded SB3 PPO ('%s.zip').", self.model_name)
        else:
            raise FileNotFoundError(
                f"No model found for '{self.model_name}': tried "
                f"{pt_path.name} and {zip_path.name} under {models}/.")

    def connect(self, **credentials) -> bool:
        """Connect to the MT5 terminal (see :meth:`MT5Bridge.connect`)."""
        return self.bridge.connect(**credentials)

    # ------------------------------------------------------------------ #
    # Observation construction (mirrors training exactly)
    # ------------------------------------------------------------------ #
    def _build_observation(self, bars: pd.DataFrame, account_state: np.ndarray) -> np.ndarray:
        """Recreate the training observation from the latest bars + account.

        Uses the same indicators and the same fitted scaler as training, then
        appends the live account-state vector in the identical layout used by
        :meth:`GoldTradingEnv._get_observation`.
        """
        featured = self.engineer.transform(bars)
        feat_cols = self.engineer.feature_columns
        normalized = self.normalizer.transform(featured)

        window = self.config.env.window_size
        window_block = normalized[feat_cols].to_numpy(dtype=np.float32)[-window:]
        if len(window_block) < window:
            raise RuntimeError(
                f"Need at least {window} feature bars after warm-up; got "
                f"{len(window_block)}. Fetch more bars."
            )
        flat = window_block.flatten()
        obs = np.concatenate([flat, account_state.astype(np.float32)])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _current_account_state(self) -> tuple[np.ndarray, list]:
        """Build the 5-element account-state vector from live MT5 data."""
        acct = self.bridge.get_account_info()
        positions = self.bridge.get_positions()
        init_bal = self.config.env.initial_balance

        direction, lots, unrealized = 0.0, 0.0, 0.0
        if positions:
            p = positions[0]
            direction = 1.0 if p["type"] == 0 else -1.0  # 0 == BUY in MT5
            lots = p["volume"]
            unrealized = p.get("profit", 0.0)

        state = np.array(
            [
                direction,
                float(np.clip(unrealized / init_bal, -1.0, 1.0)),
                lots / max(self.config.live.max_lots, 1e-9),
                float(np.clip(acct["equity"] / init_bal - 1.0, -1.0, 1.0)),
                0.0,  # holding-age unknown live; conservatively 0
            ],
            dtype=np.float32,
        )
        return state, positions

    # ------------------------------------------------------------------ #
    # Risk guards (independent of the environment)
    # ------------------------------------------------------------------ #
    def _passes_risk_checks(self, positions: list) -> bool:
        if len(positions) >= self.config.live.max_open_positions:
            logger.info("Risk guard: max open positions reached; new entries blocked.")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def step_once(self) -> None:
        """Run a single decision cycle (one bar)."""
        if self.model is None or self.normalizer is None:
            raise RuntimeError("Call load_artifacts() before trading.")

        bars = self.bridge.get_rates(count=max(500, self.config.env.window_size * 4))
        account_state, positions = self._current_account_state()
        obs = self._build_observation(bars, account_state)

        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        logger.info("Agent action: %s", ACTION_NAMES.get(action, action))

        # Map the discrete action to a broker order, respecting risk guards.
        if action in (ACTION_BUY, ACTION_SELL) and not positions:
            if not self._passes_risk_checks(positions):
                return
            direction = 1 if action == ACTION_BUY else -1
            price = float(bars["close"].iloc[-1])
            sl = price * (1 - self.config.env.stop_loss_pct * direction)
            tp = price * (1 + self.config.env.take_profit_pct * direction)
            self.bridge.market_order(
                direction=direction,
                lots=min(self.config.env.max_position_lots, self.config.live.max_lots),
                sl_price=sl,
                tp_price=tp,
            )
        elif action == ACTION_CLOSE and positions:
            self.bridge.close_position(positions[0]["ticket"])

    def run(self, max_iterations: Optional[int] = None) -> None:
        """Poll the market on ``config.live.poll_seconds`` and act each bar.

        Stops after ``max_iterations`` cycles (``None`` => run until interrupted).
        Refuses to start unless ``config.live.enabled`` is True.
        """
        if not self.config.live.enabled:
            raise RuntimeError(
                "Live trading is disabled. Set live.enabled=true in the config "
                "after thorough backtesting and only with risk capital."
            )
        self.load_artifacts()

        iteration = 0
        try:
            while max_iterations is None or iteration < max_iterations:
                try:
                    self.step_once()
                except Exception:  # keep the loop alive on transient errors
                    logger.exception("Error during live step; continuing.")
                iteration += 1
                time.sleep(self.config.live.poll_seconds)
        finally:
            self.bridge.shutdown()
