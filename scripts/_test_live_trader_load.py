"""Smoke test: LiveTrader.load_artifacts() with a .pt CleanRL ActorCritic."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config.config import Config
from live_trading.live_trader import LiveTrader

cfg = Config.from_yaml("config/config.yaml")
cfg.training.model_name = "h4_cosine_swa_v4smoke"
trader = LiveTrader(cfg)
trader.load_artifacts()
print(f"  model type: {type(trader.model).__name__}")
print(f"  normalizer features: {len(trader.normalizer.feature_columns)}")
print(f"  meta timeframe: {trader.meta['timeframe']}")
print(f"  meta swa_samples: {trader.meta['training']['swa_samples']}")

# Test predict on a synthetic observation matching the model's expected shape.
window = trader.meta["env"]["window_size"]
n_feat = len(trader.meta["feature_columns"])
obs_dim = window * n_feat + 5
obs = np.clip(np.random.randn(obs_dim).astype(np.float32), -10, 10)
action, _ = trader.model.predict(obs, deterministic=True)
print(f"  predict returned action: {action} (type {type(action).__name__})")
print("  LiveTrader.load_artifacts() with .pt + ActorCritic: OK")
