"""Internal smoke test: imports every module, then exercises the data pipeline,
environment rollout and metrics (no SB3 training required).

Run:  python scripts/_smoke_test.py
"""
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# 1. Import every package module to catch syntax/import errors early.
MODULES = [
    "config.config",
    "utils.data_loader",
    "utils.feature_engineering",
    "utils.normalization",
    "utils.visualization",
    "utils.sample_data",
    "utils.indicators",
    "utils.feature_selection",
    "env.gold_trading_env",
    "env.env_builder",
    "training.train_ppo",
    "training.callbacks",
    "backtest.metrics",
    "backtest.backtester",
    "validation.splitters",
    "validation.walk_forward",
    "validation.multi_dataset",
    "validation.robustness",
    "optimization.optuna_tuning",
    "policies.extractors",
    "policies.factory",
    "policies.ensemble",
    "env.vectorized_env",
    "backtest.baselines",
    "live_trading.mt5_bridge",
    "live_trading.live_trader",
]
for m in MODULES:
    importlib.import_module(m)
print(f"Imported {len(MODULES)} modules OK")

# 2. Exercise the data pipeline + environment with a random policy.
import numpy as np

from backtest.metrics import compute_report
from config.config import Config
from env.env_builder import TradingDataPipeline

cfg = Config.from_yaml("config/config.yaml")
pipe = TradingDataPipeline(cfg)
tr, te = pipe.prepare()
print("TRAIN bars:", len(tr), "| TEST bars:", len(te), "| features:", len(pipe.feature_columns))

env = pipe.make_env("train", random_start=False)
obs, info = env.reset()
print("obs shape:", obs.shape, "| action space:", env.action_space)
assert env.observation_space.contains(obs), "reset obs outside observation_space!"

rng = np.random.default_rng(0)
done, steps = False, 0
while not done and steps < 8000:
    obs, r, term, trunc, info = env.step(int(rng.integers(0, 4)))
    assert env.observation_space.contains(obs), f"obs outside space at step {steps}"
    done = term or trunc
    steps += 1

print(f"random rollout: {steps} steps | final equity {info['equity']:.2f} | trades {info['n_trades']}")
h = env.get_episode_history()
rep = compute_report(h["equity_curve"], h["trades"], h["initial_balance"],
                     bars_per_year=cfg.backtest.bars_per_year)
print(rep.pretty())
print("SMOKE-OK")
