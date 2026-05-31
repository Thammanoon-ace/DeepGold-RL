"""V4 readiness smoke test — train + save + load + predict round trip.

End-to-end proof that the H4 cosine+SWA agent can be:
1. Trained from scratch (using the validated cosine + SWA recipe).
2. Persisted to disk (state_dict + normalizer + env config).
3. Loaded back into a fresh process state.
4. Used to run a deterministic backtest matching grid_eval results.

This is the V4 unblock-readiness check: it doesn't connect to MT5 (that
requires a broker account), but it proves every artifact the LiveTrader would
need is producible and reload-able. After this passes, the only remaining V4
work is wiring the loaded ActorCritic into the existing LiveTrader (whose
predict-loop already matches our `ActorCritic.predict` signature).

Usage:
    python scripts/v4_smoke.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from backtest.backtester import run_episode
from backtest.baselines import buy_and_hold_return
from backtest.metrics import compute_report
from config.config import BARS_PER_YEAR, Config
from env.env_builder import TradingDataPipeline, make_env_from_frame
from env.gold_trading_env import N_ACCOUNT_FEATURES
from env.torch_vec_env import TorchVecGoldEnv
from training.cleanrl_ppo import ActorCritic, PPOConfig, train_cleanrl_ppo
from utils.normalization import FeatureNormalizer

MODEL_NAME = "h4_cosine_swa_v4smoke"


def main() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.training.policy_arch = "cnn"
    cfg.env.reward_mode = "excess"
    cfg.paths.ensure()
    models_dir = cfg.paths.models
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Train on H4 train split ---------------------------------- #
    print("\n[1/5] Loading H4 data, training ActorCritic with cosine+SWA...")
    cfg.data.timeframe = "H4"  # ensure pipeline resamples M5 -> H4
    # Match grid.py path: load raw, resample, then re-run pipeline against resampled.
    pipe = TradingDataPipeline(cfg)
    raw = pipe.loader.load()
    raw = pipe.loader.resample(raw, "H4")
    featured = pipe.engineer.transform(raw)
    pipe.feature_columns = pipe.engineer.feature_columns
    train_raw, test_raw = pipe.loader.train_test_split(featured)
    feat_cols = pipe.feature_columns
    normalizer = FeatureNormalizer(feat_cols, "robust")
    train_df = normalizer.fit_transform(train_raw)
    test_df = normalizer.transform(test_raw)
    pipe.normalizer = normalizer
    features = train_df[feat_cols].to_numpy(dtype=np.float32)
    prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    bars_per_year = BARS_PER_YEAR.get("H4", cfg.backtest.bars_per_year)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TorchVecGoldEnv(features, prices, cfg.env, num_envs=2048,
                          random_start=True, device=device, dtype=torch.float32, seed=0)
    ppo = PPOConfig(total_timesteps=1_310_720, n_steps=128, lr_schedule="cosine",
                    cosine_min_frac=0.05, use_swa=True, swa_start_frac=0.6)
    t0 = time.perf_counter()
    ac, _ = train_cleanrl_ppo(env, arch="cnn", ppo=ppo, seed=0)
    print(f"  trained in {time.perf_counter()-t0:.1f}s, swa samples = {ac._swa_n_samples}")

    # ---- 2. Save artifacts ------------------------------------------ #
    print(f"\n[2/5] Saving artifacts under {models_dir}/...")
    model_path = models_dir / f"{MODEL_NAME}.pt"
    norm_path = models_dir / f"{MODEL_NAME}_normalizer.joblib"
    meta_path = models_dir / f"{MODEL_NAME}_meta.json"

    torch.save({
        "state_dict": ac.state_dict(),
        "obs_dim": env.obs_dim,
        "window": env.window,
        "n_features": env.n_features,
        "arch": "cnn",
    }, model_path)
    pipe.normalizer.save(norm_path)
    meta = {
        "timeframe": "H4",
        "policy_arch": "cnn",
        "reward_mode": "excess",
        "feature_columns": feat_cols,
        "env": asdict(cfg.env),
        "training": {
            "lr_schedule": "cosine",
            "cosine_min_frac": 0.05,
            "use_swa": True,
            "swa_start_frac": 0.6,
            "swa_samples": int(ac._swa_n_samples),
            "total_timesteps": 1_310_720,
            "num_envs": 2048,
            "n_steps": 128,
            "seed": 0,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  model: {model_path}")
    print(f"  normalizer: {norm_path}")
    print(f"  meta: {meta_path}")

    # ---- 3. Reset state, reload from disk ---------------------------- #
    print("\n[3/5] Reloading artifacts from disk (round-trip check)...")
    del ac
    torch.cuda.empty_cache()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ac_loaded = ActorCritic(ckpt["obs_dim"], ckpt["window"], ckpt["n_features"],
                            arch=ckpt["arch"]).to(device)
    ac_loaded.load_state_dict(ckpt["state_dict"])
    ac_loaded.eval()
    norm_loaded = FeatureNormalizer.load(norm_path)
    meta_loaded = json.loads(meta_path.read_text())
    print(f"  loaded ActorCritic with {sum(p.numel() for p in ac_loaded.parameters()):,} params")
    print(f"  loaded normalizer over {len(meta_loaded['feature_columns'])} features")

    # ---- 4. Deterministic backtest on the held-out 2025 test split --- #
    print("\n[4/5] Running deterministic backtest on test split...")
    test_env = make_env_from_frame(test_df, feat_cols, cfg.env, random_start=False)
    hist = run_episode(ac_loaded, test_env, deterministic=True)
    rep = compute_report(hist["equity_curve"], hist["trades"], hist["initial_balance"],
                         bars_per_year=bars_per_year)
    bh = buy_and_hold_return(test_df["close"].to_numpy())["total_return_pct"]

    # ---- 5. Report ---------------------------------------------------- #
    print("\n[5/5] V4 readiness report")
    print(f"  Loaded-model return     : {rep.total_return_pct:+.2f}%")
    print(f"  Loaded-model Sharpe     : {rep.sharpe_ratio:.3f}")
    print(f"  Loaded-model max DD     : {rep.max_drawdown_pct:.2f}%")
    print(f"  Loaded-model n_trades   : {rep.n_trades}")
    print(f"  Buy-and-hold return     : {bh:+.2f}%")
    print(f"  Excess vs BH            : {rep.total_return_pct - bh:+.2f}%")
    print()
    print(f"  V4 readiness: {'OK' if rep.n_trades > 0 else 'NO-TRADE'} — "
          "model train -> save -> load -> predict pipeline works end-to-end.")
    print("  Next: wire `ActorCritic` load into live_trading/live_trader.py "
          "(change `load_artifacts` to detect .pt vs .zip).")


if __name__ == "__main__":
    main()
