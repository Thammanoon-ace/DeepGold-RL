"""
V4 Phase 2 — first SB3 PPO training run on ``ExecutionGoldEnv``.

Loads the real XAUUSD M5 data through the existing leakage-safe pipeline
(walk-forward split + train-only normalizer), trains a single seed of SB3
PPO with the CNN feature extractor, and evaluates deterministically on a
batch of test-set episodes. Reports the distribution of
``bps_savings_vs_twap`` — the V4 headline metric.

This is a "does it learn at all?" smoke before investing in the grid:
- if median bps_savings > 0 and a meaningful fraction of episodes beat TWAP,
  proceed to the seed × fold grid (Phase 3);
- if not, debug reward shape / network size / training budget before scaling.

Example
-------
    python scripts/train_execution.py --timesteps 100000 --n-eval-episodes 300
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor  # noqa: E402

from config.config import Config  # noqa: E402
from env.env_builder import TradingDataPipeline  # noqa: E402
from env.execution_env import ExecutionGoldEnv  # noqa: E402
from policies.factory import build_policy_kwargs  # noqa: E402
from training.train_ppo import _progress_bar_available, resolve_device  # noqa: E402

logger = logging.getLogger("v4_train")


def evaluate(model, env: ExecutionGoldEnv, n_episodes: int, seed_base: int,
             diagnose_actions: bool = False):
    """Run ``n_episodes`` deterministic episodes; return per-episode metrics
    (and, optionally, an action-frequency histogram across all steps).
    """
    savings, sf, twap_sf, n_terminal = [], [], [], 0
    action_hist = np.zeros(env.action_space.n, dtype=np.int64) if diagnose_actions else None
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            if action_hist is not None:
                action_hist[a] += 1
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc
        if "bps_savings_vs_twap" in info:
            n_terminal += 1
            savings.append(info["bps_savings_vs_twap"])
            sf.append(info["shortfall_bps"])
            twap_sf.append(info["twap_shortfall_bps"])
    return (np.asarray(savings), np.asarray(sf), np.asarray(twap_sf), n_terminal,
            action_hist)


def main() -> None:
    p = argparse.ArgumentParser(description="V4 SB3 PPO training on ExecutionGoldEnv.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval-episodes", type=int, default=300)
    p.add_argument("--policy-arch", default="cnn", choices=["mlp", "cnn"])
    p.add_argument("--deadline-min", type=int, default=32)
    p.add_argument("--deadline-max", type=int, default=128)
    p.add_argument("--fixed-cost-bps", type=float, default=1.0)
    p.add_argument("--impact-bps-per-lot", type=float, default=10.0)
    p.add_argument("--ent-coef", type=float, default=None,
                   help="Entropy bonus; overrides config (default 0.01). "
                        "Higher (0.03–0.1) prevents policy collapse to constant action.")
    p.add_argument("--diagnose-actions", action="store_true",
                   help="Print action-frequency histogram during eval.")
    p.add_argument("--execution-features", action="store_true",
                   help="Add 7 micro-timing features (ret_1/3/5, body/wick "
                        "ratios, z_score_5) targeted at intra-episode timing.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.training.policy_arch = args.policy_arch
    if args.ent_coef is not None:
        cfg.training.ent_coef = args.ent_coef
    if args.execution_features:
        cfg.features.execution_features = True
    cfg.paths.ensure()
    device = resolve_device(cfg.training.device)

    # ---- leakage-safe data (normalizer fit on train only) -------------- #
    pipeline = TradingDataPipeline(cfg)
    train_df, test_df = pipeline.prepare(args.data)
    feat_cols = pipeline.feature_columns
    train_feats = train_df[feat_cols].to_numpy(dtype=np.float32)
    train_prices = train_df[["high", "low", "close"]].to_numpy(dtype=np.float64)
    test_feats = test_df[feat_cols].to_numpy(dtype=np.float32)
    test_prices = test_df[["high", "low", "close"]].to_numpy(dtype=np.float64)

    logger.info("V4 | arch=%s | device=%s | timesteps=%d | seed=%d",
                args.policy_arch, device, args.timesteps, args.seed)
    logger.info("Train bars=%d  test bars=%d  features=%d",
                len(train_df), len(test_df), len(feat_cols))
    logger.info("Episode: deadline ∈ [%d, %d] bars; cost: fixed=%.1f bps, impact=%.1f bps/lot",
                args.deadline_min, args.deadline_max,
                args.fixed_cost_bps, args.impact_bps_per_lot)

    # ---- training env (random episode starts inside train slice) -------- #
    def make_train_env():
        return ExecutionGoldEnv(
            train_feats, train_prices, cfg.env,
            deadline_range=(args.deadline_min, args.deadline_max),
            fixed_cost_bps=args.fixed_cost_bps,
            impact_bps_per_lot=args.impact_bps_per_lot,
            random_start=True, seed=args.seed)
    vec = VecMonitor(DummyVecEnv([make_train_env]))

    tcfg = cfg.training
    model = PPO(
        policy="MlpPolicy", env=vec,
        learning_rate=tcfg.learning_rate, n_steps=tcfg.n_steps,
        batch_size=tcfg.batch_size, n_epochs=tcfg.n_epochs,
        gamma=tcfg.gamma, gae_lambda=tcfg.gae_lambda, clip_range=tcfg.clip_range,
        ent_coef=tcfg.ent_coef, vf_coef=tcfg.vf_coef, max_grad_norm=tcfg.max_grad_norm,
        policy_kwargs=build_policy_kwargs(tcfg, cfg.env.window_size),
        device=device, seed=args.seed, verbose=0,
    )
    model.learn(total_timesteps=args.timesteps,
                progress_bar=_progress_bar_available())

    # ---- deterministic eval on test slice ------------------------------ #
    eval_env = ExecutionGoldEnv(
        test_feats, test_prices, cfg.env,
        deadline_range=(args.deadline_min, args.deadline_max),
        fixed_cost_bps=args.fixed_cost_bps,
        impact_bps_per_lot=args.impact_bps_per_lot,
        random_start=True, seed=args.seed + 1000)
    savings, sf, twap_sf, n_term, action_hist = evaluate(
        model, eval_env, args.n_eval_episodes, seed_base=args.seed + 10000,
        diagnose_actions=args.diagnose_actions)
    assert n_term == args.n_eval_episodes, \
        f"only {n_term}/{args.n_eval_episodes} episodes terminated"

    print("\n" + "=" * 60)
    print("V4 PHASE 2 — FIRST TRAINING EVAL")
    print("=" * 60)
    print(f"Episodes evaluated     : {n_term}")
    print(f"Agent shortfall (bps)  : "
          f"median {np.median(sf):+.2f}  mean {np.mean(sf):+.2f}  std {np.std(sf):.1f}")
    print(f"TWAP  shortfall (bps)  : "
          f"median {np.median(twap_sf):+.2f}  mean {np.mean(twap_sf):+.2f}  std {np.std(twap_sf):.1f}")
    print(f"bps_savings_vs_twap    :")
    print(f"   median             : {np.median(savings):+.3f}")
    print(f"   mean               : {np.mean(savings):+.3f}")
    print(f"   p10 / p90          : {np.percentile(savings, 10):+.1f} / "
          f"{np.percentile(savings, 90):+.1f}")
    print(f"   beats TWAP         : {(savings > 0).mean() * 100:.1f}% of episodes")
    win_rate = (savings > 0).mean() * 100
    verdict = ("POSITIVE signal — proceed to grid"
               if np.median(savings) > 0.2 and win_rate > 52 else
               "AMBIGUOUS / NEGATIVE — debug before scaling")
    print(f"\nVerdict (this seed only): {verdict}")
    if action_hist is not None:
        names = ["pause(0)", "slow(0.5×)", "TWAP(1×)", "fast(2×)"]
        total = action_hist.sum()
        print("\nAction distribution (deterministic eval):")
        for i, (n, c) in enumerate(zip(names, action_hist)):
            print(f"   {n:<12} : {c:>6}  ({c/max(total,1)*100:5.1f}%)")
    print("\nNote: single-seed result is not a verdict — distribution + CI from\n"
          "      the grid (Phase 3) is.")


if __name__ == "__main__":
    main()
