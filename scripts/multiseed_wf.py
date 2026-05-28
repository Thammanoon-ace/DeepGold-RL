"""
Multi-seed walk-forward harness (variance estimation).

Runs the SAME walk-forward setup with several different training seeds and
reports the spread of results ACROSS seeds. The fold structure is held fixed, so
the only thing that changes is the random seed (network init, env reset offsets,
action sampling) — isolating "training-run variance".

Why this matters: a strategy that scores +44% on one seed but swings between
-40% and +90% across seeds has no reliable edge — the headline number is luck.
A trustworthy result is one that stays positive (and similar) across seeds.

Examples
--------
    python scripts/multiseed_wf.py --policy-arch cnn --seeds 4 --folds 5 --timesteps 50000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import Config
from validation.splitters import TimeSeriesSplitter
from validation.walk_forward import WalkForwardValidator


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed walk-forward variance test.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--policy-arch", default="cnn",
                        choices=["mlp", "lstm", "transformer", "cnn", "cnn_lstm"])
    parser.add_argument("--seeds", type=int, default=4, help="Number of distinct seeds.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--feature-groups", nargs="*", default=None,
                        choices=["trend", "momentum", "volatility", "candle", "structure", "volume"])
    parser.add_argument("--correlation", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.training.policy_arch = args.policy_arch
    if args.feature_groups is not None:
        cfg.features.feature_groups = args.feature_groups
    if args.correlation is not None:
        cfg.features.correlation_threshold = args.correlation
    cfg.paths.ensure()

    seeds = [i * 1000 for i in range(args.seeds)]  # widely spaced bases
    rows = []
    for s in seeds:
        cfg.training.seed = s
        splitter = TimeSeriesSplitter(n_splits=args.folds, mode="expanding", gap=cfg.env.window_size)
        validator = WalkForwardValidator(cfg, timesteps_per_fold=args.timesteps, splitter=splitter)
        logging.info("=== SEED %d ===", s)
        agg = validator.run(csv_path=args.data).aggregate()
        rows.append({
            "seed": s,
            "mean_return_pct": round(agg["mean_return_pct"], 2),
            "compounded_pct": round(agg["compounded_return_pct"], 2),
            "pct_profitable_folds": agg["pct_profitable_folds"],
            "mean_sharpe": round(agg["mean_sharpe"], 2),
            "worst_fold_pct": round(agg["worst_fold_return_pct"], 2),
        })

    df = pd.DataFrame(rows)
    out = cfg.paths.logs / "multiseed" / args.policy_arch
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)

    mret = df["mean_return_pct"].to_numpy()
    comp = df["compounded_pct"].to_numpy()
    print("\n" + "=" * 70)
    print(f"Multi-seed walk-forward — {args.policy_arch}, {len(seeds)} seeds, "
          f"{args.folds}-fold req, {args.timesteps:,} steps/fold")
    print("=" * 70)
    print(df.to_string(index=False))
    print("-" * 70)
    print(f"  mean_return_pct  across seeds : {mret.mean():+.2f}  (std {mret.std():.2f}, "
          f"min {mret.min():+.2f}, max {mret.max():+.2f})")
    print(f"  compounded_pct   across seeds : {comp.mean():+.2f}  (std {comp.std():.2f}, "
          f"min {comp.min():+.2f}, max {comp.max():+.2f})")
    print("=" * 70)
    spread = comp.max() - comp.min()
    if comp.mean() > 0 and comp.std() < abs(comp.mean()):
        verdict = "PROMISING: positive and reasonably stable across seeds."
    elif spread > 100:
        verdict = "NOT RELIABLE: results swing wildly across seeds (variance dominates)."
    else:
        verdict = "WEAK/UNCLEAR: not consistently profitable across seeds."
    print(f"Verdict: {verdict}")
    (out / "summary.json").write_text(json.dumps({
        "seeds": seeds,
        "mean_return_mean": float(mret.mean()), "mean_return_std": float(mret.std()),
        "compounded_mean": float(comp.mean()), "compounded_std": float(comp.std()),
        "compounded_min": float(comp.min()), "compounded_max": float(comp.max()),
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
