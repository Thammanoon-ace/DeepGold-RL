"""
V4 — CLI driver for ``ExecutionGridEvaluator``.

The execution analogue of ``scripts/grid_eval.py``: runs a seed × fold grid
on :class:`env.execution_env.ExecutionGoldEnv`, writes ``cells.csv`` /
``ensemble_cells.csv`` / ``summary.json`` under ``logs/grid/<tag>/``, and
prints the standard distribution + bootstrap CI summary.

Example (a 3 seed × 5 fold first verdict on real data):
    python scripts/grid_eval_execution.py --seeds 3 --folds 5 \\
        --timesteps 80000 --num-envs 4 --n-eval-episodes 200 \\
        --tag exec_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from validation.grid_execution import ExecutionGridEvaluator  # noqa: E402
from validation.splitters import TimeSeriesSplitter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="V4 execution seed x fold grid.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--data", default=None)
    p.add_argument("--policy-arch", default="cnn", choices=["mlp", "cnn"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--n-eval-episodes", type=int, default=200)
    p.add_argument("--deadline-min", type=int, default=32)
    p.add_argument("--deadline-max", type=int, default=128)
    p.add_argument("--fixed-cost-bps", type=float, default=1.0)
    p.add_argument("--impact-bps-per-lot", type=float, default=10.0)
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument("--tag", default="exec_grid")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.training.policy_arch = args.policy_arch
    cfg.paths.ensure()

    splitter = TimeSeriesSplitter(n_splits=args.folds, mode="expanding",
                                  gap=cfg.env.window_size)
    evaluator = ExecutionGridEvaluator(
        cfg, seeds=list(range(args.seeds)), splitter=splitter,
        timesteps_per_fold=args.timesteps, num_envs=args.num_envs,
        n_eval_episodes=args.n_eval_episodes,
        deadline_range=(args.deadline_min, args.deadline_max),
        fixed_cost_bps=args.fixed_cost_bps,
        impact_bps_per_lot=args.impact_bps_per_lot,
        evaluate_ensemble=not args.no_ensemble,
        run_tag=args.tag,
    )
    result = evaluator.run(csv_path=args.data)

    out = cfg.paths.logs / "grid" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    result.cells.to_csv(out / "cells.csv", index=False)
    if not result.ensemble_cells.empty:
        result.ensemble_cells.to_csv(out / "ensemble_cells.csv", index=False)
    summary = {
        "policy_arch": cfg.training.policy_arch,
        "deadline_range": [args.deadline_min, args.deadline_max],
        "fixed_cost_bps": args.fixed_cost_bps,
        "impact_bps_per_lot": args.impact_bps_per_lot,
        "n_eval_episodes": args.n_eval_episodes,
        "single": result.single.to_dict(),
        "ensemble": result.ensemble.to_dict() if result.ensemble else None,
        "median_ci": result.median_ci,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n" + result.summary())
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
