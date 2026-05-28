"""
Walk-forward A/B test for a feature group (V3 / Phase 4B).

Enforces the roadmap rule "validate every indicator with walk-forward": it runs
walk-forward twice — a baseline feature set vs. baseline + a candidate group —
and reports whether the candidate improves out-of-sample performance *and*
robustness (it must not just raise return on one lucky fold).

Compute-heavy: each side trains the agent once per fold. Use a modest
``--timesteps`` and ``--folds`` while iterating.

Examples
--------
    # Does adding the 'structure' group help, on top of core features?
    python scripts/feature_ab.py --group structure --folds 3 --timesteps 60000

    # Test 'volume' on top of an existing baseline of core + candle
    python scripts/feature_ab.py --baseline candle --group volume
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from validation.splitters import TimeSeriesSplitter  # noqa: E402
from validation.walk_forward import WalkForwardValidator  # noqa: E402


def _run(cfg: Config, groups, folds, timesteps, data):
    cfg.features.feature_groups = list(groups)
    splitter = TimeSeriesSplitter(n_splits=folds, mode="expanding", gap=cfg.env.window_size)
    validator = WalkForwardValidator(cfg, timesteps_per_fold=timesteps, splitter=splitter)
    return validator.run(csv_path=data).aggregate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward A/B test a feature group.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--group", required=True, help="Candidate feature group to evaluate.")
    parser.add_argument("--baseline", nargs="*", default=[],
                        help="Feature groups already in the baseline (default: core only).")
    parser.add_argument("--policy-arch", default=None,
                        choices=["mlp", "lstm", "transformer", "cnn"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=60_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    if args.policy_arch:
        cfg.training.policy_arch = args.policy_arch
    cfg.paths.ensure()

    logging.info("A/B baseline groups=%s | candidate=+%s", args.baseline or ["core"], args.group)
    base = _run(cfg, args.baseline, args.folds, args.timesteps, args.data)
    cand = _run(cfg, list(args.baseline) + [args.group], args.folds, args.timesteps, args.data)

    def fmt(a):
        return (f"mean_ret={a['mean_return_pct']:+.2f}% "
                f"profitable_folds={a['pct_profitable_folds']:.0f}% "
                f"std={a['std_return_pct']:.2f} sharpe={a['mean_sharpe']:.2f}")

    print("\n" + "=" * 70)
    print(f"Feature A/B — candidate group: '{args.group}'")
    print("=" * 70)
    print(f"  Baseline ({args.baseline or ['core']}):\n    {fmt(base)}")
    print(f"  + {args.group}:\n    {fmt(cand)}")

    d_ret = cand["mean_return_pct"] - base["mean_return_pct"]
    d_prof = cand["pct_profitable_folds"] - base["pct_profitable_folds"]
    d_std = cand["std_return_pct"] - base["std_return_pct"]
    print("\n  Delta:")
    print(f"    mean return     : {d_ret:+.2f} pp")
    print(f"    profitable folds: {d_prof:+.0f} pp")
    print(f"    return std      : {d_std:+.2f} (lower is better)")

    # Keep only if it helps mean return AND does not reduce robustness.
    verdict = "KEEP" if (d_ret > 0 and d_prof >= 0) else "DROP"
    print(f"\n  Verdict: {verdict} the '{args.group}' group "
          f"(rule of thumb: keep only if mean return ↑ and profitable-fold % does not drop).")
    print("=" * 70)


if __name__ == "__main__":
    main()
