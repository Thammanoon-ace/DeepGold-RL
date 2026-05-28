"""
Multi-dataset evaluation script (V2 / Phase 3).

Evaluates one trained model across several datasets/periods to gauge how well
it generalizes. Point it at a folder of CSVs or list specific files.

Examples
--------
    # Evaluate every CSV in data/ on its 2025 slice
    python scripts/evaluate_multi.py --name ppo_gold --glob "data/*.csv" --test-only

    # Evaluate explicit files over their full history
    python scripts/evaluate_multi.py --name ppo_gold --files data/XAUUSD_M5.csv data/EURUSD_M5.csv
"""
from __future__ import annotations

import argparse
import glob as globlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from validation.multi_dataset import MultiDatasetEvaluator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model across datasets.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--name", default=None, help="Model base name.")
    parser.add_argument("--glob", default=None, help="Glob of CSVs, e.g. 'data/*.csv'.")
    parser.add_argument("--files", nargs="*", default=None, help="Explicit CSV paths.")
    parser.add_argument("--test-only", action="store_true",
                        help="Restrict each dataset to the config test_start..test_end range.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()

    paths = list(args.files or [])
    if args.glob:
        paths += sorted(globlib.glob(args.glob))
    if not paths:
        raise SystemExit("Provide --files and/or --glob to specify datasets.")
    datasets = [(Path(p).stem, p) for p in paths]

    date_range = None
    if args.test_only:
        end = cfg.data.test_end or "2100-01-01"
        date_range = (cfg.data.test_start, end)

    evaluator = MultiDatasetEvaluator(cfg, model_name=args.name)
    df = evaluator.run(datasets, date_range=date_range)

    print("\nMulti-dataset evaluation")
    print("=" * 60)
    print(df.to_string(index=False))

    out = cfg.paths.logs / "multi_dataset"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    print(f"\nSaved to: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
