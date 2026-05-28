"""
Generate synthetic XAUUSD data so the pipeline is runnable out of the box.

This writes a multi-year (2019..2025) OHLCV CSV to ``data/`` for the configured
timeframe.  The data is SYNTHETIC — see ``utils/sample_data.py`` — and is only a
fixture for exercising the framework, never a basis for real conclusions.

Usage:
    python scripts/generate_data.py --timeframe M5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from utils.sample_data import write_sample_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic XAUUSD data.")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path.")
    parser.add_argument("--timeframe", default=None, help="Override timeframe (M5/M15/H1).")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    timeframe = args.timeframe or cfg.data.timeframe
    cfg.paths.ensure()

    out_path = cfg.paths.data / f"{cfg.data.symbol}_{timeframe}.csv"
    write_sample_csv(out_path, timeframe=timeframe, start=args.start, end=args.end)
    print(f"Synthetic {timeframe} data written to: {out_path}")
    print("NOTE: This is synthetic test data, not real market data.")


if __name__ == "__main__":
    main()
