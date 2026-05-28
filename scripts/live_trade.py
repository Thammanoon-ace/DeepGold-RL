"""
Example live-trading launcher (requirement #9).  WINDOWS + MT5 ONLY.

SAFETY: live trading is disabled unless ``live.enabled: true`` in the config AND
the master ``--i-understand-the-risk`` flag is passed.  ``live.dry_run`` defaults
to True, meaning orders are logged but NOT sent.  Backtest thoroughly first.

Credentials must come from environment variables, never the command line/source:
    $env:MT5_LOGIN = "12345678"
    $env:MT5_PASSWORD = "..."
    $env:MT5_SERVER = "Broker-Server"

    python scripts/live_trade.py --i-understand-the-risk --iterations 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import Config  # noqa: E402
from live_trading.live_trader import LiveTrader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RL agent live via MetaTrader 5.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--name", default=None, help="Model base name.")
    parser.add_argument("--iterations", type=int, default=None, help="Max decision cycles.")
    parser.add_argument(
        "--i-understand-the-risk",
        action="store_true",
        help="Required acknowledgement that live trading risks real capital.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if not args.i_understand_the_risk:
        raise SystemExit(
            "Refusing to start: pass --i-understand-the-risk to confirm you accept "
            "the risk of trading real capital. Keep live.dry_run=true until ready."
        )

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()

    trader = LiveTrader(cfg, model_name=args.name)
    connected = trader.connect(
        login=os.getenv("MT5_LOGIN") and int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    )
    if not connected:
        raise SystemExit("Could not connect to MetaTrader 5. Is the terminal running?")

    trader.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
