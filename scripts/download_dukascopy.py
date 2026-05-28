"""
Download real historical XAUUSD data from Dukascopy (free, no account).

Dukascopy publishes free historical bar/tick data for gold (XAU/USD). This
script fetches it via the ``dukascopy-python`` package, converts it to the
project's canonical CSV schema (``time, open, high, low, close, tick_volume``)
and writes it to ``data/`` so the rest of the pipeline can use it directly.

The download is chunked **month by month** for resilience: each chunk is
retried on failure and partial progress is cached, so a long multi-year pull can
be re-run with ``--resume`` without re-downloading completed months.

Examples
--------
    # Default: XAUUSD M5, 2020-01-01 .. 2025-12-31 (BID prices)
    python scripts/download_dukascopy.py

    # Custom range / timeframe, resumable
    python scripts/download_dukascopy.py --start 2022-01-01 --end 2025-12-31 \
        --timeframe M5 --resume
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import Config

logger = logging.getLogger(__name__)

# Map our timeframe labels to dukascopy interval constants (resolved lazily).
_INTERVAL_NAMES = {
    "M5": "INTERVAL_MIN_5",
    "M15": "INTERVAL_MIN_15",
    "H1": "INTERVAL_HOUR_1",
}


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return month-start boundaries covering [start, end] (inclusive of end)."""
    first = start.normalize().replace(day=1)
    bounds = list(pd.date_range(first, end, freq="MS"))
    if not bounds or bounds[0] > start:
        bounds = [start] + bounds
    bounds.append(end + pd.Timedelta(days=1))  # ensure the last month is covered
    # De-duplicate while preserving order.
    seen, out = set(), []
    for b in bounds:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _fetch_chunk(d, instrument, interval, offer, start, end, retries: int = 3):
    """Fetch one chunk with simple retry/backoff; return a DataFrame (maybe empty)."""
    for attempt in range(1, retries + 1):
        try:
            df = d.fetch(instrument, interval, offer,
                         start.to_pydatetime(), end.to_pydatetime())
            return df if df is not None else pd.DataFrame()
        except Exception as exc:  # transient network / server hiccups
            logger.warning("  chunk %s..%s attempt %d/%d failed: %s",
                           start.date(), end.date(), attempt, retries, exc)
            time.sleep(2 * attempt)
    logger.error("  giving up on chunk %s..%s", start.date(), end.date())
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XAUUSD data from Dukascopy.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--timeframe", default=None, help="M5/M15/H1 (defaults to config).")
    parser.add_argument("--offer-side", choices=["bid", "ask"], default="bid")
    parser.add_argument("--out", default=None, help="Output CSV path (defaults to data/<SYM>_<TF>.csv).")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse per-month chunks already cached under .dukascopy_cache/.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg.paths.ensure()
    timeframe = args.timeframe or cfg.data.timeframe
    if timeframe not in _INTERVAL_NAMES:
        raise SystemExit(f"Unsupported timeframe {timeframe!r}; choose M5/M15/H1.")

    try:
        import dukascopy_python as d
        from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD
    except ImportError as exc:
        raise SystemExit(
            "dukascopy-python is required. Install with `pip install dukascopy-python`."
        ) from exc

    interval = getattr(d, _INTERVAL_NAMES[timeframe])
    offer = d.OFFER_SIDE_BID if args.offer_side == "bid" else d.OFFER_SIDE_ASK
    instrument = INSTRUMENT_FX_METALS_XAU_USD

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    bounds = _month_starts(start, end)
    cache_dir = cfg.paths.data / ".dukascopy_cache" / f"{cfg.data.symbol}_{timeframe}_{args.offer_side}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s %s [%s] %s -> %s (%d monthly chunks)",
                cfg.data.symbol, timeframe, args.offer_side, start.date(), end.date(), len(bounds) - 1)

    frames = []
    for i, (c_start, c_end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        tag = c_start.strftime("%Y-%m")
        cache_file = cache_dir / f"{tag}.pkl"
        if args.resume and cache_file.exists():
            frames.append(pd.read_pickle(cache_file))
            logger.info("[%d/%d] %s cached (%d rows)", i, len(bounds) - 1, tag, len(frames[-1]))
            continue

        df = _fetch_chunk(d, instrument, interval, offer, c_start, c_end)
        if not df.empty:
            try:
                df.to_pickle(cache_file)  # best-effort cache for --resume
            except Exception:
                pass
            frames.append(df)
        logger.info("[%d/%d] %s -> %d rows", i, len(bounds) - 1, tag, len(df))

    if not frames:
        raise SystemExit("No data downloaded. Check your date range / network.")

    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="last")].sort_index()

    # Normalize to the project's canonical schema.
    full.index = full.index.tz_convert("UTC").tz_localize(None)
    full = full.rename(columns={"volume": "tick_volume"})
    full = full[["open", "high", "low", "close", "tick_volume"]]
    full = full.loc[(full.index >= start) & (full.index <= end + pd.Timedelta(days=1))]

    out_path = Path(args.out) if args.out else (cfg.paths.data / f"{cfg.data.symbol}_{timeframe}.csv")
    full.to_csv(out_path, index_label="time")

    logger.info("Saved %d bars to %s", len(full), out_path)
    print(f"\nReal XAUUSD {timeframe} data saved to: {out_path}")
    print(f"Rows: {len(full):,} | Range: {full.index.min()} -> {full.index.max()}")
    print("Next: python scripts/inspect_data.py --data \"%s\"" % out_path)


if __name__ == "__main__":
    main()
