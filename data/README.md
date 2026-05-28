# `data/` — market data

Place your XAUUSD OHLCV CSV here. The default filename expected by the config is
`XAUUSD_M5.csv` (change `data.csv_filename` / `data.timeframe` in
`config/config.yaml` for other instruments or timeframes).

## Expected CSV format

See [`XAUUSD_M5_example.csv`](XAUUSD_M5_example.csv) for a 10-row example.

| column        | description                              |
| ------------- | ---------------------------------------- |
| `time`        | bar open time, parseable by pandas       |
| `open`        | open price (USD/oz)                      |
| `high`        | high price                               |
| `low`         | low price                                |
| `close`       | close price                              |
| `tick_volume` | tick volume (optional; 0 is fine)        |

The loader is tolerant of MetaTrader 5 export conventions: it recognises common
column aliases (`<OPEN>`, `Date`+`Time` split columns, tab separators, etc.) and
lower-cases headers automatically.

## Getting real data

* **Dukascopy (recommended, free, no account)** — downloads real XAUUSD bars:

  ```bash
  python scripts/download_dukascopy.py --start 2019-01-01 --end 2025-12-31 --timeframe M5
  ```

* **MetaTrader 5** — `copy_rates_range` / export from the terminal, or use the
  `MT5Bridge.get_rates()` helper in `live_trading/`.
* **Generate synthetic test data** (for trying the pipeline only):

  ```bash
  python scripts/generate_data.py --timeframe M5
  ```

> ⚠️ Synthetic data is a fixture for exercising the code. Any performance on it
> is meaningless — use real historical data for actual research.

## Walk-forward split

Training uses every bar **before** `data.test_start` (default `2025-01-01`);
evaluation uses **2025 only**. Make sure your CSV spans both periods.
