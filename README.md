# DeepGold RL — Reinforcement-Learning Gold (XAUUSD) Trading Framework

A modular, production-style research framework for training a **PPO** agent to
trade spot gold (XAUUSD), evaluating it **out-of-sample on 2025 data**, and
(optionally) bridging to **MetaTrader 5** for live execution.

> **Philosophy.** This is a research framework, *not* a profit guarantee. The
> environment models realistic frictions (spread, slippage, commission,
> leverage/margin), the reward is shaped to resist reward hacking, the
> normalizer is fit on training data only, and evaluation is strictly
> walk-forward — all to avoid the classic "fake overfitted backtest" trap.

---

## Features

| # | Capability | Where |
|---|------------|-------|
| 1 | CSV OHLCV loader (M5/M15/H1), missing-value handling, resampling | [`utils/data_loader.py`](utils/data_loader.py) |
| 2 | Feature engineering: RSI, MACD, EMA fast/slow, ATR, candle return %, rolling volatility | [`utils/feature_engineering.py`](utils/feature_engineering.py) |
| 3 | Custom Gymnasium env: Buy/Sell/Hold/Close, spread, slippage, commission, SL/TP, leverage, position & balance tracking, unrealized PnL | [`env/gold_trading_env.py`](env/gold_trading_env.py) |
| 4 | Reward = net profit − drawdown penalty − overtrading penalty − costs | [`env/gold_trading_env.py`](env/gold_trading_env.py) |
| 5 | PPO training: checkpoints, TensorBoard, GPU, resume | [`training/train_ppo.py`](training/train_ppo.py) |
| 6 | Backtest on unseen 2025 data: total return, win rate, Sharpe, max drawdown, equity curve | [`backtest/`](backtest/) |
| 7 | Matplotlib charts: equity curve, trade entries/exits, reward curve | [`utils/visualization.py`](utils/visualization.py) |
| 8 | Model + scaler + normalization config save/load | [`training/train_ppo.py`](training/train_ppo.py), [`utils/normalization.py`](utils/normalization.py) |
| 9 | Live trading architecture: Python AI → MT5 bridge → broker | [`live_trading/`](live_trading/) |
| 12 | CUDA GPU support (auto-detect), vectorized envs, modest memory footprint | [`training/train_ppo.py`](training/train_ppo.py) |
| 13 | Risk management: max position size, anti-overleverage, max-drawdown stop | [`env/gold_trading_env.py`](env/gold_trading_env.py) |
| V2 | Walk-forward validation, multi-dataset eval, Optuna tuning, Calmar/expectancy/trade-distribution metrics | [`validation/`](validation/), [`optimization/`](optimization/), [`backtest/metrics.py`](backtest/metrics.py) |
| V3 | Deep policies (LSTM/Transformer/CNN), multi-timeframe + volatility-regime features, DQN off-policy, arch comparison | [`policies/`](policies/), [`utils/feature_engineering.py`](utils/feature_engineering.py), [`scripts/compare_archs.py`](scripts/compare_archs.py) |

See [`ROADMAP.md`](ROADMAP.md) for the full V0→V7 plan and progress. V0–V2 complete; V3 Phase 4A done (4B indicator-expansion / 4C structural-patterns pending).

## Project structure

```
DeepGold RL/
├── config/            # Typed dataclass config + YAML overrides
├── data/              # OHLCV CSVs (+ example format)
├── env/               # GoldTradingEnv + data pipeline factory
├── utils/             # Loader, features, normalization, plots, sample data
├── training/          # PPO trainer + custom callbacks
├── validation/        # Walk-forward, time-series splitters, multi-dataset eval
├── optimization/      # Optuna hyperparameter tuning
├── policies/          # Deep sequence feature extractors (LSTM/Transformer/CNN)
├── backtest/          # Backtester + performance metrics
├── live_trading/      # MT5 bridge + live trader skeleton
├── models/            # Saved models, scalers, checkpoints (created at runtime)
├── logs/              # TensorBoard + backtest outputs (created at runtime)
├── notebooks/         # Jupyter quickstart
├── scripts/           # Example train / backtest / live / data-gen scripts
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Environment (Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
python -m pip install --upgrade pip

# GPU build of PyTorch — install FIRST (cu128 works on recent NVIDIA drivers;
# check `nvidia-smi`). Works on any CUDA card (GTX 1650 Ti, RTX 3070, ...):
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

### 2. Get data

**Recommended — download real XAUUSD history from Dukascopy (free, no account):**

```powershell
python scripts/download_dukascopy.py --start 2019-01-01 --end 2025-12-31 --timeframe M5
```

This writes real OHLCV bars to `data/XAUUSD_M5.csv` (chunked by month, resumable
with `--resume`). Then verify it:

```powershell
python scripts/inspect_data.py --data data/XAUUSD_M5.csv
```

Alternatives: drop your own broker CSV into `data/` (the loader handles MT5
terminal and Python-API formats), or generate synthetic test data just to try
the pipeline (results are meaningless):

```powershell
python scripts/generate_data.py --timeframe M5
```

### 3. Train

```powershell
python scripts/train.py                       # full run (config-driven)
python scripts/train.py --timesteps 50000 --name ppo_gold_smoke   # quick test
```

Monitor live:

```powershell
tensorboard --logdir logs/tensorboard
```

**Deep sequence architectures (V3)** — switch the policy encoder via `--policy-arch`
(or `training.policy_arch` in the YAML). These treat the observation window as a
`(time, features)` sequence; use the GPU for them:

```powershell
python scripts/train.py --policy-arch lstm --device cuda          # PPO + LSTM
python scripts/train.py --policy-arch transformer --device cuda   # attention
python scripts/train.py --policy-arch cnn --device cuda           # 1-D CNN
python scripts/train.py --policy-arch mlp --device cpu            # baseline (CPU is faster)
python scripts/train.py --algo dqn --device cpu                  # off-policy DQN
```

Other V3 options (set in `config/config.yaml`):

```yaml
features:
  multi_timeframe: [H1, H4]   # merge higher-TF indicators onto each base bar (causal)
  regime_window: 100          # adds a vol_regime feature (short-term vol / long-run median)
```

Compare architectures out-of-sample on 2025 in one shot:

```powershell
python scripts/compare_archs.py --archs mlp lstm transformer cnn --timesteps 200000
```

### 4. Backtest on unseen 2025 data

```powershell
python scripts/backtest.py --name ppo_gold_smoke
```

This prints total return, win rate, Sharpe, Sortino, max drawdown and profit
factor, and writes `equity_curve.png`, `trades.png`, `reward_curve.png` and
`metrics.json` to `logs/backtest/`.

### 5. Quant research tools (V2)

Validate robustness instead of trusting a single backtest:

```powershell
# Inspect a (real) data CSV: schema, gaps, parse check, split sizes
python scripts/inspect_data.py --data data/XAUUSD_M5.csv

# Walk-forward: train+evaluate across rolling/expanding out-of-sample folds
python scripts/walk_forward.py --folds 5 --mode expanding --timesteps 50000

# Hyperparameter tuning on a VALIDATION slice (2025 stays untouched)
python scripts/tune.py --trials 30 --timesteps 50000 --metric sharpe

# Evaluate one model across multiple datasets/periods
python scripts/evaluate_multi.py --name ppo_gold --glob "data/*.csv" --test-only
```

Walk-forward writes per-fold metrics (`folds.csv`), an `aggregate.json` summary
and a stitched OOS equity chart to `logs/walk_forward/`. The metric suite now
includes Sortino, **Calmar, expectancy, payoff ratio, win/loss streaks** and a
trade-distribution breakdown (exit reasons, single-trade profit share).

### 6. (Optional) Live trading — Windows + MT5 only

Live trading is **off by default** (`live.enabled: false`, `live.dry_run: true`).
After you are satisfied with backtests:

```powershell
$env:MT5_LOGIN="12345678"; $env:MT5_PASSWORD="..."; $env:MT5_SERVER="Broker-Server"
python scripts/live_trade.py --i-understand-the-risk --iterations 10
```

## Using your own (real) XAUUSD data

1. Export OHLCV from MetaTrader 5 (terminal "Save as CSV", or `copy_rates_*` via
   the Python API). The loader handles both dialects: tab-separated terminal
   exports with `<DATE>`/`<TIME>` split columns and dotted dates, and
   lowercase/epoch-second API columns.
2. Save it as `data/XAUUSD_M5.csv` (or update `data.csv_filename`/`timeframe`).
3. Run `python scripts/inspect_data.py` to confirm it parses and that the
   pre-2025 / 2025 split is non-empty.
4. Make sure it spans **both** training years and 2025.

## How leakage is prevented (walk-forward)

1. **Split by date** — everything before `2025-01-01` trains; 2025 is held out.
2. **Causal features** — every indicator uses only current/past bars.
3. **Scaler fit on train only** — the `FeatureNormalizer` is fit on training
   data and merely *applied* to the test set; the saved scaler is reused for
   live trading so there is no train/serve skew.
4. **Action vs. outcome timing** — the agent acts at `close[t]`; PnL/SL/TP are
   resolved on bar `t+1`, which it could not observe when deciding.
5. **Walk-forward folds** — each fold re-fits the normalizer on its own training
   slice and inserts an embargo `gap` before the test window.
6. **Tuning never sees the test set** — Optuna optimizes a validation slice
   carved from the *training* data, so 2025 is used exactly once, for the final
   backtest.

## How reward hacking is avoided

* Reward tracks **mark-to-market equity change**, so refusing to realize a
  losing trade still hurts (no "hold the loser" exploit).
* A **drawdown penalty** discourages high-variance all-in behaviour.
* An **overtrading penalty** plus real transaction costs make churning −EV.
* A **max-drawdown stop** terminates blown-up episodes.
* Position size is **risk-based and hard-capped**; margin checks prevent
  overleveraging.

## Configuration

All knobs live in [`config/config.yaml`](config/config.yaml) (mirrored by typed
dataclasses in [`config/config.py`](config/config.py)). Load a custom config:

```python
from config.config import Config
cfg = Config.from_yaml("config/config.yaml")
```

## Notebook

[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) walks through data →
features → training → backtest interactively (great in VSCode's notebook editor).

## Disclaimer

For research and education only. Trading leveraged instruments such as gold
carries substantial risk of loss. Nothing here is financial advice. Past (or
backtested) performance does not indicate future results.
