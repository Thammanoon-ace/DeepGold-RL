"""Kalman-filter pairs trading — classical stat-arb baseline (group 3-B).

A non-RL, market-neutral strategy on a cointegrated metals pair (default
XAU/XAG). A Kalman filter tracks the time-varying hedge ratio between the two
legs; the filter's one-step forecast error (the "spread") mean-reverts around
zero, and we trade its z-score.

Why this exists: the RL-on-gold-directional project reached a rigorous negative
result (docs/NEGATIVE_RESULT.md) — the agent cannot beat a structural-bull
buy-and-hold. This script tests a *different formulation*: market-neutral
mean-reversion, where beating buy-and-hold on raw return is not the bar (the
strategy is dollar-neutral and uncorrelated with the gold trend). The question
is whether the XAU/XAG spread carries a tradeable mean-reversion edge after
realistic costs.

Method (Chan 2013, "Algorithmic Trading", ch. on Kalman pairs):
  state  = [intercept alpha_t, hedge-ratio beta_t]  (random walk)
  obs    = y_t = [1, x_t] . state + eps_t           (y = XAU, x = XAG)
  spread = forecast error e_t = y_t - [1, x_t] . state_pred
  z_t    = e_t / sqrt(forecast variance Q_t)
  trade  = enter the spread against its sign when |z_t| > entry_z,
           exit when |z_t| < exit_z. Dollar-neutral; hedge ratio fixed at
           entry beta and held to exit (no per-bar rebalance => realistic costs).

The Kalman filter is inherently *causal* (state at t uses only data <= t), so
there is no look-ahead. We still split train (cointegration check + nothing
tuned on test) vs test (2025) to mirror the project's walk-forward discipline.
Filter hyper-params (delta, R) use Chan's standard literature values and are
NOT tuned on the test set.

Usage:
    python scripts/kalman_pairs.py --timeframe H1
    python scripts/kalman_pairs.py --timeframe D1 --entry-z 1.5 --exit-z 0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.config import BARS_PER_YEAR, Config
from utils.data_loader import HistoricalDataLoader

# Chan (2013) standard Kalman pairs hyper-parameters — fixed, NOT tuned on test.
DELTA = 1e-4                  # state transition noise scale
R_OBS = 1e-3                  # observation noise variance


def _find_csv(sym: str, data_dir: Path):
    for tf in ("M5", "H1", "M15", "H4", "D1"):
        p = data_dir / f"{sym}_{tf}.csv"
        if p.exists():
            return p
    raise SystemExit(f"No CSV for {sym} under {data_dir}/")


def load_pair(tf: str, cfg: Config, y_sym: str, x_sym: str):
    """Load the y and x legs, resample to tf, align on common timestamps.

    y = the dependent / mean-reverting leg (the one the cointegration test
    regressed); x = the regressor. Loads whatever <SYM>_<TF>.csv exists.
    """
    loader = HistoricalDataLoader(cfg.data)
    data_dir = cfg.paths.data
    yp = loader.load(str(_find_csv(y_sym, data_dir)))
    xp = loader.load(str(_find_csv(x_sym, data_dir)))
    if tf != "M5":
        yp = loader.resample(yp, tf)
        xp = loader.resample(xp, tf)
    df = pd.DataFrame({"y": yp["close"], "x": xp["close"]}).dropna()
    return df


def adf_pvalue(series: np.ndarray) -> float:
    """Augmented Dickey-Fuller p-value via statsmodels if available; else NaN."""
    try:
        from statsmodels.tsa.stattools import adfuller
        return float(adfuller(series, maxlag=1, autolag=None)[1])
    except Exception:
        return float("nan")


def kalman_hedge(y: np.ndarray, x: np.ndarray):
    """Online Kalman filter for [alpha, beta]. Returns per-bar spread e_t and
    its std sqrt(Q_t), plus the beta path. Causal: index t uses data <= t."""
    n = len(y)
    delta = DELTA
    Vw = delta / (1.0 - delta) * np.eye(2)   # state transition covariance
    Ve = R_OBS                               # observation noise variance

    beta = np.zeros((2, n))                  # [alpha; beta] over time
    P = np.zeros((2, 2))                     # state covariance
    R = np.zeros((2, 2))
    state = np.zeros(2)

    e = np.full(n, np.nan)                   # forecast error (spread)
    q = np.full(n, np.nan)                   # forecast error std

    for t in range(n):
        if t > 0:
            R = P + Vw                       # predicted state covariance
        H = np.array([1.0, x[t]])            # observation matrix row
        yhat = H @ state                     # predicted observation
        et = y[t] - yhat                     # forecast error (spread)
        Qt = H @ R @ H + Ve                  # forecast error variance
        e[t] = et
        q[t] = np.sqrt(Qt)
        K = (R @ H) / Qt                     # Kalman gain
        state = state + K * et               # state update
        P = R - np.outer(K, H @ R)           # covariance update
        beta[:, t] = state
    return e, q, beta[1], beta[0]


def backtest(df: pd.DataFrame, entry_z: float, exit_z: float,
             cost_bps_xau: float, cost_bps_xag: float, bars_per_year: int):
    """Trade the Kalman spread. Dollar-neutral, hedge fixed at entry beta,
    held to exit. Returns dict of metrics + equity curve."""
    y = df["y"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    e, q, beta, alpha = kalman_hedge(y, x)
    z = e / q

    n = len(y)
    pos = 0                       # -1 short spread, +1 long spread, 0 flat
    beta_held = 0.0
    pnl = np.zeros(n)             # per-bar PnL in fraction-of-capital terms
    n_trades = 0
    # warm-up: skip first 30 bars while the filter converges
    warm = 30
    for t in range(warm, n):
        # PnL from holding the previous bar's position (fixed hedge at entry).
        if pos != 0:
            # dollar-neutral spread return: long 1 unit y, short beta_held units x,
            # normalised so the gross exposure is ~1 (|1| + |beta*x/y|).
            dy = (y[t] - y[t - 1]) / y[t - 1]
            dx = (x[t] - x[t - 1]) / x[t - 1]
            gross = 1.0 + abs(beta_held) * x[t - 1] / y[t - 1]
            spread_ret = (dy - beta_held * x[t - 1] / y[t - 1] * dx) / gross
            pnl[t] += pos * spread_ret

        zt = z[t]
        if np.isnan(zt):
            continue
        # Exit logic. exit_z == 0 => exit when the spread reverts past its mean
        # (the held position's target sign flips, i.e. z crosses 0). exit_z > 0
        # => exit inside the band |z| < exit_z.
        if pos != 0:
            reverted = (zt * pos >= 0) if exit_z == 0.0 else (abs(zt) < exit_z)
            # pos = -sign(z_entry): long spread (pos=+1) entered when z<0, target
            # is z up to 0 => exit when z >= 0 => zt*pos = zt*(+1) >= 0. Symmetric.
            if reverted:
                cost = (cost_bps_xau + cost_bps_xag) / 1e4
                pnl[t] -= cost
                pos = 0
                n_trades += 1
        # Entry logic (only when flat)
        if pos == 0 and abs(zt) > entry_z:
            pos = -int(np.sign(zt))           # spread above forecast -> short it
            beta_held = beta[t]
            cost = (cost_bps_xau + cost_bps_xag) / 1e4
            pnl[t] -= cost
            n_trades += 1

    equity = np.cumprod(1.0 + pnl)
    total_ret = (equity[-1] - 1.0) * 100.0
    rets = pnl[warm:]
    ann = np.sqrt(bars_per_year)
    sharpe = (rets.mean() / rets.std() * ann) if rets.std() > 1e-12 else 0.0
    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.max((running_max - equity) / running_max)) * 100.0
    return {
        "total_return_pct": total_ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "n_trades": n_trades,
        "equity": equity,
        "z": z, "beta": beta,
        "n_bars": n,
        "pct_in_market": float(np.mean(pnl[warm:] != 0)) * 100.0,
    }


def bh_metrics(prices: np.ndarray, bars_per_year: int):
    rets = np.diff(prices) / prices[:-1]
    eq = np.cumprod(1.0 + rets)
    ann = np.sqrt(bars_per_year)
    sharpe = rets.mean() / rets.std() * ann if rets.std() > 1e-12 else 0.0
    rm = np.maximum.accumulate(eq)
    dd = float(np.max((rm - eq) / rm)) * 100.0
    return (eq[-1] - 1.0) * 100.0, sharpe, dd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--y-symbol", default="XAUUSD", help="Dependent / mean-reverting leg.")
    p.add_argument("--x-symbol", default="XAGUSD", help="Regressor leg.")
    p.add_argument("--timeframe", default="H1", choices=["M5", "M15", "H1", "H4", "D1"])
    p.add_argument("--entry-z", type=float, default=1.0)
    p.add_argument("--exit-z", type=float, default=0.0)
    p.add_argument("--cost-bps-y", type=float, default=3.0,
                   help="Round-turn cost (bps) on the y leg.")
    p.add_argument("--cost-bps-x", type=float, default=5.0,
                   help="Round-turn cost (bps) on the x leg.")
    p.add_argument("--test-start", default="2025-01-01")
    args = p.parse_args()

    cfg = Config.from_yaml("config/config.yaml") if Path("config/config.yaml").exists() else Config()
    bpy = BARS_PER_YEAR.get(args.timeframe, 24 * 252)

    df = load_pair(args.timeframe, cfg, args.y_symbol, args.x_symbol)
    print(f"Pair {args.y_symbol}/{args.x_symbol} @ {args.timeframe}: "
          f"{len(df)} aligned bars ({df.index.min()} -> {df.index.max()})")

    # Cointegration check on the TRAIN slice only.
    test_start = pd.Timestamp(args.test_start)
    train = df[df.index < test_start]
    test = df[df.index >= test_start]
    # static OLS hedge on train, ADF on the residual
    bt = np.polyfit(train["x"], train["y"], 1)  # [slope, intercept]
    resid = train["y"].to_numpy() - (bt[0] * train["x"].to_numpy() + bt[1])
    adf_p = adf_pvalue(resid)
    print(f"\nCointegration (train, static OLS beta={bt[0]:.3f}): "
          f"ADF p-value = {adf_p:.4f} "
          f"({'cointegrated' if adf_p < 0.05 else 'NOT cointegrated at 5%'})")

    # Full-sample + test-only backtest. BH benchmark = the y leg.
    print(f"\n=== Kalman pairs backtest (entry_z={args.entry_z}, exit_z={args.exit_z}, "
          f"cost y={args.cost_bps_y}bps x={args.cost_bps_x}bps) ===")
    for label, sub in (("FULL", df), ("TEST (>=test_start)", test)):
        if len(sub) < 100:
            continue
        m = backtest(sub, args.entry_z, args.exit_z,
                     args.cost_bps_y, args.cost_bps_x, bpy)
        bh_y = bh_metrics(sub["y"].to_numpy(), bpy)
        print(f"\n  [{label}]  ({len(sub)} bars)")
        print(f"    Kalman pairs    : ret {m['total_return_pct']:+7.2f}%  "
              f"Sharpe {m['sharpe']:6.3f}  maxDD {m['max_dd_pct']:5.2f}%  "
              f"trades {m['n_trades']:4d}  in-mkt {m['pct_in_market']:.0f}%")
        print(f"    BH {args.y_symbol:<7}: ret {bh_y[0]:+7.2f}%  "
              f"Sharpe {bh_y[1]:6.3f}  maxDD {bh_y[2]:5.2f}%")
        verdict = ("BEATS BH on Sharpe" if m["sharpe"] > bh_y[1]
                   else "loses to BH on Sharpe")
        print(f"    -> {verdict} ({m['sharpe']:.2f} vs {bh_y[1]:.2f})")


if __name__ == "__main__":
    main()
