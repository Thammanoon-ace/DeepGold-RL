# DeepGold RL — Experiment Summary & Analysis Handoff

> **Purpose of this document.** A self-contained technical summary of the
> DeepGold RL project and its experiments, written so that another model/analyst
> can critique the methodology, interpret the results, and propose next steps.
> All numbers below are from real runs on real market data. Findings are
> reported honestly, including negative results.
>
> **Date:** 2026-05-27 · **Status:** V0–V2 complete; V3 (Phase 4A–4F) complete;
> **V3.5 (Variance Reduction & Stability) complete** (see §15). Key result: the
> **only lever that improved out-of-sample robustness was the reward *objective*
> (excess / beat-buy-and-hold)** — every complexity lever (hybrid arch, vol
> sizing, regime features, higher timeframe) made it *worse* (overfitting on
> low-SNR data). The best agent (CNN + excess reward, ensembled) is competitive
> with buy-and-hold — beats it on per-fold return 75% of the time — but still
> trails on mean return and Sharpe in a structural gold bull market. No clean,
> significant edge. The framework's value remains that it proves this honestly.

---

## 1. Objective

Train a Reinforcement-Learning agent to trade spot gold (XAUUSD) on M5 bars,
train on pre-2025 history, and evaluate strictly out-of-sample on 2025. The
explicit design goal is a **realistic research framework that does not fool
itself** — i.e., it must be able to detect when an apparent edge is actually
overfitting/variance, rather than manufacturing a fake profitable backtest.

## 2. Environment & stack

- Python 3.11.9 (venv).
- `gymnasium` 1.0.0, `stable-baselines3` 2.8.0, `torch` 2.11.0+cu128 (CUDA 12.8).
- Numerics: numpy 2.1.3, pandas 3.0.3, scikit-learn 1.8.
- Hyperparameter search: `optuna`. Data: `dukascopy-python`.
- **GPU:** NVIDIA GTX 1650 Ti, 4 GB (Turing). (Note: project originally specced
  for an RTX 3070; actual hardware is the 1650 Ti.) GPU is heavily
  underutilized — see §11.

## 3. Data

- **Source:** Dukascopy (free), real XAUUSD M5 OHLCV, BID side.
- **Range:** 2019-01-01 → 2025-12-31, **496,452 bars** (after cleaning).
- Clean: 0 duplicate timestamps, 0 `high<low`, gaps only on weekends/holidays.
- **Walk-forward split:** train = 425,585 bars (2019–2024), test = 70,847 bars
  (2025). The single train/test split is used by the basic train/backtest flow;
  walk-forward experiments build their own rolling splits over the full series.

## 4. System architecture (modules)

```
config/        Typed dataclass config (+ YAML overrides)
utils/         data_loader, feature_engineering, indicators (Phase 4B groups),
               feature_selection (corr/variance filter), normalization, viz
env/           GoldTradingEnv (Gymnasium) + TradingDataPipeline + make_env_from_frame
policies/      LSTM / Transformer / CNN feature extractors + config-driven factory
training/      PPOTrainer (PPO + DQN), callbacks
backtest/      Backtester, run_episode, metrics (Sharpe/Sortino/Calmar/expectancy/...)
validation/    TimeSeriesSplitter, WalkForwardValidator, MultiDatasetEvaluator
optimization/  Optuna PPOTuner (tunes on a validation slice, never the test set)
live_trading/  MT5 bridge + live trader skeleton (not used in experiments)
scripts/       train, backtest, walk_forward, multiseed_wf, compare_archs,
               tune, evaluate_multi, analyze_trades, inspect_data, download_dukascopy
```

## 5. Trading environment design

- **Action space:** `Discrete(4)` = Hold / Buy / Sell / Close. Single position;
  Buy/Sell open only when flat; opposite signals while in a position are no-ops
  (must Close first).
- **Observation:** flattened window of `window_size=32` bars × `n_features`
  (normalized) + 5 account-state scalars (direction, unrealized-PnL ratio,
  position-size ratio, equity ratio, holding age). Sequence extractors reshape
  the flat window back to `(time, features)` inside the network.
- **Costs:** spread (0.20), slippage (0.05), commission (7/lot round-turn,
  half on entry/exit). Buy fills at ask, sell at bid.
- **Risk controls:** risk-based position sizing (risk_fraction 2%/trade) capped
  at `max_position_lots`; margin check vs leverage (30:1); SL = 1%, TP = 2%;
  episode terminates if equity drops 40% (max-drawdown floor).
- **Reward:** `reward = scaled(Δequity) − drawdown_penalty − overtrading_penalty
  − holding_penalty`. Equity is marked-to-market, so unrealized losses hurt
  immediately (cannot dodge a penalty by not realizing a loss). Costs are
  embedded in Δequity (not double-counted). Clipped to [-10, 10].

## 6. Features (causal)

- **Core (9):** RSI(14), MACD(12,26,9) line/signal/hist, EMA(12)/EMA(26)
  distance-from-price, ATR%(14), candle return %, rolling volatility(20).
- **+ `vol_regime`** (V3): short-term vol ÷ its long-run median.
- **Multi-timeframe (V3, optional):** H1/H4 indicators merged causally onto base
  bars (a base bar only sees the last *closed* higher-TF bar).
- **Phase 4B groups (optional, registry in `utils/indicators.py`):**
  `trend` (SMA, rolling-VWAP dist, EMA spread), `momentum` (Stochastic RSI, ROC),
  `volatility` (Bollinger %B/bandwidth, historical vol), `candle` (body, wicks,
  close-pressure, candle momentum), `structure` (swing distance, breakout
  strength, trend slope via rolling OLS, S/R distance, structure-break),
  `volume` (relative volume, volume-spike z-score, liquidity-sweep depth).
  With all groups: 32 features total.

## 7. Leakage controls

1. Date split; 2025 held out.
2. All indicators causal (rolling/EWM/shift; no forward references).
3. Normalizer (RobustScaler) **fit on training data only**; saved scaler reused
   at backtest/live (no train/serve skew).
4. Action at `close[t]`; PnL/SL/TP resolved on bar `t+1`.
5. Walk-forward re-fits the normalizer **per fold** on that fold's train slice,
   with an embargo `gap = window_size` between train and test.
6. Feature selection (correlation/variance filter) fit per fold on train only.
7. Optuna tunes on a validation slice carved from training data, **never** 2025.

## 8. Validation methodology

- **TimeSeriesSplitter:** expanding (or rolling) windows, non-overlapping
  ordered test folds, embargo gap. (`--folds 5` typically yields 4 usable folds
  after the min-train constraint on this dataset.)
- **WalkForwardValidator:** trains a fresh agent per fold, evaluates on the
  following unseen window, aggregates (mean/std return, compounded, % profitable
  folds, mean Sharpe, etc.).
- **multiseed_wf.py:** repeats the *entire* walk-forward with different training
  seeds (fold structure fixed) to measure run-to-run variance.
- Metrics: total/compounded return, CAGR, Sharpe, Sortino, Calmar, max drawdown,
  win rate, profit factor, expectancy, payoff ratio, trade distribution
  (streaks, exit-reason mix, top-trade profit share).

## 9. Experiments & results (all on REAL data)

### 9.1 Architecture comparison — single 2025 OOS backtest, 150k steps each
| Arch | Return | Sharpe | Max DD | Win% | Trades |
|---|---|---|---|---|---|
| LSTM | **+86.67%** | 2.11 | 24.9% | 43.3% | 314 |
| CNN | +40.67% | 1.21 | 31.5% | 53.2% | 395 |
| Transformer | −26.40% | −1.07 | 31.9% | 48.1% | 1334 |
| MLP (baseline) | −40.03% | −2.89 | 41.2% | 45.4% | 4066 |

Observation: MLP overtrades (4066 trades) and bleeds costs; sequence models
trade selectively. **This single-run table is misleading (see 9.3–9.5).**

### 9.2 LSTM trade-distribution (2025, the +86.67% run)
314 trades; best trade = 8% of net PnL; **top-10 trades = 71% of net PnL**;
median trade −15.72 (negative); win rate 43%; avg hold ~219 bars; exits:
signal 147 / SL 100 / TP 66. Profile = trend-following (many small losses, few
big wins). Not a single-trade fluke, but profit is concentrated.

### 9.3 Walk-forward LSTM (4 folds, 100k steps/fold)
| Fold (test window) | Return |
|---|---|
| 2021-05→2022-07 | −28.1% |
| 2022-07→2023-08 | +8.6% |
| 2023-08→2024-10 | −40.2% |
| 2024-10→2025-12 | −40.0% |

Mean −24.9%/fold, **compounded −72%**, 25% profitable, mean Sharpe −1.76.
**The +86.67% did NOT survive walk-forward.** Note the 2024-10→2025-12 fold
(≈2025) gave −40% here vs +86% in the single-run — same period, different
training run.

### 9.4 Walk-forward CNN & Transformer (2 folds, 60k steps/fold)
- **CNN:** fold0 (2022-07→2024-04) +22.5%, fold1 (2024-04→2025-12) −40.1%;
  mean −8.8%, 50% profitable, ~108 trades/fold.
- **Transformer:** both folds ≈ −40% (std 0.13), ~1368 trades/fold (overtrades);
  mean −40.1%, 0% profitable. (Also likely undertrained — slowest model.)

### 9.5 CNN: core features vs ALL Phase-4B groups (5-fold req → 4 folds, 80k, n_envs=4, corr-filter 0.95 on the all-groups run)
| | F0 (21-22) | F1 (22-23) | F2 (23-24) | F3 (24-25) | Mean | %prof | Compounded |
|---|---|---|---|---|---|---|---|
| core (10 feat) | −1.1% | −6.5% | +97.7% | +84.8% | +43.7% | 50% | +238% |
| all groups (32 feat) | −1.2% | −40.0% | +79.5% | −40.1% | −0.4% | 25% | −36% |

Naive reading: "core beats all-groups; dumping all features in hurts
(feature explosion; all-groups fold1 had 1095 trades = overtrading)."
**But see 9.6 — this comparison is within the seed-noise band.**

### 9.6 Multi-seed core CNN — THE decisive experiment (4 seeds × 4 folds, 50k steps/fold; only the seed changes)
| Seed | Mean/fold | **Compounded** | %prof | Sharpe |
|---|---|---|---|---|
| 0 | +42.2% | **+204.4%** | 50% | +0.95 |
| 1000 | +7.9% | −12.1% | 25% | −0.25 |
| 2000 | −20.1% | **−61.9%** | 0% | −0.82 |
| 3000 | −8.0% | −38.8% | 50% | −0.41 |

Across seeds: compounded mean **+22.9%**, **std ±106%**, range **−62% to +204%**.
Mean-return-per-fold mean +5.5% ± 23.4.

## 10. Key findings & conclusions

1. **No reliable edge exists** with the current features/reward/architecture/
   training budget. Changing only the random seed moves compounded return from
   −62% to +204% — **variance dominates any signal**.
2. **Single backtests are actively misleading here.** The funnel:
   single backtest (+86%/+238%) → 1-seed walk-forward (+44%) → multi-seed
   walk-forward (−62%…+204%). Each layer of rigor erased the apparent profit.
3. **Even the "core beats all-groups" result (9.5) is unreliable** — core CNN's
   own across-seed compounded range (−62%…+204%) is wider than the core-vs-all
   gap, so that comparison is also measuring noise. The overtrading behavior of
   the all-groups run (1095 trades in one fold) is suggestive but unproven.
4. **Variance reduction is now the prerequisite** for any further comparison.
   Until run-to-run variance shrinks, architecture/feature experiments cannot be
   measured (the effect size is smaller than the noise).
5. The framework **succeeded at its actual goal**: it refuses to certify a fake
   edge. The agents do learn distinct behaviors (trend-following vs overtrading),
   but none generalize reliably.

## 11. Hardware / performance notes

- GPU utilization stays ~20–30% regardless of settings. Causes: tiny networks
  (model uses ~150 MB of 4 GB VRAM), and the bottleneck is **single-threaded,
  Python/pandas, per-bar environment stepping** plus the synchronous
  step→infer→step RL loop. CPU total also ~15–20% (≈ one core of 12 saturated).
- `DummyVecEnv` steps envs sequentially (one core). `SubprocVecEnv` was tried
  but **crashed with an out-of-memory / page-file error** (each of 8 spawned
  workers loads a full torch interpreter ~1–2 GB). Reverted to `DummyVecEnv`.
  For this lightweight env, subprocess IPC overhead also negates the parallelism.
- The only real throughput fix would be a **vectorized environment** (batch-step
  all envs in-process with numpy, no per-env Python loop, no IPC) — not yet done.
- Implication: faster hardware (more RAM to enable SubprocVecEnv, faster CPU
  cores) would speed *iteration*, not change conclusions.

## 12. Bugs found & fixed during experimentation

1. **`close()` method shadowed by `self.close` array** in GoldTradingEnv — the
   close-price ndarray was named `self.close`, masking the Gym `close()` method;
   `env.close()` raised `'numpy.ndarray' object is not callable`. Latent since
   V0; surfaced only when SubprocVecEnv called `close()` on workers. Fixed by
   renaming arrays to `_high/_low/_close`.
2. **Double normalization in the backtester** — it re-applied the saved scaler to
   an already-normalized test frame. Fixed: `TradingDataPipeline` now retains the
   raw featured splits and `apply_normalizer()` re-scales from raw exactly once.
3. **Correlation filter not wired into walk-forward** — it lived only in the
   single-split pipeline. Fixed: `WalkForwardValidator` now fits the selector per
   fold on the train slice.

## 13. Open questions for the analyst

1. **Is variance reduction likely to reveal a small edge, or is M5 gold
   effectively edge-less for this approach?** Suggested levers: much longer
   training (50k is low), seed-ensembling at inference, regularization, lower LR,
   smaller/simpler policy, reward shaping to curb the high-variance "few big
   trades" behavior. Which are most promising and why?
2. **Reward design:** how to reduce the heavy dependence on a few large winners
   (top-10 = 71% of PnL) and the overtrading failure mode, without destroying
   genuine trend-capture?
3. **Features (Phase 4C):** what *quantifiable, causal* structural features are
   most likely to carry real signal in gold (volatility compression, regime,
   breakout probability, etc.), given that naively adding all Phase-4B groups
   increased variance rather than performance?
4. **Methodology critique:** are 4 folds × 4 seeds adequate? Is expanding-window
   walk-forward the right protocol? Should we report a distribution of OOS
   equity curves rather than point metrics? Any leakage we missed?
5. **Is a Discrete(4) single-position action space too coarse?** Would continuous
   position sizing (enabling SAC) or a richer action space plausibly help — or
   just add variance?
6. **Regime dependence:** every architecture did well on 2023–2024 folds and
   poorly on 2021–2023 and (often) 2024–2025. Is the apparent profit just a bull
   regime in gold, and how should that change evaluation (regime-stratified
   metrics, regime-conditioned policies)?

## 14. Reproduction

```powershell
python scripts/download_dukascopy.py --start 2019-01-01 --end 2025-12-31 --timeframe M5
python scripts/inspect_data.py --data data/XAUUSD_M5.csv
python scripts/compare_archs.py --archs mlp lstm transformer cnn --timesteps 150000
python scripts/walk_forward.py --policy-arch cnn --folds 5 --timesteps 80000
python scripts/multiseed_wf.py --policy-arch cnn --seeds 4 --folds 5 --timesteps 50000
# V3.5 grid protocol (vectorized-env training, distribution + robustness):
python scripts/grid_eval.py --policy-arch cnn --seeds 3 --folds 5 --timesteps 30000 --num-envs 48
python scripts/grid_eval.py --policy-arch cnn --timeframe H1 ...     # 5F timeframe test
python scripts/grid_eval.py --policy-arch cnn_lstm ...               # hybrid
python scripts/grid_eval.py --policy-arch cnn --vol-sizing ...       # vol-targeted sizing
```

See `ROADMAP.md` for the full V0→V7 plan, `docs/V3_5_VARIANCE_REDUCTION.md` for
the V3.5 design, and `README.md` for setup. Artefacts under `logs/`
(walk_forward/, multiseed/, arch_comparison/, grid/, tensorboard/).

---

## 15. V3.5 — Variance Reduction & Stability (infrastructure + first experiments)

### 15.1 Infrastructure built (all tested)
* **Vectorized env** (`env/vectorized_env.py`) — NumPy-batched VecEnv, **proven
  byte-equivalent** to the scalar env (max reward err 7e-9; equity/obs err 0)
  and **2–21× faster** (7.3× at 32 lanes, 20.8× at 128). One process, ~1–2 GB
  RAM. This unblocked all multi-seed/grid work. (`SubprocVecEnv` was abandoned —
  it OOM'd, ~1.5–2 GB per worker, and IPC overhead negates parallelism for a
  light env.) **Bug found+fixed:** `self.close` array shadowed the Gym `close()`
  method (surfaced by SubprocVecEnv worker shutdown).
* **Evaluation rigor** — baselines (buy&hold/flat/random, `backtest/baselines.py`)
  + Robustness Score, bootstrap median CI, distribution stats
  (`validation/robustness.py`).
* **Grid runner** (`validation/grid.py`, `scripts/grid_eval.py`) — the protocol
  keystone: trains on the vectorized env, evaluates each (seed,fold) cell on the
  scalar env, fits normalizer (+optional corr filter) per fold, compares
  single-seed vs seed-ensemble vs buy-and-hold, reports the full distribution +
  Robustness Score + CI. TensorBoard per cell.
* **5A Ensemble** (`policies/ensemble.py`) — averages member action
  distributions + confidence gate.
* **5B Regime detection** (`utils/regime.py`) — causal 4-regime classifier
  (trend×vol), thresholds fit on train; features + labels for regime-stratified
  eval.
* **5C Trade-frequency control** — `min_hold_bars`, `max_trades_per_episode`,
  `trade_penalty_growth` in both envs (equivalence preserved on/off).
* **5D Feature selection** — variance + correlation pruning + **mutual-information
  ranking** (`utils/feature_selection.py`). Finding: volatility features
  (atr_pct, rolling vol) carry the most MI with forward-return magnitude;
  direction features (RSI/MACD) less.
* **5F Higher-timeframe** — `grid_eval.py --timeframe H1` resamples M5→H1/M15
  causally with correct per-TF annualization.
* **Reward/sizing redesign** — volatility-targeted (ATR-based) position sizing
  + ATR-based SL/TP (`env.volatility_target_sizing`), both envs, equivalence
  preserved when off.
* **Hybrid arch** — `CNNLSTMExtractor` (CNN front-end → LSTM), selectable via
  `--policy-arch cnn_lstm`.

### 15.2 Experiment: higher timeframe (5F) — matched M5 vs H1
CNN, 3 seeds × 4 folds, 30k steps. Single-seed std: **M5 45.3 vs H1 40.7** (≈
equal); H1 IQR tighter but H1 **ensemble worse** (robustness −57 vs M5 +1.6).
**Verdict: the "M5 too noisy → H1 better" hypothesis is NOT supported** at matched
settings. (An earlier tiny-unmatched smoke hinted H1≪M5 variance — a mirage the
protocol correctly debunked.)

### 15.3 Experiment: 3-way architecture / reward (M5, 3 seeds × 4 folds, 30k)
| Config | single median | single std | %pos | single robust | **ensemble median** | ens robust |
|---|---|---|---|---|---|---|
| **cnn (baseline)** | −3.2% | 45.3 | 42% | −73.2 | **+4.5%** | **+1.6** |
| cnn_lstm (hybrid) | −12.1% | 33.3 | 25% | −71.2 | −20.9% | −60.9 |
| cnn + vol-sizing | −40.2% | 19.2 | 8% | −95.0 | −6.2% | −31.6 |

Buy-and-hold ≈ **+27%**; every median CI straddles/below 0.

**Verdicts:**
* **Plain CNN is best of the three**; its ensemble is the only positive
  Robustness Score (+1.6, 75% profitable folds) — but its median CI [−15, +52]
  straddles 0 and it does not reliably beat buy-and-hold. No significant edge.
* **Hybrid CNN-LSTM is worse** (ensemble −20.9%) → added complexity overfits.
  Hypothesis rejected.
* **Vol-targeted sizing is worst** (single median −40%, profitable in only 8% of
  cells — it hits the max-drawdown floor far more often). Hypothesis rejected as
  implemented.

### 15.4 Experiment: EXCESS-return reward — the one lever that worked
Reward = equity change **minus a buy-and-hold benchmark's change** over the same
step (`env.reward_mode='excess'`), so the agent is rewarded only for *beating
passive holding*. CNN, 3 seeds × 4 folds, 30k.

| | single median | std | %pos | robust | **ensemble** median | ens %pos | **ens robust** |
|---|---|---|---|---|---|---|---|
| absolute reward | −3.2% | 45.3 | 42% | −73.2 | +4.5% | 75% | +1.6 |
| **excess reward** | **+8.2%** | 46.6 | 58% | −59.5 | **+9.9%** | **100%** | **+16.7** |

Excess reward is the **best config found** — every metric improved, the ensemble
is profitable in **100%** of folds and posts the highest Robustness Score (+16.7).
**Per-fold risk-adjusted vs buy-and-hold** (ensemble): the agent **beats BH's
return in 3 of 4 folds (75%)**, but its mean return is lower (+17.6% vs +27.0%)
because it badly underperforms the single strongest-trend fold (fold 3: +11% vs
BH +55%), and **BH still wins on Sharpe** (1.33 vs 0.94). So: competitive with
buy-and-hold, even beating it most folds — but not a clean win on raw return or
Sharpe in a bull market.

### 15.5 Experiment: regime features (5B) — rejected
Hypothesis: adding causal regime signals (`regime_trend`, `regime_vol`) to the
observation lets the agent ride strong trends and close the fold-3 gap. Result
(excess + regime vs excess alone):

| ensemble metric | excess | excess+regime |
|---|---|---|
| median return | +9.9% | **−10.7%** |
| robustness | +16.7 | **−62.3** |
| % profitable folds | 100% | 25% |

It made **every fold worse** (fold 2 collapsed +44.6% → −40.1%) and lost to
buy-and-hold on every dimension. **Verdict: rejected** — adding the two regime
features was *another instance of feature explosion* (more inputs → overfitting
on low-SNR data), exactly the failure mode seen with the Phase-4B groups.

### 15.6 Robust cross-cutting findings (V3.5)
0. **The ONLY lever that improved results was the reward *objective* (excess /
   beat-buy-and-hold).** Every *complexity* lever — bigger arch (CNN-LSTM), more
   features (all 4B groups; regime), vol-targeted sizing — made results WORSE.
   This is the central V3.5 result.
1. **Ensembling reliably cuts variance** on every config (CNN 45→21, CNN-LSTM
   33→5, vol-sizing 19→6) — but **low variance is worthless when it stabilizes a
   *loss*** (CNN-LSTM/vol-sizing ensembles are stably negative). Variance
   reduction only helped where the base had a sliver of signal (plain CNN).
2. **Complexity and "stability" levers both hurt returns** here — consistent with
   the low-SNR / efficient-market premise.
3. **Buy-and-hold (+27%) is not beaten on raw mean return or Sharpe** by any
   config. The excess-reward ensemble does beat BH's *per-fold return in 75% of
   folds*, but loses the one strong-trend fold so badly that its mean and Sharpe
   trail BH. In a sustained bull market, passive holding is a very hard benchmark.
4. **Regime dependence persists** (one seed/fold on the 2023–24 trend prints
   +85%; others hit −40%). The only large profits come from trend folds.
5. **Ensembling reliably cuts variance** on every config (~2–7×) — but is only
   *useful* when the base policy has a sliver of edge (it improved plain-CNN and
   excess; it just stabilised the losses of CNN-LSTM/vol-sizing/regime).

### 15.7 Updated open questions for the analyst
* The reward objective is the only thing that moved the needle. Should we push it
  further — e.g. reward the **information ratio vs buy-and-hold** directly, or a
  drawdown-aware excess (so the agent is rewarded for *risk-adjusted* outperformance,
  the dimension BH still wins)?
* The agent is long-biased and tracks gold; it underperforms specifically in the
  strongest uptrend (where it should just hold/lever up). Is this a
  **credit-assignment / exploration** problem (it never learns to "ride") rather
  than a features/architecture one — given that *every* feature/architecture
  addition tried so far made things worse?
* Every complexity lever overfits on this low-SNR data. Is the honest conclusion
  that **M5 XAUUSD 2019–2025 has no robust RL edge beyond buy-and-hold**, and the
  research value is the framework that proves it — not a profitable bot?
* If pursuing further: is a **different market/regime** (ranging instrument, or a
  bear/sideways period where BH is weak) where an RL timing edge could actually
  show value, a better testbed than a structural gold bull market?
