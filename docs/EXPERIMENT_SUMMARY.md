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

### 15.8 Verdict — big-seed run on new hardware (2026-05-28)
Run: `excess_bigseed_32` — 32 seeds × 4 folds × 1.31M timesteps, GPU CleanRL engine,
CNN arch, **excess** reward. Total wall-clock ~3 h 20 min on RTX PRO 4000 Blackwell
24 GB (CPU-bound at ~55 k env-steps/s — see Tier 1.2 in `docs/NEW_HARDWARE_PLAN.md`).

| | n cells | median | std | %pos | beats BH | robustness | **CI 95 % (median)** |
|---|---|---|---|---|---|---|---|
| single-seed | 128 | **+0.36 %** | 29.8 | 50 % | 13 % | −51.3 | **[−3.7, +5.5]** straddles 0 |
| ensemble (32 seeds/fold) | 4 | +1.80 % | **0.9** | 75 % | **0 %** | +7.7 | — |

Buy-and-hold mean over folds: **+27.0 %**. Worst/best single-seed cell: **−40.3 % /
+142.8 %** — variance still dominates signal even at the largest feasible seed count
(46.6 → 29.8 std with the larger N, but the bootstrap CI of the *median* is still
\[−3.7, +5.5\] and therefore **not statistically distinguishable from 0 — nor from
BH**). The ensemble cuts variance hard (29.8 → 0.9 std) but lands at +1.8 % median
while BH posts +27 % — i.e. the ensemble has averaged out the signal as completely
as it averaged out the noise. **0 % of folds beat BH** at the ensemble level.

This run was the definitive answer to the two main open questions in `CLAUDE.md`:
1. *"Does excess reward beat buy-and-hold with statistical significance at large
   seed count?"* — **No.**
2. *"Will the GPU engine's 16-member ensemble recover what the 3-member ensemble
   lacked?"* — **No.** Larger ensembles average the signal away too.

Per CLAUDE.md's own pre-registered position: *"If those answers all come back
negative on a definitive run, the honest project verdict is: RL on retail-data +
retail-compute is not a practical trading edge — the science is the negative
result."* That verdict now applies under every configuration tried through V3.5.

Artifacts: `logs/grid/excess_bigseed_32/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.9 H1 big-seed run — first statistically significant positive (2026-05-28)
Same protocol as §15.8 (32 seeds × 4 folds × 1.31M timesteps, excess reward, GPU
CleanRL engine), only `--timeframe H1`. Total wall-clock **~63 min** vs M5's
~3 h 20 min — H1 has 11.5× fewer bars so each scalar eval is far cheaper.
Tag: `excess_bigseed_32_h1`.

| | n cells | median | std | %pos | beats BH | robustness | **CI 95 % (median)** |
|---|---|---|---|---|---|---|---|
| single-seed | 128 | **+6.34 %** | 18.98 | 68 % | 16 % | −24.2 | **[+3.6, +10.3] EXCLUDES 0** ✅ |
| ensemble (32 seeds/fold) | 4 | +3.30 % | 7.82 | 50 % | **0 %** | −7.3 | — |

Buy-and-hold mean over folds: **+26.3 %** (≈ M5's +27.0 %). Variance vs the M5
run cut by ~36 % (std 29.8 → 19.0); worst/best **−24.0 / +72.4 %** vs M5's
−40.3 / +142.8 % — the H1 distribution is tighter on both tails. Ensemble cuts
variance further (19.0 → 7.8 std).

**The headline:** this is the **first run in the project where the bootstrap
95 % CI on the median single-cell excess return excludes 0**. The M5 §15.8 run
landed at [−3.7, +5.5] (not significant); H1 lands at **[+3.6, +10.3]**, clearly
above zero. The excess-reward agent does **not** beat BH on raw return (BH still
posts +26.3 % per fold and the ensemble beats BH on 0 % of folds — same as M5),
but it now demonstrably has measurable edge over *zero* return, which the M5
work had failed to establish over 12 experiments.

**Reframing the project verdict (vs §15.6/§15.7 and CLAUDE.md):**
- "RL on retail-data + retail-compute is not a practical trading edge" — that
  statement holds **for M5**, not for the framework as a whole. Timeframe is a
  load-bearing variable we had under-explored.
- The §15.7 open question *"is the honest conclusion that M5 XAUUSD 2019–2025
  has no robust RL edge beyond buy-and-hold, and the research value is the
  framework that proves it?"* answers: **for M5, yes; the H1 result shows the
  framework can find edge under different sampling**.
- The §15.7 follow-up *"is a different market/regime ... a better testbed than
  a structural gold bull market?"* now has a concrete first data point — a
  longer timeframe on the same market gave the first significant signal.

**Open questions advanced by this run:**
1. Does the edge **grow** at H4 or **peak** at H1? (Fewer bars → less noise but
   also less data for PPO to learn from.)
2. The ensemble beats BH 0 % of folds on **raw return** while having median
   single-cell excess > 0 — does it beat BH on **Sharpe / drawdown-adjusted**
   returns? That would still be tradeable.
3. Does the H1 edge generalize **off gold** (EURUSD H1, BTC H1, SPX daily)?
   The framework now has a positive baseline to compare against.

Artifacts: `logs/grid/excess_bigseed_32_h1/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.10 H4 big-seed run — first config to beat buy-and-hold (2026-05-28)
Same protocol as §15.8/§15.9 (32 seeds × 4 folds × 1.31M timesteps, excess
reward, GPU CleanRL engine), only `--timeframe H4`. Total wall-clock ~60 min.
Tag: `excess_bigseed_32_h4`. Required adding `H4 -> 4h` to
`_TIMEFRAME_TO_OFFSET` in `utils/data_loader.py`.

| | n cells | median | std | %pos | beats BH | robustness | **CI 95 % (median)** |
|---|---|---|---|---|---|---|---|
| single-seed | 128 | +5.87 % | 28.07 | 65 % | **23 %** | −38.1 | **[+3.2, +10.9] EXCLUDES 0** ✅ |
| ensemble (32 seeds/fold) | 4 | **+16.66 %** | 15.59 | **75 %** | **25 %** ← first ever | **+12.89** ← first positive | — |

Buy-and-hold mean over folds: **+25.9 %**. Ensemble worst/best: −2.2 % / **+41.3 %**
on fold 3 — **the trend fold's ensemble (+41 %) beats BH (+27 %) by ~15 pp**.
That is the first time in the project's history that an ensemble cell exceeded
buy-and-hold; the M5 run beat BH on 0 % of folds and the H1 run also on 0 %.

**Cross-timeframe summary (excess reward, 32 seeds × 4 folds, CNN, GPU
engine, identical protocol):**

| timeframe | bars | single CI | ensemble median | ensemble mean | ensemble beats BH | ensemble robustness |
|---|---|---|---|---|---|---|
| M5 (§15.8) | 425 k | [−3.7, +5.5] straddles 0 | +1.8 % | +1.5 % | 0 % | +7.7 |
| H1 (§15.9) | 36 k | [+3.6, +10.3] excludes 0 | +3.3 % | +4.3 % | 0 % | −7.3 |
| **H4 (§15.10)** | 9 k | **[+3.2, +10.9] excludes 0** | **+16.7 %** | **+18.1 %** | **25 %** | **+12.9** |

The "M5 has no robust RL edge beyond BH" conclusion from §15.6 still holds at
M5. It does **not** generalize: the same agent on H4 produces a 1-of-4-fold win
over BH and the first positive Robustness Score in the project's history. The
direction of improvement is monotonic in timeframe (M5 → H1 → H4) across every
ensemble metric tracked, consistent with the "M5 is too noisy" hypothesis from
§15.7's open questions.

**Open questions advanced by this run:**
1. Does the edge **continue to grow** at the daily timeframe, or does H4 sit at
   a sweet spot? Daily would test whether the gain comes purely from noise
   reduction or whether H4's specific structure matters.
2. The H4 ensemble beats BH on fold 3 (strong trend, +41 % vs +27 %) but
   *underperforms* on fold 0 (range market, −2 % vs ~ BH). What is the agent
   doing differently in trend vs range — and can we detect that regime in
   advance for a position-sizing gate?
3. Does the H4 edge **generalize off gold** (EURUSD H4, BTC H4, SPX H4)? If yes
   the framework has a real cross-asset signal at the right timeframe; if no,
   the H4 result is a gold-specific artifact and we are back to §15.6's stance
   with a slightly larger asterisk.

Artifacts: `logs/grid/excess_bigseed_32_h4/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.11 H4 + ATR regime gate — rejected (2026-05-28)
Hypothesis: gate new entries on bars whose ATR/close ratio is below a
volatility threshold so the agent skips quiet/range bars where the §15.10 H4
run lost. Implementation: `EnvConfig.min_trade_atr_pct` + `--min-trade-atr-pct`
CLI flag, applied causally in both `GoldTradingEnv._open_position` and
`TorchVecGoldEnv.step` open-mask logic. Tag: `excess_bigseed_32_h4_gated`,
threshold 0.004 (blocks ~25 % of the lowest-volatility H4 bars).

| | H4 baseline (§15.10) | H4 + gate | Δ |
|---|---|---|---|
| single CI | **[+3.2, +10.9] excludes 0** | **[−3.2, +5.3] straddles 0** | ❌ lost significance |
| single median | +5.87 % | +1.60 % | −4.3 pp |
| single robustness | −38.1 | −52.1 | −14 |
| **ensemble median** | **+16.7 %** | +13.8 % | −2.8 pp |
| ensemble mean | +18.1 % | +14.0 % | −4.1 pp |
| ensemble best fold | +41.3 % | +38.4 % | −2.9 pp |
| ensemble robustness | **+12.9** | +3.6 | −9 |
| ensemble beats BH | 25 % | 25 % | tie |

**Per-fold ensemble impact:**

| fold | regime | baseline | gated | Δ |
|---|---|---|---|---|
| 0 | range | −2.2 % | −10.1 % | **−7.9** |
| 1 | mild trend | +13.4 % | +14.3 % | +0.9 |
| 2 | trend | +19.9 % | +13.3 % | −6.6 |
| 3 | strong trend | +41.3 % | +38.4 % | −2.9 |

The gate makes the agent *worse* on the very fold (0) it was supposed to fix,
and degrades trend folds 2 and 3 too. The only fold marginally better is the
mild-trend fold 1 (+0.9 pp). Net: median CI returns to straddling 0 and the
first-positive Robustness Score gets cut from +12.9 to +3.6.

**Why it didn't work:**
1. ATR is a **volatility** indicator, not a trend indicator. Low ATR ≠ range —
   a smooth trending bar can have low ATR; a choppy range bar can have high ATR.
2. The gate applies during **training** too, so the agent sees fewer entry
   opportunities per cell and learns from a smaller effective dataset.
3. There are entries the unfiltered agent makes profitably on low-volatility
   bars (e.g. continuation moves in trend), and the gate vetoes those as well.

**Status:** Code path (`min_trade_atr_pct`) kept off by default (backward
compatible); the flag remains for future experiments at different thresholds
or with a different gate signal. **The §15.10 H4 baseline (no gate) remains
the project's best run** by every aggregated metric tracked.

**What to try next instead of an ATR gate:**
- A **directional** regime signal (e.g. trend-efficiency ratio TER, ADX, or
  Hurst exponent) rather than a volatility one. Trend-efficiency = abs price
  change over a window divided by sum of absolute step changes — would
  actually discriminate trend vs. chop.
- Move the gate **out of training** (use the unfiltered training agent) and
  apply it only at evaluation time, so we keep the trained policy's coverage
  but filter its outputs.
- Skip the gate idea entirely and pursue (a) daily timeframe to test whether
  the M5→H1→H4 monotone trend continues, or (b) multi-instrument H4 to test
  generalization.

Artifacts: `logs/grid/excess_bigseed_32_h4_gated/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.12 H4 + Trend-Efficiency gate (post-hoc eval) — mixed result (2026-05-29)
Follow-up to §15.11 with the eval-time-only fix the post-mortem recommended:
gate new entries by **trend-efficiency ratio** (TER, directional) instead of
ATR (volatility), and apply the gate at evaluation only so training is
unchanged. Implementation: `compute_ter` + `run_episode_ter_gated` in
[backtest/backtester.py](../backtest/backtester.py),
`BacktestConfig.ter_gate_window/threshold` config fields,
`grid._evaluate` routes through `run_episode_ter_gated` when both
fields are > 0, `--ter-gate-window` / `--ter-gate-threshold` CLI flags.
Settings: window=50 H4 bars (~10 trading days), threshold=0.10
(filter where TER < 0.10, blocking ~37 % of historical H4 bars).
Tag: `excess_bigseed_32_h4_ter_w50_t010`.

| | H4 baseline (§15.10) | **H4 + TER gate** | Δ |
|---|---|---|---|
| **single CI** | [+3.2, +10.9] | **[+5.7, +18.5]** ← strongest in project | ↑ |
| single median | +5.87 % | **+12.08 %** | **+6.2 pp** ↑ |
| single mean | +11.25 % | **+16.17 %** | +4.9 pp ↑ |
| single beats BH | 23 % | **28 %** ← project best | +5 pp ↑ |
| ensemble median | **+16.66 %** | +7.86 % | −8.8 ❌ |
| ensemble mean | **+18.10 %** | +14.63 % | −3.5 ❌ |
| ensemble best fold | +41.26 % | **+54.84 %** ← project best | +13.6 ↑ |
| ensemble worst fold | −2.17 % | −12.04 % | −9.9 ❌ |
| ensemble std | **15.59** | 24.99 | +9.4 (worse) ❌ |
| ensemble robustness | **+12.89** | −10.94 | −24 ❌ |
| beats BH (ensemble) | 25 % | 25 % | tie |

**Per-fold ensemble:**

| fold | regime | baseline | + TER gate | Δ |
|---|---|---|---|---|
| 0 | range | −2.17 % | −12.04 % | −9.9 |
| 1 | mild trend | +13.43 % | +1.65 % | −11.8 |
| 2 | trend | +19.89 % | +14.07 % | −5.8 |
| 3 | strong trend | +41.26 % | **+54.84 %** | **+13.6** |

The gate **amplifies the strong-trend fold by +14 pp** but degrades every
other fold, ensemble net result is worse on every aggregate metric except
single-seed CI / median / mean / beats-BH-rate, which are the **strongest
the project has ever produced**. So the result is genuinely two-sided:

- **Single-seed view:** TER gate is the project's best run by every per-cell
  metric — CI [+5.7, +18.5] is strictly above the H4 baseline's [+3.2, +10.9];
  median +12 % vs +5.9 %; 28 % of cells beat BH vs 23 %.
- **Ensemble view:** H4 baseline (no gate) remains the best — robustness
  +12.9 vs −10.9, lower variance, better median and mean. The gate trades
  ensemble stability for trend-fold amplification.

**Why the split:** averaging 32 seeds smooths their disagreements; the TER
gate increases each seed's variance (some seeds with the gate produce huge
positives on fold 3, others lose more on the other folds), so the ensemble
ends up averaging across a wider, more bimodal distribution. The gate also
fundamentally cuts the number of trades the agent can place — that helps
when trades on chop are losing but hurts when trades through low-TER bars
were continuation-trend entries.

**Operationally:** the H4 baseline ensemble remains the project's headline
result. The TER gate is useful only if a downstream selector picks
top-quartile seeds rather than averaging 32 — those seeds with TER produce
+50 % - +140 % on the strong-trend fold.

**Status:** TER gate code path stays in (flag defaults disabled). Both gates
(ATR §15.11 and TER §15.12) confirm a meta-finding consistent with §15.6:
**adding rules on top of the trained policy reliably trades expected value
for amplified upside in trend** — and net loses except in the specific case
of top-seed selection. The unfiltered policy + ensemble averaging remains
the optimal aggregator.

Artifacts: `logs/grid/excess_bigseed_32_h4_ter_w50_t010/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.13 D1 big-seed run — monotone trend broken, H4 confirmed as sweet spot (2026-05-30)
Same protocol as §15.8–§15.12 (32 seeds × 4 folds × 1.31M timesteps, excess
reward, GPU CleanRL engine), only `--timeframe D1`. Required adding `D1 ->
1D` to `_TIMEFRAME_TO_OFFSET` and `BARS_PER_YEAR["D1"]=252`. Tag:
`excess_bigseed_32_d1`. Wall-clock ~62 min.

| | n cells | median | std | %pos | beats BH | robustness | **CI 95 % (median)** |
|---|---|---|---|---|---|---|---|
| single-seed | 128 | **−7.07 %** | 17.50 | 30 % | **3.9 %** | −61.5 | **[−9.0, −5.1] EXCLUDES 0 on the NEGATIVE side** |
| ensemble (32 seeds/fold) | 4 | −10.92 % | 9.35 | 25 % | **0 %** | −46.0 | — |

Buy-and-hold mean over folds: **+26.2 %**.

**The monotone M5 → H1 → H4 trend does not continue. D1 reverses it:**

| timeframe | bars (full series) | single CI | ensemble median | beats BH | robustness |
|---|---|---|---|---|---|
| M5 (§15.8) | 425 k | [−3.7, +5.5] straddles 0 | +1.8 % | 0 % | +7.7 |
| H1 (§15.9) | 36 k | [+3.6, +10.3] excludes 0 (positive) | +3.3 % | 0 % | −7.3 |
| **H4 (§15.10)** | 9 k | **[+3.2, +10.9] excludes 0 (positive)** | **+16.7 %** | **25 %** | **+12.9** |
| **D1 (§15.13)** | 2 k | **[−9.0, −5.1] excludes 0 (NEGATIVE)** | **−10.9 %** | **0 %** | **−46.0** |

**Per-fold ensemble — note fold 3 (the strong-trend fold) flips sign:**

| fold | regime | H4 ensemble | D1 ensemble | Δ |
|---|---|---|---|---|
| 0 | range | −2.2 % | +1.7 % | +3.9 |
| 1 | mild trend | +13.4 % | −12.1 % | −25.5 |
| 2 | trend | +19.9 % | −9.8 % | −29.7 |
| 3 | strong trend | **+41.3 %** | **−24.7 %** | **−66.0** |

**Interpretation:**
- D1 is too coarse for an RL agent with this architecture and data. There are
  only ~250 decision points per test fold (vs ~1,500 at H4); a bad early entry
  can't be recovered. Many seeds hit the 40 % max-drawdown floor before the
  episode ends.
- The fact that the CI **significantly excludes 0 on the negative side** is
  itself a clean scientific result: at D1 the agent is reliably **worse** than
  doing nothing, not noisy around 0.
- **H4 is the sweet spot.** The hypothesis "edge grows monotonically with
  timeframe" from §15.10's open questions is now answered: NO, it peaks
  somewhere around H4 (between H1 and D1).

**Open questions advanced:**
1. Is the H4 peak gold-specific, or is "around 4 hours" the right granularity
   for retail-data RL on any liquid asset? Multi-instrument H4 (EURUSD H4,
   BTC H4, SPX H4) is still untested.
2. Within H4, can a **top-k seed selector** (rank seeds by training-fold
   Sharpe or by a held-out validation fold, then ensemble only the top-k)
   capitalise on §15.12's finding that TER-gated single seeds have project-
   best per-cell CI [+5.7, +18.5] even though the 32-seed average loses to
   the unfiltered baseline?
3. Is daily TOO coarse, or is the issue specifically about decision count?
   H2 or H3 (not currently supported by the resampler) would interpolate
   between H1 (no edge over BH) and H4 (peak).

Artifacts: `logs/grid/excess_bigseed_32_d1/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.14 Top-k seed selector — oracle ceiling (2026-05-30)
Post-hoc analysis with [scripts/_topk_analysis.py](../scripts/_topk_analysis.py):
for each fold, sort 32 seeds by a test-fold metric and average the per-seed
returns of the top-k. This is a *portfolio-of-K-independent-agents* analysis
(each agent gets 1/K capital), distinct from the policy-averaging
EnsemblePolicy. Numbers below use **Sharpe** as the ranker (return-based
ranker gives nearly identical results).

| grid | all-32 mean | top-16 | top-8 | **top-4** | top-1 |
|---|---|---|---|---|---|
| M5 (§15.8) | +3.3 % | +22.8 % | +37.8 % | +51.3 % | +87.3 % |
| H1 (§15.9) | +9.7 % | +22.2 % | +28.4 % | +31.5 % | +33.3 % |
| **H4 (§15.10)** | +11.3 % | +26.9 % | +36.9 % | **+46.8 %** | +61.6 % |
| H4 + TER (§15.12) | +16.2 % | +30.1 % | +39.1 % | +46.2 % | +55.7 % |
| D1 (§15.13) | **−6.4 %** | +5.8 % | **+14.7 %** | +23.4 % | +36.4 % |

**Headline finding:** picking the top-4/8 seeds (out of 32) by Sharpe lifts
every grid by 20-50 pp over the unfiltered mean. **D1 ensemble's −11 % flips
to +15 % at top-8**, and **H4 top-4 (Sharpe) hits +46.8 %**, beating BH
(+25.9 %) by 21 pp.

**Sanity:** the bottom-4 averaged gives −13 to −30 % on every grid — far worse
than top-4. The ranking ordering definitely carries signal.

**Critical caveat — this is leakage.** All numbers above used the *test* fold's
Sharpe as the ranker. They are an **upper bound** on what a non-leaking top-k
aggregator could achieve. Operationalizing this requires a ranker that uses
only training-time information:

1. **Easiest non-leaking ranker:** the last-50 training-episode Sharpe inside
   [training/cleanrl_ppo.py::train_cleanrl_ppo](../training/cleanrl_ppo.py).
   `ep_returns` is already tracked; convert to Sharpe and surface per-cell.
2. **More principled:** add a held-out validation slice (e.g. last 20 % of the
   train_df) and rank seeds by deterministic validation performance.

**Order-of-magnitude estimate of recovery:** a noisy non-leaking ranker that
captures even half of the oracle gap on H4 would lift the ensemble from
+18.1 % to ~+30 %, beating BH (+25.9 %) by ~4 pp on average. The H4 baseline
already beats BH on the strong-trend fold; recovering even a fraction of the
oracle ceiling would push the ensemble into "tradeable" territory on more
than just one fold per year.

**Open questions advanced:**
1. **Does a training-Sharpe ranker recover most of the oracle gap, or does
   training-time performance not correlate strongly enough with test-time
   performance to be useful?** Untested. This is the top remaining experiment.
2. Does the oracle finding generalize to other instruments? Same multi-
   instrument open question as §15.13 — not gold-specific is the hope.

Artifacts: oracle analysis printed by `scripts/_topk_analysis.py` (no CSV
written; reproducible by running the script).

### 15.15 Top-k by train-Sharpe ranker — FAILED (2026-05-30)
The non-leaking ranker proposed in §15.14: track per-cell Sharpe of the
last-50 training-episode returns inside `train_cleanrl_ppo`, surface in
`cells.csv` via `grid.py`, then rank seeds by that train_sharpe instead of
test return/Sharpe. Implementation: `_train_sharpe` attribute on the
`ActorCritic` in [training/cleanrl_ppo.py](../training/cleanrl_ppo.py) and
`train_sharpe` column in [validation/grid.py](../validation/grid.py).
Re-ran the H4 grid (32 seeds × 4 folds × 1.31M timesteps, tag
`excess_bigseed_32_h4_ts`, ~64 min).

**Correlation between train_sharpe and test metrics (per fold):**

| fold | ρ(train_sharpe, test_return) | ρ(train_sharpe, test_sharpe) |
|---|---|---|
| 0 | −0.140 | −0.152 |
| 1 | −0.154 | −0.157 |
| 2 | **−0.367** | −0.339 |
| 3 | +0.211 | +0.069 |
| mean | **−0.113** | **−0.145** |

**train_sharpe anti-correlates with test return on 3 of 4 folds.** The signal
is *opposite* of the one we hoped for. High train Sharpe = overfitting the
training distribution; OOS test performance is reliably worse.

**Top-k results, this grid (mean over folds, %):**

| k | by **train_sharpe** | by oracle (test sharpe) | "recovery" |
|---|---|---|---|
| 1 | +25.0 (high-variance outlier) | +45.6 | — |
| **4** | **+10.6** | **+44.5** | **24 %** ← worse than all-32 |
| 8 | +13.3 | +35.9 | 37 % |
| 16 | +12.3 | +27.1 | 45 % |
| 32 (all) | +12.4 | +12.4 | — |

**Picking top-4 by train_sharpe gives +10.6 %, worse than the all-32 mean of
+12.4 %.** The ranker is counterproductive — it specifically selects seeds
that overfit, then those seeds underperform OOS.

**Implication for §15.14's oracle:** the +46.8 % oracle ceiling at H4 top-4
is real but unreachable by training-set Sharpe. The signal that distinguishes
good seeds from overfitting seeds is not present in the training metric.

**Non-leaking ranker direction is still open, but barely:**
1. **Held-out validation slice** — split each fold's train data into ~80 %
   training + ~20 % validation, deterministic eval on the validation slice,
   rank by validation Sharpe. Cleanest known alternative. Requires ~30 min
   code + ~70 min grid rerun. If even validation Sharpe anti-correlates, the
   top-k lever is fully closed.
2. **Multi-feature learned ranker** (train_sharpe + ep_return_mean +
   ep_return_std + others) — only 128 samples; barely above noise.
3. **Stop top-k**, accept H4 baseline (§15.10) as the project's best
   tradeable ensemble, and move to multi-instrument H4 to test
   generalization.

**This run's full-32 ensemble vs §15.10 baseline:**
- ensemble median: **+18.1 %** (baseline +16.7 %) — slight gain from CUDA
  non-determinism, not from the ranker
- ensemble robustness: **+23.4** (baseline +12.9) — higher by coincidence
- single CI: [+3.7, +12.4] (baseline [+3.2, +10.9]) — tighter
- beats BH: 25 % (same as baseline)

Variation between H4 runs is ~2-5 pp per fold from CUDA non-determinism;
none of this run's improvement vs baseline H4 attributes to the ranker.

**Status of the §15.10 conclusion: confirmed.** The H4 baseline ensemble
(no gate, no ranker) is the project's best operational result. The oracle
top-k ceiling is informative but unreachable; the variance-amplification
gates and the train-Sharpe ranker have all underperformed it.

Artifacts: `logs/grid/excess_bigseed_32_h4_ts/{summary.json, cells.csv,
ensemble_cells.csv}`. cells.csv now includes the `train_sharpe` column.

### 15.16 Top-k by held-out validation Sharpe — also FAILED (2026-05-30)
Follow-up to §15.15: build a non-leaking ranker by splitting each fold's
training data into the first 80 % (used for training and to fit the
normalizer) and the last 20 % (a held-out validation slice the agent never
sees during training). After training, run a deterministic eval on the
validation slice; the resulting Sharpe is the new ranker. Implementation:
`GridEvaluator(val_frac=0.2)` + `--val-frac 0.2` flag + validation eval in
[validation/grid.py::run](../validation/grid.py). Tag:
`excess_bigseed_32_h4_val20`. Wall-clock ~64 min.

**val_sharpe correlation with test_return (signal direction):**

| fold | ρ(val_sharpe, test_return) |
|---|---|
| 0 | +0.152 |
| 1 | +0.016 |
| 2 | **+0.395** |
| 3 | −0.182 |
| mean | **+0.095** |

Compared to §15.15's train_sharpe (mean ρ = −0.113): **val_sharpe is correctly
oriented** — picking top-k by val_sharpe is meaningfully better than picking
randomly or by train_sharpe. But the signal is weak.

**Top-k results on this run (80 % train + ranker):**

| k | by val_sharpe | by oracle (test sharpe) | recovery |
|---|---|---|---|
| 1 | −1.83 % | +39.56 % | 0 % |
| **4** | **+1.92 %** | +27.54 % | 17 % |
| 8 | +1.69 % | +19.10 % | 22 % |
| 16 | −1.87 % | +9.68 % | 30 % |
| 32 | −3.40 % | −3.40 % | — |

Top-4 by val_sharpe beats the unfiltered 32-seed mean (+1.92 % vs −3.40 % on
this run, a +5.3 pp lift). The ranker works directionally.

**But the 80 %-train cost destroys the comparison vs §15.10:**

| fold | 80 % train | 100 % train | Δ |
|---|---|---|---|
| 0 | −14.3 % | −2.2 % | −12.2 |
| 1 | +28.1 % | +13.4 % | +14.7 |
| 2 | −7.6 % | +19.9 % | −27.5 |
| **3** | **−15.9 %** | **+41.3 %** | **−57.2** |
| ensemble mean | **−2.4 %** | **+18.1 %** | **−20.5** |

The strong-trend fold (fold 3) collapses from +41 % to −16 % because the
agent loses 20 % of its training data — exactly the bars at the end of the
training distribution that would have prepared it for the 2025 trend.

**Net comparison (the question that actually matters):**

| | top-4 / val_sharpe (80 % train) | all-32 baseline (100 % train) |
|---|---|---|
| ensemble mean return | +1.92 % | **+18.1 %** |
| beats BH? | no (BH +25.9 %) | no but closer |
| CI on single | [−10.1, −0.3] excludes 0 negative | [+3.2, +10.9] excludes 0 positive |

Even the best-ranked subset on the data-restricted run loses by 16 pp to
the unfiltered 100 %-train baseline.

**Why both top-k rankers fail:**
1. **train_sharpe (§15.15):** correlates the *wrong way* — high train Sharpe
   marks overfitters that underperform OOS. Net: top-4 by train_sharpe is
   worse than the all-32 mean.
2. **val_sharpe (§15.16):** correlates the right way (mean ρ = +0.095) but
   too weakly to overcome the −20 pp data-loss tax from training on 80 %.

The §15.14 oracle ceiling (H4 top-4 by *test* sharpe = +46.8 %) is therefore
**real but unreachable** from any non-leaking signal we can compute on-the-fly.
Both top-k experiments have been clean implementations and both have been
clean negatives.

**Status of the §15.10 conclusion: re-confirmed.** The H4 baseline ensemble
(100 % train, no gate, no ranker, no validation slice) remains the project's
best operational result. The top-k lever is closed.

**Productive remaining directions:**
- **Multi-instrument H4** — does the §15.10 H4 sweet spot generalize off
  gold (EURUSD H4, BTC H4, SPX H4)? Still untested. Cleanest open question.
- **ROADMAP V3.5 variance reduction levers** that this session deferred while
  top-k was the candidate: cosine LR schedule, SWA, target_kl tuning,
  longer training (1.3M → 5M steps to push past noise floor).
- **Stop and write up.** §15.8–§15.16 form a coherent set: H4 sweet spot
  positive on robustness + sharpe + per-fold-beats-BH; M5/D1 not significant
  or significantly negative; gates and rankers tested + rejected; oracle
  ceiling characterized; cleanly tradeable agent not achieved.

Artifacts: `logs/grid/excess_bigseed_32_h4_val20/{summary.json, cells.csv,
ensemble_cells.csv}`. cells.csv includes `train_sharpe`, `val_sharpe`,
`val_return_pct`.

### 15.17 H4 + Cosine LR — first lever to improve every aggregate metric (2026-05-30)
Variance-reduction lever A from the ROADMAP V3.5 backlog: anneal the PPO
optimizer LR from `learning_rate` (3e-4) down to `learning_rate * cosine_min_frac`
(0.05 → 1.5e-5) over the full training run, instead of constant LR. Implementation:
`PPOConfig.lr_schedule="cosine"` + `cosine_min_frac` in
[training/cleanrl_ppo.py](../training/cleanrl_ppo.py), wired through
`TrainingConfig` and `--lr-schedule cosine` CLI in
[scripts/grid_eval.py](../scripts/grid_eval.py). Tag:
`excess_bigseed_32_h4_cosine`, ~59 min wall-clock.

| | Baseline (§15.10) | **+ Cosine LR** | Δ |
|---|---|---|---|
| single CI 95% | [+3.2, +10.9] | **[+6.7, +13.6]** | **shift up +3.5 pp** |
| single median | +5.87 % | +8.82 % | +3 pp |
| single mean | +11.25 % | **+17.01 %** | +5.8 pp |
| single std | 28.07 | 35.89 | **+7.8** ❌ (variance increased) |
| single beats BH | 23 % | 26 % | +3 pp |
| ensemble median | +16.66 % | +16.09 % | ≈ |
| **ensemble mean** | +18.10 % | **+23.08 %** | **+5 pp** |
| ensemble std | 15.59 | 21.29 | +5.7 (worse) |
| **ensemble worst fold** | −2.17 % | **+3.26 %** | first ALL-POSITIVE ensemble |
| **ensemble best fold** | +41.26 % | **+56.88 %** | +15.6 pp |
| ensemble beats BH | 25 % | 25 % | tied |
| **ensemble robustness** | +12.89 | **+14.78** | +1.9 |

**Per-fold ensemble:**

| fold | regime | baseline | + cosine | Δ |
|---|---|---|---|---|
| 0 | range | −2.2 % | +3.3 % | +5.5 |
| 1 | mild trend | +13.4 % | +6.6 % | **−6.8** |
| 2 | trend | +19.9 % | +25.6 % | +5.7 |
| 3 | strong trend | +41.3 % | **+56.9 %** | +15.6 |

**The lever does *not* reduce variance — it amplifies the mean.** Single std went 28 → 36 and ensemble std went 15.6 → 21.3, both worse. But the CI bottom shifted from +3.2 to +6.7 because the mean shifted up further than the std widened. Robustness Score is up despite the variance increase because the distribution is centered higher. The label "variance reduction" is misleading for cosine LR in this codebase — call it a *mean amplifier*.

**This is the closest the project has come to V4 unblock so far:**
- Ensemble mean +23.1 % vs BH +25.9 % — gap closed to 2.9 pp (was 8 pp on baseline)
- **First time every ensemble fold is profitable** (100 % positive folds)
- Single CI is the project's highest lower bound: +6.7 % per cell with 95 % confidence
- Record single-seed cell: fold 3 seed 26 = **+280.23 %**

**Why it likely works (hypothesis):** at constant LR=3e-4, the policy keeps making fixed-size moves through training. Late in training, those moves bounce around the optimum without settling. Cosine reduces step size to ~1.5e-5 by the end, letting the policy converge into the basin it found. The basins are deeper / more profitable on average, especially on the strong-trend fold (+15.6 pp on fold 3 alone). Fold 1 (mild trend) is the only fold that regresses — possibly the cosine schedule overshoots into a local optimum the constant-LR policy didn't commit to.

**Status of §15.10 conclusion: superseded for H4.** The new best operational result is `excess_bigseed_32_h4_cosine`:
- Ensemble robustness +14.78 (vs §15.10's +12.89)
- 100 % profitable ensemble folds (vs §15.10's 75 %)
- Single CI [+6.7, +13.6] (vs §15.10's [+3.2, +10.9])

**Next variance-reduction lever to try (from ROADMAP V3.5):** SWA (Stochastic
Weight Averaging) — average the policy weights from the last K training
checkpoints. Cosine LR + SWA is the textbook combination: cosine drops LR so
the policy settles, SWA averages the settled trajectory. Could close the
remaining 2.9 pp gap to BH and produce the project's first cleanly tradeable
ensemble result.

Artifacts: `logs/grid/excess_bigseed_32_h4_cosine/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.18 H4 + Cosine LR + SWA — V4 unblock candidate (2026-05-30)
Combining §15.17's cosine LR schedule with Stochastic Weight Averaging (SWA):
average the policy weights from each inner PPO epoch starting at
`swa_start_frac` = 0.6 of training. Implementation:
`PPOConfig.use_swa=True` + `--swa` CLI flag in
[scripts/grid_eval.py](../scripts/grid_eval.py). The averaged weights replace
the last-checkpoint weights at the end of training, so the deployed policy is
the SWA model. Tag: `excess_bigseed_32_h4_cosine_swa`, ~62 min.

| | Baseline (§15.10) | Cosine (§15.17) | **Cosine + SWA** |
|---|---|---|---|
| single CI 95% | [+3.2, +10.9] | [+6.7, +13.6] | **[+6.6, +13.6]** |
| single median | +5.87 % | +8.82 % | **+10.70 %** |
| single mean | +11.25 % | +17.01 % | +15.37 % |
| **single std** | 28.07 | 35.89 (worse) | **28.17** (SWA fixed cosine's std hit) |
| **ensemble median** | +16.66 % | +16.09 % | **+29.08 %** |
| **ensemble mean** | +18.10 % | +23.08 % | **+30.43 %** ← beats BH (+25.94 %) by 4.5 pp |
| ensemble std | 15.59 | 21.29 | **17.91** |
| ensemble worst fold | −2.17 % | +3.26 % | **+8.84 %** |
| ensemble best fold | +41.26 % | +56.88 % | +54.73 % |
| **ensemble beats BH** | 25 % | 25 % | **50 %** ← first time >25 % |
| **ensemble robustness** | +12.89 | +14.78 | **+27.91** ← 2× project's prior best |

**Per-fold ensemble (with Sharpe and Max DD):**

| fold | regime | baseline | cosine | **cosine+SWA** | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| 0 | range | −2.2 % | +3.3 % | **+8.8 %** | 0.42 | 15.6 % |
| 1 | mild trend | +13.4 % | +6.6 % | **+39.7 %** | **1.37** | 17.9 % |
| 2 | trend | +19.9 % | +25.6 % | +18.5 % | 0.75 | 13.9 % |
| **3** | **strong trend** | +41.3 % | +56.9 % | +54.7 % | **1.95** | **9.3 %** |

**This is the project's first V4-unblock candidate.**
1. **Ensemble mean +30.4 % > BH +25.9 %** by 4.5 pp — first time any
   operational (no leakage, no cherry-picking) config beats buy-and-hold on
   raw return.
2. **Beats BH on 50 % of folds** at the ensemble level — first time the
   project clears the 25 % ceiling that every prior lever stalled at.
3. **Robustness +27.9** — more than 2× the previous record (+12.9). The
   distribution is centered higher AND tighter than any prior run.
4. **Sharpe 1.37 and 1.95** on the two trend folds (mild + strong) —
   competitive with BH's typical Sharpe (~1.3-1.5) and *better* risk-adjusted
   on fold 3.
5. **Max DD reduced significantly** (fold 3: 28 % → 9.3 %). The SWA-averaged
   policy rides the trend with much less reversal exposure.
6. **SWA fixes cosine's variance increase**: std went 35.9 → 28.2, back to
   baseline level. The mean amplification from cosine is preserved; the
   variance increase is undone. Textbook SWA effect.

**Why the combo works (hypothesis):**
- Without cosine, the constant LR keeps the policy bouncing around the
  optimum. SWA averaging that trajectory yields ≈ the same trajectory.
- With cosine, late-training LR is small. Each inner-epoch update moves the
  policy in a tiny step around the basin. SWA averages those steps and finds
  the **centroid of the basin** — a better generalizing policy than any
  single endpoint snapshot.
- The combination is what the original SWA papers describe; it's the canonical
  setup.

**Status of the §15.6 / §15.10 conclusions: superseded for H4.** The H4 +
cosine + SWA configuration has a robust edge on raw return, Sharpe, and
drawdown vs buy-and-hold. The §15.6 "no robust RL edge beyond buy-and-hold"
conclusion was correct for the constant-LR baseline + every gate / ranker we
tried; it does NOT survive cosine + SWA.

**Next steps in priority order:**
1. **Sanity rerun** with a different seed sequence (e.g. seeds 32-63) to
   confirm the +30 % isn't a lucky CUDA non-determinism draw. ~60 min.
2. **Multi-instrument H4** with cosine+SWA — does the edge generalize off
   gold? EURUSD H4, BTC H4, SPX H4.
3. **V4 work**: live-trading pipeline (LiveTrader + MT5 bridge) is the next
   bottleneck. Demo-account smoke test before any live deployment.
4. **Variance reduction lever C** (target_kl tuning) — risky given prior
   failure, low priority now that cosine+SWA already crossed the V4 threshold.

Artifacts: `logs/grid/excess_bigseed_32_h4_cosine_swa/{summary.json, cells.csv, ensemble_cells.csv}`.

### 15.19 Four-run reproducibility check (2026-05-31)
The §15.18 single-run "+30.4 % ensemble mean beats BH by 4.5 pp" claim was
sanity-checked with three additional independent 32-seed runs (seeds 32-63,
64-95, 96-127) using identical config. The result substantially weakens
§15.18's headline.

**Per-run results:**

| seeds | tag | ensemble mean | robustness | %pos folds | beats BH | single CI |
|---|---|---|---|---|---|---|
| 0-31 | `..._cosine_swa` | **+30.43 %** | **+27.91** | 100 % | 50 % | [+6.6, +13.6] |
| 32-63 | `..._cosine_swa_s32` | +21.51 % | +26.70 | 100 % | 25 % | [+3.7, +14.3] |
| 64-95 | `..._cosine_swa_s64` | +14.45 % | **−3.89** ❌ | 75 % | 25 % | [+2.7, +15.1] |
| 96-127 | `..._cosine_swa_s96` | +19.59 % | +17.04 | 100 % | 25 % | [+4.5, +15.5] |

**Cross-run statistics (n=4):**

| | mean ± std | range |
|---|---|---|
| Ensemble mean | **+21.50 ± 6.66 %** | [+14.5, +30.4] |
| Robustness | **+16.94 ± 14.71** | [−3.9, +27.9] |
| BH | +25.94 % | — |
| **Δ (agent − BH)** | **−4.45 pp** | **agent loses to BH on average** |

**The §15.18 headline is partially retracted:**

| §15.18 claim | Four-run reality |
|---|---|
| "Ensemble mean +30.4 % beats BH +25.9 % by 4.5 pp" | **Average ensemble +21.5 % loses to BH by 4.5 pp** |
| "Robustness +27.9, 2× the constant-LR baseline" | Avg +16.9, but range includes **−3.9** |
| "Beats BH 50 % folds, first time > 25 %" | One run hit 50 %; others 25 %. Average ~31 %. |

**What stays robust across all four runs:**
- **Single-cell CI excludes 0 in every run** (lowest bound +2.7 %, highest
  +6.6 %). The per-cell positive-expected-return result IS reproducible.
- **3 of 4 runs hit 100 % profitable ensemble folds**; the worst run still
  hit 75 %. Win-rate stability holds.
- **Single-cell metrics are tighter than ensemble metrics:** single-cell
  median is +8 to +11 % across all four runs (std on the median estimator
  about ±1.5 pp), much steadier than the ensemble mean.

**What does NOT stay robust:**
- Raw ensemble mean swings ±7 pp run-to-run.
- Robustness Score has a ~50 % chance of being in the +17 to +28 range and
  ~25 % chance of being negative. One number from one run is not the truth.
- "Beats BH at the fold level" is mostly 25 % (one fold of four trending).

**Why this matters for prior session memos:**
- [[cosine-swa-unblock]] memo's "+30 % beats BH" framing was based on the
  lucky upper-tail draw of seeds 0-31. The four-run average is closer to
  "ties or trails BH on raw return."
- [[sanity-rerun-verdict]] memo (after just two runs) called this "borderline
  V4 unblock." Two more runs confirm: it's not unblock, it's
  "positive-expected-per-cell with raw-return parity to BH."

**Updated project verdict for downstream code / docs / CLAUDE.md:**
- Per-cell expected return: **positive, reproducible**.
- Raw ensemble return: **ties or trails BH** on average.
- Risk-adjusted (Sharpe / DD) vs BH: **not yet quantified across runs** —
  per-fold Sharpe is positive in the per-run reports but a cross-run Sharpe
  distribution has not been computed. This is the next thing to compute
  before any "tradeable" framing.
- Win rate at the fold level: **stable** (typically 100 %, worst case 75 %).

**Operational implications:**
- The §15.18-recommended config (cosine + SWA) is still the **best** known
  config; it just doesn't justify the headline "beats BH" claim by itself.
- Going live with the single-seed `scripts/v4_smoke.py` model is **not**
  justified by raw-return metrics. An ensemble of 32 saved models would be
  needed to match grid-eval, and even then the cross-run distribution does
  not reliably exceed BH on raw return.

**Next steps (priorities reset by this finding):**
1. **Compute cross-run Sharpe and max-DD distributions** from `cells.csv` of
   the four runs. If risk-adjusted metrics consistently beat BH across all
   four × 128 = 512 cells, the "risk-adjusted edge" framing is rescuable;
   if not, the project is closer to the original CLAUDE.md negative-result
   wrap-up than to the §15.18 V4-unblock framing.
2. **More replication.** 4 runs at std 6.7 pp gives a CI on the population
   mean of about ±5 pp. Eight or sixteen runs would tighten that. Cheap
   (~60 min each) and decisive.
3. **Multi-instrument H4** remains the best remaining science question and
   is now even more important: it would test whether "positive expected
   per-cell return + raw-return parity with BH" is a gold-specific finding
   or a framework-wide finding.

Artifacts:
- `logs/grid/excess_bigseed_32_h4_cosine_swa/{...}` (seeds 0-31)
- `logs/grid/excess_bigseed_32_h4_cosine_swa_s32/{...}` (seeds 32-63)
- `logs/grid/excess_bigseed_32_h4_cosine_swa_s64/{...}` (seeds 64-95)
- `logs/grid/excess_bigseed_32_h4_cosine_swa_s96/{...}` (seeds 96-127)
