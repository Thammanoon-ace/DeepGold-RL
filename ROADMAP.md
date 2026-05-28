# DeepGold RL — Roadmap

> Vision: a Reinforcement-Learning system for trading XAUUSD that trains on
> historical data, backtests realistically, adapts to market regimes, connects
> to MT5/IUX, and grows into an autonomous AI trading-research platform.
>
> Long-term flow:
> `Historical Data → Market Simulation → RL Training → Walk-Forward Validation
> → Paper Trading → Live Trading → Adaptive Self-Improving AI`

## Status legend
✅ done & verified  ·  🟡 partial  ·  ⬜ not started

---

## V0 — Foundation Prototype · Phase 1: Infrastructure & Environment  ✅
**Goal:** a base the RL agent can actually train on.

| Feature | Status | Where |
|---|---|---|
| Historical data loader | ✅ | [utils/data_loader.py](utils/data_loader.py) |
| CSV pipeline | ✅ | [utils/data_loader.py](utils/data_loader.py) |
| Feature engineering (RSI/MACD/EMA/ATR/return/vol) | ✅ | [utils/feature_engineering.py](utils/feature_engineering.py) |
| Gymnasium environment | ✅ | [env/gold_trading_env.py](env/gold_trading_env.py) |
| PPO baseline | ✅ | [training/train_ppo.py](training/train_ppo.py) |
| Basic backtest | ✅ | [backtest/backtester.py](backtest/backtester.py) |
| TensorBoard logging | ✅ | [training/train_ppo.py](training/train_ppo.py), [training/callbacks.py](training/callbacks.py) |

**Deliverables:** `GoldTradingEnv`, `train.py`, `backtest.py`, `requirements.txt` — all present.
**Success criteria:** PPO trains, env stable, no crash, backtest runs — **met** (smoke-tested end-to-end).

---

## V1 — Functional RL Trading System · Phase 2: Realistic Simulation  ✅
**Goal:** make the simulation realistic.

| Feature | Status |
|---|---|
| Spread / Slippage / Commission | ✅ |
| Stop Loss / Take Profit | ✅ |
| Unrealized PnL | ✅ |
| Risk management (size caps, margin, max-DD stop) | ✅ |
| Equity tracking | ✅ |
| Reward engineering | ✅ |
| Drawdown penalty / Overtrading penalty | ✅ |
| Metrics: win rate, Sharpe, max DD, profit factor | ✅ |

**Success criteria:** reward stable, no obvious reward exploit, usable equity curve — **met**.

---

## V2 — Quant Research System · Phase 3: Validation & Robustness  ✅
**Goal:** reduce overfitting; approach real quant research.

| Feature | Status | Where |
|---|---|---|
| Walk-forward validation (rolling/expanding windows) | ✅ | [validation/walk_forward.py](validation/walk_forward.py) |
| Time-series split (expanding/rolling + embargo gap) | ✅ | [validation/splitters.py](validation/splitters.py) |
| Hyperparameter tuning (Optuna, validation-set objective) | ✅ | [optimization/optuna_tuning.py](optimization/optuna_tuning.py) |
| Multi-dataset evaluation | ✅ | [validation/multi_dataset.py](validation/multi_dataset.py) |
| Sortino ratio | ✅ | [backtest/metrics.py](backtest/metrics.py) |
| Calmar ratio | ✅ | [backtest/metrics.py](backtest/metrics.py) |
| Expectancy + payoff ratio | ✅ | [backtest/metrics.py](backtest/metrics.py) |
| Trade-distribution analysis (streaks, exit reasons, outlier share) | ✅ | [backtest/metrics.py](backtest/metrics.py) |
| Real-data ingestion hardening (MT5 dialects) + inspector | ✅ | [utils/data_loader.py](utils/data_loader.py), [scripts/inspect_data.py](scripts/inspect_data.py) |
| Tools: Optuna | ✅ | vectorbt deferred (pandas 3.0 compat); not on priority list |

**Scripts:** [scripts/walk_forward.py](scripts/walk_forward.py),
[scripts/evaluate_multi.py](scripts/evaluate_multi.py),
[scripts/tune.py](scripts/tune.py), [scripts/inspect_data.py](scripts/inspect_data.py).

**Leakage discipline:** walk-forward re-fits the normalizer per fold (train slice
only) with an embargo gap; Optuna optimizes a **validation slice carved from the
training data**, never the 2025 test set; multi-dataset uses the saved
training-time normalizer.

**Success criteria:** performance stable across periods; survives unseen data —
**machinery verified** (all four tools run end-to-end; honest losses reported for
under-trained agents on synthetic data).

---

## V3 — Advanced Deep RL · Phase 4  🟡 (4A done; 4B/4C are the new next work)
**Goal:** understand market structure; better generalization.

> The updated roadmap splits Phase 4 into **4A Sequence Learning** (done),
> **4B Indicator Expansion** (new, mostly TODO) and **4C Structural Pattern
> Intelligence** (new, TODO). V3 is therefore *not* complete — only 4A is.

### Phase 4A — Sequence Learning  ✅ (SAC deferred)

| Feature | Status | Where |
|---|---|---|
| LSTM feature extractor (PPO + LSTM) | ✅ | [policies/extractors.py](policies/extractors.py) |
| Transformer encoder + temporal attention pooling | ✅ | [policies/extractors.py](policies/extractors.py) |
| CNN chart-pattern extractor (1-D conv over time) | ✅ | [policies/extractors.py](policies/extractors.py) |
| Config-driven arch switch (`policy_arch`) + factory | ✅ | [policies/factory.py](policies/factory.py), [config/config.py](config/config.py) |
| Wired into train / walk-forward / Optuna; save+load verified | ✅ | [training/](training/), [validation/](validation/), [optimization/](optimization/) |
| Multi-timeframe observation (causal H1/H4 merge onto base) | ✅ | [utils/feature_engineering.py](utils/feature_engineering.py) |
| Volatility-regime feature | ✅ | [utils/feature_engineering.py](utils/feature_engineering.py) |
| Off-policy alternative: DQN (discrete) | ✅ | [training/train_ppo.py](training/train_ppo.py) (`algo: dqn`) |
| Architecture comparison experiment | ✅ | [scripts/compare_archs.py](scripts/compare_archs.py) |
| SAC / offline-RL | ⬜ | SAC is continuous-action only; would need a continuous-action env variant. DQN covers the discrete off-policy case. |

Switch architecture from the CLI or YAML:

```powershell
python scripts/train.py --policy-arch lstm --device cuda
python scripts/train.py --policy-arch transformer
python scripts/train.py --policy-arch cnn
```

The extractors reshape the flat observation back into a ``(time, features)``
sequence inside the network, so **no env/pipeline change is needed** and there
is no look-ahead (the window holds only causal features). GPU is worthwhile for
these encoders (the tiny MLP baseline is usually faster on CPU).

Other V3 additions:
* **Multi-timeframe** — set `features.multi_timeframe: [H1, H4]` to merge
  higher-timeframe indicators onto each base bar, aligned causally (a base bar
  only sees the last *closed* higher-TF bar).
* **Volatility regime** — `vol_regime` feature = short-term vol ÷ its long-run
  median (>1 = turbulent, <1 = calm).
* **Off-policy DQN** — `--algo dqn` (config `training.algo`). SAC is omitted
  because it is continuous-action only; DQN is the discrete counterpart.
* **Architecture comparison** — `python scripts/compare_archs.py` trains each
  architecture and tabulates out-of-sample 2025 metrics side by side.

### Phase 4B — Indicator Expansion System  🟡 (subset done)
Add a broader, **non-redundant** feature set, each validated by walk-forward.

| Group | Indicators | Status |
|---|---|---|
| Trend | EMA, SMA, VWAP, trend spread | EMA ✅ · SMA/VWAP/spread ⬜ |
| Momentum | RSI, MACD, Stochastic RSI, ROC | RSI/MACD ✅ · StochRSI/ROC ⬜ |
| Volatility | ATR, Bollinger Bands, historical vol | ATR ✅ · Bollinger/HV ⬜ |
| Candle structure | body size, wick ratio, bull/bear pressure, candle momentum | ⬜ |
| Market structure | swing highs/lows, breakout strength, trend slope, S/R distance, structure-break | ⬜ |
| Volume & liquidity | relative volume, liquidity-sweep, volume-spike | ⬜ |

**Rules (from the roadmap):** incremental addition; avoid feature explosion;
avoid highly-correlated indicators; validate every new indicator with
walk-forward before keeping it. Plan: a feature-group registry + a correlation/
importance filter + a per-feature walk-forward A/B harness.

### Phase 4C — Structural Pattern Intelligence  ⬜
Let the agent *learn* market structure rather than hardcoding chart patterns.

* Quantifiable structural features: volatility compression, breakout
  probability, rejection strength, momentum-shift, consolidation-zone,
  trend-continuation probability, reversal pressure.
* Learned representations: structural embeddings, sequence-based pattern
  understanding, transformer-attention pattern extraction, CNN structure
  encoding (the 4A extractors are the substrate for this).
* **Constraints:** no hardcoded "Head & Shoulders" / subjective patterns; only
  quantifiable structural features; the RL agent learns patterns autonomously.

## V3.5 — Variance Reduction & Stability Phase  ⬜ ← NEXT (blocks V4)
**Goal:** lower the noise floor until experiments are decidable. Profitability is
explicitly *not* the objective; **measurability and robustness** are.

Motivation (measured): multi-seed walk-forward gave compounded returns of
**−62% to +204% from the random seed alone** → variance dominates any signal,
so no architecture/feature comparison is currently trustworthy.

Pillars (✅ = implemented & tested this iteration):
* ✅ **5E Vectorized batched env** (NumPy) — [env/vectorized_env.py](env/vectorized_env.py);
  provably equivalent to the scalar env (max reward err 7e-9, obs/equity err 0)
  and **2x→21x faster** as lane count grows (7.3x at 32 lanes); PPO-compatible;
  wired via `TradingDataPipeline.make_vectorized_env`.
* ✅ **Evaluation rigor** — baselines (buy&hold/flat/random,
  [backtest/baselines.py](backtest/baselines.py)) + Robustness Score & bootstrap
  median CI ([validation/robustness.py](validation/robustness.py)). On our real
  multi-seed data: median −25.5%, 95% CI [−62, +204] (straddles 0).
* ✅ **5A Ensemble policy** — [policies/ensemble.py](policies/ensemble.py):
  multi-model action-probability averaging + confidence gate (Hold below τ);
  SB3-compatible `predict`, tested.
* ✅ **5B Regime detection** — [utils/regime.py](utils/regime.py): causal
  4-regime classifier (trend×vol), thresholds fit on train; `add_features` for
  regime-conditioning, `label` for regime-stratified evaluation.
* ✅ **5C Trade-frequency control** — `min_hold_bars`, `max_trades_per_episode`,
  `trade_penalty_growth` (+ existing cooldown) wired into both scalar and
  vectorized envs; confidence threshold lives in 5A. Equivalence preserved
  (on and off); trade-cap verified.
* ✅ **5D Feature-selection** — variance + correlation pruning ✅ and
  **mutual-information ranking** ([utils/feature_selection.py](utils/feature_selection.py),
  fit-on-train, vs forward return). SHAP deferred (needs the `shap` dep + a
  trained model). Finding: volatility features carry the most MI, not direction.
* ✅ **5F Higher-timeframe support** — `grid_eval.py --timeframe H1` resamples
  M5→H1/M15 (causal OHLC aggregation) with correct per-TF annualization; the
  real matched M5-vs-H1 grid run is the pending experiment. (Tiny smoke hinted
  H1 single-seed std ~33 vs M5 ~107 — directional, needs the full run.)
* ⬜ Variance reduction (train-to-convergence, SWA, LR/entropy schedules, `target_kl`)
  and behavior stabilization (cost-aware/Differential-Sharpe reward, vol-targeted
  sizing).
* ⬜ Algorithm research (PPO vs SAC/continuous actions vs QR-DQN).

**V4 is blocked** until V3.5 yields a config whose OOS *distribution* beats
baselines at acceptable variance — or a rigorous negative result.

Full design + analysis: [docs/V3_5_VARIANCE_REDUCTION.md](docs/V3_5_VARIANCE_REDUCTION.md).
Evidence handoff: [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md).

## V4 — Semi-Production AI Trader · Phase 5: Real-Time Trading  🟡 (blocked by V3.5)
MT5 integration 🟡 (bridge skeleton exists in [live_trading/](live_trading/)),
IUX execution bridge ⬜, real-time inference ⬜, paper-trading mode ⬜, OMS ⬜,
live dashboard ⬜, async execution + WebSocket feed + error recovery ⬜.

## V5 — Adaptive Quant AI · Phase 6: Market Adaptation  ⬜
Regime detection, dynamic retraining, online-learning experiments, adaptive risk
sizing, market-state clustering; meta-learning, ensemble RL, policy switching.

## V6 — Multi-Agent Quant Architecture · Phase 7: Portfolio Intelligence  ⬜
Trend/scalping/volatility/risk-control agents, ensemble voting, multi-asset
support, portfolio balancing, capital-allocation AI.

## V7 — Institutional Research Platform · Phase 8: Professional Ecosystem  ⬜
Distributed training (Ray RLlib), cloud inference, experiment tracking (MLflow),
automated retraining pipelines, strategy-marketplace architecture, Docker/K8s.

---

## Parallel systems roadmap

| System | V0 | V2 | V4 | V7 |
|---|---|---|---|---|
| Data | CSV ✅ | Database | Real-time streaming | Distributed pipeline |
| Visualization | matplotlib ✅ | Plotly dashboard | Live web dashboard | Analytics platform |
| Risk management | Fixed SL/TP ✅ (V1) | Dynamic risk (V3) | Adaptive sizing (V5) | Portfolio AI risk engine (V7) |

---

## V3.5 experiment outcomes (2026-05-27)

Run via the grid protocol (CNN, 3 seeds × 4 folds, M5), judged by the Robustness
Score + per-fold return vs buy-and-hold (+27%). Full detail:
[docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) §15.

| Lever | Outcome |
|---|---|
| **Excess-return reward** (beat buy&hold) | ✅ **best config** — ensemble +9.9% / robust +16.7 / 100% folds positive; beats BH return in 75% of folds (but trails BH mean/Sharpe) |
| Higher timeframe (H1) | ❌ no better than M5 (std 41 vs 45) |
| Hybrid CNN-LSTM | ❌ worse (overfits) |
| Vol-targeted sizing | ❌ worst (hits DD floor) |
| Regime features (5B) | ❌ worse (feature explosion) |
| Ensembling | ✅ cuts variance ~2–7× (only useful atop a base with edge) |

**Conclusion:** the only lever that helped was changing the reward *objective*;
every *complexity* lever overfit and hurt. Best agent is competitive with — but
does not cleanly beat — buy-and-hold in a structural gold bull. No robust edge;
the framework correctly refuses to certify one.

## Current position

**V0 ✅, V1 ✅, V2 ✅ complete; V3 🟡 — Phase 4A done, 4B/4C pending.** The
updated roadmap expanded Phase 4 with an Indicator Expansion System (4B) and
Structural Pattern Intelligence (4C), so V3 is no longer "done": we have the
deep architectures (LSTM/Transformer/CNN), DQN, multi-timeframe and the
volatility-regime feature, but the broader indicator set and learned structural
features remain.

Real data in place: `data/XAUUSD_M5.csv` = real XAUUSD 2019–2025 (496k bars,
Dukascopy). GPU: torch cu128 (CUDA 12.8).

**Decisive finding:** multi-seed walk-forward exposed compounded returns of
**−62% to +204% from the random seed alone** → no reliable edge yet; variance
dominates signal. The framework correctly refused to validate a fake edge.

**Next increment = V3.5 (Variance Reduction & Stability)** — build the
vectorized env + distributional/baseline-relative evaluation, test the H1
timeframe, and apply variance-reduction + behavior-stabilization levers before
any further feature/architecture work or V4. See
[docs/V3_5_VARIANCE_REDUCTION.md](docs/V3_5_VARIANCE_REDUCTION.md).
