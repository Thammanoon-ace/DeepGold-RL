# DeepGold RL — Final negative result

**Status:** Project conclusion. This document is the canonical writeup of what
DeepGold RL set out to test, what we actually tested, and what we concluded.
For the experiment-by-experiment log see
[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md); for raw artifacts see
`logs/grid/`.

**Date of conclusion:** 2026-05-31.

---

## 1. The question

Can a reinforcement-learning agent, trained on retail-quality market data
(XAUUSD M5 OHLC from Dukascopy, free, no broker account) and run on retail
compute (a single NVIDIA RTX PRO 4000 Blackwell), beat a passive
buy-and-hold benchmark on gold over 2019–2025?

The pre-registered acceptable outcomes (recorded in
[CLAUDE.md](../CLAUDE.md) at the start of the project) were:
1. A reproducible positive edge → continue toward live trading.
2. A rigorous negative result → publish the negative result as the
   scientific contribution.

This document records outcome 2.

---

## 2. The answer (TL;DR)

**No edge.** Across 4 independent 32-seed × 4-fold runs of the best
configuration we found (H4 timeframe + excess-return reward + cosine LR
schedule + Stochastic Weight Averaging), the agent loses to buy-and-hold on
every metric measured:

| metric | agent (512-cell median) | buy-and-hold | delta |
|---|---|---|---|
| Return per fold | +12.07 % | +25.95 % | **−13.88 pp** |
| Sharpe ratio | +0.500 | +1.219 | **−0.719** |
| Max drawdown | 18.86 % | 11.00 % | **+7.86 pp worse** |

The agent does have a measurable positive expected return per cell
(single-cell median 95 % CI [+2.7 %, +6.6 %] across the four runs excludes
zero), but BH's expected return is larger and BH's risk is lower. *Positive
expected return ≠ beats BH.* Going live with this agent would
underperform a passive gold position on every risk-adjusted measure tracked.

---

## 3. What we tested

### 3.1 Configurations

A summary of the experiments that produced this conclusion. All experiments
used the same harness (32 seeds × 4 folds expanding-window walk-forward, per-
fold normalizer fit on training data only, excess-return reward unless noted,
1.31 M PPO timesteps per cell). See [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)
for the per-experiment narrative.

| family | configuration | ensemble mean | beats BH (folds) | verdict |
|---|---|---|---|---|
| **Timeframe sweep** | M5 (constant LR) | +1.5 % | 0 % | no edge (§15.8) |
| | H1 | +4.3 % | 0 % | no edge (§15.9) |
| | **H4** | +18.1 % | 25 % | best baseline (§15.10) |
| | D1 | −11.2 % | 0 % | significantly negative (§15.13) |
| **Entry gates** | H4 + ATR gate | +14.0 % | 25 % | rejected (§15.11) |
| | H4 + TER gate | +14.6 % | 25 % | mixed; ensemble worse (§15.12) |
| **Top-k seed selection** | train-Sharpe ranker | +12.4 % | — | anti-correlates with test (§15.15) |
| | held-out validation ranker | −2.4 % | 25 % | 80%-train tax destroys baseline (§15.16) |
| **Optimization schedule** | H4 + cosine LR | +23.1 % | 25 % | first non-failed lever (§15.17) |
| | **H4 + cosine + SWA (4 runs avg)** | **+21.5 %** | **~25 %** | best, but still loses BH (§15.18 / §15.19) |

### 3.2 Methodological levers also tested (in earlier work)

These were tested in earlier (non-this-session) work and produced negative
results that still hold:

- **Architecture sweep:** MLP, LSTM, Transformer, CNN, hybrid CNN-LSTM. CNN
  is the best of these; the others overfit. Bigger nets fail consistently.
- **Reward shaping:** absolute vs excess vs DSR (Differential Sharpe Ratio).
  Excess-return reward is the only one that improved results; DSR was
  rejected (§15.6).
- **Feature expansion:** the Phase-4B indicator groups (trend / momentum /
  volatility / candle / structure / volume). Adding more features
  consistently hurt OOS performance (low-SNR overfit).
- **Regime features:** causal regime_trend / regime_vol observations.
  Rejected — collapses fold 2 from +44.6 % to −40.1 % (§15.5).
- **Volatility-targeted sizing:** rejected (every config hits the
  max-drawdown floor).

### 3.3 Infrastructure work (also tested, also irrelevant)

A full session was spent testing every infrastructure speedup proposed in
[NEW_HARDWARE_PLAN.md](NEW_HARDWARE_PLAN.md):

| infra lever | result |
|---|---|
| GPU lane scaling 16k → 65k | plateaus at ~55k env-steps/s (GPU 32-45 % utilised; CPU/Python is the bottleneck) |
| `torch.compile(ActorCritic, reduce-overhead)` | 1.07× (no Triton on Windows) |
| Vectorized eval on `TorchVecGoldEnv` | 0.2–0.8× (slower; num_envs=1 pays batched-tensor overhead the scalar env avoids) |
| SubprocVecEnv 8 workers | 0.93× (IPC > parallelism gain) |
| SB3 engine at 256 lanes | loses 5.85× to GPU engine |
| Parallel grids | 1.20× slower (GPU contention) |

**No infrastructure lever exceeded 1.07×.** The Python overhead dominates,
not the GPU compute. This is a real meta-finding (see
[memory/cpu-levers-audit](../../../.claude/projects/d--Github-DeepGold-RL/memory/cpu_levers_audit.md))
but it didn't change the science answer.

---

## 4. What we learned (the scientific contribution)

This project's contribution is **not** a profitable trading bot. The
contribution is a set of well-supported findings about *what does and
doesn't matter* for retail-RL on financial time series:

### 4.1 Timeframe is the load-bearing variable

Across 32 seeds × 4 folds at four different timeframes (M5 / H1 / H4 / D1),
the *only* axis that meaningfully moved the agent's distribution was
timeframe:

- M5 (every 5 minutes): too noisy. CI [−3.7, +5.5] straddles 0.
- H1 (every hour): marginal. CI [+3.6, +10.3] excludes 0 but ensemble loses
  to BH by 22 pp on raw return.
- **H4 (every 4 hours): sweet spot.** CI [+3.2, +10.9] excludes 0,
  ensemble robustness peaks here.
- D1 (daily): too coarse. CI [−9.0, −5.1] significantly negative.

This is the cleanest finding of the project. The H4 sweet spot is not
specific to the H4 + cosine + SWA stack — it shows up at every
configuration we tried.

### 4.2 Architecture and feature complexity do not help

Every "make the agent smarter" lever we tried (deeper nets, LSTM,
Transformer, more features, regime detection as observations) made OOS
results worse, not better. Combined with the low signal-to-noise ratio of
M5 / H4 retail gold data, the safe prior is: **for retail-RL on financial
time series, complexity is overfitting waiting to happen.**

### 4.3 Optimization-schedule complexity HELPS — but only relative to a
failing baseline

Cosine LR annealing alone lifted the ensemble mean from +18.1 % to +23.1 %.
Cosine + SWA together lifted it to +21.5 % (cross-run average) with
substantially better Sharpe and max-DD on the trend folds.

But "lifted relative to baseline" ≠ "tradeable." The lifted result still
loses to BH by 13.9 pp on return, 0.72 on Sharpe, and 7.9 pp on drawdown.

### 4.4 Variance dominates signal — including across runs

A finding we expected: per-cell std (~28 percentage points) is large vs the
signal we're chasing. A finding we didn't expect: **the cross-run std on the
ensemble mean is ±6.7 pp** — large enough that a single 32-seed run's
"+30.4 % beats BH" claim is unreliable. Multi-run replication is mandatory
before any tradeability claim.

### 4.5 Optimisation-side wins do not show up in the obvious places

The cosine + SWA stack does **not** reduce variance (single-cell std went
*up* from 28 to 35–36 in the cosine-alone run; SWA only brought it back to
baseline 28). It works by *amplifying the mean* via convergence into deeper
basins. This is the opposite of what "variance reduction" usually means and
is worth knowing if someone else replicates this kind of work.

---

## 5. What survives replication

To be clear about what is and isn't supported:

### Survives 4-run replication:
- Per-cell expected return is positive (CI excludes 0 in 4 / 4 runs).
- Win rate at the fold level is stable (3 / 4 runs hit 100 % profitable
  folds; worst was 75 %).
- Timeframe sweep verdict (M5 negative, H4 best, D1 significantly negative).
- Cosine LR improves over constant LR.
- Single-cell median return is steady (8–11 % across the 4 runs).

### Does NOT survive replication:
- "Ensemble mean beats BH" — first run +30.4 %, mean of 4 runs +21.5 % vs
  BH +25.9 %.
- "Beats BH on Sharpe" — fails on 3 of 4 folds, loses by 0.72 on overall
  median.
- "Beats BH on max drawdown" — agent's DD is 7.9 pp larger than BH's.
- "Robustness Score consistently positive" — one run-in-four produces
  negative robustness.
- "First V4-unblock candidate" — rescinded.

---

## 6. What was retracted during this session

This is a transparency record. During the session that wrapped this project,
several intermediate claims were made and then retracted by further
replication. They are recorded here so future readers don't believe them:

| Claim (when made) | Status after replication |
|---|---|
| "H4 + cosine + SWA ensemble +30.4 % beats BH by 4.5 pp" (§15.18) | Retracted: lucky upper-tail draw. 4-run mean is +21.5 %, loses to BH by 4.5 pp. |
| "Risk-adjusted edge is measurable and reproducible" | Retracted: 4-run × 512-cell Sharpe + DD distributions both lose to BH. |
| "+27.9 Robustness, 2× the prior baseline" | Retracted: one of four reruns produced −3.9 Robustness. The number is volatile. |
| "First V4 unblock candidate" | Retracted: no metric the project tracks justifies going live. |
| "Cosine + SWA is the textbook V3.5 variance-reduction combo and it worked" | Half-true: it worked relative to the failing constant-LR baseline. It did not work against BH. |
| "BB-volatility feature group beats BH (ensemble +36.3 %, robustness +38.2, 75 % of folds)" — **8-seed run, 2026-06-10** | Retracted: 16-seed replication (2026-06-11) gave ensemble +5.6 %, robustness +1.08, 0 % of folds beat BH. The cleanest small-sample-artefact example in the project (§6.1). |

The pattern: **single-run, small-seed signals on this grid are NOT reliable.**
Four independent 32-seed runs at +21.5 % ± 6.7 % is the actual cosine+SWA
signal; the BB-volatility "breakthrough" evaporated entirely when the seed
count doubled. Anyone reading this project's earlier docs should weight the
small-sample claims accordingly.

### 6.1 The BB-volatility episode — a worked example of why replication matters

This is worth recording in full because it is the cleanest illustration of
the project's central methodological lesson.

On 2026-06-10, adding the `volatility` feature group (Bollinger %B +
bandwidth + historical vol) to the M5 baseline produced, at **8 seeds × 4
folds**, the best directional result the project had ever seen:

| metric | 8-seed BB-volatility |
|---|---|
| ensemble median | **+36.3 %** |
| robustness | **+38.2** |
| beats BH (folds) | **75 %** |
| worst fold | +22.7 % (all folds positive) |

It was the first M5 result to clear buy-and-hold, and it was tempting to
declare the negative-result verdict overturned. Instead, following the V3.5
protocol, a 16-seed verdict run was pre-registered with explicit
accept/reject thresholds *before* it was run.

The **16-seed replication (2026-06-11)** collapsed the result:

| metric | 8-seed | **16-seed** |
|---|---|---|
| ensemble median | +36.3 % | **+5.6 %** |
| robustness | +38.2 | **+1.08** |
| beats BH (folds) | 75 % | **0 %** |
| worst fold | +22.7 % | **−3.8 %** |

The cause: the single-seed return distribution is **bimodal** — std ≈ 50
percentage points, with roughly half the seeds landing at +50 to +150 % and
half hitting the −40 % max-drawdown floor. Eight seeds happened to draw more
of the winners; sixteen seeds drew a balanced sample, and the ensemble of a
balanced bimodal distribution averages to roughly buy-and-hold-minus.

Nothing about the lever changed between the two runs — only the seed count.
The "+36.3 % beats BH" number was pure sampling noise dressed up as signal.
Had we acted on the 8-seed run (deployed it, published it, built the
full-stack experiment on top of it), we would have been building on nothing.

**This is why every "beats BH" claim in this project required ≥ 16 seeds
before it was believed, and why even the 32-seed cosine+SWA result was
replicated four times before being written up. The same discipline that lets
us trust the negative results is the discipline that caught this false
positive.** See EXPERIMENT_SUMMARY §15.20 for the full data.

---

## 7. What this means for the broader retail-RL question

This project is one data point. It does not prove that *no* retail-RL setup
on *any* asset can beat *any* benchmark. It does provide reasonably strong
evidence that:

1. On a structural-bull-trend asset (gold 2019–2025), an RL agent trained
   on raw OHLC with retail-quality data cannot beat passive holding — the
   bar is too high.
2. Architecture / feature complexity is the wrong axis to push. Almost
   every lever there fails.
3. Timeframe choice and optimization schedule (cosine + SWA) are real
   levers. But their cumulative gain in this project's scope is still
   smaller than the BH gap.
4. Single-run results are unreliable. Anyone publishing trading-RL results
   without multi-run replication is reporting noise.

The honest implication: if the goal is "profitable trading bot," gold-2025
+ retail data + raw-OHLC features is the wrong starting point. Lower-SNR
assets, alternative data, structural-edge formulations (volatility
arbitrage, mean reversion on uncorrelated pairs) are more promising.

### 7.1 We tested the stat-arb pivot — it also failed

To avoid leaving "mean-reversion pairs are more promising" as an untested
hope, we actually ran it (2026-06-12). A classical Kalman-filter pairs
strategy (Chan 2013 dynamic hedge ratio) on metals, deliberately tested
classically first — if the classical base has no edge, RL on top won't either,
and the classical backtest takes hours vs the multi-day RL-env rewrite.

Findings (`scripts/coint_screen.py`, `scripts/kalman_pairs.py`):
- **Most metals pairs are not cointegrated** over 2019–2025. XAU/XAG, the
  classic textbook pair, has Engle-Granger p ≈ 0.29 (gold/silver decoupled).
- **Three pairs are cointegrated** (all anchored on platinum): XAU/XPT
  (p = 0.006), XPT/XPD (p = 0.007), XAG/XPT (p = 0.010).
- **But even the cointegrated pairs have no tradeable edge.** Every active
  Kalman config loses after retail costs (8 bps round-turn); the best Sharpe
  across all pairs and parameters is ≈ 0, achieved only by a config that
  barely trades. As the strategy's in-market fraction rises from 3 % to 30 %,
  Sharpe falls from ~0 to −1.4.

The lesson: **cointegration is necessary but not sufficient.** A statistically
stationary spread does not imply a profitable one — the magnitude of
mean-reversion must exceed transaction costs, and on retail metals it does
not. The stat-arb pivot is closed on this data; the only remaining chances are
institutional cost structures or a different asset class with a larger
reversion-to-cost ratio. This is documented so the next reader doesn't repeat
it expecting a different answer.

### 7.2 The framework is the contribution

If the goal is "rigorous methodology for testing trading hypotheses,"
DeepGold RL provides one. The harness — multi-seed walk-forward grid,
per-cell distribution + bootstrap median CI + Robustness Score, GPU-native
training stack equivalence-verified against the scalar env, plus the
cointegration screen + Kalman backtest for the stat-arb branch — is the
reusable contribution. It is good at one thing above all: **refusing to
certify a false positive**, whether from a lucky 8-seed RL draw or a
cointegrated-but-untradeable spread.

---

## 8. Reading guide

For someone landing on this repo and wanting to understand the result:

1. This document.
2. [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) §15.8–§15.19 for the
   experiment-by-experiment narrative.
3. [CLAUDE.md](../CLAUDE.md) for the project-wide context and the original
   pre-registered acceptable outcomes.
4. `logs/grid/excess_bigseed_32_*` for raw cells.csv / summary.json from
   every grid.

For someone wanting to extend this work or test on a new asset:

1. The harness is in [validation/grid.py](../validation/grid.py) — use it.
2. The current best config (H4 + cosine + SWA, see
   [CLAUDE.md](../CLAUDE.md)'s "Canonical commands") is the baseline to
   beat.
3. **Run at least four independent 32-seed grids** before claiming any
   result. One run is not enough. This is the main lesson.

---

*Project status: closed in negative-result wrap-up. Infrastructure is
preserved and runnable; no live-trading deployment is justified by the data.*
