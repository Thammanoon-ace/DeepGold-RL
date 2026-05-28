# V3.5 — Variance Reduction & Stability Phase

> **Framing.** The project's greatest result so far is *not* a profit — it is
> that the framework refused to validate fake edges. Multi-seed walk-forward
> showed compounded returns of **−62% to +204% from the random seed alone**.
> That is the finding to build on: **we currently cannot measure anything,
> because the noise floor is larger than any plausible signal.** V3.5 is about
> lowering that noise floor until experiments become decidable. Profitability is
> explicitly *not* the objective of this phase; *measurability and robustness*
> are.

---

## 0. TL;DR for the impatient

- **Do not** scale models, add features blindly, or chase the +204% seed. Those
  increase variance/overfitting.
- **Do** (in order): build a **vectorized env** (compute enabler) → add
  **baselines + distributional/robustness reporting** → test the **higher-
  timeframe (H1) hypothesis** → **train to convergence with many seeds/folds**
  → **seed-ensemble** → **redesign reward/sizing** (cost-aware, vol-targeted) →
  **mandatory feature selection**.
- Success criterion for V3.5 is **not** "make money". It is: **across ≥8 seeds ×
  ≥6 folds, the OOS return distribution is (a) much tighter than today and (b)
  honestly comparable to buy-and-hold / random baselines.** If, after that, no
  config beats baselines, that is a valid, publishable scientific result.

---

## 1. Diagnosis — why the current system behaves the way it does

### 1.1 Why variance dominates signal
Financial M5 returns have an extremely low signal-to-noise ratio. The
predictable component (if any) is a few basis points; the noise is orders of
magnitude larger. On top of that low-SNR substrate, **five compounding variance
sources** stack up:

1. **Heavy-tailed reward.** Realised PnL is dominated by a handful of trades
   (measured: top-10 trades = **71%** of net PnL on the LSTM 2025 run). The
   learning signal itself is heavy-tailed, so whether a given seed's policy
   happens to catch or miss those few moves swings the result enormously.
2. **Undertraining.** 50k–150k steps is far from convergence for this problem.
   Different seeds stop at different points on a noisy, near-flat loss surface →
   materially different policies.
3. **On-policy gradient noise.** PPO estimates gradients from stochastic
   rollouts; with a tiny true edge the gradient is mostly noise, and the value
   function's explained-variance is low, making advantages noisy → unstable
   updates.
4. **Single-path OOS evaluation.** Each fold is evaluated on *one* realised price
   path with a deterministic policy. That is one sample from a high-variance
   distribution — n=1 per cell.
5. **Non-stationarity (regime).** The data-generating process changes over time,
   so "variance across folds" is partly real regime change, not estimation noise.

Net effect: measured **compounded-return std ≈ 106%** with a mean ≈ +23% over 4
seeds. The standard error of the mean is ~half the mean — i.e., **the mean is
statistically indistinguishable from zero.**

### 1.2 Why RL behavior is unstable
- PPO's value head cannot fit lumpy, heavy-tailed returns well (low explained
  variance), so the advantage estimates that drive the policy are noisy.
- The entropy bonus encourages action churn; combined with a flip-friendly
  Discrete(4) action space and weak per-trade cost in the reward, the path of
  least resistance for some seeds is **overtrading** (MLP: 4066 trades; one
  all-groups fold: 1095 trades) — costs grind equity to the drawdown floor.
- Reward scaling/clipping can flatten the already-weak gradient.

### 1.3 Why single backtests are misleading
A single backtest is one draw from a distribution with σ≈106% compounded. With
n=1 and that σ, the draw carries almost no information about the mean. Worse,
humans and search procedures select the *good* draws (the +86%/+238% we first
saw), which is implicit multiple-testing / selection bias. The deflation from
+238% (single) → +44% (1-seed WF) → −62%…+204% (multi-seed) is exactly this.

### 1.4 Why regime dependence happens
Gold trended strongly in 2023–2024; the LSTM learned **trend-following** and
shone there. 2021–2023 (chop) and parts of 2024–2025 punished the same behavior.
The learned policy class has a **conditional** edge (works in trends), not an
unconditional one. A single policy trained on mixed history either overfits the
dominant training regime or gets lucky on the test regime. This is the single
most important *substantive* clue we have (see §9.4).

---

## 2. Engineering solutions (by problem)

### 2.1 Reducing seed sensitivity
- **Seed-ensembling (inference):** train K∈[5,10] policies with different seeds;
  at each step average the action *probability distributions* (softmax of
  logits) and act on the mean. If per-seed errors are partly independent,
  variance falls ~1/K. Cheapest high-leverage win after the env rewrite.
- **Stochastic Weight Averaging (within run):** average policy weights over the
  last M checkpoints of a single run (same basin) → flatter minimum, lower
  variance. (Averaging weights *across seeds* is unsafe — different basins.)
- **Train to convergence** (see 2.5) — most undertraining variance is removable.
- **Always report distributions, never a single run.**

### 2.2 Stabilizing PPO
- Larger `n_steps` and `batch_size` → lower-variance gradient estimates.
- **LR decay** (linear schedule) and lower base LR.
- **`target_kl` early-stop** per update to prevent destructive steps.
- **Entropy schedule:** higher early (explore), decayed later (stop churning).
- Monitor **`explained_variance`** of the value function; if it stays near 0 the
  value head is useless → consider returns normalization, reward redesign, or a
  separate value network.
- Keep gradient clipping; consider value-function clipping.

### 2.3 Reducing overtrading
- **Trade cooldown:** set `min_bars_between_trades` > 0 (e.g., 6–12 M5 bars).
- **Stronger cost in reward:** raise `overtrading_penalty`; add an explicit
  per-position-change cost term so marginal trades are −EV in the reward itself.
- **Confidence threshold:** only enter when `p(action) − p(hold) > τ`, else Hold;
  tune τ on validation. Filters low-conviction noise trades.
- Consider an action space that penalises flips (e.g., position-delta cost).

### 2.4 Robustness across regimes
- **Volatility-targeted position sizing** (see 2.6) — equalises trade risk across
  regimes, shrinking the PnL-distribution tails.
- **Regime as a (causal) feature** first; **regime-conditioned policies** only
  later and cautiously.
- **Domain randomization:** randomise spread/slippage/commission (within
  realistic bounds) during training → policies robust to cost regimes.
- **Train on more regimes:** longer history and/or multiple instruments.
- **Regime-stratified evaluation** so a bull-only edge cannot masquerade as
  general skill.

### 2.5 Sample efficiency / convergence
- The real efficiency lever is **better signal** and the **vectorized env**
  (§7), which makes 1M-step, many-seed runs cheap.
- Off-policy methods (QR-DQN, discrete) reuse data via replay → more
  sample-efficient than PPO, and **distributional** value learning is often more
  stable on noisy rewards. Worth an experiment (medium priority).

### 2.6 Position sizing redesign
- Replace fixed-risk-fraction sizing with **volatility targeting**: lots ∝
  `target_risk / (ATR · contract_size)`, capped. Keeps *per-trade risk* roughly
  constant across calm/turbulent regimes → far more stable equity curve and a
  thinner-tailed return distribution (directly attacks §1.1 and §1.4).
- Lower max leverage; cap Kelly fraction well below full Kelly.

### 2.7 Reducing feature noise
- **Make feature selection mandatory** (it is already built but optional):
  variance filter → correlation prune → **per-feature/per-group walk-forward A/B**
  with the robustness score (§6). A feature is kept only if it improves the
  *distribution*, not a single run. Naively adding all 6 groups raised
  compounded std 31→89 — feature explosion is a measured failure mode here.
- Prefer few, orthogonal, economically-motivated features over many correlated
  indicators. Optional: PCA/whitening fit on train only.
- Regularise the feature extractor (dropout, weight decay).

### 2.8 Evaluation rigor (see also §6)
- Baselines first: **buy-and-hold, always-flat, random-action (cost-aware)**. Any
  result must be compared against these. If the agent cannot beat buy-and-hold
  net of costs across the distribution, there is no edge.
- Bootstrap confidence intervals; **Deflated Sharpe Ratio** to account for the
  many configs we have tried.

---

## 3. Detailed component plans

| Component | Plan | Risk |
|---|---|---|
| **Ensemble inference** | `EnsemblePolicy` wraps K saved models; `predict` averages softmax probs; act on argmax, or Hold if max mean-prob < τ. | Low; clear win |
| **Policy averaging** | SWA over last M checkpoints within a run (BatchNorm-free, so plain weight mean). Not across seeds. | Low |
| **Simple vs large arch** | Keep small CNN/MLP. Evidence: smaller generalised better; Transformer was worst + overtraded. | Scaling up = overfit trap |
| **Regularization** | AdamW weight decay, dropout in extractor, entropy + LR schedules, `target_kl`, domain randomization of costs. | Low |
| **Reward redesign** | Move to **Differential Sharpe Ratio** (incremental risk-adjusted reward) OR `Δequity − λ·turnover − κ·drawdown`, with per-trade cost emphasis and per-step reward capping to tame heavy tails. Keep mark-to-market (no hold-the-loser exploit). | Medium; validate it doesn't induce inactivity |
| **Position sizing** | Volatility targeting (constant per-trade risk), lower leverage cap. | Low–medium |
| **Trade cooldown** | `min_bars_between_trades` ∈ [6,12]; enforced in env (already supported). | Low |
| **Confidence threshold** | τ on `p(act)−p(hold)`, tuned on validation slice. | Low |
| **Regime detection** | Causal: rolling realised-vol + trend-slope → discrete regimes via train-fit thresholds or an online HMM/rolling-kmeans fit on train only. Use as feature/gate. | Medium; leakage care |
| **Regime-conditioned policy** | Phase 2: separate policies per regime or MoE gated by regime. | High overfit risk; do last |

---

## 4. Structural questions — honest assessment

| Question | Verdict | Reasoning |
|---|---|---|
| **Is M5 too noisy?** | **Very likely YES** | M5 has the lowest SNR and the highest relative transaction cost. This is probably the single biggest cause of instability. |
| **Would higher TF (H1/H4/D1) help?** | **Likely YES — top hypothesis** | Higher SNR, lower relative costs, fewer decisions → lower variance. Cost: fewer bars/less data. Test H1 first. |
| **Continuous action space?** | **Maybe, medium** | Lets the policy express conviction via size and unlocks SAC, but adds variance/complexity. Vol-targeted *discrete* sizing likely captures most of the benefit with less risk. |
| **SAC / other algos vs PPO?** | **Worth testing, not a silver bullet** | SAC (continuous, entropy-regularised) is stable + sample-efficient but needs a continuous env. QR-DQN (discrete, distributional) is a cheaper test. The bottleneck is signal/variance, not the optimiser. |
| **Offline RL?** | **Likely dead end (now)** | We synthesise data from the sim; offline RL shines when learning from a fixed log of (good) behaviour we don't have. Revisit only with real expert/trade logs. |
| **Mandatory feature selection?** | **YES** | Feature explosion is a *measured* variance amplifier. Selection must gate every feature via the robustness score. |

---

## 5. New experimental protocol (mandatory going forward)

- **Seeds:** ≥ 8 (10 preferred). Report per-seed results.
- **Folds:** ≥ 6, expanding or rolling, with embargo gap; consider
  **purged/embargoed** CV (López de Prado) to kill boundary leakage.
- **Unit of result:** the **distribution over the seed×fold grid** (≥ 48 cells),
  not a point metric. Report **median, IQR, min (worst case), % of cells
  profitable**, and an equity-curve **fan chart**.
- **Significance:** bootstrap CI on the median OOS return; **Deflated Sharpe
  Ratio** (penalises the many trials run); require the metric's CI to clear the
  **buy-and-hold and random baselines** with non-overlapping intervals. Expect
  "not significant" — and accept it as a real answer.
- **Robustness Score (single scalar to rank configs):**
  ```
  RS = median(compounded)            # central tendency
       − 0.5 · IQR(compounded)       # penalise instability
       − 1.0 · |min(compounded)|⁻    # penalise worst-case (downside only)
       + 10 · (frac_cells_profitable − 0.5)
  ```
  (weights to be calibrated). Rank by RS, never by peak return.
- **Pre-registration:** fix seeds, folds, and the metric *before* running, to
  avoid post-hoc seed/fold cherry-picking.

---

## 6. Future-proof, high-throughput architecture (the enabler)

The current `GoldTradingEnv` steps one bar at a time in Python; `DummyVecEnv`
runs envs sequentially (one CPU core, ~15–20% total CPU); `SubprocVecEnv` OOM'd
(8 torch interpreters) and adds per-step IPC that negates parallelism for a light
env. **None of the variance work is feasible at current throughput.**

**Plan: a NumPy-vectorized batched environment.**
- Maintain state for **N parallel lanes** as vectors: `position[N]`, `lots[N]`,
  `entry_price[N]`, `balance[N]`, `equity[N]`, `step_ptr[N]`, `peak[N]`.
- Pre-compute the (normalized) feature matrix and price arrays **once**; lanes
  index in with their own pointers (random starts per lane).
- `step(actions[N])` advances all lanes in **one vectorized pass** (masked numpy
  ops for open/close/SL/TP/PnL/reward) — no per-env Python loop, no IPC.
- Expose via the SB3 `VecEnv` interface (or Gymnasium vector API) so PPO/DQN use
  it unchanged. Optionally a torch-tensor version for on-GPU stepping later.
- **Validation:** assert lane-by-lane equivalence to the current scalar env on a
  fixed seed before trusting it.
- **Expected gain:** 10–100× rollout throughput on CPU; bigger GPU batches; makes
  1M-step × 8-seed × 6-fold studies routine. This is the **highest-leverage
  engineering task in V3.5** because it unblocks every experiment below.

---

## 7. Prioritization by impact

**Highest impact**
1. **Vectorized env** — unlocks the compute budget for everything else.
2. **Higher-timeframe (H1) test** — likely the biggest SNR/stability lever.
3. **Baselines + distributional/robustness evaluation** — without it we keep
   fooling ourselves.
4. **Train-to-convergence + many seeds/folds + seed-ensembling** — directly
   attacks variance.
5. **Reward/cost redesign + cooldown + confidence threshold + vol-targeted
   sizing** — stabilises behavior and the PnL distribution.
6. **Mandatory feature selection.**

**Medium impact**
- Regime-as-feature; QR-DQN experiment; SWA; domain randomization; continuous
  actions + SAC (uncertain).

**Low impact**
- Larger/Transformer tuning; exotic indicators without validation; offline RL.

**Dangerous (likely to *increase* overfitting / self-deception)**
- Scaling model capacity.
- Adding many features without selection.
- Regime-conditioned policies with few regime samples.
- Any tuning on the 2025 test set.
- Selecting the best seed/fold post-hoc; optimising for peak (not median/robust).
- Continuous sizing without strict risk caps.

---

## 8. Dead ends, promising bets, and trust levels

**Likely dead ends**
- Bigger networks; M5 high-frequency churn given costs; offline RL now; chasing
  the +204% seed; feature explosion.

**Likely promising**
- Higher timeframe; seed-ensembling; vectorized env (enabler); cost-aware reward
  + cooldown + confidence gate; volatility-targeted sizing; regime-as-feature;
  distribution-based, baseline-relative evaluation; disciplined feature selection.

**Metrics that currently CANNOT be trusted**
- Any single-run return / Sharpe / Calmar.
- Point estimates of compounded return.
- The "core CNN beats all-groups" comparison (within the seed-noise band).
- LSTM +86% / CNN +238% — artefacts of seed/period selection.

**Results that MAY still contain real signal**
- **Qualitative behavior:** LSTM → trend-following; MLP/Transformer → overtrading.
  These reproduced and are economically interpretable.
- **Regime dependence:** consistent across runs (good in trending 2023–24). This
  hints at a *conditional* trend edge worth isolating — the most promising lead.
- **Costs make M5 churning −EV:** robust and important (informs cooldown/threshold).
- **Feature explosion raises variance:** robust direction (magnitude noisy).

---

## 9. Roadmap, action plans, and 30-day experiments

### 9.1 Roadmap placement
Insert **V3.5 — Variance Reduction & Stability** between V3 (Deep RL) and V4
(Real-time). V4 (live/paper trading) is **blocked** until V3.5 produces a config
whose OOS distribution is stable and baseline-beating (or a definitive negative
result). Do not deploy anything chosen on a single backtest.

### 9.2 Engineering action plan
1. NumPy-vectorized batched env behind the SB3 VecEnv interface (+ equivalence
   test vs current env).
2. Baseline agents (buy-and-hold, flat, random) + cost-aware backtest of each.
3. Evaluation upgrade: seed×fold grid runner, distribution reporting (median/IQR/
   worst/% positive), fan charts, Robustness Score, bootstrap CI, Deflated Sharpe.
4. Reward module refactor (pluggable: current vs Differential-Sharpe vs
   turnover-penalised) + volatility-targeted sizing + cooldown + confidence gate.
5. `EnsemblePolicy` + SWA utilities.
6. Make `FeatureSelector` mandatory in the pipeline; per-group walk-forward A/B
   driven by the Robustness Score.

### 9.3 Research action plan
- Establish a **rigorous variance baseline** (8 seeds × 6 folds) for MLP and CNN
  at M5 and H1, trained to convergence. This is the new reference point.
- Test hypotheses one at a time, each judged by ΔRobustnessScore on the grid:
  H1-timeframe, vol-targeted sizing, cost-aware reward, cooldown/threshold,
  ensembling, regime-as-feature, QR-DQN.
- Keep a frozen 2025 test set; tune only on validation slices.

### 9.4 Recommended immediate next steps (this week)
1. Build the vectorized env (unblocks everything).
2. Add baselines + distribution/robustness reporting.
3. Re-run the multi-seed CNN at M5 **and H1** with the new fast env and longer
   training, to (a) confirm variance and (b) get the first read on the
   timeframe hypothesis.

### 9.5 30-day experiment schedule
- **Week 1 — Infrastructure:** vectorized env (+equivalence test); baselines;
  distributional/robustness evaluation harness.
- **Week 2 — Baseline & timeframe:** 8×6 grid for MLP/CNN at M5 vs H1, trained to
  ~1M steps. Deliverable: variance baseline + H1-vs-M5 verdict (RS + fan charts).
- **Week 3 — Stability levers:** cost-aware reward + vol-targeted sizing +
  cooldown + confidence threshold. Re-run grid; measure variance reduction vs
  Week 2. Deliverable: does stabilisation shrink IQR and lift % profitable?
- **Week 4 — Ensembling + regime + selection:** seed-ensemble + SWA;
  regime-as-feature; mandatory per-group feature A/B. Final RS-ranked comparison
  against baselines. **Go/No-Go:** does any config's OOS *distribution* beat
  buy-and-hold with non-overlapping CIs at acceptable variance? If not, document
  the negative result — it is scientifically valid and the honest outcome.

---

## 10. Success criteria for V3.5 (note: NOT profitability)
1. Compounded-return **std across seeds cut by ≥ ~2×** vs the −62%…+204% baseline.
2. Every reported result is a **distribution over ≥ 8×6 cells**, with CIs and
   baseline comparison — no more point metrics.
3. A reproducible, pre-registered protocol with a single Robustness Score.
4. Throughput high enough that an 8×6 grid at 1M steps runs in hours, not days.
5. A clear, honest verdict per hypothesis (H1, sizing, reward, ensembling,
   regime), including documented negative results. **Exposing "no edge" rigorously
   is a successful outcome of this phase.**
