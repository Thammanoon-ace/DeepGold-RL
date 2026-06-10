# V4 — Optimal Execution (pivot from directional trading)

**Status:** active research direction, opened 2026-06-10 after the V0–V3.5
*"does RL beat buy-and-hold on directional gold trades?"* question was
definitively answered (no — see [CLAUDE.md](../CLAUDE.md) and
[docs/EXPERIMENT_SUMMARY.md §15](EXPERIMENT_SUMMARY.md)).

> **Pivot rationale.** The directional verdict is settled with hard evidence
> (512 cells, 4×32-seed runs, single-cell CI [+2.7, +6.6] excludes 0 but loses
> to BH by 13.9 pp). The infrastructure — torch-resident env, walk-forward grid
> evaluator, robustness/CI protocol, seed ensembling — is high-quality and
> should be reused. Optimal execution is where RL **actually shows industrial
> edge** in finance, has higher SNR than directional prediction, and fits
> existing infra with localised changes.

---

## 1. Problem statement

> *Given an order to trade ``target_lots`` over a deadline of ``deadline_bars``,
> minimize the implementation shortfall* (the difference between the
> volume-weighted average fill price and a reference price) *vs the **arrival
> price** at the start of the execution window. Compare to a TWAP baseline.*

**Why higher SNR than V0–V3.5:**

| | V0–V3.5 directional | V4 execution |
|---|---|---|
| Decision horizon | weeks → months | minutes → hours |
| Reward time scale | episode (~thousands of bars) | episode (~64 bars) |
| Counterfactual baseline | buy-and-hold (a *return*, dominated by long-term drift) | TWAP / arrival (a *price difference*, near-zero-mean & symmetric) |
| Source of edge needed | predicting cumulative return sign | predicting *intra-episode* price wiggles |
| Theoretical floor | beating Sharpe of passive holding | beating uniform slicing |
| Industry track record | weak — gold-2025 verdict negates retail RL | strong — Almgren–Chriss family + RL improvements widely deployed |

Crucially, the *expected* reward for a TWAP agent is **exactly the arrival
price** (by definition of TWAP under no impact). Any agent that learns to
*time* slices into transient favourable price moves can improve on this in
*both* tails (long and short orders) — directional drift is irrelevant.

---

## 2. Environment design (`env/execution_env.py`)

Single-asset, single-bar tick (we use the existing M5 / H1 / H4 OHLCV data —
same `data/XAUUSD_M5.csv`). Episode = one execution task:

* Sample `side ∈ {+1, -1}` (long buy / short sell).
* Sample `target_lots ∈ [target_lots_range]` (default `(0.5, 2.0)`).
* Sample `deadline_bars ∈ [deadline_range]` (default `(32, 128)`).
* Start at a random bar (walk-forward fold's training slice). `arrival_price = close[t]`.

### Action space (discrete, 4)

The agent chooses an execution **rate multiplier** vs TWAP this bar:

| Action | Multiplier | Slice this bar |
|---|---|---|
| 0 | `0.0` | pause |
| 1 | `0.5` | half TWAP rate |
| 2 | `1.0` | TWAP rate (≈ `remaining_lots / remaining_bars`) |
| 3 | `2.0` | double TWAP — accelerate |

`slice_lots = action_mult × (remaining_lots / remaining_bars)`,
clipped to `≤ remaining_lots`.

### Fill model

`fill_price = close[t] + side × (spread/2 + slippage + impact_coef × slice_lots)`

The `impact_coef × slice_lots` term creates the **tension** between fast
execution (avoid adverse drift) and small slices (avoid impact). This is
what makes the problem non-trivial vs TWAP.

### Termination & forced completion

* If `remaining_bars == 0` and `remaining_lots > 0`: force-trade the rest at
  the current bar (final bar's close + full impact) — a hard penalty.
* If `remaining_lots ≤ 0`: clean terminal.

### Reward

**Terminal-only** (clean credit assignment):

```
shortfall = (avg_fill_price - arrival_price) * side
reward = -shortfall / arrival_price * 10_000  # bps; positive = beat arrival
```

Optionally we also track `twap_shortfall_bps` (what a TWAP agent would have
paid) so the *headline* metric is `bps_savings = twap_shortfall - shortfall`.

### Observation

Same windowed features (`window × n_features`) + **5 execution scalars**
(replacing V0–V3.5's account scalars — so `obs_dim` stays unchanged and the
existing networks plug in unmodified):

| Scalar | Range | Meaning |
|---|---|---|
| `side` | `{-1, +1}` | long buy or short sell |
| `remaining_lots / target_lots` | `[0, 1]` | how much left to do |
| `remaining_bars / deadline_bars` | `[0, 1]` | time-pressure proxy |
| `(avg_fill - arrival) / arrival × side` | small | running shortfall in *fractional* units (positive = currently winning) |
| `(close[t] - arrival) / arrival × side` | small | current-price advantage vs arrival |

---

## 3. Baselines (must be beaten)

Reported alongside RL in every grid:

1. **TWAP** — exactly `target_lots / deadline_bars` every bar. The natural floor.
2. **VWAP-naive** (we have no real volume — proxy with `|close - open|` per bar). Optional.
3. **Front-load** — execute everything on bar 0 (worst-case impact).
4. **Back-load** — wait until last bar (worst-case adverse drift).
5. **Random slicer** — uniform action sampling.

**Success criterion:** RL ensemble's median `bps_savings_vs_twap` over the
seed × fold grid is **positive** with bootstrap CI excluding 0. (Different
threshold from V3.5 — here zero IS the meaningful baseline, not BH.)

---

## 4. Walk-forward & grid protocol (unchanged)

* `validation/grid.py` re-used as-is. Cells store `bps_savings_vs_twap` rather
  than `return_pct`. The robustness / median-CI plumbing is identical.
* Per-fold normalizer fit on train-only (unchanged).
* Seed ensembling (`policies/ensemble.py`) by averaging action probabilities
  (unchanged; works for either env).

---

## 5. Phased plan

| Phase | What | Effort |
|---|---|---|
| **1** | `env/execution_env.py` (scalar Gymnasium env) + smoke test | ~half day |
| **2** | TWAP / front-load / random baselines + an "obvious" hand-coded heuristic that beats TWAP slightly on noisy bars (sanity floor) | ~few hours |
| **3** | Hook into `train_cleanrl` (no PPO changes needed if obs/action match) — single-cell training; verify it beats TWAP on a single seed | ~half day |
| **4** | First seed × fold grid (3 seeds × 5 folds excess of TWAP). If median bps_savings > 0 with CI excludes 0 → expand to 16-seed verdict. | ~few hours |
| **5** | If V4 has a real edge: hyperparameter sweep, multi-instrument generalisation (the same Tier-3 questions, but now on execution not direction) | weeks |

---

## 6. Risks / unknowns

1. **Our env has no order book — we cap fills at `close ± spread/2`.** Real
   execution would also model partial fills, queue position, etc. Our model
   is a strict simplification; if RL beats TWAP under this simplification it
   may not beat it on real venues. *But* TWAP under the same simplification
   is also unrealistic, so the *relative* comparison is honest.
2. **Impact-coef is a free parameter.** A too-low impact makes TWAP optimal;
   too-high makes any non-TWAP punishing. Pick a value that creates a
   measurable bps_savings spread between front-load and back-load (~5–20 bps
   apart on typical episodes) so there's real signal to learn from.
3. **Discrete-4 actions may be too coarse.** If grid finds RL ≈ TWAP, the next
   experiment is continuous action (Beta-policy PPO) or finer discretisation.
4. **Reward only at terminal** could be sparse. If learning is slow we can add
   a small intra-episode reward shaping (e.g., bonus for slices into bars
   where price moved favourably after the slice).

---

## 7. What stays from V0–V3.5

* Data pipeline, normalizer, walk-forward splitter, robustness/CI protocol,
  grid evaluator, ensemble policy, seed-distribution honesty.
* The training loops (`cleanrl_ppo` + SB3 train_ppo) work unchanged — they only
  see flat observations and discrete actions.
* The conservative reporting bar: **never claim from a single backtest**;
  always grid + CI + ensemble.

## 8. What's specific to V4

* New env, new reward, new baseline. The *question* has moved from "is there
  alpha?" to "is there execution improvement over TWAP?". Different answer
  space, different success criterion, different industry relevance.
