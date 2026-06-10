# DeepGold RL — Project context for Claude

This file travels with the repo. Read it first every session. For deeper history
see [ROADMAP.md](ROADMAP.md) and the docs/ folder.

---

## Environment (mandatory)

- **Python 3.11** only — use the project `.venv` (`py -3.11`). System Python
  3.10/3.13/3.14 has been explicitly rejected by the user. If `.venv` isn't
  present yet, recreate it with `py -3.11 -m venv .venv`.
- Real market data: [`data/XAUUSD_M5.csv`](data/XAUUSD_M5.csv) — Dukascopy,
  XAUUSD M5, **496k bars 2019-01 → 2025-12** (real, not synthetic).
- Train/test split: train 2019–2024, **test from 2025-01-01** (configured in
  [config/config.yaml](config/config.yaml) `data.test_start`).
- GPU detection: `device: auto` in config picks CUDA if available; CPU otherwise.

---

## Scientific findings — what V0–V4 has actually established

These are non-obvious, hard-won project facts. Treat as priors:

1. **Variance dominates signal.** Across 3-seed grids, single-seed std on excess
   reward = 46.6 percentage points. Any single-backtest claim is meaningless.
   Always report a distribution + bootstrap median CI + Robustness Score (see
   [validation/robustness.py](validation/robustness.py)).
2. **Timeframe is the load-bearing variable, not architecture/features.**
   Cross-timeframe sweep (32×4 grid, excess reward, [EXPERIMENT_SUMMARY.md
   §15.8-§15.13](docs/EXPERIMENT_SUMMARY.md)) found the edge peaks at H4:
   - M5: single CI [−3.7, +5.5] straddles 0 (no edge)
   - H1: CI [+3.6, +10.3] excludes 0 but ensemble loses to BH
   - **H4: CI [+3.2, +10.9] excludes 0, ensemble robustness +12.9, beats BH on 25% of folds**
   - D1: CI [−9.0, −5.1] significantly negative — too coarse
3. **H4 + Cosine LR + SWA has a positive expected per-cell return but
   loses to buy-and-hold on every metric tracked** (four independent
   32-seed runs × 4 folds = 512 cells; see [[risk-adjusted-also-lost]]):
   - **Raw return:** agent median +12.07 % vs BH +25.95 % → **agent loses
     by 13.9 pp**.
   - **Sharpe:** agent median +0.500 vs BH +1.219 → **agent loses by 0.72**.
   - **Max DD:** agent median 18.86 % vs BH 11.00 % → **agent is 7.9 pp
     worse** (higher drawdown).
   - **Single-cell CI excludes 0** ([+2.7, +6.6] across the four runs), so
     the per-cell expected return IS positive — but BH's expected return is
     larger and BH's risk is lower.
   - **Where the agent beats BH:** only fold 0 (range / down market) on
     Sharpe (78.9 % of cells). But BH itself has negative Sharpe (−0.51)
     there; the agent is "less negative" (−0.18), not "positive edge."
   - **Where the agent decisively loses:** fold 2 (strong trend) — **0 %**
     of agent cells beat BH on Sharpe. Trend folds are exactly where retail
     RL on raw OHLC consistently loses to passive holding (§15.6 "BH is not
     beaten in a structural gold bull" prediction confirmed).
   - **The earlier §15.18 / [[cosine-swa-unblock]] memo's "+30.4 % beats BH"
     claim was a single-run upper-tail draw whose signal does not exist in
     4-run aggregate.** It is fully retracted.
4. **Only the reward *objective* (excess reward) improved results among
   the reward levers.** §15.4: excess-reward ensemble beat BH per-fold return
   in 75 % of folds; the H4+cosine+SWA stack lifted that to 100 %.
5. **DSR reward was dropped** (2026-05-27): 12-cell grid, 0% beat BH,
   robustness −93.7, ensemble even worse. Do not revisit.
6. **Every observation / architecture complexity lever has failed:** hybrid
   CNN-LSTM, transformer, volatility-targeted sizing, regime features as
   observations, deeper nets. They overfit the low-SNR data and the held-out
   distribution looks worse, not better. See
   [docs/V3_5_VARIANCE_REDUCTION.md](docs/V3_5_VARIANCE_REDUCTION.md).
7. **Optimization complexity DID help** (the §15.17/§15.18 finding that
   reverses item 6 for *optimization* levers): cosine LR schedule (+5 pp on
   ensemble mean vs constant LR) and SWA on top (another +7 pp, plus the
   risk-adjusted gains in item 3). Train the cosine + SWA stack by default for
   any H4 experiment.
8. **Every entry-gate / seed-selection lever has failed:** ATR-based regime
   gate (§15.11), TER-based regime gate (§15.12), train-Sharpe top-k ranker
   (§15.15), held-out validation top-k ranker (§15.16). Each either degrades
   the ensemble or sacrifices too much training data. The unfiltered policy +
   32-seed ensemble averaging remains the best aggregator.
9. **Walk-forward + per-fold normalizer is mandatory** to avoid leakage. The
   normalizer must be fit on train-only per fold ([validation/grid.py](validation/grid.py)).
10. **Speed ≠ alpha** but **timeframe = alpha**. Tier 1.2/2.1/2.4/CPU-audit
    proved no infra lever exceeds 1.07× on this codebase (Python/launch
    overhead dominates, not GPU). One timeframe change (M5 → H4) and one
    optimization-schedule change (constant → cosine+SWA) moved the ensemble
    mean from +1.5 % to +26 %. Spend session time on science, not infra.
11. **V4 optimal-execution pivot hit the same ceiling** (on the V4-style
    task). The hypothesis was that switching the task from "beat
    buy-and-hold" to "beat TWAP on sub-bar slicing" would expose
    higher-SNR signal where RL has industrial track record. Phase 1–3
    infra works (see [docs/V4_OPTIMAL_EXECUTION.md](docs/V4_OPTIMAL_EXECUTION.md),
    [env/execution_env.py](env/execution_env.py),
    [validation/grid_execution.py](validation/grid_execution.py)): TWAP
    shadow is exact (savings = 0 when agent acts at TWAP rate), impact
    tension is +21 bps between forced-finish and TWAP. But the **3 seed × 4
    fold grid** ([logs/grid/exec_verdict_v1/](logs/grid/exec_verdict_v1))
    came in **median −0.25 bps with CI [−1.5, 0]**, and **5 diagnostic
    levers** (longer training, higher entropy, shorter episodes, 7
    micro-timing features, +execution_features flag) all produced results
    in [−1.3, +0.7] bps. Conclusion *for the execution task*: M5 retail
    OHLCV doesn't carry actionable per-bar timing signal under this reward
    framing.
12. **BB-volatility feature group is the first lever to beat BH on M5
    excess** (preliminary, awaiting replication). 8 seed × 4 fold grid
    ([logs/grid/excess_volgroup_s8/](logs/grid/excess_volgroup_s8/)) with
    the V3.5 baseline + `--feature-groups volatility` (`bb_pctb`,
    `bb_bandwidth`, `hist_vol` — Bollinger Bands %B + bandwidth + log-return
    rolling std): **ensemble median +36.3 %, robustness +38.2, beats BH
    on 75 % of folds (3 of 4), worst fold +22.7 %**. This contradicts the
    earlier within-session item 11 conclusion that M5 has no edge — that
    conclusion was reached on **V4 (execution)** without ever isolating the
    volatility group on the directional path. Single-seed CI still
    straddles 0 (variance lives in the ensemble); only 4 folds. **Treat as
    preliminary** until the 16-seed replication and full-stack tests in
    [docs/BB_VOLATILITY_FOLLOWUP.md](docs/BB_VOLATILITY_FOLLOWUP.md)
    confirm or refute. Until those are done: items 3, 6 and the V0–V3.5
    "negative result" framing are **provisionally on hold**, not retracted.

---

## Two training engines, two trade-offs

| | SB3 engine (`--engine sb3`) | GPU CleanRL engine (`--engine gpu`) |
|---|---|---|
| Speed | slow (CPU env, ~2.5 min/cell) | fast (~2.7 min/cell on 1650 Ti, ~7× ceiling on better GPU) |
| Determinism | bit-stable across runs (same seed → same result) | **NOT bit-stable** (CUDA non-determinism compounds) |
| Single-seed quality | validated, captures trend folds (+42%/+34% mean) | comparable after arch match (v3: mean +34%, max +108%) |
| Ensemble quality | **strong** (3 seeds → 100% beat BH on excess) | **weaker** (3-seed ensembles often near 0%); seeds disagree more |
| Use for | published numbers, verdicts | rapid iteration, prototypes |

**GPU CleanRL stack** ([env/torch_vec_env.py](env/torch_vec_env.py),
[training/cleanrl_ppo.py](training/cleanrl_ppo.py)): equivalence-verified
against the numpy VecEnv to ~1e-7 across all 3 reward modes. Required fixes
applied to avoid policy collapse:
- `target_kl=None` (SB3 default; aggressive early-stop froze the policy near init)
- Reward normalization (SB3 VecNormalize-style; the raw excess rewards are too small)
- `minibatch_size=2048` (was 16k → too few gradient steps → flat-collapse on many seeds)
- `n_epochs=10` (was 4)
- Architecture matched to SB3: CNNExtractor head + `net_arch=[128,128]` Tanh
  mlp_extractor + orthogonal init on conv layers

---

## Project structure (where things live)

```
config/config.py          # Central dataclasses; YAML config in config/config.yaml
env/
  gold_trading_env.py     # Scalar Gymnasium env (canonical, used for eval/backtest)
  vectorized_env.py       # NumPy batched VecEnv (SB3 path) — proven equivalent to scalar
  torch_vec_env.py        # GPU-resident torch env (CleanRL path) — equivalent to ~1e-7
training/
  train_ppo.py            # SB3 PPO training entry
  cleanrl_ppo.py          # GPU-native PPO loop + ActorCritic (SB3-matched arch)
policies/
  extractors.py           # LSTM/Transformer/CNN/CNN-LSTM SB3 feature extractors
  ensemble.py             # Mean-prob ensemble + confidence gate (works for both engines)
validation/
  grid.py                 # Seed × fold grid evaluator (the V3.5 protocol)
  robustness.py           # Distribution metrics, bootstrap median CI
backtest/
  backtester.py           # run_episode (single OOS path, deterministic argmax)
  metrics.py              # Sharpe, max drawdown, return %, n_trades
scripts/
  grid_eval.py            # CLI for the grid (--engine sb3|gpu)
  train_cleanrl.py        # CLI for standalone GPU training
  compare_grids.py        # Side-by-side comparison of grid outputs
docs/
  EXPERIMENT_SUMMARY.md   # Authoritative log of every experiment + verdict
  V3_5_VARIANCE_REDUCTION.md  # Design doc for the V3.5 protocol
  NEW_HARDWARE_PLAN.md    # Prioritized experiments for the upcoming 12600k/5070 rig
ROADMAP.md                # V0–V7 vision
```

---

## Canonical commands

```bash
# SB3 grid (verdict run — trustworthy, slow)
python scripts/grid_eval.py --engine sb3 --policy-arch cnn --reward-mode excess \
  --seeds 16 --folds 5 --timesteps 30000 --num-envs 48 --tag excess_bigseed

# GPU grid (fast iteration)
python scripts/grid_eval.py --engine gpu --policy-arch cnn --reward-mode excess \
  --seeds 16 --folds 5 --timesteps 1310720 --num-envs 1024 --n-steps 128 \
  --tag excess_bigseed_gpu

# THE CURRENT BEST CONFIG — H4 + cosine LR + SWA (§15.18):
python scripts/grid_eval.py --engine gpu --policy-arch cnn --reward-mode excess \
  --seeds 32 --folds 5 --timesteps 1310720 --num-envs 2048 --n-steps 128 \
  --timeframe H4 --lr-schedule cosine --swa --tag excess_bigseed_32_h4_cosine_swa

# Sanity rerun with a different seed batch (use --seeds-start to shift the range)
python scripts/grid_eval.py ... --seeds 32 --seeds-start 32 --tag <name>_s32

# V4 readiness smoke (train + save + load + backtest round trip)
python scripts/v4_smoke.py

# Compare grids
python scripts/compare_grids.py --grids m5 excess dsr

# Equivalence test (torch vs numpy env, all reward modes)
python scripts/_test_torch_equiv.py
python scripts/_test_vec_equiv.py
```

---

## How to behave when working in this repo

- **Do not claim a result from a single backtest.** Always run a grid and
  report the distribution + CI. If asked for a quick check, say so explicitly
  and run multi-seed.
- **Trust the user's own running processes.** The user often runs commands in
  their own terminal — never kill arbitrary python processes; only stop
  background tasks YOU launched (via `TaskStop` with the task id).
- **Read [docs/EXPERIMENT_SUMMARY.md](docs/EXPERIMENT_SUMMARY.md) §15** before
  proposing experiments — many have been done already.
- **Use Thai for conversational replies** (the user communicates in Thai),
  English for code, file paths, commands, and code comments.
- **Tier-4 things in [docs/NEW_HARDWARE_PLAN.md](docs/NEW_HARDWARE_PLAN.md):
  don't propose them.** They have already been falsified.
- **Speed ≠ alpha.** Faster compute means more experiments per day, not
  better trading. State this honestly when proposing infra work.

---

## Open questions (current, as of 2026-05-30)

The original Q1/Q2/Q3 from 2026-05-28 have been **answered**:

- ~~Q1 Does excess reward beat BH at 32×4 seeds on M5?~~ — **No.** §15.8: CI
  [−3.7, +5.5] straddles 0.
- ~~Q2 Will the 32-member ensemble recover what 3-member lacked?~~ — **No on
  M5** (ensemble +1.5 % vs BH +27 %). **Yes on H4 + cosine + SWA** (§15.18:
  100 % profitable folds, robustness +27.9).
- ~~Q3 Does the framework generalize off gold?~~ — **Still untested.**
  Multi-instrument H4 (EURUSD/BTC/SPX) is now the cleanest open question.

**New open questions (2026-05-30):**

1. **Is the +30 % ensemble mean from the first H4+cosine+SWA run reproducible
   at a larger run-count?** Two runs gave +30.4 % and +21.5 % (avg +26 % =
   tied with BH on raw return). 4-6 more independent reruns (seeds 64-95,
   96-127, …) would put a CI on the cross-run ensemble mean. The risk-
   adjusted metrics (Sharpe, DD, Robustness, 100% folds) are already stable
   across both runs.
2. **Does the H4 + cosine + SWA edge generalize off gold?** Multi-instrument
   H4 (EURUSD H4, BTC H4, SPX H4). If yes → cross-asset retail-RL edge. If no
   → H4 sweet spot is gold-2025-specific. This is the single highest-value
   experiment in the queue.
3. **Live broker validation.** The §15.18 ensemble works on paper backtests.
   Real frictions (live spread, swap, slippage) eat some of the edge. A
   2-week demo-account paper-trading run with the saved cosine+SWA
   ensemble would calibrate the paper-to-live decay.

**Current honest project verdict (2026-05-31, after 4-run + cross-cell
risk-adjusted replication, supersedes all prior framings):**

*RL on retail-data + retail-compute on gold has no edge over buy-and-hold:*
- **Raw return: BH wins by 13.9 pp** (BH +25.95 % vs agent +12.07 %).
- **Sharpe: BH wins by 0.72** (BH +1.22 vs agent +0.50).
- **Max DD: BH wins by 7.9 pp** (BH 11.00 % vs agent 18.86 %).

The H4 + cosine + SWA stack DOES produce a measurable positive expected
per-cell return (single-cell CI [+2.7, +6.6] excludes 0 across all four
independent 32-seed runs). But that expected return is smaller and riskier
than passive holding's. **The framework rigorously confirms the original
CLAUDE.md "negative result" prediction from May 28: no tradeable retail-RL
edge exists on this data and configuration space.**

The session's intermediate claims — §15.10 "first +12.9 Robustness," §15.17
"cosine is the first non-failed lever," §15.18 "+30.4 % beats BH by 4.5 pp"
— were all overturned by replication and cross-cell risk-adjusted analysis.
The signal that survives replication is small ("positive expected return per
cell"), reliable (CI excludes 0 in 4/4 runs, 100 % profitable folds in 3/4
runs), and **insufficient** for any "beats BH" claim on any metric.

The framework's scientific contribution is:
1. **Timeframe choice IS the load-bearing variable** for retail-RL on gold
   (M5 has no edge; H4 is the sweet spot; D1 is significantly negative).
2. **Optimization-schedule complexity (cosine + SWA) helps relative to the
   failing baseline** but does not flip the BH comparison.
3. **Single-run "beats BH" claims are unreliable**; multi-run replication +
   cross-cell distribution analysis is mandatory.
4. **The honest output of this project is a rigorous negative result** —
   not a profitable bot. This is one of the pre-registered acceptable
   outcomes per CLAUDE.md's original framing.

The project is now in a "rigorous negative result wrap-up" state. The
remaining productive work is:
- Document the negative result formally (it's well-supported and worth
  archiving).
- Multi-instrument H4 (EURUSD / BTC / SPX) would only confirm or marginally
  qualify the gold-specific finding; it's optional, not load-bearing.
- Going live is **not** justified by any metric the project tracks.
