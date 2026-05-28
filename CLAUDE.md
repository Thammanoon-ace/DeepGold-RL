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

## Scientific findings — what V0–V3.5 has actually established

These are non-obvious, hard-won project facts. Treat as priors:

1. **Variance dominates signal.** Across 3-seed grids, single-seed std on excess
   reward = 46.6 percentage points. Any single-backtest claim is meaningless.
   Always report a distribution + bootstrap median CI + Robustness Score (see
   [validation/robustness.py](validation/robustness.py)).
2. **No reliable edge has been found** on gold M5 with the configurations
   tested. The buy-and-hold baseline (+27% over the test period) is unbeaten
   on raw return or Sharpe by any single-seed config.
3. **Only the reward *objective* (excess reward) has improved results.**
   Per [docs/EXPERIMENT_SUMMARY.md §15](docs/EXPERIMENT_SUMMARY.md), the
   excess-reward ensemble beat BH per-fold return 75% of folds and 100% with
   a 3-seed ensemble. But its single-seed median CI still straddles 0 — the
   big-seed verdict is the open question.
4. **DSR reward was dropped** (2026-05-27): 12-cell grid, 0% beat BH,
   robustness −93.7, ensemble even worse. Do not revisit.
5. **Every complexity lever tested has failed:** hybrid CNN-LSTM, transformer,
   volatility-targeted sizing, regime features as observations, deeper nets.
   They overfit the low-SNR data and the held-out distribution looks worse,
   not better. See [docs/V3_5_VARIANCE_REDUCTION.md](docs/V3_5_VARIANCE_REDUCTION.md).
6. **Walk-forward + per-fold normalizer is mandatory** to avoid leakage. The
   normalizer must be fit on train-only per fold ([validation/grid.py](validation/grid.py)).

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

## Open questions (current, as of 2026-05-28)

1. **Does excess reward beat buy-hold with statistical significance** at large
   seed count (16–32 seeds × 5 folds)? — Big-seed grid in progress; the
   prior 3-seed CI was `[−14.5, +58.0]` (straddles 0).
2. **Will the GPU engine's 16-member ensemble recover** what the 3-member
   ensemble lacked? Or is GPU-policy diversity an inherent ensemble-killer?
3. **Does the framework generalize** to other instruments (EURUSD, BTC,
   indices) or is the negative result gold-M5-specific?

If those answers all come back negative on a definitive run, the honest
project verdict is: *RL on retail-data + retail-compute is not a practical
trading edge — the science is the negative result*.
