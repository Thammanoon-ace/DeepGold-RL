# BB-volatility breakthrough — follow-up plan

**Status:** preliminary 8-seed result found a major positive lever; needs
replication + stacking before claiming firm verdict change.

**Last session (2026-06-10):** added the `volatility` feature group
(`bb_pctb`, `bb_bandwidth`, `hist_vol`) to the V3.5 baseline (SB3 + CNN +
excess + M5 + 30k steps + 48 lanes), 8 seeds × 4 folds:

| | baseline excess | **+ BB-volatility** |
|---|---|---|
| ensemble median | +9.9 % | **+36.3 %** |
| robustness score | +16.7 | **+38.2** |
| beats BH (75% target) | 0 % of folds | **75 % of folds** |
| worst fold | ~0 % | **+22.7 %** |
| best fold (single seed) | (mid) | **+144 %** |

See [logs/grid/excess_volgroup_s8/](../logs/grid/excess_volgroup_s8/) for
the raw data and `compare_grids.py --grids excess excess_volgroup_s8`
once committed.

**Why this contradicts CLAUDE.md item 11:** that item was written before
the volatility group was ever tested in a grid (only "all groups at once"
had been tried in Phase 4B, which failed for *different* reasons —
overfitting on too many noisy features). Isolating the volatility group
revealed a real signal.

**Why it's "preliminary":**

- Single-seed CI is [−19.9, +18.2] — straddles 0. Per-seed reliability is
  unchanged from baseline; the signal lives in the ensemble.
- Only 4 folds in the per-fold distribution; 75 % beat-BH = 3 of 4 folds.
- One result needs replication — the V3.5 protocol always insists on it.

---

## Day-1 plan (do these in order)

### Step 1 — Replicate at 16 seeds (the verdict run)

```bash
python scripts/grid_eval.py --engine sb3 --policy-arch cnn --reward-mode excess \
  --seeds 16 --folds 5 --timesteps 30000 --num-envs 48 \
  --feature-groups volatility \
  --tag excess_volgroup_s16
```

**Wall-clock:** ~1.7 h on the GTX 1650 Ti machine; ~30 min on the new rig.

**Decision criteria:**

- If ensemble median ≥ +25 % AND vs-BH-winrate ≥ 60 % AND robustness ≥ +25
  → BB-volatility is a **confirmed** lever; proceed to Step 2.
- If ensemble median in [+10 %, +25 %] AND CI on single-seed median
  excludes 0 → softer confirmation; still worth Step 2 but flag uncertainty.
- If ensemble median < +10 % → the 8-seed run was a small-sample artefact;
  log it, revert CLAUDE.md item 11, no further work on this lever.

### Step 2 — Stack with the other best-known levers

The lever inventory now is: **BB-volatility (new), excess reward, H4
timeframe (CLAUDE.md item 2), cosine LR (item 7), SWA (item 7)**. All four
were found independently; whether they're additive isn't known.

Full stack run:

```bash
python scripts/grid_eval.py --engine gpu --policy-arch cnn --reward-mode excess \
  --seeds 32 --folds 5 --timesteps 1310720 --num-envs 2048 --n-steps 128 \
  --timeframe H4 --lr-schedule cosine --swa --feature-groups volatility \
  --tag excess_full_stack_h4_cosine_swa_volgroup
```

**Wall-clock:** ~3 h on the GTX 1650 Ti; ~45 min on the new rig.

**Decision criteria:**

- Compare to the existing 4-run H4+cosine+SWA aggregate (item 3 — median
  +12.1 % return, loses to BH +26 % by 13.9 pp on raw return).
- If stack ensemble median + BB-volatility exceeds BH on **raw return AND
  Sharpe AND drawdown** → first lever in the project to clear all three.
  This is the threshold that justifies revisiting CLAUDE.md item 3's
  "rigorous negative result" verdict.
- If stack does not clear all three → finding is M5-specific (BB features
  add timing signal that disappears at H4 sampling). Still a real result.

### Step 3 — Update CLAUDE.md based on outcomes

If Step 1 confirms and Step 2 clears BH on all three metrics:

- Item 11: rewrite — "the M5 ceiling held without group_volatility;
  with it, the H4+cosine+SWA+vol stack beats BH on all metrics".
- Item 3: rewrite — the "rigorous negative result" verdict is overturned;
  document the lever stack that did it.
- Item 6: nuance — observation-feature levers as a *group* still failed,
  but BB-volatility alone is the exception. Re-list as "every observation
  lever EXCEPT the BB-volatility group failed".

If Step 1 doesn't confirm — fold the failure back into CLAUDE.md item 6
("naively adding feature groups remains a failed lever, including the
isolated test of group_volatility at 16 seeds").

---

## Why this is worth doing carefully

The previous "no edge" verdict (item 11 from earlier in the same day)
was written based on **5 V4 levers in a single session**, with the explicit
assumption that retail M5 OHLCV doesn't carry per-bar timing signal. The
BB-volatility 8-seed run materially questions that. Reversing a verdict
this strong needs the replication; locking in a wrong verdict either
direction would mislead future sessions.

The same protocol that lets us trust the *negative* results (multi-seed,
walk-forward, distribution + CI) is what we owe the *positive* result
before claiming it.
