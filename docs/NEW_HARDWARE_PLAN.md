# แผนการทดลองเมื่อรันบน hardware ใหม่

**Target hardware (planned):** i5-12600K (10C: 6P+4E) + 32 GB RAM + RTX 5070 (12 GB VRAM, Blackwell)
**Actual hardware in hand (2026-05-28):** Intel Core Ultra 7 265 (20 cores, Arrow Lake) + 32 GB DDR5-5600 + **RTX PRO 4000 Blackwell 24 GB GDDR7** (driver 582.16, CUDA 13.0). Workstation card — 2× VRAM headroom vs the planned 5070.
**Old machine bottleneck:** GTX 1650 Ti 4 GB + RAM จำกัด → GPU lanes ≤ 8k, SubprocVecEnv OOM, eval CPU-bound

> **Honest framing first:** Hardware นี้ทำให้**ทดลองได้เร็วขึ้นและเยอะขึ้น** ไม่ได้สร้าง alpha
> finding หลักของโปรเจกต์ (variance ครอบงำ signal, no reliable edge บน gold M5) ไม่ได้แก้
> ด้วย compute — แค่ทำให้**ตัดสินผลได้เร็วขึ้น** ไปต่อหรือพอ

---

## สิ่งที่เปลี่ยนเชิงความสามารถ

| มิติ | ปัจจุบัน | บนเครื่องใหม่ |
|---|---|---|
| GPU lane count | ~8k (OOM ที่ 16k) | **32k–65k+ ทำได้สบาย** |
| Network params | 50k (overfit threshold) | 500k+ ทดลองได้ (ยัง overfit เสี่ยง) |
| SubprocVecEnv workers | OOM > 4 workers | **8–16 workers** (RAM 32 GB) |
| Big-seed grid (16×5 excess) | ~4.7 ชม. | **~1–1.5 ชม.** |
| Hyperparameter sweep | ไม่ realistic | **ทำได้จริง** |
| ทดลองต่อสัปดาห์ | ~5–10 experiments | **30–50 experiments** |

---

## Tier 1 — ทำเป็นอย่างแรก (high ROI, low effort)

### 1.1 ~~เปิด SubprocVecEnv สำหรับ SB3 path~~ **OBSOLETE (2026-05-28)**

แผนเดิม: เพิ่ม `--use-subproc` ใน [scripts/grid_eval.py](../scripts/grid_eval.py) ให้สลับ DummyVecEnv → SubprocVecEnv

**ทำไม obsolete:** [validation/grid.py:140](../validation/grid.py#L140) ใช้ `VectorizedGoldTradingEnv` (single-process **numpy batched** VecEnv ของโปรเจกต์เอง) อยู่แล้ว — ไม่ได้ใช้ `DummyVecEnv`/`SubprocVecEnv` ของ SB3. Numpy batched env เร็วกว่า SubprocVecEnv ในเคสนี้เพราะ env ของเราเบามาก (Python overhead/IPC > workload จริง). ดู docstring ใน [env/vectorized_env.py:1-23](../env/vectorized_env.py#L1-L23).

ถ้าจะใช้ SubprocVecEnv จริง ๆ อยู่แล้วใน [validation/walk_forward.py:177](../validation/walk_forward.py#L177) — เปิดผ่าน `python scripts/train.py --subproc` แทน (ไม่เกี่ยวกับ grid_eval).

**สรุป:** ข้ามไป Tier 1.2.

### 1.2 GPU benchmark 16k–65k lanes ใหม่ (สะอาด ไม่มี contention)
```bash
for N in 16384 32768 65536; do
  python scripts/train_cleanrl.py --device cuda --num-envs $N --n-steps 64 \
    --timesteps $((N*64*3)) --log-every 0 --arch cnn
done
```
คาด steady-state throughput: 32k → ~150k steps/s, 65k → ~300k+ steps/s (Isaac-Gym-style)

### 1.3 Big-seed verdict 32 seeds × 5 folds (ที่ของเดิมทำไม่ไหว)
```bash
python scripts/grid_eval.py --engine gpu --policy-arch cnn --reward-mode excess \
  --seeds 32 --folds 5 --timesteps 1310720 --num-envs 2048 --n-steps 128 \
  --tag excess_bigseed_32  # ~2 ชม.บนเครื่องใหม่
```
**ทำไม 32 seeds:** CI จะแคบลง ~√2 เท่าจาก 16 seeds → distinguish signal จาก noise ได้ชัดกว่า

---

## Tier 2 — ทำตามมา (medium effort, high value)

### 2.1 ~~Vectorize eval~~ **ATTEMPTED & REJECTED (2026-05-28)** — slower

**Implementation:** [backtest/backtester.py](../backtest/backtester.py) `run_episode_torch_vec()`,
[env/torch_vec_env.py](../env/torch_vec_env.py) `auto_reset` / `track_history`
flags + trade-lifecycle tracking, [training/cleanrl_ppo.py](../training/cleanrl_ppo.py)
`ActorCritic.predict_torch`/`action_probs_torch`,
[policies/ensemble.py](../policies/ensemble.py) `EnsemblePolicy.predict_torch`.
Equivalence test ([scripts/_test_torch_eval_equiv.py](../scripts/_test_torch_eval_equiv.py))
shows **exact match** with scalar env at CPU fp64 / < 0.001 % drift at CUDA fp32
across `absolute` / `excess` / `dsr` reward modes.

**Bench result ([scripts/_bench_torch_vec_eval.py](../scripts/_bench_torch_vec_eval.py)):**

| Test split (70 847 bars) | scalar | torch_vec | speedup | drift |
|---|---|---|---|---|
| Single-seed eval | 27.8 s | **120.4 s** | **0.2×** (5× slower) | 0.0009 % |
| Ensemble K=8 eval | 195.5 s | 256.2 s | 0.8× | 0.0001 % |

**Why it failed:** `TorchVecGoldEnv` is built for **batched** lanes (num_envs ≫ 1).
With `num_envs=1` each step still pays the per-tensor and kernel-launch overhead
that the scalar `GoldTradingEnv` (NumPy float ops) skips entirely. The GPU is
underutilized in batched training (Tier 1.2 finding) **and** in single-lane eval —
the bottleneck has been Python overhead vs. the cost of the work being done,
not host↔device transfers as the plan assumed.

**Status:** Code path kept (correctness verified) but `validation/grid.py`
`_evaluate()` reverted to the scalar path. If a future ensemble-batching design
arrives (one batched forward over K stacked actor weights per step) this
plumbing is the foundation for it.

**Skip Tier 2.1 in future plans.** The realistic speedup levers for the
ensemble-eval bottleneck remain Tier 2.4 (torch.compile) and Tier 3.4 (CUDA
graphs).

### 2.2 ~~รัน grid หลายตัวขนาน~~ **REJECTED (2026-05-28)**

Bench: sequential 2 small grids = **756 s**, parallel 2 = **906 s** = **1.20× slower**.

GPU contention (sat at 89 % util mid-run) and CPU scalar-eval competition more
than negate the wall-clock parallelism. Combined with Tier 1.2's finding that
the GPU is already underused at one grid, splitting it across two does not
free up unused capacity — it just creates contention.

Don't propose parallel grids for this codebase. Stack experiments serially.

### 2.3 Hyperparameter sweep ที่ของเดิมทำไม่ไหว
```bash
# entropy coef sweep
for ENT in 0.001 0.005 0.01 0.05; do
  python scripts/grid_eval.py --engine gpu --ent-coef $ENT \
    --seeds 8 --folds 5 --reward-mode excess --tag sweep_ent_$ENT
done
```
**ต้องเพิ่ม CLI flag** `--ent-coef`, `--learning-rate`, `--n-epochs` ใน grid_eval.py

ทำ heatmap (seed-ensemble robustness) ดู basin ที่ดีที่สุด — ระวัง overfit-to-grid ใช้ held-out fold ทดสอบสุดท้าย

### 2.4 torch.compile บน ActorCritic — **PARTIAL WIN (2026-05-28)**

**Implementation:** [training/cleanrl_ppo.py](../training/cleanrl_ppo.py) `PPOConfig.compile`
+ [scripts/train_cleanrl.py](../scripts/train_cleanrl.py) `--compile`. Wraps
`ActorCritic` with `torch.compile(mode="reduce-overhead")` after init.

**Bench result ([scripts/_bench_compile.py](../scripts/_bench_compile.py)) —
16k lanes × 5 updates × 64 n_steps = 5.24M env-steps:**

| | env-steps/s | wall-clock |
|---|---|---|
| baseline (no compile) | 55,908 | 93.8 s |
| torch.compile (reduce-overhead) | 59,951 | 87.5 s |
| **Speedup** | **1.07×** | — |

Far below the 1.5–2× expectation:
1. **Triton not installed on Windows** → inductor falls back to the C++ codegen
   path; kernel quality is lower than Triton would give.
2. Per Tier 1.2 the GPU is already 32–45 % utilised — compile only cuts the
   Python/launch overhead, which is **not** the dominant cost (the underused
   GPU is). So compile cannot recover what isn't being spent.
3. The `ActorCritic` is small (~50 k params); compile graph-build cost
   (~5–10 s one-off) is a large fraction of any short training run.

**Implication for grid_eval:** each grid cell trains for ~30 s. A 5–10 s
compile-overhead per cell would **net-slow** the grid. Do not wire `compile`
into `validation/grid.py`.

**Status:** Flag kept (`PPOConfig.compile`, `--compile`) for one-off long
training runs where amortising graph build is worth the 7 %. Not the default,
not used by the grid.

---

## Tier 3 — น่าสนใจแต่ speculative

### 3.1 Multi-instrument generalization test
ถ้า framework ทำงานบน gold ได้ ลองยิงไปสินทรัพย์อื่นว่าเจอ edge ไหม:
- EURUSD M5 (FX, ลื่นไหลต่างจาก gold)
- BTC USD H1 (volatility สูง, regime ชัดกว่า)
- SPX index daily (low noise, momentum ชัด)

```bash
# ต้องเพิ่ม CSV และ data config
python scripts/grid_eval.py --data data/EURUSD_M5.csv --reward-mode excess \
  --seeds 16 --tag eurusd_excess
```

**Honest:** ถ้า gold M5 ไม่มี edge ที่จับได้ EURUSD M5 ก็แทบไม่มี (microstructure คล้ายกัน) แต่ daily/H1 ของสินทรัพย์อื่นอาจต่างจริง — เป็นการทดสอบ generality ของ framework

### 3.2 Network ใหญ่ขึ้น (carefully)
ปัจจุบัน ActorCritic 50k params ขนาดเกือบเล็กที่สุดที่เป็นไปได้
ลอง 200k–500k params กับ overfit control เข้มงวด:
- features_dim 128 → 256
- cnn_channels (32,64) → (64,128,128)
- net_arch [128,128] → [256,256]
- **ต้องมี early stopping บน validation fold**

**คาดผล:** น่าจะ overfit เพราะ SNR ของ gold M5 ต่ำ — แต่ทดสอบเร็วบนเครื่องใหม่ ตัดทิ้งได้ใน 1 วัน

### 3.3 ทดลอง algorithm อื่น
- **SAC (Soft Actor-Critic)** — off-policy, replay buffer, อาจ stable กว่า PPO
- **IMPALA** — GPU-native off-policy distributed, scale ดี
- **A3C** — เก่าแล้วแต่ผ่าน 12600k 10 cores ดี

ใน [config/config.yaml](../config/config.yaml) อยู่แล้วมี `algo: ppo|dqn` — ขยายเป็น sac/impala

### 3.4 CUDA graphs สำหรับ rollout loop
รวบ env step + policy forward เป็น graph เดียว replay → ตัด Python overhead ออกหมด
คาดหวัง 3–5x training speedup เพิ่มจาก torch.compile
**Effort:** สูง (rewrite rollout loop, edge case rese tบน done)

---

## Tier 4 — อย่าทำ (เสียเวลา)

- ❌ **Hybrid CNN-LSTM, Transformer ใหญ่กว่าเดิม** — โปรเจกต์พิสูจน์แล้วว่า overfit ([docs/EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) §15)
- ❌ **Volatility-targeted sizing** — ทดสอบแล้วแพ้ baseline
- ❌ **Regime features เพิ่มเข้าไป** — ทดสอบแล้วไม่ช่วย ([gpu-ppo-vs-sb3 memory](../../.claude/projects/d--DeepGold-RL/memory/gpu-ppo-vs-sb3.md))
- ❌ **DSR reward** — ตัดทิ้งแล้ว 0% beat BH
- ❌ **Indicator expansion ลึกกว่า trend/momentum/volatility** — มากไป overfit

---

## ลำดับแนะนำ (วันแรกบนเครื่องใหม่)

1. **(15 นาที)** Smoke test สเปก: GPU benchmark 16k/32k/65k lanes (Tier 1.2) — confirm GPU ใช้ได้เต็ม
2. **(30 นาที)** SubprocVecEnv เปิดให้ใช้ (Tier 1.1) + smoke test
3. **(2 ชม.)** **Big-seed verdict 32×5 excess** (Tier 1.3) — ตัวเลขจริงที่จะใช้ตัดสิน "ไปต่อหรือพอ"
4. **(1 ชม.)** ระหว่างนั้น เขียน vectorize eval (Tier 2.1) — pull bottleneck ออก
5. **(2 ชม.)** Hyperparameter sweep entropy/lr (Tier 2.3) — ดูว่ามี basin ดีกว่าไหม
6. **(วันที่ 2)** Multi-instrument test (Tier 3.1) บน 2 สินทรัพย์ — generalize ได้ไหม

---

## หลังจากนั้น: คำถามที่ตั้งต้นจะ answer ได้ภายในสัปดาห์

1. **excess reward ชนะ buy-hold อย่างมีนัยสำคัญทางสถิติไหม** (32–64 seeds, CI แคบจน decisive)
2. **มี hyperparameter combo ที่ดีกว่าค่า default อย่างมีนัยสำคัญไหม**
3. **edge generalize ไปสินทรัพย์อื่นได้ไหม หรือเฉพาะ gold M5**
4. **ถ้าทุกอย่างยังคงแพ้ baseline → trading RL บน retail data + retail compute ไม่ practical** (เป็น answer ที่ valid และสำคัญ)

---

## สิ่งที่ **ไม่เปลี่ยน** ไม่ว่าจะเครื่องอะไร

- Variance dominate signal (3-seed std = 46.6% บน excess grid)
- Walk-forward + per-fold normalizer (กัน leakage)
- Distribution-based eval (ไม่เคย claim single backtest)
- Scientific honesty: report fail honestly, ไม่ shopping result

> สเปกใหม่ = ตัด wait time ให้สั้นลง คำตอบมาเร็วขึ้น — ไม่ใช่ตัวสร้างคำตอบ
