# แผนการทดลองเมื่อรันบน hardware ใหม่

**Target hardware:** i5-12600K (10C: 6P+4E) + 32 GB RAM + RTX 5070 (12 GB VRAM, Blackwell)
**Current bottleneck:** GTX 1650 Ti 4 GB + RAM จำกัด → GPU lanes ≤ 8k, SubprocVecEnv OOM, eval CPU-bound

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

### 1.1 เปิด SubprocVecEnv สำหรับ SB3 path
SubprocVecEnv ปัจจุบันถูกปิดเพราะ 8 torch workers เคย OOM ที่ RAM เดิม กับ 32 GB จะรองได้สบาย

```bash
# ทดสอบ SubprocVecEnv 8 workers
python scripts/grid_eval.py --engine sb3 --num-envs 64 --use-subproc \
  --seeds 3 --folds 5 --timesteps 30000 --tag subproc_smoke
```

**ต้องเพิ่ม CLI flag `--use-subproc`** ใน [scripts/grid_eval.py](../scripts/grid_eval.py) ที่ส่งต่อไปยัง `_train()` ใน [validation/grid.py](../validation/grid.py) เพื่อสลับ DummyVecEnv → SubprocVecEnv

**คาดหวัง:** SB3 training เร็วขึ้น 3–4x (env stepping ขนานจริง 8+ cores)

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

### 2.1 Vectorize eval (กำจัด bottleneck ที่แท้จริง)
ปัจจุบันตอน eval/ensemble ใช้ scalar env + Python loop ทีละ bar = CPU sequential, GPU ช่วยไม่ได้
**ทางแก้:** เขียน `evaluate_torch_vec()` ที่ใช้ `TorchVecGoldEnv` num_envs=1 รัน full test fold บน GPU

จุดที่ต้องแก้:
- เพิ่ม method `run_episode_torch_vec()` ใน [backtest/backtester.py](../backtest/backtester.py)
- ใน [validation/grid.py](../validation/grid.py) `_evaluate()` สลับมาใช้ตัวใหม่เมื่อ engine="gpu"
- **ระวัง:** ผลต้องเทียบได้กับ scalar env (เลย equivalence test แล้ว ~1e-7 → ปลอดภัย)

**คาดหวัง:** Ensemble eval 28 นาที/fold → ~5 นาที/fold (5–6x) → big-seed run ตัดเหลือ ~45 นาที

### 2.2 รัน grid หลายตัวขนาน
RTX 5070 12GB + 10 cores → รัน 2–3 grid experiments พร้อมกันได้

```bash
# Terminal 1
python scripts/grid_eval.py --engine gpu --reward-mode excess ... --tag excess_v1 &
# Terminal 2 (CUDA_VISIBLE_DEVICES=0 + memory partition)
python scripts/grid_eval.py --engine gpu --reward-mode absolute ... --tag abs_v1 &
```

ใช้ `torch.cuda.set_per_process_memory_fraction(0.4)` แบ่ง VRAM

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

### 2.4 torch.compile บน ActorCritic
```python
# ใน train_cleanrl_ppo() หลัง init
ac = torch.compile(ac, mode="reduce-overhead")
```
คาดหวัง 1.5–2x training speedup โดยไม่ต้อง rewrite อะไร (cuda 12.x + torch 2.4+ stable แล้ว)

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
