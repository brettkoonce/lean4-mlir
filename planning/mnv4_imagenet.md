# mnv4_imagenet.md — MobileNetV4-Conv-M paper-faithful ImageNet trainer

**Status (2026-07-19):** SPIKE. Goal = get a *unified, building, emitting* 1000-class
MNv4-Conv-M ImageNet trainer with a faithful architecture + recipe. Full accuracy run
deferred. See the faithfulness-ledger convention in `planning/paper_faithfulness.md`.

Paper: Qin et al. 2024, *MobileNetV4 — Universal Models for the Mobile Ecosystem*
(arXiv 2404.10518). Reference impl for the exact arch/recipe: timm
`mobilenetv4_conv_medium` (`timm/models/mobilenetv3.py`, `_gen_mobilenet_v4`).

---

## TL;DR

The Lean→JAX codegen gives the whole trainer (autodiff backward, optimizer, tfds
ImageNet pipeline, checkpoint/supervisor/eval) for free to any net with a working
forward. `TrainConfig` already exposes ~95% of the paper recipe. Two real gaps:

1. **The pre-existing `mobilenetV4Medium` demo is Conv-S-sized, not Conv-M.** It logs
   **4.1M params**; real Conv-M is **~9.7M** (Conv-S is 3.8M). Its stage-2/3 block
   counts are truncated. → **rebuilt from the exact timm block list below.**
2. **UIB blocks are not wired into running-BN threading** in the codegen. Paper-faithful
   eval needs running-BN; the spike ships `runningBN:=false` (batch-stat eval, works
   today) and leaves the wiring as the one remaining codegen task (template =
   `mbconv_block`, `jax/Jax/Codegen.lean:684`).

Everything else — recipe, ImageNet main, supervisor, eval — is copy-from-MNv2 mechanical.

---

## Paper recipe — MobileNetV4-Conv-M (arXiv Table 9)

| Knob | Paper Conv-M | `TrainConfig` field | Notes |
|---|---|---|---|
| Epochs | 500 | `epochs` | schedule tier — dominant compute cost |
| Optimizer | AdamW β(0.9,0.999) ε1e-7 | `useAdam:=true` | AdamW/decoupled; ε likely hardcoded 1e-8 (negligible) |
| Peak LR / warmup | 0.004 @ bs4096, 5ep, cosine | `learningRate`/`warmupEpochs`/`cosineDecay` | LR targets EFFECTIVE batch |
| Weight decay | 0.1 | `weightDecay` | consider `wdExcludeNormBias:=true` (timm excludes norm/bias) |
| Batch size | 4096 | `batchSize`×`gradAccumSteps` | e.g. 512×8 on 4×16GB |
| Label smoothing | 0.1 | `labelSmoothing` | |
| Dropout (classifier) | 0.2 | `dropout` | |
| Stochastic depth | 0.075 | `dropPath` | ✅ wired into UIB (2026-07-19), linear ramp over 21 UIB blocks |
| RandAugment | 2 layers, mag 15, p0.7 | `useRandAugment`+`randAugmentGeometric`+`randAugmentN`/`M` | ✅ faithful — m>10 EXTRAPOLATES (`m/_AA_MAX·scale`), matches timm `level/_MAX_LEVEL`, no clamp |
| Mixup/Cutmix | **none** for Conv-M | — | leave off (simpler than ViT/ConvNeXt) |
| EMA | 0.9999 (Table 9 lists for Conv-S; timm uses it) | `useEMA`/`emaDecay` | |
| Resolution | 256 (also official r224 variant) | spec `imageH/W` | **use 224** — pipeline hardcodes `_IMG_SIZE=224`; `e500_r224_in1k` is a real published variant |
| Running-BN eval | (BN net) | `runningBN` | ✅ wired (UIB+fused, 2026-07-19); EMA-weights + EMA-BN eval |

Non-distilled paper top-1: **Conv-S 73.8 / Conv-M 79.9 / Conv-L 82.9**.
(Distillation numbers, e.g. 85.9 for Conv-L, need a teacher pipeline — out of scope.)

---

## Faithful Conv-M architecture (decoded from timm → repo `Layer` vocab)

timm encodes UIB as `uir_rN_aA_kK_sS_eE_cC`: repeat N, **a**=start(pre)-DW kernel,
**k**=mid(post)-DW kernel, stride S, expand E (on input channels), out C; `a0`/`k0` = no
DW. Repo `Layer`: `.uib ic oc expand stride preDWk postDWk` — so `(pre,post) = (a,k)`.
`er_…` = FusedIB → `.fusedMbConv`; `cn_…` = plain conv → `.convBn`. Stem 32, act ReLU.

```
stem:    .convBn 3 32 3 2 .same                      -- 224→112
stage0:  .fusedMbConv 32 48 4 3 2 1 false            -- 112→56  er_r1_k3_s2_e4_c48
stage1:  .uib  48  80 4 2 3 5                         -- 56→28   ExtraDW
         .uib  80  80 2 1 3 3                         --         ExtraDW
stage2:  .uib  80 160 6 2 3 5                         -- 28→14   ExtraDW
         .uib 160 160 4 1 3 3   (×2)                  --         ExtraDW
         .uib 160 160 4 1 3 5                         --         ExtraDW
         .uib 160 160 4 1 3 3                         --         ExtraDW
         .uib 160 160 4 1 3 0                         --         ConvNeXt
         .uib 160 160 2 1 0 0                         --         FFN
         .uib 160 160 4 1 3 0                         --         ConvNeXt
stage3:  .uib 160 256 6 2 5 5                         -- 14→7    ExtraDW
         .uib 256 256 4 1 5 5                         --         ExtraDW
         .uib 256 256 4 1 3 5   (×2)                  --         ExtraDW
         .uib 256 256 4 1 0 0                         --         FFN
         .uib 256 256 4 1 3 0                         --         ConvNeXt
         .uib 256 256 2 1 3 5                         --         ExtraDW
         .uib 256 256 4 1 5 5                         --         ExtraDW
         .uib 256 256 4 1 0 0   (×2)                  --         FFN
         .uib 256 256 2 1 5 0                         --         ConvNeXt
head:    .convBn 256 960 1 1 .same                    -- cn_r1_k1_s1_c960
         .convBn 960 1280 1 1 .same                   -- conv_head (num_features, head_norm)
         .globalAvgPool
         .dense 1280 1000 .identity
```

Total blocks: stem + 1 FusedIB + 2 + 8 + 11 UIB + 2 head-conv + GAP + FC. Target param
count **~9.7M** (timm `mobilenetv4_conv_medium` = 9.72M). If the emitter reports well
under that, a block row was dropped — re-audit against the timm list.

**Minor fidelity notes (not spike blockers):**
- `.convBn` in this codegen emits conv+BN without a trailing ReLU; timm's `cn`/conv_head
  are ConvBnAct (ReLU). Head is slightly less nonlinear than timm — cosmetic for "does
  it train", worth matching for the accuracy run.
- timm applies conv_head (960→1280) with head_norm before global pool; we do
  convBn→convBn→GAP→FC (equivalent shape).

---

## Recipe tiers (the "good shorter schedule to test with" question)

Heavy paper regularization (dropPath 0.075, RandAug m15 p0.7, wd 0.1, dropout 0.2) is
tuned for **500 epochs** — at a short schedule it *underfits*. So the short tier dials
regularization DOWN. Ladder, cheapest first:

- **Tier 0 — Imagenette 80ep (plumbing smoke).** `jax/MainMobilenetV4.lean` already does
  this (10-class). Confirms UIB arch + forward/backward end-to-end in minutes. Run this
  first after any arch edit.
- **Tier 1 — ImageNet ~30ep, reduced reg (quick signal).** Confirms the 1000-class tfds
  path trains and top-1 climbs. Expect ~55–65% (climbing), a few hours duty-cycled. Use
  RandAug m9, dropPath 0, dropout 0.1.
- **Tier 2 — ImageNet ~100ep, reduced reg (confidence tier / repo default).** LR 0.004
  scaled to batch, cosine, warmup 5, LS 0.1, wd 0.05, RandAug m9, dropPath 0.0–0.05,
  dropout 0.1, EMA. Expect low-to-mid 70s. ~1–1.5 days duty-cycled. Good go/no-go before
  committing to 500ep.
- **Tier 3 — full paper 500ep (`full` recipe).** bs4096 via grad-accum, RandAug m15 p0.7,
  dropPath 0.075, dropout 0.2, wd 0.1, LS 0.1, LR 0.004@4096, EMA 0.9999 → 79.9 target.
  ~week+ duty-cycled locally, or the RunPod A100 path (`project_amd_local_cloud_strategy`).

### Measured throughput + run-time estimate (2026-07-19, klawd RTX 4060 Ti)

Two direct measurements (224px, bf16, fwd+bwd+Adam), steady-state backed out of the
cumulative ms/step avgs across in-epoch windows:
- **1× 4060 Ti**, `bench` recipe (micro-batch 128 = one GPU's shard): **160 ms/step**
  (steps 100/200/300 → 160/160 exactly) → **~800 img/s per GPU.**
- **4× 4060 Ti** (0,2,3,4), real `probe` config (micro-512 → 128/GPU, eff-batch 4096):
  **~1,720 ms/step** (100→200 and 200→300 windows both ~1,720; 300→400 excluded — epoch-1
  val at step 312) → **~2,380 img/s** = **74% scaling** off the single-GPU number (PCIe
  all-reduce, 8×/step from grad-accum). Loss fell 6.89→6.27 over steps 100–400 (real learning).

| Config | img/s | min/epoch (train) | 30ep probe | 100ep default | 500ep full |
|---|---|---|---|---|---|
| 1× 4060 Ti | 800 (measured) | ~27 | ~13 h | ~44 h | ~9 days |
| **4× 4060 Ti** | **2,380 (measured)** | **~9 (≈10 wall w/ val)** | **~4.5 h** | **~15 h** | **~3 days compute / ~4–5 duty-cycled** |
| 1× A100 bf16 (est, ref R50 1,277 fp32) | ~4–6k | ~4 | ~2 h | ~7 h | ~1.5–2 days |

ImageNet-1k = 1,281,167 train img/epoch. **Notes:** (1) multi-GPU works — the earlier "NCCL
hang" was a slow cold-cache compile (see Run notes); `JAX_COMPILATION_CACHE_DIR` now makes
the ~15-min autotune a one-time cost. (2) Single-GPU can't run the real micro-512 (OOMs >16 GB);
the 1-GPU row uses the 128-shard config, and micro-512 shards fine to 128/GPU across 4 GPUs.
(3) 4060 Ti bf16 depthwise conv isn't tensor-core-accelerated, so A100 gains are conservative.
Local 4-GPU (~4–5 duty-cycled days) is now viable; renting 1× A100 (~2 days, no thermal
babysitting) remains the low-effort path.

---

## Work remaining after the spike

1. **Wire UIB (and `fusedMbConv`) into running-BN — ✅ DONE 2026-07-19.** `Codegen.lean`:
   (a) `uib_block`/`fused_mbconv_block` running variants with `(bn, bn_start, training)` →
   `(x, new_stats)`; (b) UIB + fusedMbConv added to the `_BN_CHANNELS` enumeration (~line 452);
   (c) `runningBN` branches in the forward dispatch. Also fixed a latent bug: `fused_mbconv_block`
   was only defined in the `!runningBN` branch, so any fused+runningBN spec would have crashed —
   now emitted as its own guarded running variant. `runningBN:=true` in the config. **Verified:**
   77/77 BN layers thread + their running stats update in train; eval uses EMA-weights +
   EMA-shadowed BN stats (`ema_bn`, the ENet lesson — already generic in codegen); checkpoints
   save `bn_state`+`ema_bn` so supervisor resume is bit-for-bit incl. BN. No regression
   (ENetV2 = runningBN-off, emits/compiles unchanged).
2. **Stochastic-depth into UIB — ✅ DONE 2026-07-19.** `dropPath` now threads a per-block
   keep-prob ramp (1.0→0.925 linear over 21 UIB blocks) + per-block RNG into both uib_block
   variants; drop applies to the residual branch (batch-wise, matching mbconv). UIB added to
   `totalDrop`. Verified: two SD train forwards differ (genuinely stochastic), eval is
   drop-free, loss finite. Off (byte-identical) when `dropPath=0` — so only the `full` tier
   exercises it.
3. **RandAugment mag-15 — ✅ faithful (no work needed).** The sampler does NOT clamp M: the
   op-arg functions scale by `m/_AA_MAX` (=`m/10`) and extrapolate for m>10 (m15 → 1.5×,
   e.g. 45° rotate), exactly timm's `level/_MAX_LEVEL`. (Earlier "clamps to 0–10" note was
   wrong — the clamp only exists on the `mstd>0` jitter path, which the recipe doesn't use.)
4. **256px path (optional)** — make `_IMG_SIZE` derive from `spec.imageH` (or use
   `trainRes:=256`) for the `e500_r256` variant. 224 is faithful to `e500_r224` so not
   required.
5. **`.convBn` ReLU in head** — match timm ConvBnAct for the accuracy run.
6. New glue when going to a real run: `scripts/supervise_mnv4_*.sh` (copy MNv2) +
   `scripts/eval_mnv4_full50k.py` (copy).

## Run notes / observations (2026-07-19 spike)

- **Multi-GPU "NCCL hang" — MISDIAGNOSED; there is no hang (resolved 2026-07-19).** The
  `ncclCommRegister … Cuda failure 500 'named symbol not found'` message is a **benign
  warning**: it is NCCL user-buffer registration (a perf optimization for CollNet/NVLS) that
  is unsupported on consumer PCIe 4060 Ti's, and NCCL falls back to the non-registered path
  cleanly. Proof: a minimal batch-sharded all-reduce completes in **0.37 s on 2 GPUs and
  0.57 s on 4 GPUs** (0,2,3,4) *despite* the warning. What looked like a hang was the
  **~15-min cold-cache XLA compile** — compilation is CPU-bound, so the GPUs read 0% util
  during it (they hit 100% only during autotune kernel-timing bursts). I killed the first run
  at ~15 min, right in the compile.
  **Mitigations:** (1) set `JAX_COMPILATION_CACHE_DIR` so the ~15-min autotune is paid once,
  not every launch; (2) `XLA_FLAGS=--xla_gpu_enable_nccl_user_buffers=false` silences the
  registration warning; (3) run with `python -u` to see live progress. No NCCL env / topology
  fix needed — multi-GPU works.
- **Single-GPU micro-batch 512 OOMs** on a 16 GB 4060 Ti (needs >16 GB). In the real 4-GPU
  config the 512 micro-batch shards to 128/GPU, which fits — so per-GPU throughput was
  measured with the `bench` recipe (micro-batch 128 = one GPU's shard).
- **Generated script buffers stdout** to a file (no flush). Run with `python -u` /
  `PYTHONUNBUFFERED=1` to see live per-step timing, else prints only flush on exit/crash.
- **Cold-cache XLA compile is slow** (~15 min) — UIB's many distinct conv shapes
  (depthwise k3/k5, 1×1 expand/project at many widths) give the autotuner a lot to time.

## Files

- `jax/scripts/supervise_mnv4_convm_100ep_4gpu_duty.sh` — Tier-2 100ep supervisor (rest @33/66).
- `jax/scripts/supervise_mnv4_convm_500ep_4gpu_duty.sh` — Tier-3 paper 500ep supervisor (rest every 30ep, ~3.5 days).
- `jax/MainMobilenetV4Imagenet.lean` — NEW: faithful Conv-M 1000-class spec + recipes + `.imagenet` main.
- `jax/MainMobilenetV4.lean` — pre-existing Imagenette (Conv-S-sized) demo; keep as Tier-0 smoke.
- `apps/baselines/MainMobilenetV4Train.lean` — pre-existing Imagenette train variant.
