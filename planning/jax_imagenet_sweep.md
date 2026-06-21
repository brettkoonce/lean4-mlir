# jax_imagenet_sweep.md — re-running the convnets with paper-faithful recipes

Hand-off for a fresh session. The original sweep (R34/MNv2/ENet/ViT/ConvNeXt) ran
with SGD + crop/flip on the convnets, **before** RMSProp + AutoAugment + the full
aug pack were wired. Those features now exist (collaborator, pulled 2026-06-13), and
the three convnet configs have been corrected to paper-faithful recipes. **The prior
completed results are therefore STALE for MNv2 / ENet / ConvNeXt — re-run + re-eval,
then bump blueprint §7.2 / §8.2 / §9.3 + planning/imagenet_sweep.md.**

Status when this doc was written: recipe edits are made but **UNCOMMITTED** (held in
the working tree by request). Box = ares, 6× RTX 4060 Ti, PCIe Gen3 (BIOS-capped —
see reference_ares_pcie_aer). **Re-runs gated on cooling/fan setup** (the box runs
hot; the 6-GPU runs are ~10–16 hr each).

## UPDATE 2026-06-19 — what actually happened + paper-length TODO

The short (80–90ep) validation runs are largely done. Outcomes:

- **MNv2 — DONE.** 90ep RMSProp (ρ0.9/μ0.9/**ε1.0**, lr 0.045, crop/flip, AA off) →
  **68.77% top-1 / 88.53% top-5** (full-50k EMA). Only +0.44 over the old SGD 68.33%
  — the optimizer swap barely moved it; the gap is schedule. blueprint **§7.3** updated
  (config block, 6-GPU budget, result table, narrative, per-epoch curve, paper-length
  TODO). One AER mid-run, watchdog auto-resumed. Final ckpt `~/mnv2_imagenet_bf16.bin`.
- **ENet — PARKED (LR-fragile).** RMSProp **ε1e-3** is too hot: **lr 0.045 fully diverged**
  (val→random by ep6); **lr 0.016 eroded** (val 31%@e4 → 19% and stalled). Trains fine up
  to lr≈0.0128 → wants peak **~0.01**. AA also under-trains at 80ep. Diverged ckpts archived
  (`~/archive_enet_diverged_lr045_jun14`, `~/archive_enet_eroded_lr016_jun14`). Config edit
  to lr 0.016 is UNCOMMITTED and **known-wrong** — set to ~0.01 before the real run. See
  memory `project_enet_lr_instability`.
- **ConvNeXt — CONFIG VALIDATED.** AdamW lr 4e-4 + grad-clip 1.0 + full aug
  (Mixup/CutMix/RandErase) is stable — val climbed cleanly 0.3→2.8% (e1–3) → 39.9% (e9 eval)
  through the warmup peak, no erosion (the opposite of ENet). Power outage 2026-06-19 killed
  it at e9 (`/tmp` logs lost; ckpts e1–9 survived). 80ep is a dry-run anyway.

### TODO: paper-faithful long runs (the gap-closers)

The short runs validated the recipes/pipeline; the gap to paper numbers is **schedule
length**, not the formalization (MNv2 proved optimizer/precision/pipeline are faithful).
Each net at its full schedule on the 6-GPU box is ~**35 hr apiece**:

| net | short-run | paper | gap | full-schedule recipe |
|-----|-----------|-------|-----|----------------------|
| MNv2 | 68.77% (90ep) | ~72.0% | ~3.2pt | RMSProp + exp-decay LR, ~300–480ep |
| ENet-B0 | (no stable run) | ~77.1% | — | **lr~0.01** + AA + stoch-depth 0.2, ~350ep |
| ConvNeXt-T | 75.93% (old 80ep RandAug) | ~82.1% | ~6pt | AdamW + full aug, 300ep (where Mixup/CutMix pay off) |

Priority per user (2026-06-19): validate configs first (done-ish), long runs **later**;
ENet's 300ep+AA+lr0.01 is wanted eventually but not now. Gate long runs on stable wall
power (the 2026-06-19 outage killed the ConvNeXt smoke test) + a checkpoint/resume sanity
pass at the longer horizon.

### 80ep vs 300ep — accuracy & wall-clock estimates (R34→ViT)

6× 4060 Ti (ares, CUDA, bf16) **except ViT** = 2× 7900 XTX (mars, ROCm). **Bold top-1 =
measured this project; bold hours = run actually happened.** Everything else is an estimate.

| net | 80ep top-1 | 80ep hrs | 300ep top-1 | 300ep hrs | paper |
|-----|-----------|----------|-------------|-----------|-------|
| ResNet-34 | **72.1%** | ~12 | ~74–75% | ~45 | 73.3% |
| ResNet-50 | ~76% | ~16 | ~79–79.5% | ~60 | 79.8% (RSB-A2) |
| MobileNetV2 | **68.8%** | **~10** | ~72% | ~38 | 72.0% |
| EfficientNet-B0 | ~70–73% | **~10.5** | ~77% | ~47* | 77.1% |
| ConvNeXt-T | **75.9%** | **~15.5** | ~82% | ~63 | 82.1% |
| ViT-Ti (DeiT) | **65.6%** | ~11 | ~72% | ~47† | 72.2% |

Per-epoch (min/ep): MNv2 7.5, ENet 8, ConvNeXt 12.6 (measured); R34 ~9, R50 ~12, ViT **9.1**
(measured 2026-06-21, faithful recipe, mars 2-GPU). *ENet's paper-faithful schedule is 350ep
(~47 hr), not 300; the rest are 300ep. †ViT 300ep hrs is a measured-step-time projection
(~218 ms/step on 2× 7900 XTX with the full faithful recipe — EMA + stochastic depth + geometric
RA push it ~18% above the old ~38 hr estimate); the run itself hasn't happened yet.

Totals if the full long sweep ran: ~253 hr for the five convnets+R50 at 300ep (~10.5 days of
ares time) + ~47 hr ViT on mars (measured-step-time projection) ≈ **~12.5 days** end-to-end. Best value: ConvNeXt 300ep (~63 hr,
+6pt → ~82%); cheapest interesting win: R50 300ep (~60 hr → the ~80% milestone — see
`planning/resnet50_imagenet.md`, near-zero new code). ViT is its own codegen project
(`planning/vit_imagenet.md`, transformer stochastic depth).

Use each net's **paper-faithful epoch count** for the long runs (300 everywhere except
ENet's 350); R50 at 300 = a fair RSB-A2 comparison. The **R34+R50 pair at a matched 300ep
modern recipe** is the proposed first concrete step (basic-block vs bottleneck, ~105 hr /
~4.4 days back-to-back, R34 is just an epochs bump on the existing trainer) — see
`planning/resnet50_imagenet.md`.

## The 3 runs to do (non-ViT)

All 6-GPU (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`), batch 256 → 252 (6×42), bf16+bf16Conv,
EMA + stochastic depth on. Run order = lightest first (MNv2 → ENet → ConvNeXt).

### 1. MobileNetV2 — 90 epochs, paper-faithful (RMSProp, crop/flip only)
- Config `mobilenetV2ImagenetConfig` already set: **RMSProp** ρ0.9 / μ0.9 / **ε1.0** /
  lr 0.045, wd 4e-5, cosine + 5ep warmup, label smoothing 0.1, **crop/flip only
  (useAutoAugment=false — MNv2 paper used NO AutoAugment)**, EMA, dropPath.
- The change vs the old 68.3% run: optimizer SGD→RMSProp (+ AA was wrongly on, now off).
- ⚠ **Needs a 6-GPU supervisor** — only `supervise_mnv2_90ep.sh` (4-GPU) exists. Make
  `supervise_mnv2_90ep_6gpu.sh` by copying the 4-GPU one and editing DEVS="0,1,2,3,4,5"
  + SPE=5083 (1281167//252), like the enet/convnext 6-GPU scripts. (env-var fix already
  in those — `env CUDA_VISIBLE_DEVICES=...`.)
- ETA ~9 hr @ 6-GPU (~8 min/epoch est, but RMSProp+no-AA may shift it — measure first 400 steps).
- LR note: 0.045 is the paper value at the paper's batch; if it collapses early, drop peak to ~0.02.

### 2. EfficientNet-B0 — 80 epochs, paper-faithful (RMSProp + AutoAugment)
- Config `efficientNetB0ImagenetConfig` already correct: **RMSProp** ρ0.9 / μ0.9 /
  **ε1e-3** / lr 0.045, wd 1e-5, **useAutoAugment=true (correct — ENet paper DOES use AA)**,
  cosine + 5ep warmup, EMA, dropPath 0.2.
- Supervisor `supervise_enet_b0_80ep_6gpu.sh` exists.
- Old run was 72.3% (SGD + crop/flip, 4-GPU); this is RMSProp + AA, so a genuinely new number.
- ETA ~10.5 hr @ 6-GPU (the old 6-GPU run was ~8 min/epoch; AA adds CPU aug load — measure).
- LR fallback ~0.016 if unstable (per config TODO).

### 3. ConvNeXt-T — 80 epochs, paper-faithful (now FULL aug pack)
- Config `convNeXtTinyImagenetConfig` just corrected: AdamW lr 4e-4, wd 0.05, grad-clip 1.0,
  RandAug (color+geo), **+ Mixup α0.8 + CutMix α1.0 + Random Erasing p0.25** (these 3 were
  missing — ConvNeXt was under-augmented; now matches the DeiT-style pack), EMA, dropPath 0.1.
- Supervisor `supervise_convnext_t_80ep_6gpu.sh` exists.
- Old run was 75.93% (RandAug only); the full pack should help — though mixup/cutmix mainly
  pay off at long schedules, so the 80ep is a **dry-run for the eventual 300ep** (where it
  matters). Watch for: heavy aug can be net-neutral/slightly-negative at 80ep — don't be
  alarmed if it's flat vs 75.9%; the point is validating the full-knob recipe trains cleanly.
- ETA ~16 hr @ 6-GPU.

## Run pattern (proven, per net)

```
cd jax
lake build <exe>                 # mobilenet-v2-imagenet / efficientnet-b0-imagenet / convnext-tiny-imagenet
rm -f .lake/build/generated_<net>.py
timeout 8 ./.lake/build/bin/<exe> data/imagenet >/dev/null 2>&1   # emit .py (errors on wrong python, harmless)
# verify: grep -nE "^DT = |EPOCHS = |optimizer|_mixup|_cutmix|_random_erase|autoaugment" the .py
nohup bash scripts/supervise_<net>_6gpu.sh >/tmp/<net>_driver.log 2>&1 &
# monitor hourly: master log /tmp/<net>_master.log, train log /tmp/<net>.log, ckpts /home/skoonce/<net>_*.bin
```
- AER-watchdog + per-epoch-checkpoint auto-resume is in the supervisors. Gen3 makes 6-GPU AER
  rare-not-zero (ENet had 2 resumes, ConvNeXt 0). Watchdog kills before host reset; resumes from latest ckpt.
- On completion: `CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python -u scripts/eval_<net>_full50k.py`
  for the canonical full-50k number (uses EMA weights). Then build RESULTS.md + pgfplots curve
  (copy an existing runs/*/ as template), fill blueprint §, update planning/imagenet_sweep.md.
- **Thermal**: one run at a time, watch temps (~50°C was fine pre-reshuffle; confirm after cooling work).

## ViT — paper-faithful re-run: READY (config + codegen landed 2026-06-21)

ViT-Tiny (DeiT-Ti) is now paper-faithful (no distillation). The 2 gaps below are CLOSED, the
config is flipped, it's smoke-tested on ROCm and perf is measured. **Staged on main, not yet
committed/launched.** The completed ViT-80 (65.6%) ran without these.

1. **Geometric RandAugment — DONE.** `randAugmentGeometric := true` in `vitTinyImagenetConfig`
   (full color+geometric RandAug via ImageProjectiveTransformV3; DeiT uses full RA).

2. **Stochastic depth — DONE (wired into `transformer_block`).** dropPath 0.1, linear keep ramp
   1.000000→0.900000 across the 12 blocks (block 0 never drops, block 11 keeps 0.9). In
   jax/Jax/Codegen.lean:
   - new `_drop_branch` helper + `transformer_block(..., drop_key=None, keep_prob=1.0)`:
     per-sample inverted DropPath ((B,1,1) mask, timm/DeiT semantics), TWO independent sub-keys
     so the attention and MLP residual branches drop separately.
   - `.transformerEncoder` arm added to the forward SD-dispatch — passes `dpkeys[dbi]` + the
     per-block keep schedule; transformer blocks now counted in `totalDrop`.
   - Inference-safe (drop_key=None → identity); convnet SD rng/scaling/schedule infra reused.
   Same config also flipped `useEMA := true` (decay 0.99996) and `epochs := 300` (was 80).

3. **Repeated augmentation — STILL DEFERRED (low ROI).** DeiT uses 3× repeated-aug; data-pipeline
   change (~15-20 lines in build_imagenet_iter, flat_map each example to 3 aug copies, new
   `repeatedAug` config field). Benefit <0.5% for Ti and only at 300ep, and it adds CPU aug load.
   Skip unless doing a definitive DeiT-300 run.

**GPU smoke test (ROCm, gfx1100):** `jax/scripts/smoke_vit_droppath_gpu.py` — imports the
generated trainer (training loop behind `__main__`), runs forward+train_step on a synthetic
batch: eval determinism, drop-path activity (same-key identical / diff-key max|Δ|=1.36 / train≠eval),
6-step loss descent. PASS on RocmDevice.

**Measured perf (2026-06-21, 2× 7900 XTX / mars, batch 512, full faithful recipe):** ~218 ms/step
steady (216/218/216/221 over 4 intervals), both GPUs 100% (compute-bound), ~66s one-time compile.
→ ~9.1 min/epoch → **~45 hr train / ~47 hr with val for 300 epochs.** ~18% above the old ~38 hr
estimate — the faithful recipe adds per-step EMA (full 5.7M-param tree), 24 drop-masks/step, and
heavier geometric RA. Cheapest time lever if needed: drop EMA. CUDA-box cross-check pending.

So the paper-faithful ViT re-run is staged and ready. Skip distillation (plain DeiT-Ti ~72%
doesn't use it; only DeiT⚗ → 74.5%) and #3.

## Environment / version pinning

- **No lockfile existed** — the JAX stack lived only in the gitignored `.venv`. Captured this
  session: **`jax/requirements-cuda-lock.txt`** (`pip freeze`, 76 pkgs) — UNCOMMITTED, held.
- Pinned stack (CUDA box): **jax/jaxlib 0.9.2**, jax-cuda12 plugin 0.9.2, **cuDNN 9.20.0.48**,
  CUDA 12.9, **tensorflow 2.21.0**, tensorflow-datasets 4.9.10, numpy 2.4.4.
- The tf2.21 constraint is why geometric RandAug needed the ImageProjectiveTransformV3
  workaround (tfa unavailable on tf2.21). Recipe behavior is tied to this exact stack — a
  silent jax/cuDNN bump could shift bf16 conv kernel selection and move the published numbers.
- **CUDA lock is wrong for the ROCm box** (mars / 7900 XTX uses a different jaxlib build) — make
  a parallel `requirements-rocm-lock.txt` from mars when there. Especially relevant given the
  planned hardware reshuffle (new venvs on moved boxes).
- TODO: add a one-liner to jax/README.md — "recreate CUDA env: pip install -r requirements-cuda-lock.txt".

## Uncommitted state at hand-off (held by request — review/commit when ready)
- `jax/MainMobilenetV2Imagenet.lean` — AA off (paper-faithful), docstring updated.
- `jax/MainConvNeXtImagenet.lean` — Mixup + CutMix + Random Erasing turned on (full pack).
- `jax/requirements-cuda-lock.txt` — new, the env pin.
- `jax/Jax/Codegen.lean` — transformer stochastic depth wired (`_drop_branch` + `transformer_block`
  drop args + `.transformerEncoder` SD-dispatch + `totalDrop` count). STAGED on main 2026-06-21.
- `jax/MainVitImagenet.lean` — paper-faithful flip: dropPath 0.1, randAugmentGeometric, EMA, 300ep.
  STAGED on main 2026-06-21.
- `jax/scripts/smoke_vit_droppath_gpu.py` — new ROCm GPU smoke test for the drop-path wiring. STAGED.
