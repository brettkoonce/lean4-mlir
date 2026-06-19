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

## ViT — what's still needed before a paper-faithful re-run

ViT-Tiny (DeiT-Ti) config is the closest to faithful, but has **2 gaps** + 1 deferred. The
completed ViT-80 (65.6%) ran without these.

1. **Geometric RandAugment — one flag (free).** Config has `useRandAugment := true` but the
   comment says "color subset only (no geometric — tfa N/A)." That's now STALE — geometric
   RandAug was wired this pull (`randAugmentGeometric`, via ImageProjectiveTransformV3). Add
   `randAugmentGeometric := true` to `vitTinyImagenetConfig`. (DeiT uses full RandAug.)

2. **Stochastic depth — NOT just a flag (~half-day codegen).** DeiT-Ti uses dropPath 0.1, but
   `dropPath` is wired ONLY into conv blocks (`mbconv_block`, `convnext_block`, invres) — the
   **`transformer_block` does not participate.** To add it (in jax/Jax/Codegen.lean):
   - `transformer_block(params, x, idx, n_heads)` (~line 713): add `drop_key`/`keep_prob` args;
     DeiT drops each *sublayer residual independently* — split into 2 sub-keys, apply inverted
     drop (`branch * bernoulli/keep_prob`) to the attention-branch AND the mlp-branch before
     each residual add (mirror convnext_block lines ~757-761).
   - forward SD-dispatch (~lines 1480-1590): add a `.transformerEncoder` branch mirroring
     `.convNextStage` — loop over nBlocks, pass `dpkeys[dbi]` + linear keep schedule + n_heads.
   - include transformer blocks in the `totalDrop` count that sizes the dpkeys split.
   - Risk low (additive, inference-safe via drop_key=None default; rng/scaling/schedule infra
     all proven on the convnets). ~half-day with testing.

3. **Repeated augmentation — DEFER (low ROI).** DeiT uses 3× repeated-aug; it's a data-pipeline
   change (~15-20 lines in build_imagenet_iter, flat_map each example to 3 aug copies, new
   `repeatedAug` config field). But: benefit is <0.5% for Ti and only at 300ep, and it adds CPU
   aug load (worsens the data-pipeline bottleneck). Skip unless doing a definitive DeiT-300 run.

So a paper-faithful ViT re-run = flag #1 (free) + codegen #2 (half-day). Skip distillation
(plain DeiT-Ti ~72% doesn't use it; only DeiT⚗ does) and #3.

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
- ViT config NOT yet edited (flag #1 + codegen #2 above pending your go-ahead).
