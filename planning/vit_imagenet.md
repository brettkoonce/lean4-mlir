# vit_imagenet.md — bf16 ViT-Tiny on ImageNet (Lean→JAX, 2-GPU)

Handoff/notes for the bf16 Vision Transformer ImageNet effort. Captures
what works, the exact run recipe, the open issues (esp. what's needed for
a *legit* accuracy run), and measured time estimates. Written 2026-05-30.

## TL;DR

- **bf16 ViT-Tiny ImageNet training works end-to-end on 2× 7900 XTX.**
  This is a *pipeline-validation* milestone, NOT a SOTA-accuracy run.
- The trainer is `jax/MainVitImagenet.lean` → `vit-tiny-imagenet` exe →
  generates `.lake/build/generated_vit_tiny_imagenet.py`.
- **2-GPU requires `LD_PRELOAD=/opt/rocm/lib/librccl.so.1`** (see below).
- Current recipe is a band-aid (peak LR 1e-4) and will land well short of
  the ~72% ViT-Tiny is capable of. The fixes needed are gradient clipping
  + the DeiT augmentation suite (both codegen work).

## Architecture / spec

ViT-Tiny = DeiT-Ti: patch16, embed 192, 12 transformer blocks, 3 heads,
MLP 768, 1000-class head. ~5.72M params. 224×224 input → 196 patches + CLS.

## How to run (the exact, working invocation)

```bash
# 1. build the exe (from the jax subproject)
cd /home/skoonce/lean/claude_max/lean4-jax/jax && lake build vit-tiny-imagenet

# 2. regenerate the .py (run exe briefly from PROJECT ROOT; it writes the
#    .py to .lake/build/ before spawning python — kill after a few sec)
cd /home/skoonce/lean/claude_max/lean4-jax
timeout 12 env HIP_VISIBLE_DEVICES=1 jax/.lake/build/bin/vit-tiny-imagenet >/dev/null 2>&1

# 3. run the generated .py DIRECTLY in tmux (NOT via the exe — runJax buffers
#    all child stdout until exit, so you get no live logs through it).
#    *** LD_PRELOAD is mandatory for 2-GPU (see RCCL note) ***
tmux new-window -d -n vit2gpu \
  "LD_PRELOAD=/opt/rocm/lib/librccl.so.1 PYTHONUNBUFFERED=1 \
   .venv/bin/python3 -u .lake/build/generated_vit_tiny_imagenet.py \
   > runs/$(date +%F)-vit-tiny-imagenet-bf16/train.log 2>&1; echo EXIT_CODE=\$? >> ...log"
```

Single GPU (no RCCL needed): prepend `HIP_VISIBLE_DEVICES=0`, drop LD_PRELOAD.

## ⚠️ The 2-GPU RCCL gotcha (solved)

Multi-GPU + tfds crashes at the first all-reduce:
`RCCL operation ncclGetUniqueId failed: Unable to load NCCL library`.

**Cause:** the generated trainer imports TensorFlow (for the tfds data
pipeline). TF loads its own bundled NCCL into the process and *shadows*
jaxlib's RCCL. `scripts/jax_multigpu_probe.py` never hits this (numpy data,
no TF) — that's why multi-GPU "verified clean" earlier.

**Fix:** `LD_PRELOAD=/opt/rocm/lib/librccl.so.1` — loads the correct ROCm
RCCL first. Must be set at launch (the dynamic linker reads it at exec; you
cannot set it from inside the .py after TF is imported). `LD_LIBRARY_PATH`
does NOT work (the lib was shadowed, not missing). See memory
`project_gpu_plan` for the full writeup.

**TODO (nice-to-have):** bake `LD_PRELOAD` into `runJax`'s `IO.Process.spawn`
env, or a small launch wrapper, so multi-GPU+tfds runs work without
remembering the flag.

## Current config (provisional — band-aid)

`vitTinyImagenetConfig`: AdamW, **peak LR 1e-4**, batch 512 (256/device),
80 epochs, wd 0.05, 5-ep warmup + cosine, label smoothing 0.1, augment, bf16.

**Why LR is crippled at 1e-4:** with NO gradient clipping, the model
collapses to chance the moment warmup ramps LR past ~1.6e-4 (train loss
pins at ln(1000)≈6.9, val top1 → ~0.4%). Observed at both 5e-4 and 2e-4
peaks. 1e-4 keeps the whole cosine schedule under that threshold → trains
stably but slowly. (imagenette ViT-Ti learned fine at 3e-4 — the 1000-class
softmax + bf16 gradient noise lowers the stability threshold.)

Collapse evidence (5e-4 run): ep1 val 0.87% → ep3 0.41% → flat at chance.
Stable-but-slow (1e-4 run): ep1→ep3 train loss 6.90→6.60, val 0.70%→1.32%.

## What's needed for a LEGIT accuracy run (priority order)

1. **Gradient clipping (grad-norm 1.0) in the codegen.** THE unlock. Lets
   you use the proper DeiT LR (~1e-3) without collapse → far faster/better
   learning. Highest leverage. Not currently a TrainConfig option; needs
   adding to the emitted optimizer step (clip global grad norm before the
   Adam update).
2. **DeiT augmentation suite** — RandAugment, Mixup (0.8), CutMix (1.0),
   RandomErasing (0.25), stochastic depth (~0.1). ViT-from-scratch on
   ImageNet *depends* on these; without them it underfits/plateaus low.
   Codegen + data-pipeline work.
3. **300 epochs** (the standard ViT-Ti schedule).

## Expected accuracy

| Setup | top-1 |
|---|---|
| DeiT-Ti standard (300 ep, full recipe, no distill) | **72.2%** |
| DeiT-Ti + distillation | 74.5% |
| **Our run** (80 ep, LR 1e-4, no grad clip, minimal aug) | ~40–55% (guess; modest) |

Treat the current run as "the pipeline trains"; expect a mediocre number
until items 1–2 land.

## Performance / time estimates (measured 2026-05-30)

bf16 vs fp32 (isolated ViT-Ti bench, `jax/scripts/jax_vit_bench.py`):
**2.67×** (ViT-S: 3.6×). bf16 is a transformer/matmul win only — convnets
see ~0.98× (MIOpen bf16 conv is slow; see `reference_bf16_gfx1100_conv_vs_gemm`).

Measured step times (bf16, batch 512):
- **1 GPU:** ~390 ms/step
- **2 GPU:** ~183 ms/step (256/device, both GPUs 100%) ≈ **2.1× scaling**

Per-epoch (2-GPU bf16) ≈ 2502 steps × ~183 ms ≈ **7.6 min/epoch**.

| Epochs | 2-GPU bf16 | 1-GPU bf16 | 1-GPU fp32 (proj) |
|---|---|---|---|
| 80  | ~10 hr | ~22 hr | ~36 hr |
| 150 | ~19 hr | ~41 hr | — |
| 300 | **~38 hr** | ~82 hr | — |

A proper DeiT-recipe 300-epoch run is ~38 hr on 2 GPUs — very feasible.
Note: at ViT-Tiny scale the run can become input-bound on tfds aug; adding
the heavy aug suite (item 2) increases CPU data cost, watch throughput.

## Relevant commits / files

- `40aafb2` — bf16 codegen path (TrainConfig.bf16, mm() helper, matmul
  casts; patchEmbed → reshape+matmul which also dodges a MIOpen im2col
  crash on gfx1100)
- `8dc3dd5` / `b4fc0ab` — `jax/MainVitImagenet.lean` trainer + LR→1e-4 +
  `jax/scripts/jax_vit_bench.py`
- Minor codegen wart: the startup banner hardcodes the LR as a string
  literal (line ~1033 of generated code) instead of printing the `LR` var
  — cosmetically misleading in logs. Fix when convenient.

## Related memory

- `project_gpu_plan` — RCCL/LD_PRELOAD fix, multi-GPU status
- `reference_bf16_gfx1100_conv_vs_gemm` — bf16 conv-vs-gemm, codegen support
- `reference_iree_reduction_distribute_gfx1100` — separate gfx1100 IREE gotcha
