# ConvNeXt-T on Imagenette, verified XLA path — layer scale initialised to 1e-6

Re-run after the layer-scale init fix of 2026-09-13. The verified path initialised ConvNeXt's
per-channel layer scale to ONES (`toSpecs` kind 1, `mkParam` → 1.0) where the JAX reference
(`emitLayerScaleInit`) and the paper use 1e-6. Layer scale is now its own init kind, 3, mapped to
1e-6 in `mkParam`. Nothing else changed: same `convnext_adam_train_step.mlir` render (init is
host-side; 549 outputs, the post-2026-08-30 render with the head LN), same recipe (AdamW 1e-3,
cosine, 3-epoch warmup, 80 epochs, batch 32, wd on every parameter), one RTX 4060 Ti, stale
checkpoint removed before launch.

    CUDA_VISIBLE_DEVICES=0 .lake/build/bin/convnext-verified-adam data > convnext-imagenette-xla.log

## Result

| layer scale init | run | epoch 80 top-1 | top-5 | best epoch |
|---|---|---|---|---|
| 1e-6 (paper, = JAX reference) | this run | **81.78%** [80.55, 82.96] | 95.75% | 82.06% @ 65 |
| 1.0 (kind 1, the old verified init) | `runs/2026-08-30-convnext-imagenette-headln/` (same render) | 85.45% | 97.35% | — |
| 1.0, pre-head-LN render | `runs/2026-08-12-convnext-imagenette-xla-cuda/` (the number in ch 8) | 85.07% | 97.30% | 85.22% @ 69 |

The paper's init is **3.7 points worse** on this recipe, and the gap is resolved: the two 95%
intervals on 3,925 images (±1.2) do not overlap. Trajectory: 1e-6 leads early (epoch 20: 78.6 vs
75.7) and falls behind from epoch ~40 (epoch 50: 80.9 vs 83.1), plateauing near 82 while γ = 1
climbs to 85. Consistent with near-identity blocks — branch gradients scale with γ, so the
branches train late — on 9,469 images and 80 epochs, a regime the paper's 300-epoch recipe never
sees.

## What this does and does not settle

- The verified path now trains ConvNeXt from the same init as its reference. Every ConvNeXt
  accuracy recorded before 2026-09-13 is from γ = 1.
- It does NOT yet say 81.8 is "what 1e-6 gets on this recipe": no JAX reference has run ConvNeXt-T
  on Imagenette on the current pipeline. The only reference number, `runs/2026-05-27-convnext-tiny-ablations/`
  (bare `convnext-tiny-gelu`, 84.94%, 1e-6 init), is from the ROCm/IREE era, before the 2026-08-04
  init change and on the 9,984-image denominator, so it is not pairable. The control is one run:
  the JAX Imagenette ConvNeXt-T-GELU bare config on XLA, ~80 epochs on one card. If it lands near
  82 the init explains the gap; if it lands near 85 the verified path has a second difference.

## Wall clock is not comparable

2 h 15 m against the Aug-12 run's 1 h 19 m: this run was launched without `PJRT_FFI_RESIDENT=1`
(off by default in `ffi/pjrt_ffi.c`; the earlier logs show the RESIDENT lines), so 318 MB of
parameters crossed PCIe every step, and the CIFAR sweep shared the box's host for the first 22
minutes. Numerics are unaffected; do not update any timing column from this log.
