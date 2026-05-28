# ConvNeXt-T Imagenette ablations: mixup + random erasing

Two new data-augmentation rows for the Ch 9 ablation table, extending
the existing bare / cutmix / randaug triple. Both runs use the
`convnext-tiny-gelu` spec (27.8M params, ConvNeXt-T-GELU) and the
80-epoch base config (Adam lr=1e-3, cosine warmup=3, label smoothing
0.1, weight decay 1e-4). Augmentation knob is the only thing that
varies vs the existing `convnext-tiny-gelu` (bare) reference run.

## Setup

| Run | Config | GPU |
|---|---|---|
| mixup | `convNextTinyMixupConfig` (useMixup=true, mixupAlpha=0.8) | 7900 XTX #0 (gfx1100) |
| erase | `convNextTinyEraseConfig` (randomErasing=true, prob=0.25) | 7900 XTX #1 (gfx1100) |

Both runs kicked off in parallel via:
```
HIP_VISIBLE_DEVICES=0 IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  ./.lake/build/bin/ablation convnext-tiny-gelu-mixup > mixup.log 2>&1
HIP_VISIBLE_DEVICES=1 IREE_BACKEND=rocm IREE_CHIP=gfx1100 \
  ./.lake/build/bin/ablation convnext-tiny-gelu-erase > erase.log 2>&1
```

Wall-clock: ~13.4 hours (mixup, 604s/ep) and ~14.3 hours (erase,
645s/ep). Erase was slightly slower per epoch — small per-step
overhead from the random-erasing kernel.

## Results

| Cell | Val acc | Δ vs bare |
|---|---|---|
| `convnext-tiny-gelu-cutmix` (existing) | 87.81% | +2.87 |
| `convnext-tiny-gelu-erase` (new) | **85.63%** | +0.69 |
| `convnext-tiny-gelu-randaug` (M=9, existing) | 85.48% | +0.54 |
| `convnext-tiny-gelu` (bare, existing) | 84.94% | --- |
| `convnext-tiny-gelu-mixup` (new) | **83.45%** | **−1.49** |

## Observations

**CutMix is the load-bearing knob, by a wide margin.** Same conclusion
as the Ch 10 ViT-Tiny data-aug ablation (which found CutMix at +2.9
vs RandAugment +0.5 on a different architecture). The two new rows
sharpen this further:

- **Random Erasing lifts marginally** (+0.69), about the same tier as
  RandAugment (+0.54). Both are at the edge of seed noise.
- **Mixup actively hurts** (−1.49). The blended-label gradient signal
  is too aggressive at 9.5K-image scale; the model can't extract a
  clean target signal from a Beta(0.8, 0.8)-mixed pair when each
  half of the dataset only has ~475 images per class. (Imagenette
  is 10 classes; a typical ImageNet-scale Mixup ablation on the
  full 1.28M-image set lands +0.5 to +1.0, not −1.5.)

**Cross-architecture confirmation.** ConvNeXt-T (this run) and
ViT-Tiny (Ch 10) are architecturally different — one's a conv with
LayerNorm + GELU + LayerScale, the other's a transformer with
attention. The data-aug ranking (CutMix >> RandAug ≈ Erase > bare >
Mixup) being consistent across both is a real signal that the
ranking is data-regime-driven, not architecture-driven, at this
scale.
