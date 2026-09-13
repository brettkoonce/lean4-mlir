# ConvNeXt-T on Imagenette — JAX reference on the shared Imagenette recipe (layer-scale control)

New driver `jax/MainConvNeXt.lean` (`lake build convnext-tiny` in `jax/`): the ImageNet ConvNeXt-T body
of `MainConvNeXtImagenet.lean` with the head LN, 10 classes, and only the shared Imagenette recipe —
AdamW 1e-3, cosine, 3-epoch warmup, 80 epochs, batch 32, weight decay 1e-4 on every parameter, label
smoothing 0.1, crop + hflip, fp32. No RandAugment / Mixup / CutMix / Random Erasing / EMA / drop-path /
grad clip; `cnxInit` off so the convs take the generic fan-out init the verified path was aligned to
on 2026-08-04; layer scale at the emitter's 1e-6 (`emitLayerScaleInit`). Params 27,827,818 — the
verified count exactly. One RTX 4060 Ti (GPU 1), run from the repo root so `.venv/bin/python3` and
`data/imagenette` resolve:

    CUDA_VISIBLE_DEVICES=1 jax/.lake/build/bin/convnext-tiny data/imagenette > convnext-imagenette-jax.log 2>&1

(First launch 17:15 UTC was killed during compile and relaunched 17:18 with `PYTHONUNBUFFERED=1`;
that changed nothing — `Runner.lean` reads python's stdout with `readToEnd`, so the log fills at exit.)

## Result

**78.06 % top-1 [76.74, 79.32] / 94.73 % top-5** at epoch 80. 4959 s = 62 s/epoch.

| epoch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| top-1 | 71.08 | 75.39 | 76.59 | 74.70 | 76.10 | 78.29 | 77.99 | 78.06 |

⚠ The `Loss:` column in the log is the **validation** loss (`test_loss` in the generated script), not the
training loss the verified logs print — do not compare the two.

## What it says

The question was whether a reference at the paper's layer-scale init lands where the verified path does
(~82, then the 3-point drop from γ = 1 is the recipe's response to the init) or where γ = 1 does (~85,
then the verified path has a second difference). It landed at neither: **4.2 points under the verified
path** (`runs/2026-09-13-convnext-imagenette-ls1e-6-resident/`, 82.27), intervals apart. So the two
paths differ in more than layer scale, in the verified path's favour. First suspect, read off the
generated script: augmentation. `augment_batch` pads the 224×224 centre crop to 252 and takes a random
224 window (zero-padded translation jitter), where the verified trainer takes a random 224 window of the
stored 256×256 image. Everything else visible in the script matches the recipe (AdamW, wd 1e-4, LS 0.1,
cosine over 80 with 3-epoch warmup, seed 314159).

Not in the book: the Imagenette sections carry no reference numbers, and this one would need its
augmentation reconciled first. Proposed follow-up, not launched: the reference at γ = 1 (one line in
`emitLayerScaleInit`, ~83 min on one card) — if it gains ~3 points too, the init effect is the recipe's.
