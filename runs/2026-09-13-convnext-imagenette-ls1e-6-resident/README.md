# ConvNeXt-T on Imagenette, verified XLA path — paper layer-scale init, resident parameters

The run Chapter 8's *Run it first* transcript and its Results row come from. Shipped code as of
`a14ad2af` (layer scale = its own init kind 3 → 1e-6, the paper's and the JAX reference's value),
the post-2026-08-30 render (549 outputs, 23 LayerNorms including the head LN), one RTX 4060 Ti,
`PJRT_FFI_RESIDENT=1`, stale checkpoint removed before launch.

    rm -f .lake/build/convnext_adam_ckpt_xla.bin*
    PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0 .lake/build/bin/convnext-verified-adam data \
        > convnext-imagenette-xla.log 2>&1

## Result

**82.27 % top-1 [81.04, 83.43] / 96.51 % top-5** at epoch 80, best 82.32 % @ 77. Wall clock
17:16:45 → 18:30:35 UTC = 4430 s = 55.4 s/epoch = 188 ms/step on the book's definition (epoch wall
clock ÷ 295 batches, validation pass included). Training loss on the 0.50 label-smoothing floor.

## Against the other ConvNeXt Imagenette runs

| run | layer-scale init | render | resident | top-1 | top-5 |
|---|---|---|---|---|---|
| this | 1e-6 (paper) | 549 out, 23 LN | yes | **82.27** | 96.51 |
| `runs/2026-09-13-convnext-imagenette-ls1e-6/` | 1e-6 | same | no | 81.78 | 95.75 |
| `runs/2026-08-30-convnext-imagenette-headln/` | 1.0 (old kind 1) | same | no | 85.45 | 97.35 |
| `runs/2026-08-12-convnext-imagenette-xla-cuda/` | 1.0 | 543 out, 22 LN | yes | 85.07 | 97.30 |
| `runs/2026-09-13-convnext-imagenette-jax-ref/` (JAX reference, same recipe) | 1e-6 | JAX | n/a | 78.06 | 94.73 |

Same render and recipe, γ = 1 against γ = 1e-6, at sampled epochs:

| epoch | 5 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| γ = 1 (Aug-30) | 66.52 | 72.79 | 80.87 | 82.11 | 84.18 | 84.79 | 85.04 | 85.71 | 85.45 |
| γ = 1e-6 (this) | 62.22 | 73.35 | 77.43 | 77.96 | 80.03 | 79.82 | 81.02 | 81.94 | 82.27 |

Read: the paper's init costs about three points on this recipe (80 epochs, 9,469 images), resolved at
±1.2 per run. Level through epoch 10, three to four points behind from epoch 20 on, and the gap never
closes — with γ at 1e-6 every residual branch's gradient is scaled by γ, so the branches train late.
The 300-epoch/1.28M-image recipe the init belongs to does not pay this. The book keeps the paper's
value; the chapter says what it costs.

The JAX reference at the same init and recipe lands 4.2 points *under* this run — see that directory's
README; the two paths differ in more than layer scale, and its crop augmentation is the first suspect.
