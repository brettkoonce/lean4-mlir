import Jax

/-! ConvNeXt-Tiny on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.
    First JAX-path port of ConvNeXt (previously IREE-only). Same body as the
    IREE `MainConvNeXtTrain.lean`: patchify stem, compute ratio (3,3,9,3),
    channels (96,192,384,768), depthwise-7×7 + channel-LN + inverted-
    bottleneck + GELU + LayerScale blocks, dedicated 2×2 stride-2 downsamples.
    ~28.6M params at 224×224, 1000 classes.

    bf16 incl. bf16 conv: the depthwise-7×7 is 2.32× faster in bf16 and the
    1×1 expand/project are GEMM-like (big bf16 win) — see
    reference_bf16_depthwise_4060ti. Channel-LN / GELU stay fp32.

    DEVIATIONS from canonical ConvNeXt (both deliberate, to reuse the existing
    validated layer types):
      * stem is `convBn` (BN + a ReLU) rather than conv+LN — param-equivalent,
        mirrors the IREE spec; the extra stem ReLU is a minor variant.
      * stem is convBn (above) — the only remaining gap; EMA + stochastic
        depth are now wired (useEMA + dropPath below). -/

def convNeXtTinyImagenet : NetSpec where
  name := "ConvNeXt-T (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,                    -- patchify stem (4×4 stride 4) → 56×56
    .convNextStage 96 3 .ln .gelu,             -- stage 1: 3 blocks @ 96
    .convNextDownsample 96 192,                -- 56→28
    .convNextStage 192 3 .ln .gelu,            -- stage 2: 3 blocks @ 192
    .convNextDownsample 192 384,               -- 28→14
    .convNextStage 384 9 .ln .gelu,            -- stage 3: 9 blocks @ 384
    .convNextDownsample 384 768,               -- 14→7
    .convNextStage 768 3 .ln .gelu,            -- stage 4: 3 blocks @ 768
    .globalAvgPool,
    .dense 768 1000 .identity                  -- 1000-class head
  ]

/-- ConvNeXt-T 80-epoch recipe — first pass with the full faithful regularizer
    stack on (EMA + stochastic depth), still at the 80ep tier to validate those
    features train cleanly before the ~80-hour 300ep run (bump EPOCHS to 300 +
    re-emit for the real run). ConvNeXt needs AdamW, not SGD: decoupled weight
    decay 0.05, peak LR 4e-4 at batch 256 (≈ the 4e-3@4096 official LR linearly
    scaled), 5-epoch warmup + cosine, label smoothing 0.1, grad-clip 1.0 (cheap
    insurance — unlocked the ViT run). bf16 + bf16 conv. EMA (decay 0.9999) +
    stochastic depth (dropPath 0.1, the ConvNeXt-T paper value) now on. Geometric
    RandAugment (N=2, M=9, the ConvNeXt recipe value) is now on too — the full
    RandAugment(N,M) sampler over the color+geometric op set (shear/rotate via
    ImageProjectiveTransformV3), not the old color-only "lite". Mixup/cutmix
    still off (flip those flags when chasing the very full paper recipe).

    TODO(geo-ra): recipe CHANGE — adds geometric RandAugment vs the prior
    no-RandAugment 80ep run that hit 75.93%. The blueprint named this as the
    remaining gap to ConvNeXt-T's ~82% (with the 300ep schedule). Re-run +
    re-eval (eval_convnext_full50k.py, supervise script unchanged) for fresh
    numbers; RandAugment is CPU-side tf.data — watch input throughput. -/
def convNeXtTinyImagenetConfig : TrainConfig where
  learningRate   := 4e-4
  batchSize      := 256
  epochs         := 80
  useAdam        := true
  weightDecay    := 0.05
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  useRandAugment       := true   -- ConvNeXt recipe RandAugment...
  randAugmentGeometric := true   -- ...the full color+geometric sampler (N=2, M=9)
  labelSmoothing := 0.1
  gradClipNorm   := 1.0
  bf16           := true
  bf16Conv       := true
  useEMA         := true     -- weight averaging (decay 0.9999) — eval + ckpt use it
  dropPath       := 0.1      -- stochastic depth, ConvNeXt-T paper value

#eval convNeXtTinyImagenet.validate!

def main (args : List String) : IO Unit :=
  runJax convNeXtTinyImagenet convNeXtTinyImagenetConfig .imagenet
    (args.head? |>.getD "data/imagenet")
    "generated_convnext_tiny_imagenet.py"
