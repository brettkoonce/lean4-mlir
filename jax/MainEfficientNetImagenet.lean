import Jax

/-! EfficientNet-B0 on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.
    Same MBConv body as `MainEfficientNet.lean` (Imagenette) but with a
    1000-class head and the `.imagenet` (tfds streaming) dataset.

    bf16 incl. bf16 conv: as of the Codegen change that routes the MBConv
    expand/depthwise/project convs through `convdt`, `bf16Conv` now reaches
    the heavy convs in all MBConv blocks (the ~2x MBConv-block win from
    reference_bf16_depthwise_4060ti). The squeeze-excitation 1x1s are left in
    fp32 on purpose — they act on 1x1-spatial pooled tensors (no bf16 win)
    and the sigmoid gate is precision-sensitive. -/

def efficientNetB0Imagenet : NetSpec where
  name := "EfficientNet-B0 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                          -- 224→112
    .mbConv  32  16 1 3 1 1 true,                     -- 112
    .mbConv  16  24 6 3 2 2 true,                     -- 112→56
    .mbConv  24  40 6 5 2 2 true,                     -- 56→28
    .mbConv  40  80 6 3 2 3 true,                     -- 28→14
    .mbConv  80 112 6 5 1 3 true,                     -- 14
    .mbConv 112 192 6 5 2 4 true,                     -- 14→7
    .mbConv 192 320 6 3 1 1 true,                     -- 7
    .convBn 320 1280 1 1 .same,                       -- 1x1 head
    .globalAvgPool,
    .dense 1280 1000 .identity                        -- 1000-class head
  ]

/-- EfficientNet-B0 80-epoch recipe — the validation tier of the 80→300
    ladder (EPOCHS is baked from this spec; bump to 300 + re-emit for the
    real run). Mirrors the MobileNetV2 recipe that trained cleanly: SGD +
    momentum 0.9, base lr 0.1 at batch 256, 5-epoch warmup + cosine, weight
    decay 1e-5 (EfficientNet's small value — protects the depthwise/SE
    params), label smoothing 0.1, random-crop + flip, bf16 + bf16Conv.

    EfficientNet's original recipe is RMSProp + AutoAugment + stochastic
    depth + EMA. We now have EMA + stochastic depth (both on below); RMSProp
    and geometric AutoAugment aren't wired, so SGD+cosine at 80ep still lands
    a few points under the 77% paper number — fine for the validation tier,
    whose point is "does it train cleanly + the real per-epoch cost." If
    SE/swish make lr 0.1 unstable early, drop the peak to ~0.05. Mixup/cutmix
    left off as for MNv2 (flip the aug flags for the full recipe). -/
def efficientNetB0ImagenetConfig : TrainConfig where
  learningRate   := 0.1
  batchSize      := 256
  epochs         := 80
  useAdam        := false
  momentum       := 0.9
  weightDecay    := 1e-5
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  labelSmoothing := 0.1
  bf16           := true
  bf16Conv       := true    -- now reaches the MBConv expand/depthwise/project
  useEMA         := true     -- weight averaging (decay 0.9999) — eval + ckpt use it
  dropPath       := 0.2      -- stochastic depth, EfficientNet-B0 drop-connect rate

#eval efficientNetB0Imagenet.validate!

def main (args : List String) : IO Unit :=
  runJax efficientNetB0Imagenet efficientNetB0ImagenetConfig .imagenet
    (args.head? |>.getD "data/imagenet")
    "generated_efficientnet_b0_imagenet.py"
