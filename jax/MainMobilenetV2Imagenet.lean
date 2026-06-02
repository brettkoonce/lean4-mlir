import Jax

/-! MobileNetV2 on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.
    Same inverted-residual body as `MainMobilenetV2.lean` (Imagenette) but
    with a 1000-class head and the `.imagenet` (tfds streaming) dataset.

    bf16 incl. bf16 conv: as of the Codegen change that routes the
    inverted-residual expand/project 1x1s and the depthwise through
    `convdt`, `bf16Conv` now actually reaches the 17 inverted-residual
    blocks (previously only the stem + final 1x1 were cast). The block-level
    win is ~2x on the 4060 Ti — the 1x1 expand/project (GEMM-like) love
    bf16; the 3x3 depthwise is a wash but harmless. See
    reference_bf16_depthwise_4060ti. -/

def mobilenetV2Imagenet : NetSpec where
  name := "MobileNetV2 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                    -- 224→112
    .invertedResidual  32  16 1 1 1,            -- 112, t=1
    .invertedResidual  16  24 6 2 2,            -- 112→56, t=6
    .invertedResidual  24  32 6 2 3,            -- 56→28, t=6
    .invertedResidual  32  64 6 2 4,            -- 28→14, t=6
    .invertedResidual  64  96 6 1 3,            -- 14, t=6
    .invertedResidual  96 160 6 2 3,            -- 14→7, t=6
    .invertedResidual 160 320 6 1 1,            -- 7, t=6
    .convBn 320 1280 1 1 .same,                 -- 1x1 conv to 1280
    .globalAvgPool,
    .dense 1280 1000 .identity                  -- 1000-class head
  ]

/-- MobileNetV2 30-epoch *validation* recipe (bump `epochs` to 90 for the
    real run — EPOCHS/LR/BATCH are baked from this spec, so it's a one-line
    edit + re-emit, no env override).

    SGD + momentum 0.9, base lr 0.1 at batch 256 with a 5-epoch warmup +
    cosine decay — the proven R34 ImageNet pipeline, not MobileNet's original
    RMSProp. Two MobileNet-specific departures from the R34 recipe:
      * weight decay 4e-5 (not 1e-4): large wd hurts the tiny depthwise
        weights; 4e-5 is the standard MobileNet value.
      * no mixup/cutmix: not standard for MobileNetV2 and a net loss at
        short schedules.
    If lr 0.1 proves unstable at 30ep (BN depthwise nets can be touchy),
    drop the peak to ~0.05; the validation run is what tells us. -/
def mobilenetV2ImagenetConfig : TrainConfig where
  learningRate   := 0.1
  batchSize      := 256
  epochs         := 30      -- validation tier; 90 = the real near-paper run
  useAdam        := false
  momentum       := 0.9
  weightDecay    := 4e-5
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true    -- random-crop + horizontal flip
  labelSmoothing := 0.1
  bf16           := true
  bf16Conv       := true    -- now reaches the inverted-residual blocks

#eval mobilenetV2Imagenet.validate!

def main (args : List String) : IO Unit :=
  runJax mobilenetV2Imagenet mobilenetV2ImagenetConfig .imagenet
    (args.head? |>.getD "data/imagenet")
    "generated_mobilenet_v2_imagenet.py"
