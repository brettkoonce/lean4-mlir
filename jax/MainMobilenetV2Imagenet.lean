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

    RMSProp + momentum 0.9, base lr 0.045 at batch 256 with a 5-epoch warmup +
    paper exp-LR-decay (×0.98 per epoch) — MobileNetV2's *original* optimizer (was SGD+momentum lr 0.1,
    the borrowed R34 pipeline). Two MobileNet-specific departures kept:
      * weight decay 4e-5 (not 1e-4): large wd hurts the tiny depthwise
        weights; 4e-5 is the standard MobileNet value.
      * no mixup/cutmix: not standard for MobileNetV2 and a net loss at
        short schedules.
    RMSProp knobs: ρ=0.9 (rmspropDecay), μ=0.9 (momentum), ε=1.0 (rmspropEps —
    MobileNetV2's value, NOT 1e-8). LR schedule is the paper's exponential decay
    (×0.98 per epoch after warmup; cosineDecay off).

    TODO(recipe): this is a CHANGE on one axis — optimizer SGD→RMSProp (the
    paper's optimizer). Aug stays paper-faithful: crop/flip only, NOT
    AutoAugment (MobileNetV2 used no AA; useAutoAugment=false). Prior SGD
    results no longer apply — re-run 90ep + re-eval (supervise script +
    eval_mnv2_full50k.py unchanged). lr 0.045 is the paper value at the paper's
    batch; if unstable at batch 256, drop the peak to ~0.02. -/
def mobilenetV2ImagenetConfig : TrainConfig where
  learningRate   := 0.045   -- MobileNetV2-native RMSProp peak (was 0.1 for SGD)
  batchSize      := 256
  epochs         := 90      -- near-paper run (validation tier was 30)
  optimizer      := .rmsprop  -- MobileNetV2's original optimizer
  momentum       := 0.9       -- μ for the RMSprop momentum buffer
  rmspropDecay   := 0.9       -- ρ, the running mean-square decay
  rmspropEps     := 1.0       -- MobileNetV2 uses ε=1.0
  weightDecay    := 4e-5
  cosineDecay      := false   -- replaced by the paper exp-decay schedule (gap B)
  expLRDecayRate   := 0.98    -- MobileNetV2: ×0.98 per epoch (after warmup)
  expLRDecayEpochs := 1.0
  dropout          := 0.2     -- MobileNetV2 classifier dropout (gap C)
  warmupEpochs   := 5
  augment        := true    -- random-crop + horizontal flip (MNv2 paper aug)
  useAutoAugment := false   -- MNv2 paper used crop/flip only; AA is beyond the paper
  labelSmoothing := 0.0     -- MNv2 paper (Sandler 2018) used none
  bf16           := true
  bf16Conv       := true    -- now reaches the inverted-residual blocks
  runningBN      := true    -- paper-faithful eval (gap A): running BN stats, not eval-batch stats

#eval mobilenetV2Imagenet.validate!

/-- Paper-faithful full run: identical recipe, the long (350-epoch) schedule.
    Selected with the `full` recipe arg; writes a separate `_full.py` so a quick
    validation subrun and the multi-day run never clobber each other. -/
def mobilenetV2ImagenetConfigFull : TrainConfig :=
  { mobilenetV2ImagenetConfig with epochs := 350 }

def mobilenetV2ImagenetRecipes : List Recipe := [
  { name := "default", cfg := mobilenetV2ImagenetConfig,
    out := "generated_mobilenet_v2_imagenet.py",
    desc := "90-epoch validation tier (RMSProp, crop/flip only)" },
  { name := "full",    cfg := mobilenetV2ImagenetConfigFull,
    out := "generated_mobilenet_v2_imagenet_full.py",
    desc := "paper-faithful 350-epoch run" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "mobilenet-v2-imagenet" mobilenetV2Imagenet .imagenet
    mobilenetV2ImagenetRecipes args
