import Jax

/-! MobileNetV2 on full 1000-class ImageNet — Lean → JAX trainer.
    Same inverted-residual body as `MainMobilenetV2.lean` (Imagenette) but
    with a 1000-class head and the `.imagenet` (tfds streaming) dataset.

    bf16 incl. bf16 conv: the Codegen routes the inverted-residual
    expand/project 1x1s and the depthwise through `convdt`, so `bf16Conv`
    reaches the 17 inverted-residual blocks as well as the stem + final 1x1.
    The block-level
    win on the 4060 Ti comes from the 1x1 expand/project (GEMM-like), which
    love bf16; the 3x3 depthwise is a wash but harmless. See the per-op
    microbench in `planning/archive/imagenet_sweep.md`. -/

def mobilenetV2Imagenet : NetSpec where
  name := "MobileNetV2 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  -- The same override EfficientNet-B0 needs (`MainEfficientNetImagenet.lean`).
  -- `NetSpec.convBnAct` defaults to `.relu`
  -- (`LeanMlir/Types.lean`), so without this line the two `.convBn` layers below — the stem and
  -- the 1×1 head — emit `jax.nn.relu` while every inverted-residual interior emits ReLU6.
  -- MobileNetV2 is ReLU6 throughout, as the Imagenette twin (`MainMobilenetV2.lean`) says.
  -- The VERIFIED render is ReLU6 (35 `stablehlo.maximum` paired with 35
  -- `stablehlo.minimum`, including at the 32×112×112 stem).
  -- The published JAX reference (the 350-epoch `full` recipe, 71.90) trained a ReLU stem/head,
  -- so it does not pair one-variable with the verified run.
  convBnAct := .relu6
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

/-- MobileNetV2 90-epoch tier (`default`); the published reference is the 350-epoch `full`
    recipe below, which is this config with only `epochs` changed.

    RMSProp + momentum 0.9, base lr 0.045 at batch 256, and the paper's schedule: ×0.98 per
    epoch as a staircase from step 0, with no warmup (TF-slim's `exponential_decay(staircase=True)`).
    Two MobileNet-specific choices:
      * weight decay 4e-5 (not 1e-4): large wd hurts the tiny depthwise
        weights; 4e-5 is the standard MobileNet value. Coupled L2, off BN γ/β and biases
        (`wdExcludeNormBias`); depthwise kernels still decay, where TF-slim skips them.
      * no mixup/cutmix: not standard for MobileNetV2.
    RMSProp knobs: ρ=0.9 (rmspropDecay), μ=0.9 (momentum), ε=1.0 (rmspropEps —
    MobileNetV2's value, NOT 1e-8). BatchNorm is TF-slim's: decay 0.997, ε 1e-3.
    Aug is crop/flip only (MobileNetV2 used no AutoAugment).
    Label smoothing 0.0 and classifier dropout 0.2, as the paper. -/
def mobilenetV2ImagenetConfig : TrainConfig where
  learningRate   := 0.045   -- MobileNetV2-native RMSProp peak
  batchSize      := 256
  epochs         := 90      -- near-paper run
  optimizer      := .rmsprop  -- MobileNetV2's original optimizer
  momentum       := 0.9       -- μ for the RMSprop momentum buffer
  rmspropDecay   := 0.9       -- ρ, the running mean-square decay
  rmspropEps     := 1.0       -- MobileNetV2 uses ε=1.0
  weightDecay    := 4e-5
  wdExcludeNormBias := true   -- no decay on BN γ/β or biases (slim's rule for BN; the verified `wx`)
  cosineDecay      := false   -- the paper exp-decay schedule instead
  expLRDecayRate   := 0.98    -- MobileNetV2: ×0.98 per epoch
  expLRDecayEpochs := 1.0
  expLRStaircase   := true    -- the paper's staircase, counted from step 0
  dropout          := 0.2     -- MobileNetV2 classifier dropout
  warmupEpochs   := 0         -- the paper has no warmup
  augment        := true    -- random-crop + horizontal flip (MNv2 paper aug)
  useAutoAugment := false   -- MNv2 paper used crop/flip only; AA is beyond the paper
  labelSmoothing := 0.0     -- MNv2 paper (Sandler 2018) used none
  bf16           := true
  bf16Conv       := true    -- reaches the inverted-residual blocks
  runningBN      := true    -- paper-faithful eval: running BN stats, not eval-batch stats
  bnMomentum     := 0.997   -- TF-slim's BN decay (PyTorch momentum 0.003)
  bnEps          := 1e-3    -- TF-slim's BN ε

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
