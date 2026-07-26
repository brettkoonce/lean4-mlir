import Jax

/-! ConvNeXt-Base on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.

    Relative to `MainConvNeXtSImagenet.lean`: same (3,3,27,3) compute ratio,
    channels widened 96→128 throughout (128,256,512,1024). ~88.6M params.
    Relative to ConvNeXt-T it is both deeper (18→36 blocks) and wider.

    Same known deviation as ConvNeXt-T/S: no final LayerNorm between the global
    average pool and the head.

    This is the heaviest net in the sweep by a wide margin — deeper AND wider —
    so on a 16 GB card expect to need the `accum` recipe. The stage-1 inverted
    bottleneck alone holds 4·128 channels at 56×56 per sample per block. -/

def convNeXtBImagenet : NetSpec where
  name := "ConvNeXt-B (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .convNextStem 3 128 4,                     -- patchify stem: 4×4 s4 conv → channel-LN
    .convNextStage 128 3 .ln .gelu,            -- stage 1: 3 blocks @ 128
    .convNextDownsample 128 256,               -- 56→28
    .convNextStage 256 3 .ln .gelu,            -- stage 2: 3 blocks @ 256
    .convNextDownsample 256 512,               -- 28→14
    .convNextStage 512 27 .ln .gelu,           -- stage 3: 27 blocks @ 512
    .convNextDownsample 512 1024,              -- 14→7
    .convNextStage 1024 3 .ln .gelu,           -- stage 4: 3 blocks @ 1024
    .globalAvgPool,
    .dense 1024 1000 .identity                 -- 1000-class head
  ]

/-- ConvNeXt-B, 300-epoch paper schedule. Same as S except stochastic depth
    0.4 → 0.5 (the paper's B value). LR 2.5e-4 = 4e-3@bs4096 scaled to bs256. -/
def convNeXtBImagenetConfig : TrainConfig where
  learningRate   := 2.5e-4
  batchSize      := 256
  epochs         := 300
  useAdam        := true
  weightDecay    := 0.05
  wdExcludeNormBias := true
  cosineDecay    := true
  warmupEpochs   := 20
  augment        := true
  useRandAugment       := true
  randAugmentGeometric := true
  randAugmentMstd := 0.5
  randAugmentInc  := true
  useMixup       := true
  mixupAlpha     := 0.8
  useCutmix      := true
  cutmixAlpha    := 1.0
  randomErasing  := true
  randomErasingProb := 0.25
  labelSmoothing := 0.1
  gradClipNorm   := 1.0
  bf16           := true
  bf16Conv       := true
  useEMA         := true
  dropPath       := 0.5      -- ConvNeXt-B paper value
  valEveryEpochs := 5

#eval convNeXtBImagenet.validate!

/-- 80-epoch validation tier. dropPath pulled back 0.5 → 0.3 for the same reason
    as ConvNeXt-S's short recipe: the paper value is tuned for 300 epochs. -/
def convNeXtBImagenetConfigShort : TrainConfig :=
  { convNeXtBImagenetConfig with epochs := 80, warmupEpochs := 5, dropPath := 0.3 }

/-- Memory-safe variant: effective batch stays 256 (LR unchanged) as 4×64. -/
def convNeXtBImagenetConfigAccum : TrainConfig :=
  { convNeXtBImagenetConfig with batchSize := 64, gradAccumSteps := 4 }

/-- Effective batch 512 instead of 256, as 4×128 grad-accum. LR doubles with the
    batch under the paper's linear rule (4e-3 @ bs4096 → 5e-4 @ bs512).

    Accumulation here is not a concession — it is strictly better. Measured on
    4× 16 GB: one-shot bs512 peaks at 11.52 of 11.68 GiB and runs 747 ms/step;
    4×128 peaks at 6.19 GiB and runs **681 ms/step**. Under memory pressure XLA
    rematerializes, and that recompute costs more than the accumulation loop.
    Also beats bs256 per epoch (28.4 min vs 32.9). -/
def convNeXtBImagenetConfigBs512 : TrainConfig :=
  { convNeXtBImagenetConfig with
      batchSize := 128, gradAccumSteps := 4, learningRate := 5.0e-4 }

def convNeXtBImagenetRecipes : List Recipe := [
  { name := "default", cfg := convNeXtBImagenetConfig,
    out := "generated_convnext_b_imagenet.py",
    desc := "paper-faithful 300-epoch run (dropPath 0.5)" },
  { name := "short",   cfg := convNeXtBImagenetConfigShort,
    out := "generated_convnext_b_imagenet_short.py",
    desc := "80-epoch validation tier (dropPath pulled back to 0.3)" },
  { name := "accum",   cfg := convNeXtBImagenetConfigAccum,
    out := "generated_convnext_b_imagenet_accum.py",
    desc := "300ep, effective bs256 as 4×64 grad-accum (fits 16 GB)" },
  { name := "bs512",   cfg := convNeXtBImagenetConfigBs512,
    out := "generated_convnext_b_imagenet_bs512.py",
    desc := "300ep at effective bs512 (4×128 accum), LR 5e-4 — faster AND smaller than one-shot bs512" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "convnext-b-imagenet" convNeXtBImagenet .imagenet
    convNeXtBImagenetRecipes args
