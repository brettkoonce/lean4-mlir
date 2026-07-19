import Jax

/-! MobileNetV4-Conv-M on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.

    The pre-existing `MainMobilenetV4.lean` is a 10-class Imagenette DEMO whose
    block table is Conv-S-sized (~4.1M params). This spec is the *faithful*
    MobileNetV4-Conv-Medium (~9.7M params, paper 79.9% top-1 non-distilled),
    transcribed 1:1 from timm `mobilenetv4_conv_medium` (`_gen_mobilenet_v4`,
    `timm/models/mobilenetv3.py`). See planning/mnv4_imagenet.md for the decode
    of timm's `uir_rN_aA_kK_sS_eE_cC` encoding into `.uib ic oc expand stride
    preDWk postDWk` (a=pre/start-DW kernel, k=post/mid-DW kernel).

    SPIKE STATUS (2026-07-19): `runningBN := false` — the UIB block is not yet
    wired into the codegen's running-BN threading (template = `mbconv_block`,
    Codegen.lean:684). This trains correctly; eval uses batch stats (slightly
    noisy, not paper-faithful). Flip to `true` after the wiring lands.

    Resolution 224: the tfds pipeline hardcodes `_IMG_SIZE=224`, and timm ships
    an official `mobilenetv4_conv_medium.e500_r224_in1k` variant, so 224 is a
    published-faithful choice. (r256 would need `_IMG_SIZE` to derive from the
    spec — a separate codegen change.) -/

def mobilenetV4ConvMImagenet : NetSpec where
  name := "MobileNetV4-Conv-M (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 32 3 2 .same,                       -- 224→112  stem
    -- stage 0 (FusedIB): er_r1_k3_s2_e4_c48
    .fusedMbConv 32 48 4 3 2 1 false,             -- 112→56
    -- stage 1: 2× UIB
    .uib  48  80 4 2 3 5,                          -- 56→28   ExtraDW
    .uib  80  80 2 1 3 3,                          --         ExtraDW
    -- stage 2: 8× UIB
    .uib  80 160 6 2 3 5,                          -- 28→14   ExtraDW
    .uib 160 160 4 1 3 3,                          --         ExtraDW  (uir_r2 #1)
    .uib 160 160 4 1 3 3,                          --         ExtraDW  (uir_r2 #2)
    .uib 160 160 4 1 3 5,                          --         ExtraDW
    .uib 160 160 4 1 3 3,                          --         ExtraDW
    .uib 160 160 4 1 3 0,                          --         ConvNeXt
    .uib 160 160 2 1 0 0,                          --         FFN
    .uib 160 160 4 1 3 0,                          --         ConvNeXt
    -- stage 3: 11× UIB
    .uib 160 256 6 2 5 5,                          -- 14→7    ExtraDW
    .uib 256 256 4 1 5 5,                          --         ExtraDW
    .uib 256 256 4 1 3 5,                          --         ExtraDW  (uir_r2 #1)
    .uib 256 256 4 1 3 5,                          --         ExtraDW  (uir_r2 #2)
    .uib 256 256 4 1 0 0,                          --         FFN
    .uib 256 256 4 1 3 0,                          --         ConvNeXt
    .uib 256 256 2 1 3 5,                          --         ExtraDW
    .uib 256 256 4 1 5 5,                          --         ExtraDW
    .uib 256 256 4 1 0 0,                          --         FFN      (uir_r2 #1)
    .uib 256 256 4 1 0 0,                          --         FFN      (uir_r2 #2)
    .uib 256 256 2 1 5 0,                          --         ConvNeXt
    -- head: cn_r1_k1_s1_c960 → conv_head 1280 → GAP → FC
    .convBn 256 960 1 1 .same,                     -- 7  cn to 960
    .convBn 960 1280 1 1 .same,                    -- 7  conv_head (num_features)
    .globalAvgPool,
    .dense 1280 1000 .identity                     -- 1000-class head
  ]

/-- Tier 2 — reduced-regularization ~100-epoch *confidence* tier (repo default).

    The paper recipe (500ep, dropPath 0.075, RandAug m15 p0.7, wd 0.1,
    dropout 0.2) is tuned for the long schedule and UNDERFITS short. This tier
    dials regularization down for a real go/no-go signal in ~1–1.5 days.
    Expect low-to-mid 70s (paper 79.9 needs the full 500ep). See
    planning/mnv4_imagenet.md "Recipe tiers". -/
def mobilenetV4ConvMImagenetConfig : TrainConfig where
  learningRate         := 0.004    -- paper peak @ bs4096; here targets the effective batch
  batchSize            := 512
  gradAccumSteps       := 8        -- effective batch 4096 (paper) at one micro-batch's activation cost
  epochs               := 100
  useAdam              := true     -- AdamW (decoupled wd)
  weightDecay          := 0.05     -- reduced from paper 0.1 for the short schedule
  wdExcludeNormBias    := true     -- timm excludes norm γ/β + biases from decay
  cosineDecay          := true
  warmupEpochs         := 5
  labelSmoothing       := 0.1
  dropout              := 0.1      -- reduced from paper 0.2
  augment              := true
  useRandAugment       := true
  randAugmentGeometric := true     -- full color+geometric sampler
  randAugmentN         := 2
  randAugmentM         := 9.0      -- reduced from paper 15 for the short schedule
  useEMA               := true
  emaDecay             := 0.9999
  valEveryEpochs       := 5        -- eval every 5 ep (per-epoch 50k-img val wastes ~1.75h over 100ep)
  bf16                 := true
  bf16Conv             := true
  runningBN            := true     -- paper-faithful eval: running BN stats (UIB + fusedMbConv wired 2026-07-19)

#eval mobilenetV4ConvMImagenet.validate!

/-- Tier 1 — ~30-epoch quick signal (does the 1000-class tfds path climb?). -/
def mobilenetV4ConvMImagenetConfigProbe : TrainConfig :=
  { mobilenetV4ConvMImagenetConfig with epochs := 30 }

/-- Throughput bench: micro-batch 128 (== one GPU's shard of the real 4-GPU
    512 micro-batch), no grad-accum, single epoch. For measuring per-GPU img/s
    to extrapolate a run-time estimate; kill after a couple hundred steps. -/
def mobilenetV4ConvMImagenetConfigBench : TrainConfig :=
  { mobilenetV4ConvMImagenetConfig with
      epochs := 1, batchSize := 128, gradAccumSteps := 1, useEMA := false }

/-- Tier 3 — paper-faithful 500-epoch run. Full regularization pack; rent for it. -/
def mobilenetV4ConvMImagenetConfigFull : TrainConfig :=
  { mobilenetV4ConvMImagenetConfig with
      epochs         := 500
      weightDecay    := 0.1        -- paper
      dropout        := 0.2        -- paper
      randAugmentM   := 15.0       -- paper (NB: codegen clamps M to 0–10 — verify scale)
      dropPath       := 0.075 }    -- paper (NB: not yet wired into UIB — verify before trusting)

def mobilenetV4ConvMImagenetRecipes : List Recipe := [
  { name := "default", cfg := mobilenetV4ConvMImagenetConfig,
    out := "generated_mobilenet_v4_imagenet.py",
    desc := "Tier-2 ~100ep reduced-reg confidence run" },
  { name := "probe",   cfg := mobilenetV4ConvMImagenetConfigProbe,
    out := "generated_mobilenet_v4_imagenet_probe.py",
    desc := "Tier-1 ~30ep quick-signal run" },
  { name := "full",    cfg := mobilenetV4ConvMImagenetConfigFull,
    out := "generated_mobilenet_v4_imagenet_full.py",
    desc := "Tier-3 paper-faithful 500ep run" },
  { name := "bench",   cfg := mobilenetV4ConvMImagenetConfigBench,
    out := "generated_mobilenet_v4_imagenet_bench.py",
    desc := "throughput bench: micro-batch 128, single GPU" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "mobilenet-v4-imagenet" mobilenetV4ConvMImagenet .imagenet
    mobilenetV4ConvMImagenetRecipes args
