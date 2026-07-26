import Jax

/-! Vision Transformer (ViT-B/16, i.e. DeiT-B) on full 1000-class ImageNet —
    bf16 mixed precision, multi-GPU data-parallel (tfds streaming).

    Widths only, relative to `MainVitImagenet.lean` (ViT-Ti): embed 192→768,
    heads 3→12, MLP 768→3072. Depth 12 and patch 16 unchanged; head_dim stays
    64. ~86.6M params — 15× ViT-Ti.

    MEMORY is the one thing that is genuinely different at this size. The
    attention score tensor is `heads × N × N` per sample per block: at 12 heads
    and N = 197 tokens that is 12·197·197·4 B ≈ 1.9 MB/sample/block, 22 MB over
    12 blocks, before the MLP hidden (197·3072·4 B ≈ 2.4 MB/sample/block). There
    is no `jax.checkpoint`/remat in the phase-2 emitter, so every one of those is
    live through the backward pass. On a 16 GB card the `default` recipe at
    batch 512 / 6 devices (≈85 per device) is the thing to measure first; the
    `accum` recipe below is the fallback that keeps the effective batch (and
    therefore the LR) fixed while quartering peak activation. -/

def vitBImagenet : NetSpec where
  name := "ViT-B/16 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 768 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 768 12 3072 12,   -- 12 blocks, 12 heads, MLP 3072
    .dense 768 1000 .identity             -- 1000-class head
  ]

/-- DeiT-B: same recipe as Ti/S (DeiT deliberately uses one recipe for all three
    sizes). Grad-clip 1.0 matters more here, not less — DeiT-B is the size where
    the reference implementation is known to be touchy early in warmup. -/
def vitBImagenetConfig : TrainConfig where
  learningRate   := 0.0005          -- DeiT batch-512 peak LR
  batchSize      := 512
  epochs         := 300
  useAdam        := true
  weightDecay    := 0.05
  wdExcludeNormBias := true
  valEveryEpochs := 5
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  labelSmoothing := 0.1
  gradClipNorm   := 1.0
  useMixup       := true
  mixupAlpha     := 0.8
  useCutmix      := true
  cutmixAlpha    := 1.0
  useRandAugment := true
  randAugmentGeometric := true
  randAugmentM   := 9.0
  randAugmentMstd := 0.5
  randAugmentInc  := true
  randomErasing  := true
  randomErasingProb := 0.25
  dropPath       := 0.1             -- DeiT stochastic depth (same for Ti/S/B)
  useEMA         := true
  emaDecay       := 0.99996
  bf16           := true
  repeatedAug    := 3               -- DeiT Repeated Augmentation 3× (Hoffer et al. 2020 /
                                    -- timm RASampler). Closes the last DeiT faithfulness gap;
                                    -- steps_per_epoch is unchanged, so an epoch sees ~1/3 the
                                    -- unique images ×3 views — same aug cost, not 3×.

#eval vitBImagenet.validate!

/-- 80-epoch validation tier — same comparison point the Ti and S runs use. -/
def vitBImagenetConfigShort : TrainConfig :=
  { vitBImagenetConfig with epochs := 80, repeatedAug := 1 }
  -- repeatedAug off at the 80ep tier so it stays comparable with the ViT-Ti
  -- 80-epoch baseline; the 300ep `default` is the paper-faithful one.

/-- Memory-safe variant for 16 GB cards: effective batch stays 512 (so LR 5e-4
    stays correct) but runs as 4 micro-batches of 128, cutting peak activation
    ~4×. This net has no BN, so accumulation is numerically exact. -/
def vitBImagenetConfigAccum : TrainConfig :=
  { vitBImagenetConfig with batchSize := 128, gradAccumSteps := 4 }

/-- **ViT-B with timm/DeiT weight init** — the `default` recipe plus
    `vitInit := true`. The generic emitter path gives every transformer Linear
    Xavier-uniform, which at ViT-B is **1.8x wider** than timm's
    `trunc_normal(std=0.02)`, while the patch-embed conv comes out ~6x too narrow
    (it divides by the output fan `dim*p*p` instead of the input fan `ic*p*p`).
    The CLS token and positional embedding were already correct at 0.02, so this
    closes an inconsistency inside one file rather than changing a design.

    Kept as a SEPARATE recipe rather than folded into `default`: the A/B is the
    point, and `default` stays comparable with the runs already in `runs/`.

    Measured at init (ViT-B, batch 32, pre-clip global grad norm): Xavier 44.09
    at loss 7.4637, timm 14.28 at loss 7.1597 — 3.1x better conditioned, and a
    starting loss much nearer ln(1000)=6.908. That supports over-wide init as a
    real contributor to the documented LR-5e-4 collapse, but does NOT show
    `gradClipNorm := 1.0` becomes unnecessary: both norms still exceed the
    threshold by 10x+. Settling that needs a clip-off training arm.
    See planning/vit_imagenet.md item 0. -/
def vitBImagenetConfigDeitInit : TrainConfig :=
  { vitBImagenetConfig with vitInit := true }

def vitBImagenetRecipes : List Recipe := [
  { name := "default", cfg := vitBImagenetConfig,
    out := "generated_vit_b_imagenet.py",
    desc := "full DeiT-B 300-epoch schedule, bs512, AdamW + full DeiT aug + EMA" },
  { name := "short",   cfg := vitBImagenetConfigShort,
    out := "generated_vit_b_imagenet_short.py",
    desc := "80-epoch tier (comparison point with the Ti/S runs)" },
  { name := "accum",   cfg := vitBImagenetConfigAccum,
    out := "generated_vit_b_imagenet_accum.py",
    desc := "300ep, effective bs512 as 4×128 grad-accum (fits 16 GB)" },
  { name := "deit-init", cfg := vitBImagenetConfigDeitInit,
    out := "generated_vit_b_imagenet_deitinit.py",
    desc := "300ep + timm trunc_normal(0.02) init — the A/B against Xavier-uniform" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "vit-b-imagenet" vitBImagenet .imagenet
    vitBImagenetRecipes args
