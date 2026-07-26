import Jax

/-! Vision Transformer (ViT-S/16, i.e. DeiT-S) on full 1000-class ImageNet — bf16
    mixed precision, multi-GPU data-parallel (tfds streaming).

    Same shape as `MainVitImagenet.lean` (ViT-Ti) with the DeiT-S widths:
    embed 192→384, heads 3→6, MLP 768→1536. Depth stays 12 and patch stays 16,
    so head_dim is 64 in both (384/6 = 192/3 · 1 = 64) — the `mhsa` helper
    derives head_dim = D // n_heads, so nothing in codegen changes.
    ~22.0M params vs ViT-Ti's 5.7M.

    Recipe is DeiT's: the paper uses ONE recipe for Ti/S/B and only changes the
    width, so this is deliberately a byte-for-byte copy of the ViT-Ti config —
    including LR 5e-4 at batch 512 and stochastic depth 0.1 (DeiT Table 9 lists
    0.1 for every model size). That is the point of this file: if ViT-S needs
    anything ViT-Ti didn't, it should show up as a training failure, not as a
    config difference. -/

def vitSImagenet : NetSpec where
  name := "ViT-S/16 (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 384 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 384 6 1536 12,    -- 12 blocks, 6 heads, MLP 1536
    .dense 384 1000 .identity             -- 1000-class head
  ]

/-- DeiT-S: identical to `vitTinyImagenetConfig` (DeiT applies one recipe across
    Ti/S/B). Grad-clip 1.0 is load-bearing at LR 5e-4 — see the ViT-Ti notes. -/
def vitSImagenetConfig : TrainConfig where
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

#eval vitSImagenet.validate!

/-- 80-epoch validation tier — the schedule ViT-Ti was actually run at (65.6%),
    so it is the apples-to-apples comparison point before committing to 300ep. -/
def vitSImagenetConfigShort : TrainConfig :=
  { vitSImagenetConfig with epochs := 80, repeatedAug := 1 }
  -- repeatedAug off at the 80ep tier so it stays comparable with the ViT-Ti
  -- 80-epoch baseline; the 300ep `default` is the paper-faithful one.

/-- Memory-safe variant for 16 GB cards: same effective batch 512, but split
    into 2 micro-batches of 256 so peak activation is halved. Numerically the
    grads are averaged over the same 512 images (BN-free net, so accumulation is
    exact here); LR still targets the effective batch. -/
def vitSImagenetConfigAccum : TrainConfig :=
  { vitSImagenetConfig with batchSize := 256, gradAccumSteps := 2 }

def vitSImagenetRecipes : List Recipe := [
  { name := "default", cfg := vitSImagenetConfig,
    out := "generated_vit_s_imagenet.py",
    desc := "full DeiT-S 300-epoch schedule, bs512, AdamW + full DeiT aug + EMA" },
  { name := "short",   cfg := vitSImagenetConfigShort,
    out := "generated_vit_s_imagenet_short.py",
    desc := "80-epoch tier (apples-to-apples with the ViT-Ti 65.6% run)" },
  { name := "accum",   cfg := vitSImagenetConfigAccum,
    out := "generated_vit_s_imagenet_accum.py",
    desc := "300ep, effective bs512 as 2×256 grad-accum (16 GB headroom)" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "vit-s-imagenet" vitSImagenet .imagenet
    vitSImagenetRecipes args
