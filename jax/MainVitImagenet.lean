import Jax

/-! Vision Transformer (ViT-Tiny) on full 1000-class ImageNet — bf16
    mixed precision, data-parallel over every visible GPU (tfds streaming; the
    published run used 4, at 128 per device = global 512). Same ViT-Tiny
    spec as `MainVit.lean` but with a 1000-class head and the `.imagenet`
    dataset.

    bf16 routes every matmul (patch embed, attention QKV/scores/out, MLP,
    head) through bfloat16 while keeping LayerNorm/softmax/GELU in fp32.
    Note: at ViT-Tiny scale the run may be input-bound on tfds aug. -/

def vitTinyImagenet : NetSpec where
  name := "ViT-Tiny (ImageNet, bf16)"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP 768
    .dense 192 1000 .identity             -- 1000-class head
  ]

/-- Paper-faithful DeiT-Ti recipe (300 epochs): AdamW, 5-epoch warmup +
    cosine, weight decay 0.05, label smoothing 0.1, grad-clip 1.0, the full
    DeiT aug suite (Mixup/CutMix/RandAugment color+geometric/Random Erasing),
    stochastic depth 0.1, model EMA, bf16 matmuls. Peak LR 5e-4 at batch 512.
    Grad clipping is the unlock: 5e-4/2e-4 collapsed to chance the moment
    warmup ramped LR past ~1.6e-4 (train loss pinned at ln(1000)) without it.
    The 80-epoch grad-clip-only ancestor of this recipe is
    `jax/runs/vit_tiny_imagenet_bf16_80ep/`; the additions here (geometric RA,
    stochastic depth, EMA, 300ep) target the
    ~72% DeiT-Ti headline (no distillation). -/
def vitTinyImagenetConfig : TrainConfig where
  vitInit        := true            -- timm/DeiT trunc_normal(0.02) init, not the emitter's Xavier-uniform
  learningRate   := 0.0005          -- proper DeiT batch-512 LR
  batchSize      := 512
  epochs         := 300             -- full DeiT-Ti schedule
  optimizer      := .adam
  weightDecay    := 0.05            -- applied as AdamW decoupled decay
  wdExcludeNormBias := true          -- timm no_weight_decay: skip norm/bias/pos-embed/CLS (DeiT-faithful)
  valEveryEpochs := 5                 -- ImageNet val is data-loading-bound (~75s/ep); every-5 saves ~5h over 300ep
  cosineDecay    := true
  warmupEpochs   := 5
  minLR          := 0.00001         -- DeiT `--min-lr 1e-5` (its `--cooldown-epochs 10` is dead: DeiT's
                                    -- main.py discards create_scheduler's epoch count and stops at 300)
  augment        := true
  labelSmoothing := 0.1
  gradClipNorm   := 1.0             -- DeiT default; the unlock for the 5e-4 LR
  useMixup       := true            -- DeiT aug suite (mixup + cutmix alternate per step)
  mixupAlpha     := 0.8
  useCutmix      := true
  cutmixAlpha    := 1.0
  useRandAugment := true            -- full DeiT RandAugment (color + geometric, below)
  augBicubic     := true    -- PIL-bicubic geometry, as timm
  randAugmentGeometric := true      -- shear/rotate/translate via ImageProjectiveTransformV3
  randAugmentM   := 9.0
  randAugmentMstd := 0.5            -- DeiT rand-m9-mstd0.5
  randAugmentInc  := true           -- ...-inc1 increasing-severity mappings
  randomErasing  := true
  randomErasingProb := 0.25
  erasingPixel   := true    -- timm RandomErasing(mode='pixel'), N(0,1) fill
  dropPath       := 0.1             -- DeiT-Ti stochastic depth (linear ramp 0→0.1 over blocks)
  useEMA         := true            -- DeiT model EMA; eval + checkpoints use the shadow weights
  emaDecay       := 0.99996         -- DeiT default
  bf16           := true            -- fp32 collapsed identically, so bf16 stays on for speed
  lnEps          := 0.000001        -- DeiT `partial(nn.LayerNorm, eps=1e-6)` (the render's `eps0000001`)
  f32StemHead    := true            -- patch embed + head in fp32: the verified render's carve-out,
                                    -- so the pair differs in no precision site (2026-10-01)
  repeatedAug    := 3               -- DeiT Repeated Augmentation 3× (Hoffer et al. 2020 /
                                    -- timm RASampler). Closes the last DeiT faithfulness gap;
                                    -- steps_per_epoch is unchanged, so an epoch sees ~1/3 the
                                    -- unique images ×3 views — same aug cost, not 3×.

#eval vitTinyImagenet.validate!

/-- **ViT-Ti with the emitter's Xavier-uniform init** — the A/B arm against `default`, which
    carries timm/DeiT init (`vitInit := true`, the recipe behind the book's 72.31). The generic emitter path gives every transformer Linear
    Xavier-uniform, which at ViT-Ti is **3.6x wider** than timm's
    `trunc_normal(std=0.02)`, while the patch-embed conv comes out ~6x too narrow
    (it divides by the output fan `dim*p*p` instead of the input fan `ic*p*p`).
    The CLS token and positional embedding are at 0.02 in both arms.

    Measured once at init (ViT-B, batch 32, pre-clip global grad norm): timm's init
    starts with a grad norm about a third of Xavier's and a loss much nearer
    ln(1000)=6.908. That supports over-wide init as a real contributor to the
    documented LR-5e-4 collapse, but does NOT show `gradClipNorm := 1.0` becomes
    unnecessary: both norms still exceed the threshold by an order of magnitude. Settling that needs a clip-off training arm. -/
def vitTinyImagenetConfigXavier : TrainConfig :=
  { vitTinyImagenetConfig with vitInit := false }

/-- Named training recipes, selected by a positional CLI arg
    (`vit-tiny-imagenet <recipe> [data_dir]`, listed by `--help`). -/
def vitTinyImagenetRecipes : List Recipe := [
  { name := "default", cfg := vitTinyImagenetConfig,
    out := "generated_vit_tiny_imagenet.py",
    desc := "full DeiT-Ti 300-epoch schedule, bs512, AdamW + full DeiT aug + EMA" },
  { name := "xavier", cfg := vitTinyImagenetConfigXavier,
    out := "generated_vit_tiny_imagenet_xavier.py",
    desc := "300ep with the emitter's Xavier-uniform init — the A/B arm against `default`" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "vit-tiny-imagenet" vitTinyImagenet .imagenet
    vitTinyImagenetRecipes args
