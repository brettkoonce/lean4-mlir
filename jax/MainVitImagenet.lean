import Jax

/-! Vision Transformer (ViT-Tiny) on full 1000-class ImageNet — bf16
    mixed precision, 2-GPU data-parallel (tfds streaming). Same ViT-Tiny
    spec as `MainVit.lean` but with a 1000-class head and the `.imagenet`
    dataset.

    bf16 routes every matmul (patch embed, attention QKV/scores/out, MLP,
    head) through bfloat16 while keeping LayerNorm/softmax/GELU in fp32 —
    measured ~2.7× on the isolated ViT-Ti matmuls on gfx1100, where
    convnets see no bf16 benefit. See reference_bf16_gfx1100_conv_vs_gemm.
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
    The 80-epoch grad-clip-only ancestor of this recipe reached 65.6% top-1;
    the additions here (geometric RA, stochastic depth, EMA, 300ep) target the
    ~72% DeiT-Ti headline (no distillation). -/
def vitTinyImagenetConfig : TrainConfig where
  learningRate   := 0.0005          -- proper DeiT batch-512 LR (was crippled at 1e-4)
  batchSize      := 512
  epochs         := 300             -- full DeiT-Ti schedule (was 80; closes ~65→72%)
  useAdam        := true
  weightDecay    := 0.05            -- now applied as AdamW decoupled decay (was toxic coupled-L2)
  wdExcludeNormBias := true          -- timm no_weight_decay: skip norm/bias/pos-embed/CLS (DeiT-faithful)
  valEveryEpochs := 5                 -- ImageNet val is data-loading-bound (~75s/ep); every-5 saves ~5h over 300ep
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  labelSmoothing := 0.1             -- now actually applied (was ignored by the JAX loss)
  gradClipNorm   := 1.0             -- DeiT default; the unlock for the 5e-4 LR
  useMixup       := true            -- DeiT aug suite (mixup + cutmix alternate per step)
  mixupAlpha     := 0.8
  useCutmix      := true
  cutmixAlpha    := 1.0
  useRandAugment := true            -- full DeiT RandAugment (color + geometric, below)
  randAugmentGeometric := true      -- shear/rotate/translate via ImageProjectiveTransformV3
  randAugmentM   := 9.0
  randAugmentMstd := 0.5            -- DeiT rand-m9-mstd0.5 (gap D)
  randAugmentInc  := true           -- ...-inc1 increasing-severity mappings
  randomErasing  := true
  randomErasingProb := 0.25
  dropPath       := 0.1             -- DeiT-Ti stochastic depth (linear ramp 0→0.1 over blocks)
  useEMA         := true            -- DeiT model EMA; eval + checkpoints use the shadow weights
  emaDecay       := 0.99996         -- DeiT default
  bf16           := true            -- bf16 exonerated (fp32 collapsed identically); back on for speed

#eval vitTinyImagenet.validate!

/-- Quick validation subrun: identical recipe at the 80-epoch tier (the proven
    historical point — that schedule reached 65.6%). Selected with
    `LEAN_MLIR_SHORT=1`; writes a separate `_short.py` so it never clobbers the
    300-epoch official run. -/
def vitTinyImagenetConfigShort : TrainConfig :=
  { vitTinyImagenetConfig with epochs := 80 }

/-- A named training recipe (config + generated-file name + description). Selected
    by a positional CLI arg (`vit-tiny-imagenet <recipe> [data_dir]`), listed by
    `--help`; the `LEAN_MLIR_SHORT` env flag is kept as a deprecated fallback so the
    supervise scripts keep working. -/
structure VitRecipe where
  name : String
  cfg  : TrainConfig
  out  : String
  desc : String

def vitTinyImagenetRecipes : List VitRecipe := [
  { name := "default", cfg := vitTinyImagenetConfig,
    out := "generated_vit_tiny_imagenet.py",
    desc := "full DeiT-Ti 300-epoch schedule, bs512, AdamW + full DeiT aug + EMA" },
  { name := "short",   cfg := vitTinyImagenetConfigShort,
    out := "generated_vit_tiny_imagenet_short.py",
    desc := "80-epoch tier (the proven historical point, reached 65.6%)" }
]

def main (args : List String) : IO Unit := do
  if args.any (fun a => a == "--help" || a == "-h") then
    IO.println "usage: vit-tiny-imagenet [recipe] [data_dir]\n"
    IO.println "recipes (default if omitted: \"default\"):"
    for r in vitTinyImagenetRecipes do
      let pad := String.mk (List.replicate (10 - r.name.length) ' ')
      IO.println s!"  {r.name}{pad}{r.desc}"
    IO.println "\ndata_dir defaults to \"data/imagenet\"."
    IO.println "(legacy LEAN_MLIR_SHORT env flag still honored)"
    return
  let cliName := args.find? (fun a => vitTinyImagenetRecipes.any (·.name == a))
  let shortEnv ← IO.getEnv "LEAN_MLIR_SHORT"
  let envName := if shortEnv.isSome then some "short" else none
  let name := (cliName.orElse (fun _ => envName)).getD "default"
  match vitTinyImagenetRecipes.find? (·.name == name) with
  | none   => IO.eprintln s!"unknown recipe '{name}' — run with --help for the list"
  | some r =>
    let dataDir := (args.filter (fun a => a != r.name && !a.startsWith "-")).head?
                     |>.getD "data/imagenet"
    IO.println s!"[vit-tiny-imagenet] recipe '{r.name}': {r.desc}"
    IO.println s!"[vit-tiny-imagenet]   -> {r.out}  (data: {dataDir})"
    runJax vitTinyImagenet r.cfg .imagenet dataDir r.out
