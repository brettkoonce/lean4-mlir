import Jax

/-! ResNet-34 on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.
    Identical body to `MainResnet.lean`'s Imagenette ResNet-34 but the head
    is dense 512→1000 and the dataset kind is `.imagenet` (tfds streaming).
    Mirrors the snippet in blueprint Ch 6 §"What 90-epoch ImageNet would
    look like". -/

def resnet34Imagenet : NetSpec where
  name := "ResNet-34 (ImageNet)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 1000 .identity
  ]

/-- Paper-flavored 90-epoch recipe: SGD + momentum 0.9, base lr 0.1 at
    batch 256, 5-epoch warmup, cosine decay, weight decay 1e-4, label
    smoothing 0.1, random-crop + horizontal flip. The batch is sized to
    divide evenly across 6 NVIDIA GPUs (256 → BATCH_SIZE round-down per
    device * n_devices). Override via env if you want shorter validation
    runs (this is phase-2 codegen — you can just edit the spec). -/
def resnet34ImagenetConfig : TrainConfig where
  learningRate   := 0.1
  batchSize      := 256
  epochs         := 90      -- full paper recipe (4-GPU bf16 run, ~18 hr clean)

  useAdam        := false
  momentum       := 0.9
  weightDecay    := 1e-4
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  labelSmoothing := 0.1
  -- bf16 mixed precision (incl. bf16 conv): a CUDA/cuDNN recipe — 1.60x faster
  -- than fp32 on the 4060 Ti box, reaching 72.0% top-1 / 90.6% top-5 over the
  -- full 50k val (see jax/runs/r34_imagenet_bf16_90ep/RESULTS.md). On AMD/MIOpen
  -- set bf16Conv := false (bf16 conv is slower there); see reference_ares_pcie_aer.
  bf16           := true
  bf16Conv       := true
  runningBN      := true    -- paper-faithful eval (gap A): running BN stats, not eval-batch stats

#eval resnet34Imagenet.validate!

/-- Quick validation subrun: identical recipe at a 30-epoch tier (r34 trains
    fast; enough to confirm the curve climbs before the 90-epoch run). The
    `short` recipe arg; writes a separate `_short.py`. -/
def resnet34ImagenetConfigShort : TrainConfig :=
  { resnet34ImagenetConfig with epochs := 30 }

/-- Named training recipes, selected by a positional CLI arg
    (`resnet34-imagenet <recipe> [data_dir]`, listed by `--help`). -/
def resnet34ImagenetRecipes : List Recipe := [
  { name := "default", cfg := resnet34ImagenetConfig,
    out := "generated_resnet34_imagenet.py",
    desc := "full 90-epoch paper recipe, bs256, SGD+momentum, bf16 (-> 72.0% top-1)" },
  { name := "short",   cfg := resnet34ImagenetConfigShort,
    out := "generated_resnet34_imagenet_short.py",
    desc := "quick 30-epoch validation subrun (same recipe)" }
]

def main (args : List String) : IO Unit :=
  runRecipeMain "resnet34-imagenet" resnet34Imagenet .imagenet
    resnet34ImagenetRecipes args
