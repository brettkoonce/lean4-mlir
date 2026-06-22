import Jax

/-! ResNet-50 (bottleneck) on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.

    Architecture is `MainResnet50.lean`'s bottleneck backbone (3/4/6/3) with the
    head swapped to `dense 2048→1000` and the dataset kind set to `.imagenet`
    (tfds streaming), matching `MainResnetImagenet.lean` (the R34 ImageNet trainer).

    PHASE 5 (RSB-A2 plan): the recipe below is now the LITERAL RSB-A2 — timm's
    "ResNet Strikes Back" A2 300-epoch config (Wightman et al. 2021), the
    canonical modern ResNet-50 baseline → 79.8% top-1. All four ingredients
    (LAMB, BCE, repeated-aug, the DeiT-style aug stack) landed across phases 1-4
    and are wired together here. The phase-1 SGD skeleton lives in git history.
    See `planning/rsb_a2_resnet50.md`. -/

def resnet50Imagenet : NetSpec where
  name := "ResNet-50 (ImageNet)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    .bottleneckBlock   64  256 3 1,
    .bottleneckBlock  256  512 4 2,
    .bottleneckBlock  512 1024 6 2,
    .bottleneckBlock 1024 2048 3 2,
    .globalAvgPool,
    .dense 2048 1000 .identity
  ]

/-- Literal RSB-A2 (timm "ResNet Strikes Back" A2, Wightman et al. 2021) → 79.8%
    top-1. LAMB + BCE-with-logits over multi-hot targets, the DeiT-style aug pack
    (Mixup 0.1 + CutMix 1.0 + RandAugment m7-mstd0.5-inc1), Repeated Augmentation
    3×, stochastic depth 0.05, weight decay 0.02, model-EMA, 300 epochs with a
    5-epoch warmup + cosine.

    LR: RSB-A2's reference is lr 5e-3 @ batch 2048; LINEAR-scaled to our batch
    512 (2× 7900 XTX, 256/GPU) → 5e-3 × 512/2048 = 1.25e-3. No label smoothing —
    BCE over the mixup/cutmix soft targets subsumes it (the RSB recipe).

    ROCm note: `bf16Conv := false` — bf16 conv is SLOWER on MIOpen/gfx1100 (a
    CUDA/cuDNN win only). R50 is conv-bound, so the CUDA box (ares) is the real
    run's better home; this config runs correctly on either (set `bf16Conv :=
    true` on CUDA). RandAugment + 3× repeated-aug are CPU-side tf.data — watch
    input throughput (the warmup ETA check confirms it's not input-bound). -/
def resnet50ImagenetConfig : TrainConfig where
  learningRate   := 0.00125  -- RSB-A2 lr 5e-3 @ batch 2048, linear-scaled to batch 512
  batchSize      := 512
  epochs         := 300
  optimizer      := .lamb
  weightDecay    := 0.02
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true     -- random-resized-crop + hflip (the base aug under RA)
  labelSmoothing := 0.0      -- BCE over mixup/cutmix soft labels subsumes it (RSB)
  lossKind       := .bce     -- BCE-with-logits, multi-hot (timm --bce-loss)
  useMixup       := true     -- RSB aug pack: Mixup α0.1...
  mixupAlpha     := 0.1
  useCutmix      := true     -- ...+ CutMix α1.0 (alternates per step)...
  cutmixAlpha    := 1.0
  useRandAugment := true     -- ...+ full RandAugment (color+geometric)...
  randAugmentGeometric := true
  randAugmentN   := 2
  randAugmentM   := 7.0      -- RSB rand-m7-...
  randAugmentMstd := 0.5     -- ...-mstd0.5-...
  randAugmentInc := true     -- ...-inc1 increasing-severity mappings
  repeatedAug    := 3        -- RSB Repeated Augmentation 3× (phase 2)
  dropPath       := 0.05     -- stochastic depth, RSB-A2 value
  useEMA         := true     -- model EMA; eval + checkpoints use the shadow
  emaDecay       := 0.9999
  bf16           := true
  bf16Conv       := false    -- ROCm/MIOpen: bf16 conv is slower; keep conv fp32
  runningBN      := true     -- paper-faithful eval (gap A) + bottleneck running-BN

#eval resnet50Imagenet.validate!

/-- Quick validation subrun: identical RSB-A2 recipe at a 50-epoch tier (enough
    to confirm the curve climbs before committing the ~60-65 hr 300-ep run).
    Selected with `LEAN_MLIR_SHORT=1`; writes a separate `_short.py`. -/
def resnet50ImagenetConfigShort : TrainConfig :=
  { resnet50ImagenetConfig with epochs := 50 }

def main (args : List String) : IO Unit := do
  let short := (← IO.getEnv "LEAN_MLIR_SHORT").isSome
  let cfg := if short then resnet50ImagenetConfigShort else resnet50ImagenetConfig
  let out := if short then "generated_resnet50_imagenet_short.py"
                      else "generated_resnet50_imagenet.py"
  runJax resnet50Imagenet cfg .imagenet (args.head? |>.getD "data/imagenet") out
