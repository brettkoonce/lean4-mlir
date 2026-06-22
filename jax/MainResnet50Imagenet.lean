import Jax

/-! ResNet-50 (bottleneck) on full 1000-class ImageNet — phase-2 (Lean → JAX) trainer.

    Architecture is `MainResnet50.lean`'s bottleneck backbone (3/4/6/3) with the
    head swapped to `dense 2048→1000` and the dataset kind set to `.imagenet`
    (tfds streaming), matching `MainResnetImagenet.lean` (the R34 ImageNet trainer).

    PHASE 1 SKELETON (RSB-A2 plan): this is the *working R50/ImageNet host* — a
    real R50 trainer that builds and runs at ImageNet scale before the RSB-A2
    features (LAMB, BCE, repeated-aug) layer on. The recipe below is the R34
    SGD 90-epoch config so it trains NOW; `runningBN := true` exercises the
    bottleneck running-BN threading just landed in `Codegen.lean`. Phase 5 swaps
    in the literal RSB-A2 `resnet50ImagenetConfig`. See
    `planning/rsb_a2_resnet50.md`. -/

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

/-- Placeholder 90-epoch recipe (mirrors `resnet34ImagenetConfig`): SGD +
    momentum 0.9, base lr 0.1 @ batch 256, 5-epoch warmup, cosine decay, weight
    decay 1e-4, label smoothing 0.1, crop + hflip. `runningBN := true` for
    paper-faithful eval (gap A) and to exercise the bottleneck running-BN path.

    ROCm note: `bf16Conv := false` — bf16 conv is SLOWER on MIOpen/gfx1100 (it's
    a CUDA/cuDNN win only). R50 is conv-bound, so the CUDA box is its better home
    for the real run; this config is sized to build + smoke-test on the 7900 XTX
    box. Phase 5 replaces this with the literal RSB-A2 LAMB/BCE recipe. -/
def resnet50ImagenetConfig : TrainConfig where
  learningRate   := 0.1
  batchSize      := 256
  epochs         := 90
  useAdam        := false
  momentum       := 0.9
  weightDecay    := 1e-4
  cosineDecay    := true
  warmupEpochs   := 5
  augment        := true
  labelSmoothing := 0.1
  bf16           := true
  bf16Conv       := false   -- ROCm/MIOpen: bf16 conv is slower; keep conv fp32
  runningBN      := true    -- paper-faithful eval (gap A) + bottleneck running-BN

#eval resnet50Imagenet.validate!

/-- Quick validation subrun: identical recipe at a 30-epoch tier. Selected with
    `LEAN_MLIR_SHORT=1`; writes a separate `_short.py`. -/
def resnet50ImagenetConfigShort : TrainConfig :=
  { resnet50ImagenetConfig with epochs := 30 }

def main (args : List String) : IO Unit := do
  let short := (← IO.getEnv "LEAN_MLIR_SHORT").isSome
  let cfg := if short then resnet50ImagenetConfigShort else resnet50ImagenetConfig
  let out := if short then "generated_resnet50_imagenet_short.py"
                      else "generated_resnet50_imagenet.py"
  runJax resnet50Imagenet cfg .imagenet (args.head? |>.getD "data/imagenet") out
