import LeanMlir.VerifiedNets

/-! # `convnext-imagenet-verified` — ConvNeXt-T on full ImageNet-1k, verified renderer → XLA

The ConvNeXt peer of `resnet34-imagenet-verified` and `vit-imagenet-verified` (§2p).
Unlike those two, this one needed a renderer change first: `nClasses` was a hardcoded literal and
`-α/K` was a caller-supplied string independent of it — the two-writers-for-one-fact shape that
produced R34's K=10 gradient bug. Both are fixed; `cBS` is still private, so this renders at
batch 32 (global 128 on four replicas).

XLA-only by construction: collectives live on the PJRT path.

⚠ Does NOT move the verification tier, and is NOT the ConvNeXt paper recipe — see
`the net's Main file`'s claim-ceiling note before quoting anything from it.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 300 epochs at 32 per device — the ConvNeXt paper's schedule length. `batchSize` is PER DEVICE
    and must match the batch the selected variant was rendered at (32; it is baked into the graph,
    so a mismatch is a shape error at the first invoke rather than a silent limp). -/
def convnextImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 32

/-- Entry point. Defaults to the single-device `adam` variant rather than `adamdp`, matching the
    R34 and ViT ImageNet drivers: a DP default makes a plain invocation fail at the first step with
    a replica-count refusal, which reads as a broken build rather than a missing flag. -/
def runConvNeXtImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD convnextImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.00025   -- `convNeXtTinyImagenetConfig.learningRate`: 4e-3@bs4096 scaled to bs256.
                          -- ⚠ this run is at global 128, so the linear-scaling rule would put it
                          -- near 1.25e-4; 2.5e-4 is kept to match the reference knob and left as
                          -- the thing to tune first if it under- or over-steps.
  convnextImagenetVerified.toNet.trainAdamSched
    { convnextImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 20 variant
    -- warmup 20 epochs, not 5: `convNeXtTinyImagenetConfig.warmupEpochs := 20` is a ConvNeXt-paper
    -- value and differs from every other net in this repo.

def main (argv : List String) : IO Unit := runConvNeXtImagenet argv
