import LeanMlir.VerifiedNets

/-! # Shared body of the CIFAR-8 BN + AdamW trainer

`cifar8-bn-verified-adam` (IREE) and `cifar8-bn-verified-adam-xla` (XLA/PJRT) are
the same program — same `VerifiedNetSpec`, same
`verified_mlir/cifar8_bn_adam_train_step.mlir`, same §1a ties, same schedule and
He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point
live here rather than being duplicated; drift in `epochs`, `batchSize`, the seed,
or any of the AdamW hyperparameters would quietly invalidate the G2 comparison
(`planning/xla_pjrt_ladder.md` §3).
-/

def cifar8BnAdamConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear
    warmup then cosine decay. -/
def runCifar8BnAdam (argv : List String) : IO Unit :=
  cifar8BnVerified.toNet.trainAdamSched cifar8BnAdamConfig (argv.head?.getD "data") 0.001 0.9 0.999 3
