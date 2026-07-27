import LeanMlir.VerifiedNets

/-! # Shared body of the CIFAR-8 (no BN) + AdamW trainer

The no-BN arm of the BN/noBN × SGD/Adam ablation, shared by the IREE and XLA
executables. Same driver, hyperparameters, and seed as
`apps/cifar/Cifar8BnAdamCommon.lean` — the *only* difference between the two nets
is per-channel BatchNorm, which is what makes the pair a controlled test of where
a backend divergence comes from (`planning/xla_pjrt_ladder.md` §8, rung 2).
-/

def cifar8AdamConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear
    warmup then cosine decay. -/
def runCifar8Adam (argv : List String) : IO Unit :=
  cifar8Verified.toNet.trainAdamSched cifar8AdamConfig (argv.head?.getD "data") 0.001 0.9 0.999 3
