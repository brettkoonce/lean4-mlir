import LeanMlir.VerifiedNets

/-! # `cifar8-verified` — train the deeper 8-conv CIFAR-10 CNN (no BN) on VERIFIED codegen

The pedagogical BN-demo backbone (no-BN half): four `conv→conv→pool` stages, channels
`[16,16,32,32]`, 32→16→8→4→2 spatial, then the reused 3-dense head
(flatten 128 → 64 → relu → 64 → relu → 10) + softmax-CE. Trains on
`verified_mlir/cifar8_train_step.mlir` (`Proofs.StableHLO.cifar8TrainStepText`); the
whole-network VJP is `Proofs.cifarCnn8_has_vjp_at` — audited 3-axiom-clean.

The model is the `cifar8Verified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`). Trains
through the packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, He-init, CIFAR loader).

Companion to `cifar8-bn-verified` (with BN). Run both to compare BN's acceleration.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-verified data`
-/

def cifar8Config : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8Verified.train cifar8Config (argv.head?.getD "data")
