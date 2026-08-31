import LeanMlir.VerifiedNets

/-! # `cifar-verified` — train the CIFAR-10 CNN on the VERIFIED-rendered codegen

Chapter 5 (no BatchNorm): `conv 3→32 → relu → conv 32→32 → relu → maxpool 32→16
→ conv 32→64 → relu → conv 64→64 → relu → maxpool 16→8 → flatten 4096 →
dense 4096→512 → relu → dense 512→512 → relu → dense 512→10` + softmax-CE. Trains on
`verified_mlir/cifar_train_step.mlir` (`Proofs.StableHLO.cifarTrainStepText`), whose ops
are each proven faithful to the Mathlib `fderiv` math; the whole-network VJP is
`cifarCnn_has_vjp_at` — audited 3-axiom-clean.

The model is the `cifarVerified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`), the same
layer list whose math VJP is tied in `LeanMlir/Proofs/Foundation/SpecVJP.lean`. Trains through the
packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, He-init, CIFAR `.bin` loader).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar-verified data`
-/

def cifarConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifarVerified.train cifarConfig (argv.head?.getD "data")
