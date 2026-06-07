import LeanMlir.VerifiedNets

/-! # `cifar-bn-verified` — train the CIFAR-10 CNN **with BatchNorm** on the
VERIFIED-rendered codegen

Chapter 5, BatchNorm variant: `conv 3→32 → BN → relu → conv 32→32 → BN → relu →
maxpool → conv 32→64 → BN → relu → conv 64→64 → BN → relu → maxpool → flatten →
dense 4096→512 → relu → dense 512→512 → relu → dense 512→10` + softmax-CE. The BN is the
proven per-example normalization (`bnForward`: reduce μ/var over the whole oc·H·W feature
vec, **scalar** γ/β), inserted after each conv. Trains on
`verified_mlir/cifar_bn_train_step.mlir` (`Proofs.StableHLO.cifarBnTrainStepText`); the
whole-network VJP is `cifarCnnBn_has_vjp_at` — audited 3-axiom-clean.

The BN here is SCALAR-global (one γ/β over c·h·w), the same proven `bnForward` that ViT's
LayerNorm proof witness reduces to — NOT the per-channel `[c]` form. The model is the
`cifarBnVerified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`); its math VJP is tied in
`LeanMlir/Proofs/SpecVJP.lean`. Trains through `VerifiedNet.train` (`mlpTrainStepV`,
He-init for conv/dense, γ=1/β=0 for BN).

Companion to `cifar-verified` (no-BN). Run both to compare.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar-bn-verified data`
-/

def cifarBnConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifarBnVerified.train cifarBnConfig (argv.head?.getD "data")
