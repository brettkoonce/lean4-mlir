import LeanMlir.VerifiedNets

/-! # `cifar-bn-verified` — train the CIFAR-10 CNN **with BatchNorm** on the
VERIFIED-rendered codegen

Chapter 5, BatchNorm variant: `conv 3→32 → BN → relu → conv 32→32 → BN → relu →
maxpool → conv 32→64 → BN → relu → conv 64→64 → BN → relu → maxpool → flatten →
dense 4096→512 → relu → dense 512→512 → relu → dense 512→10` + softmax-CE. The BN is the
proven per-example **per-channel** normalization (`bnPerChannelTensor3`, `m=h·w`: reduce
μ/var over each channel's own h·w spatial cells, vector γ/β `[c]`), inserted after each
conv. Trains on `verified_mlir/cifar_bn_train_step.mlir`
(`Proofs.StableHLO.cifarBnTrainStepText`); the whole-network VJP is
`cifarCnnBn_has_vjp_at` (folds `convBnReluPC_has_vjp_at`) — audited 3-axiom-clean.

The BN here is PER-CHANNEL (per-example/instance, `γ/β : [c]`), the form that beats no-BN
(~71% vs ~67% @ lr=0.1) — NOT the earlier scalar-global `bnForward` that underperformed
(~58%). It is per-example ⇒ train=eval (no running stats). The model is the
`cifarBnVerified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`); its math VJP is tied in
`LeanMlir/Proofs/SpecVJP.lean`. Trains through `VerifiedNet.train` (`mlpTrainStepV`,
He-init for conv/dense, γ=1/β=0 `[c]` for BN).

Companion to `cifar-verified` (no-BN). Run both to compare.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar-bn-verified data`
-/

def cifarBnConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifarBnVerified.train cifarBnConfig (argv.head?.getD "data")
