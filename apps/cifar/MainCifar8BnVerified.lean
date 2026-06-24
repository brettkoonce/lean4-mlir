import LeanMlir.VerifiedNets

/-! # `cifar8-bn-verified` — train the deeper 8-conv CIFAR-10 CNN **with BatchNorm**

The pedagogical BN-acceleration demo (BN half): four `conv→BN→relu, conv→BN→relu, pool`
stages, channels `[16,16,32,32]`, 32→16→8→4→2 spatial, then the reused 3-dense head
(flatten 128 → 64 → relu → 64 → relu → 10) + softmax-CE. The BN is the proven per-example
**per-channel** normalization (`bnPerChannelTensor3`, `m=h·w`) inserted after each of the
8 convs (γ=1/β=0 init, before relu). Per-channel BN is per-example ⇒ train=eval (no running
stats). Trains on `verified_mlir/cifar8_bn_train_step.mlir`
(`Proofs.StableHLO.cifar8BnTrainStepText`); the whole-network VJP is
`Proofs.cifarCnnBn8_has_vjp_at` (folds `convBnReluPC_has_vjp_at`) — audited 3-axiom-clean.

The model is the `cifar8BnVerified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`); trains
through `VerifiedNet.train` (He-init for conv/dense, γ=1/β=0 `[c]` for BN).

Companion to `cifar8-verified` (no BN). Run both to compare BN's acceleration.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-bn-verified data`
-/

def cifar8BnConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8BnVerified.train cifar8BnConfig (argv.head?.getD "data")
