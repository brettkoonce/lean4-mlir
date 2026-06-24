import LeanMlir.VerifiedNets

/-! # `cifar8-bn-verified-adam` — the 8-conv CIFAR-10 CNN **+ per-channel BN** with **AdamW**

The Adam peer of `cifar8-bn-verified` (SGD), and the BN half of the BN/noBN × SGD/Adam
ablation. Same proof-rendered forward + backward + param gradients as `cifar8BnVerified`
(whole-net VJP `Proofs.cifarCnnBn8_has_vjp_at`, 8× per-channel BN), with the SGD update
swapped for AdamW via `ViTRender.emitAdamV` and driven by `VerifiedNet.trainAdamSched`:
`[θ|m|v]` (38 params: 22 conv/dense + 16 BN γ/β) packed + runtime `lr`/`bc₁`/`bc₂`. Trains
on `verified_mlir/cifar8_bn_adam_train_step.mlir` (rendered by `tests/TestCifar8AdamTrain.lean`).

Per-channel BN is per-example ⇒ train=eval (no running stats), so `bnChannels` stays empty
and the γ/β are Adam-updated like any other param; eval is plain `@cifar8_bn_fwd`. AdamW
lr 1e-3, β₁ .9, β₂ .999, wd 1e-4 (baked), 3-epoch warmup + cosine decay.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-bn-verified-adam data`
-/

def cifar8BnAdamConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  -- baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine decay.
  cifar8BnVerified.toNet.trainAdamSched cifar8BnAdamConfig (argv.head?.getD "data") 0.001 0.9 0.999 3
