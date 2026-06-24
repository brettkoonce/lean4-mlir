import LeanMlir.VerifiedNets

/-! # `cifar8-bn-verified-momentum` — 8-conv CIFAR-10 CNN **+ per-channel BN**, Nesterov-momentum SGD

The BN momentum peer (`cifar8BnVerified`, 38 params incl. 8× BN γ/β). Same proof-rendered
fwd/bwd/grad body as `cifar8-bn-verified{,-adam}`, with the Nesterov-momentum update
(`v ← μ·v + ∇; θ ← θ − lr·(μ·v + ∇)`, μ=0.9) from
`verified_mlir/cifar8_bn_mom_train_step.mlir`. Driven by `trainAdamSched` `variant := "mom"`.
Per-channel BN is per-example ⇒ train=eval (no running stats); eval via `@cifar8_bn_fwd`.

baseLR 0.02 (peak), μ 0.9, 3-epoch warmup + cosine decay, no weight decay.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-bn-verified-momentum data`
-/

def cifar8BnMomConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8BnVerified.toNet.trainAdamSched cifar8BnMomConfig (argv.head?.getD "data") 0.02 0.9 0.999 3 "mom"
