import LeanMlir.VerifiedNets

/-! # `cifar8-verified-momentum` — 8-conv CIFAR-10 CNN (no BN), **Nesterov-momentum SGD**

The momentum peer of `cifar8-verified` (plain SGD) and `cifar8-verified-adam`. Same
proof-rendered forward + backward + param gradients (`cifar8Verified`), with the update
`v ← μ·v + ∇; θ ← θ − lr·(μ·v + ∇)` (μ=0.9, Nesterov) rendered by `emitMomentum` in
`tests/TestCifar8AdamTrain.lean → verified_mlir/cifar8_mom_train_step.mlir`. Driven by the
shared `VerifiedNet.trainAdamSched` with `variant := "mom"` (packed `[θ|m|v]`; the momentum
step ignores the m/bc slots, reads only the runtime lr + velocity v), so it gets the same
cosine + warmup lr schedule as the Adam path — a like-for-like "modern SGD" comparison.

baseLR 0.02 (peak), μ 0.9, 3-epoch warmup + cosine decay, no weight decay.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-verified-momentum data`
-/

def cifar8MomConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  -- baseLR 0.02, μ baked 0.9 (β args unused by the momentum step), 3-epoch warmup + cosine.
  cifar8Verified.toNet.trainAdamSched cifar8MomConfig (argv.head?.getD "data") 0.02 0.9 0.999 3 "mom"
