import apps.cifar.Cifar8MomCommon

/-! # `cifar8-verified-momentum` — 8-conv CIFAR-10 CNN (no BN), **Nesterov-momentum SGD**

The momentum peer of `cifar8-verified` (plain SGD) and `cifar8-verified-adam`. Same
proof-rendered forward + backward + param gradients (`cifar8Verified`), with the update
`v ← μ·v + ∇; θ ← θ − lr·(μ·v + ∇)` (μ=0.9, Nesterov) rendered by `emitMomentum` in
`verified_mlir/cifar8_mom_train_step.mlir` — since §2i that is `pretty(provenGraph)` from
`LeanMlir/Proofs/Codegen/CnnRender.lean` (the proven `momVNextF`/`momParamF` pair), not
`emitMomentum`. Driven by the
shared `VerifiedNet.trainAdamSched` with `variant := "mom"` (packed `[θ|m|v]`; the momentum
step ignores the m/bc slots, reads only the runtime lr + velocity v), so it gets the same
cosine + warmup lr schedule as the Adam path — a like-for-like "modern SGD" comparison.

baseLR 0.02 (peak), μ 0.9, 3-epoch warmup + cosine decay, no weight decay.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-verified-momentum data`
-/

def main (argv : List String) : IO Unit := runCifar8Mom argv
