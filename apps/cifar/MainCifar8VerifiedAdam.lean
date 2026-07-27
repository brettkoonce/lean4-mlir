import apps.cifar.Cifar8AdamCommon

/-! # `cifar8-verified-adam` — the deeper 8-conv CIFAR-10 CNN (no BN) trained with **AdamW**

The Adam peer of `cifar8-verified` (SGD). Same proof-rendered forward + backward + param
gradients (`cifar8Verified`, whole-net VJP `Proofs.cifarCnn8_has_vjp_at`), with the SGD
update swapped for AdamW via `ViTRender.emitAdamV` and driven by the generic
`VerifiedNet.trainAdamSched`: `[θ|m|v]` (22 params) packed as one blob + runtime
`lr`/`bc₁`/`bc₂` (cosine + warmup + per-step bias correction). Trains on
`verified_mlir/cifar8_adam_train_step.mlir` (rendered by `tests/TestCifar8AdamTrain.lean`).

The optimizer is the *only* difference vs `cifar8-verified`: identical net, identical
gradient, plain softmax-CE (no label smoothing), mean cotangent. AdamW lr 1e-3, β₁ .9,
β₂ .999, wd 1e-4 (baked), 3-epoch warmup + cosine decay. Part of the BN/noBN × SGD/Adam
ablation.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-verified-adam data`
-/

def main (argv : List String) : IO Unit := runCifar8Adam argv
