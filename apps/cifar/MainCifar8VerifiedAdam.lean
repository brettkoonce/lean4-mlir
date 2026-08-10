import LeanMlir.VerifiedNets

/-! # `cifar8-verified-adam` — the deeper 8-conv CIFAR-10 CNN (no BN) trained with **AdamW**

The Adam peer of `cifar8-verified` (SGD). Same proof-rendered forward + backward + param
gradients (`cifar8Verified`, whole-net VJP `Proofs.cifarCnn8_has_vjp_at`), with the SGD
update swapped for AdamW via `ViTRender.emitAdamV` and driven by the generic
`VerifiedNet.trainAdamSched`: `[θ|m|v]` (22 params) packed as one blob + runtime
`lr`/`bc₁`/`bc₂` (cosine + warmup + per-step bias correction). Trains on
`verified_mlir/cifar8_adam_train_step.mlir`, rendered as `pretty(provenGraph)` by
`LeanMlir/Proofs/Codegen/CnnRender.lean` (§2a — it was `tests/`-written until 2026-07-27).

The optimizer is the *only* difference vs `cifar8-verified`: identical net, identical
gradient, plain softmax-CE (no label smoothing), mean cotangent. AdamW lr 1e-3, β₁ .9,
β₂ .999, wd 1e-4 (baked), 3-epoch warmup + cosine decay. Part of the BN/noBN × SGD/Adam
ablation.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-verified-adam data`

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
The backend is a run-time choice about transport, not a different program, which
is what the G2 gate asserts.
-/

def cifar8AdamConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear
    warmup then cosine decay. -/
def runCifar8Adam (argv : List String) : IO Unit :=
  cifar8Verified.toNet.trainAdamSched cifar8AdamConfig (argv.head?.getD "data") 0.001 0.9 0.999 3

def main (argv : List String) : IO Unit := runCifar8Adam argv
