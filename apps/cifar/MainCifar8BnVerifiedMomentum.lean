import LeanMlir.VerifiedNets

/-! # `cifar8-bn-verified-momentum` — 8-conv CIFAR-10 CNN **+ per-channel BN**, Nesterov-momentum SGD

The BN momentum peer (`cifar8BnVerified`, 38 params incl. 8× BN γ/β). Same proof-rendered
fwd/bwd/grad body as `cifar8-bn-verified{,-adam}`, with the Nesterov-momentum update
(`v ← μ·v + ∇; θ ← θ − lr·(μ·v + ∇)`, μ=0.9) from
`verified_mlir/cifar8_bn_mom_train_step.mlir`. Driven by `trainAdamSched` `variant := "mom"`.
Per-channel BN is per-example ⇒ train=eval (no running stats); eval via `@cifar8_bn_fwd`.

baseLR 0.02 (peak), μ 0.9, 3-epoch warmup + cosine decay, no weight decay.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-bn-verified-momentum data`

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
The backend is a run-time choice about transport, not a different program, which
is what the G2 gate asserts.
-/

def cifar8BnMomConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends.
    baseLR 0.02 (peak), μ 0.9, 3-epoch warmup + cosine, no weight decay. Per-channel BN is
    per-example ⇒ train = eval, so the eval forward is `@cifar8_bn_fwd` and there are no
    running stats to thread. -/
def runCifar8BnMom (argv : List String) : IO Unit :=
  cifar8BnVerified.toNet.trainAdamSched cifar8BnMomConfig (argv.head?.getD "data") 0.02 0.9 0.999 3 "mom"

def main (argv : List String) : IO Unit := runCifar8BnMom argv
