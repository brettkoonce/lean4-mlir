import LeanMlir.VerifiedNets

/-! # `mnist-linear-verified` — train MNIST on the VERIFIED-rendered codegen

Trains the Chapter-1 linear classifier on the StableHLO that the **verified
renderer** emits — `verified_mlir/linear_train_step.mlir`, which is
`Proofs.StableHLO.linearTrainStepModuleV` = `pretty (emit g)`, the text whose
denotation is machine-proven equal to the Mathlib `fderiv` math
(`LeanMlir/Proofs/Codegen/StableHLO.lean`, audited 3-axiom-clean). The forward,
softmax-CE cotangent, parameter gradients, and SGD update are all the
proof-backed ops.

**One file, one binary, either lowerer.** The proven graph is handed to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects — XLA/PJRT by default, IREE with
`=iree` — resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: this is the whole program, and it is short enough
to print. That is the point. The backend is a run-time choice about *transport*,
not a different program, which is exactly what the G2 gate asserts.

The model is expressed as a `VerifiedNetSpec` (a single dense layer) — the same
readable layer list whose **math VJP is proven** in
`LeanMlir/Proofs/Foundation/SpecVJP.lean` (`linearVerified_has_vjp`). The spec
itself lives in `LeanMlir.VerifiedNets` rather than here, deliberately: the
trainer and the theorem must name the *same* object, or the proof would be about
a different network than the one that runs.

Unlike the other verified trainers, the linear model keeps a bespoke entry point:
its train step uses the 2-argument `linearTrainStepV` FFI (separate `W0`/`b0`,
zero-init) rather than the packed-params `mlpTrainStepV` the shared
`VerifiedNet.train` driver expects. Every dimension is read from the spec.

Regenerate the `verified_mlir/*.mlir` with
`lake env lean LeanMlir/Proofs/Codegen/StableHLO.lean`.

Run:
```
lake build mnist-linear-verified
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-linear-verified data          # XLA
LEAN_MLIR_LOWERER=iree IREE_BACKEND=rocm ... mnist-linear-verified data   # IREE
```
Both produce an identical 12-epoch trajectory (final 9210/10000); XLA is ~2.3×
faster per epoch on a 7900 XTX. See `planning/xla_pjrt_ladder.md` (rung 0, G2).
-/

/-- 12 epochs at batch 128. `lr` is display-only — the real rate is baked into
    `verified_mlir/linear_train_step.mlir`, so both lowerers necessarily train
    the same recipe and cannot drift. -/
def linearConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  linearVerified.trainLinear linearConfig (argv.head?.getD "data")
