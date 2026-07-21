import LeanMlir.VerifiedNets

/-! # `mnist-linear-verified` — train MNIST on the VERIFIED-rendered codegen

Trains the Chapter-2 linear classifier on the StableHLO that the **verified
renderer** emits — `verified_mlir/linear_train_step.mlir`, which is
`Proofs.StableHLO.linearTrainStepModuleV` = `pretty (emit g)`, the text whose
denotation is machine-proven equal to the Mathlib `fderiv` math
(`LeanMlir/Proofs/StableHLO.lean`, audited 3-axiom-clean). The forward, softmax-
CE cotangent, parameter gradients, and SGD update are all the proof-backed ops.

This is the *real* path: the Lean loop → `IreeRuntime` FFI (`libiree_ffi.so`) →
in-process IREE → GPU — not the Python/CLI stand-in. The training step runs via
`IreeSession.linearTrainStepV` (the verified module's signature); eval runs the
verified `@linear_fwd` via `IreeSession.forwardF32`.

The model is expressed as a `VerifiedNetSpec` (a single dense layer) — the same
readable layer list whose **math VJP is proven** in `LeanMlir/Proofs/Foundation/SpecVJP.lean`
(`linearVerified_has_vjp`). Unlike the other verified trainers, the linear model
keeps a bespoke `main`: its train step uses the 2-argument `linearTrainStepV` FFI
(separate `W0`/`b0`, zero-init) rather than the packed-params `mlpTrainStepV` the
shared `VerifiedNet.train` driver expects. Every dimension is read from the spec.

Regenerate the `verified_mlir/*.mlir` with
`lake env lean LeanMlir/Proofs/StableHLO.lean`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-verified data`
-/

/-- The model `linearVerified` (a single dense 784→10) lives in `LeanMlir.VerifiedNets`
    so this trainer and the `Proofs.SpecVJP` VJP theorem share one object. -/
def linearConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

-- The train loop lives in the driver (`VerifiedNet.trainLinear`); linear uses the
-- 2-argument `linearTrainStepV` FFI rather than the packed-params path the other nets use.
def main (argv : List String) : IO Unit :=
  linearVerified.trainLinear linearConfig (argv.head?.getD "data")
