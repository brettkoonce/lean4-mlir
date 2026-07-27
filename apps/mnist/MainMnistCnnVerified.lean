import apps.mnist.CnnVerifiedCommon

/-! # `mnist-cnn-verified` — train the MNIST CNN on the VERIFIED-rendered codegen

Chapter 4: `conv 1→32 → relu → conv 32→32 → relu → maxpool 28→14 →
flatten → dense 6272→512 → relu → dense 512→512 → relu → dense 512→10` +
softmax-CE. Trains on `verified_mlir/cnn_train_step.mlir`
(`Proofs.StableHLO.cnnTrainStepText`), whose forward/backward/grad ops are each
proven faithful to the Mathlib `fderiv` math (`cnnFwdGraph_faithful`,
`convBack_faithful`, `maxPoolBack_faithful`, `reluF_faithful`,
`selectPos_faithful`, `wGrad/bGrad_is*Jacobian`, `lossCotGraph_isCEgrad`) —
audited 3-axiom-clean. The conv weight grad is the transpose-trick render.

The model is the `cnnVerified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`) — the same
readable layer list whose **math VJP is proven** in `LeanMlir/Proofs/Foundation/SpecVJP.lean`
(`cnnVerified_has_vjp_at`, folded through conv→relu→conv→relu→maxpool→dense→…). It trains
through the packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, He-init, 4-D kernels).

Real path: Lean loop → IreeRuntime FFI → in-process IREE → GPU. The body is shared
with the XLA build (`mnist-cnn-verified-xla`); see `apps/mnist/CnnVerifiedCommon.lean`
and `planning/xla_pjrt_ladder.md`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-verified data`
-/

def main (argv : List String) : IO Unit := runCnnVerified argv
