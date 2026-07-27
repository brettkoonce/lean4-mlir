import apps.mnist.MlpVerifiedCommon

/-! # `mnist-mlp-verified` — train the MNIST MLP on the VERIFIED-rendered codegen

Chapter 3: `dense 784→512 → relu → dense 512→512 → relu → dense 512→10` +
softmax-CE. Trains on `verified_mlir/mlp_train_step.mlir`
(`Proofs.StableHLO.mlpTrainStepText`), whose forward/backward/grad ops are each
proven faithful to the Mathlib `fderiv` math (`mlpFwdGraph_faithful`,
`mlpBackGraph_faithful`, `reluF_faithful`, `selectPos_faithful`,
`wGrad/bGrad_is*Jacobian`, `lossCotGraph_isCEgrad`) — audited 3-axiom-clean.

The model is the `mlpVerified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`) — the same
readable layer list whose **math VJP is proven** in `LeanMlir/Proofs/Foundation/SpecVJP.lean`
(`mlpVerified_has_vjp` / `mlpVerified_has_vjp_at`, the latter folded from `vjp_comp_at`).
It trains through the packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, He-init).

Real path: Lean loop → IreeRuntime FFI → in-process IREE → GPU. The body is shared
with the XLA build (`mnist-mlp-verified-xla`); see `apps/mnist/MlpVerifiedCommon.lean`
and `planning/xla_pjrt_ladder.md`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-verified data`
-/

def main (argv : List String) : IO Unit := runMlpVerified argv
