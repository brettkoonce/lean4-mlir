import LeanMlir.VerifiedNets

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
The spec stays in `VerifiedNets` on purpose: the trainer and the theorem must name
the *same* object, or the proof would be about a different network than the one
that runs.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects — XLA/PJRT by default, IREE with
`=iree` — resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file; the backend is a run-time choice about *transport*,
not a different program, which is what the G2 gate asserts. Rung 1 of the ladder
(depth + multiple parameter tensors) is now that one binary run twice.

Run:
```
lake build mnist-mlp-verified
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-mlp-verified data          # XLA
LEAN_MLIR_LOWERER=iree IREE_BACKEND=rocm ... mnist-mlp-verified data   # IREE
```
See `planning/xla_pjrt_ladder.md` (rung 1, G2).
-/

/-- 12 epochs at batch 128. `lr` is display-only — the real rate is baked into
    `verified_mlir/mlp_train_step.mlir`, so both lowerers necessarily train the
    same recipe and cannot drift. The He-init seed is shared for the same reason. -/
def mlpConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  mlpVerified.train mlpConfig (argv.head?.getD "data")
