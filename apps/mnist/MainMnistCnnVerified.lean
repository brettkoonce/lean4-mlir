import LeanMlir.VerifiedNets

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
The spec stays in `VerifiedNets` on purpose: the trainer and the theorem must name
the *same* object, or the proof would be about a different network than the one
that runs.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects — XLA/PJRT by default, IREE with
`=iree` — resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file; the backend is a run-time choice about *transport*,
not a different program. This is the first CONVOLUTIONAL rung, and the one where
that distinction is most visible: it is `transportSensitive` in the benchmark
table (84.6% of wall clock is parameter round-trip, §2d.3), so the two lowerers
differ here far more in how they move bytes than in what they compute.

Run:
```
lake build mnist-cnn-verified
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-cnn-verified data          # XLA
LEAN_MLIR_LOWERER=iree IREE_BACKEND=rocm ... mnist-cnn-verified data   # IREE
```
See `planning/xla_pjrt_ladder.md` (the conv rung, G2).
-/

/-- 10 epochs at batch 128. `lr` is display-only — the real rate is baked into
    `verified_mlir/cnn_train_step.mlir`, so both lowerers necessarily train the
    same recipe and cannot drift. He-init is random, so runs are NOT bit-reproducible
    (unlike the zero-init linear net); the tie tests, not a trajectory match, are
    what establish the two backends agree. -/
def cnnConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cnnVerified.train cnnConfig (argv.head?.getD "data")
