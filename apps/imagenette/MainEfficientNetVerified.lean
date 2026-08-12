import LeanMlir.VerifiedNets

/-! # `efficientnet-verified` — train a small EfficientNet on the VERIFIED-rendered codegen

Chapter 8: faithful EfficientNet-B0 (Tan & Le 2019) — all-swish + squeeze-excite + BATCH
norm — on IMAGENETTE 3×224×224 (native B0 resolution):

  stem 3×3-s2 conv (3→32) → BN → swish → 16 MBConv layers `[t,c,n,s,k]`
  (expand 1×1 [skip t=1] → BN → swish, depthwise k×k → BN → swish, squeeze-excite,
  project 1×1 → BN; + residual iff s=1 ∧ ic=oc) → head 1×1 conv (320→1280) → BN → swish →
  GAP → dense 1280→10 + softmax-CE.

The model is `efficientnetVerified` (in `LeanMlir.VerifiedNets`); its derived 213-tensor
layout is kernel-`#guard`ed against the audited `EfficientNetLayout`. Trains on
`verified_mlir/efficientnet_{train_step,fwd}.mlir` (rendered by tests/TestEfficientNet*)
through the packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, batch-norm, He-init).
Each op fragment is a proven-faithful emitter (swish/sigmoid/SE/depthwise k×k/batch-norm);
the whole-net VJP `efficientnet_has_vjp` is a representative witness (full B/C deferred).

Run (GPU): `.lake/build/bin/efficientnet-verified data`

⚠⚠ **THIS DRIVER CANNOT PRODUCE A MEANINGFUL ACCURACY ON THIS NET.** Measured
2026-08-12 on XLA/CUDA: `387/3925 = 9.859873%` on every epoch, byte identical, which is
chance on Imagenette's ten classes. It is a constant predictor, not a slow
curve.

`efficientnetVerified` carries BatchNorm, and **running-statistic threading lives only in
`VerifiedNet.trainAdamSched`**, not in `VerifiedNet.train`. This driver trains
parameters but never accumulates BN running stats, then evaluates through
`@efficientnet_fwd` (which needs them) rather than `@efficientnet_fwd_eval`.

▶ For a real number use `efficientnet-verified-adam`. ▶ This binary is still a useful
structural smoke test (compile, train-step arity, packed-parameter round trip).
Do not quote its accuracy. The same defect affects `resnet34-verified` and is
recorded there too; fixing it means teaching `.train` the BN threading
`trainAdamSched` already has.
-/

def efficientnetConfig : VerifiedConfig where
  epochs    := 20
  batchSize := 32

def main (argv : List String) : IO Unit :=
  efficientnetVerified.train efficientnetConfig (argv.head?.getD "data")
