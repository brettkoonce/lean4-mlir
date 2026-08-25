import LeanMlir.VerifiedNets

/-! # `cifar8b-verified-adam` — the BATCHED-render gate

Trains the identical net, identical hyperparameters and identical init as
`cifar8-verified-adam`, but on `verified_mlir/cifar8b_adam_train_step.mlir` — emitted by
`cifar8AdamTrainStepFaithfulB`, which is on ImageNet's batched op family rather than the
per-example one.

⭐ This is a GATE, not a new experiment. The two renders denote the same function
(`StableHLO.lean` l.2016 vs l.2200 are the same proven VJP; conv is linear so the primal
argument the batched op drops is free), so their f32 training curves must agree to within
run-to-run noise. A divergence is a bug in the migration, and catching it here is far
cheaper than discovering it inside a bf16 number.

Run: `LEAN_MLIR_LOWERER=xla .lake/build/bin/cifar8b-verified-adam data`
-/

def cifar8bAdamConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8bVerified.toNet.trainAdamSched cifar8bAdamConfig (argv.head?.getD "data") 0.001 0.9 0.999 3
