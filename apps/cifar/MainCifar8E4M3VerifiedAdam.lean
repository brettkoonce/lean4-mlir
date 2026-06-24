import LeanMlir.VerifiedNets

/-! # `cifar8-e4m3-verified-adam` — fp8 (E4M3) CIFAR-8 CNN, AdamW

fp8 sibling of `cifar8-verified-adam`. Same verified `cifar8_adam_train_step.mlir`
(AdamW, cosine+warmup), weights+input projected onto the E4M3 grid, fp32 accumulate,
fp32 master `[θ|m|v]` (`VerifiedNet.trainAdamSchedE4M3`, variant "adam"). Honors
LEAN_MLIR_MAX_EPOCHS for a capped comparison run.
Run: `IREE_BACKEND=rocm .lake/build/bin/cifar8-e4m3-verified-adam data` -/

def cifar8AdamE4M3Config : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8Verified.toNet.trainAdamSchedE4M3 cifar8AdamE4M3Config (argv.head?.getD "data") 0.001 0.9 0.999 3
