import LeanMlir.VerifiedNets

/-! # `cifar8-e4m3-verified` — fp8 (E4M3) CIFAR-8 CNN, plain SGD

fp8 sibling of `cifar8-verified` (plain SGD). Same verified `cifar8_train_step.mlir`,
weights (conv per-channel / dense per-column) + input projected onto the E4M3 grid,
fp32 accumulate, fp32 master (`VerifiedNet.trainE4M3`). fp8 weights+input, fp32
intermediates. Run: `IREE_BACKEND=rocm .lake/build/bin/cifar8-e4m3-verified data` -/

def cifar8E4M3Config : VerifiedConfig where
  epochs    := 20
  batchSize := 128

def main (argv : List String) : IO Unit :=
  (cifar8Verified.toNet).trainE4M3 cifar8E4M3Config (argv.head?.getD "data")
