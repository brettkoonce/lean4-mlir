import LeanMlir.VerifiedNets

/-! # `mnist-cnn-pgd` — phase-3 PGD attack on the verified MNIST CNN

The first **conv rung** of the robustness ladder (`planning/robustness_ladder.md`). Trains the
`conv 1→32 → relu → conv 32→32 → relu → maxpool → flatten → 6272→512 → relu → 512→512 → relu →
512→10` net on the proof-rendered SGD step, then runs an L∞ / L2 PGD attack through the real IREE
pipeline. Each step's input gradient is the **full proven backward** run to `dx`: the conv
input-VJPs (transpose-`o,i` + spatial-`reverse` kernel) and the maxpool `select_and_scatter`-back,
mirroring `verified_mlir/cnn_train_step.mlir` — plus the final conv1 input-VJP the train step omits.

The Lipschitz certificate is the conv-aware spectral-norm **product** (`specNormConvTapSum` for the
convs × `specNormW` for the denses; ReLU/maxpool are 1-Lipschitz). Over ~5 layers it is even looser
than the MLP's three-layer product — the linear-tight → MLP-vacuous → CNN-more-vacuous depth-cliff.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-pgd data`
-/

def cnnPgdConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit := do
  -- CNN_PGD_EPOCHS overrides the epoch count (cheap smoke test); absent → full 10.
  let ep := ((← IO.getEnv "CNN_PGD_EPOCHS").bind (·.toNat?)).getD cnnPgdConfig.epochs
  cnnVerified.attackPgdCnn { cnnPgdConfig with epochs := ep } (argv.head?.getD "data")
