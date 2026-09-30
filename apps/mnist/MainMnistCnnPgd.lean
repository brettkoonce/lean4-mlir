import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Attack

/-! # `mnist-cnn-pgd` — PGD attack on the verified MNIST CNN

The first **conv rung** of the robustness ladder. Trains the
`conv 1→32 → relu → conv 32→32 → relu → maxpool → flatten → 6272→512 → relu → 512→512 → relu →
512→10` net on the proof-rendered SGD step, then runs an L∞ / L2 PGD attack on the GPU. Each step's input gradient is the **full proven backward** run to `dx`: the conv
input-VJPs (transpose-`o,i` + spatial-`reverse` kernel) and the maxpool `select_and_scatter`-back,
mirroring `verified_mlir/cnn_train_step.mlir` — plus the final conv1 input-VJP the train step omits.

The Lipschitz certificate is the conv-aware **product** of per-layer upper bounds (`convLip`, a
tap-sum of Schatten-8 bounds, for the convs × `denseLip` for the denses; ReLU and the disjoint
2×2 max-pool are 1-Lipschitz). Over ~5 layers it is even looser
than the MLP's three-layer product — the linear-tight → MLP-vacuous → CNN-more-vacuous depth-cliff.

Run (GPU): `.lake/build/bin/mnist-cnn-pgd data`
-/

def cnnPgdConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit := do
  -- CNN_PGD_EPOCHS overrides the epoch count (cheap smoke test); absent → full 10.
  let ep := ((← IO.getEnv "CNN_PGD_EPOCHS").bind (·.toNat?)).getD cnnPgdConfig.epochs
  cnnVerified.toNet.attackPgdCnn { cnnPgdConfig with epochs := ep } (argv.head?.getD "data")
