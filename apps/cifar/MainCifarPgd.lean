import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Attack

/-! # `cifar-pgd` — PGD attack on the verified CIFAR-10 CNN

The deeper conv rung of the robustness ladder. Trains the verified
`conv 3→32 → conv 32→32 → pool → conv 32→64 → conv 64→64 → pool → 4096→512→512→10` net on the
proof-rendered SGD step, then runs L∞/L2 PGD with `genCifarPgdStep` — the full proven
input-VJP to `dx` (4 conv input-VJPs + 2 maxpool `select_and_scatter`-backs + the final conv1 VJP
the train step omits), mirroring `verified_mlir/cifar_train_step.mlir`.

The conv-aware Lipschitz certificate is a **7-layer** product (4 conv tap-sums × 3 dense spectral
norms) — even more astronomically vacuous than the 5-layer MNIST CNN. The depth-cliff, one rung
deeper. Reuses the generic `attackPgdConvNet` driver.

Run (GPU): `.lake/build/bin/cifar-pgd data`
-/

def cifarPgdConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit := do
  -- CIFAR_PGD_EPOCHS overrides the epoch count (cheap smoke test); absent → full 12.
  let ep := ((← IO.getEnv "CIFAR_PGD_EPOCHS").bind (·.toNat?)).getD cifarPgdConfig.epochs
  cifarVerified.toNet.attackPgdCifar { cifarPgdConfig with epochs := ep } (argv.head?.getD "data")
