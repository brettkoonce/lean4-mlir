import LeanMlir.VerifiedNets

/-! # `cifar-pgd` — phase-3 PGD attack on the verified CIFAR-10 CNN

The deeper conv rung of the robustness ladder (`planning/robustness_ladder.md`). Trains the
verified `conv 3→32 → conv 32→32 → pool → conv 32→64 → conv 64→64 → pool → 4096→512→512→10`
net on the proof-rendered SGD step, then runs L∞/L2 PGD through IREE with `genCifarPgdStep` —
the full proven input-VJP to `dx` (4 conv input-VJPs + 2 maxpool `select_and_scatter`-backs +
the final conv1 VJP the train step omits), mirroring `verified_mlir/cifar_train_step.mlir`.

The conv-aware Lipschitz certificate is a **7-layer** product (4 conv tap-sums × 3 dense spectral
norms) — even more astronomically vacuous than the 5-layer MNIST CNN. The depth-cliff, one rung
deeper. Reuses the generic `attackPgdConvNet` driver.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/cifar-pgd data`
-/

def cifarPgdConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit := do
  -- CIFAR_PGD_EPOCHS overrides the epoch count (cheap smoke test); absent → full 12.
  let ep := ((← IO.getEnv "CIFAR_PGD_EPOCHS").bind (·.toNat?)).getD cifarPgdConfig.epochs
  cifarVerified.attackPgdCifar { cifarPgdConfig with epochs := ep } (argv.head?.getD "data")
