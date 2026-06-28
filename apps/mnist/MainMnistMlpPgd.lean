import LeanMlir.VerifiedNets

/-! # `mnist-mlp-pgd` — phase-3 PGD attack on the verified MNIST MLP

Trains the 784→512→512→10 ReLU MLP on the proof-rendered SGD step, then runs an L∞ and L2
PGD attack through the real IREE pipeline. Each step's input gradient is the proven
`mlpInputGrad` VJP `dx = ((g·W₂ᵀ⊙relu')·W₁ᵀ⊙relu')·W₀ᵀ`, emitted as a StableHLO kernel.
The Lipschitz certificate is the **product** `‖W₀‖·‖W₁‖·‖W₂‖` — where the bound goes loose,
the contrast with the (exact) single-layer linear certificate. See `planning/robustness.md`.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-pgd data`
-/

def mlpConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  mlpVerified.attackPgdMlp mlpConfig (argv.head?.getD "data")
