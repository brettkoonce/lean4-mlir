import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Attack

/-! # `mnist-mlp-pgd` — PGD attack on the verified MNIST MLP

Trains the 784→512→512→10 ReLU MLP on the proof-rendered SGD step, then runs an L∞ and L2
PGD attack on the GPU. Each step's input gradient is the proven
`mlpInputGrad` VJP `dx = ((g·W₂ᵀ⊙relu')·W₁ᵀ⊙relu')·W₀ᵀ`, emitted as a StableHLO kernel.
The Lipschitz certificate is the **product** of per-layer upper bounds on `‖W₀‖·‖W₁‖·‖W₂‖` (the
Schatten-8 bound of `denseE_lipschitzL2_gram2`, `denseLip`) — where the bound goes loose, the
contrast with the single-layer linear certificate.

Run (GPU): `.lake/build/bin/mnist-mlp-pgd data`
-/

def mlpConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  mlpVerified.toNet.attackPgdMlp mlpConfig (argv.head?.getD "data")
