import LeanMlir.VerifiedNets

/-! # `mnist-mlp-smooth` — randomized-smoothing certificate on the verified MNIST MLP

The MLP rung of the depth-independent certificate (`planning/robustness_ladder.md` §3,
Cohen–Rosenfeld–Kolter 2019). Same forward-only Monte-Carlo procedure as `mnist-cnn-smooth`,
on the 784→512→512→10 MLP: noise-augmented training, then certify radius `σ·Φ⁻¹(p_A)` with
`p_A` a Clopper–Pearson lower bound on the top class's noise probability.

Where the MLP's three-layer spectral-norm product gave a *vacuous* cert (L = 39, 0% certified),
randomized smoothing certifies a non-vacuous radius on the same net — the contrast that motivates
the smoothing rung.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-smooth data`
-/

def mlpSmoothConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let ep := ((← IO.getEnv "SMOOTH_EPOCHS").bind (·.toNat?)).getD mlpSmoothConfig.epochs
  mlpVerified.smoothCertify { mlpSmoothConfig with epochs := ep } (argv.head?.getD "data") [0.12, 0.25, 0.5]
