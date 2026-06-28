import LeanMlir.VerifiedNets

/-! # `cifar-smooth` — randomized-smoothing certificate on the verified CIFAR-10 CNN

The deep-net payoff (`planning/robustness_ladder.md` §3, Cohen–Rosenfeld–Kolter 2019): the
7-layer conv-aware spectral-norm product was astronomically loose (global L = 942K, cert 0% at
every radius). Randomized smoothing is **depth-independent** — the exact same forward-only
procedure that ran on the MLP/CNN certifies a *non-vacuous* L2 radius here, where the Lipschitz
product is hopeless. No new kernel: sample noisy copies, run the proof-rendered `cifar_fwd`,
Clopper–Pearson lower-bound `p_A`, report `σ·Φ⁻¹(p_A)`. Base CNN trained with matched Gaussian
augmentation (host-side noise before the proof-rendered SGD step).

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/cifar-smooth data`
-/

def cifarSmoothConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let ep := ((← IO.getEnv "SMOOTH_EPOCHS").bind (·.toNat?)).getD cifarSmoothConfig.epochs
  cifarVerified.smoothCertify { cifarSmoothConfig with epochs := ep } (argv.head?.getD "data") [0.25, 0.5]
