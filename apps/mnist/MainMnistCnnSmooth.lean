import LeanMlir.VerifiedNets

/-! # `mnist-cnn-smooth` — randomized-smoothing certificate on the verified MNIST CNN

The **depth-independent** certificate (`planning/robustness_ladder.md` §3, Cohen–Rosenfeld–Kolter
2019) — the answer where the Lipschitz product is hopeless. The smoothed classifier
`ĝ(x) = argmax_c P[f(x+η)=c]`, `η ~ N(0,σ²I)`, is certified robust at L2 radius `σ·Φ⁻¹(p_A)`. It's
**forward-only**: no new kernel, no input-VJP — sample `n` noisy copies, run the proof-rendered
`cnn_fwd`, count argmax votes, Clopper–Pearson lower-bound `p_A`. The base CNN is trained with
matched Gaussian augmentation (every batch corrupted with `N(0,σ²I)` host-side before the
proof-rendered SGD step — the forward/backward graph is untouched), the Cohen recipe.

Unlike the conv-aware spectral-norm product (vacuous past the MLP), this certifies a *non-vacuous*
radius regardless of depth — the same procedure works identically on CIFAR and the deep nets.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-smooth data`
Smoke: `SMOOTH_EPOCHS=1 SMOOTH_MAXCERT=50 SMOOTH_N=200 ... mnist-cnn-smooth data`
-/

def cnnSmoothConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let ep := ((← IO.getEnv "SMOOTH_EPOCHS").bind (·.toNat?)).getD cnnSmoothConfig.epochs
  cnnVerified.smoothCertify { cnnSmoothConfig with epochs := ep } (argv.head?.getD "data") [0.12, 0.25, 0.5]
