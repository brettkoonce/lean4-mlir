import LeanMlir.VerifiedNets

/-! # `mnist-cnn-spectral` — spectral-norm-constrained CNN training (the lever, conv net)

The CNN sibling of `mnist-mlp-spectral` (`planning/robustness_ladder.md`). Trains the verified
`conv→conv→pool→512→512→10` net with **projected SGD onto the spectral ball** — after every few
proof-rendered steps each weight is rescaled so the dense `‖Wᵢ‖₂` and the conv tap-sum bound stay
`≤ c` — then runs the `cert ≤ TRUE ≤ PGD` sandwich (`genCnnPgdStep` attack, conv-aware product cert)
across a cap sweep.

Harder than the MLP: the CNN cert is a **5-layer** product (`L ≤ c⁵`) and the conv tap-sum is a
*loose* operator-norm bound, so the projection over-penalizes the convolutions — certifying the
conv net needs a tighter `c` (and pays more clean accuracy) than the MLP did. The honest
"depth + loose conv-norm ⇒ harder to certify." The verified CE gradient stays in the proven kernel;
the projection is host-side weight rescaling only.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-spectral data`
-/

def cnnSpectralConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

/-- Caps to sweep: `1e9` = unconstrained baseline, then progressively tighter spectral balls.
    Tighter than the MLP's (the 5-layer product + loose conv tap-sum compound faster). -/
def caps : List Float := [1.0e9, 2.0, 1.5, 1.2, 1.0]

def main (argv : List String) : IO Unit := do
  -- SPECTRAL_EPOCHS overrides the epoch count (cheap smoke test); absent → full 10.
  let ep := ((← IO.getEnv "SPECTRAL_EPOCHS").bind (·.toNat?)).getD cnnSpectralConfig.epochs
  cnnVerified.attackPgdSpectralCnn { cnnSpectralConfig with epochs := ep } (argv.head?.getD "data") caps
