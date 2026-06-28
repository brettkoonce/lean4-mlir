import LeanMlir.VerifiedNets

/-! # `cifar-spectral` — spectral-norm-constrained CIFAR-10 CNN training

The CIFAR rung of the gap-shrinking lever (`planning/robustness_ladder.md`). Trains the verified
CIFAR-10 CNN with **projected SGD onto the spectral ball** — every few proof-rendered steps each
weight is rescaled so the dense `‖Wᵢ‖₂` and the conv tap-sum bound stay `≤ c` — then runs the
`cert ≤ TRUE ≤ PGD` sandwich (`genCifarPgdStep` attack, conv-aware product cert) across a cap sweep.

Hardest yet: a **7-layer** product (`L ≤ c⁷`) over 4 convs (loose tap-sum bound) + 3 denses, so the
caps must be tighter than the MNIST CNN's and certify at smaller radii still. The depth-cliff from
the training side, one rung deeper. Reuses the generic `attackPgdSpectralConvNet` driver.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/cifar-spectral data`
-/

def cifarSpectralConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

/-- Caps to sweep: `1e9` = unconstrained baseline, then progressively tighter spectral balls.
    Spans the trainable→dead transition: a 7-layer net with the *loose* conv tap-sum bound
    over-shrinks the 4 convolutions, so tight caps (≲2) kill training entirely. -/
def caps : List Float := [1.0e9, 6.0, 4.0, 3.0, 2.0]

def main (argv : List String) : IO Unit := do
  -- SPECTRAL_EPOCHS overrides the epoch count (cheap smoke test); absent → full 12.
  let ep := ((← IO.getEnv "SPECTRAL_EPOCHS").bind (·.toNat?)).getD cifarSpectralConfig.epochs
  cifarVerified.attackPgdSpectralCifar { cifarSpectralConfig with epochs := ep } (argv.head?.getD "data") caps
