import LeanMlir.VerifiedNets

/-! # `mnist-mlp-spectral` — spectral-norm-constrained training (the gap-shrinking lever)

The research lever of the robustness ladder (`planning/robustness_ladder.md`). Trains the
verified 784→512→512→10 MLP with **projected SGD onto the spectral ball** — after every few
proof-rendered steps each weight `Wᵢ` is rescaled to `‖Wᵢ‖₂ ≤ c` — then runs the
`cert ≤ TRUE ≤ PGD` sandwich at a sweep of caps `c` (plus an unconstrained baseline).

Shrinking `c` pulls the global Lipschitz `L = ∏‖Wᵢ‖₂` down (`L ≤ c³`), turning the
**vacuous** product certificate **non-vacuous** — the empirical face of
`lipschitz_margin_certified_radius` (`LeanMlir/Proofs/LipschitzCert.lean`: smaller `L` ⇒ larger
certified radius `m/(√2·L)`) — at the cost of clean accuracy. The verified cross-entropy
gradient stays in the proven kernel; the projection is host-side weight rescaling only.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-spectral data`
-/

def mlpSpectralConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

/-- Caps to sweep: `1e9` = unconstrained baseline, then progressively tighter spectral balls. -/
def caps : List Float := [1.0e9, 3.0, 2.0, 1.5, 1.0]

def main (argv : List String) : IO Unit := do
  -- SPECTRAL_EPOCHS overrides the epoch count (cheap smoke test); absent → full 12.
  let ep := ((← IO.getEnv "SPECTRAL_EPOCHS").bind (·.toNat?)).getD mlpSpectralConfig.epochs
  mlpVerified.attackPgdSpectralMlp { mlpSpectralConfig with epochs := ep } (argv.head?.getD "data") caps
