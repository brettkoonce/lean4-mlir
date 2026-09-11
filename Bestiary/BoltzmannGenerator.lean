import LeanMlir.Spec

/-! # Boltzmann generator — Bestiary entry

Noé, Olsson, Köhler & Wu, "Boltzmann generators: sampling equilibrium states
of many-body systems with deep learning", Science 365 (2019). The claim is
not an architecture. It is that a generative model trained on whatever a
simulation managed to produce can be *reweighted* to the exact Boltzmann
distribution exp(-U/kT) — because the model gives every sample its own
log-density, the importance weight w = exp(-U/kT) / p_θ(x) is a number, and
with it a sample set drawn once is a free-energy estimate at any temperature.
MCMC cannot do that: a chain below a barrier sits in one well for e^(ΔU/kT)
attempts per crossing.

**The original network is a normalising flow.** Noé et al. use RealNVP
(Dinh et al. 2017): a stack of affine coupling blocks, each of which splits
the coordinates in two, passes one half through untouched, and scales and
shifts the other half by two small dense nets (the "conditioners") of the
first half. Its log-density is a sum of the log-scales, exact and cheap, and
the inverse is the same blocks run backwards. The coupling split is the one
thing our `NetSpec` language does not have — it is a branch-and-merge, not a
layer — so the conditioner is shown below as its own dense net and the block
structure is prose, the way the DCGAN entry stands in for transposed convs.

**The flow-matching version is zero new primitives, and it is the one this
repo trains.** `demos/MainDiffusion2d.lean` with the `flow` flag: the
velocity field v_θ(x, t) is the 2-D toy demo's three-layer `.dense` MLP with
the sincos time channel, trained on the linear interpolant
x_t = (1-t)·x₀ + t·ε with target ε - x₀ (Lipman et al. 2022, "Flow
matching for generative modeling"; Liu, Gong & Liu 2022, "Rectified flow";
Albergo & Vanden-Eijnden 2022, "Stochastic interpolants"). Sampling is
Euler on dx/dt = v from t = 1 to 0, and the log-density comes from the
continuity equation — the Jacobian's log-determinant integrated beside the
state — which is what turns it into a Boltzmann generator. The Müller-Brown
demo of the diffusion section trains it, and every one of its numbers is
exact by quadrature.

## Variants

- `boltzmannVelocityNet` — the demo's velocity net, 18,178 params:
                           2 coordinates + 8 sincos time channels → 128 → 128 → 2.
- `realNvpConditioner`   — one RealNVP conditioner (the s or t net of one
                           coupling block) at the width Noé et al. use for
                           the small systems: d/2 = 33 (alanine dipeptide's
                           66 Cartesian coordinates) → 128 → 128 → 33.
- `tinyBoltzmann`        — the same velocity net at width 32, small enough to
                           read in one pass.

The headline lesson is the one CLIP, NeRF and DDPM taught: the novelty is
in what the model is asked to produce — a density, not a picture — and in
what is done with it afterwards. The architecture is an MLP.
-/

-- ════════════════════════════════════════════════════════════════
-- § The flow-matching velocity net — the demo's network, verbatim
-- ════════════════════════════════════════════════════════════════
-- Input is the 2-vector plus 2·4 sincos channels of the time index
-- (`Ddpm.prependSinCosT` at H = W = 1). Output is the velocity ε - x₀,
-- a 2-vector, on the codegen's rank-2 DDPM MSE branch.

def boltzmannVelocityNet : NetSpec where
  name   := "Boltzmann generator (flow-matching velocity net, Müller-Brown)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 10 128 .relu,
    .dense 128 128 .relu,
    .dense 128 2 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § One RealNVP conditioner
-- ════════════════════════════════════════════════════════════════
-- A coupling block computes y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁) with x₁ passed
-- through; s and t are each a dense net of x₁. This is one of them. Noé et
-- al. stack several blocks with the halves swapped between them; the
-- log-density of the block is Σ s(x₁), summed over blocks.

def realNvpConditioner : NetSpec where
  name   := "RealNVP conditioner (one coupling half, alanine dipeptide width)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 33 128 .relu,
    .dense 128 128 .relu,
    .dense 128 33 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny fixture
-- ════════════════════════════════════════════════════════════════

def tinyBoltzmann : NetSpec where
  name   := "tiny-Boltzmann (velocity net, width 32)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 10 32 .relu,
    .dense 32 32 .relu,
    .dense 32 2 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Boltzmann generator"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  A generative model with an exact log-density, reweighted to"
  IO.println "  exp(-U/kT). The architecture is an MLP; the claim is physics."

  boltzmannVelocityNet.summarize (size := .omitted) (unit := .thousands)
  realNvpConditioner.summarize (size := .omitted) (unit := .thousands)
  tinyBoltzmann.summarize (size := .omitted) (unit := .thousands)

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives for the flow-matching version:"
  IO.println "    .dense / .relu and the sincos time channel of the DDPM path."
  IO.println "    ONE if the original RealNVP is shown: the affine coupling"
  IO.println "    split is a branch-and-merge the linear NetSpec cannot spell."
  IO.println "  • The exact log-density is what makes it a Boltzmann generator."
  IO.println "    RealNVP reads it off the coupling scales; the flow-matching"
  IO.println "    model integrates log|det(I + h ∂v/∂x)| beside the state."
  IO.println "  • Trained here: demos/MainDiffusion2d.lean with `flow` on the"
  IO.println "    Müller-Brown density, scored by quadrature, reweighted from"
  IO.println "    kT = 20 to 12 and 8 where a Langevin chain has not crossed."
