import LeanMlir.Spec

/-! # Physics-informed neural network (PINN) — Bestiary entry

Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks: a deep
learning framework for solving forward and inverse problems involving
nonlinear partial differential equations", J. Comput. Phys. 378 (2019);
the two 2017 preprints are arXiv:1711.10561 and 1711.10566.

**The network is a small MLP. The loss is the PDE.** A PINN is a dense net
u_θ(x, t) whose training loss has two terms: the squared error against
whatever boundary and initial data exist, and the squared residual of the
differential equation itself, evaluated at collocation points by
differentiating the network with respect to its OWN INPUTS. Burgers'
equation is u_t + u·u_x - (0.01/π)·u_xx = 0, so the residual needs u_t, u_x
and u_xx of the network — second derivatives through the graph, and then a
gradient of that with respect to the weights. Nothing in the architecture is
new; everything in the training signal is.

That second-order structure is why this is a bestiary entry and not a demo:
the VJP suite of Part 1 differentiates a loss with respect to parameters,
once. A PINN's loss is built from input-derivatives of the network, and its
gradient is a third-order object the emitter does not produce. The
Boltzmann generator beside it is the physics demo this book can train —
its physics enters through the target density and the reweighting, not
through derivatives of the network.

⚠ The paper's activation is tanh, chosen because the residual needs smooth
second derivatives — a ReLU net's u_xx is zero almost everywhere and the
PDE term would vanish. Our `Activation` enum has no tanh; the smooth `.gelu`
stands in and contributes zero parameters either way.

## Variants

- `pinnBurgers`      — the paper's Burgers' net: (x, t) → 8 hidden × 20 → u.
                       3,021 params, the number every reimplementation quotes.
- `pinnSchrodinger`  — the nonlinear Schrödinger net: (x, t) → 4 × 100 → (Re u, Im u).
- `pinnNavierStokes` — the cylinder-wake inverse problem: (x, y, t) → 8 × 20 → (ψ, p),
                       with the two unknown PDE coefficients learned as extra scalars
                       (not layers; +2 in prose).
- `tinyPinn`         — 2 hidden × 8, small enough to read in one pass.
-/

-- ════════════════════════════════════════════════════════════════
-- § Burgers' equation — the paper's forward problem
-- ════════════════════════════════════════════════════════════════
-- u_t + u u_x - (0.01/π) u_xx = 0 on x ∈ [-1, 1], t ∈ [0, 1], u(0, x) = -sin(πx),
-- u(t, ±1) = 0. 9 dense layers: 2 → 20 ×8 → 1, tanh.

def pinnBurgers : NetSpec where
  name   := "PINN (Burgers, 8 x 20 tanh)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 2 20 .gelu,        -- (x, t)
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 1 .identity     -- u(x, t)
  ]

-- ════════════════════════════════════════════════════════════════
-- § Nonlinear Schrödinger — complex-valued output as two channels
-- ════════════════════════════════════════════════════════════════
-- i h_t + ½ h_xx + |h|² h = 0, periodic in x. 5 layers of 100, two outputs.

def pinnSchrodinger : NetSpec where
  name   := "PINN (Schrödinger, 4 x 100 tanh)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 2 100 .gelu,
    .dense 100 100 .gelu,
    .dense 100 100 .gelu,
    .dense 100 100 .gelu,
    .dense 100 2 .identity    -- (Re h, Im h)
  ]

-- ════════════════════════════════════════════════════════════════
-- § Navier–Stokes, the inverse problem
-- ════════════════════════════════════════════════════════════════
-- Flow past a cylinder; the net outputs a stream function ψ and pressure p,
-- velocities come from ψ's derivatives so continuity holds by construction,
-- and the two unknown coefficients λ₁, λ₂ of the momentum equations are
-- trained alongside the weights (two scalars, not shown).

def pinnNavierStokes : NetSpec where
  name   := "PINN (Navier-Stokes inverse, 8 x 20 tanh)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 3 20 .gelu,        -- (x, y, t)
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 20 .gelu,
    .dense 20 2 .identity     -- (ψ, p)
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny fixture
-- ════════════════════════════════════════════════════════════════

def tinyPinn : NetSpec where
  name   := "tiny-PINN (2 x 8)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 2 8 .gelu,
    .dense 8 8 .gelu,
    .dense 8 1 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Physics-informed neural network"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The network is a small MLP. The loss is the PDE, evaluated by"
  IO.println "  differentiating the network with respect to its own inputs."

  pinnBurgers.summarize (size := .omitted) (unit := .thousands)
  pinnSchrodinger.summarize (size := .omitted) (unit := .thousands)
  pinnNavierStokes.summarize (size := .omitted) (unit := .thousands)
  tinyPinn.summarize (size := .omitted) (unit := .thousands)

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives: .dense and a smooth activation."
  IO.println "  • tanh is modelled as .gelu (no tanh in the enum; zero params"
  IO.println "    either way). A ReLU PINN would have u_xx = 0 a.e."
  IO.println "  • Not a demo here: the loss needs input-derivatives of the net,"
  IO.println "    and its gradient is third-order — the VJP suite of Part 1"
  IO.println "    differentiates a parameter loss once."
