import LeanMlir.Proofs.Codegen.StableHLO

/-! # `CertLayer` — composing certified backward graphs, so a NET is one object

Every `*BackB0` file in this repo stops at a **block** capstone: `r34DownBlockBackBatchedGraph_-
faithful`, `mnv2ResidBlockBackBatchedGraph_faithful`, `mbResidBlockBackBatchedGraph_faithful`,
`cnxDownChBackGraph_faithful`, and (2026-08-10) R50's three. Measured before writing this file:
**nothing in `LeanMlir/Proofs/` folds those blocks into a stage or a net.** So "the whole-net
composed backward" as §8 of `planning/mnv4_verified.md` uses the phrase means *block* capstones —
which is real, and is not a net.

⭐ **The obstacle was never the mathematics; it was that the chaining was open-coded.** Look at any
`<body>BackBatchedGraph_faithful`: it builds `G₁ x (G₂ (f₁ x) e)`, rewrites with the two component
faithfulness lemmas, and closes by `rfl` on `vjp_comp_at`'s definitional
`backward dy = f₁.backward (f₂.backward dy)`. That argument is **identical every time** and is
re-typed per composition, so a 16-block net would be 16 copies of it with ever-larger terms.

This file does it once. `CertLayer.comp` is that proof; everything else is bookkeeping.

## The structure

A `CertLayer m n` is a forward map plus, **at every input satisfying its own smoothness
precondition**, a VJP, differentiability, a backward StableHLO graph, and the theorem that the
graph *denotes* the VJP. Carrying `ok` inside the layer is what makes composition work: relu nets
are `_at`, so a block is only certified where its pre-activations miss the kinks, and the composite
of two layers is certified exactly where the first is certified **and** the second is certified at
the first's output. That is `ok x := L₁.ok x ∧ L₂.ok (L₁.fwd x)`, and it threads the deepening
hypothesis stack automatically instead of by hand.

⚠ **This is not a new trust assumption.** `CertLayer.comp` proves faithfulness of the composite
from the components' faithfulness; it introduces no axiom and no `sorry`. A chain built from
certified blocks is certified, and the fold is where that stops being a sentence and becomes a
theorem.

## Reading the list order

`chain [L₁, L₂, L₃]` runs **L₁ first**: `fwd = L₃.fwd ∘ L₂.fwd ∘ L₁.fwd`. List order is forward
execution order, which is how a block table reads. The backward graph nests the other way
automatically — `L₁.graph x (L₂.graph _ (L₃.graph _ e))` — because that is what the chain rule
says, and getting it backwards is a silent wrong-gradient rather than a type error whenever the
widths happen to agree (the §3 trap, one level up).
-/

namespace Proofs.StableHLO

/-- A layer whose backward StableHLO graph is **proven** to denote its VJP, wherever its own
    smoothness precondition `ok` holds.

    `ok` is a predicate on the INPUT because that is what relu smoothness is: a condition on the
    activations this particular layer sees. A globally-smooth layer (swish, a bare conv-bn) sets
    `ok := fun _ => True` and loses nothing. -/
structure CertLayer (m n : Nat) where
  /-- The forward map. -/
  fwd : Vec m → Vec n
  /-- Where this layer is certified — the smoothness hypotheses, as a predicate on the input. -/
  ok : Vec m → Prop
  /-- Differentiability at any certified point (needed to compose via `vjp_comp_at`). -/
  diff : ∀ x, ok x → DifferentiableAt ℝ fwd x
  /-- The proven VJP at any certified point. -/
  vjp : ∀ x, ok x → HasVJPAt fwd x
  /-- The backward graph, as a function of the forward activation and the incoming cotangent. -/
  graph : Vec m → SHlo n → SHlo m
  /-- ⭐ The theorem that makes it a *certified* layer: the graph denotes the VJP. -/
  faithful : ∀ (x : Vec m) (hx : ok x) (e : SHlo n),
    den (graph x e) = (vjp x hx).backward (den e)

namespace CertLayer

/-- The identity layer — certified everywhere, and its backward graph is the cotangent verbatim.
    The unit of `chain`. -/
noncomputable def id' (n : Nat) : CertLayer n n where
  fwd := fun y => y
  ok := fun _ => True
  diff := fun _ _ => differentiable_id.differentiableAt
  vjp := fun x _ => identity_has_vjp_at n x
  graph := fun _ e => e
  faithful := by intro _ _ _; rfl

/-- ⭐⭐ **THE COMPOSITION THEOREM — the whole point of this file.**

    Two certified layers compose into a certified layer. The backward graph nests (`L₁`'s graph
    fed `L₂`'s graph at `L₁`'s output), the smoothness preconditions conjoin, and faithfulness
    follows from the components' faithfulness plus `vjp_comp_at`'s definitional backward.

    This is the argument every `<body>BackBatchedGraph_faithful` in the repo writes out by hand.
    Proven once here, a chain of any length costs nothing. -/
noncomputable def comp {m n p : Nat} (L₁ : CertLayer m n) (L₂ : CertLayer n p) : CertLayer m p where
  fwd := L₂.fwd ∘ L₁.fwd
  ok := fun x => L₁.ok x ∧ L₂.ok (L₁.fwd x)
  diff := fun x hx => (L₂.diff _ hx.2).comp x (L₁.diff x hx.1)
  vjp := fun x hx =>
    vjp_comp_at L₁.fwd L₂.fwd x (L₁.diff x hx.1) (L₂.diff _ hx.2) (L₁.vjp x hx.1) (L₂.vjp _ hx.2)
  graph := fun x e => L₁.graph x (L₂.graph (L₁.fwd x) e)
  faithful := by
    intro x hx e
    rw [L₁.faithful x hx.1, L₂.faithful (L₁.fwd x) hx.2]
    rfl

/-- Fold a list of endo-layers into one. **List order is FORWARD execution order**:
    `chain [L₁, L₂, L₃] |>.fwd = L₃.fwd ∘ L₂.fwd ∘ L₁.fwd`. -/
noncomputable def chain {n : Nat} : List (CertLayer n n) → CertLayer n n
  | [] => id' n
  | L :: Ls => L.comp (chain Ls)

@[simp] theorem chain_nil {n : Nat} : chain ([] : List (CertLayer n n)) = id' n := rfl

@[simp] theorem chain_cons {n : Nat} (L : CertLayer n n) (Ls : List (CertLayer n n)) :
    chain (L :: Ls) = L.comp (chain Ls) := rfl

/-- The chain's forward map is the layers' composite, in list order. -/
theorem chain_fwd {n : Nat} (Ls : List (CertLayer n n)) (x : Vec n) :
    (chain Ls).fwd x = Ls.foldl (fun v L => L.fwd v) x := by
  induction Ls generalizing x with
  | nil => rfl
  | cons L Ls ih => simpa [chain, comp, Function.comp] using ih (L.fwd x)

/-- ⭐ **The net-level statement, in one line.** Whatever the chain's length, its backward graph
    denotes its VJP — so a stage, a trunk, or a whole net assembled from certified blocks is
    certified, with no per-length proof. This is just `CertLayer.faithful` at `chain Ls`; it is
    restated here because it is the theorem the fold exists to provide. -/
theorem chain_faithful {n : Nat} (Ls : List (CertLayer n n))
    (x : Vec n) (hx : (chain Ls).ok x) (e : SHlo n) :
    den ((chain Ls).graph x e) = ((chain Ls).vjp x hx).backward (den e) :=
  (chain Ls).faithful x hx e

end CertLayer

end Proofs.StableHLO
