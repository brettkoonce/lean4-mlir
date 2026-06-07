import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.StridedConv

/-! # Toward real ResNet-34 — the deep-block chain (Chapter 6 Milestone B4)

A real ResNet-34 stacks **16 basic blocks** in four stages (3+4+6+3). Within a
stage every block is a self-map `Vec n → Vec n` (same channel count) but with its
**own** weights — so it is a *composition of a list* of distinct same-type maps,
not an `iterate` of one map.

This file proves the generic enabler: if every map in a list is differentiable
and has a VJP, their composition (`chainComp`) does too — by induction chaining
`vjp_comp`. That turns "16 blocks deep" into a `List.length`, no per-block
boilerplate. The full ResNet-34 forward (strided proj blocks via `flatConvStride2`
+ chained identity blocks + per-channel BN) is assembled on top of this.

Closes under `[propext, Classical.choice, Quot.sound]`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Composition of a list of same-type self-maps
-- ════════════════════════════════════════════════════════════════

/-- Compose a list of self-maps left-to-right as data flows: `chainComp [f₁,…,fₖ]
    = f₁ ∘ … ∘ fₖ` (the last list element runs first, i.e. is the deepest). A
    ResNet stage is `chainComp` of its blocks. -/
noncomputable def chainComp {n : Nat} (fs : List (Vec n → Vec n)) : Vec n → Vec n :=
  fs.foldr (· ∘ ·) id

@[simp] theorem chainComp_nil {n : Nat} : chainComp ([] : List (Vec n → Vec n)) = id := rfl

@[simp] theorem chainComp_cons {n : Nat} (f : Vec n → Vec n) (fs : List (Vec n → Vec n)) :
    chainComp (f :: fs) = f ∘ chainComp fs := rfl

/-- A chain of differentiable maps is differentiable. -/
theorem chainComp_differentiable {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) : Differentiable ℝ (chainComp fs) := by
  induction fs with
  | nil => exact differentiable_id
  | cons f fs ih =>
    rw [chainComp_cons]
    exact (hdiff f (List.mem_cons.2 (Or.inl rfl))).comp
      (ih (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg))))

/-- **Deep-chain VJP.** A composition of a list of differentiable maps that each
    have a VJP has a VJP — the backward runs each block's backward in reverse
    order. By induction chaining `vjp_comp`; the structural heart of a deep
    ResNet stage (k distinct-weight basic blocks). -/
noncomputable def vjp_chain {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) (hvjp : ∀ f ∈ fs, HasVJP f) :
    HasVJP (chainComp fs) :=
  match fs with
  | [] => show HasVJP (id : Vec n → Vec n) from identity_has_vjp n
  | f :: rest =>
    show HasVJP (f ∘ chainComp rest) from
    vjp_comp (chainComp rest) f
      (chainComp_differentiable rest (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg))))
      (hdiff f (List.mem_cons.2 (Or.inl rfl)))
      (vjp_chain rest (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg)))
                      (fun g hg => hvjp g (List.mem_cons.2 (Or.inr hg))))
      (hvjp f (List.mem_cons.2 (Or.inl rfl)))

/-- **Deep-chain VJP correctness** (ℝ-headline): the chained backward equals the
    `pdiv`-contracted Jacobian of the whole composition. -/
theorem vjp_chain_correct {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) (hvjp : ∀ f ∈ fs, HasVJP f)
    (x dy : Vec n) (i : Fin n) :
    (vjp_chain fs hdiff hvjp).backward x dy i
      = ∑ j : Fin n, pdiv (chainComp fs) x i j * dy j :=
  (vjp_chain fs hdiff hvjp).correct x dy i

end Proofs
