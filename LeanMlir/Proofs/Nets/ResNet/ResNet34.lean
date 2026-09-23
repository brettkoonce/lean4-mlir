import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Nets.Small.MnistCNN
import LeanMlir.Proofs.Foundation.StridedConv

/-! # Toward real ResNet-34 — the deep-block chain (Chapter 5 Milestone B4)

A real ResNet-34 stacks **16 basic blocks** in four stages (3+4+6+3). Within a
stage every block is a self-map `Vec n → Vec n` (same channel count) but with its
**own** weights — so it is a *composition of a list* of distinct same-type maps,
not an `iterate` of one map.

This file proves the generic enabler: if every map in a list is differentiable
and has a VJP at its running activation, their composition (`chainComp`) does too —
by induction chaining `vjp_comp_at` (`chain_vjp_diff_at`). That turns "16 blocks
deep" into a `List.length`, no per-block boilerplate. `resnet34_has_vjp_at` is the
resulting parametric skeleton (abstract stem / downsample / blocks / head, kept for the
axiom audit); the shipped ResNet-34 is the batched chain in `ResNet34FullB` /
`ResNet34FullBVJP`, which does not build on this file.

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

-- ════════════════════════════════════════════════════════════════
-- § Deep-block chain at a smooth point (the conditional `_at` chain)
-- ════════════════════════════════════════════════════════════════

/-- Recursive hypothesis bundle for a chain of `HasVJPAt` blocks: each block is
    `DifferentiableAt` and `HasVJPAt` **at its running activation** — the point
    `chainComp rest x` feeding it (the deeper blocks run first). Residual identity
    blocks are only `HasVJPAt` at smooth points, so the chain must thread the
    point, not assume global differentiability. -/
def ChainData {n : Nat} (x : Vec n) : List (Vec n → Vec n) → Type
  | [] => PUnit
  | f :: rest =>
      -- `PProd` (not `×`): the first field `DifferentiableAt` is a `Prop`.
      PProd (DifferentiableAt ℝ f (chainComp rest x))
        (PProd (HasVJPAt f (chainComp rest x)) (ChainData x rest))

/-- The chain at a point both **has a VJP and is differentiable** there, from the
    per-block `ChainData`. The companion `DifferentiableAt` is carried alongside
    so the recursion can feed the inner-composition differentiability into each
    `vjp_comp_at` / `DifferentiableAt.comp`. -/
noncomputable def chain_vjp_diff_at {n : Nat} (x : Vec n) :
    (fs : List (Vec n → Vec n)) → ChainData x fs →
      PProd (HasVJPAt (chainComp fs) x) (DifferentiableAt ℝ (chainComp fs) x)
  | [], _ => ⟨(identity_has_vjp n).toHasVJPAt x, differentiable_id.differentiableAt⟩
  | f :: rest, d =>
      let ih := chain_vjp_diff_at x rest d.snd.snd
      ⟨vjp_comp_at (chainComp rest) f x ih.snd d.fst ih.fst d.snd.fst, d.fst.comp x ih.snd⟩

-- ════════════════════════════════════════════════════════════════
-- § The whole ResNet-34 network VJP
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network ResNet-34 VJP.** The conditional VJP of a real ResNet-34-shaped
    network at an input `x`:

      `dense ∘ GAP ∘ stage₄ ∘ stage₃ ∘ stage₂ ∘ stage₁ ∘ maxpool ∘ stem`

    with `stageᵢ = (identity-block chain) ∘ downsampleᵢ` for the three downsampling
    stages (the 3+4+6+3 = 16 basic blocks live in the `idsᵢ` lists + the three
    `downᵢ` blocks). Parametric over the component functions and their per-component
    VJP+differentiability witnesses at the running activations — so depth is a
    `List.length`, not 100 explicit weight arguments. Folded from the verified
    `vjp_comp_at` / `chain_vjp_diff_at` (`ChainData` threads each block's smooth point).

    This is the structural analogue of `cnn_has_vjp_at` scaled to 34 layers. ⚠ Its concrete
    instantiation used to be the 2-channel per-example proxy family, retired 2026-09-20: the
    non-degeneracy witness now discharges the clauses of the **batched, full-width**
    `resnet34ForwardB_full_has_vjp_at` instead, which is the one the ImageNet artifacts' tier is
    built on (`ResNet34FullBSeal`). This form survives as the audited parametric skeleton — depth
    as a `List.length` — and as the fold target of `ResNet34BackCertifiedTie`. -/
noncomputable def resnet34_has_vjp_at
    {s0 s1 s2 s3 s4 s5 s6 s7 : Nat}
    (stem : Vec s0 → Vec s1) (mp : Vec s1 → Vec s2)
    (ids1 : List (Vec s2 → Vec s2))
    (down2 : Vec s2 → Vec s3) (ids2 : List (Vec s3 → Vec s3))
    (down3 : Vec s3 → Vec s4) (ids3 : List (Vec s4 → Vec s4))
    (down4 : Vec s4 → Vec s5) (ids4 : List (Vec s5 → Vec s5))
    (gap : Vec s5 → Vec s6) (dense : Vec s6 → Vec s7)
    (x : Vec s0)
    (hstem : PProd (HasVJPAt stem x) (DifferentiableAt ℝ stem x))
    (hmp : PProd (HasVJPAt mp (stem x)) (DifferentiableAt ℝ mp (stem x)))
    (hids1 : ChainData (mp (stem x)) ids1)
    (hdown2 : PProd (HasVJPAt down2 (chainComp ids1 (mp (stem x))))
                    (DifferentiableAt ℝ down2 (chainComp ids1 (mp (stem x)))))
    (hids2 : ChainData (down2 (chainComp ids1 (mp (stem x)))) ids2)
    (hdown3 : PProd (HasVJPAt down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))
                    (DifferentiableAt ℝ down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))
    (hids3 : ChainData (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))) ids3)
    (hdown4 : PProd (HasVJPAt down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))))
                    (DifferentiableAt ℝ down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))))
    (hids4 : ChainData (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))) ids4)
    (hgap : PProd (HasVJPAt gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))))))
                  (DifferentiableAt ℝ gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))))))
    (hdense : PProd (HasVJPAt dense (gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))))))
                    (DifferentiableAt ℝ dense (gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))))))))
    : HasVJPAt
        (dense ∘ gap ∘ chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘
          chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem) x :=
  let p1 := vjp_comp_diff_at stem mp x hstem hmp
  let p2 := vjp_comp_diff_at (mp ∘ stem) (chainComp ids1) x p1 (chain_vjp_diff_at _ ids1 hids1)
  let p3 := vjp_comp_diff_at (chainComp ids1 ∘ mp ∘ stem) down2 x p2 hdown2
  let p4 := vjp_comp_diff_at (down2 ∘ chainComp ids1 ∘ mp ∘ stem) (chainComp ids2) x p3
              (chain_vjp_diff_at _ ids2 hids2)
  let p5 := vjp_comp_diff_at (chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem) down3 x p4 hdown3
  let p6 := vjp_comp_diff_at (down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              (chainComp ids3) x p5 (chain_vjp_diff_at _ ids3 hids3)
  let p7 := vjp_comp_diff_at (chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              down4 x p6 hdown4
  let p8 := vjp_comp_diff_at (down4 ∘ chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              (chainComp ids4) x p7 (chain_vjp_diff_at _ ids4 hids4)
  let p9 := vjp_comp_diff_at (chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              gap x p8 hgap
  let p10 := vjp_comp_diff_at (gap ∘ chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              dense x p9 hdense
  p10.fst

-- ════════════════════════════════════════════════════════════════
-- § ReLU helper shared by the batched seals
-- ════════════════════════════════════════════════════════════════

/-- ReLU output is always nonnegative. -/
theorem relu_nonneg (n : Nat) (v : Vec n) (k : Fin n) : 0 ≤ relu n v k := by
  simp only [relu]
  by_cases h : v k > 0
  · rw [ite_eq_left h]; exact le_of_lt h
  · rw [ite_eq_right h]

end Proofs
