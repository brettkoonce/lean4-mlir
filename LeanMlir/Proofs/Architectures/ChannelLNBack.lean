import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Architectures.ChannelLN

/-! # The channel-LayerNorm backward — the rowwise vector-LN input-VJP and its conjugation

The backward of the channel LayerNorm (`chanLNTensor3`, `ChannelLN.lean`)
and of ViT's per-token vector LayerNorm (`layerNormVec`, `LayerNorm.lean`), as the ℝ maps the
certified ties are stated about. `rowLNVecFlatBack` is the per-row input-VJP — the per-channel
`γ` scale then the three-term `bnGradInput` at unit γ, lifted rowwise — and
`chanLNTensor3Back` is that map conjugated by the same four layout permutations the forward
uses. `ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp` proves the conjugation equals
the certified `chanLNTensor3HasVJP` backward, and `rowLNVecFlatBack_eq_vecLN_vjp`
(`ViTVecLNBackCertifiedTie.lean`) does the same for the row map, so ConvNeXt's LN backward
and ViT's share `rowLNVecFlatBack`. Both ties rest on the three lemmas at the end of this file:
`bnGradInput` is the BN/LN witness's backward, and the vector-LN row and rowwise VJPs reduce to it. -/

namespace Proofs

open Proofs.StableHLO (transposeFlat)

/-- **The rowwise vector-LN input-VJP.** Per spatial row: scale the cotangent by the per-channel
    `γ` (`layerScale`'s adjoint is `diagBack γ`), then the consolidated three-term `bnGradInput`
    at `γ = 1` over that row's `c` channels (LN's adjoint = BN's, `layerNormForward = bnForward`).
    The `+β` translation contributes the identity, so it does not appear. -/
noncomputable def rowLNVecFlatBack (s c : Nat) (ε : ℝ) (γ : Vec c) (X : Vec (s * c)) :
    Vec (s * c) → Vec (s * c) :=
  perRowFlatPR s c (fun r => bnGradInput c ε 1 (Mat.unflatten X r) ∘ diagBack γ)

/-- **The channel-LN input-VJP** (as a function of the cotangent, at a saved input `x`) — the exact
    reverse of `chanLNTensor3`'s five factors. A permutation's adjoint is its inverse permutation,
    so the conjugation comes back unchanged and only the middle flips to `rowLNVecFlatBack`, read
    at the TRANSPOSED saved input (the row backward needs its own row's activation). It takes no
    `β`: the `+β` translation's VJP is the identity, and the tie proves the certified backward is
    β-free too. -/
noncomputable def chanLNTensor3Back (c h w : Nat) (ε : ℝ) (γ : Vec c) (x : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  reassocBack c h w ∘
    transposeFlat (h * w) c ∘
    rowLNVecFlatBack (h * w) c ε γ (chanLNRows c h w x) ∘
    transposeFlat c (h * w) ∘
    reassocFwd c h w

/-- **The concrete three-term BN/LN input gradient IS the certified VJP backward.** `bnGradInput`
    is not `rfl`-equal to `(bnHasVJP …).backward` — the witness is built through a
    `rw [bnForward_eq_compose]` cast — but both reduce to the canonical `∑ pdiv` form
    (`bn_input_grad_correct` and `.correct`). The function-level peer of `bnBack_faithful_fn`. -/
theorem bnGradInput_eq_vjp_backward {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (x dy : Vec n) :
    bnGradInput n ε γ x dy = (bnHasVJP n ε γ β hε).backward x dy := by
  funext i
  rw [bn_input_grad_correct n ε γ β hε x dy i]
  exact ((bnHasVJP n ε γ β hε).correct x dy i).symm

/-- **The vector-LN row backward is `bnGradInput` after the `γ` scale.** `layerNormVec` is
    `(+β) ∘ layerScale γ ∘ LN(1,0)`, so its VJP applies: the bias translation's identity backward,
    then `diagBack γ`, then the LN input gradient at `γ = 1`. The `+β` drops out — this is where
    the whole channel-LN backward story becomes β-free. -/
theorem layerNormVecHasVJP_backward_eq {D : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec D)
    (x dy : Vec D) :
    (layerNormVecHasVJP D ε γ β hε).backward x dy
      = bnGradInput D ε 1 x (diagBack γ dy) := by
  rw [bnGradInput_eq_vjp_backward ε 1 0 hε x (diagBack γ dy)]
  rfl

/-- **The rowwise vector-LN backward is `rowLNVecFlatBack`.** `rowLNVecFlatHasVJP` is the
    `rowwiseHasVJPMat` lift of the row VJP through `HasVJPMat.toHasVJP`, and
    `rowLNVecFlatBack` is `perRowFlatPR` of the row's closed form — the same per-row map at the
    same row of the saved input, so this is the row lemma read at each `(row, col)`. -/
theorem rowLNVecFlatHasVJP_backward_eq {s c : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (X dy : Vec (s * c)) :
    (rowLNVecFlatHasVJP s c ε γ β hε).backward X dy = rowLNVecFlatBack s c ε γ X dy := by
  funext idx
  show (layerNormVecHasVJP c ε γ β hε).backward
      (Mat.unflatten X (finProdFinEquiv.symm idx).1)
      (Mat.unflatten dy (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 = _
  rw [layerNormVecHasVJP_backward_eq ε hε γ β]
  rfl

end Proofs
