import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Architectures.ConvNeXtChannelLN

/-! # The channel-LayerNorm backward — the rowwise vector-LN input-VJP and its conjugation

The backward of ConvNeXt's real channel LayerNorm (`chanLNTensor3`, `ConvNeXtChannelLN.lean`)
and of ViT's per-token vector LayerNorm (`layerNormVec`, `ViTVecLN.lean`), as the ℝ maps the
certified ties are stated about. `rowLNVecFlatBack` is the per-row input-VJP — the per-channel
`γ` scale then the three-term `bn_grad_input` at unit γ, lifted rowwise — and
`chanLNTensor3Back` is that map conjugated by the same four layout permutations the forward
uses. `ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp` proves the conjugation equals
the certified `chanLNTensor3_has_vjp` backward, and `rowLNVecFlatBack_eq_vecLN_vjp`
(`ViTVecLNBackCertifiedTie.lean`) does the same for the row map, which is how ConvNeXt's LN
backward turned out to be ViT's as well.

Moved here from the float bridge that defined it beside its float twin on 2026-09-08
(`planning/archive/float_second_pass.md`). -/

namespace Proofs

open Proofs.StableHLO (transposeFlat)

/-- **The rowwise vector-LN input-VJP.** Per spatial row: scale the cotangent by the per-channel
    `γ` (`layerScale`'s adjoint is `diagBack γ`), then the consolidated three-term `bn_grad_input`
    at `γ = 1` over that row's `c` channels (LN's adjoint = BN's, `layerNormForward = bnForward`).
    The `+β` translation contributes the identity, so it does not appear. -/
noncomputable def rowLNVecFlatBack (s c : Nat) (ε : ℝ) (γ : Vec c) (X : Vec (s * c)) :
    Vec (s * c) → Vec (s * c) :=
  perRowFlatPR s c (fun r => bn_grad_input c ε 1 (Mat.unflatten X r) ∘ diagBack γ)

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

end Proofs
