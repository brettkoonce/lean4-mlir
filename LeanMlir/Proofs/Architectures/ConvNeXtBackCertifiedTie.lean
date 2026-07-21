import LeanMlir.Proofs.ConvNeXtBackFloatBridge
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie

/-! # §B: the ConvNeXt block-body backward float bridge targets the CERTIFIED VJP

The A3 backward float bridge `cnxBlockBodyBack` (`ConvNeXtBackFloatBridge.lean`) proves
**deployed-float ≈ a hand-assembled reverse-mode transcription** of the ConvNeXt block body. This file
closes §B for that body: the transcription IS the certified input-gradient VJP
`convNextBlockBody_has_vjp` (`ConvNeXt.lean`), in the SAME non-batched vocabulary — so the float
bridge's closeness is now closeness to **the certified gradient**.

The ConvNeXt block body is `convNextBlockBody = layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`,
whose certified VJP backward applies the reverses in order
`LS.back → PR.back → GE.back → EX.back → LN.back → D.back`. The float `cnxBlockBodyBack` is the exact
peer chain `depthwiseFlatBack ∘ lnB ∘ convFlatBack Wex ∘ geluB ∘ convFlatBack Wpr ∘ lsB`. The tie pins
the three smooth/diagonal/norm backs (`lsB`/`geluB`/`lnB`) to the certified `layerScale`/`gelu`/`LN`
backwards at the exact saved activations, and ties the two 1×1 convs + the depthwise to their certified
input-VJPs via the leaf gates (`convFlatBack_eq_vjp_backward`, `depthwiseFlatBack_eq_vjp_backward`).
The conv/depthwise backwards ignore their primal (linear), the pinned backs match the certified saved
activations definitionally, so the whole tie closes by rewriting the three convolution leaves + `rfl`.

This is the convnext analogue of `r34IdBlockBack_eq_rblkPC_vjp`; b1-free (the per-example body is the
non-batched object the float bridge reverses, no `batchMap` reconciliation). The certified
`convNextBlockBody_has_vjp` already existed, so the work is the depthwise leaf gate (shared, in
`DepthwiseBackCertifiedTie`) + this per-block tie. 3-axiom-clean.
-/

namespace Proofs

open Classical

/-- **The §B ConvNeXt body tie: float-bridge backward = certified VJP.** `cnxBlockBodyBack`, with its
    abstract layer-scale / GELU / LayerNorm backs pinned to the certified `layerScale` / `gelu` /
    `layerNorm` backwards at the exact saved forward activations (`depthwiseFlat … v` for LN, the deeper
    forward partials for GELU/LS), equals `(convNextBlockBody_has_vjp_at …).backward`.

    Both sides apply the six op-reverses in the order `LS → PR → GE → EX → LN → D`. The two 1×1 convs
    tie via `convFlatBack_eq_vjp_backward` (1×1 is odd) and the depthwise via
    `depthwiseFlatBack_eq_vjp_backward`; the conv/depthwise backwards ignore their (linear) primal and
    the pinned smooth backs carry the certified saved activations, so after rewriting the three leaves
    everything matches definitionally. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem cnxBlockBodyBack_eq_convNextBlockBody_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (Wdw : DepthwiseKernel c kHd kWd) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    cnxBlockBodyBack Wdw Wex Wpr
      ((layerNorm_has_vjp (c * h * w) εn γn βn hεn).backward (depthwiseFlat Wdw bdw v))
      ((layerScale_has_vjp γls).backward
        ((flatConv (h := h) (w := w) Wpr bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      ((gelu_has_vjp (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      = (convNextBlockBody_has_vjp_at Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v).backward := by
  funext dy
  unfold cnxBlockBodyBack
  rw [convFlatBack_eq_vjp_backward (W := Wex) (b := bex)
        (x := (layerNormForward (c * h * w) εn γn βn ∘ depthwiseFlat (h := h) (w := w) Wdw bdw) v)
        (by decide) (by decide),
      convFlatBack_eq_vjp_backward (W := Wpr) (b := bpr)
        (x := (gelu (cExp * h * w) ∘ flatConv (h := h) (w := w) Wex bex ∘
          layerNormForward (c * h * w) εn γn βn ∘ depthwiseFlat (h := h) (w := w) Wdw bdw) v)
        (by decide) (by decide),
      depthwiseFlatBack_eq_vjp_backward hkHd hkWd Wdw bdw v]
  rfl

/-- **The §B ConvNeXt block tie (residual-wrapped).** The full block is `residual (body)`, so the
    float block backward `residual (cnxBlockBodyBack …)` (the `dy ↦ bodyBack(dy) + dy` additive skip,
    as `floatBridges_cnxBlockBack` wraps it) equals `(convNextBlock_has_vjp_at …).backward`. Immediate
    from the body tie + the residual fan-in (`residual_has_vjp = biPath_has_vjp body id`, the skip's
    backward is `dy`): rewrite the body tie, then `rfl`. 3-axiom-clean. -/
theorem cnxBlockBack_eq_convNextBlock_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (Wdw : DepthwiseKernel c kHd kWd) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    Proofs.residual (cnxBlockBodyBack Wdw Wex Wpr
      ((layerNorm_has_vjp (c * h * w) εn γn βn hεn).backward (depthwiseFlat Wdw bdw v))
      ((layerScale_has_vjp γls).backward
        ((flatConv (h := h) (w := w) Wpr bpr ∘ gelu (cExp * h * w) ∘
          flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v))
      ((gelu_has_vjp (cExp * h * w)).backward
        ((flatConv (h := h) (w := w) Wex bex ∘ layerNormForward (c * h * w) εn γn βn ∘
          depthwiseFlat (h := h) (w := w) Wdw bdw) v)))
      = (convNextBlock_has_vjp_at Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v).backward := by
  rw [cnxBlockBodyBack_eq_convNextBlockBody_vjp hkHd hkWd Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v]
  rfl

end Proofs
