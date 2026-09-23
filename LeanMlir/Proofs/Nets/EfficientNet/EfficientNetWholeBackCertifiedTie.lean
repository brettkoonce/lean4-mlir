import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackChains

/-! # EfficientNet-B0's stem and head endpoint ties

The two concrete endpoints the sixteen-block whole-net tie (`EfficientNetFullWholeBackCertifiedTie`)
stands on: `stemBBack_eq_vjp_backward` (at the XLA-`SAME` phase every shipped B0 artifact emits) and
`headFwdBBack_eq_vjp_backward`. Each is one `rw` of a per-example leaf tie and then `rfl`: the
batched stage's VJP is `vjp_comp`-built, so its backward already reduces to the composition of the
stage backwards, and `batchMap`'s VJP reduces to the leaf backward applied row-wise.

⭐ **`batchMap_has_vjp`'s transport does not block the reduction, and the planning note that
said it would is withdrawn.** It is built as `(batchMap_eq_rowwiseFlat f).symm ▸
hasVJPMat_to_hasVJP (rowwise_has_vjp_mat …)`, and §5's standing trap is that an `Eq.mpr` blocks
`.backward` from reducing. It does not here: the transported equation holds by `funext … ; rfl`,
and proof irrelevance is definitional in Lean, so `.backward` reduces straight through the `▸`
to the leaf backward applied row-wise — checked as a bare `rfl`, and every tie below relies on it.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The two concrete endpoint ties
-- ════════════════════════════════════════════════════════════════

/-- **The STEM tie.** The hand-written `batchMap (flatConvStride2XlaBack) ∘ bnBack ∘ swishBack`
    IS `stemB`'s certified backward. One `rw` of the odd-phase leaf tie, then `rfl` — the
    stage's VJP is `vjp_comp`-built so its backward is already the composition, and the leaf's
    backward is input-independent (a convolution is linear), so the row-wise `batchMap` lift
    matches at every saved input. -/
theorem stemBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    (StableHLO.batchMap N (flatConvStride2XlaBack (h := h) (w := w) W)
      ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
          (StableHLO.batchMap N (flatConvStride2Xla W b) x)
      ∘ (swish_has_vjp (N * (oc * h * w))).backward
          (StableHLO.bnBatchLA N oc h w ε γ β (StableHLO.batchMap N (flatConvStride2Xla W b) x)))
      = (stemB_has_vjp N (h := h) (w := w) W b ε hε γ β).backward x := by
  rw [flatConvStride2XlaBack_eq_vjp_backward hkH hkW W b (fun _ => 0)]
  rfl

/-- **The HEAD tie.** The four-stage hand chain `batchMap (convFlatBack) ∘ bnBack ∘ swishBack ∘
    batchMap gapBack ∘ batchMap (dense Wᵀ 0)` IS `headFwdB`'s certified backward. `gapBack` needs
    no rewrite: it is definitionally the global-average-pool VJP's backward. -/
theorem headFwdBBack_eq_vjp_backward {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) (x : Vec (N * (c * h * w))) :
    ((StableHLO.batchMap N (convFlatBack (h := h) (w := w) Wh)
        ∘ (bnBatchLA_has_vjp N oc h w εh hεh γh βh).backward
            (StableHLO.batchMap N (flatConv Wh bh) x)
        ∘ (swish_has_vjp (N * (oc * h * w))).backward
            (StableHLO.bnBatchLA N oc h w εh γh βh (StableHLO.batchMap N (flatConv Wh bh) x)))
      ∘ StableHLO.batchMap N (gapBack oc h w)
      ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec oc)))
      = (headFwdB_has_vjp N (h := h) (w := w) Wh bh εh hεh γh βh Wfc bfc).backward x := by
  rw [convFlatBack_eq_vjp_backward (by simp) (by simp) Wh bh (fun _ => 0),
      dense_transpose_eq_vjp_backward Wfc bfc (fun _ => 0)]
  rfl

end Proofs
