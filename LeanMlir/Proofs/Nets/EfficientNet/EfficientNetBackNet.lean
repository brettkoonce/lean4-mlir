import LeanMlir.Proofs.Foundation.BatchedBackLinks
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0

/-! # EfficientNet's MBConv1 and head backward graphs

The backward graphs of the two EfficientNet stage forwards not covered in `EfficientNetBackB0.lean`:
`mbNoExpBackBatchedGraph_faithful` (MBConv1, `projB ∘ seB ∘ dwbsB` — no expand stage) and
`headBackBatchedGraph_faithful` (`dense ∘ GAP ∘ cbsB`).

## Every stage here is GLOBAL

EfficientNet is swish/sigmoid throughout, both smooth, so every layer below has `ok = True`: the
backward graph denotes the VJP at **every** input, no side conditions. Contrast MNv4/R34/R50, whose
relu kinks force `_at`.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ CLOSING §8e's HOLES AGAINST THE *NAMED* FORWARDS
-- ════════════════════════════════════════════════════════════════

/-! Where each EfficientNet stage forward's backward graph lives:

| forward | backward-graph theorem |
|---|---|
| `mbStridedFwdB` | `mbDownBodyBackBatchedGraph_faithful` certifies `mbStridedFwdBHasVJP` itself |
| `mbExpFwdB` | `mbBodyBackBatchedGraph_faithful` certifies `mbExpFwdBHasVJP` at `ic = oc` (the graph's type) |
| `mbNoExpFwdB` | `mbNoExpBackBatchedGraph_faithful` (below) |
| `headFwdB` | `headBackBatchedGraph_faithful` (below) | -/

/-- **`mbNoExpFwdB`'s backward graph** — MBConv1 has no expand stage, so the chain is
    `dwbsB⁻¹ ∘ seB⁻¹ ∘ projB⁻¹`. -/
noncomputable def mbNoExpBackBatchedGraph {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  let xD := dwbsB N (h := h) (w := w) Wd bd εd γd βd x
  let xS := seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ xD
  dwbsBackBatchedGraph Wd bd εd γd βd x
    (.seBackBatched (N := N) "%seW1" "%seb1" "%seW2" "%seb2" "%seX" Wz₁ bz₁ Wz₂ bz₂ xD
      (projBackBatchedGraph Wp bp εp γp βp xS e))

theorem mbNoExpBackBatchedGraph_faithful {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (mbNoExpBackBatchedGraph Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e)
      = (mbNoExpFwdBHasVJP N Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbNoExpBackBatchedGraph, dwbsBackBatchedGraph_faithful (hε := hεd),
      seBackBatched_faithful, projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbNoExpFwdBHasVJP, Function.comp_apply]
  rfl

/-- **`headFwdB`'s backward graph** — `cbsB⁻¹ ∘ GAP⁻¹ ∘ dense⁻¹`. The EfficientNet peer of
    MNv4's head. -/
noncomputable def headBackBatchedGraph {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (_bfc : Vec nC)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * nC)) : SHlo (N * (c * h * w)) :=
  cbsBackBatchedGraph Wh bh εh γh βh x
    (.gapBackBatched (N := N) (c := oc) (h := h) (w := w)
      (.denseRowBack (N := N) (a := oc) (c := nC) "%Wfc" Wfc e))

theorem headBackBatchedGraph_faithful {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * nC)) :
    den (headBackBatchedGraph Wh bh εh γh βh Wfc bfc x e)
      = (headFwdBHasVJP N Wh bh εh hεh γh βh Wfc bfc).backward x (den e) := by
  rw [headBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεh)]
  simp only [headFwdBHasVJP]
  rfl

end Proofs.StableHLO
