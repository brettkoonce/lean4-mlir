import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4BackB0
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0

/-! # EfficientNet's MBConv1 and head backward graphs

`planning/archive/mnv4_verified.md` §8e swept the repo for **certified batched forwards with no
BACKWARD graph**. Four were EfficientNet's; two turned out to be naming artifacts and two were
genuine (the table below). This file closes the two
genuine holes against the named forwards: `mbNoExpBackBatchedGraph_faithful` (MBConv1,
`projB ∘ seB ∘ dwbsB` — no expand stage, so nothing had composed it) and
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

/-! ⭐ Probing the named forwards found the real situation, which is **not** what §8e assumed:

| forward | verdict |
|---|---|
| `mbStridedFwdB` | ⭐ **never a hole** — `mbDownBodyB_has_vjp` is *definitionally the same object* (`rfl`), and it already has a certified graph. A duplicate NAME, not a missing proof. |
| `mbExpFwdB` | same shape as `mbBodyB_has_vjp`, which bakes in `ic = oc = c`; tied where the types meet |
| `mbNoExpFwdB` | genuine — nothing composed `projB ∘ seB ∘ dwbsB` |
| `headFwdB` | genuine — nothing composed `dense ∘ GAP ∘ cbsB` |

▶ So §8e over-counted: a sweep keyed on *names* cannot see that two names denote one object. The
lesson is the same one §4c(a) taught about the relu6 detector, in the opposite direction — there a
detector could not fire, here one fires spuriously. **Both are measurement bugs, and only re-running
the measurement after the fix catches either.** -/

/-- **`mbNoExpFwdB`'s backward graph** — genuinely new: MBConv1 has no expand stage, so
    `dwbsB⁻¹ ∘ seB⁻¹ ∘ projB⁻¹` had never been chained. -/
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
      = (mbNoExpFwdB_has_vjp N Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbNoExpBackBatchedGraph, dwbsBackBatchedGraph_faithful (hε := hεd),
      seBackBatched_faithful, projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbNoExpFwdB_has_vjp, vjp_comp, Function.comp_apply]
  rfl

/-- **`headFwdB`'s backward graph** — genuinely new: `cbsB⁻¹ ∘ GAP⁻¹ ∘ dense⁻¹`. The EfficientNet
    peer of MNv4's head, and the last stage-level hole in the repo. -/
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
      = (headFwdB_has_vjp N Wh bh εh hεh γh βh Wfc bfc).backward x (den e) := by
  rw [headBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεh)]
  simp only [headFwdB_has_vjp, vjp_comp, Function.comp_apply]
  rfl

end Proofs.StableHLO
