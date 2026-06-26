import LeanMlir.Proofs.EfficientNetBackFloatBridge
import LeanMlir.Proofs.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Resnet34BackCertifiedTie

/-! # §B: the EfficientNet MBConv body backward targets the CERTIFIED VJP

The A3 backward float bridge `mbconvBodyBack` (`EfficientNetBackFloatBridge.lean`) proves
**deployed-float ≈ a hand-assembled reverse-mode transcription** of the per-example MBConv body. This
file closes §B for that body: the transcription IS the certified input-gradient VJP `mbconvBody_has_vjp`
(`EfficientNet.lean`). Unlike mnv2/convnext, the certified per-example body VJP already exists in the
right (global-`bnForward`, non-batched) vocabulary — the per-example body is exactly what the forward
bridge `floatBridges_mbconvBody` stops at, and the batched whole-net is the separate batched-emit lift —
so no fresh certified VJP is built; b1-free.

The MBConv body is `mbconvBody = (BN∘conv Wp) ∘ SE ∘ (swish∘BN∘depthwise Wd) ∘ (swish∘BN∘conv We)`, whose
certified VJP applies `project.back → SE.back → depthwise.back → expand.back`. The float `mbconvBodyBack`
is the peer chain `(convFlatBack We ∘ bnBe ∘ swBe) ∘ (depthwiseFlatBack Wd ∘ bnBd ∘ swBd) ∘ seB ∘
(convFlatBack Wp ∘ bnBp)`. The tie pins the abstract BN backs (`bnBe/bnBd/bnBp`) to `bn_has_vjp.backward`,
the swish backs (`swBe/swBd`) to `swish_has_vjp.backward`, and the squeeze-excite back (`seB`) to the
certified `seBlockFull_has_vjp.backward` — each at its saved forward activation — and ties the two 1×1
convs + the depthwise via the leaf gates. The conv/depthwise backwards ignore their (linear) primal, the
pinned backs carry the certified saved activations, so after rewriting the three convolution leaves
everything matches definitionally. 3-axiom-clean.
-/

namespace Proofs

open Classical

/-- **The §B EfficientNet MBConv body tie: float-bridge backward = certified VJP.** `mbconvBodyBack`,
    with its abstract BN backs, swish backs, and squeeze-excite back pinned to the certified
    `bn_has_vjp` / `swish_has_vjp` / `seBlockFull_has_vjp` backwards at the exact saved activations,
    equals `(mbconvBody_has_vjp …).backward v`.

    Both sides apply the four stage-reverses `project → SE → depthwise → expand`. The two 1×1 convs tie
    via `convFlatBack_eq_vjp_backward` (1×1 odd) and the depthwise via `depthwiseFlatBack_eq_vjp_backward`;
    the conv/depthwise backwards ignore their (linear) primal, and the pinned BN/swish/SE backs carry the
    certified saved activations, so after rewriting the three leaves everything matches definitionally.
    Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem mbconvBodyBack_eq_mbconvBody_vjp {cin cmid cout h w kHe kWe kHd kWd kHp kWp r : Nat}
    (hkHe : 2 * ((kHe - 1) / 2) + 1 = kHe) (hkWe : 2 * ((kWe - 1) / 2) + 1 = kWe)
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (hkHp : 2 * ((kHp - 1) / 2) + 1 = kHp) (hkWp : 2 * ((kWp - 1) / 2) + 1 = kWp)
    (We : Kernel4 cmid cin kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 cout cmid kHp kWp) (bp : Vec cout) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (cin * h * w)) :
    mbconvBodyBack We Wd Wp
      ((bn_has_vjp (cmid * h * w) εe γe βe hεe).backward (flatConv We be v))
      ((bn_has_vjp (cmid * h * w) εd γd βd hεd).backward
        (depthwiseFlat Wd bd
          ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be) v)))
      ((swish_has_vjp (cmid * h * w)).backward
        (bnForward (cmid * h * w) εe γe βe (flatConv We be v)))
      ((swish_has_vjp (cmid * h * w)).backward
        (bnForward (cmid * h * w) εd γd βd
          (depthwiseFlat Wd bd
            ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be) v))))
      ((seBlockFull_has_vjp (h := h) (w := w) Ws₁ bs₁ Ws₂ bs₂).backward
        ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εd γd βd ∘ depthwiseFlat Wd bd)
          ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be) v)))
      ((bn_has_vjp (cout * h * w) εp γp βp hεp).backward
        (flatConv Wp bp
          (seBlockFull (h := h) (w := w) Ws₁ bs₁ Ws₂ bs₂
            ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εd γd βd ∘ depthwiseFlat Wd bd)
              ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be) v)))))
      = (mbconvBody_has_vjp We be εe γe βe hεe Wd bd εd γd βd hεd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp).backward v := by
  funext dy
  unfold mbconvBodyBack
  rw [convFlatBack_eq_vjp_backward (W := Wp) (b := bp)
        (x := seBlockFull (h := h) (w := w) Ws₁ bs₁ Ws₂ bs₂
          ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εd γd βd ∘ depthwiseFlat Wd bd)
            ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be) v))) hkHp hkWp,
      depthwiseFlatBack_eq_vjp_backward hkHd hkWd Wd bd
        ((swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be) v),
      convFlatBack_eq_vjp_backward (W := We) (b := be) (x := v) hkHe hkWe]
  rfl

end Proofs
