import LeanMlir.Proofs.Float.FloatComposeBridge
import LeanMlir.Proofs.Architectures.CifarCNN

/-! # Whole-net forward float-bridge for the deeper (8-conv, no-BN) CIFAR CNN

A1 prototype (planning/tier23_float_and_syntactic_faithfulness.md). The 4-conv
CIFAR forward already has its monolithic budget capstone (`FloatModel.cifar_float_close`,
`CifarFloatBridge.lean`). This file lands the 8-conv variant via the **existential
`FloatBridges` assembly path** (`FloatBridges.comp`) instead — no new float-forward
`F` definition, no new budget term: `FloatBridges` packages the per-op `B`/`L`/`fF`
and `.comp` threads magnitudes automatically.

This is the path the deep-net capstones (mnv2/enet/convnext/vit) will reuse, so the
prototype validates that path end-to-end: `cifarCnn8Forward` is a clean `∘`-chain of
`flatConv` / `relu` / `maxPoolFlat` / `dense`, each of which float-bridges
(`floatBridges_flatConv`, `floatBridges_relu`, `floatBridges_maxPool`,
`floatBridges_dense`). Closes under `[propext, Classical.choice, Quot.sound]`. -/

namespace Proofs

/-- **The deeper 8-conv CIFAR CNN float-bridges.** For any input magnitude there is an
    output magnitude and a `FloatClose` certificate (a specific float forward within an
    explicit error modulus of the certified `ℝ` forward) — assembled in one `.comp`
    chain over the per-op bridges. -/
theorem cifar8_floatBridges
    {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1)
    (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    {w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ w₆ β₆ w₇ β₇ w₈ β₈ w₉ β₉ wa βa wb βb : ℝ}
    (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄)
    (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅) (hw₆ : 0 ≤ w₆) (hβ₆ : 0 ≤ β₆)
    (hw₇ : 0 ≤ w₇) (hβ₇ : 0 ≤ β₇) (hw₈ : 0 ≤ w₈) (hβ₈ : 0 ≤ β₈)
    (hw₉ : 0 ≤ w₉) (hβ₉ : 0 ≤ β₉) (hwa : 0 ≤ wa) (hβa : 0 ≤ βa)
    (hwb : 0 ≤ wb) (hβb : 0 ≤ βb)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ o c kh kw, |W₃ o c kh kw| ≤ w₃) (hb₃ : ∀ o, |b₃ o| ≤ β₃)
    (hW₄ : ∀ o c kh kw, |W₄ o c kh kw| ≤ w₄) (hb₄ : ∀ o, |b₄ o| ≤ β₄)
    (hW₅ : ∀ o c kh kw, |W₅ o c kh kw| ≤ w₅) (hb₅ : ∀ o, |b₅ o| ≤ β₅)
    (hW₆ : ∀ o c kh kw, |W₆ o c kh kw| ≤ w₆) (hb₆ : ∀ o, |b₆ o| ≤ β₆)
    (hW₇ : ∀ o c kh kw, |W₇ o c kh kw| ≤ w₇) (hb₇ : ∀ o, |b₇ o| ≤ β₇)
    (hW₈ : ∀ o c kh kw, |W₈ o c kh kw| ≤ w₈) (hb₈ : ∀ o, |b₈ o| ≤ β₈)
    (hW₉ : ∀ i j, |W₉ i j| ≤ w₉) (hb₉ : ∀ j, |b₉ j| ≤ β₉)
    (hWa : ∀ i j, |Wa i j| ≤ wa) (hba : ∀ j, |ba j| ≤ βa)
    (hWb : ∀ i j, |Wb i j| ≤ wb) (hbb : ∀ j, |bb j| ≤ βb)
    (hic : 0 < ic) (hc1 : 0 < c1) (hc2 : 0 < c2) (hc3 : 0 < c3) (hc4 : 0 < c4)
    (hd1 : 0 < d1) (hh : 0 < h) (hw : 0 < w) :
    FloatBridges (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
      W₉ b₉ Wa ba Wb bb) := by
  unfold cifarCnn8Forward
  exact
    (((((((((((((((((((((((
      (floatBridges_flatConv M W₁ b₁ hw₁ hβ₁ (by positivity) hW₁ hb₁)
      |>.comp floatBridges_relu)
      |>.comp (floatBridges_flatConv M W₂ b₂ hw₂ hβ₂ (by positivity) hW₂ hb₂))
      |>.comp floatBridges_relu)
      |>.comp floatBridges_maxPool)
      |>.comp (floatBridges_flatConv M W₃ b₃ hw₃ hβ₃ (by positivity) hW₃ hb₃))
      |>.comp floatBridges_relu)
      |>.comp (floatBridges_flatConv M W₄ b₄ hw₄ hβ₄ (by positivity) hW₄ hb₄))
      |>.comp floatBridges_relu)
      |>.comp floatBridges_maxPool)
      |>.comp (floatBridges_flatConv M W₅ b₅ hw₅ hβ₅ (by positivity) hW₅ hb₅))
      |>.comp floatBridges_relu)
      |>.comp (floatBridges_flatConv M W₆ b₆ hw₆ hβ₆ (by positivity) hW₆ hb₆))
      |>.comp floatBridges_relu)
      |>.comp floatBridges_maxPool)
      |>.comp (floatBridges_flatConv M W₇ b₇ hw₇ hβ₇ (by positivity) hW₇ hb₇))
      |>.comp floatBridges_relu)
      |>.comp (floatBridges_flatConv M W₈ b₈ hw₈ hβ₈ (by positivity) hW₈ hb₈))
      |>.comp floatBridges_relu)
      |>.comp floatBridges_maxPool)
      |>.comp (floatBridges_dense M W₉ b₉ hw₉ hβ₉ (by positivity) hW₉ hb₉))
      |>.comp floatBridges_relu)
      |>.comp (floatBridges_dense M Wa ba hwa hβa hd1 hWa hba))
      |>.comp floatBridges_relu)
      |>.comp (floatBridges_dense M Wb bb hwb hβb hd1 hWb hbb)

end Proofs
