import LeanMlir.Proofs.Float.BnPerChannelFloatBridge
import LeanMlir.Proofs.Architectures.CifarCNN

/-! # Whole-net forward float-bridge for the BatchNorm CIFAR CNN

A1 capstone (planning/tier23_float_and_syntactic_faithfulness.md). The no-BN CIFAR
forwards already bridge (`FloatModel.cifar_float_close` for the 4-conv,
`Proofs.cifar8_floatBridges` for the 8-conv). This file lands the **BatchNorm** CIFAR
forward `cifarCnnBnForward` — the one with per-channel `bnPerChannelTensor3` between
each conv and ReLU.

The four BatchNorm layers enter as `FloatBridges (bnPerChannelTensor3 …)` hypotheses,
exactly as `floatBridges_mbconvBody` (EfficientNet) takes its three `hbnE/hbnD/hbnP`.
Each is now discharged by `floatBridges_bnPerChannelTensor3`
(`BnPerChannelFloatBridge.lean`) once the per-channel float-stat accuracy moduli are
supplied — so this capstone is the pure `FloatBridges.comp` *assembly* over the
conv / per-channel-BN / ReLU / maxpool / dense per-op bridges, magnitudes threaded
automatically. The BN-net peer of `cifar8_floatBridges`.
-/

namespace Proofs

/-- **The BatchNorm CIFAR CNN float-bridges.** For any input magnitude there is an
    output magnitude and a `FloatClose` certificate, assembled in one `.comp` chain
    over the per-op bridges (conv / per-channel BN / ReLU / maxpool / dense). The four
    per-channel BatchNorms are supplied as `FloatBridges` facts (discharge each with
    `floatBridges_bnPerChannelTensor3`). Closes under
    `[propext, Classical.choice, Quot.sound]`. -/
theorem cifarBn_floatBridges
    {ic c1 c2 h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    {w' β : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w') (hb₁ : ∀ o, |b₁ o| ≤ β)
    (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w') (hb₂ : ∀ o, |b₂ o| ≤ β)
    (hW₃ : ∀ o c kh kw, |W₃ o c kh kw| ≤ w') (hb₃ : ∀ o, |b₃ o| ≤ β)
    (hW₄ : ∀ o c kh kw, |W₄ o c kh kw| ≤ w') (hb₄ : ∀ o, |b₄ o| ≤ β)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w') (hb₅ : ∀ j, |b₅ j| ≤ β)
    (hW₆ : ∀ i j, |W₆ i j| ≤ w') (hb₆ : ∀ j, |b₆ j| ≤ β)
    (hW₇ : ∀ i j, |W₇ i j| ≤ w') (hb₇ : ∀ j, |b₇ j| ≤ β)
    (hic : 0 < ic) (hc1 : 0 < c1) (hc2 : 0 < c2) (hd1 : 0 < d1) (hh : 0 < h) (hw : 0 < w)
    (hbn₁ : FloatBridges (bnPerChannelTensor3 c1 (2*(2*h)) (2*(2*w)) ε₁ γ₁ β₁))
    (hbn₂ : FloatBridges (bnPerChannelTensor3 c1 (2*(2*h)) (2*(2*w)) ε₂ γ₂ β₂))
    (hbn₃ : FloatBridges (bnPerChannelTensor3 c2 (2*h) (2*w) ε₃ γ₃ β₃))
    (hbn₄ : FloatBridges (bnPerChannelTensor3 c2 (2*h) (2*w) ε₄ γ₄ β₄)) :
    FloatBridges (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
      W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇) := by
  unfold cifarCnnBnForward
  exact
    (floatBridges_flatConv (h := 2*(2*h)) (w := 2*(2*w)) M W₁ b₁ hw' hβ (by positivity) hW₁ hb₁
      |>.comp hbn₁
      |>.comp floatBridges_relu
      |>.comp (floatBridges_flatConv (h := 2*(2*h)) (w := 2*(2*w)) M W₂ b₂ hw' hβ
        (by positivity) hW₂ hb₂)
      |>.comp hbn₂
      |>.comp floatBridges_relu
      |>.comp floatBridges_maxPool
      |>.comp (floatBridges_flatConv (h := 2*h) (w := 2*w) M W₃ b₃ hw' hβ (by positivity) hW₃ hb₃)
      |>.comp hbn₃
      |>.comp floatBridges_relu
      |>.comp (floatBridges_flatConv (h := 2*h) (w := 2*w) M W₄ b₄ hw' hβ (by positivity) hW₄ hb₄)
      |>.comp hbn₄
      |>.comp floatBridges_relu
      |>.comp floatBridges_maxPool
      |>.comp (floatBridges_dense M W₅ b₅ hw' hβ (by positivity) hW₅ hb₅)
      |>.comp floatBridges_relu
      |>.comp (floatBridges_dense M W₆ b₆ hw' hβ hd1 hW₆ hb₆)
      |>.comp floatBridges_relu
      |>.comp (floatBridges_dense M W₇ b₇ hw' hβ hd1 hW₇ hb₇))

end Proofs
