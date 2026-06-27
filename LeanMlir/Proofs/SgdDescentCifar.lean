import LeanMlir.Proofs.SgdDescentCnn
import LeanMlir.Proofs.CifarCNN

/-! # CIFAR-8 last-conv SGD descent — the first non-MNIST provable descent (A2 probe)

`planning/tier23_float_and_syntactic_faithfulness.md` A2 asked the genuinely-uncertain question: does
the segment-Lipschitz SGD-descent argument (proven for the MNIST CNN, `SgdDescentCnn.lean`) reach a
CIFAR net? This file answers it.

**The finding.** CIFAR-8's *tail* — its last conv `W₈` (c4→c4) → relu → maxpool → three denses → CE —
is **byte-for-byte** the program `cnn_conv2_sgd_descends` proves descent for. So descent at the LAST
conv layer reaches CIFAR-8 **for free**, with the SAME non-vacuous admissible `lr` as MNIST: it is an
*instance* of the MNIST lemma at the frozen earlier-layer features. Made rigorous in two steps:

* `cifarCnn8Forward_factor` — the actual committed net factors as
  `head ∘ (relu ∘ flatConv W₈) ∘ prefix7` (pure `rfl`; `Function.comp` is definitionally associative).
* `cifar8_lastConv_sgd_descends` — one SGD step on `W₈` (the earlier seven conv layers held fixed, their
  output on `image` being the frozen feature map `x₁`) decreases the CIFAR-8 cross-entropy by `≥
  lr·‖∇‖²/2`. Proved by reducing the CIFAR-8 loss-as-a-function-of-`W₈` to the `cnn_conv2` program at
  `x₁` (`hfac`, via the factor lemma + `flatConv = flatten∘conv2d∘unflatten`) and applying
  `cnn_conv2_sgd_descends`.

**The honest stop (why this is the ceiling).** Descent through the *depth* of all eight conv layers is
NOT proved, by design: `cnn_conv2_sgd_descends`'s admissible-`lr` condition `hsmall` is a PRODUCT of the
per-layer operator-norm factors (the three dense bounds × spatial). Each additional conv layer would
multiply another `(spatial · weight-bound)` factor into that product, so the admissible `lr` shrinks
geometrically with depth ⇒ vacuous in any realistic regime. This is the SAME compounding mechanism that
puts deep-net descent off-limits. So last-conv descent is the honest reach of provable descent for
CIFAR; full-depth / end-to-end CIFAR descent stays open.
-/

namespace Proofs

/-- The CIFAR-8 classifier head (everything after the last conv's relu): maxpool → 3 denses. -/
noncomputable def cifar8Head {c4 h w d1 nClasses : Nat}
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) :
    Vec (c4 * (2*h) * (2*w)) → Vec nClasses :=
  dense Wb bb
  ∘ (relu d1 ∘ dense Wa ba)
  ∘ (relu d1 ∘ dense W₉ b₉)
  ∘ maxPoolFlat c4 h w

/-- The CIFAR-8 first-7-conv feature extractor (everything before the last conv W₈). -/
noncomputable def cifar8Prefix7 {ic c1 c2 c3 c4 h w kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) :
    Vec (ic * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))) → Vec (c4 * (2*h) * (2*w)) :=
  (relu (c4 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₇ b₇)
  ∘ maxPoolFlat c3 (2*h) (2*w)
  ∘ (relu (c3 * (2*(2*h)) * (2*(2*w))) ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₆ b₆)
  ∘ (relu (c3 * (2*(2*h)) * (2*(2*w))) ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₅ b₅)
  ∘ maxPoolFlat c2 (2*(2*h)) (2*(2*w))
  ∘ (relu (c2 * (2*(2*(2*h))) * (2*(2*(2*w)))) ∘ flatConv (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₄ b₄)
  ∘ (relu (c2 * (2*(2*(2*h))) * (2*(2*(2*w)))) ∘ flatConv (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₃ b₃)
  ∘ maxPoolFlat c1 (2*(2*(2*h))) (2*(2*(2*w)))
  ∘ (relu (c1 * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))) ∘ flatConv (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₂ b₂)
  ∘ (relu (c1 * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))) ∘ flatConv (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₁ b₁)

/-- **CIFAR-8 factors at the last conv** — `cifarCnn8Forward = head ∘ (relu ∘ flatConv W₈) ∘ prefix7`.
    Pure `rfl` (same `∘`-chain, regrouped; `Function.comp` is definitionally associative). -/
theorem cifarCnn8Forward_factor {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
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
    (Wb : Mat d1 nClasses) (bb : Vec nClasses) :
    cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb
      = cifar8Head W₉ b₉ Wa ba Wb bb
        ∘ (relu (c4 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₈ b₈)
        ∘ cifar8Prefix7 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ := rfl

set_option maxHeartbeats 1000000 in
/-- **CIFAR-8 last-conv SGD descent.** One SGD step on the LAST conv `W₈` of the actual
    `cifarCnn8Forward` net (the earlier seven conv layers held fixed — their output on `image` is the
    frozen feature map `x₁`) decreases the CIFAR-8 cross-entropy loss by at least `lr·‖∇‖²/2`, under
    the segment-margin conditions that freeze the ReLU/MaxPool routing along the step. Because
    CIFAR-8's tail (`W₈` → relu → maxpool → 3 denses) is byte-for-byte the architecture
    `cnn_conv2_sgd_descends` proves descent for, this is an INSTANCE of that lemma at the frozen
    features `x₁`, via `cifarCnn8Forward_factor` — the admissible `lr` is the same non-vacuous MNIST
    regime. The genuinely-distinct case (descent through the DEPTH of all eight conv layers) stays
    open by design: each extra layer multiplies another operator-norm factor into `hsmall`'s
    admissible-`lr` product, so it compounds to vacuity — the same honest stop as the deep nets. -/
theorem cifar8_lastConv_sgd_descends {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
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
    (image : Vec (ic * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))))
    (x₁ : Tensor3 c4 (2*h) (2*w)) (label : Fin nClasses) (gh : Vec (c4 * c4 * kH * kW))
    (hx₁ : x₁ = Tensor3.unflatten
      (cifar8Prefix7 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ image))
    (hc4 : 0 < c4) (hh : 0 < h) (hw : 0 < w)
    {lr η a w₉ wa wb : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₉ : 0 ≤ w₉) (hW₉ : ∀ i j, |W₉ i j| ≤ w₉)
    (hwa : 0 ≤ wa) (hWa : ∀ i j, |Wa i j| ≤ wa)
    (hwb : 0 ≤ wb) (hWb : ∀ i j, |Wb i j| ≤ wb)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (fun v' : Vec (c4 * c4 * kH * kW) =>
        crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
          (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
          label) (Kernel4.flatten W₈) idx| ≤ η)
    (hm2 : ∀ k, a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx|) +
        ((c4 * c4 * kH * kW : ℕ) : ℝ) * η)) <
      |Tensor3.flatten (conv2d W₈ b₈ x₁) k|)
    (hmq : MaxPool2MarginQ (a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx|) +
        ((c4 * c4 * kH * kW : ℕ) : ℝ) * η)))
      (Tensor3.unflatten (relu (c4 * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₈ b₈ x₁)))))
    (hm3 : ∀ l, w₉ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
        |gradAt (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx|) +
        ((c4 * c4 * kH * kW : ℕ) : ℝ) * η)))) <
      |dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₈ b₈ x₁)))) l|)
    (hm4 : ∀ q, wa * ((d1 : ℝ) * (w₉ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (lr * ((∑ idx, |gradAt
          (fun v' : Vec (c4 * c4 * kH * kW) =>
            crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
              (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₈ x₁))))))))) label)
            (Kernel4.flatten W₈) idx|) +
          ((c4 * c4 * kH * kW : ℕ) : ℝ) * η)))))) <
      |dense Wa ba (relu d1 (dense W₉ b₉ (maxPoolFlat c4 h w
        (relu (c4 * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₈ b₈ x₁)))))) q|)
    (hsmall : 2 * (wb * ((d1 : ℝ) * (wa * ((d1 : ℝ) * (w₉ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx|) +
        ((c4 * c4 * kH * kW : ℕ) : ℝ) * η))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx|) ≤
      lr * (∑ idx, gradAt
        (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx ^ 2) / 4)
    (h2 : (2 * (nClasses : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d1 : ℝ) ^ 2 *
        (d1 : ℝ) ^ 2 * w₉ ^ 2 * wa ^ 2 * wb ^ 2 * a ^ 2 /
        (1 - 2 * (wb * ((d1 : ℝ) * (wa * ((d1 : ℝ) * (w₉ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt
            (fun v' : Vec (c4 * c4 * kH * kW) =>
              crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
                (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
                  (Tensor3.flatten
                    (conv2d (Kernel4.unflatten v') b₈ x₁))))))))) label)
              (Kernel4.flatten W₈) idx|) +
            ((c4 * c4 * kH * kW : ℕ) : ℝ) * η))))))))))) *
        (lr * ((∑ idx, |gradAt
          (fun v' : Vec (c4 * c4 * kH * kW) =>
            crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
              (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₈ x₁))))))))) label)
            (Kernel4.flatten W₈) idx|) +
          ((c4 * c4 * kH * kW : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun v' : Vec (c4 * c4 * kH * kW) =>
          crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁)))))))))
            label) (Kernel4.flatten W₈) idx ^ 2) / 4) :
    crossEntropy nClasses (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
        (Kernel4.unflatten (Kernel4.flatten W₈ - lr • gh)) b₈ W₉ b₉ Wa ba Wb bb image) label ≤
      crossEntropy nClasses (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
        (Kernel4.unflatten (Kernel4.flatten W₈)) b₈ W₉ b₉ Wa ba Wb bb image) label -
        lr * (∑ idx, gradAt
          (fun v' : Vec (c4 * c4 * kH * kW) =>
            crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
              (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
                (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁))))))))) label)
            (Kernel4.flatten W₈) idx ^ 2) / 2 := by
  -- the CIFAR-8 loss as a function of the last conv's weights = the cnn_conv2 program at `x₁`
  have hfac : ∀ v' : Vec (c4 * c4 * kH * kW),
      crossEntropy nClasses (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
          (Kernel4.unflatten v') b₈ W₉ b₉ Wa ba Wb bb image) label
        = crossEntropy nClasses (dense Wb bb (relu d1 (dense Wa ba (relu d1
            (dense W₉ b₉ (maxPoolFlat c4 h w (relu (c4 * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ x₁))))))))) label := by
    intro v'
    rw [cifarCnn8Forward_factor]
    simp only [Function.comp_apply, cifar8Head, flatConv, hx₁]
  rw [hfac (Kernel4.flatten W₈ - lr • gh), hfac (Kernel4.flatten W₈)]
  exact cnn_conv2_sgd_descends W₈ b₈ x₁ W₉ b₉ Wa ba Wb bb label gh hc4 hh hw
    ha hx hw₉ hW₉ hwa hWa hwb hWb hlr hη hgh hm2 hmq hm3 hm4 hsmall h1 h2

end Proofs
