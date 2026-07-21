import LeanMlir.Proofs.LinBackFloatBridge
import LeanMlir.Proofs.Foundation.CnnChainClose

/-! # ℝ→Float32 bridge for the CNN BACKWARD: maxpool-back + conv-back → whole 8-conv fold

A3 (planning/a3_backward_deepnet_assembly.md): the deep-net backward float story for the
**convolutional** nets. The MLP backward fold (`mlpInputGrad_floatBridges`,
`LinBackFloatBridge.lean`) showed a feed-forward net's input-gradient VJP at a smooth point is
itself a *forward* composition on the cotangent — so it threads the same `FloatBridges.comp`
backbone. This file adds the two CNN-specific backward op bridges and assembles the first conv
witness (`cifar8_grad_floatBridges`, the backward peer of `cifar8_floatBridges`).

* **MaxPool backward** (1a) `maxPoolFlatBack` — the rendered `select_and_scatter`: route each
  saved-input cell's cotangent to its window's arg-max position, 0 elsewhere
  (`maxPoolBackDenote` through the flatten boundary). A masked *gather* — **exact in float**
  (no arithmetic), 1-Lipschitz, magnitude-nonincreasing, modulus `id`. The arg-max index map is
  fixed (the smooth-point `MaxPool2Smooth` assumption: float and real arg-maxes agree), exactly
  like `floatClose_reluMaskBack`'s sign mask.
* **Conv input-VJP** (1b) `convFlatBack` — `dx = convBackDenote W dy`. The codegen emits a
  `convolution(dy, reverse(transpose(W)))`, which `convBackDenote` denotes as a *forward*
  `conv2d (reverseSwap W) 0`. In flat `Vec` space that is literally `flatConv (reverseSwap W) 0`,
  so it float-bridges **for free** via `floatBridges_flatConv` at the reversed kernel
  (`|reverseSwap W| = |W|`) — no new proof, exactly as `floatBridges_linBack` reused
  `floatBridges_dense`.

Capstone `cifar8_grad_floatBridges`: the whole 8-conv CIFAR input-gradient VJP at a smooth point
float-bridges in one `.comp` chain over `convFlatBack` ×8 / `maxPoolFlatBack` ×4 /
`reluMaskBack` ×10 / `dense (transpose ·) 0` ×3 — the exact reverse of `cifarCnn8Forward`, each
op replaced by its backward. Closes under `[propext, Classical.choice, Quot.sound]`.
-/

namespace Proofs

open Proofs.IR

-- ════════════════════════════════════════════════════════════════
-- § 1a. MaxPool backward: the exact `select_and_scatter` (masked gather)
-- ════════════════════════════════════════════════════════════════

/-- **MaxPool backward in flat `Vec` space** — `maxPoolBackDenote x` crossing the flatten
    boundary (`Vec (c·h·w) → Vec (c·(2h)·(2w))`): scatter the pooled cotangent back to each
    window's arg-max input cell, 0 elsewhere. The saved input `x` fixes the arg-max map (the
    smooth-point assumption). The backward of `maxPoolFlat c h w`. -/
noncomputable def maxPoolFlatBack {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
    Vec (c * h * w) → Vec (c * (2*h) * (2*w)) :=
  fun dy => Tensor3.flatten (maxPoolBackDenote x (Tensor3.unflatten dy))

/-- The scatter never increases magnitude: each output cell is some `dy` entry or 0. -/
theorem maxPoolBackDenote_abs_le {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (dy : Tensor3 c h w) {A : ℝ} (hdy : ∀ ci hr wc, |dy ci hr wc| ≤ A)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |maxPoolBackDenote x dy ci hi wi| ≤ A := by
  simp only [maxPoolBackDenote]
  by_cases hcase : MaxPool2IsArgmax x ci hi wi
  · simp only [if_pos hcase]; exact hdy _ _ _
  · simp only [if_neg hcase, abs_zero]
    exact le_trans (abs_nonneg _) (hdy ci (winRow hi) (winCol wi))

/-- The scatter is 1-Lipschitz: at the (fixed) arg-max cells it copies `dt - da`, else 0. -/
theorem maxPoolBackDenote_sub_abs_le {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (dt da : Tensor3 c h w) {e : ℝ}
    (hd : ∀ ci hr wc, |dt ci hr wc - da ci hr wc| ≤ e)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |maxPoolBackDenote x dt ci hi wi - maxPoolBackDenote x da ci hi wi| ≤ e := by
  simp only [maxPoolBackDenote]
  by_cases hcase : MaxPool2IsArgmax x ci hi wi
  · simp only [if_pos hcase]; exact hd _ _ _
  · simp only [if_neg hcase, sub_zero, abs_zero]
    exact le_trans (abs_nonneg _) (hd ci (winRow hi) (winCol wi))

/-- **MaxPool backward is `FloatClose` with modulus `id`** — exact in float (real = float map),
    1-Lipschitz, magnitude-nonincreasing. The pooling peer of `floatClose_reluMaskBack`. -/
theorem floatClose_maxPoolFlatBack {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) (A : ℝ) :
    FloatClose A A (maxPoolFlatBack x) (maxPoolFlatBack x) (fun e => e) := by
  constructor
  · intro v hv i
    have hb : |maxPoolFlatBack x v i| ≤ A := by
      have hdy : ∀ ci hr wc, |Tensor3.unflatten v ci hr wc| ≤ A := fun _ _ _ => hv _
      exact maxPoolBackDenote_abs_le x (Tensor3.unflatten v) hdy _ _ _
    exact ⟨hb, hb⟩
  · intro vt va e _ _ hd i
    have hdt : ∀ ci hr wc,
        |Tensor3.unflatten vt ci hr wc - Tensor3.unflatten va ci hr wc| ≤ e := fun _ _ _ => hd _
    exact maxPoolBackDenote_sub_abs_le x (Tensor3.unflatten vt) (Tensor3.unflatten va) hdt _ _ _

/-- MaxPool backward float-bridges (magnitude-stable, exact). -/
theorem floatBridges_maxPoolBack {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
    FloatBridges (maxPoolFlatBack x) :=
  fun A hA => ⟨A, _, _, hA, floatClose_maxPoolFlatBack x A⟩

-- ════════════════════════════════════════════════════════════════
-- § 1b. Conv input-VJP: a reversed-kernel forward conv = `flatConv (reverseSwap W) 0`
-- ════════════════════════════════════════════════════════════════

/-- **Conv backward in flat `Vec` space** — `dx = convBackDenote W dy`. The emitted
    `convolution(dy, reverse(transpose(W)))` denotes a forward `conv2d (reverseSwap W) 0`, which
    in flat space is `flatConv (reverseSwap W) 0`. The backward of `flatConv W b`
    (`Vec (oc·h·w) → Vec (ic·h·w)`). -/
noncomputable def convFlatBack {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Vec (oc * h * w) → Vec (ic * h * w) :=
  flatConv (h := h) (w := w) (reverseSwap W) (fun _ => 0)

/-- **The conv input-VJP float-bridges** — `convFlatBack W = flatConv (reverseSwap W) 0`, so this
    is `floatBridges_flatConv` at the reversed kernel (`|reverseSwap W o c kh kw| =
    |W c o (kRev kh) (kRev kw)| ≤ w'`) with zero bias. No new proof — the conv analogue of
    `floatBridges_linBack`. -/
theorem floatBridges_convBack {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < oc * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') :
    FloatBridges (convFlatBack (h := h) (w := w) W) := by
  unfold convFlatBack
  exact floatBridges_flatConv (ic := oc) (oc := ic) M (reverseSwap W) (fun _ => 0)
    hw' le_rfl hn (fun o c kh kw => hW c o (kRev kh) (kRev kw)) (fun _ => by simp)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net fold: the 8-conv CIFAR input-gradient VJP
-- ════════════════════════════════════════════════════════════════

/-- The 8-conv CIFAR CNN input-gradient VJP at a smooth point: the **exact reverse** of
    `cifarCnn8Forward`, each op replaced by its backward — `convFlatBack` for each `flatConv`,
    `maxPoolFlatBack` (saved input `xa/xb/xc/xd`) for each `maxPoolFlat`, `reluMaskBack`
    (fixed sign mask `m1..m10`) for each `relu`, `dense (transpose ·) 0` for each dense head
    layer. The conv kinks read off the saved arg-max maps; the ReLU kinks off the sign masks. -/
noncomputable def cifar8InputGrad
    {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (W₂ : Kernel4 c1 c1 kH kW)
    (W₃ : Kernel4 c2 c1 kH kW) (W₄ : Kernel4 c2 c2 kH kW)
    (W₅ : Kernel4 c3 c2 kH kW) (W₆ : Kernel4 c3 c3 kH kW)
    (W₇ : Kernel4 c4 c3 kH kW) (W₈ : Kernel4 c4 c4 kH kW)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses)
    (xa : Tensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))))
    (xb : Tensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))))
    (xc : Tensor3 c3 (2*(2*h)) (2*(2*w)))
    (xd : Tensor3 c4 (2*h) (2*w))
    (m1 m2 : Fin d1 → Prop) [DecidablePred m1] [DecidablePred m2]
    (m3 m4 : Fin (c4 * (2*h) * (2*w)) → Prop) [DecidablePred m3] [DecidablePred m4]
    (m5 m6 : Fin (c3 * (2*(2*h)) * (2*(2*w))) → Prop) [DecidablePred m5] [DecidablePred m6]
    (m7 m8 : Fin (c2 * (2*(2*(2*h))) * (2*(2*(2*w)))) → Prop)
      [DecidablePred m7] [DecidablePred m8]
    (m9 m10 : Fin (c1 * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))) → Prop)
      [DecidablePred m9] [DecidablePred m10] :
    Vec nClasses → Vec (ic * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))) :=
  convFlatBack (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₁
  ∘ reluMaskBack m10
  ∘ convFlatBack (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₂
  ∘ reluMaskBack m9
  ∘ maxPoolFlatBack xa
  ∘ convFlatBack (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₃
  ∘ reluMaskBack m8
  ∘ convFlatBack (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₄
  ∘ reluMaskBack m7
  ∘ maxPoolFlatBack xb
  ∘ convFlatBack (h := 2*(2*h)) (w := 2*(2*w)) W₅
  ∘ reluMaskBack m6
  ∘ convFlatBack (h := 2*(2*h)) (w := 2*(2*w)) W₆
  ∘ reluMaskBack m5
  ∘ maxPoolFlatBack xc
  ∘ convFlatBack (h := 2*h) (w := 2*w) W₇
  ∘ reluMaskBack m4
  ∘ convFlatBack (h := 2*h) (w := 2*w) W₈
  ∘ reluMaskBack m3
  ∘ maxPoolFlatBack xd
  ∘ dense (Mat.transpose W₉) (0 : Vec (c4 * h * w))
  ∘ reluMaskBack m2
  ∘ dense (Mat.transpose Wa) (0 : Vec d1)
  ∘ reluMaskBack m1
  ∘ dense (Mat.transpose Wb) (0 : Vec d1)

/-- **The whole 8-conv CIFAR input-gradient VJP float-bridges.** Assembled in one `.comp` chain
    over the per-op backward bridges — `floatBridges_convBack` (each `flatConv`'s reversed-kernel
    backward), `floatBridges_maxPoolBack` (each `maxPoolFlat`'s exact scatter),
    `floatBridges_reluMaskBack` (each ReLU's exact `selectPos` mask) and `floatBridges_linBack`
    (each dense's `Wᵀ·dy`). The deployed float backward map is within an explicit budget of the
    certified real backward map — the backward peer of `cifar8_floatBridges`; closes under
    `[propext, Classical.choice, Quot.sound]`. -/
theorem cifar8_grad_floatBridges
    {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c1 ic kH kW) (W₂ : Kernel4 c1 c1 kH kW)
    (W₃ : Kernel4 c2 c1 kH kW) (W₄ : Kernel4 c2 c2 kH kW)
    (W₅ : Kernel4 c3 c2 kH kW) (W₆ : Kernel4 c3 c3 kH kW)
    (W₇ : Kernel4 c4 c3 kH kW) (W₈ : Kernel4 c4 c4 kH kW)
    (W₉ : Mat (c4 * h * w) d1) (Wa : Mat d1 d1) (Wb : Mat d1 nClasses)
    (xa : Tensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))))
    (xb : Tensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))))
    (xc : Tensor3 c3 (2*(2*h)) (2*(2*w)))
    (xd : Tensor3 c4 (2*h) (2*w))
    (m1 m2 : Fin d1 → Prop) [DecidablePred m1] [DecidablePred m2]
    (m3 m4 : Fin (c4 * (2*h) * (2*w)) → Prop) [DecidablePred m3] [DecidablePred m4]
    (m5 m6 : Fin (c3 * (2*(2*h)) * (2*(2*w))) → Prop) [DecidablePred m5] [DecidablePred m6]
    (m7 m8 : Fin (c2 * (2*(2*(2*h))) * (2*(2*(2*w)))) → Prop)
      [DecidablePred m7] [DecidablePred m8]
    (m9 m10 : Fin (c1 * (2*(2*(2*(2*h)))) * (2*(2*(2*(2*w))))) → Prop)
      [DecidablePred m9] [DecidablePred m10]
    {w₁ w₂ w₃ w₄ w₅ w₆ w₇ w₈ w₉ wa wb : ℝ}
    (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂) (hw₃ : 0 ≤ w₃) (hw₄ : 0 ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hw₆ : 0 ≤ w₆) (hw₇ : 0 ≤ w₇) (hw₈ : 0 ≤ w₈)
    (hw₉ : 0 ≤ w₉) (hwa : 0 ≤ wa) (hwb : 0 ≤ wb)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w₁) (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w₂)
    (hW₃ : ∀ o c kh kw, |W₃ o c kh kw| ≤ w₃) (hW₄ : ∀ o c kh kw, |W₄ o c kh kw| ≤ w₄)
    (hW₅ : ∀ o c kh kw, |W₅ o c kh kw| ≤ w₅) (hW₆ : ∀ o c kh kw, |W₆ o c kh kw| ≤ w₆)
    (hW₇ : ∀ o c kh kw, |W₇ o c kh kw| ≤ w₇) (hW₈ : ∀ o c kh kw, |W₈ o c kh kw| ≤ w₈)
    (hW₉ : ∀ i j, |W₉ i j| ≤ w₉) (hWa : ∀ i j, |Wa i j| ≤ wa) (hWb : ∀ i j, |Wb i j| ≤ wb)
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hc3 : 0 < c3) (hc4 : 0 < c4)
    (hd1 : 0 < d1) (hnc : 0 < nClasses) (hh : 0 < h) (hw : 0 < w) :
    FloatBridges (cifar8InputGrad W₁ W₂ W₃ W₄ W₅ W₆ W₇ W₈ W₉ Wa Wb
      xa xb xc xd m1 m2 m3 m4 m5 m6 m7 m8 m9 m10) := by
  unfold cifar8InputGrad
  exact
    (((((((((((((((((((((((
      (floatBridges_linBack M Wb hwb hnc hWb)
      |>.comp (floatBridges_reluMaskBack m1))
      |>.comp (floatBridges_linBack M Wa hwa hd1 hWa))
      |>.comp (floatBridges_reluMaskBack m2))
      |>.comp (floatBridges_linBack M W₉ hw₉ hd1 hW₉))
      |>.comp (floatBridges_maxPoolBack xd))
      |>.comp (floatBridges_reluMaskBack m3))
      |>.comp (floatBridges_convBack M W₈ hw₈ (by positivity) hW₈))
      |>.comp (floatBridges_reluMaskBack m4))
      |>.comp (floatBridges_convBack M W₇ hw₇ (by positivity) hW₇))
      |>.comp (floatBridges_maxPoolBack xc))
      |>.comp (floatBridges_reluMaskBack m5))
      |>.comp (floatBridges_convBack M W₆ hw₆ (by positivity) hW₆))
      |>.comp (floatBridges_reluMaskBack m6))
      |>.comp (floatBridges_convBack M W₅ hw₅ (by positivity) hW₅))
      |>.comp (floatBridges_maxPoolBack xb))
      |>.comp (floatBridges_reluMaskBack m7))
      |>.comp (floatBridges_convBack M W₄ hw₄ (by positivity) hW₄))
      |>.comp (floatBridges_reluMaskBack m8))
      |>.comp (floatBridges_convBack M W₃ hw₃ (by positivity) hW₃))
      |>.comp (floatBridges_maxPoolBack xa))
      |>.comp (floatBridges_reluMaskBack m9))
      |>.comp (floatBridges_convBack M W₂ hw₂ (by positivity) hW₂))
      |>.comp (floatBridges_reluMaskBack m10))
      |>.comp (floatBridges_convBack M W₁ hw₁ (by positivity) hW₁)

-- ════════════════════════════════════════════════════════════════
-- § The BatchNorm CIFAR input-gradient VJP (BN-back maps supplied as bridges)
-- ════════════════════════════════════════════════════════════════

/-- The BatchNorm CIFAR CNN input-gradient VJP at a smooth point: the **exact reverse** of
    `cifarCnnBnForward`, each op replaced by its backward. Each conv→BN→ReLU block reverses to
    `convFlatBack ∘ bnBack ∘ reluMaskBack`; the BN-backward maps `bnB1..bnB4` are supplied
    abstractly (discharged by the per-channel BN-back bridge, exactly as `cifarBn_floatBridges`
    supplies the four forward BNs as `FloatBridges` hypotheses). Dense head + maxpools mirror
    `cifar8InputGrad`. -/
noncomputable def cifarBnInputGrad
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (W₂ : Kernel4 c1 c1 kH kW)
    (W₃ : Kernel4 c2 c1 kH kW) (W₄ : Kernel4 c2 c2 kH kW)
    (W₅ : Mat (c2 * h * w) d1) (W₆ : Mat d1 d1) (W₇ : Mat d1 nClasses)
    (bnB1 bnB2 : Vec (c1 * (2*(2*h)) * (2*(2*w))) → Vec (c1 * (2*(2*h)) * (2*(2*w))))
    (bnB3 bnB4 : Vec (c2 * (2*h) * (2*w)) → Vec (c2 * (2*h) * (2*w)))
    (xp1 : Tensor3 c1 (2*(2*h)) (2*(2*w))) (xp2 : Tensor3 c2 (2*h) (2*w))
    (qA qB : Fin (c1 * (2*(2*h)) * (2*(2*w))) → Prop) [DecidablePred qA] [DecidablePred qB]
    (qC qD : Fin (c2 * (2*h) * (2*w)) → Prop) [DecidablePred qC] [DecidablePred qD]
    (p5 p6 : Fin d1 → Prop) [DecidablePred p5] [DecidablePred p6] :
    Vec nClasses → Vec (ic * (2*(2*h)) * (2*(2*w))) :=
  convFlatBack (h := 2*(2*h)) (w := 2*(2*w)) W₁
  ∘ bnB1
  ∘ reluMaskBack qA
  ∘ convFlatBack (h := 2*(2*h)) (w := 2*(2*w)) W₂
  ∘ bnB2
  ∘ reluMaskBack qB
  ∘ maxPoolFlatBack xp1
  ∘ convFlatBack (h := 2*h) (w := 2*w) W₃
  ∘ bnB3
  ∘ reluMaskBack qC
  ∘ convFlatBack (h := 2*h) (w := 2*w) W₄
  ∘ bnB4
  ∘ reluMaskBack qD
  ∘ maxPoolFlatBack xp2
  ∘ dense (Mat.transpose W₅) (0 : Vec (c2 * h * w))
  ∘ reluMaskBack p5
  ∘ dense (Mat.transpose W₆) (0 : Vec d1)
  ∘ reluMaskBack p6
  ∘ dense (Mat.transpose W₇) (0 : Vec d1)

/-- **The BatchNorm CIFAR input-gradient VJP float-bridges.** One `.comp` chain over the per-op
    backward bridges — `convFlatBack` ×4 / `maxPoolFlatBack` ×2 / `reluMaskBack` ×6 /
    `dense (transpose ·) 0` ×3 — with the four BatchNorm-backward maps supplied as `FloatBridges`
    facts (discharge each with `floatBridges_bnBack`/the per-channel lift, exactly as
    `cifarBn_floatBridges` supplies the forward BNs). The backward peer of `cifarBn_floatBridges`;
    closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem cifarBn_grad_floatBridges
    {ic c1 c2 h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c1 ic kH kW) (W₂ : Kernel4 c1 c1 kH kW)
    (W₃ : Kernel4 c2 c1 kH kW) (W₄ : Kernel4 c2 c2 kH kW)
    (W₅ : Mat (c2 * h * w) d1) (W₆ : Mat d1 d1) (W₇ : Mat d1 nClasses)
    (bnB1 bnB2 : Vec (c1 * (2*(2*h)) * (2*(2*w))) → Vec (c1 * (2*(2*h)) * (2*(2*w))))
    (bnB3 bnB4 : Vec (c2 * (2*h) * (2*w)) → Vec (c2 * (2*h) * (2*w)))
    (xp1 : Tensor3 c1 (2*(2*h)) (2*(2*w))) (xp2 : Tensor3 c2 (2*h) (2*w))
    (qA qB : Fin (c1 * (2*(2*h)) * (2*(2*w))) → Prop) [DecidablePred qA] [DecidablePred qB]
    (qC qD : Fin (c2 * (2*h) * (2*w)) → Prop) [DecidablePred qC] [DecidablePred qD]
    (p5 p6 : Fin d1 → Prop) [DecidablePred p5] [DecidablePred p6]
    {w₁ w₂ w₃ w₄ w₅ w₆ w₇ : ℝ}
    (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂) (hw₃ : 0 ≤ w₃) (hw₄ : 0 ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hw₆ : 0 ≤ w₆) (hw₇ : 0 ≤ w₇)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w₁) (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w₂)
    (hW₃ : ∀ o c kh kw, |W₃ o c kh kw| ≤ w₃) (hW₄ : ∀ o c kh kw, |W₄ o c kh kw| ≤ w₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hW₆ : ∀ i j, |W₆ i j| ≤ w₆) (hW₇ : ∀ i j, |W₇ i j| ≤ w₇)
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hd1 : 0 < d1) (hnc : 0 < nClasses) (hh : 0 < h) (hw : 0 < w)
    (hbnB1 : FloatBridges bnB1) (hbnB2 : FloatBridges bnB2)
    (hbnB3 : FloatBridges bnB3) (hbnB4 : FloatBridges bnB4) :
    FloatBridges (cifarBnInputGrad W₁ W₂ W₃ W₄ W₅ W₆ W₇ bnB1 bnB2 bnB3 bnB4
      xp1 xp2 qA qB qC qD p5 p6) := by
  unfold cifarBnInputGrad
  exact
    (((((((((((((((((
      (floatBridges_linBack M W₇ hw₇ hnc hW₇)
      |>.comp (floatBridges_reluMaskBack p6))
      |>.comp (floatBridges_linBack M W₆ hw₆ hd1 hW₆))
      |>.comp (floatBridges_reluMaskBack p5))
      |>.comp (floatBridges_linBack M W₅ hw₅ hd1 hW₅))
      |>.comp (floatBridges_maxPoolBack xp2))
      |>.comp (floatBridges_reluMaskBack qD))
      |>.comp hbnB4)
      |>.comp (floatBridges_convBack M W₄ hw₄ (by positivity) hW₄))
      |>.comp (floatBridges_reluMaskBack qC))
      |>.comp hbnB3)
      |>.comp (floatBridges_convBack M W₃ hw₃ (by positivity) hW₃))
      |>.comp (floatBridges_maxPoolBack xp1))
      |>.comp (floatBridges_reluMaskBack qB))
      |>.comp hbnB2)
      |>.comp (floatBridges_convBack M W₂ hw₂ (by positivity) hW₂))
      |>.comp (floatBridges_reluMaskBack qA))
      |>.comp hbnB1)
      |>.comp (floatBridges_convBack M W₁ hw₁ (by positivity) hW₁)

end Proofs
