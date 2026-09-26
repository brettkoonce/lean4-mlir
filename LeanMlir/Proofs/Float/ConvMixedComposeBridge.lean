import LeanMlir.Proofs.Float.FloatComposeBridge
import LeanMlir.Proofs.Float.ConvMixedFloatBridge

/-! # The mixed-precision conv as a `FloatClose`

`FloatModel.conv_close_mixed` bounds **one** bf16-mixed convolution against exact ℝ at
an **exactly-represented input**. That is not enough to compose: a net feeds each layer the
*previous* layer's already-perturbed output, so what a fold needs is an error **modulus** — a map
from inherited input error to output error — plus a magnitude bound to thread forward. That pair
is `FloatClose` (`FloatComposeBridge.lean`), and this file supplies its mixed-precision conv instance.

`FloatClose A B f fF L` says nothing about how `fF` rounds — only that it stays within
`L e` of `f`. So `floatClose_relu`, `floatClose_bn`, `floatClose_maxPool3s2`, `floatClose_gap`,
`floatClose_residualBlock`, `floatClose_iterate` and `FloatClose.comp` compose with the bf16
conv instance here unchanged. No whole-net bf16 (or f32) float bound is assembled in the repo.

What genuinely had to be proved here, none of which the `e = 0` bound gives:

1. `convFanS_le` — the data-dependent `Σ|kernel·window|` replaced by the closed form
   `n·w·A`, so the budget is a formula in dims and norms rather than in the input.
2. `conv2d_sub_abs_le` — the REAL conv is `n·w`-Lipschitz in its input. This is the term that
   carries a predecessor's error through the layer, and it has no analogue at `e = 0`.
3. `convMixedBudget` / `convMixed_close_prop` — the two combined, at an input that is both
   perturbed (`E`) and magnitude-bounded (`A`).

Note: the budget is evaluated at `A + E`, not `A`. The float conv runs on the perturbed input, so
its own rounding scales with the perturbed magnitude; only the real conv sees `A`. Writing `A`
there would understate the bound.

`n = ic·kH·kW` throughout. The fan-in amplification rides `uacc` (fp32) while `uleaf` (bf16)
enters flat, so the per-layer relative factor is `1 + O(u)`; composed depth-first, the bound is
still vacuous in absolute terms (see `convMixedGain_factor`).
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § `Tensor3.flatten` transports pointwise bounds
-- ════════════════════════════════════════════════════════════════

/-- `flatten` is a coordinate LOOKUP, so any pointwise bound on the tensor is a pointwise
    bound on the flattened vector. -/
theorem Tensor3.flatten_abs_le {c h w : Nat} {T : Tensor3 c h w} {C : ℝ}
    (hT : ∀ i j l, |T i j l| ≤ C) (k : Fin (c * h * w)) :
    |Tensor3.flatten T k| ≤ C := by
  simp only [Tensor3.flatten]; exact hT _ _ _

/-- The same for a DIFFERENCE of two tensors — `flatten` is linear because it is a lookup. -/
theorem Tensor3.flatten_sub_abs_le {c h w : Nat} {T S : Tensor3 c h w} {C : ℝ}
    (hTS : ∀ i j l, |T i j l - S i j l| ≤ C) (k : Fin (c * h * w)) :
    |Tensor3.flatten T k - Tensor3.flatten S k| ≤ C := by
  simp only [Tensor3.flatten]; exact hTS _ _ _

-- ════════════════════════════════════════════════════════════════
-- § The receptive window: magnitude and perturbation
-- ════════════════════════════════════════════════════════════════

/-- The window inherits the input's magnitude bound — the padded branch is `0`, which needs
    `0 ≤ A` rather than the hypothesis. -/
theorem convWindow3_abs_le {ic h w kH kW : Nat} {x : Tensor3 ic h w} {A : ℝ}
    (hA : 0 ≤ A) (hx : ∀ c i j, |x c i j| ≤ A) (hi : Fin h) (wi : Fin w) :
    ∀ c kh kw, |convWindow3 kH kW x hi wi c kh kw| ≤ A := by
  intro c kh kw
  simp only [convWindow3, convPad]
  split
  · exact hx _ _ _
  · simpa using hA

/-- The window inherits the input's PERTURBATION. The padding branch is the same branch for
    both tensors (it depends only on the indices), so it contributes `|0 - 0| = 0 ≤ E`. -/
theorem convWindow3_sub_abs_le {ic h w kH kW : Nat} {xt xa : Tensor3 ic h w} {E : ℝ}
    (hE : 0 ≤ E) (hd : ∀ c i j, |xt c i j - xa c i j| ≤ E) (hi : Fin h) (wi : Fin w) :
    ∀ c kh kw, |convWindow3 kH kW xt hi wi c kh kw
                 - convWindow3 kH kW xa hi wi c kh kw| ≤ E := by
  intro c kh kw
  simp only [convWindow3, convPad]
  split
  · exact hd _ _ _
  · simpa using hE

-- ════════════════════════════════════════════════════════════════
-- § `convFanS` in closed form
-- ════════════════════════════════════════════════════════════════

/-- **The data-dependent fan-in sum, bounded by dims and norms.** `conv_close_mixed` scales
    everything by `convFanS W x o hi wi = Σ|kernel·window|`; this replaces it by `n·w·A`, which
    is what turns that theorem into a budget a fold can carry. -/
theorem convFanS_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {x : Tensor3 ic h w} {w' A : ℝ} (hw' : 0 ≤ w') (hA : 0 ≤ A)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hx : ∀ c i j, |x c i j| ≤ A)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    convFanS W x o hi wi ≤ ((ic * kH * kW : ℕ) : ℝ) * w' * A := by
  have hstep : ∀ k : Fin (ic * kH * kW),
      |Tensor3.flatten (convSlice W o) k * Tensor3.flatten (convWindow3 kH kW x hi wi) k|
        ≤ w' * A := by
    intro k
    rw [abs_mul]
    exact mul_le_mul
      (Tensor3.flatten_abs_le (fun c kh kw => hW o c kh kw) k)
      (Tensor3.flatten_abs_le (convWindow3_abs_le hA hx hi wi) k)
      (abs_nonneg _) hw'
  exact (Finset.sum_le_card_nsmul _ _ _ fun k _ => hstep k).trans_eq (by simp [mul_assoc])

-- ════════════════════════════════════════════════════════════════
-- § The real convolution is Lipschitz in its input
-- ════════════════════════════════════════════════════════════════

/-- **`conv2d` is `n·w`-Lipschitz.** THE term with no analogue at `e = 0`: it is how a
    predecessor layer's error reaches this layer's output. The bias cancels (it is the same in
    both), so the difference is one dot product against the window difference. -/
theorem conv2d_sub_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW} {b : Vec oc}
    {xt xa : Tensor3 ic h w} {w' E : ℝ} (hw' : 0 ≤ w') (hE : 0 ≤ E)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    (hd : ∀ c i j, |xt c i j - xa c i j| ≤ E)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d W b xt o hi wi - conv2d W b xa o hi wi| ≤ ((ic * kH * kW : ℕ) : ℝ) * w' * E := by
  rw [conv2d_eq_flat_dot, conv2d_eq_flat_dot]
  have hrw : (b o + ∑ k, Tensor3.flatten (convSlice W o) k
                * Tensor3.flatten (convWindow3 kH kW xt hi wi) k)
           - (b o + ∑ k, Tensor3.flatten (convSlice W o) k
                * Tensor3.flatten (convWindow3 kH kW xa hi wi) k)
      = ∑ k, Tensor3.flatten (convSlice W o) k
          * (Tensor3.flatten (convWindow3 kH kW xt hi wi) k
             - Tensor3.flatten (convWindow3 kH kW xa hi wi) k) := by
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  rw [hrw]
  calc |∑ k, Tensor3.flatten (convSlice W o) k
          * (Tensor3.flatten (convWindow3 kH kW xt hi wi) k
             - Tensor3.flatten (convWindow3 kH kW xa hi wi) k)|
      ≤ ∑ k, |Tensor3.flatten (convSlice W o) k
          * (Tensor3.flatten (convWindow3 kH kW xt hi wi) k
             - Tensor3.flatten (convWindow3 kH kW xa hi wi) k)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ((ic * kH * kW : ℕ) : ℝ) * w' * E :=
        (Finset.sum_le_card_nsmul _ _ _ fun k _ => (abs_mul _ _).trans_le (mul_le_mul
          (Tensor3.flatten_abs_le (fun c kh kw => hW o c kh kw) k)
          (Tensor3.flatten_sub_abs_le (convWindow3_sub_abs_le hE hd hi wi) k)
          (abs_nonneg _) hw')).trans_eq (by simp [mul_assoc])

-- ════════════════════════════════════════════════════════════════
-- § The mixed-precision conv budget, with an INHERITED error
-- ════════════════════════════════════════════════════════════════

/-- **The mixed-precision conv budget — the `layerBudget` peer, and the object this whole
    file exists to produce.** Four terms:

    * `uacc * (… + β)` — the f32 bias add,
    * `uleaf * (1+br) * …` — the **bf16 store** of the accumulator (the bf16-TYPED conv result,
      forced by the only emit shape that reaches tensor cores),
    * `br * …` — the dot itself, fan-in `n` amplified at the ACCUMULATE precision,
    * `n·w·E` — **the inherited error**, carried through by the real conv's Lipschitz constant.

    Note: The first three are evaluated at `A + E`, not `A`: the float conv runs on the PERTURBED
    input, so its own rounding scales with the perturbed magnitude. Only the fourth term is
    linear in `E` alone. -/
noncomputable def convMixedBudget (uacc uleaf : ℝ) (n : ℕ) (w β A E : ℝ) : ℝ :=
  uacc * ((1 + uleaf) * (1 + convBrR uacc uleaf n) * ((n : ℝ) * w * (A + E)) + β)
    + uleaf * (1 + convBrR uacc uleaf n) * ((n : ℝ) * w * (A + E))
    + convBrR uacc uleaf n * ((n : ℝ) * w * (A + E))
    + (n : ℝ) * w * E

-- ════════════════════════════════════════════════════════════════
-- § The propagating bound
-- ════════════════════════════════════════════════════════════════

/-- **Mixed-precision convolution against exact ℝ at a PERTURBED input.** The composable
    peer of `conv_close_mixed`, which is this at `E = 0` and with the data-dependent `convFanS`
    left in place.

    Two steps: the float conv is `conv_close_mixed` at its OWN input `xt` (whose magnitude is
    `A + E`), and the real conv moves from `xt` to `xa` by `conv2d_sub_abs_le`. -/
theorem FloatModel.convMixed_close_prop (M L : FloatModel) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (xt xa : Tensor3 ic h w)
    {w' β A E : ℝ} (hw' : 0 ≤ w') (hA : 0 ≤ A) (hE : 0 ≤ E)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hxa : ∀ c i j, |xa c i j| ≤ A) (hd : ∀ c i j, |xt c i j - xa c i j| ≤ E)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |M.convMixed L W b xt o hi wi - conv2d W b xa o hi wi|
      ≤ convMixedBudget M.u L.u (ic * kH * kW) w' β A E := by
  have hMu := M.u_nonneg
  have hLu := L.u_nonneg
  set n := ic * kH * kW with hndef
  set br := convBrR M.u L.u n with hbrdef
  have hbr0 : 0 ≤ br := convBrR_nonneg hMu hLu n
  -- the float conv's own input magnitude is A + E, not A
  have hxt : ∀ c i j, |xt c i j| ≤ A + E := fun c i j => by
    linarith [abs_sub_abs_le_abs_sub (xt c i j) (xa c i j), hd c i j, hxa c i j]
  set S := convFanS W xt o hi wi with hSdef
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hSle : S ≤ (n : ℝ) * w' * (A + E) :=
    convFanS_le hw' (by linarith) hW hxt o hi wi
  -- step 1: conv_close_mixed at xt
  have hbase := M.conv_close_mixed L W b xt o hi wi
  rw [convBr_eq_convBrR] at hbase
  -- step 2: the real conv is Lipschitz from xt to xa
  have hlip : |conv2d W b xt o hi wi - conv2d W b xa o hi wi| ≤ (n : ℝ) * w' * E :=
    conv2d_sub_abs_le hw' hE hW hd o hi wi
  have hsplit : |M.convMixed L W b xt o hi wi - conv2d W b xa o hi wi|
      ≤ |M.convMixed L W b xt o hi wi - conv2d W b xt o hi wi|
        + |conv2d W b xt o hi wi - conv2d W b xa o hi wi| :=
    abs_sub_le _ _ _
  -- monotonicity: replace S by its closed form, |b o| by β
  have hP : (0 : ℝ) ≤ (1 + L.u) * (1 + br) := mul_nonneg (by linarith) (by linarith)
  have hm1 : (1 + L.u) * (1 + br) * S ≤ (1 + L.u) * (1 + br) * ((n : ℝ) * w' * (A + E)) :=
    mul_le_mul_of_nonneg_left hSle hP
  have hm2 : L.u * (1 + br) * S ≤ L.u * (1 + br) * ((n : ℝ) * w' * (A + E)) :=
    mul_le_mul_of_nonneg_left hSle (mul_nonneg hLu (by linarith))
  have hm3 : br * S ≤ br * ((n : ℝ) * w' * (A + E)) :=
    mul_le_mul_of_nonneg_left hSle hbr0
  have hm0 : M.u * ((1 + L.u) * (1 + br) * S + |b o|)
      ≤ M.u * ((1 + L.u) * (1 + br) * ((n : ℝ) * w' * (A + E)) + β) :=
    mul_le_mul_of_nonneg_left (by linarith [hb o]) hMu
  simp only [convMixedBudget]
  linarith

-- ════════════════════════════════════════════════════════════════
-- § Vec space, and the `FloatClose` instance
-- ════════════════════════════════════════════════════════════════

/-- **Vec-space mixed-precision conv** — the bf16 peer of `FloatModel.flatConvF`, in the flat
    space the ResNet composition actually lives in. -/
noncomputable def FloatModel.flatConvMixed (M L : FloatModel) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  fun v => Tensor3.flatten (M.convMixed L W b (Tensor3.unflatten v))

/-- The Vec-space propagating bound — `convMixed_close_prop` transported through
    `flatten`/`unflatten`, exactly as `flatConvF_close` transports `convF_close`. -/
theorem FloatModel.flatConvMixed_close (M L : FloatModel) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (vt va : Vec (ic * h * w))
    {w' β A E : ℝ} (hw' : 0 ≤ w') (hA : 0 ≤ A) (hE : 0 ≤ E)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hva : ∀ k, |va k| ≤ A) (hd : ∀ k, |vt k - va k| ≤ E)
    (k : Fin (oc * h * w)) :
    |M.flatConvMixed L W b vt k - flatConv W b va k|
      ≤ convMixedBudget M.u L.u (ic * kH * kW) w' β A E := by
  have huf_a : ∀ c i j, |Tensor3.unflatten va c i j| ≤ A := by
    intro c i j; simp only [Tensor3.unflatten]; exact hva _
  have huf_d : ∀ c i j, |Tensor3.unflatten vt c i j - Tensor3.unflatten va c i j| ≤ E := by
    intro c i j; simp only [Tensor3.unflatten]; exact hd _
  simp only [FloatModel.flatConvMixed, flatConv, Tensor3.flatten]
  exact M.convMixed_close_prop L W b _ _ hw' hA hE hW hb huf_a huf_d _ _ _

/-- **A mixed-precision (leaf `L`, accumulate `M`) convolution is `FloatClose`.** Magnitude
    `A` in, real output `≤ layerAct` and float output `≤ layerAct + convMixedBudget(E := 0)`
    out; error modulus `E ↦ convMixedBudget … E`. The other `FloatClose` instances
    (`floatClose_relu`, `floatClose_bn`, `floatClose_maxPool3s2`, `floatClose_gap`,
    `floatClose_residualBlock`, `floatClose_iterate`, `FloatClose.comp`) compose with it
    unchanged; nothing in the repo instantiates it. -/
theorem floatClose_flatConvMixed {ic oc h w kH kW : Nat} (M L : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (_hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0)
      (flatConv (h := h) (w := w) W b) (M.flatConvMixed L (h := h) (w := w) W b)
      (fun E => convMixedBudget M.u L.u (ic * kH * kW) w' β A E) :=
  FloatClose.of_close (fun v hv i => flatConv_abs_le hA hW hb hv i)
    (fun v hv i => M.flatConvMixed_close L W b v v hw' hA le_rfl hW hb hv (fun k => by simp) i)
    (fun vt va E hva _ hd i => M.flatConvMixed_close L W b vt va hw' hA
      ((abs_nonneg _).trans (hd ⟨0, hn⟩)) hW hb hva hd i)

-- ════════════════════════════════════════════════════════════════
-- § What bf16 costs the WHOLE-NET bound — the per-layer gain
-- ════════════════════════════════════════════════════════════════

/-- **The per-layer error GAIN** — the coefficient of the inherited error `E` in
    `convMixedBudget`. This is the number that compounds: a `d`-layer stack multiplies its
    input error by `gain^d`, so the gain, not the additive constant, is what decides whether a
    composed bound says anything. -/
noncomputable def convMixedGain (uacc uleaf : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  (n : ℝ) * w * (1 + convBrR uacc uleaf n + uleaf * (1 + convBrR uacc uleaf n)
    + uacc * ((1 + uleaf) * (1 + convBrR uacc uleaf n)))

/-- **`convMixedBudget` is affine in the inherited error**, with slope `convMixedGain`. So
    composing `d` of these is `gain^d` on the input error plus a geometric sum of the additive
    terms — the shape every composed forward-error bound has. -/
theorem convMixedBudget_affine (uacc uleaf : ℝ) (n : ℕ) (w β A E : ℝ) :
    convMixedBudget uacc uleaf n w β A E
      = convMixedBudget uacc uleaf n w β A 0 + convMixedGain uacc uleaf n w * E := by
  simp only [convMixedBudget, convMixedGain]; ring

/-- **The f32 peer, for comparison.** `layerBudget` is affine in `E` too, with slope
    `m·w·(1+u)^(m+2)`. Both slopes are `fan-in · weight-bound` times a factor that is
    `1 + O(roundoff)` (`convMixedGain_factor`). -/
theorem layerBudget_affine (u : ℝ) (m : ℕ) (w β A E : ℝ) :
    layerBudget u m w β A E
      = layerBudget u m w β A 0 + (m : ℝ) * w * (1 + u) ^ (m + 2) * E := by
  simp only [layerBudget]; ring

-- Illustration (arithmetic outside Lean): the f32 gain's `ε = (1+u_acc)^(n+2) − 1` is 2.7e-4 at
-- `u_acc = 2⁻²⁴`, `n = 4608`; the bf16-mixed `ε` is 1.20e-2 at `u_leaf = 2⁻⁸`, dominated by
-- `br`'s flat leaf term. `(1.012043/1.000275)^d` is 1.52× at d = 36 conv layers (R34) and 1.86×
-- at d = 53 (R50). Both bounds are vacuous in absolute terms: the shared `n·w` factor is ≫ 1 at
-- a real layer (n = 4608, w' ≈ 0.05 gives ~230), so `gain^53` is astronomical for f32 and bf16
-- alike — a property of worst-case forward-error analysis composed depth-first (every term takes
-- the adversarial sign), not of bf16. A non-vacuous absolute number needs a different analysis
-- (probabilistic rounding, or a bound that uses BN's renormalisation at each layer).
/-- **The per-layer gain factors as `n·w·(1+ε)`**, with
    `ε = br + u_leaf(1+br) + u_acc(1+u_leaf)(1+br)` (`br = convBrR u_acc u_leaf n`).
    Relative to f32 (`layerBudget_affine`, `ε = (1+u)^(n+2) − 1`), bf16-mixed changes only the
    `1+ε` factor, not the `n·w` growth rate. Composed over depth, both bounds grow as
    `(n·w)^d` and are vacuous in absolute terms at real layer sizes. -/
theorem convMixedGain_factor (uacc uleaf : ℝ) (n : ℕ) (w : ℝ) :
    convMixedGain uacc uleaf n w
      = (n : ℝ) * w * (1 + (convBrR uacc uleaf n + uleaf * (1 + convBrR uacc uleaf n)
          + uacc * ((1 + uleaf) * (1 + convBrR uacc uleaf n)))) := by
  simp only [convMixedGain]; ring

end Proofs
