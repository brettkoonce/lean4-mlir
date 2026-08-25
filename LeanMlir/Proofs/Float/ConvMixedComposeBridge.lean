import LeanMlir.Proofs.Float.FloatComposeBridge
import LeanMlir.Proofs.Float.ConvMixedFloatBridge

/-! # The mixed-precision conv as a `FloatClose` — the whole-net bf16 bound

`FloatModel.conv_close_mixed` bounds **one** bf16-mixed convolution against exact ℝ at
an **exactly-represented input**. That is not enough to compose: a net feeds each layer the
*previous* layer's already-perturbed output, so what a fold needs is an error **modulus** — a map
from inherited input error to output error — plus a magnitude bound to thread forward. That pair
is `FloatComposeBridge.FloatClose`, and this file supplies its mixed-precision conv instance.

⭐⭐ **The composition backbone is PRECISION-AGNOSTIC, and that is the whole reason this is
small.** `FloatClose A B f fF L` says nothing about how `fF` rounds — only that it stays within
`L e` of `f`. So `floatClose_relu`, `floatClose_bn`, `floatClose_maxPool3s2`, `floatClose_gap`,
`floatClose_residualBlock`, `floatClose_iterate` and `FloatClose.comp` apply to a bf16 conv
UNCHANGED. One new instance buys the entire existing fold — the `[3,4,6,3]` assembly in
`Resnet34WholeFloatBridge` included.

What genuinely had to be proved here, none of which the `e = 0` bound gives:

1. `convFanS_le` — the data-dependent `Σ|kernel·window|` replaced by the closed form
   `n·w·A`, so the budget is a formula in dims and norms rather than in the input.
2. `conv2d_sub_abs_le` — the REAL conv is `n·w`-Lipschitz in its input. This is the term that
   carries a predecessor's error through the layer, and it has no analogue at `e = 0`.
3. `convMixedBudget` / `convMixed_close_prop` — the two combined, at an input that is both
   perturbed (`E`) and magnitude-bounded (`A`).

⚠ **The budget is evaluated at `A + E`, not `A`.** The float conv runs on the PERTURBED input, so
its own rounding scales with the perturbed magnitude; only the real conv sees `A`. Writing `A`
there would understate the bound — the unsound direction.

▶ `n = ic·kH·kW` throughout, and the fan-in amplification rides `uacc` (fp32) while `uleaf`
(bf16) enters flat — the §9.3 separation that makes this non-vacuous at R50's n = 4608.
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
  simp only [convWindow3]
  split
  · exact hx _ _ _
  · simpa using hA

/-- ⭐ The window inherits the input's PERTURBATION. The padding branch is the same branch for
    both tensors (it depends only on the indices), so it contributes `|0 - 0| = 0 ≤ E`. -/
theorem convWindow3_sub_abs_le {ic h w kH kW : Nat} {xt xa : Tensor3 ic h w} {E : ℝ}
    (hE : 0 ≤ E) (hd : ∀ c i j, |xt c i j - xa c i j| ≤ E) (hi : Fin h) (wi : Fin w) :
    ∀ c kh kw, |convWindow3 kH kW xt hi wi c kh kw
                 - convWindow3 kH kW xa hi wi c kh kw| ≤ E := by
  intro c kh kw
  simp only [convWindow3]
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
  calc convFanS W x o hi wi
      ≤ ∑ _k : Fin (ic * kH * kW), w' * A :=
        Finset.sum_le_sum fun k _ => hstep k
    _ = ((ic * kH * kW : ℕ) : ℝ) * (w' * A) := by
        rw [Finset.sum_const]; simp [nsmul_eq_mul]
    _ = ((ic * kH * kW : ℕ) : ℝ) * w' * A := by ring

-- ════════════════════════════════════════════════════════════════
-- § The real convolution is Lipschitz in its input
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **`conv2d` is `n·w`-Lipschitz.** THE term with no analogue at `e = 0`: it is how a
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
    _ ≤ ∑ _k : Fin (ic * kH * kW), w' * E := by
        refine Finset.sum_le_sum fun k _ => ?_
        rw [abs_mul]
        exact mul_le_mul
          (Tensor3.flatten_abs_le (fun c kh kw => hW o c kh kw) k)
          (Tensor3.flatten_sub_abs_le (convWindow3_sub_abs_le hE hd hi wi) k)
          (abs_nonneg _) hw'
    _ = ((ic * kH * kW : ℕ) : ℝ) * (w' * E) := by
        rw [Finset.sum_const]; simp [nsmul_eq_mul]
    _ = ((ic * kH * kW : ℕ) : ℝ) * w' * E := by ring

-- ════════════════════════════════════════════════════════════════
-- § The mixed-precision conv budget, with an INHERITED error
-- ════════════════════════════════════════════════════════════════

/-- `convBr` as a function of the two roundoffs alone — the same bracket, with the `FloatModel`s
    peeled off so a concrete instance evaluates by `norm_num`. -/
noncomputable def convBrR (uacc uleaf : ℝ) (n : ℕ) : ℝ :=
  ((1 + uacc) ^ (n + 1) - 1) * (1 + uleaf) ^ 2 + (2 * uleaf + uleaf ^ 2)

theorem convBr_eq_convBrR (M L : FloatModel) (n : ℕ) :
    convBr M L n = convBrR M.u L.u n := rfl

theorem convBrR_nonneg {uacc uleaf : ℝ} (hacc : 0 ≤ uacc) (hleaf : 0 ≤ uleaf) (n : ℕ) :
    0 ≤ convBrR uacc uleaf n := by
  have h1 : (0 : ℝ) ≤ (1 + uacc) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have h2 : (0 : ℝ) ≤ (1 + uleaf) ^ 2 := sq_nonneg _
  have h3 : (0 : ℝ) ≤ 2 * uleaf + uleaf ^ 2 := by nlinarith [sq_nonneg uleaf]
  simp only [convBrR]; nlinarith

/-- ⭐⭐ **The mixed-precision conv budget — the `layerBudget` peer, and the object this whole
    file exists to produce.** Four terms:

    * `uacc * (… + β)` — the f32 bias add,
    * `uleaf * (1+br) * …` — the **bf16 store** of the accumulator (the bf16-TYPED conv result,
      forced by the only emit shape that reaches tensor cores; §9.2),
    * `br * …` — the dot itself, fan-in `n` amplified at the ACCUMULATE precision,
    * `n·w·E` — **the inherited error**, carried through by the real conv's Lipschitz constant.

    ⚠ The first three are evaluated at `A + E`, not `A`: the float conv runs on the PERTURBED
    input, so its own rounding scales with the perturbed magnitude. Only the fourth term is
    linear in `E` alone. -/
noncomputable def convMixedBudget (uacc uleaf : ℝ) (n : ℕ) (w β A E : ℝ) : ℝ :=
  uacc * ((1 + uleaf) * (1 + convBrR uacc uleaf n) * ((n : ℝ) * w * (A + E)) + β)
    + uleaf * (1 + convBrR uacc uleaf n) * ((n : ℝ) * w * (A + E))
    + convBrR uacc uleaf n * ((n : ℝ) * w * (A + E))
    + (n : ℝ) * w * E

theorem convMixedBudget_nonneg {uacc uleaf : ℝ} {n : ℕ} {w β A E : ℝ}
    (hacc : 0 ≤ uacc) (hleaf : 0 ≤ uleaf) (hw : 0 ≤ w) (hβ : 0 ≤ β)
    (hA : 0 ≤ A) (hE : 0 ≤ E) : 0 ≤ convMixedBudget uacc uleaf n w β A E := by
  have hbr := convBrR_nonneg hacc hleaf n
  have hnw : (0 : ℝ) ≤ (n : ℝ) * w := mul_nonneg (Nat.cast_nonneg n) hw
  have hS : (0 : ℝ) ≤ (n : ℝ) * w * (A + E) := mul_nonneg hnw (by linarith)
  have h1 : (0 : ℝ) ≤ (1 + uleaf) * (1 + convBrR uacc uleaf n) :=
    mul_nonneg (by linarith) (by linarith)
  simp only [convMixedBudget]
  have t1 : (0 : ℝ) ≤ uacc * ((1 + uleaf) * (1 + convBrR uacc uleaf n) * ((n : ℝ) * w * (A + E)) + β) :=
    mul_nonneg hacc (by nlinarith)
  have t2 : (0 : ℝ) ≤ uleaf * (1 + convBrR uacc uleaf n) * ((n : ℝ) * w * (A + E)) :=
    mul_nonneg (mul_nonneg hleaf (by linarith)) hS
  have t3 : (0 : ℝ) ≤ convBrR uacc uleaf n * ((n : ℝ) * w * (A + E)) := mul_nonneg hbr hS
  have t4 : (0 : ℝ) ≤ (n : ℝ) * w * E := mul_nonneg hnw hE
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The propagating bound
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **Mixed-precision convolution against exact ℝ at a PERTURBED input.** The composable
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
  have hxt : ∀ c i j, |xt c i j| ≤ A + E := by
    intro c i j
    have := abs_sub_le (xt c i j) (xa c i j) 0
    have h1 : |xt c i j| ≤ |xt c i j - xa c i j| + |xa c i j| := by simpa using this
    linarith [hd c i j, hxa c i j]
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

/-- ⭐⭐⭐ **THE INSTANCE: a bf16-mixed convolution is `FloatClose`.** Magnitude `A` in, real
    output `≤ layerAct` and float output `≤ layerAct + convMixedBudget(E := 0)` out; error
    modulus `E ↦ convMixedBudget … E`.

    ▶ This is the ONLY thing the whole-net bf16 bound needed. Everything the f32 fold already
    has — `floatClose_relu`, `floatClose_bn`, `floatClose_maxPool3s2`, `floatClose_gap`,
    `floatClose_residualBlock`, `floatClose_iterate`, `FloatClose.comp` — is stated on
    `FloatClose` and therefore applies to this verbatim. -/
theorem floatClose_flatConvMixed {ic oc h w kH kW : Nat} (M L : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0)
      (flatConv (h := h) (w := w) W b) (M.flatConvMixed L (h := h) (w := w) W b)
      (fun E => convMixedBudget M.u L.u (ic * kH * kW) w' β A E) := by
  have hB0 : 0 ≤ convMixedBudget M.u L.u (ic * kH * kW) w' β A 0 :=
    convMixedBudget_nonneg M.u_nonneg L.u_nonneg hw' hβ hA le_rfl
  refine ⟨fun v hv i => ?_, fun vt va E hva hvt hd i => ?_⟩
  · have hreal : |flatConv W b v i| ≤ layerAct (ic * kH * kW) w' β A :=
      flatConv_abs_le hA hW hb hv i
    have hround : |M.flatConvMixed L W b v i - flatConv W b v i|
        ≤ convMixedBudget M.u L.u (ic * kH * kW) w' β A 0 :=
      M.flatConvMixed_close L W b v v hw' hA le_rfl hW hb hv (fun k => by simp) i
    refine ⟨hreal.trans (le_add_of_nonneg_right hB0), ?_⟩
    calc |M.flatConvMixed L W b v i|
        ≤ |M.flatConvMixed L W b v i - flatConv W b v i| + |flatConv W b v i| := by
          simpa using abs_sub_le (M.flatConvMixed L W b v i) (flatConv W b v i) 0
      _ ≤ convMixedBudget M.u L.u (ic * kH * kW) w' β A 0
            + layerAct (ic * kH * kW) w' β A := add_le_add hround hreal
      _ = layerAct (ic * kH * kW) w' β A
            + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0 := by ring
  · have hE : 0 ≤ E := (abs_nonneg _).trans (hd ⟨0, hn⟩)
    exact M.flatConvMixed_close L W b vt va hw' hA hE hW hb hva hd i

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

/-- ⭐ **`convMixedBudget` is AFFINE in the inherited error**, with slope `convMixedGain`. So
    composing `d` of these is `gain^d` on the input error plus a geometric sum of the additive
    terms — the shape every composed forward-error bound has. -/
theorem convMixedBudget_affine (uacc uleaf : ℝ) (n : ℕ) (w β A E : ℝ) :
    convMixedBudget uacc uleaf n w β A E
      = convMixedBudget uacc uleaf n w β A 0 + convMixedGain uacc uleaf n w * E := by
  simp only [convMixedBudget, convMixedGain]; ring

/-- **The f32 peer, for comparison.** `layerBudget` is affine in `E` too, with slope
    `m·w·(1+u)^(m+2)`. ▶ Both slopes are `fan-in · weight-bound` times a factor that is
    `1 + O(roundoff)`, which is the point of the next comment. -/
theorem layerBudget_affine (u : ℝ) (m : ℕ) (w β A E : ℝ) :
    layerBudget u m w β A E
      = layerBudget u m w β A 0 + (m : ℝ) * w * (1 + u) ^ (m + 2) * E := by
  simp only [layerBudget]; ring

/-- ⭐⭐ **THE HONEST READING, and it is the useful result of this file.**

    Both gains factor as `n·w · (1 + ε)`:

    * f32: `ε = (1+u_acc)^(n+2) − 1`, which at `u_acc = 2⁻²⁴`, `n = 4608` is **2.7e-4**;
    * bf16-mixed: `ε ≈ br + u_leaf(1+br) + u_acc(1+u_leaf)(1+br)`, which at `u_leaf = 2⁻⁸` is
      **1.20e-2** — dominated by `br`'s flat leaf term, exactly as §9.3 found for one layer.

    ▶ **So bf16 does NOT change the whole-net bound's growth RATE — it changes a `1+ε` factor.**
    `(1.012043/1.000275)^d` over `d` conv layers: **1.52× at R34's 36** and **1.86× at R50's 53**.
    Under a factor of two on the certificate, for a 1.41×/1.55× speedup. (Arithmetic outside
    Lean, quoted as illustration; the affine decomposition above is what is proved.)

    ⚠⚠ **AND BOTH BOUNDS ARE VACUOUS IN ABSOLUTE TERMS, which this file will not pretend
    otherwise.** The shared `n·w` factor is ≫ 1 at any real layer (n = 4608, w' ≈ 0.05 gives
    ~230), so `gain^53` is astronomical for the f32 bound and the bf16 one alike. That is a
    property of worst-case forward-error analysis composed depth-first — every term assumes the
    adversarial sign — not a property of bf16, and the repo's existing f32 whole-net bridges
    (`Resnet34WholeFloatBridge`) carry exactly the same factor. ▶ What is meaningful here is the
    RATIO: bf16's certificate is ~2× the f32 certificate, not exponentially worse. Anyone
    wanting a non-vacuous absolute number needs a different analysis (probabilistic rounding,
    or a bound that exploits BN's renormalisation at each layer), not a tighter conv lemma. -/
theorem convMixedGain_factor (uacc uleaf : ℝ) (n : ℕ) (w : ℝ) :
    convMixedGain uacc uleaf n w
      = (n : ℝ) * w * (1 + (convBrR uacc uleaf n + uleaf * (1 + convBrR uacc uleaf n)
          + uacc * ((1 + uleaf) * (1 + convBrR uacc uleaf n)))) := by
  simp only [convMixedGain]; ring

-- ════════════════════════════════════════════════════════════════
-- § Composition — the fold, in bf16
-- ════════════════════════════════════════════════════════════════

/-- **conv→relu in bf16 is `FloatClose`** — the bf16 peer of `floatClose_reluConv`, and the
    proof is the same one line, because `floatClose_relu` never asked what precision fed it. -/
theorem floatClose_reluConvMixed {ic oc h w kH kW : Nat} (M L : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0)
      (relu (oc * h * w) ∘ flatConv (h := h) (w := w) W b)
      (relu (oc * h * w) ∘ M.flatConvMixed L (h := h) (w := w) W b)
      ((fun e => e) ∘ (fun E => convMixedBudget M.u L.u (ic * kH * kW) w' β A E)) :=
  (floatClose_flatConvMixed M L W b hw' hβ hA hn hW hb).comp
    (floatClose_relu (layerAct (ic * kH * kW) w' β A
      + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0))

/-- ⭐ **Two bf16 convs chained** — the `.comp` of two mixed-precision layers, moduli composing.
    This is the inductive step of any depth; nothing about it is conv-specific or R50-specific. -/
theorem floatClose_convMixed_twice {ic mc oc h w kH kW : Nat} (M L : FloatModel)
    (W₁ : Kernel4 mc ic kH kW) (b₁ : Vec mc) (W₂ : Kernel4 oc mc kH kW) (b₂ : Vec oc)
    {w' β A : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A)
    (hn₁ : 0 < ic * h * w) (hn₂ : 0 < mc * h * w)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w') (hb₁ : ∀ o, |b₁ o| ≤ β)
    (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w') (hb₂ : ∀ o, |b₂ o| ≤ β) :
    FloatClose A
      (layerAct (mc * kH * kW) w' β
          (layerAct (ic * kH * kW) w' β A + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0)
        + convMixedBudget M.u L.u (mc * kH * kW) w' β
            (layerAct (ic * kH * kW) w' β A
              + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0) 0)
      (flatConv (h := h) (w := w) W₂ b₂ ∘ flatConv (h := h) (w := w) W₁ b₁)
      (M.flatConvMixed L (h := h) (w := w) W₂ b₂ ∘ M.flatConvMixed L (h := h) (w := w) W₁ b₁)
      ((fun E => convMixedBudget M.u L.u (mc * kH * kW) w' β
          (layerAct (ic * kH * kW) w' β A
            + convMixedBudget M.u L.u (ic * kH * kW) w' β A 0) E)
        ∘ (fun E => convMixedBudget M.u L.u (ic * kH * kW) w' β A E)) :=
  (floatClose_flatConvMixed M L W₁ b₁ hw' hβ hA hn₁ hW₁ hb₁).comp
    (floatClose_flatConvMixed M L W₂ b₂ hw' hβ
      (by
        have h1 : 0 ≤ layerAct (ic * kH * kW) w' β A := layerAct_nonneg hw' hβ hA
        have h2 : 0 ≤ convMixedBudget M.u L.u (ic * kH * kW) w' β A 0 :=
          convMixedBudget_nonneg M.u_nonneg L.u_nonneg hw' hβ hA le_rfl
        linarith)
      hn₂ hW₂ hb₂)

/-- ⭐⭐ **R50's `[3,4,6,3]` stage fold, in bf16 — and it is `floatClose_r34_stages` verbatim.**
    ResNet-50 has the SAME stage depths as ResNet-34; the two differ in what a block contains
    (three convs with a 1×1 bottleneck vs two 3×3s), not in how many blocks a stage stacks. So
    the depth fold needs no R50-specific theorem — only an R50 block instance, which is what
    `floatClose_flatConvMixed` now makes constructible in bf16.

    ▶ Stated here under the same magnitude-stability hypothesis the f32 fold uses: a block whose
    activations stay within `A` (which is what BN buys, and what the a-posteriori probe checks). -/
theorem floatClose_r50_stages_mixed {m : Nat} {A : ℝ}
    {blk blkF : Vec m → Vec m} {Lm : ℝ → ℝ} (hblk : FloatClose A A blk blkF Lm) :
    FloatClose A A (blk^[3]) (blkF^[3]) (Lm^[3]) ∧
    FloatClose A A (blk^[4]) (blkF^[4]) (Lm^[4]) ∧
    FloatClose A A (blk^[6]) (blkF^[6]) (Lm^[6]) ∧
    FloatClose A A (blk^[3]) (blkF^[3]) (Lm^[3]) :=
  floatClose_r34_stages hblk

end Proofs
