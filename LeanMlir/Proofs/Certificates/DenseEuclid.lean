import LeanMlir.Proofs.Certificates.LipschitzCert

/-! # The dense Euclidean engine — dense and ReLU layers on `EuclideanSpace`, and their L2 bounds

The layers every Lipschitz / interval certificate in `Certificates/` is stated about: `denseE W`
(bias-free dense, `(Wx)ᵢ`) and `reluE`, with the L2 Lipschitz bounds the product certificate
multiplies — Frobenius (`denseE_lipschitzL2`), the Gram / Schatten-4 bound
(`denseE_lipschitzL2_gram`), Schatten-8 (`denseE_lipschitzL2_gram2`), and the lower bound a
witness vector gives (`lipschitzL2_lower_euclid`). `certified_at_eps` specialises the
Tsuzuku certificate to a rational radius check. The trained instances are in
`LipschitzCertInstance`; the namespace is theirs, kept so every citation keeps its name.
-/

namespace Proofs
namespace LipschitzCertDemo

open scoped BigOperators

/-- A bias-free dense (linear) layer on Euclidean space:
    `(denseE W x)ᵢ = Σⱼ Wᵢⱼ xⱼ`. -/
noncomputable def denseE {n k : ℕ} (W : Fin k → Fin n → ℝ) :
    EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k) :=
  fun x => WithLp.toLp 2 (fun i => ∑ j, W i j * x j)

@[simp] theorem denseE_apply {n k : ℕ} (W : Fin k → Fin n → ℝ)
    (x : EuclideanSpace ℝ (Fin n)) (i : Fin k) :
    denseE W x i = ∑ j, W i j * x j := rfl

/-- **Frobenius bound, proved.** If the entrywise square sum of `W` is at
    most `C²`, the dense layer is `C`-Lipschitz in L2. This is the certified
    replacement for the power-iteration estimate `specNormW`: `‖W‖₂ ≤ ‖W‖_F`,
    so any rational `C ≥ ‖W‖_F` is a sound Lipschitz constant. -/
theorem denseE_lipschitzL2 {n k : ℕ} (W : Fin k → Fin n → ℝ) {C : ℝ}
    (hC : 0 ≤ C) (hW : ∑ i, ∑ j, W i j ^ 2 ≤ C ^ 2) :
    LipschitzL2 C (denseE W) := by
  intro u w
  have hcoord : ∀ i : Fin k,
      (denseE W u - denseE W w) i = ∑ j, W i j * ((u - w) j) := by
    intro i
    show (∑ j, W i j * u j) - (∑ j, W i j * w j) = _
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by
      show W i j * u j - W i j * w j = W i j * (u j - w j); ring
  have hsq : ‖denseE W u - denseE W w‖ ^ 2 ≤ (C * ‖u - w‖) ^ 2 := by
    rw [euclid_norm_sq]
    calc ∑ i, ((denseE W u - denseE W w) i) ^ 2
        = ∑ i, (∑ j, W i j * ((u - w) j)) ^ 2 := by
          exact Finset.sum_congr rfl fun i _ => by rw [hcoord]
      _ ≤ ∑ i, ((∑ j, W i j ^ 2) * (∑ j, ((u - w) j) ^ 2)) :=
          Finset.sum_le_sum fun i _ =>
            Finset.sum_mul_sq_le_sq_mul_sq _ _ _
      _ = (∑ i, ∑ j, W i j ^ 2) * (∑ j, ((u - w) j) ^ 2) :=
          (Finset.sum_mul ..).symm
      _ ≤ C ^ 2 * (∑ j, ((u - w) j) ^ 2) :=
          mul_le_mul_of_nonneg_right hW
            (Finset.sum_nonneg fun j _ => sq_nonneg _)
      _ = (C * ‖u - w‖) ^ 2 := by rw [mul_pow, euclid_norm_sq]
  have h0 : 0 ≤ C * ‖u - w‖ := mul_nonneg hC (norm_nonneg _)
  calc ‖denseE W u - denseE W w‖
      = Real.sqrt (‖denseE W u - denseE W w‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt ((C * ‖u - w‖) ^ 2) := Real.sqrt_le_sqrt hsq
    _ = C * ‖u - w‖ := Real.sqrt_sq h0

/-- Coordinatewise ReLU on Euclidean space. -/
noncomputable def reluE {n : ℕ} :
    EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin n) :=
  fun x => WithLp.toLp 2 (fun i => max (x i) 0)

@[simp] theorem reluE_apply {n : ℕ} (x : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    reluE x i = max (x i) 0 := rfl

/-- **A one-hidden-layer net's logits from its hidden pre-activations.** Given the exact
    pre-activations `pre` of `x`, each logit of `denseE W2 ∘ reluE ∘ denseE W1` is the finite sum
    the per-image margin and argmax proofs expand; every generated scorecard reads its logits
    through this. -/
theorem mlp_out_eq {n h k : ℕ} (W1 : Fin h → Fin n → ℝ) (W2 : Fin k → Fin h → ℝ)
    {x : EuclideanSpace ℝ (Fin n)} {pre : Fin h → ℝ} (hpre : ∀ t, denseE W1 x t = pre t)
    (j : Fin k) :
    (denseE W2 ∘ reluE ∘ denseE W1) x j = ∑ t, W2 j t * max (pre t) 0 := by
  show denseE W2 (reluE (denseE W1 x)) j = _
  rw [denseE_apply]
  exact Finset.sum_congr rfl fun t _ => by rw [reluE_apply, hpre t]

/-- `√2 ≤ 14143/10000` — the rational majorant the per-image radius checks use. -/
theorem sqrt_two_le_rat : Real.sqrt 2 ≤ ((14143 : ℝ)/10000) :=
  Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩

/-- Specialize the Tsuzuku certificate to a FIXED radius ε: if the margin
    clears the rational check `(14143/10000)·L·ε ≤ m` (kernel-checkable —
    no `√2`), every `‖δ‖ < ε` leaves class `i` the strict argmax. -/
theorem certified_at_eps {n k : ℕ} {L m ε : ℝ}
    {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}
    (hf : LipschitzL2 L f) (hL : 0 < L) {x : EuclideanSpace ℝ (Fin n)}
    {i : Fin k} (hmargin : ∀ j, j ≠ i → m ≤ f x i - f x j)
    (hε : ((14143 : ℝ)/10000) * L * ε ≤ m) (hε0 : 0 ≤ ε)
    (δ : EuclideanSpace ℝ (Fin n)) (hδ : ‖δ‖ < ε) :
    ∀ j, j ≠ i → f (x + δ) j < f (x + δ) i := by
  refine lipschitz_margin_certified_radius hf hL hmargin (lt_of_lt_of_le hδ ?_)
  rw [le_div_iff₀ (mul_pos (Real.sqrt_pos.mpr (by norm_num)) hL)]
  calc ε * (Real.sqrt 2 * L) ≤ ε * (((14143 : ℝ)/10000) * L) := by
        have h2 : (0:ℝ) ≤ L := le_of_lt hL
        have := mul_le_mul_of_nonneg_right sqrt_two_le_rat h2
        exact mul_le_mul_of_nonneg_left this hε0
    _ = ((14143 : ℝ)/10000) * L * ε := by ring
    _ ≤ m := hε

/-- **ReLU is 1-Lipschitz in L2** — coordinatewise `|max(a,0) − max(b,0)| ≤ |a − b|`
    summed. The activation contributes factor 1 to the product certificate. -/
theorem reluE_lipschitzL2 {n : ℕ} : LipschitzL2 1 (reluE (n := n)) := by
  intro u w
  have hsq : ‖reluE u - reluE w‖ ^ 2 ≤ ‖u - w‖ ^ 2 := by
    rw [euclid_norm_sq, euclid_norm_sq]
    refine Finset.sum_le_sum fun i _ => ?_
    have habs : |max (u i) 0 - max (w i) 0| ≤ |u i - w i| :=
      abs_max_sub_max_le_abs (u i) (w i) 0
    have h1 : (reluE u - reluE w) i = max (u i) 0 - max (w i) 0 := rfl
    have h2 : (u - w) i = u i - w i := rfl
    rw [h1, h2, ← sq_abs (max (u i) 0 - max (w i) 0), ← sq_abs (u i - w i)]
    exact pow_le_pow_left₀ (abs_nonneg _) habs 2
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _),
       one_mul] at *


/-- Row-wise Cauchy–Schwarz summed: `‖Mv‖² ≤ ‖M‖_F²·‖v‖²` at the raw-sum level. -/
theorem sum_sq_matvec_le {k n : ℕ} (M : Fin k → Fin n → ℝ) (y : Fin n → ℝ) :
    ∑ a, (∑ b, M a b * y b) ^ 2 ≤ (∑ a, ∑ b, M a b ^ 2) * (∑ b, y b ^ 2) := by
  calc ∑ a, (∑ b, M a b * y b) ^ 2
      ≤ ∑ a, ((∑ b, M a b ^ 2) * (∑ b, y b ^ 2)) :=
        Finset.sum_le_sum fun a _ => Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    _ = (∑ a, ∑ b, M a b ^ 2) * (∑ b, y b ^ 2) := (Finset.sum_mul ..).symm

/-- **Gram (Schatten-4) bound, proved.** If `G = W·Wᵀ` (supplied as data, verified
    entrywise) and `‖G‖_F² ≤ B⁴`, then the dense layer is `B`-Lipschitz in L2.
    Since `‖G‖_F = (Σᵢσᵢ⁴)^½`, this is `‖W‖₂ ≤ (Σσᵢ⁴)^¼` — strictly tighter than
    Frobenius `(Σσᵢ²)^½` whenever the spectrum has any spread. The Gram matrix is
    only `k×k` (output-side), so the kernel arithmetic stays small even for wide
    layers. -/
theorem denseE_lipschitzL2_gram {n k : ℕ} (W : Fin k → Fin n → ℝ)
    (G : Fin k → Fin k → ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hG : ∀ a b, G a b = ∑ j, W a j * W b j)
    (hGF : ∑ a, ∑ b, G a b ^ 2 ≤ B ^ 4) :
    LipschitzL2 B (denseE W) := by
  intro u w
  set d : Fin n → ℝ := fun j => u j - w j with hdd
  set y : Fin k → ℝ := fun i => ∑ j, W i j * d j with hyy
  set z : Fin n → ℝ := fun j => ∑ i, W i j * y i with hzz
  set S : ℝ := ∑ i, y i ^ 2 with hS
  set Dq : ℝ := ∑ j, d j ^ 2 with hDq
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hDq0 : 0 ≤ Dq := Finset.sum_nonneg fun j _ => sq_nonneg _
  -- S = ⟨d, Wᵀy⟩
  have hswap : S = ∑ j, d j * z j := by
    calc S = ∑ i, y i * ∑ j, W i j * d j := by
          exact Finset.sum_congr rfl fun i _ => by rw [pow_two]
      _ = ∑ i, ∑ j, y i * (W i j * d j) := by
          exact Finset.sum_congr rfl fun i _ => Finset.mul_sum ..
      _ = ∑ j, ∑ i, y i * (W i j * d j) := Finset.sum_comm
      _ = ∑ j, d j * z j := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [hzz, Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by ring
  -- Σz² = ⟨y, Gy⟩ =: T
  have hTz : ∑ j, z j ^ 2 = ∑ a, y a * ∑ b, G a b * y b := by
    calc ∑ j, z j ^ 2
        = ∑ j, ∑ a, ∑ b, (W a j * y a) * (W b j * y b) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [pow_two, hzz, Finset.sum_mul_sum]
      _ = ∑ a, ∑ j, ∑ b, (W a j * y a) * (W b j * y b) := Finset.sum_comm
      _ = ∑ a, ∑ b, ∑ j, (W a j * y a) * (W b j * y b) := by
          exact Finset.sum_congr rfl fun a _ => Finset.sum_comm
      _ = ∑ a, ∑ b, (y a * y b) * ∑ j, W a j * W b j := by
          refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun j _ => by ring
      _ = ∑ a, y a * ∑ b, G a b * y b := by
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun b _ => by rw [hG]; ring
  have hT0 : 0 ≤ ∑ j, z j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
  -- CS1: S² ≤ Dq·T
  have hCS1 : S ^ 2 ≤ Dq * ∑ j, z j ^ 2 := by
    rw [hswap]
    exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  -- CS2: T² ≤ S · (ΣG²·S) ≤ B⁴·S²
  have hCS2 : (∑ j, z j ^ 2) ^ 2 ≤ B ^ 4 * S ^ 2 := by
    have h1 : (∑ j, z j ^ 2) ^ 2 ≤ S * ∑ a, (∑ b, G a b * y b) ^ 2 := by
      rw [hTz]
      exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    have h2 : ∑ a, (∑ b, G a b * y b) ^ 2 ≤ (∑ a, ∑ b, G a b ^ 2) * S :=
      sum_sq_matvec_le G y
    have h3 : (∑ a, ∑ b, G a b ^ 2) * S ≤ B ^ 4 * S :=
      mul_le_mul_of_nonneg_right hGF hS0
    calc (∑ j, z j ^ 2) ^ 2 ≤ S * ∑ a, (∑ b, G a b * y b) ^ 2 := h1
      _ ≤ S * (B ^ 4 * S) := by
          exact mul_le_mul_of_nonneg_left (h2.trans h3) hS0
      _ = B ^ 4 * S ^ 2 := by ring
  -- T ≤ B²·S  (both nonneg, compare squares)
  have hTle : (∑ j, z j ^ 2) ≤ B ^ 2 * S := by
    have hb2 : 0 ≤ B ^ 2 * S := mul_nonneg (sq_nonneg _) hS0
    nlinarith [hCS2, hT0, hb2]
  -- S ≤ B²·Dq  (divide S² ≤ Dq·B²·S by S, case S = 0)
  have hSle : S ≤ B ^ 2 * Dq := by
    rcases eq_or_lt_of_le hS0 with h0 | hpos
    · rw [← h0]; exact mul_nonneg (sq_nonneg _) hDq0
    · have : S ^ 2 ≤ Dq * (B ^ 2 * S) :=
        hCS1.trans (mul_le_mul_of_nonneg_left hTle hDq0)
      nlinarith [this, hpos]
  -- back to norms
  have hcoord : ∀ i, (denseE W u - denseE W w) i = y i := by
    intro i
    show (∑ j, W i j * u j) - (∑ j, W i j * w j) = _
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by
      show W i j * u j - W i j * w j = W i j * (u j - w j); ring
  have hnormsq : ‖denseE W u - denseE W w‖ ^ 2 ≤ (B * ‖u - w‖) ^ 2 := by
    rw [euclid_norm_sq, mul_pow, euclid_norm_sq]
    calc ∑ i, ((denseE W u - denseE W w) i) ^ 2
        = S := Finset.sum_congr rfl fun i _ => by rw [hcoord]
      _ ≤ B ^ 2 * Dq := hSle
      _ = B ^ 2 * ∑ j, ((u - w) j) ^ 2 := rfl
  calc ‖denseE W u - denseE W w‖
      = Real.sqrt (‖denseE W u - denseE W w‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt ((B * ‖u - w‖) ^ 2) := Real.sqrt_le_sqrt hnormsq
    _ = B * ‖u - w‖ := Real.sqrt_sq (mul_nonneg hB (norm_nonneg _))

/-- **Certified lower bound on any L2 Lipschitz constant** (the power-iteration
    direction): if `‖f u − f w‖ ≥ ℓ·‖u − w‖` at one concrete pair (verified as a
    squared-sum inequality in-kernel), then every valid `L` satisfies `ℓ ≤ L`.
    With `u` the (rationalized) power-iteration singular vector and `w = 0`,
    this certifies how close a proven upper bound sits to the true `‖W‖₂`. -/
theorem lipschitzL2_lower_euclid {n k : ℕ} {L ℓ : ℝ}
    {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}
    (hf : LipschitzL2 L f) (hℓ : 0 ≤ ℓ) (u w : EuclideanSpace ℝ (Fin n))
    (hpos : 0 < ∑ j, ((u - w) j) ^ 2)
    (hray : ℓ ^ 2 * (∑ j, ((u - w) j) ^ 2) ≤ ∑ i, ((f u - f w) i) ^ 2) :
    ℓ ≤ L := by
  have hnw : 0 < ‖u - w‖ := by
    have h2 : 0 < ‖u - w‖ ^ 2 := by rw [euclid_norm_sq]; exact hpos
    rcases (norm_nonneg (u - w)).eq_or_lt with h | h
    · exfalso; rw [← h] at h2; simp at h2
    · exact h
  have h1 : ℓ * ‖u - w‖ ≤ ‖f u - f w‖ := by
    have e : (ℓ * ‖u - w‖) ^ 2 ≤ ‖f u - f w‖ ^ 2 := by
      rw [mul_pow, euclid_norm_sq, euclid_norm_sq]
      exact hray
    calc ℓ * ‖u - w‖
        = Real.sqrt ((ℓ * ‖u - w‖) ^ 2) :=
          (Real.sqrt_sq (mul_nonneg hℓ (norm_nonneg _))).symm
      _ ≤ Real.sqrt (‖f u - f w‖ ^ 2) := Real.sqrt_le_sqrt e
      _ = ‖f u - f w‖ := Real.sqrt_sq (norm_nonneg _)
  exact le_of_mul_le_mul_right (h1.trans (hf u w)) hnw



-- ════════════════════════════════════════════════════════════
-- § Schatten-8: iterate the Gram trick once — ‖W‖₂ ≤ ‖G²‖_F^(1/4) = (Σσ⁸)^(1/8)
-- ════════════════════════════════════════════════════════════

/-- Sum-shuffle: `‖Aᵀy‖² = ⟨y, K y⟩` for `K = A·Aᵀ` supplied as data. The
    rearrangement engine both Gram bounds share. -/
theorem sum_sq_matTvec_eq {p q : ℕ} (A : Fin p → Fin q → ℝ) (y : Fin p → ℝ)
    (K : Fin p → Fin p → ℝ) (hK : ∀ a b, K a b = ∑ j, A a j * A b j) :
    ∑ j, (∑ i, A i j * y i) ^ 2 = ∑ a, y a * ∑ b, K a b * y b := by
  calc ∑ j, (∑ i, A i j * y i) ^ 2
      = ∑ j, ∑ a, ∑ b, (A a j * y a) * (A b j * y b) := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [pow_two, Finset.sum_mul_sum]
    _ = ∑ a, ∑ j, ∑ b, (A a j * y a) * (A b j * y b) := Finset.sum_comm
    _ = ∑ a, ∑ b, ∑ j, (A a j * y a) * (A b j * y b) := by
        exact Finset.sum_congr rfl fun a _ => Finset.sum_comm
    _ = ∑ a, ∑ b, (y a * y b) * ∑ j, A a j * A b j := by
        refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun j _ => by ring
    _ = ∑ a, y a * ∑ b, K a b * y b := by
        refine Finset.sum_congr rfl fun a _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun b _ => by rw [hK]; ring

/-- **Iterated Gram (Schatten-8) bound, proved.** One more squaring:
    with `G = W·Wᵀ` and `H = Gᵀ·G` (= `G²` for the symmetric `G`) supplied as
    data, `‖H‖_F² ≤ B⁸` gives `LipschitzL2 B (denseE W)` — i.e.
    `‖W‖₂ ≤ ‖G²‖_F^(1/4) = (Σσᵢ⁸)^(1/8)`, one Cauchy–Schwarz level tighter
    than the Schatten-4 bound. -/
theorem denseE_lipschitzL2_gram2 {n k : ℕ} (W : Fin k → Fin n → ℝ)
    (G : Fin k → Fin k → ℝ) (H : Fin k → Fin k → ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hG : ∀ a b, G a b = ∑ j, W a j * W b j)
    (hH : ∀ a b, H a b = ∑ c, G c a * G c b)
    (hHF : ∑ a, ∑ b, H a b ^ 2 ≤ B ^ 8) :
    LipschitzL2 B (denseE W) := by
  intro u w
  set d : Fin n → ℝ := fun j => u j - w j with hdd
  set y : Fin k → ℝ := fun i => ∑ j, W i j * d j with hyy
  set z : Fin n → ℝ := fun j => ∑ i, W i j * y i with hzz
  set S : ℝ := ∑ i, y i ^ 2 with hS
  set Dq : ℝ := ∑ j, d j ^ 2 with hDq
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hDq0 : 0 ≤ Dq := Finset.sum_nonneg fun j _ => sq_nonneg _
  -- S = ⟨d, Wᵀy⟩
  have hswap : S = ∑ j, d j * z j := by
    calc S = ∑ i, y i * ∑ j, W i j * d j := by
          exact Finset.sum_congr rfl fun i _ => by rw [pow_two]
      _ = ∑ i, ∑ j, y i * (W i j * d j) := by
          exact Finset.sum_congr rfl fun i _ => Finset.mul_sum ..
      _ = ∑ j, ∑ i, y i * (W i j * d j) := Finset.sum_comm
      _ = ∑ j, d j * z j := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [hzz, Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by ring
  -- T := Σz² = ⟨y, Gy⟩
  have hTz : ∑ j, z j ^ 2 = ∑ a, y a * ∑ b, G a b * y b :=
    sum_sq_matTvec_eq W y G hG
  have hT0 : 0 ≤ ∑ j, z j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
  -- CS1: S² ≤ Dq·T
  have hCS1 : S ^ 2 ≤ Dq * ∑ j, z j ^ 2 := by
    rw [hswap]
    exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  -- Q := Σ_a (Gy)_a² = ⟨y, Hy⟩  (the extra squaring level)
  have hQz : ∑ a, (∑ b, G a b * y b) ^ 2 = ∑ a, y a * ∑ b, H a b * y b := by
    have := sum_sq_matTvec_eq (fun i j => G j i) y H
      (fun a b => by rw [hH])
    simpa using this
  have hQ0 : 0 ≤ ∑ a, (∑ b, G a b * y b) ^ 2 :=
    Finset.sum_nonneg fun a _ => sq_nonneg _
  -- Q² ≤ S·(ΣH²·S) ≤ B⁸·S²
  have hQ2 : (∑ a, (∑ b, G a b * y b) ^ 2) ^ 2 ≤ B ^ 8 * S ^ 2 := by
    have h1 : (∑ a, (∑ b, G a b * y b) ^ 2) ^ 2
        ≤ S * ∑ a, (∑ b, H a b * y b) ^ 2 := by
      rw [hQz]
      exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    have h2 : ∑ a, (∑ b, H a b * y b) ^ 2 ≤ (∑ a, ∑ b, H a b ^ 2) * S :=
      sum_sq_matvec_le H y
    have h3 : (∑ a, ∑ b, H a b ^ 2) * S ≤ B ^ 8 * S :=
      mul_le_mul_of_nonneg_right hHF hS0
    calc (∑ a, (∑ b, G a b * y b) ^ 2) ^ 2
        ≤ S * ∑ a, (∑ b, H a b * y b) ^ 2 := h1
      _ ≤ S * (B ^ 8 * S) := mul_le_mul_of_nonneg_left (h2.trans h3) hS0
      _ = B ^ 8 * S ^ 2 := by ring
  -- Q ≤ B⁴·S
  have hQle : (∑ a, (∑ b, G a b * y b) ^ 2) ≤ B ^ 4 * S := by
    have hb4 : 0 ≤ B ^ 4 * S := mul_nonneg (by positivity) hS0
    nlinarith [hQ2, hQ0, hb4]
  -- T² ≤ S·Q ≤ B⁴·S² ⇒ T ≤ B²·S
  have hT2 : (∑ j, z j ^ 2) ^ 2 ≤ B ^ 4 * S ^ 2 := by
    have h1 : (∑ j, z j ^ 2) ^ 2 ≤ S * ∑ a, (∑ b, G a b * y b) ^ 2 := by
      rw [hTz]
      exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    calc (∑ j, z j ^ 2) ^ 2
        ≤ S * ∑ a, (∑ b, G a b * y b) ^ 2 := h1
      _ ≤ S * (B ^ 4 * S) := mul_le_mul_of_nonneg_left hQle hS0
      _ = B ^ 4 * S ^ 2 := by ring
  have hTle : (∑ j, z j ^ 2) ≤ B ^ 2 * S := by
    have hb2 : 0 ≤ B ^ 2 * S := mul_nonneg (sq_nonneg _) hS0
    nlinarith [hT2, hT0, hb2]
  -- S ≤ B²·Dq
  have hSle : S ≤ B ^ 2 * Dq := by
    rcases eq_or_lt_of_le hS0 with h0 | hpos
    · rw [← h0]; exact mul_nonneg (sq_nonneg _) hDq0
    · have : S ^ 2 ≤ Dq * (B ^ 2 * S) :=
        hCS1.trans (mul_le_mul_of_nonneg_left hTle hDq0)
      nlinarith [this, hpos]
  -- back to norms (identical tail to the Schatten-4 lemma)
  have hcoord : ∀ i, (denseE W u - denseE W w) i = y i := by
    intro i
    show (∑ j, W i j * u j) - (∑ j, W i j * w j) = _
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by
      show W i j * u j - W i j * w j = W i j * (u j - w j); ring
  have hnormsq : ‖denseE W u - denseE W w‖ ^ 2 ≤ (B * ‖u - w‖) ^ 2 := by
    rw [euclid_norm_sq, mul_pow, euclid_norm_sq]
    calc ∑ i, ((denseE W u - denseE W w) i) ^ 2
        = S := Finset.sum_congr rfl fun i _ => by rw [hcoord]
      _ ≤ B ^ 2 * Dq := hSle
      _ = B ^ 2 * ∑ j, ((u - w) j) ^ 2 := rfl
  calc ‖denseE W u - denseE W w‖
      = Real.sqrt (‖denseE W u - denseE W w‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt ((B * ‖u - w‖) ^ 2) := Real.sqrt_le_sqrt hnormsq
    _ = B * ‖u - w‖ := Real.sqrt_sq (mul_nonneg hB (norm_nonneg _))



/-- `f` is *certified at radius ε* on input `x` with class `i`: every perturbation of L2 norm
    `< ε` leaves `i` the strict argmax. The (undecidable — it quantifies over real `δ`)
    per-image certificate every scorecard's `certifiedC<i>` / `certifiedU<i>` theorems prove. -/
def CertifiedAt {n k : ℕ} (f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k))
    (ε : ℝ) (x : EuclideanSpace ℝ (Fin n)) (i : Fin k) : Prop :=
  ∀ δ : EuclideanSpace ℝ (Fin n), ‖δ‖ < ε →
    ∀ j, j ≠ i → f (x + δ) j < f (x + δ) i

end LipschitzCertDemo
end Proofs
