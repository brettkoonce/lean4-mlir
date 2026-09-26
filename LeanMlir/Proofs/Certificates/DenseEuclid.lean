import LeanMlir.Proofs.Certificates.LipschitzCert

/-! # The dense Euclidean engine — dense and ReLU layers on `EuclideanSpace`, and their L2 bounds

The layers every Lipschitz / interval certificate in `Certificates/` is stated about: `denseE W`
(bias-free dense, `(Wx)ᵢ`) and `reluE`, with the L2 Lipschitz bounds the product certificate
multiplies — Frobenius (`denseE_lipschitzL2`), the Gram / Schatten-4 bound
(`denseE_lipschitzL2_gram`), Schatten-8 (`denseE_lipschitzL2_gram2`), and the lower bound a
witness vector gives (`lipschitzL2_lower_euclid`). The upper bounds share one tail
(`denseE_lipschitzL2_of_sq`) and the Gram ones one Cauchy–Schwarz step (`sq_le_of_gram_quad`,
`quad_le_of_frob`). `certified_at_eps` specialises the
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

/-- Row-wise Cauchy–Schwarz summed: `‖Mv‖² ≤ ‖M‖_F²·‖v‖²` at the raw-sum level. -/
theorem sum_sq_matvec_le {k n : ℕ} (M : Fin k → Fin n → ℝ) (y : Fin n → ℝ) :
    ∑ a, (∑ b, M a b * y b) ^ 2 ≤ (∑ a, ∑ b, M a b ^ 2) * (∑ b, y b ^ 2) := by
  calc ∑ a, (∑ b, M a b * y b) ^ 2
      ≤ ∑ a, ((∑ b, M a b ^ 2) * (∑ b, y b ^ 2)) :=
        Finset.sum_le_sum fun a _ => Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    _ = (∑ a, ∑ b, M a b ^ 2) * (∑ b, y b ^ 2) := (Finset.sum_mul ..).symm

/-- **The common tail of every dense bound**: a raw-sum bound `‖Wd‖² ≤ B²·‖d‖²` for every `d`
    makes the dense layer `B`-Lipschitz in L2. The Frobenius and Gram bounds below differ only in
    how they prove `hW`. -/
theorem denseE_lipschitzL2_of_sq {n k : ℕ} (W : Fin k → Fin n → ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hW : ∀ d : Fin n → ℝ, ∑ i, (∑ j, W i j * d j) ^ 2 ≤ B ^ 2 * ∑ j, d j ^ 2) :
    LipschitzL2 B (denseE W) := by
  intro u w
  have hsq : ‖denseE W u - denseE W w‖ ^ 2 ≤ (B * ‖u - w‖) ^ 2 := by
    rw [euclid_norm_sq, mul_pow, euclid_norm_sq]
    refine le_of_eq_of_le (Finset.sum_congr rfl fun i _ => ?_) (hW fun j => u j - w j)
    show ((∑ j, W i j * u j) - ∑ j, W i j * w j) ^ 2 = _
    rw [← Finset.sum_sub_distrib]; simp only [mul_sub]
  exact (abs_le_of_sq_le_sq' hsq (mul_nonneg hB (norm_nonneg _))).2

/-- **Frobenius bound, proved.** If the entrywise square sum of `W` is at
    most `C²`, the dense layer is `C`-Lipschitz in L2. This is the certified
    replacement for the power-iteration estimate `specNormW`: `‖W‖₂ ≤ ‖W‖_F`,
    so any rational `C ≥ ‖W‖_F` is a sound Lipschitz constant. -/
theorem denseE_lipschitzL2 {n k : ℕ} (W : Fin k → Fin n → ℝ) {C : ℝ}
    (hC : 0 ≤ C) (hW : ∑ i, ∑ j, W i j ^ 2 ≤ C ^ 2) :
    LipschitzL2 C (denseE W) :=
  denseE_lipschitzL2_of_sq W hC fun d => (sum_sq_matvec_le W d).trans <|
    mul_le_mul_of_nonneg_right hW (Finset.sum_nonneg fun _ _ => sq_nonneg _)

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


-- ════════════════════════════════════════════════════════════
-- § Gram bounds: Schatten-4 ‖W‖₂ ≤ ‖G‖_F^(1/2), Schatten-8 ‖W‖₂ ≤ ‖G²‖_F^(1/4)
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

/-- Cauchy–Schwarz on a quadratic form: `‖My‖² ≤ c²·‖y‖²` gives `⟨y, My⟩ ≤ c·‖y‖²`. -/
theorem quad_le_of_sq_matvec {k : ℕ} (M : Fin k → Fin k → ℝ) (y : Fin k → ℝ) {c : ℝ}
    (hc : 0 ≤ c) (hM : ∑ a, (∑ b, M a b * y b) ^ 2 ≤ c ^ 2 * ∑ a, y a ^ 2) :
    ∑ a, y a * ∑ b, M a b * y b ≤ c * ∑ a, y a ^ 2 := by
  have hS0 : 0 ≤ ∑ a, y a ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  refine (abs_le_of_sq_le_sq' ?_ (mul_nonneg hc hS0)).2
  calc (∑ a, y a * ∑ b, M a b * y b) ^ 2
      ≤ (∑ a, y a ^ 2) * ∑ a, (∑ b, M a b * y b) ^ 2 := Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    _ ≤ (∑ a, y a ^ 2) * (c ^ 2 * ∑ a, y a ^ 2) := mul_le_mul_of_nonneg_left hM hS0
    _ = (c * ∑ a, y a ^ 2) ^ 2 := by ring

/-- The Frobenius form of `quad_le_of_sq_matvec`: `‖M‖_F ≤ c` gives `⟨y, My⟩ ≤ c·‖y‖²`. -/
theorem quad_le_of_frob {k : ℕ} (M : Fin k → Fin k → ℝ) (y : Fin k → ℝ) {c : ℝ}
    (hc : 0 ≤ c) (hMF : ∑ a, ∑ b, M a b ^ 2 ≤ c ^ 2) :
    ∑ a, y a * ∑ b, M a b * y b ≤ c * ∑ a, y a ^ 2 :=
  quad_le_of_sq_matvec M y hc <| (sum_sq_matvec_le M y).trans <|
    mul_le_mul_of_nonneg_right hMF (Finset.sum_nonneg fun _ _ => sq_nonneg _)

/-- **The Gram step**: with `G = W·Wᵀ` and `y = Wd`, a bound `⟨y, Gy⟩ ≤ c·‖y‖²` gives
    `‖y‖² ≤ c·‖d‖²` — because `‖y‖² = ⟨d, Wᵀy⟩ ≤ ‖d‖·‖Wᵀy‖` and `‖Wᵀy‖² = ⟨y, Gy⟩`. -/
theorem sq_le_of_gram_quad {n k : ℕ} (W : Fin k → Fin n → ℝ) (G : Fin k → Fin k → ℝ)
    (hG : ∀ a b, G a b = ∑ j, W a j * W b j) (d : Fin n → ℝ) {c : ℝ} (hc : 0 ≤ c)
    (hq : ∑ a, (∑ j, W a j * d j) * ∑ b, G a b * ∑ j, W b j * d j
      ≤ c * ∑ a, (∑ j, W a j * d j) ^ 2) :
    ∑ i, (∑ j, W i j * d j) ^ 2 ≤ c * ∑ j, d j ^ 2 := by
  set y : Fin k → ℝ := fun i => ∑ j, W i j * d j with hy
  set S := ∑ i, y i ^ 2
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hDq0 : 0 ≤ ∑ j, d j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
  have hswap : S = ∑ j, d j * ∑ i, W i j * y i := by
    calc S = ∑ i, ∑ j, y i * (W i j * d j) := by
          exact Finset.sum_congr rfl fun i _ => by rw [pow_two, ← Finset.mul_sum]
      _ = ∑ j, ∑ i, y i * (W i j * d j) := Finset.sum_comm
      _ = _ := Finset.sum_congr rfl fun j _ => by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by ring
  have h : S ^ 2 ≤ (∑ j, d j ^ 2) * (c * S) := by
    calc S ^ 2 ≤ (∑ j, d j ^ 2) * ∑ j, (∑ i, W i j * y i) ^ 2 := by
          rw [hswap]; exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
      _ ≤ _ := mul_le_mul_of_nonneg_left (by rw [sum_sq_matTvec_eq W y G hG]; exact hq) hDq0
  rcases hS0.eq_or_lt with h0 | hpos
  · rw [← h0]; exact mul_nonneg hc hDq0
  · nlinarith

/-- **Gram (Schatten-4) bound, proved.** If `G = W·Wᵀ` (supplied as data, verified
    entrywise) and `‖G‖_F² ≤ B⁴`, then the dense layer is `B`-Lipschitz in L2.
    Since `‖G‖_F = (Σᵢσᵢ⁴)^½`, this is `‖W‖₂ ≤ (Σσᵢ⁴)^¼` — strictly tighter than
    Frobenius `(Σσᵢ²)^½` whenever `W` has rank ≥ 2. The Gram matrix is
    only `k×k` (output-side), so the kernel arithmetic stays small even for wide
    layers. -/
theorem denseE_lipschitzL2_gram {n k : ℕ} (W : Fin k → Fin n → ℝ)
    (G : Fin k → Fin k → ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hG : ∀ a b, G a b = ∑ j, W a j * W b j)
    (hGF : ∑ a, ∑ b, G a b ^ 2 ≤ B ^ 4) :
    LipschitzL2 B (denseE W) :=
  denseE_lipschitzL2_of_sq W hB fun d => sq_le_of_gram_quad W G hG d (sq_nonneg B) <|
    quad_le_of_frob G _ (sq_nonneg B) (by rwa [← pow_mul])

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
    LipschitzL2 B (denseE W) :=
  denseE_lipschitzL2_of_sq W hB fun d => sq_le_of_gram_quad W G hG d (sq_nonneg B) <| by
    -- `‖Gy‖² = ⟨y, Hy⟩ ≤ B⁴·‖y‖²`, then Cauchy–Schwarz once more
    refine quad_le_of_sq_matvec G _ (sq_nonneg B) ?_
    rw [sum_sq_matTvec_eq (fun i j => G j i) (fun i => ∑ j, W i j * d j) H
      (fun a b => by rw [hH]), ← pow_mul]
    exact quad_le_of_frob H _ (by positivity) (by rwa [← pow_mul])


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



/-- `f` is *certified at radius ε* on input `x` with class `i`: every perturbation of L2 norm
    `< ε` leaves `i` the strict argmax. The (undecidable — it quantifies over real `δ`)
    per-image certificate every scorecard's `certifiedC<i>` / `certifiedU<i>` theorems prove. -/
def CertifiedAt {n k : ℕ} (f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k))
    (ε : ℝ) (x : EuclideanSpace ℝ (Fin n)) (i : Fin k) : Prop :=
  ∀ δ : EuclideanSpace ℝ (Fin n), ‖δ‖ < ε →
    ∀ j, j ≠ i → f (x + δ) j < f (x + δ) i

end LipschitzCertDemo
end Proofs
