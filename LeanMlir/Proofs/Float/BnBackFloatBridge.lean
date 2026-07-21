import LeanMlir.Proofs.Float.BnFloatBridge

/-! # ℝ→Float32 bridge for the BatchNorm BACKWARD (A3 keystone — the "other side")

The forward BN bridge (`BnFloatBridge`/`FloatComposeBridge`) certifies `bnForward`'s float
closeness. This file opens the **backward** float story — the analogue for the certified
BN gradients that the §1a ties denote (`bn_grad_gamma`/`bn_grad_beta`/`bn_grad_input`).
Every deep net's backward (`r34`, `mnv2`, `efficientnet`, `convnext` LN, `vit` LN) folds
these, so — exactly as the forward BN bridge was the "do-it-once" forward keystone — these
are the shared backward keystones.

Two rungs here:
* **Parameter gradients** (`bnBetaGrad_close`, `bnGammaGrad_close`) — the easy reductions
  `dβ = Σ dy`, `dγ = Σ dy·x̂`, float-close via `sum_close` (+ `mul_close` and the supplied
  `x̂` closeness for `dγ`). The certified-backward *parameter* side.
* **Input gradient** (`bnGradInput_close`) — the genuinely-new op: the three-term
  `dx = (1/n)·s·(n·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))` (`dx̂ = γ·dy`, `s = istd`). Float-close by
  threading `mul_close`/`sum_close`/`M.err` through the assembly, the float inverse-stddev
  `fs` and float normalized `fxh` supplied with accuracy (discharged by the forward keystones
  `bnIstd_close` and the centered closeness at instantiation, exactly as the forward did).

All in the *supplied-stats* style of the forward keystone: `rsqrt` has no IEEE spec, so the
float `istd`/`x̂` are modelled (close within `es`/`exh`), not derived.
-/

namespace Proofs

open scoped Real

-- ════════════════════════════════════════════════════════════════
-- § Parameter gradients (the easy reductions)
-- ════════════════════════════════════════════════════════════════

/-- Float β-gradient: the rounded reduction `fl(Σ dyᵢ)`. -/
noncomputable def FloatModel.bnBetaGradF {n : Nat} (M : FloatModel) (dy : Vec n) : ℝ :=
  M.sum dy

/-- Float γ-gradient: the rounded reduction `fl(Σ dyᵢ ⊙ x̂ᵢ)` at a supplied float `x̂` (`fxh`). -/
noncomputable def FloatModel.bnGammaGradF {n : Nat} (M : FloatModel) (fxh dy : Vec n) : ℝ :=
  M.sum (fun i => M.mul (dy i) (fxh i))

/-- **BN β-gradient float closeness.** `fl(Σ dy)` is within the sum fan-in budget of `Σ dy`
    (`bn_grad_beta`). Pure `sum_close`. -/
theorem bnBetaGrad_close {n : Nat} (M : FloatModel) (dy : Vec n) {Cdy : ℝ}
    (hdy : ∀ i, |dy i| ≤ Cdy) :
    |M.bnBetaGradF dy - bn_grad_beta n dy|
      ≤ ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * Cdy) := by
  have hγn0 : 0 ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  have hbnd : ∑ i, |dy i| ≤ (n : ℝ) * Cdy := by
    calc ∑ i, |dy i| ≤ ∑ _i : Fin n, Cdy := Finset.sum_le_sum (fun i _ => hdy i)
      _ = (n : ℝ) * Cdy := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  show |M.sum dy - ∑ i, dy i| ≤ _
  calc |M.sum dy - ∑ i, dy i| ≤ ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |dy i| := M.sum_close dy
    _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * Cdy) := mul_le_mul_of_nonneg_left hbnd hγn0

/-- **BN γ-gradient float closeness.** `fl(Σ dy ⊙ fx̂)` (at a float normalized `fxh` within
    `exh` of the certified `bnXhat`) is within budget of `Σ dy·x̂` (`bn_grad_gamma`): one
    `mul_close` per term (`dy` exact, `fxh` within `exh`) lifted by `sum_close`. -/
theorem bnGammaGrad_close {n : Nat} (M : FloatModel) (ε : ℝ) (x dy fxh : Vec n)
    {Cdy Xh exh : ℝ} (hn : 0 < n)
    (hdy : ∀ i, |dy i| ≤ Cdy) (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh)
    (hfxh : ∀ i, |fxh i - bnXhat n ε x i| ≤ exh) :
    |M.bnGammaGradF fxh dy - bn_grad_gamma n ε x dy|
      ≤ ((1 + M.u) ^ (n + 1) - 1)
          * ((n : ℝ) * (Cdy * Xh + FloatModel.mulErr M.u Cdy Xh 0 exh))
        + (n : ℝ) * FloatModel.mulErr M.u Cdy Xh 0 exh := by
  set eg := FloatModel.mulErr M.u Cdy Xh 0 exh with heg
  have hu := M.u_nonneg
  have hγn0 : 0 ≤ (1 + M.u) ^ (n + 1) - 1 := sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hCdy0 : 0 ≤ Cdy := (abs_nonneg _).trans (hdy ⟨0, hn⟩)
  have hXh0 : 0 ≤ Xh := (abs_nonneg _).trans (hxh ⟨0, hn⟩)
  have hexh0 : 0 ≤ exh := (abs_nonneg _).trans (hfxh ⟨0, hn⟩)
  have heg0 : 0 ≤ eg := by rw [heg]; unfold FloatModel.mulErr; positivity
  -- per-term closeness and magnitude
  have hterm : ∀ i, |M.mul (dy i) (fxh i) - dy i * bnXhat n ε x i| ≤ eg := fun i =>
    M.mul_close (show |dy i - dy i| ≤ (0 : ℝ) by simp) (hfxh i) (hdy i) (hxh i)
  have hmag : ∀ i, |M.mul (dy i) (fxh i)| ≤ Cdy * Xh + eg := by
    intro i
    have htri := abs_sub_le (M.mul (dy i) (fxh i)) (dy i * bnXhat n ε x i) 0
    simp only [sub_zero] at htri
    have hreal : |dy i * bnXhat n ε x i| ≤ Cdy * Xh := by
      rw [abs_mul]; exact mul_le_mul (hdy i) (hxh i) (abs_nonneg _) hCdy0
    calc |M.mul (dy i) (fxh i)|
        ≤ |M.mul (dy i) (fxh i) - dy i * bnXhat n ε x i| + |dy i * bnXhat n ε x i| := htri
      _ ≤ eg + Cdy * Xh := add_le_add (hterm i) hreal
      _ = Cdy * Xh + eg := by ring
  -- the two reduction pieces
  have hsumabs : ∑ i, |M.mul (dy i) (fxh i)| ≤ (n : ℝ) * (Cdy * Xh + eg) := by
    calc ∑ i, |M.mul (dy i) (fxh i)| ≤ ∑ _i : Fin n, (Cdy * Xh + eg) :=
          Finset.sum_le_sum (fun i _ => hmag i)
      _ = (n : ℝ) * (Cdy * Xh + eg) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hsumdiff :
      |∑ i, M.mul (dy i) (fxh i) - ∑ i, dy i * bnXhat n ε x i| ≤ (n : ℝ) * eg := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ i, (M.mul (dy i) (fxh i) - dy i * bnXhat n ε x i)|
        ≤ ∑ i, |M.mul (dy i) (fxh i) - dy i * bnXhat n ε x i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin n, eg := Finset.sum_le_sum (fun i _ => hterm i)
      _ = (n : ℝ) * eg := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- assemble: rounding (sum_close) + per-term shift
  show |M.sum (fun i => M.mul (dy i) (fxh i)) - ∑ i, dy i * bnXhat n ε x i| ≤ _
  calc |M.sum (fun i => M.mul (dy i) (fxh i)) - ∑ i, dy i * bnXhat n ε x i|
      ≤ |M.sum (fun i => M.mul (dy i) (fxh i)) - ∑ i, M.mul (dy i) (fxh i)|
        + |∑ i, M.mul (dy i) (fxh i) - ∑ i, dy i * bnXhat n ε x i| := abs_sub_le _ _ _
    _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |M.mul (dy i) (fxh i)| + (n : ℝ) * eg :=
        add_le_add (M.sum_close _) hsumdiff
    _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (Cdy * Xh + eg)) + (n : ℝ) * eg :=
        add_le_add (mul_le_mul_of_nonneg_left hsumabs hγn0) le_rfl

-- ════════════════════════════════════════════════════════════════
-- § Reusable reduction closeness + magnitude (float Σ of close terms)
-- ════════════════════════════════════════════════════════════════

/-- **Reduction closeness.** If each float term `f i` is within `ef` of a real `r i`
    bounded by `Mr`, then `fl(Σ f)` is within `γₙ·n·(Mr+ef) + n·ef` of `Σ r` — the
    `sum_close` fan-in plus the per-term shift. The reduction peer of the BN-back sums. -/
theorem reduction_close {n : Nat} (M : FloatModel) (f r : Vec n) {Mr ef : ℝ}
    (hf : ∀ i, |f i - r i| ≤ ef) (hr : ∀ i, |r i| ≤ Mr) :
    |M.sum f - ∑ i, r i|
      ≤ ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (Mr + ef)) + (n : ℝ) * ef := by
  have hγn0 : 0 ≤ (1 + M.u) ^ (n + 1) - 1 := sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  have hmag : ∀ i, |f i| ≤ Mr + ef := by
    intro i
    have htri := abs_sub_le (f i) (r i) 0
    simp only [sub_zero] at htri
    calc |f i| ≤ |f i - r i| + |r i| := htri
      _ ≤ ef + Mr := add_le_add (hf i) (hr i)
      _ = Mr + ef := by ring
  have hsumabs : ∑ i, |f i| ≤ (n : ℝ) * (Mr + ef) := by
    calc ∑ i, |f i| ≤ ∑ _i : Fin n, (Mr + ef) := Finset.sum_le_sum (fun i _ => hmag i)
      _ = (n : ℝ) * (Mr + ef) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hsumdiff : |∑ i, f i - ∑ i, r i| ≤ (n : ℝ) * ef := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ i, (f i - r i)| ≤ ∑ i, |f i - r i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin n, ef := Finset.sum_le_sum (fun i _ => hf i)
      _ = (n : ℝ) * ef := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc |M.sum f - ∑ i, r i|
      ≤ |M.sum f - ∑ i, f i| + |∑ i, f i - ∑ i, r i| := abs_sub_le _ _ _
    _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |f i| + (n : ℝ) * ef := add_le_add (M.sum_close f) hsumdiff
    _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (Mr + ef)) + (n : ℝ) * ef :=
        add_le_add (mul_le_mul_of_nonneg_left hsumabs hγn0) le_rfl

/-- **Rounded subtraction closeness.** `fl(a ⊖ b)` is within `u·(Ma+Mb) + (ea+eb)` of the
    real `a'−b'` (`|a−a'|≤ea`, `|b−b'|≤eb`, `|a|≤Ma`, `|b|≤Mb`): the rounding `u·|a−b|`
    plus the two input shifts. -/
theorem sub_close' {a b a' b' Ma Mb ea eb : ℝ} (M : FloatModel)
    (ha : |a - a'| ≤ ea) (hb : |b - b'| ≤ eb) (hMa : |a| ≤ Ma) (hMb : |b| ≤ Mb) :
    |M.sub a b - (a' - b')| ≤ M.u * (Ma + Mb) + (ea + eb) := by
  have hround : |M.sub a b - (a - b)| ≤ M.u * (Ma + Mb) := by
    refine (M.err (a - b)).trans ?_
    exact mul_le_mul_of_nonneg_left ((abs_sub _ _).trans (add_le_add hMa hMb)) M.u_nonneg
  have hshift : |(a - b) - (a' - b')| ≤ ea + eb := by
    calc |(a - b) - (a' - b')| = |(a - a') - (b - b')| := by ring_nf
      _ ≤ |a - a'| + |b - b'| := abs_sub _ _
      _ ≤ ea + eb := add_le_add ha hb
  calc |M.sub a b - (a' - b')|
      ≤ |M.sub a b - (a - b)| + |(a - b) - (a' - b')| := abs_sub_le _ _ _
    _ ≤ M.u * (Ma + Mb) + (ea + eb) := add_le_add hround hshift

/-- Magnitude of a rounded subtraction: `|fl(a ⊖ b)| ≤ (1+u)(Ma+Mb)`. -/
theorem sub_mag {a b Ma Mb : ℝ} (M : FloatModel) (hMa : |a| ≤ Ma) (hMb : |b| ≤ Mb) :
    |M.sub a b| ≤ (1 + M.u) * (Ma + Mb) := by
  have hround : |M.sub a b - (a - b)| ≤ M.u * (Ma + Mb) := by
    refine (M.err (a - b)).trans ?_
    exact mul_le_mul_of_nonneg_left ((abs_sub _ _).trans (add_le_add hMa hMb)) M.u_nonneg
  have htri := abs_sub_le (M.sub a b) (a - b) 0
  simp only [sub_zero] at htri
  calc |M.sub a b| ≤ |M.sub a b - (a - b)| + |a - b| := htri
    _ ≤ M.u * (Ma + Mb) + (Ma + Mb) := add_le_add hround ((abs_sub _ _).trans (add_le_add hMa hMb))
    _ = (1 + M.u) * (Ma + Mb) := by ring

-- ════════════════════════════════════════════════════════════════
-- § Input gradient — the three-term BN backward (the keystone)
-- ════════════════════════════════════════════════════════════════

/-- Float three-term BN input-gradient at output index `i`, at a supplied float
    inverse-stddev `fs` and float normalized vector `fxh`. Mirrors `bn_grad_input`
    op-for-op in `M`-rounded arithmetic (`dx̂ = γ·dy`, `1/n` exact). -/
noncomputable def FloatModel.bnGradInputF {n : Nat} (M : FloatModel) (γ fs : ℝ)
    (fxh dy : Vec n) (i : Fin n) : ℝ :=
  M.mul (M.mul (1 / (n : ℝ)) fs)
    (M.sub
      (M.sub (M.mul (n : ℝ) (M.mul γ (dy i))) (M.sum (fun k => M.mul γ (dy k))))
      (M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k))))))

/-- The assembled three-term budget (the nested `mulErr`/reduction bound the proof yields). -/
noncomputable def FloatModel.bnGradInputBudget (M : FloatModel) (n : Nat)
    (G Cdy S Xh es exh : ℝ) : ℝ :=
  let u := M.u
  let γn := (1 + u) ^ (n + 1) - 1
  let MD := G * Cdy
  let eD := FloatModel.mulErr u G Cdy 0 0
  let eSD := γn * ((n : ℝ) * (MD + eD)) + (n : ℝ) * eD
  let MSD := (n : ℝ) * MD + eSD
  let MXD := Xh * MD
  let eXD := FloatModel.mulErr u Xh MD exh eD
  let eSXD := γn * ((n : ℝ) * (MXD + eXD)) + (n : ℝ) * eXD
  let enD := FloatModel.mulErr u (n : ℝ) MD 0 eD
  let MnD := (n : ℝ) * MD + enD
  let e1 := u * (MnD + MSD) + (enD + eSD)
  let M1 := (1 + u) * (MnD + MSD)
  let eXS := FloatModel.mulErr u Xh ((n : ℝ) * MXD) exh eSXD
  let MXSf := Xh * ((n : ℝ) * MXD) + eXS
  let e2 := u * (M1 + MXSf) + (e1 + eXS)
  let MTr := (n : ℝ) * MD + (n : ℝ) * MD + Xh * ((n : ℝ) * MXD)
  let eP := FloatModel.mulErr u (1 / (n : ℝ)) S 0 es
  FloatModel.mulErr u (1 / (n : ℝ) * S) MTr eP e2

/-- **BN input-gradient float closeness (the backward keystone).** The deployed float
    three-term input-gradient is within `bnGradInputBudget` of the certified
    `bn_grad_input`, with the float inverse-stddev `fs` and normalized `fxh` supplied
    close (`es`/`exh` — discharged by the forward `bnIstd_close` + centered closeness at
    instantiation). The genuinely-new backward op every deep net's gradient folds. -/
theorem bnGradInput_close {n : Nat} (M : FloatModel) {ε γ : ℝ} (x dy fxh : Vec n) (fs : ℝ)
    {G Cdy S Xh es exh : ℝ} (hn : 0 < n)
    (hγ : |γ| ≤ G) (hdy : ∀ i, |dy i| ≤ Cdy)
    (hs : |fs - bnIstd n x ε| ≤ es) (hSabs : |bnIstd n x ε| ≤ S)
    (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh) (hfxh : ∀ i, |fxh i - bnXhat n ε x i| ≤ exh)
    (i : Fin n) :
    |M.bnGradInputF γ fs fxh dy i - bn_grad_input n ε γ x dy i|
      ≤ M.bnGradInputBudget n G Cdy S Xh es exh := by
  have hu := M.u_nonneg
  have hnR : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hCdy0 : 0 ≤ Cdy := (abs_nonneg _).trans (hdy i)
  have hXh0 : 0 ≤ Xh := (abs_nonneg _).trans (hxh i)
  have hexh0 : 0 ≤ exh := (abs_nonneg _).trans (hfxh i)
  -- real / float per-coordinate dxhat = γ·dy
  set MD := G * Cdy with hMDdef
  set eD := FloatModel.mulErr M.u G Cdy 0 0 with hEDdef
  have hdxf : ∀ k, |M.mul γ (dy k) - γ * dy k| ≤ eD := fun k =>
    M.mul_close (by simp) (by simp) hγ (hdy k)
  have hdxr : ∀ k, |γ * dy k| ≤ MD := by
    intro k; rw [abs_mul]; exact mul_le_mul hγ (hdy k) (abs_nonneg _) hG0
  -- reductions Σ dxhat  and  Σ x̂·dxhat
  set eSD := ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (MD + eD)) + (n : ℝ) * eD with hESDdef
  have hsumD : |M.sum (fun k => M.mul γ (dy k)) - ∑ k, γ * dy k| ≤ eSD :=
    reduction_close M _ _ hdxf hdxr
  have hsumDr_mag : |∑ k, γ * dy k| ≤ (n : ℝ) * MD := by
    calc |∑ k, γ * dy k| ≤ ∑ k, |γ * dy k| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _k : Fin n, MD := Finset.sum_le_sum (fun k _ => hdxr k)
      _ = (n : ℝ) * MD := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hsumDf_mag : |M.sum (fun k => M.mul γ (dy k))| ≤ (n : ℝ) * MD + eSD := by
    have htri := abs_sub_le (M.sum (fun k => M.mul γ (dy k))) (∑ k, γ * dy k) 0
    simp only [sub_zero] at htri
    calc |M.sum (fun k => M.mul γ (dy k))|
        ≤ |M.sum (fun k => M.mul γ (dy k)) - ∑ k, γ * dy k| + |∑ k, γ * dy k| := htri
      _ ≤ eSD + (n : ℝ) * MD := add_le_add hsumD hsumDr_mag
      _ = (n : ℝ) * MD + eSD := by ring
  set MXD := Xh * MD with hMXDdef
  set eXD := FloatModel.mulErr M.u Xh MD exh eD with hEXDdef
  have hxdf : ∀ k, |M.mul (fxh k) (M.mul γ (dy k)) - bnXhat n ε x k * (γ * dy k)| ≤ eXD :=
    fun k => M.mul_close (hfxh k) (hdxf k) (hxh k) (hdxr k)
  have hxdr : ∀ k, |bnXhat n ε x k * (γ * dy k)| ≤ MXD := by
    intro k; rw [abs_mul]; exact mul_le_mul (hxh k) (hdxr k) (abs_nonneg _) hXh0
  set eSXD := ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (MXD + eXD)) + (n : ℝ) * eXD with hESXDdef
  have hsumXD : |M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k)))
      - ∑ k, bnXhat n ε x k * (γ * dy k)| ≤ eSXD := reduction_close M _ _ hxdf hxdr
  have hsumXDr_mag : |∑ k, bnXhat n ε x k * (γ * dy k)| ≤ (n : ℝ) * MXD := by
    calc |∑ k, bnXhat n ε x k * (γ * dy k)| ≤ ∑ k, |bnXhat n ε x k * (γ * dy k)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _k : Fin n, MXD := Finset.sum_le_sum (fun k _ => hxdr k)
      _ = (n : ℝ) * MXD := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- n·dxhat_i
  set enD := FloatModel.mulErr M.u (n : ℝ) MD 0 eD with hEnDdef
  have hnd : |M.mul (n : ℝ) (M.mul γ (dy i)) - (n : ℝ) * (γ * dy i)| ≤ enD :=
    M.mul_close (by simp) (hdxf i) (by rw [abs_of_nonneg hnR]) (hdxr i)
  have hndr_mag : |(n : ℝ) * (γ * dy i)| ≤ (n : ℝ) * MD := by
    rw [abs_mul, abs_of_nonneg hnR]; exact mul_le_mul_of_nonneg_left (hdxr i) hnR
  have hndf_mag : |M.mul (n : ℝ) (M.mul γ (dy i))| ≤ (n : ℝ) * MD + enD := by
    have htri := abs_sub_le (M.mul (n : ℝ) (M.mul γ (dy i))) ((n : ℝ) * (γ * dy i)) 0
    simp only [sub_zero] at htri
    calc |M.mul (n : ℝ) (M.mul γ (dy i))|
        ≤ |M.mul (n : ℝ) (M.mul γ (dy i)) - (n : ℝ) * (γ * dy i)| + |(n : ℝ) * (γ * dy i)| := htri
      _ ≤ enD + (n : ℝ) * MD := add_le_add hnd hndr_mag
      _ = (n : ℝ) * MD + enD := by ring
  -- first subtraction  (n·dxhat_i) ⊖ Σdxhat
  set MnD := (n : ℝ) * MD + enD with hMnDdef
  set MSD := (n : ℝ) * MD + eSD with hMSDdef
  have hsub1 : |M.sub (M.mul (n : ℝ) (M.mul γ (dy i))) (M.sum (fun k => M.mul γ (dy k)))
      - ((n : ℝ) * (γ * dy i) - ∑ k, γ * dy k)| ≤ M.u * (MnD + MSD) + (enD + eSD) :=
    sub_close' M hnd hsumD hndf_mag hsumDf_mag
  have hsub1_mag : |M.sub (M.mul (n : ℝ) (M.mul γ (dy i))) (M.sum (fun k => M.mul γ (dy k)))|
      ≤ (1 + M.u) * (MnD + MSD) := sub_mag M hndf_mag hsumDf_mag
  -- x̂_i · Σ(x̂·dxhat)
  set eXS := FloatModel.mulErr M.u Xh ((n : ℝ) * MXD) exh eSXD with hEXSdef
  have hxs : |M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k))))
      - bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)| ≤ eXS :=
    M.mul_close (hfxh i) hsumXD (hxh i) hsumXDr_mag
  have hxsr_mag : |bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)| ≤ Xh * ((n : ℝ) * MXD) := by
    rw [abs_mul]; exact mul_le_mul (hxh i) hsumXDr_mag (abs_nonneg _) hXh0
  have hxsf_mag : |M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k))))|
      ≤ Xh * ((n : ℝ) * MXD) + eXS := by
    have htri := abs_sub_le (M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k)))))
      (bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)) 0
    simp only [sub_zero] at htri
    calc |M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k))))|
        ≤ |M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k))))
            - bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)|
          + |bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)| := htri
      _ ≤ eXS + Xh * ((n : ℝ) * MXD) := add_le_add hxs hxsr_mag
      _ = Xh * ((n : ℝ) * MXD) + eXS := by ring
  -- second subtraction  T = (first) ⊖ (x̂_i·Σ)
  set M1 := (1 + M.u) * (MnD + MSD) with hM1def
  set MXSf := Xh * ((n : ℝ) * MXD) + eXS with hMXSfdef
  set e1 := M.u * (MnD + MSD) + (enD + eSD) with hE1def
  have hT : |M.sub
        (M.sub (M.mul (n : ℝ) (M.mul γ (dy i))) (M.sum (fun k => M.mul γ (dy k))))
        (M.mul (fxh i) (M.sum (fun k => M.mul (fxh k) (M.mul γ (dy k)))))
      - (((n : ℝ) * (γ * dy i) - ∑ k, γ * dy k)
          - bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k))|
      ≤ M.u * (M1 + MXSf) + (e1 + eXS) :=
    sub_close' M hsub1 hxs hsub1_mag hxsf_mag
  have hTr_mag : |((n : ℝ) * (γ * dy i) - ∑ k, γ * dy k)
      - bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)|
      ≤ (n : ℝ) * MD + (n : ℝ) * MD + Xh * ((n : ℝ) * MXD) := by
    calc |((n : ℝ) * (γ * dy i) - ∑ k, γ * dy k)
            - bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)|
        ≤ |(n : ℝ) * (γ * dy i) - ∑ k, γ * dy k|
          + |bnXhat n ε x i * ∑ k, bnXhat n ε x k * (γ * dy k)| := abs_sub _ _
      _ ≤ ((n : ℝ) * MD + (n : ℝ) * MD) + Xh * ((n : ℝ) * MXD) :=
          add_le_add ((abs_sub _ _).trans (add_le_add hndr_mag hsumDr_mag)) hxsr_mag
      _ = (n : ℝ) * MD + (n : ℝ) * MD + Xh * ((n : ℝ) * MXD) := by ring
  -- the scale  (1/n)·s
  set eP := FloatModel.mulErr M.u (1 / (n : ℝ)) S 0 es with hEPdef
  have hps : |M.mul (1 / (n : ℝ)) fs - 1 / (n : ℝ) * bnIstd n x ε| ≤ eP :=
    M.mul_close (by simp) hs (by rw [abs_of_nonneg (by positivity)]) hSabs
  have hpr_mag : |1 / (n : ℝ) * bnIstd n x ε| ≤ 1 / (n : ℝ) * S := by
    rw [abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / (n : ℝ))]
    exact mul_le_mul_of_nonneg_left hSabs (by positivity)
  -- final product
  set MTr := (n : ℝ) * MD + (n : ℝ) * MD + Xh * ((n : ℝ) * MXD) with hMTrdef
  set e2 := M.u * (M1 + MXSf) + (e1 + eXS) with hE2def
  have hfinal := M.mul_close hps hT hpr_mag hTr_mag
  -- bn_grad_input = (1/n · s) · T  and  bnGradInputF = M.mul (M.mul (1/n) fs) (…)
  show |M.bnGradInputF γ fs fxh dy i - bn_grad_input n ε γ x dy i|
      ≤ M.bnGradInputBudget n G Cdy S Xh es exh
  simp only [FloatModel.bnGradInputF, bn_grad_input, FloatModel.bnGradInputBudget]
  exact hfinal

end Proofs
