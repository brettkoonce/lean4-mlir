import LeanMlir.Proofs.Architectures.LayerNorm
import LeanMlir.Proofs.Float.FloatComposeBridge

/-!
# ℝ→Float32 bridge: ViT — LayerNorm (a re-axis BN port) + GELU

ViT is all-smooth, like EfficientNet, so the float story is the clean one (no
sign-flip margins). The two ViT-specific primitives:

* **LayerNorm** (§2a). In this repo `layerNormForward n ε γ β = bnForward n ε γ β`
  *definitionally* (LN normalizes per-token over the feature axis; BN's per-example
  `bnForward` is exactly that reduction). So the whole BN float bridge — the `rsqrt`
  keystone, the operating-point `bnIstd_close_at`, `bnStep_close` and `floatClose_bn` —
  ports verbatim: `floatClose_layerNorm` is literally `floatClose_bn`.
* **GELU** (§2b), the one new transcendental. The tanh-form GELU
  `½·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` is bounded-Lipschitz on `|·| ≤ A` by the
  *same algebra* as Swish (`swishScalar_lipschitz_abs`): split, use `tanh` 1-Lipschitz
  and the cubic bound `|a²+ab+b²| ≤ 3A²` — no global derivative analysis. Plus the
  rounding half `gelu_close` (the GPU `fgelu` within `egelu` of `geluScalar`, like
  `eexp`/`esig`). Then `floatClose_gelu` is the composable instance.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 2a. LayerNorm = the BN bridge, re-axed (definitional)
-- ════════════════════════════════════════════════════════════════

/-- **LayerNorm is `FloatClose`** — `layerNormForward = bnForward` definitionally
    (per-token feature-axis reduction), so this is exactly `floatClose_bn`: error from
    `bnStep_close` (rounding + input-shift), float-magnitude from `bnForward_close_of`,
    modulus `bnReluBudget`. The expensive ViT primitive (LN/`rsqrt`) is reused, not
    rebuilt. Feed the operating-point `bnIstd_close_at` for the `eistd` to be non-vacuous. -/
theorem floatClose_layerNorm {m : Nat} (M : FloatModel)
    {ε γ β emean eistd D S G Bbnd A : ℝ} (fμ fistdv : Vec m → ℝ)
    (hn : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ v, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean)
    (histd : ∀ v, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd)
    (hD : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |v j - bnMean m v| ≤ D)
    (hSabs : ∀ v, (∀ k, |v k| ≤ A) → |bnIstd m v ε| ≤ S) :
    FloatClose A (G * (D * S) + Bbnd + bnNormBudget M.u D S G Bbnd emean eistd)
      (fun v => layerNormForward m ε γ β v)
      (fun v => M.bnForwardF γ β (fμ v) (fistdv v) v)
      (fun e => FloatModel.bnReluBudget M.u D S G Bbnd emean eistd A e ε) :=
  floatClose_bn M fμ fistdv hn hε hγ hβ hmean histd hD hSabs

-- ════════════════════════════════════════════════════════════════
-- § 2b. GELU: tanh 1-Lipschitz → bounded-domain GELU Lipschitz
-- ════════════════════════════════════════════════════════════════

/-- `|tanh'(x)| = |1 − tanh²x| ≤ 1` (`|tanh x| < 1`). -/
theorem Real.tanh_deriv_abs_le (x : ℝ) : |deriv Real.tanh x| ≤ 1 := by
  rw [(Real.hasDerivAt_tanh x).deriv]
  have h1 : |Real.tanh x| < 1 := abs_lt.mpr ⟨Real.neg_one_lt_tanh x, Real.tanh_lt_one x⟩
  have h2 : Real.tanh x ^ 2 < 1 := by nlinarith [sq_abs (Real.tanh x), abs_nonneg (Real.tanh x)]
  rw [abs_of_nonneg (by nlinarith [sq_nonneg (Real.tanh x)])]; linarith [sq_nonneg (Real.tanh x)]

/-- **`tanh` is 1-Lipschitz.** -/
theorem Real.tanh_lipschitz : LipschitzWith 1 Real.tanh := by
  apply lipschitzWith_of_nnnorm_deriv_le Real.differentiable_tanh
  intro x
  rw [← NNReal.coe_le_coe, coe_nnnorm, Real.norm_eq_abs]
  simpa using Real.tanh_deriv_abs_le x

/-- **`tanh` input-sensitivity:** `|tanh a − tanh b| ≤ |a − b|`. -/
theorem Real.tanh_lipschitz_abs (a b : ℝ) : |Real.tanh a - Real.tanh b| ≤ |a - b| := by
  have h := Real.tanh_lipschitz.dist_le_mul a b
  rw [Real.dist_eq, Real.dist_eq] at h; simpa using h

/-- **GELU input-sensitivity (bounded domain).** On `|a|,|b| ≤ A`,
    `|gelu(a) − gelu(b)| ≤ (1 + (√(2/π)/2)·A·(1 + 3·0.044715·A²))·|a − b|` — pure algebra:
    split `gelu(a) − gelu(b) = ½[(a−b)(1+tanh g_b) + a(tanh g_a − tanh g_b)]`, then `tanh`
    is 1-Lipschitz, `|1+tanh| ≤ 2`, and `|a²+ab+b²| ≤ 3A²` for the cubic inner term. The
    GELU analogue of `swishScalar_lipschitz_abs`; no MVT, no global derivative analysis. -/
theorem geluScalar_lipschitz_abs {a b A : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ A) :
    |geluScalar a - geluScalar b| ≤
      (1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * |a - b| := by
  have hA0 : 0 ≤ A := (abs_nonneg _).trans ha
  have hs0 : 0 ≤ Real.sqrt (2 / Real.pi) := Real.sqrt_nonneg _
  set s := Real.sqrt (2 / Real.pi) with hsdef
  set ga := s * (a + 0.044715 * a ^ 3) with hgadef
  set gb := s * (b + 0.044715 * b ^ 3) with hgbdef
  have hsplit : geluScalar a - geluScalar b
      = 0.5 * ((a - b) * (1 + Real.tanh gb) + a * (Real.tanh ga - Real.tanh gb)) := by
    unfold geluScalar; rw [← hgadef, ← hgbdef]; ring
  -- the two factor bounds
  have hb1 : |1 + Real.tanh gb| ≤ 2 := by
    have h1 := Real.neg_one_lt_tanh gb; have h2 := Real.tanh_lt_one gb
    rw [abs_le]; constructor <;> linarith
  have hquad : |1 + 0.044715 * (a ^ 2 + a * b + b ^ 2)| ≤ 1 + 3 * 0.044715 * A ^ 2 := by
    have hqa := abs_le.mp ha
    have hqb := abs_le.mp hb
    have hq0 : 0 ≤ a ^ 2 + a * b + b ^ 2 := by nlinarith [sq_nonneg (a + b), sq_nonneg a, sq_nonneg b]
    have haa : a ^ 2 ≤ A ^ 2 := by nlinarith [hqa.1, hqa.2]
    have hbb : b ^ 2 ≤ A ^ 2 := by nlinarith [hqb.1, hqb.2]
    have hab : a * b ≤ A ^ 2 := by nlinarith [sq_nonneg (a - b), haa, hbb]
    rw [abs_of_nonneg (by positivity)]; nlinarith [haa, hbb, hab]
  have hgdiff : |Real.tanh ga - Real.tanh gb| ≤ s * (1 + 3 * 0.044715 * A ^ 2) * |a - b| := by
    refine (Real.tanh_lipschitz_abs ga gb).trans ?_
    have hfac : ga - gb = s * ((a - b) * (1 + 0.044715 * (a ^ 2 + a * b + b ^ 2))) := by
      rw [hgadef, hgbdef]; ring
    rw [hfac, abs_mul, abs_mul, abs_of_nonneg hs0]
    calc s * (|a - b| * |1 + 0.044715 * (a ^ 2 + a * b + b ^ 2)|)
        ≤ s * (|a - b| * (1 + 3 * 0.044715 * A ^ 2)) := by
          gcongr
      _ = s * (1 + 3 * 0.044715 * A ^ 2) * |a - b| := by ring
  -- assemble
  rw [hsplit, abs_mul, show |(0.5 : ℝ)| = 0.5 from by norm_num]
  have hbr : |(a - b) * (1 + Real.tanh gb) + a * (Real.tanh ga - Real.tanh gb)|
      ≤ |a - b| * 2 + A * (s * (1 + 3 * 0.044715 * A ^ 2) * |a - b|) := by
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul]
    exact add_le_add
      (mul_le_mul_of_nonneg_left hb1 (abs_nonneg _))
      (mul_le_mul ha hgdiff (abs_nonneg _) hA0)
  calc 0.5 * |(a - b) * (1 + Real.tanh gb) + a * (Real.tanh ga - Real.tanh gb)|
      ≤ 0.5 * (|a - b| * 2 + A * (s * (1 + 3 * 0.044715 * A ^ 2) * |a - b|)) :=
        mul_le_mul_of_nonneg_left hbr (by norm_num)
    _ = (1 + s / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * |a - b| := by ring

/-- **GELU rounding closeness.** The GPU `fgelu` (within `egelu` of the certified
    `geluScalar`) at any input is within `egelu` of `gelu` — per coordinate (the
    `eexp`/`esig`/`ers` pattern; validate `egelu` empirically against silicon). -/
theorem gelu_close {n : Nat} (fgelu : ℝ → ℝ) {egelu : ℝ} (v : Vec n)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) (i : Fin n) :
    |fgelu (v i) - gelu n v i| ≤ egelu := by
  unfold gelu; exact hg (v i)

/-- **GELU is `FloatClose`** — the ViT MLP nonlinearity as a composable instance.
    Rounding from `gelu_close`, input-shift from `geluScalar_lipschitz_abs`. The
    smooth-world analogue of `floatClose_relu`/`floatClose_swish`; with `floatClose_dense`
    and `floatClose_layerNorm`, the ViT MLP block `dense→gelu→dense` folds through `.comp`. -/
theorem floatClose_gelu {n : Nat} (fgelu : ℝ → ℝ) {egelu A : ℝ}
    (hegelu : 0 ≤ egelu) (hA : 0 ≤ A) (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) :
    FloatClose A (A + egelu)
      (gelu n) (fun v i => fgelu (v i))
      (fun e => egelu + (1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * e) := by
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · -- magnitudes: |gelu| ≤ A (½x(1+tanh) with |1+tanh|≤2 ⟹ ≤ |x|), float ≤ A + egelu
    have hAi : |v i| ≤ A := hv i
    have hgs : |geluScalar (v i)| ≤ A := by
      set t := Real.tanh (Real.sqrt (2 / Real.pi) * (v i + 0.044715 * (v i) ^ 3)) with ht
      have htanh : |1 + t| ≤ 2 := by
        have h1 : -1 < t := by rw [ht]; exact Real.neg_one_lt_tanh _
        have h2 : t < 1 := by rw [ht]; exact Real.tanh_lt_one _
        rw [abs_le]; constructor <;> linarith
      have heq : |geluScalar (v i)| = 0.5 * |v i| * |1 + t| := by
        unfold geluScalar; rw [← ht, abs_mul, abs_mul, show |(0.5:ℝ)| = 0.5 from by norm_num]
      rw [heq]
      have hmul : |v i| * |1 + t| ≤ A * 2 := mul_le_mul hAi htanh (abs_nonneg _) hA
      nlinarith [hmul, abs_nonneg (v i), abs_nonneg (1 + t)]
    have hreal : |gelu n v i| ≤ A := hgs
    have hf : |fgelu (v i)| ≤ A + egelu := by
      calc |fgelu (v i)| ≤ |fgelu (v i) - geluScalar (v i)| + |geluScalar (v i)| := by
            simpa using abs_sub_le (fgelu (v i)) (geluScalar (v i)) 0
        _ ≤ egelu + A := add_le_add (hg (v i)) hgs
        _ = A + egelu := by ring
    exact ⟨hreal.trans (by linarith), hf⟩
  · -- error
    have h2 : |geluScalar (vt i) - geluScalar (va i)|
        ≤ (1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * |vt i - va i| :=
      geluScalar_lipschitz_abs (hvt i) (hva i)
    have hL0 : 0 ≤ 1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2) := by
      have := Real.sqrt_nonneg (2 / Real.pi); positivity
    have h3 : (1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * |vt i - va i|
        ≤ (1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * e :=
      mul_le_mul_of_nonneg_left (hd i) hL0
    unfold gelu
    calc |fgelu (vt i) - geluScalar (va i)|
        ≤ |fgelu (vt i) - geluScalar (vt i)| + |geluScalar (vt i) - geluScalar (va i)| :=
          abs_sub_le _ _ _
      _ ≤ egelu + (1 + Real.sqrt (2 / Real.pi) / 2 * A * (1 + 3 * 0.044715 * A ^ 2)) * e :=
          add_le_add (hg (vt i)) (h2.trans h3)

/-- GELU float-bridges (output magnitude `A + egelu`). -/
theorem floatBridges_gelu {n : Nat} (fgelu : ℝ → ℝ) {egelu : ℝ}
    (hegelu : 0 ≤ egelu) (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) :
    FloatBridges (gelu n) := by
  intro A hA
  exact ⟨_, _, _, by linarith, floatClose_gelu fgelu hegelu hA hg⟩

-- ════════════════════════════════════════════════════════════════
-- § 2d. The ViT MLP residual sub-block (per-token, Vec-space)
-- ════════════════════════════════════════════════════════════════

/-- **THE ViT MLP RESIDUAL SUB-BLOCK FOLD.** The per-token feed-forward half of a
    transformer block — `LN → dense → GELU → dense`, wrapped by the additive skip —
    float-bridges, built by `FloatBridges.comp` from the per-stage bridges (LayerNorm
    enters as the operating-point `FloatBridges (layerNormForward …)` hypothesis, like
    the MBConv BNs, discharged by `floatClose_layerNorm` + `bnIstd_close_at`, §3). The
    attention half mixes across tokens (Mat-space) and is the separate §2c track; this
    is the all-smooth Vec-space half, magnitudes threaded automatically. -/
theorem floatBridges_vitMlpResidual {d dff : Nat} (M : FloatModel)
    (W₁ : Mat d dff) (b₁ : Vec dff) (W₂ : Mat dff d) (b₂ : Vec d) (fgelu : ℝ → ℝ)
    {εln γln βln w' β egelu : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hegelu : 0 ≤ egelu) (hd : 0 < d) (hdff : 0 < dff)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β)
    (hln : FloatBridges (layerNormForward d εln γln βln)) :
    FloatBridges (Proofs.residual
      (Proofs.dense W₂ b₂ ∘ gelu dff ∘ Proofs.dense W₁ b₁ ∘ layerNormForward d εln γln βln)) :=
  FloatBridges.residual M
    (((hln.comp (floatBridges_dense M W₁ b₁ hw' hβ hd hW₁ hb₁)).comp
        (floatBridges_gelu (n := dff) fgelu hegelu hg)).comp
      (floatBridges_dense M W₂ b₂ hw' hβ hdff hW₂ hb₂))

end Proofs
