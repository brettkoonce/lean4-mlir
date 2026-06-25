import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.FloatBridge
import LeanMlir.Proofs.FloatComposeBridge

/-!
# ℝ→Float32 bridge: EfficientNet's smooth activations (Swish / sigmoid)

First step of the EfficientNet float bridge. enet is all-smooth (Swish + sigmoid
SE gate + conv/BN, no ReLU kinks) — so, like ViT, the float story is the clean
one (no sign-flip margins, just rounding). The conv/BN/GAP/residual machinery is
already built; the new ops are the activations.

The shared transcendental is **sigmoid** (`σ(x) = 1/(1+e^{-x})`), which drives
both Swish (`x·σ(x)`) and the SE gate. As with `exp`/`rsqrt`, the GPU `sigmoid`
has no IEEE spec, so it's modeled by a supplied `fsig` with accuracy `esig`
(`|fsig t − σ(t)| ≤ esig`, validated against silicon like `eexp`). Here: σ is
bounded in `(0,1)`, and the rounding-half closeness of `sigmoid`/`swish` at a
fixed input. (The input-sensitivity — σ is ¼-Lipschitz — is the next piece, the
analogue of `bnForward_input_close`.)
-/

namespace Proofs

/-- `σ(x) > 0`. -/
theorem sigmoidScalar_pos (x : ℝ) : 0 < sigmoidScalar x := by
  unfold sigmoidScalar; positivity

/-- `σ(x) < 1` (denominator `1 + e^{-x} > 1`). -/
theorem sigmoidScalar_lt_one (x : ℝ) : sigmoidScalar x < 1 := by
  unfold sigmoidScalar
  rw [div_lt_one (by positivity)]
  have : 0 < Real.exp (-x) := Real.exp_pos _
  linarith

/-- `|σ(x)| ≤ 1`. -/
theorem sigmoidScalar_abs_le_one (x : ℝ) : |sigmoidScalar x| ≤ 1 :=
  abs_le.mpr ⟨by linarith [sigmoidScalar_pos x], le_of_lt (sigmoidScalar_lt_one x)⟩

/-- `swish(x) = x · σ(x)` (the SiLU factorization). -/
theorem swishScalar_eq (x : ℝ) : swishScalar x = x * sigmoidScalar x := by
  unfold swishScalar sigmoidScalar; rw [mul_one_div]

/-- **Sigmoid rounding closeness.** The GPU `fsig` (within `esig` of `σ`) at any
    input is within `esig` of the certified `sigmoid` — per coordinate. -/
theorem sigmoid_close {n : Nat} (fsig : ℝ → ℝ) {esig : ℝ} (v : Vec n)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig) (i : Fin n) :
    |fsig (v i) - sigmoid n v i| ≤ esig := by
  unfold sigmoid; exact hsig (v i)

/-- **Swish rounding closeness.** `fl(xᵢ · fsig(xᵢ))` is within `mulErr u A 1 0 esig`
    of the certified `swish` — one `mul_close` (the input is exact, `σ ≤ 1`, so the
    only errors are the sigmoid accuracy `esig` and the product rounding). The
    Swish / SE-gate rounding budget. -/
theorem swish_close {n : Nat} (M : FloatModel) (fsig : ℝ → ℝ) {esig A : ℝ} (v : Vec n)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig) (hA : ∀ i, |v i| ≤ A) (i : Fin n) :
    |M.mul (v i) (fsig (v i)) - swish n v i| ≤ FloatModel.mulErr M.u A 1 0 esig := by
  unfold swish
  rw [swishScalar_eq]
  exact M.mul_close (by simp) (hsig (v i)) (hA i) (sigmoidScalar_abs_le_one (v i))

-- ════════════════════════════════════════════════════════════════
-- § Sigmoid input-sensitivity (σ is ¼-Lipschitz)
-- ════════════════════════════════════════════════════════════════

/-- `σ'(x) = e^{-x}/(1+e^{-x})²`. -/
theorem sigmoidScalar_hasDerivAt (x : ℝ) :
    HasDerivAt sigmoidScalar (Real.exp (-x) / (1 + Real.exp (-x)) ^ 2) x := by
  have h1 : HasDerivAt (fun x : ℝ => Real.exp (-x)) (-Real.exp (-x)) x := by
    simpa using (hasDerivAt_neg x).exp
  have he : HasDerivAt (fun x : ℝ => 1 + Real.exp (-x)) (-Real.exp (-x)) x := h1.const_add 1
  have h0 : (1 + Real.exp (-x)) ≠ 0 := by positivity
  have hd := (hasDerivAt_const x (1:ℝ)).div he h0
  have hval : (0 * (1 + Real.exp (-x)) - 1 * (-Real.exp (-x))) / (1 + Real.exp (-x)) ^ 2
      = Real.exp (-x) / (1 + Real.exp (-x)) ^ 2 := by ring
  rw [hval] at hd
  unfold sigmoidScalar
  exact hd

/-- `|σ'(x)| ≤ ¼` — the maximum of `e/(1+e)²` is `¼` at `e=1` (`(1-e)² ≥ 0`). -/
theorem sigmoidScalar_deriv_bound (x : ℝ) : |deriv sigmoidScalar x| ≤ 1/4 := by
  rw [(sigmoidScalar_hasDerivAt x).deriv]
  have he : 0 < Real.exp (-x) := Real.exp_pos _
  have hd : 0 < (1 + Real.exp (-x)) ^ 2 := by positivity
  rw [abs_of_pos (by positivity), div_le_iff₀ hd]
  nlinarith [sq_nonneg (1 - Real.exp (-x)), he]

/-- **σ is ¼-Lipschitz.** -/
theorem sigmoidScalar_lipschitz : LipschitzWith (1/4) sigmoidScalar := by
  apply lipschitzWith_of_nnnorm_deriv_le sigmoidScalar_diff
  intro x
  rw [← NNReal.coe_le_coe, coe_nnnorm, Real.norm_eq_abs]
  simpa using sigmoidScalar_deriv_bound x

/-- **σ input-sensitivity:** `|σ(a) − σ(b)| ≤ ¼·|a − b|`. The piece the SE-gate
    sigmoid's `FloatClose` input-shift needs (the analogue of `bnForward_input_close`). -/
theorem sigmoidScalar_lipschitz_abs (a b : ℝ) :
    |sigmoidScalar a - sigmoidScalar b| ≤ (1/4) * |a - b| := by
  have h := sigmoidScalar_lipschitz.dist_le_mul a b
  rwa [Real.dist_eq, Real.dist_eq] at h

/-- **Swish input-sensitivity (bounded domain).** `|swish(a) − swish(b)| ≤ (1+A/4)|a−b|`
    on `|a|,|b| ≤ A` — pure algebra: `a·σa − b·σb = a(σa−σb) + (a−b)σb`, then `σ` is
    ¼-Lipschitz and `σ ≤ 1`. No MVT needed. -/
theorem swishScalar_lipschitz_abs {a b A : ℝ} (ha : |a| ≤ A) (_hb : |b| ≤ A) :
    |swishScalar a - swishScalar b| ≤ (1 + A/4) * |a - b| := by
  rw [swishScalar_eq, swishScalar_eq]
  have hsplit : a * sigmoidScalar a - b * sigmoidScalar b
      = a * (sigmoidScalar a - sigmoidScalar b) + (a - b) * sigmoidScalar b := by ring
  rw [hsplit]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul, abs_mul]
  have h1 : |a| * |sigmoidScalar a - sigmoidScalar b| ≤ A * ((1/4) * |a - b|) :=
    mul_le_mul ha (sigmoidScalar_lipschitz_abs a b) (abs_nonneg _) ((abs_nonneg _).trans ha)
  have h2 : |a - b| * |sigmoidScalar b| ≤ |a - b| * 1 :=
    mul_le_mul_of_nonneg_left (sigmoidScalar_abs_le_one b) (abs_nonneg _)
  calc |a| * |sigmoidScalar a - sigmoidScalar b| + |a - b| * |sigmoidScalar b|
      ≤ A * ((1/4) * |a - b|) + |a - b| * 1 := add_le_add h1 h2
    _ = (1 + A/4) * |a - b| := by ring

/-- **Swish is `FloatClose`** — the enet smooth activation as a composable instance.
    Rounding from `swish_close` (the `mulErr` of `xᵢ·fsig(xᵢ)`), input-shift from
    `swishScalar_lipschitz_abs`. The smooth-world analogue of `floatClose_relu`. With
    `floatClose_flatConv` / the BN instance / the residual combinator, enet's MBConv
    main path folds through `.comp`. -/
theorem floatClose_swish {n : Nat} (M : FloatModel) (fsig : ℝ → ℝ) {esig A : ℝ}
    (hesig : 0 ≤ esig) (hA : 0 ≤ A) (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig) :
    FloatClose A (A + FloatModel.mulErr M.u A 1 0 esig)
      (swish n) (fun v i => M.mul (v i) (fsig (v i)))
      (fun e => FloatModel.mulErr M.u A 1 0 esig + (1 + A/4) * e) := by
  have hme0 : 0 ≤ FloatModel.mulErr M.u A 1 0 esig := by
    have := M.u_nonneg; unfold FloatModel.mulErr; nlinarith [mul_nonneg hA hesig]
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hround := swish_close M fsig v hsig hv i
    have hreal : |swish n v i| ≤ A := by
      unfold swish; rw [swishScalar_eq, abs_mul]
      calc |v i| * |sigmoidScalar (v i)|
          ≤ A * 1 := mul_le_mul (hv i) (sigmoidScalar_abs_le_one _) (abs_nonneg _) ((abs_nonneg _).trans (hv i))
        _ = A := by ring
    refine ⟨hreal.trans (le_add_of_nonneg_right hme0), ?_⟩
    have htri : |M.mul (v i) (fsig (v i))|
        ≤ |M.mul (v i) (fsig (v i)) - swish n v i| + |swish n v i| := by
      simpa using abs_sub_le (M.mul (v i) (fsig (v i))) (swish n v i) 0
    calc |M.mul (v i) (fsig (v i))|
        ≤ |M.mul (v i) (fsig (v i)) - swish n v i| + |swish n v i| := htri
      _ ≤ FloatModel.mulErr M.u A 1 0 esig + A := add_le_add hround hreal
      _ = A + FloatModel.mulErr M.u A 1 0 esig := by ring
  · have hround := swish_close M fsig vt hsig hvt i
    have hshift : |swish n vt i - swish n va i| ≤ (1 + A/4) * e := by
      unfold swish
      exact (swishScalar_lipschitz_abs (hvt i) (hva i)).trans
        (mul_le_mul_of_nonneg_left (hd i) (by positivity))
    calc |M.mul (vt i) (fsig (vt i)) - swish n va i|
        ≤ |M.mul (vt i) (fsig (vt i)) - swish n vt i| + |swish n vt i - swish n va i| :=
          abs_sub_le _ _ _
      _ ≤ FloatModel.mulErr M.u A 1 0 esig + (1 + A/4) * e := add_le_add hround hshift

-- ════════════════════════════════════════════════════════════════
-- § The Squeeze-Excitation gate (a multiplicative branch combinator)
-- ════════════════════════════════════════════════════════════════

/-- **SE scale `x ⊙ gate(x)` is `FloatClose`** — the multiplicative-branch
    combinator (residual's cousin: the input `x` is reused, gated by `gate(x)`).
    Given the (broadcast) gate map `g` is `FloatClose A Bg` — for a sigmoid gate
    `Bg = 1` — the block is `FloatClose` via `mul_close` (input error + gate error).
    The architecturally-distinctive EfficientNet op; the squeeze→excite gate net
    (`GAP → dense → swish → dense → sigmoid`, then broadcast) is the `.comp` chain
    feeding `g` here. -/
theorem floatClose_seScale {m : Nat} (M : FloatModel) {A Bg : ℝ}
    {g gF : Vec m → Vec m} {Lg : ℝ → ℝ} (hg : FloatClose A Bg g gF Lg) :
    FloatClose A (A * Bg + FloatModel.mulErr M.u A Bg 0 (Lg 0))
      (fun x i => x i * g x i)
      (fun x i => M.mul (x i) (gF x i))
      (fun e => FloatModel.mulErr M.u A Bg e (Lg e)) := by
  obtain ⟨hgm, hge⟩ := hg
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i =>
    M.mul_close (hd i) (hge vt va e hva hvt hd i) (hva i) (hgm va hva i).1⟩
  have hgv : |g v i| ≤ Bg := (hgm v hv i).1
  have hvi : |v i| ≤ A := hv i
  have hreal : |v i * g v i| ≤ A * Bg := by
    rw [abs_mul]; exact mul_le_mul hvi hgv (abs_nonneg _) ((abs_nonneg _).trans hvi)
  have hround : |M.mul (v i) (gF v i) - v i * g v i| ≤ FloatModel.mulErr M.u A Bg 0 (Lg 0) :=
    M.mul_close (by simp) (hge v v 0 hv hv (fun k => by simp) i) hvi hgv
  have hme0 : 0 ≤ FloatModel.mulErr M.u A Bg 0 (Lg 0) := (abs_nonneg _).trans hround
  refine ⟨hreal.trans (le_add_of_nonneg_right hme0), ?_⟩
  calc |M.mul (v i) (gF v i)|
      ≤ |M.mul (v i) (gF v i) - v i * g v i| + |v i * g v i| := by
        simpa using abs_sub_le (M.mul (v i) (gF v i)) (v i * g v i) 0
    _ ≤ FloatModel.mulErr M.u A Bg 0 (Lg 0) + A * Bg := add_le_add hround hreal
    _ = A * Bg + FloatModel.mulErr M.u A Bg 0 (Lg 0) := by ring

end Proofs
