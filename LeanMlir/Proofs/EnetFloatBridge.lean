import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.FloatBridge
import LeanMlir.Proofs.FloatComposeBridge
import LeanMlir.Proofs.DepthwiseFloatBridge

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

/-- **Sigmoid is `FloatClose`** — the SE gate's nonlinearity as a composable instance.
    The GPU `fsig` (within `esig` of `σ`) at any input is within `esig` of the certified
    `sigmoid` (rounding), and `σ` is ¼-Lipschitz (`sigmoidScalar_lipschitz_abs`) so the
    input-shift is `¼·e`. `σ ∈ (0,1)` ⇒ output magnitude `1 + esig` *regardless of input
    magnitude* `A` (the gate is bounded, the SE branch can't blow up). -/
theorem floatClose_sigmoid {n : Nat} (fsig : ℝ → ℝ) {esig A : ℝ}
    (hesig : 0 ≤ esig) (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig) :
    FloatClose A (1 + esig) (sigmoid n) (fun v i => fsig (v i))
      (fun e => esig + (1/4) * e) := by
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hr : |sigmoid n v i| ≤ 1 := by unfold sigmoid; exact sigmoidScalar_abs_le_one _
    have hf : |fsig (v i)| ≤ 1 + esig := by
      calc |fsig (v i)| ≤ |fsig (v i) - sigmoidScalar (v i)| + |sigmoidScalar (v i)| := by
            simpa using abs_sub_le (fsig (v i)) (sigmoidScalar (v i)) 0
        _ ≤ esig + 1 := add_le_add (hsig (v i)) (sigmoidScalar_abs_le_one _)
        _ = 1 + esig := by ring
    exact ⟨hr.trans (by linarith), hf⟩
  · have h2 : |sigmoidScalar (vt i) - sigmoidScalar (va i)| ≤ (1/4) * |vt i - va i| :=
      sigmoidScalar_lipschitz_abs _ _
    have h3 : (1/4) * |vt i - va i| ≤ (1/4) * e := mul_le_mul_of_nonneg_left (hd i) (by norm_num)
    unfold sigmoid
    calc |fsig (vt i) - sigmoidScalar (va i)|
        ≤ |fsig (vt i) - sigmoidScalar (vt i)| + |sigmoidScalar (vt i) - sigmoidScalar (va i)| :=
          abs_sub_le _ _ _
      _ ≤ esig + (1/4) * e := add_le_add (hsig (vt i)) (h2.trans h3)

/-- **Channel broadcast is `FloatClose` with modulus `id`** — `Vec c → Vec (c·h·w)`,
    the SE gate's expand-to-spatial. A pure reindex along `flatChannel` (every spatial
    cell of channel `k` copies `v k`), so it is exact in float and 1-Lipschitz, never
    growing magnitudes (the cousin of `floatClose_relu`/`_maxPool`). Turns the
    per-channel sigmoid gate into the `Vec (c·h·w)` that `floatClose_seScale` eats. -/
theorem floatClose_broadcast {c h w : Nat} (A : ℝ) :
    FloatClose A A (broadcastFlat c h w) (broadcastFlat c h w) (fun e => e) := by
  refine ⟨fun v hv idx => ?_, fun vt va e _ _ hd idx => ?_⟩
  · exact ⟨hv (flatChannel c h w idx), hv (flatChannel c h w idx)⟩
  · exact hd (flatChannel c h w idx)

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

/-- **THE SE GATE NET IS `FloatClose`** — `broadcast ∘ sigmoid ∘ dense ∘ swish ∘ dense ∘ GAP`
    folded entirely through `.comp` from the six per-op instances (`floatClose_gap`,
    `floatClose_dense` ×2, `floatClose_swish`, `floatClose_sigmoid`, `floatClose_broadcast`).
    Magnitudes thread `A → A+gap → dense → swish → dense → (1+esig)` (the sigmoid caps the
    gate at `1+esig` no matter the input scale); the modulus is the composition of the six.
    The squeeze-excite gate composes — no bespoke proof. -/
theorem floatClose_seGate {c h w r : Nat} (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' β A esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β) :
    ∃ B L gF, FloatClose A B (seGate (h := h) (w := w) W₁ b₁ W₂ b₂) gF L := by
  have hu := M.u_nonneg
  -- GAP: A → B0 (each `set` after its instance folds the instance's magnitude)
  have g0 := floatClose_gap (c := c) (h := h) (w := w) M hA hhw
  set B0 := A + (M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A)
    with hB0def
  have hB0 : 0 ≤ B0 := by
    rw [hB0def]
    have hpow : (1:ℝ) ≤ (1 + M.u) ^ (h * w + 1) := one_le_pow₀ (by linarith)
    have h1 : 0 ≤ M.u * ((1 + M.u) ^ (h * w + 1) * A) := by positivity
    have h2 : 0 ≤ ((1 + M.u) ^ (h * w + 1) - 1) * A := mul_nonneg (by linarith) hA
    linarith
  -- dense₁: B0 → B1
  have g1 := floatClose_dense M W₁ b₁ hw' hβ hB0 hc hW₁ hb₁
  set B1 := FloatModel.layerAct c w' β B0 + FloatModel.layerBudget M.u c w' β B0 0 with hB1def
  have hB1 : 0 ≤ B1 := by
    rw [hB1def]
    exact add_nonneg (FloatModel.layerAct_nonneg hw' hβ hB0)
      (FloatModel.layerBudget_nonneg M.u_nonneg hw' hβ hB0 le_rfl)
  -- swish: B1 → B2
  have g2 := floatClose_swish (n := r) M fsig hesig hB1 hsig
  set B2 := B1 + FloatModel.mulErr M.u B1 1 0 esig with hB2def
  have hB2 : 0 ≤ B2 := by
    rw [hB2def]
    have : 0 ≤ FloatModel.mulErr M.u B1 1 0 esig := by
      unfold FloatModel.mulErr; nlinarith [mul_nonneg hB1 hesig]
    linarith
  -- dense₂: B2 → B3, then sigmoid caps at 1+esig, then broadcast
  have g3 := floatClose_dense M W₂ b₂ hw' hβ hB2 hr hW₂ hb₂
  set B3 := FloatModel.layerAct r w' β B2 + FloatModel.layerBudget M.u r w' β B2 0 with hB3def
  have g4 := floatClose_sigmoid (n := c) (A := B3) fsig hesig hsig
  have g5 := floatClose_broadcast (c := c) (h := h) (w := w) (1 + esig)
  exact ⟨_, _, _, ((((g0.comp g1).comp g2).comp g3).comp g4).comp g5⟩

/-- **THE FULL SQUEEZE-EXCITE BLOCK `x ⊙ gate(x)` IS `FloatClose`** — `floatClose_seScale`
    fed the composed gate certificate `floatClose_seGate`. The architecturally-distinctive
    EfficientNet op, end to end: the gate net's fold multiplied back into the main path,
    one `FloatClose`. -/
theorem floatClose_seBlockFull {c h w r : Nat} (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' β A esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β) :
    ∃ B L Ff, FloatClose A B (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂) Ff L := by
  obtain ⟨Bg, Lg, gF, hgate⟩ :=
    floatClose_seGate M fsig W₁ b₁ W₂ b₂ hw' hβ hA hesig hhw hc hr hsig hW₁ hb₁ hW₂ hb₂
  exact ⟨_, _, _, floatClose_seScale M hgate⟩

-- ════════════════════════════════════════════════════════════════
-- § A closed smooth residual block (conv → swish → conv, + skip)
-- ════════════════════════════════════════════════════════════════

/-- **THE SMOOTH RESIDUAL FOLD: an enet-flavored `conv→swish→conv` block with the
    additive MBConv skip is `FloatClose`.** Body `conv₂ ∘ swish ∘ conv₁` folded via
    `.comp` (two conv `layerBudget` moduli through Swish's `(1+A/4)`-Lipschitz
    rounding), then wrapped by `floatClose_addResidual` into `F(x) + x` — one
    certificate, skip included. The smooth-world analogue of `floatClose_resBlock`
    (no ReLU kink, so no sign margins). The full MBConv inserts BN before each
    activation and the SE gate before the project conv; both are extra `.comp`s. -/
theorem floatClose_smoothResBlock {c h w kH kW : Nat} (M : FloatModel)
    (fsig : ℝ → ℝ) (W₁ W₂ : Kernel4 c c kH kW) (b₁ b₂ : Vec c)
    {w' β A esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hesig : 0 ≤ esig)
    (hn : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w') (hb₁ : ∀ o, |b₁ o| ≤ β)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w') (hb₂ : ∀ o, |b₂ o| ≤ β) :
    ∃ B L Ff, FloatClose A B
      (fun v => fun j => (flatConv (h := h) (w := w) W₂ b₂ ∘ swish (c*h*w)
        ∘ flatConv (h := h) (w := w) W₁ b₁) v j + v j)
      Ff L := by
  set B1 := FloatModel.layerAct (c*kH*kW) w' β A
    + FloatModel.layerBudget M.u (c*kH*kW) w' β A 0 with hB1def
  have hB1 : 0 ≤ B1 := by
    rw [hB1def]
    exact add_nonneg (FloatModel.layerAct_nonneg hw' hβ hA)
      (FloatModel.layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)
  have hB2 : 0 ≤ B1 + FloatModel.mulErr M.u B1 1 0 esig := by
    have : 0 ≤ FloatModel.mulErr M.u B1 1 0 esig := by
      have := M.u_nonneg; unfold FloatModel.mulErr; nlinarith [mul_nonneg hB1 hesig]
    linarith
  have conv₁ := floatClose_flatConv (h := h) (w := w) M W₁ b₁ hw' hβ hA hn hW₁ hb₁
  rw [← hB1def] at conv₁
  have hsw := floatClose_swish (n := c*h*w) M fsig hesig hB1 hsig
  have conv₂ := floatClose_flatConv (h := h) (w := w) M W₂ b₂ hw' hβ hB2 hn hW₂ hb₂
  exact ⟨_, _, _, floatClose_addResidual M ((conv₁.comp hsw).comp conv₂)⟩

-- ════════════════════════════════════════════════════════════════
-- § FloatBridges: the whole-MBConv fold (magnitude threading automated)
-- ════════════════════════════════════════════════════════════════

/-- Swish float-bridges (output magnitude `A + mulErr`). -/
theorem floatBridges_swish {n : Nat} (M : FloatModel) (fsig : ℝ → ℝ) {esig : ℝ}
    (hesig : 0 ≤ esig) (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig) :
    FloatBridges (swish n) := by
  intro A hA
  refine ⟨_, _, _, ?_, floatClose_swish M fsig hesig hA hsig⟩
  have hu := M.u_nonneg
  have : 0 ≤ FloatModel.mulErr M.u A 1 0 esig := by
    unfold FloatModel.mulErr; nlinarith [mul_nonneg hA hesig]
  linarith

/-- **The full Squeeze-Excite block float-bridges.** `floatClose_seBlockFull` plus the
    branch-magnitude nonnegativity (`cod_nonneg` / `modulus_zero_nonneg` of the gate),
    so the SE stage slots into the MBConv whole-net fold. -/
theorem floatBridges_seBlockFull {c h w r : Nat} (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' β esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β) :
    FloatBridges (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂) := by
  intro A hA
  obtain ⟨Bg, Lg, gF, hgc⟩ :=
    floatClose_seGate M fsig W₁ b₁ W₂ b₂ hw' hβ hA hesig hhw hc hr hsig hW₁ hb₁ hW₂ hb₂
  refine ⟨_, _, _, ?_, floatClose_seScale M hgc⟩
  have hBg : 0 ≤ Bg := hgc.cod_nonneg hA hn
  have hLg0 : 0 ≤ Lg 0 := hgc.modulus_zero_nonneg hA hn
  have hu := M.u_nonneg
  have hme : 0 ≤ FloatModel.mulErr M.u A Bg 0 (Lg 0) := by
    unfold FloatModel.mulErr
    nlinarith [mul_nonneg hA hLg0, mul_nonneg hu (mul_nonneg hA (add_nonneg hBg hLg0))]
  nlinarith [mul_nonneg hA hBg]

/-- **THE MBConv BODY FOLD.** The whole EfficientNet MBConv body
    `project∘BN ∘ SE ∘ (swish∘BN∘depthwise) ∘ (swish∘BN∘expand)` float-bridges, built by
    `FloatBridges.comp` from the per-stage bridges — expand/project conv, two swishes, the
    depthwise (§1c), and the full SE block (§1d). The three batch-norms enter as the
    operating-point hypotheses `FloatBridges (bnForward …)` (discharged by `floatClose_bn`
    fed `bnIstd_close_at`, §3) — the one piece coupled to the magnitude calibration.
    Magnitudes thread automatically: no per-stage `B` bookkeeping. -/
theorem floatBridges_mbconvBody
    {cin cmid cout h w kHe kWe kHd kWd kHp kWp r : Nat} (M : FloatModel) (fsig : ℝ → ℝ)
    (We : Kernel4 cmid cin kHe kWe) (be : Vec cmid) (εe γe βe : ℝ)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 cout cmid kHp kWp) (bp : Vec cout) (εp γp βp : ℝ)
    {w' β esig : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hcin : 0 < cin * h * w) (hcmid : 0 < cmid * h * w)
    (hcmid' : 0 < cmid) (hr : 0 < r)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ β)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ β)
    (hWs₁ : ∀ i j, |Ws₁ i j| ≤ w') (hbs₁ : ∀ j, |bs₁ j| ≤ β)
    (hWs₂ : ∀ i j, |Ws₂ i j| ≤ w') (hbs₂ : ∀ j, |bs₂ j| ≤ β)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ β)
    (hbnE : FloatBridges (bnForward (cmid * h * w) εe γe βe))
    (hbnD : FloatBridges (bnForward (cmid * h * w) εd γd βd))
    (hbnP : FloatBridges (bnForward (cout * h * w) εp γp βp)) :
    FloatBridges (mbconvBody We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp
      : Vec (cin * h * w) → Vec (cout * h * w)) := by
  have sE := floatBridges_flatConv (h := h) (w := w) M We be hw' hβ hcin hWe hbe
  have sSwE := floatBridges_swish (n := cmid * h * w) M fsig hesig hsig
  have sD := floatBridges_depthwise (h := h) (w := w) M Wd bd hw' hβ hcmid hWd hbd
  have sSwD := floatBridges_swish (n := cmid * h * w) M fsig hesig hsig
  have sSE := floatBridges_seBlockFull M fsig Ws₁ bs₁ Ws₂ bs₂ hw' hβ hesig hhw hcmid' hr hcmid
    hsig hWs₁ hbs₁ hWs₂ hbs₂
  have sP := floatBridges_flatConv (h := h) (w := w) M Wp bp hw' hβ hcmid hWp hbp
  exact (((((((sE.comp hbnE).comp sSwE).comp sD).comp hbnD).comp sSwD).comp sSE).comp sP).comp hbnP

end Proofs
