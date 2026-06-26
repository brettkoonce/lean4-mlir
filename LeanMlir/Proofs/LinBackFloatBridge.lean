import LeanMlir.Proofs.FloatComposeBridge

/-! # ℝ→Float32 bridge for the BACKWARD: linear input-VJP + ReLU-back → whole-net fold

A3 (planning/tier23…): the deep-net backward float story, the input-gradient (VJP) side.
The backward of a feed-forward net at a smooth point is itself a *forward* composition of
maps on the cotangent — so it folds through the **same** `FloatBridges.comp` backbone the
forward uses. The two op bridges needed:

* **Linear input-VJP** `dx = Wᵀ·dy` (`floatBridges_linBack`). A dense layer's input gradient
  is a bias-free dense over the **transposed** weight (`dense (Mat.transpose W) 0`), so it
  float-bridges *for free* via `floatBridges_dense` — no new proof, just the recognition.
  The conv input-VJP (reversed-kernel conv) bridges the same way via `floatBridges_flatConv`.
* **ReLU backward** `dx = select(preact>0, dy, 0)` (`floatBridges_reluMaskBack`). The rendered
  `selectPos` mask: pass `dy i` where the saved pre-activation was positive, else 0. A pure
  select — **exact in float** (no arithmetic), 1-Lipschitz, magnitude-nonincreasing, exactly
  like the forward `relu`/`maxPool`. The mask is a fixed parameter: the smooth-point assumption
  (float and real pre-activations agree in sign), mirroring the §1a backward ties' nonzero-kink
  hypotheses.

Capstone `mlpInputGrad_floatBridges`: the whole 3-layer MLP input-gradient VJP
`Wᵀ₀·(mask₁ ⊙ Wᵀ₁·(mask₂ ⊙ Wᵀ₂·dy))` float-bridges — "the deployed float backward map is
within an explicit budget of the certified real backward map." The backward peer of
`cifar8_floatBridges`, assembled in one `.comp` chain. Pair with the BatchNorm backward
(`BnBackFloatBridge`) for the BN nets.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § ReLU backward: the exact `selectPos` mask
-- ════════════════════════════════════════════════════════════════

/-- ReLU backward (the rendered `selectPos`): keep `dy i` where the saved pre-activation was
    positive (`cond i`), else 0. A select — exact in float. The mask `cond` is fixed (the
    smooth-point common sign pattern of the real and float pre-activations). -/
noncomputable def reluMaskBack {n : Nat} (cond : Fin n → Prop) [DecidablePred cond]
    (dy : Vec n) : Vec n :=
  fun i => if cond i then dy i else 0

/-- The select never increases magnitude: `|reluMaskBack cond v i| ≤ |v i|`. -/
theorem reluMaskBack_abs_le {n : Nat} (cond : Fin n → Prop) [DecidablePred cond]
    (v : Vec n) (i : Fin n) : |reluMaskBack cond v i| ≤ |v i| := by
  unfold reluMaskBack
  by_cases h : cond i
  · simp [if_pos h]
  · simp [if_neg h]

/-- **ReLU backward is `FloatClose` with modulus `id`** — exact in float (real = float map),
    1-Lipschitz, magnitude-nonincreasing. The backward peer of `floatClose_relu`. -/
theorem floatClose_reluMaskBack {n : Nat} (cond : Fin n → Prop) [DecidablePred cond] (A : ℝ) :
    FloatClose A A (reluMaskBack cond) (reluMaskBack cond) (fun e => e) := by
  refine ⟨fun v hv i => ⟨(reluMaskBack_abs_le cond v i).trans (hv i),
      (reluMaskBack_abs_le cond v i).trans (hv i)⟩, fun vt va e _ _ hd i => ?_⟩
  unfold reluMaskBack
  by_cases h : cond i
  · simp only [if_pos h]; exact hd i
  · simp only [if_neg h, sub_zero, abs_zero]; exact (abs_nonneg _).trans (hd i)

/-- ReLU backward float-bridges (magnitude-stable, exact). -/
theorem floatBridges_reluMaskBack {n : Nat} (cond : Fin n → Prop) [DecidablePred cond] :
    FloatBridges (reluMaskBack cond) :=
  fun A hA => ⟨A, _, _, hA, floatClose_reluMaskBack cond A⟩

-- ════════════════════════════════════════════════════════════════
-- § Smooth-activation backward: the diagonal `dy ⊙ act'(saved)` scale
-- ════════════════════════════════════════════════════════════════

/-- Smooth-activation backward (the rendered `emitActBack`/`scale`): multiply the cotangent
    pointwise by the **saved derivative** `s = act'(preact)`. GELU, Swish/SiLU and sigmoid all have
    a diagonal Jacobian, so their backward is this single `multiply`. A fixed vector `s` (the
    smooth-point saved derivative). -/
noncomputable def diagBack {n : Nat} (s : Vec n) (dy : Vec n) : Vec n := fun i => s i * dy i

/-- Float smooth-activation backward at a supplied float derivative `fs` (within `es` of `s` — the
    transcendental budgets `esig`/`egelu`, since the activation derivatives have no IEEE spec). -/
noncomputable def FloatModel.diagBackF {n : Nat} (M : FloatModel) (fs : Vec n) (dy : Vec n) :
    Vec n := fun i => M.mul (fs i) (dy i)

/-- **Smooth-activation backward is `FloatClose`** (over the cotangent). The deployed float
    `dy ↦ fl(fsᵢ · dyᵢ)` (at a float derivative `fs` within `es` of the saved `s`, `|s| ≤ Sd`) is
    within `mulErr(A) + Sd·e` of the certified `dy ↦ sᵢ · dyᵢ`, both outputs bounded by
    `Sd·A + mulErr(A)`. The map is linear in `dy` (one `mul_close` per coordinate), so its modulus
    is the per-coordinate rounding `mulErr` plus the real Lipschitz `Sd·e`. Covers GELU/Swish/
    sigmoid backward (diagonal Jacobian); the smooth peer of `floatClose_reluMaskBack`. -/
theorem floatClose_diagBack {n : Nat} (M : FloatModel) (s fs : Vec n) {Sd es A : ℝ}
    (hs : ∀ i, |s i| ≤ Sd) (hfs : ∀ i, |fs i - s i| ≤ es) :
    FloatClose A (Sd * A + FloatModel.mulErr M.u Sd A es 0)
      (diagBack s) (M.diagBackF fs)
      (fun e => FloatModel.mulErr M.u Sd A es 0 + Sd * e) := by
  have hu := M.u_nonneg
  refine ⟨fun v hv i => ?_, fun vt va e _ hvt hd i => ?_⟩
  · have hSd0 : 0 ≤ Sd := (abs_nonneg _).trans (hs i)
    have hAge : 0 ≤ A := (abs_nonneg _).trans (hv i)
    have hes0 : 0 ≤ es := (abs_nonneg _).trans (hfs i)
    have hmerr : 0 ≤ FloatModel.mulErr M.u Sd A es 0 := by
      unfold FloatModel.mulErr; positivity
    have hreal : |s i * v i| ≤ Sd * A := by
      rw [abs_mul]; exact mul_le_mul (hs i) (hv i) (abs_nonneg _) hSd0
    have hclose : |M.mul (fs i) (v i) - s i * v i| ≤ FloatModel.mulErr M.u Sd A es 0 :=
      M.mul_close (hfs i) (by simp) (hs i) (hv i)
    have hstep : |M.mul (fs i) (v i)| ≤ |M.mul (fs i) (v i) - s i * v i| + |s i * v i| := by
      have h := abs_sub_le (M.mul (fs i) (v i)) (s i * v i) 0
      rwa [sub_zero, sub_zero] at h
    refine ⟨hreal.trans (le_add_of_nonneg_right hmerr), ?_⟩
    calc |M.diagBackF fs v i|
        = |M.mul (fs i) (v i)| := rfl
      _ ≤ |M.mul (fs i) (v i) - s i * v i| + |s i * v i| := hstep
      _ ≤ FloatModel.mulErr M.u Sd A es 0 + Sd * A := add_le_add hclose hreal
      _ = Sd * A + FloatModel.mulErr M.u Sd A es 0 := by ring
  · have hSd0 : 0 ≤ Sd := (abs_nonneg _).trans (hs i)
    have hclose : |M.mul (fs i) (vt i) - s i * vt i| ≤ FloatModel.mulErr M.u Sd A es 0 :=
      M.mul_close (hfs i) (by simp) (hs i) (hvt i)
    have hrealdiff : |s i * vt i - s i * va i| ≤ Sd * e := by
      rw [← mul_sub, abs_mul]; exact mul_le_mul (hs i) (hd i) (abs_nonneg _) hSd0
    calc |M.diagBackF fs vt i - diagBack s va i|
        ≤ |M.mul (fs i) (vt i) - s i * vt i| + |s i * vt i - s i * va i| := abs_sub_le _ _ _
      _ ≤ FloatModel.mulErr M.u Sd A es 0 + Sd * e := add_le_add hclose hrealdiff

/-- Smooth-activation backward float-bridges. The GELU/Swish/sigmoid backward op (diagonal
    Jacobian) every smooth net's gradient folds; instantiate `s := act'(saved preact)`, `es :=`
    the activation's transcendental budget (`egelu`/`esig`). -/
theorem floatBridges_diagBack {n : Nat} (M : FloatModel) (s fs : Vec n) {Sd es : ℝ}
    (hn : 0 < n) (hs : ∀ i, |s i| ≤ Sd) (hfs : ∀ i, |fs i - s i| ≤ es) :
    FloatBridges (diagBack s) := fun A hA =>
  ⟨_, _, _, (floatClose_diagBack M s fs hs hfs (A := A)).cod_nonneg hA hn,
    floatClose_diagBack M s fs hs hfs⟩

-- ════════════════════════════════════════════════════════════════
-- § Linear input-VJP: `dx = Wᵀ·dy` = bias-free dense over the transpose
-- ════════════════════════════════════════════════════════════════

/-- **The dense input-VJP float-bridges** — `dx = Wᵀ·dy` is `dense (Mat.transpose W) 0`, so this
    is `floatBridges_dense` at the transposed weight (`|Wᵀ i j| = |W j i| ≤ w'`) and zero bias.
    The backward of `dense W b : Vec m → Vec n` is this map `Vec n → Vec m`. -/
theorem floatBridges_linBack {m n : Nat} (M : FloatModel) (W : Mat m n) {w' : ℝ}
    (hw' : 0 ≤ w') (hn : 0 < n) (hW : ∀ i j, |W i j| ≤ w') :
    FloatBridges (dense (Mat.transpose W) (0 : Vec m)) :=
  floatBridges_dense M (Mat.transpose W) 0 hw' le_rfl hn (fun i j => hW j i) (fun j => by simp)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net fold: a 3-layer MLP input-gradient VJP
-- ════════════════════════════════════════════════════════════════

/-- The 3-layer MLP input-gradient VJP at a smooth point: `dy ↦ Wᵀ₀·(mask₁ ⊙ Wᵀ₁·(mask₂ ⊙
    Wᵀ₂·dy))`. The certified backward of `dense W₂ ∘ relu ∘ dense W₁ ∘ relu ∘ dense W₀`
    (input gradient), the ReLU kinks read off the fixed sign masks `c₁`/`c₂`. -/
noncomputable def mlpInputGrad {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (c₁ : Fin d₁ → Prop) [DecidablePred c₁] (c₂ : Fin d₂ → Prop) [DecidablePred c₂] :
    Vec d₃ → Vec d₀ :=
  dense (Mat.transpose W₀) 0 ∘ reluMaskBack c₁ ∘ dense (Mat.transpose W₁) 0
    ∘ reluMaskBack c₂ ∘ dense (Mat.transpose W₂) 0

/-- **The whole MLP input-gradient VJP float-bridges.** Assembled in one `.comp` chain over the
    per-op backward bridges — `floatBridges_linBack` (each layer's `Wᵀ·dy`) and
    `floatBridges_reluMaskBack` (each ReLU's exact `selectPos` mask). The deployed float backward
    map is within an explicit budget of the certified real backward map. The backward peer of
    `cifar8_floatBridges`; closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem mlpInputGrad_floatBridges {d₀ d₁ d₂ d₃ : Nat} (M : FloatModel)
    (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (c₁ : Fin d₁ → Prop) [DecidablePred c₁] (c₂ : Fin d₂ → Prop) [DecidablePred c₂]
    {w' : ℝ} (hw' : 0 ≤ w')
    (hW₀ : ∀ i j, |W₀ i j| ≤ w') (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hW₂ : ∀ i j, |W₂ i j| ≤ w')
    (hd₁ : 0 < d₁) (hd₂ : 0 < d₂) (hd₃ : 0 < d₃) :
    FloatBridges (mlpInputGrad W₀ W₁ W₂ c₁ c₂) := by
  unfold mlpInputGrad
  exact (((floatBridges_linBack M W₂ hw' hd₃ hW₂
    |>.comp (floatBridges_reluMaskBack c₂))
    |>.comp (floatBridges_linBack M W₁ hw' hd₂ hW₁))
    |>.comp (floatBridges_reluMaskBack c₁))
    |>.comp (floatBridges_linBack M W₀ hw' hd₁ hW₀)

end Proofs
