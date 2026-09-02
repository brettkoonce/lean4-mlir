import LeanMlir.Proofs.Float.BnPerChannelFloatBridge

/-! # ℝ → Float32 bridge: inference BatchNorm **as the render emits it**

`BnEvalFloatBridge.lean` models deployed BN as a fixed per-coordinate affine `a·x + b` with
`a = γ/√(σ²+ε)` folded offline, and says so: *"there is no batch reduction and no runtime
`rsqrt`"*. The first half is right and is the whole point; the second half is not what this
repo's renderer emits. `@resnet34_fwd_eval` (and every other `_fwd_eval`) expands
`.bnPerChannelEvalF` into six runtime ops — `subtract`, `add`, `rsqrt`, `multiply`,
`multiply`, `add` (`StableHLO.lean`, the `bnPerChannelEvalF` pretty case) — i.e. it computes
`γ·((x − μ)·rsqrt(var + ε)) + β` on device, with `μ` and `var` as frozen graph inputs. The
pre-folded affine is a *different float program* from the deployed one, so a budget proved
about it is a budget about a kernel we do not ship.

This file bridges the kernel we do ship. The float peer is `FloatModel.bnForwardF γ β μ sF` —
the SAME rounded normalize chain the training-mode bridge uses — evaluated at the frozen `μ`
and at a supplied float inverse-stddev `sF` (a `rsqrt`, so modelled with an accuracy `es`,
exactly as `fexp`/`fsig` are).

⭐ **Why this leaf and not the training-mode one decides whether a whole-net number exists.**
Training-mode BN reduces its statistics out of its own input, so perturbing the input moves
the mean AND the variance, and `bnReluBudget` carries the resulting term
`G·2A·(8A·e/(2ε√ε))` — **quadratic in the window**. Folded through 33 BN sites the budget
squares at each one: the ResNet-34 interval fold's modulus comes out at ~10⁷⁴¹⁷, past the
point where `norm_num` will even evaluate the numeral. Inference BN has no reduction: `μ` and
`rsqrt(var+ε)` are constants, the map is affine in `x` with slope `γ·s`, and the modulus is
`rounding + G·S·e` — **linear**. The fold then behaves exactly like the CIFAR-8 one, budget
≈ 10⁻³ of the window, and the whole-net number is statable.
-/

namespace Proofs

open FloatModel

/-- `|1/√(t+ε)| ≤ 1/√ε` for `t ≥ 0` — the `ε`-floor caps the inference inverse-stddev, just as
    `bnIstd_abs_le` caps the batch one. -/
theorem invSqrt_shift_le {t ε : ℝ} (hε : 0 < ε) (ht : 0 ≤ t) :
    |1 / Real.sqrt (t + ε)| ≤ 1 / Real.sqrt ε := by
  have hsε : 0 < Real.sqrt ε := Real.sqrt_pos.mpr hε
  have hpos : 0 < 1 / Real.sqrt (t + ε) :=
    div_pos one_pos (Real.sqrt_pos.mpr (by linarith))
  rw [abs_of_pos hpos]
  exact one_div_le_one_div_of_le hsε (Real.sqrt_le_sqrt (by linarith))

/-- **Inference BN rounding budget.** The rounded normalize chain `fl(fl(γ ⊙ fl(fl(x ⊖ μ) ⊙ sF)) ⊕ β)`
    at the frozen mean `μ` and a float inverse-stddev `sF` is within `bnNormBudget … 0 es` of the
    certified `bnEvalForward`. `bnForward_close_of`'s chain with the mean error at **zero** — the
    frozen `μ` is the same stored constant on both sides, so it contributes no error at all. -/
theorem FloatModel.bnEvalRt_close {m : Nat} (M : FloatModel)
    {ε γ β μ v sF es D S G Bbnd : ℝ} (x : Vec m) (i : Fin m)
    (histd : |sF - 1 / Real.sqrt (v + ε)| ≤ es)
    (hD : |x i - μ| ≤ D) (hSabs : |1 / Real.sqrt (v + ε)| ≤ S)
    (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    |M.bnForwardF γ β μ sF x i - bnEvalForward m ε γ β μ v x i| ≤
      bnNormBudget M.u D S G Bbnd 0 es := by
  have hu := M.u_nonneg
  set s := 1 / Real.sqrt (v + ε) with hs
  have hD0 : 0 ≤ D := (abs_nonneg _).trans hD
  have hS0 : 0 ≤ S := (abs_nonneg _).trans hSabs
  -- stage 1: the centering subtract (μ exact on both sides)
  have hs1 : |M.sub (x i) μ - (x i - μ)| ≤ M.u * (D + 0) + 0 := by
    have h1 : |M.sub (x i) μ - (x i - μ)| ≤ M.u * |x i - μ| := M.err _
    have : M.u * |x i - μ| ≤ M.u * D := mul_le_mul_of_nonneg_left hD hu
    linarith
  -- stage 2: x̂ = centered ⊙ sF
  have hs2 : |M.mul (M.sub (x i) μ) sF - (x i - μ) * s| ≤
      FloatModel.mulErr M.u D S (M.u * (D + 0) + 0) es :=
    M.mul_close hs1 histd hD hSabs
  have hxhat : |(x i - μ) * s| ≤ D * S := by
    rw [abs_mul]; exact mul_le_mul hD hSabs (abs_nonneg _) hD0
  -- stage 3: γ ⊙ x̂
  have hs3 : |M.mul γ (M.mul (M.sub (x i) μ) sF) - γ * ((x i - μ) * s)| ≤
      FloatModel.mulErr M.u G (D * S) 0 (FloatModel.mulErr M.u D S (M.u * (D + 0) + 0) es) :=
    M.mul_close (by simp) hs2 hγ hxhat
  set es3 := FloatModel.mulErr M.u G (D * S) 0
      (FloatModel.mulErr M.u D S (M.u * (D + 0) + 0) es) with hes3
  set s3 := M.mul γ (M.mul (M.sub (x i) μ) sF) with hs3def
  have hgxhat : |γ * ((x i - μ) * s)| ≤ G * (D * S) := by
    rw [abs_mul]; exact mul_le_mul hγ hxhat (abs_nonneg _) ((abs_nonneg _).trans hγ)
  have hs3mag : |s3| ≤ G * (D * S) + es3 := by
    calc |s3| = |(s3 - γ * ((x i - μ) * s)) + γ * ((x i - μ) * s)| := by congr 1; ring
      _ ≤ |s3 - γ * ((x i - μ) * s)| + |γ * ((x i - μ) * s)| := abs_add_le _ _
      _ ≤ es3 + G * (D * S) := add_le_add hs3 hgxhat
      _ = G * (D * S) + es3 := by ring
  have hsumβ : |s3 + β| ≤ G * (D * S) + es3 + Bbnd := by
    calc |s3 + β| ≤ |s3| + |β| := abs_add_le _ _
      _ ≤ (G * (D * S) + es3) + Bbnd := add_le_add hs3mag hβ
      _ = G * (D * S) + es3 + Bbnd := by ring
  -- stage 4: ⊕ β, and assemble
  have hgoal : |M.bnForwardF γ β μ sF x i - bnEvalForward m ε γ β μ v x i| ≤
      M.u * (G * (D * S) + es3 + Bbnd) + es3 := by
    simp only [FloatModel.bnForwardF, bnEvalForward, ← hs, ← hs3def]
    have h4a : |M.add s3 β - (s3 + β)| ≤ M.u * |s3 + β| := M.err _
    have h4b : |(s3 + β) - (γ * ((x i - μ) * s) + β)| ≤ es3 := by
      have he : (s3 + β) - (γ * ((x i - μ) * s) + β) = s3 - γ * ((x i - μ) * s) := by ring
      rw [he]; exact hs3
    calc |M.add s3 β - (γ * ((x i - μ) * s) + β)|
        ≤ |M.add s3 β - (s3 + β)| + |(s3 + β) - (γ * ((x i - μ) * s) + β)| := abs_sub_le _ _ _
      _ ≤ M.u * |s3 + β| + es3 := add_le_add h4a h4b
      _ ≤ M.u * (G * (D * S) + es3 + Bbnd) + es3 := by gcongr
  exact hgoal.trans_eq (by rw [bnNormBudget, ← hes3])

/-- **Inference BN is affine in its input**, so its input-sensitivity is the plain slope
    `|γ·s|` — no mean term, no variance term, no `rsqrt` Lipschitz factor. This one line is the
    whole difference between a whole-net number that exists and one that does not. -/
theorem bnEvalForward_input_close {m : Nat} {ε γ β μ v e S G : ℝ} (xt xa : Vec m) (i : Fin m)
    (hd : ∀ k, |xt k - xa k| ≤ e) (hSabs : |1 / Real.sqrt (v + ε)| ≤ S) (hγ : |γ| ≤ G) :
    |bnEvalForward m ε γ β μ v xt i - bnEvalForward m ε γ β μ v xa i| ≤ G * S * e := by
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hkey : bnEvalForward m ε γ β μ v xt i - bnEvalForward m ε γ β μ v xa i
      = γ * (1 / Real.sqrt (v + ε)) * (xt i - xa i) := by
    simp only [bnEvalForward]; ring
  rw [hkey, abs_mul, abs_mul]
  exact mul_le_mul (mul_le_mul hγ hSabs (abs_nonneg _) hG0) (hd i) (abs_nonneg _)
    (mul_nonneg hG0 ((abs_nonneg _).trans hSabs))

/-- **One channel's inference BN is `FloatClose`** — rounding (`bnEvalRt_close`) plus the affine
    input-shift (`bnEvalForward_input_close`). Modulus `bnNormBudget … 0 es + G·S·e`: LINEAR in
    the inherited error, and linear in the window. -/
theorem floatClose_bnEvalRt {m : Nat} (M : FloatModel)
    {ε γ β μ v sF es A Mb S G Bbnd : ℝ} (hA : 0 ≤ A) (hMb : |μ| ≤ Mb)
    (hes : |sF - 1 / Real.sqrt (v + ε)| ≤ es)
    (hS : |1 / Real.sqrt (v + ε)| ≤ S) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    FloatClose A (G * ((A + Mb) * S) + Bbnd + bnNormBudget M.u (A + Mb) S G Bbnd 0 es)
      (bnEvalForward m ε γ β μ v) (fun x => M.bnForwardF γ β μ sF x)
      (fun e => bnNormBudget M.u (A + Mb) S G Bbnd 0 es + G * S * e) := by
  have hu := M.u_nonneg
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hS0 : 0 ≤ S := (abs_nonneg _).trans hS
  have hMb0 : 0 ≤ Mb := (abs_nonneg _).trans hMb
  have hD : ∀ (x : Vec m), (∀ k, |x k| ≤ A) → ∀ j, |x j - μ| ≤ A + Mb := fun x hx j => by
    calc |x j - μ| ≤ |x j| + |μ| := by
          rw [sub_eq_add_neg, ← abs_neg μ]; exact abs_add_le _ _
      _ ≤ A + Mb := add_le_add (hx j) hMb
  have hes0 : 0 ≤ es := (abs_nonneg _).trans hes
  have hAMb : (0:ℝ) ≤ A + Mb := by linarith
  have hnb0 : 0 ≤ bnNormBudget M.u (A + Mb) S G Bbnd 0 es := by
    have hin : 0 ≤ FloatModel.mulErr M.u (A + Mb) S (M.u * ((A + Mb) + 0) + 0) es :=
      mulErr_nonneg hu hAMb hS0 (by positivity) hes0
    have hout : 0 ≤ FloatModel.mulErr M.u G ((A + Mb) * S) 0
        (FloatModel.mulErr M.u (A + Mb) S (M.u * ((A + Mb) + 0) + 0) es) :=
      mulErr_nonneg hu hG0 (mul_nonneg hAMb hS0) le_rfl hin
    have hDS : (0:ℝ) ≤ G * ((A + Mb) * S) := mul_nonneg hG0 (mul_nonneg hAMb hS0)
    have hBb0 : (0:ℝ) ≤ Bbnd := (abs_nonneg _).trans hβ
    unfold bnNormBudget
    have := mul_nonneg hu (by linarith : (0:ℝ) ≤ G * ((A + Mb) * S)
      + FloatModel.mulErr M.u G ((A + Mb) * S) 0
          (FloatModel.mulErr M.u (A + Mb) S (M.u * ((A + Mb) + 0) + 0) es) + Bbnd)
    linarith
  refine ⟨fun x hx i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · -- magnitudes: real ≤ G·(D·S) + β̄, float ≤ that + the rounding budget
    have hreal : |bnEvalForward m ε γ β μ v x i| ≤ G * ((A + Mb) * S) + Bbnd := by
      simp only [bnEvalForward]
      refine (abs_add_le _ _).trans (add_le_add ?_ hβ)
      rw [abs_mul, abs_mul]
      exact mul_le_mul hγ (mul_le_mul (hD x hx i) hS (abs_nonneg _) (by linarith))
        (mul_nonneg (abs_nonneg _) (abs_nonneg _)) hG0
    refine ⟨hreal.trans (le_add_of_nonneg_right hnb0), ?_⟩
    have hround := M.bnEvalRt_close (ε := ε) (β := β) (Bbnd := Bbnd) x i hes (hD x hx i) hS hγ hβ
    have htri : |M.bnForwardF γ β μ sF x i|
        ≤ |M.bnForwardF γ β μ sF x i - bnEvalForward m ε γ β μ v x i|
          + |bnEvalForward m ε γ β μ v x i| := by
      simpa using abs_sub_le (M.bnForwardF γ β μ sF x i) (bnEvalForward m ε γ β μ v x i) 0
    linarith
  · -- error: rounding at the float input, then the affine input-shift
    have hround := M.bnEvalRt_close (ε := ε) (β := β) (Bbnd := Bbnd) vt i hes (hD vt hvt i) hS hγ hβ
    have hshift := bnEvalForward_input_close (ε := ε) (β := β) (μ := μ) (v := v)
      (S := S) (G := G) vt va i hd hS hγ
    calc |M.bnForwardF γ β μ sF vt i - bnEvalForward m ε γ β μ v va i|
        ≤ |M.bnForwardF γ β μ sF vt i - bnEvalForward m ε γ β μ v vt i|
          + |bnEvalForward m ε γ β μ v vt i - bnEvalForward m ε γ β μ v va i| := abs_sub_le _ _ _
      _ ≤ bnNormBudget M.u (A + Mb) S G Bbnd 0 es + G * S * e := add_le_add hround hshift

/-- The float per-channel inference BN (Mat-split layout): channel `c` runs the rounded
    normalize chain at its own frozen `μ c` and float inverse-stddev `sF c`. -/
noncomputable def bnPerChannelEvalFlatFV {oc m : Nat} (M : FloatModel)
    (γ β μ sF : Vec oc) : Vec (oc * m) → Vec (oc * m) :=
  perRowIdxFlat oc m (fun c => fun x => M.bnForwardF (γ c) (β c) (μ c) (sF c) x)

/-- The float per-channel inference BN in the network Tensor3 layout. -/
noncomputable def bnPerChannelEvalTensor3FV {oc h w : Nat} (M : FloatModel)
    (γ β μ sF : Vec oc) : Vec (oc * h * w) → Vec (oc * h * w) :=
  reassocBack oc h w ∘ bnPerChannelEvalFlatFV (m := h * w) M γ β μ sF ∘ reassocFwd oc h w

/-- The inference BN leaf's output window: `G·(A+μ̄)·S + β̄` plus the normalize chain's rounding. -/
noncomputable def bnEvalLeafMag (u S G Bbnd Mb es : ℝ) (A : ℝ) : ℝ :=
  G * ((A + Mb) * S) + Bbnd + bnNormBudget u (A + Mb) S G Bbnd 0 es

/-- The inference BN leaf's modulus: the rounding, plus the affine slope `G·S` on the inherited
    error. Linear in both arguments — the training-mode leaf's is quadratic in the window. -/
noncomputable def bnEvalLeafMod (u S G Bbnd Mb es : ℝ) (A e : ℝ) : ℝ :=
  bnNormBudget u (A + Mb) S G Bbnd 0 es + G * S * e

/-- ⭐ **Per-channel inference BatchNorm float-bridges TO the kernel the render emits.** The op
    is `bnPerChannelEvalTensor3` — what `SHlo.bnPerChannelEvalF` denotes
    (`bnPerChannelEvalF_faithful`), hence the ℝ semantics of every BN line in the committed
    `@resnet34_fwd_eval` / `@mobilenetv2_fwd_eval` / `@efficientnet_fwd_eval` renders. The float
    side is the same six rounded ops the emitter writes, with the device `rsqrt` modelled by a
    supplied `sF` of accuracy `es`. -/
noncomputable def floatBridgesTo_bnPerChannelEvalTensor3 {oc h w : Nat} (M : FloatModel)
    {ε : ℝ} (γ β μ v sF : Vec oc) {G Bbnd Mb S es : ℝ}
    (hoc : 0 < oc) (hhw : 0 < h * w) (hε : 0 < ε) (hv : ∀ c, 0 ≤ v c)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd) (hμ : ∀ c, |μ c| ≤ Mb)
    (hes : ∀ c, |sF c - 1 / Real.sqrt (v c + ε)| ≤ es)
    (hS : 1 / Real.sqrt ε ≤ S) :
    FloatBridgesTo (bnPerChannelEvalTensor3 oc h w ε γ β μ v)
      (bnPerChannelEvalTensor3FV M γ β μ sF) :=
  ⟨bnEvalLeafMag M.u S G Bbnd Mb es, bnEvalLeafMod M.u S G Bbnd Mb es,
   ((floatBridgesTo_gather (reassocEquiv oc h w)).comp
      (⟨bnEvalLeafMag M.u S G Bbnd Mb es, bnEvalLeafMod M.u S G Bbnd Mb es,
        fun _A hA =>
          have hg := fun c : Fin oc =>
            floatClose_bnEvalRt (m := h * w) M (μ := μ c) (v := v c) (sF := sF c)
              hA (hμ c) (hes c) ((invSqrt_shift_le hε (hv c)).trans hS) (hγ c) (hβ c)
          have hpr := FloatClose.perRowIdx (d := h * w) oc hg
          ⟨hpr.cod_nonneg hA (Nat.mul_pos hoc hhw), hpr⟩⟩ :
        FloatBridgesTo (bnPerChannelEvalFlat oc (h * w) ε γ β μ v)
          (bnPerChannelEvalFlatFV (m := h * w) M γ β μ sF))).comp
     (floatBridgesTo_gather (reassocEquiv oc h w).symm) |>.close⟩

end Proofs
