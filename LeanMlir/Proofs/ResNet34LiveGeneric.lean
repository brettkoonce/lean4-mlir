import LeanMlir.Proofs.ResNet34LiveRealistic

/-! # Limit-D strengthening: a whole-net ResNet-34 VJP with *arbitrary* downsample convs

The existing live ResNet-34 witnesses (`ResNet34LiveFull` at full depth, `ResNet34LiveRealistic`
at 224×224) discharge every smoothness/no-tie hypothesis of the parametric apex
`resnet34_has_vjp_at` — but their convolutions are **structural**: the residual bodies use the
zero kernel (`Zk2`, with BN `γ=0`, so the body collapses to the constant `1`) and the
projections use the identity kernel (`WsP2`). The audit's "limit D" residual was that the
convolutions do no genuine mixing.

This file removes that for the downsample projection: the key observation is that the
projection's positivity (hence ReLU-off-the-kink, hence smoothness) is discharged by
`bnForward_lb` — a bound on the *batch-normalized* activation that is **independent of the
convolution weights**. So the entire VJP+differentiability discharge for the strided
downsample goes through for an *arbitrary* kernel `W`, provided only the documented
β-positivity margin `√n < β`.

Result: `liveFwdW_has_vjp_at` is a whole-network ResNet-34 VJP at realistic 224×224 spatial
dims that is **universally quantified over the three downsample projection kernels**
`W₂ W₃ W₄ : Kernel4 2 2 1 1` — the convolutions are genuine free parameters, not identity.
`liveFwdW_mixing_*` instantiates it at a concrete *channel-mixing* (non-diagonal) kernel to
show the generalization is non-vacuous.

What stays specialized (the honest residual): 2 channels, the stem conv is still identity
(its maxpool no-tie discharge needs the injective ramp the identity stem preserves), the
residual *bodies* are still `γ=0` (the shortcut-collapse trick that keeps the 1+ shift
positive), and the β margin must dominate `√n`. But the downsample projection — the conv
that actually carries the strided feature transform — is now arbitrary.
-/

namespace Proofs
open Proofs ResNet34LivePC ResNet34LiveRealistic

-- ════════════════════════════════════════════════════════════════
-- § The strided downsample with an arbitrary projection kernel `W`
-- ════════════════════════════════════════════════════════════════

/-- `liveDownβ` with the identity projection kernel `WsP2` replaced by an arbitrary `W`.
    The body path is still the `γ=0` collapse (constant `1`); the projection is
    `relu(bn_{γ=1,β}(conv_W(decimate ·)) + 1)`, where `W` genuinely transforms the
    decimated input before normalization. -/
noncomputable def liveDownW (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) :
    Vec (2 * (2 * h) * (2 * w)) → Vec (2 * h * w) :=
  relu (2 * h * w) ∘ residualProj
    (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2))

/-- **Weight-independent projection positivity.** For *any* kernel `W`, the normalized
    projection `bn_{γ=1,β}(conv_W(·))` is `> 0` once `√n < β` — `bnForward_lb` bounds the
    normalized value by `±1·√…`, dominated by `β`, with no reference to `W`. -/
theorem liveDownW_proj_pos (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w)))
    (k : Fin (2 * h * w)) :
    0 < (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2) a k := by
  simp only [Function.comp_apply]
  have hlb := bnForward_lb (n := 2 * h * w) 1 1 βp (by norm_num) (flatConvStride2 W Zb2 a) k
  rw [abs_one, one_mul] at hlb
  linarith

/-- The projection + the constant-`1` body is strictly positive (for any `W`). -/
theorem liveDownW_sum_pos (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w)))
    (k : Fin (2 * h * w)) :
    0 < ((bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2) a k)
      + ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
          (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k := by
  have hp := liveDownW_proj_pos h w βp W hn a k
  have hb : ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k = 1 := by
    rw [liveDownβ_body_const h w hhw a]
  rw [hb]; linarith

/-- **The strided downsample has a VJP at every point — for an arbitrary kernel `W`.** -/
noncomputable def liveDownW_vjp (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    HasVJPAt (liveDownW h w βp W) a :=
  rblkPStrided_has_vjp_at Zk2 Zb2 Zk2 Zb2 W Zb2 1 0 1 1 0 1 1 1 βp
    (by norm_num) (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConvStride2_eq_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
      change (1 : ℝ) ≠ 0; norm_num)
    (fun k => ne_of_gt (liveDownW_sum_pos h w βp W hhw hn a k))

/-- **The strided downsample is differentiable at every point — for an arbitrary kernel `W`.** -/
theorem liveDownW_diff (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    DifferentiableAt ℝ (liveDownW h w βp W) a := by
  have hsm₁ : ∀ k, bnForward (2 * h * w) 1 0 1 (flatConvStride2 Zk2 Zb2 a) k ≠ 0 := fun k => by
    rw [flatConvStride2_eq_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
    change (1 : ℝ) ≠ 0; norm_num
  have hproj_diff : DifferentiableAt ℝ (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2) a :=
    convBnStrided_differentiable (h := h) (w := w) W Zb2 1 1 βp (by norm_num) a
  have hF_diff : DifferentiableAt ℝ
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a :=
    resblock_bodyStrided_differentiableAt (h := h) (w := w) Zk2 Zb2 Zk2 Zb2 1 0 1 1 0 1
      (by norm_num) (by norm_num) a hsm₁
  have hsm_res : ∀ k, residualProj
      (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2)
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k ≠ 0 := fun k =>
    ne_of_gt (liveDownW_sum_pos h w βp W hhw hn a k)
  show DifferentiableAt ℝ (relu (2 * h * w) ∘ residualProj
    (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2))) a
  exact (relu_differentiableAt_of_smooth (2 * h * w) _ hsm_res).comp a
    (DifferentiableAt.add hproj_diff hF_diff)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net ResNet-34 VJP, ∀ downsample kernels
-- ════════════════════════════════════════════════════════════════

/-- **224×224 live ResNet-34 with arbitrary downsample projection convs.** Identity stem +
    maxpool + three `liveDownW` downsamples carrying free kernels `W₂ W₃ W₄` + GAP + identity
    head (the 224² spatial pyramid of `liveFwd224`, but with genuine — not identity —
    strided convolutions). -/
noncomputable def liveFwdW (W₂ W₃ W₄ : Kernel4 2 2 1 1) :
    Vec (2 * (2 * 112) * (2 * 112)) → Vec 2 :=
  dense Wd2 bd2 ∘ globalAvgPoolFlat 2 7 7 ∘
    liveDownW 7 7 64 W₄ ∘ liveDownW 14 14 64 W₃ ∘ liveDownW 28 28 64 W₂ ∘
    maxPoolFlat 2 56 56 ∘ stem224

/-- **Whole-network VJP for the 224×224 live ResNet-34, for ANY downsample kernels.** Every
    smoothness/no-tie hypothesis of the parametric apex `resnet34_has_vjp_at` is discharged —
    the three downsample VJP/diff witnesses hold for *arbitrary* `W₂ W₃ W₄` because the
    β-positivity bound is weight-independent. The conv weights of the strided feature
    transforms are genuine free parameters, not fixed to identity. -/
noncomputable def liveFwdW_has_vjp_at (W₂ W₃ W₄ : Kernel4 2 2 1 1) :
    HasVJPAt (liveFwdW W₂ W₃ W₄) X224 :=
  resnet34_has_vjp_at stem224 (maxPoolFlat 2 56 56)
    ([] : List (Vec (2 * 56 * 56) → Vec (2 * 56 * 56))) (liveDownW 28 28 64 W₂)
    ([] : List (Vec (2 * 28 * 28) → Vec (2 * 28 * 28))) (liveDownW 14 14 64 W₃)
    ([] : List (Vec (2 * 14 * 14) → Vec (2 * 14 * 14))) (liveDownW 7 7 64 W₄)
    ([] : List (Vec (2 * 7 * 7) → Vec (2 * 7 * 7)))
    (globalAvgPoolFlat 2 7 7) (dense Wd2 bd2) X224
    ⟨stem224_vjp, stem224_diff⟩
    ⟨hmp_vjp224, hmp_diff224⟩
    PUnit.unit
    ⟨liveDownW_vjp 28 28 64 W₂ (by norm_num) (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num)) _,
     liveDownW_diff 28 28 64 W₂ (by norm_num) (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num)) _⟩
    PUnit.unit
    ⟨liveDownW_vjp 14 14 64 W₃ (by norm_num) (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num)) _,
     liveDownW_diff 14 14 64 W₃ (by norm_num) (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num)) _⟩
    PUnit.unit
    ⟨liveDownW_vjp 7 7 64 W₄ (by norm_num) (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)) _,
     liveDownW_diff 7 7 64 W₄ (by norm_num) (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)) _⟩
    PUnit.unit
    ⟨(globalAvgPoolFlat_has_vjp 2 7 7).toHasVJPAt _, (globalAvgPoolFlat_differentiable 2 7 7) _⟩
    ⟨(dense_has_vjp Wd2 bd2).toHasVJPAt _, (dense_differentiable Wd2 bd2) _⟩

/-- **Public correctness, ∀ downsample kernels** — the backward equals the `pdiv`-Jacobian. -/
theorem liveFwdW_has_vjp_correct (W₂ W₃ W₄ : Kernel4 2 2 1 1)
    (dy : Vec 2) (i : Fin (2 * (2 * 112) * (2 * 112))) :
    (liveFwdW_has_vjp_at W₂ W₃ W₄).backward dy i =
      ∑ j : Fin 2, pdiv (liveFwdW W₂ W₃ W₄) X224 i j * dy j :=
  (liveFwdW_has_vjp_at W₂ W₃ W₄).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Non-vacuity: a concrete channel-MIXING (non-diagonal) kernel
-- ════════════════════════════════════════════════════════════════

/-- A genuinely non-diagonal 1×1 kernel: every output channel reads every input channel
    (the all-ones channel-mixing conv). Not the identity `WsP2`. -/
noncomputable def Wmix : Kernel4 2 2 1 1 := fun _ _ _ _ => 1

/-- **The whole-net VJP holds at a real channel-mixing downsample conv.** Instantiating the
    ∀-kernel result at `Wmix` (all-ones, fully off-diagonal) gives a 224×224 ResNet-34
    whole-net backward whose three strided downsample convolutions genuinely mix channels —
    a concrete witness that the generalization past identity convs is non-vacuous. -/
noncomputable def liveFwdW_mixing_has_vjp_at : HasVJPAt (liveFwdW Wmix Wmix Wmix) X224 :=
  liveFwdW_has_vjp_at Wmix Wmix Wmix

theorem liveFwdW_mixing_has_vjp_correct (dy : Vec 2) (i : Fin (2 * (2 * 112) * (2 * 112))) :
    liveFwdW_mixing_has_vjp_at.backward dy i =
      ∑ j : Fin 2, pdiv (liveFwdW Wmix Wmix Wmix) X224 i j * dy j :=
  (liveFwdW_has_vjp_at Wmix Wmix Wmix).correct dy i

#print axioms liveFwdW_has_vjp_correct
#print axioms liveFwdW_mixing_has_vjp_correct

end Proofs
