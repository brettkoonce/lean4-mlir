import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.JacobianSeal
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# `Mnv2Live` nonzero-Jacobian seal — Item B2 discharged at a deep kinked witness

`planning/whole_network_backward.md` Item B2: take the live MobileNetV2 witness
(`Mnv2Live`, the only non-degenerate kinked witness) from **level 2**
(`mnv2Live_forward_nonconstant : forward X ≠ forward 0`) up to **level 3**, the
*nonzero-Jacobian seal*: the proven whole-net backward is genuinely non-trivial
at a witness point — `fderiv ℝ forward · ≠ 0`, hence (via the `JacobianSeal`
bridge) `backward (basisVec j₀) i₀ ≠ 0`.

## The key structural fact this exploits

`bn13_window` holds for **every** input `z` (γ=1, β=3, ε=1 puts every BN output
strictly inside `(0,6)`), so the five ReLU6 sites are the identity on the entire
BN path. The live net is therefore *globally smooth* — the kinks never bind — and
`forward` equals a closed form built only from conv / BN / GAP / dense
(`forward_eq` below). This both (a) makes `forward` differentiable everywhere and
(b) voids the worry flagged in the planning doc that "the MVT/derivative point may
not be a smooth point where the VJP holds": here *every* point is.

## Why the seal is sealed at the input `0` (exactly, constructively)

Write `forward(z)₀ = 3 + (1/4)·P(z)·L(z)` where `L(z) = chSum(conv z)` is **linear**
(`chSum_convX_smul`, with `L(X) = -3`) and `P(z)` is a product of four **positive**
BN inverse-stds (`bnIstd_pos`). Along the ray `t ↦ t·X`,
`forward(t·X)₀ = 3 - (3/4)·t·P(t·X)`, and the product-rule cross-term carries a
factor `t`, so it **vanishes at `t = 0`**: the directional derivative there is
exactly `-(3/4)·P(0) < 0`. No variance/`istd` derivative is needed — that is the
hairy piece the planning doc flags, and `t = 0` is precisely the point that avoids
it. So `fderiv ℝ forward 0 ≠ 0` is exact and constructive (no MVT, no nonconstructive
witness point). The net has genuinely nonzero weights (asymmetric stem `1,2`,
identity block-2 convs); the fact that the *output* `forward 0` happens to be the
constant `3` is irrelevant — what the seal certifies is that the *derivative* (the
backward map) at this input is not the zero map.
-/

namespace Proofs
namespace Mnv2Live

open scoped BigOperators
open Finset Filter Topology

-- ════════════════════════════════════════════════════════════════
-- § The globally-smooth closed form (ReLU6 never binds)
-- ════════════════════════════════════════════════════════════════

/-- **The live forward, for every input, equals a ReLU6-free closed form.**
    Generalization of `forward_X_eq` (which is the `v = X` case): the window
    lemma `bn13_window` discharges all five ReLU6 sites to the identity for
    *any* input, block-1's zeroed body is the constant shift `+3`, and block-2's
    identity convs leave four genuine BN layers. -/
theorem forward_eq (v : Vec (1 * 2 * 2)) :
    mobilenetv2Forward Ws bs 1 1 3 We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3
      We₂ be₂ 1 1 3 Wd₂ bd₂ 1 1 3 Wp₂ bp₂ 1 1 3 Wh bh v
      = dense Wh bh (globalAvgPoolFlat 2 2 2
          (bnForward (2 * 2 * 2) 1 1 3 (bnForward (2 * 2 * 2) 1 1 3 (bnForward (2 * 2 * 2) 1 1 3
            (fun k => 3 + bnForward (2 * 2 * 2) 1 1 3 (flatConv (h := 2) (w := 2) Ws bs v) k))))) := by
  have hb1 :
      residual (invresBody (h := 2) (w := 2) We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3)
        (bnForward (2 * 2 * 2) 1 1 3 (flatConv (h := 2) (w := 2) Ws bs v))
        = (fun k => 3 + bnForward (2 * 2 * 2) 1 1 3 (flatConv (h := 2) (w := 2) Ws bs v) k) := by
    funext k; simp only [residual, biPath, invresBody₁_const]
  simp only [mobilenetv2Forward, Function.comp_apply, relu6_bn8,
    flatConv_id2 We₂ be₂ (fun _ _ => rfl) (fun _ => rfl),
    flatConv_id2 Wp₂ bp₂ (fun _ _ => rfl) (fun _ => rfl),
    depthwiseFlat_id1 Wd₂ bd₂ (fun _ => rfl) (fun _ => rfl),
    hb1]

/-- The whole live net, as a single map (abbrev so the apex `HasVJPAt` and the
    closed form unify definitionally). -/
noncomputable abbrev fwd : Vec (1 * 2 * 2) → Vec 2 :=
  mobilenetv2Forward Ws bs 1 1 3 We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3
    We₂ be₂ 1 1 3 Wd₂ bd₂ 1 1 3 Wp₂ bp₂ 1 1 3 Wh bh

/-- The ReLU6-free closed form (`forward_eq`'s right-hand side). -/
noncomputable def fwdCF : Vec (1 * 2 * 2) → Vec 2 :=
  fun v => dense Wh bh (globalAvgPoolFlat 2 2 2
    (bnForward (2 * 2 * 2) 1 1 3 (bnForward (2 * 2 * 2) 1 1 3 (bnForward (2 * 2 * 2) 1 1 3
      (fun k => 3 + bnForward (2 * 2 * 2) 1 1 3 (flatConv (h := 2) (w := 2) Ws bs v) k)))))

theorem fwd_eq_fwdCF : fwd = fwdCF := by funext v; exact forward_eq v

/-- **The live forward is differentiable everywhere.** Composition of the
    differentiable layers (conv, four BN under `ε = 1 > 0`, the affine `+3`
    shift, GAP, dense); no ReLU6 obstruction because it never binds. -/
theorem fwdCF_differentiable : Differentiable ℝ fwdCF := by
  have hbn : Differentiable ℝ (bnForward (2 * 2 * 2) 1 1 3) :=
    bnForward_differentiable (2 * 2 * 2) 1 1 3 one_pos
  have hconv : Differentiable ℝ
      (flatConv (h := 2) (w := 2) Ws bs : Vec (1 * 2 * 2) → Vec (2 * 2 * 2)) :=
    flatConv_differentiable Ws bs
  have hgap : Differentiable ℝ (globalAvgPoolFlat 2 2 2 : Vec (2 * 2 * 2) → Vec 2) :=
    globalAvgPoolFlat_differentiable 2 2 2
  have hdense : Differentiable ℝ (dense Wh bh) := dense_differentiable Wh bh
  have hshift : Differentiable ℝ (fun u : Vec (2 * 2 * 2) => (fun k => (3 : ℝ) + u k)) := by
    fun_prop
  unfold fwdCF
  exact hdense.comp (hgap.comp (hbn.comp (hbn.comp (hbn.comp (hshift.comp (hbn.comp hconv))))))

-- ════════════════════════════════════════════════════════════════
-- § `chSum ∘ conv` is linear in the input (the carrier functional `L`)
-- ════════════════════════════════════════════════════════════════

theorem chSum_smul (t : ℝ) (z : Vec (2 * 2 * 2)) : chSum (t • z) = t * chSum z := by
  unfold chSum
  have hmean : bnMean (2 * 2 * 2) (t • z) = t * bnMean (2 * 2 * 2) z := by
    unfold bnMean
    rw [show (∑ i, (t • z) i) = t * ∑ i, z i by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i _ => by
        simp only [Pi.smul_apply, smul_eq_mul])]
    ring
  rw [hmean, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro p _
  simp only [Pi.smul_apply, smul_eq_mul]
  ring

theorem flatten_smul3 (t : ℝ) (T : Tensor3 2 2 2) :
    Tensor3.flatten (t • T) = t • Tensor3.flatten T := by
  funext k; simp only [Tensor3.flatten, Pi.smul_apply, smul_eq_mul]

theorem unflatten_smul3 (t : ℝ) (v : Vec (1 * 2 * 2)) :
    (Tensor3.unflatten (t • v) : Tensor3 1 2 2) = t • Tensor3.unflatten v := by
  funext ci hi wi; simp only [Tensor3.unflatten, Pi.smul_apply, smul_eq_mul]

theorem conv2d_Ws_smul (t : ℝ) (S : Tensor3 1 2 2) :
    conv2d Ws bs (t • S) = t • conv2d Ws bs S := by
  funext o hi wi
  simp only [Pi.smul_apply, smul_eq_mul, conv2d_1x1', bs, zero_add, Fin.sum_univ_one]
  ring

theorem flatConv_Ws_smul (t : ℝ) :
    flatConv (h := 2) (w := 2) Ws bs (t • X) = t • flatConv (h := 2) (w := 2) Ws bs X := by
  simp only [flatConv]
  rw [unflatten_smul3, conv2d_Ws_smul, flatten_smul3]

/-- **The stem carrier is linear**: `chSum(conv(t·X)) = -3·t` (so `L(X) = -3`). -/
theorem chSum_convX_smul (t : ℝ) :
    chSum (flatConv (h := 2) (w := 2) Ws bs (t • X)) = -3 * t := by
  rw [flatConv_Ws_smul, chSum_smul, chSum_convX]; ring

-- ════════════════════════════════════════════════════════════════
-- § The four BN inputs along the ray `t ↦ t·X`, and the istd product `P`
-- ════════════════════════════════════════════════════════════════

noncomputable def A0 (t : ℝ) : Vec (2 * 2 * 2) := flatConv (h := 2) (w := 2) Ws bs (t • X)
noncomputable def A1 (t : ℝ) : Vec (2 * 2 * 2) := bnForward (2 * 2 * 2) 1 1 3 (A0 t)
noncomputable def A2 (t : ℝ) : Vec (2 * 2 * 2) := fun k => 3 + A1 t k
noncomputable def A3 (t : ℝ) : Vec (2 * 2 * 2) := bnForward (2 * 2 * 2) 1 1 3 (A2 t)
noncomputable def A4 (t : ℝ) : Vec (2 * 2 * 2) := bnForward (2 * 2 * 2) 1 1 3 (A3 t)

/-- The product of the four BN inverse-stds along the ray (the nonnegative,
    nonlinear part of `forward(t·X)₀`). Always strictly positive. -/
noncomputable def Pp (t : ℝ) : ℝ :=
  bnIstd (2 * 2 * 2) (A4 t) 1 *
    (bnIstd (2 * 2 * 2) (A3 t) 1 *
      (bnIstd (2 * 2 * 2) (A2 t) 1 * bnIstd (2 * 2 * 2) (A0 t) 1))

/-- `forward(t·X)₀ = 3 + Qq t · t`; `Qq 0 = -(3/4)·P(0) < 0` is the seal value. -/
noncomputable def Qq (t : ℝ) : ℝ := -(3 / 4) * Pp t

theorem Pp_pos (t : ℝ) : 0 < Pp t := by
  unfold Pp
  have h := bnIstd_pos (2 * 2 * 2) 1 one_pos
  exact mul_pos (h _) (mul_pos (h _) (mul_pos (h _) (h _)))

theorem Qq_zero_ne : Qq 0 ≠ 0 := by
  have h := Pp_pos 0
  unfold Qq
  exact ne_of_lt (by nlinarith [h])

-- ── continuity of the istd product ──

theorem bnIstd_continuous :
    Continuous (fun v : Vec (2 * 2 * 2) => bnIstd (2 * 2 * 2) v 1) :=
  (continuous_apply (0 : Fin (2 * 2 * 2))).comp
    (bnIstdBroadcast_diff (2 * 2 * 2) 1 one_pos).continuous

theorem A0_continuous : Continuous A0 := by
  unfold A0
  exact (flatConv_differentiable (h := 2) (w := 2) Ws bs).continuous.comp
    (continuous_id.smul continuous_const)

theorem A1_continuous : Continuous A1 := by
  unfold A1
  exact (bnForward_differentiable (2 * 2 * 2) 1 1 3 one_pos).continuous.comp A0_continuous

theorem A2_continuous : Continuous A2 := by
  unfold A2
  exact continuous_pi (fun k => continuous_const.add ((continuous_apply k).comp A1_continuous))

theorem A3_continuous : Continuous A3 := by
  unfold A3
  exact (bnForward_differentiable (2 * 2 * 2) 1 1 3 one_pos).continuous.comp A2_continuous

theorem A4_continuous : Continuous A4 := by
  unfold A4
  exact (bnForward_differentiable (2 * 2 * 2) 1 1 3 one_pos).continuous.comp A3_continuous

theorem Pp_continuous : Continuous Pp := by
  unfold Pp
  exact (bnIstd_continuous.comp A4_continuous).mul
    ((bnIstd_continuous.comp A3_continuous).mul
      ((bnIstd_continuous.comp A2_continuous).mul (bnIstd_continuous.comp A0_continuous)))

theorem Qq_continuous : Continuous Qq := continuous_const.mul Pp_continuous

-- ════════════════════════════════════════════════════════════════
-- § The directional derivative at the input `0` along `X`
-- ════════════════════════════════════════════════════════════════

/-- The closed-form value along the ray: `forward(t·X)₀ = 3 + Qq t · t`. -/
theorem g_eq (t : ℝ) : fwdCF (t • X) 0 = 3 + Qq t * t := by
  unfold fwdCF
  rw [dense_Wh_apply, gap0_eq, bnForward_mean (2 * 2 * 2) (by norm_num) 1 1 3 _]
  have hch :
      chSum (bnForward (2 * 2 * 2) 1 1 3 (bnForward (2 * 2 * 2) 1 1 3 (bnForward (2 * 2 * 2) 1 1 3
        (fun k => 3 + bnForward (2 * 2 * 2) 1 1 3 (flatConv (h := 2) (w := 2) Ws bs (t • X)) k))))
        = Pp t * (-3 * t) := by
    rw [chSum_bn, chSum_bn, chSum_bn, chSum_const_add, chSum_bn, chSum_convX_smul]
    unfold Pp A4 A3 A2 A1 A0
    ring
  rw [hch]; unfold Qq; ring

/-- **`forward(·)₀` along `X` has nonzero derivative at the input `0`.** The
    product-rule cross-term carries a factor `t`, so it vanishes at `0`, leaving
    `g'(0) = Qq 0 = -(3/4)·P(0) ≠ 0`. -/
theorem g_hasDerivAt : HasDerivAt (fun (t : ℝ) => fwdCF (t • X) 0) (Qq 0) 0 := by
  have hg_eq : (fun (t : ℝ) => fwdCF (t • X) 0) = fun t => 3 + Qq t * t := funext g_eq
  rw [hg_eq]
  have hmul : HasDerivAt (fun t : ℝ => Qq t * t) (Qq 0) 0 := by
    rw [hasDerivAt_iff_tendsto_slope]
    have hslope : slope (fun t : ℝ => Qq t * t) 0 =ᶠ[𝓝[≠] (0 : ℝ)] Qq := by
      filter_upwards [self_mem_nhdsWithin] with y hy
      have hy0 : y ≠ 0 := by simpa using hy
      simp only [slope_def_field, sub_zero, mul_zero]
      rw [mul_div_assoc, div_self hy0, mul_one]
    exact Filter.Tendsto.congr' hslope.symm
      ((Qq_continuous.continuousAt).tendsto.mono_left nhdsWithin_le_nhds)
  exact hmul.const_add 3

-- ════════════════════════════════════════════════════════════════
-- § The seal: `fderiv ≠ 0` ⇒ non-trivial backward at the witness `0`
-- ════════════════════════════════════════════════════════════════

theorem fwdCF_fderiv_ne : fderiv ℝ fwdCF 0 ≠ 0 := by
  intro hzero
  -- if the Jacobian at 0 were zero, the directional derivative would vanish
  have hfd : HasFDerivAt fwdCF (0 : Vec (1 * 2 * 2) →L[ℝ] Vec 2) (0 : Vec (1 * 2 * 2)) := by
    rw [← hzero]; exact (fwdCF_differentiable 0).hasFDerivAt
  have hsmul : HasDerivAt (fun t : ℝ => t • X) X 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).smul_const X
  have hcomp : HasDerivAt (fun t : ℝ => fwdCF (t • X)) (0 : Vec 2) 0 := by
    have := HasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hfd hsmul (by simp)
    exact this
  have hcomp0 : HasDerivAt (fun t : ℝ => fwdCF (t • X) 0) (0 : ℝ) 0 := by
    have := (hasDerivAt_pi.mp hcomp) (0 : Fin 2)
    simpa using this
  exact Qq_zero_ne (g_hasDerivAt.unique hcomp0)

/-- **`fderiv ℝ fwd 0 ≠ 0`** — the live MobileNetV2's whole-net Jacobian is
    genuinely non-trivial at the witness input `0` (level-3 seal). -/
theorem mnv2Live_jacobian_nonzero : fderiv ℝ fwd 0 ≠ 0 := by
  rw [fwd_eq_fwdCF]; exact fwdCF_fderiv_ne

-- ── the pointwise VJP at an arbitrary input (window holds everywhere) ──

private theorem win' (z : Vec (2 * 2 * 2)) (k : Fin (2 * 2 * 2)) :
    bnForward (2 * 2 * 2) 1 1 3 z k ≠ 0 ∧ bnForward (2 * 2 * 2) 1 1 3 z k ≠ 6 := by
  obtain ⟨h0, h6⟩ := bn13_window (2 * 2 * 2) (by norm_num) (by norm_num) 1 one_pos z k
  exact ⟨h0.ne', h6.ne⟩

/-- The whole-net VJP holds at *every* input (the window discharges all five
    ReLU6 sites regardless of the activation), so we may seal at `0`. -/
noncomputable def mnv2Live_has_vjp_at_input (v : Vec (1 * 2 * 2)) : HasVJPAt fwd v :=
  mobilenetv2_has_vjp_at Ws bs 1 1 3 one_pos
    We₁ be₁ 1 1 3 one_pos Wd₁ bd₁ 1 1 3 one_pos Wp₁ bp₁ 1 1 3 one_pos
    We₂ be₂ 1 1 3 one_pos Wd₂ bd₂ 1 1 3 one_pos Wp₂ bp₂ 1 1 3 one_pos Wh bh v
    (fun k => win' _ k) (fun k => win' _ k) (fun k => win' _ k)
    (fun k => win' _ k) (fun k => win' _ k)

/-- **The level-3 seal for `Mnv2Live`** (Item B2): the proven whole-network
    backward of the nonzero-weight live MobileNetV2 is **not the zero map** at the
    witness input `0` — some basis-cotangent probe returns a nonzero row. This is
    strictly stronger than `mnv2Live_forward_nonconstant` (level 2): a non-constant
    forward could still have a zero Jacobian at the witness; this rules that out. -/
theorem mnv2Live_backward_nontrivial :
    ∃ (j₀ : Fin 2) (i₀ : Fin (1 * 2 * 2)),
      (mnv2Live_has_vjp_at_input 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (mnv2Live_has_vjp_at_input 0).backward_nontrivial_of_fderiv_ne mnv2Live_jacobian_nonzero

end Mnv2Live
end Proofs
