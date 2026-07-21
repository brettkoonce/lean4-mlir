import LeanMlir.Proofs.Foundation.ResNet34LiveRealistic
import LeanMlir.Proofs.Training.ResNet34LiveSeal
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Realistic-dimension live ResNet-34 nonzero-Jacobian seal (level 3, 224×224)

`ResNet34LiveRealistic` lifts the live ResNet-34 *witness* to ImageNet 224×224 (level 2:
VJP + non-vacuity). This file seals it at **level 3**: `fderiv ℝ liveFwd224 Y ≠ 0`, hence
(via the `JacobianSeal` bridge) the proven whole-net backward is **not the zero map** at `Y`.

## Why a new perturbation (vs. `ResNet34LiveSeal`)

The toy seal collapses to 1×1 before GAP, so the output channel difference *is* the top-left
carrier `cd`, perturbed by a single input coordinate; tracking it needs the maxpool to
*eventually* select the perturbed top-left (a continuity/`Eventually` argument). Here the net
GAPs over **7×7**, so the top-left alone is not the output. The clean fix is a **uniform
channel-0 perturbation** `V224u` (the whole channel, not one coordinate): along the ray
`channel 0 = channel 1 + δ` *at every position*, so

* GAP of a uniform difference is that difference (`UDiff_gap`), and
* `maxpool(ch0) = maxpool(ch1) + δ` holds for **all `t`** — `max(a+δ,b+δ) = max(a,b)+δ` —
  with **no** "eventually selects top-left" argument.

So the whole eventual-selection / topology block of the toy seal is replaced by a `UDiff δ u`
invariant ("the two channels differ by the uniform constant `δ`") threaded through the layers
exactly like `Dom2`: each BN multiplies `δ` by its positive `istd` (the exact identity
`bnForward_chan_diff`), decimate/maxpool/`+1` leave `δ` unchanged. The output difference along
the ray is therefore `t · Rr(t)` with `Rr` a product of four positive `istd`s — `g'(0) = Rr 0 ≠ 0`,
no BN-variance derivative (the same mechanism as both prior seals). The maxpool no-tie for the
VJP at `Y` is unchanged: `Y` is the per-channel positionally-injective decreasing base.
-/

namespace Proofs
namespace R34RealSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs ResNet34Live2 ResNet34LivePC ResNet34LiveSeal ResNet34LiveRealistic

-- ════════════════════════════════════════════════════════════════
-- § The maxpool shift lemma (uniform channel offset)
-- ════════════════════════════════════════════════════════════════

theorem max_add_r (a b δ : ℝ) : max (a + δ) (b + δ) = max a b + δ := by
  rcases le_total a b with h | h
  · rw [max_eq_right h]; exact max_eq_right (by linarith)
  · rw [max_eq_left h]; exact max_eq_left (by linarith)

/-- **MaxPool shifts uniformly**: if channel 0 is channel 1 plus the constant `δ` at every
    position, the maxpool of channel 0 is the maxpool of channel 1 plus `δ` (the max of a
    uniformly-shifted family). No selection/argmax argument needed. -/
theorem maxPool2_shift {h w : Nat} (x : Tensor3 2 (2 * h) (2 * w)) (δ : ℝ)
    (hx : ∀ i j, x 0 i j = x 1 i j + δ) (hi : Fin h) (wi : Fin w) :
    maxPool2 x 0 hi wi = maxPool2 x 1 hi wi + δ := by
  simp only [maxPool2, hx]
  rw [max_add_r, max_add_r, max_add_r]

-- ════════════════════════════════════════════════════════════════
-- § The uniform channel-difference invariant `UDiff`
-- ════════════════════════════════════════════════════════════════

/-- `channel 0 = channel 1 + δ` at every spatial position. The equality analogue of
    `Dom2` (which is the strict `<`); threaded through the net the same way. -/
def UDiff {h w : Nat} (δ : ℝ) (u : Vec (2 * h * w)) : Prop :=
  ∀ (i : Fin h) (j : Fin w),
    (Tensor3.unflatten u : Tensor3 2 h w) 0 i j = (Tensor3.unflatten u : Tensor3 2 h w) 1 i j + δ

/-- **BN scales the uniform channel difference by its (positive, global) `istd`.** -/
theorem UDiff_bn {h w : Nat} (β δ : ℝ) (z : Vec (2 * h * w)) (hz : UDiff δ z) :
    UDiff (δ * bnIstd (2 * h * w) z 1) (bnForward (2 * h * w) 1 1 β z) := by
  intro i j
  show bnForward (2 * h * w) 1 1 β z (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), i), j))
     = bnForward (2 * h * w) 1 1 β z (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), i), j))
       + δ * bnIstd (2 * h * w) z 1
  have hd := bnForward_chan_diff (n := 2 * h * w) 1 β z
    (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), i), j))
    (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), i), j))
  have hzδ : z (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), i), j))
           - z (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), i), j)) = δ := by
    have := hz i j; simp only [Tensor3.unflatten] at this; linarith
  rw [hzδ] at hd; linarith

/-- Decimation preserves the uniform channel difference. -/
theorem UDiff_decimate {h w : Nat} (δ : ℝ) (a : Vec (2 * (2 * h) * (2 * w))) (ha : UDiff δ a) :
    UDiff δ (decimateFlat 2 h w a) := by
  intro i j
  rw [decimate_unflatten a 0 i j, decimate_unflatten a 1 i j]
  exact ha _ _

/-- Maxpool preserves the uniform channel difference (`maxPool2_shift`). -/
theorem UDiff_maxpool {h w : Nat} (δ : ℝ) (z : Vec (2 * (2 * h) * (2 * w))) (hz : UDiff δ z) :
    UDiff δ (maxPoolFlat 2 h w z) := by
  intro i j
  have heq : (Tensor3.unflatten (maxPoolFlat 2 h w z) : Tensor3 2 h w)
      = maxPool2 (Tensor3.unflatten z) := by
    simp only [maxPoolFlat, Tensor3.unflatten_flatten]
  rw [heq]
  exact maxPool2_shift (Tensor3.unflatten z) δ (fun a b => hz a b) i j

/-- Adding a channel-symmetric constant preserves the uniform channel difference. -/
theorem UDiff_add_const {h w : Nat} (δ c : ℝ) (u : Vec (2 * h * w)) (hu : UDiff δ u) :
    UDiff δ (fun k => u k + c) := by
  intro i j
  have h0 : (Tensor3.unflatten (fun k => u k + c) : Tensor3 2 h w) 0 i j
      = (Tensor3.unflatten u) 0 i j + c := rfl
  have h1 : (Tensor3.unflatten (fun k => u k + c) : Tensor3 2 h w) 1 i j
      = (Tensor3.unflatten u) 1 i j + c := rfl
  rw [h0, h1, hu i j]; ring

/-- **GAP of a uniform channel difference is that difference.** -/
theorem UDiff_gap {h w : Nat} (hh : 0 < h) (hw : 0 < w) (δ : ℝ) (u : Vec (2 * h * w))
    (hu : UDiff δ u) :
    globalAvgPoolFlat 2 h w u 0 = globalAvgPoolFlat 2 h w u 1 + δ := by
  have hpos : (0 : ℝ) < (h : ℝ) * w := by exact_mod_cast Nat.mul_pos hh hw
  simp only [globalAvgPoolFlat_as_sum]
  have key : ∀ p : Fin h × Fin w,
      (1 / (h * w : ℝ)) * u (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), p.1), p.2))
      = (1 / (h * w : ℝ)) * u (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), p.1), p.2))
        + (1 / (h * w : ℝ)) * δ := by
    intro p
    have := hu p.1 p.2; simp only [Tensor3.unflatten] at this
    rw [this]; ring
  rw [Finset.sum_congr rfl (fun p _ => key p), Finset.sum_add_distrib, Finset.sum_const]
  simp only [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul]
  congr 1
  push_cast
  field_simp

-- ════════════════════════════════════════════════════════════════
-- § The base point `Y224` and the uniform channel-0 perturbation `V224u`
-- ════════════════════════════════════════════════════════════════

/-- Channel-symmetric, strictly spatially decreasing base (`-(i·224 + j)`) — per-channel
    positionally injective (the maxpool no-tie) and channel-symmetric (carrier vanishes). -/
noncomputable def Y224tensor : Tensor3 2 (2 * 112) (2 * 112) :=
  fun _c i j => -((i.val : ℝ) * 224 + (j.val : ℝ))

noncomputable def Y224 : Vec (2 * (2 * 112) * (2 * 112)) := Tensor3.flatten Y224tensor

theorem unflatten_Y224 : (Tensor3.unflatten Y224 : Tensor3 2 (2 * 112) (2 * 112)) = Y224tensor :=
  Tensor3.unflatten_flatten Y224tensor

/-- The uniform perturbation: all of channel 0. -/
noncomputable def V224u : Vec (2 * (2 * 112) * (2 * 112)) :=
  Tensor3.flatten (fun c _ _ => if c = (0 : Fin 2) then (1 : ℝ) else 0)

theorem unflatten_V224u (c : Fin 2) (i j : Fin (2 * 112)) :
    (Tensor3.unflatten V224u : Tensor3 2 (2 * 112) (2 * 112)) c i j = if c = (0 : Fin 2) then 1 else 0 := by
  rw [V224u, Tensor3.unflatten_flatten]

/-- **The carrier along the ray is exactly `t`.** -/
theorem UDiff_ray (t : ℝ) : UDiff t (Y224 + t • V224u) := by
  intro i j
  have hY : ∀ c : Fin 2, (Tensor3.unflatten Y224 : Tensor3 2 (2 * 112) (2 * 112)) c i j
      = -((i.val : ℝ) * 224 + (j.val : ℝ)) := by intro c; rw [unflatten_Y224]; rfl
  have e : ∀ c : Fin 2, (Tensor3.unflatten (Y224 + t • V224u) : Tensor3 2 (2 * 112) (2 * 112)) c i j
      = (Tensor3.unflatten Y224 : Tensor3 2 (2 * 112) (2 * 112)) c i j
        + t * (Tensor3.unflatten V224u : Tensor3 2 (2 * 112) (2 * 112)) c i j := by
    intro c; simp only [Tensor3.unflatten, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [e 0, e 1, hY 0, hY 1, unflatten_V224u 0 i j, unflatten_V224u 1 i j]
  simp

-- ════════════════════════════════════════════════════════════════
-- § The globally-smooth ReLU-free twin `liveFwd224S`
-- ════════════════════════════════════════════════════════════════

/-- The stem ReLU is globally off (`bn₁₆₀ ≥ 160 − √25088 > 0`), for *every* input. -/
theorem stemS224_bn_pos (v : Vec (2 * (2 * 112) * (2 * 112))) (k : Fin (2 * 112 * 112)) :
    0 < bnForward (2 * 112 * 112) 1 1 160 (flatConvStride2 WsId2 Zb2 v) k := by
  have hlb := bnForward_lb (n := 2 * 112 * 112) 1 1 160 (by norm_num) (flatConvStride2 WsId2 Zb2 v) k
  rw [abs_one, one_mul] at hlb
  linarith [sqrt25088_lt_160]

/-- The stem with the (globally-off) ReLU removed: `bn₁₆₀ ∘ decimate`. -/
noncomputable def stemS224 (v : Vec (2 * (2 * 112) * (2 * 112))) : Vec (2 * 112 * 112) :=
  bnForward (2 * 112 * 112) 1 1 160 (decimateFlat 2 112 112 v)

theorem stem224_eq_stemS224 (v : Vec (2 * (2 * 112) * (2 * 112))) : stem224 v = stemS224 v := by
  show (relu (2 * 112 * 112) ∘ bnForward (2 * 112 * 112) 1 1 160 ∘ flatConvStride2 WsId2 Zb2) v = stemS224 v
  simp only [Function.comp_apply]
  rw [relu_id_of_pos (fun k => stemS224_bn_pos v k), flatConvStride2_diag WsId2 (fun o i => rfl) v]
  rfl

/-- The live downsample with the (globally-off) ReLU removed: `bn_βp ∘ decimate + 1`. -/
noncomputable def ldSβ (h w : Nat) (βp : ℝ) (a : Vec (2 * (2 * h) * (2 * w))) : Vec (2 * h * w) :=
  fun k => bnForward (2 * h * w) 1 1 βp (decimateFlat 2 h w a) k + 1

theorem ldSβ_pos (h w : Nat) (βp : ℝ) (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp)
    (a : Vec (2 * (2 * h) * (2 * w))) (k : Fin (2 * h * w)) : 0 < ldSβ h w βp a k := by
  simp only [ldSβ]
  have hlb := bnForward_lb (n := 2 * h * w) 1 1 βp (by norm_num) (decimateFlat 2 h w a) k
  rw [abs_one, one_mul] at hlb
  linarith [hn]

theorem liveDownβ_eq_ldSβ (h w : Nat) (βp : ℝ) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    liveDownβ h w βp a = ldSβ h w βp a := by
  have hres : residualProj
      (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 WsP2 Zb2)
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a
      = ldSβ h w βp a := by
    funext k
    have hbody : ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k = 1 :=
      congrFun (liveDownβ_body_const h w hhw a) k
    simp only [residualProj, biPath, Function.comp_apply, ldSβ] at hbody ⊢
    rw [hbody, flatConvStride2_diag WsP2 (fun o i => rfl) a]
  show relu (2 * h * w) (residualProj
    (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 WsP2 Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a) = ldSβ h w βp a
  rw [hres]
  exact relu_id_of_pos (fun k => ldSβ_pos h w βp hn a k)

/-- The ReLU-free twin of `liveFwd224`. -/
noncomputable def liveFwd224S : Vec (2 * (2 * 112) * (2 * 112)) → Vec 2 :=
  dense Wd2 bd2 ∘ globalAvgPoolFlat 2 7 7 ∘
    ldSβ 7 7 64 ∘ ldSβ 14 14 64 ∘ ldSβ 28 28 64 ∘ maxPoolFlat 2 56 56 ∘ stemS224

theorem liveFwd224_eq_S : liveFwd224 = liveFwd224S := by
  show dense Wd2 bd2 ∘ globalAvgPoolFlat 2 7 7 ∘
    liveDownβ 7 7 64 ∘ liveDownβ 14 14 64 ∘ liveDownβ 28 28 64 ∘ maxPoolFlat 2 56 56 ∘ stem224 = liveFwd224S
  rw [funext stem224_eq_stemS224,
      funext (liveDownβ_eq_ldSβ 28 28 64 (by norm_num) (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num))),
      funext (liveDownβ_eq_ldSβ 14 14 64 (by norm_num) (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num))),
      funext (liveDownβ_eq_ldSβ 7 7 64 (by norm_num) (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)))]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The running activations and the positive istd product `Rr224`
-- ════════════════════════════════════════════════════════════════

noncomputable def Pr224 (t : ℝ) : Vec (2 * 56 * 56) := maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u))
noncomputable def W28r (t : ℝ) : Vec (2 * 28 * 28) := ldSβ 28 28 64 (Pr224 t)
noncomputable def W14r (t : ℝ) : Vec (2 * 14 * 14) := ldSβ 14 14 64 (W28r t)

/-- The product of the four BN inverse-stds along the ray. -/
noncomputable def Rr224 (t : ℝ) : ℝ :=
  bnIstd (2 * 112 * 112) (decimateFlat 2 112 112 (Y224 + t • V224u)) 1
  * (bnIstd (2 * 28 * 28) (decimateFlat 2 28 28 (Pr224 t)) 1
     * (bnIstd (2 * 14 * 14) (decimateFlat 2 14 14 (W28r t)) 1
        * bnIstd (2 * 7 * 7) (decimateFlat 2 7 7 (W14r t)) 1))

theorem Rr224_pos (t : ℝ) : 0 < Rr224 t := by
  unfold Rr224
  exact mul_pos (bnIstd_pos _ 1 one_pos)
    (mul_pos (bnIstd_pos _ 1 one_pos)
      (mul_pos (bnIstd_pos _ 1 one_pos) (bnIstd_pos _ 1 one_pos)))

-- ── continuity of `Rr224` (for the slope argument) ──

theorem ray224_continuous : Continuous (fun t : ℝ => Y224 + t • V224u) :=
  continuous_const.add (continuous_id.smul continuous_const)

theorem stemS224_continuous : Continuous stemS224 :=
  (bnForward_differentiable (2 * 112 * 112) 1 1 160 one_pos).continuous.comp
    (decimateFlat_differentiable 2 112 112).continuous

theorem ldSβ_continuous (h w : Nat) (βp : ℝ) : Continuous (ldSβ h w βp) := by
  have h1 : Continuous (fun a : Vec (2 * (2 * h) * (2 * w)) =>
      bnForward (2 * h * w) 1 1 βp (decimateFlat 2 h w a)) :=
    (bnForward_differentiable (2 * h * w) 1 1 βp one_pos).continuous.comp
      (decimateFlat_differentiable 2 h w).continuous
  exact continuous_pi (fun k => ((continuous_apply k).comp h1).add continuous_const)

theorem Pr224_continuous : Continuous Pr224 :=
  (maxPoolFlat_continuous 2 56 56).comp (stemS224_continuous.comp ray224_continuous)
theorem W28r_continuous : Continuous W28r := (ldSβ_continuous 28 28 64).comp Pr224_continuous
theorem W14r_continuous : Continuous W14r := (ldSβ_continuous 14 14 64).comp W28r_continuous

theorem Rr224_continuous : Continuous Rr224 := by
  have c1 : Continuous (fun t : ℝ => bnIstd (2 * 112 * 112) (decimateFlat 2 112 112 (Y224 + t • V224u)) 1) :=
    (bnIstd_cont (n := 2 * 112 * 112) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 112 112).continuous.comp ray224_continuous)
  have c2 : Continuous (fun t : ℝ => bnIstd (2 * 28 * 28) (decimateFlat 2 28 28 (Pr224 t)) 1) :=
    (bnIstd_cont (n := 2 * 28 * 28) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 28 28).continuous.comp Pr224_continuous)
  have c3 : Continuous (fun t : ℝ => bnIstd (2 * 14 * 14) (decimateFlat 2 14 14 (W28r t)) 1) :=
    (bnIstd_cont (n := 2 * 14 * 14) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 14 14).continuous.comp W28r_continuous)
  have c4 : Continuous (fun t : ℝ => bnIstd (2 * 7 * 7) (decimateFlat 2 7 7 (W14r t)) 1) :=
    (bnIstd_cont (n := 2 * 7 * 7) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 7 7).continuous.comp W14r_continuous)
  exact c1.mul (c2.mul (c3.mul c4))

-- ════════════════════════════════════════════════════════════════
-- § The output difference along the ray is `t · Rr224 t`
-- ════════════════════════════════════════════════════════════════

theorem gd_ray224 (t : ℝ) :
    liveFwd224S (Y224 + t • V224u) 0 - liveFwd224S (Y224 + t • V224u) 1 = t * Rr224 t := by
  -- thread UDiff through the layers (dims pinned: the 2h→h decimate/maxpool unify nonlinearly)
  have u0 : UDiff t (Y224 + t • V224u) := UDiff_ray t
  have us : UDiff (t * bnIstd (2 * 112 * 112) (decimateFlat 2 112 112 (Y224 + t • V224u)) 1)
      (stemS224 (Y224 + t • V224u)) :=
    UDiff_bn 160 t (decimateFlat 2 112 112 (Y224 + t • V224u))
      (UDiff_decimate (h := 112) (w := 112) t (Y224 + t • V224u) u0)
  have um := UDiff_maxpool (h := 56) (w := 56) _ (stemS224 (Y224 + t • V224u)) us
  have u28 : UDiff _ (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u)))) :=
    UDiff_add_const _ 1 _
      (UDiff_bn 64 _ (decimateFlat 2 28 28 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u))))
        (UDiff_decimate (h := 28) (w := 28) _ (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u))) um))
  have u14 : UDiff _ (ldSβ 14 14 64 (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u))))) :=
    UDiff_add_const _ 1 _
      (UDiff_bn 64 _ (decimateFlat 2 14 14 (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u)))))
        (UDiff_decimate (h := 14) (w := 14) _ (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u)))) u28))
  have u7 : UDiff _ (ldSβ 7 7 64 (ldSβ 14 14 64 (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u)))))) :=
    UDiff_add_const _ 1 _
      (UDiff_bn 64 _ (decimateFlat 2 7 7 (ldSβ 14 14 64 (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u))))))
        (UDiff_decimate (h := 7) (w := 7) _ (ldSβ 14 14 64 (ldSβ 28 28 64 (maxPoolFlat 2 56 56 (stemS224 (Y224 + t • V224u))))) u14))
  have hgap := UDiff_gap (h := 7) (w := 7) (by norm_num) (by norm_num) _ _ u7
  simp only [liveFwd224S, Function.comp_apply, dense_Wd2_apply]
  rw [hgap]
  simp only [Rr224, Pr224, W28r, W14r]
  ring

theorem gd_hasDerivAt224 :
    HasDerivAt (fun t : ℝ => liveFwd224S (Y224 + t • V224u) 0 - liveFwd224S (Y224 + t • V224u) 1)
      (Rr224 0) 0 := by
  have hmul : HasDerivAt (fun t : ℝ => t * Rr224 t) (Rr224 0) 0 := by
    rw [hasDerivAt_iff_tendsto_slope]
    have hslope : slope (fun t : ℝ => t * Rr224 t) 0 =ᶠ[𝓝[≠] (0 : ℝ)] Rr224 := by
      filter_upwards [self_mem_nhdsWithin] with y hy
      have hy0 : y ≠ 0 := by simpa using hy
      simp only [slope_def_field, sub_zero, zero_mul]
      rw [mul_comm, mul_div_assoc, div_self hy0, mul_one]
    exact Filter.Tendsto.congr' hslope.symm
      (Rr224_continuous.continuousAt.tendsto.mono_left nhdsWithin_le_nhds)
  exact hmul.congr_of_eventuallyEq (Filter.Eventually.of_forall (fun t => gd_ray224 t))

-- ════════════════════════════════════════════════════════════════
-- § The whole-net VJP at the base `Y224` (maxpool no-tie via injectivity)
-- ════════════════════════════════════════════════════════════════

theorem decimY224_val (ci : Fin 2) (r s : Fin 112) :
    decimateFlat 2 112 112 Y224 (finProdFinEquiv (finProdFinEquiv (ci, r), s))
      = -(448 * (r.val : ℝ) + 2 * (s.val : ℝ)) := by
  have e : decimateFlat 2 112 112 Y224 (finProdFinEquiv (finProdFinEquiv (ci, r), s))
      = (Tensor3.unflatten (decimateFlat 2 112 112 Y224) : Tensor3 2 112 112) ci r s := rfl
  rw [e, decimate_unflatten Y224 ci r s, unflatten_Y224]
  simp only [Y224tensor]
  push_cast
  ring

theorem stem224_Y_maxpool_smooth :
    MaxPool2Smooth (Tensor3.unflatten (stem224 Y224) : Tensor3 2 (2 * 56) (2 * 56)) := by
  apply maxPool2Smooth_of_injective
  intro ci r r' s s' heq
  rw [stem224_eq_stemS224 Y224] at heq
  have hdec := bnForward_coord_inj (n := 2 * 112 * 112) 1 160 one_pos (decimateFlat 2 112 112 Y224)
    (finProdFinEquiv (finProdFinEquiv (ci, r), s))
    (finProdFinEquiv (finProdFinEquiv (ci, r'), s')) heq
  rw [decimY224_val, decimY224_val] at hdec
  have hnat : 448 * r.val + 2 * s.val = 448 * r'.val + 2 * s'.val := by
    have h := neg_injective hdec
    have := r.isLt; have := s.isLt; have := r'.isLt; have := s'.isLt
    exact_mod_cast h
  have := r.isLt; have := s.isLt; have := r'.isLt; have := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

noncomputable def stem224_vjp_Y : HasVJPAt stem224 Y224 :=
  convBnReluStrided_has_vjp_at WsId2 Zb2 1 1 160 (by norm_num) Y224
    (fun k => ne_of_gt (stemS224_bn_pos Y224 k))

theorem stem224_diff_Y : DifferentiableAt ℝ stem224 Y224 :=
  DifferentiableAt.comp Y224
    (relu_differentiableAt_of_smooth (2 * 112 * 112) _ (fun k => ne_of_gt (stemS224_bn_pos Y224 k)))
    ((convBnStrided_differentiable WsId2 Zb2 1 1 160 (by norm_num)) Y224)

theorem mp_point_eq_Y224 :
    Tensor3.flatten (Tensor3.unflatten (stem224 Y224) : Tensor3 2 (2 * 56) (2 * 56)) = stem224 Y224 :=
  Tensor3.flatten_unflatten (stem224 Y224)

noncomputable def hmp_vjp_Y224 : HasVJPAt (maxPoolFlat 2 56 56) (stem224 Y224) := by
  have h := maxPoolFlat_has_vjp_at (Tensor3.unflatten (stem224 Y224) : Tensor3 2 (2 * 56) (2 * 56))
    stem224_Y_maxpool_smooth
  rwa [mp_point_eq_Y224] at h

theorem hmp_diff_Y224 : DifferentiableAt ℝ (maxPoolFlat 2 56 56) (stem224 Y224) := by
  have h := maxPoolFlat_differentiableAt (Tensor3.unflatten (stem224 Y224) : Tensor3 2 (2 * 56) (2 * 56))
    stem224_Y_maxpool_smooth (by norm_num) (by norm_num) (by norm_num)
  rwa [mp_point_eq_Y224] at h

/-- The whole 224×224 live ResNet-34 VJP at the seal witness base `Y224`. -/
noncomputable def liveFwd224_has_vjp_at_Y : HasVJPAt liveFwd224 Y224 :=
  resnet34_has_vjp_at stem224 (maxPoolFlat 2 56 56)
    ([] : List (Vec (2 * 56 * 56) → Vec (2 * 56 * 56))) (liveDownβ 28 28 64)
    ([] : List (Vec (2 * 28 * 28) → Vec (2 * 28 * 28))) (liveDownβ 14 14 64)
    ([] : List (Vec (2 * 14 * 14) → Vec (2 * 14 * 14))) (liveDownβ 7 7 64)
    ([] : List (Vec (2 * 7 * 7) → Vec (2 * 7 * 7)))
    (globalAvgPoolFlat 2 7 7) (dense Wd2 bd2) Y224
    ⟨stem224_vjp_Y, stem224_diff_Y⟩
    ⟨hmp_vjp_Y224, hmp_diff_Y224⟩
    PUnit.unit
    ⟨liveDownβ_vjp 28 28 64 (by norm_num) (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num)) _,
     liveDownβ_diff 28 28 64 (by norm_num) (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num)) _⟩
    PUnit.unit
    ⟨liveDownβ_vjp 14 14 64 (by norm_num) (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num)) _,
     liveDownβ_diff 14 14 64 (by norm_num) (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num)) _⟩
    PUnit.unit
    ⟨liveDownβ_vjp 7 7 64 (by norm_num) (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)) _,
     liveDownβ_diff 7 7 64 (by norm_num) (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)) _⟩
    PUnit.unit
    ⟨(globalAvgPoolFlat_has_vjp 2 7 7).toHasVJPAt _, (globalAvgPoolFlat_differentiable 2 7 7) _⟩
    ⟨(dense_has_vjp Wd2 bd2).toHasVJPAt _, (dense_differentiable Wd2 bd2) _⟩

theorem liveFwd224_diff_Y : DifferentiableAt ℝ liveFwd224 Y224 := by
  show DifferentiableAt ℝ (dense Wd2 bd2 ∘ globalAvgPoolFlat 2 7 7 ∘
    liveDownβ 7 7 64 ∘ liveDownβ 14 14 64 ∘ liveDownβ 28 28 64 ∘ maxPoolFlat 2 56 56 ∘ stem224) Y224
  exact (dense_differentiable Wd2 bd2).differentiableAt.comp Y224
    ((globalAvgPoolFlat_differentiable 2 7 7).differentiableAt.comp Y224
      ((liveDownβ_diff 7 7 64 (by norm_num) (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)) _).comp Y224
        ((liveDownβ_diff 14 14 64 (by norm_num) (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num)) _).comp Y224
          ((liveDownβ_diff 28 28 64 (by norm_num) (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num)) _).comp Y224
            (hmp_diff_Y224.comp Y224 stem224_diff_Y)))))

-- ════════════════════════════════════════════════════════════════
-- § The seal
-- ════════════════════════════════════════════════════════════════

/-- **`fderiv ℝ liveFwd224 Y224 ≠ 0`** — the 224×224 live ResNet-34's whole-net Jacobian is
    genuinely non-trivial at the witness base `Y224` (level-3 seal, realistic dims). -/
theorem liveFwd224_jacobian_nonzero : fderiv ℝ liveFwd224 Y224 ≠ 0 := by
  intro hzero
  have hfd : HasFDerivAt liveFwd224 (0 : Vec (2 * (2 * 112) * (2 * 112)) →L[ℝ] Vec 2) Y224 := by
    rw [← hzero]; exact liveFwd224_diff_Y.hasFDerivAt
  have hsmul : HasDerivAt (fun t : ℝ => Y224 + t • V224u) V224u 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const V224u).const_add Y224
  have hcomp : HasDerivAt (fun t : ℝ => liveFwd224 (Y224 + t • V224u)) (0 : Vec 2) 0 := by
    have := HasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hfd hsmul (by simp)
    exact this
  have hpi := hasDerivAt_pi.mp hcomp
  have hd : HasDerivAt (fun t : ℝ => liveFwd224 (Y224 + t • V224u) 0 - liveFwd224 (Y224 + t • V224u) 1) 0 0 := by
    have := (hpi 0).sub (hpi 1)
    simp only [Pi.zero_apply, sub_zero] at this
    exact this
  rw [liveFwd224_eq_S] at hd
  exact (Rr224_pos 0).ne' (gd_hasDerivAt224.unique hd)

/-- **The level-3 seal for the 224×224 live ResNet-34** (Item D, level 3): the proven
    whole-network backward of the nonzero-weight live ResNet-34 at real ImageNet resolution
    is **not the zero map** at the witness base `Y224`. -/
theorem liveFwd224_backward_nontrivial :
    ∃ (j₀ : Fin 2) (i₀ : Fin (2 * (2 * 112) * (2 * 112))),
      liveFwd224_has_vjp_at_Y.backward (basisVec j₀) i₀ ≠ 0 :=
  liveFwd224_has_vjp_at_Y.backward_nontrivial_of_fderiv_ne liveFwd224_jacobian_nonzero

end R34RealSeal
end Proofs
