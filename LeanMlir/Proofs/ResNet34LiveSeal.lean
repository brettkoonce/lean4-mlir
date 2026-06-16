import LeanMlir.Proofs.ResNet34LivePC
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# `ResNet34LivePC` nonzero-Jacobian seal — Item A level 3

`planning/whole_network_backward.md` Item A: take the live 2-channel ResNet-34
witness (`ResNet34LivePC.liveFwd2`) from **level 2**
(`liveFwd2_nonconstant : forward X2 ≠ forward 0`) up to **level 3**, the
*nonzero-Jacobian seal*: the proven whole-net backward is genuinely non-trivial
at a witness point — `fderiv ℝ liveFwd2 · ≠ 0`, hence (via the `JacobianSeal`
bridge) `backward (basisVec j₀) i₀ ≠ 0`.

## Why this needs a new idea (vs. `Mnv2Live`)

`Mnv2Live` is *globally* smooth (its ReLU6 never binds), so its seal is computed
along a ray from the input `0`, where the linear stem-carrier vanishes and the
product-rule cross-term drops. `liveFwd2`'s **maxpool** is a genuine kink that
binds off-witness, so the input-`0` trick does not transfer (at `0` the maxpool
has ties and the VJP fails). The fix is to keep the *same* "carrier-vanishes-at-
base" structure but move the base point:

* Seal at a **channel-symmetric, spatially-decreasing** base `Y` — the channel
  difference (the live carrier) vanishes at `Y`, the ResNet analogue of Mnv2's
  `0`, **and** `Y` is per-channel positionally injective so the maxpool has no
  ties (the VJP holds at `Y`).
* Perturb the single top-left input coordinate. The maxpool window-`(0,0)` argmax
  is the top-left corner (decreasing `Y`), so *eventually* along the ray the
  maxpool is a fixed selection — the only non-smooth step, handled by a focused
  `maxPool2_eq_at_max`.
* Track the **channel difference at the top-left** `cd` through the net. By the
  *exact* BN channel-difference identity `bn z k₀ − bn z k₁ = (z k₀ − z k₁)·istd`,
  every BN layer multiplies `cd` by a positive `istd`; so along the ray
  `liveFwd2(Y+t·V) 0 − liveFwd2(Y+t·V) 1 = t · Π(t)` with `Π` a product of four
  positive `istd`s. The factor `t` (from `cd(Y) = 0`) makes the cross-terms drop
  at `t = 0`: the directional derivative is `Π(0) ≠ 0`, with no BN-variance
  derivative ever needed — exactly the Mnv2 mechanism, transplanted.
-/

namespace Proofs
namespace ResNet34LiveSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs ResNet34Live2 ResNet34LivePC

-- ════════════════════════════════════════════════════════════════
-- § The channel-symmetric base point `Y` and the perturbation `V`
-- ════════════════════════════════════════════════════════════════

/-- The base tensor: channel-symmetric, strictly spatially **decreasing**
    (`-(i·32 + j)`), so within every 2×2 window the top-left corner is the strict
    max, and the two channels coincide (the carrier vanishes at base). -/
noncomputable def Ytensor : Tensor3 2 (2 * 16) (2 * 16) :=
  fun _c i j => -((i.val : ℝ) * 32 + (j.val : ℝ))

/-- The base point, flattened. -/
noncomputable def Y : Vec (2 * (2 * 16) * (2 * 16)) := Tensor3.flatten Ytensor

theorem unflatten_Y : (Tensor3.unflatten Y : Tensor3 2 (2 * 16) (2 * 16)) = Ytensor :=
  Tensor3.unflatten_flatten Ytensor

/-- The perturbation: the single top-left input coordinate of channel 0. -/
noncomputable def V : Vec (2 * (2 * 16) * (2 * 16)) :=
  basisVec (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), (0 : Fin (2 * 16))), (0 : Fin (2 * 16))))

-- ════════════════════════════════════════════════════════════════
-- § The exact BN channel-difference identity
-- ════════════════════════════════════════════════════════════════

/-- **Scalar BN (γ=1) acts on coordinate differences by the positive `istd`.**
    The exact identity behind `bnForward_chan_lt`: `bn z k₀ − bn z k₁ =
    (z k₀ − z k₁)·istd`. This is what propagates the channel difference through
    every BN layer undamped (and is what makes the seal's `istd`-derivative
    terms carry a factor of the carrier). -/
theorem bnForward_chan_diff {n : Nat} (ε β : ℝ) (z : Vec n) (k₀ k₁ : Fin n) :
    bnForward n ε 1 β z k₀ - bnForward n ε 1 β z k₁ = (z k₀ - z k₁) * bnIstd n z ε := by
  simp only [bnForward, bnXhat, one_mul]; ring

-- ════════════════════════════════════════════════════════════════
-- § The globally-smooth ReLU-free twin `liveFwd2S`
--   Every ReLU in `liveFwd2` is globally off (BN positivity), so `liveFwd2`
--   equals a ReLU-free composition whose only non-smooth step is the maxpool.
-- ════════════════════════════════════════════════════════════════

/-- The stem with the (globally-off) ReLU removed: `bn₃₀ ∘ decimate`. -/
noncomputable def stemS (v : Vec (2 * (2 * 16) * (2 * 16))) : Vec (2 * 16 * 16) :=
  bnForward (2 * 16 * 16) 1 1 30 (decimateFlat 2 16 16 v)

/-- The live downsample with the (globally-off) ReLU removed: `bn₂₀ ∘ decimate + 1`. -/
noncomputable def ldS (h w : Nat) (a : Vec (2 * (2 * h) * (2 * w))) : Vec (2 * h * w) :=
  fun k => bnForward (2 * h * w) 1 1 20 (decimateFlat 2 h w a) k + 1

/-- The ReLU-free twin of `liveFwd2`. -/
noncomputable def liveFwd2S : Vec (2 * (2 * 16) * (2 * 16)) → Vec 2 :=
  dense Wd2 bd2 ∘ globalAvgPoolFlat 2 1 1 ∘
    ldS 1 1 ∘ ldS 2 2 ∘ ldS 4 4 ∘ maxPoolFlat 2 8 8 ∘ stemS

-- the stem ReLU is globally off: `bn₃₀ ≥ 30 − √512 > 0` for *every* input
theorem stemS_bn_pos (v : Vec (2 * (2 * 16) * (2 * 16))) (k : Fin (2 * 16 * 16)) :
    0 < bnForward (2 * 16 * 16) 1 1 30 (flatConvStride2 WsId2 Zb2 v) k := by
  have hlb := bnForward_lb (n := 2 * 16 * 16) 1 1 30 (by norm_num) (flatConvStride2 WsId2 Zb2 v) k
  rw [abs_one, one_mul] at hlb
  linarith [sqrt512_lt_30]

theorem stem2_eq_stemS (v : Vec (2 * (2 * 16) * (2 * 16))) : stem2 v = stemS v := by
  show (relu (2 * 16 * 16) ∘ bnForward (2 * 16 * 16) 1 1 30 ∘ flatConvStride2 WsId2 Zb2) v = stemS v
  simp only [Function.comp_apply]
  rw [relu_id_of_pos (fun k => stemS_bn_pos v k), flatConvStride2_diag WsId2 (fun o i => rfl) v]
  rfl

theorem ldS_pos (h w : Nat) (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20)
    (a : Vec (2 * (2 * h) * (2 * w))) (k : Fin (2 * h * w)) : 0 < ldS h w a k := by
  simp only [ldS]
  have hlb := bnForward_lb (n := 2 * h * w) 1 1 20 (by norm_num) (decimateFlat 2 h w a) k
  rw [abs_one, one_mul] at hlb
  linarith [hn]

theorem liveDownPC_eq_ldS (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) :
    liveDownPC h w a = ldS h w a := by
  have hres : residualProj
      (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2)
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a
      = ldS h w a := by
    funext k
    have hbody : ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k = 1 :=
      congrFun (liveDownPC_body_const h w hhw a) k
    simp only [residualProj, biPath, Function.comp_apply, ldS] at hbody ⊢
    rw [hbody, flatConvStride2_diag WsP2 (fun o i => rfl) a]
  show relu (2 * h * w) (residualProj
    (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a) = ldS h w a
  rw [hres]
  exact relu_id_of_pos (fun k => ldS_pos h w hn a k)

theorem stem2_eq_stemS' : stem2 = stemS := funext stem2_eq_stemS

theorem ld_eq (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) : liveDownPC h w = ldS h w :=
  funext (liveDownPC_eq_ldS h w hhw hn)

theorem liveFwd2_eq_S : liveFwd2 = liveFwd2S := by
  show dense Wd2 bd2 ∘ globalAvgPoolFlat 2 1 1 ∘
    liveDownPC 1 1 ∘ liveDownPC 2 2 ∘ liveDownPC 4 4 ∘ maxPoolFlat 2 8 8 ∘ stem2 = liveFwd2S
  rw [stem2_eq_stemS', ld_eq 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)),
      ld_eq 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)),
      ld_eq 1 1 (by norm_num) (sqrt_lt_20 (by norm_num))]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The channel difference at the top-left, and how each layer scales it
-- ════════════════════════════════════════════════════════════════

/-- The channel difference at the top-left spatial position. -/
noncomputable def cd {h w : Nat} [NeZero h] [NeZero w] (u : Vec (2 * h * w)) : ℝ :=
  (Tensor3.unflatten u : Tensor3 2 h w) 0 0 0 - (Tensor3.unflatten u : Tensor3 2 h w) 1 0 0

/-- Decimation reads the top-left through to the top-left (`2·0 = 0`). -/
theorem decimate_val00 (h w : Nat) [NeZero h] [NeZero w] (a : Vec (2 * (2 * h) * (2 * w)))
    (c : Fin 2) :
    decimateFlat 2 h w a (finProdFinEquiv (finProdFinEquiv (c, (0 : Fin h)), (0 : Fin w)))
      = (Tensor3.unflatten a : Tensor3 2 (2 * h) (2 * w)) c 0 0 := by
  have h1 : decimateFlat 2 h w a (finProdFinEquiv (finProdFinEquiv (c, (0 : Fin h)), (0 : Fin w)))
      = (Tensor3.unflatten (decimateFlat 2 h w a) : Tensor3 2 h w) c 0 0 := rfl
  rw [h1, decimate_unflatten a c 0 0]
  congr 1

/-- **`ldS` multiplies the top-left channel difference by `bnIstd`.** -/
theorem cd_ldS (h w : Nat) [NeZero h] [NeZero w] (a : Vec (2 * (2 * h) * (2 * w))) :
    cd (ldS h w a) = cd a * bnIstd (2 * h * w) (decimateFlat 2 h w a) 1 := by
  have key : ∀ c : Fin 2, (Tensor3.unflatten (ldS h w a) : Tensor3 2 h w) c 0 0
      = bnForward (2 * h * w) 1 1 20 (decimateFlat 2 h w a)
          (finProdFinEquiv (finProdFinEquiv (c, (0 : Fin h)), (0 : Fin w))) + 1 := fun _ => rfl
  have e1 : (bnForward (2 * h * w) 1 1 20 (decimateFlat 2 h w a)
        (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), (0 : Fin h)), (0 : Fin w))) + 1)
      - (bnForward (2 * h * w) 1 1 20 (decimateFlat 2 h w a)
        (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), (0 : Fin h)), (0 : Fin w))) + 1)
      = (decimateFlat 2 h w a (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), (0 : Fin h)), (0 : Fin w)))
          - decimateFlat 2 h w a (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), (0 : Fin h)), (0 : Fin w))))
        * bnIstd (2 * h * w) (decimateFlat 2 h w a) 1 := by
    rw [← bnForward_chan_diff (n := 2 * h * w) 1 20 (decimateFlat 2 h w a)]; ring
  simp only [cd, key, e1, decimate_val00 h w a 0, decimate_val00 h w a 1]

/-- **`stemS` multiplies the top-left channel difference by `bnIstd`.** -/
theorem cd_stemS (v : Vec (2 * (2 * 16) * (2 * 16))) :
    cd (stemS v) = cd v * bnIstd (2 * 16 * 16) (decimateFlat 2 16 16 v) 1 := by
  have key : ∀ c : Fin 2, (Tensor3.unflatten (stemS v) : Tensor3 2 16 16) c 0 0
      = bnForward (2 * 16 * 16) 1 1 30 (decimateFlat 2 16 16 v)
          (finProdFinEquiv (finProdFinEquiv (c, (0 : Fin 16)), (0 : Fin 16))) := fun _ => rfl
  simp only [cd, key]
  rw [bnForward_chan_diff (n := 2 * 16 * 16) 1 30 (decimateFlat 2 16 16 v),
      decimate_val00 16 16 v 0, decimate_val00 16 16 v 1]

/-- The top-left input flat indices of the two channels are distinct. -/
theorem idx01_ne :
    (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), (0 : Fin (2 * 16))), (0 : Fin (2 * 16))))
    ≠ (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), (0 : Fin (2 * 16))), (0 : Fin (2 * 16)))) := by
  intro h
  have h2 := finProdFinEquiv.injective h
  have h3 := finProdFinEquiv.injective (congrArg Prod.fst h2)
  exact absurd (congrArg Prod.fst h3) (by decide)

/-- **The carrier `cd` along the ray is exactly `t`** (base is channel-symmetric, the
    perturbation hits only channel 0's top-left). -/
theorem cd_ray (t : ℝ) : cd (Y + t • V) = t := by
  have hY : ∀ c : Fin 2, (Tensor3.unflatten Y : Tensor3 2 (2 * 16) (2 * 16)) c 0 0 = 0 := by
    intro c; rw [unflatten_Y]; simp [Ytensor]
  have hV0 : (Tensor3.unflatten V : Tensor3 2 (2 * 16) (2 * 16)) 0 0 0 = 1 := by
    simp [Tensor3.unflatten, V]
  have hV1 : (Tensor3.unflatten V : Tensor3 2 (2 * 16) (2 * 16)) 1 0 0 = 0 := by
    simp only [Tensor3.unflatten, V, basisVec_apply]; rw [if_neg idx01_ne]
  have e : ∀ c : Fin 2, (Tensor3.unflatten (Y + t • V) : Tensor3 2 (2 * 16) (2 * 16)) c 0 0
      = (Tensor3.unflatten Y : Tensor3 2 (2 * 16) (2 * 16)) c 0 0
        + t * (Tensor3.unflatten V : Tensor3 2 (2 * 16) (2 * 16)) c 0 0 := by
    intro c; simp only [Tensor3.unflatten, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  simp only [cd, e, hY, hV0, hV1]; ring

-- ════════════════════════════════════════════════════════════════
-- § Continuity infrastructure (for the istd product along the ray)
-- ════════════════════════════════════════════════════════════════

theorem bnIstd_cont {n : Nat} (ε : ℝ) (hε : 0 < ε) (k : Fin n) :
    Continuous (fun v : Vec n => bnIstd n v ε) :=
  (continuous_apply k).comp (bnIstdBroadcast_diff n ε hε).continuous

theorem maxPoolFlat_continuous (c h w : Nat) : Continuous (maxPoolFlat c h w) := by
  apply continuous_pi
  intro k
  dsimp only [maxPoolFlat, Tensor3.flatten, maxPool2, Tensor3.unflatten]
  fun_prop

theorem ray_continuous : Continuous (fun t : ℝ => Y + t • V) :=
  continuous_const.add (continuous_id.smul continuous_const)

theorem stemS_continuous : Continuous stemS :=
  (bnForward_differentiable (2 * 16 * 16) 1 1 30 one_pos).continuous.comp
    (decimateFlat_differentiable 2 16 16).continuous

theorem ldS_continuous (h w : Nat) : Continuous (ldS h w) := by
  have h1 : Continuous (fun a : Vec (2 * (2 * h) * (2 * w)) =>
      bnForward (2 * h * w) 1 1 20 (decimateFlat 2 h w a)) :=
    (bnForward_differentiable (2 * h * w) 1 1 20 one_pos).continuous.comp
      (decimateFlat_differentiable 2 h w).continuous
  exact continuous_pi (fun k => ((continuous_apply k).comp h1).add continuous_const)

-- ── the running activations and the istd product `Rr` ──

noncomputable def Pr (t : ℝ) : Vec (2 * 8 * 8) := maxPoolFlat 2 8 8 (stemS (Y + t • V))
noncomputable def W4r (t : ℝ) : Vec (2 * 4 * 4) := ldS 4 4 (Pr t)
noncomputable def W2r (t : ℝ) : Vec (2 * 2 * 2) := ldS 2 2 (W4r t)

/-- The product of the four BN inverse-stds along the ray (the smooth, positive
    nonlinear factor of the output channel difference). -/
noncomputable def Rr (t : ℝ) : ℝ :=
  bnIstd (2 * 16 * 16) (decimateFlat 2 16 16 (Y + t • V)) 1
  * (bnIstd (2 * 4 * 4) (decimateFlat 2 4 4 (Pr t)) 1
     * (bnIstd (2 * 2 * 2) (decimateFlat 2 2 2 (W4r t)) 1
        * bnIstd (2 * 1 * 1) (decimateFlat 2 1 1 (W2r t)) 1))

theorem Pr_continuous : Continuous Pr :=
  (maxPoolFlat_continuous 2 8 8).comp (stemS_continuous.comp ray_continuous)

theorem W4r_continuous : Continuous W4r := (ldS_continuous 4 4).comp Pr_continuous
theorem W2r_continuous : Continuous W2r := (ldS_continuous 2 2).comp W4r_continuous

theorem Rr_continuous : Continuous Rr := by
  have c1 : Continuous (fun t : ℝ => bnIstd (2 * 16 * 16) (decimateFlat 2 16 16 (Y + t • V)) 1) :=
    (bnIstd_cont (n := 2 * 16 * 16) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 16 16).continuous.comp ray_continuous)
  have c2 : Continuous (fun t : ℝ => bnIstd (2 * 4 * 4) (decimateFlat 2 4 4 (Pr t)) 1) :=
    (bnIstd_cont (n := 2 * 4 * 4) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 4 4).continuous.comp Pr_continuous)
  have c3 : Continuous (fun t : ℝ => bnIstd (2 * 2 * 2) (decimateFlat 2 2 2 (W4r t)) 1) :=
    (bnIstd_cont (n := 2 * 2 * 2) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 2 2).continuous.comp W4r_continuous)
  have c4 : Continuous (fun t : ℝ => bnIstd (2 * 1 * 1) (decimateFlat 2 1 1 (W2r t)) 1) :=
    (bnIstd_cont (n := 2 * 1 * 1) 1 one_pos ⟨0, by norm_num⟩).comp
      ((decimateFlat_differentiable 2 1 1).continuous.comp W2r_continuous)
  exact c1.mul (c2.mul (c3.mul c4))

theorem Rr_pos (t : ℝ) : 0 < Rr t := by
  unfold Rr
  exact mul_pos (bnIstd_pos _ 1 one_pos)
    (mul_pos (bnIstd_pos _ 1 one_pos)
      (mul_pos (bnIstd_pos _ 1 one_pos) (bnIstd_pos _ 1 one_pos)))

-- ════════════════════════════════════════════════════════════════
-- § At the base, the window top-left is the strict argmax
-- ════════════════════════════════════════════════════════════════

/-- The decimated base value at a window position of window `(0,0)`. -/
theorem decimY_win (c a' b' : Fin 2) :
    decimateFlat 2 16 16 Y
        (finProdFinEquiv (finProdFinEquiv (c, winRowInv (0 : Fin 8) a'), winColInv (0 : Fin 8) b'))
      = -(64 * (a'.val : ℝ) + 2 * (b'.val : ℝ)) := by
  have e : decimateFlat 2 16 16 Y
        (finProdFinEquiv (finProdFinEquiv (c, winRowInv (0 : Fin 8) a'), winColInv (0 : Fin 8) b'))
      = (Tensor3.unflatten (decimateFlat 2 16 16 Y) : Tensor3 2 16 16) c
          (winRowInv (0 : Fin 8) a') (winColInv (0 : Fin 8) b') := rfl
  rw [e, decimate_unflatten Y c (winRowInv (0 : Fin 8) a') (winColInv (0 : Fin 8) b'), unflatten_Y]
  simp only [Ytensor, winRowInv, winColInv, show ((0 : Fin 8).val : ℕ) = 0 from rfl]
  push_cast
  ring

/-- At the base `Y`, the window-`(0,0)` top-left strictly dominates every other cell. -/
theorem stemS_Y_winlt (c a' b' : Fin 2) (hne : ¬ (a' = 0 ∧ b' = 0)) :
    (Tensor3.unflatten (stemS Y) : Tensor3 2 16 16) c (winRowInv (0 : Fin 8) a') (winColInv (0 : Fin 8) b')
      < (Tensor3.unflatten (stemS Y) : Tensor3 2 16 16) c (winRowInv (0 : Fin 8) 0) (winColInv (0 : Fin 8) 0) := by
  have key : ∀ p q : Fin 2,
      (Tensor3.unflatten (stemS Y) : Tensor3 2 16 16) c (winRowInv (0 : Fin 8) p) (winColInv (0 : Fin 8) q)
      = bnForward (2 * 16 * 16) 1 1 30 (decimateFlat 2 16 16 Y)
          (finProdFinEquiv (finProdFinEquiv (c, winRowInv (0 : Fin 8) p), winColInv (0 : Fin 8) q)) :=
    fun _ _ => rfl
  rw [key a' b', key 0 0]
  apply bnForward_chan_lt 1 30 (by norm_num) (decimateFlat 2 16 16 Y)
  rw [decimY_win, decimY_win]
  have hpos : (0 : ℝ) < 64 * (a'.val : ℝ) + 2 * (b'.val : ℝ) := by
    have hn : 0 < 64 * a'.val + 2 * b'.val := by
      have ha := a'.isLt; have hb := b'.isLt
      rcases Nat.eq_zero_or_pos a'.val with h | h
      · rcases Nat.eq_zero_or_pos b'.val with h' | h'
        · exact absurd ⟨Fin.ext h, Fin.ext h'⟩ hne
        · omega
      · omega
    have hr : (0 : ℝ) < ((64 * a'.val + 2 * b'.val : ℕ) : ℝ) := by exact_mod_cast hn
    push_cast at hr; linarith
  simp only [Fin.val_zero, Nat.cast_zero, mul_zero, add_zero, neg_zero]
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The focused maxpool eventual selection along the ray
-- ════════════════════════════════════════════════════════════════

/-- The window value at `(c, a', b')` of window `(0,0)`, as a function of the ray
    parameter `t`. -/
noncomputable def sval (c a' b' : Fin 2) (t : ℝ) : ℝ :=
  (Tensor3.unflatten (stemS (Y + t • V)) : Tensor3 2 16 16) c (winRowInv (0 : Fin 8) a') (winColInv (0 : Fin 8) b')

theorem sval_continuous (c a' b' : Fin 2) : Continuous (sval c a' b') := by
  unfold sval
  exact (continuous_apply
    (finProdFinEquiv (finProdFinEquiv (c, winRowInv (0 : Fin 8) a'), winColInv (0 : Fin 8) b'))).comp
    (stemS_continuous.comp ray_continuous)

theorem ray0 : Y + (0 : ℝ) • V = Y := by rw [zero_smul, add_zero]

/-- Eventually along the ray, the window-`(0,0)` top-left dominates every cell. -/
theorem dom_eventually :
    ∀ᶠ t in 𝓝 (0 : ℝ), ∀ (c a' b' : Fin 2), sval c a' b' t ≤ sval c 0 0 t := by
  rw [eventually_all]; intro c
  rw [eventually_all]; intro a'
  rw [eventually_all]; intro b'
  by_cases h : a' = 0 ∧ b' = 0
  · obtain ⟨ha, hb⟩ := h; subst ha; subst hb
    exact Filter.Eventually.of_forall (fun _ => le_refl _)
  · have hlt0 : sval c a' b' 0 < sval c 0 0 0 := by
      simp only [sval, ray0]
      exact stemS_Y_winlt c a' b' h
    have h0 : (0 : ℝ) < sval c 0 0 0 - sval c a' b' 0 := by linarith
    have hev : ∀ᶠ t in 𝓝 (0 : ℝ), 0 < sval c 0 0 t - sval c a' b' t :=
      (((sval_continuous c 0 0).sub (sval_continuous c a' b')).continuousAt.tendsto).eventually
        (Ioi_mem_nhds h0)
    filter_upwards [hev] with t ht; linarith

/-- The maxpool selects the top-left, eventually along the ray. -/
theorem maxpool_sel_eventually :
    ∀ᶠ t in 𝓝 (0 : ℝ), ∀ c : Fin 2,
      maxPool2 (Tensor3.unflatten (stemS (Y + t • V)) : Tensor3 2 (2 * 8) (2 * 8)) c 0 0
        = (Tensor3.unflatten (stemS (Y + t • V)) : Tensor3 2 16 16) c 0 0 := by
  filter_upwards [dom_eventually] with t htdom
  intro c
  rw [maxPool2_eq_at_max (Tensor3.unflatten (stemS (Y + t • V)) : Tensor3 2 (2 * 8) (2 * 8))
    c 0 0 0 0 (fun a' b' => htdom c a' b')]
  congr 1

-- ════════════════════════════════════════════════════════════════
-- § The output channel difference, as `cd` times a positive `istd` product
-- ════════════════════════════════════════════════════════════════

/-- The maxpool preserves the top-left channel difference when it selects the top-left. -/
theorem cd_maxpool_eq (u : Vec (2 * 16 * 16))
    (hsel : ∀ c : Fin 2, maxPool2 (Tensor3.unflatten u : Tensor3 2 (2 * 8) (2 * 8)) c 0 0
      = (Tensor3.unflatten u : Tensor3 2 16 16) c 0 0) :
    cd (maxPoolFlat 2 8 8 u) = cd u := by
  have e : ∀ c : Fin 2, (Tensor3.unflatten (maxPoolFlat 2 8 8 u) : Tensor3 2 8 8) c 0 0
      = maxPool2 (Tensor3.unflatten u : Tensor3 2 (2 * 8) (2 * 8)) c 0 0 := by
    intro c; simp only [maxPoolFlat, Tensor3.unflatten_flatten]
  simp only [cd, e, hsel]

/-- The whole-net output channel difference, before the maxpool selection. -/
theorem liveFwd2S_diff (v : Vec (2 * (2 * 16) * (2 * 16))) :
    liveFwd2S v 0 - liveFwd2S v 1
      = cd (ldS 1 1 (ldS 2 2 (ldS 4 4 (maxPoolFlat 2 8 8 (stemS v))))) := by
  simp only [liveFwd2S, Function.comp_apply, dense_Wd2_apply, gap_1x1, cd]

/-- The exact (all-`t`) part: the output difference is `cd(maxpool …)` times the three
    `istd`s of the downsamples. -/
theorem gd_exact (t : ℝ) :
    liveFwd2S (Y + t • V) 0 - liveFwd2S (Y + t • V) 1
      = cd (maxPoolFlat 2 8 8 (stemS (Y + t • V)))
        * (bnIstd (2 * 4 * 4) (decimateFlat 2 4 4 (maxPoolFlat 2 8 8 (stemS (Y + t • V)))) 1
           * (bnIstd (2 * 2 * 2) (decimateFlat 2 2 2 (ldS 4 4 (maxPoolFlat 2 8 8 (stemS (Y + t • V))))) 1
              * bnIstd (2 * 1 * 1)
                  (decimateFlat 2 1 1 (ldS 2 2 (ldS 4 4 (maxPoolFlat 2 8 8 (stemS (Y + t • V)))))) 1)) := by
  rw [liveFwd2S_diff, cd_ldS 1 1, cd_ldS 2 2, cd_ldS 4 4]; ring

/-- **The output difference along the ray is `t · Rr t`** (eventually near `0`): the
    maxpool selection lets the BN channel-diff identity collapse the whole chain. -/
theorem gd_ray_eventually :
    (fun t => liveFwd2S (Y + t • V) 0 - liveFwd2S (Y + t • V) 1) =ᶠ[𝓝 0] (fun t => t * Rr t) := by
  filter_upwards [maxpool_sel_eventually] with t hsel
  rw [gd_exact t, cd_maxpool_eq _ hsel, cd_stemS, cd_ray]
  simp only [Rr, Pr, W4r, W2r]; ring

/-- **The directional derivative of the output difference at the base is `Rr 0 ≠ 0`.**
    The factor `t` makes every `istd`-derivative cross-term vanish at `0`. -/
theorem gd_hasDerivAt :
    HasDerivAt (fun t : ℝ => liveFwd2S (Y + t • V) 0 - liveFwd2S (Y + t • V) 1) (Rr 0) 0 := by
  have hmul : HasDerivAt (fun t : ℝ => t * Rr t) (Rr 0) 0 := by
    rw [hasDerivAt_iff_tendsto_slope]
    have hslope : slope (fun t : ℝ => t * Rr t) 0 =ᶠ[𝓝[≠] (0 : ℝ)] Rr := by
      filter_upwards [self_mem_nhdsWithin] with y hy
      have hy0 : y ≠ 0 := by simpa using hy
      simp only [slope_def_field, sub_zero, zero_mul]
      rw [mul_comm, mul_div_assoc, div_self hy0, mul_one]
    exact Filter.Tendsto.congr' hslope.symm
      (Rr_continuous.continuousAt.tendsto.mono_left nhdsWithin_le_nhds)
  exact hmul.congr_of_eventuallyEq gd_ray_eventually

-- ════════════════════════════════════════════════════════════════
-- § The whole-net VJP at the base `Y` (maxpool no-tie via per-channel injectivity)
-- ════════════════════════════════════════════════════════════════

/-- The decimated base value at a general position `(ci, r, s)`. -/
theorem decimY_val (ci : Fin 2) (r s : Fin 16) :
    decimateFlat 2 16 16 Y (finProdFinEquiv (finProdFinEquiv (ci, r), s))
      = -(64 * (r.val : ℝ) + 2 * (s.val : ℝ)) := by
  have e : decimateFlat 2 16 16 Y (finProdFinEquiv (finProdFinEquiv (ci, r), s))
      = (Tensor3.unflatten (decimateFlat 2 16 16 Y) : Tensor3 2 16 16) ci r s := rfl
  rw [e, decimate_unflatten Y ci r s, unflatten_Y]
  simp only [Ytensor]
  push_cast
  ring

/-- **The base stem is per-channel positionally injective** ⇒ the maxpool has no ties. -/
theorem stem2_Y_maxpool_smooth :
    MaxPool2Smooth (Tensor3.unflatten (stem2 Y) : Tensor3 2 (2 * 8) (2 * 8)) := by
  apply maxPool2Smooth_of_injective
  intro ci r r' s s' heq
  rw [stem2_eq_stemS Y] at heq
  have hdec := bnForward_coord_inj (n := 2 * 16 * 16) 1 30 one_pos (decimateFlat 2 16 16 Y)
    (finProdFinEquiv (finProdFinEquiv (ci, r), s))
    (finProdFinEquiv (finProdFinEquiv (ci, r'), s')) heq
  rw [decimY_val, decimY_val] at hdec
  have hnat : 64 * r.val + 2 * s.val = 64 * r'.val + 2 * s'.val := by
    have h := neg_injective hdec
    have := r.isLt; have := s.isLt; have := r'.isLt; have := s'.isLt
    exact_mod_cast h
  have := r.isLt; have := s.isLt; have := r'.isLt; have := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

noncomputable def stem2_vjp_Y : HasVJPAt stem2 Y :=
  convBnReluStrided_has_vjp_at WsId2 Zb2 1 1 30 (by norm_num) Y
    (fun k => ne_of_gt (stemS_bn_pos Y k))

theorem stem2_diff_Y : DifferentiableAt ℝ stem2 Y :=
  DifferentiableAt.comp Y
    (relu_differentiableAt_of_smooth (2 * 16 * 16) _ (fun k => ne_of_gt (stemS_bn_pos Y k)))
    ((convBnStrided_differentiable WsId2 Zb2 1 1 30 (by norm_num)) Y)

theorem mp_point_eq_Y :
    Tensor3.flatten (Tensor3.unflatten (stem2 Y) : Tensor3 2 (2 * 8) (2 * 8)) = stem2 Y :=
  Tensor3.flatten_unflatten (stem2 Y)

noncomputable def hmp_vjp_Y : HasVJPAt (maxPoolFlat 2 8 8) (stem2 Y) := by
  have h := maxPoolFlat_has_vjp_at (Tensor3.unflatten (stem2 Y) : Tensor3 2 (2 * 8) (2 * 8))
    stem2_Y_maxpool_smooth
  rwa [mp_point_eq_Y] at h

theorem hmp_diff_Y : DifferentiableAt ℝ (maxPoolFlat 2 8 8) (stem2 Y) := by
  have h := maxPoolFlat_differentiableAt (Tensor3.unflatten (stem2 Y) : Tensor3 2 (2 * 8) (2 * 8))
    stem2_Y_maxpool_smooth (by norm_num) (by norm_num) (by norm_num)
  rwa [mp_point_eq_Y] at h

/-- **The whole 2-channel live ResNet-34 VJP at the base `Y`** — the witness point of
    the level-3 seal (the maxpool no-tie holds because `Y` is per-channel injective). -/
noncomputable def liveFwd2_has_vjp_at_Y : HasVJPAt liveFwd2 Y :=
  resnet34_has_vjp_at stem2 (maxPoolFlat 2 8 8)
    ([] : List (Vec (2 * 8 * 8) → Vec (2 * 8 * 8))) (liveDownPC 4 4)
    ([] : List (Vec (2 * 4 * 4) → Vec (2 * 4 * 4))) (liveDownPC 2 2)
    ([] : List (Vec (2 * 2 * 2) → Vec (2 * 2 * 2))) (liveDownPC 1 1)
    ([] : List (Vec (2 * 1 * 1) → Vec (2 * 1 * 1)))
    (globalAvgPoolFlat 2 1 1) (dense Wd2 bd2) Y
    ⟨stem2_vjp_Y, stem2_diff_Y⟩
    ⟨hmp_vjp_Y, hmp_diff_Y⟩
    PUnit.unit
    ⟨liveDownPC_vjp 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _,
     liveDownPC_diff 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _⟩
    PUnit.unit
    ⟨liveDownPC_vjp 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _,
     liveDownPC_diff 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _⟩
    PUnit.unit
    ⟨liveDownPC_vjp 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _,
     liveDownPC_diff 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _⟩
    PUnit.unit
    ⟨(globalAvgPoolFlat_has_vjp 2 1 1).toHasVJPAt _, (globalAvgPoolFlat_differentiable 2 1 1) _⟩
    ⟨(dense_has_vjp Wd2 bd2).toHasVJPAt _, (dense_differentiable Wd2 bd2) _⟩

theorem liveFwd2_diff_Y : DifferentiableAt ℝ liveFwd2 Y := by
  show DifferentiableAt ℝ (dense Wd2 bd2 ∘ globalAvgPoolFlat 2 1 1 ∘
    liveDownPC 1 1 ∘ liveDownPC 2 2 ∘ liveDownPC 4 4 ∘ maxPoolFlat 2 8 8 ∘ stem2) Y
  exact (dense_differentiable Wd2 bd2).differentiableAt.comp Y
    ((globalAvgPoolFlat_differentiable 2 1 1).differentiableAt.comp Y
      ((liveDownPC_diff 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _).comp Y
        ((liveDownPC_diff 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _).comp Y
          ((liveDownPC_diff 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _).comp Y
            (hmp_diff_Y.comp Y stem2_diff_Y)))))

-- ════════════════════════════════════════════════════════════════
-- § The seal
-- ════════════════════════════════════════════════════════════════

/-- **`fderiv ℝ liveFwd2 Y ≠ 0`** — the live ResNet-34's whole-net Jacobian is genuinely
    non-trivial at the witness base `Y` (level-3 seal). -/
theorem liveFwd2_jacobian_nonzero : fderiv ℝ liveFwd2 Y ≠ 0 := by
  intro hzero
  have hfd : HasFDerivAt liveFwd2 (0 : Vec (2 * (2 * 16) * (2 * 16)) →L[ℝ] Vec 2) Y := by
    rw [← hzero]; exact liveFwd2_diff_Y.hasFDerivAt
  have hsmul : HasDerivAt (fun t : ℝ => Y + t • V) V 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const V).const_add Y
  have hcomp : HasDerivAt (fun t : ℝ => liveFwd2 (Y + t • V)) (0 : Vec 2) 0 := by
    have := HasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hfd hsmul (by simp)
    simpa using this
  have hpi := hasDerivAt_pi.mp hcomp
  have hd : HasDerivAt (fun t : ℝ => liveFwd2 (Y + t • V) 0 - liveFwd2 (Y + t • V) 1) 0 0 := by
    have := (hpi 0).sub (hpi 1); simpa using this
  rw [liveFwd2_eq_S] at hd
  exact (Rr_pos 0).ne' (gd_hasDerivAt.unique hd)

/-- **The level-3 seal for the live ResNet-34** (Item A): the proven whole-network
    backward of the nonzero-weight live ResNet-34 is **not the zero map** at the witness
    base `Y` — some basis-cotangent probe returns a nonzero row. Strictly stronger than
    `liveFwd2_nonconstant` (level 2): a non-constant forward could still have a zero
    Jacobian at the witness; this rules that out. -/
theorem liveFwd2_backward_nontrivial :
    ∃ (j₀ : Fin 2) (i₀ : Fin (2 * (2 * 16) * (2 * 16))),
      liveFwd2_has_vjp_at_Y.backward (basisVec j₀) i₀ ≠ 0 :=
  liveFwd2_has_vjp_at_Y.backward_nontrivial_of_fderiv_ne liveFwd2_jacobian_nonzero

end ResNet34LiveSeal
end Proofs
