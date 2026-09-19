import LeanMlir.Proofs.Nets.ResNet.ResNet34LivePC

/-!
# Realistic-dimension live ResNet-34 witness (`planning/archive/whole_network_backward.md` Item D)

`ResNet34LivePC` builds the 2-channel non-degenerate ResNet-34 whole-net backward
witness at a **32×32** toy input. This file lifts it to the real **ImageNet spatial
resolution, 224×224** — the genuine ResNet-34 spatial pyramid (5 halvings:
`224 →stem/s2 112 →maxpool/s2 56 →down 28 →down 14 →down 7 →GAP`), exactly the
`conv1 → maxpool → layer2 → layer3 → layer4` downsampling skeleton.

The only thing that scales with the spatial size is the **BN positivity bound**: the
projection/stem ReLUs are kept off by `bnForward_lb : β − |γ|·√n ≤ bn`, so a constant
`β` no longer suffices — `β` must dominate `√n`. So the construction is made
**β-parametric** (`liveDownβ`), threading `√(2hw) < βp`, and instantiated at a `β`
large enough for the realistic `n`: `β = 64 > √1568` for the downsamples (largest BN
length `2·28·28`) and `β = 160 > √25088` for the stem (`2·112·112`). Everything else
— the channel-diagonal decimating convs, the `Dom2` channel-order carrier, the whole-
net VJP fold (`resnet34_has_vjp_at`) — is already dimension-generic and reused verbatim.
(The 2-channel carrier is inherent to the seal; channel-width realism is orthogonal.)
-/

namespace Proofs
namespace ResNet34LiveRealistic

open Proofs ResNet34Live2 ResNet34LivePC

-- ════════════════════════════════════════════════════════════════
-- § A √n-vs-β helper (the positivity bound, parametric)
-- ════════════════════════════════════════════════════════════════

theorem sqrt_lt_param (n : ℕ) (β : ℝ) (hβ : 0 ≤ β) (h : (n : ℝ) < β ^ 2) :
    Real.sqrt (n : ℝ) < β :=
  (Real.sqrt_lt n.cast_nonneg hβ).2 h

-- ════════════════════════════════════════════════════════════════
-- § GAP over a >1×1 spatial extent (the real net GAPs over 7×7)
-- ════════════════════════════════════════════════════════════════

/-- **GAP preserves the strict channel order** when one channel dominates the other at
    every spatial position (the mean of strictly larger values is strictly larger). -/
theorem gap_chan_lt {h w : Nat} (hh : 0 < h) (hw : 0 < w) (v : Vec (2 * h * w)) (hv : Dom2 v) :
    globalAvgPoolFlat 2 h w v 0 < globalAvgPoolFlat 2 h w v 1 := by
  rw [globalAvgPoolFlat_as_sum]
  have hfac : (0 : ℝ) < 1 / (h * w : ℝ) := by
    apply div_pos one_pos; exact_mod_cast Nat.mul_pos hh hw
  apply Finset.sum_lt_sum_of_nonempty
  · exact Finset.univ_nonempty_iff.mpr ⟨(⟨0, hh⟩, ⟨0, hw⟩)⟩
  · intro p _
    apply mul_lt_mul_of_pos_left _ hfac
    have := hv p.1 p.2
    simpa only [Tensor3.unflatten] using this

/-- GAP of a constant vector is that constant (over any spatial extent). -/
theorem gap_const_g {h w : Nat} (hh : 0 < h) (hw : 0 < w) (c : ℝ) :
    globalAvgPoolFlat 2 h w (fun _ => c) = fun _ => c := by
  funext j
  have hpos : (0 : ℝ) < (h : ℝ) * w := by exact_mod_cast Nat.mul_pos hh hw
  simp only [globalAvgPoolFlat_as_sum, Finset.sum_const, Finset.card_univ, Fintype.card_prod,
    Fintype.card_fin, nsmul_eq_mul]
  push_cast
  field_simp

-- ════════════════════════════════════════════════════════════════
-- § The realistic stem (112×112, β=160) and 224×224 input
-- ════════════════════════════════════════════════════════════════

/-- Positional 2-channel 224×224 input `X i = i`. -/
noncomputable def X224 : Vec (2 * (2 * 112) * (2 * 112)) := fun i => (i.val : ℝ)

theorem X224_inj : Function.Injective X224 := by
  intro a b hab; simp only [X224] at hab; exact Fin.ext (by exact_mod_cast hab)

/-- The realistic stem: `relu ∘ bn(1,1,160) ∘ conv_stride2(diag-id)` at 112×112. -/
noncomputable def stem224 : Vec (2 * (2 * 112) * (2 * 112)) → Vec (2 * 112 * 112) :=
  relu (2 * 112 * 112) ∘ bnForward (2 * 112 * 112) 1 1 160 ∘ flatConvStride2 WsId2 Zb2

theorem stem224_conv_eq : flatConvStride2 WsId2 Zb2 X224 = decimateFlat 2 112 112 X224 := by
  rw [flatConvStride2_diag WsId2 (fun o i => rfl) X224]

theorem sqrt25088_lt_160 : Real.sqrt ((2 * 112 * 112 : ℕ) : ℝ) < 160 :=
  sqrt_lt_param (2 * 112 * 112) 160 (by norm_num) (by norm_num)

/-- The stem's BN output is strictly positive: `bn ≥ 160 − √25088 > 0`. -/
theorem stem224_bn_pos : ∀ k, 0 < bnForward (2 * 112 * 112) 1 1 160 (flatConvStride2 WsId2 Zb2 X224) k := by
  intro k
  have hlb := bnForward_lb (n := 2 * 112 * 112) 1 1 160 (by norm_num) (flatConvStride2 WsId2 Zb2 X224) k
  rw [abs_one, one_mul] at hlb
  linarith [sqrt25088_lt_160]

theorem stem224_inj : Function.Injective (stem224 X224) := by
  have hstemeq : stem224 X224 = bnForward (2 * 112 * 112) 1 1 160 (flatConvStride2 WsId2 Zb2 X224) := by
    show (relu (2 * 112 * 112) ∘ bnForward (2 * 112 * 112) 1 1 160 ∘ flatConvStride2 WsId2 Zb2) X224 = _
    simp only [Function.comp_apply]
    exact relu_id_of_pos stem224_bn_pos
  rw [hstemeq, stem224_conv_eq]
  exact bnForward_injective 1 1 160 (by norm_num) (by norm_num)
    (decimateFlat_injective 2 112 112 X224_inj)

theorem stem224_maxpool_smooth :
    MaxPool2Smooth (Tensor3.unflatten (stem224 X224) : Tensor3 2 (2 * 56) (2 * 56)) := by
  apply maxPool2Smooth_of_injective
  intro ci r r' s s' heq
  simp only [Tensor3.unflatten] at heq
  have h2 := finProdFinEquiv.injective (stem224_inj heq)
  have h5 := finProdFinEquiv.injective (congrArg Prod.fst h2)
  exact ⟨congrArg Prod.snd h5, congrArg Prod.snd h2⟩

noncomputable def stem224_vjp : HasVJPAt stem224 X224 :=
  convBnReluStrided_has_vjp_at WsId2 Zb2 1 1 160 (by norm_num) X224
    (fun k => ne_of_gt (stem224_bn_pos k))

theorem stem224_diff : DifferentiableAt ℝ stem224 X224 :=
  DifferentiableAt.comp X224
    (relu_differentiableAt_of_smooth (2 * 112 * 112) _ (fun k => ne_of_gt (stem224_bn_pos k)))
    ((convBnStrided_differentiable WsId2 Zb2 1 1 160 (by norm_num)) X224)

theorem mp_point_eq224 :
    Tensor3.flatten (Tensor3.unflatten (stem224 X224) : Tensor3 2 (2 * 56) (2 * 56)) = stem224 X224 :=
  Tensor3.flatten_unflatten (stem224 X224)

noncomputable def hmp_vjp224 : HasVJPAt (maxPoolFlat 2 56 56) (stem224 X224) := by
  have h := maxPoolFlat_has_vjp_at (Tensor3.unflatten (stem224 X224) : Tensor3 2 (2 * 56) (2 * 56))
    stem224_maxpool_smooth
  rwa [mp_point_eq224] at h

theorem hmp_diff224 : DifferentiableAt ℝ (maxPoolFlat 2 56 56) (stem224 X224) := by
  have h := maxPoolFlat_differentiableAt (Tensor3.unflatten (stem224 X224) : Tensor3 2 (2 * 56) (2 * 56))
    stem224_maxpool_smooth (by norm_num) (by norm_num) (by norm_num)
  rwa [mp_point_eq224] at h

/-- The stem preserves `Dom2`. -/
theorem Dom2_stem224 (a : Vec (2 * (2 * 112) * (2 * 112))) (ha : Dom2 a)
    (hpos : ∀ k, 0 < bnForward (2 * 112 * 112) 1 1 160 (flatConvStride2 WsId2 Zb2 a) k) :
    Dom2 (stem224 a) := by
  show Dom2 (relu (2 * 112 * 112) (bnForward (2 * 112 * 112) 1 1 160 (flatConvStride2 WsId2 Zb2 a)))
  apply Dom2_relu _ hpos
  rw [flatConvStride2_diag WsId2 (fun o i => rfl) a]
  exact Dom2_bn 160 _ (Dom2_decimate a ha)

/-- The positional input `X224 i = i` has channel 1 dominating channel 0. -/
theorem Dom2_X224 : Dom2 (h := 2 * 112) (w := 2 * 112) X224 := by
  intro hi wi
  show X224 (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), hi), wi))
     < X224 (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), hi), wi))
  simp only [X224, finProdFinEquiv_apply_val, Fin.val_zero, Fin.val_one, mul_zero, mul_one]
  push_cast
  have : (0 : ℝ) < (2 * 112 : ℕ) := by positivity
  nlinarith [this]

-- ════════════════════════════════════════════════════════════════
-- § The whole realistic-dimension live ResNet-34 + its VJP
-- ════════════════════════════════════════════════════════════════

/-- **The 224×224 live ResNet-34 forward** — strided stem (112²) + maxpool (56²) + three
    `liveDownβ` downsamples (28² → 14² → 7², β=64) + per-channel GAP over 7×7 + identity
    head. Empty identity-block chains (full depth is the orthogonal `ResNet34LiveFull`
    extension); the real ImageNet spatial pyramid. -/
noncomputable def liveFwd224 : Vec (2 * (2 * 112) * (2 * 112)) → Vec 2 :=
  dense Wd2 bd2 ∘ globalAvgPoolFlat 2 7 7 ∘
    liveDownβ 7 7 64 ∘ liveDownβ 14 14 64 ∘ liveDownβ 28 28 64 ∘ maxPoolFlat 2 56 56 ∘ stem224

/-- **Whole-network VJP for the 224×224 live ResNet-34** — every smoothness/no-tie
    hypothesis discharged at realistic spatial dims (the BN positivity bounds `β > √n`
    hold by `sqrt_lt_param`). -/
noncomputable def liveFwd224_has_vjp_at : HasVJPAt liveFwd224 X224 :=
  resnet34_has_vjp_at stem224 (maxPoolFlat 2 56 56)
    ([] : List (Vec (2 * 56 * 56) → Vec (2 * 56 * 56))) (liveDownβ 28 28 64)
    ([] : List (Vec (2 * 28 * 28) → Vec (2 * 28 * 28))) (liveDownβ 14 14 64)
    ([] : List (Vec (2 * 14 * 14) → Vec (2 * 14 * 14))) (liveDownβ 7 7 64)
    ([] : List (Vec (2 * 7 * 7) → Vec (2 * 7 * 7)))
    (globalAvgPoolFlat 2 7 7) (dense Wd2 bd2) X224
    ⟨stem224_vjp, stem224_diff⟩
    ⟨hmp_vjp224, hmp_diff224⟩
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

/-- **Public correctness** — the 224×224 live ResNet-34 backward equals the `pdiv`-Jacobian. -/
theorem liveFwd224_has_vjp_correct (dy : Vec 2) (i : Fin (2 * (2 * 112) * (2 * 112))) :
    liveFwd224_has_vjp_at.backward dy i = ∑ j : Fin 2, pdiv liveFwd224 X224 i j * dy j :=
  liveFwd224_has_vjp_at.correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Non-vacuity (level 2): the realistic forward is non-constant
-- ════════════════════════════════════════════════════════════════

/-- `liveFwd224 X224` is channel-asymmetric: thread `Dom2` from the input through
    stem → maxpool → the three downsamples to the per-channel head over the 7×7 GAP. -/
theorem liveFwd224_X224_asym : liveFwd224 X224 0 < liveFwd224 X224 1 := by
  have d0 : Dom2 (stem224 X224) := Dom2_stem224 X224 Dom2_X224 stem224_bn_pos
  have d1 := Dom2_maxpool (h := 56) (w := 56) (stem224 X224) d0
  have d2 := Dom2_liveDownβ 28 28 64 (by norm_num)
    (sqrt_lt_param (2 * 28 * 28) 64 (by norm_num) (by norm_num)) _ d1
  have d3 := Dom2_liveDownβ 14 14 64 (by norm_num)
    (sqrt_lt_param (2 * 14 * 14) 64 (by norm_num) (by norm_num)) _ d2
  have d4 := Dom2_liveDownβ 7 7 64 (by norm_num)
    (sqrt_lt_param (2 * 7 * 7) 64 (by norm_num) (by norm_num)) _ d3
  simp only [liveFwd224, Function.comp_apply, dense_Wd2_apply]
  exact gap_chan_lt (by norm_num) (by norm_num) _ d4

theorem stem224_zero : stem224 (fun _ => (0 : ℝ)) = fun _ => (160 : ℝ) := by
  show (relu (2 * 112 * 112) ∘ bnForward (2 * 112 * 112) 1 1 160 ∘ flatConvStride2 WsId2 Zb2)
    (fun _ => 0) = _
  simp only [Function.comp_apply]
  rw [flatConvStride2_diag WsId2 (fun o i => rfl) (fun _ => 0), decimateFlat_const,
      bnForward_const_eq (by norm_num), ResNet34LivePC.relu_const_pos 160 (by norm_num)]

theorem liveFwd224_zero : liveFwd224 (fun _ => (0 : ℝ)) = fun _ => (65 : ℝ) := by
  simp only [liveFwd224, Function.comp_apply, stem224_zero, maxPoolFlat_const,
    liveDownβ_const 28 28 64 (by norm_num) (by norm_num),
    liveDownβ_const 14 14 64 (by norm_num) (by norm_num),
    liveDownβ_const 7 7 64 (by norm_num) (by norm_num),
    gap_const_g (h := 7) (w := 7) (by norm_num) (by norm_num), dense_Wd2_const]
  norm_num

/-- **The 224×224 live ResNet-34 is non-degenerate** (level 2): `liveFwd224 X224 ≠
    liveFwd224 0`. The first ResNet-34 whole-net backward witness at ImageNet spatial
    resolution — nonzero weights, real spatial pyramid, every smoothness/no-tie
    hypothesis discharged at `n` up to `2·112·112 = 25088`. -/
theorem liveFwd224_nonconstant : liveFwd224 X224 ≠ liveFwd224 (fun _ => (0 : ℝ)) := by
  intro h
  have h0 : liveFwd224 X224 0 = liveFwd224 X224 1 := by
    rw [h, liveFwd224_zero]
  exact absurd h0 (ne_of_lt liveFwd224_X224_asym)

end ResNet34LiveRealistic
end Proofs
