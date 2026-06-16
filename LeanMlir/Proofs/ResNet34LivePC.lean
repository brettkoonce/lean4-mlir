import LeanMlir.Proofs.ResNet34Live2

/-!
# Toward a live ResNet-34 witness — Stage 3: the 2-channel layer rebuild (A1, WIP)

Stage 2 (`ResNet34Live2.lean`) banked the channel-order invariant kit (the A2
non-vacuity crux). This file does the **mechanical 2-channel re-instantiation**
(`planning/whole_network_backward.md` Item A1): the layers `liveDown`/`stem`/…
that Stage 1 hardcoded at `1 * h * w` rebuilt at `c = 2`, reusing the
channel-generic VJP/differentiability machinery (`rblkPStrided_has_vjp_at`,
`convBnStrided_differentiable`, …) at `oc = ic = 2`.

**This file (Stage 3):**
- `liveDownPC` — the 2-channel signal-carrying strided downsample (channel-diagonal
  identity-decimate projection `WsP2`, zeroed body), with its whole VJP / `DifferentiableAt`
  / nonnegativity, mirroring Stage 1's `liveDown` at `oc = ic = 2`.
- `stem2` — the 2-channel stem (channel-diagonal identity stem conv ⇒ `flatConvStride2 = decimate`,
  so the maxpool no-tie reuses Stage-1's global `bnForward_injective` pattern; `β = 30 > √512`).
- **`liveFwd2_has_vjp_at`** — the **whole 2-channel ResNet-34 backward**: stem + maxpool + three
  `liveDownPC` downsamples + per-channel GAP + identity head (empty identity-block chains — a first
  non-degenerate witness; full depth is Item D), with *every* smoothness/no-tie hypothesis of the
  dimension-generic `resnet34_has_vjp_at` discharged. The entire **smoothness side** of the live
  witness, at 2 channels.
- `bnForward_coord_inj` — scalar BN injective per coordinate (a reusable no-tie ingredient).

Build-checked, 3-axiom clean. **Remaining for the level-2 witness:** the non-vacuity
`liveFwd2 X2 ≠ liveFwd2 0` — thread the Stage-2 channel-order invariant (input `X2 i = i` has
one channel dominating; input `0` is symmetric) through the assembly to the per-channel head.
-/

namespace Proofs
namespace ResNet34LivePC

open Proofs ResNet34Live2

-- ════════════════════════════════════════════════════════════════
-- § 2-channel kernels
-- ════════════════════════════════════════════════════════════════

/-- Channel-diagonal 1×1 identity kernel (`δ_oi`): the signal-carrying projection,
    so `flatConvStride2 WsP2` decimates each channel independently. -/
noncomputable def WsP2 : Kernel4 2 2 1 1 := fun o i _ _ => if o = i then 1 else 0
/-- Zero 2-channel kernel (the residual bodies). -/
noncomputable def Zk2 : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def Zb2 : Vec 2 := fun _ => 0

-- ════════════════════════════════════════════════════════════════
-- § The 2-channel live downsample
-- ════════════════════════════════════════════════════════════════

/-- A live 2-channel downsample: `relu(bn₂₀(decimate x) + 1)` — projection carries the
    per-channel signal (channel-diagonal identity kernel `WsP2`), body zeroed to constant 1,
    `βp = 20 > √(2·h·w)` keeps the projection positive (`bnForward_lb`). The `c = 2` peer of
    `ResNet34Live.liveDown`. -/
noncomputable def liveDownPC (h w : Nat) : Vec (2 * (2 * h) * (2 * w)) → Vec (2 * h * w) :=
  relu (2 * h * w) ∘ residualProj
    (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2))

/-- The projection is strictly positive: `bn₂₀ ≥ 20 − √(2·h·w) > 0`. -/
theorem liveDownPC_proj_pos (h w : Nat)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w)))
    (k : Fin (2 * h * w)) :
    0 < (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2) a k := by
  simp only [Function.comp_apply]
  have hlb := bnForward_lb (n := 2 * h * w) 1 1 20 (by norm_num) (flatConvStride2 WsP2 Zb2 a) k
  rw [abs_one, one_mul] at hlb
  linarith

/-- The residual body collapses to the constant `1` (zeroed convs, body bn `β = 1`). -/
theorem liveDownPC_body_const (h w : Nat) (hhw : 0 < 2 * h * w)
    (a : Vec (2 * (2 * h) * (2 * w))) :
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a
      = (fun _ => (1 : ℝ)) := by
  simp only [Function.comp_apply]
  rw [flatConv_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]

/-- The post-add ReLU input is strictly positive (`proj > 0`, body `= 1`). -/
theorem liveDownPC_sum_pos (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w)))
    (k : Fin (2 * h * w)) :
    0 < ((bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2) a k)
      + ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
          (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k := by
  have hp := liveDownPC_proj_pos h w hn a k
  have hb : ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k = 1 := by
    rw [liveDownPC_body_const h w hhw a]
  rw [hb]; linarith

/-- VJP of the live 2-channel downsample (via `rblkPStrided_has_vjp_at` at `oc = ic = 2`). -/
noncomputable def liveDownPC_vjp (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) :
    HasVJPAt (liveDownPC h w) a :=
  rblkPStrided_has_vjp_at Zk2 Zb2 Zk2 Zb2 WsP2 Zb2 1 0 1 1 0 1 1 1 20
    (by norm_num) (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConvStride2_eq_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
      change (1 : ℝ) ≠ 0; norm_num)
    (fun k => ne_of_gt (liveDownPC_sum_pos h w hhw hn a k))

/-- The live 2-channel downsample is differentiable at every point. -/
theorem liveDownPC_diff (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) :
    DifferentiableAt ℝ (liveDownPC h w) a := by
  have hsm₁ : ∀ k, bnForward (2 * h * w) 1 0 1 (flatConvStride2 Zk2 Zb2 a) k ≠ 0 := fun k => by
    rw [flatConvStride2_eq_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
    change (1 : ℝ) ≠ 0; norm_num
  have hproj_diff : DifferentiableAt ℝ (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2) a :=
    convBnStrided_differentiable (h := h) (w := w) WsP2 Zb2 1 1 20 (by norm_num) a
  have hF_diff : DifferentiableAt ℝ
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a :=
    resblock_bodyStrided_differentiableAt (h := h) (w := w) Zk2 Zb2 Zk2 Zb2 1 0 1 1 0 1
      (by norm_num) (by norm_num) a hsm₁
  have hsm_res : ∀ k, residualProj
      (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2)
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k ≠ 0 := fun k =>
    ne_of_gt (liveDownPC_sum_pos h w hhw hn a k)
  show DifferentiableAt ℝ (relu (2 * h * w) ∘ residualProj
    (bnForward (2 * h * w) 1 1 20 ∘ flatConvStride2 WsP2 Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2))) a
  exact (relu_differentiableAt_of_smooth (2 * h * w) _ hsm_res).comp a
    (DifferentiableAt.add hproj_diff hF_diff)

/-- The live 2-channel downsample output is nonnegative (it is a ReLU). -/
theorem liveDownPC_nonneg (h w : Nat) (a : Vec (2 * (2 * h) * (2 * w))) (k : Fin (2 * h * w)) :
    0 ≤ liveDownPC h w a k := relu_nonneg (2 * h * w) _ k

-- ════════════════════════════════════════════════════════════════
-- § Per-coordinate BN injectivity (for the 2-channel stem's per-channel no-tie)
-- ════════════════════════════════════════════════════════════════

/-- **Scalar BN (γ=1) is injective per coordinate.** `bn k₀ = bn k₁ ⇒ z k₀ = z k₁`
    — the contrapositive of `bnForward_chan_lt` (strict order). A reusable no-tie
    ingredient for the 2-channel stem's maxpool (whether the stem is shown globally
    injective via a large channel-weight gap, or only per-channel). -/
theorem bnForward_coord_inj {n : Nat} (ε β : ℝ) (hε : 0 < ε) (z : Vec n) (k₀ k₁ : Fin n)
    (h : bnForward n ε 1 β z k₀ = bnForward n ε 1 β z k₁) :
    z k₀ = z k₁ := by
  rcases lt_trichotomy (z k₀) (z k₁) with hlt | heq | hgt
  · exact absurd h (ne_of_lt (bnForward_chan_lt ε β hε z k₁ k₀ hlt))
  · exact heq
  · exact absurd h.symm (ne_of_lt (bnForward_chan_lt ε β hε z k₀ k₁ hgt))

-- ════════════════════════════════════════════════════════════════
-- § The 2-channel stem (maxpool no-tie via global injectivity)
--   Channel-diagonal identity stem conv ⇒ `flatConvStride2 = decimate`, so the
--   maxpool no-tie reuses Stage-1's `bnForward_injective` pattern verbatim at 2
--   channels. Asymmetry is carried by the input, not the conv (`β = 30 > √512`).
-- ════════════════════════════════════════════════════════════════

/-- Channel-diagonal identity stem kernel (1→… is 2→2 here). -/
noncomputable def WsId2 : Kernel4 2 2 1 1 := fun o i _ _ => if o = i then 1 else 0
/-- Positional (hence injective) 2-channel input `X₂ i = i`. -/
noncomputable def X2 : Vec (2 * (2 * 16) * (2 * 16)) := fun i => (i.val : ℝ)

theorem X2_inj : Function.Injective X2 := by
  intro a b hab; simp only [X2] at hab; exact Fin.ext (by exact_mod_cast hab)

/-- The 2-channel diagonal-identity 1×1 conv is the identity (2-channel, 32×32 peer of
    Stage-1 `flatConv_id_X`). -/
theorem flatConv_WsId2_X2 : flatConv (h := 2 * 16) (w := 2 * 16) WsId2 Zb2 X2 = X2 := by
  have hc : conv2d WsId2 Zb2 (Tensor3.unflatten X2) = Tensor3.unflatten X2 := by
    funext o hi wi
    rw [conv2d_1x1]
    simp only [Zb2, WsId2, ite_mul, one_mul, zero_mul, zero_add]
    rw [Finset.sum_ite_eq Finset.univ o (fun c => (Tensor3.unflatten X2) c hi wi)]
    simp [Finset.mem_univ]
  simp only [flatConv, hc, Tensor3.flatten_unflatten]

/-- The 2-channel stem `relu ∘ bn ∘ conv_stride2` (β=30, 2ch, 16×16 output). -/
noncomputable def stem2 : Vec (2 * (2 * 16) * (2 * 16)) → Vec (2 * 16 * 16) :=
  relu (2 * 16 * 16) ∘ bnForward (2 * 16 * 16) 1 1 30 ∘ flatConvStride2 WsId2 Zb2

/-- Stride-2 diagonal-identity conv = decimation. -/
theorem stem2_conv_eq : flatConvStride2 WsId2 Zb2 X2 = decimateFlat 2 16 16 X2 := by
  unfold flatConvStride2; simp only [Function.comp_apply]; rw [flatConv_WsId2_X2]

theorem sqrt512_lt_30 : Real.sqrt ((2 * 16 * 16 : ℕ) : ℝ) < 30 := by
  rw [show ((2 * 16 * 16 : ℕ) : ℝ) = 512 by norm_num,
      show (30 : ℝ) = Real.sqrt 900 by
        rw [show (900 : ℝ) = 30 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
  exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)

/-- The stem's BN output is strictly positive: `bn ≥ 30 − √512 > 0`. -/
theorem stem2_bn_pos : ∀ k, 0 < bnForward (2 * 16 * 16) 1 1 30 (flatConvStride2 WsId2 Zb2 X2) k := by
  intro k
  have hlb := bnForward_lb (n := 2 * 16 * 16) 1 1 30 (by norm_num) (flatConvStride2 WsId2 Zb2 X2) k
  rw [abs_one, one_mul] at hlb
  linarith [sqrt512_lt_30]

/-- The stem output is injective: `bn` of the injective decimated input is injective,
    ReLU is the identity (positive). Stage-1's `stem_inj` at 2 channels. -/
theorem stem2_inj : Function.Injective (stem2 X2) := by
  have hstemeq : stem2 X2 = bnForward (2 * 16 * 16) 1 1 30 (flatConvStride2 WsId2 Zb2 X2) := by
    show (relu (2 * 16 * 16) ∘ bnForward (2 * 16 * 16) 1 1 30 ∘ flatConvStride2 WsId2 Zb2) X2 = _
    simp only [Function.comp_apply]
    exact relu_id_of_pos stem2_bn_pos
  rw [hstemeq, stem2_conv_eq]
  exact bnForward_injective 1 1 30 (by norm_num) (by norm_num)
    (decimateFlat_injective 2 16 16 X2_inj)

/-- The maxpool input (`= stem₂ X₂`) is positionally injective ⇒ `MaxPool2Smooth`
    (the 2-channel no-tie, reduced to global injectivity). -/
theorem stem2_maxpool_smooth :
    MaxPool2Smooth (Tensor3.unflatten (stem2 X2) : Tensor3 2 (2 * 8) (2 * 8)) := by
  apply maxPool2Smooth_of_injective
  intro ci r r' s s' heq
  simp only [Tensor3.unflatten] at heq
  have h2 := finProdFinEquiv.injective (stem2_inj heq)
  have h5 := finProdFinEquiv.injective (congrArg Prod.fst h2)
  exact ⟨congrArg Prod.snd h5, congrArg Prod.snd h2⟩

-- ════════════════════════════════════════════════════════════════
-- § The whole 2-channel live ResNet-34 + its VJP (empty identity chains)
-- ════════════════════════════════════════════════════════════════

theorem sqrt_lt_20 {n : ℕ} (h : (n : ℝ) < 400) : Real.sqrt (n : ℝ) < 20 := by
  rw [show (20 : ℝ) = Real.sqrt 400 by
    rw [show (400 : ℝ) = 20 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
  exact Real.sqrt_lt_sqrt (by positivity) h

/-- Identity dense head: reads each channel's GAP — `dense Wd2 bd2 u = u`. -/
noncomputable def Wd2 : Mat 2 2 := fun i j => if i = j then 1 else 0
noncomputable def bd2 : Vec 2 := fun _ => 0

noncomputable def stem2_vjp : HasVJPAt stem2 X2 :=
  convBnReluStrided_has_vjp_at WsId2 Zb2 1 1 30 (by norm_num) X2
    (fun k => ne_of_gt (stem2_bn_pos k))

theorem stem2_diff : DifferentiableAt ℝ stem2 X2 :=
  DifferentiableAt.comp X2
    (relu_differentiableAt_of_smooth (2 * 16 * 16) _ (fun k => ne_of_gt (stem2_bn_pos k)))
    ((convBnStrided_differentiable WsId2 Zb2 1 1 30 (by norm_num)) X2)

theorem mp_point_eq2 :
    Tensor3.flatten (Tensor3.unflatten (stem2 X2) : Tensor3 2 (2 * 8) (2 * 8)) = stem2 X2 :=
  Tensor3.flatten_unflatten (stem2 X2)

noncomputable def hmp_vjp2 : HasVJPAt (maxPoolFlat 2 8 8) (stem2 X2) := by
  have h := maxPoolFlat_has_vjp_at (Tensor3.unflatten (stem2 X2) : Tensor3 2 (2 * 8) (2 * 8))
    stem2_maxpool_smooth
  rwa [mp_point_eq2] at h

theorem hmp_diff2 : DifferentiableAt ℝ (maxPoolFlat 2 8 8) (stem2 X2) := by
  have h := maxPoolFlat_differentiableAt (Tensor3.unflatten (stem2 X2) : Tensor3 2 (2 * 8) (2 * 8))
    stem2_maxpool_smooth (by norm_num) (by norm_num) (by norm_num)
  rwa [mp_point_eq2] at h

/-- **The 2-channel live ResNet-34 forward** — strided stem + maxpool + three
    signal-carrying `liveDownPC` downsamples + per-channel GAP + identity head, with
    empty identity-block chains (a first non-degenerate witness; full depth = Item D). -/
noncomputable def liveFwd2 : Vec (2 * (2 * 16) * (2 * 16)) → Vec 2 :=
  dense Wd2 bd2 ∘ globalAvgPoolFlat 2 1 1 ∘
    liveDownPC 1 1 ∘ liveDownPC 2 2 ∘ liveDownPC 4 4 ∘ maxPoolFlat 2 8 8 ∘ stem2

/-- **Whole-network VJP for the 2-channel live ResNet-34** — every smoothness/no-tie
    hypothesis discharged at `oc = ic = 2`. -/
noncomputable def liveFwd2_has_vjp_at : HasVJPAt liveFwd2 X2 :=
  resnet34_has_vjp_at stem2 (maxPoolFlat 2 8 8)
    ([] : List (Vec (2 * 8 * 8) → Vec (2 * 8 * 8))) (liveDownPC 4 4)
    ([] : List (Vec (2 * 4 * 4) → Vec (2 * 4 * 4))) (liveDownPC 2 2)
    ([] : List (Vec (2 * 2 * 2) → Vec (2 * 2 * 2))) (liveDownPC 1 1)
    ([] : List (Vec (2 * 1 * 1) → Vec (2 * 1 * 1)))
    (globalAvgPoolFlat 2 1 1) (dense Wd2 bd2) X2
    ⟨stem2_vjp, stem2_diff⟩
    ⟨hmp_vjp2, hmp_diff2⟩
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

theorem liveFwd2_has_vjp_correct (dy : Vec 2) (i : Fin (2 * (2 * 16) * (2 * 16))) :
    liveFwd2_has_vjp_at.backward dy i = ∑ j : Fin 2, pdiv liveFwd2 X2 i j * dy j :=
  liveFwd2_has_vjp_at.correct dy i

end ResNet34LivePC
end Proofs
