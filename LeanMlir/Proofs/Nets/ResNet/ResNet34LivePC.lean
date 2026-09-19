import LeanMlir.Proofs.Nets.ResNet.ResNet34Live2

/-!
# A live ResNet-34 witness — the 2-channel non-degenerate witness (A1 complete, level 2)

Stage 2 (`ResNet34Live2.lean`) banked the channel-order invariant kit (the A2
non-vacuity crux). This file does the **mechanical 2-channel re-instantiation**
(`planning/archive/whole_network_backward.md` Item A1): the layers `liveDown`/`stem`/…
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
- **`liveFwd2_nonconstant`** — **the non-vacuity**: `liveFwd2 X2 ≠ liveFwd2 0`. Threads the Stage-2
  channel-order invariant `Dom2` (channel 1 strictly dominates channel 0 at every position) from the
  positional input through stem → maxpool → the three downsamples to the per-channel head, so
  `liveFwd2 X2 0 < liveFwd2 X2 1`; the zero input collapses channel-symmetric (`liveFwd2 0 = const 21`).
- `bnForward_coord_inj` — scalar BN injective per coordinate (a reusable no-tie ingredient).

This is the **first non-degenerate ResNet-34 whole-net backward witness** (level 2): a real 34-layer
ResNet skeleton (strided stem + maxpool + three strided downsamples + GAP + dense), nonzero weights,
`forward X ≠ forward 0`, every smoothness/no-tie hypothesis discharged, 3-axiom clean. It retires the
"degenerate (constant-output) witness" caveat for ResNet-34. (Full identity-block depth = Item D; the
level-3 nonzero-Jacobian seal is a separate follow-up — the ReLUs/maxpool bind off-witness, so the
`Mnv2Live` input-0 global-smoothness trick does not transfer.)
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
-- § The 2-channel live downsample (any BN shift `βp`, any projection kernel `W`)
-- ════════════════════════════════════════════════════════════════

/-- A live 2-channel downsample: `relu(bn_βp(conv_W(decimate x)) + 1)` — the projection
    carries the signal, the body is zeroed to the constant 1, and `βp > √(2·h·w)` keeps the
    projection positive (`bnForward_lb`) whatever the kernel `W`. The `c = 2` peer of
    `ResNet34Live.liveDown`. -/
noncomputable def liveDownW (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) :
    Vec (2 * (2 * h) * (2 * w)) → Vec (2 * h * w) :=
  relu (2 * h * w) ∘ residualProj
    (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2))

/-- The signal-carrying downsample: the channel-diagonal identity projection `WsP2`
    decimates each channel independently. -/
noncomputable def liveDownβ (h w : Nat) (βp : ℝ) : Vec (2 * (2 * h) * (2 * w)) → Vec (2 * h * w) :=
  liveDownW h w βp WsP2

/-- The 32×32 witness's downsample: `βp = 20 > √(2·h·w)` at every stage. -/
noncomputable def liveDownPC (h w : Nat) : Vec (2 * (2 * h) * (2 * w)) → Vec (2 * h * w) :=
  liveDownβ h w 20

/-- **Weight-independent projection positivity.** For any kernel `W`, the normalized
    projection `bn_{γ=1,βp}(conv_W(·))` is `> 0` once `√n < βp` — `bnForward_lb` bounds it
    below by `βp − √n`, with no reference to `W`. -/
theorem liveDownW_proj_pos (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w)))
    (k : Fin (2 * h * w)) :
    0 < (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2) a k := by
  simp only [Function.comp_apply]
  have hlb := bnForward_lb (n := 2 * h * w) 1 1 βp (by norm_num) (flatConvStride2 W Zb2 a) k
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

/-- The post-add ReLU input is strictly positive (`proj > 0`, body `= 1`), for any `W`. -/
theorem liveDownW_sum_pos (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w)))
    (k : Fin (2 * h * w)) :
    0 < ((bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 W Zb2) a k)
      + ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
          (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a k := by
  rw [liveDownPC_body_const h w hhw a]
  linarith [liveDownW_proj_pos h w βp W hn a k]

/-- VJP of the live downsample at every point, for any `W` (via `rblkPStrided_has_vjp_at`
    at `oc = ic = 2`). -/
noncomputable def liveDownW_vjp (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    HasVJPAt (liveDownW h w βp W) a :=
  rblkPStrided_has_vjp_at Zk2 Zb2 Zk2 Zb2 W Zb2 1 0 1 1 0 1 1 1 βp
    (by norm_num) (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConvStride2_eq_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
      change (1 : ℝ) ≠ 0; norm_num)
    (fun k => ne_of_gt (liveDownW_sum_pos h w βp W hhw hn a k))

/-- The live downsample is differentiable at every point, for any `W`. -/
theorem liveDownW_diff (h w : Nat) (βp : ℝ) (W : Kernel4 2 2 1 1) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    DifferentiableAt ℝ (liveDownW h w βp W) a := by
  have hsm₁ : ∀ k, bnForward (2 * h * w) 1 0 1 (flatConvStride2 Zk2 Zb2 a) k ≠ 0 := fun k => by
    rw [flatConvStride2_eq_zero Zk2 Zb2 (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
    change (1 : ℝ) ≠ 0; norm_num
  have hsm := fun k => ne_of_gt (liveDownW_sum_pos h w βp W hhw hn a k)
  unfold liveDownW residualProj flatConvStride2 at *
  fun_prop (disch := first | assumption | norm_num)

noncomputable def liveDownβ_vjp (h w : Nat) (βp : ℝ) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    HasVJPAt (liveDownβ h w βp) a := liveDownW_vjp h w βp WsP2 hhw hn a

theorem liveDownβ_diff (h w : Nat) (βp : ℝ) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) :
    DifferentiableAt ℝ (liveDownβ h w βp) a := liveDownW_diff h w βp WsP2 hhw hn a

noncomputable def liveDownPC_vjp (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) :
    HasVJPAt (liveDownPC h w) a := liveDownβ_vjp h w 20 hhw hn a

theorem liveDownPC_diff (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) :
    DifferentiableAt ℝ (liveDownPC h w) a := liveDownβ_diff h w 20 hhw hn a

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
--   channels. Asymmetry is carried by the input, not the conv; the BN shift `β` must
--   dominate `√(2·s·s)` (`β = 30 > √512` at 16×16, `β = 160 > √25088` at 112×112).
-- ════════════════════════════════════════════════════════════════

/-- Channel-diagonal identity stem kernel (1→… is 2→2 here). -/
noncomputable def WsId2 : Kernel4 2 2 1 1 := fun o i _ _ => if o = i then 1 else 0

/-- A channel-diagonal 1×1 conv (`δ_oi`, zero bias) is the identity (generic dims). -/
theorem flatConv_diag_id {h w : Nat} (W : Kernel4 2 2 1 1) (b : Vec 2)
    (hW : ∀ o i, W o i 0 0 = if o = i then 1 else 0) (hb : ∀ o, b o = 0) (v : Vec (2 * h * w)) :
    flatConv (h := h) (w := w) W b v = v := by
  have hc : conv2d W b (Tensor3.unflatten v) = Tensor3.unflatten v := by
    funext o hi wi
    rw [conv2d_1x1, hb]
    simp only [hW, ite_mul, one_mul, zero_mul, zero_add]
    rw [Finset.sum_ite_eq Finset.univ o (fun c => (Tensor3.unflatten v) c hi wi)]
    simp [Finset.mem_univ]
  simp only [flatConv, hc, Tensor3.flatten_unflatten]

/-- The signal-carrying strided diagonal conv = decimation (generic dims). -/
theorem flatConvStride2_diag {h w : Nat} (W : Kernel4 2 2 1 1)
    (hW : ∀ o i, W o i 0 0 = if o = i then 1 else 0) (v : Vec (2 * (2 * h) * (2 * w))) :
    flatConvStride2 W Zb2 v = decimateFlat 2 h w v := by
  unfold flatConvStride2
  simp only [Function.comp_apply]
  rw [flatConv_diag_id (h := 2 * h) (w := 2 * w) W Zb2 hW (fun _ => rfl) v]

/-- Positional (hence injective) 2-channel input `X i = i` at `2s × 2s`. -/
noncomputable def Xs (s : Nat) : Vec (2 * (2 * s) * (2 * s)) := fun i => (i.val : ℝ)

theorem Xs_inj (s : Nat) : Function.Injective (Xs s) := by
  intro a b hab; simp only [Xs] at hab; exact Fin.ext (by exact_mod_cast hab)

/-- The 32×32 witness input. -/
noncomputable def X2 : Vec (2 * (2 * 16) * (2 * 16)) := Xs 16

/-- The 2-channel stem `relu ∘ bn_β ∘ conv_stride2` at `s × s` output. -/
noncomputable def stemβ (s : Nat) (β : ℝ) : Vec (2 * (2 * s) * (2 * s)) → Vec (2 * s * s) :=
  relu (2 * s * s) ∘ bnForward (2 * s * s) 1 1 β ∘ flatConvStride2 WsId2 Zb2

/-- The 32×32 witness stem (β=30, 16×16 output). -/
noncomputable def stem2 : Vec (2 * (2 * 16) * (2 * 16)) → Vec (2 * 16 * 16) := stemβ 16 30

theorem sqrt512_lt_30 : Real.sqrt ((2 * 16 * 16 : ℕ) : ℝ) < 30 :=
  (Real.sqrt_lt' (by norm_num)).2 (by norm_num)

/-- The stem's BN output is strictly positive at every input: `bn ≥ β − √(2·s·s) > 0`. -/
theorem stemβ_bn_pos (s : Nat) (β : ℝ) (hn : Real.sqrt ((2 * s * s : ℕ) : ℝ) < β)
    (v : Vec (2 * (2 * s) * (2 * s))) (k : Fin (2 * s * s)) :
    0 < bnForward (2 * s * s) 1 1 β (flatConvStride2 WsId2 Zb2 v) k := by
  have hlb := bnForward_lb (n := 2 * s * s) 1 1 β (by norm_num) (flatConvStride2 WsId2 Zb2 v) k
  rw [abs_one, one_mul] at hlb
  linarith

/-- The stem output at the positional input is injective: `bn` of the injective decimated
    input is injective, ReLU is the identity (positive). Stage-1's `stem_inj` at 2 channels. -/
theorem stemβ_inj (s : Nat) (β : ℝ) (hn : Real.sqrt ((2 * s * s : ℕ) : ℝ) < β) :
    Function.Injective (stemβ s β (Xs s)) := by
  have hstemeq : stemβ s β (Xs s) = bnForward (2 * s * s) 1 1 β (flatConvStride2 WsId2 Zb2 (Xs s)) := by
    show (relu (2 * s * s) ∘ bnForward (2 * s * s) 1 1 β ∘ flatConvStride2 WsId2 Zb2) (Xs s) = _
    simp only [Function.comp_apply]
    exact relu_id_of_pos (stemβ_bn_pos s β hn (Xs s))
  rw [hstemeq, flatConvStride2_diag WsId2 (fun o i => rfl) (Xs s)]
  exact bnForward_injective 1 1 β (by norm_num) (by norm_num)
    (decimateFlat_injective 2 s s (Xs_inj s))

/-- The maxpool input (`= stem X`) is positionally injective ⇒ `MaxPool2Smooth`
    (the 2-channel no-tie, reduced to global injectivity). -/
theorem stemβ_maxpool_smooth (m : Nat) (β : ℝ)
    (hn : Real.sqrt ((2 * (2 * m) * (2 * m) : ℕ) : ℝ) < β) :
    MaxPool2Smooth (Tensor3.unflatten (stemβ (2 * m) β (Xs (2 * m))) : Tensor3 2 (2 * m) (2 * m)) := by
  apply maxPool2Smooth_of_injective
  intro ci r r' s s' heq
  simp only [Tensor3.unflatten] at heq
  have h2 := finProdFinEquiv.injective (stemβ_inj (2 * m) β hn heq)
  have h5 := finProdFinEquiv.injective (congrArg Prod.fst h2)
  exact ⟨congrArg Prod.snd h5, congrArg Prod.snd h2⟩

noncomputable def stemβ_vjp (s : Nat) (β : ℝ) (hn : Real.sqrt ((2 * s * s : ℕ) : ℝ) < β)
    (v : Vec (2 * (2 * s) * (2 * s))) : HasVJPAt (stemβ s β) v :=
  convBnReluStrided_has_vjp_at WsId2 Zb2 1 1 β (by norm_num) v
    (fun k => ne_of_gt (stemβ_bn_pos s β hn v k))

theorem stemβ_diff (s : Nat) (β : ℝ) (hn : Real.sqrt ((2 * s * s : ℕ) : ℝ) < β)
    (v : Vec (2 * (2 * s) * (2 * s))) : DifferentiableAt ℝ (stemβ s β) v :=
  DifferentiableAt.comp v
    (relu_differentiableAt_of_smooth (2 * s * s) _ (fun k => ne_of_gt (stemβ_bn_pos s β hn v k)))
    ((convBnStrided_differentiable WsId2 Zb2 1 1 β (by norm_num)) v)

/-- The flat maxpool has a VJP at any point whose unflattened windows have no ties. -/
noncomputable def maxPoolFlat_vjp_of_smooth {c h w : Nat} (u : Vec (c * (2 * h) * (2 * w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten u : Tensor3 c (2 * h) (2 * w))) :
    HasVJPAt (maxPoolFlat c h w) u := by
  have h := maxPoolFlat_has_vjp_at _ hs
  rwa [Tensor3.flatten_unflatten] at h

theorem maxPoolFlat_diff_of_smooth {c h w : Nat} (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (u : Vec (c * (2 * h) * (2 * w)))
    (hs : MaxPool2Smooth (Tensor3.unflatten u : Tensor3 c (2 * h) (2 * w))) :
    DifferentiableAt ℝ (maxPoolFlat c h w) u := by
  have h := maxPoolFlat_differentiableAt _ hs hc hh hw
  rwa [Tensor3.flatten_unflatten] at h

-- ════════════════════════════════════════════════════════════════
-- § The whole 2-channel live ResNet-34 + its VJP (empty identity chains)
-- ════════════════════════════════════════════════════════════════

theorem sqrt_lt_20 {n : ℕ} (h : (n : ℝ) < 400) : Real.sqrt (n : ℝ) < 20 :=
  (Real.sqrt_lt' (by norm_num)).2 (by linarith)

/-- Identity dense head: reads each channel's GAP — `dense Wd2 bd2 u = u`. -/
noncomputable def Wd2 : Mat 2 2 := fun i j => if i = j then 1 else 0
noncomputable def bd2 : Vec 2 := fun _ => 0

noncomputable def stem2_vjp : HasVJPAt stem2 X2 := stemβ_vjp 16 30 sqrt512_lt_30 X2
theorem stem2_diff : DifferentiableAt ℝ stem2 X2 := stemβ_diff 16 30 sqrt512_lt_30 X2

noncomputable def hmp_vjp2 : HasVJPAt (maxPoolFlat 2 8 8) (stem2 X2) :=
  maxPoolFlat_vjp_of_smooth _ (stemβ_maxpool_smooth 8 30 sqrt512_lt_30)

theorem hmp_diff2 : DifferentiableAt ℝ (maxPoolFlat 2 8 8) (stem2 X2) :=
  maxPoolFlat_diff_of_smooth (by norm_num) (by norm_num) (by norm_num) _
    (stemβ_maxpool_smooth 8 30 sqrt512_lt_30)

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

-- ════════════════════════════════════════════════════════════════
-- § Non-vacuity via the Stage-2 channel-order invariant (`Dom2`)
--   `Dom2 v` := channel 1 strictly dominates channel 0 at every spatial position.
--   Every layer of `liveFwd2` preserves it; `liveFwd2 0` collapses symmetric.
-- ════════════════════════════════════════════════════════════════

/-- Channel 1 strictly dominates channel 0 at every spatial position. -/
def Dom2 {h w : Nat} (v : Vec (2 * h * w)) : Prop :=
  ∀ (hi : Fin h) (wi : Fin w),
    (Tensor3.unflatten v : Tensor3 2 h w) 0 hi wi < (Tensor3.unflatten v) 1 hi wi

/-- Scalar BN (γ=1) preserves the channel-order invariant. -/
theorem Dom2_bn {h w : Nat} (β : ℝ) (z : Vec (2 * h * w)) (hz : Dom2 z) :
    Dom2 (bnForward (2 * h * w) 1 1 β z) :=
  fun hi wi => bnForward_chan_lt 1 β one_pos z _ _ (hz hi wi)

/-- ReLU preserves the invariant in the kept-positive region. -/
theorem Dom2_relu {h w : Nat} (z : Vec (2 * h * w)) (hpos : ∀ k, 0 < z k) (hz : Dom2 z) :
    Dom2 (relu (2 * h * w) z) :=
  fun hi wi => relu_chan_lt z _ _ (hpos _) (hz hi wi)

/-- Adding a channel-symmetric constant preserves the invariant. -/
theorem Dom2_add_const {h w : Nat} (z c : Vec (2 * h * w)) (hz : Dom2 z)
    (hc : ∀ (hi : Fin h) (wi : Fin w),
      (Tensor3.unflatten c : Tensor3 2 h w) 0 hi wi = (Tensor3.unflatten c) 1 hi wi) :
    Dom2 (fun k => z k + c k) := by
  intro hi wi
  have h0 : (Tensor3.unflatten (fun k => z k + c k) : Tensor3 2 h w) 0 hi wi
      = (Tensor3.unflatten z) 0 hi wi + (Tensor3.unflatten c) 0 hi wi := rfl
  have h1 : (Tensor3.unflatten (fun k => z k + c k) : Tensor3 2 h w) 1 hi wi
      = (Tensor3.unflatten z) 1 hi wi + (Tensor3.unflatten c) 1 hi wi := rfl
  rw [h0, h1, hc hi wi]
  linarith [hz hi wi]

/-- Maxpool preserves the invariant (`maxPool2_chan_lt` lifted to the flat layer). -/
theorem Dom2_maxpool {h w : Nat} (z : Vec (2 * (2 * h) * (2 * w))) (hz : Dom2 z) :
    Dom2 (maxPoolFlat 2 h w z) := by
  intro hi wi
  have heq : (Tensor3.unflatten (maxPoolFlat 2 h w z) : Tensor3 2 h w)
      = maxPool2 (Tensor3.unflatten z) := by
    simp only [maxPoolFlat, Tensor3.unflatten_flatten]
  rw [heq]
  exact maxPool2_chan_lt (Tensor3.unflatten z) 1 0 (fun a b => hz a b) hi wi

/-- Decimation reads `(c, 2·hi, 2·wi)`, preserving the channel and the invariant. -/
theorem decimate_unflatten {h w : Nat} (z : Vec (2 * (2 * h) * (2 * w)))
    (c : Fin 2) (hi : Fin h) (wi : Fin w) :
    (Tensor3.unflatten (decimateFlat 2 h w z) : Tensor3 2 h w) c hi wi
      = (Tensor3.unflatten z : Tensor3 2 (2 * h) (2 * w)) c
          ⟨2 * hi.val, by have := hi.isLt; omega⟩ ⟨2 * wi.val, by have := wi.isLt; omega⟩ := by
  simp only [Tensor3.unflatten, decimateFlat, decimateIdx, Equiv.symm_apply_apply]

theorem Dom2_decimate {h w : Nat} (z : Vec (2 * (2 * h) * (2 * w))) (hz : Dom2 z) :
    Dom2 (decimateFlat 2 h w z) := by
  intro hi wi
  rw [decimate_unflatten z 0 hi wi, decimate_unflatten z 1 hi wi]
  exact hz _ _

/-- The live downsample preserves the channel-order invariant: the projection is
    `bn ∘ decimate` (order-preserving) and the body is a channel-symmetric constant. -/
theorem Dom2_liveDownβ (h w : Nat) (βp : ℝ) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < βp) (a : Vec (2 * (2 * h) * (2 * w))) (ha : Dom2 a) :
    Dom2 (liveDownβ h w βp a) := by
  show Dom2 (relu (2 * h * w) (residualProj
    (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 WsP2 Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a))
  apply Dom2_relu _ (fun k => liveDownW_sum_pos h w βp WsP2 hhw hn a k)
  apply Dom2_add_const _ _ ?_ ?_
  · -- Dom2 of the projection  bn_βp(decimate a)
    show Dom2 (bnForward (2 * h * w) 1 1 βp (flatConvStride2 WsP2 Zb2 a))
    rw [flatConvStride2_diag WsP2 (fun o i => rfl) a]
    exact Dom2_bn βp _ (Dom2_decimate a ha)
  · -- the body is the channel-symmetric constant 1
    intro hi wi
    rw [show ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) a
        = (fun _ => (1 : ℝ)) from liveDownPC_body_const h w hhw a]
    rfl

theorem Dom2_liveDownPC (h w : Nat) (hhw : 0 < 2 * h * w)
    (hn : Real.sqrt ((2 * h * w : ℕ) : ℝ) < 20) (a : Vec (2 * (2 * h) * (2 * w))) (ha : Dom2 a) :
    Dom2 (liveDownPC h w a) := Dom2_liveDownβ h w 20 hhw hn a ha

/-- The stem preserves the invariant: `relu ∘ bn ∘ decimate` of a dominating input. -/
theorem Dom2_stemβ (s : Nat) (β : ℝ) (a : Vec (2 * (2 * s) * (2 * s))) (ha : Dom2 a)
    (hpos : ∀ k, 0 < bnForward (2 * s * s) 1 1 β (flatConvStride2 WsId2 Zb2 a) k) :
    Dom2 (stemβ s β a) := by
  show Dom2 (relu (2 * s * s) (bnForward (2 * s * s) 1 1 β (flatConvStride2 WsId2 Zb2 a)))
  apply Dom2_relu _ hpos
  rw [flatConvStride2_diag WsId2 (fun o i => rfl) a]
  exact Dom2_bn β _ (Dom2_decimate a ha)

/-- The positional input `X i = i` has channel 1 dominating channel 0 (the flat index
    of channel 1 exceeds that of channel 0 at every position, by `finProdFinEquiv`). -/
theorem Dom2_Xs (s : Nat) : Dom2 (h := 2 * s) (w := 2 * s) (Xs s) := by
  intro hi wi
  show Xs s (finProdFinEquiv (finProdFinEquiv ((0 : Fin 2), hi), wi))
     < Xs s (finProdFinEquiv (finProdFinEquiv ((1 : Fin 2), hi), wi))
  simp only [Xs, finProdFinEquiv_apply_val, Fin.val_zero, Fin.val_one, mul_zero, mul_one]
  push_cast
  have : (0 : ℝ) < s := by exact_mod_cast (show 0 < s by have := hi.isLt; omega)
  nlinarith [this]

-- ── the per-channel head reads the dominating channels ──

/-- The identity dense head reads off channel `j`: `dense Wd2 bd2 u j = u j`. -/
theorem dense_Wd2_apply (u : Vec 2) (j : Fin 2) : dense Wd2 bd2 u j = u j := by
  simp only [dense, bd2, Wd2, add_zero, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_ite_eq' Finset.univ j (fun i => u i)]
  simp

/-- GAP at 1×1 reads each channel's single value: `gap v j = (unflatten v) j 0 0`. -/
theorem gap_1x1 (v : Vec (2 * 1 * 1)) (j : Fin 2) :
    globalAvgPoolFlat 2 1 1 v j = (Tensor3.unflatten v : Tensor3 2 1 1) j 0 0 := by
  rw [globalAvgPoolFlat_as_sum]
  show (∑ p : Fin 1 × Fin 1, (1 / ((1 : Nat) * (1 : Nat) : ℝ)) *
    v (finProdFinEquiv (finProdFinEquiv (j, p.1), p.2))) = _
  rw [Fintype.sum_prod_type]
  simp only [Fin.sum_univ_one, Nat.cast_one, mul_one, ne_eq, one_ne_zero,
    not_false_eq_true, div_self, one_mul]
  rfl

/-- **`liveFwd2 X2` is channel-asymmetric**: thread the `Dom2` invariant from the input
    through stem → maxpool → the three downsamples to the per-channel head. -/
theorem liveFwd2_X2_asym : liveFwd2 X2 0 < liveFwd2 X2 1 := by
  have d0 : Dom2 (stem2 X2) :=
    Dom2_stemβ 16 30 X2 (Dom2_Xs 16) (stemβ_bn_pos 16 30 sqrt512_lt_30 X2)
  have d1 := Dom2_maxpool (h := 8) (w := 8) (stem2 X2) d0
  have d2 := Dom2_liveDownPC 4 4 (by norm_num) (sqrt_lt_20 (by norm_num)) _ d1
  have d3 := Dom2_liveDownPC 2 2 (by norm_num) (sqrt_lt_20 (by norm_num)) _ d2
  have d4 := Dom2_liveDownPC 1 1 (by norm_num) (sqrt_lt_20 (by norm_num)) _ d3
  simp only [liveFwd2, Function.comp_apply, dense_Wd2_apply, gap_1x1]
  exact d4 0 0

-- ── `liveFwd2 0` collapses to a constant (channel-symmetric) ──

theorem decimateFlat_const {h w : Nat} (c : ℝ) :
    decimateFlat 2 h w (fun _ => c) = fun _ => c := by funext k; rfl

theorem maxPoolFlat_const {h w : Nat} (c : ℝ) :
    maxPoolFlat 2 h w (fun _ => c) = fun _ => c := by
  funext k
  simp only [maxPoolFlat, Tensor3.flatten, maxPool2, Tensor3.unflatten, max_self]

theorem stemβ_zero (s : Nat) (β : ℝ) (hs : 0 < 2 * s * s) (hβ : 0 < β) :
    stemβ s β (fun _ => (0 : ℝ)) = fun _ => β := by
  show (relu (2 * s * s) ∘ bnForward (2 * s * s) 1 1 β ∘ flatConvStride2 WsId2 Zb2)
    (fun _ => 0) = _
  simp only [Function.comp_apply]
  rw [flatConvStride2_diag WsId2 (fun o i => rfl) (fun _ => 0), decimateFlat_const,
      bnForward_const_eq hs, relu_const_pos _ β hβ]

theorem stem2_zero : stem2 (fun _ => (0 : ℝ)) = fun _ => (30 : ℝ) :=
  stemβ_zero 16 30 (by norm_num) (by norm_num)

/-- The downsample collapses a constant input to the constant `βp + 1`. -/
theorem liveDownβ_const (h w : Nat) (βp : ℝ) (hhw : 0 < 2 * h * w) (hβ1 : 0 < βp + 1) (c : ℝ) :
    liveDownβ h w βp (fun _ => c) = fun _ => βp + 1 := by
  have hproj : (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 WsP2 Zb2) (fun _ => c)
      = fun _ => βp := by
    simp only [Function.comp_apply]
    rw [flatConvStride2_diag WsP2 (fun o i => rfl) (fun _ => c), decimateFlat_const,
        bnForward_const_eq hhw]
  show relu (2 * h * w) (residualProj
    (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 WsP2 Zb2)
    ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
      (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) (fun _ => c)) = _
  rw [show residualProj
      (bnForward (2 * h * w) 1 1 βp ∘ flatConvStride2 WsP2 Zb2)
      ((bnForward (2 * h * w) 1 0 1 ∘ flatConv Zk2 Zb2) ∘
        (relu (2 * h * w) ∘ bnForward (2 * h * w) 1 0 1 ∘ flatConvStride2 Zk2 Zb2)) (fun _ => c)
        = fun _ => βp + 1 from ?_]
  · exact relu_const_pos _ (βp + 1) hβ1
  · funext k
    simp only [residualProj, biPath, hproj, liveDownPC_body_const h w hhw]

theorem liveDownPC_const {h w : Nat} (hhw : 0 < 2 * h * w) (c : ℝ) :
    liveDownPC h w (fun _ => c) = fun _ => (21 : ℝ) := by
  show liveDownβ h w 20 (fun _ => c) = _
  rw [liveDownβ_const h w 20 hhw (by norm_num) c]; norm_num

theorem gap_1x1_const (c : ℝ) : globalAvgPoolFlat 2 1 1 (fun _ => c) = fun _ => c := by
  funext j; rw [gap_1x1]; rfl

theorem dense_Wd2_const (c : ℝ) : dense Wd2 bd2 (fun _ => c) = fun _ => c := by
  funext j; rw [dense_Wd2_apply]

theorem liveFwd2_zero : liveFwd2 (fun _ => (0 : ℝ)) = fun _ => (21 : ℝ) := by
  simp only [liveFwd2, Function.comp_apply, stem2_zero, maxPoolFlat_const,
    liveDownPC_const (by norm_num) (h := 4) (w := 4),
    liveDownPC_const (by norm_num) (h := 2) (w := 2),
    liveDownPC_const (by norm_num) (h := 1) (w := 1),
    gap_1x1_const, dense_Wd2_const]

-- ── the level-2 live witness: forward is non-constant ──

/-- **The 2-channel live ResNet-34 is non-degenerate**: `liveFwd2 X2 ≠ liveFwd2 0`.
    At the positional input `X2` one channel strictly dominates the other through the whole
    net (`liveFwd2_X2_asym`), so the per-channel head gives `out 0 < out 1`; at the zero input
    everything collapses channel-symmetric (`liveFwd2_zero`, `out 0 = out 1 = 21`). This is the
    first **non-degenerate** ResNet-34 whole-net backward witness (level 2). -/
theorem liveFwd2_nonconstant : liveFwd2 X2 ≠ liveFwd2 (fun _ => (0 : ℝ)) := by
  intro h
  have h0 : liveFwd2 X2 0 = liveFwd2 X2 1 := by
    rw [h, liveFwd2_zero]
  exact absurd h0 (ne_of_lt liveFwd2_X2_asym)

end ResNet34LivePC
end Proofs
