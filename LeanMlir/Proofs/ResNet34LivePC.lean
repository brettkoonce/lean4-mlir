import LeanMlir.Proofs.ResNet34Live2

/-!
# Toward a live ResNet-34 witness — Stage 3: the 2-channel layer rebuild (A1, WIP)

Stage 2 (`ResNet34Live2.lean`) banked the channel-order invariant kit (the A2
non-vacuity crux). This file does the **mechanical 2-channel re-instantiation**
(`planning/whole_network_backward.md` Item A1): the layers `liveDown`/`stem`/…
that Stage 1 hardcoded at `1 * h * w` rebuilt at `c = 2`, reusing the
channel-generic VJP/differentiability machinery (`rblkPStrided_has_vjp_at`,
`convBnStrided_differentiable`, …) at `oc = ic = 2`.

**This file (Stage 3 so far):**
- `liveDownPC` — the 2-channel signal-carrying strided downsample (channel-diagonal
  identity-decimate projection `WsP2`, zeroed body), with its whole VJP / `DifferentiableAt`
  / nonnegativity, exactly mirroring Stage 1's `liveDown` at `oc = ic = 2`.
- `bnForward_coord_inj` — scalar BN is injective *per coordinate* (the ingredient the
  2-channel stem's *per-channel* maxpool no-tie needs, since the weight-asymmetric stem
  is not globally injective).

Build-checked, dimension-generic, 3-axiom clean. The stem rebuild + the whole-net
assembly + `forward X ≠ forward 0` (via the Stage-2 invariant) remain.
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

end ResNet34LivePC
end Proofs
