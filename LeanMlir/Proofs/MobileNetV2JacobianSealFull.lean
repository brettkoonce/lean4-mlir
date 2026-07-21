import LeanMlir.Proofs.MobileNetV2JacobianSeal
import LeanMlir.Proofs.Foundation.ResNet34

/-!
# Full-depth `Mnv2Live` witness + level-3 seal (17 inverted-residual blocks)

`MobileNetV2JacobianSeal` seals the live MobileNetV2 backward at the **2-block**
representative (`mobilenetv2Forward` is hard-wired to one skip + one no-skip
inverted-residual block). This file fills it out to the real MobileNetV2 **block
count** — 17 inverted-residual blocks (the paper schedule `[1,2,3,4,3,3,1]`) — and
carries both the level-2 non-vacuity and the level-3 nonzero-Jacobian seal through
to the full-depth net. (Channel realism — 224×224, the real width schedule — is the
separate "realistic dims" item; like the ResNet-34 full-depth witness this stays at
the witness's 2-channel / 2×2 dims, where the seal carrier lives.)

The clean-extension mechanism is the ResNet-34 `idBlk2` trick, and it is *cleaner*
here because the inverted-residual block has **no final activation** (linear
bottleneck): block-1's body is the zeroed-conv constant `3` (`invresBody₁_const`),
so a zeroed-body **skip** block `residual (invresBody …)` is the affine shift
`a ↦ a + 3` for **every** input — no nonnegativity bookkeeping (ResNet needed the
post-add ReLU to be off). A chain of `k` such blocks is `a ↦ a + 3k`; the shift is
channel-symmetric, so it passes through GAP (`gap (a+c) = gap a + c`) and the
identity head, surviving as a single additive constant `+45 = 3·15` that **cancels
in the directional derivative** at the witness input `0`. So the whole seal reduces
to `MobileNetV2JacobianSeal`'s `Qq` / `g_hasDerivAt` verbatim, while the whole-net
VJP is genuinely composed through all 17 block backward maps (`fwdFull_has_vjp_at`).
-/

namespace Proofs
namespace Mnv2Live

open scoped BigOperators
open Finset Filter Topology

-- ════════════════════════════════════════════════════════════════
-- § The window lemma (every ReLU6 site, every input) — public copy
-- ════════════════════════════════════════════════════════════════

/-- The five ReLU6 sites discharge through the one window lemma (length 8, γ=1,
    β=3, ε=1), regardless of the activation feeding them. (A public peer of the
    file-private `win`/`win'`, reusable in the chain assembly below.) -/
theorem winF (z : Vec (2 * 2 * 2)) (k : Fin (2 * 2 * 2)) :
    bnForward (2 * 2 * 2) 1 1 3 z k ≠ 0 ∧ bnForward (2 * 2 * 2) 1 1 3 z k ≠ 6 := by
  obtain ⟨h0, h6⟩ := bn13_window (2 * 2 * 2) (by norm_num) (by norm_num) 1 one_pos z k
  exact ⟨h0.ne', h6.ne⟩

-- ════════════════════════════════════════════════════════════════
-- § The identity inverted-residual block (= block-1: zeroed body, skip)
--   `ivId a = a + 3` for EVERY input (no relu — linear bottleneck).
-- ════════════════════════════════════════════════════════════════

/-- The 2-channel identity inverted-residual block: a zeroed-body MBConv with the
    stride-1 same-channel skip. Reuses block-1's (all-zero) weights. -/
noncomputable def ivId : Vec (2 * 2 * 2) → Vec (2 * 2 * 2) :=
  residual (invresBody (h := 2) (w := 2) We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3)

/-- **`ivId` is the affine shift `a ↦ a + 3`**, for *every* input (the zeroed body
    is the constant `3`, the residual adds the input, and there is no final relu). -/
theorem ivId_eq (a : Vec (2 * 2 * 2)) : ivId a = (fun k => a k + 3) := by
  funext k
  simp only [ivId, residual, biPath, invresBody₁_const]
  ring

/-- **A chain of `k` identity blocks shifts its input by `3k`.** -/
theorem ivId_chain_eq (k : Nat) (a : Vec (2 * 2 * 2)) :
    chainComp (List.replicate k ivId) a = (fun i => a i + 3 * (k : ℝ)) := by
  induction k with
  | zero => funext i; simp
  | succ n ih =>
    rw [List.replicate_succ, chainComp_cons, Function.comp_apply, ih, ivId_eq]
    funext i; push_cast; ring

/-- The whole-block VJP of an identity block at *any* point (window everywhere). -/
noncomputable def ivId_hasVJPAt (a : Vec (2 * 2 * 2)) : HasVJPAt ivId a :=
  invresSkip_has_vjp_at We₁ be₁ 1 1 3 one_pos Wd₁ bd₁ 1 1 3 one_pos Wp₁ bp₁ 1 1 3 one_pos
    a (fun k => winF _ k) (fun k => winF _ k)

theorem ivId_diffAt (a : Vec (2 * 2 * 2)) : DifferentiableAt ℝ ivId a :=
  invresSkip_differentiableAt We₁ be₁ 1 1 3 one_pos Wd₁ bd₁ 1 1 3 one_pos Wp₁ bp₁ 1 1 3 one_pos
    a (fun k => winF _ k) (fun k => winF _ k)

/-- Per-block VJP + differentiability data for a chain of `k` identity blocks. The
    window holds at every running activation, so no point threading is needed. -/
noncomputable def ivId_chainData (k : Nat) (a : Vec (2 * 2 * 2)) :
    ChainData a (List.replicate k ivId) := by
  induction k with
  | zero => exact PUnit.unit
  | succ n ih =>
    rw [List.replicate_succ]
    exact ⟨ivId_diffAt _, ivId_hasVJPAt _, ih⟩

-- ════════════════════════════════════════════════════════════════
-- § GAP absorbs a constant shift
-- ════════════════════════════════════════════════════════════════

/-- **GAP commutes with a constant shift** (it averages, so a global `+c` survives
    as `+c` in every channel). -/
theorem gap_add_const (z : Vec (2 * 2 * 2)) (c : ℝ) :
    globalAvgPoolFlat 2 2 2 (fun i => z i + c) = (fun j => globalAvgPoolFlat 2 2 2 z j + c) := by
  funext j
  rw [globalAvgPoolFlat_as_sum]
  simp only [mul_add, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_prod, Fintype.card_fin, nsmul_eq_mul]
  ring

-- ════════════════════════════════════════════════════════════════
-- § The full-depth (17-block) live MobileNetV2
--   stem → block1(skip) → block2(no-skip) → 15 identity skip blocks → GAP → head
-- ════════════════════════════════════════════════════════════════

/-- **The full-depth 2-channel live MobileNetV2** — 17 inverted-residual blocks
    (one skip, one no-skip, and 15 identity skip blocks), the paper block count. -/
noncomputable def fwdFull : Vec (1 * 2 * 2) → Vec 2 :=
  dense Wh bh ∘
  globalAvgPoolFlat 2 2 2 ∘
  chainComp (List.replicate 15 ivId) ∘
  invresBody (h := 2) (w := 2) We₂ be₂ 1 1 3 Wd₂ bd₂ 1 1 3 Wp₂ bp₂ 1 1 3 ∘
  residual (invresBody (h := 2) (w := 2) We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3) ∘
  (relu6 (2 * 2 * 2) ∘ bnForward (2 * 2 * 2) 1 1 3 ∘ flatConv Ws bs)

-- ════════════════════════════════════════════════════════════════
-- § The chain washes out: `fwdFull = fwd + 45`
-- ════════════════════════════════════════════════════════════════

/-- **The 15 identity blocks wash out to a single constant `+45`** (= `3·15`): they
    shift by `45`, GAP passes the shift, and the identity head preserves it. The
    constant cancels in the channel-wise directional derivative (the seal below). -/
theorem fwdFull_eq_add (v : Vec (1 * 2 * 2)) : fwdFull v = (fun j => fwd v j + 45) := by
  funext j
  simp only [fwdFull, fwd, mobilenetv2Forward, Function.comp_apply]
  rw [ivId_chain_eq 15, gap_add_const, dense_Wh_apply, dense_Wh_apply]
  norm_num

/-- **The full-depth net is differentiable everywhere** (`fwd = fwdCF` is, and a
    constant shift preserves it). -/
theorem fwdFull_differentiable : Differentiable ℝ fwdFull := by
  have heq : fwdFull = (fun v => fwdCF v + (fun _ => (45 : ℝ))) := by
    funext v j
    rw [fwdFull_eq_add, fwd_eq_fwdCF]
    simp only [Pi.add_apply]
  rw [heq]
  exact fwdCF_differentiable.add_const _

-- ════════════════════════════════════════════════════════════════
-- § The whole-net VJP, genuinely composed through all 17 blocks
-- ════════════════════════════════════════════════════════════════

/-- **The full-depth whole-network VJP at any input.** Composes the per-block VJPs:
    stem → skip block-1 → no-skip block-2 → the 15-block identity chain
    (`chain_vjp_diff_at`) → GAP → head, every ReLU6 site discharged by `winF`. The
    backward genuinely runs all 17 block backward maps in reverse. -/
noncomputable def fwdFull_has_vjp_at (x : Vec (1 * 2 * 2)) : HasVJPAt fwdFull x := by
  unfold fwdFull
  -- stem
  set S0 := (relu6 (2 * 2 * 2) ∘ bnForward (2 * 2 * 2) 1 1 3 ∘ flatConv Ws bs) with hS0
  have s0_vjp : HasVJPAt S0 x := convBnRelu6_has_vjp_at Ws bs 1 1 3 one_pos x (fun k => winF _ k)
  have s0_diff : DifferentiableAt ℝ S0 x :=
    convBnRelu6_differentiableAt Ws bs 1 1 3 one_pos x (fun k => winF _ k)
  -- block1 (skip IR)
  set B1 := residual (invresBody (h := 2) (w := 2) We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3) with hB1
  have b1_vjp : HasVJPAt B1 (S0 x) :=
    invresSkip_has_vjp_at We₁ be₁ 1 1 3 one_pos Wd₁ bd₁ 1 1 3 one_pos Wp₁ bp₁ 1 1 3 one_pos
      (S0 x) (fun k => winF _ k) (fun k => winF _ k)
  have b1_diff : DifferentiableAt ℝ B1 (S0 x) :=
    invresSkip_differentiableAt We₁ be₁ 1 1 3 one_pos Wd₁ bd₁ 1 1 3 one_pos Wp₁ bp₁ 1 1 3 one_pos
      (S0 x) (fun k => winF _ k) (fun k => winF _ k)
  have s1_vjp : HasVJPAt (B1 ∘ S0) x := vjp_comp_at S0 B1 x s0_diff b1_diff s0_vjp b1_vjp
  have s1_diff : DifferentiableAt ℝ (B1 ∘ S0) x := b1_diff.comp x s0_diff
  -- block2 (no-skip IR)
  set B2 := invresBody (h := 2) (w := 2) We₂ be₂ 1 1 3 Wd₂ bd₂ 1 1 3 Wp₂ bp₂ 1 1 3 with hB2
  have b2_vjp : HasVJPAt B2 (B1 (S0 x)) :=
    invresBody_has_vjp_at We₂ be₂ 1 1 3 one_pos Wd₂ bd₂ 1 1 3 one_pos Wp₂ bp₂ 1 1 3 one_pos
      (B1 (S0 x)) (fun k => winF _ k) (fun k => winF _ k)
  have b2_diff : DifferentiableAt ℝ B2 (B1 (S0 x)) :=
    invresBody_differentiableAt We₂ be₂ 1 1 3 one_pos Wd₂ bd₂ 1 1 3 one_pos Wp₂ bp₂ 1 1 3 one_pos
      (B1 (S0 x)) (fun k => winF _ k) (fun k => winF _ k)
  have s2_vjp : HasVJPAt (B2 ∘ (B1 ∘ S0)) x := vjp_comp_at (B1 ∘ S0) B2 x s1_diff b2_diff s1_vjp b2_vjp
  have s2_diff : DifferentiableAt ℝ (B2 ∘ (B1 ∘ S0)) x := b2_diff.comp x s1_diff
  -- the 15-block identity chain
  set CH := chainComp (List.replicate 15 ivId) with hCH
  have ch_pair := chain_vjp_diff_at (B2 (B1 (S0 x))) (List.replicate 15 ivId)
    (ivId_chainData 15 (B2 (B1 (S0 x))))
  have ch_vjp : HasVJPAt CH (B2 (B1 (S0 x))) := ch_pair.fst
  have ch_diff : DifferentiableAt ℝ CH (B2 (B1 (S0 x))) := ch_pair.snd
  have s3_vjp : HasVJPAt (CH ∘ (B2 ∘ (B1 ∘ S0))) x :=
    vjp_comp_at (B2 ∘ (B1 ∘ S0)) CH x s2_diff ch_diff s2_vjp ch_vjp
  have s3_diff : DifferentiableAt ℝ (CH ∘ (B2 ∘ (B1 ∘ S0))) x := ch_diff.comp x s2_diff
  -- GAP
  set P := CH ∘ (B2 ∘ (B1 ∘ S0)) with hP
  have gap_diff : DifferentiableAt ℝ (globalAvgPoolFlat 2 2 2) (P x) :=
    (globalAvgPoolFlat_differentiable 2 2 2) (P x)
  have s4_vjp : HasVJPAt (globalAvgPoolFlat 2 2 2 ∘ P) x :=
    vjp_comp_at P (globalAvgPoolFlat 2 2 2) x s3_diff gap_diff s3_vjp
      ((globalAvgPoolFlat_has_vjp 2 2 2).toHasVJPAt (P x))
  have s4_diff : DifferentiableAt ℝ (globalAvgPoolFlat 2 2 2 ∘ P) x := gap_diff.comp x s3_diff
  -- head
  exact vjp_comp_at (globalAvgPoolFlat 2 2 2 ∘ P) (dense Wh bh) x s4_diff
    ((dense_differentiable Wh bh) _) s4_vjp
    ((dense_has_vjp Wh bh).toHasVJPAt _)

/-- **Public correctness theorem** — the full-depth net's backward equals the
    `pdiv`-contracted Jacobian VJP at every input. -/
theorem fwdFull_has_vjp_correct (x : Vec (1 * 2 * 2)) (dy : Vec 2) (i : Fin (1 * 2 * 2)) :
    (fwdFull_has_vjp_at x).backward dy i = ∑ j : Fin 2, pdiv fwdFull x i j * dy j :=
  (fwdFull_has_vjp_at x).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § The full-depth level-3 seal (reuses `Qq` / `g_hasDerivAt` via the washout)
-- ════════════════════════════════════════════════════════════════

/-- The full net's channel-0 output along the ray has the same derivative as the
    2-block witness (the chain is transparent to it — a constant `+45` offset). -/
theorem g_full_hasDerivAt : HasDerivAt (fun t : ℝ => fwdFull (t • X) 0) (Qq 0) 0 := by
  have heq : (fun t : ℝ => fwdFull (t • X) 0) = (fun t : ℝ => fwdCF (t • X) 0 + 45) := by
    funext t
    rw [fwdFull_eq_add, fwd_eq_fwdCF]
  rw [heq]
  exact g_hasDerivAt.add_const 45

/-- **`fderiv ℝ fwdFull 0 ≠ 0`** — the full-depth (17-block) live MobileNetV2's
    whole-net Jacobian is genuinely non-trivial at the witness input `0`. -/
theorem fwdFull_jacobian_nonzero : fderiv ℝ fwdFull 0 ≠ 0 := by
  intro hzero
  have hfd : HasFDerivAt fwdFull (0 : Vec (1 * 2 * 2) →L[ℝ] Vec 2) (0 : Vec (1 * 2 * 2)) := by
    rw [← hzero]; exact (fwdFull_differentiable 0).hasFDerivAt
  have hsmul : HasDerivAt (fun t : ℝ => t • X) X 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).smul_const X
  have hcomp : HasDerivAt (fun t : ℝ => fwdFull (t • X)) (0 : Vec 2) 0 := by
    have := HasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hfd hsmul (by simp)
    exact this
  have hcomp0 : HasDerivAt (fun t : ℝ => fwdFull (t • X) 0) (0 : ℝ) 0 := by
    have := (hasDerivAt_pi.mp hcomp) (0 : Fin 2)
    simpa using this
  exact Qq_zero_ne (g_full_hasDerivAt.unique hcomp0)

/-- **The full-depth level-3 seal** (full 17-block MobileNetV2): the proven whole-
    network backward of the full-depth live MobileNetV2 is **not the zero map** at
    the witness input `0` — some basis-cotangent probe returns a nonzero row. The
    full-depth peer of `mnv2Live_backward_nontrivial`. -/
theorem fwdFull_backward_nontrivial :
    ∃ (j₀ : Fin 2) (i₀ : Fin (1 * 2 * 2)),
      (fwdFull_has_vjp_at 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (fwdFull_has_vjp_at 0).backward_nontrivial_of_fderiv_ne fwdFull_jacobian_nonzero

/-- **The full-depth witness is non-degenerate** (level 2): `fwdFull X ≠ fwdFull 0`. -/
theorem fwdFull_nonconstant : fwdFull X ≠ fwdFull (fun _ => (0 : ℝ)) := by
  rw [fwdFull_eq_add X, fwdFull_eq_add (fun _ => (0 : ℝ))]
  intro h
  apply mnv2Live_forward_nonconstant
  funext c
  have := congrFun h c
  simpa using this

end Mnv2Live
end Proofs
