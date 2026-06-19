import LeanMlir.Proofs.ResNet34

/-!
# Toward a live ResNet-34 witness — Stage 1 mechanism (WIP)

The goal was the `Mnv2Live` treatment for ResNet-34: a concrete instance whose
forward is provably non-constant. This file is **Stage 1 only** — the reusable
*mechanism* — and `liveFwd` below is **NOT yet a live witness**: it is still
constant-output, for a structural reason discovered here.

**What works (Stage 1):** `liveDown` — a signal-carrying strided downsample
(identity-decimate projection via `rblkPStrided_has_vjp_at`, body zeroed, proj
kept `> 0` by `bnForward_lb` with `βp = 20`), replacing the degenerate witness's
zeroed projection. `liveFwd_has_vjp_at` assembles the whole-network VJP from it,
reusing the (already injective/live) stem, the `idBlk` chains, and a nonzero
dense head. All 3-axiom clean.

**Why `liveFwd` is still constant (the structural finding):** a **1-channel**
net with BN-before-GAP is *necessarily* constant-output. `gap` (1 channel) =
spatial mean = BN's global mean = `β` (`bnForward_mean`); the ReLU is the
identity in the kept-positive region and the identity blocks are `+const`, so
`gap = β + consts` for **every** input. The signal lives in the BN-normalized
*deviations*, which `gap` averages away. (At 32×32 there is also a second
collapse: the chain ends 1×1, where BN-over-1-element `= β` trivially.) This is
the real reason `ResNet34Concrete`/`CnnConcrete` are degenerate — structural,
not zeroed weights. `Mnv2Live` escaped it only by being **2-channel** with an
identity head reading *per-channel* gap (`gap_c ≠ β`) and an input-dependent
channel asymmetry.

**What a live ResNet-34 needs (open):** (1) a 2-channel rebuild — stem, `Zk`,
the downsamples, maxpool-injectivity, every smoothness lemma re-proven at 2
channels; (2) non-vacuity threading an input-dependent channel asymmetry through
the **maxpool** (no `Mnv2Live` analogue: maxpool is positively homogeneous so
clean ratios survive, but the *scalar* BN over all `c·h·w` mean-subtracts and
breaks ratios — reconciling those is the crux). A multi-session project; this
file banks the `liveDown` mechanism toward it.
-/

namespace Proofs
namespace ResNet34Live

open Proofs ResNet34Concrete

-- ── Stage 1: a live strided downsample block ──
-- proj = bn(decimate(x))  (carries the signal; via the identity kernel `Ws`),
-- body = zero-collapse → constant 1.  `βp = 20 > √(h·w)` keeps proj > 0.

/-- A live downsample: `relu(bn₂₀(decimate x) + 1)`. -/
noncomputable def liveDown (h w : Nat) : Vec (1 * (2 * h) * (2 * w)) → Vec (1 * h * w) :=
  relu (1 * h * w) ∘ residualProj
    (bnForward (1 * h * w) 1 1 20 ∘ flatConvStride2 Ws bs)
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb))

/-- The live block's projection is strictly positive: `bn₂₀ ≥ 20 − √(h·w) > 0`. -/
theorem liveDown_proj_pos (h w : Nat) (_hhw : 0 < 1 * h * w)
    (hn : Real.sqrt ((1 * h * w : ℕ) : ℝ) < 20) (a : Vec (1 * (2 * h) * (2 * w))) (k : Fin (1 * h * w)) :
    0 < (bnForward (1 * h * w) 1 1 20 ∘ flatConvStride2 Ws bs) a k := by
  simp only [Function.comp_apply]
  have hlb := bnForward_lb (n := 1 * h * w) 1 1 20 (by norm_num) (flatConvStride2 Ws bs a) k
  rw [abs_one, one_mul] at hlb
  linarith

/-- The live block's residual body collapses to the constant `1` (zeroed convs). -/
theorem liveDown_body_const (h w : Nat) (hhw : 0 < 1 * h * w) (a : Vec (1 * (2 * h) * (2 * w))) :
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a = (fun _ => (1:ℝ)) := by
  simp only [Function.comp_apply]
  rw [flatConv_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]

/-- The post-add ReLU input is strictly positive (`proj > 0`, body `= 1`). -/
theorem liveDown_sum_pos (h w : Nat) (hhw : 0 < 1 * h * w)
    (hn : Real.sqrt ((1 * h * w : ℕ) : ℝ) < 20) (a : Vec (1 * (2 * h) * (2 * w))) (k : Fin (1 * h * w)) :
    0 < ((bnForward (1 * h * w) 1 1 20 ∘ flatConvStride2 Ws bs) a k)
      + ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
          (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a k := by
  have hp := liveDown_proj_pos h w hhw hn a k
  have hb : ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a k = 1 := by
    rw [liveDown_body_const h w hhw a]
  rw [hb]; linarith

/-- VJP of the live downsample (via `rblkPStrided_has_vjp_at`). -/
noncomputable def liveDown_vjp (h w : Nat) (hhw : 0 < 1 * h * w)
    (hn : Real.sqrt ((1 * h * w : ℕ) : ℝ) < 20) (a : Vec (1 * (2 * h) * (2 * w))) :
    HasVJPAt (liveDown h w) a :=
  rblkPStrided_has_vjp_at Zk Zb Zk Zb Ws bs 1 0 1 1 0 1 1 1 20
    (by norm_num) (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConvStride2_eq_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
      change (1:ℝ) ≠ 0; norm_num)
    (fun k => ne_of_gt (liveDown_sum_pos h w hhw hn a k))

/-- The live downsample is differentiable at every point. -/
theorem liveDown_diff (h w : Nat) (hhw : 0 < 1 * h * w)
    (hn : Real.sqrt ((1 * h * w : ℕ) : ℝ) < 20) (a : Vec (1 * (2 * h) * (2 * w))) :
    DifferentiableAt ℝ (liveDown h w) a := by
  have hsm₁ : ∀ k, bnForward (1 * h * w) 1 0 1 (flatConvStride2 Zk Zb a) k ≠ 0 := fun k => by
    rw [flatConvStride2_eq_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
    change (1:ℝ) ≠ 0; norm_num
  have hproj_diff : DifferentiableAt ℝ (bnForward (1 * h * w) 1 1 20 ∘ flatConvStride2 Ws bs) a :=
    convBnStrided_differentiable (h := h) (w := w) Ws bs 1 1 20 (by norm_num) a
  have hF_diff : DifferentiableAt ℝ
      ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a :=
    resblock_bodyStrided_differentiableAt (h := h) (w := w) Zk Zb Zk Zb 1 0 1 1 0 1
      (by norm_num) (by norm_num) a hsm₁
  have hsm_res : ∀ k, residualProj
      (bnForward (1 * h * w) 1 1 20 ∘ flatConvStride2 Ws bs)
      ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a k ≠ 0 := fun k =>
    ne_of_gt (liveDown_sum_pos h w hhw hn a k)
  show DifferentiableAt ℝ (relu (1 * h * w) ∘ residualProj
    (bnForward (1 * h * w) 1 1 20 ∘ flatConvStride2 Ws bs)
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb))) a
  exact (relu_differentiableAt_of_smooth (1 * h * w) _ hsm_res).comp a
    (DifferentiableAt.add hproj_diff hF_diff)

/-- The live downsample output is nonnegative (it is a ReLU). -/
theorem liveDown_nonneg (h w : Nat) (a : Vec (1 * (2 * h) * (2 * w))) (k : Fin (1 * h * w)) :
    0 ≤ liveDown h w a k := relu_nonneg (1 * h * w) _ k

-- ── Stage 1 (cont.): the whole live ResNet-34 + its VJP ──

theorem sqrt_dim_lt {n : ℕ} (h : (n : ℝ) < 400) : Real.sqrt (n : ℝ) < 20 := by
  rw [show (20 : ℝ) = Real.sqrt 400 by
    rw [show (400 : ℝ) = 20 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
  exact Real.sqrt_lt_sqrt (by positivity) h

/-- Nonzero (signal-reading) dense head: `dense WdL bdL u = (u 0, u 0)`. -/
noncomputable def WdL : Mat 1 2 := fun _ _ => 1
noncomputable def bdL : Vec 2 := fun _ => 0

/-- The live whole-network forward: same 34-layer skeleton as `ResNet34Concrete.fwd`,
    but with signal-carrying (identity-decimate) downsamples and a nonzero dense head. -/
noncomputable def liveFwd : Vec (1 * (2 * 16) * (2 * 16)) → Vec 2 :=
  dense WdL bdL ∘ globalAvgPoolFlat 1 1 1 ∘ chainComp ids4 ∘ liveDown 1 1 ∘ chainComp ids3 ∘
    liveDown 2 2 ∘ chainComp ids2 ∘ liveDown 4 4 ∘ chainComp ids1 ∘ maxPoolFlat 1 8 8 ∘ stem

/-- **Whole-network VJP for the live ResNet-34** — every smoothness/no-tie hypothesis
    discharged, on a net whose downsamples carry the signal and whose head reads it. -/
noncomputable def liveFwd_has_vjp_at : HasVJPAt liveFwd X :=
  resnet34_has_vjp_at stem (maxPoolFlat 1 8 8) ids1 (liveDown 4 4) ids2 (liveDown 2 2) ids3
    (liveDown 1 1) ids4 (globalAvgPoolFlat 1 1 1) (dense WdL bdL) X
    ⟨convBnReluStrided_has_vjp_at Ws bs 1 1 20 (by norm_num) X (fun k => ne_of_gt (stem_bn_pos k)),
     DifferentiableAt.comp X
       (relu_differentiableAt_of_smooth (1 * 16 * 16) _ (fun k => ne_of_gt (stem_bn_pos k)))
       ((convBnStrided_differentiable Ws bs 1 1 20 (by norm_num)) X)⟩
    ⟨hmp_vjp, hmp_diff⟩
    (idChainData 8 8 (by norm_num) (maxPoolFlat 1 8 8 (stem X))
      (fun k => le_of_lt (mp_stem_pos k)) 3)
    ⟨liveDown_vjp 4 4 (by norm_num) (sqrt_dim_lt (by norm_num)) _,
     liveDown_diff 4 4 (by norm_num) (sqrt_dim_lt (by norm_num)) _⟩
    (idChainData 4 4 (by norm_num) _ (fun k => liveDown_nonneg 4 4 _ k) 4)
    ⟨liveDown_vjp 2 2 (by norm_num) (sqrt_dim_lt (by norm_num)) _,
     liveDown_diff 2 2 (by norm_num) (sqrt_dim_lt (by norm_num)) _⟩
    (idChainData 2 2 (by norm_num) _ (fun k => liveDown_nonneg 2 2 _ k) 6)
    ⟨liveDown_vjp 1 1 (by norm_num) (sqrt_dim_lt (by norm_num)) _,
     liveDown_diff 1 1 (by norm_num) (sqrt_dim_lt (by norm_num)) _⟩
    (idChainData 1 1 (by norm_num) _ (fun k => liveDown_nonneg 1 1 _ k) 3)
    ⟨(globalAvgPoolFlat_has_vjp 1 1 1).toHasVJPAt _, (globalAvgPoolFlat_differentiable 1 1 1) _⟩
    ⟨(dense_has_vjp WdL bdL).toHasVJPAt _, (dense_differentiable WdL bdL) _⟩

theorem liveFwd_has_vjp_correct (dy : Vec 2) (i : Fin (1 * (2 * 16) * (2 * 16))) :
    liveFwd_has_vjp_at.backward dy i = ∑ j : Fin 2, pdiv liveFwd X i j * dy j :=
  liveFwd_has_vjp_at.correct dy i

end ResNet34Live
end Proofs
