import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Close
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChannelLN

/-! # ConvNeXt-T fold — the per-channel layer-scale γ gradient cert (the one new proof)

The committed ConvNeXt SGD net trains **per-channel** layer-scale `γ : Vec c` (the `layerScaleChF`
forward, which broadcasts `γ` over the `c·h·w` activation via `chanIdx`), NOT a per-element `Vec n`
layer-scale. So the fold needs the **per-channel** version: the γ-gradient w.r.t. the `Vec c`
parameter is the per-channel reduce `dγ_c = Σ_{k : chanIdx k = c} x_k · dy_k` (the `lsGradCh` emit:
`multiply x dy` → `reduce[0,2,3]`), and this is exactly the certified Jacobian of `layerScaleChF`'s
forward (as a function of `γ : Vec c`) contracted with the cotangent.

This is the only genuinely-NEW proof obligation for the ConvNeXt tie (the depthwise-7×7, 1×1-conv,
strided-stem/downsample and dense param grads are covered by the existing conv and dense certs, the
channel-LN γ/β by `ChannelLN`). It is linear in the parameter (`pdiv_of_linear`), with the
`chanIdx` reindex (the per-channel broadcast) — `∂(γ'(chanIdx j)·x_j)/∂γ'_c = x_j·[chanIdx j = c]`.

The `layerScaleChGammaSgd` core `SHlo` op (the per-channel layer-scale param-SGD, emitting
`lsGradCh` + the SGD wrap) `den`otes the LHS here, so `den = certified` is a one-line delegation to
`cnx_render_lsgammaCh_certified` (`layerScaleChGammaSgd_den`). -/

namespace Proofs.CnxPoC

open scoped BigOperators
open Proofs Proofs.StableHLO

/-- **Jacobian of per-channel layer-scale w.r.t. the `Vec c` parameter** —
    `∂(γ'(chanIdx j)·x_j)/∂γ'_c = x_j·[chanIdx j = c]`. The broadcast `chanIdx` reindex makes the
    basis vector read through it the channel indicator. -/
theorem pdiv_layerScaleCh_gamma {c h w : Nat} (x : Vec (c * h * w)) (γ : Vec c)
    (cc : Fin c) (j : Fin (c * h * w)) :
    pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j
      = if cc = chanIdx c h w j then x j else 0 := by
  rw [pdiv_of_linear _ (fun _ _ => by funext; simp [layerScale, add_mul])
    (fun _ _ => by funext; simp [layerScale, mul_assoc])]
  simp [layerScale, @eq_comm _ cc]

/-- **Per-channel layer-scale γ output, certified.** The rendered per-channel reduce
    `dγ_c = Σ_{k : chanIdx k = c} x_k·dy_k` (the `lsGradCh` emit) equals the certified Jacobian of
    `layerScaleChF`'s forward (as a function of `γ : Vec c`) contracted with the cotangent. The
    `den` target of the `layerScaleChGammaSgd` core op. -/
theorem cnx_render_lsgammaCh_certified {c h w : Nat} (x : Vec (c * h * w)) (γ : Vec c)
    (dy : Vec (c * h * w)) (lr : ℝ) (cc : Fin c) :
    γ cc - lr * ∑ k : Fin (c * h * w), (if chanIdx c h w k = cc then x k * dy k else 0)
      = γ cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j * dy j := by
  simp only [pdiv_layerScaleCh_gamma, ite_mul, zero_mul, @eq_comm _ cc]

-- ════════════════════════════════════════════════════════════════
-- § The §1 den-fold — each new core op `den`otes the certified loss-descent step
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel layer-scale γ op denotes the certified step.** The emitted `layerScaleChGammaSgd`
    (the `lsGradCh` per-channel reduce + SGD) `den`otes `γ − lr·(certified ∂(layerScaleChF)/∂γ · cot)`.
    One-line delegation to `cnx_render_lsgammaCh_certified`. -/
theorem layerScaleChGammaSgd_den {c h w : Nat} (gN xN lrStr cotN : String)
    (x : Vec (c * h * w)) (γ : Vec c) (dy : Vec (c * h * w)) (lr : ℝ) (cc : Fin c) :
    den (SHlo.layerScaleChGammaSgd gN xN lrStr x γ lr (.operand cotN dy)) cc
      = γ cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j * dy j := by
  simp only [denStepApp]
  exact cnx_render_lsgammaCh_certified x γ dy lr cc

/-! ## The channel-LN γ/β ops — the two the committed render actually emits

`ConvNeXtRender.lnGammaTail`/`lnBetaTail` re-emit the `[h·w, c]` transposes and then run ViT's
`veclnGammaSgd` / `rowDenseBiasSgd` on that view, so the op operands below are the transposed
views `chanLNRows` of the saved LN input and of the chain cotangent — the values those SSA names
denote. The certified Jacobian on the right is `chanLNTensor3`'s, in the `c·h·w` activation
layout the rest of the block lives in; `ChannelLN`'s permutation argument is what lets
one op serve both layouts. They cover every one of the net's 22 spatial LN sites (1 stem + 18 block + 3 downsample); the 23rd, the
head, runs after GAP and is ViT's vector-LN at `N = 1` (`ViTPoC.veclnGammaSgd_den`). -/

/-- **Channel-LN γ op denotes the certified step.** One-line delegation to
    `cnx_render_chlngamma_certified`. The free `β` is the site's LN β (the γ grad is β-free). -/
theorem chanLnGammaSgd_den {c h w : Nat} (gN xN epsStr lrStr cotN : String)
    (ε : ℝ) (β : Vec c) (x : Vec (c * h * w)) (γ : Vec c) (cot : Vec (c * h * w))
    (lr : ℝ) (k : Fin c) :
    den (SHlo.veclnGammaSgd (N := h * w) (D := c) gN xN epsStr lrStr ε
          (chanLNRows c h w x) γ lr (.operand cotN (chanLNRows c h w cot))) k
      = γ k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j := by
  simp only [denStep, denStepApp]
  exact cnx_render_chlngamma_certified ε β γ x cot lr k

/-- **Channel-LN β op denotes the certified step.** The β grad is the plain row reduce, so the
    render uses the same `rowDenseBiasSgd` op ViT's LN-β does. The free `ε`/`γ` carry the LN
    constants (the β grad is independent of both). -/
theorem chanLnBetaSgd_den {c h w : Nat} (bN lrStr cotN : String)
    (ε : ℝ) (γ : Vec c) (x : Vec (c * h * w)) (β : Vec c) (cot : Vec (c * h * w))
    (lr : ℝ) (k : Fin c) :
    den (SHlo.rowDenseBiasSgd (N := h * w) (c := c) bN lrStr β lr
          (.operand cotN (chanLNRows c h w cot))) k
      = β k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j := by
  simp only [denStep, denStepApp]
  exact cnx_render_chlnbeta_certified ε γ β x cot lr k

-- ════════════════════════════════════════════════════════════════
-- § Tie clauses — one channel-LN SGD node each (each its `_den` lemma's statement with the index
--   bound, over the flat input `x`; each `…_holds` below proves it)
-- ════════════════════════════════════════════════════════════════

/-- A channel-LN γ SGD node, tied (`chanLnGammaSgd_den`). -/
def ChanLNGammaSgdTied (h w : Nat) {c : Nat} (gN xN epsStr lrStr cotN : String) (ε : ℝ)
    (β : Vec c) (x : Vec (c * h * w)) (γ : Vec c) (cot : Vec (c * h * w)) (lr : ℝ) : Prop :=
  ∀ k : Fin c,
    den (SHlo.veclnGammaSgd (N := h * w) (D := c) gN xN epsStr lrStr ε
          (chanLNRows c h w x) γ lr (.operand cotN (chanLNRows c h w cot))) k
      = γ k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j

/-- A channel-LN β SGD node, tied (`chanLnBetaSgd_den`). -/
def ChanLNBetaSgdTied (h w : Nat) {c : Nat} (bN lrStr cotN : String) (ε : ℝ) (γ : Vec c)
    (x : Vec (c * h * w)) (β : Vec c) (cot : Vec (c * h * w)) (lr : ℝ) : Prop :=
  ∀ k : Fin c,
    den (SHlo.rowDenseBiasSgd (N := h * w) (c := c) bN lrStr β lr
          (.operand cotN (chanLNRows c h w cot))) k
      = β k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j

/-! Each clause holds, every argument implicit (read off the goal by a step tie's constructor). -/

theorem chanLNGammaSgdTied_holds {h w c : Nat} {gN xN epsStr lrStr cotN : String} {ε : ℝ}
    {β : Vec c} {x : Vec (c * h * w)} {γ : Vec c} {cot : Vec (c * h * w)} {lr : ℝ} :
    ChanLNGammaSgdTied h w gN xN epsStr lrStr cotN ε β x γ cot lr := fun k =>
  chanLnGammaSgd_den gN xN epsStr lrStr cotN ε β x γ cot lr k

theorem chanLNBetaSgdTied_holds {h w c : Nat} {bN lrStr cotN : String} {ε : ℝ} {γ : Vec c}
    {x : Vec (c * h * w)} {β : Vec c} {cot : Vec (c * h * w)} {lr : ℝ} :
    ChanLNBetaSgdTied h w bN lrStr cotN ε γ x β cot lr := fun k =>
  chanLnBetaSgd_den bN lrStr cotN ε γ x β cot lr k

end Proofs.CnxPoC
