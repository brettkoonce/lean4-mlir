import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChainClose
import LeanMlir.Proofs.Architectures.ChannelLN

/-! # ConvNeXt's channel LayerNorm — the render's γ/β outputs and the graph transport

The op itself — `chanLNTensor3`, its VJP, `chanLNRows` and the γ/β gradient contractions — is
`Architectures/ChannelLN`. This file holds what is ConvNeXt's: the two statements that the render's
LN tail (`veclnGammaSgd` / `rowDenseBiasSgd` on the transposed views) denotes the certified γ/β
step (`cnx_render_chlngamma_certified`, `cnx_render_chlnbeta_certified`), and the graph-side `▸`
transport (`den_reassocS` / `den_unassocS`) that keeps `ConvNeXtRender`'s `reassoc` and
`chanLNTensor3` describing one function. -/

namespace Proofs

open scoped BigOperators

/-- **Channel-LN γ output, certified.** The rendered per-channel reduce — ViT's
    `vecLNGradGamma` on the two transposed views the tail emits — equals the certified Jacobian
    of `chanLNTensor3` in its `Vec c` γ, contracted with the activation-layout cotangent. The
    `den` target of the render's `veclnGammaSgd` LN tail. -/
theorem cnx_render_chlngamma_certified {c h w : Nat} (ε : ℝ) (β γ : Vec c)
    (x cot : Vec (c * h * w)) (lr : ℝ) (k : Fin c) :
    γ k - lr * vecLNGradGamma (h * w) c ε (Mat.unflatten (chanLNRows c h w x))
                  (Mat.unflatten (chanLNRows c h w cot)) k
      = γ k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j := by
  rw [chanLN_gamma_contract ε β γ x cot k]
  exact congrArg (fun t => γ k - lr * t)
    (vit_veclnGamma_grad_bridge ε β γ (Mat.unflatten (chanLNRows c h w x))
      (chanLNRows c h w cot) k)

/-- **Channel-LN β output, certified.** The β grad is the plain reduce `Σ_rows dy`, so the same
    `rowDenseBiasSgd` op ViT's LN-β uses denotes it here too. -/
theorem cnx_render_chlnbeta_certified {c h w : Nat} (ε : ℝ) (γ β : Vec c)
    (x cot : Vec (c * h * w)) (lr : ℝ) (k : Fin c) :
    β k - lr * vecLNGradBeta (h * w) c (Mat.unflatten (chanLNRows c h w cot)) k
      = β k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j := by
  rw [chanLN_beta_contract ε γ β x cot k]
  exact congrArg (fun t => β k - lr * t)
    (vit_veclnBeta_grad_bridge ε γ β (Mat.unflatten (chanLNRows c h w x))
      (chanLNRows c h w cot) k)

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The graph-side transport
-- ════════════════════════════════════════════════════════════════

/-- **The graph's `▸` transport IS the math's Mat-split bridge** — `den_castIdx` composed with
    `reassocFwdIdx_val`. This is the lemma that keeps `ConvNeXtRender`'s `reassoc` and
    `chanLNTensor3` describing one function. -/
theorem den_reassocS {c h w : Nat} (e : SHlo (c * h * w)) :
    den ((Nat.mul_assoc c h w) ▸ e) = reassocFwd c h w (den e) := by
  refine (den_castIdx (Nat.mul_assoc c h w) e).trans ?_
  funext k
  exact congrArg (den e) (Fin.ext (reassocFwdIdx_val c h w k).symm)

theorem den_unassocS {c h w : Nat} (e : SHlo (c * (h * w))) :
    den ((Nat.mul_assoc c h w).symm ▸ e) = reassocBack c h w (den e) := by
  refine (den_castIdx (Nat.mul_assoc c h w).symm e).trans ?_
  funext k
  exact congrArg (den e) (Fin.ext (reassocBackIdx_val c h w k).symm)

end Proofs.StableHLO
