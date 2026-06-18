import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.ConvNeXtClose

/-! # ConvNeXt-T §1 fold — the per-channel layer-scale γ gradient cert (the one new proof)

The committed ConvNeXt SGD net trains **per-channel** layer-scale `γ : Vec c` (the `layerScaleChF`
forward, which broadcasts `γ` over the `c·h·w` activation via `chanIdx`), NOT the per-element `Vec n`
layer-scale that `ConvNeXtClose.cnx_render_lsgamma_certified` certifies. So the §1 fold needs the
**per-channel** version: the γ-gradient w.r.t. the `Vec c` parameter is the per-channel reduce
`dγ_c = Σ_{k : chanIdx k = c} x_k · dy_k` (the `lsGradCh` emit: `multiply x dy` → `reduce[0,2,3]`),
and this is exactly the certified Jacobian of `layerScaleChF`'s forward (as a function of `γ : Vec c`)
contracted with the cotangent.

This is the only genuinely-NEW proof obligation for the ConvNeXt tie (the depthwise-7×7, 1×1-conv,
strided-stem/downsample, dense, and scalar-LN γ/β param grads are all covered by existing
`ConvNeXtClose` / M2 / M3 certs). It chains `pdiv_mul`/`pdiv_reindex`/`pdiv_const` exactly like
`ConvNeXtClose.pdiv_layerScale_gamma`, but with the `chanIdx` reindex (the per-channel broadcast)
in place of the identity — `∂(γ'(chanIdx j)·x_j)/∂γ'_c = x_j·[chanIdx j = c]`.

Once the `layerScaleChGammaSgd` core `SHlo` op lands (the per-channel layer-scale param-SGD, emitting
`lsGradCh` + the SGD wrap), its `den` reduces to the LHS here, so `den = certified` is a one-line
delegation to `cnx_render_lsgammaCh_certified`. -/

namespace Proofs.CnxPoC

open scoped BigOperators
open Proofs Proofs.StableHLO

/-- **Jacobian of per-channel layer-scale w.r.t. the `Vec c` parameter** —
    `∂(γ'(chanIdx j)·x_j)/∂γ'_c = x_j·[chanIdx j = c]`. The per-channel mirror of
    `ConvNeXtClose.pdiv_layerScale_gamma` (which is per-element); the broadcast `chanIdx` reindex
    replaces the identity, so `pdiv_reindex` supplies the channel-indicator. -/
theorem pdiv_layerScaleCh_gamma {c h w : Nat} (x : Vec (c * h * w)) (γ : Vec c)
    (cc : Fin c) (j : Fin (c * h * w)) :
    pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j
      = if cc = chanIdx c h w j then x j else 0 := by
  have h_eq : (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x)
      = (fun y : Vec c => fun k => (fun y' : Vec c => fun k' => y' (chanIdx c h w k')) y k
                                    * (fun _ : Vec c => x) y k) := by
    funext y k; rfl
  rw [h_eq,
      pdiv_mul (fun y' : Vec c => fun k' => y' (chanIdx c h w k')) (fun _ : Vec c => x) γ
        (reindexCLM (chanIdx c h w)).differentiableAt (differentiableAt_const x) cc j,
      pdiv_reindex (chanIdx c h w) γ cc j, pdiv_const x γ cc j]
  split_ifs with h <;> ring

/-- **Per-channel layer-scale γ output, certified.** The rendered per-channel reduce
    `dγ_c = Σ_{k : chanIdx k = c} x_k·dy_k` (the `lsGradCh` emit) equals the certified Jacobian of
    `layerScaleChF`'s forward (as a function of `γ : Vec c`) contracted with the cotangent. The
    `Vec c` peer of `ConvNeXtClose.cnx_render_lsgamma_certified`; the `den` target of the (pending)
    `layerScaleChGammaSgd` core op. -/
theorem cnx_render_lsgammaCh_certified {c h w : Nat} (x : Vec (c * h * w)) (γ : Vec c)
    (dy : Vec (c * h * w)) (lr : ℝ) (cc : Fin c) :
    γ cc - lr * ∑ k : Fin (c * h * w), (if chanIdx c h w k = cc then x k * dy k else 0)
      = γ cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j * dy j := by
  congr 1
  apply congrArg (lr * ·)
  apply Finset.sum_congr rfl
  intro j _
  rw [pdiv_layerScaleCh_gamma]
  by_cases hcc : chanIdx c h w j = cc
  · rw [if_pos hcc, if_pos hcc.symm]
  · rw [if_neg hcc, if_neg (fun h => hcc h.symm)]; ring

end Proofs.CnxPoC
