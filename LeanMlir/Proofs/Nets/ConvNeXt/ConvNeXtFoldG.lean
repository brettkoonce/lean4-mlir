import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFold
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Fold
import LeanMlir.Proofs.Nets.ViT.ViTFoldG
import LeanMlir.Proofs.Nets.ResNet.ResNet34Fold

/-! # ConvNeXt-T un-fused gradient nodes at the per-example index

Three per-example lemmas that `ConvNeXtFoldGB` lifts over the batch: the per-channel layer-scale
γ gradient (`layerScaleChGammaGrad_den`) and the channel-LN γ/β gradients (`chanLnGammaGrad_den`,
`chanLnBetaGrad_den`). Each is `den`-faithful at the RAW gradient node, the one every optimizer
tail (AdamW, the clipped and weight-decayed variants, SGD) consumes. The fusion is `rfl`
(`StableHLO.lean`'s `*Sgd_eq_grad` family), so each proof is its fused peer's in `ConvNeXtFold`
with the `θ − lr·` wrapper dropped. Every Adam artifact of this net renders from the batched chain;
its fold is `ConvNeXtFoldGB`.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel layer scale — the one op kind unique to this net
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel layer-scale γ GRADIENT denotes the certified gradient.** The emitted `lsGradCh`
    reduce `dγ_c = Σ_{k : chanIdx k = c} x_k·dy_k` IS the certified Jacobian of `layerScaleChF`
    as a function of `γ : Vec c`, contracted with the cotangent. -/
theorem layerScaleChGammaGrad_den {c h w : Nat} (xN cotN : String)
    (x : Vec (c * h * w)) (γ : Vec c) (dy : Vec (c * h * w)) (cc : Fin c) :
    den (SHlo.layerScaleChGammaGrad xN x (.operand cotN dy)) cc
      = ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => layerScale (fun k => γ' (chanIdx c h w k)) x) γ cc j * dy j := by
  simp only [den, Proofs.CnxPoC.pdiv_layerScaleCh_gamma, ite_mul, zero_mul, @eq_comm _ cc]

-- ════════════════════════════════════════════════════════════════
-- § The 22 spatial LayerNorm sites — the CHANNEL-LN form the render actually emits
--   The op operands are the `[h·w, c]` transposed views the render re-emits; the certified
--   Jacobian is `chanLNTensor3`'s in the `c·h·w` activation layout, and `ConvNeXtChannelLN`'s
--   permutation argument is what lets one op serve both.
-- ════════════════════════════════════════════════════════════════

/-- **Channel-LN γ GRADIENT denotes the certified γ gradient.** All 22 spatial sites (1 stem +
    18 block + 3 downsample). -/
theorem chanLnGammaGrad_den {c h w : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (β : Vec c) (x : Vec (c * h * w)) (γ : Vec c) (cot : Vec (c * h * w)) (k : Fin c) :
    den (SHlo.veclnGammaGrad (N := h * w) (D := c) xN epsStr ε
          (chanLNRows c h w x) (.operand cotN (chanLNRows c h w cot))) k
      = ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j := by
  simp only [den]
  rw [chanLN_gamma_contract ε β γ x cot k]
  exact vit_veclnGamma_grad_bridge ε β γ (Mat.unflatten (chanLNRows c h w x))
    (chanLNRows c h w cot) k

/-- **Channel-LN β GRADIENT denotes the certified β gradient.** The β gradient is the plain row
    reduce, so the render uses the same `rowDenseBiasGrad` op ViT's LN β does. -/
theorem chanLnBetaGrad_den {c h w : Nat} (cotN : String)
    (ε : ℝ) (γ : Vec c) (x : Vec (c * h * w)) (β : Vec c) (cot : Vec (c * h * w)) (k : Fin c) :
    den (SHlo.rowDenseBiasGrad (N := h * w) (c := c)
          (.operand cotN (chanLNRows c h w cot))) k
      = ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j := by
  simp only [den]
  rw [chanLN_beta_contract ε γ β x cot k]
  exact vit_veclnBeta_grad_bridge ε γ β (Mat.unflatten (chanLNRows c h w x))
    (chanLNRows c h w cot) k

end Proofs.CnxPoCG
