import LeanMlir.Proofs.Nets.Small.CifarBnClose
import LeanMlir.Proofs.Nets.Small.CifarFold

/-! # PoC: the per-channel BatchNorm γ/β ops, proof-tied

The per-channel-BatchNorm peer of `CifarFold`'s conv and dense folds, used by the cifar8-BN
tie (`Cifar8BnStepTie`), `ResNet34Fold` and `GradNodesB`.

**The BN scale/shift ops.** The per-channel γ/β updates use the core
ops `bnGammaSgd`/`bnBetaSgd`, whose `den` is `γ − lr·bnPerChannel_grad_gamma` /
`β − lr·bnPerChannel_grad_beta` (the certs work in the `oc·m` flat-spatial layout; the
op's `den` bridges its `oc·h·w` activation layout via `reassocFwd`, exactly as the BN
forward/back ops `bnPerChannelF`/`bnPerChannelBack` do — `bnPerChannelTensor3 =
reassocBack ∘ bnPerChannelFlat ∘ reassocFwd`). The two theorems below close them via
`cifar_bn_render_{gamma,beta}_certified` (CifarBnClose.lean) — `den` reduces (`rfl`) to
each cert's LHS.

## Honest residual
Same as the non-BN fold (conv cotangents are free vars; cotangent-subgraph⇄SHlo pin;
per-op `pretty` lexing; ℝ→Float32), plus the BN input-grad `0<ε` smoothness hypothesis
(inherited — γ/β grads themselves are affine and need no `0<ε`).
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CifarBnPoC

/-- **Per-channel BN γ op = certified.** The emitted `bnGammaSgd`, fed the BN-output
    cotangent `c` and the saved conv output `v`, denotes `γ − lr·(certified ∂(per-channel
    BN)/∂γ · c)` — via `reassocFwd` into the `oc·m` cert layout. -/
theorem bnGamma_den {oc h w : Nat}
    (gN vN epsStr lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ v lr (.operand cotN c)) idx
      = γ idx - lr * ∑ j : Fin (oc*(h*w)),
          pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' β (reassocFwd oc h w v))
               γ idx j * reassocFwd oc h w c j :=
  cifar_bn_render_gamma_certified oc (h*w) ε γ β (reassocFwd oc h w v) (reassocFwd oc h w c) lr idx

/-- **Per-channel BN β op = certified.** Likewise `β − lr·(certified ∂BN/∂β · c)`. -/
theorem bnBeta_den {oc h w : Nat}
    (bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnBetaSgd bN lrStr β lr (.operand cotN c)) idx
      = β idx - lr * ∑ j : Fin (oc*(h*w)),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ β' (reassocFwd oc h w v))
               β idx j * reassocFwd oc h w c j :=
  cifar_bn_render_beta_certified oc (h*w) ε γ β (reassocFwd oc h w v) (reassocFwd oc h w c) lr idx

/-- The emitted `bnGammaSgd` and `bnBetaSgd` ops of one per-channel BN layer, fed its BN-output
    cotangent `c` at the saved conv output `v`, are the certified SGD steps on `γ` and `β` — the
    statements of `bnGamma_den` and `bnBeta_den` under `∀`. The per-example peer of
    `EnetPoC.BnSgdPairTiedB`. -/
def BnSgdPairTied {oc h w : Nat} (gN vN bN epsStr lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v c : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  (∀ idx : Fin oc,
    den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ v lr (.operand cotN c)) idx
      = γ idx - lr * ∑ j : Fin (oc*(h*w)),
          pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' β (reassocFwd oc h w v))
               γ idx j * reassocFwd oc h w c j) ∧
  (∀ idx : Fin oc,
    den (SHlo.bnBetaSgd bN lrStr β lr (.operand cotN c)) idx
      = β idx - lr * ∑ j : Fin (oc*(h*w)),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ β' (reassocFwd oc h w v))
               β idx j * reassocFwd oc h w c j)

theorem bnSgdPairTied_holds {oc h w : Nat} (gN vN bN epsStr lrStr cotN : String) (ε : ℝ)
    (γ β : Vec oc) (v c : Vec (oc*h*w)) (lr : ℝ) :
    BnSgdPairTied gN vN bN epsStr lrStr cotN ε γ β v c lr :=
  ⟨bnGamma_den gN vN epsStr lrStr cotN ε γ β v c lr, bnBeta_den bN lrStr cotN ε γ β v c lr⟩

end Proofs.CifarBnPoC
