import LeanMlir.Proofs.CifarBnClose
import LeanMlir.Proofs.CifarFaithfulPoC

/-! # PoC: the CIFAR-BN (Chapter 5, per-channel BatchNorm) train step, proof-tied

The per-channel-BatchNorm peer of `CifarFaithfulPoC`: `conv→BN→relu ×4, 2 pools,
3 dense` — 22 params (4 conv kernels/biases, 4 BN scale/shift pairs γ/β, 3 dense
layers). `MainCifarBnVerified` trains on `verified_mlir/cifar_bn_train_step.mlir`.

**Reuses everything from the non-BN fold:**
* **Conv layers (W₁…W₄):** `CifarPoC.convW_den`/`convB_den` (generic, no new ops).
* **Dense head (W₅/W₆/W₇):** `CifarPoC.{dW,db}{5,6,7}_den` (same head — pool₂ input).

**New here — the BN scale/shift ops.** The per-channel γ/β updates use the new core
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

end Proofs.CifarBnPoC
