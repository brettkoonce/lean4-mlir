import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Architectures.ConvGrad
import LeanMlir.Proofs.Architectures.PerChannelBNGrad

/-! # The per-example fused-SGD nodes — one `*_den` per op kind, shared by every per-example chain

The per-example train steps (the MNIST / CIFAR chapter nets, and the per-example tiers of
ResNet-34, MobileNetV2, ConvNeXt and ViT) emit fused `θ − lr·∂Loss/∂θ` ops. Each lemma here says
one such op denotes the certified SGD step at an arbitrary cotangent, so a net's fold is these at
its layers. The batched, un-fused peers are in `GradNodesB`.

| op | lemma | namespace |
|---|---|---|
| dense weight / bias (`weightSgd`, `biasSgd`) | `denseW_den`, `denseB_den` | `Cifar8PoC` |
| conv weight / bias (`convWeightSgd`, `convBiasSgd`) | `convW_den`, `convB_den` | `CifarPoC` |
| per-channel BN γ / β (`bnGammaSgd`, `bnBetaSgd`), and the pair as one clause | `bnGamma_den`, `bnBeta_den`, `BnSgdPairTied` | `CifarBnPoC` |

Namespaces are the net that first needed each op, kept so that every citation keeps its name.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Cifar8PoC

open Proofs.StableHLO Proofs.IR

/-- **Any emitted dense weight op = certified.** Generic in the layer dims, the
    activation `a`, bias `b` and cotangent `c`: `den (weightSgd a W (.operand _ c)) =
    W − lr·(certified ∂dense/∂W · c)`. Every small net's dense layers are instances. -/
theorem denseW_den {m n : Nat} (aN wN lrStr cotN : String)
    (a : Vec m) (W : Mat m n) (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin m) (j : Fin n) :
    den (SHlo.weightSgd aN wN lrStr a W lr (.operand cotN c)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ k : Fin n,
          pdiv (fun v : Vec (m*n) => dense (Mat.unflatten v) b a) (Mat.flatten W)
               (finProdFinEquiv (i, j)) k * c k := by
  have step : den (SHlo.weightSgd aN wN lrStr a W lr (.operand cotN c)) (finProdFinEquiv (i, j))
            = W i j - lr * emitWeightGrad a Back.cotangent c i j := by
    simp only [denStepApp, emitWeightGrad, Mat.outer, Back.denote, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W b a Back.cotangent c i j]; rfl

/-- **Any emitted dense bias op = certified.** Generic peer of `denseW_den`. -/
theorem denseB_den {m n : Nat} (bN lrStr cotN : String)
    (W : Mat m n) (a : Vec m) (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin n) :
    den (SHlo.biasSgd bN lrStr b lr (.operand cotN c)) i
      = b i - lr * ∑ j : Fin n,
          pdiv (fun b' : Vec n => dense W b' a) b i j * c j := by
  have step : den (SHlo.biasSgd bN lrStr b lr (.operand cotN c)) i
            = b i - lr * emitBiasGrad Back.cotangent c i := by
    simp only [denStepApp, emitBiasGrad, Back.denote]
  rw [step, bias_grad_bridge W b a Back.cotangent c i]; rfl

end Proofs.Cifar8PoC

namespace Proofs.CifarPoC

/-- **Any emitted conv weight op = certified.** Generic in the conv dims and the
    cotangent `c`: the `convWeightSgd` op denotes `flatten W − lr·(certified
    ∂conv/∂W · c)`. Instantiated at each layer's `(b,x,W,c)` it certifies W₁…W₄. -/
theorem convW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 ic h w)
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               (Kernel4.flatten W) idx j * c j :=
  cnn_render_convW_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted conv bias op = certified.** Generic peer of `convW_den`. -/
theorem convB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  cnn_render_convb_certified W x b c lr o

end Proofs.CifarPoC

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

