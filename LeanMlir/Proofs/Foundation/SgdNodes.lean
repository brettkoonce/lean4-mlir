import LeanMlir.Proofs.Codegen.StableHLO.Basic
import LeanMlir.Proofs.Architectures.ConvGrad
import LeanMlir.Proofs.Architectures.PerChannelBNGrad

/-! # The per-example fused-SGD nodes — one `*_den` per op kind, shared by every per-example chain

The per-example train steps (the MNIST / CIFAR chapter nets, and the per-example tiers of
ResNet-34, MobileNetV2, ConvNeXt and ViT) emit fused `θ − lr·∂Loss/∂θ` ops. Each lemma here says
one such op denotes the certified SGD step at an arbitrary cotangent, so a net's fold is these at
its layers. The batched, un-fused peers are in `GradNodesB`.

| op | lemma | namespace |
|---|---|---|
| dense weight / bias (`weightSgd`, `biasSgd`) | `denseW_den`, `denseB_den` | `SgdNode` |
| conv weight / bias (`convWeightSgd`, `convBiasSgd`) | `convW_den`, `convB_den` | `SgdNode` |
| per-channel BN γ / β (`bnGammaSgd`, `bnBetaSgd`), and the pair as one clause | `bnGamma_den`, `bnBeta_den`, `BnSgdPairTied` | `SgdNode` |
| conv weight / bias as tie clauses | `ConvWSgdTied`, `ConvBSgdTied`, `convWSgdTied_holds`, `convBSgdTied_holds` | `Proofs` |
| stride-1 depthwise weight / bias (`depthwiseWeightSgd`, `depthwiseBiasSgd`) | `depthwiseW_den`, `depthwiseB_den` | `SgdNode` |
| stride-2 conv weight / bias (`convStridedWeightSgd`, `convStridedBiasSgd`) | `convStridedW_den`, `convStridedB_den` | `SgdNode` |
| vector-LN γ / β (`veclnGammaSgd`, `rowDenseBiasSgd` at the LN forward), and their clauses | `veclnGammaSgd_den`, `rowDenseBiasSgd_den_lnbeta`, `VecLN{Gamma,Beta}SgdTied`, `vecLN{Gamma,Beta}SgdTied_holds` | `SgdNode` |

-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.SgdNode

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

end Proofs.SgdNode

namespace Proofs.SgdNode

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
  conv_weight_sgd_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted conv bias op = certified.** Generic peer of `convW_den`. -/
theorem convB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  conv_bias_sgd_certified W x b c lr o

end Proofs.SgdNode

namespace Proofs.SgdNode

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
  bnPerChannel_gamma_sgd_certified oc (h*w) ε γ β (reassocFwd oc h w v) (reassocFwd oc h w c) lr idx

/-- **Per-channel BN β op = certified.** Likewise `β − lr·(certified ∂BN/∂β · c)`. -/
theorem bnBeta_den {oc h w : Nat}
    (bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc)
    (v c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin oc) :
    den (SHlo.bnBetaSgd bN lrStr β lr (.operand cotN c)) idx
      = β idx - lr * ∑ j : Fin (oc*(h*w)),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ β' (reassocFwd oc h w v))
               β idx j * reassocFwd oc h w c j :=
  bnPerChannel_beta_sgd_certified oc (h*w) ε γ β (reassocFwd oc h w v) (reassocFwd oc h w c) lr idx

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

theorem bnSgdPairTied_holds {oc h w : Nat} {gN vN bN epsStr lrStr cotN : String} {ε : ℝ}
    {γ β : Vec oc} {v c : Vec (oc*h*w)} {lr : ℝ} :
    BnSgdPairTied gN vN bN epsStr lrStr cotN ε γ β v c lr :=
  ⟨bnGamma_den gN vN epsStr lrStr cotN ε γ β v c lr, bnBeta_den bN lrStr cotN ε γ β v c lr⟩

end Proofs.SgdNode

namespace Proofs

/-! Clause Props for the per-example conv ties: each is a conv `_den` lemma's statement
(`SgdNode.convW_den` / `convB_den`) under `∀`, so a tie theorem states one line per parameter
tensor and `intro` unfolds it back; each `…_holds` proves it with every argument implicit (read off
the goal by a step tie's constructor). The batched peers are `GradNodeB.ConvWTiedB` and
`EnetPoC.ConvWSgdTiedB`. -/

/-- The emitted `convWeightSgd` op, fed the cotangent `c` at the conv output, is the certified SGD
    step on the kernel `W`. -/
def ConvWSgdTied {ic oc h w kH kW : Nat} (xN wN lrStr cotN : String) (b : Vec oc)
    (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  ∀ idx : Fin (oc*ic*kH*kW),
    den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               (Kernel4.flatten W) idx j * c j

/-- The emitted `convBiasSgd` op, fed the cotangent `c` at the conv output, is the certified SGD
    step on the bias `b`. -/
def ConvBSgdTied {ic oc h w kH kW : Nat} (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW)
    (x : Tensor3 ic h w) (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) : Prop :=
  ∀ o : Fin oc,
    den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j

theorem convWSgdTied_holds {ic oc h w kH kW : Nat} {xN wN lrStr cotN : String}
    {b : Vec oc} {x : Tensor3 ic h w} {W : Kernel4 oc ic kH kW} {c : Vec (oc*h*w)} {lr : ℝ} :
    ConvWSgdTied xN wN lrStr cotN b x W c lr := fun idx => SgdNode.convW_den xN wN lrStr cotN b x W c lr idx

theorem convBSgdTied_holds {ic oc h w kH kW : Nat} {bN lrStr cotN : String}
    {W : Kernel4 oc ic kH kW} {x : Tensor3 ic h w} {b : Vec oc} {c : Vec (oc*h*w)} {lr : ℝ} :
    ConvBSgdTied bN lrStr cotN W x b c lr := fun o => SgdNode.convB_den bN lrStr cotN W x b c lr o

end Proofs

namespace Proofs.SgdNode

/-- **Flat stride-1 depthwise weight render bridge.** The emitted op's flat weight grad
    `flatten W − lr·flatten((dwconv_weight_grad₃ b x).backward W (unflatten c))` equals the flat
    pdiv-Jacobian form. Via `HasVJP3.toHasVJP.correct` (the triple→flat reindex), modulo
    `unflatten (flatten W) = W`. The stride-1 depthwise peer of `conv_weight_sgd_certified`. -/
theorem mnv2_render_depthwiseW_flat_certified {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW)
    (cot : Vec (c*h*w)) (lr : ℝ) (idx : Fin (c*kH*kW)) :
    Tensor3.flatten W idx
        - lr * Tensor3.flatten
            ((depthwiseWeightGradHasVJP3 b x).backward W (Tensor3.unflatten cot)) idx
      = Tensor3.flatten W idx - lr * ∑ j : Fin (c*h*w),
          pdiv (fun v' : Vec (c*kH*kW) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
               (Tensor3.flatten W) idx j * cot j := by
  congr 1
  congr 1
  rw [← (HasVJP3.toHasVJP (depthwiseWeightGradHasVJP3 b x)).correct (Tensor3.flatten W) cot idx]
  simp only [HasVJP3.toHasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **Stride-1 depthwise weight op = certified.** The `depthwiseWeightSgd` op denotes
    `flatten W − lr·(certified ∂(depthwiseConv2d)/∂W · c)` (flat pdiv form). The stride-1 depthwise
    peer of `SgdNode.convW_den`. -/
theorem depthwiseW_den {c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec c) (x : Tensor3 c h w)
    (W : DepthwiseKernel c kH kW) (cot : Vec (c*h*w)) (lr : ℝ) (idx : Fin (c*kH*kW)) :
    den (SHlo.depthwiseWeightSgd xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ j : Fin (c*h*w),
          pdiv (fun v' : Vec (c*kH*kW) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
               (Tensor3.flatten W) idx j * cot j := by
  show depthwiseWeightSgdDen b x W lr cot idx = _
  exact mnv2_render_depthwiseW_flat_certified b x W cot lr idx

/-- **Stride-1 depthwise bias op = certified.** Delegates to `depthwise_bias_sgd_certified`. -/
theorem depthwiseB_den {c h w kH kW : Nat}
    (bN lrStr cotN : String) (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (cot : Vec (c*h*w)) (lr : ℝ) (o : Fin c) :
    den (SHlo.depthwiseBiasSgd bN lrStr W x b lr (.operand cotN cot)) o
      = b o - lr * ∑ j : Fin (c*h*w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b o j * cot j := by
  show depthwiseBiasSgdDen W x b lr cot o = _
  exact depthwise_bias_sgd_certified W x b cot lr o

end Proofs.SgdNode

namespace Proofs.SgdNode

/-- **Any emitted STRIDED conv weight op = certified.** Generic in the conv dims, the kernel size
    (covers the 7×7 stem AND every 3×3 downsample/projection) and the cotangent `c`: the
    `convStridedWeightSgd` op denotes `flatten W − lr·(certified ∂(flatConvStride2)/∂W · c)`, the
    emitted op's `den` reduced (`rfl`) to the LHS of the generic strided weight bridge. The strided
    peer of `SgdNode.convW_den`. -/
theorem convStridedW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (ic*(2*h)*(2*w)))
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convStridedWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) => flatConvStride2 (Kernel4.unflatten v') b x)
               (Kernel4.flatten W) idx j * c j :=
  convStride2_weight_sgd_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted STRIDED conv bias op = certified.** The bias peer of `convStridedW_den`; the
    `convStridedBiasSgd` op (which emits the same `reduce` text as `convBiasSgd`) denotes
    `b − lr·(certified ∂(flatConvStride2)/∂b · c)`. -/
theorem convStridedB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w)))
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convStridedBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * c j :=
  convStride2_bias_sgd_certified W x b c lr o

end Proofs.SgdNode

namespace Proofs.SgdNode

/-! The vector-LN γ / β nodes (every LN of ViT, and the LN-style norms of ConvNeXt's per-example
tier). The β node is the per-token bias op `rowDenseBiasSgd` read against the LN forward: the LN β
gradient is the same `Σ_tokens dy` reduce as a dense bias. -/

/-- **Vector-LN γ op denotes the certified step.** `den(veclnGammaSgd)` = `γ − lr·(Σ_tokens dy·x̂)`,
    the certified ∂(rowwise vector-LN)/∂γ contraction. Covers all 25 LN-γ sites (LN1/LN2 × 12 + final).
    One-line delegation to `layerNormVec_gamma_sgd_certified` (the den's sum IS `vecLNGradGamma`). -/
theorem veclnGammaSgd_den {N D : Nat} (gN xN epsStr lrStr cotN : String)
    (ε : ℝ) (βv : Vec D) (x : Vec (N * D)) (γ : Vec D) (dy : Vec (N * D)) (lr : ℝ) (k : Fin D) :
    den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε x γ lr (.operand cotN dy)) k
      = γ k - lr * ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (Mat.unflatten x r))) γ k o * dy o := by
  simp only [denStep, denStepApp]
  exact layerNormVec_gamma_sgd_certified ε βv γ (Mat.unflatten x) dy lr k

/-- **The SAME per-token bias op, certified against the vector-LN β forward.** The LN β grad is
    `Σ_tokens dy` — identical reduce to the dense bias — so `rowDenseBiasSgd` ALSO denotes the certified
    ∂(rowwise vector-LN)/∂β contraction. Covers all 25 LN-β sites. Delegation to `layerNormVec_beta_sgd_certified`. -/
theorem rowDenseBiasSgd_den_lnbeta {N D : Nat} (bN lrStr cotN : String)
    (ε : ℝ) (γv : Vec D) (X : Mat N D) (β : Vec D) (dy : Vec (N * D)) (lr : ℝ) (i : Fin D) :
    den (SHlo.rowDenseBiasSgd bN lrStr β lr (.operand cotN dy)) i
      = β i - lr * ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o * dy o := by
  simp only [denStep, denStepApp]
  exact layerNormVec_beta_sgd_certified ε γv β X dy lr i

/-- A vector-LN γ SGD node, tied (`veclnGammaSgd_den`). -/
def VecLNGammaSgdTied (N : Nat) {D : Nat} (gN xN epsStr lrStr cotN : String) (ε : ℝ)
    (βv : Vec D) (x : Vec (N * D)) (γ : Vec D) (dy : Vec (N * D)) (lr : ℝ) : Prop :=
  ∀ k : Fin D,
    den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε x γ lr (.operand cotN dy)) k
      = γ k - lr * ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (Mat.unflatten x r))) γ k o * dy o

/-- A vector-LN β SGD node, tied (`rowDenseBiasSgd_den_lnbeta`). -/
def VecLNBetaSgdTied (N : Nat) {D : Nat} (bN lrStr cotN : String) (ε : ℝ) (γv : Vec D)
    (x : Vec (N * D)) (β : Vec D) (dy : Vec (N * D)) (lr : ℝ) : Prop :=
  ∀ i : Fin D,
    den (SHlo.rowDenseBiasSgd bN lrStr β lr (.operand cotN dy)) i
      = β i - lr * ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (Mat.unflatten x r))) β i o * dy o

theorem vecLNGammaSgdTied_holds {N D : Nat} {gN xN epsStr lrStr cotN : String} {ε : ℝ}
    {βv : Vec D} {x : Vec (N * D)} {γ : Vec D} {dy : Vec (N * D)} {lr : ℝ} :
    VecLNGammaSgdTied N gN xN epsStr lrStr cotN ε βv x γ dy lr := fun k =>
  veclnGammaSgd_den gN xN epsStr lrStr cotN ε βv x γ dy lr k

theorem vecLNBetaSgdTied_holds {N D : Nat} {bN lrStr cotN : String} {ε : ℝ} {γv : Vec D}
    {x : Vec (N * D)} {β : Vec D} {dy : Vec (N * D)} {lr : ℝ} :
    VecLNBetaSgdTied N bN lrStr cotN ε γv x β dy lr := fun i =>
  rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε γv (Mat.unflatten x) β dy lr i

end Proofs.SgdNode
