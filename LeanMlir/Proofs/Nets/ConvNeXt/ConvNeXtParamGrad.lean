import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB
import LeanMlir.Proofs.Foundation.ParamGradNodes

/-! # ConvNeXt-T — every parameter gradient node IS the loss's derivative in that parameter

`cnx_net_tiedGB` says each of the 182 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it, the chain's top
being the smoothed-loss cotangent. `cnx_net_lossGrad` composes that with the chain: for any loss
`L` of the logits whose gradient at the net's output is `g`, the loss of the WHOLE net `cnxNetB`
with that one parameter varied is differentiable in it and every node is its gradient
(`HasGradAt`). `cnxNetB` is the canonical forward, batched: `batchMap N` of `convNextForwardTCh`
(`cnxNetB_eq_convNextForwardTCh`). `cnx_net_lossGrad_smoothedCE` discharges `hL` for
the label-smoothed loss the artifacts ship (`smoothedBatchLossDiv`, whose gradient is the
`softmaxDiv` cotangent the render emits).

**How.** No ConvNeXt op couples examples, so the work is per example and lifted once:

* **Per example** (at variable widths): the loss read at each activation inside a block, a
  downsample, the stem and the head has the tie's own per-example cotangent as its gradient
  (`cnxBlk_hasGradAt`, `cnxDown_hasGradAt`, …). Every stage VJP is global — GELU has no kink and
  LayerNorm needs only `0 < ε` — so each step is one `HasGradAt.comp_global`, and the channel-LN
  step is `chanLNTensor3Back_eq_chanLN_vjp`.
* **Lifted** (`HasGradAt.param_batchMap_through`, ParamGrad): read against the linear loss
  `⟨·, dyₙ⟩` per example, those gradients turn each tied node's `Σ_n Σ_j ∂per/∂θ · cotₙ` into
  `∂G/∂θ` of the batched block, `G` the loss at the block's output.
* **Per net**: the loss read after each stage (`cnxSuf*`), pulled back through the certified
  batched block, downsample and head VJPs (`cnxBlockCotInB_eq_vjp`, `cnxDownCotInB_eq_vjp`,
  `cnxHeadDyB_eq_vjp`), and each `Φ` identified with the whole net at updated weights by a
  standalone `cnx_factor_*` theorem.

**Two nodes are stated differently from the tie.** The stem's bias node is emitted as a stride-1
`convBiasGradB` over a free `xstem`; its Jacobian in the bias is the channel indicator whatever the
conv, so it equals the patchify conv's (`GradNodeB.pdiv_bias_of_split`). The classifier bias node
`biasGradB` is the identity on its operand and the batch reduce is emitted text, so the statement
is the sum over the batch of the node's per-example slices.

**Hypotheses.** `0 < ε` (the LayerNorms' VJPs); no smoothness hypothesis. For the smoothed loss,
every example's target sums to one and `0 < nC`. Drop-path and the bf16 nodes are outside this
statement, as they are outside the tie.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTiePoCGB

open scoped BigOperators
open Proofs.CnxTiePoC (cnxStemFwdO cnxBlockFwdChO cnxDownFwdChO cnxBlockCotInChAt cnxDownCotInChAt
  CnxTieWeights CnxTieBlk CnxTieDown)

-- ════════════════════════════════════════════════════════════════
-- § Parameter differentiability of the ConvNeXt-specific ops
-- ════════════════════════════════════════════════════════════════

theorem rowLNVecFlat_gamma_differentiable (s c : Nat) (ε : ℝ) (β : Vec c) (x : Vec (s * c)) :
    Differentiable ℝ (fun γ : Vec c => rowLNVecFlat s c ε γ β x) := by
  unfold rowLNVecFlat Mat.flatten layerNormVec; fun_prop

theorem rowLNVecFlat_beta_differentiable (s c : Nat) (ε : ℝ) (γ : Vec c) (x : Vec (s * c)) :
    Differentiable ℝ (fun β : Vec c => rowLNVecFlat s c ε γ β x) := by
  unfold rowLNVecFlat Mat.flatten layerNormVec; fun_prop

theorem chanLNTensor3_gamma_differentiable (c h w : Nat) (ε : ℝ) (β : Vec c)
    (x : Vec (c * h * w)) : Differentiable ℝ (fun γ : Vec c => chanLNTensor3 c h w ε γ β x) := by
  unfold chanLNTensor3
  exact (reassocBack_differentiable c h w).comp ((transposeFlat_differentiable (h * w) c).comp
    (rowLNVecFlat_gamma_differentiable (h * w) c ε β _))

theorem chanLNTensor3_beta_differentiable (c h w : Nat) (ε : ℝ) (γ : Vec c)
    (x : Vec (c * h * w)) : Differentiable ℝ (fun β : Vec c => chanLNTensor3 c h w ε γ β x) := by
  unfold chanLNTensor3
  exact (reassocBack_differentiable c h w).comp ((transposeFlat_differentiable (h * w) c).comp
    (rowLNVecFlat_beta_differentiable (h * w) c ε γ _))

theorem layerScaleCh_gamma_differentiable (c h w : Nat) (x : Vec (c * h * w)) :
    Differentiable ℝ (fun γ : Vec c => layerScale (fun k => γ (chanIdx c h w k)) x) := by
  unfold layerScale; fun_prop

theorem flatConvStride4_weight_differentiable {ic oc h w kH kW : Nat} (b : Vec oc)
    (y : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) :
    Differentiable ℝ (fun θ : Vec (oc * ic * kH * kW) =>
      (flatConvStride4 (Kernel4.unflatten θ) b y : Vec (oc * h * w))) := by
  unfold flatConvStride4
  exact (decimateFlat_differentiable oc h w).comp
    ((decimateOddFlat_differentiable oc (2 * h) (2 * w)).comp
      (conv2d_weight_differentiable (h := 2 * (2 * h)) (w := 2 * (2 * w)) b (Tensor3.unflatten y)))

theorem flatConvStride4_bias_differentiable {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (y : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) :
    Differentiable ℝ (fun θ : Vec oc => (flatConvStride4 W θ y : Vec (oc * h * w))) := by
  unfold flatConvStride4
  exact (decimateFlat_differentiable oc h w).comp
    ((decimateOddFlat_differentiable oc (2 * h) (2 * w)).comp
      (conv2d_bias_differentiable (h := 2 * (2 * h)) (w := 2 * (2 * w)) W (Tensor3.unflatten y)))

-- ════════════════════════════════════════════════════════════════
-- § A ConvNeXt block, per example
--   xin → d (depthwise) → nl (channel LN) → e (expand) → g (GELU) → p (project) → layer scale → + xin
--   `cnxPost*` is the rest of the block after each activation, as a function of it.
-- ════════════════════════════════════════════════════════════════

section Block
variable {c cExp : Nat}

/-- The block after the project conv: layer scale, then the identity skip. -/
noncomputable def cnxPostP (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun u i => layerScale (fun k => p.sL (chanIdx c h w k)) u i + y i

/-- The block after the expand conv (pre-GELU). -/
noncomputable def cnxPostE (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (cExp * h * w) → Vec (c * h * w) :=
  fun u => cnxPostP h w p y (flatConv (h := h) (w := w) p.pW p.pB (gelu (cExp * h * w) u))

/-- The block after the channel LN. -/
noncomputable def cnxPostN (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun u => cnxPostE h w p y (flatConv (h := h) (w := w) p.eW p.eB u)

/-- The block after the depthwise conv. -/
noncomputable def cnxPostD (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun u => cnxPostN h w p y (chanLNTensor3 c h w ε p.nG p.nB u)

/-- The depthwise conv's output. -/
noncomputable def cnxActD (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) :=
  depthwiseFlat (h := h) (w := w) p.aW p.aB y

/-- The channel LN's output. -/
noncomputable def cnxActNl (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) :=
  chanLNTensor3 c h w ε p.nG p.nB (cnxActD h w p y)

/-- The GELU's output (the project conv's input). -/
noncomputable def cnxActG (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (cExp * h * w) :=
  gelu (cExp * h * w) (flatConv (h := h) (w := w) p.eW p.eB (cnxActNl h w ε p y))

/-- The project conv's output (the layer scale's input). -/
noncomputable def cnxActP (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) :=
  flatConv (h := h) (w := w) p.pW p.pB (cnxActG h w ε p y)

theorem cnxPostP_differentiable (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Differentiable ℝ (cnxPostP h w p y) := by
  unfold cnxPostP layerScale; fun_prop

theorem cnxPostE_differentiable (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Differentiable ℝ (cnxPostE h w p y) :=
  (cnxPostP_differentiable h w p y).comp
    ((flatConv_differentiable p.pW p.pB).comp (gelu_differentiable _))

theorem cnxPostN_differentiable (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Differentiable ℝ (cnxPostN h w p y) :=
  (cnxPostE_differentiable h w p y).comp (flatConv_differentiable p.eW p.eB)

theorem cnxPostD_differentiable (h w : Nat) (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (y : Vec (c * h * w)) : Differentiable ℝ (cnxPostD h w ε p y) :=
  (cnxPostN_differentiable h w p y).comp (chanLNTensor3_differentiable c h w ε p.nG p.nB hε)

/-- **A block's cotangents are loss gradients**, per example: from the gradient `dy` at the block
    output, the loss read after each activation has the tie's cotangent there — `dy` at the layer
    scale's output, `cnxCotP` at the project conv's, then `blkCotE`, `blkCotN`, `blkCotD`. -/
theorem cnxBlk_hasGradAt {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (y dy : Vec (c * h * w)) {G : Vec (c * h * w) → Vec 1}
    (hG : HasGradAt G (p.fwdO (h := h) (w := w) ε y) dy) :
    HasGradAt (fun u => G (fun i => u i + y i))
        (layerScale (fun k => p.sL (chanIdx c h w k)) (cnxActP h w ε p y)) dy
      ∧ HasGradAt (fun u => G (cnxPostP h w p y u)) (cnxActP h w ε p y)
          (cnxCotP (fun k => p.sL (chanIdx c h w k)) dy)
      ∧ HasGradAt (fun u => G (cnxPostE h w p y u))
          (flatConv (h := h) (w := w) p.eW p.eB (cnxActNl h w ε p y))
          (blkCotE (h := h) (w := w) ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
      ∧ HasGradAt (fun u => G (cnxPostN h w p y u)) (cnxActNl h w ε p y)
          (blkCotN (h := h) (w := w) ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
      ∧ HasGradAt (fun u => G (cnxPostD h w ε p y u)) (cnxActD h w p y)
          (blkCotD (h := h) (w := w) ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy) := by
  have hL : HasGradAt (fun u => G (fun i => u i + y i))
      (layerScale (fun k => p.sL (chanIdx c h w k)) (cnxActP h w ε p y)) dy :=
    HasGradAt.comp (f := fun u i => u i + y i)
      (x := layerScale (fun k => p.sL (chanIdx c h w k)) (cnxActP h w ε p y)) hG
      (differentiableAt_id.add_const y)
      (addConstHasVJPAt (fun u => u) y _ differentiableAt_id (identityHasVJPAt _ _))
  have hP : HasGradAt (fun u => G (cnxPostP h w p y u)) (cnxActP h w ε p y)
      (cnxCotP (fun k => p.sL (chanIdx c h w k)) dy) :=
    HasGradAt.comp_global (f := layerScale (fun k => p.sL (chanIdx c h w k)))
      (x := cnxActP h w ε p y) hL (layerScale_differentiable _) (layerScaleHasVJP _)
  have hGg := HasGradAt.comp_global (f := flatConv (h := h) (w := w) p.pW p.pB)
    (x := cnxActG h w ε p y) hP (flatConv_differentiable p.pW p.pB) (flatConvHasVJP p.pW p.pB)
  have hE := HasGradAt.comp_global (f := gelu (cExp * h * w))
    (x := flatConv (h := h) (w := w) p.eW p.eB (cnxActNl h w ε p y)) hGg (gelu_differentiable _)
    (geluHasVJP _)
  have hN := HasGradAt.comp_global (f := flatConv (h := h) (w := w) p.eW p.eB)
    (x := cnxActNl h w ε p y) hE (flatConv_differentiable p.eW p.eB) (flatConvHasVJP p.eW p.eB)
  have hD := (HasGradAt.comp_global (f := chanLNTensor3 c h w ε p.nG p.nB) (x := cnxActD h w p y)
    hN (chanLNTensor3_differentiable c h w ε p.nG p.nB hε)
    (chanLNTensor3HasVJP c h w ε p.nG p.nB hε)).of_eq
    (congrFun (chanLNTensor3Back_eq_chanLN_vjp ε hε p.nG p.nB (cnxActD h w p y)) _).symm
  exact ⟨hL, hP, hE, hN, hD⟩

/-- **ConvNeXt block, every parameter node a loss derivative** — the nine nodes
    `cnxBlockChTiedGB` ties, at the tie's batched activations and cotangents, `Φ` the loss at the
    block's output as a function of the block's weight record. -/
def cnxBlockLossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : CnxTieBlk c cExp) (xin : Vec (N * (c * h * w))) (Φ : CnxTieBlk c cExp → Vec 1)
    (dyOut : Vec (N * (c * h * w))) : Prop :=
  let γlsB : Vec (c * h * w) := fun k => p.sL (chanIdx c h w k)
  -- forward activations
  let dB  : Vec (N * (c * h * w))    := batchMap N (depthwiseFlat (h := h) (w := w) p.aW p.aB) xin
  let nlB : Vec (N * (c * h * w))    := batchMap N (chanLNTensor3 c h w ε p.nG p.nB) dB
  let gB  : Vec (N * (cExp * h * w)) :=
    batchMap N (fun nl => gelu (cExp * h * w) (flatConv (h := h) (w := w) p.eW p.eB nl)) nlB
  let pB  : Vec (N * (c * h * w))    := batchMap N (flatConv (h := h) (w := w) p.pW p.pB) gB
  -- backward chain cotangents
  let cotPB : Vec (N * (c * h * w))    := batchMap N (cnxCotP γlsB) dyOut
  let cotEB : Vec (N * (cExp * h * w)) :=
    batchMapAux N (blkCotE ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin dyOut
  let cotNB : Vec (N * (c * h * w))    :=
    batchMapAux N (blkCotN ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin dyOut
  let cotDB : Vec (N * (c * h * w))    :=
    batchMapAux N (blkCotD ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin dyOut
  -- depthwise 7×7 W/b
  (HasGradAt (fun θ => Φ { p with aW := Tensor3.unflatten θ }) (Tensor3.flatten p.aW)
        (den (SHlo.depthwiseWeightGradB xN p.aB xin p.aW (.operand cotN cotDB))))
  ∧ (HasGradAt (fun θ => Φ { p with aB := θ }) p.aB
        (den (SHlo.depthwiseBiasGradB p.aW xin p.aB (.operand cotN cotDB))))
  -- channel-LN γ/β
  ∧ (HasGradAt (fun θ => Φ { p with nG := θ }) p.nG
        (den (SHlo.veclnGammaGradB (N := N) (R := h * w) (D := c) xN epsStr ε
          (batchMap N (chanLNRows c h w) dB)
          (.operand cotN (batchMap N (chanLNRows c h w) cotNB)))))
  ∧ (HasGradAt (fun θ => Φ { p with nB := θ }) p.nB
        (den (SHlo.rowDenseBiasGradB (N := N) (R := h * w) (c := c)
          (.operand cotN (batchMap N (chanLNRows c h w) cotNB)))))
  -- expand 1×1 conv W/b
  ∧ (HasGradAt (fun θ => Φ { p with eW := Kernel4.unflatten θ }) (Kernel4.flatten p.eW)
        (den (SHlo.convWeightGradB xN p.eB nlB p.eW (.operand cotN cotEB))))
  ∧ (HasGradAt (fun θ => Φ { p with eB := θ }) p.eB
        (den (SHlo.convBiasGradB (h := h) (w := w) p.eW nlB p.eB (.operand cotN cotEB))))
  -- project 1×1 conv W/b
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradB xN p.pB gB p.pW (.operand cotN cotPB))))
  ∧ (HasGradAt (fun θ => Φ { p with pB := θ }) p.pB
        (den (SHlo.convBiasGradB (h := h) (w := w) p.pW gB p.pB (.operand cotN cotPB))))
  -- per-channel layer-scale γ
  ∧ (HasGradAt (fun θ => Φ { p with sL := θ }) p.sL
        (den (SHlo.layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) xN pB
          (.operand cotN dyOut))))

theorem cnx_block_lossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (hε : 0 < ε) (p : CnxTieBlk c cExp) (xin : Vec (N * (c * h * w)))
    {Lb : Vec (N * (c * h * w)) → Vec 1} {dyOut : Vec (N * (c * h * w))}
    (hLb : HasGradAt Lb (batchMap N (p.fwdO (h := h) (w := w) ε) xin) dyOut)
    {Φ : CnxTieBlk c cExp → Vec 1}
    (hΦ : ∀ p', Φ p' = Lb (batchMap N (p'.fwdO (h := h) (w := w) ε) xin)) :
    cnxBlockLossTiedGB N xN epsStr cotN ε p xin Φ dyOut := by
  rw [show Φ = fun p' => Lb (batchMap N (p'.fwdO (h := h) (w := w) ε) xin) from funext hΦ]
  have hc := fun y dy => cnxBlk_hasGradAt (h := h) (w := w) ε hε p y dy (hasGradAt_linLoss dy _)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => depthwiseFlat (h := h) (w := w) (Tensor3.unflatten θ : DepthwiseKernel c 7 7) p.aB y)
        (cnxPostD h w ε p) (fun y dy => blkCotD ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
        xin (θ := Tensor3.flatten p.aW) (by rw [Tensor3.unflatten_flatten]; exact hLb)
        (fun y => (depthwise_weight_differentiable p.aB (Tensor3.unflatten y)) _)
        (fun y => cnxPostD_differentiable h w ε hε p y)
        (fun y dy => by rw [Tensor3.unflatten_flatten]; exact (hc y dy).2.2.2.2)
        xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun idx => (GradNodeB.depthwiseWGradB_den xN cotN p.aB xin p.aW _ idx).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => depthwiseFlat (h := h) (w := w) p.aW θ y)
        (cnxPostD h w ε p) (fun y dy => blkCotD ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
        xin (θ := p.aB) hLb
        (fun y => (depthwise_bias_differentiable p.aW (Tensor3.unflatten y)) _)
        (fun y => cnxPostD_differentiable h w ε hε p y)
        (fun y dy => (hc y dy).2.2.2.2)
        xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun o => (GradNodeB.depthwiseBGradB_den cotN p.aW xin p.aB _ o).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActD h w p y)
        (fun θ d => chanLNTensor3 c h w ε θ p.nB d)
        (cnxPostN h w p) (fun y dy => blkCotN ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
        xin (θ := p.nG) hLb
        (fun y => (chanLNTensor3_gamma_differentiable c h w ε p.nB y) _)
        (fun y => cnxPostN_differentiable h w p y)
        (fun y dy => (hc y dy).2.2.2.1)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun k => (CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε p.nB _ p.nG _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActD h w p y)
        (fun θ d => chanLNTensor3 c h w ε p.nG θ d)
        (cnxPostN h w p) (fun y dy => blkCotN ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
        xin (θ := p.nB) hLb
        (fun y => (chanLNTensor3_beta_differentiable c h w ε p.nG y) _)
        (fun y => cnxPostN_differentiable h w p y)
        (fun y dy => (hc y dy).2.2.2.1)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun k => (CnxPoCGB.chanLnBetaGradB_den cotN ε p.nG _ p.nB _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActNl h w ε p y)
        (fun θ nl => flatConv (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 cExp c 1 1) p.eB nl)
        (cnxPostE h w p) (fun y dy => blkCotE ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
        xin (θ := Kernel4.flatten p.eW) (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (conv2d_weight_differentiable p.eB (Tensor3.unflatten y)) _)
        (fun y => cnxPostE_differentiable h w p y)
        (fun y dy => by rw [Kernel4.unflatten_flatten]; exact (hc y dy).2.2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun idx => (GradNodeB.convWGradB_den xN cotN p.eB _ p.eW _ idx).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActNl h w ε p y)
        (fun θ nl => flatConv (h := h) (w := w) p.eW θ nl)
        (cnxPostE h w p) (fun y dy => blkCotE ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y dy)
        xin (θ := p.eB) hLb
        (fun y => (conv2d_bias_differentiable p.eW (Tensor3.unflatten y)) _)
        (fun y => cnxPostE_differentiable h w p y)
        (fun y dy => (hc y dy).2.2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun o => (GradNodeB.convBGradB_den cotN p.eW _ p.eB _ o).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActG h w ε p y)
        (fun θ g => flatConv (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 c cExp 1 1) p.pB g)
        (cnxPostP h w p) (fun _ dy => cnxCotP (fun k => p.sL (chanIdx c h w k)) dy)
        xin (θ := Kernel4.flatten p.pW) (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (conv2d_weight_differentiable p.pB (Tensor3.unflatten y)) _)
        (fun y => cnxPostP_differentiable h w p y)
        (fun y dy => by rw [Kernel4.unflatten_flatten]; exact (hc y dy).2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        (fun n => batchSlice_batchMap _ _ n)).of_eq
      (funext fun idx => (GradNodeB.convWGradB_den xN cotN p.pB _ p.pW _ idx).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActG h w ε p y)
        (fun θ g => flatConv (h := h) (w := w) p.pW θ g)
        (cnxPostP h w p) (fun _ dy => cnxCotP (fun k => p.sL (chanIdx c h w k)) dy)
        xin (θ := p.pB) hLb
        (fun y => (conv2d_bias_differentiable p.pW (Tensor3.unflatten y)) _)
        (fun y => cnxPostP_differentiable h w p y)
        (fun y dy => (hc y dy).2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        (fun n => batchSlice_batchMap _ _ n)).of_eq
      (funext fun o => (GradNodeB.convBGradB_den cotN p.pW _ p.pB _ o).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => cnxActP h w ε p y)
        (fun θ u => layerScale (fun k => θ (chanIdx c h w k)) u)
        (fun y u i => u i + y i) (fun _ dy => dy)
        xin (θ := p.sL) hLb
        (fun y => (layerScaleCh_gamma_differentiable c h w y) _)
        (fun y => differentiable_id.add_const y)
        (fun y dy => (hc y dy).1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl) (fun _ => rfl)).of_eq
      (funext fun cc => (CnxPoCGB.layerScaleChGammaGradB_den xN cotN _ p.sL _ cc).symm)

end Block

-- ════════════════════════════════════════════════════════════════
-- § A downsample, per example — channel LN at `2h × 2w`, then the 2×2/s2 conv
-- ════════════════════════════════════════════════════════════════

section Down
variable {ci co : Nat}

/-- **Downsample, every parameter node a loss derivative** — the four nodes `cnxDownChTiedGB`
    ties. -/
def cnxDownLossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : CnxTieDown ci co) (xin : Vec (N * (ci * (2 * h) * (2 * w))))
    (Φ : CnxTieDown ci co → Vec 1) (dyOut : Vec (N * (co * h * w))) : Prop :=
  let nB : Vec (N * (ci * (2 * h) * (2 * w))) :=
    batchMap N (chanLNTensor3 ci (2 * h) (2 * w) ε p.G p.T) xin
  let cotNB : Vec (N * (ci * (2 * h) * (2 * w))) :=
    batchMapAux N (dnCotN (h := h) (w := w) ε p.G p.T p.W p.B) xin dyOut
  (HasGradAt (fun θ => Φ { p with G := θ }) p.G
        (den (SHlo.veclnGammaGradB (N := N) (R := (2 * h) * (2 * w)) (D := ci) xN epsStr ε
          (batchMap N (chanLNRows ci (2 * h) (2 * w)) xin)
          (.operand cotN (batchMap N (chanLNRows ci (2 * h) (2 * w)) cotNB)))))
  ∧ (HasGradAt (fun θ => Φ { p with T := θ }) p.T
        (den (SHlo.rowDenseBiasGradB (N := N) (R := (2 * h) * (2 * w)) (c := ci)
          (.operand cotN (batchMap N (chanLNRows ci (2 * h) (2 * w)) cotNB)))))
  ∧ (HasGradAt (fun θ => Φ { p with W := Kernel4.unflatten θ }) (Kernel4.flatten p.W)
        (den (SHlo.convStridedWeightGradB xN p.B nB p.W (.operand cotN dyOut))))
  ∧ (HasGradAt (fun θ => Φ { p with B := θ }) p.B
        (den (SHlo.convStridedBiasGradB (h := h) (w := w) p.W nB p.B (.operand cotN dyOut))))

theorem cnx_down_lossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : CnxTieDown ci co) (xin : Vec (N * (ci * (2 * h) * (2 * w))))
    {Lb : Vec (N * (co * h * w)) → Vec 1} {dyOut : Vec (N * (co * h * w))}
    (hLb : HasGradAt Lb (batchMap N (p.fwdO (h := h) (w := w) ε) xin) dyOut)
    {Φ : CnxTieDown ci co → Vec 1}
    (hΦ : ∀ p', Φ p' = Lb (batchMap N (p'.fwdO (h := h) (w := w) ε) xin)) :
    cnxDownLossTiedGB N xN epsStr cotN ε p xin Φ dyOut := by
  rw [show Φ = fun p' => Lb (batchMap N (p'.fwdO (h := h) (w := w) ε) xin) from funext hΦ]
  have hc : ∀ (y : Vec (ci * (2 * h) * (2 * w))) (dy : Vec (co * h * w)),
      HasGradAt (fun u => linLoss dy (flatConvStride2 (h := h) (w := w) p.W p.B u))
        (chanLNTensor3 ci (2 * h) (2 * w) ε p.G p.T y) (dnCotN ε p.G p.T p.W p.B y dy) :=
    fun y dy => HasGradAt.comp_global (f := flatConvStride2 (h := h) (w := w) p.W p.B)
      (x := chanLNTensor3 ci (2 * h) (2 * w) ε p.G p.T y) (hasGradAt_linLoss dy _)
      (flatConvStride2_differentiable p.W p.B) (flatConvStride2HasVJP p.W p.B)
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => chanLNTensor3 ci (2 * h) (2 * w) ε θ p.T y)
        (fun _ u => flatConvStride2 (h := h) (w := w) p.W p.B u)
        (fun y dy => dnCotN ε p.G p.T p.W p.B y dy) xin (θ := p.G) hLb
        (fun y => (chanLNTensor3_gamma_differentiable ci (2 * h) (2 * w) ε p.T y) _)
        (fun _ => flatConvStride2_differentiable p.W p.B) hc
        xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun k => (CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε p.T xin p.G _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => chanLNTensor3 ci (2 * h) (2 * w) ε p.G θ y)
        (fun _ u => flatConvStride2 (h := h) (w := w) p.W p.B u)
        (fun y dy => dnCotN ε p.G p.T p.W p.B y dy) xin (θ := p.T) hLb
        (fun y => (chanLNTensor3_beta_differentiable ci (2 * h) (2 * w) ε p.G y) _)
        (fun _ => flatConvStride2_differentiable p.W p.B) hc
        xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun k => (CnxPoCGB.chanLnBetaGradB_den cotN ε p.G xin p.T _ k).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => chanLNTensor3 ci (2 * h) (2 * w) ε p.G p.T y)
        (fun θ n => flatConvStride2 (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 co ci 2 2) p.B n)
        (fun _ z => z) (fun _ dy => dy) xin (θ := Kernel4.flatten p.W)
        (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (GradNodeB.flatConvStride2_weight_differentiable p.B y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)).of_eq
      (funext fun idx => (GradNodeB.convStridedWGradB_den xN cotN p.B _ p.W _ idx).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => chanLNTensor3 ci (2 * h) (2 * w) ε p.G p.T y)
        (fun θ n => flatConvStride2 (h := h) (w := w) p.W θ n)
        (fun _ z => z) (fun _ dy => dy) xin (θ := p.B) hLb
        (fun y => (GradNodeB.flatConvStride2_bias_differentiable p.W y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)).of_eq
      (funext fun o => (GradNodeB.convStridedBGradB_den cotN p.W _ p.B _ o).symm)

end Down

-- ════════════════════════════════════════════════════════════════
-- § The stem — 4×4/s4 patchify conv, then channel LN
-- ════════════════════════════════════════════════════════════════

section Stem
variable {c : Nat}

/-- **Stem, every parameter node a loss derivative** — the four nodes `cnxStemChTiedGB` ties. The
    bias node is the emitted stride-1 `convBiasGradB` over a free `xstem`, as in the tie. -/
def cnxStemLossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (x : Vec (N * (3 * (2 * (2 * h)) * (2 * (2 * w)))))
    (xstem : Vec (N * (3 * h * w))) (Φ : Kernel4 c 3 4 4 → Vec c → Vec c → Vec c → Vec 1)
    (dyStem : Vec (N * (c * h * w))) : Prop :=
  let patchB : Vec (N * (c * h * w)) := batchMap N (flatConvStride4 Wst psb) x
  let cotPatchB : Vec (N * (c * h * w)) :=
    batchMapAux N (stemCotPatch (h := h) (w := w) ε Wst psb psng) x dyStem
  (HasGradAt (fun θ => Φ Wst psb θ psnbt) psng
        (den (SHlo.veclnGammaGradB (N := N) (R := h * w) (D := c) xN epsStr ε
          (batchMap N (chanLNRows c h w) patchB)
          (.operand cotN (batchMap N (chanLNRows c h w) dyStem)))))
  ∧ (HasGradAt (fun θ => Φ Wst psb psng θ) psnbt
        (den (SHlo.rowDenseBiasGradB (N := N) (R := h * w) (c := c)
          (.operand cotN (batchMap N (chanLNRows c h w) dyStem)))))
  ∧ (HasGradAt (fun θ => Φ Wst θ psng psnbt) psb
        (den (SHlo.convBiasGradB (N := N) (ic := 3) (oc := c) (h := h) (w := w) (kH := 4) (kW := 4)
          Wst xstem psb (.operand cotN cotPatchB))))
  ∧ (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) psb psng psnbt) (Kernel4.flatten Wst)
        (den (SHlo.convStride4WeightGradB xN psb x Wst (.operand cotN cotPatchB))))

theorem cnx_stem_lossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ) (hε : 0 < ε)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (x : Vec (N * (3 * (2 * (2 * h)) * (2 * (2 * w)))))
    (xstem : Vec (N * (3 * h * w))) {Lb : Vec (N * (c * h * w)) → Vec 1}
    {dyStem : Vec (N * (c * h * w))}
    (hLb : HasGradAt Lb (batchMap N (cnxStemFwdO (h := h) (w := w) ε Wst psb psng psnbt) x) dyStem)
    {Φ : Kernel4 c 3 4 4 → Vec c → Vec c → Vec c → Vec 1}
    (hΦ : ∀ W b γ β, Φ W b γ β = Lb (batchMap N (cnxStemFwdO (h := h) (w := w) ε W b γ β) x)) :
    cnxStemLossTiedGB N xN epsStr cotN ε Wst psb psng psnbt x xstem Φ dyStem := by
  rw [show Φ = fun W b γ β => Lb (batchMap N (cnxStemFwdO (h := h) (w := w) ε W b γ β) x) from
    funext fun W => funext fun b => funext fun γ => funext fun β => hΦ W b γ β]
  have hc : ∀ (y : Vec (3 * (2 * (2 * h)) * (2 * (2 * w)))) (dy : Vec (c * h * w)),
      HasGradAt (fun u => linLoss dy (chanLNTensor3 c h w ε psng psnbt u))
        (flatConvStride4 Wst psb y) (stemCotPatch ε Wst psb psng y dy) := fun y dy =>
    (HasGradAt.comp_global (f := chanLNTensor3 c h w ε psng psnbt) (x := flatConvStride4 Wst psb y)
      (hasGradAt_linLoss dy _) (chanLNTensor3_differentiable c h w ε psng psnbt hε)
      (chanLNTensor3HasVJP c h w ε psng psnbt hε)).of_eq
      (congrFun (chanLNTensor3Back_eq_chanLN_vjp ε hε psng psnbt _) dy).symm
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact (HasGradAt.param_batchMap_through (fun y => flatConvStride4 (h := h) (w := w) Wst psb y)
        (fun θ u => chanLNTensor3 c h w ε θ psnbt u) (fun _ z => z) (fun _ dy => dy) x (θ := psng)
        hLb (fun y => (chanLNTensor3_gamma_differentiable c h w ε psnbt y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)).of_eq
      (funext fun k => (CnxPoCGB.chanLnGammaGradB_den xN epsStr cotN ε psnbt _ psng _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => flatConvStride4 (h := h) (w := w) Wst psb y)
        (fun θ u => chanLNTensor3 c h w ε psng θ u) (fun _ z => z) (fun _ dy => dy) x (θ := psnbt)
        hLb (fun y => (chanLNTensor3_beta_differentiable c h w ε psng y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)).of_eq
      (funext fun k => (CnxPoCGB.chanLnBetaGradB_den cotN ε psng _ psnbt _ k).symm)
  · -- the emitted bias node reads the channel sum; any conv's bias Jacobian is the indicator
    refine (HasGradAt.param_batchMap_through (fun y => y)
      (fun θ y => flatConvStride4 (h := h) (w := w) Wst θ y)
      (fun _ u => chanLNTensor3 c h w ε psng psnbt u) (fun y dy => stemCotPatch ε Wst psb psng y dy)
      x (θ := psb) hLb (fun y => (flatConvStride4_bias_differentiable Wst y) _)
      (fun _ => chanLNTensor3_differentiable c h w ε psng psnbt hε) hc
      x _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq (funext fun o => ?_)
    have hb : ∀ n j,
        pdiv (fun b' : Vec c => Tensor3.flatten (conv2d Wst b'
            (Tensor3.unflatten (batchSlice N (3 * h * w) xstem n)))) psb o j
          = pdiv (fun b' : Vec c => (flatConvStride4 Wst b'
              (batchSlice N (3 * (2 * (2 * h)) * (2 * (2 * w))) x n) : Vec (c * h * w))) psb o j :=
      fun n j => (GradNodeB.pdiv_bias_of_split (fun θ y => flatConv (h := h) (w := w) Wst θ y)
          (GradNodeB.flatConv_bias_split Wst) _ psb o j).trans
        (GradNodeB.pdiv_bias_of_split (fun θ y => flatConvStride4 (h := h) (w := w) Wst θ y)
          (GradNodeB.flatConvStride4_bias_split Wst) _ psb o j).symm
    refine (Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl fun j _ =>
      congrArg (· * _) (hb n j)).symm.trans ?_
    exact (GradNodeB.convBGradB_den cotN Wst xstem psb _ o).symm
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => flatConvStride4 (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 c 3 4 4) psb y)
        (fun _ u => chanLNTensor3 c h w ε psng psnbt u) (fun y dy => stemCotPatch ε Wst psb psng y dy)
        x (θ := Kernel4.flatten Wst) (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (flatConvStride4_weight_differentiable psb y) _)
        (fun _ => chanLNTensor3_differentiable c h w ε psng psnbt hε)
        (fun y dy => by rw [Kernel4.unflatten_flatten]; exact hc y dy)
        x _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun idx => (GradNodeB.psWGradB_den xN cotN psb x Wst _ idx).symm)

end Stem

-- ════════════════════════════════════════════════════════════════
-- § The head — GAP, the head LN at one row, the dense classifier
-- ════════════════════════════════════════════════════════════════

section Head

/-- The head per example: GAP, then LayerNorm at one row, then the dense classifier. -/
noncomputable def cnxHeadO (h w : Nat) {nC : Nat} (ε : ℝ) (hng hnbt : Vec 768) (Wfc : Mat 768 nC)
    (bfc : Vec nC) : Vec (768 * h * w) → Vec nC :=
  dense Wfc bfc ∘ rowLNVecFlat 1 768 ε hng hnbt ∘ globalAvgPoolFlat 768 h w

/-- **Head, every parameter node a loss derivative** — the four nodes `cnxHeadChTiedGB` ties. The
    classifier bias node is the identity on its operand (the batch reduce is emitted text), so its
    statement is the batch sum of the node's slices. -/
def cnxHeadLossTiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC) (xhead : Vec (N * (768 * h * w)))
    (Φ : Vec 768 → Vec 768 → Mat 768 nC → Vec nC → Vec 1) (g : Vec (N * nC)) : Prop :=
  let gapB   : Vec (N * (1 * 768)) := batchMap N (globalAvgPoolFlat 768 h w) xhead
  let hnB    : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε hng hnbt) gapB
  let cotHnB : Vec (N * (1 * 768)) := batchMapAux N (headCotHn Wfc bfc) hnB g
  (HasGradAt (fun θ => Φ θ hnbt Wfc bfc) hng
        (den (SHlo.veclnGammaGradB (N := N) (R := 1) (D := 768) xN epsStr ε gapB
          (.operand cotN cotHnB))))
  ∧ (HasGradAt (fun θ => Φ hng θ Wfc bfc) hnbt
        (den (SHlo.rowDenseBiasGradB (N := N) (R := 1) (c := 768) (.operand cotN cotHnB))))
  ∧ HasGradAt (fun θ => Φ hng hnbt (Mat.unflatten θ) bfc) (Mat.flatten Wfc)
      (den (SHlo.weightGradB (N := N) (m := 768) (n := nC) dN hnB (.operand cotN g)))
  ∧ HasGradAt (fun θ => Φ hng hnbt Wfc θ) bfc
      (fun i => ∑ n : Fin N,
        batchSlice N nC (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g))) n i)

theorem cnx_head_lossTiedGB (N : Nat) {h w nC : Nat} (xN epsStr cotN dN : String) (ε : ℝ)
    (hng hnbt : Vec 768) (Wfc : Mat 768 nC) (bfc : Vec nC) (xhead : Vec (N * (768 * h * w)))
    {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)}
    (hL : HasGradAt L (batchMap N (cnxHeadO h w ε hng hnbt Wfc bfc) xhead) g)
    {Φ : Vec 768 → Vec 768 → Mat 768 nC → Vec nC → Vec 1}
    (hΦ : ∀ a b W bb, Φ a b W bb = L (batchMap N (cnxHeadO h w ε a b W bb) xhead)) :
    cnxHeadLossTiedGB N xN epsStr cotN dN ε hng hnbt Wfc bfc xhead Φ g := by
  rw [show Φ = fun a b W bb => L (batchMap N (cnxHeadO h w ε a b W bb) xhead) from
    funext fun a => funext fun b => funext fun W => funext fun bb => hΦ a b W bb]
  have hc : ∀ (y : Vec (768 * h * w)) (dy : Vec nC),
      HasGradAt (fun u => linLoss dy (dense Wfc bfc u))
        (rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y))
        (headCotHn Wfc bfc (rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y)) dy) :=
    fun y dy => HasGradAt.comp_global (f := dense Wfc bfc)
      (x := rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y)) (hasGradAt_linLoss dy _)
      (dense_differentiable Wfc bfc) (denseHasVJP Wfc bfc)
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact (HasGradAt.param_batchMap_through (fun y => globalAvgPoolFlat 768 h w y)
        (fun θ u => rowLNVecFlat 1 768 ε θ hnbt u) (fun _ z => dense Wfc bfc z)
        (fun y dy => headCotHn Wfc bfc (rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y)) dy)
        xhead (θ := hng) hL (fun y => (rowLNVecFlat_gamma_differentiable 1 768 ε hnbt y) _)
        (fun _ => dense_differentiable Wfc bfc) hc
        _ _ (fun n => batchSlice_batchMap _ _ n)
        (fun n => by rw [batchSlice_batchMapAux, batchSlice_batchMap, batchSlice_batchMap])).of_eq
      (funext fun k => (GradNodeB.veclnGammaGradB_den xN epsStr cotN ε hnbt _ hng _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => globalAvgPoolFlat 768 h w y)
        (fun θ u => rowLNVecFlat 1 768 ε hng θ u) (fun _ z => dense Wfc bfc z)
        (fun y dy => headCotHn Wfc bfc (rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y)) dy)
        xhead (θ := hnbt) hL (fun y => (rowLNVecFlat_beta_differentiable 1 768 ε hng y) _)
        (fun _ => dense_differentiable Wfc bfc) hc
        _ _ (fun n => batchSlice_batchMap _ _ n)
        (fun n => by rw [batchSlice_batchMapAux, batchSlice_batchMap, batchSlice_batchMap])).of_eq
      (funext fun i => (GradNodeB.rowDenseBiasGradB_den_lnbeta cotN ε hng
      (fun n => Mat.unflatten (batchSlice N (1 * 768) (batchMap N (globalAvgPoolFlat 768 h w) xhead) n))
      hnbt _ i).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y))
        (fun θ z => dense (Mat.unflatten θ) bfc z) (fun _ l => l) (fun _ dy => dy)
        xhead (θ := Mat.flatten Wfc) (by rw [Mat.unflatten_flatten]; exact hL)
        (fun y => (denseWeightMap_differentiable bfc y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => by rw [batchSlice_batchMap, batchSlice_batchMap]) (fun _ => rfl)).of_eq
      (funext fun idx => by
        obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
        exact (GradNodeB.headWGradB_den dN cotN _ Wfc bfc g i j).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => rowLNVecFlat 1 768 ε hng hnbt (globalAvgPoolFlat 768 h w y))
        (fun θ z => dense Wfc θ z) (fun _ l => l) (fun _ dy => dy) xhead (θ := bfc) hL
        (fun y => (GradNodeB.dense_bias_differentiable Wfc y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => by rw [batchSlice_batchMap, batchSlice_batchMap]) (fun _ => rfl)).of_eq
      (funext fun i => (Finset.sum_congr rfl fun n _ =>
      GradNodeB.headBGradB_den cotN Wfc (batchSlice N 768 (batchMap N (rowLNVecFlat 1 768 ε hng hnbt)
        (batchMap N (globalAvgPoolFlat 768 h w) xhead)) n) bfc g n i).symm)

end Head

-- ════════════════════════════════════════════════════════════════
-- § The whole net: the prefix before each stage, the loss after it, the net with one stage varied
-- ════════════════════════════════════════════════════════════════

/-- Pull the loss gradient back through a batched block's certified VJP: the cotangent is the tie's
    `batchMapAux N (p.cotIn ε)` (`cnxBlockCotInB_eq_vjp`). -/
theorem cnxBlkB_hasGradAt_comp (N : Nat) {c cExp h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : CnxTieBlk c cExp) (X : Vec (N * (c * h * w))) {G : Vec (N * (c * h * w)) → Vec 1}
    {dY : Vec (N * (c * h * w))} (hG : HasGradAt G (batchMap N (p.fwdO (h := h) (w := w) ε) X) dY) :
    HasGradAt (fun y => G (batchMap N (p.fwdO (h := h) (w := w) ε) y)) X
      (batchMapAux N (p.cotIn (h := h) (w := w) ε) X dY) :=
  (HasGradAt.comp (f := batchMap N (p.fwdO (h := h) (w := w) ε)) (x := X) hG
    ((batchMap_differentiable _ (cnxBlockChW_differentiable
      (⟨p.aW, p.aB, ε, p.nG, p.nB, p.eW, p.eB, p.pW, p.pB, p.sL⟩ : CnxBlockParamsCh c cExp h w 7 7)
      hε)) X)
    (batchMapHasVJPAt (cnxBlockChW (h := h) (w := w)
        ⟨p.aW, p.aB, ε, p.nG, p.nB, p.eW, p.eB, p.pW, p.pB, p.sL⟩) X
      (fun _ => (cnxBlockChWHasVJP ⟨p.aW, p.aB, ε, p.nG, p.nB, p.eW, p.eB, p.pW, p.pB, p.sL⟩ hε).toHasVJPAt _)
      (fun _ => (cnxBlockChW_differentiable ⟨p.aW, p.aB, ε, p.nG, p.nB, p.eW, p.eB, p.pW, p.pB, p.sL⟩
        hε) _))).of_eq
    (congrFun (cnxBlockCotInB_eq_vjp N ε hε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL X) dY).symm

/-- …through a batched downsample (`cnxDownCotInB_eq_vjp`). -/
theorem cnxDownB_hasGradAt_comp (N : Nat) {ci co h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : CnxTieDown ci co) (X : Vec (N * (ci * (2 * h) * (2 * w)))) {G : Vec (N * (co * h * w)) → Vec 1}
    {dY : Vec (N * (co * h * w))} (hG : HasGradAt G (batchMap N (p.fwdO (h := h) (w := w) ε) X) dY) :
    HasGradAt (fun y => G (batchMap N (p.fwdO (h := h) (w := w) ε) y)) X
      (batchMapAux N (p.cotIn (h := h) (w := w) ε) X dY) :=
  (HasGradAt.comp (f := batchMap N (p.fwdO (h := h) (w := w) ε)) (x := X) hG
    ((batchMap_differentiable _ (cnxDownChW_differentiable h w ⟨ε, p.G, p.T, p.W, p.B⟩ hε)) X)
    (batchMapHasVJPAt (cnxDownChW h w ⟨ε, p.G, p.T, p.W, p.B⟩) X
      (fun _ => (cnxDownChWHasVJP h w ⟨ε, p.G, p.T, p.W, p.B⟩ hε).toHasVJPAt _)
      (fun _ => (cnxDownChW_differentiable h w ⟨ε, p.G, p.T, p.W, p.B⟩ hε) _))).of_eq
    (congrFun (cnxDownCotInB_eq_vjp N ε hε p.G p.T p.W p.B X) dY).symm

/-- …and through the batched head (`cnxHeadDyB_eq_vjp`). -/
theorem cnxHeadB_hasGradAt_comp (N : Nat) {h w nC : Nat} (ε : ℝ) (hε : 0 < ε) (hng hnbt : Vec 768)
    (Wfc : Mat 768 nC) (bfc : Vec nC) (X : Vec (N * (768 * h * w))) {L : Vec (N * nC) → Vec 1}
    {g : Vec (N * nC)} (hL : HasGradAt L (batchMap N (cnxHeadO h w ε hng hnbt Wfc bfc) X) g) :
    HasGradAt (fun y => L (batchMap N (cnxHeadO h w ε hng hnbt Wfc bfc) y)) X
      (batchMapAux N (cnxHeadDyXheadChN (h := h) (w := w) ε hng hnbt Wfc bfc) X g) :=
  (HasGradAt.comp (f := batchMap N (cnxHeadO h w ε hng hnbt Wfc bfc)) (x := X) hL
    ((batchMap_differentiable _ ((dense_differentiable Wfc bfc).comp
      ((rowLNVecFlat_differentiable 1 768 ε hng hnbt hε).comp
        (globalAvgPoolFlat_differentiable 768 h w)))) X)
    (batchMapHasVJPAt (dense Wfc bfc ∘ rowLNVecFlat 1 768 ε hng hnbt ∘ globalAvgPoolFlat 768 h w) X
      (fun _ => (cnxHeadHasVJP h w ε hε hng hnbt Wfc bfc).toHasVJPAt _)
      (fun _ => ((dense_differentiable Wfc bfc).comp
        ((rowLNVecFlat_differentiable 1 768 ε hng hnbt hε).comp
          (globalAvgPoolFlat_differentiable 768 h w))) _))).of_eq
    (congrFun (cnxHeadDyB_eq_vjp N ε hε hng hnbt Wfc bfc X) g).symm

/-- **ConvNeXt-T, batched**: the tie's forward, stage by stage — `batchMap N` of the stem, each
    block and downsample, then of the head. It is `batchMap N (convNextForwardTCh (w.toCh ε))`
    (`cnxNetB_eq_convNextForwardTCh`). -/
noncomputable def cnxNetB (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    Vec (N * nC) :=
  batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc)
    (batchMap N (w.b18.fwdO (h := 7) (w := 7) ε)
    (batchMap N (w.b17.fwdO (h := 7) (w := 7) ε)
    (batchMap N (w.b16.fwdO (h := 7) (w := 7) ε)
    (batchMap N (w.d2.fwdO (h := 7) (w := 7) ε)
    (batchMap N (w.b15.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b14.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b13.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b12.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b11.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b10.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b9.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b8.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b7.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.d1.fwdO (h := 14) (w := 14) ε)
    (batchMap N (w.b6.fwdO (h := 28) (w := 28) ε)
    (batchMap N (w.b5.fwdO (h := 28) (w := 28) ε)
    (batchMap N (w.b4.fwdO (h := 28) (w := 28) ε)
    (batchMap N (w.d0.fwdO (h := 28) (w := 28) ε)
    (batchMap N (w.b3.fwdO (h := 56) (w := 56) ε)
    (batchMap N (w.b2.fwdO (h := 56) (w := 56) ε)
    (batchMap N (w.b1.fwdO (h := 56) (w := 56) ε)
    (batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x))))))))))))))))))))))

/-- The stem's output — block `b1`'s input (the tie's `ib1`). -/
noncomputable def cnxPreS (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ)

/-- Stage `b1`'s output. -/
noncomputable def cnxPreB1 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMap N (w.b1.fwdO (h := 56) (w := 56) ε) ∘ cnxPreS N ε w

/-- Stage `b2`'s output. -/
noncomputable def cnxPreB2 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMap N (w.b2.fwdO (h := 56) (w := 56) ε) ∘ cnxPreB1 N ε w

/-- Stage `b3`'s output. -/
noncomputable def cnxPreB3 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMap N (w.b3.fwdO (h := 56) (w := 56) ε) ∘ cnxPreB2 N ε w

/-- Stage `d0`'s output. -/
noncomputable def cnxPreD0 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) ∘ cnxPreB3 N ε w

/-- Stage `b4`'s output. -/
noncomputable def cnxPreB4 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMap N (w.b4.fwdO (h := 28) (w := 28) ε) ∘ cnxPreD0 N ε w

/-- Stage `b5`'s output. -/
noncomputable def cnxPreB5 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMap N (w.b5.fwdO (h := 28) (w := 28) ε) ∘ cnxPreB4 N ε w

/-- Stage `b6`'s output. -/
noncomputable def cnxPreB6 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMap N (w.b6.fwdO (h := 28) (w := 28) ε) ∘ cnxPreB5 N ε w

/-- Stage `d1`'s output. -/
noncomputable def cnxPreD1 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB6 N ε w

/-- Stage `b7`'s output. -/
noncomputable def cnxPreB7 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b7.fwdO (h := 14) (w := 14) ε) ∘ cnxPreD1 N ε w

/-- Stage `b8`'s output. -/
noncomputable def cnxPreB8 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b8.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB7 N ε w

/-- Stage `b9`'s output. -/
noncomputable def cnxPreB9 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b9.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB8 N ε w

/-- Stage `b10`'s output. -/
noncomputable def cnxPreB10 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b10.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB9 N ε w

/-- Stage `b11`'s output. -/
noncomputable def cnxPreB11 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b11.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB10 N ε w

/-- Stage `b12`'s output. -/
noncomputable def cnxPreB12 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b12.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB11 N ε w

/-- Stage `b13`'s output. -/
noncomputable def cnxPreB13 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b13.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB12 N ε w

/-- Stage `b14`'s output. -/
noncomputable def cnxPreB14 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b14.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB13 N ε w

/-- Stage `b15`'s output. -/
noncomputable def cnxPreB15 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.b15.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB14 N ε w

/-- Stage `d2`'s output. -/
noncomputable def cnxPreD2 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) ∘ cnxPreB15 N ε w

/-- Stage `b16`'s output. -/
noncomputable def cnxPreB16 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMap N (w.b16.fwdO (h := 7) (w := 7) ε) ∘ cnxPreD2 N ε w

/-- Stage `b17`'s output. -/
noncomputable def cnxPreB17 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMap N (w.b17.fwdO (h := 7) (w := 7) ε) ∘ cnxPreB16 N ε w

/-- Stage `b18`'s output. -/
noncomputable def cnxPreB18 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMap N (w.b18.fwdO (h := 7) (w := 7) ε) ∘ cnxPreB17 N ε w

theorem cnxPreS_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreS N ε w x = batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x := by
  rw [cnxPreS]

theorem cnxPreB1_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB1 N ε w x = batchMap N (w.b1.fwdO (h := 56) (w := 56) ε) (cnxPreS N ε w x) := by
  rw [cnxPreB1, Function.comp_apply]

theorem cnxPreB2_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB2 N ε w x = batchMap N (w.b2.fwdO (h := 56) (w := 56) ε) (cnxPreB1 N ε w x) := by
  rw [cnxPreB2, Function.comp_apply]

theorem cnxPreB3_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB3 N ε w x = batchMap N (w.b3.fwdO (h := 56) (w := 56) ε) (cnxPreB2 N ε w x) := by
  rw [cnxPreB3, Function.comp_apply]

theorem cnxPreD0_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreD0 N ε w x = batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) (cnxPreB3 N ε w x) := by
  rw [cnxPreD0, Function.comp_apply]

theorem cnxPreB4_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB4 N ε w x = batchMap N (w.b4.fwdO (h := 28) (w := 28) ε) (cnxPreD0 N ε w x) := by
  rw [cnxPreB4, Function.comp_apply]

theorem cnxPreB5_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB5 N ε w x = batchMap N (w.b5.fwdO (h := 28) (w := 28) ε) (cnxPreB4 N ε w x) := by
  rw [cnxPreB5, Function.comp_apply]

theorem cnxPreB6_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB6 N ε w x = batchMap N (w.b6.fwdO (h := 28) (w := 28) ε) (cnxPreB5 N ε w x) := by
  rw [cnxPreB6, Function.comp_apply]

theorem cnxPreD1_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreD1 N ε w x = batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) (cnxPreB6 N ε w x) := by
  rw [cnxPreD1, Function.comp_apply]

theorem cnxPreB7_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB7 N ε w x = batchMap N (w.b7.fwdO (h := 14) (w := 14) ε) (cnxPreD1 N ε w x) := by
  rw [cnxPreB7, Function.comp_apply]

theorem cnxPreB8_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB8 N ε w x = batchMap N (w.b8.fwdO (h := 14) (w := 14) ε) (cnxPreB7 N ε w x) := by
  rw [cnxPreB8, Function.comp_apply]

theorem cnxPreB9_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB9 N ε w x = batchMap N (w.b9.fwdO (h := 14) (w := 14) ε) (cnxPreB8 N ε w x) := by
  rw [cnxPreB9, Function.comp_apply]

theorem cnxPreB10_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB10 N ε w x = batchMap N (w.b10.fwdO (h := 14) (w := 14) ε) (cnxPreB9 N ε w x) := by
  rw [cnxPreB10, Function.comp_apply]

theorem cnxPreB11_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB11 N ε w x = batchMap N (w.b11.fwdO (h := 14) (w := 14) ε) (cnxPreB10 N ε w x) := by
  rw [cnxPreB11, Function.comp_apply]

theorem cnxPreB12_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB12 N ε w x = batchMap N (w.b12.fwdO (h := 14) (w := 14) ε) (cnxPreB11 N ε w x) := by
  rw [cnxPreB12, Function.comp_apply]

theorem cnxPreB13_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB13 N ε w x = batchMap N (w.b13.fwdO (h := 14) (w := 14) ε) (cnxPreB12 N ε w x) := by
  rw [cnxPreB13, Function.comp_apply]

theorem cnxPreB14_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB14 N ε w x = batchMap N (w.b14.fwdO (h := 14) (w := 14) ε) (cnxPreB13 N ε w x) := by
  rw [cnxPreB14, Function.comp_apply]

theorem cnxPreB15_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB15 N ε w x = batchMap N (w.b15.fwdO (h := 14) (w := 14) ε) (cnxPreB14 N ε w x) := by
  rw [cnxPreB15, Function.comp_apply]

theorem cnxPreD2_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreD2 N ε w x = batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) (cnxPreB15 N ε w x) := by
  rw [cnxPreD2, Function.comp_apply]

theorem cnxPreB16_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB16 N ε w x = batchMap N (w.b16.fwdO (h := 7) (w := 7) ε) (cnxPreD2 N ε w x) := by
  rw [cnxPreB16, Function.comp_apply]

theorem cnxPreB17_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB17 N ε w x = batchMap N (w.b17.fwdO (h := 7) (w := 7) ε) (cnxPreB16 N ε w x) := by
  rw [cnxPreB17, Function.comp_apply]

theorem cnxPreB18_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB18 N ε w x = batchMap N (w.b18.fwdO (h := 7) (w := 7) ε) (cnxPreB17 N ε w x) := by
  rw [cnxPreB18, Function.comp_apply]

/-- The net after block `b18` — the head. -/
noncomputable def cnxSufB18 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc)

/-- The net after stage `b17`: stage `b18`, then the rest. -/
noncomputable def cnxSufB17 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  fun y => cnxSufB18 N ε w (batchMap N (w.b18.fwdO (h := 7) (w := 7) ε) y)

/-- The net after stage `b16`: stage `b17`, then the rest. -/
noncomputable def cnxSufB16 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  fun y => cnxSufB17 N ε w (batchMap N (w.b17.fwdO (h := 7) (w := 7) ε) y)

/-- The net after stage `d2`: stage `b16`, then the rest. -/
noncomputable def cnxSufD2 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  fun y => cnxSufB16 N ε w (batchMap N (w.b16.fwdO (h := 7) (w := 7) ε) y)

/-- The net after stage `b15`: stage `d2`, then the rest. -/
noncomputable def cnxSufB15 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufD2 N ε w (batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) y)

/-- The net after stage `b14`: stage `b15`, then the rest. -/
noncomputable def cnxSufB14 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB15 N ε w (batchMap N (w.b15.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b13`: stage `b14`, then the rest. -/
noncomputable def cnxSufB13 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB14 N ε w (batchMap N (w.b14.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b12`: stage `b13`, then the rest. -/
noncomputable def cnxSufB12 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB13 N ε w (batchMap N (w.b13.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b11`: stage `b12`, then the rest. -/
noncomputable def cnxSufB11 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB12 N ε w (batchMap N (w.b12.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b10`: stage `b11`, then the rest. -/
noncomputable def cnxSufB10 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB11 N ε w (batchMap N (w.b11.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b9`: stage `b10`, then the rest. -/
noncomputable def cnxSufB9 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB10 N ε w (batchMap N (w.b10.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b8`: stage `b9`, then the rest. -/
noncomputable def cnxSufB8 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB9 N ε w (batchMap N (w.b9.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b7`: stage `b8`, then the rest. -/
noncomputable def cnxSufB7 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB8 N ε w (batchMap N (w.b8.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `d1`: stage `b7`, then the rest. -/
noncomputable def cnxSufD1 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB7 N ε w (batchMap N (w.b7.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b6`: stage `d1`, then the rest. -/
noncomputable def cnxSufB6 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufD1 N ε w (batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b5`: stage `b6`, then the rest. -/
noncomputable def cnxSufB5 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufB6 N ε w (batchMap N (w.b6.fwdO (h := 28) (w := 28) ε) y)

/-- The net after stage `b4`: stage `b5`, then the rest. -/
noncomputable def cnxSufB4 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufB5 N ε w (batchMap N (w.b5.fwdO (h := 28) (w := 28) ε) y)

/-- The net after stage `d0`: stage `b4`, then the rest. -/
noncomputable def cnxSufD0 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufB4 N ε w (batchMap N (w.b4.fwdO (h := 28) (w := 28) ε) y)

/-- The net after stage `b3`: stage `d0`, then the rest. -/
noncomputable def cnxSufB3 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufD0 N ε w (batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) y)

/-- The net after stage `b2`: stage `b3`, then the rest. -/
noncomputable def cnxSufB2 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufB3 N ε w (batchMap N (w.b3.fwdO (h := 56) (w := 56) ε) y)

/-- The net after stage `b1`: stage `b2`, then the rest. -/
noncomputable def cnxSufB1 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufB2 N ε w (batchMap N (w.b2.fwdO (h := 56) (w := 56) ε) y)

/-- The net after the stem: stage `b1`, then the rest. -/
noncomputable def cnxSufS (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufB1 N ε w (batchMap N (w.b1.fwdO (h := 56) (w := 56) ε) y)

/-- **The net with the stem's parameters varied** is the suffix after the stem at the varied stem. -/
theorem cnx_factor_stem (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 96 3 4 4) (b γ β : Vec 96) :
    cnxNetB N ε { w with sW := W, sb := b, sγ := γ, sβ := β } x
      = cnxSufS N ε w (batchMap N (cnxStemFwdO (h := 56) (w := 56) ε W b γ β) x) := rfl

/-- **The net with stage `b1`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b1 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 96 384) :
    cnxNetB N ε { w with b1 := p } x
      = cnxSufB1 N ε w (batchMap N (p.fwdO (h := 56) (w := 56) ε) (cnxPreS N ε w x)) := by
  rw [cnxPreS_apply]; rfl

/-- **The net with stage `b2`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b2 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 96 384) :
    cnxNetB N ε { w with b2 := p } x
      = cnxSufB2 N ε w (batchMap N (p.fwdO (h := 56) (w := 56) ε) (cnxPreB1 N ε w x)) := by
  rw [cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b3`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b3 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 96 384) :
    cnxNetB N ε { w with b3 := p } x
      = cnxSufB3 N ε w (batchMap N (p.fwdO (h := 56) (w := 56) ε) (cnxPreB2 N ε w x)) := by
  rw [cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `d0`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_d0 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieDown 96 192) :
    cnxNetB N ε { w with d0 := p } x
      = cnxSufD0 N ε w (batchMap N (p.fwdO (h := 28) (w := 28) ε) (cnxPreB3 N ε w x)) := by
  rw [cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b4`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b4 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 192 768) :
    cnxNetB N ε { w with b4 := p } x
      = cnxSufB4 N ε w (batchMap N (p.fwdO (h := 28) (w := 28) ε) (cnxPreD0 N ε w x)) := by
  rw [cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b5`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b5 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 192 768) :
    cnxNetB N ε { w with b5 := p } x
      = cnxSufB5 N ε w (batchMap N (p.fwdO (h := 28) (w := 28) ε) (cnxPreB4 N ε w x)) := by
  rw [cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b6`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b6 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 192 768) :
    cnxNetB N ε { w with b6 := p } x
      = cnxSufB6 N ε w (batchMap N (p.fwdO (h := 28) (w := 28) ε) (cnxPreB5 N ε w x)) := by
  rw [cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `d1`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_d1 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieDown 192 384) :
    cnxNetB N ε { w with d1 := p } x
      = cnxSufD1 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB6 N ε w x)) := by
  rw [cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b7`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b7 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b7 := p } x
      = cnxSufB7 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreD1 N ε w x)) := by
  rw [cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b8`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b8 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b8 := p } x
      = cnxSufB8 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB7 N ε w x)) := by
  rw [cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b9`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b9 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b9 := p } x
      = cnxSufB9 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB8 N ε w x)) := by
  rw [cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b10`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b10 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b10 := p } x
      = cnxSufB10 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB9 N ε w x)) := by
  rw [cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b11`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b11 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b11 := p } x
      = cnxSufB11 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB10 N ε w x)) := by
  rw [cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b12`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b12 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b12 := p } x
      = cnxSufB12 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB11 N ε w x)) := by
  rw [cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b13`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b13 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b13 := p } x
      = cnxSufB13 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB12 N ε w x)) := by
  rw [cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b14`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b14 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b14 := p } x
      = cnxSufB14 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB13 N ε w x)) := by
  rw [cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b15`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b15 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB N ε { w with b15 := p } x
      = cnxSufB15 N ε w (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB14 N ε w x)) := by
  rw [cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `d2`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_d2 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieDown 384 768) :
    cnxNetB N ε { w with d2 := p } x
      = cnxSufD2 N ε w (batchMap N (p.fwdO (h := 7) (w := 7) ε) (cnxPreB15 N ε w x)) := by
  rw [cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b16`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b16 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 768 3072) :
    cnxNetB N ε { w with b16 := p } x
      = cnxSufB16 N ε w (batchMap N (p.fwdO (h := 7) (w := 7) ε) (cnxPreD2 N ε w x)) := by
  rw [cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b17`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b17 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 768 3072) :
    cnxNetB N ε { w with b17 := p } x
      = cnxSufB17 N ε w (batchMap N (p.fwdO (h := 7) (w := 7) ε) (cnxPreB16 N ε w x)) := by
  rw [cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b18`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b18 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 768 3072) :
    cnxNetB N ε { w with b18 := p } x
      = cnxSufB18 N ε w (batchMap N (p.fwdO (h := 7) (w := 7) ε) (cnxPreB17 N ε w x)) := by
  rw [cnxPreB17_apply, cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with the head varied** is the head at the varied parameters. -/
theorem cnx_factor_head (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224)))
    (a b : Vec 768) (W : Mat 768 nC) (bb : Vec nC) :
    cnxNetB N ε { w with hG := a, hT := b, Wfc := W, bfc := bb } x
      = batchMap N (cnxHeadO 7 7 ε a b W bb) (cnxPreB18 N ε w x) := by
  rw [cnxPreB18_apply, cnxPreB17_apply, cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- The net's output is the head at block `b18`'s output. -/
theorem cnx_forward_eq_head (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxNetB N ε w x = batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc) (cnxPreB18 N ε w x) := by
  rw [cnxPreB18_apply, cnxPreB17_apply, cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The logits the tie's loss cotangent reads are `cnxNetB`'s.** The tie spells the head as
    three batched ops (`batchMap_comp`). -/
theorem cnx_logitsB_eq (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    batchMap N (dense w.Wfc w.bfc) (batchMap N (rowLNVecFlat 1 768 ε w.hG w.hT)
      (batchMap N (globalAvgPoolFlat 768 7 7) (cnxPreB18 N ε w x))) = cnxNetB N ε w x := by
  rw [cnx_forward_eq_head, cnxHeadO, batchMap_comp, batchMap_comp]; rfl

/-- **`cnxNetB` is the canonical ConvNeXt-T forward, batched**: `convNextForwardTCh` at
    `w.toCh ε`, the FullT forward whose VJP is `convNextForwardTChHasVJP`, applied per example. The
    per-example identity is `CnxTieWeights.forward_eq_convNextForwardTCh`; `batchMap_comp` splits
    the batched composite into the capstone's stage-by-stage chain. -/
theorem cnxNetB_eq_convNextForwardTCh (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC)
    (x : Vec (N * (3 * 224 * 224))) :
    cnxNetB N ε w x = batchMap N (convNextForwardTCh (w.toCh ε)) x := by
  have hper : ∀ y, convNextForwardTCh (w.toCh ε) y
      = (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc ∘ w.b18.fwdO (h := 7) (w := 7) ε
        ∘ w.b17.fwdO (h := 7) (w := 7) ε ∘ w.b16.fwdO (h := 7) (w := 7) ε
        ∘ w.d2.fwdO (h := 7) (w := 7) ε ∘ w.b15.fwdO (h := 14) (w := 14) ε
        ∘ w.b14.fwdO (h := 14) (w := 14) ε ∘ w.b13.fwdO (h := 14) (w := 14) ε
        ∘ w.b12.fwdO (h := 14) (w := 14) ε ∘ w.b11.fwdO (h := 14) (w := 14) ε
        ∘ w.b10.fwdO (h := 14) (w := 14) ε ∘ w.b9.fwdO (h := 14) (w := 14) ε
        ∘ w.b8.fwdO (h := 14) (w := 14) ε ∘ w.b7.fwdO (h := 14) (w := 14) ε
        ∘ w.d1.fwdO (h := 14) (w := 14) ε ∘ w.b6.fwdO (h := 28) (w := 28) ε
        ∘ w.b5.fwdO (h := 28) (w := 28) ε ∘ w.b4.fwdO (h := 28) (w := 28) ε
        ∘ w.d0.fwdO (h := 28) (w := 28) ε ∘ w.b3.fwdO (h := 56) (w := 56) ε
        ∘ w.b2.fwdO (h := 56) (w := 56) ε ∘ w.b1.fwdO (h := 56) (w := 56) ε
        ∘ cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) y := by
    intro y
    rw [← CnxTiePoC.CnxTieWeights.forward_eq_convNextForwardTCh w ε y]
    simp only [Function.comp_apply, cnxHeadO, mnistLinear]
  rw [show convNextForwardTCh (w.toCh ε) = _ from funext hper, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp]
  unfold cnxNetB
  simp only [Function.comp_apply]

/-- **Every ConvNeXt-T parameter gradient node is the derivative of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 182 nodes
    `cnx_net_tiedGB` ties, each at the cotangent the tie threads to it from `g`, stated against
    `L` of `cnxNetB` with that one parameter varied. -/
def CnxNetLossTiedGB (xN epsStr cotN dN : String) (N : Nat) {nC : Nat} (ε : ℝ)
    (w : CnxTieWeights nC) (xstem : Vec (N * (3 * 56 * 56))) (x : Vec (N * (3 * 224 * 224)))
    (L : Vec (N * nC) → Vec 1) (g : Vec (N * nC)) : Prop :=
  let dyO18 := batchMapAux N (cnxHeadDyXheadChN (h := 7) (w := 7) ε w.hG w.hT w.Wfc w.bfc)
    (cnxPreB18 N ε w x) g
  let dyO17 := batchMapAux N (w.b18.cotIn (h := 7) (w := 7) ε) (cnxPreB17 N ε w x) dyO18
  let dyO16 := batchMapAux N (w.b17.cotIn (h := 7) (w := 7) ε) (cnxPreB16 N ε w x) dyO17
  let dyD2 := batchMapAux N (w.b16.cotIn (h := 7) (w := 7) ε) (cnxPreD2 N ε w x) dyO16
  let dyO15 := batchMapAux N (w.d2.cotIn (h := 7) (w := 7) ε) (cnxPreB15 N ε w x) dyD2
  let dyO14 := batchMapAux N (w.b15.cotIn (h := 14) (w := 14) ε) (cnxPreB14 N ε w x) dyO15
  let dyO13 := batchMapAux N (w.b14.cotIn (h := 14) (w := 14) ε) (cnxPreB13 N ε w x) dyO14
  let dyO12 := batchMapAux N (w.b13.cotIn (h := 14) (w := 14) ε) (cnxPreB12 N ε w x) dyO13
  let dyO11 := batchMapAux N (w.b12.cotIn (h := 14) (w := 14) ε) (cnxPreB11 N ε w x) dyO12
  let dyO10 := batchMapAux N (w.b11.cotIn (h := 14) (w := 14) ε) (cnxPreB10 N ε w x) dyO11
  let dyO9 := batchMapAux N (w.b10.cotIn (h := 14) (w := 14) ε) (cnxPreB9 N ε w x) dyO10
  let dyO8 := batchMapAux N (w.b9.cotIn (h := 14) (w := 14) ε) (cnxPreB8 N ε w x) dyO9
  let dyO7 := batchMapAux N (w.b8.cotIn (h := 14) (w := 14) ε) (cnxPreB7 N ε w x) dyO8
  let dyD1 := batchMapAux N (w.b7.cotIn (h := 14) (w := 14) ε) (cnxPreD1 N ε w x) dyO7
  let dyO6 := batchMapAux N (w.d1.cotIn (h := 14) (w := 14) ε) (cnxPreB6 N ε w x) dyD1
  let dyO5 := batchMapAux N (w.b6.cotIn (h := 28) (w := 28) ε) (cnxPreB5 N ε w x) dyO6
  let dyO4 := batchMapAux N (w.b5.cotIn (h := 28) (w := 28) ε) (cnxPreB4 N ε w x) dyO5
  let dyD0 := batchMapAux N (w.b4.cotIn (h := 28) (w := 28) ε) (cnxPreD0 N ε w x) dyO4
  let dyO3 := batchMapAux N (w.d0.cotIn (h := 28) (w := 28) ε) (cnxPreB3 N ε w x) dyD0
  let dyO2 := batchMapAux N (w.b3.cotIn (h := 56) (w := 56) ε) (cnxPreB2 N ε w x) dyO3
  let dyO1 := batchMapAux N (w.b2.cotIn (h := 56) (w := 56) ε) (cnxPreB1 N ε w x) dyO2
  let dyStem := batchMapAux N (w.b1.cotIn (h := 56) (w := 56) ε) (cnxPreS N ε w x) dyO1
  cnxStemLossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε w.sW w.sb w.sγ w.sβ x xstem
      (fun W b γ β => L (cnxNetB N ε { w with sW := W, sb := b, sγ := γ, sβ := β } x)) dyStem
  ∧ cnxBlockLossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε w.b1 (cnxPreS N ε w x)
      (fun p => L (cnxNetB N ε { w with b1 := p } x)) dyO1
  ∧ cnxBlockLossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε w.b2 (cnxPreB1 N ε w x)
      (fun p => L (cnxNetB N ε { w with b2 := p } x)) dyO2
  ∧ cnxBlockLossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε w.b3 (cnxPreB2 N ε w x)
      (fun p => L (cnxNetB N ε { w with b3 := p } x)) dyO3
  ∧ cnxDownLossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.d0 (cnxPreB3 N ε w x)
      (fun p => L (cnxNetB N ε { w with d0 := p } x)) dyD0
  ∧ cnxBlockLossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.b4 (cnxPreD0 N ε w x)
      (fun p => L (cnxNetB N ε { w with b4 := p } x)) dyO4
  ∧ cnxBlockLossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.b5 (cnxPreB4 N ε w x)
      (fun p => L (cnxNetB N ε { w with b5 := p } x)) dyO5
  ∧ cnxBlockLossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.b6 (cnxPreB5 N ε w x)
      (fun p => L (cnxNetB N ε { w with b6 := p } x)) dyO6
  ∧ cnxDownLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.d1 (cnxPreB6 N ε w x)
      (fun p => L (cnxNetB N ε { w with d1 := p } x)) dyD1
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b7 (cnxPreD1 N ε w x)
      (fun p => L (cnxNetB N ε { w with b7 := p } x)) dyO7
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b8 (cnxPreB7 N ε w x)
      (fun p => L (cnxNetB N ε { w with b8 := p } x)) dyO8
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b9 (cnxPreB8 N ε w x)
      (fun p => L (cnxNetB N ε { w with b9 := p } x)) dyO9
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b10 (cnxPreB9 N ε w x)
      (fun p => L (cnxNetB N ε { w with b10 := p } x)) dyO10
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b11 (cnxPreB10 N ε w x)
      (fun p => L (cnxNetB N ε { w with b11 := p } x)) dyO11
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b12 (cnxPreB11 N ε w x)
      (fun p => L (cnxNetB N ε { w with b12 := p } x)) dyO12
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b13 (cnxPreB12 N ε w x)
      (fun p => L (cnxNetB N ε { w with b13 := p } x)) dyO13
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b14 (cnxPreB13 N ε w x)
      (fun p => L (cnxNetB N ε { w with b14 := p } x)) dyO14
  ∧ cnxBlockLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.b15 (cnxPreB14 N ε w x)
      (fun p => L (cnxNetB N ε { w with b15 := p } x)) dyO15
  ∧ cnxDownLossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.d2 (cnxPreB15 N ε w x)
      (fun p => L (cnxNetB N ε { w with d2 := p } x)) dyD2
  ∧ cnxBlockLossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.b16 (cnxPreD2 N ε w x)
      (fun p => L (cnxNetB N ε { w with b16 := p } x)) dyO16
  ∧ cnxBlockLossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.b17 (cnxPreB16 N ε w x)
      (fun p => L (cnxNetB N ε { w with b17 := p } x)) dyO17
  ∧ cnxBlockLossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.b18 (cnxPreB17 N ε w x)
      (fun p => L (cnxNetB N ε { w with b18 := p } x)) dyO18
  ∧ cnxHeadLossTiedGB N (h := 7) (w := 7) xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc (cnxPreB18 N ε w x)
      (fun a b W bb => L (cnxNetB N ε { w with hG := a, hT := b, Wfc := W, bfc := bb } x)) g

/-- **Every ConvNeXt-T parameter gradient node is the derivative of the loss in that parameter.**
    For any loss `L` of the logits with gradient `g` at the net's output, each of the 182 nodes
    `cnx_net_tiedGB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net, `cnxNetB` with that
    one parameter varied (a stem field, a block's or downsample's record `w.bk := p` with one slot
    changed, or a head field).

    Hypothesis: `0 < ε`, the LayerNorms' (the tie itself needs none). The loss enters only through
    `hL`; `cnx_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship. -/
theorem cnx_net_lossGrad (xN epsStr cotN dN : String) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : CnxTieWeights nC) (xstem : Vec (N * (3 * 56 * 56))) (x : Vec (N * (3 * 224 * 224)))
    {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)} (hL : HasGradAt L (cnxNetB N ε w x) g) :
    CnxNetLossTiedGB xN epsStr cotN dN N ε w xstem x L g := by
  unfold CnxNetLossTiedGB
  intro dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  have hL' : HasGradAt L (batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc) (cnxPreB18 N ε w x)) g :=
    hL.congr_point (cnx_forward_eq_head N ε w x)
  have hB18 : HasGradAt (fun y => L (cnxSufB18 N ε w y)) (cnxPreB18 N ε w x) dyO18 :=
    cnxHeadB_hasGradAt_comp N ε hε w.hG w.hT w.Wfc w.bfc _ hL'
  have hB17 : HasGradAt (fun y => L (cnxSufB17 N ε w y)) (cnxPreB17 N ε w x) dyO17 :=
    cnxBlkB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.b18 _ (hB18.congr_point (cnxPreB18_apply N ε w x))
  have hB16 : HasGradAt (fun y => L (cnxSufB16 N ε w y)) (cnxPreB16 N ε w x) dyO16 :=
    cnxBlkB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.b17 _ (hB17.congr_point (cnxPreB17_apply N ε w x))
  have hD2 : HasGradAt (fun y => L (cnxSufD2 N ε w y)) (cnxPreD2 N ε w x) dyD2 :=
    cnxBlkB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.b16 _ (hB16.congr_point (cnxPreB16_apply N ε w x))
  have hB15 : HasGradAt (fun y => L (cnxSufB15 N ε w y)) (cnxPreB15 N ε w x) dyO15 :=
    cnxDownB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.d2 _ (hD2.congr_point (cnxPreD2_apply N ε w x))
  have hB14 : HasGradAt (fun y => L (cnxSufB14 N ε w y)) (cnxPreB14 N ε w x) dyO14 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b15 _ (hB15.congr_point (cnxPreB15_apply N ε w x))
  have hB13 : HasGradAt (fun y => L (cnxSufB13 N ε w y)) (cnxPreB13 N ε w x) dyO13 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b14 _ (hB14.congr_point (cnxPreB14_apply N ε w x))
  have hB12 : HasGradAt (fun y => L (cnxSufB12 N ε w y)) (cnxPreB12 N ε w x) dyO12 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b13 _ (hB13.congr_point (cnxPreB13_apply N ε w x))
  have hB11 : HasGradAt (fun y => L (cnxSufB11 N ε w y)) (cnxPreB11 N ε w x) dyO11 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b12 _ (hB12.congr_point (cnxPreB12_apply N ε w x))
  have hB10 : HasGradAt (fun y => L (cnxSufB10 N ε w y)) (cnxPreB10 N ε w x) dyO10 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b11 _ (hB11.congr_point (cnxPreB11_apply N ε w x))
  have hB9 : HasGradAt (fun y => L (cnxSufB9 N ε w y)) (cnxPreB9 N ε w x) dyO9 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b10 _ (hB10.congr_point (cnxPreB10_apply N ε w x))
  have hB8 : HasGradAt (fun y => L (cnxSufB8 N ε w y)) (cnxPreB8 N ε w x) dyO8 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b9 _ (hB9.congr_point (cnxPreB9_apply N ε w x))
  have hB7 : HasGradAt (fun y => L (cnxSufB7 N ε w y)) (cnxPreB7 N ε w x) dyO7 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b8 _ (hB8.congr_point (cnxPreB8_apply N ε w x))
  have hD1 : HasGradAt (fun y => L (cnxSufD1 N ε w y)) (cnxPreD1 N ε w x) dyD1 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b7 _ (hB7.congr_point (cnxPreB7_apply N ε w x))
  have hB6 : HasGradAt (fun y => L (cnxSufB6 N ε w y)) (cnxPreB6 N ε w x) dyO6 :=
    cnxDownB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.d1 _ (hD1.congr_point (cnxPreD1_apply N ε w x))
  have hB5 : HasGradAt (fun y => L (cnxSufB5 N ε w y)) (cnxPreB5 N ε w x) dyO5 :=
    cnxBlkB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.b6 _ (hB6.congr_point (cnxPreB6_apply N ε w x))
  have hB4 : HasGradAt (fun y => L (cnxSufB4 N ε w y)) (cnxPreB4 N ε w x) dyO4 :=
    cnxBlkB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.b5 _ (hB5.congr_point (cnxPreB5_apply N ε w x))
  have hD0 : HasGradAt (fun y => L (cnxSufD0 N ε w y)) (cnxPreD0 N ε w x) dyD0 :=
    cnxBlkB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.b4 _ (hB4.congr_point (cnxPreB4_apply N ε w x))
  have hB3 : HasGradAt (fun y => L (cnxSufB3 N ε w y)) (cnxPreB3 N ε w x) dyO3 :=
    cnxDownB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.d0 _ (hD0.congr_point (cnxPreD0_apply N ε w x))
  have hB2 : HasGradAt (fun y => L (cnxSufB2 N ε w y)) (cnxPreB2 N ε w x) dyO2 :=
    cnxBlkB_hasGradAt_comp N (h := 56) (w := 56) ε hε w.b3 _ (hB3.congr_point (cnxPreB3_apply N ε w x))
  have hB1 : HasGradAt (fun y => L (cnxSufB1 N ε w y)) (cnxPreB1 N ε w x) dyO1 :=
    cnxBlkB_hasGradAt_comp N (h := 56) (w := 56) ε hε w.b2 _ (hB2.congr_point (cnxPreB2_apply N ε w x))
  have hS : HasGradAt (fun y => L (cnxSufS N ε w y)) (cnxPreS N ε w x) dyStem :=
    cnxBlkB_hasGradAt_comp N (h := 56) (w := 56) ε hε w.b1 _ (hB1.congr_point (cnxPreB1_apply N ε w x))
  refine ⟨cnx_stem_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.sW w.sb w.sγ w.sβ x xstem
      (hS.congr_point (cnxPreS_apply N ε w x)) (fun W b γ β => by rw [cnx_factor_stem]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.b1 _
      (hB1.congr_point (cnxPreB1_apply N ε w x)) (fun p => by rw [cnx_factor_b1]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.b2 _
      (hB2.congr_point (cnxPreB2_apply N ε w x)) (fun p => by rw [cnx_factor_b2]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.b3 _
      (hB3.congr_point (cnxPreB3_apply N ε w x)) (fun p => by rw [cnx_factor_b3]), ?_⟩
  refine ⟨cnx_down_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.d0 _
      (hD0.congr_point (cnxPreD0_apply N ε w x)) (fun p => by rw [cnx_factor_d0]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε hε w.b4 _
      (hB4.congr_point (cnxPreB4_apply N ε w x)) (fun p => by rw [cnx_factor_b4]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε hε w.b5 _
      (hB5.congr_point (cnxPreB5_apply N ε w x)) (fun p => by rw [cnx_factor_b5]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε hε w.b6 _
      (hB6.congr_point (cnxPreB6_apply N ε w x)) (fun p => by rw [cnx_factor_b6]), ?_⟩
  refine ⟨cnx_down_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.d1 _
      (hD1.congr_point (cnxPreD1_apply N ε w x)) (fun p => by rw [cnx_factor_d1]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b7 _
      (hB7.congr_point (cnxPreB7_apply N ε w x)) (fun p => by rw [cnx_factor_b7]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b8 _
      (hB8.congr_point (cnxPreB8_apply N ε w x)) (fun p => by rw [cnx_factor_b8]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b9 _
      (hB9.congr_point (cnxPreB9_apply N ε w x)) (fun p => by rw [cnx_factor_b9]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b10 _
      (hB10.congr_point (cnxPreB10_apply N ε w x)) (fun p => by rw [cnx_factor_b10]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b11 _
      (hB11.congr_point (cnxPreB11_apply N ε w x)) (fun p => by rw [cnx_factor_b11]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b12 _
      (hB12.congr_point (cnxPreB12_apply N ε w x)) (fun p => by rw [cnx_factor_b12]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b13 _
      (hB13.congr_point (cnxPreB13_apply N ε w x)) (fun p => by rw [cnx_factor_b13]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b14 _
      (hB14.congr_point (cnxPreB14_apply N ε w x)) (fun p => by rw [cnx_factor_b14]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b15 _
      (hB15.congr_point (cnxPreB15_apply N ε w x)) (fun p => by rw [cnx_factor_b15]), ?_⟩
  refine ⟨cnx_down_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.d2 _
      (hD2.congr_point (cnxPreD2_apply N ε w x)) (fun p => by rw [cnx_factor_d2]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε hε w.b16 _
      (hB16.congr_point (cnxPreB16_apply N ε w x)) (fun p => by rw [cnx_factor_b16]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε hε w.b17 _
      (hB17.congr_point (cnxPreB17_apply N ε w x)) (fun p => by rw [cnx_factor_b17]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε hε w.b18 _
      (hB18.congr_point (cnxPreB18_apply N ε w x)) (fun p => by rw [cnx_factor_b18]), ?_⟩
  exact cnx_head_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc _ hL'
    (fun a b W bb => by rw [cnx_factor_head])

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLossDiv`, `g` the `softmaxDiv` cotangent the render emits — the
    tie's own `g`, whose logits are `cnxNetB N ε w x` (`cnx_logitsB_eq`). -/
theorem cnx_net_lossGrad_smoothedCE (xN epsStr cotN dN aStr negAK bStr logN ohN : String)
    (N : Nat) {nC : Nat} (hK : 0 < nC) (ε α B : ℝ) (hε : 0 < ε) (w : CnxTieWeights nC)
    (xstem : Vec (N * (3 * 56 * 56))) (x : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC))
    (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    CnxNetLossTiedGB xN epsStr cotN dN N ε w xstem x (smoothedBatchLossDiv N nC α B t)
      (den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN (cnxNetB N ε w x) t)) :=
  cnx_net_lossGrad xN epsStr cotN dN N ε hε w xstem x
    ⟨(smoothedBatchLossDiv_differentiable N nC α B t) _,
      fun J => smoothedBatchLossDiv_grad N nC hK α B aStr negAK bStr logN ohN t _ ht J⟩

end Proofs.CnxTiePoCGB
