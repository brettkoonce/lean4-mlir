import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB

/-! # ConvNeXt-T — every parameter gradient node IS the loss's derivative in that parameter

`cnx_net_tiedGB` says each of the 182 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it, the chain's top
being the smoothed-loss cotangent. `cnx_net_lossGrad` composes that with the chain: for any loss
`L` of the logits whose gradient at the net's output is `g`, the loss of the WHOLE net `cnxNetB`
with that one parameter varied is differentiable in it and every node is its gradient
(`HasGradAt`). `cnxNetB` is the tie's forward, batched, at its stochastic-depth sites `sd`; at
`none` it is `batchMap N` of the canonical `convNextForwardTCh` (`cnxNetB_eq_convNextForwardTCh`). `cnx_net_lossGrad_smoothedCE` discharges `hL` for
the label-smoothed loss the artifacts ship (`smoothedBatchLossDiv`, whose gradient is the
`softmaxDiv` cotangent the render emits).

**How.** No ConvNeXt op couples examples, so the work is per example and lifted once:

* **Per example** (at variable widths): the loss read at each activation inside a block, a
  downsample, the stem and the head has the tie's own per-example cotangent as its gradient
  (`cnxBlk_hasGradAt`, `cnxDown_hasGradAt`, …). Every stage VJP is global — GELU has no kink and
  LayerNorm needs only `0 < ε` — so each step is one `HasGradAt.comp_global`, and the channel-LN
  step is `chanLNTensor3Back_eq_chanLN_vjp`.
* **Lifted** (`HasGradAt.param_batchMapIdx_through`, `Foundation.Batched.Indexed`, for a block;
  `param_batchMap_through` for the stem, downsamples and head): read against the linear loss
  `⟨·, dyₙ⟩` per example, those gradients turn each tied node's `Σ_n Σ_j ∂per/∂θ · cotₙ` into
  `∂G/∂θ` of the batched block, `G` the loss at the block's output. Example `n`'s block suffix
  ends in its own drop site (`cnxPostP`'s `siteScale`), so every block cotangent is the
  drop-free one at `s ⊙ dy`.
* **Per net**: the loss read after each stage (`cnxSuf*`), pulled back through the certified
  batched block, downsample and head VJPs (`cnxBlockCotInB_eq_vjp`, `cnxDownCotInB_eq_vjp`,
  `cnxHeadDyB_eq_vjp`), and each `Φ` identified with the whole net at updated weights by a
  standalone `cnx_factor_*` theorem.

**The stem's bias node** is emitted as a stride-1 `convBiasGradB` at the output resolution, and
reads no input; its Jacobian in the bias is the channel indicator whatever the conv, so it is the
patchify conv's (`GradNodeB.pdiv_flatConvStride4_bias_eq_conv2d`), as in the tie.

**Hypotheses.** `0 < ε` (the LayerNorms' VJPs); no smoothness hypothesis. For the smoothed loss,
every example's target sums to one and `0 < nC`. Stochastic depth is the tie's `sd` binder,
`none` the drop-free chain.

**Scope.** One replica, at either precision: every conv, depthwise, strided-conv and patchify
weight node is stated on the renderers' switch (`convWeightGradBAt bf16 id …`,
`depthwiseWeightGradBAt bf16 id …`, `convStridedWeightGradBAt bf16 id …`,
`convStride4WeightGradBAt bf16 id …`; `StableHLO.PrecisionSwitch`), so `bf16 := true` is the bf16
kind the `convnextin_*bf16` artifacts emit, read over ℝ at the identity rounding (`Bf16Erasure`),
and `bf16 := false` the f32 artifacts'. The bias, LayerNorm, layer-scale and classifier nodes carry
no flag because no render switches them.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnxTieGB

open scoped BigOperators
open Proofs.CnxTie (cnxStemFwdO cnxBlockFwdChO cnxDownFwdChO cnxBlockCotInChAt cnxDownCotInChAt
  CnxTieWeights CnxTieBlk CnxTieDown)

-- ════════════════════════════════════════════════════════════════
-- § Parameter differentiability of the ConvNeXt-specific ops
-- ════════════════════════════════════════════════════════════════

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

/-- The block after the project conv: layer scale, the drop site, then the identity skip. -/
noncomputable def cnxPostP (h w : Nat) (p : CnxTieBlk c cExp) (s : Option ℝ) (y : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun u i => siteScale s (layerScale (fun k => p.sL (chanIdx c h w k)) u i) + y i

/-- The block after the expand conv (pre-GELU). -/
noncomputable def cnxPostE (gf : GeluForm) (h w : Nat) (p : CnxTieBlk c cExp) (s : Option ℝ)
    (y : Vec (c * h * w)) : Vec (cExp * h * w) → Vec (c * h * w) :=
  fun u => cnxPostP h w p s y (flatConv (h := h) (w := w) p.pW p.pB (gf.map (cExp * h * w) u))

/-- The block after the channel LN. -/
noncomputable def cnxPostN (gf : GeluForm) (h w : Nat) (p : CnxTieBlk c cExp) (s : Option ℝ)
    (y : Vec (c * h * w)) : Vec (c * h * w) → Vec (c * h * w) :=
  fun u => cnxPostE gf h w p s y (flatConv (h := h) (w := w) p.eW p.eB u)

/-- The block after the depthwise conv. -/
noncomputable def cnxPostD (gf : GeluForm) (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (s : Option ℝ)
    (y : Vec (c * h * w)) : Vec (c * h * w) → Vec (c * h * w) :=
  fun u => cnxPostN gf h w p s y (chanLNTensor3 c h w ε p.nG p.nB u)

/-- The depthwise conv's output. -/
noncomputable def cnxActD (h w : Nat) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) :=
  depthwiseFlat (h := h) (w := w) p.aW p.aB y

/-- The channel LN's output. -/
noncomputable def cnxActNl (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) :=
  chanLNTensor3 c h w ε p.nG p.nB (cnxActD h w p y)

/-- The GELU's output (the project conv's input). -/
noncomputable def cnxActG (gf : GeluForm) (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (cExp * h * w) :=
  gf.map (cExp * h * w) (flatConv (h := h) (w := w) p.eW p.eB (cnxActNl h w ε p y))

/-- The project conv's output (the layer scale's input). -/
noncomputable def cnxActP (gf : GeluForm) (h w : Nat) (ε : ℝ) (p : CnxTieBlk c cExp) (y : Vec (c * h * w)) :
    Vec (c * h * w) :=
  flatConv (h := h) (w := w) p.pW p.pB (cnxActG gf h w ε p y)

theorem cnxPostP_differentiable (h w : Nat) (p : CnxTieBlk c cExp) (s : Option ℝ) (y : Vec (c * h * w)) :
    Differentiable ℝ (cnxPostP h w p s y) :=
  ((dropScalarOpt_differentiable s).comp (layerScale_differentiable _)).add (differentiable_const y)

theorem cnxPostE_differentiable {gf : GeluForm} (h w : Nat) (p : CnxTieBlk c cExp) (s : Option ℝ)
    (y : Vec (c * h * w)) : Differentiable ℝ (cnxPostE gf h w p s y) :=
  (cnxPostP_differentiable h w p s y).comp
    ((flatConv_differentiable p.pW p.pB).comp (gf.map_differentiable _))

theorem cnxPostN_differentiable {gf : GeluForm} (h w : Nat) (p : CnxTieBlk c cExp) (s : Option ℝ)
    (y : Vec (c * h * w)) : Differentiable ℝ (cnxPostN gf h w p s y) :=
  (cnxPostE_differentiable h w p s y).comp (flatConv_differentiable p.eW p.eB)

theorem cnxPostD_differentiable {gf : GeluForm} (h w : Nat) (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (s : Option ℝ) (y : Vec (c * h * w)) : Differentiable ℝ (cnxPostD gf h w ε p s y) :=
  (cnxPostN_differentiable h w p s y).comp (chanLNTensor3_differentiable c h w ε p.nG p.nB hε)

/-- **A block's cotangents are loss gradients**, per example at its drop site `s`: from the gradient
    `dy` at the block output, the loss read after each activation has the tie's cotangent there —
    `s ⊙ dy` at the layer scale's output, `cnxCotP` of it at the project conv's, then `blkCotE`,
    `blkCotN`, `blkCotD` at `s ⊙ dy`. -/
theorem cnxBlk_hasGradAt {gf : GeluForm} {h w : Nat} (ε : ℝ) (hε : 0 < ε) (p : CnxTieBlk c cExp)
    (s : Option ℝ) (y dy : Vec (c * h * w)) {G : Vec (c * h * w) → Vec 1}
    (hG : HasGradAt G (p.fwdOD gf (h := h) (w := w) ε s y) dy) :
    HasGradAt (fun u => G (fun i => siteScale s (u i) + y i))
        (layerScale (fun k => p.sL (chanIdx c h w k)) (cnxActP gf h w ε p y)) (dropScalarOpt s dy)
      ∧ HasGradAt (fun u => G (cnxPostP h w p s y u)) (cnxActP gf h w ε p y)
          (cnxCotP (fun k => p.sL (chanIdx c h w k)) (dropScalarOpt s dy))
      ∧ HasGradAt (fun u => G (cnxPostE gf h w p s y u))
          (flatConv (h := h) (w := w) p.eW p.eB (cnxActNl h w ε p y))
          (blkCotE gf (h := h) (w := w) ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y (dropScalarOpt s dy))
      ∧ HasGradAt (fun u => G (cnxPostN gf h w p s y u)) (cnxActNl h w ε p y)
          (blkCotN gf (h := h) (w := w) ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y (dropScalarOpt s dy))
      ∧ HasGradAt (fun u => G (cnxPostD gf h w ε p s y u)) (cnxActD h w p y)
          (blkCotD gf (h := h) (w := w) ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y (dropScalarOpt s dy)) := by
  have hL : HasGradAt (fun u => G (fun i => siteScale s (u i) + y i))
      (layerScale (fun k => p.sL (chanIdx c h w k)) (cnxActP gf h w ε p y)) (dropScalarOpt s dy) :=
    HasGradAt.comp (f := fun u i => dropScalarOpt s u i + y i)
      (x := layerScale (fun k => p.sL (chanIdx c h w k)) (cnxActP gf h w ε p y)) hG
      (((dropScalarOpt_differentiable s) _).add_const y)
      (addConstHasVJPAt (dropScalarOpt s) y _ ((dropScalarOpt_differentiable s) _)
        ((dropScalarOptHasVJP s).toHasVJPAt _))
  have hP : HasGradAt (fun u => G (cnxPostP h w p s y u)) (cnxActP gf h w ε p y)
      (cnxCotP (fun k => p.sL (chanIdx c h w k)) (dropScalarOpt s dy)) :=
    HasGradAt.comp_global (f := layerScale (fun k => p.sL (chanIdx c h w k)))
      (x := cnxActP gf h w ε p y) hL (layerScale_differentiable _) (layerScaleHasVJP _)
  have hGg := HasGradAt.comp_global (f := flatConv (h := h) (w := w) p.pW p.pB)
    (x := cnxActG gf h w ε p y) hP (flatConv_differentiable p.pW p.pB) (flatConvHasVJP p.pW p.pB)
  have hE := HasGradAt.comp_global (f := gf.map (cExp * h * w))
    (x := flatConv (h := h) (w := w) p.eW p.eB (cnxActNl h w ε p y)) hGg (gf.map_differentiable _)
    (gf.hasVJP _)
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
def cnxBlockLossTiedGB (gf : GeluForm) (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : CnxTieBlk c cExp) (bf16 : Bool) (sd : Option (Vec N)) (xin : Vec (N * (c * h * w)))
    (Φ : CnxTieBlk c cExp → Vec 1) (dyOut : Vec (N * (c * h * w))) : Prop :=
  let γlsB : Vec (c * h * w) := fun k => p.sL (chanIdx c h w k)
  -- the drop site: the whole branch reads `s ⊙ dyOut`
  let dyOutD : Vec (N * (c * h * w)) := dropPathOpt N (c * h * w) sd dyOut
  -- forward activations
  let dB  : Vec (N * (c * h * w))    := batchMap N (depthwiseFlat (h := h) (w := w) p.aW p.aB) xin
  let nlB : Vec (N * (c * h * w))    := batchMap N (chanLNTensor3 c h w ε p.nG p.nB) dB
  let gB  : Vec (N * (cExp * h * w)) :=
    batchMap N (fun nl => gf.map (cExp * h * w) (flatConv (h := h) (w := w) p.eW p.eB nl)) nlB
  let pB  : Vec (N * (c * h * w))    := batchMap N (flatConv (h := h) (w := w) p.pW p.pB) gB
  -- backward chain cotangents
  let cotPB : Vec (N * (c * h * w))    := batchMap N (cnxCotP γlsB) dyOutD
  let cotEB : Vec (N * (cExp * h * w)) :=
    batchMapAux N (blkCotE gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin dyOutD
  let cotNB : Vec (N * (c * h * w))    :=
    batchMapAux N (blkCotN gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin dyOutD
  let cotDB : Vec (N * (c * h * w))    :=
    batchMapAux N (blkCotD gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin dyOutD
  -- depthwise 7×7 W/b
  (HasGradAt (fun θ => Φ { p with aW := Tensor3.unflatten θ }) (Tensor3.flatten p.aW)
        (den (SHlo.depthwiseWeightGradBAt bf16 id xN p.aB xin p.aW (.operand cotN cotDB))))
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
        (den (SHlo.convWeightGradBAt bf16 id xN p.eB nlB p.eW (.operand cotN cotEB))))
  ∧ (HasGradAt (fun θ => Φ { p with eB := θ }) p.eB
        (den (SHlo.convBiasGradB (h := h) (w := w) p.eW nlB p.eB (.operand cotN cotEB))))
  -- project 1×1 conv W/b
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradBAt bf16 id xN p.pB gB p.pW (.operand cotN cotPB))))
  ∧ (HasGradAt (fun θ => Φ { p with pB := θ }) p.pB
        (den (SHlo.convBiasGradB (h := h) (w := w) p.pW gB p.pB (.operand cotN cotPB))))
  -- per-channel layer-scale γ
  ∧ (HasGradAt (fun θ => Φ { p with sL := θ }) p.sL
        (den (SHlo.layerScaleChGammaGradB (N := N) (c := c) (h := h) (w := w) xN pB
          (.operand cotN dyOutD))))

theorem cnx_block_lossTiedGB {gf : GeluForm} (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (hε : 0 < ε) (p : CnxTieBlk c cExp) (bf16 : Bool) (sd : Option (Vec N)) (xin : Vec (N * (c * h * w)))
    {Lb : Vec (N * (c * h * w)) → Vec 1} {dyOut : Vec (N * (c * h * w))}
    (hLb : HasGradAt Lb (batchMapIdx N (fun n => p.fwdOD gf (h := h) (w := w) ε (exampleSite sd n)) xin) dyOut)
    {Φ : CnxTieBlk c cExp → Vec 1}
    (hΦ : ∀ p', Φ p' = Lb (batchMapIdx N (fun n => p'.fwdOD gf (h := h) (w := w) ε (exampleSite sd n)) xin)) :
    cnxBlockLossTiedGB gf N xN epsStr cotN ε p bf16 sd xin Φ dyOut := by
  rw [show Φ = fun p' => Lb (batchMapIdx N (fun n => p'.fwdOD gf (h := h) (w := w) ε (exampleSite sd n)) xin)
    from funext hΦ]
  have hc := fun (n : Fin N) y dy => cnxBlk_hasGradAt (gf := gf) (h := h) (w := w) ε hε p (exampleSite sd n) y dy
    (hasGradAt_linLoss dy _)
  -- the cotangents' slices: example `n`'s chain at its own dropped cotangent
  have hCD : ∀ n, batchSlice N (c * h * w)
      (batchMapAux N (blkCotD gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin
        (dropPathOpt N (c * h * w) sd dyOut)) n
      = blkCotD gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL (batchSlice N (c * h * w) xin n)
          (dropScalarOpt (exampleSite sd n) (batchSlice N (c * h * w) dyOut n)) := fun n => by
    rw [batchSlice_batchMapAux, batchSlice_dropPathOpt]
  have hCN : ∀ n, batchSlice N (c * h * w)
      (batchMapAux N (blkCotN gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin
        (dropPathOpt N (c * h * w) sd dyOut)) n
      = blkCotN gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL (batchSlice N (c * h * w) xin n)
          (dropScalarOpt (exampleSite sd n) (batchSlice N (c * h * w) dyOut n)) := fun n => by
    rw [batchSlice_batchMapAux, batchSlice_dropPathOpt]
  have hCE : ∀ n, batchSlice N (cExp * h * w)
      (batchMapAux N (blkCotE gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL) xin
        (dropPathOpt N (c * h * w) sd dyOut)) n
      = blkCotE gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL (batchSlice N (c * h * w) xin n)
          (dropScalarOpt (exampleSite sd n) (batchSlice N (c * h * w) dyOut n)) := fun n => by
    rw [batchSlice_batchMapAux, batchSlice_dropPathOpt]
  have hCP : ∀ n, batchSlice N (c * h * w)
      (batchMap N (cnxCotP (fun k => p.sL (chanIdx c h w k))) (dropPathOpt N (c * h * w) sd dyOut)) n
      = cnxCotP (fun k => p.sL (chanIdx c h w k))
          (dropScalarOpt (exampleSite sd n) (batchSlice N (c * h * w) dyOut n)) := fun n => by
    rw [batchSlice_batchMap, batchSlice_dropPathOpt]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => y)
        (fun θ y => depthwiseFlat (h := h) (w := w) (Tensor3.unflatten θ : DepthwiseKernel c 7 7) p.aB y)
        (fun n => cnxPostD gf h w ε p (exampleSite sd n)) (fun n y dy => blkCotD gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y
          (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := Tensor3.flatten p.aW) (by rw [Tensor3.unflatten_flatten]; exact hLb)
        (fun y => (depthwise_weight_differentiable p.aB (Tensor3.unflatten y)) _)
        (fun n y => cnxPostD_differentiable h w ε hε p (exampleSite sd n) y)
        (fun n y dy => by rw [Tensor3.unflatten_flatten]; exact (hc n y dy).2.2.2.2)
        xin _ (fun _ => rfl) hCD).of_eq
      ((funext fun idx => (GradNodeB.depthwiseWGradB_den xN cotN p.aB xin p.aW _ idx).symm).trans
        (Bf16Fold.den_depthwiseWeightGradBAt_id bf16 xN _ _ _ _).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => y)
        (fun θ y => depthwiseFlat (h := h) (w := w) p.aW θ y)
        (fun n => cnxPostD gf h w ε p (exampleSite sd n)) (fun n y dy => blkCotD gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y
          (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := p.aB) hLb
        (fun y => (depthwise_bias_differentiable p.aW (Tensor3.unflatten y)) _)
        (fun n y => cnxPostD_differentiable h w ε hε p (exampleSite sd n) y)
        (fun n y dy => (hc n y dy).2.2.2.2)
        xin _ (fun _ => rfl) hCD).of_eq
      (funext fun o => (GradNodeB.depthwiseBGradB_den cotN p.aW xin p.aB _ o).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActD h w p y)
        (fun θ d => chanLNTensor3 c h w ε θ p.nB d)
        (fun n => cnxPostN gf h w p (exampleSite sd n)) (fun n y dy => blkCotN gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y
          (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := p.nG) hLb
        (fun y => (chanLNTensor3_gamma_differentiable c h w ε p.nB y) _)
        (fun n y => cnxPostN_differentiable h w p (exampleSite sd n) y)
        (fun n y dy => (hc n y dy).2.2.2.1)
        _ _ (fun n => batchSlice_batchMap _ _ n) hCN).of_eq
      (funext fun k => (CnxFoldGB.chanLnGammaGradB_den xN epsStr cotN ε p.nB _ p.nG _ k).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActD h w p y)
        (fun θ d => chanLNTensor3 c h w ε p.nG θ d)
        (fun n => cnxPostN gf h w p (exampleSite sd n)) (fun n y dy => blkCotN gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y
          (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := p.nB) hLb
        (fun y => (chanLNTensor3_beta_differentiable c h w ε p.nG y) _)
        (fun n y => cnxPostN_differentiable h w p (exampleSite sd n) y)
        (fun n y dy => (hc n y dy).2.2.2.1)
        _ _ (fun n => batchSlice_batchMap _ _ n) hCN).of_eq
      (funext fun k => (CnxFoldGB.chanLnBetaGradB_den cotN ε p.nG _ p.nB _ k).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActNl h w ε p y)
        (fun θ nl => flatConv (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 cExp c 1 1) p.eB nl)
        (fun n => cnxPostE gf h w p (exampleSite sd n)) (fun n y dy => blkCotE gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y
          (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := Kernel4.flatten p.eW) (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (conv2d_weight_differentiable p.eB (Tensor3.unflatten y)) _)
        (fun n y => cnxPostE_differentiable h w p (exampleSite sd n) y)
        (fun n y dy => by rw [Kernel4.unflatten_flatten]; exact (hc n y dy).2.2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        hCE).of_eq
      ((funext fun idx => (GradNodeB.convWGradB_den xN cotN p.eB _ p.eW _ idx).symm).trans
        (Bf16Fold.den_convWeightGradBAt_id bf16 xN _ _ _ _).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActNl h w ε p y)
        (fun θ nl => flatConv (h := h) (w := w) p.eW θ nl)
        (fun n => cnxPostE gf h w p (exampleSite sd n)) (fun n y dy => blkCotE gf ε p.aW p.aB p.nG p.nB p.eW p.eB p.pW p.pB p.sL y
          (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := p.eB) hLb
        (fun y => (conv2d_bias_differentiable p.eW (Tensor3.unflatten y)) _)
        (fun n y => cnxPostE_differentiable h w p (exampleSite sd n) y)
        (fun n y dy => (hc n y dy).2.2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        hCE).of_eq
      (funext fun o => (GradNodeB.convBGradB_den cotN p.eW _ p.eB _ o).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActG gf h w ε p y)
        (fun θ g => flatConv (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 c cExp 1 1) p.pB g)
        (fun n => cnxPostP h w p (exampleSite sd n)) (fun n _ dy => cnxCotP (fun k => p.sL (chanIdx c h w k)) (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := Kernel4.flatten p.pW) (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (conv2d_weight_differentiable p.pB (Tensor3.unflatten y)) _)
        (fun n y => cnxPostP_differentiable h w p (exampleSite sd n) y)
        (fun n y dy => by rw [Kernel4.unflatten_flatten]; exact (hc n y dy).2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        hCP).of_eq
      ((funext fun idx => (GradNodeB.convWGradB_den xN cotN p.pB _ p.pW _ idx).symm).trans
        (Bf16Fold.den_convWeightGradBAt_id bf16 xN _ _ _ _).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActG gf h w ε p y)
        (fun θ g => flatConv (h := h) (w := w) p.pW θ g)
        (fun n => cnxPostP h w p (exampleSite sd n)) (fun n _ dy => cnxCotP (fun k => p.sL (chanIdx c h w k)) (dropScalarOpt (exampleSite sd n) dy))
        xin (θ := p.pB) hLb
        (fun y => (conv2d_bias_differentiable p.pW (Tensor3.unflatten y)) _)
        (fun n y => cnxPostP_differentiable h w p (exampleSite sd n) y)
        (fun n y dy => (hc n y dy).2.1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl)
        hCP).of_eq
      (funext fun o => (GradNodeB.convBGradB_den cotN p.pW _ p.pB _ o).symm)
  · exact (HasGradAt.param_batchMapIdx_through (fun _ y => cnxActP gf h w ε p y)
        (fun θ u => layerScale (fun k => θ (chanIdx c h w k)) u)
        (fun n y u i => siteScale (exampleSite sd n) (u i) + y i)
        (fun n _ dy => dropScalarOpt (exampleSite sd n) dy)
        xin (θ := p.sL) hLb
        (fun y => (layerScaleCh_gamma_differentiable c h w y) _)
        (fun n y => ((dropScalarOpt_differentiable (exampleSite sd n)).add (differentiable_const y)))
        (fun n y dy => (hc n y dy).1)
        _ _ (fun n => by simp only [batchSlice_batchMap]; rfl) (fun n => batchSlice_dropPathOpt _ _ n)).of_eq
      (funext fun cc => (CnxFoldGB.layerScaleChGammaGradB_den xN cotN _ p.sL _ cc).symm)

end Block

-- ════════════════════════════════════════════════════════════════
-- § A downsample, per example — channel LN at `2h × 2w`, then the 2×2/s2 conv
-- ════════════════════════════════════════════════════════════════

section Down
variable {ci co : Nat}

/-- **Downsample, every parameter node a loss derivative** — the four nodes `cnxDownChTiedGB`
    ties. -/
def cnxDownLossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : CnxTieDown ci co) (bf16 : Bool) (xin : Vec (N * (ci * (2 * h) * (2 * w))))
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
        (den (SHlo.convStridedWeightGradBAt bf16 id xN p.B nB p.W (.operand cotN dyOut))))
  ∧ (HasGradAt (fun θ => Φ { p with B := θ }) p.B
        (den (SHlo.convStridedBiasGradB (h := h) (w := w) p.W nB p.B (.operand cotN dyOut))))

theorem cnx_down_lossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (p : CnxTieDown ci co) (bf16 : Bool) (xin : Vec (N * (ci * (2 * h) * (2 * w))))
    {Lb : Vec (N * (co * h * w)) → Vec 1} {dyOut : Vec (N * (co * h * w))}
    (hLb : HasGradAt Lb (batchMap N (p.fwdO (h := h) (w := w) ε) xin) dyOut)
    {Φ : CnxTieDown ci co → Vec 1}
    (hΦ : ∀ p', Φ p' = Lb (batchMap N (p'.fwdO (h := h) (w := w) ε) xin)) :
    cnxDownLossTiedGB N xN epsStr cotN ε p bf16 xin Φ dyOut := by
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
      (funext fun k => (CnxFoldGB.chanLnGammaGradB_den xN epsStr cotN ε p.T xin p.G _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => chanLNTensor3 ci (2 * h) (2 * w) ε p.G θ y)
        (fun _ u => flatConvStride2 (h := h) (w := w) p.W p.B u)
        (fun y dy => dnCotN ε p.G p.T p.W p.B y dy) xin (θ := p.T) hLb
        (fun y => (chanLNTensor3_beta_differentiable ci (2 * h) (2 * w) ε p.G y) _)
        (fun _ => flatConvStride2_differentiable p.W p.B) hc
        xin _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      (funext fun k => (CnxFoldGB.chanLnBetaGradB_den cotN ε p.G xin p.T _ k).symm)
  · exact (HasGradAt.param_batchMap_through
        (fun y => chanLNTensor3 ci (2 * h) (2 * w) ε p.G p.T y)
        (fun θ n => flatConvStride2 (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 co ci 2 2) p.B n)
        (fun _ z => z) (fun _ dy => dy) xin (θ := Kernel4.flatten p.W)
        (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (GradNodeB.flatConvStride2_weight_differentiable p.B y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)).of_eq
      ((funext fun idx => (GradNodeB.convStridedWGradB_den xN cotN p.B _ p.W _ idx).symm).trans
        (Bf16Fold.den_convStridedWeightGradBAt_id bf16 xN _ _ _ _).symm)
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
    bias node is the emitted stride-1 `convBiasGradB`, which reads no input (stated at zero). -/
def cnxStemLossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (bf16 : Bool)
    (x : Vec (N * (3 * (2 * (2 * h)) * (2 * (2 * w)))))
    (Φ : Kernel4 c 3 4 4 → Vec c → Vec c → Vec c → Vec 1)
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
          Wst 0 psb (.operand cotN cotPatchB))))
  ∧ (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) psb psng psnbt) (Kernel4.flatten Wst)
        (den (SHlo.convStride4WeightGradBAt bf16 id xN psb x Wst (.operand cotN cotPatchB))))

theorem cnx_stem_lossTiedGB (N : Nat) {h w : Nat} (xN epsStr cotN : String) (ε : ℝ) (hε : 0 < ε)
    (Wst : Kernel4 c 3 4 4) (psb psng psnbt : Vec c) (bf16 : Bool)
    (x : Vec (N * (3 * (2 * (2 * h)) * (2 * (2 * w)))))
    {Lb : Vec (N * (c * h * w)) → Vec 1}
    {dyStem : Vec (N * (c * h * w))}
    (hLb : HasGradAt Lb (batchMap N (cnxStemFwdO (h := h) (w := w) ε Wst psb psng psnbt) x) dyStem)
    {Φ : Kernel4 c 3 4 4 → Vec c → Vec c → Vec c → Vec 1}
    (hΦ : ∀ W b γ β, Φ W b γ β = Lb (batchMap N (cnxStemFwdO (h := h) (w := w) ε W b γ β) x)) :
    cnxStemLossTiedGB N xN epsStr cotN ε Wst psb psng psnbt bf16 x Φ dyStem := by
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
      (funext fun k => (CnxFoldGB.chanLnGammaGradB_den xN epsStr cotN ε psnbt _ psng _ k).symm)
  · exact (HasGradAt.param_batchMap_through (fun y => flatConvStride4 (h := h) (w := w) Wst psb y)
        (fun θ u => chanLNTensor3 c h w ε psng θ u) (fun _ z => z) (fun _ dy => dy) x (θ := psnbt)
        hLb (fun y => (chanLNTensor3_beta_differentiable c h w ε psng y) _)
        (fun _ => differentiable_id) (fun _ dy => hasGradAt_linLoss dy _)
        _ _ (fun n => batchSlice_batchMap _ _ n) (fun _ => rfl)).of_eq
      (funext fun k => (CnxFoldGB.chanLnBetaGradB_den cotN ε psng _ psnbt _ k).symm)
  · -- the emitted bias node reads the channel sum; any conv's bias Jacobian is the indicator
    refine (HasGradAt.param_batchMap_through (fun y => y)
      (fun θ y => flatConvStride4 (h := h) (w := w) Wst θ y)
      (fun _ u => chanLNTensor3 c h w ε psng psnbt u) (fun y dy => stemCotPatch ε Wst psb psng y dy)
      x (θ := psb) hLb (fun y => (flatConvStride4_bias_differentiable Wst y) _)
      (fun _ => chanLNTensor3_differentiable c h w ε psng psnbt hε) hc
      x _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq (funext fun o => ?_)
    have hb : ∀ n j,
        pdiv (fun b' : Vec c => Tensor3.flatten (conv2d Wst b'
            (Tensor3.unflatten (batchSlice N (3 * h * w) (0 : Vec (N * (3 * h * w))) n)))) psb o j
          = pdiv (fun b' : Vec c => (flatConvStride4 Wst b'
              (batchSlice N (3 * (2 * (2 * h)) * (2 * (2 * w))) x n) : Vec (c * h * w))) psb o j :=
      fun n j => (GradNodeB.pdiv_flatConvStride4_bias_eq_conv2d Wst _ _ psb o j).symm
    refine (Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl fun j _ =>
      congrArg (· * _) (hb n j)).symm.trans ?_
    exact (GradNodeB.convBGradB_den cotN Wst 0 psb _ o).symm
  · exact (HasGradAt.param_batchMap_through (fun y => y)
        (fun θ y => flatConvStride4 (h := h) (w := w) (Kernel4.unflatten θ : Kernel4 c 3 4 4) psb y)
        (fun _ u => chanLNTensor3 c h w ε psng psnbt u) (fun y dy => stemCotPatch ε Wst psb psng y dy)
        x (θ := Kernel4.flatten Wst) (by rw [Kernel4.unflatten_flatten]; exact hLb)
        (fun y => (flatConvStride4_weight_differentiable psb y) _)
        (fun _ => chanLNTensor3_differentiable c h w ε psng psnbt hε)
        (fun y dy => by rw [Kernel4.unflatten_flatten]; exact hc y dy)
        x _ (fun _ => rfl) (fun n => batchSlice_batchMapAux _ _ _ n)).of_eq
      ((funext fun idx => (GradNodeB.psWGradB_den xN cotN psb x Wst _ idx).symm).trans
        (Bf16Fold.den_convStride4WeightGradBAt_id bf16 xN _ _ _ _).symm)

end Stem

-- ════════════════════════════════════════════════════════════════
-- § The head — GAP, the head LN at one row, the dense classifier
-- ════════════════════════════════════════════════════════════════

section Head

/-- The head per example: GAP, then LayerNorm at one row, then the dense classifier. -/
noncomputable def cnxHeadO (h w : Nat) {nC : Nat} (ε : ℝ) (hng hnbt : Vec 768) (Wfc : Mat 768 nC)
    (bfc : Vec nC) : Vec (768 * h * w) → Vec nC :=
  dense Wfc bfc ∘ rowLNVecFlat 1 768 ε hng hnbt ∘ globalAvgPoolFlat 768 h w

/-- **Head, every parameter node a loss derivative** — the four nodes `cnxHeadChTiedGB` ties. -/
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
      (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g)))

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
      (funext fun i => (GradNodeB.headBGradB_den cotN Wfc
        (batchSlice N 768 (batchMap N (rowLNVecFlat 1 768 ε hng hnbt)
          (batchMap N (globalAvgPoolFlat 768 h w) xhead))) bfc g i).symm)

end Head

-- ════════════════════════════════════════════════════════════════
-- § The whole net: the prefix before each stage, the loss after it, the net with one stage varied
-- ════════════════════════════════════════════════════════════════

/-- Pull the loss gradient back through a batched block's certified VJP at its drop site: the
    cotangent is the tie's `batchMapAuxIdx N (p.cotInD ε …)` (`cnxBlockCotInB_eq_vjp`). -/
theorem cnxBlkB_hasGradAt_comp {gf : GeluForm} (N : Nat) {c cExp h w : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : CnxTieBlk c cExp) (sd : Option (Vec N)) (X : Vec (N * (c * h * w)))
    {G : Vec (N * (c * h * w)) → Vec 1} {dY : Vec (N * (c * h * w))}
    (hG : HasGradAt G (batchMapIdx N (fun n => p.fwdOD gf (h := h) (w := w) ε (exampleSite sd n)) X) dY) :
    HasGradAt (fun y => G (batchMapIdx N (fun n => p.fwdOD gf (h := h) (w := w) ε (exampleSite sd n)) y)) X
      (batchMapAuxIdx N (fun n => p.cotInD gf (h := h) (w := w) ε (exampleSite sd n)) X dY) :=
  (HasGradAt.comp (x := X) hG
    (batchMapIdx_differentiableAt _ X (fun _ => fwdOD_differentiable ε hε p _ _))
    (batchMapIdxHasVJPAt _ X (fun n => (p.fwdODHasVJP gf ε hε (exampleSite sd n)).toHasVJPAt _)
      (fun _ => fwdOD_differentiable ε hε p _ _))).of_eq
    (congrFun (cnxBlockCotInB_eq_vjp N ε hε p sd X) dY).symm

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

/-- **ConvNeXt-T, batched**: the tie's forward, stage by stage — `batchMap N` of the stem,
    `batchMapIdx N` of each block at its site, `batchMap N` of each downsample and the head. At
    `sd = none` it is `batchMap N (convNextForwardTCh (w.toCh ε))`
    (`cnxNetB_eq_convNextForwardTCh`). -/
noncomputable def cnxNetB (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    Vec (N * nC) :=
  batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc)
    (batchMapIdx N (fun n => w.b18.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n))
    (batchMapIdx N (fun n => w.b17.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n))
    (batchMapIdx N (fun n => w.b16.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n))
    (batchMap N (w.d2.fwdO (h := 7) (w := 7) ε)
    (batchMapIdx N (fun n => w.b15.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n))
    (batchMapIdx N (fun n => w.b14.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n))
    (batchMapIdx N (fun n => w.b13.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n))
    (batchMapIdx N (fun n => w.b12.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n))
    (batchMapIdx N (fun n => w.b11.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n))
    (batchMapIdx N (fun n => w.b10.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n))
    (batchMapIdx N (fun n => w.b9.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n))
    (batchMapIdx N (fun n => w.b8.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n))
    (batchMapIdx N (fun n => w.b7.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n))
    (batchMap N (w.d1.fwdO (h := 14) (w := 14) ε)
    (batchMapIdx N (fun n => w.b6.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n))
    (batchMapIdx N (fun n => w.b5.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n))
    (batchMapIdx N (fun n => w.b4.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n))
    (batchMap N (w.d0.fwdO (h := 28) (w := 28) ε)
    (batchMapIdx N (fun n => w.b3.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n))
    (batchMapIdx N (fun n => w.b2.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n))
    (batchMapIdx N (fun n => w.b1.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n))
    (batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x))))))))))))))))))))))

/-- The stem's output — block `b1`'s input (the tie's `ib1`). -/
noncomputable def cnxPreS (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ)

/-- Stage `b1`'s output. -/
noncomputable def cnxPreB1 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMapIdx N (fun n => w.b1.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n)) ∘ cnxPreS N ε w

/-- Stage `b2`'s output. -/
noncomputable def cnxPreB2 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMapIdx N (fun n => w.b2.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n)) ∘ cnxPreB1 gf N ε w sd

/-- Stage `b3`'s output. -/
noncomputable def cnxPreB3 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (96 * 56 * 56)) :=
  batchMapIdx N (fun n => w.b3.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n)) ∘ cnxPreB2 gf N ε w sd

/-- Stage `d0`'s output. -/
noncomputable def cnxPreD0 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) ∘ cnxPreB3 gf N ε w sd

/-- Stage `b4`'s output. -/
noncomputable def cnxPreB4 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMapIdx N (fun n => w.b4.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n)) ∘ cnxPreD0 gf N ε w sd

/-- Stage `b5`'s output. -/
noncomputable def cnxPreB5 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMapIdx N (fun n => w.b5.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n)) ∘ cnxPreB4 gf N ε w sd

/-- Stage `b6`'s output. -/
noncomputable def cnxPreB6 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 28 * 28)) :=
  batchMapIdx N (fun n => w.b6.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n)) ∘ cnxPreB5 gf N ε w sd

/-- Stage `d1`'s output. -/
noncomputable def cnxPreD1 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) ∘ cnxPreB6 gf N ε w sd

/-- Stage `b7`'s output. -/
noncomputable def cnxPreB7 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b7.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n)) ∘ cnxPreD1 gf N ε w sd

/-- Stage `b8`'s output. -/
noncomputable def cnxPreB8 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b8.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n)) ∘ cnxPreB7 gf N ε w sd

/-- Stage `b9`'s output. -/
noncomputable def cnxPreB9 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b9.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n)) ∘ cnxPreB8 gf N ε w sd

/-- Stage `b10`'s output. -/
noncomputable def cnxPreB10 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b10.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n)) ∘ cnxPreB9 gf N ε w sd

/-- Stage `b11`'s output. -/
noncomputable def cnxPreB11 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b11.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n)) ∘ cnxPreB10 gf N ε w sd

/-- Stage `b12`'s output. -/
noncomputable def cnxPreB12 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b12.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n)) ∘ cnxPreB11 gf N ε w sd

/-- Stage `b13`'s output. -/
noncomputable def cnxPreB13 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b13.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n)) ∘ cnxPreB12 gf N ε w sd

/-- Stage `b14`'s output. -/
noncomputable def cnxPreB14 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b14.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n)) ∘ cnxPreB13 gf N ε w sd

/-- Stage `b15`'s output. -/
noncomputable def cnxPreB15 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (384 * 14 * 14)) :=
  batchMapIdx N (fun n => w.b15.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n)) ∘ cnxPreB14 gf N ε w sd

/-- Stage `d2`'s output. -/
noncomputable def cnxPreD2 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) ∘ cnxPreB15 gf N ε w sd

/-- Stage `b16`'s output. -/
noncomputable def cnxPreB16 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMapIdx N (fun n => w.b16.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n)) ∘ cnxPreD2 gf N ε w sd

/-- Stage `b17`'s output. -/
noncomputable def cnxPreB17 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMapIdx N (fun n => w.b17.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n)) ∘ cnxPreB16 gf N ε w sd

/-- Stage `b18`'s output. -/
noncomputable def cnxPreB18 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (768 * 7 * 7)) :=
  batchMapIdx N (fun n => w.b18.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n)) ∘ cnxPreB17 gf N ε w sd

theorem cnxPreS_apply (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreS N ε w x = batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x := by
  rw [cnxPreS]

theorem cnxPreB1_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB1 gf N ε w sd x = batchMapIdx N (fun n => w.b1.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n)) (cnxPreS N ε w x) := by
  rw [cnxPreB1, Function.comp_apply]

theorem cnxPreB2_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB2 gf N ε w sd x = batchMapIdx N (fun n => w.b2.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n)) (cnxPreB1 gf N ε w sd x) := by
  rw [cnxPreB2, Function.comp_apply]

theorem cnxPreB3_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB3 gf N ε w sd x = batchMapIdx N (fun n => w.b3.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n)) (cnxPreB2 gf N ε w sd x) := by
  rw [cnxPreB3, Function.comp_apply]

theorem cnxPreD0_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreD0 gf N ε w sd x = batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) (cnxPreB3 gf N ε w sd x) := by
  rw [cnxPreD0, Function.comp_apply]

theorem cnxPreB4_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB4 gf N ε w sd x = batchMapIdx N (fun n => w.b4.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n)) (cnxPreD0 gf N ε w sd x) := by
  rw [cnxPreB4, Function.comp_apply]

theorem cnxPreB5_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB5 gf N ε w sd x = batchMapIdx N (fun n => w.b5.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n)) (cnxPreB4 gf N ε w sd x) := by
  rw [cnxPreB5, Function.comp_apply]

theorem cnxPreB6_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB6 gf N ε w sd x = batchMapIdx N (fun n => w.b6.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n)) (cnxPreB5 gf N ε w sd x) := by
  rw [cnxPreB6, Function.comp_apply]

theorem cnxPreD1_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreD1 gf N ε w sd x = batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) (cnxPreB6 gf N ε w sd x) := by
  rw [cnxPreD1, Function.comp_apply]

theorem cnxPreB7_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB7 gf N ε w sd x = batchMapIdx N (fun n => w.b7.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n)) (cnxPreD1 gf N ε w sd x) := by
  rw [cnxPreB7, Function.comp_apply]

theorem cnxPreB8_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB8 gf N ε w sd x = batchMapIdx N (fun n => w.b8.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n)) (cnxPreB7 gf N ε w sd x) := by
  rw [cnxPreB8, Function.comp_apply]

theorem cnxPreB9_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB9 gf N ε w sd x = batchMapIdx N (fun n => w.b9.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n)) (cnxPreB8 gf N ε w sd x) := by
  rw [cnxPreB9, Function.comp_apply]

theorem cnxPreB10_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB10 gf N ε w sd x = batchMapIdx N (fun n => w.b10.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n)) (cnxPreB9 gf N ε w sd x) := by
  rw [cnxPreB10, Function.comp_apply]

theorem cnxPreB11_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB11 gf N ε w sd x = batchMapIdx N (fun n => w.b11.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n)) (cnxPreB10 gf N ε w sd x) := by
  rw [cnxPreB11, Function.comp_apply]

theorem cnxPreB12_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB12 gf N ε w sd x = batchMapIdx N (fun n => w.b12.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n)) (cnxPreB11 gf N ε w sd x) := by
  rw [cnxPreB12, Function.comp_apply]

theorem cnxPreB13_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB13 gf N ε w sd x = batchMapIdx N (fun n => w.b13.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n)) (cnxPreB12 gf N ε w sd x) := by
  rw [cnxPreB13, Function.comp_apply]

theorem cnxPreB14_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB14 gf N ε w sd x = batchMapIdx N (fun n => w.b14.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n)) (cnxPreB13 gf N ε w sd x) := by
  rw [cnxPreB14, Function.comp_apply]

theorem cnxPreB15_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB15 gf N ε w sd x = batchMapIdx N (fun n => w.b15.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n)) (cnxPreB14 gf N ε w sd x) := by
  rw [cnxPreB15, Function.comp_apply]

theorem cnxPreD2_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreD2 gf N ε w sd x = batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) (cnxPreB15 gf N ε w sd x) := by
  rw [cnxPreD2, Function.comp_apply]

theorem cnxPreB16_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB16 gf N ε w sd x = batchMapIdx N (fun n => w.b16.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n)) (cnxPreD2 gf N ε w sd x) := by
  rw [cnxPreB16, Function.comp_apply]

theorem cnxPreB17_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB17 gf N ε w sd x = batchMapIdx N (fun n => w.b17.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n)) (cnxPreB16 gf N ε w sd x) := by
  rw [cnxPreB17, Function.comp_apply]

theorem cnxPreB18_apply {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxPreB18 gf N ε w sd x = batchMapIdx N (fun n => w.b18.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n)) (cnxPreB17 gf N ε w sd x) := by
  rw [cnxPreB18, Function.comp_apply]

/-- The net after block `b18` — the head. -/
noncomputable def cnxSufB18 (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc)

/-- The net after stage `b17`: stage `b18`, then the rest. -/
noncomputable def cnxSufB17 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  fun y => cnxSufB18 N ε w (batchMapIdx N (fun n => w.b18.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n)) y)

/-- The net after stage `b16`: stage `b17`, then the rest. -/
noncomputable def cnxSufB16 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  fun y => cnxSufB17 gf N ε w sd (batchMapIdx N (fun n => w.b17.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n)) y)

/-- The net after stage `d2`: stage `b16`, then the rest. -/
noncomputable def cnxSufD2 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (768 * 7 * 7)) → Vec (N * nC) :=
  fun y => cnxSufB16 gf N ε w sd (batchMapIdx N (fun n => w.b16.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n)) y)

/-- The net after stage `b15`: stage `d2`, then the rest. -/
noncomputable def cnxSufB15 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufD2 gf N ε w sd (batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) y)

/-- The net after stage `b14`: stage `b15`, then the rest. -/
noncomputable def cnxSufB14 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB15 gf N ε w sd (batchMapIdx N (fun n => w.b15.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n)) y)

/-- The net after stage `b13`: stage `b14`, then the rest. -/
noncomputable def cnxSufB13 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB14 gf N ε w sd (batchMapIdx N (fun n => w.b14.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n)) y)

/-- The net after stage `b12`: stage `b13`, then the rest. -/
noncomputable def cnxSufB12 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB13 gf N ε w sd (batchMapIdx N (fun n => w.b13.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n)) y)

/-- The net after stage `b11`: stage `b12`, then the rest. -/
noncomputable def cnxSufB11 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB12 gf N ε w sd (batchMapIdx N (fun n => w.b12.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n)) y)

/-- The net after stage `b10`: stage `b11`, then the rest. -/
noncomputable def cnxSufB10 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB11 gf N ε w sd (batchMapIdx N (fun n => w.b11.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n)) y)

/-- The net after stage `b9`: stage `b10`, then the rest. -/
noncomputable def cnxSufB9 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB10 gf N ε w sd (batchMapIdx N (fun n => w.b10.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n)) y)

/-- The net after stage `b8`: stage `b9`, then the rest. -/
noncomputable def cnxSufB8 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB9 gf N ε w sd (batchMapIdx N (fun n => w.b9.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n)) y)

/-- The net after stage `b7`: stage `b8`, then the rest. -/
noncomputable def cnxSufB7 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB8 gf N ε w sd (batchMapIdx N (fun n => w.b8.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n)) y)

/-- The net after stage `d1`: stage `b7`, then the rest. -/
noncomputable def cnxSufD1 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (384 * 14 * 14)) → Vec (N * nC) :=
  fun y => cnxSufB7 gf N ε w sd (batchMapIdx N (fun n => w.b7.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n)) y)

/-- The net after stage `b6`: stage `d1`, then the rest. -/
noncomputable def cnxSufB6 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufD1 gf N ε w sd (batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) y)

/-- The net after stage `b5`: stage `b6`, then the rest. -/
noncomputable def cnxSufB5 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufB6 gf N ε w sd (batchMapIdx N (fun n => w.b6.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n)) y)

/-- The net after stage `b4`: stage `b5`, then the rest. -/
noncomputable def cnxSufB4 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufB5 gf N ε w sd (batchMapIdx N (fun n => w.b5.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n)) y)

/-- The net after stage `d0`: stage `b4`, then the rest. -/
noncomputable def cnxSufD0 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (192 * 28 * 28)) → Vec (N * nC) :=
  fun y => cnxSufB4 gf N ε w sd (batchMapIdx N (fun n => w.b4.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n)) y)

/-- The net after stage `b3`: stage `d0`, then the rest. -/
noncomputable def cnxSufB3 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufD0 gf N ε w sd (batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) y)

/-- The net after stage `b2`: stage `b3`, then the rest. -/
noncomputable def cnxSufB2 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufB3 gf N ε w sd (batchMapIdx N (fun n => w.b3.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n)) y)

/-- The net after stage `b1`: stage `b2`, then the rest. -/
noncomputable def cnxSufB1 (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufB2 gf N ε w sd (batchMapIdx N (fun n => w.b2.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n)) y)

/-- The net after the stem: stage `b1`, then the rest. -/
noncomputable def cnxSufS (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) :
    Vec (N * (96 * 56 * 56)) → Vec (N * nC) :=
  fun y => cnxSufB1 gf N ε w sd (batchMapIdx N (fun n => w.b1.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n)) y)

/-- **The net with the stem's parameters varied** is the suffix after the stem at the varied stem. -/
theorem cnx_factor_stem {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 96 3 4 4) (b γ β : Vec 96) :
    cnxNetB gf N ε { w with sW := W, sb := b, sγ := γ, sβ := β } sd x
      = cnxSufS gf N ε w sd (batchMap N (cnxStemFwdO (h := 56) (w := 56) ε W b γ β) x) := rfl

/-- **The net with stage `b1`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b1 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 96 384) :
    cnxNetB gf N ε { w with b1 := p } sd x
      = cnxSufB1 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n)) (cnxPreS N ε w x)) := by
  rw [cnxPreS_apply]; rfl

/-- **The net with stage `b2`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b2 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 96 384) :
    cnxNetB gf N ε { w with b2 := p } sd x
      = cnxSufB2 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n)) (cnxPreB1 gf N ε w sd x)) := by
  rw [cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b3`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b3 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 96 384) :
    cnxNetB gf N ε { w with b3 := p } sd x
      = cnxSufB3 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n)) (cnxPreB2 gf N ε w sd x)) := by
  rw [cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `d0`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_d0 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieDown 96 192) :
    cnxNetB gf N ε { w with d0 := p } sd x
      = cnxSufD0 gf N ε w sd (batchMap N (p.fwdO (h := 28) (w := 28) ε) (cnxPreB3 gf N ε w sd x)) := by
  rw [cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b4`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b4 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 192 768) :
    cnxNetB gf N ε { w with b4 := p } sd x
      = cnxSufB4 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n)) (cnxPreD0 gf N ε w sd x)) := by
  rw [cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b5`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b5 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 192 768) :
    cnxNetB gf N ε { w with b5 := p } sd x
      = cnxSufB5 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n)) (cnxPreB4 gf N ε w sd x)) := by
  rw [cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b6`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b6 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 192 768) :
    cnxNetB gf N ε { w with b6 := p } sd x
      = cnxSufB6 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n)) (cnxPreB5 gf N ε w sd x)) := by
  rw [cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `d1`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_d1 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieDown 192 384) :
    cnxNetB gf N ε { w with d1 := p } sd x
      = cnxSufD1 gf N ε w sd (batchMap N (p.fwdO (h := 14) (w := 14) ε) (cnxPreB6 gf N ε w sd x)) := by
  rw [cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b7`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b7 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b7 := p } sd x
      = cnxSufB7 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n)) (cnxPreD1 gf N ε w sd x)) := by
  rw [cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b8`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b8 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b8 := p } sd x
      = cnxSufB8 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n)) (cnxPreB7 gf N ε w sd x)) := by
  rw [cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b9`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b9 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b9 := p } sd x
      = cnxSufB9 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n)) (cnxPreB8 gf N ε w sd x)) := by
  rw [cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b10`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b10 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b10 := p } sd x
      = cnxSufB10 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n)) (cnxPreB9 gf N ε w sd x)) := by
  rw [cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b11`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b11 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b11 := p } sd x
      = cnxSufB11 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n)) (cnxPreB10 gf N ε w sd x)) := by
  rw [cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b12`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b12 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b12 := p } sd x
      = cnxSufB12 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n)) (cnxPreB11 gf N ε w sd x)) := by
  rw [cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b13`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b13 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b13 := p } sd x
      = cnxSufB13 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n)) (cnxPreB12 gf N ε w sd x)) := by
  rw [cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b14`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b14 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b14 := p } sd x
      = cnxSufB14 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n)) (cnxPreB13 gf N ε w sd x)) := by
  rw [cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b15`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b15 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 384 1536) :
    cnxNetB gf N ε { w with b15 := p } sd x
      = cnxSufB15 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n)) (cnxPreB14 gf N ε w sd x)) := by
  rw [cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `d2`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_d2 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieDown 384 768) :
    cnxNetB gf N ε { w with d2 := p } sd x
      = cnxSufD2 gf N ε w sd (batchMap N (p.fwdO (h := 7) (w := 7) ε) (cnxPreB15 gf N ε w sd x)) := by
  rw [cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b16`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b16 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 768 3072) :
    cnxNetB gf N ε { w with b16 := p } sd x
      = cnxSufB16 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n)) (cnxPreD2 gf N ε w sd x)) := by
  rw [cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b17`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b17 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 768 3072) :
    cnxNetB gf N ε { w with b17 := p } sd x
      = cnxSufB17 gf N ε w sd (batchMapIdx N (fun n => p.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n)) (cnxPreB16 gf N ε w sd x)) := by
  rw [cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with stage `b18`'s weights varied** is the suffix after it at the varied stage. -/
theorem cnx_factor_b18 {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (p : CnxTieBlk 768 3072) :
    cnxNetB gf N ε { w with b18 := p } sd x
      = cnxSufB18 N ε w (batchMapIdx N (fun n => p.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n)) (cnxPreB17 gf N ε w sd x)) := by
  rw [cnxPreB17_apply, cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The net with the head varied** is the head at the varied parameters. -/
theorem cnx_factor_head {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (a b : Vec 768) (W : Mat 768 nC) (bb : Vec nC) :
    cnxNetB gf N ε { w with hG := a, hT := b, Wfc := W, bfc := bb } sd x
      = batchMap N (cnxHeadO 7 7 ε a b W bb) (cnxPreB18 gf N ε w sd x) := by
  rw [cnxPreB18_apply, cnxPreB17_apply, cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- The net's output is the head at block `b18`'s output. -/
theorem cnx_forward_eq_head {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    cnxNetB gf N ε w sd x = batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc) (cnxPreB18 gf N ε w sd x) := by
  rw [cnxPreB18_apply, cnxPreB17_apply, cnxPreB16_apply, cnxPreD2_apply, cnxPreB15_apply, cnxPreB14_apply, cnxPreB13_apply, cnxPreB12_apply, cnxPreB11_apply, cnxPreB10_apply, cnxPreB9_apply, cnxPreB8_apply, cnxPreB7_apply, cnxPreD1_apply, cnxPreB6_apply, cnxPreB5_apply, cnxPreB4_apply, cnxPreD0_apply, cnxPreB3_apply, cnxPreB2_apply, cnxPreB1_apply, cnxPreS_apply]; rfl

/-- **The logits the tie's loss cotangent reads are `cnxNetB`'s.** The tie spells the head as
    three batched ops (`batchMap_comp`). -/
theorem cnx_logitsB_eq {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    batchMap N (dense w.Wfc w.bfc) (batchMap N (rowLNVecFlat 1 768 ε w.hG w.hT)
      (batchMap N (globalAvgPoolFlat 768 7 7) (cnxPreB18 gf N ε w sd x))) = cnxNetB gf N ε w sd x := by
  rw [cnx_forward_eq_head, cnxHeadO, batchMap_comp, batchMap_comp]; rfl

/-- **Without stochastic depth `cnxNetB` is the canonical ConvNeXt-T forward, batched**: `convNextForwardTCh` at
    `w.toCh ε`, the FullT forward whose VJP is `convNextForwardTChHasVJP`, applied per example. The
    per-example identity is `CnxTieWeights.forward_eq_convNextForwardTCh`; `batchMap_comp` splits
    the batched composite into the capstone's stage-by-stage chain. -/
theorem cnxNetB_eq_convNextForwardTCh {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (w : CnxTieWeights nC)
    (x : Vec (N * (3 * 224 * 224))) :
    cnxNetB gf N ε w none x = batchMap N (convNextForwardTCh gf (w.toCh ε)) x := by
  have hper : ∀ y, convNextForwardTCh gf (w.toCh ε) y
      = (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc ∘ w.b18.fwdO gf (h := 7) (w := 7) ε
        ∘ w.b17.fwdO gf (h := 7) (w := 7) ε ∘ w.b16.fwdO gf (h := 7) (w := 7) ε
        ∘ w.d2.fwdO (h := 7) (w := 7) ε ∘ w.b15.fwdO gf (h := 14) (w := 14) ε
        ∘ w.b14.fwdO gf (h := 14) (w := 14) ε ∘ w.b13.fwdO gf (h := 14) (w := 14) ε
        ∘ w.b12.fwdO gf (h := 14) (w := 14) ε ∘ w.b11.fwdO gf (h := 14) (w := 14) ε
        ∘ w.b10.fwdO gf (h := 14) (w := 14) ε ∘ w.b9.fwdO gf (h := 14) (w := 14) ε
        ∘ w.b8.fwdO gf (h := 14) (w := 14) ε ∘ w.b7.fwdO gf (h := 14) (w := 14) ε
        ∘ w.d1.fwdO (h := 14) (w := 14) ε ∘ w.b6.fwdO gf (h := 28) (w := 28) ε
        ∘ w.b5.fwdO gf (h := 28) (w := 28) ε ∘ w.b4.fwdO gf (h := 28) (w := 28) ε
        ∘ w.d0.fwdO (h := 28) (w := 28) ε ∘ w.b3.fwdO gf (h := 56) (w := 56) ε
        ∘ w.b2.fwdO gf (h := 56) (w := 56) ε ∘ w.b1.fwdO gf (h := 56) (w := 56) ε
        ∘ cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) y := by
    intro y
    rw [← CnxTie.CnxTieWeights.forward_eq_convNextForwardTCh w ε y]
    simp only [Function.comp_apply, cnxHeadO, mnistLinear]
  rw [show convNextForwardTCh gf (w.toCh ε) = _ from funext hper, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp, batchMap_comp,
    batchMap_comp, batchMap_comp]
  have hb : ∀ {c cExp h w : Nat} (p : CnxTieBlk c cExp) (y : Vec (N * (c * h * w))),
      batchMapIdx N (fun n => p.fwdOD gf (h := h) (w := w) ε (exampleSite (none : Option (Vec N)) n)) y
        = batchMap N (p.fwdO gf (h := h) (w := w) ε) y := fun _ _ => rfl
  unfold cnxNetB
  simp only [cnxSd_none, hb, Function.comp_apply]

/-- **Every ConvNeXt-T parameter gradient node is the derivative of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 182 nodes
    `cnx_net_tiedGB` ties, each at the cotangent the tie threads to it from `g`, stated against
    `L` of `cnxNetB` with that one parameter varied. -/
def CnxNetLossTiedGB (gf : GeluForm) (xN epsStr cotN dN : String) (N : Nat) {nC : Nat} (ε : ℝ)
    (w : CnxTieWeights nC) (bf16 : Bool) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (L : Vec (N * nC) → Vec 1) (g : Vec (N * nC)) : Prop :=
  let dyO18 := batchMapAux N (cnxHeadDyXheadChN (h := 7) (w := 7) ε w.hG w.hT w.Wfc w.bfc)
    (cnxPreB18 gf N ε w sd x) g
  let dyO17 := batchMapAuxIdx N (fun n => w.b18.cotInD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n)) (cnxPreB17 gf N ε w sd x) dyO18
  let dyO16 := batchMapAuxIdx N (fun n => w.b17.cotInD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n)) (cnxPreB16 gf N ε w sd x) dyO17
  let dyD2 := batchMapAuxIdx N (fun n => w.b16.cotInD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n)) (cnxPreD2 gf N ε w sd x) dyO16
  let dyO15 := batchMapAux N (w.d2.cotIn (h := 7) (w := 7) ε) (cnxPreB15 gf N ε w sd x) dyD2
  let dyO14 := batchMapAuxIdx N (fun n => w.b15.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n)) (cnxPreB14 gf N ε w sd x) dyO15
  let dyO13 := batchMapAuxIdx N (fun n => w.b14.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n)) (cnxPreB13 gf N ε w sd x) dyO14
  let dyO12 := batchMapAuxIdx N (fun n => w.b13.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n)) (cnxPreB12 gf N ε w sd x) dyO13
  let dyO11 := batchMapAuxIdx N (fun n => w.b12.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n)) (cnxPreB11 gf N ε w sd x) dyO12
  let dyO10 := batchMapAuxIdx N (fun n => w.b11.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n)) (cnxPreB10 gf N ε w sd x) dyO11
  let dyO9 := batchMapAuxIdx N (fun n => w.b10.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n)) (cnxPreB9 gf N ε w sd x) dyO10
  let dyO8 := batchMapAuxIdx N (fun n => w.b9.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n)) (cnxPreB8 gf N ε w sd x) dyO9
  let dyO7 := batchMapAuxIdx N (fun n => w.b8.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n)) (cnxPreB7 gf N ε w sd x) dyO8
  let dyD1 := batchMapAuxIdx N (fun n => w.b7.cotInD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n)) (cnxPreD1 gf N ε w sd x) dyO7
  let dyO6 := batchMapAux N (w.d1.cotIn (h := 14) (w := 14) ε) (cnxPreB6 gf N ε w sd x) dyD1
  let dyO5 := batchMapAuxIdx N (fun n => w.b6.cotInD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n)) (cnxPreB5 gf N ε w sd x) dyO6
  let dyO4 := batchMapAuxIdx N (fun n => w.b5.cotInD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n)) (cnxPreB4 gf N ε w sd x) dyO5
  let dyD0 := batchMapAuxIdx N (fun n => w.b4.cotInD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n)) (cnxPreD0 gf N ε w sd x) dyO4
  let dyO3 := batchMapAux N (w.d0.cotIn (h := 28) (w := 28) ε) (cnxPreB3 gf N ε w sd x) dyD0
  let dyO2 := batchMapAuxIdx N (fun n => w.b3.cotInD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n)) (cnxPreB2 gf N ε w sd x) dyO3
  let dyO1 := batchMapAuxIdx N (fun n => w.b2.cotInD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n)) (cnxPreB1 gf N ε w sd x) dyO2
  let dyStem := batchMapAuxIdx N (fun n => w.b1.cotInD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n)) (cnxPreS N ε w x) dyO1
  cnxStemLossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε w.sW w.sb w.sγ w.sβ bf16 x
      (fun W b γ β => L (cnxNetB gf N ε { w with sW := W, sb := b, sγ := γ, sβ := β } sd x)) dyStem
  ∧ cnxBlockLossTiedGB gf N (h := 56) (w := 56) xN epsStr cotN ε w.b1 bf16 (cnxSd sd 0) (cnxPreS N ε w x)
      (fun p => L (cnxNetB gf N ε { w with b1 := p } sd x)) dyO1
  ∧ cnxBlockLossTiedGB gf N (h := 56) (w := 56) xN epsStr cotN ε w.b2 bf16 (cnxSd sd 1) (cnxPreB1 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b2 := p } sd x)) dyO2
  ∧ cnxBlockLossTiedGB gf N (h := 56) (w := 56) xN epsStr cotN ε w.b3 bf16 (cnxSd sd 2) (cnxPreB2 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b3 := p } sd x)) dyO3
  ∧ cnxDownLossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.d0 bf16 (cnxPreB3 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with d0 := p } sd x)) dyD0
  ∧ cnxBlockLossTiedGB gf N (h := 28) (w := 28) xN epsStr cotN ε w.b4 bf16 (cnxSd sd 3) (cnxPreD0 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b4 := p } sd x)) dyO4
  ∧ cnxBlockLossTiedGB gf N (h := 28) (w := 28) xN epsStr cotN ε w.b5 bf16 (cnxSd sd 4) (cnxPreB4 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b5 := p } sd x)) dyO5
  ∧ cnxBlockLossTiedGB gf N (h := 28) (w := 28) xN epsStr cotN ε w.b6 bf16 (cnxSd sd 5) (cnxPreB5 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b6 := p } sd x)) dyO6
  ∧ cnxDownLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.d1 bf16 (cnxPreB6 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with d1 := p } sd x)) dyD1
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b7 bf16 (cnxSd sd 6) (cnxPreD1 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b7 := p } sd x)) dyO7
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b8 bf16 (cnxSd sd 7) (cnxPreB7 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b8 := p } sd x)) dyO8
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b9 bf16 (cnxSd sd 8) (cnxPreB8 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b9 := p } sd x)) dyO9
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b10 bf16 (cnxSd sd 9) (cnxPreB9 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b10 := p } sd x)) dyO10
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b11 bf16 (cnxSd sd 10) (cnxPreB10 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b11 := p } sd x)) dyO11
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b12 bf16 (cnxSd sd 11) (cnxPreB11 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b12 := p } sd x)) dyO12
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b13 bf16 (cnxSd sd 12) (cnxPreB12 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b13 := p } sd x)) dyO13
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b14 bf16 (cnxSd sd 13) (cnxPreB13 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b14 := p } sd x)) dyO14
  ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b15 bf16 (cnxSd sd 14) (cnxPreB14 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b15 := p } sd x)) dyO15
  ∧ cnxDownLossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.d2 bf16 (cnxPreB15 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with d2 := p } sd x)) dyD2
  ∧ cnxBlockLossTiedGB gf N (h := 7) (w := 7) xN epsStr cotN ε w.b16 bf16 (cnxSd sd 15) (cnxPreD2 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b16 := p } sd x)) dyO16
  ∧ cnxBlockLossTiedGB gf N (h := 7) (w := 7) xN epsStr cotN ε w.b17 bf16 (cnxSd sd 16) (cnxPreB16 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b17 := p } sd x)) dyO17
  ∧ cnxBlockLossTiedGB gf N (h := 7) (w := 7) xN epsStr cotN ε w.b18 bf16 (cnxSd sd 17) (cnxPreB17 gf N ε w sd x)
      (fun p => L (cnxNetB gf N ε { w with b18 := p } sd x)) dyO18
  ∧ cnxHeadLossTiedGB N (h := 7) (w := 7) xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc (cnxPreB18 gf N ε w sd x)
      (fun a b W bb => L (cnxNetB gf N ε { w with hG := a, hT := b, Wfc := W, bfc := bb } sd x)) g

/-- **Every ConvNeXt-T parameter gradient node is the derivative of the loss in that parameter.**
    For any loss `L` of the logits with gradient `g` at the net's output, each of the 182 nodes
    `cnx_net_tiedGB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net, `cnxNetB` with that
    one parameter varied (a stem field, a block's or downsample's record `w.bk := p` with one slot
    changed, or a head field).

    Hypothesis: `0 < ε`, the LayerNorms' (the tie itself needs none). The loss enters only through
    `hL`; `cnx_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship. -/
theorem cnx_net_lossGrad {gf : GeluForm} (xN epsStr cotN dN : String) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : CnxTieWeights nC) (bf16 : Bool) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)} (hL : HasGradAt L (cnxNetB gf N ε w sd x) g) :
    CnxNetLossTiedGB gf xN epsStr cotN dN N ε w bf16 sd x L g := by
  unfold CnxNetLossTiedGB
  intro dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9 dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem
  have hL' : HasGradAt L (batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc) (cnxPreB18 gf N ε w sd x)) g :=
    hL.congr_point (cnx_forward_eq_head N ε w sd x)
  have hB18 : HasGradAt (fun y => L (cnxSufB18 N ε w y)) (cnxPreB18 gf N ε w sd x) dyO18 :=
    cnxHeadB_hasGradAt_comp N ε hε w.hG w.hT w.Wfc w.bfc _ hL'
  have hB17 : HasGradAt (fun y => L (cnxSufB17 gf N ε w sd y)) (cnxPreB17 gf N ε w sd x) dyO17 :=
    cnxBlkB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.b18 (cnxSd sd 17) _ (hB18.congr_point (cnxPreB18_apply N ε w sd x))
  have hB16 : HasGradAt (fun y => L (cnxSufB16 gf N ε w sd y)) (cnxPreB16 gf N ε w sd x) dyO16 :=
    cnxBlkB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.b17 (cnxSd sd 16) _ (hB17.congr_point (cnxPreB17_apply N ε w sd x))
  have hD2 : HasGradAt (fun y => L (cnxSufD2 gf N ε w sd y)) (cnxPreD2 gf N ε w sd x) dyD2 :=
    cnxBlkB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.b16 (cnxSd sd 15) _ (hB16.congr_point (cnxPreB16_apply N ε w sd x))
  have hB15 : HasGradAt (fun y => L (cnxSufB15 gf N ε w sd y)) (cnxPreB15 gf N ε w sd x) dyO15 :=
    cnxDownB_hasGradAt_comp N (h := 7) (w := 7) ε hε w.d2 _ (hD2.congr_point (cnxPreD2_apply N ε w sd x))
  have hB14 : HasGradAt (fun y => L (cnxSufB14 gf N ε w sd y)) (cnxPreB14 gf N ε w sd x) dyO14 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b15 (cnxSd sd 14) _ (hB15.congr_point (cnxPreB15_apply N ε w sd x))
  have hB13 : HasGradAt (fun y => L (cnxSufB13 gf N ε w sd y)) (cnxPreB13 gf N ε w sd x) dyO13 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b14 (cnxSd sd 13) _ (hB14.congr_point (cnxPreB14_apply N ε w sd x))
  have hB12 : HasGradAt (fun y => L (cnxSufB12 gf N ε w sd y)) (cnxPreB12 gf N ε w sd x) dyO12 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b13 (cnxSd sd 12) _ (hB13.congr_point (cnxPreB13_apply N ε w sd x))
  have hB11 : HasGradAt (fun y => L (cnxSufB11 gf N ε w sd y)) (cnxPreB11 gf N ε w sd x) dyO11 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b12 (cnxSd sd 11) _ (hB12.congr_point (cnxPreB12_apply N ε w sd x))
  have hB10 : HasGradAt (fun y => L (cnxSufB10 gf N ε w sd y)) (cnxPreB10 gf N ε w sd x) dyO10 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b11 (cnxSd sd 10) _ (hB11.congr_point (cnxPreB11_apply N ε w sd x))
  have hB9 : HasGradAt (fun y => L (cnxSufB9 gf N ε w sd y)) (cnxPreB9 gf N ε w sd x) dyO9 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b10 (cnxSd sd 9) _ (hB10.congr_point (cnxPreB10_apply N ε w sd x))
  have hB8 : HasGradAt (fun y => L (cnxSufB8 gf N ε w sd y)) (cnxPreB8 gf N ε w sd x) dyO8 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b9 (cnxSd sd 8) _ (hB9.congr_point (cnxPreB9_apply N ε w sd x))
  have hB7 : HasGradAt (fun y => L (cnxSufB7 gf N ε w sd y)) (cnxPreB7 gf N ε w sd x) dyO7 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b8 (cnxSd sd 7) _ (hB8.congr_point (cnxPreB8_apply N ε w sd x))
  have hD1 : HasGradAt (fun y => L (cnxSufD1 gf N ε w sd y)) (cnxPreD1 gf N ε w sd x) dyD1 :=
    cnxBlkB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.b7 (cnxSd sd 6) _ (hB7.congr_point (cnxPreB7_apply N ε w sd x))
  have hB6 : HasGradAt (fun y => L (cnxSufB6 gf N ε w sd y)) (cnxPreB6 gf N ε w sd x) dyO6 :=
    cnxDownB_hasGradAt_comp N (h := 14) (w := 14) ε hε w.d1 _ (hD1.congr_point (cnxPreD1_apply N ε w sd x))
  have hB5 : HasGradAt (fun y => L (cnxSufB5 gf N ε w sd y)) (cnxPreB5 gf N ε w sd x) dyO5 :=
    cnxBlkB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.b6 (cnxSd sd 5) _ (hB6.congr_point (cnxPreB6_apply N ε w sd x))
  have hB4 : HasGradAt (fun y => L (cnxSufB4 gf N ε w sd y)) (cnxPreB4 gf N ε w sd x) dyO4 :=
    cnxBlkB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.b5 (cnxSd sd 4) _ (hB5.congr_point (cnxPreB5_apply N ε w sd x))
  have hD0 : HasGradAt (fun y => L (cnxSufD0 gf N ε w sd y)) (cnxPreD0 gf N ε w sd x) dyD0 :=
    cnxBlkB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.b4 (cnxSd sd 3) _ (hB4.congr_point (cnxPreB4_apply N ε w sd x))
  have hB3 : HasGradAt (fun y => L (cnxSufB3 gf N ε w sd y)) (cnxPreB3 gf N ε w sd x) dyO3 :=
    cnxDownB_hasGradAt_comp N (h := 28) (w := 28) ε hε w.d0 _ (hD0.congr_point (cnxPreD0_apply N ε w sd x))
  have hB2 : HasGradAt (fun y => L (cnxSufB2 gf N ε w sd y)) (cnxPreB2 gf N ε w sd x) dyO2 :=
    cnxBlkB_hasGradAt_comp N (h := 56) (w := 56) ε hε w.b3 (cnxSd sd 2) _ (hB3.congr_point (cnxPreB3_apply N ε w sd x))
  have hB1 : HasGradAt (fun y => L (cnxSufB1 gf N ε w sd y)) (cnxPreB1 gf N ε w sd x) dyO1 :=
    cnxBlkB_hasGradAt_comp N (h := 56) (w := 56) ε hε w.b2 (cnxSd sd 1) _ (hB2.congr_point (cnxPreB2_apply N ε w sd x))
  have hS : HasGradAt (fun y => L (cnxSufS gf N ε w sd y)) (cnxPreS N ε w x) dyStem :=
    cnxBlkB_hasGradAt_comp N (h := 56) (w := 56) ε hε w.b1 (cnxSd sd 0) _ (hB1.congr_point (cnxPreB1_apply N ε w sd x))
  refine ⟨cnx_stem_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.sW w.sb w.sγ w.sβ bf16 x
      (hS.congr_point (cnxPreS_apply N ε w x)) (fun W b γ β => by rw [cnx_factor_stem]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.b1 bf16 (cnxSd sd 0) _
      (hB1.congr_point (cnxPreB1_apply N ε w sd x)) (fun p => by rw [cnx_factor_b1]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.b2 bf16 (cnxSd sd 1) _
      (hB2.congr_point (cnxPreB2_apply N ε w sd x)) (fun p => by rw [cnx_factor_b2]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε hε w.b3 bf16 (cnxSd sd 2) _
      (hB3.congr_point (cnxPreB3_apply N ε w sd x)) (fun p => by rw [cnx_factor_b3]), ?_⟩
  refine ⟨cnx_down_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.d0 bf16 _
      (hD0.congr_point (cnxPreD0_apply N ε w sd x)) (fun p => by rw [cnx_factor_d0]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε hε w.b4 bf16 (cnxSd sd 3) _
      (hB4.congr_point (cnxPreB4_apply N ε w sd x)) (fun p => by rw [cnx_factor_b4]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε hε w.b5 bf16 (cnxSd sd 4) _
      (hB5.congr_point (cnxPreB5_apply N ε w sd x)) (fun p => by rw [cnx_factor_b5]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε hε w.b6 bf16 (cnxSd sd 5) _
      (hB6.congr_point (cnxPreB6_apply N ε w sd x)) (fun p => by rw [cnx_factor_b6]), ?_⟩
  refine ⟨cnx_down_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.d1 bf16 _
      (hD1.congr_point (cnxPreD1_apply N ε w sd x)) (fun p => by rw [cnx_factor_d1]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b7 bf16 (cnxSd sd 6) _
      (hB7.congr_point (cnxPreB7_apply N ε w sd x)) (fun p => by rw [cnx_factor_b7]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b8 bf16 (cnxSd sd 7) _
      (hB8.congr_point (cnxPreB8_apply N ε w sd x)) (fun p => by rw [cnx_factor_b8]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b9 bf16 (cnxSd sd 8) _
      (hB9.congr_point (cnxPreB9_apply N ε w sd x)) (fun p => by rw [cnx_factor_b9]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b10 bf16 (cnxSd sd 9) _
      (hB10.congr_point (cnxPreB10_apply N ε w sd x)) (fun p => by rw [cnx_factor_b10]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b11 bf16 (cnxSd sd 10) _
      (hB11.congr_point (cnxPreB11_apply N ε w sd x)) (fun p => by rw [cnx_factor_b11]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b12 bf16 (cnxSd sd 11) _
      (hB12.congr_point (cnxPreB12_apply N ε w sd x)) (fun p => by rw [cnx_factor_b12]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b13 bf16 (cnxSd sd 12) _
      (hB13.congr_point (cnxPreB13_apply N ε w sd x)) (fun p => by rw [cnx_factor_b13]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b14 bf16 (cnxSd sd 13) _
      (hB14.congr_point (cnxPreB14_apply N ε w sd x)) (fun p => by rw [cnx_factor_b14]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε hε w.b15 bf16 (cnxSd sd 14) _
      (hB15.congr_point (cnxPreB15_apply N ε w sd x)) (fun p => by rw [cnx_factor_b15]), ?_⟩
  refine ⟨cnx_down_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.d2 bf16 _
      (hD2.congr_point (cnxPreD2_apply N ε w sd x)) (fun p => by rw [cnx_factor_d2]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε hε w.b16 bf16 (cnxSd sd 15) _
      (hB16.congr_point (cnxPreB16_apply N ε w sd x)) (fun p => by rw [cnx_factor_b16]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε hε w.b17 bf16 (cnxSd sd 16) _
      (hB17.congr_point (cnxPreB17_apply N ε w sd x)) (fun p => by rw [cnx_factor_b17]), ?_⟩
  refine ⟨cnx_block_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε hε w.b18 bf16 (cnxSd sd 17) _
      (hB18.congr_point (cnxPreB18_apply N ε w sd x)) (fun p => by rw [cnx_factor_b18]), ?_⟩
  exact cnx_head_lossTiedGB N (h := 7) (w := 7) xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc _ hL'
    (fun a b W bb => by rw [cnx_factor_head])

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLossDiv`, `g` the `softmaxDiv` cotangent the render emits — the
    tie's own `g`, whose logits are `cnxNetB N ε w x` (`cnx_logitsB_eq`). -/
theorem cnx_net_lossGrad_smoothedCE {gf : GeluForm} (xN epsStr cotN dN aStr negAK bStr logN ohN : String)
    (N : Nat) {nC : Nat} (hK : 0 < nC) (ε α B : ℝ) (hε : 0 < ε) (w : CnxTieWeights nC) (bf16 : Bool)
    (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC))
    (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    CnxNetLossTiedGB gf xN epsStr cotN dN N ε w bf16 sd x (smoothedBatchLossDiv N nC α B t)
      (den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN (cnxNetB gf N ε w sd x) t)) :=
  cnx_net_lossGrad xN epsStr cotN dN N ε hε w bf16 sd x
    ⟨(smoothedBatchLossDiv_differentiable N nC α B t) _,
      fun J => smoothedBatchLossDiv_grad N nC hK α B aStr negAK bStr logN ohN t _ ht J⟩

/-- **The emitted ConvNeXt-T step's gradient nodes ARE the loss's gradient, at one chain.** For
    each of the 182 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`cnx_net_tiedGB`), and the batched smoothed loss of `cnxNetB` with that one slot varied is
    differentiable there with the node as its gradient (`cnx_net_lossGrad_smoothedCE`). The tie
    spells each block input as its own let; the proof rewrites the loss side's `cnxPre*` into those
    lets (`cnxPreS_apply`, …) and the loss side's logits into the tie's (`cnx_logitsB_eq`). -/
theorem cnx_net_tied_lossGrad {gf : GeluForm} (N : Nat) {nC : Nat}
    (xN epsStr cotN dN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    (w : CnxTieWeights nC) (bf16 : Bool) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3*224*224))) (t : Vec (N * nC))
    (hK : 0 < nC) (hε : 0 < ε) (ht : ∀ n, ∑ k : Fin nC, batchSlice N nC t n k = 1) :
    -- forward block inputs (the prefixes of the committed render's forward)
    let ib1 : Vec (N * (96*56*56)) := batchMap N (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x
    let ib2 : Vec (N * (96*56*56)) := batchMapIdx N (fun n => w.b1.fwdOD gf ε (exampleSite (cnxSd sd 0) n)) ib1
    let ib3 : Vec (N * (96*56*56)) := batchMapIdx N (fun n => w.b2.fwdOD gf ε (exampleSite (cnxSd sd 1) n)) ib2
    let ibD0 : Vec (N * (96*56*56)) := batchMapIdx N (fun n => w.b3.fwdOD gf ε (exampleSite (cnxSd sd 2) n)) ib3
    let ib4 : Vec (N * (192*28*28)) := batchMap N (w.d0.fwdO (h := 28) (w := 28) ε) ibD0
    let ib5 : Vec (N * (192*28*28)) := batchMapIdx N (fun n => w.b4.fwdOD gf ε (exampleSite (cnxSd sd 3) n)) ib4
    let ib6 : Vec (N * (192*28*28)) := batchMapIdx N (fun n => w.b5.fwdOD gf ε (exampleSite (cnxSd sd 4) n)) ib5
    let ibD1 : Vec (N * (192*28*28)) := batchMapIdx N (fun n => w.b6.fwdOD gf ε (exampleSite (cnxSd sd 5) n)) ib6
    let ib7 : Vec (N * (384*14*14)) := batchMap N (w.d1.fwdO (h := 14) (w := 14) ε) ibD1
    let ib8 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b7.fwdOD gf ε (exampleSite (cnxSd sd 6) n)) ib7
    let ib9 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b8.fwdOD gf ε (exampleSite (cnxSd sd 7) n)) ib8
    let ib10 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b9.fwdOD gf ε (exampleSite (cnxSd sd 8) n)) ib9
    let ib11 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b10.fwdOD gf ε (exampleSite (cnxSd sd 9) n)) ib10
    let ib12 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b11.fwdOD gf ε (exampleSite (cnxSd sd 10) n)) ib11
    let ib13 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b12.fwdOD gf ε (exampleSite (cnxSd sd 11) n)) ib12
    let ib14 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b13.fwdOD gf ε (exampleSite (cnxSd sd 12) n)) ib13
    let ib15 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b14.fwdOD gf ε (exampleSite (cnxSd sd 13) n)) ib14
    let ibD2 : Vec (N * (384*14*14)) := batchMapIdx N (fun n => w.b15.fwdOD gf ε (exampleSite (cnxSd sd 14) n)) ib15
    let ib16 : Vec (N * (768*7*7)) := batchMap N (w.d2.fwdO (h := 7) (w := 7) ε) ibD2
    let ib17 : Vec (N * (768*7*7)) := batchMapIdx N (fun n => w.b16.fwdOD gf ε (exampleSite (cnxSd sd 15) n)) ib16
    let ib18 : Vec (N * (768*7*7)) := batchMapIdx N (fun n => w.b17.fwdOD gf ε (exampleSite (cnxSd sd 16) n)) ib17
    let xhead : Vec (N * (768*7*7)) := batchMapIdx N (fun n => w.b18.fwdOD gf ε (exampleSite (cnxSd sd 17) n)) ib18
    -- head forward + the SMOOTHED loss cotangent, at a general target `t`
    let gapB    : Vec (N * (1*768)) := batchMap N (globalAvgPoolFlat 768 7 7) xhead
    let hnB     : Vec (N * 768)     := batchMap N (rowLNVecFlat 1 768 ε w.hG w.hT) gapB
    let logitsB : Vec (N * nC)      := batchMap N (dense w.Wfc w.bfc) hnB
    let g       : Vec (N * nC)      :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    -- backward cotangents (composed from the loss; residual fan-in at each skip, LN-back at each
    -- downsample and at the stem)
    let dyO18 : Vec (N * (768*7*7)) := batchMapAux N (cnxHeadDyXheadChN (h := 7) (w := 7) ε w.hG w.hT w.Wfc w.bfc) xhead g
    let dyO17 : Vec (N * (768*7*7)) := batchMapAuxIdx N (fun n => w.b18.cotInD gf ε (exampleSite (cnxSd sd 17) n)) ib18 dyO18
    let dyO16 : Vec (N * (768*7*7)) := batchMapAuxIdx N (fun n => w.b17.cotInD gf ε (exampleSite (cnxSd sd 16) n)) ib17 dyO17
    let dyD2 : Vec (N * (768*7*7)) := batchMapAuxIdx N (fun n => w.b16.cotInD gf ε (exampleSite (cnxSd sd 15) n)) ib16 dyO16
    let dyO15 : Vec (N * (384*14*14)) := batchMapAux N (w.d2.cotIn (h := 7) (w := 7) ε) ibD2 dyD2
    let dyO14 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b15.cotInD gf ε (exampleSite (cnxSd sd 14) n)) ib15 dyO15
    let dyO13 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b14.cotInD gf ε (exampleSite (cnxSd sd 13) n)) ib14 dyO14
    let dyO12 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b13.cotInD gf ε (exampleSite (cnxSd sd 12) n)) ib13 dyO13
    let dyO11 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b12.cotInD gf ε (exampleSite (cnxSd sd 11) n)) ib12 dyO12
    let dyO10 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b11.cotInD gf ε (exampleSite (cnxSd sd 10) n)) ib11 dyO11
    let dyO9 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b10.cotInD gf ε (exampleSite (cnxSd sd 9) n)) ib10 dyO10
    let dyO8 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b9.cotInD gf ε (exampleSite (cnxSd sd 8) n)) ib9 dyO9
    let dyO7 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b8.cotInD gf ε (exampleSite (cnxSd sd 7) n)) ib8 dyO8
    let dyD1 : Vec (N * (384*14*14)) := batchMapAuxIdx N (fun n => w.b7.cotInD gf ε (exampleSite (cnxSd sd 6) n)) ib7 dyO7
    let dyO6 : Vec (N * (192*28*28)) := batchMapAux N (w.d1.cotIn (h := 14) (w := 14) ε) ibD1 dyD1
    let dyO5 : Vec (N * (192*28*28)) := batchMapAuxIdx N (fun n => w.b6.cotInD gf ε (exampleSite (cnxSd sd 5) n)) ib6 dyO6
    let dyO4 : Vec (N * (192*28*28)) := batchMapAuxIdx N (fun n => w.b5.cotInD gf ε (exampleSite (cnxSd sd 4) n)) ib5 dyO5
    let dyD0 : Vec (N * (192*28*28)) := batchMapAuxIdx N (fun n => w.b4.cotInD gf ε (exampleSite (cnxSd sd 3) n)) ib4 dyO4
    let dyO3 : Vec (N * (96*56*56)) := batchMapAux N (w.d0.cotIn (h := 28) (w := 28) ε) ibD0 dyD0
    let dyO2 : Vec (N * (96*56*56)) := batchMapAuxIdx N (fun n => w.b3.cotInD gf ε (exampleSite (cnxSd sd 2) n)) ib3 dyO3
    let dyO1 : Vec (N * (96*56*56)) := batchMapAuxIdx N (fun n => w.b2.cotInD gf ε (exampleSite (cnxSd sd 1) n)) ib2 dyO2
    let dyStem : Vec (N * (96*56*56)) := batchMapAuxIdx N (fun n => w.b1.cotInD gf ε (exampleSite (cnxSd sd 0) n)) ib1 dyO1
    let L := smoothedBatchLossDiv N nC α B t
    -- the stem, every block, every downsample, the head, the dense total-loss fold + loss cot
    (cnxStemChTiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ bf16 x dyStem
      ∧ cnxStemLossTiedGB N (h := 56) (w := 56) xN epsStr cotN ε w.sW w.sb w.sγ w.sβ bf16 x
        (fun W b γ β => L (cnxNetB gf N ε { w with sW := W, sb := b, sγ := γ, sβ := β } sd x)) dyStem)
  ∧ (w.b1.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 0) ib1 dyO1
      ∧ cnxBlockLossTiedGB gf N (h := 56) (w := 56) xN epsStr cotN ε w.b1 bf16 (cnxSd sd 0) ib1
        (fun p => L (cnxNetB gf N ε { w with b1 := p } sd x)) dyO1)
  ∧ (w.b2.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 1) ib2 dyO2
      ∧ cnxBlockLossTiedGB gf N (h := 56) (w := 56) xN epsStr cotN ε w.b2 bf16 (cnxSd sd 1) ib2
        (fun p => L (cnxNetB gf N ε { w with b2 := p } sd x)) dyO2)
  ∧ (w.b3.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 2) ib3 dyO3
      ∧ cnxBlockLossTiedGB gf N (h := 56) (w := 56) xN epsStr cotN ε w.b3 bf16 (cnxSd sd 2) ib3
        (fun p => L (cnxNetB gf N ε { w with b3 := p } sd x)) dyO3)
  ∧ (w.d0.TiedGB N xN epsStr cotN ε bf16 ibD0 dyD0
      ∧ cnxDownLossTiedGB N (h := 28) (w := 28) xN epsStr cotN ε w.d0 bf16 ibD0
        (fun p => L (cnxNetB gf N ε { w with d0 := p } sd x)) dyD0)
  ∧ (w.b4.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 3) ib4 dyO4
      ∧ cnxBlockLossTiedGB gf N (h := 28) (w := 28) xN epsStr cotN ε w.b4 bf16 (cnxSd sd 3) ib4
        (fun p => L (cnxNetB gf N ε { w with b4 := p } sd x)) dyO4)
  ∧ (w.b5.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 4) ib5 dyO5
      ∧ cnxBlockLossTiedGB gf N (h := 28) (w := 28) xN epsStr cotN ε w.b5 bf16 (cnxSd sd 4) ib5
        (fun p => L (cnxNetB gf N ε { w with b5 := p } sd x)) dyO5)
  ∧ (w.b6.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 5) ib6 dyO6
      ∧ cnxBlockLossTiedGB gf N (h := 28) (w := 28) xN epsStr cotN ε w.b6 bf16 (cnxSd sd 5) ib6
        (fun p => L (cnxNetB gf N ε { w with b6 := p } sd x)) dyO6)
  ∧ (w.d1.TiedGB N xN epsStr cotN ε bf16 ibD1 dyD1
      ∧ cnxDownLossTiedGB N (h := 14) (w := 14) xN epsStr cotN ε w.d1 bf16 ibD1
        (fun p => L (cnxNetB gf N ε { w with d1 := p } sd x)) dyD1)
  ∧ (w.b7.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 6) ib7 dyO7
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b7 bf16 (cnxSd sd 6) ib7
        (fun p => L (cnxNetB gf N ε { w with b7 := p } sd x)) dyO7)
  ∧ (w.b8.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 7) ib8 dyO8
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b8 bf16 (cnxSd sd 7) ib8
        (fun p => L (cnxNetB gf N ε { w with b8 := p } sd x)) dyO8)
  ∧ (w.b9.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 8) ib9 dyO9
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b9 bf16 (cnxSd sd 8) ib9
        (fun p => L (cnxNetB gf N ε { w with b9 := p } sd x)) dyO9)
  ∧ (w.b10.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 9) ib10 dyO10
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b10 bf16 (cnxSd sd 9) ib10
        (fun p => L (cnxNetB gf N ε { w with b10 := p } sd x)) dyO10)
  ∧ (w.b11.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 10) ib11 dyO11
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b11 bf16 (cnxSd sd 10) ib11
        (fun p => L (cnxNetB gf N ε { w with b11 := p } sd x)) dyO11)
  ∧ (w.b12.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 11) ib12 dyO12
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b12 bf16 (cnxSd sd 11) ib12
        (fun p => L (cnxNetB gf N ε { w with b12 := p } sd x)) dyO12)
  ∧ (w.b13.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 12) ib13 dyO13
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b13 bf16 (cnxSd sd 12) ib13
        (fun p => L (cnxNetB gf N ε { w with b13 := p } sd x)) dyO13)
  ∧ (w.b14.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 13) ib14 dyO14
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b14 bf16 (cnxSd sd 13) ib14
        (fun p => L (cnxNetB gf N ε { w with b14 := p } sd x)) dyO14)
  ∧ (w.b15.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 14) ib15 dyO15
      ∧ cnxBlockLossTiedGB gf N (h := 14) (w := 14) xN epsStr cotN ε w.b15 bf16 (cnxSd sd 14) ib15
        (fun p => L (cnxNetB gf N ε { w with b15 := p } sd x)) dyO15)
  ∧ (w.d2.TiedGB N xN epsStr cotN ε bf16 ibD2 dyD2
      ∧ cnxDownLossTiedGB N (h := 7) (w := 7) xN epsStr cotN ε w.d2 bf16 ibD2
        (fun p => L (cnxNetB gf N ε { w with d2 := p } sd x)) dyD2)
  ∧ (w.b16.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 15) ib16 dyO16
      ∧ cnxBlockLossTiedGB gf N (h := 7) (w := 7) xN epsStr cotN ε w.b16 bf16 (cnxSd sd 15) ib16
        (fun p => L (cnxNetB gf N ε { w with b16 := p } sd x)) dyO16)
  ∧ (w.b17.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 16) ib17 dyO17
      ∧ cnxBlockLossTiedGB gf N (h := 7) (w := 7) xN epsStr cotN ε w.b17 bf16 (cnxSd sd 16) ib17
        (fun p => L (cnxNetB gf N ε { w with b17 := p } sd x)) dyO17)
  ∧ (w.b18.TiedGB gf N xN epsStr cotN ε bf16 (cnxSd sd 17) ib18 dyO18
      ∧ cnxBlockLossTiedGB gf N (h := 7) (w := 7) xN epsStr cotN ε w.b18 bf16 (cnxSd sd 17) ib18
        (fun p => L (cnxNetB gf N ε { w with b18 := p } sd x)) dyO18)
  ∧ (cnxHeadChTiedGB N xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc xhead g
      ∧ cnxHeadLossTiedGB N (h := 7) (w := 7) xN epsStr cotN dN ε w.hG w.hT w.Wfc w.bfc xhead
        (fun a b W bb => L (cnxNetB gf N ε { w with hG := a, hT := b, Wfc := W, bfc := bb } sd x)) g) := by
  intro ib1 ib2 ib3 ibD0 ib4 ib5 ib6 ibD1 ib7 ib8 ib9 ib10 ib11 ib12 ib13 ib14 ib15 ibD2 ib16 ib17
    ib18 xhead gapB hnB logitsB g dyO18 dyO17 dyO16 dyD2 dyO15 dyO14 dyO13 dyO12 dyO11 dyO10 dyO9
    dyO8 dyO7 dyD1 dyO6 dyO5 dyO4 dyD0 dyO3 dyO2 dyO1 dyStem L
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
    t20, t21, t22⟩ :=
    cnx_net_tiedGB N xN epsStr cotN dN aStr negAK bStr logN ohN ε α B w bf16 sd x t
  have hl :=
    cnx_net_lossGrad_smoothedCE (gf := gf) xN epsStr cotN dN aStr negAK bStr logN ohN N hK ε α B hε w
      bf16 sd x t ht
  -- the loss side's activations and logits, in the tie's spelling
  have e0 : cnxPreS N ε w x = ib1 := by rw [cnxPreS_apply N ε w x]
  have e1 : cnxPreB1 gf N ε w sd x = ib2 := by rw [cnxPreB1_apply N ε w sd x, e0]
  have e2 : cnxPreB2 gf N ε w sd x = ib3 := by rw [cnxPreB2_apply N ε w sd x, e1]
  have e3 : cnxPreB3 gf N ε w sd x = ibD0 := by rw [cnxPreB3_apply N ε w sd x, e2]
  have e4 : cnxPreD0 gf N ε w sd x = ib4 := by rw [cnxPreD0_apply N ε w sd x, e3]
  have e5 : cnxPreB4 gf N ε w sd x = ib5 := by rw [cnxPreB4_apply N ε w sd x, e4]
  have e6 : cnxPreB5 gf N ε w sd x = ib6 := by rw [cnxPreB5_apply N ε w sd x, e5]
  have e7 : cnxPreB6 gf N ε w sd x = ibD1 := by rw [cnxPreB6_apply N ε w sd x, e6]
  have e8 : cnxPreD1 gf N ε w sd x = ib7 := by rw [cnxPreD1_apply N ε w sd x, e7]
  have e9 : cnxPreB7 gf N ε w sd x = ib8 := by rw [cnxPreB7_apply N ε w sd x, e8]
  have e10 : cnxPreB8 gf N ε w sd x = ib9 := by rw [cnxPreB8_apply N ε w sd x, e9]
  have e11 : cnxPreB9 gf N ε w sd x = ib10 := by rw [cnxPreB9_apply N ε w sd x, e10]
  have e12 : cnxPreB10 gf N ε w sd x = ib11 := by rw [cnxPreB10_apply N ε w sd x, e11]
  have e13 : cnxPreB11 gf N ε w sd x = ib12 := by rw [cnxPreB11_apply N ε w sd x, e12]
  have e14 : cnxPreB12 gf N ε w sd x = ib13 := by rw [cnxPreB12_apply N ε w sd x, e13]
  have e15 : cnxPreB13 gf N ε w sd x = ib14 := by rw [cnxPreB13_apply N ε w sd x, e14]
  have e16 : cnxPreB14 gf N ε w sd x = ib15 := by rw [cnxPreB14_apply N ε w sd x, e15]
  have e17 : cnxPreB15 gf N ε w sd x = ibD2 := by rw [cnxPreB15_apply N ε w sd x, e16]
  have e18 : cnxPreD2 gf N ε w sd x = ib16 := by rw [cnxPreD2_apply N ε w sd x, e17]
  have e19 : cnxPreB16 gf N ε w sd x = ib17 := by rw [cnxPreB16_apply N ε w sd x, e18]
  have e20 : cnxPreB17 gf N ε w sd x = ib18 := by rw [cnxPreB17_apply N ε w sd x, e19]
  have e21 : cnxPreB18 gf N ε w sd x = xhead := by rw [cnxPreB18_apply N ε w sd x, e20]
  have eg : den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN
      (cnxNetB gf N ε w sd x) t) = g := by rw [← cnx_logitsB_eq, e21]
  unfold CnxNetLossTiedGB at hl
  rw [eg, e21, e20, e19, e18, e17, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3,
    e2, e1, e0] at hl
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19,
    l20, l21, l22⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, l12⟩, ⟨t13, l13⟩, ⟨t14, l14⟩, ⟨t15, l15⟩, ⟨t16, l16⟩,
    ⟨t17, l17⟩, ⟨t18, l18⟩, ⟨t19, l19⟩, ⟨t20, l20⟩, ⟨t21, l21⟩, ⟨t22, l22⟩⟩

end Proofs.CnxTieGB
