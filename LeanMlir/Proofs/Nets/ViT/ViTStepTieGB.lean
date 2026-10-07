import LeanMlir.Proofs.Nets.ViT.ViTFoldGB
import LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTieB
import LeanMlir.Proofs.Nets.ViT.ViTDropBlock

/-! # ViT-Tiny's step tie at the batched index, the un-fused gradient and the smoothed loss

`ViTStepTie.lean` ties all 200 parameters of the SGD-inline `vit_train_step.mlir`: each fused
`θ − lr·g` op `den`s to the certified step at the cotangent the emitted backward chain delivers,
per example, at a hard label. This file is that statement re-pointed along the three axes
`ConvNeXtStepTieGB.lean` moved for ConvNeXt, applied to ViT's per-example capstone.

**Axis 1 — the gradient node.** Every conjunct is at the raw gradient node (`*GradB`), which is
what `vit_adam_train_step.mlir` and every f32 `vitin_*` artifact emit; the fused op appears only
in the SGD-inline file. The optimizer update that consumes the node (AdamW, the `wx`/`clip`
variants, EMA) is outside this statement. `ViTFoldGB.lean` is the fold each conjunct delegates to.

**Precision is a flag on the statement.** The bf16 artifacts (`vitin_adamwxclipdropbf16` and its
EMA, data-parallel, ε and exact-GELU twins) emit the bf16 per-token dense weight gradient
(`rowDenseWeightGradBBf16`) at the six denses of every block, and — only if the render's
`bf16ConvW` is set, which no shipped artifact does — the bf16 patch-embed weight gradient
(`Foundation.Bf16GradNodes`). Both are stated on the renderers' switch:
`ViTFoldGB.RowDenseWTiedBAt bf16` and `patchEmbedWeightGradBAt bf16 id …`
(`StableHLO.PrecisionSwitch`), `false` the f32 node, `true` the bf16 one. The capstone
`vit_net_tiedGB` takes the renderer's two backward flags, `bf16` and `bf16ConvW`, and passes
`bf16 && bf16ConvW` to the embed as `vitBackAllB` does, so at `bf16 := true, bf16ConvW := false`
it reaches the shipped bf16 artifacts' gradient nodes read over ℝ exactly as it reads the f32 ones
(`Bf16Erasure`: at the identity rounding the bf16 kind denotes what its f32 peer does). The bias,
LayerNorm, CLS, position and classifier nodes carry no flag because no render switches them.

**Axis 2 — the loss.** `g` is a binder, instantiated at `smoothedLossCotGraphDiv` — the six-op
chain `expe → softmaxDiv → subB → scaleB → addVB → shiftB → divConstB` this render emits at the
plain width `N·K`, at a GENERAL target arriving as `%onehot`. The fused file pins it to
`softmax − oneHot`.

**Axis 3 — the index.** `N` is a binder. Every activation is a per-example lift of the prefix
the fused file threads (`patchEmbedFlat`, the block forward, the final LN, `clsSliceFlat`) and
every cotangent a per-example lift of the chain (`vitCotTowerOutV`, the block's input cotangent,
the `vitCot*` family). The lift is exact for this net because no ViT op couples examples —
LayerNorm, attention, GELU and the denses are all per-example, and the `*B` constructors' `den`
arms say so. A block's lift is the INDEXED one (`batchMapIdx` / `batchMapAuxIdx`,
`Foundation.Batched.Indexed`): with stochastic depth example `n` runs the block at its own mask
entries. `nC` is a binder too (10 on Imagenette, 1000 on ImageNet).

**Stochastic depth is a binder.** `sd : Option (Fin 12 → Vec N × Vec N)` is the render's `drop`
flag: `none` the drop-free artifacts' chain — every family constant, so the lifts are `batchMap` /
`batchMapAux` and the statement is the drop-free one by `rfl` — and `some` the `*drop*` artifacts',
the book's `vitin_emadp128x4wxclipdropeps0000001erfbf16` among them: block `k`'s two sites (`vitSdA` / `vitSdM`,
the `%dp<2k>` / `%dp<2k+1>` inputs) scale the out-projection's and fc2's outputs by example `n`'s
entries (`BlockParamsV.fwdOD`), the out-projection and fc2 weight and bias nodes read the dropped
cotangent (`dropPathOpt`, where `ViTRenderB` emits `dropPathB` on it), and both skip fan-ins read
the raw one (`BlockParamsV.cotInD`, `ViTDropBlock`).

**The CLS-token conjunct.** The CLS token is one
shared `[192]` vector; its gradient is the sum of every example's CLS-row cotangent. The fused
file's `vit_cls_den` is at `denseBiasSgdB (N := 1)` — "sum one thing", correct there because
`pretty B` performed the batch lift outside the AST. `vitEmbedTiedGB`'s third conjunct is
`ViTFoldGB.clsGrad_denB` at the batched node, `denseBiasGradB (N := N)` on `batchMap N clsSliceFlat`
of the embed cotangent, with the batch sum inside `den`.

## Relation to the per-example file

Every save, every chain cotangent and every Jacobian witness is `ViTStepTie.lean`'s, lifted; every
conjunct's proof is one `ViTFoldGB.*_den` lemma. The per-example saves and internal cotangents are
repackaged as functions of the block INPUT and its two sites (`blkSaves`, `cAtt` … `cM1`) so that
the indexed lift has something to lift — the `let` chains of `vitBlockTiedAtMHV` and
`vitBlockCotInAtMHV` with the sites in.
ViT has no `*BackBatchedGraph_faithful` family and needs none here: the `batchMap` lift is the
batched statement, as it was for ConvNeXt.

**Conventions carried unchanged from the fused file:** the VECTOR LayerNorm (`γ β : Vec D`) at
all 25 sites, 3 heads × d_head 64, depth 12, D 192, MLP 768, 16×16 patches, GELU (no kink — no
smoothness hypothesis anywhere). Stated at ViT-Tiny's literal dims; S and B are other nets.

**Scope.** One replica: in `vitin_adamdp128x4*` (four replicas of 128) every gradient node feeds
`allReduceMeanF`, and `DataParallel.Node` composes the per-replica statement with the replica
mean.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTTieGB

open scoped BigOperators
open Proofs.ViTTie (vitBlockFwdOMHV vitBlockCotInAtMHV ViTTieWeights)
open Proofs.ViTFoldGB (rowDenseBTiedB_holds rowDenseWTiedBAt_holds)
open Proofs.GradNodeB (vecLNBetaTiedB_holds vecLNGammaTiedB_holds)

/-! ## Per-example saves and internal cotangents as functions of a block's INPUT

`vitBlockTiedMHV` takes the nine saved activations as arguments; `batchMapAux` lifts a function
of (one saved value, one input), so the batched block tie needs each save as a function of `xin`
and each internal cotangent as a function of `(xin, dyOut)` — exactly the `let` chains of
`ViTTie.vitBlockTiedAtMHV` and `vitBlockCotInAtMHV`, packaged. -/

/-- The nine saved activations of one multi-head block, flattened. -/
structure BlkSaves (Np1 heads d mlpDim : Nat) where
  ln1 : Vec (Np1 * (heads * d))
  q   : Vec (Np1 * (heads * d))
  k   : Vec (Np1 * (heads * d))
  v   : Vec (Np1 * (heads * d))
  att : Vec (Np1 * (heads * d))
  h   : Vec (Np1 * (heads * d))
  ln2 : Vec (Np1 * (heads * d))
  m1  : Vec (Np1 * mlpDim)
  g   : Vec (Np1 * mlpDim)

/-- The saves from the block input — `vitBlockTiedAtMHV`'s `let` chain, verbatim. -/
noncomputable def blkSaves (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (sA : Option ℝ)
    (xin : Vec (Np1 * (heads * d))) : BlkSaves Np1 heads d mlpDim :=
  let X    : Mat Np1 (heads * d) := Mat.unflatten xin
  let ln1  : Mat Np1 (heads * d) := fun r kk => layerScale γ1 (fun s => layerNormForward (heads * d) ε 1 0 (X r) s) kk + β1 kk
  let Q    : Mat Np1 (heads * d) := fun r => dense Wq bq (ln1 r)
  let K    : Mat Np1 (heads * d) := fun r => dense Wk bk (ln1 r)
  let V    : Mat Np1 (heads * d) := fun r => dense Wv bv (ln1 r)
  let att  : Mat Np1 (heads * d) := ∑ hh : Fin heads, headPadMat Np1 heads d hh
    (Mat.mul (rowSoftmax (fun i j => sdpaScale d *
        Mat.mul (headSliceMat Np1 heads d hh Q) (Mat.transpose (headSliceMat Np1 heads d hh K)) i j))
      (headSliceMat Np1 heads d hh V))
  let h    : Mat Np1 (heads * d) := fun r s => X r s + siteScale sA (dense Wo bo (att r) s)
  let ln2  : Mat Np1 (heads * d) := fun r kk => layerScale γ2 (fun s => layerNormForward (heads * d) ε 1 0 (h r) s) kk + β2 kk
  let m1   : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g    : Mat Np1 mlpDim := fun r => gf.map mlpDim (m1 r)
  ⟨Mat.flatten ln1, Mat.flatten Q, Mat.flatten K, Mat.flatten V, Mat.flatten att, Mat.flatten h,
   Mat.flatten ln2, Mat.flatten m1, Mat.flatten g⟩

/-- Per example, the attention-output cotangent (`vitCotAttV`), from the block input and output cotangent. -/
noncomputable def cAtt (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut

/-- Per example, the Q cotangent, per head (`vitCotDQmh`), from the block input and output cotangent. -/
noncomputable def cQ (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotDQmh Np1 heads d s.q s.k s.v (vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut)

/-- Per example, the K cotangent, per head, from the block input and output cotangent. -/
noncomputable def cK (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotDKmh Np1 heads d s.q s.k s.v (vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut)

/-- Per example, the V cotangent, per head, from the block input and output cotangent. -/
noncomputable def cV (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotDVmh Np1 heads d s.q s.k s.v (vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut)

/-- Per example, the LN₁-output cotangent: the three-way Q/K/V fan-in (`vitCotLn1`), from the block input and output cotangent. -/
noncomputable def cLn1 (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotLn1 Wq Wk Wv
    (vitCotDQmh Np1 heads d s.q s.k s.v (vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut))
    (vitCotDKmh Np1 heads d s.q s.k s.v (vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut))
    (vitCotDVmh Np1 heads d s.q s.k s.v (vitCotAttVD gf ε γ2 Wo Wfc1 Wfc2 sA sM s.h s.m1 dyOut))

/-- Per example, the MLP-residual fan-in at `h` (`vitCotHV`), from the block input and output cotangent. -/
noncomputable def cH (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotHVD gf ε γ2 Wfc1 Wfc2 sM s.h s.m1 dyOut

/-- Per example, the LN₂-output cotangent (`vitCotLn2`), from the block input and output cotangent. -/
noncomputable def cLn2 (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotLn2 gf Wfc1 Wfc2 s.m1 (dropScalarOpt sM dyOut)

/-- Per example, the fc1-output cotangent through the GELU mask (`vitCotM1`), from the block input and output cotangent. -/
noncomputable def cM1 (gf : GeluForm) {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (sA sM : Option ℝ) (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * mlpDim) :=
  let s := blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 sA xin
  vitCotM1 gf Wfc2 s.m1 (dropScalarOpt sM dyOut)

/-! ## The multi-head block — all 16 gradient nodes, batched -/

/-- **One multi-head vector-LN transformer block, tied at the batched gradient nodes.** Every one of
    the block's 16 params, fed the indexed lift of the cotangent the real backward chain delivers
    at its site, `den`otes the certified `Σ_n` gradient — the MLP-residual and attention-residual
    fan-ins and the three-way LN₁ fan-in, per head, exactly as `vitBlockTiedMHV` has them per
    example. At the two drop sites `sA sM` the out-projection's and fc2's nodes read the dropped
    cotangent (`dropPathOpt`); `none none` is the drop-free block. -/
def vitBlockTiedGB (gf : GeluForm) (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (bf16 : Bool) (sA sM : Option (Vec N)) (xin dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  -- forward saves — each the indexed lift of the per-example save the emitted node lifts
  let ln1B : Vec (N * (Np1 * (heads * d))) :=
    batchMapIdx N (fun n x => (blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 (exampleSite sA n) x).ln1) xin
  let attB : Vec (N * (Np1 * (heads * d))) :=
    batchMapIdx N (fun n x => (blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 (exampleSite sA n) x).att) xin
  let hB   : Vec (N * (Np1 * (heads * d))) :=
    batchMapIdx N (fun n x => (blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 (exampleSite sA n) x).h) xin
  let ln2B : Vec (N * (Np1 * (heads * d))) :=
    batchMapIdx N (fun n x => (blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 (exampleSite sA n) x).ln2) xin
  let gB   : Vec (N * (Np1 * mlpDim)) :=
    batchMapIdx N (fun n x => (blkSaves gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 (exampleSite sA n) x).g) xin
  -- backward chain cotangents — the indexed lift of the per-example chain
  let cotLn1B : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cLn1 gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let dQB     : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cQ gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let dKB     : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cK gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let dVB     : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cV gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let cotHB   : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cH gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let cotLn2B : Vec (N * (Np1 * (heads * d))) := batchMapAuxIdx N (fun n => cLn2 gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  let cotM1B  : Vec (N * (Np1 * mlpDim))      := batchMapAuxIdx N (fun n => cM1 gf ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 (exampleSite sA n) (exampleSite sM n)) xin dyOut
  -- the two sites on the branch cotangents (the skips read the raw ones)
  let cotOB   : Vec (N * (Np1 * (heads * d))) := dropPathOpt N (Np1 * (heads * d)) sA cotHB
  let dyOutD  : Vec (N * (Np1 * (heads * d))) := dropPathOpt N (Np1 * (heads * d)) sM dyOut
  -- LN₁ γ/β  (cot = cotLn1B, LN input = xin)
  GradNodeB.VecLNGammaTiedB N Np1 xN epsStr cotN ε β1 xin γ1 cotLn1B
  ∧GradNodeB.VecLNBetaTiedB N Np1 cotN ε γ1 xin β1 cotLn1B
  -- Q dense W/b  (cot = dQB, dense input = ln1B)
  ∧ViTFoldGB.RowDenseWTiedBAt bf16 N Np1 xN cotN bq ln1B Wq dQB
  ∧ViTFoldGB.RowDenseBTiedB N Np1 cotN Wq ln1B bq dQB
  -- K dense W/b  (cot = dKB)
  ∧ViTFoldGB.RowDenseWTiedBAt bf16 N Np1 xN cotN bk ln1B Wk dKB
  ∧ViTFoldGB.RowDenseBTiedB N Np1 cotN Wk ln1B bk dKB
  -- V dense W/b  (cot = dVB)
  ∧ViTFoldGB.RowDenseWTiedBAt bf16 N Np1 xN cotN bv ln1B Wv dVB
  ∧ViTFoldGB.RowDenseBTiedB N Np1 cotN Wv ln1B bv dVB
  -- out-proj dense W/b  (cot = sA ⊙ cotHB, dense input = attB)
  ∧ViTFoldGB.RowDenseWTiedBAt bf16 N Np1 xN cotN bo attB Wo cotOB
  ∧ViTFoldGB.RowDenseBTiedB N Np1 cotN Wo attB bo cotOB
  -- LN₂ γ/β  (cot = cotLn2B, LN input = hB)
  ∧GradNodeB.VecLNGammaTiedB N Np1 xN epsStr cotN ε β2 hB γ2 cotLn2B
  ∧GradNodeB.VecLNBetaTiedB N Np1 cotN ε γ2 hB β2 cotLn2B
  -- fc1 dense W/b  (cot = cotM1B, dense input = ln2B)
  ∧ViTFoldGB.RowDenseWTiedBAt bf16 N Np1 xN cotN bfc1 ln2B Wfc1 cotM1B
  ∧ViTFoldGB.RowDenseBTiedB N Np1 cotN Wfc1 ln2B bfc1 cotM1B
  -- fc2 dense W/b  (cot = sM ⊙ dyOut, dense input = gB)
  ∧ViTFoldGB.RowDenseWTiedBAt bf16 N Np1 xN cotN bfc2 gB Wfc2 dyOutD
  ∧ViTFoldGB.RowDenseBTiedB N Np1 cotN Wfc2 gB bfc2 dyOutD

theorem vit_block_tiedGB {gf : GeluForm} (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (bf16 : Bool) (sA sM : Option (Vec N)) (xin dyOut : Vec (N * (Np1 * (heads * d)))) :
    vitBlockTiedGB gf N xN epsStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 bfc2
      bf16 sA sM xin dyOut := by
  unfold vitBlockTiedGB
  exact ⟨vecLNGammaTiedB_holds, vecLNBetaTiedB_holds, rowDenseWTiedBAt_holds bf16, rowDenseBTiedB_holds,
    rowDenseWTiedBAt_holds bf16, rowDenseBTiedB_holds, rowDenseWTiedBAt_holds bf16, rowDenseBTiedB_holds,
    rowDenseWTiedBAt_holds bf16, rowDenseBTiedB_holds, vecLNGammaTiedB_holds, vecLNBetaTiedB_holds,
    rowDenseWTiedBAt_holds bf16, rowDenseBTiedB_holds, rowDenseWTiedBAt_holds bf16, rowDenseBTiedB_holds⟩

/-! ## Final LN, classifier and patch embedding — batched -/

/-- **Final vector-LN γF/βF, tied at the batched classifier-back cotangent** `vitCotFl` per
    example (the `clsPad` of `Wclsᵀ g_n`, exactly what the render's `dotOut → clsPad` computes). -/
def vitFinalLNTiedGB (N : Nat) {nC : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (b12out : Vec (N * (197 * 192))) (g : Vec (N * nC)) :
    Prop :=
  let cotFlB : Vec (N * (197 * 192)) := batchMap N (vitCotFl 196 192 nC Wcls) g
  GradNodeB.VecLNGammaTiedB N 197 xN epsStr cotN ε βF b12out γF cotFlB
  ∧ GradNodeB.VecLNBetaTiedB N 197 cotN ε γF b12out βF cotFlB

theorem vit_finalLN_tiedGB (N : Nat) {nC : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (b12out : Vec (N * (197 * 192))) (g : Vec (N * nC)) :
    vitFinalLNTiedGB N xN epsStr cotN ε γF βF Wcls b12out g := by
  unfold vitFinalLNTiedGB
  intro cotFlB
  refine ⟨?_, ?_⟩
  · exact vecLNGammaTiedB_holds
  · intro i
    exact GradNodeB.rowDenseBiasGradB_den_lnbeta cotN ε γF
      (fun n => Mat.unflatten (batchSlice N (197 * 192) b12out n)) βF cotFlB i

/-- **Classifier Wcls/bcls, tied at the loss cotangent `g`** — the weight at the batched CLS row,
    the bias summed over the batch (`GradNodeB.headBGradB_den`). -/
def vitHeadTiedGB (N : Nat) {nC : Nat} (aN cotN : String)
    (hn : Vec (N * 192)) (Wcls : Mat 192 nC) (bcls : Vec nC) (g : Vec (N * nC)) : Prop :=
  (∀ (i : Fin 192) (j : Fin nC),
      den (SHlo.weightGradB (N := N) (m := 192) (n := nC) aN hn (.operand cotN g))
          (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ k : Fin nC,
            pdiv (fun v : Vec (192 * nC) => dense (Mat.unflatten v) bcls (batchSlice N 192 hn n))
                 (Mat.flatten Wcls) (finProdFinEquiv (i, j)) k * batchSlice N nC g n k)
  ∧ (∀ i : Fin nC,
      den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g)) i
        = ∑ n : Fin N, ∑ j : Fin nC,
            pdiv (fun b' : Vec nC => dense Wcls b' (batchSlice N 192 hn n)) bcls i j
              * batchSlice N nC g n j)

theorem vit_head_tiedGB (N : Nat) {nC : Nat} (aN cotN : String)
    (hn : Vec (N * 192)) (Wcls : Mat 192 nC) (bcls : Vec nC) (g : Vec (N * nC)) :
    vitHeadTiedGB N aN cotN hn Wcls bcls g := by
  unfold vitHeadTiedGB
  refine ⟨?_, ?_⟩
  · intro i j; exact GradNodeB.headWGradB_den aN cotN hn Wcls bcls g i j
  · intro i; exact GradNodeB.headBGradB_den cotN Wcls (batchSlice N 192 hn) bcls g i

/-- **Patch embed wConv/bConv/cls/pos, tied at the batched embed-output cotangent.** The third
    conjunct is the CLS token's gradient with the batch sum INSIDE `den` — the statement the
    per-example capstone made only at `N = 1`. -/
def vitEmbedTiedGB (N : Nat) (xN cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (bf16 : Bool) (img : Vec (N * (3 * 224 * 224))) (dyEmbed : Vec (N * (197 * 192))) : Prop :=
  (∀ (dd : Fin 192) (c : Fin 3) (kh kw : Fin 16),
      den (SHlo.patchEmbedWeightGradBAt bf16 (N := N) (ic := 3) (H := 224) (W := 224) (P := 16)
            (tk := 196) (D := 192) id xN img (.operand cotN dyEmbed))
          (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (dd, c), kh), kw))
        = ∑ n : Fin N, ∑ o : Fin ((196 + 1) * 192),
            pdiv (fun v : Vec (192 * 3 * 16 * 16) =>
                    patchEmbedFlat 3 224 224 16 196 192 (Kernel4.unflatten v) bc cls pos
                      (batchSlice N (3 * 224 * 224) img n))
              (Kernel4.flatten Wc)
              (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (dd, c), kh), kw)) o
              * batchSlice N ((196 + 1) * 192) dyEmbed n o)
  ∧ (∀ i : Fin 192,
      den (SHlo.patchEmbedBiasGradB (N := N) (tk := 196) (c := 192) (.operand cotN dyEmbed)) i
        = ∑ n : Fin N, ∑ o : Fin ((196 + 1) * 192),
            pdiv (fun b' : Vec 192 =>
                    patchEmbedFlat 3 224 224 16 196 192 Wc b' cls pos
                      (batchSlice N (3 * 224 * 224) img n)) bc i o
              * batchSlice N ((196 + 1) * 192) dyEmbed n o)
  ∧ (∀ i : Fin 192,
      den (SHlo.denseBiasGradB (N := N) (c := 192)
            (.operand cotN (batchMap N (clsSliceFlat 196 192) dyEmbed))) i
        = ∑ n : Fin N, ∑ j : Fin (197 * 192),
            pdiv (fun cl : Vec 192 =>
                    patchEmbedFlat 3 224 224 16 196 192 Wc bc cl pos
                      (batchSlice N (3 * 224 * 224) img n)) cls i j
              * batchSlice N (197 * 192) dyEmbed n j)
  ∧ (∀ i : Fin ((196 + 1) * 192),
      den (SHlo.posEmbedGradB (N := N) (tk := 196) (D := 192) (.operand cotN dyEmbed)) i
        = ∑ n : Fin N, ∑ o : Fin ((196 + 1) * 192),
            pdiv (fun p : Vec ((196 + 1) * 192) =>
                    patchEmbedFlat 3 224 224 16 196 192 Wc bc cls (Mat.unflatten p)
                      (batchSlice N (3 * 224 * 224) img n))
              (Mat.flatten pos) i o
              * batchSlice N ((196 + 1) * 192) dyEmbed n o)

theorem vit_embed_tiedGB (N : Nat) (xN cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (bf16 : Bool) (img : Vec (N * (3 * 224 * 224))) (dyEmbed : Vec (N * (197 * 192))) :
    vitEmbedTiedGB N xN cotN Wc bc cls pos bf16 img dyEmbed := by
  unfold vitEmbedTiedGB
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro dd c kh kw
    rw [Bf16Fold.den_patchEmbedWeightGradBAt_id]
    exact ViTFoldGB.patchEmbedWeightGradB_den xN cotN bc cls pos img Wc dyEmbed dd c kh kw
  · intro i; exact ViTFoldGB.patchEmbedBiasGradB_den cotN Wc bc cls pos img dyEmbed i
  · intro i; exact ViTFoldGB.clsGrad_denB cotN Wc bc cls pos img dyEmbed i
  · intro i; exact ViTFoldGB.posEmbedGradB_den cotN Wc bc cls pos img dyEmbed i

/-! ## Every cotangent the capstone threads is a certified VJP backward

The capstone below threads two per-example constructors: the head's `vitCotTowerOutV`
(`batchMapAux N`) and each block's `cotInD` at its sites (`batchMapAuxIdx N`). Their per-example
ties to the certified VJPs are `ViTStepTie`'s `vitCotTowerOutV_eq_vjp` and `ViTDropBlock`'s
`cotInD_eq_vjp`; the two lemmas below lift them over the batch. -/

/-- **Batched: a block's cotangent at its sites is the lifted block VJP's backward.** Example `n`
    runs the block at its own two mask entries, so the lift is the indexed one; `cotInD_eq_vjp` per
    example. At `none none` it is the drop-free block's (`cotInD_none`, `fwdOD_none`). -/
theorem vitBlockCotInB_eq_vjp {gf : GeluForm} (N : Nat) {Np1 heads d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d) mlpDim) (sA sM : Option (Vec N))
    (xin : Vec (N * (Np1 * (heads * d)))) :
    StableHLO.batchMapAuxIdx N (fun n => p.cotInD gf (Np1 := Np1) ε (exampleSite sA n) (exampleSite sM n)) xin
      = (batchMapIdxHasVJPAt (fun n => p.fwdOD gf (Np1 := Np1) ε (exampleSite sA n) (exampleSite sM n)) xin
          (fun n => (p.fwdODHasVJP gf ε hε (exampleSite sA n) (exampleSite sM n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε p _ _ _)).backward :=
  batchMapAuxIdx_eq_batchMapIdxHasVJPAt _ _ xin _ _ fun _ => by
    funext dy; exact cotInD_eq_vjp ε hε p _ _ _ dy

/-- **Batched: the head's `batchMapAux` cotangent is the lifted head VJP's backward.** -/
theorem vitCotTowerOutB_eq_vjp (N : Nat) {n D nC : Nat} (ε : ℝ) (hε : 0 < ε) (γF βF : Vec D)
    (Wcls : Mat D nC) (bcls : Vec nC) (b : Vec (N * ((n + 1) * D))) :
    StableHLO.batchMapAux N (vitCotTowerOutV n D nC ε γF Wcls) b
      = (batchMapHasVJPAt (classifierFlat n D nC Wcls bcls ∘
            fun v : Vec ((n + 1) * D) => Mat.flatten (fun r => layerNormVec D ε γF βF (Mat.unflatten v r)))
          b (fun _ => (vitHeadHasVJP n D nC ε hε γF βF Wcls bcls).toHasVJPAt _)
          (fun _ => ((classifierFlat_differentiable n D nC Wcls bcls).comp
            (layerNormVec_per_token_flat_differentiable (n + 1) D ε γF βF hε)) _)).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ b _ _ fun _ => by
    funext g; exact vitCotTowerOutV_eq_vjp n D nC ε hε γF βF Wcls bcls _ g

/-! ## The whole-net capstone — all 200 params through the REAL batched forward + composed cotangent

The fused file's thread, lifted: `ib1` is `batchMap N` of the patch embedding, `ib_{k+1}` is
`batchMapIdx N` of the multi-head block forward at block `k`'s sites, the final LN / CLS slice /
dense head are `batchMap N` of theirs, `g` is the smoothed loss cotangent at a general target, and
every cotangent is the lift of the per-example chain — `vitCotTowerOutV` at the top, then twelve
`cotInD` attention-residual fan-ins down to the embed-output cotangent. -/

/-- Block `k`'s attention-site masks, when the render carries stochastic depth: the first of the
    pair the `*drop*` artifacts feed block `k` (`%dp<2k>`, `ViTRenderB.vitSiteIdx k 0`). -/
def vitSdA {N L : Nat} (sd : Option (Fin L → Vec N × Vec N)) (k : Fin L) : Option (Vec N) :=
  sd.map fun f => (f k).1

/-- Block `k`'s MLP-site masks (`%dp<2k+1>`). -/
def vitSdM {N L : Nat} (sd : Option (Fin L → Vec N × Vec N)) (k : Fin L) : Option (Vec N) :=
  sd.map fun f => (f k).2

@[simp] theorem vitSdA_none {N L : Nat} (k : Fin L) :
    vitSdA (none : Option (Fin L → Vec N × Vec N)) k = none := rfl

@[simp] theorem vitSdM_none {N L : Nat} (k : Fin L) :
    vitSdM (none : Option (Fin L → Vec N × Vec N)) k = none := rfl

/-- The block's batched tie (`vitBlockTiedGB`), over its `BlockParamsV` record. -/
abbrev _root_.Proofs.BlockParamsV.TiedGB (gf : GeluForm) {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (bf16 : Bool)
    (sA sM : Option (Vec N)) (xin dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  vitBlockTiedGB gf N xN epsStr cotN ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo
    p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 bf16 sA sM xin dyOut

theorem _root_.Proofs.BlockParamsV.tied_gb {gf : GeluForm} {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (N : Nat) (xN epsStr cotN : String) (ε : ℝ) (bf16 : Bool)
    (sA sM : Option (Vec N)) (xin dyOut : Vec (N * (Np1 * (heads * d)))) :
    p.TiedGB gf N xN epsStr cotN ε bf16 sA sM xin dyOut :=
  vit_block_tiedGB N xN epsStr cotN ε _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ bf16 sA sM xin dyOut

/-- **The whole depth-12 multi-head ViT-Tiny train step, tied at the batched index, the
    gradient nodes and the smoothed loss — all 200 parameters.** The real forward
    `patchEmbed → 12 multi-head vector-LN blocks → final vector-LN → CLS-slice → dense head` as
    per-example lifts of the prefixes, the smoothed loss cotangent at a general target `t`,
    and the backward chain as the lift of the per-example one (the per-block multi-head
    fan-ins, `vitCotTowerOutV` at the top, the embed-output cotangent at the bottom): the twelve
    blocks' 192 params, the final-LN γ/β, the classifier and the patch-embed wConv/bConv/cls/pos
    all denote the certified batched `Σ_n` gradient at their chain cotangent, and each chain
    cotangent is the backward of its stage's lifted certified VJP (`vitCotTowerOutB_eq_vjp`,
    `vitBlockCotInB_eq_vjp`) — at the gradient nodes `vit_adam_train_step.mlir` and every
    `vitin_*` train step emit: `bf16` selects the per-token dense weight nodes' kind and
    `bf16 && bf16ConvW` the patch embed's, `false` the f32 artifacts', `true` the `*bf16` ones', read
    over ℝ at the identity rounding (`Bf16Erasure`); the right-hand side is the same certified
    gradient at every value. The CLS token's gradient sums over the batch inside `den`,
    which the per-example capstone could state only at `N = 1`.

    `sd` is stochastic depth: `none` the drop-free artifacts' chain, `some` the `*drop*`
    artifacts' — `vitin_emadp128x4wxclipdropeps0000001erfbf16`'s, the book's run — each block's out-projection
    and fc2 nodes at the dropped cotangent and both skips at the raw one, so the book's ViT job is
    reached at every gradient node with only its EMA tail outside.

    `N` and `nC` are binders and there is no smoothness hypothesis (GELU, no kink). The batch
    enters only through the per-example lifts, because no ViT op couples examples. The statement
    is at one replica and at ViT-Tiny's literal dims. -/
theorem vit_net_tiedGB {gf : GeluForm} (N : Nat) {nC : Nat}
    (xN aN epsStr cotN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    (w : ViTTieWeights nC) (bf16 bf16ConvW : Bool) (sd : Option (Fin 12 → Vec N × Vec N))
    (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC)) :
    let ib1    : Vec (N * (197 * 192)) := batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img
    let ib2    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) ib1
    let ib3    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) ib2
    let ib4    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) ib3
    let ib5    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) ib4
    let ib6    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) ib5
    let ib7    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) ib6
    let ib8    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) ib7
    let ib9    : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) ib8
    let ib10   : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) ib9
    let ib11   : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) ib10
    let ib12   : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) ib11
    let b12out : Vec (N * (197 * 192)) := batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) ib12
    -- final LN → CLS row → dense head, then the SMOOTHED loss cotangent at a general target `t`
    let flB     : Vec (N * (197 * 192)) :=
      batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b r))) b12out
    let hnB     : Vec (N * 192) := batchMap N (clsSliceFlat 196 192) flB
    let logitsB : Vec (N * nC)  := batchMap N (dense w.Wcls w.bcls) hnB
    let g       : Vec (N * nC)  :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    let dy12    : Vec (N * (197 * 192)) := batchMapAux N (vitCotTowerOutV 196 192 nC ε w.γF w.Wcls) b12out g
    let dy11   : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b12.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n)) ib12 dy12
    let dy10   : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b11.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n)) ib11 dy11
    let dy9    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b10.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n)) ib10 dy10
    let dy8    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b9.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n)) ib9 dy9
    let dy7    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b8.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n)) ib8 dy8
    let dy6    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b7.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n)) ib7 dy7
    let dy5    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b6.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n)) ib6 dy6
    let dy4    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b5.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n)) ib5 dy5
    let dy3    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b4.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n)) ib4 dy4
    let dy2    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b3.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n)) ib3 dy3
    let dy1    : Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b2.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n)) ib2 dy2
    let dyEmbed: Vec (N * (197 * 192)) := batchMapAuxIdx N (fun n => w.b1.cotInD gf (Np1 := 197) (heads := 3) (d := 64) ε (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n)) ib1 dy1
    w.b1.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 0) (vitSdM sd 0) ib1 dy1
  ∧ w.b2.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 1) (vitSdM sd 1) ib2 dy2
  ∧ w.b3.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 2) (vitSdM sd 2) ib3 dy3
  ∧ w.b4.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 3) (vitSdM sd 3) ib4 dy4
  ∧ w.b5.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 4) (vitSdM sd 4) ib5 dy5
  ∧ w.b6.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 5) (vitSdM sd 5) ib6 dy6
  ∧ w.b7.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 6) (vitSdM sd 6) ib7 dy7
  ∧ w.b8.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 7) (vitSdM sd 7) ib8 dy8
  ∧ w.b9.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 8) (vitSdM sd 8) ib9 dy9
  ∧ w.b10.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 9) (vitSdM sd 9) ib10 dy10
  ∧ w.b11.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 10) (vitSdM sd 10) ib11 dy11
  ∧ w.b12.TiedGB gf N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 11) (vitSdM sd 11) ib12 dy12
  ∧ vitFinalLNTiedGB N xN epsStr cotN ε w.γF w.βF w.Wcls b12out g
  ∧ vitHeadTiedGB N aN cotN hnB w.Wcls w.bcls g
  ∧ vitEmbedTiedGB N xN cotN w.Wc w.bc w.cls w.pos (bf16 && bf16ConvW) img dyEmbed := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out flB hnB logitsB g dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact w.b1.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 0) (vitSdM sd 0) ib1 dy1
  · exact w.b2.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 1) (vitSdM sd 1) ib2 dy2
  · exact w.b3.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 2) (vitSdM sd 2) ib3 dy3
  · exact w.b4.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 3) (vitSdM sd 3) ib4 dy4
  · exact w.b5.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 4) (vitSdM sd 4) ib5 dy5
  · exact w.b6.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 5) (vitSdM sd 5) ib6 dy6
  · exact w.b7.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 6) (vitSdM sd 6) ib7 dy7
  · exact w.b8.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 7) (vitSdM sd 7) ib8 dy8
  · exact w.b9.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 8) (vitSdM sd 8) ib9 dy9
  · exact w.b10.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 9) (vitSdM sd 9) ib10 dy10
  · exact w.b11.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 10) (vitSdM sd 10) ib11 dy11
  · exact w.b12.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε bf16 (vitSdA sd 11) (vitSdM sd 11) ib12 dy12
  · exact vit_finalLN_tiedGB N xN epsStr cotN ε w.γF w.βF w.Wcls b12out g
  · exact vit_head_tiedGB N aN cotN hnB w.Wcls w.bcls g
  · exact vit_embed_tiedGB N xN cotN w.Wc w.bc w.cls w.pos (bf16 && bf16ConvW) img dyEmbed

end Proofs.ViTTieGB
