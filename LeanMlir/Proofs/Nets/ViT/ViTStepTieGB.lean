import LeanMlir.Proofs.Nets.ViT.ViTFoldGB
import LeanMlir.Proofs.Nets.ViT.ViTStepTie
import LeanMlir.Proofs.Foundation.SmoothedLossCot

/-! # ViT-Tiny's T3 §1a TIE at the BATCHED index, the UN-FUSED gradient and the SMOOTHED loss

`ViTStepTie.lean` ties all 200 parameters of the SGD-inline `vit_train_step.mlir`: each fused
`θ − lr·g` op `den`s to the certified step at the cotangent the emitted backward chain delivers,
per example, at a hard label. This file is that statement re-pointed along the THREE axes
`ConvNeXtStepTieGB.lean` (§4b.6) moved for ConvNeXt, and it is that file's transformation applied
to ViT's per-example capstone — the 4b capstone that closes the set at five of five.

⭐ **Axis 1 — the OPTIMIZER FORM.** Every conjunct is at the RAW gradient node (`*GradB`), which is
what `vit_adam_train_step.mlir` and every `vitin_*` artifact emit since 4c leg 4; the fused op
appears only in the SGD-inline file. One statement covers AdamW, the `wx`/`clip` variants, EMA,
the 4× accumulation and the data-parallel twins, because they all consume this node.
`ViTFoldGB.lean` (§4c-ter) is the fold each conjunct delegates to.

⭐ **Axis 2 — the LOSS.** `g` is a binder, instantiated at `smoothedLossCotGraphDiv` — the six-op
chain `expe → softmaxDiv → subB → scaleB → addVB → shiftB → divConstB` this render emits at the
plain width `N·K`, at a GENERAL target arriving as `%onehot`. The fused file pins it to
`softmax − oneHot`.

⭐ **Axis 3 — the INDEX.** `N` is a binder. Every activation is `batchMap N` of the per-example
prefix the fused file threads (`patchEmbedFlat`, `vitBlockFwdOMHV`, the final LN, `clsSliceFlat`)
and every cotangent is `batchMapAux N` of the per-example chain (`vitCotB2outV`,
`vitBlockCotInAtMHV`, the `vitCot*` family). Honest for this net because no ViT op couples
examples — LayerNorm, attention, GELU and the denses are all per-example, and the `*B`
constructors' `den` arms say so. `nC` is a binder too (10 on Imagenette, 1000 on ImageNet).

⭐⭐ **The one conjunct the per-example capstone could not state is here.** The CLS token is one
shared `[192]` vector; its gradient is the sum of every example's CLS-row cotangent. The fused
file's `vit_cls_den` is at `denseBiasSgdB (N := 1)` — "sum one thing", correct there because
`pretty B` performed the batch lift outside the AST. `vitEmbedTiedGB`'s third conjunct is
`ViTPoCGB.clsGrad_denB` at the batched node, `denseBiasGradB (N := N)` on `batchMap N clsSliceFlat`
of the embed cotangent, with the batch sum inside `den`.

## What is NOT new

Every save, every chain cotangent and every Jacobian witness is `ViTStepTie.lean`'s, lifted; every
conjunct's proof is one `ViTPoCGB.*_den` lemma. The per-example saves and internal cotangents are
repackaged as functions of the block INPUT (`blkSaves`, `cAtt` … `cM1`) so that `batchMapAux` has
something to lift — the `let` chains of `vitBlockTiedAtMHV` and `vitBlockCotInAtMHV`, verbatim.
⚠ ViT has no `*BackBatchedGraph_faithful` family and needs none here: the lift is the honesty
argument, as it was for ConvNeXt.

⛔ **Conventions carried unchanged from the fused file:** the VECTOR LayerNorm (`γ β : Vec D`) at
all 25 sites, 3 heads × d_head 64, depth 12, D 192, MLP 768, 16×16 patches, GELU (no kink — no
smoothness hypothesis anywhere). Stated at ViT-Tiny's literal dims; S and B are other nets.
⛔ ONE REPLICA: in `vitin_adamdp128x4*` every gradient node feeds `allReduceMeanF`, the collective
as an AST node since 4d piece 2 (2026-09-07; `DataParallelNode.lean` composes the per-replica
statement with the replica mean), and the 4× accumulation is `momVNextF` at its other reading on
top. ⛔ Stated at the drop-free chain.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTTiePoCGB

open scoped BigOperators
open Proofs.ViTTiePoC (vitBlockFwdOMHV vitBlockCotInAtMHV ViTTieWeights)
open Proofs.ViTPoCGB (rowDenseBTiedB_holds rowDenseWTiedB_holds vecLNBetaTiedB_holds
  vecLNGammaTiedB_holds)

/-! ## Per-example saves and internal cotangents as functions of a block's INPUT

`vitBlockTiedMHV` takes the nine saved activations as arguments; `batchMapAux` lifts a function
of (one saved value, one input), so the batched block tie needs each save as a function of `xin`
and each internal cotangent as a function of `(xin, dyOut)` — exactly the `let` chains of
`ViTTiePoC.vitBlockTiedAtMHV` and `vitBlockCotInAtMHV`, packaged. -/

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
noncomputable def blkSaves {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim)
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
  let h    : Mat Np1 (heads * d) := fun r s => X r s + dense Wo bo (att r) s
  let ln2  : Mat Np1 (heads * d) := fun r kk => layerScale γ2 (fun s => layerNormForward (heads * d) ε 1 0 (h r) s) kk + β2 kk
  let m1   : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g    : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  ⟨Mat.flatten ln1, Mat.flatten Q, Mat.flatten K, Mat.flatten V, Mat.flatten att, Mat.flatten h,
   Mat.flatten ln2, Mat.flatten m1, Mat.flatten g⟩

/-- Per example, the attention-output cotangent (`vitCotAttV`), from the block input and output cotangent. -/
noncomputable def cAtt {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut

/-- Per example, the Q cotangent, per head (`vitCotDQmh`), from the block input and output cotangent. -/
noncomputable def cQ {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotDQmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut)

/-- Per example, the K cotangent, per head, from the block input and output cotangent. -/
noncomputable def cK {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotDKmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut)

/-- Per example, the V cotangent, per head, from the block input and output cotangent. -/
noncomputable def cV {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotDVmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut)

/-- Per example, the LN₁-output cotangent: the three-way Q/K/V fan-in (`vitCotLn1`), from the block input and output cotangent. -/
noncomputable def cLn1 {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotLn1 Wq Wk Wv
    (vitCotDQmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut))
    (vitCotDKmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut))
    (vitCotDVmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut))

/-- Per example, the MLP-residual fan-in at `h` (`vitCotHV`), from the block input and output cotangent. -/
noncomputable def cH {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotHV ε γ2 Wfc1 Wfc2 s.h s.m1 dyOut

/-- Per example, the LN₂-output cotangent (`vitCotLn2`), from the block input and output cotangent. -/
noncomputable def cLn2 {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotLn2 Wfc1 Wfc2 s.m1 dyOut

/-- Per example, the fc1-output cotangent through the GELU mask (`vitCotM1`), from the block input and output cotangent. -/
noncomputable def cM1 {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * mlpDim) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotM1 Wfc2 s.m1 dyOut

/-! ## The multi-head block — all 16 gradient nodes, batched -/

/-- **One multi-head vector-LN transformer block, tied at the batched gradient nodes.** Every one of
    the block's 16 params, fed `batchMapAux N` of the cotangent the real backward chain delivers at
    its site, `den`otes the certified `Σ_n` gradient — the MLP-residual and attention-residual
    fan-ins and the three-way LN₁ fan-in, per head, exactly as `vitBlockTiedMHV` has them per
    example. -/
def vitBlockTiedGB (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  -- forward saves — each `batchMap N` of the per-example save the emitted node lifts
  let ln1B : Vec (N * (Np1 * (heads * d))) :=
    batchMap N (fun x => (blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 x).ln1) xin
  let attB : Vec (N * (Np1 * (heads * d))) :=
    batchMap N (fun x => (blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 x).att) xin
  let hB   : Vec (N * (Np1 * (heads * d))) :=
    batchMap N (fun x => (blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 x).h) xin
  let ln2B : Vec (N * (Np1 * (heads * d))) :=
    batchMap N (fun x => (blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 x).ln2) xin
  let gB   : Vec (N * (Np1 * mlpDim)) :=
    batchMap N (fun x => (blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 x).g) xin
  -- backward chain cotangents — `batchMapAux N` of the per-example chain
  let cotLn1B : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cLn1 ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  let dQB     : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cQ ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  let dKB     : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cK ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  let dVB     : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cV ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  let cotHB   : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cH ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  let cotLn2B : Vec (N * (Np1 * (heads * d))) := batchMapAux N (cLn2 ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  let cotM1B  : Vec (N * (Np1 * mlpDim))      := batchMapAux N (cM1 ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2) xin dyOut
  -- LN₁ γ/β  (cot = cotLn1B, LN input = xin)
  ViTPoCGB.VecLNGammaTiedB N Np1 xN epsStr cotN ε β1 xin γ1 cotLn1B
  ∧ViTPoCGB.VecLNBetaTiedB N Np1 cotN ε γ1 xin β1 cotLn1B
  -- Q dense W/b  (cot = dQB, dense input = ln1B)
  ∧ViTPoCGB.RowDenseWTiedB N Np1 xN cotN bq ln1B Wq dQB
  ∧ViTPoCGB.RowDenseBTiedB N Np1 cotN Wq ln1B bq dQB
  -- K dense W/b  (cot = dKB)
  ∧ViTPoCGB.RowDenseWTiedB N Np1 xN cotN bk ln1B Wk dKB
  ∧ViTPoCGB.RowDenseBTiedB N Np1 cotN Wk ln1B bk dKB
  -- V dense W/b  (cot = dVB)
  ∧ViTPoCGB.RowDenseWTiedB N Np1 xN cotN bv ln1B Wv dVB
  ∧ViTPoCGB.RowDenseBTiedB N Np1 cotN Wv ln1B bv dVB
  -- out-proj dense W/b  (cot = cotHB, dense input = attB)
  ∧ViTPoCGB.RowDenseWTiedB N Np1 xN cotN bo attB Wo cotHB
  ∧ViTPoCGB.RowDenseBTiedB N Np1 cotN Wo attB bo cotHB
  -- LN₂ γ/β  (cot = cotLn2B, LN input = hB)
  ∧ViTPoCGB.VecLNGammaTiedB N Np1 xN epsStr cotN ε β2 hB γ2 cotLn2B
  ∧ViTPoCGB.VecLNBetaTiedB N Np1 cotN ε γ2 hB β2 cotLn2B
  -- fc1 dense W/b  (cot = cotM1B, dense input = ln2B)
  ∧ViTPoCGB.RowDenseWTiedB N Np1 xN cotN bfc1 ln2B Wfc1 cotM1B
  ∧ViTPoCGB.RowDenseBTiedB N Np1 cotN Wfc1 ln2B bfc1 cotM1B
  -- fc2 dense W/b  (cot = dyOut, dense input = gB)
  ∧ViTPoCGB.RowDenseWTiedB N Np1 xN cotN bfc2 gB Wfc2 dyOut
  ∧ViTPoCGB.RowDenseBTiedB N Np1 cotN Wfc2 gB bfc2 dyOut

theorem vit_block_tiedGB (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) :
    vitBlockTiedGB N xN epsStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 bfc2
      xin dyOut := by
  unfold vitBlockTiedGB
  exact ⟨vecLNGammaTiedB_holds, vecLNBetaTiedB_holds, rowDenseWTiedB_holds, rowDenseBTiedB_holds,
    rowDenseWTiedB_holds, rowDenseBTiedB_holds, rowDenseWTiedB_holds, rowDenseBTiedB_holds,
    rowDenseWTiedB_holds, rowDenseBTiedB_holds, vecLNGammaTiedB_holds, vecLNBetaTiedB_holds,
    rowDenseWTiedB_holds, rowDenseBTiedB_holds, rowDenseWTiedB_holds, rowDenseBTiedB_holds⟩

/-! ## Final LN, classifier and patch embedding — batched -/

/-- **Final vector-LN γF/βF, tied at the batched classifier-back cotangent** `vitCotFl` per
    example (the `clsPad` of `Wclsᵀ g_n`, exactly what the render's `dotOut → clsPad` computes). -/
def vitFinalLNTiedGB (N : Nat) {nC : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (b12out : Vec (N * (197 * 192))) (g : Vec (N * nC)) :
    Prop :=
  let cotFlB : Vec (N * (197 * 192)) := batchMap N (vitCotFl 196 192 nC Wcls) g
  ViTPoCGB.VecLNGammaTiedB N 197 xN epsStr cotN ε βF b12out γF cotFlB
  ∧ ViTPoCGB.VecLNBetaTiedB N 197 cotN ε γF b12out βF cotFlB

theorem vit_finalLN_tiedGB (N : Nat) {nC : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (b12out : Vec (N * (197 * 192))) (g : Vec (N * nC)) :
    vitFinalLNTiedGB N xN epsStr cotN ε γF βF Wcls b12out g := by
  unfold vitFinalLNTiedGB
  intro cotFlB
  refine ⟨?_, ?_⟩
  · exact vecLNGammaTiedB_holds
  · intro i
    exact ViTPoCGB.rowDenseBiasGradB_den_lnbeta cotN ε γF
      (fun n => Mat.unflatten (batchSlice N (197 * 192) b12out n)) βF cotFlB i

/-- **Classifier Wcls/bcls, tied at the loss cotangent `g`** — the weight at the batched CLS row,
    the bias PER EXAMPLE (`biasGradB` is the identity on its operand; the batch reduce is emitted
    text — `ViTPoCGB.headBGradB_den`). -/
def vitHeadTiedGB (N : Nat) {nC : Nat} (aN cotN : String)
    (hn : Vec (N * 192)) (Wcls : Mat 192 nC) (bcls : Vec nC) (g : Vec (N * nC)) : Prop :=
  (∀ (i : Fin 192) (j : Fin nC),
      den (SHlo.weightGradB (N := N) (m := 192) (n := nC) aN hn (.operand cotN g))
          (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ k : Fin nC,
            pdiv (fun v : Vec (192 * nC) => dense (Mat.unflatten v) bcls (batchSlice N 192 hn n))
                 (Mat.flatten Wcls) (finProdFinEquiv (i, j)) k * batchSlice N nC g n k)
  ∧ (∀ (n : Fin N) (i : Fin nC),
      batchSlice N nC (den (SHlo.biasGradB (N := N) (n := nC) (.operand cotN g))) n i
        = ∑ j : Fin nC,
            pdiv (fun b' : Vec nC => dense Wcls b' (batchSlice N 192 hn n)) bcls i j
              * batchSlice N nC g n j)

theorem vit_head_tiedGB (N : Nat) {nC : Nat} (aN cotN : String)
    (hn : Vec (N * 192)) (Wcls : Mat 192 nC) (bcls : Vec nC) (g : Vec (N * nC)) :
    vitHeadTiedGB N aN cotN hn Wcls bcls g := by
  unfold vitHeadTiedGB
  refine ⟨?_, ?_⟩
  · intro i j; exact ViTPoCGB.headWGradB_den aN cotN hn Wcls bcls g i j
  · intro n i; exact ViTPoCGB.headBGradB_den cotN Wcls (batchSlice N 192 hn n) bcls g n i

/-- **Patch embed wConv/bConv/cls/pos, tied at the batched embed-output cotangent.** ⭐ The third
    conjunct is the CLS token's gradient with the batch sum INSIDE `den` — the statement the
    per-example capstone made only at `N = 1`. -/
def vitEmbedTiedGB (N : Nat) (xN cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (N * (3 * 224 * 224))) (dyEmbed : Vec (N * (197 * 192))) : Prop :=
  (∀ (dd : Fin 192) (c : Fin 3) (kh kw : Fin 16),
      den (SHlo.patchEmbedWeightGradB (N := N) (ic := 3) (H := 224) (W := 224) (P := 16)
            (tk := 196) (D := 192) xN img (.operand cotN dyEmbed))
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
    (img : Vec (N * (3 * 224 * 224))) (dyEmbed : Vec (N * (197 * 192))) :
    vitEmbedTiedGB N xN cotN Wc bc cls pos img dyEmbed := by
  unfold vitEmbedTiedGB
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro dd c kh kw
    exact ViTPoCGB.patchEmbedWeightGradB_den xN cotN bc cls pos img Wc dyEmbed dd c kh kw
  · intro i; exact ViTPoCGB.patchEmbedBiasGradB_den cotN Wc bc cls pos img dyEmbed i
  · intro i; exact ViTPoCGB.clsGrad_denB cotN Wc bc cls pos img dyEmbed i
  · intro i; exact ViTPoCGB.posEmbedGradB_den cotN Wc bc cls pos img dyEmbed i

/-! ## The whole-net capstone — all 200 params through the REAL batched forward + composed cotangent

The fused file's thread, lifted: `ib1` is `batchMap N` of the patch embedding, `ib_{k+1}` is
`batchMap N` of the multi-head block forward, the final LN / CLS slice / dense head are `batchMap N`
of theirs, `g` is the smoothed loss cotangent at a general target, and every cotangent is
`batchMapAux N` of the per-example chain — `vitCotB2outV` at the top, then twelve
`vitBlockCotInAtMHV` attention-residual fan-ins down to the embed-output cotangent. -/

/-- The block's batched tie (`vitBlockTiedGB`), over its `BlockParamsV` record. -/
abbrev _root_.Proofs.BlockParamsV.TiedGB {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (N : Nat) (xN epsStr cotN : String) (ε : ℝ)
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  vitBlockTiedGB N xN epsStr cotN ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo
    p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 xin dyOut

theorem _root_.Proofs.BlockParamsV.tied_gb {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (N : Nat) (xN epsStr cotN : String) (ε : ℝ)
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) : p.TiedGB N xN epsStr cotN ε xin dyOut :=
  vit_block_tiedGB N xN epsStr cotN ε _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ xin dyOut

/-- ⭐⭐ **The whole depth-12 multi-head ViT-Tiny train step, tied at the BATCHED index, the
    GRADIENT nodes and the SMOOTHED loss — all 200 parameters.** The real forward
    `patchEmbed → 12 multi-head vector-LN blocks → final vector-LN → CLS-slice → dense head` as
    `batchMap N` of the per-example prefixes, the smoothed loss cotangent at a general target `t`,
    and the backward chain as `batchMapAux N` of the per-example one (the per-block multi-head
    fan-ins, `vitCotB2outV` at the top, the embed-output cotangent at the bottom): the twelve
    blocks' 192 params, the final-LN γ/β, the classifier and the patch-embed wConv/bConv/cls/pos
    all denote the certified batched `Σ_n` gradient — at the nodes `vit_adam_train_step.mlir` and
    every `vitin_*` artifact emit. ⭐ The CLS token's gradient sums over the batch INSIDE `den`,
    which the per-example capstone could state only at `N = 1`.

    ⭐ **`N` and `nC` are binders and there is no smoothness hypothesis** (GELU, no kink). The batch
    enters only through `batchMap`/`batchMapAux`, honest because no ViT op couples examples.
    ⛔ ONE REPLICA (4d); the 4× accumulation is `momVNextF`'s other reading on top; stated at the
    drop-free chain and at ViT-Tiny's literal dims. -/
theorem vit_net_tiedGB (N : Nat) {nC : Nat}
    (xN aN epsStr cotN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
    (w : ViTTieWeights nC)
    (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC)) :
    let ib1    : Vec (N * (197 * 192)) := batchMap N (patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos) img
    let ib2    : Vec (N * (197 * 192)) := batchMap N (w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib1
    let ib3    : Vec (N * (197 * 192)) := batchMap N (w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib2
    let ib4    : Vec (N * (197 * 192)) := batchMap N (w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib3
    let ib5    : Vec (N * (197 * 192)) := batchMap N (w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib4
    let ib6    : Vec (N * (197 * 192)) := batchMap N (w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib5
    let ib7    : Vec (N * (197 * 192)) := batchMap N (w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib6
    let ib8    : Vec (N * (197 * 192)) := batchMap N (w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib7
    let ib9    : Vec (N * (197 * 192)) := batchMap N (w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib8
    let ib10   : Vec (N * (197 * 192)) := batchMap N (w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib9
    let ib11   : Vec (N * (197 * 192)) := batchMap N (w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib10
    let ib12   : Vec (N * (197 * 192)) := batchMap N (w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib11
    let b12out : Vec (N * (197 * 192)) := batchMap N (w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε) ib12
    -- final LN → CLS row → dense head, then the SMOOTHED loss cotangent at a general target `t`
    let flB     : Vec (N * (197 * 192)) :=
      batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b r))) b12out
    let hnB     : Vec (N * 192) := batchMap N (clsSliceFlat 196 192) flB
    let logitsB : Vec (N * nC)  := batchMap N (dense w.Wcls w.bcls) hnB
    let g       : Vec (N * nC)  :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    let dy12    : Vec (N * (197 * 192)) := batchMapAux N (vitCotB2outV 196 192 nC ε w.γF w.Wcls) b12out g
    let dy11   : Vec (N * (197 * 192)) := batchMapAux N (w.b12.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib12 dy12
    let dy10   : Vec (N * (197 * 192)) := batchMapAux N (w.b11.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib11 dy11
    let dy9    : Vec (N * (197 * 192)) := batchMapAux N (w.b10.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib10 dy10
    let dy8    : Vec (N * (197 * 192)) := batchMapAux N (w.b9.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib9 dy9
    let dy7    : Vec (N * (197 * 192)) := batchMapAux N (w.b8.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib8 dy8
    let dy6    : Vec (N * (197 * 192)) := batchMapAux N (w.b7.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib7 dy7
    let dy5    : Vec (N * (197 * 192)) := batchMapAux N (w.b6.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib6 dy6
    let dy4    : Vec (N * (197 * 192)) := batchMapAux N (w.b5.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib5 dy5
    let dy3    : Vec (N * (197 * 192)) := batchMapAux N (w.b4.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib4 dy4
    let dy2    : Vec (N * (197 * 192)) := batchMapAux N (w.b3.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib3 dy3
    let dy1    : Vec (N * (197 * 192)) := batchMapAux N (w.b2.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib2 dy2
    let dyEmbed: Vec (N * (197 * 192)) := batchMapAux N (w.b1.cotIn (Np1 := 197) (heads := 3) (d := 64) ε) ib1 dy1
    w.b1.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib1 dy1
  ∧ w.b2.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib2 dy2
  ∧ w.b3.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib3 dy3
  ∧ w.b4.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib4 dy4
  ∧ w.b5.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib5 dy5
  ∧ w.b6.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib6 dy6
  ∧ w.b7.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib7 dy7
  ∧ w.b8.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib8 dy8
  ∧ w.b9.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib9 dy9
  ∧ w.b10.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib10 dy10
  ∧ w.b11.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib11 dy11
  ∧ w.b12.TiedGB N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib12 dy12
  ∧ vitFinalLNTiedGB N xN epsStr cotN ε w.γF w.βF w.Wcls b12out g
  ∧ vitHeadTiedGB N aN cotN hnB w.Wcls w.bcls g
  ∧ vitEmbedTiedGB N xN cotN w.Wc w.bc w.cls w.pos img dyEmbed := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out flB hnB logitsB g dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact w.b1.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib1 dy1
  · exact w.b2.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib2 dy2
  · exact w.b3.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib3 dy3
  · exact w.b4.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib4 dy4
  · exact w.b5.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib5 dy5
  · exact w.b6.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib6 dy6
  · exact w.b7.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib7 dy7
  · exact w.b8.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib8 dy8
  · exact w.b9.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib9 dy9
  · exact w.b10.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib10 dy10
  · exact w.b11.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib11 dy11
  · exact w.b12.tied_gb N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε ib12 dy12
  · exact vit_finalLN_tiedGB N xN epsStr cotN ε w.γF w.βF w.Wcls b12out g
  · exact vit_head_tiedGB N aN cotN hnB w.Wcls w.bcls g
  · exact vit_embed_tiedGB N xN cotN w.Wc w.bc w.cls w.pos img dyEmbed

end Proofs.ViTTiePoCGB
