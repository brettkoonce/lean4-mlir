import LeanMlir.Proofs.Nets.ViT.ViTFold
import LeanMlir.Proofs.Nets.ViT.ViTMultiHeadChain
import LeanMlir.Proofs.Nets.ViT.ViTDepthK

/-! # ViT-Tiny step tie — the SGD-inline train step, all 200 parameters at the real backward chain

**What this file is.** The tie of `verified_mlir/vit_train_step.mlir` — the SGD-inline step
`ViTRender.lean` writes — at the fused `θ − lr·g` ops: `vit_net_tied_certified` (the last theorem)
threads all 200 ViT-Tiny parameters through the committed multi-head (3 heads, d_head 64),
depth-12, vector-LayerNorm forward and the loss-driven backward cotangent chain. The statement is
per example, at one image and a hard label; the artifact runs a batch of 32, and that batch and
its mean lie outside this statement. Its batched peer at the un-fused gradient node, the smoothed
loss and a batch binder — the chain of the drop-free f32 `vitin_*` artifacts — is
`ViTTiePoCGB.vit_net_tiedGB`, built from this file's block ties by `batchMap` / `batchMapAux`.

The file has two layers:
* `vit_block_tiedMHV` / `vit_block_tiedAtMHV` — one multi-head (3 heads, d_head 64) vector-LN
  block, generic in the cotangent: the two residual fan-ins (`vitCotHV`, `vitCotXinV`), the
  three-way fan-in at LN₁ (`vitCotLn1`), and the per-head SDPA backward (`ViTMultiHeadChain`'s
  `vitCotD{Q,K,V}mh`).
* `vit_net_tied_certified` — the 200-parameter capstone: the 192 block parameters threaded
  through all 12 blocks, the final LN, the classifier and the patch embed (`vit_cls_den` covers the
  CLS token at `N = 1`).

Every conjunct delegates to a `ViTPoC.*_den` fold lemma at the chain cotangent — zero new ops,
zero new bridges. The vector-LN granularity that ships (`[192]` γ/β) is what is modelled. -/

namespace Proofs.ViTTiePoC

open scoped BigOperators
open Proofs Proofs.StableHLO
open Proofs.ViTPoC (rowDenseBSgdTied_holds rowDenseWSgdTied_holds vecLNBetaSgdTied_holds
  vecLNGammaSgdTied_holds)

/-! ## Multi-head promotion (3 heads, d_head=64) — the committed-render block tie

The committed `vitTrainStepRenderV` is multi-head: the SDPA-internal backward `dAtt → dQ/dK/dV`
runs per head (`vitCotD{Q,K,V}mh`, `ViTMultiHeadChain`), so the Q/K/V dense cotangents are the
multi-head `…mh` ones rather than the single-head `vitCotD{Q,K,V}`; everything else (the out-proj
`Wo`, LN₂, the MLP) is head-agnostic. `vitBlockTiedMHV` states the block's 16 parameter ties with
those cotangents (no separate `ss`/`p` saves — the per-head scores/weights are recomputed inside the
`…mh` cots from the saved Q/K); every conjunct delegates to a head-agnostic fold generic
`ViTPoC.*_den`. -/

def vitBlockTiedMHV {Np1 heads d mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin ln1 q k v att h ln2 : Vec (Np1 * (heads * d))) (g : Vec (Np1 * mlpDim))
    (m1 : Vec (Np1 * mlpDim))
    (dyOut : Vec (Np1 * (heads * d))) (lr : ℝ) : Prop :=
    let dAtt   : Vec (Np1 * (heads * d))      := vitCotAttV ε γ2 Wo Wfc1 Wfc2 h m1 dyOut
    let dQ     : Vec (Np1 * (heads * d))      := vitCotDQmh Np1 heads d q k v dAtt
    let dK     : Vec (Np1 * (heads * d))      := vitCotDKmh Np1 heads d q k v dAtt
    let dV     : Vec (Np1 * (heads * d))      := vitCotDVmh Np1 heads d q k v dAtt
    let cotLn1 : Vec (Np1 * (heads * d))      := vitCotLn1 Wq Wk Wv dQ dK dV
    let cotH   : Vec (Np1 * (heads * d))      := vitCotHV ε γ2 Wfc1 Wfc2 h m1 dyOut
    let cotLn2 : Vec (Np1 * (heads * d))      := vitCotLn2 Wfc1 Wfc2 m1 dyOut
    let cotM1  : Vec (Np1 * mlpDim) := vitCotM1 Wfc2 m1 dyOut
    -- LN₁ γ/β  (cot = cotLn1, LN input = xin)
    ViTPoC.VecLNGammaSgdTied Np1 gN xN epsStr lrStr cotN ε β1 xin γ1 cotLn1 lr
  ∧ ViTPoC.VecLNBetaSgdTied Np1 bN lrStr cotN ε γ1 xin β1 cotLn1 lr
    -- Q dense W/b  (cot = dQ, dense input = ln1)
  ∧ ViTPoC.RowDenseWSgdTied Np1 xN wN lrStr cotN bq ln1 Wq dQ lr
  ∧ ViTPoC.RowDenseBSgdTied Np1 bN lrStr cotN Wq ln1 bq dQ lr
    -- K dense W/b  (cot = dK)
  ∧ ViTPoC.RowDenseWSgdTied Np1 xN wN lrStr cotN bk ln1 Wk dK lr
  ∧ ViTPoC.RowDenseBSgdTied Np1 bN lrStr cotN Wk ln1 bk dK lr
    -- V dense W/b  (cot = dV)
  ∧ ViTPoC.RowDenseWSgdTied Np1 xN wN lrStr cotN bv ln1 Wv dV lr
  ∧ ViTPoC.RowDenseBSgdTied Np1 bN lrStr cotN Wv ln1 bv dV lr
    -- out-proj dense W/b  (cot = cotH, dense input = att)
  ∧ ViTPoC.RowDenseWSgdTied Np1 xN wN lrStr cotN bo att Wo cotH lr
  ∧ ViTPoC.RowDenseBSgdTied Np1 bN lrStr cotN Wo att bo cotH lr
    -- LN₂ γ/β  (cot = cotLn2, LN input = h)
  ∧ ViTPoC.VecLNGammaSgdTied Np1 gN xN epsStr lrStr cotN ε β2 h γ2 cotLn2 lr
  ∧ ViTPoC.VecLNBetaSgdTied Np1 bN lrStr cotN ε γ2 h β2 cotLn2 lr
    -- fc1 dense W/b  (cot = cotM1, dense input = ln2)
  ∧ ViTPoC.RowDenseWSgdTied Np1 xN wN lrStr cotN bfc1 ln2 Wfc1 cotM1 lr
  ∧ ViTPoC.RowDenseBSgdTied Np1 bN lrStr cotN Wfc1 ln2 bfc1 cotM1 lr
    -- fc2 dense W/b  (cot = dyOut, dense input = g)
  ∧ ViTPoC.RowDenseWSgdTied Np1 xN wN lrStr cotN bfc2 g Wfc2 dyOut lr
  ∧ ViTPoC.RowDenseBSgdTied Np1 bN lrStr cotN Wfc2 g bfc2 dyOut lr

theorem vit_block_tiedMHV {Np1 heads d mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin ln1 q k v att h ln2 : Vec (Np1 * (heads * d))) (g : Vec (Np1 * mlpDim))
    (m1 : Vec (Np1 * mlpDim))
    (dyOut : Vec (Np1 * (heads * d))) (lr : ℝ) :
    vitBlockTiedMHV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
      Wfc1 bfc1 Wfc2 bfc2 xin ln1 q k v att h ln2 g m1 dyOut lr := by
  unfold vitBlockTiedMHV
  exact ⟨vecLNGammaSgdTied_holds, vecLNBetaSgdTied_holds, rowDenseWSgdTied_holds,
    rowDenseBSgdTied_holds, rowDenseWSgdTied_holds, rowDenseBSgdTied_holds, rowDenseWSgdTied_holds,
    rowDenseBSgdTied_holds, rowDenseWSgdTied_holds, rowDenseBSgdTied_holds, vecLNGammaSgdTied_holds,
    vecLNBetaSgdTied_holds, rowDenseWSgdTied_holds, rowDenseBSgdTied_holds, rowDenseWSgdTied_holds,
    rowDenseBSgdTied_holds⟩


/-! ## Multi-head forward + cot-in + input-only block wrappers (the thread template) -/

/-- Multi-head forward block step (the committed render's block forward = `vitBlockSpelledMHV`,
    which IS `transformerBlockV` at general `heads` by `vitBlockSpelledMHV_eq`). -/
noncomputable def vitBlockFwdOMHV {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  Mat.flatten (vitBlockSpelledMHV Np1 heads d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
    γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten xin))

/-- Multi-head attention-residual fan-in: the block-input cotangent the chain hands upstream
    (`vitCotXinV` at the multi-head Q/K/V dense cots — `vitCotLn1 Wq Wk Wv dQmh dKmh dVmh` IS the
    multi-head LN₁ fan-in `vitCotLn1MH`). Recomputes the saves from `xin` (the `vitBlockSpelledMHV`
    let-chain, multi-head `att`). -/
noncomputable def vitBlockCotInAtMHV {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
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
  let dAtt := vitCotAttV ε γ2 Wo Wfc1 Wfc2 (Mat.flatten h) (Mat.flatten m1) dyOut
  let dQ   := vitCotDQmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let dK   := vitCotDKmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let dV   := vitCotDVmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let cotH := vitCotHV ε γ2 Wfc1 Wfc2 (Mat.flatten h) (Mat.flatten m1) dyOut
  vitCotXinV ε γ1 Wq Wk Wv xin dQ dK dV cotH

/-- Multi-head input-only block tie — recompute the 9 saves from `xin`, then `vit_block_tiedMHV`. -/
def vitBlockTiedAtMHV {Np1 heads d mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) (lr : ℝ) : Prop :=
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
  vitBlockTiedMHV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
    Wfc1 bfc1 Wfc2 bfc2 xin (Mat.flatten ln1) (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V)
    (Mat.flatten att) (Mat.flatten h) (Mat.flatten ln2) (Mat.flatten g) (Mat.flatten m1) dyOut lr

/-- The multi-head input-only block tie holds — unfold the saves, delegate to `vit_block_tiedMHV`. -/
theorem vit_block_tiedAtMHV {Np1 heads d mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) (lr : ℝ) :
    vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
      Wfc1 bfc1 Wfc2 bfc2 xin dyOut lr := by
  unfold vitBlockTiedAtMHV
  exact vit_block_tiedMHV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
    Wfc1 bfc1 Wfc2 bfc2 xin _ _ _ _ _ _ _ _ _ dyOut lr

/-! ## The non-block param bundle + the all-200-params capstone (committed ViT-Tiny config)

`vitFinalLNTied`/`vitHeadTied`/`vitEmbedTied` bundle the final vector-LN γ/β, the classifier Wcls/bcls,
and the patch-embed wConv/bConv/cls/pos as `den = certified` at their chain cotangents — each a direct
delegation to the fold generics (`ViTPoC.*_den`), with the cls op (`denseBiasSgdB` N=1) folded by
`vit_cls_den` (its row-0 batch slice IS `clsTokenGrad`, closed by `vit_render_cls_certified`). Then
`vit_net_tied_certified` threads the REAL forward + loss-driven backward and bundles all 200 params. -/

-- cls-param op den at the committed ViT-Tiny dims
theorem vit_cls_den (clsN lrStr cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (lr : ℝ) (i : Fin 192) :
    den (SHlo.denseBiasSgdB (N := 1) (c := 192) clsN lrStr cls lr
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
      = cls i - lr * ∑ j : Fin (197 * 192),
          pdiv (fun cl : Vec 192 =>
                  patchEmbedFlat 3 224 224 16 196 192 Wc bc cl pos img) cls i j * dyEmbed j := by
  have hstep : den (SHlo.denseBiasSgdB (N := 1) (c := 192) clsN lrStr cls lr
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
      = cls i - lr * clsTokenGrad dyEmbed i := by
    simp only [denStepApp, batchSlice, clsSliceFlat, clsTokenGrad]; rw [Fin.sum_univ_one]; rfl
  rw [hstep, vit_render_cls_certified Wc bc cls pos img dyEmbed lr i]

/-- Final vector-LN γF/βF tied at the classifier-back cot `vitCotFl`. -/
def vitFinalLNTied (gN xN bN epsStr lrStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 10) (b12out : Vec (197 * 192)) (g : Vec 10) (lr : ℝ) : Prop :=
  (∀ k : Fin 192,
      den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε b12out γF lr (.operand cotN (vitCotFl 196 192 10 Wcls g))) k
        = γF k - lr * ∑ o : Fin (197 * 192),
            pdiv (fun gv : Vec 192 => Mat.flatten (fun r => layerNormVec 192 ε gv βF (Mat.unflatten b12out r))) γF k o * vitCotFl 196 192 10 Wcls g o)
  ∧ (∀ i : Fin 192,
      den (SHlo.rowDenseBiasSgd bN lrStr βF lr (.operand cotN (vitCotFl 196 192 10 Wcls g))) i
        = βF i - lr * ∑ o : Fin (197 * 192),
            pdiv (fun bv : Vec 192 => Mat.flatten (fun r => layerNormVec 192 ε γF bv (Mat.unflatten b12out r))) βF i o * vitCotFl 196 192 10 Wcls g o)

theorem vit_finalLN_tied (gN xN bN epsStr lrStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 10) (b12out : Vec (197 * 192)) (g : Vec 10) (lr : ℝ) :
    vitFinalLNTied gN xN bN epsStr lrStr cotN ε γF βF Wcls b12out g lr := by
  exact ⟨vecLNGammaSgdTied_holds, vecLNBetaSgdTied_holds⟩

/-- Classifier Wcls/bcls tied at the loss cotangent `g`. -/
def vitHeadTied (aN wN bN lrStr cotN : String)
    (hn : Vec 192) (Wcls : Mat 192 10) (bcls : Vec 10) (g : Vec 10) (lr : ℝ) : Prop :=
  (∀ (i : Fin 192) (j : Fin 10),
      den (SHlo.weightSgd aN wN lrStr hn Wcls lr (.operand cotN g)) (finProdFinEquiv (i, j))
        = Wcls i j - lr * ∑ k : Fin 10,
            pdiv (fun v : Vec (192 * 10) => dense (Mat.unflatten v) bcls hn) (Mat.flatten Wcls) (finProdFinEquiv (i, j)) k * g k)
  ∧ (∀ i : Fin 10,
      den (SHlo.biasSgd bN lrStr bcls lr (.operand cotN g)) i
        = bcls i - lr * ∑ j : Fin 10, pdiv (fun b' : Vec 10 => dense Wcls b' hn) bcls i j * g j)

theorem vit_head_tied (aN wN bN lrStr cotN : String)
    (hn : Vec 192) (Wcls : Mat 192 10) (bcls : Vec 10) (g : Vec 10) (lr : ℝ) :
    vitHeadTied aN wN bN lrStr cotN hn Wcls bcls g lr := by
  refine ⟨?_, ?_⟩
  · intro i j; exact Cifar8PoC.denseW_den aN wN lrStr cotN hn Wcls bcls g lr i j
  · intro i;   exact Cifar8PoC.denseB_den bN lrStr cotN Wcls hn bcls g lr i

/-- Patch embed wConv/bConv/cls/pos tied at the embed-output cot `dyEmbed`. -/
def vitEmbedTied (wN xN bN clsN pN lrStr cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (lr : ℝ) : Prop :=
  (∀ (d : Fin 192) (c : Fin 3) (kh kw : Fin 16),
      den (SHlo.patchEmbedWeightSgd wN xN lrStr img Wc lr (.operand cotN dyEmbed))
          (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
        = Wc d c kh kw - lr * ∑ o : Fin (197 * 192),
            pdiv (fun v : Vec (192 * 3 * 16 * 16) =>
                    patchEmbedFlat 3 224 224 16 196 192 (Kernel4.unflatten v) bc cls pos img)
              (Kernel4.flatten Wc)
              (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o * dyEmbed o)
  ∧ (∀ i : Fin 192,
      den (SHlo.patchEmbedBiasSgd bN lrStr bc lr (.operand cotN dyEmbed)) i
        = bc i - lr * ∑ o : Fin (197 * 192),
            pdiv (fun b' : Vec 192 => patchEmbedFlat 3 224 224 16 196 192 Wc b' cls pos img) bc i o * dyEmbed o)
  ∧ (∀ i : Fin 192,
      den (SHlo.denseBiasSgdB (N := 1) (c := 192) clsN lrStr cls lr (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
        = cls i - lr * ∑ j : Fin (197 * 192),
            pdiv (fun cl : Vec 192 => patchEmbedFlat 3 224 224 16 196 192 Wc bc cl pos img) cls i j * dyEmbed j)
  ∧ (∀ i : Fin (197 * 192),
      den (SHlo.posEmbedSgd pN lrStr pos lr (.operand cotN dyEmbed)) i
        = Mat.flatten pos i - lr * ∑ j : Fin (197 * 192),
            pdiv (fun p : Vec (197 * 192) => patchEmbedFlat 3 224 224 16 196 192 Wc bc cls (Mat.unflatten p) img) (Mat.flatten pos) i j * dyEmbed j)

theorem vit_embed_tied (wN xN bN clsN pN lrStr cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (lr : ℝ) :
    vitEmbedTied wN xN bN clsN pN lrStr cotN Wc bc cls pos img dyEmbed lr := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro d c kh kw; exact ViTPoC.patchEmbedWeightSgd_den wN xN lrStr cotN bc cls pos img Wc dyEmbed lr d c kh kw
  · intro i; exact ViTPoC.patchEmbedBiasSgd_den bN lrStr cotN Wc bc cls pos img dyEmbed lr i
  · intro i; exact vit_cls_den clsN lrStr cotN Wc bc cls pos img dyEmbed lr i
  · intro i; exact ViTPoC.posEmbedSgd_den pN lrStr cotN Wc bc cls pos img dyEmbed lr i


/-! ## The ties' weight records

Twelve blocks as `BlockParamsV` (ViTDepthK's bundle), plus the patch embed, final LN and head.
The ties share ONE `ε` across every LN. The dot-notation wrappers are `abbrev`s over the
constructors above: a statement over a record unfolds to the unpacked one. -/

/-- ViT-Tiny as the ties bind it. -/
structure ViTTieWeights (nC : Nat) where
  Wc : Kernel4 192 3 16 16
  bc : Vec 192
  cls : Vec 192
  pos : Mat 197 192
  b1 : BlockParamsV 192 768
  b2 : BlockParamsV 192 768
  b3 : BlockParamsV 192 768
  b4 : BlockParamsV 192 768
  b5 : BlockParamsV 192 768
  b6 : BlockParamsV 192 768
  b7 : BlockParamsV 192 768
  b8 : BlockParamsV 192 768
  b9 : BlockParamsV 192 768
  b10 : BlockParamsV 192 768
  b11 : BlockParamsV 192 768
  b12 : BlockParamsV 192 768
  γF : Vec 192
  βF : Vec 192
  Wcls : Mat 192 nC
  bcls : Vec nC

/-- The block's forward (`vitBlockFwdOMHV`). -/
noncomputable abbrev _root_.Proofs.BlockParamsV.fwdO {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (ε : ℝ) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  vitBlockFwdOMHV ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1
    p.Wfc2 p.bfc2

/-- The block's input cotangent (`vitBlockCotInAtMHV`). -/
noncomputable abbrev _root_.Proofs.BlockParamsV.cotIn {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (ε : ℝ) :
    Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) → Vec (Np1 * (heads * d)) :=
  vitBlockCotInAtMHV ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1
    p.Wfc2

/-- The block's per-example tie (`vitBlockTiedAtMHV`). -/
abbrev _root_.Proofs.BlockParamsV.TiedAt {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (xin dyOut : Vec (Np1 * (heads * d))) (lr : ℝ) : Prop :=
  vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 xin dyOut lr

theorem _root_.Proofs.BlockParamsV.tied_at {Np1 heads d mlpDim : Nat}
    (p : BlockParamsV (heads * d) mlpDim) (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (xin dyOut : Vec (Np1 * (heads * d))) (lr : ℝ) :
    p.TiedAt xN wN bN gN epsStr lrStr cotN ε xin dyOut lr :=
  vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ xin dyOut lr

/-- **The whole depth-12 multi-head ViT-Tiny train step, tied — all 200 params** (the vit peer of
    convnext's `cnx_net_tied_certified`, at the committed config: 3 heads, d_head=64, D=192, N=196,
    mlpDim=768, 10 classes, 16×16 patches). The real forward `patchEmbed → 12 multi-head vector-LN
    blocks → final vector-LN → CLS-slice → dense head` and the loss-driven backward cotangent chain
    (the per-block multi-head fan-ins, the final-LN-back `vitCotB2outV`, the classifier-back `vitCotFl`,
    the embed-output cot = block-1's `vitBlockCotInAtMHV` output) are threaded, and EVERY param op
    `den`otes the certified loss-descent step: the 12 blocks' 192 params (`vitBlockTiedAtMHV`), the
    final-LN γ/β, the classifier Wcls/bcls, and the patch-embed wConv/bConv/cls/pos — 200/200.
    The statement is per example, at one image `img` and a hard label `label`; the artifact's
    batch of 32 and its mean lie outside it (the batched form is `ViTTiePoCGB.vit_net_tiedGB`). -/
theorem vit_net_tied_certified
    (xN wN bN gN aN clsN pN epsStr lrStr cotN : String) (ε : ℝ)
    (w : ViTTieWeights 10)
    (img : Vec (3 * 224 * 224)) (label : Fin 10) (lr : ℝ) :
    let ib1    : Vec (197 * 192) := patchEmbedFlat 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos img
    let ib2    : Vec (197 * 192) := w.b1.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib1
    let ib3    : Vec (197 * 192) := w.b2.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib2
    let ib4    : Vec (197 * 192) := w.b3.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib3
    let ib5    : Vec (197 * 192) := w.b4.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib4
    let ib6    : Vec (197 * 192) := w.b5.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib5
    let ib7    : Vec (197 * 192) := w.b6.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib6
    let ib8    : Vec (197 * 192) := w.b7.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib7
    let ib9    : Vec (197 * 192) := w.b8.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib8
    let ib10   : Vec (197 * 192) := w.b9.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib9
    let ib11   : Vec (197 * 192) := w.b10.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib10
    let ib12   : Vec (197 * 192) := w.b11.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib11
    let b12out : Vec (197 * 192) := w.b12.fwdO (Np1 := 197) (heads := 3) (d := 64) ε ib12
    let fl     : Vec (197 * 192) := Mat.flatten (fun r => layerNormVec 192 ε w.γF w.βF (Mat.unflatten b12out r))
    let hn     : Vec 192 := clsSliceFlat 196 192 fl
    let logits : Vec 10 := dense w.Wcls w.bcls hn
    let g      : Vec 10 := fun c => softmax 10 logits c - oneHot 10 label c
    let dy12   : Vec (197 * 192) := vitCotB2outV 196 192 10 ε w.γF w.Wcls b12out g
    let dy11   : Vec (197 * 192) := w.b12.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib12 dy12
    let dy10   : Vec (197 * 192) := w.b11.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib11 dy11
    let dy9    : Vec (197 * 192) := w.b10.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib10 dy10
    let dy8    : Vec (197 * 192) := w.b9.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib9 dy9
    let dy7    : Vec (197 * 192) := w.b8.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib8 dy8
    let dy6    : Vec (197 * 192) := w.b7.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib7 dy7
    let dy5    : Vec (197 * 192) := w.b6.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib6 dy6
    let dy4    : Vec (197 * 192) := w.b5.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib5 dy5
    let dy3    : Vec (197 * 192) := w.b4.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib4 dy4
    let dy2    : Vec (197 * 192) := w.b3.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib3 dy3
    let dy1    : Vec (197 * 192) := w.b2.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib2 dy2
    let dyEmbed: Vec (197 * 192) := w.b1.cotIn (Np1 := 197) (heads := 3) (d := 64) ε ib1 dy1
    w.b1.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib1 dy1 lr
  ∧ w.b2.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib2 dy2 lr
  ∧ w.b3.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib3 dy3 lr
  ∧ w.b4.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib4 dy4 lr
  ∧ w.b5.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib5 dy5 lr
  ∧ w.b6.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib6 dy6 lr
  ∧ w.b7.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib7 dy7 lr
  ∧ w.b8.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib8 dy8 lr
  ∧ w.b9.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib9 dy9 lr
  ∧ w.b10.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib10 dy10 lr
  ∧ w.b11.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib11 dy11 lr
  ∧ w.b12.TiedAt (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib12 dy12 lr
  ∧ vitFinalLNTied gN xN bN epsStr lrStr cotN ε w.γF w.βF w.Wcls b12out g lr
  ∧ vitHeadTied aN wN bN lrStr cotN hn w.Wcls w.bcls g lr
  ∧ vitEmbedTied wN xN bN clsN pN lrStr cotN w.Wc w.bc w.cls w.pos img dyEmbed lr := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out fl hn logits g dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact w.b1.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib1 dy1 lr
  · exact w.b2.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib2 dy2 lr
  · exact w.b3.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib3 dy3 lr
  · exact w.b4.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib4 dy4 lr
  · exact w.b5.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib5 dy5 lr
  · exact w.b6.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib6 dy6 lr
  · exact w.b7.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib7 dy7 lr
  · exact w.b8.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib8 dy8 lr
  · exact w.b9.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib9 dy9 lr
  · exact w.b10.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib10 dy10 lr
  · exact w.b11.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib11 dy11 lr
  · exact w.b12.tied_at (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε ib12 dy12 lr
  · exact vit_finalLN_tied gN xN bN epsStr lrStr cotN ε w.γF w.βF w.Wcls b12out g lr
  · exact vit_head_tied aN wN bN lrStr cotN hn w.Wcls w.bcls g lr
  · exact vit_embed_tied wN xN bN clsN pN lrStr cotN w.Wc w.bc w.cls w.pos img dyEmbed lr

end Proofs.ViTTiePoC
