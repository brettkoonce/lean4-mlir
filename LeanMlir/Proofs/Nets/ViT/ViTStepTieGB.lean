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
prefix the fused file threads (`patchEmbed_flat`, `vitBlockFwdOMHV`, the final LN, `clsSliceFlat`)
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
open Proofs.ViTTiePoC (vitBlockFwdOMHV vitBlockCotInAtMHV)

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
    (Mat.mul (rowSoftmax (fun i j => sdpa_scale d *
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
  (∀ kk : Fin (heads * d),
      den (SHlo.veclnGammaGradB (N := N) (R := Np1) (D := heads * d) xN epsStr ε xin
            (.operand cotN cotLn1B)) kk
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun gv : Vec (heads * d) =>
                    Mat.flatten (fun r => layerNormVec (heads * d) ε gv β1
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) xin n) r)))
                 γ1 kk o * batchSlice N (Np1 * (heads * d)) cotLn1B n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotLn1B)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun bv : Vec (heads * d) =>
                    Mat.flatten (fun r => layerNormVec (heads * d) ε γ1 bv
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) xin n) r)))
                 β1 i o * batchSlice N (Np1 * (heads * d)) cotLn1B n o)
  -- Q dense W/b  (cot = dQB, dense input = ln1B)
  ∧(∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
      den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := (heads * d)) (c := (heads * d)) xN ln1B
            (.operand cotN dQB)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun vmat : Vec ((heads * d) * (heads * d)) =>
                    Mat.flatten (fun r => dense (Mat.unflatten vmat) bq
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n) r)))
                 (Mat.flatten Wq) (finProdFinEquiv (i, j)) o * batchSlice N (Np1 * (heads * d)) dQB n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := (heads * d)) (.operand cotN dQB)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun b' : Vec (heads * d) =>
                    Mat.flatten (fun r => dense Wq b'
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n) r)))
                 bq i o * batchSlice N (Np1 * (heads * d)) dQB n o)
  -- K dense W/b  (cot = dKB)
  ∧(∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
      den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := (heads * d)) (c := (heads * d)) xN ln1B
            (.operand cotN dKB)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun vmat : Vec ((heads * d) * (heads * d)) =>
                    Mat.flatten (fun r => dense (Mat.unflatten vmat) bk
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n) r)))
                 (Mat.flatten Wk) (finProdFinEquiv (i, j)) o * batchSlice N (Np1 * (heads * d)) dKB n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := (heads * d)) (.operand cotN dKB)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun b' : Vec (heads * d) =>
                    Mat.flatten (fun r => dense Wk b'
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n) r)))
                 bk i o * batchSlice N (Np1 * (heads * d)) dKB n o)
  -- V dense W/b  (cot = dVB)
  ∧(∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
      den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := (heads * d)) (c := (heads * d)) xN ln1B
            (.operand cotN dVB)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun vmat : Vec ((heads * d) * (heads * d)) =>
                    Mat.flatten (fun r => dense (Mat.unflatten vmat) bv
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n) r)))
                 (Mat.flatten Wv) (finProdFinEquiv (i, j)) o * batchSlice N (Np1 * (heads * d)) dVB n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := (heads * d)) (.operand cotN dVB)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun b' : Vec (heads * d) =>
                    Mat.flatten (fun r => dense Wv b'
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n) r)))
                 bv i o * batchSlice N (Np1 * (heads * d)) dVB n o)
  -- out-proj dense W/b  (cot = cotHB, dense input = attB)
  ∧(∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
      den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := (heads * d)) (c := (heads * d)) xN attB
            (.operand cotN cotHB)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun vmat : Vec ((heads * d) * (heads * d)) =>
                    Mat.flatten (fun r => dense (Mat.unflatten vmat) bo
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) attB n) r)))
                 (Mat.flatten Wo) (finProdFinEquiv (i, j)) o * batchSlice N (Np1 * (heads * d)) cotHB n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := (heads * d)) (.operand cotN cotHB)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun b' : Vec (heads * d) =>
                    Mat.flatten (fun r => dense Wo b'
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) attB n) r)))
                 bo i o * batchSlice N (Np1 * (heads * d)) cotHB n o)
  -- LN₂ γ/β  (cot = cotLn2B, LN input = hB)
  ∧(∀ kk : Fin (heads * d),
      den (SHlo.veclnGammaGradB (N := N) (R := Np1) (D := heads * d) xN epsStr ε hB
            (.operand cotN cotLn2B)) kk
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun gv : Vec (heads * d) =>
                    Mat.flatten (fun r => layerNormVec (heads * d) ε gv β2
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) hB n) r)))
                 γ2 kk o * batchSlice N (Np1 * (heads * d)) cotLn2B n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := heads * d) (.operand cotN cotLn2B)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun bv : Vec (heads * d) =>
                    Mat.flatten (fun r => layerNormVec (heads * d) ε γ2 bv
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) hB n) r)))
                 β2 i o * batchSlice N (Np1 * (heads * d)) cotLn2B n o)
  -- fc1 dense W/b  (cot = cotM1B, dense input = ln2B)
  ∧(∀ (i : Fin (heads * d)) (j : Fin mlpDim),
      den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := (heads * d)) (c := mlpDim) xN ln2B
            (.operand cotN cotM1B)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ o : Fin (Np1 * mlpDim),
            pdiv (fun vmat : Vec ((heads * d) * mlpDim) =>
                    Mat.flatten (fun r => dense (Mat.unflatten vmat) bfc1
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln2B n) r)))
                 (Mat.flatten Wfc1) (finProdFinEquiv (i, j)) o * batchSlice N (Np1 * mlpDim) cotM1B n o)
  ∧(∀ i : Fin mlpDim,
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := mlpDim) (.operand cotN cotM1B)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * mlpDim),
            pdiv (fun b' : Vec mlpDim =>
                    Mat.flatten (fun r => dense Wfc1 b'
                      (Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln2B n) r)))
                 bfc1 i o * batchSlice N (Np1 * mlpDim) cotM1B n o)
  -- fc2 dense W/b  (cot = dyOut, dense input = gB)
  ∧(∀ (i : Fin mlpDim) (j : Fin (heads * d)),
      den (SHlo.rowDenseWeightGradB (N := N) (tk := Np1) (a := mlpDim) (c := (heads * d)) xN gB
            (.operand cotN dyOut)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun vmat : Vec (mlpDim * (heads * d)) =>
                    Mat.flatten (fun r => dense (Mat.unflatten vmat) bfc2
                      (Mat.unflatten (batchSlice N (Np1 * mlpDim) gB n) r)))
                 (Mat.flatten Wfc2) (finProdFinEquiv (i, j)) o * batchSlice N (Np1 * (heads * d)) dyOut n o)
  ∧(∀ i : Fin (heads * d),
      den (SHlo.rowDenseBiasGradB (N := N) (R := Np1) (c := (heads * d)) (.operand cotN dyOut)) i
        = ∑ n : Fin N, ∑ o : Fin (Np1 * (heads * d)),
            pdiv (fun b' : Vec (heads * d) =>
                    Mat.flatten (fun r => dense Wfc2 b'
                      (Mat.unflatten (batchSlice N (Np1 * mlpDim) gB n) r)))
                 bfc2 i o * batchSlice N (Np1 * (heads * d)) dyOut n o)

theorem vit_block_tiedGB (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) :
    vitBlockTiedGB N xN epsStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 bfc2
      xin dyOut := by
  unfold vitBlockTiedGB
  intro ln1B attB hB ln2B gB cotLn1B dQB dKB dVB cotHB cotLn2B cotM1B
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro kk; exact ViTPoCGB.veclnGammaGradB_den xN epsStr cotN ε β1 xin γ1 cotLn1B kk
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den_lnbeta cotN ε γ1 (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) xin n)) β1 cotLn1B i
  · intro i j; exact ViTPoCGB.rowDenseWeightGradB_den xN cotN bq ln1B Wq dQB i j
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den cotN Wq (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n)) bq dQB i
  · intro i j; exact ViTPoCGB.rowDenseWeightGradB_den xN cotN bk ln1B Wk dKB i j
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den cotN Wk (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n)) bk dKB i
  · intro i j; exact ViTPoCGB.rowDenseWeightGradB_den xN cotN bv ln1B Wv dVB i j
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den cotN Wv (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln1B n)) bv dVB i
  · intro i j; exact ViTPoCGB.rowDenseWeightGradB_den xN cotN bo attB Wo cotHB i j
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den cotN Wo (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) attB n)) bo cotHB i
  · intro kk; exact ViTPoCGB.veclnGammaGradB_den xN epsStr cotN ε β2 hB γ2 cotLn2B kk
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den_lnbeta cotN ε γ2 (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) hB n)) β2 cotLn2B i
  · intro i j; exact ViTPoCGB.rowDenseWeightGradB_den xN cotN bfc1 ln2B Wfc1 cotM1B i j
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den cotN Wfc1 (fun n => Mat.unflatten (batchSlice N (Np1 * (heads * d)) ln2B n)) bfc1 cotM1B i
  · intro i j; exact ViTPoCGB.rowDenseWeightGradB_den xN cotN bfc2 gB Wfc2 dyOut i j
  · intro i; exact ViTPoCGB.rowDenseBiasGradB_den cotN Wfc2 (fun n => Mat.unflatten (batchSlice N (Np1 * mlpDim) gB n)) bfc2 dyOut i

@[irreducible] def vitBlockTiedGBAt (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) : Prop :=
  vitBlockTiedGB N xN epsStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 bfc2 xin dyOut

theorem vit_block_tiedGBAt (N : Nat) {Np1 heads d mlpDim : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin dyOut : Vec (N * (Np1 * (heads * d)))) :
    vitBlockTiedGBAt N xN epsStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 bfc2
      xin dyOut := by
  unfold vitBlockTiedGBAt
  exact vit_block_tiedGB N xN epsStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 Wfc2 bfc2
    xin dyOut

/-! ## Final LN, classifier and patch embedding — batched -/

/-- **Final vector-LN γF/βF, tied at the batched classifier-back cotangent** `vitCotFl` per
    example (the `clsPad` of `Wclsᵀ g_n`, exactly what the render's `dotOut → clsPad` computes). -/
def vitFinalLNTiedGB (N : Nat) {nC : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (b12out : Vec (N * (197 * 192))) (g : Vec (N * nC)) :
    Prop :=
  let cotFlB : Vec (N * (197 * 192)) := batchMap N (vitCotFl 196 192 nC Wcls) g
  (∀ k : Fin 192,
      den (SHlo.veclnGammaGradB (N := N) (R := 197) (D := 192) xN epsStr ε b12out
            (.operand cotN cotFlB)) k
        = ∑ n : Fin N, ∑ o : Fin (197 * 192),
            pdiv (fun gv : Vec 192 =>
                    Mat.flatten (fun r => layerNormVec 192 ε gv βF
                      (Mat.unflatten (batchSlice N (197 * 192) b12out n) r)))
                 γF k o * batchSlice N (197 * 192) cotFlB n o)
  ∧ (∀ i : Fin 192,
      den (SHlo.rowDenseBiasGradB (N := N) (R := 197) (c := 192) (.operand cotN cotFlB)) i
        = ∑ n : Fin N, ∑ o : Fin (197 * 192),
            pdiv (fun bv : Vec 192 =>
                    Mat.flatten (fun r => layerNormVec 192 ε γF bv
                      (Mat.unflatten (batchSlice N (197 * 192) b12out n) r)))
                 βF i o * batchSlice N (197 * 192) cotFlB n o)

theorem vit_finalLN_tiedGB (N : Nat) {nC : Nat} (xN epsStr cotN : String) (ε : ℝ)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (b12out : Vec (N * (197 * 192))) (g : Vec (N * nC)) :
    vitFinalLNTiedGB N xN epsStr cotN ε γF βF Wcls b12out g := by
  unfold vitFinalLNTiedGB
  intro cotFlB
  refine ⟨?_, ?_⟩
  · intro k; exact ViTPoCGB.veclnGammaGradB_den xN epsStr cotN ε βF b12out γF cotFlB k
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
                    patchEmbed_flat 3 224 224 16 196 192 (Kernel4.unflatten v) bc cls pos
                      (batchSlice N (3 * 224 * 224) img n))
              (Kernel4.flatten Wc)
              (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (dd, c), kh), kw)) o
              * batchSlice N ((196 + 1) * 192) dyEmbed n o)
  ∧ (∀ i : Fin 192,
      den (SHlo.patchEmbedBiasGradB (N := N) (tk := 196) (c := 192) (.operand cotN dyEmbed)) i
        = ∑ n : Fin N, ∑ o : Fin ((196 + 1) * 192),
            pdiv (fun b' : Vec 192 =>
                    patchEmbed_flat 3 224 224 16 196 192 Wc b' cls pos
                      (batchSlice N (3 * 224 * 224) img n)) bc i o
              * batchSlice N ((196 + 1) * 192) dyEmbed n o)
  ∧ (∀ i : Fin 192,
      den (SHlo.denseBiasGradB (N := N) (c := 192)
            (.operand cotN (batchMap N (clsSliceFlat 196 192) dyEmbed))) i
        = ∑ n : Fin N, ∑ j : Fin (197 * 192),
            pdiv (fun cl : Vec 192 =>
                    patchEmbed_flat 3 224 224 16 196 192 Wc bc cl pos
                      (batchSlice N (3 * 224 * 224) img n)) cls i j
              * batchSlice N (197 * 192) dyEmbed n j)
  ∧ (∀ i : Fin ((196 + 1) * 192),
      den (SHlo.posEmbedGradB (N := N) (tk := 196) (D := 192) (.operand cotN dyEmbed)) i
        = ∑ n : Fin N, ∑ o : Fin ((196 + 1) * 192),
            pdiv (fun p : Vec ((196 + 1) * 192) =>
                    patchEmbed_flat 3 224 224 16 196 192 Wc bc cls (Mat.unflatten p)
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

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
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
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (γF βF : Vec 192) (Wcls : Mat 192 nC) (bcls : Vec nC)
    -- block 1
    (lnG1_1 lnB1_1 lnG2_1 lnB2_1 : Vec 192) (mWq_1 mWk_1 mWv_1 mWo_1 : Mat 192 192) (mbq_1 mbk_1 mbv_1 mbo_1 : Vec 192)
    (fW1_1 : Mat 192 768) (fb1_1 : Vec 768) (fW2_1 : Mat 768 192) (fb2_1 : Vec 192)
    -- block 2
    (lnG1_2 lnB1_2 lnG2_2 lnB2_2 : Vec 192) (mWq_2 mWk_2 mWv_2 mWo_2 : Mat 192 192) (mbq_2 mbk_2 mbv_2 mbo_2 : Vec 192)
    (fW1_2 : Mat 192 768) (fb1_2 : Vec 768) (fW2_2 : Mat 768 192) (fb2_2 : Vec 192)
    -- block 3
    (lnG1_3 lnB1_3 lnG2_3 lnB2_3 : Vec 192) (mWq_3 mWk_3 mWv_3 mWo_3 : Mat 192 192) (mbq_3 mbk_3 mbv_3 mbo_3 : Vec 192)
    (fW1_3 : Mat 192 768) (fb1_3 : Vec 768) (fW2_3 : Mat 768 192) (fb2_3 : Vec 192)
    -- block 4
    (lnG1_4 lnB1_4 lnG2_4 lnB2_4 : Vec 192) (mWq_4 mWk_4 mWv_4 mWo_4 : Mat 192 192) (mbq_4 mbk_4 mbv_4 mbo_4 : Vec 192)
    (fW1_4 : Mat 192 768) (fb1_4 : Vec 768) (fW2_4 : Mat 768 192) (fb2_4 : Vec 192)
    -- block 5
    (lnG1_5 lnB1_5 lnG2_5 lnB2_5 : Vec 192) (mWq_5 mWk_5 mWv_5 mWo_5 : Mat 192 192) (mbq_5 mbk_5 mbv_5 mbo_5 : Vec 192)
    (fW1_5 : Mat 192 768) (fb1_5 : Vec 768) (fW2_5 : Mat 768 192) (fb2_5 : Vec 192)
    -- block 6
    (lnG1_6 lnB1_6 lnG2_6 lnB2_6 : Vec 192) (mWq_6 mWk_6 mWv_6 mWo_6 : Mat 192 192) (mbq_6 mbk_6 mbv_6 mbo_6 : Vec 192)
    (fW1_6 : Mat 192 768) (fb1_6 : Vec 768) (fW2_6 : Mat 768 192) (fb2_6 : Vec 192)
    -- block 7
    (lnG1_7 lnB1_7 lnG2_7 lnB2_7 : Vec 192) (mWq_7 mWk_7 mWv_7 mWo_7 : Mat 192 192) (mbq_7 mbk_7 mbv_7 mbo_7 : Vec 192)
    (fW1_7 : Mat 192 768) (fb1_7 : Vec 768) (fW2_7 : Mat 768 192) (fb2_7 : Vec 192)
    -- block 8
    (lnG1_8 lnB1_8 lnG2_8 lnB2_8 : Vec 192) (mWq_8 mWk_8 mWv_8 mWo_8 : Mat 192 192) (mbq_8 mbk_8 mbv_8 mbo_8 : Vec 192)
    (fW1_8 : Mat 192 768) (fb1_8 : Vec 768) (fW2_8 : Mat 768 192) (fb2_8 : Vec 192)
    -- block 9
    (lnG1_9 lnB1_9 lnG2_9 lnB2_9 : Vec 192) (mWq_9 mWk_9 mWv_9 mWo_9 : Mat 192 192) (mbq_9 mbk_9 mbv_9 mbo_9 : Vec 192)
    (fW1_9 : Mat 192 768) (fb1_9 : Vec 768) (fW2_9 : Mat 768 192) (fb2_9 : Vec 192)
    -- block 10
    (lnG1_10 lnB1_10 lnG2_10 lnB2_10 : Vec 192) (mWq_10 mWk_10 mWv_10 mWo_10 : Mat 192 192) (mbq_10 mbk_10 mbv_10 mbo_10 : Vec 192)
    (fW1_10 : Mat 192 768) (fb1_10 : Vec 768) (fW2_10 : Mat 768 192) (fb2_10 : Vec 192)
    -- block 11
    (lnG1_11 lnB1_11 lnG2_11 lnB2_11 : Vec 192) (mWq_11 mWk_11 mWv_11 mWo_11 : Mat 192 192) (mbq_11 mbk_11 mbv_11 mbo_11 : Vec 192)
    (fW1_11 : Mat 192 768) (fb1_11 : Vec 768) (fW2_11 : Mat 768 192) (fb2_11 : Vec 192)
    -- block 12
    (lnG1_12 lnB1_12 lnG2_12 lnB2_12 : Vec 192) (mWq_12 mWk_12 mWv_12 mWo_12 : Mat 192 192) (mbq_12 mbk_12 mbv_12 mbo_12 : Vec 192)
    (fW1_12 : Mat 192 768) (fb1_12 : Vec 768) (fW2_12 : Mat 768 192) (fb2_12 : Vec 192)
    (img : Vec (N * (3 * 224 * 224))) (t : Vec (N * nC)) :
    let ib1    : Vec (N * (197 * 192)) := batchMap N (patchEmbed_flat 3 224 224 16 196 192 Wc bc cls pos) img
    let ib2    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ) ib1
    let ib3    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ) ib2
    let ib4    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ) ib3
    let ib5    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ) ib4
    let ib6    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ) ib5
    let ib7    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ) ib6
    let ib8    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ) ib7
    let ib9    : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ) ib8
    let ib10   : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ) ib9
    let ib11   : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ) ib10
    let ib12   : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ) ib11
    let b12out : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ) ib12
    -- final LN → CLS row → dense head, then the SMOOTHED loss cotangent at a general target `t`
    let flB     : Vec (N * (197 * 192)) :=
      batchMap N (fun b => Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten b r))) b12out
    let hnB     : Vec (N * 192) := batchMap N (clsSliceFlat 196 192) flB
    let logitsB : Vec (N * nC)  := batchMap N (dense Wcls bcls) hnB
    let g       : Vec (N * nC)  :=
      den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t)
    let dy12    : Vec (N * (197 * 192)) := batchMapAux N (vitCotB2outV 196 192 nC ε γF Wcls) b12out g
    let dy11   : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 ) ib12 dy12
    let dy10   : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 ) ib11 dy11
    let dy9    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 ) ib10 dy10
    let dy8    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 ) ib9 dy9
    let dy7    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 ) ib8 dy8
    let dy6    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 ) ib7 dy7
    let dy5    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 ) ib6 dy6
    let dy4    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 ) ib5 dy5
    let dy3    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 ) ib4 dy4
    let dy2    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 ) ib3 dy3
    let dy1    : Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 ) ib2 dy2
    let dyEmbed: Vec (N * (197 * 192)) := batchMapAux N (vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 ) ib1 dy1
    vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9 dy9
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10 dy10
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11 dy11
  ∧ vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12
  ∧ vitFinalLNTiedGB N xN epsStr cotN ε γF βF Wcls b12out g
  ∧ vitHeadTiedGB N aN cotN hnB Wcls bcls g
  ∧ vitEmbedTiedGB N xN cotN Wc bc cls pos img dyEmbed := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out flB hnB logitsB g dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9 dy9
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10 dy10
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11 dy11
  · exact vit_block_tiedGBAt N (Np1 := 197) (heads := 3) (d := 64) xN epsStr cotN ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12
  · exact vit_finalLN_tiedGB N xN epsStr cotN ε γF βF Wcls b12out g
  · exact vit_head_tiedGB N aN cotN hnB Wcls bcls g
  · exact vit_embed_tiedGB N xN cotN Wc bc cls pos img dyEmbed

end Proofs.ViTTiePoCGB
