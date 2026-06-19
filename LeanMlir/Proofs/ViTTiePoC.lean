import LeanMlir.Proofs.ViTFaithfulPoC
import LeanMlir.Proofs.ViTVecLN
import LeanMlir.Proofs.ViTChainClose
import LeanMlir.Proofs.ViTMultiHeadChain

/-! # ViT-Tiny §1a tie — the transformer block tied through the real backward cotangent chain

The ViT peer of `ConvNeXtTiePoC`/`MobileNetV2TiePoCPaper`: thread the §1-fold param-SGD `den`otations
(`ViTPoC.*_den`, generic in the cotangent) at the **actual** cotangents the block backward chain
delivers (`ViTVecLN`/`ViTChainClose`'s `vitCot*` family), so every block param's `den = θ − lr·(certified
∂/∂θ · the real-chain cotangent)` — one composed fact, the real loss-driven backward, not a free `∀ c`.

The new structural content vs every prior net: **two residual fan-ins per block** (the MLP residual
`vitCotHV = dyOut + LN₂-back(…)` and the attention residual `vitCotXinV = cotH + LN₁-back(…)`), and the
**three-way fan-in at LN₁'s output** (the Q/K/V dense-backs all read `LN₁ x`, so their cotangents SUM in
`vitCotLn1`). The per-head SDPA backward is pinned to the audited `sdpa_back_{Q,K,V}` suite
(`ViTChainClose.vitCotD{Q,K,V}`). Pure thread + fan-in — ZERO new core ops, ZERO new bridges; each
conjunct is a delegation to a §1-fold generic at the chain cotangent.

**Scope (honest, the mnv2 reduced-net situation).** The vector-LN chain infrastructure
(`vitForward2V`, the `*V` cots, the `vecln*_chain_certified` certs) is **single-head** (`heads = 1`);
the committed `vitTrainStepRenderV` render is **multi-head (3 heads), depth-12**. So this ties the
per-block **vector-LN representative**; promoting to the multi-head/depth-12 committed render needs
multi-head chain cotangents (the per-head `headSlice`/`headPad` backs summed over heads) — the analogue
of mnv2's reduced→full and the next vit step. The vector-LN granularity (the `[D]` LN that SHIPS) IS
modeled here (NOT the scalar-LN `vitNetBackGraph` parallel universe). -/

namespace Proofs.ViTTiePoC

open scoped BigOperators
open Proofs Proofs.StableHLO

/-- **One vector-LN transformer block, tied.** Every one of the block's 16 params, fed the cotangent
    the real backward chain delivers at its site, `den`otes `θ − lr·(certified ∂/∂θ · chain-cot)`.
    Threads the saved single-head activations (`xin`→`ln1`→`q/k/v`→`ss`/`p`→`att`→`h`→`ln2`→`m1`→`g`)
    and the chain cotangents (the MLP-residual + attention-residual + three-way LN₁ fan-ins). -/
def vitBlockTiedV {Np1 D mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec D) (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (xin ln1 q k v att h ln2 : Vec (Np1 * D)) (g : Vec (Np1 * mlpDim))
    (ss p : Vec (Np1 * Np1)) (m1 : Vec (Np1 * mlpDim))
    (dyOut : Vec (Np1 * D)) (lr : ℝ) : Prop :=
    let dAtt   : Vec (Np1 * D)      := vitCotAttV ε γ2 Wo Wfc1 Wfc2 h m1 dyOut
    let dQ     : Vec (Np1 * D)      := vitCotDQ D ss k v dAtt
    let dK     : Vec (Np1 * D)      := vitCotDK D ss q v dAtt
    let dV     : Vec (Np1 * D)      := vitCotDV p dAtt
    let cotLn1 : Vec (Np1 * D)      := vitCotLn1 Wq Wk Wv dQ dK dV
    let cotH   : Vec (Np1 * D)      := vitCotHV ε γ2 Wfc1 Wfc2 h m1 dyOut
    let cotLn2 : Vec (Np1 * D)      := vitCotLn2 Wfc1 Wfc2 m1 dyOut
    let cotM1  : Vec (Np1 * mlpDim) := vitCotM1 Wfc2 m1 dyOut
    -- LN₁ γ/β  (cot = cotLn1, LN input = xin)
    (∀ kk : Fin D,
        den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε xin γ1 lr (.operand cotN cotLn1)) kk
          = γ1 kk - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun gv : Vec D => Mat.flatten (fun r => layerNormVec D ε gv β1 (Mat.unflatten xin r))) γ1 kk o * cotLn1 o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr β1 lr (.operand cotN cotLn1)) i
          = β1 i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γ1 bv (Mat.unflatten xin r))) β1 i o * cotLn1 o)
    -- Q dense W/b  (cot = dQ, dense input = ln1)
  ∧ (∀ (i : Fin D) (j : Fin D),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln1 Wq lr (.operand cotN dQ)) (finProdFinEquiv (i, j))
          = Wq i j - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun vmat : Vec (D * D) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bq (Mat.unflatten ln1 r))) (Mat.flatten Wq) (finProdFinEquiv (i, j)) o * dQ o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr bq lr (.operand cotN dQ)) i
          = bq i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun b' : Vec D => Mat.flatten (fun r => dense Wq b' (Mat.unflatten ln1 r))) bq i o * dQ o)
    -- K dense W/b  (cot = dK)
  ∧ (∀ (i : Fin D) (j : Fin D),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln1 Wk lr (.operand cotN dK)) (finProdFinEquiv (i, j))
          = Wk i j - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun vmat : Vec (D * D) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bk (Mat.unflatten ln1 r))) (Mat.flatten Wk) (finProdFinEquiv (i, j)) o * dK o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr bk lr (.operand cotN dK)) i
          = bk i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun b' : Vec D => Mat.flatten (fun r => dense Wk b' (Mat.unflatten ln1 r))) bk i o * dK o)
    -- V dense W/b  (cot = dV)
  ∧ (∀ (i : Fin D) (j : Fin D),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln1 Wv lr (.operand cotN dV)) (finProdFinEquiv (i, j))
          = Wv i j - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun vmat : Vec (D * D) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bv (Mat.unflatten ln1 r))) (Mat.flatten Wv) (finProdFinEquiv (i, j)) o * dV o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr bv lr (.operand cotN dV)) i
          = bv i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun b' : Vec D => Mat.flatten (fun r => dense Wv b' (Mat.unflatten ln1 r))) bv i o * dV o)
    -- out-proj dense W/b  (cot = cotH, dense input = att)
  ∧ (∀ (i : Fin D) (j : Fin D),
        den (SHlo.rowDenseWeightSgd xN wN lrStr att Wo lr (.operand cotN cotH)) (finProdFinEquiv (i, j))
          = Wo i j - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun vmat : Vec (D * D) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bo (Mat.unflatten att r))) (Mat.flatten Wo) (finProdFinEquiv (i, j)) o * cotH o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr bo lr (.operand cotN cotH)) i
          = bo i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun b' : Vec D => Mat.flatten (fun r => dense Wo b' (Mat.unflatten att r))) bo i o * cotH o)
    -- LN₂ γ/β  (cot = cotLn2, LN input = h)
  ∧ (∀ kk : Fin D,
        den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε h γ2 lr (.operand cotN cotLn2)) kk
          = γ2 kk - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun gv : Vec D => Mat.flatten (fun r => layerNormVec D ε gv β2 (Mat.unflatten h r))) γ2 kk o * cotLn2 o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr β2 lr (.operand cotN cotLn2)) i
          = β2 i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γ2 bv (Mat.unflatten h r))) β2 i o * cotLn2 o)
    -- fc1 dense W/b  (cot = cotM1, dense input = ln2)
  ∧ (∀ (i : Fin D) (j : Fin mlpDim),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln2 Wfc1 lr (.operand cotN cotM1)) (finProdFinEquiv (i, j))
          = Wfc1 i j - lr * ∑ o : Fin (Np1 * mlpDim),
              pdiv (fun vmat : Vec (D * mlpDim) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bfc1 (Mat.unflatten ln2 r))) (Mat.flatten Wfc1) (finProdFinEquiv (i, j)) o * cotM1 o)
  ∧ (∀ i : Fin mlpDim,
        den (SHlo.rowDenseBiasSgd bN lrStr bfc1 lr (.operand cotN cotM1)) i
          = bfc1 i - lr * ∑ o : Fin (Np1 * mlpDim),
              pdiv (fun b' : Vec mlpDim => Mat.flatten (fun r => dense Wfc1 b' (Mat.unflatten ln2 r))) bfc1 i o * cotM1 o)
    -- fc2 dense W/b  (cot = dyOut, dense input = g)
  ∧ (∀ (i : Fin mlpDim) (j : Fin D),
        den (SHlo.rowDenseWeightSgd xN wN lrStr g Wfc2 lr (.operand cotN dyOut)) (finProdFinEquiv (i, j))
          = Wfc2 i j - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun vmat : Vec (mlpDim * D) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bfc2 (Mat.unflatten g r))) (Mat.flatten Wfc2) (finProdFinEquiv (i, j)) o * dyOut o)
  ∧ (∀ i : Fin D,
        den (SHlo.rowDenseBiasSgd bN lrStr bfc2 lr (.operand cotN dyOut)) i
          = bfc2 i - lr * ∑ o : Fin (Np1 * D),
              pdiv (fun b' : Vec D => Mat.flatten (fun r => dense Wfc2 b' (Mat.unflatten g r))) bfc2 i o * dyOut o)

/-- **The block tie holds** — every conjunct is a delegation to a §1-fold generic (`ViTPoC.*_den`) at
    the chain cotangent. Pure thread + fan-in; ZERO new core ops/bridges. -/
theorem vit_block_tiedV {Np1 D mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec D) (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (xin ln1 q k v att h ln2 : Vec (Np1 * D)) (g : Vec (Np1 * mlpDim))
    (ss p : Vec (Np1 * Np1)) (m1 : Vec (Np1 * mlpDim))
    (dyOut : Vec (Np1 * D)) (lr : ℝ) :
    vitBlockTiedV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
      Wfc1 bfc1 Wfc2 bfc2 xin ln1 q k v att h ln2 g ss p m1 dyOut lr := by
  unfold vitBlockTiedV
  intro dAtt dQ dK dV cotLn1 cotH cotLn2 cotM1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro kk;  exact ViTPoC.veclnGammaSgd_den gN xN epsStr lrStr cotN ε β1 xin γ1 cotLn1 lr kk
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε γ1 (Mat.unflatten xin) β1 cotLn1 lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bq ln1 Wq dQ lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wq (Mat.unflatten ln1) bq dQ lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bk ln1 Wk dK lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wk (Mat.unflatten ln1) bk dK lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bv ln1 Wv dV lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wv (Mat.unflatten ln1) bv dV lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bo att Wo cotH lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wo (Mat.unflatten att) bo cotH lr i
  · intro kk;  exact ViTPoC.veclnGammaSgd_den gN xN epsStr lrStr cotN ε β2 h γ2 cotLn2 lr kk
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε γ2 (Mat.unflatten h) β2 cotLn2 lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bfc1 ln2 Wfc1 cotM1 lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wfc1 (Mat.unflatten ln2) bfc1 cotM1 lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bfc2 g Wfc2 dyOut lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wfc2 (Mat.unflatten g) bfc2 dyOut lr i

/-! ## Whole-net thread (single-head vector-LN, 2-block representative) — the convnext pattern

`vitBlockFwdOV` is the forward block step (= `vitBlockSpelledV`, exposing the block output);
`vitBlockTiedAtV` recomputes the 11 saved activations from the block INPUT (the `vitBlockSpelledV`
let-chain) and delegates to `vit_block_tiedV` — so the block's params tie at the REAL forward + the
threaded `dyOut`. `vitBlockCotInAtV` is the attention-residual fan-in (`vitCotXinV`), the block's input
cotangent (= the previous block's `dyOut`). `@[irreducible]` so the nested 2-block composition stays
opaque (the r34/mnv2 heartbeat lesson). -/

@[irreducible] noncomputable def vitBlockFwdOV {Np1 D mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec D) (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (xin : Vec (Np1 * D)) : Vec (Np1 * D) :=
  -- inline `vitBlockSpelledV` at D (it is stated at `1 * D`, not defeq) — the block output
  let X    : Mat Np1 D := Mat.unflatten xin
  let ln1  : Mat Np1 D := fun r k => layerScale γ1 (fun s => layerNormForward D ε 1 0 (X r) s) k + β1 k
  let Q    : Mat Np1 D := fun r => dense Wq bq (ln1 r)
  let K    : Mat Np1 D := fun r => dense Wk bk (ln1 r)
  let V    : Mat Np1 D := fun r => dense Wv bv (ln1 r)
  let P    : Mat Np1 Np1 := rowSoftmax (fun i j => sdpa_scale D * Mat.mul Q (Mat.transpose K) i j)
  let h    : Mat Np1 D := fun r s => X r s + dense Wo bo (Mat.mul P V r) s
  let ln2  : Mat Np1 D := fun r k => layerScale γ2 (fun s => layerNormForward D ε 1 0 (h r) s) k + β2 k
  let g    : Mat Np1 mlpDim := fun r => gelu mlpDim (dense Wfc1 bfc1 (ln2 r))
  Mat.flatten (fun r s => h r s + dense Wfc2 bfc2 (g r) s)

/-- Attention-residual fan-in: the block-input cotangent the chain hands upstream (`vitCotXinV`,
    recomputing the saves from `xin`). -/
@[irreducible] noncomputable def vitBlockCotInAtV {Np1 D mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec D) (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D)
    (xin dyOut : Vec (Np1 * D)) : Vec (Np1 * D) :=
  let X    : Mat Np1 D := Mat.unflatten xin
  let ln1  : Mat Np1 D := fun r k => layerScale γ1 (fun s => layerNormForward D ε 1 0 (X r) s) k + β1 k
  let Q    : Mat Np1 D := fun r => dense Wq bq (ln1 r)
  let K    : Mat Np1 D := fun r => dense Wk bk (ln1 r)
  let V    : Mat Np1 D := fun r => dense Wv bv (ln1 r)
  let ss   : Mat Np1 Np1 := fun i j => sdpa_scale D * Mat.mul Q (Mat.transpose K) i j
  let P    : Mat Np1 Np1 := rowSoftmax ss
  let att  : Mat Np1 D := Mat.mul P V
  let h    : Mat Np1 D := fun r s => X r s + dense Wo bo (att r) s
  let ln2  : Mat Np1 D := fun r k => layerScale γ2 (fun s => layerNormForward D ε 1 0 (h r) s) k + β2 k
  let m1   : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let dAtt := vitCotAttV ε γ2 Wo Wfc1 Wfc2 (Mat.flatten h) (Mat.flatten m1) dyOut
  let dQ   := vitCotDQ D (Mat.flatten ss) (Mat.flatten K) (Mat.flatten V) dAtt
  let dK   := vitCotDK D (Mat.flatten ss) (Mat.flatten Q) (Mat.flatten V) dAtt
  let dV   := vitCotDV (Mat.flatten P) dAtt
  let cotH := vitCotHV ε γ2 Wfc1 Wfc2 (Mat.flatten h) (Mat.flatten m1) dyOut
  vitCotXinV ε γ1 Wq Wk Wv xin dQ dK dV cotH

/-- **Input-only block tie** — recompute the 11 saves from `xin` (the `vitBlockSpelledV` let-chain),
    then the generic block tie holds. The vit peer of `cnxBlockTiedAt`. -/
@[irreducible] def vitBlockTiedAtV {Np1 D mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec D) (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (xin dyOut : Vec (Np1 * D)) (lr : ℝ) : Prop :=
  let X    : Mat Np1 D := Mat.unflatten xin
  let ln1  : Mat Np1 D := fun r k => layerScale γ1 (fun s => layerNormForward D ε 1 0 (X r) s) k + β1 k
  let Q    : Mat Np1 D := fun r => dense Wq bq (ln1 r)
  let K    : Mat Np1 D := fun r => dense Wk bk (ln1 r)
  let V    : Mat Np1 D := fun r => dense Wv bv (ln1 r)
  let ss   : Mat Np1 Np1 := fun i j => sdpa_scale D * Mat.mul Q (Mat.transpose K) i j
  let P    : Mat Np1 Np1 := rowSoftmax ss
  let att  : Mat Np1 D := Mat.mul P V
  let h    : Mat Np1 D := fun r s => X r s + dense Wo bo (att r) s
  let ln2  : Mat Np1 D := fun r k => layerScale γ2 (fun s => layerNormForward D ε 1 0 (h r) s) k + β2 k
  let m1   : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g    : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  vitBlockTiedV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
    Wfc1 bfc1 Wfc2 bfc2 xin (Mat.flatten ln1) (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V)
    (Mat.flatten att) (Mat.flatten h) (Mat.flatten ln2) (Mat.flatten g)
    (Mat.flatten ss) (Mat.flatten P) (Mat.flatten m1) dyOut lr

/-- **The input-only block tie holds.** Unfold the saves, delegate to `vit_block_tiedV`. -/
theorem vit_block_tiedAtV {Np1 D mlpDim : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec D) (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (xin dyOut : Vec (Np1 * D)) (lr : ℝ) :
    vitBlockTiedAtV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
      Wfc1 bfc1 Wfc2 bfc2 xin dyOut lr := by
  unfold vitBlockTiedAtV
  exact vit_block_tiedV xN wN bN gN epsStr lrStr cotN ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
    Wfc1 bfc1 Wfc2 bfc2 xin _ _ _ _ _ _ _ _ _ _ _ dyOut lr

set_option maxHeartbeats 4000000 in
/-- **The 2-block vector-LN ViT, tied through the real forward + the inter-block cotangent fan-in.**
    Block inputs are the real forward prefixes (`ib1` the embedded tokens, `ib2 = vitBlockFwdOV(ib1)`);
    the block-2 output cotangent is the final-LN input-VJP of the classifier-back (`vitCotB2outV`), and
    block 1's `dyOut` is what block 2 hands upstream (`vitBlockCotInAtV` — the attention-residual fan-in).
    BOTH blocks' 16 params then `den = θ − lr·(certified ∂/∂θ · real-chain cotangent)`. The vit peer of
    convnext's `cnx_net_tied_certified` at the 2-block single-head representative. (Final-LN γ/β, the
    classifier, and the patch embed reuse the §1-fold + chain certs directly; not bundled here.) -/
theorem vit_net_tiedV {N D mlpDim nClasses : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ) (γF : Vec D) (Wcls : Mat D nClasses)
    -- block 1
    (γ1₁ β1₁ γ2₁ β2₁ : Vec D) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat D D) (bq₁ bk₁ bv₁ bo₁ : Vec D)
    (Wfc1₁ : Mat D mlpDim) (bfc1₁ : Vec mlpDim) (Wfc2₁ : Mat mlpDim D) (bfc2₁ : Vec D)
    -- block 2
    (γ1₂ β1₂ γ2₂ β2₂ : Vec D) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat D D) (bq₂ bk₂ bv₂ bo₂ : Vec D)
    (Wfc1₂ : Mat D mlpDim) (bfc1₂ : Vec mlpDim) (Wfc2₂ : Mat mlpDim D) (bfc2₂ : Vec D)
    (ib1 : Vec ((N + 1) * D)) (dy : Vec nClasses) (lr : ℝ) :
    let ib2    : Vec ((N + 1) * D) := vitBlockFwdOV ε γ1₁ β1₁ γ2₁ β2₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ ib1
    let b2out  : Vec ((N + 1) * D) := vitBlockFwdOV ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ ib2
    let dyOut2 : Vec ((N + 1) * D) := vitCotB2outV N D nClasses ε γF Wcls b2out dy
    let dyOut1 : Vec ((N + 1) * D) := vitBlockCotInAtV ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ Wfc1₂ bfc1₂ Wfc2₂ ib2 dyOut2
    -- block 1 tied at its real input (ib1) + the cotangent block 2 hands upstream (dyOut1)
    vitBlockTiedAtV xN wN bN gN epsStr lrStr cotN ε γ1₁ β1₁ γ2₁ β2₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁
        Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ ib1 dyOut1 lr
    -- block 2 tied at its real input (ib2) + the final-LN-back cotangent (dyOut2)
  ∧ vitBlockTiedAtV xN wN bN gN epsStr lrStr cotN ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂
        Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ ib2 dyOut2 lr := by
  intro ib2 b2out dyOut2 dyOut1
  refine ⟨?_, ?_⟩
  · exact vit_block_tiedAtV xN wN bN gN epsStr lrStr cotN ε γ1₁ β1₁ γ2₁ β2₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ ib1 dyOut1 lr
  · exact vit_block_tiedAtV xN wN bN gN epsStr lrStr cotN ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ ib2 dyOut2 lr


/-! ## Multi-head promotion (3 heads, d_head=64) — the committed-render block tie

The committed `vitTrainStepRenderV` is multi-head: the SDPA-internal backward `dAtt → dQ/dK/dV`
runs per head (`vitCotD{Q,K,V}mh`, `ViTMultiHeadChain`), so the Q/K/V dense cots change from the
single-head `vitCotD{Q,K,V}` to the multi-head `…mh` cots; everything else (the out-proj `Wo`, LN₂,
the MLP) is head-agnostic and unchanged. `vitBlockTiedMHV` is `vitBlockTiedV` with that swap (and
no separate `ss`/`p` saves — the per-head scores/weights are recomputed inside the `…mh` cots from
the saved Q/K). The 16 conjuncts and proof delegations are otherwise identical (the §1-fold generics
`ViTPoC.*_den` are head-agnostic). -/

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
    (∀ kk : Fin (heads * d),
        den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε xin γ1 lr (.operand cotN cotLn1)) kk
          = γ1 kk - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun gv : Vec (heads * d) => Mat.flatten (fun r => layerNormVec (heads * d) ε gv β1 (Mat.unflatten xin r))) γ1 kk o * cotLn1 o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr β1 lr (.operand cotN cotLn1)) i
          = β1 i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun bv : Vec (heads * d) => Mat.flatten (fun r => layerNormVec (heads * d) ε γ1 bv (Mat.unflatten xin r))) β1 i o * cotLn1 o)
    -- Q dense W/b  (cot = dQ, dense input = ln1)
  ∧ (∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln1 Wq lr (.operand cotN dQ)) (finProdFinEquiv (i, j))
          = Wq i j - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun vmat : Vec ((heads * d) * (heads * d)) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bq (Mat.unflatten ln1 r))) (Mat.flatten Wq) (finProdFinEquiv (i, j)) o * dQ o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr bq lr (.operand cotN dQ)) i
          = bq i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun b' : Vec (heads * d) => Mat.flatten (fun r => dense Wq b' (Mat.unflatten ln1 r))) bq i o * dQ o)
    -- K dense W/b  (cot = dK)
  ∧ (∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln1 Wk lr (.operand cotN dK)) (finProdFinEquiv (i, j))
          = Wk i j - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun vmat : Vec ((heads * d) * (heads * d)) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bk (Mat.unflatten ln1 r))) (Mat.flatten Wk) (finProdFinEquiv (i, j)) o * dK o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr bk lr (.operand cotN dK)) i
          = bk i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun b' : Vec (heads * d) => Mat.flatten (fun r => dense Wk b' (Mat.unflatten ln1 r))) bk i o * dK o)
    -- V dense W/b  (cot = dV)
  ∧ (∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln1 Wv lr (.operand cotN dV)) (finProdFinEquiv (i, j))
          = Wv i j - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun vmat : Vec ((heads * d) * (heads * d)) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bv (Mat.unflatten ln1 r))) (Mat.flatten Wv) (finProdFinEquiv (i, j)) o * dV o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr bv lr (.operand cotN dV)) i
          = bv i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun b' : Vec (heads * d) => Mat.flatten (fun r => dense Wv b' (Mat.unflatten ln1 r))) bv i o * dV o)
    -- out-proj dense W/b  (cot = cotH, dense input = att)
  ∧ (∀ (i : Fin (heads * d)) (j : Fin (heads * d)),
        den (SHlo.rowDenseWeightSgd xN wN lrStr att Wo lr (.operand cotN cotH)) (finProdFinEquiv (i, j))
          = Wo i j - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun vmat : Vec ((heads * d) * (heads * d)) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bo (Mat.unflatten att r))) (Mat.flatten Wo) (finProdFinEquiv (i, j)) o * cotH o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr bo lr (.operand cotN cotH)) i
          = bo i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun b' : Vec (heads * d) => Mat.flatten (fun r => dense Wo b' (Mat.unflatten att r))) bo i o * cotH o)
    -- LN₂ γ/β  (cot = cotLn2, LN input = h)
  ∧ (∀ kk : Fin (heads * d),
        den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε h γ2 lr (.operand cotN cotLn2)) kk
          = γ2 kk - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun gv : Vec (heads * d) => Mat.flatten (fun r => layerNormVec (heads * d) ε gv β2 (Mat.unflatten h r))) γ2 kk o * cotLn2 o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr β2 lr (.operand cotN cotLn2)) i
          = β2 i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun bv : Vec (heads * d) => Mat.flatten (fun r => layerNormVec (heads * d) ε γ2 bv (Mat.unflatten h r))) β2 i o * cotLn2 o)
    -- fc1 dense W/b  (cot = cotM1, dense input = ln2)
  ∧ (∀ (i : Fin (heads * d)) (j : Fin mlpDim),
        den (SHlo.rowDenseWeightSgd xN wN lrStr ln2 Wfc1 lr (.operand cotN cotM1)) (finProdFinEquiv (i, j))
          = Wfc1 i j - lr * ∑ o : Fin (Np1 * mlpDim),
              pdiv (fun vmat : Vec ((heads * d) * mlpDim) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bfc1 (Mat.unflatten ln2 r))) (Mat.flatten Wfc1) (finProdFinEquiv (i, j)) o * cotM1 o)
  ∧ (∀ i : Fin mlpDim,
        den (SHlo.rowDenseBiasSgd bN lrStr bfc1 lr (.operand cotN cotM1)) i
          = bfc1 i - lr * ∑ o : Fin (Np1 * mlpDim),
              pdiv (fun b' : Vec mlpDim => Mat.flatten (fun r => dense Wfc1 b' (Mat.unflatten ln2 r))) bfc1 i o * cotM1 o)
    -- fc2 dense W/b  (cot = dyOut, dense input = g)
  ∧ (∀ (i : Fin mlpDim) (j : Fin (heads * d)),
        den (SHlo.rowDenseWeightSgd xN wN lrStr g Wfc2 lr (.operand cotN dyOut)) (finProdFinEquiv (i, j))
          = Wfc2 i j - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun vmat : Vec (mlpDim * (heads * d)) => Mat.flatten (fun r => dense (Mat.unflatten vmat) bfc2 (Mat.unflatten g r))) (Mat.flatten Wfc2) (finProdFinEquiv (i, j)) o * dyOut o)
  ∧ (∀ i : Fin (heads * d),
        den (SHlo.rowDenseBiasSgd bN lrStr bfc2 lr (.operand cotN dyOut)) i
          = bfc2 i - lr * ∑ o : Fin (Np1 * (heads * d)),
              pdiv (fun b' : Vec (heads * d) => Mat.flatten (fun r => dense Wfc2 b' (Mat.unflatten g r))) bfc2 i o * dyOut o)

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
  intro dAtt dQ dK dV cotLn1 cotH cotLn2 cotM1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro kk;  exact ViTPoC.veclnGammaSgd_den gN xN epsStr lrStr cotN ε β1 xin γ1 cotLn1 lr kk
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε γ1 (Mat.unflatten xin) β1 cotLn1 lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bq ln1 Wq dQ lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wq (Mat.unflatten ln1) bq dQ lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bk ln1 Wk dK lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wk (Mat.unflatten ln1) bk dK lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bv ln1 Wv dV lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wv (Mat.unflatten ln1) bv dV lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bo att Wo cotH lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wo (Mat.unflatten att) bo cotH lr i
  · intro kk;  exact ViTPoC.veclnGammaSgd_den gN xN epsStr lrStr cotN ε β2 h γ2 cotLn2 lr kk
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den_lnbeta bN lrStr cotN ε γ2 (Mat.unflatten h) β2 cotLn2 lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bfc1 ln2 Wfc1 cotM1 lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wfc1 (Mat.unflatten ln2) bfc1 cotM1 lr i
  · intro i j; exact ViTPoC.rowDenseWeightSgd_den xN wN lrStr cotN bfc2 g Wfc2 dyOut lr i j
  · intro i;   exact ViTPoC.rowDenseBiasSgd_den bN lrStr cotN Wfc2 (Mat.unflatten g) bfc2 dyOut lr i


/-! ## Multi-head forward + cot-in + input-only block wrappers (`@[irreducible]`, the thread template) -/

/-- Multi-head forward block step (the committed render's block forward = `vitBlockSpelledMHV`,
    which IS `transformerBlockV` at general `heads` by `vitBlockSpelledMHV_eq`). -/
@[irreducible] noncomputable def vitBlockFwdOMHV {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (xin : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  Mat.flatten (vitBlockSpelledMHV Np1 heads d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
    γ2 β2 Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten xin))

/-- Multi-head attention-residual fan-in: the block-input cotangent the chain hands upstream
    (`vitCotXinV` at the multi-head Q/K/V dense cots — `vitCotLn1 Wq Wk Wv dQmh dKmh dVmh` IS the
    multi-head LN₁ fan-in `vitCotLn1MH`). Recomputes the saves from `xin` (the `vitBlockSpelledMHV`
    let-chain, multi-head `att`). -/
@[irreducible] noncomputable def vitBlockCotInAtMHV {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
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
  let dAtt := vitCotAttV ε γ2 Wo Wfc1 Wfc2 (Mat.flatten h) (Mat.flatten m1) dyOut
  let dQ   := vitCotDQmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let dK   := vitCotDKmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let dV   := vitCotDVmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) dAtt
  let cotH := vitCotHV ε γ2 Wfc1 Wfc2 (Mat.flatten h) (Mat.flatten m1) dyOut
  vitCotXinV ε γ1 Wq Wk Wv xin dQ dK dV cotH

/-- Multi-head input-only block tie — recompute the 9 saves from `xin`, then `vit_block_tiedMHV`. -/
@[irreducible] def vitBlockTiedAtMHV {Np1 heads d mlpDim : Nat}
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
    (Mat.mul (rowSoftmax (fun i j => sdpa_scale d *
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


set_option maxHeartbeats 4000000 in
/-- **The 2-block MULTI-HEAD vector-LN ViT, tied** (the multi-head peer of `vit_net_tiedV`) — a
    validation thread for the multi-head per-block infra at depth 2 before the depth-12 promotion. -/
theorem vit_net_tiedMHV2 {N heads d mlpDim nClasses : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ) (γF : Vec (heads * d)) (Wcls : Mat (heads * d) nClasses)
    (γ1₁ β1₁ γ2₁ β2₁ : Vec (heads * d)) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d) (heads * d)) (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d))
    (Wfc1₁ : Mat (heads * d) mlpDim) (bfc1₁ : Vec mlpDim) (Wfc2₁ : Mat mlpDim (heads * d)) (bfc2₁ : Vec (heads * d))
    (γ1₂ β1₂ γ2₂ β2₂ : Vec (heads * d)) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d) (heads * d)) (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d))
    (Wfc1₂ : Mat (heads * d) mlpDim) (bfc1₂ : Vec mlpDim) (Wfc2₂ : Mat mlpDim (heads * d)) (bfc2₂ : Vec (heads * d))
    (ib1 : Vec ((N + 1) * (heads * d))) (dy : Vec nClasses) (lr : ℝ) :
    let ib2    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε γ1₁ β1₁ γ2₁ β2₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ ib1
    let b2out  : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ ib2
    let dyOut2 : Vec ((N + 1) * (heads * d)) := vitCotB2outV N (heads * d) nClasses ε γF Wcls b2out dy
    let dyOut1 : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ Wfc1₂ bfc1₂ Wfc2₂ ib2 dyOut2
    vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε γ1₁ β1₁ γ2₁ β2₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁
        Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ ib1 dyOut1 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂
        Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ ib2 dyOut2 lr := by
  intro ib2 b2out dyOut2 dyOut1
  refine ⟨?_, ?_⟩
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε γ1₁ β1₁ γ2₁ β2₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ ib1 dyOut1 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε γ1₂ β1₂ γ2₂ β2₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ ib2 dyOut2 lr


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
/-- **The whole depth-12 MULTI-HEAD vector-LN ViT train step, tied through the real forward.**
    The 12 transformer blocks (3 heads, d_head=64), each fed the cotangent the real multi-head
    backward chain delivers at its site, denote the certified loss-descent step. Block inputs are
    the real forward prefixes (`ib1` the embedded tokens, `ib_{k+1} = vitBlockFwdOMHV(ib_k)`); the
    block-12 output cotangent is the final-LN input-VJP of the classifier-back (`vitCotB2outV`), and
    each upstream `dyOut_k` is the attention-residual fan-in the next block hands back
    (`vitBlockCotInAtMHV`, now multi-head). The depth-12 promotion of `vit_net_tiedMHV2`; the vit
    peer of convnext's 18-block `cnx_net_tied_certified`. (Final-LN γ/β, classifier, patch embed
    reuse the §1-fold + chain certs directly; bundled in the next step, not here.) -/
theorem vit_net_tiedMHV {N heads d mlpDim nClasses : Nat}
    (xN wN bN gN epsStr lrStr cotN : String) (ε : ℝ) (γF : Vec ((heads * d))) (Wcls : Mat ((heads * d)) nClasses)
    -- block 1
    (lnG1_1 lnB1_1 lnG2_1 lnB2_1 : Vec ((heads * d))) (mWq_1 mWk_1 mWv_1 mWo_1 : Mat ((heads * d)) (heads * d)) (mbq_1 mbk_1 mbv_1 mbo_1 : Vec ((heads * d)))
    (fW1_1 : Mat ((heads * d)) mlpDim) (fb1_1 : Vec mlpDim) (fW2_1 : Mat mlpDim ((heads * d))) (fb2_1 : Vec ((heads * d)))
    -- block 2
    (lnG1_2 lnB1_2 lnG2_2 lnB2_2 : Vec ((heads * d))) (mWq_2 mWk_2 mWv_2 mWo_2 : Mat ((heads * d)) (heads * d)) (mbq_2 mbk_2 mbv_2 mbo_2 : Vec ((heads * d)))
    (fW1_2 : Mat ((heads * d)) mlpDim) (fb1_2 : Vec mlpDim) (fW2_2 : Mat mlpDim ((heads * d))) (fb2_2 : Vec ((heads * d)))
    -- block 3
    (lnG1_3 lnB1_3 lnG2_3 lnB2_3 : Vec ((heads * d))) (mWq_3 mWk_3 mWv_3 mWo_3 : Mat ((heads * d)) (heads * d)) (mbq_3 mbk_3 mbv_3 mbo_3 : Vec ((heads * d)))
    (fW1_3 : Mat ((heads * d)) mlpDim) (fb1_3 : Vec mlpDim) (fW2_3 : Mat mlpDim ((heads * d))) (fb2_3 : Vec ((heads * d)))
    -- block 4
    (lnG1_4 lnB1_4 lnG2_4 lnB2_4 : Vec ((heads * d))) (mWq_4 mWk_4 mWv_4 mWo_4 : Mat ((heads * d)) (heads * d)) (mbq_4 mbk_4 mbv_4 mbo_4 : Vec ((heads * d)))
    (fW1_4 : Mat ((heads * d)) mlpDim) (fb1_4 : Vec mlpDim) (fW2_4 : Mat mlpDim ((heads * d))) (fb2_4 : Vec ((heads * d)))
    -- block 5
    (lnG1_5 lnB1_5 lnG2_5 lnB2_5 : Vec ((heads * d))) (mWq_5 mWk_5 mWv_5 mWo_5 : Mat ((heads * d)) (heads * d)) (mbq_5 mbk_5 mbv_5 mbo_5 : Vec ((heads * d)))
    (fW1_5 : Mat ((heads * d)) mlpDim) (fb1_5 : Vec mlpDim) (fW2_5 : Mat mlpDim ((heads * d))) (fb2_5 : Vec ((heads * d)))
    -- block 6
    (lnG1_6 lnB1_6 lnG2_6 lnB2_6 : Vec ((heads * d))) (mWq_6 mWk_6 mWv_6 mWo_6 : Mat ((heads * d)) (heads * d)) (mbq_6 mbk_6 mbv_6 mbo_6 : Vec ((heads * d)))
    (fW1_6 : Mat ((heads * d)) mlpDim) (fb1_6 : Vec mlpDim) (fW2_6 : Mat mlpDim ((heads * d))) (fb2_6 : Vec ((heads * d)))
    -- block 7
    (lnG1_7 lnB1_7 lnG2_7 lnB2_7 : Vec ((heads * d))) (mWq_7 mWk_7 mWv_7 mWo_7 : Mat ((heads * d)) (heads * d)) (mbq_7 mbk_7 mbv_7 mbo_7 : Vec ((heads * d)))
    (fW1_7 : Mat ((heads * d)) mlpDim) (fb1_7 : Vec mlpDim) (fW2_7 : Mat mlpDim ((heads * d))) (fb2_7 : Vec ((heads * d)))
    -- block 8
    (lnG1_8 lnB1_8 lnG2_8 lnB2_8 : Vec ((heads * d))) (mWq_8 mWk_8 mWv_8 mWo_8 : Mat ((heads * d)) (heads * d)) (mbq_8 mbk_8 mbv_8 mbo_8 : Vec ((heads * d)))
    (fW1_8 : Mat ((heads * d)) mlpDim) (fb1_8 : Vec mlpDim) (fW2_8 : Mat mlpDim ((heads * d))) (fb2_8 : Vec ((heads * d)))
    -- block 9
    (lnG1_9 lnB1_9 lnG2_9 lnB2_9 : Vec ((heads * d))) (mWq_9 mWk_9 mWv_9 mWo_9 : Mat ((heads * d)) (heads * d)) (mbq_9 mbk_9 mbv_9 mbo_9 : Vec ((heads * d)))
    (fW1_9 : Mat ((heads * d)) mlpDim) (fb1_9 : Vec mlpDim) (fW2_9 : Mat mlpDim ((heads * d))) (fb2_9 : Vec ((heads * d)))
    -- block 10
    (lnG1_10 lnB1_10 lnG2_10 lnB2_10 : Vec ((heads * d))) (mWq_10 mWk_10 mWv_10 mWo_10 : Mat ((heads * d)) (heads * d)) (mbq_10 mbk_10 mbv_10 mbo_10 : Vec ((heads * d)))
    (fW1_10 : Mat ((heads * d)) mlpDim) (fb1_10 : Vec mlpDim) (fW2_10 : Mat mlpDim ((heads * d))) (fb2_10 : Vec ((heads * d)))
    -- block 11
    (lnG1_11 lnB1_11 lnG2_11 lnB2_11 : Vec ((heads * d))) (mWq_11 mWk_11 mWv_11 mWo_11 : Mat ((heads * d)) (heads * d)) (mbq_11 mbk_11 mbv_11 mbo_11 : Vec ((heads * d)))
    (fW1_11 : Mat ((heads * d)) mlpDim) (fb1_11 : Vec mlpDim) (fW2_11 : Mat mlpDim ((heads * d))) (fb2_11 : Vec ((heads * d)))
    -- block 12
    (lnG1_12 lnB1_12 lnG2_12 lnB2_12 : Vec ((heads * d))) (mWq_12 mWk_12 mWv_12 mWo_12 : Mat ((heads * d)) (heads * d)) (mbq_12 mbk_12 mbv_12 mbo_12 : Vec ((heads * d)))
    (fW1_12 : Mat ((heads * d)) mlpDim) (fb1_12 : Vec mlpDim) (fW2_12 : Mat mlpDim ((heads * d))) (fb2_12 : Vec ((heads * d)))
    (ib1 : Vec ((N + 1) * (heads * d))) (dy : Vec nClasses) (lr : ℝ) :
    let ib2    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1
    let ib3    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2
    let ib4    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3
    let ib5    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4
    let ib6    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5
    let ib7    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6
    let ib8    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7
    let ib9    : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8
    let ib10   : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9
    let ib11   : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10
    let ib12   : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11
    let b12out : Vec ((N + 1) * (heads * d)) := vitBlockFwdOMHV ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12
    let dy12   : Vec ((N + 1) * (heads * d)) := vitCotB2outV N ((heads * d)) nClasses ε γF Wcls b12out dy
    let dy11   : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 ib12 dy12
    let dy10   : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 ib11 dy11
    let dy9    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 ib10 dy10
    let dy8    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 ib9 dy9
    let dy7    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 ib8 dy8
    let dy6    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 ib7 dy7
    let dy5    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 ib6 dy6
    let dy4    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 ib5 dy5
    let dy3    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 ib4 dy4
    let dy2    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 ib3 dy3
    let dy1    : Vec ((N + 1) * (heads * d)) := vitBlockCotInAtMHV ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 ib2 dy2
    vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9 dy9 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10 dy10 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11 dy11 lr
  ∧ vitBlockTiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12 lr := by
  intro ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9 dy9 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10 dy10 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11 dy11 lr
  · exact vit_block_tiedAtMHV xN wN bN gN epsStr lrStr cotN ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12 lr


/-! ## Task 3 — the non-block param bundle + the all-200-params capstone (committed ViT-Tiny config)

`vitFinalLNTied`/`vitHeadTied`/`vitEmbedTied` bundle the final vector-LN γ/β, the classifier Wcls/bcls,
and the patch-embed wConv/bConv/cls/pos as `den = certified` at their chain cotangents — each a direct
delegation to the §1-fold generics (`ViTPoC.*_den`), with the cls op (`denseBiasSgdB` N=1) folded by
`vit_cls_den` (its row-0 batch slice IS `cls_token_grad`, closed by `vit_render_cls_certified`). Then
`vit_net_tied_certified` threads the REAL forward + loss-driven backward and bundles all 200 params. -/

-- cls-param op den at the committed ViT-Tiny dims
set_option linter.unusedSimpArgs false in
theorem vit_cls_den (clsN lrStr cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (lr : ℝ) (i : Fin 192) :
    den (SHlo.denseBiasSgdB (N := 1) (c := 192) clsN lrStr cls lr
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
      = cls i - lr * ∑ j : Fin (197 * 192),
          pdiv (fun cl : Vec 192 =>
                  patchEmbed_flat 3 224 224 16 196 192 Wc bc cl pos img) cls i j * dyEmbed j := by
  have hstep : den (SHlo.denseBiasSgdB (N := 1) (c := 192) clsN lrStr cls lr
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
      = cls i - lr * cls_token_grad dyEmbed i := by
    simp only [den, batchSlice, clsSliceFlat, cls_token_grad]; rw [Fin.sum_univ_one]; rfl
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
  refine ⟨?_, ?_⟩
  · intro k; exact ViTPoC.veclnGammaSgd_den (N := 197) gN xN epsStr lrStr cotN ε βF b12out γF (vitCotFl 196 192 10 Wcls g) lr k
  · intro i; exact ViTPoC.rowDenseBiasSgd_den_lnbeta (N := 197) bN lrStr cotN ε γF (Mat.unflatten b12out) βF (vitCotFl 196 192 10 Wcls g) lr i

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
  · intro i j; exact ViTPoC.headW_den aN wN lrStr cotN hn Wcls bcls g lr i j
  · intro i;   exact ViTPoC.headB_den bN lrStr cotN Wcls hn bcls g lr i

/-- Patch embed wConv/bConv/cls/pos tied at the embed-output cot `dyEmbed`. -/
def vitEmbedTied (wN xN bN clsN pN lrStr cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (lr : ℝ) : Prop :=
  (∀ (d : Fin 192) (c : Fin 3) (kh kw : Fin 16),
      den (SHlo.patchEmbedWeightSgd wN xN lrStr img Wc lr (.operand cotN dyEmbed))
          (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
        = Wc d c kh kw - lr * ∑ o : Fin (197 * 192),
            pdiv (fun v : Vec (192 * 3 * 16 * 16) =>
                    patchEmbed_flat 3 224 224 16 196 192 (Kernel4.unflatten v) bc cls pos img)
              (Kernel4.flatten Wc)
              (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o * dyEmbed o)
  ∧ (∀ i : Fin 192,
      den (SHlo.patchEmbedBiasSgd bN lrStr bc lr (.operand cotN dyEmbed)) i
        = bc i - lr * ∑ o : Fin (197 * 192),
            pdiv (fun b' : Vec 192 => patchEmbed_flat 3 224 224 16 196 192 Wc b' cls pos img) bc i o * dyEmbed o)
  ∧ (∀ i : Fin 192,
      den (SHlo.denseBiasSgdB (N := 1) (c := 192) clsN lrStr cls lr (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
        = cls i - lr * ∑ j : Fin (197 * 192),
            pdiv (fun cl : Vec 192 => patchEmbed_flat 3 224 224 16 196 192 Wc bc cl pos img) cls i j * dyEmbed j)
  ∧ (∀ i : Fin (197 * 192),
      den (SHlo.posEmbedSgd pN lrStr pos lr (.operand cotN dyEmbed)) i
        = Mat.flatten pos i - lr * ∑ j : Fin (197 * 192),
            pdiv (fun p : Vec (197 * 192) => patchEmbed_flat 3 224 224 16 196 192 Wc bc cls (Mat.unflatten p) img) (Mat.flatten pos) i j * dyEmbed j)

theorem vit_embed_tied (wN xN bN clsN pN lrStr cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (lr : ℝ) :
    vitEmbedTied wN xN bN clsN pN lrStr cotN Wc bc cls pos img dyEmbed lr := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro d c kh kw; exact ViTPoC.patchEmbedWeightSgd_den wN xN lrStr cotN bc cls pos img Wc dyEmbed lr d c kh kw
  · intro i; exact ViTPoC.patchEmbedBiasSgd_den bN lrStr cotN Wc bc cls pos img dyEmbed lr i
  · intro i; exact vit_cls_den clsN lrStr cotN Wc bc cls pos img dyEmbed lr i
  · intro i; exact ViTPoC.posEmbedSgd_den pN lrStr cotN Wc bc cls pos img dyEmbed lr i


set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
/-- **The whole depth-12 MULTI-HEAD ViT-Tiny train step, tied — ALL 200 params** (the vit peer of
    convnext's `cnx_net_tied_certified`, at the committed config: 3 heads, d_head=64, D=192, N=196,
    mlpDim=768, 10 classes, 16×16 patches). The real forward `patchEmbed → 12 multi-head vector-LN
    blocks → final vector-LN → CLS-slice → dense head` and the loss-driven backward cotangent chain
    (the per-block multi-head fan-ins, the final-LN-back `vitCotB2outV`, the classifier-back `vitCotFl`,
    the embed-output cot = block-1's `vitBlockCotInAtMHV` output) are threaded, and EVERY param op
    `den`otes the certified loss-descent step: the 12 blocks' 192 params (`vitBlockTiedAtMHV`), the
    final-LN γ/β, the classifier Wcls/bcls, and the patch-embed wConv/bConv/cls/pos — 200/200, the
    FIRST net with zero param gaps (vit has the patch-weight cert). 3-axiom clean. -/
theorem vit_net_tied_certified
    (xN wN bN gN aN clsN pN epsStr lrStr cotN : String) (ε : ℝ)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (γF βF : Vec 192) (Wcls : Mat 192 10) (bcls : Vec 10)
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
    (img : Vec (3 * 224 * 224)) (label : Fin 10) (lr : ℝ) :
    let ib1    : Vec (197 * 192) := patchEmbed_flat 3 224 224 16 196 192 Wc bc cls pos img
    let ib2    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1
    let ib3    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2
    let ib4    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3
    let ib5    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4
    let ib6    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5
    let ib7    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6
    let ib8    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7
    let ib9    : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8
    let ib10   : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9
    let ib11   : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10
    let ib12   : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11
    let b12out : Vec (197 * 192) := vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12
    let fl     : Vec (197 * 192) := Mat.flatten (fun r => layerNormVec 192 ε γF βF (Mat.unflatten b12out r))
    let hn     : Vec 192 := clsSliceFlat 196 192 fl
    let logits : Vec 10 := dense Wcls bcls hn
    let g      : Vec 10 := fun c => softmax 10 logits c - oneHot 10 label c
    let dy12   : Vec (197 * 192) := vitCotB2outV 196 192 10 ε γF Wcls b12out g
    let dy11   : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 ib12 dy12
    let dy10   : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 ib11 dy11
    let dy9    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 ib10 dy10
    let dy8    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 ib9 dy9
    let dy7    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 ib8 dy8
    let dy6    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 ib7 dy7
    let dy5    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 ib6 dy6
    let dy4    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 ib5 dy5
    let dy3    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 ib4 dy4
    let dy2    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 ib3 dy3
    let dy1    : Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 ib2 dy2
    let dyEmbed: Vec (197 * 192) := vitBlockCotInAtMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 ib1 dy1
    vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9 dy9 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10 dy10 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11 dy11 lr
  ∧ vitBlockTiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12 lr
  ∧ vitFinalLNTied gN xN bN epsStr lrStr cotN ε γF βF Wcls b12out g lr
  ∧ vitHeadTied aN wN bN lrStr cotN hn Wcls bcls g lr
  ∧ vitEmbedTied wN xN bN clsN pN lrStr cotN Wc bc cls pos img dyEmbed lr := by
  intro ib1 ib2 ib3 ib4 ib5 ib6 ib7 ib8 ib9 ib10 ib11 ib12 b12out fl hn logits g dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dyEmbed
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9 dy9 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10 dy10 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11 dy11 lr
  · exact vit_block_tiedAtMHV (Np1 := 197) (heads := 3) (d := 64) xN wN bN gN epsStr lrStr cotN ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12 lr
  · exact vit_finalLN_tied gN xN bN epsStr lrStr cotN ε γF βF Wcls b12out g lr
  · exact vit_head_tied aN wN bN lrStr cotN hn Wcls bcls g lr
  · exact vit_embed_tied wN xN bN clsN pN lrStr cotN Wc bc cls pos img dyEmbed lr

end Proofs.ViTTiePoC
