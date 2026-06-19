import LeanMlir.Proofs.ViTFaithfulPoC
import LeanMlir.Proofs.ViTVecLN
import LeanMlir.Proofs.ViTChainClose

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

end Proofs.ViTTiePoC
