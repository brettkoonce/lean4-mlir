import LeanMlir.Proofs.Nets.ViT.ViTFwdGraph

/-! # ViT — the attention-block cotangent chain

`TokenParamGrad.lean` certifies each ViT param output for *any* cotangent `dy` at that
site's output. This file defines the cotangent the **actual backward chain delivers** at each
site — the ViT analogue of `ConvNeXtChainClose` — and the step ties (`ViTStepTie`,
`ViTStepTieGB`) feed those cotangents to those bridges at the real forward. The definitions are
per example; everything in a ViT is per-example separable, so the batched tie lifts them with
`batchMapAux`.

The chain composes the *rendered* backward denotations — exactly the render's
backward tokens: per-token dense input-VJP (`denseRowBack`'s denotation
`rowDenseBackFlat` = rowwise `dX = W·dy`), the GELU mask (`dy ⊙ geluScalarDeriv` at the
saved pre-GELU), the rowwise scalar-LN input-VJP (`lnRowBack`'s denotation
`rowLNBackFlat` = rowwise `bnGradInput`), the row-softmax backward (`softmaxRowBack`'s
denotation `rowSoftmaxBackFlat`, recomputing the weights from the saved pre-softmax
scores), and the **SDPA matmuls spelled with the forward `matmulF`/`transposeF` on
cotangents** (`matMulFlat`/`transposeFlat`):

  block: bout = h + fc2(gelu(fc1(LN₂ h))),  h = x + Wo·SDPA(Wq·LN₁x, Wk·LN₁x, Wv·LN₁x)

The MLP residual passes `dyOut` straight to the fc2 output AND down the LN₂ branch
(the cotangent at `h` is `dyOut + LN₂-back(…)`); the attention residual likewise (the block
input's is `cotH + LN₁-back(…)`). The new wrinkle vs all prior nets is the **three-way
fan-in at LN₁'s output** — the Q/K/V dense-backs all read from `LN₁ x`, so their three
cotangents SUM (`vitCotLn1`), the `biPath` fan-in at width 3.

**The substantive new ties** (`vitCotD{Q,K,V}_eq_sdpaBack{Q,K,V}`): at the pinned saved
activations (pre-softmax scores = the scaled `Q·Kᵀ`, post-softmax weights =
`sdpaWeights`), the matmul-spelled chain segments ARE the proven closed forms
`sdpaBack{Q,K,V}` (Attention.lean) — `dP = dO·Vᵀ → softmax-back → ·1/√d → dQ = dS·K /
dK = dSᵀ·Q / dV = Pᵀ·dO`, flattened. So the rendered attention backward is pinned to the
audited SDPA backward suite.
-/

namespace Proofs

open scoped BigOperators
open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The cotangent the block backward chain delivers at each site output
--   (saved activations named as in the Item B render:
--    xin → ln1 → q/k/v → ss → p → att → h → ln2 → m1 → g → bout)
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the **GELU output** (= the fc2 input): the MLP residual passes the
    block cotangent `dyOut` straight to the fc2 output (`bout = h + fc2(…)`, no
    post-add activation), and fc2's input-VJP is the per-token `dX = W·dy`
    (`denseRowBack`'s denotation). -/
noncomputable def vitCotG {Np1 D mlpDim : Nat} (Wfc2 : Mat mlpDim D)
    (dyOut : Vec (Np1 * D)) : Vec (Np1 * mlpDim) :=
  rowDenseBackFlat Np1 mlpDim D Wfc2 dyOut

/-- Cotangent at the **fc1 output** (pre-GELU): the GELU mask at the saved
    pre-activation `m1` (`geluBack`'s denotation). -/
noncomputable def vitCotM1 {Np1 D mlpDim : Nat} (Wfc2 : Mat mlpDim D)
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) : Vec (Np1 * mlpDim) :=
  fun i => vitCotG Wfc2 dyOut i * geluScalarDeriv (m1 i)

/-- Cotangent at the **LN₂ output** (= the fc1 input): fc1's per-token input-VJP. -/
noncomputable def vitCotLn2 {Np1 D mlpDim : Nat} (Wfc1 : Mat D mlpDim)
    (Wfc2 : Mat mlpDim D) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) :
    Vec (Np1 * D) :=
  rowDenseBackFlat Np1 D mlpDim Wfc1 (vitCotM1 Wfc2 m1 dyOut)

/-- `dP = dAtt·Vᵀ` — the rendered `matmulF`/`transposeF` on the cotangent against the
    saved `v`. -/
noncomputable def vitCotDP {Np1 D : Nat} (v dAtt : Vec (Np1 * D)) : Vec (Np1 * Np1) :=
  matMulFlat Np1 D Np1 dAtt (transposeFlat Np1 D v)

/-- `dS` — `softmaxRowBack`'s denotation at the saved pre-softmax scaled scores `ss`. -/
noncomputable def vitCotDS {Np1 D : Nat} (ss : Vec (Np1 * Np1))
    (v dAtt : Vec (Np1 * D)) : Vec (Np1 * Np1) :=
  rowSoftmaxBackFlat Np1 Np1 ss (vitCotDP v dAtt)

/-- `dQ = (1/√d · dS)·K` against the saved `k`. -/
noncomputable def vitCotDQ {Np1 D : Nat} (d : Nat) (ss : Vec (Np1 * Np1))
    (k v dAtt : Vec (Np1 * D)) : Vec (Np1 * D) :=
  matMulFlat Np1 Np1 D (fun i => sdpaScale d * vitCotDS ss v dAtt i) k

/-- `dK = (1/√d · dS)ᵀ·Q` against the saved `q`. -/
noncomputable def vitCotDK {Np1 D : Nat} (d : Nat) (ss : Vec (Np1 * Np1))
    (q v dAtt : Vec (Np1 * D)) : Vec (Np1 * D) :=
  matMulFlat Np1 Np1 D
    (transposeFlat Np1 Np1 (fun i => sdpaScale d * vitCotDS ss v dAtt i)) q

/-- `dV = Pᵀ·dAtt` against the saved post-softmax weights `p`. -/
noncomputable def vitCotDV {Np1 D : Nat} (p : Vec (Np1 * Np1))
    (dAtt : Vec (Np1 * D)) : Vec (Np1 * D) :=
  matMulFlat Np1 Np1 D (transposeFlat Np1 Np1 p) dAtt

/-- The **three-way fan-in at LN₁'s output**: the Q/K/V dense-backs all read from
    `LN₁ x`, so their cotangents SUM — the `biPath` fan-in at width 3, the new
    structural wrinkle vs every prior net. -/
noncomputable def vitCotLn1 {Np1 D : Nat} (Wq Wk Wv : Mat D D)
    (dQ dK dV : Vec (Np1 * D)) : Vec (Np1 * D) :=
  fun i => rowDenseBackFlat Np1 D D Wq dQ i + rowDenseBackFlat Np1 D D Wk dK i +
           rowDenseBackFlat Np1 D D Wv dV i

/-- Cotangent at the **final-LN output**: classifier-back (`dotOut`'s denotation
    `Mat.mulVec Wcls`) scattered to row 0 (`clsPadF`'s denotation `clsPadFlat`). -/
noncomputable def vitCotFl (N D nClasses : Nat) (Wcls : Mat D nClasses)
    (dy : Vec nClasses) : Vec ((N + 1) * D) :=
  clsPadFlat N D (Mat.mulVec Wcls dy)

-- ════════════════════════════════════════════════════════════════
-- § The SDPA ties — the rendered matmul chain IS the proven closed backward
-- ════════════════════════════════════════════════════════════════

/-- `dP`-segment tie: the rendered `matmulF(dOut, transposeF V)` is the proven
    `sdpaDWeights V dOut = dOut·Vᵀ`, flattened. -/
theorem vitCotDP_eq_sdpaDWeights (Np1 d : Nat) (V dOut : Mat Np1 d) :
    vitCotDP (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpaDWeights V dOut) := by
  unfold vitCotDP sdpaDWeights
  rw [transposeFlat_flat, matMulFlat_flat]

/-- **`dV` tie**: at the saved post-softmax weights (`sdpaWeights Q K`), the rendered
    `matmulF(transposeF P, dOut)` IS the proven `sdpaBackV = weightsᵀ·dOut`. -/
theorem vitCotDV_eq_sdpaBackV (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDV (Mat.flatten (sdpaWeights Np1 d Q K)) (Mat.flatten dOut)
      = Mat.flatten (sdpaBackV Np1 d Q K V dOut) := by
  unfold vitCotDV sdpaBackV
  rw [transposeFlat_flat, matMulFlat_flat]

/-- `dS`-segment tie: `softmaxRowBack`'s denotation, recomputing the weights from the
    saved pre-softmax scaled scores, applied to the flattened `sdpaDWeights`, IS the
    proven `sdpaDScaled` (the per-row `pᵢ⊙(dwᵢ − ⟨pᵢ,dwᵢ⟩)` closed form). -/
theorem vitCotDS_eq_sdpaDScaled (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDS (Mat.flatten (fun i j => sdpaScale d * Mat.mul Q (Mat.transpose K) i j))
        (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpaDScaled Np1 d Q K V dOut) := by
  unfold vitCotDS
  rw [vitCotDP_eq_sdpaDWeights]
  unfold rowSoftmaxBackFlat sdpaDScaled sdpaWeights rowSoftmax
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]

/-- **`dQ` tie**: at the saved activations, the rendered
    `matmulF(scaleF(softmaxRowBack(matmulF(dOut, transposeF V))), K)` IS the proven
    `sdpaBackQ = (1/√d · softmax-back(dOut·Vᵀ))·K`. -/
theorem vitCotDQ_eq_sdpaBackQ (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDQ d (Mat.flatten (fun i j => sdpaScale d * Mat.mul Q (Mat.transpose K) i j))
        (Mat.flatten K) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpaBackQ Np1 d Q K V dOut) := by
  unfold vitCotDQ
  rw [vitCotDS_eq_sdpaDScaled, scale_flat, matMulFlat_flat]
  unfold sdpaBackQ sdpaDScores
  rfl

/-- **`dK` tie**: likewise the rendered transposed chain IS the proven
    `sdpaBackK = (1/√d · softmax-back(dOut·Vᵀ))ᵀ·Q`. -/
theorem vitCotDK_eq_sdpaBackK (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDK d (Mat.flatten (fun i j => sdpaScale d * Mat.mul Q (Mat.transpose K) i j))
        (Mat.flatten Q) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpaBackK Np1 d Q K V dOut) := by
  unfold vitCotDK
  rw [vitCotDS_eq_sdpaDScaled, scale_flat, transposeFlat_flat, matMulFlat_flat]
  unfold sdpaBackK sdpaDScores
  rfl

end Proofs
