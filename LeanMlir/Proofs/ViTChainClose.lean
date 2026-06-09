import LeanMlir.Proofs.ViTClose

/-! # ViT Item D — pinning the attention-block cotangent chain

`ViTClose.lean` (Item C) certifies each ViT param output for *any* cotangent `dy` at that
site's output. This file pins `dy` to the cotangent the **actual backward chain delivers**
— the ViT analogue of `ConvNeXtChainClose` (`planning/vit_close.md` Item D). Pure-Lean,
batch-1 — everything in a ViT is per-example separable.

The chain composes the *rendered* backward denotations — exactly the Item B render's
backward tokens: per-token dense input-VJP (`denseRowBack`'s denotation
`rowDenseBackFlat` = rowwise `dX = W·dy`), the GELU mask (`dy ⊙ geluScalarDeriv` at the
saved pre-GELU), the rowwise scalar-LN input-VJP (`lnRowBack`'s denotation
`rowLNBackFlat` = rowwise `bn_grad_input`), the row-softmax backward (`softmaxRowBack`'s
denotation `rowSoftmaxBackFlat`, recomputing the weights from the saved pre-softmax
scores), and the **SDPA matmuls spelled with the forward `matmulF`/`transposeF` on
cotangents** (`matMulFlat`/`transposeFlat`):

  block: bout = h + fc2(gelu(fc1(LN₂ h))),  h = x + Wo·SDPA(Wq·LN₁x, Wk·LN₁x, Wv·LN₁x)

The MLP residual passes `dyOut` straight to the fc2 output AND down the LN₂ branch
(`vitCotH = dyOut + LN₂-back(…)`); the attention residual likewise
(`vitCotXin = cotH + LN₁-back(…)`). The new wrinkle vs all prior nets is the **three-way
fan-in at LN₁'s output** — the Q/K/V dense-backs all read from `LN₁ x`, so their three
cotangents SUM (`vitCotLn1`), the `biPath` fan-in at width 3.

**The substantive new ties** (`vitCotD{Q,K,V}_eq_sdpa_back_{Q,K,V}`): at the pinned saved
activations (pre-softmax scores = the scaled `Q·Kᵀ`, post-softmax weights =
`sdpa_weights`), the matmul-spelled chain segments ARE the proven closed forms
`sdpa_back_{Q,K,V}` (Attention.lean) — `dP = dO·Vᵀ → softmax-back → ·1/√d → dQ = dS·K /
dK = dSᵀ·Q / dV = Pᵀ·dO`, flattened. So the rendered attention backward is pinned to the
audited SDPA backward suite, and each param output denotes
`θ − lr·(certified ∂/∂θ · the actual-chain cotangent)`. 3-axiom clean.
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

/-- Cotangent at the **attention-sublayer output `h`**: the MLP residual fan-in —
    `dyOut` (the skip) plus the LN₂ input-VJP (`lnRowBack`'s denotation, recomputing
    x̂/istd from the saved pre-LN₂ input `h`). -/
noncomputable def vitCotH {Np1 D mlpDim : Nat} (ε γ2 : ℝ) (Wfc1 : Mat D mlpDim)
    (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D)) (m1 : Vec (Np1 * mlpDim))
    (dyOut : Vec (Np1 * D)) : Vec (Np1 * D) :=
  fun i => dyOut i + rowLNBackFlat Np1 D ε γ2 h (vitCotLn2 Wfc1 Wfc2 m1 dyOut) i

/-- Cotangent at the **SDPA output `att`** (= the out-proj input): Wo's per-token
    input-VJP of `vitCotH`. -/
noncomputable def vitCotAtt {Np1 D mlpDim : Nat} (ε γ2 : ℝ) (Wo : Mat D D)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) : Vec (Np1 * D) :=
  rowDenseBackFlat Np1 D D Wo (vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut)

-- ── The SDPA backward segment (taking the cot at the SDPA output, `dAtt`, as input;
--    the pins below instantiate `dAtt := vitCotAtt …`) ──

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
  matMulFlat Np1 Np1 D (fun i => sdpa_scale d * vitCotDS ss v dAtt i) k

/-- `dK = (1/√d · dS)ᵀ·Q` against the saved `q`. -/
noncomputable def vitCotDK {Np1 D : Nat} (d : Nat) (ss : Vec (Np1 * Np1))
    (q v dAtt : Vec (Np1 * D)) : Vec (Np1 * D) :=
  matMulFlat Np1 Np1 D
    (transposeFlat Np1 Np1 (fun i => sdpa_scale d * vitCotDS ss v dAtt i)) q

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

/-- Cotangent at the **block input**: the attention residual fan-in — the
    LN₁ input-VJP of the three-way fan-in, plus the skip's `cotH`. This is what the
    block hands upstream (the previous block's `dyOut`; at block 1, the cotangent the
    embed params contract with). -/
noncomputable def vitCotXin {Np1 D : Nat} (ε γ1 : ℝ) (Wq Wk Wv : Mat D D)
    (xin : Vec (Np1 * D)) (dQ dK dV cotH : Vec (Np1 * D)) : Vec (Np1 * D) :=
  fun i => cotH i + rowLNBackFlat Np1 D ε γ1 xin (vitCotLn1 Wq Wk Wv dQ dK dV) i

/-- Cotangent at the **final-LN output**: classifier-back (`dotOut`'s denotation
    `Mat.mulVec Wcls`) scattered to row 0 (`clsPadF`'s denotation `clsPadFlat`). -/
noncomputable def vitCotFl (N D nClasses : Nat) (Wcls : Mat D nClasses)
    (dy : Vec nClasses) : Vec ((N + 1) * D) :=
  clsPadFlat N D (Mat.mulVec Wcls dy)

/-- Cotangent at **block 2's output**: the final-LN input-VJP at the saved pre-LN
    input `b2out`, of `vitCotFl`. -/
noncomputable def vitCotB2out (N D nClasses : Nat) (ε γF : ℝ)
    (Wcls : Mat D nClasses) (b2out : Vec ((N + 1) * D)) (dy : Vec nClasses) :
    Vec ((N + 1) * D) :=
  rowLNBackFlat (N + 1) D ε γF b2out (vitCotFl N D nClasses Wcls dy)

-- ════════════════════════════════════════════════════════════════
-- § The SDPA ties — the rendered matmul chain IS the proven closed backward
-- ════════════════════════════════════════════════════════════════

/-- `dP`-segment tie: the rendered `matmulF(dOut, transposeF V)` is the proven
    `sdpa_dWeights V dOut = dOut·Vᵀ`, flattened. -/
theorem vitCotDP_eq_sdpa_dWeights (Np1 d : Nat) (V dOut : Mat Np1 d) :
    vitCotDP (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpa_dWeights V dOut) := by
  unfold vitCotDP sdpa_dWeights
  rw [transposeFlat_flat, matMulFlat_flat]

/-- **`dV` tie**: at the saved post-softmax weights (`sdpa_weights Q K`), the rendered
    `matmulF(transposeF P, dOut)` IS the proven `sdpa_back_V = weightsᵀ·dOut`. -/
theorem vitCotDV_eq_sdpa_back_V (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDV (Mat.flatten (sdpa_weights Np1 d Q K)) (Mat.flatten dOut)
      = Mat.flatten (sdpa_back_V Np1 d Q K V dOut) := by
  unfold vitCotDV sdpa_back_V
  rw [transposeFlat_flat, matMulFlat_flat]

/-- `dS`-segment tie: `softmaxRowBack`'s denotation, recomputing the weights from the
    saved pre-softmax scaled scores, applied to the flattened `sdpa_dWeights`, IS the
    proven `sdpa_dScaled` (the per-row `pᵢ⊙(dwᵢ − ⟨pᵢ,dwᵢ⟩)` closed form). -/
theorem vitCotDS_eq_sdpa_dScaled (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDS (Mat.flatten (fun i j => sdpa_scale d * Mat.mul Q (Mat.transpose K) i j))
        (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpa_dScaled Np1 d Q K V dOut) := by
  unfold vitCotDS
  rw [vitCotDP_eq_sdpa_dWeights]
  unfold rowSoftmaxBackFlat sdpa_dScaled sdpa_weights rowSoftmax
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]

/-- **`dQ` tie**: at the saved activations, the rendered
    `matmulF(scaleF(softmaxRowBack(matmulF(dOut, transposeF V))), K)` IS the proven
    `sdpa_back_Q = (1/√d · softmax-back(dOut·Vᵀ))·K`. -/
theorem vitCotDQ_eq_sdpa_back_Q (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDQ d (Mat.flatten (fun i j => sdpa_scale d * Mat.mul Q (Mat.transpose K) i j))
        (Mat.flatten K) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpa_back_Q Np1 d Q K V dOut) := by
  unfold vitCotDQ
  rw [vitCotDS_eq_sdpa_dScaled, scale_flat, matMulFlat_flat]
  unfold sdpa_back_Q sdpa_dScores
  rfl

/-- **`dK` tie**: likewise the rendered transposed chain IS the proven
    `sdpa_back_K = (1/√d · softmax-back(dOut·Vᵀ))ᵀ·Q`. -/
theorem vitCotDK_eq_sdpa_back_K (Np1 d : Nat) (Q K V dOut : Mat Np1 d) :
    vitCotDK d (Mat.flatten (fun i j => sdpa_scale d * Mat.mul Q (Mat.transpose K) i j))
        (Mat.flatten Q) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (sdpa_back_K Np1 d Q K V dOut) := by
  unfold vitCotDK
  rw [vitCotDS_eq_sdpa_dScaled, scale_flat, transposeFlat_flat, matMulFlat_flat]
  unfold sdpa_back_K sdpa_dScores
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The chain-pinned closes — the Item C bridges at the actual cotangents
-- ════════════════════════════════════════════════════════════════

/-- **fc2 W, chain-certified.** The chain cotangent at the fc2 output IS the block
    cotangent `dyOut` (the MLP residual is the outermost op, no post-add activation);
    the saved GELU output `g` is the layer input. -/
theorem vit_render_Wfc2_chain_certified {Np1 D mlpDim : Nat}
    (bfc2 : Vec D) (g : Vec (Np1 * mlpDim)) (Wfc2 : Mat mlpDim D)
    (dyOut : Vec (Np1 * D)) (lr : ℝ) (i : Fin mlpDim) (j : Fin D) :
    Wfc2 i j - lr * rowDense_weight_grad (Mat.unflatten g) (Mat.unflatten dyOut) i j
      = Wfc2 i j - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun v : Vec (mlpDim * D) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bfc2
                    ((Mat.unflatten g) r)))
               (Mat.flatten Wfc2) (finProdFinEquiv (i, j)) o * dyOut o :=
  vit_render_rowdenseW_certified bfc2 (Mat.unflatten g) Wfc2 dyOut lr i j

/-- **fc2 b, chain-certified.** -/
theorem vit_render_bfc2_chain_certified {Np1 D mlpDim : Nat}
    (Wfc2 : Mat mlpDim D) (g : Vec (Np1 * mlpDim)) (bfc2 : Vec D)
    (dyOut : Vec (Np1 * D)) (lr : ℝ) (i : Fin D) :
    bfc2 i - lr * rowDense_bias_grad (Mat.unflatten dyOut) i
      = bfc2 i - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun b' : Vec D =>
                  Mat.flatten (fun r => dense Wfc2 b' ((Mat.unflatten g) r)))
               bfc2 i o * dyOut o :=
  vit_render_rowdenseb_certified Wfc2 (Mat.unflatten g) bfc2 dyOut lr i

/-- **fc1 W, chain-certified** at `vitCotM1` (fc2-back → GELU mask); the saved LN₂
    output `ln2` is the layer input. -/
theorem vit_render_Wfc1_chain_certified {Np1 D mlpDim : Nat}
    (bfc1 : Vec mlpDim) (ln2 : Vec (Np1 * D)) (Wfc1 : Mat D mlpDim)
    (Wfc2 : Mat mlpDim D) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D))
    (lr : ℝ) (i : Fin D) (j : Fin mlpDim) :
    Wfc1 i j - lr * rowDense_weight_grad (Mat.unflatten ln2)
        (Mat.unflatten (vitCotM1 Wfc2 m1 dyOut)) i j
      = Wfc1 i j - lr * ∑ o : Fin (Np1 * mlpDim),
          pdiv (fun v : Vec (D * mlpDim) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bfc1
                    ((Mat.unflatten ln2) r)))
               (Mat.flatten Wfc1) (finProdFinEquiv (i, j)) o
            * vitCotM1 Wfc2 m1 dyOut o :=
  vit_render_rowdenseW_certified bfc1 (Mat.unflatten ln2) Wfc1
    (vitCotM1 Wfc2 m1 dyOut) lr i j

/-- **fc1 b, chain-certified.** -/
theorem vit_render_bfc1_chain_certified {Np1 D mlpDim : Nat}
    (Wfc1 : Mat D mlpDim) (ln2 : Vec (Np1 * D)) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D))
    (lr : ℝ) (i : Fin mlpDim) :
    bfc1 i - lr * rowDense_bias_grad (Mat.unflatten (vitCotM1 Wfc2 m1 dyOut)) i
      = bfc1 i - lr * ∑ o : Fin (Np1 * mlpDim),
          pdiv (fun b' : Vec mlpDim =>
                  Mat.flatten (fun r => dense Wfc1 b' ((Mat.unflatten ln2) r)))
               bfc1 i o * vitCotM1 Wfc2 m1 dyOut o :=
  vit_render_rowdenseb_certified Wfc1 (Mat.unflatten ln2) bfc1
    (vitCotM1 Wfc2 m1 dyOut) lr i

/-- **LN₂ γ, chain-certified** at `vitCotLn2` (fc2-back → GELU mask → fc1-back), with
    the saved attention-sublayer output `h` as the LN input. -/
theorem vit_render_ln2gamma_chain_certified {Np1 D mlpDim : Nat}
    (ε β : ℝ) (γ : Vec 1) (h : Vec (Np1 * D)) (Wfc1 : Mat D mlpDim)
    (Wfc2 : Mat mlpDim D) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) :
    γ 0 - lr * rowLN_grad_gamma Np1 D ε (Mat.unflatten h)
        (Mat.unflatten (vitCotLn2 Wfc1 Wfc2 m1 dyOut))
      = γ 0 - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun γ' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε (γ' 0) β
                    ((Mat.unflatten h) r)))
            γ 0 o * vitCotLn2 Wfc1 Wfc2 m1 dyOut o :=
  vit_render_rowlngamma_certified Np1 D ε β γ (Mat.unflatten h)
    (vitCotLn2 Wfc1 Wfc2 m1 dyOut) lr

/-- **LN₂ β, chain-certified.** -/
theorem vit_render_ln2beta_chain_certified {Np1 D mlpDim : Nat}
    (ε γsc : ℝ) (β : Vec 1) (h : Vec (Np1 * D)) (Wfc1 : Mat D mlpDim)
    (Wfc2 : Mat mlpDim D) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) :
    β 0 - lr * rowLN_grad_beta Np1 D (Mat.unflatten (vitCotLn2 Wfc1 Wfc2 m1 dyOut))
      = β 0 - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun β' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε γsc (β' 0)
                    ((Mat.unflatten h) r)))
            β 0 o * vitCotLn2 Wfc1 Wfc2 m1 dyOut o :=
  vit_render_rowlnbeta_certified Np1 D ε γsc β (Mat.unflatten h)
    (vitCotLn2 Wfc1 Wfc2 m1 dyOut) lr

/-- **Wo, chain-certified** at `vitCotH` (the MLP-sublayer residual fan-in), with the
    saved SDPA output `att` as the layer input. -/
theorem vit_render_Wo_chain_certified {Np1 D mlpDim : Nat}
    (bo : Vec D) (att : Vec (Np1 * D)) (Wo : Mat D D) (ε γ2 : ℝ)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) (i j : Fin D) :
    Wo i j - lr * rowDense_weight_grad (Mat.unflatten att)
        (Mat.unflatten (vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut)) i j
      = Wo i j - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun v : Vec (D * D) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bo
                    ((Mat.unflatten att) r)))
               (Mat.flatten Wo) (finProdFinEquiv (i, j)) o
            * vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut o :=
  vit_render_rowdenseW_certified bo (Mat.unflatten att) Wo
    (vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut) lr i j

/-- **bo, chain-certified.** -/
theorem vit_render_bo_chain_certified {Np1 D mlpDim : Nat}
    (Wo : Mat D D) (att : Vec (Np1 * D)) (bo : Vec D) (ε γ2 : ℝ)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) (i : Fin D) :
    bo i - lr * rowDense_bias_grad
        (Mat.unflatten (vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut)) i
      = bo i - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun b' : Vec D =>
                  Mat.flatten (fun r => dense Wo b' ((Mat.unflatten att) r)))
               bo i o * vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut o :=
  vit_render_rowdenseb_certified Wo (Mat.unflatten att) bo
    (vitCotH ε γ2 Wfc1 Wfc2 h m1 dyOut) lr i

/-- **Wq, chain-certified** at `vitCotDQ` of the full chain (out-proj back → SDPA
    backward at the saved activations), with the saved LN₁ output as the layer input. -/
theorem vit_render_Wq_chain_certified {Np1 D mlpDim : Nat} (d : Nat)
    (bq : Vec D) (ln1 : Vec (Np1 * D)) (Wq : Mat D D)
    (ss : Vec (Np1 * Np1)) (k v : Vec (Np1 * D)) (ε γ2 : ℝ) (Wo : Mat D D)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) (i j : Fin D) :
    Wq i j - lr * rowDense_weight_grad (Mat.unflatten ln1)
        (Mat.unflatten (vitCotDQ d ss k v
          (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut))) i j
      = Wq i j - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun w : Vec (D * D) =>
                  Mat.flatten (fun r => dense (Mat.unflatten w) bq
                    ((Mat.unflatten ln1) r)))
               (Mat.flatten Wq) (finProdFinEquiv (i, j)) o
            * vitCotDQ d ss k v (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut) o :=
  vit_render_rowdenseW_certified bq (Mat.unflatten ln1) Wq
    (vitCotDQ d ss k v (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut)) lr i j

/-- **Wk, chain-certified** at `vitCotDK`. -/
theorem vit_render_Wk_chain_certified {Np1 D mlpDim : Nat} (d : Nat)
    (bk : Vec D) (ln1 : Vec (Np1 * D)) (Wk : Mat D D)
    (ss : Vec (Np1 * Np1)) (q v : Vec (Np1 * D)) (ε γ2 : ℝ) (Wo : Mat D D)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) (i j : Fin D) :
    Wk i j - lr * rowDense_weight_grad (Mat.unflatten ln1)
        (Mat.unflatten (vitCotDK d ss q v
          (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut))) i j
      = Wk i j - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun w : Vec (D * D) =>
                  Mat.flatten (fun r => dense (Mat.unflatten w) bk
                    ((Mat.unflatten ln1) r)))
               (Mat.flatten Wk) (finProdFinEquiv (i, j)) o
            * vitCotDK d ss q v (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut) o :=
  vit_render_rowdenseW_certified bk (Mat.unflatten ln1) Wk
    (vitCotDK d ss q v (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut)) lr i j

/-- **Wv, chain-certified** at `vitCotDV`. -/
theorem vit_render_Wv_chain_certified {Np1 D mlpDim : Nat}
    (bv : Vec D) (ln1 : Vec (Np1 * D)) (Wv : Mat D D)
    (p : Vec (Np1 * Np1)) (ε γ2 : ℝ) (Wo : Mat D D)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) (lr : ℝ) (i j : Fin D) :
    Wv i j - lr * rowDense_weight_grad (Mat.unflatten ln1)
        (Mat.unflatten (vitCotDV p
          (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut))) i j
      = Wv i j - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun w : Vec (D * D) =>
                  Mat.flatten (fun r => dense (Mat.unflatten w) bv
                    ((Mat.unflatten ln1) r)))
               (Mat.flatten Wv) (finProdFinEquiv (i, j)) o
            * vitCotDV p (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut) o :=
  vit_render_rowdenseW_certified bv (Mat.unflatten ln1) Wv
    (vitCotDV p (vitCotAtt ε γ2 Wo Wfc1 Wfc2 h m1 dyOut)) lr i j

/-- **LN₁ γ, chain-certified** at `vitCotLn1` — the THREE-WAY fan-in of the Q/K/V
    dense-backs (the structural wrinkle no prior net had), with the saved block input
    `xin` as the LN input. -/
theorem vit_render_ln1gamma_chain_certified {Np1 D : Nat}
    (ε β : ℝ) (γ : Vec 1) (xin : Vec (Np1 * D)) (Wq Wk Wv : Mat D D)
    (dQ dK dV : Vec (Np1 * D)) (lr : ℝ) :
    γ 0 - lr * rowLN_grad_gamma Np1 D ε (Mat.unflatten xin)
        (Mat.unflatten (vitCotLn1 Wq Wk Wv dQ dK dV))
      = γ 0 - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun γ' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε (γ' 0) β
                    ((Mat.unflatten xin) r)))
            γ 0 o * vitCotLn1 Wq Wk Wv dQ dK dV o :=
  vit_render_rowlngamma_certified Np1 D ε β γ (Mat.unflatten xin)
    (vitCotLn1 Wq Wk Wv dQ dK dV) lr

/-- **LN₁ β, chain-certified.** -/
theorem vit_render_ln1beta_chain_certified {Np1 D : Nat}
    (ε γsc : ℝ) (β : Vec 1) (xin : Vec (Np1 * D)) (Wq Wk Wv : Mat D D)
    (dQ dK dV : Vec (Np1 * D)) (lr : ℝ) :
    β 0 - lr * rowLN_grad_beta Np1 D (Mat.unflatten (vitCotLn1 Wq Wk Wv dQ dK dV))
      = β 0 - lr * ∑ o : Fin (Np1 * D),
          pdiv (fun β' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε γsc (β' 0)
                    ((Mat.unflatten xin) r)))
            β 0 o * vitCotLn1 Wq Wk Wv dQ dK dV o :=
  vit_render_rowlnbeta_certified Np1 D ε γsc β (Mat.unflatten xin)
    (vitCotLn1 Wq Wk Wv dQ dK dV) lr

/-- **Final-LN γ, chain-certified** at `vitCotFl` (classifier-back scattered to row 0),
    with the saved block-2 output as the LN input. -/
theorem vit_render_lnFgamma_chain_certified {N D nClasses : Nat}
    (ε β : ℝ) (γ : Vec 1) (b2out : Vec ((N + 1) * D)) (Wcls : Mat D nClasses)
    (dy : Vec nClasses) (lr : ℝ) :
    γ 0 - lr * rowLN_grad_gamma (N + 1) D ε (Mat.unflatten b2out)
        (Mat.unflatten (vitCotFl N D nClasses Wcls dy))
      = γ 0 - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun γ' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε (γ' 0) β
                    ((Mat.unflatten b2out) r)))
            γ 0 o * vitCotFl N D nClasses Wcls dy o :=
  vit_render_rowlngamma_certified (N + 1) D ε β γ (Mat.unflatten b2out)
    (vitCotFl N D nClasses Wcls dy) lr

/-- **pos-embed, chain-certified** at the block-1 input cotangent `vitCotXin` — the
    cotangent the whole chain delivers at the embed output. -/
theorem vit_render_pos_chain_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (ε γ1 : ℝ) (Wq Wk Wv : Mat D D)
    (xin : Vec ((N + 1) * D)) (dQ dK dV cotH : Vec ((N + 1) * D)) (lr : ℝ)
    (i : Fin ((N + 1) * D)) :
    Mat.flatten pos i - lr * vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH i
      = Mat.flatten pos i - lr * ∑ j : Fin ((N + 1) * D),
          pdiv (fun p : Vec ((N + 1) * D) =>
                  patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
            (Mat.flatten pos) i j * vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH j :=
  vit_render_pos_certified Wc bc cls pos img
    (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) lr i

/-- **CLS token, chain-certified** — the row-0 slice of the block-1 input cotangent. -/
theorem vit_render_cls_chain_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (ε γ1 : ℝ) (Wq Wk Wv : Mat D D)
    (xin : Vec ((N + 1) * D)) (dQ dK dV cotH : Vec ((N + 1) * D)) (lr : ℝ)
    (i : Fin D) :
    cls i - lr * cls_token_grad (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) i
      = cls i - lr * ∑ j : Fin ((N + 1) * D),
          pdiv (fun cl : Vec D =>
                  patchEmbed_flat ic H W P N D Wc bc cl pos img) cls i j
            * vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH j :=
  vit_render_cls_certified Wc bc cls pos img
    (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) lr i

/-- **Patch kernel, chain-certified** — the patch-grid reduce at the block-1 input
    cotangent. -/
theorem vit_render_patchW_chain_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (ε γ1 : ℝ) (Wq Wk Wv : Mat D D)
    (xin : Vec ((N + 1) * D)) (dQ dK dV cotH : Vec ((N + 1) * D)) (lr : ℝ)
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    Wc d c kh kw - lr * patchEmbed_weight_grad ic H W P N D img
        (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) d c kh kw
      = Wc d c kh kw - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
            (Kernel4.flatten Wc)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o
            * vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH o :=
  vit_render_patchW_certified Wc bc cls pos img
    (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) lr d c kh kw

/-- **Patch bias, chain-certified.** -/
theorem vit_render_patchb_chain_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (ε γ1 : ℝ) (Wq Wk Wv : Mat D D)
    (xin : Vec ((N + 1) * D)) (dQ dK dV cotH : Vec ((N + 1) * D)) (lr : ℝ)
    (i : Fin D) :
    bc i - lr * patchEmbed_bias_grad N D
        (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) i
      = bc i - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun b' : Vec D =>
                  patchEmbed_flat ic H W P N D Wc b' cls pos img) bc i o
            * vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH o :=
  vit_render_patchb_certified Wc bc cls pos img
    (vitCotXin ε γ1 Wq Wk Wv xin dQ dK dV cotH) lr i

end Proofs
