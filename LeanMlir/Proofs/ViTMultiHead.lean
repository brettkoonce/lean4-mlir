import LeanMlir.Proofs.ViTVecLN

/-!
# ViT scaling pass — multi-head rendering + faithfulness

The representative ViT close (Items A–D) rendered at heads = 1, where MHSA
collapses to three matmuls + a row-softmax (`mhsa_layer_one_head`). The MATH
was always general in `heads` (`mhsa_has_vjp_mat`, `transformerBlock(V)_has_vjp_mat`,
`vitForward2(V)_has_vjp`); what was missing is RENDERING + faithfulness at
heads > 1. This file closes that:

1. **`mhsa_layer_spelled`** — the load-bearing tie, the general-`heads`
   successor of `mhsa_layer_one_head`: `mhsa_layer N heads d` IS, per head,
   slice → matmul-spelled SDPA → pad-scatter, summed over heads. The concat
   is spelled as `Σ_h headPadMat h ∘ (per-head SDPA)` — every output column
   receives exactly one head's value, and the sum stays at the single index
   `N·(heads·d)` (no `(N·a)+(N·b)` Nat-cast trouble a binary concat would hit).

2. **`vitBlockGraphMH(V)`** over the two new ch10 tokens
   `headSliceF`/`headPadF` (+ `headsSumG`, a left-assoc `addV` fold), with
   **`vitFwdGraphMH(V)_faithful`**: the multi-head forward graphs denote the
   proven `vitForward2`/`vitForward2V` at `heads := hm1 + 1` — faithfulness
   against `mhsa_layer N heads d` DIRECTLY, not a 1-head specialization.

The graph layer is stated at `heads = hm1 + 1` (the head fold needs a first
head); the Mat-level spelling is fully general in `heads`.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 1. Per-head slice/pad at the Mat level + the spelled MHSA
-- ════════════════════════════════════════════════════════════════

/-- Head `h`'s `[N,d]` column block of an `[N,heads·d]` matrix — the
    `finProdFinEquiv (h, ·)` column gather `mhsa_layer` feeds each head's SDPA. -/
noncomputable def headSliceMat (N heads d : Nat) (h : Fin heads)
    (A : Mat N (heads * d)) : Mat N d :=
  fun r j => A r (finProdFinEquiv (h, j))

/-- Scatter an `[N,d]` head block into head `h`'s columns of a zero
    `[N,heads·d]`. Summed over heads this is `mhsa_layer`'s concat; it is also
    the slice's VJP. -/
noncomputable def headPadMat (N heads d : Nat) (h : Fin heads)
    (A : Mat N d) : Mat N (heads * d) :=
  fun r hj =>
    if (finProdFinEquiv.symm hj).1 = h then A r (finProdFinEquiv.symm hj).2 else 0

/-- **The pad-sum IS the head concat**: every column `hj` lands in exactly one
    head's block, so the sum over heads of pad-scatters reads off head
    `(symm hj).1` at column `(symm hj).2` — `mhsa_layer`'s concat indexing. -/
lemma sum_headPadMat_apply {N heads d : Nat} (G : Fin heads → Mat N d)
    (n : Fin N) (hj : Fin (heads * d)) :
    (∑ h : Fin heads, headPadMat N heads d h (G h)) n hj =
      G (finProdFinEquiv.symm hj).1 n (finProdFinEquiv.symm hj).2 := by
  rw [Finset.sum_apply, Finset.sum_apply]
  unfold headPadMat
  rw [Finset.sum_ite_eq]
  simp

/-- **MHSA at general `heads` is per-head slice → matmul-spelled SDPA →
    pad-scatter, summed over heads.** The general-`heads` successor of
    `mhsa_layer_one_head` and the load-bearing tie for the multi-head graph
    faithfulness: each head's SDPA is exactly the ch10 token spelling
    (`Q_h·K_hᵀ` → `·1/√d` → row-softmax → `P_h·V_h`) on the sliced Q/K/V, and
    the concat is the pad-sum (`sum_headPadMat_apply`). -/
lemma mhsa_layer_spelled (Np1 heads d : Nat)
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (X : Mat Np1 (heads * d)) :
    mhsa_layer Np1 heads d Wq Wk Wv Wo bq bk bv bo X =
      fun n => dense Wo bo
        ((∑ h : Fin heads, headPadMat Np1 heads d h
            (Mat.mul
              (rowSoftmax (fun i j => sdpa_scale d *
                Mat.mul (headSliceMat Np1 heads d h (fun r c => dense Wq bq (X r) c))
                  (Mat.transpose (headSliceMat Np1 heads d h
                    (fun r c => dense Wk bk (X r) c))) i j))
              (headSliceMat Np1 heads d h
                (fun r c => dense Wv bv (X r) c)))) n) := by
  funext n j
  unfold mhsa_layer sdpa sdpa_scale dense headSliceMat
  dsimp only
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  rw [sum_headPadMat_apply]

-- ════════════════════════════════════════════════════════════════
-- § 2. The spelled multi-head blocks (Mat level)
-- ════════════════════════════════════════════════════════════════

/-- The ch10 spelled pre-norm transformer block at general `heads` (Mat level) —
    the exact op sequence `vitBlockGraphMH` denotes: LN₁ → Q/K/V per-token dense
    → per head (slice → `Q_h·K_hᵀ` → `·1/√d` → row-softmax → `P_h·V_h` → pad) →
    Σ heads → output dense → +res → LN₂ → fc1 → GELU → fc2 → +res. -/
noncomputable def vitBlockSpelledMH (Np1 heads d mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (X : Mat Np1 (heads * d)) : Mat Np1 (heads * d) :=
  let ln1 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε γ1 β1 (X r)
  let Q : Mat Np1 (heads * d) := fun r => dense Wq bq (ln1 r)
  let K : Mat Np1 (heads * d) := fun r => dense Wk bk (ln1 r)
  let V : Mat Np1 (heads * d) := fun r => dense Wv bv (ln1 r)
  let att : Mat Np1 (heads * d) := ∑ h : Fin heads, headPadMat Np1 heads d h
    (Mat.mul
      (rowSoftmax (fun i j => sdpa_scale d *
        Mat.mul (headSliceMat Np1 heads d h Q)
          (Mat.transpose (headSliceMat Np1 heads d h K)) i j))
      (headSliceMat Np1 heads d h V))
  let O : Mat Np1 (heads * d) := fun r => dense Wo bo (att r)
  let hres : Mat Np1 (heads * d) := fun r s => X r s + O r s
  let ln2 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε γ2 β2 (hres r)
  let m1 : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  let m2 : Mat Np1 (heads * d) := fun r => dense Wfc2 bfc2 (g r)
  fun r s => hres r s + m2 r s

/-- **The spelled multi-head block IS `transformerBlock` at general `heads`.**
    The sublayer/residual structure matches definitionally once
    `mhsa_layer_spelled` turns the per-head plumbing into the pad-sum. -/
lemma vitBlockSpelledMH_eq (Np1 heads d mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (X : Mat Np1 (heads * d)) :
    vitBlockSpelledMH Np1 heads d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X =
      transformerBlock Np1 heads d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X := by
  unfold transformerBlock transformerMlpSublayer transformerAttnSublayer
         transformerMlp biPathMat
  simp only [Function.comp_apply]
  rw [mhsa_layer_spelled]
  rfl

/-- The spelled multi-head block at vector-[D] LN — each LN site decomposed as
    the graph (and `ViTRender`) emit it: pure normalize (scalar-LN at 1,0) →
    per-channel scale → per-channel bias; attention as `vitBlockSpelledMH`. -/
noncomputable def vitBlockSpelledMHV (Np1 heads d mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : Vec (heads * d))
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (γ2 β2 : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (X : Mat Np1 (heads * d)) : Mat Np1 (heads * d) :=
  let xh1 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε 1 0 (X r)
  let sc1 : Mat Np1 (heads * d) := fun r => layerScale γ1 (xh1 r)
  let ln1 : Mat Np1 (heads * d) := fun r k => sc1 r k + β1 k
  let Q : Mat Np1 (heads * d) := fun r => dense Wq bq (ln1 r)
  let K : Mat Np1 (heads * d) := fun r => dense Wk bk (ln1 r)
  let V : Mat Np1 (heads * d) := fun r => dense Wv bv (ln1 r)
  let att : Mat Np1 (heads * d) := ∑ h : Fin heads, headPadMat Np1 heads d h
    (Mat.mul
      (rowSoftmax (fun i j => sdpa_scale d *
        Mat.mul (headSliceMat Np1 heads d h Q)
          (Mat.transpose (headSliceMat Np1 heads d h K)) i j))
      (headSliceMat Np1 heads d h V))
  let O : Mat Np1 (heads * d) := fun r => dense Wo bo (att r)
  let hres : Mat Np1 (heads * d) := fun r s => X r s + O r s
  let xh2 : Mat Np1 (heads * d) := fun r => layerNormForward (heads * d) ε 1 0 (hres r)
  let sc2 : Mat Np1 (heads * d) := fun r => layerScale γ2 (xh2 r)
  let ln2 : Mat Np1 (heads * d) := fun r k => sc2 r k + β2 k
  let m1 : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  let m2 : Mat Np1 (heads * d) := fun r => dense Wfc2 bfc2 (g r)
  fun r s => hres r s + m2 r s

/-- **The spelled multi-head vector-LN block IS `transformerBlockV` at general
    `heads`** — the three-stage LN decomposition collapses to `layerNormVec`
    definitionally; the per-head plumbing via `mhsa_layer_spelled`. -/
lemma vitBlockSpelledMHV_eq (Np1 heads d mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : Vec (heads * d))
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (γ2 β2 : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d)) (bfc2 : Vec (heads * d))
    (X : Mat Np1 (heads * d)) :
    vitBlockSpelledMHV Np1 heads d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X =
      transformerBlockV Np1 heads d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X := by
  unfold transformerBlockV transformerMlpSublayerV transformerAttnSublayerV
         transformerMlp biPathMat vitBlockSpelledMHV
  simp only [Function.comp_apply]
  rw [mhsa_layer_spelled]
  rfl

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 3. Graph-level: the head fold + flat ↔ Mat bridges
-- ════════════════════════════════════════════════════════════════

/-- Left-assoc `addV` fold of one graph per head — the token-level Σ over
    heads (`heads = hm1 + 1`: the fold needs a first head). -/
def headsSumG {n : Nat} : {hm1 : Nat} → (Fin (hm1 + 1) → SHlo n) → SHlo n
  | 0, f => f 0
  | hm1 + 1, f => .addV (headsSumG (fun i => f i.castSucc)) (f (Fin.last (hm1 + 1)))

/-- The head fold denotes the pointwise sum over heads. -/
lemma den_headsSumG {n : Nat} {hm1 : Nat} (f : Fin (hm1 + 1) → SHlo n) :
    den (headsSumG f) = fun j => ∑ h, den (f h) j := by
  induction hm1 with
  | zero =>
      funext j
      simp [headsSumG]
  | succ k ih =>
      funext j
      show den (headsSumG (fun i => f i.castSucc)) j + den (f (Fin.last (k + 1))) j =
        ∑ h, den (f h) j
      rw [ih (fun i => f i.castSucc), Fin.sum_univ_castSucc]

/-- Per-head slice commutation bridge. -/
lemma headSliceFlat_flat {N heads d : Nat} (h : Fin heads) (A : Mat N (heads * d)) :
    headSliceFlat N heads d h (Mat.flatten A) =
      Mat.flatten (headSliceMat N heads d h A) := by
  unfold headSliceFlat headSliceMat
  rw [Mat.unflatten_flatten]

/-- Per-head pad commutation bridge. -/
lemma headPadFlat_flat {N heads d : Nat} (h : Fin heads) (A : Mat N d) :
    headPadFlat N heads d h (Mat.flatten A) =
      Mat.flatten (headPadMat N heads d h A) := by
  unfold headPadFlat headPadMat
  rw [Mat.unflatten_flatten]

/-- Pointwise sums over heads commute with `Mat.flatten`. -/
lemma flatten_sum {m n H : Nat} (G : Fin H → Mat m n) :
    (fun j => ∑ h : Fin H, Mat.flatten (G h) j) = Mat.flatten (∑ h, G h) := by
  funext j
  unfold Mat.flatten
  simp [Finset.sum_apply]

-- ════════════════════════════════════════════════════════════════
-- § 4. The multi-head block graphs + denotations
-- ════════════════════════════════════════════════════════════════

/-- One spelled pre-norm transformer block over the ch10 tokens at
    `heads = hm1 + 1`: `lnRowF` → Q/K/V `denseRowF` → per head
    (`headSliceF` → `matmulF`(Q_h, `transposeF` K_h) → `scaleF` →
    `softmaxRowF` → `matmulF`(P_h, V_h) → `headPadF`) → `headsSumG` →
    output `denseRowF` → `addV` residual → `lnRowF` → fc1 → `geluF` → fc2 →
    `addV` residual. -/
def vitBlockGraphMH {Np1 hm1 d mlpDim : Nat} (pfx epsStr sStr : String)
    (ε s : ℝ) (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq bk bv bo : Vec ((hm1 + 1) * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat ((hm1 + 1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1 + 1) * d)) (bfc2 : Vec ((hm1 + 1) * d))
    (x : SHlo (Np1 * ((hm1 + 1) * d))) : SHlo (Np1 * ((hm1 + 1) * d)) :=
  let ln1 := SHlo.lnRowF s!"%{pfx}g1" s!"%{pfx}bt1" epsStr ε γ1 β1 x
  let q := SHlo.denseRowF s!"%{pfx}Wq" s!"%{pfx}bq" Wq bq ln1
  let k := SHlo.denseRowF s!"%{pfx}Wk" s!"%{pfx}bk" Wk bk ln1
  let v := SHlo.denseRowF s!"%{pfx}Wv" s!"%{pfx}bv" Wv bv ln1
  let att := headsSumG (fun h : Fin (hm1 + 1) => SHlo.headPadF h
    (SHlo.matmulF
      (SHlo.softmaxRowF (SHlo.scaleF sStr s
        (SHlo.matmulF (SHlo.headSliceF h q)
          (SHlo.transposeF (SHlo.headSliceF h k)))))
      (SHlo.headSliceF h v)))
  let o := SHlo.denseRowF s!"%{pfx}Wo" s!"%{pfx}bo" Wo bo att
  let hres := SHlo.addV x o
  let ln2 := SHlo.lnRowF s!"%{pfx}g2" s!"%{pfx}bt2" epsStr ε γ2 β2 hres
  let m2 := SHlo.denseRowF s!"%{pfx}Wfc2" s!"%{pfx}bfc2" Wfc2 bfc2
    (SHlo.geluF (SHlo.denseRowF s!"%{pfx}Wfc1" s!"%{pfx}bfc1" Wfc1 bfc1 ln2))
  SHlo.addV hres m2

/-- **Multi-head block-graph denotation** (generalized over the input's Mat
    form): the spelled token block denotes `Mat.flatten ∘ vitBlockSpelledMH ∘
    Mat.unflatten` at `heads := hm1 + 1`, `s := sdpa_scale d`. -/
private lemma vitBlockGraphMH_den_aux {Np1 hm1 d mlpDim : Nat}
    (pfx epsStr sStr : String) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq bk bv bo : Vec ((hm1 + 1) * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat ((hm1 + 1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1 + 1) * d)) (bfc2 : Vec ((hm1 + 1) * d))
    (e : SHlo (Np1 * ((hm1 + 1) * d))) (A : Mat Np1 ((hm1 + 1) * d))
    (hA : den e = Mat.flatten A) :
    den (vitBlockGraphMH pfx epsStr sStr ε (sdpa_scale d) γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 e) =
      Mat.flatten (vitBlockSpelledMH Np1 (hm1 + 1) d mlpDim ε γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 A) := by
  simp only [vitBlockGraphMH, lnRowF_faithful, denseRowF_faithful, matmulF_faithful,
             transposeF_faithful, scaleF_faithful, softmaxRowF_faithful, geluF_faithful,
             headSliceF_faithful, headPadF_faithful, den_headsSumG, den_addV, hA]
  simp only [rowLNFlat_flat, rowDenseFlat_flat, headSliceFlat_flat, transposeFlat_flat,
             matMulFlat_flat, scale_flat, rowSoftmaxFlat_flat, headPadFlat_flat,
             flatten_sum, gelu_flat, add_flat_pt]
  rfl

/-- The vector-LN multi-head block over the tokens: each LN site is
    `lnRowF`(1,0) → `rowScaleF γ` → `rowBiasF β` (the `ViTRender`
    decomposition); attention as `vitBlockGraphMH`. -/
def vitBlockGraphMHV {Np1 hm1 d mlpDim : Nat}
    (pfx epsStr sStr oneStr zeroStr : String)
    (ε s : ℝ) (γ1 β1 : Vec ((hm1 + 1) * d))
    (Wq Wk Wv Wo : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq bk bv bo : Vec ((hm1 + 1) * d))
    (γ2 β2 : Vec ((hm1 + 1) * d))
    (Wfc1 : Mat ((hm1 + 1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1 + 1) * d)) (bfc2 : Vec ((hm1 + 1) * d))
    (x : SHlo (Np1 * ((hm1 + 1) * d))) : SHlo (Np1 * ((hm1 + 1) * d)) :=
  let ln1 := SHlo.rowBiasF s!"%{pfx}bt1" β1
    (SHlo.rowScaleF s!"%{pfx}g1" γ1
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 x))
  let q := SHlo.denseRowF s!"%{pfx}Wq" s!"%{pfx}bq" Wq bq ln1
  let k := SHlo.denseRowF s!"%{pfx}Wk" s!"%{pfx}bk" Wk bk ln1
  let v := SHlo.denseRowF s!"%{pfx}Wv" s!"%{pfx}bv" Wv bv ln1
  let att := headsSumG (fun h : Fin (hm1 + 1) => SHlo.headPadF h
    (SHlo.matmulF
      (SHlo.softmaxRowF (SHlo.scaleF sStr s
        (SHlo.matmulF (SHlo.headSliceF h q)
          (SHlo.transposeF (SHlo.headSliceF h k)))))
      (SHlo.headSliceF h v)))
  let o := SHlo.denseRowF s!"%{pfx}Wo" s!"%{pfx}bo" Wo bo att
  let hres := SHlo.addV x o
  let ln2 := SHlo.rowBiasF s!"%{pfx}bt2" β2
    (SHlo.rowScaleF s!"%{pfx}g2" γ2
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 hres))
  let m2 := SHlo.denseRowF s!"%{pfx}Wfc2" s!"%{pfx}bfc2" Wfc2 bfc2
    (SHlo.geluF (SHlo.denseRowF s!"%{pfx}Wfc1" s!"%{pfx}bfc1" Wfc1 bfc1 ln2))
  SHlo.addV hres m2

/-- Multi-head vector-LN block-graph denotation. -/
private lemma vitBlockGraphMHV_den_aux {Np1 hm1 d mlpDim : Nat}
    (pfx epsStr sStr oneStr zeroStr : String) (ε : ℝ)
    (γ1 β1 : Vec ((hm1 + 1) * d))
    (Wq Wk Wv Wo : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq bk bv bo : Vec ((hm1 + 1) * d))
    (γ2 β2 : Vec ((hm1 + 1) * d))
    (Wfc1 : Mat ((hm1 + 1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1 + 1) * d)) (bfc2 : Vec ((hm1 + 1) * d))
    (e : SHlo (Np1 * ((hm1 + 1) * d))) (A : Mat Np1 ((hm1 + 1) * d))
    (hA : den e = Mat.flatten A) :
    den (vitBlockGraphMHV pfx epsStr sStr oneStr zeroStr ε (sdpa_scale d) γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 e) =
      Mat.flatten (vitBlockSpelledMHV Np1 (hm1 + 1) d mlpDim ε γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 A) := by
  simp only [vitBlockGraphMHV, lnRowF_faithful, rowScaleF_faithful, rowBiasF_faithful,
             denseRowF_faithful, matmulF_faithful, transposeF_faithful, scaleF_faithful,
             softmaxRowF_faithful, geluF_faithful, headSliceF_faithful, headPadF_faithful,
             den_headsSumG, den_addV, hA]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat, rowDenseFlat_flat,
             headSliceFlat_flat, transposeFlat_flat, matMulFlat_flat, scale_flat,
             rowSoftmaxFlat_flat, headPadFlat_flat, flatten_sum, gelu_flat, add_flat_pt]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 5. The multi-head forward graphs + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Whole **multi-head ViT forward** graph (scalar LN): patch embed → 2 spelled
    multi-head blocks (distinct params) → final per-token LN → CLS slice →
    dense head, at `heads = hm1 + 1`, `D = (hm1+1)·d`. -/
def vitFwdGraphMH {ic H W P N hm1 d mlpDim nClasses : Nat}
    (epsStr sStr : String) (ε s : ℝ)
    (Wc : Kernel4 ((hm1 + 1) * d) ic P P) (bc : Vec ((hm1 + 1) * d))
    (cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (γ1₁ β1₁ : ℝ) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₁ bk₁ bv₁ bo₁ : Vec ((hm1 + 1) * d))
    (γ2₁ β2₁ : ℝ) (Wfc1₁ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₁ : Vec ((hm1 + 1) * d))
    (γ1₂ β1₂ : ℝ) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₂ bk₂ bv₂ bo₂ : Vec ((hm1 + 1) * d))
    (γ2₂ β2₂ : ℝ) (Wfc1₂ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₂ : Vec ((hm1 + 1) * d))
    (γF βF : ℝ) (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) : SHlo nClasses :=
  let embed : SHlo ((N + 1) * ((hm1 + 1) * d)) :=
    .patchEmbedF "%Wp" "%bp" "%cls" "%pos" Wc bc cls pos (.operand "%x" x)
  let b1 := vitBlockGraphMH "b1_" epsStr sStr ε s γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ embed
  let b2 := vitBlockGraphMH "b2_" epsStr sStr ε s γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ b1
  let fl := SHlo.lnRowF "%gF" "%btF" epsStr ε γF βF b2
  denseF "%Wcls" "%bcls" Wcls bcls (.clsSliceF fl)

/-- **Multi-head ViT forward faithfulness** — the scaling-pass apex: the
    multi-head forward graph denotes the proven distinct-param 2-block
    `vitForward2` at `heads := hm1 + 1` — against `mhsa_layer` directly
    (per-block `vitBlockGraphMH_den_aux` + `vitBlockSpelledMH_eq`,
    `mhsa_layer_spelled` under the hood). -/
theorem vitFwdGraphMH_faithful
    (ic H W patchSize N hm1 d mlpDim nClasses : Nat)
    (epsStr sStr : String)
    (Wc : Kernel4 ((hm1 + 1) * d) ic patchSize patchSize) (bc : Vec ((hm1 + 1) * d))
    (cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (ε : ℝ)
    (γ1₁ β1₁ : ℝ) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₁ bk₁ bv₁ bo₁ : Vec ((hm1 + 1) * d))
    (γ2₁ β2₁ : ℝ) (Wfc1₁ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₁ : Vec ((hm1 + 1) * d))
    (γ1₂ β1₂ : ℝ) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₂ bk₂ bv₂ bo₂ : Vec ((hm1 + 1) * d))
    (γ2₂ β2₂ : ℝ) (Wfc1₂ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₂ : Vec ((hm1 + 1) * d))
    (γF βF : ℝ) (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    den (vitFwdGraphMH epsStr sStr ε (sdpa_scale d) Wc bc cls pos
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x)
      = vitForward2 ic H W patchSize N mlpDim (hm1 + 1) d nClasses Wc bc cls pos ε
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x := by
  have h0 : den (SHlo.patchEmbedF (P := patchSize) "%Wp" "%bp" "%cls" "%pos"
        Wc bc cls pos (.operand "%x" x))
      = Mat.flatten (Mat.unflatten
          (patchEmbed_flat ic H W patchSize N ((hm1 + 1) * d) Wc bc cls pos x)) := by
    simp only [patchEmbedF_faithful, den_operand]
    rw [Mat.flatten_unflatten]
    rfl
  have h1 := vitBlockGraphMH_den_aux "b1_" epsStr sStr ε γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
    _ _ h0
  have h2 := vitBlockGraphMH_den_aux "b2_" epsStr sStr ε γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
    _ _ h1
  simp only [vitFwdGraphMH, denseF_faithful, clsSliceF_faithful, lnRowF_faithful, h2]
  simp only [rowLNFlat_flat, vitBlockSpelledMH_eq]
  unfold vitForward2 classifier_flat
  simp only [Function.comp_apply, Mat.unflatten_flatten]
  rfl

/-- Whole **multi-head vector-LN ViT forward** graph: patch embed → 2 spelled
    multi-head vector-LN blocks (distinct params) → final vector-LN (the same
    three-token decomposition) → CLS slice → dense head. The production
    `ViTRender` shape at the graph layer. -/
def vitFwdGraphMHV {ic H W P N hm1 d mlpDim nClasses : Nat}
    (epsStr sStr oneStr zeroStr : String) (ε s : ℝ)
    (Wc : Kernel4 ((hm1 + 1) * d) ic P P) (bc : Vec ((hm1 + 1) * d))
    (cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (γ1₁ β1₁ : Vec ((hm1 + 1) * d))
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₁ bk₁ bv₁ bo₁ : Vec ((hm1 + 1) * d))
    (γ2₁ β2₁ : Vec ((hm1 + 1) * d))
    (Wfc1₁ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₁ : Vec ((hm1 + 1) * d))
    (γ1₂ β1₂ : Vec ((hm1 + 1) * d))
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₂ bk₂ bv₂ bo₂ : Vec ((hm1 + 1) * d))
    (γ2₂ β2₂ : Vec ((hm1 + 1) * d))
    (Wfc1₂ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₂ : Vec ((hm1 + 1) * d))
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) : SHlo nClasses :=
  let embed : SHlo ((N + 1) * ((hm1 + 1) * d)) :=
    .patchEmbedF "%Wp" "%bp" "%cls" "%pos" Wc bc cls pos (.operand "%x" x)
  let b1 := vitBlockGraphMHV "b1_" epsStr sStr oneStr zeroStr ε s γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ embed
  let b2 := vitBlockGraphMHV "b2_" epsStr sStr oneStr zeroStr ε s γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ b1
  let fl := SHlo.rowBiasF "%btF" βF
    (SHlo.rowScaleF "%gF" γF
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 b2))
  denseF "%Wcls" "%bcls" Wcls bcls (.clsSliceF fl)

/-- **Multi-head vector-LN ViT forward faithfulness** — the multi-head graph
    at the production LN form denotes `vitForward2V` at `heads := hm1 + 1`. -/
theorem vitFwdGraphMHV_faithful
    (ic H W patchSize N hm1 d mlpDim nClasses : Nat)
    (epsStr sStr oneStr zeroStr : String)
    (Wc : Kernel4 ((hm1 + 1) * d) ic patchSize patchSize) (bc : Vec ((hm1 + 1) * d))
    (cls : Vec ((hm1 + 1) * d)) (pos : Mat (N + 1) ((hm1 + 1) * d))
    (ε : ℝ)
    (γ1₁ β1₁ : Vec ((hm1 + 1) * d))
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₁ bk₁ bv₁ bo₁ : Vec ((hm1 + 1) * d))
    (γ2₁ β2₁ : Vec ((hm1 + 1) * d))
    (Wfc1₁ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₁ : Vec ((hm1 + 1) * d))
    (γ1₂ β1₂ : Vec ((hm1 + 1) * d))
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq₂ bk₂ bv₂ bo₂ : Vec ((hm1 + 1) * d))
    (γ2₂ β2₂ : Vec ((hm1 + 1) * d))
    (Wfc1₂ : Mat ((hm1 + 1) * d) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim ((hm1 + 1) * d)) (bfc2₂ : Vec ((hm1 + 1) * d))
    (γF βF : Vec ((hm1 + 1) * d))
    (Wcls : Mat ((hm1 + 1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    den (vitFwdGraphMHV epsStr sStr oneStr zeroStr ε (sdpa_scale d) Wc bc cls pos
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x)
      = vitForward2V ic H W patchSize N mlpDim (hm1 + 1) d nClasses Wc bc cls pos ε
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x := by
  have h0 : den (SHlo.patchEmbedF (P := patchSize) "%Wp" "%bp" "%cls" "%pos"
        Wc bc cls pos (.operand "%x" x))
      = Mat.flatten (Mat.unflatten
          (patchEmbed_flat ic H W patchSize N ((hm1 + 1) * d) Wc bc cls pos x)) := by
    simp only [patchEmbedF_faithful, den_operand]
    rw [Mat.flatten_unflatten]
    rfl
  have h1 := vitBlockGraphMHV_den_aux "b1_" epsStr sStr oneStr zeroStr ε γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
    _ _ h0
  have h2 := vitBlockGraphMHV_den_aux "b2_" epsStr sStr oneStr zeroStr ε γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
    _ _ h1
  simp only [vitFwdGraphMHV, denseF_faithful, clsSliceF_faithful, rowBiasF_faithful,
             rowScaleF_faithful, lnRowF_faithful, h2]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat, vitBlockSpelledMHV_eq]
  unfold vitForward2V classifier_flat
  simp only [Function.comp_apply, Mat.unflatten_flatten]
  rfl

end Proofs.StableHLO
