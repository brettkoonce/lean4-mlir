import LeanMlir.Proofs.Architectures.ViTMultiHead

/-! # ViT multi-head backward — the per-head SDPA cotangents the real chain delivers

The single-head representative (`ViTChainClose`) pins the rendered attention backward to the
audited `sdpa_back_{Q,K,V}` suite at `d = D` (one head). The committed `vitTrainStepRenderV`
render is **multi-head** (`D = heads · d`, the committed ViT-Tiny: `heads = 3`, `d = 64`): the
Q/K/V denses produce all heads' projections at once (`[N, heads·d]`), then `headSliceF h` slices
head `h`'s `[N, d]` block, per-head SDPA runs at `sdpa_scale d` (d_head, NOT D), and `headPadF h`
scatters the result back, summed over heads (`mhsa_layer_spelled`).

This file is the backward symmetry of `ViTMultiHead`'s forward: the cotangent the multi-head
chain delivers at each of the Q/K/V dense **outputs** (`[N, heads·d]`). The out-proj `Wo` is a
single `[D, D]` dense unchanged by heads, so the cotangent at the SDPA output `dAtt`
(= `vitCotAttV`) is exactly the single-head one — multi-head changes ONLY the SDPA-internal
backward `dAtt → dQ/dK/dV`:

  * slice the SDPA-output cotangent per head (`headSliceFlat h dAtt` — the pad's VJP),
  * run the EXISTING per-head rendered SDPA backward at `d_head` (`vitCotD{Q,K,V} d …`),
  * scatter each head's `dQ_h`/`dK_h`/`dV_h` back into the full `[N, heads·d]` cotangent and
    sum over heads (`Σ_h headPadFlat h …` — the slice's VJP).

The **composition theorems** (`vitCotD{Q,K,V}mh_eq`) are the substantive content: the rendered
slice → per-head SDPA backward → pad chain IS `Mat.flatten (Σ_h headPadMat h (sdpa_back_{Q,K,V}))`
— the concat of the proven per-head SDPA backwards. Each reuses the single-head pin
(`vitCotD{Q,K,V}_eq_sdpa_back_{Q,K,V}`) at `d_head` head-by-head, plus the `headSliceFlat`/
`headPadFlat`/`Mat.flatten`-sum commutation bridges (`ViTMultiHead`). The downstream LN₁ fan-in is
unchanged in shape (`vitCotLn1MH = vitCotLn1` at the multi-head Q/K/V cots); the §1-fold param-SGD
`den` lemmas (`rowDenseWeightSgd_den`, …) are head-agnostic (generic in the cotangent), so only the
COTANGENT changes from the single-head `vitCotD{Q,K,V}` to `vitCotD{Q,K,V}mh`. -/

namespace Proofs

open scoped BigOperators
open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Per-head saved scores/weights (recomputed from the full saved Q/K)
-- ════════════════════════════════════════════════════════════════

/-- Head `h`'s pre-softmax scaled scores `Q_h·K_hᵀ · 1/√d`, recomputed from the full saved
    Q/K — the `ss` argument the rendered per-head `vitCotDQ`/`vitCotDK` consume (`d` = d_head). -/
noncomputable def headScoresF (Np1 heads d : Nat) (h : Fin heads)
    (q k : Vec (Np1 * (heads * d))) : Vec (Np1 * Np1) :=
  Mat.flatten (fun i j => sdpa_scale d *
    Mat.mul (headSliceMat Np1 heads d h (Mat.unflatten q))
      (Mat.transpose (headSliceMat Np1 heads d h (Mat.unflatten k))) i j)

/-- Head `h`'s post-softmax weights `softmax(Q_h·K_hᵀ · 1/√d)`, recomputed from the full saved
    Q/K — the `p` argument the rendered per-head `vitCotDV` consumes. -/
noncomputable def headWeightsF (Np1 heads d : Nat) (h : Fin heads)
    (q k : Vec (Np1 * (heads * d))) : Vec (Np1 * Np1) :=
  Mat.flatten (sdpa_weights Np1 d
    (headSliceMat Np1 heads d h (Mat.unflatten q))
    (headSliceMat Np1 heads d h (Mat.unflatten k)))

-- ════════════════════════════════════════════════════════════════
-- § The multi-head Q/K/V dense cotangents (Σ_h pad ∘ per-head SDPA back ∘ slice)
-- ════════════════════════════════════════════════════════════════

/-- **Multi-head Q-dense cotangent.** Per head: slice the SDPA-output cot (`headSliceFlat h
    dAtt`), run the rendered per-head SDPA Q-backward (`vitCotDQ d` at d_head, recomputing the
    softmax from `headScoresF`), then pad-scatter back; summed over heads. The cotangent the Q
    dense `Wq`/`bq` SGD ops contract with (vs the single-head `vitCotDQ D …`). -/
noncomputable def vitCotDQmh (Np1 heads d : Nat)
    (q k v dAtt : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  ∑ h : Fin heads, headPadFlat Np1 heads d h
    (vitCotDQ d (headScoresF Np1 heads d h q k)
      (headSliceFlat Np1 heads d h k)
      (headSliceFlat Np1 heads d h v)
      (headSliceFlat Np1 heads d h dAtt))

/-- **Multi-head K-dense cotangent.** As `vitCotDQmh`, with the per-head `vitCotDK` (K-back reads
    the saved `Q_h`). -/
noncomputable def vitCotDKmh (Np1 heads d : Nat)
    (q k v dAtt : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  ∑ h : Fin heads, headPadFlat Np1 heads d h
    (vitCotDK d (headScoresF Np1 heads d h q k)
      (headSliceFlat Np1 heads d h q)
      (headSliceFlat Np1 heads d h v)
      (headSliceFlat Np1 heads d h dAtt))

/-- **Multi-head V-dense cotangent.** Per head `Pᵀ_h · dAtt_h` (`vitCotDV` at the post-softmax
    `headWeightsF`), pad-scattered and summed. -/
noncomputable def vitCotDVmh (Np1 heads d : Nat)
    (q k _v dAtt : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  ∑ h : Fin heads, headPadFlat Np1 heads d h
    (vitCotDV (headWeightsF Np1 heads d h q k)
      (headSliceFlat Np1 heads d h dAtt))

/-- **Multi-head LN₁ cotangent** — the three-way Q/K/V dense-back fan-in (the structural wrinkle
    no prior net had), now at the multi-head Q/K/V dense cots. Same `vitCotLn1` shape; only the
    summed cotangents change. -/
noncomputable def vitCotLn1MH (Np1 heads d : Nat)
    (Wq Wk Wv : Mat (heads * d) (heads * d)) (q k v dAtt : Vec (Np1 * (heads * d))) :
    Vec (Np1 * (heads * d)) :=
  vitCotLn1 Wq Wk Wv (vitCotDQmh Np1 heads d q k v dAtt)
    (vitCotDKmh Np1 heads d q k v dAtt) (vitCotDVmh Np1 heads d q k v dAtt)

-- ════════════════════════════════════════════════════════════════
-- § Per-head: the rendered slice → SDPA-back → pad IS the proven per-head SDPA backward
-- ════════════════════════════════════════════════════════════════

/-- One head's Q term: `headPadFlat h (vitCotDQ d …)` at the saved slices of the full Q/K/V/dOut
    IS `Mat.flatten (headPadMat h (sdpa_back_Q d Q_h K_h V_h dOut_h))` — the single-head pin
    `vitCotDQ_eq_sdpa_back_Q` at `d_head`, slid through the slice/pad commutation bridges. -/
lemma vitCotDQ_head_eq (Np1 heads d : Nat) (h : Fin heads) (Q K V dOut : Mat Np1 (heads * d)) :
    headPadFlat Np1 heads d h
        (vitCotDQ d (headScoresF Np1 heads d h (Mat.flatten Q) (Mat.flatten K))
          (headSliceFlat Np1 heads d h (Mat.flatten K))
          (headSliceFlat Np1 heads d h (Mat.flatten V))
          (headSliceFlat Np1 heads d h (Mat.flatten dOut)))
      = Mat.flatten (headPadMat Np1 heads d h
          (sdpa_back_Q Np1 d (headSliceMat Np1 heads d h Q) (headSliceMat Np1 heads d h K)
            (headSliceMat Np1 heads d h V) (headSliceMat Np1 heads d h dOut))) := by
  unfold headScoresF
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]
  rw [headSliceFlat_flat, headSliceFlat_flat, headSliceFlat_flat, vitCotDQ_eq_sdpa_back_Q,
      headPadFlat_flat]

/-- One head's K term — `vitCotDK_eq_sdpa_back_K` at `d_head` through the bridges. -/
lemma vitCotDK_head_eq (Np1 heads d : Nat) (h : Fin heads) (Q K V dOut : Mat Np1 (heads * d)) :
    headPadFlat Np1 heads d h
        (vitCotDK d (headScoresF Np1 heads d h (Mat.flatten Q) (Mat.flatten K))
          (headSliceFlat Np1 heads d h (Mat.flatten Q))
          (headSliceFlat Np1 heads d h (Mat.flatten V))
          (headSliceFlat Np1 heads d h (Mat.flatten dOut)))
      = Mat.flatten (headPadMat Np1 heads d h
          (sdpa_back_K Np1 d (headSliceMat Np1 heads d h Q) (headSliceMat Np1 heads d h K)
            (headSliceMat Np1 heads d h V) (headSliceMat Np1 heads d h dOut))) := by
  unfold headScoresF
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]
  rw [headSliceFlat_flat, headSliceFlat_flat, headSliceFlat_flat, vitCotDK_eq_sdpa_back_K,
      headPadFlat_flat]

/-- One head's V term — `vitCotDV_eq_sdpa_back_V` at `d_head` through the bridges. -/
lemma vitCotDV_head_eq (Np1 heads d : Nat) (h : Fin heads) (Q K V dOut : Mat Np1 (heads * d)) :
    headPadFlat Np1 heads d h
        (vitCotDV (headWeightsF Np1 heads d h (Mat.flatten Q) (Mat.flatten K))
          (headSliceFlat Np1 heads d h (Mat.flatten dOut)))
      = Mat.flatten (headPadMat Np1 heads d h
          (sdpa_back_V Np1 d (headSliceMat Np1 heads d h Q) (headSliceMat Np1 heads d h K)
            (headSliceMat Np1 heads d h V) (headSliceMat Np1 heads d h dOut))) := by
  unfold headWeightsF
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]
  rw [headSliceFlat_flat, vitCotDV_eq_sdpa_back_V, headPadFlat_flat]

-- ════════════════════════════════════════════════════════════════
-- § The composition theorems — the multi-head dense backward, pinned to `sdpa_back_*`
-- ════════════════════════════════════════════════════════════════

/-- **Multi-head Q-dense backward, pinned.** At the saved full activations, the rendered
    `Σ_h headPadFlat h (vitCotDQ d …)` IS `Mat.flatten (Σ_h headPadMat h (sdpa_back_Q d …))`
    — the concat of the audited per-head SDPA Q-backwards. So the cotangent the Q dense ops
    contract with is the genuine multi-head attention Q-gradient. -/
theorem vitCotDQmh_eq (Np1 heads d : Nat) (Q K V dOut : Mat Np1 (heads * d)) :
    vitCotDQmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (∑ h : Fin heads, headPadMat Np1 heads d h
          (sdpa_back_Q Np1 d (headSliceMat Np1 heads d h Q) (headSliceMat Np1 heads d h K)
            (headSliceMat Np1 heads d h V) (headSliceMat Np1 heads d h dOut))) := by
  unfold vitCotDQmh
  rw [Finset.sum_congr rfl (fun h _ => vitCotDQ_head_eq Np1 heads d h Q K V dOut),
      ← flatten_sum]
  funext j
  rw [Finset.sum_apply]

/-- **Multi-head K-dense backward, pinned** — concat of the per-head `sdpa_back_K`. -/
theorem vitCotDKmh_eq (Np1 heads d : Nat) (Q K V dOut : Mat Np1 (heads * d)) :
    vitCotDKmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (∑ h : Fin heads, headPadMat Np1 heads d h
          (sdpa_back_K Np1 d (headSliceMat Np1 heads d h Q) (headSliceMat Np1 heads d h K)
            (headSliceMat Np1 heads d h V) (headSliceMat Np1 heads d h dOut))) := by
  unfold vitCotDKmh
  rw [Finset.sum_congr rfl (fun h _ => vitCotDK_head_eq Np1 heads d h Q K V dOut),
      ← flatten_sum]
  funext j
  rw [Finset.sum_apply]

/-- **Multi-head V-dense backward, pinned** — concat of the per-head `sdpa_back_V`. -/
theorem vitCotDVmh_eq (Np1 heads d : Nat) (Q K V dOut : Mat Np1 (heads * d)) :
    vitCotDVmh Np1 heads d (Mat.flatten Q) (Mat.flatten K) (Mat.flatten V) (Mat.flatten dOut)
      = Mat.flatten (∑ h : Fin heads, headPadMat Np1 heads d h
          (sdpa_back_V Np1 d (headSliceMat Np1 heads d h Q) (headSliceMat Np1 heads d h K)
            (headSliceMat Np1 heads d h V) (headSliceMat Np1 heads d h dOut))) := by
  unfold vitCotDVmh
  rw [Finset.sum_congr rfl (fun h _ => vitCotDV_head_eq Np1 heads d h Q K V dOut),
      ← flatten_sum]
  funext j
  rw [Finset.sum_apply]

end Proofs
