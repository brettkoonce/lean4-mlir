import LeanMlir.Proofs.SoftmaxBackFloatBridge
import LeanMlir.Proofs.ViTAttentionFloatBridge

/-! # ℝ→Float32 bridge: the SDPA BACKWARD — the Mat-space assembly (A3 §1f, the vit crux)

A3 (planning/a3_backward_deepnet_assembly.md §1f): the §1f softmax-Jacobian (`softmaxBack`) is the
genuinely-new op; this file is the **Mat-space assembly** around it — the backward peer of the
forward `sdpa_close` (`ViTAttentionFloatBridge.lean`). The certified scaled-dot-product-attention
backward (`Attention.lean`) is, working back from a cotangent `dOut`:

  dw      = dOut · Vᵀ                         -- `sdpa_dWeights`  (a matmul, rounded dot per entry)
  dScaled = softmaxBack(p, dw)  per query row  -- `sdpa_dScaled`   (the §1f row VJP `diag(p)−p·pᵀ`)
  dScores = (1/√d) · dScaled                  -- `sdpa_dScores`   (a scalar scale)
  dQ = dScores · K,  dK = dScoresᵀ · Q,  dV = pᵀ · dOut             -- three more matmuls

Every piece already has a float bridge: the matmuls are rounded dots (`attnScore_close` for `dw`,
`attnDot_close` for `dQ`/`dK`/`dV` — the latter at *perturbed* weights, exactly the forward output
matmul's shape), the row VJP is `softmaxBack_close` (rounding) + `softmaxBack_sub_abs_le` (the
`dw`-perturbation Lipschitz half), and the scale is `mul_close`. So there is **no new analysis** — the
backward threads the same per-entry rounding budgets the forward does, in the same `Mat`-space.

The float weights `fp` are supplied within `ew` of the forward softmax weights `p = sdpa_weights`
(`ew = attnWeightErr`, discharged by `sdpa_close`'s forward weights), and `p`/`dw`/`dScaled` magnitudes
are the saved-activation operating point — the honest smooth-point framing every backward bridge in
this codebase uses (`softmaxBack_close`/`bnGradInput_close` supply their float stats abstractly too).

Capstones: `sdpaBackV_close`, `sdpaBackQ_close`, `sdpaBackK_close` — each output entry of the deployed
float backward is within an explicit budget of the certified `sdpa_back_{V,Q,K}`. With these, the
attention sublayer's backward is `Mat`-space float-bridged; pair with the `Vec`-space MLP-half backward
for the transformer-block / whole-net fold. A3 = gradient *closeness* at a smooth point (NOT descent).
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Real magnitudes (saved-activation operating point)
-- ════════════════════════════════════════════════════════════════

/-- `|dw| = |dOut · Vᵀ|` bound: the `d`-fan-in matmul magnitude. -/
noncomputable def sdpaDwMag (d : Nat) (dA vA : ℝ) : ℝ := (d : ℝ) * dA * vA

/-- `|dScaled| = |softmaxBack(p, dw)|` bound (`softmaxBack_abs_le` at `P = 1`, `A = sdpaDwMag`). -/
noncomputable def sdpaDScaledMag (n d : Nat) (dA vA : ℝ) : ℝ :=
  1 * (sdpaDwMag d dA vA + (n : ℝ) * (1 * sdpaDwMag d dA vA))

/-- `|dScores| = |(1/√d)·dScaled|` bound: the scale `scaleA` times `sdpaDScaledMag`. -/
noncomputable def sdpaDScoresMag (n d : Nat) (dA vA scaleA : ℝ) : ℝ :=
  scaleA * sdpaDScaledMag n d dA vA

/-- **Forward softmax weights are probabilities** — `|sdpa_weights i j| ≤ 1`. -/
theorem sdpa_weights_abs_le_one {n d : Nat} (Q K : Mat n d) (i j : Fin n) :
    |sdpa_weights n d Q K i j| ≤ 1 := by
  unfold sdpa_weights rowSoftmax
  exact softmax_abs_le_one _ j

/-- **`dw` magnitude** — `|sdpa_dWeights V dOut i j| ≤ d·dA·vA` (the score-matmul bound at the
    cotangent/value magnitudes; `sdpa_dWeights = dOut · Vᵀ`, so this is `attnScore_abs_le`). -/
theorem sdpa_dWeights_abs_le {n d : Nat} (V dOut : Mat n d) {vA dA : ℝ}
    (hdA : 0 ≤ dA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (i j : Fin n) :
    |sdpa_dWeights V dOut i j| ≤ (d : ℝ) * dA * vA :=
  attnScore_abs_le dOut V hdA hdOut hV i j

/-- **`dScaled = softmaxBack(p, dw)`** — `sdpa_dScaled` is literally the per-row softmax-Jacobian
    backward at the saved weights `p` and the cotangent `dw`. Definitional. -/
theorem sdpa_dScaled_eq {n d : Nat} (Q K V dOut : Mat n d) (i j : Fin n) :
    sdpa_dScaled n d Q K V dOut i j
      = softmaxBack (sdpa_weights n d Q K i) (sdpa_dWeights V dOut i) j := rfl

/-- **`dScaled` magnitude** — `|sdpa_dScaled i j| ≤ sdpaDScaledMag` (`softmaxBack_abs_le`, `P = 1`). -/
theorem sdpa_dScaled_abs_le {n d : Nat} (Q K V dOut : Mat n d) {vA dA : ℝ}
    (hdA : 0 ≤ dA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (i j : Fin n) :
    |sdpa_dScaled n d Q K V dOut i j| ≤ sdpaDScaledMag n d dA vA := by
  rw [sdpa_dScaled_eq]
  unfold sdpaDScaledMag sdpaDwMag
  exact softmaxBack_abs_le (sdpa_weights n d Q K i) (sdpa_dWeights V dOut i)
    zero_le_one (fun k => sdpa_weights_abs_le_one Q K i k)
    (fun k => sdpa_dWeights_abs_le V dOut hdA hV hdOut i k) j

/-- **`dScores` magnitude** — `|sdpa_dScores i j| ≤ sdpaDScoresMag` (`|scale·dScaled| ≤ scaleA·…`). -/
theorem sdpa_dScores_abs_le {n d : Nat} (Q K V dOut : Mat n d) {vA dA scaleA : ℝ}
    (hdA : 0 ≤ dA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hscaleA : |sdpa_scale d| ≤ scaleA) (i j : Fin n) :
    |sdpa_dScores n d Q K V dOut i j| ≤ sdpaDScoresMag n d dA vA scaleA := by
  unfold sdpa_dScores sdpaDScoresMag
  rw [abs_mul]
  exact mul_le_mul hscaleA (sdpa_dScaled_abs_le Q K V dOut hdA hV hdOut i j)
    (abs_nonneg _) ((abs_nonneg _).trans hscaleA)

-- ════════════════════════════════════════════════════════════════
-- § The float backward maps (deployed, op-for-op in M-rounded arithmetic)
-- ════════════════════════════════════════════════════════════════

/-- **Float `dWeights`** `dOut · Vᵀ` — rounded dot of a `dOut` row with a `V` row. -/
noncomputable def FloatModel.sdpaDwF (M : FloatModel) {n d : Nat} (V dOut : Mat n d) : Mat n n :=
  fun i j => M.dot (dOut i) (fun k => V j k)

/-- **Float per-row softmax-Jacobian backward** — `softmaxBackF` applied row-wise at the saved float
    weights `fp` (within `ew` of `p`). -/
noncomputable def FloatModel.sdpaDScaledF (M : FloatModel) {n : Nat} (fp dwF : Mat n n) : Mat n n :=
  fun i => M.softmaxBackF (fp i) (dwF i)

/-- **Float `dScores`** — undo the `1/√d` scale (one rounded multiply). -/
noncomputable def FloatModel.sdpaDScoresF (M : FloatModel) {n : Nat} (d : Nat)
    (dScaledF : Mat n n) : Mat n n :=
  fun i j => M.mul (sdpa_scale d) (dScaledF i j)

/-- **Float backward w.r.t. V** — `pᵀ · dOut`, rounded dot of a weight column with a `dOut` column. -/
noncomputable def FloatModel.sdpaBackVF (M : FloatModel) {n d : Nat} (fp : Mat n n)
    (dOut : Mat n d) : Mat n d :=
  fun i j => M.dot (fun k => fp k i) (fun k => dOut k j)

/-- **Float backward w.r.t. Q** — `dScores · K`. -/
noncomputable def FloatModel.sdpaBackQF (M : FloatModel) {n d : Nat} (dScoresF : Mat n n)
    (K : Mat n d) : Mat n d :=
  fun i j => M.dot (dScoresF i) (fun k => K k j)

/-- **Float backward w.r.t. K** — `dScoresᵀ · Q`. -/
noncomputable def FloatModel.sdpaBackKF (M : FloatModel) {n d : Nat} (dScoresF : Mat n n)
    (Q : Mat n d) : Mat n d :=
  fun i j => M.dot (fun k => dScoresF k i) (fun k => Q k j)

-- ════════════════════════════════════════════════════════════════
-- § The per-stage rounding budgets
-- ════════════════════════════════════════════════════════════════

/-- **`dScaled` budget** — the softmax-Jacobian backward's rounding (`softmaxBackBudget` at the
    perturbed float weights, input magnitude `sdpaDwMag + attnScoreErr`) plus the `dw`-perturbation
    Lipschitz term (`softmaxBack_sub_abs_le` at `e = attnScoreErr`, `P = 1`, `c = n`). -/
noncomputable def FloatModel.sdpaDScaledErr (M : FloatModel) (n d : Nat) (dA vA ew : ℝ) : ℝ :=
  M.softmaxBackBudget n 1 ((d : ℝ) * dA * vA + M.attnScoreErr d dA vA) ew
    + (1 : ℝ) * (M.attnScoreErr d dA vA + (n : ℝ) * (1 * M.attnScoreErr d dA vA))

/-- **`dScores` budget** — the (exact) `1/√d` scale times the `dScaled`-perturbed gradient
    (`mul_close`, scale error `0`, second-operand error `sdpaDScaledErr`). -/
noncomputable def FloatModel.sdpaDScoresErr (M : FloatModel) (n d : Nat)
    (dA vA scaleA ew : ℝ) : ℝ :=
  FloatModel.mulErr M.u scaleA (sdpaDScaledMag n d dA vA) 0 (M.sdpaDScaledErr n d dA vA ew)

/-- **Final-matmul budget** — a rounded dot at perturbed weights (`attnDot_close`): the Higham γ over
    the `n` fan-in (weights bounded by `wMag + eweight`) plus the weight perturbation `n·eweight·vA`. -/
noncomputable def FloatModel.sdpaBackErr (M : FloatModel) (n : Nat) (wMag eweight vA : ℝ) : ℝ :=
  ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (wMag + eweight) * vA) + (n : ℝ) * eweight * vA

-- ════════════════════════════════════════════════════════════════
-- § The per-stage closeness lemmas
-- ════════════════════════════════════════════════════════════════

/-- **`dWeights` closeness.** The rounded `dOut · Vᵀ` entry is within `attnScoreErr` of the real
    `sdpa_dWeights` (`attnScore_close`, no input perturbation). -/
theorem sdpaDwF_close (M : FloatModel) {n d : Nat} (V dOut : Mat n d)
    {vA dA : ℝ} (hdA : 0 ≤ dA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (i j : Fin n) :
    |M.sdpaDwF V dOut i j - sdpa_dWeights V dOut i j| ≤ M.attnScoreErr d dA vA :=
  M.attnScore_close dOut V hdA hdOut hV i j

/-- **`dScaled` closeness.** The deployed float per-row softmax-Jacobian backward at the float
    weights `fp` and float cotangent `dwF` is within `sdpaDScaledErr` of the certified `sdpa_dScaled`.
    Triangle: `softmaxBack_close` (rounding at the common cotangent `dwF`) + `softmaxBack_sub_abs_le`
    (the `dwF`-vs-`dw` perturbation, `softmaxBack` linear in the cotangent). -/
theorem sdpaDScaledF_close (M : FloatModel) {n d : Nat} (Q K V dOut : Mat n d) (fp : Mat n n)
    {vA dA ew : ℝ} (hdA : 0 ≤ dA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ i j, |fp i j - sdpa_weights n d Q K i j| ≤ ew) (i j : Fin n) :
    |M.sdpaDScaledF fp (M.sdpaDwF V dOut) i j - sdpa_dScaled n d Q K V dOut i j|
      ≤ M.sdpaDScaledErr n d dA vA ew := by
  -- saved weights / real & float cotangents at row i
  have hpabs : ∀ k, |sdpa_weights n d Q K i k| ≤ (1 : ℝ) := fun k => sdpa_weights_abs_le_one Q K i k
  have hfpclose : ∀ k, |fp i k - sdpa_weights n d Q K i k| ≤ ew := fun k => hfp i k
  have hdwF_close : ∀ k, |M.sdpaDwF V dOut i k - sdpa_dWeights V dOut i k| ≤ M.attnScoreErr d dA vA :=
    fun k => sdpaDwF_close M V dOut hdA hV hdOut i k
  have hdwmag : ∀ k, |sdpa_dWeights V dOut i k| ≤ (d : ℝ) * dA * vA :=
    fun k => sdpa_dWeights_abs_le V dOut hdA hV hdOut i k
  have hdwFmag : ∀ k, |M.sdpaDwF V dOut i k| ≤ (d : ℝ) * dA * vA + M.attnScoreErr d dA vA := by
    intro k
    have h := abs_sub_le (M.sdpaDwF V dOut i k) (sdpa_dWeights V dOut i k) 0
    simp only [sub_zero] at h
    linarith [hdwF_close k, hdwmag k]
  -- the rounding half (common cotangent dwF) and the perturbation half (dwF vs dw)
  have h1 : |M.softmaxBackF (fp i) (M.sdpaDwF V dOut i) j
              - softmaxBack (sdpa_weights n d Q K i) (M.sdpaDwF V dOut i) j|
            ≤ M.softmaxBackBudget n 1 ((d : ℝ) * dA * vA + M.attnScoreErr d dA vA) ew :=
    softmaxBack_close M (sdpa_weights n d Q K i) (fp i) (M.sdpaDwF V dOut i)
      zero_le_one hpabs hfpclose hdwFmag j
  have h2 : |softmaxBack (sdpa_weights n d Q K i) (M.sdpaDwF V dOut i) j
              - softmaxBack (sdpa_weights n d Q K i) (sdpa_dWeights V dOut i) j|
            ≤ (1 : ℝ) * (M.attnScoreErr d dA vA + (n : ℝ) * (1 * M.attnScoreErr d dA vA)) :=
    softmaxBack_sub_abs_le (sdpa_weights n d Q K i) (M.sdpaDwF V dOut i) (sdpa_dWeights V dOut i)
      zero_le_one hpabs hdwF_close j
  show |M.softmaxBackF (fp i) (M.sdpaDwF V dOut i) j - sdpa_dScaled n d Q K V dOut i j|
      ≤ M.sdpaDScaledErr n d dA vA ew
  rw [sdpa_dScaled_eq]
  unfold FloatModel.sdpaDScaledErr
  calc |M.softmaxBackF (fp i) (M.sdpaDwF V dOut i) j
          - softmaxBack (sdpa_weights n d Q K i) (sdpa_dWeights V dOut i) j|
      ≤ |M.softmaxBackF (fp i) (M.sdpaDwF V dOut i) j
          - softmaxBack (sdpa_weights n d Q K i) (M.sdpaDwF V dOut i) j|
        + |softmaxBack (sdpa_weights n d Q K i) (M.sdpaDwF V dOut i) j
          - softmaxBack (sdpa_weights n d Q K i) (sdpa_dWeights V dOut i) j| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add h1 h2

/-- **`dScores` closeness.** The (exact) `1/√d` rescale of the float `dScaled` is within
    `sdpaDScoresErr` of the certified `sdpa_dScores` (`mul_close`). -/
theorem sdpaDScoresF_close (M : FloatModel) {n d : Nat} (Q K V dOut : Mat n d) (fp : Mat n n)
    {vA dA scaleA ew : ℝ} (hdA : 0 ≤ dA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hscaleA : |sdpa_scale d| ≤ scaleA)
    (hfp : ∀ i j, |fp i j - sdpa_weights n d Q K i j| ≤ ew) (i j : Fin n) :
    |M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) i j - sdpa_dScores n d Q K V dOut i j|
      ≤ M.sdpaDScoresErr n d dA vA scaleA ew := by
  have hclose := sdpaDScaledF_close M Q K V dOut fp hdA hV hdOut hfp i j
  have hmag := sdpa_dScaled_abs_le Q K V dOut hdA hV hdOut i j
  have key := M.mul_close (ea := 0) (xt := sdpa_scale d) (x := sdpa_scale d)
    (yt := M.sdpaDScaledF fp (M.sdpaDwF V dOut) i j) (y := sdpa_dScaled n d Q K V dOut i j)
    (by simp) hclose hscaleA hmag
  show |M.mul (sdpa_scale d) (M.sdpaDScaledF fp (M.sdpaDwF V dOut) i j)
        - sdpa_scale d * sdpa_dScaled n d Q K V dOut i j| ≤ M.sdpaDScoresErr n d dA vA scaleA ew
  unfold FloatModel.sdpaDScoresErr
  exact key

-- ════════════════════════════════════════════════════════════════
-- § The three backward capstones (per output entry)
-- ════════════════════════════════════════════════════════════════

/-- **SDPA backward w.r.t. V — float-close.** Each entry of the deployed `pᵀ · dOut` is within
    `sdpaBackErr n 1 ew dA` of the certified `sdpa_back_V` (`attnDot_close` at the perturbed softmax
    weights `fp`, weights bounded by `1`, `dOut` by `dA`). The simplest path: V flows only through the
    final matmul, so this needs no score/softmax chain. -/
theorem sdpaBackV_close (M : FloatModel) {n d : Nat} (Q K V dOut : Mat n d) (fp : Mat n n)
    {dA ew : ℝ} (hdA : 0 ≤ dA) (hew : 0 ≤ ew)
    (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ i j, |fp i j - sdpa_weights n d Q K i j| ≤ ew) (i : Fin n) (j : Fin d) :
    |M.sdpaBackVF fp dOut i j - sdpa_back_V n d Q K V dOut i j| ≤ M.sdpaBackErr n 1 ew dA := by
  have hwabs : ∀ k, |sdpa_weights n d Q K k i| ≤ (1 : ℝ) := fun k => sdpa_weights_abs_le_one Q K k i
  have hwclose : ∀ k, |fp k i - sdpa_weights n d Q K k i| ≤ ew := fun k => hfp k i
  have hv : ∀ k, |dOut k j| ≤ dA := fun k => hdOut k j
  have key := M.attnDot_close (fun k => fp k i) (fun k => sdpa_weights n d Q K k i)
    (fun k => dOut k j) hew zero_le_one hdA hwclose hwabs hv
  show |M.dot (fun k => fp k i) (fun k => dOut k j) - sdpa_back_V n d Q K V dOut i j|
      ≤ M.sdpaBackErr n 1 ew dA
  unfold sdpa_back_V Mat.mul Mat.transpose FloatModel.sdpaBackErr
  exact key

/-- **SDPA backward w.r.t. Q — float-close.** Each entry of the deployed `dScores · K` is within
    `sdpaBackErr n (sdpaDScoresMag) (sdpaDScoresErr) kA` of the certified `sdpa_back_Q` (`attnDot_close`
    at the perturbed gradient weights `dScoresF`, the full `dw → dScaled → dScores` chain discharging
    their closeness/magnitude). -/
theorem sdpaBackQ_close (M : FloatModel) {n d : Nat} (Q K V dOut : Mat n d) (fp : Mat n n)
    {kA vA dA scaleA ew : ℝ}
    (hkA : 0 ≤ kA) (hdA : 0 ≤ dA)
    (hscaleA : |sdpa_scale d| ≤ scaleA)
    (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ i j, |fp i j - sdpa_weights n d Q K i j| ≤ ew) (i : Fin n) (j : Fin d) :
    |M.sdpaBackQF (M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut))) K i j
        - sdpa_back_Q n d Q K V dOut i j|
      ≤ M.sdpaBackErr n (sdpaDScoresMag n d dA vA scaleA) (M.sdpaDScoresErr n d dA vA scaleA ew) kA := by
  have hedScores0 : 0 ≤ M.sdpaDScoresErr n d dA vA scaleA ew :=
    (abs_nonneg _).trans (sdpaDScoresF_close M Q K V dOut fp hdA hV hdOut hscaleA hfp i i)
  have hmag0 : 0 ≤ sdpaDScoresMag n d dA vA scaleA :=
    (abs_nonneg _).trans (sdpa_dScores_abs_le Q K V dOut hdA hV hdOut hscaleA i i)
  have hwclose : ∀ k,
      |M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) i k - sdpa_dScores n d Q K V dOut i k|
        ≤ M.sdpaDScoresErr n d dA vA scaleA ew :=
    fun k => sdpaDScoresF_close M Q K V dOut fp hdA hV hdOut hscaleA hfp i k
  have hwabs : ∀ k, |sdpa_dScores n d Q K V dOut i k| ≤ sdpaDScoresMag n d dA vA scaleA :=
    fun k => sdpa_dScores_abs_le Q K V dOut hdA hV hdOut hscaleA i k
  have hv : ∀ k, |K k j| ≤ kA := fun k => hK k j
  have key := M.attnDot_close (M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) i)
    (sdpa_dScores n d Q K V dOut i) (fun k => K k j) hedScores0 hmag0 hkA hwclose hwabs hv
  show |M.dot (M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) i) (fun k => K k j)
        - sdpa_back_Q n d Q K V dOut i j|
      ≤ M.sdpaBackErr n (sdpaDScoresMag n d dA vA scaleA) (M.sdpaDScoresErr n d dA vA scaleA ew) kA
  unfold sdpa_back_Q Mat.mul FloatModel.sdpaBackErr
  exact key

/-- **SDPA backward w.r.t. K — float-close.** Each entry of the deployed `dScoresᵀ · Q` is within
    `sdpaBackErr n (sdpaDScoresMag) (sdpaDScoresErr) qA` of the certified `sdpa_back_K` (same
    `attnDot_close`, with the transposed `dScores` column and `Q` as the vector). -/
theorem sdpaBackK_close (M : FloatModel) {n d : Nat} (Q K V dOut : Mat n d) (fp : Mat n n)
    {qA vA dA scaleA ew : ℝ}
    (hqA : 0 ≤ qA) (hdA : 0 ≤ dA)
    (hscaleA : |sdpa_scale d| ≤ scaleA)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hV : ∀ j k, |V j k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ i j, |fp i j - sdpa_weights n d Q K i j| ≤ ew) (i : Fin n) (j : Fin d) :
    |M.sdpaBackKF (M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut))) Q i j
        - sdpa_back_K n d Q K V dOut i j|
      ≤ M.sdpaBackErr n (sdpaDScoresMag n d dA vA scaleA) (M.sdpaDScoresErr n d dA vA scaleA ew) qA := by
  have hedScores0 : 0 ≤ M.sdpaDScoresErr n d dA vA scaleA ew :=
    (abs_nonneg _).trans (sdpaDScoresF_close M Q K V dOut fp hdA hV hdOut hscaleA hfp i i)
  have hmag0 : 0 ≤ sdpaDScoresMag n d dA vA scaleA :=
    (abs_nonneg _).trans (sdpa_dScores_abs_le Q K V dOut hdA hV hdOut hscaleA i i)
  have hwclose : ∀ k,
      |M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) k i - sdpa_dScores n d Q K V dOut k i|
        ≤ M.sdpaDScoresErr n d dA vA scaleA ew :=
    fun k => sdpaDScoresF_close M Q K V dOut fp hdA hV hdOut hscaleA hfp k i
  have hwabs : ∀ k, |sdpa_dScores n d Q K V dOut k i| ≤ sdpaDScoresMag n d dA vA scaleA :=
    fun k => sdpa_dScores_abs_le Q K V dOut hdA hV hdOut hscaleA k i
  have hv : ∀ k, |Q k j| ≤ qA := fun k => hQ k j
  have key := M.attnDot_close (fun k => M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) k i)
    (fun k => sdpa_dScores n d Q K V dOut k i) (fun k => Q k j)
    hedScores0 hmag0 hqA hwclose hwabs hv
  show |M.dot (fun k => M.sdpaDScoresF d (M.sdpaDScaledF fp (M.sdpaDwF V dOut)) k i) (fun k => Q k j)
        - sdpa_back_K n d Q K V dOut i j|
      ≤ M.sdpaBackErr n (sdpaDScoresMag n d dA vA scaleA) (M.sdpaDScoresErr n d dA vA scaleA ew) qA
  unfold sdpa_back_K Mat.mul Mat.transpose FloatModel.sdpaBackErr
  exact key

-- ════════════════════════════════════════════════════════════════
-- § Multi-head wrap — the per-head sdpa-core backward over the head axis
--
-- Multi-head attention runs `sdpa` independently on each of the `h` head-slabs of the (projected)
-- Q/K/V, then concatenates (`mhsa_layer`, `Attention.lean`). At a column `j : Fin (h·dh)` of a
-- `Mat N (h·dh)`, `finProdFinEquiv.symm j = (head hd, within-head c)` decodes which head owns it
-- (matching the certified concat `output[n, fPF(hd, c)] = perHead hd n c`). So the multi-head sdpa
-- backward is the per-head concatenation of the certified single-head `sdpa_back_{V,Q,K}` over the
-- head slabs — a thin reindexing wrap, with the budget head-INDEPENDENT (every head is dim `dh`,
-- scale `1/√dh`). Each output entry reduces to its head's single-head capstone (`sdpaBack*_close`).
--
-- Scope: this is the attention-CORE backward (Q/K/V/dOut already in head-concat layout). The full
-- MHSA backward composes this with the Q/K/V/O projection denses' `linBack` (input/param VJPs) and
-- the three-way Q+K+V fan-in at X (`biPathSum`/`residual`) — existing combinators, no new op.
-- ════════════════════════════════════════════════════════════════

/-- The column slab `[hd·dh, (hd+1)·dh)` of a `Mat n (h·dh)` as a `Mat n dh` (head hd's view) — the
    `finProdFinEquiv (hd, ·)` column restriction, matching `mhsa_layer`'s per-head extraction. -/
noncomputable def mhSlab {n h dh : Nat} (hd : Fin h) (Q : Mat n (h * dh)) : Mat n dh :=
  fun i c => Q i (finProdFinEquiv (hd, c))

/-- **Multi-head sdpa backward w.r.t. V** (real) — per head, the certified `sdpa_back_V` on the head
    slabs; concatenated by the `finProdFinEquiv` column layout. -/
noncomputable def mhsaSdpaBackV {h N dh : Nat} (Q K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => sdpa_back_V N dh (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
              (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. Q** (real). -/
noncomputable def mhsaSdpaBackQ {h N dh : Nat} (Q K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => sdpa_back_Q N dh (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
              (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. K** (real). -/
noncomputable def mhsaSdpaBackK {h N dh : Nat} (Q K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => sdpa_back_K N dh (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
              (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

/-- **Float multi-head sdpa backward w.r.t. V** — per head, `sdpaBackVF` at the saved float weights
    `fp hd` (within `ew` of head hd's softmax weights) and the head-slab of the cotangent. -/
noncomputable def FloatModel.mhsaSdpaBackVF (M : FloatModel) {h N dh : Nat} (fp : Fin h → Mat N N)
    (dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j => M.sdpaBackVF (fp (finProdFinEquiv.symm j).1) (mhSlab (finProdFinEquiv.symm j).1 dOut)
              i (finProdFinEquiv.symm j).2

/-- **Float multi-head sdpa backward w.r.t. Q** — per head, the full `dw → dScaled → dScores → dScores·K`
    float chain on head hd's slabs. -/
noncomputable def FloatModel.mhsaSdpaBackQF (M : FloatModel) {h N dh : Nat} (fp : Fin h → Mat N N)
    (K V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j =>
    M.sdpaBackQF (M.sdpaDScoresF dh (M.sdpaDScaledF (fp (finProdFinEquiv.symm j).1)
        (M.sdpaDwF (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut))))
      (mhSlab (finProdFinEquiv.symm j).1 K) i (finProdFinEquiv.symm j).2

/-- **Float multi-head sdpa backward w.r.t. K** — per head, the float chain feeding `dScoresᵀ·Q`. -/
noncomputable def FloatModel.mhsaSdpaBackKF (M : FloatModel) {h N dh : Nat} (fp : Fin h → Mat N N)
    (Q V dOut : Mat N (h * dh)) : Mat N (h * dh) :=
  fun i j =>
    M.sdpaBackKF (M.sdpaDScoresF dh (M.sdpaDScaledF (fp (finProdFinEquiv.symm j).1)
        (M.sdpaDwF (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut))))
      (mhSlab (finProdFinEquiv.symm j).1 Q) i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. V — float-close.** Each entry reduces to head hd's single-head
    `sdpaBackV_close`; the budget `sdpaBackErr N 1 ew dA` is head-independent. The per-head float
    weights `fp hd` are within `ew` of head hd's softmax weights `sdpa_weights N dh (mhSlab hd Q) …`. -/
theorem mhsaSdpaBackV_close (M : FloatModel) {h N dh : Nat} (Q K V dOut : Mat N (h * dh))
    (fp : Fin h → Mat N N) {dA ew : ℝ} (hdA : 0 ≤ dA) (hew : 0 ≤ ew)
    (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew)
    (i : Fin N) (j : Fin (h * dh)) :
    |M.mhsaSdpaBackVF fp dOut i j - mhsaSdpaBackV Q K V dOut i j| ≤ M.sdpaBackErr N 1 ew dA := by
  unfold FloatModel.mhsaSdpaBackVF mhsaSdpaBackV
  exact sdpaBackV_close M (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
    (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
    (fp (finProdFinEquiv.symm j).1) hdA hew
    (fun a b => hdOut a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (hfp (finProdFinEquiv.symm j).1) i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. Q — float-close.** Each entry reduces to head hd's
    `sdpaBackQ_close`; the budget is head-independent (dim `dh`, scale `1/√dh`). -/
theorem mhsaSdpaBackQ_close (M : FloatModel) {h N dh : Nat} (Q K V dOut : Mat N (h * dh))
    (fp : Fin h → Mat N N) {kA vA dA scaleA ew : ℝ}
    (hkA : 0 ≤ kA) (hdA : 0 ≤ dA) (hscaleA : |sdpa_scale dh| ≤ scaleA)
    (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ i k, |V i k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew)
    (i : Fin N) (j : Fin (h * dh)) :
    |M.mhsaSdpaBackQF fp K V dOut i j - mhsaSdpaBackQ Q K V dOut i j|
      ≤ M.sdpaBackErr N (sdpaDScoresMag N dh dA vA scaleA) (M.sdpaDScoresErr N dh dA vA scaleA ew) kA := by
  unfold FloatModel.mhsaSdpaBackQF mhsaSdpaBackQ
  exact sdpaBackQ_close M (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
    (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
    (fp (finProdFinEquiv.symm j).1) hkA hdA hscaleA
    (fun a b => hK a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (fun a b => hV a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (fun a b => hdOut a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (hfp (finProdFinEquiv.symm j).1) i (finProdFinEquiv.symm j).2

/-- **Multi-head sdpa backward w.r.t. K — float-close.** Each entry reduces to head hd's
    `sdpaBackK_close`. -/
theorem mhsaSdpaBackK_close (M : FloatModel) {h N dh : Nat} (Q K V dOut : Mat N (h * dh))
    (fp : Fin h → Mat N N) {qA vA dA scaleA ew : ℝ}
    (hqA : 0 ≤ qA) (hdA : 0 ≤ dA) (hscaleA : |sdpa_scale dh| ≤ scaleA)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hV : ∀ i k, |V i k| ≤ vA) (hdOut : ∀ i k, |dOut i k| ≤ dA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew)
    (i : Fin N) (j : Fin (h * dh)) :
    |M.mhsaSdpaBackKF fp Q V dOut i j - mhsaSdpaBackK Q K V dOut i j|
      ≤ M.sdpaBackErr N (sdpaDScoresMag N dh dA vA scaleA) (M.sdpaDScoresErr N dh dA vA scaleA ew) qA := by
  unfold FloatModel.mhsaSdpaBackKF mhsaSdpaBackK
  exact sdpaBackK_close M (mhSlab (finProdFinEquiv.symm j).1 Q) (mhSlab (finProdFinEquiv.symm j).1 K)
    (mhSlab (finProdFinEquiv.symm j).1 V) (mhSlab (finProdFinEquiv.symm j).1 dOut)
    (fp (finProdFinEquiv.symm j).1) hqA hdA hscaleA
    (fun a b => hQ a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (fun a b => hV a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (fun a b => hdOut a (finProdFinEquiv ((finProdFinEquiv.symm j).1, b)))
    (hfp (finProdFinEquiv.symm j).1) i (finProdFinEquiv.symm j).2

end Proofs
