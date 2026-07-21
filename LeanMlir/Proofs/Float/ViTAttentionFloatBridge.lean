import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Float.FloatBridge

/-!
# ℝ→Float32 bridge: ViT — §2c scaled dot-product attention (the `Mat`-space track)

The other ViT pieces (LayerNorm §2a, GELU §2b, the per-token MLP residual §2d) live in
`Vec`-space and fold through `FloatClose`/`FloatBridges`. **Attention is different**: it
mixes *across tokens*, so it lives in `Mat n d` space and the per-row softmax couples a
whole row of logits at once. This file is that track — a direct per-entry float closeness
for `sdpa` (the repo's scaled-dot-product attention, `Attention.lean`), assembled from
pieces that already exist:

* **the two score/output matmuls** are each a *rounded dot* (`Mat.mul A B i j = `the dot of
  a row of `A` with a column of `B`), so `FloatModel.dot_close` (Higham γ) bounds them;
* **the `1/√d` scale** is one rounded multiply — `FloatModel.mul_close`;
* **the per-row softmax** is `FloatModel.softmaxF` at the float scores vs `softmax` at the
  real scores, within `smErr` by `softmaxF_close_at` (rounding `softmaxF_close` + the logit
  perturbation `softmax_perturb`).

The capstone `sdpa_close` chains them: each output entry of the float attention `sdpaF` is
within `attnOutErr` of the real `sdpa`. All-smooth, so no sign-flip margins (unlike the
ReLU/maxpool kinked ops). The budget is **a-posteriori in magnitudes** (supplied `qA`/`kA`/
`vA`/`scaleA` bounds on the Q/K/V/scale), **proved in rounding** — the project's standard
honest framing (`planning/floatbridge_enet_vit.md` §3.4). The transformer-block fold
(LN→MHSA→+→LN→MLP→+) then pairs this `Mat`-space MHSA with the `Vec`-space MLP half.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Score entry = a real dot of a Q-row and a K-row
-- ════════════════════════════════════════════════════════════════

/-- One attention score entry is the dot of a `Q`-row with a `K`-row:
    `(Q · Kᵀ) i j = ∑ k, Q i k · K j k`. -/
theorem matScore_eq {n d : Nat} (Q K : Mat n d) (i j : Fin n) :
    Mat.mul Q (Mat.transpose K) i j = ∑ k, Q i k * K j k := rfl

/-- **Real score magnitude:** `|(Q · Kᵀ) i j| ≤ d · qA · kA` under uniform entry bounds. -/
theorem attnScore_abs_le {n d : Nat} (Q K : Mat n d) {qA kA : ℝ}
    (hqA : 0 ≤ qA) (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ j k, |K j k| ≤ kA)
    (i j : Fin n) :
    |Mat.mul Q (Mat.transpose K) i j| ≤ (d : ℝ) * qA * kA := by
  rw [matScore_eq]
  calc |∑ k, Q i k * K j k| ≤ ∑ k, |Q i k * K j k| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _k : Fin d, qA * kA := by
        refine Finset.sum_le_sum fun k _ => ?_
        rw [abs_mul]; exact mul_le_mul (hQ i k) (hK j k) (abs_nonneg _) hqA
    _ = (d : ℝ) * (qA * kA) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    _ = (d : ℝ) * qA * kA := by ring

/-- `mulErr` is nonnegative when all of its arguments are. -/
theorem mulErr_nonneg {u A C ea ec : ℝ} (hu : 0 ≤ u) (hA : 0 ≤ A) (hC : 0 ≤ C)
    (hea : 0 ≤ ea) (hec : 0 ≤ ec) : 0 ≤ FloatModel.mulErr u A C ea ec := by
  unfold FloatModel.mulErr
  have t1 : 0 ≤ u * ((A + ea) * (C + ec)) :=
    mul_nonneg hu (mul_nonneg (by linarith) (by linarith))
  have t2 : 0 ≤ A * ec := mul_nonneg hA hec
  have t3 : 0 ≤ ea * C := mul_nonneg hea hC
  have t4 : 0 ≤ ea * ec := mul_nonneg hea hec
  linarith

namespace FloatModel

variable (M : FloatModel)

-- ════════════════════════════════════════════════════════════════
-- § The per-entry attention budgets
-- ════════════════════════════════════════════════════════════════

/-- Forward error of one rounded attention score `M.dot (Q i) (K j)` against the real
    `(Q · Kᵀ) i j`: the Higham γ over the fan-in `d`, on magnitude `d · qA · kA`. -/
noncomputable def attnScoreErr (d : Nat) (qA kA : ℝ) : ℝ :=
  ((1 + M.u) ^ (d + 1) - 1) * ((d : ℝ) * qA * kA)

/-- Error of the scaled score `M.mul (1/√d) (M.dot (Q i) (K j))` against `(1/√d)·(Q·Kᵀ)ᵢⱼ`:
    `mulErr` with the scale exact (`ea = 0`) and the score error `attnScoreErr` inherited. -/
noncomputable def attnScaledErr (d : Nat) (qA kA scaleA : ℝ) : ℝ :=
  FloatModel.mulErr M.u scaleA ((d : ℝ) * qA * kA) 0 (M.attnScoreErr d qA kA)

/-- Error of one per-row softmax weight `M.softmaxF` at the float scaled scores against
    `softmax` at the real ones: `smErr` at logit-perturbation `δ = attnScaledErr`. -/
noncomputable def attnWeightErr (n d : Nat) (qA kA scaleA eexp : ℝ) : ℝ :=
  smErr M.u eexp (M.attnScaledErr d qA kA scaleA) n

/-- Full per-entry SDPA output error: the output matmul's rounding (Higham γ over the
    `n` tokens, weights bounded by `1 + attnWeightErr`, `V` by `vA`) plus the weight
    perturbation `n · attnWeightErr · vA`. -/
noncomputable def attnOutErr (n d : Nat) (qA kA vA scaleA eexp : ℝ) : ℝ :=
  ((1 + M.u) ^ (n + 1) - 1)
      * ((n : ℝ) * (1 + M.attnWeightErr n d qA kA scaleA eexp) * vA)
    + (n : ℝ) * (M.attnWeightErr n d qA kA scaleA eexp) * vA

/-- The score/scaled budgets are nonnegative. -/
theorem attnScoreErr_nonneg (d : Nat) {qA kA : ℝ} (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) :
    0 ≤ M.attnScoreErr d qA kA := by
  unfold FloatModel.attnScoreErr
  exact mul_nonneg (sub_nonneg.mpr (one_le_pow₀ (by have := M.u_nonneg; linarith)))
    (mul_nonneg (mul_nonneg (Nat.cast_nonneg d) hqA) hkA)

theorem attnScaledErr_nonneg (d : Nat) {qA kA scaleA : ℝ}
    (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hscaleA : 0 ≤ scaleA) :
    0 ≤ M.attnScaledErr d qA kA scaleA :=
  mulErr_nonneg M.u_nonneg hscaleA (mul_nonneg (mul_nonneg (Nat.cast_nonneg d) hqA) hkA)
    le_rfl (M.attnScoreErr_nonneg d hqA hkA)

theorem attnWeightErr_nonneg (n d : Nat) {qA kA scaleA eexp : ℝ}
    (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hscaleA : 0 ≤ scaleA) (heexp0 : 0 ≤ eexp)
    (hρ1 : smRho M.u eexp n < 1) : 0 ≤ M.attnWeightErr n d qA kA scaleA eexp :=
  M.smErr_nonneg heexp0 (M.attnScaledErr_nonneg d hqA hkA hscaleA) hρ1

theorem attnOutErr_nonneg (n d : Nat) {qA kA vA scaleA eexp : ℝ}
    (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hvA : 0 ≤ vA) (hscaleA : 0 ≤ scaleA) (heexp0 : 0 ≤ eexp)
    (hρ1 : smRho M.u eexp n < 1) : 0 ≤ M.attnOutErr n d qA kA vA scaleA eexp := by
  unfold FloatModel.attnOutErr
  have hG : (0 : ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by have := M.u_nonneg; linarith))
  have hew := M.attnWeightErr_nonneg n d hqA hkA hscaleA heexp0 hρ1
  have h1 : 0 ≤ ((1 + M.u) ^ (n + 1) - 1)
      * ((n : ℝ) * (1 + M.attnWeightErr n d qA kA scaleA eexp) * vA) :=
    mul_nonneg hG (mul_nonneg (mul_nonneg (Nat.cast_nonneg n) (by linarith)) hvA)
  have h2 : 0 ≤ (n : ℝ) * (M.attnWeightErr n d qA kA scaleA eexp) * vA :=
    mul_nonneg (mul_nonneg (Nat.cast_nonneg n) hew) hvA
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The three closeness stages
-- ════════════════════════════════════════════════════════════════

/-- **Stage A — score closeness.** The rounded score dot is within the Higham γ budget
    `attnScoreErr` of the real `(Q · Kᵀ) i j` (`dot_close`, no input perturbation). -/
theorem attnScore_close {n d : Nat} (Q K : Mat n d) {qA kA : ℝ}
    (hqA : 0 ≤ qA) (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ j k, |K j k| ≤ kA) (i j : Fin n) :
    |M.dot (Q i) (K j) - Mat.mul Q (Mat.transpose K) i j| ≤ M.attnScoreErr d qA kA := by
  have hsum : (∑ k, |Q i k * K j k|) ≤ (d : ℝ) * qA * kA := by
    calc (∑ k, |Q i k * K j k|) ≤ ∑ _k : Fin d, qA * kA := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [abs_mul]; exact mul_le_mul (hQ i k) (hK j k) (abs_nonneg _) hqA
      _ = (d : ℝ) * (qA * kA) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ = (d : ℝ) * qA * kA := by ring
  have hG0 : (0 : ℝ) ≤ (1 + M.u) ^ (d + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by have := M.u_nonneg; linarith))
  rw [matScore_eq]
  exact (M.dot_close (Q i) (K j)).trans (mul_le_mul_of_nonneg_left hsum hG0)

/-- **Stage B — scaled closeness.** Multiplying the float score by the (exact) scale
    `scale` lands within `attnScaledErr` of `scale · (Q · Kᵀ) i j` (`mul_close`, the
    score error `attnScoreErr` riding in as the second operand's inherited error). -/
theorem attnScaled_close {n d : Nat} (Q K : Mat n d) {qA kA scale scaleA : ℝ}
    (hqA : 0 ≤ qA) (hscaleA : |scale| ≤ scaleA)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ j k, |K j k| ≤ kA) (i j : Fin n) :
    |M.mul scale (M.dot (Q i) (K j)) - scale * Mat.mul Q (Mat.transpose K) i j|
      ≤ M.attnScaledErr d qA kA scaleA :=
  M.mul_close (by simp) (M.attnScore_close Q K hqA hQ hK i j) hscaleA
    (attnScore_abs_le Q K hqA hQ hK i j)

/-- **Stage D — output matmul closeness.** A rounded dot of perturbed softmax weights `wF`
    (within `eweight` of the real weights `w`, themselves `≤ wMag`) against a column `vcol`
    of `V` (`≤ vA`): rounding (`dot_close`, weights bounded by `wMag + eweight`) plus the
    weight perturbation (`n · eweight · vA`). The output half of attention's backward-free
    forward; reused with `wMag = 1` (softmax outputs are probabilities). -/
theorem attnDot_close {nn : Nat} (wF w vcol : Vec nn)
    {eweight wMag vA : ℝ} (heweight : 0 ≤ eweight) (hwMag : 0 ≤ wMag) (_hvA : 0 ≤ vA)
    (hwclose : ∀ k, |wF k - w k| ≤ eweight) (hwabs : ∀ k, |w k| ≤ wMag)
    (hv : ∀ k, |vcol k| ≤ vA) :
    |M.dot wF vcol - ∑ k, w k * vcol k|
      ≤ ((1 + M.u) ^ (nn + 1) - 1) * ((nn : ℝ) * (wMag + eweight) * vA)
        + (nn : ℝ) * eweight * vA := by
  have hG0 : (0 : ℝ) ≤ (1 + M.u) ^ (nn + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by have := M.u_nonneg; linarith))
  have hwFabs : ∀ k, |wF k| ≤ wMag + eweight := by
    intro k
    have h := abs_sub_le (wF k) (w k) 0
    simp only [sub_zero] at h
    have h1 := hwclose k; have h2 := hwabs k; linarith
  have hround : |M.dot wF vcol - ∑ k, wF k * vcol k|
      ≤ ((1 + M.u) ^ (nn + 1) - 1) * ((nn : ℝ) * (wMag + eweight) * vA) := by
    refine (M.dot_close wF vcol).trans ?_
    apply mul_le_mul_of_nonneg_left _ hG0
    calc (∑ k, |wF k * vcol k|) ≤ ∑ _k : Fin nn, (wMag + eweight) * vA := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [abs_mul]
          exact mul_le_mul (hwFabs k) (hv k) (abs_nonneg _) (by linarith)
      _ = (nn : ℝ) * ((wMag + eweight) * vA) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ = (nn : ℝ) * (wMag + eweight) * vA := by ring
  have hpert : |(∑ k, wF k * vcol k) - ∑ k, w k * vcol k| ≤ (nn : ℝ) * eweight * vA := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc (∑ k, |wF k * vcol k - w k * vcol k|) ≤ ∑ _k : Fin nn, eweight * vA := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [show wF k * vcol k - w k * vcol k = (wF k - w k) * vcol k from by ring, abs_mul]
          exact mul_le_mul (hwclose k) (hv k) (abs_nonneg _) heweight
      _ = (nn : ℝ) * (eweight * vA) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      _ = (nn : ℝ) * eweight * vA := by ring
  calc |M.dot wF vcol - ∑ k, w k * vcol k|
      ≤ |M.dot wF vcol - ∑ k, wF k * vcol k|
        + |(∑ k, wF k * vcol k) - ∑ k, w k * vcol k| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add hround hpert

-- ════════════════════════════════════════════════════════════════
-- § The float SDPA and the capstone
-- ════════════════════════════════════════════════════════════════

/-- **Float row-softmax** — the float peer of `rowSoftmax`, per-row `M.softmaxF`. -/
noncomputable def rowSoftmaxF (fexp : ℝ → ℝ) {m n : Nat} (A : Mat m n) : Mat m n :=
  fun i => M.softmaxF fexp (A i)

/-- **Per-row softmax closeness.** Under a coordinatewise logit error `δ` across each row,
    the float row-softmax is within `smErr` of the real one (the `Mat`-space wrap of
    `softmaxF_close_at`). -/
theorem rowSoftmaxF_close (fexp : ℝ → ℝ) {m n : Nat} (At Aa : Mat m n) {eexp δ : ℝ}
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1)
    (hδ : ∀ i k, |At i k - Aa i k| ≤ δ) (i : Fin m) (j : Fin n) :
    |M.rowSoftmaxF fexp At i j - rowSoftmax Aa i j| ≤ smErr M.u eexp δ n :=
  M.softmaxF_close_at fexp (At i) (Aa i) heexp0 heexp1 hfexp hρ1 (fun k => hδ i k) j

/-- **Float scaled dot-product attention** — the rounded peer of `sdpa`: rounded score
    dots `M.dot (Q i) (K j)`, the `1/√d` scale via `M.mul`, per-row `M.softmaxF`, and the
    rounded output dot `M.dot (weightsF i) (V·ⱼ)`. Matches `emitMHSAForward`'s op order. -/
noncomputable def sdpaF (fexp : ℝ → ℝ) {n d : Nat} (Q K V : Mat n d) : Mat n d :=
  let scoresF : Mat n n := fun i j => M.dot (Q i) (K j)
  let scaledF : Mat n n := fun i j => M.mul (1 / Real.sqrt (d : ℝ)) (scoresF i j)
  let weightsF : Mat n n := fun i => M.softmaxF fexp (scaledF i)
  fun i j => M.dot (weightsF i) (fun k => V k j)

theorem sdpaF_eq (fexp : ℝ → ℝ) {n d : Nat} (Q K V : Mat n d) (i : Fin n) (j : Fin d) :
    M.sdpaF fexp Q K V i j
      = M.dot (M.softmaxF fexp (fun jj => M.mul (1 / Real.sqrt (d : ℝ)) (M.dot (Q i) (K jj))))
          (fun k => V k j) := rfl

theorem sdpa_eq {n d : Nat} (Q K V : Mat n d) (i : Fin n) (j : Fin d) :
    sdpa n d Q K V i j
      = ∑ k, softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k
              * V k j := rfl

/-- **THE ATTENTION CAPSTONE — `sdpa_close`.** Each output entry of the float attention
    `sdpaF` is within `attnOutErr` of the real `sdpa n d Q K V`. The four stages chained:
    score `dot_close` (A) → `1/√d` `mul_close` (B) → per-row `softmaxF_close_at` (C) →
    output `dot_close`-at-perturbed-weights (D). All-smooth ⇒ no sign-flip margins; the
    budget is a-posteriori in the supplied magnitudes `qA`/`kA`/`vA`/`scaleA`, proved in
    rounding. With this, the transformer block's MHSA half is `Mat`-space float-bridged;
    pair with the §2d `Vec`-space MLP half for the whole-block fold. -/
theorem sdpa_close (fexp : ℝ → ℝ) {n d : Nat} (Q K V : Mat n d)
    {eexp qA kA vA scaleA : ℝ}
    (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hvA : 0 ≤ vA)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ j k, |K j k| ≤ kA) (hV : ∀ k j, |V k j| ≤ vA)
    (i : Fin n) (j : Fin d) :
    |M.sdpaF fexp Q K V i j - sdpa n d Q K V i j| ≤ M.attnOutErr n d qA kA vA scaleA eexp := by
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  -- (C) the per-row logit error feeding the softmax is the (B) scaled error
  have hδ : ∀ k', |M.mul (1 / Real.sqrt (d : ℝ)) (M.dot (Q i) (K k'))
                - (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i k'|
            ≤ M.attnScaledErr d qA kA scaleA :=
    fun k' => M.attnScaled_close Q K hqA hscaleA hQ hK i k'
  have hwclose : ∀ k,
      |M.softmaxF fexp (fun jj => M.mul (1 / Real.sqrt (d : ℝ)) (M.dot (Q i) (K jj))) k
        - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k|
      ≤ M.attnWeightErr n d qA kA scaleA eexp :=
    fun k => M.softmaxF_close_at fexp _ _ heexp0 heexp1 hfexp hρ1 hδ k
  have heweight : 0 ≤ M.attnWeightErr n d qA kA scaleA eexp :=
    M.smErr_nonneg heexp0 (M.attnScaledErr_nonneg d hqA hkA hscaleA0) hρ1
  -- (D) the output matmul over the perturbed weights
  have key := M.attnDot_close
    (M.softmaxF fexp (fun jj => M.mul (1 / Real.sqrt (d : ℝ)) (M.dot (Q i) (K jj))))
    (softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj))
    (fun k => V k j)
    heweight zero_le_one hvA hwclose (fun k => softmax_abs_le_one _ k) (fun k => hV k j)
  rw [sdpaF_eq, sdpa_eq]
  exact key

-- ════════════════════════════════════════════════════════════════
-- § Attention INPUT-sensitivity (the Lipschitz-through-softmax bound)
--
-- The real-vs-real companion of sdpa_close: how the real `sdpa` output moves when
-- the inputs Q/K/V are each perturbed by `e` per entry. This is the genuinely new
-- analysis (no longer assembly) — the softmax is the only nonlinearity, and its
-- input-sensitivity is `softmax_perturb` (e^(2δ)−1, NOT a derivative bound). With
-- sdpa_close (rounding), this discharges the attention sublayer's FloatClose modulus.
-- ════════════════════════════════════════════════════════════════

/-- Score-matmul input-sensitivity budget: `d·((qA+e)·e + kA·e)`. -/
noncomputable def attnScoreInErr (d : Nat) (qA kA e : ℝ) : ℝ :=
  (d : ℝ) * ((qA + e) * e + kA * e)

/-- Per-row softmax-weight input-sensitivity (`softmax_perturb` at logit shift `scaleA·attnScoreInErr`). -/
noncomputable def attnWeightInErr (d : Nat) (qA kA scaleA e : ℝ) : ℝ :=
  Real.exp (2 * (scaleA * attnScoreInErr d qA kA e)) - 1

/-- Full per-entry SDPA output input-sensitivity: `n·(e + vA·attnWeightInErr)` — V's own
    perturbation (weights ≤ 1) plus the weight perturbation against V (≤ vA). -/
noncomputable def attnOutInErr (n d : Nat) (qA kA vA scaleA e : ℝ) : ℝ :=
  (n : ℝ) * (e + vA * attnWeightInErr d qA kA scaleA e)

/-- **Score-matmul input-sensitivity.** `|(Qt·Ktᵀ)ᵢⱼ − (Qa·Kaᵀ)ᵢⱼ| ≤ attnScoreInErr` — each
    term `QtKt − QaKa = Qt(Kt−Ka) + Ka(Qt−Qa)`, bounded by `(qA+e)·e + kA·e`, summed over `d`. -/
theorem attnScore_input_close {n d : Nat} (Qt Kt Qa Ka : Mat n d) {qA kA e : ℝ}
    (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (he : 0 ≤ e)
    (hQa : ∀ i k, |Qa i k| ≤ qA) (hKa : ∀ j k, |Ka j k| ≤ kA)
    (hQe : ∀ i k, |Qt i k - Qa i k| ≤ e) (hKe : ∀ j k, |Kt j k - Ka j k| ≤ e)
    (i j : Fin n) :
    |Mat.mul Qt (Mat.transpose Kt) i j - Mat.mul Qa (Mat.transpose Ka) i j|
      ≤ attnScoreInErr d qA kA e := by
  have hbound : (∑ k, |Qt i k * Kt j k - Qa i k * Ka j k|) ≤ attnScoreInErr d qA kA e := by
    unfold attnScoreInErr
    calc (∑ k, |Qt i k * Kt j k - Qa i k * Ka j k|)
        ≤ ∑ _k : Fin d, ((qA + e) * e + kA * e) := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [show Qt i k * Kt j k - Qa i k * Ka j k
              = Qt i k * (Kt j k - Ka j k) + Ka j k * (Qt i k - Qa i k) from by ring]
          refine (abs_add_le _ _).trans ?_
          rw [abs_mul, abs_mul]
          have hQtik : |Qt i k| ≤ qA + e := by
            have h := abs_sub_le (Qt i k) (Qa i k) 0; simp only [sub_zero] at h
            linarith [hQe i k, hQa i k]
          have t1 : |Qt i k| * |Kt j k - Ka j k| ≤ (qA + e) * e :=
            mul_le_mul hQtik (hKe j k) (abs_nonneg _) (by linarith)
          have t2 : |Ka j k| * |Qt i k - Qa i k| ≤ kA * e :=
            mul_le_mul (hKa j k) (hQe i k) (abs_nonneg _) hkA
          linarith
      _ = (d : ℝ) * ((qA + e) * e + kA * e) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [matScore_eq, matScore_eq, ← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans hbound

/-- **SDPA INPUT-SENSITIVITY (the attention Lipschitz bound).** When the inputs `Q/K/V` are
    each perturbed by `e` per entry (on magnitudes `qA/kA/vA`), each output entry of the real
    `sdpa` moves by at most `attnOutInErr`. The chain: score sensitivity (`attnScore_input_close`)
    → `1/√d` scale → **per-row softmax sensitivity `softmax_perturb`** (the `e^(2δ)−1` bound, the
    only nonlinear step, no derivatives) → output matmul (V's own shift + the weight shift). This
    is the piece `sdpa_close` was missing: with it, the attention sublayer is a full `FloatClose`. -/
theorem sdpa_input_close {n d : Nat} (Qt Kt Vt Qa Ka Va : Mat n d)
    {qA kA vA scaleA e : ℝ}
    (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hvA : 0 ≤ vA) (he : 0 ≤ e)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA)
    (hQa : ∀ i k, |Qa i k| ≤ qA) (hKa : ∀ j k, |Ka j k| ≤ kA) (hVa : ∀ k j, |Va k j| ≤ vA)
    (hQe : ∀ i k, |Qt i k - Qa i k| ≤ e) (hKe : ∀ j k, |Kt j k - Ka j k| ≤ e)
    (hVe : ∀ k j, |Vt k j - Va k j| ≤ e) (i : Fin n) (j : Fin d) :
    |sdpa n d Qt Kt Vt i j - sdpa n d Qa Ka Va i j| ≤ attnOutInErr n d qA kA vA scaleA e := by
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  set Δsc : ℝ := scaleA * attnScoreInErr d qA kA e with hΔ
  have hΔ0 : 0 ≤ Δsc := by
    rw [hΔ]; refine mul_nonneg hscaleA0 ?_
    unfold attnScoreInErr
    exact mul_nonneg (Nat.cast_nonneg d)
      (by have := mul_nonneg (by linarith : (0:ℝ) ≤ qA + e) he
          have := mul_nonneg hkA he; linarith)
  -- (B) the scaled per-row logit shift
  have hsc : ∀ k', |(1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i k'
                  - (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i k'| ≤ Δsc := by
    intro k'
    rw [← mul_sub, abs_mul]
    exact mul_le_mul hscaleA (attnScore_input_close Qt Kt Qa Ka hqA hkA he hQa hKa hQe hKe i k')
      (abs_nonneg _) hscaleA0
  -- (C) the per-row softmax sensitivity
  have hw : ∀ k, |softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k
                - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i jj) k|
              ≤ Real.exp (2 * Δsc) - 1 :=
    fun k => softmax_perturb _ _ hsc k
  -- (D) output: V's own shift (weights ≤ 1) + the weight shift against V
  rw [sdpa_eq, sdpa_eq, ← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  have hbound : (∑ k, |softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k * Vt k j
                     - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i jj) k * Va k j|)
              ≤ attnOutInErr n d qA kA vA scaleA e := by
    unfold attnOutInErr attnWeightInErr
    rw [← hΔ]
    calc (∑ k, |softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k * Vt k j
              - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i jj) k * Va k j|)
        ≤ ∑ _k : Fin n, (e + vA * (Real.exp (2 * Δsc) - 1)) := by
          refine Finset.sum_le_sum fun k _ => ?_
          rw [show softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k * Vt k j
                 - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i jj) k * Va k j
              = softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k * (Vt k j - Va k j)
                + Va k j * (softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k
                          - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i jj) k)
              from by ring]
          refine (abs_add_le _ _).trans ?_
          rw [abs_mul, abs_mul]
          have he1 : 0 ≤ Real.exp (2 * Δsc) - 1 := by
            have := Real.add_one_le_exp (2 * Δsc); linarith [hΔ0]
          have t1 : |softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k|
                    * |Vt k j - Va k j| ≤ 1 * e :=
            mul_le_mul (softmax_abs_le_one _ k) (hVe k j) (abs_nonneg _) (by norm_num)
          have t2 : |Va k j| * |softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qt (Mat.transpose Kt) i jj) k
                              - softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Qa (Mat.transpose Ka) i jj) k|
                    ≤ vA * (Real.exp (2 * Δsc) - 1) :=
            mul_le_mul (hVa k j) (hw k) (abs_nonneg _) hvA
          linarith
      _ = (n : ℝ) * (e + vA * (Real.exp (2 * Δsc) - 1)) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact hbound

/-- **Softmax rows sum to 1** (`n ≥ 1`) — `∑ exp(zₖ)/S = S/S = 1`. -/
theorem softmax_sum_one {n : ℕ} (hn : 0 < n) (z : Vec n) : ∑ k, softmax n z k = 1 := by
  have hS : (0 : ℝ) < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨⟨0, hn⟩, Finset.mem_univ _⟩
  have hk : ∀ k, softmax n z k = Real.exp (z k) / ∑ j, Real.exp (z j) := fun k => rfl
  simp_rw [hk, div_eq_mul_inv, ← Finset.sum_mul]
  rw [mul_inv_cancel₀ hS.ne']

/-- **Attention is magnitude-stable** — `|sdpa Q K V i j| ≤ A` whenever `|V| ≤ A`: the output
    is a convex combination of `V`'s rows (softmax weights are a probability distribution, by
    `softmax_sum_one` + nonnegativity), so it never exceeds `V`'s magnitude. The attention
    analogue of `relu`/`maxpool` magnitude-stability — what makes the sublayer compose to depth
    with a FIXED `A` (no magnitude growth). -/
theorem sdpa_abs_le {n d : Nat} (hn : 0 < n) (Q K V : Mat n d) {A : ℝ}
    (hV : ∀ k j, |V k j| ≤ A) (i : Fin n) (j : Fin d) :
    |sdpa n d Q K V i j| ≤ A := by
  rw [sdpa_eq]
  have hw0 : ∀ k, 0 ≤ softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k :=
    fun k => div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun jj _ => (Real.exp_pos _).le)
  calc |∑ k, softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k * V k j|
      ≤ ∑ k, |softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k * V k j| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k, softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k * A := by
        refine Finset.sum_le_sum fun k _ => ?_
        rw [abs_mul, abs_of_nonneg (hw0 k)]
        exact mul_le_mul_of_nonneg_left (hV k j) (hw0 k)
    _ = (∑ k, softmax n (fun jj => (1 / Real.sqrt (d : ℝ)) * Mat.mul Q (Mat.transpose K) i jj) k) * A := by
        rw [Finset.sum_mul]
    _ = 1 * A := by rw [softmax_sum_one hn]
    _ = A := one_mul A

end FloatModel

end Proofs
