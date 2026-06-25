import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.FloatBridge

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

end FloatModel

end Proofs
