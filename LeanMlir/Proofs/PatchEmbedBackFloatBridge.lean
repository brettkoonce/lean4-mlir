import LeanMlir.Proofs.MhsaBackFloatBridge

/-! # ℝ→Float32 bridge: the ViT PATCH-EMBED backward (A3 §1f — the last whole-net endpoint)

A3 (planning/a3_backward_deepnet_assembly.md §1f): the one concrete endpoint left for the ViT whole-net
backward (`vit_grad_floatBridges`). The certified patch-embedding input gradient
(`patchEmbed_input_grad_formula`, `Attention.lean`) is the transposed convolution: at input pixel
`idx` (decoded to channel `c`, row `hh`, col `ww`),

  ∑ p:Fin N, ∑ kh ∑ kw, (if patch p offset (kh,kw) covers (hh,ww) then ∑ d, W_conv[d,c,kh,kw]·dy[p+1,d] else 0)

a guarded triple-sum of dots, **linear in the cotangent `dy`**. So — like every other backward op here —
its `FloatClose` modulus is the rounding budget plus the real magnitude at the input-error `e`. The
float model rounds each inner dot (`dot_close`, fan-in `D`) and the three sums (`reduction_close`, nested
over `N`/`patchSize`/`patchSize`); the guard is a fixed index predicate (identical in float and real), so
the closeness threads cleanly. `floatBridges_patchEmbedBack` discharges the whole-net's `hPatch`
hypothesis — the ViT whole-net backward is then fully concrete (head/cls-slice/patch-embed) over the
supplied LN/block backwards. A3 = gradient *closeness* at a smooth point (NOT descent).
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § A generic triple-sum magnitude bound
-- ════════════════════════════════════════════════════════════════

/-- `|∑ p ∑ kh ∑ kw t| ≤ NN·P·P·c` when every term is `≤ c`. -/
theorem triple_sum_abs_le {NN P : Nat} (t : Fin NN → Fin P → Fin P → ℝ) {c : ℝ}
    (ht : ∀ p kh kw, |t p kh kw| ≤ c) :
    |∑ p, ∑ kh, ∑ kw, t p kh kw| ≤ (NN : ℝ) * ((P : ℝ) * ((P : ℝ) * c)) := by
  calc |∑ p, ∑ kh, ∑ kw, t p kh kw|
      ≤ ∑ p, |∑ kh, ∑ kw, t p kh kw| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _p : Fin NN, ((P : ℝ) * ((P : ℝ) * c)) := Finset.sum_le_sum fun p _ => by
        calc |∑ kh, ∑ kw, t p kh kw| ≤ ∑ kh, |∑ kw, t p kh kw| := Finset.abs_sum_le_sum_abs _ _
          _ ≤ ∑ _kh : Fin P, ((P : ℝ) * c) := Finset.sum_le_sum fun kh _ => by
              calc |∑ kw, t p kh kw| ≤ ∑ kw, |t p kh kw| := Finset.abs_sum_le_sum_abs _ _
                _ ≤ ∑ _kw : Fin P, c := Finset.sum_le_sum fun kw _ => ht p kh kw
                _ = (P : ℝ) * c := by
                    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          _ = (P : ℝ) * ((P : ℝ) * c) := by
              rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    _ = (NN : ℝ) * ((P : ℝ) * ((P : ℝ) * c)) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- One rounded-reduction budget level: the `reduction_close` bound `γₙ·n·(Mr+ef) + n·ef`. -/
noncomputable def redErr (u : ℝ) (n : Nat) (Mr ef : ℝ) : ℝ :=
  ((1 + u) ^ (n + 1) - 1) * ((n : ℝ) * (Mr + ef)) + (n : ℝ) * ef

-- ════════════════════════════════════════════════════════════════
-- § The float patch-embed backward + budgets
-- ════════════════════════════════════════════════════════════════

/-- **Float patch-embed backward** — the rounded peer of `patchEmbed_input_grad_formula`: each inner
    `∑ d` is a rounded dot `M.dot`, the three patch/offset sums are nested `M.sum`. Same guard. -/
noncomputable def FloatModel.patchEmbedBackF (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (dy : Vec ((N + 1) * D)) : Vec (ic * H * W) :=
  fun idx_in =>
    let c  := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).1
    let hh := (finProdFinEquiv.symm (finProdFinEquiv.symm idx_in).1).2
    let ww := (finProdFinEquiv.symm idx_in).2
    M.sum fun p : Fin N => M.sum fun kh : Fin patchSize => M.sum fun kw : Fin patchSize =>
      let W' := W / patchSize
      let h' := p.val / W'
      let w' := p.val % W'
      if _h_match : h' * patchSize + kh.val = hh.val ∧ w' * patchSize + kw.val = ww.val then
        M.dot (fun d : Fin D => W_conv d c kh kw) (fun d : Fin D => dy (finProdFinEquiv (p.succ, d)))
      else 0

/-- Inner-dot rounding budget (`dot_close`, fan-in `D`, magnitudes `wconv`/`A`). -/
noncomputable def FloatModel.patchEmbedDotErr (M : FloatModel) (D : Nat) (wconv A : ℝ) : ℝ :=
  ((1 + M.u) ^ (D + 1) - 1) * ((D : ℝ) * wconv * A)

/-- Full nested patch-embed rounding budget: dot rounding folded through the `kw`/`kh`/`p` reductions. -/
noncomputable def FloatModel.patchEmbedBackBudget (M : FloatModel) (patchSize N D : Nat)
    (wconv A : ℝ) : ℝ :=
  let Mg := (D : ℝ) * wconv * A
  let ekw := redErr M.u patchSize Mg (M.patchEmbedDotErr D wconv A)
  let ekh := redErr M.u patchSize ((patchSize : ℝ) * Mg) ekw
  redErr M.u N ((patchSize : ℝ) * ((patchSize : ℝ) * Mg)) ekh

-- ════════════════════════════════════════════════════════════════
-- § Real magnitude + linearity (the formula is linear in the cotangent)
-- ════════════════════════════════════════════════════════════════

/-- **Real patch-embed-back magnitude** — `|formula z idx| ≤ N·P²·D·wconv·Z` for `|z| ≤ Z`. -/
theorem patchEmbed_formula_abs_le (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) {wconv Z : ℝ} (hwconv0 : 0 ≤ wconv) (hZ : 0 ≤ Z)
    (hwconv : ∀ d c kh kw, |W_conv d c kh kw| ≤ wconv) (z : Vec ((N + 1) * D)) (hz : ∀ k, |z k| ≤ Z)
    (idx : Fin (ic * H * W)) :
    |patchEmbed_input_grad_formula ic H W patchSize N D W_conv z idx|
      ≤ (N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * Z))) := by
  unfold patchEmbed_input_grad_formula
  refine triple_sum_abs_le _ (fun p kh kw => ?_)
  by_cases hg : (p.val / (W / patchSize)) * patchSize + kh.val
        = ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2).val
      ∧ (p.val % (W / patchSize)) * patchSize + kw.val = ((finProdFinEquiv.symm idx).2).val
  · simp only [dif_pos hg]
    calc |∑ d, W_conv d _ kh kw * z (finProdFinEquiv (p.succ, d))|
        ≤ ∑ d, |W_conv d _ kh kw * z (finProdFinEquiv (p.succ, d))| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _d : Fin D, (wconv * Z) := Finset.sum_le_sum fun d _ => by
          rw [abs_mul]; exact mul_le_mul (hwconv _ _ kh kw) (hz _) (abs_nonneg _) hwconv0
      _ = (D : ℝ) * wconv * Z := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  · simp only [dif_neg hg, abs_zero]
    exact mul_nonneg (mul_nonneg (Nat.cast_nonneg D) hwconv0) hZ

/-- **Real patch-embed-back is linear in `dy`** — `|formula dyt − formula dya| ≤ N·P²·D·wconv·e`. -/
theorem patchEmbed_formula_sub_abs_le (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) {wconv e : ℝ} (hwconv0 : 0 ≤ wconv) (he : 0 ≤ e)
    (hwconv : ∀ d c kh kw, |W_conv d c kh kw| ≤ wconv) (dyt dya : Vec ((N + 1) * D))
    (hd : ∀ k, |dyt k - dya k| ≤ e) (idx : Fin (ic * H * W)) :
    |patchEmbed_input_grad_formula ic H W patchSize N D W_conv dyt idx
        - patchEmbed_input_grad_formula ic H W patchSize N D W_conv dya idx|
      ≤ (N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * e))) := by
  unfold patchEmbed_input_grad_formula
  simp only [← Finset.sum_sub_distrib]
  refine triple_sum_abs_le _ (fun p kh kw => ?_)
  by_cases hg : (p.val / (W / patchSize)) * patchSize + kh.val
        = ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2).val
      ∧ (p.val % (W / patchSize)) * patchSize + kw.val = ((finProdFinEquiv.symm idx).2).val
  · simp only [dif_pos hg, ← Finset.sum_sub_distrib]
    calc |∑ d, (W_conv d _ kh kw * dyt (finProdFinEquiv (p.succ, d))
              - W_conv d _ kh kw * dya (finProdFinEquiv (p.succ, d)))|
        ≤ ∑ d, |W_conv d _ kh kw * dyt (finProdFinEquiv (p.succ, d))
              - W_conv d _ kh kw * dya (finProdFinEquiv (p.succ, d))| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _d : Fin D, (wconv * e) := Finset.sum_le_sum fun d _ => by
          rw [← mul_sub, abs_mul]
          exact mul_le_mul (hwconv _ _ kh kw) (hd _) (abs_nonneg _) hwconv0
      _ = (D : ℝ) * wconv * e := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  · simp only [dif_neg hg, sub_zero, abs_zero]
    exact mul_nonneg (mul_nonneg (Nat.cast_nonneg D) hwconv0) he

-- ════════════════════════════════════════════════════════════════
-- § The rounding closeness (nested reduction_close) + the FloatClose
-- ════════════════════════════════════════════════════════════════

/-- **Patch-embed-back rounding closeness** — the deployed float backward is within
    `patchEmbedBackBudget` of the certified one (the inner-dot `dot_close` folded through the three
    nested `reduction_close`s over `kw`/`kh`/`p`). -/
theorem patchEmbedBack_round_close (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) {wconv A : ℝ} (hwconv0 : 0 ≤ wconv) (hA : 0 ≤ A)
    (hwconv : ∀ d c kh kw, |W_conv d c kh kw| ≤ wconv) (dy : Vec ((N + 1) * D)) (hdy : ∀ k, |dy k| ≤ A)
    (idx : Fin (ic * H * W)) :
    |M.patchEmbedBackF ic H W patchSize N D W_conv dy idx
        - patchEmbed_input_grad_formula ic H W patchSize N D W_conv dy idx|
      ≤ M.patchEmbedBackBudget patchSize N D wconv A := by
  set Mg : ℝ := (D : ℝ) * wconv * A with hMg
  set edot : ℝ := M.patchEmbedDotErr D wconv A with hedot
  have hMg0 : 0 ≤ Mg := mul_nonneg (mul_nonneg (Nat.cast_nonneg D) hwconv0) hA
  have hedot0 : 0 ≤ edot := by
    rw [hedot, FloatModel.patchEmbedDotErr]
    exact mul_nonneg (sub_nonneg.mpr (one_le_pow₀ (by have := M.u_nonneg; linarith))) hMg0
  -- the leaf, as a function of (p, kh, kw): float `dite` of M.dot, real `dite` of the ℝ sum
  set gF : Fin N → Fin patchSize → Fin patchSize → ℝ := fun p kh kw =>
    if _h : (p.val / (W / patchSize)) * patchSize + kh.val
          = ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2).val
        ∧ (p.val % (W / patchSize)) * patchSize + kw.val = ((finProdFinEquiv.symm idx).2).val then
      M.dot (fun d : Fin D => W_conv d (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1 kh kw)
        (fun d : Fin D => dy (finProdFinEquiv (p.succ, d)))
    else 0 with hgF
  set gR : Fin N → Fin patchSize → Fin patchSize → ℝ := fun p kh kw =>
    if _h : (p.val / (W / patchSize)) * patchSize + kh.val
          = ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2).val
        ∧ (p.val % (W / patchSize)) * patchSize + kw.val = ((finProdFinEquiv.symm idx).2).val then
      ∑ d : Fin D, W_conv d (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1 kh kw
        * dy (finProdFinEquiv (p.succ, d))
    else 0 with hgR
  -- leaf closeness + leaf magnitude
  have hleafclose : ∀ p kh kw, |gF p kh kw - gR p kh kw| ≤ edot := by
    intro p kh kw
    rw [hgF, hgR]
    by_cases hg : (p.val / (W / patchSize)) * patchSize + kh.val
          = ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2).val
        ∧ (p.val % (W / patchSize)) * patchSize + kw.val = ((finProdFinEquiv.symm idx).2).val
    · simp only [dif_pos hg, hedot, FloatModel.patchEmbedDotErr]
      refine (M.dot_close _ _).trans ?_
      apply mul_le_mul_of_nonneg_left _ (sub_nonneg.mpr (one_le_pow₀ (by have := M.u_nonneg; linarith)))
      calc (∑ d, |W_conv d _ kh kw * dy (finProdFinEquiv (p.succ, d))|)
          ≤ ∑ _d : Fin D, (wconv * A) := Finset.sum_le_sum fun d _ => by
            rw [abs_mul]; exact mul_le_mul (hwconv _ _ kh kw) (hdy _) (abs_nonneg _) hwconv0
        _ = (D : ℝ) * wconv * A := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
    · simp only [dif_neg hg, sub_zero, abs_zero]; exact hedot0
  have hleafmag : ∀ p kh kw, |gR p kh kw| ≤ Mg := by
    intro p kh kw
    rw [hgR]
    by_cases hg : (p.val / (W / patchSize)) * patchSize + kh.val
          = ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2).val
        ∧ (p.val % (W / patchSize)) * patchSize + kw.val = ((finProdFinEquiv.symm idx).2).val
    · simp only [dif_pos hg, hMg]
      calc |∑ d, W_conv d _ kh kw * dy (finProdFinEquiv (p.succ, d))|
          ≤ ∑ d, |W_conv d _ kh kw * dy (finProdFinEquiv (p.succ, d))| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _d : Fin D, (wconv * A) := Finset.sum_le_sum fun d _ => by
            rw [abs_mul]; exact mul_le_mul (hwconv _ _ kh kw) (hdy _) (abs_nonneg _) hwconv0
        _ = (D : ℝ) * wconv * A := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
    · simp only [dif_neg hg, abs_zero]; exact hMg0
  -- partial real magnitudes
  have hkwmag : ∀ p kh, |∑ kw, gR p kh kw| ≤ (patchSize : ℝ) * Mg := by
    intro p kh
    calc |∑ kw, gR p kh kw| ≤ ∑ kw, |gR p kh kw| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _kw : Fin patchSize, Mg := Finset.sum_le_sum fun kw _ => hleafmag p kh kw
      _ = (patchSize : ℝ) * Mg := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hkhmag : ∀ p, |∑ kh, ∑ kw, gR p kh kw| ≤ (patchSize : ℝ) * ((patchSize : ℝ) * Mg) := by
    intro p
    calc |∑ kh, ∑ kw, gR p kh kw| ≤ ∑ kh, |∑ kw, gR p kh kw| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _kh : Fin patchSize, ((patchSize : ℝ) * Mg) := Finset.sum_le_sum fun kh _ => hkwmag p kh
      _ = (patchSize : ℝ) * ((patchSize : ℝ) * Mg) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- nested reduction closeness
  have hkw : ∀ p kh, |M.sum (fun kw => gF p kh kw) - ∑ kw, gR p kh kw|
      ≤ redErr M.u patchSize Mg edot := fun p kh =>
    reduction_close M (fun kw => gF p kh kw) (fun kw => gR p kh kw)
      (fun kw => hleafclose p kh kw) (fun kw => hleafmag p kh kw)
  have hkh : ∀ p, |M.sum (fun kh => M.sum (fun kw => gF p kh kw)) - ∑ kh, ∑ kw, gR p kh kw|
      ≤ redErr M.u patchSize ((patchSize : ℝ) * Mg) (redErr M.u patchSize Mg edot) := fun p =>
    reduction_close M (fun kh => M.sum (fun kw => gF p kh kw)) (fun kh => ∑ kw, gR p kh kw)
      (fun kh => hkw p kh) (fun kh => hkwmag p kh)
  have hp : |M.sum (fun p => M.sum (fun kh => M.sum (fun kw => gF p kh kw)))
        - ∑ p, ∑ kh, ∑ kw, gR p kh kw|
      ≤ redErr M.u N ((patchSize : ℝ) * ((patchSize : ℝ) * Mg))
          (redErr M.u patchSize ((patchSize : ℝ) * Mg) (redErr M.u patchSize Mg edot)) :=
    reduction_close M (fun p => M.sum (fun kh => M.sum (fun kw => gF p kh kw)))
      (fun p => ∑ kh, ∑ kw, gR p kh kw) (fun p => hkh p) (fun p => hkhmag p)
  -- conclude
  show |M.patchEmbedBackF ic H W patchSize N D W_conv dy idx
      - patchEmbed_input_grad_formula ic H W patchSize N D W_conv dy idx|
      ≤ M.patchEmbedBackBudget patchSize N D wconv A
  rw [FloatModel.patchEmbedBackBudget]
  simp only [← hMg, ← hedot]
  exact hp

/-- **Patch-embed backward is `FloatClose`** — magnitude `N·P²·D·wconv·A + rounding`, modulus the
    rounding budget + the real magnitude at `e` (the formula is linear in the cotangent). -/
theorem floatClose_patchEmbedBack (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) {wconv A : ℝ} (hwconv0 : 0 ≤ wconv) (hA : 0 ≤ A)
    (hD : 0 < D) (hwconv : ∀ d c kh kw, |W_conv d c kh kw| ≤ wconv) :
    FloatClose A
      ((N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * A)))
        + M.patchEmbedBackBudget patchSize N D wconv A)
      (patchEmbed_input_grad_formula ic H W patchSize N D W_conv)
      (M.patchEmbedBackF ic H W patchSize N D W_conv)
      (fun e => M.patchEmbedBackBudget patchSize N D wconv A
        + (N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * e)))) := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hreal := patchEmbed_formula_abs_le ic H W patchSize N D W_conv hwconv0 hA hwconv v hv idx
    have hround := patchEmbedBack_round_close M ic H W patchSize N D W_conv hwconv0 hA hwconv v hv idx
    have hB0 : 0 ≤ M.patchEmbedBackBudget patchSize N D wconv A := (abs_nonneg _).trans hround
    refine ⟨le_trans hreal (by linarith), ?_⟩
    calc |M.patchEmbedBackF ic H W patchSize N D W_conv v idx|
        ≤ |M.patchEmbedBackF ic H W patchSize N D W_conv v idx
            - patchEmbed_input_grad_formula ic H W patchSize N D W_conv v idx|
          + |patchEmbed_input_grad_formula ic H W patchSize N D W_conv v idx| := by
          simpa using abs_sub_le (M.patchEmbedBackF ic H W patchSize N D W_conv v idx)
            (patchEmbed_input_grad_formula ic H W patchSize N D W_conv v idx) 0
      _ ≤ M.patchEmbedBackBudget patchSize N D wconv A
          + (N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * A))) :=
          add_le_add hround hreal
      _ = (N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * A)))
          + M.patchEmbedBackBudget patchSize N D wconv A := by ring
  · have he0 : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, Nat.mul_pos (Nat.succ_pos N) hD⟩)
    have hround := patchEmbedBack_round_close M ic H W patchSize N D W_conv hwconv0 hA hwconv vt hvt idx
    have hsens := patchEmbed_formula_sub_abs_le ic H W patchSize N D W_conv hwconv0 he0 hwconv vt va hd idx
    calc |M.patchEmbedBackF ic H W patchSize N D W_conv vt idx
            - patchEmbed_input_grad_formula ic H W patchSize N D W_conv va idx|
        ≤ |M.patchEmbedBackF ic H W patchSize N D W_conv vt idx
            - patchEmbed_input_grad_formula ic H W patchSize N D W_conv vt idx|
          + |patchEmbed_input_grad_formula ic H W patchSize N D W_conv vt idx
            - patchEmbed_input_grad_formula ic H W patchSize N D W_conv va idx| := abs_sub_le _ _ _
      _ ≤ M.patchEmbedBackBudget patchSize N D wconv A
          + (N : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * ((D : ℝ) * wconv * e))) :=
          add_le_add hround hsens

/-- **The patch-embed backward float-bridges** — discharges the `hPatch` hypothesis of
    `vit_grad_floatBridges`, making the whole ViT input-gradient backward fully concrete. -/
theorem floatBridges_patchEmbedBack (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) {wconv : ℝ} (hwconv0 : 0 ≤ wconv)
    (himg : 0 < ic * H * W) (hD : 0 < D)
    (hwconv : ∀ d c kh kw, |W_conv d c kh kw| ≤ wconv) :
    FloatBridges (patchEmbed_input_grad_formula ic H W patchSize N D W_conv) := by
  intro A hA
  exact ⟨_, _, _,
    (floatClose_patchEmbedBack M ic H W patchSize N D W_conv hwconv0 hA hD hwconv).cod_nonneg hA himg,
    floatClose_patchEmbedBack M ic H W patchSize N D W_conv hwconv0 hA hD hwconv⟩

/-- **THE FULLY-CONCRETE ViT WHOLE-NET BACKWARD.** `vit_grad_floatBridges` with the patch-embed
    endpoint discharged by `floatBridges_patchEmbedBack` — EVERY endpoint (patch-embed, cls-slice,
    head) is now concrete; only the per-block backwards and the final LN are supplied as `FloatBridges`
    (dischargeable by `floatBridges_vitBlockBack` / `bnBack`), exactly as `r34_grad_floatBridges` supplies
    its blocks. The deployed float ViT input-gradient backward ≈ the certified ℝ gradient, end to end. -/
theorem vit_grad_floatBridges_concrete (M : FloatModel)
    (ic H W patchSize N D nClasses : Nat) (Wcls : Mat D nClasses) (finalLNBack : Vec D → Vec D)
    (blockBacks : List (Vec ((N + 1) * D) → Vec ((N + 1) * D)))
    (W_conv : Kernel4 D ic patchSize patchSize)
    {w' wconv : ℝ} (hw' : 0 ≤ w') (hnc : 0 < nClasses) (himg : 0 < ic * H * W) (hD : 0 < D)
    (hwconv0 : 0 ≤ wconv) (hWcls : ∀ i j, |Wcls i j| ≤ w')
    (hwconv : ∀ d c kh kw, |W_conv d c kh kw| ≤ wconv)
    (hFinalLN : FloatBridges finalLNBack) (hblocks : ∀ f ∈ blockBacks, FloatBridges f) :
    FloatBridges (vitGradFlat Wcls finalLNBack blockBacks
      (patchEmbed_input_grad_formula ic H W patchSize N D W_conv)) :=
  vit_grad_floatBridges M Wcls finalLNBack blockBacks _ hw' hnc hWcls hFinalLN hblocks
    (floatBridges_patchEmbedBack M ic H W patchSize N D W_conv hwconv0 himg hD hwconv)

end Proofs
