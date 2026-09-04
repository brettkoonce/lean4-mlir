import LeanMlir.Proofs.Float.CnnBackFloatBridge

/-! # ℝ→Float32 bridge for the 3×3/s2 MAX-POOL BACKWARD — the accumulating scatter

⛔⛔ **Why this file exists: `r34InputGrad` was the reverse of the wrong pool.** The committed
ResNet-34 forward (`resnet34Forward_full_pc`) pools with `maxPool3s2Flat` — He et al.'s 3×3
stride-2 stem pool, restored 2026-08-03 — but the whole-net backward used `maxPoolFlatBack`, the
**2×2** pool's backward, while its docstring claimed to be "the exact reverse of
`resnet34Forward_full_pc`". `MaxPool3s2.lean`'s own header warns that the two have the same TYPE
and are different functions; nothing had forced the two statements to unify until the whole-net
CERTIFIED TIE needed them to. (`imagenet_specs_drift_from_twins`; the same failure mode as
ConvNeXt's stale head-LayerNorm slot, `planning/float_budget_numbers.md` §3.3(b).)

⭐⭐ **The structural difference, and it is the only one: `maxPool2`'s windows TILE, so its
backward is a LOOKUP and is exact in float. 3×3/s2 windows OVERLAP, so an input cell can be the
argmax of up to FOUR outputs and the backward must ACCUMULATE** — a reduction, with rounding, and
with a magnitude that is no longer envelope-preserving. `Maps.maxPoolBack` carries `Ā ↦ Ā`;
`Maps.maxPool3s2Back` cannot, and `4` is the constant.

⭐ **The `4` is proved, not assumed** (`maxPool3s2Back_mask_sum_abs_le`). `win3Row_mem_le_two`
(`MaxPool3s2.lean`) says an input row lies in at most two windows — `p/2` and `(p+1)/2` — so the
fibre of the smooth-point reindex map is contained in a `1 × 2 × 2` box and the masked sum has at
most four live terms. ⛔ Taking the trivial `c·h·w` bound instead (the shape
`floatClose_broadcastBack` settles for) costs **six orders** on the r34 backward number, which
lands it at `10²⁵¹` against §3.7(a)'s shape-dependent `norm_num` wall at ~`10²⁵³`. The count is
not tidiness.

The float peer is the masked reduction `fl(Σ_k [σ(k) = idx] · dy k)` — the same modelling choice
`FloatModel.broadcastBackFlatF` makes for the squeeze-excite gate's spatial reduce, and the same
`M.sum_close` Higham factor, at `n = c·h·w` (the length the kernel reduces over) but with the
sum-of-magnitudes bounded by the fibre count rather than by the length.
-/

namespace Proofs

open Finset

-- ════════════════════════════════════════════════════════════════
-- § Counting the fibre: at most 2 rows, 2 columns, 1 channel
-- ════════════════════════════════════════════════════════════════

/-- At most one `Fin n` carries a given `val`, so a `val`-indicator sums to at most `1`. -/
theorem sum_ite_val_eq_le_one {n : Nat} (k : Nat) :
    ∑ x : Fin n, (if x.val = k then (1:ℝ) else 0) ≤ 1 := by
  have hcard : (Finset.univ.filter (fun x : Fin n => x.val = k)).card ≤ 1 := by
    rw [Finset.card_le_one]
    intro a ha b hb
    simp only [Finset.mem_filter] at ha hb
    exact Fin.ext (ha.2.trans hb.2.symm)
  calc ∑ x : Fin n, (if x.val = k then (1:ℝ) else 0)
      = ((Finset.univ.filter (fun x : Fin n => x.val = k)).card : ℝ) := by
        simp [Finset.sum_boole]
    _ ≤ 1 := by exact_mod_cast hcard

/-- The disjunction `win3Row_mem_le_two` produces pins `x.val` to one of two values, so its
    indicator sums to at most `2`. -/
theorem sum_ite_twoval_le_two {n : Nat} (k₁ k₂ : Nat) :
    ∑ x : Fin n, (if x.val = k₁ ∨ 2 * x.val = k₂ then (1:ℝ) else 0) ≤ 2 := by
  have hle : ∀ x : Fin n,
      (if x.val = k₁ ∨ 2 * x.val = k₂ then (1:ℝ) else 0)
        ≤ (if x.val = k₁ then (1:ℝ) else 0) + (if x.val = k₂ / 2 then (1:ℝ) else 0) := by
    intro x
    by_cases h1 : x.val = k₁
    · simp [h1]; positivity
    · by_cases h2 : 2 * x.val = k₂
      · have hd : x.val = k₂ / 2 := by omega
        rw [if_pos (Or.inr h2), if_neg h1, if_pos hd]
        norm_num
      · rw [if_neg (by tauto), if_neg h1, zero_add]
        split <;> norm_num
  calc ∑ x : Fin n, (if x.val = k₁ ∨ 2 * x.val = k₂ then (1:ℝ) else 0)
      ≤ ∑ x : Fin n, ((if x.val = k₁ then (1:ℝ) else 0) + (if x.val = k₂ / 2 then (1:ℝ) else 0)) :=
        Finset.sum_le_sum (fun x _ => hle x)
    _ = (∑ x : Fin n, (if x.val = k₁ then (1:ℝ) else 0))
          + ∑ x : Fin n, (if x.val = k₂ / 2 then (1:ℝ) else 0) := Finset.sum_add_distrib
    _ ≤ 1 + 1 := add_le_add (sum_ite_val_eq_le_one _) (sum_ite_val_eq_le_one _)
    _ = 2 := by norm_num

/-- ⭐ **The overlap count, on the row axis.** At most two output rows have a 3×3 window
    containing input row `hi` (`win3Row_mem_le_two`), so the row indicator sums to `≤ 2`. The
    column peer is the same statement at `w`. -/
theorem sum_ite_win3Row_le_two {h : Nat} (hi : Fin (2 * h)) :
    ∑ ho : Fin h, (if ∃ a : Fin 3, win3RowInv ho a = hi then (1:ℝ) else 0) ≤ 2 := by
  refine le_trans (Finset.sum_le_sum (fun ho _ => ?_))
    (sum_ite_twoval_le_two (n := h) (hi.val / 2) (hi.val + 1))
  by_cases hmem : ∃ a : Fin 3, win3RowInv ho a = hi
  · simp only [if_pos hmem]
    rw [if_pos (win3Row_mem_le_two hi ho hmem)]
  · simp only [if_neg hmem]
    positivity

/-- Column peer of `sum_ite_win3Row_le_two` (`win3ColInv` is `win3RowInv` at `w`). -/
theorem sum_ite_win3Col_le_two {w : Nat} (wi : Fin (2 * w)) :
    ∑ wo : Fin w, (if ∃ b : Fin 3, win3ColInv wo b = wi then (1:ℝ) else 0) ≤ 2 :=
  sum_ite_win3Row_le_two wi

/-- Row-major re-indexing of a `Fin (c*h*w)` sum as a triple sum — the shape every
    `maxPool3s2` statement is written in. -/
theorem sum_flat3 {c h w : Nat} (g : Fin (c*h*w) → ℝ) :
    ∑ k : Fin (c*h*w), g k
      = ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          g (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) := by
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin (c*h) × Fin w ≃ Fin (c*h*w)) g,
      Fintype.sum_prod_type,
      ← Equiv.sum_comp (finProdFinEquiv : Fin c × Fin h ≃ Fin (c*h))
        (fun a => ∑ b : Fin w, g (finProdFinEquiv (a, b))),
      Fintype.sum_prod_type]

-- ════════════════════════════════════════════════════════════════
-- § The fibre bound: at most FOUR live terms
-- ════════════════════════════════════════════════════════════════

/-- Reading the smooth-point reindex backwards: if output `(co,ho,wo)`'s argmax is input
    `(ci,hi,wi)`, then the channel matches exactly and `hi`/`wi` lie in `(ho,wo)`'s 3×3 window.
    Two applications of `finProdFinEquiv.injective`; the argmax offsets are the witnesses. -/
theorem maxPool3s2_reindex_decompose {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w))
    (hcond : maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
      = finProdFinEquiv (finProdFinEquiv (ci, hi), wi)) :
    co = ci ∧ (∃ a : Fin 3, win3RowInv ho a = hi) ∧ (∃ b : Fin 3, win3ColInv wo b = wi) := by
  rw [maxPool3s2LocalReindex] at hcond
  simp only [Equiv.symm_apply_apply] at hcond
  have h1 := Prod.mk.injEq _ _ _ _ ▸ finProdFinEquiv.injective hcond
  have h2 := Prod.mk.injEq _ _ _ _ ▸ finProdFinEquiv.injective h1.1
  exact ⟨h2.1, ⟨_, h2.2⟩, ⟨_, h1.2⟩⟩

/-- ⭐⭐ **The `4`.** The masked sum the pool backward reduces has at most four live terms: the
    fibre of the smooth-point reindex over one input cell is contained in
    `{ci} × {rows containing hi} × {cols containing wi}`, and `sum_ite_win3Row_le_two` bounds each
    spatial factor by two. Both `FloatClose` clauses ride on this one statement — the magnitude
    with `B := A`, the error with `B := e`. -/
theorem maxPool3s2Back_mask_sum_abs_le {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (idx : Fin (c*(2*h)*(2*w))) (v : Fin (c*h*w) → ℝ)
    {B : ℝ} (hB : 0 ≤ B) (hv : ∀ k, |v k| ≤ B) :
    ∑ k : Fin (c*h*w), (if maxPool3s2LocalReindex x k = idx then |v k| else 0) ≤ 4 * B := by
  set q := finProdFinEquiv.symm idx with hq
  set r := finProdFinEquiv.symm q.1 with hr
  have hidx : idx = finProdFinEquiv (finProdFinEquiv (r.1, r.2), q.2) := by
    rw [Prod.mk.eta, hr, Equiv.apply_symm_apply, Prod.mk.eta, hq, Equiv.apply_symm_apply]
  set ci := r.1; set hi := r.2; set wi := q.2
  set iCh : Fin c → ℝ := fun co => if co = ci then (1:ℝ) else 0 with hiCh
  set iR : Fin h → ℝ := fun ho => if ∃ a : Fin 3, win3RowInv ho a = hi then (1:ℝ) else 0 with hiR
  set iC : Fin w → ℝ := fun wo => if ∃ b : Fin 3, win3ColInv wo b = wi then (1:ℝ) else 0 with hiC
  have h0Ch : ∀ co, 0 ≤ iCh co := by intro co; rw [hiCh]; dsimp only; split <;> norm_num
  have h0R : ∀ ho, 0 ≤ iR ho := by intro ho; rw [hiR]; dsimp only; split <;> norm_num
  have h0C : ∀ wo, 0 ≤ iC wo := by intro wo; rw [hiC]; dsimp only; split <;> norm_num
  rw [sum_flat3]
  have hstep : ∀ co : Fin c,
      (∑ ho : Fin h, ∑ wo : Fin w,
        (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = idx
          then |v (finProdFinEquiv (finProdFinEquiv (co, ho), wo))| else 0)) ≤ (4 * B) * iCh co := by
    intro co
    have hinner : ∀ ho : Fin h,
        (∑ wo : Fin w,
          (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = idx
            then |v (finProdFinEquiv (finProdFinEquiv (co, ho), wo))| else 0))
          ≤ (2 * B * iCh co) * iR ho := by
      intro ho
      have hpt : ∀ wo ∈ (Finset.univ : Finset (Fin w)),
          (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = idx
            then |v (finProdFinEquiv (finProdFinEquiv (co, ho), wo))| else 0)
            ≤ (B * iCh co * iR ho) * iC wo := by
        intro wo _
        by_cases hcond : maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = idx
        · obtain ⟨hc, ha, hb⟩ := maxPool3s2_reindex_decompose x co ho wo ci hi wi (hidx ▸ hcond)
          rw [if_pos hcond, hiCh, hiR, hiC]
          dsimp only
          rw [if_pos hc, if_pos ha, if_pos hb, mul_one, mul_one, mul_one]
          exact hv _
        · rw [if_neg hcond]
          have := h0Ch co; have := h0R ho; have := h0C wo
          have : 0 ≤ B * iCh co * iR ho := by positivity
          exact mul_nonneg this (h0C wo)
      calc (∑ wo : Fin w,
              (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = idx
                then |v (finProdFinEquiv (finProdFinEquiv (co, ho), wo))| else 0))
          ≤ ∑ wo : Fin w, (B * iCh co * iR ho) * iC wo := Finset.sum_le_sum hpt
        _ = (B * iCh co * iR ho) * ∑ wo : Fin w, iC wo := by rw [← Finset.mul_sum]
        _ ≤ (B * iCh co * iR ho) * 2 := by
            refine mul_le_mul_of_nonneg_left (sum_ite_win3Col_le_two wi) ?_
            have := h0Ch co; have := h0R ho; positivity
        _ = (2 * B * iCh co) * iR ho := by ring
    calc (∑ ho : Fin h, ∑ wo : Fin w,
            (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = idx
              then |v (finProdFinEquiv (finProdFinEquiv (co, ho), wo))| else 0))
        ≤ ∑ ho : Fin h, (2 * B * iCh co) * iR ho := Finset.sum_le_sum (fun ho _ => hinner ho)
      _ = (2 * B * iCh co) * ∑ ho : Fin h, iR ho := by rw [← Finset.mul_sum]
      _ ≤ (2 * B * iCh co) * 2 := by
          refine mul_le_mul_of_nonneg_left (sum_ite_win3Row_le_two hi) ?_
          have := h0Ch co; positivity
      _ = (4 * B) * iCh co := by ring
  calc (∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = idx
            then |v (finProdFinEquiv (finProdFinEquiv (co, ho), wo))| else 0))
      ≤ ∑ co : Fin c, (4 * B) * iCh co := Finset.sum_le_sum (fun co _ => hstep co)
    _ = (4 * B) * ∑ co : Fin c, iCh co := by rw [← Finset.mul_sum]
    _ = 4 * B := by rw [hiCh]; simp

-- ════════════════════════════════════════════════════════════════
-- § The leaf: definition, float peer, `FloatClose`, and the bridges
-- ════════════════════════════════════════════════════════════════

/-- **3×3/s2 max-pool backward in flat `Vec` space** — the accumulating scatter: each input cell
    collects `dy` from every output whose 3×3 window selects it (at most four,
    `maxPool3s2Back_mask_sum_abs_le`). Spelled as the masked reduction the kernel performs, which
    is `maxPool3s2_has_vjp_at3`'s backward reindexed (`maxPool3s2FlatBack_eq_vjp_backward`). ⛔ The
    2×2 peer `maxPoolFlatBack` is a LOOKUP and is not this function. -/
noncomputable def maxPool3s2FlatBack {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
    Vec (c*h*w) → Vec (c*(2*h)*(2*w)) :=
  fun dy idx => ∑ k : Fin (c*h*w), (if maxPool3s2LocalReindex x k = idx then dy k else 0)

/-- **The float 3×3/s2 pool backward** — the rounded masked reduction `fl(Σ [σ(k) = idx]·dy k)`.
    The peer of `FloatModel.broadcastBackFlatF`; `maxPoolFlatBack`'s float peer is ITSELF (a
    lookup rounds nothing), and that is exactly what overlap costs. -/
noncomputable def FloatModel.maxPool3s2FlatBackF {c h w : Nat} (M : FloatModel)
    (x : Tensor3 c (2*h) (2*w)) : Vec (c*h*w) → Vec (c*(2*h)*(2*w)) :=
  fun dy idx => M.sum (fun k : Fin (c*h*w) => if maxPool3s2LocalReindex x k = idx then dy k else 0)

/-- The output index type is inhabited only when every dimension is, and then so is the input's. -/
private theorem maxPool3s2_chw_pos {c h w : Nat} (i : Fin (c*(2*h)*(2*w))) : 0 < c * h * w := by
  have h1 : 0 < c * (2*h) * (2*w) := lt_of_le_of_lt (Nat.zero_le i.val) i.isLt
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · simp at h1
  rcases Nat.eq_zero_or_pos h with rfl | hh
  · simp at h1
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp at h1
  positivity

/-- **The 3×3/s2 pool backward is `FloatClose`.** Both outputs land in `4A·(1+γ)` and the float
    reduction is within `γ·4A + 4e` of the real one, at `γ = (1+u)^(c·h·w+1) − 1` (the kernel
    reduces over that many terms) and the `4` from `maxPool3s2Back_mask_sum_abs_le`.
    ⛔ Compare `floatClose_maxPoolFlatBack`, which is `FloatClose A A f f id`: overlap costs a
    factor of four on the window and a rounding term where the 2×2 pool has neither. -/
theorem floatClose_maxPool3s2Back {c h w : Nat} (M : FloatModel)
    (x : Tensor3 c (2*h) (2*w)) (A : ℝ) :
    FloatClose A (4 * A + ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A))
      (maxPool3s2FlatBack x) (M.maxPool3s2FlatBackF x)
      (fun e => ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) + 4 * e) := by
  have hu := M.u_nonneg
  have hγ0 : 0 ≤ (1 + M.u) ^ (c*h*w + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hA0 : 0 ≤ A := (abs_nonneg _).trans (hv ⟨0, maxPool3s2_chw_pos i⟩)
    set mv : Fin (c*h*w) → ℝ :=
      fun k => if maxPool3s2LocalReindex x k = i then v k else 0 with hmv
    have habs : ∀ k, |mv k| = if maxPool3s2LocalReindex x k = i then |v k| else 0 := by
      intro k; rw [hmv]; dsimp only; split
      · rfl
      · exact abs_zero
    have hsum : ∑ k, |mv k| ≤ 4 * A := by
      simp only [habs]
      exact maxPool3s2Back_mask_sum_abs_le x i v hA0 hv
    have hreal : |maxPool3s2FlatBack x v i| ≤ 4 * A :=
      (Finset.abs_sum_le_sum_abs _ _).trans hsum
    have hround : |M.sum mv - ∑ k, mv k| ≤ ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) :=
      (M.sum_close mv).trans (mul_le_mul_of_nonneg_left hsum hγ0)
    refine ⟨hreal.trans (le_add_of_nonneg_right (mul_nonneg hγ0 (by linarith))), ?_⟩
    calc |M.maxPool3s2FlatBackF x v i|
        ≤ |M.sum mv - ∑ k, mv k| + |∑ k, mv k| := by
          simpa [FloatModel.maxPool3s2FlatBackF, hmv] using
            abs_sub_le (M.sum mv) (∑ k, mv k) 0
      _ ≤ ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) + 4 * A := add_le_add hround hreal
      _ = 4 * A + ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) := by ring
  · have hpos := maxPool3s2_chw_pos i
    have hA0 : 0 ≤ A := (abs_nonneg _).trans (hva ⟨0, hpos⟩)
    have he0 : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, hpos⟩)
    set mvt : Fin (c*h*w) → ℝ :=
      fun k => if maxPool3s2LocalReindex x k = i then vt k else 0 with hmvt
    set mva : Fin (c*h*w) → ℝ :=
      fun k => if maxPool3s2LocalReindex x k = i then va k else 0 with hmva
    have habst : ∀ k, |mvt k| = if maxPool3s2LocalReindex x k = i then |vt k| else 0 := by
      intro k; rw [hmvt]; dsimp only; split
      · rfl
      · exact abs_zero
    have hsumt : ∑ k, |mvt k| ≤ 4 * A := by
      simp only [habst]; exact maxPool3s2Back_mask_sum_abs_le x i vt hA0 hvt
    have hdiff : ∀ k, |mvt k - mva k| = if maxPool3s2LocalReindex x k = i
        then |vt k - va k| else 0 := by
      intro k; rw [hmvt, hmva]; dsimp only; split
      · rfl
      · simp
    have hsumd : ∑ k, |mvt k - mva k| ≤ 4 * e := by
      simp only [hdiff]
      exact maxPool3s2Back_mask_sum_abs_le x i (fun k => vt k - va k) he0 hd
    have hround : |M.sum mvt - ∑ k, mvt k| ≤ ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) :=
      (M.sum_close mvt).trans (mul_le_mul_of_nonneg_left hsumt hγ0)
    have hreal : |(∑ k, mvt k) - ∑ k, mva k| ≤ 4 * e := by
      rw [← Finset.sum_sub_distrib]
      exact (Finset.abs_sum_le_sum_abs _ _).trans hsumd
    calc |M.maxPool3s2FlatBackF x vt i - maxPool3s2FlatBack x va i|
        = |M.sum mvt - ∑ k, mva k| := rfl
      _ ≤ |M.sum mvt - ∑ k, mvt k| + |(∑ k, mvt k) - ∑ k, mva k| := abs_sub_le _ _ _
      _ ≤ ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) + 4 * e := add_le_add hround hreal

/-- The 3×3/s2 pool backward float-bridges (the accumulating scatter). -/
theorem floatBridges_maxPool3s2Back {c h w : Nat} (M : FloatModel)
    (x : Tensor3 c (2*h) (2*w)) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    FloatBridges (maxPool3s2FlatBack x) := fun A hA =>
  ⟨_, _, _, (floatClose_maxPool3s2Back M x A).cod_nonneg hA (by positivity),
    floatClose_maxPool3s2Back M x A⟩

/-- The 3×3/s2 pool backward float-bridges TO the model's rounded masked reduction. -/
noncomputable def floatBridgesTo_maxPool3s2Back {c h w : Nat} (M : FloatModel)
    (x : Tensor3 c (2*h) (2*w)) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    FloatBridgesTo (maxPool3s2FlatBack x) (M.maxPool3s2FlatBackF x) :=
  ⟨fun A => 4 * A + ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A),
   fun A e => ((1 + M.u) ^ (c*h*w + 1) - 1) * (4 * A) + 4 * e,
   fun A hA => ⟨(floatClose_maxPool3s2Back M x A).cod_nonneg hA (by positivity),
     floatClose_maxPool3s2Back M x A⟩⟩

-- ════════════════════════════════════════════════════════════════
-- § The certified tie
-- ════════════════════════════════════════════════════════════════

/-- **3×3/s2 pool input-VJP leaf tie (smooth point).** `maxPool3s2FlatBack x` IS the certified
    pool input-VJP `(maxPool3s2Flat_has_vjp_at x h_smooth).backward`: the certified backward is the
    triple sum `∑_{co,ho,wo} [σ(co,ho,wo) = idx]·dy(co,ho,wo)`, and this is that sum re-indexed
    row-major (`sum_flat3`). The 3×3/s2 peer of `maxPoolFlatBack_eq_vjp_backward`. -/
theorem maxPool3s2FlatBack_eq_vjp_backward {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (h_smooth : MaxPool3s2Smooth x) :
    maxPool3s2FlatBack x = (maxPool3s2Flat_has_vjp_at x h_smooth).backward := by
  funext dy idx
  show (∑ k : Fin (c*h*w), (if maxPool3s2LocalReindex x k = idx then dy k else 0)) = _
  rw [sum_flat3 (fun k => if maxPool3s2LocalReindex x k = idx then dy k else 0)]
  have hidx : finProdFinEquiv
      (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1,
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
        (finProdFinEquiv.symm idx).2) = idx := by
    rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
  simp only [maxPool3s2Flat_has_vjp_at, hasVJPAt3_to_hasVJPAt, maxPool3s2_has_vjp_at3,
    Tensor3.unflatten]
  refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl fun ho _ =>
    Finset.sum_congr rfl fun wo _ => ?_
  rw [hidx]
  split <;> simp

/-- ⭐ **The pool VJP at a `Vec` point, with its backward DEFINITIONALLY `maxPool3s2FlatBack`.**
    `maxPool3s2Flat_has_vjp_at` is stated at `Tensor3.flatten x`, and a whole-net chain needs it at
    the stem's `Vec` output. ⛔ Transporting with `▸`/`rwa` would work for the TYPE and leave a
    `backward` field behind an `Eq.mpr` that will not reduce — the `FloatBridgesTo.ofEq` trap
    (`planning/float_budget_numbers.md` §3.5.2 item 5) one tier down. Building the structure
    field-by-field instead keeps `backward` the leaf itself, which is what lets the whole-net tie
    close by `rfl` at this stage rather than by a rewrite. -/
noncomputable def maxPool3s2Flat_has_vjp_at_vec {c h w : Nat} (v : Vec (c * (2*h) * (2*w)))
    (h_smooth : MaxPool3s2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    HasVJPAt (maxPool3s2Flat c h w) v where
  backward := maxPool3s2FlatBack (Tensor3.unflatten v)
  correct := by
    intro dy i
    have hc := (maxPool3s2Flat_has_vjp_at (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))
      h_smooth).correct dy i
    rw [← maxPool3s2FlatBack_eq_vjp_backward _ h_smooth] at hc
    rwa [Tensor3.flatten_unflatten] at hc

/-- The `Vec`-point differentiability companion of `maxPool3s2Flat_has_vjp_at_vec`. -/
theorem maxPool3s2Flat_differentiableAt_vec {c h w : Nat} (v : Vec (c * (2*h) * (2*w)))
    (h_smooth : MaxPool3s2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w)))
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    DifferentiableAt ℝ (maxPool3s2Flat c h w) v := by
  have h := maxPool3s2Flat_differentiableAt (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))
    h_smooth hc hh hw
  rwa [Tensor3.flatten_unflatten] at h

end Proofs
