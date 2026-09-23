import LeanMlir.Proofs.Architectures.CNN

/-! # Conv and max-pool index facts — the flat ↔ tensor index vocabulary

The flat-index plumbing every conv-net proof reads tensors through (`t3Idx`, the window-tiling
sums, `sum_s2`), and the 2×2 max-pool's window facts: the window max is Lipschitz in its cells,
the pool is 1-Lipschitz per entry and ℓ1-contractive, and a selection margin beyond `2δ` freezes
the argmax (`MaxPool2MarginQ`). The ℝ conv as a dense layer with weight sharing and its float
forward are in `ConvFloat`.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Max is Lipschitz: the 2×2 window max moves no more than its cells
-- ════════════════════════════════════════════════════════════════

/-- A four-way max moves by at most the largest cell movement (`ℓ∞`). -/
theorem max4_sub_abs_le {a b c d a' b' c' d' δ : ℝ}
    (h1 : |a - a'| ≤ δ) (h2 : |b - b'| ≤ δ)
    (h3 : |c - c'| ≤ δ) (h4 : |d - d'| ≤ δ) :
    |max (max a b) (max c d) - max (max a' b') (max c' d')| ≤ δ := by
  have hab : |max a b - max a' b'| ≤ δ :=
    le_trans (abs_max_sub_max_le_max a b a' b') (max_le h1 h2)
  have hcd : |max c d - max c' d'| ≤ δ :=
    le_trans (abs_max_sub_max_le_max c d c' d') (max_le h3 h4)
  exact le_trans (abs_max_sub_max_le_max _ _ _ _) (max_le hab hcd)

/-- A four-way max moves by at most the *sum* of the cell movements
    (`ℓ1`) — the per-window step of the pool's `ℓ1` contraction. -/
theorem max4_sub_abs_le_sum {a b c d a' b' c' d' : ℝ} :
    |max (max a b) (max c d) - max (max a' b') (max c' d')| ≤
      |a - a'| + |b - b'| + |c - c'| + |d - d'| := by
  refine max4_sub_abs_le (δ := |a - a'| + |b - b'| + |c - c'| + |d - d'|)
    ?_ ?_ ?_ ?_ <;>
    nlinarith [abs_nonneg (a - a'), abs_nonneg (b - b'),
      abs_nonneg (c - c'), abs_nonneg (d - d')]

-- ════════════════════════════════════════════════════════════════
-- § Index plumbing: window cells tile the input, flat sums = tensor sums
-- ════════════════════════════════════════════════════════════════

/-- Flat index of a `Tensor3` coordinate (the suite's row-major layout).

    ⚠ `@[reducible]` is load-bearing on Lean ≥ 4.33 (see planning/archive/lean_434_and_cleanup.md):
    `t3Idx_def` folds the raw encoding into the `ite` CONDITION below, but simp does not rewrite
    inside the `Decidable` INSTANCE argument, so the goal carries a folded condition over an
    unfolded instance and every `ite_eq_left`/`ite_eq_right` here fails to match. `basisVec`, which produces
    that `ite`, is `@[reducible]` for the same reason. -/
@[reducible] def t3Idx {c h w : Nat} (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    Fin (c * h * w) :=
  finProdFinEquiv (finProdFinEquiv (ci, hi), wi)

/-- `t3Idx` reads back through `Tensor3.unflatten`. -/
theorem unflatten_t3Idx {c h w : Nat} (v : Vec (c * h * w))
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten v ci hi wi = v (t3Idx ci hi wi) := rfl

/-- `Tensor3.flatten` reads off at a `t3Idx`. -/
theorem flatten_t3Idx {c h w : Nat} (T : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.flatten T (t3Idx ci hi wi) = T ci hi wi := by
  unfold Tensor3.flatten t3Idx
  simp

/-- A flat sum is the triple tensor sum. -/
theorem sum_t3 {c h w : Nat} (f : Fin (c * h * w) → ℝ) :
    ∑ k, f k = ∑ ci : Fin c, ∑ hi : Fin h, ∑ wi : Fin w,
      f (t3Idx ci hi wi) := by
  simp only [sum_finProdFinEquiv]

/-- Every flat spatial index is a `t3Idx` — lets a per-cell bound be lifted to
    the whole flattened conv-output vector (`∀ k`), the form `relu_close` /
    `maxPoolFlat_close` / `dense_close` consume. -/
theorem t3Idx_surj {c h w : Nat} (k : Fin (c * h * w)) :
    ∃ (ci : Fin c) (hi : Fin h) (wi : Fin w), k = t3Idx ci hi wi := by
  refine ⟨(finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1,
    (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2,
    (finProdFinEquiv.symm k).2, ?_⟩
  simp only [t3Idx, Prod.mk.eta, Equiv.apply_symm_apply]

/-- The window-cell parameterization `(out-row, sub-row) ↦ in-row` is a
    bijection — pooled windows tile the rows. -/
def winRowEquiv (h : Nat) : Fin h × Fin 2 ≃ Fin (2 * h) where
  toFun p := winRowInv p.1 p.2
  invFun hi := (winRow hi, winRowMod hi)
  left_inv p := by
    ext
    · exact congrArg Fin.val (winRow_winRowInv p.1 p.2)
    · exact congrArg Fin.val (winRowMod_winRowInv p.1 p.2)
  right_inv hi := winRowInv_winRow hi

/-- Column version of `winRowEquiv`. -/
def winColEquiv (w : Nat) : Fin w × Fin 2 ≃ Fin (2 * w) where
  toFun p := winColInv p.1 p.2
  invFun wi := (winCol wi, winColMod wi)
  left_inv p := by
    ext
    · exact congrArg Fin.val (winCol_winColInv p.1 p.2)
    · exact congrArg Fin.val (winColMod_winColInv p.1 p.2)
  right_inv wi := winColInv_winCol wi

/-- Summing a function over all window cells of all windows is summing it
    over the whole spatial grid — the 2×2 stride-2 windows partition the
    input. -/
theorem sum_window_cells {h w : Nat} (g : Fin (2 * h) → Fin (2 * w) → ℝ) :
    ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
        g (winRowInv ho ab.1) (winColInv wo ab.2) =
      ∑ hi : Fin (2 * h), ∑ wi : Fin (2 * w), g hi wi := by
  have hcol : ∀ g' : Fin (2 * w) → ℝ,
      ∑ wo : Fin w, ∑ b : Fin 2, g' (winColInv wo b) = ∑ wi, g' wi := by
    intro g'
    calc ∑ wo : Fin w, ∑ b : Fin 2, g' (winColInv wo b)
        = ∑ q : Fin w × Fin 2, g' (winColInv q.1 q.2) :=
          (Fintype.sum_prod_type
            (fun q : Fin w × Fin 2 => g' (winColInv q.1 q.2))).symm
      _ = ∑ wi, g' wi := Equiv.sum_comp (winColEquiv w) g'
  have hrow : ∀ g' : Fin (2 * h) → ℝ,
      ∑ ho : Fin h, ∑ a : Fin 2, g' (winRowInv ho a) = ∑ hi, g' hi := by
    intro g'
    calc ∑ ho : Fin h, ∑ a : Fin 2, g' (winRowInv ho a)
        = ∑ p : Fin h × Fin 2, g' (winRowInv p.1 p.2) :=
          (Fintype.sum_prod_type
            (fun p : Fin h × Fin 2 => g' (winRowInv p.1 p.2))).symm
      _ = ∑ hi, g' hi := Equiv.sum_comp (winRowEquiv h) g'
  calc ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
        g (winRowInv ho ab.1) (winColInv wo ab.2)
      = ∑ ho : Fin h, ∑ a : Fin 2, ∑ wo : Fin w, ∑ b : Fin 2,
          g (winRowInv ho a) (winColInv wo b) := by
        refine Finset.sum_congr rfl fun ho _ => ?_
        calc ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
              g (winRowInv ho ab.1) (winColInv wo ab.2)
            = ∑ wo : Fin w, ∑ a : Fin 2, ∑ b : Fin 2,
                g (winRowInv ho a) (winColInv wo b) :=
              Finset.sum_congr rfl fun wo _ => Fintype.sum_prod_type _
          _ = ∑ a : Fin 2, ∑ wo : Fin w, ∑ b : Fin 2,
                g (winRowInv ho a) (winColInv wo b) := Finset.sum_comm
    _ = ∑ ho : Fin h, ∑ a : Fin 2, ∑ wi : Fin (2 * w),
          g (winRowInv ho a) wi := by
        refine Finset.sum_congr rfl fun ho _ => ?_
        exact Finset.sum_congr rfl fun a _ =>
          hcol (fun wi => g (winRowInv ho a) wi)
    _ = ∑ hi : Fin (2 * h), ∑ wi : Fin (2 * w), g hi wi :=
        hrow (fun hi => ∑ wi : Fin (2 * w), g hi wi)

-- ════════════════════════════════════════════════════════════════
-- § The pool is 1-Lipschitz per entry and ℓ1-contractive across entries
-- ════════════════════════════════════════════════════════════════

/-- The pooled entry at `(ci, ho, wo)` is the four-way max of its window
    cells, in flat coordinates. -/
theorem maxPoolFlat_apply {c h w : Nat} (u : Vec (c * (2*h) * (2*w)))
    (ci : Fin c) (ho : Fin h) (wo : Fin w) :
    maxPoolFlat c h w u (t3Idx ci ho wo) =
      max (max (u (t3Idx ci (winRowInv ho 0) (winColInv wo 0)))
               (u (t3Idx ci (winRowInv ho 1) (winColInv wo 0))))
          (max (u (t3Idx ci (winRowInv ho 0) (winColInv wo 1)))
               (u (t3Idx ci (winRowInv ho 1) (winColInv wo 1)))) := by
  show Tensor3.flatten (maxPool2 (Tensor3.unflatten u)) (t3Idx ci ho wo) = _
  rw [flatten_t3Idx, winRowInv_zero, winRowInv_one, winColInv_zero,
    winColInv_one]
  rfl

/-- `ℓ1` contraction: the pooled drift, summed over all pooled entries, is
    at most the input drift summed over all input entries (windows are
    disjoint, max is 1-Lipschitz). The pool passes `ℓ1` budgets through
    unamplified. -/
theorem maxPoolFlat_l1_contract {c h w : Nat}
    (u v : Vec (c * (2*h) * (2*w))) :
    ∑ q, |maxPoolFlat c h w u q - maxPoolFlat c h w v q| ≤
      ∑ k, |u k - v k| := by
  rw [sum_t3 (fun q => |maxPoolFlat c h w u q - maxPoolFlat c h w v q|),
    sum_t3 (fun k => |u k - v k|)]
  refine Finset.sum_le_sum fun ci _ => ?_
  calc ∑ ho : Fin h, ∑ wo : Fin w,
        |maxPoolFlat c h w u (t3Idx ci ho wo) -
          maxPoolFlat c h w v (t3Idx ci ho wo)|
      ≤ ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
          |u (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2)) -
            v (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2))| := by
        refine Finset.sum_le_sum fun ho _ => Finset.sum_le_sum fun wo _ => ?_
        have hexp : ∑ ab : Fin 2 × Fin 2,
            |u (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2)) -
              v (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2))| =
            ∑ a : Fin 2, ∑ b : Fin 2,
              |u (t3Idx ci (winRowInv ho a) (winColInv wo b)) -
                v (t3Idx ci (winRowInv ho a) (winColInv wo b))| :=
          Fintype.sum_prod_type _
        rw [maxPoolFlat_apply, maxPoolFlat_apply, hexp, Fin.sum_univ_two,
          Fin.sum_univ_two, Fin.sum_univ_two]
        refine le_trans max4_sub_abs_le_sum (le_of_eq ?_)
        ring
    _ = ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |u (t3Idx ci hi wi) - v (t3Idx ci hi wi)| :=
        sum_window_cells (fun hi wi =>
          |u (t3Idx ci hi wi) - v (t3Idx ci hi wi)|)

-- ════════════════════════════════════════════════════════════════
-- § The selection margin: window gaps beyond 2δ freeze the argmax
-- ════════════════════════════════════════════════════════════════

/-- Two values farther apart than `2δ` cannot be equalized by
    `δ`-perturbations. -/
theorem ne_of_gap_of_close {xa xb ya yb δ : ℝ} (hgap : 2 * δ < |xa - xb|)
    (ha : |ya - xa| ≤ δ) (hb : |yb - xb| ≤ δ) : ya ≠ yb := by
  intro heq
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  have heq' : ya - yb = 0 := by rw [heq]; ring
  have hle : |xa - xb| ≤ 2 * δ :=
    abs_le.mpr ⟨by linarith [h1.1, h1.2, h2.1, h2.2],
      by linarith [h1.1, h1.2, h2.1, h2.2]⟩
  linarith

/-- Strict order survives `δ`-perturbations across a `2δ` gap. -/
theorem lt_of_lt_gap_of_close {xa xb ya yb δ : ℝ}
    (hlt : 2 * δ < xb - xa) (ha : |ya - xa| ≤ δ) (hb : |yb - xb| ≤ δ) :
    ya < yb := by
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  linarith [h1.1, h1.2, h2.1, h2.2]

/-- **Quantitative pool-selection margin**: every two cells of every 2×2
    window differ by more than `2δ`. The quantitative form of
    `MaxPool2Smooth` — a perturbation of at most `δ` per entry can neither
    create a tie nor reorder a window, so the pool's argmax routing
    freezes. The pool peer of the ReLU margin `a·D < |zⱼ|`. -/
def MaxPool2MarginQ {c h w : Nat} (δ : ℝ)
    (x : Tensor3 c (2*h) (2*w)) : Prop :=
  ∀ (ci : Fin c) (ho : Fin h) (wo : Fin w)
    (ab ab' : Fin 2 × Fin 2), ab ≠ ab' →
    2 * δ < |x ci (winRowInv ho ab.1) (winColInv wo ab.2) -
             x ci (winRowInv ho ab'.1) (winColInv wo ab'.2)|

/-- Every point within `δ` of a margined point is smooth (no window
    ties). -/
theorem MaxPool2MarginQ.smooth_of_close {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ) :
    MaxPool2Smooth y := fun ci ho wo ab ab' hne =>
  ne_of_gap_of_close (hm ci ho wo ab ab' hne)
    (hclose ci (winRowInv ho ab.1) (winColInv wo ab.2))
    (hclose ci (winRowInv ho ab'.1) (winColInv wo ab'.2))

/-- A margined point is itself smooth. -/
theorem MaxPool2MarginQ.smooth {c h w : Nat} {δ : ℝ} (hδ0 : 0 ≤ δ)
    {x : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x) :
    MaxPool2Smooth x :=
  hm.smooth_of_close (fun ci hi wi => by simp [hδ0])

/-- **The argmax freezes**: within `δ` of a margined point, every window's
    argmax cell is the same as at the margined point. -/
theorem MaxPool2MarginQ.isArgmax_iff {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    MaxPool2IsArgmax y ci hi wi ↔ MaxPool2IsArgmax x ci hi wi := by
  have hxw : x ci (winRowInv (winRow hi) (winRowMod hi))
      (winColInv (winCol wi) (winColMod wi)) = x ci hi wi := by
    rw [winRowInv_winRow, winColInv_winCol]
  have hyw : y ci (winRowInv (winRow hi) (winRowMod hi))
      (winColInv (winCol wi) (winColMod wi)) = y ci hi wi := by
    rw [winRowInv_winRow, winColInv_winCol]
  constructor
  · -- y-argmax at (hi,wi) ⇒ x-argmax at (hi,wi), by contraposition on cells
    intro hy a b
    by_contra hnot
    have hlt : x ci hi wi <
        x ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) :=
      not_le.mp hnot
    have hne : ((a, b) : Fin 2 × Fin 2) ≠ (winRowMod hi, winColMod wi) := by
      rintro ⟨⟩
      rw [hxw] at hlt
      exact lt_irrefl _ hlt
    have hgap := hm ci (winRow hi) (winCol wi) (a, b)
      (winRowMod hi, winColMod wi) hne
    rw [hxw] at hgap
    have hgap' : 2 * δ <
        x ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) -
          x ci hi wi := by
      rwa [abs_of_pos (by linarith)] at hgap
    have hylt : y ci hi wi <
        y ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) :=
      lt_of_lt_gap_of_close hgap' (hclose ci hi wi)
        (hclose ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b))
    exact absurd (hy a b) (not_le.mpr hylt)
  · -- x-argmax at (hi,wi) ⇒ y-argmax at (hi,wi)
    intro hx a b
    by_cases hEq : ((a, b) : Fin 2 × Fin 2) = (winRowMod hi, winColMod wi)
    · cases hEq; rw [hyw]
    · have hle := hx a b
      have hgap := hm ci (winRow hi) (winCol wi)
        (winRowMod hi, winColMod wi) (a, b) (Ne.symm hEq)
      rw [hxw] at hgap
      have hgap' : 2 * δ < x ci hi wi -
          x ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) := by
        rwa [abs_of_nonneg (sub_nonneg.mpr hle)] at hgap
      exact le_of_lt (lt_of_lt_gap_of_close hgap'
        (hclose ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b))
        (hclose ci hi wi))


/-- The spatial `(hi, wi)` sum collapses to one flat sum over `Fin (h·w)`. -/
theorem sum_s2 {h w : Nat} (g : Fin (h * w) → ℝ) :
    ∑ s, g s = ∑ hi : Fin h, ∑ wi : Fin w, g (finProdFinEquiv (hi, wi)) :=
  sum_finProdFinEquiv g

/-- The padded input read that multiplies kernel entry `(·, c, kh, kw)` at
    output position `(hi, wi)` — names the `dite` inside `conv2d` so the
    affine-in-the-kernel structure can be stated. Depends on the input
    only, never the kernel. -/
noncomputable def convPad {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    ℝ :=
  if hpad : (kH - 1) / 2 ≤ kh.val + hi.val ∧
      kh.val + hi.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + wi.val ∧
      kw.val + wi.val - (kW - 1) / 2 < w then
    x c ⟨kh.val + hi.val - (kH - 1) / 2, hpad.2.1⟩
        ⟨kw.val + wi.val - (kW - 1) / 2, hpad.2.2.2⟩
  else 0

end Proofs
