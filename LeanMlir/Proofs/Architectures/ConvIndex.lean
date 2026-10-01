import LeanMlir.Proofs.Architectures.CNN

/-! # Conv and max-pool index facts — the flat ↔ tensor index vocabulary

The flat-index plumbing every conv-net proof reads tensors through (`t3Idx` and its
flat-sum `sum_t3`), and the 2×2 max-pool's window facts: the pool with its routing frozen
(`poolGatherFlat`, which agrees with the pool at its selection) is ℓ1-contractive
(`poolGatherFlat_l1_contract`), and a selection margin beyond `2δ` freezes the argmax
(`MaxPool2MarginQ`). The conv reads its input through the zero-padded window `convPad`
and its kernel through the flat index `k4Idx`; cells with identical padded patches are twins
(`ConvPatchEq`, and `ConvPatchEq2` two convs deep), equal under every kernel and bias
(`conv2d_eq_of_convPatchEq`). The conv as a dense layer with weight sharing and its float forward
are in `ConvFloat`.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Index plumbing: window cells tile the input, flat sums = tensor sums
-- ════════════════════════════════════════════════════════════════

/-- Flat index of a `Tensor3` coordinate (the suite's row-major layout).

    Note: on Lean ≥ 4.33 the proofs below fail without `@[reducible]`:
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
-- § The 2×2 gather: the pool with its routing frozen
-- ════════════════════════════════════════════════════════════════

/-- The 2×2 window gather at selection `σ` (`windowGather` at the 2×2 windows), flattened like
    `maxPoolFlat`: pooled entry `(ci, ho, wo)` reads window cell `σ ci ho wo`. -/
noncomputable def poolGatherFlat {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) :
    Vec (c * (2*h) * (2*w)) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (windowGather winRowInv winColInv σ (Tensor3.unflatten v))

theorem poolGatherFlat_apply {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (u : Vec (c * (2*h) * (2*w))) (ci : Fin c) (ho : Fin h) (wo : Fin w) :
    poolGatherFlat σ u (t3Idx ci ho wo) =
      u (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2)) := by
  rw [poolGatherFlat, flatten_t3Idx]; rfl

/-- The 2×2 gather as a continuous linear map: each output reads one input coordinate. -/
noncomputable def poolGatherCLM {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) :
    Vec (c * (2*h) * (2*w)) →L[ℝ] Vec (c * h * w) where
  toFun := poolGatherFlat σ
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  cont := continuous_pi fun _ => continuous_apply _

theorem poolGatherCLM_coe {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) :
    ⇑(poolGatherCLM σ) = poolGatherFlat σ := rfl

@[fun_prop]
theorem poolGatherFlat_differentiable {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2) :
    Differentiable ℝ (poolGatherFlat σ) := by
  rw [← poolGatherCLM_coe]; exact (poolGatherCLM σ).differentiable

/-- **The gather's Jacobian**: input `(ci, hi, wi)` feeds pooled `(co, ho, wo)` exactly when it is
    that window's selected cell. -/
theorem pdiv_poolGatherFlat {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (u : Vec (c * (2*h) * (2*w))) (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv (poolGatherFlat σ) u (t3Idx ci hi wi) (t3Idx co ho wo) =
      if co = ci ∧ ho = winRow hi ∧ wo = winCol wi ∧
          σ co ho wo = (winRowMod hi, winColMod wi) then 1 else 0 := by
  rw [← poolGatherCLM_coe, pdiv_eq_of_hasFDerivAt (poolGatherCLM σ).hasFDerivAt,
    poolGatherCLM_coe, poolGatherFlat_apply, basisVec_apply]
  congr 1
  apply propext
  constructor
  · intro hA
    have h1 := finProdFinEquiv.injective hA
    have h2 := Prod.mk.inj h1
    have h3 := Prod.mk.inj (finProdFinEquiv.injective h2.1)
    refine ⟨h3.1, ?_, ?_, ?_⟩
    · rw [← h3.2, winRow_winRowInv]
    · rw [← h2.2, winCol_winColInv]
    · rw [← h3.2, ← h2.2, winRowMod_winRowInv, winColMod_winColInv]
  · rintro ⟨rfl, rfl, rfl, hσ⟩
    rw [hσ, winRowInv_winRow, winColInv_winCol]

/-- **The gather is `ℓ1`-contractive**: each pooled entry reads a distinct input cell (the
    2×2 windows partition the input). -/
theorem poolGatherFlat_l1_contract {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (u v : Vec (c * (2*h) * (2*w))) :
    ∑ q, |poolGatherFlat σ u q - poolGatherFlat σ v q| ≤ ∑ k, |u k - v k| := by
  rw [sum_t3 (fun q => |poolGatherFlat σ u q - poolGatherFlat σ v q|),
    sum_t3 (fun k => |u k - v k|)]
  refine Finset.sum_le_sum fun ci _ => ?_
  calc ∑ ho : Fin h, ∑ wo : Fin w,
        |poolGatherFlat σ u (t3Idx ci ho wo) - poolGatherFlat σ v (t3Idx ci ho wo)|
      ≤ ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
          |u (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2)) -
            v (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2))| := by
        refine Finset.sum_le_sum fun ho _ => Finset.sum_le_sum fun wo _ => ?_
        rw [poolGatherFlat_apply, poolGatherFlat_apply]
        exact Finset.single_le_sum (f := fun ab : Fin 2 × Fin 2 =>
          |u (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2)) -
            v (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2))|)
          (fun _ _ => abs_nonneg _) (Finset.mem_univ (σ ci ho wo))
    _ = ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |u (t3Idx ci hi wi) - v (t3Idx ci hi wi)| :=
        sum_window_cells (fun hi wi => |u (t3Idx ci hi wi) - v (t3Idx ci hi wi)|)

/-- **The flat pool is the flat gather at a dominating selection.** -/
theorem maxPoolFlat_eq_poolGatherFlat {c h w : Nat} (σ : Fin c → Fin h → Fin w → Fin 2 × Fin 2)
    (u : Vec (c * (2*h) * (2*w)))
    (hdom : ∀ ci ho wo (cd : Fin 2 × Fin 2),
      u (t3Idx ci (winRowInv ho cd.1) (winColInv wo cd.2)) ≤
        u (t3Idx ci (winRowInv ho (σ ci ho wo).1) (winColInv wo (σ ci ho wo).2))) :
    maxPoolFlat c h w u = poolGatherFlat σ u := by
  unfold maxPoolFlat poolGatherFlat
  rw [maxPool2_eq_windowMax, windowMax_eq_windowGather winRowInv winColInv σ _ hdom]

-- ════════════════════════════════════════════════════════════════
-- § The selection margin: window gaps beyond 2δ freeze the argmax
-- ════════════════════════════════════════════════════════════════

/-- Strict order survives `δ`-perturbations across a `2δ` gap. -/
theorem lt_of_lt_gap_of_close {xa xb ya yb δ : ℝ}
    (hlt : 2 * δ < xb - xa) (ha : |ya - xa| ≤ δ) (hb : |yb - xb| ≤ δ) :
    ya < yb := by
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  linarith [h1.1, h1.2, h2.1, h2.2]

/-- **Quantitative pool-selection margin**: in every 2×2 window, a cell that dominates the
    window is more than `2δ` above every other cell of it. The quantitative form of
    `MaxPool2Smooth` — a perturbation of at most `δ` per entry can neither tie the max nor move
    it to another cell, so the pool's argmax routing freezes. The other cells may tie with each
    other. The pool peer of the ReLU margin `a·D < |zⱼ|`. -/
def MaxPool2MarginQ {c h w : Nat} (δ : ℝ)
    (x : Tensor3 c (2*h) (2*w)) : Prop :=
  ∀ (ci : Fin c) (ho : Fin h) (wo : Fin w)
    (ab ab' : Fin 2 × Fin 2), ab ≠ ab' →
    (∀ cd : Fin 2 × Fin 2,
      x ci (winRowInv ho cd.1) (winColInv wo cd.2) ≤
      x ci (winRowInv ho ab.1) (winColInv wo ab.2)) →
    2 * δ < x ci (winRowInv ho ab.1) (winColInv wo ab.2) -
             x ci (winRowInv ho ab'.1) (winColInv wo ab'.2)

/-- Within `δ` of a margined point, a cell that dominates its window at the perturbed point is
    the cell that dominates it at the margined point. -/
private theorem MaxPool2MarginQ.dom_eq {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ)
    (ci : Fin c) (ho : Fin h) (wo : Fin w) (ab : Fin 2 × Fin 2)
    (hy : ∀ cd : Fin 2 × Fin 2,
      y ci (winRowInv ho cd.1) (winColInv wo cd.2) ≤
      y ci (winRowInv ho ab.1) (winColInv wo ab.2)) :
    ab = maxPool2Argmax x ci ho wo := by
  by_contra hne
  have hgap := hm ci ho wo _ ab (Ne.symm hne) (maxPool2Argmax_max x ci ho wo)
  exact absurd (hy (maxPool2Argmax x ci ho wo))
    (not_le.mpr (lt_of_lt_gap_of_close hgap (hclose _ _ _) (hclose _ _ _)))

/-- Every point within `δ` of a margined point is smooth (every window's max is attained once). -/
theorem MaxPool2MarginQ.smooth_of_close {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ) :
    MaxPool2Smooth y := fun ci ho wo ab ab' hne hy => by
  have hab := hm.dom_eq hclose ci ho wo ab hy
  subst hab
  exact lt_of_lt_gap_of_close (hm ci ho wo _ ab' hne (maxPool2Argmax_max x ci ho wo))
    (hclose _ _ _) (hclose _ _ _)

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
  · -- a y-argmax cell is the x-argmax cell, which dominates at x
    intro hy
    have hdom := hm.dom_eq hclose ci (winRow hi) (winCol wi) (winRowMod hi, winColMod wi)
      (fun cd => by rw [hyw]; exact hy cd.1 cd.2)
    intro a b
    have h := maxPool2Argmax_max x ci (winRow hi) (winCol wi) (a, b)
    rw [← hdom, hxw] at h
    exact h
  · -- x-argmax at (hi,wi) ⇒ y-argmax at (hi,wi)
    intro hx a b
    by_cases hEq : ((a, b) : Fin 2 × Fin 2) = (winRowMod hi, winColMod wi)
    · cases hEq; rw [hyw]
    · have hgap := hm ci (winRow hi) (winCol wi)
        (winRowMod hi, winColMod wi) (a, b) (Ne.symm hEq)
        (fun cd => by rw [hxw]; exact hx cd.1 cd.2)
      rw [hxw] at hgap
      exact le_of_lt (lt_of_lt_gap_of_close hgap
        (hclose ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b))
        (hclose ci hi wi))

/-- **Quantitative pool margin up to twins** (2×2), on the PRE-activation: `WindowMarginUpTo`
    at the 2×2 windows. Every window is dead (all cells `≤ 0`), or a cell dominating it is more
    than `2δ` above every other cell except its `T`-twins. Unlike `MaxPool2MarginQ`, a window of
    equal cells qualifies when its cells are twins: the descent rungs take `T` to relate cells
    that read identical input patches, which are equal for every value of the moving parameter.
    That is the case real MNIST forces (a constant background patch makes a window's cells
    equal for every kernel); see scripts/probes/mnist_pool_twin_probe.py. -/
abbrev MaxPool2MarginQUpTo {c h w : Nat} (δ : ℝ)
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (x : Tensor3 c (2*h) (2*w)) : Prop :=
  WindowMarginUpTo winRowInv winColInv δ T x

/-- The post-ReLU `MaxPool2MarginQ` gives the pre-activation `MaxPool2MarginQUpTo`, with any
    twins: the dominating pre-activation cell is positive unless the window is dead, and below a
    positive cell the pre-activation gaps are at least the post-ReLU ones. -/
theorem MaxPool2MarginQ.to_marginQUpTo {c h w : Nat} {δ : ℝ} {x : Tensor3 c (2*h) (2*w)}
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hm : MaxPool2MarginQ δ (fun ci hi wi => max (x ci hi wi) 0)) :
    MaxPool2MarginQUpTo δ T x := by
  intro ci ho wo
  by_cases hd : ∀ cd : Fin 2 × Fin 2, x ci (winRowInv ho cd.1) (winColInv wo cd.2) ≤ 0
  · exact Or.inl hd
  refine Or.inr fun ab ab' hne hdom => Or.inl ?_
  have hpos : 0 < x ci (winRowInv ho ab.1) (winColInv wo ab.2) := by
    by_contra hle
    exact hd fun cd => (hdom cd).trans (not_lt.mp hle)
  have hab : ab ≠ ab' := fun h => hne (by rw [h])
  have hgap : 2 * δ < max (x ci (winRowInv ho ab.1) (winColInv wo ab.2)) 0 -
      max (x ci (winRowInv ho ab'.1) (winColInv wo ab'.2)) 0 :=
    hm ci ho wo ab ab' hab fun cd => max_le_max (hdom cd) le_rfl
  rw [max_eq_left hpos.le] at hgap
  linarith [le_max_left (x ci (winRowInv ho ab'.1) (winColInv wo ab'.2)) 0]

/-- `MaxPool2MarginQ.to_marginQUpTo` in the flat spelling the descent rungs use, the post-ReLU
    tensor as `unflatten ∘ relu ∘ flatten`. -/
theorem MaxPool2MarginQ.to_marginQUpTo_flat {c h w : Nat} {δ : ℝ} {x : Tensor3 c (2*h) (2*w)}
    (T : Fin (2*h) × Fin (2*w) → Fin (2*h) × Fin (2*w) → Prop)
    (hm : MaxPool2MarginQ δ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten x)))) :
    MaxPool2MarginQUpTo δ T x := by
  refine MaxPool2MarginQ.to_marginQUpTo T ?_
  have hx : Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten x)) =
      fun ci hi wi => max (x ci hi wi) 0 := by
    funext ci hi wi
    rw [unflatten_t3Idx, relu_apply_eq_max, flatten_t3Idx]
  rwa [hx] at hm

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

-- ════════════════════════════════════════════════════════════════
-- § Conv-kernel drift: a dense layer with weight sharing
-- ════════════════════════════════════════════════════════════════

/-- `conv2d` through `convPad`: bias plus the kernel-linear form. -/
theorem conv2d_eq_convPad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi =
      b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * convPad kH kW x c kh kw hi wi := rfl

/-- Padded reads are bounded by the input bound (out-of-bounds reads are
    zero). -/
theorem abs_convPad_le {ic h w kH kW : Nat} (x : Tensor3 ic h w) {a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ c i j, |x c i j| ≤ a)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW x c kh kw hi wi| ≤ a := by
  unfold convPad
  split_ifs with h
  · exact hx _ _ _
  · simpa using ha

/-- Flat index of a `Kernel4` entry (the suite's row-major layout). -/
def k4Idx {oc ic kH kW : Nat} (o : Fin oc) (c : Fin ic)
    (kh : Fin kH) (kw : Fin kW) : Fin (oc * ic * kH * kW) :=
  finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, c), kh), kw)

/-- `k4Idx` reads back through `Kernel4.unflatten`. -/
theorem unflatten_k4Idx {oc ic kH kW : Nat} (v : Vec (oc * ic * kH * kW))
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Kernel4.unflatten v o c kh kw = v (k4Idx o c kh kw) := rfl

/-- `Kernel4.flatten` reads off at a `k4Idx` — the forward peer of
    `unflatten_k4Idx`, lifting a per-entry kernel bound to the flattened vector. -/
theorem flatten_k4Idx {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Kernel4.flatten W (k4Idx o c kh kw) = W o c kh kw := by
  simp only [Kernel4.flatten, k4Idx, Equiv.symm_apply_apply]

/-- Every flat kernel index is a `k4Idx` — lets the abstract `∀ idx` gradient
    accuracy be discharged per `(o,cc,kh,kw)` by `cnn_conv2_grad_close`. -/
theorem k4Idx_surj {oc ic kH kW : Nat} (idx : Fin (oc * ic * kH * kW)) :
    ∃ (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
      idx = k4Idx o c kh kw := by
  refine ⟨(finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1).1,
    (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1).2,
    (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2,
    (finProdFinEquiv.symm idx).2, ?_⟩
  simp only [k4Idx, Prod.mk.eta, Equiv.apply_symm_apply]

/-- The output-channel slabs tile the kernel: summing the slab masses over
    the output channels recovers the total `ℓ1` mass. -/
theorem sum_abs_k4 {oc ic kH kW : Nat} (e : Vec (oc * ic * kH * kW)) :
    ∑ idx, |e idx| =
      ∑ o : Fin oc, ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        |e (k4Idx o c kh kw)| := by
  simp only [sum_finProdFinEquiv]; rfl

/-- The `ℓ1` mass of one output-channel slab is at most the total `ℓ1`
    mass — the conv analogue of a dense column being part of the flat
    parameter vector. -/
theorem sum_abs_kernel_slab_le {oc ic kH kW : Nat}
    (e : Vec (oc * ic * kH * kW)) (o : Fin oc) :
    ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, |e (k4Idx o c kh kw)| ≤
      ∑ idx, |e idx| := by
  rw [sum_abs_k4 e]
  exact Finset.single_le_sum (f := fun o => ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
    |e (k4Idx o c kh kw)|) (fun _ _ => by positivity) (Finset.mem_univ o)

/-- Folds the raw `finProdFinEquiv` encoding back into `t3Idx`. -/
theorem t3Idx_def {c h w : Nat} (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    finProdFinEquiv (finProdFinEquiv (ci, hi), wi) = t3Idx ci hi wi := rfl

-- ════════════════════════════════════════════════════════════════
-- § Twin cells: identical zero-padded conv patches
-- ════════════════════════════════════════════════════════════════

/-- **Twin cells of a conv input**: positions `p` and `q` whose zero-padded `kH×kW` input
    patches are identical in every input channel. The conv output is then the same at `p` and
    `q` for EVERY kernel and bias (`conv2d_eq_of_convPatchEq`), so a tie between them is a tie
    along any step on the conv's parameters. A flat image region gives such pairs. -/
def ConvPatchEq {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w) (p q : Fin h × Fin w) : Prop :=
  ∀ (cc : Fin ic) (kh : Fin kH) (kw : Fin kW),
    convPad kH kW x cc kh kw p.1 p.2 = convPad kH kW x cc kh kw q.1 q.2

theorem ConvPatchEq.symm {ic h w kH kW : Nat} {x : Tensor3 ic h w} {p q : Fin h × Fin w}
    (hpq : ConvPatchEq kH kW x p q) : ConvPatchEq kH kW x q p :=
  fun cc kh kw => (hpq cc kh kw).symm

theorem ConvPatchEq.trans {ic h w kH kW : Nat} {x : Tensor3 ic h w} {p q u : Fin h × Fin w}
    (hpq : ConvPatchEq kH kW x p q) (hqu : ConvPatchEq kH kW x q u) : ConvPatchEq kH kW x p u :=
  fun cc kh kw => (hpq cc kh kw).trans (hqu cc kh kw)

/-- Two cells whose patches are entirely zero are twins: the form a blank image region gives,
    and the one a concrete instance discharges cell by cell. -/
theorem ConvPatchEq.of_zero {ic h w kH kW : Nat} {x : Tensor3 ic h w} {p q : Fin h × Fin w}
    (hp : ∀ cc kh kw, convPad kH kW x cc kh kw p.1 p.2 = 0)
    (hq : ∀ cc kh kw, convPad kH kW x cc kh kw q.1 q.2 = 0) : ConvPatchEq kH kW x p q :=
  fun cc kh kw => (hp cc kh kw).trans (hq cc kh kw).symm

/-- Twin cells have equal conv outputs, for every kernel and bias. -/
theorem conv2d_eq_of_convPatchEq {ic oc h w kH kW : Nat} {x : Tensor3 ic h w}
    {p q : Fin h × Fin w} (hpq : ConvPatchEq kH kW x p q) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (o : Fin oc) : conv2d W b x o p.1 p.2 = conv2d W b x o q.1 q.2 := by
  unfold ConvPatchEq at hpq
  simp only [conv2d_eq_convPad, hpq]

/-- **Two-layer twins**: positions `p` and `q` whose two-conv receptive fields are identical.
    For every offset of the outer `kH×kW` window, both reads fall in the zero padding, or both
    land on cells whose inner patches of `x` are identical (`ConvPatchEq`). Then the outer conv's
    input patches at `p` and `q` agree for every inner kernel and bias
    (`convPatchEq_relu_conv`), so a tie between the outer outputs there is a tie along any step
    on the inner conv's parameters. A flat image region away from the border gives such pairs. -/
def ConvPatchEq2 {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w) (p q : Fin h × Fin w) :
    Prop :=
  ∀ (kh : Fin kH) (kw : Fin kW),
    ((kH - 1) / 2 ≤ kh.val + p.1.val ∧ kh.val + p.1.val - (kH - 1) / 2 < h ∧
        (kW - 1) / 2 ≤ kw.val + p.2.val ∧ kw.val + p.2.val - (kW - 1) / 2 < w ↔
      (kH - 1) / 2 ≤ kh.val + q.1.val ∧ kh.val + q.1.val - (kH - 1) / 2 < h ∧
        (kW - 1) / 2 ≤ kw.val + q.2.val ∧ kw.val + q.2.val - (kW - 1) / 2 < w) ∧
    ∀ (hp : (kH - 1) / 2 ≤ kh.val + p.1.val ∧ kh.val + p.1.val - (kH - 1) / 2 < h ∧
        (kW - 1) / 2 ≤ kw.val + p.2.val ∧ kw.val + p.2.val - (kW - 1) / 2 < w)
      (hq : (kH - 1) / 2 ≤ kh.val + q.1.val ∧ kh.val + q.1.val - (kH - 1) / 2 < h ∧
        (kW - 1) / 2 ≤ kw.val + q.2.val ∧ kw.val + q.2.val - (kW - 1) / 2 < w),
      ConvPatchEq kH kW x
        (⟨kh.val + p.1.val - (kH - 1) / 2, hp.2.1⟩, ⟨kw.val + p.2.val - (kW - 1) / 2, hp.2.2.2⟩)
        (⟨kh.val + q.1.val - (kH - 1) / 2, hq.2.1⟩, ⟨kw.val + q.2.val - (kW - 1) / 2, hq.2.2.2⟩)

/-- Two-layer twins are twins of the outer conv's input, for every inner kernel and bias. -/
theorem convPatchEq_relu_conv {ic oc h w kH kW : Nat} {x : Tensor3 ic h w}
    {p q : Fin h × Fin w} (hpq : ConvPatchEq2 kH kW x p q) (W : Kernel4 oc ic kH kW)
    (b : Vec oc) :
    ConvPatchEq kH kW (Tensor3.unflatten (relu (oc * h * w) (Tensor3.flatten (conv2d W b x))))
      p q := by
  intro cc kh kw
  obtain ⟨hiff, hin⟩ := hpq kh kw
  unfold convPad
  by_cases hp : (kH - 1) / 2 ≤ kh.val + p.1.val ∧ kh.val + p.1.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + p.2.val ∧ kw.val + p.2.val - (kW - 1) / 2 < w
  · have hq := hiff.mp hp
    rw [dite_eq_left hp, dite_eq_left hq, unflatten_t3Idx, unflatten_t3Idx, relu_apply_eq_max,
      relu_apply_eq_max, flatten_t3Idx, flatten_t3Idx, conv2d_eq_of_convPatchEq (hin hp hq)]
  · rw [dite_eq_right hp, dite_eq_right fun hq => hp (hiff.mpr hq)]

-- ════════════════════════════════════════════════════════════════
-- § The pool backward selector under the selection margin
-- ════════════════════════════════════════════════════════════════

/-- **Float pool-backward closeness.** Under the pool
    margin the float post-relu argmax matches the real one
    (`isArgmax_iff`), so the pool's backward selector
    `𝟙[(ci,hi,wi) is its window's argmax]·(pooled cotangent)` differs from the
    certified one only through the pooled cotangent value — an indicator
    pass-through (`indicator ∈ {0,1}`), the pool peer of `reluMask_close`.
    The two cotangent values `ay` (float) / `ax` (real) enter only via their
    closeness `|ay − ax| ≤ e`. -/
theorem MaxPool2MarginQ.poolBack_close {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w))
    {ay ax e : ℝ} (ha : |ay - ax| ≤ e) :
    |(if MaxPool2IsArgmax y ci hi wi then ay else 0) -
      (if MaxPool2IsArgmax x ci hi wi then ax else 0)| ≤ e := by
  rw [if_congr (hm.isArgmax_iff hclose ci hi wi) rfl rfl]
  split_ifs <;> simp [ha, (abs_nonneg _).trans ha]

end Proofs
