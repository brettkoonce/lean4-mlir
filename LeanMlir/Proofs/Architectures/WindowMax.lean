import LeanMlir.Proofs.Foundation.Tensor

/-! # `windowMax` — the max pool over a family of product windows

One max pool, proved once. Output cell `(ch, hi, wi)` is the max of the input over the window
`{(r hi a, s wi b) : a, b ∈ Fin k}`, where `r` and `s` are the row and column index maps of the
window. The pools the nets use are instances:

* `maxPool2` (CNN.lean), 2×2 stride 2: `r hi a = 2·hi + a` (`winRowInv`), `k = 2`. Its own
  four-way `max` spelling is kept, because the graph ties and the generated certificates read it;
  `maxPool2_eq_windowMax` is the bridge.
* `maxPool3s2` (MaxPool3s2.lean), He et al.'s 3×3 stride-2 stem pool: `r hi a = 2·hi + a − 1`
  (`win3RowInv`), `k = 3`, and it IS `windowMax` at those maps.

## Smoothness is stated over positions

`WindowSmooth` asks that a cell dominating its window be strictly above every cell at another
input **position**, not at another offset. The two differ when `r` or `s` is not injective: the
3×3/s2 pool's clamped first window names one input cell at two offsets, and those two values are
the same number, so a smoothness condition over offsets could never hold there. For the 2×2 pool
offsets and positions coincide (`windowSmooth_of_maxPool2Smooth`).

## Why overlapping windows cost nothing extra

At a smooth point the pool is locally the reindexing `y ↦ y ∘ σ` (`windowMax_flat_hasFDerivAt`),
`σ` sending each output to its argmax's input position. The argmax is the first maximal offset
in row-major order (`windowArgmax`), the cell the emitted `select_and_scatter` (`GE` select)
picks, so at a tie the gather and the printed scatter still name one cell. Overlapping windows
only make `σ` non-injective, and `reindexCLM`'s adjoint already sums over preimages, so the VJP
(`windowMaxHasVJPAt3`) accumulates over every output whose window selects an input: one term for
tiling windows, up to four for the 3×3/s2 pool. -/

namespace Proofs

open Finset

variable {c h w H W k : Nat} [NeZero k]

-- ════════════════════════════════════════════════════════════════
-- § The forward
-- ════════════════════════════════════════════════════════════════

/-- **The window max pool**, `[c, H, W] → [c, h, w]`: the max of `x ch` over the window
    `{(r hi a, s wi b)}`. -/
noncomputable def windowMax (r : Fin h → Fin k → Fin H) (s : Fin w → Fin k → Fin W)
    (x : Tensor3 c H W) : Tensor3 c h w :=
  fun ch hi wi =>
    (univ : Finset (Fin k × Fin k)).sup' univ_nonempty (fun ab => x ch (r hi ab.1) (s wi ab.2))

variable (r : Fin h → Fin k → Fin H) (s : Fin w → Fin k → Fin W)

/-- Every window cell is ≤ the pooled value. -/
theorem le_windowMax (x : Tensor3 c H W) (ch : Fin c) (hi : Fin h) (wi : Fin w)
    (ab : Fin k × Fin k) :
    x ch (r hi ab.1) (s wi ab.2) ≤ windowMax r s x ch hi wi :=
  le_sup' (f := fun ab : Fin k × Fin k => x ch (r hi ab.1) (s wi ab.2)) (mem_univ ab)

/-- The pooled value is attained by some window cell. -/
theorem windowMax_attained (x : Tensor3 c H W) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    ∃ ab : Fin k × Fin k, windowMax r s x ch hi wi = x ch (r hi ab.1) (s wi ab.2) := by
  obtain ⟨ab, _, hab⟩ := exists_mem_eq_sup' (univ_nonempty (α := Fin k × Fin k))
    (fun ab : Fin k × Fin k => x ch (r hi ab.1) (s wi ab.2))
  exact ⟨ab, hab⟩

-- ════════════════════════════════════════════════════════════════
-- § Magnitude and closeness — what the float tier needs
-- ════════════════════════════════════════════════════════════════

/-- **The pool never grows magnitudes**: it selects an existing window cell. -/
theorem windowMax_abs_le {x : Tensor3 c H W} {A : ℝ} (hx : ∀ ci hi wi, |x ci hi wi| ≤ A)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) : |windowMax r s x ci hi wi| ≤ A := by
  obtain ⟨ab, hab⟩ := windowMax_attained r s x ci hi wi
  rw [hab]; exact hx _ _ _

/-- **The pool is 1-Lipschitz in the sup norm**: each side is ≤ the other plus `e`, from the
    attained cell on one side and `le_windowMax` on the other. -/
theorem windowMax_close (xt xa : Tensor3 c H W) {e : ℝ}
    (hx : ∀ ci hi wi, |xt ci hi wi - xa ci hi wi| ≤ e) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    |windowMax r s xt ci hi wi - windowMax r s xa ci hi wi| ≤ e := by
  have key : ∀ (u v : Tensor3 c H W), (∀ a b d, |u a b d - v a b d| ≤ e) →
      windowMax r s u ci hi wi - windowMax r s v ci hi wi ≤ e := by
    intro u v huv
    obtain ⟨ab, hab⟩ := windowMax_attained r s u ci hi wi
    have hle := le_of_abs_le (huv ci (r hi ab.1) (s wi ab.2))
    have hv := le_windowMax r s v ci hi wi ab
    rw [hab]; linarith
  have h1 := key xt xa hx
  have h2 := key xa xt (fun a b d => by rw [abs_sub_comm]; exact hx a b d)
  rw [abs_sub_le_iff]; exact ⟨h1, by linarith⟩

/-- **The pool shifts with a uniform offset.** If one slab's channel is another's plus the
    constant `δ`, so are their pooled values (`Finset.apply_sup'_eq_sup'_comp` at `(· + δ)`). It
    holds at every point, with no argmax argument. -/
theorem windowMax_shift (x y : Tensor3 c H W) (δ : ℝ) (ci : Fin c)
    (hxy : ∀ p q, x ci p q = y ci p q + δ) (hi : Fin h) (wi : Fin w) :
    windowMax r s x ci hi wi = windowMax r s y ci hi wi + δ := by
  have hg : ∀ p q : ℝ, (p ⊔ q) + δ = (p + δ) ⊔ (q + δ) := fun p q => (max_add_add_right p q δ).symm
  simp only [windowMax, hxy]
  exact (Finset.apply_sup'_eq_sup'_comp Finset.univ_nonempty (fun z : ℝ => z + δ) hg).symm

/-- The pool keeps a nonnegative slab nonnegative. -/
theorem windowMax_nonneg (x : Tensor3 c H W) (hx : ∀ ci p q, 0 ≤ x ci p q) (ci : Fin c)
    (hi : Fin h) (wi : Fin w) : 0 ≤ windowMax r s x ci hi wi := by
  obtain ⟨ab, hab⟩ := windowMax_attained r s x ci hi wi
  rw [hab]; exact hx _ _ _

-- ════════════════════════════════════════════════════════════════
-- § Smoothness predicates
-- ════════════════════════════════════════════════════════════════

/-- **Smoothness**: every window attains its max at exactly one input POSITION. A cell that
    dominates its window is strictly above every cell at another position; other cells may tie
    with each other. A window whose max sits at two positions does not qualify, and there the pool
    has no derivative. -/
def WindowSmooth (x : Tensor3 c H W) : Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w) (ab ab' : Fin k × Fin k),
    (r hi_out ab.1, s wi_out ab.2) ≠ (r hi_out ab'.1, s wi_out ab'.2) →
    (∀ cd : Fin k × Fin k, x ci (r hi_out cd.1) (s wi_out cd.2) ≤ x ci (r hi_out ab.1) (s wi_out ab.2)) →
    x ci (r hi_out ab'.1) (s wi_out ab'.2) < x ci (r hi_out ab.1) (s wi_out ab.2)

omit [NeZero k] in
/-- Windows whose cells at distinct positions have distinct values are smooth. -/
theorem windowSmooth_of_pairwise (x : Tensor3 c H W)
    (hd : ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w) (ab ab' : Fin k × Fin k),
      (r hi_out ab.1, s wi_out ab.2) ≠ (r hi_out ab'.1, s wi_out ab'.2) →
      x ci (r hi_out ab.1) (s wi_out ab.2) ≠ x ci (r hi_out ab'.1) (s wi_out ab'.2)) :
    WindowSmooth r s x :=
  fun ci ho wo ab ab' hne hmax => lt_of_le_of_ne (hmax ab') (hd ci ho wo ab' ab (Ne.symm hne))

omit [NeZero k] in
/-- **Positional injectivity ⇒ smoothness.** If on each channel `(p, q) ↦ x ci p q` is
    injective, no two positions tie. Because smoothness is quantified over positions, the
    injectivity lands directly on the hypothesis, with no decoding from positions back to
    offsets. -/
theorem windowSmooth_of_injective (x : Tensor3 c H W)
    (hinj : ∀ (ci : Fin c) (p p' : Fin H) (q q' : Fin W), x ci p q = x ci p' q' → p = p' ∧ q = q') :
    WindowSmooth r s x := by
  refine windowSmooth_of_pairwise r s x ?_
  intro ci hi_out wi_out ab ab' hne hval
  obtain ⟨hr, hs⟩ := hinj ci _ _ _ _ hval
  exact hne (Prod.ext_iff.mpr ⟨hr, hs⟩)

/-- **Smooth or dead**: every window either has its maximum at one input position, or has every
    cell `≤ 0`. The second case is what a pool AFTER a ReLU needs: a window of dead ReLUs ties at
    `0`, so the pool alone has no derivative there, but `pool ∘ relu` is locally constant at a
    pre-activation whose cells are all strictly negative. -/
def WindowSmoothOrDead (x : Tensor3 c H W) : Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w),
    (∀ cd : Fin k × Fin k, x ci (r hi_out cd.1) (s wi_out cd.2) ≤ 0) ∨
    ∀ ab ab' : Fin k × Fin k,
      (r hi_out ab.1, s wi_out ab.2) ≠ (r hi_out ab'.1, s wi_out ab'.2) →
      (∀ cd : Fin k × Fin k,
        x ci (r hi_out cd.1) (s wi_out cd.2) ≤ x ci (r hi_out ab.1) (s wi_out ab.2)) →
      x ci (r hi_out ab'.1) (s wi_out ab'.2) < x ci (r hi_out ab.1) (s wi_out ab.2)

omit [NeZero k] in
/-- A smooth pool input is smooth-or-dead. -/
theorem windowSmoothOrDead_of_smooth {x : Tensor3 c H W} (hx : WindowSmooth r s x) :
    WindowSmoothOrDead r s x :=
  fun ci ho wo => Or.inr (fun ab ab' => hx ci ho wo ab ab')

/-- **Smooth, dead, or tied only between twins**: every window is entirely `≤ 0`, or its maximum
    is strictly above every cell at another position except positions `T` relates to the
    maximum's. With `T` empty this is `WindowSmoothOrDead`. The twins a parameter gradient can
    afford are cells that are the SAME function of the moving parameter: a tie between them
    persists along the parameter, and the pool may pick either. -/
def WindowSmoothUpTo (T : Fin H × Fin W → Fin H × Fin W → Prop) (x : Tensor3 c H W) : Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w),
    (∀ cd : Fin k × Fin k, x ci (r hi_out cd.1) (s wi_out cd.2) ≤ 0) ∨
    ∀ ab ab' : Fin k × Fin k,
      (r hi_out ab.1, s wi_out ab.2) ≠ (r hi_out ab'.1, s wi_out ab'.2) →
      (∀ cd : Fin k × Fin k,
        x ci (r hi_out cd.1) (s wi_out cd.2) ≤ x ci (r hi_out ab.1) (s wi_out ab.2)) →
      x ci (r hi_out ab'.1) (s wi_out ab'.2) < x ci (r hi_out ab.1) (s wi_out ab.2) ∨
        T (r hi_out ab.1, s wi_out ab.2) (r hi_out ab'.1, s wi_out ab'.2)

omit [NeZero k] in
/-- A smooth-or-dead pool input is smooth up to any twin relation. -/
theorem windowSmoothUpTo_of_smoothOrDead (T : Fin H × Fin W → Fin H × Fin W → Prop)
    {x : Tensor3 c H W} (hx : WindowSmoothOrDead r s x) : WindowSmoothUpTo r s T x :=
  fun ci ho wo => (hx ci ho wo).imp id fun h ab ab' hne hd => Or.inl (h ab ab' hne hd)

/-- **Margin up to twins**, the quantitative `WindowSmoothUpTo`: every window is entirely `≤ 0`,
    or a cell dominating it is more than `2δ` above every cell at another position, except
    positions `T` relates to its own. A perturbation of at most `δ` per entry then keeps every
    such cell strictly below, so the dominating cell keeps dominating; the twins it ties with
    must be the same function of whatever moves the input, and then they stay tied. Stated on the
    pool's input before any ReLU: the descent rungs apply it to the pre-activation, where a dead
    window is one whose cells are all `≤ 0`. -/
def WindowMarginUpTo (δ : ℝ) (T : Fin H × Fin W → Fin H × Fin W → Prop) (x : Tensor3 c H W) :
    Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w),
    (∀ cd : Fin k × Fin k, x ci (r hi_out cd.1) (s wi_out cd.2) ≤ 0) ∨
    ∀ ab ab' : Fin k × Fin k,
      (r hi_out ab.1, s wi_out ab.2) ≠ (r hi_out ab'.1, s wi_out ab'.2) →
      (∀ cd : Fin k × Fin k,
        x ci (r hi_out cd.1) (s wi_out cd.2) ≤ x ci (r hi_out ab.1) (s wi_out ab.2)) →
      2 * δ < x ci (r hi_out ab.1) (s wi_out ab.2) - x ci (r hi_out ab'.1) (s wi_out ab'.2) ∨
        T (r hi_out ab.1, s wi_out ab.2) (r hi_out ab'.1, s wi_out ab'.2)

omit [NeZero k] in
/-- A margin up to twins holds at every smaller margin. -/
theorem WindowMarginUpTo.mono {δ δ' : ℝ} (hδ : δ' ≤ δ) {T : Fin H × Fin W → Fin H × Fin W → Prop}
    {x : Tensor3 c H W} (hx : WindowMarginUpTo r s δ T x) : WindowMarginUpTo r s δ' T x :=
  fun ci ho wo => (hx ci ho wo).imp id fun h ab ab' hne hd =>
    (h ab ab' hne hd).imp (fun hlt => by linarith) id

omit [NeZero k] in
/-- **A margin up to twins from one designated cell per window.** If `T` is an equivalence and
    every live window has a cell `m` with every other cell at `m`'s position, a twin of it, or
    more than `2δ` below it, the margin holds: a cell dominating the window is `m` or a twin of
    `m` (it cannot sit `2δ` below), and twins of `m` inherit `m`'s gaps. The form a concrete
    instance discharges, one certificate per window. -/
theorem windowMarginUpTo_of_cert {δ : ℝ} (hδ : 0 ≤ δ) (T : Fin H × Fin W → Fin H × Fin W → Prop)
    (hsymm : ∀ p q, T p q → T q p) (htrans : ∀ p q u, T p q → T q u → T p u) {x : Tensor3 c H W}
    (hx : ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w),
      (∀ cd : Fin k × Fin k, x ci (r hi_out cd.1) (s wi_out cd.2) ≤ 0) ∨
      ∃ m : Fin k × Fin k, ∀ cd : Fin k × Fin k,
        (r hi_out m.1, s wi_out m.2) = (r hi_out cd.1, s wi_out cd.2) ∨
        T (r hi_out m.1, s wi_out m.2) (r hi_out cd.1, s wi_out cd.2) ∨
        x ci (r hi_out cd.1) (s wi_out cd.2) + 2 * δ < x ci (r hi_out m.1) (s wi_out m.2)) :
    WindowMarginUpTo r s δ T x := by
  intro ci ho wo
  rcases hx ci ho wo with hdead | ⟨m, hm⟩
  · exact Or.inl hdead
  refine Or.inr fun ab ab' hne hdom => ?_
  -- the dominating cell is at `m`'s position or a twin of `m`
  have hab : (r ho m.1, s wo m.2) = (r ho ab.1, s wo ab.2) ∨
      T (r ho m.1, s wo m.2) (r ho ab.1, s wo ab.2) := by
    rcases hm ab with h | h | h
    · exact Or.inl h
    · exact Or.inr h
    · have := hdom m; linarith
  have hxm : x ci (r ho m.1) (s wo m.2) ≤ x ci (r ho ab.1) (s wo ab.2) := hdom m
  rcases hm ab' with h' | h' | h'
  · -- `ab'` sits at `m`'s position: `ab` is `m`'s twin
    rcases hab with h | h
    · exact absurd (h.symm.trans h') hne
    · exact Or.inr (by rw [← h']; exact hsymm _ _ h)
  · rcases hab with h | h
    · exact Or.inr (by rw [← h]; exact h')
    · exact Or.inr (htrans _ _ _ (hsymm _ _ h) h')
  · exact Or.inl (by linarith)

omit [NeZero k] in
/-- At a nonnegative margin the margined input is smooth up to the same twins. -/
theorem windowSmoothUpTo_of_margin {δ : ℝ} (hδ : 0 ≤ δ) (T : Fin H × Fin W → Fin H × Fin W → Prop)
    {x : Tensor3 c H W} (hx : WindowMarginUpTo r s δ T x) : WindowSmoothUpTo r s T x :=
  fun ci ho wo => (hx ci ho wo).imp id fun h ab ab' hne hd =>
    (h ab ab' hne hd).imp (fun hlt => by linarith) id

-- ════════════════════════════════════════════════════════════════
-- § Argmax extractor and the window-max characterisation
-- ════════════════════════════════════════════════════════════════

/-- The offsets attaining the max of the window at output `(co, ho, wo)`, as row-major flat
    indices `a·k + b` (`finProdFinEquiv`). -/
noncomputable def windowMaxOffsets (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w) :
    Finset (Fin (k * k)) :=
  univ.filter fun j => ∀ cd : Fin k × Fin k,
    x co (r ho cd.1) (s wo cd.2) ≤
      x co (r ho (finProdFinEquiv.symm j).1) (s wo (finProdFinEquiv.symm j).2)

theorem windowMaxOffsets_nonempty (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w) :
    (windowMaxOffsets r s x co ho wo).Nonempty := by
  obtain ⟨ab, -, hab⟩ := (univ : Finset (Fin k × Fin k)).exists_max_image
    (fun ab => x co (r ho ab.1) (s wo ab.2)) univ_nonempty
  exact ⟨finProdFinEquiv ab, mem_filter.mpr ⟨mem_univ _, fun cd => by
    rw [Equiv.symm_apply_apply]; exact hab cd (mem_univ cd)⟩⟩

/-- **The first argmax of the window** at output `(co, ho, wo)`, as an offset: the least maximal
    offset in row-major order (`a` major, `windowArgmax_first`). That is the cell the emitted
    `select_and_scatter` routes to: its `GE` select keeps the current pick while it is `≥` the
    next cell, so it ends on the first maximum in window iteration order. Unique under
    `WindowSmooth` up to position, and only `windowArgmax_max` is needed off a tie. -/
noncomputable def windowArgmax (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w) :
    Fin k × Fin k :=
  finProdFinEquiv.symm ((windowMaxOffsets r s x co ho wo).min' (windowMaxOffsets_nonempty r s x co ho wo))

theorem windowArgmax_max (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w)
    (ab : Fin k × Fin k) :
    x co (r ho ab.1) (s wo ab.2) ≤
      x co (r ho (windowArgmax r s x co ho wo).1) (s wo (windowArgmax r s x co ho wo).2) :=
  (mem_filter.mp ((windowMaxOffsets r s x co ho wo).min'_mem
    (windowMaxOffsets_nonempty r s x co ho wo))).2 ab

/-- **No earlier offset attains the max.** Every offset before `windowArgmax` in row-major order
    is strictly below the window max: the first-maximum half of the `select_and_scatter` reading. -/
theorem windowArgmax_first (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w)
    (cd : Fin k × Fin k) (hcd : finProdFinEquiv cd < finProdFinEquiv (windowArgmax r s x co ho wo)) :
    x co (r ho cd.1) (s wo cd.2) <
      x co (r ho (windowArgmax r s x co ho wo).1) (s wo (windowArgmax r s x co ho wo).2) := by
  by_contra hle
  push Not at hle
  have hmem : finProdFinEquiv cd ∈ windowMaxOffsets r s x co ho wo :=
    mem_filter.mpr ⟨mem_univ _, fun ab => by
      rw [Equiv.symm_apply_apply]; exact (windowArgmax_max r s x co ho wo ab).trans hle⟩
  have hmin := (windowMaxOffsets r s x co ho wo).min'_le _ hmem
  rw [windowArgmax, Equiv.apply_symm_apply] at hcd
  exact absurd hmin (not_le.mpr hcd)

/-- If offset `ab` dominates every window cell, the pooled value is the value there. -/
theorem windowMax_eq_at_max (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w)
    (ab : Fin k × Fin k)
    (h_max : ∀ cd : Fin k × Fin k, x co (r ho cd.1) (s wo cd.2) ≤ x co (r ho ab.1) (s wo ab.2)) :
    windowMax r s x co ho wo = x co (r ho ab.1) (s wo ab.2) :=
  le_antisymm (sup'_le _ _ (fun cd _ => h_max cd)) (le_windowMax r s x co ho wo ab)

theorem windowMax_eq_argmax_value (x : Tensor3 c H W) (co : Fin c) (ho : Fin h) (wo : Fin w) :
    windowMax r s x co ho wo =
      x co (r ho (windowArgmax r s x co ho wo).1) (s wo (windowArgmax r s x co ho wo).2) :=
  windowMax_eq_at_max r s x co ho wo _ (windowArgmax_max r s x co ho wo)

-- ════════════════════════════════════════════════════════════════
-- § The fixed gather: the pool with its routing frozen
-- ════════════════════════════════════════════════════════════════

omit [NeZero k] in
/-- **The window gather at a fixed selection** `σ`: output `(ch, hi, wi)` reads the window cell
    at offset `σ ch hi wi`. Linear in `x`, with no argmax to decide. Wherever `σ` names a cell
    dominating every window, it IS the pool (`windowMax_eq_windowGather`); that is how a pool
    with tied windows is handled, the ties being routed to one fixed cell. -/
noncomputable def windowGather (σ : Fin c → Fin h → Fin w → Fin k × Fin k) (x : Tensor3 c H W) :
    Tensor3 c h w :=
  fun ch hi wi => x ch (r hi (σ ch hi wi).1) (s wi (σ ch hi wi).2)

/-- **The pool is the gather at a dominating selection.** -/
theorem windowMax_eq_windowGather (σ : Fin c → Fin h → Fin w → Fin k × Fin k) (x : Tensor3 c H W)
    (hdom : ∀ ch hi wi (cd : Fin k × Fin k),
      x ch (r hi cd.1) (s wi cd.2) ≤ x ch (r hi (σ ch hi wi).1) (s wi (σ ch hi wi).2)) :
    windowMax r s x = windowGather r s σ x :=
  funext fun ch => funext fun hi => funext fun wi =>
    windowMax_eq_at_max r s x ch hi wi _ (hdom ch hi wi)

-- ════════════════════════════════════════════════════════════════
-- § Local linearisation, the smooth-point Jacobian and the VJP
-- ════════════════════════════════════════════════════════════════

/-- For each output flat index, the flat index of its argmax's input position: the carrier of
    the local linearisation. Not injective when windows overlap; `reindexCLM`'s adjoint sums over
    preimages, which is where the backward accumulates. -/
noncomputable def windowLocalReindex (x : Tensor3 c H W) (k_out : Fin (c * h * w)) :
    Fin (c * H * W) :=
  let r1 := finProdFinEquiv.symm k_out
  let wo : Fin w := r1.2
  let r2 := finProdFinEquiv.symm r1.1
  let co : Fin c := r2.1
  let ho : Fin h := r2.2
  let ab := windowArgmax r s x co ho wo
  finProdFinEquiv (finProdFinEquiv (co, r ho ab.1), s wo ab.2)

/-- **Smooth-point local linearisation.** Near `flatten x` the flattened pool agrees with the
    reindex `y ↦ y ∘ σ`: every window keeps its argmax, since finitely many strict inequalities
    persist on a neighbourhood (`Filter.eventually_all`). An offset naming the argmax's own
    position is equal to it, not below, so the domination argument branches on positions. -/
theorem windowMax_flat_hasFDerivAt (x : Tensor3 c H W) (h_smooth : WindowSmooth r s x) :
    HasFDerivAt (fun v : Vec (c * H * W) => Tensor3.flatten (windowMax r s (Tensor3.unflatten v)))
      (reindexCLM (windowLocalReindex r s x)) (Tensor3.flatten x) := by
  refine (reindexCLM (windowLocalReindex r s x)).hasFDerivAt.congr_of_eventuallyEq ?_
  have hmax : ∀ᶠ y in nhds (Tensor3.flatten x), ∀ (co : Fin c) (ho : Fin h) (wo : Fin w)
      (cd : Fin k × Fin k), Tensor3.unflatten y co (r ho cd.1) (s wo cd.2) ≤
        Tensor3.unflatten y co (r ho (windowArgmax r s x co ho wo).1)
          (s wo (windowArgmax r s x co ho wo).2) := by
    have hcont : ∀ co hi wi, ContinuousAt
        (fun y : Vec (c * H * W) => Tensor3.unflatten y co hi wi) (Tensor3.flatten x) :=
      fun _ _ _ => (continuous_apply _).continuousAt
    simp only [Filter.eventually_all]
    intro co ho wo cd
    by_cases hab : (r ho cd.1, s wo cd.2) =
        (r ho (windowArgmax r s x co ho wo).1, s wo (windowArgmax r s x co ho wo).2)
    · obtain ⟨hr, hs⟩ := Prod.mk.inj hab
      exact Filter.Eventually.of_forall fun _ => by rw [hr, hs]
    · refine ((hcont _ _ _).eventually_lt (hcont _ _ _) ?_).mono fun _ => le_of_lt
      simpa [Tensor3.unflatten_flatten] using
        h_smooth co ho wo _ cd (Ne.symm hab) (windowArgmax_max r s x co ho wo)
  filter_upwards [hmax] with y hy
  funext k_out
  exact windowMax_eq_at_max r s (Tensor3.unflatten y) _ _ _ _ (hy _ _ _)

/-- **Smooth-point Jacobian.** `pdiv3` is the 0/1 indicator that the local reindex sends output
    `(co, ho, wo)` to input `(ci, hi_in, wi_in)`. Left as the reindex equation: with overlapping
    windows an input has no single owning window to decode it into. -/
theorem pdiv3_windowMax_smooth (x : Tensor3 c H W) (h_smooth : WindowSmooth r s x)
    (ci : Fin c) (hi_in : Fin H) (wi_in : Fin W) (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv3 (windowMax r s) x ci hi_in wi_in co ho wo =
      (if windowLocalReindex r s x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)
        then (1 : ℝ) else 0) := by
  have h_fderiv := windowMax_flat_hasFDerivAt r s x h_smooth
  unfold pdiv3
  rw [pdiv_eq_of_hasFDerivAt h_fderiv]
  show reindexCLM (windowLocalReindex r s x)
        (basisVec (finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)))
        (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = _
  -- `basisVec j i` IS `if i = j then 1 else 0`, so the reindex equation is the condition.
  rw [reindexCLM_apply]

/-- **The VJP witness.** The backward accumulates `dy` over every output whose window selects
    this input. -/
noncomputable def windowMaxHasVJPAt3 (x : Tensor3 c H W) (h_smooth : WindowSmooth r s x) :
    HasVJPAt3 (windowMax r s : Tensor3 c H W → Tensor3 c h w) x where
  backward dy ci hi_in wi_in :=
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      (if windowLocalReindex r s x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)
        then (1 : ℝ) else 0) * dy co ho wo
  correct dy ci hi_in wi_in := by
    refine Finset.sum_congr rfl (fun co _ => Finset.sum_congr rfl
      (fun ho _ => Finset.sum_congr rfl (fun wo _ => ?_)))
    rw [pdiv3_windowMax_smooth r s x h_smooth]

-- ════════════════════════════════════════════════════════════════
-- § The flat bridge
-- ════════════════════════════════════════════════════════════════

/-- The flattened window max, the `Vec`-level form a codegen op denotes. -/
noncomputable def windowMaxFlat : Vec (c * H * W) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (windowMax r s (Tensor3.unflatten v))

theorem windowMaxFlat_differentiableAt (x : Tensor3 c H W) (h_smooth : WindowSmooth r s x) :
    DifferentiableAt ℝ (windowMaxFlat (c := c) r s) (Tensor3.flatten x) :=
  (windowMax_flat_hasFDerivAt r s x h_smooth).differentiableAt

/-- Flattened magnitude bound. -/
theorem windowMaxFlat_abs_le {v : Vec (c * H * W)} {A : ℝ} (hv : ∀ i, |v i| ≤ A)
    (i : Fin (c * h * w)) : |windowMaxFlat r s v i| ≤ A := by
  have huf : ∀ ci hi wi, |Tensor3.unflatten v ci hi wi| ≤ A := by
    intro ci hi wi; simp only [Tensor3.unflatten]; exact hv _
  simp only [windowMaxFlat, Tensor3.flatten]
  exact windowMax_abs_le r s huf _ _ _

/-- Flattened closeness. -/
theorem windowMaxFlat_close (vt va : Vec (c * H * W)) {e : ℝ} (hv : ∀ i, |vt i - va i| ≤ e)
    (i : Fin (c * h * w)) : |windowMaxFlat r s vt i - windowMaxFlat r s va i| ≤ e := by
  have huf : ∀ ci hi wi,
      |Tensor3.unflatten vt ci hi wi - Tensor3.unflatten va ci hi wi| ≤ e := by
    intro ci hi wi; simp only [Tensor3.unflatten]; exact hv _
  simp only [windowMaxFlat, Tensor3.flatten]
  exact windowMax_close r s (Tensor3.unflatten vt) (Tensor3.unflatten va) huf _ _ _

/-- `windowMaxFlat` is continuous (a `sup'` of coordinates). -/
@[fun_prop]
theorem windowMaxFlat_continuous : Continuous (windowMaxFlat (c := c) r s) := by
  refine continuous_pi (fun i => ?_)
  show Continuous (fun v => Tensor3.flatten (windowMax r s (Tensor3.unflatten v)) i)
  simp only [Tensor3.flatten, windowMax, Tensor3.unflatten]
  exact Continuous.finset_sup'_apply Finset.univ_nonempty (fun ab _ => continuous_apply _)

end Proofs
