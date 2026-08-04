import LeanMlir.Proofs.Architectures.CNN

/-! # `maxPool3s2` — the 3×3 stride-2 max pool of He et al.'s ResNet stem

**Written 2026-08-03.** `planning/rsb_a3_r50_verified.md` / the stem-pool finding.

The verified path's only pooling op is `maxPool2` — 2×2, stride 2, **non-overlapping**. Every
ResNet in He et al. (18/34/50/101/152) specifies a **3×3 stride-2** pool after the stem conv, so
`resnet34Verified` has been pooling a different function from the paper *and* from the reference
it is paired against. This file is the missing op.

## ⚠⚠ WHICH 3×3 pool — the PAPER's, which is not what the JAX reference emits today

This file implements **He et al. / torchvision**: `nn.MaxPool2d(3, stride=2, padding=1)` —
**symmetric** padding, so **window `i` covers input `[2i−1, 2i+1]`**.

⚠ The repo's JAX references emit `reduce_window(…, (1,1,3,3), (1,1,2,2), 'SAME')`, and XLA's
`SAME` on a 112→56 axis gives `pad_total = max((56−1)·2 + 3 − 112, 0) = 1`, split
`pad_low = 0, pad_high = 1` — window `i = [2i, 2i+2]`, padded at the **end**. Measured on device
at `n = 12`: `SAME` windows peak at `[2,4,6,8,10,11]`, symmetric at `[1,3,5,7,9,11]`. The two
grids are **offset by one input position** and are different functions everywhere.

Paper-faithfulness is the goal, so this file is symmetric and the JAX `max_pool2d` helper moves to
match. ⚠ Until both land and are re-run, verified and JAX disagree at the stem pool.

⭐ Measured: symmetric `(k−1)//2` padding is **bit-identical to `SAME` for every 2×2 pool**, so no
cifar/mnist net moves — only the 3×3 users (R34-ImageNet, R50).

## ⭐ THE PADDING NEEDS NO EXTENDED-REALS TYPE

`reduce_window` pads with `-∞`, which `Tensor3 _ _ _ = … → ℝ` cannot hold. It does not need to:
**for `max`, clamping the index is equivalent to `-∞` padding.** The only out-of-range read is
`2i−1` at `i = 0`, and Nat's truncated subtraction clamps it to `0` — a cell the window already
contains at offset `a = 1`. So `max` over the clamped triple equals `max` over the unpadded pair,
which is exactly what `-∞` padding computes. `win3RowInv_first_dup` is that statement.

⭐ The symmetric form needs **no `min`**: the upper end `2(h−1)+2−1 = 2h−1` is in range by
construction, so truncated subtraction is the whole story.

## The shape of the VJP, and why overlap costs less than it looks

`maxPool2`'s backward is a **lookup** (`dy` at `hi/2`), sound only because 2×2 windows are
disjoint. Here windows overlap — odd input `p` lies in windows `(p−1)/2` and `(p+1)/2` — so an
input feeds up to **4** outputs and the backward must **accumulate**.

That needs no new analytic argument. `HasVJPAt3.correct` already states the backward as
`∑ co ∑ ho ∑ wo, pdiv3 f x … * dy co ho wo` — a sum over *all* outputs; `maxPool2`'s
`codegen_matches_canonical` merely *collapses* it to one term using disjointness. Here it
collapses to ≤4. The generic route (`maxPool2LocalReindex` → `reindexCLM` → `pdiv3`) is
indifferent: at a smooth point the pool is locally a reindexing map, and overlap only makes that
map non-injective, which `reindexCLM`'s adjoint already handles by summing over preimages.

⚠ **This file is the DEFINITIONAL layer.** The `HasVJPAt3` witness mirrors
`maxPool2_has_vjp_at3` (33 declarations in `CNN.lean`) and is not built here yet. Nothing
downstream references these definitions, so nothing renders or moves. -/

namespace Proofs

open Finset

-- ════════════════════════════════════════════════════════════════
-- § Window index helpers — 3 wide, stride 2, symmetric pad 1
-- ════════════════════════════════════════════════════════════════

/-- Input row of offset `a ∈ Fin 3` inside output window `hi_out`: `2·hi + a − 1`, in **Nat**.
    The truncated subtraction *is* the low pad (see the header) — it is the only clamp needed,
    because the high end `2(h−1)+2−1 = 2h−1` is in range by construction. -/
def win3RowInv {h : Nat} (hi_out : Fin h) (a : Fin 3) : Fin (2 * h) :=
  ⟨2 * hi_out.val + a.val - 1, by
    have h1 := hi_out.isLt; have h2 := a.isLt; omega⟩

/-- Column peer of `win3RowInv`. -/
def win3ColInv {w : Nat} (wi_out : Fin w) (b : Fin 3) : Fin (2 * w) :=
  ⟨2 * wi_out.val + b.val - 1, by
    have h1 := wi_out.isLt; have h2 := b.isLt; omega⟩

@[simp] theorem win3RowInv_val {h : Nat} (hi_out : Fin h) (a : Fin 3) :
    (win3RowInv hi_out a).val = 2 * hi_out.val + a.val - 1 := rfl

@[simp] theorem win3ColInv_val {w : Nat} (wi_out : Fin w) (b : Fin 3) :
    (win3ColInv wi_out b).val = 2 * wi_out.val + b.val - 1 := rfl

/-- ⭐ **The padding statement.** In the FIRST window offset `a = 0` duplicates `a = 1` rather
    than reading out of range — exactly what a `-∞` pad contributes to a `max`. -/
theorem win3RowInv_first_dup {h : Nat} (hi_out : Fin h) (hfirst : hi_out.val = 0) :
    win3RowInv hi_out ⟨0, by omega⟩ = win3RowInv hi_out ⟨1, by omega⟩ := by
  apply Fin.ext
  show 2 * hi_out.val + 0 - 1 = 2 * hi_out.val + 1 - 1
  omega

theorem win3ColInv_first_dup {w : Nat} (wi_out : Fin w) (hfirst : wi_out.val = 0) :
    win3ColInv wi_out ⟨0, by omega⟩ = win3ColInv wi_out ⟨1, by omega⟩ := by
  apply Fin.ext
  show 2 * wi_out.val + 0 - 1 = 2 * wi_out.val + 1 - 1
  omega

-- ════════════════════════════════════════════════════════════════
-- § The forward
-- ════════════════════════════════════════════════════════════════

/-- **3×3 stride-2 symmetrically-padded max pool**, `[c, 2h, 2w] → [c, h, w]`: the max over the
    window `[2i−1, 2i+1] × [2j−1, 2j+1]`, clamped at the near edge (= `-∞` padded, header). -/
noncomputable def maxPool3s2 {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Tensor3 c h w :=
  fun ch hi wi =>
    (univ : Finset (Fin 3 × Fin 3)).sup' univ_nonempty
      (fun ab => x ch (win3RowInv hi ab.1) (win3ColInv wi ab.2))

/-- Every window cell is ≤ the pooled value. -/
theorem le_maxPool3s2 {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (ch : Fin c) (hi : Fin h) (wi : Fin w) (ab : Fin 3 × Fin 3) :
    x ch (win3RowInv hi ab.1) (win3ColInv wi ab.2) ≤ maxPool3s2 x ch hi wi :=
  le_sup' (f := fun ab : Fin 3 × Fin 3 =>
    x ch (win3RowInv hi ab.1) (win3ColInv wi ab.2)) (mem_univ ab)

/-- The pooled value is attained by some window cell. -/
theorem maxPool3s2_attained {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    ∃ ab : Fin 3 × Fin 3,
      maxPool3s2 x ch hi wi = x ch (win3RowInv hi ab.1) (win3ColInv wi ab.2) := by
  obtain ⟨ab, _, hab⟩ := exists_mem_eq_sup' (univ_nonempty (α := Fin 3 × Fin 3))
    (fun ab : Fin 3 × Fin 3 => x ch (win3RowInv hi ab.1) (win3ColInv wi ab.2))
  exact ⟨ab, hab⟩

-- ════════════════════════════════════════════════════════════════
-- § Magnitude and closeness — what the FLOAT bridge needs
-- ════════════════════════════════════════════════════════════════
--
-- ⚠⚠ These exist because `r34Forward` — the skeleton `r34_floatBridges` is stated over — supplies
-- its pool CONCRETELY (`floatBridges_maxPool`) where it supplies all 16 blocks abstractly. So
-- moving the net's pool moved a `rfl` that had nothing to do with the codegen, and it surfaced as
-- a **`(deterministic) timeout at whnf`** on `resnet34Forward_full_pc_eq_skeleton` rather than as
-- a type error. ⚠ Raising the heartbeat budget — the recorded fix for the superficially identical
-- symptom in `xla_pjrt_handoff.md` §0.2 increment 2 — would have spent unbounded compute on a
-- proposition that was FALSE. *A `whnf` timeout on an `rfl` is not evidence about the budget; the
-- first question is whether the two sides should be equal at all.*

/-- **The 3×3 pool never grows magnitudes** — it selects an existing window cell.
    ⭐ `Finset.sup'` again makes the window size stop mattering: `maxPool2_abs_le` needs a nested
    `abs_max_le (abs_max_le _ _) (abs_max_le _ _)` for 4 cells, which at 9 would be worse; here it
    is `sup'_le` plus one `le_sup'`, independent of the window. Fourth collapse of the same kind. -/
theorem maxPool3s2_abs_le {c h w : Nat} {x : Tensor3 c (2 * h) (2 * w)} {A : ℝ}
    (hx : ∀ ci hi wi, |x ci hi wi| ≤ A) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    |maxPool3s2 x ci hi wi| ≤ A := by
  obtain ⟨ab, hab⟩ := maxPool3s2_attained x ci hi wi
  rw [hab]; exact hx _ _ _

/-- **The 3×3 pool is 1-Lipschitz in the sup norm** — the peer of `maxPool2_close`. Standard
    sup-vs-sup argument: each side is ≤ the other plus `e`, from `sup'_le` and `le_sup'`. -/
theorem maxPool3s2_close {c h w : Nat} (xt xa : Tensor3 c (2 * h) (2 * w)) {e : ℝ}
    (hx : ∀ ci hi wi, |xt ci hi wi - xa ci hi wi| ≤ e)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    |maxPool3s2 xt ci hi wi - maxPool3s2 xa ci hi wi| ≤ e := by
  have key : ∀ (u v : Tensor3 c (2 * h) (2 * w)),
      (∀ a b d, |u a b d - v a b d| ≤ e) →
      maxPool3s2 u ci hi wi - maxPool3s2 v ci hi wi ≤ e := by
    intro u v huv
    obtain ⟨ab, hab⟩ := maxPool3s2_attained u ci hi wi
    have hle : u ci (win3RowInv hi ab.1) (win3ColInv wi ab.2)
        - v ci (win3RowInv hi ab.1) (win3ColInv wi ab.2) ≤ e :=
      le_of_abs_le (huv _ _ _)
    have hv := le_maxPool3s2 v ci hi wi ab
    rw [hab]; linarith
  have h1 := key xt xa hx
  have h2 := key xa xt (fun a b d => by rw [abs_sub_comm]; exact hx a b d)
  rw [abs_sub_le_iff]; exact ⟨h1, by linarith⟩

-- (the two flattened peers live at the end of the file, after `maxPool3s2Flat` is defined.)

-- ════════════════════════════════════════════════════════════════
-- § Smoothness and the argmax predicate
-- ════════════════════════════════════════════════════════════════

/-- **Smoothness**: every 3×3 window has a *strict* argmax. Stated as "distinct offsets that land
    on distinct input POSITIONS have distinct values", so the clamped duplicate in the first
    window (`a = 0` ≡ `a = 1`) is not counted as a tie. ⚠ That carve-out is forced by the padding
    and has no `maxPool2` analogue — there, distinct offsets always meant distinct positions. -/
def MaxPool3s2Smooth {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w) (ab ab' : Fin 3 × Fin 3),
    (win3RowInv hi_out ab.1, win3ColInv wi_out ab.2) ≠
      (win3RowInv hi_out ab'.1, win3ColInv wi_out ab'.2) →
    x ci (win3RowInv hi_out ab.1) (win3ColInv wi_out ab.2) ≠
      x ci (win3RowInv hi_out ab'.1) (win3ColInv wi_out ab'.2)

/-- ⭐ **Positional injectivity ⇒ `MaxPool3s2Smooth`** — the discharge lemma for the whole-net live
    and seal witnesses, the peer of `MnistCNN`'s `maxPool2Smooth_of_injective`. One injectivity
    argument in place of `36·c·h·w` per-window `decide`s (9 offsets pairwise, against 2×2's 6), which
    at ResNet-34's stem is why case-bashing is not an option.

    ⚠⚠ **And it is STRICTLY SHORTER than its 2×2 peer, for the reason the padding forced.**
    `MaxPool2Smooth` is quantified over **offsets**, so its discharge lemma has to get from
    "the two input positions coincide" back to "the two offsets coincide" — two `Fin.mk.injEq` +
    `omega` decodes, valid only because there distinct offsets always meant distinct positions.
    `MaxPool3s2Smooth` is quantified over **positions** precisely because that is false here (the
    clamped duplicate `a = 0 ≡ a = 1` in the first window, `win3RowInv_first_dup`), so injectivity
    lands directly on the hypothesis and the decode step does not exist. **The carve-out that made
    the predicate awkward to state is what makes it cheap to discharge.**

    The hypothesis is the same one the 2×2 sites already supply: on each channel the position map
    `(r, s) ↦ x ci r s` is injective. -/
theorem maxPool3s2Smooth_of_injective {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (hinj : ∀ (ci : Fin c) (r r' : Fin (2 * h)) (s s' : Fin (2 * w)),
              x ci r s = x ci r' s' → r = r' ∧ s = s') :
    MaxPool3s2Smooth x := by
  intro ci hi_out wi_out ab ab' hne hval
  obtain ⟨hr, hs⟩ := hinj ci _ _ _ _ hval
  exact hne (Prod.ext_iff.mpr ⟨hr, hs⟩)

/-- Input `(ci, hi_in, wi_in)` attains the max of the window at output `(ho, wo)`.
    ⚠ Unlike `MaxPool2IsArgmax` this takes the OUTPUT position explicitly: with overlapping
    windows an input has no single owning window, which is the whole difference. -/
def MaxPool3s2IsArgmaxAt {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (ci : Fin c) (ho : Fin h) (wo : Fin w)
    (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w)) : Prop :=
  (∃ ab : Fin 3 × Fin 3, win3RowInv ho ab.1 = hi_in ∧ win3ColInv wo ab.2 = wi_in) ∧
  x ci hi_in wi_in = maxPool3s2 x ci ho wo

/-- ⭐ **The overlap fact, stated rather than assumed**: an input row lies in at most TWO windows —
    `p/2` and `(p+1)/2`. With symmetric padding the shared cell is at ODD `p` (window `(p−1)/2`
    takes it at offset 2, window `(p+1)/2` at offset 0); even `p` lies in exactly one. So an input
    feeds at most 4 outputs and the backward accumulates at most 4 terms — the count `maxPool2`
    does not have. -/
theorem win3Row_mem_le_two {h : Nat} (p : Fin (2 * h)) (hi_out : Fin h)
    (hmem : ∃ a : Fin 3, win3RowInv hi_out a = p) :
    hi_out.val = p.val / 2 ∨ 2 * hi_out.val = p.val + 1 := by
  obtain ⟨a, ha⟩ := hmem
  have hv : 2 * hi_out.val + a.val - 1 = p.val := congrArg Fin.val ha
  have h3 := a.isLt
  have hi := hi_out.isLt
  have hp := p.isLt
  omega

-- ════════════════════════════════════════════════════════════════
-- § Argmax extractor and the window-max characterisation
-- ════════════════════════════════════════════════════════════════

/-- A (not necessarily unique) argmax of the 3×3 window at output `(co, ho, wo)`.
    Unique under `MaxPool3s2Smooth` *up to position* — the clamped duplicate in the first
    window is two offsets naming one cell. -/
noncomputable def maxPool3s2Argmax {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) : Fin 3 × Fin 3 :=
  Classical.choose
    ((univ : Finset (Fin 3 × Fin 3)).exists_max_image
      (fun ab => x co (win3RowInv ho ab.1) (win3ColInv wo ab.2)) univ_nonempty)

theorem maxPool3s2Argmax_max {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) (ab : Fin 3 × Fin 3) :
    x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) ≤
      x co (win3RowInv ho (maxPool3s2Argmax x co ho wo).1)
            (win3ColInv wo (maxPool3s2Argmax x co ho wo).2) :=
  (Classical.choose_spec
    ((univ : Finset (Fin 3 × Fin 3)).exists_max_image
      (fun ab' => x co (win3RowInv ho ab'.1) (win3ColInv wo ab'.2))
      univ_nonempty)).2 ab (mem_univ ab)

/-- If `(a, b)` dominates every window cell, the pooled value is the value there.
    ⭐ The `sup'` formulation makes this two lines where `maxPool2_eq_at_max` needs a
    four-way `fin_cases` against an explicit `max (max _ _) (max _ _)` — and nine ways here. -/
theorem maxPool3s2_eq_at_max {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) (a b : Fin 3)
    (h_max : ∀ a' b' : Fin 3,
      x co (win3RowInv ho a') (win3ColInv wo b') ≤
        x co (win3RowInv ho a) (win3ColInv wo b)) :
    maxPool3s2 x co ho wo = x co (win3RowInv ho a) (win3ColInv wo b) :=
  le_antisymm
    (sup'_le _ _ (fun ab' _ => h_max ab'.1 ab'.2))
    (le_sup' (f := fun ab : Fin 3 × Fin 3 =>
      x co (win3RowInv ho ab.1) (win3ColInv wo ab.2)) (mem_univ (a, b)))

theorem maxPool3s2_eq_argmax_value {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    maxPool3s2 x co ho wo =
      x co (win3RowInv ho (maxPool3s2Argmax x co ho wo).1)
            (win3ColInv wo (maxPool3s2Argmax x co ho wo).2) :=
  maxPool3s2_eq_at_max x co ho wo _ _ (fun a' b' => maxPool3s2Argmax_max x co ho wo (a', b'))

-- ════════════════════════════════════════════════════════════════
-- § Local linearisation (the reindex σ)
-- ════════════════════════════════════════════════════════════════

/-- For each output flat index, the flat index of its argmax's input position.
    ⚠ **Not injective** — two overlapping windows may select the same input. That is exactly
    what makes the backward accumulate, and `reindexCLM`'s adjoint already sums over preimages,
    so nothing here needs to change relative to `maxPool2LocalReindex`. -/
noncomputable def maxPool3s2LocalReindex {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (k_out : Fin (c * h * w)) : Fin (c * (2 * h) * (2 * w)) :=
  let r1 := finProdFinEquiv.symm k_out
  let wo : Fin w := r1.2
  let r2 := finProdFinEquiv.symm r1.1
  let co : Fin c := r2.1
  let ho : Fin h := r2.2
  let ab := maxPool3s2Argmax x co ho wo
  finProdFinEquiv (finProdFinEquiv (co, win3RowInv ho ab.1), win3ColInv wo ab.2)

/-- **Smooth-point local linearisation.** On a metric ball around `flatten x` the flattened
    pool agrees with the reindex `y ↦ y ∘ σ`. Mirrors `maxPool2_flat_hasFDerivAt`; the one
    structural change is that the gap function and the domination argument branch on whether the
    two offsets name the same **position** (the clamped duplicate), not on whether the offsets
    are equal — `MaxPool3s2Smooth` says nothing about coincident positions because there the
    values are literally the same number. -/
theorem maxPool3s2_flat_hasFDerivAt {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (h_smooth : MaxPool3s2Smooth x)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    HasFDerivAt
      (fun v : Vec (c * (2 * h) * (2 * w)) =>
        Tensor3.flatten (maxPool3s2 (Tensor3.unflatten v)))
      (reindexCLM (maxPool3s2LocalReindex x))
      (Tensor3.flatten x) := by
  haveI : Nonempty (Fin c) := ⟨⟨0, hc⟩⟩
  haveI : Nonempty (Fin h) := ⟨⟨0, hh⟩⟩
  haveI : Nonempty (Fin w) := ⟨⟨0, hw⟩⟩
  let samePos : Fin h → Fin w → (Fin 3 × Fin 3) → (Fin 3 × Fin 3) → Prop :=
    fun ho wo ab ab' =>
      (win3RowInv ho ab.1, win3ColInv wo ab.2) = (win3RowInv ho ab'.1, win3ColInv wo ab'.2)
  let gap : Fin c × Fin h × Fin w × (Fin 3 × Fin 3) × (Fin 3 × Fin 3) → ℝ :=
    fun p => if samePos p.2.1 p.2.2.1 p.2.2.2.1 p.2.2.2.2 then 1
             else |x p.1 (win3RowInv p.2.1 p.2.2.2.1.1) (win3ColInv p.2.2.1 p.2.2.2.1.2)
                  - x p.1 (win3RowInv p.2.1 p.2.2.2.2.1) (win3ColInv p.2.2.1 p.2.2.2.2.2)|
  have hgap_pos : ∀ p, 0 < gap p := by
    intro ⟨co, ho, wo, ab, ab'⟩
    show 0 < (if samePos ho wo ab ab' then (1 : ℝ) else _)
    by_cases hab : samePos ho wo ab ab'
    · rw [if_pos hab]; norm_num
    · rw [if_neg hab]
      exact abs_pos.mpr (sub_ne_zero.mpr (h_smooth co ho wo ab ab' hab))
  let univ_S : Finset (Fin c × Fin h × Fin w × (Fin 3 × Fin 3) × (Fin 3 × Fin 3)) := univ
  set r_raw := univ_S.inf' univ_nonempty gap with hr_raw_def
  have hr_raw_pos : 0 < r_raw := by
    refine (Finset.lt_inf'_iff _).mpr ?_
    intro p _; exact hgap_pos p
  set r := r_raw / 4 with hr_def
  have hr_pos : 0 < r := by show 0 < r_raw / 4; linarith
  have hgap_le : ∀ co ho wo ab ab', gap (co, ho, wo, ab, ab') ≥ r_raw :=
    fun co ho wo ab ab' => Finset.inf'_le _ (mem_univ _)
  have h_local : Set.EqOn
      (fun v : Vec (c * (2 * h) * (2 * w)) =>
        Tensor3.flatten (maxPool3s2 (Tensor3.unflatten v)))
      (reindexCLM (maxPool3s2LocalReindex x) : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w))
      (Metric.ball (Tensor3.flatten x) r) := by
    intro y hy
    have hy_norm : ‖y - Tensor3.flatten x‖ < r := by
      rwa [Metric.mem_ball, dist_eq_norm] at hy
    have hy_coord : ∀ k, |y k - Tensor3.flatten x k| < r := by
      intro k
      have h1 : ‖(y - Tensor3.flatten x) k‖ ≤ ‖y - Tensor3.flatten x‖ :=
        norm_le_pi_norm (y - Tensor3.flatten x) k
      rw [Real.norm_eq_abs] at h1
      have : |y k - Tensor3.flatten x k| ≤ ‖y - Tensor3.flatten x‖ := by
        show |(y - Tensor3.flatten x) k| ≤ _
        exact h1
      linarith
    funext k_out
    set r1 := finProdFinEquiv.symm k_out with hr1
    set wo : Fin w := r1.2 with hwo
    set r2 := finProdFinEquiv.symm r1.1 with hr2
    set co : Fin c := r2.1 with hco
    set ho : Fin h := r2.2 with hho
    set ab := maxPool3s2Argmax x co ho wo with hab
    have h_max_y : ∀ a' b' : Fin 3,
        Tensor3.unflatten y co (win3RowInv ho a') (win3ColInv wo b') ≤
        Tensor3.unflatten y co (win3RowInv ho ab.1) (win3ColInv wo ab.2) := by
      intro a' b'
      by_cases h_eq : samePos ho wo (a', b') ab
      · -- The two offsets name ONE cell (the clamped duplicate): equal by congruence.
        have h1 : win3RowInv ho a' = win3RowInv ho ab.1 := congrArg Prod.fst h_eq
        have h2 : win3ColInv wo b' = win3ColInv wo ab.2 := congrArg Prod.snd h_eq
        rw [h1, h2]
      · have h_le_x : x co (win3RowInv ho a') (win3ColInv wo b') ≤
                      x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) :=
          maxPool3s2Argmax_max x co ho wo (a', b')
        have h_ne_x : x co (win3RowInv ho a') (win3ColInv wo b') ≠
                      x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) :=
          h_smooth co ho wo (a', b') ab h_eq
        have h_diff_x : x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) -
                        x co (win3RowInv ho a') (win3ColInv wo b') ≥ r_raw := by
          have h_gap := hgap_le co ho wo (a', b') ab
          have h_gap_expanded : gap (co, ho, wo, (a', b'), ab) =
              |x co (win3RowInv ho a') (win3ColInv wo b') -
               x co (win3RowInv ho ab.1) (win3ColInv wo ab.2)| := by
            show (if samePos ho wo (a', b') ab then (1 : ℝ) else _) = _
            rw [if_neg h_eq]
          rw [h_gap_expanded] at h_gap
          rw [abs_sub_comm] at h_gap
          have h_pos : 0 ≤ x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) -
                       x co (win3RowInv ho a') (win3ColInv wo b') := by
            have := lt_of_le_of_ne h_le_x h_ne_x; linarith
          rwa [abs_of_nonneg h_pos] at h_gap
        set k_ab : Fin (c * (2 * h) * (2 * w)) :=
          finProdFinEquiv (finProdFinEquiv (co, win3RowInv ho ab.1), win3ColInv wo ab.2)
        set k_ab' : Fin (c * (2 * h) * (2 * w)) :=
          finProdFinEquiv (finProdFinEquiv (co, win3RowInv ho a'), win3ColInv wo b')
        have h_unflat_ab : Tensor3.unflatten y co (win3RowInv ho ab.1) (win3ColInv wo ab.2)
                          = y k_ab := rfl
        have h_unflat_ab' : Tensor3.unflatten y co (win3RowInv ho a') (win3ColInv wo b')
                           = y k_ab' := rfl
        have h_flat_x_ab : Tensor3.flatten x k_ab =
            x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) := by
          show x (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab).1).1
                  (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab).1).2
                  (finProdFinEquiv.symm k_ab).2 = _
          simp [k_ab, Equiv.symm_apply_apply]
        have h_flat_x_ab' : Tensor3.flatten x k_ab' =
            x co (win3RowInv ho a') (win3ColInv wo b') := by
          show x (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab').1).1
                  (finProdFinEquiv.symm (finProdFinEquiv.symm k_ab').1).2
                  (finProdFinEquiv.symm k_ab').2 = _
          simp [k_ab', Equiv.symm_apply_apply]
        rw [h_unflat_ab, h_unflat_ab']
        have hy_ab := hy_coord k_ab
        have hy_ab' := hy_coord k_ab'
        rw [h_flat_x_ab] at hy_ab
        rw [h_flat_x_ab'] at hy_ab'
        have h_lhs_ge : y k_ab - y k_ab' ≥
            (x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) -
             x co (win3RowInv ho a') (win3ColInv wo b')) - 2 * r := by
          have h1 := abs_sub_lt_iff.mp hy_ab
          have h2 := abs_sub_lt_iff.mp hy_ab'
          linarith
        have h_2r_lt : 2 * r < r_raw := by show 2 * (r_raw / 4) < r_raw; linarith
        linarith
    show maxPool3s2 (Tensor3.unflatten y) co ho wo = y (maxPool3s2LocalReindex x k_out)
    rw [maxPool3s2_eq_at_max (Tensor3.unflatten y) co ho wo ab.1 ab.2 h_max_y]
    show Tensor3.unflatten y co (win3RowInv ho ab.1) (win3ColInv wo ab.2) =
         y (maxPool3s2LocalReindex x k_out)
    rfl
  exact (reindexCLM (maxPool3s2LocalReindex x)).hasFDerivAt.congr_of_eventuallyEq
    (h_local.eventuallyEq_of_mem (Metric.ball_mem_nhds _ hr_pos))

-- ════════════════════════════════════════════════════════════════
-- § The smooth-point Jacobian and the VJP witness
-- ════════════════════════════════════════════════════════════════

/-- **Smooth-point Jacobian.** `pdiv3` is the 0/1 indicator that the local reindex sends output
    `(co, ho, wo)` to input `(ci, hi_in, wi_in)`.

    ⚠⚠ **This is where the overlapping case genuinely differs from `maxPool2`, and it differs by
    being SIMPLER to state.** `pdiv3_maxPool2_smooth` decodes the condition into
    `co = ci ∧ ho = winRow hi_in ∧ wo = winCol wi_in ∧ IsArgmax` — legitimate there because each
    input has exactly ONE owning window, so `winRow`/`winCol` name it. Here an input has up to
    two windows per axis and no such decoding exists. Leaving the condition as the reindex
    equation is both correct and shorter; the accumulation then happens in `correct`'s sum over
    outputs, with no extra argument. -/
theorem pdiv3_maxPool3s2_smooth {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x)
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv3 maxPool3s2 x ci hi_in wi_in co ho wo =
      (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)
        then (1 : ℝ) else 0) := by
  have hc : 0 < c := Fin.pos ci
  have hh : 0 < h := Fin.pos ho
  have hw : 0 < w := Fin.pos wo
  have h_fderiv := maxPool3s2_flat_hasFDerivAt x h_smooth hc hh hw
  unfold pdiv3 pdiv
  rw [h_fderiv.fderiv]
  show reindexCLM (maxPool3s2LocalReindex x)
        (basisVec (finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)))
        (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) = _
  -- ⭐ `rw` closes this by `rfl`: `basisVec j i` IS `if i = j then 1 else 0`, and the reindex
  -- equation is exactly the condition. `maxPool2`'s peer needs a further `h_sigma` decoding step
  -- to reach its `winRow`/`winCol` form; leaving the condition as the reindex equation (§ above)
  -- means there is nothing left to decode.
  rw [reindexCLM_apply]

/-- **The VJP witness.** The backward accumulates `dy` over every output whose window selects
    this input — at most 4 of them (`win3Row_mem_le_two` squared). `maxPool2`'s peer is a single
    lookup; this is the same statement without the disjointness collapse. -/
noncomputable def maxPool3s2_has_vjp_at3 {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x) :
    HasVJPAt3 (maxPool3s2 : Tensor3 c (2 * h) (2 * w) → Tensor3 c h w) x where
  backward dy ci hi_in wi_in :=
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)
        then (1 : ℝ) else 0) * dy co ho wo
  correct dy ci hi_in wi_in := by
    refine Finset.sum_congr rfl (fun co _ => Finset.sum_congr rfl
      (fun ho _ => Finset.sum_congr rfl (fun wo _ => ?_)))
    rw [pdiv3_maxPool3s2_smooth x h_smooth]

-- ════════════════════════════════════════════════════════════════
-- § The flat bridge — what the `SHlo` op's `den` will name
-- ════════════════════════════════════════════════════════════════

/-- Flattened 3×3/s2 pool, the `Vec`-level form the codegen denotes. -/
noncomputable def maxPool3s2Flat (c h w : Nat) :
    Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (maxPool3s2 (Tensor3.unflatten v))

theorem maxPool3s2Flat_differentiableAt {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    DifferentiableAt ℝ (maxPool3s2Flat c h w) (Tensor3.flatten x) :=
  (maxPool3s2_flat_hasFDerivAt x h_smooth hc hh hw).differentiableAt

noncomputable def maxPool3s2Flat_has_vjp_at {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x) :
    HasVJPAt (maxPool3s2Flat c h w) (Tensor3.flatten x) :=
  hasVJPAt3_to_hasVJPAt (maxPool3s2_has_vjp_at3 x h_smooth)

/-- Flattened magnitude bound — the form the R34 forward float bridge threads
    (`maxPoolFlat_abs_le`'s peer). -/
theorem maxPool3s2Flat_abs_le {c h w : Nat} {v : Vec (c * (2 * h) * (2 * w))} {A : ℝ}
    (hv : ∀ k, |v k| ≤ A) (k : Fin (c * h * w)) :
    |maxPool3s2Flat c h w v k| ≤ A := by
  have huf : ∀ ci hi wi, |Tensor3.unflatten v ci hi wi| ≤ A := by
    intro ci hi wi; simp only [Tensor3.unflatten]; exact hv _
  simp only [maxPool3s2Flat, Tensor3.flatten]
  exact maxPool3s2_abs_le huf _ _ _

/-- Flattened closeness — `maxPoolFlat_close`'s peer. -/
theorem maxPool3s2Flat_close {c h w : Nat} (vt va : Vec (c * (2 * h) * (2 * w)))
    {e : ℝ} (hv : ∀ k, |vt k - va k| ≤ e) (k : Fin (c * h * w)) :
    |maxPool3s2Flat c h w vt k - maxPool3s2Flat c h w va k| ≤ e := by
  have huf : ∀ ci hi wi,
      |Tensor3.unflatten vt ci hi wi - Tensor3.unflatten va ci hi wi| ≤ e := by
    intro ci hi wi; simp only [Tensor3.unflatten]; exact hv _
  simp only [maxPool3s2Flat, Tensor3.flatten]
  exact maxPool3s2_close (Tensor3.unflatten vt) (Tensor3.unflatten va) huf _ _ _

end Proofs
