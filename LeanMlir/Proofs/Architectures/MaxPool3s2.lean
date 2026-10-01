import LeanMlir.Proofs.Architectures.WindowMax

/-! # `maxPool3s2` — the 3×3 stride-2 max pool of He et al.'s ResNet stem

`maxPool2` (CNN.lean) is 2×2, stride 2, non-overlapping. Every ResNet in He et al.
(18/34/50/101/152) specifies a 3×3 stride-2 pool after the stem conv; this file is that op.

## Which 3×3 pool: the paper's symmetric padding

This file implements He et al. / torchvision: `nn.MaxPool2d(3, stride=2, padding=1)` —
symmetric padding, so window `i` covers input `[2i−1, 2i+1]`.

XLA's `reduce_window(…, 'SAME')` would be a different function: on a 112→56 axis it gives
`pad_total = max((56−1)·2 + 3 − 112, 0) = 1`, split `pad_low = 0, pad_high = 1` — window
`i = [2i, 2i+2]`, padded at the end. Measured on device at `n = 12`: `SAME` windows peak at
`[2,4,6,8,10,11]`, symmetric at `[1,3,5,7,9,11]`; the two grids are offset by one input position.
The JAX reference's `max_pool2d` uses the same symmetric padding `(p, p)` with `p = (size−1)//2`.

At 2×2 symmetric `(k−1)//2` padding is `p = 0`, bit-identical to `SAME`, so the choice only
matters for the 3×3 pools (the ResNet-34 and ResNet-50 ImageNet stems).

## The padding needs no extended-reals type

`reduce_window` pads with `-∞`, which `Tensor3 _ _ _ = … → ℝ` cannot hold. It does not need to:
**for `max`, clamping the index is equivalent to `-∞` padding.** The only out-of-range read is
`2i−1` at `i = 0`, and Nat's truncated subtraction clamps it to `0` — a cell the window already
contains at offset `a = 1`. So `max` over the clamped triple equals `max` over the unpadded pair,
which is exactly what `-∞` padding computes. `win3RowInv_first_dup` is that statement, and
`win3ColInv_first_dup` its column half.

The symmetric form needs **no `min`**: the upper end `2(h−1)+2−1 = 2h−1` is in range by
construction, so truncated subtraction is the whole story.

## The pool is a `windowMax` instance

`maxPool3s2` is `windowMax win3RowInv win3ColInv` (WindowMax.lean), and every analytic fact here
is that file's lemma at these window maps. What is particular to this pool is the window
geometry: the clamped duplicate (`win3RowInv_first_dup`), which is why smoothness is stated over
positions, and the overlap (`win3Row_mem_le_two`): an input lies in up to four windows, so where
`maxPool2`'s backward is a lookup, this one accumulates. The witness is `maxPool3s2HasVJPAt3`, with
its flat form `maxPool3s2FlatHasVJPAt`; the ResNet-34/50 stems, their seals and the float stem
bridge build on it. -/

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

/-- **The padding statement.** In the FIRST window offset `a = 0` duplicates `a = 1` rather
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
-- § The pool and its predicates, as `windowMax` instances
-- ════════════════════════════════════════════════════════════════

/-- **3×3 stride-2 symmetrically-padded max pool**, `[c, 2h, 2w] → [c, h, w]`: the max over the
    window `[2i−1, 2i+1] × [2j−1, 2j+1]`, clamped at the near edge (= `-∞` padded, header). -/
noncomputable abbrev maxPool3s2 {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Tensor3 c h w :=
  windowMax win3RowInv win3ColInv x

/-- **Smoothness**: every 3×3 window attains its max at exactly one input position
    (`WindowSmooth`). Stated over positions, so the clamped duplicate in the first window is not a
    tie; a window whose max sits at two positions (an all-zero post-ReLU window) does not
    qualify. -/
abbrev MaxPool3s2Smooth {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  WindowSmooth win3RowInv win3ColInv x

/-- **Positional injectivity ⇒ `MaxPool3s2Smooth`**, the discharge used by
    `BatchSeal.ctConv_pool_smooth` for the ResNet-34 and ResNet-50 full-width seals: one
    injectivity argument in place of a per-window case split. -/
theorem maxPool3s2Smooth_of_injective {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (hinj : ∀ (ci : Fin c) (r r' : Fin (2 * h)) (s s' : Fin (2 * w)),
              x ci r s = x ci r' s' → r = r' ∧ s = s') :
    MaxPool3s2Smooth x :=
  windowSmooth_of_injective _ _ x hinj

/-- **Smooth or dead** (`WindowSmoothOrDead`): every 3×3 window has its maximum at one position,
    or every cell `≤ 0`. The ResNet stems state their pool condition in this form
    (`StemPoolSmoothAt`), and `maxPool3s2Flat_relu_eventuallyEq` is the lemma that uses it. -/
abbrev MaxPool3s2SmoothOrDead {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  WindowSmoothOrDead win3RowInv win3ColInv x

/-- A smooth pool input is smooth-or-dead. -/
theorem maxPool3s2SmoothOrDead_of_smooth {c h w : Nat} {x : Tensor3 c (2 * h) (2 * w)}
    (hx : MaxPool3s2Smooth x) : MaxPool3s2SmoothOrDead x :=
  windowSmoothOrDead_of_smooth _ _ hx

/-- **Smooth, dead, or tied only between twins** (`WindowSmoothUpTo`): the ResNet stem's twins are
    cells reading identical input patches (`StemPoolTwinAt`). -/
abbrev MaxPool3s2SmoothUpTo {c h w : Nat}
    (T : Fin (2 * h) × Fin (2 * w) → Fin (2 * h) × Fin (2 * w) → Prop)
    (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  WindowSmoothUpTo win3RowInv win3ColInv T x

/-- A smooth-or-dead pool input is smooth up to any twin relation. -/
theorem maxPool3s2SmoothUpTo_of_smoothOrDead {c h w : Nat}
    (T : Fin (2 * h) × Fin (2 * w) → Fin (2 * h) × Fin (2 * w) → Prop)
    {x : Tensor3 c (2 * h) (2 * w)} (hx : MaxPool3s2SmoothOrDead x) : MaxPool3s2SmoothUpTo T x :=
  windowSmoothUpTo_of_smoothOrDead _ _ T hx

/-- **The overlap fact, stated rather than assumed**: an input row lies in at most TWO windows —
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
-- § Argmax, local reindex and the VJP witness
-- ════════════════════════════════════════════════════════════════

/-- A (not necessarily unique) argmax of the 3×3 window at output `(co, ho, wo)`. -/
noncomputable abbrev maxPool3s2Argmax {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) : Fin 3 × Fin 3 :=
  windowArgmax win3RowInv win3ColInv x co ho wo

theorem maxPool3s2Argmax_max {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) (ab : Fin 3 × Fin 3) :
    x co (win3RowInv ho ab.1) (win3ColInv wo ab.2) ≤
      x co (win3RowInv ho (maxPool3s2Argmax x co ho wo).1)
            (win3ColInv wo (maxPool3s2Argmax x co ho wo).2) :=
  windowArgmax_max _ _ x co ho wo ab

/-- If `(a, b)` dominates every window cell, the pooled value is the value there. -/
theorem maxPool3s2_eq_at_max {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) (a b : Fin 3)
    (h_max : ∀ a' b' : Fin 3,
      x co (win3RowInv ho a') (win3ColInv wo b') ≤ x co (win3RowInv ho a) (win3ColInv wo b)) :
    maxPool3s2 x co ho wo = x co (win3RowInv ho a) (win3ColInv wo b) :=
  windowMax_eq_at_max _ _ x co ho wo (a, b) fun cd => h_max cd.1 cd.2

/-- For each output flat index, the flat index of its argmax's input position. Not injective:
    two overlapping windows may select the same input. -/
noncomputable abbrev maxPool3s2LocalReindex {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (k_out : Fin (c * h * w)) : Fin (c * (2 * h) * (2 * w)) :=
  windowLocalReindex win3RowInv win3ColInv x k_out

/-- **The VJP witness.** The backward accumulates `dy` over every output whose window selects
    this input, at most 4 of them (`win3Row_mem_le_two` squared). The backward is spelled out
    rather than taken from `windowMaxHasVJPAt3`, so that unfolding this name exposes the sum the
    backward ties match. -/
noncomputable def maxPool3s2HasVJPAt3 {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x) :
    HasVJPAt3 (maxPool3s2 : Tensor3 c (2 * h) (2 * w) → Tensor3 c h w) x where
  backward dy ci hi_in wi_in :=
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      (if maxPool3s2LocalReindex x (finProdFinEquiv (finProdFinEquiv (co, ho), wo))
            = finProdFinEquiv (finProdFinEquiv (ci, hi_in), wi_in)
        then (1 : ℝ) else 0) * dy co ho wo
  correct := (windowMaxHasVJPAt3 _ _ x h_smooth).correct

-- ════════════════════════════════════════════════════════════════
-- § The flat bridge — what the `SHlo` op's `den` names
-- ════════════════════════════════════════════════════════════════

/-- Flattened 3×3/s2 pool, the `Vec`-level form the codegen denotes. Spelled through
    `maxPool3s2`, which is `windowMaxFlat` at the 3×3/s2 maps by unfolding. -/
noncomputable def maxPool3s2Flat (c h w : Nat) :
    Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (maxPool3s2 (Tensor3.unflatten v))

theorem maxPool3s2Flat_differentiableAt {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x) :
    DifferentiableAt ℝ (maxPool3s2Flat c h w) (Tensor3.flatten x) :=
  windowMaxFlat_differentiableAt _ _ x h_smooth

/-- The VJP of `maxPool3s2Flat` at a flattened input satisfying `MaxPool3s2Smooth`:
    `maxPool3s2HasVJPAt3` moved to `Vec` form. -/
noncomputable def maxPool3s2FlatHasVJPAt {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool3s2Smooth x) :
    HasVJPAt (maxPool3s2Flat c h w) (Tensor3.flatten x) :=
  HasVJPAt3.toHasVJPAt (maxPool3s2HasVJPAt3 x h_smooth)

/-- Flattened magnitude bound, the form `floatClose_maxPool3s2` threads. -/
theorem maxPool3s2Flat_abs_le {c h w : Nat} {v : Vec (c * (2 * h) * (2 * w))} {A : ℝ}
    (hv : ∀ k, |v k| ≤ A) (k : Fin (c * h * w)) :
    |maxPool3s2Flat c h w v k| ≤ A :=
  windowMaxFlat_abs_le _ _ hv k

/-- Flattened closeness: the pool is 1-Lipschitz in the sup norm. -/
theorem maxPool3s2Flat_close {c h w : Nat} (vt va : Vec (c * (2 * h) * (2 * w)))
    {e : ℝ} (hv : ∀ k, |vt k - va k| ≤ e) (k : Fin (c * h * w)) :
    |maxPool3s2Flat c h w vt k - maxPool3s2Flat c h w va k| ≤ e :=
  windowMaxFlat_close _ _ vt va hv k

/-- `maxPool3s2Flat` is continuous (a `sup'` of coordinates). -/
@[fun_prop]
theorem maxPool3s2Flat_continuous (c h w : Nat) : Continuous (maxPool3s2Flat c h w) :=
  windowMaxFlat_continuous _ _

/-- **The 3×3/s2 pool shifts with a uniform offset**, at every point of the ray, which is what
    lets the carrier cross the only real kink in the net. -/
theorem maxPool3s2_shift {c h w : Nat} (x y : Tensor3 c (2 * h) (2 * w)) (δ : ℝ) (ci : Fin c)
    (hxy : ∀ r s, x ci r s = y ci r s + δ) (hi : Fin h) (wi : Fin w) :
    maxPool3s2 x ci hi wi = maxPool3s2 y ci hi wi + δ :=
  windowMax_shift _ _ x y δ ci hxy hi wi

/-- The pool keeps a nonnegative slab nonnegative (it selects a window cell). -/
theorem maxPool3s2_nonneg {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (hx : ∀ ci r s, 0 ≤ x ci r s) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    0 ≤ maxPool3s2 x ci hi wi :=
  windowMax_nonneg _ _ x hx ci hi wi

end Proofs
