import LeanMlir.Proofs.Foundation.IntervalBound
import LeanMlir.Proofs.Foundation.ListDot

/-! # CROWN: a linear-relaxation bound on the seam IBP already certifies through

`IntervalBound.lean` concretizes to an interval after *every* layer, so the box
grows multiplicatively with depth: each layer throws away the correlations
between neurons. CROWN (Zhang et al. 2018) never concretizes in the middle. It
carries a **linear function of the input** backward through the net — relaxing
each unstable ReLU by a linear lower/upper envelope — and concretizes **once**,
at the end.

Why that is tighter, in one line: IBP bounds each pre-activation with
`⟨W1ᵢ, x₀⟩ ∓ ε‖W1ᵢ‖₁` and *then* combines; CROWN forms the composite row
`A = Σₜ aₜ · W1ₜ` **first** and takes `ε‖A‖₁` **once**. Cancellation between the
rows of `W1` survives in `A` and is destroyed by the per-row `‖·‖₁`.

**CROWN-IBP**, which is what this file supports: the per-neuron pre-activation
bounds `[l, u]` that decide each ReLU's relaxation are taken from **IBP** — here
literally `denseLo W1 …` / `denseHi W1 …`, the box `IntervalBound.lean` already
proves — and only the output pass is CROWN. (Full CROWN would re-derive `[l,u]`
by running itself on every prefix: strictly tighter, much more work.)

## What is certified, and why it is a MARGIN and not a box

`certified_of_boxSound` (`IntervalBound.lean`) consumes a bracket on the logit
*vector* and asks the boxes to separate. CROWN does not produce one usefully:
bounding `f · y` and `f · j` independently and then separating discards the
correlation between them, which is a large fraction of the available tightening.
So the capstone here is stated on the margin `f · y − f · j` directly, via
`certified_of_marginPos` — the same seam idea (the certificate never inspects
*how* the bound was obtained), one rung lower. `certified_of_marginPos` is
generic: a margin-direct *interval* bound plugs into it too, with no CROWN.

## Relaxation, and the rational-size question

Both envelopes are stated so a generator can discharge them by `norm_num` on
rationals, and — critically — so the slope may be **rounded**:

* lower: `α · z ≤ relu z` for ANY `α ∈ [0,1]` (`relu_lower_envelope`) — soundness
  is insensitive to `α`, so it is purely a tightness knob;
* upper: `relu z ≤ s · (z − l)` on `[l,u]` whenever `l ≤ 0`, `0 ≤ s` and
  `u ≤ s · (u − l)` (`relu_upper_envelope`). The hypothesis is stated
  multiplicatively, NOT as `s = u/(u−l)`, so `s` may be any rational at or above
  the chord slope — in particular the chord ROUNDED UP to a `/2^k` grid.

That rounding is what keeps the coefficients small. Left unrounded, `u/(u−l)`
carries the layer-1 denominators into every entry of `A` and reproduces the
LipSDP tier's ~230-digit blow-up. Measured (`scripts/crown_ibp_probe.py`,
`planning/crown_ibp.md` §5.5): at `k = 8` the rounding costs ZERO images on both
trained nets at every radius, so the coefficients stay at the same `/256` scale
as the weights themselves.

Depth: this file does the single backward step a `dense ∘ relu ∘ dense` net
needs, where `crownRow` IS the matrix-level back-substitution. A general
multi-layer `LinSound` would be real machinery with no consumer yet — the conv
tier stays on IBP (max-pool is not an elementwise nonlinearity), and the trained
dense nets are two-layer.

Everything is elementary and closes under `propext / Classical.choice /
Quot.sound`. Engine only; the generated instance is a separate file. -/

namespace Proofs
namespace LipschitzCertDemo

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The margin seam: a positive margin on the box ⇒ L∞ certificate
-- ════════════════════════════════════════════════════════════════

/-- **Margin positivity ⇒ `L∞` certificate.** If for every wrong class the
    margin `f · y − f · j` is strictly positive at every point of the box
    `x ∓ ε`, then `y` is the strict argmax under every perturbation with
    `|δ i| ≤ ε` coordinatewise.

    The peer of `certified_of_boxSound` one rung lower: it says nothing about
    HOW the margin was bounded, so a CROWN bound and a margin-direct interval
    bound discharge it the same way. Bounding the margin rather than the two
    logits separately is itself a tightening — it keeps the correlation between
    `f · y` and `f · j` that a box separation throws away. -/
theorem certified_of_marginPos {n k : ℕ}
    {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}
    {x : EuclideanSpace ℝ (Fin n)} {ε : ℝ} {y : Fin k}
    (h : ∀ j, j ≠ y → ∀ x' : EuclideanSpace ℝ (Fin n),
      InBoxE (fun i => x i - ε) (fun i => x i + ε) x' → 0 < f x' y - f x' j) :
    CertifiedAtLinf f ε x y := by
  intro δ hδ j hj
  have hbox : InBoxE (fun i => x i - ε) (fun i => x i + ε) (x + δ) := by
    intro i
    have h1 := abs_le.mp (hδ i)
    have hxi : (x + δ) i = x i + δ i := rfl
    exact ⟨by rw [hxi]; linarith [h1.1], by rw [hxi]; linarith [h1.2]⟩
  linarith [h j hj (x + δ) hbox]

-- ════════════════════════════════════════════════════════════════
-- § The ReLU envelopes
-- ════════════════════════════════════════════════════════════════

/-- **Lower envelope.** `α · z ≤ relu z` for ANY `α ∈ [0,1]`, with no reference
    to `[l,u]` at all. Soundness is therefore insensitive to the choice of `α`:
    optimizing it (α-CROWN) is a *tightness* question, never a correctness one,
    and a rounded `α` needs no re-verification. -/
theorem relu_lower_envelope {α z : ℝ} (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    α * z ≤ max z 0 := by
  by_cases hz : z ≤ 0
  · rw [max_eq_right hz]; nlinarith
  · rw [max_eq_left (le_of_not_ge hz)]; nlinarith [le_of_not_ge hz]

/-- **Upper envelope.** On `[l, u]` with `l ≤ 0`, any nonneg `s` at or above the
    chord slope dominates `relu`. The chord condition is stated as
    `u ≤ s * (u - l)` rather than `s = u / (u - l)`, which is what lets a
    generator ROUND `s` up to a `/2^k` grid and still discharge it by
    `norm_num` — the rational-size lever (see the header).

    Proof: `g z = s·(z−l) − relu z` is affine on `[0,u]` with `g 0 = −s·l ≥ 0`
    and `g u = s·(u−l) − u ≥ 0`, hence nonneg between; below `0` it is trivial. -/
theorem relu_upper_envelope {l u s z : ℝ} (hl : l ≤ 0) (hlz : l ≤ z) (hzu : z ≤ u)
    (hs0 : 0 ≤ s) (hs : u ≤ s * (u - l)) : max z 0 ≤ s * (z - l) := by
  by_cases hz : z ≤ 0
  · rw [max_eq_right hz]
    exact mul_nonneg hs0 (by linarith)
  · have hz0 : 0 ≤ z := le_of_not_ge hz
    rw [max_eq_left hz0]
    nlinarith [mul_nonneg (sub_nonneg.mpr hzu) (mul_nonneg hs0 (neg_nonneg.mpr hl)),
               mul_nonneg hz0 (sub_nonneg.mpr hs)]

-- ════════════════════════════════════════════════════════════════
-- § Per-neuron relaxation
-- ════════════════════════════════════════════════════════════════

/-- **`a · z + c` is a sound linear lower bound for `v · relu z` on `[l, u]`.**

    One neuron's contribution to a margin, relaxed. `v` is that neuron's
    coefficient in the margin row `W2 y · − W2 j ·`; its SIGN decides which
    envelope may be used, which is why the instances below split on it and not
    on the neuron alone. -/
def ReluLB (v l u a c : ℝ) : Prop :=
  ∀ z, l ≤ z → z ≤ u → a * z + c ≤ v * max z 0

/-- A neuron that is dead on the whole box (`u ≤ 0`) contributes exactly `0` —
    both envelopes are exact and the coefficient vanishes. -/
theorem reluLB_dead {v l u : ℝ} (hu : u ≤ 0) : ReluLB v l u 0 0 := by
  intro z _ hzu
  rw [max_eq_right (hzu.trans hu)]
  simp

/-- A neuron that is active on the whole box (`0 ≤ l`) is EXACT: `relu z = z`,
    so it passes its margin coefficient through unrelaxed. This is where CROWN
    keeps everything IBP would have widened. -/
theorem reluLB_active {v l u : ℝ} (hl : 0 ≤ l) : ReluLB v l u v 0 := by
  intro z hlz _
  rw [max_eq_left (hl.trans hlz)]
  simp

/-- Unstable neuron, POSITIVE margin coefficient: take the lower envelope, whose
    slope `α` is unconstrained in `[0,1]`. -/
theorem reluLB_unstable_pos {v l u α : ℝ} (hv : 0 ≤ v) (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    ReluLB v l u (v * α) 0 := by
  intro z _ _
  calc v * α * z + 0 = v * (α * z) := by ring
    _ ≤ v * max z 0 := mul_le_mul_of_nonneg_left (relu_lower_envelope h0 h1) hv

/-- Unstable neuron, NEGATIVE margin coefficient: multiplying by `v ≤ 0` flips
    the inequality, so the *upper* envelope is what lower-bounds the
    contribution. This is the only branch that carries a constant, and the only
    one whose slope must be verified against the chord. -/
theorem reluLB_unstable_neg {v l u s : ℝ} (hv : v ≤ 0) (hl : l ≤ 0) (hs0 : 0 ≤ s)
    (hs : u ≤ s * (u - l)) : ReluLB v l u (v * s) (-(v * s * l)) := by
  intro z hlz hzu
  calc v * s * z + -(v * s * l) = v * (s * (z - l)) := by ring
    _ ≤ v * max z 0 :=
        mul_le_mul_of_nonpos_left (relu_upper_envelope hl hlz hzu hs0 hs) hv

/-- Transport a `ReluLB` along equalities of its coefficient and constant — the
    join between the *engine's* branch-selected form and the *generator's*
    emitted literals. -/
theorem ReluLB.congr {v l u a a' c c' : ℝ} (h : ReluLB v l u a c)
    (ha : a' = a) (hc : c' = c) : ReluLB v l u a' c' := by
  rw [ha, hc]; exact h

/-- The relaxation coefficient, with the branch chosen by the box and the sign
    of the margin coefficient `v`. -/
noncomputable def relaxA (l u α s v : ℝ) : ℝ :=
  if u ≤ 0 then 0 else if 0 ≤ l then v else if 0 ≤ v then v * α else v * s

/-- The relaxation constant. Only the unstable/negative branch carries one. -/
noncomputable def relaxC (l u α s v : ℝ) : ℝ :=
  if u ≤ 0 then 0 else if 0 ≤ l then 0 else if 0 ≤ v then 0 else -(v * s * l)

/-- **All four branches in one lemma, uniform in `v`.** The chord condition is
    required only where it is used — on a genuinely unstable neuron — so a
    stable neuron discharges it vacuously and may carry `s = 0`.

    This is what keeps a generated instance affordable: the relaxation is proved
    once per NEURON (16 facts), not once per (class, neuron) pair (144), because
    nothing here depends on which wrong class is being separated. -/
theorem reluLB_relax {l u α s v : ℝ} (h0 : 0 ≤ α) (h1 : α ≤ 1) (hs0 : 0 ≤ s)
    (hs : 0 < u → l < 0 → u ≤ s * (u - l)) :
    ReluLB v l u (relaxA l u α s v) (relaxC l u α s v) := by
  unfold relaxA relaxC
  split_ifs with hu hl hv
  · exact reluLB_dead hu
  · exact reluLB_active hl
  · exact reluLB_unstable_pos hv h0 h1
  · exact reluLB_unstable_neg (le_of_not_ge hv) (le_of_not_ge hl) hs0
      (hs (lt_of_not_ge hu) (lt_of_not_ge hl))

-- ════════════════════════════════════════════════════════════════
-- § Back-substitution and concretization
-- ════════════════════════════════════════════════════════════════

/-- **The CROWN row.** Back-substitute the per-neuron relaxation coefficients
    through the first layer: `A = Σₜ aₜ · W1ₜ`. Forming this composite BEFORE
    taking any norm is the entire mechanism — cancellation between the rows of
    `W1` survives here and is destroyed by IBP's per-row `‖·‖₁`. -/
noncomputable def crownRow {n h : ℕ} (a : Fin h → ℝ) (W1 : Fin h → Fin n → ℝ) :
    Fin n → ℝ := fun i => ∑ t, a t * W1 t i

/-- Back-substitution is exact: relaxing in pre-activation space and then
    substituting `z = W1 x'` is the same linear function of `x'` as the CROWN
    row applied directly. -/
theorem crownRow_dot {n h : ℕ} (a : Fin h → ℝ) (W1 : Fin h → Fin n → ℝ)
    (x' : EuclideanSpace ℝ (Fin n)) :
    (∑ t, a t * denseE W1 x' t) = ∑ i, crownRow a W1 i * x' i := by
  simp only [denseE_apply, crownRow, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun t _ => by ring

/-- **Concretization, once.** On the uniform box `x ∓ ε` a linear functional
    bottoms out at `⟨A, x₀⟩ − ε‖A‖₁`. This is `denseLo_uniform` read at the
    one-row matrix `A` — reused, not re-derived. ONE `ℓ1` norm per
    `(image, class)`, where IBP pays one per `(image, neuron)`. -/
theorem linf_lower_bound {n : ℕ} (A : Fin n → ℝ) (x : EuclideanSpace ℝ (Fin n)) (ε : ℝ)
    {x' : EuclideanSpace ℝ (Fin n)}
    (hbox : InBoxE (fun i => x i - ε) (fun i => x i + ε) x') :
    (∑ i, A i * x i) - ε * (∑ i, |A i|) ≤ ∑ i, A i * x' i := by
  have h := denseLo_le (fun _ : Fin 1 => A) hbox 0
  rwa [denseLo_uniform (fun _ : Fin 1 => A) x ε 0, denseE_apply] at h

-- ════════════════════════════════════════════════════════════════
-- § Assembly: per-neuron relaxations ⇒ a margin lower bound
-- ════════════════════════════════════════════════════════════════

/-- The two-layer net, evaluated one logit at a time — `denseE`/`reluE` peeled
    exactly once, so the inner `denseE W1 x'` stays folded (unfolding it would
    put a 784-term sum under every `max`). -/
theorem mlp2_apply {n h k : ℕ} (W1 : Fin h → Fin n → ℝ) (W2 : Fin k → Fin h → ℝ)
    (x' : EuclideanSpace ℝ (Fin n)) (c : Fin k) :
    (denseE W2 ∘ reluE ∘ denseE W1) x' c = ∑ t, W2 c t * max (denseE W1 x' t) 0 := by
  show denseE W2 (reluE (denseE W1 x')) c = _
  rw [denseE_apply]
  exact Finset.sum_congr rfl fun t _ => by rw [reluE_apply]

/-- **The margin bound.** Summing the per-neuron relaxations gives a linear
    lower bound on `f · y − f · j` in pre-activation space. `z` is abstract
    here: the bound is a statement about the relaxation, not about the net. -/
theorem crown_margin_ge {h k : ℕ} (W2 : Fin k → Fin h → ℝ) (y j : Fin k)
    (a cc l u z : Fin h → ℝ)
    (hz : ∀ t, l t ≤ z t ∧ z t ≤ u t)
    (hr : ∀ t, ReluLB (W2 y t - W2 j t) (l t) (u t) (a t) (cc t)) :
    (∑ t, a t * z t) + (∑ t, cc t)
      ≤ (∑ t, W2 y t * max (z t) 0) - (∑ t, W2 j t * max (z t) 0) := by
  rw [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  refine Finset.sum_le_sum fun t _ => ?_
  have := hr t (z t) (hz t).1 (hz t).2
  nlinarith [this]

-- ════════════════════════════════════════════════════════════════
-- § The capstone
-- ════════════════════════════════════════════════════════════════

/-- **CROWN-IBP `L∞` certificate for `dense ∘ relu ∘ dense`.**

    For each wrong class `j`, the generator supplies relaxation coefficients
    `a j` and constants `cc j` for the margin row `W2 y · − W2 j ·`, discharged
    against the IBP pre-activation box by the `reluLB_*` instances; the
    certificate fires when the ONE concretized linear bound is positive:

        ⟨A, x₀⟩ − ε‖A‖₁ + Σ cc  >  0,    A = crownRow (a j) W1.

    Note the box in `hrelax` is literally `denseLo W1 …` / `denseHi W1 …` — the
    interval box `IntervalBound.lean` proves — not a float recomputation. That
    is forced by the statement, which is the point: CROWN-IBP is sound only if
    the `[l,u]` you relax against are the ones that actually hold on the box the
    certificate quantifies over. -/
theorem crown2_certified_at_eps {n h k : ℕ}
    (W1 : Fin h → Fin n → ℝ) (W2 : Fin k → Fin h → ℝ)
    {x : EuclideanSpace ℝ (Fin n)} {ε : ℝ} {y : Fin k}
    (a cc : Fin k → Fin h → ℝ)
    (hrelax : ∀ j, j ≠ y → ∀ t,
      ReluLB (W2 y t - W2 j t)
        (denseLo W1 (fun i => x i - ε) (fun i => x i + ε) t)
        (denseHi W1 (fun i => x i - ε) (fun i => x i + ε) t)
        (a j t) (cc j t))
    (hcert : ∀ j, j ≠ y →
      0 < (∑ i, crownRow (a j) W1 i * x i)
            - ε * (∑ i, |crownRow (a j) W1 i|) + ∑ t, cc j t) :
    CertifiedAtLinf (denseE W2 ∘ reluE ∘ denseE W1) ε x y := by
  refine certified_of_marginPos fun j hj x' hbox => ?_
  -- the IBP pre-activation box, from the phase-0 per-layer bracket
  have hz : ∀ t, denseLo W1 (fun i => x i - ε) (fun i => x i + ε) t ≤ denseE W1 x' t ∧
      denseE W1 x' t ≤ denseHi W1 (fun i => x i - ε) (fun i => x i + ε) t :=
    denseE_boxSound W1 _ _ _ hbox
  -- per-neuron relaxations, summed
  have hmar := crown_margin_ge W2 y j (a j) (cc j) _ _ (denseE W1 x') hz (hrelax j hj)
  -- back-substitute, then concretize once
  rw [crownRow_dot] at hmar
  have hconc := linf_lower_bound (crownRow (a j) W1) x ε hbox
  rw [mlp2_apply, mlp2_apply]
  linarith [hcert j hj, hmar, hconc]

-- ════════════════════════════════════════════════════════════════
-- § The CROWN row as ONE kernel fact (`ListDot.lean` at matrix scale)
-- ════════════════════════════════════════════════════════════════

/-! `‖A‖₁ = Σᵢ |Σₜ aₜ·W1ₜᵢ|` does NOT decompose over `t` — the absolute value is
taken after the combination, which is exactly the point of CROWN. So it needs a
fact of its own, and the naive route (emit `A`'s 784 numerators per
`(image, class)`) makes the exhibit enormous (gotcha 2).

Instead the kernel *forms* `A` from the weight rows the corpus already commits:
a generator emits the 16 coefficient numerators, and `absSumZ (combZ …)` folds
the 16×784 combination and the absolute sum in one `decide +kernel`.

`⟨A, x₀⟩` needs no new fact at all — `crownRow_dot` turns it back into
`Σₜ aₜ·⟨W1ₜ, x₀⟩`, i.e. a 16-term rational sum over the committed `hpre` dots. -/

/-- Entrywise `c · row + acc`. -/
def scaleAddZ (c : ℤ) (row acc : List ℤ) : List ℤ :=
  List.zipWith (fun r a => c * r + a) row acc

/-- `Σₜ csₜ · rowsₜ`, entrywise: the CROWN row `A`, at integer scale. -/
def combZ (n : ℕ) : List ℤ → List (List ℤ) → List ℤ
  | [], _ => List.replicate n 0
  | _ :: _, [] => List.replicate n 0
  | c :: cs, r :: rs => scaleAddZ c r (combZ n cs rs)

theorem getD_replicate_zero (n i : ℕ) : (List.replicate n (0 : ℤ)).getD i 0 = 0 := by
  induction n generalizing i with
  | zero => simp
  | succ m ih =>
      rw [List.replicate_succ]
      cases i with
      | zero => simp
      | succ k => simpa using ih k

theorem length_combZ (n : ℕ) : ∀ (cs : List ℤ) (rows : List (List ℤ)),
    (∀ r ∈ rows, r.length = n) → (combZ n cs rows).length = n
  | [], _, _ => by simp [combZ]
  | _ :: _, [], _ => by simp [combZ]
  | c :: cs, r :: rs, hr => by
      have hrn : r.length = n := hr r (by simp)
      have hrec := length_combZ n cs rs (fun q hq => hr q (by simp [hq]))
      simp [combZ, scaleAddZ, hrn, hrec]

theorem getD_scaleAddZ (c : ℤ) : ∀ (row acc : List ℤ), row.length = acc.length → ∀ i : ℕ,
    (scaleAddZ c row acc).getD i 0 = c * row.getD i 0 + acc.getD i 0
  | [], [], _, _ => by simp [scaleAddZ]
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | _ :: _, _ :: _, _, 0 => by simp [scaleAddZ]
  | _ :: rs, _ :: as, h, i + 1 => by
      simp only [scaleAddZ, List.zipWith_cons_cons, List.getD_cons_succ]
      exact getD_scaleAddZ c rs as (by simpa using h) i

theorem getD_map_getD (rows : List (List ℤ)) (i : ℕ) : ∀ t : ℕ,
    (rows.map (fun r => r.getD i 0)).getD t 0 = (rows.getD t []).getD i 0
  | 0 => by cases rows <;> simp
  | t + 1 => by
      cases rows with
      | nil => simp
      | cons r rs => simpa using getD_map_getD rs i t

/-- The combined row, entry by entry, is a plain `dotZ` of the coefficients
    against the rows' `i`-th column. -/
theorem getD_combZ (n : ℕ) : ∀ (cs : List ℤ) (rows : List (List ℤ)),
    cs.length = rows.length → (∀ r ∈ rows, r.length = n) → ∀ i : ℕ,
    (combZ n cs rows).getD i 0 = dotZ cs (rows.map (fun r => r.getD i 0))
  | [], [], _, _, i => by
      rw [show combZ n ([] : List ℤ) ([] : List (List ℤ)) = List.replicate n 0 from rfl,
        getD_replicate_zero]
      simp [dotZ]
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | c :: cs, r :: rs, hl, hr, i => by
      have hrn : r.length = n := hr r (by simp)
      have hlen : r.length = (combZ n cs rs).length := by
        rw [hrn, length_combZ n cs rs (fun q hq => hr q (by simp [hq]))]
      rw [show combZ n (c :: cs) (r :: rs) = scaleAddZ c r (combZ n cs rs) from rfl,
        getD_scaleAddZ c r _ hlen i,
        getD_combZ n cs rs (by simpa using hl) (fun q hq => hr q (by simp [hq])) i]
      simp [dotZ]

/-- **The CROWN row, in one kernel-checkable object.** With coefficients
    `aₜ = csₜ / dc` and weights `W1ₜᵢ = rowsₜᵢ / dw`, the ℝ-level `crownRow`
    is the integer `combZ` over `dc·dw`. -/
theorem crownRow_comb {n h : ℕ} (cs : List ℤ) (rows : List (List ℤ))
    (W1 : Fin h → Fin n → ℝ) (dc dw : ℝ)
    (hW : ∀ (t : Fin h) (i : Fin n), W1 t i = (((rows.getD (t : ℕ) []).getD (i : ℕ) 0 : ℤ) : ℝ) / dw)
    (hcl : cs.length = h) (hrl : rows.length = h) (hrn : ∀ r ∈ rows, r.length = n)
    (i : Fin n) :
    crownRow (fun t => ((cs.getD (t : ℕ) 0 : ℤ) : ℝ) / dc) W1 i
      = (((combZ n cs rows).getD (i : ℕ) 0 : ℤ) : ℝ) / (dc * dw) := by
  have hmap : (rows.map (fun r => r.getD (i : ℕ) 0)).length = h := by simpa using hrl
  rw [getD_combZ n cs rows (by rw [hcl, hrl]) hrn (i : ℕ)]
  rw [← sum_getD_div hcl hmap rfl dc dw]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [hW t i, getD_map_getD rows (i : ℕ) (t : ℕ)]

/-- **`‖A‖₁` from one `absSumZ … := by decide +kernel` fact.** One `ℓ1` norm per
    `(image, class)`, where IBP pays one per `(image, neuron)` — and the 784
    entries of `A` are never emitted, only folded. -/
theorem crownRow_l1 {n h : ℕ} (cs : List ℤ) (rows : List (List ℤ))
    (W1 : Fin h → Fin n → ℝ) (dc dw : ℝ)
    (hW : ∀ (t : Fin h) (i : Fin n), W1 t i = (((rows.getD (t : ℕ) []).getD (i : ℕ) 0 : ℤ) : ℝ) / dw)
    (hcl : cs.length = h) (hrl : rows.length = h) (hrn : ∀ r ∈ rows, r.length = n)
    (hd : 0 ≤ dc * dw) {v : ℤ} (hv : absSumZ (combZ n cs rows) = v) :
    (∑ i : Fin n, |crownRow (fun t => ((cs.getD (t : ℕ) 0 : ℤ) : ℝ) / dc) W1 i|)
      = (v : ℝ) / (dc * dw) := by
  rw [Finset.sum_congr rfl fun i _ =>
    congrArg abs (crownRow_comb cs rows W1 dc dw hW hcl hrl hrn i)]
  exact sum_getD_abs_div (length_combZ n cs rows hrn) hv hd

end LipschitzCertDemo
end Proofs
