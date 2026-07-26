# CROWN-IBP: a tighter box on the seam that already exists

*(2026-07-26. Status: **phase 0 done**, rest design-only. Written as a pick-up
doc. Assumes the certificate corpus as of `4276787`: IBP L∞ tiers generated and
CI-verified, `certs-heavy.yml` green.)*

## 0. Why — the one number this is aimed at

The IBP L∞ scorecard, measured over the first 100 MNIST test images, against
its PGD-L∞ bracket:

| net | ε = 1/255 | 2/255 | 4/255 | 8/255 |
|---|---|---|---|---|
| **SF** (σ≤2 capped) certified | 92 | 88 | 69 | **24** |
| SF, PGD-robust (upper bracket) | 93 | 93 | 92 | **88** |
| **TF** (unconstrained) certified | 87 | 42 | 2 | **0** |
| TF, PGD-robust | 95 | 92 | 85 | 36 |

At ε = 1/255 the sandwich is `92 ≤ TRUE ≤ 93` — closed, nothing to win. At
ε = 8/255 it is **`24 ≤ TRUE ≤ 88`**. That column is the entire prize, and it is
exactly where interval propagation is worst: the box grows multiplicatively with
depth because every layer throws away the correlations between neurons.

This is a **capability** change, not hygiene — the axis
`planning/scorecard_trim.md` §7 says trimming explicitly does not move.

## 1. What CROWN-IBP is, and what it is not

* **IBP** (what we have): concretize to an interval after *every* layer. Cheap,
  linear in width, loses all correlation.
* **CROWN** (Zhang et al. 2018): never concretize in the middle. Carry a *linear
  function of the input* backward through the net, relaxing each ReLU by a
  linear upper/lower envelope, and concretize **once** at the end.
* **CROWN-IBP** (the target here): take the per-neuron pre-activation bounds
  `[l, u]` — which decide each ReLU's relaxation — from **IBP**, which is
  already implemented and proved in this repo. Do one CROWN backward pass for
  the output. Full CROWN would instead re-derive `[l,u]` by running itself on
  every prefix; that is strictly tighter and much more work.
* **Not α-CROWN.** The relaxation slope `α` is a fixed heuristic here. Optimizing
  it is a separate project and needs a different soundness story (any `α ∈ [0,1]`
  is sound, so optimization is a *tightness* question, not a correctness one —
  which is a nice property to lean on later).

### Why it is tighter, in one line

For a 2-layer net `f = denseE W2 ∘ reluE ∘ denseE W1` (no biases, as `mlpSF`/
`mlpTF`), IBP bounds each pre-activation with `⟨W1ᵢ, x₀⟩ ∓ ε‖W1ᵢ‖₁` and *then*
combines. CROWN forms the composite row `A = Σᵢ coefᵢ · W1ᵢ` **first** and takes
`ε‖A‖₁` **once**. Cancellation between the rows of `W1` survives in `A` and is
destroyed by the per-row `‖·‖₁`. That is the whole mechanism.

Concretely, certify the *margin directly* (as the LipSDP tier already does
pairwise): for each `j ≠ y` set `v = W2 y · − W2 j ·`, sign-split `v` against the
ReLU envelopes, back-substitute to a single `A`, and certify when

    ⟨A, x₀⟩ − ε‖A‖₁ + const  >  0

One ℓ1 norm per `(image, class)` instead of one per `(image, neuron)`.

## 2. The seam: what already accepts this, and what does not

**The conv tier is ready.** `Foundation/IntervalBoundConv.lean`:

```lean
def BoxSound3V (f : Tensor3 c h w → Vec k)
    (Flo Fhi : Tensor3 c h w → Tensor3 c h w → Vec k) : Prop :=
  ∀ lo hi u, InBox3 lo hi u → InBox (Flo lo hi) (Fhi lo hi) (f u)

theorem ibp3_certified_of_boxSound (hs : BoxSound3V f Flo Fhi)
    (hsep : ∀ j, j ≠ y → Fhi … j < Flo … y) : CertifiedAtLinf3 f ε x y
```

`BoxSound3V` is a pure *bracketing* predicate — it says nothing about how
`Flo`/`Fhi` are computed. Any CROWN-derived pair that satisfies it plugs in and
yields `CertifiedAtLinf3` with **no new certification layer**.

**The dense tier was not** (fixed by phase 0, §3). Its entry point used to
hardwire the interval transformers into its statement:

```lean
theorem ibp2_certified_at_eps (W1 : Fin h → Fin n → ℝ) (W2 : Fin k → Fin h → ℝ)
    (hcmp : ∀ j, j ≠ y → denseHi W2 (reluLo (denseLo W1 …)) (reluHi (denseHi W1 …)) j
                       < denseLo W2 … y)
    (δ …) (hδ : ∀ i, |δ i| ≤ ε) : …
```

so CROWN could not reuse it. `certified_of_boxSound` / `BoxSoundE` now sit
underneath it and take an arbitrary bracket.

**Already-built machinery to reuse, not rebuild:**

| need | already in the repo |
|---|---|
| `⟨w, x₀⟩ ∓ ε‖w‖₁` collapse on a uniform box | `denseLo_uniform` / `denseHi_uniform` |
| exact ℓ1 norm of an ℤ-row, in-kernel | `absSumZ` + `sum_getD_abs_div` (`ListDot.lean`) |
| exact 784-term dot, in-kernel | `dotZ` + `sum_getD_div` |
| per-image pre-activations at real weights | the committed `hpre{SF,TF}<i>` facts |
| per-row ℓ1 norms of `W1` | `absr{SF,TF}` / `absrow{SF,TF}` in the IBP scorecards |
| certification from a bracket (conv) | `ibp3_certified_of_boxSound` |
| certification from a bracket (dense) | `certified_of_boxSound` (phase 0) |

## 3. Phase 0 — abstract the dense seam ✅ DONE

`IntervalBound.lean` now carries the seam, in the shape `IntervalBoundConv.lean`
already uses (predicate named with the file's own `…E` convention, since it
lives in `EuclideanSpace` while the conv engine lives in `Vec`):

```lean
def InBoxE (lo hi : Fin n → ℝ) (u : EuclideanSpace ℝ (Fin n)) : Prop
def BoxSoundE (f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m))
    (Flo Fhi : (Fin n → ℝ) → (Fin n → ℝ) → (Fin m → ℝ)) : Prop
theorem BoxSoundE.comp    -- depth by composition
theorem denseE_boxSound   -- denseLo_le + le_denseHi, packaged
theorem reluE_boxSound    -- relu_box, packaged
theorem certified_of_boxSound (hs : BoxSoundE f Flo Fhi)
    (hsep : ∀ j, j ≠ y → Fhi … j < Flo … y) : CertifiedAtLinf f ε x y
theorem mlp2_boxSound     -- the interval bracket for dense ∘ relu ∘ dense
```

`ibp2_certified_at_eps` is now a one-line corollary
(`certified_of_boxSound (mlp2_boxSound W1 W2) hcmp δ hδ`) with its **statement
byte-identical** — so the generated corpus and `scripts/…_ibp.py` are untouched,
no regeneration needed, and the `scorecard_trim.md` §2.2 diff is empty by
construction. New decls audited in `tests/AuditAxioms.lean`; all 3-axiom clean.

CROWN is therefore *additive*: a second sound `BoxSoundE` witness on the same
seam, not a parallel stack.

## 4. Phase 1 — the engine (`Foundation/CrownBound.lean`) ✅ DONE

Built, 3-axiom clean, engine-only (no generated data), in the `Certs` lib.

**The margin seam.** `certified_of_marginPos` — a strictly positive margin
`f · y − f · j` at every point of the box certifies. This replaces the planned
"discharge `BoxSoundV`, close with `certified_of_boxSound`" route: bounding the
two logits independently and then separating them discards the correlation
between them, which §5.5 measures as a large share of the available tightening.
Same seam idea, one rung lower — it never inspects how the margin was bounded.

**The ReLU envelopes.** Split into the two directions, because they have very
different characters:

```lean
theorem relu_lower_envelope (h0 : 0 ≤ α) (h1 : α ≤ 1) : α * z ≤ max z 0
theorem relu_upper_envelope (hl : l ≤ 0) (hlz : l ≤ z) (hzu : z ≤ u)
    (hs0 : 0 ≤ s) (hs : u ≤ s * (u - l)) : max z 0 ≤ s * (z - l)
```

* the lower envelope needs **no reference to `[l,u]` at all** — any `α ∈ [0,1]`
  works for any `z`, so α is a pure tightness knob and a rounded α needs no
  re-verification (the α-CROWN soundness story, for free);
* the upper envelope takes the chord condition **multiplicatively**,
  `u ≤ s*(u-l)` and not `s = u/(u-l)`. That is the whole gotcha-1 lever: `s` may
  be the chord slope ROUNDED UP to a `/2^k` grid, and a generator discharges the
  side condition by `norm_num`. No division, no denominator growth.

Proof of the upper envelope: `g z = s·(z−l) − relu z` is affine on `[0,u]` with
`g 0 = −s·l ≥ 0` and `g u = s·(u−l) − u ≥ 0`, so it is nonneg between; below `0`
it is trivial. `nlinarith` closes it from those two products.

**Per-neuron relaxation.** `ReluLB v l u a c := ∀ z, l ≤ z → z ≤ u →
a * z + c ≤ v * max z 0`, with four instances — `reluLB_dead` (`u ≤ 0`),
`reluLB_active` (`0 ≤ l`, EXACT, no relaxation), `reluLB_unstable_pos`
(`0 ≤ v`, lower envelope), `reluLB_unstable_neg` (`v ≤ 0`, upper envelope, the
only branch carrying a constant). The split is on the sign of the *margin
coefficient* `v`, not on the neuron alone — that is what decides which envelope
lower-bounds the contribution.

**Back-substitution and concretization.** `crownRow a W1 i = ∑ t, a t * W1 t i`
with `crownRow_dot` proving the substitution exact, then `linf_lower_bound` =
`denseLo_uniform` read at the one-row matrix `A` (reused, not re-derived). One
`ℓ1` norm per `(image, class)` where IBP pays one per `(image, neuron)`.

**Capstone.** `crown2_certified_at_eps` takes `a`, `cc`, the per-neuron `ReluLB`
facts **stated against `denseLo W1 …`/`denseHi W1 …` literally** (gotcha 4 is
enforced by the statement — you cannot relax against a float recomputation), and
the one concretized comparison per wrong class:

    ⟨A, x₀⟩ − ε‖A‖₁ + Σ cc  >  0

*Not built:* a general multi-layer `LinSound`. For a two-layer net `crownRow`
IS the matrix-level back-substitution, and there is no consumer for the general
version — the conv tier stays on IBP (gotcha 3) and the trained dense nets are
two-layer. Add it when something needs depth.

## 5. Phase 2 — the generated instance ✅ DONE

`scripts/crown_ibp_scorecard.py` →
`LipschitzCertScorecardCrown{,Uncon}.lean` (2,967 + 5,333 lines), in
`CertsHeavy`. Same nets, same first-100 subset, same ε grid — a new COLUMN,
not a new experiment:

| net | ε=1/255 | 2/255 | 4/255 | 8/255 |
|---|---|---|---|---|
| SF IBP (was) | 92 | 88 | 69 | 24 |
| **SF CROWN** | **93** | **93** | **92** | **81** |
| SF PGD (ceiling) | 93 | 93 | 92 | 88 |
| TF IBP (was) | 87 | 42 | 2 | 0 |
| **TF CROWN** | **94** | **92** | **76** | **15** |
| TF PGD (ceiling) | 95 | 92 | 85 | 36 |

**Success criterion met.** 4/255: 69 → 92 (= PGD, closed). 8/255: 24 → 81
against a ceiling of 88. The exact-rational counts reproduce the float probe of
§5.5 digit for digit, which is the cross-check that matters — the probe was a
prediction, this is the certificate.

Two things the plan did not anticipate, both about keeping the instance
affordable:

* **`‖A‖₁` needed new engine support.** `Σᵢ |Σₜ aₜW1ₜᵢ|` does not decompose over
  `t` — the absolute value is taken *after* the combination, which is the whole
  point of CROWN — so it needs a fact of its own, and emitting `A`'s 784
  numerators per `(image, class)` would have made the exhibit enormous (gotcha 2
  biting exactly as predicted). Fix: `combZ` in `CrownBound.lean` — the kernel
  *forms* `A` from the committed `w1z` rows, so a generator emits 16 coefficient
  numerators and one `absSumZ (combZ …) := by decide +kernel`. `⟨A, x₀⟩` needs
  no new fact at all: `crownRow_dot` turns it back into `Σₜ aₜ·⟨W1ₜ, x₀⟩` over
  the committed `hpre` dots.
* **`![…] 7` does not reduce under `norm_num`** (only `cons_val_zero/one/two`
  exist), and `rw` cannot match a numeral index against `fin_cases`'s `⟨j, _⟩`.
  So all per-class data lives in NAMED defs and the `![…]` form is assembled
  only at the end, where `exact` bridges it by defeq. Worth remembering — it
  will bite any generator that indexes a committed weight matrix at a literal.
* **Being stronger than IBP has a corpus cost.** CROWN certifies images IBP
  never did, and the corpus only ever emitted per-image `hpre` dot data for
  images some earlier tier needed — `hpreTF1` exists *only* in the DISABLED
  LipSDP file. Exhibiting one of those would mean emitting new 784-term dot
  data, i.e. not reuse. So the images carrying THEOREMS are drawn from what is
  already committed and in the build. Measured counts are unaffected (they run
  over all 100 either way); the visible effect is that the TF ε=8/255 aggregate
  lists 3 witnesses rather than 8, since only 3 of its 15 certified images have
  committed data. If a future tier wants the full 8 there, the fix is to widen
  the base generator's per-image emission, not to change this one.

`N_EMIT` contract honoured: counts MEASURED over all 100 in exact rationals,
first 8 certifying images per radius carry theorems, header states which is
which. Each emitted image is proved ONCE, at the largest radius it is emitted
at, and carried down the grid by `CertifiedAtLinf.mono` — 23 proved images and
207 kernel facts instead of 4× that. Not pooled, so no `REDUCED CERTIFICATE
MODEL` disclaimer applies (these are the full 784-dim nets).

## 5.5. Measured payoff — `scripts/crown_ibp_probe.py` ✅ (gotcha 5, done)

Float simulation at the committed `/256` weights, first 100 test images. The
`ibp_box` column reproduces the certified corpus exactly (92/88/69/24 and
87/42/2/0), which is what validates the rest of the table.

| net | ε | unstable/16 | IBP box (now) | **IBP margin** | **CROWN** | PGD |
|---|---|---|---|---|---|---|
| SF | 1/255 | 0.69 | 92 | 93 | **93** | 93 |
| SF | 2/255 | 1.46 | 88 | 92 | **93** | 93 |
| SF | 4/255 | 2.79 | 69 | 81 | **92** | 92 |
| SF | 8/255 | 5.23 | 24 | 45 | **81** | 88 |
| TF | 1/255 | 1.24 | 87 | 91 | **94** | 95 |
| TF | 2/255 | 2.34 | 42 | 64 | **92** | 92 |
| TF | 4/255 | 4.42 | 2 | 13 | **76** | 85 |
| TF | 8/255 | 7.72 | 0 | 0 | **15** | 36 |

* **Gotcha 5 answers "yes".** 33 % (SF) / 48 % (TF) of neurons are unstable at
  8/255 — neither degenerate regime. Phase 1 is worth doing.
* **The prize column moves 24 → 81** against a PGD ceiling of 88, and the
  sandwich *closes exactly* at SF 4/255 (92 = 92), SF 2/255, SF 1/255 and
  TF 2/255 (92 = 92). CROWN gains 57 SF images and loses none.
* **Float risk is nil.** Tightest certifying margin 3.4e-2 against a dynamic
  range of ~53 (ratio 6e-4); zero `(image, class)` pairs within 1e-9 of the
  decision boundary. Exact rationals will reproduce these counts.
* **Gotcha 1 is defused: `k = 8` suffices.** Rounding the upper-envelope slope
  up to a `/2^k` grid costs nothing from k=8 on (SF 8/255: k=2 → 80, k=4 → 81,
  k≥4 → 81; TF 4/255: k=2 → 63, k=4 → 72, k=6 → 75, k≥8 → 76). So relaxation
  coefficients live at the SAME `/256` denominator scale as the weights, and
  `A = Σᵢ coefᵢ·W1ᵢ` lands at `/2^16` — nowhere near the LipSDP 230-digit
  regime. **Emit the `/2^k` grid from the start; do not prototype unrounded.**

### Margin-direct IBP ("phase 0.5") — considered and DROPPED as a tier

The `IBP margin` column uses no CROWN at all: same interval hidden box, but
sign-split the margin row `v = W2 y · − W2 j ·` instead of separating the two
logit boxes. It looked like a cheap independent rung (24 → 45 at SF 8/255) with
none of CROWN's risk.

**It is strictly dominated by CROWN, so it is not worth a tier.** Setting
`α := 0` — the *weakest* legal lower envelope — already beats it everywhere:

| | SF 1 | SF 2 | SF 4 | SF 8 | TF 1 | TF 2 | TF 4 | TF 8 |
|---|---|---|---|---|---|---|---|---|
| IBP margin | 93 | 92 | 81 | 45 | 91 | 64 | 13 | 0 |
| CROWN α=0 | 93 | 93 | 92 | **80** | 94 | 91 | 70 | 12 |
| CROWN heur | 93 | 93 | 92 | **81** | 94 | 92 | 76 | 15 |

That is not a coincidence of these weights. Per neuron: an unstable neuron with
`v ≥ 0` contributes `0` under both (IBP uses `relu l = 0`; CROWN α=0 uses
coefficient 0); an unstable neuron with `v < 0` contributes `v·u` under IBP,
which is exactly CROWN's linear envelope evaluated at its worst point `z = u`;
and a *stable-active* neuron keeps its coefficient linear in `x` under CROWN, so
its cancellation against the other rows survives into `A` instead of being
normed away per-row. CROWN α=0 therefore dominates term by term, and the α
heuristic is pure upside on top (+1 SF, +6 TF at 4/255).

Building it anyway would mean maintaining a second generated corpus whose every
number is worse. **Skip it.** The one piece worth keeping from the idea — the
margin seam itself — is already in `CrownBound.lean` as `certified_of_marginPos`,
generic and reusable if a margin-direct tier is ever wanted.

## 6. Gotchas

1. **Rational coefficient blow-up.** `u/(u−l)` has a denominator that grows with
   the product of the layer-1 denominators; `A = Σᵢ coefᵢ W1ᵢ` then carries it
   into every entry. Left alone this reproduces the LipSDP tier's ~230-digit
   problem — the thing that costs 16 GB per goal and keeps those modules out of
   CI. **Round the relaxation coefficients to a `/2^k` grid and round in the
   SOUND direction**: the lower envelope's `α` may be any value in `[0,1]`, so
   round it however; the upper envelope's slope must be rounded **up** and its
   intercept adjusted to stay above the chord. Verify the rounded envelope
   exactly before emitting, the way `rational_cert` verifies its LDLᵀ.
   **✅ MEASURED (§5.5): `k = 8` costs zero images, so the coefficients sit at
   the same `/256` scale as the weights and `A` lands at `/2^16`. This gotcha
   is defused — but only if the grid is emitted from the start.**
2. **Exhibit size goes up, not down.** IBP emits two intervals per neuron;
   CROWN emits a coefficient row per `(image, class)`. The `N_EMIT` cap matters
   *more* here, not less.
3. **Max-pool.** Fine under IBP (monotone — pool the endpoints). Genuinely
   awkward under CROWN: it is not an elementwise nonlinearity. **Do the dense
   tier first and leave `IbpConvScorecard` on IBP.** Extending CROWN to the conv
   net is a separate decision.
4. **The `[l,u]` you relax against must be the ones you proved.** CROWN-IBP's
   soundness depends on `l ≤ z ≤ u` holding for the *same* box the certificate
   quantifies over. Take them from the IBP layer-1 collapse
   (`denseLo_uniform`), not from a float recomputation in the generator.
5. **Unstable-neuron count is the whole story.** If almost every neuron is
   stable at ε = 8/255 then CROWN ≈ IBP and there is nothing to win; if almost
   none are, the relaxation is loose. **Measure the unstable fraction per radius
   in Python before writing any Lean** — it predicts the payoff for free.
   **✅ DONE (§5.5): 33 % (SF) / 48 % (TF) unstable at 8/255 — the productive
   middle. The probe went further and simulated CROWN outright: 24 → 81.**

## 7. What this does not address

* The **LipSDP 16 GB problem** — that is `linarith` over a 256-monomial slack
  with huge LDLᵀ fractions, a different bottleneck. See
  `planning/certs_heavy_psd_memory.md`; still needs the diagonally-dominant
  witnesses.
* The **per-push weight-side `simp` cost** (`W*_abs_le`, `G*_eq`/`H*_eq`) — that
  wants kernel bounds over the ℤ-list weights, the `ListDot.lean` trick.
  Unrelated to this work. `scorecard_trim.md` §5.
* **Resolution.** CROWN tightens the bound at the current 8×8 / 784-dim scales;
  it does not lift the conv tier's resolution ceiling.

## 8. Suggested order

```
phase 0    abstract the dense seam, re-derive ibp2_certified_at_eps  ✅ DONE
gotcha 5   measure unstable fraction + simulate CROWN in Python      ✅ DONE (§5.5)
phase 0.5  margin-direct IBP                                         ✗ DROPPED (dominated, §5.5)
phase 1    CrownBound.lean: relaxation + substitution + concretize   ✅ DONE (§4)
phase 2    generated instance, new column in the IBP table          ✅ DONE (§5)
```

**The arc is complete.** ε = 8/255 on the capped net went `24 ≤ TRUE ≤ 88` →
`81 ≤ TRUE ≤ 88`, and 4/255 closed outright at 92. What is left is no longer
this plan:

* **α-CROWN** — optimize the lower-envelope slope instead of the `u > −l`
  heuristic. Worth +1 (SF) / +6 (TF) images at 4/255 by the probe's α=0-vs-heuristic
  spread, so the headroom is real but small. Soundness needs no new story:
  `relu_lower_envelope` already accepts any `α ∈ [0,1]`, so this is pure
  generator work.
* **Full CROWN buys NOTHING here — do not attempt it.** §1 calls full CROWN
  "strictly tighter" than CROWN-IBP, which is true in general and vacuous at two
  layers. The only pre-activation bounds this net needs are layer 1's, and over
  an L∞ ball those are already *exact*: `ε‖w‖₁` is the exact support function of
  a linear functional on that ball, so `denseLo_uniform` is tight, not a
  relaxation. CROWN-IBP ≡ full CROWN for `dense ∘ relu ∘ dense`. The remaining
  8/255 gap (81 vs 88 SF, 15 vs 36 TF) is therefore entirely the **ReLU
  relaxation gap on unstable neurons**, not loose `[l,u]`.
* **What would actually close it: branch-and-bound over unstable neurons.** Fix
  the sign of each unstable ReLU and the net becomes exactly linear on that
  cell, so the bound is exact and the union over sign patterns is complete. §5.5
  measures 5.2 (SF) / 7.7 (TF) unstable neurons on average at 8/255, i.e. tens
  to hundreds of cells — plausible, but `2^k` certificates per image, so it
  needs partial branching on the highest-impact neurons and a new exhaustiveness
  argument in the engine. That is a NEW arc, not a continuation of this one.
* **Conv.** Still blocked by max-pool not being an elementwise nonlinearity
  (gotcha 3); `IbpConvScorecard` stays on IBP.
