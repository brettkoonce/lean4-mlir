import LeanMlir.Proofs.Codegen.AdamStep

/-! # Global-norm gradient clipping over ℝ — the ViT / ConvNeXt recipe's last v1.4 piece

The ℝ reference for `planning/grad_clip.md` (`recipe_gaps.md` v1.4b). Coordinatewise over `Vec`
where it can be, mirroring the emitted StableHLO op-for-op so the faithfulness theorems in
`StableHLO.lean` are structural matches (`rfl`), exactly as `AdamStep` is for the AdamW triple and
`RmsPropStep` for RMSProp's buffer.

**The reference** (`jax/Jax/Codegen.lean:2262`, emitted verbatim into every trainer whose config
sets `gradClipNorm`, and placed AFTER `value_and_grad` and BEFORE weight decay + the optimizer):

```python
gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, CLIP / (gn + 1e-6)), grads)
```

⚠⚠ **THE NORM IS GLOBAL — ONE SCALAR ACROSS EVERY PARAMETER, NOT ONE PER PARAMETER.** That is the
entire semantic content of the feature, and it is the thing a per-parameter check cannot see: a
per-parameter clip compiles, renders, trains and descends, and differs only in that the factor stops
being shared. `clipFactor_shared` below is the statement the numeric gate drives, and it is why that
gate measures the ratio's CONSTANCY across parameters rather than the presence of scaling
(`planning/grad_clip.md` §7 — `wdx-tie`'s *gate the partition, not the count*, one feature over).

**Who uses it**: ViT 1.0 (`jax/MainVitImagenet.lean:45`, *"DeiT default; the unlock for the 5e-4
LR"*) and ConvNeXt 1.0 (`jax/MainConvNeXtImagenet.lean:74`). **EfficientNet sets it to 0.0
deliberately** — its own comment says the TF-RMSProp fix (ε-inside-sqrt + ms-init 1.0) removed the
blow-up it was compensating for. R34 and mnv2 do not use it. Do not add it to any of the three.

**What is new here and what is not.** Nothing in the *shape* of this is new — the sum-to-rank-0
reduce is `lnBetaGrad`'s, and broadcasting a rank-0 scalar to a parameter shape and multiplying is
emitted 200× per ViT render already, inside `adamWParamF`. What the kit lacked is a way to scale a
tensor by a **runtime** scalar (`scaleF`/`scaleB` look like they take one and bake a literal), and a
rank-0 arithmetic tail. Four small ops, all in the `ds : List Nat` parameter-shape family.

**Claim ceiling.** Like `AdamStep`, the verified target is **faithfulness** (the rendered clip
denotes these functions) plus **well-definedness** (`clipDenom_pos`) — *not* that clipping improves
anything. Say "the clipped render is certified", never "clipping is proven to help". -/

namespace Proofs

variable {n : Nat}

/-- **The single ℝ inside a rank-0 render slot.** `SHlo 1` is how this kit spells a rank-0
    `tensor<f32>` (`lnBetaGrad`'s established reading), so its denotation is a `Vec 1` carrying one
    number, and the clip ops need to read it out.

    ⚠ **It exists so that `StableHLO.den` never APPLIES a recursive `den` call to an index.** Every
    other arm of that ~200-case dependent match passes `den e` along whole; writing `den e 0` in the
    new arms instead made nine unrelated `simp only [… den …]` proofs elsewhere in the file die with
    a `whnf` timeout that **4× the heartbeat budget did not fix** — the cost is in unfolding the
    match, not in arithmetic, so raising the limit is not the remedy. Passing `Vec 1 → ℝ` as an
    ordinary function keeps the new arms the same shape as the other 200. -/
def scalarOf (v : Vec 1) : ℝ := v ⟨0, Nat.zero_lt_one⟩

-- ════════════════════════════════════════════════════════════════
-- § The two halves: the per-parameter contribution, and the shared factor
-- ════════════════════════════════════════════════════════════════

/-- **One parameter's contribution to the global squared norm**: `∑ᵢ (gᵢ)²`.

    This is the reference's `jnp.sum(g * g)` for a single tree leaf. It is deliberately the SQUARED
    norm and deliberately per-leaf: the reference sums these across every leaf and takes ONE square
    root at the end, so a per-leaf `Real.sqrt` here would be a different function (and the usual
    `√(a+b) ≠ √a + √b`). `noncomputable` for `bn_grad_beta`'s reason — `Vec` is `Fin n → ℝ`. -/
noncomputable def gradSumSq (g : Vec n) : ℝ := ∑ i : Fin n, (g i) ^ 2

/-- **The clip factor**, `min 1 (c / (√s + ε))`, where `s` is the summed squared norm across ALL
    parameters and `ε` is the reference's `1e-6`.

    Takes `s` rather than `√s` so it mirrors the emit exactly: the graph carries the sum through the
    scalar fold and roots it once, here. -/
noncomputable def clipFactor (c ε s : ℝ) : ℝ := min 1 (c / (Real.sqrt s + ε))

/-- **The clipped gradient**: `g` scaled by a factor it does not itself determine.

    ⚠ The factor is a PARAMETER of this definition, not something computed from `g`, and that is the
    global-vs-local distinction made structural — this function cannot express a per-parameter clip
    because it never sees enough to compute one. -/
def clipScale (fac : ℝ) (g : Vec n) : Vec n := fun i => fac * g i

-- ════════════════════════════════════════════════════════════════
-- § Well-definedness
-- ════════════════════════════════════════════════════════════════

/-- The divisor `√s + ε` is strictly positive whenever `ε > 0` — `Real.sqrt` is unconditionally
    nonnegative, including at the negative arguments `Real.sqrt` maps to 0, so this needs no
    hypothesis on `s`. The `AdamStep.adam_denom_pos` argument verbatim. This is what makes the
    reference's `+ 1e-6` load-bearing rather than cosmetic: at `ε = 0` and a zero gradient the
    factor is `0/0`. -/
theorem clipDenom_pos (ε s : ℝ) (hε : 0 < ε) : 0 < Real.sqrt s + ε := by
  have hs : 0 ≤ Real.sqrt s := Real.sqrt_nonneg _
  linarith

/-- The factor never exceeds 1 — clipping only ever shrinks. Immediate from `min`, and stated
    because it is the half of the specification a "scale by `c/‖g‖`" misreading would drop: without
    the `min`, a SMALL gradient gets AMPLIFIED, which trains and descends and is not the recipe. -/
theorem clipFactor_le_one (c ε s : ℝ) : clipFactor c ε s ≤ 1 := min_le_left _ _

/-- The factor is nonnegative for a nonnegative clip threshold. -/
theorem clipFactor_nonneg (c ε s : ℝ) (hc : 0 ≤ c) (hε : 0 < ε) : 0 ≤ clipFactor c ε s := by
  have hd : 0 < Real.sqrt s + ε := clipDenom_pos ε s hε
  exact le_min zero_le_one (div_nonneg hc (le_of_lt hd))

-- ════════════════════════════════════════════════════════════════
-- § The known answers the numeric gates drive
-- ════════════════════════════════════════════════════════════════

/-- ▶ **BELOW THE THRESHOLD THE CLIP IS THE EXACT IDENTITY.** When `√s + ε ≤ c` the factor is
    `1` — not "approximately 1", *the literal constant 1* — so the clipped render must return its
    input unchanged.

    ⚠ This is what makes `planning/grad_clip.md`'s gate 3 a **bit-exactness** claim rather than a
    tolerance: `x * 1.0` is exact in IEEE-754 binary32, so a clip-on render at a large `c` must
    agree with the clip-off render on every byte. `dropPath_ones_id` licensed the stochastic-depth
    gate the same way, and the same warning applies here twice over — an identity gate CANNOT see
    where the intervention is applied, so gate 3 alone is blind to a per-parameter clip and to a
    misplaced clip site. It has to be run alongside gate 4, in the clipping regime. -/
theorem clipFactor_eq_one_below (c ε s : ℝ) (h : Real.sqrt s + ε ≤ c) (hε : 0 < ε) :
    clipFactor c ε s = 1 := by
  have hd : 0 < Real.sqrt s + ε := clipDenom_pos ε s hε
  exact min_eq_left ((one_le_div hd).mpr h)

/-- `clipScale 1 g = g`, the `Vec`-level reading of the line above. -/
@[simp] theorem clipScale_one (g : Vec n) : clipScale 1 g = g := by
  funext i; simp [clipScale]

/-- `clipScale 0 g = 0` — the zero-factor control, and the reason a clip site on the wrong side of
    an update is detectable at all. -/
@[simp] theorem clipScale_zero (g : Vec n) : clipScale (0 : ℝ) g = fun _ => 0 := by
  funext i; simp [clipScale]

/-- ▶ **THE FACTOR IS SHARED — the theorem the numeric gate exists to check.**

    Two parameters clipped by the same global factor have `g'ᵢ/gᵢ` equal, coordinate for coordinate,
    across BOTH of them. Stated as the cross-multiplied form so it needs no nonzero hypothesis and
    no division: `g'₁ᵢ · g₂ⱼ = g'₂ⱼ · g₁ᵢ`.

    ⚠ **A per-parameter clip satisfies every other theorem in this file.** It scales, it never
    amplifies, it is the identity below the threshold — it differs from the reference only here. So
    this is the load-bearing statement, and the harness's job is to drive it on the real render
    across all 200 (ViT) / 180 (ConvNeXt) parameters rather than to check that any single parameter
    got smaller. -/
theorem clipFactor_shared {m : Nat} (fac : ℝ) (g₁ : Vec n) (g₂ : Vec m) (i : Fin n) (j : Fin m) :
    (clipScale fac g₁) i * g₂ j = (clipScale fac g₂) j * g₁ i := by
  simp only [clipScale]; ring

/-- The composite the render computes for one parameter, folded into a single statement:
    `θ`'s gradient is scaled by the factor derived from the summed squared norm of ALL gradients.
    `sTotal` arrives as data precisely because it is not derivable from `g` — see `clipScale`. -/
noncomputable def clipGrad (c ε sTotal : ℝ) (g : Vec n) : Vec n :=
  clipScale (clipFactor c ε sTotal) g

/-- The reference's own composition, spelled out: `g * min(1, CLIP / (gn + 1e-6))` where
    `gn = sqrt(sum of the per-leaf sums of squares)`. Holds by `rfl`; it exists so the renderer's
    fold has a name to be tied to, and so a reader can check the transcription against the Python
    without unfolding three definitions. -/
theorem clipGrad_eq (c ε sTotal : ℝ) (g : Vec n) :
    clipGrad c ε sTotal g = fun i => min 1 (c / (Real.sqrt sTotal + ε)) * g i := rfl

/-- Below the threshold, `clipGrad` is the identity on the whole vector. The `Vec`-level form of
    `clipFactor_eq_one_below`, which is the shape the whole-net gate quotes. -/
theorem clipGrad_id_below (c ε sTotal : ℝ) (g : Vec n)
    (h : Real.sqrt sTotal + ε ≤ c) (hε : 0 < ε) : clipGrad c ε sTotal g = g := by
  simp [clipGrad, clipFactor_eq_one_below c ε sTotal h hε]

end Proofs
