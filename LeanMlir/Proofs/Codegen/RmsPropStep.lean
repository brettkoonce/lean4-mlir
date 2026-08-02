import LeanMlir.Proofs.Codegen.AdamStep
import LeanMlir.Proofs.Codegen.SgdMomentumStep

/-! # RMSProp with momentum over ℝ — the optimizer MobileNetV2 and EfficientNet actually use

The ℝ reference for the `mnv2in`/`enetin` ImageNet train steps (`planning/recipe_gaps.md` v1.2).
Coordinatewise over `Vec`, mirroring the emitted StableHLO op-for-op so the faithfulness theorem in
`StableHLO.lean` is a structural match (`rfl`), exactly as `AdamStep` is for the AdamW triple and
`SgdMomentumStep` for the SGD/Nesterov pair.

**⚠ THIS IS TENSORFLOW'S RMSPROP, NOT THE TEXTBOOK ONE — and that is the whole point of the file.**
The JAX reference (`jax/Jax/Codegen.lean`, the `.rmsprop` branch) says so in its own comment: ε goes
**inside** the square root and the running mean-square **initialises to 1.0**. `timm` ships
`RMSpropTF` for exactly this reason. Reaching for the textbook spelling — `g / (√s + ε)` — would
compile, render, train, descend, and be **a different optimizer than the one this exists to match**;
`rmsBufNext_eps_placement_at_zero` below is the exact difference, and it is what the numeric tie's
negative control drives. This is §2k's Nesterov-vs-heavy-ball trap in a second place, and the
transferable rule is the same one: **check which variant the reference uses before rendering one.**

**What is new here and what is not.** Three of the four steps of the update are already-certified ops
and this file only gives them their RMSProp reading:

| step | reference | verified path |
|---|---|---|
| `g ← g + wd·θ` (COUPLED L2) | `grads = g + WD * p` | `momVNext (μ := wd) (v := θ)` — `momVNext_as_coupled_l2` |
| `s' = ρ·s + (1−ρ)·g²` | `RHO * s + (1-RHO) * g * g` | `adamVNext (β₂ := ρ)` — `rmsSqNext_eq_adamVNext`, by `rfl` |
| `b' = μ·b + g/√(s'+ε)` | `MOMENTUM * b + g / sqrt(s + EPS)` | **`rmsBufNext` — the one genuinely new op** |
| `θ' = θ − lr·b'` | `p - lr * b` | `sgdParam` |

So the packed `[θ|m|v]` protocol is reused verbatim with **`m` carrying the momentum buffer and `v`
the running mean-square** — the same slot-reinterpretation `SgdMomentumStep` does for the velocity,
and the reason no driver or signature change is needed.

**Claim ceiling.** Like `AdamStep`, the verified target is **faithfulness** (the rendered update
denotes these functions) plus **well-definedness** (`rms_denom_pos`) — *not* a loss-decrease bound.
Say "the RMSProp render is certified", never "RMSProp is proven to descend". -/

namespace Proofs

variable {n : Nat}

/-- **Running mean-square of the gradient**: `s' = ρ·s + (1−ρ)·g²`.

    `ρ` is the reference's `rmspropDecay` (0.9 on both nets). Note this is *definitionally*
    `adamVNext` at `β₂ := ρ` — see `rmsSqNext_eq_adamVNext`, which is what licenses the render
    reusing the existing `adamVNextF` op rather than adding a second emitter for the same
    arithmetic. Defined under its own name anyway so the RMSProp chain reads as RMSProp. -/
def rmsSqNext (ρ : ℝ) (sq g : Vec n) : Vec n :=
  fun i => ρ * sq i + (1 - ρ) * (g i) ^ 2

/-- **The mean-square update IS Adam's second moment.** Stated, not assumed: the render emits
    `adamVNextF` for this slot, and this is the theorem that says doing so computes RMSProp's `s'`.
    Holds by `rfl` — same arithmetic, different reading of the two hyperparameter slots, exactly as
    §2k's heavy-ball reuses `momVNext` at `(μ := wd, v := θ)`. -/
theorem rmsSqNext_eq_adamVNext (ρ : ℝ) (sq g : Vec n) :
    rmsSqNext ρ sq g = adamVNext ρ sq g := rfl

/-- **Coupled L2 as a momentum update.** `momVNext wd θ g = g + wd·θ` — the reference's
    `grads = jax.tree.map(lambda g, p: g + WD * p, grads, params)`, which is COUPLED (the decay
    flows through the accumulator) and not AdamW's decoupled form. Same reuse §2k found for
    heavy-ball; restated here because RMSProp's `wd` is coupled for the same reason and a reader
    checking this chain should not have to rediscover it. -/
theorem momVNext_as_coupled_l2 (wd : ℝ) (θ g : Vec n) :
    momVNext wd θ g = fun i => g i + wd * θ i := by
  funext i; simp only [momVNext]; ring

/-- **Momentum buffer on the normalised gradient — TENSORFLOW's placement**:
    `b' = μ·b + g / √(s' + ε)`.

    ⚠ **ε is INSIDE the square root.** The textbook form is `g / (√s' + ε)`; see
    `rmsBufNextVanilla` and `rmsBufNext_eps_placement_at_zero` for the exact difference and why it
    is not cosmetic. -/
noncomputable def rmsBufNext (ρ μ ε : ℝ) (sq buf g : Vec n) : Vec n :=
  fun i => μ * buf i + g i / Real.sqrt (rmsSqNext ρ sq g i + ε)

/-- **The textbook spelling, `g / (√s' + ε)` — rendered by NOTHING.** It exists solely so the
    difference from `rmsBufNext` is a theorem rather than a comment, and so the tie harness has a
    named target to build its negative control from. -/
noncomputable def rmsBufNextVanilla (ρ μ ε : ℝ) (sq buf g : Vec n) : Vec n :=
  fun i => μ * buf i + g i / (Real.sqrt (rmsSqNext ρ sq g i) + ε)

/-- One RMSProp step: `(θ', b', s')` — the triple the rendered train step returns per parameter,
    laid out on the packed `[θ|m|v]` signature as `θ' | b' | s'`.

    The gradient `g` passed here is the **weight-decayed** one (`momVNext wd θ g_raw`) on the nets
    that use L2, matching the reference's ordering: decay first, then the accumulator sees it. -/
noncomputable def rmsPropStep (ρ μ ε lr : ℝ) (θ sq buf g : Vec n) : Vec n × Vec n × Vec n :=
  (sgdParam lr θ (rmsBufNext ρ μ ε sq buf g), rmsBufNext ρ μ ε sq buf g, rmsSqNext ρ sq g)

/-- **Mean-square invariant.** `s'` stays nonnegative when `0 ≤ ρ ≤ 1` and the incoming `s` is —
    so the square root below is real at every step.

    ⚠ The reference starts `s` at **1.0**, not 0, so the hypothesis holds from step 0 and the FIRST
    step is damped (`g/√(1−ρ+…)`) rather than amplified. Adam's `v = 0` start is the opposite
    convention and is bias-corrected for it; RMSProp here is not bias-corrected, which is why the
    init value is part of the recipe rather than an implementation detail. -/
theorem rmsSqNext_nonneg {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    {sq g : Vec n} (hsq : ∀ i, 0 ≤ sq i) (i : Fin n) : 0 ≤ rmsSqNext ρ sq g i := by
  have h1 : 0 ≤ ρ * sq i := mul_nonneg hρ0 (hsq i)
  have h2 : 0 ≤ (1 - ρ) * (g i) ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
  show 0 ≤ ρ * sq i + (1 - ρ) * (g i) ^ 2
  linarith

/-- **Well-definedness of the RMSProp update.** With `ε > 0` and a nonnegative mean-square the
    denominator `√(s' + ε)` is strictly positive, so there is no division by zero.

    Note this needs the `rmsSqNext_nonneg` hypothesis where `adam_denom_pos` needs nothing: Adam's
    `√v̂ + ε` is positive from `ε` alone because `Real.sqrt` is unconditionally nonnegative, but the
    TF placement puts the only positive term *inside* the root, so a negative `s'` would make the
    root `0` and the divide undefined. **The ε placement moves the side condition** — a small,
    concrete instance of why the two spellings are different functions. -/
theorem rms_denom_pos {ρ ε : ℝ} (hε : 0 < ε) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    {sq g : Vec n} (hsq : ∀ i, 0 ≤ sq i) (i : Fin n) :
    0 < Real.sqrt (rmsSqNext ρ sq g i + ε) := by
  refine Real.sqrt_pos.mpr ?_
  have := rmsSqNext_nonneg (g := g) hρ0 hρ1 hsq i
  linarith

/-- **Coordinate closed form** — the spec the emitted RMSProp graph must denote, the `rmsBufNext`
    analogue of `adamWParam_apply`. Holds definitionally; stated so the faithfulness proof has an
    explicit per-coordinate target. -/
theorem rmsBufNext_apply (ρ μ ε : ℝ) (sq buf g : Vec n) (i : Fin n) :
    rmsBufNext ρ μ ε sq buf g i =
      μ * buf i + g i / Real.sqrt (ρ * sq i + (1 - ρ) * (g i) ^ 2 + ε) := rfl

/-- **`μ = 0` drops the momentum buffer**, leaving plain (TF-flavoured) RMSProp. The bridge to a
    momentum-free configuration, and a cheap check that the buffer term carries no stray factor. -/
theorem rmsBufNext_mu_zero (ρ ε : ℝ) (sq buf g : Vec n) :
    rmsBufNext ρ 0 ε sq buf g = fun i => g i / Real.sqrt (rmsSqNext ρ sq g i + ε) := by
  funext i; simp [rmsBufNext]

/-- ▶ **THE ε-PLACEMENT DIFFERENCE, made exact.** At a coordinate whose running mean-square has
    collapsed to zero the two spellings scale the gradient by `1/√ε` (TensorFlow) against `1/ε`
    (textbook) — a factor of `1/√ε`, which at EfficientNet's `ε = 1e-3` is **31.6×** and at
    MobileNetV2's `ε = 1.0` is exactly **1×**.

    That asymmetry is the content: the two nets in this repo that use RMSProp sit on opposite sides
    of it, so a render that got the placement wrong would be **invisible on MobileNetV2 and wrong on
    EfficientNet**. Gate both, and do not let an mnv2 tie license the enet render.

    This is the theorem the JAX reference's comment asserts informally (*"with ε=1e-3 the placement
    is ~30× on the effective step when sq is small; vanilla diverges/erodes at the paper LR"*).

    Stated without a `0 < ε` hypothesis deliberately — the two identities hold for any `ε`, and the
    factor `1/√ε` versus `1/ε` is a reading of them, not a side condition on them. -/
theorem rmsBufNext_eps_placement_at_zero {ρ μ ε : ℝ}
    {sq buf g : Vec n} {i : Fin n} (h : rmsSqNext ρ sq g i = 0) :
    rmsBufNext ρ μ ε sq buf g i = μ * buf i + g i / Real.sqrt ε ∧
    rmsBufNextVanilla ρ μ ε sq buf g i = μ * buf i + g i / ε := by
  constructor
  · show μ * buf i + g i / Real.sqrt (rmsSqNext ρ sq g i + ε) = _
    rw [h, zero_add]
  · show μ * buf i + g i / (Real.sqrt (rmsSqNext ρ sq g i) + ε) = _
    rw [h, Real.sqrt_zero, zero_add]

/-- **The buffer is NOT affine in the gradient, even from a zero buffer** — recorded because the
    shard/DP gates lean on the opposite property for every other optimizer here.

    `momVNext_v_zero` gives `v' = g` and `adamMNextF` gives `m' = (1−β₁)·g`, both exactly linear, so
    those renders can use a moment slot as a gradient proxy and average it across replicas
    (`shard-check`, handoff §5). At `buf = 0` RMSProp gives `g/√((1−ρ)g² + ε)` — the gradient
    divided by a function OF the gradient. So `mean(step(gA), step(gB)) ≠ step(mean(gA, gB))` and
    **the asymmetric-batch `shard-check` construction does not transfer to an RMSProp DP render**.
    The duplicated-batch identity (`*-dp-check`) is unaffected: both replicas see the same `g`, so
    `all_reduce(add)/2` is the identity regardless of what the tail does with the result. -/
theorem rmsBufNext_buf_zero (ρ μ ε : ℝ) (sq g : Vec n) :
    rmsBufNext ρ μ ε sq (fun _ => 0) g = fun i => g i / Real.sqrt (rmsSqNext ρ sq g i + ε) := by
  funext i; simp [rmsBufNext]

/-- **Scalar RMSProp buffer update** — one coordinate of `rmsBufNext`, the form a per-entry render
    close applies to a single weight/bias entry's certified gradient (the `adamWScalar` analogue). -/
noncomputable def rmsBufScalar (ρ μ ε sq buf g : ℝ) : ℝ :=
  μ * buf + g / Real.sqrt (ρ * sq + (1 - ρ) * g ^ 2 + ε)

/-- The `Vec` spec is the scalar update applied coordinatewise. -/
theorem rmsBufNext_eq_scalar (ρ μ ε : ℝ) (sq buf g : Vec n) (i : Fin n) :
    rmsBufNext ρ μ ε sq buf g i = rmsBufScalar ρ μ ε (sq i) (buf i) (g i) := rfl

end Proofs
