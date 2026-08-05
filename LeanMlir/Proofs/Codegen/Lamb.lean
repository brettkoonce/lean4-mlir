import LeanMlir.Proofs.Codegen.GradClip

/-! # LAMB over ℝ — RSB-A3's optimizer, and the one item `rsb_a3_r50_verified.md` §2.3 ESTIMATED

The ℝ reference for LAMB (You et al. 2019), coordinatewise over `Vec` where it can be and mirroring
the emitted StableHLO op-for-op, so the faithfulness theorems in `StableHLO.lean` are structural
matches — exactly as `AdamStep` is for the AdamW triple, `RmsPropStep` for RMSProp and `GradClip`
for the clip.

**The reference**, `jax/Jax/Codegen.lean`'s `.lamb` branch, emitted verbatim into
`jax/MainResnet50Imagenet.lean`'s `rsb-faithful` recipe (which reached **76.66% top-1 @ ep100**):

```python
BETA1 = 0.9; BETA2 = 0.999; EPS = 1e-6
m  = BETA1 * m + (1 - BETA1) * g
v  = BETA2 * v + (1 - BETA2) * g * g
mc = m / (1 - BETA1 ** t);  vc = v / (1 - BETA2 ** t)
r  = mc / (jnp.sqrt(vc) + EPS) + WD * msk * p        # DECOUPLED, folded in BEFORE the trust ratio
wn = jnp.sqrt(jnp.sum(p * p));  rn = jnp.sqrt(jnp.sum(r * r))
trust = jnp.where(wn > 0, jnp.where(rn > 0, wn / rn, 1.0), 1.0)
p  = p - lr * trust * r
```

## ⚠⚠ The three things that make this LAMB and not AdamW-with-extra-steps

1. ⭐ **The trust ratio is PER PARAMETER TENSOR** — `jax.tree.map` over leaves. That is the exact
   opposite of `GradClip`'s global norm, whose whole semantic content is that ONE scalar is shared
   across every parameter (`clipFactor_shared`). Reading one as the other is a live confusion: they
   are both "compute a norm, scale by a ratio", and they differ in the quantifier.
2. ⭐ **`ε` is added to `√v̂`, OUTSIDE the root, and the weight decay goes INSIDE the trust ratio.**
   Both placements are load-bearing and both have a plausible wrong neighbour: `√(v̂ + ε)` is
   RMSProp-TF's placement (`RmsPropStep`'s whole point), and decaying after the trust ratio is
   AdamW's. `lambDir_wd_inside` below states the second as an inequality rather than as prose.
3. ⭐ **`trust = 1` when either norm vanishes**, not `0/0` and not `wn/rn` — a zero-initialised β
   (every BN β in this repo's driver init) has `wn = 0` on step 1, so this branch is taken on real
   runs from the first step, not in some corner case.

## Claim ceiling

Like `AdamStep` and `GradClip`, the verified target is **faithfulness** (the rendered LAMB denotes
these functions) plus **well-definedness** — *not* that LAMB converges or that it beats AdamW. Say
"the LAMB render is certified", never "LAMB is proven to work at batch 2048".
-/

namespace Proofs

variable {n : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The direction, before any trust ratio
-- ════════════════════════════════════════════════════════════════

/-- **LAMB's update direction**, `r = m̂/(√v̂ + ε) + wd·θ`, computed from the INCOMING moments and
    this step's gradient in one pass — the shape `adamWParamF` already uses, so the render carries
    one op per parameter here rather than a chain.

    `bc₁`/`bc₂` are the bias-correction denominators `1 − β₁ᵗ` / `1 − β₂ᵗ`, runtime scalars in the
    graph exactly as they are for AdamW, so one render serves every step.

    ⚠ **`Real.sqrt v̂ + ε`, not `Real.sqrt (v̂ + ε)`.** The second is RMSProp-TF's placement and is a
    different optimizer; `RmsPropStep`'s file exists because that distinction was worth ~30× on the
    effective step there. The reference is literal here: `mc / (jnp.sqrt(vc) + EPS)`.

    ⚠ **The decay is DECOUPLED and lands INSIDE `r`**, hence inside the norm the trust ratio takes.
    That is timm's `Lamb`, and it is not AdamW's `θ' = … − lr·wd·θ` moved around: there the decay
    never enters a norm. -/
noncomputable def lambDir (β₁ β₂ ε wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) : Vec n := fun i =>
  let m' := β₁ * m i + (1 - β₁) * g i
  let v' := β₂ * v i + (1 - β₂) * (g i) ^ 2
  (m' / bc₁) / (Real.sqrt (v' / bc₂) + ε) + wd * θ i

-- ════════════════════════════════════════════════════════════════
-- § The trust ratio — the layer-wise part
-- ════════════════════════════════════════════════════════════════

/-- **LAMB's trust ratio**, `‖θ‖ / ‖r‖`, with the reference's guard: `1` when either norm is zero.

    Takes the SQUARED norms, because that is what the graph carries — `gradSumSqAccF` accumulates
    `∑ᵢ gᵢ²` and the root is taken here, once, exactly as `clipFactor` takes `s` rather than `√s`.

    ⚠ **The guard is reached on real runs at step 1**, not in a corner case: the driver initialises
    every BN β and every dense bias to 0 (`VerifiedTrain.mkParam` kind 2), so `wn = 0` for those
    tensors before a single update. `nested jnp.where` in the reference; a single `if` here because
    `0 < wn2 ∧ 0 < rn2` is exactly the conjunction it spells. -/
noncomputable def lambTrust (wn2 rn2 : ℝ) : ℝ :=
  if 0 < wn2 ∧ 0 < rn2 then Real.sqrt wn2 / Real.sqrt rn2 else 1

/-- **The trust-scaled direction**, `trust · r`, where `‖r‖²` is computed FROM `r` and `‖θ‖²` is
    supplied.

    ⚠ The asymmetry is deliberate and it is what keeps the emitted op inside a shape the AST already
    has. `clipScaleF` is `SHlo 1 → SHlo n → SHlo n`; a LAMB scale that took both norms as children
    would be the kit's first TERNARY constructor, and `StableHLO.lean`'s own note records what
    adding an unfamiliar constructor SHAPE cost last time — nine unrelated `simp only [… den …]`
    proofs dying with a `whnf` timeout that 4× the heartbeat budget did not fix. `r` is already the
    tensor child, so `‖r‖²` is free to recompute and `‖θ‖²` is the one scalar that must be threaded.

    ▶ **`gradSumSq` is shared with `GradClip`, not re-derived.** The two features compute the same
    per-leaf quantity and differ only in what they do with it: the clip sums across every leaf and
    shares one factor, LAMB keeps them separate. Writing a second `∑ᵢ gᵢ²` would put that quantity
    in two places, which is exactly the double-writer failure this repo keeps paying for. -/
noncomputable def lambScale (wn2 : ℝ) (r : Vec n) : Vec n := fun i =>
  lambTrust wn2 (gradSumSq r) * r i

-- ════════════════════════════════════════════════════════════════
-- § Well-definedness, and the clauses a plausible wrong LAMB would drop
-- ════════════════════════════════════════════════════════════════

/-- The direction's divisor is strictly positive for `ε > 0` — `Real.sqrt` is unconditionally
    nonnegative, including at the negative arguments it maps to 0, so this needs no hypothesis on
    `v'`. `clipDenom_pos` / `adam_denom_pos` verbatim. This is what makes the reference's `+ 1e-6`
    load-bearing rather than cosmetic: at `ε = 0` with `v = 0` and `g = 0` the ratio is `0/0`. -/
theorem lambDenom_pos (ε x : ℝ) (hε : 0 < ε) : 0 < Real.sqrt x + ε := by
  have hx : 0 ≤ Real.sqrt x := Real.sqrt_nonneg _
  linarith

/-- The trust ratio is nonnegative. -/
theorem lambTrust_nonneg (wn2 rn2 : ℝ) : 0 ≤ lambTrust wn2 rn2 := by
  unfold lambTrust
  split
  · exact div_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  · norm_num

/-- ⭐ **The guard fires, and it gives exactly 1.** A zero-norm parameter — every BN β at init — is
    stepped by `lr · 1 · r`, i.e. plain AdamW-without-decay, NOT by `0` and NOT by `0/0`. -/
@[simp] theorem lambTrust_zero_weight (rn2 : ℝ) : lambTrust 0 rn2 = 1 := by
  unfold lambTrust; simp

/-- The same at a zero direction, the other half of the reference's nested `where`. -/
@[simp] theorem lambTrust_zero_dir (wn2 : ℝ) : lambTrust wn2 0 = 1 := by
  unfold lambTrust; simp

/-- ⭐⭐ **LAMB is NOT AdamW: the trust ratio is per-tensor, so it does not factor out.**

    `GradClip`'s `clipFactor_shared` says the clip's factor is one scalar for every parameter; this
    is the statement that LAMB's is not, made structural — `lambScale` sees only `r` and one norm,
    so its factor is determined by THAT tensor and two tensors with different norms get different
    scales. Stated as the explicit witness rather than as prose, because "compute a norm and scale
    by a ratio" describes both features and the quantifier is the whole difference. -/
theorem lambScale_not_shared :
    ∃ (wn2 : ℝ) (r₁ r₂ : Vec 1),
      lambTrust wn2 (gradSumSq r₁) ≠ lambTrust wn2 (gradSumSq r₂) := by
  refine ⟨1, (fun _ => 1), (fun _ => 2), ?_⟩
  have h1 : gradSumSq (n := 1) (fun _ => (1:ℝ)) = 1 := by
    simp [gradSumSq]
  have h2 : gradSumSq (n := 1) (fun _ => (2:ℝ)) = 4 := by
    simp [gradSumSq]; norm_num
  rw [h1, h2]
  have h4 : Real.sqrt 4 = 2 := by
    rw [show (4:ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]
  unfold lambTrust
  rw [if_pos ⟨by norm_num, by norm_num⟩, if_pos ⟨by norm_num, by norm_num⟩, h4, Real.sqrt_one]
  norm_num

/-- ⭐ **The decay is inside the ratio, and that is observable.** At `wd = 0` the direction is pure
    bias-corrected Adam; any other `wd` moves it by exactly `wd·θ`, BEFORE `lambScale` takes its
    norm. An implementation that decayed after the trust ratio (AdamW's placement) would leave
    `‖r‖` — and therefore the whole step, not just the decay term — unchanged. -/
theorem lambDir_wd_inside (β₁ β₂ ε wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) (i : Fin n) :
    lambDir β₁ β₂ ε wd bc₁ bc₂ θ m v g i
      = lambDir β₁ β₂ ε 0 bc₁ bc₂ θ m v g i + wd * θ i := by
  unfold lambDir; ring

end Proofs
