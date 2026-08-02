import LeanMlir.Proofs.Architectures.ConvNeXt

/-! # Stochastic depth (drop-path) over ℝ — the per-example branch scale

The ℝ reference for `planning/stochastic_depth.md`. The JAX reference emits, verbatim
(`jax/Jax/Codegen.lean:1037`):

```python
def _drop_branch(branch, drop_key, keep_prob):
    shape = (branch.shape[0],) + (1,) * (branch.ndim - 1)
    keep  = jax.random.bernoulli(drop_key, keep_prob, shape).astype(branch.dtype)
    return branch * keep / keep_prob
```

**⚠ It is a per-SAMPLE scale, not a switch on the block** — the mask is `(B, 1, …, 1)` and
broadcasts over every non-batch axis, so "stochastic depth" is a diagonal linear map at the batched
index and nothing about the architecture changes. `recipe_gaps.md` files this as Tier E, *"a new
layer family"*; it is not one. What it genuinely costs is on the *plumbing* side (a per-step random
graph INPUT), not here.

**▶ Everything in this file is a reading of `layerScale`, so it adds NO new proof obligation.**
`dropPath` is `layerScale` at a per-example-broadcast scale vector, hence:

| piece | where it comes from |
|---|---|
| forward | `layerScale (dropScale …)` — `rfl` |
| VJP | **`layerScale_has_vjp`, verbatim.** The map is diagonal, so its own transpose |
| the backward emitter | **none needed** — `dropPath_vjp_is_self` says the backward IS the forward at the same mask |
| float story | `floatBridges_diagBack` already covers the shape (`LinBackFloatBridge.lean:75`) |

That is the fourth time enumerating a reference feature against existing ops *at their other
readings* has collapsed a scoped op family (§2k heavy-ball, recipe_gaps v1.2 RMSProp, the EMA
shadow, here) — and note the pattern is now strong enough to be a first move rather than a lucky
one: **read the reference's update, then look for it among the ops you have before adding one.**

**⚠ WHERE THE RANDOMNESS IS NOT.** The mask is a graph **input**, drawn on the host next to the
augmentation seed. `stablehlo.rng` is disqualified and not on taste: every numeric gate in this repo
is a bit-exactness or known-answer argument over a *deterministic* graph — the tie harnesses' A-vs-A
floor, `residency_gate.sh`'s bit-identity, the duplicated-batch DP identity, the cross-lowerer
IREE-vs-XLA agreement. A graph that draws its own randomness makes every one of those either
impossible or contingent on seeding an XLA RNG identically across two lowerers and two vendors.

**Claim ceiling.** The verified target is **faithfulness** (the rendered op denotes `dropPath`) plus
the exact identity at a ones mask. Stochastic depth is a *regulariser*; nothing here says it
improves generalisation, and no theorem in this repo could. -/

namespace Proofs

/-! ### ⚠⚠ WHERE `1/keep_prob` LIVES — a tension in the spec, settled here

`planning/stochastic_depth.md` asks for two things that **cannot both hold**, and neither §1 nor §3
noticed:

* **§1 fact 3** — the keep probability is *"emitted as a constant, exactly like every other
  hyperparameter"*;
* **§3** — the FORWARD render emits the same drop sites at an all-ones mask, so the
  `forward ⊂ train-step` prefix audit survives and *"eval is the identity"* becomes a claim about
  one graph at a particular input.

With `1/keep_i` baked, a ones mask computes `x / keep_i`, **not** `x` — and `keep_i < 1` at every
site but the first, so eval would silently rescale every residual branch upward. The reference is
unambiguous that this must be exact: `if drop_key is None or keep_prob >= 1.0: return branch`.

**Settled by folding `1/keep_i` into the supplied mask**: the graph is a pure per-example scale, the
driver passes `bernoulli(keep_i)/keep_i` at train and `1.0` at eval. Then §3 holds exactly, gate 1
holds at `dropRate = 0`, and the emitted text is identical in the forward and the train step.

⚠ **That moves the ramp from the graph to the driver, and that is the repo's own strongest
precedent rather than a concession.** `%lr` is a runtime operand for exactly this reason — one graph
serves a whole schedule — and a learning rate baked into a graph constant is the documented
`RenderCifar8Sgd02` / enet-16× silent-hyperparameter failure. The keep ramp is a per-site
hyperparameter schedule; it belongs in the same place. What it costs is that no *render-level* check
can see a wrong ramp, which is what `keepProb` below and the known-answer gate exist for. -/

/-- **The drop-path scale vector** — the supplied per-example scale `s`, broadcast over the
    per-example width `n`.

    `finProdFinEquiv.symm idx` splits the batched index exactly as `batchMapAux` does, so `.1` is
    the example and `.2` the position within it. That the scale is indexed by `.1` ALONE is the
    formal content of "the mask is `(B, 1, …, 1)` and broadcasts": two positions in the same example
    are scaled identically, and two examples are scaled independently. Emitting a per-ELEMENT scale
    instead typechecks and trains — it is per-element dropout, a different regulariser — which is
    what `tests/TestBatchedEmitTie.lean`'s `dims = [0]` assertion pins. -/
noncomputable def dropScale (N n : Nat) (s : Vec N) : Vec (N * n) :=
  fun idx => s (finProdFinEquiv.symm idx).1

/-- **Drop-path forward** — the per-sample residual-branch scale, at the batched index. -/
noncomputable def dropPath (N n : Nat) (s : Vec N) : Vec (N * n) → Vec (N * n) :=
  layerScale (dropScale N n s)

@[simp] theorem dropPath_apply (N n : Nat) (s : Vec N) (x : Vec (N * n)) (idx : Fin (N * n)) :
    dropPath N n s x idx = s (finProdFinEquiv.symm idx).1 * x idx := rfl

/-- ⭐ **The supplied scale IS the reference's `keep / keep_prob`.** Stated rather than assumed,
    because folding the inversion into the input is precisely the step at which "inverted
    stochastic depth" could quietly become the *un*-inverted kind — which trains, and shifts every
    activation's scale at eval. At `s b = keep b / kp` this op computes the reference's
    `branch * keep / keep_prob` coordinate for coordinate. -/
theorem dropPath_eq_reference (N n : Nat) (keep : Vec N) (kp : ℝ) (x : Vec (N * n))
    (idx : Fin (N * n)) :
    dropPath N n (fun b => keep b / kp) x idx
      = x idx * keep (finProdFinEquiv.symm idx).1 / kp := by
  simp [dropPath, dropScale, layerScale]; ring

/-- ⭐ **EVAL IS THE IDENTITY, EXACTLY — and this is a theorem about ONE graph, not two.**

    `planning/stochastic_depth.md` §3's design turns the train/eval divergence into a *data*
    difference: the forward render emits the drop sites too, and the driver supplies an all-ones
    scale there. So the emitted text is identical in train and eval, the `forward ⊂ train-step`
    prefix audit survives untouched (it is one of the two load-bearing structural gates in the repo
    — it caught `resnet34_fwd` and `mobilenetv2_fwd` scoring nets they had not trained), and "eval
    is the identity" stops being a claim about two graphs and becomes this:

    `s ≡ 1 ⇒ dropPath = id`, exact in IEEE because `1 * x = x` is exact. ⚠ This is why `1/keep_i` is
    folded into the supplied scale rather than baked (see the note above): a baked `1/keep_i` would
    make the ones-mask forward compute `x / keep_i`, and the reference is explicit that eval returns
    the branch untouched.

    The render-side peer is `fwd-tie convnext` coming back BIT-EXACT against the committed
    pre-change forward when fed a ones mask. -/
@[simp] theorem dropPath_ones_id (N n : Nat) (x : Vec (N * n)) :
    dropPath N n (fun _ => 1) x = x := by
  funext idx; simp [dropPath, dropScale, layerScale]

/-- **A zero mask kills the branch exactly.** The other endpoint, and the render's second control:
    an all-zero mask on one site must make that branch contribute nothing, which is what pins the
    site to where the renderer claims it is. -/
@[simp] theorem dropPath_zeros_zero (N n : Nat) (x : Vec (N * n)) :
    dropPath N n (fun _ => 0) x = fun _ => 0 := by
  funext idx; simp [dropPath, dropScale, layerScale]

/-- **The VJP, and it is `layerScale`'s verbatim.** A diagonal linear map is its own transpose, so
    there is no second emitter, no `*Grad` peer and no new certificate — the whole backward story
    for stochastic depth is this line. -/
noncomputable def dropPath_has_vjp (N n : Nat) (s : Vec N) :
    HasVJP (dropPath N n s) :=
  layerScale_has_vjp (dropScale N n s)

/-- ⭐ **THE BACKWARD IS THE FORWARD.** `y = c ⊙ x ⇒ dx = c ⊙ dy` at the same `c`, so the renderer
    emits the *same op* on the cotangent that it emitted on the activation — the same mask, the same
    `invKeep`. Stated rather than assumed, because "reuse the forward op on the backward path" is
    exactly the kind of step that is obvious right up until the mask is per-example and someone
    reaches for a transposed index. -/
theorem dropPath_vjp_is_self (N n : Nat) (s : Vec N) (x dy : Vec (N * n)) :
    (dropPath_has_vjp N n s).backward x dy = dropPath N n s dy := rfl

/-- The `correct` field spelled out, matching every other `_has_vjp_correct` in the kit. -/
theorem dropPath_has_vjp_correct (N n : Nat) (s : Vec N)
    (x dy : Vec (N * n)) (i : Fin (N * n)) :
    (dropPath_has_vjp N n s).backward x dy i =
      ∑ j : Fin (N * n), pdiv (dropPath N n s) x i j * dy j :=
  (dropPath_has_vjp N n s).correct x dy i

/-- **The keep-probability ramp**, `keep_i = 1 − dropPath · i / (totalDrop − 1)`.

    ⚠⚠ **`totalDrop` counts ALL blocks, including ones the drop never fires on.** The reference sums
    the block count of every stage (`Codegen.lean:1888`) and its own comment says *"the drop only
    actually fires where a skip exists … so no-skip blocks just carry a unit keep"* — so the ramp
    index advances over blocks that do not drop. **Deriving the denominator from the drop-ELIGIBLE
    blocks instead silently changes every keep probability in the net**, which compiles, runs,
    descends, and trains a different objective. That is §2k's `α/K` bug in a new place and it has
    the same signature; the gate for it is a known answer, not a tie (every tie compares the render
    against a peer built from the same constant, so none of them can see it).

    Kept in ℝ here so the denotation side has one definition of the ramp; the renderer's Float peer
    must agree with it, and does, because both read the same `(i, totalDrop)` out of the same block
    traversal. -/
noncomputable def keepProb (dropRate : ℝ) (i totalDrop : Nat) : ℝ :=
  1 - dropRate * (i : ℝ) / ((totalDrop : ℝ) - 1)

/-- **Block 0 keeps everything.** The cheap end of the ramp, and the first thing a wrong denominator
    breaks. -/
@[simp] theorem keepProb_zero (dropRate : ℝ) (totalDrop : Nat) :
    keepProb dropRate 0 totalDrop = 1 := by simp [keepProb]

/-- **The last block drops the most**: at `i = totalDrop − 1` the ramp reaches `1 − dropRate`
    exactly. Requires `totalDrop ≥ 2`, which every net that sets `dropPath` satisfies (ConvNeXt-T
    has 18 blocks, ViT-Tiny 12, EfficientNet-B0 16). -/
theorem keepProb_last (dropRate : ℝ) (totalDrop : Nat) (h : 2 ≤ totalDrop) :
    keepProb dropRate (totalDrop - 1) totalDrop = 1 - dropRate := by
  have hd : ((totalDrop : ℝ) - 1) ≠ 0 := by
    have : (2 : ℝ) ≤ (totalDrop : ℝ) := by exact_mod_cast h
    intro hc; rw [sub_eq_zero] at hc; rw [hc] at this; norm_num at this
  have hcast : ((totalDrop - 1 : Nat) : ℝ) = (totalDrop : ℝ) - 1 := by
    have : 1 ≤ totalDrop := le_trans (by norm_num) h
    push_cast [Nat.cast_sub this]; ring
  rw [keepProb, hcast, mul_div_assoc, div_self hd, mul_one]

/-- **`dropRate = 0` is the identity ramp** — every site keeps everything, so the whole feature is
    inert. This is the denotation-side peer of gate 1's strong form (*"at `dropPath = 0` every
    committed artifact re-renders byte-identically"*): the render emits nothing, and had it emitted
    something, this says it would have computed nothing either. -/
@[simp] theorem keepProb_zero_rate (i totalDrop : Nat) : keepProb 0 i totalDrop = 1 := by
  simp [keepProb]

end Proofs
