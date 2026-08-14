# verified_optimizer_parity.md — carrying the OPTIMIZER knobs to the verified path

**Opened 2026-08-14**, out of the timm-fidelity work (`planning/resize_eval_reconciliation.md`,
`planning/recipe_fidelity_diffs.md` D1/D2). Companion to
`planning/next_session_verified_trainer_code.md` §3, which lists the same knobs from the RSB-A2
side; this doc is the *how*.

---

## 0. THE SPLIT, and it is the reason this doc exists

Making the JAX reference timm-faithful reached the verified path **unevenly**, and the boundary is
worth naming once because it predicts every future item of this kind.

| | reaches the verified path | why |
|---|---|---|
| **data / augmentation** | ⭐ **automatically** | the verified `.imagenet` trainers do no augmentation at all — batches arrive pre-transformed from that net's `generated_*_imagenet_shim.py`, emitted by the SAME `JaxCodegen`. One definition of the transform, and it is the reference's |
| **optimizer** | ⛔ **not at all** | the verified optimizer is a RENDERED GRAPH (`Proofs/Codegen/`), independent of `Jax/Codegen.lean`. A knob added to the reference does not exist here until it is rendered |

⚠ "Automatically" still means **regenerate the shims**: they are build products, and they were
stale for the whole afternoon of 2026-08-14 until `scripts/gen_shims.sh` ran. ▶ *Run it after any
`Jax/Codegen.lean` change.* The file on disk is the one that trains.

---

## 1. THE KNOBS AND WHERE THEY STAND

| knob | reference | verified | note |
|---|---|---|---|
| `wdExcludeNormBias` | ✅ | ✅ `ccca380` | `%wdz` vs `%wd` per parameter |
| **D2** trust-ratio guard | ✅ | ✅ 2026-08-14 | done on both sides together |
| **D1** LAMB grad clip | ✅ 2026-08-14 | ✅ 2026-08-14 | **§2** — and §2b-bis is the part this doc had wrong |
| stochastic depth | ✅ | ⛔ §3 | RSB-A2 needs `0.05`; R50 has no `sd` flag |
| EMA | ✅ | ⛔ §3 | RSB-A2 needs `0.9999`; R50 has no 4th region |
| runtime weight decay | ✅ | ⛔ §3 | `%wd` is a BAKED constant; A1 needs `0.01` |

⭐ D2 is worth one line of retrospect: the verified fix was **cheaper** than the reference's,
because feeding `%lzero` for an excluded parameter lands on `Proofs.lambTrust_zero_weight`
(`lambTrust 0 rn2 = 1`, already `@[simp]`). No new op, no new theorem, one op *fewer* emitted.
▶ Look for that shape first — the proven kit often already contains the case you want.

---

## 2. ✅ D1 — THE GRADIENT CLIP. **DONE 2026-08-14.** Read §2b-bis before trusting §2b.

**Shipped**: `resnet50in160_lambaccdp8x64wxclipbce_train_step.mlir` and its single-device peer.
`gradClip` on `resnet50TrainStepFaithfulB`, the `clip` axis on `r34AdamVariant`, `preAvg`/`accIn` on
`optOne`, and `clipNormStr`/`clipEpsStr`/`clipZeroConst`/`optAccumK` beside `wdzConst`. Inertness
held: with the flag off **every one of the 174 committed artifacts re-rendered byte-identically**,
and the two new files are additions beside the old ones, never a re-pointing of a run's slug.

⚠ The estimate below was right about the proof work and wrong about the difficulty. The clip needed
no new op, constructor or driver change exactly as §2a said — and then §2b's actual recipe was
unsafe for the one variant RSB-A3 renders. §2b-bis is what happened; it is left in full because the
*shape* of the miss is the reusable part.

`timm.optim.lamb.Lamb.__init__` defaults `max_grad_norm = 1.0` and clips the global gradient norm
inside the optimizer every step. Read from source, not assumed:

```python
clip_global_norm = (global_norm / max_grad_norm).clamp_(min=1.0)   # _get_clip_grad_norm
grad.div_(clip_global_norm)                                        # step()
```

which is `g · min(1, C/‖g‖)` — algebraically the emitted `clipScaleF`, and the reference side's
`g * jnp.minimum(1.0, C / (gn + 1e-6))`.

### 2a. What already exists — this is a WIRING job, not a proof job

* `Proofs/Codegen/GradClip.lean` — **12 theorems**, including `clipFactor_shared` (the clip's factor
  is ONE scalar across every parameter, which is the whole difference from LAMB's per-tensor ratio,
  `lambScale_not_shared`).
* `.gradSumSqAccF` and `.clipScaleF` — both already `SHlo` constructors, both already used.
* `ConvNeXtRender.lean:933–955` — a **working, committed emission** of exactly this shape.
* `tests/TestVariantPredicates.lean` — already carries `clip` rows proving the marker disturbs none
  of the five driver axes.

▶ **So: no new op, no new constructor, no new theorem, and no driver change.**

### 2b. ⚠⚠ THE ONE REAL DESIGN ISSUE — the all-reduce is in the wrong place

The clip must be taken on the **averaged** gradient, or each replica clips its own and the result is
not a clip of anything. ConvNeXt gets this right by all-reducing *inside* the clip branch
(`ConvNeXtRender.lean:936–939`) before folding the norm.

R50 cannot copy that directly, because **`optOne` does its own all-reduce, per parameter**:

```
ResNet34RenderB.lean:332
  let (arS, gAvg) := ViTRender.emitGradAllReduce g.grad g.ds g.nm replicas
```

So a naive clip in the caller would either reduce twice or clip pre-reduction.

⭐ **The fix needs no interface change at all.** `emitGradAllReduce` at `replicas ≤ 1` emits
**nothing** and threads its input name straight through (its own docstring says so, and that is the
existing byte-identity self-check for single-device renders). So:

1. in the caller, when `clip` is on: all-reduce every parameter's gradient, fold
   `gradSumSqAccF` across **all** of them into one `SHlo 1` seeded at `%zero`, `pretty` it to
   `normSSA`, then emit `clipScaleF` per parameter;
2. rewrite each `PGrad.grad` to the clipped SSA name;
3. call `optOne` with **`replicas := 1`** — its all-reduce then vanishes and it consumes the clipped,
   already-averaged gradient.

⚠ **The inertness condition, and it is the check that the change is scoped:** when `clip` is off,
take the existing path unchanged (`optOne` with the real `replicas`), so **every committed artifact
re-renders byte-identically**. That is how D2 was validated — only the two `wx` LAMB files moved.

⚠ `clipStr` is a **baked string constant**, like `%wd` and unlike `%lr`. The clip norm lives in the
artifact, so changing it is a re-render. Fine for timm's fixed `1.0`; worth knowing before anyone
tries to sweep it. ⭐ It shipped as a **`Float`** rather than a `String`, unlike ConvNeXt's, for
§2b-bis's reason: under accumulation the baked threshold is arithmetic in `k`, not a literal.

### 2b-bis. ⚠⚠ WHAT §2b MISSED — there are TWO orderings, and it named one

§2b says: all-reduce in the caller, fold, clip, then `optOne` with `replicas := 1`. That is right
for `.adamw`/`.lamb`/`.heavyBall` and **wrong for `.adamwAccum`/`.lambAccum`** — which is
`lambAccum 8`, i.e. the only optimizer RSB-A3 and RSB-A2 ever render. Under §2b's recipe the clip
lands on the **micro-batch** gradient, before the accumulator.

The reference settles it, and it is worth quoting rather than paraphrasing
(`jax/Jax/Codegen.lean:2439-2441`):

```python
grads = jax.tree.map(lambda _a: _a / _K, _gsum)   # the MEAN over k micro-batches
loss  = jnp.mean(_ls)
gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, C / (gn + 1e-6)), grads)
```

The clip is on the **mean accumulated** gradient. So the second ordering is *the clip goes after the
accumulation*, and it forces two things §2b did not anticipate:

1. **The accumulator hoists too.** `Gt` is computed per parameter inside `optOne`, and the fold needs
   every parameter's `Gt` at once — so `optOne` grew `accIn` beside `preAvg`, and the caller emits
   the `momVNextF` itself.
2. **⚠⚠ The fourth region must keep the RAW `Gt`, not the clipped one.** The reference's carry
   (`_gsum`) is raw and is clipped only on the way into the optimizer. Returning the clipped total
   as `%<p>a` would compound the clip across every one of the k micro-batches — a contraction that
   trains, descends, and is not the recipe. That is why `accIn` names the raw accumulator while
   `g.grad` carries the clipped one, and why they cannot collapse into one name.

⭐ **The threshold moves with the fold.** The graph never materialises the mean (the `1/k` is folded
into `%ob1`/`%ob2`, split because `v` is quadratic), so the fold runs on `Gt = k·mean` and the baked
constants become `k·C` and `k·ε`:

`min(1, kC/(‖Gt‖ + kε)) = min(1, C/(‖Gt‖/k + ε))`

▶ That identity is **`Proofs.clipFactor_accum`**, with `Proofs.clipGrad_accum` as its vector form and
`Proofs.gradSumSq_smul` underneath — added to `GradClip.lean` rather than left as a comment, because
"algebraically equal" is the class of claim this repo has been wrong about before, and because
**every other theorem in that file is blind to it**: `k·C`-on-the-sum and `C`-on-the-mean both scale,
neither amplifies, and both are the identity below threshold. ⚠ Note `ε` scales too — leaving it
alone breaks the identity in exactly the near-zero regime the guard exists for.

⚠ On an ACCUMULATE micro-batch the clip is taken on a partial `Gt` and then discarded (`%lr = 0`,
`%b1 = %b2 = 1`, `%ob1 = %ob2 = 0`), so only the APPLY micro-batch's factor can reach a weight. That
is what makes this expressible without a second buffer, and it is worth knowing before anyone reads
the partial-sum clip as a bug.

▶ **The generalisable lesson, and it is §0's boundary again.** A knob that is "just wiring" on the
optimizer still has to be placed against **every other optimizer axis already in the graph**, not
just against the one it was specified with. D1 was specified against `Lamb`; it shipped against
`Lamb × accumulation × data-parallel`, and two of those three interactions changed the answer.

### 2c. The variant name — the defect ConvNeXt shipped twice

Add a trailing, defaulted `gradClip` flag, spelled like `wdExclude`:

* `ResNet34RenderB.r34AdamVariant` — currently `(B, replicas, opt, wdExclude)`, **no clip axis**.
* `ResNet50RenderB.resnet50TrainStepFaithfulB` — currently
  `(B, nClasses, epsStr, replicas, opt, slug, bce, vSuffix, q, wdExclude)`; the flag goes **last**
  (§2m: a parameter inserted mid-list captures an existing positional argument).

⚠⚠ **It must reach `r34AdamVariant`, not just the renderer.** R50 derives its ENTRY NAME from the
variant, so a flag that reaches the emission but not the name produces an artifact whose declared
entry disagrees with its own path, and the shim refuses the call outright. `r34AdamVariant`'s own
docstring records that ConvNeXt shipped exactly this defect **twice** — once for `wx`, once for
`clip`. The `#guard`s below that function pin every spelling; extend them.

▶ Placement: `wx` trails the batch and `bce` trails everything, giving `lambaccdp8x64wxbce` today.
Follow ConvNeXt's order — `wxclip` — for `lambaccdp8x64wxclipbce`. ⚠ The order is a *choice*, and the
`#guard`s are what make it a fixed one.

### 2d. Artifacts and gates — ✅ all green 2026-08-14

New renders need a literal `IO.FS.writeFile "verified_mlir/…"` writer in `Proofs/Codegen/`, which is
what `scripts/regen_verified_mlir.sh check` pins. Then:

```
lake build Proofs                       # ⚠ NOT `lake build LeanMlir` — see below
scripts/regen_verified_mlir.sh          # only the NEW files should appear
scripts/regen_verified_mlir.sh check
scripts/check_render_coverage.py
lake env lean tests/TestVariantPredicates.lean
```

⚠⚠ **`lake build LeanMlir` DOES NOT BUILD THESE RENDERERS, and it exits 0.** `lean_lib LeanMlir` has
`roots := #[LeanMlir]`, but `LeanMlir.Proofs.Codegen.*` reaches the build only through the `Proofs`
lib's explicit root list. A `LeanMlir`-only build of an edited renderer reports *"Build completed
successfully"* while re-elaborating nothing — and since the `#eval` writers run at elaboration time,
`git diff verified_mlir/` is then clean *because nothing was written*, which reads exactly like the
byte-identity result you were looking for. ▶ Confirm by mtime, not by exit code: the artifacts must
carry the timestamp of the build you just ran. (The line above said `lake build LeanMlir Proofs`,
which is correct but invites dropping the second word.)

⭐ `check_render_coverage.py` **caught the two new artifacts as unguarded** and named the fix — they
are now in the drift-guard step of `.github/workflows/proofs.yml`. That gate works; do not skip it
on the assumption that a new render is automatically covered.

⭐⭐ And the numeric one, which is the only gate that can see a clip applied in the wrong place — a
clip on unreduced or unaccumulated gradients still trains, still descends, and is a different
optimizer. ▶ **This now exists: §5b, `scripts/opt_step_tie.py`.** It ties the rendered optimizer to
the reference's own at ~1e-7 on four variants, and both of D1's decisions were reverted to confirm
it turns red. Run it before believing any change to the optimizer stage.

⚠ The emitted MLIR was also read directly, and that reading is what the gate was built to replace.
On `resnet50in160_lambacc8x64wxclipbce`:

* one `%czero`-seeded fold, and **every** `clipScale` block re-roots the *same* summed scalar —
  `Proofs.clipFactor_shared`'s property, present;
* `dense<8.000000000000>` / `dense<0.000008000000>` — the `k = 8` instance;
* the clipped value feeds the moments and `lambDirF`, while the **raw** accumulator is what appears
  in the return list at index 484 = 161·3 + 1, i.e. region 4;
* at 4 replicas: exactly **161** `all_reduce` ops (one per parameter, not two), each *before* its
  accumulator, which consumes `%armean*` and not the raw gradient.

---

## 3. THE OTHER THREE (RSB-A2/A1, not timm-fidelity)

Ordered by cost. All three have working precedents; none needs a new `SHlo` op.

* **`wdStr`** — ✅ **DONE 2026-08-14.** `optWdDefault` (1e-4 for the AdamW family, 0.02 for LAMB's),
  `optWdStr` (the override, empty = the optimizer's own), `optConstsB opt wdStr`, and a trailing
  `wdStr` on `resnet50TrainStepFaithfulB`. Byte-inert: all 176 artifacts unchanged.
  ⚠⚠ **It had to reach the NAME, which §3 did not anticipate.** `%wd` is a baked constant, so an A1
  render (0.01) and an A3 render (0.02) at the same optimizer/batch/replicas would otherwise both be
  `lambaccdp8x64wxclipbce` — the last-writer-wins race §2a already cost this repo an artifact.
  `wdVariantMark` appends `wd001`, and it is PER-OPTIMIZER because the same value means different
  things to the two families (0.02 is LAMB's default and a 200× override for AdamW).
  ⭐ **And it is a MEASURED knob, not a threaded string**: `scripts/opt_step_tie.py` gained a
  `lambacc8wxclipwd001` row against `generated_resnet50_imagenet_a1.py`, which bakes `WD = 0.010000`.
  Pointed at the wd = 0.02 reference instead it fails at 9.5e-05 against a 1e-7 floor, and the worst
  rows are `θ'[0]` and `θ'[2]` — the two DECAYING parameters, with the rank-1 one `wx` excludes
  untouched. A decay knob that never reached the graph could not produce that pattern.
  ▶ ⚠ A full A1 render still needs `sd` and `ema` below; this removes the decay from that list.
  (original note: thread the decay VALUE as a render parameter, so A1's `0.01` costs a re-render
  rather than a new op.) ⚠ It stays a baked `stablehlo.constant` either way: ConvNeXt's
  `convnextAdamConsts (wdExclude) (wdStr := "0.0001")` emits
  `%wd = stablehlo.constant dense<{wdStr}>` (`ConvNeXtRender.lean:844,854`). So this is a
  *parameterised constant*, not a runtime operand — copy that shape verbatim. Cheapest of the three.
* **`ema`** — a FOURTH blob region `[θ|m|v|ema]` and 5 scalars, exactly as ConvNeXt/ViT.
  `VerifiedVariant.nRegions`/`nScalars` already return 4/5 for an `ema*` variant, so the driver is
  done. ⚠⚠ **It cannot compose with accumulation, and RSB-A2 needs both** — verified, not assumed:
  `MainResnet50Imagenet.lean:73` sets `useEMA := true` / `emaDecay := 0.9999` on the A2 base (A3's
  `short` turns it off at :127), and `a2-accum` sets `gradAccumSteps := 4` at :194. `trainAdamSched`
  throws on `emaOn && accOn` *deliberately*, because the EMA shadow and the gradient accumulator
  occupy the same fourth region of `[θ|m|v|·]`.
  ▶ **This is a real blocker for A2 on a 16 GB box**, where accumulation is what makes bs2048 fit at
  all, and it wants deciding — a FIFTH region, or `a2-true-2048` on rented memory — *before* anything
  is rendered, not after.
* **`sd`** — stochastic depth, the biggest of the three and the only one with a design question:
  where the site goes in a bottleneck, and whether the stage-first blocks (which carry a PROJECTION
  shortcut) get one. R50 has 16 blocks so the ramp denominator is 15. ⚠⚠ Read
  `planning/stochastic_depth.md` §7b before placing a single site: an all-ones-mask gate is
  structurally blind to placement, and `scripts/misplace_drop_sites.py` is what makes a green run
  mean anything.

---

## 4. SUGGESTED ORDER

1. ~~**D1, the clip** (§2).~~ ✅ **done 2026-08-14.** ⚠ The estimate — "no proof work, and the
   all-reduce fix is the whole of its difficulty" — was wrong twice over: the accumulation ordering
   was a second, unnamed difficulty (§2b-bis), and it *did* want proof work, three theorems of it,
   because the fix rests on an algebraic identity rather than on a placement.
2. **`wdStr`** — cheap, unblocks A1. ▶ **Next.**
3. **`ema` × accumulation** — decide the fourth-region conflict *before* rendering anything.
4. **`sd`**, with the misplacement control.
5. The A2/A1 `#eval`s, which are then a slug and a `q`.

⚠ None of §2 or §3 produces a number. Every one of them is a code change awaiting the same re-run
that `next_session_verified_trainer_code.md` §1 already mandates — which is the point: when the
compute question is answered, the runs should be a scheduling decision.

⚠ D1 does not change that, and it adds a **third** un-run RSB-A3 render beside the two `wx` ones. It
is worth saying plainly: `resnet50in160_lambaccdp8x64wxclipbce` is now the artifact that most nearly
implements the recipe on paper, and it has never executed a step. Nothing about it can be compared
to 77.43% except by a fresh run.

---

## 5. ⭐ THE GAP THIS DOC IS REALLY ABOUT

The reference and the verified path share a data pipeline **by construction** and share an optimizer
**by nobody's construction** — the second is two independent implementations that happen to agree,
maintained by hand, and D1 is what a drift between them looks like.

`tests/vjp_oracle` diffs them at the *gradient*; nothing diffs them at the *update*. ▶ A worthwhile
follow-up, and cheaper than it sounds: one step of each optimizer on the same `(θ, g, state)` and a
tolerance, per variant. That is the same shape as `score-checkpoint`'s equality gate — re-derive the
number the other side would print, and demand it — which is the pattern that has caught the most
this month.

## ✅ 5b. THE ONE-STEP UPDATE DIFF — **BUILT 2026-08-14**

    lake build opt-step-fixtures && .lake/build/bin/opt-step-fixtures
    .venv/bin/python scripts/opt_step_tie.py

D1 is what forced this. The gap was never just that the two optimizers *might* drift — it is that
**the verified side has now made a semantic decision the reference never had to make**: the
reference divides by `k` and clips, the render folds on the sum and scales the threshold. Those
agree by `Proofs.clipFactor_accum`, which is machine-checked — but the theorem covers the *algebra*,
not the *wiring*: that the render passes the same `k` to `clipNormStr` as `accumScalarConsts` folds
into `%ob1`/`%ob2`, on the same tensor, in the same order.

**What it is.** `tests/TestOptStepFixtures.lean` emits the optimizer stage ALONE — one step as a
function of `(θ, g, m, v, G)`, gradients as function ARGUMENTS so the update is isolated from the
gradient half `vjp_oracle` already covers — and `scripts/opt_step_tie.py` runs it through XLA
against the reference's own optimizer lines. Seconds, on CPU. Result:

```
✓ lamb            K=1  ref=r50.py           worst rel 8.05e-08
✓ lambwxclip      K=1  ref=r50_a2accum.py   worst rel 8.71e-08
✓ lambacc4wxclip  K=4  ref=r50_a2accum.py   worst rel 1.09e-07
✓ lambacc8wxclip  K=8  ref=r50_a2accum.py   worst rel 1.09e-07
```

⭐ **Two things make it a gate rather than a decoration**, and both are the lessons this repo keeps
re-learning:

1. **It drives the SHIPPED emission.** `optAllParams` was factored out of
   `resnet50TrainStepFaithfulB` (byte-identically) so the fixture calls the same function the real
   render calls. §5's own point one level down: *a gate on a copy is not a gate on the thing copied.*
2. **The reference is EXECUTED, not re-implemented** — the optimizer lines are extracted from
   `generated_resnet50_imagenet*.py` and `exec`d verbatim, no line edited, added or skipped. The
   variants were chosen to match references that ship, which is why the list is four rows and not
   the seven a coverage instinct would have written.

⭐⭐ **And it was shown to fail.** Both of D1's new decisions were reverted and re-run: dropping the
`k` scaling gives **1.8e-01 / 2.1e-01**, and returning the clipped total as region 4 gives
**9.4e-01 / 8.8e-01** — six orders above the 1e-7 floor, and in both cases **only on the
accumulating rows**, which is what the `k = 1` control exists to say. Under the second the reported
worst rows are `G'[0..2] (RAW)`, so the gate names the accumulator rather than leaving a bisect. The
table is in the script's header.

⚠ **`.adamw` is not covered**, and it is a real gap: no generated reference bakes R50's AdamW
constants (`%eps = 1e-8`, `%wd = 1e-4`) — the Adam-family references in the tree are EfficientNet's
and MNv2's TF-RMSProp recipes. Closing it wants a config that GENERATES that reference, not a
transcription written into the harness.

▶ **What this now buys the rest of §3.** `wdStr`, `ema` and `sd` are all code changes to the same
optimizer stage, and each one now has a place to prove itself in seconds instead of in a run: add a
config that generates the reference, add a fixture row, and the diff either ties or names the
tensor it disagrees on.
