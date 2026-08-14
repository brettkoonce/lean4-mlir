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
| **D1** LAMB grad clip | ✅ 2026-08-14 | ⛔ **§2** | timm's `Lamb` default `max_grad_norm = 1.0` |
| stochastic depth | ✅ | ⛔ §3 | RSB-A2 needs `0.05`; R50 has no `sd` flag |
| EMA | ✅ | ⛔ §3 | RSB-A2 needs `0.9999`; R50 has no 4th region |
| runtime weight decay | ✅ | ⛔ §3 | `%wd` is a BAKED constant; A1 needs `0.01` |

⭐ D2 is worth one line of retrospect: the verified fix was **cheaper** than the reference's,
because feeding `%lzero` for an excluded parameter lands on `Proofs.lambTrust_zero_weight`
(`lambTrust 0 rn2 = 1`, already `@[simp]`). No new op, no new theorem, one op *fewer* emitted.
▶ Look for that shape first — the proven kit often already contains the case you want.

---

## 2. ▶ D1 — THE GRADIENT CLIP. Start here; it is the only one timm requires.

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
tries to sweep it.

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

### 2d. Artifacts and gates

New renders need a literal `IO.FS.writeFile "verified_mlir/…"` writer in `Proofs/Codegen/`, which is
what `scripts/regen_verified_mlir.sh check` pins. Then:

```
lake build LeanMlir Proofs
scripts/regen_verified_mlir.sh          # only the NEW files should appear
scripts/regen_verified_mlir.sh check
scripts/check_render_coverage.py
lake env lean tests/TestVariantPredicates.lean
```

⭐ And the numeric one, which is the only gate that would catch a clip applied in the wrong place:
`tests/vjp_oracle` / the R50 gradient check against the reference. A clip on unreduced gradients
still trains, still descends, and is a different optimizer.

---

## 3. THE OTHER THREE (RSB-A2/A1, not timm-fidelity)

Ordered by cost. All three have working precedents; none needs a new `SHlo` op.

* **`wdStr`** — thread the decay VALUE as a render parameter, so A1's `0.01` costs a re-render
  rather than a new op. ⚠ It stays a baked `stablehlo.constant` either way: ConvNeXt's
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

1. **D1, the clip** (§2). It is the only timm-required item, it needs no proof work, and the
   all-reduce fix (§2b) is the whole of its difficulty.
2. **`wdStr`** — cheap, unblocks A1.
3. **`ema` × accumulation** — decide the fourth-region conflict *before* rendering anything.
4. **`sd`**, with the misplacement control.
5. The A2/A1 `#eval`s, which are then a slug and a `q`.

⚠ None of §2 or §3 produces a number. Every one of them is a code change awaiting the same re-run
that `next_session_verified_trainer_code.md` §1 already mandates — which is the point: when the
compute question is answered, the runs should be a scheduling decision.

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
