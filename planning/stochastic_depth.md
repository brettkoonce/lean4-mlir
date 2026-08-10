# stochastic_depth.md — scoping stochastic depth (drop-path) for the verified renderers

**Written 2026-08-02.** `planning/recipe_gaps.md` files this as **Tier E**, *"architectural: a
per-block Bernoulli mask plus its VJP, and a train/eval divergence. This is a new layer family, not
a knob."* Scoped against the reference and the existing op kit, **that tiering is wrong in both
directions**, which is why it was worth reading before building:

* the **math is smaller** than Tier E implies — it is not a new layer family, it is a per-example
  scale, and its VJP is itself;
* the **plumbing is larger** — it introduces the first *per-step random graph INPUT* this repo has
  ever had, and that touches the driver, the shim's shard mask, and every gate on three nets.

Nothing here is built. This is the measurement that decides whether and how to build it.

---

## 1. What the reference actually does — read, not assumed

`jax/Jax/Codegen.lean:1037`, emitted verbatim into every reference trainer that sets `dropPath`:

```python
def _drop_branch(branch, drop_key, keep_prob):
    if drop_key is None or keep_prob >= 1.0:
        return branch
    shape = (branch.shape[0],) + (1,) * (branch.ndim - 1)
    keep = jax.random.bernoulli(drop_key, keep_prob, shape).astype(branch.dtype)
    return branch * keep / keep_prob
```

Five facts, each of which changes the design:

1. **Per-SAMPLE, not per-block.** The mask is `(B, 1, …, 1)` and broadcasts over every non-batch
   axis. So it is a **per-example scalar multiply**, not a switch on the block.
2. **Inverted.** Survivors are scaled by `1/keep_prob`, so the expectation is preserved and
   **inference is exactly the identity** (`drop_key = None`).
3. **The keep probability RAMPS by depth and is a compile-time literal**:
   `keep_i = 1 − dropPath · i / (totalDrop − 1)`. Block 0 keeps everything; the last block drops
   the most. It is emitted as a constant, exactly like every other hyperparameter.
4. ⚠ **`totalDrop` counts ALL blocks, including ones the drop never fires on.** `Codegen.lean:1888`
   sums the block count of every stage/encoder; the emitter's own comment says *"the drop only
   actually fires where a skip exists (guarded inside `mbconv_block`), so no-skip blocks just carry
   a unit keep."* The ramp index therefore advances over blocks that do not drop. **Deriving the
   denominator from the drop-eligible blocks instead would silently change every keep probability
   in the net.**
5. **Sites per block differ by architecture:**

| net | `dropPath` | blocks | drop SITES | note |
|---|---|---|---|---|
| **ViT-Tiny** | 0.1 | 12 | **24** | each block drops the attention and MLP branches **independently** (`ka, km = split(drop_key)`) but **both use the same `keep_i`** — sites ≠ ramp index |
| **ConvNeXt-T** | 0.1 | 18 (`[3,3,9,3]`) | **18** | one per block, on the residual branch |
| **EfficientNet-B0** | 0.2 | 16 MBConv | ≤16 | fires only where a skip exists; the others carry keep = 1 |

---

## 2. ▶ THE CRUX: where the randomness lives. It must NOT be in the graph.

This is the whole design decision and everything else follows from it.

**In-graph RNG (`stablehlo.rng`) is disqualified**, and not on taste: every numeric gate in this
repo is a bit-exactness or known-answer argument over a *deterministic* graph — the tie harnesses'
A-vs-A floor (§2b), `residency_gate.sh`'s bit-identity, the duplicated-batch DP identity, the
cross-lowerer IREE-vs-XLA agreement (§2h). A graph that draws its own randomness makes every one of
those either impossible or dependent on seeding an XLA RNG identically across two lowerers and two
vendors. That is a large amount of evidence to trade away for a mask.

**▶ Recommended: the mask is a host-drawn graph INPUT**, one `tensor<Bxf32>` per drop site. The
graph stays a deterministic function of its inputs; the randomness lives in the driver, next to the
augmentation seed it already owns. This is §2k's own argument one level over — *the transform has
exactly one definition, and the render owns no augmentation code*.

Consequences, all good:
* no new proof obligation about randomness — the op is a multiply by a supplied vector;
* every existing gate keeps working by feeding a **fixed** mask (all-ones, or a pinned pattern);
* **eval is the same graph with an all-ones mask**, which is the next point and is the best part.

---

## 3. ▶ The design that keeps the prefix audit alive — make train/eval a DATA difference

The obvious reading of "train/eval divergence" is that the train step emits drop ops and the eval
forward does not. **That would break `regen_verified_mlir.sh`'s forward ⊂ train-step prefix audit**
on all three nets — the audit that caught `resnet34_fwd` and `mobilenetv2_fwd` scoring nets they had
not trained (§2a, §2g). It is one of the two load-bearing structural gates in the repo and it should
not be spent on this.

**Emit the drop sites in the forward too, at an all-ones mask.** Then:

* the emitted text is **identical** in both, so the byte-identical prefix property survives
  untouched;
* "eval is the identity" stops being a claim about two graphs and becomes a claim about one graph at
  a particular input — `keep = 1, mask ≡ 1 ⇒ branch · 1 / 1.0 = branch`, exact in IEEE;
* it is directly checkable: **the existing `fwd-tie <net>` must stay BIT-EXACT against the committed
  pre-change forward when fed a ones mask.** That is a free, strong gate that costs nothing to write.

Cost: N extra `tensor<Bxf32>` inputs and N broadcast-multiplies on the eval path. Negligible against
a conv stack, and worth measuring rather than assuming (§2m's transpose measurement is the template).

⚠ This DOES mean the forward artifacts change, contrary to the first-glance reading that only train
steps move. Trading that for the prefix audit is the right way round.

---

## 4. The op inventory — it is ONE op, and its VJP is itself

The check that made heavy-ball (§2k) and RMSProp (v1.2) one-op jobs, run again:

| piece | status |
|---|---|
| per-example broadcast scale, `fun i => mask (i / n) * x i` | ⛔ **does not exist.** `rowScaleF`/`layerScaleChF` scale by a per-**feature** vector (the LN/layer-scale affine) — the other axis. `scaleB` scales by a **literal**. None is reusable at a different reading |
| the shape it needs | ✅ **`batchMapAux`** (`StableHLO.lean:96`) — "batchMap with per-example auxiliary data", built in §2b for exactly this |
| the route | ✅ **`selectMidB`** (`:468`) is the precedent: own constructor riding the generic `.batched` tag, ~4-6 sites, no `Raw`/`Tok`/`toToks`/`parseStack` work |
| ⚠ why NOT a descriptor | §4's rule — a `BatchableOp` descriptor may carry only batch-INVARIANT data, and the mask is per-example. `selectMidB` is in this file precisely because the original scope got that wrong once (§2f) |
| the backward | **the same op on the cotangent**: `y = c ⊙ x ⇒ dx = c ⊙ dy`. No second emitter, no `*Grad` peer |
| the cert | a diagonal linear map. `diagBack s dy = fun i => s i * dy i` and `floatBridges_diagBack` already exist (`LinBackFloatBridge.lean:75/125`) for the float story; the VJP itself is the cheapest shape in the kit |

**So: one `SHlo` constructor, one `den`, one faithfulness theorem, one `has_vjp`, one
`TestBatchedEmitTie` case.** That is the whole render-side math. Tier E's "new layer family" is
wrong — nothing about the architecture changes; a residual branch gains a scale.

---

## 5. What it ACTUALLY costs: a new input class

This is where the work is, and none of it is proof work.

**5a. The driver.** Draw `sites × B` Bernoulli values per step and pass them in.
* no `F32.bernoulli` exists — `F32.heInit` / `F32.perturbUnit` are the nearest. One new primitive,
  or a uniform + threshold.
* ⚠ **it must be seeded from the global step**, like `augSeed := (ep*nb + bi + 1)`, or no run is
  reproducible and every gate that replays a step breaks.
* ⚠ **the masks must NOT go on the resident path.** They change every step, so they are per-step
  host→device traffic — `sites × B` floats (ViT: 24 × 32 = 768 floats, trivial) but they must be
  wired as ordinary inputs, not retained ones, or §2d.3's residency invariant is violated.

**5b. The shim's shard mask — and this is the one with a real trap.** Under data parallelism the
mask is **per-example, so it must be SHARDED like `x`**, not replicated like the parameters. The
shard mask is built per call site (`iree_lean_ffi.c:1022/1040`).

> ⚠⚠ **The duplicated-batch `*-dp-check` gates are STRUCTURALLY BLIND to getting this wrong.**
> They hand every replica the *same* rows, so a correctly-sharded mask and a wrongly-replicated one
> produce **identical** results — the harness passes bit-exact either way. This is the same class of
> hole §5 found for sharding itself (*"a duplicated-batch gate does NOT test that sharding
> SHARDS"*), and it is why `shard-check` exists. Any stochastic-depth DP render needs an
> **asymmetric-batch** gate, not the duplicated-batch one.
>
> ⚠ But `shard-check`'s construction is `DP([A|B]) = mean(single(A), single(B))`, which needs the
> gated slot linear in the gradient — fine for AdamW's `m`, **already known false for RMSProp's
> buffer** (2026-08-02). EfficientNet is the net that wants stochastic depth *and* RMSProp, so on
> that net **neither existing DP construction works as-is.** This is an open design question and it
> should be settled before the render lands, not after.

**5c. Blast radius — 26 train steps and 8 forwards across three nets.**

```
convnext, convnext_adam, convnext_adamdp, convnextin_adam, convnextin_adamdp
vit, vit_adam, vit_adam64, vit_adam128, vit_adamdp, vit_adamdp64,
  vit_adamdp32x4, vit_adamdp128x4, vitin_adam128, vitin_adamdp128x4
efficientnet, efficientnet_adam, efficientnet_adam128, efficientnet_adamdp,
  efficientnet_adamdp128, efficientnet_rms, efficientnetin_adam64, efficientnetin_adamdp64,
  efficientnetin_rms64, efficientnetin_rmsdp64
+ the 8 forwards (§3): convnext_fwd, convnextin_fwd, vit_fwd, vitin_fwd,
  efficientnet_fwd{,_eval}, efficientnetin_fwd{,_eval}
```

Every one changes **arity**, so every harness that supplies a parameter blob to them moves:
`fwd-tie`, `sgd-render-tie`, the AdamW ties, `rms-tie`, `*-dp-check`, `shard-check`,
`residency_gate*.sh`. None is a hard change — they build their inputs from the spec — but it is
breadth, and §2m's lesson applies exactly: **two arity routes must agree, and only pinning one is
how the mnv2 swap shipped a 160-param forward past a green tie.**

---

## 6. ⚠ Traps, ranked by how expensive they'd be

1. **The ramp denominator is a derived constant that depends on depth** — §2k's `α/K` bug in a new
   place, and it has the same signature: it compiles, it runs, it descends, and it trains a
   different objective. `keep_i` must be derived from the same block traversal that emits the sites,
   never a parallel hand-list (§2e's silent-slot rule), and gated by a byte-identical re-render at
   `dropPath = 0`.
2. **`dropPath = 0` must re-render every existing artifact byte-identically.** That is gate 1 in its
   strong form and it is available here for free — take it.
3. **ViT's two branches share one `keep_i` but need two independent masks.** Emitting one mask per
   *block* instead of per *site* is a plausible mistake that halves the noise and is invisible in
   any structural check.
4. **A skipped drop is not a dropped block.** EfficientNet's no-skip blocks consume a ramp index but
   never fire; rendering the op there anyway (at keep = 1) is harmless, but *shifting the ramp* to
   skip them is not.
5. **Nothing in the numeric ties will catch a wrong keep probability** — a wrong `keep_i` is a
   different function, but every tie compares the render against *itself* or against a peer built
   from the same constant. The gate has to be a **known answer**: at a chosen mask, the output is
   `branch · mask / keep`, computed on the host. The `rms-tie` / `r34-mom-tie` construction.

---

## 7. Gates this needs (all cheap, all in the house style)

| gate | what it establishes | state |
|---|---|---|
| **`dropPath = 0` re-renders all ~34 artifacts byte-identically** | the threading is inert (gate 1, strong form) | ✅ 2026-08-02 |
| **`fwd-tie <net>` BIT-EXACT vs the committed forward, ones mask** | §3's identity: eval really is the identity, on the *same* graph | ✅ **B1** below — ⚠ and it is BLIND TO PLACEMENT, see the finding |
| **known-answer tie**: emitted output vs host-computed `branch · mask / keep` | the keep ramp and the inversion are right — the ONE thing no self-tie can see | ✅ **gate A**, bit-exact 40/40 |
| **control**: perturb one `keep_i` | the known-answer gate fires; verify it goes red | ✅ `--break`, 35/40 |
| **control**: all-zero mask on one site | that branch's contribution vanishes ⇒ the site is where it is claimed to be | ✅ **gate B**, + a misplaced render that goes red |
| **prefix audit unchanged** | §3 held | ✅ 1953-line prefix, both scales |
| **asymmetric-batch DP gate** (§5b) | the mask is SHARDED, not replicated | ✅ **DONE 2026-08-02** — `lake build drop-shard-check`, and it found the defect §5b predicted. §7d |
| **residency gate unchanged** | masks stayed off the resident path | ✅ 2026-08-02 |

### ✅ 7e. **ConvNeXt-T — DONE 2026-08-03.** The recommendation in §8 executed

`ConvNeXtRenderB.lean` renders the 18 sites, one per block, on the residual branch (between
`layerScaleCh` and the `addVB`); the backward emits the same constructor on the cotangent between
`dy` and `layerScaleCh`. Six artifacts, all NEW — `convnext_{adamdrop,adamdpdrop}_train_step`,
`convnext_drop_fwd`, `convnextin_{adamwxclipdrop,adamdpwxclipdrop}_train_step`, `convnextin_drop_fwd` — so no
committed artifact moved (0 lines of diff with the writers forced).

| gate | result | its control |
|---|---|---|
| the drop-free chain is unchanged | `convnext-fwd-b-tie` still byte-identical, gradMap 180/180 | — |
| `verified_mlir/` diff, writers FORCED | **0 lines** on all 118 committed artifacts | — |
| **keep = 1 ⇒ `adamdrop` trains what `adam` trains** | **0 of 83,478,846** floats differ after 3 AdamW steps, against a **bit-exact** A-vs-A floor | real 0.1 ramp fires at norm-rel **0.399** (100% of coords); broken conv-VJP flip at **0.0343** |
| **the misplacement control** | `misplace_drop_sites.py` → **rc=1** at B2's collapse check | — |
| gate B (all-zero mask), `droppath-tie --net convnext` | floor 320/320, B1 320/320, **B3 0.259**, **B4 1.104** | the misplaced render, above |
| gate A (the op, `B ≠ n`) | bit-exact 40/40 | C1 1.29, C2 35/40 |
| **DP mask shard**, `drop-shard-check convnext` | ① **83,478,846/83,478,846** bit-identical, ②a `%loss` 4.5413 → 4.3645 MOVED | `DROP_FAULT=replicate` fires ① at **81,799,716** moving, rel 1.00005; `PJRT_DP_NO_MASK_SHARD=1` refuses on arity |
| prefix audit | `convnext_drop_fwd` ⊂ `convnext_adamdrop`, **1580 lines**; same for `convnextin` | — |

**Three findings that outlive the feature:**

1. ⚠⚠ **§7b's ones-mask blindness REPRODUCED ON A SECOND NET, and it is what makes the control
   necessary rather than tidy.** On the misplaced ConvNeXt render, B1 — the ones-mask identity
   against the drop-free `convnext_fwd` — passes **BIT-EXACT 320/320**, and the keep = 1 train-step
   gate would too. The gate that fires is B2's *collapse* check. Two nets, same conclusion: every
   endpoint gate is blind to placement, and only `s = 0` separates the two wirings.
2. ⚠⚠ **ONE PARAMETER GRADIENT READS THE COTANGENT AT THE DROP SITE, and it is silent.** LayerScale's
   γ gradient is `Σ (cot ⊙ p)` at the cotangent of the LayerScale *output* — which the drop scales.
   Every other block gradient descends from `cot_p` and inherits the scale; `%…lg` is the one that
   would be computed against an undropped cotangent. It type-checks, trains and descends: **18 of
   180 gradients wrong by a per-example factor, on the parameter stochastic depth is about.** Found
   by tracing the operands, not by any gate — the keep = 1 tie cannot see it (at `s = 1` the two
   cotangents are equal) and the placement controls do not touch the backward.
3. ⚠⚠ **AN LN NET HAS NO BATCH STATISTICS, SO §7d's ANTI-VACUITY HALF SHRINKS TO ONE SCALAR.**
   EfficientNet witnesses "replica 0 received a different mask" with 42,016 replica-0-local floats;
   ConvNeXt has exactly one, `%loss`. ① is unchanged and still discriminating (its control moves
   81.8M outputs), but the check pulling the other way is now a single float plus the harness's
   mask-halves-differ refusal. The harness **says so at run time** rather than reporting the BN
   line as `0/0 MOVED` and letting a reader take it for a pass.

### ✅ 7a. The interior gates — DONE 2026-08-02, `lake build droppath-tie`

The six gates run when the feature landed all pin an **endpoint**: `dropPath = 0` is inert, keep = 1
is bit-identical to AdamW, and `TestDropPathRamp` pins the ramp across the driver/renderer seam.
None of them says what a scale strictly *between* those endpoints does, and none of them can —
every existing tie compares the render against itself or against a peer built from the SAME
constants (§6.5). `tests/TestDropPathTie.lean` closes both, and the split is deliberate: **gate A is
site-local, gate B is whole-net, and neither subsumes the other.**

Run under `scripts/det_shim.sh` (gate B compares two HLO programs; §2d.3's Finding 1 is
ROCm-specific). The A-vs-A floor is measured first and read out, per the 2026-08-02 finding.

**▶ Gate A — the known answer.** `dropPathB` through the *same* `pretty` emitter at `B 8 × n 5`
(`n ≠ B` on purpose, so a wrong broadcast axis is a type error rather than a different function),
against a host-computed `s[j]·x[j,i]`:

| check | result |
|---|---|
| emitted vs host closed form | **BIT-EXACT 40/40**, `\|ref\|max` 1.452 |
| `dropPath_ones_id` on device | the `s = 1` row equals `x`, **5/5** |
| `dropPath_zeros_zero` on device | the `s = 0` row is zero, **5/5** |
| ⚠ CONTROL — the descriptor bug (every example gets `s[0]`) | **fires**, rel 1.29, only 5/40 exact (row 0, correctly) |
| ⚠ CONTROL `--break` — one scale perturbed 1% | **fires**, 35/40 exact — exactly that row's 5 coordinates |

**Bit-exact is the bar rather than a tolerance, and that is an argument**: both operands are f32, so
their exact product needs ≤48 mantissa bits and is EXACT in the f64 the host multiplies in; rounding
that to f32 is by definition what f32 multiplication returns. No double rounding, so any difference
at all is a different function. The mask is read back out of the buffer the device receives, never
from the literal, so both sides see the identical f32.

⚠ **The mask spans the values the driver ACTUALLY supplies.** `F32.dropScales` emits
`bernoulli(keep_i)/keep_i`, i.e. `0` or `1/keep_i` — and every `1/keep_i` is **greater than 1**
(1.027397 / 1.136364 / 1.229508 at sites 0/4/8). A gate written only over `(0,1)`, which is how §6.5
and the handoff both phrase it, would test a range the feature never uses. The mask carries those
three, both endpoints, and three interior values.

**▶ Gate B — the all-zero-mask control**, at `@efficientnet_drop_fwd` and `@…_fwd_eval`:

| | train | eval |
|---|---|---|
| FLOOR — same artifact, two compiles | **bit-exact 320/320** | **320/320** |
| **B1** ones mask vs the drop-free `@efficientnet_fwd` | **BIT-EXACT 320/320** | **320/320** |
| **B2** zero at site 0 (block 2) vs ones | rel 0.661 | rel 0.153 |
| **B3** with site 0 zeroed, ALSO zero site 8 (block 14) | rel 0.305, 0/320 exact | rel 0.448, 0/320 |
| **B4** all 9 zeroed, two different `x` | rel 0.631, \|logits\|max 0.874 | rel 0.0131, \|max\| 0.744 |

B3 and B4 are the load-bearing pair and B2 is not: **B2 is satisfied by both placements.** Under
`out = s ⊙ (branch + x)` the activation is already identically zero at the upstream site, so no
later mask can matter (B3 would be bit-exact) and every later layer is a function of zero, so the
logits stop depending on `x` (B4 would collapse). Under the correct `out = s ⊙ branch + x` a zeroed
site leaves the block an **identity** and the net intact.

**▶ The control that makes gate B mean anything: a misplaced render.** `--cand` takes a candidate,
the `vit-dp-check` lesson (§2j — that harness hardcoded both paths, so its bit-exact PASS was
unfalsifiable until an argument was added). The control moves all 9 drop sites from the residual
branch onto the block output — two lines swapped per site, **same SSA names, same order, same types,
same line count**, so arity, op counts and the prefix audit are all unchanged:

```
correct     %B = multiply %mask, %branch ; %C = add %B, %skip        s·branch + x
misplaced   %B = add %branch, %skip      ; %C = multiply %mask, %B   s·(branch + x)
```

It goes red at B2's collapse check, `|logits|max 0.000000`, rc=1. The rewrite is
`scripts/misplace_drop_sites.py` — committed rather than ad-hoc, because a control that has to be
retyped is a control nobody re-runs.

### ⚠⚠ 7b. THE FINDING: the ones-mask identity gate is STRUCTURALLY BLIND TO PLACEMENT

**§7's own second row — *"`fwd-tie <net>` BIT-EXACT vs the committed forward, ones mask"* — passes
BIT-EXACT on the misplaced render.** 320 of 320, against a bit-exact floor. And it must:
`1 ⊙ (branch + x) = branch + x`, exactly, so at an all-ones mask the two placements are the *same
function*. §3 proposes that gate as *"a free, strong gate that costs nothing to write"* and it is —
but it gates the **identity at keep = 1**, not the wiring, and this doc read it as covering both.

The keep = 1 train-step gate the feature shipped with (bit-identical to AdamW, 0 of 4,020,358) is
the same statement one level up and inherits the same blindness. So **every endpoint gate in the
original set is blind to the placement**, which is exactly why the interior ones were owed:

> An identity gate at `s = 1` cannot see where `s` is applied. Only a gate at `s ≠ 1` can, and
> the sharpest one is `s = 0` — the other endpoint, where the two placements differ maximally.

Generalisable, and it is §5b's shape one axis over: *a gate whose input makes the intervention
inert cannot test the intervention.* The duplicated-batch DP gates are blind to sharding for the
same reason; this is that, at `keep = 1` instead of at a replicated batch.

### ⚠ 7c. A control proves the gate goes red FOR THE REASON IT CLAIMS, not merely that it goes red

The first misplaced-render run fired — at the wrong check, with a message naming the wrong cause.
`cmpBufs` normalised by the *first* buffer's magnitude, the misplacement drove that buffer to
identically zero, the denominator went to zero, and `rel` returned `0.0`. A **total collapse of the
logits** was therefore reported as *"zeroing site 0 barely moves the logits — the mask input is not
reaching that site"*: right verdict, wrong diagnosis, and one that would have sent the next reader
to the driver's mask wiring instead of to the render. Fixed by normalising over both buffers (what
`fwd-tie` already does) and by checking the collapse explicitly, first, so it reports itself.

Worth the ink because the harness was green on the real artifact throughout — the defect existed
only on the failure path, which is the half a passing run never exercises.

---

## 8. Honest cost, and the recommendation

> ⚠⚠ **STATUS 2026-08-03 — the blocker named in §5 is GONE, and the cost table below is now
> pessimistic for ConvNeXt.** This spec was written when ConvNeXt rendered at the PER-EXAMPLE index,
> where a per-example mask is not expressible at all (a node denotes ONE example; `pretty B` lifts
> it, so the node cannot see `j` — handoff §0.2 ▶2). ConvNeXt is now **fully re-instantiated at the
> batched index `N := B`** (`LeanMlir/Proofs/Codegen/ConvNeXtRenderB.lean`), forward byte-identical
> to `convnext_fwd.mlir` and the whole train step tied against `convnext_adam_train_step.mlir`.
> So the rows below that assume a per-example chain no longer apply, and the recommendation —
> **ConvNeXt only, single-device** — stands and is now cheap. **Handoff §0.10 is the work order**:
> what exists, the 18 sites, the gates in order, and the two decisions to take first.
>
> ⚠ Two rows of the table below are also settled since: the **DP sharding gate** is built and its
> construction is optimizer-agnostic (§7d), and **the interior gates** are done (§7a-c) — including
> the finding that every ENDPOINT gate was blind to placement, which is the trap to respect when
> siting the 18 drops.


| phase | cost |
|---|---|
| the op + cert + emit tie | **small** — one `selectMidB`-shaped ctor, VJP is itself |
| the ramp + site traversal in ONE renderer (ConvNeXt) | small-medium; the derived-constant care is most of it |
| driver: Bernoulli, seeding, wiring as non-resident inputs | medium |
| the other two renderers | medium (ViT's two-sites-per-block; enet's skip guard) |
| **the DP sharding gate** | ⚠ **unknown — the construction does not exist for the RMSProp nets** |
| re-running every arity-sensitive harness on 3 nets | breadth, not depth |

> ✅ **DONE 2026-08-03 — and it went past "single-device".** §7e is the record: 18 sites, five
> artifacts, seven gates with five controls, including the DP mask-shard gate at two replicas. The
> cost came in at what §8 predicted for the render half and **less** for the plumbing half, because
> the driver, the op, the cert and the shard gate were all already built. What was NOT predicted:
> the LayerScale-γ cotangent (§7e finding 2), which no gate in this table would have caught.

**Recommendation: do ConvNeXt only, single-device, and stop there for a decision point.** It is the
net with one site per block (no ViT branch-splitting, no enet skip guard), it already has the
tightest DP agreement in the repo, and — unlike EfficientNet — it does **not** also need RMSProp, so
§5b's open question can be deferred rather than solved first. That yields the op, the cert, the
derived ramp, the known-answer gate and a real answer on cost, against one renderer and four
artifacts instead of twenty-six.

⚠ **And know what it buys before spending it.** ConvNeXt's reference number is **75.93%, which is
the EMA shadow's** — so stochastic depth alone does not make that pair comparable; EMA is still
required for the headline. On the verified side ConvNeXt's Imagenette run is *overfitting* already
(train loss 0.502, val flat from epoch 40, ~28M params on 9,469 images — §2o Part B), so a
regulariser is directionally right there and its effect should be *measurable* at Imagenette scale,
which makes it cheap to evaluate. That is the argument for doing it; it is not an argument that it
closes the ImageNet gap on its own.


---

### ✅ 7d. THE DP SHARD GATE — DONE 2026-08-02, and §5b's prediction was CORRECT

§5b left this as *"an open design question [that] should be settled before the render lands, not
after"*, on the grounds that the duplicated-batch gates are blind to it and `shard-check` needs a
linearity EfficientNet's RMSProp variant does not have. Both halves of that turned out right, and
the answer was a construction neither gate uses.

#### ⛔ The defect was REAL and it was in the shim, before any DP drop render existed

The masks ride in the **parameter blob** (`dropShapes` is appended to `adamShapes`), and the DP
shim's rule is *"`x` and the labels shard, everything between them replicates"*. So every replica
would have received replica 0's mask and applied it to its own rows. **Two halves, and both were
needed for it to be silent:**

1. the shard flag was never set on the mask inputs;
2. `dropShapes` and `dropScales` were sized at the **per-device** batch, so the buffer type-checked
   as a replicated input and nothing complained.

Fixed by `pjrt_ffi_invoke_f32_dp2` (a **renamed** entry taking `n_shard_tail`, per §4's rule — a
stale `.so` against a new binary shifts every argument, which is garbage rather than a link error)
plus sizing the mask buffer at the **global** batch. ⚠ Once the buffer is global, replication is not
expressible: the shim refuses on arity. **The sizing fix is what turns the flag from a correctness
question into a type-checked one.**

#### ▶ The construction — optimizer-agnostic, which is what §5b said did not exist

Duplicate the **data** and make only the **mask** asymmetric, then **swap the halves**:

> run 1: `[x|x]` with `[m₀|m₁]`  ·  run 2: `[x|x]` with `[m₁|m₀]`

| | run 1 | run 2 | swap-invariant? |
|---|---|---|---|
| sharded (correct) | `mean(g(x,m₀), g(x,m₁))` | `mean(g(x,m₁), g(x,m₀))` | **YES, bit-identical** |
| replicated (defect) | `g(x,m₀)` | `g(x,m₁)` | no |

Bit-identity is an **argument**: at two replicas the collective is `(a+b)/2` and IEEE-754 addition
is **commutative**. ⚠ Commutativity, *not* associativity — above two replicas the reduction is a
tree whose order a permutation changes, so the harness refuses at `DROP_REPLICAS ≠ 2`.
**No linearity is required anywhere**, because it compares two runs of the SAME graph rather than a
device answer against a host one — so it transfers to `emarmsdrop` unchanged.

#### ⚠⚠ AND IT NEEDS A SECOND CHECK PULLING THE OTHER WAY

Swap-invariance alone is satisfied by a mask that **reaches nothing** — the ones-mask blindness of
§7b one level up. What witnesses that replica 0 really received a *different* mask is the output
that is **replica-0-LOCAL**: the batch statistics (never all-reduced, read from replica 0 only) and
`%loss` (report-only, computed per replica). Those must **MOVE**.

| | measured |
|---|---|
| ① all-reduced `θ'/m'/v'` BIT-IDENTICAL under the swap | **12,061,074 / 12,061,074** |
| ②a `%loss` MOVED (replica-0-local) | 2.430799 → 2.428901 |
| ② batch statistics MOVED | **38,966 / 42,016** |
| mask halves distinguishable (anti-vacuity refusal) | 50 of 288 slots differ |
| ⚠ CONTROL `DROP_FAULT=replicate` — the pre-fix world reconstructed | ① fires, **12,044,574 / 12,061,074** move, rel 0.691, rc=1 |
| ⚠ CONTROL `PJRT_DP_NO_MASK_SHARD=1` — flag off, buffer global | the shim **REFUSES on arity**, rc=1 |

⚠ Both controls pass ② — replica 0 still sees a different mask between the two runs — so **① is the
discriminating gate and ② is the anti-vacuity one.** Neither is evidence alone.

#### Three findings

1. ⚠⚠ **A SHARDED INPUT THAT IS ALSO AN OUTPUT NEEDS THE PER-REPLICA SIZE ON THE WAY BACK.** The
   drop masks are the first: `x` and the labels are inputs only, so the DP output walk had always
   been able to take the declared size for granted. It **caught itself** —
   `output 740 size mismatch: graph 128 bytes, caller 256` — rather than reading past the end. That
   is the shim's G4 guard earning its keep a third time (it also caught the missing BN arity when
   `shard-check` was generalised).
2. ⚠ **`%loss` IS REPLICA-LOCAL, NOT ALL-REDUCED, AND THE RETURN LAYOUT HIDES IT.** The layout is
   `θ' ++ m' ++ v' ++ [%loss, %bc1, %bc2] ++ bnstats`, so any range reaching the scalars sweeps
   `%loss` up with them. Counting it as all-reduced made ① read **one** differing output of
   12,061,077 — and *one* is the tell, because a wiring defect moves millions. It belongs with the
   batch statistics, and it is extra evidence rather than noise.
3. ⚠ **§5b'S FRAMING WAS THE OBSTACLE.** It asked which of the two EXISTING constructions to use,
   and both answers were "neither". Dropping the requirement that the gate compare against a
   host-computable or single-device answer — comparing two DP runs of the same graph instead —
   removes the linearity constraint entirely. *When two known constructions both fail, check
   whether the property they share is actually required.*
