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
convnext, convnext_adam, convnext_adamdp, cnxin_adam, cnxin_adamdp
vit, vit_adam, vit_adam64, vit_adam128, vit_adamdp, vit_adamdp64,
  vit_adamdp32x4, vit_adamdp128x4, vitin_adam128, vitin_adamdp128x4
efficientnet, efficientnet_adam, efficientnet_adam128, efficientnet_adamdp,
  efficientnet_adamdp128, efficientnet_rms, enetin_adam64, enetin_adamdp64,
  enetin_rms64, enetin_rmsdp64
+ the 8 forwards (§3): convnext_fwd, cnxin_fwd, vit_fwd, vitin_fwd,
  efficientnet_fwd{,_eval}, enetin_fwd{,_eval}
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

| gate | what it establishes |
|---|---|
| **`dropPath = 0` re-renders all ~34 artifacts byte-identically** | the threading is inert (gate 1, strong form) |
| **`fwd-tie <net>` BIT-EXACT vs the committed forward, ones mask** | §3's identity: eval really is the identity, on the *same* graph |
| **known-answer tie**: emitted output vs host-computed `branch · mask / keep` | the keep ramp and the inversion are right — the ONE thing no self-tie can see |
| **control**: perturb one `keep_i` | the known-answer gate fires; verify it goes red |
| **control**: all-zero mask on one site | that branch's contribution vanishes ⇒ the site is where it is claimed to be |
| **prefix audit unchanged** | §3 held |
| **asymmetric-batch DP gate** (§5b) | the mask is SHARDED, not replicated — ⚠ needs a construction that does not exist yet for the RMSProp nets |
| **residency gate unchanged** | masks stayed off the resident path |

---

## 8. Honest cost, and the recommendation

| phase | cost |
|---|---|
| the op + cert + emit tie | **small** — one `selectMidB`-shaped ctor, VJP is itself |
| the ramp + site traversal in ONE renderer (ConvNeXt) | small-medium; the derived-constant care is most of it |
| driver: Bernoulli, seeding, wiring as non-resident inputs | medium |
| the other two renderers | medium (ViT's two-sites-per-block; enet's skip guard) |
| **the DP sharding gate** | ⚠ **unknown — the construction does not exist for the RMSProp nets** |
| re-running every arity-sensitive harness on 3 nets | breadth, not depth |

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
