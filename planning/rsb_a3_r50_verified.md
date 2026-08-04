# rsb_a3_r50_verified.md — RSB-A3 ResNet-50 as the SIXTH verified net

**Scoped 2026-08-03 by measurement, not estimate** — every number below was read off the kit or the
reference, the §2l/§2m way. **Partly built 2026-08-03/04**; §0a is the state.

**The ask**: bring RSB-A3 ResNet-50 into the verified/MLIR path, paired against the JAX reference.

⚠ **The reference number is currently VOID and that is deliberate.** `rsb-faithful` measured
**76.66% top-1 / 93.03% top-5 @ ep100** (`planning/rsb_a2_resnet50.md`) against paper RSB-A3 78.1%
— but the stem-pool fix (§4b) moved the JAX net, so that number belongs to a net the repo no longer
emits. It has to be re-run. Same for JAX R34/ImageNet's 72.1% / 90.7%.

---

## 0a. ▶ START HERE — state as of 2026-08-04

**Nothing about R50 itself is built.** What exists is the layout skeleton, and — from a stem-pool
deviation found while scoping it (§4b) — a finished max-pool VJP witness and a fixed JAX emitter.

| | state | where |
|---|---|---|
| `VLayer.bottleneckStage` + layout helpers | ✅ built, 7 `#guard`s, controls verified to redden | `VerifiedSpec.lean` |
| `resnet50Verified` / `resnet50ImagenetVerified` | ✅ **LAYOUT ONLY** — params land on **25,557,032** exactly | `VerifiedNets.lean` |
| ⭐ `maxPool3s2` **full VJP witness** | ✅ 438 lines, 21 decls, **0 `sorry`**, every theorem 3-axiom | `Proofs/Architectures/MaxPool3s2.lean` |
| ⭐ JAX `max_pool2d` → symmetric padding | ✅ landed + gated both ways (§4b) | `jax/Jax/Codegen.lean` |
| the bottleneck architecture (3 block VJPs) | ⛔ not started | Phase 1 |
| `ResNet50RenderB.lean` | ⛔ not started | Phase 2 |
| the pool's **codegen** | ⛔ not started — site map in §4b | — |

**`verified_mlir/` is 0 lines of diff.** Nothing emits yet, on either thread.

### ⚠⚠ THE ONE THING THAT BITES IF YOU DO NOTHING ELSE

**JAX now pools the paper's 3×3 window; the verified path still pools 2×2.** The two disagree at
the stem *more* than before this work started. That is the cost of fixing the cheaper side first,
and it closes when the pool codegen + the 14-artifact re-render land.

⛔ **Do not start an R34 verified-vs-JAX pair run in this window** — it would compare two nets that
differ at the stem. The R34/ImageNet 30-epoch run (~26 h, `xla_pjrt_handoff.md` §0.3) is otherwise
ready and was the queued next run.

### ▶ Suggested order for a fresh session

1. **The pool codegen** (§4b's site map) — 10 sites across two forms, then the R34 renderer swap.
   Unblocks everything and closes the inconsistency window.
2. **Re-render the 14 `resnet34*`/`resnet34in*` artifacts**, re-run R34's ties / `dp-check` /
   `shard-check` / residency, re-run the **90.06%** Imagenette baseline (~1h03m).
3. **Re-run the voided JAX numbers** (RSB-A3, R34/ImageNet).
4. **Then** Phase 1 — the bottleneck block VJPs. It is CPU proof work, so §0.1 of the handoff says
   it pairs well with a long run burning in the background.

---

## 0. ▶ THE ONE-PARAGRAPH VERSION

**The architecture is nearly free and the recipe is not.** Every SHlo op a bottleneck block needs
already exists and is kernel-generic, so the render costs **zero new ops** — the fifth time reading a
scoped op family against existing ops at their other readings has collapsed it (§2k heavy-ball,
RMSProp, EMA ×3, dropPath's VJP, ViT's `matmulF`). The verified param count lands on the reference's
**25,557,032 exactly, with no adjustment**, because §2m already dropped conv biases repo-wide. What
is genuinely new is **three block forms on the proof side** (not two — see §2.1), **BCE**
(~1 descriptor), **LAMB** (2–3 ops), and **one blocker that is not a code question**: the verified
driver has **no gradient accumulation**, and the 76.66% (⚠ now VOID — §4b) depends on effective
bs2048 obtained through it. Phases 1–3 get a verified R50 that exists, trains and is gated. Phase 4
is the RSB-A3 signature.

---

## 1. WHAT WAS MEASURED — the collapses, and why they are collapses

| scoped as | measured | evidence |
|---|---|---|
| "bottleneck conv ops" | ⭐ **ZERO new SHlo ops** | `SHlo.conv`/`convStrided` are `{ic oc h w kH kW}` — kernel-generic. 1×1 and 3×3 are the *same op at different indices* |
| "the 1×1 convs are a padding risk" (§2f-bis's even-kernel scar) | ⭐ **already shipped and gated, BOTH cases** | 1×1 **stride-1**: `mobilenetv2_fwd` carries **68** 1×1-kernel tensors and `efficientnet_fwd` **64**, in committed bit-exact-tied artifacts. 1×1 **strided**: R34's six, from §2l. The odd-kernel side condition `2·((k−1)/2)+1 = k` is discharged at k=1 by R34's projection weight-gradient already shipping |
| "R34's render lacks bottleneck pieces" | **its op set IS the bottleneck's** | `ResNet34RenderB.lean` uses `.conv`, `.convStrided`, `.bnBatchF`, `.relu`, `.addVB` + every grad/back peer. A bottleneck is those ops, three convs deep instead of two |
| "param count will need reconciling" | ⭐ **exact on the first try** | conv 23,454,912 + BN 53,120 + fc 2,049,000 = **25,557,032** = the reference's own reported count = torchvision's. **Because the verified world carries no conv biases** (§2m) and torchvision's R50 carries none either — the two conventions already agree |
| "BCE is a new loss family" | **~1 descriptor** | BCE-with-logits cotangent is `σ(z) − t`, i.e. CE's `softmax(z) − onehot` with one op swapped. `sigmoidF` exists with a **global, hypothesis-free** `sigmoid_has_vjp`; `subB`/`divConstB` exist. Missing: `BatchableOp.sigmoid` — pointwise and batch-invariant, so a **legal descriptor**, 4 sites |
| "LAMB is a new optimizer family" | **2–3 ops, shapes already built** | LAMB's trust ratio is `min(‖θ‖ / ‖r + wd·θ‖, clip)`. GradClip (§0.8) already built sum-of-squares-reduce + scale-by-**runtime** scalar. Same shapes at **per-parameter** scope instead of global. ⚠ Estimate, not a measurement — see §2.3 |
| "53 BN layers" | **confirmed, and it decomposes** | 1 stem + 16 blocks×3 + **4** shortcuts = 53. All four stages get a shortcut (stage 1 changes 64→256), unlike R34's three |

**Render size, extrapolated (labelled as such):** `resnet34in_mom256_train_step` is **5,321 lines /
107 `stablehlo.convolution`** at 36 BN layers. R50 at 53 BN layers ⇒ ~**8,000 lines**, between R34's
5.3k and ConvNeXt's 11.6k. Nothing here is a new scale.

**Param tensors: 161** (53 W + 53 γ + 53 β + fc W + fc b), against R34's 146. So the DP render emits
161 collectives and `[θ|m|v]` is **307 MB** — comparable to R34/ImageNet's ~255 MB, i.e. residency
(§2d.3) and the 4-GPU numbers transfer without re-deriving.

*(The param arithmetic above is computed, not hand-derived: conv 23,454,912 + BN 53,120 + fc
2,049,000 = 25,557,032, landing on the reference's own reported count to the unit.)*

---

## 2. WHAT IS GENUINELY NEW

### 2.1 ⚠⚠ THREE block forms, not two — and the third is the one a first look misses

R34 has two: `rblkPC` (identity) and `rblkPStridedPC` (downsample + projection). R50 needs **three**,
and the extra one is *not* a bottleneck-vs-basic difference:

| form | where | shape | exists? |
|---|---|---|---|
| **identity** bottleneck | 12 blocks | `Vec (c*h*w) → Vec (c*h*w)`, skip = id | new (3-conv peer of `rblkPC`) |
| **strided projection** | stages 2/3/4 block 0 | `Vec (ic*(2h)*(2w)) → Vec (oc*h*w)` | new (3-conv peer of `rblkPStridedPC`) |
| ⭐ **stride-1 projection** | **stage 1 block 0 ONLY** | `Vec (ic*h*w) → Vec (oc*h*w)` — channels change, **spatial does not** | ⛔ **no analogue anywhere** |

R34 never needed the third because its stage 1 is `ic = oc = 64`, so the skip is the identity.
R50's stage 1 goes 64→256 at stride 1, so it carries a **1×1 stride-1 projection**.
`rblkPStridedPC` hardcodes `flatConvStride2` for the projection *and* bakes the `(2*h)`/`(2*w)`
input index, so it cannot express this — it is a different type, not a different argument.

⚠ **This is the kind of thing that type-checks its way into being noticed late.** Writing only the
two obvious forms and reaching for the strided one at stage 1 is a *shape* error, so it fails loudly
— but writing an identity skip there is a **silent** wrong net that trains and descends. Build the
stride-1 projection form first, deliberately.

### 2.2 ⭐ The stride is on the 3×3 — this is ResNet **v1.5**, measured not assumed

`jax/Jax/Codegen.lean:602` — `bottleneck_block_down` puts `stride=(stride,stride)` on
`params[idx+1]` (the **3×3**) and on the shortcut, with the leading 1×1 at stride 1. That is
torchvision / ResNet-v1.5, **not** He et al.'s original v1 (stride on the first 1×1).

⚠ **Getting this backwards compiles, trains, descends and is a different net** — §2k's heavy-ball-vs-
Nesterov trap exactly, one layer up. It is also worth +~0.5 pt of accuracy in the literature, so it
is not cosmetic. Record it in the spec blurb, per §2l's lesson that the sin was the blurb, not the
deviation.

### 2.3 LAMB — the one estimate in this doc that is NOT a measurement

Everything else above was read off the kit. LAMB was not: I read GradClip's op *shapes* and matched
them against LAMB's algebra, which is the same reasoning that called grad clip "Tier E" when it was
Tier D and RMSProp "an op family" when it was one op. **Cost it separately before starting, the §0.8
way**: read `Jax/Codegen.lean`'s emitted LAMB `train_step` and grep the kit, rather than reasoning
from the formula.

The specific thing to check: LAMB needs **two** per-parameter norms (`‖θ‖` and `‖r + wd·θ‖`) where
grad clip needed **one** global one. Whether that is `gradSumSqAccF` at a different scope or a new
constructor decides whether this is 2 ops or 5.

⚠ And note LAMB **breaks every gate that recovers `g` from `m'`** — `r34-mom-tie`, `rms-tie`,
`wdx-tie`, `shard-check` all use `m' = (1−β₁)·g` at `m = 0` as their oracle, and the trust ratio
rescales it per layer. §0.8 finding on grad clip, one optimizer over. Plan a variant that keeps LAMB
off in those harnesses, exactly as `wx` is.

---

## 3. ⛔⛔ THE BLOCKER, and it is not a code question

**The verified driver has NO gradient accumulation.** `grep gradAccum LeanMlir/VerifiedTrain.lean`
returns nothing. The JAX side has it (`planning/grad_accum.md`, GPU-validated) and **the 76.66%
depends on it**: `rsb-faithful` is 512 micro × 4 = effective bs2048, and the whole finding recorded
in `rsb_a2_resnet50.md` is that **LAMB at bs512 gives 40.8% instead of 78.1%** — it is a large-batch
optimizer run at a quarter of its design batch.

⚠ The 76.66% itself is VOID pending §4b's re-run, but **this finding is not**: it is about batch
size, not about the pool, so it survives the re-run and still shapes the plan.

So a verified R50 has three ways to a comparable number, and none is free:

| route | cost | what it buys |
|---|---|---|
| **grad accum in the verified driver** | driver work; the accumulate-then-step loop, plus a BN-regime decision (Ghost-BN, as the JAX side took) | the honest pair against 76.66% |
| **real bs2048** | ~80 GB of activations at 160px — **this box cannot** | removes Ghost-BN; the cleanest reproduction, ungated here |
| **accept bs512** | free | reproduces the **40.8%** regime. A valid *engineering* pair (same starved recipe both sides) and a useless *accuracy* one |

▶ **Recommendation: phase the recipe last and decide this before phase 4, not during it.** Phases
1–3 use the existing certified AdamW kit at bs32/bs64, where none of this arises, and produce a
verified R50 that exists, trains and is gated. That is real value even if phase 4 never runs.

⚠ **A second, smaller structural point.** RSB-A3 is train@160 / eval@224. The verified path bakes
resolution into the render, so that is two artifacts at different indices — which *works*, but
**the §2g prefix audit (`_fwd` ⊂ `_train_step`) cannot hold across them**. That audit is one of the
load-bearing structural gates (it caught R34 and mnv2 scoring nets they had not trained). Decide
explicitly: either a documented carve-out, or a 160-res `_fwd` for the audit plus a 224-res
`_fwd_eval` for scoring. ⚠ Do not let it silently degrade to "the audit doesn't cover R50".

---

## 4. THE PHASES

Ordered cheapest-first, each landing something gated on its own. Phases 1–3 are the committed core;
phase 4 is the RSB-A3 signature and is where §3's decision lands.

### Phase 0 — ⭐ mostly ALREADY DISCHARGED (see §1's second row)
§2l's ten-minute 1×1-padding check and the odd-kernel side condition at k=1 were both going to be
phase 0. **Neither is owed** — mnv2/EfficientNet ship 68/64 1×1-kernel tensors in bit-exact-tied
artifacts and R34's projection already carries a 1×1 strided weight gradient. §2f-bis's even-kernel
scar does not reach this net: every R50 kernel is 7, 3 or 1, all odd.

What remains of phase 0 is **one item: cost LAMB the §0.8 way** (§2.3) — read `Jax/Codegen.lean`'s
emitted LAMB `train_step` and grep the kit, rather than reasoning from the formula.

### Phase 1 — the bottleneck architecture (proof side)
The three forms of §2.1 + their `_diff`/`_has_vjp`, built by `vjp_comp` over the existing
`flatConv`/`flatConvStride2`/`bnPerChannelTensor3`/`relu`/`residual`/`residualProj` pieces — the same
construction `rblkPStridedPC` uses, one conv deeper. Then the stage/net chain and **rung E**.
*Gate*: 3-axiom clean on every new declaration; `AuditAxioms` grows and the coverage checker stays
green (§2n's lesson: a swap can silently drop a rung).

### Phase 2 — the render + AdamW, Imagenette scale
`Proofs/Codegen/ResNet50RenderB.lean` at `N := B` from the start (every net is there now — no
per-example detour, and stochastic depth stays expressible). `resnet50Verified` in `VerifiedNets.lean`
with the 53-entry `bnChannels`, and the `toSpecs == Layout.specs` `#guard` — **which has fired on
three nets and is the thing that stops the two hand-lists drifting**.
*Gates*: `verified_mlir/` 0 lines of diff with writers **forced** (§2n) · `fwd-tie resnet50` bit-exact
with a firing control · a numeric AdamW tie against… ⚠ **nothing, and that is the honest gap** —
there is no incumbent hand-written R50 render to tie against. The substitute is the layer-level VJP
oracle (`tests/vjp_oracle/run.sh`) plus a keep-1-style known-answer check; **say which one licensed
the swap**, per §5.

### Phase 3 — ImageNet scale + DP + residency
Three `#eval`s (`B := 64`, `nClasses := 1000`, `replicas := 4`) if the renderer is parameterised the
way R34's is. Param count must land on **25,557,032** — free, and it is the §2k check that caught two
different "ResNet-34"s.
*Gates*: `shard-check resnet50` (batch-BN net ⇒ the asymmetric-batch construction, not the
duplicated-batch one) · `residency_gate_all.sh` with the right fault mode · a 40-step smoke on 4 GPUs.
⚠ **And per §0.3's open row, render the DP artifact with the SAME regulariser set as the
single-device one** — ViT and EfficientNet are currently a regulariser behind on exactly this axis.

### Phase 4 — the RSB-A3 recipe
BCE (`BatchableOp.sigmoid` + the `σ(z) − t` cotangent + a `%loss` carve-out), LAMB, the 160/224
split, and §3's grad-accum decision.
*Gates*: BCE by a known-answer check against a host-computed `σ(z) − t` (the `dropout-tie` gate-A
shape) · LAMB by recovering the trust ratio per layer and requiring it **constant within a layer and
different across layers** — the `clip-tie` construction inverted, and the control is a *global* trust
ratio, which is what a naive port produces.

---

## 4b. ⭐ THE STEM POOL — found while scoping; JAX side + VJP witness LANDED 2026-08-03/04

**Every ResNet in He et al. pools 3×3/s2 after the stem conv. The verified path pools 2×2/s2.**
Found while placing R50's stem; it is **pre-existing on R34**, documented nowhere (not the spec
blurb, not `planning/`), and R50 would have inherited it. Output shape is identical (112→56) so
nothing ever failed — the *function* differs (overlapping vs tiling windows).

⚠⚠ **And the JAX reference was not the paper either.** It emitted
`reduce_window(…, (1,1,3,3), (1,1,2,2), 'SAME')`, and XLA's `SAME` on a 112→56 axis pads
`(low 0, high 1)` — window `i = [2i, 2i+2]` — where He et al./torchvision pad **symmetrically**,
`[2i−1, 2i+1]`. Measured on device at `n = 12`: `SAME` window maxima `[2,4,6,8,10,11]` against
symmetric `[1,3,5,7,9,11]`. **The two grids are offset by one input position.** So "make the
verified net match the reference" and "make it match the paper" were different targets, and only
checking the emit rather than the spec line surfaced it.

**Decision (Brett, 2026-08-03): paper-faithfulness. Fix BOTH sides.**

### ✅ What landed

| | state |
|---|---|
| ⭐ `Proofs/Architectures/MaxPool3s2.lean` — the **complete VJP witness** | **438 lines, 21 decls, 0 `sorry`, every theorem 3-axiom.** `maxPool3s2` → `Argmax` → `eq_at_max` → `LocalReindex` → `flat_hasFDerivAt` → `pdiv3_…_smooth` → `has_vjp_at3` → `maxPool3s2Flat_has_vjp_at` |
| `jax/Jax/Codegen.lean`'s `max_pool2d` | `'SAME'` → explicit symmetric `(k−1)//2`. **True A/B diff (via `git stash`, not a stale snapshot): the helper and nothing else** |
| ⚠ discrimination gate | R50 logits **move**, rel **0.075** — the change is real, not vacuous |
| ⭐ inertness gate | `mnist-cnn` (2×2, calls the same shared helper twice) **BIT-IDENTICAL** end to end. At `k = 2`, `(k−1)//2 = 0` ⇒ symmetric ≡ `SAME`, so **no cifar/mnist net moves** |
| R50 forward after the change | param count **25,557,032** exact, 53 BN buffers, shape `(2,1000)`, finite |

⚠ **A methodology note worth keeping**: the first A/B diff of the JAX change was WRONG and looked
plausible. The on-disk `generated_*.py` were stale — they predated the AUG_SEED determinism, shim
sharding and EMA-warmup codegen changes — so diffing against them showed four unrelated deltas as
if they were mine. **Diff against a regenerated baseline (`git stash` the change, rebuild, generate,
restore), never against whatever is on disk.**

⭐ **Four things came in under scope, and three are about the op's SHAPE rather than its size:**

1. **`-∞` padding needs no extended-reals type.** For a `max`, clamping the index is equivalent, and
   Nat's truncated subtraction *is* the low clamp — so the symmetric form needs no `min` at all
   (`win3RowInv_first_dup`). The SAME-padded version I wrote first *did* need one.
2. **The accumulating backward needs no new analytic argument.** This looked like the hard part:
   overlapping windows mean an input feeds up to 4 outputs, so the backward must sum where
   `maxPool2`'s merely looks up. But `HasVJPAt3.correct` already states the backward as a sum over
   *all* outputs, and `maxPool2` only *collapses* it to one term using disjointness. Here it
   collapses to ≤4. The `LocalReindex → reindexCLM → pdiv3` route is indifferent to the reindex
   being non-injective — `reindexCLM`'s adjoint sums over preimages by construction.
3. **`Finset.sup'` beats nested `max`.** `maxPool3s2_eq_at_max` is 3 lines where
   `maxPool2_eq_at_max` needs a 4-way `fin_cases` against `max (max _ _) (max _ _)` — which would
   have been **9 ways** at 3×3. Choosing `sup'` for the forward made window size stop mattering.
4. **`pdiv3_…_smooth` is SHORTER than its 2×2 peer.** `maxPool2`'s decodes the condition into
   `co = ci ∧ ho = winRow hi_in ∧ …`, valid only because each input has one owning window. With
   overlap no such decoding exists, so the condition stays the reindex equation — and
   `rw [reindexCLM_apply]` then closes the goal by `rfl`.

⚠ **The one place the overlap really costs something**: `MaxPool3s2Smooth` must be stated over
**positions**, not offsets — two offsets can name one cell (the clamped duplicate at the first
window), where the values are equal by construction and smoothness must say nothing. `maxPool2` has
no analogue because there distinct offsets always meant distinct positions. That distinction
propagates into the gap function and the domination step of `flat_hasFDerivAt`.

### ⛔ What this VOIDS, and it is not small

* **JAX R50 RSB-A3 `rsb-faithful` 76.66% / 93.03%** — the number this whole pair was anchored on.
* **JAX R34/ImageNet 72.1% / 90.7%** (90 ep).
* Every other 3×3-pool JAX number. ⚠ 2×2 nets are provably unaffected (gate above).

### ▶ NEXT: the codegen — site map MEASURED, so it is mechanical

~~1. The `HasVJPAt3` witness~~ ✅ **DONE 2026-08-04** (above). The "33 declarations to mirror"
estimate was right about the source region and pessimistic about the work — see the four collapses.

⭐ **And it is NOT §4's ten-site route, which this doc previously assumed.** `ResNet34RenderB` uses
exactly **two** forms for the pool (measured — `grep maxPool` on that file returns two lines), and
the forward is a legal `BatchableOp` **descriptor** because pooling is per-example and reduces no
batch axis. So:

| form | route | sites, with line numbers in `Proofs/Codegen/StableHLO.lean` |
|---|---|---|
| `BatchableOp.maxPool3s2` (forward) | descriptor | ctor `:182` · `denOp` `:1280` · den theorem `:1954` · skel/Raw `:3879` · emit `:5843` |
| `SHlo.maxPool3s2BackB` (backward) | own ctor on the generic `.batched` tag | ctor `:528` · `den` `:1559` · den theorem `:1957` · skel `:3952` · emit |

`denOp` maps to `maxPool3s2Flat c h w` — the bridge this session's file already provides, so the
`den` side is a one-liner and its `rfl` theorem should close immediately.

**The two emits are the only real content:**
* forward `reduce_window`: window `2 → 3`, add `padding = [[1,1],[1,1]]` on the spatial dims;
* backward `select_and_scatter`: same window/padding change — ⭐ and **nothing else**, because
  `select_and_scatter` already scatters with an **add** reduction, which is exactly the
  accumulation overlapping windows need. The emitter was always general enough.

⚠ Both descriptors and `.batched`-tag ctors route through the generic Raw/Tok machinery, so per
§0.2 increment 2 expect **~5 sites each, not 10** — but *verify* rather than assume, and remember
`lake build Proofs Certs Codegen` (not bare `lake build`) is what checks the parser round-trip.

**Then:**
2. `ResNet34RenderB` swap (2 lines — `:343` forward, `:402` backward) → **14 artifacts re-render**
   (`resnet34*`, `resnet34in*`). ⚠ Gate 1 becomes load-bearing here for the first time: force the
   writers with `lake env lean` (§2n's vacuous-green trap), because until now `verified_mlir/`'s
   0 diff has been inert-by-construction, not evidence.
3. Re-run R34's ties, `dp-check`, `shard-check`, residency; re-run the **90.06%** Imagenette
   baseline (~1h03m); re-run the voided JAX numbers.
4. ⚠ **The R50 spec must adopt the new pool too** — `resnet50Verified`/`resnet50ImagenetVerified`
   currently say `.maxPool 2 2` with the deviation documented in their docstrings. Both those
   notes come out when the op lands.

⚠ Until 1–2 land, **verified and JAX disagree at the stem pool** — see §0a.

## 5. CLAIM CEILING

Unchanged from §5 of the handoff, plus two R50-specific ones:

* **There is no incumbent render to tie against**, so phase 2's swap is licensed differently from
  every other net's. Say so at the swap; do not describe it as "tied" in the §2b sense.
* **A verified-vs-JAX R50 accuracy comparison is only meaningful if both sides are in the same batch
  regime** (§3). At bs512 both sides land near 40.8% and the pair says something about the
  *recipe*, not the *port*. State the regime with the number, always.
* ⚠ **"Paper-faithful stem pool" is a claim about the WINDOW, not about the whole stem.** What was
  fixed is 2×2-non-overlapping → 3×3-symmetric. It says nothing about the rest of the net, and it
  does not make the R34 or R50 renders paper-faithful overall — §2.1's three block forms and §2.2's
  v1.5 stride are separate claims with separate evidence.
* ⚠ **No accuracy number anywhere may currently be quoted against a JAX reference**, because the
  references were re-emitted and not re-run (§4b). That includes numbers this doc cites.

## 6. COST

**The stem-pool sub-thread (§4b), which was not in the original estimate at all**: the VJP witness
took ~half a session against a "33 declarations" estimate — the four collapses are why. What is
left of it (codegen + re-render + re-gate) is **~1 session of work plus ~1h03m of GPU**, and the
JAX re-runs on top of that.

**Phases 1–3**: comparable to ConvNeXt's channel-LN thread (§2m/§2n/§2o) — call it **3–5 sessions**,
and that is an estimate anchored on a net whose op cost collapsed the same way. **Phase 4** is
**1–2** for BCE + LAMB, plus whatever §3's grad-accum decision turns out to be.

⚠ Every figure in this section is an estimate; §1's table and §4b's gates are the measured parts.
⭐ Calibration for whoever plans off it: **three of this doc's own estimates came in low-side-wrong
in the same direction** — the op cost (measured: zero new SHlo ops), Phase 0 (measured: already
discharged), and the pool's ten-site route (measured: two forms, ~5 sites each). The pattern is the
repo's own: *reading a scoped family against existing ops at their other readings collapses it.*
Do that before believing any number here.

## 7. RELATED

* `planning/rsb_a2_resnet50.md` — the JAX side: phases 1–5, the fidelity ledger, the (now VOID)
  76.66%, and the bs512→40.8% saga that §3 rests on. ⚠ **Its fidelity ledger needs a stem-pool
  row** — that doc still reads as if the JAX pool were settled.
* `planning/resnet50_imagenet.md` — the original R50 decision + compute estimates.
* `planning/xla_pjrt_handoff.md` §2l — the 1×1 projection, the odd-kernel side condition, and the
  blast-radius method this doc copies. §2m — the conv-bias drop that makes §1's param count exact.
  §0.1 — why CPU proof work pairs with a long run. ⚠ **§0.3's owed-list needs a stem-pool row too.**
* `planning/grad_accum.md` — the JAX-side grad accum §3 would need a verified peer of.
* `LeanMlir/Proofs/Architectures/MaxPool3s2.lean` — the witness itself; its header carries the
  padding argument and the `SAME`-vs-symmetric measurement in full.
* `LeanMlir/Proofs/Architectures/CNN.lean` 1780–2480 — `maxPool2`'s peer machinery, which the
  witness mirrors and which is the place to look when the codegen's `den` theorems misbehave.
