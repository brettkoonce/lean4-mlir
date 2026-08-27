# verified_side_quest_counterparts.md — a verified peer for every side quest, and one naming scheme for all of them

**Opened 2026-08-27.** Three things that are really one thing:

1. **Appendix B Track 4 lists eight rows and the book has six side quests**, none of which appear
   there. MNv4 is in the main table where it does not belong, and the four S/B variants and the two
   RSB tiers are nowhere.
2. **Four namespaces name the same run and none of them agree** — the Lean exe, the MLIR slug +
   `LEAN_MLIR_VARIANT`, `LEAN_MLIR_RECIPE`, and the job config. `r34-imagenet-4gpu` and
   `r50-imagenet-4gpu` are the same shape of name for two different recipes.
3. **Most side quests have no verified counterpart**, and the reasons differ per net — one is four
   `#eval` lines, one is a renderer feature, one is a whole project.

▶ Sibling docs, read first: `planning/blueprint_lowerer_pattern.md` (the §5.7 chapter shape, and
why the verified path is the artifact and JAX the oracle) and `planning/chapter_makeover.md` §5
(the verification discipline). This doc is about the *targets*; those are about the *prose*.

---

## 0. START HERE — the state this doc was written against

Everything below is a survey of the tree at the commit that added this file. Nothing in §4 has
been built. What HAS landed, and why the book now looks the way it does:

* **Every side quest's phase-2 cost is measured and printed.** `sec:r50_a2_a1_cost`,
  `sec:convnext_sb` and `sec:vit_sb` each carry a table of ms/step, min/epoch and full-schedule
  wall clock on four 4060 Ti, bf16. Raw logs, method and the probe's own caveats are in
  `runs/2026-08-27-jax-sb-tier-step-probe/`. **Do not re-measure these to start** — they reproduce
  the 2026-07-25 figures to within a few percent on a newer JAX, and the numbers are not what is
  blocking anything.
* **`jax/scripts/step_probe.py` was repaired** to run against current emits (EMA signature, absent
  EMA, FixRes resolution). Any new phase-2 pricing goes through it.
* ⚠ **Regenerate before probing.** All four S/B `generated_*.py` on disk were a month stale when
  this survey ran, and a month of augmentation work had landed. `lake exe <net>-imagenet <recipe>`
  in `jax/` first, every time.

✅ **§6's ITEM 1 LANDED 2026-08-27**, in five commits: the ViT-B docstring fix, the five exe
renames, the job renames, the `RECIPE=` precheck, and the Track 4 restructure. §2c, §3 and §6
below are marked up with what happened — including **three places the survey contradicted this
plan**, each flagged ⚠⚠ where it bites.

✅ **§6a's EMA HALF LANDED 2026-08-27**, in two commits (`c4fa680` the region arithmetic + its
gate, `799865b` the renderer, driver and four renders). The fifth region exists, `emaOn` and `accOn`
are independent, and `emalambaccdp8x64wxclipbce` ties at **1.20e-07** against the reference's own
`ema_update`. Evidence: `runs/2026-08-27-r50-a2-a1-ema-fifth-region/`.

⬅⬅ **NEXT: the OTHER A2/A1 delta — stochastic depth 0.05 on the residual family — and it opened a
repo-wide finding that has to be settled before it can be rendered.** See §6b. ViT accumulation
(§4d) is untouched and its TPU question still stands.

## 6a. THE TWO FEATURES — what a next session is actually picking up

Everything landed so far was a parameter that already existed. These two are not, and each has one
thing worth knowing before any code is written.

### ✅ A2/A1 deeper — the EMA fifth region. DONE 2026-08-27, and §6a's estimate held

**Landed as scoped.** `nRegions` and `nScalars` are sums over two independent axes (3/4/5 and
3/5/7), the layout is `[θ|m|v|G|E]`, and `trainAdamSched`'s refusal came off LAST. Four renders:
A2 and A1 × {1, 4} replicas.

⭐ **The three things this section predicted were all true.** `nRegions`/`nScalars` were two
one-line functions and the 26 call sites were written against them; `emalambaccdp8x64wxclipbce`
already parsed with no new marker; the refusal was four lines. What it did NOT predict:

* ⚠⚠ **The EMA must advance on APPLY micro-batches ONLY, and nothing in §6a saw it.** The reference
  EMAs once per OPTIMIZER step — `ema_update` follows the `train_step` call and JAX's accumulation
  lives *inside* that call — while this driver invokes the graph per micro-batch. So on an
  accumulate micro-batch the driver hands the graph `%emad = 1, %oemad = 0`, which is `e' = e`
  exactly. ▶ Got wrong it is not merely a k× faster filter: θ is FROZEN on those micro-batches, so
  k−1 of every k updates pull the shadow toward a weight that did not move. It trains and descends.
  ⭐ Costs nothing in the graph — the decay was already a runtime scalar, because it is
  warmup-corrected. Baking 0.9999 would have forced a second artifact.
* ⚠⚠ **"26 reference sites, none of them a hardcoded count" was the DRIVER's sites only.** Four
  gates — `TestViTDpCheck`, `TestConvNeXtDpCheck`, `TestEfficientNetDpCheck`, `TestR50GradCheck` —
  each carried a PRIVATE COPY of the predicates, and two of them used a SUBSTRING test where the
  driver uses a prefix one. That is `TestVariantPredicates`' own header one level up: *a gate on a
  transcription is not a gate on the thing transcribed.* All four now call `VerifiedVariant`.
* ⚠ **`TestVariantPredicates`' region guard needed more than a new row.** It read `nRegions v ==
  (if e || accOn v then 4 else 3)` — true of every spelling that existed and silently true of a
  five-region one. Rewritten as a SUM over the two axes, with a check that all three buckets are
  POPULATED, so the guards cannot be vacuously true of a table with no five-region row.
* ⭐ **The tie row was cheap and its controls were not obvious.** `opt_step_tie.py`'s
  `emalambacc8wxclip` runs `a2accum.py`'s own `ema_update`, and derives `%emad`/`%oemad` from it too
  (`ema_update(1, 0) = d` exactly) rather than transcribing the warmup formula this repo has already
  got wrong once. ⚠ Its sharpest control: a shadow reading the INCOMING θ instead of θ′ is off by
  **1.1%** — four orders above the gate, invisible to a 1e-2 tolerance, and it trains.
* ⚠ **No bf16 twins, and that is reasoned rather than skipped.** The shadow runs on MASTER weights,
  which stay f32 in every bf16 render in this tree, so a bf16 twin would differ from its fp32 peer
  in no line of the EMA block. §4c's "a tier without its precision peer reads as a decision" is why
  the reason is written down instead of left as a gap.

*The original scoping follows, since it is what the work was done against.*

### ⭐ A2/A1 deeper — the EMA fifth region, and it is smaller than §4a implied

§4a records that model EMA is *unrepresentable* alongside accumulation because both want the fourth
region of `[θ|m|v|·]`, and calls lifting it "a change to the blob layout, not a flag". True, but the
layout was built to move:

* `VerifiedVariant.nRegions` and `nScalars` (`VerifiedTrain.lean:242,246`) are **two one-line
  functions**, and the driver is written against them rather than against a literal `3` or `4` —
  **26 reference sites, none of them a hardcoded count.** That was deliberate (§1125's comment says
  so) and it is what makes five regions an edit rather than a rewrite.
* ⭐⭐ **The variant spelling already exists and already parses.** `emaOn` is `startsWith "ema"` and
  `accOn` is a substring test, so `emalambaccdp8x64wxclipbce` fires BOTH today. Nothing needs a new
  marker; `VerifiedTrain.lean:1156`'s refusal is the only thing standing in the way, and it is four
  lines.
* What genuinely has to be built: the renderer emitting shadow AND accumulator as separate regions,
  the optimizer stage returning a fifth name list beside `aNames`
  (`ResNet34RenderB.lean:751` returns a 5-tuple today), and the driver's pack/unpack for 5.

⚠ **Order matters here.** Lift the refusal LAST. While it stands, a wrong render fails loudly at
load; the moment it is gone, an EMA-plus-accumulation graph that packs regions wrongly trains and
reports a number. ▶ And `TestVariantPredicates` needs the 5-region case in its partition BEFORE the
driver can produce one — that table's `nRegions v == (if e || accOn v then 4 else 3)` guard is
currently a two-way split and would go green on a graph it had never seen.

⚠ The other A2/A1 delta, **stochastic depth 0.05**, is unrelated work: `DropPath.lean` exists and
ENet/ConvNeXt render off it, but neither residual renderer imports it and `r34AdamVariant` has no
`drop` marker. Cheaper than the region work and independent of it.

## 6b. ✅ STOCHASTIC DEPTH ON THE RESIDUAL FAMILY — DONE 2026-08-27, after fixing the REFERENCE

✅ **LANDED**, `49ef99a` (the nine JAX emitters) + `5508110` (sixteen sites, four renders, the
placement gate). **Option A was taken** (brett, 2026-08-27): fix the emitters, then render.

⭐ **Both of `sec:r50_a2_a1_cost`'s ⛔ rows are now closed.** The only A2 delta left is the ghost-BN
group — see §6c, which corrects the number this doc had wrong.

⚠⚠ **THE PLACEMENT IS THE CORRECTNESS QUESTION AND NO STRUCTURAL CHECK SEES IT.**
`scripts/misplace_drop_sites.py` moves all sixteen sites onto the block output and produces a render
with the same SSA names, order, types, **18,186 ops** and arity. Run at `q = 1` / `B = 2` on CPU:
TEST at a real mask **2.19e+08**, CONTROL at an all-ones mask **0.00e+00**. The second number is the
argument — at ones the misplaced render is bit-identical, so every endpoint gate passes on it.

⚠ **The `sd`-vs-no-`sd` identity is exact on the FORWARD and not on θ′**, and that is stated rather
than rounded: the loss and all 106 BN statistics differ by 0.0, the gradients by 3.29e-06, and θ′
amplifies that through AdamW's `m̂/(√v̂+ε)`. The extra `multiply` changes XLA's fusion, so the
161-parameter reduction reassociates. A bit-exactness claim on θ′ would have been false.

✅ **THE PHASE-2 CAVEAT IS CLOSED — re-probed, and the CONTROL is what settles it**
(`runs/2026-08-27-r50-droppath-reprobe/`). The figures were taken against the scalar-mask reference,
so they were re-run on the fixed one: A2 reads **1368.3 → 1359.7 ms/step (−0.63 %)**, and A3 —
whose recipe sets `dropPath := 0.0` so its trainer emits **no `dpkeys` at all** — reads
**715.3 → 711.5 (−0.53 %)**. A net with no stochastic depth moved by the same amount, so A2's move
is session drift and not the mask shape. ⚠ Checked first that the probe passes a live `drop_key`;
one passing `None` would early-return and compare nothing. ▶ The book's committed 1,368 and 715
stand; this session's are NOT promoted, per §8.6's practice.

⚠ **Two defects the placement gate itself had**, both found by running it: `hash()` is salted per
PROCESS, so its headline figure moved four orders between two invocations (now `crc32`); and the
`v` region was initialised random-signed, so `√v` gave NaN for 50 of 161 parameters beside a
perfectly finite loss. Recorded because a gate whose number is not reproducible is a demo.

*The original analysis follows, since it is what the decision was made against.*

## 6b (original). ⛔ STOCHASTIC DEPTH — blocked on a REFERENCE defect, not on renderer work

§6a called this *"unrelated work … cheaper than the region work and independent of it"*. The
renderer half is indeed small. What it ran into is not.

⚠⚠⚠ **EVERY CONVOLUTIONAL NET'S JAX REFERENCE DRAWS A SCALAR BERNOULLI PER BLOCK PER STEP, SHARED
BY THE WHOLE BATCH — and the verified renders are per-EXAMPLE.** Read off `jax/Jax/Codegen.lean`,
not assumed:

| block emitter | line | mask |
|---|---|---|
| `bottleneck_block` / `_down` (ResNet-50) | 703, 714, 725, 737 | `bernoulli(key, keep)` — **scalar** |
| `convnext_block` | 1204 | `bernoulli(key, keep)` — **scalar** |
| `mbconv_block` (EfficientNet) | 860, 914 | `bernoulli(key, keep)` — **scalar** |
| `uib_block` (MNv4) | 1036 | `bernoulli(key, keep)` — **scalar** |
| `_drop_branch` (ViT/DeiT) | 1150 | `bernoulli(key, keep, (B,1,…,1))` — **per-example** ✅ |

⭐ **The verified side is the one that is right, and the pinned spec says so.** timm 1.0.28's
`drop_path` (read from `.venv-timm`, not from memory):

```python
shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
```

`planning/stochastic_depth.md` quotes `_drop_branch` — the per-example one — as *the* reference and
builds the whole op family (`dropPathB`, `F32.dropScales`, the per-example shard rule) around it.
So the four conv emitters are the outlier, and the same file already contains the correct
definition eleven hundred lines away.

⚠ **This is not hypothetical for artifacts that already ship.** 26 committed `convnext*` and
`efficientnet*` drop renders are per-example against a per-block-scalar oracle. The ViT family's 12
are fine. Nothing in the tree records the divergence.

▶ **Why it blocks R50 rather than merely annoying it.** A verified render's gate is agreement with
the JAX oracle. Rendering R50's stochastic depth per-example — the only shape the kit's op and the
driver's `F32.dropScales` have, and the correct one — produces an artifact that cannot be tied
against its own reference. Rendering it scalar would mean a second drop op, faithful to a defect.

**The fork, and it is a scoping decision rather than a technical one:**

| | what it costs | what it leaves |
|---|---|---|
| **A. Fix the four emitters to call `_drop_branch`, then render** | one definition in one file; regenerate 9 references | every drop render tie-able; ⚠ changes what a future ConvNeXt/ENet JAX run computes |
| **B. Render per-example, record the divergence, leave JAX alone** | the render only | a knowingly-wrong oracle for 26+4 artifacts |
| **C. Leave sd unrendered; A2/A1 keep the one remaining delta** | nothing | §4a's second ⛔ stands |

⭐⭐ **A INVALIDATES NOTHING MEASURED, and that was CHECKED rather than assumed** (2026-08-27).
`dropPath > 0` appears in **ImageNet-tier configs only** — ConvNeXt-T/S/B, ENet-B0, ViT-Ti/S/B, R50
and MNv4's `full`. Every Imagenette config sets none, which is where the completed JAX runs are.
At ImageNet scale: R50's A3 reference sets `dropPath := 0.0`, and MNv4-Conv-M's 75.51% ran the
`default` recipe — only `full` sets 0.075, and its own comment says it is *"not yet wired into
UIB — verify before trusting"*. ▶ **So no quoted number in this repo was produced by a JAX run
with stochastic depth active.** The defect is entirely latent: it reaches future ImageNet-tier
reference runs, and it means 26 shipping verified artifacts have no correct oracle today.

---

## 6c. GHOST-BN — the gap is 8×, not 2×, and only half of it is reachable on this box

⚠⚠ **THIS DOC AND THE BOOK BOTH HAD THE NUMBER WRONG.** §4a said *"These render 8×64, i.e. 64-image
ghosts. The reference reaches 2048 as 4 accumulation steps × 512 global (128 per device on four
cards), i.e. **128-image ghosts**."* The 128 is the per-device **tensor**; the BN **group** is the
full **512**, because `bm = jnp.mean(x, axis=(0,2,3))` reduces over an axis the reference
mesh-shards (`P(None,'batch')`), so XLA inserts the collective. `a3_paper_fidelity.md` §2.2 already
had this right for A3 and said so in the same words; §4a restated it from the wrong side.

▶ **So the gap is 64 → 512, an 8×, recorded as a 2×.** Corrected in the book (`ed8de4e`'s successor)
and here.

### What landed

✅ **The reference's own factorisation, rendered** — `k = 4` × 128 per device × 4 replicas = 2048,
which is `GRAD_ACCUM = 4` / `MICRO_BATCH = 512` exactly. Group 64 → **128**. Four artifacts
(`emalambacc{,dp}4x128wxclipdropbce{,wd001}bf16`).

⚠⚠⚠ **"bf16-ONLY" WAS WRONG, AND THE ERROR WAS READING A DEFAULT AS A LIMIT** (brett caught it,
2026-08-27). `4×128` fp32 was called unusable at 95–97 % of an 11.68 GiB budget. **11.68 GiB is not
the card** — it is the GPU plugin's BFC default of `memory_fraction = 0.75` on a 16 GB 4060 Ti, and
it left four gigabytes unreachable.

⛔ **And the verified path had no way to raise it.** `ffi/pjrt_ffi.c` called `PJRT_Client_Create`
with **no create options**. ⚠ `XLA_PYTHON_CLIENT_MEM_FRACTION` does not help: the plugin never
reads it (checked — the string is absent from `xla_cuda_plugin.so`); JAX's *Python* layer reads it
and passes `memory_fraction` as a create option. So the JAX trainers could always be told to use
the whole card and the verified trainers could not, and nothing in the tree said so.

⭐ **Fixed**: the shim passes `memory_fraction` and `preallocate` as create options, off
`LEAN_MLIR_MEM_FRACTION` / `LEAN_MLIR_PREALLOCATE`. XLA's own log line is the measurement —
*"XLA backend allocating 15.11GiB (16221470720 bytes) … for BFCAllocator"* against
*"11.68GiB (12543066112 bytes)"* unset. **+3.43 GiB, 29 %.**

| render | peak | of 11.68 (default) | of 15.11 (at 0.97) |
|---|---|---|---|
| `8×64` fp32 | 6.18 G | 53 % | 41 % |
| `8×64` fp32 + EMA + sd | 6.37 G | 55 % | 42 % |
| `8×64` bf16 | 4.32 G | 37 % | 29 % |
| `4×128` bf16 + EMA + sd | 8.09 G | 69 % | 54 % |
| `4×128` fp32 + EMA + sd | **11.91 G** | ⛔ over | ✅ **79 %** |

✅ **So the fp32 peers ARE rendered** and the full (precision × factorisation) square exists —
§4c's rule holds after all. Running the `4×128` fp32 pair needs `LEAN_MLIR_MEM_FRACTION=0.97`,
which is a job-config line rather than a missing artifact.

⚠ **A compile-time peak is not independent of the allocator it was compiled against**: the fp32
`4×128` reads 11.52 G under the default budget and 11.91 G under the raised one, because XLA
rematerialises less when it has room. Quote the budget with the peak.

⚠ The shim change is opt-in and was regression-checked: `shard-check convnext` returns
`77.999521 e-9` / `0.041654` **bit-identically** with the option set and unset.

⚠ The EMA fifth region costs **0.19 G** and it is all `args` (0.42 → 0.51 G): 25.6 M params × 4 B =
102 MB, exactly one more region. Temporaries are unchanged at 5.38 G. **Stochastic depth costs
nothing measurable** — 16 masks of `tensor<64×f32>` is 4 KB.

### What is still owed: SYNC-BN, and it is a feature

Closing the last 4× (128 → 512) means all-reducing the statistics themselves. Scoped, not built:

* ⛔ **A new batched operator.** `bnBatchF` computes its own mean/var internally. There is
  `bnEval`/`bnPerChannelEvalF` which normalise by GIVEN μ/σ² — but both are per-example; the
  batched family has no eval-BN peer. `bnBatchMeanB`/`bnBatchVarB` exist and already compute the
  statistics, but only for the REPORTED running stats, not for the normalisation.
* ⚠⚠ **VARIANCE DOES NOT AVERAGE.** `Var_global ≠ mean(Var_replica)` unless the replica means
  agree. The correct combination all-reduces μ and `E[x²]` and forms `Var = E[x²] − μ²`; from what
  the ops give you that is `all_reduce(var + mean²)`. A render that averaged the variances would
  train, descend, and be a different normalisation.
* ⛔ **The BACKWARD changes too.** `bnBatchBack`'s two internal reduction sums are over the local
  batch; under sync-BN they must be all-reduced. That is not plumbing.
* ⚠ **53 BN layers × 2 collectives per step**, against a run whose whole throughput story is that
  the all-reduce term is 56 ms (`a3_paper_fidelity.md` §2.2: *"Measure before adopting"*).

▶ Comparable in size to §4d's ViT accumulation. Its own planning doc if it is picked up.

---

### ⚠ ViT accumulation (§4d) — answer the hardware question first

§4d's own warning is the thing to settle before touching `ViTRenderB`: **a TPU deletes this item.**
v3 is 16 GB/core and v4 is 32, the FFI is plugin-agnostic, and `$PJRT_PLUGIN` always wins — so
reaching DeiT's global 512 becomes an env var rather than a renderer feature. ▶ Building the
accumulation loop for one box is a medium-to-large job that a different box makes unnecessary.

⭐ **What changed since §4d was written**: ViT-B now has a MEASURED wall clock — 423 → 317 ms/step,
**356 → 268 h** at global 128 — so the cost of *not* having accumulation is now a number rather
than a shrug. Whatever is decided, that is the figure to weigh it against.

✅ **The correction this doc owed the book is PAID** (`40f63d7`). `sec:r50_a2_a1_cost`'s table said
A1's renderer work was "same render as A2"; the two artifacts now exist and differ in **exactly one
line of 18,000** — the baked `%wd`, 0.01 against 0.02. The cell changed when the render that proves
it existed, which is what this note asked for.

---

## 1. THE INVENTORY

Surveyed 2026-08-27 against the tree at `253fb75`. "Render" means a committed
`verified_mlir/*_train_step.mlir`; "run" means an ImageNet-scale verified training result.

| side quest | book § | JAX peer | verified render | verified run | what is actually owed |
|---|---|---|---|---|---|
| **RSB-A2** | `sec:r50_a2_a1_cost` | ✅ `a2-accum` | ✅ fp32 ×2 + bf16 ×2 + **EMA + sd ×2** | — | ✅ nothing — **train it** |
| **RSB-A1** | `sec:r50_a2_a1_cost` | ✅ `a1` | ✅ fp32 ×2 + bf16 ×2 + **EMA + sd ×2** + own shim | — | ✅ nothing — **train it** |
| **MNv4-Conv-M** | `sec:mnv4_side_quest` | ✅ | ✅ 4 variants incl. DP + bf16 | ✅ **4-GPU, tied** | ✅ nothing — **run it** (§4b) |
| **ConvNeXt-S** | `sec:convnext_sb` | ✅ | ✅ fp32 ×2 + **bf16 ×2** | steps only | ✅ nothing — **train it** (§4c) |
| **ConvNeXt-B** | `sec:convnext_sb` | ✅ | ✅ fp32 ×2 + bf16 ×2 | ✅ **benchmarked** | ✅ nothing — **train it** (§4c) |
| **ViT-S** | `sec:vit_sb` | ✅ | ✅ fp32 + **bf16** | ✅ measured **1.89×** | ✅ nothing — **train it** (§4c) |
| **ViT-B** | `sec:vit_sb` | ✅ | ✅ fp32 + **bf16** @ global 128 | ✅ measured 1.40× | **an accum render** (§4d) |
| **EfficientNetV2-S** | `sec:enet_side_quests` | ⚠ Imagenette only | ⛔ nothing | — | spec, peer, shim, render (§4e) |
| Noisy Student | `sec:enet_side_quests` | — | — | — | out of scope: needs JFT-300M |
| EfficientDet | `sec:enet_side_quests` | — | — | — | out of scope: detection, not this book's head |

⭐ **The surprise is how little is missing.** Six of the eight buildable rows already have a
committed render. The book reads as though the S/B variants are unbuilt because Track 4 does not
list them and their cost columns say `[todo]` — but `convnextsin`, `convnextbin`, `vitsin` and
`vitbin` all emit, tie and step today. What is missing is mostly **bf16 twins and one gate**, not
architecture work.

✅ **`planning/chapter_makeover.md`'s DP row was stale and is fixed** (`fd9981e`). It said MNv4
data parallel was "**not rendered** … Deliberately NOT emitted"; the artifacts had been on disk
since 2026-08-25, so it went through "rendered, ungated" and straight out the other side — it is
now rendered AND tied. ⭐ The adjacent **bf16** row in that table was stale the same way ("0 bf16
artifacts in `verified_mlir/`") and was fixed in the same pass; leaving a known-false claim beside
one being corrected is worse than fixing both.

---

## 2. NAMING — four namespaces, and the rule that ties them

### 2a. Where they are today

| namespace | example | who writes it |
|---|---|---|
| Lean exe | `resnet50-imagenet-verified`, `mobilenetv4-imagenet-verified`, `vit-s-imagenet-verified` | `lakefile.lean` |
| MLIR slug | `resnet50in`, `resnet50in160`, `convnextsin`, `vitbin` | `VerifiedNets.lean` `slug` |
| variant | `LEAN_MLIR_VARIANT=lambaccdp8x64wxclipbce` | the renderer's `#eval` names |
| recipe | `LEAN_MLIR_RECIPE=2018` → `generated_resnet50_imagenet_2018_shim.py` | `scripts/gen_shims.sh` |
| job | `scripts/jobs/r50-a3-wxclip-4gpu.conf` | one file per run |

### 2b. The four disagreements, with the bug each one caused or invites

✅ **All four are closed as of 2026-08-27**, by `4ee304a` (1, 2, 4), `d023220` (3) and `796cacf`
(the check that keeps 1 and 2 closed). Kept as written, because the *reasons* are what a later
naming decision has to re-read.

1. ⛔ **`-imagenet-` in a job name means "the default recipe", and the default differs per net.**
   `r34-imagenet-4gpu` is 2018 at the 30-epoch short tier; `r50-imagenet-4gpu` is RSB-A3 without
   `wx`/`clip`. Two names of identical shape, two unrelated recipes, and neither says so. This is
   the same failure class as the `LEAN_MLIR_RECIPE=2018` omission that `2859cd7` just fixed in the
   book's own reproduction command — there, a 2018 run streamed RSB-A2's RandAugment for
   want of one variable.
2. **The recipe is in some job names and not others.** `r50-2018-4gpu` names it; `mnv2-imagenet-4gpu`
   does not, and does not set `LEAN_MLIR_RECIPE` at all — it inherits `default` implicitly.
3. **Abbreviation is per-target, and MNv4 disagrees with itself**: `mobilenetv4-verified-adam`
   (Imagenette) beside `mnv4-imagenet-verified` (ImageNet), while MobileNetV2 spells itself out in
   both. Size variants concatenate where the base does not: `vits-`, `convnextb-`.
4. **Axes leak into names unevenly**: epochs in `r34-imagenet-90ep-4gpu`, precision in
   `r50-2018-bf16-4gpu`, an optimizer detail in `r50-a3-wxclip-4gpu`, nothing in the rest.

### 2c. The scheme

⭐ **N1 — a job is `<net>-<recipe>[-<axis>…]-<n>gpu`, and `<recipe>` is EXACTLY the string
`LEAN_MLIR_RECIPE` takes.** Not "imagenet", not omitted. A net whose emitter has one recipe still
spells it (`default`), because unnamed is precisely what let `r34-imagenet` and `r50-imagenet` mean
different things.

▶ **This is mechanically checkable and should be a precheck, not a convention.** `supervise.sh`
already runs per-job prechecks; add one that asserts the second dash-field of the job's own
filename equals the `LEAN_MLIR_RECIPE` in its `ENV_EXTRA`. A convention nothing enforces is how
namespace 4 drifted from namespace 5 in the first place.

✅ **Landed as an ENGINE check, not a per-job one** (`796cacf`), so a new job cannot forget it.
⚠⚠ **But not against `LEAN_MLIR_RECIPE`, because that rule cannot hold for the jobs it most needs
to.** `LEAN_MLIR_RECIPE` is read by exactly one driver — `MainResnet50Imagenet.lean:84` — and A3
is selected there by `LEAN_MLIR_RES=160`; line 93 *throws* on any non-default recipe at 160, since
2018 is a 224/224 recipe and the two selectors cannot both move. So `r50-a3-4gpu` could never have
set `LEAN_MLIR_RECIPE=a3` to be checked against, and the four nets whose drivers ignore the
variable would have carried a load-bearing-looking line that nothing reads.

⭐ **What landed instead: a conf-level `RECIPE=` field.** The engine asserts (a) every `-<n>gpu`
job sets it, (b) it equals the filename's second dash-field, and (c) any `LEAN_MLIR_RECIPE` in
`ENV_EXTRA` equals it too. Rule (c) is vacuous for six of nine jobs today and deliberately so —
the slot gets checked the day those nets gain a second recipe. The *net-specific* half lives in
the net that has one: both A3 confs now assert `LEAN_MLIR_RES=160` is present **and**
`LEAN_MLIR_RECIPE` is absent, the second being a refusal to write the config that looks most
correct and does not start. Six controls, all firing — see `796cacf`.

✅ **What landed** (`4ee304a`). The two `r34` rows are the plan's, corrected:

| today | became | why |
|---|---|---|
| `r34-imagenet-90ep-4gpu` | `r34-default-4gpu` | 90 ep is the paper schedule; unmarked = paper |
| `r34-imagenet-4gpu` | ⚠⚠ **dropped** — it is `EPOCHS=30` on the above | see below |
| `r50-imagenet-4gpu` | `r50-a3-4gpu` | it is A3, and only its contents said so |
| `r50-a3-wxclip-4gpu` | unchanged | already the target shape |
| `r50-2018-4gpu` / `-bf16-` | unchanged | already the target shape |
| `mnv2-imagenet-4gpu` | `mnv2-default-4gpu` | + `RECIPE=default`, checked by the engine |
| `enet-imagenet-4gpu` | `enet-default-4gpu` | ditto |
| `cnx-imagenet-4gpu` | `cnx-default-4gpu` | ditto |
| `vit-imagenet-4gpu` | `vit-default-4gpu` | ditto |

⚠⚠ **R34's recipe is `default`, not `2018` — this table said `2018` and N1's own rule said
otherwise.** `jax/MainResnetImagenet.lean:65` registers R34's recipes as `default` (90 ep, the
paper schedule) and `short` (30 ep). "2018" is R50's recipe name, `resnet50Imagenet2018Verified`,
and R34 has nothing by it — so `r34-2018-4gpu` would have named a string no registry or driver
accepts. N1's "a net whose emitter has one recipe still spells it (`default`)" was right and this
table was the slip. ▶ The book still *calls* R34's recipe "the 2018 recipe" in prose
(`content.tex:5531`), which is why Track 4's recipe column glosses `default` as such.

⚠ **The 30-epoch R34 job was dropped rather than renamed.** The two confs never differed in
recipe — same `momdp64` render, same shim, same DEVS, same per-replica batch — and *neither* set
`LEAN_MLIR_EPOCHS`, so the short tier was always the 90-epoch cosine stopped early. `EPOCHS=30`
reproduces it exactly. Its provenance moved into `r34-default-4gpu.conf`'s header rather than
being deleted with it, and its schedule-fusion PRECHECK stayed — folding makes that collision
*easier* to hit, since one file now writes the checkpoint both tiers resume from.

**N2 — the slug is the NET, the variant is the RECIPE.** `resnet50in160` looks like an exception
and is not: the driver opens `<slug>_fwd.mlir` and `<slug>_fwd_eval.mlir` **by name**, so anything
the eval forward's shape depends on has to be in the slug. Resolution is; optimizer, accumulation,
replica count, decay and precision are not.

⭐ This rule *predicts* §4a's answer rather than needing a decision: RSB-A2 trains and evaluates at
224, so it reuses the existing `resnet50in` slug and its existing `_fwd_eval`, and everything that
differs goes in the variant. No new slug, no new forward.

**N3 — variant grammar, in this order**: `<opt>[acc][dp]<batch>[x<replicas>][wx][clip][drop][do][bce][wd<d>][bf16]`.
Every committed name already obeys it (`lambaccdp8x64wxclipbce`, `adamdp128x4wxclipdrop`,
`emarmsdp64dropdo`, `momdp64bf16`); it has just never been written down, and §4 adds four names
that have to land in the same order to stay sortable.

**N4 — one abbreviation policy for Lean exes: the chapter's spelling, with the size hyphenated.**
✅ Landed `d023220`.

| today | became |
|---|---|
| `mnv4-imagenet-verified` | `mobilenetv4-imagenet-verified` |
| `vits-imagenet-verified` / `vitb-` | `vit-s-imagenet-verified` / `vit-b-` |
| `convnexts-imagenet-verified` / `convnextb-` | `convnext-s-imagenet-verified` / `convnext-b-` |

Five renames, all `lean_exe` labels plus their references in the blueprint, `scripts/jobs/` and the
planning docs. ⚠ **Grep the blueprint for each old name before renaming** — `vits-imagenet-verified`
appears in `sec:vit_sb`'s prose and `convnexts-imagenet-verified` in `sec:convnext_sb`'s, and a
rename that misses those leaves the book naming a target that no longer builds.

⚠ **Do the renames in one commit with no other change**, so `git log --follow` on any target stays
readable and so a bisect over a later run failure never lands mid-rename.

✅ Done, and split further: exe renames (`d023220`, label-only, no file moves) and job renames
(`4ee304a`, pure `git mv`) are separate commits, because only the second moves files.
⚠ **A `lean_exe` rename leaves the OLD binary in `.lake/build/bin/`.** All five were deleted in the
same pass — a gate that greps for a binary would otherwise keep finding an Aug-14 build of a target
that no longer exists, which is the `vit-fwd-b-tie` failure mode.
⭐ **One drive-by the rename forced** (`4f87a64`): `MainViTBImagenet.lean`'s header was copied from
ViT-S and never retargeted — it claimed `D = 384`, MLP 1536 and 22,050,664 parameters, and
contradicted its own ⚠⚠ batch paragraph two lines down. Line 3 was a rename target, so half-fixing
it was not an option.

---

## 3. TRACK 4 — the restructure

✅ **LANDED 2026-08-27, `7c457a3`.** Built and checked: `xelatex print.tex` exits 0 at 171 pages
with zero undefined references (all five `\S\ref` targets and `chap:residual` resolve), and
overfull hboxes are 79 before and 79 after, none in the Track 4 region. The side-quest table needed
`\small` to clear a 40.3 pt overrun.

Appendix B Track 4 (`blueprint/src/content.tex:13966`) opened "Seven nets train on full
ImageNet-1k" over a table of eight rows that mixes main-line recipes, a second recipe for R50, and
one side quest with a `\withheld` job.

⚠ **Track 4 has no `\label`** — Track 3 has `sec:getting_started_track3` and Track 4 has nothing, so
no chapter can point a reader at it. Add `\label{sec:getting_started_track4}` in the same pass; the
side-quest table below is exactly the thing six chapter sections will want to `\ref`.

**Split it in two.** Main table = one row per (net, recipe) that a chapter's *ImageNet recipe*
section reports. Side-quest table = one row per side-quest variant, with a **status** column,
because most are not runnable and `\withheld` cannot say why.

```
\section*{Track 4: ImageNet-1k runners}
  <intro: the scale tier, chapter order, and that a row is a (net, recipe) pair>
  <main table>                       ← 7 rows
  \prosesection{This tier needs Python}      (unchanged)
  \prosesection{Running one}                 (unchanged)
  \prosesection{Running one for a day and a half}   (unchanged)
  \prosesection{Side quests}         ← NEW, after the running instructions
  <side-quest table with a status column>
```

⭐ **The side-quest table goes AFTER the run instructions, not beside the main table.** A reader
working through Track 4 wants the seven runnable things and the commands; the side quests are a
different question ("what else is rendered?") and putting them first buries the tier's actual
purpose. It also means the status column's `render only` rows never sit next to a job name a reader
might try to run.

**Main table** — target, recipe, job, in chapter order. ✅ What landed:

| target | recipe | job |
|---|---|---|
| `resnet34-imagenet-verified` | `default` (the 2018 recipe) | `r34-default-4gpu` |
| `resnet50-imagenet-verified` | `2018` | `r50-2018-bf16-4gpu` |
| `resnet50-imagenet-verified` | `a3` (RSB-A3, train@160) | `r50-a3-wxclip-4gpu` |
| `mobilenetv2-imagenet-verified` | `default` | `mnv2-default-4gpu` |
| `efficientnet-imagenet-verified` | `default` | `enet-default-4gpu` |
| `convnext-imagenet-verified` | `default` | `cnx-default-4gpu` |
| `vit-imagenet-verified` | `default` (DeiT-Ti) | `vit-default-4gpu` |

⚠⚠ **The R50 / 2018 row is the `bf16` job, not `r50-2018-4gpu` as this plan had it.** The rule
above — "one row per (net, recipe) that a chapter's ImageNet recipe section *reports*" — decides
it, and this table had named the job that merely exists. §5.4's 77.07% row is 4× 3060 **bf16**,
and the book's own reproduction verbatim spells `LEAN_MLIR_VARIANT=momdp64bf16`. Same check on the
other two: §5.4 names the A3 artifact outright as `resnet50in160_lambaccdp8x64wxclipbce_train_step`
(so `r50-a3-wxclip-4gpu` was right, and `r50-a3-4gpu` carries the earlier `lambaccdp8x64bce`), and
R34's 74.14% is fp32 / 90 ep, which is `r34-default-4gpu`'s `momdp64`.

▶ **So `scripts/jobs/` holds more jobs than the table has rows**, and one caveat sentence after it
says so: the unlisted ones are *axis* siblings differing in precision or one optimizer knob, never
in recipe. Listing them would have put two job names on one (net, recipe) row, which is Rule 1 of
`blueprint_lowerer_pattern.md` at table scale.

**Side-quest table** — target, variant, status, and the chapter section it belongs to:

| target | variant | status | § |
|---|---|---|---|
| `resnet50-imagenet-verified` | `a2-accum` | render owed | `sec:r50_a2_a1_cost` |
| `resnet50-imagenet-verified` | `a1` | render + shim owed | `sec:r50_a2_a1_cost` |
| `mobilenetv4-imagenet-verified` | Conv-M | rendered; **single-device only** (no shard gate) | `sec:mnv4_side_quest` |
| `convnext-s-imagenet-verified` | — | rendered fp32; bf16 owed | `sec:convnext_sb` |
| `convnext-b-imagenet-verified` | — | rendered fp32 + bf16; **unbenchmarked** | `sec:convnext_sb` |
| `vit-s-imagenet-verified` | — | rendered fp32; bf16 owed | `sec:vit_sb` |
| `vit-b-imagenet-verified` | — | rendered fp32 @ global 128; bf16 + accum owed | `sec:vit_sb` |
| EfficientNetV2-S | — | no spec | `sec:enet_side_quests` |

⚠ **MNv4's row keeps its reason.** The current table's bare `\withheld` reads as a redaction; the
status column has to say *single-device only*, because that is a 63.5 h run rather than a missing
one, and the difference is one gate (§4b).

---

## 4. THE RENDERER WORK, ITEM BY ITEM

### 4a. RSB-A2 at 224 — four `#eval` lines. And A1 is NOT free. ✅ RENDERED 2026-08-27

✅ **LANDED**, `6f89701` (eight renders + A1's spec, shim and recipe arm) + `40f63d7` (the book).

⚠⚠⚠ **BUT "it closes a whole book section" WAS WRONG, and this is the finding of the item.** The
`#eval`s were four lines as predicted. The section does not close, because **two of A2's
regularisers have no expression on this path** — checked field by field against
`resnet50ImagenetConfigA2Accum`, not assumed:

* ⛔ **Model EMA (`emaDecay := 0.9999`) is UNREPRESENTABLE, not merely unrendered.** The EMA shadow
  and the gradient accumulator are the **same fourth region** of `[θ|m|v|·]`, and
  `VerifiedTrain.lean:1156` already throws on the pairing: *"they occupy the same fourth region …
  Render one or the other."* Accumulation is not optional for A2 — effective batch 2048 at 224² has
  no other route on a 16 GB card. Lifting it means a **fifth region** in the driver's pack/unpack
  and in every optimizer's return list. That is its own project, not a flag.
* ⛔ **Stochastic depth (`dropPath := 0.05`).** `DropPath.lean` exists and ENet/ConvNeXt render
  `drop` variants off it, but neither `ResNet34RenderB` nor `ResNet50RenderB` imports it (0 hits in
  both) and `r34AdamVariant` has no `drop` marker.
* ⚠ Ghost-BN group 64 against the reference's 128.

⭐ **A3 met neither obstacle because A3's own recipe sets `useEMA := false` and `dropPath := 0.0`.**
That is why this was invisible from A3's success, and it is the reason the item read as cheap.

✅ **Both of the section's guesses about the shims were right, and both were MEASURED.** The
`a2-accum` shim is **byte-identical** to `default`'s (md5 `d42c412beb4f`), so A2 needed no new spec
and no new shim; A1's differs in **one line**, `_MIX_A` 0.1 → 0.2, so A1 got
`resnet50ImagenetA1Verified`, a `gen_shims.sh` row and a `LEAN_MLIR_RECIPE=a1` arm.
⚠⚠ **And that α is also an env override** — `SHIM_MIXUP_ALPHA` — so A1's mixup could have been had
from the default shim with one variable and nothing in a 600-epoch log recording which α trained.
That is the `LEAN_MLIR_RECIPE=2018` failure exactly; a named shim the driver refuses to start
without is the version that cannot be got wrong.

⭐ **Eight renders, not four**: both tiers got the full (precision × replicas) square in one pass,
so neither ships without its bf16 twin the way ConvNeXt-S did — §4c's own complaint, avoided.

⭐ **Two gates earned their keep and one gap turned up:**
* `opt_step_tie.py` **refused on stale references** before running — §0's "regenerate before
  probing" trap, caught by the gate rather than by me. After regenerating: 6/6 within rtol 2e-6,
  `lambacc8wxclipwd001` at **1.09e-07** against the emitted A1 trainer that bakes `WD = 0.010000`.
* `check_render_coverage.py` **refused** until all eight artifacts were added to CI's drift guard.
* ⚠ `tests/TestVariantPredicates.lean` had **no bf16 spelling at all**, though every bf16 render in
  the tree goes through its five predicates. Added the four new ones plus `momdp64bf16` as the
  non-accumulating counter-case. ⚠ It also has **no lake target** despite being cited as *the* gate
  in ten places — run it with `lake env lean`.

---

*The original text follows, since it is what the work was done against.*


⭐⭐ **The renderer already takes every parameter this needs.**
`Proofs.StableHLO.resnet50TrainStepFaithfulB` (`LeanMlir/Proofs/Codegen/ResNet50RenderB.lean:521`)
carries `replicas`, `opt`, `slug`, `bce`, `wdStr`, `q`, `wdExclude`, `gradClip`, `clipNorm` and
`bf16`, all as trailing defaulted parameters. A2 is the committed A3 call at `q := 7` with the 224
slug:

```lean
-- A2: RSB-A3's composition at 224 instead of 160. Same LAMB × BCE × k=8 × 4 replicas ⇒ 2048.
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambaccdp8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true))
-- Its 1-replica peer — `r50-accum-tie` and `r50-accum-shard-tie` both compare against one.
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambacc8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true))
```

plus the same pair at `(bf16 := true)`, which is what the book's `renderer work` cell calls
"LAMB + BCE at 224, bf16". `resnet50in_momdp64bf16` proves the bf16 path renders for this net at
this resolution, so the twin is a flag, not an investigation.

⚠⚠ **The book's A1 cell says "same render as A2" and that is wrong.** `wdStr` is a **baked
`stablehlo.constant`**, not a runtime operand — the renderer's own comment at
`ResNet50RenderB.lean:534` says so, and says A1's 0.01 against A2's 0.02 "is what makes that a
re-render rather than a new op". `wdVariantMark` (`ResNet34RenderB.lean:637`) appends `wd001`, so
A1's artifact is a distinct path. **A1 needs its own render.** Correct that cell when this lands.

⚠ **A1 also needs its own shim.** A1 differs from A2 in three fields: epochs (600, a driver knob —
free), weight decay (baked — the re-render above), and **Mixup α 0.2 against A2's 0.1**. Mixup is
data-side and rides the shim, so `generated_resnet50_imagenet_a1_shim.py` has to be added to
`scripts/gen_shims.sh`'s list, and `resnet50Imagenet…A1Verified` needs its own `shimScript` field.
That is exactly the `shimScript`-is-per-net mechanism `VerifiedNets.lean:662` was written for.

⭐ **A1's optimizer arm is already tied, against A1's own reference.**
`scripts/opt_step_tie.py:132` carries `("lambacc8wxclipwd001", "generated_resnet50_imagenet_a1.py",
8, True)` — the wd-0.01 LAMB the A1 render would bake, checked against the emitted A1 trainer that
bakes `WD = 0.010000`. So the optimizer stage is gated before the whole-net render exists, and its
own comment says the row is what turns "the string reaches the constant block" from a code-reading
claim into a measurement.

⚠ **Check the A2 shim rather than assuming it.** The emitter's `default` R50 recipe *is* RSB-A2, so
`generated_resnet50_imagenet_shim.py` is plausibly already A2's augmentation and `a2-accum` differs
from `default` only in optimizer-side fields. Confirm by diffing the two emitted shims before
wiring — this is the exact trap `2859cd7` hit from the other side, where a 2018 run silently
streamed A2's RandAugment.

**Cost: small.** Four to six `#eval`s, one shim entry, one `VerifiedNetSpec`, and the existing gates.
It closes a whole book section.

### 4b. MNv4 — the missing shard gate, not a missing render ✅ DONE 2026-08-27

✅ **LANDED**, `b1208ab` (the gates) + `fd9981e` (the job and the caveat lift) + `f8f0943` (the
book). Evidence in `runs/2026-08-27-mnv4-dp-shard-gates/`.

| gate | result |
|---|---|
| `mnv4-dp-check` fp32 | `bnstat` **bit-exact 67,904/67,904**, gradient norm-rel **8.45e-7** |
| `mnv4-dp-check` bf16 | **bit-exact on all 9,715,512 floats** — θ, m, v *and* bnstat |
| `shard-check mnv4in` | TEST **1.10e-6** vs CONTROL **2.00** — 1.8e6× apart |

⭐ **Both go red on a sum-not-mean render, and the shard TEST lands on exactly `3.000000`** — which
is `|4g − g| / |g|`, the arithmetic the failure mode implies, not merely a big number.

⚠⚠ **FOUR GPUs, not the two this section budgeted.** MNv4's only DP renders are 4-replica and it
has no 2-replica peer, so `PJRT_REPLICAS=2` hits the shim's replica-count guard rather than
degrading. It is also the 1000-class 224² net — `mnv4_adam_train_step.mlir` is single-device, so
there is no Imagenette-scale MNv4 DP render to gate more cheaply.

⭐ **The shard half was ONE ROW, not a harness.** `TestShardCheck.lean` was already generalised to
N replicas and to non-bare variant names (`SHARD_REPLICAS`, `SHARD_VARIANT{,_DP}`), so MNv4 needed
a line in `netOf`. This section budgeted "an MNv4 arm" as though it were work.

⭐ **Both precision arms are gated.** `mnv4in_adamdp64bf16` was exactly as untied as its f32 peer;
the precision axis did not get to inherit a tie it was not given. The `DP_VARIANT` knobs made that
a re-run rather than a second file.

⚠⚠ **"63.5 h on one card to roughly a quarter of that" below is wrong on both halves.** 63.5 h
derives from 114 ms/step measured on Conv-**S**, before the Conv-M conversion (4.1M → 9.7M), so it
never described this network — and no single-card Conv-M figure has ever been measured. The 4×
number did not need projecting: `bf16_renderer.md` §21.1 already had it measured at **177 ms/step
→ 25.6 h** f32, **126 ms/step → 18.6 h** bf16. The job ships f32 by default, to keep a first run
from moving two axes at once.

---

*The original text follows, since it is what the work was done against.*

`mnv4in_adamdp64` and `mnv4in_adamdp64bf16` exist. What does not exist is MNv4 in
`shard-check`'s net list — `lakefile.lean:2102` reads `<convnext|efficientnet|mobilenetv2|vit>` —
and there is no `TestMnv4DpCheck.lean`. The principle the original decision rested on is right and
should survive: **an untied collective artifact looks exactly as trustworthy as a tied one**, so
the DP render must not be quotable until the gate exists.

Needed: an MNv4 arm in `tests/TestShardCheck.lean`, a `TestMnv4DpCheck.lean` on the pattern of its
siblings, and a two-GPU run of both. Then a `mnv4-default-4gpu` job and MNv4's Track-4 status
changes from *single-device only* to a job name.

**Cost: small**, and it is the highest-leverage item here: it takes MNv4's ImageNet run from 63.5 h
on one card to roughly a quarter of that on four.

### 4c. ConvNeXt-S and ViT-S/B bf16 twins — three per-net emits

✅ **ConvNeXt-S LANDED 2026-08-27** (`e17b61c`). Two `#eval`s, no new operator, and the variant
strings needed no new `#guard` — `cnxAdamVariant` keys on replicas and flags, never on the size, so
S reuses `adamwxclipdropbf16` at a different SLUG. That is N2 doing its job.

⭐⭐ **MEASURED, and it beat the estimate. This section predicted ~1.29× for S** on the reasoning
that T's ratio would carry. One session, one 4060 Ti, bs32, `scripts/bf16_device_step.py`:

| model | fp32 | bf16 | speedup |
|---|---|---|---|
| ConvNeXt-T | 157.85 ms | 121.12 ms | 1.30× |
| **ConvNeXt-S** | **268.35 ms** | **192.65 ms** | **1.39×** |
| ConvNeXt-B | 395.73 ms | 294.25 ms | 1.34× |

⭐ **T and B reproduce their committed figures to within 0.3%** (§21.6 has 157.8 → 121.0 and
396.0 → 293.3), which is what licenses S's row rather than S licensing itself. ▶ And it closes
ConvNeXt-B's "**benchmark it**" item in §1 in the same pass, since B was run as a control.

⭐ **S is the FASTEST of the three**, which "deeper costs more" does not predict: depth adds blocks
at widths bf16 already suits, where B's widening moves every stage.

⚠ **1.39× is the GRAPH's, not a wall clock.** The trainer dilutes it — T reads 1.30× here and
1.18× inside a four-card run on real data, B 1.34× and 1.20×. Two rows of §21.6's shape are still
unmeasured for S: peak memory and the 4×bs32 trainer figure.

✅ **AND THE TWO ViT TWINS LANDED THE SAME DAY** (`d78feb6`). ⚠⚠ The section's trap did NOT bite,
and the reason is worth keeping: ViT's stem wgrad does run **0.19×**, but `bf16Conv`/`bf16ConvW`
are already FALSE in Tiny's bf16 render and S and B inherit that, so the one op where bf16 loses on
this architecture was never in the emit at any size. The renderer had been taught the lesson before
this section wrote it down.

⭐⭐ **Measured at FOUR replicas — ViT-S and ViT-B have no single-device render at all**, which is
why `scripts/bf16_device_step.py` grew a `--replicas` flag (a separate compile path; widening the
shared correctness gate's would be how a gate stops gating):

| model | per-dev bs | fp32 | bf16 | speedup |
|---|---|---|---|---|
| ViT-Tiny | 128 | 171.6 ms | 96.4 ms | 1.78× |
| **ViT-S** | 128 | **464.0 ms** | **245.5 ms** | **1.89×** |
| ViT-B | 32 | 383.8 ms | 273.7 ms | 1.40× |

⭐ **1.89× is the largest bf16 payoff in the repo.** ⚠⚠ And checking that claim caught a basis
mismatch: the first draft set it against R50's **1.55×**, which is R50's SINGLE-DEVICE figure.
R50 at four replicas is **1.66×** (336.9 → 203.4 ms). S still leads, by less.

✅ **ViT-B DOES get a wall clock — by measuring, 2026-08-27.** This section first said it could not
have one: Tiny's multiplier gave B a 1.12× trainer speedup, below B's own 1.40× device ratio, and
two defensible models sat 100 h apart. That was true and the answer was to stop modelling. All
three ViTs were then probed directly on real ImageNet (40 steps, 4 cards, `LEAN_MLIR_SKIP_EVAL=1`):

| net | global | steps/epoch | trainer ms/step | 300 epochs |
|---|---|---|---|---|
| ViT-Tiny | 512 | 2,502 | 237 → 168 | 53 → 38 h |
| ViT-S | 512 | 2,502 | 528 → 325 | 113 → 71 h |
| **ViT-B** | 128 | 10,009 | **423 → 317** | **356 → 268 h** (14.8 → 11.1 d) |

⚠⚠ **AND IT SHOWED THE ViT-S FIGURE I HAD ALREADY PUBLISHED WAS WRONG BY 19%** — 627 → 415 ms
derived against 528 → 325 measured. Corrected in `93ec229`'s successor.

⭐⭐ **The finding, and it is why one net's model transfers and another's does not.** ViT's device
figures are FOUR-REPLICA, so the all-reduce is already inside them and the only thing a trainer
adds is the data feed — ~65 ms at 128 img/dev for BOTH Tiny and S, i.e. an additive constant.
ConvNeXt's device figures are SINGLE-DEVICE, so its trainer must also absorb an all-reduce that
grows with the parameter count, and a multiplier is the right shape. Same box, same two
quantities, two different laws.
▶ Checked rather than assumed: a direct probe of ConvNeXt-S returns 368 → 301 ms against the
366 → 292 derived in §4c, fp32 within 0.5%, so that net's derived row stands.
⚠ Controls both ways — ViT-Tiny reads 237 → 168 against a committed 232 → 163 (2–3% slow this
session); ConvNeXt-T reads 231 against 217, which is why §8.6's committed-basis numbers were NOT
overwritten with this session's.

---

*The original text follows, since it is what the work was done against.*


| owed | pattern already proven by |
|---|---|
| `convnextsin_adamwxclipdropbf16`, `convnextsin_adamdpwxclipdropbf16` | `convnextbin_*bf16` (same net family, both variants) |
| `vitsin_adamdp128x4wxclipdropbf16` | `vitin_adamdp128x4wxclipdropbf16` |
| `vitbin_adamdp32x4wxclipdropbf16` | same |

No new proved operator: every bf16 twin these nets need already exists. ConvNeXt-S lacking one is
an accident of B landing after S.

⚠⚠ **Three traps this repo has already paid for, all recorded:**
- A bf16 conv needs a **bf16-typed result plus a convert**. The f32-result shape works for `dot` and
  is silently folded away on `conv`.
- A bf16 dot's **result type** is a ~1.2× performance issue, not a correctness one — ViT-Tiny went
  1.23× → 1.46× on that line alone.
- ⚠ A bf16 op can be **slower** than its f32 peer with every gate green (ViT's stem wgrad at 0.19×).
  Profile each new twin standalone with `scripts/bf16_device_step.py` before believing a whole-step
  number — and never read a trainer's own ms/step for this, because `PJRT_FFI_RESIDENT` is off by
  default and `BENCH_SYNTH` does not control for it.

**Cost: small.** Expect roughly a fifth of the wall clock back on ConvNeXt-S (T's bf16 render runs
1.29× its fp32 peer) and about a third on the ViT pair (Ti's runs 1.46×).

### 4d. ViT-B gradient accumulation — the one real feature

This is the only item that is not a flag. **ViT has no accumulation render.** `ViTRenderB`'s ten
`acc` hits are the *attention* accumulator, a different thing. R50 has the real one
(`R34Opt.lambAccum`, `adamwAccum`, and `resnet50in_accdp4x64` and peers), so the template exists.

Why it matters beyond speed: without it `vitbin` renders at **global 128**, and DeiT's recipe is
global 512. That is a recipe deviation, not a hardware footnote — a ViT-B number produced at global
128 is not comparable to DeiT-B's 81.8% even in principle. The phase-2 side has the same constraint
and solves it the same way: `sec:vit_sb` now records that one-shot global 512 peaks at 11.41 of the
allocator's 11.68 GiB and dies `RESOURCE_EXHAUSTED`, and the JAX trainer reaches 512 as 4×128.

Needed: the accumulation loop in `ViTRenderB`, an `adamwAccum`-shaped `R34Opt` arm for ViT's
optimizer stage, `vitbin_adamaccdp<k>x<b>wxclipdrop` renders at 1 and 4 replicas, and a
`vit-accum-tie` on the pattern of `r50-accum-tie`.

⚠ **A TPU deletes this item.** v3 is 16 GB/core and v4 is 32, and the FFI is plugin-agnostic —
`$PJRT_PLUGIN` always wins — so it is closer to an env var than a port. Weigh that before building
the feature for one box.

**Cost: medium-to-large.** The only item here that should get its own planning doc if it is picked
up.

### 4e. EfficientNetV2-S — a project, not a task

Nothing exists on the verified side, and the JAX peer (`jax/MainEfficientNetV2.lean`) is
**Imagenette-scale only** (`runJax … .imagenette`). So the chain is: an ImageNet JAX peer → a shim →
a `VerifiedNetSpec` → renders → gates. The book is right that no new *primitive* is needed —
`.fusedMbConvNB` is already in the kit as MobileNetV4's stage 0 — but "no new primitive" is the
cheapest quarter of this.

⚠⚠ **And one part of the recipe the render shape cannot currently express.** V2's headline second
idea is **progressive learning**: the image size *rises across training*, with regularization rising
with it. Batch and resolution are both **baked into the graph** on this path (`q` is a render
parameter, `LEAN_MLIR_BATCH` must match what the artifact was rendered at). So a faithful V2 needs
either N renders — one per resolution stage, with the driver switching artifacts mid-run — or an
explicit, stated deviation to a fixed resolution.

⭐ **That is worth writing down whatever happens to V2**, because it is the first recipe in the book
whose *schedule* the verified path cannot represent. Every previous deviation has been a missing
flag; this one is a property of baking shapes into a graph. RSB-A3's train@160 / eval@224 split is
the nearest existing thing and it works precisely because it is **two** instantiations of one
renderer, which is the N-renders answer at N = 2.

**Cost: large.** Recommend it stays a bestiary entry until §4a–§4d are done.

---

## 5. GATES — what a new render must pass before it is quotable

Non-negotiable, and in this order:

1. ⭐ **Byte-identity FIRST.** After any parameterisation, `scripts/regen_verified_mlir.sh proofs`
   must leave `git diff verified_mlir/` **empty** — *before* the new instance is added. That is what
   separates "the refactor broke something" from "the new size is wrong". ViT-S and ConvNeXt-S both
   record this as the gate that made their parameterisation safe.
2. **`#guard` the parameter count** against the JAX emitted count. All six S/B and Tiny probes on
   2026-08-27 printed back the blueprint's guarded figures exactly
   (`runs/2026-08-27-jax-sb-tier-step-probe/`); that is what says a render is the net it claims.
3. **`scripts/check_render_coverage.py`** and **`scripts/render_parity.py`**.
4. **The tie gates for whatever axis moved**: `scripts/opt_step_tie.py` (six variants) for an
   optimizer or decay change, `r50-accum-tie` / `r50-accum-shard-tie` for accumulation,
   `shard-check` and the net's `*-dp-check` for collectives.
5. ⚠ **`lake build <gate>`, not just the binary.** `vit-fwd-b-tie` printed green from a three-week-old
   binary while failing to compile. A gate you did not rebuild is not a gate.
6. ⚠ **`scripts/verify_excerpt.py`** over any chapter before its prose is touched. It has found
   fabricated log lines in chapters already marked done.
7. ⚠ **The PJRT shim is not a lake target.** Editing `ffi/pjrt_ffi.c` and running `lake build` reports
   SUCCESS without rebuilding it. Use the `gcc` one-liner or you get a false green.

---

## 6. ORDER

1. ✅ **Naming + Track 4 restructure (§2, §3) — DONE 2026-08-27.** Five commits, not two: the
   ViT-B docstring the rename forced (`4f87a64`), the exe renames (`d023220`), the job renames
   (`4ee304a`), the `RECIPE=` precheck (`796cacf`), and the Track 4 split (`7c457a3`). Splitting
   exe renames from job renames matters — only the second moves files, so only the second has a
   `git log --follow` to protect.
2. ✅ **MNv4 shard gate (§4b) — DONE 2026-08-27.** Three commits: the gates (`b1208ab`), the job
   and the caveat lift (`fd9981e`), the book (`f8f0943`). It did turn a `\withheld` into a job.
   ⚠ Four GPUs, not two, and the shard half was one row rather than a harness — see §4b.
3. ✅ **RSB-A2 + A1 (§4a) — RENDERED 2026-08-27.** `6f89701` + `40f63d7`. The "same render as A2"
   cell is fixed. ⚠ It did NOT close `sec:r50_a2_a1_cost`: EMA is unrepresentable alongside
   accumulation (one fourth region, two claimants) and stochastic depth has no importer in the
   residual family. Both are now stated in the book rather than owed. See §4a.
4. ✅ **The three bf16 twins (§4c) — ALL DONE 2026-08-27.** ConvNeXt-S (`e17b61c`) at **1.39×**,
   ViT-S and ViT-B (`d78feb6`) at **1.89×** and 1.40×, book in `93ec229`. ConvNeXt-B's and R50's
   benchmarks fell out of the same sessions. §4c is closed.
5. ✅ **A2/A1's EMA half — DONE 2026-08-27.** `c4fa680` (the region arithmetic + its gate) and
   `799865b` (the renderer, the driver, four renders, the tie row). The fifth region exists and
   `emalambacc8wxclip` ties at 1.20e-07. §6a's estimate held; what it missed is the apply-only EMA
   cadence and four gates carrying private copies of the predicates. See §6a.
   ✅ **A2/A1's OTHER delta — stochastic depth — DONE the same day** (`49ef99a` + `5508110`),
   after option A: nine JAX emitters were drawing a per-block SCALAR bernoulli where timm and every
   verified render are per-example, so the reference was fixed first and the sixteen sites rendered
   against it. §6b. ⭐ Both of `sec:r50_a2_a1_cost`'s ⛔ rows are closed and only the ghost-BN group
   remains.
   ⬅ **ViT-B accumulation (§4d → §6a) is untouched** and its TPU question still stands.
   ⬅ **Nothing has been TRAINED.** No five-region graph has been loaded by the trainer; every gate
   in this pass is CPU (XLA-on-CPU, text, `#guard`). The shim's buffer-count refusal is the check
   that would say so, and `runs/2026-08-27-r50-a2-a1-ema-fifth-region/arity_check.py` reproduces
   that arithmetic statically — the strongest cheap substitute and not a replacement.
6. **EfficientNetV2 (§4e)** — its own project, after the rest. ⚠ Its row is OUT of the book's
   Track-4 side-quest table as of `93ec229` (brett, 2026-08-27): a "no spec" row in a table of
   rendered artifacts reads as a commitment rather than a possibility. The ch.7 bestiary entry
   keeps the idea, and §4e below keeps the analysis — including the progressive-learning finding,
   which is the part worth not losing.

⚠ **Do not convert a chapter and re-run its net in the same pass.** §5.7's conversion was interleaved
with a live run and produced three separate stale-number corrections.

---

## 7. WHAT IS DELIBERATELY NOT HERE

* **Imagenette peers for the size variants.** `ViTRender.lean` — the per-example renderer — is still
  pinned at Tiny by roughly 154 dimension literals, and `ConvNeXtRender`'s per-example half is the
  same shape of job. Unpinning them buys a *pedagogical demo* for a *side quest*, which is the wrong
  trade twice over: the Imagenette tier exists to teach a primitive, and the size variants introduce
  none. `sec:vit_sb` already states this as a limitation and should keep stating it.
* **A book-wide results table.** Rule 1 of `blueprint_lowerer_pattern.md` applies at book scale: a
  number with two homes is a number that will disagree with itself.
* **Promoting the planning docs' verified fp32 figures into the book.** `vit_convnext_sb_scaleup.md`
  holds ViT-S at 525 ms/step and ViT-B at 432 (2026-08-14, 4× 4060 Ti, fp32). They are real, and they
  are *not* in `sec:vit_sb`, deliberately — this repo has one commit whose whole subject is quoting a
  plan instead of a run. Re-measure before they enter the book.
* **Noisy Student and EfficientDet.** JFT-300M and a detection head respectively; neither is a
  variant of something already here.
