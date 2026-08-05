# next_session_pipeline_then_r50.md — R34/ImageNet is base camp; bring the rest to parity

**Written 2026-08-05**, at the end of the session that produced the first trustworthy verified
ImageNet number.

▶ **The frame for everything below: R34/ImageNet is now a known-good reference configuration.**
Not "a net that works" — a *standard*. Every other net is behind it on a small number of
enumerable axes, and most of the remaining work is bringing them to it rather than inventing
anything. R50 is the one net that also needs building.

⚠ Read `planning/rsb_a3_r50_verified.md` for the R50 *architecture* scoping — §1's op-cost
collapse, §2.1's three block forms, §2.2's v1.5 stride. This file does not repeat it.

---

## ✅ WHAT LANDED 2026-08-05 (late) — §3.2's debt PAID, §4's blocker BUILT

Two commits. **R50's gradient is gated and gradient accumulation exists, is certified, and trains.**

| | was | now |
|---|---|---|
| **§3.2 R50's gradient** | ⚠⚠ ungated, "say which one licensed the swap" | ✅ `lake build r50-gradcheck` — two tiers on the committed bytes |
| **§4 grad accumulation** | ⛔⛔ `grep gradAccum` returns 0 | ✅ rendered, certified, and smoked on ImageNet |

### The gradient gate — and §3.2 named the wrong two checks

⚠⚠ **§3.2's `vjp_oracle` cases would have gated the WRONG EMITTER.** `tests/vjp_oracle/run.sh`
drives `NetSpec.train`, i.e. `MlirCodegen.emitBottleneckBlock`; `resnet50-imagenet-verified-xla`
runs `ResNet50RenderB`. Different lowerings. ▶ Those cases are still worth adding — they cover
`apps/baselines/MainResnet50Train.lean` — and they are not the R50 gate.

`tests/TestR50GradCheck.lean` drives the committed artifact and needs no second framework: the train
step returns its own loss beside its own `m'`, so one invoke from `m = v = 0` gives `L(θ)` **and**
`g = 10·m'` (§2k's construction).

⭐ **TIER 1 is two closed-form identities with no finite differences anywhere.** Every conv is
BN-followed and bias-free, so `L` is 0-homogeneous in each kernel and Euler gives `⟨g_W, W⟩ = 0` on
53 sites. ⭐⭐ The BN-AFFINE version is the one with teeth: `⟨g_W,W⟩` factors as `⟨c, J_W W⟩` with
`J_W W = 0`, so it holds for **any** arriving cotangent, while `⟨g_γ,γ⟩+⟨g_β,β⟩ = 0` is a statement
*about* that cotangent — and the stem's is the only gradient path through `maxPool3s2`. All 86 sites
land ≤ **6.1e-5**, which is BN's `ε/var` and not a coincidence.

**TIER 2 is the adjoint probe** `⟨g,δ⟩ ≟ (L₊−L₋)/2`, one direction per block, which pins the SCALE
that tier 1 is blind to. ⚠ Its resolution is depth-dependent and MEASURED (`R50_GC_SCAN=1`): 0.00097
at the head, 0.17 at the stem, where the perturbation drags the forward across millions of ReLU
kinks. The stem's fp32 noise floor is read off R50's own BN scale invariance — 0.0012, i.e. 140×
below the residual, so that residual is truncation and not arithmetic.

▶ **The honest sentence: the structure is pinned to 6e-5 everywhere, the magnitude to ~0.1% at
stage 4 and ~17% at the stem.** The two tiers cover each other's blind spots; that is the design.

⚠ Controls, all live: 21 sites where the homogeneity is FALSE violate it with **no overlap at all**
(weakest 12–34× the worst passing site, median ~450×), and the 3e-4 tolerance sits *between* the two
populations rather than being a round number. A doubled gradient misses by 5.9×.

⚠ **Two designs were tried and DROPPED rather than quietly kept** — a whole-net random-sign
direction (degenerate at n = 25.5M) and a projection-blind control (the shortcut is only ~4% of
`‖m'‖²` at s1b0, so it could not fire). Both are recorded in the file. §6, paid again.

⚠ `tests/r50_dp_render_tie.py` carries the verdict to the DP artifacts. The DP render all-reduces
the gradient but **NOT the loss** — `%loss` is replica 0's own shard — so tier 2 cannot run there.
Instead: substitute each `%arsum<P>`'s named gradient SSA on both sides, drop the all-reduce blocks,
and the files must be IDENTICAL line for line. They are (12,273 lines, 161 params), for both the
plain and the accumulation pair. Negative control verified.

### Gradient accumulation — `R34Opt.adamwAccum k`, and it cost ONE op per parameter

A fourth region `[θ|m|v|G]` and two runtime scalars, so **one graph is both phases** — one compile,
one resident parameter set, and no way for an "accumulate" and an "apply" render to drift.

* `Gt = akeep·G + g` is `momVNextF` read the way `.heavyBall` reads it for coupled L2, so **no new
  `SHlo` constructor**. ⭐ `%akeep` is 0 on the first micro-batch of a cycle, so the accumulator
  RESETS by dropping the previous total — there is no zeroing step that could be skipped.
* `%aup ∈ {0,1}` selects the phase by arithmetic: at 0, `β₁ = β₂ = 1` and `(1−β₁) = (1−β₂) = 0`, so
  both moments are exact passthroughs, and `lr = 0` freezes θ COMPLETELY because AdamW's decay is
  DECOUPLED. The 161 per-parameter tails stay byte-identical to `.adamw`'s; the carve-out is eight
  lines of SCALAR arithmetic emitted once.
* ⭐⭐ **`1/k` is folded in ASYMMETRICALLY** — `%ob1` carries `(1−β₁)/k` and `%ob2` carries
  `(1−β₂)/k²`, because `v` is QUADRATIC in the gradient. A shared scale gives the mean of the
  per-micro-batch second moments instead of the second moment of the mean: a different optimizer
  that descends and looks entirely normal.
* ⚠ `fmt12`, not `fmt6`: at k = 4, `(1−β₂)/k² = 6.25e-5`, which `fmt6` emits as `0.000063` — **0.8%
  wrong, baked, in the optimizer.** Same class as §2k's hardcoded label-smoothing mass.

**`lake build r50-accum-tie`** — k micro-steps on the SAME batch must reproduce ONE step of the
committed `adam64` artifact, which was rendered before accumulation existed. Measured:
`[θ|m|v]` **76,671,096/76,671,096 bit-exact** across every accumulate micro-batch, and the apply
ties to rel ≤ **1e-6** on all three regions, against a WRONG-k control that misses by **1778×**.
⚠ Duplicated batch, so it is blind to the combination of DIFFERENT micro-batches, exactly as every
`*-dp-check` is blind to shard offset. The complementary exact check needs a BN-free net (R50
normalises per micro-batch by design — that IS Ghost-BN) and is not built.

⭐ **§4.1 item 2 is discharged BY CONSTRUCTION.** The driver's step loop is unchanged: one iteration
is still one micro-batch, one shim read, one invoke. Only the scalars written per iteration changed,
so the depth-1 prefetch already follows the MICRO-step. What splits is `mstep` (micro-batches — the
augmentation seed, the drop masks, the prefetch) from `gstep` (updates — the LR schedule and Adam's
bias correction). ⚠ `totalSteps`/`warmSteps` are now counted in UPDATES; left in micro-batches the
cosine would run k× too fast.

⚠ `k` is IN THE ARTIFACT NAME (`acc4x64`, `accdp8x64`) and the driver reads it off the same string
that selects the file, because the graph has `1/k` baked and the driver picks the cadence — a
disagreement is silent (a wrong effective LR, no error anywhere). Round-tripped by `#guard` on the
producing side and by `tests/TestVariantPredicates.lean` on the consuming side, which now runs
**55 spellings across 5 axes**.

**Smoked on ImageNet**: 8 micro-batches at k=4 single-GPU, losses 7.44 / 7.65 / 7.78 against §3.3's
7.609 / 7.461 / 7.671, warmup LR correct on the update clock.

### ✅ AND THEN, same session — both of those gaps closed

⭐⭐ **`lake build r50-accum-shard-tie` — the different-micro-batch identity, and it is EXACT.**
The naive complement ("k micro-batches of b == one step at k·b") is false by design: k micro-batches
give k BatchNorm groups where one big batch gives one. ▶ **But R50 already has a render that
computes exactly that grouping — the DATA-PARALLEL one**, whose replicas each normalise over their
own b rows (`shard-check`'s own docstring is where that fact is recorded). So

    acc(x₁..x_k) on 1 device  ==  adamdp([x₁|..|x_k]) on k replicas

exactly, with the two sides reaching it through completely different machinery: a serial accumulator
with a folded `1/k` against a collective `all_reduce` and a divide. Measured **rel ≤ 1e-6 on θ', m'
AND v'** — every region, unlike `shard-check`, which averages two separately-optimised steps and so
can only compare the linear `m`. ⟂ The control is the blind spot itself: the DUPLICATED batch (what
`r50-accum-tie` runs) misses by **17,244× the tie**.

⚠ `PJRT_REPLICAS=4` is required and is not redundant with `CUDA_VISIBLE_DEVICES` — the shim decides
per SESSION (`reps = (g_replicas > 1 && strstr(mlir, "all_reduce")) ? g_replicas : 1`), so setting it
is safe even though half this harness is single-device. Without it the DP invoke refuses outright,
which is the right failure: a silent single-device run would have tied against itself.

✅ **`accdp8x64` smoked at EFFECTIVE BATCH 2048** — 4 replicas × 64 micro × k=8, 16 micro-batches =
2 updates, losses 7.52 / 7.51 / 7.61. RSB-A3's design batch now runs.

### ⛔ What this did NOT do
* **LAMB, BCE-with-logits and 160/224 are still absent**, so this is AdamW at a large batch and must
  not be called `rsb-faithful`. §4's table is otherwise unchanged.
* The two `vjp_oracle` projection cases, now correctly scoped to `MlirCodegen`.

---

## ✅ WHAT LANDED 2026-08-05 (evening) — read this first, then skip to §4

Seven commits on `perf/shim-prefetch`. **§1, §2 and §3 are done.** The next session is §4, and its
one blocker is gradient accumulation.

| | was | now |
|---|---|---|
| **§2 pipeline** | 377 ms/step | **224**, 1.68× — depth-1 prefetch, bit-identity gated |
| **§2.4 uint8** | "the next lever" | ⛔ **dropped, measured** — worth 2.5%, see §2.2 |
| **§1.2 the 6–10× mystery** | cause unknown, blocked 2 nets | ✅ **solved** — it was the producer |
| **enetin** | 2,023 ms/step | **203** (10.2×) |
| **cnxin** | 895 ms/step | **235** (3.8×) |
| **vitin** | 4,348 ms/step | **665** (6.5×) |
| **§1.1 job configs** | 1 of 6 | ✅ **5 of 6**, each with a `SHIM_WORKERS` precheck |
| **§3 ResNet-50** | did not exist | ✅ **trains on ImageNet**, phases 1–3 |
| the whole 80-ep sweep | 792 h ≈ 33 d | **159 h ≈ 6.6 d** |

⚠⚠ **THE ONE DEBT THIS SESSION CREATED, and it must not be inherited quietly: R50's GRADIENT is
not gated.** §3.2's problem is unchanged and §3.3 below spells it out. R50 trains and descends,
and per §6 that is worth nothing on its own.

---

## 0. BASE CAMP — what R34/ImageNet now establishes

**70.735% top-1 / 90.140% top-5** at 30 epochs, 4×bs64, fp32, 15.5 h — above the phase-2
30-epoch reference's 69.26%.

⭐ **The load-bearing result is not the endpoint, it is the epoch-5 cross-check.** Same weights,
same val split, same box, scored by both paths: **17,312 against 17,313 correct out of 49,920 —
one image** — with training losses agreeing to **0.055 nats** at epochs 1, 2, 5 and 6. That is
the strongest statement this repo can make about the verified lowering at scale.

**The eight things that had to be true for that number to exist**, all of which are now in
`main`, and all of which apply to every other net because they live in shared code:

| | commit | scope |
|---|---|---|
| argmax honours the class count | `0a168f0` | driver — **all nets** |
| eval labels decoded full-width int32 | `f42e28a` | driver — **all nets** |
| top-5 exists at all | `16a4f33` | driver — **all nets** |
| init: He fan-out conv / Glorot dense | `fa30fb0` | driver — **all nets** |
| BN running-stat EMA 0.01, not 0.1 | `fa30fb0` | driver — **all nets** |
| strided convs pad symmetrically | `d078a6d` | JAX emitter — ResNet family |
| residency required, enforced by PRECHECK | `ddf8ad0` | **R34's job config only** |
| a DP gate that reads shard content | `c44a758` | **R34 only** |

⭐ **The first five are free for everyone** — they are in the shared driver, so mnv2, enet,
ConvNeXt, ViT and R50 inherit them the moment they run. The last two are per-net and are the
parity work.

⚠ **Nothing retroactive is voided by the eval fixes, and that is luck rather than design**: no
verified *ImageNet* accuracy existed before 2026-08-05 (§0.13 was a 40-step smoke; R34's
30-epoch was blocked on hardware). Imagenette/CIFAR/MNIST numbers are provably untouched —
labels 0..9 fit in byte 0 and 10 classes fit in a 10-wide argmax. **Had any net been run at
ImageNet scale before tonight, its number would have been ~4× low and believed.**

---

## 1. PARITY — what each net needs to reach base camp

| net | job config | DP shard gate | regularisers in the DP artifact | throughput | other |
|---|---|---|---|---|---|
| **resnet34in** | ✅ `r34-imagenet-4gpu.conf` | ✅ `r34-dp-shard` | ✅ | ✅ **224** ms/step (w=1) | **base camp** |
| **mnv2in** | ✅ `mnv2-imagenet-4gpu.conf` | ✅ `shard-check mnv2in` | ✅ | ✅ **201** ms/step (w=1; w=4 is WORSE) | ⚠ 80-ep re-run owed since the conv-bias swap |
| **enetin** | ✅ `enet-imagenet-4gpu.conf` | ✅ `shard-check enetin` | ✅ conf points at `emarmsdp64**dropdo**` | ✅ **203** ms/step (w=8, was 2,061) | |
| **cnxin** | ✅ `cnx-imagenet-4gpu.conf` | ✅ `shard-check cnxin` | ✅ `cnxin_adamdpwxclipdrop` | ✅ **235** ms/step (w=8, was 895); **batch 32 ⇒ 10,009 steps/epoch, double everyone** | ▶ batch-64 rescope is now the top lever — it is compute-bound at 235 |
| **vitin** | ✅ `vit-imagenet-4gpu.conf` | ⛔ **still only `vit-dp-check`, duplicated-batch — structurally blind to shard OFFSET** | ✅ conf points at `adamdp128x4wxclip**drop**` | ⚠ **665** ms/step (w=8, was 4,348) against a **250** floor — still data-bound | ⚠ w=16 is SLOWER (710); 32 cores cannot make the ~2,048 img/s it wants |
| **resnet50in** | ⛔ none | ⛔ none | ✅ AdamW kit | ⛔ unmeasured | ✅ **the net exists and trains** (§3); ⚠⚠ gradient ungated |

### 1.1 ✅ DONE — job configs, and the regulariser gap closed by pointing at the right artifact

Four new configs (`mnv2`, `enet`, `cnx`, `vit`), each carrying the R34 template's residency
precheck **plus a new one for `SHIM_WORKERS`**, whose failure mode is identical: an absent line, a
run that reads completely normal, and up to 10× the wall-clock. Both refusals verified against
negative controls including a wrong-value case.

⭐ **Item 3 needed no `#eval` at all** — `enetin_emarmsdp64dropdo` and
`vitin_adamdp128x4wxclipdrop` were already on disk beside their under-regularised peers. The
configs now name them. ⚠ That is a *file exists* check, not a *bake* check: confirm by listing what
the artifact contains, never off the variant name.

⛔ **Item 2 is still open** — ViT's shard gate is still the duplicated-batch `vit-dp-check`, and
vitin's config launches 4 genuinely-sharded replicas against it. It is the weakest sharding
evidence in the set and the config says so.

### 1.2 ✅ SOLVED — it was the producer, and the fix was a knob that already existed

⚠⚠ **The cause was the data pipeline, not the net, and `SHIM_WORKERS` fixes it.**

`LEAN_MLIR_BENCH_SYNTH` (once §2.2's fix made it mean anything) plus `PJRT_FFI_TIMING` split it in
one run: EfficientNet real **2,061** ms/step against synth **196**, with a *byte-identical invoke* —
188.7 real vs 188.5 synth. So all 1,865 ms was the producer and none of it the net; enet's device
compute is **144.5 ms, cheaper than R34's 188.3**.

⭐ **The three slow nets are exactly the three whose shims run AutoAugment + RandAugment** (plus
erasing on ConvNeXt/ViT, plus repeated augmentation on ViT); R34 and mnv2 do flip only. That is the
whole fast/slow split, and it matches the per-image ordering exactly.

⚠ **Nothing regressed.** `imagenet_sweep.md`'s 310/424 were measured when every net was hardcoded
to R34's *light* shim — the bug fixed 2026-08-02. §0.13's 2,023/4,489 came after each net got its
own *correct* heavy shim, still with one producer. **The pipeline became correct and the producer
count did not follow.**

⚠ §0.13's isolation probes tested dropPath masks and mixup and correctly cleared them — both are
GPU-side, and the cause was on the producer side, which none of those probes touched. ▶ Transferable:
*a probe that clears every hypothesis you had is evidence your hypotheses share a blind spot.*

| net | 1 worker | chosen | floor | 80 ep was → is |
|---|---|---|---|---|
| mnv2 | 201 | **201** (w=1) | 152 | 44 h → 22.4 h |
| enet | 2,061 | **203** (w=8) | 196 | 225 h → 22.6 h |
| cnx | 895 | **235** (w=8) | 214 | 205 h → 52.3 h |
| vit | 4,348 | **665** (w=8) | 250 | 250 h → 37.0 h |

⚠ **It is NOT `tf.data` autoscaling doing this.** `num_parallel_calls=AUTOTUNE` leaves one producer
at ~3 of 32 cores; scaling is near-linear across **processes**, so the worker count is a number
someone has to set per net — which is why it belongs in a config with a precheck rather than in a
default. A C rewrite of the augmentation would attack the constant factor on kernels that are
already compiled TF ops, i.e. the part that is least broken.

⚠ **ViT stays data-bound** at 665 against a 250 floor, and **16 workers is slower than 8** (710).
It needs ~2,048 img/s of RandAugment+erasing+repeated-aug and this box cannot produce it. That is
hardware, not a bug. ⚠ uint8 does not help it either — the constraint is CPU augmentation, not
transport.

▶ **Unexamined and worth a look:** ViT's LAUNCH time is **118 ms** against R34's 9.4 — 47% of its
non-data step.

---

## 2. THE DATA PIPELINE — helps every net, and it is small

**Why before R50:** RSB-A3 is **100 epochs**. R34 took 15.5 h for 30. A ~1.8× that is already
understood multiplies across every net and every future run.

### 2.1 What is wrong

The step is two blocking calls back to back — `readShimBatchRR`, then the invoke — and nothing
drains the pipe during compute. ⭐ **A batch is 154 MB; a pipe's buffer is 64 KB** (this box caps
`pipe-max-size` at 1 MB, still 0.6% of a batch). The producer fills 64 KB, blocks in `write()`,
and sleeps through the entire compute. It measures **258% CPU on a 32-core box at load 5.6** — it
is not slow, **we are throttling it**.

⚠ Which is why "the producer does 1,530 img/s and R34 needs 380" was never the relevant
comparison. Capacity is irrelevant when the consumer pulls one batch and walks away.

### 2.2 The measured target

| | ms/step |
|---|---|
| verified, fp32, resident | **368** (measured, steps 200–1200) |
| JAX fp32, same 4 cards | ~201 (recorded) |
| JAX bf16, same 4 cards | **101** (measured 2026-08-05: 503.8 s/epoch ÷ 5004) |

The generated JAX trainer calls `prefetch_to_device(it, sharding, depth=2)`. **368→201 is
transport we don't overlap and they do.** 201→101 is bf16, a separate axis.

#### ✅ MEASURED 2026-08-05 — the split is real, and it is not where this file put it

`LEAN_MLIR_BENCH_SYNTH=1` **did not work on ImageNet** until today: `imgStreams` was spawned on
`net.data == .imagenet` alone, so a "synthetic" run still did the full 154 MB blocking pipe read
every step. On every other dataset synth replaces a preloaded host array; ImageNet never had one,
so the flag replaced nothing and announced nothing. Fixed in `VerifiedTrain.lean` (skip the spawn
under synth, announce it, and give `mkSynthData` a real `.imagenet` case — it had been falling
through to **mnist's 60,000**, making a synthetic epoch 234 steps where the real one is 5,004).

Same binary, same config (4×bs64, momdp64, resident, fp32, devices 0,2,3,4), `MAX_STEPS=120`:

| | ms/step | what it is |
|---|---|---|
| real, shim on | **377** | `t_read + t_rest` |
| synth, shim off | **219** | `t_rest` (identical at `MAX_STEPS=40`) |
| **difference** | **158** | **`t_read`** |

⭐ Validated against production: 5004 × 377 ms × 30 ep = **15.7 h** against the run's actual
**15.5 h**. The probe tracks the in-run figure (368, steps 200–1200) to 2.4%.

**What this changes:**

1. ⭐ **The prefetch ceiling is `max(158, 219)` = 219 ms/step — 377 → 219, `1.72×`**, i.e. 30
   epochs 15.7 h → **9.1 h**. Close to this file's 1.8× guess, but now a bound rather than a hope.
2. ⭐ **`t_read < t_rest`, so one producer already has the slack.** The read needs 158 ms and
   compute gives it 219 — confirming `SHIM_WORKERS>1` does not bind for R34, and now by
   measurement rather than by the img/s argument §2.1 says was never the relevant one.
3. ⚠⚠ **The decomposition above — "~105 compute · ~5 params · ~250 image path" — is WRONG, and
   the error matters.** The un-hidable floor is **219**, not ~110. Device compute is ~105 (the
   residency measurement's own control), so **~114 ms of the floor is host-side work plus the
   154 MB H2D of `x`** — which lives *inside* `mlpTrainStepVDP` and which prefetching the **pipe**
   does not touch. Overlap cannot go below 219; only shrinking or overlapping the H2D can.
4. ▶ **Therefore uint8 (§2.4) is worth more than "then", and for a second reason.** It cuts the
   158 ms read *and* the H2D term inside the 219 ms floor — the only lever identified that touches
   the post-overlap floor at all. It stays second in ORDER (overlap first, it is cheaper and
   gateable on bit-identity), but it is not a cleanup item.

#### ✅ AND THEN MEASURED — the 219 splits, and item 3 above was wrong too

`PJRT_FFI_TIMING=40` (already in `ffi/pjrt_ffi.c`, opt-in, zero cost when unset) reports the invoke
directly. R34/ImageNet 4×bs64, resident, prefetch ON, steps 81–120:

| term | ms | share |
|---|---|---|
| **device compute** | **188.3** | **86%** |
| one-hot h2d (4 × 0.25 MB) | 11.3 | 5.2% |
| launch | 9.5 | 4.3% |
| **image h2d (147 MiB)** | **7.4** | **3.4%** |
| d2h | 2.4 | 1.1% |
| = invoke | 218.5 | (driver step 224) |

⚠⚠ **This kills the uint8 case, and it kills the paragraph I wrote three hours earlier saying
uint8 "hits the floor twice".** The 147 MiB image h2d is **7.4 ms — 3.4% of the step** — and it
moves at **20.4 GiB/s, PCIe 4.0 x16 line rate**. It is not a bottleneck; it is already optimal.
Cutting the bytes 4× saves ~5.5 ms of a 219 ms step: **≈2.5%**. Against that: a changed wire
protocol, a changed shim (shared with the JAX reference), normalize moved into the certified
renderer, and the G4 gated interface. ▶ **Do not build it for R34.**

⚠ The whole host↔device budget is **21 ms = 9.6%**. That is the hard ceiling on *every* transfer
optimisation combined, uint8 included. §2.1's framing — that the image path was the dominant term
residency had exposed — was right about the PIPE READ (158 ms, now hidden) and wrong about the
TRANSFER. Two different 154 MB costs, and only one of them was ever real.

⭐ **The surprise: the one-hot costs more than the image.** 4 × 0.25 MB takes **11.3 ms** while
147 MiB takes 0.2 ms to issue. Not back-pressure — **reversing the issue order so the one-hot goes
first left it at 11.6 ms**, and not bytes, since x is 600× larger for 1/50th the issue cost. So
~5% of the step is spent inside `BufferFromHostBuffer` on four small buffers, cause unknown.
▶ That is a bigger prize than uint8's entire ceiling and it needs no interface change; the likely
route is the residency mechanism (a retained buffer updated in place) rather than a fresh
allocation per step.

▶▶ **The real lever is bf16, and it is the only one that touches the 86%.** §2.2's own table has
JAX at 201 ms fp32 → **101 ms bf16 on these same cards**. Our 188 ms of device compute is
consistent with JAX's 201 ms fp32 step, which is the cross-check that says the compute is not
anomalous — it is simply what this net costs in fp32 here.

⚠ The 188 ms does NOT reconcile with the residency note's "compute UNCHANGED (105.6 → 104.8 ms)"
recorded for "exactly this shape". One of the two is a different configuration; the JAX agreement
above says today's number is the trustworthy one. Unresolved, and flagged rather than smoothed.

### 2.3 The change

## ✅ BUILT AND GATED 2026-08-05 — **377 → 224 ms/step, 1.68×**

`LeanMlir/VerifiedTrain.lean`, depth-1 double buffer, **`LEAN_MLIR_PREFETCH` default ON** with
`=0` as the gate's control. 30 epochs of R34/ImageNet: **15.7 h → 9.3 h**. Landed 5 ms above the
219 ms ceiling §2.2 measured, and that residual is the real path allocating a fresh 154 MB
`ByteArray` per step where the synth floor reuses one buffer.

⭐ **`Task.Priority.default` (the pool), not `.dedicated` — worth 12 ms/step** (236 → 224). The
usual advice for a blocking read is `.dedicated`, so a long `read()` cannot occupy a pool worker
and starve other tasks. That reasoning does not apply at depth 1 — there is **exactly one
outstanding task by construction**, so there is nothing to starve — while `.dedicated` spawns a
fresh OS thread every step, 150,120 of them over a 30-epoch run. The doc's sketch (and the first
implementation) had `.dedicated`; it was a 5% error.

**How depth 1 is actually bought:** read `i+1` is issued *after* the wait on read `i` and *before*
the invoke. After-the-wait is the correctness condition (one read outstanding ⇒ one thread on the
handles, in issue order); before-the-invoke is the entire point. Two concurrent reads on one pipe
would interleave and corrupt both batches, and the resident path's `res_gen` requires strict step
order. ⚠ The prefetch is skipped on the last step of the last epoch — otherwise a pool worker is
left blocked in `read()` on a live producer while `main` returns. The guard spans the epoch
boundary rather than stopping at `bi + 1 < nb`, so the overlap survives it: `ep*nb + bi + 1` at the
end of epoch e is exactly `ep'*nb + 0` for e+1.

### The gate — `tests/prefetch_tie.sh`, PASS

Three runs: A1/A2 both OFF (**the control**), B ON (the verdict), `[θ|m|v]` dumps compared
bit-for-bit over 2 epochs × 12 steps so it **crosses an epoch boundary** — the one place the index
arithmetic can be wrong on its own. **Control 0 bytes, verdict 0 bytes.**

⚠⚠ **The control failed on the first attempt and that is the finding worth keeping.** A1 vs A2
differed in **145,301,829 of 261,572,064 bytes** — two runs of the *identical* path. Cause: XLA
autotuning picks convolution algorithms per process on CUDA, exactly what `scripts/det_shim.sh`
exists for and records (191,094,739 of 255,477,624 at 10 steps). With the deterministic shim on
`LD_LIBRARY_PATH` the floor is 0 and the gate reads. **Had the gate been written as two runs
instead of three, it would have reported a 145 MB "failure" of a change that is bit-exact** — or,
run once in the other order, a green that meant nothing. §6's lesson, paid immediately.

⚠ `LEAN_MLIR_BENCH_SYNTH` must NOT be set for this gate — it now skips the shim spawn, so there
would be no read to prefetch and the gate would compare a path against itself and pass forever.

⚠ Still open, both deliberately: **`SHIM_WORKERS>1` keeps depth 1**, where the natural depth is n
(one in flight per handle); and the non-shim datasets are untouched — Imagenette/CIFAR augment
host-side off `augSeed` in the loop and have no pipe to drain.

### 2.4 Then

* ⛔ **uint8 wire — MEASURED AND DROPPED for R34.** 154 MB → 38.5 MB/step buys **≈2.5%**: the image
  h2d is 7.4 ms of a 219 ms step and already runs at PCIe line rate (§2.2). Cost is a changed wire
  protocol, a changed shim, normalize inside the certified renderer, and the G4 gated interface.
  ⚠ **Revisit only after bf16.** At 101 ms of compute the 158 ms pipe read no longer hides behind
  it, and the arithmetic inverts — this is a conditional "no", not a permanent one.
* ▶ **The one-hot's 11.3 ms** (§2.2) — 5% of the step, no interface change, bigger than uint8's
  entire ceiling. The first thing to try on this path.
* **`SHIM_WORKERS>1`** — built, one env var, defaults to 1. 2 workers **1.71×**, 4 **2.36×**. The
  **ViT** lever; does not bind for R34 — measured, the read has 61 ms of slack (158 vs 219).

---

## 3. RESNET-50 — the one net that must be built

`rsb_a3_r50_verified.md` §1 measured **zero new SHlo ops** and the param count **exact on the
first try** (25,557,032). Still true. What exists: `VLayer.bottleneckStage`, the layout specs, the
`maxPool3s2` witness + codegen, and (new) the shim. What does not: the block VJPs, the renderer,
any artifact.

### 3.1 ✅ Phase 1 DONE — `LeanMlir/Proofs/Foundation/Resnet50BlocksCertified.lean`

All three forms, forward + certified VJP, **3-axiom clean** (`propext, Classical.choice,
Quot.sound`). Compiled first pass.

| form | where | name |
|---|---|---|
| identity bottleneck | 12 blocks | `bblkPC` / `bblkPC_has_vjp_at` |
| strided projection | stages 2/3/4 block 0 | `bblkPStridedPC` / `…_has_vjp_at` |
| ⭐ **stride-1 projection** | **stage 1 block 0 ONLY** | `bblkPProjPC` / `…_has_vjp_at` |

⭐ **Zero new foundation, and that is §1's op-cost collapse showing up on the proof side.**
`convBnReluPC_has_vjp_at`, `flatConv_has_vjp`, `flatConvStride2_has_vjp`,
`bnPerChannelTensor3_has_vjp` and `residualProj_has_vjp_at` were **already generic** in
`{ic oc h w kH kW}` / `{m n}`, so the bottleneck's third conv is one more `vjp_comp_at` link.

Confirmed while building: `rblkPStridedPC` reads `Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w)`, so the
halving is in the **type**. Kernel extents stayed binders, so 1×1-vs-3×3 is an argument.

Also carries a wiring section pinning the blocks at R50's real dims and composing whole stages —
it catches the bottleneck's easiest inversion, that an identity block's `mid` is a **quarter** of
its `c` (256 with mid 64), not a multiple.

### 3.2 ✅ Phase 2 DONE — `LeanMlir/Proofs/Codegen/ResNet50RenderB.lean`

Four artifacts, all from one renderer: `resnet50in_{adam64,adamdp64}_train_step.mlir` (594 in / 592
out) and `resnet50in_fwd{,_eval}.mlir` (162 / 268 in). Both forwards share ONE chain via `bnSite`'s
train/eval switch, so the scored net and the differentiated net cannot drift — §2g's
`mobilenetv2_fwd` defect. `optOne`/`optConstsB`/`bnSite` went private→public rather than being
copied; no emitted bytes change and R34's artifacts re-render byte-identical (checked).

✅ **`tests/TestR50Contract.lean`** gates the layout: **161 tensors**, **25,557,032 params** (exact,
torchvision's published count), **shapes elementwise** against the spec the *driver* walks. Verified
non-blind by cutting stage 3 to 5 blocks.

### ✅ PAID 2026-08-05 (late) — `lake build r50-gradcheck`

The debt below is discharged; the two checks it names are **not** how, and one of them would have
gated the wrong emitter. See "WHAT LANDED (late)" at the top. Kept verbatim because the reasoning
about *what licenses a render with no incumbent* is still the standing rule.

⚠⚠ **BUT §3.2's original warning still stands and is now a DEBT, not a plan.** There is no incumbent
R50 render, so the bit-exact-tie license every other swap had does not exist. **The gradient is
ungated.** Before any number is quoted:

* ⛔ `tests/vjp_oracle/run.sh bottleneck` covers the **identity** block only — its case is
  `.bottleneckBlock 8 8 1 1`, **no projection**. It covers neither projection form and specifically
  **not the stride-1 one**. Two new oracle cases would close that.
* ⛔ Nothing exercises the **whole-net wiring**. The check that would is a loss-sequence agreement
  against `jax/MainResnet50Imagenet.lean` at matched init — the evidence class R34's own ImageNet
  number rests on, and the machinery exists (the oracle already does init-dump/init-load + NO_SHUFFLE).

**Say which one licensed the swap.** Neither has been run.

### 3.3 ✅ Phase 3 DONE (partly) — it trains

`resnet50-imagenet-verified-xla` + `apps/imagenette/Resnet50ImagenetCommon.lean`. Single-GPU smoke:
483 resident tensors (292.5 MB), 53 BN layers / 53,120 stat floats, losses **7.609 / 7.461 / 7.671**
against ln(1000) = 6.91 — the excess is label smoothing plus this net's mixup soft targets (R34
starts at 7.14 with mixup off).

⛔ Still owed from this phase: `resnet50-dp-shard` (copy `TestR34DpShard.lean`, change the slug),
`residency_gate_all.sh` with the right fault mode, a job config, a 4-GPU smoke, and a ms/step
measurement — R50's throughput is **unmeasured**.

▶ Fixed on the way: under `LEAN_MLIR_SKIP_EVAL` the driver printed
`acc = 0/49920 = 0.000000%  top5 = 0/49920` on a run that scored nothing — indistinguishable from a
catastrophically broken net, and on R50's first run that is exactly how it read. Exact zeros on
**both** top-1 and top-5 are the tell (chance at 1000 classes is ~50 and ~250). It now says eval was
skipped.

**Phases 1–3 give a verified R50 that exists, trains and is gated at the certified AdamW kit —
real value even if §4 never runs.** ⚠ "Gated" there means the LAYOUT. See §3.2.

---

## 4. ▶▶ RSB-A3 — **THE NEXT SESSION.** Gradient accumulation is the blocker

JAX already has the recipes (`jax/MainResnet50Imagenet.lean`): `rsb-faithful`, `true-2048`,
`a2-accum`, `a1`, `adam-probe`, and `rsb-faithful` has already run: **76.66% top-1 @ ep100**.

| piece | cost | state |
|---|---|---|
| ✅ **gradient accumulation** | ~~driver work~~ | ✅ **BUILT AND CERTIFIED 2026-08-05 (late)** — `R34Opt.adamwAccum k`, a fourth `[θ\|m\|v\|G]` region, one op per parameter, `lake build r50-accum-tie`. No longer the blocker |
| **LAMB** | 2–3 ops *estimated* | ⚠ **the one number in `rsb_a3_r50_verified.md` §2.3 that is NOT a measurement.** Cost it the §0.8 way first. ⚠ LAMB **breaks every gate that recovers `g` from `m'`** — `r34-mom-tie`, `rms-tie`, `wdx-tie`, `shard-check` |
| **BCE with logits** | ~1 descriptor | `BatchableOp.sigmoid` is pointwise + batch-invariant ⇒ legal descriptor. `sigmoidF` exists with a global hypothesis-free `sigmoid_has_vjp`; the cotangent `σ(z) − t` is CE's with one op swapped |
| **160 train / 224 eval** | two artifacts | ⚠ the §2g prefix audit (`_fwd` ⊂ `_train_step`) **cannot hold across them**. Decide explicitly; do not let it degrade to "the audit doesn't cover R50" |
| ✅ **mixup / cutmix** | **NOT A GAP — this row was stale** | corrected 2026-08-05 |

⭐ **The mixup row was wrong and cost nothing only because nobody acted on it.** It claimed "no
`softLabelCE` cotangent in any renderer — `grep` finds none". `lean_exe soft-target-tie` /
`tests/TestSoftTargetTie.lean` exist **specifically to retire that claim**: every render takes the
target as a float `[batch, nClasses]` and the emitted cotangent is affine in it, so
`grad(λ·y_a + (1−λ)·y_b) = λ·grad(y_a) + (1−λ)·grad(y_b)` holds **by construction**, gated on the
committed bytes. Producer half ✅, wire v2 ✅ — and R50's smoke already streamed `SHIM_MIX=both`.
▶ *A "grep finds none" is evidence about a name, not about a capability.*

**Why grad accum is the blocker.** `rsb-faithful` is `gradAccumSteps := 4` — 512 micro × 4 =
effective **bs2048**, LAMB's design batch, at `learningRate := 0.008` (the bs2048 rate, explicitly
*not* the 512-scaled 0.002). And `rsb_a2_resnet50.md` records **LAMB at bs512 giving 40.8%
against 78.1%**. ⚠ That finding is about batch size, not the stem pool, so it survives the §4b
re-runs.

| route | cost | buys |
|---|---|---|
| **grad accum in the driver** | accumulate-then-step + a BN-regime decision (Ghost-BN, as JAX took) | the honest pair |
| **real bs2048** | ~80 GB activations at 160px — **this box cannot** | cleanest reproduction, ungated here |
| **accept bs512** | free | the **40.8%** regime: a valid *engineering* pair, a useless *accuracy* one |

▶ **Decide before phase 4, not during it.** ✅ **DECIDED AND BUILT: grad accum in the driver**, with
Ghost-BN inherited rather than chosen (each micro-batch normalises over itself, which is what the
per-micro-batch forward does and what JAX took). ⚠ The DP shard gate's argument now has that second
axis and its text has not been updated — §4.1 item 3's warning is still open.

### 4.1 What the grad-accum session should know before it starts

1. ⚠⚠ **Settle §3.2's gradient debt first, or at least decide not to.** RSB-A3 is a ~1.5–2 day fp32
   run on an R50 whose gradient has never been checked against anything. Two oracle cases plus a
   matched-init loss tie is a few hours; discovering the net was wrong after two days is not.
2. ✅ **DISCHARGED BY CONSTRUCTION.** The accumulation implemented 2026-08-05 changes no loop
   structure at all: one iteration is still one micro-batch, one shim read, one invoke, and only the
   scalars written per iteration differ. So the depth-1 prefetch already follows the MICRO-step, and
   `tests/prefetch_tie.sh` needs no extension to remain valid. ⚠ What DID split is the step counter:
   `mstep` (micro-batches — the augmentation seed, the drop masks, the prefetch index) from `gstep`
   (updates — the LR schedule and Adam's bias correction), and `totalSteps`/`warmSteps` moved to the
   update clock. Left in micro-batches the cosine runs k× too fast.

   *Original warning, kept because it is what the design had to satisfy:* §2.3's depth-1 prefetch
   means the shim read is issued *inside* the step, one in flight, with a strict-order requirement
   (`res_gen`); the prefetch index must follow the MICRO-step or the pipeline desynchronises.
3. **BN is the real design question, not the accumulation.** Accumulating gradients is
   arithmetic; deciding what BN normalises over across 4 micro-batches is a modelling choice. JAX
   took Ghost-BN. ⚠ Whatever is chosen, the DP shard gate's argument (`N×b ≠ 1×(N·b)` by design)
   now has a second axis and the gate text needs to say which.
4. **fp32 is the regime.** bf16 is untouched (§2.2 item 4: it is the only lever on the 86% that is
   device compute) so activation memory is what it is — which is exactly why grad accum is the
   route and real bs2048 is not.
5. Wall-clock to budget: R50/A3 at 160px, 100 epochs, extrapolated off R34's measured 224 ms/step,
   is roughly **1.5–2 days**. ⚠ Extrapolated, not measured — **R50's ms/step is still unmeasured**
   (§3.3). Measure it first; it is one 40-step probe.

---

## 5. HOW CLOSE

**Parity: ✅ done, and it was one knob.** All five configs exist, and the 6–10× "comprehension
problem" that gated two nets turned out to be a producer count. ⛔ Left: ViT's shard gate, and
ConvNeXt's batch-64 rescope (now the largest single lever on the sweep — it is compute-bound at
235 ms/step, so halving its step count per epoch should nearly halve its wall-clock).

**A plain verified R50 (§3): ✅ it exists and trains.** The estimate was 3–5 sessions with a note
that three of that doc's own estimates came in low-side-wrong; it took **one**, because the op-cost
collapse §1 measured turned out to hold on the proof side too — every underlying VJP lemma was
already generic, so three block VJPs compiled on the first pass. ⚠ **The estimate was low-side-wrong
in the OTHER direction this time, and that is worth remembering with suspicion rather than
satisfaction: what came in fast was the part with a template. The part with no template — licensing
a render with no incumbent — is untouched.**

**RSB-A3: still not close, and the gap is still the recipe.** But it is one item smaller than this
file claimed: mixup was never blocked. What remains is grad accumulation (driver work, not written),
LAMB (an estimate the doc flags as its only non-measurement), and the 160/224 audit decision. Plus
100 epochs is a ~1.5–2 day fp32 run — which was the argument for §2 first, and §2 is now done.

⚠ **The claim ceiling does not move for any of it, and R50 sits BELOW R34 on it.** R34/ImageNet has
a pair agreement (0.055 nats, and 17,312 vs 17,313 on the same weights). R50 has a *layout* gate and
a smoke. Until §3.2's debt is paid, the honest sentence about R50 is "it renders from the certified
renderer and it runs" — not "one architecture, two independent lowerings, agreeing", which is R34's
sentence and has to be earned separately.

⚠ **The claim ceiling does not move for any of it.** The proof-carrying tier stops at Imagenette.
ImageNet inherits *provenance* plus whatever a pair agreement shows — "one architecture, two
independent lowerings, agreeing", never "proven".

---

## 6. THE LESSON FROM 2026-08-05, because it will apply again

Five defects in one session. Two were the **same defect wearing different clothes** — argmax over
10 of 1000 logits, and labels read one byte wide — in the same scoring loop, both **correct on
every net the repo gates** (Imagenette/CIFAR/MNIST are all 10-class) and wrong on exactly the tier
with no accuracy gate.

⭐ **The signal that mattered was not the accuracy gap — it was that the TRAIN LOSSES AGREED.**
They matched JAX to 0.055 nats from epoch 1 while top-1 was 4× off. Four investigations went at
the accuracy gap directly (DP sharding, He-init mode, BN momentum, conv padding); three found
real-but-unrelated bugs and **none was the bug**. Loss agreeing while accuracy did not meant the
defect was **downstream of the weights** — true and visible from epoch 1.

▶ **Transferable form: when two paths disagree, find the earliest quantity that AGREES, and look
strictly downstream of it.**

And: **a setting with no output and no gate is a setting that can be silently wrong.**
`PJRT_FFI_RESIDENT` announced itself when ON and printed *nothing* when OFF — which is how a
benchmark and its production config diverged for a week. ⚠ In this session two monitors and one
probe *also* failed silently, each caught only by its own control. **Build the control before
trusting the green.**
