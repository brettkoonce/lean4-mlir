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
| **resnet34in** | ✅ `r34-imagenet-4gpu.conf` | ✅ `r34-dp-shard` | ✅ | ✅ 368 ms/step | **base camp** |
| **mnv2in** | ⛔ none | ✅ `shard-check mnv2in` | ✅ | ✅ 396 ms/step | ⚠ 80-ep re-run owed since the conv-bias swap |
| **enetin** | ⛔ none | ✅ `shard-check enetin` | ⛔ **`enetin_emarmsdp64` has NEITHER dropPath NOR classifier dropout** | ⚠⚠ **2,023 ms/step vs 310 recorded** | |
| **cnxin** | ⛔ none | ✅ `shard-check cnxin` | ✅ `cnxin_adamdpwxclipdrop` | ⚠ 922 ms/step; **batch 32 ⇒ 10,009 steps/epoch, double everyone** | batch-64 scoped, not built (§2p: "the single best pre-run optimisation available") |
| **vitin** | ⛔ none | ⛔ **only `vit-dp-check`, duplicated-batch — structurally blind to shard OFFSET** | ⛔ **`vitin_adamdp128x4wxclip` has NO dropPath** | ⚠⚠ **4,489 ms/step vs 424 recorded** | needs `SHIM_WORKERS≥2` (wants ~1,940 img/s, one producer gives ~1,530) |
| **resnet50in** | ⛔ none | ⛔ none | — | — | ✅ shim wired 2026-08-05; **the net itself does not exist** (§3) |

### 1.1 The cheap parity items, in the order I would do them

1. **Job configs for the other five.** `scripts/jobs/r34-imagenet-4gpu.conf` is the template and
   it now carries the `PJRT_FFI_RESIDENT=1` line **and a PRECHECK that refuses to launch without
   it**. ⚠ Copy the precheck too — the flag's failure mode is an *absent* line in a 5,000-line
   log, which is how a 16 h benchmark and a 26 h reality diverged for a week.
2. **ViT's shard gate.** `vit-dp-check` hands both replicas the *same* rows, so a shard-offset bug
   leaves the halves identical and it passes bit-exact. It establishes "the collective averages",
   **not** "the replicas saw different data". Either add `vitin` to `TestShardCheck`'s table (it
   needs a single-device render at the same per-replica batch) or copy `TestR34DpShard.lean` for
   the weaker discrimination form.
3. **The two regulariser-behind DP artifacts.** One `#eval` each — `replicas` is already a
   parameter on both renderers — plus a `drop-shard-check` run, which transfers unchanged. ⚠ A
   4-replica ViT or EfficientNet run today silently trains **without** the regularisers its
   reference sets. The check that finds this is: *list what the artifact bakes*, never read the
   recipe matrix.

### 1.2 The one that is not cheap, and blocks two nets

⚠⚠ **EfficientNet and ViT measure 6–10× their recorded ImageNet ms/step and the cause is not
found.** `imagenet_sweep.md` says enetin 310 / vitin 424; §0.13 measured **2,023 / 4,489** on the
current renders. The obvious hypothesis — that the new regularisers caused it — was **tested and
refuted**: masks cost nothing (2,161 without vs 2,023 with), mixup ~10% on ViT only.

Per image the split is stark: R34 2.41 ms and mnv2 1.55 ms (neither carries a feature added that
session) against ConvNeXt 7.20, EfficientNet 7.90, ViT 8.77 (all three do).

▶ **Do not budget a run for either net off either number.** This is the first job for those two,
ahead of any parity work, because you cannot tell whether §2 fixes it if you don't know what it is.

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

▶ **Next measurement:** split the 219 into compute / H2D / host blob patching. Item 3 is arithmetic
off a borrowed compute figure, not a measurement, and now that the prefetch has landed (§2.3, 224
ms/step) it is **the** binding term — every remaining lever on this path aims at it.

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

* **uint8 wire** — 154 MB → 38.5 MB/step. *After* overlap, since overlap hides latency and what
  remains exposed is bandwidth. Compatible with the one-definition rule **iff** only the affine
  normalize moves to device. ⚠ Changes a gated interface (the preamble's G4 check).
  ▶ **Now the only identified lever on the 224 ms floor, and it hits that floor twice** — §2.2's
  measurement puts the 154 MB H2D of `x` and a 154 MB host allocation *inside* the un-overlappable
  term, and uint8 cuts both 4×. This is no longer a cleanup item.
* **`SHIM_WORKERS>1`** — built, one env var, defaults to 1. 2 workers **1.71×**, 4 **2.36×**. The
  **ViT** lever; does not bind for R34.

---

## 3. RESNET-50 — the one net that must be built

`rsb_a3_r50_verified.md` §1 measured **zero new SHlo ops** and the param count **exact on the
first try** (25,557,032). Still true. What exists: `VLayer.bottleneckStage`, the layout specs, the
`maxPool3s2` witness + codegen, and (new) the shim. What does not: the block VJPs, the renderer,
any artifact.

### 3.1 Phase 1 — three bottleneck block VJPs (CPU proof work)

⚠⚠ **THREE forms, and the third is the one a first look misses:**

| form | where | exists? |
|---|---|---|
| identity bottleneck | 12 blocks | new (3-conv peer of `rblkPC`) |
| strided projection | stages 2/3/4 block 0 | new (3-conv peer of `rblkPStridedPC`) |
| ⭐ **stride-1 projection** | **stage 1 block 0 ONLY** | ⛔ **no analogue anywhere** |

R34 never needed it (stage 1 is `ic = oc = 64`). R50's stage 1 goes 64→256 at stride 1.
`rblkPStridedPC` hardcodes `flatConvStride2` *and* bakes the `(2*h)`/`(2*w)` index — a different
type, not a different argument.

⚠ **Build the stride-1 projection FIRST.** Reaching for the strided form there is a *shape* error
and fails loudly; an identity skip is a **silent wrong net that trains and descends**. After
2026-08-05, "trains and descends" is worth nothing.

⚠ **The stride is on the 3×3** (v1.5/torchvision), not the leading 1×1. Backwards compiles,
trains, descends, and is a different net — worth ~0.5 pt.

### 3.2 Phase 2 — `ResNet50RenderB.lean` at `N := B`

⚠ **No incumbent hand-written R50 render to tie against.** Every other net's swap was licensed by
a bit-exact numeric tie; this one cannot be. Substitutes: the layer-level VJP oracle
(`tests/vjp_oracle/run.sh`) and a keep-1 known-answer check. **Say which one licensed the swap.**

### 3.3 Phase 3 — scale, DP, residency

Param count must land on **25,557,032**. Then a `resnet50-dp-shard` (copy `TestR34DpShard.lean`,
change the slug), `residency_gate_all.sh` with the right fault mode, a job config with the
residency precheck, and a 40-step smoke.

**Phases 1–3 give a verified R50 that exists, trains and is gated at the certified AdamW kit —
real value even if §4 never runs.**

---

## 4. RSB-A3 — the recipe, and the blocker that is not code

JAX already has the recipes (`jax/MainResnet50Imagenet.lean`): `rsb-faithful`, `true-2048`,
`a2-accum`, `a1`, `adam-probe`.

| piece | cost | state |
|---|---|---|
| **BCE with logits** | ~1 descriptor | `BatchableOp.sigmoid` is pointwise + batch-invariant ⇒ legal descriptor. `sigmoidF` exists with a global hypothesis-free `sigmoid_has_vjp`; the cotangent `σ(z) − t` is CE's with one op swapped |
| **LAMB** | 2–3 ops *estimated* | ⚠ **§2.3: the one number in that doc that is NOT a measurement.** Cost it the §0.8 way first. ⚠ LAMB **breaks every gate that recovers `g` from `m'`** — `r34-mom-tie`, `rms-tie`, `wdx-tie`, `shard-check` |
| **160 train / 224 eval** | two artifacts | ⚠ the §2g prefix audit (`_fwd` ⊂ `_train_step`) **cannot hold across them**. Decide explicitly; do not let it degrade to "the audit doesn't cover R50" |
| **mixup / cutmix** | ⛔ cotangent missing | producer half ✅ (R50's shim defaults `SHIM_MIX=both`), driver speaks wire v2 ✅ (`nclasses > 0` ⇒ `SHIM_NCLASSES`). **No `softLabelCE` cotangent in any renderer** — `grep` finds none |
| ⛔⛔ **gradient accumulation** | driver work | **`grep gradAccum LeanMlir/VerifiedTrain.lean` returns 0** |

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

▶ **Decide before phase 4, not during it.**

---

## 5. HOW CLOSE

**Parity for the other four nets: close, and mostly mechanical.** Job configs from a template,
two `#eval`s for the regulariser gap, one shard gate for ViT. ⚠ Except the 6–10× throughput
mystery, which is a comprehension problem and gates two of them.

**A plain verified R50 (§3): genuinely close.** Zero new ops, exact param count, shim wired,
scale path proven, every gate transfers by changing a slug. The work is three block VJPs and a
renderer. §6 of the R50 doc estimates **3–5 sessions** and notes three of its own estimates came
in low-side-wrong in the same direction.

**RSB-A3 specifically: not close, and the gap is the recipe, not the net.** Grad accumulation is a
driver feature that has not been written, mixup needs a cotangent that does not exist, and LAMB's
cost is an estimate the doc itself flags. Plus 100 epochs is a multi-day run — which is the
argument for §2 first.

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
