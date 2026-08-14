# ViT-S/B + ConvNeXt-S/B on ImageNet — what the scale-up actually costs

**Question asked:** to add the S/B variants of the two ImageNet demos, is it
enough to change the model parameters, or is something else needed?

**Answer:** for the *model*, yes — the specs are pure parameter changes and the
phase-2 JAX emitter handles all four without a line of codegen work. Everything
that needed doing was infrastructure around the runs, listed at the bottom.

Measured 2026-07-25 on ares (6× RTX 4060 Ti, 16 GB; JAX 0.10.2 CUDA).

---

## STATUS (2026-08-14) — the two sides are at different places

⚠ **This doc was written about the JAX (phase-2) side and everything below the
next section is about THAT side.** All four JAX trainers landed in `f35c4d6`
(2026-07-26). The VERIFIED (Lean render → StableHLO → PJRT) side is newer and
only half done:

| model | JAX phase 2 | verified / PJRT |
|---|---|---|
| ViT-Ti | ✅ | ✅ `vitin` |
| **ViT-S** | ✅ Jul 26 | ✅ **Aug 14** `vitsin` |
| **ViT-B** | ✅ Jul 26 | ✅ **Aug 14** `vitbin` (global 128 only, see §Verified) |
| ConvNeXt-T | ✅ | ✅ `convnextin` |
| **ConvNeXt-S** | ✅ Jul 26 | ✅ **Aug 14** `convnextsin` |
| **ConvNeXt-B** | ✅ Jul 26 | ✅ **Aug 14** `convnextbin` |

⭐ **All four are done.** ConvNeXt-B was the last one and the only one that needed
the dimension literals threaded; everything below the ViT section is the record.

▶ **The JAX answer ("pure parameter changes") did NOT transfer to the verified
side unmodified**, because the verified renderers hardcode their dimensions where
the JAX emitter derives them. It transferred for ConvNeXt-**S** (pure depth, so it
never touches a dimension literal); ConvNeXt-**B** is the case it does not cover,
and closing it cost ~27 literals across the two renderers. ⭐ **The final answer,
now that all four are done: the model is free, the RENDERER is the work, and the
work is proportional to how many dimensions move — not to how many parameters.**

## The specs are parameter-only

| new file | change vs the existing demo |
|---|---|
| `jax/MainVitSImagenet.lean` | `patchEmbed 3 384 16 196`, `transformerEncoder 384 6 1536 12` (Ti: 192/3/768) |
| `jax/MainVitBImagenet.lean` | `patchEmbed 3 768 16 196`, `transformerEncoder 768 12 3072 12` |
| `jax/MainConvNeXtSImagenet.lean` | stage-3 depth 9 → 27; channels unchanged |
| `jax/MainConvNeXtBImagenet.lean` | stage-3 depth 9 → 27; channels 96/192/384/768 → 128/256/512/1024 |

Nothing else moved. `mhsa` derives `head_dim = D // n_heads` at runtime, the
transformer/ConvNeXt block emitters loop over `nBlocks`, and the init / load /
save param walks are all generated from the same spec — so width and depth are
genuinely free parameters. Emitted parameter counts land on the published
figures exactly:

| model | emitted params | reference |
|---|---|---|
| ViT-S/16 | 22,050,664 | 22.05 M (DeiT-S) |
| ViT-B/16 | 86,567,656 | 86.57 M (DeiT-B) |
| ConvNeXt-S | 50,222,152 | 50.22 M |
| ConvNeXt-B | 88,589,416 | 88.59 M |

Recipes are copied from the existing Ti demos unchanged (DeiT applies one recipe
across Ti/S/B), with two deliberate exceptions, both commented in-file:
stochastic depth follows the ConvNeXt paper's per-size values (S 0.4, B 0.5) on
the 300-epoch recipe and is pulled back (0.2 / 0.3) on the 80-epoch tier, where
the paper values underfit.

## Measured cost per step

`jax/scripts/step_probe.py`: real `train_step` + `ema_update` on
synthetic batches, params replicated / batch sharded exactly as the trainer does.
Excludes the tf.data input pipeline, so ms/step is a **compute-only lower bound**
(the ConvNeXt-T run measured 143 ms/step live vs 130 ms here — ~10% pipeline tax).
`peak` is `peak_bytes_in_use` on device 0; the ceiling is 11.68 GiB, not 16, because
XLA preallocates 75% by default.

6 GPUs, no gradient accumulation:

| model | batch | ms/step | peak | min/epoch | hr @80ep | hr @300ep |
|---|---|---|---|---|---|---|
| ViT-Ti (existing) | 510 | 110 | 1.92 GiB | 4.6 | 6.1 | 23.1 |
| **ViT-S** | 510 | 243 | 4.05 GiB | 10.2 | 13.5 | 50.8 |
| **ViT-B** | 510 | 595 | 9.32 GiB | 24.9 | 33.2 | 124.5 |
| ConvNeXt-T (existing) | 252 | 130 | 3.01 GiB | 11.0 | 14.7 | 55.1 |
| **ConvNeXt-S** | 252 | 208 | 4.92 GiB | 17.6 | 23.5 | 88.0 |
| **ConvNeXt-B** | 252 | 303 | 7.23 GiB | 25.7 | 34.2 | 128.4 |

All four run out of the box on 6 GPUs at the Tiny demos' batch sizes, loss finite
and decreasing. ViT-B at 9.32 of 11.68 GiB is the only one without comfortable
headroom.

### 4 GPUs

The historical runs used 4 GPUs (0,2,3,4) to dodge the PCIe AER issue. At 4
devices the per-device batch is 1.5× larger.

| model | batch | ms/step | peak | hr @80ep | hr @300ep |
|---|---|---|---|---|---|
| ViT-S | 512 | 313 | 7.60 GiB | 17.4 | 65.2 |
| ViT-B | 512 | — | **OOM** | — | — |
| ViT-B (4×128 accum) | 512 eff | 712 | 5.65 GiB | 39.6 | 148.5 |
| ConvNeXt-S | 256 | 262 | 6.77 GiB | 29.1 | 109.3 |
| ConvNeXt-B | 256 | 395 | 9.56 GiB | 43.9 | 164.6 |

Three of four run unchanged; **ViT-B OOMs** (`RESOURCE_EXHAUSTED`, peak 11.41 of
11.68 GiB). Going 6→4 devices costs ~30% wall-clock, not 50% — these are
compute-bound enough that fewer, fuller devices lose less than the device count
suggests.

This is why each new file carries an `accum` recipe: same effective batch (so the
LR stays correct), micro-batched. These nets have no BN, so accumulation is
numerically exact. The existing `gradAccumSteps` path needed no changes — it just
had never been exercised on ViT/ConvNeXt before.

There is no `jax.checkpoint`/remat in the phase-2 emitter, but XLA does its own
rematerialization under memory pressure — see the ConvNeXt-B row below.

### ConvNeXt at effective batch 512 on 4× 16 GB

bs512 is not just a memory question: the LR doubles with the batch under the
paper's linear rule (4e-3 @ bs4096 → **5e-4** @ bs512, vs 2.5e-4 @ bs256). The
`bs512` recipes on the S and B files carry that.

| model | recipe | ms/step | peak of 11.68 GiB | min/epoch | hr @80ep |
|---|---|---|---|---|---|
| ConvNeXt-T | bs512 direct | 313 | 7.60 GiB | 13.0 | 17.4 |
| ConvNeXt-S | bs512 direct | 481 | 11.26 GiB | 20.1 | 26.8 |
| ConvNeXt-S | 2×256 accum | 521 | **6.91 GiB** | 21.7 | 29.0 |
| ConvNeXt-B | bs512 direct | 747 | 11.52 GiB | 31.1 | 41.5 |
| ConvNeXt-B | 4×128 accum | **681** | **6.19 GiB** | 28.4 | 37.9 |

All three fit one-shot, but S leaves 0.42 GiB and B leaves 0.16 GiB — and this
probe `device_put`s a pre-made array, so it does not model the tf.data
depth-2 prefetch buffers that also sit on device. Run the accum recipes.

**ConvNeXt-B's accum recipe is strictly better than one-shot**: 681 vs 747
ms/step at 5.3 GiB less. Same mechanism as the 3060 case — under memory pressure
XLA rematerializes, and the recompute costs more than the accumulation loop. For
ConvNeXt-S the trade is real but mild (8% for 4.35 GiB).

Larger batch also wins per epoch for both: S 21.7 min at bs512 vs 21.9 at bs256,
B 28.4 vs 32.9.

Accumulation correctness cross-check: direct and accum losses agree to 3–4
decimals at matched steps (S 6.1816 vs 6.1828, B 5.7601 vs 5.7625), the residue
being drop-path RNG splitting across micro-batches.

### Why ViT-B on 6 GPUs runs at 510, not 512

512 = 2^9 has no factor of 3, so it cannot be split evenly across 6 devices;
`BATCH_SIZE = (512 // n_devices) * n_devices` yields 510. Grad accumulation does
not help — the total is `micro × accum` with `micro` divisible by 6, so every
reachable total is divisible by 6. Exactly 512 on 6 GPUs is unavailable.

Uneven sharding is not an escape hatch — JAX 0.10.2 rejects it three ways
(`IndivisibleError`): `device_put` of a 512 leading dim over 6 devices, a jitted
op over such an array, and even hand-building it via
`make_array_from_single_device_arrays` from explicit 85/85/85/85/86/86 buffers.
`NamedSharding` validates against its computed shard shape before it looks at
your buffers; `shard_map` and `pmap` carry the same constraint.

It does not matter anyway. DeiT's rule is `lr = 5e-4 × batch/512`, so batch 510 at
LR 4.98e-4 is simply a correct config at a slightly smaller batch — not an
approximation of a 512 run. (A literal 512 *is* reachable by padding to 516 and
zero-weighting 4 samples in the loss, but that burns 0.8% of compute to fix a
0.4% discrepancy.)

**Round up, not down.** Measured on ViT-B / 6 GPUs:

| batch | ms/step | peak | min/epoch |
|---|---|---|---|
| 510 (6×85) | 594.7 | 9.32 GiB | 24.9 |
| **516 (6×86)** | **576.8** | 9.40 GiB | **23.9** |

516 is 3% faster per step *and* carries 1.2% more samples — ~4% more throughput.
Step times do not overlap (510: 592–596 ms over five samples; 516: 573–578), so
86/device genuinely tiles better than 85/device.

General note for this box: 6 devices means every batch must be `6 × per_device`,
and no power of two ever is. The clean choices are **384** (6×64) and **768**
(6×128); near the usual literature values, 510/516 stand in for 512 and 252/258
for 256. Scale LR linearly to whatever you pick.

## Would 4× RTX 3060 12 GB work?

**Memory: yes, all four.** XLA preallocates ~73% of a card (11.68 GiB observed of
16380 MiB), so a 12 GB 3060 gets ~8.7 GiB. Emulated by capping these cards at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.548` → an 8.54 GiB limit, slightly *tighter*
than a real 3060, so these results are conservative. 4 devices throughout:

| model | recipe | ms/step | peak of 8.54 GiB |
|---|---|---|---|
| ViT-S | bs512 | 313.5 | 7.60 GiB |
| ConvNeXt-S | bs256 | 262.7 | 6.77 GiB |
| ConvNeXt-B | bs256 | 401.2 | 8.54 GiB — exactly at the ceiling |
| ConvNeXt-B | 4×64 accum | **375.6** | **4.18 GiB** |
| ViT-B | 4×128 accum | 712.5 | 5.66 GiB |

Two things worth knowing:

- **ConvNeXt-B fits at bs256 without accumulation**, contrary to the naive
  read of its 9.56 GiB peak at the 16 GB budget. XLA rematerialized to fit the
  smaller budget and paid only 1.6% (395 → 401 ms/step). But peak landing
  *exactly* on the limit means zero headroom for prefetch buffers or
  fragmentation.
- **The accum recipe is strictly better for ConvNeXt-B here**: 375.6 ms/step at
  4.18 GiB beats 401.2 ms/step at 8.54 GiB on both axes. Micro-batches fit
  natively, so XLA skips the recompute that rematerialization forces. Use
  `accum` for ConvNeXt-B on 12 GB cards — it is faster *and* leaves 4 GiB spare.

**Speed: ~1.7× slower per card, estimated, not benchmarked** (no 3060 on hand).
RTX 3060 12 GB is 12.74 TFLOPS FP32 / ~51 TFLOPS bf16-tensor vs the 4060 Ti's
22.06 / ~88 — a 1.73× compute deficit — but 360 vs 288 GB/s of bandwidth, a 1.25×
surplus. These runs sit at 95–100% GPU util (compute-bound), so compute dominates.
Applying 1.7× to the measured rows:

| model | recipe | est. hr @80ep | est. hr @300ep |
|---|---|---|---|
| ViT-S | bs512 | ~30 | ~111 (4.6 d) |
| ConvNeXt-S | bs256 | ~50 | ~186 (7.8 d) |
| ViT-B | 4×128 accum | ~67 | ~252 (10.5 d) |
| ConvNeXt-B | 4×64 accum | ~71 | ~266 (11.1 d) |

Caveats specific to a 3060 box: Ampere (sm_86) has native bf16 tensor cores, so
`bf16` and `bf16Conv` work unchanged — the sm_89 workaround in the IREE notes is
IREE-specific and does not apply to this JAX path. If one 3060 drives a display,
subtract a few hundred MB and ViT-S's 0.9 GiB margin gets thin; run it with
`accum` there. The **8 GB** 3060 variant would give only ~5.9 GiB, which all four
would need `accum` to clear.

## What actually needed doing (the "something else")

1. **`jax/` did not build at HEAD.** `DatasetKind.brats224` was added for the
   BraTS work and three matches in `Jax/Codegen.lean` were never updated
   (two `.brats` panics, one trace-header name). Fixed.
2. **`MainMobilenetV3.lean` did not build either** — long-standing, since 574b1f9
   changed `mbConvV3`'s last field from `useHSwish : Bool` to `act : Activation`.
   Fixed mechanically (`false` → `.relu`, `true` → `.hSwish`; the per-line RE/HS
   comments confirm each). Unrelated to this work, but it blocks a clean
   all-executables build.
3. **Disk.** The supervise scripts run with `LEAN_MLIR_CKPT_EVERY=1`, and `.bin`
   weight files were never pruned (only `.state.npz` was). At ~340 MB per
   checkpoint, a 300-epoch ViT-B or ConvNeXt-B run writes ~100 GB of weights, and
   the box has ~120 GB free. Added `_prune_bin_ckpts`, gated on
   `LEAN_MLIR_KEEP_BIN` and **defaulting to keep-everything** so existing runs are
   byte-identical; set it to 3 for the big nets.
4. **Eval scripts hardcoded the Tiny trainer path.** `eval_vit_full50k.py` and
   `eval_convnext_full50k.py` now take `GEN` from the environment (same default).
   The rest of both scripts is already architecture-agnostic.
5. **Supervise scripts are per-run copies.** A new variant needs a copy with
   `DEVS` / `PY` / `CKPT_BASE` / `SPE` edited. Also note `supervise_vit_80ep.sh`
   points `VENV_PY` at `/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python`,
   which no longer exists; the working CUDA venv is
   `/home/skoonce/lean/klawd_max_power/lean4-jax/.venv`.
6. **Compile time.** ConvNeXt-S/B spend ~85–105 s per jit compile and there are
   several (train_step, eval_batch, ema_update, mixup, cutmix) — roughly 3–5 min
   of startup per (re)start. With AER auto-resume that is a per-resume tax the
   Tiny runs did not really pay.

## Suggested order if these get run

ConvNeXt-S at the 80-epoch tier is the best value: 23.5 hr, and ConvNeXt-T
already holds the sweep accuracy lead at 75.93%, so S is the most likely to move
the headline number. ViT-S 80ep (13.5 hr) is the cheapest new data point and the
clean apples-to-apples against ViT-Ti's 65.6%. The B variants at 300 epochs are
~5-day runs each and should wait until an S result confirms the recipe.


---

# The VERIFIED side (Lean render → StableHLO → PJRT)

Everything above is phase-2 JAX. This section is the verified peer, and the
short version is that the JAX conclusion does not carry: the JAX emitter derives
every width at runtime, while the verified renderers were written *at* one size.

## ✅ ViT-S and ViT-B — DONE 2026-08-14 (`b9ea36f`, `9affd2f`)

**What it took: parameterising one renderer.** `ViTRenderB.lean` had six
`private def` width constants pinned at Tiny. They became a `VitDims` record
threaded as a TRAILING DEFAULTED parameter — the same move the file's own header
records for `vbB` (batch) and calls "the whole prerequisite". Three instances now
exist (`vitTiDims`, `vitSDims`, `vitBDims`) and one renderer serves all three.

⭐ **The proof side needed NOTHING.** `Proofs.vitForwardKV_has_vjp` is already
`∀ heads d_head mlpDim k` and is a GLOBAL `HasVJP`, not the pointwise `_at` form
the relu-family nets carry, because GELU/softmax/LayerNorm have no kink. One
theorem covers Ti/S/B at different arguments.

⭐ **Two design decisions that made the bodies not move**, both worth copying:
- `d` and `tok` are DERIVED (`d = heads * hd`), not stored fields. So the sites
  that wrote `vbD` and the sites that wrote `vbH * vbHd` (head-slice operands)
  remain the same type with no cast. Stored with a `heads * hd = d` proof field
  they would be only PROPOSITIONALLY equal and every such site would need one.
- `heads_pos : 0 < heads` is a field because the attention loop builds a
  `Fin heads`; `by decide` closed that against the literal 3 and cannot close it
  against a record field.

**Cross-check, no GPU needed:** the verified specs derive **22,050,664** and
**86,567,656** from `VLayer.toSpecs`, matching this doc's JAX emitted counts and
published DeiT-S/DeiT-B exactly — in the SAME 200 parameter tensors as Tiny. S
and B widen every tensor and add none.

### Measured on the verified path (4× 4060 Ti, fp32, driver's own steady-state probe)

| model | config | ms/step | min/epoch | 300 ep |
|---|---|---|---|---|
| ViT-S | 4×128 = global 512 | **525** | 21.9 | 109.5 h |
| ViT-B | 4×32 = global 128 | **432** | 72.1 | ~360 h |

For reference this doc's JAX ViT-S at 4× measured 313 ms/step compute-only
(~345 live at its stated ~10% pipeline tax), so verified fp32 is ~1.5× the JAX
bf16 peer — the same ratio MNv4-Conv-M showed.

### ⚠⚠ ViT-B cannot reach the DeiT recipe on this box, and it is measured

Rendered at 4×128 and run: **OOM on all four devices**, asking 11.90 GiB against
the 11.68 GiB a 16 GB card's BFC allocator gets. This doc predicted it for bf16;
fp32 is worse. That probe artifact was deleted, not committed.

Reaching global 512 needs GRADIENT ACCUMULATION, and **ViT has no accumulation
render** — only R50 does (`resnet50in_accdp4x64_train_step.mlir` and peers).
`ViTRenderB`'s ten `acc` hits are the *attention* accumulator, a different thing.
That is a renderer feature, and it is the real remaining work for a quotable
ViT-B. ▶ **A TPU would delete this item entirely** (v3 is 16 GB/core, v4 32), and
the FFI is plugin-agnostic — `$PJRT_PLUGIN` always wins — so it is closer to an
env var than a port. See the end of this section.

### ⚠ Traps hit, none of which the build catches

1. **A bare `.map f` silently defaults the new parameter.**
   `(List.range vDEPTH).map blkArgSig` passes only `i`, so `V` fell back to Tiny
   and a ViT-S *body* rendered under a ViT-Tiny *block signature*. It type-checks
   and the artifact is wrong. Caught by shape-checking the emitted MLIR
   (`%wConv` was 384 while `%b0_Wfc1` was still 192×768). ▶ After adding a
   defaulted parameter, grep for bare point-free call sites of every function
   that took it.
2. **The driver loads `<slug>_fwd.mlir` BY NAME.** `vitsin_drop_fwd` does not
   satisfy it — the drop forward declares 24 `%dp<n>` mask inputs an eval forward
   has no values for. Caught by running the trainer, not by any build-time check.
3. **The trailing-defaulted parameter must reach the SIGNATURE builders too.**
   The batched train step delegates its parameter list to `vitParamSig` and
   `blkArgSig` in the per-example file; without threading `V` there the body is S
   and the func signature is Ti.

⭐ **The gate that made all of this safe: byte-identity.** After the
parameterisation, `scripts/regen_verified_mlir.sh proofs` must leave
`git diff verified_mlir/` EMPTY. It did. That is what says the refactor was inert
before any new artifact was added. ▶ Run it before adding the new instance, not
after — it separates "the refactor broke something" from "the new size is wrong".

### ⚠ What ViT-S/B is NOT

- **ImageNet only.** The 10-class `vit_*` artifacts come from the per-example
  `ViTRender.lean`, still pinned at Tiny by ~154 dimension literals. Only the
  BATCHED renderer was parameterised. There is no ViT-S/B Imagenette peer.
- **Nothing trained.** Artifacts render, shapes tie to `VLayer.toSpecs`, param
  counts are `#guard`ed, a few steps run. No accuracy, and the forward/gradient
  numeric ties against the JAX peers have not been run.
- Expected if trained (the only calibration point that exists): this repo's JAX
  ViT-Ti reached **70.28%** against DeiT-Ti's 72.2%, a 1.9-point recipe gap the
  book attributes to repeated augmentation, Mixup/CutMix alternation, and
  Random-Erase-to-zero. If that carries, ViT-S ≈ 77.5–78% against DeiT-S's 79.8%
  — but the missing items are all REGULARISATION and S has 4× Tiny's capacity, so
  the gap plausibly widens rather than holds.

---

## ✅ ConvNeXt-S — DONE 2026-08-14

**The prediction below was right, and it was the cheapest of the three.** S is
pure depth (`[3,3,9,3] → [3,3,27,3]`, dims unchanged), so it needed **one
`Array Nat` threaded as a trailing defaulted parameter** through both renderers —
no record, no dims work, and every hardcoded `96`/`768` left alone because no
dimension moves. ViT-S needed six constants become a `VitDims`; this needed a
loop bound.

⭐ **The proof side needed nothing, for a DIFFERENT reason than ViT-S.** ViT's was
that `vitForwardKV_has_vjp` is already `∀ heads d_head mlpDim k`. ConvNeXt's is
more basic: its certificates are per-SITE and already generic in `c`/`e`/`h`, so
18 more blocks is 18 more uses of theorems that were never indexed by depth.
Depth was not a hypothesis. ▶ **Deepening is structurally cheaper than widening**
on this codebase, and that is the transferable finding.

**342 parameter tensors, 50,222,152 scalars** at K=1000 — this doc's own JAX
emitted count and the published ConvNeXt-S figure, to the digit. T's 180 /
28,587,592 plus 18 stage-3 blocks at 9 tensors / 1,201,920 scalars each.

### What was threaded

| file | what took the parameter |
|---|---|
| `ConvNeXtRender.lean` | `cnxDropTotal`, `cnxBlockIdx`, `cnxDropSites`, `cnxDropSig`, `allParams`, `cnxWdCounts`, `convNextFwdChain`, `convNextFwdFaithfulV`, `convNextBackAll`, `convNextTrainStepFaithfulV`, `convNextAdamTrainStepFaithful` |
| `ConvNeXtRenderB.lean` | `convNextFwdChainB`, `convNextFwdRenderB`, `convNextBackAllB`, `convNextAdamTrainStepFaithfulB`, `cnxDropFwdBanner` |

⭐ **Byte-identity held before the new instance was added** — all 22 committed
`convnext*`/`convnextin*` artifacts re-render unchanged, which is what says the
parameterisation was inert. Run it in that order; it separates "the refactor
broke something" from "the new size is wrong".

### Artifacts (4, slug `convnextsin`)

`convnextsin_adamwxclipdrop_train_step.mlir` (1 replica),
`convnextsin_adamdpwxclipdrop_train_step.mlir` (4 replicas),
`convnextsin_drop_fwd.mlir` (the SD prefix partner),
`convnextsin_fwd.mlir` (the eval forward — see the traps).

### Measured, single GPU, fp32 — and ConvNeXt-T run identically as the control

`LEAN_MLIR_MAX_STEPS=20` on **one** 4060 Ti, `adamwxclipdrop`, batch 32,
40,036 steps/epoch. The T row is the same probe on the same card in the same
session, which is what makes the ratio mean anything:

| model | ms/step | init loss | compile | resident θ+m+v | h/epoch (1 GPU) |
|---|---|---|---|---|---|
| ConvNeXt-T | 167 | 7.84 | 8.1 s | 561 out | 1.86 |
| **ConvNeXt-S** | **280** | **8.09** | 11.9 s | 1065 out, 574.7 MiB | **3.11** |

**S costs 1.68× T per step.** The JAX bf16 peers in this doc's own 6-GPU table
are 130 → 208 ms, a ratio of **1.60** — so the deepening costs the same relative
amount on both paths, and the fp32 tax is a constant factor, not something that
grows with depth.

⭐ **Running the T control is what makes the init loss readable.** S starts at
8.09 against `ln(1000) = 6.91`, which looks high on its own — but T starts at
7.84 on the same probe, so the offset is this repo's init scheme at K=1000 and
not something the deepening introduced. A number this shape is exactly the kind
that gets quoted as a bug or waved through as fine, depending on nothing.

### Measured, 4 GPUs, `adamdpwxclipdrop` — global 128, 10,009 steps/epoch

Same probe, `PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4`, T again as the control:

| model | 1×32 | 4×32 | DP tax | min/epoch | 80 ep | 300 ep |
|---|---|---|---|---|---|---|
| ConvNeXt-T | 167 ms | **220 ms** | +53 ms (1.32×) | 37 | 49 h (2.0 d) | 184 h (7.6 d) |
| **ConvNeXt-S** | 280 ms | **366 ms** | +86 ms (1.31×) | **61** | **81 h (3.4 d)** | **305 h (12.7 d)** |

⭐⭐ **THE DP TAX IS EXACTLY GRADIENT BYTES OVER A 3.5 GB/s LINK, and that is the
most useful number in this doc.** S all-reduces 50,222,152 f32 = 200.9 MB per
device per step; ring traffic is `2·(N−1)/N ·size` = 301 MB, and 301 MB / 86 ms
= **3.50 GB/s**. Feeding T's 114.4 MB through the *same* constant predicts
49 ms; **measured 53 ms**. Two nets, one line, no fitting.

▶ **So the collective is PCIe-bound, not algorithmic**, and on NVLink (300–900
GB/s) it goes to ~1 ms. A machine with NVLink deletes 24% of this wall clock
before a single kernel gets faster. That is a hardware choice, not a code change.

### ⚠⚠ TF32 buys 10%, so this net is NOT tensor-FLOP bound — buy BANDWIDTH

The render carries **no `precision_config`**, so XLA is free to use TF32 on
Ampere+/Ada. `NVIDIA_TF32_OVERRIDE=0` turns it off driver-wide, which measures
what it was worth:

| | ms/step (1×32) |
|---|---|
| TF32 on (default) | 280 |
| TF32 off | 308 |

**10%.** Against ~2× peak-FLOP headroom, that says the convolutions are not the
bottleneck — the 335 `stablehlo.convolution`s are outnumbered by 6562
`multiply`, 5607 `broadcast_in_dim`, 4236 `add`, 695 `transpose` and 896
`reduce`, which is ConvNeXt's channel-LN chain plus the depthwise convs, all
memory-bound. Achieved 5.97 TFLOPS/card = 27% of the 4060 Ti's 22.06 fp32 peak.

▶ **Consequence for hardware, and it is counter-intuitive: an A100 is a weak
choice for this net and an H100 is worse value than its price implies.** A100's
non-tensor fp32 is 19.5 TFLOPS — *below* a 4060 Ti's 22.06 — so its advantage
here is its 2039 GB/s (7.1×) memory bandwidth, not its 156 TFLOPS of TF32, which
this graph can only reach for 10% of its time. Rank candidates by GB/s and by
interconnect, not by tensor TFLOPS.

⚠ And at **batch 32 per device** a large card is starved regardless: the kernels
are too small to saturate an 80 GB part, so much of the bandwidth advantage will
not land either. **`cBS` being a private constant is the real blocker to using
big hardware well** — see the §`cBS` note below, which is now the highest-value
item in this file rather than a tidy-up.

### Host RAM: 36.5 GB peak, and it is the val drain

Sampled RSS over the whole process tree during the 4-GPU probe: **36.5 GB peak**
(ConvNeXt-T 35.4 GB — so it is the DATA, not the model). Almost all of it is the
unconditional ImageNet validation drain (~28 GB, `imagenet_smoke_knobs`), which
`LEAN_MLIR_SKIP_EVAL` does not skip. ▶ **A rental with 32 GB of host RAM will
OOM on the host before the first step, whatever its GPU is.** Budget ≥ 64 GB.

Device memory: fits inside the 11.68 GiB a 16 GB card's BFC allocator gets — the
run is the proof. ⚠ The exact peak is NOT measured: the shim exposes no
`peak_bytes_in_use`, and `nvidia-smi` shows the 75% preallocation rather than
use. It is also not currently actionable, because the batch is baked and cannot
be traded against the headroom.

### ⚠ Renting: what the two measurements above imply (EXTRAPOLATED, not measured)

Scaled from the one measured point by **memory bandwidth** (justified by the TF32
result) with the all-reduce priced at each machine's interconnect. Compute derated
0.55 on the 80 GB parts because bs32 cannot fill them. **Every number in this
table is an extrapolation from a single 4060 Ti; treat the ORDERING as the
finding and the magnitudes as ±2×**, biased optimistic on the big cards (AMP
halves memory traffic on a bandwidth-bound net and these runs are fp32).

| 8× | ms/step | of which all-reduce | 80 ep | 300 ep |
|---|---|---|---|---|
| RTX 4090 (PCIe) | ~180 | **~100 ms (56%)** | 20 h | 75 h |
| L40S (PCIe) | ~194 | ~100 ms | 22 h | 81 h |
| A100 80G SXM (NVLink) | ~73 | ~1 ms | 8 h | 31 h |
| H100 80G SXM (NVLink) | ~45 | ~1 ms | 5 h | 19 h |
| *4× RTX 4060 Ti (measured)* | *366* | *86 ms (24%)* | *81 h* | *305 h* |

⭐ **The headline is the all-reduce column, not the compute column.** On a
PCIe-only 8×4090 box the collective becomes **the majority of the step** — 8
replicas raise ring traffic to `2·(7/8)·200.9` = 352 MB per device against a link
this repo has measured at 3.5 GB/s. Buying more consumer cards buys a bigger
collective. NVLink is worth more here than any FLOP number on the spec sheet.

⚠ **8 replicas × 32 = global 256 is EXACTLY the ConvNeXt paper's batch**, which
also makes the driver's `learningRate := 2.5e-4` correct rather than ~2× high as
it is at the global 128 this box runs. So 8 cards is the right count for recipe
reasons independently of speed — 5,004 steps/epoch, the reference's own figure.

▶ **The 80-epoch tier is the experiment to buy first** (~$100 either way, and
ConvNeXt-T already holds this repo's sweep accuracy lead at 75.93%), with
`LEAN_MLIR_DROP_RATE_U=200000`. The 300-epoch run is 4× the money for the
paper's schedule.

### ⚠ Traps, and only one was ViT's

1. **The banners were LITERALS.** `"ConvNeXt-T"` and `"18 drop sites"` are
   emitted INTO the artifact, so an S render would have opened by announcing
   itself as a net with half its blocks. Fixed by deriving the name from the
   depth table (`cnxModelName`) rather than passing it — a second parameter
   beside `D` is two writers for one fact. ▶ **Grep the renderer for its own
   model name before adding a size**; nothing gates a banner.
2. **`cnxBlockIdx si j D` must be spelled with `D` at every site**, and the
   `#guard` that checks the numbering had a point-free `.map (cnxBlockIdx si)`
   that would silently keep checking T. This is ViT-S trap 1 exactly, and it is
   worth stating as the rule: *after adding a defaulted parameter, grep for bare
   point-free call sites of every function that took it.*
3. **The eval forward is loaded BY NAME** (`<slug>_fwd.mlir`), so
   `convnextsin_drop_fwd` does not satisfy it — ViT-S trap 2, avoided by
   rendering `convnextsin_fwd.mlir` from the per-example renderer, as
   `convnextin` already does.
4. **The drop RATE moves with model size and nothing derives it.** The ConvNeXt
   paper uses 0.4 for S against T's 0.1. It is DATA (`dropKeeps` in the spec),
   not a render knob — so copying Tiny's ramp would have rendered, trained and
   descended at ¼ the reference's regularisation. `tests/TestDropPathRamp.lean`
   now gates the S ramp against `cnxBlockIdx` at the S table AND against not
   being Tiny's.

### ⚠ Found while doing it: ViT-S's SD forward has no prefix partner

`regen_verified_mlir.sh`'s `check_fwd_prefix` pairs a forward with a train step
byte-exactly. `vitsin_drop_fwd` is rendered at `vbB = 32` while ViT-S's only
train step is at 128, so they disagree on body line 0 (`tensor<32x150528>` vs
`tensor<128x150528>`) — a BATCH difference, not a semantic one, and the gate
cannot span it. **ViT-S is therefore the one net whose SD forward nothing
audits.** ConvNeXt-S is unaffected (`cBS` is 32 on both sides) and its row is in
the list. ▶ The fix is a bs-32 ViT-S train step or a bs-128 SD forward, not an
entry in that table. Recorded in the script beside where the row would go.

---

## ✅ ConvNeXt-B — DONE 2026-08-14

**The doc predicted the shape of this one and got it right: the literals were the
job.** B is S's depth table (`[3,3,27,3]` — *identical*, 36 blocks) at
`[128,256,512,1024]`, so nothing about the traversal changed and everything about
the widths did.

**342 parameter tensors — the same count as S — and 88,589,416 scalars.** The
published 88.59M, and this doc's own JAX emitted count. B widens every tensor and
adds none, which is the mirror of ViT-S's "same 200 tensors" claim.

⭐ **The proof side needed nothing, and B is the stronger evidence for that than S
was.** S reused the per-site certificates at the SAME widths; B instantiates them
at four widths no committed artifact had ever used. They are generic in
`c`/`e`/`h`. Neither depth nor width was ever a hypothesis.

### What it cost: one record, ~27 literals

`D : Array Nat` (S's depth table) became **`CnxDims`**, a record of `depths` and
`dims`, with `cnxTiny`/`cnxSmall`/`cnxBase`. ⚠ **The bundling is the point**: two
bare arrays admit `(S depths, T dims)` — a net that exists nowhere, yet
type-checks, renders and trains. One record makes it unspellable.

Threaded literals, all of them in three places: the **stem** (96 → `dims[0]`, in
`allParams`, the patchify conv/LN, and the stem backward's bias+weight grads),
the **head** (768 → `dims[3]`, in `gap`/`dense`/`dotOut`/`weightGrad`), and the
**hand-written GAP backward**.

⚠⚠ **The GAP backward is a declared §5 carve-out — hand-written TEXT on both
renderers, so nothing type-checks its width.** `%dgi`/`%dgb`/`%dgn`/`%dgd`/`%dgapf`
had `768` baked into five interpolated type strings. At T and S they were right by
accident of the dims not moving. This is the one place a missed literal would have
produced a graph the *lowerer* rejects rather than the compiler — i.e. after the
artifact was written and committed. Verified on the emitted B artifact:
`tensor<32x1024x7x7>` throughout and `1024*7*7 = 50176`.

### ⚠⚠ The trap B sprang that S could not

**`cnxModelName` keyed on the BLOCK COUNT.** With S as the only new size that was
fine — 18 → T, 36 → S. B has **36 blocks too**, so every B artifact would have
opened by introducing itself as a ConvNeXt-S, in the banner the renderer emits
into the file. Now it matches on the whole `CnxDims` record, and the guard beside
it compares S's name to B's rather than spot-checking either.

▶ **The general form: a derived label must key on everything that varies, and
"everything" grows when a second axis is added.** Deriving the name instead of
passing it (the S lesson) was necessary and not sufficient.

### ⭐ The gate: byte-identity at T **and** S

The dims refactor had to leave BOTH earlier sizes untouched — all 22 committed
`convnext*`/`convnextin*` artifacts, and the four `convnextsin_*` files, checked
by hash across the change. Both held before a single B artifact was written. That
is a strictly stronger gate than S had, and it is the whole reason a ~27-literal
edit inside a hand-written text block was safe to make.

### Artifacts (4, slug `convnextbin`)

`convnextbin_adamwxclipdrop_train_step.mlir` (1 replica),
`convnextbin_adamdpwxclipdrop_train_step.mlir` (4 replicas),
`convnextbin_drop_fwd.mlir` (SD prefix partner — a byte-identical 2984-line
prefix, same as S's), `convnextbin_fwd.mlir` (the eval forward).

Stochastic depth is **0.5** — the paper's B value, against S's 0.4 and T's 0.1.
Three sizes, three rates, all of them data rather than render knobs.

---

## ⭐⭐ Measured: all three ConvNeXt sizes, one box, one probe

`LEAN_MLIR_MAX_STEPS`, 4060 Ti, fp32, batch 32/device, `*wxclipdrop`.

| model | params | 1×32 | 4×32 | DP tax | **predicted tax** | min/epoch | 80 ep | 300 ep |
|---|---|---|---|---|---|---|---|---|
| ConvNeXt-T | 28.6 M | 167 ms | 220 ms | 53 ms | 49 ms | 37 | 49 h | 184 h |
| ConvNeXt-S | 50.2 M | 280 ms | 366 ms | 86 ms | 86 ms | 61 | 81 h | 305 h |
| **ConvNeXt-B** | **88.6 M** | **408 ms** | **539 ms** | **131 ms** | **152 ms** | **90** | **120 h (5.0 d)** | **450 h (18.7 d)** |

⭐⭐ **THE DP TAX IS PREDICTED BY ONE CONSTANT ACROSS A 3× RANGE OF GRADIENT
VOLUME.** The rule — ring traffic `2·(N−1)/N ·(params × 4 B)` over a **3.5 GB/s**
effective link — was fitted on ConvNeXt-S alone and then predicted T to 8% and B
to 14%, in the direction that says bigger messages amortize slightly better
(B's implied rate is 4.06 GB/s against S's 3.50). ▶ **The collective is a
bandwidth line, not an algorithmic cost**, so on NVLink all three taxes go to
~1 ms and B alone gets back 24% of its wall clock.

⭐ **B FITS at bs32, and that is not obvious** — ViT-B, at a comparable 86.6 M,
OOMs on this box (11.90 of 11.68 GiB) and needs gradient accumulation ConvNeXt
does not have either. ConvNeXt-B ran first time: resident θ+m+v is **1013.8 MiB**
and the activations clear the 11.68 GiB budget. The difference is the batch —
ViT-B was rendered at 128/device where ConvNeXt is at 32 — which is another way
of saying `cBS` is the axis everything on this net turns on.

Init loss rises with size: T 7.84, S 8.09, B 8.98 against `ln(1000) = 6.91`. All
three sit above it, so the offset is this repo's init scheme rather than anything
the scale-up introduced — which is only knowable because T was re-probed as a
control in the same session.

Host RAM: 36.5 GB peak (S), 35.4 GB (T) — dominated by the unconditional ~28 GB
val drain, so it barely moves with model size. Budget ≥64 GB on any rental.

---

## Appendix: the ConvNeXt-B plan as written BEFORE it was done

Kept because the estimate was accurate and that is worth being able to check.
⚠ Everything below is superseded by the section above.

### The shape of the job

Same as ViT: parameterise `ConvNeXtRenderB.lean` (the batched renderer), then
instantiate. ⭐ **The depth half is already done** — `D` is threaded through both
renderers and `#eval`-tested at two tables. What remains is the DIMS, which is
the expensive half. Confirmed reachable — the recipe-complete variants come from
there:

| variant | writer |
|---|---|
| `convnextin_adamdpwxclipdrop` (DP, drop, wd-exclude, clip) | **ConvNeXtRenderB** |
| `convnextin_adamwxclipdrop` | **ConvNeXtRenderB** |
| `convnextin_adamdp`, `convnextin_adam` (the driver's DEFAULT) | ConvNeXtRender (per-example) |

⚠ The driver defaults to `LEAN_MLIR_VARIANT=adam`, which the per-example
renderer owns. As with ViT, target the `*drop` DP variants and say so in the app
docstring rather than silently shipping a driver whose default variant does not
exist for the new size.

### ✅ The depth half is already threaded (ConvNeXt-S, above)

`D : Array Nat := cDepths` / `:= bDepths` is a trailing defaulted parameter on
every function listed in the ConvNeXt-S table, and `cnxDepthsS` exercises it. So
**B does not need any depth work at all** — `[3,3,27,3]` is already a rendered,
gated table. What B adds is the dims, and only the dims.

### ⚠⚠ ConvNeXt-**B** is where the dim literals bite

B is depth AND width: dims `[96,192,384,768] → [128,256,512,1024]`. The batched
renderer does not route the final-stage width through `bDims`:

```
ConvNeXtRenderB.lean:173  (.gap (c := 768) (h := 7) (w := 7))
                  :176  (.dense "%Wd" "%bd" (zMB : Mat 768 nClasses) zVB)
                  :395  (.dotOut "%Wd" (zMB : Mat 768 nClasses))
                  :397  (.weightGradB (m := 768) (n := nClasses) ...)
                  :403  %dgi = reshape ... ty [bB,768] -> ty [bB,768,1,1]
                  :404  %dgb = broadcast_in_dim ... -> ty [bB,768,7,7]
```

Twelve `768` occurrences in the batched renderer and fifteen in the per-example
one, concentrated in the head and the GAP backward; `96` appears 13 and 21 times.
Against that, the tables are used symbolically only 10 times (RenderB) and 15
(Render). ▶ **So the literals outnumber the symbolic uses** — this is NOT the
clean 6-constants-and-2-literals situation `ViTRenderB` was in, and the estimate
should be set accordingly. Thread them through `bDims[3]!` (and `bDims[0]!`)
FIRST, verify byte-identity at ConvNeXt-T, and only then add the B instance.

### ⭐⭐ `cBS` is still a private constant — and it is now the TOP item, not a tidy-up

`ConvNeXtRender.lean:41  private def cBS : Nat := 32`. This doc's ViT header note
called making it a parameter "the whole prerequisite" for the stronger
split-identity gate. **The 2026-08-14 measurements promoted it from hygiene to
the main blocker on ever running these nets economically**, for two independent
reasons:

1. **It starves big hardware.** bs32 kernels cannot fill an A100/H100, so most of
   what you would rent is unusable. The bandwidth advantage that the TF32 result
   says is the ONLY advantage that matters here is the part bs32 fails to collect.
2. **It is the only lever on the all-reduce.** The collective cost is per STEP
   and independent of batch (342 gradients either way), while the step count
   falls linearly with batch. Going bs32 → bs128 per device cuts the number of
   all-reduces per epoch 4×. On the PCIe box that is 24% of wall clock; on an
   8×4090 rental it is 56%.

▶ Do `cBS` BEFORE ConvNeXt-B. B is bigger and will hit both walls harder, and the
threading is the same trailing-defaulted-parameter move `D` just took — with the
same byte-identity gate at T and now S.

### Suggested order

1. ~~ConvNeXt-S by changing `bDepths`/`cDepths` alone~~ — **done**, and it was
   the one-array job predicted.
2. Thread the DIMS: a second `Array Nat` beside `D`, or fold both into a
   `CnxDims` record now that there are two tables to keep in step. ⚠ The record
   is the better shape here for a reason ViT's was not: `cnxModelName` already
   has to key on depth alone and therefore **cannot distinguish B from S**
   (both are `[3,3,27,3]`) — it must take the dims in the same pass, or B's
   artifacts will introduce themselves as ConvNeXt-S in their own banners.
3. Byte-identity at ConvNeXt-T **and now at ConvNeXt-S** — two sizes must come
   back unchanged, which is a strictly stronger gate than the one S had.
4. ConvNeXt-B instance. Its drop rate is **0.5** at 300 epochs (T 0.1, S 0.4),
   and it is data, not a render knob.
5. Probe with `LEAN_MLIR_MAX_STEPS` before promising a wall clock. This doc's JAX
   rows say ConvNeXt-B needs `accum` on 4 cards even in bf16 — expect the
   verified fp32 path to need it more, and **ConvNeXt has no accumulation render
   either**, the same wall ViT-B hit.
