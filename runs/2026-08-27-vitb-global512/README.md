# 2026-08-27 — ViT-B reaches DeiT's global 512, and §4d's accumulation loop is not needed

**What this closes.** `planning/verified_side_quest_counterparts.md` §4d called ViT-B gradient
accumulation *"the one real feature"* on the side-quest list — the only item that was not a flag —
and §6a wrote out a four-step test for whether 2026-08-27's allocator fix deleted it. The test ran.
It did.

> ⭐⭐⭐ **THERE IS NOW A CANDIDATE ANSWER THAT NEEDS NO TPU** […] *"one-shot global 512 peaks at
> 11.41 of the allocator's 11.68 GiB and dies `RESOURCE_EXHAUSTED`"*. **That 11.68 is the plugin's
> `memory_fraction = 0.75` default, not the card.**

⛔ **Nothing has been TRAINED.** These are the graph's own figures on four cards. No accuracy, no
trainer wall clock.

---

## The answer, in one control

The same four artifact bytes, the same four 4060 Ti, ten steps each. The only difference between
the two rows is whether one environment variable is set.

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97   1295.61 ms/step   vitbin_adamdp128x4wxclipdrop
                            (unset)   RESOURCE_EXHAUSTED: Out of memory
                                      while trying to allocate 11.96GiB
```

▶ So the `RESOURCE_EXHAUSTED` that justified building an accumulation loop was never a statement
about ViT-B or about a 16 GB card. It was a statement about an unset `memory_fraction`.

## What was rendered

Four artifacts, `LeanMlir/Proofs/Codegen/ViTRenderB.lean`. The 128×4 shape is not new to this
renderer — `vitsin_adamdp128x4wxclipdrop` and `vitin_adamdp128x4wxclipdrop` both ship — so this is
four `#eval`s, not a feature.

| artifact | replicas | per-dev | global | why |
|---|---|---|---|---|
| `vitbin_adamdp128x4wxclipdrop_train_step.mlir` | 4 | 128 | **512** | the DeiT batch, fp32 |
| `vitbin_adamdp128x4wxclipdropbf16_train_step.mlir` | 4 | 128 | **512** | its precision peer |
| `vitbin_adam128wxclipdrop_train_step.mlir` | 1 | 128 | 128 | ⚠ a CONTROL, not a recipe |
| `vitbin_adam128wxclipdropbf16_train_step.mlir` | 1 | 128 | 128 | ⚠ ditto |

⚠ **The two 1-replica renders are not a second recipe and must not be quoted as one.** ViT-S and
ViT-B had no single-device render at all, which is what made §6a's third failure mode —
*"the all-reduce buffer is not in a single-device peak"* — unfalsifiable: it cannot be checked
without a graph that has no all-reduce in it. These are that graph.

## `peak_memory.log` — and the budget must be quoted with the peak

| render | at 11.68 GiB (default) | at 15.11 GiB (0.97) |
|---|---|---|
| `adamdp128x4wxclipdrop` (fp32) | 13.97 G — **120 %** ⛔ | 13.99 G — 93 % ✅ |
| `adamdp128x4wxclipdropbf16` | 10.88 G — 93 % ✅ | 12.61 G — 83 % ✅ |
| `adamdp32x4wxclipdrop` (the shipped pin) | 6.46 G — 55 % | not re-measured |

⚠ **The bf16 row moves 10.88 → 12.61 when given MORE room**, which is not a measurement error:
XLA rematerialises less when it has budget. §6c saw the same on R50's fp32 `4×128` (11.52 → 11.91).
A compile-time peak is not independent of the allocator it was compiled against, so the budget goes
next to the number every time.

⭐ **XLA states the un-rematerialised size itself**, which is what makes the 13.99 legible:

```
Can't reduce memory use below 10.16GiB by rematerialization; only reduced to
13.36GiB (14341998740 bytes), down from 20.39GiB (21889508632 bytes) originally
```

▶ 20.39 G is where a naive `4× the 32×4 temporaries` estimate lands, and it is wrong by 6.4 G
because it prices the graph rather than what the compiler does to it under pressure.

⭐ **The all-reduce costs nothing measurable in the peak.** The DP render and the collective-free
1-replica render at the same batch read the **same 13.99 / 12.61** (temporaries 11.98 against 12.06
— the DP graph is marginally *smaller*, which is scheduling noise, not a buffer). Compiling the DP
artifact at `--replicas 1` and at `--replicas 4` also gives identical figures. §6a's third way this
could fail does not bite.

## `device_step.log` — and the basis is measured in the same session

```
  1295.61 ms/step   vitbin_adamdp128x4wxclipdrop        ← global 512
   745.34 ms/step   vitbin_adamdp128x4wxclipdropbf16    speedup 1.74x
   386.04 ms/step   vitbin_adamdp32x4wxclipdrop         ← global 128, the shipped pin
   273.79 ms/step   vitbin_adamdp32x4wxclipdropbf16     speedup 1.41x
```

⭐ **The 32×4 pair reproduced its committed figures to 0.6 % and 0.03 %** (§4c has 383.8 and 273.7).
That control is what licensed this session's 128×4 rows, rather than the 128×4 rows licensing
themselves. ⚠ It cannot be re-run — those artifacts are deleted (see the wall clock below); the
live control is ViT-Tiny's trainer probe.

### The wall clock — MEASURED, and global 512 is faster as well as more faithful

`trainer_probe.log`. 40 steps on real ImageNet, four cards, median of steps 9–40, eval skipped.
1,281,167 images: 2,502 steps/epoch at global 512, 10,009 at 128.

| | fp32 | bf16 |
|---|---|---|
| ViT-B trainer ms/step @ 512 | **1,381** | **837** |
| ViT-B, 300 epochs (incl. 37.5 s/epoch eval + ckpt) | **291 h** (12.1 d) | **178 h** (7.4 d) |
| the deleted ViT-B 32×4 pair, for comparison | 322 h | 228 h |
| ViT-S trainer ms/step @ 512 | 528 | 319 |
| ViT-Tiny trainer ms/step @ 512 | 241 | 166 |

⭐ **Tiny and S are the session controls** and they say the box had not drifted: **241 → 166** and
**528 → 319** against the 237 → 168 and 528 → 325 committed in the book — S's fp32 lands on its
committed figure exactly. So ViT-B's row sits on the same basis as the two above it.

⭐⭐ **AND PROBING S ANSWERED A SEPARATE QUESTION: it was launchable all along.** The book had said
*"B is launchable and S is not"*, which was true of job configs and of nothing else. ViT-S has its
driver, its spec, its shim and both 128×4 renders, it defaults to 128 per device, and at **10.27
GiB** it fits even the DEFAULT arena. Nothing blocked it. `vits-default-g512-4gpu` now exists.

⚠ **Its allocator line is HEADROOM, not necessity, and the config says so rather than copying B's
refusal.** 10.27 of 11.68 is 88 %, close enough to the wall that a fragmented pool could fail a
300-epoch run; 0.97 makes it 68 %. But the graph fits without it, so the precheck does NOT enforce
it — a refusal there would claim a limit this net does not have, which is the same class of error
as the one this whole session corrected, pointing the other way.

⭐ **And the derivation this README used to carry was close.** It predicted 1,361 / 810 ms from the
device figures plus §4c's ~65 ms feed; measured is 1,381 / 837, i.e. **−1.4 % and −3.2 %**. The
additive-constant model for a four-replica net holds at ViT-B's width. ▶ It was still right to
measure: a table with two probed rows and one modelled row is not one table.

⚠ **The feed is ~85 ms here, and it hides.** ViT-B wants 512 / 1.381 ≈ 371 img/s where ViT-Tiny at
the same global batch wants ~2,048 and gets ~770 — which is why Tiny's job is data-bound at 665 ms
over a 250 ms compute floor and this one is not. ⚠ The bf16 arm is nearer the edge at ~612 img/s.

⛔ **The 32×4 pair is DELETED** (brett, 2026-08-27), the ViT half of what `2b2b15b` did to R50's
`8×64`. It was global 128 where DeiT's recipe is 512, it applied the batch-512 LR to a quarter of
the images that rate was set for, and it was slower per epoch on both arms. The measurements above
are kept because they are what licensed the deletion; the artifacts are not.

⭐ **Why it gets faster on a bigger batch**: the same per-invoke-overhead amortisation R50's
`4×128` showed against `8×64` (§6c) — a bigger per-device batch spreads the fixed cost of an
invocation over four times the images. §6a explicitly did not predict this ("the wall clock does not
necessarily improve at global 512 … though R50's `4×128` did turn out faster … so check"). Checked.

## What this does to the recipe, which is the part that is not about speed

DeiT-B's recipe is global 512 and `baseLR = 5e-4` is the **batch-512** rate. At 32×4 the driver
applied that rate to a batch four times too small and recorded it as a deviation. At 128×4 there is
no deviation: the batch is the reference's and the LR is the reference's for that batch. ▶ Both
halves of *"no run of this driver is comparable to DeiT-B's 81.8 %"* are gone.

## Three docstrings said the opposite, and all three are fixed

The claim had propagated to three places, each stating it as a hardware fact:

* `apps/imagenette/MainViTBImagenet.lean` — *"it OOMs, asking 11.90 GiB against the 11.68 GiB the
  BFC allocator gets on a 16 GB card"*. The 11.90 is right and reproduces (11.96 here); "gets on a
  16 GB card" is the error.
* `LeanMlir/VerifiedNets.lean`, `vitBImagenetVerified` — *"per-device batch 32 … a memory fact
  rather than a recipe choice"*.
* `LeanMlir/Proofs/Codegen/ViTRenderB.lean` — *"the phase-2 JAX probe measured ViT-B OOM at 4×128
  on these 16 GB cards IN bf16"*.

⚠ **The bf16 half of that last one is doubly wrong**: the verified bf16 render at 128×4 reads
10.88 G and RUNS at the default budget (row G of `oom_control.log`, 760.52 ms/step). So bf16 alone
would have bought global 512 — in bf16 only. §4c's rule that a tier shipping without its precision
peer "reads as a decision" is what makes that insufficient, and the fp32 twin is exactly what the
allocator fix buys.

## Gates

| gate | result |
|---|---|
| `regen_verified_mlir.sh proofs`, byte-identity **before** the new instances | `git diff verified_mlir/` empty |
| the four `#guard`s pinning path ↔ `vitAdamVariant` | pass (build fails otherwise) |
| `parse_verified_mlir.py` | ✓ 233 artifacts parse (was 229) |
| `check_render_coverage.py` | refused until all four entered the CI drift guard; then 207/225 diffed |
| `tests/TestVariantPredicates.lean` | ✓ 90 spellings, 5 axes, no collision |

⚠⚠ **`TestVariantPredicates` was missing the spelling three nets already ship.**
`adamdp128x4wxclipdrop` — ViT-Tiny's, ViT-S's and now ViT-B's — appeared only as three scattered
single-predicate `#guard`s and never as a table row, so it was never run through all five
predicates. No ViT bf16 spelling was in the table at all, though `dropbf16` is a marker adjacency
R50's `bcewd001bf16` rows do not exercise. Three rows added. That is the file's own header rule
("run every CONCATENATION") having failed against the most-shipped ViT name in the tree.

## The job config

`scripts/jobs/vitb-default-g512-4gpu.conf` — the first job in the tree that sets
`LEAN_MLIR_MEM_FRACTION`, closing §6c's *"a job-config line rather than a missing artifact"*.

⚠⚠ **Two new prechecks, both run RED before being trusted.** A precheck that has never refused is
decoration.

| control | what it printed |
|---|---|
| `LEAN_MLIR_MEM_FRACTION` deleted from `ENV_EXTRA` | *"…the FIRST STEP dies RESOURCE_EXHAUSTED on all four devices — it does not run slower, it does not fall back to a smaller batch, it stops."* → refused |
| the shim check pointed at a binary without the string | *"…predates the allocator-options change and LEAN_MLIR\_MEM\_FRACTION will be IGNORED."* → refused |
| `RECIPE=` disagreeing with the filename (the engine's own) | refused, naming both spellings |

⭐ **The shim check is the one that is not obvious.** `ffi/pjrt_ffi.c` is not a lake target, so a
tree can hold the source that reads the variable and still run a binary that ignores it — and that
failure is indistinguishable from a card being too small, which is exactly the misreading this
whole session corrected. `strings ffi/libpjrt_ffi.so | grep memory_fraction` is cheap and decides it.

⚠⚠ **The job token is `vitb`, not `vit-b`, and that is forced.** `supervise.sh` reads the recipe as
the filename's second dash-field and its header states *"a net whose name contains a dash would
break the field split — none does"*. N4's exe rename made that false: `vit-b-default-g512-4gpu`
would have parsed its recipe as `b`. Jobs use the short token every existing job already uses
(`cnx`, `enet`, `r34`); the exe namespace and the job namespace are separate.
▶ `convnext-s` and `convnext-b` hit this next.

⚠ **Its throughput line is a PREDICTION.** The conf argues ViT-B will be compute-bound where
ViT-Tiny is data-bound — 512 img / 1.296 s ≈ 395 img/s needed against ~770 that 8 producers were
measured to deliver — and says so as a prediction with the disconfirming observation named. No
trainer probe has been run.

✅ **AND THE SHIM PATH IS CONFIRMED ON THIS GRAPH.** Every memory figure above went through JAX's
Python compile path and `XLA_PYTHON_CLIENT_MEM_FRACTION`; the job goes through the C shim and
`LEAN_MLIR_MEM_FRACTION`, which is a different code path. The trainer probe exercised it and the
shim announced itself: `[pjrt_ffi] allocator: 1 create option(s), memory_fraction=0.970`. Forty
steps then ran at global 512 in both precisions.

⭐ **The driver refuses too**, and it is the only driver in the tree that does. Without the variable
it throws a sentence naming it, exempting the bf16 twin because that render fits the default arena.
The refusal exists because the failure it replaces is *misleading* rather than merely unhelpful —
`RESOURCE_EXHAUSTED … 11.96GiB` reads as "this card is too small for ViT-B", which is exactly the
reading that pinned this net at a quarter of its recipe's batch.

## Reproduce

```
runs/2026-08-27-vitb-global512/probe.sh
```

⚠ Needs the pinned venv (`jax/requirements-cuda-lock.txt`) and four free GPUs. About four minutes.
