# 2026-08-27 — RSB-A2/A1 get the model-EMA shadow: the fifth blob region

**What this closes.** `planning/verified_side_quest_counterparts.md` §4a found that the eight A2/A1
renders it had just landed were **not faithful A2/A1**, because two of A2's regularisers had no
expression on the verified path. This run closes the first of the two:

> ⛔ **Model EMA (`emaDecay := 0.9999`) is UNREPRESENTABLE, not merely unrendered.** The EMA shadow
> and the gradient accumulator are the **same fourth region** of `[θ|m|v|·]`, and
> `VerifiedTrain.lean:1156` already throws on the pairing. […] Lifting it means a **fifth region**
> in the driver's pack/unpack and in every optimizer's return list. That is its own project, not a
> flag.

It is now a fifth region. `nRegions` is 3, 4 or 5; `nScalars` is 3, 5 or 7; the layout is
`[θ|m|v|G|E]` and the scalar tail is `lr,bc₁,bc₂,aup,akeep,emad,oemad`.

✅ **AND THE SECOND DELTA IS CLOSED TOO** — stochastic depth `dropPath := 0.05`, sixteen sites on
the residual branch, one per bottleneck. It needed a fix on the *reference* side first: see §6b
below. So `sec:r50_a2_a1_cost`'s two ⛔ rows are both gone.

⚠ **What is still NOT A2**: the ghost-BN group — see `peak_memory.log` below. These render 8×64, i.e. 64-image ghosts, against
the reference's 4 × 512-global (128 per device on four cards). Immaterial to a wall clock; a
different regime for any accuracy claim. That is now the only delta, and it is the one §4a listed
as ⚠ rather than ⛔.

---

⚠⚠ **THE `8×64` ARTIFACTS ARE GONE** (2026-08-27) — superseded by `4×128`, the reference's own
factorisation, which doubles the batch-norm group AND runs faster. The measurements below stay
because they are what licensed the move; the artifacts do not.

## The artifacts

| artifact | replicas | baked `%wd` | regions |
|---|---|---|---|
| `resnet50in_emalambaccdp8x64wxclipbce_train_step.mlir` | 4 | 0.02 (A2) | 5 |
| `resnet50in_emalambacc8x64wxclipbce_train_step.mlir` | 1 | 0.02 (A2) | 5 |
| `resnet50in_emalambaccdp8x64wxclipbcewd001_train_step.mlir` | 4 | 0.01 (A1) | 5 |
| `resnet50in_emalambacc8x64wxclipbcewd001_train_step.mlir` | 1 | 0.01 (A1) | 5 |

920 inputs, 918 outputs, 1.6 MB each. The 1-replica peers are not optional: `r50-accum-tie` and
`r50-accum-shard-tie` both compare a DP render against a single-device one.

⚠ **No bf16 twins, and that is a decision.** §4c's rule is that a tier shipping without its
precision peer "reads as a decision and is really an accident of ordering" — but the shadow's
arithmetic is `e' = d·e + (1−d)·θ′` on MASTER weights, which stay f32 in every bf16 render in this
tree. A bf16 twin would differ from the fp32 twin in no line of the EMA block. Render them the day
a bf16 A2 run is scheduled.

---

## `opt_step_tie.log` — the composition, measured

⭐ The new `emalambacc8wxclip` row runs `generated_resnet50_imagenet_a2accum.py`'s **own**
`ema_update` — extracted and `exec`'d, not re-implemented — against the `E` slot the render emits.

```
  ✓ emalambacc8wxclip K=8  ref=r50_a2accum.py  worst rel 1.20e-07
  worst 3: emalambacc8wxclip/E'[0] (shadow) 1.20e-07, …
✓ 7 variants agree with the shipped reference within rtol 2e-06
```

⭐⭐ **The decay scalars come out of the reference too, not out of a formula written here.**
`ema_update` at `(ema := 1, params := 0)` returns `d·1 + (1−d)·0 = d` exactly, so `%emad`/`%oemad`
are tied to the reference as well as the op that consumes them. This repo has shipped the
warmup-corrected decay wrong once already (`planning/ema.md`: a shadow holding 12.8% of its init at
3.1 tau, scoring 0.00% top-1 beside live weights at 70.48%).

## `ema_control.log` — and it goes RED on both defects it claims to catch

```
  rtol gate                                 2e-06
  TEST     E' vs ema_update(e0, theta')     1.20e-07
  CONTROL a  E' vs ema_update(e0, theta)    1.14e-02   <- one-step lag
  CONTROL b  E' read at offset 9 (G's slot) 1.91e+01   <- swapped region order
  CONTROL c  G' read at offset 12 (E's slot)1.01e+00   <- swapped region order
```

⚠⚠ **Control (a) is the one worth reading twice.** A shadow wired to the INCOMING θ instead of the
updated θ′ is off by only **1.1%** — because at lr 0.005 one step barely moves θ. It is four orders
above this gate and would be invisible to a 1e-2 tolerance, and it trains, descends and prints a
normal curve. The reference's ordering (`ema_update` follows the `train_step` call) is the only
thing that says which is right.

⚠ The incoming shadow `e0` is deliberately NOT `theta`, even though the driver seeds it there: with
`e0 = θ` the update reads `d·θ + (1−d)·θ′ ≈ θ′` and the row would pass with its two operands
swapped.

## `arity_check.log` — the driver's blob against the artifact's signature

The shim refuses a count mismatch, but only at run time on a GPU. This reproduces the count
arithmetic against the committed text, in both directions — the driver does `pbuf := out`, so the
previous output IS the next input and the two layouts have to line up either way.

```
✓ 23/23 artifacts agree with the driver's packed layout
```

⭐ All three region counts are exercised: 12 three-region, 7 four-region, 4 five-region.

## `variant_predicates.log`

`✓ 75 variant spellings, 5 axes, no collision`, and its region guard is now a **three-way** sum
with a populated-bucket check beside it. It used to read `nRegions v == (if e || accOn v then 4
else 3)` — true of every spelling that existed, and silently true of a five-region one.

---

## §6b — the reference defect the sd half ran into

⛔ **Four of the five JAX block emitters drew a per-BLOCK SCALAR bernoulli**, shared by the whole
batch — nine sites across `bottleneck_block`/`_down`, `convnext_block`, `mbconv_block`,
`uib_block`, `fused_mbconv_block`. timm 1.0.28's `drop_path` is explicitly *"per sample"*, and the
same file already emitted the correct `_drop_branch` for the transformer family alone.

⭐ **Verified renders were right; the oracle was wrong**, for 26 committed `convnext*`/
`efficientnet*` artifacts. `_drop_branch` is now one definition and all nine sites call it.

### `droppath_shape.log` — measured, because the two forms have the SAME EXPECTATION

```
  per-example (now)      examples dropped/step: min 0, max 9, mean 3.23 (expect 3.2)
                         steps where ALL 64 or NONE dropped:   5/200
  per-block scalar (was) examples dropped/step: min 0, max 64, mean 2.88
                         steps where ALL 64 or NONE dropped: 200/200
```

⚠ No mean-based check separates them, which is why it survived. The graph type-checks, eval is the
identity either way, and no gate compares two Bernoulli streams.

⭐ Nothing measured changed — `dropPath > 0` is ImageNet-tier only, R50's A3 reference sets 0.0,
and MNv4-Conv-M's 75.51% ran `default`.

## `drop_site_placement.log` — is the site on the BRANCH or on the block OUTPUT?

⚠⚠ **NO STRUCTURAL CHECK IN THIS REPO CAN TELL THEM APART.** `scripts/misplace_drop_sites.py` moves
all 16 sites onto the block output and produces a render with the same SSA names, the same order,
the same types, the same 18,186 ops and the same arity. Run at `q = 1` / `B = 2` so it fits on CPU:

```
  ⭐ TEST     at a REAL draw, correct vs misplaced      : 2.19e+08
  CONTROL  at an ALL-ONES mask, correct vs misplaced   : 0.00e+00
  CONTROL  ones mask, sd vs NO-sd render, FORWARD      : 0.00e+00
  CONTROL  ones mask, sd vs NO-sd render, m' and v'    : 3.29e-06
```

⭐ The first control is the argument: at an all-ones mask the misplaced render is **bit-identical**,
so every endpoint gate, prefix audit and arity check passes on it. Only a non-ones mask sees the
placement, and it sees it by eight orders of magnitude.

⚠ The last row is **not** bit-exact and correctly so — the extra `multiply` changes XLA's fusion,
so the 161-parameter reduction chain reassociates. The FORWARD is exact (loss and all 106 BN
statistics, difference 0.0); the gradients agree at ~1e-6; θ′ amplifies that through AdamW's
`m̂/(√v̂+ε)` wherever `v` is small. Claiming bit-exactness on θ′ would have been false.

▶ Two defects this file had, both found by running it rather than reading it: `hash()` is salted
per PROCESS, so the TEST figure moved four orders between two invocations (now `crc32`); and the
`v` region was initialised random-signed, so `sqrt(v)` gave NaN for 50 of 161 parameters beside a
perfectly finite loss.

---

## `peak_memory.log` — what A1/A2 actually cost, and what ghost-BN costs

`scripts/bf16_peak_memory.py` reads XLA's own `peak_memory_in_bytes` — arguments + outputs +
temporaries for one execution, i.e. the number that decides whether a batch FITS. ⚠ NOT
`nvidia-smi`, which reports the BFC allocator's PREALLOCATED pool (~73 % of the card) and makes two
very different artifacts read the same. Budget: **11.68 GiB** per card on a 16 GB 4060 Ti.

| render | peak | of 11.68 (default) | of 15.11 (at 0.97) |
|---|---|---|---|
| A2/A1 `8×64` fp32 | 6.18 G | 53 % | 41 % |
| A2/A1 `8×64` fp32 + EMA + sd | 6.37 G | 55 % | 42 % |
| A2/A1 `8×64` bf16 | 4.32 G | 37 % | 29 % |
| ghost-BN `4×128` bf16 + EMA + sd | 8.09 G | 69 % | 54 % |
| ghost-BN `4×128` fp32 + EMA + sd | **11.91 G** | ⛔ over | ✅ **79 %** |

⭐ **The fifth region costs 0.19 G and it is ALL `args`** (0.42 → 0.51): 25.6 M params × 4 B,
exactly one more region, temporaries unchanged. **Stochastic depth costs nothing measurable** —
16 masks of `tensor<64×f32>` is 4 KB.

⚠⚠⚠ **THE TWO BUDGET COLUMNS ARE THE FINDING, and the first draft of this file had only the
first.** 11.68 GiB is not the card — it is the GPU plugin's BFC default (`memory_fraction = 0.75`)
taken because `ffi/pjrt_ffi.c` called `PJRT_Client_Create` with **no create options**. Four
gigabytes of a 16 GB card were unreachable on the verified path, and
`XLA_PYTHON_CLIENT_MEM_FRACTION` could not reach it: the plugin does not read that variable — JAX's
Python layer reads it and passes `memory_fraction` as a create option. The shim now does the same.
XLA's own log line, both ways:

```
  unset                        : XLA backend allocating 11.68GiB (12543066112 bytes) for BFCAllocator
  LEAN_MLIR_MEM_FRACTION=0.97  : XLA backend allocating 15.11GiB (16221470720 bytes) for BFCAllocator
```

⚠ **A compile-time peak is not independent of the allocator it was compiled against** — the fp32
`4×128` reads 11.52 G under the default and 11.91 G under 0.97, because XLA rematerialises less
when it has room. Quote the budget alongside the peak.

⚠ Opt-in, and regression-checked: `shard-check convnext` returns `77.999521 e-9` / `0.041654`
bit-identically with the option set and unset.

▶ Host side, not in the table: the checkpoint blob is `nRegions × 25,557,032 × 4 B` — **409 MB** at
four regions, **511 MB** at five.

---

## What was NOT run here, and why

* **No A2/A1 render has been LOADED by the trainer.** Every gate above is CPU: XLA-on-CPU for the optimizer tie, text for the arity
  check, `#guard` for the predicates. A five-region graph has not yet been *loaded* by the trainer,
  and the shim's buffer-count refusal is the check that would say so. `arity_check.py` reproduces
  that arithmetic statically, which is the strongest cheap substitute and not a replacement.
* **No A2 or A1 training run.** These are 300- and 600-epoch schedules.
* **No re-measure of the phase-2 costs.** `runs/2026-08-27-jax-sb-tier-step-probe/` has them and §0
  of the planning doc says not to start there.

## Order of operations, recorded because the reverse is tempting

The refusal in `trainAdamSched` came off **LAST**. While it stood, a wrong render failed loudly at
load; the moment it was gone, a five-region graph packed wrongly trains and reports a number. The
renderer, the driver's pack/unpack, the three-way partition and the tie row all landed first.
