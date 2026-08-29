# next_session_execution_and_parity.md — the execution path, then JAX→paper, then PJRT→JAX

**Opened 2026-08-29**, at the end of the session that measured all five 4-GPU nets on this box.
**Track 1 was executed the same day and is CLOSED** — it overturned the premise it was written on
and produced a 2.29× fix. §0 and §1 below are the corrected version; tracks 2 and 3 are unchanged.

▶ Companion artifact (tables, per-net A/B/C/D lists, the full sweep):
`https://claude.ai/code/artifact/0653ef09-028d-48ab-81b6-4ec2f2f90612`
⭐ Its §3 tells the *transpose* story, and §1 below now says that story was **right**. The header
warning that it was superseded is withdrawn.

---

## 0. START HERE — what changed, in one block

⛔⛔ **The "phase-4 is GPU-idle" finding is WITHDRAWN. It was a profiler artifact.** The capture
behind it ran `nsys profile -t cuda`, whose default `--cuda-graph-trace=graph` records a CUDA graph
as one opaque row and — in nsys' own help text — *"node activities will not be collected"*. XLA puts
most of the step inside command-buffer graphs, so **22 760 of 28 740 kernels/step (79%) never
appeared in the kernel table**, and every graph execution was scored as idle. Re-measured with
`--cuda-graph-trace=node`, EfficientNet-B0 bf16 @64, one 4060 Ti, 20 steady-state steps:

| per step | verified | JAX | (what graph-mode said) |
|---|---|---|---|
| kernels | **1437** | 1600 | 292 / 575 |
| wall ms | 135.96 | 62.43 | 135.60 / 63.21 |
| **GPU busy ms** | **133.02** | **57.70** | 24.19 / 25.63 |
| GPU idle ms | **2.94 (2.2%)** | 4.73 (7.6%) | 111.41 / 37.59 |
| **transpose ms** | **72.61 (54.6%)** | 0.75 (1.3%) | 0.476 / 0.675 |

The GPU is **97.8% busy**. There is no idle to find. Every claim resting on the 292-kernel table is
void: "half the kernels", "arithmetic 6% faster", "27.9 stalls/step", "transposes are 2% of GPU",
and the ~89 ms supposedly hidden inside one `Execute`. ⭐ `PJRT_FFI_TIMING`'s `device` bucket was
**right the whole time** — it reported 127 ms and the device really was busy for ~133. It needs no
split; §1.2's first move in the old plan is cancelled.

⭐⭐ **The transpose tax is real and it is the entire gap**, exactly as the companion artifact's §3
said. Transpose share predicts which nets pay, with no exceptions:

| net | wall vs JAX | transpose ms/step | share of GPU |
|---|---|---|---|
| EfficientNet-B0 | 2.21× slower | **72.61** | 54.6% |
| ConvNeXt-T | 1.89× slower | **62.73** | 53.2% |
| MobileNetV2 | 1.21× **faster** | 0.73 | 1.3% |
| ViT-Ti | 1.23× **faster** | 0.15 | 0.8% |

72.61 ms measured against 72.8 ms predicted by the 288 GB/s bandwidth model on 10.486 GB — the model
§1.3 dismissed as coincidence is accurate to 0.3%.

⭐⭐ **And it is fixed.** Rewriting the flattened pointwise activations back to 4-D (§1.2) removes
99.6% of B0's transpose bytes and **2.29×** its step, taking it past its JAX reference:

| net | transpose GB → | ms/step → | speedup |
|---|---|---|---|
| **EffNet-B0** | 10.486 → **0.045** | 135.53 → **59.07** | **2.29×** (JAX 61.84) |
| **ConvNeXt-T** | 7.704 → **3.301** | 120.58 → **88.94** | **1.36×** (partial, see §1.4) |
| MobileNetV2 | 0.043 → 0.043 | 58.17 → 58.59 | 0.99× |
| R50 A3 | 0.573 → 0.573 | 88.85 → 89.55 | 0.99× |

The two nets with nothing to remove are unchanged — the control that says the fix acts on the
mechanism claimed and not on something else.

---

## 1. TRACK 1 — CLOSED. The mechanism, the fix, and what the old §1.3 got wrong

### 1.1 The mechanism

The proof IR renders pointwise activations over a **flat vector** — `CnnRender.lean:14`, *"the proof
IR works in flat vectors … we bridge that with explicit reshape glue"* — so the ImageNet renderer
brackets every activation with a flatten and an unflatten. B0's swish, artifact lines 192–195:

```mlir
%v145 = stablehlo.reshape %v144 : (tensor<64x96x112x112xf32>) -> tensor<64x1204224xf32>
%v146 = stablehlo.logistic %v145 : tensor<64x1204224xf32>
%v147 = stablehlo.multiply %v145, %v146 : tensor<64x1204224xf32>
%v148 = stablehlo.reshape %v147 : (tensor<64x1204224xf32>) -> tensor<64x96x112x112xf32>
```

A `[N,C,H,W] → [N,C·H·W]` flatten is a free bitcast **only in NCHW layout**. XLA runs bf16 convs in
NHWC for the tensor cores (`dim_labels=b01f_o01i->b01f` on all 146 of B0's convs), so the flatten
pins the tensor to NCHW between two NHWC convs and XLA materialises a physical relayout each way.
B0's 266 optimized-HLO transposes are:

| permutation | bytes | count | what |
|---|---|---|---|
| `{0,2,3,1}` NCHW→NHWC | **5.843 GB** | 137 | the relayouts |
| `{0,3,1,2}` NHWC→NCHW | **4.636 GB** | 97 | the relayouts |
| `{3,1,2,0}` | 0.006 GB | 32 | the wgrad weight-transpose trick |

⚠ **The wgrad trick is 0.006 of 10.486 GB.** Every old-§1.3 item that aimed at it was aiming at
0.06% of the cost — which is why they all read as nulls.

⭐ **The dtype is the switch.** A 2×2×dtype synthetic (conv → BN → activation → conv, with and
without a flatten, with and without squeeze-excite):

| arm | transposes | bytes | conv layout |
|---|---|---|---|
| **f32**, any | **0** | 0 | `bf01_oi01->bf01` NCHW |
| bf16, 4-D activation | 4 | 0.174 GB | `b01f_o01i->b01f` NHWC |
| bf16, **flat** activation | 6 | **0.482 GB** | NHWC |

In f32 the conv never leaves NCHW and the effect **cannot appear**. Squeeze-excite makes **no
difference at all** — the SE and no-SE arms are identical in every row.

### 1.2 The fix, measured

`reshape` and pointwise ops commute, so the flat shape can simply be replaced by its 4-D partner:
both bracketing reshapes collapse to 4D→4D identities that XLA folds away and nothing else changes.
A scratch pass that does this per flat shape produced the §0 numbers.

**Numerically faithful**, seeded inputs, all outputs compared:

| artifact | differing words | max abs Δ |
|---|---|---|
| `efficientnetin_rms64` (f32) | 180 of 15,907,663 | **0.0** — signed zeros only |
| `convnextin_adamwxclipdrop` (f32) | 365 of 85,763,355 | 7.45e-09 |
| `efficientnetin_rms64bf16` | 4,688,885 of 15,907,663 | 2.19e-05 |
| `convnextin_adamwxclipdropbf16` | 10 of 85,763,355 | 1.86e-09 |

The f32 arms are exact; the bf16 spread is low-bit fusion-order noise at 2e-05.

⭐⭐ **B0's long-open "bf16 is only 1.03–1.10×" is SOLVED, and it was a render defect after all.**
The relayout traffic is **f32** — it happens before the convert — so it was a fixed cost bf16 could
not touch. Remove it and the cast pays properly:

| B0, f32 vs bf16 | f32 | bf16 | ratio |
|---|---|---|---|
| stock | 150.57 | 136.32 | **1.10×** |
| unflattened | 148.40 | **59.33** | **2.50×** |

The f32 arm barely moves (150.57 → 148.40): in f32 there were no relayouts to remove. That is the
mechanism confirmed from the other side.

### 1.2b ✅ LANDED in the renderer (2026-08-29)

`StableHLO.lean` now emits pointwise ops at their 4-D shape. Three pieces:

* **`ShapeTbl` / `EmitS`** — the emitter state is `Nat × ShapeTbl` (fresh counter + flat width ↦
  `[c,h,w]`) instead of a bare `Nat`. ⚠ It has to live in the STATE, not in a `pretty` argument: a
  net renderer calls `pretty` once per graph FRAGMENT and a conv lands in a different call from the
  activation that consumes it. Only the state is threaded across them.
* **`shapeTblOf`** — recovers the table from a token stream; conv/depthwise/BN/pool `Tok.batched`
  descriptors already carry their `[c,h,w]`. `pretty` folds each fragment's shapes in as it goes.
* **`liftPointwise` / `liftPointwise2`** — wrap one pointwise block in an inverse reshape pair so
  its ops run 4-D. **18 arms** converted (`swish`, `relu`, `relu6`, `gelu`, `sigmoidP`, `expeP`,
  `scale`, `shift`, `divConst`, `dropPathP`, `dropoutP`, the five `*BackP`/`select*P` peers, and
  `addV`/`sub`). Every value crossing a token boundary keeps its flat type, so **no other emit arm
  changed** and no type mismatch is possible.

⭐ A width with no table entry — or one whose entry came from a different layer that happens to
share it — costs a MISSED optimisation and never a wrong program: the bracket is an inverse reshape
pair, the identity at any shape with the same element count.

**Measured on the re-rendered artifacts** (59 changed; `scripts/regen_verified_mlir.sh proofs`):

| net | transpose GB → | bare graph ms → | speedup |
|---|---|---|---|
| **EffNet-B0 bf16** | 10.486 → **0.045** | 135.64 → **59.42** | **2.28×** |
| **ConvNeXt-T bf16 (DP)** | 7.704 → **2.434** | *(4-replica)* | see runner below |
| mnv2 / R50 / ViT bf16 | unchanged | unchanged | 0.99× — inert |
| **EffNet-B0 f32** (the production job) | 0 → 0 | 150.44 → 148.61 | 1.01× — no regression |

⭐ The renderer beats the scratch text pass on ConvNeXt (2.434 GB vs 3.301) because it knows the
true `[c,h,w]`, where inference from the emitted text could not: channel-LN only ever reshapes
those three shapes to rank-3.

⭐⭐ **End-to-end on the ImageNet runner** — ConvNeXt-T, real ImageNet, 4× 4060 Ti, the
`cnx-default-4gpu` env at `adamdpwxclipdropbf16`, `LEAN_MLIR_MAX_STEPS=40`:

```
before  180 ms/step        after  144 ms/step      1.25×
losses  7.622482 / 7.636338 / 7.541610   →   7.622416 / 7.636274 / 7.540822
```

30.0 → 24.0 min/epoch; over the paper's 300 epochs, 150 h → 120 h. The bare graph gains more
(1.36×) than the runner (1.25×) because ~55 ms/step of the runner's step is not the graph.

**Correctness.** `TestBatchedEmitTie` green on all six converted pointwise ops, plus the
EfficientNet / ConvNeXt / MobileNetV2 Adam ties; `regen_verified_mlir.sh`'s prefix and silent-emit
audits clean. Numerically the f32 render is **exact** — 177 differing words of 15,907,663 at
**max |Δ| = 0.0**, i.e. signed zeros only; bf16 differs at 2.19e-05, low-bit fusion-order noise.

⚠⚠ **Every production job artifact is f32 today**, so this buys them nothing until they move to
bf16 (§4's two missing renders). The win is real and measured, but it is a **bf16-only** win — in
f32 XLA keeps convs in NCHW and there are no relayouts to remove.

▶ The scratch pass and the `.hlo` dumps are in
`/tmp/claude-1000/-home-skoonce-lean-klawd-max-power-lean4-jax-mlir/ae511b87-3bd2-4b84-8cd3-ebc966d0902c/scratchpad/`
— `unflatten.py` (the pass), `busy2.py` (busy union with/without graph windows, the instrument that
caught the artifact), `kbreak.py` (per-kernel breakdown), `hlo_tr.py` (transpose bytes by
permutation), `gaps.py` (gap attribution).

### 1.3 The old "⛔ DEAD" list, corrected

The list was right that the *wgrad transpose trick* is not the cost. It was wrong to conclude the
**transpose line** was dead — that conclusion came from the broken profile.

1. **bf16 wgrad, cast before transpose** — still a genuine null. It targeted `{3,1,2,0}`, 0.006 GB.
2. **wgrad via permuted `dim_numbers`** — still correct: `transpose-folding` does delete the emitted
   weight transposes at pass 11. ⭐ The `iree#21955` corollary (`CNN.lean:1098` protects nothing
   worth having) stands.
3. **SE `dot_general` dragging the layout** — dead, and now for a *sufficient* reason: the synthetic
   in §1.1 varies SE directly and it changes nothing. The old evidence ("every conv is `{3,2,1,0}`")
   was true but not discriminating — mnv2's convs are identical and it pays nothing.
4. **The flat `[B, C·H·W]` reshape barrier** — ⭐⭐ **REINSTATED. This is the cause.** The synthetic
   that killed it was **f32**, where the conv never leaves NCHW, so the test was inert. Its other
   leg — "mnv2 flattens as much and pays nothing" — is true and still unexplained (§1.4).
5. **Large depthwise kernels** — that synthetic was f32 too. Re-test in bf16 before trusting it.
6. **Volume of non-default layout** — the observation stands; see §1.4.

⚠ The note that `layout_normalization` injects the bytes (0.006 → 8.597 GB against mnv2's 0.108) was
recorded and then dismissed as *"True, and irrelevant: the bytes are not the cost."* **The bytes are
the cost.** It was the right lead, discarded on the strength of the broken profile.

⭐ **The transferable lesson, sharpened.** "Profile before diagnosing" was the right lesson and it
still cost six hypotheses, because the profile itself was silently incomplete. Add: **check what
your profiler is not recording.** The tell was in the trace all along — a `CUDA_GRAPH_TRACE` table
with 82 rows/step sitting next to a kernel table that could not account for the wall clock.

### 1.4 What is still open

**Why does MobileNetV2 escape?** It has 138 flat activation regions of the same shape, between bf16
convs XLA also puts in NHWC (`b01f_o01i->b01f`, all 155), and it pays 0.043 GB. Ruled out: dtype
(both bf16), conv layout (identical), SE (mnv2 has none, but the synthetic says SE is irrelevant),
and fan-out of the flattened value (mnv2 73% multi-use vs B0 77%). Unexplained. It does not block
the fix — the fix is inert on mnv2 — but whatever it is may be the cheaper lever.

**ConvNeXt is only partly fixed** (7.704 → 3.301 GB, 1.36×). Three of its seven flat shapes are
still bracketed, blocked by channel-LN reshaping the flat tensor to rank-3 `[32,384,196]` rather
than back to 4-D. Closing those should take it toward JAX's 63.94 ms.

---

## 2. TRACK 2 — bring JAX to paper faithfulness

Audited from the **emitted** `jax/.lake/build/generated_*.py` (regenerated 2026-08-27), not the Lean
config. Ranked by (likely points) ÷ (cost).

### 2.1 Cross-cutting

* ⭐⭐ **NEW — label smoothing is silently dropped on every mixed step.** `_mixup`/`_cutmix` build
  their target from a raw `jax.nn.one_hot`, and `loss_fn` smooths only the hard-label branch
  (`y.ndim == 1`). timm's `Mixup(label_smoothing=0.1)` folds smoothing INTO the mixed target
  (`on = 1−s+s/K`, `off = s/K`). Mixup or CutMix fires **every** step for ConvNeXt-T, ViT-Ti and
  RSB-A2/A1, so those runs trained at an effective **ls = 0.0** while config, banner and ledger all
  say 0.1. RSB-A3 is unaffected (ls 0 by recipe).
  ▶ Fix in `Jax/Codegen.lean` where `_mixup`/`_cutmix` emit the one-hot — **not** in `loss_fn`: the
  soft-target branch must stay a pass-through or the shim's wire-v2 targets get double-smoothed.
* **BN momentum is hard-coded 0.99.** Correct for EfficientNet (TF's value). **Wrong by 10× for
  R50** — timm's PyTorch default is momentum 0.1 ⇒ decay 0.9, a 100-step window against our 1000.
  MobileNetV2's TF-slim reference is 0.997 `[unverified]`.
* **Mixup/CutMix switching is deterministic** (`step % 2`) vs timm's `switch_prob=0.5`. Same
  expectation, zero variance.
* **`jax.nn.gelu` defaults to the tanh approximation**; timm's `nn.GELU()` is exact erf. ConvNeXt
  and ViT. ⚠ Both paths agree with each other, so it is a paper gap, not a port gap.
* ✅ **CLOSED, do not re-open:** antialiasing (now `antialias=True` on both the train RRC and the
  eval resize, with timm's resize-then-crop ordering) and ViT's repeated augmentation.

### 2.2 Per net

| net | paper | ours | gap | what is left |
|---|---|---|---|---|
| **ViT-Ti** | 72.2 / 91.1 | **72.31 / 91.12** | **+0.1 ✅** | `default` recipe still Xavier — collapse it into `deit-init` so nobody quotes the wrong arm |
| **EffNet-B0** | 77.1 / 93.3 | 76.80 / 93.26 | −0.3 | BN eps 1e-5 vs the TF reference's 1e-3 `[unverified — one grep of timm]` |
| **ConvNeXt-T** | 82.1 | 81.10 / 95.37 | −1.0 | **no final LayerNorm between GAP and head** (paper is `GAP → LN → Linear`; re-confirmed in the emitted Python). Plus the ls=0 defect above. ~0.8 pt is unattributed after antialias/test-crop were spent |
| **R50 A3** | 78.1 | 77.22 | −0.9 | Ghost-BN (accum micro-steps normalise over 512, not 2048; `true-2048` needs ~80 GB, unrun); BN momentum 0.99 vs 0.9; BCE target thresholding `[check timm's bce_target_thresh default]` |
| **mnv2** | 72.0 | 68.77 @ 90ep | −3.23 | **the schedule** — `full` (350 ep) exists and has never run; LR 0.045 unscaled at bs256. The published number also used an offline post-hoc EMA the trainer does not do, and predates the ls→0 fix |

▶ **Highest value here is mnv2's 350-epoch run and ConvNeXt's final LN + ls fix.** Everything else
is sub-0.5 pt or already closed.

---

## 3. TRACK 3 — bring PJRT to JAX parity

Scope: the five `scripts/jobs/*-4gpu.conf` and the artifact each names.

⭐ **What needs no work:** the data pipeline is the *same code*. `generateShim` reuses
`build_imagenet_iter` verbatim and `scripts/gen_shims.sh` writes one shim per net from that net's own
`TrainConfig`, with no fallback. Augmentation, antialiasing and the eval protocol are identical by
construction. The one exception is mixup's λ (numpy vs `jax.random`) — distributional agreement
only, permanently.

| net | job artifact | what is missing vs JAX |
|---|---|---|
| **ViT** | `adamdp128x4wxclipdrop` | ✅ **all three deltas closed** by `68842de` (`vitInit`, `emadp128x4wxclipdropbf16`, bf16). ⛔ **but the job conf still names the pre-EMA, pre-bf16 artifact** — `scripts/jobs/vit-default-emabf16-4gpu.conf` (the other box's) is the model. Run it |
| **mnv2** | `rmsdp64` | ⛔ the render **bakes label smoothing α = 0.1** where the reference sets 0.0 — `MobileNetV2RenderB.lean:652`, a literal with no way to set 0, and it is on the **gradient** path (`dense<-0.000100>` in the artifact). No classifier dropout (JAX sets 0.2, no `do` variant exists). bf16 |
| **ConvNeXt** | `adamdpwxclipdrop` | **no EMA** — and the JAX headline *is* the EMA shadow's number. `convnextin_emadp` exists but nothing composes it with `wxclipdrop`. ⭐ ViT closed the identical gap in one call; ConvNeXt **derives** its entry name from the variant where ViT takes `funcName`, so the flag must reach the variant too. Also global 128 not 256 (`cBS` private) with the LR left at the bs-256 value, and no `cnxInit` |
| **enet** | `emarmsdp64dropdo` | bf16 only — this artifact already carries RMSProp + EMA(+`ema_bn`) + drop-path + classifier dropout + exp ×0.97/2.4 + LR 0.016 |
| **R50 A3** | `lambaccdp8x64bce` | the completed 77.43% run had **neither `wx` nor `clip`**; `lambaccdp8x64wxclipbce` + `scripts/jobs/r50-a3-wxclip-4gpu.conf` close it and the clip profiled free (206 ms/step). Unrun |

⚠ **D-list (differences that are NOT in JAX either)** — do not "fix" these toward JAX:
Ghost-BN granularity (verified accumulates 8×(4×64), JAX 4×(4×128) — neither is timm's);
ConvNeXt's `%loss` divisor carve-out (starts at 10.42 vs ln(1000)=6.91, report-only);
180–213 separate `stablehlo.all_reduce` ops vs JAX's single GSPMD collective;
`LEAN_MLIR_G2_STEPS=5000` dropping 4 micro-batches/epoch (0.08%) so 5000 = 8×625.

---

## 4. bf16 across the board

⭐ **Track 1 re-valued this UPWARD, not downward.** The "75% idle" premise is withdrawn (§0), and
bf16 is worth *more* than the ladder assumed once the relayouts go: B0 goes from 1.10× to **2.50×**
(§1.2). Re-measure each net's bf16 ratio on the unflattened graph — the stock ratios understate it
wherever the transpose share is large.

**What exists vs what a job would load:**

| net | job variant | bf16 peer | status |
|---|---|---|---|
| ConvNeXt | `adamdpwxclipdrop` | `…dropbf16` | ✅ exists |
| ViT | `adamdp128x4wxclipdrop` | `…dropbf16`, `emadp128x4wxclipdropbf16` | ✅ exists |
| R50 A3 | `lambaccdp8x64bce` | `lambaccdp8x64wxclipbcebf16` | ✅ exists |
| **mnv2** | `rmsdp64` | ⛔ **none** — the bf16 renders are the **AdamW** family | render `rmsdp64bf16` |
| **enet** | `emarmsdp64dropdo` | ⛔ **none** — `rmsdp64bf16` carries no EMA/drop/dropout | render `emarmsdp64dropdobf16` |

⭐ **B0's long-open "bf16 is only 1.03–1.10×" is SOLVED** — and it *was* a render defect. The
flat-activation relayouts are f32 and sit before the cast, so bf16 could not touch them. Unflattened,
B0's bf16 ratio is **2.50×** (§1.2).

---

## 5. What we are targeting

| net | paper | JAX (phase 2) | PJRT (phase 4) |
|---|---|---|---|
| **R50 RSB-A3** | 78.1 | 77.22 ✅ | **77.43 ✅ done** — ahead of the reference |
| **ViT-Ti / DeiT-Ti** | 72.2 / 91.1 | **72.31 / 91.12 ✅** | ◀ next run, artifacts ready |
| **ConvNeXt-T** | 82.1 | 81.10 / 95.37 | TBD |
| **EfficientNet-B0** | 77.1 / 93.3 | 76.80 / 93.26 | TBD |
| **MobileNetV2** | 72.0 | 68.77 @ 90ep (350ep unrun) | TBD |

**Throughput, measured on this box 2026-08-29** (4× 4060 Ti, AER-clean four, real ImageNet,
600-step steady state, job-accurate variants):

| net | JAX bf16 | min/ep | verified | min/ep | ratio |
|---|---|---|---|---|---|
| mnv2 | 85.0 ms @g256 | 7.09 | 153 **f32** @g256 | 12.76 | 1.80× |
| enet | 106.5 @g256 | 8.88 | 357 **f32** @g256 | 29.8 | 3.35× |
| cnx | 179.0 @g256 | 14.93 | 178 bf16 @g128 | 29.70 | 1.99× |
| vit | 159.0 @g512 | 6.63 | **146 bf16-EMA** @g512 | **6.09** | **0.92×** |

⭐ **ViT's verified path is already FASTER than its reference here.** ⚠ The same artifact runs at
375 ms/step on the 24-core box — phase-2 is ~equal on both machines while phase-4 is 2.6× apart, so
that box is producer-starved and `SHIM_WORKERS` is worth probing there (ares' "16 is worse than 8"
was measured on a box that was not starved and does not transfer).

⛔ **This whole table is PRE-FIX.** The enet and ConvNeXt rows cost graphs carrying the
flat-activation relayouts §1.2 removes; on the bare graph B0 goes 135.53 → 59.07 ms and ConvNeXt
120.58 → 88.94. Re-take both rows after the renderer change. mnv2, ViT and R50 are unaffected.

⚠ **The old §21 table costed the wrong graphs.** It probed mnv2 as `adamdp64` (AdamW, 195 ms) where
the job runs RMSProp (167), and enet as bare `rmsdp64` (186) where the job runs
`emarmsdp64dropdo` (**357** — EMA + drop-path + dropout nearly double B0's step). Re-take any number
quoted from it.

---

## 6. Order of work

0. ✅ **Track 1 — DONE** (2026-08-29). No idle existed; the cost was the flat-activation relayouts,
   and a scratch pass removes them (§1.2). Nothing further is needed to *understand* it.
1. **Land the unflatten in the renderer** — emit pointwise activations on the 4-D type (§1.2). This
   is now the single largest item in the plan: **2.29× on B0, 1.36× on ConvNeXt**, inert on the
   other three, and it is worth more than every bf16 item in §4 combined. Re-render, re-run the
   verified-vs-JAX gates, then re-take §5's throughput table — the enet and ConvNeXt rows there are
   pre-fix and will be badly wrong.
2. **Track 3's cheap wins** — run the ViT EMA/bf16 job (artifacts ready, one conf to write); fix
   mnv2's baked label smoothing (a literal, and it is on the gradient path); compose ConvNeXt's EMA.
3. **Track 2's two real items** — ConvNeXt's final LN + the mixup/ls fix, then mnv2's 350-epoch run.
4. **bf16 gaps** (§4), re-measured on unflattened graphs — the stock ratios understate them.
5. **Optional, cheap:** finish ConvNeXt's remaining three flat shapes, and settle why mnv2 escapes
   (§1.4).
