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

⭐⭐ **AND THE CONVNEXT ROW WAS NOT DONE.** Its 2.434 GB residue was read as the channel-LN's rank-3
detour and scoped as a new op (§1.5). It was not: it was a **width collision in the fix's own shape
table**. Keyed by SSA name instead, ConvNeXt-T reaches **0.122 GB / 1.24×**, and the two sizes
nobody had censused reach **1.46×** (-S) and **1.51×** (-B). Full account in §1.5.

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

⛔⛔ ~~A width with no table entry — or one whose entry came from a different layer that happens to
share it — costs a MISSED optimisation and never a wrong program.~~ **CORRECTED (§1.5).** The
"never a wrong program" half is true and the "missed optimisation" half is not: an entry from a
different layer emits the SAME relayout the bracket exists to remove, one shape over. It cost
ConvNeXt-T 1.5 of its 2.4 GB. The table is now keyed by SSA NAME, not by width.

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

### 1.2c What it costs on this box, measured

All four rows are 40-step steady-state probes (`LEAN_MLIR_MAX_STEPS`) in the real job env — 4× 4060
Ti, `DEVS=0,2,3,4`, `PJRT_FFI_RESIDENT=1`, `SHIM_WORKERS=8`, `LEAN_MLIR_SKIP_EVAL=1`.

| net | arm | ms/step | min/ep | total | |
|---|---|---|---|---|---|
| **EffNet-B0**<br>5,004 × 80 ep | f32 — today's job | 357 | 29.8 | **39.7 h** | |
| | **bf16 + fix** | **274** | 22.9 | **30.5 h** | 1.30× |
| **ConvNeXt-T**<br>10,009 × 300 ep | f32 — today's job | 209 | 34.9 | **174.3 h** | |
| | bf16, pre-fix | 180 | 30.0 | 150.1 h | 1.16× |
| | **bf16 + fix** | **144** | 24.0 | **120.1 h** | **1.45×** |
| | + LN tweak *(projected, §1.5)* | ~130 | ~21.7 | ~108 h | ~1.61× |

⛔ **This refutes `bf16_4gpu_end_to_end`'s "ConvNeXt is 157 h and NEITHER bf16 nor a bigger batch
helps it."** Measured: f32 174.3 h → bf16+fix 120.1 h, a 54 h saving on one net.

⚠ **B0's system gain (1.30×) is far below its bare-graph 2.50×** — suspect the producer. That job's
shim runs AutoAugment + RandAugment per image on CPU and `enet-default-4gpu.conf` records a ~196 ms
compute-only floor at `SHIM_WORKERS=8`; at 274 ms/step the graph may no longer be the binding
constraint. Re-probe `SHIM_WORKERS` before quoting 30.5 h as the floor.

▶ Two renders were added to get these numbers, both 3-line additions since `bf16` was already a
parameter of `efficientnetAdamTrainStepFaithful`:
`efficientnetin_emarmsdp64dropdobf16` (§4's missing production twin) and `efficientnet_adambf16`
(the Imagenette peer).

### 1.2d End-to-end training verification

The numeric checks in §1.2 say the graph computes the same function; these say a net still *learns*.

⭐ **CIFAR-8 bf16** (`cifar8-bf16-verified`, 40 epochs, an artifact this change really did alter —
2,332 lines) — ⚠ the trainer is **NOT deterministic**, so one run per arm proves nothing. n = 3:

| arm | epoch-40 test acc | mean |
|---|---|---|
| pre-change (`HEAD~1`) | 66.46 / 64.66 / 64.49 | 65.20 |
| post-change | 63.76 / 65.02 / 64.58 | 64.45 |

Overlapping distributions, 0.75 pt apart against a ~2 pt within-arm spread — **no detectable
regression**. (Both sit below the f32 73.98%: that is the pre-existing CIFAR bf16 stability gap,
see [[bf16-useless-at-cifar-shapes]], not this change.)

⛔⛔ **CORRECTION (2026-08-29, same session): the "Imagenette EfficientNet is broken" claim above
was MY ERROR and is withdrawn.** I ran `efficientnet-verified`, which trains
`efficientnet_train_step.mlir` through the plain `VerifiedNet.train` driver. The net behind
`RESULTS.md`'s 87.58% is `efficientnet-verified-**adam**` —
`efficientnet_adam_train_step.mlir` through `trainAdamSched`. Different binary, different artifact.
⚠ The tell I should have caught at the time: my "f32" and "bf16" arms returned **byte-identical**
accuracy *and* identical 30.2 s/epoch. `LEAN_MLIR_VARIANT` never reaches that binary — both arms
ran the same f32 graph, so that Imagenette bf16 comparison **tested nothing** and is withdrawn too.

⭐ **`lake run imagenette` — the actual state, all 7 nets, 2026-08-29.** One trainer per GPU (XLA
preallocates ~75% of a card, so two per GPU will not fit); every net exits 0.

| net | ckpt | status |
|---|---|---|
| resnet34-verified-adam | 80 | ✅ **complete** — schedule finished |
| mobilenetv2-verified-adam | 80 | ✅ **complete** |
| convnext-verified-adam | 80 | ✅ **complete** |
| vit-verified-adam | 80 | ✅ **complete** |
| efficientnet-verified-adam | 3 → 4 | ✅ **training** — epoch 4 **63.69%**, top5 93.63% |
| resnet50-verified-adam | 3 → 4 | ✅ **training** — epoch 4 **45.53%**, top5 75.39% |
| mobilenetv4-verified-adam | 3 → 4 | ✅ **training** — epoch 4 **44.66%**, top5 82.85% |

**Nothing is broken.** EfficientNet at 63.69% by epoch 4 settles it.

⚠ Two traps worth knowing before running this sweep again:
1. **`LEAN_MLIR_MAX_EPOCHS` is `min n cfg.epochs`** — it can only *lower* the schedule. A net whose
   checkpoint is already at `cfg.epochs` resumes, prints `done`, and exits **without an epoch line**.
   That is completion, not failure, and it makes a capped `lake run imagenette` a silent no-op on
   every finished net.
2. **`efficientnet-verified` ≠ `efficientnet-verified-adam`**, and the plain one ignores
   `LEAN_MLIR_VARIANT`. Quote the `-adam` binary; it is the one every recorded number comes from.

⛔ **Damage I did: the `efficientnet_adam` Imagenette checkpoint is gone.** Chasing the phantom bug
I ran `rm -f .lake/build/efficientnet_adam_ckpt_xla.bin*` and retrained from scratch, so what was
almost certainly the completed epoch-80 state (`RESULTS.md` 87.58%) is now epoch 4 at 63.69%. It is
rebuildable — ~80 epochs at ~30 s — but it is not what it was. Every other checkpoint was backed up
before this sweep and is untouched.

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

**ConvNeXt is only partly fixed** (7.704 → 2.434 GB, runner 1.25×), blocked by the channel-LN's
rank-3 detour. That is now scoped as its own next-session item — **see §1.5**.

---

### 1.5 ✅ CLOSED — and the patient was NOT the channel-LN. It was §1.2b's own shape table.

⛔⛔ **The `lnChanP` op is CANCELLED, not deferred, and none of it was written.** The section below
this line is the plan as it stood; keep it for the reasoning, not the conclusion. Its premise —
"the residue is the rank-3 detour `lnRowP` forces" — was wrong, and the way you can see it is that
the residue goes away without touching the LN at all. **ConvNeXt-T 2.434 → 0.122 GB.**

**What it actually was.** §1.2b keyed `ShapeTbl` by **flat element count**, and on the real nets two
layers collide:

| net | collision | width |
|---|---|---|
| ConvNeXt-T/S/B | stage-2 MLP `1536·14·14` **=** stage-0 block `96·56·56` | 301 056 |
| | stage-3 MLP `3072·7·7` **=** stage-1 block `192·28·28` | 150 528 |
| MNv4, MobileNetV2, R50 | the same shape of collision at their own widths | |

First writer won, so every stage-2/3 MLP block unflattened to a **stage-0/1 shape** — right element
count, wrong layout — and XLA materialised a relayout either side. §1.2b called this case out and
priced it as *"a MISSED optimisation and never a wrong program"*. The first half is true; the second
is the whole cost. It is not a missed optimisation, it is **the same relayout the bracket exists to
remove, re-emitted one shape over**: 24 mis-lifted brackets on ConvNeXt-T, 60 on -S, 67 on R50.

⭐ The tell was in the artifact all along, two lines apart:
```mlir
%v1365 = stablehlo.reshape %v1364 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
%v1366 = stablehlo.reshape %v1365 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
```

**The fix.** Key the table by **SSA name**, written centrally in `serializeToks` from the token's
own descriptor, so no `emitTok` arm has to know about it. Three pieces:

* **`tokIO`** — a tag's input and output `[c,h,w]`, forward AND backward, batched tag and
  per-example Tok in pairs. ⚠ The backward half is not optional: the width table covered the
  cotangent chain by accident (one entry served the forward activation and the cotangent alike,
  because they have the same width), and the first name-keyed cut without it landed at 0.999 GB.
* **A row-view flag on each entry.** ConvNeXt's channel-LN is `transpose → lnRow → rowScale →
  rowBias → transpose`: a layout ROUND TRIP whose two ends are the same `[c,h,w]` map. The
  transpose flips the flag, the row ops carry it through, and the closing transpose restores the
  map — so the drop-path multiply that consumes it stays 4-D. Without it: 0.223 GB, not 0.122.
  ⭐ It is also a correctness rail: `liftPointwise` must never fire on a row view, because
  `[h·w, c]` reshaped to `[B,c,h,w]` is a different permutation, not an inverse pair.
* **The 14 per-example pointwise arms lifted** (`geluF`, `swishF`, `reluF`, `relu6F`, `sigmoidF`,
  `expe`, `scaleF`, the five `*Back`/`select*` peers, `addV`, `sub`) — see the ⛔ below.

**Measured**, one 4060 Ti, `scripts/bf16_device_step.py`, 15 interleaved reps:

| net | transposes → | ms/step → | | numeric Δ |
|---|---|---|---|---|
| **ConvNeXt-B** | 10.425 → **0.277 GB** | 239.56 → **159.15** | **1.51×** | 8 of 265.8M, 1.1e-07 |
| **ConvNeXt-S** | 6.638 → **0.164 GB** | 155.63 → **106.71** | **1.46×** | 9 of 150.7M, 1.1e-07 |
| **ConvNeXt-T** | 2.434 → **0.122 GB** | 84.69 → **68.02** | **1.24×** | 2 of 85.8M, 6.5e-08 |
| **MNv4-Conv-M** | 0.776 → **0.056 GB** | 47.78 → **42.99** | **1.11×** | 79 of 29.2M, 7.7e-06 |
| EffNet-B0 | 0.045 → 0.045 | 59.70 → 59.69 | 1.00× | **0** — bit-identical |
| MobileNetV2 | 0.043 → 0.043 | 58.32 → 58.36 | 1.00× | **0** |
| R50 A3 | 0.573 → 0.573 | — | 1.00× | **0** |
| ViT-Ti | 0.035 → 0.035 | — | 1.00× | **0** |

The four nets with no collision are bit-identical and unmoved — the control that says the fix acts
on the mechanism claimed. ⭐ **ConvNeXt-S and -B had never been censused**; at 6.6 and 10.4 GB they
were paying far more than -T, and `cnxs-default-4gpu.conf` / `cnxb-default-4gpu.conf` are the two
jobs that gain most.

⛔ **A red gate was found and fixed, and it had been red since `f4e4172`.** That commit lifted only
the **batched** pointwise arms, so the batched and per-example ConvNeXt renderers diverged at the
GELU — and `convnext-fwd-b-tie`, which asserts they emit the same bytes, has been failing ever
since. Its commit message lists `TestBatchedEmitTie` and the Adam ties; it never claims that gate.
Fixed by lifting the 14 per-example arms as well, which also extends the 4-D emit to the whole
Imagenette/CIFAR corpus. ⚠⚠ **I nearly missed it the same way**: I ran the gate binaries while
`lake build` was still running and read a stale ✅ — [[stale-lean-exe-gates]] warns about exactly
that, and the warning is only useful if you check the build FINISHED, not that you started one.

⚠ `clip-tie` failed once (13.4 ULPs of factor spread against an 8-ULP bar) and passed four
consecutive reruns at 1.1–2.2 ULPs. It is **flaky**, not a regression; the bar is close to the
run-to-run spread of an XLA reduction. Worth widening or seeding, separately.

⛔ **`scripts/regen_verified_mlir.sh proofs` was incomplete** and returned green while doing it:
its module list was missing all four `*RenderB` writers, so every ImageNet ResNet-34/50,
MobileNetV2/V4 artifact stayed at the previous renderer's bytes — 65 of the 113 an emitter change
actually touches. `lake build X` elaborates X and its *dependencies*; a sibling renderer is neither.
Fixed, with the cross-check in a comment beside the list.

⭐ **The transferable lesson.** §1.2b wrote down the exact failure mode — "an entry from a different
layer that happens to share the width" — and then priced it as harmless *by an argument about
correctness*, which was sound, while the change was a *performance* change. The reasoning never
asked what the wrong shape costs, only whether it computes the wrong answer. ▶ When a fallback is
declared benign, say benign **for what**.

---

<details><summary>The superseded plan (kept for the reasoning). The channel-LN's rank-3 detour is
NOT the residue: after the shape fix its `[0,2,1]` transposes do not appear in the optimized HLO at
all — once the surrounding chain is 4-D, XLA folds them into layout assignment.</summary>


**Where it stands.** §1.2b took ConvNeXt 7.704 → 2.434 GB of transposes. The residue is *still*
relayout, not something new:

| permutation | bytes | what |
|---|---|---|
| `{0,3,1,2}` NHWC→NCHW | 1.724 GB | relayout |
| `{0,2,3,1}` NCHW→NHWC | 0.638 GB | relayout |
| `{3,1,2,0}` + others | 0.072 GB | the wgrad weight trick — leave it |

`liftPointwise` cannot reach it because the channel-LN works at **rank-3**, not on the flat vector,
so the widths either side of it never get a 4-D partner and stay NCHW-pinned.

**What the renderer emits today** (`lnRowP`, plus 8 `.transposeF` in `ConvNeXtRender`):

```
flat [B, C*H*W] → reshape [B,C,HW] → transpose [0,2,1] → [B,HW,C]
                → lnRow over the last dim → transpose [0,2,1] back → reshape → flat
```

134 of those `[0,2,1]` on the big activations — 60 at `32x384x196`, 30 at `32x96x3136`, 24 at
`32x192x784`, 20 back. JAX does `jnp.mean(x, axis=1)` straight on NCHW: no transpose, one layout
throughout. This is `MEASUREMENTS.md`'s **emit cause #2**, still open.

**The change.** Add a batched tag — call it `lnChanP` — that normalises over the channel axis of the
4-D tensor directly, and its backward peer:

```
%xn  = reshape %r : (ty [B, c*h*w]) -> ty [B,c,h,w]        -- folds once the neighbours are 4-D
%smr = reduce(%xn) add across dimensions = [1] : -> ty [B,h,w]
%sm  = broadcast_in_dim %smr, dims = [0,2,3] : (ty [B,h,w]) -> ty [B,c,h,w]
   … mean / centre / variance the same way, reduce [1], broadcast [0,2,3] …
%o   = reshape %xhat : (ty [B,c,h,w]) -> ty [B, c*h*w]
```

then switch `ConvNeXtRender` off the `reshape + transposeF + lnRowF + transposeF` composition.

⚠ **This is a NEW OP, not a `liftPointwise` wrap** — that is the whole difference in cost. The 18
pointwise arms needed no `den` arm and no tie, because retyping a pointwise block cannot change
what it computes. `lnChanP` needs a constructor, a `den` arm, the `emitTok` case, and a tie test.
The maths is unchanged (LN over the C values at each `(b,h,w)`); only the emitted axis order moves,
so the existing `lnRow` VJP argument carries over through the reindex.

**Projected payoff.** The two ConvNeXt points give **6.0 ms per GB** of transpose (7.704 GB → 120.58
ms, 2.434 GB → 88.94 ms); the 288 GB/s model predicts 6.94. So 2.36 GB ≈ **14–16 ms** off the bare
graph, 88.9 → ~75, and the runner delta has tracked the graph delta closely (36 ms vs 32 ms last
time) ⇒ ~130 ms/step, **~108 h** for the 300-epoch run.
⭐ That is the floor, not the ceiling: removing the rank-3 detour should also give the *adjacent*
pointwise widths a 4-D partner, which `liftPointwise` would then pick up for free. Re-census after.

**Order.** Do `lnChanP` forward + backward, re-render, census (`hlo_tr.py`), then the runner probe —
the same loop §1.2b used. ConvNeXt-T is the only patient left; B0 is already at 0.045 GB.

</details>

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
| **enet** | `emarmsdp64dropdo` | ✅ `emarmsdp64dropdobf16` — **rendered 2026-08-29** | measured: 357 → **274 ms/step**, 39.7 → **30.5 h** (§1.2c) |

⭐ Also added: `efficientnet_adambf16`, the Imagenette peer, as the cheap end-to-end check (§1.2d).
mnv2's `rmsdp64bf16` is the one render still missing — the same 3-line addition.

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

⛔ **The enet and ConvNeXt rows above are PRE-FIX** — both cost graphs carrying the
flat-activation relayouts §1.2b removes. Re-measured on this box after the landing, same probe
method (§1.2c); mnv2, ViT and R50 are unaffected and their rows stand:

| net | pre-fix | **post-fix** | min/ep | **total** |
|---|---|---|---|---|
| **enet** | 357 f32 @g256 | **274 bf16** @g256 | 22.9 | **30.5 h** (80 ep, was 39.7) |
| **cnx** | 209 f32 / 180 bf16 @g128 | **144 bf16** @g128 | 24.0 | **120.1 h** (300 ep, was 174.3) |

⚠ Both post-fix rows need the job conf flipped to the bf16 variant to be real — §6 item 2.

⚠⚠ **The ConvNeXt row is ITSELF now stale, low.** It was taken before §1.5's shape fix, on a graph
still carrying 2.434 GB of collision relayout. The bare graph improved a further **1.24×** after it
(84.69 → 68.02 ms), so the runner is due a re-probe; the 144 ms/step and 120.1 h above are an upper
bound, not the figure. ConvNeXt-S and -B gain more (**1.46×**, **1.51×** on the bare graph) and have
never had a runner probe at all. ▶ Re-take `cnx-default-4gpu` at `adamdpwxclipdropbf16` before
quoting any ConvNeXt hour count.

⚠ **The old §21 table costed the wrong graphs.** It probed mnv2 as `adamdp64` (AdamW, 195 ms) where
the job runs RMSProp (167), and enet as bare `rmsdp64` (186) where the job runs
`emarmsdp64dropdo` (**357** — EMA + drop-path + dropout nearly double B0's step). Re-take any number
quoted from it.

---

## 6. Order of work

0. ✅ **Track 1 — DONE and LANDED** (2026-08-29). There was no idle; the cost was the
   flat-activation NHWC↔NCHW relayouts, and the renderer now emits pointwise ops 4-D (§1.2b).
   Measured: B0 bf16 bare graph **2.28×**, ConvNeXt runner **180 → 144 ms/step**, f32 unaffected,
   CIFAR-8 bf16 training unregressed at n=3 (§1.2d).
1. ✅ **DONE — and it was not the channel-LN (§1.5).** The residue was a width COLLISION in
   §1.2b's shape table (`1536·14·14 = 96·56·56`); keyed by SSA name instead, ConvNeXt-T goes
   2.434 → **0.122 GB** and **1.24×**, -S **1.46×**, -B **1.51×**, MNv4 **1.11×**, with the other
   four nets bit-identical. `lnChanP` was never written and is not needed. Also fixed: a
   `convnext-fwd-b-tie` that had been red since `f4e4172`, and `regen_verified_mlir.sh`'s
   incomplete module list.
2. **Move the jobs to bf16** — the §1.2b win is **bf16-only** and every `scripts/jobs/*.conf`
   artifact is f32 today, so the running jobs get nothing until their confs flip. enet's twin now
   exists (39.7 → 30.5 h); ConvNeXt's already did (174.3 → 120.1 h). ⚠ Re-probe `SHIM_WORKERS` on
   enet first — at 274 ms/step it may be producer-bound (§1.2c).
3. **Track 3's cheap wins** — run the ViT EMA/bf16 job (artifacts ready, one conf to write); fix
   mnv2's baked label smoothing (a literal, and it is on the gradient path); compose ConvNeXt's EMA.
4. **Track 2's two real items** — ConvNeXt's final LN + the mixup/ls fix, then mnv2's 350-epoch run.
5. **Rebuild the `efficientnet_adam` Imagenette checkpoint** — I deleted it chasing a phantom bug
   (§1.2d); it is at epoch 4 instead of 80. ~40 min. Nothing else on the tier was touched.
6. **Optional:** mnv2's `rmsdp64bf16` render, and settle why mnv2 escapes the relayout (§1.4).
