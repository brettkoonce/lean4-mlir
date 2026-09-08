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

⛔⛔ **THE EFFNET ROWS BELOW COSTED THE WRONG SCHEDULE.** They divide by **80 epochs**;
`efficientnetImagenetConfig.epochs` is **350** (`MainEfficientNetImagenet.lean:35`), and the trainer
anneals over the constant, not over the job conf. ⚠ Worse, `enet-default-4gpu.conf` *set* `EPOCHS=80`
— and that is supervise.sh's **STOP** condition (`last_epoch >= EPOCHS` ⇒ ✅ COMPLETE), so the run
would have been killed a quarter of the way through its own LR curve and reported success. Fixed in
the conf 2026-08-30. The ConvNeXt rows are fine: 300 is that net's real schedule.

| net | arm | ms/step | min/ep | total | |
|---|---|---|---|---|---|
| **EffNet-B0**<br>5,004 × ~~80~~ **350 ep** | f32 — today's job | 357 | 29.8 | ~~39.7~~ **173.7 h** | |
| | **bf16 + fix** | **280** | 23.4 | ~~30.5~~ **136.2 h** | 1.28× |
| **ConvNeXt-T**<br>10,009 × 300 ep | f32 — today's job | 209 | 34.9 | **174.3 h** | |
| | bf16, pre-fix | 180 | 30.0 | 150.1 h | 1.16× |
| | bf16 + pointwise fix | 144 | 24.0 | 120.1 h | 1.45× |
| | **bf16 + shape fix (§1.5)** | **131** | 21.9 | **109.3 h** | **1.60×** |

⭐ The last ConvNeXt row was the *projection* "~130 ms, ~108 h, ~1.61×" for the channel-LN op. The op
was never written — a different fix landed (§1.5) — and it arrived within 1 ms of the projection.
Re-probed 2026-08-30 alongside -S (**192 ms**, 160.1 h) and -B (**301 ms**, 251.1 h).

⛔ **This refutes `bf16_4gpu_end_to_end`'s "ConvNeXt is 157 h and NEITHER bf16 nor a bigger batch
helps it."** Measured: f32 174.3 h → bf16+fix 120.1 h, a 54 h saving on one net.

⚠ **B0's system gain (1.28×) is far below its bare-graph 2.50×** — the suspicion was the producer,
since that job's shim runs AutoAugment + RandAugment per image on CPU and `enet-default-4gpu.conf`
records a ~196 ms compute-only floor at `SHIM_WORKERS=8`.
✅ **ANSWERED 2026-08-30: it is not the producer.** `SHIM_WORKERS=16` measures **293 ms/step against
8's 280** — worse, not better — so 8 is not starving this net and the remaining gap is elsewhere.
`SHIM_WORKERS=8` stands and 136.2 h is the floor as measured.

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

* ✅ **Label smoothing was silently dropped on every mixed step — CLOSED 2026-08-30, see §7.2.**
  `_mixup`/`_cutmix` now emit timm's smoothed one-hot (`on = 1−s+s/K`, `off = s/K`) through one
  shared `softOneHot` binding; `loss_fn`'s soft branch stays a pass-through. ⚠ Two corrections to
  this bullet as it stood: it was **phase 2 only** (the verified cotangent already smooths whatever
  `%onehot` carries), and **RSB-A2/A1 were never affected** — they set `labelSmoothing := 0.0` by
  recipe. The nets that were: ViT-Ti/S/B and ConvNeXt-T/S/B, all six.
* ✅ **BN momentum was hard-coded 0.99 — CLOSED 2026-08-30, see §7.3.** It is now
  `TrainConfig.bnMomentum` / `VerifiedConfig.bnMomentum`, default 0.99, and **R50 is set to 0.9**
  on both paths. ⚠ The timm audit there corrects this bullet: timm 1.0.28 runs decay 0.9 on
  *every* BN net including `tf_efficientnet_b0`, which changes only `bn_eps`.
* **Mixup/CutMix switching is deterministic** (`step % 2`) vs timm's `switch_prob=0.5`. Same
  expectation, zero variance.
* **`jax.nn.gelu` defaults to the tanh approximation**; timm's `nn.GELU()` is exact erf. ConvNeXt
  and ViT. ⚠ Both paths agree with each other, so it is a paper gap, not a port gap.
* ✅ **CLOSED, do not re-open:** antialiasing (now `antialias=True` on both the train RRC and the
  eval resize, with timm's resize-then-crop ordering) and ViT's repeated augmentation.

### 2.2 Per net

| net | paper | ours | gap | what is left |
|---|---|---|---|---|
| **ViT-Ti** | 72.2 / 91.1 | ~~**72.31 / 91.12**~~ ⚠ stale | ~~+0.1~~ | ⭐⭐ **mixup dropped label smoothing** — ✅ fixed 2026-08-30 (§7.2), so this cell describes a net the code no longer emits and the **+0.1 no longer stands**. Plus: `default` recipe still Xavier — collapse it into `deit-init` so nobody quotes the wrong arm |
| **EffNet-B0** | 77.1 / 93.3 | 76.80 / 93.26 | −0.3 | ⭐⭐ **the JAX stem and head ran ReLU where the net is swish** — ✅ fixed 2026-08-30, so this cell is stale. Plus BN eps 1e-5 vs the TF reference's 1e-3 — ✅ **now verified**, and it is the one axis on which timm agrees with the paper; see §2.3 |
| **ConvNeXt-T** | 82.1 | ~~81.10 / 95.37~~ ⚠ stale | −1.0 | ✅ **both items fixed 2026-08-30** — the missing head LN (§7.1, which also brought all three sizes onto timm's parameter count exactly) and the ls=0 defect (§7.2). Nothing here is open; the cell is stale and the −1.0 is unattributed until a re-run. ~0.8 pt was unattributed after antialias/test-crop even before these |
| **R50 A3** | 78.1 | 77.22 | −0.9 | Ghost-BN (accum micro-steps normalise over 512, not 2048; `true-2048` needs ~80 GB, unrun); ~~BN momentum 0.99 vs 0.9~~ ✅ fixed 2026-08-30, §7.3 — **so this cell is stale, unrun**; BCE target thresholding `[check timm's bce_target_thresh default]` |
| **mnv2** | 72.0 | 68.77 @ 90ep | −3.23 | ⭐⭐ **the JAX stem and head ran ReLU where the net is ReLU6** — ✅ fixed 2026-08-30, so this cell is stale. Plus **the schedule** — `full` (350 ep) exists and has never run; LR 0.045 unscaled at bs256. The published number also used an offline post-hoc EMA the trainer does not do, and predates the ls→0 fix |

⭐⭐ **✅ FIXED 2026-08-30 — TWO JAX references ran the wrong activation on their stem and head
convs, and the second one was found by auditing for the first.** `Types.lean:378` defaults `NetSpec.convBnAct := .relu`; `jax/MainEfficientNet.lean:14`
(the **Imagenette** B0) sets `.swish`, and `jax/MainEfficientNetImagenet.lean:14` — the ImageNet one,
which is the phase-2 reference behind **76.80 / 93.26** — never got the same line. Verified in the
emitted Python: `generated_efficientnet_b0.py` has `swish` at both sites, and
`generated_efficientnet_b0_imagenet.py` has `jax.nn.relu` (lines 1595 and 1630).
▶ **Three consequences.** (1) The fix is one line, and `Codegen.lean:2107`'s own comment says this
emitter *"was an unconditional `jax.nn.relu(x)` … EfficientNet's was ReLU where the net is swish"* —
so the fix was made and applied to one of the two specs. (2) It is a plausible share of the −0.3.
(3) ⚠⚠ **B0's phase-2 ↔ phase-4 comparison is not like-for-like**: the VERIFIED render is swish
throughout (194 `stablehlo.logistic`, **zero** `stablehlo.maximum`), i.e. the port is more
paper-faithful than the reference it is scored against.
⛔ **`scripts/enet_forward_tie.py` cannot catch this** — it ties the verified render against
`generated_efficientnet_b0.py`, the Imagenette file, which is the one that is already correct.
A tie that green-lights the net whose twin has the defect is testing the wrong pair.

⭐⭐ **AND THE SAME OMISSION HIT MobileNetV2.** Auditing `convBnAct` across all 25 `NetSpec`s: only
two specs in the repo set it at all, and in **both** cases the Imagenette twin has the line and the
ImageNet one does not.

| spec | should be | was emitting | twin that had it right |
|---|---|---|---|
| `efficientNetB0Imagenet` | `.swish` | `jax.nn.relu` | `efficientNetB0` |
| `mobilenetV2Imagenet` | `.relu6` | `jax.nn.relu` | `mobilenetV2` |

Both verified renders were already correct — B0 is 194 `stablehlo.logistic` / **zero**
`stablehlo.maximum`; MNv2 is 35 `maximum` paired with 35 `minimum`, the ReLU6 clamp, at the
32×112×112 stem included. ⚠ **MobileNetV2 is the net with the largest paper gap in the fleet**
(−3.23, 68.77 @ 90 ep vs 72.0) and the schedule was the only suspect on the list.

**✅ Both fixed**, one line each; regenerated and diffed to exactly two lines per file (the stem and
the 1×1 head), shims **byte-identical** (`generateShim` emits the data pipeline, which `convBnAct`
cannot reach), both trainers parse and run a forward pass.

▶ **Also added: `--emit`** (`jax/Jax/Runner.lean`), the counterpart of the existing `--shim`.
`runJax` writes the trainer and then immediately spawns python on it, so before this the only way
to refresh a `generated_*.py` after a spec change was to start a real ImageNet run and race a
timeout against it.

⚠⚠ **BOTH PHASE-2 NUMBERS ARE NOW REFERENCES FOR A NET THE CODE NO LONGER EMITS.** EfficientNet's
**76.80 / 93.26** and MobileNetV2's **68.77** were trained with ReLU stems and heads. They are not
wrong as history, but they no longer describe what `lake exe … ` will train, and the phase-4 rows
they are compared against never had the defect. ▶ Re-running them is the only way to know what the
fix is worth; until then treat both phase-2 cells as **stale, direction unknown**.

▶ **Highest value here is mnv2's 350-epoch run and ConvNeXt's final LN.** Everything else
is sub-0.5 pt or already closed. (The ls fix landed 2026-08-30, §7.2.)

---

### 2.3 ⚠ EfficientNet-B0 is the one net chasing a reference timm does NOT implement

Written down 2026-08-30 because B3 (§7.3) kept tripping over it, and because every future
"just check timm" on this net will trip over it the same way. **B0's target, 77.1 / 93.3, is Tan &
Le's TensorFlow number.** timm has no weights trained to that recipe — its `efficientnet_b0` tags
are `ra_in1k` and `ra4_e3600_r224_in1k`, both timm's own recipes, and the `tf_efficientnet_b0` tags
are *ported* TF weights, not a reproduction. So for this net timm is a **transcription of the
architecture, not of the recipe**, and reading a hyperparameter off it can silently move us AWAY
from the number we are scored against.

**Measured against the pinned timm 1.0.28 in `.venv-timm`** (`create_model(name).modules()` and
`get_pretrained_cfg`), beside the TF paper and beside what we emit:

| | TF paper (our target) | timm `efficientnet_b0` | timm `tf_efficientnet_b0` | **we emit** |
|---|---|---|---|---|
| BN **momentum** (decay) | 0.99 | 0.1 → **0.9** | 0.1 → **0.9** | **0.99** ✅ paper |
| BN **eps** | 1e-3 | 1e-5 | **1e-3** | **1e-5** ⛔ |
| crop_pct | 0.875 | 0.875 | 0.875 | 0.875 ✅ |
| interpolation | bicubic | bicubic | bicubic | bicubic ✅ |

⛔⛔ **`tf_efficientnet_b0` does NOT carry TF's BN momentum, and this is a live trap.**
`BN_MOMENTUM_TF_DEFAULT = 1 - 0.99` sits in `timm/models/_efficientnet_builder.py:36` next to
`BN_EPS_TF_DEFAULT = 1e-3`, and reads exactly like the pair a `tf_*` model would install. It is
not: its only consumer `get_bn_args_tf()` **has no caller in 1.0.28**, and every `tf_*` constructor
does `kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)` and nothing else. So the port kept TF's eps
and silently took PyTorch's momentum. ▶ A grep that finds the constant and stops there concludes
the opposite of the truth.

▶ **Consequences, and they cut both ways:**
1. **B0 stays at `bnMomentum := 0.99`** (§7.3 left it there) and that is now a recorded decision,
   not an untouched default. It tracks the paper. Flipping it to timm's 0.9 would be tracking a
   recipe we are not being scored against.
2. **BN eps is the one axis where timm and the paper AGREE and we differ.** We emit `eps=1e-5`
   because `Jax/Codegen.lean`'s `_bn` still hard-codes it — the exact shape B3 just fixed for
   momentum, one field over. ▶ `TrainConfig.bnEps` is the same one-line addition; it would change
   B0's emitted graph, so it is a decision rather than a cleanup, and it is UNRUN either way.
   ⚠ It is not obviously small: 1e-3 vs 1e-5 is 100× the variance floor, and B0's depthwise
   activations are exactly where a variance floor bites.
3. ⚠ **Do not "fix" B0 toward timm on any axis without saying which reference the run is claiming.**
   The same question decides `rmspropEps` (we use 1e-3, EfficientNet's value, not PyTorch's 1e-8),
   the `×0.97 / 2.4 ep` exponential LR, and AutoAugment-vs-RandAugment.

⚠ **The epoch axis is the same question again, and it is unsettled.** `efficientNetB0ImagenetConfig`
runs **80** epochs; the paper's schedule is **350**, and the `full` recipe (`epochs := 350`) exists
and has never run. The 76.80 / 93.26 that sits 0.3 under a 350-epoch paper number was produced by
the 80-epoch tier — which is either a very good result or a sign the comparison is not what it
looks like. ▶ Worth resolving before attributing that −0.3 to anything.
⭐ For contrast, **R50 is the opposite case**: its target IS timm's (RSB, `arXiv:2110.00476`, and
`resnet50.a1_in1k` even carries the paper id), which is why reading BN momentum off timm was the
right move there and would be the wrong one here.

### ⭐⭐ The general rule, and it is predictable rather than case-by-case

`tf_efficientnet_b0` differs from `efficientnet_b0` in **exactly two** things — 5 `Conv2dSame`
(TF's asymmetric SAME on the strided convs) and `bn_eps = 1e-3`. Same SiLU, same channel rounding,
same BN **momentum**. That is not an oversight; it is the port's scope showing through:

> **timm's `tf_*` models port what changes the FUNCTION, not what changes the OPTIMIZATION** —
> they exist to load TF *weights*, and BN momentum has no effect at inference.

| ported into a `tf_*` model | left at PyTorch defaults |
|---|---|
| padding (`Conv2dSame`), BN eps, activation, channel rounding | **BN momentum**, optimizer, LR schedule, EMA, augmentation |

⚠⚠ **That split lands exactly on our fault line, because we TRAIN FROM SCRATCH.** A `tf_*` model is
a good reference for the forward pass and a bad one for the recipe, and nothing in timm marks which
half you are reading. ▶ Applies to every TF-origin net here: MobileNetV2/V3, EfficientNet-B0/V2,
MNv4.

⭐ **The near-miss that proves it is already in this repo, and we got it right by other means.**
Plain `mobilenetv2_100` has **0** `Conv2dSame` — timm's MobileNetV2 is not the TF MobileNetV2
either, differing on every strided conv. We emit TF `SAME` there deliberately; `Jax/Codegen.lean`'s
`conv2d` comment says so: *"Scoped to the ResNet-family helpers ON PURPOSE: MobileNetV2/
EfficientNet emit their own `conv_general_dilated` with 'SAME' and are TF-origin ports, where
asymmetric 'SAME' IS the reference. Do not 'fix' those."* Anyone "checking against timm" would have
concluded the opposite — the same way `BN_MOMENTUM_TF_DEFAULT` reads.

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
| **enet** | `emarmsdp64dropdo` | ✅ `emarmsdp64dropdobf16` — **rendered 2026-08-29** | measured 2026-08-30 post-fix: **188 → 107 ms/step**, 91.5 → **52.1 h** over 350 ep (§6.1) |

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
| **ViT-Ti / DeiT-Ti** | 72.2 / 91.1 | 72.31 / 91.12 ⚠ stale | ◀ next run, artifacts ready |
| **ConvNeXt-T** | 82.1 | 81.10 / 95.37 ⚠ stale | TBD |
| **EfficientNet-B0** | 77.1 / 93.3 | 76.80 / 93.26 ⚠ stale | TBD |
| **MobileNetV2** | 72.0 | 68.77 @ 90ep ⚠ stale (350ep unrun) | TBD |

⚠⚠ **EVERY cell in this table is now stale, and R50's is stale on the OTHER side.** B0 and mnv2
ran ReLU stems and heads where the nets are swish / ReLU6 (§2.1); ViT-Ti and ConvNeXt-T trained at
an effective ls = 0.0 because mixup dropped the smoothing (§7.2); **ConvNeXt-T additionally had no
head LayerNorm on EITHER path** (§7.1); and R50's verified BN decay moved 0.99 → 0.9 (§7.3). All
fixed 2026-08-30, none of it measured. §7.6 has the run list.

⚠⚠ **None of these phase-2 ↔ phase-4 comparisons is like-for-like until the references re-run.**
⭐ **The pattern held three times and then broke on the fourth, and the exception is the
interesting one.** For `convBnAct`, mixup-smoothing and BN momentum the verified renderer was the
more faithful side — hand-written per net and audited op by op, against a JAX emitter whose
defaults are shared across 25 `NetSpec`s where a missing line is silent. **B1 is the opposite**:
the port was made wrong ON PURPOSE, in §2m, to agree with the reference. ▶ So "the port is usually
right" is not the lesson. The lesson is that agreement between our two phases is worth nothing as
evidence — both were wrong together on B1 and on B3 — and the only thing that settled it was an
external number, timm's parameter count.

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

⛔⛔ **THE ENET ROW IS SUPERSEDED TWICE OVER — see §6.1b for the live numbers.** Its 274/22.9/30.5 h
costed the wrong variant (bare `rmsdp64bf16`, not the job's `emarmsdp64dropdobf16`) *and* the wrong
epoch count (80, not 350); the job-accurate figure was **281 ms**, and after the `dropoutMask` fix
(§6.1) it is **107 ms — 8.92 min/ep, 52.1 h over 350 epochs, 1.00× its JAX reference**. Every enet
throughput number written before 2026-08-30 afternoon is void.
⚠ The **cnx** row's 144 ms is likewise superseded by §1.5's 131 and then by §6.1b's 142 — take
§6.1b's, which was measured in the same session as everything it is compared against.

⭐⭐ **RE-PROBED 2026-08-29 after the shape fix — all three ConvNeXt sizes, both precisions, one
session, 4× 4060 Ti, 40-step probes through the real job env.** The 144 ms/step row above is
superseded. Hours are `ms × 10,009 × 300 / 3.6e6` (this doc's convention, eval+ckpt EXCLUDED; the
job-conf headers include the extra ~3.1 h):

| net | f32 (what the conf names) | bf16 pre-fix | **bf16 now** | flip is worth |
|---|---|---|---|---|
| **ConvNeXt-T** | 206 ms — 171.8 h | 144 ms — 120.1 h | **131 ms — 109.3 h** | 63 h |
| **ConvNeXt-S** | 345 ms — 287.8 h | 243 ms — 202.7 h | **192 ms — 160.1 h** | 128 h |
| **ConvNeXt-B** | 521 ms — 434.7 h | 379 ms — 316.1 h | **301 ms — 251.1 h** | 184 h |

⭐ **The f32 arm is INERT and that was A/B'd, not assumed** — ConvNeXt-T 207 → 206, ConvNeXt-B
528 → 521, same swap discipline as the bf16 pair. f32 keeps its convs in NCHW, so there is no
layout to convert.

⭐ **ConvNeXt-S and -B gain more than -T, in ratio AND in hours** (1.27× / 1.26× vs 1.10× on the
runner), because the graph is a larger share of a bigger step. ⚠ The runner ratios are below the
bare graph's (1.46× / 1.51× / 1.24×) — ~55 ms/step of -T's step is not the graph at all, and as the
graph shrinks the shim feed and all-reduce become a larger fraction: the trainer step was ~1.52× the
graph when §8.6 was written and is ~1.85× now.

▶ Losses tracked step for step: -S bit-identical over the first three, -T and -B differing in the
last bits (bf16 fusion order). ▶ `blueprint/src/content.tex` §8.5 and §8.6 carry these numbers now.

⚠ **The old §21 table costed the wrong graphs.** It probed mnv2 as `adamdp64` (AdamW, 195 ms) where
the job runs RMSProp (167), and enet as bare `rmsdp64` (186) where the job runs
`emarmsdp64dropdo` (**357** — EMA + drop-path + dropout nearly double B0's step). Re-take any number
quoted from it.

---

## 6. Order of work

⭐ **Everything below §6.0 was closed on 2026-08-29/30 and pushed to `main` at `c4272d5`** (five CI
workflows green). ⭐ §6.1 and §6.2 were then CLOSED on 2026-08-30 afternoon — EfficientNet's
2.63× turned out to be a 150 ms/step host-side RNG loop and is now 1.00×.

▶▶ **THE LIVE QUEUE IS NOW §7**, a self-contained brief for the three open paper diffs
(ConvNeXt's missing final LN, mixup dropping label smoothing, BN momentum). Throughput work is
done; §6.3/§6.4 are the leftovers and the rest of §6 is the record.

### 6.1 ✅ CLOSED — EfficientNet's 2.63× was a host-side RNG loop, and it is gone

⭐⭐ **B0 is now at parity with its JAX reference: 281 → 107 ms/step, 2.63×.** The gap was never
the lowerer and never the producer. It was `F32.dropoutMask`, a pure-Lean Bernoulli loop the
trainer ran once per step, costing **150 ms** — more than twice B0's whole 71.6 ms graph.

**The measurement §6.1 asked for, taken first** (`bf16_device_step.py --replicas 4`, the job's own
`emarmsdp64dropdo` pair). The graph was never the problem — 71.6 ms puts it alongside ConvNeXt-T's
68.0 and MNv2's 68.4, and bf16 pays a clean 2.23× on it:

| B0, 4 replicas | bare graph | trainer step | **not the graph** |
|---|---|---|---|
| f32 (`emarmsdp64dropdo`) | 159.94 ms | 357 | 197 ms (55%) |
| bf16 (`…dropdobf16`) | **71.60 ms** | 281 | **208 ms (74%)** |

⭐ The overhead is the SAME ~200 ms in both arms — a fixed cost, not a scaling one. That is the
tell: it cannot be anything the graph does.

**Then the split that named it.** `LEAN_MLIR_BENCH_SYNTH=1` removes the shim read and nothing else:

| bf16 term | ms/step |
|---|---|
| bare graph @ 4 replicas | 71.6 |
| **host-side, not the graph** | **173.4** |
| shim read not hidden by the depth-8 prefetch | 36 |
| trainer step | 281 |

⛔ **This retires the `SHIM_WORKERS=16` argument, which was true but not evidence.** "16 measures
293 against 8's 280, so B0 is not producer-bound" happens to reach the right conclusion from a
premise that does not support it — 16 being worse is equally consistent with CPU oversubscription
on top of a starved feed. Deleting the producer outright is the measurement, and it recovers 36 ms.

**The cost, timed against the compiled objects:**

| per step | ms |
|---|---|
| **`F32.dropoutMask`, enet job shape (256×1280)** | **150.07** |
| the same at 32×1280 — the shape its docstring costed | 18.74 |
| `F32.dropScales` (9 sites × 256) | 1.83 |
| `F32.const`, same 327,680 floats, `@[extern]` C | **0.073** |

It pushed 327,680 boxed Floats into an `Array Float` and tiled them out through ~109k `write3`
calls, in a file whose own header says *"Heavy-lift operations … are `@[extern]` to C for speed —
avoids millions of Lean-level push calls."* Its docstring priced it at *"~2 ULP of a ~310 ms step"*
from a **40,960**-draw shape; the job runs global batch 256, so it draws **8× that**.

⭐ **B0 is the only net that pays it.** `dropoutKeep` is set on the two EfficientNet specs
(`VerifiedNets.lean:930,990`) and nowhere else, and no other job variant carries `do` in its name.
That is the whole reason B0 read as an outlier while ViT, ConvNeXt and MNv2 sat near parity.
⭐ **JAX never paid it**: it draws the same mask with `jax.random.bernoulli` *inside* the jitted
`train_step`, on device. The host-side draw here is deliberate and stays — `stablehlo.rng` would
make every bit-exactness gate contingent on seeding an XLA RNG identically across two lowerers —
so the fix was to make the host draw fast, not to move it.

**The control, before touching anything** (f32, synth, 4 replicas):

| variant | bare graph | synth step | host term |
|---|---|---|---|
| `emarmsdp64dropdo` (drop + do) | 159.94 | 346 | **186.1 ms** |
| `emarmsdp64` (neither) | 161.59 | 176 | **14.4 ms** |

The two graphs cost the same to within noise (the drop/do graph is 1.65 ms *cheaper*), so all
171.7 ms of the step difference is host-side — and 281 − 171.7 = 109 predicted the post-fix step
to within 2 ms.

### ✅ LANDED — `dropoutFill`, the loop in C

`lean_f32_dropout_fill` in `ffi/f32_helpers.c`, called from a `dropoutMask` that keeps every guard
it had (`keep ≥ 1`, `n == 0`, `n < 3` still throws). **150.07 → 1.21 ms/call.**

**Byte-identical**, checked against the pre-change Lean loop transcribed verbatim: six shapes
(including the job's 256×1280 and the `n % 3 = 1` case), both guard paths, survivor fractions
tracking `keep` (261,759/327,680 = 0.799) so the agreement is not a degenerate all-ones one.
`dropout-tie` green, its per-example control C1 red as designed.

| B0, 4 replicas, 40-step probes | before | **after** | |
|---|---|---|---|
| **bf16 real** | 281 | **107 ms/step** | **2.63×** |
| bf16 synth | 245 | 98 | |
| **f32 real** (what the conf names today) | 357 | **188 ms/step** | **1.90×** |
| host term (synth − graph) | 173.4 | **26.4** | |
| shim read (real − synth) | 36 | **9** | |

⭐ The producer did **not** become the binding constraint at the shorter step — the unhidden read
*fell*, 36 → 9 ms. Over 5,004 × 350 steps: bf16 **136.7 → 52.1 h**, f32 173.7 → 91.5 h.

| net | JAX min/ep | verified min/ep | ratio |
|---|---|---|---|
| **EfficientNet-B0** | 8.88 | ~~23.44~~ **8.92** | ~~2.63×~~ **1.00×** |

⚠⚠ **THE LESSON, and it is not "the estimate was stale".** The estimate was taken **at a shape no
production job runs**. A host-side per-step cost is a function of the BATCH, and this one was
costed at B=32 while every ImageNet job runs global 256. ▶ When you write down what a host-side
helper costs, write down the batch you costed it at — and re-cost it when a job's batch changes.
⚠ A second one: `lake build LeanMlir.F32Array` regenerates the **`.c`** but not the `.c.o.export`,
which is only built under an executable target. The first byte-identity run linked an object from
the day before and compared the old loop against itself — a green that proved nothing, and the
timing line (unchanged at 150 ms) was the only thing that gave it away. [[stale-lean-exe-gates]]
again, one layer down.

### 6.1b ▶ The same measurement on every other net — and two corrections

Job-accurate bf16 variants, 4× 4060 Ti, `DEVS=0,2,3,4`, `PJRT_FFI_RESIDENT=1`, `SHIM_WORKERS=8`,
40-step probes, **all taken in one session** (§6.3's method), with the bare graph at 4 replicas:

| net | bare graph | synth | real | **host** | **read** | min/ep | JAX | ratio |
|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B0** | 71.60 | 98 | 107 | 26.4 | 9 | **8.92** | 8.88 | **1.00×** |
| **ViT-Ti** | 97.45 | 144 | 158 | **46.5** | 14 | 6.59 | 6.63 | **0.99×** |
| MobileNetV2 | 69.62 | 86 | 140 | 16.4 | **54** | 11.68 | 7.09 | 1.65× |
| ConvNeXt-T | 106.66 | 119 | 142 | 12.3 | 23 | 23.69 | 14.93 | 1.59× |
| MNv4-Conv-M | 61.08 | 83 | 122 | 21.9 | **39** | 10.17 | 9.03 ⚠ | 1.13× ⚠ |

`host` = synth − graph (blob patching, host draws, the Lean loop); `read` = real − synth (the shim
feed the depth-8 prefetch did not hide). **Two of five nets are now at or past their JAX reference.**

⚠ **The MNv4 row's JAX cell is not like-for-like** — that reference runs effective batch **4096**
with EMA where the verified job runs 256 without, so its 1.13× is two recipes compared, not two
lowerers. Full account in §6.2. ⚠ MobileNetV2's `real` is a median of three (140/141/142); an
earlier single sample read 132, and its fp32 arm spans 159–196 across three runs, so treat any
single mnv2 probe as ±15%.

⛔⛔ **CORRECTION — §6.1's old "bare graph (4-rep)" column mixed replica counts.** ConvNeXt-T's
68.0 ms was §1.5's **single-device** number (that table says "one 4060 Ti"); at 4 replicas, with the
all-reduce, its graph is **106.66 ms**. So the "48% of ConvNeXt's step is not the graph" reading
was wrong in the direction that matters — it is **25%**, and ConvNeXt is the net where the graph
really is the cost. Corrected shares: B0 33%, ViT 38%, MNv2 47%, ConvNeXt-T **25%**.
⚠ The ViT and MNv2 rows were fine (97.4 and 68.4 reproduce here at 97.45 and 69.62); only the
ConvNeXt row imported a 1-replica number into a 4-replica column.

⚠ ConvNeXt-T's runner reads **142 ms** here against §1.2c's 131 — 8%, which is exactly the
cross-day drift §6.3 warns about, and is why this table was taken in one sitting. Prefer these.

▶ **Where the remaining two gaps live, and they are different problems:**
* **MobileNetV2 (1.55×) and MNv4 are PRODUCER-bound** — 46 and 39 ms of unhidden read against a
  16 and 22 ms host term. Their graphs are the smallest on the box, so the feed is what is left.
* **ConvNeXt-T (1.59×) is GRAPH-bound** — 75% of its step is the graph, and its host and read terms
  are the smallest of the five. Nothing in the trainer will fix that one.
* ⭐ **ViT-Ti now carries the largest host term on the box (46.5 ms of a 158 ms step)** and is
  still 0.99× its reference. It is the only remaining net where the B0 lever — a host-side cost
  bigger than it looks — is even plausible. Worth one `bf16_device_step` + synth pass before
  assuming it is blob patching.

---

### 6.1c ✅ CLOSED — `SHIM_WORKERS=8` was over-provisioned on six of seven nets (2026-08-31)

⭐⭐ **Every job conf shipped `SHIM_WORKERS=8`. On six nets 4 is measurably better, worth ~89 h
across the fleet.** All bf16, 4× 4060 Ti `0,2,3,4`, `PJRT_FFI_RESIDENT=1`, 120-step probes,
every row replicated with non-overlapping bands.

| net | w8 | **w4** | floor | job | w8 h | **w4 h** | saved |
|---|---|---|---|---|---|---|---|
| ConvNeXt-B | 324 | **300** | 294 | 300 ep | 270.2 | **250.2** | **20.0 h** |
| ConvNeXt-S | 212.5 | **191** | 190 | 300 ep | 177.2 | **159.3** | **17.9 h** |
| ConvNeXt-T | 145.5 | **126** | 118 | 300 ep | 121.4 | **105.1** | **16.3 h** |
| MobileNetV2 | 131 | 114 / **109** @w3 | 86 | 350 ep | 63.7 | **~55** | **~9 h** |
| MNv4-Conv-M | 126.5 | **85** | 82 | 100 ep | 17.6 | **11.8** | **5.8 h** |
| R50 / RSB-A3 | 166.5 | **147** | 141 | 100 ep | 23.6 | **20.9** | **2.7 h** |
| ViT-Ti | 154.5 | **143** | — | 300 ep | 32.2 | **29.8** | **2.4 h** |

▶ **The mechanism is CONTENTION, not capacity**, and two hypotheses were killed getting there:

* ⛔ **Not the payload.** The shim streams f32 CHW — 154 MB/step for MNv2 — so serialization looked
  like the obvious cost. `SHIM_HASH` (computes batches, streams nothing) runs *slower* than
  streaming, 190 vs 172 ms/batch. The pipe is free. Sending uint8 would buy nothing.
* ⛔ **Not producer capacity.** Standalone, MNv2's producers saturate at ~15 batch/s between 4 and 8
  workers (1→4 buys 2.3×, 4→8 buys 8%) — faster than the 86 ms compute floor. And
  `SHIM_DETERMINISM=0`, which this repo records as making producers 5.3× faster, moved the step
  **0%** (137 vs 131, 115 vs 114) — the same signature §6.1b's ViT precedent left.
* ✅ **It is the trainer's own host thread losing cores.** Producers past ~4 add no throughput and
  take CPU from blob patching / host draws / the Lean loop. ⭐ The saving is near-CONSTANT in
  absolute terms across the ConvNeXt family — 19.5, 21.5, 24 ms/step as the model triples in cost —
  which is what a fixed contention cost looks like and not what a compute-proportional one does.

⛔⛔ **NOT a fleet-wide constant. Three nets keep 8**, and the reasoning differs:

* **EfficientNet-B0 — INCONCLUSIVE, and this is the cautionary row.** A first pass read
  w6 **103** / w4 145 / w8 169 and looked like the biggest win on the board. Replication reversed
  it: w6 = 103/180/155, w8 = 169/111. **±30% spread, bands fully overlapping.** B0's probe cannot
  resolve a worker effect at all, so it stays at 8. ⚠ Its conf's old "w4 346 vs w8 203" is
  pre-`dropoutMask`-fix and should not be quoted either.
* **R34, R50-2018, ViT-S/B (g512)** — simply unmeasured. Do not change on the strength of this table;
  ViT-S/B run global batch 512 at a different shape.

⚠⚠ **`vit-default-4gpu.conf`'s 665 ms/step was STALE BY 4.3×.** Measured 154.5 today at the same
4×128 shape, which vindicates §6.1b's 158 and retires the conf's ~138.7 h projection: the real
300-epoch ViT-Ti job is **~32 h**. ▶ Any conf timing not re-measured this session is suspect the
same way; `cnxs`/`cnxb`'s bf16 figures turn out to be *bare-graph* numbers, so those jobs were
budgeted at their floor and ran 8–11% over it at w8.

⚠⚠ **THREE ConvNeXt ImageNet jobs would not have STARTED**, and nothing in the precheck said so.
`convnext{,-s,-b}-imagenet-verified` were built 08-29 23:22 against renders regenerated 08-30 20:08
(the head-LN fix added 2 param tensors ⇒ 6 outputs). Launching any of them died on
`G4 VIOLATION: … returns 567 outputs, caller supplied 561`. All three rebuilt.
▶ The confs' prechecks verify the RENDER exists but never that the BINARY matches it — a gate
worth adding, and [[stale-lean-exe-gates]] for the fourth time.

**Landed:** `SHIM_WORKERS=4` in `cnx`, `cnxs`, `cnxb`, `mnv2`, `mnv4`, `vit`, `vit-emabf16`,
`r50-a3`, `r50-a3-wxclip`, each with its measurement in a comment, and the five prechecks that
*required* 8 now require 4. MNv4's carries an explicit floor: **w3 measured 276 ms/step** — its
heavy shim starves below 4, so 4 is a floor and not a midpoint.

### 6.2 ✅ CLOSED — MNv4 re-costed, and the shape fix never reached the runner

Job-accurate, 4× 4060 Ti, `SHIM_WORKERS=8`, 40-step probes, medians of three, one session:

| MNv4-Conv-M | bare graph @4rep | synth | **real** | host | feed |
|---|---|---|---|---|---|
| **f32 `adamdp64`** — what the conf names | 122.03 | 143 | **176** (176/180/176) | 21 | 33 |
| **bf16 `adamdp64bf16`** | 61.08 | 83 | **122** (125/122/120) | 22 | 39 |

⛔ **The headline is a NULL, and it is the answer §6.2 was opened to get.** MNv4 gained **1.11×**
on the bare graph from the shape fix (0.776 → 0.056 GB), and the conf's pre-fix numbers were
177 / 126. Post-fix they are **176 / 122** — statistically identical. The graph win did not reach
the runner because **exactly half the step is not the graph**: 61.1 ms of the 122 ms bf16 step is
the graph (all-reduce included, these are 4-replica compiles), 39 ms is unhidden feed, 22 ms host.
▶ bf16 is worth **2.00×** on the graph and **1.44×** on the whole step. Nothing here needs redoing;
the numbers the conf and the book carried were right.

⚠ Its artifact prefix is `mnv4in_`, **not** `mobilenetv4in_` — the exe is
`mobilenetv4-imagenet-verified` but the render slug is `mnv4in`, and a probe naming the exe's
prefix silently finds no file rather than erroring.

### ⛔⛔ The denominator exists — and it is not a like-for-like one

`/home/skoonce/mnv4_convm_100ep` (outside the repo) records **541.8 s train per epoch** at
steady state, i.e. **9.03 min/ep**, against the verified path's 10.17. That is **1.13×** — but do
not quote it as a lowerer result, because the two sides are not running the same job:

| | JAX reference | verified `adamdp64` |
|---|---|---|
| effective batch | **4096** (512 micro × `GRAD_ACCUM = 8`) | **256** (4 × 64) |
| optimizer | AdamW @ **0.004** | AdamW @ **1e-3** |
| EMA | ✅ `ema_params` **and** `ema_bn`, every step | ⛔ none (`emaOn` needs an `ema` prefix) |
| RandAugment m9, drop-path | ✅ | ✅ (same shim) |

⚠ **The batch difference runs in our favour and the EMA runs against it.** JAX does 312 optimizer
steps/epoch against 5004, and each of its forward/backward passes is 2× wider per device, so it
pays the per-invoke overhead half as often at better occupancy; against that it does two full
`tree_map`s over 9.7M parameters per step that the verified path does not. Neither effect has been
separated, so 1.13× is a **throughput comparison of two different recipes**, not a statement about
the lowerer. ▶ The conf already says the optimizer does not match; this quantifies what that costs
the comparison. A clean number needs the JAX side re-run at global 256 without EMA, or the verified
side brought up to the reference recipe — the latter is the real goal and is `chapter_makeover.md`'s
open MNv4 gap list.

⚠ Its phase-2 accuracy reference is the 100-epoch JAX run at **75.51%**, in that same out-of-repo
directory. `EPOCHS=100` matches `mnv4ImagenetConfig.epochs` (audited 2026-08-30).
⭐ `convBnAct` is `.relu` on **both** MNv4 specs, which agree — so MNv4 escaped the stem/head
defect that hit B0 and MNv2 (§2.1). Verified render: 54 `stablehlo.maximum`, 0 `minimum`.

### 6.3 ▶ The JAX end-to-end reference run

⛔⛔ **Only the numerator of every ratio in §6.1 has been audited.** Three times on 2026-08-29/30 the
*verified* side turned out to be costing a graph its job does not run — B0's book row was a variant
without EMA/drop-path/dropout (181 ms against the real 280), mnv2's was AdamW where the job is
RMSProp, ConvNeXt's was pre-fix. **Nobody has put the JAX rows through the same check.** If a JAX
reference row costs a lighter recipe than its own job runs, these ratios are wrong in the flattering
direction and for the wrong reason.

▶ Method that paid off: for each net, confirm the JAX config bakes the same recipe as the verified
artifact **before** quoting the ratio, and take both sides in ONE session on this box rather than
comparing across days. Cross-day drift on this hardware measured 1–9% today, which is the same size
as several of the effects being chased.

### 6.4 The rest of the queue

* **Move the jobs to bf16.** Every `scripts/jobs/*.conf` still names its f32 artifact by default, and
  every bf16 twin now exists (mnv2's `rmsdp64bf16` was the last gap, rendered 2026-08-30). Worth
  ~39 h on enet (91.5 → 52.1, re-measured post-`dropoutMask` — §6.1), ~25 h on mnv2,
  63/128/184 h on ConvNeXt-T/S/B. The confs carry the numbers.
* **Track 3's cheap wins** — run the ViT EMA/bf16 job (`vit-default-emabf16-4gpu`, artifacts ready,
  and its EMA arm measures 151 ms, one *under* the plain bf16 arm, so the shadow is free); fix mnv2's
  baked label smoothing (a literal at `MobileNetV2RenderB.lean:652`, and it is on the **gradient**
  path); compose ConvNeXt's EMA.
* **Track 2's real items — now written up as §7**, with the sites, the blast radius and the
  ordering. ⭐ Two of the three are in BOTH phases, not phase-2 only like `convBnAct` was.
* **Rebuild the `efficientnet_adam` Imagenette checkpoint** — deleted chasing a phantom bug (§1.2d);
  at epoch 4 instead of 80. ~40 min.
* **Optional:** settle why mnv2 escapes the relayout (§1.4) — still unexplained, and it is the one
  net whose emit was clean before either fix.

### 6.5 Closed 2026-08-29/30 — the record

0. ✅ **Track 1.** No idle; the cost was the flat-activation NHWC↔NCHW relayouts, and the renderer
   emits pointwise ops 4-D (§1.2b). B0 bare graph **2.28×**, f32 unaffected, CIFAR-8 bf16 training
   unregressed at n=3 (§1.2d).
1. ✅ **The residue was NOT the channel-LN (§1.5)** — a width COLLISION in §1.2b's own shape table
   (`1536·14·14 = 96·56·56`). Keyed by SSA name: ConvNeXt-T 2.434 → **0.122 GB** and **1.24×**,
   -S **1.46×**, -B **1.51×**, MNv4 **1.11×**, four nets bit-identical. `lnChanP` never written.
2. ✅ **`convnext-fwd-b-tie` was RED since `f4e4172`** and nobody knew — that commit lifted only the
   batched pointwise arms, so the two ConvNeXt renderers diverged at the GELU. Fixed by lifting the
   14 per-example arms too, which also extends the 4-D emit to the Imagenette/CIFAR corpus.
3. ✅ **Three artifacts were landing outside the CI drift guard** (two since `f4e4172`).
   ⚠ `scripts/check_render_coverage.py` is the ONLY thing that catches this and no local build runs
   it — `regen_verified_mlir.sh check` passes, because it audits writers and entry names, not CI.
4. ✅ **`regen_verified_mlir.sh proofs` was skipping four writers and exiting 0** — the `*RenderB`
   modules were missing from its hardcoded list, leaving 65 of 113 artifacts stale.
5. ✅ **Two job confs would have truncated their own LR schedules.** `EPOCHS` is supervise.sh's STOP
   condition, and enet had 80 / mnv2 had 300 against constants of 350 — the run would have been
   killed a quarter (resp. a seventh) of the way through its own curve and reported ✅ COMPLETE.
   ▶ **Every other conf audited 2026-08-30 and clean**: cnx/cnxs/cnxb/mnv4/vit/vits/vitb all match
   their constants; r34 and the four r50 confs use `${EPOCHS:-N}` and cite no constant.
6. ✅ **`mobilenetv2in_rmsdp64bf16` rendered** — that net's only bf16 artifacts were the AdamW family,
   so the job that trains it had no bf16 arm at all. 187 → **136 ms/step**, ~95 → **~70 h**.
7. ✅ **`scripts/bf16_device_step.py` reworked** — interleaved reps, shared input buffers, min and
   p10–p90 printed, a flag inside ±5%. It had reported **0.92× on MobileNetV2** — an 8% regression —
   for a bit-identical graph, because six compiles were running at once and a bare median hid it.
8. ✅ **Five book sections re-costed** (§5.9, §6.5, §7.5, §8.5, §8.6, §9.6, §9.7) and cut to configs
   and numbers. All throughput tables are fp32 → bf16 pairs; the 3060 rows are blanked.
---

## 7. ▶▶ NEXT SESSION (clean start) — the three open paper diffs

Everything in §1–§6 is closed or measured. **This section is self-contained**: it is the whole
brief for a session that has read nothing else. Three defects, all found by audit rather than by a
failing gate. **ALL THREE ARE CLOSED (2026-08-30, §7.1 / §7.2 / §7.3).** What is open is
downstream of them: nothing is measurable until the affected nets re-run — see §7.6.

⛔⛔ **THE OLD HEADLINE — "TWO OF THE THREE ARE IN BOTH PHASES" — WAS WRONG ON BOTH COUNTS, AND
IT WAS WRONG IN OPPOSITE DIRECTIONS.** It was written from an audit of the phase-2 emitter, and
which phase a defect lives in turned out not to be predictable from that. Settled 2026-08-30:

| | where it actually lived | how it was found |
|---|---|---|
| **B3** — BN momentum | **BOTH** — the item called it phase-2 | grep for the literal on the verified side |
| **B2** — mixup drops ls | **phase 2 ONLY** — the item called it both | reading the artifact's cotangent chain |
| **B1** — ConvNeXt head LN | **BOTH** — and the port was made wrong ON PURPOSE, to match the reference (§2m) | the timm parameter count |

⭐ **And B1 is the one that should change how the next audit starts.** It was found by an
architecture read, but the thing that would have found it in seconds was
`sum(p.numel() for p in timm.create_model('convnext_tiny').parameters())` — 28,589,128 against our
28,587,592. ▶ **Check the parameter count against timm before checking anything else**, and treat
"the counts nearly cancel" as a reason to look harder rather than a reason to stop.

⭐⭐ **RUN ACROSS THE WHOLE FLEET, 2026-08-30 — every ImageNet net now matches timm EXACTLY**,
`VerifiedNetSpec.toSpecs` against `timm.create_model(name, num_classes=1000)`:

| net | timm model | params | |
|---|---|---|---|
| R34 / R50 | `resnet34` / `resnet50` | 21,797,672 / 25,557,032 | ✅ |
| MobileNetV2 / MNv4-Conv-M | `mobilenetv2_100` / `mobilenetv4_conv_medium` | 3,504,872 / 9,715,512 | ✅ |
| EfficientNet-B0 | `efficientnet_b0` | 5,288,548 | ✅ |
| ViT-Ti | `deit_tiny_patch16_224` | 5,717,416 | ✅ |
| **ConvNeXt-T / -S / -B** | `convnext_{tiny,small,base}` | 28,589,128 / 50,223,688 / 88,591,464 | ✅ **after B1** |

⚠ ConvNeXt was the ONLY family that did not, and all three sizes were wrong the same way. ▶ This
is now the cheapest standing check on the fleet — one `create_model` call per net, no GPU, no data.
⛔ It is a check on the **shape**, not on the recipe: §2.3's point is that B0 matches timm's
architecture while deliberately not matching timm's hyperparameters.

⭐ **The transferable rule, and B2 is the expensive case.** The old §7.2 named the shim's
`_targets` as one of three writers to change; the verified graph smooths downstream, so doing that
would have DOUBLE-SMOOTHED every ConvNeXt and ViT run — a plausible-looking edit, made on a
plausible-looking instruction, that trains a wrong objective and reports a number. ▶ **Read the
artifact before believing a claim about the verified side.** The evidence is 8 lines of MLIR and it
took minutes; the claim it overturned had been written down twice.

### 7.1 ✅ CLOSED 2026-08-30 — B1, ConvNeXt's missing head LayerNorm

Paper is `GAP → LN → Linear`; we did `GAP → Linear`, in **both** phases. Now fixed in both.

⭐⭐ **THE PARAMETER COUNT WAS THE TELL, AND IT HAD BEEN WRITTEN DOWN AS A REASON NOT TO LOOK.**
`ConvNeXtLayout.specs`' own docstring said, of the §2m stem-LN/head-LN swap: *"the last two nearly
cancel — `+2·768 − 2·96 = +1,344` of 28.6M — so a matching parameter count is a decomposition
test, not an architecture check."* Right that they nearly cancel; wrong that the residue was
noise. **The residue IS the missing layer, exactly:**

| | ours (before) | timm 1.0.28 | short by |
|---|---|---|---|
| ConvNeXt-T | 28,587,592 | **28,589,128** | 1,536 = 2×768 |
| ConvNeXt-S | 50,222,152 | **50,223,688** | 1,536 = 2×768 |
| ConvNeXt-B | 88,589,416 | **88,591,464** | 2,048 = 2×1024 |

All three now match `timm.create_model(...)` **exactly**. ▶ That is the cheapest possible check on
an architecture claim and it was available the whole time.

⛔⛔ **THE HISTORY IS THE POINT: §2m/§2n DELETED THIS LAYER ON PURPOSE.** The pre-§2m render had a
head LN and no stem LN. §2m added the stem LN — correct — and removed the head LN to agree with
`jax/MainConvNeXtImagenet.lean`, which was itself missing it. Both references have **both**:
`facebookresearch/ConvNeXt` does `self.norm(x.mean([-2,-1]))` with `nn.LayerNorm(dims[-1],
eps=1e-6)`, and timm's head is `NormMlpClassifierHead(global_pool → LayerNorm2d(768) → flatten →
fc)`. ▶ This is §7.2's lesson with the sign flipped: there the port was right and the reference
wrong; here **the port was made wrong in order to match the reference**. Converging the two phases
is not the same as being correct, and nothing in this repo can tell the difference — which is why
the timm parameter count matters more than any internal tie.

⛔ **The renderer is SHARED between the Imagenette and ImageNet ConvNeXts, so this necessarily
pulled the proof chain** — §7.1 as written did not say that. `denoteConvnextT` pattern-matches the
exact layer list and ties it to `convNextForwardTCh` by `rfl`, so the layer could not be added on
one side only.

**What it took, and it was smaller than the blast-radius note suggested:**

| piece | change |
|---|---|
| `ConvNeXtFullT.lean` | `CnxTWeightsCh` gains `hε/hγ/hβ`; one composition step; **one** `vjp_comp` link + one `0 < hε` |
| rung E | `headLNGraph` + `headLNGraph_faithful` (3 ops, one `rw`); one more rewrite in the apex |
| `ConvNeXtRender.lean` | `headLn{Fwd,Back}Site` + γ/β tails; `allParams`; fwd + bwd wiring |
| `ConvNeXtRenderB.lean` | the same four, batched |
| `Types.lean` / `Spec.lean` / `SpecHelpers.lean` | a bare `.layerNorm dim` layer + shapes + He-init |
| `Jax/Codegen.lean` | `head_layer_norm` helper + arms in forward / init / from-file / to-file |
| 3 JAX specs, 4 verified specs, `ConvNeXtLayout` | one line each |

⭐ **No new mathematics.** The head LN is `rowLNVecFlat 1 768` — ViT's per-token LN at one row —
whose `_diff`, `_has_vjp` and graph bridge (`rowLN_affine_eq`) were all already proven. And in the
render it is `lnFwdSite` with the two transposes deleted, since at one spatial position "normalise
each row over its channels" and "normalise the feature vector" are the same function.

**Evidence, all taken 2026-08-30:**
* ⭐ **Two independent routes agree on the weight-decay split** — the renderer's `allParams` gives
  **59 decayed / 123 excluded**, and running the regenerated reference's OWN `_wd_mask` over its
  OWN `init_params` gives 59 / 123. (Was 59 / 121; the head γ/β are 1-D, so they must land in the
  EXCLUDED column — a head LN that showed up in the decayed one would be getting weight decay the
  recipe excludes.)
* `convnext-fwd-b-tie` ✅ — forward **byte-identical**, backward `gradMap IDENTICAL — 182
  parameters, same names, same SSA, same order`.
* `convnext-adam-tie` ✅ — 0 of 182 params disagree, gradient norm-rel 0.000000.
* `convnext-dp-check` ✅ (2 replicas) and `convnext-shard-check` ✅ (TEST 8.1e-8 vs CONTROL 0.075,
  9.1e5× apart).
* `regen_verified_mlir.sh proofs` + `check` ✅ (36 artifacts rewritten, prefix/silent-emit/empty-slot/
  prelude audits clean); `check_render_coverage.py` ✅ 227 files.
* The regenerated phase-2 trainer runs: `init_params` → **182 tensors / 28,589,128**, head LN at
  γ=1 β=0, `forward` returns `(2, 1000)`.
* **Cross-phase numeric check** (there is no ConvNeXt tie script, so this was done by hand): the
  emitted `head_layer_norm` against a numpy transcription of the artifact's ops agrees to
  **4.8e-07**, against 8.6 for the no-LN control.

⚠⚠ **THE ε CONTROL IS WEAK, AND THAT IS WORTH KNOWING.** eps 1e-5 instead of 1e-6 moves the same
comparison by only **2.4e-06** — at realistic activation scales the variance dwarfs both, so a
wrong ε here is numerically invisible. It was matched by READING (`ConvNeXtRender.cEPS`,
`channel_layer_norm`'s 1e-6, the paper's `eps=1e-6`), not by testing, and the JAX emitter now has
a `head_layer_norm` separate from the transformer nets' `layer_norm` precisely so the two epsilons
cannot be confused. ▶ Do not treat the cross-phase number above as evidence about ε.

⛔ **Every ConvNeXt checkpoint is now incompatible** (182 tensors, not 180). `trainAdamSched`'s
checkpoint size guard refuses loudly rather than misaligning, which is the good failure — but the
Imagenette `convnext-verified-adam` epoch-80 state and any ConvNeXt ImageNet checkpoint must be
retrained, not resumed.

⚠ **ConvNeXt has NO phase-2 ↔ phase-4 forward tie script** — `scripts/` has `enet_forward_tie.py`,
`mnv2_forward_tie.py` and `mnv4_forward_tie.py` and no ConvNeXt peer. That is a real part of why
this survived §2m: nothing compares the two ConvNeXt phases numerically, so a layer could leave one
side and the other would follow it by hand. ▶ Worth writing; it is the gate that would have caught
B1 and would catch the next one.

▶ **Worth: part of ConvNeXt-T's −1.0 (82.1 paper vs 81.10), unquantified until a re-run.** B1 and
B2 both hit this net, both are now in, and §7.4's advice stands — run ConvNeXt-T once with both
rather than twice to split a gap that may not decompose.

### 7.2 ✅ CLOSED 2026-08-30 — B2, mixup and cutmix threw away label smoothing (phase 2 ONLY)

⛔⛔ **THE HEADLINE OF THE OLD §7.2 IS WRONG, AND ACTING ON ITS INSTRUCTIONS WOULD HAVE INTRODUCED
A BUG.** It said the defect was in both phases and named three writers to change — `_mixup`,
`_cutmix`, **and the shim's `_targets`**. The shim is *right as it stands*: the verified graph
smooths downstream, so smoothing the shim's target too would **double-smooth** every ConvNeXt and
ViT run on the phase-4 path. Only the two phase-2 writers were touched.

**Why phase 4 was already correct.** The verified renderers fold the smoothing into the COTANGENT
rather than into the target, and they apply it to whatever `%onehot` carries — one-hot or mixed.
`convnextin_adamdpwxclipdrop_train_step.mlir:1728–1735` (and `vitin_adamdp128x4wxclipdrop`:2537):

```mlir
%v1537 = subtract %v1536, %onehot            // softmax − t
%v1538 = constant dense<0.100000>            // α
%v1539 = multiply %onehot, %v1538            // α·t
%v1540 = add %v1537, %v1539
%v1541 = constant dense<-0.000100>           // −α/K, K = 1000
%v1542 = add %v1540, %v1541
%v1544 = divide %v1542, dense<32.0>          // ÷B
```

That is `(p − ((1−α)·t + α/K))/B` — the gradient of CE against the SMOOTHED target — and it holds
for any `t` with `Σt = 1`, which a mixup or cutmix target is. ⭐ **Checked numerically, not just
algebraically**: feeding the shim's raw mixed target through the transcribed chain reproduces
`timm.data.mixup.mixup_target(y, K, lam, smoothing=0.1)`'s CE gradient to **6.1e-09**, where an
unsmoothed target is off by 0.013.

▶ So B2 is the same SHAPE as the `convBnAct` fix (§2.1): a **port divergence** where the verified
render was already the paper-faithful one and phase 2 was the side that was wrong.

**The fix**, one Lean-level binding (`softOneHot` in `emitLossAndTraining`) consumed by both
`_mixup` and `_cutmix`, so the two cannot disagree:

```python
y1 = (jax.nn.one_hot(y, 1000) * (1.0 - 0.100000) + 0.100000 / 1000)
```

deliberately spelled the same way as `loss_fn`'s hard-label branch. Smoothing and mixing COMMUTE —
both are convex combinations of rows summing to 1 — so smoothing the one-hot before mixing equals
smoothing the mixed target; this is simply where timm puts it. Verified against timm 1.0.28's
`mixup_target` at **2.5e-08** (float32 vs float64), and the emitted `_mixup`/`_cutmix` were then
executed under the pinned jax: rows sum to 1, off-value exactly `s/K` = 1e-4.

**Blast radius, from the regenerated files** — the two one-hot lines and nothing else:

| net | recipes changed | |
|---|---|---|
| **ViT-Ti / -S / -B** | every one | ls 0.1 + mixup + cutmix |
| **ConvNeXt-T / -S / -B** | every one | ls 0.1 + mixup + cutmix |
| R50 (all nine, incl. A1/A2/2018) | **none — byte-identical** | `labelSmoothing := 0.0` by recipe, so the emit is the bare one-hot character for character |
| R34 / MNv2 / B0 / MNv4 | none | no mixing |

⛔ **The old §7.2's "RSB-A2/A1 trained at ls = 0.0" was also wrong, and harmlessly so** —
`resnet50ImagenetConfig` sets `labelSmoothing := 0.0` deliberately (*"BCE over mixup/cutmix soft
labels subsumes it"*), and every RSB tier inherits it. There was nothing to drop.

⚠ **SIX phase-2 accuracies are now stale**, not one: ConvNeXt-T's **81.10 / 95.37** and ViT-Ti's
**72.31 / 91.12** describe nets the code no longer emits, and the four larger sizes never ran.
⚠ ViT-Ti was at **+0.1 over paper** *with* the defect, so do not assume the sign.
⚠⚠ **And the phase-4 column never had it**, so ConvNeXt's and ViT's phase-2 ↔ phase-4 comparisons
were not like-for-like either — the same trap `convBnAct` set for B0 (§2.1).

### 7.3 ✅ CLOSED 2026-08-30 — B3, BN momentum was a hard-coded 0.99

**It was hard-coded in BOTH phases, not just the emitter**, which the item did not say:
`jax/Jax/Codegen.lean:614` emitted the literal `"0.99"` into `_bn`, and `VerifiedTrain.lean`'s
host-side `F32.ema` weight was a matching literal `0.01` in two drivers (`trainAdamSched` and the
fp8 twin). They agreed with each other, so nothing could ever have caught it.

⭐ **Cheap for the reason the item guessed, but a different reason than it gave.** It needs no
re-render on either side — the graph emits raw per-layer BATCH stats and the HOST does the EMA, so
the decay never enters the MLIR at all. Nothing in `verified_mlir/` moved.

**The knob.** `TrainConfig.bnMomentum` (shared with the JAX emitter) and its peer
`VerifiedConfig.bnMomentum`, both defaulting to **0.99**, so every net that does not set one is
unchanged. On the verified side the two branches live in ONE place, `VerifiedConfig.bnEmaWeight`,
consumed by the training loop, the fp8 trainer, and a new startup banner line — the file's own
`VerifiedVariant` docstring is the record of what a second copy costs.

⚠ **The default arm returns the historic `0.01` DOUBLE, not `1.0 - 0.99`** (= 0.010000000000000009,
different bits). Below every gate's bar, but it is what makes "the other four nets are unchanged" a
statement about bytes. Verified by `#eval` on the real configs: R34 / MNv2 / B0 / MNv4 all report
`bnEmaWeight none == 0.01` **true**.

⚠ **The sense is TF's, and it is the RECIPROCAL of PyTorch's.** `bnMomentum` weights the OLD
estimate; `torch.nn.BatchNorm2d(momentum=m)` weights the NEW batch. timm's default `momentum = 0.1`
is `bnMomentum = 0.9`. (The old §7.3 said "a 100-step averaging window against our 1000" — the
windows are **10 against 100**, `1/(1−d)`.)

⭐⭐ **AND THE AUDIT SAYS THE ITEM'S PER-NET REASONING WAS RIGHT BUT ITS TIMM CLAIM IS NOT.**
Read off the pinned timm 1.0.28 in `.venv-timm`, `create_model(name).modules()`:

| timm model | BN momentum | ⇒ decay | eps |
|---|---|---|---|
| `resnet50`, `resnet34` | 0.1 | 0.9 | 1e-5 |
| `mobilenetv2_100` | 0.1 | 0.9 | 1e-5 |
| `efficientnet_b0` | 0.1 | 0.9 | 1e-5 |
| **`tf_efficientnet_b0`** | **0.1** | **0.9** | **1e-3** |
| `mobilenetv4_conv_medium` | 0.1 | 0.9 | 1e-5 |

⛔ **timm runs 0.9 on EVERY one of them, `tf_efficientnet_b0` included.** `BN_MOMENTUM_TF_DEFAULT
= 1 - 0.99` exists in `models/_efficientnet_builder.py:36`, but its only consumer `get_bn_args_tf()`
**has no caller in 1.0.28** — the `tf_*` ports `kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)` and
nothing else, so they inherit PyTorch's momentum and change only eps. So "0.99 is correct for
EfficientNet, which is TF's value" is true **of the original TF codebase and false of timm**.

▶ **What was changed: R50 only, both phases.** `jax/MainResnet50Imagenet.lean` and
`apps/imagenette/MainResnet50Imagenet.lean` set `0.9`. All nine JAX R50 recipes inherit it —
verified by parsing `_bn`'s default out of every regenerated trainer: the six unaccumulated
recipes read `momentum=0.9`, the three K=4 ones `0.974004` = 0.9^(1/4), and **no other net's file
moved a byte**. At A3's k=8 the verified driver now compensates to `1 − 0.9^(1/8)` = **0.013084**
against the old 0.001256.

▶ **Left deliberately at 0.99, because each chases a different reference — this is the user's call,
not an oversight:**
* **EfficientNet-B0** — its target is the TF paper's 77.1/93.3, and TF's decay there really is
  0.99. ⚠ timm is NOT the reference for this net and reading hyperparameters off it moves us away
  from the number we are scored against — written up as **§2.3**, which this decision produced.
* **MobileNetV2** — target is the TF-slim paper's 72.0; that reference's 0.997 is still
  `[unverified]` (no offline TF-slim source here). ⚠ Do not move it while the 350-epoch run on the
  other box is in flight.
* **MNv4** — its denominator is our own 100-ep JAX run, not a paper (§6.2).
* ⚠ **R34 is the live question.** Same family, same timm reference, same argument as R50 — but
  flipping it stales the landed **74.16%**. One line if wanted.

⚠ **EVAL-ONLY, on no gradient path.** The completed A3 **77.43%** is not invalidated as history; it
is simply no longer what this config trains. ▶ Direction unknown: a longer window is lower-variance
and higher-bias against statistics that are still moving.

⚠ This supersedes the PREMISE of `a3_paper_fidelity.md` §2.3, which compensated the per-micro EMA so
k updates compose to one 0.99/step update, treating 0.99 as the reference throughout. The
compensation was right and stands — it now composes to one 0.9/step update.

▶ Worth: part of R50-A3's **−0.9**, alongside Ghost-BN. Unquantified until a re-run.

### 7.4 Order, and what to do first

1. ✅ **B3 — DONE 2026-08-30** (§7.3). The R50 re-run it unblocks is now the open half: nothing
   about the change is measurable until R50 trains again on both paths.
2. ✅ **B2 — DONE 2026-08-30** (§7.2). ⭐ Its own warning — *"do check whether the verified
   artifacts bake a smoothing constant anywhere before assuming"* — is what saved it: they do, and
   following the rest of the item would have double-smoothed the verified path.
3. ✅ **B1 — DONE 2026-08-30** (§7.1). ⚠ It pulled the proof chain, which the item did not say:
   the renderer is shared with the Imagenette net and `denoteConvnextT` ties the exact layer list
   by `rfl`. Smaller than feared all the same — no new mathematics, one `vjp_comp` link.

▶ **Then re-run ConvNeXt-T once with B1+B2 together.** They hit the same net and the same −1.0;
running them separately costs two 300-epoch runs (~112 h each at the bf16 arm) to split a gap
that may not decompose.

### 7.5 State of play a clean session should know

* ⚠⚠ **FOUR phase-2 accuracies are stale, not two** — B0's 76.80/93.26 and mnv2's 68.77 predate
  the `convBnAct` fix (§2.1); ViT-Ti's 72.31/91.12 and ConvNeXt-T's 81.10/95.37 predate the mixup
  label-smoothing fix (§7.2). All four describe nets the code no longer emits, all four had the
  port as the MORE faithful side, and in every case the direction is unknown. R50's 77.22 is the
  only phase-2 cell still describing what would train — and even it is now stale on the verified
  side, where R50's BN decay moved to 0.9 (§7.3).
* ⚠ **B0 is scored against a reference timm does not implement** — §2.3. Do not read a
  hyperparameter off timm for that net without deciding which reference the run is claiming.
* ⚠ **An mnv2 350-epoch run is in flight on another box and does NOT carry that fix**, so its
  result will need the same caveat when it lands.
* ⛔ **`scripts/enet_forward_tie.py` cannot catch a spec divergence of this family** — it ties the
  verified render against the *Imagenette* generated file, which is the twin that was already
  correct. A tie that green-lights the wrong pair.
* ⛔ **Every ConvNeXt checkpoint is incompatible** since B1 (182 param tensors, not 180). The
  size guard refuses loudly; the Imagenette `convnext-verified-adam` epoch-80 state needs
  retraining, not resuming.
* ⚠ **ConvNeXt has no phase-2 ↔ phase-4 forward tie script.** enet, mnv2 and mnv4 have one; that
  gap is part of why B1 survived §2m.
* ▶ Throughput is settled and lives in §6.1b; do not re-measure it to start this work.

### 7.6 ▶▶ WHAT IS ACTUALLY OPEN NOW — all of it is runs, none of it is code

Every paper diff on the list is fixed and every gate is green. Nothing about any of it is
*measured*, because all three changes alter what gets trained and none of them can be seen in a
loss curve.

| run | why | rough cost |
|---|---|---|
| **ConvNeXt-T, B1 + B2 together** | the −1.0 both were aimed at; running them apart costs two 300-epoch runs to split a gap that may not decompose | ~109 h at the bf16 arm (§1.2c) |
| **R50 A3, B3** | BN decay 0.9; part of the −0.9, alongside Ghost-BN | its own 100-ep schedule |
| **B0 and mnv2, `convBnAct`** | §2.1's fix; both phase-2 cells are stale until then | 52.1 h / ~70 h at bf16 |
| **ViT-Ti, B2** | its +0.1 over paper was measured WITH the defect, so the sign is unknown | artifacts ready |

⚠ **Six phase-2 accuracies are stale** (§7.5), and in every case the port was the more faithful
side — so no phase-2 ↔ phase-4 ratio in this document is like-for-like until the references are
re-run. ▶ That, not more auditing, is the next thing worth the box's time.
⭐ The one piece of *code* worth doing first is cheap and would have caught B1: a ConvNeXt
phase-2 ↔ phase-4 forward tie, the peer of `scripts/enet_forward_tie.py`.
