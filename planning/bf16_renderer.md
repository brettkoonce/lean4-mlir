# bf16_renderer.md — bf16 on the verified render path, as a ladder from mnist-linear

Scoped 2026-08-01 on ares (6× RTX 4060 Ti, CUDA 12.9).

---

## ⭐⭐ STATUS 2026-08-25 — SEVEN NETS WIRED, ALL SEVEN COSTED AT 4 GPUs (§21). READ THIS BLOCK, THEN §16, §19 AND §20 BEFORE QUOTING ANY ms/step.

Branch `bf16/verified-conv-ops`. Seven nets render in bf16, every one gate-1 and gate-2 green.
**The speedups are NOT interchangeable and the differences are the interesting part** — read the
replica AND residency columns before quoting any number.

| net | artifact variant | R | resident | f32 → bf16 ms/step | speedup | § |
|---|---|---|---|---|---|---|
| ResNet-34 | `momdp64bf16` | 4 | ? | 222 → 157 | 1.41× | §STATUS |
| ResNet-50 | `momdp64bf16` | 4 | on | 360 → 232 | **1.55×** | §10.1 |
| MobileNetV2 | `adamdp64bf16` | 4 | ? | 191 → 139 | 1.37× | §12 |
| MobileNetV2 | `adam64bf16` | **1** | ? | 136 → 71 | **1.92×** | §13.2 |
| MobileNetV4-Conv-M | `adam64bf16` | **1** | ? | 120 → 64 | **1.88×** | §13.1 |
| EfficientNet-B0 | `rms64bf16` | **1** | ? | 163 → 149 | ⛔ **1.09×** | §14 |
| ConvNeXt-T | `adamwxclipdropbf16` | **1** | **on** | 165 → 128 | **1.29×** | §16 |
| ConvNeXt-T | *same graph* | **1** | **off** | 312 → 280 | 1.11× | §16.2 |
| ViT-Tiny | `adamwxclipdropbf16` | **1** | *bare device* | 29.9 → 20.5 | **1.46×** | §20 |
| ViT-Tiny | *all six ops on* | **1** | *bare device* | 29.9 → 57.2 | ⛔⛔ **0.52×** | §19.1 |

⭐⭐ **ALL SEVEN NOW HAVE A 4-GPU NUMBER ON REAL IMAGENET AND AN END-TO-END COST — §21.** Every net
gained a DP bf16 render (three did not have one). Totals: **f32 533.9 h (22.2 d), bf16 425.3 h
(17.7 d)** of continuous 4-GPU compute — bf16 saves **4.5 days, 20 %**. ⛔ ConvNeXt alone is 157 h of
that, because it is the only net at global batch 128 and runs 10,009 steps/epoch for 300 epochs.
⛔⛔ **And raising its batch does NOT fix that — §21.5**: the block interior's per-image cost is FLAT
from batch 8 to 128 (1.03×), so the GPU is already saturated at 32 and doubling the batch just
doubles the step. ⚠ §21.5's *memory* arguments were measured with `nvidia-smi`, which reports XLA's
preallocated pool rather than the graph, and are WITHDRAWN — the real peaks come from
`scripts/bf16_peak_memory.py`. ⭐⭐ **And on ConvNeXt-B they matter: 10.93 GiB f32 (94 % of budget)
against 9.75 GiB bf16, and bf16 saves that net 73 hours — the biggest single saving on this
branch. §21.6.**

⚠ **`?` means the probe did not record it**, not "off" — the column is new with §16 and only R50's
§10.1 and ConvNeXt's §16 state it. ⭐ For MNv2 and B0 the bare-device timings in §16.4 bound how
much it can have mattered (both trainer numbers sit ~12 ms above their device step, on parameter
blobs of 42 MB and 64 MB against ConvNeXt's 327 MB), so neither figure moves much either way. R34
and R50 hold 21.8M and 25.6M parameters and have NOT been checked — they are the ones to re-probe.

⚠⚠ **THE FIVE RESULTS THAT MATTER MOST:**

1. **A 4-replica bf16 number is a SYSTEM result, not a renderer result** (§13.2). MobileNetV2 is
   **1.92× on one GPU and 1.37× on four — same graph**. The loss is the shim feed first and the
   f32 all-reduce second. On one GPU the verified renderer is at PARITY with the JAX reference
   (1.92× vs 1.94×). ▶ Check `SHIM_WORKERS` before ever blaming the emit.
2. **`PJRT_FFI_RESIDENT=1` IS OFF BY DEFAULT AND IT IS A THIRD WAY TO MEASURE THE SYSTEM** (§16.2,
   new 2026-08-24). ConvNeXt-T is **1.29× resident and 1.11× not — same graph, same GPU, same
   batch.** Its 540 parameter tensors are 327 MB, and without residency that crosses PCIe every
   step: 154 ms of a 312 ms step is not the graph at all. ⚠⚠ **`LEAN_MLIR_BENCH_SYNTH` DOES NOT
   CONTROL FOR THIS** — it removes the data FEED, not the parameter round trip, so §14.1's synth
   check on B0 ruled out less than it was read as ruling out. ▶ Quote a number only with its
   residency state, and prefer the bare-device timing (`scripts/`-style direct execute) when what
   you mean is "the renderer".
3. ⛔⛔ **GATE 2 IS NOT A STATEMENT ABOUT SPEED, AND ON ViT THAT COST 32.8 ms** (§19.1, new
   2026-08-25). ViT's patchify-stem WEIGHT GRADIENT in bf16 is **0.19× its f32 peer** — one op, one
   site, +32.8 ms on a 29.89 ms step — because its transpose-trick shape gives cuDNN a **209×209
   window** for which it has a direct f32 kernel and no bf16 one. Every gate was green on that arm:
   gate 1 clean, gate 2 green, histograms identical, losses inside an ulp. ▶ **Profile the
   standalone op against its f32 peer at the net's own shape** — §19.4 adds it to the recipe, §9.1
   predicted this class ("kernel selection, not bandwidth") and §14.3 item 2 asked for the check by
   name. Turned off, ViT is **1.23×**.
4. ⭐⭐⭐ **A bf16 DOT'S RESULT TYPE IS A CORRECTNESS NON-ISSUE AND A ~1.2× PERFORMANCE ISSUE**
   (§20.1, new 2026-08-25). §9.2 measured that `dot_general` reaches the tensor cores with either
   result type and that was read as "inert for dot". Inert for correctness; **not** for speed — an
   f32 result makes the gemm write twice the bytes. Giving ViT's dots bf16-typed results took it
   from 1.23× to **1.46×** with no dtype in the IR, no new op and no new theorem.
   ⛔⛔ **And that measurement REFUTES the successor project** (§20.3): once the producing op writes
   bf16, XLA already propagates it across the intervening ops, so a dtype-carrying emit stack buys
   ~0 % on both nets it was scoped for. ▶ `planning/bf16_dtype_ir.md` is now a record of a
   correctly-scoped project that measurement cancelled — read its §0 before planning off it.
5. **The f32 carve-outs (BN, activations, SE, dense, optimizer) are NOT a fixed tax — their cost
   is architecture-dependent** (§14). They cost MobileNetV2 nothing and EfficientNet-B0 almost
   everything. ⭐ B0's 1.09× is now CONFIRMED on the bare device (150.98 → 136.00 = 1.10×, §16.4),
   so it is a real property of that net and not a driver artifact. ▶ Do NOT order future bf16 work
   by §9.1's JAX-side speedups: B0 is the largest there (2.41×) and the smallest here.

### The op kit as it now stands — 27 bf16 ops

```
flatConvFBf16                                            per-example forward (CNN rung)
convBf16  convStridedBf16  convStridedXlaBf16            batched forward
depthwiseBf16  depthwiseStridedBf16  depthwiseStridedXlaBf16
convBackBatchedBf16  convStridedBackBatchedBf16          dgrad
depthwiseBackBatchedBf16  depthwiseStridedBackBatchedBf16  depthwiseStridedXlaBackBatchedBf16
convWeightGradBBf16  convStridedWeightGradBBf16  convStridedXlaWeightGradBBf16   wgrad
depthwiseWeightGradBBf16  depthwiseStridedWeightGradBBf16  depthwiseStridedXlaWeightGradBBf16
convStride4Bf16                                          ConvNeXt's 4×4/s4 patchify stem
convStride4WeightGradBBf16                               its wgrad — the stem has NO dgrad
dotInBf16                                                dense (depth-1 PoC shape)

denseRowBf16  denseRowBackBf16  rowDenseWeightGradBBf16    ViT's six per-block matmuls
matmulFBBf16                                             SDPA QKᵀ / P·V — ACTIVATION × ACTIVATION
patchEmbedBf16  patchEmbedWeightGradBBf16                ⛔ BUILT, GATE-2 CORRECT, DELIBERATELY OFF
```

⛔⛔ **The last two are not used by any render and that is a measurement, not an omission** — ViT's
stem weight gradient in bf16 is 0.19× its f32 peer (§19.1). They are kept because the ops and the
emit shape are right; what is wrong is cuDNN's kernel for one shape.

⚠ **Every BIAS gradient stays f32 in every net** — `Σ_{batch,spatial} dy` is a reduction, not a
contraction, so there is nothing for a tensor core to do. Same for BN, the loss, the dense head
and the whole optimizer tail.

### Proof side

| what | state |
|---|---|
| `flatConvFBf16_faithful` / `_id` | ✅ the op adds rounding and nothing else |
| `conv_close_mixed` | ✅ `Proofs/Float/ConvMixedFloatBridge.lean` |
| `depthwise_close_mixed` | ✅ `Proofs/Float/DepthwiseMixedFloatBridge.lean`, §12.3 |
| whole-net composition — the mixed conv is `FloatClose` | ✅ `ConvMixedComposeBridge.lean`, §11 |
| ViT's six ops | ✅ nothing new needed — `dot_close_mixed` rounds BOTH operands and never asks which is a weight (§19.5) |
| a full training run on ANY of the seven | ⛔ not started |

### ⛔⛔ THE SUCCESSOR PROJECT WAS SCOPED AND THEN CANCELLED BY MEASUREMENT — §20.3

`planning/bf16_dtype_ir.md` scoped letting activations STAY bf16 between ops, on the strength of
§16.3 and §17.3. Its own §8 step 3 said to hand-build the bf16-through block first and that failing
to beat the bf16-with-boundary block would mean *"this project is not worth doing and the honest
outcome is to write that down"*. **Run on both nets it was scoped for, it does not beat it** —
1.77× vs 1.76× on ConvNeXt's block interior, 1.48× vs 1.47× on ViT's pass-through chain, with the
pass-through arms compiling to the *same* 73 converts.

⭐ **The thing that WAS worth doing turned out to need none of it**: the producing op's RESULT TYPE
(§20.1). Once a dot writes bf16, XLA propagates it downstream on its own. ▶ Do not start that
project. Do §18's first item instead — profile each net's bf16 ops against their f32 peers.

### ▶ ViT was surveyed, measured, NOT built — and then BUILT anyway at 1.23× — §17, then §19

§17 scoped six ops, settled all three emit shapes, ran §16.3's method **before** writing them, and
stopped the build at a predicted **1.03×**. ⭐ **The artifact measures 1.23×** — §17.3's isolated
op-set model was a lower bound on the real net, not an estimate of it (§19.7), because in the
artifact many boundary converts fuse into the LayerNorm/GELU/softmax between the matmuls.

⛔⛔ **And building it surfaced the branch's first bf16 op that is SLOWER than its f32 peer in a real
net** — the stem weight gradient, 0.19×, §19.1. Read that before wiring any new op kind.

▶ ⭐ And ViT then went to **1.46×** (§20) on one change — the dots' result type — which needed no
type-system work and which §9.2 had measured for correctness and never for speed.

### ⚠ Read the correction sections before trusting anything older

§9 refutes four load-bearing claims from the original scoping. §12.1 is refuted by §13.2. §14
refutes §9.1's expectation for B0. §16.2 adds a residency axis that every ms/step above §16 was
taken without stating. §17 refutes §10.3's "the OPS are the work" for ViT. The reasoning is kept in
place; the conclusions moved.

---

## Why

Measured this session on the JAX reference path, 4× 4060 Ti, synthetic input, `scripts/jax_imagenet_bench.py`:

| net | fp32 | bf16 | speedup |
|---|---|---|---|
| ResNet-34 | 378 ms/step · 1355 img/s | 201 ms/step · 2546 img/s | **1.88×** |
| ViT-Tiny | 450 ms/step · 2273 img/s | 286 ms/step · 3574 img/s | **1.57×** |

The verified Lean→MLIR→XLA path **cannot access any of this**: there is no bf16 in any renderer
or codegen module and no bf16 artifact in `verified_mlir/`. `TrainConfig.bf16` / `bf16Conv` are
consumed by the **JAX codegen only** — which is why `jax/MainResnetImagenet.lean` gets the 1.88×
and the verified R34 does not.

⚠ The 1.88× is **CUDA/cuDNN-specific**. On AMD/MIOpen bf16 conv is *slower*, which is why
`jax/MainResnetImagenet.lean` carries `bf16Conv := false` advice for that box. Anything built here
must keep bf16 opt-in per net AND per backend, exactly as the JAX side already does. Do not make it
a default.

## What already exists — more than you would guess

* **The structural render-tie, for MNIST-linear, at depth 1.** `LeanMlir/Proofs/Float/Bf16FaithfulPoC.lean`
  (75 lines) proves `bf16LinearGraph` **denotes** the exact-ℝ linear on the rounded operands, for
  *any* rounding `rnd`. Built only from `den`-faithful `SHlo` ops (`operand` → `dotIn` → `addBcast`).
* **The accuracy bound.** `FloatBridge.dense_close_mixed` (line 356), with `u_leaf = 2⁻⁸` for the
  bf16 leaf and `u_acc = 2⁻²⁴` for the fp32 accumulate. Plus `dot_close_mixed`,
  `dot_close_mixed_uniform`, `dense_close_mixed_uniform_budget`.
* **It lowers.** A `bf16`-in / `f32`-accumulate `dot_general` compiles for `sm_86`/`sm_89`. This is
  the inverse of the fp8 situation (`upstream-issues/2026-06-iree-cuda-fp8-nvptx-lowering/`), where
  the tie is proven but the graph will not lower on CUDA.

So render-tie ∘ accuracy = a tied-and-lowered bf16 forward, **for dense, at depth 1**. Rung 0 is
mostly plumbing, not proof.

## ▶ The trap that would waste the whole exercise

`Bf16FaithfulPoC` says depth-1 needs **no new `SHlo` op**, because the leaf cast is folded into the
operand value `rnd ∘ x`. That is true *of the proof* and false *of the emitter*.

If the emitted MLIR still carries `tensor<…xf32>` operands, XLA does fp32 math and **the speedup is
zero** — you would have a correct bf16 proof attached to a graph that never runs bf16. Tensor cores
engage only when the IR actually says `bf16`, with an `f32` accumulation type on the `dot_general`.

**So the emitter needs a real convert node and bf16 tensor types from rung 0**, even though the
proof does not need one until depth > 1. Good news: it is the *same* op — the `convertF` round node
the PoC names for depth > 1, with `den (convertF rnd e) = rnd ∘ den e`. Build it once, at rung 0,
and the depth > 1 proof ingredient comes free.

**Gate this explicitly.** Every rung's perf check must show a wall-clock move OR state that the rung
is too small to show one (see rung 0). A rung that reads ~1.00× has almost certainly emitted fp32.

### ⚠⚠ The trap is WORSE than written above — measured 2026-08-01, and it already bit

The paragraph above predicts "emit f32, get f32 speed". The reality on ares (jax 0.10.2, CUDA 12.9)
is that **XLA deletes the rounding**, so you get neither the speed nor the numerics:

```
.astype(bfloat16).astype(float32)   on  1.7640524
  eager   → 1.765625      (rounds correctly)
  jitted  → 1.7640524     (UNCHANGED — and the optimized HLO has no `convert` at all)
```

The algebraic simplifier treats a `convert(f32→bf16) → convert(bf16→f32)` **pair** as removable.
Confirmed a second way: a `bf16`-rounded 784×10 matmul scored error `3.8e-5` against a float64
reference — *identical* to true f32, where genuine bf16 rounding would give ~`1e-1`.

**This refuted the first emit strategy for `convertF`** (`5dc1df0`), which emitted exactly that pair.
The op, its `den`, and every tie built on it remain correct — what is refuted is the *emit*. The
node is kept because it is the proof-side round and the depth > 1 ingredient; it just must not be
read as "this graph runs bf16".

**Consequence for the ladder: the type-changing emitter is required at rung 0, not rung 2.** A
round trip cannot survive the optimizer, so the value has to stay bf16 *across an operation* — a
`dot_general` with bf16-typed operands and `preferred_element_type = f32`. Because that changes the
value's type, it cannot be an `SHlo n → SHlo n` node, which is the real design constraint this
whole discovery surfaces:

> `SHlo n` is indexed by WIDTH ONLY. It has no element type. Every existing op is implicitly f32
> (`ty` hardcodes `xf32`). Mixed precision needs the IR to carry a dtype — either a second index on
> `SHlo`, or a dtype field on the ops that consume/produce tensors.

That is the actual scoping question for this work, and it is bigger than "add a flag to the
renderer". Options, cheapest first:
1. **A bf16 `dotIn` variant** (`dotInBf16`) that emits bf16 operands + f32 accumulate as ONE node.
   No dtype in the type system; the cast lives inside the op. Matches how `flatConvF` already bundles
   conv+bias. Probably the right first move — it is rung 0-through-3 for dense with no IR redesign.
2. **A dtype index on `SHlo`.** Principled, and what a general mixed-precision story wants, but it
   touches every op and every existing proof.

Option 1 is strongly preferred to start. It also keeps the `den` story trivial: `den (dotInBf16 W e)
= dense (rnd∘W) (rnd∘den e)`, which is `bf16_render_faithful` with the rounding moved inside — the
tie already proven in `Bf16FaithfulPoC`.

**And this is why gate 2 is non-negotiable.** A numerical check alone would have passed: the
denotation says "rounded", the proof says "rounded", and the hardware quietly says "not rounded".
Only running it caught this.

## ▶ The proof gap that gates the money

✅ **CLOSED 2026-08-24 — `Proofs/Float/ConvMixedFloatBridge.lean`, and it was not the big item
this section claims. See §9.3.** The text below is kept for the reasoning that motivated it.

`FloatBridge` has `dot_close_mixed` / `dense_close_mixed`. It has **no `conv_close_mixed`.** The
mixed-precision accuracy story covers dense only.

Every net where the 1.88× lives — R34, MNv2, EfficientNet, ConvNeXt — is conv-dominated. So the
convnet rungs need a genuinely new accuracy theorem, not plumbing. That is the single biggest item
in this document, and it is why the ladder below puts conv late and says so up front rather than
discovering it at rung 3.

(ViT is the interesting exception: it is matmul-bound, so `dense_close_mixed` may carry most of it.
Its 1.57× is nearly the R34 number and it may be reachable *without* `conv_close_mixed` — worth
checking before committing to the conv proof. The patch embed is a conv, but it is one layer.)

## ▶ Per-op-class measurements — which rungs are even worth building

Measured 2026-08-01 on one 4060 Ti, 30 reps. **A** = plain f32; **B** = the `convertF` round
trip; **C** = true bf16 operands with f32 accumulate (the `dotInBf16` target).

| case | A f32 | B round trip | C true bf16 | C vs A |
|---|---|---|---|---|
| gemm 4096³ | 6.29 ms | 6.31 ms | 4.22 ms | **1.49×** |
| conv 256c 28² | 3.15 ms | 3.12 ms | 1.97 ms | **1.61×** |
| **depthwise 256c** | 0.41 ms | 0.41 ms | 0.81 ms | **0.50×** |

Three conclusions, and the third reorders the ladder:

1. **The round trip never buys speed.** With `xla_allow_excess_precision=true` (the default) B ≡ A
   because the pair is folded; with it `=false` the converts become real and B goes *slower* than
   f32 (conv 3.17 → 3.55, depthwise 0.41 → 0.80). So the flag recovers bf16 NUMERICS at a cost and
   never performance. Useful for an accuracy gate, useless as an optimisation.
2. **True bf16 pays on dense and standard conv** — 1.49× / 1.61×, consistent with the whole-net
   1.88× (R34) and 1.57× (ViT-Tiny) measured the same day.
3. **⚠ bf16 DEPTHWISE conv is 0.50× — twice as SLOW.** MobileNetV2, EfficientNet, MNv4 and ConvNeXt
   are all depthwise-dominated, so they capture far less of the win than R34, which is all standard
   convs and is therefore the best case rather than a typical one.
   ⛔⛔ **THE CONCLUSION IN THIS POINT IS REFUTED — see §9.1.** The 0.50× is a real number for the
   shape it was taken at and an unrepresentative one for any real net: at MNv2's own depthwise
   layers it is 0.86×, depthwise is ~13% of the step, and MNv2/B0 measure **1.94× / 2.41×** whole-
   step — *more* than R34. Do not read this point as a reason to skip those nets.

Point 3 independently confirms a decision already in the code: `LeanMlir/Types.lean` on `bf16Conv`
says *"Depthwise/separable convs (MobileNet/EfficientNet) still stay fp32."* That was asserted
without a number; it now has one, on CUDA. (The AMD note that bf16 conv is slower on MIOpen is a
*separate* effect — this is depthwise-specific and it is on NVIDIA.)

**Consequence: prioritise R34/dense/ViT, and do NOT expect the depthwise nets to pay.** Any bf16
renderer must keep depthwise on the fp32 path, exactly as the JAX side already does.

⛔⛔ **BOTH SENTENCES ARE WRONG — §9.1.** The depthwise nets pay MORE than R34 (1.94× / 2.41×),
and the JAX side does NOT keep depthwise on the fp32 path: `MainMobilenetV2Imagenet.lean` and
`MainEfficientNetImagenet.lean` both set `bf16Conv := true` and route `depthwise_conv` through
`convdt`. This paragraph asserted the JAX behaviour without checking it, and got it backwards.

## The ladder

Mirrors the fp8 ladder, which went exactly this route and whose `lean_exe` targets are the naming
template: `mnist-linear-e4m3-verified` → `mnist-mlp-e4m3-verified` → `mnist-cnn-e4m3-verified` →
`cifar-e4m3-verified` → `cifar8-e4m3-verified`.

| rung | net | new proof needed | new emitter needed |
|---|---|---|---|
| **0** | mnist-linear | **none** — `Bf16FaithfulPoC` is exactly this graph | `convertF` + bf16 types + f32-accum `dot_general` |
| **1** | mnist-mlp | depth > 1 tie: `den (convertF rnd e) = rnd ∘ den e` on intermediate activations | reuse rung 0 |
| **2** | mnist-cnn / cifar | **`conv_close_mixed`** — the big one | conv path emits bf16 operands, f32 accumulate |
| **3** | cifar8 / R34 | compose rung 2 over the block structure | reuse |
| **4** | ViT | possibly none beyond rung 1 — matmul-bound | reuse |

⛔ **THE LADDER WAS NOT THE ROUTE TAKEN, and the reader should know that before following it.**
The work went straight to rung 3 (R34/ImageNet) in 2026-08-24, skipping rungs 0–2 entirely,
because the ops needed for R34 turned out to be the same ops rungs 2–3 would have built and the
accuracy theorem turned out to be an instantiation rather than a new result (§9.3). Rung 2's
"the big one" is closed; rung 4's guess ("possibly none beyond rung 1") is right on the THEOREM
and wrong on the OPS — ViT's attention is activation × activation and needs a bundled op that
does not exist (§10.3).

▶ The ladder is still the right shape for a *proof-first* route. It was not the right shape for
a *payoff-first* one, and payoff-first is what got a net training. §10 is the plan that replaced it.

**Backward is a separate axis and is NOT in the table.** Every rung above is the *forward*. The
train step also carries the VJPs (`convBack`, `bnBack`, `denseRowBack`, the `*Sgd` tail folds). JAX
keeps master weights in fp32 and only casts the GEMM operands; the verified path must decide the
same question explicitly, and the `*Sgd` ops must stay fp32 or the optimizer state degrades.

⚠ **CORRECTION 2026-08-01 — an earlier draft of this paragraph said "keep the backward fp32, that
alone is most of the win, since the forward GEMMs dominate". That is FALSE and it inverts the
plan.** Measured on ares, a 4-layer 256ch 28² conv stack at batch 64:

```
fp32: fwd 12.91 ms   fwd+bwd 41.66 ms   bwd share 69.0%
bf16: fwd  5.85 ms   fwd+bwd 17.35 ms   bwd share 66.3%
```

**The backward is ~69% of a training step**, so:

| bf16 covers | step | speedup |
|---|---|---|
| nothing | 41.66 ms | 1.00× |
| forward only | 34.6 ms | **1.20×** |
| forward + backward | 17.35 ms | **2.40×** |

Forward-only captures about a SIXTH of the available win. So the conv **dgrad** and **wgrad** nodes
are not a follow-on — they are where the money is, and any plan that ships forward-only should
expect ~1.2× and say so.

**This is also the whole reason JAX gets this in one line and we do not.** JAX autodiffs the
backward *from* the forward, so casting the forward's operands propagates to the backward for free.
This repo's VJPs are hand-written (`convBack`, `convWeightSgd`, …) and inherit nothing — every
backward op needs its own bundled bf16 twin. The cost difference is structural, not incidental, and
it is the honest answer to "why not just flip a flag".

## Where the code changes

* **Tensor types: one chokepoint.** `LeanMlir/Proofs/Codegen/IRPrint.lean:30` is the whole tensor
  type emitter — `"tensor<" ++ … ++ "xf32>"`. The per-net renderers are near-dtype-agnostic
  (`MlpRender.lean` mentions `f32` twice). IRPrint has ~116 other `f32` mentions, but most are
  scalar/arith literals rather than tensor types; the parameter to thread is an element type, and
  it starts at line 30.
* **The op:** `convertF` in `LeanMlir/Proofs/Codegen/StableHLO.lean`'s `SHlo`, with its `den` lemma.
  Emits `stablehlo.convert`.
* **Plumbing:** copy the fp8 shape — a variant slug, a `lean_exe` per rung, a `verified_mlir/*_bf16_*.mlir`
  artifact per render, and `LEAN_MLIR_VARIANT` selecting it. `E4M3Quant.lean` + `trainE4M3` are the
  worked precedent for a precision variant that already threads end to end.

## Gates

Per rung, and the first two are the ones that catch the real failures:

1. **Render-tie** — the emitted graph denotes the rounded-operand computation. Rung 0 already has it.
2. **The bf16 actually reached the hardware** — `grep bf16` the emitted artifact AND show a
   wall-clock move at a size where one is expected. See the trap above.
3. **Accuracy** — instantiate the abstract `rnd` at bf16 round-to-nearest, feed `|rnd x − x| ≤ 2⁻⁸|x|`
   into the `*_close_mixed` bound for that op class.
4. **Trains** — loss descends, and final accuracy is within the fp32 arm's noise on the small nets.

⚠ **Rung 0 will show NO speedup and that is expected.** MNIST-linear is a 784×10 matmul: latency- and
transfer-bound, nowhere near tensor-core-bound. Do not read `1.00×` at rung 0 as "bf16 does not
help" — read it as "this rung's gate is gates 1/3/4, not gate 2". The first rung where a wall-clock
move should appear at all is rung 2, and the first where it should be *large* is rung 3. This is
worth writing down because the fp8 thread had the mirror-image confusion available to it.

## 9. ⚠⚠ WHAT THIS DOCUMENT GOT WRONG — measured 2026-08-24

Four corrections. Each is stated against the section that made the claim, so the reasoning
above stays readable as the reasoning it was.

### 9.1 ⛔ "bf16 depthwise is 0.50×, so do NOT expect the depthwise nets to pay" — REFUTED

§"Per-op-class measurements" point 3 concluded MobileNetV2 / EfficientNet / MNv4 / ConvNeXt
"capture far less of the win" and told the reader not to expect them to pay. **They pay more
than ResNet-34 does.** Whole train steps, real ImageNet, one 4060 Ti, the repo's own generated
modules with only `DT`/`CONV_DT` differing:

| net | fp32 | bf16 | speedup |
|---|---|---|---|
| ResNet-34 | 141.3 ms | 73.7 ms | 1.92× |
| **MobileNetV2** | 125.95 ms | 64.75 ms | **1.94×** |
| **EfficientNet-B0** | 139.83 ms | 57.98 ms | **2.41×** |

Two things made the original reading wrong:

* **The 0.50× was one unrepresentative shape** (a generic 256-channel depthwise). At MNv2's
  OWN depthwise layers the figure is **0.86×** forward+backward — a modest loss, not a rout.
* **Depthwise is only ~13% of the step.** ~16 ms of MNv2's 126 ms. The 1×1 expand/project
  convs dominate and win big, which is exactly what `jax/MainMobilenetV2Imagenet.lean` already
  said ("the 3×3 depthwise is a wash but harmless") while setting `bf16Conv := true`. ▶ The
  net config and this planning doc disagreed for three weeks, and **the config was right**.

⭐ A control arm settles it: `bf16` with `bf16Conv := false` on MNv2 measured **126.03 ms,
exactly 1.00×**. MNv2 is all convolution with no matmul to speak of, so the conv flag is the
only one that does anything there.

⭐ **And the depthwise loss is KERNEL SELECTION, not bandwidth or hardware.** Depthwise is
memory-bound; against the 4060 Ti's 288 GB/s peak, fp32 achieves 93 / 86 / 77 % across three
MNv2 layers while bf16 achieves only 45 / 44 / 58 %. cuDNN has an excellent fp32 depthwise
kernel on Ada and a poor bf16 one. Nothing fundamental — and on hardware whose bf16 tensor-core
advantage over fp32 is larger than Ada's, the compute-bound layers that carry the win should
gain more, not less. ⚠ That last sentence is INFERENCE, not measurement; no such card was tested.

### 9.2 ⛔ "The ONLY emit shape that reaches tensor cores: operands bf16, result f32" — HALF WRONG

`dotInBf16`'s emitter comment says this, and `StableHLO.lean` still carries it. Measured on
jax 0.11.0 and 0.10.2 alike, and NOT rescued by `xla_allow_excess_precision=false`:

| op | bf16 operands → **f32** result | bf16 operands → **bf16** result + convert |
|---|---|---|
| `dot_general` | ✅ bf16 reaches the hardware | ✅ reaches |
| `convolution` | ⛔ **FOLDED to f32** — cuDNN gets f32 params, zero converts | ✅ reaches |

True for dot, false for conv. Six ops copying `dotInBf16`'s shape would have shipped correct
proofs attached to graphs running fp32 — precisely the failure §"The trap" exists to prevent,
recurring inside the fix for the original trap. The shape that survives is a **bf16-TYPED
result then a separate convert**, which is what `jax/Jax/Codegen.lean`'s `conv2d` already
emits, and why the JAX lowerer gets bf16 on ImageNet and the verified path did not.

⚠⚠ **Check this by resolving the OPERAND SSA names in the optimized HLO.** The op line carries
only the RESULT type, and grepping it reports "bf16 reached" for a graph that folded. That
mistake was made twice in one session before the checker was written.

▶ Consequence for the `den`: a bf16-typed result means the hardware rounds the OUTPUT too
(f32 MAC accumulate, bf16 store), so the conv ops' `den` carries an OUTER rounding that
`dotInBf16`'s does not. Copying `den_dotInBf16` would claim more precision than the hardware
delivers — the unsound direction for an accuracy bound.

### 9.3 ✅ "`conv_close_mixed` — the single biggest item in this document" — DONE, and it was not big

§"The proof gap that gates the money" called it "a genuinely new accuracy theorem, not
plumbing". It is `dot_close_mixed_uniform` instantiated at fan-in `ic·kH·kW`, because **a
convolution output IS a dot product over its flattened receptive field**
(`conv2d_eq_flat_dot`, via `Tensor3.sum_flatten`). Plus one leaf rounding for the bf16 store
and one accumulate rounding for the bias — three terms, one per rounding the emit performs.
`LeanMlir/Proofs/Float/ConvMixedFloatBridge.lean`, builds in 3.3 s, no `sorryAx`.

⭐ Non-vacuous, and the fan-in is not what costs. `convBr` at fp32 accumulate / bf16 leaf on
R34's layers: **0.0078** (stem, n=147) → **0.0081** (stage-4, n=4608). The fan-in term is
2.8e-4 against the leaf's 7.83e-3 — a ~28× gap. Set the accumulate to bf16 as well and
`((1+u)^(n+1) − 1)` at n=4608 is **6.4e7**, i.e. vacuous. That contrast is the whole argument
for bf16-mixed over bf16, now at real fan-ins rather than in the abstract.

⚠ It bounds ONE conv against exact ℝ at an EXACTLY-REPRESENTED input. A whole-net bound needs an
error MODULUS, not a single-layer error. ✅ **DONE 2026-08-24 — `ConvMixedComposeBridge.lean`,
§11 below**, and the answer to "how bad is the composed bf16 bound" is ~1.86× the f32 one.

### 9.4 ⚠ "Forward-only captures about a sixth — expect ~1.2×" — measured 1.09×

Close enough in spirit (the conclusion "do not ship forward-only" stands, and is stronger),
but the number was optimistic. Measured on R34's own layer shapes: conv work alone is 1.68×
fwd+bwd, and forward-only is **1.09×**. The backward is **59.9%** of the fp32 conv step.

### 9.5 ▶ Where the remaining gap to JAX is

The verified R34 gets 1.41× end to end where JAX's whole step gets 1.92×. Most of the
difference is nameable, not mysterious: the verified emit **converts back to f32 after every
conv**, so activations cross layer boundaries in f32. JAX keeps them bf16 and lets XLA fuse the
converts into neighbours. The rest is BN, the residual adds, the loss, the heavy-ball tail, the
4-replica all-reduce and the shim feed, all f32 in both. ▶ Chasing the boundary converts is the
next perf lever and does NOT need new ops — only a decision about where the converts sit.

---

## 10. ⭐ THE PLAN FOR THE REMAINING NETS

Three categories, and they are genuinely different amounts of work. ⚠ "Mechanical" applies to
exactly one of them.

### 10.1 ✅ ResNet-50 — DONE 2026-08-24, and it WAS mechanical. **1.55×**, better than R34.

The prediction in this section held: no new op, no new theorem, `bf16 : Bool := false` threaded
through `ResNet50RenderB` and one `#eval`. 35 conv sites became a two-way choice, 32 block calls
were threaded, and R34's diff (`2739e34`) was followed line for line.

MEASURED, real ImageNet, `CUDA_VISIBLE_DEVICES=0,2,3,4`, `PJRT_REPLICAS=4`, 4×bs64 = global 256,
`LEAN_MLIR_MAX_STEPS=40` (median of steps 9..40), residency on, prefetch depth 8:

| arm | ms/step | step 0 | step 1 | step 2 |
|---|---|---|---|---|
| `momdp64` (f32) | 360 | 7.517632 | 7.527534 | 7.629933 |
| `momdp64bf16` | **232** | 7.526365 | 7.536716 | 7.625086 |
| | **1.55×** | | | |

⭐ **1.55× beats R34's 1.41×, and the direction is the expected one.** R50 is more conv-dominated
than R34 — the bottleneck's 1×1s are the layers tensor cores like best — so the f32 remainder this
render does not touch (BN, the residual adds, the loss, the heavy-ball tail, the all-reduce, the
shim feed; §9.5) is a smaller fraction of a heavier net.

⚠ **The loss agreement is LOOSER than R34's and that is worth stating rather than glossing.**
R34's arms matched to the 3rd–4th decimal; R50's match to ~**0.12 % relative** at step 0 and
0.06 % at step 2. Both are well inside one bf16 ulp (`2⁻⁸` = 0.39 %), and step 0 is a pure forward
comparison — no update has happened yet — so the whole difference is 53 conv layers of bf16
rounding composing. R50 stores through 53 bf16-typed conv results where R34 stores through 36,
which is the shape of the net rather than a defect. Both arms trace the same non-monotone
7.52 → 7.53 → 7.63 through step 2, which is the signal that says it is the same graph.

▶ **Gate 2, the one that actually needed checking, is green on a shape R34 never had.**
`scripts/bf16_gate2.py` resolves 158/158 convolutions with all operands bf16 in the OPTIMIZED HLO;
the f32 control arm reports 158/158 f32. **34 of those forward sites are stride-1 1×1 convs**
(`convBf16` at `kH = kW = 1`, so `pad = 0`) — R50's characteristic layer and the one thing about
this render that was not already exercised by R34, whose only 1×1s are its strided projections.
XLA neither folded them nor rewrote them into `dot_general`.

⭐ A structural check backs the loss comparison: the two artifacts have **identical counts across
all 24 `stablehlo` op kinds**, differing only by 474 added `stablehlo.convert` — exactly
158 convs × 3 (two operand casts, one result cast back). Same graph, one node kind added.

▶ **Gate 3 is an instantiation, not work.** `conv_close_mixed` is stated over arbitrary
`ic`/`kH`/`kW`. At R50's fan-ins, `convBr` at `M.u = 2⁻²⁴` / `L.u = 2⁻⁸` (arithmetic outside Lean,
quoted as illustration):

    layer                    fan-in n   fan-in term   leaf term   convBr
    1×1, ic=64                     64     3.9e-06     7.83e-03    0.0078
    stem 7×7, ic=3                147     8.9e-06     7.83e-03    0.0078
    3×3, ic=512 (widest)         4608     2.8e-04     7.83e-03    0.0081
    1×1, ic=2048 (widest 1×1)    2048     1.2e-04     7.83e-03    0.0080

R50's WIDEST fan-in is 4608 — the same as R34's — so the §9.3 numbers carry over unchanged, and
its characteristic 1×1s all sit BELOW its 3×3s. The leaf term still dominates by ~28×.

⚠ Still not done for R50, exactly as for R34: a full training run. The 40-step probe says the
graph is right and fast; it does not say what it converges to. ✅ The whole-net error bound is
no longer on this list — see §11.

### 10.2 ✅ MobileNetV2 — DONE 2026-08-24 (§12). EfficientNet-B0 / MNv4 — new op KIND

1.94× and 2.41× measured (§9.1), so these are worth more than R34. But they need
**depthwise bf16 twins**: `BatchableOp.depthwise` / `.depthwiseStrided` / `.depthwiseStridedXla`
plus `depthwiseBackBatched` and the depthwise weight-grads. Same emit discipline as the conv
ops — bf16 operands, bf16-typed result, convert back — and `feature_group_count = c` unchanged.

⚠ The accuracy side needs a `depthwise_close_mixed`, but it should go the way
`conv_close_mixed` did: a depthwise output is a dot product over a fan-in of `kH·kW` (one
channel, no `ic` sum), so it is the same instantiation at a much smaller `n`. Expect it to be
easier, not harder.

⚠ EfficientNet leaves its squeeze-excitation 1×1s in fp32 on purpose — they act on
1×1-spatial pooled tensors, where there is no bf16 win. Keep that.

### 10.3 ViT / ConvNeXt — genuine design work, do last

ViT's attention uses `matmulFB`: **activation × activation**, where every bf16 op built so far
is weight × activation. `dotInBf16` bundles the rounding of a constant weight and does not
cover it. Needs a new bundled op, and the four SDPA backward matmuls each need one.

▶ The accuracy side may come free: `dot_close_mixed` already rounds BOTH operands, so it does
not care that neither is a weight. The OPS are the work, not the theorem.

⛔⛔ **"THE OPS ARE THE WORK" IS REFUTED FOR ViT — §17, measured 2026-08-24.** Both sentences above
survive: `matmulFB` really is the new kind, and the theorem really does come free. But building
those ops in the established emit shape buys **1.03×** at ViT's rendered batch, because ViT's
matmuls are skinny and bandwidth-bound and the f32 boundary between ops costs what the tensor cores
save. Its whole 1.71–1.93× lives in keeping activations bf16 ACROSS ops. ▶ The work is neither the
ops nor the theorem — it is the type-system decision (Option 2) this document deferred at scoping.

ConvNeXt is partly §10.2 (depthwise) and partly this (large 1×1s that are matmuls in disguise).

### 10.4 ⚠⚠ THE RECIPE, and the trap that bit on R34's first run

Per net, in order:

1. **Ops first, gate 2 before wiring.** Emit one op standalone, compile it, and resolve the
   convolution's operand dtypes in the optimized HLO. Do not proceed on a grep of the op line.
2. **Thread `bf16 : Bool := false` as a TRAILING defaulted parameter**, the `wx`/`clip` idiom.
   Every existing render then re-renders byte-identical and gate 1 holds for free.
3. **⚠⚠ PASS THE FLAG TO THE VARIANT-NAME DERIVATION, not just to the block renderers.**
   `resnet34AdamTrainStepFaithfulB` derives its entry name from `r34AdamVariant`. Passing
   `bf16` to the renderers but not to THAT call wrote an artifact to
   `…momdp64bf16_train_step.mlir` that declared `@resnet34in_momdp64_train_step` inside, and
   the driver refused at load with an entry mismatch. `r34AdamVariant`'s own docstring warns
   about this for `wx` and `clip` and records ConvNeXt shipping it twice. **bf16 made three.**
   ▶ The failure is LOUD — the driver refuses rather than running the wrong graph.
   ✅ **R50 followed this and it did not bite** — `resnet50TrainStepFaithfulB` passes `bf16` to
   `r34AdamVariant` alongside `wdExclude`/`gradClip`/`bce`/`wdStr`, and the artifact declared
   `@resnet50in_momdp64bf16_train_step` on the first render. Three defects, then one clean one.
4. **Guard the slug against the DRIVER's variant predicates.** They read the same string to
   size the checkpoint blob. `cdOn` is a substring test for `"do"`; a false positive silently
   adds a region. `#guard` that the new slug contains no `acc`, no `ema`, no `do`.
5. **Check gate 1**: rebuild the render module and confirm `git status verified_mlir/` shows
   only the new file.
6. **Probe on real ImageNet** with `LEAN_MLIR_MAX_STEPS=40` — steady-state ms/step, then exits
   without the val drain. ⚠ Set `LEAN_MLIR_CKPT_TAG` or a finished run's checkpoint will make
   the probe exit instantly at "resuming from checkpoint at epoch 90".
7. **Compare the losses**, not only the time. The bf16 and f32 arms should agree to the 3rd–4th
   decimal on the first steps from the same init. That is the cheapest correctness signal there
   is, and it is what says the graph is the same graph.

---

## Sequencing against the other open lever

Device-resident parameters (§2d.3 in `xla_pjrt_handoff.md`) is fully scoped, has its gate written,
and is worth ~20 points of DP efficiency at 4 replicas. bf16 is worth up to 1.88× but needs a new
accuracy theorem to reach the convnets. **They are independent** — residency is transport, bf16 is
arithmetic — so either order works, and neither blocks the other.

If the goal is "match the JAX reference on R34/ImageNet", note it needs **both**: the JAX 4× number
is bf16 *and* device-resident, and this session measured the verified path missing both.

---

## 11. ✅ THE WHOLE-NET BOUND — done 2026-08-24, and the answer is a factor of 1.86

`LeanMlir/Proofs/Float/ConvMixedComposeBridge.lean`. No `sorryAx`; the three standard Mathlib
axioms only, audited in `tests/AuditAxioms.lean`.

### 11.1 ⛔ First, a BUG this uncovered: `lake build LeanMlir` had been broken since `b956efa`

`ConvMixedFloatBridge` declared `Proofs.convWindow` for the receptive field at type
`Tensor3 ic kH kW`. `SgdDescentCnn` **already** declared `Proofs.convWindow` for the same field at
type `Vec (ic·kH·kW)`. Two constants cannot share a full name, so the root module failed outright:

```
import LeanMlir.Proofs.Float.ConvMixedFloatBridge failed,
  environment already contains 'Proofs.convWindow' from LeanMlir.Proofs.Training.SgdDescentCnn
```

▶ **`conv_close_mixed` was therefore unreachable from the rest of the float stack** — which is
exactly why no composition had happened. It was not that the composition was hard; the file could
not be imported alongside the thing it had to compose with. The R34/R50 render work never noticed
because neither `ResNet50RenderB` nor the trainer imports that file. ⚠ Renamed to `convWindow3`;
the `Tensor3` shape is load-bearing (`conv2d_eq_flat_dot` needs `Tensor3.sum_flatten`), so the
name moved rather than the type. **`lake build LeanMlir` is green again.**

### 11.2 ⭐⭐ The backbone is PRECISION-AGNOSTIC — that is why this was one instance, not a rewrite

`FloatClose A B f fF L` says only: inputs bounded by `A` give outputs bounded by `B`, and an
input error `e` gives an output error `≤ L e`. **It never mentions how `fF` rounds.** So
`floatClose_relu`, `floatClose_bn`, `floatClose_maxPool3s2`, `floatClose_gap`,
`floatClose_residualBlock`, `floatClose_iterate` and `FloatClose.comp` accept a bf16 conv verbatim.

⭐ `floatClose_r50_stages_mixed` is *literally* `floatClose_r34_stages` — **R50 has the same
`[3,4,6,3]` stage depths as R34**; the nets differ in what a block contains, not in how many
blocks a stage stacks. The depth fold needed no R50-specific theorem at all.

Three things did have to be proved, none of which the `e = 0` bound gives:

| lemma | what it does |
|---|---|
| `convFanS_le` | replaces the data-dependent `Σ|kernel·window|` by the closed form `n·w·A` |
| `conv2d_sub_abs_le` | **the real conv is `n·w`-Lipschitz** — how a predecessor's error crosses the layer |
| `convMixed_close_prop` | the two combined, at an input both perturbed (`E`) and bounded (`A`) |

⚠ The budget is evaluated at **`A + E`, not `A`**: the float conv runs on the perturbed input, so
its own rounding scales with the perturbed magnitude. Writing `A` there understates it — the
unsound direction.

### 11.3 ⭐⭐ What bf16 actually costs the whole-net bound — and the part that is vacuous

Both budgets are **affine in the inherited error** (`convMixedBudget_affine`, `layerBudget_affine`),
and both slopes factor as `n·w·(1 + ε)`:

| arm | ε at `n = 4608` | gain factor |
|---|---|---|
| f32 (`layerBudget`) | `(1+2⁻²⁴)^(n+2) − 1` = **2.75e-4** | 1.000275 |
| bf16-mixed (`convMixedBudget`) | `br + u_leaf(1+br) + u_acc(1+u_leaf)(1+br)` = **1.20e-2** | 1.012043 |

▶ **bf16 does not change the growth RATE — it moves a `1+ε` factor.** Compounded:
**1.52× at R34's 36 conv layers, 1.86× at R50's 53.** Under a factor of two on the certificate,
for a 1.41×/1.55× speedup. That is the useful result of this section.

⚠⚠ **AND BOTH BOUNDS ARE VACUOUS IN ABSOLUTE TERMS.** The shared `n·w` factor is ~230 at
`n = 4608, w' ≈ 0.05`, so `gain^53` is astronomical for f32 and bf16 alike. This is a property of
worst-case forward-error analysis composed depth-first — every term assumes the adversarial sign —
**not** a property of bf16, and the repo's existing f32 whole-net bridges
(`Resnet34WholeFloatBridge`) carry exactly the same factor. ▶ The meaningful statement is the
RATIO. A non-vacuous ABSOLUTE number needs a different analysis — probabilistic rounding, or one
that exploits BN renormalising the activation scale at every layer — not a tighter conv lemma.
Saying otherwise would be the kind of claim §9 exists to correct.

---

## 12. ✅ MOBILENETV2 — 1.37×, the first GROUPED bf16 convs, and the number is BELOW R34's

`mobilenetv2in_adamdp64bf16_train_step.mlir`. Eight new ops in `StableHLO.lean`; render threaded
the same way R34/R50 were. Gate 1 green (only the new file), gate 2 green on both arms.

### 12.1 The measurement — and it is the WEAKEST of the three nets so far

4×bs64 on four 4060 Ti, real ImageNet, `LEAN_MLIR_MAX_STEPS=40` (median of steps 9..40):

| arm | ms/step | step 0 | step 1 | step 2 |
|---|---|---|---|---|
| `adamdp64` (f32) | 191 | 6.974680 | 7.041726 | 7.012378 |
| `adamdp64bf16` | **139** | 6.981724 | 7.046127 | 7.002637 |
| | **1.37×** | | | |

⚠⚠ **1.37× is below R34's 1.41× and R50's 1.55×, and FAR below the 1.94× §9.1 recorded for
MNv2.** That is not a contradiction — §9.1's 1.94× was measured on the repo's **JAX-codegen**
modules, which keep activations bf16 across layer boundaries. The verified emit converts back to
f32 after **every** conv (§9.5), and that overhead is proportionally worst on the lightest net:

| net | verified | JAX-side | share of the available win captured |
|---|---|---|---|
| ResNet-34 | 1.41× | 1.92× | 45 % |
| MobileNetV2 | **1.37×** | 1.94× | **39 %** |

MNv2 holds **40.1 MB** of parameters where R50 holds 292.5 MB. It is memory-bound, so 465 extra
`stablehlo.convert` nodes cost relatively more than they do in a compute-bound net. ▶ **This is
the first net where chasing §9.5's boundary converts would clearly pay more than adding ops.**

⛔⛔ **THE PARAGRAPH ABOVE IS REFUTED — §13.2.** The same graph measures **1.92× on ONE GPU**,
which is within noise of the JAX-side 1.94×. The verified emit's boundary converts cost MNv2
almost nothing; the 1.37× is the **data pipeline and the f32 collective**, not the converts and
not memory-boundedness. The "39 % of the win captured" figure is an artifact of the 4-replica
configuration, not a property of the renderer. ⚠ It also puts §9.5's whole "the boundary converts
are most of the remainder" reading in doubt — see §13.3.

Losses agree to **0.10 % / 0.06 % / 0.14 %** relative over the first three steps — the same
~0.1 % band R50 showed, and inside one bf16 ulp (2⁻⁸ = 0.39 %).

### 12.2 ⭐ The first GROUPED bf16 convolutions, and §9.2 has no exception for them

Before writing any Lean, a hand-built StableHLO module at a real MNv2 depthwise layer
(c = 144, 56², 3×3, `feature_group_count = 144`) was compiled three ways:

| emit shape | result |
|---|---|
| f32 | operands f32 (control) |
| bf16 operands → **f32-typed** result | ⛔ **FOLDED to f32** |
| bf16 operands → **bf16-typed** result + convert | ✅ bf16 reaches the hardware |

Identical to the ordinary-conv finding. **Grouping buys no exemption**, and §10.4 step 1 ("ops
first, gate 2 before wiring") is what made this a 15-minute check instead of a wasted render.

Eight new ops: `convStridedXlaBf16`, `depthwiseBf16`, `depthwiseStridedXlaBf16`,
`depthwiseBackBatchedBf16`, `depthwiseStridedXlaBackBatchedBf16`, `depthwiseWeightGradBBf16`,
`depthwiseStridedXlaWeightGradBBf16`, `convStridedXlaWeightGradBBf16`. Each keeps its f32 peer's
padding verbatim — notably the `[p+1, p-1]` dgrad shift that is the OPPOSITE of the weight grads'
`[p-1, p+1]`, which `scripts/xla_pad_op_check.py` caught being "fixed by symmetry" once already.

Whole-net gate 2: **155/155 convolutions with bf16 operands**, of which **34 are grouped**; the
f32 control reports 155/155 f32. Op histograms match on every kind, differing only by 465 added
`convert` = 155 × 3.

### 12.3 ✅ `depthwise_close_mixed` — §10.2 said "expect it to be easier", and it was

`LeanMlir/Proofs/Float/DepthwiseMixedFloatBridge.lean`, no `sorryAx`. A depthwise output IS a dot
product of length `kH·kW` — one channel, no `ic` sum — so it is `dot_close_mixed_uniform` at that
fan-in plus the bf16 store and the f32 bias add.

⭐ **The fan-in shrinks 461× and the bound moves 3.5 %:**

| layer | fan-in n | fan-in term | leaf term | bracket |
|---|---|---|---|---|
| depthwise 3×3 (every MNv2 block) | 9 | 6.01e-07 | 7.83e-03 | 0.0078 |
| R50 3×3, ic=512 | 4608 | 2.77e-04 | 7.83e-03 | 0.0081 |

▶ A depthwise layer is **not** meaningfully more accurate in bf16 than a dense conv — it is the
same 0.8 %. The fan-in rides the fp32 accumulate; the flat leaf term is what costs. That is §9.3's
separation, now confirmed at both ends of the fan-in range.

### 12.4 ⚠⚠ THE DUPLICATE-NAME TRAP FIRED AGAIN, one section after being documented

§11.1 records `Proofs.convWindow` colliding and breaking `lake build LeanMlir` for three commits.
The first draft of `DepthwiseMixedFloatBridge` defined `Proofs.dwWindow` — which
`DepthwiseFloatBridge` **already declares, with the identical type and meaning**. Same refusal:

```
import … DepthwiseMixedFloatBridge failed,
  environment already contains 'Proofs.dwWindow' from LeanMlir.Proofs.Float.DepthwiseFloatBridge
```

▶ **The fix was to REUSE, not to rename.** `dwWindow`, `dwKernelMat` and
`depthwiseConv2d_eq_dense` all existed; the new file now imports them and is shorter for it.
⚠ When a `conv*`/`dw*` helper looks missing, **grep before defining** — twice now the collision
was with a lemma that already did the job.

---

## 13. ✅ MNv4 — 1.88×; and ⛔ THE MEASUREMENT THAT REFUTES §12.1 AND CASTS DOUBT ON §9.5

### 13.1 MobileNetV4-Conv-M — three new ops, and MNv2 had already built the rest

`mnv4in_adam64bf16_train_step.mlir`. 47 conv/depthwise call sites threaded; **only three new ops**
(`depthwiseStridedBf16` and its dgrad/wgrad) because MNv4's UIB blocks use the **symmetric-pad**
`depthwiseStrided` family where MNv2 used the XLA-`SAME` one. Everything else — the stride-1
depthwise, the 1×1s, `convStridedXla` — came from §12 unchanged.

⚠ **SINGLE-DEVICE, deliberately.** MNv4 renders no DP variant at all: nothing has tied its
collectives, and a bf16 DP artifact would inherit that untied status while looking as trustworthy
as the rest. The precision axis does not get to quietly introduce the replica axis.

| arm | ms/step | step 0 | step 1 | step 2 |
|---|---|---|---|---|
| `adam64` (f32), 1 GPU | 120 | 6.960926 | 7.108018 | 7.074213 |
| `adam64bf16`, 1 GPU | **64** | 6.953143 | 7.115940 | 7.072668 |
| | **1.88×** | | | |

Gate 1 green; gate 2 **230/230 convolutions bf16, 60 of them grouped**, f32 control 230/230 f32;
histograms differ only by 690 `convert` = 230 × 3. Losses agree to 0.11 / 0.11 / 0.02 %.
Gate 3 needs nothing new: `depthwise_close_mixed` is stated over arbitrary `kH`/`kW`, so MNv4's
**5×5** depthwise is an instance (n = 25, bracket still 0.0078).

### 13.2 ⛔⛔ THE CONTROL THAT CHANGES THE STORY — MNv2 at 1 GPU is **1.92×**

MNv4's 1.88× and MNv2's 1.37× differ in **both** architecture and replica count, so neither
explains the other. `mobilenetv2in_adam64bf16` was rendered purely as a control — same net, same
graph, replica count the only variable:

| MNv2 arm | f32 | bf16 | speedup |
|---|---|---|---|
| 1 GPU, real data | 136 | 71 | **1.92×** |
| 4 GPU, **synthetic** input | 150 | 89 | 1.69× |
| 4 GPU, real data | 191 | 139 | 1.37× |

Per-step cost, decomposed:

| term | f32 | bf16 |
|---|---|---|
| compute (1 GPU) | 136 | 71 |
| + collective + DP overhead | +14 | +18 |
| + shim feed | +41 | +50 |

▶ **MNv2's bf16 arm on one GPU is 1.92×, against the 1.94× §9.1 measured on the JAX-codegen
module.** The verified renderer is at parity. §12.1's explanation — boundary converts, memory
boundedness — is **wrong**; the loss is the data pipeline first and the f32 all-reduce second.
⚠ Note both non-compute terms are LARGER in the bf16 arm (18 vs 14, 50 vs 41): they are f32 and
host-side, they do not shrink, and a faster GPU step simply spends longer waiting on them. At
4×bs64 the bf16 arm needs **1,842 img/s** from the shim, which is the regime
`next_session_a3_the_run.md` §1 warns about by name.

### 13.3 ⚠ WHAT THIS DOES AND DOES NOT SAY ABOUT §9.5 AND THE OTHER NETS

§9.5 attributes the verified path's gap to JAX to the boundary converts ("most of the remainder
has a name"). For MNv2 that is now measured to be **false**. ⚠ But R34's 1.41× and R50's 1.55×
are **4-replica** numbers, and no single-GPU bf16 render exists for either — so whether their gap
is also mostly pipeline-and-collective is **INFERENCE, NOT MEASUREMENT**, and this document has
been burned by exactly that kind of extrapolation before (§9).

▶ **The experiment that would settle it, named so it can be run:** render
`resnet34in_mom64bf16` / `resnet50in_mom64bf16` at `replicas := 1` and probe them on one GPU
against their f32 peers. One `#eval` each, no new ops, no new proof.

⭐ **The actionable conclusion that IS measured:** on a single device the verified bf16 path
reaches the reference's speedup. Any 4-replica bf16 number on this box should be read as a
*system* result — pipeline and collective included — not as a statement about the renderer, and
raising `SHIM_WORKERS` is the first thing to try before touching the emit.

▶▶ **RESOLVED 2026-08-24 — §16.3, and both sides were right.** The doubt this section casts on
§9.5 is settled by architecture rather than by one of them being wrong. MobileNetV2's convolutions
sit next to each other, so XLA fuses its boundary converts away and they cost nothing — §13.2's
finding. ConvNeXt's have a **LayerNorm between every pair**, so its converts can never fuse, and
measured on ConvNeXt's own conv set they turn **2.70× into 1.68×** — §9.5's claim, now with a
number. ▶ The lever is real and it is per-net; do not generalise either measurement.

▶ And this section named a third system term without knowing it. §16.2 adds it: the **parameter
round trip**, off by default, and the first thing to check on any net with a large parameter blob.

---

## 14. ⛔ EFFICIENTNET-B0 — ZERO new ops, every gate green, and **1.09×**. The worst result yet.

`efficientnetin_rms64bf16_train_step.mlir`. 23 conv/depthwise call sites threaded and **not one new
op**: every kind B0 uses on its AdamW/RMSProp path already had a bf16 twin from §12 (MNv2, 8 ops)
and §13 (MNv4, 3 ops). That part went exactly as §10.2 predicted.

### 14.1 The measurement, and the pipeline is NOT the excuse

Single-device (per §13.2 — a 1-GPU pair is what isolates the renderer), B = 64, RMSProp:

| arm | ms/step | step 0 | step 1 | step 2 |
|---|---|---|---|---|
| `rms64` (f32) | 163 | 6.946810 | 6.946942 | 6.965819 |
| `rms64bf16` | **149** | 6.945035 | 6.946255 | 6.963548 |
| | **1.09×** | | | |

⚠ **`LEAN_MLIR_BENCH_SYNTH=1` gives 163 → 148 = 1.10×** — indistinguishable. Unlike MNv2's
4-replica number (§13.2), this is **not** the data pipeline. B0's step is compute-bound and bf16
genuinely buys ~9 %.

⚠⚠ **THE SYNTH CONTROL IS WEAKER THAN THIS PARAGRAPH READS — §16.2.** `BENCH_SYNTH` removes the
data FEED and leaves the **parameter round trip** untouched, and that round trip is off-by-default
(`PJRT_FFI_RESIDENT`) and cost ConvNeXt 154 ms of a 312 ms step. So "not the data pipeline" was
established; "therefore the graph" was not. ✅ **The verdict survives anyway**: §16.4 timed B0's
two artifacts as bare device executables and got **151.0 → 136.0 = 1.10×**. B0's 1.09× is real,
and this section's conclusion stands on a measurement it did not originally have.

Gate 1 green. Gate 2 **146/146 convolutions bf16** (32 grouped); f32 control 146/146 f32; the
**179 `dot_general`s carry ZERO bf16**, confirming the SE gates and classifier stayed f32 by
design. Histograms differ only by 438 `convert` = 146 × 3. Losses agree to ≤ 0.03 %.

▶ So the graph is right, the bf16 reached the hardware, and the answer is still 1.09×.

### 14.2 ⛔ This REFUTES §9.1's expectation for B0 — it was the biggest payoff and is now the least

§9.1 records **2.41×** for EfficientNet-B0, the largest of the three nets it measured, on the
repo's own **JAX-codegen** modules where `bf16Conv := true` routes the WHOLE net through bf16.
The verified render converts back to f32 after every conv and keeps BN, swish, SE, the dense head
and the optimizer in f32. On MobileNetV2 those carve-outs cost **nothing** (1.92× verified vs
1.94× JAX, §13.2). On B0 they cost nearly everything.

▶ **The carve-outs are not a fixed tax — their cost is architecture-dependent, and B0 is the
worst case measured.** That is the actionable finding, and it is the opposite of what §10.2's
ordering assumed when it put the depthwise nets first for having "the biggest payoff".

### 14.3 ⚠ WHY — OPEN. The swish/SE reading below was NOT accepted; chase the OPS instead

▶▶ **STEER, 2026-08-24: EfficientNet is a different problem and wants a different investigation —
at the OP level, not at the "which tensors stayed f32" level.** The hypothesis below is recorded
because it is what the structural evidence suggested, **not** because it is the working theory.
Do not open §14.3 by building bf16 swish/SE ops.

▶ **What a fresh look should establish first**, in order:
1. **Which ops actually consume B0's step.** Time B0's conv work in isolation at its own layer
   shapes (the §9.4 method), and profile the step to attribute the rest per op kind. Everything
   below is inference until this exists.
2. **Whether B0's bf16 convs are even fast.** Gate 2 proves the operands are bf16; it proves
   NOTHING about whether cuDNN picked a good kernel for those shapes. §9.1 already found one
   depthwise shape where bf16 was **0.50×** — kernel selection, not bandwidth. B0's depthwise
   set (3×3 and 5×5 at many widths, plus SE) is the most varied in the repo and is exactly where
   a per-shape bf16 regression would hide.
3. **Whether the emit shape is right for B0's shapes specifically** — the §9.2 fold was found once
   for conv and once for grouped conv; a third variant is not impossible.

⚠ The 1.09× is solid and reproducible (real 1.09×, synth 1.10×) and gate 2 is green. The open
question is not *whether* B0 underperforms but *which ops* make it so.

---

#### The structural evidence, and the hypothesis it suggested (NOT the working theory)

What *is* measured, structurally: B0 carries **194 `stablehlo.logistic`** where MobileNetV2 carries
**zero** — the swish activations and the SE sigmoid gates. The elementwise-op-per-convolution ratio
is otherwise similar (49 vs 43), so this is **not** simply "B0 has more elementwise work"; it is
that B0's f32 remainder contains full-resolution **swish** and **SE gating** where MNv2's contains
relu6.

⚠⚠ **The causal claim — that the f32 swish/SE remainder is what eats the win — is INFERENCE.**
Op counts are not timings, and this document has been burned by exactly that kind of extrapolation
(§9). ▶ Also worth noting: §10.2 justifies keeping SE in f32 because "its 1×1s act on 1×1-spatial
pooled tensors, where there is no bf16 win". That reasoning covers the SE **matmuls** only.
`seBlock` / `seBackBatched` are BUNDLED ops that also perform the GAP and a **full-resolution gate
multiply** over `[B,c,h,w]` — which is not pooled-tensor work, and which that justification never
addressed.

▶ **Two experiments that would settle it, cheapest first:**
1. Time B0's conv work in isolation at its own layer shapes, the way §9.4 did for R34. That gives
   the conv share of the step directly and needs no new ops.
2. If the share is small, build bf16 twins for `swishB`/`swishBackB` and the full-resolution parts
   of `seBlock`/`seBackBatched`, and re-measure. ⚠ That widens the bf16 surface beyond
   convolution for the first time and needs its own accuracy story — do not start it before (1).

---

## 15. ✅ ConvNeXt — THE SURVEY, and it was right. **Exactly two new ops.** Result in §16.

Surveyed 2026-08-24 against `LeanMlir/Proofs/Codegen/ConvNeXtRenderB.lean` (the batched render —
`ConvNeXtRender.lean` is the per-example/fused-SGD peer and is NOT what the ImageNet artifacts use).

### 15.1 The op audit — what ConvNeXt uses, and what is missing

| ConvNeXt-T site | op | bf16 twin |
|---|---|---|
| block 7×7 depthwise | `.depthwise` | ✅ built for MNv2 |
| block 1×1 expand (c→4c) | `.conv` at kH=kW=1 | ✅ |
| block 1×1 project (4c→c) | `.conv` at kH=kW=1 | ✅ |
| stage downsample 2×2/s2 | `.convStrided` | ✅ |
| **patchify stem 4×4/s4** | **`.convStride4`** | ⛔ **NEW** |
| dgrads | `convBackBatched`, `convStridedBackBatched`, `depthwiseBackBatched` | ✅ |
| wgrads | `convWeightGradB`, `convStridedWeightGradB`, `depthwiseWeightGradB` | ✅ |
| **stem wgrad** | **`.convStride4WeightGradB`** | ⛔ **NEW** |
| bias grads ×4 | `*BiasGradB` | — stays f32, as in every net |
| classifier head | `.dotOut` / `.weightGradB` | — stays f32, as in every net |

⭐ **`convStride4` has NO dgrad** — it is the stem, so there is no input gradient. Two ops, not
three. ▶ §10.3's guess that ConvNeXt's "large 1×1s are matmuls in disguise" needing new
matmul ops is **wrong for this render**: they are `.conv` at kH=kW=1 and already covered. The
only true matmul is the classifier head, which stays f32 like everyone else's.

### 15.2 ⚠⚠ The three traps this net specifically carries

1. **`convStrided`'s ASYMMETRIC pad at an EVEN kernel.** ConvNeXt's 2×2/s2 downsample is the only
   even strided kernel in the repo, and `convStridedBackBatched`'s pad `[[kH-1-pH, pH], …]` is
   only observably different there — the symmetric spelling agrees at every odd kernel and is
   wrong at k=2. ✅ `convStridedBackBatchedBf16` already preserves it verbatim; **do not "tidy"
   it**, and re-read the comment on that emit case before touching anything nearby.
2. **The entry-name bug has now shipped FOUR times and ConvNeXt owns two of them** (`wx`, then
   `clip`). It has three distinct routes, all seen: (a) the flag never reaches the variant
   function; (b) it reaches the variant function but the variant's returned STRING never appends
   the marker (EfficientNet, §14); (c) the artifact path and the `#eval` disagree. ▶ `#guard` the
   variant spelling AND the slug against `cdOn`/`accOn`/`emaOn` before rendering.
3. **`cdOn` is a substring test for `"do"`.** ConvNeXt's own variants already carry `do`/`drop`
   markers, so check the composed slug, not just the `bf16` suffix.

### 15.3 The recipe, unchanged from §10.4 and now proven over five nets

1. Build the two ops (ctor → `denOp`/`den` → `batchOpDescr`/`skel` → emit). Mirror
   `convStridedBf16` exactly; the emit shape is **bf16 operands, bf16-TYPED result, convert back**.
   ⚠ That is the CONV shape. `dotInBf16` uses a **different** shape (bf16 operands, f32 result) and
   it is correct for `dot_general` — §9.2. Do not unify them.
2. Gate 2 on ONE op standalone before wiring. Cost: 15 minutes; it has paid twice.
3. Thread `bf16 : Bool := false` as a TRAILING defaulted parameter everywhere, including the
   variant-name function AND its return string.
4. Render single-device first (§13.2) — a 1-GPU pair is what isolates the renderer.
5. Gate 1 (`git status verified_mlir/`), gate 2 on BOTH arms, op-histogram diff (must differ only
   by `3 × nconv` converts), then a 40-step probe with `LEAN_MLIR_CKPT_TAG` set.

### 15.4 ✅ THE SURVEY HELD — every prediction in §15.1–15.3 was confirmed by the build

* **Two ops, not three.** `convStride4Bf16` and `convStride4WeightGradBBf16`. The "`convStride4`
  has NO dgrad" call was right: it is the stem, `%x` is its input, there is no input gradient.
* **§10.3's "large 1×1s are matmuls in disguise" was wrong for this render**, as §15.1 said. The
  block 1×1s are `.conv` at `kH = kW = 1` and `convBf16` covers them. The artifact's only three
  `dot_general`s are the classifier head and its two grads, all f32 by design.
* **Trap 1 (the even-kernel asymmetric pad) held** — see §16.1's geometry check.
* **Trap 2 (the entry name) did not fire.** Threading `bf16` into `cnxAdamVariant`'s signature AND
  its returned string, and into both halves of `convNextAdamTrainStepFaithfulB`, was enough; both
  artifacts declared the right entry on the first render.

---

## 16. ✅ ConvNeXt-T — **1.29×**, two new ops, and the measurement that had to be redone

`convnextin_adamwxclipdropbf16_train_step.mlir` (single device) and
`convnextin_adamdpwxclipdropbf16_train_step.mlir` (4 replicas, §0.5's rule). Two new ops, exactly
as §15.1 predicted. Gate 1 green — every committed artifact re-renders byte-identically, and
`convnext-fwd-b-tie` is unmoved.

### 16.1 The gates

| gate | result |
|---|---|
| **1** — nothing else moved | ✅ `git status verified_mlir/` shows only the two new files |
| **entry name** | ✅ both declare `@convnextin_adam…bf16_train_step`, first render |
| **2** — bf16 reached the hardware | ✅ **173/173 convolutions with all operands bf16**; f32 control **173/173 f32** |
| **2b** — the head stayed f32 | ✅ the 3 `dot_general`s carry **zero** bf16 |
| **histogram** | ✅ identical across **all 21** `stablehlo` op kinds, differing only by **519 `convert` = 173 × 3** |
| **geometry** | ✅ the two arms' convolution `window` specs are the **same multiset** — 5 distinct specs, 173 sites |
| **3** — accuracy | ✅ instantiation, no new theorem (below) |
| losses, first 3 steps | ✅ agree to **0.0005 % / 0.026 % / 0.004 %** — well inside one bf16 ulp (2⁻⁸ = 0.39 %) |

⭐ **The standalone gate-2 check (§15.3 step 2) paid for the third time.** Before a line of Lean:
at the stem's own shape (B=32, 3→96, 224²→56², 4×4/s4) and at the stem wgrad's
(`[3,B,224,224] × [96,B,221,221] → [3,96,4,4]`), `bf16 operands → f32-TYPED result` **folds to
pure f32** and `bf16 operands → bf16-TYPED result → convert` reaches the hardware. **Stride buys no
exemption from §9.2** any more than grouping did (§12.2). There is no third emit variant — which
is the answer to §14.3's item 3, at least for stride.

⭐ **Trap 1 held, and it is visible in the geometry check.** The 5 distinct window specs include
`pad = [[1, 0], [1, 0]]` at 3 sites — the 2×2/s2 downsample's **dgrad**, `[[kH-1-pH, pH]]` at
`k = 2`, the repo's only site where the asymmetric spelling differs observably from the symmetric
one. `convStridedBackBatchedBf16` preserved it verbatim.

▶ **Gate 3 is an instantiation, not work**, for the reason R50's was (§10.1): `conv_close_mixed` is
stated over arbitrary `ic`/`kH`/`kW`, and a stride-4 output *is* a `conv2d` output read at the
decimated position `4i+1` — decimation selects which outputs survive, it never changes what one
output computes. That is the same argument stride-2 already rides on. `M.u = 2⁻²⁴` / `L.u = 2⁻⁸`
(arithmetic outside Lean, quoted as illustration):

    layer                          fan-in n   fan-in term   leaf term   convBr
    stem 4×4/s4, ic=3                    48     2.94e-06     7.83e-03   0.0078
    7×7 depthwise (every block)          49     3.00e-06     7.83e-03   0.0078
    1×1 expand, ic=96                    96     5.83e-06     7.83e-03   0.0078
    2×2/s2 downsample, ic=384          1536     9.23e-05     7.83e-03   0.0079
    1×1 project, ic=3072 (widest)      3072     1.85e-04     7.83e-03   0.0080

ConvNeXt's widest fan-in (3072) is smaller than R34/R50's 4608, so §9.3's separation is if anything
wider here: the leaf term beats the fan-in term by **42×** at the widest layer and **2659×** at the
stem. ⚠ Set the accumulate to bf16 as well and `(1+u)^(n+1) − 1` at n = 3072 is **1.6e5**, i.e.
vacuous — the whole argument for bf16-**mixed**, restated at a sixth net's fan-ins.

### 16.2 ⛔⛔ THE MEASUREMENT, AND THE FIRST ONE WAS WRONG — `PJRT_FFI_RESIDENT` IS OFF BY DEFAULT

Single device (§13.2: a 1-GPU pair is what isolates the renderer), B = 32, AdamW + wx + clip + drop,
real ImageNet, `LEAN_MLIR_MAX_STEPS=40` (median of steps 9..40):

| arm | resident | ms/step | speedup |
|---|---|---|---|
| `adamwxclipdrop` (f32) | **off** | 312 | |
| `adamwxclipdropbf16` | **off** | 280 | 1.11× |
| `adamwxclipdrop` (f32) | **on** | 165 | |
| `adamwxclipdropbf16` | **on** | **128** | **1.29×** |
| *the bare device executable, no driver at all* | — | 157.6 → 120.9 | **1.30×** |

⚠⚠ **The 1.11× is a SYSTEM result and it is a NEW way to get one.** §13.2 named two — the shim
feed and the f32 all-reduce. This is a third: **the parameter round trip.** ConvNeXt-T holds
**540 parameter tensors, 327.2 MB**; without `PJRT_FFI_RESIDENT=1` they cross PCIe every step, and
**154 ms of that 312 ms step is not the graph**. Turn residency on and the trainer's number
(1.29×) lands on the bare-device number (1.30×) — i.e. the driver is then accounted for entirely.

⚠⚠ **AND `LEAN_MLIR_BENCH_SYNTH` DOES NOT CONTROL FOR IT.** Synthetic input gave 312 → 275 = 1.13×,
indistinguishable from real — which is *true* and says nothing about residency, because
`BENCH_SYNTH` removes the data FEED and leaves the parameter traffic untouched. §14.1 used exactly
that control on B0 and read it as "the pipeline is not the excuse"; that reading happens to survive
(§16.4), but the control was weaker than the sentence claimed. ▶ **The check that actually
separates renderer from driver is timing the compiled artifact directly**, with no trainer around
it. It takes one script and it should now be the default for any bf16 claim.

▶ **ConvNeXt is the first net on this branch where residency binds**, and the reason is size, not
architecture: 28.6M parameters against MobileNetV2's 3.5M and B0's 5.3M — 327 MB per step against
42 and 64.

### 16.3 ⭐⭐ THE FULL ATTRIBUTION OF A STEP — the first on this branch, and the method §14.3 asked for

§14.3 item 1 said: *time the conv work in isolation at the net's own layer shapes, and profile the
rest per op kind; everything else is inference until this exists.* Done, for ConvNeXt, against the
**157.6 ms bare-device f32 step**:

| term | ms | share |
|---|---|---|
| **convolutions**, fwd+bwd, at ConvNeXt-T's own 58 sites | **111.7** | **71 %** |
| channel-LayerNorm × 22 sites | 8.1 | 5 % |
| GELU × 18 sites | 9.1 | 6 % |
| LayerScale × 18 sites | 2.5 | 2 % |
| AdamW + global-norm clip, 180 tensors / 28,587,592 scalars | 4.9 | 3 % |
| **accounted** | **136.2** | **86 %** |

⭐ **The isolated conv stack lowers to exactly 173 convolutions fwd+bwd — the artifact's own
count.** That is what licenses reading 111.7 ms as "the artifact's conv work" rather than as a
lookalike.

⚠ One attempt at the carve-out row was **wrong and looked plausible**: broadcasting a single
scalar into every site let XLA fold the whole stack away and reported 22 LayerNorms in 1.56 ms.
Real independent inputs per site is what makes this a measurement.

⭐⭐ **And it MEASURES §9.5's boundary converts for the first time.** The same conv stack, two ways:

| conv-only arm | fwd+bwd | speedup | converts in the optimized HLO |
|---|---|---|---|
| converts free to fuse across layers | 111.7 → 41.3 | **2.70×** | 290 |
| f32 boundary **forced** (as LayerNorm forces it) | 111.7 → 66.5 | **1.68×** | 347 |
| the artifact | — | — | **519** = 3 × 173 |

▶ **The boundary converts cost ConvNeXt 25 ms of conv time — 2.70× becomes 1.68×.** §9.5 claimed
this was "the next perf lever" and §13.3 put that claim in doubt after MNv2 showed it costing
nothing. Both are right, and the resolution is architecture: MNv2's convs sit next to each other,
ConvNeXt's have a LayerNorm between every pair, so its converts can never fuse away. ⚠ It is
still the SECOND-order term here — see the ceiling below.

**The ceilings, so nobody chases the wrong lever:**

* If bf16 made every convolution **free**: 157.6 → 45.9 = **3.4×**. Conv is 71 % of ConvNeXt's
  device step, so the carve-outs are NOT what caps this net — unlike B0.
* At the boundary-constrained conv speedup (1.68×): 157.6 → 112.5 = **1.40×**. Realized **1.30×**;
  the residue is the artifact's 519 converts against the barrier arm's 347.
* ▶ So ConvNeXt's remaining bf16 headroom is **the boundary converts, and only them** — worth
  perhaps 1.30× → 1.40×, and needing **no new ops**, only a decision about where the converts sit.
  Building bf16 LayerNorm/GELU twins would chase 13 % of the step for a new accuracy story.

### 16.4 ⭐ THE CROSS-CHECKS — the method validated, and §14's B0 conclusion CONFIRMED

Bare-device timings of the already-committed pairs, taken with the same tool:

| net | device f32 → bf16 | device speedup | what its trainer probe reported |
|---|---|---|---|
| MobileNetV2 `adam64` | 123.0 → 58.4 | **2.11×** | 1.92× (§13.2) |
| EfficientNet-B0 `rms64` | 151.0 → 136.0 | **1.10×** | ⛔ 1.09× (§14) |
| ConvNeXt-T `adamwxclipdrop` | 157.6 → 120.9 | **1.30×** | 1.11× off / 1.29× on |

* **MNv2 validates the method**: its device number sits 12 ms under its trainer number and moves
  the speedup 1.92 → 2.11×, i.e. *better* than the JAX reference's 1.94×. Small blob, small driver
  tax.
* ⭐ **B0's 1.09× is CONFIRMED, not refuted.** On the bare device it is 1.10×. So §14's conclusion
  — that B0's f32 carve-outs eat nearly the whole win — stands on its own, and §14.3 remains the
  open question it was. What §16.2 corrects is only the *strength* of §14.1's synth control, not
  its verdict.
* ⚠ **R34 and R50 have NOT been checked this way** and hold 21.8M / 25.6M parameters — nearer
  ConvNeXt's blob than MNv2's. Their 4-replica numbers may be carrying a residency tax nobody has
  measured.
  ⛔ **AND IT IS NOT "just run the script" — TRIED 2026-08-24 AND IT CANNOT WORK AS IS.** The only
  bf16 R34/R50 renders are `momdp64`, i.e. **4-replica**: they carry `all_reduce`, so compiling
  them for one device drives XLA's conv autotuner into a core dump rather than a clean refusal.
  ▶ The prerequisite is the render §13.3 already named — `resnet34in_mom64bf16` /
  `resnet50in_mom64bf16` at `replicas := 1`, one `#eval` each, no new ops and no new proof — and
  the bare-device timing follows. MNv2, B0 and ConvNeXt were checkable because their bf16 renders
  are single-device to begin with.

### 16.5 What ConvNeXt did NOT need

* **No new theorem.** `conv_close_mixed` covers the stem by instantiation (§16.1);
  `depthwise_close_mixed` is stated over arbitrary `kH`/`kW`, so the **7×7** depthwise is an
  instance at n = 49 exactly as MNv4's 5×5 was at n = 25.
* **No `convStride4BackBatchedBf16`.** The stem has no input gradient.
* **No new matmul op.** §10.3's ConvNeXt half is retired; its ViT half stands.
* **No driver change.** `bf16` adds no marker the driver's substring predicates read
  (`emaOn`/`cdOn`/`accOn`), which the `#guard`s check on the full concatenations.

### 16.6 ⚠ What is NOT measured

* **The DP arm.** `convnextin_adamdpwxclipdropbf16` renders and its gates are the single-device
  ones; it has not been probed. Per §13.2 a 4-replica number is a system result anyway, and per
  §16.2 this net now has a third system term to control for.
* **Anything about convergence.** Three steps of loss agreement is not training.

---

## 17. ⛔⛔ ViT — SURVEYED AND MEASURED, **NOT BUILT**, and the measurement is why

> ⛔⛔ **SUPERSEDED 2026-08-25 — IT WAS BUILT, AND IT IS 1.23×, NOT 1.03×. SEE §19.** This section's
> op audit (§17.1) and emit shapes (§17.2) were confirmed exactly by the build. Its *decision* —
> stop, the established shape buys 1.03× — rested on §17.3's isolated op-set timing, and the
> artifact beat it (§19.7): an isolated stack pays the f32 boundary in full, while the real net
> fuses many of those converts into the LayerNorm/GELU/softmax that sit between its matmuls.
> ▶ §17.4's conclusion still stands and is the successor project: ViT's remaining 1.23× → ~1.7×
> is in keeping activations bf16 BETWEEN ops. The reasoning below is kept as the reasoning it was.

Six ops were scoped and the emit shape for every one of them was settled. Then §16.3's method was
run **prospectively** — before writing any of them — and it says the six ops in the established
emit shape would buy **1.03×** at the batch ViT actually renders at. §15.3 step 2 exists to stop a
wasted render; this is the same discipline one level up, stopping a wasted *op set*.

### 17.1 The op audit — six ops, and one of them is a new KIND

Surveyed against `ViTRenderB.lean` (the batched render; `ViTRender.lean` is the per-example peer,
the same split ConvNeXt has). ViT-Tiny: B = **32**, 197 tokens, D = 192 = 3 × 64, MLP 768, depth 12.

| ViT site | op | bf16 twin |
|---|---|---|
| patch embed 16×16/s16 | `patchEmbed` (a **convolution**) | ⛔ NEW |
| its weight grad | `patchEmbedWeightGradB` | ⛔ NEW |
| Q/K/V/O/fc1/fc2 — 6 per block × 12 | `denseRow` | ⛔ NEW |
| their input-VJPs | `denseRowBack` | ⛔ NEW |
| their weight grads | `rowDenseWeightGradB` | ⛔ NEW |
| **attention QKᵀ and P·V**, + the 4 backward matmuls | `matmulFB` | ⛔ NEW — **activation × activation** |
| classifier head | `dense`/`dotOut`/`weightGradB` | — stays f32, as in every net |
| LN, GELU, softmax, transposes, slices, AdamW tail | — | — stays f32 |

⭐ §10.3's claim that `matmulFB` is the genuinely new kind is **right**: every bf16 op built for the
five convnets is weight × activation, and `dotInBf16` bundles the rounding of a *constant* weight.
▶ And §10.3's other claim — that the accuracy side comes free — is also right: `dot_close_mixed`
rounds BOTH operands and never asks which is a weight.

### 17.2 ⭐ The emit shape, settled by measurement for all THREE forms

Standalone, at ViT's own shapes, before any Lean (§15.3 step 2):

| form | shape | `bf16 → f32` result | `bf16 → bf16` result + convert |
|---|---|---|---|
| `denseRow` | `[32,197,192] × [192,768]` | ✅ reaches (as a **cuBLAS** gemm) | ✅ reaches (as a fused `dot`) |
| `matmulFB` | `[32,197,64] × [32,64,197]`, batched | ✅ reaches | ✅ reaches |
| `patchEmbed` | 16×16/s16 convolution | ⛔ **FOLDS to f32** | ✅ reaches |

▶ **§9.2's split holds exactly, at a fourth and fifth op form**: `dot_general` is unaffected by the
result type and `convolution` is not. Batching dims buy no exemption, and neither does
activation × activation. That is now checked at dot (dense), conv (stride 1/2/4), grouped conv, and
batched dot — the rule is general.

### 17.3 ⛔⛔ THE MEASUREMENT THAT STOPPED THE BUILD

The **whole f32 ViT-Tiny train step is 29.95 ms** on the bare device
(`vitin_adamwxclipdrop`, B = 32), and its **matmul set alone is 27.0 ms — about 90 %**. ViT really
is matmul-bound, exactly as §10.3 and rung 4 assumed. So the ops are pointed at the right 90 %.

Then, the same matmul set timed three ways — every arm gate-2 checked (387 dots, 3655 bf16-typed
values in the bf16 arms, so the bf16 genuinely reached the hardware):

| B | f32 | **bf16 in the verified emit shape** | bf16 **throughout** (the JAX reference's world) |
|---|---|---|---|
| **32** (what ViT renders at) | 26.9 | 26.2 — **1.03×** | 15.7 — **1.71×** |
| 128 | 136.7 | 129.0 — **1.06×** | 70.9 — **1.93×** |
| 256 | 317.0 | 280.7 — 1.13× | — |

⭐ The "throughout" column brackets §9.1's independently-measured JAX-side **1.57×** for ViT-Tiny,
which is what says this model of the two worlds is the right one.

▶▶ **The entire ViT bf16 win lives in keeping activations bf16 ACROSS op boundaries, and the
verified emit shape cannot do that.** Six ops built the established way would capture 0.7 ms of an
11.2 ms prize.

⛔⛔ **MEASURED WRONG — the six ops were built and the artifact is 1.23×, not 1.03× (§19).** The
table above is a correct measurement OF THE ISOLATED MATMUL SET and a wrong prediction of the net:
in isolation every boundary convert is paid, and in the artifact many fuse into the LayerNorm, GELU
and softmax between the matmuls. ▶ Read an isolated op-set timing as a **lower bound** on the
artifact. The "throughout" column is unaffected and is still what the successor project targets.

### 17.4 ⭐⭐ WHY ViT DIFFERS FROM THE CONVNETS — and why this is the net that forces the deferred decision

The f32 boundary is not new; §16.3 measured it costing ConvNeXt 2.70× → 1.68×. What is new is how
much of the saving survives it:

| net | saving with converts free to fuse | saving through the f32 boundary | **kept** |
|---|---|---|---|
| ConvNeXt-T conv work | 70.3 ms | 45.1 ms | **64 %** |
| ViT-Tiny matmul work (B=32) | 11.2 ms | 0.7 ms | **6 %** |

⚠ The explanation is arithmetic intensity, and it is REASONING FROM THESE NUMBERS rather than a
separate measurement: a convolution reuses each loaded input across many output positions, so a
cast amortises over a lot of arithmetic; ViT's matmuls are **skinny** (contracting dim 192, or 768
at the MLP), so they are bandwidth-bound on the activations and an f32→bf16→f32 round trip at every
op costs about what the tensor cores save.

▶▶ **So ViT is where the original scoping's Option 1 runs out.** That scoping offered two routes
and chose the cheap one:

> 1. **A bf16 `dotIn` variant** … No dtype in the type system; the cast lives inside the op.
>    Probably the right first move.
> 2. **A dtype index on `SHlo`.** Principled … but it touches every op and every existing proof.

Option 1 has now carried **six nets** and it was the right call every time. ViT is the first net
where it does not work: bundling the cast inside each op *forces* the f32 boundary, and for ViT the
f32 boundary is the whole cost. ▶ **Getting ViT's 1.7–1.9× needs activations that stay bf16 between
ops.** That is a design project, not an op-writing project, and it should be scoped as one.

⚠ **THE SENTENCE ABOVE ORIGINALLY SAID "i.e. the IR carrying a dtype (Option 2)" AND THAT WAS TOO
STRONG — corrected 2026-08-25, `planning/bf16_dtype_ir.md` §2.** What forces the f32 boundary is
Option 1 **as practised** — every bf16 op built so far takes f32 in and gives f32 out. A
`bf16-in / bf16-out` variant bundles its casts exactly as today and forces no boundary. The real
question is what stops a bf16-producing op being wired to an f32-consuming one, and that has a
cheap answer (a defaulted `dt` FIELD + a dtype-carrying emit stack, no proof churn) as well as an
expensive one (an index on `SHlo`: 236 constructors, 230 theorems, 65 files). ▶ The overstatement
mattered because it made the cheap route look unavailable.

⭐ **And it is the same project ConvNeXt's remaining headroom needs** (§16.3: 1.30× → ~1.40×, "the
boundary converts, and only them"). One decision serves both nets — which is a much better reason
to take it on than either net alone.

### 17.5 ⚠ What would make ViT worth doing in the CURRENT shape

Nothing measured here does. For completeness, the two things that would change the answer:

* **A bigger batch.** ViT renders at B = 32 because `vB` is a private constant (ConvNeXt has the
  same constraint at 32). The verified-emit speedup rises 1.03 → 1.06 → 1.13× at B = 32 → 128 →
  256, so even at 8× the batch it is still under 1.2×. Batch alone does not rescue it.
* **Hardware whose bf16 tensor-core advantage over fp32 is larger than Ada's.** ⚠ INFERENCE, not
  measurement; no such card was tested — the same caveat §9.1 attaches to its version of this.

---

## 19. ✅ ViT-Tiny — **BUILT after all, and 1.23×**. §17 predicted 1.03× and was wrong twice over.

`vitin_adamwxclipdropbf16_train_step.mlir` (single device, B = 32). Six ops, exactly the set §17.1
scoped. ⚠⚠ **And the headline result of this section is NOT the speedup — it is the op that had to
be turned OFF to get it.**

### 19.1 ⛔⛔ THE FINDING: ViT's STEM WEIGHT GRADIENT HAS NO USABLE bf16 cuDNN KERNEL

Bare device, `scripts/bf16_device_step.py`, B = 32, median of 25, f32 control **29.89 ms**:

| arm | `bf16Conv` | `bf16ConvW` | ms/step | speedup |
|---|---|---|---|---|
| **shipping** | false | false | **24.39** | **1.23×** |
| stem forward bf16 too | true | false | 24.64 | 1.21× |
| stem **wgrad** bf16 too | false | **true** | **57.22** | ⛔ **0.52×** |
| both convolutions bf16 | true | true | 57.29 | ⛔ 0.52× |

▶ **One op, one site, +32.8 ms on a 29.89 ms step.** The first render of this net turned all six ops
on and measured **0.52× — nearly twice as SLOW as f32** — with every gate green: gate 1 clean, gate
2 green on both arms, op histograms identical but for the converts, losses inside an ulp. Nothing
structural saw it and nothing structural could have.

⭐ **An `nsys` profile named it in one line.** The bf16 arm's `__cudnn$convBackwardFilter` lowers to
`sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_*` at **~30 ms**; the f32
arm's same op lowers to `conv2d_grouped_direct_kernel<float>` at **~5.8 ms**.

⚠ **The shape is why, and it is specific to the transpose-trick wgrad.** That convolution is
`[3,B,224,224] × [192,B,209,209] → [3,192,16,16]` — the DILATED cotangent as the filter — so cuDNN
sees a **209×209 window**. It has a direct f32 kernel for that and no bf16 one, so bf16 falls back
to implicit GEMM over a window ~170× larger than any real kernel, plus an NCHW→NHWC layout
transform the f32 path does not pay. ▶ The stem FORWARD (16×16/s16, an ordinary shape) is fine —
24.39 vs 24.64 ms, i.e. free and no gain. It is the weight gradient alone.

⚠⚠ **THIS IS §9.1's DEPTHWISE FINDING ON A NEW OP CLASS, AND §14.3 ITEM 2 ASKED FOR EXACTLY THIS
CHECK.** §9.1: *"the depthwise loss is KERNEL SELECTION, not bandwidth or hardware."* §14.3 item 2:
*"Gate 2 proves the operands are bf16; it proves NOTHING about whether cuDNN picked a good kernel
for those shapes."* That check had never actually been run on any net. ViT is where it fired, and
it fired at **13× on a single op**. ▶ **Add it to the recipe** — §19.4.

### 19.2 The gates, and what they did and did not catch

| gate | result |
|---|---|
| **1** — nothing else moved | ✅ `git status verified_mlir/` shows only the new file |
| **entry name** | ✅ `@vitin_adamwxclipdropbf16_train_step`, first render, `#guard`ed against the path |
| **2** — bf16 reached the hardware | ✅ **384 of 387 dot/gemm instructions with all operands bf16** |
| **2b** — the head stayed f32 | ✅ the remaining **3** are the classifier head and its two gradients |
| **2c** — the convolutions | ✅ **2/2 f32, deliberately** — §19.1 |
| **histogram** | ✅ identical across every `stablehlo` op kind, differing only by **864 `convert` = 432 bf16 dot sites × 2** |
| **numeric** | ✅ 627 outputs on identical seeded inputs: worst relative deviation **4.00e-04**, aggregate 3.89e-06, **zero** outputs beyond one bf16 ulp (2⁻⁸ = 3.91e-3) |
| **3** — accuracy | ✅ instantiation, no new theorem (§19.5) |
| a training run | ⛔ not started, as on all seven nets |

⚠⚠ **EVERY ONE OF THOSE GATES WAS ALSO GREEN ON THE 0.52× ARM.** They check that the graph is the
same graph and that bf16 reached the hardware. Neither is a statement about speed, and this is the
first net where that gap cost something rather than being a caveat.

⭐ **A note on the numeric gate, which is new here.** §10.4 step 7's check is the trainer's first
three losses. This one runs both artifacts on identical seeded inputs and compares **all 627
outputs**, so a divergence anywhere in the traversal shows rather than only one that moves the loss.
⚠ 200 outputs contain non-finite entries — a random second-moment state makes the AdamW tail
`rsqrt` a negative — which is the SYNTHETIC INPUT's doing, not the graph's; the check asserts both
arms agree on exactly which entries are finite rather than skipping them silently. The first draft
of that script did skip them silently (`nan > worst` is `False`) and reported a `nan` aggregate,
which is how the hole was found.

### 19.2b ⛔⛔ A THIRD GATE WAS GREEN FROM A STALE BINARY — `vit-fwd-b-tie`, since 2026-08-14

Running the ViT byte tie to confirm gate 1 printed three ✅ lines **from a binary dated 2026-08-03**.
`lake build vit-fwd-b-tie` had exited 1 the whole time:

```
tests/TestViTFwdBTie.lean:72: Application type mismatch: the argument `smooth` … expected ℕ
```

`vitBackAllB` gained a **leading** `vbB` when ViT stopped being batch-32-only (the `128x4`/`256x2`
renders), and this one call site was never updated. So the target has failed to BUILD since, while
`.lake/build/bin/vit-fwd-b-tie` sat on disk reporting byte-identity for a render three weeks old.

▶ **Fixed** (`vitBackAllB 32 10 smooth`), and it now passes fresh: forward, backward, gradient
list and whole train step all byte-identical, which is what actually confirms this section's gate 1.

⚠⚠ **The lesson generalises past ViT: run the BUILD, not the binary.** A tie whose binary is stale
is worse than one that fails, because it prints green. Every `lean_exe` gate in this repo has the
same failure mode, and nothing checks for it. ⭐ Note the shape it shares with §19.1: both are
checks that *look* green and are not measuring what the reader thinks. That is two in one net.

⚠ **And a fourth, found the same way and NOT fixed here**: `tests/AuditAxioms.lean:4010` does
`#print axioms Proofs.dw_sum_pair` on a constant that **exists nowhere in the repo** — it is a
depthwise helper (§12.3) that was renamed or inlined and the audit line was left behind. The audit
reports `error(lean.unknownIdentifier)` and still **exits 0**, so CI would not notice. Left alone
because the right fix is to name the lemma that replaced it, and guessing would be worse than the
dangling line. ▶ Unrelated to ViT; belongs to whoever owns `DepthwiseMixedFloatBridge`.

### 19.3 The six ops, and the two that are built-but-off

| ViT site | op | emit shape | in the shipping render |
|---|---|---|---|
| Q/K/V/O/fc1/fc2, ×12 | `denseRowBf16` | dot: bf16 operands, **f32** result | ✅ |
| their input-VJPs | `denseRowBackBf16` | dot | ✅ |
| their weight grads | `rowDenseWeightGradBBf16` | dot, contracting batch AND token | ✅ |
| SDPA `QKᵀ`/`P·V` + 4 backward | `matmulFBBf16` | batched dot | ✅ |
| patchify stem 16×16/s16 | `patchEmbedBf16` | conv: **bf16-typed** result + convert | ⛔ off — free, no gain |
| its weight grad | `patchEmbedWeightGradBBf16` | conv | ⛔⛔ off — **§19.1** |

⭐ **`matmulFBBf16` is the kit's first ACTIVATION × ACTIVATION bf16 op**, and §10.3's two predictions
about it both held: it really is the new KIND, and the theorem really does come free
(`dot_close_mixed` rounds both operands and never asks which is a weight).

⭐ **§9.2's split held at a sixth and seventh op form.** `dot_general` is unaffected by its result
type; `convolution` is not. Now checked at dense, batched dense, conv stride 1/2/4/16, and grouped
conv. ⚠ The two conv ops are kept, gate-2 correct and unused: they are the right ops in the right
emit shape, and what is wrong is cuDNN's kernel for one of their shapes, which is not something the
renderer can spell its way out of.

### 19.4 ⚠⚠ THE RECIPE GAINS A STEP, and it is the one this net needed

§10.4/§15.3's recipe has carried seven nets. It is missing the check that would have caught §19.1
before a full render:

> **1b. PROFILE the standalone op, not just gate-2 it.** After confirming the operands are bf16,
> time the op against its f32 peer AT THE NET'S OWN SHAPE. A bf16 op that is *slower* than its f32
> peer is not hypothetical: §9.1 measured 0.50× on a depthwise shape and §19.1 measured **0.19×** on
> ViT's stem wgrad. ▶ Cost: minutes. It has now been skipped seven times and been wrong once, and
> the once was worth 32.8 ms on a 29.89 ms step.

⭐ And the per-op-class flag that made §19.1 attributable is worth copying: `bf16Conv` and
`bf16ConvW` are separate from `bf16` and from each other, so a single render can be bisected by op
class instead of guessed at. The name matches the JAX side's own `TrainConfig.bf16Conv`, which has
always been separate for this class of reason.

### 19.5 What ViT did NOT need

* **No new theorem.** `dot_close_mixed` rounds both operands, so the activation × activation case is
  an instance rather than a result. The dot ops' `den` carries **two** roundings and no outer one —
  the f32-typed result means the hardware does not round the output, which is the opposite of every
  conv op in the kit and the thing to get right when copying one.
* **No `patchEmbedBackBf16`.** The stem's input is `%x`; it has no input gradient. ConvNeXt's
  `convStride4` exactly (§16.5), and the reason this net needs six ops and not seven.
* **No new `Raw`/`Tok`.** `matmulFBBf16` rides the generic binary `.batched2` skeleton that
  `addVB`/`subB` already use; the two weight grads ride `.batched`. So `parse_toToks` is untouched.
* **No driver change**, and no variant marker for the two conv flags — they are a measured fact
  about cuDNN, not a recipe choice, and a marker would invite someone to flip them.

### 19.6 ⚠ What is NOT measured

* **A trainer run.** The 1.23× is bare-device (§16.2's rule). No trainer probe and no loss-agreement
  run on real ImageNet has been done, so nothing here says the driver loads this artifact.
* **The DP arm.** Not rendered, deliberately — §13.2 makes a 4-replica number a system result, and
  at 1.23× there is nothing to ship.
* **Anything about convergence**, as on all seven nets.

### 19.7 ⭐ AND §17.3's PROSPECTIVE MEASUREMENT WAS WRONG IN THE FAVOURABLE DIRECTION

§17.3 timed ViT's matmul set in isolation and predicted **1.03×** for the established emit shape,
which is why §17 stopped the build. The artifact measures **1.23×**. ▶ The model was of the right
thing and got the number wrong: in the isolated stack the f32 boundary was paid in full, and in the
real net many of those converts fuse into the LayerNorm/GELU/softmax elementwise ops that sit
between the matmuls. ⚠ So a prospective measurement bought a wrong decision here — but it also cost
almost nothing and the same discipline is what settled the emit shapes (§17.2) correctly. The
lesson is not "stop measuring prospectively", it is that **an isolated op-set model is a lower bound
on the artifact, not an estimate of it**.

---

## 20. ⭐⭐ ViT-Tiny at **1.46×** — and the successor project's premise is REFUTED

Stage 2 of the ViT work. §19 wired the six ops in the established emit shape and got 1.23×. This
section changes **one thing** — the dot's result type — and gets **1.46×**, and then measures that
the scoped `bf16_dtype_ir.md` project would buy **nothing more**.

### 20.1 ⭐⭐⭐ THE FINDING: §9.2 MEASURED THE DOT RESULT TYPE FOR CORRECTNESS AND NEVER FOR SPEED

§9.2's table says `dot_general` reaches the tensor cores with a bf16-typed result **or** an f32 one,
and `convolution` only with a bf16 one. That is exactly right, and it was read as *"the result type
is inert for dot"*. **It is inert for correctness and it is not inert for speed.**

Standalone, ViT-Tiny's own MLP chain (`dot → bias → GELU → dot`, ×12, B=32, D=192, M=768), each arm
gate-2 checked:

| arm | ms | speedup |
|---|---|---|
| f32 | 2.77 | 1.00× |
| `boundary` — bf16 operands, **f32** result (what §19 shipped, and `dotInBf16`'s shape) | 2.35 | 1.18× |
| **`bf16out` — bf16 operands, bf16-TYPED result, convert back** | **1.70** | **1.63×** |
| `elemonly` — f32 result, but the GELU runs bf16 | 1.73 | 1.60× |
| `through` — bf16 result **and** bf16 GELU | 1.72 | 1.61× |

▶ **The whole difference is the dot's result type.** An f32 result makes the gemm write twice the
bytes and take a worse epilogue. ⚠ And it needs **no dtype in the IR** — it is a one-line change in
each op's emit plus an outer `rnd` in its `den`, i.e. `convBf16`'s shape applied to a dot.

**On the artifact**, changing `denseRowBf16`, `denseRowBackBf16` and `matmulFBBf16` to bf16-typed
results (bare device, B = 32, median of 25):

| arm | ms/step | vs f32 |
|---|---|---|
| f32 | 29.91 | — |
| §19, f32-result dots | 24.32 | 1.23× |
| **§20, bf16-result dots** | **20.52** | **1.46×** |

⚠ `rowDenseWeightGradBBf16` **keeps its f32 result deliberately** — its output is the weight
`[a,c]`, 147K elements against the activations' 4.8M, so there is no bandwidth to save and a bf16
store would only cost precision on the optimizer's input. It is the only bf16 dot in the kit shaped
that way, and its constructor says so.

⭐ Gates: histograms identical bar the converts; **384/387 dot/gemm instructions with all operands
bf16**; 2/2 convolutions f32 by design; **627 outputs on identical seeded inputs, worst relative
deviation 4.86e-04** against a 3.91e-3 ulp (§19's was 4.00e-04 — the added bf16 store costs 21 % of
the deviation budget and stays 8× inside it).

⚠ `dotInBf16`'s f32-result shape is now annotated as a PoC artefact rather than a template. It is
rendered by **no** net, and the six convnets carry **zero** bf16 dots (their dense heads and SE
gates are f32 by design), so this finding is ViT-specific in effect — but the next net with a bf16
matmul would have inherited it.

### 20.2 ⛔⛔ AND §5.1's GATE IS MEASURED-WRONG: THE CONVERT COUNT WENT **UP**

`bf16_dtype_ir.md` §5.1 makes the increment's gate *"the op histogram differs … by **fewer**
`stablehlo.convert`"*, and calls a count that does not fall proof that "the round trip is still
there and the increment did nothing".

ViT's convert count went **864 → 1224** and the step got **1.19× faster**. The extra 360 are exactly
one convert-back per bf16-out dot site (72 + 72 + 216), and they are the *point*, not a symptom.

▶ **Replace that gate with the bare-device wall clock plus the numeric check.** Convert COUNT is
not a proxy for convert COST — and in the standalone chains the correlation is actively inverted:
`through` had 132 converts against `boundary`'s 48 and beat it by 1.37×.

### 20.3 ⛔⛔⛔ THE SUCCESSOR PROJECT IS REFUTED BY ITS OWN PRESCRIBED TEST

`bf16_dtype_ir.md` §8 step 3 says to hand-build the bf16-through block before writing any Lean, and
that *"if the hand-built bf16-through block does not beat the bf16-with-boundary block at
ConvNeXt's real shapes, this project is not worth doing and the honest outcome is to write that
down here."* **Run on both nets it was scoped for, it does not beat it.**

**ConvNeXt-T block interior** — `conv1×1 → GELU → conv1×1`, ×3, at stage-1 shapes
(c=96, 4c=384, 56², B=32), which is §8 step 3 verbatim and §5's named first increment:

| arm | ms | speedup | converts |
|---|---|---|---|
| f32 | 8.42 | 1.00× | 0 |
| `bf16out` — **what ConvNeXt already ships** | **4.76** | **1.77×** | 14 |
| `through` — the project's proposal, no f32 in the middle | 4.79 | 1.76× | 26 |

**ViT pass-through chain** — `dot → slice → transpose → dot`, ×36:

| arm | ms | speedup | converts |
|---|---|---|---|
| f32 | 2.34 | 1.00× | 0 |
| `bf16out` — what §20.1 now ships | **1.58** | **1.48×** | 73 |
| `through` — slice and transpose typed bf16 | 1.59 | 1.47× | 73 |

▶ **XLA already propagates bf16 across the intervening ops once the PRODUCING op writes bf16.** In
the pass-through case the two arms compile to *the same 73 converts*. Spelling the dtype explicitly
in the emitter buys nothing because the emitter was never what was forcing the boundary — the
**f32 result type** was, and that is fixed in §20.1 without any type-system work at all.

⚠⚠ **WHAT THIS DOES AND DOES NOT REFUTE.** It refutes the *route*: a `dt : Dtype` field plus a
dtype-carrying `emitTok` stack (Route B) is measured at ~0 % on both nets it was scoped for, so the
196-constructor Route A is a fortiori not worth it. It does **not** refute §16.3's measurement that
ConvNeXt's conv work is 2.70× with converts free and 1.68× with an f32 boundary forced — that
experiment forced barriers over the whole 173-conv set and mine is 6 pointwise convs at one stage's
shapes. ▶ The reconciliation worth having is: **which of §16.3's 519 converts are actually costing
anything**, measured per-convert rather than in aggregate. That is a profiling question now, not a
type-system one.

⭐ **And the cheap thing to do instead is already named**: §18's first item, profiling each net's
bf16 ops against their f32 peers (§19.1's method). ViT's stem wgrad was 0.19× and its dots were
leaving 1.19× on the table — both found by measurement, neither by any gate, and neither needing a
line of type-system work.

---

## 21. ⭐⭐ ALL SEVEN NETS AT 4 GPUs ON REAL IMAGENET — and what a full run would cost

Measured 2026-08-25. `CUDA_VISIBLE_DEVICES=0,2,3,4` (the AER-clean four, per `scripts/jobs/*.conf`),
`PJRT_REPLICAS=4`, `PJRT_FFI_RESIDENT=1`, `SHIM_WORKERS=8`, real ImageNet, `LEAN_MLIR_MAX_STEPS=40`
(median of steps 9..40), `LEAN_MLIR_CKPT_TAG` set so no finished checkpoint short-circuits the probe.

| net | steps/epoch | epochs | f32 ms | bf16 ms | speedup | f32 end-to-end | bf16 end-to-end |
|---|---|---|---|---|---|---|---|
| ResNet-34 | 5,004 | 90 | 225 | 164 | 1.37× | 29.1 h | **21.5 h** |
| ResNet-50 | 5,004 | 100 | 359 | 233 | **1.54×** | 50.9 h | **33.4 h** |
| MobileNetV2 | 5,004 | 350 | 195 | 129 | **1.51×** | 98.5 h | **66.4 h** |
| MobileNetV4-M | 5,004 | 100 | 177 | 126 | 1.40× | 25.6 h | **18.6 h** |
| EfficientNet-B0 | 5,004 | 350 | 186 | 181 | ⛔ 1.03× | 94.1 h | 91.7 h |
| ConvNeXt-T | 10,009 | 300 | 217 | 184 | 1.18× | 184.1 h | **156.6 h** |
| ViT-Tiny | 2,502 | 300 | 232 | 163 | 1.42× | 51.5 h | **37.1 h** |
| **all seven** | | | | | **1.26×** | **533.9 h (22.2 d)** | **425.3 h (17.7 d)** |

▶ **bf16 saves 4.5 days of continuous 4-GPU compute across the seven, 20 %.**

### 21.1 How the end-to-end numbers are built, and what they do NOT include

`steps/epoch × epochs × ms/step`, plus a measured **37.5 s/epoch** for eval and the checkpoint.
Steps/epoch is the driver's own figure, read off its `step 0/N` line, not derived. Epochs are each
net's `*ImagenetConfig.epochs` (⚠ three of the `scripts/jobs/*.conf` files run a SHORTER tier —
R34 30, B0 80, MNv2 300 — so those confs are cheaper than the table by the obvious ratio).

⭐ **The 30 GB val drain is ONE-TIME, not per-epoch** — the driver says so (`draining the val split
into RAM (~30 GB, one time)`) and the measurement confirms it: 1 epoch of 20 steps + eval was 142 s,
3 epochs 226 s, so the marginal epoch is 42 s of which 4.5 s is the 20 train steps. ⚠ Eval also runs
on **1 replica** (`@<slug>_fwd_eval`, 1 replica in the log), so it does not scale with the four.

⚠ **NOT included, and all three push the real wall-clock UP:**
* **Thermal governance.** The confs run under `supervise.sh` with `TEMP_MAX=78 / TEMP_RESUME=62`;
  a sustained multi-day run rests. Nothing here models that, and these probes ran from a 30 °C idle.
* **The 37.5 s/epoch eval figure is R34's.** ConvNeXt's and ViT's forwards are heavier; their eval
  cost is larger and was not measured separately.
* **Restarts.** `supervise.sh` exists because these runs get reaped and resumed.

### 21.2 ⚠ Three nets needed a 4-replica bf16 render that did not exist

§13.1/§14/§19 rendered MNv4, B0 and ViT single-device only. Added here so all seven can be costed at
the same geometry: `efficientnetin_rmsdp64bf16`, `vitin_adamdp128x4wxclipdropbf16`, and — with a
caveat — `mnv4in_adamdp64` / `mnv4in_adamdp64bf16`.

✅ **MNv4's DP pair carried §13.1's caveat and it was LIFTED 2026-08-27.** That section declined a
DP render because *nothing has tied MNv4's collectives*, and named its own release condition. Both
halves of the tie now exist and are green: `tests/TestMnv4DpCheck.lean` (duplicated batch — fp32
`bnstat` bit-exact 67,904/67,904 and gradient 8.45e-7; **bf16 bit-exact on all 9,715,512 floats**)
and the `mnv4in` row in `tests/TestShardCheck.lean` (asymmetric batch — TEST 1.10e-6 against a
CONTROL of 2.00). Both go red on a sum-not-mean render, the shard TEST landing on exactly
`3.000000 = |4g − g| / |g|`. ▶ `runs/2026-08-27-mnv4-dp-shard-gates/`.
⭐ So MNv4's row in the table above is no longer costing-only, and `scripts/jobs/mnv4-default-4gpu.conf`
trains off it. ⚠ Four GPUs, forced: `adamdp64` is 4-replica only, with no 2-replica peer.

⚠ A 4-replica ms/step is a **system** result in all seven rows (§13.2) — shim feed and f32
all-reduce included. The RENDERER numbers are the single-device/bare-device ones in §14, §16 and §20.

### 21.3 ⭐ What the 4-replica figures say that the single-device ones did not

* **R50 reproduces exactly** — 359 → 233 = 1.54× against §10.1's 360 → 232 = 1.55×.
* **MNv2 is 1.51× here against §12.1's 1.37×**, same geometry. The difference is `SHIM_WORKERS`:
  §12's probe did not set it and this one uses 8. ▶ That is §13.2's "check `SHIM_WORKERS` before
  ever blaming the emit", now costing 0.14× on a committed number.
* **ViT holds up best under DP** — 1.42× at 4 replicas against 1.46× on the bare device, i.e. it
  loses almost nothing to the system. It is also the only net whose global batch is 512, so it runs
  the fewest steps per epoch of the seven.
* **B0 is 1.03× at 4 replicas against 1.09× single-device** — the worst result on the branch gets
  worse with replicas, and §14.3 is still open. ▶ Of the 108.7 h bf16 saves across the seven, B0
  contributes 2.4 h.
* ⛔ **ConvNeXt is the most expensive net by a wide margin** — 184 h f32, 157 h bf16 — because it
  is the only one at global batch 128, so it runs **10,009 steps/epoch for 300 epochs**. Its ms/step
  is mid-pack; its step COUNT is what costs.
  ⛔⛔ **AND A BIGGER BATCH DOES NOT FIX IT — §21.5.** An earlier draft said "the cheapest available
  speedup for ConvNeXt is a bigger batch, and `cBS` being a private constant is what stands in the
  way". The conclusion is right and the reason given was not: batch 64 **would** fit (T uses 42 % of
  the BFC budget, not the ~100 % `nvidia-smi` appeared to show), and it still would not pay, because
  the block interior's ms per IMAGE is flat from batch 8 to 128. ▶ §21.5 has the corrected numbers
  and says which of its own arguments are withdrawn.

### 21.5 ⛔⛔ "CAN WE JUST RAISE ConvNeXt's BATCH?" — NO, BUT NOT FOR THE REASON FIRST GIVEN

⚠⚠ **THIS SECTION'S FIRST TWO ARGUMENTS WERE MEASURED WITH THE WRONG TOOL AND ARE WITHDRAWN.**
They read peak memory off `nvidia-smi --query-gpu=memory.used`, which reports XLA's **preallocated
BFC pool** (~73 % of the card, ~11.68 GiB of 16,380 MiB) and not the graph. Every artifact looks
like ~12 GB through that lens. The corrected numbers, from XLA's own
`peak_memory_in_bytes` (`scripts/bf16_peak_memory.py`), invert both:

| artifact (bs32, 1 device) | peak | args | temp | of the 11.68 GiB budget |
|---|---|---|---|---|
| ConvNeXt-T f32 | 4.88 GiB | 0.34 | 4.24 | 42 % |
| ConvNeXt-T bf16 | **4.24 GiB** | 0.34 | 3.58 | **36 %** |
| ConvNeXt-B f32 | 10.93 GiB | 1.01 | 8.95 | **94 %** |
| ConvNeXt-B bf16 | **9.75 GiB** | 1.01 | 7.75 | 83 % |

* ⛔ **"bf16 costs 200 MiB rather than saving any" — WRONG. bf16 SAVES 13 % on T and 11 % on B.**
  The reasoning behind the wrong claim was not silly (this emit does convert back to f32 after every
  op, so the f32 activation really is still alive) — it was just not what the number said, and the
  number was not measuring the graph.
* ⛔ **"Only ~4 GB free, largest batch ~40" — WRONG.** T uses 42 % of the budget at bs32. Batch 64
  extrapolates to 8.82 GiB f32 / 7.50 GiB bf16 — **76 % / 64 %, comfortably fitting.**

✅ **The third argument stands, was measured correctly, and is the one that actually decides it:
the compute is ALREADY SATURATED at batch 32.** ConvNeXt's own stage-1 block interior, ms per IMAGE:

| B | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| f32 | 263.5 | 268.6 | **264.5** | 257.0 | 256.9 |
| bf16 | 88.8 | 147.4 | **148.5** | 144.5 | 142.7 |

▶ Flat. 32 → 128 buys **1.03×**. Doubling the batch halves the step count and doubles the step, and
they cancel. The only thing a bigger batch amortises is the FIXED per-step overhead — the all_reduce
(parameter-sized, batch-independent) and the launch, part of the ~59 ms between the 217 ms trainer
step and the 157.8 ms bare-device one, and the shim feed inside that gap scales with batch anyway.

⭐ **So the conclusion is unchanged and the reasoning is not**: raising ConvNeXt-T's batch would FIT
(memory was never the constraint) and would still not pay, because the GPU has no idle throughput to
sell. The `bB`/`cBS` refactor is still not ConvNeXt-T's lever — but "it would not fit" was never a
true reason to say so.

▶ **What drives ConvNeXt-T's 157 h is the recipe against the hardware**: a 300-epoch schedule at a
global batch of 128 rather than the paper's 4096, i.e. ~32× the paper's step count.

### 21.5b ✅ ConvNeXt-**S** — the twin that was missing, and it is the FASTEST of the three

`convnextsin` carried the fp32 half of the square and not the bf16 half, where T and B carried
both. ⚠ **An accident of ordering, not a decision**: B landed after S and brought bf16 with it.
Rendered 2026-08-27 (`verified_side_quest_counterparts.md` §4c) — two `#eval`s, no new operator,
because S is T's depth table at T's widths.

Measured with `scripts/bf16_device_step.py`, one session, one 4060 Ti, bs32, `adamwxclipdrop`
against its bf16 twin:

| model | fp32 | bf16 | speedup |
|---|---|---|---|
| ConvNeXt-T | 157.85 ms | 121.12 ms | 1.30× |
| **ConvNeXt-S** | **268.35 ms** | **192.65 ms** | **1.39×** |
| ConvNeXt-B | 395.73 ms | 294.25 ms | 1.34× |

⭐⭐ **T AND B REPRODUCE THEIR COMMITTED FIGURES, WHICH IS WHAT LICENSES S's ROW.** §21.6's table
has T at 157.8 → 121.0 and B at 396.0 → 293.3; this session returns 157.85 → 121.12 and
395.73 → 294.25. Two independent controls landing on the record to within 0.3% is the difference
between "S is 1.39×" and "S measured 1.39× once".

⭐ **S is the fastest of the three, which "deeper costs more" does not predict.** Depth adds blocks
at widths bf16 already suits; B's widening moves every stage. Worth not smoothing over — the plan
that scheduled this work expected ~1.29× for S, on the reasoning that T's ratio would carry.

⚠ **Two rows of §21.6's shape are NOT measured for S**: peak memory (`bf16_peak_memory.py`) and the
4×bs32 trainer figure on real ImageNet. The trainer number dilutes the graph's — T reads 1.30×
here and 1.18× there, B 1.34× and 1.20× — so do not quote 1.39× as a wall-clock saving.

### 21.5c ✅ ViT-**S** and ViT-**B** — and S is the largest bf16 win in the repo

`vitsin` and `vitbin` carried fp32 only. Rendered 2026-08-27 (§4c) — one `(bf16 := true)` each,
no new operator, because S and B are Tiny WIDENED and every bf16 op they need is one Tiny already
instantiates at a narrower shape.

⚠⚠ `bf16Conv`/`bf16ConvW` stay FALSE, inherited from Tiny's bf16 render and load-bearing: §19.1
measured this net's stem weight gradient at **0.19×** its f32 peer (a 209×209 window with no bf16
cuDNN kernel), and the width axis does not touch the stem. The one op where bf16 is a LOSS on this
architecture is out of the emit at every size, which is why a green gate here means what it usually
means.

⭐⭐ **These are 4-REPLICA measurements**, because ViT-S and ViT-B have no single-device render —
`scripts/bf16_device_step.py` grew a `--replicas` flag for exactly this (see below). All-reduce
included, four 4060 Ti:

| model | per-dev bs | global | steps/epoch | fp32 | bf16 | speedup |
|---|---|---|---|---|---|---|
| ViT-Tiny | 128 | 512 | 2,502 | 171.6 ms | 96.4 ms | 1.78× |
| **ViT-S** | 128 | 512 | 2,502 | **464.0 ms** | **245.5 ms** | **1.89×** |
| ViT-B | 32 | 128 | 10,009 | 383.8 ms | 273.7 ms | 1.40× |

⭐ **ViT-S's 1.89× is the largest bf16 speedup measured anywhere in this repo**, ahead of R50's
1.55×. The shape of it: Tiny is narrow enough (`D = 192`) that its matmuls underuse the tensor
cores even in bf16, S's `D = 384` suits them better, and B pays a **346 MB f32 all-reduce** every
step (86.6M parameters) at only bs32 — a fixed cost in both arms, which is what pulls its ratio
back down.

⚠ **ViT-T's 4-replica ratio (1.78×) is not its committed 1.42×** and neither is wrong: 1.42× is the
TRAINER's, at 232 → 163 ms. Trainer/device is 1.352 in fp32 here, against ConvNeXt's 1.362 —
which is the cross-check that says the new `--replicas` path measures the same thing the old one
does at a different device count.

⛔ **ViT-B GETS NO PROJECTED WALL CLOCK, deliberately.** T is the only ViT with a measured trainer
figure, and B runs at ¼ T's per-device batch, so T's overhead multiplier does not transfer:
applied to B it returns a 1.12× trainer speedup, i.e. BELOW B's own 1.40× device ratio, which no
overhead model may do. An additive model scaled by images/step gives 1.37× and a total 100 h
lower. Two defensible models 100 h apart is not an estimate. ▶ ViT-S is at T's batch, so its
projection stands.

### 21.6 ⭐⭐ ConvNeXt-**B** — where the memory question is real, and where bf16 buys headroom

ConvNeXt-B (88.59M parameters, 3.1× Tiny) is the first size at which fitting is a question rather
than a formality, and the first where bf16's memory saving is load-bearing rather than incidental.

| | ConvNeXt-T | ConvNeXt-B |
|---|---|---|
| parameters | 28.6 M | **88.6 M** |
| peak memory, bs32 f32 | 4.88 GiB (42 %) | **10.93 GiB (94 %)** |
| peak memory, bs32 bf16 | 4.24 GiB (36 %) | **9.75 GiB (83 %)** |
| bare device, 1 GPU bs32 | 157.8 → 121.0 ms = **1.30×** | 396.0 → 293.3 ms = **1.35×** |
| trainer, 4×bs32 real ImageNet | 217 → 184 ms = 1.18× | **534 → 446 ms = 1.20×** |
| steps/epoch | 10,009 | 10,009 |
| **end-to-end, 300 epochs** | 184.1 h → **156.6 h** | **448.5 h (18.7 d) → 375.1 h (15.6 d)** |

⭐ **bf16 saves ConvNeXt-B 73.4 hours — three full days — the largest single saving on this branch**,
and it is also the net where bf16 is *fastest* on the bare device (1.35×, beating T's 1.30× and every
other conv net except R50's 1.55×).

⚠ **B's f32 arm is at 94 % of the BFC budget at bs32 and it DID run at 4 replicas** — no
`RESOURCE_EXHAUSTED`, 534 ms/step, 40 steps clean. So this is not "f32 does not fit". It is that it
fits with ~0.7 GiB spare, before the shim's prefetch buffers and any allocator fragmentation, and
`vit_convnext_sb_scaleup.md` records the JAX side OOMing at exactly that kind of margin
(ViT-B at 11.41 of 11.68 GiB). ▶ **bf16's 83 % is the difference between "ran once" and "has room".**

▶ **And it makes the batch question different for B than for T.** T at bs64 would fit and not pay.
B at bs64 f32 extrapolates past the budget and would need bf16 to fit at all — so if a bigger batch
is ever wanted at this size, bf16 is a prerequisite rather than an optimisation. ⚠ Whether it would
PAY at B is unmeasured: §21.5's flat-ms/image curve is Tiny's stage-1 shapes, and B is 1.33× wider
at every stage. That check is one run of `scripts/bf16_boundary_probe.py` at `cnxBase`'s dims.

⚠ **NOTHING HAS BEEN TRAINED ON ConvNeXt-B**, in either arm — the app docstring's warning still
stands and these are shape-and-speed results only.

### 21.7 ⛔ A THIRD STALE BINARY, found by this probe

`vit-imagenet-verified` was dated 2026-08-12 and failed both ViT probes outright:
`imagenet shim: wire version 3, expected 1 (nclasses=0 ⇒ v1)` — the shim wire format had been
bumped and the binary never rebuilt. A `lake build` fixed it in 1 s and both arms then ran.

▶ That is the **third** stale-artifact defect in two days: `vit-fwd-b-tie` printing green from an
Aug-3 binary (§19.2b), `AuditAxioms.lean` erroring on a missing constant and exiting 0 (§19.2b), and
now this. ⭐ Unlike the other two this one FAILED LOUDLY, which is why it cost a minute rather than a
wrong conclusion. **The pattern is worth a gate**: nothing in this repo checks that a committed
`lean_exe` still builds before its output is trusted.

---

## 18. Still open, in rough priority order

* **R34/R50 bare-device timings** — §16.4. Two runs of one script, no render and no proof, and
  they close the last unmeasured inference about the four-replica numbers.
* **EfficientNet-B0's 1.09×** — §14.3, confirmed on device by §16.4 and still unexplained. Now
  approachable with §16.3's method, which did not exist when §14.3 was written: attribute B0's
  device step per op kind the way ConvNeXt's is attributed, and the answer falls out.
* **ConvNeXt's boundary converts** — §16.3 puts the prize at 1.30× → ~1.40× and says it needs no
  new ops. The first net where this lever is measured rather than argued.
* ⭐⭐ **PROFILE THE OTHER SIX NETS' bf16 OPS AGAINST THEIR f32 PEERS — new, and §19.1 is why.**
  ViT's stem weight gradient is **0.19×** its f32 peer with every gate green, and that check has
  never been run on any net on this branch. §9.1 found a 0.50× depthwise shape and §14.3 item 2
  asked for exactly this on B0 and never got it. ▶ **It is also the most likely explanation still
  standing for B0's 1.09×**, and it is cheap: time each op standalone at the net's own shapes.
* ⭐ **ViT IS BUILT — §19/§20**, at **1.46×** on the bare device with its two convolutions
  deliberately left f32 and its dots writing bf16-typed results.
* ✅ **A trainer probe on ViT's bf16 arm — DONE (§21)**: 232 → 163 ms/step at 4×bs128 on real
  ImageNet, so the driver does load it. ⭐ Its DP arm keeps 1.42× against the bare device's 1.46×,
  the smallest system loss of the seven.
* ⭐ **A gate that a committed `lean_exe` still BUILDS.** Three stale-artifact defects in two days
  (§19.2b ×2, §21.7), one of which printed green for three weeks. Nothing checks this.
* ⛔⛔ **The boundary converts — SCOPED, THEN CANCELLED BY ITS OWN TEST (§20.3).** Do not start
  `planning/bf16_dtype_ir.md`. ⚠ Its §5.1 gate ("the convert count must FALL") is separately
  measured-wrong: ViT's count went 864 → 1224 and the step got 1.19× faster (§20.2).
  ▶ **What remains genuinely open** is narrower and is a profiling question: §16.3 measured
  ConvNeXt's conv work at 2.70× with converts free and 1.68× with an f32 boundary forced, over its
  whole 173-conv set. §20.3's 6-conv block-interior probe does not reproduce that gap. Find out
  **which** of the artifact's 519 converts cost anything, per-convert, before concluding either way.
* ⭐⭐ **CIFAR IN bf16 AND fp8 — ✅ SCOPED 2026-08-25 in `planning/cifar_lowprec_stability.md`.**
  The cheapest answer to the item below: CIFAR converges in HOURS where the seven need 18.6 h to
  15.6 d (§21), and cifar8's 23 convs + 9 dots all already have bf16 twins, so it needs no new ops.
  ⭐ It also closes `fp8_lowering.md`'s blocking gate, which this session opened by measurement —
  XLA lowers f8E4M3FN to real fp8 silicon at 2.71× f32, where IREE could not lower it at all.
* **A full training run** on any of the seven. Nothing has been trained to convergence in bf16, and
  every accuracy claim on this branch is still a three-step loss comparison — except ViT's, which is
  a 627-output comparison on identical seeded inputs (§19.2) and is stronger but still not training.
