# bf16_renderer.md — bf16 on the verified render path, as a ladder from mnist-linear

Scoped 2026-08-01 on ares (6× RTX 4060 Ti, CUDA 12.9). **NOT started.**

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

Point 3 independently confirms a decision already in the code: `LeanMlir/Types.lean` on `bf16Conv`
says *"Depthwise/separable convs (MobileNet/EfficientNet) still stay fp32."* That was asserted
without a number; it now has one, on CUDA. (The AMD note that bf16 conv is slower on MIOpen is a
*separate* effect — this is depthwise-specific and it is on NVIDIA.)

**Consequence: prioritise R34/dense/ViT, and do NOT expect the depthwise nets to pay.** Any bf16
renderer must keep depthwise on the fp32 path, exactly as the JAX side already does.

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

## Sequencing against the other open lever

Device-resident parameters (§2d.3 in `xla_pjrt_handoff.md`) is fully scoped, has its gate written,
and is worth ~20 points of DP efficiency at 4 replicas. bf16 is worth up to 1.88× but needs a new
accuracy theorem to reach the convnets. **They are independent** — residency is transport, bf16 is
arithmetic — so either order works, and neither blocks the other.

If the goal is "match the JAX reference on R34/ImageNet", note it needs **both**: the JAX 4× number
is bf16 *and* device-resident, and this session measured the verified path missing both.
