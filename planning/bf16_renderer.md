# bf16_renderer.md — bf16 on the verified render path, as a ladder from mnist-linear

Scoped 2026-08-01 on ares (6× RTX 4060 Ti, CUDA 12.9).

---

## ⭐ STATUS 2026-08-24 — ResNet-34 IS DONE AND RUNNING; four claims below are REFUTED

Branch `bf16/verified-conv-ops`. **The verified R34 trains on ImageNet in bf16 at 1.41× end to
end** (222 → 157 ms/step, 4×bs64 on four 4060 Ti), with losses tracking the f32 arm step-for-step
to the 3rd–4th decimal — and **R50 now does too, at 1.55×** (360 → 232 ms/step, same box, same
4×bs64, both arms measured back to back in one session).

| what | state |
|---|---|
| 7 bf16 conv ops (per-example fwd, batched fwd ×2, dgrad ×2, wgrad ×2) | ✅ `StableHLO.lean` |
| `flatConvFBf16_faithful` / `_id` | ✅ the op adds rounding and nothing else |
| `conv_close_mixed` | ✅ `Proofs/Float/ConvMixedFloatBridge.lean` |
| R34 render wired + `resnet34in_momdp64bf16_train_step.mlir` | ✅ gate 1 and gate 2 green |
| R50 render wired + `resnet50in_momdp64bf16_train_step.mlir` | ✅ gate 1 and gate 2 green, **1.55×** |
| MNv2, B0, MNv4, ViT, ConvNeXt | ⛔ not started — §10 is the plan |
| whole-NET error bound; the 90-epoch run | ⛔ not started |

⚠ **Read §9 before trusting anything below it.** Four of this document's load-bearing claims
were refuted by measurement on 2026-08-24. The reasoning is kept; the conclusions moved.

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

⚠ It bounds ONE conv against exact ℝ. A whole-net bound needs `FloatComposeBridge` and is not
started.

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

⚠ Still not done for R50, exactly as for R34: the whole-NET error bound (`FloatComposeBridge`) and
a full training run. The 40-step probe says the graph is right and fast; it does not say what it
converges to.

### 10.2 MobileNetV2 / EfficientNet-B0 / MNv4 — new op KIND, biggest payoff

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
