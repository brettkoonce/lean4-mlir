# xla_pjrt_handoff.md — where the XLA/PJRT work stands, and what to do next

**Written 2026-07-27.** Handoff for a fresh session. The full history, gate
definitions, and every measurement live in `planning/xla_pjrt_ladder.md`; this
file is the short version — state, next moves, and the things that cost time to
learn the first time.

Branch **`xla-pjrt-backend`**, four commits on top of `cfbdccd`:

| commit | what |
|---|---|
| `b44caaa` | PJRT C-API backend; ladder rungs 0–3 |
| `20dfd29` | emit cross-replica `all_reduce` for data-parallel AdamW |
| `9a7957a` | `String.dropEnd` deprecation fix |
| `b5b843e` | N-replica execute; data-parallel validated against 1×N |

---

## 1. What works today

**A second trusted lowerer, with no Python at run time.** `ffi/pjrt_ffi.c`
implements the same C surface as `ffi/iree_ffi.c` (symbol-identical under
`nm -D`), so nothing above the shim changed; the backend is whichever `.so` is
linked. Backend detection is a **weak symbol**, so a binary cannot disagree with
the library it linked.

**Rungs 0–3 complete** — linear, MLP, CNN, cifar8-bn+Adam, ResNet-34/Imagenette
(513 in / 513 out, 36 BN layers). **No rung ever needed new shim code**: BN
running stats and rank-0 scalars all ride `iree_ffi_invoke_f32`. The
`train_step_adam*` family in `iree_ffi.h` is still stubbed and was never reached.

**Data-parallel multi-GPU**, validated on cifar8 (no BN) where the
batch-decomposition identity is exact: 1×256 vs 2×128+all_reduce agree on the
gradient to **1.015e-06**.

**Speed, R34/Imagenette bs32:** IREE 1702 → XLA **162 ms/step** (10.5×);
52.5 s/epoch. Within **1.04×** of hand-written JAX per step at bs32, but see §3.

### Running it

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-verified-adam-xla data

# multi-GPU: render at REPLICAS=N first (tests/TestResnet34Train.lean), then
unset HIP_VISIBLE_DEVICES
LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 .lake/build/bin/resnet34-verified-adam-xla data
```

Env knobs added by this work: `PJRT_REPLICAS`, `PJRT_PLUGIN`, `PJRT_FFI_TRACE`,
`LEAN_MLIR_REPLICAS`, `LEAN_MLIR_SKIP_EVAL`, `LEAN_MLIR_G2_STEPS`,
`LEAN_MLIR_DUMP_PARAMS`, `LEAN_MLIR_PERTURB_R`.

---

## 2. Next up

### 2a. Move the imagenette `_fwd` renders from `tests/` into `Proofs/` — R34 ✅ DONE, and it found a real bug

**Status 2026-07-27 (later).** `resnet34_fwd` is moved. Doing it surfaced that the two writers
were not two spellings of one function — **they were two different functions**, and the SGD
trainer had a train/eval skew:

| artifact | writer | BN |
|---|---|---|
| `resnet34_train_step.mlir` | `Proofs/…/ResNet34Render.lean` (certified) | **per-example**, reduce `[2,3]`, n = H·W = 12544 |
| `resnet34_fwd.mlir` *(was)* | `tests/TestResnet34Fwd.lean` (hand-written) | **batch**, reduce `[0,2,3]`, n = B·H·W = 401408 |
| `resnet34_adam_train_step.mlir` | `tests/TestResnet34Train.lean` | batch, 401408 |
| `resnet34_fwd_eval.mlir` | `tests/TestResnet34Fwd.lean` | affine, running stats (correct — see §2a-bis) |

So `resnet34-verified` (SGD) trained a per-channel **per-example** net and scored it with **batch**
statistics. Measured: on one shared (θ, x) the two forwards disagree at **rel 1.13** on
non-degenerate logits (`|logit|max` 2.86) — not a rounding difference, a different function.

Worse, `resnet34_train_step.mlir` *itself* has the same split across its two writers: the
`tests/` writer renders it at 401408 (batch), the `Proofs/` writer at 12544 (per-example).
Whichever ran last decided what the trainer optimised. That is what had clobbered the working
tree at the start of this session.

**What landed:**

- `resnet34FwdFaithfulV` in `LeanMlir/Proofs/Codegen/ResNet34Render.lean`, plus a shared
  `r34FwdChain` / `r34SigList` that the forward *and* the train step both render from. The
  train-step artifact came back **byte-identical to the committed one**, which is what proves the
  refactor changed nothing it shouldn't.
- `verified_mlir/resnet34_fwd.mlir` is now a **byte-identical 1106-line prefix** of
  `verified_mlir/resnet34_train_step.mlir`, ending exactly where the loss cotangent begins. Eval
  is now literally the forward the trainer differentiates.
- `BFwd` carries its own `xin`, and the backward takes the block record instead of a separately
  passed input name — the forward/backward pairing can no longer be miswired at the call site.
- `tests/TestResnet34Fwd.lean` no longer writes `resnet34_fwd.mlir` (and is deleted outright in
  §2a-bis, once `_fwd_eval` moved too).
- `scripts/regen_verified_mlir.sh` — the missing canonical regeneration entry point, with two
  audits: duplicate writers, and *forward ⊂ train-step prefix*.
- `lean_exe resnet34-fwd-tie` (`tests/TestResnet34FwdTie.lean`) — feeds two renders the same
  (θ, x) and compares logits, refusing a degenerate all-zero/non-finite agreement.

**Open decision for the reader:** R34-SGD now normalises per-example everywhere. That is the
certified semantics (`bnPerChannelTensor3`, tied in `SpecVJP` to `resnet34Forward_full_pc`), and
per-channel-per-example BN is why train == eval with no running stats. But it is *not* textbook
ResNet batch-norm, and any published `resnet34-verified` accuracy came from the skewed pairing.
Re-measure before quoting — this folds naturally into the verified-path table re-run.

**Still `tests/`-rendered, in the original §2a scope:** `mobilenetv2_fwd{,_eval}`,
`efficientnet_fwd{,_eval}`, `convnext_fwd`, `resnet34_fwd_eval`. `vit_fwd`/`vit_train_step`/
`cifar8_train_step` also have two writers, but those `tests/` files *delegate* to the `Proofs/`
renderer — verified byte-identical for `vit_fwd`, so they are redundant, not divergent. The four
that own hand-written emitters (convnext, efficientnet, mobilenetv2, resnet34 train steps) are
the ones that can drift.

### 2a-bis. `resnet34_fwd_eval` — new proof-kit op ✅ DONE

The kit had no running-stats/affine BN, only the batch-statistic `bnPerChannelF`. Added
**`bnPerChannelEvalF`** — the frozen-statistics BN, `γ·(x−μ)·rsqrt(var+ε)+β` with μ/var arriving
as graph inputs. No reduction, so it is pointwise in the activation; that is the formal content of
"eval is class-batch-independent". Forward-only by design — eval has no backward.

Math (`Proofs/Foundation/PerChannelBN.lean`) mirrors the training chain one-for-one:
`bnEvalForward → bnPerChannelEvalMat → bnPerChannelEvalFlat → bnPerChannelEvalTensor3`, through
the same `reassoc` bridge. Being affine in `x`, it is differentiable with **no `0 < ε`
hypothesis** — ε only enters a constant scale factor.

Wired at all eight sites: `SHlo` constructor, `den`, the `rfl` faithfulness theorem, `Raw`,
`skel`, `Tok`, `toToks`, `emitTok`. `r34FwdChain` now takes an `R34Bn` mode (`.train` / `.eval`),
so all three renders come from one chain.

Result — `verified_mlir/resnet34_fwd_eval.mlir` now renders as `pretty(provenGraph)`:

- 219 inputs, **arg types identical in order** to the retired render, all 72 stat names identical
  (only the stem bias moves `%sb`→`%sbi`, positional).
- **Ties the retired hand-written render EXACTLY — max rel diff 0.0** on non-degenerate logits
  (`resnet34-fwd-tie --eval`). Unlike `_fwd`, this one was always the right function; it is now
  certified rather than trusted. So this change is behaviour-preserving for the AdamW drivers.
- `resnet34_train_step.mlir` and `resnet34_fwd.mlir` both came back **byte-identical** after the
  BN-mode refactor.
- `tests/TestResnet34Fwd.lean` is deleted — with both artifacts moved it wrote nothing.

**Gotcha this cost time:** `LeanMlir/Proofs/Codegen/StableHLOParse.lean` is under the **`Certs`**
lib, *not* the default target. A plain `lake build` passes while the R4 `roundtrip` theorem
(`parse (toToks (skel a)) = some (skel a)`, quantified over **all** `SHlo`) is missing a case for
your new op. Adding an `SHlo` constructor means **`lake build Certs`** as well — two edits there:
a `parseStack` case and a `parse_toToks` induction case.

**Still open:** `@resnet34_fwd_eval` is the eval partner of `resnet34_adam_train_step.mlir`, which
is *still* the hand-written batch-BN render in `tests/TestResnet34Train.lean`. The eval forward is
now certified while the train step it partners is not — that asymmetry is the Adam half of §2a
(`emitAdamV` in `LeanMlir/ViTRender.lean`), untouched.

**Still `tests/`-rendered:** `mobilenetv2_fwd{,_eval}`, `efficientnet_fwd{,_eval}`,
`convnext_fwd`. Same recipe applies; `bnPerChannelEvalF` now exists for their `_fwd_eval` halves.

### 2a-ter. The Adam half — optimizer ops ✅ DONE, whole-net renders still open

§2a's finding was that *"the dividing line is Adam, not the net"*: `emitAdamV` lives in
`LeanMlir/ViTRender.lean` as a hand-written String emitter, is only ever called from `tests/`, and
so every `_adam_train_step.mlir` for every net sat outside the proven kit. The structural reason
turned out to be that **every param op in the kit fuses gradient and update** (`convWeightSgd`,
`weightSgd`, `bnGammaSgd`, …, all `θ − lr·g`). Adam needs the gradient *itself*, three times over
(θ', m', v'), and there were no gradient-only ops and no optimizer ops at all.

**Added — 7 ops.** Four param gradients un-fused from their SGD peers (`weightGrad`, `biasGrad`,
`convWeightGrad`, `convBiasGrad`), and three shape-generic AdamW ops (`adamMNextF`, `adamVNextF`,
`adamWParamF`) whose child expression is the gradient and which carry θ/m/v as name+value fields,
exactly as the `*Sgd` ops carry their param. Three ops rather than one because `SHlo` is
single-result while an AdamW step produces the triple `Proofs.adamWStep` returns.

Both halves of the trust boundary are now closed, and they were closed by *different* means:

- **Denotation** — `adamW_triple_faithful` : `(den adamWParamF, den adamMNextF, den adamVNextF)
  = Proofs.adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v (den e)`, by `rfl`. The ℝ optimizer and its
  invariants (`adamVNext_nonneg`, `adam_denom_pos`) were already proven in `AdamStep.lean`;
  they had no op to attach to. Still no descent claim — Adam is not monotone (AMSGrad).
- **Emission** — `tests/TestAdamOpTie.lean`: `adamWParamF` emits `emitAdamV`'s **exact 26-op
  sequence**, and the two moment ops are contiguous runs of it. `emitAdamV`'s docstring claim
  ("op-for-op the coordinate formula `Proofs.adamWParam`") is now checked, not asserted.
- **Consistency** — four `*Sgd_eq_grad` theorems (`rfl`): `den (xSgd …) = θ − lr · den (xGrad …)`.
  Un-fusing the update did not quietly change the gradient.

**First whole-net Adam render: `cifar8` ✅ DONE.** `cifar8AdamTrainStepFaithfulV` in
`Proofs/Codegen/CnnRender.lean` — same forward/backward as `cifar8TrainStepFaithfulV`, with the 22
fused SGD ops replaced by 22 un-fused gradients each feeding the three AdamW ops. Chosen because
cifar8 is 8 convs + 3 denses with **no BN**, so the four new gradient ops are exactly what it needs
and no BN fork arises.

- Signature reproduces the packed `trainAdamSched` protocol **exactly** — 71 inputs, 69 outputs,
  identical names *and* types in order:
  `(x, θ×22, m×22, v×22, %lr, %bc1, %bc2, onehot) → (θ'×22, m'×22, v'×22, loss, bc1, bc2)`.
- **Ties the retired hand-written render EXACTLY** — all **158577** returned floats bit-identical,
  every one non-zero (`.lake/build/bin/cifar8-adam-tie`). Behaviour-preserving.
- Two lines stay outside the proven surface, marked in the emitted text: the report-only scalar
  `%loss` (the kit has no rank-0 loss op, and it feeds no parameter) and the `%bc` passthroughs.
  The mean-loss `1/B` **is** proven — it is `scaleF`, not hand-written. Unlike the SGD render it
  cannot be folded into `lr`, because `lr` is a runtime scalar here.

Two gotchas worth keeping:
- **`%sc`/`%sa`/`%sb`/`%sd` are reserved SSA names.** `maxPoolBack`'s `select_and_scatter` emitter
  hardcodes them as region block arguments, so a top-level constant of the same name is a
  *redefinition* parse error — and it only surfaces at XLA compile time, not in Lean. Hence
  `%lzero`. (`tests/TestResnet34TrainPC.lean:84` records the same trap for the stem bias.)
- **Conv biases had to be renamed `%cb1…%cb8`**: `%b1` is β₁ in the Adam constant block.
- Emit each gradient **once** and let θ'/m'/v' read it back by SSA name (`.operand gradSSA`).
  `pretty` has no CSE, so passing the gradient subtree to all three ops would emit three copies of
  every conv-weight-gradient convolution.

### 2a-quater. Batch-BN at R34 scale — scoped, and it exposes a den/emit gap ▶ NEXT

Decision (2026-07-27): keep the AdamW trainer's **batch-BN** semantics rather than moving it to
the per-example chain, i.e. build batch-BN at R34 scale in the proven kit. Scoping it turned up
something that has to be fixed first, because it is the same defect.

**`bnBatchF`'s emitter discards the proof-side batch `N`.** It emits
`dense<{B*h*w}>` + `reduce [0,2,3]`, where `B` is `pretty`'s runtime batch — the `N` in
`[_N, oc, h, w]` is literally an underscore. Meanwhile `den_bnBatchF` says
`den (.bnBatchF …) = bnBatchLA N oc h w …`. Measured with a probe:

- rendering one `bnBatchF` node at `N := 32` and at `N := 1`, both with `pretty 32`, gives
  **byte-identical text**;
- `N := 32` elaborates and renders fine — `dense<512.0>` (= 32·4·4), `reduce [0,2,3]`. There is
  no type-level blowup at `N = B`.

`EfficientNetRender.lean` instantiates **every** batched op at `N := 1` and renders at `B := 32`.
Its module docstring states this deliberately: *"every `(N := 1)` below is the SHlo batch-unit;
`pretty B` carries the actual batch"*. So this is a disclosed convention, not a hidden defect —
but it does not carry uniformly, and the split is exactly along the op R34 needs:

- **Parallel-index ops — convention sound.** `den (.batchOp …) = batchMap N (denOp op)`, and at
  `N = 1` that is just the per-example op, which is what the emitter applies across the batch.
  Pointwise ops (`swishF`, `reluF`, `selectPos`, `addV`) are index-agnostic for the same reason.
  Here the batch really is a parallel repetition and `N` is free.
- **BN family — convention does NOT carry.** `bnBatchF`, `bnBatchBack`, `bnGammaSgdB`,
  `bnBetaSgdB` reduce **across** the batch: their `den`s are over `N*(h*w)`
  (`bnBatchLA N`, `bnPerChannel_grad_gamma oc (N*(h*w))`, …). At `N = 1` that is `h*w`, i.e.
  per-example — while the emitter reduces `[0,2,3]` over `B*h*w`. For these the batch is a
  reduction axis, not a parallel index, so `N` is *not* free and `N = 1` denotes a different
  function than the one that runs.

That is the real content of the "batched emit" blocker: not a missing op, and not an undisclosed
one — a convention that is sound for the parallel ops and silently wrong for the four BN ops.

**So the job is: instantiate the batched chain at `N := B`.** That makes the denotation say what
the emitter does, and it is what a genuine batch-BN R34 needs anyway. Concretely:

| piece | status |
|---|---|
| `batchOp .conv` / `.convStrided` / `.gap` / `.dense` | ✅ exist |
| `bnBatchF`, `bnBatchBack`, `gapBackBatched`, `conv{,Strided}BackBatched` | ✅ exist |
| relu / `selectPos` / `addV` / `sub` | ✅ index-agnostic — pointwise, so they work at the batched index unchanged |
| loss cotangent | ✅ `softmaxRowF (m := B) (n := nClasses)` (EfficientNet uses `m := 1`) |
| **batched maxPool fwd + back** | ❌ **missing** — R34's stem pools 2×2; EfficientNet downsamples with strided convs, which is why these were never needed |
| **batched conv-bias param grad** | ❌ **missing** — there is `convWeightSgdB`/`convStridedWeightSgdB` but no bias peer |
| **un-fused `*GradB`** for AdamW | ❌ **missing** — the `*SgdB` family is fused (`θ − lr·g`), same problem §2a-ter solved for the per-example ops |

Order of work: (1) re-instantiate EfficientNet's render at `N := B` and confirm its artifact is
unchanged — that alone closes the den/emit gap and is the cheapest possible check of the whole
idea; (2) add batched maxPool fwd/back + the conv-bias param grad; (3) un-fuse the `*SgdB` family
into `*GradB` the way §2a-ter did; (4) render `resnet34_adam_train_step` batched at `N := B := 32`
and tie it numerically against the committed artifact, which should be **exact**, since the
emitted text is what already runs.

**Still open — R34 (per-example route, now not taken).** Needs four more gradient ops (`convStridedWeightGrad`,
`convStridedBiasGrad`, `bnGammaGrad`, `bnBetaGrad` — same trimming recipe) **and, first, an answer
to the BN question in §2a**: `resnet34_adam_train_step.mlir` is batch-BN, while the proven R34
chain is per-example. Rendering it from `Proofs/` as-is would change the AdamW trainer's BN
semantics the way the `_fwd` fix changed the SGD trainer's, and would need the `REPLICAS` knob
moved out of `tests/` too. The alternative — a batch-BN R34 in the proven kit — needs `bnBatchF`
plus **batched param-grads**, which `planning/` already records as the open EfficientNet blocker.
`mobilenetv2`/`efficientnet`/`convnext`/`vit` Adam steps are untouched.

---

#### Original scope notes (kept — the analysis that set this up)

The provenance audit (`xla_pjrt_ladder.md` §8, and the tables below) found the
repo splits cleanly along an axis nobody chose deliberately:

| artifact class | rendered by |
|---|---|
| mnist/cifar `_fwd` and `_train_step` (SGD) | **`Proofs/Codegen/`** — `pretty(emit g)` of a proven graph |
| imagenette `_train_step` (SGD) | **`Proofs/Codegen/`** |
| imagenette `_fwd`, `_fwd_eval` | `tests/` |
| **every** `_adam_train_step`, all nets | `tests/` |

The dividing line is **Adam**, not the net: `emitAdamV` lives in
`LeanMlir/ViTRender.lean` (a hand-written string emitter) and is only ever called
from `tests/`. The `_fwd` split is the smaller, more tractable half — moving the
imagenette forwards into `Proofs/Codegen/` makes them match mnist/cifar.

**Scope, checked 2026-07-27: this is a rewrite, not a lift-and-shift.**
`tests/TestResnet34Fwd.lean` (308 lines) carries its *own* hand-written string
fragments — `conv`, `convStem`, `bnPC`, `relu`, `maxpool`, `addOp`, `idBlock`,
`downBlock`, `idChain` — and barely references `Proofs.` at all.
`LeanMlir/Proofs/Codegen/ResNet34Render.lean` (307 lines) independently has
`idFwd`/`downFwd` built on the proven graph ops and renders through `pretty`, but
it only emits the **train step**; there is no forward-only module. So the two are
parallel independent implementations of the same forward, which is precisely the
drift risk worth closing.

The work is: assemble a forward-only `#eval` in `ResNet34Render.lean` from the
fragments it already has, then point `verified_mlir/resnet34_fwd.mlir` at it and
retire the tests copy. The fiddly part is the **signature** — the driver feeds
parameters in `net.paramShapes` order (147 inputs for `resnet34_fwd`, 219 for
`resnet34_fwd_eval`, the latter taking running BN stats named `...mu`/`...var`).
Diff the new render against the committed artifact before switching over.

Watch out for:
- **Seven artifacts already have two writers** with no consistency check
  (`resnet34_train_step.mlir` ← both `Proofs/ResNet34Render.lean:306` and
  `tests/TestResnet34Train.lean:464`; also convnext, efficientnet, mobilenetv2,
  cifar8, and both vit files). Whoever runs last wins, silently. Moving a render
  is a good moment to add the diff check.
  → *Now audited by `scripts/regen_verified_mlir.sh check`; still 7, because the fix is
  per-net. Three are benign delegators; four own independent emitters.*
- `LeanMlir/Proofs/Codegen/ViTRender.lean` is the existing **drift guard** — it
  renders the same forward via `pretty` of the proven graph. Copy that pattern.
  → *Verified: the two `vit_fwd.mlir` writers do agree byte-for-byte.*
- `vit_adam_train_step.mlir` is written by the **trainer app itself** at run time
  (`apps/imagenette/MainViTVerifiedAdam.lean:31`) — unique lifecycle, don't be
  surprised by it.
- **There is no canonical regeneration entry point.** No lake target or script
  rebuilds `verified_mlir/`; two `lakefile.lean` comments describe it in prose.
  Worth adding while you are in here. → *Added: `scripts/regen_verified_mlir.sh`.*

### 2b. R34 on 2 GPUs ✅ RUNS — 1.46×, and the shortfall is diagnosed

`tests/TestResnet34Train.lean` has a `REPLICAS` knob (at 1 it re-renders
byte-identical; at 2 it emits 146 collectives, one per parameter). Measured,
3 epochs of Imagenette across both 7900 XTXs:

| | steps/epoch | s/epoch | ms/img |
|---|---|---|---|
| 1 GPU, bs32 | 295 | 52.5 | 5.06 |
| **2 GPU, global 64** | 147 | **~36** | **3.81** |

**1.46×, not 2×** — and the cause is already identified in `xla_pjrt_ladder.md`
§10.3a: parameters are still host-resident, so each step pushes the full 272 MB
`[θ|m|v]` to *every* replica. Compute halves while transfer doubles.
**Device-resident parameters is a prerequisite for multi-GPU scaling, not an
independent optimisation** — this measurement is the evidence.

It learns: val 36.4 / 41.9 / **49.4%** over three epochs. Lower per-epoch than the
single-GPU run (which reached ~51.7% by epoch 2) because global batch 64 with an
**unscaled** LR does half as many steps per epoch. That is the expected
large-batch recipe cost, not a defect — apply the linear-scaling rule before
comparing convergence.

A gradient check against 1×64 would be **inexact by design** here: BN normalises
per replica, so 2×32 ≠ 1×64 (§10.3b). The exact check lives on cifar8 (no BN),
where it passed at 1.015e-06.

Also fixed while running this: **the replica count is per-graph, not
per-process.** A module with no collective (the eval forwards) is compiled
single-replica; otherwise `Execute` rejects it with *"Attempted to execute with 1
argument lists when local device count is 2"*. The single-device invoke now also
refuses a multi-replica executable rather than mis-executing it.

### 2c. Then, in value order

1. **bs256 re-render + measure.** Batch is worth **1.8×** on this net
   (5.06 → 2.87 ms/img from bs32 → bs256, measured), it is a one-line `BS` edit,
   and bs256 **fits** on a 7900 XTX. Needed for ImageNet anyway.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified
   end to end.
3. **Device-resident parameters.** Two rounds of transfer work are already done
   (batching: 256→205 ms; killing the per-step host memcpys: 205→162 ms). What
   remains is smaller than it looks — see §3.
4. **Executable cache** (`PJRT_Executable_Serialize` / `DeserializeAndLoad`).
   Worth **0.1%** on an R34 training run and **53%** on the MNIST-MLP demo: a
   dev-loop and CI win, not a throughput one.

---

## 3. Corrections a new session should not have to re-derive

**"Within 1.04× of JAX" was measured in a data-bound regime and does not
generalise.** JAX on Imagenette is flat at 46.0 / 45.0 / 44.1 s per epoch for
bs32 / bs192 / bs256 — both paths idle the GPU on data loading. Compute-bound at
bs256 the honest number is **1.42×** (711 vs 501 ms/step, same box, same net,
synthetic data, `jax/scripts/jax_r34_bf16_bench.py 256`).

**XLA is NOT deterministic at epoch scale.** It is bit-identical run-to-run for a
*single step*, which is why the 1-step gate is sound — but three runs of the same
binary at the same seed gave epoch-1 val accuracy of 43.21 / 46.80 / 47.29%. Run
determinism controls **at the scale you care about**; a 1-step check is necessary,
not sufficient.

**R34's raw G2 number (gradient rel 5.20e-02) is not a defect.** The gradient
does not reproduce to better than ~6e-3 against the *same backend* under a
sub-ULP nudge, so a 1e-4 gate fails XLA-vs-XLA by 60×. Correctness comes from the
layer-level oracle instead: all six R34 layer families tie to JAX autodiff at
≤ 6.4e-06 (`tests/vjp_oracle/run.sh`).

**bf16 is worse than fp32 on this box** — measured ×0.96 for R34. Do not reach
for it here. It matters on ares only if someone measures it there.

**For Adam nets, gate G2 on the gradient (`m` after one step), never on θ.**
Adam's update is scale-free, so a near-zero-gradient parameter flips sign on a
1-ULP difference and moves a full ±lr. θ lands at ~1e-4 regardless of whether
anything is wrong.

---

## 4. Gotchas that cost time

- **`use_global_device_ids` must NOT be set** on `stablehlo.all_reduce`. It needs
  a positive `channel_id`; without one, compilation fails with *"channel_id must
  be positive when useGlobalDeviceIds is set"*.
- **Multi-GPU needs `HIP_VISIBLE_DEVICES` unset**, or the client sees one device.
  `ffi/test_pjrt_dp.c` and `test_pjrt_allreduce.c` refuse rather than silently
  running single-replica.
- **Empty `compile_options` is not "defaults"** — proto3 zeros give
  `replica_count = 0` and XLA aborts. Hence the generated blob table.
- **`.venv/bin/python3` must be a wrapper, not a symlink.** A symlinked
  interpreter derives `sys.prefix` from the symlink's location and cannot find
  jax. (Also: `python3 -c 'import jax'` from the repo root imports the local
  `jax/` *directory*.)
- **Checkpoints and `.vmfb` paths are now backend-scoped.** They were shared, so
  an XLA run could resume from, or reuse, an IREE artifact while looking normal.
  Any new driver with resume needs the same treatment.
- **`git stash -u` here would write ~17 GB** (`runs/` 5 GB, `figures/` 12 GB)
  into `.git`. Stash tracked modifications only; use `.gitignore` for the rest.
- Rendering at a non-default batch breaks eval unless the forward graph is
  re-rendered too — that is what `LEAN_MLIR_SKIP_EVAL` is for.

---

## 5. The claim ceiling

Adding a collective does not make this "verified multi-GPU", and nothing here
should be described that way. Each replica evaluates the *same tied graph* at the
batch size it was rendered for, and the collective averages gradients of that
function over disjoint equal batches. The strongest honest statement is:

> the gradient averaging is a proven identity; the collective implementing it is
> trusted, exactly like the lowerer.

And prefer **scaling** the global batch over **splitting** a fixed one: scaling
keeps each replica's BatchNorm group at the size it was tied at, so the BN caveat
never arises (`xla_pjrt_ladder.md` §10.3b).
