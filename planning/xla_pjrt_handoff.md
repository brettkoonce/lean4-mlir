# xla_pjrt_handoff.md — where the XLA/PJRT work stands, and what to do next

**Written 2026-07-27; rewritten same day after the codegen-provenance thread.** Handoff for a
fresh session. The full history, gate definitions, and every measurement live in
`planning/xla_pjrt_ladder.md`; this file is the short version — state, next moves, and the things
that cost time to learn the first time.

Branch **`xla-pjrt-backend`**, on top of `cfbdccd`. Two threads, in order:

| commit | what |
|---|---|
| `b44caaa` | PJRT C-API backend; ladder rungs 0–3 |
| `20dfd29` | emit cross-replica `all_reduce` for data-parallel AdamW |
| `9a7957a` | `String.dropEnd` deprecation fix |
| `b5b843e` | N-replica execute; data-parallel validated against 1×N |
| `92b6ac0` | R34 data-parallel on 2 GPUs; replica count per-graph |
| — | *↓ the codegen-provenance thread (§2a)* |
| `c6665e5` | `resnet34_fwd` from the proven graph — **it was computing a different function** |
| `e57613d` | `bnPerChannelEvalF`; `resnet34_fwd_eval` from the proven graph |
| `30a6918` | AdamW + un-fused param gradients as proven `SHlo` ops |
| `01724c3` | `cifar8_adam_train_step` from the proven graph |
| `7faa7fb` | the strided + BN un-fused param gradients |
| `345c9ea`, `f72903f` | batch-BN-at-R34-scale scoping (§2b) |
| *uncommitted* | §2b step 1: the batched pointwise forms; EfficientNet renders at `N := B` |

---

## 1. What works today

**A second trusted lowerer, with no Python at run time.** `ffi/pjrt_ffi.c` implements the same C
surface as `ffi/iree_ffi.c` (symbol-identical under `nm -D`), so nothing above the shim changed;
the backend is whichever `.so` is linked. Backend detection is a **weak symbol**, so a binary
cannot disagree with the library it linked.

**Rungs 0–3 complete** — linear, MLP, CNN, cifar8-bn+Adam, ResNet-34/Imagenette (513 in / 513 out,
36 BN layers). **No rung ever needed new shim code**: BN running stats and rank-0 scalars all ride
`iree_ffi_invoke_f32`. The `train_step_adam*` family in `iree_ffi.h` is still stubbed and was never
reached.

**Data-parallel multi-GPU**, validated on cifar8 (no BN) where the batch-decomposition identity is
exact: 1×256 vs 2×128+all_reduce agree on the gradient to **1.015e-06**.

**Speed, R34/Imagenette bs32:** IREE 1702 → XLA **162 ms/step** (10.5×); 52.5 s/epoch. Within
**1.04×** of hand-written JAX per step at bs32, but see §3.

**Four artifacts moved from `tests/` into `Proofs/Codegen/`** and now render as
`pretty(provenGraph)` — `resnet34_fwd`, `resnet34_fwd_eval`, `cifar8_adam_train_step`, and (already
there) `resnet34_train_step`. The optimizer itself is now a proven op family rather than a
hand-written string emitter. See §2a.

### Running it

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-verified-adam-xla data

# multi-GPU: render at REPLICAS=N first (tests/TestResnet34Train.lean), then
unset HIP_VISIBLE_DEVICES
LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 .lake/build/bin/resnet34-verified-adam-xla data

# regenerate + audit verified_mlir/  (the canonical entry point; did not exist before §2a)
scripts/regen_verified_mlir.sh          # or `check` to audit without writing
```

Env knobs added by this work: `PJRT_REPLICAS`, `PJRT_PLUGIN`, `PJRT_FFI_TRACE`,
`LEAN_MLIR_REPLICAS`, `LEAN_MLIR_SKIP_EVAL`, `LEAN_MLIR_G2_STEPS`, `LEAN_MLIR_DUMP_PARAMS`,
`LEAN_MLIR_PERTURB_R`.

---

## 2. Next up

### 2a. Codegen provenance — moving renders from `tests/` into `Proofs/` ✅ four done

The audit (`xla_pjrt_ladder.md` §8) found the repo split along an axis nobody chose: `Proofs/`
rendered `pretty(provenGraph)`, `tests/` rendered hand-written strings, and **the dividing line was
Adam, not the net**. Closing it turned up a real bug and produced a reusable op family.

**The bug.** `resnet34_fwd.mlir` and `resnet34_train_step.mlir` each had *two* writers, and they
were not two spellings of one function:

| artifact | `Proofs/` writer | `tests/` writer |
|---|---|---|
| `resnet34_train_step` | **per-example** BN, reduce `[2,3]`, n = H·W = 12544 | batch BN, reduce `[0,2,3]`, n = B·H·W = 401408 |
| `resnet34_fwd` | (didn't exist) | batch BN, 401408 |

So `resnet34-verified` (SGD) *trained* a per-channel per-example net and *scored* it with batch
statistics: on one shared (θ, x) the two forwards disagree at **rel 1.13** on non-degenerate logits.
And whichever writer ran last decided what the trainer optimised — that is what had clobbered the
working tree at the start of the session.

**Current state.** `verified_mlir/resnet34_fwd.mlir` is now a byte-identical **1106-line prefix**
of `resnet34_train_step.mlir`, ending exactly where the loss cotangent begins: eval is literally
the forward the trainer differentiates. `resnet34_train_step.mlir` came back **byte-identical** to
the committed artifact through every refactor since, which is what proves each one behaviour-
preserving.

**What got built along the way:**

- **`bnPerChannelEvalF`** — inference BN with frozen stats, `γ·(x−μ)·rsqrt(var+ε)+β`. No reduction,
  so pointwise: that is the formal content of "eval is class-batch-independent". Math mirrors the
  training chain one-for-one (`bnEvalForward → …Mat → …Flat → bnPerChannelEvalTensor3`) and, being
  affine in `x`, needs **no `0 < ε` hypothesis`**. `resnet34_fwd_eval` ties the retired render
  **exactly** (rel 0.0) — that one was always the right function, it is now certified not trusted.
- **Un-fused param gradients** — `weightGrad`, `biasGrad`, `conv{,Strided}WeightGrad`,
  `conv{,Strided}BiasGrad`, `bnGammaGrad`, `bnBetaGrad`. Every param op in the kit fused gradient
  and update (`θ − lr·g`), so there was no gradient to hand an optimizer; that, not Adam, was the
  actual blocker. Eight `*Sgd_eq_grad` `rfl` theorems say `den (xSgd …) = θ − lr · den (xGrad …)`.
- **AdamW as proven ops** — `adamMNextF` / `adamVNextF` / `adamWParamF`. Both halves of the trust
  boundary close, by different means: **denotation**, `adamW_triple_faithful` says the three ops
  denote `Proofs.adamWStep` (`rfl`); **emission**, `tests/TestAdamOpTie.lean` shows `adamWParamF`
  emits `emitAdamV`'s exact 26-op sequence. `emitAdamV`'s docstring claim is now checked rather
  than asserted. Still no descent claim — Adam is not monotone (AMSGrad).
- **`cifar8_adam_train_step` from `Proofs/`** — first whole-net AdamW render out of the kit. 71
  inputs / 69 outputs, names *and* types identical in order; ties the retired render **exactly**,
  all 158577 returned floats. cifar8 was the right first target: no BN, so no fork.
- **`scripts/regen_verified_mlir.sh`** — the canonical regeneration entry point that did not exist,
  with two audits: duplicate writers, and *forward ⊂ train-step prefix*.
- Tie harnesses: `resnet34-fwd-tie [--eval]`, `cifar8-adam-tie`. Both refuse a degenerate
  all-zero/non-finite agreement rather than reporting a green tie.

**Still `tests/`-rendered:** `mobilenetv2_fwd{,_eval}`, `efficientnet_fwd{,_eval}`, `convnext_fwd`,
and every `_adam_train_step` except cifar8. Of the 7 remaining double-writers, three (`vit_fwd`,
`vit_train_step`, `cifar8_train_step`) just *delegate* to the `Proofs/` renderer — verified
byte-identical for `vit_fwd` — so they are redundant, not divergent. The four that own independent
emitters (convnext, efficientnet, mobilenetv2, resnet34 train steps) can drift.

### 2b. Batch-BN at R34 scale ▶ step 1 DONE — and it is also a den/emit fix

Decision: keep the AdamW trainer's **batch-BN** semantics rather than moving R34-Adam onto the
per-example chain. Scoping that turned up the thing to fix first, because it is the same defect.

**`bnBatchF`'s emitter discards the proof-side batch `N`.** It emits `dense<{B*h*w}>` +
`reduce [0,2,3]` off `pretty`'s runtime batch — `N` is an underscore in the pattern — while
`den_bnBatchF` says `den = bnBatchLA N oc h w`. Probed:

- one `bnBatchF` node at `N := 32` and at `N := 1`, both under `pretty 32`, emit **byte-identical
  text**;
- `N := 32` elaborates and renders correctly (`dense<512.0>` = 32·4·4, `reduce [0,2,3]`). **No
  type-level blowup at `N = B`** — so this is a re-instantiation, not a rewrite.

`EfficientNetRender.lean` instantiated every batched op at `N := 1` and rendered at `B := 32`. Its
module docstring said so outright, so this was a **disclosed convention, not a hidden defect** — but
it did not carry uniformly, and the split falls exactly on the ops R34 needs:

- **Parallel-index ops — sound.** `den (.batchOp …) = batchMap N (denOp op)`; at `N = 1` that is
  the per-example op, which is what the emitter applies across the batch. `N` is free.
- **Batch-reduction ops — not sound.** Their `den`s reduce **across** the batch, so at `N = 1` they
  describe a ONE-EXAMPLE function while the emitter reduces over all `B`.

**The gap set is ten ops, not four.** The earlier scoping named only `bnBatchF`, `bnBatchBack`,
`bnGammaSgdB`, `bnBetaSgdB`. But the whole `*SgdB` param family has the same defect — every one of
them is `θ − lr · ∑ n : Fin N, (per-example grad on batchSlice n)`, and every one of their emitters
discards `N` and contracts the runtime batch instead (`dot_general` over dim 0, `reduce [0]`, or the
transpose-trick wgrad). So add `denseWeightSgdB`, `denseBiasSgdB`, `convWeightSgdB`,
`convStridedWeightSgdB`, `depthwiseWeightSgdB`, `depthwiseStridedWeightSgdB`.

**Step 1 is done: `EfficientNetRender` now renders at `N := B`, byte-identical.** All ten `den`s are
now honest. But the plan's framing of step 1 as a one-line re-instantiation was **wrong**, and the
reason is worth keeping:

> The single-node probe (one `bnBatchF` at `N := 1` and `N := 32` emitting identical text) was
> right about the batch-coupled ops and did not generalise to a graph. The **pointwise** ops
> (`swishF`, `swishBack`, `sigmoidBack`, `addV`, `sub`) carry only the SHlo index and emit
> `tensor<B×n>` from it, so at the batched index `N·s` they emit `tensor<B×(N·s)>` — a type that
> does not even match their own operand. Measured on the whole net: re-instantiating at `N := B`
> left the graph structure byte-identical (same ops, same order, same SSA numbering) and changed
> **597 of 7466 lines**, every one a pointwise op's trailing dim inflated 32×.

So step 1 cost seven new codegen forms rather than a `sed`. All of them separate the batch `N` from
the per-example emit width `n`, and all are reusable for R34:

| form | kind | why |
|---|---|---|
| `BatchableOp.swish`, `.softmaxRow`, `.denseRowBack` | descriptors | what they lift is a *fixed* function (no data, or a shared `W`) |
| `swishBackB`, `sigmoidBackB` | own `SHlo` ctors | see the trap below |
| `addVB`, `subB` | own ctors, via a new binary `batched2` Raw/Tok tag | `batchOp` is unary |

**The trap, which cost a wrong commit's worth of time and would not have been caught by the
artifact.** `BatchableOp` lifts a fixed function by `batchMap N`, so a descriptor **must not carry
batch-varying data**. `swishBack`/`sigmoidBack` look pointwise but their VJP is
`fun x dy i => dy i * deriv (x i)` — it depends on the saved pre-activation, which differs per
example. As descriptors their `den` would say *every example shares one example's activation*.
The byte-identical artifact **cannot** witness this: the render is value-independent (`skel` erases
values), so the wrong descriptor renders exactly the same bytes. Only the `rfl` faithfulness
theorem does. Any op carrying a saved activation needs its own constructor holding the
**whole-batch** `x : Vec (N*n)` — which is what the emitted `xName` holds at runtime anyway.
**`selectPos` is the same shape, so R34's relu backward hits this too.**

`scripts/regen_verified_mlir.sh check` grew a third audit for the related silent-failure mode: the
emit catch-alls (`// MALFORMED token stream`, `// … render TODO`) are *comments*, so a missing
`emitTok` case fails nothing — not the Lean build, not iree-compile — and surfaces only as a wrong
number much later.

**Inventory for a batched R34:**

| piece | status |
|---|---|
| `batchOp .conv` / `.convStrided` / `.gap` / `.dense` | ✅ exist |
| `bnBatchF`, `bnBatchBack`, `gapBackBatched`, `conv{,Strided}BackBatched` | ✅ exist |
| relu / `selectPos` / `addV` / `sub` | ✅ index-agnostic, work unchanged |
| loss cotangent | ✅ `softmaxRowF (m := B) (n := nClasses)` (EfficientNet uses `m := 1`) |
| **batched maxPool fwd + back** | ❌ missing — R34's stem pools 2×2; EfficientNet downsamples with strided convs, so these were never needed |
| **batched conv-bias param grad** | ❌ missing — `conv{,Strided}WeightSgdB` exist, no bias peer |
| **un-fused `*GradB`** | ❌ missing — the `*SgdB` family is fused, the same problem §2a solved for the per-example ops |

**Order of work:**

1. ~~Re-instantiate EfficientNet's render at `N := B`~~ ✅ **DONE** — `verified_mlir/`
   `efficientnet_train_step.mlir` is byte-identical, all ten `den`s honest, full
   `lake build Proofs Certs Codegen` green (the parser round-trip included — `batched2` needed a
   `parseStack` case and a `parse_toToks` induction case).
2. ~~The R34 pointwise forms~~ ✅ **DONE** — `BatchableOp.relu` (descriptor) and `selectPosB` (own
   ctor, whole-batch `x` — it is `swishBack`'s shape, not `swish`'s). With `addVB`/`subB` from
   step 1, R34's whole pointwise set is covered. Ties: `den_batchOp_relu_eq_reluF`,
   `selectPosB_faithful`.
3. ~~Batched maxPool fwd/back + the conv-bias param grad~~ ✅ **DONE** — `BatchableOp.maxPool`
   (descriptor), `maxPoolBackB` (own ctor: it routes `dy` to the saved input's window argmax, so
   per-example data), `convBiasSgdB` + `convStridedBiasSgdB`. The bias grad is stride-INDEPENDENT
   (`Σ_{batch,spatial} dy`), so both bias ops `skel` to ONE Raw and share an emit case — the
   established `convStridedBiasSgd`-aliases-`convBiasSgd` pattern; only `den` differs. Also added
   **`batchMapAux`**, the "batchMap with per-example auxiliary data" combinator that `maxPoolBackB`
   needs and `seBackBatched` had inlined — it is the shape every saved-activation backward takes,
   and naming it makes the descriptor restriction legible.
4. **▶ NEXT: un-fuse the `*SgdB` family into `*GradB`**, the way §2a did for the per-example ops.
   That is what hands AdamW a gradient instead of a fused `θ − lr·g`; §2a's note applies verbatim
   — the fusion, not Adam, was the blocker. Peers needed for R34: `conv{,Strided}WeightGradB`,
   `conv{,Strided}BiasGradB`, `bnGammaGradB`, `bnBetaGradB`.
5. Render `resnet34_adam_train_step` batched at `N := B := 32` and tie it numerically against the
   committed artifact. It should be **exact** — the emitted text is what already runs.

**`tests/TestBatchedEmitTie.lean`** pins all thirteen batched forms against their per-example peers,
byte-for-byte, one case each. Add a case with every new batched form in steps 3–4 — it is what
catches an emitter that reads its width off the SHlo index again. The tie was verified to actually
fail by deliberately breaking `relu`'s emit case. Note it fails via `throw`, not
`IO.Process.exit 1`: under `#eval` the elaborator buffers output and prints it only after the eval
returns, so `exit` discards **every** diagnostic and you get a bare non-zero status. Several older
`tests/*.lean` use the `exit` form and fail blind.

Cheapest sanity check for any of these, learned from step 1: render the *whole net* at both `N`
values into temp files and `diff` them, and separately `diff` with all `tensor<…>` annotations
stripped. Structure-identical + types-differ localises the breakage to the exact op family in one
pass; a single-node probe does not.

The per-example route is *not* being taken, but if it is ever revisited: it needs no new ops
(§2a's eight gradients cover it), and its consequences are that the AdamW trainer's BN semantics
change, the `REPLICAS` knob must move out of `tests/`, and — because per-example BN makes
train == eval — `bnChannels` goes empty and `@resnet34_fwd_eval` loses its caller.

### 2c. R34 on 2 GPUs ✅ RUNS — 1.46×, and the shortfall is diagnosed

`tests/TestResnet34Train.lean` has a `REPLICAS` knob (at 1 it re-renders byte-identical; at 2 it
emits 146 collectives, one per parameter). Measured, 3 epochs of Imagenette across both 7900 XTXs:

| | steps/epoch | s/epoch | ms/img |
|---|---|---|---|
| 1 GPU, bs32 | 295 | 52.5 | 5.06 |
| **2 GPU, global 64** | 147 | **~36** | **3.81** |

**1.46×, not 2×** — cause identified in `xla_pjrt_ladder.md` §10.3a: parameters are still
host-resident, so each step pushes the full 272 MB `[θ|m|v]` to *every* replica. Compute halves
while transfer doubles. **Device-resident parameters is a prerequisite for multi-GPU scaling, not
an independent optimisation** — this measurement is the evidence.

It learns: val 36.4 / 41.9 / **49.4%** over three epochs — but see §3, those numbers predate the
BN fix. Lower per-epoch than the single-GPU run because global batch 64 with an **unscaled** LR
does half as many steps per epoch; that is the expected large-batch recipe cost, not a defect.

A gradient check against 1×64 would be **inexact by design**: BN normalises per replica, so
2×32 ≠ 1×64 (§10.3b). The exact check lives on cifar8 (no BN), where it passed at 1.015e-06.

Also fixed while running this: **the replica count is per-graph, not per-process.** A module with
no collective (the eval forwards) is compiled single-replica; otherwise `Execute` rejects it with
*"Attempted to execute with 1 argument lists when local device count is 2"*. The single-device
invoke now also refuses a multi-replica executable rather than mis-executing it.

### 2d. Then, in value order

1. **bs256 re-render + measure.** Batch is worth **1.8×** on this net (5.06 → 2.87 ms/img from
   bs32 → bs256, measured), it is a one-line `BS` edit, and bs256 **fits** on a 7900 XTX. Needed
   for ImageNet anyway.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified end to end.
3. **Device-resident parameters.** Two rounds of transfer work are already done (batching:
   256→205 ms; killing the per-step host memcpys: 205→162 ms). What remains is smaller than it
   looks — see §3.
4. **Executable cache** (`PJRT_Executable_Serialize` / `DeserializeAndLoad`). Worth **0.1%** on an
   R34 training run and **53%** on the MNIST-MLP demo: a dev-loop and CI win, not throughput.

---

## 3. Corrections a new session should not have to re-derive

**Published `resnet34-verified` accuracy predates the BN fix.** The SGD trainer used to score with
batch statistics a net it trained with per-example ones (§2a). Re-measure before quoting; this
folds into the verified-path table re-run that was already decided.

**"Within 1.04× of JAX" was measured in a data-bound regime and does not generalise.** JAX on
Imagenette is flat at 46.0 / 45.0 / 44.1 s per epoch for bs32 / bs192 / bs256 — both paths idle the
GPU on data loading. Compute-bound at bs256 the honest number is **1.42×** (711 vs 501 ms/step,
same box, same net, synthetic data, `jax/scripts/jax_r34_bf16_bench.py 256`).

**XLA is NOT deterministic at epoch scale.** It is bit-identical run-to-run for a *single step*,
which is why the 1-step gate is sound — but three runs of the same binary at the same seed gave
epoch-1 val accuracy of 43.21 / 46.80 / 47.29%. Run determinism controls **at the scale you care
about**; a 1-step check is necessary, not sufficient.

**R34's raw G2 number (gradient rel 5.20e-02) is not a defect.** The gradient does not reproduce to
better than ~6e-3 against the *same backend* under a sub-ULP nudge, so a 1e-4 gate fails XLA-vs-XLA
by 60×. Correctness comes from the layer-level oracle instead: all six R34 layer families tie to
JAX autodiff at ≤ 6.4e-06 (`tests/vjp_oracle/run.sh`).

**bf16 is worse than fp32 on this box** — measured ×0.96 for R34. Do not reach for it here. It
matters on ares only if someone measures it there.

**For Adam nets, gate G2 on the gradient (`m` after one step), never on θ.** Adam's update is
scale-free, so a near-zero-gradient parameter flips sign on a 1-ULP difference and moves a full
±lr. θ lands at ~1e-4 regardless of whether anything is wrong.

---

## 4. Gotchas that cost time

**Codegen / proof kit**

- **A plain `lake build` does not check the parser round-trip.**
  `LeanMlir/Proofs/Codegen/StableHLOParse.lean` is under the **`Certs`** lib, not the
  `@[default_target]` `Proofs` lib. Add an `SHlo` constructor and `lake build` stays green while
  `roundtrip` (`parse (toToks (skel a)) = some (skel a)`, over **all** `SHlo`) is missing a case.
  Use `lake build Proofs Certs Codegen`.
- **Adding an `SHlo` op touches ten sites**: constructor, `den`, the `rfl` faithfulness theorem,
  `Raw`, `skel`, `Tok`, `toToks`, `emitTok` — plus `parseStack` and the `parse_toToks` induction
  case in `StableHLOParse.lean`. Grep an existing op (`bnPerChannelF`) as the template. Note
  `serializeToks` just folds `emitTok`; the case goes in `emitTok`, which **ends in a catch-all
  emitting `// MALFORMED token stream`**, so a missing case there is silent, not an error.
  (`scripts/regen_verified_mlir.sh check` now greps the artifacts for that marker.)
- **A `BatchableOp` descriptor is the four-site version of that** — constructor, `denOp`,
  `batchOpDescr`, `emitTok` — because `skel`/`Raw`/`Tok`/`toToks`/`parseStack`/`parse_toToks` all
  route through the generic `.batched` tag already. Prefer it. **But a descriptor may only carry
  batch-INVARIANT data**: `den` is `batchMap N (denOp op)`, which lifts one *fixed* function across
  the batch. A shared weight is fine (`denseRowBack`'s `W`); a saved per-example activation is not
  (`swishBack`, `sigmoidBack`, `selectPos`) — those need their own constructor holding the
  whole-batch `x : Vec (N*n)`. Nothing in the render catches the mistake; see §2b.
- **Binary batched ops go through `batched2`**, added for `addVB`/`subB`. `batchOp` is unary.
- **`%sc` / `%sa` / `%sb` / `%sd` are reserved SSA names.** `maxPoolBack`'s `select_and_scatter`
  emitter hardcodes them as region block arguments, so a top-level constant of the same name is a
  *redefinition* parse error — and it surfaces only at XLA compile time, not in Lean.
- **`pretty` has no CSE.** If three ops consume the same gradient subtree, the gradient is emitted
  three times. Emit it once and thread the SSA name (`.operand gradSSA`).
- **Two writers for one artifact is a silent last-writer-wins race.** Run
  `scripts/regen_verified_mlir.sh check` before trusting anything in `verified_mlir/`.

**Runtime / PJRT**

- **`use_global_device_ids` must NOT be set** on `stablehlo.all_reduce`. It needs a positive
  `channel_id`; without one, compilation fails with *"channel_id must be positive when
  useGlobalDeviceIds is set"*.
- **Multi-GPU needs `HIP_VISIBLE_DEVICES` unset**, or the client sees one device.
  `ffi/test_pjrt_dp.c` and `test_pjrt_allreduce.c` refuse rather than silently running
  single-replica.
- **Empty `compile_options` is not "defaults"** — proto3 zeros give `replica_count = 0` and XLA
  aborts. Hence the generated blob table.
- **`.venv/bin/python3` must be a wrapper, not a symlink.** A symlinked interpreter derives
  `sys.prefix` from the symlink's location and cannot find jax. (Also: `python3 -c 'import jax'`
  from the repo root imports the local `jax/` *directory*.)
- **Checkpoints and `.vmfb` paths are backend-scoped.** They were shared, so an XLA run could
  resume from, or reuse, an IREE artifact while looking normal. Any new driver with resume needs
  the same treatment.
- **`git stash -u` here would write ~17 GB** (`runs/` 5 GB, `figures/` 12 GB) into `.git`. Stash
  tracked modifications only; use `.gitignore` for the rest.
- Rendering at a non-default batch breaks eval unless the forward graph is re-rendered too — that
  is what `LEAN_MLIR_SKIP_EVAL` is for.

---

## 5. The claim ceiling

Adding a collective does not make this "verified multi-GPU", and nothing here should be described
that way. Each replica evaluates the *same tied graph* at the batch size it was rendered for, and
the collective averages gradients of that function over disjoint equal batches. The strongest
honest statement is:

> the gradient averaging is a proven identity; the collective implementing it is trusted, exactly
> like the lowerer.

Prefer **scaling** the global batch over **splitting** a fixed one: scaling keeps each replica's
BatchNorm group at the size it was tied at, so the BN caveat never arises (§10.3b).

And on the renders: `pretty(provenGraph)` means the committed bytes are the certified render *of
the graph that was proven* — it does not mean the emitter is verified. The `Tok → StableHLO-text`
lexing stays audited-but-trusted, which is why every move in §2a is backed by a numeric tie against
what it replaced, not by the faithfulness theorem alone. Two places currently emit text that is
**not** `pretty` of an AST node, and both say so in the emitted output: `cifar8_adam_train_step`'s
report-only scalar `%loss` and its `%bc` passthroughs.
