# xla_pjrt_handoff.md — where the XLA/PJRT work stands, and what to do next

**Written 2026-07-27; rewritten 2026-07-28 after the batched-index thread (§2b) closed.** Handoff
for a fresh session. The full history, gate definitions, and every measurement live in
`planning/xla_pjrt_ladder.md`; this file is the short version — state, next moves, and the things
that cost time to learn the first time.

Branch **`xla-pjrt-backend`**, on top of `cfbdccd`. Three threads, in order:

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
| — | *↓ the batched-index thread (§2b), 2026-07-28* |
| `a56eab4` | EfficientNet renders at `N := B`; 7 batched pointwise/row forms |
| `828875f` | batched ReLU forms; `tests/TestBatchedEmitTie.lean` |
| `79f2a65` | batched max-pool + conv bias grads; `batchMapAux` |
| `ce9c1df` | un-fuse the batched `*SgdB` family into `*GradB` (8 ops) |
| `4af61ff` | `bnBatchMeanB`/`bnBatchVarB` — the BN running stats |
| `2618ba4` | **`ResNet34RenderB.lean`** — the batched R34 AdamW train step |
| `b856deb` | the numeric tie: **forward bit-exact, backward norm-rel 1e-6** |

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

**The batched renderers are honest** (§2b). EfficientNet and the new batch-BN ResNet-34 AdamW train
step both sit at the batched index `N := B`, so the ten batch-*reducing* `den`s describe what the
emitted text actually computes. `ResNet34RenderB.lean` → `resnet34_adam_train_step_b.mlir` **ties
the hand-written render numerically**: forward bit-exact, backward norm-relative 1e-6. It is not yet
what the trainer runs — see §2b.

### Running it

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-verified-adam-xla data

# multi-GPU (§2b-quater): the DP render is certified-graph + trusted collective, written by the
# same `lake build LeanMlir.Proofs.Codegen.ResNet34RenderB` as the single-device one. To change the
# replica count, edit the `(replicas := 2)` #eval there and rebuild — `replica_groups` is baked, so
# it must match PJRT_REPLICAS.
unset HIP_VISIBLE_DEVICES
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/resnet34-verified-adam-xla data

# regenerate + audit verified_mlir/  (the canonical entry point; did not exist before §2a)
scripts/regen_verified_mlir.sh          # or `check` to audit without writing

# the render ties (§2a, §2b). The R34 AdamW one needs a GPU; the rest are CPU-only.
lake env lean tests/TestBatchedEmitTie.lean            # 13 emit ties + 8 grad-prefix checks
lake build resnet34-adam-tie && .lake/build/bin/resnet34-adam-tie
lake build cifar8-adam-tie   && .lake/build/bin/cifar8-adam-tie

# the step-time bench (§2b-bis). Takes both paths so it can be run in either compile order,
# which is the control for the ~2.1 s first-compile-in-process cost.
lake build resnet34-adam-bench
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-adam-bench \
  verified_mlir/resnet34_adam_train_step.mlir verified_mlir/resnet34_adam_train_step.mlir 20

# the data-parallel semantics gate (§2b-quater). Needs TWO GPUs.
unset HIP_VISIBLE_DEVICES
lake build cifar8-dp-check && PJRT_REPLICAS=2 .lake/build/bin/cifar8-dp-check
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
and every `_adam_train_step` except cifar8 **and resnet34** (§2b-ter). Counting writers across all
50 artifacts: **21 `Proofs/`-only, 22 `tests/`-only, 4 contested.**

The `tests/`-only 22 include four whole-net AdamW renders with **live drivers** —
`vit-verified-adam`, `convnext-verified-adam`, `efficientnet-verified-adam`,
`mobilenetv2-verified-adam` all train on hand-written bytes. So the AdamW scorecard is
**2 of 6 certified** (cifar8 §2a, resnet34 §2b-ter).

**The double-writers were 7; they are now 4** — see §2a-quater. The four that remain
(`convnext`/`efficientnet`/`mobilenetv2`/`resnet34` **SGD** train steps) each own an independent
emitter, so they can genuinely diverge; `resnet34_train_step` is *demonstrated* to flip
(md5 `3184522f` ↔ `929074f6`) simply by elaborating its test file.

### 2a-quater. The three delegators, retired ✅

Three of the 7 contested artifacts had `tests/` writers that merely re-invoked the `Proofs/`
renderer. Retired 2026-07-28, each only after being **proved** byte-redundant rather than assumed:

- **`vit_fwd`, `vit_train_step`** — `tests/TestViT{Fwd,Train}.lean` called
  `vitFwdRenderV "vit_fwd"` and `vitTrainStepRenderV "vit_train_step" "0.003125"`, the identical
  arguments the `Proofs/` `#eval` uses. Ran both writers and diffed: md5 `626cc192…` / `f57aff00…`
  unchanged. The write is gone; **the `iree-compile` smoke stays**, now reading the committed bytes
  instead of re-rendering them — that is the part `lake build` genuinely cannot do, since it needs
  the compiler on PATH. They now fail loudly if the artifact is missing rather than quietly
  recreating it.
- **`cifar8_train_step`** — `tests/RenderCifar8Sgd02.lean` was **not** a delegator, despite being
  classified as one. It called the same certified renderer at a **different learning rate**
  (`0.00015625` = 0.02/128, against the committed `0.00078125` = 0.1/128), so elaborating it
  silently replaced a committed certified artifact with different hyperparameters — its own
  docstring said "Temporary" and told you to `git checkout` afterwards. **Deleted.** It backed the
  momentum/lr-0.02 column of `runs/ablation_cifar8/README.md`, so that README now records where to
  recover it (`git show 57e7a12:tests/RenderCifar8Sgd02.lean`) and to restore the artifact after.

The lesson worth keeping: *"currently byte-identical" is not a property that maintains itself.* A
redundant writer costs nothing until someone edits one of the two — which is precisely how
`resnet34_train_step` (§2a) and `resnet34_adam_train_step` (§2b-ter) went wrong.

### 2a-quinquies. The last four double-writers ✅ DONE — the audit is at 0

Done 2026-07-28. `scripts/regen_verified_mlir.sh check` now reports **"OK — 50 artifacts, one writer
each"**. All four were SGD `_train_step` artifacts whose `tests/` writer owned an *independent*
emitter:

| artifact | `Proofs/` writer | retired `tests/` writer | interface (in/out) | tie result |
|---|---|---|---|---|
| `convnext_train_step` | `ConvNeXtRender.lean:254` | `TestConvNeXtTrain.lean` | **182/180 both** | ✅ **gradient BIT-EXACT** |
| `efficientnet_train_step` | `EfficientNetRender.lean:534` | `TestEfficientNetTrain.lean` | **264/262 both** | ❌ **FAILED — different function** |
| `mobilenetv2_train_step` | `MobileNetV2Render.lean:577` | `TestMobilenetV2Train.lean` | **212/210 vs 84/82** | n/a — different ARITY |
| `resnet34_train_step` | `ResNet34Render.lean:438` | `TestResnet34Train.lean` | — | n/a — per-example vs batch BN (§2a) |

**The committed bytes were already the `Proofs/` render for all four**, so the retirement changed no
bytes and no behaviour: every artifact's md5 is unchanged, and `verified_mlir/` stays git-clean
through a full `lake build` and through elaborating all four edited files. It removed clobber
hazards; it swapped nothing.

Each file was reduced to the §2a-quater shape rather than deleted outright — the `iree-compile`
smoke **stays**, now reading the *committed* bytes instead of re-rendering them, and throwing if
they are missing. That is the part `lake build` genuinely cannot do (it needs the compiler on PATH),
and all four smokes were run green with `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm`. Group B
(`TestConvNeXtTrain`, `TestEfficientNetTrain`) keeps its AdamW render, which is still that
artifact's sole writer. Side benefit: the mnv2 smoke now compiles the **full** 212/210 render
instead of the reduced 84/82 one it used to write.

#### The tie harness — `tests/TestSgdRenderTie.lean` → `lake build sgd-render-tie`

```bash
.lake/build/bin/sgd-render-tie <slug> <pathA> <lrA> <pathB> <lrB>
```

Adapted from `TestResnet34AdamTie.lean`, with **two changes that matter for an SGD render**:

1. **It compares the recovered gradient `g = (θ − θ')/lr`, not `θ'`.** An SGD train step returns
   `θ' = θ − lr·g` and *nothing else* — no loss, no BN stats, 180 in / 180 out. Comparing `θ'`
   directly is close to meaningless: `θ'` is dominated by `θ`, the **same input on both sides**, so
   a wholly wrong gradient still lands within `lr·|g| / |θ|` of a match. Recovering `g` removes the
   shared term exactly.
2. **The lr is passed per side and is not defaulted** — because the two emitters do not always agree
   on it (see below), and a wrong lr silently rescales one side of the comparison.

The gate is **per-parameter** norm-relative (`max|gA−gB| / max|gA|` within each parameter): a global
denominator would hide a whole small-gradient layer under the largest layer's scale, and a
per-coordinate ratio is meaningless on a near-zero gradient entry (§3). It runs **A against itself
first** (skippable with `TIE_SKIP_AA=1`) — bit-exact on both nets, so any A-vs-B difference is
graph-attributable, not backend noise.

**What it does NOT establish, unlike the R34 AdamW tie:** there is no forward-only output here (the
R34 one gates on `bnstat`, the batch μ/var of all 36 BN inputs), so a forward disagreement and a
backward one both land in `g` and cannot be separated. The gradient is still a strong forward check
— a mis-wiring at layer k perturbs every layer ≥ k and the cotangent reaching every layer < k — but
say "the two emitters compute the same gradient", not "the same forward".

**ConvNeXt: bit-exact.** 27,811,542 gradient coordinates, `max|gA−gB| = 0.0`, 0/180 parameters
disagreeing, against a bit-exact A-vs-A floor. Two structurally different emitters (4483 vs 3590
lines) computing the identical gradient to the last bit ⇒ that deletion is *provably* lossless.

#### ▶ EfficientNet's tie FAILED — and it is a real divergence, not a tolerance argument

All 262 parameters disagreed at norm-relative **0.96875 = 31/32 exactly** — the signature of
`g_tests = g_Proofs / 32`. Root-caused in the emitted text, not inferred from the number:

| | loss cotangent | baked lr | effective lr on the MEAN loss |
|---|---|---|---|
| committed (`Proofs/`) | **sum**-CE — `subtract %v1794, %onehot` straight into `dot_general` | 0.05 | 0.05 × 32 = **1.6** |
| retired (`tests/`) | **mean**-CE — `%bnc = dense<32.0>` then `divide %dyr, %bnc` | 0.1 | **0.1** |

So `tests/TestEfficientNetTrain.lean` was a live instance of the `RenderCifar8Sgd02` hazard
(§2a-quater): a `tests/` writer that, on elaboration, silently replaced a committed certified
artifact with **different hyperparameters** — a 16× smaller effective step. Deleting it is strictly
a fix, and the tie is what made the divergence visible.

**The convention split it exposed — checked, and BENIGN.** The same probe over all committed SGD
renders shows two of them are off the house convention:

| committed render | cotangent | lr | effective mean-loss lr | |
|---|---|---|---|---|
| `resnet34_train_step` | sum-CE | 0.003125 = 0.1/32 | **0.1** | ✅ the "r34 convention" — mean folded into lr |
| `vit_train_step` | sum-CE | 0.003125 | **0.1** | ✅ same |
| `convnext_train_step` | mean-CE (÷32) | 0.1 | **0.1** | ✅ same effective value, spelled differently |
| `efficientnet_train_step` | sum-CE | 0.05 | **1.6** | ⚠️ 32× the apparent lr |
| `mobilenetv2_train_step` | sum-CE | 0.3 | **9.6** | ⚠️ and `MainMobilenetV2Verified.lean:19` documents itself as *"mean-loss SGD lr=0.3"* |

**These are tuned values, not a 32× slip** — settled from the existing 80-epoch runs, no re-run
needed. `runs/efficientnet_verified_crop_gpu1.log` goes 40.6% → **87.81%** and
`runs/mobilenetv2_verified_crop_gpu0.log` 32.9% → **86.89%**, matching README's 87.58 / 87.09. Both
descend smoothly from epoch 1. (BN makes a net scale-invariant in its weights, so a large *nominal*
lr is not the same thing as a large step.)

What is left is therefore a **documentation** inconsistency, not a defect:
`MainMobilenetV2Verified.lean:19` describes itself as *"mean-loss SGD lr=0.3"* when its graph is
sum-loss at 0.3 — 32× apart in what a reader would compute. Same for EfficientNet, which documents
no lr at all. Worth one line in each docstring stating the convention and the effective value; not
worth changing a number that demonstrably trains.

Traps that still apply to anyone touching these files:
- **Elaborating a `tests/` render file can rewrite an artifact.** Always `git diff verified_mlir/`
  after. The four SGD writers are gone, but the AdamW writers in Group B remain.
- **mnv2 fails loudly, the others silently.** A clobbered mnv2 artifact has the wrong *arity*, so
  the driver refuses it; convnext/efficientnet/resnet34 clobbers produce a runnable graph computing
  something else. Do not let mnv2's noisiness set your expectations for the other three.
- `scripts/regen_verified_mlir.sh check` is the scoreboard: **0**, and it should stay there.

### Then, separately: the four uncertified whole-net AdamW renders

Distinct job, much larger, and *not* a prerequisite for the above. `vit`, `convnext`,
`efficientnet`, `mobilenetv2` `_adam_train_step` are hand-written with **live drivers**
(`*-verified-adam`), so the AdamW scorecard is 2 of 6. From what §2b cost at R34 scale:

- **ViT is likely cheapest** — `vit_fwd`/`vit_train_step` already render from `Proofs/`, so only the
  AdamW tail is missing, and `emitAdamV` is already a proven op family (§2a).
- **EfficientNet is next** — it already renders at `N := B` and came back byte-identical through the
  §2b move, but its AdamW render needs `depthwise{,Strided}WeightGradB` plus a depthwise bias peer
  (real kit work, ~4 new forms × the 4-site `BatchableOp` recipe).
- **MobileNetV2** needs those same depthwise gradients, so it comes free-ish after EfficientNet.
- **ConvNeXt** carries the 4 even-kernel gaps from its §1a tie; check whether they touch the
  backward before scoping.

The R34 playbook that worked: un-fuse the gradients → render AdamW from the proven ops → numeric
tie against the hand-written render → bench → swap and retire in one change.

### 2b. Batch-BN at R34 scale ✅ DONE — the batched index, and a certified R34 AdamW render

Decision taken and carried out: keep the AdamW trainer's **batch-BN** semantics rather than moving
R34-Adam onto the per-example chain. The result is
`LeanMlir/Proofs/Codegen/ResNet34RenderB.lean` → `verified_mlir/resnet34_adam_train_step_b.mlir`,
the first whole-net batch-BN AdamW render out of the proven kit, **numerically tied** to the
hand-written one.

**The defect this closed.** `pretty B` renders a graph whose SHlo index is the *per-example* width
and whose emitted tensors are `[B, index]`. For ops where the batch is a **parallel** index that is
sound at any `N` — `den (.batchOp …) = batchMap N (denOp op)`, and at `N = 1` that is the
per-example op the emitter applies across the batch. For ops where the batch is a **reduction axis**
it is not: at `N = 1` their `den` describes a ONE-EXAMPLE function while the emitted text reduces
over all `B`. **Ten ops in these graphs are of the second kind**, not the four originally scoped:
`bnBatchF`, `bnBatchBack`, `bnGammaSgdB`, `bnBetaSgdB`, **and the whole `*SgdB` param family**
(`dense{Weight,Bias}SgdB`, `conv{,Strided}WeightSgdB`, `depthwise{,Strided}WeightSgdB` — each
`θ − lr·∑ n : Fin N, …` against an emitter that contracts the runtime batch). Moving the graphs to
`N := B` makes all ten honest.

**What it cost, and why the original estimate was wrong.** The plan called step 1 a one-line
re-instantiation, on the strength of a single-node probe: one `bnBatchF` at `N := 1` and `N := 32`
emits byte-identical text. That was true and did not generalise. The **pointwise** ops read their
emit width off the SHlo index, so at index `N·s` they emit `tensor<B×(N·s)>` — a type that does not
match their own operand. Measured on the whole EfficientNet: re-instantiating left the graph
structure byte-identical (same ops, same order, same SSA numbering) and changed **597 of 7466
lines**, every one a pointwise trailing dim inflated 32×. So it cost **20 new codegen forms**, all
of which separate the batch `N` from the per-example emit width `n`:

| forms | kind |
|---|---|
| `BatchableOp.swish` `.relu` `.maxPool` `.softmaxRow` `.denseRowBack` | descriptors (4 sites each) |
| `swishBackB` `sigmoidBackB` `selectPosB` `maxPoolBackB` | own ctors — they carry per-example saved data |
| `addVB` `subB` | own ctors via the new binary `batched2` Raw/Tok tag |
| `scaleB` `shiftB` `divConstB` | pointwise affine-by-a-literal (the cotangent pieces) |
| `conv{,Strided}{Weight,Bias}GradB`, `bnGammaGradB`, `bnBetaGradB`, `denseWeight/BiasGradB` | the un-fused `*GradB` peers of §2a's eight |
| `convBiasSgdB` `convStridedBiasSgdB`, `bnBatchMeanB` `bnBatchVarB` | the missing fused bias grads + BN running stats |

plus **`batchMapAux`**, the "batchMap with per-example auxiliary data" combinator — the shape every
saved-activation backward takes, and the thing that makes the descriptor restriction legible.

**Artifacts:** `efficientnet_train_step.mlir` came back **byte-identical** through the whole move,
which is what proves it behaviour-preserving. `resnet34_adam_train_step_b.mlir` is new.

#### The R34 tie — passed, and it caught a bug

`lake build resnet34-adam-tie && .lake/build/bin/resnet34-adam-tie` (XLA/PJRT, 7900 XTX, one step,
all **68,040,737** returned floats):

| check | result |
|---|---|
| interface vs the hand-written artifact | 515 in / 513 out, all types positionally identical ✅ |
| structural ops (conv 107, transpose 143, reverse 35, pad 13, `reduce_window` 1, `select_and_scatter` 1, sqrt 146, dot_general 3) | exact match ✅ |
| **forward** (`bnstat` = batch μ/var of all 36 BN inputs) | **BIT-EXACT 17024/17024** ✅ |
| **loss** | bit-exact 3/3 ✅ |
| **backward** (θ/m/v) | **norm-rel 1–2e-6** ✅ |

The backward figure has **run-to-run spread**: four of five runs report 1e-6, one reported 2e-6, on
identical binaries and artifacts. The forward stayed bit-exact `17024/17024` in every run. Most
likely XLA autotuning picking a different convolution algorithm between processes — note this does
NOT contradict §3's "bit-identical for a single step", which is about repeated execution *within*
a process, and which the A-vs-A run confirms. Quote the tie as **≤ 2e-6**, not as a fixed number;
the 1e-4 gate has 50× headroom either way.

**The gate is deliberately not per-coordinate relative** — see §3: R34's gradient does not reproduce
to better than ~6e-3 per-coordinate even XLA-vs-XLA under a sub-ULP nudge, so a 1e-4 per-coordinate
gate fails a *correct* render by 60×. Here per-coordinate max rel is **0.32** while max|a−b| is
below 1e-6; that number is noise. The two gates that mean something: the **forward must be
bit-exact** (`bnstat` pins stem, every block and every BN, so a real mis-wiring is a hard failure
rather than a tolerance argument), and the backward must agree **norm-relative** (max|a−b|/max|a|).
The harness runs A against itself first — **bit-exact on all 68M outputs** — so any A-vs-B
difference is graph-attributable, never backend noise. Without that baseline the 1e-6 is an
assertion, not a measurement.

**The tie caught a real bug:** the first render computed PLAIN cross-entropy for `%loss` instead of
the SMOOTHED one its own cotangent implies. 0.28% loss disagreement against an otherwise
bit-identical forward. `%loss` is report-only, on no gradient path, so no proof in the repo could
have seen it — it would have surfaced as a training curve that quietly failed to match the
reference. It is exactly the hand-written non-AST carve-out §5 flags.

#### ▶ What is left here

1. ~~**Measure a step.**~~ ✅ **DONE 2026-07-28 — the 1.68× op count costs nothing.** See §2b-bis
   below. The render is free at run time and free in memory; the driver swap is unblocked.
2. ~~**Then swap the driver**, and retire `tests/TestResnet34Train.lean`'s AdamW writer.~~
   ✅ **DONE 2026-07-28 — see §2b-ter.**
3. **EfficientNet's own Adam render** still needs `depthwise{,Strided}WeightGradB` and a depthwise
   bias peer. Out of R34's scope, so not built.

#### 2b-bis. The step measurement — XLA CSE collapses the whole gap

`tests/TestResnet34AdamBench.lean` → `lake build resnet34-adam-bench`. Both renders are compiled in
**one process** and their steps **interleaved** (A,B,A,B…) so clock drift hits both equally; the
statistic is the **min**, which is the robust one for a bench. Inputs are byte-identical to the tie,
so the two harnesses measure the same two executables. Run in both compile orders, 20 rounds each:

| | emitted `stablehlo` ops | min ms/step | median | peak device memory |
|---|---|---|---|---|
| A hand-written | 5,971 | 151–152 | 156 | 1.44 GiB |
| B `pretty(proven)` | 10,014 | 151–152 | 156 | 1.44 GiB (4.8 MB **less**) |

**No measurable difference — the delta is 0–1 ms either way, and its sign flips with run order.**

And the mechanism is confirmed rather than inferred. Dumping post-optimisation HLO (see the gotcha
in §4 — `XLA_FLAGS` alone does *not* work here) shows the two modules converge to the same program:

| | A | B |
|---|---|---|
| HLO instrs, before optimisation | 5,901 | 8,768 |
| **`ENTRY` instrs, after optimisation** | **2,690** | **2,691** |
| fusions | 429 | 428 |
| **`rsqrt`** | **36** | **36** |
| triton / gemm / conv calls | 260 / 18 / 214 | 260 / 18 / 214 |
| `dot`, `reduce-window`, `bitcast`, `copy` | 3, 1, 260, 3 | 3, 1, 260, 3 |

The `rsqrt` row is the direct answer to the concern that raised this item: the batched render *emits*
108 = 36 × 3 of them because `bnBatchF`, `bnBatchBack` and `bnGammaGradB` each rebuild x̂
independently, and **XLA deduplicates them back to 36** — identical subgraphs on identical inputs,
exactly as hypothesised. The 1.68× emitted-text ratio is already down to 1.49× at HLO import and to
**1.0004×** after optimisation.

**Conclusion: `pretty`'s lack of CSE (§4) is a codegen-readability problem, not a performance one**,
at least for recomputes this regular. Do not spend effort on emit-side CSE for this net on that
rationale. The one caveat worth carrying: this says XLA *can* undo a redundant recompute of a pure
subgraph on unchanged operands; it is not a licence to assume any emitted redundancy is free.

**Compile time is NOT a differentiator either, though it looks like one.** The first module compiled
in a process takes ~5.8 s and the second ~3.6 s — *whichever render is which*. Swap the argument
order and the numbers swap with it. The ~2.1 s gap is one-time per-process cost, not a property of
the artifact. (This is why the bench takes both paths as arguments.)

#### 2b-ter. The swap ✅ DONE — the trainer now runs the certified render

`verified_mlir/resnet34_adam_train_step.mlir` is now written by
`LeanMlir/Proofs/Codegen/ResNet34RenderB.lean` as `pretty(provenGraph)`, and that `#eval` is its
**only** writer. The driver needed **no change at all** — it resolves the path from the net slug, so
taking over the canonical name *is* the swap. `…_b.mlir` is deleted; the bytes now at the canonical
path are byte-identical to the `_b.mlir` render that passed the tie (checked before deleting).

Gates run for the swap, in order:

| gate | result |
|---|---|
| regenerated canonical == the tied `_b.mlir` | byte-identical ✅ |
| `resnet34-adam-tie` retired-render vs new canonical | forward **bit-exact** 17024/17024, backward norm-rel **2e-6** ✅ |
| `regen_verified_mlir.sh check` | `resnet34_adam_train_step` has **one** writer; no new duplicate ✅ |
| smoke-train, `LEAN_MLIR_G2_STEPS=40` | loss 2.49 → 1.83, val **22.3 → 26.8 → 27.4 → 43.3%** over 4 epochs ✅ |

**The retirement was not clean, and the reason matters.** `tests/TestResnet34Train.lean`'s AdamW
writer was also the *only* producer of the **data-parallel** render — it calls `emitAdamVDP` under
the `REPLICAS` knob (§2c), and **`ResNet34RenderB` has no replica support**. Deleting the writer
outright would have silently removed the multi-GPU capability. So it is retired only at
`REPLICAS = 1`; above that it still renders, to its **own path**
`verified_mlir/resnet34_adam_train_step_dp.mlir`.

That split also fixes a hazard that was live rather than hypothetical: producing a DP render meant
editing `REPLICAS` and re-running the writer, which **overwrote the single-device artifact the
trainer runs** — the §2a last-writer-wins race, on the one artifact that had been declared clean.
The DP render now cannot touch it.

**The sibling race is still live, and this change did not touch it.** `tests/TestResnet34Train.lean`
also writes the **SGD** `resnet34_train_step.mlir`, against `Proofs/Codegen/ResNet34Render.lean`.
Elaborating that file to produce a DP render therefore still clobbers the committed SGD artifact
with the batch-BN spelling — observed while running these gates (md5 `3184522f` → `929074f6`),
restored with `git checkout`. **After any `lake env lean tests/TestResnet34Train.lean`, check
`git diff verified_mlir/`.** Closing it is §2a's remaining four-emitter cleanup, not this thread's.

~~**New named gap: the certified batched renderer cannot emit collectives.**~~ ✅ **Closed the same
day — §2b-quater.**

#### 2b-quater. Multi-GPU brought over ✅ — the collective, as a declared carve-out

Motivated by ImageNet: multi-GPU has to be a supported path, so the DP render was moved onto the
certified renderer rather than retired. `resnet34AdamTrainStepFaithfulB` now takes
`(replicas : Nat := 1)`, and `LeanMlir/Proofs/Codegen/ResNet34RenderB.lean` writes **both**
artifacts — so it is the only writer of either:

| artifact | variant | entry |
|---|---|---|
| `resnet34_adam_train_step.mlir` | `adam` (default) | `@resnet34_adam_train_step` |
| `resnet34_adamdp_train_step.mlir` | `adamdp` | `@resnet34_adamdp_train_step` |

The hand-written AdamW emitter in `tests/TestResnet34Train.lean` (`adamConsts`, `adamCot`,
`trainStepAdamSched`, `bnLayers`, the `REPLICAS` knob — 87 lines) is **deleted**, not repointed: a
second emitter that cannot write is still one more thing to drift. Recover from
`git show b856deb:tests/TestResnet34Train.lean` if ever needed. The SGD `trainStep` is untouched.

**What is and is not verified here.** The insertion is one call to `ViTRender.emitGradAllReduce`
between the gradient SSA and the AdamW triple, so the graph is: *certified gradient →* **trusted
collective** *→ certified AdamW*. The collective is emitted text, not `pretty` of an AST node —
a **declared carve-out**, and the render says so in its own output banner at `replicas > 1`, per
the §5/§2b `%loss` lesson that an undeclared carve-out is how wrong things ship. The `den` side is
untouched: the AdamW ops consume the averaged gradient as an `.operand` exactly as they consumed
the raw one. Claim ceiling is unchanged from §5 — *"the gradient averaging is a proven identity;
the collective implementing it is trusted, exactly like the lowerer."*

Gates run:

| gate | result |
|---|---|
| `replicas = 1` re-render vs the committed artifact | **byte-identical** — the insertion is provably inert on the single-device path ✅ |
| collectives emitted | **146**, one per parameter, matching §2c ✅ |
| syntax | `all_reduce(add)` over `[[0,1]]`, **no `use_global_device_ids`** (§4), then `/2.0` ✅ |
| 2 GPUs, 2 replicas | compiles at 2 replicas, runs, loss descends in both runs — **2.34 → 2.00** at 30 steps/epoch, **2.58 → 2.22** at 10 ✅ |

#### The cifar8 exact gate ✅ — the collective's semantics, pinned

`tests/TestCifar8DpCheck.lean` → `lake build cifar8-dp-check`. cifar8 has **no BN**, so the loss is a
plain mean over examples and the decomposition is an identity:
`(1/2)[(1/128)Σ_A + (1/128)Σ_B] = (1/256)Σ_{A∪B}`. A correct DP step must therefore reproduce the
single-device step at the global batch. `cifar8AdamTrainStepFaithfulV` now takes the same trailing
`replicas` parameter, so both sides come from one renderer:

| | region | norm-rel |
|---|---|---|
| **1×256 vs 2×128 + all_reduce** | gradient (`m`) | **2e-6** ✅ |
| | `θ` | 0 (bit-exact 52441/52858) |
| | `v` | 1e-6 |
| | `%loss` | 1.6e-3 — **not gated, and correctly so** |

The `%loss` gap is the check working, not failing: the DP render computes loss per replica (÷128)
and it is read back from replica 0, so it is that half's mean. **That it differs is independent
evidence the sharding is real** — if both replicas saw the same data it would match exactly. Sharding
real + gradients matching ⇒ the collective genuinely combined both halves.

**The gate was verified to fail**, two ways:

- *collective removed* → never reaches the numbers; the shim's replica-count guard refuses first:
  *"DP invoke asked for 2 replicas but … was compiled for 1 (does the graph contain an all_reduce?)"*
- *collective present but wrong* (`%arn` divisor 2.0 → 1.0, i.e. sum not mean) → **`m` norm-rel
  0.96** against a passing 2e-6. Five and a half orders of separation from the gate.

That second control also confirms §3's rule empirically: on the same broken render **θ moved only
2.7e-4 while `m` moved 0.96**. A θ-based gate would have passed a 2× gradient error.

`replicas = 1` re-renders `cifar8_adam_train_step.mlir` byte-identical, same self-check as R34.

**A guard earned its keep:** the first DP render kept the entry name `@resnet34_adam_train_step`
while the driver asked for `@resnet34_adamdp_train_step`, and the shim's entry-name check refused
the call outright ("entry mismatch") instead of running the wrong graph. The name now follows the
variant.

The per-example route is *not* being taken, but if it is ever revisited: it needs no new ops
(§2a's eight gradients cover it), and its consequences are that the AdamW trainer's BN semantics
change, the `REPLICAS` knob must move out of `tests/`, and — because per-example BN makes
train == eval — `bnChannels` goes empty and `@resnet34_fwd_eval` loses its caller.

### 2c. R34 on 2 GPUs ✅ RUNS — 1.46×, and the shortfall is diagnosed

The DP render emits 146 collectives, one per parameter. Since §2b-quater it comes from the
**certified** renderer (`ResNet34RenderB`, `replicas := 2`) to
`verified_mlir/resnet34_adamdp_train_step.mlir`, selected with `LEAN_MLIR_VARIANT=adamdp`. The
numbers below predate that and were measured with the retired hand-written emitter — the emitted
collective text is the same `emitGradAllReduce` output, but the surrounding graph is not the same
one, so **re-measure before quoting these**. 3 epochs of Imagenette across both 7900 XTXs:

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

0. ~~**Finish §2b's tail**~~ ✅ **DONE** — measured (§2b-bis: no cost), swapped (§2b-ter), multi-GPU
   brought onto the certified renderer and gated (§2b-quater). This thread is closed.
0b. **▶ The last four double-writers — §2a-quinquies.** Zero-behaviour-change cleanup: the committed
   bytes are *already* the certified render for all four, so this only removes clobber hazards.
   Cheapest remaining item by a wide margin, and it takes the audit to 0.
1. ~~**bs256 re-render + measure.**~~ ✅ **DONE 2026-07-28 — 1.78×, gated.** See §2d.1 below.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified end to end.
3. **Device-resident parameters.** Two rounds of transfer work are already done (batching:
   256→205 ms; killing the per-step host memcpys: 205→162 ms). What remains is smaller than it
   looks — see §3.
4. **Executable cache** (`PJRT_Executable_Serialize` / `DeserializeAndLoad`). Worth **0.1%** on an
   R34 training run and **53%** on the MNIST-MLP demo: a dev-loop and CI win, not throughput.

### 2d.1. bs256 ✅ DONE — 1.78× img/s, and the gate that licensed it

`verified_mlir/resnet34_adam256_train_step.mlir`, written by a third `#eval` in
`ResNet34RenderB.lean` at `B := 256`. It renders to its **own** path, so the artifact the trainer
runs is untouched and the §2b tie/bench baselines stay valid; select it with
`LEAN_MLIR_VARIANT=adam256` and `cfg.batchSize := 256`. The eval forwards are still bs32, so train
with `LEAN_MLIR_SKIP_EVAL=1` or re-render them.

`r34AdamVariant B replicas` is now the single source for the entry name, the artifact path and
`LEAN_MLIR_VARIANT`; three `#guard`s pin the literal `#eval` paths against it, so a rename fails at
`lake build` instead of at run time as an "entry mismatch". Note the audit greps for the **literal**
string `IO.FS.writeFile "verified_mlir/`, so those paths must stay literals — do not interpolate them.

**Structurally the re-render is a no-op:** identical to the bs32 artifact at 10014 ops / 9838 lines
with the same op profile (conv 107, dot_general 3, rsqrt 108, reduce_window 1, select_and_scatter 1).
Only tensor dimensions and the mean-CE divisor (32.0 → 256.0) move.

| | ms/step (min) | **ms/image** | ops |
|---|---|---|---|
| bs32 | 150 | 4.69 | 10,014 |
| **bs256** | 674 | **2.63** | 10,014 |

**1.78×**, against the 1.8× the ladder predicted. 8× the batch for 4.49× the step, and it fits on a
7900 XTX with no OOM. Most of the win is amortising the ~272 MB `[θ|m|v]` host↔device round-trip
(§2c) over 8× more images — the transfer is not faster, it is paid less often per image, which is
the same reason device-resident parameters (§2d.3) is the next structural lever.

`tests/TestResnet34AdamBench.lean` now reads the entry name and batch **off each artifact**
(`entryAndBatch`), so one bench compares renders at different `B` with no new argument, and reports
ms/image when the batches differ.

#### The gate — `lake build resnet34-batch-check`, and what it took to make it honest

`tests/TestResnet34BatchCheck.lean`. Feed the bs256 render **8 identical copies** of one bs32 batch.
Then batch-BN μ/σ² over 256 rows are exactly those over the 32, and the mean-CE cotangent divides by
256 against 8× as many terms — so as real numbers the two renders produce the **same** step. All
68,040,737 returned floats must agree.

Measured, and the first two attempts at a gate were both wrong in instructive ways:

| comparison | forward (`bnstat`) | gradient (`m`) |
|---|---|---|
| bs32 vs bs32, same input | **bit-exact** 68040737/68040737 | 0 |
| bs32 vs bs32, batch **reversed** | **bit-exact** | 1e-6 |
| **CONTROL** — bs256 duplicated vs bs256 **regrouped** | 1e-6 | **2.7e-3** |
| **TEST** — bs32 vs bs256 duplicated | 1e-6 | **2.2e-3** |

- *First wrong gate: absolute, at 1e-4.* It failed at 2.9e-3 in the gradient and looked like a
  defect. It is not — §3 already says R34's gradient does not reproduce to better than ~6e-3 under a
  sub-ULP forward nudge.
- *Second wrong gate: calibrate against a reversed bs32 batch.* That control comes back **forward
  bit-exact**, so it measures nothing — the ratio was 0/0 and the gate passed by accident. A control
  that produces no difference cannot calibrate a difference.
- *The control that works:* run the **same bs256 graph** on the **same 256 rows in a different
  order** (each example repeated 8× consecutively, instead of the 32-batch repeated 8×). Same
  multiset ⇒ provably the same real-number result, but the 256-wide reductions accumulate
  differently. That is the same *class* of fp difference the bs32↔bs256 comparison has, measured
  where correctness is not in question.

The answer is then unambiguous: the bs32↔bs256 gradient difference (**2.2e-3**) is *smaller* than
what the bs256 graph produces against **itself** under a semantics-preserving reordering
(**2.7e-3**). So it is this net's conditioning at batch 256, not a defect in the re-render.

The gate is therefore two-sided: **forward absolute ≤ 1e-4** (a plain symmetric mean/variance that
amplifies nothing — a real mis-wiring lands here), and **gradient ≤ 4× the control**, never absolute.

Worth carrying: **32-wide reductions on this net are order-insensitive (forward bit-exact) while
256-wide ones are not** (2.7e-3 in the gradient). Do not assume a reordering control is free just
because it was at bs32.

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

**Refinement of that (2026-07-28, §2b tie).** "Bit-identical for a single step" holds *within* a
process — two executables compiled and run in one process agree to the bit, which the R34 tie's
A-vs-A run confirms on all 68M outputs. **Across processes it is not quite bit-stable**: repeated
runs of the same tie binary on the same artifacts gave a backward norm-relative difference of 1e-6
four times and 2e-6 once. Forward stayed bit-exact every time. Presumably autotuning. So gate with
headroom and quote a bound, not a value.

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

**Guards, and how to use them**

- **`tests/TestBatchedEmitTie.lean`** — two sections: thirteen batched forms tied byte-for-byte to
  their per-example peers, and eight un-fused `*GradB` renders checked to be byte-PREFIXES of their
  fused `*SgdB` peers (the tail being exactly the const-lr/multiply/subtract update — the emit-side
  twin of the `*SgdB_eq_grad` theorems). **Add a case for every new batched form**; it is what
  catches an emitter that reads its width off the SHlo index again. It was verified to actually fail
  by deliberately breaking `relu`'s emit case.
- **A `#eval`-based test must fail via `throw`, not `IO.Process.exit 1`.** Under `#eval` the
  elaborator buffers the eval's output and prints it only after the eval returns, so `exit` discards
  **every** diagnostic — you get a bare non-zero status and no idea what broke. Flushing does not
  help. `tests/TestAdamOpTie.lean` and `tests/TestCifar8AdamTie.lean` still use the `exit` form and
  fail blind.
- **Fastest way to localise a render disagreement**: render the *whole net* both ways into temp
  files and `diff`; then `diff` again with all `tensor<…>` annotations stripped. Structure-identical
  + types-differ pins the breakage to one op family in a single pass. A single-node probe does not —
  that is exactly how §2b's step 1 was mis-estimated.
- **In a numeric tie, run A against itself first.** It establishes the determinism floor (XLA is
  bit-identical for a single step), which is what turns "the difference is 1e-6" from an assertion
  into a measurement. And report **per region**: in `resnet34-adam-tie`, `bnstat` depends only on
  the forward, so it separates a forward disagreement from a backward one in one run.

**Runtime / PJRT**

- **`use_global_device_ids` must NOT be set** on `stablehlo.all_reduce`. It needs a positive
  `channel_id`; without one, compilation fails with *"channel_id must be positive when
  useGlobalDeviceIds is set"*.
- **Multi-GPU needs `HIP_VISIBLE_DEVICES` unset**, or the client sees one device.
  `ffi/test_pjrt_dp.c` and `test_pjrt_allreduce.c` refuse rather than silently running
  single-replica.
- **Empty `compile_options` is not "defaults"** — proto3 zeros give `replica_count = 0` and XLA
  aborts. Hence the generated blob table.
- **`XLA_FLAGS` is silently INERT on this path.** The generated blob embeds a fully-populated
  `DebugOptions`, so it overrides the environment — `XLA_FLAGS=--xla_dump_to=…` produces no dump and
  no warning. The blob captures whatever `XLA_FLAGS` was set when
  `scripts/gen_pjrt_compile_options.py` ran, so to get a dump, regenerate the header with the flag
  set and build a throwaway shim against it, leaving the committed one alone:

  ```bash
  XLA_FLAGS="--xla_dump_to=$D/hlo --xla_dump_hlo_as_text" \
    $VENV/bin/python3 scripts/gen_pjrt_compile_options.py > $D/pjrt_compile_options.h
  cp ffi/pjrt_ffi.c $D/                      # the quoted #include must resolve to $D first
  gcc -fPIC -O2 -shared $D/pjrt_ffi.c -I$D -Iffi -ldl -o $D/libpjrt_ffi.so
  LD_LIBRARY_PATH=$D .lake/build/bin/<binary>   # the shim's rpath is RUNPATH, so this wins
  ```

  Diff the generated header against the committed one to confirm the flag actually landed. Note the
  dump-enabled blob is a *different* `DebugOptions`, so read its HLO for structure, not its timings.
- **`.venv/bin/python3` must be a wrapper, not a symlink.** A symlinked interpreter derives
  `sys.prefix` from the symlink's location and cannot find jax. (Also: `python3 -c 'import jax'`
  from the repo root imports the local `jax/` *directory*.)
- **Checkpoints and `.vmfb` paths are backend-scoped.** They were shared, so an XLA run could
  resume from, or reuse, an IREE artifact while looking normal. Any new driver with resume needs
  the same treatment. They are **variant-scoped too**, so `adam` and `adamdp` do not collide.
- **Check the `.epoch` marker before a long run.** Once, after a run killed mid-epoch (SIGPIPE from
  a `| head` in the invocation), `resnet34_adamdp_ckpt_xla.bin.epoch` held **`80`** — `cfg.epochs` —
  so the next run "resumed" past the end and exited having done nothing, printing only
  *"resuming from checkpoint at epoch 80"*. **Not reproduced**: a clean bounded run writes the
  correct value (`2` after two epochs), and the only writer is `writeFile epPath (toString (ep+1))`
  inside the epoch loop. Cause unknown, so treat it as a thing to *check* rather than a known bug:
  a silent no-op run is an expensive way to discover it at ImageNet scale. `cat` the marker, or
  delete both `ckpt` files, before starting something long.
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
lexing stays audited-but-trusted, which is why every move in §2a and §2b is backed by a numeric tie
against what it replaced, not by the faithfulness theorem alone.

Three places currently emit text that is **not** `pretty` of an AST node, and all say so in the
emitted output: `cifar8_adam_train_step`'s report-only scalar `%loss` and its `%bc` passthroughs,
and `resnet34_adam_train_step_b`'s `%loss`. **That last one shipped wrong** — plain CE instead of
the smoothed CE its own cotangent implies — and only the numeric tie found it, because nothing on a
gradient path touches it and no theorem covers it. Treat every such carve-out as unverified text
that needs its own numeric check, not as a harmless annotation.

A note on what the §2b tie does and does not establish. It says the batched render computes what the
hand-written render computes — forward to the bit, backward to norm-relative 1e-6. It does **not**
say either one is the mathematically intended net; that comes from the `den` side (the faithfulness
theorems, now honest at `N := B`) and from the layer-level VJP oracle in §3. The two halves are
independent, and both are needed: the artifact cannot witness a wrong `den` (the render is
value-independent), and the theorems cannot witness a wrong emitter.
