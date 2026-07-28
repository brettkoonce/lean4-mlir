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
| — | *↓ the AdamW-scorecard thread, 2026-07-28 — ViT, EfficientNet (§2e), ConvNeXt (§2f-bis)* |
| `16fa0f5`, `70b2da3`, `21a09c3` | ViT AdamW: certified render, DP variant, the MIOpen blocker |
| `1119b4f`, `c96bd36` | the depthwise weight gradients — EfficientNet's last blockers |
| `b496fa1` | **EfficientNet AdamW**: `enetBackAll`, the certified render, the bit-exact tie, the swap |
| `b84ae09` | EfficientNet **data-parallel** — gated by the exact identity on the real net, on 2 GPUs |
| `0978cd9`, `3620569`, `9349541` | the DP bench: 1.75× on-GPU, bs128 (1.50×, 93% DP efficiency), and the loader correction |
| `e872ca1` | scoping the last two — ConvNeXt first, and mnv2's real task named |
| `b94e8e9`, `283bb2d` | **ConvNeXt AdamW**: five ops, the render, the conditioning finding, the spread gate, the swap |
| — | *↓ the MobileNetV2 thread (§2f), 2026-07-28 — the last net* |
| `205cd87` | the four batched-index ops: `BatchableOp.relu6`, `selectMidB`, `depthwiseBias{,Strided}GradB` |
| `75a9f8e` | `MobileNetV2RenderB.lean` + the tie — **bit-exact**, three controls fire |
| *(this)* | **the swap — the AdamW scorecard is 6 of 6** |

---

## 0. ▶ START HERE — the AdamW scorecard is **6 of 6**. This thread is CLOSED.

Done 2026-07-28. **cifar8, resnet34, vit, efficientnet, convnext and mobilenetv2** all train on
`pretty(provenGraph)`, each swap licensed by a numeric tie that was verified to fail, and the writer
audit reports one writer per artifact. There is no "next AdamW render".

**What is left in this file is §2d's value-ordered list**, none of it on the AdamW track:
rung 4 (the FPN detector), **device-resident parameters** (two independent multi-GPU measurements
point at it — §2c's 1.46× on R34 and §2e-ter's 13–16% per-step DP overhead on EfficientNet), and the
executable cache. Read §2d before picking one.

Two named gaps stay open and should be quoted whenever the DP renders are described: **ViT's DP
render does not execute on this box** (`miopenStatusUnknownError` in the patch-embed weight-grad
convolution — refuted as a shape or memory problem, see §2a's ViT block), and **ConvNeXt keeps two
weight-gradient carve-outs** (§5).

*The MobileNetV2 write-up below (§2f) is kept because its scoping correction — that mnv2 was
R34-shaped, not EfficientNet-shaped — is the reusable lesson, not because anything is owed.*

The four gates every one of these swaps passed:

1. the SGD artifact re-renders **byte-identical** after the `adam : Bool` threading — ⚠ this one
   **did not transfer to mnv2**, which threads nothing (§2f); there it degraded to "`git diff
   verified_mlir/mobilenetv2_train_step.mlir` is empty", which is strictly weaker because nothing
   forces the two renderers to stay in step. If a seventh net is ever added, check which shape it
   is BEFORE assuming the enet playbook applies;
2. the interface matches the hand-written render **positionally** (arity + arg/return types);
3. a numeric tie with an **A-vs-A determinism floor**, gated per region, that is **verified to fail**
   on a deliberately perturbed render — see §2f-bis for why a green tie you have not tried to break
   is not evidence;
4. after the swap: regenerated canonical == the tied bytes, `regen_verified_mlir.sh check` still at
   one-writer-each, and elaborating the retired test file rewrites nothing.

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

**The AdamW scorecard is 6 of 6** — `cifar8`, `resnet34`, `vit`, `efficientnet`, `convnext` and
`mobilenetv2` all train on `pretty(provenGraph)`, each swap licensed by a numeric tie that was
verified to fail. The writer audit reports **one writer per artifact**. Every whole-net AdamW
render in the repo is now certified; the hand-written emitters are retired to `iree-compile`
smokes that read the committed bytes.

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

# the render ties (§2a, §2b, §2e). The AdamW ones need a GPU; TestBatchedEmitTie is CPU-only.
lake env lean tests/TestBatchedEmitTie.lean            # 13 emit ties + 16 grad-prefix checks
lake build resnet34-adam-tie && .lake/build/bin/resnet34-adam-tie
lake build cifar8-adam-tie   && .lake/build/bin/cifar8-adam-tie
# EfficientNet (§2e) — IREE, not XLA, because efficientnet-verified-adam is an IREE binary
git show c96bd36:verified_mlir/efficientnet_adam_train_step.mlir > /tmp/retired.mlir
lake build efficientnet-adam-tie && IREE_BACKEND=rocm .lake/build/bin/efficientnet-adam-tie \
  /tmp/retired.mlir verified_mlir/efficientnet_adam_train_step.mlir
# ConvNeXt (§2f-bis). Runs its OWN reorder control and gates spread as well as magnitude; pass a
# perturbed render as argv[2] to see it go red, and TIE_XSEED / TIE_REVERSE for the probes.
git show b94e8e9:verified_mlir/convnext_adam_train_step.mlir > /tmp/retired_cnx.mlir
lake build convnext-adam-tie && IREE_BACKEND=rocm .lake/build/bin/convnext-adam-tie \
  /tmp/retired_cnx.mlir verified_mlir/convnext_adam_train_step.mlir
# MobileNetV2 (§2f) — IREE. Gates bnstat (52 BN layers ⇒ the forward is pinned bit-exactly),
# %loss, gradient magnitude AND spread. Deletes its .vmfb before every compile, so re-runs with
# different candidates are safe (unlike resnet34/vit/cifar8/sgd-render — see §4).
git show 75a9f8e:verified_mlir/mobilenetv2_adam_train_step.mlir > /tmp/retired_mnv2.mlir
lake build mobilenetv2-adam-tie && IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-adam-tie \
  /tmp/retired_mnv2.mlir verified_mlir/mobilenetv2_adam_train_step.mlir

# the step-time bench (§2b-bis). Takes both paths so it can be run in either compile order,
# which is the control for the ~2.1 s first-compile-in-process cost.
lake build resnet34-adam-bench
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-adam-bench \
  verified_mlir/resnet34_adam_train_step.mlir verified_mlir/resnet34_adam_train_step.mlir 20

# the data-parallel semantics gates. Need TWO GPUs.
unset HIP_VISIBLE_DEVICES
lake build cifar8-dp-check && PJRT_REPLICAS=2 .lake/build/bin/cifar8-dp-check   # §2b-quater (proxy)
lake build efficientnet-dp-check && PJRT_REPLICAS=2 .lake/build/bin/efficientnet-dp-check  # §2e-bis
#   ^ the exact identity ON THE REAL NET (duplicated batch ⇒ all_reduce/2 is the identity, and BN
#     does not spoil it because both replicas' groups are the same 32 examples). Pass a broken
#     render as argv[1] to run the sum-not-mean control.
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/efficientnet-verified-adam-xla data

# the DP throughput bench (§2e-ter). Interleaved, min statistic, SYNTHETIC inputs so the loader is
# out of it — an end-to-end run measures a DIFFERENT thing (1.67× vs 1.75×) and both are reported.
lake build efficientnet-dp-bench && PJRT_REPLICAS=2 .lake/build/bin/efficientnet-dp-bench 30
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

*The paragraph below is the **snapshot when §2a closed**, kept because it is what the later sections
are measured against. Current state: the writer audit is at **0** (§2a-quinquies), and the AdamW
scorecard is **4 of 6** — cifar8 §2a, resnet34 §2b-ter, vit §2a-quinquies-follow-on, efficientnet
§2e; `convnext` and `mobilenetv2` remain.*

**Still `tests/`-rendered:** `mobilenetv2_fwd{,_eval}`, `efficientnet_fwd{,_eval}`, `convnext_fwd`,
and every `_adam_train_step` except cifar8 **and resnet34** (§2b-ter). Counting writers across all
50 artifacts: **21 `Proofs/`-only, 22 `tests/`-only, 4 contested.**

The `tests/`-only 22 include four whole-net AdamW renders with **live drivers** —
`vit-verified-adam`, `convnext-verified-adam`, `efficientnet-verified-adam`,
`mobilenetv2-verified-adam` all train on hand-written bytes. So the AdamW scorecard is
**2 of 6 certified**.

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

### The four uncertified whole-net AdamW renders — ✅ three done, mnv2 left

*This section is the ViT thread, written when four AdamW renders were hand-written. **All four are
now certified**: **ViT** below, **EfficientNet** §2e, **ConvNeXt** §2f-bis, **MobileNetV2** §2f —
the scorecard is **6 of 6** and nothing is owed. Kept because the ViT write-up below is where the
shared-traversal playbook was worked out.*

Distinct job, much larger, and *not* a prerequisite for the above. `vit`, `convnext`,
`efficientnet`, `mobilenetv2` `_adam_train_step` were hand-written with **live drivers**
(`*-verified-adam`). From what §2b cost at R34 scale:

- **ViT is cheapest, but "only the AdamW tail is missing" was WRONG** — corrected 2026-07-28 by
  measurement. The tail is the *done* part: `emitAdamV` is a proven op family and
  `adamW_triple_faithful` closes the denotation side (§2a). What was actually missing is that **§2a
  un-fused only the CNN gradient family**; ViT's certified backward spends every gradient inside a
  `*Sgd` op, so there was nothing to hand `adamWParamF`. Six ops were needed, now **built and
  gated** (see below). Two further facts worth knowing before picking this up:
  - `verified_mlir/vit_adam_train_step.mlir` is written by **`apps/imagenette/MainViTVerifiedAdam.lean:31`
    — the driver itself, at startup**, from the hand-written `LeanMlir/ViTRender.lean`. That is a
    third writer pattern, distinct from the `tests/` double-writers §2a-quinquies retired: the
    committed bytes are not authoritative because every run overwrites them. The swap therefore also
    means *stopping the driver writing*, so the `Proofs/` `#eval` becomes the sole writer.
  - the audit reports one writer for it today only because the driver is not a `tests/` file; do not
    read that as "clean".

  **✅ Step 1 done — the six transformer gradients are un-fused.** In `Proofs/Codegen/StableHLO.lean`:
  `rowDenseWeightGrad`, `rowDenseBiasGrad` (also carries the vector-LN β), `veclnGammaGrad`,
  `patchEmbedWeightGrad`, `patchEmbedBiasGrad`, `posEmbedGrad`. Each has a `*Sgd_eq_grad` theorem
  (`den (xSgd …) = θ − lr · den (xGrad …)`, all `rfl`) and all six are byte-PREFIXES of their fused
  peers — `tests/TestBatchedEmitTie.lean` is now **13 emit ties + 14 grad-prefix checks**.

  Cheaper than the §2a eight, because all six ride the generic **`.batched`** tag: constructor,
  `den`, the `skel` line, the `emitTok` case, the theorem, the test case — no new `Raw`/`Tok`/
  `toToks`/`parseStack`/`parse_toToks`. Prefer that route (the memory note about 10 sites applies
  only to ops needing their own tag). Verify with `lake build Proofs Certs Codegen`, never bare
  `lake build`, and remember an unmatched `.batched` name falls through to `// MALFORMED` **silently**
  — which is what the prefix test is there to catch.

  **✅ Step 2a done — one backward traversal, two tails.** `vitBackAll (lrStr) (adam : Bool)` in
  `Proofs/Codegen/ViTRender.lean` is the shared depth-12 backward: it returns one SSA per parameter
  in func-arg order — the *updated param* at `adam := false`, the *un-fused gradient* at `true`. The
  ~20 leaf sites dispatch through two small helpers (`rdW`/`rdB`) plus inline `if adam`. Duplicating
  the traversal instead would have been the double-writer disease one level down, in code.
  **`vit_train_step.mlir` came back byte-identical (md5 `f57aff00…`)**, which is what proves the
  refactor inert. `vitParamSig` was added as the single source for the 200 `(name, shape)` pairs —
  the arg signature, the return types, and the `%<nm>m`/`%<nm>v` moment slots.

  **✅ Step 2b done — `vitAdamTrainStepFaithful` exists and its interface matches.** Not yet tied,
  and deliberately **not yet wired to a `#eval`**: the driver still writes that artifact at startup,
  so adding a writer now would create the very double-writer §2a-quinquies just removed. The swap is
  step 3, after the tie.

  | | hand-written | certified |
  |---|---|---|
  | entry / arity | `@vit_adam_train_step`, 605 in / 603 out | **same** |
  | arg + return TYPES, in order | | **identical** ✅ (names differ: `%Wq_0` vs `%b0_Wq`) |
  | emitted `stablehlo` ops | 7,700 | 14,881 (1.9×) |
  | `// MALFORMED` markers | 0 | 0 ✅ |

  The 1.9× is `pretty`'s missing CSE, the same story as R34's 1.68× which §2b-bis measured to cost
  **nothing** after XLA optimisation — do not treat it as a problem without measuring.

  **No new ops were needed for the smoothing after all.** `shiftB`/`divConstB` emit at `ty [B, n]`
  and **ignore their `N`**, so at `N := 1` they are exactly the per-example forms; both are
  POINTWISE, so the §2b `N := 1` hazard (which is about batch-*reducing* ops) does not apply. The
  cotangent is now `vitBackAll`'s `smooth` parameter: `none` → plain CE, mean folded into lr (the
  SGD path, byte-identical); `some (α, −α/K, B)` → `((softmax − onehot) + α·onehot − α/K)/B`.
  One textual difference from the hand-written render, numerically inert: `shiftB` emits
  `add x, dense<−0.01>` where the other emits `subtract x, 0.01` — IEEE subtraction *is* addition of
  the exact negation, so bit-identical.

  **✅ Step 3 done — tied and swapped. The AdamW scorecard is 3 of 6.**
  `verified_mlir/vit_adam_train_step.mlir` is now written by `Proofs/Codegen/ViTRender.lean`'s
  `#eval`, and that is its **only** writer; `MainViTVerifiedAdam.lean`'s `IO.FS.writeFile` is gone
  and the driver now throws if the artifact is missing instead of recreating it. The committed bytes
  are byte-identical to the render that passed the tie (checked before and after the swap).

  `tests/TestViTAdamTie.lean` → `lake build vit-adam-tie`, one AdamW step, all **16,579,041**
  returned floats:

  | check | result |
  |---|---|
  | interface vs the retired render | 605 in / 603 out, arg+return types positionally identical ✅ |
  | A-vs-A determinism floor | **bit-exact 16579041/16579041** ✅ |
  | θ | norm-rel 0, bit-exact 5512498/5526346 ✅ |
  | **gradient (`m`)** | **norm-rel 1e-6**, 0/200 params above 1e-4 ✅ |
  | `v` | norm-rel 1e-6 ✅ |
  | **`%loss`** | **BIT-EXACT 3/3** ✅ |

  **`%loss` is the load-bearing check here, not a footnote.** ViT has no BN, so unlike
  `resnet34-adam-tie` there is no `bnstat` region and *nothing* in the output depends on the forward
  alone — except `%loss`. It is report-only, on no gradient path, and covered by no theorem, which
  is exactly the configuration in which §2b shipped plain CE against a smoothed-CE cotangent. So the
  harness **gates** it rather than reporting it.

  **`vit-adam-tie` links `ireeLink`, not `xlaLink`** — the only tie that does. `vit-verified-adam`
  is an IREE binary, so this graph has never run under XLA/PJRT, and on XLA it dies in the
  patch-embed weight-grad convolution with `miopenStatusUnknownError` (this box is MIOpen-conv-weak
  — see the ROCm note). A tie should run on the backend the trainer actually uses regardless.

  **The data-parallel variant ✅ rendered and gated as far as this box allows.**
  `vitAdamTrainStepFaithful` now takes `(replicas : Nat := 1)` and
  `verified_mlir/vit_adamdp_train_step.mlir` is its own artifact (`LEAN_MLIR_VARIANT=adamdp`,
  entry `@vit_adamdp_train_step`) — rendering to its own path is what stops the §2a race where
  producing a DP render meant editing a knob and clobbering the artifact the trainer runs.

  | gate | result |
  |---|---|
  | `replicas = 1` re-render vs the committed artifact | **byte-identical** — the insertion is provably inert ✅ |
  | collectives emitted | **200**, one per parameter ✅ |
  | syntax | `all_reduce(add)` over `[[0,1]]`, **no `use_global_device_ids`** (§4), then `/2.0` ✅ |
  | carve-out declared in the emitted output | yes, at `replicas > 1` ✅ |
  | `// MALFORMED` | 0 ✅ |
  | **2 GPUs, 2 replicas** | ⛔ **not run — see below** |

  **⛔ Named gap: the ViT DP render cannot be executed on this box — MEASURED on 2 GPUs, not
  inferred.** `tests/TestViTDpCheck.lean` → `lake build vit-dp-check` was written and run
  (`unset HIP_VISIBLE_DEVICES && PJRT_REPLICAS=2`). Result:

  * the graph **compiles fine** — XLA accepts it, 603 outputs, 29 s, with both devices visible. So
    this is not a malformed render or a bad collective;
  * it dies at **execution**, `miopenStatusUnknownError`, *"Failed to enqueue convolution on
    stream"*;
  * and it dies on the **single-device** step, before the DP invoke is ever reached. The blocker is
    therefore the convolution, and **has nothing to do with the collective or the replica count** —
    two GPUs cannot help.

  **The obvious suspect was REFUTED — do not repeat it.** The graph contains exactly two
  convolutions: the forward patch-embed `(32,3,224,224)×(192,3,16,16)/s16 → (32,192,14,14)`, and its
  weight gradient `(3,32,224,224)×(192,32,209,209) → (3,192,16,16)` — a 209×209 filter, which looked
  like an obvious MIOpen limitation. It is not: a minimal JAX repro of that exact convolution
  (`scratchpad/miopen_repro.py`, jax 0.10.0 + rocm plugin, same box) **runs fine and returns the
  right shape**. So this is *not* a shape MIOpen cannot handle.

  **The memory hypothesis is REFUTED too** (`scripts/miopen_mem_probe.py`). Holding a ballast tensor
  and re-running the same conv: it succeeds at 0, 4, 8 and **12 GiB** of ballast on top of its own
  1.02 GiB of inputs. And when memory genuinely runs out (14 GiB ballast) XLA reports a **clean
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 14.00GiB`** — a completely different,
  well-formed error. So `miopenStatusUnknownError` is not an allocation failure wearing a disguise.

  **Strongest remaining lead: XLA fuses the dilation into the convolution, changing the descriptor.**
  The render emits `stablehlo.pad` with `interior = [0,0,15,15]` and then a *stride-1* conv against
  the materialised 209×209 filter — which is what both probes reproduce, successfully. XLA is free
  to fuse that pad into the convolution instead, as `rhs_dilation = 16` over the original 14×14
  cotangent, which is a **different MIOpen call** that the isolated repro never makes. That would
  explain precisely why the op works alone and fails in the graph.

  To confirm: dump post-optimisation HLO for this executable and read the actual `convolution`
  descriptor. **`XLA_FLAGS` is inert on this path** — the compile-options blob overrides it; §4 has
  the regenerate-a-throwaway-shim recipe, which §2b-bis already used successfully for exactly this
  kind of question. If the fused descriptor is the culprit, that IS the filable bug, and the repro
  is a two-op MLIR module rather than anything ViT-specific.

  `vit-adam-tie` links IREE for the same reason, and there the identical graph runs fine, which
  localises this to the XLA/MIOpen path rather than to the render.

  **No bug has been filed and none should be until there is a self-contained repro.** The isolated
  convolution does not reproduce it, so a report today would point maintainers at the wrong thing.

  `vit-verified-adam` is an IREE binary besides, where the shim refuses the DP entry point outright
  rather than silently running single-device.

  **The gate is written and will pass the moment the graph runs.** `vit-dp-check` uses an identity
  R34 could never use: ViT has **no BatchNorm**, so giving both replicas the *same* batch makes
  `all_reduce(add)/2` the identity, and the DP step must reproduce the single-device step exactly —
  the cifar8-grade exact check (§2b-quater), on the real net rather than a proxy. Run it on ares, or
  after rerouting that convolution.

  So this is **strictly weaker than R34's §2b-quater**, which had a 2-GPU descent run plus the
  cifar8 exact decomposition gate behind it. Do not describe ViT multi-GPU as working. What is
  established is that the emitted text is the same collective R34 uses, inserted at the same place,
  and provably inert at one replica. To finish it: run on a box whose conv library handles the
  patch-embed weight-grad (ares), or route that one convolution differently.

  Why the gap below was worth recording — the two cotangents WERE different functions, so a render
  built on `vitBackAll` without `smooth` would have failed the tie in a way that looks like a bug in
  the new gradient ops:

  | | certified SGD render | hand-written AdamW (what `vit-verified-adam` trains on) |
  |---|---|---|
  | cross-entropy | plain `softmax − onehot` | **label-smoothed**, α = 0.1: `+α·onehot`, then `−α/K` (`%lsa` 0.1, `%lsaik` 0.01) |
  | batch mean | folded into lr (0.003125 = 0.1/32) | **explicit** `divide %dyr, dense<32.0>` |

  So step 2b needs the smoothing chain at the ViT **per-example** index (`Vec 10`, emitted
  `tensor<32x10>`). R34 got exactly this in §2b as `softmaxRow → subB → scaleB → addVB → shiftB →
  divConstB`, but those are the **batched** forms; check which per-example peers already exist
  (`.scaleF` and `.addV` do) and add the missing shift / divide-by-constant. Two ops at most, same
  `.batched`-tag recipe as the six above.

  Then: assemble `vitAdamTrainStepFaithful` (605 in / 603 out — `%x`, 200 θ, 200 m, 200 v, `%lr`,
  `%bc1`, `%bc2`, `%onehot` → 200 θ', 200 m', 200 v', loss, bc1, bc2; the hand-written arg ORDER is
  positionally identical to `vitParamSig`, only the names differ), copy `adamOne`/`adamConstsB` from
  `ResNet34RenderB.lean`, and tie.

  **Two traps for step 2b/3, both already paid for elsewhere:**
  - `%loss` is report-only and on no gradient path, so **no theorem covers it** — this is precisely
    where §2b shipped plain CE against a smoothed-CE cotangent and only the numeric tie caught it.
    Render it from the *same* smoothed chain the cotangent implies.
  - ViT has **no BN**, so unlike R34 there is no `bnstat` region to pin the forward bit-exactly.
    Gate on the gradient (`m`) and the loss, and expect the same limitation the SGD ties have.
- **EfficientNet ✅ DONE 2026-07-28 — the AdamW scorecard is 4 of 6.** See §2e below; the tie came
  back **bit-exact on all 12,166,117 returned floats**.
- **MobileNetV2** needs those same depthwise gradients, so it comes free-ish after EfficientNet.
- **ConvNeXt** carries the 4 even-kernel gaps from its §1a tie; check whether they touch the
  backward before scoping.

The R34 playbook that worked: un-fuse the gradients → render AdamW from the proven ops → numeric
tie against the hand-written render → bench → swap and retire in one change.

### 2e. EfficientNet-B0 AdamW ✅ DONE — scorecard 3 of 6 → **4**, and the tie is BIT-EXACT

Done 2026-07-28, in the R34/ViT order: un-fuse → one shared backward → assemble → tie → swap.
`verified_mlir/efficientnet_adam_train_step.mlir` is now written by
`Proofs/Codegen/EfficientNetRender.lean`'s `efficientnetAdamTrainStepFaithful`, and that `#eval` is
its **only** writer. `efficientnet-verified-adam` needed **no change** — it resolves the path from
the net slug, so taking over the canonical name *is* the swap.

**Step 1 — `enetBackAll`, one traversal, two tails.** `adam : Bool` threads through the six backward
functions the plan named (`seBack`, `eBackBody`, `eBack`, `eBackNoSkip`, `eBackStrided`,
`eBackNoExp`) plus the head and stem tails, which had to come out of
`efficientnetTrainStepFaithfulV` into a shared `enetBackAll` — the `vitBackAll` shape. The ~20 leaf
sites dispatch through six small helpers (`bnG`, `bnBt`, `convW1`, `dwW`, `dwWS`, `dnW`, `dnB`).
**`efficientnet_train_step.mlir` came back byte-identical (md5 `f17aef2c…`)** at every step, which
is what proves the refactor inert.

*Worth keeping:* `bnBt` covers **every conv bias in the net** as well as BN β, because
`Σ_{batch,spatial} dy` *is* a conv bias gradient. That reuse is why the scoping note "EfficientNet
needs no depthwise bias peer" holds — the depthwise convs are followed by BN, so their bias is
folded, and `bnBetaGradB` serves the rest.

**Step 2 — the render.** Two things differ from the SGD render and both are load-bearing:

| | certified SGD render | AdamW render |
|---|---|---|
| cross-entropy | plain `softmax − onehot` | **label-smoothed** α = 0.1, composed `scaleB → addVB → shiftB → divConstB` |
| batch mean | folded into lr (0.05 ⇒ effective 1.6) | **explicit** `divConstB "32.0"` |
| BN running stats | none | **98 outputs** — `bnBatchMeanB`/`bnBatchVarB` per BN layer |

No new ops were needed. The BN stat layout is derived from the **same forward traversal that
computes them** (each `EFwd` carries its own `bns` list) rather than from an independent 49-entry
table — a misaligned stat slot is silent, since the arities still match and the wrong layer's
statistics simply flow into the wrong `@efficientnet_fwd_eval` slot.

| | hand-written | certified |
|---|---|---|
| entry / arity | `@efficientnet_adam_train_step`, 889 in / 887 out | **same** |
| arg + return TYPES, in order | | **identical** ✅ (only the 98 unused BN slot names differ) |
| emitted `stablehlo` ops | 10,421 | 17,545 (1.68×) |
| `// MALFORMED` | 0 | 0 ✅ |

The 1.68× is exactly R34's ratio, which §2b-bis measured to cost **nothing** after optimisation.

**Step 3 — `tests/TestEfficientNetAdamTie.lean` → `lake build efficientnet-adam-tie`.** One AdamW
step, all **12,166,117** returned floats:

| check | result |
|---|---|
| interface vs the retired render | 889 in / 887 out, arg+return types positionally identical ✅ |
| A-vs-A determinism floor | **bit-exact 12166117/12166117** ✅ |
| **forward** (`bnstat` = batch μ/var of all 49 BN inputs) | **BIT-EXACT 42016/42016** ✅ |
| **`%loss`** | **BIT-EXACT** ✅ |
| **gradient (`m`)** | **BIT-EXACT**, 0/262 params above 1e-4 ✅ |
| θ, `v` | bit-exact ✅ |

**Bit-exact, not 1e-6 — and the reason is the backend, so do not generalise it.** This tie links
**IREE** (`efficientnet-verified-adam` is an IREE binary, same as `vit-adam-tie`), where the
pipeline is deterministic; R34's ≤2e-6 spread is XLA autotuning picking different convolution
algorithms across processes (§3). A bit-exact result here does not predict a bit-exact one on XLA.

**Three negative controls, because a tie that reports bit-exact everywhere has to be shown capable
of failing.** All three were rendered from the same renderer with one thing perturbed:

| control | perturbation | result |
|---|---|---|
| A | cotangent α 0.1 → 0.11 | gradient gate **fires** at 2.6e-3, 112/262 params; forward stays bit-exact |
| B | BN ε 1e-5 → 1e-4 (forward) | forward gate **fires** — `bnstat` bit-exact only **76/42016**, and those 76 are the stem BN, the one layer upstream of any ε effect |
| C | `%loss` constant 0.9 → 0.8 | loss gate **fires** at 0.1 with every other region bit-exact |

Control A is also a clean re-demonstration of §3's rule: on a wrong gradient **θ reported norm-rel
0.000000** while `m` moved 2.6e-3. A θ-based gate would have passed it.

#### ⚠ A trap this thread found in the tie harnesses — it produced a false PASS

`mkSession`/`compileVmfb` reuse any existing `.vmfb` **newer than the `.mlir`**. The cache key is
the *output* path and an mtime, not the source. So running a tie twice with different candidates
under the same tag silently reuses the **first** candidate's binary and reports the second as a
perfect match. That is exactly what happened while building control B above: it printed
`(cached vmfb)` and reproduced control A's numbers to the digit, for a render that had never been
compiled.

`efficientnet-adam-tie` now **deletes the target `.vmfb` (both the bare and the `_$IREE_BACKEND`
scoped name) before every compile**. `resnet34-adam-tie`, `vit-adam-tie`, `sgd-render-tie` and
`cifar8-adam-tie` still have the hazard — it only bites when the same binary is run more than once
with different arguments, which is precisely what running a negative control looks like. Their
committed results are not in doubt (each compiled fresh into an empty `.lake/build`), but **check
for `(cached vmfb)` in the output before believing any re-run.**

#### 2e-bis. Data-parallel EfficientNet ✅ DONE — and it is the **best-gated multi-GPU render here**

`efficientnetAdamTrainStepFaithful` takes `(replicas : Nat := 1)` and
`verified_mlir/efficientnet_adamdp_train_step.mlir` is its own artifact (`LEAN_MLIR_VARIANT=adamdp`,
entry `@efficientnet_adamdp_train_step`), so producing a DP render can never clobber the one the
trainer runs. Unlike R34's, this variant never had a hand-written emitter to migrate off — the
certified renderer is the only writer of both EfficientNet AdamW artifacts from the start.

**It runs on XLA/PJRT, which was not a given.** ViT's DP render is still unexecutable on this box
(`miopenStatusUnknownError` in the patch-embed weight-grad convolution), so this was measured before
anything was built: `efficientnet-verified-adam-xla` — a new target, the shared-body split
`Resnet34AdamCommon` already uses — compiles in 20.5 s and reproduces the IREE run within fp noise
(val 14.98/21.83/27.13% vs IREE's 15.11/21.58/27.97% over 3 epochs). EfficientNet is depthwise
convolutions throughout; that MIOpen handles them and not ViT's 209×209 patch-embed weight-grad
narrows the ViT blocker further.

| gate | result |
|---|---|
| `replicas = 1` re-render vs the committed artifact | **byte-identical** — the insertion is provably inert ✅ |
| collectives emitted | **262**, one per parameter ✅ |
| syntax | `all_reduce(add)` over `[[0, 1]]`, **no `use_global_device_ids`** (§4), then `/2.0` ✅ |
| carve-out declared in the emitted output | yes, at `replicas > 1` ✅ |
| `// MALFORMED` | 0 ✅ |
| single-device artifact contains a collective | **0** ✅ |
| **2 GPUs, 2 replicas — the exact identity** | **PASSES** ✅ (below) |
| 2-GPU descent | loss 2.317 → 2.278 → 2.246, val 14.3 → 17.5 → **21.5%** ✅ |

**`tests/TestEfficientNetDpCheck.lean` → `lake build efficientnet-dp-check`.** Give both replicas the
**same** 32 examples: each computes the same gradient `g`, so `all_reduce(add)/2 = (g+g)/2 = g` is
the identity and the DP step must reproduce the single-device step.

**BatchNorm does not spoil this, and that is worth being precise about.** The §10.3b caveat that
blocked an exact gate for R34 is about **splitting** a batch — 2×32 genuinely is not 1×64, because
the halves get different statistics. **Duplicating** a batch is the other case: BN normalises per
replica, both replicas' groups are the same 32 examples, so their statistics are identical by
construction. R34's collective had to be gated on cifar8 as a proxy; this one is gated **on the real
net**.

| region | result (2× 7900 XTX, `PJRT_REPLICAS=2`) |
|---|---|
| **`bnstat`** (forward, 49 BN layers) | **BIT-EXACT 42016/42016** ✅ |
| **`%loss`** | **BIT-EXACT** ✅ |
| `v` | **BIT-EXACT 4041366/4041366** ✅ |
| **gradient (`m`)** | norm-rel **3.3e-8**; 14 of 4,041,366 coordinates differ, max abs **1.9e-9** ✅ |
| θ | norm-rel 1e-12, 13 coordinates differ |

That is ~3000× inside the 1e-4 gate. It is *not* bit-exact on `m`/θ and the harness says so —
`Float.toString` gives six decimals, so a genuine 3e-8 prints as `0.000000` and reads as bit-exact.
The gate now also reports nano-units and the exact-coordinate count. The residue is not the
collective: `(g+g)/2 = g` holds to the bit in binary floating point. It is that the DP module is a
**different HLO program**, so XLA orders the backward reductions differently. Consistent with that,
the forward is untouched — `bnstat` is bit-exact — and the count varies run to run (4041310 then
4041352 of 4041366 on identical inputs), which is §3's cross-process autotuning again.

**Verified to fail.** `%arn` divisor 2.0 → 1.0 (sum, not mean; 262 lines changed) → gradient
norm-rel **0.246** against a passing 3.3e-8, seven orders of separation. It also re-demonstrates §3
on a real net: the same broken render moved **θ by only 7e-6** while `m` moved 0.246, and `bnstat`
stayed bit-exact — a θ-based gate would have passed a 2× gradient error outright.

#### 2e-quinquies. Wall clock — **80 epochs ≈ 1 h 35 m on XLA**, and how to measure an epoch

Marginal epoch, `(T₃ − T₁)/2`, which cancels compile, the one-time dataset load and startup:

| | marginal epoch | 80 epochs |
|---|---|---|
| XLA, 1 GPU, bs 32, train + eval | **71.1 s** | **≈ 1 h 35 m** |
| XLA, 1 GPU, bs 32, train only | 65.8 s | ≈ 1 h 28 m |
| XLA, 2 GPU, global 64, train only | 39.5 s | ≈ 53 m |
| IREE, 1 GPU, bs 32, train + eval | 354 s | ≈ 7 h 50 m |

Eval costs **5.3 s/epoch** (3925 images at bs 32 through `@efficientnet_fwd_eval`). **XLA is 4.6×
IREE** on this net, so use the `-xla` binary. Cross-check: `runs/efficientnet_verified_crop_gpu1.log`
ran 8 h 30 m for 80 epochs on IREE with mobilenetv2 training concurrently on the other GPU —
consistent with 7 h 50 m solo.

**Always take the marginal difference, not wall-clock-minus-compile.** The latter still carries the
~7–12 s dataset load and produced a wrong DP ratio twice in this thread (§2e-ter).

#### 2e-ter. The throughput bench — **1.75× on-GPU, 1.67× end-to-end**

`tests/TestEfficientNetDpBench.lean` → `lake build efficientnet-dp-bench`. **This supersedes the
~1.23× first quoted here**, which came from two short end-to-end runs and was contaminated by data
loading and process startup — precisely §3's standing trap, and it understated the GPU result by
30%. Do not quote 1.23× as a scaling figure.

Method is §2b-bis's: both executables compiled in ONE process, steps **interleaved** A,B,A,B so
drift hits both equally, **min** statistic, and **synthetic in-process inputs** so the loader is out
of the measurement. (One process holding a 1-replica and a 2-replica executable is fine — the
replica count is per-GRAPH, §2c.) Two runs, 20 and 30 rounds, on 2× 7900 XTX:

| | ms/step (min) | **ms/image** |
|---|---|---|
| A — 1 GPU, bs 32 | 209–211 | 6.53–6.59 |
| B — 2 GPU, global 64 | 238–243 | **3.72–3.80** |

* **On-GPU throughput 1.72–1.77×**, i.e. **86–89% parallel efficiency**. Perfect is 2.00×, because
  each replica of B does exactly the work A does — so with zero overhead B's ms/*step* would equal
  A's at twice the images.
* **DP overhead 27–34 ms/step = 13–16%** of the single-GPU step. That covers 15.4 MiB all-reduced
  plus the 46.2 MiB `[θ|m|v]` pushed to the *second* replica, ≈ **1.8 GiB/s** effective — consistent
  with PCIe, and it is why device-resident parameters (§2d.3) is the lever: the `[θ|m|v]` half of
  that grows with the replica count and the collective cannot amortise it.

**Beware the Amdahl form.** Writing the serial share as `s = 2·T_B/T_A − 1` returns **125%** here —
nonsense, because it models B as halving A's work. B does not: each replica does A's work, on twice
the data. The cost of going data-parallel is the step-time excess `T_B − T_A`, full stop.

**End-to-end, with the real loader: 1.67×.** ⚠ **This corrects a 1.43× first recorded here**, which
was measured as *total wall clock minus compile* on a single run and therefore still carried the
**one-time dataset load** — Imagenette is read from disk once and expanded to ~7.45 GiB of f32 in
host RAM, ~7–12 s. The right measurement is the **marginal** epoch, `(T₃ − T₁)/2`, which cancels
compile, dataset load and process startup exactly:

| | marginal epoch (train only) | on-GPU part | **host overhead** |
|---|---|---|---|
| 1 GPU, bs 32 | **65.8 s** | 295 × 209 ms = 61.7 s | 4.1 s = **6.3%** |
| 2 GPU, global 64 | **39.5 s** | 147 × 238 ms = 35.0 s | 4.5 s = **11.3%** |

**So the loader is NOT the dominant lever, and the earlier "21% / 34%" figures here were wrong** —
they were one-time dataset loading misattributed to per-step work. The real per-epoch host cost is
**~4.1–4.5 s and roughly constant**, which is what it should be: it is proportional to images, not
to GPU count, so it takes a larger *share* of the shorter 2-GPU epoch (6.3% → 11.3%) without
growing. That constant is also the entire gap between the on-GPU 1.75× and the end-to-end 1.67×.

**What that 4.3 s actually is** — no disk I/O; the dataset is resident. Per epoch,
`F32.shuffle` Fisher-Yates over the whole ~7.45 GiB array (in place: the pristine copy used to stay
live and leak one training set per epoch, see the comment at `VerifiedTrain.lean:471`). Per step,
`sliceImages` → `randomCrop` 256²→224² → `randomHFlip`, about **64 MB of host memcpy per bs32
step**, all of it serial with the GPU — there is no prefetch and no overlap. At ~4.3 s per 9469
images that pipeline is running at roughly 4 GB/s, which is unremarkable but not pathological.

Worth doing eventually (overlapping it would recover ~6% single-GPU, ~11% at two), but **it is not
where the time is**, and this is the second time in this thread that a wall-clock-minus-compile
measurement has produced a wrong ratio. Use the marginal-epoch difference.

#### Memory: **4.7 GiB at bs32, 17.3 GiB at bs128** — and why `rocm-smi` says 19

Peak VRAM sampled during training reads **18.7–19.0 GiB per GPU under XLA/PJRT at every batch**,
which is the tell that it is *not* the model. `ffi/pjrt_ffi.c:38` documents it — *"each
StreamExecutor GPU client reserves ~19 GB for its BFC allocator"* — i.e. XLA pre-reserves ~78% of
the card at client creation regardless of what the graph needs, so the figure is pinned to the
reservation and carries no information. **Never quote it as a memory requirement.** Measure on IREE,
which does not pre-reserve:

| batch | working set | share of the 23.98 GiB card |
|---|---|---|
| 32 | **4.67 GiB** | 19% |
| 128 | **17.30 GiB** | 72% |

4× the batch costs 3.7× the memory — close to linear, because activations dominate and only the
~140 MiB of `[θ|m|v]` is fixed. **That makes bs128 the practical ceiling on a 7900 XTX**: bs256
extrapolates to ~30 GiB and will not fit. (XLA ran bs128 inside its 19 GiB reservation, so its
planner is at least as tight as IREE's.)

Batch is baked into the render, not a runtime dimension, so each one is its own artifact —
`LEAN_MLIR_BATCH` selects the batch and **must match the variant**, or the first invoke is a shape
error. The eval forwards are still bs32, so anything else needs `LEAN_MLIR_SKIP_EVAL=1`.

Claim ceiling is unchanged from §5: the graph is *certified gradient → trusted collective →
certified AdamW*. What is new is that the collective's semantics are now pinned by an exact
known-answer check **on the net itself**, not on a no-BN proxy.

### 2f-bis. ConvNeXt-T AdamW ✅ DONE — scorecard 4 of 6 → **5**, and a gate lesson

Done 2026-07-28, following §2e's playbook. `verified_mlir/convnext_adam_train_step.mlir` is now
written by `Proofs/Codegen/ConvNeXtRender.lean`'s `#eval`, its only writer; the hand-written emitter
in `tests/TestConvNeXtTrain.lean` is retired to smoke-only. The driver needed no change.

**Step 1 — five ops.** `depthwiseWeightGrad`, `depthwiseBiasGrad`, `lnGammaGrad`, `lnBetaGrad`,
`layerScaleChGammaGrad`, each with a `*Sgd_eq_grad` `rfl` theorem and a byte-PREFIX case
(`TestBatchedEmitTie` is now 13 emit ties + **21** grad-prefix checks). All ride the generic
`.batched` tag — the four-site route. `depthwiseWeightGrad` deliberately **aliases the batched op's
tag**, distinguished by ARITY (five nats vs six): the emitted text is identical because that emitter
ignores its `N`, so only `den` differs — the same aliasing `convStridedBiasGrad` does against
`convBiasGrad`. The prefix guard paid for itself immediately: `depthwiseBiasGrad` emitted its zero
constant *before* the reshape where the fused op does it after — equivalent MLIR, not a prefix.

**Step 2 — the threading was as cheap as §2f predicted.** ConvNeXt's param tails were already
factored out of the cotangent traversal, so `adam` goes through `blockParamSgd`/`downParamSgd` and
`bwdBlock` is untouched. `convnext_train_step.mlir` re-renders **byte-identical**. Interface: 545 in
/ 543 out, arg *and* return types positionally identical — and here even the parameter names agree.
10,407 emitted ops against 7,488.

#### ▶ The gate lesson: this tie is where an absolute bound would have been wrong, AND where the
obvious fix would have passed a real bug

The first run failed: `%loss` bit-exact, 179 of 180 parameter gradients bit-exact, and **one**
parameter — `s3b2lg`, the last block's per-channel layer-scale γ — off at norm-rel 6.9e-3.

*Not the change.* Both certified renders (SGD and AdamW) compute the identical chain for it, so the
un-fusing is inert; the disagreement is against the hand-written emitter and predates this work.

*Not a formula difference either.* Three probes:

| probe | result | reading |
|---|---|---|
| three input seeds | max abs err pinned at 3.2e-4 while values move; `Σ\|b\|/Σ\|a\|` = 1.00054 / 0.99987 / 0.99993 | straddles 1 ⇒ no consistent bias |
| sign flips | **0/768** | not a sign/formula error |
| cancellation | `\|Σa\|/Σ\|a\|` = 0.094 across channels, worse within one | a cancelling reduce |

`lg` is **the only parameter in the block whose gradient reads a forward VALUE** (the project
output) rather than a cotangent — everything else reads `g`, `n`, `d` or a cotangent. So a tiny
forward difference lands only there, amplified by the cancellation.

**The control that settles it.** Run the reference render against **itself** on the same batch in
reversed row order. Every gradient here sums over the batch, so that is semantics-preserving as real
arithmetic but changes the accumulation order:

| | gradient norm-rel | params disturbed |
|---|---|---|
| TEST — certified vs hand-written | 0.003772 | **1**/180 |
| CONTROL — hand-written vs itself, batch reversed | 0.003690 | **6**/180 |

The two emitters agree **better** than the reference agrees with itself under a reordering that
cannot change the answer. So the gate is calibrated against the control (`≤ 4×`), never absolute —
§2d.1's rule, now with a second net behind it.

**But the obvious fix is not enough, and this is the part worth keeping.** With only that magnitude
gate, a **deliberately perturbed cotangent** (α 0.1 → 0.11) **PASSES**: it lands at 9.1e-3, under
4 × 0.00369 = 1.48e-2. What separates them is not magnitude but **spread** — it disturbs **178/180**
parameters where the reorder control disturbs 6, and where the real tie disturbs 1:

> Floating-point conditioning is **local** to the ill-conditioned op. A different function is
> **global**. Gate the spread as well as the magnitude, or a control-relative bound will wave a real
> cotangent bug through.

The harness therefore gates three things: `%loss` (absolute, 1e-4 — it is the only forward-only
output on a net with no BN), gradient magnitude ≤ 4× the reorder control, and gradient spread ≤ the
control's. Verified to fail on both controls: the cotangent perturbation fires the **spread** gate,
a `%loss`-constant perturbation fires the **loss** gate.

**Claim ceiling.** ConvNeXt keeps its two documented weight-grad gaps — the stem 4×4/s4 patchify and
the even-kernel 2×2/s2 downsample, neither of which has a VJP-cert `SHlo` op — and those stay
hand-written. But the SGD render also hand-wrote the *update* for exactly those params (the `sgd`
helper); this render replaces it with the proven AdamW triple. So for `psW` and the three `d{i}W`
the tier moves from *hand-written gradient + hand-written update* to *hand-written gradient +
certified update*, and the other 176 params are `pretty(AST)` end to end. A strict improvement.

### 2f. ▶ The last AdamW render — MobileNetV2. Scoped by measurement; ConvNeXt went first ✅

Scoped 2026-07-28 by enumerating every `*Sgd` op each renderer uses and checking for a `*Grad` peer,
then checking two things that decide the order.

| | ConvNeXt | MobileNetV2 |
|---|---|---|
| renderer size | **255 lines** | 583 lines |
| interface | 545 in / 543 out (180 params) | 739 / 737 (212 params) |
| `*Grad` peers present | 5 of 10 | 8 of 12 |
| **missing ops** | `lnGammaGrad`, `lnBetaGrad`, `layerScaleChGammaGrad`, `depthwiseWeightGrad`, `depthwiseBiasGrad` | `depthwise{,Strided}{Weight,Bias}Grad` |
| backward structure | **param-SGD already factored out** (`blockParamSgd`/`downParamSgd` are separate from `bwdBlock`) | enet's exact shape (`irBack{,Strided,NoExp,NoSkip}`) |
| BatchNorm | **none** (LayerNorm) ⇒ ViT-grade tie | yes, 52 layers ⇒ enet-grade tie |
| cotangent gap | label-smoothed + explicit ÷B (same shape as ViT/enet) | same |

**The decider is a semantic fork in mnv2, not op count.** `MobileNetV2Render.lean` renders at the
**per-example** index (`.bnGammaSgd`, `.relu6F`, `.addV`, `.gapF` — all non-`B` forms), while
`verified_mlir/mobilenetv2_adam_train_step.mlir` — what `mobilenetv2-verified-adam` trains on — is
**batch BN** (52 `rsqrt`, 364 `reduce[0,2,3]`, the saved-`%xh` spelling). **That is exactly the R34
§2a two-worlds situation**, and resolving it is the §2b batched-index move, which was the single
most expensive and most badly mis-estimated step in this whole thread (planned as "a one-line
re-instantiation", actually 20 new codegen forms and 597 changed lines). Most of those forms now
exist, but mnv2 additionally needs a batched `relu6` and `depthwise{,Strided}BiasGradB` — enet
needed neither, because its depthwise convs are followed by BN so the bias is folded.

**So the honest name for the mnv2 task is "MobileNetV2 at the batched index", not "MobileNetV2
AdamW".** Scoping it as the latter is how the estimate goes wrong again.

**ConvNeXt had no such fork** — done, §2f-bis. Its param-SGD was already separate from the
cotangent traversal, three of its five missing ops were plain reductions, and its SGD tie was
already gradient-bit-exact, which is why it went first.

#### ▶ The MobileNetV2 work order

**Decide the semantics first — everything else follows from it, and it is the only irreversible
choice.** The certified SGD render and the AdamW trainer are two different functions today
(per-example BN vs batch BN). Two ways out, and R34 has already paid for one of them:

* **(a) move mnv2 to the batched index `N := B`** — what §2b did for R34. Keeps the trainer's
  semantics, so the accuracy story and the 80-epoch logs stay valid. This is the recommended one,
  and it is what "MobileNetV2 at the batched index" means.
* **(b) move the trainer onto the per-example chain** — cheaper in codegen, but it changes what
  `mobilenetv2-verified-adam` computes, voids its published accuracy, and (as §2b-quater notes for
  R34) makes train == eval so `bnChannels` goes empty and `@mobilenetv2_fwd_eval` loses its caller.

#### ▶ Scoped by measurement 2026-07-28 — mnv2 is **R34-shaped, not EfficientNet-shaped**

The instruction above to "follow §2e's playbook" (thread `adam : Bool` through ONE renderer, both
artifacts out of one file) **does not transfer**, and the BN reduce-dim census is why:

| renderer's SGD artifact | `reduce[2,3]` | `reduce[0,2,3]` | `rsqrt` | BN world |
|---|---|---|---|---|
| `efficientnet_train_step` | 81 | 539 | 147 (= 49×3) | **batch** |
| `resnet34_train_step` | 289 | 108 (= 36×3) | 108 | **per-example** |
| `mobilenetv2_train_step` | **417** | 156 (= 52×3) | **156** | **per-example** |

EfficientNet's SGD render was *already* batch-BN, which is the only reason §2b's `N := B` move came
back byte-identical there and one file could serve both tails. **mnv2's SGD render is per-example**,
so re-instantiating `MobileNetV2Render.lean` at `N := B` changes `mobilenetv2_train_step.mlir`'s
bytes *and its function* — voiding the 80-epoch SGD log (`runs/mobilenetv2_verified_crop_gpu0.log`,
86.89%). That is the §2a two-worlds trap a second time, and taking the enet shape walks into it.

**So the batched AdamW render goes in a NEW `MobileNetV2RenderB.lean`**, exactly like
`ResNet34RenderB.lean`, leaving `MobileNetV2Render.lean` untouched. Two consequences:

- **Gate 1 changes meaning.** "the SGD artifact re-renders byte-identical after the `adam : Bool`
  threading" (§0) was written from the enet/convnext playbook. Here there is no threading, so the
  gate degrades to *"`git diff verified_mlir/mobilenetv2_train_step.mlir` is empty"* — strictly
  weaker, because nothing forces the two renderers to stay in step. Note it as such; and §2b-ter's
  hazard applies, a second file writing a sibling artifact is exactly how R34's SGD render still
  gets clobbered today.
- **The fused `*SgdB` tail is NOT needed.** `ResNet34RenderB.lean` is AdamW-only (three `#eval`s,
  all `_adam*`), so the new file needs only the un-fused `adam := true` gradients. Do not build
  `depthwise{,Strided}BiasSgdB`.

**Verified target interface:** `@mobilenetv2_adam_train_step`, **739 in / 737 out**, **210 params**,
**52 BN layers**, 104 running-stat slots in *and* out. The arithmetic closes exactly —
`210×3 + x + lr + bc1 + bc2 + onehot + 104 = 739` and `630 + loss + bc1 + bc2 + 104 = 737` — so a
slot-count mistake shows up as an arity error rather than silently. (The 212/210 in §2a-quinquies is
the **SGD** train step; the AdamW render has 210 params.)

**What (a) needs, measured — four ops, and two of them are free.** Every one of mnv2's 12 `*Sgd` ops
and all its forward/backward ops were enumerated against existing batched peers:

| missing | route | cost |
|---|---|---|
| **`BatchableOp.relu6`** (11 uses) | 4-site descriptor — ctor, `denOp`, `batchOpDescr`, `emitTok`. Pointwise and batch-INVARIANT so a descriptor is legal; template is `.relu` at `StableHLO.lean:136/964/2897/4549` | small |
| **`selectMidB`** (11 uses) | ⚠ **the original scope missed this.** relu6's backward mask carries the saved per-example `x`, so per §4 it *cannot* be a descriptor — own ctor riding `.batched "selectMidP"`, template `selectPosB` (`:415/1187/1521/2951`, emit `:4598`) | small |
| **`depthwiseBiasGradB`** | **zero new emitter code** — see below | proof-side only |
| **`depthwiseStridedBiasGradB`** | **zero new emitter code** | proof-side only |

**Why the two bias grads are free.** `convBiasGrad` (`:4674`), `bnBetaGrad` (`:4705`) and ConvNeXt's
`depthwiseBiasGrad` (`:5201`) emit **character-identical text** modulo SSA freshness —
`reshape (B, c*h*w) → (B,c,h,w)`, `constant dense<0.0>`, `reduce add across dimensions = [0, 2, 3]`.
`convStridedBiasGradB` already aliases `convBiasGradB`'s tag (`:3071`/`:3073`), and
`StableHLOParse.lean` handles `.batched` generically. So each is **ctor + `den` + `skel` +
`*SgdB_eq_grad` theorem + a `TestBatchedEmitTie` case** — no `Raw`/`Tok`/`toToks`/`emitTok`/
`parseStack`/`parse_toToks` work at all. That is §2f-bis's `depthwiseWeightGrad` aliasing route.

#### ✅ Step 1 DONE 2026-07-28 — all four ops built, gated, and verified to fail

`lake build Proofs Certs Codegen` green; `verified_mlir/` **unmodified** (no tracked artifact
changed, checked before and after); all four new theorems **3-axiom clean** (`propext`,
`Classical.choice`, `Quot.sound`). **UNCOMMITTED.**

| op | sites touched | theorem |
|---|---|---|
| `BatchableOp.relu6` | ctor, `denOp`, `batchOpDescr`, `emitTok` (4) | `den_batchOp_relu6_eq_relu6F` |
| `selectMidB` | ctor, `den`, `skel`, `emitTok` (4) + `den_selectMidB` simp | `selectMidB_faithful` (two-sided hyp) |
| `depthwiseBiasGradB` | ctor, `den`, `skel` (3) — **no `emitTok`** | `depthwiseBiasGradB_faithful` |
| `depthwiseStridedBiasGradB` | ctor, `den`, `skel` (3) — **no `emitTok`** | `depthwiseStridedBiasGradB_faithful` |

`tests/TestBatchedEmitTie.lean` is now **15 emit ties + 23 grad-prefix checks** (was 13 + 21).

**The aliasing claim is confirmed empirically, not just by reading the emitters.** All six bias
gradients — `convBiasGradB`, `convStridedBiasGradB`, `bnBetaGradB`, ConvNeXt's per-example
`depthwiseBiasGrad`, and both new ones — render to **exactly 280 of 444 bytes** of their fused peer.
Both new ones are checked against the *per-example* fused `depthwise{,Strided}BiasSgd`, since
`MobileNetV2RenderB` is AdamW-only and no batched fused peer exists.

**Both new guard groups were verified to go red**, per §4's "prove a green tie can go red":

| control | perturbation | result |
|---|---|---|
| A | `relu6` batched emit clamps to `5.0` instead of `6.0` | **`✖ relu6`** fires alone; all 14 other emit ties stay green |
| B | `depthwiseStridedBiasGradB` skels to an unmatched tag | **`✖ … is NOT a prefix`** fires — i.e. the prefix check *does* catch the silent `// MALFORMED` fallthrough §4 warns about, and `depthwiseBiasGradB` stayed green beside it |

#### ✅ Step 3 DONE 2026-07-28 — `MobileNetV2RenderB.lean` renders, and the interface is EXACT

`LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean` (added to the `Proofs` lib roots), AdamW-only at
`N := B`, 17-block paper spec. **UNCOMMITTED.** `lake build Proofs Certs Codegen` green.

**It renders to its own `verified_mlir/mobilenetv2_adam_train_step_b.mlir`**, deliberately — the
canonical path is still owned by the hand-written emitter in `tests/TestMobilenetV2TrainPC.lean`,
and a second writer would be the §2a last-writer-wins race. Same staging §2b-ter used for R34.

| gate | result |
|---|---|
| entry name | `@mobilenetv2_adam_train_step` — matches ✅ |
| arity | **739 in / 737 out**, matches the committed artifact ✅ |
| arg + return TYPES, positionally | **identical** ✅ — and here even the arg NAMES agree, unlike ViT/enet |
| `// MALFORMED` | **0** ✅ |
| gate 1 (weakened, §2f) | `mobilenetv2_train_step.mlir` md5 unchanged `a79d2e8b…` ✅ |
| `regen_verified_mlir.sh check` | 56 artifacts, **one writer each** ✅ |

**Structural agreement with the committed render, where it must hold:**

| op | committed | certified |
|---|---|---|
| convolution | 155 | **155** |
| dot_general | 3 | **3** |
| transpose / reverse / pad | 173 / 51 / 164 | **173 / 51 / 164** |
| minimum, maximum, select, and (the 35 relu6 sites) | 35 each | **35 each** |
| `rsqrt` | 52 | 156 = 52×3 |
| total `stablehlo` ops | 8,844 | 13,701 (1.55×) |

The `rsqrt` 3× and the 1.55× op ratio are the expected `pretty`-has-no-CSE story: `bnBatchF`,
`bnBatchBack` and `bnGammaGradB` each rebuild x̂. §2b-bis measured XLA collapsing exactly this on
R34 (108 → 36, 1.68× → 1.0004× after optimisation) at **no** run-time cost. Do not treat it as a
problem without measuring — and note mnv2's driver is IREE, so re-measure rather than assume.

**Structural choices worth knowing.** The two stride-1 block kinds (identity-skip and no-skip) are
ONE function `irBackStride1GradB` with a `skip : Bool`, because the only difference is the `addVB`
fan-in — duplicating the traversal would be the double-writer disease one level down, in code
(§2a-quater). BN stat slots are derived from the same forward record that computes them
(`f.ec`/`f.dc`/`f.pc`), never a parallel 52-entry table, per §2e — a misaligned slot is silent.

#### ✅ Step 4 DONE 2026-07-28 — the tie is BIT-EXACT, and all three controls fire

`tests/TestMobilenetV2AdamTie.lean` → `lake build mobilenetv2-adam-tie`, IREE-linked
(`mobilenetv2-verified-adam` is an `ireeLink` binary). One AdamW step, all **6,795,329** returned
floats. It deletes both `.vmfb` paths before every compile, and no run reported `(cached vmfb)`.

| check | result |
|---|---|
| interface vs the hand-written render | 739 in / 737 out, arg+return types positionally identical ✅ |
| A-vs-A determinism floor | **bit-exact 6795329/6795329** ✅ |
| **forward** (`bnstat` = batch μ/var of all 52 BN inputs) | **BIT-EXACT 34112/34112** ✅ |
| **`%loss`** | **BIT-EXACT 3/3** ✅ |
| **gradient (`m`)** | **BIT-EXACT 2253738/2253738** ✅ |
| θ, `v` | bit-exact ✅ |
| **spread** (§2f-bis) | **0/210 params** ✅ |

**Bit-exact, not 1e-6 — and the reason is the backend, so do not generalise it.** Same story as
EfficientNet (§2e): IREE's pipeline is deterministic, while R34's ≤2e-6 spread is XLA autotuning
picking different convolution algorithms across processes (§3). A bit-exact result here does not
predict one on XLA.

**Three negative controls, because a tie that reports bit-exact everywhere is indistinguishable
from a harness comparing a buffer with itself.** Each was the certified render with ONE thing
changed, compared against the certified render, and each fired **its own** gate while the others
stayed clean:

| control | perturbation | result |
|---|---|---|
| A | cotangent α 0.1 → 0.11 (1 line) | **gradient** gate fires at 1.17e-2, spread **111/210**; forward `bnstat` and `%loss` stay BIT-EXACT |
| B | BN ε 1e-5 → 1e-4, **re-rendered** (156 lines) | **forward** gate fires — `bnstat` bit-exact only **80/34112**; gradient 0.121, spread 131/210 |
| C | `%loss` constant 0.9 → 0.8 (1 line) | **loss** gate fires at 0.100; θ, `m`, `v` and `bnstat` ALL stay bit-exact, spread **0/210** |

Control C is the sharpest: it isolates `%loss` completely, which is the direct evidence that the
report-only scalar really is on no gradient path — and therefore that the loss gate is doing
independent work rather than shadowing the gradient gate. Control A is also a clean
re-demonstration of §3's rule on a fourth net: **θ moved only 1e-6 while `m` moved 1.17e-2**, so a
θ-based gate would have passed a real cotangent bug outright. And A-vs-B is exactly §2f-bis's
spread signature — a different function is GLOBAL (111 and 131 of 210) where the true tie disturbs 0.

(On control B, ε enters the *normalisation* `rsqrt(var+ε)`, not the statistics themselves, so the
stem BN's own μ/var — 64 of the 34112 slots — are structurally upstream of any ε effect. 80 stayed
exact, so 16 further slots coincided numerically; that residue was not investigated.)

#### ✅ Step 5 DONE 2026-07-28 — the swap. **The AdamW scorecard is 6 of 6.**

`verified_mlir/mobilenetv2_adam_train_step.mlir` is now written by
`Proofs/Codegen/MobileNetV2RenderB.lean`'s `#eval`, and that is its **only** writer. The driver
needed **no change at all** — it resolves the path from the net slug, so taking over the canonical
name *is* the swap. `…_b.mlir` is deleted; the bytes now at the canonical path are byte-identical
to the `_b.mlir` render that passed the tie (checked before deleting).

Gates run for the swap, in order:

| gate | result |
|---|---|
| regenerated canonical == the tied `_b.mlir` | **byte-identical** ✅ (md5 `9e8280d5…`, retired was `73e425ad…`) |
| `mobilenetv2-adam-tie` retired-render vs new canonical | forward **BIT-EXACT** 34112/34112, `%loss` bit-exact, gradient **bit-exact**, spread 0/210 ✅ |
| gate 1 (weakened) — SGD artifact | `mobilenetv2_train_step.mlir` untouched ✅ |
| elaborating the retired `tests/TestMobilenetV2TrainPC.lean` | rewrites **nothing**; canonical md5 unchanged ✅ |
| `regen_verified_mlir.sh check` | **55 artifacts, one writer each**, no MALFORMED ✅ |
| `iree-compile` smoke on the certified bytes | **OK** ✅ (both SGD and AdamW smokes green) |
| `lake build Proofs Certs Codegen` | clean ✅ |

**The retirement went further than repointing, following §2b-quater.** The hand-written AdamW
emitter in `tests/TestMobilenetV2TrainPC.lean` (`adamParams`, `adamConsts`, `adamCot`, `bnLayers`,
`trainStepAdamSched` — 110 lines) is **deleted**, not left dormant: a second emitter that can write
is one more thing to drift, and this net's own history is the argument. Recover it from
`git show 75a9f8e:tests/TestMobilenetV2TrainPC.lean` if ever needed. The SGD `trainStep` and the
shared `renderBody` are untouched, and the file keeps both `iree-compile` smokes — the AdamW one now
reads the *committed* bytes and **throws** if they are missing rather than recreating them.

**Not run: a smoke-train.** R34's swap (§2b-ter) included `LEAN_MLIR_G2_STEPS=40` showing the loss
descend on the swapped artifact. That was not done here. The evidence that stands in its place is
strictly graph-level — a bit-exact tie plus a successful `iree-compile` — so if you want end-to-end
confirmation that the swapped artifact *trains*, that run is still owed. It is cheap
(`mobilenetv2-verified-adam` with a small step budget) and worth doing before quoting new accuracy.

**Everything else already exists** — `bnBatchF`/`bnBatchBack`, `bnBatchMeanB`/`bnBatchVarB`, all four
`*BackBatched` convs, `gapBackBatched`, `addVB`/`subB`, the `scaleB → addVB → shiftB → divConstB`
smoothing chain, `depthwise{,Strided}WeightGradB`, and
`BatchableOp.{conv,convStrided,depthwise,depthwiseStrided,gap,softmaxRow,denseRowBack}`. mnv2's head
must move `expe`/`softmaxDiv`/`dotOut` → `softmaxRow`/`denseRowBack`, as enet's did.

**A `den`-vs-emit observation, flagged not fixed.** mnv2 and ConvNeXt both use *per-example-typed*
param-gradient ops (`convBiasGrad`, `depthwiseBiasGrad`, `bnBetaGrad`) whose emitters reduce over
`[0, 2, 3]` — contracting the runtime batch. That is the same shape §2b named as "the second kind".
It is benign for the emitted text (summing per-example gradients over a batch *is* the batch
gradient) and it is precisely why the depthwise bias grads come free here. But if §2b's honesty
argument is meant to cover the per-example renders too, `ConvNeXtRender.lean:138-164`
(`blockParamSgd`) is in the same position. A decision, not a defect.

**Traps specific to mnv2, all already paid for elsewhere:**

* **mnv2 fails LOUDLY where the others fail silently** (§2a-quinquies): a clobbered mnv2 artifact has
  the wrong *arity*, so the driver refuses it. Do not let that set your expectations for the rest.
* its `_train_step` docstring (`MainMobilenetV2Verified.lean:19`) says *"mean-loss SGD lr=0.3"* while
  the graph is **sum-loss at 0.3** — an effective 9.6. That is a **tuned** value, not a slip
  (`runs/mobilenetv2_verified_crop_gpu0.log` 32.9% → 86.89%, matching README's 87.09). Leave the
  number alone; it is only the docstring that misleads.
* the §2b `N := 1` hazard is the whole point of this task — re-read §2b's list of which ops are
  batch-*reducing* before assuming a re-instantiation is a one-liner. It was not, twice.

**Its tie will be enet-grade, which is the good case**: mnv2 has 52 BN layers, so the returned batch
statistics give a `bnstat` region that pins the forward **bit-exactly**. Derive those stat slots from
the traversal that computes them, never from a parallel table (§2e — a misaligned slot is silent).
And gate the **spread** as well as the magnitude (§2f-bis).

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
an independent optimisation** — this measurement is the evidence, and §2e-ter is a second,
independent one: EfficientNet reaches **1.75× on-GPU** and **1.67× end-to-end**, and the
13–16% per-step DP overhead it measures is dominated by the `[θ|m|v]` push to the second replica.

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
0a. ~~**MobileNetV2 at the batched index — §2f.**~~ ✅ **DONE 2026-07-28 — the scorecard is 6 of 6.**
   Four ops, a new `MobileNetV2RenderB.lean` (mnv2 is R34-shaped, not enet-shaped), a bit-exact tie
   with three firing controls, and the swap. The AdamW track is closed.
0b. ~~**The last four double-writers — §2a-quinquies.**~~ ✅ DONE Zero-behaviour-change cleanup: the committed
   bytes are *already* the certified render for all four, so this only removes clobber hazards.
   Cheapest remaining item by a wide margin, and it takes the audit to 0.
1. ~~**bs256 re-render + measure.**~~ ✅ **DONE 2026-07-28 — 1.78×, gated.** See §2d.1 below.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified end to end.
3. **Device-resident parameters.** Two rounds of transfer work are already done (batching:
   256→205 ms; killing the per-step host memcpys: 205→162 ms). What remains is smaller than it
   looks — see §3. **Two independent multi-GPU measurements now point here** (§2c 1.46× on R34,
   §2e-ter's measured 13–16% per-step DP overhead on EfficientNet, most of it the `[θ|m|v]` push to
   the second replica), so it is the highest-value structural item left. (On EfficientNet the data loader is
   NOT competitive with this: measured at only 6.3% of a 1-GPU epoch and 11.3% of a 2-GPU one,
   §2e-ter — an earlier claim here that it dominated was a measurement artefact.)
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
- **Gate the SPREAD, not just the magnitude.** §2f-bis: ConvNeXt's layer-scale γ gradient is a
  cancelling reduce that does not reproduce to 1e-4 against *any* reordering, so the gradient gate
  has to be relative to a reorder control. But a control-relative MAGNITUDE gate alone **passes a
  deliberately perturbed cotangent** — measured, α 0.1 → 0.11 lands under 4× the control. What
  separates them is how many parameters move: the perturbation disturbs 178/180 where the control
  disturbs 6 and the real tie disturbs 1. Floating-point conditioning is LOCAL to the
  ill-conditioned op; a different function is GLOBAL.
- **In a numeric tie, run A against itself first.** It establishes the determinism floor (XLA is
  bit-identical for a single step), which is what turns "the difference is 1e-6" from an assertion
  into a measurement. And report **per region**: in `resnet34-adam-tie`, `bnstat` depends only on
  the forward, so it separates a forward disagreement from a backward one in one run.
- **A tie harness must DELETE its `.vmfb` before compiling.** `mkSession` → `compileVmfb` reuses any
  existing output file **newer than the `.mlir`**: the cache key is the output path plus an mtime,
  never the source. So running the same tie binary twice with different candidates under the same
  tag silently reuses the FIRST candidate's binary and reports the second as a perfect match —
  observed while building §2e's negative controls, where it reproduced the previous control's
  numbers to the digit for a render that had never been compiled. `efficientnet-adam-tie` deletes
  both the bare and the `_$IREE_BACKEND`-scoped path first; **`resnet34-adam-tie`, `vit-adam-tie`,
  `sgd-render-tie` and `cifar8-adam-tie` still have the hazard.** It only bites on a re-run with
  different arguments — which is exactly what running a control looks like. Grep the output for
  `(cached vmfb)` before believing any re-run.
- **Prove a green tie can go red.** A tie that reports bit-exact everywhere is indistinguishable
  from a harness comparing a buffer with itself. §2e ran three controls off the same renderer with
  one thing perturbed each — cotangent, BN ε, `%loss` constant — and checked that the *matching*
  gate fired while the others stayed exact. That also localises what each gate is actually sensitive
  to: the ε control left exactly 76/42016 statistics bit-exact, and those 76 are the stem BN, the
  one layer upstream of the perturbation.

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

The three DP renders are **not** equally evidenced, and the difference is worth stating whenever any
of them is described: **EfficientNet** (§2e-bis) is gated by an exact known-answer check on the real
net, run on two GPUs; **ResNet-34** (§2b-quater) by that same check on a cifar8 proxy plus a 2-GPU
descent run; **ViT** by nothing numeric at all — its graph does not execute on this box. Only the
first two should be called working.

And on the renders: `pretty(provenGraph)` means the committed bytes are the certified render *of
the graph that was proven* — it does not mean the emitter is verified. The `Tok → StableHLO-text`
lexing stays audited-but-trusted, which is why every move in §2a and §2b is backed by a numeric tie
against what it replaced, not by the faithfulness theorem alone.

ConvNeXt additionally keeps two **weight-gradient** gaps that no other net has — the stem 4×4/s4
patchify and the even-kernel 2×2/s2 downsample, neither of which has a VJP-cert `SHlo` op — so its
render is `pretty(AST)` for 176 of 180 params and *hand-written gradient + certified update* for the
other four (§2f-bis). Say "ConvNeXt's AdamW render is certified except for two documented
weight-gradient gaps", not "ConvNeXt is certified".

Four places currently emit text that is **not** `pretty` of an AST node, and all say so in the
emitted output: `cifar8_adam_train_step`'s report-only scalar `%loss` and its `%bc` passthroughs,
and the `resnet34`/`vit`/`efficientnet` AdamW renders' `%loss`. **The R34 one shipped wrong** — plain CE instead of
the smoothed CE its own cotangent implies — and only the numeric tie found it, because nothing on a
gradient path touches it and no theorem covers it. Treat every such carve-out as unverified text
that needs its own numeric check, not as a harmless annotation.

A note on what the §2b tie does and does not establish. It says the batched render computes what the
hand-written render computes — forward to the bit, backward to norm-relative 1e-6. It does **not**
say either one is the mathematically intended net; that comes from the `den` side (the faithfulness
theorems, now honest at `N := B`) and from the layer-level VJP oracle in §3. The two halves are
independent, and both are needed: the artifact cannot witness a wrong `den` (the render is
value-independent), and the theorems cannot witness a wrong emitter.
