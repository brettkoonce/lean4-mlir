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
| `5920848` | the swap — the AdamW scorecard is 6 of 6 |
| `9bb00f5` | close ConvNeXt's two weight-gradient gaps — all 180 params `pretty(AST)` |
| `17413f0` | docs(handoff): §0b per-net leftovers |
| — | *↓ the forward-artifact thread (§2g), 2026-07-28 — the last five* |
| `019e09d` | **all five `_fwd` artifacts certified; `mobilenetv2_fwd` was the wrong BN world** |
| `4bbcb6e` | §2h: the `-xla` trainer plan, de-risked — mnv2/ConvNeXt backward graphs RUN on XLA, ViT's SGD one does not |
| — | *↓ the `-xla` trainer thread (§2h), 2026-07-29* |
| `d4da271` | **`mobilenetv2-` and `convnext-verified-adam-xla` built, gated and MEASURED**; ViT plumbed, blocker reproduced |
| `2a0aafd` | MobileNetV2 data-parallel — gated by the exact identity on the real net |
| `1a9a7a1` | **ConvNeXt data-parallel** — the last net with no DP path; 1.68×, and the first rank-0 collectives |
| `772815f` | eval batch decoupled from train batch (`evalBs`); R34 bs128×2 DP render |
| `068a494` | R34 bs64 + bs128 single-device renders; **the BN-split cost, measured** |
| `9f7304f` | §2d.2 — the sweep is monotone in step count |
| `9a78491` | **`convnext-shard-check`** — prove the replicas saw DIFFERENT data; §2i scoped |
| — | *↓ the §2i cifar8 port, 2026-07-29/30* |
| `769d0ab` | **plain-SGD + Nesterov optimizer ops** — `SgdMomentumStep.lean` + 3 `SHlo` ops, 10 sites each |
| `d5fbf32` | **cifar8 `sgd` + `mom` certified and swapped** — bit-exact tie, controls fire. 2 of 13 |
| `d0f87ac` | §2i — the three BN variants scoped by measurement |
| `f107c47` | **cifar8 BN `{adam,mom,sgd}` certified and swapped** — `opt` threaded, one `bnSig` source, 4 controls; the `.epoch`/`head` trap solved (§4). **5 of 13** |
| `eab3ecf` | **the 8 `cifar8w*` certified** — `cifar8` at `d1 := 512`, not a second net; a Chapter-5 table, not an ablation. **§2i CLOSED, 13 of 13** |
| `cf07259` | `lake run {mnist,cifar,imagenette}-xla` — the XLA demo groups, with a shim-build and plugin guard |
| `eb3632c` | the four missing cifar `-xla` peers — **`lake run cifar-xla` is all six**; six stale writer docstrings corrected |

---

## 0. ▶ START HERE — the AdamW scorecard is **6 of 6**. This thread is CLOSED.

Done 2026-07-28. **cifar8, resnet34, vit, efficientnet, convnext and mobilenetv2** all train on
`pretty(provenGraph)`, each swap licensed by a numeric tie that was verified to fail, and the writer
audit reports one writer per artifact. There is no "next AdamW render".

**What is left in this file is §2d's value-ordered list**, none of it on the AdamW track: rung 4
(the FPN detector), **device-resident parameters** (now **four** independent multi-GPU measurements
point at it — §2c's 1.46× on R34, §2e-ter's 13–16% per-step DP overhead on EfficientNet, and
mnv2's and ConvNeXt's end-to-end 1.67×/1.68×), and the executable cache. Read §2d before picking one.

**The DATA-PARALLEL scorecard is 4 of 5 as of 2026-07-29 — §2h-quater.** ConvNeXt was the last large
net with no DP path at all (its renderer took no `replicas` and emitted no collective); it now has
one, gated by the exact duplicated-batch identity on the real net and measured at **1.68×** on two
GPUs. **ViT is the only net without a working DP path**, and for a reason nothing in this thread can
fix from here — its graph does not execute on this box at all.

**▶ THE ACTIVE THREAD IS §2i — the last 13 `tests/`-rendered artifacts, 5 of 13 DONE, and every
LIVE TRAINER in the set is now certified.** This is the §2a provenance axis being finished. State as
of 2026-07-30:

* ✅ **`cifar8_sgd` + `cifar8_mom`** are `pretty(provenGraph)`, swapped, sole-writer. Three new
  proven ops (`sgdParamF`, `momVNextF`, `momParamF`) + `SgdMomentumStep.lean`, all 3-axiom clean;
  `CifarOpt` threaded through ONE renderer so all three variants share one forward/backward and only
  the tail differs. Tie **bit-exact 52858/52858**, both controls fire, gate 1 held.
* ✅ **`cifar8_bn_{adam,mom,sgd}`** — done 2026-07-30, no new ops, `opt : Option CifarOpt` threaded
  through the existing renderer as scoped. Gate 1 held byte-identically on all four incumbents; ties
  ≤3.8e-5 against a reorder control they match or beat; four controls fire; **it descends on the
  swapped bytes** (test 13.2 → 30.1% over 4 epochs). **`cifar8_bn_adam` was the only one of the 13
  backing a real trainer**, so what is left is ablation-only.
* ✅ **the eight `cifar8w*`** — done 2026-07-30. Both assumptions about them were wrong: `cifar8w` is
  `cifar8` at `d1 := 512` (no second net), and they are not ablation-only, they are the **Chapter-5
  "bridge" table**. Three ties BIT-EXACT, four controls fire, it descends.

**▶ §2i IS CLOSED — 13 of 13. `regen_verified_mlir.sh check`: 61 artifacts, one writer each, and
"every artifact is `Proofs/`-rendered" is now true with NO carve-out.** The §2a provenance axis,
open since 2026-07-27, is finished.

**Also landed 2026-07-30: `lake run {mnist,cifar,imagenette}-xla`** (`cf07259`, `eb3632c`). mnist and
cifar are EXACT mirrors of their IREE groups — the four missing cifar peers were built with the §2h
recipe, so `lake run cifar-xla` now runs the whole six-way Chapter-5 optimizer ablation on the
second lowerer. imagenette-xla is 4 of 5; ViT is excluded by measurement, not omission.

**▶ `lake run benchmark-xla` IS DONE — 2026-07-30, §2j below. UNCOMMITTED.** Both commands now
label their lowerer, each scales from its own measured reference column, and the mismatched-baseline
trap is structural rather than documented: `BenchRef` bundles a lowerer's probe binaries with its
anchors, so a probe cannot be divided by the other path's reference. A `cifar8-bn-verified-xla`
peer was added for the conv anchor (`.train` **does** drive the PJRT shim — that was §2j's open
question). On-reference factors read 0.99-1.02× on both paths.

**▶ AND VIT RUNS ON XLA — the last named hard gap in this whole thread is GONE, but NOT for the
reason first recorded.** `vit-verified-adam-xla` executes on this box, **with no workaround**, and
is numerically gated: first three step losses agree with the IREE peer to **3e-6** from identical
fresh init (3.015268/3.005852/2.987083 vs 3.015261/3.005857/2.987081); it descends
(39.7 → 46.7 → **49.6%** over 3 epochs); **`vit-dp-check` passes BIT-EXACT on all 16,579,041
floats** against a sum-not-mean control that fires at **0.996** — so **DP is 5 of 5**. Speed:
**128 ms/step** median vs IREE's 1188 (**9.2×**), marginal epoch **43.5 s** ⇒ 80 ep ≈ 0.97 h vs
IREE's 7.8 h.

⚠ **A correction to my own first finding here, recorded because it nearly shipped.** I first hit the
documented im2col failure, found `MIOPEN_DEBUG_CONV_GEMM=0` made it run, and reported that as the
fix — it is **not one**. Measured properly afterwards: the graph runs *without* the variable (11
consecutive runs, including the byte-identical invocation that had just failed), and setting it
**costs ~7%** (attn probe 136 vs 128 ms/step; marginal epoch 46.5 s vs 43.5 s). It had already been
wired into `benchmark-xla` and `imagenette-xla` and was **backed out**. **The real finding is that
the failure is NON-DETERMINISTIC**: it fired once, on the session's first ViT/XLA execution, and
never again; the MIOpen on-disk cache shows no writes in that window, so cache population does not
explain it, and the mechanism is unidentified. Keep the variable as a documented escape hatch only.
The lesson is the thread's own recurring one, applied to a *negative* result this time: **one sample
is not a measurement — I inferred causation from a single before/after pair.** The same mistake, in
the same session, produced the retracted conv-probe "thermal" story (§2j).

`upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/README.md`'s **diagnosis was already right
and already confirmed** on 2026-07-28 (the fused interior-dilated pad+conv selects the broken
`GemmFwdRest`/im2col path), with a 20-line JAX reproducer — so this handoff's own description of that
as a *"leading unconfirmed hypothesis"* was stale and is retired. Its line 312 ❌ for
`MIOPEN_DEBUG_CONV_GEMM=0` (*"MIOpen hung … Killed it"*) is also stale — it completes in 39 s here —
but since the variable is not needed, that is a footnote, not a fix.

**NOT established on ViT:** no 80-epoch run and no accuracy number (the other four Imagenette nets
have both). See §2j's tail for the exact ledger.

After these, §2d's value-ordered list: rung 4 (the FPN detector) or device-resident parameters
(four measurements behind it; a calibrated model says the case roughly quadruples at 4 GPUs —
§2d.3) — plus the ViT follow-ups the line above unlocks.

Read §2i before starting — it has the measurements, not guesses.

Also queued, both small: port `convnext-shard-check` to the enet/mnv2 DP gates (§2h-quater, ~20
lines each), and try relinking `resnet34-adam-tie`/`cifar8-adam-tie` to `ireeLink` — the four
IREE-linked ties all report **bit-exact** while the XLA-linked ones report ≤2e-6 with run-to-run
spread, which is the backend, not the net. **Rule worth adopting: ties link IREE (deterministic),
DP checks link XLA (collectives only exist there), training runs XLA.**

**The `-xla` trainers are DONE for mnv2 and ConvNeXt as of 2026-07-29 — §2h.** Both now have an
XLA/PJRT peer sharing one body with the IREE binary, both agree with IREE to fp noise, and both
**descend on their current certified bytes** — which closes §0b's first open item for the two nets
that had no training run at all. ViT is plumbed and its blocker was reproduced on the new binary;
it still does not run here.

**The five forward artifacts are certified too, as of 2026-07-28 — §2g.** `convnext_fwd`,
`efficientnet_fwd{,_eval}` and `mobilenetv2_fwd{,_eval}` now render `pretty(provenGraph)` off the
same forward chain their train step differentiates. **That found a live §2a skew**: `mobilenetv2_fwd`
was batch-BN against a per-example-BN train step, so `mobilenetv2-verified` scored a different net
than it trained (measured: logits rel **1.86**). Every `verified_mlir/` artifact is now
`Proofs/`-rendered.

~~**One named gap stays open**: **ViT's DP render does not execute on this box**~~ ✅ **CLOSED
2026-07-30 (§2j's tail).** It executes, with no workaround; `vit-dp-check` passes **bit-exact on all
16,579,041 floats** against a control that fires at 0.996, so **the DP scorecard is 5 of 5**. ⚠ The
underlying MIOpen fault is **non-deterministic, not fixed** — it fired once and not again in 11 runs
— so a recurrence is possible; and ViT still has no 80-epoch run. *ConvNeXt's two
weight-gradient carve-outs were CLOSED on 2026-07-28 (`9bb00f5`, §2f-bis, §5) — an earlier version
of this line still listed them as open.*

*The MobileNetV2 write-up below (§2f) is kept because its scoping correction — that mnv2 was
R34-shaped, not EfficientNet-shaped — is the reusable lesson, not because anything is owed.*

---

## 0b. ▶ Per-net leftovers — read this before a training run

Written 2026-07-28 at the end of the AdamW thread. **Nothing here blocks the scorecard**; it is what
is *not* done on each of the four large nets, ordered by what would bite first.

### ⚠ The three things to do before trusting a long run

1. ~~**MobileNetV2 and ConvNeXt have NO training run on their current artifacts.**~~ ✅ **CLOSED
   2026-07-29 — §2h.** Both now descend on their current certified bytes, on both lowerers:
   **mnv2** reaches val **59.9%** after 4 full epochs on XLA (26.7 → 38.0 → 52.3 → 59.9) and
   **68.2%** over an 80-epoch capped-step run; **ConvNeXt** descends over 17 epochs on XLA and 1 on
   IREE, epoch-1 loss **identical on both backends**. The gap between "computes the same function"
   and "trains" is closed for both.
2. ~~**Only EfficientNet is validated end to end.**~~ ✅ **CLOSED for three nets 2026-07-29 —
   and all four now have an 80-EPOCH run on their certified bytes:**

   | net | final (ep 80) | best | wall (XLA, 1 GPU) |
   |---|---|---|---|
   | **ResNet-34** | **90.39%** | 90.39% | 1h11m |
   | **EfficientNet-B0** | **88.20%** | 88.46% | 1h34m |
   | **MobileNetV2** | **86.73%** | 86.96% | 1h25m |
   | **ConvNeXt-T** | **82.75%** | 82.98% | 1h56m |

   All `rc=0` with the epoch marker on 80 (so genuinely to the end, not a resumed no-op), scored
   through their `Proofs/`-rendered eval forwards. Logs: `runs/<net>_xla_80ep_jul29.log` (untracked).
   **ConvNeXt is the outlier and it is NOT a plumbing problem** — it reaches the LOWEST val accuracy
   while reaching the LOWEST train loss (0.500), which is overfitting: ~28M params against 9,469
   images, curve flat from epoch 40, peak at 68. Read the ordering as a dataset-size story, not a
   ranking of the renders.

   ⚠ **The §2g wrong-forward number, 86.89%, is STILL owed** — it belongs to `mobilenetv2-verified`,
   the **SGD** binary, and these were the AdamW trainers. Cheap now that the `-xla` path exists.
   EfficientNet, MobileNetV2 (§2h-bis/ter) and ConvNeXt (§2h-quater) each now have single-GPU
   XLA-vs-IREE agreement, a descent run, measured wall clock, a DP artifact gated by the exact
   identity **on the real net**, and a 2-GPU descent run with a measured scaling ratio. **ViT is the
   remaining answer to "not validated"**, and its blocker is execution, not evidence.
3. **`mobilenetv2-verified`'s published 86.89% is measured through the WRONG forward** and needs
   re-running (§2g). Its eval artifact was batch-BN while it trained per-example BN — the §2a defect,
   fixed 2026-07-28. Only the *scoring* was wrong, so the training log itself stands; the accuracy
   number does not. This is the same footnote §3 already carries for `resnet34-verified`, and it now
   applies to two nets.

### Per net

| | AdamW certified | data-parallel | `-xla` trainer | forward artifact | run on current bytes |
|---|---|---|---|---|---|
| **EfficientNet** | ✅ | ✅ **best-gated** (exact identity on the real net, 2 GPUs) | ✅ | ✅ `Proofs/` (§2g) | ✅ 3 epochs + 2-GPU |
| **MobileNetV2** | ✅ | ✅ **§2h-bis** — exact identity on the real net, 2 GPUs, **1.67×** | ✅ **§2h** — 58.0 s/epoch | ✅ `Proofs/` (§2g — **BN skew fixed**) | ✅ **4 full epochs → val 59.9%**, + 2-GPU descent |
| **ConvNeXt** | ✅ (all 180) | ✅ **§2h-quater** — exact identity on the real net, 2 GPUs, **1.68×** | ✅ **§2h** — 84.5 s/epoch | ✅ `Proofs/` (§2g) | ✅ **4 full epochs → val 60.6%**, + 12-epoch 2-GPU descent |
| **ViT** | ✅ | ✅ **§2j tail** — `vit-dp-check` BIT-EXACT 16,579,041/16,579,041, control 0.996 | ✅ **RUNS 2026-07-30**, no workaround — 128 ms/step, 43.5 s/epoch | ✅ `Proofs/` | ✅ 3 epochs → val **49.6%**; ⛔ still no 80-epoch run |

**MobileNetV2** — ~~no smoke-train~~ ✅ §2h; **no `mobilenetv2_adamdp_train_step.mlir`** (the
renderer takes `replicas` and `mnv2AdamVariant` returns `adamdp` — both re-verified 2026-07-29 — but
no `#eval` writes it, and there is no `mobilenetv2-dp-check`); ~~no `-xla` target~~ ✅ §2h;
(`mobilenetv2_fwd{,_eval}` were `tests/`-rendered — ✅ fixed, §2g, and the fix was a real bug); and
`MainMobilenetV2Verified.lean:19` still documents itself as *"mean-loss SGD lr=0.3"* against a
sum-loss graph (effective 9.6) — a docstring fix, the number is tuned and trains. **The one
remaining gap is the DP artifact**, and it is now the only thing between mnv2 and multi-GPU.

**ConvNeXt** — ~~no smoke-train on either changed artifact~~ ✅ §2h; ~~no DP path whatsoever~~
✅ **§2h-quater**; ~~no `-xla` target~~ ✅ §2h; (`convnext_fwd` was `tests/`-rendered — ✅ fixed, §2g;
there is still no `_fwd_eval` and there should not be, LayerNorm means train == eval, and
`fwd-tie convnext --eval` now refuses outright rather than looking for a missing file).
**Nothing is owed on ConvNeXt.**

> ⚠ **A claim this file carried was FALSE and is retired** (2026-07-29): *"the scalar-LN γ/β render
> as `tensor<1xf32>` where the committed signature says `tensor<f32>`"*. Measured — `ty [] =
> "tensor<f32>"`, and `grep -c 'tensor<1xf32>'` is **0** in both `convnext_train_step.mlir` and
> `convnext_adam_train_step.mlir`. The scalar params are `tensor<f32>` on both sides and always
> were; there is no signature mismatch and nothing to fix. The same stale note in
> `ConvNeXtRender.lean`'s module docstring is retired too. It cost time in this thread because it
> reads like a live interface hazard for anyone touching the DP render.

**ViT** — the one with a *hard* blocker. Its DP render is numerically ungated because the graph will
not execute on this box: `miopenStatusUnknownError` in the patch-embed weight-grad convolution. Both
obvious hypotheses are **refuted** (the 209×209 shape runs fine standalone in JAX; memory is not it
— a genuine OOM gives a clean `RESOURCE_EXHAUSTED`). **Measured 2026-07-28: the SGD train step fails
identically**, so this is not an AdamW-render problem — it is in every ViT graph with a backward,
and a self-contained repro can start from the much smaller `vit_train_step.mlir` (§2h). The leading unconfirmed hypothesis is that XLA
fuses the `stablehlo.pad` into the conv as `rhs_dilation = 16`, a different MIOpen call than either
probe makes. To settle it: dump post-optimisation HLO via the §4 throwaway-shim recipe (`XLA_FLAGS`
is inert here) and read the `convolution` descriptor. **No bug filed and none should be** until
there is a self-contained repro. `vit-dp-check` is written and will pass the moment the graph runs —
run it on ares. Everything else on ViT is the cleanest of the four: it is the **only** net whose
forward artifacts are all `Proofs/`-rendered.

### Cross-cutting

* **The `.vmfb` false-PASS hazard is still live in four harnesses** — `resnet34-adam-tie`,
  `vit-adam-tie`, `cifar8-adam-tie`, `sgd-render-tie` (§4). `efficientnet`/`convnext`/`mobilenetv2`
  delete first. It only bites on a re-run with different arguments, which is exactly what running a
  control looks like. Grep for `(cached vmfb)`.
* ~~**Five forward artifacts are still hand-written**~~ ✅ **DONE 2026-07-28 — §2g.** All five now
  render `pretty(provenGraph)`, each off the same forward chain its train step differentiates, each
  swap licensed by a numeric tie that was verified to fail. The §2a *provenance* axis is closed:
  **every** `verified_mlir/` artifact is `Proofs/`-rendered. It turned up one real bug —
  `mobilenetv2_fwd` was the wrong BN world.

---

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

**Every artifact is `Proofs/`-rendered — TRUE with no carve-out as of 2026-07-30 (§2i).** This line
was flatly FALSE from 2026-07-27 to 2026-07-29 (13 hand-written artifacts with live drivers), and it
is worth knowing why it read as true anyway: the writer audit counts *duplicate* writers, not
provenance, so 13 hand-written artifacts with one writer each reported green. Four moved out of
`tests/` in §2a (`resnet34_fwd`,
`resnet34_fwd_eval`, `cifar8_adam_train_step`, and — already there — `resnet34_train_step`); the
last five followed on 2026-07-28 (§2g: `convnext_fwd`, `efficientnet_fwd{,_eval}`,
`mobilenetv2_fwd{,_eval}`). The optimizer itself is a proven op family rather than a hand-written
string emitter. `scripts/regen_verified_mlir.sh check` reports **55 artifacts, one writer each**,
with **four** forward ⊂ train-step prefix pairs green — the check that caught ResNet-34 scoring a
net it had not trained, and, in §2g, MobileNetV2 doing the same thing.

**The batched renderers are honest** (§2b). EfficientNet and the new batch-BN ResNet-34 AdamW train
step both sit at the batched index `N := B`, so the ten batch-*reducing* `den`s describe what the
emitted text actually computes. `ResNet34RenderB.lean` → `resnet34_adam_train_step_b.mlir` **ties
the hand-written render numerically**: forward bit-exact, backward norm-relative 1e-6. It is not yet
what the trainer runs — see §2b.

### Running it

```bash
# ── the one-command entry points (added 2026-07-30): the XLA peers of the three demo groups.
# mnist-xla and cifar-xla are EXACT mirrors of their IREE groups; imagenette-xla is 4 of 5.
# Each builds `ffi/libpjrt_ffi.so` if it is missing or older than `ffi/pjrt_ffi.c` (it is NOT a
# lake target, just the gcc line below), reports whether $PJRT_PLUGIN resolves, then builds and
# runs each trainer in sequence via run.sh. IREE_BACKEND is irrelevant on these — the backend is
# whichever .so the target linked.
lake run mnist-xla        # linear / MLP / CNN — an EXACT mirror of `lake run mnist`
lake run cifar-xla        # all 6 of `lake run cifar` (SGD/momentum/AdamW × BN/no-BN)
lake run imagenette-xla   # ⚠ 4 of 5: ViT is excluded BY MEASUREMENT (MIOpen, §0b/§2h)

gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-verified-adam-xla data

# the other -xla trainers. r34 / efficientnet / mnv2 / convnext all work; vit BUILDS but does NOT
# RUN on this box (§2h — miopenStatusUnknownError at execution, not at compile).
lake build mobilenetv2-verified-adam-xla convnext-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mobilenetv2-verified-adam-xla data
HIP_VISIBLE_DEVICES=0 .lake/build/bin/convnext-verified-adam-xla data
#   ^ each shares ONE body with its IREE peer (apps/imagenette/<Net>AdamCommon.lean), so the
#     schedule and seed cannot drift. The IREE peers need the venv on PATH for `iree-compile`:
#     PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified-adam data
#   ⚠ CHECK FOR A STALE CHECKPOINT FIRST (§4). `.lake/build/<slug>_adam_ckpt{,_xla}.bin.epoch`
#     survives across artifact swaps, so a June checkpoint from a RETIRED render will silently be
#     resumed — and one at `epoch=80` makes the run a silent no-op. Hit on 2026-07-29 on all three.

# the honest wall-clock measurement — marginal epoch (T3-T1)/2, never wall-clock-minus-compile
scripts/marginal_epoch.sh runs/mnv2_xla.log -- .lake/build/bin/mobilenetv2-verified-adam-xla data

# multi-GPU (§2b-quater): the DP render is certified-graph + trusted collective, written by the
# same `lake build LeanMlir.Proofs.Codegen.ResNet34RenderB` as the single-device one. To change the
# replica count, edit the `(replicas := 2)` #eval there and rebuild — `replica_groups` is baked, so
# it must match PJRT_REPLICAS.
unset HIP_VISIBLE_DEVICES
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/resnet34-verified-adam-xla data

# regenerate + audit verified_mlir/  (the canonical entry point; did not exist before §2a)
scripts/regen_verified_mlir.sh          # or `check` to audit without writing

# the FORWARD ties (§2g) — one harness, every net, XLA-linked so it compiles in seconds.
#   fwd-tie <slug> [--eval] [<pathA> [<pathB>]]; defaults to the committed artifact vs itself.
lake build fwd-tie
git show 17413f0:verified_mlir/convnext_fwd.mlir > /tmp/retired.mlir
HIP_VISIBLE_DEVICES=0 .lake/build/bin/fwd-tie convnext /tmp/retired.mlir verified_mlir/convnext_fwd.mlir
HIP_VISIBLE_DEVICES=0 .lake/build/bin/fwd-tie efficientnet --eval   # self-tie smoke
#   ^ `--eval` ties @<slug>_fwd_eval (params + 2 running-stat inputs per BN layer). ConvNeXt has no
#     _fwd_eval and the harness refuses --eval for it: LayerNorm ⇒ train == eval.

# the render ties (§2a, §2b, §2e). The AdamW ones need a GPU; TestBatchedEmitTie is CPU-only.
lake env lean tests/TestBatchedEmitTie.lean            # 16 emit ties + 23 grad-prefix checks
lake build resnet34-adam-tie && .lake/build/bin/resnet34-adam-tie
lake build cifar8-adam-tie   && .lake/build/bin/cifar8-adam-tie
# §2i — SIX cifar8 optimizer variants through one harness. Gates the RECOVERED GRADIENT (never θ'),
# the m/v passthrough slots, %loss, and the per-param SPREAD against a reorder control it runs
# itself (A vs A on the reversed batch). Deletes its .vmfb. TIE_SKIP_REORDER=1 skips the control run.
lake build cifar8-opt-tie
for v in adam sgd mom bn_adam bn_sgd bn_mom; do HIP_VISIBLE_DEVICES=0 .lake/build/bin/cifar8-opt-tie $v; done
#   ^ no args = self-tie smoke. To tie a candidate against the committed bytes:
#     .lake/build/bin/cifar8-opt-tie bn_adam /tmp/candidate.mlir
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
lake build mobilenetv2-dp-check  && PJRT_REPLICAS=2 .lake/build/bin/mobilenetv2-dp-check   # §2h-bis
lake build convnext-dp-check     && PJRT_REPLICAS=2 .lake/build/bin/convnext-dp-check      # §2h-quater
#   ^ all three are the EXACT identity on the REAL net. Pass a broken render as argv[1] for the
#     sum-not-mean control: mnv2 goes 8.7e-8 → 1.037 on the gradient while θ moves only 8.4e-5;
#     ConvNeXt goes ≤1.1e-8 → 1.114 while θ moves 9.7e-5, i.e. UNDER a 1e-4 θ gate. Build it with:
#       sed -E 's/^(    %arn[A-Za-z0-9_]+ = stablehlo\.constant dense<)2\.0(>)/\11.0\2/' \
#         verified_mlir/convnext_adamdp_train_step.mlir > /tmp/cnx_dp_sum.mlir
#   ConvNeXt gates %loss as well as the gradient: LayerNorm ⇒ no bnstat region, so %loss is the
#   ONLY forward-only output. It is also the only DP render with RANK-0 collectives (44 scalar LN).
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/mobilenetv2-verified-adam-xla data
#   ^ the exact identity ON THE REAL NET (duplicated batch ⇒ all_reduce/2 is the identity, and BN
#     does not spoil it because both replicas' groups are the same 32 examples). Pass a broken
#     render as argv[1] to run the sum-not-mean control.
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/efficientnet-verified-adam-xla data
LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 \
  .lake/build/bin/convnext-verified-adam-xla data
#   ^ §2h-quater: 1.68× (marginal epoch, train-only, 77.5 s → 46.0 s). Measure with
#     LEAN_MLIR_SKIP_EVAL=1 on BOTH sides — eval runs single-replica and is not part of the ratio.

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

**Still `tests/`-rendered** *(as of when §2a closed — all five were moved on 2026-07-28, §2g)*:
`mobilenetv2_fwd{,_eval}`, `efficientnet_fwd{,_eval}`, `convnext_fwd`,
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

**Claim ceiling.** ConvNeXt keeps two documented weight-grad gaps — the stem 4×4/s4 patchify
(`psW`) and the even-kernel 2×2/s2 downsample (`d1W`/`d2W`/`d3W`) — which stay hand-written. But the
SGD render also hand-wrote the *update* for exactly those params (the `sgd` helper); this render
replaces it with the proven AdamW triple. So for those four the tier moves from *hand-written
gradient + hand-written update* to *hand-written gradient + certified update*, and the other 176
params are `pretty(AST)` end to end. A strict improvement.

> ✅ **BOTH GAPS CLOSED 2026-07-28 — ConvNeXt is now `pretty(AST)` for all 180 params.** The
> paragraph above is the pre-closure state, kept because the *diagnosis* is the reusable part. What
> was done, and the gate, is at the end of this note.
>
> ⚠ **These two gaps were NOT the same kind of thing, and this section used to say they were**
> ("neither of which has a VJP-cert `SHlo` op"). Measured:
>
> * **`psW` (4×4/s4) is a real CERT gap.** `flatConvStride4` (forward, `StridedConv.lean:243`) and
>   `flatConvStride4_has_vjp` (the INPUT grad, `:262`) both exist; **`flatConvStride4_weight_grad_has_vjp`
>   does not.** Its construction is already spelled out by the stride-2 sibling at `:141` —
>   `vjp_comp` of `conv2d_weight_grad_has_vjp` with `decimateFlat` — and the extra piece stride-4
>   needs, `decimateOddFlat_has_vjp`, exists at `:226`.
> * **The downsample is NOT a cert gap.** `flatConvStride2_weight_grad_has_vjp {ic oc h w kH kW}` is
>   **kernel-generic** (no parity assumption — it is `vjp_comp (conv2d_weight_grad_has_vjp)
>   decimateFlat`), and so is the `SHlo` op `convStridedWeightGrad`. The downsample's forward AND
>   input-grad already use certified ops at 2×2 (`ConvNeXtRender.lean:111`/`:116`), so
>   `flatConvStride2` at an even kernel is already trusted in this very render. **The blocker is the
>   EMITTER**, which hardcodes symmetric SAME padding `pad = [[pH,pH],[pW,pW]]`, `pH = (kH-1)/2`.
>   Rendered both cases at input 8×8 → output 4×4:
>
>   | kernel | emitted pad | conv result | declared type | |
>   |---|---|---|---|---|
>   | 3×3 | `[[1,1],[1,1]]` | 3×3 | `2x3x3x3` | ✅ |
>   | **2×2** | `[[0,0],[0,0]]` | **1×1** | **`2x3x2x2`** | ❌ **type-invalid MLIR** |
>
>   At `kH = 2`, `(2−1)/2 = 0` in Nat division. Even kernels need ASYMMETRIC padding, which that form
>   cannot express; the hand-written `downWGrad` sidesteps it by omitting the trailing
>   `high = [0,0,1,1]` pad (cotangent to `2h−1`, not `2h`), after which a VALID conv yields 2×2.
>   So closing it is EMIT-side only — the `den` side is already covered.
>
> **The latent trap this closed:** `convStridedWeightGrad` at an even kernel typechecked in Lean and
> denoted correctly, so nothing in the proof layer stopped someone using it at `kH = 2`; it failed
> only at `iree-compile`. Loud rather than silent, but the "emitted is not verified" pattern again.
>
> **▶ What was built.**
> * **`StableHLO.sWGradGeom (k s)`** — the odd/even window geometry, shared by all FOUR strided
>   weight-grad emitters (`convStridedWeight{Grad,Sgd}`, per-example and batched) so they cannot
>   drift. Odd is *provably* the old inline formula, and that is measured: **every existing artifact
>   came back byte-identical** (R34, EfficientNet and mnv2 are all odd-kernel — 1, 3, 7).
>   ConvNeXt's `d{i}W` then moved onto the existing `.convStridedWeightGrad`. **No new cert.**
> * **`flatConvStride4_weight_grad_has_vjp`** (`StridedConv.lean`) — the genuinely missing one, plus
>   a `_correct` peer. Two `vjp_comp` steps over `conv2d_weight_grad_has_vjp` and the two
>   decimations, mirroring the stride-2 sibling; compiled first try, **3-axiom clean**. It backs the
>   new **`.convStride4WeightGrad`** `SHlo` op (ctor + `den` + `skel` on the generic `.batched` tag +
>   `emitTok` + faithfulness theorem — the four-site route), and `psW` is wired to it.
> * `patchWGrad`, `downWGrad`, `rs4` and the old `sgd` helper are **deleted**, not left dormant
>   (§2b-quater). Verified byte-inert: the artifact md5 is unchanged across the deletion.
>
> **▶ The gate — `convnext-adam-tie`, previously committed hand-written render vs the fully
> certified one.** BIT-EXACT on all **83,434,629** returned floats: θ, `m`, `v` each
> 27,811,542/27,811,542, `%loss` 3/3, **spread 0/180** against a reorder control that disturbs 5,
> and a bit-exact A-vs-A floor. Bit-exact means the new emitters compute *literally the same
> convolutions* as the hand-written ones. (Run twice — the artifact was rewritten mid-run the first
> time by an inert dead-code deletion, so it was re-run against the settled bytes rather than
> reasoned about.)
>
> **Claim now:** "ConvNeXt's AdamW render is certified" — the two-weight-gradient-gaps caveat is
> retired. `%loss` remains report-only and outside the AST, as on every net.

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

### 2g. The five forward artifacts ✅ DONE 2026-07-28 — and it found a live §2a skew

`convnext_fwd`, `efficientnet_fwd{,_eval}` and `mobilenetv2_fwd{,_eval}` were the last
`tests/`-rendered artifacts. All five now render `pretty(provenGraph)` off the **same forward chain
their train step differentiates**, and `scripts/regen_verified_mlir.sh check` reports **55 artifacts,
one writer each** with the prefix audit green on four nets. The §2a provenance axis is closed.

**The point is not provenance for its own sake.** A separately-written forward can silently be a
different function from the one the trainer optimises — and one of these was:

#### ▶ `mobilenetv2_fwd` was the WRONG BN WORLD — the §2a defect, still live

| | BN reduce | n | |
|---|---|---|---|
| retired `mobilenetv2_fwd` (hand-written) | `[0,2,3]`, 104 sites | B·H·W | **batch** BN |
| `mobilenetv2_train_step` (certified, `Proofs/`) | `[2,3]` | H·W | **per-example** BN |

So `mobilenetv2-verified` *trained* a per-example-BN net and *scored* it with batch statistics —
byte-for-byte the ResNet-34 bug §2a found on 2026-07-27, on a second net, uncaught for as long.
The old emitter's own docstring said "matches the reference's batch-norm", which is what made it
look deliberate.

Measured on one shared (θ, x) with the driver's real He init, `lake build fwd-tie`:

| | max rel | bit-exact logits |
|---|---|---|
| retired (batch BN) vs certified (per-example BN) | **1.857** | **0/320** |
| certified vs ITSELF — the determinism floor | 0 | **320/320** |

A relative difference above 1 means the logits disagree in *sign*. The bit-exact floor is what makes
that graph-attributable rather than backend noise. **`runs/mobilenetv2_verified_crop_gpu0.log`'s
86.89% therefore went through the wrong forward** and needs re-running (§0b).

`mobilenetv2_fwd_eval` was NOT affected and ties **bit-exact**: it is frozen-stat affine BN, which
performs no reduction, so it is the same graph in either BN world. EfficientNet was not skewed
either — both its sides were already batch-BN — and ConvNeXt cannot be, being LayerNorm.

#### The results

| artifact | tie vs the retired render | prefix of its train step |
|---|---|---|
| `convnext_fwd` | **BIT-EXACT 320/320** | ✅ 1192 lines |
| `efficientnet_fwd` | **BIT-EXACT 320/320** | ✅ 1935 lines |
| `efficientnet_fwd_eval` | **BIT-EXACT 320/320** | n/a (frozen stats) |
| `mobilenetv2_fwd` | ❌ **rel 1.857 — the bug above** | ✅ 1614 lines |
| `mobilenetv2_fwd_eval` | **BIT-EXACT 320/320** | n/a (frozen stats) |

Interfaces are positionally identical (arity, arg + return types) on all five; ConvNeXt and
EfficientNet match down to the argument NAMES, mnv2's 210 params do not (`%Ws` vs `%sW`) — the
104 stat names do.

**Gate 1 held everywhere**: every train-step artifact re-rendered **byte-identical** through the
refactor — ConvNeXt ×2, EfficientNet ×5, MobileNetV2 ×3 — which is what proves the chain extraction
and the mode threading inert. The only bytes that moved in `verified_mlir/` are the five forwards.

#### What got built

- **`BnMode`** (`StableHLO.lean`) — `.train | .eval`, the batched-index peer of `R34Bn`, shared by
  the EfficientNet and MobileNetV2 chains. One BN-site helper per renderer (`bnSiteB`, `bnSiteP`) is
  the *only* place the two worlds are chosen between, so the two artifacts cannot drift apart.
- **`BatchableOp.bnEval`** — inference BN at the batched index, the frozen-stats peer of the
  own-ctor `bnBatchF`. **A descriptor is legal here and that is the content**: γ/β/μ/var are the
  driver's running stats, graph inputs shared by the whole batch, i.e. batch-INVARIANT data (§4's
  rule). `den_batchOp_bnEval` says `den = batchMap N (bnPerChannelEvalTensor3 …)` — every example
  normalised by the SAME frozen stats, independently, which is the formal statement of
  "eval is class-batch-independent". `bnBatchF` needs its own constructor precisely because it
  *reduces* across the batch and `batchMap` cannot express that. Four sites, and it emits
  byte-for-byte what `bnPerChannelEvalF` emits — `TestBatchedEmitTie` is now **16 emit ties + 23
  grad-prefix checks**, and the new case was verified to go red.
- **`tests/TestFwdTie.lean` → `lake build fwd-tie`** — one net-agnostic forward tie replacing
  `resnet34-fwd-tie` (which was it with one net hardcoded; `fwd-tie resnet34` is the same check).
  Two improvements: it **deletes its `.vmfb` before every compile** (§4's false-PASS hazard, which
  bites exactly when running a control), and it **counts bit-exact coordinates**, because
  `Float.toString` prints a genuine 3e-8 as `0.000000`.
- **The prefix audit now covers four nets**, not one, and is anchored on the first `%v0 = ` line
  rather than a fixed line count — ConvNeXt's train step emits two header constants its forward
  does not, so the fixed offset only ever worked for R34. Verified to fail: flipping one `rsqrt` in
  `mobilenetv2_fwd` reports the exact diverging body line.

#### Controls — every green tie was shown capable of going red

| net | control | result |
|---|---|---|
| ConvNeXt | 22 LayerNorm ε, 1e-6 → 1e-3 | **fires**, rel 5.9e-2, 0/320 |
| ConvNeXt | GELU cubic coeff 0.044715 → 0.05 | **fires**, rel 3.7e-1, 0/320 |
| EfficientNet | 49 BN ε, 1e-5 → 1e-3 | **fires**, rel 4.4e-1, 0/320 |
| EfficientNet | **stat-slot misalignment** — `b13en` ↔ `b14en` reads swapped | **fires**, rel 5.6e-1 |
| ConvNeXt | GAP divisor 49 → 48 | ⚠ **does NOT fire** — see below |

The stat-slot control is the valuable one: §2e names "a misaligned stat slot is silent" as a hazard
you have to be careful about, and it is now an executable check. Getting it right took two tries —
renaming the parameter in the signature *and* at the use site is an alpha-rename and came back
bit-identical. **The func-arg POSITION is what binds a statistic to a slot.**

And the GAP control is a genuine finding rather than a gap in the gate: ConvNeXt's head LayerNorm
immediately normalises the pooled features, and LN is scale-invariant up to ε, so rescaling the GAP
divisor is absorbed before the dense layer (rel 5.1e-5). It is a near-identity perturbation of *that
net*. Same lesson as §2d.1's reversed-bs32 control: **a control that produces no difference cannot
calibrate one** — pick a different perturbation rather than loosening the gate.

#### What is NOT established

The ties compare **logits only** — 320 floats, not a whole-net fingerprint like the AdamW ties'
`bnstat` region. A forward render has no other output to gate. That is strong (a mis-wiring at
layer k perturbs every logit) but it is not the AdamW ties' 12M-float comparison, so say
"the two renders compute the same logits", not "the same intermediates". The prefix audit is what
covers the intermediates, and only for the `.train` artifacts.

### 2h. The `-xla` trainers for mnv2, ConvNeXt and ViT ✅ DONE 2026-07-29 (ViT: plumbed, unrunnable)

**mnv2 and ConvNeXt now have working XLA/PJRT trainers; ViT is plumbed and still dies in MIOpen.**
The 2026-07-28 scoping below held exactly — 3 files + 2 lakefile entries per net, no driver change,
and the two nets predicted clear were clear.

#### ▶ The result

| | `-xla` binary | agreement with IREE | descends on certified bytes | marginal epoch (XLA) |
|---|---|---|---|---|
| **MobileNetV2** | ✅ `mobilenetv2-verified-adam-xla` | first step **4.9e-5**; epoch-1 loss 3.5e-5, val 594 vs 600 / 3925 | ✅ 26.7 → 38.0 → 52.3 → **59.9%** (4 full epochs); **68.2%** over 80 capped-step epochs | **58.0 s** → 80 ep ≈ **1 h 17 m** |
| **ConvNeXt-T** | ✅ `convnext-verified-adam-xla` | first three steps ≤ **1.6e-5**; epoch-1 loss and val_acc **IDENTICAL** (2.903300, 502/3925) | ✅ 36.0 → 47.7 → 54.3 → **60.6%** (4 full epochs) | **84.5 s** → 80 ep ≈ **1 h 53 m** |
| **ViT-Tiny** | ⛔ builds, **does not run** | — | — | — |

Per-epoch deltas were **58/58/59 s** and **84/85/85 s**, so the marginal is stable, not a lucky pair.
Both figures are train **+ eval**, matching §2e-quinquies' primary row, and both were measured solo.

**The IREE side was deliberately NOT benchmarked** — the point of this work is that IREE is the slow
path, and a ratio was not worth the wall clock. So **no XLA:IREE speedup is quoted for these two
nets**; EfficientNet's 4.6× (§2e-quinquies) remains the only measured one and is still one net's
number. The one incidental data point: mnv2's IREE epoch 1 stamped at 336 s against XLA's 75 s, both
including startup — indicative of a ~5× gap, **not a marginal-epoch measurement**, do not quote it.

**ConvNeXt reproduces epoch 1 EXACTLY across two independent lowerers** — same loss to all six
printed decimals *and* the same 502/3925 val_acc, over a 20-step epoch plus a full 3925-image eval.
mnv2 does not quite (6 images of 3925), and the difference tracks **BatchNorm**: ConvNeXt is
LayerNorm, so it has no batch statistics for fp differences to land in. That is sharper than the
"close but not bit-exact" this section predicted, and it is a *cross-lowerer* result, not a
same-backend one. Re-measured independently at the full-epoch config (295 steps rather than 20):
ConvNeXt's first three steps agree to 1.6e-5 / 2e-6 / 9e-6, mnv2's to 4.9e-5 / 6.3e-4 / 1.5e-4.

**⚠ The trap this thread actually hit was not in the plan — a stale CHECKPOINT.** All three nets had
a June `.lake/build/<slug>_adam_ckpt.bin` from the *retired* hand-written renders, and mnv2's was at
`epoch=80`. Nothing fingerprints the graph, so the first gate run would have silently resumed June's
optimizer state on the new certified bytes, and mnv2's would have been a silent no-op. See §4.

#### Why it was worth doing

* **Speed.** XLA is **4.6× IREE** on EfficientNet (§2e-quinquies: 71 s/epoch vs 354 s; 80 epochs
  1 h 35 m vs 7 h 50 m). `mobilenetv2-verified-adam` and `convnext-verified-adam` are IREE-only
  today, so every long run on those two nets is paying that factor. Re-measure per net rather than
  assuming 4.6× — it is one net's number, on a net that is depthwise-convolution-heavy.
* **Multi-GPU is only reachable this way.** Collectives exist on the PJRT path only; the IREE shim
  refuses a DP entry point outright. ConvNeXt has no `replicas` support in its renderer at all and
  mnv2 has support but no artifact (§0b), so DP is a *later* step for both — but it is unreachable
  without the `-xla` binary first.
* **§0b's first open item gets cheap.** mnv2 and ConvNeXt have **no training run at all** on their
  current artifacts — the evidence for both swaps is graph-level. The smoke-train that closes that
  gap is the natural first use of the new binary, at 4.6×-ish less wall clock.

#### ✅ The risk was measured, and mnv2 + ConvNeXt are CLEAR

The obvious worry was ViT's blocker generalising: `miopenStatusUnknownError` in a strided-conv
**weight gradient**. ConvNeXt has a structurally similar op — the 4×4/s4 patchify weight grad
(`convStride4WeightGrad`, §2f-bis), whose cotangent dilates to a large filter exactly as ViT's
16×16/s16 does. So both backward graphs were run on XLA/PJRT before any plumbing was written, using
`sgd-render-tie` (which is `xlaLink`) as an A-vs-A probe:

| net | XLA compile | execute | A-vs-A gradient |
|---|---|---|---|
| **ConvNeXt** `@convnext_train_step` (180 params) | 6.6 s | ✅ **runs** | bit-exact, 0/180 disagree |
| **MobileNetV2** `@mobilenetv2_train_step` (210 params) | 3.0 s | ✅ **runs** | **bit-exact 2253738/2253738** |
| **ViT** `@vit_train_step` (200 params) | ✅ compiles | ⛔ **`miopenStatusUnknownError`** | — |

**So ConvNeXt is NOT ViT-shaped**: its stride-4 patchify weight gradient runs fine on this box's
MIOpen, which further narrows the ViT blocker to that one 16×16/s16 patch-embed shape rather than
to strided weight gradients as a class. mnv2 was never at risk (its depthwise weight grads are
EfficientNet's, which §2e-bis already runs on XLA), and now it is measured rather than argued.

**New ViT fact, and it changes the repro:** the **SGD** train step fails *identically*. The blocker
is therefore **not AdamW-specific** — it is in every ViT graph carrying a backward, since both
renders contain the same two convolutions. A minimal repro needs no AdamW machinery at all, which
makes the §2a "no bug filed until there is a self-contained repro" bar materially easier to clear:
start from `vit_train_step.mlir` (202 in / 200 out, 2 convolutions) and cut down.

(`sgd-render-tie` now accepts `vit` in its `netBySlug` so this probe is repeatable — one line, and
ViT's train step is already the `(x, θ, onehot) → θ'` shape that harness drives.)

##### ▶ Is it purely a ROCm/MIOpen thing? Probably — and the CUDA box settles it cheaply

Raised 2026-07-29. **Everything known points at MIOpen rather than at the graph**, and three of the
facts are already measured:

* the failure is `miopenStatusUnknownError`, a **ROCm conv-library enqueue failure** with no cuDNN
  analogue — it is not a shape rejection, an OOM, or an XLA verifier error;
* the **identical graph runs under IREE on the same AMD GPU** (`vit-adam-tie` is `ireeLink`), so the
  render is fine and the fault is in the XLA→MIOpen lowering;
* the isolated convolution **succeeds** in JAX on this box, and ConvNeXt's structurally similar
  stride-4 patchify weight-grad **succeeds** under XLA/ROCm — so MIOpen handles the shape when
  called directly, and what fails is the *fused* call (the `rhs_dilation = 16` hypothesis above).

A ViT-Tiny patch-embed backward is also about as ordinary as a convolution gets, and ViT trains
under JAX/XLA on CUDA routinely. **But nobody has run it on CUDA**, so this is inference from the
error surface, not a measurement — hold it to the same bar as the bug report itself.

To settle it, do NOT carry the trainer over. `@vit_train_step` (202 in / 200 out, two convolutions)
fails *identically* and is far smaller, so **`sgd-render-tie vit` is the probe**. One prerequisite:
a **CUDA PJRT plugin** via `PJRT_PLUGIN` (the jax CUDA plugin ships one) — `ffi/pjrt_ffi.c` is
plugin-agnostic but does not conjure one. If it runs there, the open ViT item converts from
"unexplained" into a filable ROCm bug with a two-op repro, which is exactly the bar §2a set.

#### The recipe — 3 files + 2 lakefile entries per net, and NO driver change

`VerifiedNet.trainAdamSched` already takes `(variant : String := "adam")` and already picks its
banner off `IreeSession.backendName`; checkpoints and `.vmfb` paths are already backend- AND
variant-scoped (§4). **Nothing in `LeanMlir/` needs to change.** Copy the shape of
`apps/imagenette/EfficientNetAdamCommon.lean` (48 lines) exactly:

1. **`apps/imagenette/<Net>AdamCommon.lean`** — the config + a `run<Net>Adam (argv) : IO Unit`,
   moved verbatim out of the existing main. Lake requires a distinct root module per executable, so
   this exists to stop the two binaries drifting: *"drift in `epochs`, `batchSize`, the seed, or any
   AdamW hyperparameter would quietly invalidate any cross-backend comparison."* Keep each net's own
   numbers — mnv2/ConvNeXt are baseLR 1e-3 with 3-epoch warmup, ViT is 3e-4 with 5.
2. **`Main<Net>VerifiedAdam.lean`** — reduced to `def main (argv) := run<Net>Adam argv`.
3. **`Main<Net>VerifiedAdamXla.lean`** — the same one line, plus the docstring naming the backend.
4. **lakefile**: the existing `lean_exe` keeps `ireeLink`; the new `<net>-verified-adam-xla` gets
   `xlaLink` and the new root.

**Thread `variant` while you are in there.** `MainMobilenetV2VerifiedAdam.lean` and
`MainConvNeXtVerifiedAdam.lean` call `trainAdamSched … 3` and never pass a variant, so they cannot
select anything but `adam`. EfficientNet's common reads `LEAN_MLIR_VARIANT` (and `LEAN_MLIR_BATCH`);
copy that, or the DP step later needs the file edited again. ViT's main additionally throws if its
artifact is missing — keep that.

#### The gates, as run

| gate | mnv2 | ConvNeXt | ViT |
|---|---|---|---|
| both binaries build; `ldd` shows the right `.so` per target | ✅ | ✅ | ✅ |
| refactor inert — config values and hyperparameters moved **verbatim**, `variant` defaults to `adam` so the previously hardcoded path is reproduced | ✅ | ✅ | ✅ (`vit_{variant}_…` = the old literal at `adam`; the missing-artifact throw was re-verified by asking for a bogus variant) |
| IREE-vs-XLA agreement, same seed | ✅ (above) | ✅ (above) | n/a — XLA cannot run |
| loss descends on the XLA binary | ✅ 4 full epochs | ✅ 4 full epochs | ⛔ |
| marginal-epoch wall clock `(T₃ − T₁)/2` | ✅ 58.0 s | ✅ 84.5 s | ⛔ |
| the IREE peer still trains after the refactor | ✅ | ✅ | ✅ 3.505 → 3.284 → 2.836 |
| `regen_verified_mlir.sh check` unchanged | ✅ 55 artifacts, one writer each, prefix audit green on four nets | | |

None of this touched `verified_mlir/` — no tracked artifact changed, which is what makes the whole
change a plumbing change rather than a re-render.

**`scripts/marginal_epoch.sh` was added** to make the wall-clock gate hard to get wrong: it stamps
each epoch line and prints `(T₃ − T₁)/2`, because wall-clock-minus-compile has produced a wrong
number twice in this thread (§2e-ter) and the ~7.45 GiB one-time dataset load lands entirely in
epoch 1.

#### What is NOT done

* ~~**The mnv2 DP render**~~ ✅ **DONE 2026-07-29 — §2h-bis.** ~~**ConvNeXt's is still open**~~
  ✅ **DONE 2026-07-29 — §2h-quater.** The estimate held: it was a renderer change rather than an
  `#eval` (`grep -c replicas ConvNeXtRender.lean` was 0), the driver needed nothing, and it came to
  ~40 lines plus a harness. **Every large net except ViT now has a gated DP path.**
* **Re-measuring the published accuracies** — mnv2 owes this anyway for an unrelated reason (§2g: it
  was scored through the wrong forward). The 4-epoch runs here are descent evidence, not accuracy.
* **An XLA:IREE ratio for these two nets** — deliberately skipped, see above.
* **ViT on XLA anywhere.** Try the CUDA box with `sgd-render-tie vit` first (above).

### 2h-bis. MobileNetV2 data-parallel ✅ DONE 2026-07-29 — gated by the EXACT identity on the real net

`verified_mlir/mobilenetv2_adamdp_train_step.mlir`, written by a second `#eval` in
`Proofs/Codegen/MobileNetV2RenderB.lean` at `replicas := 2`; `LEAN_MLIR_VARIANT=adamdp`, entry
`@mobilenetv2_adamdp_train_step`. It renders to its **own** path, so the artifact the trainer runs
is untouched. Needs `mobilenetv2-verified-adam-xla` (§2h) — collectives exist only on the PJRT path.

This was the cheap half of the DP work, exactly as §2h predicted: the renderer already took
`replicas`, already called `emitGradAllReduce`, and `mnv2AdamVariant` already returned the slug.
**One `#eval` and one harness**; no renderer logic changed.

| gate | result |
|---|---|
| `replicas = 1` re-render vs the committed artifact | **byte-identical** — the insertion is provably inert ✅ |
| collectives emitted | **210**, one per parameter, each paired with a `/2.0` ✅ |
| syntax | `all_reduce(add)` over `[[0, 1]]`, **no `use_global_device_ids`** (§4) ✅ |
| carve-out declared in the emitted output | yes, at `replicas > 1` ✅ |
| `// MALFORMED` | 0 ✅ |
| single-device artifact contains a collective | **0** ✅ |
| `regen_verified_mlir.sh check` | **56 artifacts, one writer each** ✅ |
| **2 GPUs, 2 replicas — the exact identity** | **PASSES** ✅ |

**`tests/TestMobilenetV2DpCheck.lean` → `lake build mobilenetv2-dp-check`.** The §2e-bis identity,
which mnv2 can use for the same reason EfficientNet can: give both replicas the **same** 32
examples, so their BN groups are identical by construction, `all_reduce(add)/2 = (g+g)/2 = g` is the
identity, and the DP step must reproduce the single-device one. **Gated on the real net, not a
proxy** — R34's collective still only has the cifar8 proxy (§2b-quater).

| region | result (2× 7900 XTX, `PJRT_REPLICAS=2`) |
|---|---|
| **`bnstat`** (forward, 52 BN layers) | **BIT-EXACT 34112/34112** ✅ |
| **`%loss`/bc** | **BIT-EXACT 3/3** ✅ |
| `v` | **BIT-EXACT 2253738/2253738** ✅ |
| **gradient (`m`)** | norm-rel **8.7e-8**; 35 of 2,253,738 coords differ, max abs 8.1e-9 ✅ |
| θ | norm-rel 5.7e-12, 22 coords differ |

~1150× inside the 1e-4 gate. As on EfficientNet it is **not** bit-exact on `m`/θ, and for the same
reason — the DP module is a different HLO program so XLA orders the backward reductions differently;
the collective itself is exact, `(g+g)/2 = g` holds to the bit. The forward being bit-exact is the
evidence for that split. `Float.toString`'s six decimals print 8.7e-8 as `0.000000`, so the harness
reports nano-units and the exact-coordinate count — without them this reads as bit-exact when it is
not.

**Verified to fail.** `%arn` divisor 2.0 → 1.0 (sum, not mean; 210 lines changed) → gradient
norm-rel **1.037** against a passing 8.7e-8, **seven orders of separation**, and the harness exits 1.
It is also the fifth net to re-demonstrate §3's rule: the same broken render moved **θ by only
8.4e-5** — *under* a 1e-4 θ gate — while `m` moved 1.037. And `bnstat` stayed **bit-exact**, which
correctly localises the fault to the backward rather than the forward.

Claim ceiling unchanged (§5): *certified gradient → trusted collective → certified AdamW*. What is
new is that mnv2 joins EfficientNet as a net whose collective semantics are pinned by an exact
known-answer check **on the net itself**. The DP renders now rank: **EfficientNet and MobileNetV2**
(exact identity, real net, 2 GPUs) > **ResNet-34** (cifar8 proxy + 2-GPU descent) > **ViT** (nothing
numeric — will not execute here).

#### 2h-ter. mnv2 DP end to end — it trains, and it scales at **1.67×**

The identity gate says the DP step computes the right thing; these say what it costs and that the
trainer actually drives it.

**It trains.** `LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2` on
`mobilenetv2-verified-adam-xla`: the banner reports *"DATA-PARALLEL: 2 replicas x bs 32 = global
batch 64"*, loss descends 2.506 → 2.291 → 2.128 and val reaches 17.8% at epoch 1.

**Marginal epoch, train-only, measured solo on 2× 7900 XTX** (`scripts/marginal_epoch.sh`):

| | marginal epoch | deltas |
|---|---|---|
| 1 GPU, bs 32 | **55.0 s** | 55, 55 |
| **2 GPU DP, global 64** | **33.0 s** | 33, 33 |

**1.67× end-to-end** — the *same* figure EfficientNet's DP reaches (§2e-ter), which is a useful
independent confirmation that the shortfall from 2× is structural rather than net-specific: params
are host-resident, so each step pushes the full `[θ|m|v]` to every replica (§2c). That is the
standing argument for device-resident parameters (§2d.3), now with a third measurement behind it.

**Eval is NOT the bottleneck on either path**, which is worth stating because a single ad-hoc run
suggested otherwise and did not reproduce. Single-device eval costs **3.0 s/epoch** (58.0 train+eval
vs 55.0 train-only, §2h). Under a 2-replica process it costs **~5 s/epoch** — measured with a
5-train-step cap so the epoch time *is* essentially the eval pass: stamps at 29/35/40/45 s, deltas
6/5/5. Eval runs single-replica while the process holds a 2-replica executable (§2c: the replica
count is per-graph), and that costs a couple of seconds, not the order of magnitude first suspected.

**Still not done:** an interleaved ms/step DP bench of the §2e-ter kind (both executables compiled in
one process, synthetic inputs, min statistic). The numbers above are marginal epochs, which is the
right measurement for planning a run but is not the same thing as an on-GPU throughput ratio —
EfficientNet's on-GPU 1.75× and end-to-end 1.67× differ for exactly that reason.

### 2h-quater. ConvNeXt data-parallel ✅ DONE 2026-07-29 — the last net, and the first rank-0 collectives

`verified_mlir/convnext_adamdp_train_step.mlir`, written by a second `#eval` in
`Proofs/Codegen/ConvNeXtRender.lean` at `replicas := 2`; `LEAN_MLIR_VARIANT=adamdp`, entry
`@convnext_adamdp_train_step`. It renders to its **own** path, so the artifact the trainer runs is
untouched. Needs `convnext-verified-adam-xla` (§2h) — collectives exist only on the PJRT path.

**This was the expensive half of the DP work, as §2h predicted** — unlike mnv2 (§2h-bis, one `#eval`
because its renderer already took `replicas` and already called `emitGradAllReduce`), ConvNeXt had
**no `replicas` support at all**: `grep -c replicas ConvNeXtRender.lean` was 0. So the parameter had
to be threaded through `convnextAdamOne` and the render, plus the import of `LeanMlir.ViTRender`
that no ConvNeXt file had. Still small — ~40 lines of renderer, one `#eval`, two `#guard`s, and the
harness. `cnxAdamVariant` is the single source for the entry name, the artifact path and
`LEAN_MLIR_VARIANT`, pinned by `#guard`s per §2d.1.

#### ▶ The one genuinely new thing: **44 of the 180 collectives are RANK-0**

ConvNeXt's scalar LayerNorm γ/β params have `ds = []`, i.e. `tensor<f32>`. **No other net's DP render
has an operand below rank 1** — R34, EfficientNet, mnv2 and ViT are all ≥1-D — so
`stablehlo.all_reduce` on a scalar had never been emitted, let alone executed, anywhere in this repo.
It compiles and runs:

```mlir
%arsums0b0ng = "stablehlo.all_reduce"(%v3399) ({
^bb0(%aras0b0ng: tensor<f32>, %arbs0b0ng: tensor<f32>):
  %aradds0b0ng = stablehlo.add %aras0b0ng, %arbs0b0ng : tensor<f32>
  stablehlo.return %aradds0b0ng : tensor<f32>
}) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (tensor<f32>) -> tensor<f32>
```

Worth knowing because it is the kind of thing that would have failed at *execution* rather than at
render or compile, which is the ViT failure mode — so it was checked before anything was measured.

| gate | result |
|---|---|
| `replicas = 1` re-render vs the committed artifact | **byte-identical** — the insertion is provably inert ✅ |
| collectives emitted | **180**, one per parameter, each paired with a `/2.0` ✅ |
| syntax | `all_reduce(add)` over `[[0, 1]]`, **no `use_global_device_ids`** (§4) ✅ |
| carve-out declared in the emitted output | yes, at `replicas > 1` ✅ |
| `// MALFORMED` | 0 ✅ |
| single-device artifact contains a collective | **0** ✅ |
| `regen_verified_mlir.sh check` | **58 artifacts, one writer each**, prefix audit green ✅ |
| **2 GPUs, 2 replicas — the exact identity** | **PASSES** ✅ |
| 2-GPU descent | loss 2.127 → **0.897** over 12 epochs ✅ |

**`tests/TestConvNeXtDpCheck.lean` → `lake build convnext-dp-check`.** The §2e-bis duplicated-batch
identity — and **ConvNeXt is the one net that needs no BatchNorm argument to justify it.** The
§10.3b caveat is a statement about batch BN: 2×32 genuinely is not 1×64 because the halves get
different statistics. EfficientNet and mnv2 have to argue that *duplicating* a batch keeps both
replicas' BN groups identical. ConvNeXt normalises with **LayerNorm**, which reduces within one
example and never across the batch, so nothing couples the replicas at all and there is nothing to
argue.

| region | run A | run B |
|---|---|---|
| **`%loss`** (the only forward-only output) | **BIT-EXACT 1/1** ✅ | **BIT-EXACT** ✅ |
| `v` | **BIT-EXACT 27811542/27811542** ✅ | **BIT-EXACT** ✅ |
| **gradient (`m`)** | norm-rel **1.07e-8**; **3** of 27,811,542 coords differ, max abs 9.3e-10 | **BIT-EXACT 27811542/27811542** |
| θ | norm-rel 9.1e-13, 3 coords differ | **BIT-EXACT** |

**Quote it as ≤ 1.1e-8, not as bit-exact.** Two runs of the same binary on the same artifacts gave
different answers — one bit-exact on all **83,434,629** floats, one with three gradient coordinates
off at ~1e-9. That is §3's cross-process autotuning again, and it is exactly why the rule is to gate
with headroom and quote a bound. ~9000× inside the 1e-4 gate either way.

It is nonetheless the **tightest DP agreement of any net here** — 3 coords against mnv2's 35 and
EfficientNet's 14, and the only one that has ever come back bit-exact — which is the LayerNorm story
again: with no batch statistics there is less for a reordered reduction to land in.

**The forward gate is `%loss`, and that is a real difference from the BN nets.** ConvNeXt returns no
batch statistics, so there is no `bnstat` region to pin the forward bit-exactly the way mnv2's 52 BN
layers do. `%loss` is the only output that reads the forward alone — report-only, on no gradient
path, covered by no theorem, i.e. precisely the configuration in which §2b shipped plain CE against
a smoothed-CE cotangent. So the harness **gates** it at 1e-4, the same split `convnext-adam-tie`
uses on this net.

**Verified to fail.** `%arn` divisor 2.0 → 1.0 (sum, not mean; 180 lines changed) → gradient norm-rel
**1.114** against a passing ≤1.1e-8, **eight orders of separation**, and the harness exits 1. Two
things it re-demonstrates, both worth having on a sixth net:

* **§3's rule, more sharply than anywhere else.** The broken render moved **θ by 9.7e-5** — *under*
  a 1e-4 θ gate — while `m` moved 1.114. A θ-based gate would have passed a 2× gradient error by a
  margin of 3%.
* **The fault localises.** `%loss` stayed **BIT-EXACT** through the control, which is correct: the
  collective sits downstream of the loss, so a broken divisor cannot touch it. The loss gate is
  doing independent work rather than shadowing the gradient gate.

#### It trains, and it scales at **1.68×**

`LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2` on `convnext-verified-adam-xla`: the
banner reports *"DATA-PARALLEL: 2 replicas x bs 32 = global batch 64, 147 steps/epoch"* and the loss
descends **2.127 → 1.834 → 1.564 → … → 0.897** over 12 epochs.

Marginal epoch (`scripts/marginal_epoch.sh`), **train-only on both sides** so the ratio is not
contaminated by an eval pass that runs single-replica, measured solo on 2× 7900 XTX:

| | marginal epoch | deltas |
|---|---|---|
| 1 GPU, bs 32 | **77.5 s** | 78, 77, 78, 78, 78 |
| **2 GPU DP, global 64** | **46.0 s** | 46, 46, 45, 46, 46 |

**1.68×** — within noise of the **1.67×** that EfficientNet (§2e-ter) and mnv2 (§2h-ter) both reach.
Three architecturally unlike nets landing on the same ratio is the strongest evidence yet that the
shortfall from 2× is **structural**: parameters are host-resident, so every step pushes the full
`[θ|m|v]` to every replica (§2c). That is now **four** independent measurements pointing at
device-resident parameters (§2d.3).

Note the single-GPU 77.5 s here is **train-only** and so is not the 84.5 s in §2h, which is
train + eval on the same net. Eval costs ConvNeXt ~7 s/epoch. Do not mix the two.

**Still not done**, same two as mnv2: an interleaved ms/step DP bench of the §2e-ter kind (these are
marginal epochs, not an on-GPU throughput ratio), and a re-measured published accuracy.

#### ▶ A stale claim inside a committed artifact, found and fixed here

`convnext_adam_train_step.mlir`'s own banner still said its stem 4×4/s4 and 2×2/s2 downsample weight
gradients *"have no VJP-cert SHlo op and stay hand-written (the two documented gaps)"* — true when
written, **false since `9bb00f5`** (§2f-bis) closed both. The artifact was **under**-describing its
own certification level, which is the benign direction but is still a wrong statement in the one
place a reader would trust it.

Fixed, as a **separate** change from the DP threading and in that order deliberately: the DP work's
cheapest gate is *"`replicas = 1` re-renders byte-identical"*, and editing the banner destroys it.
So the threading landed first and was shown byte-inert, then the banner moved on its own, with the
diff as the check — **3 changed lines, all comments, 0 non-comment lines changed**. MLIR discards
comments lexically, so no re-tie is owed; the changed bytes were nonetheless compiled and executed
by the dp-check re-run above.

*(The same `git diff verified_mlir/` discipline §2a-quinquies asks for is what makes this checkable
at a glance — it is the only tracked artifact this whole thread modified.)*

### 2i. ▶ NEXT THREAD — the last 13 `tests/`-rendered artifacts (the cifar8 / cifar8w port)

Scoped 2026-07-29. **§1's "every artifact is `Proofs/`-rendered" is FALSE** and should be corrected
whenever it is quoted: it holds for the five large nets and the demo ladder, but **two files in
`tests/` still own independent emitters** (`emitMomentum`, `emitSgd` — not delegators) and write 13
artifacts between them:

| writer | artifacts |
|---|---|
| `tests/TestCifar8AdamTrain.lean` | `cifar8_{bn_adam, mom, bn_mom, sgd, bn_sgd}_train_step` — **5** |
| `tests/TestCifar8WideTrain.lean` | `cifar8w_{adam, bn_adam, mom, bn_mom, sgd, bn_sgd}_train_step` + `cifar8w_fwd` + `cifar8w_bn_fwd` — **8** |

**They are LIVE**, not dead ablation bytes: `cifar8-bn-verified-adam{,-xla}`,
`cifar8-verified-momentum`, `cifar8-bn-verified-momentum`, `cifar8-verified-sgdsched`,
`cifar8-bn-verified-sgdsched`, `cifar8w{,-bn}-ablation`. **The ablations are the POINT** — that
section of the demo exists to show SGD several ways — so these get ported, not deleted.

**The audit does not flag them.** `regen_verified_mlir.sh check` counts *duplicate* writers, not
provenance; 13 hand-written artifacts with one writer each report green. And no automated tie covers
any of them — `fwd-tie` and `sgd-render-tie` both hardcode `netBySlug ∈ {convnext, efficientnet,
mobilenetv2, resnet34}` (+`vit`).

**Two probes already run, both CLEAN** — so these are *unverified*, not *suspect*:
* **BN semantics** — every cifar/cifar8/cifar8w artifact computes statistics per (example, channel);
  the batch axis survives everywhere. `cifar8_bn_fwd` is a byte-identical **285-line prefix** of the
  `Proofs/` `cifar8_bn_train_step`. ⚠ See §4's layout-trap note: an earlier version of this section
  reported a two-worlds skew here that **does not exist** — the census had compared reduce-dimension
  literals across two different tensor layouts.
* **lr / cotangent** (the §2a-quinquies probe that caught EfficientNet's 16×) — all 13 are
  **mean-CE ÷128 with a runtime `%lr`**, matching their certified siblings; they differ only in
  SPELLING (`divide by 128.0` vs `multiply by 0.0078125`). The silent-hyperparameter failure mode
  does **not** reproduce.

#### The op gap, measured

`adamWParamF` is an **own constructor** (`SHlo` case, `den` case, faithfulness theorem — the 10-site
route, *not* the 4-site `.batched` descriptor). **Nothing momentum-shaped exists anywhere in
`Proofs/`** — not emit-side, not denotation-side. So:

| piece | cost | status |
|---|---|---|
| **scheduled SGD**, no-BN | one new op, 10 sites — `θ − lr·g` with **lr as a runtime operand** (the fused `*Sgd` ops bake lr as a *literal*, which is why it was absent) | ✅ **DONE** `769d0ab` + `d5fbf32` |
| **Nesterov momentum**, no-BN | two new ops + a denotation-side `momStep` — new *math*, not just emit | ✅ **DONE** `769d0ab` + `d5fbf32` |
| **the three BN variants** | ⚠ bigger than a tail swap (shape table, conditional signature, explicit cotangent) — and the shape table turned out to be **avoidable**: one `bnSig` list is the single source for the signature, the m/v blocks, the return types and every `optTail` call | ✅ **DONE** 2026-07-30 |
| **`cifar8w*`** (8 artifacts) | ⚠ NOT a second net and NOT ablation-only — see below | ✅ **DONE** 2026-07-30 |

#### ✅ Done 2026-07-29 — the two no-BN variants, and the shape that made them cheap

`cifar8_sgd_train_step` and `cifar8_mom_train_step` are `pretty(provenGraph)` out of `CnnRender`,
each `#eval` its artifact's only writer. **`CifarOpt = adamw | sgd | nesterov` threaded through ONE
renderer** (`adamTail` → `optTail opt`, 22 call sites), so all three variants share one forward, one
backward and all 22 un-fused gradients and only the tail differs — the ablation section is now
genuinely "the same net with the optimizer swapped". `%mu` emits only for `nesterov`, which is what
kept the three AdamW artifacts **byte-identical** (gate 1).

Tie: `cifar8-opt-tie {sgd,mom}` — **BIT-EXACT 52858/52858** recovered-gradient coords, `%loss`
identical, m/v passthrough bit-exact. Controls fire: ÷B 0.0078125→0.008 ⇒ 0.024; μ 0.9→0.91 ⇒
1.6e-4 with **0/52858** exact. ⚠ That second control clears the 1e-4 gate by only **1.6×** (μ→0.91
is a 1% perturbation entering as `0.01·v`, v≈0.05) — **the exact COUNT is the decisive
discriminator there, not the magnitude.**

**`cifar8-opt-tie` gates the RECOVERED GRADIENT, never θ′**, and for SGD that is the whole point:
θ' = θ − lr·g is dominated by θ, the same input on both sides, so at lr 1e-3 a wholly wrong gradient
still looks like a match (§2a-quinquies). Recovery is exact per variant — adam from `m'`, sgd from
`θ'`, mom from `v' = μv + g`. It also gates the m/v **passthrough** slots bit-exactly (a tail that
silently dropped a moment would still yield a plausible θ'), counts bit-exact coordinates (the first
run printed `0.000000`, which six decimals cannot distinguish from 3e-8 — §2e-bis), and **deletes its
`.vmfb`** before every compile, which `cifar8-adam-tie` still does not.

#### ✅ Done 2026-07-30 — the three BN variants. **5 of 13**, and the ONE real trainer is certified

`cifar8_bn_{adam,mom,sgd}_train_step.mlir` are `pretty(provenGraph)` out of
`cifar8BnTrainStepFaithfulV`, each `#eval` its artifact's only writer; the three `IO.FS.writeFile`
calls in `tests/TestCifar8AdamTrain.lean` are retired (the renders stay, as the ties' references —
the §2a-quater shape). **`cifar8_bn_adam` was the only one of §2i's 13 backing a real trainer**
(`cifar8-bn-verified-adam{,-xla}`), so the remaining 8 are ablation-only.

**No new ops, exactly as scoped.** `opt : Option CifarOpt` threaded through the EXISTING renderer:
`none` renders the fused 40/38 incumbent, `some o` the packed 119/117 variant. The three branch
points the scoping predicted were the three that were needed — cotangent, conv-bias names
(`%b1..%b8` → `%cb1..%cb8`, since AdamW bakes β₁/β₂ as `%b1`/`%b2`), and signature/return.

**The 38-entry shape table was avoidable and that is the better answer.** One `bnSig : List (String
× List Nat)` is the single source for the arg signature, the packed `m`/`v` blocks, the return types
AND each `optTail`'s `(pName, n, ds)` — derived from one entry per param rather than a parallel
hand-list, so a shape or slot cannot drift between signature and tail (§2e's silent-slot hazard).
Gate 1 came for free and held: `cifar8_bn_train_step.mlir` re-renders **byte-identical** (md5
`e877dfa1…`) — which also proves the rebuilt-from-`bnSig` signature reproduces the hand-written
literal it replaced — and so do `cifar8_{adam,sgd,mom}_train_step`.

| gate | result |
|---|---|
| gate 1 — fused artifact + the 3 no-BN packed artifacts re-render | **byte-identical**, all four ✅ |
| interface vs the hand-written render | **119 in / 117 out**, arg + return types **and NAMES** positionally identical, all three ✅ |
| `// MALFORMED` | 0 ✅ |
| A-vs-A determinism floor | **bit-exact 53242/53242**, all three ✅ |
| regenerated canonical == the tied bytes | **byte-identical** before deleting the staged `_cert` paths ✅ |
| post-swap tie, retired render vs new canonical | passes, all three ✅ |
| `regen_verified_mlir.sh check` | **61 artifacts, one writer each**, prefix audit green ✅ |
| elaborating the retired `tests/TestCifar8AdamTrain.lean` | rewrites **nothing**; all six `iree-compile` smokes OK on the new bytes ✅ |
| **descends on the swapped bytes** | ✅ loss 2.92 → 1.94, test **13.2 → 20.1 → 25.6 → 30.1%** (4 epochs, 40 steps/epoch, XLA) |

#### ▶ The tie numbers, and why an ABSOLUTE per-param spread gate is WRONG on this net

`cifar8-opt-tie` now drives **six** slugs: the `bn_` prefix picks the net (`cifar8BnVerified` — 38
params, shapes and init kinds from the `VerifiedNetSpec`), the rest picks the recovery formula. All
three BN ties, against a bit-exact A-vs-A floor:

| | gradient norm-rel | reorder control | spread | control's spread |
|---|---|---|---|---|
| `bn_adam` | **1.02e-6** | 1.30e-6 | 8/38 | **8/38 — the same param indices** |
| `bn_mom` | **1.02e-6** | 1.29e-6 | 8/38 | **8/38, same indices** |
| `bn_sgd` | **3.81e-5** | 1.90e-5 | 11/38 | 12/38 — a strict **subset** |

**The two emitters agree with each other as well as the reference agrees with ITSELF** under a
semantics-preserving batch reversal — for adam/mom the test is *tighter* than the control. That is
§2f-bis's ConvNeXt result on a second net, and it arrived the hard way: a first draft of this harness
gated the spread at an **absolute** 1e-4 per parameter, and **that gate fails the REAL tie** (8/38
and 11/38). The gate has to be control-relative, per §2d.1.

**The 8 are the CONV BIASES** — indices 1, 5, 9, … 29 in `(W, cb, γ, β)×8` order — and the harness now
prints the indices, because "the same 8 params in test and control" is much stronger evidence than
"8 in each". Their gradient `Σ_{b,h,w} dy` is a cancelling reduce over 128·H·W terms, so it does not
reproduce to 1e-4 against *any* reordering. Not the BN γ/β, which was the obvious guess.

**`bn_sgd`'s 3.81e-5 is NOT a looser agreement, and the harness now says so in its own output.**
sgd's gradient is recovered as `(θ − θ')/lr` at lr 1e-3, which **amplifies the output-level
difference 1000×**; adam's `(m' − β₁m)/(1−β₁)` amplifies 10×. The new `raw slot` row measures the
un-amplified quantity: **max|θ'A − θ'B| = 3.0e-8** on θ' ≈ 1.0, i.e. ~4 ULPs of binary32 — the same
graph disagreement adam sees through a smaller lens. Without that row the number reads like a
near-miss at 2.6× under the gate; with it, the gate is 234× *below* where the cotangent control lands.

**Four negative controls, each firing its own gate:**

| control | perturbation | result |
|---|---|---|
| A | cotangent `1/128 → 0.008` (1 line) | **magnitude** fires at **2.34e-2** (617× the sgd pass, 23000× the adam pass); spread goes **38/38** — GLOBAL, vs the control's 8–13 |
| B | BN ε `1e-5 → 1e-4` (24 sites) | **`%loss`** fires at 1.52e-4, gradient 6.4e-3, spread 34/38 |
| C | `%loss` divisor `128 → 130` (1 line) | **loss** fires at 4.8e-2 with the gradient **bit-exact 53242/53242** and spread **0/38** |
| D | Nesterov μ `0.9 → 0.91` (1 line) | magnitude fires at 1.01e-3, **0/53242** bit-exact, spread 38/38 |

Control C is the sharpest, as it was on mnv2: it isolates `%loss` completely, which is the direct
evidence that the report-only scalar really is on no gradient path and that its gate does
independent work. Control D lands **10× further above the gate than the no-BN μ control did**
(1.01e-3 vs 1.6e-4), so this net does not have the "clears by only 1.6×" problem the no-BN thread
recorded.

#### ▶ The three BN variants — the 2026-07-29 scoping, kept because it held

Target interface, all three identical and the arithmetic closes exactly:
`1 + 3×38 + 3 + 1 = 119` in, `3×38 + 3 = 117` out (38 params: 8 conv W + 8 conv b + 8 BN γ +
8 BN β + 3 dense W + 3 dense b).

`cifar8BnTrainStepFaithfulV` is the **fused-SGD** renderer — 102 `pretty` calls, of which **38 are
fused `*Sgd` ops** (8 `convWeightSgd`, 8 `convBiasSgd`, 8 `bnGammaSgd`, 8 `bnBetaSgd`, 3 `weightSgd`,
3 `biasSgd`). All six un-fused peers already exist from §2a (`convWeightGrad`, `convBiasGrad`,
`bnGammaGrad`, `bnBetaGrad`, `weightGrad`, `biasGrad`), so **no new ops are needed**. Three things
make it more than the no-BN edit:

1. **`optTail` needs each param's `(n, ds)` EXPLICITLY**, but the fused ops carry their shapes
   implicitly through dependent types (`{ic oc h w kH kW}` inferred from the operand). So the BN
   render needs a **38-entry shape table**, and a misaligned entry is the §2e "silent slot" hazard —
   derive it from the traversal that computes the gradients, never a parallel hand-list.
2. **The signature and return are conditional** — 40/38 fused versus 119/117 packed — so the
   `func.func` line, the return list and the constants block all branch, not just the tail.
3. **The cotangent differs**: the fused render folds the mean into `lrStr`; the packed variants need
   an explicit `scaleF invB` plus a report-only `%loss`, exactly as the no-BN AdamW render does.
   Splitting the currently-nested `.sub (.softmaxDiv (.expe …))` into separate `pretty` calls should
   be byte-neutral (the fresh-name counter continues), but that is an assumption gate 1 must check.

**Do it by threading `opt : Option CifarOpt` through the existing renderer, NOT by writing a twin** —
duplicating 102 `pretty` calls is the double-writer disease one level down, in code, which is exactly
what §2a-quater warns about and what `vitBackAll`/`enetBackAll` exist to avoid. Gate 1 then applies
for free: at `none` the render must reproduce `cifar8_bn_train_step.mlir` byte-identically.

**`cifar8_bn_adam` is still the highest-value one** — it is the only artifact in the whole set of 13
backing a REAL trainer (`cifar8-bn-verified-adam{,-xla}`); the other 12 are ablations. Once the
threading exists, all three BN variants fall out of it together.

**Claim ceiling for this thread:** the goal is `pretty(provenGraph)` parity, **not** a descent
theorem — Adam has none either (§2a: "Still no descent claim — Adam is not monotone"). Say "the
momentum render is certified", never "momentum is proven to descend".

#### The `-xla` demo groups, and the six-way cifar ablation on both lowerers ✅ 2026-07-30

`lake run {mnist,cifar,imagenette}-xla`. `runDemoGroup` grew an `xla : Bool` rather than a twin, so
the two backends cannot drift on GPU selection, backend detection or the `run.sh` contract, and the
XLA path gets two guards the IREE path does not need: `ffi/libpjrt_ffi.so` is **not a lake target**
(it is the documented gcc one-liner), so it is built when missing *or older than* `ffi/pjrt_ffi.c`;
and the PJRT plugin is `dlopen`'d at run time from `$PJRT_PLUGIN` or a path compiled into the shim,
so a missing one is reported up front instead of surfacing as a `dlopen` failure at the first step.

**`cifar-xla` is a complete mirror**: the four missing peers (`cifar8{,-bn}-verified-{momentum,
sgdsched}`) were built with the §2h recipe — a shared `apps/cifar/Cifar8*Common.lean` per arm (plus
a `lean_lib` each, which lake needs to build a module for two roots), the existing main reduced to
one line, a new `…Xla` root, one lakefile entry. Config and hyperparameters moved **verbatim**
(40 epochs, bs 128; momentum 0.02, sgdsched 0.1), and both peers still build.

**Cross-backend agreement, measured on `cifar8-verified-sgdsched` — and CONTROLLED, which matters
more than the headline here:**

| | epoch-1 loss | epoch-1 val |
|---|---|---|
| XLA ×3, same seed | 2.199141 / 2.195176 / 2.199194 | 32.30 / 28.57 / **23.19%** |
| IREE | 2.201572 | 17.92% |

* the **first three steps are BIT-IDENTICAL across the two lowerers** (3.269639 / 2.662815 /
  2.394230) — the strongest form of the §2h gate, matching ConvNeXt's result;
* the cross-backend epoch-1 loss difference (**2.5e-3**) is **inside XLA's own run-to-run spread
  (3.9e-3)**, so it is not backend-attributable;
* epoch-1 **val is a high-variance statistic on this config** — 9.1 points of spread from the same
  binary at the same seed — so the 14-point IREE-vs-XLA gap says nothing. This is §3's
  "XLA is NOT deterministic at epoch scale" on a sixth net. **Do not gate on epoch-1 val here.**

⚠ A capped run (`LEAN_MLIR_G2_STEPS`) is worse still: the warmup/cosine schedule is computed against
the *full* 390-step epoch, so a 20-step epoch runs a mismatched schedule and both backends thrash
near chance (10–19%). Fine for a descent smoke, useless for a comparison.

#### ✅ Done 2026-07-30 — the wide family. **§2i is 13 of 13; the provenance axis is CLOSED**

`scripts/regen_verified_mlir.sh check` reports **61 artifacts, one writer each**, and for the first
time **§1's "every artifact is `Proofs/`-rendered" is true with no carve-out.**

**Both of this section's own assumptions about `cifar8w` were WRONG, and each in the expensive
direction.** They are recorded because the checks that refuted them are cheap and general:

* ~~"plus the wide net rendered at all"~~ — **`cifar8w` IS `cifar8` at `d1 := 512`.** The two
  `VerifiedNetSpec` pairs agree layer-for-layer up to the head width, and the decisive check is a
  byte one: the committed `cifar8w_bn_adam_train_step.mlir` is **byte-identical modulo the entry
  name** to the width sweep's `cifar8_bn_512_adam_train_step.mlir`, which the same hand-written
  emitter already wrote. So no renderer was needed — the six train steps are the two renderers this
  thread had just threaded, at 512, and the two forwards are `cifar8{,Bn}FwdModuleV`, **certified
  renderers that already existed** and that the `cifar8_bn_{d}` sweep already uses. The wide
  `_fwd`s were the last two artifacts still written by hand-written *text* emitters
  (`cifar8FwdText`, `cifar8BnFwdTextPC`).
* ~~"ablation-only — confirm the sweep still earns its keep"~~ — **they are a BOOK table.**
  `runs/ablation_cifar8w/README.md` is the **Chapter-5 "bridge" net**: a 6-cell table (no-BN and BN
  × SGD/Nesterov/AdamW) whose claim — *"head width barely matters: 7.1× the params, accuracy within
  a point, 1.36× wall clock; the depth, not the head, is the lever"* — exists only by comparing wide
  against narrow. All six train steps and both forwards are load-bearing, and the `cifar8_bn_{d}`
  sweep is **adam-only**, covering exactly 1 of the 6 cells. Deleting the family would have voided a
  chapter. **Check what an "ablation" backs before pruning it** — `runs/ablation_<slug>/README.md`
  is where that is written down.

| | tie (incumbent vs certified@512) | reorder control | spread |
|---|---|---|---|
| `cifar8w_{adam,mom,sgd}` | **gradient BIT-EXACT** | bit-exact | **0/22** |
| `cifar8w_bn_{adam,mom}` | 1.0e-6 | 1.3e-6 | 8/38 — the control's own 8 |
| `cifar8w_bn_sgd` | 3.4e-5 | 3.3e-5 | 12/38 ⊂ the control's 14 |
| `cifar8w_fwd`, `cifar8w_bn_fwd` | **logits BIT-EXACT 1280/1280** | — | — |

Every number is the narrow net's, at 7.1× the parameters — the evidence that `d1` is the only thing
that moved.

**Four controls, because three of these ties are BIT-EXACT** and a bit-exact tie is indistinguishable
from a harness comparing a buffer with itself (§4):

| control | fires |
|---|---|
| cotangent `1/128 → 0.008` on `w_adam` (the bit-exact one) | **2.34e-2**, spread **22/22** vs the control's 0 |
| same on `w_bn_sgd` (the thinnest) | **2.34e-2**, spread **38/38** vs the control's 13 |
| BN ε `1e-5 → 1e-3` on `cifar8w_bn_fwd` | logits rel **0.728**, 0/1280 bit-exact |
| max-pool `−inf` init → `0.5` on `cifar8w_fwd` | logits rel **1.977**, 0/1280 bit-exact |

Swap gates as run: regenerated canonical byte-identical to the tied bytes on all 8 before deleting
the staged `_cert` paths; post-swap ties against the retired renders all pass; audit 61/one-writer;
elaborating the retired `tests/TestCifar8WideTrain.lean` rewrites **nothing** and all 8
`iree-compile` smokes pass on the new bytes; and **it descends** — `cifar8w-bn-ablation` on the
swapped bytes goes loss 2.99 → 1.79, test **13.7 → 34.2%** over 8 epochs at 15 steps/epoch.

**Both tie harnesses grew a family rather than a special case.** `cifar8-opt-tie` decomposes its slug
into three independent choices — optional `w_`, optional `bn_`, then the optimizer — so **twelve**
variants share one harness and nothing about a net is hardcoded (shapes and init kinds come from the
selected `VerifiedNetSpec`). `fwd-tie` gained the four cifar8 slugs; its one net-specific constant,
`bs := 32`, became per-family (**the cifar8 forwards are bs 128**), and it refuses `--eval` for them
because per-channel BN keeps no running stats, so train == eval.

### 2j. ✅ DONE 2026-07-30 — `lake run benchmark-xla`, and the mismatched-baseline trap in it

**UNCOMMITTED.** All four scoped items landed, plus the conv peer. What follows is the original
scoping (kept because its reasoning is what the implementation was measured against), then the
results and the corrections it produced.

#### ✅ What landed

| item | state |
|---|---|
| 1. reference constants on XLA, on the 7900 XTX | ✅ `probeDenseRefMsXla := 610`, `probeConvRefMsXla := 3650` — **medians**, see the spread finding below |
| 2. a conv peer | ✅ **`cifar8-bn-verified-xla`** added. §2j's open question — *"check `.train` drives the XLA shim first"* — is **answered yes**: `mnist-mlp-verified-xla` already rides `VerifiedNet.train`, which has an `xla` branch at `VerifiedTrain.lean:216`. So the peer was preferable to re-pointing the probe at a different optimizer, which would have cost a second reference constant and made the two tables non-comparable row by row |
| 3. the script | ✅ `runBenchmark (ref : BenchRef)`, one body, two `script`s. Reuses `ensurePjrtShim`/`notePjrtPlugin`; needs no venv |
| 4. label the output | ✅ both print `lowerer:` in the header and `ref(IREE)` / `ref(XLA/PJRT)` as the column head, plus a footer forbidding cross-table comparison. `lake run benchmark` gained the word IREE, as §2j asked |
| 5. *(not scoped)* the attn probe + ch.10 | ✅ **`benchmark-xla` is 3 probes / 9 chapters**, because ViT turned out to run — see this section's tail. `probeAttnRefMsXla := 128` (median of 8), ch.10 `refSecXla := 3480` (marginal epoch 43.5 s × 80) |
| 6. *(not scoped)* `imagenette-xla` | ✅ **now all five nets**, ViT included |

**The trap is now structural, not documentary.** `BenchRef` bundles one lowerer's probe *binaries*
with its *anchors*, and `yourSecOf` takes a `BenchRef` — so there is no expression that divides an
XLA probe by an IREE anchor. `BenchItem.refSecXla : Option Nat` carries the second column;
`BenchItem.refOn xla` selects. A `none` row prints `n/a`, is excluded from the totals **and** from
the tier subtotals, and flips the footer label to `Part-1 training (8 of 9 ch.)` — so a short total
cannot read as a whole-Part-1 number.

#### The measured XLA reference column

All on the reference 7900 XTX, 2026-07-30, same construction as the IREE column (steady-state
ms/epoch × the trainer's own epoch count), MNIST/CIFAR real-data-plus-eval, last of 3 epochs:

| | IREE ms/ep | XLA ms/ep | XLA speedup |
|---|---|---|---|
| ch2 linear (×12) | 539 | **239** | 2.26× |
| ch3 MLP (×12) | 3032 | **676** | 4.49× |
| ch4 CNN (×10) | 17659 | **4103** | 4.30× |
| ch5 cifar8-bn (×40×6) | 8782 | **3698** | 2.37× |
| dense anchor (synth) | 2819 | **610** | 4.62× |
| conv anchor (synth) | 8183 | **3650** | 2.24× |

Imagenette rows come from §0b's 80-epoch XLA runs on the current certified bytes (R34 1h11m, mnv2
1h25m, ENet 1h34m, ConvNeXt 1h56m). **The spread between 2.2× and 4.6× is the whole argument for a
per-lowerer column** — a single blended factor would be wrong in both directions.

Free sanity check while measuring: MNIST-MLP epoch-1 val is **92.43% on both backends**, CNN 97.77
vs 97.71, cifar8-bn 41.52 vs 40.08.

#### ⚠ Two corrections this produced

* **The conv probe has ±6% run-to-run spread, and a single sample is not an anchor.** Ten runs of
  the same binary: 3449 / 3473 / 3482 / 3528 / 3565 / 3733 / 3774 / 3778 / 3792 / 3865 ms/epoch.
  Anchoring on one sample made the on-reference factor read **0.92×**, which looks like a
  regression and is not. Both XLA anchors are now medians (dense is stable to ±1.5%: 601-619).
  *A wrong inference worth recording because the check was cheap:* the first six samples split
  cleanly by context — ~3465 inside `lake run benchmark-xla`, ~3780 standalone — and a thermal
  story was written for it. The 11th sample (3865, in-benchmark) **refuted it**; it is plain noise.
  The footer now says 0.94-1.06× is agreement, not signal.
* **`lake run benchmark`'s ch4 (MNIST CNN) row is ~1.35× pessimistic.** It reads 23764 ms/epoch;
  re-measured on the same card, same basis, it is **17659**. ch2/ch3/ch5 reproduce (535→539,
  3200→3032, 8490→8782) and all three IREE probe anchors reproduce (0.92× / 1.02× / 1.01×), so this
  is one stale row, not a drifted table. **Left as-is, not silently changed** — it is a
  user-facing published estimate and the correction is Brett's call. Both caveats are recorded in
  the lakefile header comment, along with the fact that the IREE Imagenette rows predate the
  2026-07-28 codegen swaps and so were measured on **retired hand-written renders**, while the XLA
  ones are the current certified bytes.

#### ▶ ViT on XLA — it runs; the workaround was a false lead; DP is 5 of 5

Found while confirming this section's own "attn probe unavailable on XLA" assumption, which was true
when written and is now stale.

| | result |
|---|---|
| runs at all, **no workaround** | ✅ rc=0 on 11 consecutive runs |
| step time | **128 ms/step** median of 8 (123/125/126/127/128/129/132/137) vs IREE **1188** ⇒ **9.2×** |
| marginal epoch, `(T₃−T₁)/2` | **43.5 s** ⇒ 80 ep ≈ **0.97 h**, against IREE's 7.8 h |
| cross-backend numerics, fresh init both sides | losses **3.015268 / 3.005852 / 2.987083** (XLA) vs **3.015261 / 3.005857 / 2.987081** (IREE) — **3e-6**, descending |
| descent | 39.7 → 46.7 → **49.6%** over 3 real epochs |
| **`vit-dp-check`** | ✅ **BIT-EXACT 16,579,041/16,579,041** (θ, m, v, loss/bc), control **0.996**, rc=1 ⇒ **DP is 5 of 5** |

⚠ The IREE side needed its epoch-1 `vit_adam_ckpt.bin` moved aside to start fresh (§4's checkpoint
trap — restored afterwards). Without that the loss comparison is meaningless.

**`vit-dp-check` needed a code change before its green could be believed.** It hardcoded both
artifact paths and took no argv, so its bit-exact PASS was **unfalsifiable** — §4's "a tie that is
bit-exact everywhere is indistinguishable from a harness comparing a buffer with itself", in its
purest form. Added `argv[1]` (the mnv2/ConvNeXt shape) and built the sum-not-mean control by
flipping all 200 divisors; it fires at **0.996** with rc=1. It also re-demonstrates §3's θ rule on a
**sixth** net: θ moved **1.99e-4** while `m` moved 0.996 — and note that lands *just above* a 1e-4 θ
gate where ConvNeXt's 9.7e-5 landed *under* one, so the rule holds but the margin is net-dependent.
`loss/bc` stayed bit-exact through the control, correctly localising the fault to the backward.

**⚠ THE WORKAROUND WAS A FALSE LEAD — the honest account.** The documented im2col failure fired on
the session's **first** ViT/XLA execution. `MIOPEN_DEBUG_CONV_GEMM=0` then made it run, and that was
recorded — and wired into `benchmark-xla` and `imagenette-xla` — as the fix. It is not:

* the graph runs **without** the variable, 11 runs including the byte-identical invocation that had
  just failed;
* setting it **costs ~7%**: attn probe 136 vs 128 ms/step median, marginal epoch 46.5 s vs 43.5 s;
* the MIOpen on-disk cache shows **no writes** in that window, so "the cache got populated" does not
  explain the change either. **Mechanism unidentified; the failure is non-deterministic.**

The wiring was **backed out** (`BenchRef.attnEnv`, `runProbe`'s `extraEnv`, and `runDemoGroup`'s
per-target env are all gone), and every constant measured under the variable was **re-measured
without it**. Keep it as a documented escape hatch in `probeAttnRefMsXla`, not a default.

*Both of this session's wrong turns were the same error* — inferring a cause from a single
before/after pair. The other was the conv probe's retracted "thermal/context" story. Neither
survived a fourth sample. **One sample is not a measurement**, and that applies to negative results
and to explanations, not just to headline numbers.

**What is still owed on ViT:** no 80-epoch run and no accuracy number; the other four Imagenette
nets have both. ch.10's XLA reference is therefore a marginal-epoch extrapolation (43.5 s × 80),
flagged as such in `benchTable`.

**The upstream doc's diagnosis was already right and already confirmed** on 2026-07-28 (fused
interior-dilated pad+conv → the no-workspace `GemmFwdRest` path → `MIOpenIm2d2Col.cpp` fails under
HIPRTC on an OpenCL builtin), with a 20-line JAX reproducer. So **this handoff's own "leading
unconfirmed hypothesis" wording was stale and is retired.** Its line 312 ❌ for the env var
(*"MIOpen hung … Killed it"*) is stale too — it completes in 39 s here — but since the variable is
not needed, that is a footnote worth one line in that doc, not a rewrite.

#### Not touched, deliberately

**README.** It documents no `-xla` target at all — not the three demo groups from `cf07259`/
`eb3632c`, not this. The whole XLA track is absent from it because the branch is unmerged, so
adding one command in isolation would be inconsistent; that is a merge-time decision.

---

*The original scoping follows.*

`lake run benchmark` runs three **synthetic-input** probes, measures
ms/step, and scales a per-chapter reference table to estimate full Part-1 training time on the
user's GPU. The estimator is one line (`lakefile.lean`, `yourSecOf`):

> `yourSec = refSec × yourMs / refMs`

#### ⚠ The trap: `refMs` and `refSec` are IREE numbers, and nothing says so at the call site

`probeDenseRefMs := 3030`, `probeConvRefMs := 8020`, `probeAttnRefMs := 1173` and every `refSec` in
`benchTable` were measured **on IREE, on a 7900 XTX**. Run the probes on XLA and divide by those,
and the ratio conflates *your GPU vs a 7900 XTX* with *XLA vs IREE* — and the second factor is
**4.6× on EfficientNet** (§2e-quinquies). So the naive `benchmark-xla` would tell a user with an
identical card that Part-1 trains ~4.6× faster than it does, with no warning. That is this thread's
recurring failure mode exactly: a plausible number from a silently mismatched baseline, the same
shape as the 1.23× and 1.43× DP ratios and the wall-clock-minus-compile epoch.

**So `benchmark-xla` needs its own reference column, measured on XLA on the reference card.** Much
of it already exists — §0b's 80-epoch table is XLA on a 7900 XTX (R34 1h11m, EfficientNet 1h34m,
MobileNetV2 1h25m, ConvNeXt 1h56m) — so the imagenette rows are largely measured; what is missing is
the three probe constants and the MNIST/CIFAR chapter rows.

#### Probe coverage on the XLA side, measured

| probe | IREE driver | XLA peer | state |
|---|---|---|---|
| **dense** | `mnist-mlp-verified` | `mnist-mlp-verified-xla` | ✅ exists |
| **conv** | `cifar8-bn-verified` | — | ⛔ **no peer.** That driver is plain-SGD via `.train`, NOT `trainAdamSched`; every cifar `-xla` target goes through the latter. Either add the peer (check `.train` drives the XLA shim first) or move the probe to `cifar8-bn-verified-adam-xla` and re-measure its constant — do NOT reuse 8020 for a different optimizer |
| **attn** | `vit-verified-adam` | builds, **will not run** | ⛔ MIOpen (§0b). The script already degrades to the conv proxy at `aMs == 0` and prints `*proxy`, so this is graceful — but it means benchmark-xla is structurally **2 of 3 probes** on this box |

#### What it needs, in order

1. **The reference constants, on XLA, on the 7900 XTX** — three synthetic probes, minutes each.
   Without these the whole output is wrong, so do them first, not last.
2. **A conv peer or a re-pointed conv probe** (above).
3. **The script itself** — reuse `ensurePjrtShim` / `notePjrtPlugin` from `runDemoGroup (xla := true)`;
   it does NOT need the venv on PATH, since the XLA binaries compile in-process rather than shelling
   out to `iree-compile`.
4. **Label the output.** Print which lowerer produced the estimate and which reference it scaled, in
   the table header. The current one prints `ref = single AMD 7900 XTX` and does not say IREE.

Worth doing at the same time: `lake run benchmark`'s own header should gain the word IREE, because
today the two commands would print indistinguishable tables from different baselines.

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
1b. ~~**The `-xla` trainers for mnv2 / ConvNeXt / ViT — §2h.**~~ ✅ **DONE 2026-07-29.** mnv2 and
   ConvNeXt both have an XLA peer, agreeing with IREE to fp noise and descending on their certified
   bytes; ViT is plumbed and its MIOpen blocker was reproduced on the new binary. The estimate held
   exactly — 3 small files + 2 lakefile entries per net, no driver change.
1c. ~~**The DP artifacts the `-xla` binaries unlocked.**~~ ✅ **DONE 2026-07-29 — §2h-bis (mnv2, one
   `#eval`) and §2h-quater (ConvNeXt, `replicas` built from scratch).** Both gated by the exact
   duplicated-batch identity on the real net, both verified to fail, both measured (1.67× / 1.68×).
   **This axis is closed except for ViT**, which cannot execute here at all. What it produced is the
   fourth independent measurement pointing at item 3 below.
1d. **`lake run benchmark-xla` — §2j.** ▶ the next thread. Small, but it has a silent-wrong-number
   trap in it (the reference constants are IREE's), so read §2j before writing any of it.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified end to end.
3. **Device-resident parameters.** Two rounds of transfer work are already done (batching:
   256→205 ms; killing the per-step host memcpys: 205→162 ms). What remains is smaller than it
   looks — see §3. **FOUR independent multi-GPU measurements now point here** (§2c 1.46× on R34,
   §2e-ter's measured 13–16% per-step DP overhead on EfficientNet, and mnv2's **1.67×** and
   ConvNeXt's **1.68×** end-to-end — three architecturally unlike nets landing on the same ratio,
   which is what makes the ceiling structural rather than net-specific), so it is the
   highest-value structural item left. (On EfficientNet the data loader is
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

### 2d.2. The batch / step-count study, and the FIRST measurement of the BN-split cost

Run 2026-07-29. Five R34 configurations, **same certified renderer, same 80 epochs, same
cosine+warmup schedule, unscaled LR**, so the only moving parts are global batch, how the batch is
assembled, and the step count that follows:

| config | variant | steps/epoch | final (ep 80) | best | marginal epoch | wall |
|---|---|---|---|---|---|---|
| 1 GPU, bs32 (global 32) | `adam` | 295 | **90.39%** | 90.39% | — | 71m27 |
| **1 GPU, bs64** (global 64) | `adam64` | 147 | **89.40%** | 89.58% | 37.5 s | 50m20 |
| 2 GPU, bs32×2 (global 64) | `adamdp` | 147 | **89.22%** | 89.40% | 33.5 s | 45m15 |
| 2 GPU, bs128×2 (global 256) | `adamdp128` | 36 | **86.98%** | 87.11% | 21.0 s | 28m27 |
| 1 GPU, bs128 (global 128) | `adam128` | 73 | **88.64%** | 88.71% | 34.0 s | 45m20 |

**Accuracy tracks the STEP COUNT, and is indifferent to how the batch was assembled.** Monotone
across all five, at a cost of roughly one point per halving — 295→147→73→36 gives −0.99, −0.76,
−1.66, accelerating at the bottom as 2,880 total updates starts being genuinely too few:

| steps/epoch | 295 | 147 | 73 | 36 |
|---|---|---|---|---|
| total updates (×80 ep) | 23,600 | 11,760 | 5,840 | 2,880 |
| final val | 90.39% | 89.40% | 88.64% | 86.98% |

That is the large-batch recipe cost at unscaled LR (§2c), not a defect — **final train loss is
0.5016 / 0.5016 / 0.5021 / 0.5019 / 0.5025**, a spread of 9e-4 across a 3.4-point accuracy range.
Every config fits the training set identically; the whole difference is generalisation from fewer
updates. The fix, if the wall clock is wanted, is LR scaling, not a graph change.

#### ▶ The controlled pair: **splitting BatchNorm across 2 replicas costs nothing measurable**

`adam64` and `adamdp` are the same global batch (64) at the same step count (147). The **only**
difference is where BN gets its statistics:

| | BN statistics over | final | best | train loss |
|---|---|---|---|---|
| bs64 × 1 GPU | **64 examples** | 89.40% | 89.58% | 0.5016 |
| bs32 × 2 GPU | **32 per replica** | 89.22% | 89.40% | 0.5021 |

**0.18 points**, with train losses 5e-4 apart — inside §3's documented epoch-scale nondeterminism
(three same-seed runs spanning 43.21–47.29% at epoch 1). So the training consequence of the §10.3b
split is below noise here.

**Be careful what this licenses.** It does **not** weaken §10.3b: `2×32` still is not `1×64` as a
*function*, which is why R34's collective cannot be gated by splitting and needs the cifar8 proxy
(§2b-quater). It says the *accuracy* consequence is unmeasurable **at 32 per replica**. Per-replica
batch is what shrinks as replicas are added at fixed global batch — at 8 replicas you would be at
bs8 per replica, where BN statistics genuinely degrade. This measurement says 32 is fine and gives
**no** comfort about 8.

*(This run is also what the `evalBs` change bought: `adam64`/`adam128` train at a batch the eval
forward was not rendered at, and before that they could only run under `LEAN_MLIR_SKIP_EVAL=1`,
i.e. with no accuracy number at all.)*

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
- **`lake env lean tests/X.lean` does NOT rebuild an edited IMPORT.** It links the committed
  `.olean`s of everything `X` imports, so editing `StableHLO.lean` and re-running the test silently
  exercises the OLD emitter. Found while running a negative control on the new `bnEval` emit tie:
  the perturbation was in the file, the guard stayed green, and the guard was right — it had never
  seen the change. `lake build LeanMlir.Proofs.Codegen.StableHLO` first, then run the test. This is
  the `lake env lean` cousin of the exe-cache trap, and it makes "I broke it and the guard stayed
  green" worthless as evidence unless you rebuilt.
- **A `#eval`-based test must fail via `throw`, not `IO.Process.exit 1`.** Under `#eval` the
  elaborator buffers the eval's output and prints it only after the eval returns, so `exit` discards
  **every** diagnostic — you get a bare non-zero status and no idea what broke. Flushing does not
  help. `tests/TestAdamOpTie.lean` and `tests/TestCifar8AdamTie.lean` still use the `exit` form and
  fail blind.
- **⚠ A BN reduce-dim census is only valid WITHIN ONE TENSOR LAYOUT.** §2f's table (`reduce[2,3]` =
  per-example, `reduce[0,2,3]` = batch) is the standard way to settle which BN world a render is in,
  and on 2026-07-29 it produced a **false alarm**: the `Proofs/` CNN renders use 4-D `[B,C,H,W]`
  where per-example BN is `reduce [2,3]`, while the hand-written cifar8/cifar8w emitters use 3-D
  `[B,C,H*W]` where the IDENTICAL semantics is `reduce [2]`. Comparing the literals across the two
  "found" a two-worlds skew in `cifar8_bn` that does not exist. **Ask the layout-independent
  question instead: does the reduce that FEEDS the `rsqrt` contract axis 0?** (Equivalently: does
  the broadcast after it keep the batch axis — `dims = [0,1]` does.) And note `reduce [0,2,3]` in a
  train step is usually a conv-bias / BN-β **param gradient** legitimately contracting the batch
  (§2b's "second kind"), not a BN statistic — check what it feeds before counting it.
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
- **A checkpoint OUTLIVES the artifact it was trained on, and nothing notices.** Found 2026-07-29
  while running §2h's gates: `mobilenetv2_adam_ckpt.bin` (epoch **80**), `convnext_adam_ckpt.bin`
  and `vit_adam_ckpt.bin` were all still sitting in `.lake/build/` from mid-**June** — i.e. from the
  *retired hand-written* renders, before the 2026-07-28 swaps. The checkpoint is keyed by slug +
  variant + backend and carries no fingerprint of the graph, so a fresh run on the new certified
  bytes silently resumes June's optimizer state; and mnv2's `epoch=80` would have made the run a
  **silent no-op** (§4's next bullet). Neither is detectable from the output. **Delete or rename
  `.lake/build/<slug>_<variant>_ckpt{,_xla}.bin{,.epoch}` after any artifact swap** — and note the
  IREE and XLA peers have *separate* checkpoints, so a resumed IREE run against a fresh XLA one
  quietly compares two different programs, which is exactly what a cross-backend gate must not do.
- **✅ SOLVED 2026-07-30 — the `.epoch` marker mystery was `| head`, and the marker was never
  wrong.** The symptom: after a run "killed mid-epoch" by a `| head` in the invocation,
  `resnet34_adamdp_ckpt_xla.bin.epoch` held **`80`** = `cfg.epochs`, so the next run resumed past the
  end and did nothing. Recorded here for a year as *"cause unknown, not reproduced"*. Reproduced
  deliberately on `cifar8-bn-verified-adam-xla`: pipe it into `head -1`, and **the trainer is still
  running 45 s later** — `pgrep` finds it alive long after `head` exited, and it goes on to complete
  all 40 epochs and write `40` **correctly**. **SIGPIPE does not kill these trainers.** So there is
  no marker bug and nothing to fix in the writer; what there is, is a trap:
  *piping a training run into `head`/`grep -m1` does not stop it* — it detaches it, and it keeps a
  GPU busy and finishes invisibly. Consequences: a later run silently no-ops on the completed
  checkpoint (the original symptom); two "sequential" runs can overlap and contend for the GPU; and a
  wall-clock measurement taken this way is meaningless. **Redirect to a file and read the file**, and
  still `cat` the marker or delete `.lake/build/<slug>_<variant>_ckpt{,_xla}.bin{,.epoch}` before
  starting something long.
  (Also why `pgrep -f <binary>` reads 2 when nothing is running: it matches the invoking shell's own
  command line. Check `ps -eo comm` instead — the pkill self-kill trap, one level over.)
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

The five DP renders are **not** equally evidenced, and the difference is worth stating whenever any
of them is described: **EfficientNet** (§2e-bis), **MobileNetV2** (§2h-bis) and **ConvNeXt**
(§2h-quater) are gated by an exact known-answer check on the real net, run on two GPUs, each with a
2-GPU descent run and a measured scaling ratio behind it; **ResNet-34** (§2b-quater) by that same
check on a cifar8 proxy plus a 2-GPU descent run; **ViT** (§2j's tail, 2026-07-30) by that same exact
check on the real net, **bit-exact on all 16,579,041 floats** against a control that fires at 0.996 —
but with **no 2-GPU descent run and no measured scaling ratio**, so it is the least-evidenced of the
five even though its gate is the tightest. All five now execute; four of the five are gated on the
net they actually run.

ConvNeXt's is the one whose identity needs **no BatchNorm argument**: LayerNorm reduces within an
example, so the replicas are uncoupled by construction and the duplicated-batch identity is exact
without the §10.3b reasoning the BN nets need. Its flip side is that it returns no batch statistics,
so `%loss` — report-only, outside the AST — is the *whole* of its forward evidence, which is why
that gate is load-bearing there and merely confirmatory on mnv2/EfficientNet.

#### ✅ CLOSED for ConvNeXt 2026-07-29 — `convnext-shard-check`, the asymmetric-batch known answer

The gap below is real and the diagnosis stands; what follows is the gate that closes it, and the
construction **transfers unchanged to `efficientnet-` and `mobilenetv2-dp-check`** (unlike the
cifar8 split identity, it works on batch-BN nets too — each replica normalises over its own `b`
rows either way, and the single-device reference runs are at that same `b`).

`tests/TestConvNeXtShardCheck.lean` → `lake build convnext-shard-check`. Give the replicas
**different** data and check against two single-device steps:

> `DP( [xA | xB] )` must equal `mean( single(xA), single(xB) )`

| | result |
|---|---|
| **TEST** — `\|DP − mean(A,B)\|` / max\|mean\| | **8.2e-8** ✅ (~1200× inside the 1e-4 gate) |
| **CONTROL** — `\|DP − A\|` / max\|mean\| — where a broken shard lands | **0.137** |
| separation | **1.7 × 10⁶** |

**Two design points worth keeping.**

*It gates `m`, not `θ'`, and that is forced.* AdamW's parameter update is NONLINEAR in the gradient
(`m̂/(√v̂+ε)`), so `(θ'_A + θ'_B)/2 ≠ θ'(ḡ)` and a θ' comparison would be meaningless. Feeding
**`m = 0`** makes `adamMNextF` give `m' = (1−β₁)·g = 0.1·g` — exactly linear in the gradient, hence
exactly averagable. `v' = 0.001·g²` is quadratic and is reported but NOT gated. Same conclusion as
§3's "gate the gradient, never θ", reached from the other direction.

*The control is inside the same run.* `|DP − A|` is what a shard-offset bug returns, computed every
time, and the harness **refuses as VACUOUS** if it is below 1e-3 — i.e. if the two shards were not
actually distinguishable. So this gate cannot go vacuously green, which is the failure §2d.1 hit
with a reversed-batch control that produced no difference at all.

#### ⚠ The gap it closes: a duplicated-batch gate does NOT test that sharding SHARDS

Found 2026-07-29 while tracing the shard path. `efficientnet-`, `mobilenetv2-` and `convnext-dp-check`
all hand both replicas the **same** rows (`x2 = concat #[x1, x1]`). That is what makes
`all_reduce/2` an exact identity — and it is also why those gates are **blind to a shard-offset
bug**: if the shim gave replica 1 rows `[0,b)` instead of `[b,2b)`, the two halves are identical and
every one of those gates still passes, bit-exact.

What they establish is *"the collective averages correctly"*, not *"the replicas saw different
data"*. Do not quote them as the latter.

**`cifar8-dp-check` is the one that tests sharding**, via the split identity `1×256` vs `2×128`, and
its own `%loss` row is the evidence: the DP loss is replica 0's half-mean and *differs* from the
single-device one by 1.6e-3, deliberately ungated — *"if both replicas saw the same data it would
match exactly"*. A difference there is the proof the shard is real.

**How worried to be: not very, and for a structural reason.** The split is `n_replicas`-generic C in
`pjrt_ffi.c:555-566` (`src + rep * (elems / n_replicas)`, plus the divisibility refusal), shared by
every net — cifar8 exercises the same lines. So the risk is "tested once, in shared code" rather
than "untested". Still worth closing per net, because the *shard mask* is built per call site
(`iree_lean_ffi.c:1022/1040` — `x` and onehot sharded, params replicated) and that is the part that
could be wrong for one net's interface without cifar8 noticing.

**Which nets could take the strong gate.** The split identity needs no BN, so it is available
exactly to the LayerNorm/no-norm nets — **ConvNeXt and ViT** — and needs a `2b` render to compare
against. ConvNeXt's is blocked on `cBS` being a *private constant* rather than a renderer parameter
(unlike R34/mnv2/enet, which already take `B`); making it a parameter is the whole prerequisite.
R34/mnv2/EfficientNet cannot take it at all — batch BN means `2×b ≠ 1×2b` by design.

*(That paragraph is why the **asymmetric-batch** construction above is the better answer: it needs
no `2b` render, needs no absence of BN, and closes the same hole. The split identity remains
strictly stronger where it applies, but nothing now depends on it.)*

**Still open:** port `convnext-shard-check` to `efficientnet-` and `mobilenetv2-dp-check`. Same
construction, ~20 lines each, and R34's DP would gain its first real-net sharding evidence
(it currently has only the cifar8 proxy).

And on the renders: `pretty(provenGraph)` means the committed bytes are the certified render *of
the graph that was proven* — it does not mean the emitter is verified. The `Tok → StableHLO-text`
lexing stays audited-but-trusted, which is why every move in §2a and §2b is backed by a numeric tie
against what it replaced, not by the faithfulness theorem alone.

**ConvNeXt's two weight-gradient gaps are CLOSED (2026-07-28)** — the stem 4×4/s4 patchify (`psW`)
and the even-kernel 2×2/s2 downsample (`d1W`/`d2W`/`d3W`). Its render is now `pretty(AST)` for
**all 180 params**, licensed by a bit-exact `convnext-adam-tie` against the previously committed
hand-written render (83,434,629 floats, spread 0/180). The old caveat — "certified except for two
documented weight-gradient gaps" — is retired.

**They were different kinds of gap, and an earlier version of this paragraph said they were the
same** ("neither of which has a VJP-cert `SHlo` op"), which overstated the second by a lot. `psW`
needed a genuinely missing cert (`flatConvStride4_weight_grad_has_vjp`, now built and 3-axiom
clean); the downsample's cert and `SHlo` op both already existed and were kernel-generic, and only
the emitter's symmetric-SAME-padding formula could not spell an even kernel. §2f-bis has the
measurement and what was built.

Four places currently emit text that is **not** `pretty` of an AST node, and all say so in the
emitted output: `cifar8_adam_train_step`'s report-only scalar `%loss` and its `%bc` passthroughs,
and the `resnet34`/`vit`/`efficientnet` AdamW renders' `%loss`. **The R34 one shipped wrong** — plain CE instead of
the smoothed CE its own cotangent implies — and only the numeric tie found it, because nothing on a
gradient path touches it and no theorem covers it. Treat every such carve-out as unverified text
that needs its own numeric check, not as a harmless annotation.

On the forward artifacts (§2g): `pretty(provenGraph)` and the prefix audit together say the graph
the driver *evals* with is the graph the train step *differentiates* — which is a structural claim,
and the one that was false for `resnet34_fwd` and `mobilenetv2_fwd`. The numeric ties behind the
five swaps compare **logits only** (320 floats), because a forward render returns nothing else; they
are not the AdamW ties' whole-net fingerprint. Say "the same logits", not "the same intermediates".

A note on what the §2b tie does and does not establish. It says the batched render computes what the
hand-written render computes — forward to the bit, backward to norm-relative 1e-6. It does **not**
say either one is the mathematically intended net; that comes from the `den` side (the faithfulness
theorems, now honest at `N := B`) and from the layer-level VJP oracle in §3. The two halves are
independent, and both are needed: the artifact cannot witness a wrong `den` (the render is
value-independent), and the theorems cannot witness a wrong emitter.
