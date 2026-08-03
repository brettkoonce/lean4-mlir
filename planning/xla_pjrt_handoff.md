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
| — | *↓ the v1.3/v1.4 recipe threads, 2026-08-02 (§0.4)* |
| `969c21c` | **stochastic depth's two INTERIOR gates** (`droppath-tie`) — and the endpoint gates were blind to PLACEMENT |
| `bfa4ea2` | **mixup + cutmix, the producer half** (`scripts/mixup_gate.py`) — and the second definition's cost made precise |
| `bc4bf7d` | the val shim inherited `SHIM_MIX` and died — **mixing is gated on the SPLIT**; found by the end-to-end smoke, not by any of the three gates |
| `c54b408` | **`wdExcludeNormBias` on ViT** (`wdx-tie`) — and the ViT renders were baking a **500× wrong** weight decay |
| `ed79c99` | **`wdExcludeNormBias` on ConvNeXt** — and the same edit behaved differently on the two renderers |

---

## 0. ▶ START HERE — **next: the BATCHED-INDEX MOVE. §0.2 has the order.**

**Rewritten 2026-08-02 for a fresh session; updated the same day when grad clip landed.** State:
`lake build Proofs Certs Codegen` **3,913** green · **115** artifacts, one writer each ·
`verified_mlir/` clean · drift-guard coverage **72/103** with `render_guard_baseline.txt` unchanged
(that file may only shrink).

**Read in this order:** this section (§0), then `planning/recipe_gaps.md` — which is the plan and
supersedes every ordering further down this file. The feature specs are `planning/ema.md` and
`planning/stochastic_depth.md`. Everything from §0a onward is HISTORY and context; nothing there is
a live thread unless §0.3 says it is owed.

**§0 is laid out for someone starting cold:**

| | |
|---|---|
| **§0.1** | ⚠ the box constraint — it re-orders everything |
| **§0.2** | ▶ **THE NEXT THREADS, in order** — batched index → the rest |
| **§0.3** | ⚠ what is OWED, collected in one place |
| **§0.9** | ✅ the per-net data shims — wired, gated, and the streams measured apart |
| **§0.5** | ✅ DP ImageNet recipe parity — and the DP control it broke |
| **§0.6** | ✅ stochastic depth's DP shard gate — §5b's predicted defect, found |
| **§0.7** | ✅ the ViT / EfficientNet EMA DP peers, gated — and the shadow re-measured |
| **§0.8** | ✅ grad clip — what it cost, and the three traps it found |
| **§0.4** | ✅ what landed 2026-08-02, with the findings worth keeping |

### §0.1 ⚠⚠ THE THING THAT CHANGES WHAT "NEXT" MEANS: **this box cannot do long runs**

Brett, 2026-08-02, stopping four concurrent 80-epoch Imagenette trainers mid-flight: *"um no long
runs this box will crash."* Sustained multi-GPU load destabilises ares. That is not a footnote — it
**re-orders this whole file**, because:

* **§0's former headline, the R34/ImageNet 30-epoch run, is ~16 h on 4 GPUs and is NOT currently
  runnable here.** Nothing about it is wrong; it is blocked on hardware, not on code. Same for
  `recipe_gaps.md` §4's ~203 h budget for all five nets at reference epochs.
* **What IS available is build-and-gate work**, and 2026-08-02 is six threads of evidence that it
  goes fine: every gate in §0.4 is a known answer, a bit-identity check or a few-step smoke, all
  single-GPU and all minutes.
* **So prefer threads whose deliverable is a certified render + a numeric gate**, and bank the runs
  for a box that can sustain them. State any accuracy number as what it is — a 3-4 epoch smoke is
  descent evidence, never an accuracy claim.
* ⚠ **Ask before starting anything long**, and use `scripts/supervise.sh` (AER restart, thermal
  resting, stall guard) if a long run is ever sanctioned.

### §0.2 ▶ THE NEXT THREADS, IN ORDER — ▶1 (the shim wiring) is DONE; **▶2 is the live one**

**All five nets are at PJRT parity** (`<net>-verified-adam-xla` + `<net>-imagenet-verified-xla`, both
scales, all five execute). R34 and mnv2 are feature-complete against their references except bf16.
What is left is concentrated on **EfficientNet, ViT and ConvNeXt**, and it goes in this order.

---

#### ▶ 1. ✅ THE SHIM WIRING — **DONE 2026-08-02.** §0.9 is the record

Was: `spawnShim` hardcoded `generated_resnet34_imagenet_shim.py` for every net, so a verified
EfficientNet / ViT / ConvNeXt ImageNet run streamed **R34's** augmentation. Now each net names its
own (`VerifiedNet.shimScript`), there is **no fallback**, and the streams are measured apart. The
finding that outlives it: **a definition is not a call site** — see §0.9.

---

#### ▶ 2. THE BATCHED-INDEX MOVE — the real prerequisite for stochastic depth on ViT/ConvNeXt

**Measured, not recalled** — distinct AST forms per renderer:

| renderer | batched | per-example |
|---|---|---|
| EfficientNet · MobileNetV2 · ResNet-34 | **28 · 21 · 18** | 1 each |
| **ViT** | 4 | **14** |
| **ConvNeXt** | 2 | **8** |

Every net that HAS stochastic depth is in the batched world; ViT and ConvNeXt are not.

⚠⚠ **AND THE REQUIREMENT IS STRUCTURAL, not a convention.** The drop mask is per-EXAMPLE (example
`j` gets `sⱼ`). In a per-example-indexed AST a node denotes ONE example (`Vec n`) and `pretty B`
lifts it across the batch — **the node cannot see `j`**. §4's descriptor rule is exactly this: `den`
is `batchMap N (denOp op)`, one FIXED function across the batch.

⚠ **The trap is that the emit would typecheck.** `pretty B` already emits `tensor<Bxn>`, so a
`broadcast_in_dim %mask, dims = [0]` + multiply against a per-example node compiles, trains and
descends — with no faithful `den` behind it. Same shape as `swishBack`/`selectPos`, which needed
their own constructors holding the whole-batch `x`.

**Once it lands, stochastic depth is nearly free**: `dropPathB`, its cert (`layerScale_has_vjp`
verbatim), the driver, and the DP mask-shard gate (§0.6) all already exist.

##### ⚠⚠ COSTED 2026-08-03, and the "14 and 8" above UNDERSTATES it — do not plan off those numbers

The counts above are of per-example *index conventions*; the **work** is per missing batched FORM,
and it was measured by classifying every `SHlo` constructor each renderer uses (ViT **49**,
ConvNeXt **54**) against the 14 existing `BatchableOp` descriptors and the 42 existing batched
`SHlo` ctors:

| | descriptors needed (4 sites) | own ctors needed (10 sites) | total |
|---|---|---|---|
| **ConvNeXt** | ~8 | ~6 | **~14** |
| **ViT** | ~11 | ~10, **+1 new combinator** | **~22** |

⚠⚠ **DO ConvNeXt FIRST, and the reason is not its size.** ConvNeXt is a CNN, so **every one of its
conv / depthwise / dense / gap forms already exists** — `conv`, `convStrided`, `depthwise`,
`depthwiseStrided`, `dense`, `gap` as descriptors, and `convBackBatched`, `convWeightGradB`,
`depthwiseBackBatched`, `convBiasGradB`, … as ctors, all built for R34/mnv2/EfficientNet in §2b.
What ConvNeXt is missing is only its *own* layers, and **~10 of those ~14 forms are ALSO ViT's**
(`lnRow`, `lnRowBack`, `rowScale`, `rowBias`, `gelu`, `geluBack`, `transpose`, `veclnGammaGrad`,
`rowDenseBiasGrad`, `expe`/`softmaxDiv`). So the cheaper net pays down roughly half of the dearer
one, and nothing is paid twice.

⚠ **ViT has one piece that is not a copy of an existing pattern, and it is the schedule risk:
`matmulF`.** Attention's `QKᵀ` and `·V` have **both operands per-example**, where every batched
binary in the kit (`addVB`, `subB`) is pointwise-same-shape and needs no combinator at all. A
batched matmul needs a `batchMap2`-shaped combinator (`batchMap`/`batchMapAux` do not cover it), its
own VJP, and a `dot_general` emit carrying a batching dimension. Cost that separately before
starting ViT; everything else on both nets is a template copy.

⚠ **What is NOT a cost, checked rather than assumed**: descriptors need **no `StableHLOParse`
roundtrip case** (they route through the generic `.batched` tag — §4), and the row ops need no
`Nat.mul_assoc` gymnastics, because a descriptor carries its own internal `(m, n)` and the SHlo
index stays `N * (m*n)`. Those were the two things that looked like they would dominate.

##### ✅ INCREMENT 1 LANDED 2026-08-03 — the five forms BOTH nets need

`gelu` · `transpose` · `lnRow` · `rowScale` · `rowBias`, as `BatchableOp` descriptors (4 sites each,
no parse case). Gates: `tests/TestBatchedEmitTie.lean` **16 → 21 batched forms**, each emitting its
per-example peer's text **byte-for-byte**; 7 new `rfl` declarations, all **3-axiom** (audit
1509 → 1511); build **3,913** green; `verified_mlir/` **0 lines of diff** — nothing renders them
yet, which is exactly what makes this increment safe to land on its own.

⚠ **Control run and it fires**: perturbing the batched `lnRowP` emit by one reduce axis
(`dimensions = [2]` → `[1]`) turns the tie red on `lnRow` and only `lnRow`, rc=1. ⚠⚠ And the trap
§4 warns about is live here — `lake env lean` links the COMMITTED `.olean` of its imports, so the
control is worthless unless `lake build LeanMlir.Proofs.Codegen.StableHLO` runs between the edit and
the test. Both runs above did.

**Next increment**: ConvNeXt's remaining ~9 (`flatConvStride4`, `layerScaleCh` as descriptors;
`lnRowBack`, `geluBack`, `layerScaleChGammaGrad`, `veclnGammaGrad`, `convStride4WeightGrad`,
`rowDenseBiasGrad` as own ctors — all `batchMapAux`-shaped), then the renderer re-instantiation and
the whole-net tie against the committed `convnext_adam_train_step.mlir`.

---

#### ▶ 3. THE REST — exactly two items after that

* **mixup / cutmix run** — producer done (`scripts/mixup_gate.py`, 3 gates + the end-to-end smoke
  that caught the split defect). ⚠ Its λ agreement is **permanently distribution-only**: the
  reference draws `jax.random.beta(fold_in(...))`, the shim draws from numpy's `Generator`, and no
  seeding makes them equal. Never quote it as per-step agreement.
* **bf16 / bf16Conv** — the only gap left on R34 and mnv2 too. ⚠ Worth much less on the depthwise
  nets: `bf16_renderer.md` measured **bf16 depthwise conv at 0.50×, twice as SLOW**. Throughput,
  not accuracy.

⚠ **And one unlisted RENDER gap found the same day**: EfficientNet's reference sets
`dropout := 0.2` (classifier dropout, `MainEfficientNetImagenet.lean:68`) and there are **ZERO
dropout sites in any verified EfficientNet render**. The matrix has a row for stochastic depth and
none for dropout — they are different regularisers, and `StableHLO.lean:5329`'s own comment says so.

---

### §0.3 ⚠ WHAT IS OWED — collected here so it is not spread over 6,000 lines

| owed | why it matters | where |
|---|---|---|
| ~~⛔ the four ImageNet renders bake `wd = 1e-4`~~ ✅ **CLOSED 2026-08-02** | all four bake **0.05** now; the re-render diff was exactly 4 lines, all `%wd`, every other artifact byte-identical, and both pairs re-gate bit-exact at 4 replicas | §0.5 |
| ~~⛔ stochastic depth's **asymmetric-batch DP gate**~~ ✅ **CLOSED 2026-08-02** | `lake build drop-shard-check` — and §5b's prediction was right: the masks WERE being replicated, in the shim, before any DP drop render existed. Both existing constructions were unusable and the answer was neither of them | §0.6 |
| ~~⛔ the **DP clip artifact + its numeric gate**~~ ✅ **CLOSED 2026-08-02** | `vitin_adamdp128x4wxclip` / `cnxin_adamdpwxclip` — the shipping recipe at 4 replicas, **bit-exact on 17,152,251 / 85,762,779 floats** | §0.5 |
| ⚠⚠ **every DP render's sum-not-mean control is BLIND once a clip is on** | a NEW hole, found by running it: grad clip is scale-invariant where it saturates, so the standard control passes bit-exact on a deliberately broken collective. The composed control (`perturb_clip.py hi` + sum-not-mean) is documented in `TestViTDpCheck.lean`. ⚠ **Any future clipped render must use it** | §0.5 |
| ~~⚠ the ViT / EfficientNet EMA DP peers are RENDERED BUT UNGATED~~ ✅ **CLOSED 2026-08-02** | both gated at **4 replicas, every region BIT-EXACT** — ViT 22,869,669 floats, EfficientNet 21,196,213 (incl. the 4th region and 49 BN layers), sum-not-mean controls at **2.96 / 2.39**. The EMA scorecard is 3 of 3 on DP as well as single-device | §0.7 |
| ~~⛔ **the per-net data SHIMS are not wired**~~ ✅ **CLOSED 2026-08-02** | `VerifiedNet.shimScript`, no fallback, `scripts/gen_shims.sh` + `scripts/shim_wiring_gate.py`. Measured: ViT / ConvNeXt / EfficientNet each stream a **different** train digest from R34's now, mnv2 ≡ R34 to the bit (predicted — its config is R34's), and all five validation streams stay identical | §0.9 |
| ⚠ **ViT/ConvNeXt at wire v1 now run WITHOUT their reference's mixup/cutmix, announced** | their shims bake `SHIM_MIX=both`; a mixed target cannot ride int32 labels, so the driver passes `off` and prints that it did. `SHIM_SOFT=1` gets soft targets **and** that net's own mixing — which is the reference recipe and has never had a long run | §0.9 |
| ⚠ **EfficientNet's classifier dropout 0.2 is missing and UNLISTED** | `MainEfficientNetImagenet.lean:68` sets it; there are **zero dropout sites in any verified enet render**. The matrix has a stochastic-depth row and no dropout row — different regularisers | §0.2 ▶3 |
| ⚠ mixup/cutmix has **no long run**, and its λ stream is numpy's, not `jax.random`'s | a paired run agrees **in distribution, not per step**. Never quote it as the augmentation pipeline's byte-identity | §2b |
| ⚠ mnv2's **80-epoch re-run** after the conv-bias swap | 86.73% was measured on the 210-param net | §2m |
| ⛔ **R34/ImageNet, 30 epochs** | ~16 h on 4 GPUs. Blocked on hardware, not code; the preflight is green and the rig smoke-tested | §0.4's R34 block |

---

### §0.9 ✅ THE PER-NET DATA SHIMS — WIRED AND GATED (2026-08-02), and the streams measured apart

**The defect**: `spawnShim` hardcoded `generated_resnet34_imagenet_shim.py`, `$SHIM_SCRIPT` was set
nowhere, and R34's was the only generated shim on disk — so *every* verified ImageNet trainer
streamed RandomResizedCrop + hflip, whatever its reference asked for. Nothing failed; the recipe
matrix read ✅ on a **capability** (`generateShim` honours every flag) rather than on the **state**.

**What landed** — no render change, no new op, no proof:

* **`VerifiedNet.shimScript`** (threaded through `VerifiedNetSpec.toNet`), set on all five nets.
  ⚠ **No fallback and no default**: an empty value REFUSES at spawn, because the failure it
  replaces was silent. `spawnShim` also **prints the script it resolved** — the old banner named
  only the shape, so a run streaming the wrong augmentation read exactly like a right one.
* **`scripts/gen_shims.sh`** — the ONE writer of the five files (the double-writer rule applied to
  the data path). It verifies each exe wrote the file the wiring names, rather than assuming the
  recipe's `out` still matches its exe name.
* **`scripts/shim_wiring_gate.py`** — four gates, three controls. Static run is instant;
  `--stream` measures the actual bytes; `--break` runs the negatives.

| gate | result |
|---|---|
| ⓪ the wiring is a **bijection** | five `.imagenet` nets, five **distinct** shims, all present |
| ① each shim is the right net's | `shimScript` == that Main file's `default` recipe `out`, derived not restated; banner names the reference |
| ② the augmentation **partition** | config flags vs generated **call sites**, 5 features × 5 nets, all match |
| ③ the baked `SHIM_MIX` default | == the config's `useMixup`/`useCutmix` on all five |
| ⚠ control A — the pre-fix wiring | REJECTED, and on **exactly the 3** affected nets |
| ⚠ control B — swap ViT↔ConvNeXt | five distinct shims still, **partition fires on 2** |
| ⚠ control C1 — count definitions | **mis-classifies 2 of 5** (see finding 1) |

**Measured** (`--stream`, `SHIM_HASH`, 2 batches × 8, seed 7):

| | train digest | |
|---|---|---|
| resnet34in | `f6dca723f5e9e535` | — |
| **mnv2in** | `f6dca723f5e9e535` | ⚠ **known answer: ≡ R34 to the bit** |
| **vitin** | `98712d1e21443405` | ≠ R34 |
| **enetin** | `0e3a0c4c84a10707` | ≠ R34 |
| **cnxin** | `fa85309026d0ee36` | ≠ R34 |
| all five, **validation** | `0431fa5caacb8c74` | identical — eval is untouched |

mnv2 is the **inertness** half and it is a prediction, not a coincidence: its config sets
`useAutoAugment := false` and nothing else, so its generated shim differs from R34's in **exactly
one line** — the banner. The other three are the **discrimination** half: they did not differ from
R34 yesterday. Determinism re-measured per net (same seed twice ⇒ same digest).

**End to end**: ViT/ImageNet, 1 GPU, 3 steps — both spawns name
`generated_vit_tiny_imagenet_shim.py`, loss 8.0399 → 7.8709 → 7.6713, `rc=0`. ⚠ Three steps is
*"it runs and the loss moves"*, nothing more.

**Three findings that outlive the feature:**

1. ⚠⚠ **A DEFINITION IS NOT A CALL SITE, and a census cannot tell them apart.** `generateShim`
   emits the shared `_aa_*` op block whenever AutoAugment **or** geometric RandAugment is on, so
   **ViT and ConvNeXt contain `def _autoaugment` and never call it**. A gate grepping
   `_autoaugment` reports AutoAugment on three nets when it is on one. Gate ② reads `img =
   _autoaugment(` instead, and control C1 measures the census being wrong on 2 of 5 rather than
   arguing it. This is `wdx-tie`'s *"gate the partition, not the count"* one layer down — the
   count is right and the meaning is wrong.
2. ⚠⚠ **WIRING THE RIGHT ARTIFACT CAN BREAK THE DEFAULT PATH, and the gate found it before any
   trainer ran.** ViT's and ConvNeXt's shims bake `SHIM_MIX=both` (their references mix), and a
   mixed target is a distribution that **wire v1's int32 labels cannot carry** — so the shim exits
   before its preamble. Through the driver that surfaces as `shim closed the pipe after 0 of 16
   bytes`, since the child's stderr is not captured: the §0.4 mixup split defect's symptom exactly,
   one layer up. The driver now passes `SHIM_MIX=off` at v1 **and prints that it did and what it
   dropped** — silence there would be this thread's own defect recommitted. It is inert on the
   other three (their default is already `off`; R34's digest is `f6dca723f5e9e535` with the
   variable unset *and* set to `off`).
3. ⚠ **"Free via the shim" was a claim about the EMITTER, and two docstrings stated it as state.**
   `VerifiedNets.lean` told a reader that ConvNeXt's and ViT's pipeline augs *"do come across free"*
   — true of `generateShim`, false of every run. When a capability and a state have the same
   sentence, the doc will drift to the flattering one. Both are corrected and dated.

⚠ **Still owed**: the mixup/cutmix **run** (`SHIM_SOFT=1` now gets ViT/ConvNeXt their reference's
mixing with no extra flag, and its λ stream is still numpy's — §0.3), and EfficientNet's classifier
dropout 0.2, which is a RENDER gap and unaffected by any of this.

---

### §0.8 ✅ **GRAD CLIP — BUILT AND GATED 2026-08-02.** `planning/grad_clip.md` §11 is the record

**Two `SHlo` ops, no new proof machinery, no driver change** — `recipe_gaps`' Tier E was wrong by a
tier, the second time an estimate there was (RMSProp's "op family" was one op). Four artifacts
(`vit_adamclip`, `vitin_adam128wxclip`, `convnext_adamclip`, `cnxin_adamwxclip`), `lake build
clip-tie {vit,convnext}`, **all six controls firing**, 17 new declarations 3-axiom clean.

| gate | ViT (200 params) | ConvNeXt (180) |
|---|---|---|
| ⓪ clip ACTIVE | factor **0.038283**, ‖g‖ = 26.12 | **0.022005**, ‖g‖ = 45.44 |
| ① **the factor is ONE SHARED SCALAR** | **1.120 ULPs** of spread | **1.149 ULPs** |
| ② = `min(1, c/(‖g‖+ε))` | **0.105 ppm** | **0.0070 ppm** |
| ③ `%loss` bit-exact | ✅ | ✅ |
| ④ inert above the threshold | **16,579,039/16,579,039** | **83,478,847/83,478,847** |

Controls: `perparam` (per-parameter norm) fires ① at **7.6M / 30.4M ULPs** against an 8-ULP bar —
**while passing ⓪, ③ and ④**, because it is a working clip, just not a global one; `nosqrt` fires ②
at ~960,000 ppm **while passing ①**, because ‖g‖² is still one shared scalar; `epsout` fires ② at
**26.28 / 45.46 ppm** against the analytically predicted `ε/fac` of 26.1 / 45.5.

**Three findings that outlive the feature:**

1. ⚠⚠ **A NEW `SHlo` CONSTRUCTOR MUST BE PARAMETRIC IN `n`.** The first design had two ops with no
   `{n : Nat}` binder (`SHlo 1 → SHlo 1`), a shape the AST does not otherwise have, and **nine
   unrelated `simp only [… den …]` proofs elsewhere in `StableHLO.lean` died with a `whnf`
   timeout** — `den` is a ~200-case dependent match and fully-index-fixed arms make unfolding it
   markedly more expensive. **4× the heartbeat budget did not fix it**; fusing to two parametric ops
   cleared it first try, and is 20 sites instead of 40. The constraint pushed toward the better
   design. (Smaller cousin: `den` must never APPLY a recursive `den` call to an index — hence
   `Proofs.scalarOf`.)
2. ⚠⚠ **A BOOL-DERIVED VARIANT NAME CANNOT DISTINGUISH TWO RENDERS THAT DIFFER ONLY IN A BAKED
   CONSTANT.** A second render at a different threshold spelled the same `cnxAdamVariant`, so
   `convnext_adamcliphi_train_step.mlir` declared `@convnext_adamclip_train_step` — an entry
   disagreeing with its own path. ViT takes `funcName` explicitly and hid it. §0.4's
   derived-vs-explicit finding meeting §2a-quater's silent-hyperparameter one, and the fix is better
   than the bug: the below-threshold render is **generated** by `scripts/perturb_clip.py hi`, never
   committed, because an artifact baking a threshold no config sets *is* a silent-hyperparameter
   artifact.
3. ⚠⚠ **MEASURE THE FLOOR WITH AN INSTRUMENT THAT HAS THE INTERVENTION'S STRUCTURE BUT NOT ITS
   EFFECT — and the det shim is mandatory, for the THIRD recorded time.** Without
   `scripts/det_shim.sh`, ConvNeXt's ④ read **137,229 of 83,478,847 differing at a 24,149-ULP
   floor**, ①'s spread sat *below* that floor, and the number **moved between runs**. The floor was
   carried by **exactly ONE parameter of 180** — `d0W`, the even-kernel 2×2/s2 downsample weight
   gradient. Under the det shim: bit-identical, ① at 1.15 ULPs. *Nothing about the render changed.*
   ⚠ **ViT is clean either way**, which is how the trap stays hidden — a gate developed on ViT and
   ported to ConvNeXt inherits ViT's conditioning (§0.4 finding 1, now in a **fourth** place). ④ is
   run FIRST now, because it is also ①'s floor, and it refuses with the det-shim recipe.

✅ **The DP artifact and its gate landed the same day — §0.5.** `vitin_adamdp128x4wxclip` /
`cnxin_adamdpwxclip`, bit-exact at 4 replicas on 17,152,251 / 85,762,779 floats. ⚠ And running its
control turned up a **new hole in a gate every DP render uses**: the sum-not-mean control is
structurally blind on any clipped render, because grad clip is scale-invariant where it saturates.
§0.5 has the mechanism and the composed control that replaces it.

---

#### The scoping that produced it, kept for the method (written before the code)

⚠⚠ **FULLY SCOPED 2026-08-02 — `planning/grad_clip.md` is the spec and SUPERSEDES this section.**
The sketch below is right about the op shapes and wrong about the two things that matter most:
**the shared-DAG worry dissolves** (`SHlo` is single-OUTPUT, not single-INPUT — `sub`/`addV`/
`matmulF` are already binary, and every gradient is already an `.operand` leaf, so the 200-way fold
is an ordinary tree and **no carve-out is needed on the norm**), and **the "breaks four gates" list
is one gate**, not four. It also missed that the shipping variant is `wx` *composed with* clip.
Measured cost: **four small ops, no new proof machinery, no driver change** — Tier D, not Tier E.

`recipe_gaps` calls it *"the unlock for the 5e-4 LR"* on ViT, and it is the last v1.4 item. **It is
the first feature in this file whose shape the op kit genuinely does not have**, so it was scoped
the §2l/§2m way — by reading the reference and grepping the kit, not by estimating.

**The reference** (`jax/Jax/Codegen.lean:2262`, emitted verbatim into every trainer that sets it):

```python
gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, CLIP / (gn + 1e-6)), grads)
```

⚠ **A GLOBAL norm across ALL parameters**, then one scalar factor applied to every gradient. Who
uses it: **ViT 1.0**, **ConvNeXt 1.0**. EfficientNet sets it to **0.0 deliberately** — its comment
says the TF-RMSProp fix (ε-inside-sqrt + ms-init 1.0) removed the blow-up it was compensating for —
so do **not** add it there.

#### What the kit ALREADY has (measured today)

| piece | evidence |
|---|---|
| reduce a whole gradient tensor to **rank-0** | `lnBetaGrad : SHlo n → SHlo 1` emits a real `reduce … -> tensor<f32>`. The shape is certified, not new |
| **rank-0 broadcast then multiply** | emitted **200×** in `vit_adam_train_step` already — `broadcast_in_dim %lr, dims = []` inside `adamWParamF` |
| `stablehlo.sqrt` | 200× in the same artifact (AdamW's √v̂) |
| `stablehlo.minimum` | 35× in `mobilenetv2_adam` (relu6). New to ViT/ConvNeXt's renders, **not** new to the repo |
| **the insertion point** | `vitBackAll` returns `gradNames` (all 200) and `convNextBackAll` returns `gradMap`, both **before** the optimizer loop — every gradient SSA coexists exactly where the clip goes |

#### What it does NOT have — one primitive, and it is small

⚠ **There is no "scale a tensor by a RUNTIME scalar" op.** `scaleF`/`scaleB` look like they take
one (they carry `sStr : String` beside the ℝ), but they **emit `stablehlo.constant dense<{sStr}>`**
— they bake a literal. Checked in the emitter, because the constructor signature reads the other
way and that is exactly the sort of thing this file keeps paying for.

What is needed is `dropPathB`'s emit at **`dims = []`** instead of `dims = [0]`:

```mlir
%b = stablehlo.broadcast_in_dim %gn, dims = [] : (tensor<f32>) -> tensor<Bxn>
%o = stablehlo.multiply %g, %b : tensor<Bxn>
```

So the inventory is roughly: **`sumSqF : SHlo n → SHlo 1`** (`lnBetaGrad` with a multiply first) and
**`scaleByScalarF`** (the op above) — plus the scalar tail (add N scalars → sqrt → `min(1, c/(gn+ε))`),
which is a handful of rank-0 ops. Check each against an existing reading before building one:
that check has collapsed a scoped op family to **zero** four times now (§2k heavy-ball, RMSProp,
EMA ×3, dropPath's VJP).

#### ⚠⚠ THE REAL QUESTION, and it is not the op count: the norm is a SHARED DAG NODE

`SHlo` is a single-output expression **tree**. The global norm is one scalar **consumed by all 200
sites**, and produced by folding 200 subtrees together. `.operand <ssa>` expresses the sharing in
the *emit* (that is how gradients are already threaded), but **`den` of "one scalar 200 subtrees
depend on" is not a tree**, so decide what is certified and what is a DECLARED CARVE-OUT *before*
building. §2b-quater's collective is the template — emitted text, outside every faithfulness
theorem, said so in the artifact's own banner.

#### ⚠ Two ordering traps, both of which change the function

1. **UNDER DATA PARALLELISM, THE CLIP MUST GO AFTER THE `all_reduce`.** The reference clips inside
   the train step on the whole (already-combined) gradient. In our DP renders the collective sits
   between the gradient and the optimizer, so clipping *before* it makes every replica clip its own
   partial gradient — a different function that trains, descends, and no structural check sees.
   `emitGradAllReduce` is called inside `vitAdamOne`/`convnextAdamOne`, i.e. **per-param, after the
   point where the clip would be computed** — so this is a real restructuring question, not a
   line-ordering one.
2. **IT CHANGES THE GRADIENT THE OPTIMIZER SEES, so it breaks every gate that recovers `g` from
   `m'`.** `r34-mom-tie`, `rms-tie`, `wdx-tie` and `shard-check` all use `m' = (1−β₁)·g` at `m = 0`
   as an oracle. With clipping on, that recovers the *clipped* gradient. Either keep the clip off in
   those harnesses (a new variant, as `wx` is) or account for the factor explicitly.

#### The gates it should have

**Known answer**, the `r34-mom-tie`/`wdx-tie` construction: at `m = v = 0`, recover `g` per param
from the unclipped render's `m'`, compute `gn` and the factor on the host, and require the clipped
render's `m'` to be `(1−β₁)·factor·g` — **exactly**. Then the controls that matter: a gradient whose
norm is BELOW the threshold must come back **bit-identical** to the unclipped render (the factor is
exactly 1.0, so this is a bit-exactness claim, not a tolerance), and one above it must be scaled by
the predicted factor. ⚠ Condition the instrument so both regimes are reachable — `LEAN_MLIR_BASE_LR_U`
does not help here; the norm is a property of the *data*, so drive it with a scaled input.

---

### §0.7 ✅ THE ViT / EfficientNet EMA DP PEERS, GATED (2026-08-02) — and the shadow re-measured

⚠ **§0.3's row said "nothing renders either" and that was STALE**: v1.2c (`e5f0c9bc`) rendered both
`vitin_emadp128x4` and `enetin_emarmsdp64`. What was actually owed was the NUMERIC GATE — ConvNeXt's
`convnext_emadp` was gated when it landed, these two came across as a carry-forward and no check had
ever run on them. **No new renders were needed**; the single-device peers at the matching per-device
batch (`vitin_ema128`, `enetin_emarms64`) already existed.

The two `dp-check` harnesses could not feed them: an `ema*` render carries a **fourth
`[θ|m|v|ema]` region and a 5-slot scalar tail**, and only ConvNeXt's harness knew that. Both now
build the blob they feed. ⚠ The predicate is a **substring**, not `startsWith` — EfficientNet's
recipe is `emarms`, and this is the exact axis where a prefix test already failed once
(`planning/ema.md`: `emarms` does not start with `"rms"`).

| duplicated-batch identity, 4 replicas | ViT (`vitin_ema*`) | EfficientNet (`enetin_emarms*`) |
|---|---|---|
| θ / m / v / **ema** | **bit-exact 5,717,416 each** | **bit-exact 5,288,548 each** |
| `%loss` + scalars · BN stats | 5/5 · — | 5/5 · **42,016/42,016** |
| total | **22,869,669 floats** | **21,196,213** |
| ⚠ sum-not-mean CONTROL, on `m` | **2.9647**, rc=1 | **2.3915**, rc=1 |

⚠⚠ **AND THE CONTROLS RE-MEASURED `ema.md`'s RULE ON TWO MORE NETS, more sharply than before.**
*Never gate on the EMA shadow.* In the same runs that fire at 2.96 / 2.39 on the gradient:

| | gradient `m` | θ | **the shadow** | sensitivity ratio |
|---|---|---|---|---|
| ViT | 2.9647 | 3.73e-4 | **1.94e-4** | **15,000×** |
| EfficientNet | 2.3915 | 6.94e-4 | **5.42e-4** | **4,400×** |

The shadow is the most damped region of the three on **both** nets — below θ, which §3 already says
never to gate. ⚠ ViT's shadow lands at **1.94× a 1e-4 gate**, i.e. it would have *barely* fired;
ConvNeXt's earlier measurement was 1.00e-4, exactly ON it. Three nets, same conclusion: the shadow
is θ's low-pass filter, so gating it is §3's rule one step worse. It is REPORTED in both harnesses —
a mis-threaded 4th region has to be visible — and it decides nothing.

⚠ Not a training run: this gates the collective's semantics on a duplicated batch. Neither net has a
long EMA DP run, and §0.4's caveat stands — an Imagenette `ema` run is a gate vehicle, not a pair.

### §0.6 ✅ STOCHASTIC DEPTH'S DP SHARD GATE (2026-08-02) — §5b was right, and the defect was real

`stochastic_depth.md` §5b left this as *"an open design question [that] should be settled before the
render lands, not after"*. Settled, and the prediction held: **the drop masks were being
REPLICATED, not sharded** — in the shim, before any DP drop render existed to expose it.

The masks ride in the PARAMETER blob, and the DP shim's rule is *"`x` and the labels shard,
everything between replicates"*. **Two halves, both needed for it to be silent**: the shard flag was
never set, and the buffer was sized at the PER-DEVICE batch so it type-checked as a replicated
input. Fixed by `pjrt_ffi_invoke_f32_dp2` (renamed, taking `n_shard_tail` — §4's rule) plus global
sizing. ⚠ Once the buffer is global, **replication is not expressible**: the shim refuses on arity,
so the sizing fix is what turns the flag from a correctness question into a type-checked one.

**▶ The construction, and it is optimizer-agnostic — which §5b said did not exist.** Duplicate the
DATA, make only the MASK asymmetric, and SWAP the halves. A sharded mask is swap-invariant TO THE
BIT (the 2-replica collective is `(a+b)/2` and f32 addition is COMMUTATIVE); a replicated one is
not. ⚠ Commutativity, *not* associativity — above two replicas the reduction order changes under a
permutation, so the harness refuses there. It compares two runs of the SAME graph rather than a
device answer against a host one, so **no linearity is required** and it transfers to `emarmsdrop`.

⚠⚠ **And it needs a second check pulling the other way.** Swap-invariance alone is satisfied by a
mask that reaches NOTHING — §7b's ones-mask blindness one level up. What witnesses that replica 0
actually received a different mask is the replica-0-LOCAL output: the batch statistics and `%loss`,
which must MOVE.

| | measured |
|---|---|
| ① all-reduced `θ'/m'/v'` BIT-IDENTICAL under the swap | **12,061,074 / 12,061,074** |
| ② `%loss` + batch stats MOVED (replica-0-local) | 2.430799 → 2.428901 · **38,966 / 42,016** |
| ⚠ CONTROL `DROP_FAULT=replicate` (the pre-fix world) | ① fires, **12,044,574** move, rel 0.691, rc=1 |
| ⚠ CONTROL `PJRT_DP_NO_MASK_SHARD=1` (flag off, buffer global) | the shim REFUSES on arity, rc=1 |

Both controls pass ②, so **① discriminates and ② is anti-vacuity.** Neither is evidence alone.

**Three findings:**

1. ⚠⚠ **A SHARDED INPUT THAT IS ALSO AN OUTPUT NEEDS THE PER-REPLICA SIZE ON THE WAY BACK.** The
   drop masks are the first — `x` and the labels are inputs only — so the DP output walk had always
   taken the declared size for granted. It **caught itself** (`output 740 size mismatch: graph 128
   bytes, caller 256`) rather than reading past the end: the shim's G4 guard, third time.
2. ⚠ **`%loss` IS REPLICA-LOCAL AND THE RETURN LAYOUT HIDES IT.** `θ' ++ m' ++ v' ++ [%loss, %bc1,
   %bc2] ++ bnstats`, so any range reaching the scalars sweeps `%loss` in. Counting it as
   all-reduced made ① read **one** differing output of 12,061,077 — and *one* is the tell, because a
   wiring defect moves millions.
3. ⚠ **§5b'S FRAMING WAS THE OBSTACLE.** It asked which of two EXISTING constructions to use, and
   both answers were "neither". Dropping the requirement that the gate compare against a
   host-computable or single-device answer removes the linearity constraint entirely. **When two
   known constructions both fail, check whether the property they share is actually required.**

⚠ Still single-device for the FEATURE: this gates the DP render's mask plumbing, it is not a DP
stochastic-depth training run, and ConvNeXt/ViT still need the §2b batched-index move first.

### §0.5 ✅ DP IMAGENET RECIPE PARITY (2026-08-02) — and the control it broke

**The DP ImageNet renders were THREE features behind their single-device peers**: `wd = 1e-4`
(500× off), no `wdExcludeNormBias`, no grad clip. An ImageNet run loads the DP render, so *the
artifact a real ViT/ConvNeXt pair run would have used matched none of its reference's optimizer
recipe*, while the corrected single-device renders sat beside it unused. Found by **listing what
each artifact bakes** — §0.4 finding 5 one axis over (there Imagenette-vs-ImageNet, here
single-device-vs-DP).

* **decay fixed on all four** (`vitin_adam128`, `vitin_adamdp128x4`, `cnxin_adam`, `cnxin_adamdp`).
  The re-render diff is **exactly 4 lines, all `%wd`**; every other committed artifact is
  byte-identical. ⚠ They stay short of the reference in the ways their docstrings list (no `wx`, no
  clip, one-hot targets) — **the decay was never on that list, it was a typo**, and no config
  anywhere says "ImageNet ViT at 1e-4".
* **the shipping renders exist**: `vitin_adamdp128x4wxclip` (global 512) and `cnxin_adamdpwxclip`,
  `wd = 0.05`, decay partitions 126/74 and 121/59 matching their single-device peers,
  **200 / 180 all_reduces (not 400 / 360)**, every one before the norm fold.
* **gated at 4 replicas, bit-exact** under the det shim: ViT **17,152,251/17,152,251** floats,
  ConvNeXt **85,762,779/85,762,779**. The wd-fixed no-clip pair re-gates bit-exact too, so the
  committed DP numbers survive the change.

#### ⚠⚠ AND THE STANDARD DP CONTROL IS BLIND ON A CLIPPED RENDER — a new hole in a gate every DP render uses

The sum-not-mean control (200 divisors 4.0 → 1.0) **passes bit-exact, rc=0**, on
`vitin_adamdp128x4wxclip`. Not a harness fault:

> **Global-norm clipping is SCALE-INVARIANT where it saturates.** `g · min(1, c/‖g‖) = c·g/‖g‖`
> whenever `‖g‖ ≥ c`, and that is invariant under `g → λg`. A collective wrong by ANY scalar —
> sum-not-mean, a wrong replica count, a doubled gradient — is **exactly** normalised away.

Pinned to the clip rather than to anything else that changed, by two runs: the same control on the
**no-clip** DP render fires at norm-rel **2.965**, and composed with `perturb_clip.py hi` (threshold
above the norm ⇒ clip inert) it fires on the **clipped** render at **identically 2.965**.

⚠ **The composed control is the RIGHT control, not a workaround.** Where the clip saturates a scale
error is *both* invisible to the gate *and* harmless to training — the update is `c·g/‖g‖` either
way. Where it does not (late training, ‖g‖ < c) the clip is the identity and the error is fully live
*and* fully visible. Raising the threshold moves the control to the only regime in which the defect
it hunts has consequences. Generalised: **a gate whose downstream NORMALISES the perturbation away
cannot test for it** — §0.4 finding 1 one level over. The recipe is in `TestViTDpCheck.lean`.

⚠ **This binds every clipped render anyone adds from here.** The saturated regime is the default:
the reference measured the init norm at 10×+ the threshold, and `clip-tie` measures 26-45×.

### §0.4 ✅ WHAT LANDED 2026-08-02 — the day's threads, and the findings that outlive them

*Everything below this line is DONE. It is kept for the FINDINGS, not because anything is live —
§0.3 is the complete list of what is owed. In order: ViT's EMA peer · stochastic depth's two
interior gates · mixup + cutmix (producer + the smoke that caught a defect) · `wdExcludeNormBias`
on ViT then ConvNeXt. The `▶ WHAT LANDED` blocks further down are earlier sessions.*

**The five findings from these threads that generalise past their features:**

1. **An identity gate at the neutral value cannot see WHERE that value is applied.** The ones-mask
   forward gate passes bit-exact on a deliberately MISPLACED drop site — `1·(branch+x) = branch+x`.
   Every endpoint gate in the original stochastic-depth set was blind to the placement.
2. **A gate that exercises one split/mode cannot see a split-dependent defect.** All three
   producer-side mixup gates were green while `SHIM_MIX` was killing the validation shim; only the
   end-to-end smoke saw it. Same shape as the duplicated-batch DP hole.
3. **Recover a constant by READING it, not by FITTING it.** A least-squares λ recovery was ~1 ULP
   off and produced a phantom failure in a correct producer feeding a bit-exact assertion.
4. **A tolerance must be in the unit the instrument has.** The `wdx-tie` offset gate failed a
   correct render at 1.39e-2 against an absolute 1e-3 bound; in ULPs of θ' it reads **0.49**.
5. **Gate the PARTITION, not the count** — and, one level up, *the same edit can behave differently
   on two renderers* (ConvNeXt derives its entry name from the variant, ViT takes it explicitly;
   only running both found it).

#### ✅ ViT's EMA peer — the EMA scorecard is 3 of 3

`verified_mlir/vit_ema_train_step.mlir`, `LEAN_MLIR_VARIANT=ema`. **807 in / 805 out** =
605/603 + 200 + 2, zero new `SHlo` ops (fourth time). It was the cheapest of the three exactly as
scoped — LayerNorm ⇒ `hasBn` false ⇒ the driver's whole `ema_bn` arm is skipped rather than
special-cased — and **the driver needed nothing**: `nRegions`/`nScalars` and the checkpoint size
guard were already generic. 69 functional lines in the renderer, plus the caller's two knobs.

Gates: gate 1 **0 diff lines** across all 12 committed `vit*`/`vitin*` artifacts (writers FORCED) ·
**`decay = 0` ⇒ shadow BIT-IDENTICAL, 0 of 5,526,346** · ratio known answer **5.96e-5** against a
no-warmup-correction control at **150,995×** · **residency at 4 regions, 0 of 88,421,536 bytes**,
both controls firing · size guard throws on a forged 3-region file · shadow **tracks then exceeds
from epoch 1** (42.37/50.65/54.34/56.61% vs live 40.64/42.93/48.41/51.64%, chance 10%) · writer
audit **96 artifacts, one writer each** · build 3,911 green. Full write-up in `planning/ema.md`.

**Three things from it that outlive the feature:**

1. ⚠⚠ **A GATE PORTED BETWEEN NETS INHERITS THE SOURCE NET'S CONDITIONING, NOT THE TARGET'S.** The
   ratio gate measures `ema₁ − θ₁ = d₀·(θ₀ − θ₁)`, a difference of two nearly-equal f32 numbers. At
   ViT's step 1 the warmup LR is `3e-4/(5·295) ≈ 2e-7`, putting `ema − θ` at ~**one ULP of θ** — so
   the first run read 1.96e-2 with its control only 455× away, and *nothing was wrong with the
   render*. ConvNeXt never saw this because its baseLR/warmup give a **50× larger** step 1. Fixed by
   conditioning the instrument (`LEAN_MLIR_BASE_LR_U=100000`, the `r34-mom-tie` move), after which
   the same gate reads 5.96e-5 against 150,995×. §2j's rule in a fourth place.
2. **The train losses are IDENTICAL to six decimals across the `ema`/`adam` pair, all four epochs**,
   over two different HLO programs (805 outputs vs 603). ConvNeXt's were merely close. That is what
   turns the shadow-vs-live accuracy table from "two similar runs" into **one θ trajectory read two
   ways** — a control, not a coincidence — and it is the cheapest evidence that the EMA op perturbs
   the optimizer not at all.
3. ⚠ **An `ema` run on Imagenette is a GATE VEHICLE, not a matched pair.** `vitTinyConfig` sets no
   EMA at all; 0.99996 is `vitTinyImagenetConfig`'s DeiT value, carried as the knob's default so a
   future ImageNet pair needs no edit. Do not quote the 4-epoch numbers as a reference comparison.

⚠ Still owed: the `emadp` DP peers on **both** ViT and EfficientNet (`vitAdamVariant 32 2 true`
names ViT's; nothing renders either), and any long run.

#### ✅ Stochastic depth — the two INTERIOR gates (`lake build droppath-tie`)

Both interior gates landed, and the tail cost what it was scoped at. Full write-up in
`stochastic_depth.md` §7a-c; the headline is that **it found the endpoint gates were blind to the
thing they were assumed to cover.**

* **Gate A, the known answer** — `dropPathB` through the same `pretty` emitter at `B 8 × n 5`
  (`n ≠ B` so a wrong broadcast axis is a type error, not a different function), against a
  host-computed `s[j]·x[j,i]`: **BIT-EXACT 40/40**, with `dropPath_ones_id` and
  `dropPath_zeros_zero` read out on device (5/5 each). Two controls fire — the descriptor bug
  (every example gets `s[0]`) at rel 1.29, and `--break`'s 1% wrong scale at 35/40, exactly that
  row's five coordinates. **Bit-exact is the bar by argument, not by luck**: an f32 product needs
  ≤48 mantissa bits, so it is exact in the host's f64 and rounds to what f32 multiplication returns.
* **Gate B, the all-zero-mask control** — at `@efficientnet_drop_fwd` and `@…_fwd_eval`, floor
  bit-exact 320/320 under the det shim. B1 ones-mask vs the drop-free `efficientnet_fwd`
  **BIT-EXACT 320/320** on both. Then the placement pair: with site 0 zeroed, zeroing site 8 still
  moves the logits (rel 0.305 / 0.448, 0/320 exact), and with all nine zeroed the net still depends
  on its input (rel 0.631 / 0.0131, non-degenerate). Both are impossible if the scale sat on the
  block output.
* **The control that licenses gate B**: a render with all 9 sites moved onto the block output — two
  lines swapped per site, **same SSA names, order, types and line count**, so arity, op counts and
  the prefix audit are all unchanged. `--cand` drives it (`scripts/misplace_drop_sites.py`) and it goes red, rc=1.

Gates held: `verified_mlir/` **0 lines of diff**, writer audit **107 artifacts, one writer each**,
all four content audits OK, render coverage unchanged at 64/95, `TestDropPathRamp` green, `lake
build Proofs Certs Codegen` **3,912** jobs (unchanged — this adds a test, not a proof).

**Two findings that outlive the feature:**

1. ⚠⚠ **AN IDENTITY GATE AT `s = 1` CANNOT SEE WHERE `s` IS APPLIED.** `stochastic_depth.md` §7's
   own row — *"`fwd-tie` BIT-EXACT vs the committed forward, ones mask"* — **passes bit-exact on the
   misplaced render**, and must: `1 ⊙ (branch + x) = branch + x` exactly, so at keep = 1 the two
   placements are the SAME FUNCTION. The keep = 1 train-step gate the feature shipped with (0 of
   4,020,358 against AdamW) is that statement one level up and inherits the blindness. So **every
   endpoint gate in the original set was blind to the placement** — which is precisely why the
   interior ones were owed. It is §5's duplicated-batch hole one axis over: *a gate whose input
   makes the intervention inert cannot test the intervention.*
2. ⚠ **A CONTROL PROVES THE GATE GOES RED FOR THE REASON IT CLAIMS, not merely that it goes red.**
   The first misplaced-render run fired — at the wrong check, with the wrong cause. The comparator
   normalised by the *first* buffer's magnitude; the misplacement drove that buffer to identically
   zero, so the denominator went to zero and `rel` returned `0.0`. A **total collapse** was reported
   as *"the logits did not move — the mask is not reaching that site"*, which would have sent the
   next reader to the driver instead of to the render. The harness was green on the real artifact
   the whole time: the defect lived only on the failure path, the half a passing run never runs.

⚠ **Still open on stochastic depth**: the **asymmetric-batch DP gate** (§5b of that doc). The
duplicated-batch `*-dp-check` harnesses are structurally blind to a mask that is replicated instead
of sharded, and `shard-check`'s construction needs the gated slot linear in the gradient — false for
RMSProp's buffer, which is the net that wants both. Everything above is single-device.

#### ✅ mixup + cutmix — the producer half (`scripts/mixup_gate.py`)

`recipe_gaps.md` v1.3, and it cost what it was scoped at: **producer-side Python only, no render
change, no new op, and — verified rather than assumed — no Lean change either.** `generateShim`'s
`_mix` is a line-for-line numpy transcription of the reference's `_mixup`/`_cutmix`, selected by
`SHIM_MIX=off|mixup|cutmix|both` (`both` alternates per step, as the reference does), alphas from
the config with env overrides. Two startup refusals: `SHIM_MIX` without `SHIM_NCLASSES` (a mixed
label is a distribution; int32 cannot carry one) and an unknown mode.

**The gates, all three green, controls firing:**

| gate | result |
|---|---|
| **inert when off** | v1 `c375ad0f…`, v2 `f3a4b2a0…` — **byte-identical to the pre-mixing shim**, measured on the pre-change script |
| **determinism** | same seed ⇒ same digest on all three modes; **different seed ⇒ DIFFERENT** on all three |
| **known answer** | mixed stream vs the OFF stream at one seed: `t' = λ·t + (1−λ)·flip(t)` and `x' = λ·x + (1−λ)·flip(x)` **BIT-EXACT**, images and labels alike |
| cutmix structure | the box recovered **from the pixels** is a RECTANGLE, and λ = `1 − area/(H·W)` to 1e-6 — the label follows the CLIPPED box, not the drawn λ |
| ⚠ control A | unmixed images against a mixed target — **rejected**, 1.15 |
| ⚠ control B | λ wrong by 1% — **rejected**, 7.5e-3 |

**Four findings that outlive the feature:**

1. ⚠⚠ **RECOVER A CONSTANT BY READING IT, NOT BY FITTING IT.** The first λ recovery solved
   `tm = λ·t + (1−λ)·flip(t)` by least squares in float64 and cast to float32 — about a ULP off,
   which is fine for a tolerance check and **useless feeding a bit-exact assertion**. It produced a
   phantom failure (mixup batch 1 at 1.5e-08, batch 0 clean) in a correct producer. `t` is one-hot,
   so λ is *literally stored* at `tm[i, y_i]`; reading it is exact, needs no tolerance, and gates a
   real property for free — λ is a per-STEP scalar, so every identified row must agree.
2. ⚠ **A GATE THAT ONLY PRINTS CANNOT FAIL.** The inertness gate first printed its two digests with
   "compare against the doc" — the `vit-dp-check` defect in its purest form. It now checks pinned
   baselines and **SKIPS LOUDLY** at any other batch/batch-count instead of comparing incomparable
   numbers. Note "inert when off" is a **cross-version** claim: that the new shim's two modes agree
   with each other is trivial (one calls the other's code path), so it needs a constant measured on
   the shim as it was *before* mixing existed.
3. ⚠⚠ **THE SECOND DEFINITION'S COST IS NOW PRECISE.** The reference draws
   `jax.random.beta(fold_in(PRNGKey(seed), step), α, α)`; the shim draws from numpy's `Generator`.
   Same distribution, **different numbers, and no seeding makes them the same.** So a verified-vs-JAX
   pair under mixup agrees **in distribution, not per step** — strictly weaker than the augmentation
   pipeline, which is reused verbatim and cannot drift at all. Never quote the two as the same kind
   of agreement; and it is why the gates are known-answer rather than cross-path byte comparisons —
   *that comparison does not exist to be made.*
4. ⚠ **Gate the IMAGES, not just the labels.** A mixed label against unmixed pixels compiles,
   streams, trains and descends — it is simply a worse objective. Control A is exactly that, and it
   fires at 1.15 against a bit-exact pass.

#### ✅ …and its end-to-end smoke FOUND A DEFECT ALL THREE GATES MISSED

**After the fix** (mnv2/ImageNet, 1 GPU, 6 steps, fresh checkpoint per config; the trio run TWICE so
the floors are measured rather than assumed):

| run | step 0 | step 1 | step 2 | epoch 1 |
|---|---|---|---|---|
| **A** v1 hard labels ×2 | 7.019591 / 7.019591 | 7.017135 / 7.017126 | 7.069839 / 7.069849 | 7.057056 / 7.057081 |
| **B** v2 one-hot ×2 | 7.019591 / 7.020040 | 7.017126 / 7.016245 | 7.069456 / 7.068997 | 7.056798 / 7.056218 |
| **C** v2 **MIXED** | **7.029974** | 7.026065 | 7.063550 | 7.052764 |

**C is ~1.0e-2 above the unmixed band, ≈20× the B-vs-B run-to-run floor of 4.5e-4**, and *higher* —
what a higher-entropy target must do. A-vs-B sits INSIDE that floor. ⚠ 2 samples per config and
**not** under `det_shim.sh` — my own miss against the 2026-08-02 finding that the det shim is
mandatory for cross-process numeric comparison on CUDA. Quote the separation as ≈20× a measured
floor, nothing tighter.

#### The defect it found

The smoke was nearly skipped as confirmatory. It was not: **`SHIM_MIX=both` killed the trainer
before its first step.**

`SHIM_MIX` is an ordinary environment variable, so **every shim the driver spawns inherits it — and
it spawns TWO**: the train stream at `nclasses = K`, and the **validation drain at `nclasses = 0`**
(wire v1 hard labels, because eval scores against a label, not a distribution). Gating the mixing on
the variable alone made the *"needs `SHIM_NCLASSES>0`"* refusal fire on the **val** shim, which died
before writing its preamble; the trainer then reported `imagenet shim closed the pipe after 0 of 16
bytes`.

**Silencing the refusal would have been the wrong fix.** Mixup/CutMix are TRAIN-time augmentations —
the reference applies them in the train loop only, and a mixed validation target would score the net
against a convex combination of two labels, which is not the metric. The rule is *mix the train
split, never the eval one*: `_MIX_ON = (_MIX_MODE != 'off') and training`.

> ⚠⚠ **A GATE THAT EXERCISES ONE SPLIT CANNOT SEE A SPLIT-DEPENDENT DEFECT.** Gates 1-3 were green
> throughout and always would be — every one of them drives the **train** split. Nothing
> producer-side would have found this. It is §5's duplicated-batch hole in a third place (*a gate
> whose input makes the failure impossible cannot test for it*), and it is the argument for running
> the cheap end-to-end check even when every component gate is green.

`mixup_gate.py` grew **gate 1b**: at every mode the validation split must survive `SHIM_MIX` at wire
v1 — the driver's own spawn — and hash **identically to the unmixed** validation stream.

⚠ **Three ImageNet-smoke traps found on the way**, all of which will bite again:
* **`LEAN_MLIR_MAX_STEPS` is NOT the step cap for `trainAdamSched`** — that is `LEAN_MLIR_G2_STEPS`.
  `MAX_STEPS` means *"time a step window then exit"*, so it does not bound the run; a smoke set with
  it ran past step 300.
* **The ImageNet validation drain is UNCONDITIONAL** — 195 × 256 = 49,920 images, ~28 GB into host
  RAM, before the first train step, and **`LEAN_MLIR_SKIP_EVAL` does not skip it** (it gates the
  eval *pass*, not `loadData`'s drain). Budget ~4-5 min and 30 GB per smoke config.
* ⚠ Configs sharing slug+variant **must clear `.lake/build/<slug>_<variant>_ckpt_xla.bin{,.epoch}`
  between runs**, or the second silently resumes the first and the comparison is meaningless (§4).

#### ✅ `wdExcludeNormBias` — BOTH NETS (`lake build wdx-tie {vit,convnext}`)

The timm/DeiT `no_weight_decay` rule: decay only ≥2-D weight matrices, skip every 1-D param
(biases, LayerNorm γ/β, the CLS token) **and the positional embedding**. Run over the reference's
own `init_params`, that is **74 decayed / 126 excluded of 200**, every mask leaf uniform — so the
decision is per-TENSOR, which is what licenses a scalar operand.

**No new op, no interface change, no driver change.** `adamWParamF` already takes `wd` as a runtime
OPERAND NAME (the `%lr` shape, for the `%lr` reason), so "exclude" is binding it to a zero constant:
126 of 200 operand strings move and arity, types and regions do not. That is also why — unlike
`ema` (a 4th region) or `drop` (extra inputs) — the `wx` marker needs no driver predicate at all.

| gate | result |
|---|---|
| gate 1 | every committed artifact **byte-identical**; only the two new `wx` paths appear |
| ① decayed | θ' **BIT-EXACT** between `adam` and `adamwx`, 74/74 |
| ② excluded | `θ'_wx − θ'_adam` = `lr·wd·θ` to **0.49 ULPs of θ'**, 126/126 |
| ③ `m'`,`v'` | **bit-exact on all 11,052,692** coords — decoupled decay touches neither |
| ④ `%loss` | bit-exact |
| ⚠ control `invert` | **fires**, 200 misclassified, rc=1 |
| ⚠ control `swap1` | **fires on exactly 2**, rc=1 |

Audit **109 artifacts / one writer each**; drift-guard coverage **64/95 → 66/97** with
`render_guard_baseline.txt` **unchanged** (that file may only shrink); `TestVariantPredicates`
**30 spellings**, `wx` composed with all three axes.

**Three findings:**

1. ⚠⚠ **GATE THE PARTITION, NOT THE COUNT.** `swap1` flips one param each way, so the counts still
   read 74/126 — a gate checking *how many* params moved passes it. The gate instead recovers per
   parameter which bucket it EMPIRICALLY falls in (bit-exact ⇒ decayed; offset by `lr·wd·θ` ⇒
   excluded) and requires that partition to equal `vitWdDecays`' **name for name**. A mask
   excluding the wrong 126 is otherwise silent in the arity, the types and the prefix audit.
2. ⚠ **A TOLERANCE MUST BE IN THE UNIT THE INSTRUMENT HAS.** ② first gated at an absolute 1e-3 and
   **failed a correct render at 1.39e-2**. `θ'_wx − θ'_adam` is a difference of two nearly-equal
   f32s, so its best achievable relative accuracy is `ulp(θ')/(lr·wd·|θ|)` ≈ 6e-2 here — and
   **independent of `lr`**, because both the difference and `|θ'|` scale with it. (Turning `%lr` up
   is what conditioned the EMA ratio gate; it does nothing for this one — the same trap, immune to
   the same fix.) Restated in **ULPs of θ'** it reads **0.49**, i.e. sub-ULP.
3. ⚠ **A "non-degenerate" input can be worse than a degenerate one.** The driver's init zeroes every
   kind-2 param — which is most of the 126 — making `lr·wd·θ = 0` and ② vacuous on the half it
   exists to test. My first fix gave *every* param a value centred at 0.6, which overflowed the ViT
   forward: `%loss NaN`, `0/5,526,346` bit-exact, `max abs 0.000000`. That combination is the tell
   (NaN ≠ NaN makes every coord "differ" while every `>` stays false) and it now refuses up front.
   Keep the driver's He scaling; move only the zeros.

#### ⛔ AND IT TURNED UP A SEPARATE 500× GAP — the ViT renders bake the WRONG weight decay

`vitAdamConsts` baked `%wd = 1e-4` for **every** ViT render. That is `vitTinyConfig`'s (Imagenette)
value — but **`vitTinyImagenetConfig.weightDecay := 0.05`**, the DeiT one. An ImageNet ViT render
was training at **1/500th of its reference's decay**: the `RenderCifar8Sgd02` / EfficientNet-16×
shape (§2a-quater), a silently wrong hyperparameter that compiles, runs and descends.

`wdStr` is a parameter now (default unchanged ⇒ gate 1 stayed free) and `vitin_adam128wx` renders at
**0.05** — both halves of the recipe, the magnitude and the mask.

⚠ **The same 500× gap is on ConvNeXt** (`convnextTinyImagenetConfig.weightDecay := 0.05`), and
`cnxin_adamwx` renders at 0.05 for the same reason `vitin_adam128wx` does.

⚠ **`vitin_adam128`, `vitin_adamdp128x4`, `cnxin_adam` and `cnxin_adamdp` are STILL at 1e-4 and
were NOT touched** — a separate call with its own blast radius (the DP peers, the residency rows,
the committed `vit-dp-check`/`convnext-dp-check` numbers). **Owed, and it must be closed before any
ViT or ConvNeXt ImageNet pair run**: a matched pair at the wrong decay is not a matched pair.

#### ✅ ConvNeXt followed the same day — 59 decayed / 121 excluded of 180

The prediction held: **the plain rank test, no name carve-out.** ConvNeXt's generated reference sets
`_WD_POS_SHAPE = None`, so ViT's `nm != "pos"` has no analogue — carrying it over would have
transcribed a rule this net does not have. Excluded: every LN γ/β, every conv bias, and LayerScale
γ (1-D, so structurally rather than as a special case).

| | ViT | ConvNeXt |
|---|---|---|
| params (decayed / excluded) | 200 (74/126) | 180 (**59/121**) |
| ① decayed θ' bit-exact | 74/74 | **59/59** |
| ② excluded offset = `lr·wd·θ` | 0.49 ULPs | **0.48 ULPs** |
| ③ `m'`,`v'` bit-exact | 11,052,692 | **55,652,564** |
| controls `invert`/`swap1` | 200 / 2 | **180 / 2** |

**ONE harness for both** (`wdx-tie <net>`), per `rms-tie`/`shard-check` — a second copy is the
double-writer disease one level down, in code. ⚠ A green ViT run does not license ConvNeXt: the two
rules genuinely differ, the `rms-tie` ε-placement lesson one knob over.

⚠⚠ **AND THE SAME EDIT BEHAVED DIFFERENTLY ON THE TWO RENDERERS.** ConvNeXt DERIVES its entry name
(`{slug}_{cnxAdamVariant …}`) where ViT takes `funcName` explicitly, so the flag had to reach the
*variant* too. It did not, and `convnext_adamwx_train_step.mlir` came out declaring
`@convnext_adam_train_step` — an entry disagreeing with its own path. **The shim refused the call**
(`mlp train step failed` at the first invoke) rather than running the wrong graph, which is
§2b-quater's entry check earning its keep a second time. `#guard`s pin the `wx` spellings now.
*Only running both nets found it.*

Audits after both: **111 artifacts / one writer each**, drift-guard coverage **64/95 → 68/99** with
`render_guard_baseline.txt` unchanged, prefix audit green, `TestVariantPredicates` 30 spellings.


#### ▶ Earlier that day, sessions three and four (4 commits)

* **ViT's EMA peer** — the section above. EMA is **3 of 3**; zero new ops on all three nets.
* **Stochastic depth, end to end on EfficientNet single-device** — the op (`dropPathB`), its cert,
  the render, the driver, and an exact endpoint gate. `LEAN_MLIR_VARIANT=adamdrop` trains.
* **`recipe_gaps` v1.2c** — EMA and dropPath carried to the ImageNet slugs (8 renders).
* The **CI drift guard widened twice more**: coverage **40/81 → 64/95**, baseline **46 → 36**.
  That file may only shrink.

**▶ Stochastic depth, where it stands.** One `SHlo` op whose **VJP is `layerScale_has_vjp`
verbatim** — no new certificate, and the backward emits the SAME constructor on the cotangent. The
exact gate, under `det_shim.sh` over 3 steps: at every keep = 1 the drop op is the identity, so
`adamdrop` must train what `adam` trains — **0 of 4,020,358** against a **0** floor, with the real
recipe firing at norm-rel **1.89**. That pins the wiring: a site on the wrong side of the skip add,
a scale reaching the identity path, or a backward that dropped the SKIP cotangent each fail it.

**Five findings from those two sessions that outlive the features:**

1. ⚠⚠ **`sd` COLLIDED WITH `rmsdp`, AND THE COLLISION WAS BETWEEN TWO *OTHER* MARKERS MEETING.**
   `rms` ++ `dp` spells `rmsdp`, which contains "sd" — so the stochastic-depth predicate fired on
   every RMSProp data-parallel variant, including the committed and gated `enetin_rmsdp64`. This is
   `ema.md`'s `emarms` defect a second time, one axis on, and *reading names one at a time cannot
   find it*. **With N markers, check every CONCATENATION, not every marker.** Renamed to `drop`;
   `tests/TestVariantPredicates.lean` now runs all 23 spellings × 3 axes.
2. ⚠⚠ **A GATE PORTED BETWEEN NETS INHERITS THE SOURCE NET'S CONDITIONING.** ViT's EMA ratio gate
   read **1.96e-2 with nothing wrong**, because ViT's step-1 warmup LR (~2e-7) puts `ema − θ` at one
   ULP of θ. ConvNeXt never saw it — its baseLR/warmup give a 50× larger step 1. Conditioning the
   instrument (`LEAN_MLIR_BASE_LR_U`) took the same gate to 5.96e-5 against a 150,995× control.
3. ⚠⚠ **THE DET SHIM IS MANDATORY FOR ANY CROSS-GRAPH NUMERIC COMPARISON ON CUDA.** The keep = 1
   gate first read as a **2.5e-2 failure**; it is bit-identical under `scripts/det_shim.sh`. §2d.3's
   Finding 1 ("the floor IS bit-exact across processes") is ROCm-specific. **Measure the A-vs-A
   floor before reading any cross-graph number** — twice now that has been the difference between a
   green feature and a phantom defect.
4. ⚠ **SCOPE BY THE INDEX CONVENTION, NOT BY OP COUNT.** `stochastic_depth.md` recommends ConvNeXt
   as "the cheapest"; measured, ConvNeXt and ViT render at the **per-example** index (10 and 13
   per-example forms against ~2 batched), where a per-example mask is §4's descriptor trap. Only
   EfficientNet/R34/mnv2 sit at `N := B`. §2f's lesson one net over.
5. ⚠ **A FEATURE IS NOT DONE WHEN ITS IMAGENETTE ARTIFACT RENDERS.** Found by *listing* artifacts:
   RMSProp had been carried to both scales, EMA and dropPath had not — so the ImageNet trainers did
   not carry the features their reference numbers depend on. Both scales are one `#eval` apart,
   which is exactly why it is easy to stop at one. Closed as v1.2c.

### ▶ WHAT LANDED 2026-08-02, second session (5 commits)

* **RMSProp's DRIVER half** — mean-square init 1.0, exponential LR decay, both DP renders compiled
  and gated at 4 replicas. mnv2 is now at feature parity with its reference except bf16.
* **EMA on ConvNeXt** (single-device + DP peer) **and EfficientNet** (`emarms` = the reference's own
  RMSProp + exp-decay + EMA recipe, plus the `ema_bn` BN-buffer shadow). **Zero new `SHlo` ops** on
  either — the third time reading a reference update against existing ops at their other readings
  has collapsed a scoped op family to nothing.
* **Two specs**, `planning/ema.md` and `planning/stochastic_depth.md`, both written before building
  and both re-tiering their `recipe_gaps` entry.

**Four findings from it that outlive the features:**

1. ⚠ **`shard-check` cannot gate an optimizer whose tail is NONLINEAR in the gradient.** Its known
   answer `DP([A|B]) = mean(single(A), single(B))` needs the gated slot linear — true of AdamW's `m`
   at `m = 0`, false of RMSProp's buffer. Use the duplicated-batch identity in `*-dp-check`, which
   is optimizer-agnostic. All three `dp-check` harnesses now take `DP_NET`/`DP_VARIANT{,_DP}`/
   `DP_REPLICAS`/`DP_BATCH` and still reproduce their committed 2-replica results with no arguments.
2. ⚠ **NEVER GATE ON THE EMA SHADOW.** §3 says gate the gradient, never θ; the shadow is θ's
   low-pass filter, so it is that failure one level worse — measured, a sum-not-mean control moved
   `m` by 0.94, θ by 1.95e-4 and the **shadow by 1.00e-4, exactly ON a 1e-4 gate**.
3. ⚠ **A prefix test on a variant name that encodes TWO axes fails quietly.** `emarms` does not
   start with `"rms"`, so the RMSProp mean-square would have initialised to 0 — the exact defect
   that thread existed to fix, reintroduced through a name. Substring tests now, eight spellings
   checked.
4. ⚠ **`residency_gate.sh` deletes `<slug>_<variant>_ckpt_xla.bin` between its four passes**, so
   running it against a live trainer on the same slug AND variant produces a **false green**. Run it
   on an idle box; `ps -eo comm | grep verified` first.

### ▶ WHAT LANDED 2026-08-01/02 (7 commits) — read this before assuming anything below is current

* **All five nets now have a gated ImageNet trainer** — `resnet34in`, `vitin`, `cnxin`, `enetin`,
  `mnv2in` (§2p). **Every one matches its JAX reference param count exactly**: 21,797,672 ·
  5,717,416 · 28,587,592 · 5,288,548 · 3,504,872. Each is gated on DP, residency and an
  end-to-end run on real ImageNet over the shim. **None has a descent run** — the smokes are 40
  steps, which shows they RUN, not that they LEARN.
* **`planning/recipe_gaps.md` is new and is the plan.** Its headline: **ResNet-34 is already at
  feature parity with its JAX reference except bf16**, so v1 for R34 is a run, not a build.
* **Soft targets need NO render change** — the committed renders are AFFINE in `%onehot`
  (`lake build soft-target-tie`; ViT 492× / ConvNeXt 309× separation). §2p's earlier claim that
  mixup needs a `softLabelCE` cotangent is **retired**. `lean_fill_targets` in the C shim takes
  either int32 labels or a float32 distribution, dispatched on buffer size.
* **Shim wire v2** carries `float32[batch·nClasses]` soft targets (`SHIM_NCLASSES` / `SHIM_SOFT`),
  gated bit-identical against v1 with a refusal control. **Shim sharding** (`SHIM_SHARD`,
  `SHIM_WORKERS`) exists and is inert at the default — ⚠ and is NOT needed: the loader was measured
  and is **not** the bottleneck (synthetic 424 ms/step vs real-data 427).
* **⛔ FIVE COPIES OF ONE BUG were found across FOUR nets** — a label-smoothing constant hardcoded
  at the K=10 value: R34's cotangent (§2k), ConvNeXt's loss, EfficientNet's loss, and mnv2's
  cotangent **and** loss. All derived now. **It survives the obvious check** because that greps the
  cotangent's negative spelling `-0.010000` and the loss copy is positive-signed and on no gradient
  path. Any new emitted constant depending on `nClasses`/`alpha`/batch must be derived and gated by
  a byte-identical re-render — which is how a wrong `α` was caught mid-edit on mnv2.
* **`shard-check` is N-replica now** (`SHARD_REPLICAS`, `SHARD_VARIANT{,_DP}`), gated by
  reproducing the 2-replica harness it replaces to the digit. ⚠ That run **retired a stale number**:
  §5 records ConvNeXt's shard CONTROL as 0.137; both harnesses agree on **0.041645** today — 0.137
  predates §2o's channel-LN flip.
* **`residency_gate.sh` takes `$GATE_DEVICES`** (default "0"), which is what makes DP renders
  gatable at all; and **ViT is in `residency_gate_all.sh` at fault mode 2**, because ViT+AdamW is
  CONTRACTIVE — it absorbs a 1-ULP fault to 1 byte of 66M where R34 amplifies to ~184M of 255M.
  "AdamW ⇒ chaotic ⇒ mode 1" is refuted.

### ▶ ALSO LANDED 2026-08-02 (branch `rmsprop-render`, 2 commits) — **RMSProp, both nets**

`recipe_gaps.md` v1.2's render half is **done and certified**, and its own cost estimate was wrong
by a tier: filed as Tier D *"a new proven `SHlo` op family, ten sites each"*, it is **ONE op**.
Three of the four steps are existing certified ops read differently — `momVNextF` at
`(μ := wd, v := θ)` is the coupled L2, and **`adamVNextF` at `β₂ := ρ` IS the running mean-square**
(`rmsSqNext_eq_adamVNext`, by `rfl`). Only the ε-inside-the-root normalise had to be built.

* **`Proofs/Codegen/RmsPropStep.lean`** + `rmsBufNextF` at all ten sites + 13 theorems, 3-axiom clean.
* **Six artifacts**: `{mobilenetv2,efficientnet}_rms`, `{mnv2in,enetin}_rms64`, `…_rmsdp64`.
  `[θ|m|v]` reused with `m` = buffer, `v` = mean-square ⇒ **signature byte-identical to each net's
  AdamW render apart from the entry name**; no driver change.
* **`lake build rms-tie && rms-tie [mobilenetv2|efficientnet]`** — ①②③ ≤ 1.1e-6 on both, with the
  **textbook-ε control missing by 365,412× (mnv2) and 774,497× (enet)**.
* ⚠ **The two nets sit on opposite sides of the ε placement** and it is now measured, not argued:
  the control fires ~4× harder on enet (ε = 1e-3) than on mnv2 (ε = 1.0), exactly as
  `rmsBufNext_eps_placement_at_zero` predicts. **Neither net's green run licenses the other.**
* **Gate 1 held at 0 diff lines** through both threadings; writer audit 91/one-writer-each.
* The **CI drift guard was widened** because the coverage check demanded it: `MobileNetV2RenderB`
  had no step at all and `EfficientNetRender` diffed 1 of the 14 artifacts it writes. Coverage
  **13/74 → 33/80**, `render_guard_baseline.txt` **61 → 47** — that file may only shrink.
* ~~⛔ **NOT a matched pair yet.**~~ ✅ **THE DRIVER HALF LANDED 2026-08-02 — see the next
  section.** All four things that block were owed are done: the mean-square init, the exponential
  schedule, the DP renders compiled and numerically gated at 4 replicas, and descent runs on both
  nets.

### ▶ AND THE DRIVER HALF LANDED 2026-08-02 — all four owed items, ⚠ but see the run caveat

* **`trainAdamSched` gained `expDecayRate`/`expDecayEpochs`** (trailing, optional, default 0.0 =
  cosine) and **initialises the mean-square slot `v` to 1.0** on any `rms*` variant. Six functional
  lines. The reference's peak LR + decay live in ONE record — `RmsSchedule` in `VerifiedNets.lean`,
  read by all four mnv2/enet entry points — while the **emitted** half (ρ/μ/ε/wd) stays in
  `Proofs.StableHLO.RmsHyper`. Two modules deliberately: `%lr` is a runtime operand precisely so one
  graph serves a whole schedule, and a learning rate that becomes a graph constant is the
  `RenderCifar8Sgd02` / enet-16× silent-hyperparameter failure (§2a-quater) waiting to recur.
* **INERT when off, measured**: cifar8-bn AdamW is bit-identical across the change — **0 bytes of
  638,904** — against a cross-process FLOOR of **0** under `scripts/det_shim.sh`. ⚠ That floor is
  NOT free on CUDA (§2d.3's Finding 1 is ROCm-specific); without the det shim the comparison has no
  resolution and neither verdict means anything.
* **Each half fires, and separately.** The ms-init alone moves **20,929,404 of 26,840,184 bytes** —
  pre- vs post-change driver on the same mnv2 render, same LR, entirely inside warmup so the
  schedule is identical, i.e. a genuine single-variable control. The schedule is gated as a **known
  answer**: the six per-epoch LRs across the warmup boundary match the reference formula recomputed
  in Python to **≤3.5e-7**, the rounding of the driver's own six-decimal print.
  ⚠ **`_global_step` in the emitted reference is 0-BASED** where this driver's `gstep` is 1-based,
  so the epoch is `(gstep−1)/nb`. One step of offset is invisible in a 5004-step epoch — read
  `jax/Jax/Codegen.lean`'s generator, not the prose.
* **Both DP renders compile and RUN at 4 replicas**, the third owed item. `mnv2in_rmsdp64` /
  `enetin_rmsdp64` pass the duplicated-batch identity: forward **BIT-EXACT** (34,112 / 42,016 BN
  statistics), buffer norm-rel **7.8e-7 / 8.1e-7**, against sum-not-mean controls firing at
  **2.22 / 2.39** with rc=1 — six orders of separation.
* ⚠ **`shard-check` CANNOT gate these, and that is a general finding about that harness.** Its known
  answer `DP([A|B]) = mean(single(A), single(B))` requires the gated slot to be **linear in the
  gradient** — true of AdamW's `m` at `m = 0` (`m' = (1−β₁)·g`, which is *why* it gates `m` and not
  `θ'`) and **false of RMSProp's buffer** `b' = μ·b + gw/√(ρ·s + (1−ρ)·gw² + ε)`. The
  duplicated-batch identity in `*-dp-check` is optimizer-AGNOSTIC — both sides get the identical
  gradient — so that is the construction that transfers. Both dp-check harnesses were generalised
  (`DP_NET` / `DP_VARIANT{,_DP}` / `DP_REPLICAS` / `DP_BATCH`, the `TestShardCheck` shape) instead of
  a third being written, and each still reproduces its committed 2-replica result with **no
  arguments** — the gate on the generalisation itself.
* **Gate 1 held**: `verified_mlir/` **0 lines of diff** after FORCING both renderers' `#eval`s with
  `lake env lean` (§2n's vacuous-green trap — a plain build can leave the writers unrun); writer
  audit **91 artifacts, one writer each**; `lake build Proofs Certs Codegen` **3,911** green;
  `rms-tie` still certifies both nets (textbook-ε controls at 255,266× / 858,014× the tie).

⛔ **IT DESCENDS — BUT THESE ARE NOT 80-EPOCH NUMBERS. The runs were KILLED mid-flight**, at
Brett's instruction, because sustained 4-GPU load destabilises this box. Read the table as descent
evidence and nothing more:

| run | stopped at | first val | best val |
|---|---|---|---|
| **mnv2 `rms`** | ep 60 of 80 | 19.80% | **76.48%** |
| mnv2 `adam` (control) | ep 61 of 80 | 28.66% | 82.42% |
| **EfficientNet `rms`** | ep 50 of 80 | 38.78% | **85.50%** |
| EfficientNet `adam` (control) | ep 50 of 80 | 37.58% | 80.36% |

⚠ **Not comparable to §0b's 80-epoch table** (mnv2 86.73%, enet 88.20%) on three axes at once: a
different box, **224² with NO random crop** (`crop := (px == 256)`, and this box stores Imagenette
train at 224² — §0's ViT note), and stopped early. The AdamW columns are there precisely so the
RMSProp ones have a same-box, same-augmentation peer; they are the only fair comparison available,
and even they are interrupted at different epochs.

✅ **The residency gate on the `rms` variant PASSES on both nets** — re-run uncontended
2026-08-02, and **`scripts/residency_gate_all.sh` carries them now** (`mnv2-rms`, `enet-rms`), so
they run by default rather than by anyone remembering:

| | bytes | FLOOR | TEST | init control | staleness fault |
|---|---|---|---|---|---|
| **mnv2 `rms`** | 26,840,184 | 0 | **0** | 16,197,508 | 19,311,310 |
| **EfficientNet `rms`** | 48,244,296 | 0 | **0** | 11,920,572 | 29,233,518 |

⚠ **RMSProp is CONTRACTIVE, so these rows are fault mode 2** — measured uncontended on mnv2, the
1-ULP fault lands at **2 bytes of 26,840,184** at ten steps, i.e. ViT's situation exactly: it
passes mode 1 only because 2 > 0, and one more step of contraction would report VACUOUS. This is
the **third** data point against "AdamW ⇒ chaotic ⇒ mode 1" and the first where the property tracks
the *optimizer* rather than the net — both nets contract, despite sitting on opposite sides of the
ε placement.

⚠ **And the first two attempts at this gate were VOID for a reason worth knowing before reusing
it**: `residency_gate.sh` deletes `<slug>_<variant>_ckpt_xla.bin` between each of its four passes,
so running it while a trainer is live on the same slug **and variant** lets that trainer's
per-epoch checkpoint write land mid-gate. It does not crash — mnv2 reported a clean **PASS** and
enet a saturated floor, and only the second looked wrong. Run it on an idle box; the runner now
says so in its own header.

### ▶ R34/ImageNet — ⛔ BLOCKED ON HARDWARE, NOT ON CODE (2026-08-02)

⚠ **This was §0's headline until 2026-08-02 and it is still the right run — but it cannot be done on
this box.** It is ~16 h on 4 GPUs, and sustained multi-GPU load destabilises ares (see the top of
§0). Nothing below is stale; the preflight was green and the rig smoke-tested. **Everything here
stands for the day a box can sustain it.** Do not re-derive it, and do not start it here without
asking.

### ▶ THE JOB: get R34/ImageNet over the line

§2d.3 (device-resident parameters) made this **feasible for the first time** — 4-GPU R34/ImageNet
went **596 → 386 ms/step**, so a 30-epoch run is **~16 h** where it was ~25 h. The 4-GPU rig was
smoke-tested end to end: 6 epochs × 400 steps, `rc=0`, loss **7.03 → 5.65** monotone, all four cards
47-57 °C at ~80 W. Nothing structural is in the way.

**Preflight run 2026-08-02 and it is GREEN**: binary rebuilt against today's `spawnShim` /
`readShimBatch` signature changes, shim carries both new features with v1 output byte-identical,
artifact untouched, all 6 GPUs idle, `scripts/jobs/r34-imagenet-4gpu.conf` correct.
⚠ **The epoch-6 checkpoint from the 08-01 smoke WAS ARMED and has been moved aside** (to the
session scratchpad, not deleted) — it came from a 400-capped-step run, so resuming it would have
carried 8%-of-an-epoch momentum into epoch 7 and done 24 epochs, `rc=0`, silently.
⚠ Note `r34-imagenet-4gpu.conf` uses `DEVS="0,2,3,4"`, **not** 0-3 — it excludes the AER-unclean
cards. The smokes in this file used 0,1,2,3, which is less conservative.
⚠ 30 epochs is the `resnet34ImagenetConfigShort` tier; the JAX **72.02%** is a **90-epoch** number,
and no JAX 30-epoch run exists to compare against.

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-imagenet-verified-xla
scripts/gen_shims.sh                                      # the five per-net data shims (§0.9)
cat .lake/build/resnet34in_momdp64_ckpt_xla.bin.epoch 2>/dev/null   # ⚠ READ THIS FIRST (§4)

PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LEAN_MLIR_VARIANT=momdp64 LEAN_MLIR_BATCH=64 LEAN_MLIR_BASE_LR_U=100000 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/resnet34-imagenet-verified-xla data 2>&1 | tee runs/r34in_4gpu_30ep_<date>.log
#   ⚠ do NOT pipe a long run into `head`/`grep -m1` — §4: SIGPIPE does not kill these trainers,
#     it DETACHES them. Redirect to a file and read the file. `scripts/supervise.sh` has AER
#     restart, thermal resting and a stall guard for exactly this (§2d.3a).
```

**Five things that will bite, in the order they will bite:**

1. **`PJRT_FFI_RESIDENT=1` is opt-in and there is no warning if you forget it** — the run just
   takes ~25 h instead of ~16. The banner `[pjrt_ffi] RESIDENT: … holds 330 parameter tensors` is
   the confirmation; if it is absent, residency is off.
2. **The checkpoint is per-variant** (`resnet34in_momdp64_ckpt_xla.bin`) and a marker at the epoch
   budget makes the run a **silent no-op that exits `done` with rc=0** (§4, §2o). `cat` the
   `.epoch` file before launching, every time.
3. **Do not rebuild `ffi/libpjrt_ffi.so` while a run is live.** It is `dlopen`'d and overwriting it
   in place can corrupt the mapping. (Got away with it once today; do not rely on that.)
4. **`pjrt_ffi_invoke_f32_resident_v2` — bump the suffix on ANY signature change**, and rebuild
   **every** `-xla` binary after one. A stale binary calls the new shim with the old argument list
   and every argument shifts; it is not a link error, it is garbage (§2d.3).
5. **ares has no P2P between any pair** (`nvidia-smi topo -p2p r` is `CNS` everywhere), so every
   `all_reduce` stages through host. That is the remaining 4-GPU shortfall and residency cannot
   remove it — it is real work.

**What the run is FOR** (§2k): a matched pair against `jax/MainResnetImagenet.lean`, same net, same
optimizer, same augmentation via the generated shim. The param counts already agree exactly at
**21,797,672** (§2l/§2m), which is the precondition that made the pair meaningful. Claim ceiling
stays §5's: *"one architecture, two independent lowerings, agreeing"*, never "proven".

### ▶ Then, in value order — **superseded by `planning/recipe_gaps.md`, read that first**

`recipe_gaps.md` enumerates every difference between the five verified ImageNet trainers and their
JAX references, classified by which LAYER each gap lives in (pipeline / producer / driver / render /
render+theorem), with the measured wall-clock budget: **~203 h ≈ 8.5 days** of 4-GPU time for all
five at reference epochs. Its ordering, in brief:

**v1.0 run R34** (zero build) · **v1.1** exponential LR decay + EMA (driver only) · **v1.2**
RMSProp (the ONLY thing between mnv2 and parity) · **v1.3** mixup+cutmix (wire is done) · **v1.4**
`wdExcludeNormBias`, grad clip · **v2** stochastic depth, bf16.

⚠ **Read that as HISTORY.** As of 2026-08-02 v1.0-v1.4 are done or hardware-blocked, and the live
order is **§0.2's**: ~~v1.5a the SHIM WIRING~~ (done, §0.9) → v1.5b the BATCHED-INDEX MOVE (then SD on ViT/ConvNeXt)
→ the mixup/cutmix run → bf16.

The list below is the pre-ImageNet ordering and is kept because the bf16 measurements are still the
best ones on file.

1. **bf16** — and its case just got much stronger, twice: measured **×1.76 on ares** (three runs;
   §3's ×0.96 is an RDNA3 result and does not carry), and residency raised what it is worth on the
   verified R34 step from **1.21× to 1.56×** because transport was masking the arithmetic. 4-6
   sessions, needs `conv_close_mixed`; `planning/bf16_renderer.md`.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified end to end.
3. **Executable cache** — 0.1% on a training run, 53% on the MNIST-MLP demo. Dev-loop only, and
   note residency already took the demo groups 2.8-5.2×, so re-measure the case before starting.

### ▶ ViT on FOUR replicas — ✅ RENDERED, GATED AND MEASURED 2026-08-01, not yet run long

Built as a code/test pass; **no training run yet, deliberately.** `verified_mlir/vit_adamdp32x4_train_step.mlir`
(bs32 × 4 = global 128), one `#eval` in `ViTRender.lean` — no renderer change, `replicas` was
already a parameter.

| gate | result |
|---|---|
| **gate 1** — the six existing `vit_*` artifacts re-render | byte-identical (`md5sum -c`), only the new path appears |
| **structure** vs the committed 2-replica render | the ONLY diff is the banner + 200 × `tensor<1x2xi64>` → `tensor<1x4xi64>`; forward and backward bodies byte-identical |
| **`vit-dp-check` at `VIT_DP_REPLICAS=4`** | **bit-exact 16,579,041 / 16,579,041**, control (200 divisors 4.0 → 1.0) fires at **2.99** |
| **residency**, 4 replicas | **0 bytes differ of 66,316,152**, floor 0, both controls fire |

**Speed — ms/step probe, `LEAN_MLIR_SKIP_EVAL=1`, median of 22 steps:**

| config | copying | resident | residency gain | ms/image (resident) |
|---|---|---|---|---|
| 1 GPU, bs32 | 95 | **52** | 1.83× | 1.63 |
| 2 GPU, bs32×2 | — | **59** | — | 0.92 |
| **4 GPU, bs32×4** | 168 | **76** | **2.21×** | **0.59** |

So **4 GPUs is 2.74× on images/s with residency (68% parallel efficiency), against 2.26× (57%)
copying** — better than R34/ImageNet's 54%, and the reason is structural: ViT-Tiny pushes 63.2 MB of
`[θ|m|v]` where R34 pushes ~255 MB, so the O(N−1) term §2d.3a identifies is far smaller relative to
compute. 73 steps/epoch, loss descends 3.441 → 3.056 → 2.902.

**▶ bs128 per device FITS, and global 512 is where the scaling looks best.** Added the same day:
`vit_adam128` + `vit_adamdp128x4` (the single-device peer is not scaffolding — `vit-dp-check` gates
a DP render against the single-device one *at the same per-device batch*). Gated identically:
**bit-exact 16,579,041/16,579,041** at 4 replicas, control fires at **2.98**.

| config | ms/step | **ms/image** | scaling vs 1 GPU at the same batch |
|---|---|---|---|
| 1 GPU bs32 | 52 | 1.625 | — |
| 4 GPU bs32×4 (global 128) | 76 | 0.594 | 2.74× (68%) |
| 1 GPU bs128 | 217 | 1.695 | — |
| **4 GPU bs128×4 (global 512)** | 264 | **0.516** | **3.29× (82%)** |

Two readings, and the second is the useful one. **On ONE card a bigger batch buys nothing here**
(1.695 vs 1.625 ms/image — slightly worse). **On four it buys 15%**, and parallel efficiency goes
68% → 82%, because the per-step collective is amortised over 4× the images while the `all_reduce`
payload is fixed at the parameter count. That is the same O(N−1)-vs-O(1) argument as §2d.3a, read in
the batch direction instead of the replica direction.

⚠ **What "fits" does and does not rest on.** It ran to completion at bs128×4, which is the
operational answer. It is **not** backed by a memory measurement: `nvidia-smi memory.used` reads
~12,228 MiB/device, but XLA's BFC allocator pre-reserves 11,962 MiB regardless of the net, so that
number is the *pool* and says nothing about headroom. Whether bs256 fits is **untested**.

**⚠ Three things to know before running it long:**

1. **This box stores Imagenette train at 224², not 256² — every Imagenette run here needs
   `LEAN_MLIR_IMAGENETTE_TRAIN=224` or it dies with `uncaught exception: short read`.** `train.bin`
   is 1,425,359,105 bytes = exactly 9469 × 150,529 (224² records). It is a **pre-existing box
   condition, not a DP one** — the single-device `adam` variant fails identically. ⚠ And it is not
   only a loader flag: `crop := (px == 256)`, so at 224 the run trains with **no random crop**, i.e.
   a weaker augmentation than the 71.31% 80-epoch run in §0b. Do not compare accuracy across the two
   without saying so.
2. **Global batch 128 is a DIFFERENT EXPERIMENT from the 71.31% run**, which was global 32. §2d.2
   measured accuracy tracking *step count*: 295 → 73 steps/epoch is two halvings, worth roughly
   −1.5 to −2.5 points at unscaled LR. To reproduce the single-device number on four cards, render
   bs8×4; to measure *scaling*, bs32×4 is the right config. Pick one on purpose.
3. **NCCL logs `ncclCommRegister … unhandled cuda error` at 4 replicas** and the run is correct
   anyway (bit-exact against single-device). It is a memory-registration fast-path failing, not the
   collective. Not diagnosed — worth a look only if 4-GPU throughput needs to improve.

**⚠ And two traps this session walked into, both worth the ink because both LOOK like results:**

* **`LEAN_MLIR_MAX_STEPS=20` at 18 steps/epoch silently became a full 80-EPOCH RUN.** §2d.3-scope
  already documents it (*"keep it under nTrain/(bs·replicas) or the probe never fires and you get an
  80-epoch run"*) and it still bit, because at global 512 the epoch is only 18 steps and 20 looks
  like a small number. It also left an `epoch=80` checkpoint, i.e. §4's silent-no-op trap armed for
  the next run. **At global 512 the ceiling is 18; the probe warms 8, so the usable window is 9-18.**
* **A SKIPPED eval prints `val_acc = 0/3925 = 0.000000%`.** `VerifiedTrain.lean:810` runs the eval
  loop over `[0:0]` under `LEAN_MLIR_SKIP_EVAL`, then prints `correct/total` regardless — so the
  output of a *deliberately* eval-free probe is indistinguishable at a glance from a net that
  collapsed to below chance. Nothing is wrong with the training; the print is. Worth fixing to say
  `skipped` if anyone is in there.

### ▶ What is DONE and needs nothing

The AdamW scorecard is **6 of 6**, DP is **5 of 5**, the provenance axis is closed (67 artifacts,
one writer each), all five nets match their JAX reference param count, and §2d.3 is built, gated
and measured. Details below; none of it is a live thread.

---

## 0a. The AdamW scorecard is **6 of 6**. This thread is CLOSED.

Done 2026-07-28. **cifar8, resnet34, vit, efficientnet, convnext and mobilenetv2** all train on
`pretty(provenGraph)`, each swap licensed by a numeric tie that was verified to fail, and the writer
audit reports one writer per artifact. There is no "next AdamW render".

**▶ §2o Part A is ✅ DONE (2026-07-31) — the channel-LN backward capstone landed, and it covered
the op, block body, residual block AND the downsample §2n dropped.** Six declarations in
`ConvNeXtBackB0.lean`, all 3-axiom, audit 1503 → 1509, build 3,910 green, `verified_mlir/` 0 lines
of diff. The gap was **wider than §2n's commit message said** — the channel-LN net had no `den`-level
backward capstone at ANY level, because the survivors in `ConvNeXtBackB0` are over the ch9 scalar-LN
net. Two traps it hit are written up in §2o (a stale-olean one and a `simp`-won't-fire one).

**▶ §2o Part B is ✅ DONE (2026-07-31) — and with it §2o, and the last two things ConvNeXt owed.**
The fresh 80-epoch channel-LN re-run scores **84.41% final / 84.82% best in 1h54m** (`rc=0`, epoch
marker 80), replacing the VOID 82.75%; §0b's table is updated. The DP ratio re-measures at
**1.70×** (78.0 s → 46.0 s marginal, train-only), within noise of the pre-flip 1.68×, so the
channel LN costs nothing at the scaling level. **Nothing is owed on ConvNeXt.**

**▶ §2d.3 — DEVICE-RESIDENT PARAMETERS — IS ✅ BUILT, GATED, MEASURED AND COMMITTED
(2026-08-01).** The last structural item in the file, done in one session rather than the scoped
3-4. Opt-in behind `PJRT_FFI_RESIDENT=1`; the `[θ|m|v]` blob stops crossing PCIe. Measured on ares
(RTX 4060 Ti — **not** the 7900 XTX the rest of §2d.3 was measured on):
**cifar8-bn 3.1× · ResNet-34 bs32 2.03× · EfficientNet 1.62× · R34 2-GPU 2.09×**, and on
R34/ImageNet at a fixed global batch of 256, **4-GPU parallel efficiency 38% → 54%** (596 → 386
ms/step), i.e. a 30-epoch verified ImageNet run goes **~25 h → ~16 h**. Gated bit-identical on 1,
2 and 4 GPUs with the fault control firing on the resident path itself.

Four of §2d.3's five predictions held; the fifth — *"DP efficiency barely moves"* — is **refuted at
4 replicas** (1.52 → 2.16×) for the reason §2d.3a had already written down one section up: the push
is O(N−1) against O(1) compute. Two new instrument findings are in §2d.3, and one is a live trap:
**§2d.3's Finding 1 ("the floor IS bit-exact across processes") is ROCm-specific** — on CUDA the
gate saturates completely, to the point where a deliberate 1-ULP fault scored *cleaner than the
noise*, and `scripts/det_shim.sh` exists now because of it.

**▶ `VerifiedNet.train` IS CONVERTED TOO** — and it is where the gains are largest, because that
loop reads *nothing* out of a step, so every parameter is resident and the round trip goes to a
literal zero: **MNIST CNN 6.49× · MNIST MLP 4.39× · cifar8-bn SGD 2.03× · MNIST linear 1.21×**. `lake run
mnist-xla` is the headline, and `trainLinear` came over with it, so **every training loop behind an
`-xla` target is converted**; `scripts/residency_gate_all.sh` gates all seven in one command. A second generalisation broke on the way: **§2d.3's Finding 2 ("the system is
chaotic, bit-identity or nothing") is R34/AdamW-specific** — MLP+SGD *absorbs* a 1-ULP fault, which
left that gate with no working control until `PJRT_FFI_FAULT=2` (drop a step's retained params) was
added. Left to do: `trainLinear`, and a long run.

**▶ §2n IS CLOSED (2026-07-31) — bridges discharged, §B tied, scalar chain DROPPED (−852 lines).**
Gates: build **3,910** green · `verified_mlir/` 0 lines of diff after a real re-render ·
`regen_verified_mlir.sh check` 67/one-writer · audit **1506/1506** · parity smoke `REF RUNS ✓`.
Four things the drop taught are in §2n, and one is a **live trap worth knowing before you touch any
renderer**: `lake build Codegen` does NOT build `LeanMlir/Proofs/Codegen/*.lean` (different lean_lib
roots), so a green build + empty artifact diff after editing a renderer can be **vacuous** — the
`#eval` writers never ran. Force them with `lake env lean` on the render file.
~~⚠ Two things ConvNeXt still owes~~ ✅ **both CLOSED by §2o Part B (2026-07-31)**: the 82.75% is
replaced by a measured **84.41%** (best 84.82%, 1h54m) and the DP ratio re-measures at **1.70×**.
Below is the §2n history.

**▶ §2n STEP 1 (2026-07-31) — the FloatBridges are discharged.** The channel-LN net now has a whole-net float story in both directions
(`convnextCh_floatBridges` / `convnextCh_grad_floatBridges`), tied to the committed
`convNextForwardTCh` by `convNextForwardTCh_eq_skeleton`, on a new
`Proofs/Float/ChannelLNFloatBridge.lean`. Build **3,910** jobs green, `verified_mlir/` 0 lines of
diff, audit **1509/1509** three-axiom. The keystone paid off exactly as scoped: the block bridge is
now one theorem over `cnxBodyWith` and the ch9 net instantiates it as a *term*.

**Both of step 1's leftovers are also closed (2026-07-31).** The **§B tie** landed —
`chanLNTensor3Back_eq_chanLN_vjp` + the body/block peers, so the channel-LN net's §B coverage now
matches the scalar net's and the float backward is tied to the CERTIFIED gradient (it is also
**β-free**, proved not assumed). And the **third stale-scope file** the §2n table missed —
`ConvNeXtTiePoC.lean`'s §1a whole-net tie, which is scalar-LN *with a head LN*, i.e. `chLN := false`
— now carries an explicit scope label; porting it was measured as chapter-sized, so it is labelled,
not ported. `ConvNeXtRender.lean`'s matching stale docstring (*"`chLN := false` is the committed …"*)
is fixed.

**▶ The original §2n plan — discharge ConvNeXt's FloatBridges, then DROP the scalar chain.**
§2m is CLOSED as of 2026-07-31: **all five nets now match the JAX reference param count exactly**
(R34 21,797,672 · ViT 5,717,416 · mnv2 3,504,872 · enet 5,288,548 · **ConvNeXt 28,587,592**), the
last one by landing ConvNeXt's REAL channel LayerNorm — math forward, whole-net VJP, rung E, the
spec flip, both 2-GPU gates and the render capstone back at 180/180 bit-identical parity.
Nothing is broken and the build is green at 3,909 jobs; what is left is that the **float story
and two smaller files still describe the scalar-LN net the repo stopped shipping**. §2n has the
scope, the measured dependency table, the verified keystone (`convNextBlockBody = cnxBodyWith
(layerNormForward …)` by `rfl`, so ONE generic bridge serves both worlds) and the gates.
~~⚠ Two things are OWED on ConvNeXt regardless~~ ✅ **both CLOSED by §2o Part B (2026-07-31)** —
**84.41%** final / 84.82% best / 1h54m, and the scaling ratio re-measured at **1.70×**.

**▶ §2l IS DONE (2026-07-31) — the render is the paper's ResNet-34 and the param counts match the
JAX reference at 21,797,672.** ▶ Then: the **80-epoch re-run** to restore the headline (done),
then the `mom` numeric gate. Read §2l for the two things its own plan got wrong. Original note: §2k found that this repo's "ResNet-34" is **not** He et al.'s: the downsample projection
is **3×3 strided** where the paper's option-B shortcut is **1×1**, which is +1,376,256 params
(+6.3%), documented only in a codegen docstring while the spec blurb says *"Real ResNet-34"*. It was
caught by the ImageNet reference-pairing attempt, where the two paths' param counts disagreed by
exactly that plus the conv biases. **§2l has the plan, the blast radius (13 artifacts, the 90.39%
run, §2d.2) and the ten-minute check to run first.** Start there, not here.

**What is left in this file is §2d's value-ordered list**, none of it on the AdamW track: rung 4
(the FPN detector), **device-resident parameters** — whose case is now **measured, not estimated**
(§2d.3, 2026-07-30: the parameter round trip is **55% of a bs32 step**, projected **~2.2×**, and the
cost is per-buffer rather than per-byte) — and the executable cache. Read §2d before picking one.

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

**ViT's ledger is now CLOSED**: 80-epoch run done 2026-07-30 (**val 71.31%**, best 71.62%, 58m11s —
§0b's table), DP gated bit-exact with a firing control, and a measured 2-GPU scaling ratio
(**1.62×** at bs32, 1.59× at bs64). Nothing is owed on any of the five nets. See §2j's tail.

After these, §2d's value-ordered list: rung 4 (the FPN detector) or device-resident parameters
(four measurements behind it; a calibrated model says the case roughly quadruples at 4 GPUs —
§2d.3) — plus the ViT follow-ups the line above unlocks.

Read §2i before starting — it has the measurements, not guesses.

~~Also queued: port `convnext-shard-check` to the enet/mnv2 DP gates~~ ✅ **DONE 2026-07-30 —
`lake build shard-check`, ONE harness for all three nets** (§5). Still queued: try relinking `resnet34-adam-tie`/`cifar8-adam-tie` to `ireeLink` — the four
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
   | **ResNet-34** *(re-run 2026-07-31 at the PAPER net — §2l)* | **90.06%** | 90.06% | **1h03m** |
   | **EfficientNet-B0** | **88.20%** | 88.46% | 1h34m |
   | **MobileNetV2** | **86.73%** | 86.96% | 1h25m |
   | **ConvNeXt-T** *(re-run 2026-07-31 at the CHANNEL-LN net — §2o Part B)* | **84.41%** | 84.82% (ep 69) | **1h54m** |
   | **ViT-Tiny** *(2026-07-30)* | **71.31%** | 71.62% | **58m11s** |

   **All five nets now have an 80-epoch run on their certified bytes.** ViT was added 2026-07-30
   (`runs/vit_xla_80ep_jul30.log`, epoch marker 80, `rc=0`) once its graph would execute here at
   all — same config as the other four (1 GPU, bs32, 80 ep), so the column is apples-to-apples.
   **ViT is the lowest by a wide margin and that is the expected result, not a defect**: a ViT with
   no pretraining, on 9,469 images, with label smoothing as its only strong regulariser, is the
   most data-hungry architecture in the set against the smallest dataset in the book. Read the
   whole ordering as a dataset-size story — it is ConvNeXt's overfitting note one step further.
   It is also the **fastest** of the five, which is the `[[rocm-is-the-transformer-box]]` result:
   gfx1100 is GEMM-strong and MIOpen-conv-weak.

   ⚠ Its wall clock did double duty: the ch10 `benchmark-xla` reference was **3480 s extrapolated**
   from a 43.5 s marginal epoch × 80, and the real run came in at **3491 s — 0.3% out**. So
   `scripts/marginal_epoch.sh` × epochs is validated at this scale, which matters because the
   ch6-9 rows and every DP ratio in this file rest on that method.

   All `rc=0` with the epoch marker on 80 (so genuinely to the end, not a resumed no-op), scored
   through their `Proofs/`-rendered eval forwards. Logs: `runs/<net>_xla_80ep_jul29.log` (untracked).

   ⚠ **The old "ConvNeXt is the outlier" reading is RETIRED** (2026-07-31, §2o Part B). It rested
   on the 82.75% of the scalar-LN net. At the real channel LayerNorm ConvNeXt scores **84.41%**
   final / **84.82%** best — **+1.66 / +1.84 points**, moving it from last of the four CNNs to
   third, ahead of nothing it was behind before only in the sense that ViT was always below it.
   The *overfitting* diagnosis survives and is the reason the gain is modest: train loss still
   lands at **0.502** (the same floor the scalar net reached) with ~28M params against 9,469
   images, and the val curve peaks at **epoch 69** and is flat over the last ten. What the channel
   LN bought is a better-conditioned optimisation, not more data — it is much faster early
   (**69.66% at epoch 7**, where the scalar-LN run was at 42.88% by epoch 12) and then hits the
   same dataset-size ceiling. Read the whole ordering as a dataset-size story, not a ranking of
   the renders.

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
| **ConvNeXt** | ✅ (all 180) | ✅ **§2h-quater** — exact identity on the real net, 2 GPUs, **1.70×** (re-measured on the channel-LN bytes, §2o) | ✅ **§2h** — 84.5 s/epoch | ✅ `Proofs/` (§2g) | ✅ **80 epochs → val 84.41%** (best 84.82%, 1h54m — §2o Part B) |
| **ViT** | ✅ | ✅ **§2j tail** — `vit-dp-check` BIT-EXACT 16,579,041/16,579,041, control 0.996; **1.62×** on 2 GPUs | ✅ **RUNS 2026-07-30** — 128 ms/step, 43.0 s/epoch | ✅ `Proofs/` | ✅ **80 epochs → val 71.31%** (best 71.62%, 58m11s) |

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

# the other -xla trainers. ⚠ This line used to say "vit BUILDS but does NOT RUN on this box" and
# that is STALE — §2j's tail records ViT executing on XLA with NO workaround (128 ms/step, 9.2x
# IREE) and §0b's table has its 80-epoch run at 71.31%. **All five nets run on PJRT, at both
# scales**: `<net>-verified-adam-xla` (Imagenette) and `<net>-imagenet-verified-xla` (ImageNet),
# for r34 / vit / convnext / efficientnet / mobilenetv2 — plus cifar8's six and mnist's three.
# ⚠ The underlying MIOpen fault was NON-DETERMINISTIC rather than fixed (it fired once and not
# again in 11 runs), so a recurrence is possible; it is a ROCm-box hazard, not a ViT one.
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

# ── device-resident parameters (§2d.3, 2026-08-01). Opt-in; the copying path is the default
#    and is byte-identical to pre-change, so nothing else in this file is affected by it.
PJRT_FFI_RESIDENT=1 .lake/build/bin/resnet34-verified-adam-xla data     # 2.0x at bs32
scripts/residency_gate_all.sh           # ALL seven converted loops, bit-identity, one command
scripts/eval_residency_gate.sh          # the eval forward's hold mode — a DIFFERENT property
scripts/det_shim.sh /tmp/detshim        # ⚠ both gates need this on CUDA, or the floor is noise
#   ^ eval_residency_gate needs GATE_CKPTS for any net that checkpoints (trainAdamSched does):
#     GATE_CKPTS='.lake/build/cifar8*_ckpt_xla.bin*' scripts/eval_residency_gate.sh <bins...>

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

# MIXUP / CUTMIX — the producer half of wire v2 (`recipe_gaps.md` v1.3). Shim-side Python only;
# no render, no op, no Lean change. `SHIM_MIX` reaches the shim child because IO.Process.spawn's
# env array EXTENDS the inherited environment (checked, not assumed).
scripts/gen_shims.sh                      # regenerate ALL FIVE after touching generateShim
scripts/mixup_gate.py                     # inert-when-off, the split gate, determinism, known answer
scripts/mixup_gate.py --break             # + the two negative controls

# THE PER-NET SHIMS (§0.9). `VerifiedNet.shimScript` names each net's own; there is NO fallback,
# and `spawnShim` prints the script it resolved. ⚠ ViT's and ConvNeXt's bake SHIM_MIX=both, so the
# driver passes `off` at wire v1 and says so — SHIM_SOFT=1 is what turns their mixing back on.
scripts/gen_shims.sh                      # the ONE writer of the five files
scripts/shim_wiring_gate.py               # gates 0-3 + the definition-vs-call-site control: instant
scripts/shim_wiring_gate.py --stream      # + SHIM_HASH: mnv2 ≡ R34, the other three ≠ R34, val equal
scripts/shim_wiring_gate.py --break       # + the pre-fix wiring and a ViT↔ConvNeXt swap, both red
#   ^ to stream mixed batches into a trainer: SHIM_SOFT=1 (wire v2) and SHIM_MIX=both.
#     ⚠ SHIM_MIX without SHIM_NCLASSES is REFUSED on the TRAIN split — a mixed label is a
#       distribution. The VALIDATION split ignores SHIM_MIX entirely (gate 1b): the driver spawns
#       it at wire v1, it inherits the variable, and mixing eval data is not the metric anyway.
#     ⚠ the step cap for trainAdamSched is LEAN_MLIR_G2_STEPS, NOT LEAN_MLIR_MAX_STEPS (which
#       means "time a step window then exit"), and the 28 GB ImageNet val drain is UNCONDITIONAL
#       — LEAN_MLIR_SKIP_EVAL does not skip it. Budget ~3-4 min per ImageNet smoke config.

# wdExcludeNormBias — timm/DeiT no_weight_decay on ViT (`recipe_gaps.md` v1.4). No new op, no
# interface change, no driver change: `adamWParamF` takes `wd` as a runtime OPERAND NAME, so
# "exclude" binds a zero constant at 126 of 200 sites.
lake build wdx-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/wdx-tie {vit|convnext}
#   ^ the controls — the partition, not the count, is what is gated:
#     python3 scripts/perturb_wd_mask.py verified_mlir/vit_adamwx_train_step.mlir /tmp/w.mlir swap1
#     .lake/build/bin/wdx-tie --cand /tmp/w.mlir          # rc=1, fires on exactly 2 params
#     (`swap1` keeps the counts at 74/126 — a count-only gate passes it. `invert` is the blunt one.)

# GRAD CLIP — global-norm clipping (`recipe_gaps.md` v1.4b, `planning/grad_clip.md`). Two SHlo ops,
# no driver change. ⚠⚠ RUN IT UNDER THE DET SHIM — without it ConvNeXt's gate 4 reads 137,229 of
# 83,478,847 outputs differing at a 24,149-ULP floor carried by ONE parameter (`d0W`), the number
# MOVES between runs, and gate 1's spread sits BELOW that floor. ViT is clean either way, which is
# exactly how the trap stays hidden.
  lake build clip-tie
  scripts/det_shim.sh /tmp/detshim
  python3 scripts/perturb_clip.py verified_mlir/vit_adamclip_train_step.mlir \
    .lake/build/clip_hi_vit.mlir hi        # gate 4's vehicle — GENERATED, never committed
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie vit
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie convnext
#   ^ the gate is the CONSTANCY of `m'_clip/m'_adam` across params (1.12 / 1.15 ULPs), because a
#     per-parameter clip gets nothing else wrong — it scales, never amplifies, and is the identity
#     below the threshold, so it passes gates 0, 3 and 4. Its control fires at 7.6M / 30.4M ULPs:
#     python3 scripts/perturb_clip.py verified_mlir/vit_adamclip_train_step.mlir /tmp/c.mlir perparam
#     LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie vit --cand /tmp/c.mlir    # rc=1
#     (also `nosqrt` and `epsout`; all six control runs are red across the two nets.)

# STOCHASTIC DEPTH — the two gates covering the op's INTERIOR (§0's tail, `stochastic_depth.md`
# §7a-c). Every other SD gate pins an ENDPOINT, and an identity gate at s = 1 cannot see WHERE s is
# applied — measured: the ones-mask gate passes bit-exact on a deliberately misplaced render.
lake build droppath-tie
scripts/det_shim.sh /tmp/detshim          # ⚠ REQUIRED: gate B compares two HLO programs
LD_LIBRARY_PATH=/tmp/detshim CUDA_VISIBLE_DEVICES=0 .lake/build/bin/droppath-tie
.lake/build/bin/droppath-tie --op --break            # gate A falsifiable: a 1% wrong scale
.lake/build/bin/droppath-tie --net --eval            # at @efficientnet_drop_fwd_eval
#   ^ the control that licenses gate B — move all 9 sites onto the BLOCK OUTPUT (2 lines per site,
#     same SSA names/order/types/line count, so no structural check moves) and it goes red, rc=1:
#     python3 scripts/misplace_drop_sites.py verified_mlir/efficientnet_drop_fwd.mlir /tmp/mis.mlir
#     .lake/build/bin/droppath-tie --net --cand /tmp/mis.mlir

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

Env knobs added by this work: `PJRT_REPLICAS`, `PJRT_PLUGIN`, `PJRT_FFI_TRACE`, `PJRT_FFI_TIMING`
(§2d.3 — per-phase transfer accounting, opt-in and inert when unset), `PJRT_FFI_PINNED`
(**vendor-dependent — see §2d.3a. A 17% regression on ROCm, a ~17% WIN on CUDA.** It is also the
second transport `scripts/residency_gate.sh` validates against), `PJRT_FFI_FAULT`
(§2d.3 — 1-ULP fault injection, the phase-3 gate's transport control; on the resident path it hits
the parameter SEED, since there is no returned float left to flip; **`=2` is a STALENESS fault** —
drop one step's retained parameters — because a 1-ULP one is *absorbed* by nets whose updates
contract, e.g. MLP+SGD),
**`PJRT_FFI_RESIDENT`** (§2d.3, 2026-08-01 — **device-resident `[θ|m|v]`**: 2.0× on R34 bs32, 3.1×
on cifar8-bn, and 4-GPU parallel efficiency 38% → 54%. Opt-in, and deliberately so — the copying
path stays the default and is byte-identical to pre-change, so every existing gate is untouched),
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

##### ✅ 2h-cuda. SETTLED 2026-08-01 — it IS MIOpen. The probe ran clean on CUDA.

The section above says the CUDA box settles this cheaply and names the exact probe. It did.
`PJRT_PLUGIN=<jax cuda plugin> sgd-render-tie vit verified_mlir/vit_train_step.mlir 0.003125 ×2`
on ares (RTX 4060 Ti, CUDA 12.9, jax 0.10.2 plugin):

```
compiled verified_mlir/vit_train_step.mlir (@vit_train_step, 200 outputs, 1 replica) in 4949 ms
  A-vs-A floor: max|a−a'| = 0.000000, bit-exact 5526346/5526346
  5526346 gradient coordinates; 5526254 non-zero; max|g| = 1503.883514
  params whose gradient disagrees (>1e-4 norm-rel): 0/200
✓ renders TIE
```

**No workaround, no MIOPEN_* variable, no failure.** The graph that dies on ROCm with
`miopenStatusUnknownError` — same file, same two convolutions, same backward — compiles in 4.9 s
and produces a live gradient on NVIDIA. Every remaining hypothesis that blamed the *render* is
therefore dead; the fault is in the XLA→MIOpen lowering, exactly where the error surface pointed.

That clears §2a's bar: the open item converts from "unexplained" into a **filable ROCm bug with a
minimal repro** — `vit_train_step.mlir` is 202 in / 200 out with two convolutions, and it is now
known-good on another backend, which is the control a bug report wants.

⚠ Scope it honestly: this shows the graph is fine and CUDA runs it. It does NOT identify which
MIOpen solver is picked on the AMD side — the `rhs_dilation = 16` hypothesis above is still a
hypothesis, and cutting the repro down to the single offending convolution is still unstarted.

**Second, unplanned finding:** the A-vs-A determinism floor is **bit-exact over 5,526,346 floats**
here. §2d.3's Finding 1 established that floor on R34/ROCm and flagged it as "one net, 10 steps" —
it now also holds on a different vendor, a different net, and a graph with two convolutions. That
strengthens the case that `scripts/residency_gate.sh`'s bit-identity gate is buildable as written.

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
* ~~**ViT on XLA anywhere.** Try the CUDA box with `sgd-render-tie vit` first (above).~~
  ✅ **DONE 2026-08-01 — the probe ran on ares and it is a ROCm/MIOpen fault.** See §2h-cuda below.

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

✅ **RE-MEASURED 2026-07-31 on the channel-LN bytes (§2o Part B): 78.0 s → 46.0 s = 1.70×**, within
noise of the 1.68× below. The numbers in this table are the pre-§2m ones; the conclusion is unchanged.

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

**Nothing is owed on ViT any more.** The 80-epoch run landed 2026-07-30 — **val 71.31%**, best
71.62%, wall **3491 s**, epoch marker 80 (`runs/vit_xla_80ep_jul30.log`) — so ch.10's `benchmark-xla`
reference is a MEASUREMENT, not the 43.5 s × 80 extrapolation it replaced. Those two agree to
**0.3%** (3480 vs 3491), which validates `marginal_epoch.sh` × epochs at this scale and therefore
the ch6-9 rows and every DP ratio in this file. Throughput at bs64 and the 2-GPU ratios: §2j tail.

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

### 2k. ▶ R34 heavy-ball momentum — the optimizer the ImageNet reference actually uses

**Rendered 2026-07-30.** `verified_mlir/resnet34_mom_train_step.mlir`, selected by
`LEAN_MLIR_VARIANT=mom`. Purpose: make the verified/XLA path and `jax/MainResnetImagenet.lean`
runnable as a **matched pair** rather than as two different experiments — the two differed on
optimizer, augmentation and precision, so no number from one said anything about the other.

#### ⚠ The trap, and it would have been silent

The JAX reference (`Jax/Codegen.lean`, `.sgd` branch at `hasMomentum`) is:

```python
grads    = g + WD * p          # COUPLED L2, wd = 1e-4, every param (no wdExclude)
velocity = MOMENTUM * v + g    # μ = 0.9
params   = p - lr * velocity   # HEAVY-BALL
```

The repo's existing momentum op is **Nesterov** — `momParam μ lr θ v g = θ − lr·(g + μ·v')`. Reaching
for `momParamF`, which is what "add the momentum variant" obviously means, would have compiled,
rendered, trained, descended, and produced **a different optimizer than the one it exists to match**.
`Proofs.momParam_heavyBall_diff` states the exact difference and is what settled it.
**Check which momentum a reference uses before rendering one.**

#### Zero new `SHlo` ops — the whole rule composes from the existing family

Worth knowing because "adding an op touches ten sites" (§4) is normally the cost of this kind of job:

| step | op | why it works |
|---|---|---|
| `g + wd·θ` | **`momVNextF`** at `(μ := wd, v := θ)` | `momVNext μ v g = μ·v + g`, so this denotes `wd·θ + g` — the *same function*, just a different reading of the two slots. Faithfulness carries over untouched |
| `v' = μ·v + g` | `momVNextF` | as intended |
| `θ' = θ − lr·v'` | **`sgdParamF`** applied to `v'` | `sgdParam lr θ g = θ − lr·g`; feed it the velocity and it is heavy-ball |

So no constructor, no `den`, no `rfl` theorem, no `Raw`/`skel`/`Tok`/`toToks`/`emitTok`, and — the one
that actually bites — **no `StableHLOParse` roundtrip case**.

#### Gates: two held, one is OWED

* ✅ **Gate 1 — all six `adam*` artifacts re-render BYTE-IDENTICAL** through the `opt` threading
  (`md5sum -c`, and `verified_mlir/` stays git-clean). This is the strong form §2f could not use, and
  it is what says the threading is inert.
* ✅ **Interface** — the `func.func` signature is **byte-identical to `adam`'s** apart from the entry
  name: 515 in, 513 out, same names, types and order. `m` is a passthrough (the `CnnRender.optTail`
  `.sgd` convention), so the packed `[θ|m|v]` protocol and the driver do not move. And the
  forward+backward **body is byte-identical over all 4409 lines** — only the banner and the tail
  differ, which is the structural version of the same claim.
* ✅ **Descent, on the real net and real data**: loss **2.369 → 1.814 → 1.610** over 3 epochs, val
  38.7 → 33.1 → **57.5%** (val bounces because LR is still warming: 0.033 → 0.067 → 0.100).
* ✅ **CERTIFIED 2026-07-31 — `lake build r34-mom-tie`** (`c8cb9e0`). The known-answer check
  below, run on the post-§2l net: **① `v' = g + wd·θ` rel 7.3e-8, ② `θ' = θ − lr·v'` rel 5.8e-8,
  ③ `m` passthrough bit-exact 21,289,802/21,289,802.** The controls are what make it a gate: the
  **NESTEROV** prediction (`θ − lr·(1+μ)·v'`, = 1.9× the step at `v = 0`) misses by **2.99e-2,
  515,403× the tie**, and dropping `wd·θ` misses by 2.93e-4, 4,034× the tie. So the harness
  demonstrably separates the two optimizers — which is the whole risk §2k identified, since
  `momParamF` is Nesterov and is what "add the momentum variant" reaches for. *(Original note: the
  exact known-answer check is available and cheap: from `m = v = 0` the `adam` render's
  `m' = (1−β₁)·g` recovers the gradient exactly (`g = 10·m'`), so on shared `(θ, x, onehot)` the
  momentum render must satisfy `v' = g + wd·θ` and `θ' = θ − lr·v'` — a cross-render known answer,
  not a tolerance argument. It is the `shard-check` construction reused.)*

#### A driver knob it needed, and why it is not optional

`LEAN_MLIR_BASE_LR_U` — base LR in **micro-units** (`100000` = 0.1), integer-encoded because this
toolchain has no `String.toFloat?` (the `LEAN_MLIR_PERTURB_R` dodge, 1e-9 there). Default 0.001 is
unchanged so every existing run is unaffected.

It is effectively **required** for `mom`: 0.001 is an *AdamW* rate, the reference uses **0.1**, and
running the momentum render at the Adam default under-steps by ~100× — which looks exactly like a
broken render rather than a wrong knob.

#### ✅ The tfds batch shim — landed 2026-07-30, and determinism was NOT free

`JaxCodegen.generateShim` → `scripts/gen_shims.sh` → one
`jax/.lake/build/generated_*_imagenet_shim.py` **per net** (§0.9 — it was R34's for every net until
2026-08-02). Streams the **same** tfds pipeline the JAX
reference trainer consumes to stdout, so the verified XLA trainer can eat identical bytes.

**It is GENERATED, from the same `TrainConfig` the recipe trains on — that is the whole point.** The
augmentation *is* the pipeline (Inception RandomResizedCrop area 8-100% / aspect 3/4-4/3 / bicubic,
hflip, ImageNet mean-std). A hand-written `imagenet_stream.py` would be a **second definition** of
"how an ImageNet image becomes a tensor" — agreeing today, drifting the first time one side is tuned.
That is §2a's double-writer disease one level up, and `resnet34_train_step`'s md5 flipping between two
writers computing genuinely different functions is the precedent. Reusing `emitDataLoading .imagenet`
verbatim means it **cannot drift by construction**. Bonus: switching on RandAugment / AutoAugment /
random-erasing in the config moves both paths at once, and the verified side owns no aug code at all.

**Wire format** — 16-byte preamble once (`LMSH` | version | batch | flat_len), then per batch
`int32[batch]` labels ++ `float32[batch*flat_len]` images. The preamble is the shim's G4: a shape
mismatch fails loudly at the reader instead of silently misaligning every later batch.

**f32 on the wire, deliberately** — reversing an earlier note in this file that said uint8. The
pipeline already normalizes and flattens to `(B, 3·224·224)` f32, exactly the train step's `%x`;
sending uint8 is 4× cheaper but moves normalization to the Lean side, i.e. a second writer for part
of the transform. ~154 MB/batch against a ~670 ms step is ~230 MB/s and a pipe does GB/s, so the
bandwidth is not worth the seam.

**▶ DETERMINISM TOOK THREE THINGS, AND THE FIRST TWO WERE NOT ENOUGH.** Recorded because the
"obvious" version is silently wrong:

| attempt | result |
|---|---|
| `tf.random.set_seed(seed)` alone | ❌ **two same-seed runs hash DIFFERENTLY** (`6c005967…` vs `3b62deda…`) — the crop/flip ops draw from TF's global state and `num_parallel_calls=AUTOTUNE` varies the draw order |
| + `enable_op_determinism()` | ❌ TF **refuses to run**: *"sample_distorted_bounding_box requires a non-zero seed when determinism is enabled"* |
| + op-level `seed=_AUG_SEED` | ✅ three runs, one hash: `58e972d1…` |

`_AUG_SEED` reads `$AUG_SEED` **at module scope** (so the shim must install the default *before*
importing the block, not in `_main` — an ordering bug that costs one confusing run). Unset ⇒ `None`
⇒ the pre-existing stateful behaviour, so **the reference trainers are byte-for-byte unaffected by
default**; only the shim turns it on.

**Verified to fail**: seeds 0/1/2 give three *distinct* hashes, so the determinism is not vacuous —
without that control, a pipeline that ignored its seed entirely would look identical to a
correctly-seeded one. The validation split is stable across runs too (no randomness there).

**Throughput: 1679 img/s** with determinism on, marginal `(t₂₀−t₄)/16` at bs256 — and that *includes*
sha256 over 154 MB/batch, so it is a lower bound. The GPU needs ~380 img/s to keep a 673 ms bs256
step fed, so there is 4.4× of headroom; `enable_op_determinism` costs little against the reference
run's 1840 img/s. **The decode-throughput worry this idea was gated on is settled, measured.**

```bash
scripts/gen_shims.sh                              # emit all five (each from its recipe's own cfg)
SHIM_BATCH=64 SHIM_HASH=3 python3 jax/.lake/build/generated_resnet34_imagenet_shim.py
#   ^ hash N batches to stderr and exit. RUN THIS TWICE after touching the pipeline —
#     it is the whole determinism gate, and it is two commands.
```

#### ✅ The Lean-side reader — landed 2026-07-30, and VERIFIED against the producer

`VerifiedData.imagenet` + `spawnShim` / `readShimBatch` / `readExact` in `VerifiedTrain.lean`.

**Split strategy, and it is what keeps the change small:** the **train** split streams (1.28M images
= ~938 GiB at f32, so preloading is not an option), while the **val** split is drained into RAM once
(195 batches × 256 after tfds `drop_remainder` = **49,920 images, 30 GB**, which fits in the box's
175 GB). So the eval loop is **completely unchanged** — and 49,920 is the same count
`jax/runs/r34_imagenet_bf16_90ep/RESULTS.md` reports, so both paths score the identical set.

**The Lean side does NO augmentation for `.imagenet`.** The batch arrives already RRC'd, flipped,
normalized and flattened, so the `.imagenet` arm bypasses both the slice and the aug `match`. One
definition of the transform, and it lives in the generated shim.

Three things the reader has to get right, all of which fail silently if skipped:

* **`readExact` loops.** A pipe read returns what is *available*, not what was asked for — at 154 MB
  per batch a short read is the normal case, and treating one `read` as a batch misaligns the stream
  from then on. It throws with a pointer to `SHIM_HASH` if the pipe closes early.
* **The preamble is CHECKED, not skipped** (`LMSH` | version | batch | flat). A batch or resolution
  mismatch between render and shim would otherwise read as garbage pixels and look like a broken
  net. Same reasoning as the FFI's G4 arity guard.
* **The stream is spawned ONCE, not per epoch**, and the per-epoch `F32.shuffle` is skipped for it —
  tf.data's `.shuffle(seed=42, reshuffle_each_iteration=True).repeat()` already re-shuffles across
  the epoch boundary, and there is no resident array to shuffle anyway.

**Verified against the producer, which is the part that makes it evidence rather than "it ran":**
Lean reads batch 0 labels `[26, 948, 227, 24, 582, 614, 141, 155]` and batch 1
`[15, 34, 141, 671, 268, 988, 658, 350]`; driving `build_imagenet_iter` directly from Python yields
**exactly those**, at `(8, 150528) float32`. Byte counts are exact on both batches
(4,816,896 = 8 × 150,528 × 4), so the framing is right and two consecutive records stay aligned.

⚠ `jax/` is its own lake project, so `--shim` writes under **its** build dir; `spawnShim` looks in
`jax/.lake/build/` then `.lake/build/`, and `$SHIM_SCRIPT` overrides. `$SHIM_PYTHON` picks the venv.

#### ✅ The ImageNet net — renders, spec and driver, 2026-07-30. IT RUNS.

`lake build resnet34-imagenet-verified-xla`. Three artifacts under slug **`resnet34in`**
(`mom256_train_step`, `fwd`, `fwd_eval`) at `B := 256, nClasses := 1000`, heavy-ball;
`resnet34ImagenetVerified` in `VerifiedNets.lean`; `Resnet34ImagenetCommon` + a `-xla` main.

**No renderer change was needed** — `B`, `nClasses`, `opt` and `slug` are all parameters, so this is
three `#eval`s. (`slug` was added here, defaulting to `"resnet34"`; all nine existing artifacts
re-render byte-identical.)

⚠ **The slug is load-bearing.** Forward artifacts carry no variant in their path
(`<slug>_fwd.mlir`), so a 1000-class forward under the `resnet34` slug would have silently
**overwritten** the 10-class Imagenette one that five committed runs and the prefix audit depend on.

**End-to-end smoke:** all three compile on XLA, val drains to **49,920 images / 28,665 MB** (the
count the reference scores), the train stream spawns, and it steps —
loss **8.10 → 7.86 → 7.95**, `rc=0`.

#### ⛔ AND IT FOUND A REAL BUG — the label smoothing was hardcoded for K = 10

**`α/K` was the literal `0.010000`** (α = 0.1, K = 10) in **two** places: the report-only `%loss`
*and* — the one that matters — **the COTANGENT** (`.shiftB "-0.010000"`). The docstring one line
above it already said "K = nClasses"; the literal disagreed.

At `nClasses = 1000` that is **100× too large**: the smoothing term removes **10.0** of probability
mass instead of 0.1. It is on the gradient path, so the first ImageNet render was training on a
**different objective**, silently.

**What caught it: the number was implausible.** The first smoke reported loss ≈ **87** where
1000-class CE at init must be ≈ ln(1000) = **6.9**. Nothing else would have — no proof covers it
(`α` is a literal in emitted text, exactly §5's carve-out class), the render compiled, the graph ran,
and the loss *descended*, so every structural check was green. §2b's `%loss` bug is the standing
precedent and this is the same failure one level over: **a descending loss curve is not evidence the
objective is right.**

Fixed: `alphaOverK nClasses` + a `fmt6` fixed-decimal formatter, so the emitted form is identical at
K = 10 (**all nine artifacts byte-identical** — the fix is inert there) and correct elsewhere. After:
loss **8.10 → 7.86 → 7.95**, which is where it belongs.

*The transferable bit: when a net is re-rendered at a new `nClasses`, every emitted constant that
depends on K is suspect. Here there was exactly one, in two places, and only one of them was on a
gradient path.*

#### ⛔ THE STEP-LEVEL TIE CANNOT RUN — the two "ResNet-34"s are DIFFERENT NETS

Attempted 2026-07-30, and it stopped at the first check, which is the cheapest one: **the param
counts disagree.**

| | params |
|---|---|
| verified (`resnet34ImagenetVerified`, from the dumped `[θ|m|v]`) | **23,182,440** |
| JAX reference (`resnet34Imagenet.totalParams`) | **21,797,672** |
| gap | **1,384,768** |

The gap is not noise and it decomposes **exactly**:

| cause | params |
|---|---|
| downsample projection is **3×3 strided** in the render, **1×1** in the JAX spec | 1,376,256 |
| the render carries **conv biases**; `.convBn` does not | 8,512 |
| **total** | **1,384,768** ✓ |

Confirmed in the emitted artifact, not inferred: `%d2Wp: tensor<128x64x3x3xf32>`,
`%d4Wp: tensor<512x256x3x3xf32>`. `ResNet34RenderB.downFwdB` builds the projection from
`zk1 : Kernel4 c cin 3 3` — the *same* kernel as the block's first conv.

**So the repo's "ResNet-34" is not He et al.'s ResNet-34.** The paper's option-B shortcut is a **1×1**
projection; this is 3×3, which is **+6.3% params** over the standard 21.8M. It is **documented
nowhere** — `grep` finds no mention of the shortcut kernel in any `.lean` or `.md`, and the render's
own docstring says only "strided projection skip". Every R34 number in this repo (the 90.39%
Imagenette run included) is for that net, and `resnet34Verified`'s blurb calls it "Real ResNet-34".

**This is precisely the seam §2k's "structural" note predicted**, and it is the argument for the
`#guard` in its sharpest form: `NetSpec` and `VerifiedNetSpec` are two independent descriptions of
"ResNet-34" in two layer vocabularies, nothing forces them to agree, **and they don't**. A
`toSpecs`-vs-`totalParams` guard would have failed at `lake build` the day the ImageNet spec was
written, instead of after the renders, the driver and a smoke run.

It also confirms the reason the tie was worth attempting *before* a 28-hour run: a JAX-vs-verified
accuracy comparison would have been **uninterpretable**, and nothing else in the pipeline would have
said so — both nets train, both descend, both are "ResNet-34".

### 2l. ✅ DONE 2026-07-31 — the render IS the paper's ResNet-34 now

**Landed in three commits: `a1414ef` (the check + the generalization), `e9c2729` (step B, the conv
biases), `08f4800` (step A, the 1×1 projection).** The param counts now **match the JAX reference
exactly at 21,797,672**, which was step 5's gate and the whole point of the detour.

**What the plan below got right:** the ordering (B before A), the blast radius, and that the
1×1-padding check was the thing to run first.

**What it got wrong, both worth keeping:**

1. ⛔ **"the conv-bias gradient is exactly zero, so they stay 0 forever" is FALSE in f32.** The BN
   mean is a rounded sum, so the cancellation leaves a residue ~1e-6 of the conv-weight gradient on
   ~93% of coordinates (two runs disagree on *which* — that is what identifies it as noise). And
   under AdamW's **scale-free** update (§3's own warning) that residue still moves θ by ~lr per
   step: in the 80-epoch checkpoint **all 8,512 conv biases had drifted off zero**, to |θ|max
   0.041. What makes B safe is a claim about the FUNCTION — zeroing all of them moves the trained
   logits by rel **1e-6**, against **0.786** for the same ablation on BN β — not about the
   gradient. `tests/TestConvBiasZero.lean`.
2. ⛔ **"parameterised, so this is arguments not proofs" was true of `rblkPStridedPC` and false of
   everything stated about it.** ~20 declarations over 6 files pinned `Wp : Kernel4 oc ic 3 3`.
   Generalizing them first is what made A cost **15 binder literals and 3 record fields** — and it
   surfaced a hidden **odd-kernel side condition** (`2·((k−1)/2)+1 = k`) that
   `flatConvStride2Back_eq_vjp_backward` was discharging with `by decide` at a pinned 3×3. It is an
   explicit hypothesis now: true at 1 and 3, provably false at 2 (ConvNeXt's exclusion, §2f-bis).

**The bug A produced, and how it was caught:** the AdamW tail carried its own hardcoded
`[c,cin,3,3]` for the projection, so the signature said 1×1 while the optimizer emitted a 3×3
multiply. **`iree-compile` rejected it** — a shape that disagrees with itself cannot run. No gate
found it; the type checker did.

**▶ STILL OWED:**
* ✅ **the 80-epoch re-run is DONE** — 2026-07-31, `runs/r34_1x1_smoke.log`, 80 epochs, marker at
  80, `done`. **val 90.06%** (best 90.06%, epoch 80), **1h03m** / 47.2 s marginal epoch. The old
  **90.39% is retired**: it belonged to the 3×3 net. **−0.33 pp for −6.3% params, and that is
  inside this setup's run-to-run spread** — §3 records three same-seed runs of one binary landing
  at 43.2 / 46.8 / 47.3% epoch-1 val, so do not read the delta as a regression from the shortcut
  change. It is also **8 minutes faster** (1h03m vs 1h11m), which is the param drop showing up
  where you would expect it.
  ⚠ A corroboration worth recording: README's *baseline* (non-verified) table already listed
  ResNet-34 at **21.3M** params — i.e. the baselines were the paper net all along and the VERIFIED
  render was the odd one out at 22.7M. Nobody compared the two columns. That is the same class of
  miss as §2k itself, one table over.
* **§2d.2's five-config batch/step-count study is VOID** and not re-run.
* §2d.1's 1.78×, §2c's 1.46× and §2d.3's transfer numbers **shift** — the `[θ|m|v]` blob goes
  272 MB → ~255 MB. Directionally unchanged; re-measure before quoting.
* `resnet34-adam-tie` against the retired 3×3 render is **meaningless by construction** now.
* ~~the `mom` numeric gate (§2k)~~ ✅ **DONE 2026-07-31** (`r34-mom-tie`, ≤7.3e-8, Nesterov control
  fires at 515,403× the tie), and **the `NetSpec`/`VerifiedNetSpec` `#guard`** — the latter
  turned out to already exist (`resnet34Verified.toSpecs == ResNet34Layout.specs`) and **it FIRED
  on step B**, forcing `VLayer` to grow a bias-free `convBnNB` rather than the hand-list being
  quietly edited to match. It earned its keep on its first real test.

*The plan as written on 2026-07-30 follows, for the record.*

#### (the original plan) make the render the PAPER's ResNet-34 (1×1 projection)

**Brett's call, and it is the expensive option deliberately** — the alternatives were to bend the
JAX spec to match the render, or to keep two nets and give up the oracle. Correctness of the
artifact wins: `resnet34Verified`'s blurb says *"Real ResNet-34"* and it should be true.

**▶ THIS IS THE NEXT SESSION'S JOB. Read §2k's finding above first — it is the why.**

#### The change is two things, and only ONE of them is functional

| | change | functional? |
|---|---|---|
| **A. projection shortcut** | `downFwdB`'s `%{p}Wp` from **3×3 strided** to **1×1 strided** | ✅ **YES** — different net, −1,376,256 params |
| **B. conv biases** | drop `%sbi`, `%{p}b1/b2/bp` (every conv is BN-followed) | ⛔ **NO** — see below, −8,512 params |

**B is inert and that matters for sequencing.** Every conv here is immediately followed by BN, and
BN subtracts the batch mean: `(x + b) − mean(x + b) = x − mean(x)`. So a conv bias cannot affect the
BN output, its gradient is **exactly zero**, and — since biases are zero-initialised (`initKind 2`) —
it stays 0 forever, with `m`/`v` at 0 too. **Dropping the conv biases changes the parameter layout
but NOT the function.** ⚠ That is an argument, not a measurement: verify it in ten seconds by
dumping `[θ|m|v]` after a few steps and checking the bias slots are ~0 before relying on it.

Consequence: **A alone invalidates accuracy numbers; B alone invalidates only layouts.** They can be
landed and gated separately, and B is the one that makes `init_params_from_file` line up.

#### What makes this much cheaper than it looks

Both sides are **already kernel-generic** — this is re-instantiation, not restructuring:

* `SHlo.convStrided {ic oc h w kH kW}` and `convStridedBack {… kH kW}` take the kernel as
  parameters.
* `Proofs.rblkPStridedPC {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp}` — **the proof-side downsample block
  already takes the projection kernel as `kHp kWp`**, instantiated at 3×3.

⚠ **The one thing to check FIRST, before touching anything**: whether the emitter's symmetric-SAME
padding formula can spell `kH = kW = 1`. §2f-bis's ConvNeXt work found that formula *could not*
spell an **even** kernel (2×2) and needed emitter work; 1 is odd, so `pad = (k−1)/2 = 0` should fall
out — but **measure it, do not assume**. Render one 1×1 strided conv and read the emitted
`stablehlo.convolution` padding. That is a ten-minute check that decides whether this is a
half-session or a two-session job.

*(Also: `RenderPC.lean:13,49` DID document "3×3 strided projection skip" all along. The failure was
that the spec blurb and README said "Real ResNet-34" while the deviation lived only in a codegen
docstring. Whatever lands, put the shortcut kernel in `resnet34Verified`'s blurb.)*

#### Blast radius — what this voids, and it is not small

**13 artifacts re-render** (every `resnet34*` and `resnet34in*` in `verified_mlir/`).

| what | status after |
|---|---|
| `ResNet34Layout.specs` (`IreeRuntime.lean:284`) + its `#guard` | **must be rewritten** — 146 params → fewer, and the hand-list is audited |
| the §1a tie, `SpecVJP` witnesses, `ResNet34LiveSeal`, `Resnet34WholeFloatBridge` | **re-instantiate at `kHp = kWp = 1`** — parameterised, so this is arguments not proofs |
| §2b's numeric tie vs the retired hand-written render | **gone** — that emitter is deleted; the tie was against the 3×3 net. Recover with `git show 75a9f8e:` if a comparison is wanted |
| **the 90.39% Imagenette 80-epoch run** | ~~VOID~~ ✅ **RE-RUN 2026-07-31: 90.06%, 1h03m** |
| §2d.2's five-config batch/step-count study | **VOID** — all five are the 3×3 net. Re-run if the finding is still wanted (it is a good finding: accuracy tracks step count) |
| §2d.1 bs256 1.78×, §2c 1.46×, the §2d.3 transfer measurements | **shift slightly** — 6.3% fewer params moves the `[θ\|m\|v]` blob from 272 MB to ~256 MB. Directionally unchanged; re-measure before quoting |
| `resnet34-adam-tie`, `resnet34-batch-check`, the DP artifacts + gates | **re-run**, unchanged in construction |
| README's R34 row | **re-measure** |

#### Suggested order

1. **The 1×1-padding check** above. Ten minutes, and it sizes everything else.
2. **B first (conv biases), alone.** Layout-only, so the *forward is unchanged* — which means you
   can gate it the strongest possible way: the trained net must behave identically. Lands
   `init_params_from_file` compatibility early.
3. **Then A (1×1 projection)**, re-instantiate the proof side, rewrite `ResNet34Layout.specs`,
   re-render all 13, re-run the ties.
4. **Re-run the 80-epoch Imagenette** to restore the headline number, then §2d.2 if wanted.
5. **Then the step-level tie** — the param counts should now match at **21,797,672** on both sides,
   which is itself the first gate and is free to check.

#### Still owed, independent of all this

* **The `mom` numeric gate** (§2k) — the `m = v = 0` ⇒ `g = 10·m'` known answer.
* **The `NetSpec`/`VerifiedNetSpec` `#guard`.** Write it regardless and write it FIRST: it has a
  demonstrated catch, and after this change it is what stops the two specs drifting apart again.
  A `totalParams` equality would have failed at `lake build` and saved this whole detour.

#### What this does NOT do

It does not make R34/ImageNet runnable — that still needs the streaming loader (`VerifiedData` has no
`.imagenet` case, and `loadData` preloads f32 into host RAM: 938 GiB at 1.28M×256²). The momentum
render is deliberately at **bs32 / 10 classes**, the shape the existing gates and the Imagenette
trainer exercise *today*. The ImageNet shape is `B := 256, nClasses := 1000` — both are true renderer
parameters, so it is one more `#eval` and **no renderer change** whenever the loader lands.

Also not done: `momdp` (the DP peer — `r34AdamVariant` already names it, nothing renders it), and
matching the remaining pair axes (precision: reference is bf16 on CUDA, this is fp32 — and §3
measured bf16 at ×0.96 here, so fp32-both is the sane match).

### 2m. ▶ The other four nets, audited the §2k way — 2026-07-31

`§2k`'s check (verified param count vs the JAX reference's own reported count) run on every
remaining net. **Every gap decomposes exactly**, which is what makes these findings rather than
suspicions. References are `jax/.lake/build/generated_*_imagenet.py`, whose headers report the
count the reference itself trains.

| net | verified @1000 | reference | gap | cause |
|---|---|---|---|---|
| **ResNet-34** | 21,797,672 | 21,797,672 | **0** | ✅ fixed, §2l |
| **ViT-Tiny** | 5,717,416 | 5,717,416 | **0** | ✅ architecture clean |
| **MobileNetV2** | ~~3,521,928~~ **3,504,872** | 3,504,872 | ~~+17,056~~ **0** | ✅ fixed — 52 conv biases dropped, `6b48389` |
| **EfficientNet-B0** | ~~5,309,556~~ **5,288,548** | 5,288,548 | ~~+21,008~~ **0** | ✅ fixed — 49 conv biases dropped (SE's KEPT) |
| **ConvNeXt-T** | ~~28,572,852~~ **28,587,592** | 28,587,592 | ~~−14,740~~ **0** | ✅ fixed — the REAL channel LN, all 22 sites, math forward + rung E landed 2026-07-31 (below) |

**mnv2 and EfficientNet have R34's step-B defect and ONLY that.** Every conv is BN-followed, the
reference's `.convBn` carries no bias, and ours does. The fix is `e9c2729` re-run on two more nets:
thread `convBias`, bind the operand to a zero constant, gate on all artifacts re-rendering
byte-identical, tie (`x + 0.0` is exact ⇒ the forward comes out bit-exact), swap the layout.
⚠ **Not verified**: whether the enet reference's SE convs *should* carry biases (the real
EfficientNet's do). The gap closes exactly against THIS reference; whether the reference is itself
paper-faithful there is a separate question nobody has asked.

#### ConvNeXt — the scalar LayerNorm, and it is CHEAPER than it looks

**The deviation.** All 22 LayerNorm sites carry **rank-0 scalar γ/β** where ConvNeXt specifies
per-channel vectors. Per-channel γ+β over the reference's own site widths (5×96, 4×192, 10×384,
3×768) is 14,784 floats; the render carries 44 scalars; **14,784 − 44 = 14,740**, the measured gap
to the float. Site count matches (22 both sides), every other tensor shape matches, and the tensor
count is 180 on both sides.

⚠ A correction recorded because it nearly shipped: an intermediate hand inventory said ConvNeXt-T
has 23 LN sites and therefore that one 768-wide site was *also* missing. Counting `jnp.ones(` in
the reference directly gives **22**. There is no missing site; the scalar affine is the whole gap.
Count the reference, do not enumerate the architecture from memory.

**It is the §2k failure mode exactly**: `convnextVerified`'s blurb says *"depthwise-7×7 + LN +
GELU"* while the deviation lives only in codegen docstrings (`VerifiedSpec.lean:66` "scalar-LN",
`StableHLO.lean:2835` "Scalar LN (`layerNormForward = bnForward`)").

**⛔⛔ THE SURVEY'S CENTRAL CLAIM IS REFUTED — 2026-07-31, and it doubles the job.** The bullet
below said *"the normalization AXIS is correct"*. **It is not.** The claim came from comparing the
literal `across dimensions = [1]` in the artifact against `jnp.mean(x, axis=1)` in the reference —
but **the artifact's tensor is rank-2 `[B, C·H·W]` and the reference's is rank-4 NCHW.** Same
literal, different function. This is §4's own documented trap — *"a BN reduce-dim census is only
valid WITHIN ONE TENSOR LAYOUT"* — hit a second time, one net over from where it was written.

Measured, from the committed `convnext_fwd.mlir` and `generated_convnext_tiny_imagenet.py`:

| | reduces over | statistics per example | affine |
|---|---|---|---|
| **the render** (`.bnF` ⇒ `bnForward`) | the **whole C·H·W map** | **1** | scalar γ, β |
| **the reference** (`channel_layer_norm`) | **C only**, per (h,w) | **H·W** | per-channel γ,β `[C]` |

`bnForward n ε γ β x = fun i => γ * bnXhat n ε x i + β` over the whole `Vec n` — one mean and one
variance for all 301,056 values of a stage-1 site, where ConvNeXt wants 3,136 of them, one per
spatial position over 96 channels. **21 of the 22 sites are on the wrong axis.** The 22nd is the
**head LN, and it is correct**: it runs after GAP on a flat `[768]`, where there is no spatial
extent left, so reducing "everything" *is* reducing over channels. (Site census from the artifact:
flat lengths 768×1, 37632×3, 75264×10, 150528×4, 301056×4.)

**▶ The consequence for the plan, and it is the sharp bit.** §2m scoped this as *"the scalar affine
is the whole gap"* — and doing only that would make the parameter count match the reference
**exactly**, at 28,587,592, **while the function stayed wrong.** The count is what the audit checks.
A net that matches its reference's parameter count and computes a different function is a worse
outcome than the honest deviation we have now, because the one cheap signal that something is off
would go green. **Whatever is done here, do not land the affine alone.**

*The original survey bullet follows, corrected:*

* ~~the normalization AXIS is correct~~ ⛔ **the axis is WRONG on 21 of 22 sites** (above). The
  reference is `channel_layer_norm`, `jnp.mean(x, axis=1)` over NCHW; the artifact reduces over the
  flattened feature map.
* **per-channel LN already exists in the kit and SHIPS** — ViT does `lnRowF` (pure normalize) →
  `rowScaleF` (per-channel γ) → `rowBiasF` (per-channel β), with `veclnGammaGrad`/`veclnGammaSgd`
  on the backward and `[192]` vector params. So this is not a new capability, it is a capability
  ConvNeXt does not use.

**What is actually needed — RESCOPED 2026-07-31 after the refutation above.** Both the reduction
axis and the affine have to change, so this is no longer "assemble one op family from two existing
pieces". Two routes, and the choice is a real one:

**Route A — transpose, and reuse ViT's row-LN family unchanged.** ConvNeXt's channel-LN *is* ViT's
row-LN under an NCHW→NHWC transpose: view one example as `[C, H·W]`, transpose to `[H·W, C]`, and
each row is then a spatial position with `C` features — exactly what `rowLNFlat m n ε γ β v =
flatten (fun i => bnForward n ε γ β (unflatten v i))` computes. **Every op this needs already
exists and is proven**: `transposeF {m n} : SHlo (m*n) → SHlo (n*m)` (den `Mat.flatten ∘
Mat.transpose ∘ Mat.unflatten`), `lnRowF` at γ=1/β=0, `rowScaleF (γ : Vec n)`, `rowBiasF
(β : Vec n)`, `lnRowBack`, `veclnGammaGrad`. **No new `SHlo` op, so §4's ten-sites cost never
arises.** What it costs instead: restructuring 21 sites in both directions, the proof-side
denotation (`convnextForward`'s LN is `bnForward` today), and **2 transposes per site per
direction** on tensors up to 301,056 elements — a throughput question that must be measured, not
assumed.

**Route B — a new rank-4-aware channel-LN family** (forward, input-VJP, γ-grad, β-grad) reducing
axis 1 of `[B,C,H,W]` with a per-channel affine. No transposes, so it is the faster graph; but it
is four new ops at §4's ten sites each, plus their `den`, faithfulness theorems and parse cases.

**Recommendation: A first.** It is strictly less *proof* work — it reuses ops whose VJP theorems
are already discharged and shipping in ViT — and this repo trades throughput for provenance
everywhere else. Measure the transposes before committing; fall back to B only if they cost enough
to matter.

#### ✅ ROUTE A STEP 1 IS DONE — `lake build channel-ln`, and it needs NO new op in EITHER direction

The §2l-shaped check, run before touching any of the 21 sites: settle the primitive on device
first. `tests/TestChannelLN.lean` renders the composition at `B 2, C 4, S 6` and drives it against
the closed form recomputed from the inputs — a reference implementation, not a second render.

| gate | result |
|---|---|
| forward: `transposeF ∘ lnRowF ∘ rowScaleF ∘ rowBiasF ∘ transposeF` vs closed-form channel-LN | **rel 0.000000** |
| ⚠ **CONTROL** — the incumbent `.bnF` vs the same closed form | **rel 0.821** ✅ fires |
| backward `dx` = `transposeF ∘ lnRowBack ∘ rowScaleF ∘ transposeF` | **rel 0.000000** |
| backward `dγ` = `veclnGammaGrad` | **rel 0.000000** |
| backward `dβ` = `rowDenseBiasGrad` | **rel 0.000000** |

**The backward is the half that decided it.** A correct forward would still have left Route A
needing new ops if the γ/β gradients or the input VJP had no row-layout peer. They do, and all
three are generic in their dims — ViT merely instantiates them at 197×192. `rowDenseBiasGrad`
even contracts the **batch** as well as the rows (`dims = [0,1]`), which is what a shared param
gradient needs.

**The control is the load-bearing gate**, and it does double duty: it says the probe is measuring
the axis (not something both paths satisfy), and — at **rel 0.82** — it is an independent
measurement that the deviation is real and large, from the op level rather than from reading the
artifact.

#### ▶ THE FLIP IS BLOCKED ON EXACTLY ONE THEOREM — measured 2026-07-31, not estimated

The flip was **attempted and reverted**, the §2l way: flip everything, let the build say what breaks,
then decide with a number instead of a guess. Flipped `VLayer` (a new `convNextBlockCh`, `.layerNorm`
for the stem/downsample sites), `convnextVerified.layers`, `ConvNeXtLayout.specs` and the render
default; re-rendered; **`#guard convnextVerified.toSpecs == ConvNeXtLayout.specs` passed**, the four
artifacts came out at the exact reference count, and then:

> `lake build Proofs Certs Codegen` — **3,907 of 3,908 targets green.** The one failure is
> `SpecVJP.lean:814`, `convnextVerified_denote_eq`.

**That is the whole proof-side cost, and it is not small.** `denoteConvnextT` maps the layer list to
`convNextForwardTC`, whose LN is scalar `layerNormForward` — so a channel-LN layer list must map to
a channel-LN math forward, and behind that sits `ConvNeXtFullT.lean`'s chain: `cnxBlockW`,
`convNextStageK`, `cnxDownW`, each with its own `_diff` and `_has_vjp`, all built on
`layerNormForward`. Rung E (`convnextVerified_fwd_faithful`) additionally needs the proof-side
graph `convNextFwdGraphTC` rebuilt. ⚠ **Do not "fix" this by re-pointing the pattern at
`convNextForwardTC`** — it typechecks by `rfl` and would assert that the channel-LN architecture
denotes the scalar-LN function, which is §2k's exact sin one level down.

**What makes it tractable:** the math side can mirror Route A. `transpose_has_vjp` is proven
(`Tensor.lean:1467`), ViT's row-LN math and its γ/β gradients are proven, and ConvNeXt's own
`layerScale` is already a per-channel affine — so `channelLN = transpose ∘ rowLN ∘ per-channel
affine ∘ transpose` composes through `vjp_comp` exactly as the render composes through `pretty`.

**Reverted to inert rather than rushed or left red.** The render work is committed and green
(`e4b7815`, `4628cc8`); the flip is a one-line change (`chLN : Bool := false` → `true` in four
signatures) plus the `VLayer`/layout edits, and it lands the moment the math forward exists.

#### ✅ THE MATH FORWARD EXISTS AND THE FLIP IS LANDED — 2026-07-31. **UNCOMMITTED.**

`lake build Proofs Certs Codegen` is green at **3,909 jobs**; `SpecVJP.lean:814` — the one target
that blocked this for a session — builds. Every new declaration is **3-axiom clean**.

**▶ Route A cost no new `SHlo` op AND no new VJP**, exactly as scoped. `chanLNTensor3` is five
already-proven pieces glued by `vjp_comp`:

| piece | reused from |
|---|---|
| `reassocFwd`/`reassocBack` + VJPs | `PerChannelBN.lean` — the per-channel BN layout bridge |
| `transposeFlat` VJP | `Tensor.lean`'s `transpose_has_vjp` through `hasVJPMat_to_hasVJP` (a re-typing, not a proof) |
| the rowwise vector-LN | `ViTVecLN.lean`'s `layerNormVec_per_token_has_vjp_mat` — ViT's `[192]` LN, re-read with "token" = "spatial position" |

The only hypothesis is `0 < ε`, exactly as the scalar `layerNorm_has_vjp` it replaces, so the
22 LN positivities are the whole hypothesis set of `convNextForwardTCh_has_vjp` — same count as
the scalar peer, since the stem LN it gains and the head LN it loses cancel.

**▶ THE SEAM THAT NEARLY SHIPPED, and it is the reusable part.** The render transports its index
with a `▸` cast (`ConvNeXtRender.reassoc`, because `c*h*w = (c*h)*w` is not defeq to `c*(h*w)`);
the math uses `PerChannelBN`'s `finProdFinEquiv` re-association, whose "row `c` is channel `c`"
reading is the *only* reason the composition is legibly a **channel** LN. **Nothing forced those
two to be the same map** — and if they are not, the math and the artifact are different functions
with no gate between them, which is §2k's own sin one level down.

They are the same map. `reassocFwdIdx_val` proves it: row-major `finProdFinEquiv` sends both
`((c,hi),wi)` and `(c,(hi,wi))` to the same linear offset, so the bridge preserves `Fin.val` and
*is* the type cast. `den_cast`/`den_reassocS` lift that to the graph, which is what lets
`chanLNGraph_faithful` close. **Ask this question of any layout bridge that gets spelled two
ways** — it is §4's one-tensor-layout rule in the type system rather than in a reduce-dim census.

**▶ What was built** (`LeanMlir/Proofs/Architectures/ConvNeXtChannelLN.lean`, new, + additions to
`ConvNeXtFullT.lean`):

* `chanLNTensor3 c h w ε γ β` + `_diff` + `_has_vjp` — channel LN at the conv activation layout;
* `reassocFwdIdx_val` / `reassocBackIdx_val` / `den_cast` / `den_reassocS` / `den_unassocS` — the seam;
* `rowLN_affine_eq` — the emitted three-op affine tail (`lnRowF(1,0) → rowScaleF → rowBiasF`) IS
  ViT's per-token `layerNormVec`, which is what collapses the graph's five denotations onto three;
* `cnxBodyWith` — the ConvNeXt block body abstracted over its normalisation, so the channel-LN
  world costs one definition rather than a second copy of the six-piece `vjp_comp` chain;
* `CnxBlockParamsCh` / `cnxBlockChW` / `convNextStageChK` / `CnxDownParamsCh` / `cnxDownChW` /
  `CnxTWeightsCh` / **`convNextForwardTCh`** + `_has_vjp` + `_eq_chain` + `_has_vjp_correct`;
* the graph side: `chanLNGraph`, `cnxBlockChGraphW`, `cnxStageChGraphK`, `cnxDownChGraphW`,
  **`convNextFwdGraphTCh`** + `convNextFwdGraphTCh_faithful` — rung E's new apex.

Built as a **parallel** chain, not by re-instantiating the scalar one, for `MobileNetV2RenderB`'s
reason (§2f): `convNextForwardTC` backs the retired render's tie, `ConvNeXtChainClose` and
`ConvNeXtWholeFloatBridge`. The scalar chain is untouched.

**▶ The gates, as run.** Note there is **no A-vs-B tie across this flip and there cannot be** —
the function AND the parameter shapes both change, so one param blob cannot feed both renders.
That is the honest difference from the conv-bias drops (§2m), which were bit-exact.

| gate | result |
|---|---|
| `lake build Proofs Certs Codegen` | ✅ **3,909 jobs**, `SpecVJP` green |
| `#guard convnextVerified.toSpecs == ConvNeXtLayout.specs` | ✅ holds — both hand-lists rewritten INDEPENDENTLY from the render's `allParams chLN := true` |
| parameter count | **27,826,282 at K = 10 ⇒ 28,587,592 at K = 1000** — the JAX reference's own reported count, EXACTLY. Still **180 param tensors** (stem/head LN swap one-for-one) |
| `regen_verified_mlir.sh check` | ✅ **67 artifacts, one writer each**; prefix audit green (`convnext_fwd` is a 1544-line prefix); empty-slot and `%zb` audits clean |
| **`fwd-tie convnext`** (XLA, GPU) | ✅ compiles and RUNS on the driver's 180-param blob, self-tie **bit-exact 320/320**, logits \|max\| 7.61 (non-degenerate) |
| **`sgd-render-tie convnext`** A-vs-A (XLA, GPU) | ✅ the whole channel-LN BACKWARD at full depth — **bit-exact 27,826,282/27,826,282**, max\|g\| 4.15, 27,826,157 non-zero |
| **`convnext-adam-tie`** A-vs-A (IREE) | ✅ **bit-exact 83,478,849/83,478,849**, reorder control 1e-6, spread 0/180 |
| **`channel-ln`** op gate (XLA, GPU) | ✅ re-run after the flip: forward and all three backward pieces **rel 0.000000** vs the closed form, incumbent `.bnF` control fires at **0.820856** |
| descent on the swapped bytes | ✅ loss 3.94 → 2.32, val **19.97 → 25.32 → 33.27%** (capped 25-step epochs, XLA) |

`fwd-tie` and `sgd-render-tie` are the load-bearing ones: they are the instrument §2m named
(*"compile the candidate before believing a count"*), and the forward one is exactly the check
whose absence killed the first mnv2 swap — the driver builds the blob from the LAYOUT and the
graph either accepts it or PJRT refuses the call.

**▶ A claim this thread published is now FALSE and is retired.** §2h-quater's headline —
*"44 of the 180 collectives are RANK-0"* — **was** ConvNeXt's scalar LayerNorm. Measured on the
re-rendered DP artifact: the LN collectives are `tensor<96xf32>` … `tensor<768xf32>` and **0 of
180 are rank-0**. Nothing in the repo exercises a rank-0 `stablehlo.all_reduce` any more. Both
the render docstring and the `#eval` comment are corrected (byte-inert: the four artifact md5s
are unchanged across that edit).

#### ✅ THE 2-GPU GATES, RE-RUN ON THE CHANNEL-LN ARTIFACT — 2026-07-31

Both pass on the re-rendered DP artifact, each with a control that fires:

| gate | TEST | CONTROL |
|---|---|---|
| **`convnext-dp-check`** (duplicated batch) | **BIT-EXACT on all 83,478,849 floats** — θ, `m`, `v` each 27,826,282/27,826,282, `%loss` 1/1, `bc` 2/2 | sum-not-mean (180 divisors 2.0 → 1.0): gradient **0.942569**, rc=1 |
| **`shard-check convnext`** (asymmetric batch) | `\|DP − mean(A,B)\|` = **7.8e-8** | `\|DP − A\|` = **0.0416**, 5.3e5 apart |
| 2-GPU descent | loss 13.97 → **2.15**, val 13.86 → **31.95%** over 6 epochs (25-step, XLA) | — |

**The DP agreement got TIGHTER, not looser** — bit-exact where the scalar-LN render was ≤1.1e-8
with 3 coordinates off. Consistent with §2h-quater's own LayerNorm reading: with no batch
statistics there is less for a reordered reduction to land in, and the channel LN has *more*
per-example structure than the scalar one, not less.

⚠ **The θ rule flipped sign on this net and it is worth recording.** §2h-quater found the broken
render moved **θ by 9.7e-5** — *under* a 1e-4 θ gate, its sharpest demonstration that a θ-based
gate passes a 2× gradient error. On the channel-LN net the same control moves θ by **1.95e-4**,
i.e. *just above* the same gate. The rule (gate the gradient, never θ — §3) is unchanged and is
what saved it either way; what moved is the margin, which is net-dependent and not something to
rely on. `%loss` stayed BIT-EXACT through the control on both, correctly localising the fault to
the backward.

**▶ STILL OWED — none of it blocks the build, and the first is the real one:**
* ~~**the 80-epoch re-run**~~ ✅ **DONE 2026-07-31 (§2o Part B)** — **84.41%** final / 84.82% best
  / 1h54m, replacing the VOID 82.75%.
* ~~**the 2-GPU scaling ratio**~~ ✅ **DONE 2026-07-31 (§2o Part B)** — **1.70×** (78.0 → 46.0 s
  marginal, train-only). The worry was that the channel LN's 44 extra transposes per direction,
  free at the op level per `channel-ln --bench`, might not be free end to end. **They are**: both
  the single-GPU marginal (77.5 → 78.0 s) and the ratio (1.68 → 1.70×) are unchanged within noise.
* ~~`ConvNeXtWholeFloatBridge` / `ConvNeXtBackB0` / `WholeNetForwardTies` still ride the SCALAR
  chain~~ — `ConvNeXtWholeFloatBridge` was discharged by §2n and **`ConvNeXtBackB0` by §2o Part A**,
  which added the channel-LN backward capstones alongside the scalar ones (the file now carries both
  worlds, explicitly labelled). `WholeNetForwardTies` is the remainder, and `ConvNeXtTiePoC`'s §1a
  tie carries a scope label rather than a port (§2n measured the port as chapter-sized).
* ~~the §1a tie (176/180)~~ — the **render capstone** (`tests/TestConvNeXtTTrainPC.lean`) is
  ✅ **RE-ALIGNED 2026-07-31 and back at PARITY: 180 of 180 outputs BIT-IDENTICAL** against the
  channel-LN artifact, control (EPS 1.0e-6 → 1.0e-3) fires at 0/180 with rc=1. It got **strictly
  stronger**: the LN γ/β gradients are now proof-rendered (`veclnGammaGrad`/`rowDenseBiasGrad`
  under the transposes) where the scalar LN needed ~16 hand-emitted lines of x̂ recomputation per
  site, so **44 params moved from hand-emitted to `pretty`**. The dead scalar emitter is deleted.
  ⚠ `render_parity.py` needs `iree-run-module`, which is **not** in this repo's `.venv/bin` (only
  `iree-compile` is) — it lives in `../lean4-mlir/.venv/bin`; put both on PATH.
* §2f-bis's conditioning/spread findings were measured on the scalar-LN render; re-read before
  quoting.

#### ▶ The remaining proof-alignment work, scoped by reading the files — 2026-07-31

**Nothing is broken**: all 3,909 targets build and every scalar-LN theorem is still TRUE. The
issue is SCOPE — they describe a net the repo no longer ships. And the job is much smaller than
it looks, for a reason worth knowing:

> **The ConvNeXt float story is ALREADY abstract in the LN.** `convnext_floatBridges` takes
> `FloatBridges lnStem`, `FloatBridges lnHead` and every stage/downsample as *hypotheses*, and
> its own docstring says the LNs enter abstractly "because `layerNormForward = bnForward` has the
> rsqrt keystone". `floatBridges_convNextStageK` likewise takes `∀ i, FloatBridges (layerNorm…)`.

So the whole-net float theorem needs **zero change** — the channel-LN net instantiates it at
`lnHead := id` (`floatBridges_idVec` already exists in that file) with the Ch slots. What is left:

| item | what it needs |
|---|---|
| `FloatBridges (chanLNTensor3 …)` | the transposes/reassoc are exact reindexes (cheap); the per-row normalise reuses `floatBridges_bn`'s rsqrt keystone rowwise |
| `floatBridges_cnxBlockW` → Ch | ⚠ **restate it over `cnxBodyWith`** — the LN-abstract body already built for the VJP — so ONE theorem serves both worlds instead of two copies. Same generic-in-the-LN move that made the VJP cheap |
| `floatBridges_convNextStageK` / `_cnxDownW` → Ch | mechanical mirrors |
| `ConvNeXtBackFloatBridge` | the backward peer of the above |
| `ConvNeXtBackB0.lean` (164 lines) | `cnxDownBackGraph` ties the emitted backward to `cnxDownW_has_vjp`'s backward; needs a Ch peer |
| `WholeNetForwardTies.convNextForwardT_eq_skeleton` | the skeleton already has LN slots ⇒ `lnHead := id` + an `eq_chain` unfold |

**▶ THE DECISION IS TO DROP IT, NOT RELABEL IT — Brett's call, and it REVERSES the
recommendation this section carried an hour earlier.** The full scalar-LN chain comes out once
the FloatBridges are discharged; it is no longer on the primary path and a retired net that
still elaborates is one more thing to drift. §2n has the scope, the order and the measured
dependency table. *(The relabel argument — §2k's "the sin was the blurb" — still applies to
anything that STAYS, which is the ch9 representative.)*

#### ⛔⛔ AND THE LN PLACEMENT IS WRONG TOO — found 2026-07-31 by the count NOT matching

With the axis and the affine both fixed in the render, the parameter count came to **28,588,936
against the reference's 28,587,592** — a residual of **+1,344**. It decomposes exactly, and into
two more deviations:

| | reference (`forward`, read directly) | our render |
|---|---|---|
| **stem** | patchify conv → **`channel_layer_norm`** | patchify conv → *(nothing)* ⛔ |
| **head** | GAP → dense | GAP → **LN** → dense ⛔ |

`+2×768 (our extra head LN) − 2×96 (our missing stem LN) = +1,344`, to the float. The reference's
22 sites are **1 stem + 18 block + 3 downsample**, which is also exactly §2m's own measured width
census (5×96 = stem + 3 blocks + down0; 4×192; 10×384; 3×768 = the three stage-4 blocks and
nothing else). Ours are 18 block + 3 downsample + 1 head.

⚠ **This is the sharpest argument yet against treating a matching parameter count as an
architecture check, and it nearly went the other way.** The two deviations are in opposite
directions and *nearly* cancel: 1,344 out of 28.6M is **0.005%**. Had the extra and the missing LN
been at the same width they would have cancelled EXACTLY, the count would have matched, and a net
with its final normalisation in the wrong place would have passed the audit clean. The count is a
decomposition test — it is only evidence when the residual is explained, not when it is small.

**So ConvNeXt has FOUR deviations, not one**: the reduction axis (21 sites), the scalar affine (22
sites), the missing stem LN, and the extra head LN. §2m's original scope named only the second.

#### ✅ AND THE TRANSPOSES ARE FREE — measured 2026-07-31, `channel-ln --bench`

The one number that could still have sent this to Route B. Every LN shape ConvNeXt-T actually has,
**at its real site count**, so nothing is extrapolated — 4×(96,3136), 4×(192,784), 10×(384,196),
3×(768,49), forward, B=32, median of 12 samples:

| (C, S) | sites | Route A | `.bnF` | Δ |
|---|---|---|---|---|
| (96, 3136) | 4 | 8.30 ms | 8.35 ms | **−0.05** |
| (192, 784) | 4 | 4.10 | 3.95 | +0.15 |
| (384, 196) | 10 | 2.55 | 2.65 | −0.10 |
| (768, 49) | 3 | 1.15 | 1.15 | 0.00 |
| **whole-net forward LN** | **21** | **16.10 ms** | **16.10 ms** | **0.00** |

**Δ = 0.0 ms on 16.1 ms of LN, against a ~286 ms step.** The per-stage deltas are ±0.15 ms and
**signed in both directions**, which is the signature of noise rather than of a cost — the same
spread-not-magnitude reasoning §2f-bis used on the gradient gate. XLA folds a transpose into the
consumer's layout assignment rather than materialising it, which is the expected behaviour and is
now measured rather than hoped for. **Route B's only advantage over A was throughput, and it is
zero.**

⚠ Three caveats, none of which move the conclusion: `.bnF` is a *proxy* for Route B (same
reduce→normalize→affine shape, no transpose) because Route B does not exist to be timed; the sites
are chained back-to-back here, so no transpose can fuse into a conv consumer — an **upper** bound
on the delta, and the bound is already zero; and this is one box, one run each.

⚠ **A resolution trap this hit first, worth keeping.** The first pass timed ONE invoke per sample
with `IO.monoMsNow` — integer milliseconds — and the stage totals were 1-9 ms, so the tick was
comparable to the quantity being measured. It reported "A is 1 ms faster", which is one tick and
means nothing. Fixed by timing 20 invokes per sample (0.05 ms effective resolution). **§2j's rule
in a new place: check that your instrument can resolve the thing you are measuring before reading
the answer off it.**

**Route C, and it is legitimate: fix the DOC, not the net.** Characterise the deviation precisely
in `convnextVerified`'s blurb — the way §2l says R34's should have said "3×3 projection" — and
leave the net alone. That is what §2k's whole finding was *about*: the sin was the blurb claiming
"Real ResNet-34", not the deviation existing.

**Blast radius:** ConvNeXt's **82.75%** run, the §1a tie (176/180), §2f-bis's AdamW tie +
conditioning/spread findings, `convnext-dp-check`, `convnext-shard-check`, 5 artifacts. ⚠ And
**§2h-quater's headline evaporates**: *"44 of the 180 collectives are RANK-0"* IS this deviation —
per-channel LN makes them rank-1 and that novelty goes away.

**Recommended order** (cheapest-first, and it front-loads the certain wins): mnv2 conv biases →
enet conv biases → ConvNeXt LN. The first two are a known recipe run twice; the third is a
decision like §2l was. ✅ **All three DONE 2026-07-31** — and the order held: the two conv-bias
drops validated the swap recipe (bit-exact ties) before ConvNeXt spent it on a change no tie can
license, because the function genuinely moves.

#### ▶ STATE 2026-07-31 — both renderers THREADED and INERT, the swap tail is what is left

`5dbbbd2` (mnv2) and `44848af` (enet) thread `convBias` through the renderers and stop while the
change provably emits nothing. **Gate 1 in its strong form holds on both: every artifact in
`verified_mlir/` is byte-identical, across all nets** — which is also the check that the shared
helpers moved into `StableHLO` (`biasName`, `zeroBiasPrelude`, `fmt6`, `alphaOverK`) are inert.

`#guard`s pin BOTH arities on each net, so the drop cannot happen silently:

| net | with biases | without | dropped | audit predicted |
|---|---|---|---|---|
| MobileNetV2 | 210 | **158** | **52** | 52 ✓ |
| EfficientNet | 262 | **213** | **49** | 49 ✓ |

Two independent routes — the renderer's signature list and the layout walk — landing on the same
number is what makes these safe to proceed on.

**The remaining tail, identical for both:** wire `zeroBiasPrelude` with the net's own channel
widths → the `VLayer` spec (`invertedResidual` / `mbConvSE`, plus `.convBnNB` for stem and head) →
the `*Layout.specs` hand-list → flip the default, re-render, tie, smoke. **Finish ONE end to end
first** (mnv2 — no SE complication) so the tie harness and swap recipe are validated once before
the second net reuses them.

**Two things learned doing the threading, both of which cost time:**

* ⚠ **enet trains its conv biases through `bnBt`** — the BN-β op — fed the BN *input* cotangent,
  NOT through anything named `convBiasGrad`. Grepping the obvious name finds nothing. It is also
  the same mathematical statement as R34's: BN's input cotangent sums to zero per channel in ℝ,
  which is exactly why the conv-bias gradient is zero and the parameter is safe to drop. One
  reason, three nets.
* ⚠ **enet's SE biases must NOT be dropped** (`zb1`/`zb2`): those 1×1 convs are followed by an
  ACTIVATION, not BN, so nothing absorbs them and the reference carries them. The audit's rule —
  a rank-1 kind-2 param immediately after a **rank-4** kernel — excluded them automatically
  because SE's params are rank-2, which is why the gap closed exactly on the first attempt rather
  than by luck. Any net with non-BN-followed convs needs the same care.

*Method note for whoever continues: thread these with a LINE-AWARE, paren-balanced pass, not
blanket regexes. Two attempts with `re.sub` over these files split a multi-line `pretty B (...)`
across an `else`, and backtracked `\d+` into `160` to produce `convBias0`. Also put `convBias`
LAST in every public signature — inserted mid-list it captures an existing positional argument
(`efficientnetTrainStepFaithfulV`'s `funcName`, `enetBackAll`'s optional `smooth`).*

#### ⛔ The swap was attempted, BLOCKED and reverted (`b8971ce`) — ✅ then DIAGNOSED AND FIXED

`b8971ce` flipped the default, re-rendered all 6 mnv2 artifacts, and the trainer died at
`f32 forward failed` — **the EVAL forward, not the train step**, with the train step compiling, the
tie passing (bit-exact on every forward-only output) and the audit green. It was reverted
undiagnosed, with the BN running-stat inputs named as the suspect. **That was the wrong suspect:
the stat slots tie exactly, 104 on both sides.**

**The cause is a two-signature-sources skew, and it is §2a's own disease one level down — in code
rather than in artifacts.** mnv2 has TWO renderers describing one net:

| | writes | drops at `convBias := false` |
|---|---|---|
| `MobileNetV2RenderB.mnv2SigList` | the AdamW train step (+ DP) | **52** ✓ `#guard`ed 210/158 |
| `MobileNetV2Render.paperSig` | `_fwd`, `_fwd_eval`, `_train_step` | **50** ⛔ not guarded |

All 15 block conv sites threaded `biasName convBias …` correctly; **the stem `%bs` and the head
`%bh` were hardcoded** in `paperSig` and in the two forward chains. So the driver walks a 158-param
layout while the eval graph wants 160, and PJRT refuses to run it. **Measured, and the shim had
already printed it — one stderr line above the Lean exception:**

```
[pjrt_ffi] Execute: Execution supplied 315 buffers but compiled program expected 265 buffers
uncaught exception: f32 forward failed          ← 265 = 1 + 160 + 104, driver supplies 1 + 158 + 104
```

After the fix the same probe reads **`expected 263`** — exactly what the driver supplies. Before
263, after 265: two numbers, one skew, and both come from XLA rather than from a reading of the
source.

**A SECOND defect it was hiding, found the same way and not yet hit.** The backward `names` lists
were ungated while the ops filling them were gated, so a dropped bias sat in the return list as the
empty string `pure ("", "")` hands back:

| render | at `convBias := false` |
|---|---|
| `mobilenetv2_train_step` | `return %v5358, , %v5379 …` — **210 names, 52 EMPTY, against 160 types** |
| `mobilenetv2_reduced_train_step` | 82 names, 20 empty, 64 types |
| **`efficientnet_train_step`** | **262 names, 49 empty, 213 types** — enet has this one too |

This is the same class as the AdamW-tail bug `b8971ce` describes fixing in RenderB (`stablehlo.multiply
%v5793,  :`), still live in all three SGD renderers. ⚠ **An arity `#guard` cannot catch it** — the
list keeps its full length; only the lowerer ever sees it.

**A THIRD defect, found by compiling rather than counting.** With the arities right and the return
lists clean, `iree-compile` still rejected enet: *"use of undeclared SSA value name"*. **None of
EfficientNet's four renders emitted `zeroBiasPrelude`** — 15 `%zb` widths used, **0 declared**,
across `_fwd`, `_fwd_eval`, `_train_step` and the AdamW step. It is the first item of §2m's own
enet tail, so it was owed rather than broken; what is worth recording is that **two clean audits
and a green `lake build` preceded it.** Counting arities and counting empty slots are both static
checks over text, and neither knows whether a name is *defined*. The render is `enetBiasWidths`
now, one list feeding all four calls — ⚠ **and it is the CONV widths only**: SE's 1×1 convs are
followed by an activation, not BN, so their biases stay real parameters (§2m above).

**What landed (gate 1 re-verified: every artifact in `verified_mlir/` byte-identical, 0 lines of
`git diff`):**

* `StableHLO.biasSlot` beside `biasName` — the return-list peer of the operand gate;
* `paperSig`'s stem/head gated, the two forward chains take `biasName convBias "%bs" 32` /
  `"%bh" 1280`, and the reduced renderer's signature — **written out twice, which is the same
  two-lists-one-net shape** — collapsed to one `reducedSig`;
* the four block `names` lists + both `outNames` in mnv2, and the three block lists + `outNames`
  in enet;
* **`enetBiasWidths` + `zeroBiasPrelude` wired into all four enet renders** — the first item of
  enet's tail, done here because it is what makes "the bias-free renders are well-formed" true for
  both nets rather than one;
* **`#guard`s on `paperSig` (210/158) and `reducedSig` (82/62)** — the missing peer of RenderB's.
  With both routes pinned the two lists are one contract;
* **`regen_verified_mlir.sh check` grew TWO audits**: an empty-SSA-slot check, verified to fire on
  both real shapes (`return %a, , %b` and `= stablehlo.multiply %v,  :`) and stay quiet on clean
  text; and a `%zb`-used-but-not-declared check that reproduces the enet finding. The second is
  vacuous while the committed bytes carry biases and goes live the moment a net is swapped.

**Verified, and by compiling rather than by counting: all TEN bias-free renders now compile on
XLA** with the output counts the signatures promise — mnv2 `_fwd`/`_fwd_eval`/`_train_step` and the
AdamW/DP steps at **158** params (581 outputs), reduced 62, enet at **213** (740 outputs). Gate 1
in its strong form still holds: `git diff verified_mlir/` is **0 lines** across all 67 artifacts,
and `lake build Proofs Certs Codegen` is green at 3,908 jobs.

#### ✅ AND THE mnv2 SWAP IS DONE — 2026-07-31. **158 params, 3,504,872 at K = 1000.**

The tail ran as scoped, in this order. **The param count now matches the JAX reference exactly**
(`jax/.lake/build/generated_mobilenet_v2_imagenet.py` header: `Parameters: 3504872`), which is what
the whole §2m detour was for. At the shipped K = 10 it is 158 tensors / 2,236,682 floats.

**▶ The gate the first attempt did not have, built FIRST and run BEFORE the swap.** `--tie` gates
the AdamW *train step* and nothing else, which is precisely how a 160-param forward shipped past a
green tie. `conv-bias-zero --fwd <candidate> [--eval]` now feeds a candidate forward **the same
bias-free blob the driver will build**, so an arity or ordering skew is a hard failure in the gate
rather than at run time:

| gate | result |
|---|---|
| `--fwd` (`@mobilenetv2_fwd`) | **BIT-EXACT 320/320 logits**, 158 params vs A's 210 |
| `--fwd --eval` (`@mobilenetv2_fwd_eval`) | **BIT-EXACT 320/320**, + 52 layers of frozen stats |
| **negative control** — candidate = the committed 210-param render | `Execution supplied 263 buffers but compiled program expected 315`, rc=1 |

The control is the point: it fails with the *same PJRT message* that killed the first swap, so the
gate demonstrably catches that bug. Bit-exact is the right bar here because `mkParam` gives conv
biases their real init (kind 2, **zeros**) and `x + 0.0` is exact — A and B differ only in whether
the zero arrives as an argument or a constant. ⚠ Do **not** "strengthen" it with non-zero biases in
`--eval`: the frozen `μ` was estimated on whatever net produced it, so `(x+b) − μ` with a biased
`μ` is not `x − μ`, and a correct render would fail.

**The train-step tie, re-run:** A (210, committed) vs B (158) over 6,744,161 shared floats —
**every forward-only output BIT-EXACT** (`%loss` 3/3, all 34,112 BN running stats), differences
confined to the backward at max **2.5e-7**, against an A-vs-A floor bit-exact on all 6,795,329
outputs. 6,726,967 shared floats bit-exact.

**The `#guard` earned its keep again.** `mobilenetv2Verified.toSpecs == MobileNetV2Layout.specs`
FIRED when only one of the two routes was changed (verified deliberately: revert the layout's stem
and `lake build` fails at `VerifiedNets.lean:418`), so the vocabulary had to grow rather than the
hand-list being edited to match. `VLayer` gained **`invertedResidualNB`** beside `convBnNB`, for
`convBnNB`'s reason — `invertedResidual` should still be sayable by a net whose blocks do ship
biases.

**One thing the R34 precedent predicted exactly:** `SpecVJP.denoteMobilenetPaper` pattern-matches
the committed layer list, so it needed the NB spelling — the same one-line change `e9c2729` made to
`denoteR34Full`. It fails loudly (`lake build`), which is the behaviour you want from a denotation
that is deliberately drift-sensitive.

**It trains on the swapped bytes** (`runs/mnv2_nobias_swap_jul31.log`, fresh checkpoint):
val **30.93% → 43.77%** over 2 epochs, against the pre-swap XLA baseline's 26.7 → 38.0 (§2h) — same
band, and §3's run-to-run spread is wider than the gap. The checkpoint is **26,840,184 B** =
3 × 2,236,682 × 4, i.e. the bias-free size, which is its own confirmation that the driver and the
graph agree. **And the eval forward runs** — the step that killed the first attempt.

**Still owed on mnv2, and deliberately not done here:**
* **the 80-epoch re-run.** `86.73%` was measured on the 210-param net. The forward is bit-exact at
  `b = 0` and the two renders' forward-only outputs are bit-exact, so it should stand — but that is
  an argument, and the *trained*-checkpoint ablation R34 ran (`--ablate`, which needs the biased
  layout) was never run for mnv2 and now cannot be without reverting. Re-run before quoting; it is
  1h25m.

**The two multi-GPU gates were re-run on the re-rendered DP artifact and both hold**, at the new
581-output arity: `mobilenetv2-dp-check` — forward **BIT-EXACT** (all 34,112 bnstat), `v`
bit-exact 2,236,682/2,236,682, gradient norm-rel **6.0e-8** (was 8.7e-8 at 210 params);
`shard-check mobilenetv2` — TEST **7.0e-8** against a built-in CONTROL of **0.953**, 1.4e7 apart.
Nothing about the collective or the shard offsets moved with the layout, which is what you would
expect and is now measured rather than assumed.

#### ✅ AND EfficientNet FOLLOWED THE SAME DAY — **213 params, 5,288,548 at K = 1000**

The recipe transferred unchanged and the prediction held: enet was cheaper, because its arity was
always right (one `enetSig` source) and the two render defects were already fixed. **The count
matches `generated_efficientnet_b0_imagenet.py`'s `Parameters: 5288548` exactly**, closing §2m's
+21,008.

| gate | result |
|---|---|
| `--fwd` (`@efficientnet_fwd`) | **BIT-EXACT 320/320**, 213 params vs A's 262 |
| `--fwd --eval` (`@efficientnet_fwd_eval`) | **BIT-EXACT 320/320**, + 49 layers of frozen stats |
| `--tie` (AdamW train step) | **BIT-EXACT on all 12,103,093 shared floats** — floor bit-exact on 12,166,117 |

**enet's tie is bit-exact all the way through the BACKWARD**, where mnv2's left 2.5e-7 there. Not
worth over-reading — it is XLA fusing a 49-chain-lighter graph the same way rather than differently
— but it is the strongest of the three conv-bias ties in the repo.

**⚠ The candidate had to be re-rendered before the tie would mean anything.** The first one used
`bStr = "0.900000"` where the committed `#eval` bakes **`"32.0"`** (the explicit mean-CE divisor),
so it differed from the reference in *two* ways and a failure would have been unattributable. A
swap tie is only evidence if the candidate differs in **exactly** the thing being gated — check the
committed `#eval`'s literals, do not reconstruct them from the signature.

**`classifyAll` handled the SE params correctly with no change**, which is the one thing that could
have gone wrong on this net: it walks the layout and treats a rank-1 kind-2 param as a conv bias
only when it *immediately follows a rank-4 kernel*, and SE's are rank-2. Confirmed by the drop
count — `--fwd` reports exactly **49** fewer, never 49+2×16. `VLayer` gained **`mbConvSENB`**
(expand/depthwise/project lose their biases, SE keeps both of its).

`SpecVJP.denoteEfficientnetB0` needed the NB spelling, same one-line change as mnv2 and R34.

**It trains on the swapped bytes** (`runs/enet_nobias_swap_jul31.log`, fresh checkpoint): val
**34.68% → 49.38%** over 2 epochs, checkpoint **48,244,296 B** = 3 × 4,020,358 × 4. ⚠ Do **not**
read that against §2e-bis's 14.98/21.83/27.13 — that smoke ran a different schedule, so it is not
an apples-to-apples pair. The evidence that the function is unchanged is the **bit-exact tie**, not
the smoke; the smoke's job is only to show the swapped bytes train and that the driver and the
graph agree on the layout.

**Both multi-GPU gates re-run** on the re-rendered DP artifact (740 outputs, 2 replicas):
`efficientnet-dp-check` — forward **BIT-EXACT** (all 42,016 bnstat), `%loss` bit-exact, `v`
bit-exact 4,020,358/4,020,358, gradient norm-rel **3.3e-8** (the same figure it reported at 262
params); `shard-check efficientnet` — TEST **8.5e-8** against a built-in CONTROL of **1.339**,
1.6e7 apart. The `tests/`-side `iree-compile` smokes over the committed bytes still pass.

**▶ §2m IS CLOSED — the audit table at the top of this section is all zeros, and as of
2026-07-31 that is literally true.** All five verified nets now agree with the JAX reference they
are paired against: R34 21,797,672 (§2l), ViT 5,717,416 (always), MobileNetV2 3,504,872,
EfficientNet-B0 5,288,548, and **ConvNeXt-T 28,587,592** — the last one closed by landing the
channel-LN math forward and flipping the render, not by matching a count (see the warning above
about why matching the count alone would have been the WORSE outcome). ConvNeXt's 82.75% was void
with it; the re-run landed 2026-07-31 at **84.41%** (§2o Part B).

⚠ **What did NOT change on any of the three nets: the proof side.** `B0Weights` / `MNV2PaperWeights`
still carry their bias fields, `den` and every faithfulness theorem are untouched, and no `SHlo` op
was added — the bias operand simply becomes a zero CONSTANT instead of a function argument, which
is `e9c2729`'s spelling and why §4's ten-sites-per-op cost never came up. So comments like
`AuditAxioms`'s *"covers all 262 params"* are describing the **math-level** param set and are still
correct; the 213 is the render's signature. Do not "fix" them to match.

**The lesson, and it is the thread's own:** the guards were real, fired, and pinned the WRONG HALF.
Two arity routes agreeing was the check — and only one route was pinned, so the half that ships
(RenderB, which the tie exercises) was gated and the half the driver *evals* through was not. A tie
that never touches an artifact cannot license it: `fwd-tie mobilenetv2 --eval` existed the whole
time and was not run. The second-order version, from the third defect: **static audits over
emitted text cannot see an undefined name.** Three of the four things wrong here were invisible to
`lake build`, and the cheapest instrument that found all of them was
`.lake/build/bin/fwd-tie <net> --eval <candidate>` — it XLA-compiles any path you hand it in
seconds, whatever the entry name, and prints the compiler's own complaint. **Compile the candidate
before believing a count.**

### 2n. ✅ CLOSED 2026-07-31 — bridges discharged, §B tied, and the scalar chain DROPPED

#### ✅ The drop, executed — 852 lines deleted, and the four things it taught

`lake build Proofs Certs Codegen` green at **3,910** jobs · `verified_mlir/` **0 lines of `git
diff`** after a real re-render · `regen_verified_mlir.sh check` **67 artifacts / one writer each**,
all four prefix checks OK · `AuditAxioms` **1506/1506** three-axiom · `render_parity.py` smoke on
gfx1100: **180/180 outputs finite and non-zero, REF RUNS ✓**.

Deleted: the `[3,3,9,3]` scalar-LN chain and its graph section from `ConvNeXtFullT.lean`
(979 → 473 lines), the three scalar float bridges + `CnxBlockBounded`, `convNextForwardT_eq_skeleton`,
`ConvNeXtBackB0`'s downsample capstone, 18 `AuditAxioms` lines, and the renderer's `chLN` flag
(76 sites). Net **−852 / +231** across 11 files.

**1. `ConvNeXtBackB0` — the §2n plan said "port to Ch"; measured, the answer was "delete".** Three
facts decided it: the capstone had no consumer but an audit line; the ch9 representative
(`Architectures/ConvNeXt.lean`) is a **2-block net with NO downsample**, so `CnxDownParams`/`cnxDownW`
were full-T scalar scope rather than shared; and porting needs a *graph-side* channel-LN backward
faithfulness that does not exist (§2m built the render and the math VJP; today's
`chanLNTensor3Back_eq_chanLN_vjp` is the MATH-side tie). ⚠ **So the channel-LN downsample now has
no graph-side backward capstone.** Recorded in `ConvNeXtBackB0`'s docstring. It is a gap the drop
**exposed rather than created** — it was always the scalar net that had the capstone.

**2. The drop found an audit hole §2m left.** The channel-LN graph faithfulness theorems
(`chanLNGraph_faithful`, `cnxBlockChGraphW_faithful`, `cnxStageChGraphK_den`,
`cnxDownChGraphW_faithful`, `convNextFwdGraphTCh_faithful`) were **never in `AuditAxioms`**, while
the scalar graphs they replaced were. Deleting the scalar lines is what surfaced it; all five are
now audited and 3-axiom clean. Generalisable: *when you replace X with Y, diff the audit coverage of
X against Y — a swap can silently drop a rung.*

**3. ⚠ `lake build Codegen` does NOT build `LeanMlir/Proofs/Codegen/*.lean`.** The `Codegen`
lean_lib's roots are `LeanMlir.MlirCodegen`/`Train`/`Spec`; `ConvNeXtRender.lean` lives under the
`Proofs`/`Certs` roots. After the renderer refactor I ran `lake build Codegen`, got "Build completed
successfully (11 jobs)" and an empty `git diff verified_mlir/` — and **both were vacuous**: the
module never rebuilt, so the `#eval` writers never re-ran. The full build then failed on three real
errors. **A byte-identical artifact check only means something if the writers actually ran** — force
it with `lake env lean LeanMlir/Proofs/Codegen/<Render>.lean`, which elaborates the `#eval`s. This
is §4's "`lake env lean` does not rebuild an edited import" trap wearing the opposite hat.

**4. The renderer's `chLN` flag deletion was inert, as predicted** — bytes identical across all
three artifacts it writes. Two residues the compiler caught: a dangling `else` where a two-line
`if chLN then … else …` collapsed, and a signature line eaten with the flag (`convNextBackAll`'s
`:`). Both loud; neither could have reached an artifact.

#### ✅ Step 1 DONE 2026-07-31 — the FloatBridges are discharged

Scoped 2026-07-31 by reading the files, not by estimating. **Nothing is broken today**: all
3,909 targets build and every scalar-LN theorem is still TRUE. What is wrong is SCOPE — the
float story and two smaller files describe a net the repo stopped shipping when §2m flipped
ConvNeXt to its real channel LayerNorm.

#### ✅ What landed (step 1, 2026-07-31) — UNCOMMITTED

**The channel-LN net now has a whole-net float story in BOTH directions, and the scalar ch9 net
paid nothing for it.** Build green at **3,910** jobs (3,909 + the new module), `verified_mlir/` is
**0 lines of `git diff`** (this touched no codegen), and every new theorem is 3-axiom clean.

| what | where |
|---|---|
| **`LeanMlir/Proofs/Float/ChannelLNFloatBridge.lean`** (new, in `Certs` roots) | the LN op itself, fwd + bwd |
| `floatBridges_transposeFlat` | the transpose **IS** a `gather` — `transposeFlat_eq_gather` is `rfl` |
| `floatBridges_biasAdd` | the `+β` token; magnitude `(1+u)(A+Bb)`, modulus `e ↦ u(A+Bb)+e` |
| `floatBridges_layerNormVec` | `layerNormVec = (+β) ∘ layerScale γ ∘ LN(1,0)` by `rfl` |
| **`floatBridges_chanLNTensor3`** | four gathers around one `FloatBridges.perRow` — `floatBridges_bnPerChannelTensor3`'s blueprint with a transpose inserted |
| **`floatBridges_chanLNTensor3Back`** | same conjugation, `rowLNVecFlatBack` = `perRowIdx(bn_grad_input ∘ diagBack γ)` in the middle |
| **`floatBridges_cnxBodyWith`** / `floatBridges_cnxBlockWith` | the keystone in float form — the block bridge stated ONCE over `cnxBodyWith` |
| `floatBridges_convNextBlock` | now the scalar **instantiation** of it. Same statement, same signature, `rfl`-level — no `unfold` needed |
| `floatBridges_cnxBlockChW` / `floatBridges_convNextStageChK` / `floatBridges_cnxDownChW` | the Ch peers + the two bound bundles |
| **`convnextCh_floatBridges`** | the whole [3,3,9,3] forward at the channel LN, `lnHead := id` |
| **`convnextCh_grad_floatBridges`** | the whole-net backward peer, `lnBhead := id` |
| **`convNextForwardTCh_eq_skeleton`** | ⚠ this was step 2's first bullet — done early, because without it the capstone is about a look-alike |
| `tests/AuditAxioms.lean` | 13 new `#print axioms` lines + the coverage check |

**Three findings worth keeping.**

1. **The §2n keystone held exactly as scoped, and it is the whole reason this was cheap.**
   `convNextBlockBody … = cnxBodyWith (layerNormForward …) …` is `rfl`, so `floatBridges_convNextBlock`
   is now a *term* — `floatBridges_cnxBlockWith M fgelu (layerNormForward …) …` — with no tactic at
   all. One theorem, two worlds, and the ch9 net's statement is byte-identical to what it was.
2. **The backward needed ONE op, not a rewrite.** `ConvNeXtBackFloatBridge.lean` was already
   LN-abstract at all three levels (`cnxBlockBodyBack`, `cnxDownBack`, `convnextInputGrad` each take
   `lnB` as a supplied `FloatBridges`), so the §2m flip costs the whole-net gradient exactly
   `chanLNTensor3Back` plus `id` in the head slot. §2n predicted "the backward peer of all of the
   above"; the measured answer is smaller.
3. **`floatBridges_idVec` moved** from `ConvNeXtWholeFloatBridge` to `ChannelLNFloatBridge` — both
   ConvNeXt bridge files now need it (the head LN slot is `id` in the channel-LN world) and that is
   the file they share. Name unchanged, so nothing downstream moved. ⚠ It is still a duplicate of
   `floatBridges_id` (`MhsaBackFloatBridge`); de-duplicating means editing `AuditAxioms`, which
   prints the other name — not worth the 11-minute rebuild, but it is drift and it is now recorded.

#### ✅ The §B tie — DONE 2026-07-31, right after step 1

The gap step 1 left: `chanLNTensor3Back` was the *hand-composed* reverse chain, with nothing proving
it equals `chanLNTensor3_has_vjp.backward`. **Closed** — three theorems in
`ConvNeXtBackCertifiedTie.lean`, so the float bridge's closeness is closeness to the **certified**
gradient:

* **`chanLNTensor3Back_eq_chanLN_vjp`** — the op tie. Piecewise: the two re-associations collapse by
  the existing permutation-adjoint lemmas (`reassoc{Fwd,Back}_has_vjp_backward_eq`), the two
  transposes by `rfl`, and the row map through `bn_grad_input`;
* **`cnxBodyWithChanLNBack_eq_vjp`** / **`cnxBlockChBack_eq_vjp`** — the body and residual-wrapped
  block, so the channel-LN net's §B coverage now MATCHES the scalar net's, with the LN op covered
  on top.

**Three things worth keeping from doing it.**

1. **The block ties never had to look inside a LayerNorm; this one did.** `cnxBlockBodyBack` takes an
   abstract `lnB`, so the scalar §B could pin it to any certified object it liked. `chanLNTensor3Back`
   is concrete — it has to be, or `floatClose_bnBack` could not run in its middle — so it owed a tie
   the scalar world never owed. That asymmetry is the reason this was a separate piece of work and
   not a copy of `cnxBlockBodyBack_eq_convNextBlockBody_vjp`.
2. **`bn_grad_input` is NOT `rfl`-equal to `(bn_has_vjp …).backward`** — the witness carries a
   `rw [bnForward_eq_compose]` cast. `EfficientNetBackB0`'s `bnBack_faithful_fn` already documented
   this trap at the *graph* level; the function-level peer (`bn_grad_input_eq_vjp_backward`) did not
   exist and is added here. Both sides meet through the canonical `∑ pdiv` form.
3. **`HasVJP.backward_unique`** — any two VJP witnesses for one map have the same backward (both
   `.correct` to the same sum). That is what makes the tie robust to *how* `chanLNTensor3_has_vjp`
   was built: it is tactic-assembled with a leading `unfold`, so its `.backward` does not reduce
   syntactically. Build the same chain in term mode, compute THAT, and transfer. Reusable — any
   future tie against a tactic-built witness can take the same route instead of fighting the term.

**The tie is β-free**, and that is a content result rather than a convenience: the `+β` translation's
VJP is the identity, so neither the float chain nor the certified backward depends on the LN bias.
`chanLNTensor3Back` never took a `β`; now that is proved rather than assumed.

**The decision (Brett's, 2026-07-31): once the bridges are discharged, the scalar full chain
comes OUT.** Not relabelled — deleted. It is off the primary path, and §2a's whole lesson is
that a retired thing which still elaborates is one more thing to drift.

#### ▶ Why this is much smaller than it looks — the float story is ALREADY LN-abstract

`convnext_floatBridges` takes `FloatBridges lnStem`, `FloatBridges lnHead` and every stage and
downsample as **hypotheses**; its own docstring says the LNs enter abstractly *"because
`layerNormForward = bnForward` has the rsqrt keystone"*. `floatBridges_convNextStageK` likewise
takes `∀ i, FloatBridges (layerNormForward …)`. So:

> **`convnext_floatBridges` needs ZERO change.** The channel-LN net instantiates the existing
> theorem at `lnHead := id` — `floatBridges_idVec` is already in that file — with the Ch slots.

#### ▶ The keystone, VERIFIED not assumed

```lean
convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls
  = cnxBodyWith (layerNormForward (c*h*w) εn γn βn) Wdw bdw Wex bex Wpr bpr γls   -- := rfl ✅
```

Checked on device 2026-07-31. `cnxBodyWith` (the LN-abstract body built for the §2m VJP) is
**definitionally** the ch9 representative's body at the scalar LN. So generalising
`floatBridges_convNextBlock` over `cnxBodyWith` + an abstract LN bridge gives **ONE theorem that
serves both the retained ch9 net and the shipped channel-LN net** — ch9's instantiation is free.
Do that first; it is what makes the rest mechanical.

#### ✅ Step 1 — discharge the bridges (DONE — the original scope, for the record)

| item | what it needs | outcome |
|---|---|---|
| `FloatBridges (chanLNTensor3 …)` | the transposes and the reassoc are exact reindexes (permutations — magnitude/modulus pass through); the per-row normalise reuses `floatBridges_bn`'s rsqrt keystone rowwise | ✅ exactly this |
| `floatBridges_convNextBlock` → generic over `cnxBodyWith` | the keystone above; instantiate at scalar (ch9, `rfl`) and at `chanLNTensor3` | ✅ and the scalar one is now a **term**, no tactic |
| `floatBridges_convNextStageChK` / `floatBridges_cnxDownChW` | mechanical mirrors of the scalar peers | ✅ mechanical as predicted |
| `ConvNeXtBackFloatBridge` | the backward peer of all of the above | ✅ **smaller than scoped** — the file was already `lnB`-abstract, so it needed ONE op + `id` |

#### ✅ Step 2 — the drop, and EXACTLY what it did and did not mean (executed; scope below as planned)

**IT MEANS** the full `[3,3,9,3]` scalar-LN chain in `ConvNeXtFullT.lean`: `CnxBlockParams`,
`cnxGls`, `cnxBlockW`, `convNextStageK` (+`_diff`/`_has_vjp`), `CnxDownParams`, `cnxDownW`
(+`_diff`/`_has_vjp`), `CnxTWeights`, `convNextForwardT`/`TC` (+`_has_vjp`/`_eq_chain`/
`_has_vjp_correct`), and the graph section `cnxBlockGraphW` / `cnxStageGraphK` / `cnxDownGraphW`
/ `convNextFwdGraphT` / `convNextFwdGraphTC` + their faithfulness.

⛔ **IT DOES NOT MEAN the ch9 representative** — `convNextForward`, `convNextBlock`,
`convNextBlockBody`, `convnext_has_vjp` in `Architectures/ConvNeXt.lean`. Those are used by
`tests/comparator/{Challenge,Solution}.lean` (**the Project Diderot FV model**),
`apps/imagenette/MainConvNeXtVerified.lean`, `ConvNeXtClose`, `StableHLO`, `EfficientNet.lean`
and `ViTFwdGraph.lean`. **Deleting them would void a book chapter and the comparator.** This is
the `cifar8w` lesson (§2i) one net over: *check what a thing backs before pruning it.*

**The measured dependency table** — every external reference to the scalar full chain, so the
drop is a checklist rather than a search:

| symbol | external refs | where |
|---|---|---|
| `convNextFwdGraphT`, `cnxStageGraphK` | **0** | already dead |
| `convNextFwdGraphTC`, `cnxBlockGraphW`, `cnxDownGraphW` | docstrings only | `tests/TestConvNeXtTTrainPC.lean` ✅ already repointed to the `…Ch` names |
| `convNextForwardTC` | 3 | `VerifiedNets` docstring, `AuditAxioms`, `SpecVJP`'s §2m warning — **no live consumer** |
| `convNextForwardT` | 13 | `WholeNetForwardTies`, both Float bridges, `AuditAxioms`, lakefile **comments** |
| `CnxTWeights` | 3 | `SpecVJP`, `WholeNetForwardTies`, `AuditAxioms` |
| `CnxBlockParams` / `cnxBlockW` / `convNextStageK` | 5/3/9 | `ConvNeXtWholeFloatBridge`, `WholeNetForwardTies`, lakefile comments |
| `CnxDownParams` / `cnxDownW` | 3/9 | + `ConvNeXtBackB0.lean` (164 lines — `cnxDownBackGraph` ties the emitted backward to `cnxDownW_has_vjp`'s backward; needs a Ch peer) |

So the drop is: ~~**step 1 first**~~ ✅ done, ~~then `WholeNetForwardTies.convNextForwardT_eq_skeleton`
→ a Ch peer~~ ✅ **`convNextForwardTCh_eq_skeleton` landed with step 1** (`lnHead := id` + the
`_eq_chain` unfold, exactly as scoped — 5 lines), then `ConvNeXtBackB0` → Ch, then delete, then
sweep `AuditAxioms` / the lakefile comments / `VerifiedNets`'s and `SpecVJP`'s docstrings.

**▶ The four float-side declarations that die WITH the scalar chain** — measured 2026-07-31 while
building step 1, because the table above names files and the drop needs names:

* `floatBridges_cnxBlockW` (takes a `CnxBlockParams`), `floatBridges_convNextStageK`
  (`convNextStageK`), `floatBridges_cnxDownW` (`CnxDownParams`/`cnxDownW`) —
  `ConvNeXtWholeFloatBridge.lean`; each has a live `…Ch` peer now, so deleting them loses nothing;
* `convNextForwardT_eq_skeleton` — `WholeNetForwardTies.lean`; its Ch peer exists.

⛔ **Three things in that file must SURVIVE the drop** and it is easy to get wrong: `convnextForward`
(the skeleton takes its blocks abstractly — it is LN-world-neutral), `convnext_floatBridges` (same),
and `floatBridges_convNextBlock` / `floatBridges_cnxBodyWith` (the ch9 representative's block, which
§2n's own ⛔ keeps).

**`AuditAxioms.lean` loses 17 `#print axioms` lines** (counted, not estimated): 13 in the
`ConvNeXtFullT` block — `convNextStageK_has_vjp`, `cnxDownW_has_vjp`, `convNextForwardT_has_vjp`
/`_eq_chain`/`_has_vjp_correct`, `cnxBlockGraphW_faithful`, `cnxStageGraphK_den`,
`cnxDownGraphW_faithful`, `convNextFwdGraphT_faithful`, `convNextForwardTC_has_vjp`/`_eq_chain`/
`_has_vjp_correct`, `convNextFwdGraphTC_faithful` — plus the three float bridges and
`convNextForwardT_eq_skeleton`. Sweep them in the same commit or the audit stops elaborating.

#### ✅ A THIRD stale-scope file, found while doing step 1 — LABELLED 2026-07-31, not ported

**Decision: it carries a scope label, not a port** — and the reason is measured, not stylistic.
Re-pointing the ch7 §1a tie at the channel LN is *not* a relabel: the 22 LN sites would need
vector-γ/β param certs (ViT has that shape — `vit_render_vecln{gamma,beta}_certified` — ConvNeXt
has only the `Vec 1` scalar embedding, `ConvNeXtClose` §C), the head-LN tie would be deleted rather
than ported, and the whole forward thread would have to be re-derived through the transpose/reassoc
conjugation. That is chapter-sized. So `ConvNeXtTiePoC.lean` now opens with a ⚠ that says exactly
what it ties (`chLN := false`, plus a head LN), that every theorem in it is still TRUE, and that it
should be read as the ch9-representative §1a tie — the same status
`Architectures/ConvNeXt.lean` has as the ch9-representative forward. Its two other "the committed
render" claims are corrected in place.

⚠ **`ConvNeXtRender.lean:113` was stale in the same direction and is fixed**: it called
`chLN := false` *"the committed scalar-global `.bnF`"* while the flag has defaulted to `true` since
§2m. A reader auditing the LN world from the renderer down would have been told the wrong mode is
the shipped one. Nothing in `ConvNeXtFaithfulPoC`/`ConvNeXtClose` needed touching — their
"scalar-LN γ/β" lines describe the *ops*, which still exist and are still certified, not the net.

The original finding, for the record:

**`LeanMlir/Proofs/Architectures/ConvNeXtTiePoC.lean` — the ch7 §1a whole-net tie
(`CnxTiePoC.cnx_net_tied_certified`, 6 audit lines) is written against the SCALAR LN, and it still
has a HEAD LN** (`hng`/`hnbt` at the GAP output, line ~219) — the site §2m deleted. Its own
docstring says it feeds "the **real forward activations** of the committed
`convNextTrainStepFaithfulV` render"; that render now takes `chLN : Bool := true`
(`ConvNeXtRender.lean:532`), so **the tie describes the `chLN := false` spelling — the mode the repo
stopped shipping.** The theorem is still TRUE (it is stated over its own local forward defs) and
`grep` says it references **none** of the dying symbols, so *the drop cannot break it* — which is
exactly why it will be missed. Decide before the drop lands: re-point it at the Ch chain, or label
it explicitly as a ch9-representative tie the way §2n's ⛔ labels `Architectures/ConvNeXt.lean`.
This is §2a's lesson again — a thing that still elaborates is not a thing that still describes what
ships.

#### ▶ Gates for the drop

* `lake build Proofs Certs Codegen` green — **3,910 jobs** as of step 1 (was 3,909; +1 module);
* **`verified_mlir/` is 0 lines of `git diff`** — this is a proof-side deletion and must move no
  artifact. If bytes move, something in the drop was not inert;
* `regen_verified_mlir.sh check` still 67 artifacts / one writer each;
* the **capstone parity** re-run (`render_parity.py`, 180/180 bit-identical) — it is the cheapest
  end-to-end confirmation that nothing in the delete reached the shipped graph;
* `#print axioms` on the surviving Ch theorems still 3-axiom clean.

#### ▶ Traps, all already paid for elsewhere

* **`lake env lean tests/X.lean` does NOT rebuild an edited import** (§4). Deleting declarations
  in `ConvNeXtFullT.lean` and then running the capstone without a `lake build` first exercises
  the OLD oleans and tells you nothing.
* **`render_parity.py` needs `iree-run-module`**, which is NOT in this repo's `.venv/bin` (only
  `iree-compile` is) — it is in `../lean4-mlir/.venv/bin`. Put both on PATH.
* **`AuditAxioms.lean` is the heaviest module in the repo** (`vit-back-b0` ~11 min / ~14 GB —
  see the memory note); budget for it when the sweep touches it.
* A deleted `noncomputable def` that something references fails at `lake build`, loudly — but a
  stale **docstring** does not, and this section's own dependency table is mostly docstrings.

---

### 2o. ▶ NEXT SESSION — the channel-LN BACKWARD capstone, then ConvNeXt's 80-epoch re-run

Two independent pieces of work, scoped 2026-07-31 by measurement. **Do them in the order below**:
Part B is a ~2-hour GPU run and Part A is CPU proof work, so **start the run first and prove while
it burns**. They do not touch the same files.

---

#### ✅ Part A — DONE 2026-07-31. The channel-LN backward capstone, and it went one level further

**Landed** in `ConvNeXtBackB0.lean` (+1 import, `ConvNeXtBackCertifiedTie`) — six new declarations,
all **3-axiom clean**, `AuditAxioms` **1503 → 1509** directives and the coverage checker green:

| new | what |
|---|---|
| `Proofs.rowLNBack_affine_eq` | the backward peer of `rowLN_affine_eq` — the emitted `rowScaleF γ` on the COTANGENT is the per-row `diagBack γ` that `rowLNVecFlatBack` folds in, so the LN gradient runs at `γ = 1`. β-free |
| `chanLNBackGraph` + `_faithful` | **the keystone** — the `den`-level peer of §2m's `chanLNGraph_faithful` |
| `chanLNBackGraph_eq_vjp` | chains it through §B onto `(chanLNTensor3_has_vjp …).backward` |
| `cnxBlockBodyChBackGraph_faithful` | block body ↔ `cnxBodyWith_has_vjp` at the shipped LN |
| `cnxResidBlockChBackGraph_faithful` | the residual block ↔ `cnxBlockChW_has_vjp` |
| `cnxDownChBackGraph_faithful` | **the downsample §2n dropped, restored at the channel LN** |

Gates: `lake build Proofs Certs Codegen` **3,910 green** · `git diff verified_mlir/` **0 lines** ·
`regen_verified_mlir.sh check` **67 artifacts, one writer each** + all four content audits OK.
Every capstone lands on the CERTIFIED VJP, not on a hand-composed chain — that is what tying §B
first bought, and `chanLNBackGraph_eq_vjp` is the single place it is spent.

⚠ **Two things that cost time and would cost it again:**

1. **`lake env lean` typechecks but does NOT write the `.olean`.** `tests/AuditAxioms.lean` then
   reads the STALE one and reports `unknown constant` for every new theorem — which reads exactly
   like "my declaration doesn't exist". Run `lake build` before the audit, always. (Same family as
   §2n's `lake build Codegen` trap and `[[lake-exe-cache-gotcha]]`.)
2. **`simp only` will not fire `chanLNBackGraph_eq_vjp`; `rw` will.** The graph's `c h w` are
   implicit and appear only inside `Vec (c * h * w)`, so simp gives up unifying them and silently
   leaves the term alone — the whole rewrite chain below it then stalls and the failure surfaces as
   a confusing `rfl` error about `id { backward := … }`. The **`unusedSimpArgs` linter names the
   culprit outright**; read it. Fix: `unfold` the graph def, `rw` the LN step, then `simp only` the
   rest. Both capstones needed this.

Below is the original scoping, kept for the reasoning.

**The measured state after §2n.** `regen`-level and math-level coverage of the shipped channel-LN
net was complete; the `den`-level backward was empty:

| shipped (channel-LN) net | `den`-level coverage |
|---|---|
| **forward** | ✅ `chanLNGraph_faithful` → `cnxBlockChGraphW_faithful` → `cnxStageChGraphK_den` → `cnxDownChGraphW_faithful` → `convNextFwdGraphTCh_faithful` (all five, audited as of §2n) |
| **backward** | ~~❌ **nothing, at any level**~~ ✅ **CLOSED above** — op, block body, residual block, downsample |

⚠ **This is wider than the §2n commit message says.** That message says "the channel-LN *downsample*
has no graph-side backward capstone". Measured afterwards: the two capstones that SURVIVED in
`ConvNeXtBackB0` (`cnxBlockBodyBackGraph_faithful`, `cnxResidBlockBackGraph_faithful`) are over
`convNextBlockBody` — **the ch9 representative's SCALAR LN** — and `ConvNeXtChainClose` is scalar-LN
too. The only `den`-level declarations touching `chanLNTensor3` anywhere are the five FORWARD graph
theorems plus `den_reassocS`. So the correct statement is **the channel-LN net has no `den`-level
backward capstone at all**. The drop did not cause that; it removed the scalar capstone that was
making the column look populated.

**The missing keystone is ONE definition + ONE theorem.** §2m built `chanLNGraph` +
`chanLNGraph_faithful` (forward); its backward peer was never built. Everything above it — block,
downsample — is assembly on top, exactly as `ConvNeXtBackB0` assembles the scalar ones.

**Every piece it needs already exists**, which is why this is small:

| piece | where | note |
|---|---|---|
| `lnRowBack_faithful` | `StableHLO.lean:2250` | `rfl`. Denotes `rowLNBackFlat m n ε γ x dy = Mat.flatten (fun i => bn_grad_input n ε γ (unflatten x i) (unflatten dy i))` — which is **literally `rowLNVecFlatBack`'s shape**, modulo folding `diagBack γ` in |
| `transposeF_faithful`, `rowScaleF_faithful` | `StableHLO.lean` | the other three ops of the emitted subtree |
| `den_reassocS` / `den_unassocS` | `ConvNeXtChannelLN.lean` | the `▸`-transport bridge (§2m's seam) |
| `chanLNTensor3Back_eq_chanLN_vjp` | `ConvNeXtBackCertifiedTie.lean` | §B, landed 2026-07-31 — the certified target to chain onto |

**The recipe.** Mirror `chanLNGraph_faithful`, whose whole proof is six `rw`s and an `rfl`:

1. `chanLNBackGraph` — the subtree `ConvNeXtRender.lnBackSite` already emits: `transposeF(reassoc x)`,
   `transposeF(reassoc cot)`, `rowScaleF γ`, `lnRowBack "%one" … 1`, `transposeF`, `unassoc`;
2. a `rowLN_affine_eq` peer for the backward — `rowLNBackFlat s c ε 1 X (rowScaleFlat s c γ dy)
   = rowLNVecFlatBack s c ε γ X dy` (per row: `rowScaleFlat` is `layerScale γ` = `diagBack γ`, so
   this is the same fold `rowLN_affine_eq` does forward);
3. `chanLNBackGraph_faithful : den (chanLNBackGraph …) = chanLNTensor3Back c h w ε γ x (den e)`;
4. **then chain through §B** so the public statement lands on `(chanLNTensor3_has_vjp …).backward`
   rather than on the hand-composed chain — that is the whole point of having tied it;
5. the block + downsample Ch capstones in `ConvNeXtBackB0`, restoring what the drop removed at the
   LN the net actually uses. `ConvNeXtBackB0` already imports `ConvNeXtFullT`.

**Gates:** `lake build Proofs Certs Codegen` green (3,910 today) · `verified_mlir/` 0 lines of diff
(this adds proof, emits nothing) · `#print axioms` 3-axiom on each new theorem · **add them to
`AuditAxioms`** — §2n's own lesson was that §2m's five forward graph theorems sat unaudited for a
week because nobody diffed audit coverage across a swap.

---

#### ✅ Part B — DONE 2026-07-31. The re-run landed, and the DP ratio survives

```
final (epoch 80)   84.41%  (3313/3925)      wall  1h54m01s (6841 s)
best               84.82%  (epoch 69)       rc=0 · epoch marker 80 · 80 epochs logged
runs/convnext_chln_80ep_jul31.log           ckpt kept as convnext_adam_ckpt_xla.bin.chln80-jul31
```

Ran **fresh** (the epoch-12 checkpoint moved aside), so the row is apples-to-apples with the other
four. Against the VOID scalar-LN numbers — 82.75% / 82.98% / 1h56m — that is **+1.66 points final,
+1.84 best at the same wall clock**. §0b's table is updated and the VOID note is gone.

**Read the gain correctly: the overfitting diagnosis SURVIVES, and is why it is only 1.7 points.**
Train loss still lands at **0.502** — the same floor the scalar net reached — with ~28M params
against 9,469 images, val peaking at epoch 69 and flat over the last ten. What the channel LN bought
is a better-conditioned optimisation, not more data: **69.66% by epoch 7**, where the scalar-LN run
needed 12 epochs to reach 42.88%, and then the same dataset-size ceiling. What IS retired is the old
"ConvNeXt is the outlier" reading, which rested on a number belonging to a net the repo dropped.

Below is the original scoping.

**Why:** §2m changed the function at 21 of 22 LN sites, so the **82.75%** in §0b's table was VOID —
it belonged to a net the repo no longer ships. The other four rows are clean 80-epoch runs on their
certified bytes; ConvNeXt needed to rejoin them.

⚠ **THERE IS A LIVE CHECKPOINT AT EPOCH 12 RIGHT NOW, and it is not stale junk — read this first.**

```
.lake/build/convnext_adam_ckpt_xla.bin        333,915,384 B   Jul 31 17:54
.lake/build/convnext_adam_ckpt_xla.bin.epoch  = 12
runs/convnext_chln_smoke_jul31.log            Epoch 12/80: loss=1.865542 lr=0.000967
                                              epoch 12: val_acc = 1683/3925 = 42.878981%
```

This is §2m's channel-LN run, **interrupted at epoch 12** — same artifact, same 80-epoch cosine
schedule (the `lr=0.000967` is consistent), params + Adam m/v all in the file. So it is a genuine
choice, not the §4 stale-checkpoint trap:

* **Fresh (recommended)** — move it aside, run all 80. Comparable with the other four rows, which
  are uninterrupted. ~1h56m.
* **Resume** — ~68 epochs, ~1h39m. Defensible (same bytes, same schedule) but the table does not
  record "resumed", and nobody has checked whether the data-order seed depends on run start. If you
  take it, **say so in the log line**.

Either way, `cat .lake/build/convnext_adam_ckpt_xla.bin.epoch` BEFORE launching. The neighbouring
`*.prechln-jul31` marker holds **80** — restore that by accident and the run is a silent no-op,
which is exactly §4's trap and it has already bitten three nets once.

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so   # if older than pjrt_ffi.c
lake build convnext-verified-adam-xla
mv .lake/build/convnext_adam_ckpt_xla.bin{,.epoch} /tmp/           # the FRESH option
HIP_VISIBLE_DEVICES=0 .lake/build/bin/convnext-verified-adam-xla data \
  2>&1 | tee runs/convnext_chln_80ep_<date>.log
```

**What to record when it lands:** final (epoch 80) and best val, wall clock, `rc=0`, and that the
epoch marker really reads 80 (not a resumed no-op). Then update **§0b's five-net table** — replace
`ConvNeXt-T | 82.75% | 82.98% | 1h56m` with the new row and drop the VOID note in §0/§2m/§2n.
Expect the accuracy to move: the net genuinely changed. ConvNeXt was the lowest of the four CNNs
and the §0b reading was *overfitting* (~28M params on 9,469 images, curve flat from epoch 40); the
channel LN is a better-conditioned normaliser, so up is plausible, but **do not predict a number in
the doc — record the measured one**.

**Companion ✅ DONE 2026-07-31 — the DP ratio survives the flip at 1.70×.** Re-measured on the
channel-LN bytes, train-only on both sides, solo on the box, 4 epochs each
(`runs/convnext_chln_dp{1,2}_jul31.log`):

| | marginal epoch | deltas | pre-flip (§2h-quater) |
|---|---|---|---|
| 1 GPU, bs 32 | **78.0 s** | 78, 78, 78 | 77.5 s |
| **2 GPU DP, global 64** | **46.0 s** | 46, 46, 45 | 46.0 s |
| **ratio** | **1.70×** | | 1.68× |

Within noise of the pre-flip number and of the **1.67×** EfficientNet (§2e-ter) and mnv2 (§2h-ter)
reach, so §2h-quater's conclusion is unchanged: the shortfall from 2× is **structural** (host-resident
parameters, §2c/§2d.3), not LayerNorm-shaped. The channel LN costs nothing at the scaling level.

⚠ **Two traps this 10-minute job hit, both worth knowing before any bench re-run:**

1. **The checkpoint path is per-VARIANT** — `{slug}_{variant}_ckpt_xla.bin`, so the DP side reads
   `convnext_adamdp_ckpt_xla.bin`, NOT the `_adam_` one. Clearing only the `adam` checkpoint left a
   stale `adamdp` marker at **epoch 7**; with `LEAN_MLIR_MAX_EPOCHS=4` the loop is `[7:4]`, which
   runs **zero epochs and exits `done` with rc=0**. That is §4's silent no-op wearing a new hat —
   the only symptom was `▸ resuming from checkpoint at epoch 7` in the banner and an empty timing
   table. Clear BOTH paths, and read the resume line.
2. **`LEAN_MLIR_SKIP_EVAL=1` does NOT suppress the `  epoch N:` line** `marginal_epoch.sh` keys on
   — the eval loop just iterates zero times and the line prints with `val_acc = 0/3925 = 0%`. So
   train-only timing works, and a `0.000000%` in one of these logs is expected, not a broken run.

`LEAN_MLIR_MAX_EPOCHS=4` is the clean way to bound these (≥3 epochs are needed for the marginal);
no `timeout`/kill is required, and `| head` would not stop the trainer anyway.

---

### 2p. ✅ ViT-Tiny / full ImageNet-1k — BUILT AND GATED 2026-08-01, NOT RUN

The ViT peer of §2k, stood up as a code/test pass. **No long run yet, deliberately.** Everything
below is measured on ares (4 × RTX 4060 Ti of the box's 6).

`lake build vit-imagenet-verified-xla`. Three artifacts under slug **`vitin`**
(`adam128_train_step`, `adamdp128x4_train_step`, `fwd`) at `nClasses := 1000`, `bs := 128`;
`vitImagenetVerified` in `VerifiedNets.lean`; `ViTImagenetCommon` + a `-xla` main.
**No renderer change** — `nClasses`, `bs` and `replicas` were already parameters, so it is three
`#eval`s, exactly as §2k found for R34.

Batch is **128 per device × 4 = global 512**, which is `vitTinyImagenetConfig.batchSize`. Matching
the reference's global batch is what makes the pair comparable rather than two experiments.

#### The gates — all four pass

| gate | result |
|---|---|
| **param count vs the JAX reference** | **5,717,416 == 5,717,416**, exact (`vitTinyImagenet.totalParams`) — the check §2k built after R34's two "ResNet-34"s turned out to be different nets |
| **gate 1** — the ten `vit_*` artifacts re-render | byte-identical; only the three `vitin_*` paths appear |
| **`vit-dp-check`** at `VIT_DP_NET=vitin VIT_DP_REPLICAS=4` | **bit-exact 17,152,251 / 17,152,251**; sum-not-mean control fires at **2.96** |
| **residency**, 4 replicas | **0 of 68,608,992 bytes**, floor 0, both controls fire (init 32,416,743 · staleness 49,332,018) |
| **end-to-end** | 4 GPUs, real ImageNet over the shim, `rc=0`, 2502 steps/epoch, loss 7.84/7.96/7.79 |

⚠ The **label-smoothing constant is right at K = 1000** — `-0.000100`, and `-0.010000` occurs zero
times. That is §2k's real bug (`α/K` hardcoded for K=10, *on the gradient path*, caught only
because loss ≈ 87 was implausible against ln(1000) = 6.9) checked rather than assumed. `%loss`
reads **7.99** at init here, which is where it belongs.

#### ⛔ THE PROJECTION WAS WRONG, AND THE CAUSE IS UNEXPLAINED

This section first estimated 264 ms/step by carrying the Imagenette ViT's measured rate over,
on the reasoning that 224² ViT-Tiny costs the same per step regardless of class count. **Measured,
it does not:**

| net | classes | ms/step (4×bs128, synthetic) |
|---|---|---|
| `vit_adamdp128x4` | 10 | **274** |
| `vitin_adamdp128x4` | 1000 | **424** |

**+150 ms, 1.55×, for a head that is 3.4% of the parameters** (5,526,346 → 5,717,416). The two
renders are *structurally identical* — 15,859 lines each, 435 `dot_general`, 2 `convolution`, 438
`reduce` — so it is not an op-count explosion; only tensor dimensions move. **The cause is not
identified.** Before committing multi-day wall clock, dump post-optimisation HLO (the §4
throwaway-shim recipe, since `$XLA_FLAGS` is inert here) and read what the head's `dot_general` and
the CE reduce lower to. A candidate worth checking first: `pretty` has **no CSE** (§4), so a
cotangent subtree consumed more than once is emitted more than once, and at `[128, 1000]` the
intermediates are 100× the Imagenette ones.

**ETA on the measured number**: 424 ms × 2,502 steps = 17.7 min/epoch train-only ⇒ 300 epochs ≈
**88 h ≈ 3.7 days**, plus eval. (The reference sets `valEveryEpochs := 5` for exactly this reason;
this driver evals every epoch and does not have that knob.) If the 1.55× is recovered it is ~2.4
days.

#### ⛔ AND THE LOADER IS **NOT** THE BOTTLENECK — a second refuted prediction

This section also predicted ViT would be the first loader-bound config (~1,940 img/s wanted against
~1,530 delivered). **Refuted by the synthetic control**, which is the measurement that settles it:

| | ms/step |
|---|---|
| synthetic inputs (no loader at all) | **424** |
| real ImageNet, `SHIM_WORKERS=1` | 427 |
| real ImageNet, `SHIM_WORKERS=2` | 432 |
| real ImageNet, `SHIM_WORKERS=4` | 436 |

Within noise of each other, and *slightly worse* with more workers — so at 424 ms/step the demand
is only ~1,200 img/s, which one producer already covers. The arithmetic that predicted otherwise
used the 264 ms/step that turned out to be wrong; **the loader ceiling was an artefact of the bad
compute estimate**, which is the same "one number carried into a new regime" mistake this file
records for the 1.04×-vs-JAX claim (§3). Sharding is built, gated and inert at `SHIM_WORKERS=1` —
keep it for when a faster render or bf16 changes the balance, but **do not set it today**.

#### ▶ ALL FIVE ImageNet trainers — the scoreboard (2026-08-02)

`vitin` is written up in detail above; the other three followed the same recipe. **No descent runs
yet** — every smoke is 40 steps.

| net | target | params (verified == JAX) | DP gate | residency | ms/step (4 GPU) | steps/ep |
|---|---|---|---|---|---|---|
| `resnet34in` | `mom256`/`momdp64` | **21,797,672** | ✅ (cifar8 proxy, §2b-quater) | ✅ | 386 (4×64) | 5004 |
| `vitin` | `adam128`/`adamdp128x4` | **5,717,416** | ✅ bit-exact 17,152,251 | ✅ | 424 (4×128) | 2502 |
| `cnxin` | `adam`/`adamdp` | **28,587,592** | ✅ bit-exact 85,762,779 | ✅ | 270 (4×32) | 10009 |
| `enetin` | `adam64`/`adamdp64` | **5,288,548** | ✅ **sharding** 2.06e6 sep | ✅ | 310 (4×64) | 5004 |
| `mnv2in` | `adam64`/`adamdp64` | **3,504,872** | ✅ **sharding** 2.18e6 sep | ✅ | 294 (4×64) | 5004 |

**What each needed, and it varied a lot:**

* **ViT** — nothing but `#eval`s; `nClasses`/`bs`/`replicas` were already parameters.
* **ConvNeXt** — `nClasses` was a hardcoded literal in ~20 places and `−α/K` was a string
  independent of it. `cBS` is STILL a private constant in 96 places, which is why `cnxin` renders at
  batch 32 (global 128, 10,009 steps/epoch). **Threading `cBS` would roughly halve its 60-hour run**
  and is the single best pre-run optimisation available.
* **EfficientNet / mnv2** — `B` and `nClasses` were already parameters; both needed a `slug`
  (entry names were baked, and mnv2's forwards live in a DIFFERENT file from its train step), and
  both carried the K=10 constant. These two are the batch-BN nets, so they have `_fwd_eval` peers
  and are gated by `shard-check` rather than the duplicated-batch harness.

⚠ **The variant names encode the PER-DEVICE BATCH, not the replica count** (`enetAdamVariant`,
`mnv2AdamVariant`, `r34AdamVariant` all do this), so `adamdp64` would name both a 2- and a
4-replica render at B=64. Only the 4-replica ones exist. Anyone adding a 2-replica peer must rename
first or it is a silent clobber.

#### The shim sharding, since it exists now

`SHIM_SHARD=i/N` in the generated shim (`ds.shard` on the RAW dataset, before the map, so JPEG
decode is what parallelises), plus `spawnShimSharded` / `readShimBatchRR` behind `$SHIM_WORKERS`
(default 1). Gated three ways:

* **inert when unsharded** — `SHIM_HASH` returns `3da05cb8…`, byte-identical to pre-change;
* **shards are disjoint, covering and order-preserving** — `interleave(shard 0/2, shard 1/2)`
  reproduces the unsharded stream element-for-element over 240 elements, same at N = 4, against a
  control that the shards differ from each other. That is the check that a "shards hash
  differently" gate would pass while silently dropping half the data;
* **`SHIM_WORKERS=1` is the old path end to end** — R34/ImageNet steps normally at 926 ms.

⚠ Round-robin over **batches** is not the unsharded stream: `ds.shard` interleaves *elements*, so
batch composition differs. Both are valid shuffled streams; they are not byte-comparable, so a
determinism hash does not carry across a shard-count change.

⚠ Measured throughput, for the record (bs128, marginal so TF startup is out): **1,527 img/s**
single producer, 2 processes **1.71×**, 4 processes **2.36×** on 32 cores. And
`enable_op_determinism()` is **free** — 1,527 with it against 1,463 without, i.e. ON is marginally
*faster*. That refutes the obvious "determinism is throttling AUTOTUNE" hypothesis and confirms
§2k's "costs little".

#### ⚠ What this is NOT: the DeiT recipe

`jax/MainVitImagenet.lean`'s `vitTinyImagenetConfig` targets DeiT-Ti's ~72.0% with a suite the
verified path does not have. The split is clean:

| | verified path | why |
|---|---|---|
| RandAugment (geometric, m9/mstd0.5/inc1), random erasing, repeated aug 3×, RRC+hflip | ✅ **free** | they live inside `build_imagenet_iter`, which the shim reuses verbatim — flipping them in the config moves both paths |
| **mixup + cutmix** | ❌ | applied in the JAX *train step* (`_mixup`/`_cutmix`), not the pipeline, and they emit **soft labels `float[B,NC]`** where the shim wire is `int32[B]` **and** this render's cotangent is smoothed-CE over a one-hot. Needs a wire v2 **and** a `softLabelCE` cotangent |
| **stochastic depth** (`dropPath := 0.1`) | ❌ | architectural — needs the renderer |
| **EMA** (`emaDecay := 0.99996`) | ❌ | driver-level |
| **grad clip 1.0** | ❌ | which the reference config calls *"the unlock for the 5e-4 LR"* |

The reference's own **grad-clip-only 80-epoch ancestor reached 65.6%**, and this has less than that.
**Do not compare a number from this run to 72.0%.** `softLabelCE` already exists as a worked design
in the *unverified* track (`LeanMlir/Types.lean:349`, `LeanMlir/Train.lean`), so the mixup/cutmix
half is a port with a precedent rather than a research problem — but note that track is a different
one, and `grep -rn 'softLabel|mixup|cutmix' LeanMlir/Proofs/ VerifiedTrain.lean VerifiedNets.lean`
returns **nothing**.

#### Traps hit while building this

* **The `MAX_STEPS` probe trap fired TWICE in one session**, both times because the step ceiling is
  tiny at large global batch: 18 steps/epoch at global 512 on Imagenette, so `MAX_STEPS=20` and
  `=40` both silently became full 80-epoch runs (and the first left an `epoch=80` checkpoint, i.e.
  §4's silent-no-op armed). **Compute `nTrain/(bs·replicas)` before choosing the cap**; the probe
  warms 8, so the usable window at global 512 on Imagenette is 9-18.
* **A skipped eval prints `val_acc = 0/N = 0.000000%`** (`VerifiedTrain.lean:810` loops over
  `[0:0]` and prints `correct/total` anyway) — indistinguishable at a glance from a collapsed net.
* **§2k's recorded "batch 0 labels" `[26, 948, 227, 24, 582, 614, 141, 155]` are the VALIDATION
  split's**, not the train split's (train unshuffled starts `[196, 40, 522, …]`, shuffled
  `[633, 304, …]`). No bug — `trainAdamSched` does spawn `"train"` (`VerifiedTrain.lean:693`) and
  the val drain is separate — but §2k reads as if the *train* stream had been verified against the
  producer, and it has not been. The framing/preamble/`readExact` evidence does carry.

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
3. **Device-resident parameters — ▶ SCOPED IN §2d.3, and its prescribed first measurement is now
   DONE (2026-07-30). Read §2d.3 before starting.** Two rounds of transfer work are
   already done (batching: 256→205 ms; killing the per-step host memcpys: 205→162 ms). What remains
   is smaller than it looks — see §3. ⚠ **"the payoff is multi-GPU only; at bs256 on one GPU it is
   worth ~nothing" was this item's headline and it is REFUTED by measurement**: the param round trip
   is **55% of a bs32 step on ONE GPU** (88 ms of 160) and the prize is *small-batch*, not
   multi-GPU — ~2.2× at bs32 on one GPU and on two, 1.15× at bs256. The cost is dominated by
   **per-buffer overhead** (513-887 param buffers/step), not bandwidth, so it scales with parameter
   *count* and is roughly uniform across nets. **FIVE independent multi-GPU measurements also point
   here** (§2c 1.46× on R34,
   §2e-ter's measured 13–16% per-step DP overhead on EfficientNet, mnv2's **1.67×**, ConvNeXt's
   **1.68×** and **ViT's 1.62×** end-to-end — four architecturally unlike nets landing on the same
   ratio, and the fourth is a TRANSFORMER WITH NO CONVOLUTIONAL BACKBONE, which is what makes the
   ceiling structural rather than net-specific), so it is the
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

### 2d.3a. ⚠ CORRECTIONS from the CUDA box — measured 2026-08-01 on ares

Two claims below were measured on the 7900 XTX and written as if they were properties of the
*code*. Both are properties of the **vendor**. Read them before planning off §2d.3.

**1. `PJRT_FFI_PINNED` is NOT refuted — it was untestable on CUDA until 2026-08-01.** The pinned
arena only ever `dlopen`'d `libamdhip64`/`hipHostMalloc`, so on NVIDIA the flag printed
"no libamdhip64", fell back to the direct path, and *looked* supported while doing nothing. With the
`libcudart`/`cudaHostAlloc` leg added, on the 2-replica EfficientNet DP bench:

| | DP overhead | parallel efficiency | effective |
|---|---|---|---|
| unpinned | 27 ms/step | 81.88% | 2.22 GiB/s |
| **pinned** | **19 ms/step** | **86.81%** | **3.15 GiB/s** |

So the standing advice "leave it off, it is a 17% regression" is a **ROCm** result. On CUDA it is
roughly a 17% win in the other direction. ⚠ Single A/B pair — indicative, not anchored; the ROCm
refutation was the more careful measurement. Anyone re-anchoring should take medians on both boxes.
This does not weaken its role as the gate's second transport: it stays numerically inert either way.

**2. The Lean half of phase 2 is SMALLER than "~150 lines / 7 loops / 11 call sites" for the path
that matters.** Reading the actual call path (`trainAdamSched` → `IreeSession.mlpTrainStepV` →
`lean_iree_mlp_train_step_v` → `iree_ffi_invoke_f32`) settles the shape:

* inputs are `[x, params×n, onehot]`; outputs are `[θ'|m'|v'|loss|bc1|bc2|bn…]`, and the **param
  inputs correspond 1:1 to the leading param outputs**. §2d.3's "retain the outputs, pass them as the
  next call's inputs" is therefore literally an index mapping, not a search.
* **per step the host reads only two small things**: `F32.read out (3*nParams)` for the loss, and,
  when `hasBn`, an `nBnStats`-float slice. Everything else it does with `out` is `pbuf := out`, i.e.
  hand it straight back. The 272 MB `[θ|m|v]` prefix crosses PCIe twice per step *solely to be
  re-uploaded*.
* the epoch boundary ALREADY isolates the host's real need — `thetamv := pbuf.extract 0 mvBytes`,
  commented "One 272 MB copy per EPOCH (for eval + checkpoint), not per step". That is exactly
  `resident_read`'s call site, already written and already once-per-epoch.

So for `trainAdamSched` the change is: keep the handle, copy back only the tail outputs per step,
and point the existing per-epoch extract at `resident_read`. The "7 loops / 11 call sites" figure
counts loops that §2d.3 itself then tells you to leave alone (E4M3, `train`, `trainLinear`).

**Sequencing consequence:** do single-replica end-to-end FIRST. It is gate-testable on its own, and
§2d.3's own measurement (the parameter round trip is 55% of a bs32 step) says it is most of the
win on one GPU. The DP path — where the 4-GPU money is — lands second, since that is the part
§2d.3 correctly calls "the fiddly part".

**Why 4 GPUs needs this more than the doc assumes** (measured on ares 2026-08-01): the box has **no
P2P between any pair** (`nvidia-smi topo -p2p r` is `CNS` across the whole matrix), so every
`all_reduce` already stages through host memory, and PCIe is Gen3 x8. The `[θ|m|v]` push is
**O(N−1)** while compute is O(1) per replica, so measured 81.9% at 2 replicas projects to ~60-69%
at 4. The JAX reference on the same four cards gets ~4.10× because its parameters are
device-resident and only gradients cross. That gap IS this work item.

##### ▶ 4 GPUs is now MEASURED on the real net, and it is worse than the projection

The projection above said ~60-69% at 4 replicas. The real R34/ImageNet number, 2026-08-01, on the
new `resnet34in_momdp64` render (4 × bs64 = global 256, so the same global batch and 5004
steps/epoch as the single-device peer):

| config | ms/step | min/epoch | 30 ep |
|---|---|---|---|
| 1 GPU, bs256 | 914 | 76.2 | 38 h |
| **4 GPU, 4×64** | **625** | **52.1** | **26 h** |

**4 GPUs buys 1.46× — 36.6% parallel efficiency.** Per-replica compute fell 4× while the push rose
3×: R34 is 21.3M params, so ≈170 MiB per extra replica per step, ≈510 MiB/step of pure overhead,
all of it through host because there is no P2P.

⚠ That 36.6% conflates DP transfer overhead with the lower arithmetic intensity of bs64 vs bs256
per replica. Separating them needs a single-GPU bs64 render, which does not exist. Treat it as the
end-to-end figure, not a pure transfer measurement — but note that either way it is what residency
is buying back.

**This makes §2d.3 the top lever, ahead of bf16** (`planning/bf16_renderer.md`, which is 4-6
sessions for ~2× and needs a new `conv_close_mixed` theorem to reach the convnets). Roughly
two-thirds of a 26-hour verified ImageNet run is currently the parameter round trip. The two
compose — bf16 is arithmetic, residency is transport — but residency is cheaper and its gate is
already written.

##### Practical notes for whoever picks this up

* **`LEAN_MLIR_REPLICAS` and `PJRT_REPLICAS` are BOTH required and are not the same knob.**
  `PJRT_REPLICAS` configures the shim's client and compile options; `LEAN_MLIR_REPLICAS` makes the
  Lean driver call the data-parallel invoke. Setting only the former fails at the first step with
  `@… was compiled for N replicas; use the data-parallel invoke` — which reads like a missing
  feature and is not one. Cost me a wrong conclusion in-session.
* **Compile options for 1/2/4/8 replicas already exist** (`kPjrtCompileOptions{1,2,4,8}` in
  `ffi/pjrt_compile_options.h`), so no regeneration is needed to go to 4.
* **A long run is now supervisable**: `scripts/supervise.sh r34-imagenet-4gpu` (engine +
  `scripts/jobs/*.conf`) has AER restart, temperature-driven resting for the fanless box, a stall
  guard, and a precheck. `scripts/supervise.sh selftest` exercises it in ~90 s with no GPU.
* **The residency gate is unchanged and still the right one**: `GATE_ALT=PJRT_FFI_RESIDENT=1
  scripts/residency_gate.sh`. Its bit-exact floor was re-confirmed this session on a *different
  vendor and net* — `sgd-render-tie vit` on CUDA gave 5,526,346/5,526,346 bit-exact — which is
  further evidence the gate is buildable as written.

### 2d.3. ✅ Device-resident parameters — **BUILT, GATED, MEASURED AND COMMITTED 2026-08-01.**

Phases 1 and 2 landed in one session, not the scoped 3-4. **The projections below are now
results**, and the headline is that the parameter round trip is gone: 55-75% of a step becomes
1-7%, and a 30-epoch verified ImageNet run goes from ~25 h to **~16 h** on four cards.

⚠ **Every number in this block is from ares — RTX 4060 Ti, PCIe Gen3 x8, CUDA 12.9 — NOT the
7900 XTX the rest of §2d.3 was measured on.** They are not row-for-row comparable with the table
further down; what transfers is the shape of the result, and it transfers well (R34 bs32 read 55%
transfer there, 59.4% here).

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
PJRT_FFI_RESIDENT=1 .lake/build/bin/resnet34-verified-adam-xla data   # ONE env var, opt-in
scripts/det_shim.sh /tmp/detshim                                      # ⚠ REQUIRED on CUDA, below
LD_LIBRARY_PATH=/tmp/detshim GATE_ALT=PJRT_FFI_RESIDENT=1 scripts/residency_gate.sh
```

#### ▶ The measurement — PROBE median ms/step, shim-side `step=` in parentheses

| config | copying | resident | **gain** | param share |
|---|---|---|---|---|
| **cifar8-bn**, bs128, 1 GPU | 10 (7.7) | 3 (2.5) | **3.1×** | 75.5% → 6.9% |
| **EfficientNet-B0**, bs32, 1 GPU | 144 (133.8) | 89 (80.8) | **1.62×** | 46.7% → 5.3% |
| **ResNet-34**, bs32, 1 GPU | 199 (202.7) | 98 (90.1) | **2.03×** | 59.4% → 3.9% |
| **ResNet-34**, 2×bs32, 2 GPU | 269 (253.8) | 129 (117.2) | **2.09×** | 58.1% → 2.9% |
| **R34/ImageNet**, bs256, 1 GPU | 905 | 832 | 1.09× | 16.0% → 0.4% |
| **R34/ImageNet**, 4×bs64, 4 GPU | 596 | 386 | **1.54×** | 55.5% → 1.2% |

*(cifar8-bn's PROBE is integer-ms at 3 ms/step — one tick of resolution, §2m's own trap — so quote
the shim's float-precision 7.7 → 2.5 there, not 10 → 3.)*

**COMPUTE IS UNCHANGED, and that is the control that makes the rest readable**: R34 81.5 → 81.2 ms,
DP 105.6 → 104.8. The transport moved; the arithmetic did not. Consistent with the gate.

**▶ 4 GPUs: parallel efficiency 38% → 54%, at a fixed global batch of 256.**

| | 1 GPU bs256 | 4 GPU 4×64 | scaling |
|---|---|---|---|
| copying | 905 ms | 596 ms | **1.52×** (38% of 4) |
| **resident** | 832 ms | **386 ms** | **2.16×** (54% of 4) |

§2d.3a measured 1.46× and this reproduces it at 1.52×. At 5004 steps/epoch that is **24.9 h → 16.1 h**
for the 30-epoch run. §2d.3a's *"roughly two-thirds of a 26-hour verified ImageNet run is currently
the parameter round trip"* is confirmed as the right order and slightly over-stated: it was 55.5%.

#### ▶ AND THE DEMO LOOP TOO — `VerifiedNet.train`, where the gains are LARGEST

Converted the same day, and it was **~4 lines** because this loop is simpler than `trainAdamSched`:
the step is `params ← trainStep(x, params, y)` and the host reads **nothing** out of the result per
step — no loss slot, no BN stats. So **every** parameter is resident, not a prefix, and the param
round trip goes to a literal **0.0 ms**: no parameter buffer moves in either direction, ever.

| probe | copying | resident | **gain** | param share |
|---|---|---|---|---|
| **MNIST CNN** (ch4) | 6753 ms/ep | **1040** | **6.49×** | 84.6% → **0%** |
| **MNIST MLP** (ch2-3, the *dense* anchor) | 899 | **205** | **4.39×** | 80.6% → **0%** |
| **cifar8-bn SGD** (ch5, the *conv* anchor) | 1898 | **937** | **2.03×** | 59.4% → **0%** |
| **MNIST linear** (ch2, `trainLinear`) | 261 | **216** | **1.21×** | *unresolvable — see below* |

This loop was explicitly OUT of §2d.3's scope — *"`train`/`trainLinear` stay on the copying path,
they are the demo loops, not the throughput ones"*. That was written before §2d.3's own later
finding that the demo nets are the most transfer-bound in the set, and the finding wins: **the two
biggest gains in this entire work item are `lake run mnist-xla`'s two nets.** §2d.3 predicted "~4×"
for the MLP and got **4.39×**. It is an *interactivity* item, exactly as it said.

**`trainLinear` came over too** — a different FFI entry point (`linearTrainStepV`, `(x, W0, b0,
onehot) → (W0n, b0n)`), so `nResident := 2` is the whole parameter set and the same input-`i+1` /
output-`i` correspondence holds. One wrinkle worth knowing: `W0`/`b0` are **separate** FFI
arguments, so the copying path must re-slice them from the packed result every step; under
residency that slices an unwritten buffer, which is harmless because the epoch boundary re-derives
both from `readParams` before anything reads them. It is called out at the call site.

⚠ **The linear net is where this stops paying, and the reason is worth recording.** At 1.21× it is
the smallest gain in the set: 2 parameter buffers totalling 31 KB against a **0.3 ms** step. Do NOT
quote its `PARAM round trip` row — the shim reports 33.9% → 35.1%, which is *not* a measurement of
anything: with residency there is no parameter transfer at all, so that 0.1 ms is the timer's own
overhead against a step of the same order. §2m's resolution trap in a third place; the epoch number
is the only readable one.

#### ▶ THE EVAL PASS TOO — hold mode, and it roughly doubles the demo gains again

Done the same session. `forwardF32` was pushing the whole parameter set on **every eval batch**;
measured on the MNIST MLP, **73% of an eval step was the param push** (0.6 ms of 0.8 — compute is
0.1). It needs a different mechanism from the training loops and that is the content: that graph
returns **logits, not parameters**, so there is nothing to retain from the output. `res_out < 0`
selects **HOLD** mode — seed once, reuse across every batch — and `res_gen`, a generation token the
caller advances (the epoch number), is what re-seeds it.

⚠ **The token is not bookkeeping, it is the safety property.** A held set that went stale would
score the *previous* epoch's weights **silently**, which reads as a training plateau rather than as
an error — a nastier failure than anything the update mode has, and one the `[θ|m|v]` gate is
structurally blind to (eval does not feed training, so the trained parameters stay bit-identical
either way). Hence a second gate, below.

Full epochs, real data, train **+ eval**, committed shim:

| | copying | resident | **gain** | was, train-only |
|---|---|---|---|---|
| **MNIST CNN** | 5286 ms/ep | **1016** | **5.20×** | 6.49× |
| **MNIST MLP** | 930 | **187** | **4.97×** | 4.39× |
| **MNIST linear** | 206 | **113** | **1.82×** | 1.21× |

The linear net is the tell: **1.21× → 1.82×**, because eval was about half its epoch. Holding the
eval parameters is worth roughly as much again as the training loop was, on the nets small enough
for transport to dominate.

**`scripts/eval_residency_gate.sh`** is the gate the param one cannot be: it compares the reported
accuracy **epoch by epoch**, since a stale held set repeats an earlier epoch's number. It refuses
as VACUOUS if the accuracy never moves across the run — without that, a net that plateaus would
make a stale set indistinguishable from a correct one. All three demo nets PASS with 3 distinct
values each.

**▶ AND IT BUYS NOTHING ON THE CIFAR GROUP — measured, and worth knowing as a BOUND.**
All six `cifar-xla` variants pass the eval gate (3 epochs, 3 distinct values each), so hold mode
engages there; it just does not pay. `lake run cifar-xla` is **335 s train-only residency → 333 s
with eval held**, i.e. nothing outside noise, against 937 s copying. The reason is the one the
demo table shows from the other side: **eval only benefits where eval is TRANSPORT-bound.**
cifar8's eval is a conv forward over 10,000 images against ~38 small parameter tensors, so it is
compute-bound and there is nothing to remove; the MNIST linear net's eval is almost pure transport,
which is why its epoch went 1.21× → 1.82×. Expect this to hold for every larger net — R34's eval
param push is ~1.6% of its epoch. **Do not quote the demo-net eval numbers as general.**

⚠ On the **committed** shim the CNN's final accuracy reads 9838 against the copying path's 9833 —
and that is autotuning, not residency: three copying runs give **9833 / 9846 / 9847**, so the
resident number sits inside the A-vs-A spread, and under the deterministic shim all three epochs
match exactly. Same discipline as the cifar group above; the gate is the det-shim run.

**▶ FOUR of §2d.3's five predictions held; one is REFUTED and it is the interesting one.**

| predicted | measured |
|---|---|
| ~2.2× at R34 bs32 | **2.03×** ✅ |
| 1.15× at bs256 | **1.09×** ✅ |
| 1.9× on EfficientNet | **1.62×** — right direction, under |
| *"the SMALL demo nets benefit MOST"* | ✅ **strongly** — the two biggest gains in the set are `mnist-xla`'s, at **6.49×** (CNN) and **4.39×** (MLP, predicted "~4×") |
| *"DP efficiency barely moves… the ratio is nearly unchanged"* | ⛔ **true at 2 replicas (1.48 → 1.52×), FALSE at 4 (1.52 → 2.16×)** |

The refutation has a clean cause and §2d.3a had already written it down one section up: the
`[θ|m|v]` push is **O(N−1)** while compute is O(1) per replica, so removing it cannot leave the
*ratio* alone once N is big enough for the push to dominate. Two replicas is not big enough; four
is. **The DP argument for this work was right for a reason the DP section itself got wrong.**

**▶ AND THE REMAINING DP SHORTFALL IS NOW A DIFFERENT THING.** With the push gone, 2-GPU compute is
105.6 ms against single-device 81.5 at the same per-replica batch — so the +24 ms is the
`all_reduce`, staged through host because ares has no P2P (§2d.3a). That is real work, not
transport, and residency neither can nor should remove it. §2c's diagnosis of the 1.46-1.7× ceiling
was correct *and* is now spent; anything further is a collective/interconnect question.

#### ▶ How it is built — and why the Lean side has no backend branch

The insight §2d.3 opens with held exactly: **no buffer donation, no XLA-side aliasing.** The train
step's outputs already *are* device buffers; the copying path d2h'd them and destroyed them.
Residency is the pointer swap that keeps them instead.

* **`pjrt_ffi_invoke_f32_resident`** — ONE function for 1 and N replicas (a single-device copy beside
  a DP copy would be the double-writer disease one level down, in code). Inputs
  `[res_in, res_in+n)` and outputs `[res_out, res_out+n)` are the same tensors one step apart; the
  caller states both offsets and the shim **refuses unless the element counts agree tensor for
  tensor**, which is the structural check that replaces the copying path's per-output size query.
  Each replica keeps its own set on its own device.
* **`pjrt_ffi_resident_read`** — the per-epoch read-back for eval and the checkpoint. That call site
  (`thetamv := pbuf.extract 0 mvBytes`) was *already* once-per-epoch, which is why phase 2 came in
  far under the scoped ~150 lines: §2d.3a read the call path correctly.
* **The switch lives in C** (`iree_lean_ffi.c` reads `$PJRT_FFI_RESIDENT` and picks an entry point,
  behind the same weak-symbol pattern `pjrt_ffi_invoke_f32_dp` already uses). The Lean driver calls
  the same function with the same arguments on both backends, so the "one shared body, cannot drift"
  property every §2h cross-backend gate rests on is untouched.
* **`nResident` is an OPTIONAL argument defaulting to 0**, so not one existing call site changed —
  and the tie / DP-check harnesses, which read the whole returned blob, keep the copying path by
  construction rather than by anyone remembering to exclude them.

Total: ~330 lines of C, ~10 lines of Lean, no renderer change, no artifact moved.

#### ▶ THE GATE PASSES — and getting it to *mean* anything was most of the work

| gate | FLOOR | TEST | controls |
|---|---|---|---|
| **1 GPU**, R34, 10 steps | **0** | **0** on all 255,477,624 bytes | init 192,845,715 · 1-ULP fault 184,034,575 |
| **2 GPU DP**, 10 and 30 steps | **0** | **0** at both | resident-path fault 184,034,575 / 196,284,711 |
| **4 GPU DP**, 10 steps | **0** | **0** on all 261,572,064 bytes | resident-path fault 134,879,565 |
| **MNIST MLP** (`train` loop) | **0** | **0** on all 2,678,824 bytes | init 679,055 · staleness fault 267,983 |
| **MNIST CNN** | **0** | **0** on all 13,956,520 bytes | init 3,579,613 · staleness 4,401 |
| **cifar8-bn SGD** | **0** | **0** on all 212,968 bytes | init 60,064 · staleness 1,528 |
| **MNIST linear** (`trainLinear`) | **0** | **0** on all 31,400 bytes | both fire |
| **cifar8-bn AdamW**, **EfficientNet** | **0** | **0** (638,904 / 48,244,296 bytes) | both fire |

**▶ ViT was added to the gate 2026-08-01, and it REFUTES "AdamW ⇒ chaotic ⇒ fault mode 1".**
ViT-Tiny on the same AdamW schedule as R34 absorbs a 1-ULP fault down to **1 byte of 66,316,152** at
10 steps — single-device and at 4 replicas alike — where R34 amplifies one into ~184M of 255M. It
passes mode 1 only because 1 > 0; one more step of contraction and it reports VACUOUS, so the row is
**mode 2** (42,938,702 bytes). Finding 2's real content is therefore narrower than §2d.3 wrote it:
chaos is a property of the **net**, and the optimizer alone does not predict it — two nets under the
same optimizer land on opposite sides.

**▶ And the gate is single-device by construction, which made every DP render ungatable.**
`residency_gate.sh` hardcoded `HIP_/CUDA_VISIBLE_DEVICES=0`, so a DP variant died at session create
(*"PJRT_REPLICAS=4 but only 1 device(s) are addressable"* — the good failure mode, but a dead end).
Now `$GATE_DEVICES`, default `"0"`, so every pre-existing row is byte-for-byte the run it was.
The 4-replica ViT DP row passes with it: **0 of 66,316,152 bytes**, floor 0, both controls firing.

▶ **`scripts/residency_gate_all.sh` runs all eight as ONE command** and prints a scoreboard. It
exists because a readable verdict needs three things that were being retyped every time and are
each easy to get wrong: the deterministic shim on `LD_LIBRARY_PATH` (or the floor is noise), the
right `GATE_FAULT` for the net, and the slug. They live in one table there and nowhere else —
`TestShardCheck.lean`'s move, applied to the gate. A **new net whose gate reports VACUOUS on the
fault should have its mode flipped to 2 before anyone suspects the implementation**; which mode
applies is a fact about the net's optimizer, not about residency.

The 30-step DP run is the one that matters beyond the obvious: the copying path *force-resyncs* the
replicas from replica 0 every step, and residency does not, so replicas that drifted would show up
there and nowhere else. They do not.

**⚠ §2d.3's Finding 1 — "the floor IS bit-exact across processes" — is ROCm-SPECIFIC and does not
carry to CUDA.** With the committed compile options the R34 floor on ares is **191,094,739 of
255,477,624 bytes at 10 steps**, and **2,647,005 at ONE step**. The cause is XLA autotuning picking
different convolution algorithms per process; AdamW then amplifies one differing bit into most of
the blob within a few steps, which is Finding 2 seen from the other side. It IS suppressible —
`--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true`, captured at generation time because
`$XLA_FLAGS` is inert on this path (§4) — and then the floor is **0** and the gate reads as
designed. `scripts/det_shim.sh` is that recipe as one command.

**⚠⚠ §2d.3's Finding 2 DOES NOT GENERALISE EITHER, and it cost the `train`-loop gate a control.**
Finding 2 says *"the system is chaotic, so a 'small' transport error does not stay small — one
flipped mantissa bit becomes 64% of the blob in ten steps; bit-identity or nothing."* Measured on
**MNIST MLP under plain SGD, the opposite holds**: the 1-ULP fault moves **1 byte at 3 steps and
0 at 10**, on real data as well as synthetic. Read directly out of the dumps, the mechanism is
plain — `W0[0]` is byte-identical at step 3 and step 10, so the update has reached a fixed point and
`(w ⊕ 1) − lr·g` rounds straight back to `w`. **The perturbation is absorbed, not amplified.** A
macroscopic change to the same net (a different init seed) still moves 2,528,413 of 2,678,824 bytes,
so the harness is fine; it is the *fault* that is powerless there. Chaos is a property of the
net/optimizer/data, not of training — R34+AdamW is chaotic, MLP+SGD is contractive.

So `PJRT_FFI_FAULT=2` was added: **drop one step's retained parameters**, i.e. a stale retained
handle, injected deliberately. It is macroscopic, no contraction can absorb it, and unlike mode 1 it
is a defect residency *actually has* — the fault comment already named "a stale retained handle" as
one of the three plausible ways this breaks. `GATE_FAULT` selects it. Run D also now faults **the
alt path** rather than the default one, which is the path the control is supposed to be about.

**⚠ AND THE GATE'S OWN STATISTIC SATURATES, which nearly produced a false FAIL.** On the committed
shim, across two consecutive runs of the script:

| | FLOOR | FAULT (a deliberate 1-ULP corruption) |
|---|---|---|
| run 1 | 191,094,739 | **190,642,729** — the fault scored *cleaner than the noise* |
| run 2 | 189,531,906 | 191,037,525 — 0.8% apart |

At that point a real transport defect and run-to-run nondeterminism are the same number. The script
now refuses rather than reporting: it requires the fault to reach **2× the floor** before either
verdict branch is allowed to run. ⚠ A first version of that check tested the *ordering* instead and
**passed run 2 while calling a 0.9%-above-floor TEST a real regression** — the ordering is a coin
flip in a saturated band. The bar has to be a margin, which is what every tie in this file already
demands; verified both ways (saturates red on the committed shim, passes green on the deterministic
one).

#### ▶ What is NOT done

* ~~**`VerifiedNet.train` is still on the copying path**~~ ✅ **DONE the same day** — see the demo
  table above; it turned out to be the biggest win in the item (MLP 4.39×, CNN 6.49×).
  ✅ **`trainLinear` too**, so every training loop that drives an `-xla` target is converted. The
  E4M3 loops stay on the copying path for §2d.3's original reason — they quantise on the host every
  step, so they cannot go resident without moving quantisation into the graph.
* ~~**The EVAL pass still pushes parameters every batch**~~ ✅ **DONE** — hold mode, above.
* **A 4-GPU run is done**: R34/ImageNet `momdp64`, 4 × bs64, residency on, 6 epochs × 400 steps,
  `rc=0`, loss **7.03 → 5.65** monotone, all four cards at 47-57 °C and ~80 W. The rig holds.
* ⚠ **`pjrt_ffi_invoke_f32_resident_v2` — BUMP THE SUFFIX ON ANY SIGNATURE CHANGE.** The reference
  is weak and resolved at RUN time, so a binary linked before a signature change calls the new shim
  with the old argument list and every argument shifts. That is not a link error, it is garbage:
  caught here as *"returns 740 outputs, caller supplied -886575312 destinations"* from the one
  binary that had not been rebuilt after `res_gen` was inserted. With a versioned name a stale weak
  reference resolves to NULL and the caller falls back to the copying path instead — slower and
  correct, which is the right way round for a mismatch nothing else can detect.
* **`benchmark-xla` is untouched and would report wrong numbers if residency were wired in
  naively** — §2d.3's own analysis below still stands, and residency being opt-in is what keeps it
  correct today.
* ~~**No long run on the resident path**~~ — partially closed 2026-08-01 by running the WHOLE
  `lake run cifar-xla` group both ways: six nets × 40 epochs, **937 s → 335 s = 2.80×**
  (~15.6 min → ~5.6 min), `rc=0` on all twelve trainers.

  ⚠ **Final accuracies are NOT identical between the two passes, and that is the expected result,
  not a defect** — the committed shim autotunes, so this had to be read against an A-vs-A control
  rather than against equality. A second copying pass gives it:

  | | max | mean |
  |---|---|---|
  | **A-vs-A** — copying vs copying, two runs | **1.88 pp** | 0.59 pp |
  | copying vs **resident** (train only) | **1.22 pp** | 0.49 pp |
  | copying vs **resident** (train + eval) | **0.87 pp** | 0.39 pp |

  **Residency agrees with the copying path better than the copying path agrees with itself**, and
  the net that looked worst (`bn-momentum`, 1.22 pp) is exactly the one whose own A-vs-A spread is
  the largest at 1.88 pp. Same shape as §2f-bis's ConvNeXt result. Bit-identity is separately
  established by the gate under the deterministic shim; this is the end-to-end confirmation that it
  survives 40 epochs, which the 10-step gate cannot say.

  Still owed: an 80-epoch Imagenette run on a large net. Device memory is unchanged by construction
  (the retained set replaces buffers the copying path already held live), but nothing large has run
  for hours yet.
* **The IREE peer could not be link-checked on this box** — `resnet34-verified-adam` fails at
  `ld.lld` on four `iree_ffi_train_step_adam_*` symbols, and that reproduces at **pristine HEAD**,
  so it is a pre-existing ares condition and not this change. The weak-symbol pattern used here is
  the one `pjrt_ffi_invoke_f32_dp` already ships, but *verify the IREE build on the AMD box*.

*The original scoping follows, kept because the measurement is calibration against it.*

### 2d.3-scope. ▶ Device-resident parameters — scoped 2026-07-30

**This section did not exist until now, and five places in this file referenced it.** One of those
references — *"a calibrated model says the case roughly quadruples at 4 GPUs"* — had **no written
derivation anywhere**; treat it as an unverified intuition (see "payoff" below for what is actually
measured). The nearest real prior content is `xla_pjrt_ladder.md` §10.3a, which is a diagnosis with
no implementation scoping, and whose sequencing advice (*"device-resident params first, `all_reduce`
second"*) was **overtaken by events** — DP landed first and works, at 1.6-1.7× on five nets.

#### The problem, in one paragraph

`iree_ffi_invoke_f32` (`ffi/pjrt_ffi.c:322`) is **fully stateless**: every input is
`BufferFromHostBuffer`'d, `Execute` runs, every output is `ToHostBuffer`'d, and all buffers are
deleted. So the whole `[θ|m|v]` blob crosses PCIe **twice per step** — 272 MB each way at R34 — and
under DP it is pushed to *every* replica. That is §2c's diagnosed cause of the 1.46-1.68× ceiling.

#### ▶ The insight that makes this much smaller than it looks

**No buffer donation and no XLA-side aliasing are needed.** The train step's outputs *already are*
device buffers (`PJRT_Buffer*`); the shim currently d2h-copies them and then deletes them. Device
residency is: **retain the output buffers and pass them as the next call's parameter inputs.** A
pointer swap. The hardest-sounding part of the job evaporates before it starts.

#### The work, measured against the code

| phase | what | size |
|---|---|---|
| **1. shim** | `resident_create` (one h2d at startup) / `invoke_f32_resident` (params from the handle, param-outputs *replace* the handle, only `%loss`/`bnstat` come to host) / `resident_read` (eval, checkpoint, param dump) / `resident_release`. The fiddly part is the DP path — per-replica buffer arrays, which is exactly where the money is | **~250-350 lines C** |
| **2. Lean** | the bigger half. `params : ByteArray` is threaded functionally through **7 training loops / 11 call sites**. Convert **`trainAdamSched` only** — it drives all five Imagenette nets plus the cifar ablation. Touch points: loop carrier type, eval, checkpoint save/resume, `LEAN_MLIR_DUMP_PARAMS` | **~150 lines** |
| **3. gate + measure** | ✅ **DONE 2026-07-30, ahead of phases 1-2** — `scripts/residency_gate.sh` | ~110 lines |

**≈ 3-4 focused sessions.** Calibration for phase 1: `ladder.md` §10.2 estimated the DP shim change
at "~150 lines" and that landed accurately.

**Leave the E4M3 loops alone.** They quantise parameters on the host every step, so they cannot go
resident without moving quantisation into the graph. `train` / `trainLinear` likewise stay on the
copying path — they are the demo loops, not the throughput ones.

#### ▶ The design decision that protects every existing gate

**Make residency OPT-IN via env var, with the copying path staying the default.** The FFI surface is
symbol-identical across `iree_ffi.c` and `pjrt_ffi.c` by design (`nm -D`), and **every §2h
cross-backend gate depends on IREE and XLA running the same Lean code path**. An XLA-only residency
with a backend branch inside the training loop would break the "one shared body, cannot drift"
property those gates are built on.

Opt-in also hands you the gate for free: **residency must be BIT-IDENTICAL to the copying path over
N steps**, same seed, same data. That is the same known-answer shape as every other gate in this
file, it is cheap, and it is trivially verified to fail (perturb one retained buffer).

#### ✅ PHASE 3 IS DONE — 2026-07-30, written BEFORE phases 1-2 on purpose

`scripts/residency_gate.sh`. Authoring the gate first is not tidiness: a gate written *after* the
thing it gates tends to get written until it passes, and this one immediately earned its keep by
settling two questions that would otherwise have been discovered mid-implementation.

Four runs, each a fresh process from a **deleted** checkpoint (§4's most expensive trap), 10 steps,
1 epoch, synthetic inputs, eval skipped, comparing the whole dumped `[θ|m|v]`:

| | | R34, 272,094,840 bytes |
|---|---|---|
| **FLOOR** | default vs **itself**, two processes | **0 bytes differ** |
| **TEST** | default vs the alternative transport | **0 bytes differ** |
| **CONTROL** | default vs a perturbed init (`LEAN_MLIR_PERTURB_R`) | 193,307,970 differ |
| **FAULT** | default vs a deliberate **1-ULP** hit on ONE float | **175,479,358 differ** |

**Verified to fail**: point the alt path at the fault (`GATE_ALT=PJRT_FFI_FAULT=1`) and TEST goes to
175,479,358 with **rc=1**.

**▶ Finding 1 — the floor IS bit-exact across processes, so §2d.3's stated gate is buildable as
written.** This was in genuine doubt: §3 says *"XLA is NOT deterministic at epoch scale"* and
*"across processes it is not quite bit-stable"*. **That claim is narrower than it reads, and this
refines it**: §3's cross-process instability was measured on the *difference between two different
graphs* in a tie harness. The **same** graph run twice in two processes is bit-identical here — 272
MB, 10 full AdamW steps, nothing. Autotuning does not perturb a fixed program.
⚠ Scope it honestly: one net, 10 steps, constant synthetic inputs. It is not a claim about 80 epochs.

**▶ Finding 2 — a tolerance gate would be USELESS here, so bit-identity is not merely available, it
is required.** One flipped mantissa bit in one of 68 million floats becomes **64% of the blob** in
ten steps. The system is chaotic, so a "small" transport error does not stay small and there is no
tolerance that separates a 1-ULP bug from a 1-ULP nothing. Bit-identity or nothing — and the floor
happens to support it.

**▶ Why there are TWO controls.** The perturbed-init control (C) proves only that the harness can see
*a* difference. It does **not** prove the harness can see a **transport** difference, which is the
only thing this gate is for — precisely §4's *"a tie that is bit-exact everywhere is
indistinguishable from a harness comparing a buffer with itself"*. So `PJRT_FFI_FAULT=1` was added to
the shim: it flips the low mantissa bit of one returned float, the **weakest fault that is still a
fault**. A gate that catches that will catch a dropped buffer, a stale retained handle, or an
off-by-one replica offset — the plausible ways residency breaks.

**▶ It is already validated against a REAL second path, not a stub.** `PJRT_FFI_PINNED=1` is a
genuinely different way of getting outputs to the host, so TEST is a live comparison today rather
than a self-tie waiting for phase 1. Side benefit: it establishes that the pinned path is
**numerically inert**, which the previous commit asserted from a timing argument and never checked.

```bash
scripts/residency_gate.sh                       # default: r34, 10 steps, vs the pinned path
GATE_ALT=PJRT_FFI_FAULT=1 scripts/residency_gate.sh    # the red run — expect rc=1
GATE_ALT=PJRT_FFI_RESIDENT=1 scripts/residency_gate.sh # ▶ what phase 1 will run. ONE env var.
scripts/residency_gate.sh efficientnet-verified-adam-xla efficientnet 10
```

**Pointing it at residency is a one-variable change** — nothing else in the harness knows what it is
comparing, which is what makes it usable the day phase 1 has its first working invoke rather than
after it is finished.

#### ✅ THE MEASUREMENT IS DONE — 2026-07-30. The estimate below was **4× low, and low for the
#### wrong reason.** Read this before the retired arithmetic that follows it.

Run with `PJRT_FFI_TIMING=<report interval>`, which is the opt-in accounting added to
`ffi/pjrt_ffi.c` for exactly this (see "how to re-run it" below). Steady-state windows, synthetic
inputs so the loader is out of it, `LEAN_MLIR_SKIP_EVAL=1`:

| config | step | h2d params | compute | d2h params | **param round trip** | **share** |
|---|---|---|---|---|---|---|
| **R34, 1 GPU, bs32** | 160 ms | 23.5 ms (260 MB) | 63 ms | 64.6 ms (260 MB) | **88 ms** | **55%** |
| **R34, 1 GPU, bs256** | 709 ms | 27.0 ms | 569 ms | 66.1 ms | **93 ms** | **13%** |
| **R34, 2 GPU, bs32×2** | 196 ms | 40.8 ms (519 MB) | 72 ms | 65.1 ms | **106 ms** | **54%** |
| **R34, 2 GPU, bs128×2** | 460 ms | 43.8 ms (519 MB) | 296 ms | 71.4 ms | **116 ms** | **28%** |
| **EfficientNet, 1 GPU, bs32** | 215 ms | 25.2 ms (**46 MB**) | 106 ms | 75.9 ms (46 MB) | **101 ms** | **49%** |

*(`step` is the Lean-side `PROBE` median; the shim's own phase sum runs ~6% under it, the difference
being host-side batch assembly. Shares are against the PROBE number, i.e. the conservative one.)*

**▶ The cost is PER-BUFFER, not per-byte — that is the whole finding.** EfficientNet moves **5.6×
less** parameter data than R34 (46 MB vs 260 MB) and pays **the same ~100 ms**. Two nets, two
unknowns, solved exactly:

> **h2d ≈ 26 µs/buffer + 26 GB/s.  d2h ≈ 81 µs/buffer + 11 GB/s.**

§2d.3's assumed "~25 GB/s real PCIe 4.0 x16" was **right**. What the estimate missed is the fixed
per-buffer term, and these graphs hand the shim **513 (R34) to 887 (EfficientNet) separate
parameter buffers** every step, so 13-23 ms of pure per-call overhead lands on each direction before
a byte moves. That is also why d2h costs **2.8× h2d for identical bytes** — 81 µs against 26 µs per
buffer. ⚠ A two-point fit of two unknowns is *exactly determined, not validated*; the DP and bs256
cells are consistent with it but were not used to fit it. A third net would test it.

**What that changes, in order of how much it matters:**

1. **The prize is NOT multi-GPU only** — that claim is retired. It is **small-batch**, on any
   replica count: **~2.2× at bs32** on one GPU *and* on two, falling to 1.15× at bs256. The round
   trip is near-constant (88-116 ms) because it is set by parameter *count* and replica count, not
   by batch; the share therefore collapses as compute per step grows.
2. **It compounds with §2d.2's accuracy finding.** Accuracy tracks step count (295 steps/epoch at
   bs32 → 90.39%, 36 → 86.98%), so **bs32 is the configuration you actually want to run**, and bs32
   is exactly where residency is worth 2.2×. The two findings point the same way, which the "large
   batch is the answer" reading of §2d.1 did not.
3. **The payoff scales with parameter COUNT, not parameter bytes.** So it is roughly uniform across
   nets rather than concentrated in the param-heavy ones — EfficientNet, the *lightest* `[θ|m|v]` in
   the set, gains **1.9×**.
4. **DP efficiency barely moves.** Today 160→196 ms for 2× the images = **1.63×**; after residency
   ~72→~83 ms = **~1.73×**. Both paths get ~2.2× faster, so the *ratio* is nearly unchanged. §2e-ter
   was right that the replica-2 param push dominates the DP *overhead* (17 of the 27 ms added by
   going to 2 GPUs at bs32) — but removing it mostly makes one GPU faster too.

**Projected after residency** (param round trip → ~0; per-step host traffic left is x at ~1 buffer /
18 MB, plus `%loss`/bnstat; eval and checkpoint become per-*epoch* reads):

| config | now | projected | gain |
|---|---|---|---|
| R34 1 GPU bs32 | 160 ms | ~72 ms | **2.2×** |
| R34 2 GPU bs32×2 | 196 ms | ~90 ms | **2.2×** |
| R34 1 GPU bs256 | 709 ms | ~616 ms | 1.15× |
| R34 2 GPU bs128×2 | 460 ms | ~344 ms | 1.34× |
| EfficientNet 1 GPU bs32 | 215 ms | ~114 ms | **1.9×** |

**Verdict: the 3-4 sessions are justified**, and on a stronger and broader case than the section was
written on. ⚠ These are projections from a measured subtraction, not results — the gate in "the
design decision" above (residency bit-identical to the copying path over N steps) is what turns them
into results.

#### ▶ The two probe nets — and what residency would do to `benchmark-xla`

Same accounting, same recipe as the benchmark's own probes (3 epochs, `LEAN_MLIR_BENCH_SYNTH=1`,
`LEAN_MLIR_MAX_EPOCHS=3`). Both reproduce their committed anchors, so these are comparable numbers:
dense **614 ms/epoch** against `probeDenseRefMsXla := 610`, conv **3742** against
`probeConvRefMsXla := 3650` (inside that constant's documented ±6% spread).

| | step | param round trip | share | param buffers | param bytes |
|---|---|---|---|---|---|
| **dense probe** (`mnist-mlp-verified-xla`) | 1.3 ms | 1.0 ms | **75%** | 6 | 3 MB |
| **conv probe** (`cifar8-bn-verified-xla`) | 9.5 ms | 3.2 ms | **33.5%** | 38 | **~0.5 MB** |
| R34 bs32 | 160 ms | 88 ms | 55% | 513 | 260 MB |
| EfficientNet bs32 | 215 ms | 101 ms | 49% | 887 | 46 MB |

**cifar8-bn is the cleanest demonstration of the per-buffer story in the repo: 3.2 ms to move half a
megabyte, an effective 0.15 GB/s.** Nothing about that is bandwidth.

These are also the third and fourth points against the two-point fit, and they **split the verdict**:
h2d holds (predicted 0.99 ms for cifar8-bn vs 1.1 measured; the MLP's whole round trip predicted at
1.05 vs 1.0), while **d2h is looser than the fit implies** — 81 µs/buffer over-predicts cifar8-bn
(3.08 predicted vs 2.1 measured), and the per-buffer d2h across the four nets ranges **55-100 µs**.
Quote it as: **h2d ≈ 26 µs/buffer, solid on four nets; d2h is 2-4× h2d and NOT tightly pinned.**

**▶ `benchmark-xla` would NOT become "automagically" right, and the failure is §2j's trap one axis
over.** Two reasons, in order:

1. **It would not move at all**, because residency is opt-in by design (see "the design decision"
   above) and the benchmark does not set the variable. Nothing changes until someone wires it in or
   makes residency the default.
2. **Wired in naively, it reports wrong numbers.** The probes run the real trainers
   (`lakefile.lean`'s `runProbe`), so `yourMs` genuinely drops — but `probe*RefMsXla` and every
   `refSecXla` are constants measured on the copying path, and `yourSec = refSec × yourMs / refMs`
   only self-corrects when the probe and the chapter it scales speed up by the **same** factor. They
   do not, because the speedup tracks parameter count against compute per step:

   | rows | probe | probe gain | chapter gain | estimate |
   |---|---|---|---|---|
   | ch1-2 MNIST linear/MLP | dense | ~4.0× | ~4.0× (the same nets) | ✅ right |
   | ch3-4 MNIST CNN, CIFAR×6 | conv | 1.50× | ~1.50× (ch4 **is** cifar8-bn) | ✅ right |
   | **ch5-8 R34/mnv2/ENet/ConvNeXt** | conv | **1.50×** | **~2.0-2.2×** | ⚠️ **~1.4× pessimistic** |
   | ch9 ViT | attn | — | the same net | ✅ right |

   The error is **pessimistic**, which is the safer direction — but ch5-8 are **90% of the XLA total**
   (21,960 s of 24,391), so the headline Part-1 number reads ~1.4× too slow. The fix is what §2j did
   when it added this column in the first place: **re-measure it** — 3 probe constants + 9 chapter
   rows. Note ch9's `refSecXla := 3491` is a *measured* 80-epoch wall validated to 0.3%; that
   validation goes void with it.

   One real mitigation: on the reference card the on-reference factor (§2j's "0.94-1.06× is
   agreement") would immediately read far outside the band, so this fails loudly rather than
   silently — *if* anyone runs `benchmark-xla` on a 7900 XTX.

**▶ And the surprise worth carrying: the SMALL demo nets benefit MOST.** The MNIST MLP is **75%**
transfer against R34's 55%, so residency is worth ~4× there and ~2.2× on R34. That inverts the
premise this whole section was written on. Device-resident parameters is not primarily an
Imagenette-throughput item — it is a **`lake run mnist-xla` / `cifar-xla` interactivity** item, and
those are the demo groups a reader actually sits and watches. Weigh that when deciding the 3-4
sessions: the case is stronger than §2d.3 assumed *and* it points somewhere else.

#### ⛔ The pinned-d2h idea — proposed here, then MEASURED AND REFUTED the same day

**This section briefly recommended pinning as "a cheaper thing to try FIRST, ~50 lines against ~500."
Do not do it.** The recommendation rested on a mechanism that turned out to be wrong, and the
retraction is worth more than the proposal was.

*The argument, as made.* d2h moves bytes at a fitted ~11 GB/s while h2d hits ~26 (PCIe 4.0 x16 line
rate) on identical buffers. A d2h into **pageable** host memory — and these destinations are
Lean-owned `ByteArray` slices — cannot be DMA'd into directly, so the runtime must bounce through a
pinned staging buffer and then memcpy out. In series that predicts
`1/(1/26 + 1/15.5) = 9.7 GB/s` against the 11.0 measured, where 15.5 GB/s is this host's
single-threaded memcpy at 260 MB (measured). **Two independent numbers agreeing to 13% — and it was
still wrong.**

*The test.* `PJRT_FFI_PINNED=1` (`ffi/pjrt_ffi.c`) DMAs into an arena allocated with `hipHostMalloc`,
i.e. one that is definitely pinned, then memcpys out — which separates the DMA leg from the copy leg.
If the bounce were the mechanism, the DMA leg alone should reach line rate (~10 ms for 260 MB).

| R34 bs32, d2h await (the DMA leg) | run 1 | run 2 | run 3 |
|---|---|---|---|
| pageable destination (default) | 62.3 | 60.7 | 61.6 ms |
| **pinned destination** | 59.3 | 64.2 | **62.0 ms** |

**Overlapping distributions. Pinning does nothing.** So d2h's ~11 GB/s is simply what this path
costs — not a pageable-destination artefact — and it is not addressable this way. Turning the flag on
is a **17% regression** (161 → 189 ms/step): the explicit memcpy costs ~30.5 ms and buys no DMA gain.

**⇒ The route to d2h is FEWER BUFFERS, not faster ones — which is device residency, and there is no
cheaper partial win to take first.** The per-buffer term was always the larger leg anyway: of R34's
64 ms d2h, 41.6 is per-buffer and 24.8 is bytes; for cifar8-bn it is 2.05 vs 0.05, i.e. **the two
nets that gain most from residency are ~100% per-buffer**, where pinning could not have helped even
if the mechanism had been right. Best case across the four nets was 12% (the MNIST MLP) and typically
1-7%.

The flag is kept, **off by default**, as the falsification instrument rather than as a feature: this
is a hardware/driver-specific negative, and on a box with a different PCIe or ROCm setup pinning may
well help. Re-checking is two minutes (`PJRT_FFI_PINNED=1 PJRT_FFI_TIMING=10`) and beats re-deriving
the argument from scratch — which is exactly what this section did once already.

⚠ **The lesson, and it is this thread's own, committed against me this time.** I inferred a mechanism
from a single arithmetic coincidence (9.7 predicted vs 11.0 measured) and wrote it into the plan as
a recommendation. That is the same error as §2j's retracted "thermal" story for the conv probe and
the retracted `MIOPEN_DEBUG_CONV_GEMM=0` "fix" — **one agreement is not a mechanism**, and a
prediction matching to 13% is *weaker* evidence than it feels, because plausible wrong models land
there routinely. The measurement that refuted it took twenty minutes; acting on the proposal would
have cost a session and shipped a regression.

**How to re-run it** — the accounting is opt-in and costs nothing when off (control: `PROBE` 161 vs
160 ms at bs32, 195 vs 197 DP, i.e. inside noise):

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
HIP_VISIBLE_DEVICES=0 LEAN_MLIR_BENCH_SYNTH=1 LEAN_MLIR_SKIP_EVAL=1 \
  LEAN_MLIR_MAX_STEPS=48 PJRT_FFI_TIMING=10 \
  .lake/build/bin/resnet34-verified-adam-xla data
#   ^ windows of 10 steps, each RESET after reporting — never cumulative, or the ~2 s first compile
#     and the cold warmup steps ride in every later number. Window 1 is warmup; read window 2+.
#   ^ LEAN_MLIR_MAX_STEPS exits via `return ()` BEFORE any checkpoint write, so it cannot leave a
#     stale marker. It also caps steps WITHIN an epoch, so keep it under nTrain/(bs*replicas)
#     (bs256 → 36, bs128×2 → 36) or the probe never fires and you get an 80-epoch run.
```

⚠ **Two traps this run hit, both already documented in §4, both still worth the reminder**: the
EfficientNet and `adamdp`/`adamdp128` checkpoints were at `epoch=80`, so the first attempt at each
resumed past the end and did **nothing** while printing a clean `done` — and the timing report never
appeared, which is the only reason it was caught. Move `.lake/build/<slug>_<variant>_ckpt_xla.bin{,.epoch}`
aside first.

⚠ **`Execute` is ASYNCHRONOUS, and ignoring that gives a 97% answer.** The first version of this
measurement timed `Execute` at 4 ms and the d2h await at 123 ms, and read out "the param round trip
is 94% of the step" — wrong, because the GPU compute was landing inside the d2h await. The shim now
requests `device_complete_events` and waits on it **when timing only**, which splits compute from
transfer and costs nothing (this step's d2h depends on this step's compute, so there was nothing to
overlap). Any future phase timing on this path has to do the same.

---

*The original estimate follows, kept because the measurement is calibration against it.*

#### Payoff — and it is NOT uniform, which §10.6's "worth doing alone" oversells

R34's `[θ|m|v]` is 272 MB ⇒ 544 MB per step. At ~25 GB/s real PCIe 4.0 x16 that is ~22 ms, and the
shim is synchronous so it is serial, not overlapped:

| case | step time | transfer share | expected gain |
|---|---|---|---|
| 1 GPU, bs32 | 162 ms | ~13% | ~1.15× |
| 1 GPU, bs256 | 674 ms | ~3% | **~nothing** |
| **2 GPU** | — | **13-16%, MEASURED** (§2e-ter) | **1.67× → ~1.95×** |

**The prize is multi-GPU only.** The single-GPU large-batch case buys almost nothing, because
§2d.1's bs256 win *already* amortised this same transfer 8× — that is the same fact seen from the
other side. §2e-ter measured the per-step DP overhead directly and found it dominated by the
`[θ|m|v]` push to replica 2, so removing it should recover most of the gap.

The case does strengthen with replica count (each replica costs another full push against another
slice of compute), which is the honest version of the retired "quadruples at 4 GPUs" claim — but
**nobody has measured 4 replicas**, that needs ares, and §10.6 measured the whole 4×4060 Ti ares box
at only **2.1×** one 7900 XTX for R34. So the absolute ceiling on the box where this matters most is
modest. Also note the easy transfer wins are already taken: 256 → 205 → 162 ms.

*(Everything in the three paragraphs above except "the easy transfer wins are already taken" is
**superseded**: the bandwidth figure was right, the conclusion drawn from it was not.)*

#### ▶ Do this ONE-HOUR measurement before writing any of it ✅ DONE 2026-07-30

Instrument the existing shim to time the parameter h2d/d2h **specifically** (not the whole step), on
1 and 2 GPUs, at bs32 and bs256. That converts the 13% / 3% / 15% arithmetic above from an estimate
into a measurement and says whether the 3-4 sessions are justified. This is the `ladder.md` §10.5
"measure before building" move, and **it has never been done for this item** — §10.5's own
prescribed first measurement was the bs256 re-render, which §2d.1 completed.

*It was worth the hour: it did not merely confirm the estimate at higher precision, it inverted the
conclusion. The bs256 cell — the one the estimate called "~nothing" at 3% — is 13%, and the bs32
cell it called 13% is 55%.*

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

**bf16 is worse than fp32 on the 7900 XTX** — measured ×0.96 for R34 — but that is an **RDNA3**
result and it does NOT carry. This line used to say "on this box … it matters on ares only if
someone measures it there"; measured there 2026-08-01, `jax/scripts/jax_r34_bf16_bench.py 32`,
three runs: **×1.79 / ×1.73 / ×1.77** (fp32 94.1-94.5 ms/step, bf16 52.7-54.4). Ada has bf16
tensor cores and gfx1100 effectively does not. **Quote the vendor with the number.**

⚠ **And residency roughly DOUBLED what bf16 is worth**, which is the compounding §2d.3a predicted
("bf16 is arithmetic, residency is transport"). On the verified R34 bs32 step, ares:

| | step | compute | share | with bf16 at ×1.76 |
|---|---|---|---|---|
| copying (pre-§2d.3) | 199 ms | 81.5 | 41% | 199 → 164 = **1.21×** |
| **resident** | 98 ms | 81.2 | **83%** | 98 → 63 = **1.56×** |

Transport was masking the arithmetic; removing it is what makes the *next* lever pay. Together
199 → ~63 ms/step, **3.2×**. ⚠ The bf16 column is a **projection** — it applies the JAX bench's
hardware ratio to the verified render's measured compute, and there is no bf16 verified render to
check it against (`planning/bf16_renderer.md`, 4-6 sessions, needs `conv_close_mixed`). The
committed R34 artifact contains **0** bf16 ops today.

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
- **`LEAN_MLIR_BENCH_SYNTH` was sized at the PER-REPLICA batch, so every DP run read past the end of
  it.** Found 2026-07-30 (§2d.3) and fixed in `VerifiedTrain.lean`: `mkSynthData` allocated `bs`
  images while a DP step consumes `bs * replicas`. At bs32×2 the overread was silent — it produced
  plausible timings for months; at bs128×2 it aborted with `free(): invalid next size (normal)`
  *before the first invoke*, which is what exposed it. The fix reads `replicas` before the data is
  built and sizes it at `bs * replicas`. Re-measured after: the bs32×2 numbers moved by ~1 ms, so
  nothing measured through it needs redoing — **but that is luck, not design**. The lesson is the
  familiar one in a new place: the synthetic path exists to remove a variable from a measurement,
  which makes it exactly the code least likely to be looked at when the measurement comes out clean.
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

✅ **DONE 2026-07-30 — and as ONE generic harness, not two more copies.**
`tests/TestShardCheck.lean` → `lake build shard-check <convnext|efficientnet|mobilenetv2>
[<dpPath>]`. Everything per-net derives from `net.slug`
(`verified_mlir/<slug>_adam{,dp}_train_step.mlir` + the matching `m.<slug>_adam{,dp}_train_step`),
so the only table in it is spec + batch. Writing it per net would have been the double-writer
disease one level down, in code — the same reasoning that made `vitBackAll` one shared traversal.

**All three nets now have real-net sharding evidence:**

| net | BN layers | TEST `\|DP − mean(A,B)\|` | CONTROL `\|DP − A\|` | separation |
|---|---|---|---|---|
| **EfficientNet** | 49 | **9.7e-8** | 0.862 | 8.9e6 |
| **MobileNetV2** | 52 | **1.2e-7** | 0.922 | 7.9e6 |
| **ConvNeXt** | 0 | **8.2e-8** | 0.137 | 1.7e6 |

**Verified to fail:** a sum-not-mean DP render (all 210 divisors 2.0 → 1.0) drives mnv2's TEST to
**1.000000** with rc=1 — seven orders above the passing value. The built-in CONTROL only proves the
two shards are distinguishable; this proves the TEST comparison itself is wired correctly.

**The generalisation is gated by reproducing the specific harness it replaces:** ConvNeXt comes back
at TEST **8.2e-8** / CONTROL **0.136877** against `convnext-shard-check`'s committed 8.2e-8 / 0.137.
(§3's cross-process nondeterminism applies to the TEST value — quote the bound, not the digits.)

⚠ **The BN nets needed one thing the ConvNeXt-only harness did not have**, and it is worth knowing
because the failure mode was ideal: batch-BN nets carry running-stat **inputs** and return the batch
statistics, so their arity is 2·(BN layers) wider on both sides. Omitting that is **not** a silent
wrong answer — the shim's G4 guard refuses the call outright (*"returns 887 outputs, caller supplied
789 destinations"*), which is exactly how it was caught. `bnChannels` is empty for ConvNeXt, so the
added lines degrade to a no-op there, which is why its numbers are unchanged.

*`tests/TestConvNeXtShardCheck.lean` and its `convnext-shard-check` target are now subsumed and
could be retired to save a 152-line near-duplicate — deliberately NOT done here, since it is a
working committed gate and removing it is a separate call.*

R34 still has only the cifar8 proxy for sharding: it has no `adamdp` peer at a batch this
construction can pair with, so it is out of scope for this harness rather than merely undone.

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
