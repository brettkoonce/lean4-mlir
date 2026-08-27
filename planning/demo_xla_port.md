# demo_xla_port.md — moving the 34 demos off IREE onto XLA/PJRT

Scoped 2026-08-25 on ares (6× RTX 4060 Ti, sm_89, CUDA 12.9, PJRT 0.114).
Successor to `planning/detector_pjrt_port.md`, which ported **one** demo (the VisDrone FPN
detector) and left the recipe behind. This document applies that recipe to the other 33.

---

## 0. ⭐⭐ THE ONE-PARAGRAPH VERSION

The link modes are gone (`ireeLink`/`xlaLink` retired 2026-08-25, all 167 exes on `lowererLink`),
so **nothing is "an IREE binary" any more** — every demo already dlopens whichever shim
`$LEAN_MLIR_LOWERER` names and defaults to XLA. What still pins a demo to IREE is one of exactly
two things: (1) its driver hardcodes a `.vmfb` path instead of calling `Train.lean`'s
`graphArtifact`, or (2) it needs one of the **six `not_ported` stubs** in `ffi/pjrt_ffi.c`.
▶ Those are very different costs, and the useful finding of this scoping is that **13 of the 34
demos need only (1)** — a driver edit against an FFI that already works. ⛔ 12 need (2), which is
trusted-shim work and is where the real risk lives. 9 need nothing at all.

---

## 1. ⭐⭐⭐ THE THREE TIERS — this is the whole document

| tier | what it needs | demos | risk |
|---|---|---|---|
| **0** nothing | no GPU session at all | **8** | none |
| **1** driver only, generic FFI | `graphArtifact` instead of a hardcoded `.vmfb` | **9** | low |
| **2** driver only, specialised FFI **already ported** | same edit; `train_step_adam_ddpm` exists | **4** | low |
| **3** ⛔ new PJRT entry point | one of 6 `not_ported` stubs in the trusted shim | **12** | **high** |
| ✅ done | — | **1** (`yolov1-visdrone-fpn`) | — |

▶ **Tiers 1+2 are 13 demos for one mechanical edit each.** Do them first, in one pass. They are
the cheap two-thirds of the surface and they need no change to `ffi/pjrt_ffi.c` — i.e. no change
to the trusted lowerer, which is the thing this repo is most careful about.

---

## 2. ⛔ THE BLOCKER — 6 of 7 specialised entry points are still stubs

`detector_pjrt_port.md` §9 got this right and it is still true, verified in `ffi/pjrt_ffi.c`
2026-08-25. The verified nets reach XLA through the **generic** `iree_ffi_invoke_f32`; the
`Train.lean` demos go through **specialised** entry points, and almost none exist:

| entry point | pjrt_ffi.c | line | who needs it |
|---|---|---|---|
| `iree_ffi_invoke_f32` (generic) | ✅ implemented | 761 | every forward / every verified net |
| `train_step_adam_ddpm` | ✅ implemented | 1747 | DDPM ×4, FPN detector |
| `train_step_adam` | ⛔ `not_ported` | 1697 | autoencoder, bigram, grad-fd-probe |
| `train_step_adam_seg` | ⛔ `not_ported` | 1708 | UNet ×3, tinygpt, tinystories |
| `train_step_adam_yolov1` | ⛔ `not_ported` | 1830 | YOLO ×4 |
| `train_step_adam_softlabel` | ⛔ `not_ported` | 1719 | mixup/cutmix runs |
| `train_step_mlp` | ⛔ `not_ported` | 1677 | — |
| `train_step_generic` | ⛔ `not_ported` | 1687 | — |

The stubs are **loud** — `return not_ported("train_step_adam_seg")` — so a tier-3 demo fails
immediately and visibly on XLA rather than misbehaving. That is the one piece of good news here.

⚠ `detector_pjrt_port.md` §9 deliberately shipped only the DDPM one: *"the other six are still
`not_ported` — deliberately. They now have a working template, but shipping untested marshalling
into the trusted shim is worse than a loud stub."* ▶ That judgement stands. Tier 3 is not "five
more of the same edit"; each entry point has its own layout and needs its own gate (§5).

---

## 3. ⚠⚠ THE TRAP: "the driver is clean" ≠ "it runs on XLA"

Six demos route through `Train.lean` and never mention `.vmfb`, so a driver-level sweep calls them
backend-agnostic. **They are not.** `unet-pets-train`, `unet-brats-train`, `unet-brats-r34`,
`autoencoder-pets-train`, `grad-fd-probe` and `yolov1-pets-train-bootstrap` have clean drivers and
will still hit a `not_ported` stub the moment they take a training step on XLA.

▶ The two layers are independent and **both** must be checked:

```
driver layer   :  hardcoded ".vmfb"  vs  graphArtifact pfx suffix     → Train.lean:83
FFI layer      :  which specialised entry point the dispatch picks    → Train.lean:911-961
```

The dispatch at `Train.lean:911-961` is what decides the FFI layer, and it derives from `lossKind`,
not from anything written in the demo file:

```
useFpnRun      := !cfg.fpnScales.isEmpty                        → train_step_adam_ddpm   ✅
useYolov1Run   := lossKind == .yolov1Masked && !useFpnRun       → train_step_adam_yolov1 ⛔
useSeg         := lossKind.isSeg                                → train_step_adam_seg    ⛔
useSoftLabels  := cfg.useMixup || useCutmix || useKnnMixup      → train_step_adam_soft   ⛔
(else)                                                          → train_step_adam        ⛔
```

⭐ Note the FPN detector is on XLA **because it borrows the DDPM protocol** — `Train.lean:911`
routes FPN through `trainStepAdamF32Ddpm` verbatim (same x + single flat target + lr + t
signature). That is why one port bought two demos, and it is worth looking for the same trick
before writing a new entry point.

---

## 4. THE FULL MAP

### 4.1 Tier 0 — nothing to do (8)

`anchor-loss-probe`, `diou-loss-probe`, `flash-probe`, `fpn-detect-probe`, `fpn-loss-probe`,
`fpn-neck-probe`, `seg-loss-probe`, `fpn-train-emit`. Zero `LowererSession` calls — they emit MLIR
or compute on the host. ⚠ `fpn-train-emit` reads as IREE-flavoured (its docstring mentions
`iree-compile --compile-to=input`) but makes no session call at all; that mention is advice to the
reader, not a dependency.

### 4.2 Tier 1 — driver only, generic FFI already works (9)

All call **only** `forwardF32` → `iree_ffi_invoke_f32`, which is ✅ implemented on PJRT. The single
blocker is a hardcoded `.vmfb`.

| demo | fix |
|---|---|
| `gradcam` | `graphArtifact`; also caches + compiles its own cam vmfb |
| `yolov1-pets-infer` | `graphArtifact` (guard + `create`) |
| `pets-predict` | `graphArtifact` |
| `brats-predict` | `graphArtifact` |
| `inspect-convnext` | `graphArtifact` |
| `cifar-ddpm-sample` | `graphArtifact` + drop its own `iree-compile` subprocess |
| `cifar-ddpm-attn-sample` | ″ |
| `cifar-ddpm-sincos-sample` | ″ |
| `mnist-ddpm-sample` | ″ |

### 4.3 Tier 2 — driver only, specialised FFI already ported (4)

`cifar-ddpm-train`, `cifar-ddpm-attn-train`, `cifar-ddpm-sincos-train`, `mnist-ddpm-train`.
They use `trainStepAdamF32Ddpm`, which the FPN port implemented. The edit: replace the hardcoded
`.vmfb` with **`Train.graphArtifact pfx suffix`** (the demos' helper — the verified trainers use
`VerifiedTrain.mkSession` instead; do not mix them up), and delete the local `iree-compile`
subprocess, which is already redundant because `Train.lean:runIreeCached` returns early on XLA.

⭐ **The edit is proven, not hypothesised — UPDATED 2026-08-25.** Exactly this swap was landed on
the three fp8 trainers (`trainE4M3`, `trainAdamSchedE4M3`, `trainLinearE4M3`), which were IREE-only
for the identical reason: they printed the "XLA/PJRT" banner and then died in `iree-compile`. The
swap worked first time and the f32 artifacts stayed byte-identical. ▶ Use that commit as the
template; the tier-1/2 demos are the same shape of change.

⚠⚠ **EXPECT THE EDIT TO BE CHEAP AND THEN EXPECT G4 VIOLATIONS — that is the point, not a
setback.** The moment the fp8 MLP/CNN could reach XLA, the PJRT shim's G4 arity guard caught a
latent marshalling bug IREE had never surfaced (the driver supplied N output destinations for a
graph returning N+1). These demo drivers hand-roll their marshalling too, so the same class of
bug is *likely* here. Budget for it: the port is a few lines, and the bugs it exposes are the
actual work — and the actual value, since they were wrong on IREE too and nothing was checking.

⚠ **Correction to this section's stated payoff.** It previously promised "an actual XLA-vs-IREE
speed and correctness datapoint". That is NOT obtainable on this box as configured: `iree-compile`
is not on PATH (it resolves only via `.venv/bin` inside `runDemoGroup`), so the IREE arm of any
such comparison cannot be run here. ▶ The port still stands on its own — these demos gain XLA, and
gate A still checks IREE is unchanged *by construction* (every changed expression evaluates to the
same string) — but a measured cross-backend comparison needs the IREE toolchain fixed first.

### 4.4 Tier 3 — ⛔ needs a new PJRT entry point (12)

| entry point needed | demos |
|---|---|
| `train_step_adam_seg` | `tinygpt-shakespeare`, `tinystories`, `unet-pets-train`, `unet-brats-train`, `unet-brats-r34` |
| ~~`train_step_adam`~~ ✅ **ported 2026-08-26** | `bigram-shakespeare`, `grad-fd-probe` (⚠ NOT `autoencoder-pets-train` — see §4.5) |
| `train_step_adam_yolov1` | `yolov1-pets-train-bootstrap`, `yolov1-visdrone448`, `yolov1-visdrone448s16`, `yolov1-visdrone-anchor` |

⭐ `train_step_adam_seg` unlocks **5 demos across two unrelated families** (UNet segmentation and
GPT next-token CE — the GPT drivers reuse the per-token CE ride, `MainTinyStories.lean:19`). ▶ It
is the highest-leverage stub and should be first.

---

## 4.5 ⭐ STATUS 2026-08-26 — tiers 1+2 swept, `train_step_adam_seg` ported

Done and on `main` (`bc946fe`):

- **Tiers 1+2 swept.** 18 hardcoded `.vmfb` paths across 16 drivers moved to
  `NetSpec.graphArtifact`. ⚠ There was a **second** IREE dependency this doc's
  two-layer model does not name: 8 demos carry their own `compileMlir`/`runIree`
  that shells out to `iree-compile` unconditionally. `Train.lean:102` already
  guards that but is `private`, so each demo reimplemented the call without it.
  Guarded. A driver sweep that only greps for `.vmfb` misses these.
- **`train_step_adam_seg` is implemented**, which §4.4 correctly called the
  highest-leverage stub. The blocker was not in either layer this doc models:
  `iree_ffi_invoke_f32` hardcoded `PJRT_Buffer_Type_F32` for **every** input,
  and the seg graph takes a real int32 label (`%y_seg: tensor<BxHxWxi32>`).
  Fixed by splitting an `invoke_typed` core out with a per-input dtype array;
  `iree_ffi_invoke_f32` is now a wrapper passing NULL, so f32 callers are
  byte-identical (gated on `score-checkpoint {convnext,vit}`, bit-identical).
- **`bigram-shakespeare` did not compile**, independent of any of this —
  verified against the pristine file. Fixed (`return b` → `pure b`).

### ✅ DONE 2026-08-26: `unet-brats-train` — `train_step_adam_seg` VALIDATED

Run: `CUDA_VISIBLE_DEVICES=0 .lake/build/bin/unet-brats-train data/brats 3`
(default `wce` arm, 484 volumes / 14,415 train / 2,569 val slices, exit 0,
~201 s/epoch, 217 ms/step).

| epoch | train loss | mIoU | c1 | c2 | c3 | WT | TC | ET | WT pred/gt |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.402 | **0.640** | 0.477 | 0.433 | 0.668 | **0.765** | **0.808** | 0.801 | 1.56× |
| 2 | 0.329 | 0.494 | 0.322 | 0.182 | 0.511 | 0.590 | 0.508 | 0.676 | 2.38× |
| 3 | 0.308 | 0.611 | 0.411 | 0.390 | 0.664 | 0.715 | 0.763 | 0.798 | 1.77× |

All three gates from the previous revision pass. The one that matters is #3:
**per-class IoU is strongly non-zero on all three tumour classes in every
epoch.** Coherent 240×240 four-class segmentation is not reachable through a
mis-indexed int32 label, so the `invoke_typed` per-input dtype path is correct
on the real per-pixel case — the case tinygpt's `[B, T, 1]` label could never
exercise. ▶ `unet-brats-r34` and `tinystories` are now unblocked.

⭐ **Gate #2 ("loss decreases") would have passed on the worst epoch.** Training
loss fell monotonically 0.402 → 0.329 → 0.308 while val mIoU went
0.640 → 0.494 → 0.611. Epoch 2 is simultaneously the best loss so far and the
most inverted model (WT pred/gt 2.38×, precision 41.9% at 99.6% recall). This
is the concrete vindication of the previous revision's warning that loss is not
a sufficient gate; it is now a measured fact rather than a caution.

#### ⚠⚠ The γ=2 warning in the previous revision was misaimed

It said to read `MainUnetBratsTrain.lean:186-195` because "γ=2 is a documented
collapse setting on this data, so a zero there may be the loss config rather
than the port." γ is `focal`'s only knob and **the default arm is `wce`**, not
`focal` — the plain invocation this section prescribes never touches γ. The
caution was real but attached to an arm nobody was running.

#### ⛔⛔ The prescribed baseline does not exist — `brats_demo.md` is VOID

This section told the reader to "compare against the IREE-era numbers before
concluding anything about the port." **Do not.** Every trained number in
`planning/brats_demo.md` — the FINAL VERDICT table, the wave-2 ablation, the
whole collapse-vs-inversion loss thread — predates the 2026-07-22 shuffle fix
(`430ba2c`, `ca83835`). `lean_f32_shuffle` permuted images by a full record and
labels by a **hardcoded 4 bytes**; BraTS has `labelBytes = 240²`, so it trained
on mismatched image/mask pairs every epoch. `planning/post_shuffle_fix.md` §1
lists BraTS as invalidation item **#1, "every trained result void"**.

▶ The trap is cross-file. The invalidation IS recorded — `post_shuffle_fix.md`
§1, `demos/README.md`, `planning/r34_brats_retrain.md`, and blueprint §10.2.3,
which retracted its whole loss chapter on 2026-07-25 (`d565f2c`). **`brats_demo.md`
is the one file that was missed**, and it is the one a reader reaches for first —
so following this section's own instruction leads straight into it. ⚠ The run
above is therefore NOT the first BraTS training on fixed data (the R34 retrain
behind §10.2.3 was); it is the first for the from-scratch UNet and the first
loss-arm sweep. Its epoch-1 `ce` numbers (mIoU 0.689, WT 0.873, TC 0.805, ET
0.838) independently replicate `d565f2c`'s (~0.69, 0.875/0.813/0.837) to ~0.01,
a month apart. The untuned default arm at epoch 1
(WT 0.765 / TC 0.808 / ET 0.801) beats that doc's best tuned recipe
(`wcesqrt cos pb aug` best-by-val, WT 0.661 / TC 0.489 / ET 0.322).

#### ⭐⭐ The follow-up that mattered more than the port: plain `ce` now wins

`post_shuffle_fix.md` §1a named the decisive test — *"If plain `ce` now
segments, the entire loss-design thread was chasing a data bug."* Run same-day,
same config, 3 epochs:

| arm | best mIoU | WT | TC | ET | WT pred/gt | shape |
|---|---|---|---|---|---|---|
| **`ce`** | **0.728** | **0.903** | **0.858** | **0.851** | 0.83 → 0.94× | improves smoothly |
| `wce` β=1 | 0.640 | 0.765 | 0.808 | 0.801 | 1.56 → 2.38× | oscillates |

`brats_demo.md` records this same `ce` arm as predicting **zero tumour pixels
across all 2,569 val slices**. It is now the best *and* best-calibrated arm
(94.9% precision at 85.3% recall). ▶ So `wce`'s over-prediction, which the
previous paragraph here called "still real", is real but is now **damage**: the
196× weights were compensating for the data bug, and with the bug fixed they are
pure over-correction. The constant-LR oscillation goes with them — `ce` improves
monotonically. `planning/brats_demo.md` now carries a STOP banner; `post_shuffle_fix.md`
§1a is marked ANSWERED and ledger item #1 closed.

⚠ Not comparable to published BraTS — this eval is slice-level over
tumour-bearing slices only (`min_tumor_px: 1` applies to val), where the
literature reports per-volume Dice. Ours is the easier number.

⚠ Best-by-val checkpointing still earns its place on both arms (`wce` peaked at
epoch 1, score 0.791359; `ce` advanced twice, 0.838 → 0.867).

Widened the same day to all six arms (3 ep each, 4 concurrent on separate GPUs
— 214 s/epoch vs 201 s solo, so ~7% for 4× throughput):

| `dice` | `dicece` | `ce` | `focal` γ=2 | `wcesqrt` | `wce` β=1 |
|---|---|---|---|---|---|
| **0.736** | 0.734 | 0.728 | 0.719 | 0.709 | 0.640 |

▶ Consequence for the demo, not just the doc: **`wce` was the shipped default
and is the worst arm.** Default moved to `dicece` in `MainUnetBratsTrain.lean`,
and the void loss-theory docstrings in that file were retracted in place — one
of them ("the single number that has predicted every arm we have run") had
already been refuted by `brats_demo.md` in July and never updated.

### ✅ DONE 2026-08-26: `tinystories` — the second `train_step_adam_seg` demo

`§4.5` predicted this would "come free off the same entry point". It does.
Data prepared (BPE 4096 → 50.3M train / 4.86M val tokens; the tokenizer went in
its own `.venv-tokenizers`, NOT the pinned `.venv`). 600-step probe on XLA,
8.49M params, B=32, T=256:

* loss 9.24 → 3.06 nats/tok (13.3 → 4.4 bits/tok), val 3.080 tracking train 3.062
* train → checkpoint → sample → BPE decode round-trips, and the sample is
  coherent English ("Once upon a time, there was a little girl named Lily…")

⭐ Note tinystories was **never exposed to the shuffle bug** — it uses
`F32.loadTokenStream` + contiguous windows and never touches `DatasetIO` /
`lean_f32_shuffle`, whose hardcoded 4-byte label stride is what voided BraTS.
Its prior results stand.

⚠ 600 of a 12,000-step recipe — this is an entry-point gate, not a trained model.

### ✅ DONE 2026-08-26: `unet-brats-r34` — blueprint §10.2.3's anchor demo, on XLA

The third and last `train_step_adam_seg` demo. 24.5M params, 224², 502 graph
outputs, `scratch` (He-init) arm — the arm blueprint §10.2.3 quotes. Data built
at `data/brats224` (`--size 224 --seed 0`); split.json confirms the SAME val
patients and 14,415 / 2,569 slice counts as `data/brats`.

| | mIoU | WT | TC | ET |
|---|---|---|---|---|
| blueprint §10.2.3, 10 ep (IREE, 2026-07-25) | 0.740 | 0.910 | 0.869 | 0.856 |
| **this run, 3 ep (XLA, 2026-08-26)** | **0.730** | **0.908** | **0.856** | **0.844** |

Within 0.010–0.013 on every metric. ⚠ Read that correctly: r34 has
`cosineDecay` on, so a 3-epoch run runs a **complete** compressed cosine cycle
(lr 0.001 → 0.0005 → 0) rather than the first 3 epochs of the 10-epoch
schedule. This is "a shorter recipe converging slightly lower", not "on
trajectory". A true reproduction of the table needs the 10-epoch run.

### ✅ DONE 2026-08-26: `unet-pets-train` — and it closes ledger #3

Ran on XLA the same day (7.85M params, 224², 3 classes, 3 epochs, 44 s/epoch).
**mIoU 0.649, per-class 0.712 / 0.832 / 0.404.** Compare `RESULTS.md:90`, which
recorded this exact config at mIoU 0.344 with **boundary IoU 0.000**.

⭐ So the Pets boundary collapse was the shuffle bug too — the same bug, the
same signature, and it was the collapse `brats_demo.md` originally cited as its
*motivating evidence*. Flagged in `RESULTS.md`, `planning/unet_demo_v2.md`
(which had zero mention of the bug), and `post_shuffle_fix.md` ledger #3.

⚠ The skip ablation there is **unresolved, not inconclusive**: the skipless
`autoencoder-pets-train` arm needs `train_step_adam`, still `not_ported`.

▶ **All four available `train_step_adam_seg` demos are verified on XLA** — the
from-scratch BraTS UNet, `tinystories`, `unet-brats-r34`, and `unet-pets-train`
— plus `tinygpt-shakespeare`, the original gate. §4.4's "unlocks 5 demos across
two unrelated families" is fully confirmed.

## 5. ▶ STEP ORDER, by risk

1. **Tiers 1+2 in one pass (13 demos, no shim change).** Mechanical: hardcoded `.vmfb` →
   `graphArtifact pfx suffix`, delete local `iree-compile` subprocesses. Gate: each demo runs on
   **both** backends and IREE is unchanged.
2. **`train_step_adam_seg`** — the 5-demo stub. Follow the DDPM adapter's shape: a pure adapter
   over `iree_ffi_invoke_f32` that makes no PJRT call of its own, so it inherits the G4
   output-count guard (`detector_pjrt_port.md` §9).
3. **`train_step_adam`** (3 demos), then **`train_step_adam_yolov1`** (4).
4. ⛔ **Do NOT port `train_step_mlp` / `train_step_generic` / `train_step_softlabel` speculatively** —
   no demo in this sweep needs the first two, and softlabel only activates under mixup/cutmix.
   A loud stub is better than untested marshalling in the trusted shim.

### 5.1 ⭐ The gate, and why trajectory agreement is the wrong one

`detector_pjrt_port.md` §9 settled this and it should not be re-litigated per demo. Training
trajectories **diverge legitimately** across backends (8.3e-5 by step 100, 3.4e-2 by step 700) —
that is chaos from already-divergent weights, not a wrong graph. The decisive test is a
**fixed-parameter forward on the same checkpoint**, which gave max abs 1.45e-4 on logits spanning
[-134.7, 63.8] (7e-7 of full scale).

▶ Per demo, four gates, inherited verbatim:

| gate | what |
|---|---|
| **A** | IREE unchanged — bit-identical losses pre/post edit (the edit must be a no-op on IREE) |
| **B** | same function — fixed-param forward on one checkpoint, both backends |
| **C** | speed — the FPN port measured **9.47×**; expect large but do not assume |
| **D** | task metric — mAP / val loss / sample quality identical within noise |

⚠ Gate A is the cheap one and it is the one that catches a botched `graphArtifact` swap: on IREE
every changed expression must evaluate to the identical string.

---

## 6. ⚠ WHAT THIS DOES NOT COVER

* **IREE is not being retired, and must keep working.** `vjp_oracle` is the differential compiler
  for the 14 layer-family axiom checks and is IREE-only by design; `lake run {mnist,cifar,
  imagenette}-iree` are deliberate mirrors of their XLA peers. ▶ This port is "demos gain XLA",
  never "demos lose IREE" — hence gate A on every one.
* **`iree-compile` is not on PATH on this box** (it resolves via `.venv/bin` inside
  `runDemoGroup`). Any tier-1/2 demo that keeps its own subprocess will keep tripping on that;
  deleting the subprocess is part of the fix, not incidental to it.
* **No claim about which backend is *better* for a given demo.** Gate C measures it per demo.
  The FPN detector's 9.47× is one shape on one box.
* **This is not a correctness argument about the graphs.** Every demo here is a `Train.lean`
  demo, not a verified-render trainer — the proof obligations live elsewhere and are untouched.

---

## 7. File index

* `planning/detector_pjrt_port.md` — the precedent. §9 is the outcome, the two things that plan got
  wrong, and the gate design this document reuses wholesale.
* `planning/xla_pjrt_ladder.md` — had the "specialised entry points are the blocker" call right
  (rung 4) when `detector_pjrt_port.md` had it wrong.
* `LeanMlir/Train.lean:83` — `graphArtifact`, the driver-layer fix; `:911-961` — the dispatch that
  decides which specialised entry point a demo needs.
* `ffi/pjrt_ffi.c:1677-1830` — the seven specialised entry points, six of them `not_ported`.
* `ffi/lowerer.c` — the dlopen dispatch; `$LEAN_MLIR_LOWERER` = `xla` (default) | `iree`.


## 4.6 ✅ `train_step_adam` PORTED 2026-08-26 — and §4.4 mis-assigned one demo

Implemented in `ffi/pjrt_ffi.c` against the `train_step_adam_seg` template. The
two differ in exactly one input — the label tensor's rank:

```
seg    %y_seg : tensor<b x H x W x i32>   one label per PIXEL
adam   %y     : tensor<b x i32>           one label per EXAMPLE
```

⚠⚠ **The arity was read off the emitted graph, not inferred — and inferring it
would have been wrong.** `bigram_shakespeare_train_step.mlir` @main is
`(params…, %x_flat, %y, %lr, %t)` = np + 4. But `resnet50_train_step.mlir` has
**no `%t` at all**. The difference is that the verified nets reach XLA through
the *generic* `iree_ffi_invoke_f32` and are not clients of this entry point;
copying their shape would have produced a silent arity mismatch.

⛔⛔ **`ffi/libpjrt_ffi.so` IS NOT A LAKE TARGET.** `lake build` reports success
without rebuilding the shim, so an edited `pjrt_ffi.c` gives a **false green**
and the demo still prints `not_ported`. It cost a debug cycle here.
`lakefile.lean:2739` documents this; the rebuild is:

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
```

### Verification

| gate | result |
|---|---|
| `bigram-shakespeare train 3` | loss 2.619 → **2.473 nats** — the bigram optimum for tinyshakespeare's 65-char vocab (random = ln 65 = 4.17). A wrong gradient does not land on the known optimum. |
| `bigram-shakespeare sample` | character-level Shakespeare structure (`ROMEO:`, line breaks, punctuation), no word-level coherence — exactly a bigram's ceiling |
| `grad-fd-probe mlp` | 91.65% MNIST, `BN layers: 0` |
| `grad-fd-probe convbn` | 96.81%, `BN layers: 1` — first exercise of the `have_bn` output branch |
| `grad-fd-probe res` | 97.81%, `BN layers: 3` |

⭐ The BN rungs matter specifically: the eval is "val accuracy (**running BN**)",
so a misplaced BN stat output would surface as a garbage eval rather than
passing quietly. Both BN rungs land on correct numbers for their architecture.

⚠ The proper gate — `scripts/grad_fd_bisect.py`, which compares the emitted
gradient against finite differences using the probe's exact-step property — was
**NOT run**: it needs `data/mnist16` (absent here) and hardcodes ROCm env
(`IREE_BACKEND=rocm`, `HIP_VISIBLE_DEVICES`). That remains the strongest
available check on this port and is worth doing.

### ⚠ §4.4 mis-assigned `autoencoder-pets-train`

It is listed under `train_step_adam`. It is a **3-class trimap segmentation**
net, so `lossKind.isSeg` holds and it dispatches to `train_step_adam_seg` — it
has been runnable since that port and was never blocked. Verified: it trains on
XLA at exit 0.

▶ Consequence: **the Pets skip ablation was never blocked either**, and it is
now resolved (§4.7).

## 4.7 ✅ The Pets skip ablation, resolved — Gate B passes

3 epochs each, matched config, on shuffle-fixed data:

| decoder | mIoU | bg | fg | **boundary** |
|---|---|---|---|---|
| Autoencoder (skipless) | 0.596 | 0.667 | 0.800 | 0.320 |
| **UNet (skips)** | **0.649** | 0.712 | 0.832 | **0.404** |

`RESULTS.md` concluded from void data that "UNet does **NOT** yet beat the
skipless autoencoder (0.344 vs 0.360)... inconclusive". On correct data UNet
wins by **+0.053 mIoU and +0.084 on the boundary class** — Gate B passes.

⭐ Same signature as the BraTS skip ablation (+0.105 mIoU, **+0.123 on ET**, the
thinnest structure): skips help, and they help most on the thin class. Two
datasets, same mechanism, and both were invisible under the shuffle bug.
