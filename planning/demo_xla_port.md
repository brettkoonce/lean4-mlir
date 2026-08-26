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
| `train_step_adam` | `bigram-shakespeare`, `autoencoder-pets-train`, `grad-fd-probe` |
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

### ▶▶ NEXT: `unet-brats-train` — the validation that matters

`train_step_adam_seg` is proven on **tinygpt only**, and tinygpt's label is
`[B, T, 1]` — it never exercises `W > 1`. UNet/BraTS is the real per-pixel case
and is what should gate the entry point before anything else builds on it.

```
CUDA_VISIBLE_DEVICES=0 .lake/build/bin/unet-brats-train data/brats 3
#                                                       ^dataDir  ^epochs (default 3)
```

Data is prepared (`data/brats/{train,val}.bin`, ~4.2 GB / 0.74 GB). What to
check, in order:

1. It reaches the training loop at all — i.e. no `not_ported`, which is the
   thing the port fixed.
2. Loss decreases. ⚠ Not sufficient on its own: a mis-indexed label tensor
   descends too. This is exactly why tinygpt was gated on *sampling* rather
   than on loss.
3. **mIoU + per-class IoU**, which this trainer already prints every 10 epochs.
   That is the real gate — the labels are either aligned or the minority class
   IoU is zero. ⚠ Read `MainUnetBratsTrain.lean:186-195` first: γ=2 is a
   documented **collapse setting** on this data, so a zero there may be the
   loss config rather than the port. Compare against the IREE-era numbers
   before concluding anything about the port.
4. Then `unet-brats-r34` and `tinystories` come free off the same entry point.

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
