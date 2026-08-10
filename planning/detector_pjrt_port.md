# detector_pjrt_port.md — the VisDrone FPN detector on XLA/PJRT

**Written 2026-08-06.** Plan for a session that moves the FPN detector off IREE onto the
XLA/PJRT backend. Companions: `planning/visdrone_detector.md` (the detector itself),
`planning/coco_visdrone_two_stage.md` (what this unblocks, §4b), `planning/xla_pjrt_handoff.md`
(the backend, and the §2h recipe this follows).

Status: **DONE 2026-08-07 — all four gates green.** See §9 for results and for the two
places this plan was wrong. Body below is the original plan, kept as written.

---

## 9. Outcome (2026-08-07)

**All four gates pass. The detector trains on XLA at 9.47× IREE.**

| gate | result |
|---|---|
| **A** IREE unchanged | **bit-identical** — all 11 logged losses *and* the epoch mean (391.078521) match pre-port exactly |
| **B** same function | step 0 rel err **4.6e-6**; fixed-param forward (below) max abs **1.45e-4** over 101.5M logits |
| **C** speed | **9.47×** — 175.4 vs 1660.7 ms/step; 12-epoch run **4.47 h → 28.3 min** |
| **D** infer | identical mAP **0.1128** both backends; 1 detection differs out of 493,093 |

### What this plan got wrong

1. **The blocker was not `Train.lean`.** Threading `mkSession` was necessary but not
   sufficient. The FPN graph *compiles fine* on PJRT (28.8 s, `@main, 424 outputs`) — neither
   the upsample nor the reduction was a problem. The real blocker was that **all seven**
   specialised train-step entry points in `ffi/pjrt_ffi.c` are `not_ported` stubs. The verified
   nets reach XLA through the *generic* `iree_ffi_invoke_f32`; the `Train.lean` demos go through
   the specialised ones, and none existed. `planning/xla_pjrt_ladder.md` had this right as
   rung 4 — the two docs disagreed and the ladder was correct.
2. **It missed a 9th site** (`Train.lean:1292`, the train session) and, more importantly,
   mislocated the risk: the four eval sites are wrapped in `if ← pathExists evalVmfb`. On XLA
   no `.vmfb` exists, so three of those guards go **silently false** — training looks healthy
   and just stops reporting val metrics. That is worse than a load failure.

### What was actually built

* `LeanMlir/Train.lean` — new `NetSpec.graphArtifact pfx suffix`, returning `.vmfb` on IREE and
  the `.mlir` on XLA. Every guard and `IreeSession.create` routes through it, so the guards keep
  working. Chosen over adopting `VerifiedTrain.mkSession`, whose IREE branch would have changed
  this file's cache semantics and path scoping for all 39 consumers. On IREE every changed
  expression evaluates to the identical string — that is why Gate A is bit-identical.
* `ffi/pjrt_ffi.c` — **`iree_ffi_train_step_adam_ddpm` implemented** as a ~70-line pure adapter
  over `iree_ffi_invoke_f32`. It makes no PJRT call of its own, so it inherits the G4
  output-count guard. Layout check: 351 params + x + y + lr + t = **355 inputs**;
  351 + loss + 72 BN = **424 outputs**, which is what the compiler reports.
  ⚠ The other **six** stubs are still `not_ported` — deliberately. They now have a working
  template, but shipping untested marshalling into the trusted shim is worse than a loud stub.
* `demos/Yolov1VisdroneFpnCommon.lean` + two thin roots + `lean_lib` (a new shared module needs
  its own lake target, same as `Resnet34AdamCommon`); `lean_exe yolov1-visdrone-fpn`.

### The fixed-parameter check — the real proof of "same function"

Training-trajectory agreement decays (8.3e-5 by step 100, 3.4e-2 by step 700) because each side
trains from already-divergent weights; that is chaos, not a wrong graph. The decisive test ran
`infer` on **the same checkpoint** on both backends:

* max **abs** diff **1.45e-4** on logits spanning [-134.7, 63.8] — 7e-7 of full scale; mean 1.6e-6
* max *rel* diff reads 5.78, but **98.6%** of the elements exceeding rel 1e-3 have |logit| < 1e-2
  (largest 0.049) — near-zero denominators, where relative error is meaningless. Verified, not assumed.

### Gotchas this cost time on

* **`IREE_BACKEND=rocm` is required** and defaults to `cuda`. Without it `iree-compile` dies with
  `'func.func' op failed to distribute` on the multi-scale loss reduction — the workaround flag in
  `ireeCompileArgs` is gated on `rocm`. This looks exactly like a codegen regression and is not one.
* **`pgrep -f "lake build <target>"` matches its own shell** (this plan's own gotcha #4, via pgrep
  rather than pkill). Use a bracketed pattern.
* `| tail` on a build swallows all output if the build is killed — redirect to a file.
* On XLA `runIreeCached` logs `(cached)` when nothing was compiled or cached. Cosmetic, but it
  reads as though an IREE artifact was reused. Worth a backend-aware message.

### Left open

* The six remaining `not_ported` stubs — needed before the other 38 `spec.train` consumers can
  move. `adam_ddpm` is the template.
* **Pre-existing, both backends:** `WARN: bootstrap BN stats size 87190688 ≠ expected 68096; using
  zeros` fires on every detector run. Not caused by this port and it cancels out of the
  comparison, but the R34 bootstrap BN stats are not reaching the detector.
* `runs/gateD_{iree,xla}/logits.bin` are 406 MB each — delete when done.
* §6 step 7 (the COCO smoke on `data/coco_small`) not run.

---

## 0. TL;DR

The detector runs on IREE at **1660 ms/step** (bs 8) — 1341 s/epoch over 808 batches, measured
in `runs/fpn_long30_gpu0.log`. XLA is **6–10.5× faster on this box** on comparable nets. So a
12-epoch VisDrone run goes from **4.5 h to roughly 30–45 min**, and the COCO→VisDrone two-stage
experiment stops needing the PyTorch twin as its primary vehicle.

**The blocker is not the link flag.** The verified nets are backend-agnostic because they go
through `VerifiedTrain.mkSession`. The detector goes through `Train.lean`'s `train`, which is
**IREE-hardcoded in 8 places**. The work is threading `mkSession` through that file. It is a
small change in one file with a **39-file blast radius**, so the plan is mostly about not
breaking the other 38.

---

## 1. Why — the payoff is measured, not assumed

| comparison | IREE | XLA | ratio | source |
|---|---|---|---|---|
| R34 / Imagenette bs32 | 1702 ms/step | 162 ms/step | **10.5×** | handoff §L409 |
| ViT bs32 | 1188 ms/step | 128 ms/step | **9.2×** | handoff §0 |
| **whole Part-1 tier** | **42.8 h** | **7.2 h** | **~6×** | `lake run benchmark` vs `benchmark-xla`, both re-run 2026-08-05, every probe within 6% of its reference |

The detector is R34 + FPN, so R34 is the closest analogue. Projected at bs 8:

| | IREE (measured) | XLA @6× | XLA @10× |
|---|---|---|---|
| ms/step | 1660 | ~277 | ~166 |
| VisDrone epoch | 1341 s | ~224 s | ~134 s |
| 12-epoch run | 4.5 h | **~45 min** | **~27 min** |

⚠ Both projections are extrapolations from *other nets*. The measured ratios are at bs 32; the
detector is bs 8, and XLA's advantage generally grows with batch — so expect the low end. The
first honest number comes from Gate C, not from this table.

---

## 2. The actual blocker, with the site inventory

Two training paths exist and only one is backend-aware:

* **`LeanMlir/VerifiedTrain.lean`** — the five verified nets. `mkSession (mlirPath outVmfb)`
  (`:199`) branches on `IreeSession.backendName`: **XLA** hands the `.mlir` straight to PJRT
  (compiled in-process), **IREE** runs `iree-compile` to a `.vmfb` first. Backend-agnostic.
* **`LeanMlir/Train.lean`** — the demos and baselines, including the detector. **IREE only.**

The eight sites in `Train.lean`:

| line | what |
|---|---|
| `73` | `private def runIreeCached` — the `iree-compile` wrapper |
| `229` | `runIreeCached … _fwd.mlir → _fwd.vmfb` |
| `232` | `runIreeCached … _fwd_eval.mlir → _fwd_eval.vmfb` |
| `235` | `runIreeCached … _train_step.mlir → _train_step.vmfb` |
| `1000`, `1101`, `1159`, `1235` | `IreeSession.create evalVmfb` — four eval-session sites |

Plus one in the demo itself: `demos/MainYolov1VisdroneFpn.lean:190` calls
`IreeSession.create evalVmfb` directly for the `infer` subcommand.

**The shape of the fix:** on XLA, `runIreeCached` should be a no-op that reports the `.mlir`
path, and every `IreeSession.create <x>.vmfb` becomes
`mkSession "<x>.mlir" "<x>.vmfb"`. That is exactly what `mkSession` already encapsulates —
this is *adopting* an existing abstraction, not inventing one.

---

## 3. Blast radius, and why it is safe anyway

**39 files** call `spec.train`: every `apps/baselines/*` (VGG, R50, ConvNeXt, EfficientNet(V2),
ViT-Muon/Shampoo, CIFAR CNN ±BN, ResNet), both MNIST grid apps, `apps/cifar/MainCifar*Verified`,
`apps/imagenette/Main{EfficientNet,ViT}Verified`, and five demos (UNet/BraTS, UNet/Pets,
autoencoder/Pets, YOLO/Pets bootstrap, YOLO/VisDrone FPN).

The safety argument is that **`mkSession` is conditional on a symbol probe, not a config flag**:
`IreeSession.backendName` is resolved from which shim the binary linked (a weak symbol — a binary
cannot disagree with its library). So for all 38 other consumers, which still link `ireeLink`,
`mkSession` takes the IREE branch and does what `runIreeCached` + `IreeSession.create` do today.
**Behaviour on IREE must be bit-identical; that is Gate A.**

---

## 4. Risks, in the order they will bite

1. **FPN upsampling.** The detector uses ops none of the six verified nets do: the multi-scale
   loss, cross-scale concat, and the P5→P4→P3 upsample path. Upsampling can lower to a transposed
   or **dilated** convolution — the exact descriptor class behind the ViT/MIOpen failure
   (a 14×14 filter at `rhs_dilation = 16`, selecting `ConvDirectNaiveConvFwd`). If the port dies
   in the conv, that is the first thing to check.
2. **Cold MIOpen kernel cache.** That ViT failure is *cache-dependent*: with a warm
   `~/.cache/miopen` it passes bit-exact, with an empty one it fails deterministically with
   `miopenStatusUnknownError`. So a green run here does **not** prove a fresh machine works.
   Re-run the gate with `MIOPEN_CUSTOM_CACHE_DIR` pointed at an empty dir before believing it.
3. **`.vmfb` paths leaking into demo code.** `MainYolov1VisdroneFpn.lean:190` builds an explicit
   `_fwd_eval.vmfb` string. On XLA no `.vmfb` is ever produced, so that path must go through
   `mkSession` too or `infer` breaks while `train` works — a split failure that is easy to miss
   because training is what gets exercised first.
4. **`IREE_BACKEND` is inert on the XLA path.** The vmfb scoping at `VerifiedTrain.lean:~210`
   reads it; PJRT does not. Anything that sets it expecting an effect will silently get none.

---

## 5. Gates — each must be able to fail

* **Gate A — IREE is unchanged.** Before/after the `Train.lean` edit, run one IREE demo
  (`yolov1-visdrone-fpn`, 1 epoch, fixed seed) and require the per-step losses to be
  **bit-identical**. This is the gate that protects the other 38 consumers, and it is the one
  worth writing first. A "looks the same" eyeball is not this gate.
* **Gate B — the XLA detector computes the same function.** Same fixed seed and init, one epoch,
  IREE vs XLA: first N step losses must agree to ~1e-5. This is the §2h cross-backend check every
  verified net had to pass. **Speed without this gate is worthless** — a 10× speedup on a graph
  computing something slightly different is worse than no speedup, and this thread has already
  lost one run (`long30`) to a silent mismatch.
* **Gate C — the speed claim.** Median ms/step over ≥100 steady-state steps, XLA vs the measured
  IREE 1660. Report the ratio; do not reuse §1's projection.
* **Gate D — `infer` still works on both backends.** Run the inference dump and score it; mAP
  must match the pre-port number on IREE and be finite on XLA.

---

## 6. Run order

```bash
# 0. baseline FIRST, before touching anything — Gate A needs a before-image
FPN_TAG=preport FPN_EPOCHS=1 .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn \
  > runs/fpn_preport_iree.log 2>&1

# 1. thread mkSession through LeanMlir/Train.lean (8 sites, §2)
# 2. Gate A: re-run step 0 with a different tag, diff the loss column — must be bit-identical
# 3. add the exe:  lean_exe «yolov1-visdrone-fpn» with moreLinkArgs := xlaLink
#    (§2h recipe: a Common module split if the demo body needs sharing)
# 4. Gate B: same seed, 1 epoch, IREE vs XLA losses to ~1e-5
# 5. Gate C: ms/step on the XLA build
# 6. Gate D: infer + score on both
# 7. THEN the COCO smoke — data/coco_small is already cut (2,000 train / 400 val, 3.2 GB,
#    --check passed), and its records are byte-identical in geometry to data/visdrone_fpn
#    (1,342,992 B/record, Ntot 185,220), so it is a path argument and nothing else.
```

---

## 7. Gotchas carried in

* **`FPN_TAG` on every train AND eval.** The missing-tag trap scored the wrong checkpoint six
  times and voided the entire `long30` run.
* **Checkpoints live in `.lake/build/<sanitizedName>[_<tag>]_params[_e<N>].bin`** and are
  per-variant. They **outlive the render that trained them** — if the render generation changed,
  park the checkpoint rather than resuming onto it (this bit the Imagenette re-runs on 2026-08-06:
  Jul-31 checkpoints against Aug-2 renders).
* **`| head` does not stop a trainer** — SIGPIPE detaches it, it keeps burning GPU. Redirect to a
  file and read the file.
* **`pkill -f <pattern>` can match your own shell** and kill the command issuing it. Use a bracket
  (`resnet34-imagenet-verified-x[l]a`) or kill by PID.
* **Build the exe you are about to run.** `lake build A` will not rebuild exe B.
* **PJRT plugin resolution on this box** works only via the third entry of `kDefaultPlugins`
  (`/home/skoonce/lean/claude_max/lean4-jax/.venv/…/xla_rocm_plugin.so`). It resolves today; the
  repo-local `.venv` paths do not exist here. A repo-local symlink would make this robust.

---

## 8. What would make this not worth doing

* Gate B cannot be made to pass — the detector's op set lowers differently on XLA and the graphs
  genuinely disagree. Then the port is a research task, not an engineering one; fall back to the
  PyTorch twin (`demos/visdrone`, ~26× IREE) as the two-stage vehicle.
* Gate C comes in under ~2×. The projection is 6–10×, but at bs 8 with a small per-step workload
  the all-reduce-free single-device case may not amortise XLA's advantages. Under 2×, the
  `Train.lean` churn is not worth it for this net alone — though note the other 38 consumers
  would inherit the option for free.
