# coco_visdrone_two_stage.md — COCO pretrain → VisDrone fine-tune

**Written 2026-08-01.** Plan for a session that builds a two-stage detector: pretrain the
FPN detector on MS-COCO, fine-tune on VisDrone. Companion to `planning/visdrone_detector.md`
(the single-stage detector, which is the baseline this must beat) and the memories
`yolo-fpn-thread`, `visdrone-pytorch-twin`, `visdrone-fetch-and-wsa`.

> **⚠ STATUS CORRECTED 2026-08-28 — the built data is GONE, and step 0 is DONE.**
>
> - `data/coco_vd/` (94 GB) and `data/coco/` were deleted to free disk, along with
>   `data/visdrone*`. VisDrone has been rebuilt; **COCO has not**. The "already on
>   disk" claim below is false — rebuilding costs a 19 GB fetch (~50 min at 12
>   ranged connections) plus preprocessing, and §5's disk warning still applies.
>   Note the transfer build reads `data/visdrone` for its anchors, so **VisDrone
>   must exist first**.
> - **§4b is satisfied.** "Step 0 is not a training run: port the detector to
>   XLA/PJRT" was completed 2026-08-07 at **9.47×** (`detector_pjrt_port.md`). The
>   §9 run order still lists it as a to-do; ignore that line.
> - The box is now **6× RTX 4060 Ti on CUDA**, so arms run in parallel and the
>   `IREE_BACKEND=rocm` lines here are dead weight. But the **host data loader is
>   the shared bottleneck** — epochs stretch from 208 s to ~400 s at five
>   concurrent trainers, so parallelism is sublinear.
> - **The control this must beat is being re-measured right now** (see
>   `visdrone_detector.md` §12b). Do not run stage 2 against the stale 0.1386
>   until `ctrl12` lands, because the rebuilt R34 backbone is a *different, better*
>   checkpoint than the one behind that number.
>
> **This remains the lowest-priority thread**, on its own evidence: §3's inverted
> scale distributions, and §10's prediction that arm B ties arm A. The augmentation
> A/B and the long-schedule question are cheaper and are running first.

Status: ~~**DATA PREP IS DONE (2026-08-01).** The COCO transfer set is built, checked and on
disk;~~ everything below the data section is still plan.

```
data/coco_vd/train.bin   70,082 records   94.12 GB   (--check passed)
data/coco_vd/val.bin      2,968 records    3.99 GB   (--check passed)
                          1,342,992 B/record — byte-identical to data/visdrone_fpn
```

Built with `--classes vdmap --fpn data/visdrone --size 448 --grid 14`, i.e. VisDrone's own
anchor priors and index space (§2), so the existing trainer reads it with a path change.
Measured: 332,526 GT boxes over 70,082 images, 97.7% encoded into unique multi-scale
slots, **0 records skipped** across 118,287 real JPEGs.

---

## 0. TL;DR — read this before scheduling anything

The idea is sound and the plumbing already exists (`preprocess_coco.py --classes vdmap`
was built for exactly this). But three things in the obvious version of the plan are
wrong by large factors, and all three are measured, not guessed:

1. **500 epochs is a real convention but the wrong one *for this configuration*, and it
   costs ~84 days on the Lean trainer.** 500/100 is the YOLO/DETR lineage — from scratch,
   with heavy augmentation. Here the backbone is already ImageNet-pretrained and the aug
   pack has no mosaic, which puts us in the Detectron2 1×/3× regime (12/36) instead; and
   `long30` already plateaued at epoch 10 on this data. The schedule is coupled to the
   aug decision — see §4, it is the most important call in this plan.
2. **The COCO build needs 94 GB of disk** and the box has 170 GB free at 91% used. §5.
3. **The scale distributions are nearly inverted**, so the mechanism most people assume
   ("COCO gives more small-object examples") is *false* here. VisDrone is P3-dominated;
   COCO is P5-dominated. §3. This does not kill the idea, but it changes what the
   experiment is testing and it lowers the prior.

**Step 0 is not a training run: port the detector to the XLA/PJRT backend (§4b).** It is
measured at **10.5× IREE on ResNet-34** — the detector's own backbone — and the shim is
drop-in by construction. That turns a 12-epoch VisDrone run from 4.5 h into ~27 min and
makes the whole two-stage experiment feasible *in Lean*, without a PyTorch round trip.
If the port stalls, fall back to developing on the twin (`demos/visdrone`, ~26×) and
porting after — the practice this repo already validated for the single-stage detector. §4.

---

## 1. What already exists (nothing here needs building)

| piece | where | state |
|---|---|---|
| COCO → VisDrone-index-space preprocessor | `preprocess_coco.py --classes vdmap` | built, validated |
| COCO fetch | `download_coco.sh` | built; uses the S3 path-style URL (the canonical host fails TLS) |
| per-scale anchors | `scripts/coco_anchors.py` | built — **but see §2, we do NOT use COCO's own anchors** |
| FPN trainer | `lake exe yolov1-visdrone-fpn` (`demos/MainYolov1VisdroneFpn.lean`) | trained to mAP 0.1386 |
| PyTorch twin | `demos/visdrone` | reproduces mAP 0.1531, ~26× faster |
| scorer | `scripts/yolo_map_visdrone.py --fpn ... --grid 14` | scores against the uncapped GT sidecar |
| aug pack | `FPN_AUG=1` (committed `9280a3d`) | **never A/B'd at scale** — a confound, see §7 |

The **baseline to beat is mAP@0.5 = 0.1386** (full GT, 12 epochs, recall 0.676,
class-agnostic AP 0.376), with an ImageNet-pretrained R34 backbone this stack trained
itself to 72% top-1. Note that the backbone is *already pretrained* — so this experiment
is testing the marginal value of COCO pretraining **on top of** ImageNet pretraining, not
against a cold start. That is a much narrower claim than "pretraining helps".

---

## 2. The design that makes the head transferable

`--classes vdmap` emits COCO **into VisDrone's own 10-class index space**:

```
person→pedestrian(0)  bicycle(2)  car(3)  truck(5)  bus(8)  motorcycle(9)
```

Four VisDrone classes have no COCO counterpart and stay permanently empty during stage 1:
**people(1), van(4), tricycle(6), awning-tricycle(7)**.

Two consequences, both load-bearing:

* **Use VisDrone's anchors for the COCO build — non-negotiable.**
  `--fpn data/visdrone`, not `data/coco`. The head regresses residuals *off the anchor
  priors*; if the two stages use different priors, every box-regression weight is
  meaningless on transfer and only the backbone carries over. `scripts/coco_anchors.py`
  exists for studying COCO in its own right; it must NOT be used here.
* **Four class channels get zero positive gradient in stage 1.** Their logits are pure
  prior at the start of stage 2. Gate D (§6) exists specifically to check that stage 1
  does not *hurt* those four relative to the single-stage baseline.

With VisDrone's anchors and VisDrone's index space, the COCO record geometry is
**byte-identical** to the VisDrone FPN build (1,342,992 B/record, Ntot=185,220), so the
existing trainer reads it with a path change — no codegen, no Lean, no C change.

---

## 3. The measurement that should lower your prior

Both rows measured, not estimated (COCO from `instances_val2017.json`; VisDrone from all
343,204 eval-relevant train boxes):

| GT scale assignment @448, thresholds 24/64 px | P3 (56²) | P4 (28²) | P5 (14²) |
|---|---|---|---|
| **VisDrone** train | **77.2%** | 20.8% | 2.0% |
| **COCO** vdmap | 17.2% | 28.5% | **54.3%** |

They are nearly inverted. COCO spends 54% of its supervision on the scale that carries
**2%** of VisDrone's objects, and only 17% on the scale carrying 77%.

Worse for the naive story, **VisDrone is not data-poor at its dominant scale**: 77.2% of
343,204 = **264,953 P3 boxes** already. Now measured on the real full build rather than
estimated: COCO vdmap contributes **332,526 boxes**, of which 17.5% = **58,192 are P3** —
a **22.0% increase** in P3 supervision, not a step change. (Total box counts are
comparable — 332,526 vs 343,204 — over 10.8× the images.)

**So the mechanism cannot be "more small objects."** If COCO pretraining helps here, it
helps by giving the backbone/neck better generic object features — on top of a backbone
that is *already* ImageNet-pretrained. That is a real but modest hypothesis, and it is
worth stating plainly before spending days on it.

It also suggests a cheaper variant worth testing in the same session (§9, arm C):
**pretrain on COCO but keep only P3/P4 supervision**, or re-threshold COCO so more of its
boxes land on P3. The preprocessor already supports `--fpn-thresh LO HI`, and it prints
the resulting histogram, so this costs one flag.

---

## 4. Schedule arithmetic — why "500 epochs" is wrong

Measured throughput, Lean/IREE FPN trainer, single GPU, from `runs/fpn_long30_gpu0.log`:

```
Epoch 1/30: loss=391.078521 ... (1337419ms)
Epoch 30/30: loss=66.359002 ... (1347516ms)
```

**1,341 s/epoch on 6,471 images at bs=8 ⇒ 4.83 img/s.** Flat across 30 epochs, so it
extrapolates cleanly by image count.

COCO vdmap train2017 ≈ **70,200 images** (measured keep rate 59.4% of 118,287) —
**10.9× VisDrone's image count**, so ~4.0 h per COCO epoch on the Lean trainer.

| stage | epochs | Lean trainer | PyTorch twin (÷26) |
|---|---|---|---|
| COCO pretrain | **500** (DETR-style, as proposed) | **~84 days** ❌ | ~3.2 days |
| COCO pretrain | 300 (YOLOv5/v8/YOLOX default) | ~50 days ❌ | ~1.9 days |
| COCO pretrain | 36 (3× schedule) | ~6.1 days | **~5.6 h** |
| COCO pretrain | **12 (1× schedule)** | ~2.0 days | **~1.9 h** |
| VisDrone fine-tune | **100** (as proposed) | ~37 h | ~1.4 h |
| VisDrone fine-tune | 12 (baseline-matched) | ~4.5 h | ~10 min |

Note the twin column makes the long schedules *possible* — 300 epochs is ~1.9 days there,
not 50 — so if the aug question gets answered first, the YOLO-standard schedule is
affordable on the twin and only the Lean port stays out of reach. On the Lean trainer,
anything past ~36 COCO epochs is off the table until the trainer gets faster.

Two corrections to the proposal:

* **500 → 12 or 36 *in this configuration* — but the number is coupled to augmentation,
  not free-standing.** There are two live conventions and they apply to different regimes:

  | convention | epochs | regime it belongs to |
  |---|---|---|
  | YOLOv5/v8/YOLOX | 300 | from scratch, **heavy aug** (mosaic, mixup, scale jitter) |
  | DETR (original long schedule) | **500** | from scratch, slow-converging set-matching |
  | Detectron2 / mmdet 1× / 3× | **12 / 36** | head on an **ImageNet-pretrained backbone** |

  500/100 is a real convention — it is the YOLO/DETR lineage, and 100-epoch VisDrone
  fine-tunes are common there. But that lineage's long schedules are inseparable from
  heavy augmentation: the aug is what keeps a 300–500 epoch run from simply memorising.

  This setup sits in the *third* row on both deciding axes: the R34 backbone is already
  ImageNet-pretrained by this stack (72% top-1), and the aug pack is minimal (HSV +
  hflip, **no mosaic**) and off by default.

  And there is a local measurement: `long30` **plateaued in val mAP at epoch 10 while
  train loss halved through epoch 30** (`planning/visdrone_detector.md` §6.1) — this
  dataset, this head, this aug setting. Running 500 in that configuration buys more
  overfitting, not accuracy.

  **So the schedule decision is really an aug decision.** If a long schedule is wanted,
  it has to be bought with real augmentation first — and the current pack has no mosaic,
  which is the single most important one for a YOLO-style long run. Sequence it: get
  signal at 12–36 on the twin, and if the val curve is *still climbing* at 36 (unlike
  VisDrone's plateau at 10), extend and reconsider. That is an empirical call, not a
  dogmatic one.
* **100 → 12 for the headline A/B**, because the control is the 12-epoch 0.1386 run and
  **the fine-tune must be schedule-matched to it** or the comparison is confounded. Run a
  longer fine-tune afterwards as a separate arm if 12 looks promising.

## 4b. The lever that changes everything: put the detector on XLA/PJRT

The detector runs on **IREE** (`yolov1-visdrone-fpn`, `moreLinkArgs := ireeLink`). The
XLA/PJRT backend exists, is validated on six nets, and is **measured far faster on this
box** — including on a pure CNN:

| net | IREE | XLA | ratio | source |
|---|---|---|---|---|
| **R34 / Imagenette bs32** | 1702 ms/step | **162 ms/step** | **10.5×** | handoff §L409 |
| ViT bs32 | 1188 ms/step | 128 ms/step | 9.2× | handoff §0 / §2h |

R34 is the detector's own backbone, so this is about as close an analogue as exists. The
"gfx1100 is MIOpen-conv-weak" caveat does *not* mean XLA is bad at convs here — it means
IREE is worse.

Projected for the detector (measured IREE baseline: 1341 s/epoch ÷ 808 batches =
**1660 ms/step** at bs=8):

| | IREE (measured) | XLA @10× (projected) | PyTorch twin |
|---|---|---|---|
| ms/step | 1660 | ~166 | ~64 |
| VisDrone epoch | 1341 s | ~134 s | ~52 s |
| VisDrone 12 ep | 4.5 h | **~27 min** | ~10 min |
| COCO 12 ep (70k) | ~2.0 days | **~4.9 h** | ~1.9 h |
| COCO 300 ep (70k) | ~50 days | **~5 days** | ~1.9 days |

With 2-GPU DP (measured 1.67–1.70× on mnv2 / ConvNeXt / EfficientNet) that is
approximately twin parity — i.e. **the two-stage experiment becomes a Lean-native
experiment**, and the twin reverts to being a cross-check rather than the primary
vehicle.

**Why the port should be cheap.** From the lakefile: *"`libpjrt_ffi.so` exports the
identical C surface as `libiree_ffi.so`, so nothing above the shim moves."* The §2h
recipe is 3 files + 2 lakefile entries per net with **no driver change** — split a
`Common` module out of `demos/MainYolov1VisdroneFpn.lean`, add a
`yolov1-visdrone-fpn` exe with `xlaLink`. The detector is a demo, not a verified net,
so there is no render or tie work to do.

**Two risks, both concrete:**

* The 10.5× is R34 *classification* at **bs32**; the detector is **bs8**. XLA's advantage
  generally grows with batch, so expect less than 10×. Measure before believing.
* The detector's graph uses ops none of the six verified nets do: the multi-scale loss,
  cross-scale concat, and **FPN upsampling**. Upsampling can lower to a transposed or
  dilated convolution — precisely the descriptor class behind the ViT/MIOpen failure
  (a 14×14 filter at `rhs_dilation=16` that selects `ConvDirectNaiveConvFwd` and dies on
  a cold kernel cache). If the port fails, look there first, and run with a cold
  `MIOPEN_CUSTOM_CACHE_DIR` to check whether a warm cache is masking it.

**Gate for the port:** the XLA detector must reproduce the IREE detector's first N step
losses to ~1e-5 from identical fresh init, the same cross-backend check §2h used for
every other net. Speed is worthless if the graph computes something else.

### End-to-end, the recommended plan

**⚠ The first two rows are now MEASURED and both were badly over-estimated — corrected
in place, originals struck through.**

| step | cost | note |
|---|---|---|
| download train2017 (19 GB) | ~~~10 h~~ → **50 min** | 12 ranged connections (§9a). 1 conn 0.65 MB/s, 12 conns **6.0 MB/s** |
| preprocess to 448 FPN bins | ~~~1–3 h~~ → **5m57s** | 212 img/s measured; single-threaded PIL was never the bottleneck |
| stage 1: COCO 12 ep (twin) | ~1.9 h | |
| stage 2: VisDrone 12 ep (twin) | ~10 min | |
| control arm (no pretrain) | ~10 min | must run, same seed |
| scoring + per-class analysis | ~30 min | |
| **total on the twin** | **~14–16 h**, dominated by download | one overnight |
| **same experiment, Lean + XLA** (§4b) | **~18–20 h** | ≈ the twin, and it is the real trainer |
| **same experiment, Lean + IREE** | **~2.5 days** of GPU | the status quo; avoid |

The XLA row is why §4b is step 0: it costs a few hours of porting and collapses the Lean
column to roughly the twin's, which removes the develop-then-port round trip entirely.

The download is the single largest line item and it is pure wall-clock — **start it
first**, before anything else in the session.

---

## 5. The disk problem

Record size is **1,342,992 B** (verified against `data/visdrone_fpn/train.bin`:
8,287.91 MB / 6,471 records). At ~70,200 COCO records:

```
70,200 × 1,342,992 B ≈ 94 GB          (+ 19 GB zip + ~19 GB extracted tree during prep)
free on / : 170 GB at 91% used
```

It fits, but only just, and not alongside the extracted tree. Options, in order of
preference:

1. **Subsample COCO to ~30k images (~40 GB).** With 333k boxes total, 30k images still
   gives ~142k boxes — more than enough to move a backbone, and it keeps the whole
   experiment comfortably on disk. `--limit N` already exists.
2. **Delete the extracted `train2017/` tree immediately after preprocessing** (the
   download script already prints this) — recovers ~19 GB.
3. Full 70k build only if steps 1–2 prove the effect is real and worth the disk.

**Recommendation: start with `--limit 30000`.** If the effect does not show at 30k
images, 70k is unlikely to rescue it, and the disk headroom is not worth the risk on a
91%-full volume.

---

## 6. Gates — each must be able to fail

Following the repo's rule that a gate which cannot fail proves nothing.

* **Gate 0 — data pairing.** `preprocess_coco.py ... --check`. Already built and already
  falsified (short file / swapped records / zeroed one-hot are all rejected). Non-optional:
  every recent detection bug in this repo was silent mispairing, not bad math.
* **Gate A — stage 1 actually learned.** COCO-val mAP@0.5 on the **six populated classes**
  must be clearly > 0 after stage 1. If it is ~0, the pretrain is noise and stage 2 is
  measuring nothing. (Score with `yolo_map_visdrone.py` — it is index-space agnostic.)
* **Gate B — the checkpoint actually loads (plumbing, not science).** Load the stage-1
  checkpoint into the VisDrone trainer and evaluate on VisDrone **with zero fine-tuning**.
  Expect *low but non-zero* mAP on the shared classes (car, bus, truck, pedestrian).
  **Exactly 0.0000 means a plumbing bug**, not a domain gap — that is the failure mode
  that has bitten this thread twice. Do not proceed past a 0.0000 here.
* **Gate C — the headline A/B.** Fine-tuned-from-COCO vs from-scratch-head, **same
  schedule (12 ep), same seed, same aug setting, full uncapped GT**. Must beat 0.1386.
  Report recall and class-agnostic AP too: the known single-stage gap is
  ranking/classification, not localization (recall was already tied at 0.676 vs 0.677),
  so a win that does not move classification is suspicious.
* **Gate D — the four never-pretrained classes.** Per-class AP for people(1), van(4),
  tricycle(6), awning-tricycle(7) must not regress versus the single-stage baseline. A
  headline gain that comes with a regression on 4 of 10 classes is a different result
  than it looks like, and must be reported as such.

---

## 7. The confound that will ruin this if ignored

**Augmentation is unrun and is the known-mandatory lever.** `planning/visdrone_detector.md`
established by measurement that "train longer" is refuted and therefore *aug is where the
remaining headroom is* — and the aug pack (`FPN_AUG=1`) has still never been A/B'd at
scale.

So COCO-pretraining and augmentation are **competing explanations for the same headroom.**
If stage 2 runs with `FPN_AUG=1` and the baseline did not, a win proves nothing.

**Rule for this session: aug setting must be identical in both arms.** Simplest is
`FPN_AUG=0` in both, matching the 0.1386 baseline exactly. If you want the aug answer too,
that is a 2×2 (pretrain × aug), which on the twin costs ~40 min of fine-tuning total and
is the better experiment.

---

## 8. Gotchas carried in from the thread's own history

* **`FPN_TAG` must be set on every train AND every eval.** The missing-tag trap scored the
  wrong (untagged) checkpoint six times and voided the entire `long30` run. Verify the
  infer log prints `params : …__<tag>_params.bin`.
* **Checkpoints are per-variant and outlive the artifact that trained them.** A stale
  checkpoint will load and train happily against a changed render.
* **The stage-1 → stage-2 handoff is a FILE COPY, not a flag.** Checkpoints are written to
  `.lake/build/<sanitizedName>[_<buildTag>]_params[_e<N>].bin` (`LeanMlir/Train.lean:30`),
  and a run resumes from *its own tag's* file. So warm-starting VisDrone from the COCO
  pretrain is:

  ```bash
  P=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone_
  cp ${P}_cocopre_params.bin ${P}_ftcoco_params.bin     # stage 1 -> stage 2's tag
  FPN_TAG=ftcoco FPN_EPOCHS=12 .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
  ```

  Confirm the exact prefix with `ls .lake/build/*fpn*params*` first — it is derived from
  the spec name, so it moves if the spec is renamed. **Forget the copy and stage 2
  silently trains from scratch**, producing a null result that looks like "pretraining
  didn't help". That is the single most likely way to get a wrong answer from this
  experiment, and it is indistinguishable from a real negative without checking.
* **Checkpoints live under `.lake/build/`**, so `lake clean` destroys every trained
  artifact in this thread. Copy anything precious to `runs/` before cleaning.
* **Score against `data/visdrone448/val.full_gt.bin`, never the training record** (note the
  dir — it sits next to the 448 build, not in `data/visdrone/`; 548 records).
  `MAX_BBOXES=56` truncates
  and silently drops ~35% of VisDrone val GT.
* **`| head` does not stop a trainer** — it detaches and keeps burning GPU.
* **Lake exe-cache:** build the exact exe you are about to run; `lake build A` will not
  rebuild exe B.
* **`preprocess_coco.py` is single-threaded** over ~70k JPEG decode+resize. If prep time
  matters, that loop is the thing to parallelize (`multiprocessing.Pool` over the
  `per_image` list) — but note the records must stay in `per_image` order or the
  bin/sidecar pairing breaks, so gather results by index, never by completion.

---

## 9. Run order

Start the download first; it is ~10 h of pure wall-clock and everything else waits on it.

```bash
# 0a. kick off immediately, in the background — ~10 h of pure wall-clock
./download_coco.sh                      # 19 GB; S3 path-style URL, resumable

# 0b. WHILE IT DOWNLOADS: port the detector to XLA/PJRT (§4b).
#     Common-module split + one lakefile entry with xlaLink. Gate it by
#     reproducing the IREE detector's first N step losses to ~1e-5 from
#     identical fresh init. This is the highest-leverage hour in the session.

# 1. build the transfer set — VisDrone's anchors and index space (§2)   [DONE]
#    NOTE: extract BOTH annotation files. Extracting only instances_val2017.json
#    is an easy mistake and the train build then aborts with
#    "expected data/coco/train2017 and .../instances_train2017.json".
unzip -o data/coco/annotations_trainval2017.zip \
      "annotations/instances_train2017.json" "annotations/instances_val2017.json" \
      -d data/coco/
python3 preprocess_coco.py data/coco data/coco_vd \
    --size 448 --grid 14 --classes vdmap --fpn data/visdrone \
    --train-only --check                        # Gate 0 — 70,082 records, 94 GB
python3 preprocess_coco.py data/coco data/coco_vd \
    --size 448 --grid 14 --classes vdmap --fpn data/visdrone \
    --val-only --check                          # 2,968 records, 4 GB

# 2. stage 1 on the PyTorch twin, 12 epochs               (Gate A)
# 3. Gate B: load the stage-1 ckpt, eval on VisDrone, ZERO fine-tuning
# 4. stage 2, three arms, all 12 epochs / same seed / same aug:
#      A  from-scratch head   (reproduce 0.1386 — this is the control)
#      B  COCO-pretrained
#      C  COCO-pretrained with --fpn-thresh re-tuned toward P3   (§3)
# 5. score all arms on full GT; per-class table                 (Gates C, D)
# 6. port to the Lean trainer ONLY if B or C beats A
```

---

## 10. What would refute the whole idea

Worth writing down in advance so the session cannot rationalize afterwards:

* Gate B returns 0.0000 and it is *not* a plumbing bug ⇒ the representations do not
  transfer at all; stop.
* Arm B ties arm A within noise ⇒ COCO adds nothing on top of an ImageNet-pretrained
  backbone. Given §3's inverted scale distributions, **this is the most likely single
  outcome**, and it is a perfectly good result to report — it is the measurement that
  justifies spending the next session on augmentation instead.
* Arm B wins but Gate D shows the four unpretrained classes regressed ⇒ report as a
  class-imbalance artifact, not a transfer win.
