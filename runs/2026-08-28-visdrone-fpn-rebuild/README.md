# 2026-08-28 — the VisDrone detector is rebuilt from nothing, and augmentation buys the long schedule

## ⭐ The headline: mAP@0.5 0.1386 → 0.1674, and the lever is one flag

| arm | epochs | aug | train loss | mAP@0.5 | recall | ca-AP |
|---|---|---|---|---|---|---|
| baseline on record | 12 | off | — | 0.1386 | 0.676 | 0.376 |
| `ctrl12` | 12 | off | 112.7 | 0.1526 | 0.682 | 0.393 |
| `long50` | 50 | off | 47.9 | **0.1243** | 0.669 | 0.369 |
| **`aug50`** | 50 | **on** | 88.5 | **0.1674** | **0.703** | **0.429** |
| PyTorch twin (same architecture) | 12 | off | — | 0.1532 | 0.677 | 0.400 |

**Training longer without augmentation actively HURTS** — 4× the schedule, less
than half the train loss, and 19% *lower* mAP, with detections rising 437k → 463k
while recall falls. Textbook overfitting on 6,471 images.

**Augmentation inverts that.** Same 50-epoch schedule, same seed, aug the only
difference: −19% becomes +10% over the best short run. One flag turns a
regression into the best number this project has produced. It beats the recorded
baseline by 21% and the architecture-matched PyTorch twin by 9%.

⭐ This vindicates a prediction that was written down and never tested —
`coco_visdrone_two_stage.md` §4: *"the schedule decision is really an aug
decision … if a long schedule is wanted, it has to be bought with real
augmentation first."* Correct.

It also retroactively vindicates the observation the thread VOIDED: the old
`long30` run was reported as plateauing in val mAP around epoch 10 while train
loss kept halving. That was discarded because the shuffle bug contaminated the
era. The plateau was real, and past it the curve turns down.

⚠ 8 of 10 classes improve, and the gains land where the detector was weakest
rather than on the easy class: bus 0.165→0.227, truck 0.095→0.108,
awning-tricycle 0.020→0.030. Car improves 0.573→0.605 but no longer carries the
mean alone. (bicycle and tricycle slip slightly.)

**The recipe is therefore: augmentation, at a MEDIUM schedule.** Not one or the
other, and not simply "longer" — see the curves below, where both arms decline
after their peak and the aug optimum sits near 30 epochs, not 50.

---

# The rebuild (how the arms above became possible)

The detector thread paused 2026-07-24. Everything it produced was later deleted to
free disk: the data, every checkpoint, the ImageNet backbone file, and the PyTorch
twin's environment. The code and the planning docs survived. This session rebuilt
the arm end to end and **beat the number it stopped on**.

| arm | mAP@0.5 | recall | class-agnostic AP |
|---|---|---|---|
| baseline on record (12 ep, no aug) | 0.1386 | 0.676 | 0.376 |
| **`ctrl12` — rebuilt, same recipe** | **0.1526** | **0.682** | **0.393** |
| PyTorch twin (ultralytics, matched) | 0.1532 | 0.677 | 0.400 |
| YOLOv8s, 100 ep, full aug, 640 px | 0.391 | 0.400 | — |

**+10.1% on the baseline at identical schedule, and now 99.6% of the twin** — the
Lean/PyTorch gap was 90.5% and is closed. Scored against all 38,759 uncapped val
GT boxes, never the 56-box-truncated training record.

⚠ **The gain is the backbone, not the recipe.** `jax_r34_imagenet.bin` was deleted
and is only regenerable by a full ImageNet run — but it is exactly the first
85,138,688 bytes of the packed `resnet34in_momdp64_ckpt_xla.bin`. Reconstructed that
way it is the *30-epoch verified* R34, not the older ~72% one behind 0.1386. Epoch-1
loss 354.7 vs the historical 391.1. `ctrl12` exists to quantify exactly this, which
is why it ran at 12 epochs alongside the long arms.

## Throughput — the number the edge demo needs

548 images in **8.34 s / 8.36 s** over two runs on one RTX 4060 Ti = **65 fps**,
*including* process start, PJRT init, graph load, a 625 MB read and a 406 MB write.
Pure forward is faster. Projections to Jetson by fp32 ratio (arithmetic, not
measured): AGX Orin ~16 fps, ~30 fps at bf16 (R34's bf16 speedup is measured at
1.92x); Orin Nano ~4 / ~7 fps, i.e. the Nano wants MnV4 instead of R34.

## The rebuild reproduces the pipeline exactly

| quantity | historical | rebuilt |
|---|---|---|
| train GT boxes | 343,204 | 343,204 |
| val GT boxes | 38,759 | 38,759 |
| val boxes/img | 70.7 | 70.7 |
| encodability | 88.2% | 88.2% |

⚠ Do **not** regenerate the anchor priors by k-means. Write
`data/visdrone/anchors_fpn_{p3,p4,p5}.txt` from the values hardcoded in
`demos/MainYolov1VisdroneFpn.lean`, so encoder and model cannot disagree.

## Augmentation finally has a matched A/B — and it helps

`FPN_AUG` was committed 2026-07-24 and never run at scale. Same 50-epoch schedule,
aug the only difference:

| at epoch 5 | mAP@0.5 | class-agnostic AP | recall | train loss |
|---|---|---|---|---|
| `long50` (no aug) | 0.1114 | 0.304 | 0.654 | 186.8 |
| `aug50` (aug) | **0.1207** | **0.333** | **0.659** | 194.0 |

Ahead on all three val metrics *while carrying higher train loss* — the signature of
augmentation working rather than noise. Early (epoch 5 of 50), but three metrics move
together with the expected train/val inversion. It is also confirmed simply *active*:
epoch time 305 s vs 208 s for the host-side jitter and flip.

## Two codegen limits, found by putting the head on an R50

- **`.fpnDetect` could only sit on a `residualBlock` backbone.** It taps `fpnStages`,
  which only that layer populated; on a bottleneck backbone the list was empty, the
  head emitted a comment instead of a graph, and the backward then referenced
  `%d_logits`, a classifier gradient that does not exist under a detector. Fixed —
  bottleneck stages now register as taps in both walks. Additive; nothing else reads
  that list.
- ⚠ **`.maxPool` with size > stride has no correct backward.** The tile-compare-select
  routes gradient to the argmax and is valid only for non-overlapping windows; with
  overlap one input can be the max of several and its gradient is a sum the tiling
  never forms. R50's `.maxPool 3 2` *also* fails to parse first (forward pads 224→225,
  backward reads the unpadded `inShape`), so fixing that type error alone would trade a
  loud failure for a silent one. Use `.maxPool 2 2` — pooling is parameter-free, so a
  pretrained prefix still aligns and every downstream shape is identical.
- Incidental: `convPadStyle` is declared in `Types.lean` and **read nowhere** in
  `MlirCodegen.lean`. Setting it on a spec that trains through the generic walk is a
  no-op.

## ⛔ R50-A3's checkpoint cannot be bootstrapped raw

Its conv weights are perfect — std 0.17, every segment matching `ckpt_e100.state.npz`
— but **every per-channel BN slot runs to 5.2e6**, against R34's γ∈[0,0.60] /
β∈[−0.68,0.77]. Loaded as-is the detector starts at loss **5.8e8** where R34 starts at
717.7, and epoch 1 averages 4.3e7 against 354.7.

Not a codegen bug, and an earlier read of this blamed `.bottleneckBlock` and was wrong.
The control that settled it: the same spec under `FPN_NOBOOTSTRAP=1` (pure He init)
starts at **8,284.8** and reaches ~1,018 by step 100, i.e. trains normally. Those BN
values are presumably meaningful to the render that wrote them — that run did score
77.2% — most likely a scale folded against running statistics kept in the `.state.npz`,
which the generic bootstrap loads as zeros, so nothing cancels.

**Workaround in place:** keep A3's conv weights, reset BN to γ=1/β=0
(`r50_a3convonly_params.bin`). Starts at 8,567.6 and is *ahead of He init* by step 100
(923.0 vs 1,018.1), so the conv transfer is real. Clean fix if full transfer is ever
wanted: re-export A3 through `LEAN_MLIR_PARAMS_OUT`, the path that produced R34's file.

## The figure

`demos/figures/visdrone_fpn.png` — truth over prediction on the four densest val
frames. Built by `scripts/fpn_render.py`, which imports `decode_fpn` from the scorer
so the drawn boxes are by construction the scored boxes. The Pets-era renderer could
not read this head at all (it assumes a 7×7 grid and 1470 outputs).

## Where the difficulty actually is

Per-class AP on `ctrl12`, and the mean is not the story:

| class | AP@0.5 | GT |
|---|---|---|
| car | **0.573** | 14,064 |
| bus | 0.165 | 251 |
| van | 0.166 | 1,975 |
| motor | 0.155 | 4,886 |
| people | 0.133 | 5,125 |
| pedestrian | 0.132 | 8,844 |
| truck | 0.095 | 750 |
| tricycle | 0.072 | 1,045 |
| awning-tricycle | 0.020 | 532 |
| bicycle | **0.015** | 1,287 |

Car alone carries the headline. A 38x spread between best and worst class says the
demo's honest subject is *why aerial detection collapses on small rare classes*,
which is more instructive than one mediocre mAP.

## The plateau curves — ⚠ BOTH arms peak and then decline

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 (final) |
|---|---|---|---|---|---|---|---|---|---|---|
| `long50` no aug | 0.1114 | 0.1223 | **0.1320** | 0.1313 | 0.1315 | 0.1283 | 0.1261 | 0.1268 | 0.1244 | 0.1243 |
| `aug50` +aug | 0.1207 | — | — | 0.1703 | — | **0.1731** | — | 0.1694 | — | 0.1674 |

⚠ **Augmentation delays and softens overfitting; it does not eliminate it.**
Without aug the peak is ~e15–25 at 0.132. With aug the peak is **e30 at 0.1731**,
and by e50 it has given back 3%. So the correct reading is *not* "longer + aug
wins" — it is **aug moves the optimum from ~15 epochs to ~30 and raises it 31%**.
Running to 50 overshoots in both arms.

(Intermediate checkpoints are mid-schedule and therefore NOT annealed, which makes
0.1731 an underestimate of what a properly-annealed 30-epoch run should reach.)

Note the aug arm was already ahead at epoch 5 (0.1207 vs 0.1114) while carrying
higher train loss — the correct early signature, visible 45 epochs before the
finals confirmed it.

⚠ `ctrl12` and `long50` have different cosine schedules, so `long50` at its epoch
12 is NOT a control for `ctrl12`. Only the finals are comparable.

## What to run next

1. ⭐ **A 30-epoch arm with aug, annealed at 30.** The best checkpoint seen is
   `aug50` @ e30 = **0.1731**, but that is a truncated 50-epoch cosine that never
   got its annealing. A schedule that actually ends at 30 should beat it, and
   costs 40% less than the 50-epoch run.
2. **Augmentation at 12 epochs.** The 2×2 is missing its fourth cell: aug helps at
   50, but nobody has checked whether it helps at 12, which is 4× cheaper.
3. **Do NOT run longer than 50.** Both arms are declining by then.
3. Mosaic, which the current pack deliberately omits — deferred on the grounds
   that 4-into-1 halves already-tiny objects. Worth re-testing now that plain aug
   is measured as clearly positive.
