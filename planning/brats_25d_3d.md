# BraTS: 2.5D first, and let it decide 3D

**Opened 2026-09-09.** Scoped, not started. **2026-09-29: §1, §2, §3, §5 done (committed
`145b911b`); §2 is a tie, §3 passes, §4a/§4b measured; §4c is the plan for the long 3D
session that decides the port.** **2026-10-04: §4d — two 20k-step 3D seeds against two 2D seeds; the §4c
rule says go, on both 3D seeds.** Companions: `planning/archive/unet3d.md` (the 3D scope,
with the XLA compile spike and the DECIDED-2026-07-15 entry: **codegen + FD, no proofs** — Phase 5
struck, not deferred) and `planning/archive/brats_demo.md` §Dimensionality / Workstream D (Gate D:
2.5D must beat 2D at matched budget before 3D starts). ⚠ `brats_demo.md:315,347` say "3D is not
implied" — those lines are inside the retracted appendix (banner at :270). The live position is
the opposite: affordable, unwarranted until Gate D.

## §0 The demo today

* **Data:** MSD Task01 (BraTS-derived), 484 volumes, modalities FLAIR/T1w/T1gd/T2w, MSD labels
  0/1/2/3 (1↔2 permuted vs raw BraTS; `Train.lean:457` has the WT/TC/ET map). Axial slices,
  tumour-bearing only (`--min-tumor-px 1`), **`--stride 2`**, z-scored over brain voxels then u8
  at ±5σ. 411/73 patients → 14,415 / 2,569 slices. Records: `C·S·S` image bytes + `S·S` mask,
  no volume id or z index in `train.bin`/`val.bin`. **Since 2026-09-29:** `--context K` stacks
  the `K` neighbours each side into the channels (`C = 4·(2K+1)`, `DatasetKind.brats224Ctx K`,
  `data/brats224c<2K+1>/`); `--val-full` / `--train-full` write every slice of every volume in
  volume order with a per-volume index (`*_full.bin/.idx/.json`), which is what per-volume
  scoring (`brats-eval`) and the 3D patch sampler read.
* **Nets:** `unetBrats` from scratch (7.85M, 240²) and `ReferenceNets.r34UnetBratsOf skips ctx`
  (24.5M, 224², ImageNet-bootstrapped encoder; one definition since 09-29, shared by the trainer,
  the predictor and the scorer — the predictor used to carry a "MUST match" copy). The stem is
  FRESH `[64, 4·(2·ctx+1), 7, 7]`; the bootstrap skips it by byte range (`stemFloats ic`), so
  `unet-brats-r34 ctx=k` is the 2D trainer with one tensor wider.
* ⚠ **Every published number is IREE-era** (0.740/0.910/0.869/0.856 at 10 ep); the only XLA row
  is 3 epochs (0.730). `content.tex` ~12496–12502 labels the **scratch** arm's number as the
  ResNet-34 arm. XLA is ~12× the ROCm box (217 ms/step, ~201 s/epoch from-scratch 240²);
  `unet-brats-r34` has no recorded ms/step at all. `runs/brats_*.log` are pre-shuffle-fix and void.
* Tier: codegen-only, FD-validated (`grep -i unet LeanMlir/Proofs/` → only `check_jacobians.py`).
  bf16: not available on the generic walk.

## §1 Stage A — close the 2D ledger (½ day, ~2 GPU-h)

Both arms at 10 epochs on XLA via `scripts/sweeps/run_brats_r34_ab.sh 10 data/brats224` (drop the
`IREE_BACKEND` default at :23); record ms/step for `unet-brats-r34`. **Gate:** 0.740 ± 0.01
reproduces. Fix the `content.tex` label while there.

**DONE 2026-09-25** (`runs/2026-09-25-brats-r34-xla/`): r34 0.743 / 0.912 / 0.869 / 0.856,
scratch 0.741 / 0.910 / 0.867 / 0.856, noskip 0.633; the IREE numbers reproduce within 0.003;
252 / 259 ms/step at batch 16 on one 4060 Ti, ~4.5–5.5 min per epoch.

## §2 Stage B — 2.5D, the gate that decides 3D (1–2 days + a regen + ~2 GPU-days)

**Code LANDED 2026-09-29** (items 1–5 below, as written; the regen took 8 min per build); the
A/B (item 6) ran the same day — results in §2a.

Neighbours are NOT recoverable from the records (no z, stride 2, non-uniform filter, and the
loader shuffles) — a regen is required.

1. `scripts/datasets/preprocess_brats.py:184-227` — `--context k`; neighbours from the UNFILTERED volume, kept
   centres still filtered/strided, replicate at volume ends. Regen to `data/brats224c3/`.
2. `ffi/f32_helpers.c:342,354` — `BRATS_CHANNELS` #define → parameter; `F32Array.lean:313` follows.
3. `Types.lean` ~:1084 `DatasetKind.brats224c3`; `Train.lean` ~:483 `brats224c3IO := { bratsIO
   with trainPixels := 12*224*224, channels := 12, labelBytesPerRecord := 224*224 }`; dispatch ~:497.
4. `MainUnetBratsR34.lean:119` `.convBn 12 64 7 2`; from-scratch `.unetDown 12 32`. Bootstrap:
   nothing.
5. `MainBratsPredict.lean` — T1gd backdrop is channel `4k+2`.
6. Run k=1 and the 2D control at **matched epochs, schedule and seed** (the `r34UnetBratsOf` A/B
   discipline). Add k=2 only if k=1 moved.

⚠ Host RAM: the loader holds the split as f32 — 11.6 GB today → **34.7 GB at 12 channels, 57.9
GB at 20**. ⚠ Decide ±1 slice (true adjacency) vs ±stride (matched spacing) deliberately; no doc
ever did. **Gate D:** ≥ +0.02 mIoU or ≥ +0.03 ET → Stage C. Small or zero → a FINDING; write it
into `brats_demo.md` §5 and stop. This is the highest information per GPU-hour on the table.

### §2a Gate D result, 2026-09-29: a tie. ±1 slice (1 mm) is worth +0.001 to +0.004.

`runs/2026-09-29-brats-25d/` (README, logs, per-patient CSVs). ±1 slice at step 1 (true
adjacency; 12 channels, 34.7 GB host), both arms, 10 epochs, matched to the 09-25 2D run in every
other respect:

| pooled peak | mIoU | WT | TC | ET |
|---|---|---|---|---|
| 2D r34 → 2.5D r34 | 0.744 → 0.744 | 0.913 → 0.913 | 0.869 → 0.874 | 0.857 → 0.858 |
| 2D scratch → 2.5D scratch | 0.741 → 0.744 | 0.910 → 0.911 | 0.867 → 0.871 | 0.857 → 0.858 |

Per patient (brats-eval, best checkpoints): r34 0.893/0.821/0.790 → 0.893/0.824/0.788; scratch
0.889/0.819/0.783 → 0.891/0.819/0.790. Curves coincide from epoch 2; epochs-to-target identical.
The 2.5D trainer costs nothing (240 vs 252 ms/step) and stays in the tree as `ctx=k`.

**Reading.** At 1 mm spacing on skull-stripped brain the neighbours are nearly the centre slice
again, so this is the narrow finding "adjacent slices add nothing the slice model cannot infer",
not "through-plane context is useless": a ±4 mm build (`--context 2 --context-step 2`, 20
channels, 58 GB) is untested, and the 3D probe (§4a) is the direct measurement. Gate D as
written is failed; the decision it was meant to make is taken over by §4a's number.

## §3 Stage C — Gate 0 for real (1–2 days, only if D passed)

Extend `planning/archive/conv3d_spike.mlir` from 16³ toys to a 128³×32ch patch at batch 2
through `ffi/libpjrt_ffi.so`; measure ms/step against Stage A. Add the **rank-8 maxPool backward
tile** (`MlirCodegen.lean:7773` widened; 512 MiB per pool at 128³/B2) — the one op NOT in the
spike and the largest memory risk. Gate: per-voxel throughput within ~3–5× of the 2D conv and no
OOM at 11.68 GiB; else re-plan at 96³ or drop.

**MEASURED 2026-09-29 — the gate passes, by a wide margin.** `jax/scripts/unet3d_gate0.py`: the
whole 3D UNet (unetBrats's shape with 3³ kernels and 2³ pools, 23.5M params), forward + backward
+ SGD update on a 4-channel patch under XLA on one 4060 Ti (jax 0.11, the same compiler the
Lean path drives through the PJRT shim), against the 2D twin at the from-scratch trainer's
16 × 240² batch from the same harness:

| net | input | ms/step | Mvox/s | per voxel vs 2D | peak GiB |
|---|---|---|---|---|---|
| 2D UNet (unetBrats shape) | 16 × 240² | 155 | 5.95 | 1× | 2.9 |
| 3D UNet | 128³ × B2 | 897 | 4.68 | 1.3× slower | 8.6 |
| 3D UNet | 128³ × B1 | 452 | 4.64 | 1.3× | — |
| 3D UNet | 96³ × B2 | 378 | 4.67 | 1.3× | — |
| 3D UNet | 64³ × B4 | 221 | 4.75 | 1.3× | — |

Rank-5 convolution runs at 77% of the 2D conv's per-voxel rate — not the 3–5× the gate allowed
for, let alone the 20× that would have sunk it — and nnU-Net's 128³ × B2 patch fits with 6 GiB
to spare. The rank-8 pool-backward tile is inside those numbers (XLA's `reduce_window`
gradient, which is what the Lean codegen's tile lowers to). 3D is affordable; §2 decides
whether it is warranted. The log: `runs/2026-09-29-brats-25d/gate0.log`.

## §4 Stages D–F — the 3D UNet (~4–5 weeks total if every gate passes)

* **D, rank-generic refactor (~1 week):** `convDimNumbers` (~:62) is the ONE definition every conv
  attr block calls — parameterize by spatial rank and ~35 of 37 conv sites follow; `samePad`
  (:54) list-ifies; `imageD : Nat := 1` on `NetSpec`; 7 hardcoded `ic*imageH*imageW` sites;
  58 four-element pattern matches; 6 dispatch walkers grow arms (parameterize `Layer` by rank,
  do not add `conv3d`/`maxPool3d` constructors). **Gate: emitted MLIR byte-identical for every
  existing 2D spec.** Make the `conv2dHasVJP3` / `maxPool2HasVJP3` citations at :4130-4131 /
  :7769 rank-conditional or the 3D MLIR states theorems that do not exist.
* **E, ops one at a time behind FD probes (~2 weeks):** maxPool3d fwd+VJP first (the unknown),
  conv3d fwd/dW/dx (transpose `[1,0,2,3,4]`, `reverse [2,3,4]`), trilinear fwd/VJP
  (`bilinearWeights1D` is already 1-D — a third `dot_general` factor; the 8-corner gather would
  not have been cheap), BN-3D (`[0,2,3,4]`), concat/split, the two loss blocks (re-verify the
  IREE-miscompile workaround at :4886-4898 on XLA at rank 5). Each op: a
  `tests/vjp_oracle/phase3/MainVjpOracle*3d.lean` clone + a `scripts/*_probe_check.py` clone;
  JAX side is `conv_general_dilated` with `NCDHW`. Bar: ≤ 1e-6 vs central FD. ⚠ Whole-net FD is
  NOT usable here — the family carries a ~15% analytic-vs-FD gap (`MainUnetBratsR34.lean:207-210`).
* **F, loader + net (~1 week + GPU):** whole-volume u8 corpus (~17 GB, fits the read-all pattern),
  patch sampler with ~33% foreground oversampling (the 2D deferral of oversampling does not
  transfer), `DatasetKind.brats3d`, `unet3dBrats` + `lean_exe unet3d-brats-train`. Memory: 96³×B2
  (~2.8 GiB activations) or 128³×B1–2 fits the 11.68 GiB arena; ~23M params ×3 with Adam.

### §4a The direct test, 2026-09-29: a JAX 3D UNet on 128³ patches, one hour, ties the 2D anchor

`jax/scripts/unet3d_brats.py` (the reference a Lean 3D UNet would tie to; whole volumes from
`--train-full`, nnU-Net's patch/oversampling/mirroring, Dice + CE, BN with running stats). 4,000
steps × B2 in 63.5 min on one 4060 Ti. Per patient: **WT 0.894 / TC 0.810 / ET 0.790** against the
2D R34 anchor's 0.893 / 0.821 / 0.790 (pooled 0.910 / 0.859 / 0.848 vs 0.912 / 0.869 / 0.855).
Between the step-2,000 and step-4,000 evals every per-patient number rose (+0.012 / +0.027 /
+0.010) and the loss was still falling, so this is a floor on what 3D does here, not its ceiling.

**Standing verdict on Stages D–F.** Not started. Gate 0 says 3D is affordable; Gate D says 1 mm
of context is nothing; the 3D probe says a from-scratch volumetric net reaches the bootstrapped
slice model's per-patient Dice in an hour and was still improving. The experiment that decides
the port is a longer JAX run (~20k steps, ~5 h on one card; a data-parallel loop over four would
be ~1.5 h) with a per-patient eval every 2k steps: past the anchor by more than seed noise → the
port has its reason; a plateau at parity → the slice model is the right model and Stages D–F
stay unbuilt. Either way the codegen work waits for that number.

## §4b The tail is the needle (2026-09-29, from the per-patient CSVs)

The clinical question is not the mean Dice but how many patients get an unusable segmentation.
From `runs/2026-09-29-brats-25d/pervol_*.csv` (73 patients; worst-10% = the 7 lowest):

| model | WT worst-10% · n<0.7 | TC worst-10% · n<0.7 | ET worst-10% · n<0.7 | ET-absent patient |
|---|---|---|---|---|
| 2D R34 anchor | 0.713 · 3 | 0.470 · 14 | 0.341 · 14 | 55 false ET voxels → 0 |
| 2D R34 scratch | 0.696 · 4 | 0.465 · 13 | 0.338 · 17 | 101 → 0 |
| 2.5D R34 | 0.713 · 3 | 0.467 · 11 | 0.320 · 14 | 122 → 0 |
| 3D UNet, 1 h | 0.744 · 1 | 0.457 · 17 | 0.397 · 15 | 0 → 1 |

* ⭐ **The anchor's headline for a clinician is 14 of 73**: one patient in five gets a core or
  enhancing segmentation under Dice 0.7. Invisible in the pooled 0.869.
* ⭐ **The one-hour 3D net moved the tail, not the mean.** Paired per patient it is worse than
  the anchor on the middle of the distribution (TC: worse on 38, better on 21, mean −0.011 — an
  undertrained model) but better on the anchor's ten worst patients on every region: WT
  0.739 → 0.767, TC 0.527 → 0.562, ET 0.415 → 0.528. The only ET-absent validation patient
  gets 55–272 spurious ET voxels from every slice model and none from the 3D net.
* ⚠ **Not yet a finding.** n = 1 run of one hour; the tail counts carry ~3 patients of seed
  noise (the two 2D arms differ by that on ET n<0.7); one ET-absent patient is an anecdote;
  and the 55-voxel false call is also cleared by the standard BraTS post-process (drop ET under
  a few hundred voxels), so on that case 3D buys what a threshold buys.
* The deep tail is shared: volume 8 (a 2,466-voxel core) scores 0.00–0.11 for every model,
  volume 55 (large, atypical) 0.11–0.37 for all. Training longer will not fix those two; they
  need the images and possibly the labels looked at.

**Current belief:** probably yes on the missed-small-core side, unproven on the false-alarm
side. Mechanism: a slice model decides from one plane, so a speck or a sliver on a single slice
has nothing to check itself against; a 3D model sees it has no extent above or below.

## §4c The long 3D session — the experiment that decides Stages D–F

**Status 2026-09-29:** everything above is committed (`145b911b`, not pushed); the session is
scheduled for when the box can be held for 1–2 days. Box: the 2× RX 7900 XTX (ROCm 7.2,
`jax/requirements-rocm-lock.txt`: jax-rocm7 0.11), or the 4× 4060 Ti here with a data-parallel
loop. Throughput there is unmeasured: on paper 2–3× a 4060 Ti per card (24 GB, ~3× the
bandwidth), in practice MIOpen's 3D conv kernels are the unknown, and the only ROCm number this
repo ever recorded for BraTS (the 2D UNet at ~40 min/epoch, IREE era) was 17× slower than XLA.

**Step 0 on the target box (10 min):** `unet3d_gate0.py`. Its ms/step sets steps-per-day and
whether batch 4 (≈16 GiB at 128³) is on; nothing else is planned until it prints.

**Prep, software, before the box-days (~1 day, here, no GPU-hours to speak of):**
1. Tail metrics by default in `brats-eval` and `unet3d_brats.py`'s eval: worst-decile mean,
   n<0.7, n<0.5 per region; the ET-absent count under the convention; and a per-slice ET
   false-alarm rate (slices with 0 ET in the ground truth on which ≥ k ET voxels are predicted,
   over all 11,315 slices — more cases than the one ET-absent patient gives).
2. A post-processing control applied to BOTH sides at eval: `--min-et N` (an ET prediction under
   N voxels in a volume is relabelled to core, the standard BraTS trick), so the tail comparison
   is threshold-matched and 3D is credited only for what a threshold does not do.
3. `--tta` (mirror test-time augmentation, 8 flips averaged) in the 3D eval; the 2D scorer gets
   the h-flip equivalent or is left as is and said so.
4. Checkpoint + eval every 2,000 steps with the tail metrics, resumable (`--init` plus a step
   offset), so a killed run loses at most 20 minutes and the curve reports as it goes.
5. Batch-4 and 2-card `pmap` options (one file, both behind flags).
6. ⭐ A second 2D anchor seed (50 min on one 4060 Ti, `unet-brats-r34 10 r34 tag=s2`) and its
   per-patient CSV — the seed noise on the tail metrics, without which no tail delta can be
   called. Cheap; do it here before the session.

**The run:** ~170k steps at B2 (or ~85k at B4) ≈ one day at 0.5 s/step — nnU-Net's schedule
within 2×; Adam 3e-4, 200-step warmup, cosine to zero; Dice + CE; eval every 2k. A second day,
if available, is a second seed rather than a longer first run: the decision below needs the
noise more than it needs more steps.

**Decision rule (write the answer into §4a either way):** with the same post-processing on both
sides, the port has its reason if the 3D net beats the 2D anchor on the worst decile by more
than the seed noise (≈3 points) on TC or ET, or lowers the per-slice ET false-alarm rate by
more than noise, or clears the mean by > 2 points on TC. A plateau at parity on all three means
the slice model is the right model for this data and Stages D–F stay unbuilt. Expected
(§"projection" on the results page): mean +1 / +3 / +1.5 points (WT / TC / ET), most of the
value in the tail; the floor is parity.

**If go, the port (§4 Stages D–F):** 4–5 weeks, no proofs (the 2026-07-15 decision), gated by
byte-identical 2D MLIR after the rank-generic refactor; the JAX trainer above is what the Lean
train step ties to, per op through FD probes and end to end through per-patient Dice.

## §4d The 20k-step runs, 2026-10-03/04: the decision rule says go, on two seeds a side

Run dir `runs/2026-10-03-brats-3d/` (README has the commands). Prep items 1, 2, 4 and 6 of §4c
landed for it: the tail and the per-slice ET false-alarm rate in both scorers (`brats-eval`,
`unet3d_brats.py`; the CSVs carry `ET_clear_slices`, `ET_fa1`, `ET_fa10`),
`scripts/probes/brats_tail.py --min-et` as the post-processing control applied to every CSV alike,
checkpoint + per-patient eval every 2k steps with `--resume`, and a second 2D anchor seed —
which needed `LEAN_MLIR_SEED` first, because every Lean run's init, shuffle and augmentation seeds
were constants (`tag=s2` alone would have replayed seed 0). Items 3 (TTA) and 5 (B4 / pmap) not
done. The 3D run is §4a's recipe with the cosine over 20k steps (318 min at 911 ms/step); its
second seed (`--seed 1`, `unet3d_20k_s1.sh`) ran the same recipe in 318 min.

At min-ET 200 on every model (73 patients; worst-10% = the 7 lowest):

| model | WT mean · worst-10% | TC mean · worst-10% · n<0.7 | ET mean · worst-10% · n<0.7 | ET false alarms, ≥1 px |
|---|---|---|---|---|
| 2D R34 anchor, seed 0 | 0.893 · 0.713 | 0.821 · 0.470 · 14 | 0.804 · 0.418 · 13 | 0.0249 |
| 2D R34 anchor, seed 1 | 0.891 · 0.699 | 0.818 · 0.453 · 12 | 0.801 · 0.421 · 15 | 0.0242 |
| 3D UNet, 4k steps (§4a) | 0.894 · 0.744 | 0.810 · 0.457 · 17 | 0.790 · 0.397 · 15 | 0.0136 |
| **3D UNet, 20k steps, seed 0** | **0.900 · 0.747** | **0.827 · 0.495 · 13** | **0.807 · 0.464 · 13** | **0.0107** |
| **3D UNet, 20k steps, seed 1** | **0.903 · 0.772** | **0.830 · 0.521 · 13** | **0.805 · 0.440 · 13** | **0.0103** |

The false-alarm rate is over the 8,060 slices with no ET in the ground truth.

* **Seed noise, measured:** the two 2D seeds differ by 0.002–0.003 on the means, 0.014–0.020 on
  the worst-10% means, ~2 patients on n<0.7, and 0.0007 on the false-alarm rate; the two 3D seeds
  by 0.002–0.003 on the means, 0.024–0.026 on the worst-10% means, 0 patients on n<0.7, and
  0.0004 on the false-alarm rate.
* **The rule (§4c), clause by clause, on both 3D seeds:** worst decile beyond seed noise on TC or
  ET — TC +0.025 to +0.068 over the 2D seeds, ET +0.019 to +0.046: **yes on TC** (the 3D seeds'
  worst 0.495 is above the 2D seeds' best 0.470), **marginal on ET** (seed 1's +0.019 is inside the
  3D seeds' own 0.024 spread). Per-slice ET false-alarm rate lower beyond noise — 0.0103–0.0107
  against 0.0242–0.0249, under half on either seed and ~20× either side's spread: **yes**. Mean
  > 2 points on TC — +0.009: **no**. One clause suffices; two hold on both seeds.
* Paired over patients, the two 3D seeds' mean minus the two 2D seeds' mean: WT +0.009 (95 %
  bootstrap CI [+0.003, +0.017]), TC +0.009 [−0.005, +0.024], ET +0.004 [−0.005, +0.013]. The mean
  is a tie or better, never worse; the tail and the false alarms are where the volume pays. (On
  seed 0 alone ET read +0.018; the second seed halves the ET mean gain and widens nothing else.)
* min-ET 200 and 500 give identical tables; on ET it closes most of the 2D seeds' tail gap from
  the raw numbers (0.341 → 0.418), which is why the post-process had to be on both sides.
* The curve (`unet3d_20k_pervol_s*.csv`): under the high learning rate the evals swing (step 2k
  WT 0.657 — under-segmentation, partly a BN running-stat lag; step 10k ET n<0.7 = 19), and they
  settle from 16k on (16k / 18k / 20k within 0.003 on every mean).
* The two 2D-dead patients stay dead on both 3D seeds: volume 8 TC 0.053 / 0.110 (2D 0.114),
  volume 55 TC 0.464 / 0.378 (2D 0.369) — the gain is from the patients just above them.

**Verdict:** the port (Stages D–F) has its reason, on two seeds a side: the ET false-alarm rate
halves on either 3D seed, and the TC tail clears both 2D seeds on either. The ET tail, which carried
the one-seed verdict, is the weaker clause on two. The verdict is in the book (2026-10-05: the BraTS
section's per-patient table and through-plane paragraph); Stages D–F are Brett's call.

## §5 Stage G — sliding-window whole-volume inference (separate line item)


The only thing that makes ANY BraTS number here comparable to the literature
(`brats_demo.md:110-118`: slice-level over tumour-bearing slices). Do it for the **2D** model
first, where it is cheap and immediately makes the 0.910 WT honest.

**DONE 2026-09-29: `lake exe brats-eval`** (`demos/MainBratsEval.lean`). `preprocess_brats.py
--val-full` writes every slice of every validation volume in volume order with a per-volume
index (`val_full.bin/.idx/.json`); the scorer runs the eval forward over all 11,315 slices and
keeps one confusion matrix per patient. No sliding window is needed in-plane: the 224 crop
holds every brain voxel. Three numbers per checkpoint — the trainer's pooled Dice recomputed on
the tumour-bearing slices (agrees with the trainer's own log line to 0.0004, the instrument
check), the pooled Dice over every slice, and the literature's per-volume mean. The best-by-val
2D checkpoints of 09-25:

| checkpoint | pooled, tumour slices | pooled, every slice | per-volume mean ± sd (median) |
|---|---|---|---|
| r34 (bootstrap) | WT 0.912 TC 0.869 ET 0.855 | 0.910 / 0.868 / 0.854 | WT 0.893 ± 0.074 (0.914) · TC 0.821 ± 0.152 (0.867) · ET 0.790 ± 0.185 (0.858) |
| scratch | WT 0.910 TC 0.867 ET 0.855 | 0.907 / 0.866 / 0.854 | WT 0.889 ± 0.077 (0.917) · TC 0.819 ± 0.158 (0.869) · ET 0.783 ± 0.186 (0.851) |

Two things the pooled number hid. The tumour-free slices cost almost nothing — every-slice Dice
is within 0.003 of tumour-slice Dice, so the slice model does not paint tumour where there is
none. The per-volume mean is 2–7 points lower than the pooled number and carries a wide spread:
the mean is pulled down by a tail of small-tumour patients (medians sit at the pooled values),
which is what the literature's protocol is designed to expose and the pooled protocol cannot.
One of the 73 validation patients has no enhancing tumour; the BraTS convention scores it 1 or 0
outright, and it is included above (the present-only ET means are 0.801 / 0.794).
