# brats_demo.md — brain-tumour segmentation, the seg demo's next rung

Goal: take the segmentation stack that `unet_demo.md` / `unet_demo_v2.md`
built for Oxford-IIIT Pets trimaps and push it to a task where the hard
part is real. Pets was a good scaffold and a weak result: mIoU 0.344 at 3
epochs with **boundary-class IoU 0.000** (`RESULTS.md:90`) — a class
collapse we logged and shrugged at, because a trimap boundary is not what
that demo was about.

BraTS is the rung where that same collapse becomes the whole point.
Enhancing tumour is ~1% of pixels, and it is the sub-region clinicians
actually care about. The failure pets tolerated, BraTS cannot — which is
precisely why medical segmentation invented Dice. That is the through-line
of this demo: **the thin-class collapse, taken seriously.**

Prerequisite reading: `planning/unet_demo_v2.md` (what the seg stack is and
what's still open on the pets side). Volumetric follow-on:
`planning/unet3d.md`.

**Audience, decided 2026-07-15: a working demo a medical student can pick up
and run.** Not a proofs chapter. This is a real constraint with teeth, and it
already explains several choices below — MSD Task01 over the Synapse-gated
BraTS 2021, a dependency-free NIfTI reader instead of a `pip install nibabel`
that PEP 668 blocks, a downloader that parallelizes 3 h down to 12 min. The
correctness story is FD validation (`scripts/seg_loss_probe_check.py`:
gradient vs central finite differences, 1.3e-07), which is a claim a reader
can check by running one script. Priorities follow from this: **reproducibility
and legibility outrank verification depth.** The two things a newcomer needs
that don't exist yet are a per-class **Dice** score (mIoU is not the metric
this field reports) and a **visualizer** — see Workstreams F and G.

## Two decisions, made up front

### Data: MSD Task01_BrainTumour, not the BraTS 2021 tarball

BraTS 2021 proper (1,251 cases, Synapse `syn25829067`) is gated behind a
free account **and a signed research-use agreement**. No script can fetch
it. Every other dataset in this repo is one `./download_*.sh` away, and a
demo whose first step is "go register on Synapse" is a demo readers don't
run.

Medical Segmentation Decathlon **Task01_BrainTumour** is the same task from
open storage: built from the BraTS 2016/2017 cases, 484 labelled 4D
volumes, same four modalities, same glioma sub-regions, 7.6 GB over
unauthenticated S3 (verified `curl -LO`-able 2026-07-15). It preserves the
`download_pets.sh` pattern exactly.

The cost of this choice, stated plainly: 484 volumes instead of 1,251, and
numbers that are **not** comparable to a BraTS 2021 leaderboard. We are not
entering a challenge; we want a real segmentation result on real medical
data that any reader can reproduce. If a 2021 comparison ever matters,
`preprocess_brats.py` reads BraTS-2021-layout NIfTI too — only the label
remap differs (2021 uses 0/1/2/4; MSD pre-remaps to 0/1/2/3).

Citation: Antonelli et al., *The Medical Segmentation Decathlon*, Nature
Communications 13, 4128 (2022) — plus the underlying BraTS papers (Menze et
al. 2015, Bakas et al. 2017).

### Dimensionality: 2D axial slices first — but 3D is far cheaper than it looks

2D slices reuse the entire existing stack and cost one integer: `.unetDown
3 32` → `.unetDown 4 32`. Per the repo's demo-first protocol we take the
runnable demo and its A/B signal first, and let the numbers argue for 3D
rather than assuming it. 2D slice-wise BraTS is a legitimate published
baseline, not a toy — it just leaves through-plane context on the table.

**Scoped properly on 2026-07-15, and the earlier "research-scale lift"
framing in this doc was wrong by ~10×.** Recording the real numbers,
because they change what 3D is worth:

- **IREE lowers rank-5 fine — MEASURED, not assumed.** A hand-written
  NCDHW spike (`conv3d` fwd, the transpose+reverse+convolve `dx`,
  `maxPool3d` via `reduce_window`, bias-grad `reduce` across `[0,2,3,4]`,
  and a trilinear `dot_general` factor) compiles clean through
  `iree-compile --iree-rocm-target=gfx1100`, exit 0, all five functions.
  Kept at `planning/conv3d_spike.mlir` so the claim is re-runnable:
  `.venv/bin/iree-compile planning/conv3d_spike.mlir
  --iree-hal-target-backends=rocm --iree-rocm-target=gfx1100
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false -o /dev/null`
  This was the one genuine unknown — the repo has hit IREE op gaps before
  (`MlirCodegen.lean:6182` routes around `select_and_scatter` because IREE
  lacks it). It is not a gap here.
- **The codegen surface is ~675 lines across 10 functions, not 8.5k.** The
  rank-4 locking is *incidental*, not structural: shapes are plain
  `List Nat` (there is no `Shape` type — rank is not in any codegen type),
  `tensorTy : List Nat → String` (`MlirCodegen.lean:44`) is already
  rank-generic, and **rank-6 tensors already ship** — the maxPool backward
  emits `[b, c, oH, stride, oW, stride]` (`MlirCodegen.lean:6191-6196`).
  Of 37 `stablehlo.convolution` sites, **only 2** emit
  `input_spatial_dimensions` themselves; the rest route through
  `convDimNumbers` (`:62-70`). Of 111 rank-4 list patterns, only ~51 are on
  a 3D UNet's path; the rest are ViT/Mamba/Swin/MobileNet/EffNet/ConvNeXt/
  YOLO/DDPM code a 3D UNet never touches.
- **`bilinearWeights1D` (`:846`) is strictly 1-D** — trilinear is that
  helper called three times plus a third `dot_general` factor. The
  factorized `Wy·X·Wxᵀ` structure *helps* here; a naive 8-corner 3D gather
  would have been the hard path and the demo didn't take it.
- **Layer/NetSpec extend without breaking anything**: every field is
  defaulted (`Types.lean:290-291`), so `imageD : Nat := 1` and a `rank`
  arg on `convBn`/`maxPool`/`unetDown`/`unetUp` migrate zero existing specs.
- **The streaming-loader objection was overstated.** The 550 GB figure
  assumes loading whole volumes as f32. 3D UNets train on *patches*
  (nnU-Net-style 128³ crops), and all 484 volumes held as uint8 are ~17 GB —
  which fits the existing read-everything-into-a-ByteArray pattern on a
  188 GB box. Patch sampling is a loader change, not a rearchitecture.

**The real cost is the proof layer, and it is optional by precedent.**
`Tensor3 := Fin c → Fin h → Fin w → ℝ` (`Proofs/Tensor.lean:1490`) puts
rank *in the type*, so `conv2d_has_vjp3` (~490 lines) and
`conv2d_weight_grad_has_vjp` (~494 lines) plus the whole 2D windowing
algebra (`winRow`/`winCol`/... `Proofs/CNN.lean:1857-1930`) do not
instantiate to 3D — a 3D twin is ~2000 new lines of Lean. **But
`bilinearUpsample`, `unetDown`, and `unetUp` have no Lean proofs either**
(`grep -rln "unet" LeanMlir/Proofs/` finds only `check_jacobians.py`).
The entire UNet decoder already ships codegen-only + FD-validated. conv3d
can follow that exact precedent.

So the honest estimate: **~3-5 weeks codegen-only + FD** (the way the
decoder shipped), or multi-month if it must be certified. That is an
editorial decision about what the demo claims, not a technical wall.

**Cheapest real win stays 2.5D** — stack 2k+1 adjacent slices × 4
modalities into the channel dim (5 slices → `ic=20`). Zero codegen,
loader-only, recovers much of the through-plane context. It is also the
number full 3D has to beat, and worth having *before* spending 3-5 weeks.

## What's built (2026-07-15)

The data path and the demo shell, mirroring the pets stack one-for-one:

| Piece | File | Notes |
|---|---|---|
| Downloader | `download_brats.sh` | MSD tar → extract → preprocess; resumable |
| Preprocessor | `preprocess_brats.py` | NIfTI → flat binary; **dependency-free NIfTI-1 reader** (numpy only, no nibabel — PEP 668 makes `pip install` a reader-hostile first step). Cross-validated against nibabel 5.4.2 on 9 cases incl. column-major, big-endian, `scl_slope`/`scl_inter`. |
| C loader | `lean_f32_load_brats` (`ffi/f32_helpers.c`) | 4-channel, size-parameterized, z-score dequant. **Not** the pets loader — that one hardcodes 224 / 3ch / ImageNet RGB mean-std, none of which is meaningful for MRI. |
| Lean decl | `F32.loadBrats` (`LeanMlir/F32Array.lean`) | |
| Dataset | `DatasetKind.brats` + `bratsIO` (`Types.lean`, `Train.lean`) | `labelBytesPerRecord := 240*240` auto-selects `.perPixelCE` |
| Spec + exe | `demos/MainUnetBratsTrain.lean`, `lakefile.lean` | `unetBrats`, `lake exe unet-brats-train` |
| Region Dice | `DatasetIO.segRegions` + eval block (`Train.lean`) | WT/TC/ET from the existing confusion matrix; host-only, `eval forward (cached)` proves zero codegen impact. Checked by `scripts/seg_region_dice_check.py` |
| Visualizer | `demos/MainBratsPredict.lean` | `lake exe brats-predict [out.ppm] [params.bin] [bn_stats.bin]`; T1gd backdrop + GT/pred overlays, tumour-burden slice picking |
| Weighted CE | `LossKind.perPixelWeightedCE` (`Types.lean`), `weights` param on `emitPerPixelCEBlock` | `unet-brats-train … wce`, now the default. FD-verified 9.41e-08 + `wce1 ≡ ce` exactly; weights from `scripts/brats_class_weights.py` |
| Focal CE | `LossKind.perPixelFocalCE`, `emitSegFocalBlock` (`MlirCodegen.lean`) | `unet-brats-train … focal`. True (non-detached) gradient; FD 1.64e-07 + `focal0 ≡ ce` exactly |
| Gradient scorecard | `scripts/seg_grad_scorecard.py` | All 5 arms vs the emitted MLIR; the (C) ratio predicts which loss can work, for ~2 min of CPU instead of 40 h of GPU |
| Prior-bias init | `TrainConfig.headPriorBias`, `NetSpec.applyHeadPriorBias` | `unet-brats-train … pb`; head bias = log π_c, verified `softmax(bias) == π` to 1.9e-09 |
| Per-arm artifacts | `NetSpec.buildTag` / `withBuildTag` (`Types.lean`, `Train.lean`) | Each arm gets its own `_params.bin` / `_train_step.vmfb`, so the ablation runs **concurrently, one arm per GPU** instead of clobbering itself |

Everything else — `unetDown`/`unetUp` skip codegen, bilinear upsample
(fwd+bwd, FD ~1e-11), channel concat, per-pixel softmax-CE, the seg train
ABI, the mIoU + per-class-IoU confusion harness — is reused **unchanged**
from the pets demo. That reuse is itself a result worth stating in the
writeup: the seg stack generalized to a new modality count, a new class
count, and a new resolution without touching the codegen.

### Two preprocessing decisions that would silently ruin the numbers

Both are the kind of thing that produces a plausible-looking training curve
and a meaningless val score:

1. **Normalize over brain voxels only.** BraTS volumes are skull-stripped,
   so most of each volume is exact zero. Z-scoring over the whole volume
   computes the statistics of mostly-background and crushes tissue contrast
   into a narrow band. `znorm_brain` takes mean/std over nonzero voxels,
   per modality, per volume.
2. **Split by patient, not by slice.** Adjacent axial slices of one brain
   are near-duplicates. A random slice-level split puts near-copies of the
   same patient on both sides and reports a val score that is partly
   memorization. `preprocess_brats.py` splits the 484 volumes, then slices;
   the manifest lands in `data/brats/split.json`.

Storage: uint8 on disk, z-score quantized over ±5σ (step ~0.039σ), inverted
in the C loader. Matches the repo's on-disk convention (pets/imagenette/
cifar all store uint8 and normalize in C) and keeps train.bin 4× smaller.
The step is far below the contrasts that define tumour boundaries. Noted
because it *is* a real (if small) information loss the RGB datasets don't
have — their sources are 8-bit to begin with.

## Workstream A — the baseline run (do first)

`unet-brats-train data/brats <epochs>`, per-pixel CE, no augmentation. The
existing mIoU harness prints per-class IoU every 10 epochs and at the end.

**Gate A: a number exists, per class.** Any number. This is the baseline
row and — more importantly — the measurement that decides Workstream B.

### Status 2026-07-15: 1-epoch smoke DONE, prediction confirmed (at this budget)

`runs/brats_unet_ce_smoke_gpu0.log`, gfx1100, batch 16, lr 1e-3, 900 steps,
40.2 min/epoch:

```
Epoch 1/1: loss=0.132800
val mIoU: 0.243445  (c0=0.973529  c1=0.000000  c2=0.000250  c3=0.000000)
```

**All three tumour classes collapsed, not just the enhancing one.** The
prediction below said class 3 would go; in fact 1, 2 and 3 all did. mIoU
0.243 versus 0.250 for the trivial predict-background-everywhere model —
i.e. the net is, to three decimals, the trivial predictor.

The loss trace is the informative part, and it argues this is **not purely
an underfitting artifact**: loss reaches ~0.12 by step 100/900 and then
oscillates (0.088 / 0.145 / 0.142 / 0.032 / 0.110 / 0.158) without
descending for the remaining 800 steps. It found the trivial minimum inside
the first ninth of one epoch and stayed. For reference the entropy of the
class prior — the CE of the best *constant* predictor — is 0.1417, and the
epoch mean was 0.1328: barely below the constant-predictor floor.

Honest limits of this datapoint: it is one epoch, no augmentation, constant
LR. It does **not** establish that CE cannot eventually learn the tumour
classes (it can, slowly — that is the well-known behaviour Dice exists to
fix). It establishes that (a) the whole pipeline works end-to-end on real
data, and (b) at a smoke budget CE lands exactly on the degenerate solution.
The real budget run is still owed before the CE row is published.

What this DOES already validate, and it is the reuse claim: the entire seg
stack ran on a new modality count, class count, and resolution with **zero
codegen changes**.

The prediction on record, so the result can falsify it: **per-pixel CE will
collapse the enhancing-tumour class**, the way it collapsed the pets
boundary class, only worse, because the imbalance is worse. If mean IoU
looks respectable while class 3 IoU is ~0, that is not a disappointment;
that is the demo's thesis reproducing on cue. Report per-class IoU always —
mean-of-4 alone would hide exactly the failure we're here to show.

If CE *doesn't* collapse class 3, that's the more interesting outcome and
Workstream B gets re-scoped rather than assumed.

### Result, 1-epoch smoke (2026-07-15, `runs/brats_unet_ce_smoke_gpu0.log`)

```
Epoch 1/1: loss=0.132800    (900 steps, batch 16, lr 1e-3, ~40 min on gfx1100)
val mIoU: 0.243445  (c0=0.973529  c1=0.000000  c2=0.000250  c3=0.000000)
```

**The net converged to the all-background predictor, to four decimals.** A
model that predicts background on every pixel scores, on this val split's
measured balance, c0 IoU 0.973670 and mIoU 0.243418. Observed: 0.973529 and
0.243445 — Δ mIoU = 0.000027. All three tumour classes are gone.

**The instructive part — a trap for whoever reads the loss curve.** Final CE
0.1328 is *below* the 0.1417 CE of the best constant (class-prior)
predictor, which looks like evidence the net learned something real. It did
not. BraTS is skull-stripped, so ~83% of every volume is exactly zero
(measured: brain is 17.0% of voxels). The net drives CE under the prior
floor purely by being very confident on trivially-separable background,
while its argmax never fires a tumour class. **CE below the prior floor is
not evidence against collapse.** This is precisely why Workstream A (the
per-class IoU harness) had to precede any loss work — no scalar in the
training log could have told us this.

**Caveat, stated so nobody over-reads it: one epoch is not a budget.** The
pets demo's 3-epoch collapse was read as a budget artifact
(`unet_demo_v2.md`) and that reading was defensible. Gate A is met in the
sense that a per-class number exists; it is *not* yet a verdict on CE. The
verdict needs the matched-budget run below. What the exactness of the
trivial-predictor match does establish is that after 900 steps the CE
gradient has produced no pressure whatsoever toward the thin classes.

**Next**: run CE and Dice at matched budget (30-60 ep), one per GPU, and let
the ablation decide. Do not publish the 1-epoch row as the CE verdict.

## Workstream B — Dice loss (~~the lever this demo exists to pull~~ — it isn't; see the Gate B result)

`unet_demo_v2.md` scoped Dice out as "medical-imaging-grade complexity, and
that judgment stands." **That judgment inverts here** — this *is* medical
imaging, and Dice is table stakes rather than a refinement.

### Status 2026-07-15: codegen DONE + FD-verified

`LossKind.perPixelDice` / `.perPixelDiceCE` (`Types.lean`), `SegLoss` selector
+ `LossKind.isSeg`/`.segLoss` helpers, `emitSegDiceBlock` + `emitSegLossBlock`
(`MlirCodegen.lean`), wired through `generateTrainStep` → `compileVmfbs`.
Selectable per-run: `unet-brats-train data/brats <epochs> [ce|dice|dicece]`,
default `dicece`.

**Why the gradient needed a real check.** CE's backward seed is
`(softmax - onehot)/N` — the softmax Jacobian cancels against the log, which
is why it's one line and verifiable by eye. Dice gets no cancellation, so it
carries the Jacobian-vector product explicitly:

```
g_{c,k} = ∂L/∂p_{c,k} = (1/NC)·(B_c - A_c·y_{c,k}),  A_c = 2/den_c, B_c = num_c/den_c²
dz_i    = p_i·(g_i - Σ_j g_j·p_j)
```

All elementwise ops + one channel reduce — no new primitive.

**FD verification** (`scripts/seg_loss_probe_check.py` + `seg-loss-probe` exe,
mirroring the flash-attn probe pattern; CPU, no GPU needed):

| loss | fwd vs numpy | grad vs central FD |
|---|---|---|
| `ce` | 2.8e-09 | 8.5e-08 |
| `dice` | 2.2e-08 | **1.3e-07** |
| `dicece` | 3.5e-08 | 1.0e-07 |
| `dice` / `dicece`, class absent from batch | ≤1.7e-07 | gradient finite |

That last row is why ε exists: a class absent from the batch gives `0/0`
without it. On BraTS this is not a corner case — enhancing tumour is missing
from many slices, so it fires constantly.

**Zero regression on the pets path**: with `segLoss = .ce` the emitted train
step is byte-identical to before the refactor (verified by diffing the
generated MLIR — the only delta is the `(SegLoss.ce)` provenance comment in
the header).

Design notes worth keeping: **batch** Dice (reduce over B,H,W) not per-sample —
with 0.5% classes and batch 16 a sample often contains none of a class, which
makes per-sample Dice degenerate for exactly the classes we care about.
`.diceCE` one-hots raw for its Dice half and applies label smoothing only to
its CE half (smoothing a set-overlap ratio is meaningless); `.perPixelDice`
+ smoothing is rejected rather than silently ignored.

- New `LossKind.dice` (and likely `.diceCE`, the standard sum — Dice alone
  can be unstable early, and the CE term regularizes it).
- Emitter alongside `emitPerPixelCEBlock` (`MlirCodegen.lean:4303`), which
  is the structural model: same `[B, NC, H, W]` softmax, different
  reduction. Soft Dice over the softmax probabilities:
  `1 - (2·Σ p·y + ε) / (Σ p + Σ y + ε)`, per class, meaned. Its gradient is
  a quotient rule over two spatial sums — no new primitive, just a new
  reduction pattern.
- FD-verify like every other loss here.
- Ablation: CE vs Dice vs Dice+CE at matched budget, scored by per-class
  IoU. **That table is the demo's money slide** — the same shape as the
  pets skip-ablation, but about the loss.

**Gate B: Dice rescues class 3** (enhancing tumour IoU decisively off the
floor) at matched epochs. If it doesn't, the collapse is not the loss and
we go looking — most likely at the class-weighted CE variant
(`unet_demo_v2.md` Workstream E scopes it at ~half a session) as the cheaper
counterfactual.

### GATE B RESULT 2026-07-15: FAILED at 1 epoch — and the mechanism is measured

Matched 1-epoch runs, identical in every respect but the loss:

| loss | val mIoU | c0 bg | c1 edema | c2 non-enh | c3 enhancing |
|---|---|---|---|---|---|
| `ce`     | 0.243445 | 0.9735 | 0.000000 | 0.000250 | 0.000000 |
| `dicece` | 0.243402 | 0.9736 | 0.000000 | 0.000000 | 0.000000 |

**Dice rescued nothing** — identical to four decimals, both the trivial
background-only predictor. diceCE's epoch loss (0.871783) is the
all-background value (Dice 0.75 + CE ~0.12 ≈ 0.87) to three decimals.

**Why — measured, not guessed** (`scripts/seg_dice_vanishing_grad_probe.py`,
run against the emitted MLIR). Dice's gradient carries a `p_i` factor from the
softmax Jacobian; CE's does not:

```
dice: dz_i = p_i·(g_i - Σ_j g_j·p_j)      ← ∝ p_i
ce:   dz_i = (p_i - y_i)/N                ← = -1/N at p_i = 0
```

Sweeping class 3's logit down and watching the gradient that is supposed to
rescue it (mean |dz₃| at true-class-3 pixels):

| logit₃ | p₃ | dice \|dz₃\| | ce \|dz₃\| | dice/ce |
|---|---|---|---|---|
| 0 | 1.78e-01 | 1.929e-03 | 6.423e-03 | 0.300 |
| -4 | 5.84e-03 | 4.535e-04 | 7.767e-03 | 0.058 |
| -8 | 1.03e-04 | 9.306e-06 | 7.812e-03 | 0.0012 |
| -10 | 2.09e-05 | 1.886e-06 | 7.812e-03 | 0.00024 |

From -8 to -10, p₃ falls 4.9× and the Dice gradient falls 4.9× — **exactly
linear**, confirming the `p_i` factor. CE's gradient is **flat at 7.8e-03
throughout**, wholly indifferent to the collapse.

**Two conclusions, and they reframe this workstream:**

1. **Dice is not a rescue mechanism.** Its signal is weakest exactly where
   it is needed — at p₃ = 2e-5 it is 0.02% of CE's. It cannot un-collapse a
   class. It can only help if the collapse never happens.
2. **In `.diceCE`, Dice is the junior partner from step 0.** Even at init
   (p₃ ≈ 0.25, logit 0) its gradient is only 0.30× CE's, and it decays from
   there. `.diceCE` therefore *is* CE with a rounding error attached — which
   is precisely what the matched run showed.

So the header framing of this workstream ("the lever this demo exists to
pull") was **wrong**. The collapse happens in the first ~100 steps
(`runs/brats_unet_ce_smoke_gpu0.log`: loss reaches ~0.12 by step 100 and
never descends again), long before Dice has any say. The lever has to act
*there*.

**Revised plan — prevent the collapse, don't try to reverse it:**

- **Class-weighted CE** (`unet_demo_v2.md` WS-E, ~half a session): a weight
  vector on the per-pixel loss and gradient seed. The mechanism above says
  this is the *right* tool — CE's gradient provably does not vanish at
  p→0, so up-weighting rare classes there keeps a live signal all the way
  down. Cheapest and most likely to work; do it first.
- **Prior-bias init on the head** (RetinaNet's trick): initialize the final
  conv's bias to `log(π_c)` so class c starts at its prior rather than at
  uniform. Directly targets the first-100-steps window where this is decided.
  Nearly free.
- **Foreground oversampling** in the sampler (nnU-Net forces ~33% of patches
  to contain tumour). Raises Y_c per batch; complements the above.
- **Then** re-test Dice on top. Dice may well be a good *polish* on a model
  that already predicts the class at all — that is a different claim from
  "Dice fixes imbalance", and it is the one the evidence supports.
- Also still worth running once for the record: **pure `.dice`** (no CE to
  drive the collapse). The mechanism predicts it collapses too, just slower.
  A cheap falsification of the story above.

**None of this impugns the codegen** — the Dice emitter is FD-verified to
1.3e-07 and does exactly what soft Dice is defined to do. The loss is
correct; the *expectation* of what it would buy was wrong.

**~~Free win to fold in~~ — DONE 2026-07-15, and it was not free.** This doc
said label smoothing on `perPixelCE` was already in the emitter
(`emitPerPixelCEBlock`'s smoothOn/smoothOff) and that "only the guard needs
lifting". The emitter part was true. The conclusion was wrong: **nothing in
`seg_loss_probe_check.py` ever ran at `ls > 0`** — the numpy side modelled
smoothing, but `build_and_run` never passed `ls=` to the probe, so no FD check
ever touched the path. Lifting the guard on that basis would have shipped
unverified codegen on the strength of the code *existing*.

Verified first, then lifted. FD at ls=0.1: `ce` **1.07e-07**, `wce`
**1.31e-07**, `dicece` **1.25e-07** — the emitter was right all along, which is
the good outcome and not one we were entitled to assume. Probe check count
14 → 17. Guards lifted for `.perPixelCE` and `.perPixelWeightedCE`; still
rejected for `.perPixelDice` (smoothing a set-overlap ratio is meaningless) and
`.perPixelFocalCE` (`p_t` is the probability of *the* true class, which a
softened target does not name).

For `wce` the composition is the one you want and it is not an accident: the
weight rides on `%seg_mask`, the raw EQ comparison, so it stays a weight on the
**true** class while the *target* softens.

**The general lesson, worth more than the feature:** "the codegen is already
there" is not the same as "the codegen is verified". This repo's whole claim on
the seg path is FD, and an emitted-but-unexercised branch has exactly the status
of code nobody has run.

## WAVE 2 RESULT 2026-07-16: wcesqrt works; focal collapses at every γ; the loss-floor axis is REFUTED

Five arms now run, and they force a correction to this doc's own theory. The
first-epoch per-class result (`evalEveryNEpochs := 1`, so this is visible in 40
min, not 7 h):

| arm | weights / γ | c3 enh IoU | WT Dice | % of brain | outcome |
|---|---|---|---|---|---|
| `ce` | uniform | 0.000000 | 0.0000 | 0.00% | collapse |
| `focal g=2` | γ=2 | 0.000000 | 0.0000 | 0.00% | collapse |
| `focal g=8` | γ=8 | 0.000000 | 0.0000 | 0.00% | collapse |
| `wcesqrt` | π^-0.5 | **0.143432** | **0.3983** | 0.94% | **finds the tumour** |
| `wce` | π^-1 | 0.017823 (@10ep) | 0.1650 | 28.95% | over-predicts |

### The loss-floor ratio does NOT predict the outcome — I committed that it did, and it's wrong

Last commit (`44a6eb3`) claimed the "cost of the prior vs uniform" ratio called
every arm, target ≈ 1. Then `focal g=8` ran. Its ratio is **1.30×**; `wcesqrt`
is **1.35×** — all but identical — and they land on opposite outcomes (total
collapse vs finds-the-tumour). **The floor ratio is refuted.** It was a
post-hoc fit to four points and the fifth broke it. Recorded here rather than
quietly dropped, because it is the third time this session a tidy theory got
ahead of the measurement (the others: the gradient scorecard ranking focal_pb
best, and a script comment asserting focal's γ moved the wrong way).

### What actually survives all five

The clean split is **absolute amplification of the rare class**:

| | leaves rare-class gradient at CE's magnitude? | outcome |
|---|---|---|
| `ce`, `focal g=2`, `focal g=8` | **yes** — focal only scales the *majority* | collapse, every one |
| `wcesqrt`, `wce` | **no** — the weight multiplies the rare class directly | escapes |

Focal collapses at *every* γ because its whole mechanism is suppressing the
easy majority, and that leaves the rare-class gradient — the quantity that must
actually move the head's weights — exactly where CE has it. CE collapses with
that gradient. Suppressing its rival does not change it. The gradient
scorecard's (C) ratio missed this because a *ratio* has no magnitude: focal
sends (C) sky-high by shrinking the denominator while the numerator, the part
that matters, never moves. **(C) is a diagnostic of one mechanism, not a
ranking, and it should never have been read as one.**

### wcesqrt: the honest claim is narrower than "segments"

It finds the tumour *region* and cannot type the *sub-regions*. Per-epoch:

| ep | mIoU | c1 edema | c2 non-enh | c3 enh | WT Dice | % brain (truth 2.63%) |
|---|---|---|---|---|---|---|
| 1 | 0.2808 | 0.001608 | 0.000000 | 0.143432 | 0.3983 | 0.94% |
| 2 | 0.2750 | 0.000004 | 0.000020 | 0.123750 | 0.5071 | 2.16% |

c3 IoU *fell* while WT Dice *rose* — not a regression: it is learning to cover
the whole lesion and calling all of it enhancing, so its class-3 precision drops
as its region coverage improves. c1 and c2 stay ≈ 0. **This is exactly the
loss-floor ladder's `1.0397` reference predictor — "knows tumour location,
uniform over the classes there" — now observed rather than hypothesized.** So
`wcesqrt` is a binary tumour/background segmenter wearing a 4-class head. That
is still the best result in the demo by a mile (CE: nothing; wce: 29% of brain),
but "it segments the tumour" over-claims and the record says region-not-type.

Whether the sub-regions ever separate is a harder problem — likely capacity/data
bound, not loss-bound (note c2 has the *highest* weight of any tumour class yet
the lowest IoU, so more weight is not the lever). The doc always flagged WT easy
/ TC-ET hard; this is that, quantified.

### The axis, unified: `wceb b=<n>` sweeps β — and it is a CLIFF, not a plateau

The discrete arms are one knob — weights `π_c^(-β)`, β as a percent on the CLI
(β=0 → CE, β=0.5 → wcesqrt exactly, β=1 → wce exactly). `wceb b=70` mapped the
middle, and the transition is sharp (confirmed, 2 stable epochs):

| β | WT recall | WT precision | % of brain (truth 2.63%) | behaviour | stability |
|---|---|---|---|---|---|
| 0.5 `wcesqrt` | 27–46% | 56–76% | 0.9–2.2% | finds tumour | **oscillates** |
| 0.7 `wceb70` | 94–99% | ~11% | 21–23% | over-predicts | stable |
| 1.0 `wce` | 99% | 1.8% | 29% | paints brain | stable |

The jump from "finds it" to "over-predicts" happens between β=0.5 and 0.7 — by
0.7 the model already covers 72% of the way to β=1's full over-prediction. The
usable band is narrow and near β=0.5.

**The stability column is the real finding, and it reframes the whole
workstream.** The over-prediction basin (β≥0.7) is *wide and stable* — both
epochs sit at ~22% of brain. The find-the-tumour basin (β=0.5) is *narrow and
unstable* — `wcesqrt` rotated its predicted class every epoch (enhancing @1-2,
edema @3-4), WT Dice 0.40/0.51/0.045. So the exponent that gives the right
coverage is also the one the optimizer cannot sit still in at a constant 0.001
LR. The good solution is a knife-edge; the bad one is a broad valley. That is
why this is hard, and it is a *training-dynamics* statement, not a loss one.

**Pivotal open experiment: `wcesqrt cos`** (running). Same β=0.5 weights,
cosine LR 0.001→0 — high early to find the knife-edge, low late to stay on it.
If it converges stably to the epoch-2 quality (WT Dice ~0.5), the demo has a
clean winner and the story is "right loss + right schedule". If it still
oscillates or collapses, the honest conclusion is that a loss choice alone does
not solve severe imbalance on this architecture/budget — you need the nnU-Net
machinery (patch sampling, deep supervision, long schedule), and *that* is the
real cost of the thin-class problem the demo set out to show.

## GATE B' RESULT 2026-07-16: matched 10-epoch run — wce did not fix the collapse, it INVERTED it

`runs/brats_ablation_{ce,wce}_10ep_gpu{0,1}.log`. 10 epochs, 9,000 steps, one arm
per GPU, identical in every respect but the loss. **The CE row this doc has owed
since Workstream A now exists.**

| arm | val mIoU | c0 bg | c1 edema | c2 non-enh | c3 enh | WT Dice | TC Dice | ET Dice |
|---|---|---|---|---|---|---|---|---|
| `ce` | 0.243402 | 0.9736 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `wce` | 0.186536 | 0.7283 | 0.000000 | 0.000000 | **0.017823** | **0.165003** | **0.065377** | **0.035021** |

### 1. CE: the collapse is an absorbing state, not a budget artifact — settled

**`ce` predicted ZERO tumour pixels across all 2,569 val slices** (`pred=0` on
every region). Not one voxel of any tumour class, out of 148 million.

And it is **worse than the 1-epoch run**, which at least had `c2 = 0.000250`. Ten
times the budget moved it *further* into the trivial predictor. That kills the
"1 epoch is not a budget" caveat this doc has been carrying since Gate A — the
collapse is not underfitting waiting to resolve, it is an absorbing state that
training deepens. The loss-floor ladder said exactly this and now the run agrees:
under CE the cheapest constant predictor *is* the collapse.

### 2. wce: 99.4% recall, 1.8% precision — the opposite degenerate predictor

The numbers that matter are not in the IoU column, they are in the counts:

```
wce  ET:  inter=763457   gt=768149   pred=42831676
```

- **Recall 99.39%** — it finds essentially every enhancing-tumour voxel there is.
- **Precision 1.78%** — it paints **28.95% of every brain** as enhancing tumour,
  against a truth of 0.52%. A **55.8× over-prediction**.
- `pred_WT == pred_TC == pred_ET == 42831676`, exactly. That identity proves it
  **only ever predicts class 0 or class 3** — it never once says edema or
  non-enhancing, which is why c1 and c2 are still 0.000000.

So `wce` did not un-collapse the net. **It replaced "everything is background"
with "everything is enhancing tumour."** The figure
(`demos/figures/brats_ce_vs_wce.png`) is the cleanest thing this demo has
produced: column 3 is an empty brain, column 4 is a brain painted entirely
yellow, and only the loss differs.

**The mechanism, and it is our own arithmetic.** `w₃/w₀ = 195.6` means a false
*negative* on enhancing tumour costs 196× a false *positive* on background. Under
that exchange rate, a capacity-limited net's best move is to over-predict the
rare class enormously — 56× is not a bug in the weighting, it is the weighting
working exactly as specified. We asked for every class to own 25% of the loss;
we got a net that acts like every class owns 25% of the *brain*.

### 3. mIoU and Dice disagree about which arm is better — Gate F earns its keep

| metric | says | ce | wce |
|---|---|---|---|
| mean IoU | **ce is better** | 0.2434 | 0.1865 |
| WT Dice | **wce is better** | 0.0000 | 0.1650 |

mIoU ranks the arm that predicts *nothing* above the arm that at least finds the
tumour, because 3 of its 4 classes are 0 either way and background IoU dominates.
This is not a curiosity — it is the argument for Workstream F, arriving
unprompted. Had we shipped mIoU alone we would have concluded `ce` was the better
model.

### 4. What called it, and what didn't — honest scoreboard

- **The 64-slice overfit probe was the best predictor.** It showed exactly this
  failure at 9× over-prediction and this doc wrote down the fallback
  (inverse-sqrt) in case it survived to budget. It survived, at 56×. The cheap
  plumbing test earned its keep.
- **The gradient scorecard was right but insufficient.** It said `wce` escapes
  the trivial predictor (constant 196× balance advantage). It did escape. But
  **(C) measures whether the rare class's gradient can compete, not whether the
  answer is useful** — it is necessary, not sufficient, and *higher is not
  better*. A 196× pull is enough to overshoot past the target into the mirror
  failure. The scorecard has no term for overshoot, and should not be read as
  ranking arms.
- **The loss floors were right and also insufficient.** They said the trivial
  predictor is unbounded-bad under wce. True, and it fled. Neither tool models
  the *opposite* degenerate state, because both were built to explain a collapse.
- **My prediction was half right**: I called "all three tumour classes off the
  floor + over-prediction". Only c3 came off the floor, and the over-prediction
  was 6× worse than the probe suggested.

### 5. Next, and it is cheap

**Inverse-sqrt frequency weights `[1.0, 7.80, 14.84, 13.99]`** — already printed
by `scripts/brats_class_weights.py`, already flagged in this doc as the fallback.
They drop the FN:FP exchange rate from **196:1 to 14:1** and tumour's share of
the loss from 75% to ~21%. The prediction to falsify: precision rises off 1.8%
without recall returning to 0.

Also newly interesting: **`focal`**, which never amplifies the rare class and so
has no FN:FP asymmetry to overshoot on — it defunds the majority instead. Its
open question was always timing, and `focal pb` is the arm with a mechanism for
that. The 5-arm run is now clearly worth it, with per-epoch evals
(`evalEveryNEpochs := 1`) so the next one is not blind for seven hours.

## Workstream B' — prevent the collapse (the real lever)

### B'1 class-weighted CE — Status 2026-07-15: BUILT + FD-VERIFIED

`LossKind.perPixelWeightedCE (weights : List Float)` / `SegLoss.weightedCE`,
emitted by a `weights` parameter on `emitPerPixelCEBlock` (`[]` = the old
unweighted path, byte-identical). Selectable: `unet-brats-train data/brats
<epochs> wce`, and it is now the **default** — `dicece` lost that job when it
turned out to be CE with a rounding error.

**The weights, and why.** Measured over `train.bin` by
`scripts/brats_class_weights.py` (14,415 slices): background 97.46% / edema
1.60% / non-enhancing 0.44% / enhancing 0.50% of voxels. Inverse frequency
gives `[1.0, 60.9033, 220.0868, 195.5835]`, which makes every class contribute
**exactly 25%** of the loss, against 97.46 / 1.60 / 0.44 / 0.50 under plain CE.

That equal-contribution property is the argument for this workstream, because
it is *Dice's own stated goal* — "a ratio per class, so every class carries
equal weight no matter how few pixels it owns" (`Types.lean`, `perPixelDice`).
Dice reaches for it and cannot deliver, because its gradient carries a `p_i`
factor and vanishes on the collapsed class. CE's gradient is flat at `p → 0`.
**So B'1 is Dice's objective pursued with a gradient that still exists where it
is needed** — which is a sharper framing of the demo than "Dice vs CE" ever was.

**Reduction: `Σ_k w_{y_k}·CE_k / Σ_k w_{y_k}`, not `/N`.** Both sums are linear
in `w`, so the loss is **invariant to the overall scale of the weight vector** —
verified exactly (`|Δ| = 0.00e+00` for `w` vs `1000·w`). Two consequences worth
having: a caller cannot change the effective LR by normalizing its weights
differently, and the stock objection to inverse frequency ("a 200× dynamic range
destabilizes training") does not apply, since that objection is about `/N`, where
the weights inflate the gradient scale outright. Here the loss stays a weighted
*mean*, on CE's scale, self-normalizing per batch. `Σ_k w_{y_k}` is labels-only,
hence constant w.r.t. the logits and contributing no gradient term — which is
what makes the normalization affordable at all.

A pleasing side effect of scale-invariance: **median-frequency balancing
(SegNet) is not a separate option here.** It is `median(f)/f_c` = inverse
frequency times a constant, and the constant cancels. Under a `/N` reduction the
two schemes differ; under this one they are the same scheme, and
`brats_class_weights.py` asserts it rather than offering a phantom choice.

**The loss landscape flips, and it is the gradient story's other half.**
`scripts/brats_class_weights.py` now prints what each reference predictor
scores under each loss (measured class balance, exact):

| predictor | plain CE | weighted CE |
|---|---|---|
| predict the class **prior** everywhere | **0.1417** ← best constant | 3.7206 |
| predict **uniform** everywhere | 1.3863 | **1.3863** ← best constant |
| predict **background** everywhere | 17.5549 | **inf** |
| knows tumour location, uniform there | 0.0352 | 1.0397 |

**Under plain CE the cheapest constant predictor is "predict the prior" —
i.e. mostly background — and it is 10× cheaper than uniform. The doorway to
the collapse is the most attractive thing in the landscape, and descent walks
straight in.** Under weighted CE that inverts: the cheapest constant is
*uniform*, predicting the prior is 2.7× **worse**, and predicting background
everywhere is unbounded. The trivial answer stops being a trap and becomes the
worst place in the space.

This is the scorecard's claim restated as geometry rather than gradients — the
scorecard says *what the gradient does*, this says *where the minima are* — and
the two agree. Worth having both: the gradient argument explains why Dice can't
escape once collapsed; this explains why CE goes there in the first place.

It also makes the wce curve readable, which matters because the CE row was
misread once already (0.1328 < the 0.1417 floor looks like learning and wasn't).
On the weighted curve the number to beat is **1.3863** (any constant), and
**1.0397** is roughly "found the tumour, hasn't typed it yet". Same caveat as
ever, and it is the doc's own hard-won lesson: **these make the curve readable,
not conclusive.** No scalar in a training log can tell you a thin class
survived — only per-class IoU / region Dice at eval can.

**FD verification** (`scripts/seg_loss_probe_check.py`, CPU, no GPU) — 10/10:

| check | result |
|---|---|
| `wce` grad vs central FD | **9.41e-08** |
| `wce1` (all-ones weights) ≡ `ce` | `|Δloss| = 0.00e+00`, `max|Δgrad| = 3.73e-09` |
| `wce` ≠ `ce` (the weighting must actually do something) | 1.4205 vs 1.4747 |
| scale-invariance, `w` vs `1000·w` | `|Δ| = 0.00e+00` |

The last three exist because **FD alone cannot catch the bugs that matter here.**
FD only asks that the gradient match whatever loss was emitted — a no-op emitter
that silently ignored `weights` would sail through every FD check. `wce1 ≡ ce`
pins the `Σw` denominator (since `Σ_k 1 = N` by construction), and `wce ≠ ce`
pins that the weights are wired at all.

**An IREE bug, found and routed around.** The obvious spelling — broadcast the
weight to `[B,NC,H,W]` and reduce all four dims at once — **miscompiles**: IREE
fuses `reduce(multiply(x, broadcast(w)))` into a `vector.contract` and derives an
invalid accumulator shape (`llvm-cpu`: "invalid accumulator/result vector
shape"), on the *same* reduce op that compiles fine unweighted. Same family as
the `select_and_scatter` gap routed around at the maxPool backward
(`MlirCodegen.lean:6182`). The fix is to collapse the channel axis first, giving
per-pixel CE at `[B,H,W]`, and only then apply the weight — equal by
distributivity (`w_k` doesn't depend on `c`), keeps the weight out of any
reduce's operand tree, and is the more honest spelling anyway since the weight is
a per-pixel quantity. Recorded because the next person to touch this block will
otherwise re-derive it the obvious way and hit the same wall.

**Live on GPU, and it moves the model off the trivial predictor.** A 64-slice
overfit probe (val = train, 20 epochs) — a *plumbing* test, not a result:

| | c0 bg | c1 edema | c2 non-enh | c3 enhancing |
|---|---|---|---|---|
| `ce`/`dicece`, real val, 1 ep | 0.9735 | **0.000000** | 0.000250 | **0.000000** |
| `wce`, 64-slice overfit, 10 ep | 0.7052 | **0.035623** | 0.022263 | **0.080366** |

Every tumour class is off the floor. Two honest caveats: this is 64 images with
val = train, so it says nothing about generalization; and `wce` **over-predicts
tumour ~9×** (WT pred 1,176,442 vs gt 127,394), which is the expected failure
mode of aggressive inverse-frequency weighting and the opposite of collapse.
If that over-prediction survives to real budget, inverse-sqrt frequency
(`[1.0, 7.80, 14.84, 13.99]`, tumour ~21% of the loss) is the fallback and
`brats_class_weights.py` already prints it. Being able to trade *along* that
axis is the point; CE and Dice offered no axis at all.

### B'5 focal CE — Status 2026-07-15: BUILT + FD-VERIFIED

`LossKind.perPixelFocalCE (gamma : Float)` — `-(1-p_t)^γ·log p_t`, meaned.
`unet-brats-train … focal` (γ=2). The **third** answer to the imbalance, and
mechanically the opposite of B'1, which is the reason to have both:

- **wce amplifies the rare class.** Static, per-*class*, from label frequencies.
- **focal defunds the easy class.** Dynamic, per-*pixel*, from the current
  prediction. It does **not** amplify the rare class at all — measured, below.

α is deliberately omitted: the paper's α_t *is* wce's mechanism, and folding it
in would confound the two arms of the ablation it exists for.

The gradient is the **true** one — `A = γ·p_t·omp^(γ-1)·log p_t - omp^γ`, with
`dz_c = A·(y_c - p_c)` — not the detached-weight approximation the YOLOv1
objectness path uses (`%y1f_w0`). Detaching is defensible but yields a
`d_logits` that is not the derivative of the loss being reported, and FD would
correctly reject it. FD 14/14: `focal` at **1.64e-07**, and **`focal0` (γ=0)
≡ `ce` exactly** (Δloss 0.00e+00, Δgrad 3.73e-09) — the same species of
structural check as `wce1`, pinning the `/N` normalizer and `A`'s sign.

Probe gained a decimal parser (`w=1.0:60.9033:…`, `g=2.0`) so weight vectors
live in one place instead of being duplicated into the Lean probe and drifting.
Lean core has no `String.toFloat?`; `.drop` returns a `String.Slice` in this
toolchain, so `.toString` before reaching for the String API.

### The gradient scorecard — the framework, and it earns its keep immediately

`scripts/seg_grad_scorecard.py` (CPU, ~2 min, against the **emitted MLIR**)
generalizes the Dice vanishing-gradient probe to every arm. The reframing it
forces: **a loss does not save a rare class by having a large gradient on it.**
CE's gradient on a collapsed class is already the largest here and CE collapses
anyway. What matters is the rare class's gradient *relative to the easy 97% it
competes with* — table (C), `Σ|dz|` over class-3 px / `Σ|dz|` over background px.

Measured, from init (`z0=0`) to deep collapse (`z0=10`, `p₀@bg = 0.9998`):

| loss | (C) at init | (C) collapsed | mechanism |
|---|---|---|---|
| `ce` | 5.09e-03 | 2.87e+01 | — the rare class is out-voted from the start |
| `dice` | 2.84e-02 | **8.57e-02** | flat, and *declining* past z0=6 |
| `dicece` | 6.83e-03 | 1.00e+01 | ≈ ce, as Gate B showed |
| `wce` | **9.96e-01** | 5.61e+03 | constant **196×** ce = exactly `w₃/w₀` |
| `focal` | **5.15e-03** | **1.20e+08** | nil at init → ~4e6× ce once confident |

Three findings worth keeping:

1. **Dice's (C) starts ~5× better balanced than CE and ends ~335× worse.** Its
   correction *weakens* as the collapse deepens — positive feedback into the
   very state it was meant to prevent. That is "Dice can only help if the
   collapse never happens", quantified.
2. **focal never amplifies the rare class** — its (A) column sits on CE's to
   three digits. Its whole mechanism is (B): as background confidence goes
   0.25 → 0.9998, CE's majority gradient decays 4300× and focal's decays
   **2e10×**. It silences the crowd rather than handing the rare class a
   megaphone. This is invisible in (A) and was worth building the table to see.
3. **focal is a no-op at initialization** (1.01× CE) and only engages as the net
   grows confident — which *is* the collapse. Since the collapse is decided in
   the first ~100 steps, focal's open question is whether its feedback arrives
   in time, or shows up, like Dice, to a decision already made. wce has no such
   timing risk: it is a blunt state-independent constant.

**The probe was wrong first, and the wrongness is the lesson.** v1 pushed class
3's logit down and left the background at random logits. That measures (A)
correctly and (B) as a *flat line* — under which focal is indistinguishable from
CE, and we would have shipped that as a finding. With random logits `p₀ ≈ 0.25`:
the background is not **confident**, so `(1-p_t)^γ` has nothing to bite on. The
fix is to sweep `z₀` **up**, which is what the collapse actually is (it is how CE
gets below the class-prior floor) and moves both halves — `p₀@bg → 1` and
`p₃@tum → 0` — with one knob.

**First evidence the timing risk is real.** On the 64-slice overfit probe
(val = train, 10 ep — a plumbing test, not a result), `focal` collapsed where
`wce` did not:

| arm | c1 edema | c2 non-enh | c3 enhancing |
|---|---|---|---|
| `wce` | 0.035623 | 0.022263 | **0.080366** |
| `focal` | 0.000013 | 0.000108 | **0.000000** |

Two independent lines — the scorecard's (C)-at-init and this run — point the
same way. Not a verdict at 64 images, but it is the predicted failure showing up
where predicted, and it is worth knowing *before* buying 40 h of GPU.

### B'2 prior-bias init — Status 2026-07-15: BUILT + VERIFIED

`TrainConfig.headPriorBias : List Float := []` + `NetSpec.applyHeadPriorBias`.
Sets the head's bias to `log π_c` instead of zero, so the net starts predicting
the class prior rather than a uniform softmax. `unet-brats-train … pb`,
orthogonal to the loss arm because it is an init, not a loss.

Implementation is a splice, not a new primitive: `heInitParams` emits each
conv's bias right after its weights in layer order, so the head bias is the
**final NC floats** of the buffer. Applied after the bootstrap patch (which only
rewrites a backbone prefix) and before checkpoint resume (a full restore, which
must win). Priors need no normalization — a constant added to every logit is a
no-op under softmax, the same scale-invariance `perPixelWeightedCE` enjoys.

**Verified exactly**: `softmax(head bias) == π` to **1.9e-09** off the emitted
checkpoint, giving `z₀ - z₃ = 5.2726`.

**And that number is the point.** 5.2726 is `log(π₀/π₃)` — it lands the net
exactly on the scorecard's `z0 ≈ 5.3` row, which was already measured before
this was built:

| (C) at | ce | dice | wce | focal |
|---|---|---|---|---|
| `z0 = 0` (uniform init) | 5.09e-03 | 2.84e-02 | 9.96e-01 | **5.15e-03** |
| `z0 = 5.27` (prior-bias) | 2.60e-01 | 1.12e-01 | 5.08e+01 | **9.90e+01** |

**One bias vector is worth ~19,000× to focal, at step 0, and flips it from the
worst arm — tied with CE, a literal no-op — to the best, ~2× wce.** This is the
scorecard paying rent: the prediction was sitting in the table before the code
existed, and the code landed on the predicted row. It also explains why Lin et
al. ship prior-bias init *with* focal rather than as a separate trick — focal's
mechanism needs confidence to suppress, and at a uniform softmax there is none.
They are one idea. (It helps every arm — wce rises 51× too, since starting at
the prior beats starting uniform. focal is the one that goes inert → leading.)

**Preliminary, and it does not fully agree.** On the 64-slice overfit probe
(10 ep, val = train — a plumbing test):

| arm | c1 edema | c3 enhancing | WT Dice |
|---|---|---|---|
| `focal` | 0.000013 | 0.000000 | 0.000659 |
| `focal pb` | 0.001999 | 0.000000 | **0.021299** (32×) |
| `wce` | 0.035623 | **0.080366** | — |

`pb` helps focal by 32× on WT Dice and 154× on edema IoU, directionally as
predicted. But `wce` alone still beats `focal pb`, where the (C) table says
`focal pb` (98.98) should beat `wce` alone (0.996). Two honest readings: 64
images at val = train is too weak to rank arms, **and** (C) describes the
gradient regime at *init*, not the trajectory — wce holds its ratio for the
whole run by construction, while focal's tracks wherever the net actually goes.
Which is exactly the question the matched-budget run exists to settle, and the
disagreement is worth having on record before it does.

### B'3 foreground oversampling — Status 2026-07-15: MEASURED, then NOT built

Measured before building (`scripts/brats_oversample_probe.py`, CPU, ~2 min),
which is what the scorecard habit is for. **The result retires it as a headline
arm, and the reason is structural.**

nnU-Net forces ~33% of patches to contain foreground, and that lever is
enormous when most patches are pure background. On this dataset it is a **no-op
by construction**: `preprocess_brats.py` keeps only slices with ≥1 tumour voxel,
so **zero of the 14,415 training slices lack foreground** (measured). The win
nnU-Net gets was already banked at preprocessing time.

What remains is reweighting *within* an already-filtered set — biasing toward
tumour-rich slices, since a polar cross-section holds a few voxels and an
equatorial one holds thousands. The ceiling on that:

| scheme | enhancing voxels | vs uniform |
|---|---|---|
| uniform (status quo) | 0.498% | 1.00× |
| top-25%, f=0.5 | 0.852% | 1.71× |
| top-10%, f=0.67 (best practical) | 1.194% | **2.40×** |
| train on ONLY the richest decile (a bound, not a config) | 1.537% | 3.08× |

**~2.4×, against levers already built and FD-verified at 196× (wce), ~4e6×
(focal at a collapsed net), and ~19,000× (prior-bias, for focal at step 0).**

Not an argument that it is worthless — it is orthogonal, it composes with every
arm, and 2.4× is 2.4×. It *is* an argument against spending the next block of
GPU on it, and against it being an arm of the ablation. It also has a cost the
loss-side levers don't: implemented by duplication it doubles epoch time for
that 2.4×, and implemented by resampling it needs a new weighted sampler in the
shared train loop. wce gets 196× for a constant vector.

**Deferred, deliberately, with the measurement on record** — so this is a
decision rather than an oversight, and so nobody re-derives it from the nnU-Net
paper and assumes the 33% rule transfers. It does not; the filter already ate it.

**Still owed:** B'4 pure `.dice` for the record, and the matched-budget run —
now 5 arms worth running (`ce`, `dicece`, `wce`, `focal`, `focal pb`), not 2.

## Workstream C — mask-aware augmentation

`bratsIO.augmentBatch` is identity, same starting point pets had. Paired
hflip + scale/crop applied identically to image and mask (nearest-neighbour
for the mask — **never interpolate labels**). This is the same primitive
`unet_demo_v2.md` Workstream C wants for pets; build once, both demos get
it. Note the modality subtlety: brains are *near*-symmetric, so hflip is
safe here in a way it wouldn't be for an organ with strong laterality.

Gate C: aug beats bare on val mIoU at matched epochs.

## Workstream D — the 2.5D upgrade

Loader-only change (stack adjacent slices into channels, `ic = 4*(2k+1)`);
tests whether through-plane context is where the 2D ceiling actually is.
Cheap, and it's the measurement that tells us whether 3D is worth its
enormous price.

Gate D: 2.5D beats 2D at matched budget. The delta is the honest estimate of
what full 3D could buy — and if it's small, that's a *finding*, and it
retires the conv3d question rather than leaving it as vague ambition.

## Workstream E — verified-gradient tie-in — STRUCK 2026-07-15

Was: "Dice's VJP is a quotient of two linear reductions over the softmax —
well inside what the existing proof library composes. 'Verified Dice gradient
over a verified UNet, on brain MRI' is the segmentation peer of the classifier
chapters' claim."

**Struck per the audience decision.** This demo's claim is codegen + FD, not
proofs (see the header). The FD probe already gives a correctness story a
newcomer can *run* — `scripts/seg_loss_probe_check.py`, gradient vs central
finite differences to 1.3e-07 — and that is the right rigor for someone whose
question is "does this segment a tumour", not "is the adjoint certified".

Not deleted, because it is still a *true* observation and a future chapter
may want it. But it is not on this demo's path, and leaving it listed as a
"stretch" invites it to be treated as owed. It isn't.

## Workstream F — report Dice on the BraTS regions (the field's metric)

**The gap a medical student would notice first.** We report mIoU over the raw
label classes. Nobody in this field does. BraTS reports **Dice** — and on
*nested regions*, not raw classes:

| region | MSD Task01 labels | meaning |
|---|---|---|
| WT (whole tumour) | 1 ∪ 2 ∪ 3 | everything abnormal |
| TC (tumour core)  | 2 ∪ 3 | the resectable core (excludes edema) |
| ET (enhancing tumour) | 3 | contrast-enhancing, the surgical target |

(MSD pre-remapped the native BraTS 1/2/4 → 2/1/3, so the region definitions
have to be read off *MSD's* labels, not the BraTS paper's. Easy to get
backwards: MSD 1 = edema = BraTS 2; MSD 2 = non-enhancing/necrotic = BraTS 1.)

Until this lands, our numbers are not comparable to a single published BraTS
result — which for this audience is most of the point. Worse, the regions are
*nested*, so per-class IoU can't be converted to them by inspection.

**It is cheap.** `F32.segConfusion` already returns the `[NC*NC]` confusion
matrix, and region Dice falls straight out of it — for a region R ⊆ classes,
from `C[gt][pred]`:

```
inter_R = Σ_{g∈R} Σ_{p∈R} C[g][p]
|gt_R|  = Σ_{g∈R} Σ_p C[g][p]
|pr_R|  = Σ_{p∈R} Σ_g C[g][p]
Dice_R  = 2·inter_R / (|gt_R| + |pr_R|)
```

No new C helper, no new kernel — pure host arithmetic on the matrix we already
accumulate in exact `Nat`. Print WT/TC/ET Dice alongside the existing per-class
IoU (keep both: IoU stays the cross-demo comparable to pets).

Gate F: WT/TC/ET Dice printed at eval. Rough orientation for whether the demo
is in the right postcode — a competent 2D BraTS baseline lands around
WT ≈ 0.85+, TC ≈ 0.7-0.8, ET ≈ 0.6-0.7. We will be under that (484 volumes, no
aug, 2D, short budget) and that is fine; being able to *say* by how much is the
deliverable.

### Status 2026-07-15: DONE — Gate F met

Landed as predicted: **pure host arithmetic on the confusion matrix, no new
kernel and no second pass over val.** `iree-compile` reports `eval forward
(cached)` on the first run after the change, which is the mechanical proof that
the codegen surface was untouched.

- `DatasetIO.segRegions : List (String × List Nat) := []` (`Train.lean`), set to
  `[("WT",[1,2,3]), ("TC",[2,3]), ("ET",[3])]` in `bratsIO`. Empty default, so
  pets is byte-for-byte unaffected — it has no regions to report and prints
  none. Any future dataset with nested regions is now a one-line change.
- The eval block prints `val Dice <R>: <d>  (inter= gt= pred=)` per region,
  right after the existing per-class IoU line. IoU is **kept**, not replaced:
  it is what keeps this comparable to pets, while Dice is what makes it
  comparable to the literature.
- Counts stay in exact `Nat` until the final divide, same as the IoU path.

**Verified** (`scripts/seg_region_dice_check.py`, CPU, no GPU, ~15 s):

| check | result |
|---|---|
| confusion-matrix identity vs direct per-pixel Dice, 300 region-trials across class priors from uniform to Dirichlet α=0.02 | max abs error **0.000e+00** (exact) |
| all-background predictor scores 0, not the vacuous 1 | PASS all three regions |
| ground-truth region counts over the real `val.bin` | WT 3,896,201 / TC 1,455,581 / ET 768,187 voxels |

The last row is the one worth keeping: those counts depend only on the data, so
they pin the harness's `gt=` output against an independent number and catch a
transposed `conf[g][p]` — the one bug in this formula that produces a plausible
result instead of a crash. (Dice itself is *insensitive* to that transposition,
since `gt` and `pred` only ever appear as their sum. The `gt=`/`pred=`
diagnostics are what expose it, which is why they are printed.)

Measured on the real val split: ET is **0.52%** of all voxels and the whole
tumour is 2.6% — so the doc's "~1% of pixels" framing was, if anything,
generous about how thin the target class is.

## Workstream G — see the segmentation (`brats-predict`)

`MainPetsPredict` renders image | GT | pred PPM strips. There is no BraTS
equivalent, and for this audience the picture *is* the demo — a medical student
evaluates a segmentation by looking at it, not by reading a scalar.

Wants a little more than the pets version: pick a modality to render as the
grayscale backdrop (T1gd is the one clinicians read for enhancing tumour), then
overlay GT and prediction as colour masks with the standard region colours.
Choose slices with a real tumour cross-section, not the near-empty edge slices
`min_tumor_px = 1` lets through.

Gate G: a figure in `demos/figures/` that makes the CE-vs-Dice difference
*visible* — CE's collapse should be a picture of an empty brain next to a
labelled one. That figure is the demo's money slide, more than any table.

### Status 2026-07-15: DONE — Gate G met, and the picture is the argument

`demos/MainBratsPredict.lean` + `lake exe brats-predict`. Run against the Gate-B
checkpoint, it renders `demos/figures/brats_pred_dicece.ppm`: four val slices ×
three panels (T1gd | +ground truth | +prediction).

**The right-hand column is an empty brain.** Per-slice, against ~5,000
ground-truth tumour pixels:

```
slice 932:  gt tumour px=5350   predicted tumour px=0
slice 127:  gt tumour px=5330   predicted tumour px=0
slice 1185: gt tumour px=5046   predicted tumour px=0
slice 2490: gt tumour px=4939   predicted tumour px=0
```

That is the same fact as `mIoU 0.243402`, and it is not remotely the same
*claim*. The table's version is a number that looks mediocre; the figure's
version is a model that has never once fired a tumour class. This is the demo's
money slide.

Three decisions the pets renderer didn't have to make:

- **Backdrop = one modality (T1gd, channel 2).** Four co-registered modalities
  have no honest rendering as one image, and T1gd is what a clinician reads for
  enhancing tumour. This turned out to be self-checking: in the rendered figure
  the yellow enhancing label lands exactly on the bright rim of the backdrop,
  which is only true if the channel index is right. A wrong index would have
  put the label on unremarkable tissue.
- **Overlay, not a side-by-side mask.** A trimap beside a photo is legible; a
  tumour mask on black is not — a tumour is only interpretable against the
  anatomy it sits in. Alpha 0.55, standard BraTS colours (edema green,
  non-enhancing/necrotic red, enhancing yellow). The rendered result reads as
  a textbook ring-enhancing glioblastoma: yellow rim, red necrotic core, green
  edema halo.
- **Slices chosen, not taken in order.** `min_tumor_px = 1` means the head of
  val is near-empty tumour edges that would show nothing either way. Slices are
  ranked by tumour burden (on a stride-2 mask subsample — it is a ranking, not
  a measurement), preferring slices with ≥25 enhancing voxels since a slice
  without ET cannot show whether ET collapsed, then picked greedily with a
  60-index minimum gap. **The gap is what stops the figure being one tumour
  rendered four times** — slices are written volume-by-volume, so index
  distance proxies for "different patient". Confirmed by eye: the four
  rendered brains are visibly four different people.

**A gotcha worth knowing before the CE-vs-Dice figure:** every run writes the
*same* `<prefix>_params.bin`, so training a second loss clobbers the first.
`brats-predict` therefore takes optional params/bn-stats paths:

```
lake exe unet-brats-train data/brats 30 ce
cp .lake/build/unet_brats_*_params.bin   /tmp/ce_params.bin
cp .lake/build/unet_brats_*_bn_stats.bin /tmp/ce_bn.bin
lake exe unet-brats-train data/brats 30 dicece
lake exe brats-predict demos/figures/brats_ce.ppm /tmp/ce_params.bin /tmp/ce_bn.bin
lake exe brats-predict demos/figures/brats_dicece.ppm
```

Given the Gate B result — `ce` and `dicece` are the same trivial predictor to
four decimals — the honest CE-vs-Dice figure today is **two identical empty
brains**, which is a finding rather than a fizzle. The figure gets interesting
when Workstream B's revised plan (class-weighted CE) produces a model that
predicts the class at all.

## Sequencing

```
Phase 0  DONE   data path + spec + exe
Phase 1  A      baseline run, per-class IoU             (Gate A — DONE: collapsed)
Phase 2  B      Dice / Dice+CE + the loss ablation      (Gate B — DONE: FAILED,
                                                         mechanism measured;
                                                         Dice ∝ p_i vanishes)
Phase 3  F      WT/TC/ET Dice — the field's metric      (Gate F — DONE)
Phase 4  G      brats-predict visualizer                (Gate G — DONE)
Phase 5  B'     PREVENT the collapse — the real lever   (Gate B' — IN FLIGHT)
                  B'1 class-weighted CE               DONE + FD 9.41e-08
                  B'5 focal CE                        DONE + FD 1.64e-07
                  ——  gradient scorecard              DONE — the framework:
                      measures which arm CAN work for 2 min of CPU
                  B'2 prior-bias head init   (nearly free; targets the
                      first-100-steps window — and is focal's natural
                      complement, since focal is a no-op at a uniform softmax)
                  B'3 foreground oversampling
                  B'4 pure `.dice` for the record — the mechanism predicts it
                      collapses too, just slower. Cheap falsification.
                  B'6 re-test Dice on top, as *polish* not rescue
Phase 6  C      mask-aware aug                          (Gate C)
Phase 7  D      2.5D                                    (Gate D) → gates planning/unet3d.md
        (E struck — codegen + FD is the claim)
```

F and G moved ahead of aug/2.5D deliberately: they are what turn this from
"a training run that prints numbers" into something a newcomer can pick up,
and they are both cheap. Quality levers come after the demo is legible. That
ordering paid off exactly as argued — Gate G's figure is what makes the Gate B
failure legible at a glance, and neither the mIoU table nor the loss curve
could do that.

**Phase 5 is Workstream B re-scoped, not repeated.** The original Gate B asked
whether Dice *rescues* a collapsed class; the answer is measured and it is no,
because Dice's gradient carries a `p_i` factor that vanishes precisely where
the rescue would have to happen. Do not re-run the Dice ablation expecting a
different number. The remaining lever acts in the first ~100 steps, before the
collapse — see the Gate B result for the mechanism.

**Still owed regardless: the matched-budget run.** Every number in this doc is
1 epoch. That is enough to establish the collapse and its mechanism (the net
sits on the trivial minimum from step 100 onward and the gradient analysis is
budget-independent), and it is *not* enough to publish a CE row. Gate B' should
be scored at 30-60 epochs, one loss per GPU.

## On replacing the pets demo

The opening question was whether BraTS *replaces* pets. Recommendation:
**not a replacement — a promotion, with pets retained.** They teach
different things and the pair is worth more than either alone:

- Pets answers "**what do skip connections buy for dense prediction?**" —
  it has the skipless `autoencoderPets` twin, which is the controlled A/B
  for that question. BraTS has no such twin and shouldn't grow one.
- BraTS answers "**what happens when the class you care about is 1% of the
  pixels?**" — and pets *cannot* answer that convincingly, because on a
  trimap nobody believes the boundary class matters.

Pets is also 738 MB and trains in minutes; BraTS is multi-GB and needs a
real budget. Keeping the cheap one as the smoke test has practical value
beyond pedagogy.

The honest move is to demote pets from "the segmentation demo" to "the skip
ablation", and let BraTS carry the segmentation *result*. That does leave
`unet_demo_v2.md`'s Workstreams B/C (real pets budget, aug) open — they're
worth finishing only insofar as the skip ablation needs a real budget to be
credible, which it does (the 3-epoch A/B currently shows the autoencoder
*ahead*, 0.360 vs 0.344 — a budget artifact that should not be left standing
as the published number).

## Out of scope

Full 3D conv **for this demo's first result** — but not out of scope for the
repo, and the scoping above deliberately de-risked it (IREE rank-5 verified,
codegen surface measured at ~675 lines, proofs optional by the decoder's own
precedent). It is its own workstream with its own doc — `planning/unet3d.md`
— gated on Gate D
telling us through-plane context is actually where the ceiling is. Also out:
BraTS 2021 leaderboard comparison (different data,
see the data decision), instance-level tumour counting, survival prediction
(the other MSD/BraTS task), transpose-conv upsampling (`unet_demo.md`
declined it on checkerboard grounds and that still holds), the other nine
Decathlon tasks (Task01 is the family representative; a second task is a
loader change and its own scoping line, not a rider).
