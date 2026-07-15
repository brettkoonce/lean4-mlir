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
what's still open on the pets side).

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

## Workstream B — Dice loss (the lever this demo exists to pull)

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

**Free win to fold in:** label smoothing on `perPixelCE` is already
implemented in the emitter (`MlirCodegen.lean:4312-4314`) but gated off
host-side (`Train.lean:127-128` throws). The codegen is there; only the
guard needs lifting.

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

## Workstream E — verified-gradient tie-in (stretch, on-brand)

Same as `unet_demo_v2.md` Workstream F, and the argument is stronger here:
if Dice lands, its VJP is a quotient of two linear reductions over the
softmax — well inside what the existing proof library composes. "Verified
Dice gradient over a verified UNet, on brain MRI" is the segmentation peer
of the classifier chapters' claim.

## Sequencing

```
Phase 0  DONE   data path + spec + exe (this session)
Phase 1  A      baseline run, per-class IoU            (Gate A)
Phase 2  B      Dice / Dice+CE + the loss ablation     (Gate B)
Phase 3  C      mask-aware aug                         (Gate C)
Phase 4  D      2.5D                                   (Gate D)
Phase 5  E      Dice VJP proofs                        (optional)
```

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
precedent). It is its own workstream with its own doc, gated on Gate D
telling us through-plane context is actually where the ceiling is. Also out:
BraTS 2021 leaderboard comparison (different data,
see the data decision), instance-level tumour counting, survival prediction
(the other MSD/BraTS task), transpose-conv upsampling (`unet_demo.md`
declined it on checkerboard grounds and that still holds), the other nine
Decathlon tasks (Task01 is the family representative; a second task is a
loader change and its own scoping line, not a rider).
