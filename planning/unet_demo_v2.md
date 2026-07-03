# unet_demo_v2.md — UNet segmentation demo, second pass

Goal: finish what v1 started. The UNet demo's *infrastructure* all
landed — and then the effort pivoted to DDPM (which reused the skip
codegen the same week) before anyone ran a real training. Today the
repo has a working seg pipeline, a smoke-test checkpoint, a render
figure, and **no quantitative segmentation number anywhere**: mIoU
is still a `TODO` in `Train.lean` ("seg eval skipped — Phase 0"),
augment is an identity placeholder, the trainer config is 3 epochs
at lr 1e-3, and RESULTS.md has no UNet row. v2 is: metric, real
training run, augmentation, then the one big quality lever
(pretrained encoder) that shares codegen with the YOLO v2 plan.

Prerequisite reading: `planning/unet_demo.md` (v1 plan + the
2026-05-05/06 progress log). Companion: `planning/yolo_demo_v2.md`
Workstream D (`.saveFeature`) — Workstream D here is the same
primitive; whoever builds it first, the other demo gets it free.

## Where v1 landed (recap, one paragraph)

Everything on v1's "new primitives" list shipped and is
FD-verified to ~1e-11: `bilinearUpsample` forward+backward codegen
(factorized `Wy·X·Wxᵀ` dot_generals), channel concat/split
sub-primitives, per-pixel softmax-CE (`useSeg` /
`emitPerPixelCEBlock`), the seg train-step ABI
(`trainStepAdamF32Seg`), and — the hard one — `unetDown`/`unetUp`
skip-state codegen via the emit-time LIFO stack
(`MlirCodegen.lean:1490`), which the DDPM demo then built on. Data
side: `download_pets.sh` + `preprocess_pets.py` (3,680 train /
3,669 val, 3-class trimap), `F32.loadPets`, `DatasetKind.pets`.
Demos: `MainAutoencoderPetsTrain` (skipless Phase-1 smoke),
`MainUnetPetsTrain` (`unetPets`, depth 4 / base 32, 7.85M params),
`MainPetsPredict` (image | GT | pred PPM strips →
`demos/figures/unet_pets.png`).

## What's actually open (v1's own list, audited 2026-07-03)

| v1 item | Status now |
|---|---|
| #1 Train.lean seg path | ✓ done (`useSeg` branch, seg ABI call) |
| #2 `autoencoderPets` | ✓ done |
| #3 `unetDown`/`unetUp` skip codegen | ✓ done (reused by DDPM) |
| #4 Mask-aware augmentation | ✗ `petsIO.augmentBatch` is identity |
| #5 mIoU eval | ✗ `Train.lean:693` TODO; eval block skips seg |
| (implicit) a real training run | ✗ 3-epoch smoke config; no runs/, no RESULTS.md row |

So v2 is not a redesign — it's the last 20% that turns "the
pipeline works" into a bestiary-grade result. Same governing lesson
as the DDPM and YOLO v2 docs: **no scalar we currently print is a
quality measure** (train CE says nothing about boundary quality or
class collapse), so the metric comes first.

## Workstream A — mIoU + per-class IoU (do first, no codegen)

1. **mIoU harness** (~half session). Host-side over the val set:
   eval forward (the path `MainPetsPredict` already exercises) →
   argmax over the 3 channels → per-class intersection/union
   accumulated in Lean (or dump argmax maps and score in a
   `scripts/pets_miou.py`, mirroring the YOLO-v2 harness choice).
   Report per-class IoU (fg / bg / boundary) + mean + per-pixel
   accuracy. Boundary is ~12% of pixels and genuinely thin — the
   per-class breakdown is the honest view; mean-of-3 alone would
   hide a boundary collapse.
2. **Wire it into the epoch-eval block**, replacing the
   `Train.lean:693` "seg eval skipped" branch, so every future run
   logs mIoU alongside loss (classification's eval cadence, every
   10 ep). The eval-vmfb-at-train-batch constraint applies here
   too; full batches + drop the tail is fine at 3,669 val records.

Gate A: score the existing smoke checkpoint (or retrain 3 ep if
stale). Any number is fine — it's the baseline row.

## Workstream B — the real baseline run + the skip ablation

The compute is trivial next to the other demos (7.85M params,
230 steps/epoch at batch 16 — minutes per epoch on mars), which
makes the full recipe-ablation pattern from the classification
chapters affordable:

1. **Real config**: 60–80 ep, cosine decay + warmup, wd 1e-4 —
   the classification-chapter defaults, tuned only if it visibly
   misbehaves.
2. **The skip ablation** — `unetPets` vs `autoencoderPets` at
   matched budget, scored by mIoU. v1's smoke test claimed "loss
   drops as fast or faster"; v2 turns the *entire pedagogical point
   of UNet* ("what do skip connections buy for dense prediction?")
   into a two-row table. This A/B is the demo's money slide and
   costs one extra short run.
3. RESULTS.md gets its first segmentation rows; refresh
   `demos/figures/unet_pets.png` from the real checkpoint.

Gate B: UNet beats the skipless autoencoder on mIoU (expected
decisively, especially boundary-class IoU — skips carry exactly
the high-frequency detail the bottleneck destroys). If it
*doesn't*, that's a skip-plumbing bug worth finding before any
other demo trusts the stack.

## Workstream C — mask-aware augmentation (v1 item #4, verbatim)

Replace the identity `petsIO.augmentBatch` (~1 session, host-side
FFI like the classification aug pack): paired hflip + random
scale/crop applied identically to the f32 image and the uint8
mask (nearest-neighbor resample for the mask — never interpolate
labels), plus image-only color jitter. Pets is 3,680 training
images and the model is 7.85M params — augmentation is likely the
single biggest cheap win on val mIoU. Ship as the v1-planned
two-cell ablation: `unet-pets` vs `unet-pets-aug`.

Gate C: aug beats bare on val mIoU at matched epochs (and by how
much — the classification chapters ask the same question, so the
delta is itself bestiary content).

## Workstream D — pretrained encoder (the quality lever, shared codegen)

On 3.7K images, a pretrained encoder usually dominates every other
lever. The pieces all exist separately; the missing 20% is the same
primitive the YOLO v2 doc wants:

- **Encoder**: the R34-ImageNet checkpoint + `bootstrapBackbone`
  prefix loading are proven (YOLOv1 does exactly this). But
  `unetDown` is its own conv-pair structure — the bootstrap only
  works if the encoder *is* the R34 layers.
- **The missing piece**: skips off `residualBlock` stage outputs.
  That is precisely `.saveFeature` from `yolo_demo_v2.md`
  Workstream D (a no-param marker pushing SSA + shape onto the
  existing skip stack). Here it feeds `unetUp`-style decoder
  stages (upsample + concat + convBn — concat/split sub-primitives
  already exist from v1) instead of an FPN head.
- Spec sketch: R34 stem + 4 residual stages with `.saveFeature`
  after stages 1–3, decoder of 3–4 `unetUp`-equivalents, 1×1 head.
  Roughly a ResNet-UNet / light DeepLab-family hybrid, ~25M params.
- Estimate: ~1–2 sessions *if* `.saveFeature` lands here first
  (budget the pidx/BN-stat audit tax, as always), ~1 session if
  the YOLO effort already paid for it. Coordinate: build it once.

Gate D: R34-encoder UNet vs `unetPets`+aug at matched wall-clock.
Expected: large mIoU jump, especially fg/boundary. This checkpoint
also becomes the natural backbone for any future DeepLab/SegFormer
promotion.

## Workstream E — loss refinements (only if the numbers ask for it)

Deliberately gated behind D, not before: on the trimap task the
boundary class is thin (~12%) and per-class IoU from Workstream A
will say whether the loss is the problem.

- **Class-weighted per-pixel CE** (~half session): a 3-vector
  weight in `emitPerPixelCEBlock` — one constant multiply on the
  per-pixel loss and gradient seed. Cheapest counter to boundary
  neglect.
- **Dice loss** (new codegen, ~1 session): only if weighted CE
  demonstrably fails; v1 already scoped it out as
  medical-imaging-grade complexity, and that judgment stands.

## Workstream F — verified-gradient tie-in (stretch, on-brand)

The seg stack's codegen is FD-verified but none of it has VJP
theorems: `bilinearUpsample` is the appealing target — it's a
*linear* op (fixed interpolation matrices), so its VJP proof is a
transpose-of-linear-map statement the existing matmul lemma family
should nearly close by composition. Per-pixel CE is the softmax-CE
VJP lifted spatially — the row-wise lifting machinery is exactly
what the proof library already does well. Either or both promote
the segmentation demo to the same "verified loss gradient over a
verified backbone" claim as the classifiers. ~1–2 sessions; not on
the demo's critical path.

## Sequencing

```
Phase 0 (1 session):             A  mIoU harness + wire into epoch eval (Gate A)
Phase 1 (1 session + short runs): B  real baseline + skip ablation (Gate B)
Phase 2 (1 session + runs):      C  mask-aware aug, two-cell ablation (Gate C)
Phase 3 (1–2 sessions + runs):   D  .saveFeature + R34-encoder UNet (Gate D)
Phase 4 (conditional):           E  weighted CE (only if boundary IoU says so)
Phase 5 (optional):              F  bilinearUpsample / per-pixel-CE VJP proofs
```

Phases 0–2 are the committed core — **the cheapest v2 of the three
demo docs** (≈3 sessions, runs measured in minutes-to-an-hour, no
overnights) because v1 built everything and stopped one metric
short of a result. Phase 3 is where the demo becomes genuinely
good, and its codegen is shared with (or free from) the YOLO v2
plan — sequence whichever demo wants it first.

## Deliverables

- mIoU in the seg epoch-eval path (delete the `Train.lean:693`
  TODO) + `scripts/pets_miou.py` if the Python-side route is chosen
- RESULTS.md segmentation table: autoencoder / UNet / UNet+aug /
  R34-UNet, per-class IoU + mIoU
- Refreshed `demos/figures/unet_pets.png` from the best checkpoint
  (image | GT | pred strips already render; pick examples showing
  boundary quality)
- `petsIO.augmentBatch` real implementation + the `unet-pets-aug`
  cell
- If D lands: `.saveFeature` Layer kind (shared with
  `yolo_demo_v2.md`) + `r34UnetPets` spec
- Bestiary/blueprint: UNet entry graduates from "the pipeline
  works" to a scored result with the skip ablation as the teaching
  table

## Out of scope (unchanged from v1, plus)

ADE20K / Cityscapes / VOC-seg (Pets stays the demo target until a
metric exists to justify bigger data), instance segmentation,
transpose-conv upsampling, full DeepLabv3+ (`asppModule` stays
shape-only; the R34-encoder UNet is the family representative),
SegFormer/MobileViT promotion (unlocked by the DDPM
`spatialFlatten` work but each deserves its own scoping line, not
a rider), diffusion-style uses of the UNet (that's
`ddpm_demo_v2.md`'s lane).
