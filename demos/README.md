# Demos

Trainers and inference exes that ride on top of the chapter-aligned
classification stack. Top-level `Main*Train.lean` files are the
chapters themselves (MLP, CNN, ResNet, MobileNet, EfficientNet,
ConvNeXt, ViT); these demos extend the framework into adjacent
domains — segmentation, generative models, language modeling,
explainability — without changing the underlying codegen path.

Build any of these with `lake exe <name>` after the relevant
chapter trainer has produced its checkpoint.

---

## ResNet-34 UNet — brain-tumour segmentation (BraTS)

The segmentation demo. A ResNet-34 encoder (the Ch-5 architecture, reused
verbatim as the contracting path) + a UNet decoder, on MSD Task01_BrainTumour:
224×224 axial slices, 4 co-registered MRI modalities (FLAIR / T1w / T1gd / T2w)
→ 4 tumour classes. 24.5M params, plain per-pixel CE, 10 epochs.

`MainUnetBratsR34.lean`, `MainBratsPredict.lean`. See
`planning/r34_brats_retrain.md`.

```bash
./download_brats.sh
python3 preprocess_brats.py data/brats/Task01_BrainTumour data/brats224 \
        --size 224 --seed 0            # same patient split as data/brats
./scripts/run_brats_r34_ab.sh 10 data/brats224 # both arms, one per GPU
lake exe brats-predict net=r34 arm=scratch,r34 out.ppm
```

| arm | mIoU | WT | TC | ET |
|---|---|---|---|---|
| `r34` (ImageNet bootstrap) | 0.742 | 0.911 | 0.870 | 0.858 |
| `scratch` (He-init) | 0.740 | 0.910 | 0.869 | 0.856 |

![R34 UNet transfer on BraTS](figures/brats_r34_skip_transfer.png)

`T1gd | ground truth | +scratch | +r34`. Edema green, non-enhancing/necrotic
core red, enhancing tumour yellow — the yellow rim around a red core is a
textbook ring-enhancing glioblastoma.

**Two things this demo measures, and they are not the same size.**

*Skips are worth ~10 points.* Same backbone and schedule, decoder with and
without the encoder concat: **0.635 → 0.740 mIoU**, the largest gain on ET
(+0.12), the thinnest structure — a skipless decoder has to rebuild every
boundary from a 7×7 bottleneck. Run the ablation with `noskip`.

*Transfer buys one epoch, not a better model.* The two arms differ in exactly
one field (`bootstrapBackboneRange`), so 86.8% of params start pretrained vs
random and everything else is identical. At **epoch 1** the bootstrapped arm is
already at ET Dice 0.818 while the control sits at 0.184, still collapsed on
the hard classes. By epoch 2 the control has caught up, and the peaks above are
a tie (+0.002, noise at n=1). The honest claim is sample-efficiency: same
quality, one epoch sooner. Transfer's payoff scales inversely with dataset
size, and 14,415 slices is a lot — a data-fraction sweep is the experiment that
would show it properly.

The backbone is `.lake/build/jax_r34_imagenet.bin`, trained by this stack on
ImageNet to 72% top-1. Nothing is downloaded. Its stem is 3-channel RGB and
BraTS needs 4, so the transferable weights are not a prefix — hence
`bootstrapBackboneRange`, which patches a byte *range* and leaves the fresh
stem He-init. It self-checks on every run: the patched window must be
byte-equal to the checkpoint and the stem must be untouched, or it throws.

---

## FPN detector — object detection on VisDrone

Multi-scale detection on real drone-altitude imagery, and the best demo in this
repo for showing what the stack does end to end: a ResNet-34 backbone **trained
by this stack on ImageNet** feeds an FPN top-down neck into three anchor heads at
strides 8/16/32, with a DIoU box loss, focal objectness and focal class CE.
448 px input, 10 VisDrone classes.

VisDrone is the point: a median image holds **70 objects** and many are 2–5 px
after the resize, which is the regime where a single coarse grid structurally
cannot work and multi-scale detection stops being decoration.

`MainYolov1VisdroneFpn.lean`. See `planning/visdrone_detector.md`.

```bash
./download_visdrone.sh
# ⚠ write the anchor priors from the values hardcoded in the demo — do NOT
# re-run k-means, or encoder and model silently disagree
python3 preprocess_visdrone.py data/visdrone data/visdrone_fpn \
    --size 448 --grid 14 --fpn data/visdrone
python3 preprocess_visdrone.py data/visdrone data/visdrone448 --size 448 --grid 14

# the current best recipe — ~2 h on one RTX 4060 Ti
CUDA_VISIBLE_DEVICES=0 FPN_BACKBONE=r34 FPN_TAG=run1 \
  FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_EPOCHS=30 \
  lake exe yolov1-visdrone-fpn data/visdrone_fpn

CUDA_VISIBLE_DEVICES=0 FPN_BACKBONE=r34 FPN_TAG=run1 \
  lake exe yolov1-visdrone-fpn infer data/visdrone_fpn runs/fpn_run1

python3 scripts/yolo_map_visdrone.py runs/fpn_run1/logits.bin \
    data/visdrone448/val.bin --fpn data/visdrone --grid 14 \
    --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05

python3 scripts/fpn_render.py runs/fpn_run1/logits.bin data/visdrone_fpn/val.bin \
    --gt data/visdrone448/val.full_gt.bin --diverse --scale 2 --topk-per-gt \
    --layout cols --n 4 --out demos/figures/visdrone_fpn.png
```

⚠ Three flags that fail *silently* rather than loudly:
- **`FPN_TAG` must be set on `infer` too.** Without it the eval loads a different
  arm's weights, and the only tell is an epoch sweep whose rows are identical.
- **`FPN_BACKBONE` defaults to `r50`, not `r34`** — omit it and you train a
  different arm than the one these numbers come from.
- **`--topk` defaults to 1000**, which truncates the multilabel candidate list.
  The same checkpoint reads 0.1919 instead of 0.1961 at the default.

Knobs: `FPN_BACKBONE` (`r34`/`r50`), `FPN_AUG`, `FPN_AFFINE` (percent probability
of the box-aware scale/translate transform), `FPN_CLSW`, `FPN_CLSFOCAL`,
`FPN_EPOCHS`, `FPN_TOWER`, `FPN_NOBOOTSTRAP`.

![VisDrone FPN detection](figures/visdrone_fpn.png)

Truth on top, prediction below, on four val frames. **mAP@0.5 = 0.2363**
(recall 0.769, class-agnostic AP 0.487) at 30 epochs, and 65 fps on one RTX
4060 Ti — or **35.7 fps on a 25 W Jetson Orin Nano** under TensorRT fp16, which
is the deployment this dataset implies. That beats a hand-written PyTorch replica
of this same architecture (0.1532) by **54%**.

⚠ Frames are picked with `--diverse`. Picking the *densest* frames selects
consecutive frames of one VisDrone sequence — val records are video — so the
figure ends up showing a single street corner four times.

⭐ **Augmentation and schedule length are one decision, not two.** At 50 epochs
*without* augmentation the same arm scores 0.1243 — worse than 12 epochs
(0.1526), with half the train loss: ordinary overfitting on 6,471 images.
Photometric augmentation recovers it, but at that strength 12 epochs still beats
50 (0.1961 vs 0.1674). Augmentation that changes object **scale** inverts the
ordering again, making 30 epochs worth 44% more than 12. A stronger augmentation
needs a longer schedule to absorb it, so the optimum epoch count is a property of
the augmentation pack — there is no schedule to tune once and carry across packs.

⭐ **The result is in the per-class split, not the mean.** Per-class AP runs from
car (0.685) to bicycle (0.036), and scale augmentation narrowed that spread from
43× to 19× by lifting exactly the classes that were worst: awning-tricycle +59%,
bicycle +41%, tricycle +40%, against car's +7%. Reweighting the loss toward rare
classes buys their recall at the expense of precision, and average precision
charges for the trade; supplying the scales those classes are missing raises both
at once. Detection on aerial imagery does not degrade uniformly — it collapses on
whatever is small *and* rare, and an averaged mAP hides exactly that.

![VisDrone predictions coloured by correctness](figures/visdrone_fpn_match.png)

The same frames coloured by **correctness** rather than class — green hit, red
false positive, yellow missed ground truth — with the 30-epoch arm on the left
and the 12-epoch one on the right. Read the per-frame counts in the labels: the
gain on any single dense frame is a few boxes, because most of the improvement is
rare-class ranking spread across all 548 val images and no one frame displays it.

A YOLOv8s at the same budget scores 0.140; its published-style 0.391 comes from
8× the epochs, higher resolution, full augmentation and COCO pretraining, so that
gap is recipe rather than architecture — and scale augmentation alone has closed
31% of it.

---

## DDPM — diffusion generative models

Denoising diffusion on MNIST. A tiny UNet predicts the noise
ε(x_t, t) that was added to an image; sampling runs that prediction backwards.
Cosine ᾱ schedule, DDIM (η=0) with 50 steps subsampled from T=1000, time
conditioning via a tiled `t/T_max` channel — which needs no new codegen
primitive, the UNet just sees one extra input channel.

`MainMnistDdpmTrain.lean` + `Sample`. Tiny UNet, base 16, 50 epochs.
See `planning/ddpm_demo.md`.

```bash
lake exe mnist-ddpm-train data 50
lake exe mnist-ddpm-sample runs/mnist_samples.ppm          # 4x4 grid of samples

# the two-row trajectory figure below
lake exe mnist-ddpm-sample trajectory data=data img=7
python3 scripts/ddpm_trajectory_figure.py \
    runs/2026-09-02-mnist-ddpm/trajectory.ppm \
    --out demos/figures/ddpm_mnist_trajectory.png
```

![DDPM forward and reverse trajectories on MNIST](figures/ddpm_mnist_trajectory.png)

**Top: the forward process.** A real MNIST training digit at nine points along
the schedule, x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε. Nothing is learned here — it is the
fixed corruption the model is trained to invert, and by the right-hand end the
digit is gone.

**Bottom: the reverse process**, read right to left. Sampling starts from fresh
N(0, I) noise and walks back down the same schedule, and a **different** digit
condenses out of it. That is the point of the row: it is not a reconstruction of
the 3 above it. The model never sees that image during sampling — it has learned
what MNIST digits look like at every noise level, and any noise seed lands
somewhere on that manifold.

⭐ **The two rows are aligned by noise level, not by step index.** Column *c* of
both rows sits at the same ᾱ, so reading down a column compares "a real digit
this corrupted" against "what the model can still recover from that much noise."
Indexing the bottom row by sampler step instead would have made the columns
incomparable and the figure decorative. Both rows are emitted by the sampler
itself (`mnist-ddpm-sample trajectory`), using the same ᾱ table and the same
`ddimStep` primitive as an ordinary run, so the picture cannot drift from the
process it illustrates; `scripts/ddpm_trajectory_figure.py` only upscales and
labels.

⚠ Most of the visible change happens in the last few columns. That is the cosine
schedule, not a rendering artifact — ᾱ stays low across most of the trajectory
and the image resolves late.

---

## TinyGPT — character-level language model

Char-level transformer on Karpathy's tinyshakespeare. Three new
codegen primitives shipped to support it:

- `tokenPositionEmbed` (one-hot → embed + learnable position)
- `lmHead` (per-position dense + reshape into `useSeg` loss path)
- `causalMask` flag on `transformerEncoder`

212K params (T=64, D=64, 4 layers, 2 heads). Trains in ~11 min on
gfx1100 for 10K Adam steps. See `planning/tinygpt_demo.md`.

```bash
./download_shakespeare.sh             # downloads tinyshakespeare.txt
python3 preprocess_shakespeare.py     # builds train.bin / val.bin / vocab.txt
lake exe tinygpt-shakespeare train    # trains, saves params
lake exe tinygpt-shakespeare sample 600 80 "ROMEO:"
```

Sample output after 10K steps (loss 1.45 nats/char ≈ 2.10 bits/char):

```text
ROMEO:
I prately I head.

LORD CAY:
God the goodness hath storn, so given to my love
To request and of the faces; with sun
Do not only to witness musrer Claudio.

POMPEY:
Alack, perpeal to amend my heart desires,
That one you to thine more, would I know,
And spuress and the seals destaint in heirs;
is more news, that she now to Lamentio.

KING RICHARD III:
What all strangthes me not I am not me:
To hich dost Grey, If thou know me not to such ounts?
```

Real Shakespeare character names (KING RICHARD III, KING HENRY
VI, ISABELLA, POMPEY, PRINCE, ROMEO), reference to Claudio (from
*Measure for Measure*), coherent multi-line dialog with proper
cadence and punctuation. Semantic coherence drops past the 64-char
context window — exactly what the planning doc predicted.

A `bigram-shakespeare` baseline (single dense V→V predicting next
char given current char) also lives here as a smoke test that the
data pipeline + sampler work end-to-end without the transformer.

---

## Layout

The four demos above are the maintained set. Everything else lives in one of two
subfolders — moving a file does **not** change its executable name, so every
`lake exe <name>` in this repo, in `scripts/` and in CI still works unchanged.

```
demos/
├── README.md                              # this file
├── figures/                               # rendered outputs for the README
│
│   # ── the four demos ──
├── MainUnetBratsR34.lean                  # R34-UNet on BraTS (segmentation)
├── MainUnetBratsTrain.lean                # from-scratch UNet on BraTS
├── MainBratsPredict.lean                  # render predicted masks from a checkpoint
├── MainYolov1VisdroneFpn.lean             # R34+FPN detector on VisDrone, train + infer
├── MainMnistDdpmTrain.lean / Sample       # DDPM on MNIST (Sample also writes the
│                                          #   two-row trajectory figure)
├── MainTinyGptShakespeare.lean            # char-level transformer
├── MainBigramShakespeare.lean             # bigram baseline (validates the data pipeline)
├── MainTinyStories.lean                   # the same transformer at a larger corpus
│
├── probes/                                # gates and tools, not demos — these RUN IN CI
│   ├── MainFpnLossProbe.lean              #   finite-difference gate on the detector loss
│   ├── MainFpnNeckProbe.lean              #   FPN neck shapes
│   ├── MainFpnDetectProbe.lean            #   detector head
│   ├── MainFpnTrainEmit.lean              #   emit the detector train step
│   ├── MainAnchorLossProbe.lean           #   anchor loss
│   ├── MainDiouLossProbe.lean             #   DIoU box loss
│   ├── MainSegLossProbe.lean              #   segmentation losses
│   ├── MainGradFdProbe.lean               #   generic finite-difference gradient check
│   ├── MainFlashProbe.lean                #   flash-attention
│   ├── MainMnistDdpmScore.lean            #   DDPM sample scoring
│   ├── MainGradCAM.lean                   #   closed-form CAM for GAP+dense nets
│   └── MainInspectConvNeXt.lean           #   checkpoint diagnostics
│
└── archive/                               # superseded; kept building, not maintained
    ├── MainUnetPetsTrain.lean             #   UNet on Pets, superseded by BraTS
    ├── MainAutoencoderPetsTrain.lean      #   autoencoder baseline (no skips)
    ├── MainPetsPredict.lean               #   Pets mask rendering
    ├── MainYolov1PetsTrainBootstrap.lean  #   YOLOv1 on Pets, superseded by VisDrone
    ├── MainYolov1PetsInfer.lean           #   Pets detection dump
    ├── MainYolov1VisDrone448.lean         #   single-scale VisDrone arms, superseded
    ├── MainYolov1VisDrone448S16.lean      #     by the FPN detector
    ├── MainYolov1VisDroneAnchor.lean      #
    ├── MainCifarDdpmTrain.lean / Sample   #   DDPM on CIFAR-10
    ├── MainCifarDdpmAttnTrain.lean / …    #   bottleneck-attention variant (codegen ✓, recipe ✗)
    ├── MainCifarDdpmSincosTrain.lean / …  #   sincos t-embed variant (small negative)
    └── MainDiffusion2d.lean               #   2-D toy diffusion
```

Per-demo planning docs live in `planning/` at the repo root.
