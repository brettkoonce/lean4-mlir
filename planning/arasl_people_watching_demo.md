# arasl_people_watching_demo.md — the chapter-4 CNN on Arabic sign-language letters

Goal: a *People watching* entry for the bestiary. Run chapter 4's
CIFAR-CNN8-wide-BN, unchanged bar the stem and the head, on ArASL — 54,049
grey 64×64 crops of hands spelling the 32 letters of the Arabic sign-language
alphabet — under two splits of the same images: the random split every
published number uses, and a split that keeps consecutive frames together.
The images are video bursts (§2), so the random split puts near-identical
frames of the same hand on both sides of the line, and the pair of numbers
shows how much of a 98% is the hand and how much is the frame. Written
2026-09-17; the data is already at `data/arasl/` (zip + labels, 66 MB).

Prerequisite reading: `planning/gw_detection_demo.md` §4 (the host-loop
trainer that put the chapter CNN on a new input with zero codegen —
`demos/MainGwDetect.lean` is the file to copy), `planning/neu_det_fpn_demo.md`
(the two-tables shape and its §11 log for how a plan meets its data), and
`demos/README.md` for the section style: picture, table, the problem
explained; process stays in `runs/`.

## 0. The one-paragraph version

ArASL (Latif, Mohammad, Alghazo, AlKhalaf & AlKhalaf, *Data in Brief* 23,
103777, 2019; Mendeley Data doi 10.17632/y7pckrw6z2.1, CC BY 4.0) is 54,049
grayscale 64×64 images of static hand signs for the 32 letters of the Arabic
alphabet, performed by "more than 40 people", 1,293–2,114 images per class.
The published classifiers report 95–99.6% on random splits — CNNs at
96.6–97.6%, transfer-learned EfficientNets and ViTs at 99.3–99.6%. The
images were captured as bursts: 73% of consecutively numbered files differ by
under 6 grey levels per pixel (a random pair differs by ~33), and the
54,049 images fall into 14,335 chains of near-duplicate frames. No signer
identity is recorded, so the closest honest protocol is a split by capture
order within each class. The demo is the chapter-4 CNN under both splits,
with the chapter's linear and MLP rungs beside it as the bracket, scored
with a Wilson interval, a per-class breakdown and a leak audit that counts,
for every test image, whether a near-duplicate sits in train. Table 1 is the
chapter ladder × the two splits beside the published rows; Table 2 is the
per-class result on the honest split and the letter pairs that confuse. The
claim is the gap between the two columns and the audit that explains it.

## 1. Why this and not another people demo

- Nothing in the codegen moves. The GW demo already ran this exact net on a
  2×64×128 input with a 2-way head through the standard train step; ArASL is
  a 1×64×64 input with a 32-way head. Two numbers change in a spec.
- It is the dataset the Arabic sign-language literature reports on, with a
  clean licence (CC BY 4.0) and a plain-HTTPS download — the first
  people dataset in the book that needs no account and redistributes freely.
- The lesson is one only two splits of the same data can teach, and it is
  the people-watching lesson in general: a classifier of people generalises
  to *new* people or it does not, and a random split of burst frames cannot
  tell. The literature's protocol is measured here, beside the honest one,
  by the same net.
- It is cheap: 66 MB, ~12 s per epoch for the chapter net on one card, the
  whole ladder in under an hour, and the figure is 32 hands.

## 2. The data

Mendeley Data, dataset `y7pckrw6z2` version 1 (published 2018-11-05):
`ArASL_Database_54K_Final.zip` (66 MB, `ArASL_Database_54K_Final/<class>/
<CLASS> (<n>).jpg`) and `ArSL_Data_Labels.csv` (`#, File_Name, Class`).
Direct URLs are `https://data.mendeley.com/public-files/datasets/y7pckrw6z2/
files/<uuid>/file_downloaded`; the uuids come from the public API
(`…/public-api/datasets/y7pckrw6z2/files?folder_id=root&version=1`), so
`download_arasl.sh` resolves them at run time rather than hardcoding.
Institutions: Prince Mohammad Bin Fahd University (Al Khobar) and Universiti
Malaysia Sarawak. The 32 folder names, which are the class labels:

    ain al aleff bb dal dha dhad fa gaaf ghain ha haa jeem kaaf khaa la
    laam meem nun ra saad seen sheen ta taa thaa thal toot waw ya yaa zay

Census, measured 2026-09-17 on the download (`Gate 0` reproduces it):

- 54,049 files, 32 classes, per class 1,293 (`yaa`) to 2,114 (`ain`), a
  1.6× imbalance — mild; report per class, do not reweight.
- 53,391 are 64×64 grayscale. **648 are not**: 638 at 256×256 (`kaaf` 320,
  `meem` 318), 10 at 768×1024 (`haa`), and 10 are 64×64 RGB. The
  preprocessor resizes everything to 64×64 with one resampler and converts
  to L; the census prints the offenders so a mirror that differs is caught.
- **The images are bursts.** Sorting each class by the number in its file
  name, the median mean-absolute difference between consecutive frames is
  3.4 grey levels (random pairs within a class: 33); 73% of consecutive
  pairs are under 6, and cutting where the difference exceeds 6 gives
  14,335 chains — 3.8 frames per chain on average, the longest 58. The
  numbering is capture order, and no participant, session or date is
  recorded anywhere in the release. ⚠ Verify the chain statistic holds in
  every class before trusting the blocked split (it did: 64–78% per class).

`preprocess_arasl.py data/arasl data/arasl` writes, per split protocol
`{random,blocked}` and per part `{train,val,test}`, the GW demo's format:
`<protocol>_<part>.bin` as flat f32 `[N, 1, 64, 64]` in [0,1] (⚠ or the
chapter's normalisation — match whatever `apps/cifar/MainCifar8WideAblation`
feeds, so the net sees the range it was designed for), `labels_<protocol>_
<part>.bin` as int32, and `meta_<protocol>.npz` with each image's class,
file index, chain id and part, so the scorer and the audit never re-derive a
split. At 64×64 f32 the whole set is 885 MB; the trainer reads a part into
one ByteArray and gathers batches by index, exactly as `MainGwDetect` does.

**The two splits**, both 80/10/10 per class, both from one seed:

- `random`: a stratified permutation of images — the literature's protocol,
  and the plumbing check.
- `blocked`: within each class, the images in file-number order, cut at the
  80% and 90% marks — train is the first 80% of the capture order, val the
  next 10%, test the last 10%. Contiguous rather than chain-shuffled because
  sessions and signers are presumably contiguous in the numbering too, and
  a contiguous cut separates them as far as the release allows. ⚠ A
  contiguous cut can land inside a chain; the cut is moved to the nearest
  chain boundary so no chain straddles two parts, and the audit confirms it.

The **leak audit** is part of the preprocessor's `--stats` and the scorer:
for every test image, the minimum mean-absolute difference to any train
image of any class, at 16×16 (a 5,400 × 43,000 distance matrix in chunks,
seconds in numpy), and the fraction under 6. Expectation: tens of percent
under `random`, near zero under `blocked`. That fraction is the number the
section's gap is explained by, and it is printed before any training.

## 3. What is and is not "just a dataloader"

1. **The split is the demo**, not the loader (§2). Everything else is the
   GW demo's trainer with different constants.
2. **Stem and head.** Chapter 4's net is `cifar8w`: eight 3×3 convs in four
   conv-conv-pool stages at 16/16/32/32 channels into 512-512-out
   (`demos/MainGwDetect.lean` spells it out; the chapter's own is
   `apps/cifar/MainCifar8WideAblation.lean`). On a 1×64×64 input the flatten
   is 32·4·4 = 512 and the head is `dense 512 32`. Those are the only edits
   — the same two the GW demo made — and the plan holds the net at 64×64
   native. The chapter net's own input is 32×32; a `size=32` arm (Hu et
   al. 2022 trained at 32×32 and reported 95%) is a one-flag ablation, not
   a phase.
3. **The chapter ladder.** `mnist-linear` and `mnist-mlp` are the chapter 1–3
   rungs; on 64×64 they are `dense 4096 32` and `dense 4096 512 → 512 → 32`,
   and the same host loop trains them (the train step is the ordinary
   `generateTrainStep`, which already serves dense-only specs). They are the
   bracket that says what a convolution buys on hands, and they are free.
4. **Class count and normalisation** are the two places a wrong constant
   reads as a low number rather than a crash (a 10-way head on 32 classes
   throws at the label gather; a [0,255] input into a net designed for
   [0,1] or CIFAR-normalised trains, slowly). Print both at start.
5. **No bootstrap.** The chapter net trains from He init on CIFAR and does
   here; an ImageNet-R34 arm (the BraTS/VisDrone/NEU bootstrap) is an
   optional upper row (§4), not the demo.

The Lean side is one file, `demos/MainAraslSigns.lean` (`lake exe
arasl-signs [net=cifar8w|mlp|linear] [split=blocked|random] [epochs=30]
[batch=64] [lr=0.001] [seed=1] [size=64] [tag=] [eval]`), copied from
`MainGwDetect.lean` with the specs swapped, the two data files per split and
`scoreSet` writing `[N, 32]` logits for the test part. Recipe: the GW
demo's — Adam 1e-3, one-epoch linear warmup then cosine, weight decay 1e-4,
batch 64, label smoothing 0 — unless the chapter-4 ablation's own recipe
transfers unchanged, in which case use that and say so.

## 4. The arms

Table 1, ArASL, 80/10/10 per class, test accuracy with a 95% Wilson interval
(5,400 test images → about ±0.5 points):

| arm | random split | blocked split | what it tests |
|---|---|---|---|
| chapter 4 CIFAR-CNN8-wide-BN, 64×64, 30 ep | | | the demo |
| chapter 2–3 MLP 4096-512-512-32 | | | what convolutions buy on hands |
| chapter 1 linear 4096-32 | | | the floor |
| CNN8-wide at 32×32 (`size=32`) | | | optional: the chapter's native input |
| ImageNet R34, bootstrap, 224 upsample | | | optional: what pretraining buys, if the blocked column is low |
| published CNNs, random split | 96.6–97.6 | — | Alani & Cosma 2021 (ArSL-CNN, 96.59); Latif et al. 2020 (~97.6); Abdelghfar et al. 2023 (QSLRS-CNN, 97.31) |
| published transfer / transformer, random split | 99.3–99.6 | — | Al Nabih et al. 2024 (ViT, 99.3); EfficientNet-B2 at 224 px (99.48, arXiv:2501.08169, 2025); arXiv:2410.00681 (2024, up to 99.6) |

The random column of the chapter net beside the published CNN rows is
Gate 1; the blocked column is the result, and it may be anything from a
few points down to a collapse. Both go in the table as they land. The ratio
of the two columns, not either alone, is the sentence.

Table 2, the blocked split, per class: accuracy and the five most confused
letter pairs (predicted ↔ true, counts). Expectation from the alphabet: the
pairs that differ by finger position rather than hand shape — `ta`/`taa`,
`dal`/`thal`, `ra`/`zay`, `saad`/`dhad`, `seen`/`sheen`, `ha`/`haa`/`khaa` —
and the audit says whether a confused pair is also a leaked one.

Epochs: the chapter net at 43k images and batch 64 is 675 steps per epoch,
~12 s; 30 epochs is 6 minutes. Choose the epoch on val, report test at it;
run seeds 1–3 on the two chapter-net arms since three runs cost 40 minutes
and the book's convention is a mean with an interval where one is cheap.

## 5. The instrument

`scripts/arasl_score.py <logits.bin> data/arasl/labels_<protocol>_test.bin
--meta data/arasl/meta_<protocol>.npz [--train-bin …]`: accuracy with Wilson
interval, per-class accuracy, the confusion matrix and its top pairs, and
the leak audit from §2 (test → nearest train image, fraction under 6 grey
levels) — the audit needs the train part's images, so it takes the path and
caches the 16×16 thumbnails. One scorer, both protocols, so the two columns
of Table 1 are one code path with one constant changed.

⚠ Accuracy is the field's metric here and the only one the published rows
give; do not add a macro-F1 column unless the per-class spread makes the
mean misleading, and if it does, say so beside the table, do not swap it in.

## 6. Figure and section

Figure: (a) the alphabet — one crop per class, 32 tiles with the letter
label, the data article's own `Signs_32_New.png` in the demo's own crops;
(b) the five most confused pairs on the blocked split, a true and a
predicted crop each, side by side. `scripts/arasl_figure.py`, 2 rows. If
the leak audit is the story, a third panel: a test image and its nearest
train neighbour under each split — the same hand twice under `random`, a
different hand under `blocked`. A preview of (a) plus the burst-vs-random
strip exists already: `runs/2026-09-17-arasl-preview/{preview_figure.py,
arasl_preview.png}` (untracked), the template for the real script the way
`scripts/mock_gw_figure.py` (now deleted) was for the GW figure — the user's reaction to
it was "if we had that pic for real that would look great", so build (a)
to that layout.

Section: *People watching — demo: Arabic sign-language letters on ArASL*, a
`\subsection` in the bestiary. Placement: after *Industrial inspection* —
the two demos are the same move, a chapter artifact carried unchanged to a
domain it was not built for — or wherever the people-watching family gets
its home if it grows. Shape, learned on NEU: two lead paragraphs (the data
and the two splits; what changes in the net and what does not), Table 1,
Table 2 or the pairs, the figure, one closing paragraph on what the gap
means and one sentence on what a static-handshape classifier is not (§9).
No history, no gates, no plan in the book — that lives in `runs/<date>-
arasl-*/README.md` and §11 of this file. Data appendix: an ArASL row in book
order, and a "Building the ArASL dataset." entry with the licence, the
census, the burst structure and the two splits.

## 7. Phases

```
Phase 0 (½ session, CPU):   download_arasl.sh (API-resolved URLs), preprocess_arasl.py
                            with --stats: census, chain statistic, both splits, leak audit
                            Gate 0: 54,049 files / 32 names exactly / 648 non-64 listed;
                                    chain fraction ≥ 0.6 in every class; leak audit
                                    printed for both protocols with random ≫ blocked
Phase 1 (½ session, GPU):   demos/MainAraslSigns.lean from MainGwDetect; cifar8w on `random`
                            Gate 1: test accuracy ≥ 96% (the published CNN rows; a wrong
                                    head, label offset or normalisation reads as < 90)
Phase 2 (½ session, GPU):   cifar8w on `blocked`; MLP and linear on both; seeds 1–3 on
                            the chapter net
                            Gate 2: the blocked column is scored by the same scorer at the
                                    same epoch rule, and the audit's leak fraction is in
                                    the runs README beside it, whatever the numbers are
Phase 3 (½ session):        Table 2 (per class, confused pairs), the figure, the section,
                            the data-appendix row and entry, demos/README.md
Optional:                   size=32 arm; the ImageNet-R34 bootstrap arm if blocked is low
```

## 8. Gates that fail loudly

- Gate 0's chain statistic is the premise. If consecutive files are NOT
  near-duplicates in some class, the numbering is not capture order there
  and the blocked split is just a different random split for that class —
  the plan is void for that class and the audit will show it.
- Gate 1 is the plumbing: class ids in the folder → label map (alphabetical,
  as the CSV lists them), the normalisation, the flatten width. Each fails
  as a low number; the per-class printout tells them apart (one class at
  0 = label map; all classes at ~3% = head or labels shifted; slow training
  = input range).
- The leak audit must say what the split says: `blocked` with a leak
  fraction above a few percent means a chain straddles the cut, and the
  preprocessor's boundary snap is wrong.
- A blocked column ABOVE the random one is not an error to hide; it is a
  result about the data and belongs in the table.

## 9. Out of scope

- Signer-independent evaluation proper: the release records no signer,
  session or date; the blocked split is the closest proxy and the section
  says so in one sentence.
- Sign *language*: this is fingerspelling of static handshapes, one letter
  per image, no motion, no grammar, no continuous signing. The section's
  one sentence on what the classifier is not.
- Other sign datasets (ASL alphabet on Kaggle, AASL, KuSL2023): the same
  demo would run on any of them; ArASL is the one with the licence, the
  citation trail and the published bracket.
- Hand-keypoint pipelines (MediaPipe-style landmarks → classifier): a
  different method, not a comparison arm for a pixel CNN.
- Deployment (the Orin path) and the transfer-learning arms beyond the one
  optional R34 row.

## 10. Notes before starting

- The trainer is `lake exe arasl-signs …`; nothing here is a `lake run`
  job. Runs are minutes; nothing needs asking about.
- ⛔ Size the packed parameter buffer from `heInitParams`, not
  `spec.totalParams` (the GW file's comment; irrelevant for this net, which
  has no SE, but keep the line).
- XLA backend only, as `gw-detect`; the standard train step with int32
  labels — no `DatasetKind`, no new FFI.
- The zip has 10 RGB files and 648 non-64×64 files; the preprocessor
  converts and resizes, and `--stats` prints the census so a mirror that
  differs is caught before it trains.
- `data/arasl/` already holds the zip, the labels CSV and the extracted
  tree from 2026-09-17; `download_arasl.sh` must be idempotent over it.
- The book's accuracy convention is a Wilson interval on every number;
  `scripts/arasl_score.py` prints it, and Table 1 carries it.

## 11. Log — how the plan met its data (2026-09-17)

- **Gate 0 held on every line.** Census exact (54,049 / 32 names / numbered
  1..n without a gap in every class / 638 at 256² + 10 at 768×1024 + 10 RGB).
  Chain statistic 73.5% of consecutive pairs under 6 grey levels, 14,336
  chains of 3.8 frames (longest 58), weakest class `al` at 63.9%. Random
  pair within a class: 33. `runs/2026-09-17-arasl/gate0.log`.
- **The leak audit's two numbers are 92.3% and 6.4%** (16×16; 88.3% / 3.7%
  at 64×64). The blocked residual is NOT a straddled chain (the check
  passed): 91% of the flagged pairs are same-class, a median **306 files
  apart** — the same hand returning in a later sitting, which a release with
  no signer ids cannot separate. §2's "near zero" expectation was wrong by
  that much; the section reports it as the protocol's floor. The
  preprocessor now also stores the 64×64 distance of the chosen neighbour.
- **The val tenth is 37.5% leaked** under `blocked` (it sits next to train in
  capture order), the test tenth 6.4%; that is the val 92% / test 79%
  asymmetry in every blocked log, and it means the epoch is chosen on a
  partly-leaked set. Stated, not fixed: a val slice further from train would
  have to come out of test.
- **Two rows the plan did not list, both free from the audit:** accuracy on
  the leaked vs the non-leaked test images (random: 99.7% vs 84.1%), and the
  1-NN-on-16×16-thumbnails floor (random 95.7%, blocked 28.6%) — a
  zero-parameter lookup that gets within three points of the published CNNs
  on the literature's protocol. Both are in `scripts/arasl_score.py`.
- **Gate 1 met at epoch 5** (96.78%); seed 1 finished at 98.52% random /
  78.78% blocked, best-val epochs 23 / 24. ~9 s per epoch with three cards
  busy (8 ms/step alone), 30 epochs ≈ 5–6 min.
- **Confused pairs are not §4's guesses.** Blocked, seed 1: `taa`↔`thaa`
  (82), `kaaf`↔`thaa` (65), `fa`→`waw` (60, one-directional), `gaaf`→`dhad`
  (44), `kaaf`→`seen` (43). Per-class spread 22.7% (`fa`) to 100%; distance
  to the nearest training image explains about half of it (Spearman −0.51
  between class accuracy and median nearest-train |Δ|) — the rest is which
  letter the new hand's shape resembles.
- **`--size 32` reuses the 64×64 chains and split** (first cut recomputed
  them on the downsampled images, a different blocked split). The 32 arm is
  the same split downsampled, as it must be to sit in the same column.
- Recipe carried from the GW demo unchanged (Adam 1e-3, one-epoch warmup +
  cosine, wd 1e-4, batch 64, ls 0); the chapter's own CIFAR ablation runs a
  constant lr with hflip, which would be wrong here (a mirrored hand is a
  different sign for some letters).
- Files: `download_arasl.sh`, `preprocess_arasl.py`, `demos/MainAraslSigns.lean`
  (`lake exe arasl-signs`), `scripts/arasl_score.py`, `scripts/arasl_figure.py`;
  runs under `runs/2026-09-17-arasl*/`.
- **DONE 2026-09-17, Table 1 as landed** (test, 3 seeds for the chapter net,
  pooled Wilson ±0.2 / ±0.6): cifar8w **98.62 ± 0.10 / 77.94 ± 0.80**; on the
  leaked test images 99.73 / 98.46, on the rest 85.26 / 76.53; size=32 98.21 /
  71.03; MLP 94.86 / 42.09; linear 53.20 / 15.37; 1-NN 95.65 / 28.60. Blocked
  per class 16.5% (`fa`) – 100% (`sheen`); pooled pairs `fa`→`gaaf` 181:1,
  `kaaf`↔`thaa`, `taa`↔`thaa`, `gaaf`→`dhad` 117:0, `fa`→`waw` 113:0. Section
  in the book after Industrial inspection (`sec:bestiary_people`), data
  appendix (bullet, row, "Building the ArASL dataset."), demos/README, figure
  `demos/figures/arasl_signs.png`. Optional R34-bootstrap arm NOT run (the
  blocked column is a result, not a failure to fix).
