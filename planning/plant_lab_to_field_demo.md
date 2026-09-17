# plant_lab_to_field_demo.md — chapter 6's ResNet-34 from lab leaves to field leaves

Goal: an *Agriculture* entry for the bestiary, and the book's first use of a
class-activation map as a debugger rather than a decoration. Fine-tune chapter
6's ResNet-34 (ImageNet prefix bootstrap, the BraTS/VisDrone/NEU move) on
PlantVillage — 54k lab photographs of single picked leaves on a grey background,
38 classes, the most-cited dataset in the user's agriculture corpus — to the
~99% every one of those papers reports; evaluate the same weights on PlantDoc,
2.6k field photographs of the same crops and diseases, and watch it fall to a
third; show with the CAM and with a leaf mask that the 99% was partly the
photography; then close part of the gap three ways, each measured on the same
2,578 field images. Written 2026-09-17; data cloned the same day. Four acts,
one net, one scorer; the edge (Orin, a Pi) is a later session — user
decision 2026-09-17, software first.

Prerequisite reading: `planning/arasl_people_watching_demo.md` §2 and §11 (the
leak audit and the two-splits instrument — both run here unchanged on
PlantVillage, and §11 is the shape of a plan meeting its data),
`demos/MainAraslSigns.lean` (the host loop this trainer copies),
`demos/MainUnetBratsR34.lean` §"bootstrap" (`patchInitWithPretrainedPrefix` /
`Range` on `jax_r34_imagenet.bin`), `demos/probes/MainGradCAM.lean` (the
closed-form CAM for GAP+dense nets, already wired for an Imagenette R34),
and `demos/README.md` §Industrial inspection for the section shape.

## 0. The one-paragraph version

PlantVillage (Hughes & Salathé 2015, arXiv:1511.08060; Mohanty, Hughes &
Salathé 2016, *Front. Plant Sci.* 7:1419) is 54,305 colour images of single
leaves, 14 crops × (diseases + healthy) = 38 classes, every leaf picked, laid
on a uniform grey or paper background and photographed in a lab; it ships a
`segmented` twin with the background blacked out. Mohanty et al. reported
99.35% with GoogLeNet on a random 80/20 split — and, in the same paper, ~31%
on images taken under other conditions. PlantDoc (Singh et al. 2020,
CODS-COMAD; github `pratikkayal/PlantDoc-Dataset`) is 2,578 field photographs
scraped from the web, 13 of PlantVillage's 14 crops, 27 classes that all map
into PlantVillage's 38: leaves on plants, several per frame, soil and sky
behind them, phone lighting. Noyan (2022, arXiv:2206.04374) showed that a
classifier given only PlantVillage's *backgrounds* — leaf removed — scores far
above chance on the 38 classes. The demo is: (1) the 99% on PlantVillage,
with the ArASL leak audit run on it first; (2) the same weights on PlantDoc,
the collapse; (3) the CAM on PlantVillage test images with the fraction of
class-evidence mass that lands *outside* the leaf mask, and the
background-only accuracy, as the numbers that predicted (2); (4) three
fixes — leaves composited onto other backgrounds, stronger augmentation,
and a few hundred field labels — each scored on the same 2,578 PlantDoc
images, the gap closing partway and not fully. Table 1 is the four-row fixes table with
the lab column beside the field column; Table 2 is the CAM-mass and
background-only numbers; the figure is a lab leaf, a field leaf, and the
CAM on each before and after the background fix.

## 1. Why this and not another ag demo

- Nothing in the codegen moves. ResNet-34 with `.dense 512 38` from the
  ImageNet prefix is the NEU/VisDrone/BraTS backbone with a classifier head;
  the trainer is `MainAraslSigns.lean` reading the Imagenette record format
  (`F32.loadImagenetteSized`), which chapter 6 already trains from.
- PlantVillage is the corpus's most-used dataset (25 papers using it, 31
  naming it, ahead of BraTS) and the most famously flawed one in applied
  vision, and the flaw *is* the curriculum: a random split of lab photographs
  says nothing about a field. It is the ArASL lesson (same hand on both sides
  of the line) with the leak moved from the subject to the background, and
  the fix acts are what the ArASL section could not offer.
- Both datasets are plain-HTTPS `git clone`s with no account, and PlantDoc's
  classes were designed to overlap PlantVillage's, so lab→field is one
  mapping table, not a reconciliation.
- It is the first place the book *uses* a CAM. `lake exe gradcam` exists with
  figures for R34 and ConvNeXt-T, and the book cites neither (0 mentions of
  Grad-CAM in `content.tex` as of 2026-09-17). Here the CAM produces a number
  that a later act tests.
- Cost: ~3 min per PlantVillage epoch on one 4060 Ti (chapter 6's R34 runs
  Imagenette at ~270 images/s), a fine-tune is 10 epochs, the whole ladder
  including seeds is an evening on one card. ⚠ Still ask before the ladder
  (`user_runtime_prefs`): the individual runs are 30 min, not 5.

## 2. The data

**PlantVillage** — `git clone --depth 1
https://github.com/spMohanty/PlantVillage-Dataset` (4.8 GB on disk; `raw/color`,
`raw/grayscale`, `raw/segmented`, 38 folders each; every image 256×256 RGB).
Licence: no LICENSE file in the clone; `README_HF.md`'s dataset card says
**CC BY-SA 3.0**, and `CITATION.cff` asks for Mohanty et al. 2016 — the
appendix states both. ⛔ Do NOT use Kaggle's "New Plant Diseases Dataset"
(87k images): PlantVillage with augmented copies, the same image on both
sides of its split. The 38 folder names, as cloned (Gate 0 checks them
byte-for-byte; note `Cherry_(including_sour)`, `Corn_(maize)` and the
trailing underscore on `Common_rust_`):

    Apple___{Apple_scab, Black_rot, Cedar_apple_rust, healthy}   Blueberry___healthy
    Cherry_(including_sour)___{Powdery_mildew, healthy}
    Corn_(maize)___{Cercospora_leaf_spot Gray_leaf_spot, Common_rust_, Northern_Leaf_Blight, healthy}
    Grape___{Black_rot, Esca_(Black_Measles), Leaf_blight_(Isariopsis_Leaf_Spot), healthy}
    Orange___Haunglongbing_(Citrus_greening)   Peach___{Bacterial_spot, healthy}
    Pepper,_bell___{Bacterial_spot, healthy}   Potato___{Early_blight, Late_blight, healthy}
    Raspberry___healthy   Soybean___healthy   Squash___Powdery_mildew   Strawberry___{Leaf_scorch, healthy}
    Tomato___{Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot,
              Spider_mites Two-spotted_spider_mite, Target_Spot, Tomato_Yellow_Leaf_Curl_Virus,
              Tomato_mosaic_virus, healthy}

Census (2026-09-17, the clone): 54,305 colour files (54,303 .jpg, 1 .png, 1
.jpeg), per class 152 (`Potato___healthy`) to 5,507 (`Orange___Haunglongbing`),
a 36× imbalance — report per class, do not reweight, and say the mean is
tomato-heavy (10 of 38 classes, 18,160 images). `segmented` has 54,306 files:
one orphan in `Grape___Esca`, and four colour files (two Peach, one
Strawberry, one Tomato spider-mites) whose twin is spelled differently —
dropped from the mask parts and printed. ⚠ `Corn_(maize)___Common_rust_`'s
colour files carry NO uuid prefix (`RS_Rust 1563.JPG`) while their segmented
twins do; the twin is matched on the suffix after `___` (lower-cased), which
is also the leaf-map key.

**PlantVillage ships its own leaf grouping.** A file is
`<uuid>___<lab-code>_<name> <n>.JPG` (`FREC_Scab 3335`), the suffix being the
original camera file from one of ~20 labs (`RS`, `FREC`, `JR`, `FAM`,
`GCREC`, `UF.GRC`, `Com.G`, …, printed per class by Gate 0).
`leaf_grouping/leaf-map.json` maps 40,328 suffixes to a leaf id
(`"uf.citrus_hlb_lab 1654" → "Orange___…:::104.0"`, consecutive numbers →
the same leaf): **7,946 leaves, the mode is exactly 4 photographs per leaf
(5,811 leaves), max 33**, covering 41,111 of the 54,305 colour files; 13
classes have no entries at all (YLCV 2,619, Squash 1,835, Target spot 1,404,
the four Corn classes, Tomato late blight 989, …). `plant_village.py`, the
maintainers' Hugging Face loader, gives an uncovered file a singleton
`fallback_<suffix>` id, and its `splits/color_{train,test}.txt` (fetched to
`data/plant/splits/`, 43,596 + 10,709 = 54,305) is an 80/20 split that
"strictly respects the leaf grouping to prevent data leakage" — added to the
dataset years after the 25 papers. So the two columns are: the random split
the literature used, and the maintainers' own grouped split; and the ArASL
audit (16×16 nearest neighbour, chains along the suffix numbers within
(class, lab)) says how much the fallback singletons in the 13 uncovered
classes still leak. Both numbers are printed before anything trains, and the
grouped column is the section's lab number.

**PlantDoc** — `git clone https://github.com/pratikkayal/PlantDoc-Dataset`
(1.9 GB on disk; `LICENSE.txt` = **CC BY 4.0**). `train/` 2,342 and `test/`
236 images in 28 class folders (the paper says 27 and 2,578: the 28th,
`Tomato two spotted spider mites leaf`, has 2 training images and no test
image, and 20 files are missing from the clone). 13 species. Web photographs
of every size (1600×1200 down to 300×300) and mode (a few RGBA/CMYK →
`convert("RGB")`); the preprocessor resizes the shorter side to 256 and
centre-crops 224 for evaluation. Several leaves per frame, the labelled
disease on at least one; the label is per image and the task stays
classification.

**The class map**, PlantDoc → PlantVillage, all 28 folders, spelled once in
`preprocess_plant.py` and printed by Gate 0:

| PlantDoc | PlantVillage | | PlantDoc | PlantVillage |
|---|---|---|---|---|
| Apple Scab Leaf | Apple___Apple_scab | | Raspberry leaf | Raspberry___healthy |
| Apple leaf | Apple___healthy | | Soyabean leaf | Soybean___healthy |
| Apple rust leaf | Apple___Cedar_apple_rust | | Squash Powdery mildew leaf | Squash___Powdery_mildew |
| Bell_pepper leaf | Pepper,_bell___healthy | | Strawberry leaf | Strawberry___healthy |
| Bell_pepper leaf spot | Pepper,_bell___Bacterial_spot | | Tomato Early blight leaf | Tomato___Early_blight |
| Blueberry leaf | Blueberry___healthy | | Tomato Septoria leaf spot | Tomato___Septoria_leaf_spot |
| Cherry leaf | Cherry_(including_sour)___healthy | | Tomato leaf | Tomato___healthy |
| Corn Gray leaf spot | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | | Tomato leaf bacterial spot | Tomato___Bacterial_spot |
| Corn leaf blight | Corn_(maize)___Northern_Leaf_Blight | | Tomato leaf late blight | Tomato___Late_blight |
| Corn rust leaf | Corn_(maize)___Common_rust_ | | Tomato leaf mosaic virus | Tomato___Tomato_mosaic_virus |
| Peach leaf | Peach___healthy | | Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| Potato leaf early blight | Potato___Early_blight | | Tomato mold leaf | Tomato___Leaf_Mold |
| Potato leaf late blight | Potato___Late_blight | | Tomato two spotted spider mites leaf | Tomato___Spider_mites Two-spotted_spider_mite |
| grape leaf | Grape___healthy | | | |
| grape leaf black rot | Grape___Black_rot | | | |

The ten PlantVillage classes with no PlantDoc twin (Apple black rot, Cherry
powdery mildew, Corn healthy, Grape esca, Grape leaf blight, Orange HLB, Peach
bacterial spot, Potato healthy, Strawberry leaf scorch, Tomato target spot)
stay in the 38-way head; the scorer reports PlantDoc accuracy both ways —
38-way argmax (a prediction of an unmapped class is wrong) and argmax
restricted to the 28 mapped classes — and the section quotes the restricted
one, since the published PlantDoc numbers are 27-way.

`preprocess_plant.py <pv_dir> <plantdoc_dir> data/plant` writes the Imagenette
record format (`count` header, then `label` byte + 3×S×S u8, channel-planar) at
S = 256 for training parts and 224 for evaluation parts, so
`F32.loadImagenetteSized` and the C batch helpers read them unchanged:

- `pv_{train,val,test}.bin` — PlantVillage `color` under the random split,
  80/10/10 per class, one seed; `pvg_{train,val,test}.bin` under the grouped
  split: test = the maintainers' `color_test.txt` exactly (10,709), val = one
  tenth of the whole carved out of their train by leaf id, train = the rest;
- `pv{,g}_test_seg.bin` — the same test images from `segmented` (leaf on
  black); `pv{,g}_test_bg.bin` — the complement: `color` with the leaf region
  filled with the image's median background colour (⚠ the silhouette still
  carries the leaf's shape, which is class information — say so; the CAM-mass
  number is the primary evidence, this is the corroboration);
  `pv{,g}_test_mask.npy` — the leaf mask at 7×7 (the CAM's grid) and 224×224;
  `pv{,g}_test_{leaf,none}.bin` — the two-player Shapley's other counterfactuals:
  the leaf on the median fill, and the flat fill alone (`_test_bg` is the third);
- `pv_train_comp.bin` — Act 4's composites: every training leaf (mask from
  `segmented`, 2-px feathered edge) pasted at its own position onto a random
  224 crop of a random Imagenette training image (`data/imagenette/train.bin`,
  already in the repo, no agriculture in it, so nothing field-like leaks in),
  one composite per leaf, background drawn from a seed;
- `pd_all.bin` (2,578, eval), `pd_train.bin` (2,342 at 256), `pd_test.bin`
  (236), `pd_fold{0..4}_{train,test}.bin` — five folds over all 2,578
  stratified by class, so the fine-tune arm scores every field image once
  held-out and the whole of Table 1 has the same 2,578 denominator;
- `meta_*.npz` with the class map, the PlantVillage suffix numbers and lab
  codes, the audit, and the fold ids; `classes_pv.txt`, `classes_pd.txt`.

Sizes: PlantVillage at 256 u8 is 54,305 × 196,608 B = 10.7 GB, at 224 8.2 GB;
with the composite and augmented twins of the training part the directory
is ~35 GB (114 GB free). The loader reads a part whole. PlantDoc is small.

## 3. What is and is not "just a dataloader"

1. **The bootstrap is the demo's premise** (Act 1 is "what the median paper
   does"). `p0 ← spec.heInitParams` then `patchInitWithPretrainedPrefix p0
   ".lake/build/jax_r34_imagenet.bin"` — the 21,284,672-float prefix is every
   conv and BN of the R34 and the ImageNet dense head is NOT in the prefix, so
   the `dense 512 38` stays He-initialised; the NEU detector's
   `bootstrapBackbone := some (path, 21284672)` is the same call. ⚠ The
   BN running statistics are not in that file (memory: "BN stats are in the
   .state.npz not the .bin" for R50-A3; check the R34 export the same way) —
   the fine-tune recomputes them from step 1, which is fine for 10 epochs and
   must be said in the runs README.
2. **The head and the class count**: `.dense 512 38 .identity`; the label
   check at start is the ArASL one (max label + 1 = 38).
3. **Augmentation is what the C helper gives** — random 224 crop from 256 and
   horizontal flip, the Imagenette recipe — for every arm but Act 4's second
   fix, which is stronger photometric augmentation. ⚠ Decision: the shim
   (a Python producer feeding batches over a pipe, `Train.lean`'s ImageNet
   path) can do any augmentation but drags in the TFDS layout; the cheap
   route is to PRECOMPUTE the stronger augmentation as extra training records
   (colour jitter, ±30° rotation, random-erasing, 2 variants per image →
   `pv_train_aug.bin`) and train the arm on the union. Precompute. It is
   honest (the section says the variants are fixed per epoch) and needs no
   runtime code.
4. **Evaluation at 224 from a 256 shorter side, centre crop** for both
   datasets, one code path, so a PlantDoc number and a PlantVillage number are
   the same function of an image.
5. **The CAM is closed-form** (`F32.camCompute`: the dense row for the target
   class weights the 512×7×7 pre-GAP map; no autodiff), and the probe already
   compiles a `forward_cam` graph for an R34 spec. What is new: a `plant`
   entry in `pickModel` reading this demo's checkpoint, a raw dump of the
   7×7 CAM per image beside the PPM strip, and `scripts/plant_cam.py` that
   reads the dump and the 7×7 leaf mask and reports the mass fraction inside
   the leaf (CAM normalised to sum 1, ReLU'd first, as Grad-CAM does).
6. **Shapley values, exact where they can be.** With two players — the leaf
   region and the background, from the same mask — the Shapley value of each
   for the class logit is exact in four forward passes: full, leaf only,
   background only, neither (the "removed" region filled with the image's
   median background colour, the same fill for all four), and
   φ_leaf + φ_bg = f(full) − f(neither) holds to the float. That runs on every
   test image from the eval graph alone (`pv{,g}_test_{leaf,bg,none}.bin`
   beside `_test`), model-agnostic, and asks the CAM's question a second way:
   what share of the evidence is the leaf. For the figure, a sampled Shapley
   heatmap on the CAM's 7×7 grid for the same few leaves — permutation
   sampling over the 49 patches with a standard error, a few thousand forwards
   per image written as one probe part and scored by `plant-leaf eval
   extra=<file>` — next to the CAM, so the reader sees a closed-form and a
   model-agnostic explainer agree, or not. The book has no Shapley content;
   the definition (marginal contributions averaged over orderings), the
   efficiency axiom and the Monte-Carlo estimate are the four sentences this
   section adds, beside the CAM's four.
7. **No PlantDoc image touches training** in Acts 1–3 and in Act 4's first
   two rows; only the fifth-fold fine-tune sees field images, and never the
   ones it is scored on. The scorer asserts it from the fold ids.

The Lean side is one file, `demos/MainPlantLeaf.lean` (`lake exe plant-leaf
[arm=base|comp|aug|field] [split=random|blocked] [epochs=10] [batch=64]
[lr=0.001] [seed=1] [fold=0..4] [init=imagenet|scratch] [tag=] [out=] [eval]`),
copied from `MainAraslSigns.lean` with `loadImagenetteSized` in place of the
flat-f32 loader, the prefix bootstrap, and `scoreSet` run over every
evaluation part it is given (`pv_test`, `pv_test_seg`, `pv_test_bg`, `pd_all`
or the fold's test), writing `[N, 38]` logits for each. `arm=field` starts
from a saved `base` checkpoint (`init=<prefix>`) and continues on
`pd_fold{k}_train` for a few epochs at a lower lr. Recipe: the ArASL/GW one —
Adam 1e-3, one-epoch warmup, cosine, wd 1e-4, batch 64, ls 0 — with lr 1e-4
for the field fine-tune; 10 epochs on PlantVillage (the 99% arrives by epoch
3; the schedule is short on purpose).

## 4. The acts, as arms and tables

**Table 1 — the fixes, scored on the same 2,578 PlantDoc images (restricted
argmax, Wilson), with the PlantVillage test column beside them:**

| arm | PlantVillage test | PlantDoc, all 2,578 | what it tests |
|---|---|---|---|
| Act 1: R34 ImageNet-bootstrap, 10 ep, 3 seeds | ~99 | | the median paper, and the collapse |
| R34 from scratch, 10 ep | | | what ImageNet is worth in the field |
| Act 4a: + leaves on Imagenette backgrounds (`pv_train_comp` ∪ `pv_train`) | | | the CAM's diagnosis, attacked directly |
| Act 4b: + stronger augmentation (`pv_train_aug` ∪ `pv_train`) | | | the modern-recipe angle |
| Act 4c: + fine-tune on ~250 field labels (5-fold, ~10 per class) | — | | the "50-label" move |
| Act 4c′: + fine-tune on all field labels (5-fold, 2,078 per fold) | — | | the ceiling the field labels buy |
| Act 4a + 4c′ | — | | do the fixes stack |
| published, PlantVillage random split | 99.35 (Mohanty 2016, GoogLeNet); 99.5–99.8 (transfer, 2018–23) | — | |
| published, PlantVillage-trained on other conditions | — | ~31 (Mohanty 2016, ⚠ verify the figure) | |
| published, PlantDoc-trained on PlantDoc test | — | ⚠ from Singh et al. 2020 | 27-way, 236 images |

The first row's field column is Act 2. The PlantDoc paper's own
PlantVillage→PlantDoc number, if it gives one, goes in the table with its
denominator; the section says the 236-image official test split has a ±6-point
interval and that is why every row here is scored on 2,578.

**Table 2 — the diagnosis (Act 3), on the PlantVillage test tenth, 3 seeds:**

| | base | + backgrounds (4a) | + field labels (4c′) |
|---|---|---|---|
| CAM mass inside the leaf mask (mean over images, and the fraction of images with < 50% inside) | | | |
| Shapley share of the class logit attributable to the leaf, exact 2-player (mean; fraction of images < 50%) | | | |
| accuracy on `pv_test_seg` (leaf only, black background) | | | |
| accuracy on `pv_test_bg` (background only, leaf blacked out) — chance is 1/38 = 2.6% | | | |
| PlantDoc (from Table 1) | | | |

The claim of the section is the diagonal: the base row's background-only
accuracy and outside-leaf CAM mass are high, the background fix drops both,
and its PlantDoc number rises; if the CAM number moves and PlantDoc does not,
the section says so — the instrument was tested, not vindicated. Noyan's
background-only figure (⚠ ~49% on the full 38 classes; verify against the
paper) is the published row.

Seeds: three on Act 1 and on whichever Act 4 row is the section's headline;
one elsewhere. Per-class PlantDoc accuracy in the runs README; in the section
only the sentence about which crops survive the move (expectation: tomato and
potato, where the lesions are large; corn and grape collapse).

## 5. The instrument

`scripts/plant_score.py <logits.bin> --part pv_test|pv_test_seg|pv_test_bg|pd_all|pd_fold<k>
[--restrict] [--json]`: accuracy with Wilson, 38-way or restricted to the
mapped classes for a PlantDoc part, per class, the confusion matrix and its
top pairs, and for `pv_test_*` the leak audit read from `meta_pv.npz` as
ArASL's scorer does. `scripts/plant_cam.py <cam_dump.bin> data/plant/pv_test_mask.npy`:
the mass-inside-leaf statistic, per image and pooled, and the images at the
two extremes for the figure. `scripts/plant_shapley.py two-player <logits of
test, test_leaf, test_bg, test_none>`: φ_leaf, φ_bg per image for the true
(and the predicted) class logit, the efficiency check, the leaf share pooled
and per class; `plant_shapley.py grid --images … --perms 40` writes the probe
part of masked variants, and `plant_shapley.py grid --score <logits>` turns
the scored probe into 7×7 Shapley maps with standard errors for the figure. One scorer for both datasets and every arm; the
`--restrict` flag is the one constant that changes between the two columns.

⚠ Accuracy is the field's metric and the only one the published rows give;
PlantDoc's class imbalance (⚠ measure it) may make macro-F1 worth a sentence
beside the table, not a column.

## 6. Figure and section

Figure, three rows of 224 crops with the CAM (viridis, α-blended, the probe's
existing rendering; the mock is `runs/2026-09-17-plant/mock_figure.py`, which
the user liked — build to that layout): (a) four PlantVillage test leaves — a
tomato, a corn, an apple, a grape — input | base CAM | base Shapley (7×7,
sampled) | +backgrounds CAM, with the mass-inside-leaf and the leaf-share
number under each map; (b) the same four classes from PlantDoc, input | base
CAM | +field-labels CAM, with the predicted label under each; (c) one
composite from `pv_train_comp` beside its source, so the reader sees what the
fix trained on. `scripts/plant_figure.py`, from the probe's PPM/dump output.

Section: *Agriculture — demo: lab leaves to field leaves on PlantVillage and
PlantDoc*, a `\subsection` after *People watching* (the same move one step
further: the chapter net on the field's dataset, the number the field
reports, the number it should report, and then the fixes). Shape: two lead
paragraphs (the two datasets and the map; what changes in the net and what
does not — the bootstrap, the head, the precomputed augmentations), Table 1,
the figure, Table 2 and the paragraph that reads it, one closing paragraph on
what closes and what does not, one sentence on what a leaf classifier is not
(a diagnosis: no severity, no field-level incidence, no early symptom). No
acts, no gates, no plan in the book. Data appendix: two rows (PlantVillage,
PlantDoc) and one "Building the PlantVillage and PlantDoc datasets." entry —
the clone sizes, the licences as read, the census, the map, the audit, the
composites, the folds. The CAM: since nothing in the book defines it, four
sentences where Table 2 is introduced (GAP+dense ⇒ the class's dense row
weights the last feature map; ReLU; normalise; that is the whole method), and
a pointer to `lake exe gradcam`.

Deploy is not in this demo. The Orin route (the PJRT plugin runs the
verified graph itself; TensorRT through the PyTorch replica) and a Pi are a
later session (`planning/orin_rerun.md` §1 lists the blockers when it comes).

## 7. Phases

```
Phase 0 (½ session, CPU):   download_plant.sh (two clones, sizes + licences printed),
                            preprocess_plant.py --stats: census, the class map, the ArASL
                            audit on PlantVillage, the seg/bg/mask parts, composites, folds
                            Gate 0: 38 / 54,305 (color) with segmented twins for every file;
                                    27 PlantDoc folders each mapped or listed as unmapped;
                                    audit printed (chain stat + leak fraction) and the split
                                    decision written into the runs README before training;
                                    a composite strip eyeballed
Phase 1 (½ session, GPU):   demos/MainPlantLeaf.lean from MainAraslSigns; base arm, seed 1,
                            scored on pv_test / pv_test_seg / pv_test_bg / pd_all
                            Gate 1: pv_test ≥ 98.5% (the published row; a wrong map or head
                                    reads as < 90); pv_test_bg well above 2.6% (else the
                                    background claim is dead and the section changes);
                                    pd_all restricted between 15% and 60% (outside that
                                    range, check the map and the resize before believing it)
Phase 2 (½ session, GPU):   gradcam `plant` model + dump; plant_cam.py; Table 2's base column
                            Gate 2: mass-inside-leaf reproduces on the probe's Imagenette R34
                                    as a sanity number first (no mask there — skip) → the
                                    plant dump's per-image sums are 1 and the strip renders
Phase 3 (1 session, GPU):   Acts 4a/4b/4c/4c′ and 4a+4c′; seeds on Act 1 and the headline
                            row; the scratch arm
                            Gate 3: every PlantDoc number is on the same 2,578 with fold
                                    hygiene asserted by the scorer; Table 2's fix columns
Phase 4 (½ session):        figure, section, appendix, demos/README, runs README
Later (own session):        deploy — Orin PJRT + TensorRT, a Pi if one appears
```

## 8. Gates that fail loudly

- The census. Mirrors of PlantVillage differ in count (54,305 / 54,303 /
  61,486 augmented) and in whether `segmented` has a twin for every `color`
  file; a missing twin means no mask for that image and it is dropped from
  the seg/bg/mask parts, counted, and printed.
- The class map is the whole of Act 2. A PlantDoc folder with no row in the
  map is an error, not a skip; a PlantVillage class mapped from two PlantDoc
  folders is an error (PlantDoc has no duplicate targets).
- Gate 1's three ranges tell the plumbing failures apart: pv_test low = head
  or labels; pd_all near 1/27 = the map is shuffled or the resize is wrong
  (PlantDoc's aspect ratios are all over the place; a squash instead of a
  crop reads as a 10-point loss); pv_test_bg at chance = the mask is inverted
  (the seg part would score ~0 too).
- Fold hygiene: the scorer reads the fold id of every scored image and the
  fold the checkpoint trained on from the run's tag, and refuses to score a
  training image. Everything in Table 1's field column is held-out or
  zero-shot, no exceptions.
- A fix that lowers PlantDoc is a row, not a bug — unless it also lowers
  PlantVillage below Gate 1, which is a broken pipeline.

## 9. Out of scope

- Detection or segmentation of lesions (PlantDoc ships boxes in a sibling
  repo; a different task).
- MobileNet arms. MNv4-Conv-M's ImageNet checkpoint exists outside the repo
  (`/home/skoonce/mnv4_convm_100ep`, 75.51%), no bootstrap loader or replica
  has read it, and the deploy story does not need a smaller net to be true on
  the Orin. An optional second row if the deploy session wants the edge-net framing —
  after everything above, never instead.
- The fog/haze coda. No restoration section exists in the book to link to.
- Severity, incidence, early symptoms, multi-label frames: the section's one
  sentence on what the classifier is not.
- Deploy. Hardware lowerings are a later session (user, 2026-09-17).
- Redistributing either dataset: the scripts clone from the maintainers.

## 10. Notes before starting

- ⛔ `lake exe plant-leaf` is not a `lake run` job; runs are 30 min each on
  one card, the full ladder ~6 h of GPU time across seeds. Ask before the
  ladder; the base arm alone (Phase 1) does not need asking.
- ⛔ Size the packed parameter buffer from `heInitParams` (the SE-net
  warning; R34 has no SE, keep the line).
- The ImageNet R34 prefix is 21,284,672 floats and its BN *statistics* are
  not in it; the first steps run with fresh stats. If the base arm's epoch-1
  accuracy is not far above the from-scratch arm's at the same epoch (the
  bootstrap should show by the end of epoch 1 — no Imagenette R34 has been
  run from this prefix, so there is no number to quote), suspect the prefix
  offset before anything else — `MainUnetBratsR34.lean` documents the two
  offsets and the NEU detector's bootstrap self-check is the pattern.
- The audit code is imported from `preprocess_arasl.py`, not copied; if the
  function signatures need to move into `scripts/dup_audit.py`, do it in this
  session and point both preprocessors at it.
- `data/imagenette/train.bin` must exist for the composites
  (`download_imagenette.sh`); it does on this box.
- The `gradcam` probe writes to `blueprint/src/figures/gradcam/`; the demo's
  figure goes to `demos/figures/plant_lab_to_field.png` and a copy to
  `blueprint/src/figures/demos/`, as every demo's does.
- Every published number in §0 and §4 marked ⚠ is from memory; look each one
  up before it goes in the section (Mohanty's cross-condition figure, Noyan's
  background-only figure, Singh's PlantDoc numbers and the licence).

## 11. Log — how the plan met its data (2026-09-17)

- **Gate 0.** Census exact (54,305 / 38; every colour file has a `segmented`
  twin once matched on the suffix key — `Corn_(maize)___Common_rust_`'s
  colour files carry no uuid prefix). PlantDoc 2,578 in 28 folders, all
  mapped (no "Potato leaf"; spider mites n=2, no test image). Licences as
  read: PV CC BY-SA 3.0 (the HF card), PlantDoc CC BY 4.0. **PlantVillage
  ships a leaf grouping** (`leaf-map.json`: 7,946 leaves, mode 4 photos per
  leaf, 76% coverage) and the maintainers' own leaf-grouped 80/20 split on
  HF; the `grouped` column's test is their 10,709 exactly.
- **The ArASL audit finds PlantVillage pixel-clean** — §2's expectation of
  "some near-duplicates" was wrong: 0.1% of consecutive pairs under 6 grey
  levels (ArASL 73.5%), 1.3% of random-split test images with a near-twin
  (same leaf 6.9%), 1.0% under grouped. The four photographs of a leaf are
  different poses. The grouped and random columns then agree on the lab
  number (99.64 / 99.72): the split was never the leak; the lab was. This
  is the section's opening move, and it is the opposite of ArASL's.
- **Act 1–2 landed where the literature says.** R34 ImageNet prefix, 10
  epochs: 99.64 lab, **15.79** PlantDoc (restricted 28-way; 38-way 13.62;
  chance 3.6); random split 16.56; from scratch 99.07 / 11.60. Gate 1's
  range (15–60) held at its bottom edge. Bootstrap fit exactly:
  21,304,166 − 19,494 head = 21,284,672.
- **Act 3 had a twist the plan did not predict.** Both attribution
  instruments say the evidence is mostly the leaf — two-player Shapley 85.5%
  leaf share (bg helps in 89.6% of images), CAM 71.5% of mass inside the
  mask (+12.6 over uniform) — and the counterfactuals say the background is
  decisive anyway: leaf on black **49.08**, on its median background colour
  94.98, background silhouette only 27.64, a flat image of the colour 3.83
  (random-split net: 9.32). §4's "CAM mass outside the leaf is high" is not
  what happened; "saliency is not sensitivity" is the line instead. The
  sampled 7×7 Shapley agrees with the CAM leaf by leaf (66/78, 87/96, 78/77,
  74/81) with Σφ = f(full) − f(none) to 1e-3 and SE 0.06–0.09 vs top-patch
  φ 1.2–2.0 (40 permutations, 7,844 forwards).
- **Act 4a is the clean result.** Composites (leaf on Imagenette, train ∪
  comp): every diagnostic moves as predicted — black 49 → 96.57, bg-only
  → 14.06, flat colour → 2.33 (chance), Shapley leaf share → 95.9, CAM lift
  → +19.7 — and PlantDoc 15.79 → **20.09** (intervals disjoint). The fix
  that repairs the lab artefact buys the least in the field, which is the
  honest shape. Over three seeds each: base **17.80 ± 1.77**, composites
  **21.48 ± 1.25** (every fixed seed above every base seed). 4b (aug) 19.24,
  inside the base spread and no diagnostic moved. 4c: 250 field labels
  5-fold **22.69**, all labels **45.00**; from the composite net 26.57 / 42.05
  — the fixes stack at few labels and not at many. Base seeds on the lab
  number 99.57 ± 0.06; the one diagnostic the seed changes is leaf-on-black
  (49 / 59 / 72); Shapley and CAM shares move by under two points.
- **Design changes made on the way:** the whole-part CAM dump lives in
  `plant-leaf cam=1` (the old `gradcam` probe is IREE-era; under XLA the
  `fwd_cam` MLIR compiles like `fwd_eval`), so the statistic is over all
  10,709 test images; u8-resident records via new FFI (`F32.imagenetteGather`
  / `imagenetteLabels`) because the f32 form of a part is 34 GB; `extra=`
  scores any part (the Shapley probe); the two-player Shapley needed the
  `_test_leaf` / `_test_none` parts (`--only-shapley-parts`), all four
  counterfactuals sharing one fill so efficiency holds exactly (4e-6). PlantDoc
  figure picks are the *misread* case at a fixed seed (23) — a random draw
  gave 3 of 4 correct for a 16% net — and two web photos with watermarks were
  avoided by the seed, not by hand. After the user's review the figure went
  eight tiles wide (two leaves per row) and panel (b) gained a fourth column,
  the fold net's CAM after field labels (five `cam=1` evals of the field
  checkpoints on their held-out folds); the section lost every reference to
  other demos and gained a second table with the explainer numbers.
- ⚠ Two traps: `random.Random(numpy.int64)` raises on Python 3.12 (the first
  preprocess died at the composites); a background waiter's `pgrep -f
  <pattern>` matches its own shell.
