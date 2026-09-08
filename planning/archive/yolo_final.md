# YOLO detection demo — final (Oxford-IIIT Pets, cat/dog)

Consolidates the `yolo_*` planning family (demo, v2, v3, v4, v5, with_r34) into one
doc. The demo went from "cursed, won't converge" to a working cat/dog detector — and
the *journey* is the pedagogy.

## What it is
- ResNet-34 backbone (ImageNet `jax_r34_imagenet.bin`, ~21M, bootstrapped) + a deep
  conv head (`conv 512→256 3×3 relu → 256→30 1×1`), 224 input → 7×7×30 grid
  (2 boxes/cell, 20-class slot layout; cat=7/dog=11).
- Loss: `yolov1Masked` + sigmoid focal-BCE objectness (`useFocal`).
- Data: Oxford-IIIT Pets head boxes, 2×2 mosaic, class-balanced
  (`preprocess_pets_mosaic.py` → `data/pets_mosaic_bal/`).
- Dataset kind `DatasetKind.petsDet`; exes `yolov1-pets-train-bootstrap` /
  `yolov1-pets-infer`; render `scripts/yolo_render.py`.
- Runs through the verified train-step codegen — no new VJP machinery.

## The lesson (the real deliverable): marginal dominance
Detection is hard here not because of the architecture but because of the data
distribution. On a coarse 7×7 grid, when objects cluster (centered people in VOC,
centered pet heads in Pets), predicting the *marginal* ("objects are usually here")
beats localizing each one → objectness collapses to a fixed, image-independent
center-prior (acrossImgStd → ~0.001, every image peaks the same cell). The training
loss keeps dropping the whole time — loss ≠ detector quality.

## Two independent failure modes, two fixes
1. **Positional marginal → mosaic.** A 2×2 mosaic of 4 natural pet images decenters
   the heads (spreads them across the grid) and keeps backgrounds natural — a flat
   gray canvas is out-of-distribution for the ImageNet backbone and *regresses*.
   Localization: 0 → **64/64**.
2. **Class marginal → balance.** Pets is 32% cat / 68% dog (25 dog vs 12 cat breeds)
   → class head collapses to always-dog (6/58). 50/50 cat/dog mosaic sampling →
   balanced (**30/34 ≈ GT**), ~64% per-cell accuracy (cat-vs-dog is genuinely hard
   at the small quadrant scale).

## Results (e20, stopped)
- Localization: 64/64 precision+recall (top objectness peaks land on GT heads).
- Class: balanced labels, ~64% cat-vs-dog accuracy.
- Render: clean 4-on-4 at threshold ~0.27. Focal objectness saturates ~0.55 (the
  α-balance equilibrium); score = obj×class is dog-biased in *ranking*, so rank by
  objectness to read the true labels. `--sigmoid-conf` decodes the focal conf logit;
  `--max-per-image` caps to top-K.

## Artifacts
- `preprocess_pets_{det,canvas,mosaic}.py` — det = single head boxes; canvas = the
  failed gray-canvas attempt (kept as the OOD lesson); mosaic = the winner
  (balanced 50/50 species via capital-letter filename = cat).
- `scripts/yolo_render.py`: `--sigmoid-conf`, `--max-per-image`.
- `demos/MainYolov1Pets{TrainBootstrap,Infer}.lean`.
- (The deep head + `weightDecay=0` were tried BEFORE we found the cause was data —
  incidental, not load-bearing.)

## Honest caveats / open
- Trained on mosaics → does NOT transfer to a single full-frame pet (2/16 — centered
  marginal again; needs mixed single+mosaic training, and single centered pets always
  lean on the prior).
- cat-vs-dog ~64% — tops out at this scale.
- If chased further: per-quadrant random-crop (de-cheat the quadrant structure),
  mixed training (single-image support), 448/14×14 (finer grid). None needed for the
  demo to work.

## History (condensed, from the deleted yolo_* docs)
- **demo / v2 / v3** — original plan: R34/ENet backbone + grid head + 5-term MSE on
  VOC; Phases 1–5 shipped (codegen, FFI, VOC trainer, R34 bootstrap, infer/render).
- **with_r34_imagenet** — bootstrapped the 1000-class ImageNet R34 (69.26%) backbone.
- **v4** — SSE→CE class loss (0→16/16 detecting), but class-collapse to "person".
- **v5** — conv head + person-only + the planned focal objectness.
- **final (this doc)** — v5 plus the realization that centered data (VOC-person,
  centered Pets) is *intrinsically* marginal-dominated → pivot to Pets mosaic + class
  balance. VOC retired; rebuild `download_voc.sh` / `preprocess_voc.py` from git
  history (or the pre-sweep commit) if VOC is ever chased again.
