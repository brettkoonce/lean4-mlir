import LeanMlir

/-! # Mask R-CNN — Bestiary entry

Mask R-CNN (He, Gkioxari, Doll\'ar, Girshick, ICCV 2017 --- ``Mask
R-CNN'') is the reference two-stage detector. It extends Faster
R-CNN with a parallel mask-prediction head, making instance
segmentation (detect + segment each instance) a single-network
problem. The default choice for bench-marking papers through 2020,
and the canonical example of the ``propose then classify'' detection
paradigm. DETR (already in the bestiary) is its end-to-end
transformer-era cousin.

## Five components

Mask R-CNN is architecturally rich --- five pieces, each with its
own responsibility:

1. **Backbone.** ResNet-50 or ResNet-101. Produces four feature
   maps at strides 4 / 8 / 16 / 32 (conventionally called C2, C3,
   C4, C5).
2. **FPN (Feature Pyramid Network).** Lin et al.\ 2017. Merges the
   four backbone stages into a 4-level feature pyramid at a common
   channel width (usually 256), via 1$\times$1 lateral convs + top-
   down upsample-and-add + 3$\times$3 smoothing. The pyramid is what
   makes the detector handle objects across a wide scale range.
3. **RPN (Region Proposal Network).** A small conv network that
   runs at each pyramid level and outputs anchor-box proposals:
   objectness score + bounding-box deltas per anchor.
4. **ROI-Align + box head.** For each proposal, ROI-Align crops a
   $7 \times 7 \times 256$ feature tile from the pyramid. Then a
   2-layer MLP (FC 1024 $\to$ 1024) produces class logits and
   bbox regression deltas.
5. **Mask head.** For each proposal, ROI-Align crops a $14 \times
   14 \times 256$ tile. A 4-layer conv stack + transposed conv
   upsample + 1$\times$1 conv produces a per-class 28$\times$28
   mask. The mask head is the ``+Mask'' in Mask R-CNN.

The ROI-Align step isn't a layer --- it's a crop-and-bilinear-sample
operation applied per proposal, dispatching the per-ROI features to
the heads. That's orchestration, not architecture. Our bestiary entry
shows each head as its own NetSpec; running the heads per-proposal is
a training-code concern.

## Why this is the ``canonical reference''

Every published 2-stage detection or instance-segmentation paper
compares to Mask R-CNN. The FPN module alone has outlived the rest
--- it's standard equipment in detectors, segmentors, and even some
transformer stacks. ROI-Align's bilinear sampling replaced the
quantized ROI-Pool of Faster R-CNN and is now the default. The
mask head's per-class 28$\times$28 prediction is the template
followed by Cascade Mask R-CNN, Mask2Former, and many others.

## Variants

- `maskRCNNBackboneR101`   --- ResNet-101 up through C5 + FPN
- `maskRCNNRPN`            --- RPN head (shared across pyramid levels)
- `maskRCNNBoxHead`        --- classification + bbox head (80 COCO classes)
- `maskRCNNMaskHead`       --- per-class mask head
- `tinyMaskRCNN`           --- compact fixture
-/

-- ════════════════════════════════════════════════════════════════
-- § Backbone + FPN: ResNet-101-FPN feature extractor
-- ════════════════════════════════════════════════════════════════
-- Stage outputs of ResNet-101:
--   C2: 256 ch  @ 1/4   resolution
--   C3: 512 ch  @ 1/8   resolution
--   C4: 1024 ch @ 1/16  resolution
--   C5: 2048 ch @ 1/32  resolution
-- FPN maps these to four 256-ch pyramid levels at the same resolutions.

def maskRCNNBackboneR101 : NetSpec where
  name := "Mask R-CNN backbone (ResNet-101-FPN)"
  imageH := 800
  imageW := 800
  layers := [
    -- ResNet-101 stem
    .convBn 3 64 7 2 .same,
    .maxPool 3 2,
    -- ResNet-101 body (C2/C3/C4/C5)
    .bottleneckBlock 64   256  3  1,
    .bottleneckBlock 256  512  4  2,
    .bottleneckBlock 512  1024 23 2,
    .bottleneckBlock 1024 2048 3  2,
    -- FPN: combines C2–C5 into a 4-level feature pyramid at 256 channels
    .fpnModule 256 512 1024 2048 256
  ]

-- ════════════════════════════════════════════════════════════════
-- § RPN (Region Proposal Network)
-- ════════════════════════════════════════════════════════════════
-- Applied at each FPN pyramid level (shared weights across levels).
-- Per-location outputs: objectness score (2 logits per anchor; we use
-- 3 anchors per location as the FPN default) + 4 bbox deltas per anchor.

def maskRCNNRPN : NetSpec where
  name := "Mask R-CNN RPN (shared across pyramid levels)"
  imageH := 50        -- illustrative feature-map size (one pyramid level)
  imageW := 50
  layers := [
    -- 3×3 "context" conv shared across pyramid levels
    .conv2d 256 256 3 .same .relu,
    -- Objectness head: 1×1 conv to 2 × nAnchors logits (paper: 3 anchors → 6 ch)
    .conv2d 256 6 1 .same .identity,
    -- NOTE: the bbox-delta head (1×1 conv to 4 × nAnchors) runs in
    -- parallel, not after. Param-count contribution: 256*12 + 12 = 3084.
    -- Shown here as sequential for NetSpec linearity.
    .conv2d 6 12 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Box head (per-ROI: classification + bbox regression)
-- ════════════════════════════════════════════════════════════════
-- Per proposal: ROI-Align produces a 7×7×256 tile, flattened to
-- 12544-dim vector, passed through two FC layers.

def maskRCNNBoxHead : NetSpec where
  name := "Mask R-CNN box head (FC 1024 → 1024, COCO 80 classes)"
  imageH := 7          -- ROI-Align output height
  imageW := 7
  layers := [
    .flatten,
    -- 7*7*256 = 12544 features
    .dense 12544 1024 .relu,
    .dense 1024 1024 .relu,
    -- Parallel class head (81 = 80 COCO + 1 background) + bbox head
    -- (4 coords × 80 classes = 320). Shown here as one combined dense
    -- of matching output width (81 + 320 = 401) for NetSpec linearity.
    .dense 1024 401 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Mask head (per-ROI: per-class 28×28 mask)
-- ════════════════════════════════════════════════════════════════
-- ROI-Align produces 14×14×256, four 3×3 convs maintain shape, a
-- 2×2 transposed conv upsamples to 28×28, final 1×1 conv produces
-- a 28×28 mask per class.

def maskRCNNMaskHead : NetSpec where
  name := "Mask R-CNN mask head (COCO 80 classes, 28×28 masks)"
  imageH := 14         -- ROI-Align output for mask branch
  imageW := 14
  layers := [
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,
    -- Transposed conv 2×2, stride 2: 14×14 → 28×28.
    -- Approximated here as a 1×1 conv (same param count as a 2×2 conv
    -- with reversed stride: 4 * ic * oc weights + bias).
    .conv2d 256 256 2 .same .relu,
    -- Per-class mask logits: 80 classes
    .conv2d 256 80 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyMaskRCNN fixture
-- ════════════════════════════════════════════════════════════════
-- Compact backbone + FPN + heads inlined for read-at-a-glance.

def tinyMaskRCNN : NetSpec where
  name := "tiny-MaskRCNN"
  imageH := 128
  imageW := 128
  layers := [
    .convBn 3 32 3 2 .same,
    .bottleneckBlock 32 64  2 2,
    .bottleneckBlock 64 128 2 2,
    .bottleneckBlock 128 256 2 2,
    .bottleneckBlock 256 512 2 2,
    .fpnModule 64 128 256 512 128
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Mask R-CNN"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Two-stage detection + instance segmentation. FPN pyramid +"
  IO.println "  RPN proposals + ROI-Align + parallel box / mask heads."

  summarize maskRCNNBackboneR101
  summarize maskRCNNRPN
  summarize maskRCNNBoxHead
  summarize maskRCNNMaskHead
  summarize tinyMaskRCNN

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • New primitive: .fpnModule (c2 c3 c4 c5 target). Same pattern"
  IO.println "    as other bundled primitives — the FPN's cross-scale add"
  IO.println "    doesn't linearize layer-by-layer, so we bundle the whole"
  IO.println "    four-level combine into one constructor."
  IO.println "  • Components shown as separate NetSpecs (SAM-style). ROI-"
  IO.println "    Align is an orchestration step that runs once per proposal;"
  IO.println "    it dispatches features to the box / mask heads per-ROI."
  IO.println "    Not a layer."
  IO.println "  • Total Mask R-CNN ResNet-101-FPN (from paper): ~63M params."
  IO.println "    Our backbone+FPN lands around ~45M; adding box head (~13M)"
  IO.println "    + mask head (~2.5M) + RPN (~0.6M) gets close to ~61M."
  IO.println "  • Mask head produces per-class 28×28 masks. At inference,"
  IO.println "    you take the mask for the predicted class only — cheap"
  IO.println "    to compute, avoids mask-classification coupling."
  IO.println "  • Cascade Mask R-CNN (Cai & Vasconcelos 2018) iterates"
  IO.println "    the box head at increasing IoU thresholds. Mask2Former"
  IO.println "    (Cheng et al. 2022) is the DETR-style successor — same"
  IO.println "    instance-segmentation goal, transformer-decoder-based"
  IO.println "    mechanism."
