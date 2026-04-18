import LeanMlir

/-! # SegFormer — Bestiary entry

SegFormer (Xie, Wang, Yu, Anandkumar, Alvarez, Luo, NeurIPS 2021 ---
"SegFormer: Simple and Efficient Design for Semantic Segmentation
with Transformers") made two design moves that together displaced
the CNN-plus-ASPP family (DeepLab v3+, PSPNet, etc.) as the default
semantic-segmentation backbone for transformer-era workflows:

1. **Hierarchical transformer encoder (MiT — Mix Transformer).**
   Four stages of transformer blocks, each at a progressively lower
   spatial resolution with a progressively wider channel dim, just
   like a CNN feature pyramid. Output: four multi-scale feature maps,
   no CLS token, no positional embedding.
2. **Lightweight all-MLP decoder.** No upsample convs, no atrous
   pyramid, no skip-connections of the UNet kind. Four small dense
   layers project each encoder scale to the same channel width, a
   bilinear upsample aligns them to the highest-resolution grid, one
   more dense fuses them, and a 1$\times$1 conv produces per-pixel
   logits. The decoder is a few M parameters for every encoder size
   --- the pyramid did the hard work.

What makes SegFormer practically attractive is the \textbf{decoder is
trivial}. Segmentation gains of competitive-with-SOTA quality fall
out of a pretrained transformer backbone and a handful of dense
layers. Compare to DeepLab v3+ where the ASPP module alone has
$\sim$15M parameters and has to be hand-tuned per receptive field.

## MiT variants

| Variant | Stages (ch)       | Blocks        | Heads        | Params |
|---------|-------------------|---------------|--------------|--------|
| MiT-B0  | 32 / 64 / 160 / 256  | 2 / 2 / 2 / 2   | 1 / 2 / 5 / 8  | 3.7M   |
| MiT-B2  | 64 / 128 / 320 / 512 | 3 / 4 / 6 / 3   | 1 / 2 / 5 / 8  | 25.4M  |
| MiT-B5  | 64 / 128 / 320 / 512 | 3 / 6 / 40 / 3  | 1 / 2 / 5 / 8  | 82M    |

All use mlpDim = 4 $\times$ dim per stage and the same 1/4, 1/8, 1/16,
1/32 spatial ratios (relative to the $224 \times 224$ input). SegFormer
with decoder is a couple M over the encoder at each size.

## NetSpec simplifications

- **Efficient self-attention.** Real MiT compresses the key/value
  sequence by 8$\times$ / 4$\times$ / 2$\times$ / 1$\times$ at the four
  stages (``spatial reduction'' to control compute at high res). Our
  \texttt{.transformerEncoder} uses full attention; param count is
  unaffected but compute per forward pass is higher.
- **Overlapping patch embeddings.** MiT's inter-stage transitions use
  overlapping strided convs (stride 4 then stride 2's) rather than
  non-overlapping patch merging. We use \texttt{.patchEmbed} for the
  initial stem and \texttt{.patchMerging} for stages 2--4; the
  resulting param counts are close enough for shape purposes.
- **Decoder upsample.** Bilinear upsample is parameter-free; we show
  the decoder as a chain of \texttt{.dense} projections and a final
  1$\times$1 output conv.
-/

-- ════════════════════════════════════════════════════════════════
-- § SegFormer MiT-B0 encoder — smallest, 3.7M paper
-- ════════════════════════════════════════════════════════════════

def segformerB0 : NetSpec where
  name := "SegFormer MiT-B0 encoder"
  imageH := 224
  imageW := 224
  layers := [
    -- Stage 1: 224 → 56, 32 channels, 2 blocks
    .patchEmbed 3 32 4 (56 * 56),
    .transformerEncoder 32 1 128 2,
    -- Stage 2: 56 → 28, 64 channels, 2 blocks
    .patchMerging 32 64,
    .transformerEncoder 64 2 256 2,
    -- Stage 3: 28 → 14, 160 channels, 2 blocks
    .patchMerging 64 160,
    .transformerEncoder 160 5 640 2,
    -- Stage 4: 14 → 7, 256 channels, 2 blocks
    .patchMerging 160 256,
    .transformerEncoder 256 8 1024 2
  ]

-- ════════════════════════════════════════════════════════════════
-- § SegFormer MiT-B2 encoder — mid-size, 25M paper
-- ════════════════════════════════════════════════════════════════

def segformerB2 : NetSpec where
  name := "SegFormer MiT-B2 encoder"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 64 4 (56 * 56),
    .transformerEncoder 64 1 256 3,
    .patchMerging 64 128,
    .transformerEncoder 128 2 512 4,
    .patchMerging 128 320,
    .transformerEncoder 320 5 1280 6,
    .patchMerging 320 512,
    .transformerEncoder 512 8 2048 3
  ]

-- ════════════════════════════════════════════════════════════════
-- § SegFormer MiT-B5 encoder — largest, 82M paper
-- ════════════════════════════════════════════════════════════════
-- Stage 3 blows up to 40 blocks; that's where almost all the params land.

def segformerB5 : NetSpec where
  name := "SegFormer MiT-B5 encoder"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 64 4 (56 * 56),
    .transformerEncoder 64 1 256 3,
    .patchMerging 64 128,
    .transformerEncoder 128 2 512 6,
    .patchMerging 128 320,
    .transformerEncoder 320 5 1280 40,
    .patchMerging 320 512,
    .transformerEncoder 512 8 2048 3
  ]

-- ════════════════════════════════════════════════════════════════
-- § SegFormer MLP decoder (shared shape, illustrated at B2 width)
-- ════════════════════════════════════════════════════════════════
-- Real decoder takes 4 feature maps (at 1/4, 1/8, 1/16, 1/32 scales),
-- projects each to a common channel dim `embedDim` (256 for B0, 768
-- elsewhere), bilinearly upsamples all to 1/4, concats, and fuses.
-- We show a single-scale approximation at the B2 embedDim.

def segformerDecoder : NetSpec where
  name := "SegFormer MLP decoder (single-scale approx at B2 widths)"
  imageH := 7          -- last-stage output resolution (1/32)
  imageW := 7
  layers := [
    -- Real decoder has 4 parallel dense projections (one per encoder
    -- scale), bilinear-upsampled to 1/4 resolution, concatenated, then
    -- fused by one more dense. A linear NetSpec can't express the four
    -- parallel projections, so we approximate with a single-scale
    -- chain at the deepest-stage width (512) threading through the
    -- same 768-dim embedding space + the 3072-dim fusion width.
    .dense 512 768 .identity,      -- scale projection (last stage 512 → 768)
    .dense 768 768 .identity,      -- stands for upsample + concat-then-fuse
    -- Per-pixel classifier (ADE20K: 150 classes)
    .dense 768 150 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinySegFormer — compact fixture
-- ════════════════════════════════════════════════════════════════

def tinySegFormer : NetSpec where
  name := "tiny-SegFormer"
  imageH := 64
  imageW := 64
  layers := [
    .patchEmbed 3 16 4 (16 * 16),
    .transformerEncoder 16 1 64 2,
    .patchMerging 16 32,
    .transformerEncoder 32 2 128 2,
    .patchMerging 32 64,
    .transformerEncoder 64 4 256 2,
    -- Tiny MLP decoder (single-scale for simplicity)
    .dense 64 128 .identity,
    .dense 128 10 .identity
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
  IO.println "  Bestiary — SegFormer"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Hierarchical transformer encoder + lightweight MLP decoder."
  IO.println "  Transformer feature pyramid made segmentation-via-CNN obsolete."

  summarize segformerB0
  summarize segformerB2
  summarize segformerB5
  summarize segformerDecoder
  summarize tinySegFormer

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Encoder is 4 × (.patchMerging +"
  IO.println "    .transformerEncoder); decoder is a handful of .dense calls."
  IO.println "  • Real MiT uses 'efficient self-attention' — KV-sequence"
  IO.println "    reduction at each stage (8x/4x/2x/1x). Param count is"
  IO.println "    unchanged; compute per forward pass is lower. Our spec"
  IO.println "    uses standard attention, which is the right param budget"
  IO.println "    even if it's not the compute-optimal implementation."
  IO.println "  • Decoder is trivially small — a few .dense calls aligning"
  IO.println "    4 feature scales to a common dim, then a fusion dense and"
  IO.println "    a class projection. Bilinear upsample in between is"
  IO.println "    parameter-free."
  IO.println "  • The decoder paragraph is the whole architectural argument"
  IO.println "    of the paper: a good pretrained transformer pyramid makes"
  IO.println "    the segmentation head cheap. Compare DeepLab v3+'s ASPP"
  IO.println "    module (~15M params, hand-tuned dilation rates) to"
  IO.println "    SegFormer's ~3M decoder that works across all B0-B5 sizes."
