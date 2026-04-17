import LeanMlir

/-! # DETR — Bestiary entry

DETR (Carion et al., 2020 — "End-to-End Object Detection with Transformers")
killed anchor boxes and non-max suppression in one paper. The trick: frame
detection as **set prediction**. A transformer decoder takes `N` learned
"object queries" and a transformer-encoded image; each query ends up
predicting one (class, box) pair or "no object." The bipartite-matching
loss (Hungarian) handles the set-to-set alignment during training.

```
    Image (3, H, W)
        │
        ▼
    ResNet-50 backbone            → (2048, H/32, W/32)
        │
        ▼
    patchEmbed(2048 → 256, 1×1)   ← 1×1 conv + flatten spatial + pos-embed
        │                           → (N_tokens, 256) where N = (H/32)·(W/32)
        ▼
    TransformerEncoder ×6          → context tokens (N, 256)
        │
        ▼
    TransformerDecoder ×6          ← 100 learned object queries as input
        │                           → 100 refined queries (100, 256)
        ▼
    DETRHeads                      → per query: class(92) + box(4)
```

The elegant bit: the whole pipeline is differentiable end-to-end. No
hand-engineered components (anchors, ROI pool, NMS) — just a transformer
with learned queries. DETR itself is slower to converge than
anchor-based detectors, but its successors (Deformable DETR, DINO,
Co-DETR) inherited the clean set-prediction formulation.

## Variants

| Model    | Backbone  | Encoder | Decoder | Params | COCO AP |
|----------|-----------|---------|---------|--------|---------|
| DETR-R50 | ResNet-50 | 6 × 8h  | 6 × 8h  | 41M    | 42.0    |
| DETR-R101| ResNet-101| 6 × 8h  | 6 × 8h  | 61M    | 43.5    |
| tinyDETR | 2-stage miniRN | 2 × 4h | 2 × 4h | ~3M | n/a   |

All use dim = 256, mlp_dim = 2048, 100 object queries, 91 COCO classes
(+ 1 "no object" slot = 92 output classes on the class head).
-/

-- ════════════════════════════════════════════════════════════════
-- § DETR-R50 (canonical)
-- ════════════════════════════════════════════════════════════════

/-- Canonical DETR: ResNet-50 backbone + 6-layer encoder/decoder with
    100 object queries + class/box heads over 91 COCO classes.

    Input shape: 800 × 800 (representative; DETR trains with aspect-
    preserving resize). After the backbone (stride 32) the spatial grid
    is 25 × 25 = 625 tokens for the transformer. -/
def detrR50 : NetSpec where
  name := "DETR-R50"
  imageH := 800
  imageW := 800
  layers := [
    -- ResNet-50 backbone (stem + 4 stages)
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock 64 256 3 1,              -- C2
    .bottleneckBlock 256 512 4 2,             -- C3
    .bottleneckBlock 512 1024 6 2,            -- C4
    .bottleneckBlock 1024 2048 3 2,           -- C5 (final feature map)
    -- patchEmbed with patchSize=1: channel projection (2048→256) + flatten
    -- spatial + positional embed. 25×25 = 625 tokens from 800×800 input.
    .patchEmbed 2048 256 1 625,
    -- Transformer encoder: 6 blocks, 8 heads, mlp 2048.
    .transformerEncoder 256 8 2048 6,
    -- Transformer decoder: 6 blocks, 8 heads, 100 object queries.
    .transformerDecoder 256 8 2048 6 100,
    -- Per-query class head (→ 92) + box head (→ 4).
    .detrHeads 256 91
  ]

-- ════════════════════════════════════════════════════════════════
-- § DETR-R101 (heavier backbone)
-- ════════════════════════════════════════════════════════════════

def detrR101 : NetSpec where
  name := "DETR-R101"
  imageH := 800
  imageW := 800
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock 64 256 3 1,
    .bottleneckBlock 256 512 4 2,
    .bottleneckBlock 512 1024 23 2,           -- ResNet-101: 23 blocks in C4 vs 6
    .bottleneckBlock 1024 2048 3 2,
    .patchEmbed 2048 256 1 625,
    .transformerEncoder 256 8 2048 6,
    .transformerDecoder 256 8 2048 6 100,
    .detrHeads 256 91
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyDETR — fixture (tiny backbone, small transformer)
-- ════════════════════════════════════════════════════════════════

def tinyDETR : NetSpec where
  name := "tiny-DETR"
  imageH := 128
  imageW := 128
  layers := [
    .convBn 3 32 7 2 .same,
    .maxPool 2 2,
    .bottleneckBlock 32 64 2 1,
    .bottleneckBlock 64 128 2 2,
    .patchEmbed 128 64 1 64,                   -- 8×8 = 64 tokens
    .transformerEncoder 64 4 256 2,
    .transformerDecoder 64 4 256 2 20,          -- 20 object queries
    .detrHeads 64 20                           -- 20 classes for a toy dataset
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
  IO.println "  Bestiary — DETR"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Object detection as set prediction. No anchors, no NMS."
  IO.println "  Killed a decade of hand-engineered detection heads in one paper."

  summarize detrR50
  summarize detrR101
  summarize tinyDETR

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.transformerDecoder` and `.detrHeads` are new Layer"
  IO.println "    constructors for this bestiary. Decoder wraps 6 blocks of"
  IO.println "    (self-attn + cross-attn + FFN) with a learned query matrix;"
  IO.println "    `detrHeads` bundles the per-query class (→ nClasses+1) and"
  IO.println "    box (→ 4) MLPs. Codegen emits UNSUPPORTED for both."
  IO.println "  • `patchEmbed` with patchSize=1 does channel projection"
  IO.println "    (2048→256 for R50) + flatten spatial + positional embed"
  IO.println "    in one step — absorbs the explicit 1×1 conv that most"
  IO.println "    reference implementations name separately."
  IO.println "  • The bipartite-matching (Hungarian) loss is a training-time"
  IO.println "    concern, not architecture. It lives outside the NetSpec."
  IO.println "  • Followups (Deformable DETR, DINO, Co-DETR) swap attention"
  IO.println "    flavor or add denoising; the NetSpec shape stays the same."
