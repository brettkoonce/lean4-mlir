import LeanMlir

/-! # ShuffleNet v1 — Bestiary entry

ShuffleNet v1 (Zhang, Zhou, Lin, Sun, 2017) is one of the classic
mobile-optimized CNNs. Two tricks make it cheap:

1. **Grouped 1×1 convolutions.** Splitting channels into `g` independent
   groups cuts the cost of a 1×1 conv by a factor of `g`. The
   bottleneck block of a ResNet-ish unit already used 1×1 convs
   (expand then project); ShuffleNet makes them grouped.

2. **Channel shuffle.** Grouped convs are cheap but trap channels
   inside their group — information can't cross group boundaries, so
   stacked grouped convs become strictly less expressive than a full
   conv. ShuffleNet inserts a **channel shuffle** permutation between
   the two grouped 1×1s: rearrange `(g × (c/g))` as `((c/g) × g)` so
   the next grouped conv sees a fresh mixture. Costs zero params and
   zero FLOPs — it's just an index permutation.

## The shuffle unit

```
         x (in)
           │
           ▼
     1×1 GConv + BN + ReLU      ← grouped convolution, g groups
           │
           ▼
     channel shuffle            ← the trick (no params, no FLOPs)
           │
           ▼
     3×3 DWConv + BN            ← depthwise, cheap by construction
           │
           ▼
     1×1 GConv + BN             ← second grouped 1×1, no activation
           │
           ▼
    + x  (residual)  or  concat with avg-pooled x (downsample variant)
           │
           ▼
         ReLU
```

The downsampling variant uses the DWConv at stride 2 and replaces the
additive skip with a concatenation of the avg-pooled input — so the
channel count can increase across stages without an extra 1×1
projection on the skip path.

## Variants

The paper's canonical setting is `g = 3`. The per-stage channel counts
scale with a width multiplier:

| Variant                | Stage 2 / 3 / 4 channels | Params | Top-1 err |
|------------------------|---------------------------|--------|-----------|
| ShuffleNet 0.5× (g=3) | 120 / 240 / 480          | ~1.0M  | 43.2%     |
| ShuffleNet 1.0× (g=3) | 240 / 480 / 960          | ~2.4M  | 32.6%     |
| ShuffleNet 1.5× (g=3) | 360 / 720 / 1440         | ~3.4M  | 31.3%     |
| ShuffleNet 2.0× (g=3) | 480 / 960 / 1920         | ~5.4M  | 29.1%     |

Per-stage unit counts are `[4, 8, 4]` across the three shuffle stages,
matching the paper's Table 1 (a total of 16 shuffle units plus the
stem + classifier).
-/

-- ════════════════════════════════════════════════════════════════
-- § ShuffleNet 1.0× (g=3)  —  the canonical spec
-- ════════════════════════════════════════════════════════════════

def shuffleNet1x : NetSpec where
  name := "ShuffleNet 1.0× (g=3)"
  imageH := 224
  imageW := 224
  layers := [
    -- Stem: standard conv + max-pool (like early ResNets)
    .convBn 3 24 3 2 .same,
    .maxPool 2 2,
    -- Three shuffle stages: [4, 8, 4] units, doubling channels each stage
    .shuffleBlock 24  240 3 4,                  -- Stage 2
    .shuffleBlock 240 480 3 8,                  -- Stage 3
    .shuffleBlock 480 960 3 4,                  -- Stage 4
    -- Global pool + classifier
    .globalAvgPool,
    .dense 960 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ShuffleNet 0.5× (g=3)  —  smallest canonical variant
-- ════════════════════════════════════════════════════════════════

def shuffleNet0_5x : NetSpec where
  name := "ShuffleNet 0.5× (g=3)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 24 3 2 .same,
    .maxPool 2 2,
    .shuffleBlock 24  120 3 4,
    .shuffleBlock 120 240 3 8,
    .shuffleBlock 240 480 3 4,
    .globalAvgPool,
    .dense 480 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § ShuffleNet 2.0× (g=3)  —  largest canonical variant
-- ════════════════════════════════════════════════════════════════

def shuffleNet2x : NetSpec where
  name := "ShuffleNet 2.0× (g=3)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 24 3 2 .same,
    .maxPool 2 2,
    .shuffleBlock 24  480  3 4,
    .shuffleBlock 480 960  3 8,
    .shuffleBlock 960 1920 3 4,
    .globalAvgPool,
    .dense 1920 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyShuffle  —  bestiary fixture
-- ════════════════════════════════════════════════════════════════

def tinyShuffle : NetSpec where
  name := "tiny-ShuffleNet (g=2)"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .shuffleBlock 16 32 2 2,
    .shuffleBlock 32 64 2 2,
    .shuffleBlock 64 128 2 2,
    .globalAvgPool,
    .dense 128 10 .identity                      -- CIFAR-10 head
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
  IO.println "  Bestiary — ShuffleNet v1"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Mobile-optimized CNN. Grouped 1×1 convs for cheap bottlenecks"
  IO.println "  + channel shuffle to restore cross-group information flow."

  summarize shuffleNet0_5x
  summarize shuffleNet1x
  summarize shuffleNet2x
  summarize tinyShuffle

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.shuffleBlock ic oc groups nUnits` bundles a full stage:"
  IO.println "    one stride-2 down-unit + (nUnits-1) residual units. Each"
  IO.println "    unit is (1×1 GConv → channel shuffle → 3×3 DWConv → 1×1 GConv)"
  IO.println "    with a residual or avg-pool-concat skip."
  IO.println "  • Channel shuffle is parameter-free — just an index permutation."
  IO.println "    Param counts are governed by the grouped 1×1 convs (ic/g ×"
  IO.println "    oc/g × g = ic·oc/g). Formula reflects this."
  IO.println "  • ShuffleNet v2 (Ma et al. 2018) replaces grouping with a"
  IO.println "    channel split per unit, trading FLOPs for hardware-friendliness."
  IO.println "    Worth its own bestiary entry if anyone wants it."
