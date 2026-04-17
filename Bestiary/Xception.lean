import LeanMlir

/-! # Xception — Bestiary entry

Xception (Chollet, 2017 — "Xception: Deep Learning with Depthwise
Separable Convolutions") takes the Inception hypothesis to its logical
extreme. Inception says "cross-channel correlations and spatial
correlations can be decoupled — use parallel 1×1 projections and
then do spatial convs on the reduced channels." Xception says: fine,
let's **completely** decouple them. **Every conv in the network is
a depthwise-separable conv**: a pointwise 1×1 for cross-channel
mixing, followed by a depthwise $k\times k$ for spatial mixing, with
*no* cross-channel interaction in the spatial step.

The paper's title is a pun: "Extreme Inception."

## Architecture

```
                        ┌─────────────────────┐
                        │     Entry flow      │
                        │  (downsample 299→19)│
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │    Middle flow      │
                        │      × 8 times      │     ← 8 identical
                        │  (same 19×19, 728   │       sep-conv blocks
                        │   channels)         │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │     Exit flow       │
                        │  (channels → 2048)  │
                        └──────────┬──────────┘
                                   │
                           Global avg pool
                           Dense → 1000
```

The key building block is just `.separableConv`, which we already have
in `Types.lean`. Every block has:
- 3 depthwise-separable convs in series
- Residual skip around them
- Max-pool for downsampling (entry/exit flows only)

~22M params on ImageNet. Slightly better accuracy than Inception-v3 at
~similar cost — proof that the "aggressive channel-space decoupling"
hypothesis beats Inception's hand-designed parallel branches.

## Why this matters

Xception is the link between Inception (2014–16) and MobileNet v1
(2017). MobileNet basically said "separable convs + stride for
downsampling + linear bottlenecks" and got to 4M params for mobile
deployment. Xception had already proven the separable-conv thesis
worked at server scale.
-/

-- ════════════════════════════════════════════════════════════════
-- § Xception (canonical)
-- ════════════════════════════════════════════════════════════════

def xception : NetSpec where
  name := "Xception"
  imageH := 299
  imageW := 299
  layers := [
    -- Entry flow
    .convBn 3 32 3 2 .same,
    .convBn 32 64 3 1 .same,

    -- Entry flow: three "separable conv + residual-projection" blocks
    .separableConv 64 128 1,
    .separableConv 128 128 1,
    .maxPool 2 2,

    .separableConv 128 256 1,
    .separableConv 256 256 1,
    .maxPool 2 2,

    .separableConv 256 728 1,
    .separableConv 728 728 1,
    .maxPool 2 2,

    -- Middle flow: 8 blocks of 3 separable convs each at 728 channels.
    -- Each block has a residual (not shown in linear NetSpec — the
    -- pattern is the Inception-flavored skip around 3 sep convs, repeated 8×).
    -- Bestiary linearizes: 24 separableConvs in series, residuals implicit.
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,
    .separableConv 728 728 1, .separableConv 728 728 1, .separableConv 728 728 1,

    -- Exit flow
    .separableConv 728 728  1,
    .separableConv 728 1024 1,
    .maxPool 2 2,
    .separableConv 1024 1536 1,
    .separableConv 1536 2048 1,

    .globalAvgPool,
    .dense 2048 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyXception fixture
-- ════════════════════════════════════════════════════════════════

def tinyXception : NetSpec where
  name := "tiny-Xception"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 16 3 1 .same,
    .separableConv 16 32 1,
    .separableConv 32 32 1,
    .maxPool 2 2,
    .separableConv 32 64 1,
    .separableConv 64 64 1,
    .maxPool 2 2,
    .separableConv 64 128 1,
    .separableConv 128 128 1,
    .globalAvgPool,
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
  IO.println "  Bestiary — Xception"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Inception hypothesis taken to its extreme: every conv is a"
  IO.println "  depthwise-separable conv. 'Extreme Inception.' ~22M params,"
  IO.println "  better than Inception-v3 at similar cost."

  summarize xception
  summarize tinyXception

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Zero new Layer primitives. `.separableConv` was already in"
  IO.println "    Types.lean — Xception is what it's for."
  IO.println "  • Residual connections (one per block, around each middle-flow"
  IO.println "    + entry-flow-block trio of separable convs) are implicit."
  IO.println "    Our linear NetSpec shows the 36 conv layers in series; the"
  IO.println "    skip-adds are every-third-sep-conv in prose."
  IO.println "  • Lineage: Xception (2017, server-scale, 22M) → MobileNet v1"
  IO.println "    (2017, mobile-scale, 4M) → EfficientNet (2019, scaling law)."
  IO.println "    All downstream of 'cross-channel and spatial correlations"
  IO.println "    can be decoupled.'"
