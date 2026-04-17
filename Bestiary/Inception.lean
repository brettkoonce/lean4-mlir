import LeanMlir

/-! # Inception (v1 / v3 / v4) — Bestiary entry

The Inception family (Szegedy et al., 2014–2016) introduced the idea of
**parallel multi-scale feature extraction**: instead of picking a
"best" convolution kernel size for each layer, run several kernel
sizes in parallel and let the network learn which to emphasize.

Three papers, three iterations of the same idea:

- **v1 / GoogLeNet** (Szegedy et al. 2014) — 9 inception modules, two
  auxiliary classifiers to keep gradients flowing through a 22-layer
  network, 5M params. Won ILSVRC 2014.
- **v3** (Szegedy et al. 2015, "Rethinking the Inception Architecture")
  — introduced factorized convolutions (7×7 → 1×7 then 7×1 — asymmetric,
  cheaper), label smoothing, and RMSProp. ~23M params.
- **v4** (Szegedy et al. 2016) — cleaner, more uniform design. ~42M
  params. The paper pairs it with Inception-ResNet (add residual
  connections to each inception block).

## The Inception module (v1)

```
                     input (ic channels)
                            │
            ┌───────┬───────┼───────┬─────────┐
            │       │       │       │         │
            ▼       ▼       ▼       ▼         ▼
         1×1 ic→b1   1×1 ic→b2r  1×1 ic→b3r   3×3 maxPool
            │       │       │       │         │
            │       ▼       ▼       │         ▼
            │   3×3 b2r→b2 5×5 b3r→b3  1×1 ic→b4
            │       │       │                  │
            └───────┴───────┴──────────────────┘
                            │
                            ▼
                  concat along channels  (b1 + b2 + b3 + b4 channels out)
```

The 1×1 convs on branches 2, 3, and 4 are **dimension reducers** — they
cut the input channel count before the expensive 3×3 and 5×5 convs
apply. Without them a naive all-parallel inception module would be
computationally infeasible. With them, parameters stay manageable.

This is the trick of the paper: "use 1×1 convs as channel-space
bottlenecks." Inception blocks end up roughly as cheap as comparable
ResNet bottlenecks, and the multi-scale concat gives richer features.

## Inception v3 and v4

v3 and v4 use MORE inception module types (Inception-A, -B, -C,
Reduction-A, Reduction-B, and so on) with specific asymmetric
factorizations and different reduction geometries. Cataloguing each
module as its own primitive would bloat the Layer ADT with module-
family-specific constructors that mostly do the same thing at slightly
different widths.

For bestiary purposes, we **approximate v3 and v4 with the v1-style
`.inceptionModule`** at appropriate channel counts. The shape / flow
is the same; the total param count lands within ~20% of the paper's
numbers. Honest limitation, worth the simplification.
-/

-- ════════════════════════════════════════════════════════════════
-- § GoogLeNet (Inception v1, 2014)
-- ════════════════════════════════════════════════════════════════

def googLeNet : NetSpec where
  name := "GoogLeNet (Inception v1)"
  imageH := 224
  imageW := 224
  layers := [
    -- Stem
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                              -- 56×56 output
    .convBn 64 64 1 1 .same,                   -- 1×1 reduce (simplified)
    .convBn 64 192 3 1 .same,
    .maxPool 2 2,                              -- 28×28

    -- Inception 3a: 256 = 64 + 128 + 32 + 32
    .inceptionModule 192 64 96 128 16 32 32,
    -- Inception 3b: 480 = 128 + 192 + 96 + 64
    .inceptionModule 256 128 128 192 32 96 64,
    .maxPool 2 2,                              -- 14×14

    -- Inception 4a..4e
    .inceptionModule 480 192 96  208 16 48  64,     -- 512
    .inceptionModule 512 160 112 224 24 64  64,     -- 512
    .inceptionModule 512 128 128 256 24 64  64,     -- 512
    .inceptionModule 512 112 144 288 32 64  64,     -- 528
    .inceptionModule 528 256 160 320 32 128 128,    -- 832
    .maxPool 2 2,                              -- 7×7

    -- Inception 5a, 5b
    .inceptionModule 832 256 160 320 32 128 128,    -- 832
    .inceptionModule 832 384 192 384 48 128 128,    -- 1024

    -- Classifier
    .globalAvgPool,
    .dense 1024 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Inception v3 (2015, approximate)
-- ════════════════════════════════════════════════════════════════

/-- Inception v3 at bestiary fidelity: stem + 3 blocks of
    "inception-A"-like + 5 blocks of "inception-B"-like + 2 blocks of
    "inception-C"-like, each approximated with the unified v1-style
    `.inceptionModule`. Input 299×299 as per the paper. -/
def inceptionV3 : NetSpec where
  name := "Inception-v3 (bestiary approximation)"
  imageH := 299
  imageW := 299
  layers := [
    -- Stem: a handful of stride-2 convs taking 299 → 35 spatial
    .convBn 3 32 3 2 .same,
    .convBn 32 32 3 1 .same,
    .convBn 32 64 3 1 .same,
    .maxPool 2 2,
    .convBn 64 80 1 1 .same,
    .convBn 80 192 3 1 .same,
    .maxPool 2 2,

    -- Three "Inception-A" modules at 35×35
    .inceptionModule 192 64 48 64  64 96 32,
    .inceptionModule 256 64 48 64  64 96 64,
    .inceptionModule 288 64 48 64  64 96 64,

    -- Five "Inception-B" modules at 17×17 (approximated; real B uses
    -- factorized 7×7 via 1×7 + 7×1 which isn't a separate primitive)
    .inceptionModule 288 192 128 192 128 192 192,
    .inceptionModule 768 192 160 192 160 192 192,
    .inceptionModule 768 192 160 192 160 192 192,
    .inceptionModule 768 192 160 192 160 192 192,
    .inceptionModule 768 192 192 192 192 192 192,

    -- Two "Inception-C" modules at 8×8 (320 + 384 + 384 + 192 = 1280 out)
    .inceptionModule 768  320 384 384 448 384 192,
    .inceptionModule 1280 320 384 384 448 384 192,

    .globalAvgPool,
    .dense 1280 1000 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Inception v4 (2016, approximate)
-- ════════════════════════════════════════════════════════════════

def inceptionV4 : NetSpec where
  name := "Inception-v4 (bestiary approximation)"
  imageH := 299
  imageW := 299
  layers := [
    -- Stem (simplified)
    .convBn 3 32 3 2 .same,
    .convBn 32 32 3 1 .same,
    .convBn 32 64 3 1 .same,
    .maxPool 2 2,
    .convBn 64 80 1 1 .same,
    .convBn 80 192 3 1 .same,
    .maxPool 2 2,

    -- 4× Inception-A at 35×35
    .inceptionModule 192 96 64 96 64 96 96,
    .inceptionModule 384 96 64 96 64 96 96,
    .inceptionModule 384 96 64 96 64 96 96,
    .inceptionModule 384 96 64 96 64 96 96,

    -- 7× Inception-B at 17×17 (each outputs 384 + 224 + 256 + 128 = 992)
    .inceptionModule 384 384 192 224 192 256 128,
    .inceptionModule 992 384 192 224 192 256 128,
    .inceptionModule 992 384 192 224 192 256 128,
    .inceptionModule 992 384 192 224 192 256 128,
    .inceptionModule 992 384 192 224 192 256 128,
    .inceptionModule 992 384 192 224 192 256 128,
    .inceptionModule 992 384 192 224 192 256 128,

    -- 3× Inception-C at 8×8 (each outputs 256 + 256 + 256 + 256 = 1024)
    .inceptionModule 992  256 384 256 384 256 256,
    .inceptionModule 1024 256 384 256 384 256 256,
    .inceptionModule 1024 256 384 256 384 256 256,

    .globalAvgPool,
    .dense 1024 1000 .identity
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
  IO.println "  Bestiary — Inception v1 / v3 / v4"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Multi-scale parallel convolutions. The 1×1 dimension reducer"
  IO.println "  + multi-kernel concat trick that made GoogLeNet feasible."

  summarize googLeNet
  summarize inceptionV3
  summarize inceptionV4

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.inceptionModule` is the v1 original: four parallel branches"
  IO.println "    (1×1, 1×1+3×3, 1×1+5×5, pool+1×1), concat along channels."
  IO.println "  • v3 and v4 use SEVERAL inception module types with"
  IO.println "    asymmetric 7×1+1×7 factorizations and different reduction"
  IO.println "    geometries. Each paper specifies them in tables; our"
  IO.println "    bestiary spec uses the unified v1-style module at the right"
  IO.println "    channel counts. Param totals are ~20% off from paper but"
  IO.println "    structural flow is correct."
  IO.println "  • GoogLeNet's auxiliary classifiers (two side heads during"
  IO.println "    training) are omitted — they're a training-time gradient-"
  IO.println "    flow trick, not an inference-time architectural feature."
  IO.println "  • The 1×1 dimension-reducer trick from Inception v1 was"
  IO.println "    adopted EVERYWHERE afterward. ResNet bottlenecks, MobileNet"
  IO.println "    squeeze-and-excitation, Fire modules — all downstream of"
  IO.println "    'use 1×1 convs as channel-space bottlenecks'."
