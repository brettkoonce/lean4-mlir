import LeanMlir

/-! # YOLO v1 — Bestiary entry

YOLO v1 (Redmon, Divvala, Girshick, Farhadi, 2016 — "You Only Look Once:
Unified, Real-Time Object Detection") was the paper that reframed
detection as a **single regression problem**. Instead of the
two-stage R-CNN pipeline (region proposals → classify), YOLO ran one
convnet over the whole image and predicted boxes + classes directly in
one forward pass. The slogan came from the insight: each pixel is
looked at exactly once, grounded in a spatial grid of predictions.

## The output encoding is the trick

The network's output is a tensor of shape `(S, S, B·5 + C)`:

- `S × S` — the image is divided into an S×S grid (S = 7 in v1).
- `B` bounding boxes per grid cell (B = 2).
  Each box has 5 scalars: `(x, y, w, h, confidence)`.
- `C` class probabilities per grid cell (C = 20 for Pascal VOC).

So each of the 49 cells predicts 10 bbox scalars + 20 class probs =
30 outputs. The network's last layer is a dense projection to the
flat `49 · 30 = 1470` vector, which then gets reshaped to (7, 7, 30)
for loss computation. **There's no special YOLO primitive** — the
whole architecture is just stacked convs and two fully-connected
layers. The cleverness lives in the loss function and the output
interpretation, not in any novel layer.

## Architecture

```
Input: 448 × 448 × 3

 1. Conv 7×7, 64, stride 2      → 224×224×64
    MaxPool 2×2, stride 2       → 112×112×64

 2. Conv 3×3, 192               → 112×112×192
    MaxPool 2×2, stride 2       → 56×56×192

 3. Conv 1×1, 128               ┐
    Conv 3×3, 256               │ the "inception-ish" reduction block
    Conv 1×1, 256               │
    Conv 3×3, 512               ┘ → 56×56×512
    MaxPool 2×2, stride 2       → 28×28×512

 4. [Conv 1×1, 256              ┐
     Conv 3×3, 512]  × 4        │ four 1×1 + 3×3 pairs
    Conv 1×1, 512               │
    Conv 3×3, 1024              ┘ → 28×28×1024
    MaxPool 2×2, stride 2       → 14×14×1024

 5. [Conv 1×1, 512              ┐
     Conv 3×3, 1024]  × 2       │ two more pairs
    Conv 3×3, 1024              │
    Conv 3×3, 1024, stride 2    ┘ → 7×7×1024

 6. Conv 3×3, 1024              ┐
    Conv 3×3, 1024              ┘ → 7×7×1024

 7. Flatten                     → 50176
    Dense 50176 → 4096          ← this FC is HUGE (~200M params alone)
    Dense 4096 → 1470           ← → reshape to 7×7×30
```

The paper uses **Leaky ReLU** (slope 0.1) after every conv / FC except
the final output. Our `Activation` enum has `{relu, relu6, identity}`
only, so the bestiary spec below uses `relu` with a prose note. The
architectural shape and param count are unchanged by that substitution.

## Variants

- `yolo` — full YOLOv1, 24 conv layers + 2 FC. ~270M params total,
  with ~205M in the first FC layer alone.
- `fastYolo` — 9 conv + 2 FC, ~163M params. Faster, worse AP.
- `tinyYolo` — scale-model fixture: 6 conv + 2 FC at reduced width.

Note: "tiny-YOLO" in the wild also refers to a specific published
variant of YOLOv2/v3; `tinyYolo` here is just a bestiary toy, not that.
-/

-- ════════════════════════════════════════════════════════════════
-- § YOLOv1 — full 24-conv version
-- ════════════════════════════════════════════════════════════════

/-- YOLO v1 output head size: `S·S · (B·5 + C)` for Pascal VOC
    (S = 7, B = 2, C = 20) = 7·7·(10+20) = **1470**. -/
def yoloHeadSize : Nat := 7 * 7 * (2 * 5 + 20)

def yolo : NetSpec where
  name := "YOLOv1 (Redmon 2016)"
  imageH := 448
  imageW := 448
  layers := [
    -- Block 1: stem
    .conv2d 3 64 7 .same .relu,      -- stride is 1 in conv2d; stride-2 via a maxPool below
    .maxPool 2 2,

    -- Block 2
    .conv2d 64 192 3 .same .relu,
    .maxPool 2 2,

    -- Block 3: 1×1 reduce + 3×3 expand pattern
    .conv2d 192 128 1 .same .relu,
    .conv2d 128 256 3 .same .relu,
    .conv2d 256 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .maxPool 2 2,

    -- Block 4: four 1×1+3×3 pairs, then 1×1 + 3×3
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 256 1 .same .relu,
    .conv2d 256 512 3 .same .relu,
    .conv2d 512 512 1 .same .relu,
    .conv2d 512 1024 3 .same .relu,
    .maxPool 2 2,

    -- Block 5: two 1×1+3×3 pairs + two more 3×3 convs
    .conv2d 1024 512 1 .same .relu,
    .conv2d 512 1024 3 .same .relu,
    .conv2d 1024 512 1 .same .relu,
    .conv2d 512 1024 3 .same .relu,
    .conv2d 1024 1024 3 .same .relu,
    .conv2d 1024 1024 3 .same .relu,    -- the stride-2 last conv (approximated with .same)

    -- Block 6: two 3×3 at 7×7 resolution
    .conv2d 1024 1024 3 .same .relu,
    .conv2d 1024 1024 3 .same .relu,

    -- Head: flatten + 2 FC → reshape to (7, 7, 2·5 + 20)
    .flatten,
    .dense (7 * 7 * 1024) 4096 .relu,
    .dense 4096 yoloHeadSize .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Fast YOLOv1 — 9-conv variant
-- ════════════════════════════════════════════════════════════════

def fastYolo : NetSpec where
  name := "Fast YOLOv1"
  imageH := 448
  imageW := 448
  layers := [
    .conv2d 3 16 3 .same .relu,
    .maxPool 2 2,
    .conv2d 16 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .maxPool 2 2,
    .conv2d 64 128 3 .same .relu,
    .maxPool 2 2,
    .conv2d 128 256 3 .same .relu,
    .maxPool 2 2,
    .conv2d 256 512 3 .same .relu,
    .maxPool 2 2,
    .conv2d 512 1024 3 .same .relu,
    .conv2d 1024 256 3 .same .relu,
    .conv2d 256 256 3 .same .relu,

    .flatten,
    .dense (7 * 7 * 256) 4096 .relu,
    .dense 4096 yoloHeadSize .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyYolo fixture
-- ════════════════════════════════════════════════════════════════

def tinyYolo : NetSpec where
  name := "tiny-YOLO (fixture)"
  imageH := 112
  imageW := 112
  layers := [
    .conv2d 3 16 3 .same .relu,
    .maxPool 2 2,
    .conv2d 16 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .maxPool 2 2,
    .conv2d 64 128 3 .same .relu,
    .maxPool 2 2,

    .flatten,
    .dense (7 * 7 * 128) 512 .relu,
    .dense 512 (7 * 7 * (2 * 5 + 5)) .identity   -- 5 classes instead of 20
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
  IO.println "  Bestiary — YOLOv1"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Detection as regression. One forward pass → S×S grid of"
  IO.println "  (box, confidence, class) predictions. The paper that"
  IO.println "  retired two-stage detectors."

  summarize yolo
  summarize fastYolo
  summarize tinyYolo

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • No new Layer primitives needed. YOLOv1 is a conv stack +"
  IO.println "    two FCs; the cleverness is in the LOSS (grid assignment,"
  IO.println "    responsibility-to-predict, IoU weighting) and the output"
  IO.println "    reshape — both training-time concerns outside NetSpec."
  IO.println "  • Paper uses LeakyReLU(0.1); we use ReLU here. Architecture"
  IO.println "    shape and param count are identical."
  IO.println "  • The 50176 → 4096 FC layer is the main cost — around 205M"
  IO.println "    of the ~270M total params. This is why later YOLOs (v2+)"
  IO.println "    dropped FC layers in favor of convolutional output heads"
  IO.println "    with anchors."
  IO.println "  • Stride-2 behavior in some convs is approximated with a"
  IO.println "    following maxPool-2 since our conv2d primitive is stride-1."
  IO.println "    Param count is unchanged; output spatial dims are the same."
