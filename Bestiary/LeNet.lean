import LeanMlir

/-! # LeNet — Bestiary entry

LeNet (LeCun, Bottou, Bengio, Haffner, 1998 — "Gradient-Based Learning
Applied to Document Recognition") is **the** original CNN — the paper
that anchors modern deep learning's lineage. Yann LeCun's group built
it in 1989 and refined it for the 1998 paper to recognize handwritten
digits on bank cheques. A $\sim$60K-parameter network running on 32$\times$32
grayscale images, 99%+ accuracy on MNIST.

Everything in modern CNNs — conv layers, pooling, stacking depth,
learned features instead of hand-designed — is already here. The
architecture looks comically simple now, but in 1998 the paper had to
spend pages defending that convolutions were a useful structural
prior. Thirty years later every ResNet you see is still, at the
abstract shape level, a LeNet with more blocks.

## LeNet-5 (the convolutional one)

```
    Input: 1 × 32 × 32 (grayscale digit)
         │
         ▼
    Conv 5×5, 6 filters       → 6  × 28 × 28
    AvgPool 2×2               → 6  × 14 × 14
         │
         ▼
    Conv 5×5, 16 filters      → 16 × 10 × 10
    AvgPool 2×2               → 16 × 5  × 5
         │
         ▼
    Flatten                   → 400
    Dense 400 → 120
    Dense 120 → 84
    Dense 84  → 10
```

The paper actually uses subsampling with learned scalars (each pool
channel had a bias and trainable scaling) plus tanh activations
throughout. We use maxPool + ReLU because the param counts are
almost identical and nobody hand-tunes tanh-sigmoid networks in 2026.

## LeNet-300-100 (the MLP version)

Same paper also has a pure-MLP baseline: 784 pixels → 300 → 100 → 10.
Zero convolutions, ~266K params, ~1.5% error on MNIST (worse than
LeNet-5). Useful fixture for showing the MLP-vs-CNN comparison without
any frill.
-/

-- ════════════════════════════════════════════════════════════════
-- § LeNet-5 (the conv one)
-- ════════════════════════════════════════════════════════════════

def leNet5 : NetSpec where
  name := "LeNet-5"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d 1 6 5 .valid .relu,          -- 32×32 → 28×28
    .maxPool 2 2,                          -- 28×28 → 14×14
    .conv2d 6 16 5 .valid .relu,          -- 14×14 → 10×10
    .maxPool 2 2,                          -- 10×10 → 5×5
    .flatten,
    .dense (16 * 5 * 5) 120 .relu,         -- 400 → 120
    .dense 120 84 .relu,
    .dense 84 10 .identity                  -- 10-class MNIST output
  ]

-- ════════════════════════════════════════════════════════════════
-- § LeNet-300-100 (the MLP baseline)
-- ════════════════════════════════════════════════════════════════

def leNet300_100 : NetSpec where
  name := "LeNet-300-100 (MLP baseline)"
  imageH := 28
  imageW := 28
  layers := [
    .flatten,
    .dense (28 * 28) 300 .relu,
    .dense 300 100 .relu,
    .dense 100 10 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000}K)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — LeNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  The grandfather of modern CNNs. 1998. Yann LeCun on bank"
  IO.println "  cheques. 60K parameters, 99%+ MNIST accuracy."

  summarize leNet5
  summarize leNet300_100

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Zero new Layer primitives. LeNet is the ground truth; every"
  IO.println "    later CNN in the bestiary is a LeNet with more expressive"
  IO.println "    tricks bolted on."
  IO.println "  • Paper uses tanh activations + learned subsampling; we use"
  IO.println "    ReLU + maxPool. Param counts and training trajectory are"
  IO.println "    essentially identical with the modern substitution."
  IO.println "  • LeNet-5 is ~60K params; LeNet-300-100 is ~266K. The MLP"
  IO.println "    baseline is 4× bigger and performs worse: direct evidence"
  IO.println "    for 'convolution is a useful structural prior' — the whole"
  IO.println "    point of the 1998 paper."
  IO.println "  • If you trained this on an N6800 GPU today it would finish"
  IO.println "    in under a second per epoch."
