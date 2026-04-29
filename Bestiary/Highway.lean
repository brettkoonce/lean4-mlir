import LeanMlir

/-! # Highway Networks — Bestiary entry

Highway Networks (Srivastava, Greff, Schmidhuber 2015,
[arXiv:1505.00387](https://arxiv.org/abs/1505.00387)) are the
gating-style precursor to ResNet. The architecture predates He et
al.'s residual networks by ~6 months and motivates them: a learned
per-element bypass that lets very deep stacks train.

The block is two parallel paths — a *main path* `H(x)` (any
transformation, typically dense + ReLU) and a *transform gate*
`T(x)` (its own dense + sigmoid, output in `[0, 1]`) — recombined
as

    y = T(x) * H(x) + (1 - T(x)) * x.

`T` learned per element gives a continuous knob between
``pass through identity'' (`T → 0`) and ``transform fully''
(`T → 1`). The gradient through the blend is exactly the product
rule + additive fan-in from Chapter 2's VJP toolkit, so no new
calculus is needed.

```
                  Input x
                     │
              ┌──────┴──────┐
              ▼             ▼
         Main: H(x)    Gate: T(x)
         (dense+ReLU)  (dense+sigmoid)
              │             │
              └──────┬──────┘
                     ▼
        y = T·H + (1−T)·x          ← carry path bypasses x verbatim
```

ResNet (Chapter 6) is the special case where `T ≡ 1` on the
transform side and the carry path becomes a constant identity skip
— same calculus, fewer parameters, marginally better training.
Highway showed in 2015 that *some kind of bypass* made very deep
networks trainable; ResNet showed half a year later that the gate
could be a constant.

Our `NetSpec` is a linear list of layers, so we represent the two
paths as two independent specs in the same body-sharing style as
the AlphaZero / MuZero entries. The combination `y = T·H + (1−T)·x`
is the architectural story; the calculus is in the book.

## Variants

- `highway50Main` / `highway50Gate` — canonical Highway-MLP from the
  paper, 50-dim hidden width, used for the 50-layer experiments.
- `highway100Main` / `highway100Gate` — same shape, 100-dim variant
  used for the deeper 100-layer experiments.
- `tinyHighwayMain` / `tinyHighwayGate` — 8-dim fixture, useful for
  quick architectural inspection / testing.

All are pure `NetSpec` values; no training runs here. The bestiary
uses these as read-only architecture examples.
-/

-- ════════════════════════════════════════════════════════════════
-- § Highway-MLP, 50-dim hidden (paper canonical)
-- ════════════════════════════════════════════════════════════════

/-- Main path H(x): dense → ReLU. -/
def highway50Main : NetSpec where
  name := "Highway-50 — main path H(x)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 50 50 .relu
  ]

/-- Transform gate T(x): dense; sigmoid is applied at the blend
    (not in the spec). The gate has its own learned weights, the
    same shape as the main path. -/
def highway50Gate : NetSpec where
  name := "Highway-50 — transform gate T(x)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 50 50 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Highway-MLP, 100-dim hidden (deeper experiments)
-- ════════════════════════════════════════════════════════════════

def highway100Main : NetSpec where
  name := "Highway-100 — main path H(x)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 100 100 .relu
  ]

def highway100Gate : NetSpec where
  name := "Highway-100 — transform gate T(x)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 100 100 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny Highway (fixture)
-- ════════════════════════════════════════════════════════════════

def tinyHighwayMain : NetSpec where
  name := "tiny-Highway — main path H(x)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 8 8 .relu
  ]

def tinyHighwayGate : NetSpec where
  name := "tiny-Highway — transform gate T(x)"
  imageH := 1
  imageW := 1
  layers := [
    .dense 8 8 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary of every Bestiary entry in this file.
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams}"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK (channel dims chain cleanly)"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Highway Networks"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Two-path block: main H(x) + transform gate T(x)."
  IO.println "  Combined as y = T·H + (1−T)·x — product rule + additive fan-in,"
  IO.println "  both already in Ch 2's VJP toolkit. ResNet (Ch 6) is T ≡ 1."

  summarize highway50Main
  summarize highway50Gate
  summarize highway100Main
  summarize highway100Gate
  summarize tinyHighwayMain
  summarize tinyHighwayGate

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Each main / gate pair shares the same input shape but learns"
  IO.println "    independent weights. The blend y = T·H + (1−T)·x lives in"
  IO.println "    the orchestration, not in NetSpec — a linear list of layers"
  IO.println "    can't express the per-element gate at the spec level."
  IO.println "  • Sigmoid on the gate is applied at the blend, not in the spec."
  IO.println "    Same convention as the AlphaZero value head's tanh."
  IO.println "  • ResNet (Ch 6) replaced T with a constant 1 — same calculus,"
  IO.println "    fewer parameters. The bestiary entry pairs the two for"
  IO.println "    pedagogical contrast."
