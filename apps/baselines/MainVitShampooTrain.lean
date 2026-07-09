import LeanMlir

/-! ViT-Tiny on Imagenette with the **Shampoo** optimizer (UNVERIFIED perf path).

    Step-2 demo from `planning/shampoo.md`: Shampoo (Kronecker `L^{-1/4}·G·R^{-1/4}`)
    on every SQUARE 2D weight matrix — the Q/K/V/O projections (`[192,192]`) of all
    12 transformer blocks — with AdamW on everything else (patch-embed conv, CLS
    token, positional embedding, LayerNorm γ/β, biases, the two non-square MLP
    weights `[192,768]`/`[768,192]`, and the small classifier head). Identical
    architecture + recipe to `MainVitTrain` (AdamW) and `MainVitMuonTrain` (Muon),
    so the three are a compute-matched 3-way A/B on the same net — the ladder point
    `planning/shampoo.md` cashes out: AdamW (diagonal) vs Muon (one-step polar =
    single-step Shampoo) vs Shampoo (polar WITH accumulated memory).

    Patch 16×16 → 192-dim, 12 blocks, 3 heads, MLP 768. ~5.5M params, 224×224, 10 classes.

    Like Muon, the generated module is **signature-identical to the AdamW module**:
    Shampoo's L/R accumulators reuse the m/v slots (the routed weights are square,
    so `[n,n]` L/R match the `[n,n]` slots). So the host drives it through the
    existing Adam FFI — hence `useAdam := true` AND `useShampoo := true`.

    Shampoo's effective step scale differs from Adam's; the LR here (`3e-4`, same as
    the Muon/AdamW baselines) is the A/B's *matched* starting point — sweep it per
    `planning/shampoo.md §3` if the matched run underperforms. -/

def vitTiny : NetSpec where
  name := "ViT-Tiny-Shampoo"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity               -- classification head (non-square → AdamW)
  ]

def vitTinyShampooConfig : TrainConfig where
  learningRate := 0.0003
  batchSize    := 32
  epochs       := 80
  useAdam      := true      -- host state path (m+v, %t) — the Shampoo module is Adam-signature
  useShampoo   := true      -- codegen routes SQUARE 2D weight matrices to Shampoo
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true
  labelSmoothing := 0.1

def main (args : List String) : IO Unit :=
  vitTiny.train vitTinyShampooConfig (args.head?.getD "data/imagenette")
