import LeanMlir

/-! ViT-Tiny on Imagenette with the **Muon** optimizer (UNVERIFIED perf path).

    Step-1 demo from `planning/muon.md`: Muon (Newton–Schulz polar projection) on
    every 2D weight matrix — the Q/K/V/O projections and the two MLP layers of all
    12 transformer blocks — with AdamW on the edges (patch-embed conv, CLS token,
    positional embedding, every LayerNorm γ/β, biases, and the small classifier head).
    Identical architecture + recipe to `MainVitTrain` (the AdamW baseline) so the two
    are a compute-matched A/B on the same net — the comparison `planning/muon.md`'s
    ViT section cashes out.

    Patch 16×16 → 192-dim, 12 blocks, 3 heads, MLP 768. ~5.5M params, 224×224, 10 classes.

    The generated train-step module is **signature-identical to the AdamW module**
    (same m/v optimizer-state slots, same `%lr`/`%t` scalars): Muon stores its momentum
    in the m-slot and passes the v-slot through. So the host drives it through the
    existing Adam FFI path — hence `useAdam := true` AND `useMuon := true` below. The
    only behavioral change is internal to codegen (2D weights routed to Muon). -/

def vitTiny : NetSpec where
  name := "ViT-Tiny-Muon"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity               -- classification head (1D-ish → AdamW)
  ]

def vitTinyMuonConfig : TrainConfig where
  learningRate := 0.0003
  batchSize    := 32
  epochs       := 80
  useAdam      := true      -- host state path (m+v, %t) — the Muon module is Adam-signature
  useMuon      := true      -- codegen routes 2D weight matrices to Newton–Schulz
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 5
  augment      := true
  labelSmoothing := 0.1

def main (args : List String) : IO Unit :=
  vitTiny.train vitTinyMuonConfig (args.head?.getD "data/imagenette")
