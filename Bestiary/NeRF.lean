import LeanMlir

/-! # NeRF — Bestiary entry

NeRF (Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng, 2020 —
"NeRF: Representing Scenes as Neural Radiance Fields for View
Synthesis") is the "it's literally just an MLP" paper. A tiny,
8-layer fully-connected network, pointed at the right problem with
the right input preprocessing, learns to represent a 3D scene from a
few dozen photos + camera poses. No convolutions. No attention. No
3D data structure. Just query the MLP for every pixel along every
ray and composite.

## The architectural secret

```
      (x, y, z) ∈ ℝ³             (θ, φ) viewing direction
            │                             │
            ▼                             ▼
     γ(·) positional encoding       γ(·) positional encoding
     (L = 10 frequencies)            (L = 4 frequencies)
            │                             │
            ▼                             │
       60-dim input                       │ 24-dim
            │                             │
    ┌───────┴─────┐                       │
    │ FC 60 → 256 │                       │
    │    + ReLU   │                       │
    └───────┬─────┘                       │
            ▼                             │
       FC 256 → 256                       │
       ... (4 layers)                     │
            │                             │
       ─── concat γ(x) ─── (skip at layer 5)
            │                             │
       FC 316 → 256                       │
       ... (3 more layers, 8 total)       │
            │                             │
      ┌─────┴─────┐                       │
      ▼           ▼                       │
  σ head      FC 256 → 256                │
  (density)       │                       │
                  ▼                       │
               [concat γ(d)]  ──── 280-dim
                  │
               FC 280 → 128 + ReLU
                  │
               FC 128 → 3   (RGB)
```

The **entire architecture** is described by the diagram above — 11 FC
layers total, under 1M params. What makes NeRF work is not the network;
it's the positional encoding (no encoding → blurry mush) + the
**volumetric rendering loss** that composites densities along rays.
Both of those are ingredients around the network, not in it.

## Why positional encoding matters

An MLP with ReLU activations is a piecewise-linear function. It cannot
represent high-frequency signals (sharp edges, fine textures) on its
own. The sinusoidal positional encoding
$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))$
lifts a low-dim coordinate into a high-dim, high-frequency feature
space where the MLP has enough wiggle room to interpolate sharp
details. Same basis as transformer positional embeddings; different
use case.

## Variants

- `nerf` — canonical paper architecture (L_pos=10, L_dir=4, 8×256 hidden).
  Params: ~528K per network. Full NeRF trains two such networks
  (coarse + fine) for hierarchical sampling.
- `nerfFast` — smaller hidden dim (128) for faster training.
- `tinyNeRF` — minimalist fixture.
-/

-- ════════════════════════════════════════════════════════════════
-- § NeRF (canonical Mildenhall 2020)
-- ════════════════════════════════════════════════════════════════

/-- Canonical NeRF: positional encoding of (x, y, z) with L=10, plus
    the dual-conditioned MLP. 256 hidden units, 8 layers, ~528K params.

    NOTE: our linear NetSpec can't express the direction-input branch
    (which enters the MLP after the density head); the `.nerfMLP`
    primitive bundles the whole MLP structure including that branch,
    so the NetSpec just shows one positional encoding + one MLP. The
    direction input and its encoding are handled internally. -/
def nerf : NetSpec where
  name := "NeRF (canonical)"
  imageH := 1
  imageW := 1
  layers := [
    -- Positional encoding of 3D position (3 coords × 2 × 10 freqs = 60-dim)
    .positionalEncoding 3 10,
    -- 8-layer MLP with mid-skip and dual output heads (density + RGB).
    -- encodedDirDim = 2·2·4 = 16 (NeRF uses 2D direction; 4 frequencies)
    .nerfMLP 60 16 256
  ]

-- ════════════════════════════════════════════════════════════════
-- § NeRF-fast (smaller hidden)
-- ════════════════════════════════════════════════════════════════

def nerfFast : NetSpec where
  name := "NeRF-fast (hidden=128)"
  imageH := 1
  imageW := 1
  layers := [
    .positionalEncoding 3 10,
    .nerfMLP 60 16 128
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyNeRF fixture
-- ════════════════════════════════════════════════════════════════

def tinyNeRF : NetSpec where
  name := "tiny-NeRF"
  imageH := 1
  imageW := 1
  layers := [
    .positionalEncoding 3 4,                    -- just 4 frequencies (24-dim)
    .nerfMLP 24 8 64                             -- 64 hidden units
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000}K)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — NeRF"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Neural Radiance Fields: an MLP that learns a 3D scene from"
  IO.println "  photos. The whole network fits in a page. The magic is in"
  IO.println "  positional encoding + volumetric rendering, not the architecture."

  summarize nerf
  summarize nerfFast
  summarize tinyNeRF

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • `.positionalEncoding (d L)` is the sinusoidal frequency basis"
  IO.println "    (Vaswani 2017 / NeRF 2020): output dim = d · 2 · L, zero"
  IO.println "    parameters. Same primitive shows up in transformers, NeRF,"
  IO.println "    SIREN, etc."
  IO.println "  • `.nerfMLP (encodedPosDim encodedDirDim hiddenDim)` bundles"
  IO.println "    the paper's specific 8-layer MLP with mid-skip at layer 5"
  IO.println "    and dual density + RGB heads. The direction input branch"
  IO.println "    enters internally — NetSpec's linear shape doesn't see it."
  IO.println "  • Total params ≈ 528K at canonical config. Full NeRF pipelines"
  IO.println "    train TWO of these (coarse + fine) for hierarchical sampling,"
  IO.println "    so ~1M trainable params total."
  IO.println "  • What NeRF the PAPER is really about: volumetric rendering"
  IO.println "    (alpha-composite samples along rays) + positional encoding."
  IO.println "    Both sit OUTSIDE the network — inputs and loss, not layers."
  IO.println "    This is the bestiary's clearest \"the architecture is not"
  IO.println "    where the innovation is\" entry."
  IO.println "  • Successors (mip-NeRF, Instant-NGP, Plenoxels, Gaussian"
  IO.println "    Splatting) keep the volumetric-rendering idea but swap the"
  IO.println "    MLP for faster representations (hash grids, voxel grids,"
  IO.println "    literal 3D Gaussians). The MLP was a means, not an end."
