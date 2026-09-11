import LeanMlir.Spec

/-! # FourCastNet — Bestiary entry

Pathak, Subramanian, Harrington, Raja, Chattopadhyay, Mardani, Kurth, Hall,
Li, Azizzadenesheli, Hassanzadeh, Kashinath & Anandkumar, "FourCastNet: a
global data-driven high-resolution weather model using adaptive Fourier
neural operators", 2022 (arXiv:2202.11214).

**A ViT whose attention is a Fourier mixer.** The global weather state on
ERA5's 0.25° grid — 720 × 1440 pixels, 20 variables — is cut into 8 × 8
patches (16,200 tokens of dimension 768) and pushed through 12 transformer
blocks whose token mixing is not attention but the **adaptive Fourier neural
operator** (AFNO, Guibas et al.\ 2022, arXiv:2111.13587): FFT over the token
grid, a block-diagonal complex two-layer MLP applied per mode with soft
thresholding, inverse FFT. That is the FNO's spectral convolution made
data-dependent and sparse, and it costs O(N log N) in tokens where
attention costs O(N²) — at 16,200 tokens the difference is the model. A
linear head maps each token back to its 8 × 8 × 20 patch, and the network
is trained to predict the state six hours ahead; rollouts of that map are
the forecast.

The AFNO mixer is the one primitive here our `NetSpec` lacks (the same
spectral convolution FNO's entry counts in prose). The spec below shows the
backbone with the `.transformerEncoder` primitive standing in for the AFNO
block — same patch embedding, dimension, depth and MLP, with attention where
the Fourier mixer goes. ⚠ That stand-in OVERCOUNTS: an attention block at
768 spends 4 × 768² ≈ 2.36 M parameters on Q, K, V and the output projection,
where AFNO's block-diagonal weights (8 blocks of 96 × 96, two layers, complex)
are ≈ 0.30 M. The MLPs, which are ¾ of a block either way, are the same.

## Variants

- `fourCastNet`     — the paper's model: 720 × 1440 × 20, patch 8, dim 768,
                      12 blocks, MLP 3072, head to 8·8·20 = 1280 per token.
- `fourCastNetLite` — a 2° version: 90 × 180 grid, 4 variables, patch 6,
                      dim 256, 6 blocks. The shape at a size one card trains.
- `tinyFourCastNet` — 16 × 32 grid, 4 variables, patch 4, dim 32, 2 blocks.
-/

-- ════════════════════════════════════════════════════════════════
-- § FourCastNet (AFNO shown as an attention encoder — see the header)
-- ════════════════════════════════════════════════════════════════
-- 720/8 × 1440/8 = 90 × 180 = 16,200 patches. The head is per-token in the
-- paper (each token reconstructs its own patch); the count is the same.

def fourCastNet : NetSpec where
  name   := "FourCastNet (AFNO-ViT, 0.25°, attention standing in for AFNO)"
  imageH := 720
  imageW := 1440
  layers := [
    .patchEmbed 20 768 8 16200,           -- 20 variables, 8×8 patches → 16,200 tokens of 768
    .transformerEncoder 768 12 3072 12,   -- 12 blocks; the paper's mixer is AFNO, not attention
    .dense 768 1280 .identity             -- back to an 8×8×20 patch
  ]

-- ════════════════════════════════════════════════════════════════
-- § A 2° version
-- ════════════════════════════════════════════════════════════════

def fourCastNetLite : NetSpec where
  name   := "FourCastNet-lite (2°, 90x180x4, dim 256, 6 blocks)"
  imageH := 90
  imageW := 180
  layers := [
    .patchEmbed 4 256 6 450,              -- 15 × 30 = 450 tokens
    .transformerEncoder 256 8 1024 6,
    .dense 256 144 .identity              -- 6×6×4 patch
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny fixture
-- ════════════════════════════════════════════════════════════════

def tinyFourCastNet : NetSpec where
  name   := "tiny-FourCastNet (16x32x4, dim 32, 2 blocks)"
  imageH := 16
  imageW := 32
  layers := [
    .patchEmbed 4 32 4 32,                -- 4 × 8 = 32 tokens
    .transformerEncoder 32 4 128 2,
    .dense 32 64 .identity                -- 4×4×4 patch
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — FourCastNet"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  A ViT on the global weather grid whose token mixer is an"
  IO.println "  adaptive Fourier neural operator; trained to step 6 h ahead."

  fourCastNet.summarize (size := .image) (unit := .millions)
  fourCastNetLite.summarize (size := .image) (unit := .millions)
  tinyFourCastNet.summarize (size := .image) (unit := .thousands)

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ONE new primitive, the AFNO mixer, shown here with"
  IO.println "    .transformerEncoder in its place: same tokens, dim, depth"
  IO.println "    and MLP, attention where the Fourier mixer goes."
  IO.println "  • That OVERCOUNTS by ~2 M per block (attention's 4·768² vs"
  IO.println "    AFNO's block-diagonal 0.3 M); the MLPs are ¾ of a block"
  IO.println "    either way."
  IO.println "  • Attention at 16,200 tokens is O(N²); AFNO is O(N log N)."
  IO.println "    At this token count that difference is the model."
