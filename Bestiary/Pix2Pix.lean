import LeanMlir

/-! # Pix2Pix — Bestiary entry

Pix2Pix (Isola, Zhu, Zhou, Efros, CVPR 2017 --- "Image-to-Image
Translation with Conditional Adversarial Networks") is the direct
ancestor of CycleGAN, from the same lab a few months earlier. It's
the paper that established the template: \textbf{UNet generator +
PatchGAN discriminator}, trained with adversarial loss + an L1
reconstruction term for paired data.

The catch is exactly that --- paired data. Pix2Pix requires matched
before/after images: a sketch and its photograph, a map and its
aerial view, a black-and-white frame and its colorized version. For
most interesting translation problems you can't collect pairs at
scale, which is why CycleGAN landed later that year with the cycle-
consistency trick for unpaired data. But when you \emph{have} pairs,
Pix2Pix is simpler and better --- the L1 reconstruction loss is a
direct training signal that cycle consistency only approximates.

## Anatomy

```
    Source image (256×256×3)
         │
         ▼    Encoder: 8 × strided-conv (1/2 spatial, 2× channels)
         │
         ▼         bottleneck at 1×1×512
         │
         ▼    Decoder: 8 × transposed-conv + skip (mirror)
         │
         ▼
    Target image (256×256×3)

    Discriminator: 70×70 PatchGAN (see CycleGAN.lean)
    Loss: L_GAN(G, D) + λ · L_1(G(x), y)     [paired data]
```

The 8-level UNet has channel progression
$64 \to 128 \to 256 \to 512 \to 512 \to 512 \to 512 \to 512$ in the
encoder, mirrored symmetrically in the decoder. Each pair of
symmetric layers gets a skip connection --- classic Ronneberger
UNet, just deeper and used generatively. The bottleneck at
$1 \times 1 \times 512$ is where the full $256 \times 256$ image
has been compressed to a single 512-dim vector; the skips carry
everything else.

## Pix2Pix vs CycleGAN

| Aspect              | Pix2Pix (2017)           | CycleGAN (2017)            |
|---------------------|--------------------------|----------------------------|
| Data                | Paired (x, y)            | Unpaired (X, Y) sets        |
| Generator           | UNet, $\sim$54M           | 9-ResBlock ResNet, $\sim$11M |
| Discriminator       | PatchGAN                 | PatchGAN                   |
| Loss                | GAN + L1 reconstruction  | GAN + cycle consistency    |
| Use case            | Sketch→photo, map→aerial | Horse↔zebra, summer↔winter |
| Year                | Jan 2017                 | Oct 2017                   |

Same lab, 9 months apart. The CycleGAN paper explicitly starts from
Pix2Pix and asks ``what if we relax the paired-data requirement'' ---
so this entry is the reference for that relaxation.

## Hardware context

Pix2Pix came out when consumer GPUs had 8--12GB VRAM (GTX 1080 /
1080 Ti era). Training a 54M UNet on $256 \times 256$ images meant
batch size 1 or 2; the paper explicitly calls this out as a design
constraint. A couple years later 12--24GB GPUs were commodity and
the constraint evaporated, but the ``start with a small
architecture that fits'' habit is still visible across the early
image-translation literature.

## Variants

- `pix2pixGenerator`      --- 8-level UNet generator ($\sim$70M in our
                              approximation, paper reports $\sim$54M --- see
                              NetSpec simplifications below)
- `pix2pixDiscriminator`  --- PatchGAN discriminator (same as
                              \texttt{cycleganDiscriminator})
- `tinyPix2Pix_G` / `_D`  --- fixture

## NetSpec simplifications

- Our \texttt{.unetDown} / \texttt{.unetUp} primitives use 2 convs
  per level (Ronneberger style); Pix2Pix uses 1 strided conv per
  level. Our approximation over-counts by $\sim$30\%. Architecture
  shape (8 levels, channel progression, skip connections) is correct.
- Transposed convs in the decoder are bundled inside \texttt{.unetUp}
  (upsample + concat-skip + 2 convs). Param accounting matches paper
  shape even if the exact per-layer breakdown differs.
-/

-- ════════════════════════════════════════════════════════════════
-- § Pix2Pix generator (8-level UNet, 256×256 RGB → 256×256 RGB)
-- ════════════════════════════════════════════════════════════════

def pix2pixGenerator : NetSpec where
  name := "Pix2Pix generator (8-level UNet, 256×256 RGB)"
  imageH := 256
  imageW := 256
  layers := [
    -- Encoder: 8 × unetDown halving spatial each time, channels
    -- following 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512.
    .unetDown 3    64,        -- 256 → 128, skip =  64
    .unetDown 64  128,        -- 128 →  64, skip = 128
    .unetDown 128 256,        --  64 →  32, skip = 256
    .unetDown 256 512,        --  32 →  16, skip = 512
    .unetDown 512 512,        --  16 →   8, skip = 512
    .unetDown 512 512,        --   8 →   4, skip = 512
    .unetDown 512 512,        --   4 →   2, skip = 512
    .unetDown 512 512,        --   2 →   1, skip = 512 (bottleneck)
    -- Decoder: 8 × unetUp mirroring, pairing skip widths from last
    -- unetDown backwards to first.
    .unetUp 512 512,          --   1 →   2, pairs skip = 512
    .unetUp 512 512,          --   2 →   4, pairs skip = 512
    .unetUp 512 512,          --   4 →   8, pairs skip = 512
    .unetUp 512 512,          --   8 →  16, pairs skip = 512
    .unetUp 512 512,          --  16 →  32, pairs skip = 512
    .unetUp 512 256,          --  32 →  64, pairs skip = 256
    .unetUp 256 128,          --  64 → 128, pairs skip = 128
    .unetUp 128  64,          -- 128 → 256, pairs skip =  64
    -- Final 1×1 conv to RGB (tanh in the real model)
    .conv2d 64 3 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Pix2Pix discriminator — identical to CycleGAN's PatchGAN
-- ════════════════════════════════════════════════════════════════
-- Shown here for self-containment. 70×70 receptive field; outputs an
-- N×N grid of per-patch real/fake logits.

def pix2pixDiscriminator : NetSpec where
  name := "Pix2Pix PatchGAN discriminator (256×256 → N×N grid)"
  imageH := 256
  imageW := 256
  layers := [
    .convBn 3    64  4 2 .same,         -- 256 → 128
    .convBn 64   128 4 2 .same,         -- 128 →  64
    .convBn 128  256 4 2 .same,         --  64 →  32
    .convBn 256  512 4 1 .same,         -- stride 1; RF grows to 70×70
    .conv2d 512  1   4 .same .identity  -- per-patch logit
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyPix2Pix — compact fixture
-- ════════════════════════════════════════════════════════════════

def tinyPix2Pix_G : NetSpec where
  name := "tiny-Pix2Pix generator (4-level UNet, 64×64)"
  imageH := 64
  imageW := 64
  layers := [
    .unetDown 3  16,          -- 64 → 32
    .unetDown 16 32,          -- 32 → 16
    .unetDown 32 64,          -- 16 →  8
    .unetDown 64 64,          --  8 →  4 (bottleneck)
    .unetUp 64 64,            --  4 →  8
    .unetUp 64 32,            --  8 → 16
    .unetUp 32 16,            -- 16 → 32
    .unetUp 16 3,             -- 32 → 64 (output widths)
    -- No final conv needed since unetUp already outputs 3 channels
    .conv2d 3 3 1 .same .identity
  ]

def tinyPix2Pix_D : NetSpec where
  name := "tiny-Pix2Pix discriminator"
  imageH := 64
  imageW := 64
  layers := [
    .convBn 3  16 4 2 .same,
    .convBn 16 32 4 2 .same,
    .convBn 32 64 4 1 .same,
    .conv2d 64  1 4 .same .identity
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
  IO.println "  Bestiary — Pix2Pix"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  UNet generator + PatchGAN discriminator. Paired image-to-"
  IO.println "  image translation. The ancestor CycleGAN built on."

  summarize pix2pixGenerator
  summarize pix2pixDiscriminator
  summarize tinyPix2Pix_G
  summarize tinyPix2Pix_D

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Generator is an 8-level UNet"
  IO.println "    via existing .unetDown / .unetUp; discriminator is"
  IO.println "    identical to CycleGAN's PatchGAN."
  IO.println "  • Our 8-level UNet overcounts vs paper (paper ~54M, ours"
  IO.println "    ~70M) because .unetDown / .unetUp use 2 convs per level"
  IO.println "    (Ronneberger) while Pix2Pix uses 1 strided conv per level."
  IO.println "    Architectural shape — 8 levels, 64→128→256→512→512×5,"
  IO.println "    skip connections — is faithful."
  IO.println "  • Paired vs unpaired is the key pedagogical contrast with"
  IO.println "    CycleGAN: Pix2Pix's L1 reconstruction loss is a direct"
  IO.println "    supervision signal available only when pairs exist."
  IO.println "    CycleGAN's cycle consistency is what you fall back on"
  IO.println "    when they don't."
  IO.println "  • 2016–2017 hardware context: 54M UNet on 256×256 images"
  IO.println "    meant batch size 1 on a GTX 1080 Ti. The 'use batch size"
  IO.println "    1, rely on InstanceNorm' pattern in the paper is a"
  IO.println "    direct response to that constraint — it's not a deliberate"
  IO.println "    design decision so much as 'this is what fit'."
  IO.println "  • Successors: Pix2PixHD (2017) scales to 2048×1024 via"
  IO.println "    multi-resolution + coarse-to-fine; SPADE (2019) replaces"
  IO.println "    the UNet with a semantic image-synthesis generator. Same"
  IO.println "    paired-supervision template."
