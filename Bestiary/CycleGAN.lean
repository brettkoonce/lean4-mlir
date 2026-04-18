import LeanMlir

/-! # CycleGAN — Bestiary entry

CycleGAN (Zhu, Park, Isola, Efros, ICCV 2017 --- "Unpaired Image-to-
Image Translation using Cycle-Consistent Adversarial Networks") is
the paper that removed the paired-data requirement from image
translation. Pix2Pix (Isola 2017) needed matched before/after
image pairs (a sketch-and-its-photo, a map-and-its-aerial-view); in
most real settings you can't get those. CycleGAN's insight: if you
have two \emph{unpaired} sets of images --- say, horses and zebras
--- you can train a pair of generators $G: X \to Y$ and $F: Y \to
X$, and enforce that $F(G(x)) \approx x$ (the ``cycle consistency''
loss) so the round trip preserves content.

The architecture is the easy part. The content is in the training
objective.

## Four networks, trained jointly

```
         X (horses)               Y (zebras)
           ▲                          │
           │                          │
     F ────┤                          ├──── G
           │                          │
           ▼                          ▼
       Y (fake zebras)            X (fake horses)

     D_X judges real vs fake X    D_Y judges real vs fake Y
```

The loss has three terms:

- **Adversarial loss** (standard GAN): $D_Y$ tries to tell real
  zebras from $G(x)$; $D_X$ tries to tell real horses from $F(y)$.
- **Cycle consistency loss** (the new idea): $\|F(G(x)) - x\|_1$
  and $\|G(F(y)) - y\|_1$. After a round trip through both
  generators, you should get back (approximately) the image you
  started with.
- **Identity loss** (optional): $\|G(y) - y\|_1$ penalizes $G$ for
  changing images that are already in the target domain.

The cycle consistency is what makes unpaired training work. Without
it, $G$ could map every horse to the same ``best zebra'' --- mode
collapse. With it, $G$ has to preserve enough information that $F$
can invert the mapping.

## Generator architecture (same for $G$ and $F$)

Johnson-et-al.\ 2016's perceptual-loss-network design:

```
  Input: H × W × 3
      │
      ▼ conv 7×7, 3→64, s=1      (padding for 7-kernel edge)
      ▼ conv 3×3, 64→128, s=2    (downsample)
      ▼ conv 3×3, 128→256, s=2   (downsample)
      ▼ ResBlock × 9 at 256 ch   (the bottleneck)
      ▼ trans-conv 3×3, 256→128, s=2 (upsample)
      ▼ trans-conv 3×3, 128→64, s=2  (upsample)
      ▼ conv 7×7, 64→3, s=1, tanh    (output)
```

Same 256$\times$256 in, 256$\times$256 out. $\sim$11M params per
generator.

## Discriminator architecture (PatchGAN, same for $D_X$ and $D_Y$)

The discriminator doesn't classify whole images --- it outputs an
$N \times N$ grid of real/fake predictions, each corresponding to a
70$\times$70 patch of the input (the receptive field). Patch-level
discrimination penalizes high-frequency artifacts (texture) while
the adversarial + cycle losses together handle low-frequency
structure.

```
  Input: 256 × 256 × 3
      │
      ▼ conv 4×4, 3→64, s=2      (no BN, LeakyReLU)
      ▼ convBN 4×4, 64→128, s=2
      ▼ convBN 4×4, 128→256, s=2
      ▼ convBN 4×4, 256→512, s=1 (stride 1 this time!)
      ▼ conv 4×4, 512→1, s=1     (output logit map)
```

$\sim$2.8M params per discriminator.

## Variants

- `cycleganGenerator`      --- 9-block ResNet generator (shared
                              shape for both $G$ and $F$)
- `cycleganDiscriminator`  --- PatchGAN discriminator (shared shape)
- `tinyCycleGAN_G` / `_D`  --- fixture

## NetSpec simplifications

- Only one generator and one discriminator are shown; in a real
  setup you instantiate two of each. Architecture is identical ---
  only weights differ.
- The generator's upsample transposed convs are approximated by
  \texttt{.convBn} stride-1 stand-ins (same params, spatial
  doubling is forward-pass-only). Same trick DCGAN uses.
-/

-- ════════════════════════════════════════════════════════════════
-- § CycleGAN generator (Johnson-style ResNet, 9 residual blocks)
-- ════════════════════════════════════════════════════════════════

def cycleganGenerator : NetSpec where
  name := "CycleGAN generator (9-block ResNet, 256×256 RGB)"
  imageH := 256
  imageW := 256
  layers := [
    -- Head: 7×7 s=1 conv to 64 channels
    .convBn 3 64 7 1 .same,
    -- Downsample ×2 via strided 3×3 convs
    .convBn 64  128 3 2 .same,         -- 256 → 128
    .convBn 128 256 3 2 .same,         -- 128 →  64
    -- Bottleneck: 9 residual blocks at 256 channels
    .residualBlock 256 256 9 1,
    -- Upsample ×2 via transposed-conv-shaped .convBn (param-count stand-in)
    .convBn 256 128 3 1 .same,         --  64 → 128 (stand-in)
    .convBn 128  64 3 1 .same,         -- 128 → 256 (stand-in)
    -- Head: 7×7 s=1 conv back to 3 channels (tanh in real)
    .conv2d 64 3 7 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § CycleGAN discriminator (PatchGAN, 70×70 receptive field)
-- ════════════════════════════════════════════════════════════════

def cycleganDiscriminator : NetSpec where
  name := "CycleGAN PatchGAN discriminator (256×256 RGB → N×N grid)"
  imageH := 256
  imageW := 256
  layers := [
    -- First layer: no BN in paper; we use convBn for compactness.
    .convBn 3    64  4 2 .same,        -- 256 → 128
    .convBn 64   128 4 2 .same,        -- 128 →  64
    .convBn 128  256 4 2 .same,        --  64 →  32
    .convBn 256  512 4 1 .same,        -- stride 1! receptive field grows to 70×70
    .conv2d 512  1   4 .same .identity -- output: per-patch real/fake logit
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyCycleGAN — compact fixture
-- ════════════════════════════════════════════════════════════════

def tinyCycleGAN_G : NetSpec where
  name := "tiny-CycleGAN generator"
  imageH := 64
  imageW := 64
  layers := [
    .convBn 3 16 7 1 .same,
    .convBn 16 32 3 2 .same,
    .convBn 32 64 3 2 .same,
    .residualBlock 64 64 3 1,          -- 3 res blocks instead of 9
    .convBn 64 32 3 1 .same,
    .convBn 32 16 3 1 .same,
    .conv2d 16  3 7 .same .identity
  ]

def tinyCycleGAN_D : NetSpec where
  name := "tiny-CycleGAN discriminator"
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
  IO.println "  Bestiary — CycleGAN"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Unpaired image translation via cycle consistency. Two"
  IO.println "  generators, two discriminators, one clever loss."

  summarize cycleganGenerator
  summarize cycleganDiscriminator
  summarize tinyCycleGAN_G
  summarize tinyCycleGAN_D

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Generator reuses .residualBlock"
  IO.println "    (9 blocks at 256-ch) surrounded by downsample / upsample"
  IO.println "    convs. PatchGAN discriminator is a 5-conv strided stack."
  IO.println "  • Four-network pattern shown as two specs (G, D); the real"
  IO.println "    setup instantiates two of each for the bidirectional"
  IO.println "    X→Y + Y→X mapping. Architecture is identical; only"
  IO.println "    weights differ."
  IO.println "  • The cycle-consistency loss is THE contribution. With"
  IO.println "    ||F(G(x)) - x||_1 as a training objective, you can train"
  IO.println "    without paired data — a huge practical unlock. No layer"
  IO.println "    expresses this; it's a loss term."
  IO.println "  • PatchGAN's 70×70 receptive field was a deliberate design"
  IO.println "    choice — whole-image discrimination pushes towards low-"
  IO.println "    frequency features (overall brightness, structure);"
  IO.println "    patch-level discrimination pushes towards high-frequency"
  IO.println "    features (texture, edges). The adversarial + cycle losses"
  IO.println "    split responsibility cleanly."
  IO.println "  • Successors: Pix2PixHD for paired high-res (2017),"
  IO.println "    StarGAN for multi-domain translation (2018), U-GAT-IT"
  IO.println "    adds attention (2020). All reuse the CycleGAN training"
  IO.println "    template with architectural tweaks."
