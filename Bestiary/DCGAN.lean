import LeanMlir

/-! # DCGAN — Bestiary entry

DCGAN (Radford, Metz, Chintala, 2015 --- "Unsupervised Representation
Learning with Deep Convolutional Generative Adversarial Networks")
was the paper that made GAN training reliably work. Vanilla GANs
from Goodfellow 2014 were mostly MLPs and notoriously unstable;
DCGAN gave you a recipe of eight design decisions that turned GANs
into something you could actually train. Those eight decisions are
now standard in every deep-learning textbook.

## The DCGAN design guidelines

From the paper, section 3:

1. Replace all pooling with strided convs (discriminator) and
   strided transposed convs (generator).
2. Use BatchNorm in both $G$ and $D$ (but not on $G$'s output or
   $D$'s input).
3. Remove fully-connected hidden layers for deeper architectures.
4. ReLU activation in $G$ (plus tanh at the output).
5. LeakyReLU (slope $0.2$) in $D$.
6. Use Adam with lr $= 0.0002$, $\beta_1 = 0.5$.
7. Noise vector $z$ is 100-dim, drawn from $\mathcal{N}(0, I)$.
8. The generator's first layer projects $z$ to a spatial feature
   map via a dense layer + reshape (not shown as a layer in the
   linear NetSpec).

Every GAN paper after this one either follows these guidelines or
explicitly justifies its deviation.

## Anatomy (64$\times$64 RGB output, e.g.\ CelebA)

```
  Generator G:                     Discriminator D:

  z ~ N(0, I_100)                  Image (64×64×3)
      │                                 │
      ▼ dense 100 → 4×4×1024            ▼ conv 3→128   (no BN)
      │   + reshape (prose, not NetSpec) │
      ▼ transposed-conv, 4×4, s=2        ▼ convBN 128→256
       1024 → 512, BN, ReLU              │
      ▼ transposed-conv                   ▼ convBN 256→512
       512 → 256, BN, ReLU                │
      ▼ transposed-conv                   ▼ convBN 512→1024
       256 → 128, BN, ReLU                │
      ▼ transposed-conv                   ▼ flatten → dense → 1
       128 → 64, BN, ReLU                 │ (real/fake logit)
      ▼ transposed-conv
       64 → 3, tanh
      │
      ▼ 64×64×3 fake image
```

## NetSpec simplifications

- **Transposed conv approximation.** Our \texttt{.conv2d} doesn't
  have stride-2 upsampling; transposed conv and regular conv share
  kernel-size and input/output channels, so the \emph{parameter
  count} matches. We use \texttt{.convBn ic oc 4 1 .same} as the
  stand-in; the spatial doubling is a forward-pass detail.
- **Noise projection.** Generator's 100-dim $z \to 4 \times 4 \times
  1024$ dense + reshape is shown as a separate \texttt{.dense} spec
  (\texttt{dcganProjector}) since a linear NetSpec can't express the
  reshape.
- **Activation nits.** LeakyReLU (slope 0.2) in $D$ is
  parameter-free; our \texttt{Activation} enum has \texttt{.relu}
  and \texttt{.relu6}, so we use \texttt{.relu} in both. Same for
  the $\tanh$ at $G$'s output.

## Variants

- `dcganProjector`     --- noise $\to$ $4 \times 4 \times 1024$ projection (1.65M)
- `dcganGenerator`     --- the transposed-conv stack (~11M)
- `dcganDiscriminator` --- the strided-conv stack (~11M)
- `tinyDCGAN_G` / `tinyDCGAN_D` --- fixture
-/

-- ════════════════════════════════════════════════════════════════
-- § DCGAN noise projection (z → 4×4×1024 via dense)
-- ════════════════════════════════════════════════════════════════

def dcganProjector : NetSpec where
  name := "DCGAN noise projector (100 → 4×4×1024)"
  imageH := 1         -- noise as a scalar input
  imageW := 1
  layers := [
    -- Dense layer projects the 100-dim noise vector to 4*4*1024 = 16384
    -- flat values, which are then reshape-to-4x4x1024 before going into
    -- the generator's transposed-conv stack.
    .dense 100 16384 .relu
  ]

-- ════════════════════════════════════════════════════════════════
-- § DCGAN generator (4×4×1024 → 64×64×3)
-- ════════════════════════════════════════════════════════════════

def dcganGenerator : NetSpec where
  name := "DCGAN generator (64×64 RGB)"
  imageH := 4         -- starting spatial resolution (post-projection)
  imageW := 4
  layers := [
    -- Each layer is a transposed conv 4×4 stride 2 in the real DCGAN,
    -- doubling the spatial resolution. We use .convBn as a param-count
    -- stand-in (same kernel + ic + oc = same params).
    .convBn 1024 512 4 1 .same,   --  4×4 →  8×8
    .convBn 512  256 4 1 .same,   --  8×8 → 16×16
    .convBn 256  128 4 1 .same,   -- 16×16 → 32×32
    .convBn 128   64 4 1 .same,   -- 32×32 → 64×64
    .conv2d   64   3 4 .same .identity  -- output (tanh in real DCGAN)
  ]

-- ════════════════════════════════════════════════════════════════
-- § DCGAN discriminator (64×64×3 → scalar real/fake logit)
-- ════════════════════════════════════════════════════════════════

def dcganDiscriminator : NetSpec where
  name := "DCGAN discriminator (64×64 RGB → 1)"
  imageH := 64
  imageW := 64
  layers := [
    -- Strided convs halve the spatial each step. BN on layers 2-4 only
    -- (paper's guideline; first layer has no BN, but we use convBn
    -- throughout for spec compactness — extra ~2·128 params, trivial).
    .convBn 3    128  4 2 .same,  -- 64 → 32
    .convBn 128  256  4 2 .same,  -- 32 → 16
    .convBn 256  512  4 2 .same,  -- 16 →  8
    .convBn 512 1024  4 2 .same,  --  8 →  4
    .flatten,
    .dense (4 * 4 * 1024) 1 .identity   -- real/fake logit
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyDCGAN — compact fixture (32×32 output)
-- ════════════════════════════════════════════════════════════════

def tinyDCGAN_G : NetSpec where
  name := "tiny-DCGAN generator (32×32)"
  imageH := 4
  imageW := 4
  layers := [
    .convBn 128 64 4 1 .same,
    .convBn 64  32 4 1 .same,
    .convBn 32  16 4 1 .same,
    .conv2d 16   3 4 .same .identity
  ]

def tinyDCGAN_D : NetSpec where
  name := "tiny-DCGAN discriminator (32×32)"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3   32 4 2 .same,
    .convBn 32  64 4 2 .same,
    .convBn 64 128 4 2 .same,
    .flatten,
    .dense (4 * 4 * 128) 1 .identity
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
  IO.println "  Bestiary — DCGAN"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Deep Convolutional GAN (Radford 2015). The eight guidelines"
  IO.println "  that made GANs trainable. Canonical GAN reference."

  summarize dcganProjector
  summarize dcganGenerator
  summarize dcganDiscriminator
  summarize tinyDCGAN_G
  summarize tinyDCGAN_D

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Generator uses .convBn / .conv2d"
  IO.println "    as param-count stand-ins for transposed convs (same ic/oc"
  IO.println "    and kernel size = same params, spatial doubling is"
  IO.println "    forward-pass-only). Discriminator is a strided-conv stack."
  IO.println "  • Projector: 1.65M. Generator convs: ~11M. Discriminator:"
  IO.println "    ~11M. Total: ~24M for a 64×64 GAN. Paper says ~12M for G"
  IO.println "    and ~11M for D — close match."
  IO.println "  • The paper's real contribution is the eight design"
  IO.println "    guidelines, not any single layer. Every trainable GAN"
  IO.println "    since has followed those guidelines as a baseline and"
  IO.println "    justified each deviation."
  IO.println "  • Successors: Progressive GAN (Karras 2017) grows layers"
  IO.println "    during training; StyleGAN (2019) introduces AdaIN /"
  IO.println "    mapping network for high-quality faces; BigGAN (2018)"
  IO.println "    scales DCGAN to class-conditional ImageNet. All use"
  IO.println "    DCGAN's guidelines as the default and add their own"
  IO.println "    innovations on top."
  IO.println "  • GAN training is the hard part. Mode collapse, equilibrium"
  IO.println "    stability, and sample diversity are all training-procedure"
  IO.println "    problems — the DCGAN architecture is just the substrate"
  IO.println "    that lets the training procedure work at all."
