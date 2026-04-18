import LeanMlir

/-! # VAE — Bestiary entry

VAE (Kingma \& Welling, 2013 --- "Auto-Encoding Variational Bayes") is
the autoencoder that learned a \emph{distribution} over latents
instead of a single point. The encoder outputs $(\mu, \log \sigma^2)$
rather than a code; you sample $z = \mu + \sigma \odot \epsilon$
(where $\epsilon \sim \mathcal{N}(0, I)$ is fresh noise per sample)
and decode. The KL divergence between the learned $(\mu, \sigma)$
and a standard normal goes into the loss as a regularizer, which
pushes the latent distribution to stay approximately Gaussian and
keeps the decoder tolerant to interpolation in latent space.

That's the entire generative trick. Everything architecturally
interesting follows from it:

- **The encoder outputs $2 \times D$ not $D$.** One half is $\mu$,
  one is $\log \sigma^2$. Our NetSpec shows the doubled output width
  as a \texttt{.dense} stand-in; the two halves are a convention on
  top of a single tensor, not a new layer.
- **The reparameterization trick** ($z = \mu + \sigma \odot \epsilon$)
  lives in training code, not the network. NetSpec doesn't express
  randomness.
- **The decoder mirrors the encoder.** Same depth, same widths,
  reversed. For convolutional VAEs, transposed convs or upsamples
  replace strided convs.

## Where VAEs show up

- Textbook generative model for MNIST / CIFAR (the canonical
  $\sim$20-dim latent VAE you learn from the Kingma paper).
- \textbf{Stable Diffusion's first stage} (see
  \texttt{StableDiffusion.lean}) --- a conv VAE that compresses
  512$\times$512 pixel images to 64$\times$64$\times$4 latents. The
  generative model operates on the latents, not pixels, so the VAE
  is the key efficiency win.
- VQ-VAE (van den Oord 2017) swaps the continuous Gaussian latent
  for a discrete codebook. Same encoder-decoder shape, different
  bottleneck semantics. Discrete codebook lookup isn't a standard
  layer; bestiary-style treatment would need a new primitive.
- $\beta$-VAE (Higgins 2017) multiplies the KL term by a scalar.
  Architecturally identical; the change lives in the loss.

## Variants

- `mnistVAEEncoder` / `mnistVAEDecoder` --- the canonical MLP VAE
  on 28$\times$28 MNIST with a 20-dim latent. The textbook example.
- `cifarVAEEncoder` / `cifarVAEDecoder` --- a convolutional VAE on
  32$\times$32 CIFAR with a 128-dim latent. What you'd see in a
  modern diffusion-precursor.
- `tinyVAEEncoder` / `tinyVAEDecoder` --- 16$\times$16 fixture.

## Parameter count conventions

Encoder's final layer outputs $2 \times$ latent-dim (e.g.\ 40 for
a 20-dim latent) --- the $(\mu, \log \sigma^2)$ concatenation. Decoder
input is the post-sample $z$, shape latent-dim.
-/

-- ════════════════════════════════════════════════════════════════
-- § MNIST VAE — MLP, 20-dim latent (the textbook example)
-- ════════════════════════════════════════════════════════════════

def mnistVAEEncoder : NetSpec where
  name := "MNIST VAE encoder (MLP, 20-dim latent)"
  imageH := 28
  imageW := 28
  layers := [
    .flatten,
    .dense 784 400 .relu,
    -- Output is 2 × 20 = 40: first 20 are μ, last 20 are log σ².
    .dense 400 40 .identity
  ]

def mnistVAEDecoder : NetSpec where
  name := "MNIST VAE decoder (MLP, 20-dim latent)"
  imageH := 20         -- latent dim (post-reparameterization)
  imageW := 1
  layers := [
    .dense 20 400 .relu,
    .dense 400 784 .identity     -- reshape to 28×28 outside the network
  ]

-- ════════════════════════════════════════════════════════════════
-- § CIFAR VAE — convolutional, 128-dim latent
-- ════════════════════════════════════════════════════════════════

def cifarVAEEncoder : NetSpec where
  name := "CIFAR VAE encoder (conv, 4×4×4 spatial latent)"
  imageH := 32
  imageW := 32
  layers := [
    -- Staying in conv-land: the latent is a 4×4 spatial grid with 4
    -- channels (64-dim effective). Same SD/LDM approach. Pure-MLP
    -- global-latent VAEs exist too; see mnistVAE above.
    .convBn 3   64  3 2 .same,   -- 32 → 16
    .convBn 64  128 3 2 .same,   -- 16 →  8
    .convBn 128 256 3 2 .same,   --  8 →  4
    .convBn 256 256 3 1 .same,   -- bottleneck
    -- 2 × 4 = 8 channels per spatial location (μ + log σ² for 4-ch latent)
    .conv2d 256 8 1 .same .identity
  ]

def cifarVAEDecoder : NetSpec where
  name := "CIFAR VAE decoder (conv, 4×4×4 spatial latent)"
  imageH := 4
  imageW := 4
  layers := [
    .conv2d 4 256 1 .same .identity,   -- expand 4-ch latent to feature dim
    .convBn 256 256 3 1 .same,
    .unetUp 256 128,                    --  4 →  8
    .unetUp 128 64,                     --  8 → 16
    .unetUp 64  32,                     -- 16 → 32
    .conv2d 32 3 3 .same .identity      -- final RGB
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyVAE fixture — 16×16 input, 4-dim latent
-- ════════════════════════════════════════════════════════════════

def tinyVAEEncoder : NetSpec where
  name := "tiny-VAE encoder (4×4×2 spatial latent)"
  imageH := 16
  imageW := 16
  layers := [
    .convBn 1 8  3 2 .same,      -- 16 →  8
    .convBn 8 16 3 2 .same,      --  8 →  4
    .conv2d 16 4 1 .same .identity   -- 2 × 2 = 4 channels (μ + log σ² for 2-ch latent)
  ]

def tinyVAEDecoder : NetSpec where
  name := "tiny-VAE decoder (4×4×2 spatial latent)"
  imageH := 4
  imageW := 4
  layers := [
    .conv2d 2 16 1 .same .identity,
    .unetUp 16 8,                --  4 →  8
    .unetUp 8  4,                --  8 → 16
    .conv2d 4 1 3 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000} K)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — VAE"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Variational autoencoder. Encoder outputs (μ, log σ²); sample"
  IO.println "  z = μ + σ·ε; decoder reconstructs. KL regularizer in the loss."

  summarize mnistVAEEncoder
  summarize mnistVAEDecoder
  summarize cifarVAEEncoder
  summarize cifarVAEDecoder
  summarize tinyVAEEncoder
  summarize tinyVAEDecoder

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ZERO new Layer primitives. Encoder / decoder are plain"
  IO.println "    MLPs (MNIST) or conv stacks (CIFAR). The '2 × latent-dim'"
  IO.println "    convention on the encoder's final layer packs μ and log σ²"
  IO.println "    into one tensor; the split is a training-code decision."
  IO.println "  • The reparameterization trick (z = μ + σ ⊙ ε with ε ~ N(0,I))"
  IO.println "    lives OUTSIDE the network. NetSpec doesn't express random"
  IO.println "    sampling; that's the training loop's job."
  IO.println "  • Stable Diffusion's VAE is a much larger conv VAE of this"
  IO.println "    same shape (see StableDiffusion.lean). The architectural"
  IO.println "    template scales; 2013's 20-dim MNIST latent and 2022's"
  IO.println "    64×64×4 SD latent are the same idea applied at different"
  IO.println "    scales."
  IO.println "  • VQ-VAE (van den Oord 2017) replaces the Gaussian latent"
  IO.println "    with a discrete codebook. Same encoder/decoder shape;"
  IO.println "    discrete codebook lookup isn't a standard layer, so VQ-VAE"
  IO.println "    isn't included as a bestiary entry (would need a new"
  IO.println "    primitive). β-VAE scales the KL term; architecturally"
  IO.println "    identical."
