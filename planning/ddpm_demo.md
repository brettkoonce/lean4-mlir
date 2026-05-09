# ddpm_demo.md — DDPM diffusion-model demo on CIFAR-10

Goal: a worked generative-model example for the bestiary that shows
the framework handles iterative-sampling architectures end-to-end
with verified gradients. The "framework can do generative" demo,
parallel to UNet ("framework can do segmentation") and YOLO/BiFPN
("framework can do detection").

## Status (2026-05-09)

**Phase 1 ✓ + Phase 2 ✓ shipped in 2 sessions.** MNIST produces
legible digits from noise; CIFAR-10 produces recognizable cars /
birds / horses from noise. Plan below estimated 4-6 weeks; the
simplified path landed in ~12 hours of human-active code work plus
a couple of overnight runs.

**Bottleneck self-attention + sin/cos time embedding attempted
(2026-05-08 → 2026-05-09): codegen primitives ship, neither recipe
is a quality win.** See "Phase 3 partial" below for the full
narrative — five training runs, definitive 50-ep A/B diagnostic.
Verdict: (a) bottleneck attention collapses to a degenerate
low-MSE solution, real fix needs architectural surgery (layer-scale
init or per-block time-MLP); (b) sincos t-channels alone are a
small *negative* vs the plain `t/T_max` tile at matched budget;
(c) the codegen changes did not regress the existing path. The
spatialFlatten/Unflatten primitives, keepSequence flag, and sincos
helpers are real reusable infrastructure, but neither directly
improves the demo's sample quality at the current training budget.

What was simplified vs. the original plan:

- **Sin/cos time embedding partially landed (2026-05-09).** Original
  plan called for sinusoidal encoding fed through a dense MLP and
  added to *each block's* feature map. The first session shipped a
  cruder single-channel `t/T_max` tile; a follow-up session shipped
  multi-channel sincos-at-log-spaced-frequencies tile (Vaswani / NeRF
  convention), but **still tile-only — no per-block MLP injection.**
  Files: `LeanMlir/Ddpm.lean::prependSinCosT{,Scalar}`,
  `ffi/f32_helpers.c::lean_ddpm_prepend_sincos_t{,_scalar}`. Per-block
  conditioning remains future work and is now suspected to be one of
  the missing pieces for the bottleneck-attention recipe (see Phase 3
  partial below).
- **Bottleneck self-attention codegen ✓, recipe ✗ (2026-05-09).**
  Spatial-flatten primitives (`spatialFlatten`, `spatialUnflatten C H W`)
  shipped as Layer kinds with full forward+backward emit, plus a
  `keepSequence : Bool := false` flag on `transformerEncoder` so
  ViT specs (slice CLS) and DDPM specs (preserve [B,N,D]) can share
  the same primitive. Two training runs (50 ep at lr=5e-4; 30 ep at
  lr=1e-4) both produce monotone-blob samples despite low MSE
  (0.063-0.068 — same range as the plain base80 baseline at 70 ep).
  See "Phase 3 partial" below for the failure mode and remaining
  hypotheses.
- **No FID.** Visual inspection only. PPM grids of 16 samples each.

What's actually built and working:

| Piece | File | Notes |
|---|---|---|
| Cosine α schedule | `ffi/f32_helpers.c::lean_ddpm_cosine_schedule` | T=1000, clamped [1e-4, 0.9999] |
| Per-step noise application | `lean_ddpm_step_inputs` | Returns (x_t, ε, t) per batch |
| t-channel prepend | `lean_ddpm_prepend_t_channel{,_scalar}` | Generalized to arbitrary C |
| Per-pixel MSE loss + gradient | `MlirCodegen.lean` `useDdpm` flag | Parallel to `useSeg` / `useSoftLabels` |
| Adam train-step ABI | `iree_ffi_train_step_adam_ddpm` | f32 [B, C, H, W] target |
| DDIM η=0 sampler | `lean_ddim_step` | One affine pass |
| Data centering | `F32.scaleShift` | Generic affine — `out = scale·in + shift` |
| MNIST trainer/sampler | `MainMnistDdpm{Train,Sample}.lean` | 1 image channel |
| CIFAR trainer/sampler | `MainCifarDdpm{Train,Sample}.lean` | 3 RGB channels |

### Run-by-run results

| Run | Spec | Params | Epochs | Wall time | End loss | Visual |
|---|---|---|---|---|---|---|
| MNIST no-tcond | base 16 (1ch in) | 118K | 3 | ~5 min | 0.054 | Blob shapes (no t signal) |
| MNIST tcond | base 16 (2ch in) | 118K | 3 | ~5 min | 0.054 | Same — t-cond hadn't trained |
| MNIST tcond long | base 16 (2ch in) | 118K | 50 | ~65 min | 0.032 | **Legible digits 0-9** (`runs/2026-05-07-mnist-ddpm-tcond/samples_50ep.png`) |
| CIFAR base16 | base 16 [0,1] | 120K | 50 | ~70 min | 0.047 | Mean blob collapse |
| CIFAR base48 centered | base 48 [-1,1] | 1M | 100 | ~5 hr | 0.063 | Diverse soft structure |
| CIFAR base80 centered | base 80 [-1,1] | 3M | 70 | ~7 hr | 0.062 | **Recognizable cars/birds/animals** (`runs/2026-05-08-cifar-ddpm-base80/samples_70ep.png`) |

Key empirical finding: **MSE plateaus while sample quality keeps
improving**. base48 → base80 had only a 0.001 loss difference but a
dramatic visual quality jump from "soft chromatic shapes" to
"recognizable categories". Don't rely on MSE as a sample-quality proxy.

### Phase 3 partial: bottleneck attention attempted (2026-05-08 → 09)

The planning doc flagged bottleneck self-attention as the #1 quality
lever. Implementation went in cleanly. Training results did not.

What shipped (codegen — these primitives are real and reusable):
- `spatialFlatten` / `spatialUnflatten C H W` Layer kinds:
  `[B,C,H,W] ↔ [B,H·W,C]` via paired transpose+reshape. No params.
  Forward + backward + train-step plumbing all clean.
- `transformerEncoder` gained a `keepSequence : Bool := false` flag,
  decoupling "no CLS slice" from "causal mask". ViT default false
  (slice CLS for the classifier head); DDPM uses `keepSequence := true`
  to preserve the [B,N,D] sequence flowing into a downstream
  `spatialUnflatten`; tinyGPT continues to set `causalMask := true`
  which implicitly preserves the sequence.
- `prependSinCosT{,Scalar}` C helpers: replaces the cruder single
  `t/T_max` tile with `2·n_freq` channels of sin/cos at log-spaced
  frequencies (Vaswani / NeRF convention).
- `collectBnLayers` pidx accounting fixed for transformerEncoder /
  tokenPositionEmbed / lmHead / patchEmbed (the existing fall-through
  was silently corrupting BN-stat indexing on any spec mixing convBn
  with these layers — an old latent bug we tripped here).

Five training runs across three architectures, plus a 50-ep
A/B-diagnostic on the proven baseline:

| Spec | Epochs | LR | Loss | Sample |
|---|---|---|---|---|
| base 64 + attn (plain `t/T` tile) | 20 | 5e-4 | 0.068 | dark-red monotone blobs |
| base 64 + attn + sincos t-channels | 50 | 5e-4 | 0.064 | dark-red monotone blobs |
| base 64 + attn + sincos, lower LR | 30 | 1e-4 | 0.068 | orange-yellow monotone blobs |
| base 80 + sincos (no attn) | 50 | 5e-4 | 0.062 | orange blobs, less structure |
| base 80 plain (diagnostic A/B) | 50 | 5e-4 | 0.062 | yellow blobs, animal-ish silhouettes |
| **base 80 plain conv (reference)** | **70** | **5e-4** | **0.062** | **recognizable cars/birds/animals** |

Failure signature: low MSE + structural collapse. Same MSE range
across all five runs, dramatically different visual outcomes. The
model finds a low-MSE plateau without learning the data manifold;
sample quality lives in the residual fluctuations that MSE doesn't
distinguish.

Three reads from the table (post-diagnostic 2026-05-09):

1. **Training-budget hypothesis: confirmed in part.** The proven
   baseline is *also* worse at 50 ep than at 70 ep (yellow
   animal-ish blobs vs the reference's clean recognizable cars).
   The reference checkpoint crosses a quality threshold around
   epoch 60-70 that pure MSE doesn't capture. So part of the
   gap on every 50-ep run is just budget.

2. **Sincos t-channels are a small negative, not a win.** Same
   architecture (base80 plain conv), same training budget (50 ep,
   lr=5e-4), same MSE (0.062 both) — the sincos variant produces
   *less* structure than the single-channel `t/T_max` tile. The 8
   sincos channels apparently confuse the early conv layers more
   than they help. So the planning doc's `#2 — proper sin/cos time
   embedding` was tested and isn't a near-term lever; it likely
   needs the per-block MLP injection that didn't ship.

3. **Codegen changes did not regress the existing path.** The
   diagnostic 50-ep base80 plain run uses the modified codegen
   (collectBnLayers pidx accounting, transformerEncoder arity bump,
   spatialFlatten arms) and still produces structured samples. So
   the bottleneck-attention failure is genuinely architectural,
   not a side-effect bug in shared codegen.

Most plausible root cause for the attention failure (still
untested): pre-norm + residual skip lets the network bypass
attention entirely. With small init, MHSA and FC2 outputs are
~zero, residuals route around them, and the model effectively
trains a base-64 conv UNet — *smaller capacity than the base80
baseline*, which directly explains the worse-than-baseline
quality. The transformer block sits dormant at init and never
lights up because the rest of the network can already reach a
low-MSE plateau without it.

Remaining hypotheses, ordered by likelihood (untested):

1. **Per-block time-MLP conditioning is missing.** Standard DDPM
   architectures route the time embedding through a small MLP per
   block, scaled and added to feature maps. Our sincos channels
   only enter at the input, so the bottleneck attention has no
   strong per-block time signal. Effort: ~3-5 hr (need a new "add
   time projection" primitive or a cleaner per-block emit path).
2. **Layer-scale init.** ConvNeXt / DiT both initialize transformer
   block contributions at ~zero (a learnable scalar γ multiplied
   onto MHSA/FFN outputs) so the residual path dominates at init,
   then the block gradually "turns on" as γ grows. We use uniform
   He-init which gives the block real magnitude immediately, but
   apparently it doesn't train through. Effort: ~1-2 hr (one new
   per-channel γ param plus a multiply in the block emit).
3. **Post-norm instead of pre-norm.** Pre-norm is the modern default
   but post-norm forces the gradient through the LN+block instead of
   skipping. Old transformers used post-norm successfully. Effort:
   ~2-3 hr (re-order ops in `emitTransformerBlockForward` and the
   matching backward; touches every existing transformer use site).
4. **Stronger attention position (16×16 instead of 8×8).** Counter-
   intuitive: more tokens means each spatial location resolves more,
   but compute scales with N². Worth trying once one of the above
   works. Effort: 0 (just spec edit).
5. **More attention blocks (1 → 2 or 4).** If a single block is
   bypassed, two might force more useful work. ~0 effort, but adds
   train time.

For now, the codegen primitives are checked in (commits below).
A revisit needs at least one of the architectural surgery items
landed first. The investment isn't wasted — these primitives also
unlock SegFormer / MobileViT / CCT promotion from shape-only to
codegen-backed.

### Commits (DDPM stack)

```
45808ac diagnostic: base80 plain at 50ep — answers the A/B question
042261a sincos baseline 50-ep: loss matches baseline, samples don't
1554211 sincos baseline: base80 plain conv + sincos t-embed (no attn)
70e7b3a plan: capture bottleneck-attention attempt (codegen ✓, recipe ✗)
5c1bb88 attn debug: LR=1e-4 not the lever (still monotone)
fbddb13 attn+sincos 50-epoch run: pipeline works, samples don't beat baseline
2f80d25 polish: sin/cos time embedding (replaces tile-channel)
a713f32 attention 20-epoch run: preliminary samples (loss 0.068)
2c28523 bottleneck attention: spatialFlatten/Unflatten primitives + new spec
3053fc3 CIFAR base80 spec for overnight run
62d5ce4 CIFAR base48 + data centering for quality bump
ef48460 CIFAR-10 trainer + sampler (multi-channel)
c1a998a time conditioning + 50-epoch run produces legible digits
452ce6b Phase 1 plumbing + MNIST smoke test
```

### What's left to push CIFAR quality further

Ranked by impact-per-hour-of-code (revised after the 2026-05-09
attention-experiment results):

1. **Make bottleneck attention actually train** — codegen ✓, recipe ✗.
   Pick one or more from the Phase-3-partial hypotheses above. Best
   bang-for-effort is probably **layer-scale init (~1-2 hr)** because
   it's a one-line change and directly addresses the suspected
   "attention sits dormant at init" failure mode. If layer-scale
   alone doesn't fix it, layer in **per-block time-MLP conditioning
   (~3-5 hr)**, then iterate.
2. **Tile-only sincos t-embedding ✓ shipped 2026-05-09 — small negative
   on its own.** A/B'd against the plain `t/T_max` tile at the same
   architecture and training budget (base80, 50 ep). Sincos variant
   produces *less* structure than the single-channel tile — the 8
   sincos channels apparently mostly add noise to the early conv
   layers without compensating signal. Per-block MLP injection still
   missing and is the next thing to try; sincos channels at the
   input alone aren't the right primitive at this depth.
3. **More capacity** (free; existing code) — base 80 → 96 or 128.
   Per-step time scales quadratically with channel base; base 128
   would be ~500ms/step vs base 80's 230ms. ~50 epochs of base 128
   = ~13 hours.
4. **Class-conditional generation** (~1-2 hr code) — embed CIFAR
   labels as a one-hot channel alongside the t channel. "Free
   performance" because labels give the model a strong prior, but
   changes the demo's framing from unconditional to conditional.

The plan as originally written (sin/cos + bottleneck attention +
800K iter + larger model) would close the gap to paper FID 3.17;
that's ~weeks of compute and is out of scope unless someone wants
to actually compete on benchmarks.

---

## The rule

DDPM (Ho, Jain, Abbeel 2020) is **a UNet trained to denoise**.
The architecture is UNet with three additions:

1. **Time embedding** (sinusoidal encoding of the diffusion timestep)
2. **Time-conditioning** of UNet feature maps (small dense projection added to each block)
3. **Self-attention** at one or two intermediate resolutions (typically 8×8 and 16×16 in the bottleneck)

All three pieces decompose to existing primitives once `unet_demo.md`
lands. The new work is the **training-time noise scheduling** and
the **sampling-time iterative denoising loop** — both data-side
plumbing, not codegen.

**Prerequisite: UNet demo done first** (this is built on top of UNet).

## Architecture: time-conditioned UNet

| Piece | Source | New work |
|---|---|---|
| UNet backbone | `unet_demo.md` (encoder + decoder + skips) | none |
| Sinusoidal time embedding | timestep → fixed-dim embedding via sin/cos at multiple frequencies | pure data, no codegen |
| Time-conditioning | small dense (embed → channels) added to each block's feature map | dense + add, both existing |
| Self-attention at conv resolutions | `transformerEncoder` applied to (B, HW, C) reshape of conv feature map | existing primitives, novel shape |

For CIFAR-10 32×32: a small UNet (3 stages, channels 64→128→256→256
or similar), self-attention at the 8×8 bottleneck, ~30-40M params.
Trains comfortably on a single 7900 XTX in ~6-12 hours for 100K
iterations.

## New primitives

**Sinusoidal time embedding**: takes integer timestep `t`, computes
`[sin(t/10000^(2i/d)), cos(t/10000^(2i/d))]` for each frequency `i`.
Pure deterministic function, no params, no learnable VJP — just a
data-side computation that's run once per batch and broadcast.

**Time-conditioning add**: existing primitives (dense + broadcast +
elementwise add). No new codegen.

**No genuinely new VJPs needed.** Everything reduces to: UNet
primitives + dense layers + elementwise ops + self-attention, all
already proved.

## Loss function

Standard DDPM loss is MSE between predicted noise and actual noise:

```
L = ||ε_predicted(x_t, t) - ε_actual||²
```

The model takes noised image `x_t` and timestep `t`, predicts the
noise that was added. Already have MSE VJP. ~1 day to wire up.

## Training loop modifications

This is the substantive new work. Per training step:

1. Sample timestep `t ~ Uniform(1, T)` (T=1000 standard)
2. Sample noise `ε ~ N(0, I)`
3. Compute noised image `x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε` (using precomputed cumulative alpha schedule)
4. Predict `ε_θ(x_t, t)` via the UNet
5. Loss = MSE(ε_θ, ε)
6. Backprop and update

| Item | Effort |
|---|---|
| Beta schedule (linear or cosine, precomputed table of α, ᾱ, β) | 1-2 days |
| Noise sampling + scheduled noising forward | 2-3 days |
| Timestep sampling | 1 day |
| Modified train step that does steps 1-6 above | 1 week |

## Sampling loop (inference)

DDPM's "inference" is the iterative denoising process:

```
x_T ~ N(0, I)
for t in T..1:
    ε_θ = model(x_t, t)
    x_{t-1} = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ) + σ_t · z
return x_0
```

Pure inference (no gradient), so it's a separate exe that loads the
trained checkpoint. ~1 week to implement DDPM sampling correctly.

For faster sampling: **DDIM** (50-100 steps instead of 1000) is the
canonical speedup. Same trained weights, different sampling
schedule. ~3 days to add as a separate exe.

## Data pipeline

CIFAR-10 is already in the codebase as a `DatasetIO`. The DDPM
training loop wraps the existing data loader — each batch goes
through the noise-scheduling step before hitting the model. **Zero
new dataset work.** This is the cheapest demo on the data side
because diffusion is unsupervised (no labels needed beyond the
images themselves, and we're not even using class labels for
unconditional generation).

## Eval

Two paths, easiest first:

| Item | Effort |
|---|---|
| **Visual inspection** — sample 64 images, save as a grid PNG | 2-3 days (just sampling + image writing) |
| **FID** (Fréchet Inception Distance) — paper-standard metric | ~2 weeks (needs an Inception network for feature extraction, statistics computation) |

For a bestiary demo, visual inspection is enough. The output is "a
grid of 64 generated CIFAR-style images" and the demo line is
"these came out of nothing but Gaussian noise + the trained UNet."
That sells the story without paying the FID complexity tax.

FID is the right metric if we want to compare against published
numbers. Paper DDPM on CIFAR-10 reports FID ~3.17. Our demo would
probably hit FID 10-30 with limited training; honest framing is
"the pipeline works, longer training would close the gap."

## Sequencing

**Prerequisite: UNet demo (`unet_demo.md`) complete.**

**Phase 1 — diffusion plumbing (1-2 weeks):**
- Beta schedule (precomputed α, ᾱ tables)
- Sinusoidal time embedding
- Time-conditioning blocks in the UNet variant

**Phase 2 — training loop (1-2 weeks):**
- Noise sampling + timestep sampling per batch
- Modified train step (steps 1-6 above)
- Self-attention at the bottleneck (reuse existing transformer primitives)

**Phase 3 — sampling + eval (1-2 weeks):**
- DDPM sampling exe (1000-step iterative denoising)
- DDIM sampling exe (50-100 step variant)
- Image grid output for visual demo
- (Optional) FID against held-out CIFAR test set

**Total: ~4-6 weeks if UNet done first.** Standalone (without UNet
prerequisite): add another 4-6 weeks for UNet work.

## Cells to add

```
("ddpm-cifar10-unet32",  ⟨ddpmUnet32Spec, ddpmCifarConfig, .cifar10, "data"⟩),
```

Single cell since the architecture is the demo. Could add a
linear-vs-cosine beta-schedule ablation as a second cell if useful.

## Compute budget

- ~50K-200K training iterations standard for CIFAR-10 DDPM
- Per-iter on the 7900 XTX with a 30M-param UNet at 32×32: ~150-200ms
- 100K iter × 200ms = ~5.5 hours
- Sampling: 1000 steps × ~50ms forward = ~50s per generated image (DDPM)
- DDIM: 50 steps × ~50ms = 2.5s per generated image (much friendlier)

Generating a 64-image grid: ~3 minutes with DDIM, ~50 minutes with full DDPM.

## What this unlocks

The diffusion training pattern (noise schedule + denoising loss +
iterative sampling) is the **gateway pattern** for a large modern
generative-model family:

- **Stable Diffusion** — same architecture pattern at scale, latent space
- **Score-based models** (Song & Ermon) — same math, different parameterization
- **Flow-matching** (Lipman et al.) — same training framework, simpler loss
- **Conditional diffusion** (text-to-image, image-to-image) — adds conditioning, same backbone
- **Video diffusion** — extends time dimension, same primitives

Once DDPM works, "Stable Diffusion's training loop" is a
straight-line extension (replace UNet with a bigger one, add
text-conditioning via cross-attention, train on a latent space
instead of pixels).

## Honest tradeoff

DDPM on CIFAR-10 with limited training:
- **Visual quality**: recognizable CIFAR-style images (planes, cars,
  birds, etc. as blurry-but-recognizable shapes)
- **FID**: probably 10-30 vs paper's 3.17 — would need 200K+ iter
  + careful tuning to close
- **Demo value**: "the framework trains a generative model that
  samples coherent images from noise" is the story — beating the
  paper isn't required

## Out of scope (deferred)

- **CIFAR-100 or ImageNet diffusion** — bigger compute, no
  pedagogical gain over CIFAR-10 demo
- **Conditional diffusion** (class-conditional, text-conditional) — adds
  cross-attention; defer until unconditional demo is solid
- **Latent diffusion / Stable Diffusion** — needs a separate
  encoder-decoder (VAE) for the latent space; significantly more
  scaffolding
- **Score-based parameterization** — equivalent math, alternative
  framing; pick one
- **Variational lower bound loss** — paper uses simple MSE
  (`L_simple`), which works fine; the VLB is a more principled
  but harder-to-implement variant
- **Classifier-free guidance** — for conditional models; not
  applicable to unconditional demo
