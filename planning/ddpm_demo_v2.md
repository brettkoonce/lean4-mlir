# ddpm_demo_v2.md — DDPM demo, second pass

Goal: take the CIFAR-10 DDPM demo from "recognizable blobs at 70
epochs" to a sample grid that sells itself, and in the process land
the one architectural primitive v1 proved is missing: **per-block
time conditioning**. Secondary goals, in order: make the bottleneck
attention block actually train (v1 shipped the codegen, the recipe
failed), a class-conditional variant, a self-contained FID-style
metric, and a verified-gradient tie-in that connects the DDPM train
step to the project's proof arc.

Prerequisite reading: `planning/ddpm_demo.md` (v1 plan + full
post-mortem of the 2026-05-08/09 attention experiments). This doc
assumes those results and does not repeat the tables.

## Where v1 landed (recap, one paragraph)

Eight mains ship today (`demos/Main{Mnist,Cifar}Ddpm*` ×
{Train,Sample} × {plain, sincos, attn}). MNIST produces legible
digits; CIFAR base80 plain conv at 70 ep produces recognizable
cars/birds/animals. The infrastructure — cosine schedule, per-step
noising (`Ddpm.stepInputs`), `useDdpm` MSE codegen branch, Adam DDPM
train-step ABI, DDIM η=0 sampler, `spatialFlatten/Unflatten`,
`transformerEncoder keepSequence` — is all real and committed. What
failed: bottleneck attention collapses to monotone blobs at matched
MSE (suspected cause: pre-norm residual routes around a
He-initialized block that never lights up), and input-level sincos
t-channels are a small *negative* vs the single `t/T_max` tile. The
governing empirical lesson: **MSE is not a sample-quality proxy** —
every run in the 0.062–0.068 band looked wildly different.

## What v1's failures actually tell us

Three independent signals all point at the same missing piece:

1. The attention post-mortem's top-ranked untested hypothesis is
   per-block time-MLP conditioning (the block has no time signal
   where it sits).
2. The sincos-tile A/B showed input-level time channels are already
   saturated — more input channels add noise, not signal. The time
   information has to enter *deeper*.
3. Our own Bestiary entry (`Bestiary/Diffusion.lean`) says it
   outright: real DDPM UNets give **every residual block** an
   additive projection of the time embedding, use **GroupNorm**
   (BN misbehaves when a batch mixes independent timesteps — v1's
   `convBn` UNet has exactly this problem), and init attention
   contributions near zero.

So v2's core bet is: per-block time conditioning is the load-bearing
change, layer-scale init is the cheap enabler for attention, and
everything else is training recipe.

## Workstream A — per-block time conditioning (the new codegen)

The one genuinely new primitive. Design that fits the linear
`NetSpec` without branch/merge syntax:

- **ABI**: the DDPM train step gains one side input `%t_embed :
  [B, D_t]` (sincos features of the per-image timesteps, computed
  host-side — `prependSinCosT`'s math, minus the spatial tiling;
  new FFI helper `lean_ddpm_t_embed`, trivial). The sampler's eval
  ABI gains the scalar-broadcast variant.
- **New Layer kind** `.timeCondAdd C` — emits
  `dense(D_t → C) → broadcast [B,C] → [B,C,1,1] → add` onto the
  current feature map. Two params (W, b). Backward: gradient of the
  add reduces over H,W into the dense VJP (existing dense backward);
  no gradient flows to `%t_embed` itself (it's data). Precedent for
  threading a side value through emit: the UNet skip stack
  (`MlirCodegen.lean:1490`) already does LIFO side-channel plumbing;
  this is simpler — one global SSA value read by every
  `.timeCondAdd` site.
- **Optionally** a shared two-layer time MLP (`D_t → 4·D_t → silu →
  per-block dense`) as in Ho et al. Defer: start with per-block
  dense straight off the sincos features; add the shared trunk only
  if conditioning is still weak.
- Spec usage: interleave `.timeCondAdd` after each `unetDown` /
  bottleneck `convBn` / `unetUp`. Input reverts to plain 3-channel
  RGB (drop the prepended t-channels entirely — cleaner demo story
  too: "the network is told the time through the side door, like
  the paper").

Touchpoints: `Types.lean` (Layer kind + param accounting),
`MlirCodegen.lean` (forward/backward/train-step arms + the new side
input in `generateTrainStep` / `generateEval`), `IreeRuntime.lean` +
`ffi/iree_ffi.c` (ABI variant), `Ddpm.lean` + `f32_helpers.c`
(t-embed helper). Estimate: **1–2 sessions** including the
`collectBnLayers`-style pidx audit that bit v1.

Gate A: base80-equivalent UNet + timeCondAdd, 50 ep, fixed-seed
sample grid vs the v1 50-ep diagnostic. Win = visibly more structure
at 50 ep than v1's yellow-blob 50-ep baseline. If it loses at
matched budget, stop and re-plan before touching attention.

## Workstream B — training recipe (cheap, do first)

Ordered by effort; B1–B3 need zero new codegen and can run before
Workstream A lands.

1. **EMA of weights** (~1 hr). Standard DDPM (Ho et al. use decay
   0.9999) and famously a large sample-quality lever — v1 never
   tried it. Host-side only: `F32.ema` already exists (v1 uses it
   for BN stats). Trainer keeps a shadow param buffer, saves
   `{pfx}_params_ema.bin`; sampler prefers the EMA file. This is
   the single cheapest untested lever in the whole plan.
2. **Fixed-seed sample grid every N epochs** (~1 hr). v1's core
   lesson is that MSE can't pick checkpoints. Fold the DDIM sampler
   into the trainer (or shell out to the sample exe) every 10 ep
   with a fixed noise seed, writing `samples_ep{N}.ppm`. All
   subsequent A/Bs become honest.
3. **Horizontal-flip augment** (~30 min, host-side). Standard for
   CIFAR DDPM; v1 trained with `augment := false`.
4. **Budget/capacity**: v1 established the quality threshold sits
   around ep 60–70 at base80. Default v2 run: base96–128, 100 ep,
   overnight on mars (base128 ≈ 500 ms/step × 1562 steps/ep ≈ 13 h
   per 60 ep — split across two nights or duty-cycle on klawd per
   the thermal notes). `gradAccumSteps` codegen exists if a larger
   effective batch is wanted, but it's unvalidated on GPU — don't
   make it load-bearing here.
5. **GroupNorm** (defer unless A stalls). The principled fix for
   mixed-timestep batches, but it's a new norm primitive with
   forward+backward emit (~1 session). BN demonstrably *works* at
   base80; only pay this if conditioning + EMA + budget still
   plateau below demo quality.

## Workstream C — attention, second attempt

Only after Gate A passes. Two changes, applied together this time:

1. **Layer-scale init on the transformer block.** Per-channel γ
   initialized ~0 multiplying the MHSA and FFN outputs before the
   residual add, so the block starts dormant-by-construction and
   grows in — directly targets the v1 failure mode. The emit
   pattern already exists inside `convNextStage`
   (`MlirCodegen.lean:1418`); extract it as an optional
   `layerScaleInit : Option Float` on `transformerEncoder`.
   ~1–2 hr.
2. **Time conditioning at the bottleneck** — free once Workstream A
   lands (a `.timeCondAdd` before `spatialFlatten`).

Gate C: base64+attn+timeCond+layerScale vs base80+timeCond at
matched wall-clock, 50 ep, fixed-seed grids. Also log the mean |γ|
per epoch — if γ stays ~0 the block is still bypassed and we finally
have a direct measurement instead of vibes. If attention still
loses, document it and ship v2 as conv-only; the demo doesn't need
attention to be honest.

## Workstream D — class-conditional + classifier-free guidance

The biggest *demo-value* jump per hour once the backbone is solid:
"give me a grid of horses" beats "here are some unconditional
samples" in every showing.

- Conditioning: reuse `.timeCondAdd` verbatim — concatenate a
  learned or one-hot class embedding onto the time-embedding side
  input (`[B, D_t + 10]`). Zero new codegen beyond Workstream A.
- Train with label dropout (10% → zero embedding) so one checkpoint
  supports classifier-free guidance.
- CFG at sampling: two forward passes per step (cond + uncond),
  `ε = ε_u + w·(ε_c − ε_u)` — host-side affine, `ddimStep` shape
  already fits. Guidance scale w as a CLI arg; a w-sweep grid
  (w ∈ {0, 1, 2, 4}) makes a great figure.
- New mains: `MainCifarDdpmCondTrain/Sample`. ~1 session on top of A.

## Workstream E — eval beyond eyeballing

Skip true FID (Inception-V3 port is weeks). Instead a
**self-contained Fréchet distance in our own feature space**: the
repo already has trained, *verified* CIFAR-scale classifiers — use a
frozen checkpoint's penultimate features, compute Fréchet distance
between 10K generated samples and the CIFAR test set. Not comparable
to paper FID, but monotone-ish in quality, fully reproducible inside
the repo, and on-brand ("scored by our own verified classifier").
Host-side mean/cov + a small matrix-sqrt (2×2 blocks or
power-series; feature dim can be kept small by projecting). ~1
session. Report it alongside grids in RESULTS.md; never let it
replace the fixed-seed grids (the MSE lesson generalizes: trust no
scalar alone).

Also: **DDPM stochastic sampler** (η>0) as a sampler flag — one
extra `sampleNoise` + affine per step, ~1 hr — so the demo can show
the classic 1000-step ancestral chain and the 50-step DDIM shortcut
side by side.

## Workstream F — verified-gradient tie-in (stretch, on-brand)

The DDPM train step is currently outside the proof story: the
`useDdpm` MSE branch has no VJP theorem and no faithful-sweep entry.
Two graded options:

1. **Cheap**: add the per-pixel-MSE loss VJP (`d = 2(ŷ−y)/N`) to the
   proved-VJP set — it's a composition of subtract/square/mean
   already in the library. Then the DDPM train step is "verified
   loss gradient over a verified backbone", same claim strength as
   the classifier demos. ~1 session.
2. **Stretch**: a descent-at-trained-weights witness for one DDPM
   step (the machinery from the robustness/descent arc), giving the
   generative demo the same level-3 seal as the CNN. Only if the
   proof side wants a generative example; not on the demo's
   critical path.

## Sequencing

```
Phase 0 (1 session, no codegen):   B1 EMA + B2 fixed-seed grids + B3 flip
                                   → rerun base80 70ep as the new reference
Phase 1 (1–2 sessions):            A  timeCondAdd primitive + Gate A run
Phase 2 (1 session + overnights):  B4 capacity/budget push on the Gate-A winner
Phase 3 (1 session):               C  attention retry (layer-scale + t-cond) + Gate C
Phase 4 (1 session):               D  class-conditional + CFG
Phase 5 (1 session):               E  Fréchet-in-own-features + DDPM sampler
Phase 6 (optional):                F  proof tie-in
```

Phases 0–2 are the committed core (≈3–4 sessions of code + overnight
runs) and alone should clear the "demo-quality grid" bar. 3–6 are
independently droppable.

## Deliverables

- `demos/MainCifarDdpmV2Train/Sample.lean` (t-cond backbone; retire
  the sincos mains to `historical/` — the A/B is documented, keep
  plain + attn for reproducibility)
- `MainCifarDdpmCondTrain/Sample` (Phase 4)
- New reference grids + the w-sweep figure in `demos/figures/`,
  RESULTS.md row with the Fréchet score
- Bestiary: promote the `Diffusion.lean` prose from "our NetSpec
  can't express per-block time injection" to pointing at
  `.timeCondAdd`

## Out of scope (unchanged from v1, plus)

Latent diffusion / VAE, text conditioning, video, score-based
reparameterization, true Inception FID, flow matching (tempting as
a "same backbone, simpler loss" follow-on — if v2 lands cleanly it
deserves its own one-pager, not a rider here).
