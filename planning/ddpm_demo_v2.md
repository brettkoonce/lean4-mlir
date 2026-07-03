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

**Status 2026-07-03: SHIPPED (a design better than the plan below).**
`.timeCondAdd channels nFreq` is implemented and validated. The plan
below called for a new `[B, D_t]` side input threaded through the
DDPM ABI — but that would force a `libiree_ffi.so` relink (the
fragile stale-FFI path). Instead, **the timestep is read in-graph**
from the last channel of the network input (`prependTChannel`'s
`t/Tmax` plane, sliced from `%x_flat`/`%x` at offset `(cIn-1)·H·W`),
the `2·nFreq` sin/cos embedding is built in the MLIR (frequencies
`π·2^k`), projected through a learned dense `[2·nFreq, C]`, and
broadcast-added onto the `[B, C, H, W]` feature map — **zero ABI
change**, same philosophy as the tinygpt in-graph one-hot. W and b
init to **zero** so conditioning starts as a no-op and grows in (the
layer-scale trick Workstream C wanted, for free). Backward: residual
add ⇒ gradient passes through unchanged, plus standard dense-transpose
param grads (`d_W = embᵀ·d_proj`, `d_b = Σ d_proj`) — the same idiom
as the FD-verified tokenPositionEmbed/lmHead. Validation: iree-compile
passes for train + eval (strict shape check over all ~15 codegen
sites), param count exact (+19,040 for the 6-site base80 variant =
Σ 17·C), loss drops on a smoke run. Follow-up: an isolated FD gradient
test (awkward because the timestep enters via the input t-channel; the
backward mirrors verified patterns by construction). Use it via
`cifar-ddpm-train data <epochs> 0 tc` (`tinyCifarDdpmTC` spec).
Remaining for a full Gate A: a matched-budget 50-ep train + fixed-seed
grid vs the base80 baseline.

The original side-input design (superseded, kept for context):

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

### Gate A — how to run it (2026-07-03)

The codegen is done; Gate A is now purely two training runs + a
visual compare. Both use the Phase-0 trainer (EMA + fixed-seed grids
+ hflip), same seed, same 50-ep budget — the ONLY difference is the
spec (`tinyCifarDdpm` vs `tinyCifarDdpmTC`, selected by the trailing
`tc` arg). Grids go to separate dirs (`runs/ddpm_v2_base80/` vs
`runs/ddpm_v2_tc/`), checkpoints to distinct build prefixes (spec
names differ), so the two runs never collide and can run in parallel
on two GPUs.

```bash
lake build cifar-ddpm-train
# Baseline (plain base80, ~2.94M params):
CUDA_VISIBLE_DEVICES=0 .lake/build/bin/cifar-ddpm-train data 50      &
# timeCondAdd variant (~2.96M params, 6 per-block conditioning sites):
CUDA_VISIBLE_DEVICES=2 .lake/build/bin/cifar-ddpm-train data 50 0 tc &
wait
```

Cost: base80 is **~19 min/epoch host-bound on klawd** → ~16 h per
run (they overlap on 2 GPUs, so ~16 h wall). Each writes
`samples_ep{5,10,...,50}.ppm` from EMA weights every 5 epochs, so the
comparison is watchable mid-run (ep 20–30 already tells the story)
and the runs survive interruption (checkpoint every 5 ep). On mars
(7900 XTX) this is materially faster — prefer it if free.

Judging (visual, per the "MSE ≠ quality" lesson — do NOT rank by the
loss printout):
- Put `runs/ddpm_v2_base80/samples_ep50.ppm` beside
  `runs/ddpm_v2_tc/samples_ep50.ppm` (convert to PNG:
  `convert x.ppm x.png`). Also compare the ep-25 pair.
- **Win** = the tc grid shows visibly more object structure /
  cleaner category shapes at matched epoch than plain base80 (which
  v1 characterized as yellow animal-ish blobs at 50 ep). Log the
  call + attach both grids to RESULTS.md.
- **Loss ≈ same, tc grid better** → conditioning is the lever;
  proceed to Workstream C (attention retry) with t-cond in place.
- **Loss ≈ same, grids indistinguishable** → per-block conditioning
  alone isn't enough at this budget; before re-planning, try the
  shared time-MLP trunk (Workstream A "Optionally") and/or a longer
  70-ep run (v1's quality threshold sits at ep 60–70).
- **tc worse** → likely the zero-init hasn't grown in at 50 ep;
  check whether the timeCondAdd `%d_W` grads are non-trivial (they
  are emitted per site) and consider a small non-zero W init.

Not yet run (the remaining Gate A work): launch the two commands
above and record the verdict. Everything upstream (primitive,
compile, param accounting, smoke train) is done and committed.

## Workstream B — training recipe (cheap, do first)

**Status 2026-07-03: B1–B3 DONE (Phase 0 shipped).** All three
zero-codegen levers landed in `demos/MainCifarDdpmTrain.lean` and
smoke-verified end-to-end (2.94M-param base80, train+eval compile,
20-step loop, EMA `_params_ema.bin` written, fixed-seed grid
rendered, hflip clean):
- **B1 EMA** — shadow buffer `ema = 0.9999·ema + 0.0001·p` via the
  existing `F32.ema`; saved as `_params_ema.bin`, and the periodic
  grids sample from it.
- **B2 fixed-seed grid** — `sampleGrid` folds the 50-step DDIM
  sampler into the trainer, fixed noise seed `0xfeed5eed`, EMA
  weights → `runs/ddpm_v2_base80/samples_ep{N}.ppm` every 5 epochs.
  The honest-A/B infrastructure the whole plan leans on.
- **B3 hflip** — new `F32.hflipNCHW` FFI (per-image p=0.5), applied
  to each batch.
Plus periodic checkpointing (params/EMA/BN every 5 ep) so long runs
survive interruption, and a `maxSteps` arg for smoke tests. A base80
40-epoch reference run is baking (klawd GPU 2, ~18 min/epoch
host-bound → ~12h; grids every 5 ep show the EMA sample evolution).
Still open: wiring the standalone sampler to prefer `_params_ema.bin`
(currently the in-trainer grid is the EMA view), and B4/B5 below.

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

## Running it (Phase 0, reproduce)

```bash
# CIFAR-10 must be at data/cifar-10/ (download_cifar.sh). Then:
lake build cifar-ddpm-train

# Train base80 (2.94M params) with EMA + fixed-seed grids + hflip.
# args: [dataDir] [epochs] [maxSteps>0 caps steps/epoch for smoke].
# ~18 min/epoch host-bound on klawd (RTX 4060 Ti); base80 quality
# threshold is ep ~60–70 (v1). Grids + checkpoints land every 5 ep.
CUDA_VISIBLE_DEVICES=2 .lake/build/bin/cifar-ddpm-train data 40

# Outputs:
#   runs/ddpm_v2_base80/samples_ep{5,10,...}.ppm   ← EMA sample grids
#   .lake/build/<pfx>_params_ema.bin               ← EMA weights
#   .lake/build/<pfx>_params.bin / _bn_stats.bin   ← raw weights + BN
# Smoke test (compile + 20 steps + one grid): … cifar-ddpm-train data 1 20
```

The in-trainer grid already samples from the EMA weights, so
`samples_ep{N}.ppm` is the EMA view over training. The standalone
`cifar-ddpm-sample` exe still reads `_params.bin` (raw) — the small
open item noted in Workstream B is pointing it at `_params_ema.bin`.

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
