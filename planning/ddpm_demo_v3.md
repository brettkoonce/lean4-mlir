# ddpm_demo_v3.md — DDPM demo, third pass

Goal: v2 fixed the sampler and produced the project's best grid so far
(base80 ep90, constant-LR run). v3 makes that quality *reproducible
and steerable*: a normalization layer that doesn't fight diffusion
(GroupNorm), weight-EMA that actually works at sampling time,
time-conditioning that *wins* its A/B instead of losing it, and then
the two demo-value multipliers on top (class-conditional + CFG, and
the self-scored Fréchet metric). Everything here builds on v2's
honest-A/B infrastructure: fixed-seed raw-weight grids every 5 ep.

Prerequisite reading: `planning/ddpm_demo_v2.md` (Gate-A RESOLVED
section + matched-budget verdict + Phase-2 wander finding). This doc
assumes those results and does not repeat them.

## Where v2 landed (recap)

Shipped and validated:
- **Sampler regression root-caused + fixed** (committed 408f15a): the
  Gate-A confetti was EMA params paired with raw-weight BN running
  stats — a normalization mismatch DDIM amplifies into noise. Grids
  now render from raw weights; the same ep-50 checkpoint that looked
  like confetti produces clear cars/animals.
- **η-generalized sampler**: `cifar-ddpm-sample` takes `ddpm` (η=1
  ancestral, default 1000 steps), `<nat>` step count, `ckpt=<pfx>`
  (archived checkpoints), `tc`, `ema` (diagnostic). Seed-scramble
  gotcha: per-step noise seeds must be Knuth-multiplied or the
  correlated xorshift streams accumulate into color speckles.
- **Gate A, matched 50-ep**: `.timeCondAdd` LOSES to plain base80
  (replicated across two runs: saturated color-mode collapse). The
  primitive is correct code but the wrong architecture — see C.
- **Phase-2 100-ep constant-LR**: color-phase wander at dead-flat MSE
  (grid cycles global color modes every ~5 ep; best ep90, magenta
  collapse ep100). Cosine LR wired into the trainer as the default
  (`cosineDecay := true`, per-step, Train.lean:560 idiom).
- **Trainer variants**: `b96` capacity spec (`tinyCifarDdpm96`,
  ~4.2M params).

Interrupted 2026-07-10 (GPUs handed back), resumable-by-rerun:
- base80 100-ep **cosine** leg — stopped at ep~8. The v3 must-rerun.
- base96 100-ep **constant-LR** leg — stopped at ep~72; grids through
  ep70 live in `runs/ddpm_v2_base96/` (read with the constant-LR
  wander caveat).

Archives (all with checkpoints + grids): `runs/ddpm_v2_gateA_matched/`
(base80 + tc ep50, `ep50_*` sampler symlinks),
`runs/ddpm_v2_base80_100ep_constlr/` (incl. the ep90 keeper grid),
`runs/ddpm_v2_{base80,tc}_emabug/` (the confetti evidence).

## The v2 through-line: every failure was BatchNorm

Three independent v2 failures share one root:

1. **EMA confetti** — BN running stats don't match EMA weights.
2. **tc loss** — per-image, per-timestep uniform offsets pollute BN
   batch statistics (the batch mixes independent timesteps, so the
   per-channel batch mean/var depend on the batch's t draw); at eval
   the sampler feeds a whole batch at the SAME t, off-distribution.
3. **Color-phase wander** — global color modes are exactly the
   degrees of freedom BN leaves loosest (per-channel affine on top of
   batch-dependent normalization), and a constant LR lets them drift.

The Bestiary (`Bestiary/Diffusion.lean`) said it from the start: real
DDPM UNets use GroupNorm. v2 deferred it ("BN demonstrably works at
base80"); v3 stops deferring. GroupNorm is batch-independent, so it
simultaneously: makes EMA-weight sampling consistent (no running
stats to mismatch — there ARE none), makes mixed-timestep batches
statistically sane, and removes the eval/train BN gap entirely.

## Workstream A — GroupNorm primitive (the unlock)

New layer kind `.convGn ic oc k s pad groups` (or a `norm` enum on
the existing conv block): GroupNorm(G) over channel groups, per-image
— forward + backward emit, no running stats, no batched-stats ABI.

- Forward: reshape `[B, C, H, W] → [B, G, C/G·H·W]`, per-(image,
  group) mean/var, normalize, per-channel γ/β. All within-image ⇒
  the emit is per-example (no cross-batch reduce, UNLIKE convBn).
- Backward: the standard GN VJP (same shape as LN-over-groups; the
  repo already has LayerNorm fwd+bwd emit patterns from ConvNeXt/ViT
  to crib from — `lnGamma/lnBeta` machinery).
- G = 8 default (Ho et al. use 32 on wider nets; base80's thinnest
  stage is 80 channels → 8 or 16 both divide cleanly; make it a spec
  param).
- Trainer: GN specs have `nBnStats = 0`; the BN-stats plumbing
  (extract/EMA/append) must no-op cleanly.
- Est. ~1 session codegen + the pidx/shape audit, then compile-check
  all sites like timeCondAdd did.

**Gate A3**: base80-GN vs base80-BN, matched 50 ep, cosine LR, same
seed. GN wins if grids are ≥ BN quality AND the EMA-weights grid (now
legal: no stats to mismatch) beats the raw grid — that second clause
is the real prize, it unlocks the papers' single biggest quality
lever. If GN loses on quality at matched budget, fall back to A′.

## Workstream A′ — EMA via BN-recalibration (cheap fallback/bridge)

If GN slips or loses: make EMA usable on the BN net. After training,
run a forward-stats pass — ~200 batches through the *train-step
graph* with lr := 0 under EMA weights (zero codegen: the train step
already returns batch BN stats; lr=0 makes it read-only) —
accumulate running stats, save as `_bn_stats_ema.bin`, sample EMA
weights + recalibrated stats. ~half a session, host-side only.
Worth doing even if GN lands, as the A/B that *measures* how much of
the EMA win GN captured.

## Workstream B — finish Phase 2 (recipe + hero run)

1. **Rerun the cosine base80 100-ep leg** (interrupted at ep8). Read:
   does the cosine endpoint freeze the color-phase wander (ep95 ≈
   ep100 ≈ stable palette)? If yes, recipe settled; if no, the wander
   is not LR-driven — prioritize A/A′ (EMA is the other damper).
2. **Capacity read**: finish or judge base96 (constant-LR grids
   through ep70 already on disk; a cosine rerun only if base96-ep70
   clearly beats base80-ep70 — check first, it's free).
3. **Checkpoint-resume** (~1 hr, pure host): the trainer already
   writes params/EMA/BN every 5 ep; add `resume` arg that loads them
   + skips to the saved epoch. Two multi-hour runs died this cycle;
   stop paying that tax. (Adam m/v aren't checkpointed — either add
   them to the 5-ep write or accept a warm-restart moment.)
4. **Hero run**: best config from 1+2 (+A if landed) × 150–200 ep →
   the README grid, DDIM-50 and DDPM-1000 side by side.

## Workstream C — conditioning done right (FiLM inside the block)

v2's `.timeCondAdd` verdict: a spatially-uniform per-channel *add*
between stages lets the net use time as a global color knob — and
that's what it did (saturated color-mode collapse, twice). The fix is
where and how real UNets inject time:

- **FiLM scale+shift, inside the residual block, after the norm**:
  `h = norm(conv(x)); h = h·(1+s(t)) + b(t); h = relu(h)` with
  `s, b` zero-init dense projections of the sin/cos embedding.
  Post-norm placement means the injection can't be normalized away
  (pre-norm injection into BN is literally erased batch-wise) and
  scale (not just shift) modulates *features*, not color.
- Requires opening up `unetDown`/`unetUp` (currently monolithic
  emit): either add an optional `timeCond : Bool` to their emit
  (injection sites inside), or decompose the spec into primitive
  layers + explicit skip push/pop layer kinds. The former is less
  surgery; the latter is more honest NetSpec design. Decide at
  implementation time.
- Shared time-MLP trunk (`emb → dense → silu → dense`, per-site
  heads) as part of the same change — v2 deferred it, every paper
  has it, and per-site zero-init heads keep the growing-in property.
- **Depends on A** (GN) ideally: FiLM-after-GN is the literature
  pattern. FiLM-after-BN still suffers the mixed-t stats pollution.

**Gate C3**: base80-GN-FiLM vs base80-GN, matched 50 ep cosine. Win =
visibly better structure (v2 criteria). This gate has now failed once
(v2 tc) — if the *proper* injection also loses at this scale, write
the negative result up honestly and ship v3 unconditional; the ladder
says the collapse IS the signal.

## Workstream D — attention retry (unchanged from v2)

Only after C3 passes. Layer-scale-init transformer block at the
bottleneck + FiLM t-cond in place; log mean |γ| per epoch (the direct
"is the block alive" measurement). Criteria per v2 Workstream C.

## Workstream E — class-conditional + CFG (unchanged from v2)

Reuses C's FiLM path verbatim (concat one-hot/learned class embedding
onto the time embedding; 10% label dropout; CFG at sampling = two
forwards + host affine, w-sweep grid figure). Gated behind C3.

## Workstream F — Fréchet-in-own-features (recipe ready, independent)

Full recon recipe in v2 doc (Workstream E section): `cifar8_bn_1024`
classifier (72.17%, 1024-d penultimate, ckpt = θ|m|v slice), needs a
`stopBeforeLastDense` codegen twin of the GradCAM `stopAtGAP` path,
features projected to ~64-d, mean/cov/sqrtm via a tiny IREE matmul
vmfb (NOT new FFI — avoid the .so relink), Song-style power-series
sqrt. ~1 session, independent of A–E.

**Validation for free**: v2's archives are a known-ordered quality
ladder — emabug confetti < gateA ep50 < constlr ep90 > constlr ep100
(magenta). The metric must reproduce that ordering before it's
trusted on anything new. (This is the validation-ladder method
applied to the metric itself.)

## Workstream G — verified-gradient tie-in (unchanged from v2 F)

Per-pixel-MSE loss VJP into the proved set (cheap, ~1 session);
descent-at-trained-weights witness for one DDPM step (stretch).
Not on the demo critical path.

## Sequencing

```
Phase 0 (½ session + reruns):  B1 cosine rerun + B2 base96 read + B3 resume
Phase 1 (1–2 sessions):        A  GroupNorm primitive + Gate A3 (A′ fallback)
Phase 2 (1 session):           C  FiLM-in-block + time-MLP trunk + Gate C3
Phase 3 (overnight):           B4 hero run on the Phase-1/2 winner
Phase 4 (1 session):           E  class-conditional + CFG      (needs C3)
Phase 5 (1 session):           F  Fréchet metric                (anytime)
Phase 6 (1 session):           D  attention retry               (needs C3)
Phase 7 (optional):            G  proof tie-in
```

Phases 0–3 are the committed core. F is the best rainy-day slot (no
GPU contention, validates against archives). D and E are independently
droppable; E > D in demo value per hour.

## Out of scope (unchanged from v2, plus)

Latent diffusion / VAE, text conditioning, video, score-based
reparameterization, true Inception FID, flow matching (still wants
its own one-pager if the backbone lands), v-prediction / EDM-style
preconditioning (same "separate one-pager" rule — tempting because
it directly attacks the high-t x₀ amplification, but it changes the
training target and every A/B baseline with it).

## Housekeeping at handoff (2026-07-10)

Uncommitted at time of writing: trainer (cosine LR + b96 spec + raw
grids), sampler (η/ckpt/tc/ema args + seed scramble), v2 plan-doc
updates (Gate-A verdict, Phase-2 wander, FD recipe), this doc.
GPUs freed (both runs killed 2026-07-10 ~mid-day; rerun via
`run_ddpm_base80.sh`-style launches — cosine is now the compiled
default, no flag needed).
