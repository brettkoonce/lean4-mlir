# Planning — `mobilenet-v2-train` → `mobilenetv2-verified-adam`

Bring MobileNetV2 to the same standard `vit-verified-adam` reached (see
`planning/vit_train_to_vit_verified.md`): a verified-AdamW trainer that
**matches the non-verified `mobilenet-v2-train` reference**, differential-tested
on identical data. mnv2 is the cheapest next target (smallest imagenette net, 82
params) — but it carries the one thing ViT didn't: **BatchNorm**.

---
## STATUS (2026-06-11) — DONE, with one BN caveat
`mobilenetv2-verified-adam` is **built, GPU-validated, committed (`90e9bb6`), and pushed** — along
with the whole verified-AdamW set (vit, mnv2, enet `19c487b`, r34 `845077b`, convnext `26e91d1`).
The §1 AdamW packed render and §3 Main/recipe/exe below all landed as planned (the `emitAdamV` swap
+ packed `[θ|m|v]` + scalar-tail through `trainAdamSched`).

**§2 (BatchNorm) was the swing factor, exactly as predicted — and it split three ways:**
- **mnv2 kept per-sample BN** (reduce `[2,3]`, each image self-normalized). This is a *different
  operation* than the reference's true batch-norm (reduce `[0,2,3]`), so mnv2's train curve doesn't
  tightly match the reference — but its **eval works fine** (self-normalizing → not degenerate on the
  class-sorted val set). **Accepted as a documented caveat**, because mnv2's BN is rendered through
  PROOF TOKENS (`.bnPerChannelF`) and `.bnBatchF` has no `pretty`/emit case (no batch-back token
  either) — so giving mnv2 batch-norm means HAND-rebuilding it in the proof-token PC render, trading
  away its proof-rendered-BN property, for the weakest net.
- **r34 + enet got full exact BN parity** (committed `293cb1c`, `cceea14`): true `[0,2,3]` batch-norm
  (r34 swapped; enet already was) **+ running-stats eval**. Running-stats is now SHARED INFRA in the
  generic `trainAdamSched` driver — a new `<slug>_fwd_eval.mlir` (affine BN with per-layer running
  μ/var), the adam step carries batch mean/var in PASSTHROUGH slots (keeps FFI `#out=#in`), the
  driver EMAs them (`bnMom` 1.0→0.1) into `runningBnStats`, and a `bnChannels : Array Nat` field
  drives it (empty for LN nets). Fixed degenerate sorted-val eval: r34 16%→37.4%, enet 12%→40.1%.

**If mnv2 exact-parity is wanted later:** the driver + metadata are done; only the mnv2-SIDE render
work remains — hand-emit batch-norm fwd+bwd into `TestMobilenetV2TrainPC` (replace the
`.bnPerChannelF`/`.bnPerChannelBack` `pretty` calls with hand-emitted batch fragments, like r34/enet),
add `bnLayers` + train-step stat passthrough, a `mobilenetv2_fwd_eval` render, and set `bnChannels`
on `mobilenetv2Verified`. See `[[verified-adam-net-progress]]` memory + the r34 commit `293cb1c` for
the template.

---
## Headline: mnv2 reuses ~80% of the ViT spine; the real new work is BatchNorm.

The ViT effort built a net-generic spine. mnv2 reuses almost all of it.

## Reused for free (already net-generic)
- **Same optimizer.** mnv2's reference (`MainMobilenetV2Train.lean`) is **AdamW**:
  `useAdam=true`, lr 1e-3, wd 1e-4, cosine, warmup 3, label-smoothing 0.1, augment,
  80 ep, bs 32 — and (like vitTinyConfig) **no EMA, no grad-clip**. So `emitAdamV`
  and the whole `trainAdamSched` driver (scheduled lr + bias correction, shuffle,
  aug, in-graph loss logging, checkpoint/resume) apply **directly**.
- **Recipe ≈ identical** to ViT's — only `lr 1e-3` (vs 3e-4) and `warmup 3` (vs 5).
  A two-number change in a new Main.
- **Loader already fixed** — the 224²/256² env var (`LEAN_MLIR_IMAGENETTE_TRAIN`)
  resolves the "short read" that had `mobilenetv2-verified` listed as broken
  ([[running-verified-trainers-locally]]).
- **Proof side done** — mnv2's verified VJP + param-grad closes exist
  (`MobileNetV2Close`/`MobileNetV2RenderPC`/`MobileNetV2ChainClose`,
  `invres_render_*_chain_certified`, `mnv2_render_*_certified`,
  `mobilenetv2FwdGraphFullPC_faithful`). ReLU6 two-sided kinks are already handled by
  the render's subgradient (`selectMid`). The whole-net VJP witness
  `mobilenetv2_has_vjp_at` is conditional/degenerate — fine for training (the render
  computes the real gradient at the real weights; the proof certifies it at smooth
  points), a doc caveat only.

## Net-specific work (in order)

### 1. mnv2 AdamW packed render
The analogue of `vitTrainStepModuleAdamSched`, but mnv2 renders through the **PC/SHlo
path** (`tests/TestMobilenetV2TrainPC.lean` → `verified_mlir/mobilenetv2_train_step.mlir`),
not ViT's hand-written strings. Work: take mnv2's already-rendered forward / backward /
softmax-CE cotangent, and bolt on:
- the AdamW update via `ViTRender.emitAdamV` (generic per-param, given the param/grad
  SSA names + shapes — mnv2 exposes these analogously to `vitParamNames`/`vitGradNames`),
- the packed `[θ|m|v]` layout + scalar `lr/bc₁/bc₂` slots + the in-graph smoothed-CE
  loss output (same mechanism as the ViT sched render),
- label smoothing in the cotangent.
The fiddly part is integrating `emitAdamV` (a string emitter) into mnv2's render
mechanism — but it's the same op-graph, so the SSA-name plumbing is the whole job.
`trainAdamSched` already drives any `VerifiedNet` (uses `net.specs`/`slug`/`paramShapes`),
so a `mnv2-verified-adam` Main just points it at the rendered mnv2 AdamW step.

### 2. BatchNorm running stats — the swing factor (ViT had none)
mnv2 has per-channel BN; the reference (`Train.lean`, "Adam + cosine-LR +
**running-BN-stats** training loop") maintains running mean/var (BN EMA) and uses them
at eval. **The verified driver has none** (it batch-norms everywhere). Consequence:
- **Training / loss curve matches for free** — both use batch BN in train mode, so the
  differential *loss-curve* test should pass with zero BN work.
- **Eval / val-accuracy is where it diverges** — the reference evals with running stats;
  verified would eval with the eval-batch's own stats (noisier, not exact parity).
- **The fork:**
  - **(a) batch-BN eval** — reuse the train forward at eval. Cheap, gets a trainer whose
    *loss curve* tracks the reference; val-acc will be close but not exact.
  - **(b) running-stats BN** — thread BN EMA each step (the reference's `F32.ema` on batch
    stats; there's a `_bn_stats.bin` companion convention in `Train.lean`) + an
    **inference-BN forward** that consumes running mean/var. Needed for exact val-acc
    parity; a render variant (affine-only BN with passed-in stats) + driver threading.
    Possibly a small new faithfulness lemma (inference-BN = the affine map).

### 3. Main exe + recipe-match + differential test
- `MainMobilenetV2VerifiedAdam` + `lean_exe mobilenetv2-verified-adam`; recipe = mnv2's
  (lr 1e-3, warmup 3, else as ViT).
- Diff-test vs `mobilenet-v2-train` on identical 256² Imagenette: **loss curve first**
  (should track immediately), then val-acc once BN stats are in. The diff-test is the
  safety net that catches mnv2's own "shuffle-bug equivalent."

## Effort & the one real decision
- Reused: optimizer, driver, loader, proofs — done.
- New: AdamW render integration into the PC path (moderate) + **BN stats** (moderate→
  significant for exact eval parity; ~zero if batch-BN eval is acceptable).
- ~1–2 days, BN the swing. Recommended path: get the **loss-curve diff-test passing first**
  (render + batch-BN, no stats), confirm tracking, then add running-stats BN for val-acc parity.

## After mnv2 — DONE (all imagenette nets)
The template applied to the rest as planned, cheapest-first by param count: mnv2 (82) → resnet34
(146) → convnext (180) → efficientnet (262), all AdamW-reference, all reusing the spine — all
**committed + pushed** (see STATUS at top). convnext is all-smooth like ViT (LayerNorm → exact
parity, no BN gap); r34 + enet got exact BN parity; mnv2 keeps per-sample BN (caveat).

**Remaining (next clean session):** the small **cnn/cifar** nets (MNIST/CIFAR, not imagenette) —
their references also use AdamW (lr 1e-3, warmup 1–2, no label-smoothing, bs 128), same `emitAdamV`
swap recipe, but the render path differs (no per-net `allParams`/`sgd`/`cot` test — likely a shared
`CnnRender`-style library; needs investigation first). Lower value (re-proves the swap on small
nets, no new capability). `mnist-linear` uses a separate 2-param `trainLinear` driver — skip.
