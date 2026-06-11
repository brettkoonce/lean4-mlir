# Planning — `mobilenet-v2-train` → `mobilenetv2-verified-adam`

Bring MobileNetV2 to the same standard `vit-verified-adam` reached (see
`planning/vit_train_to_vit_verified.md`): a verified-AdamW trainer that
**matches the non-verified `mobilenet-v2-train` reference**, differential-tested
on identical data. mnv2 is the cheapest next target (smallest imagenette net, 82
params) — but it carries the one thing ViT didn't: **BatchNorm**.

---
## STATUS (2026-06-11) — exact-parity DONE + GPU-validated (UNCOMMITTED, awaiting sign-off)
`mobilenetv2-verified-adam` originally landed (committed `90e9bb6`, pushed) with **per-sample BN** —
the documented caveat. Exact BN parity (true `[0,2,3]` batch-norm + running-stats eval, matching r34/
enet) is now **implemented and GPU-validated** on the 7900 XTX (gfx1100): all 4 artifacts
iree-compile; a fresh run trains cleanly (epoch-1 loss 2.18, descending into epoch 2 at ~1.9), and
the running-stats eval through `@mobilenetv2_fwd_eval` is **non-degenerate: epoch-1 val_acc
1283/3904 = 32.9%** (vs ~10% for degenerate batch-BN-eval on the class-sorted val set; right in the
r34 37.4% / enet 40.1% family). UNCOMMITTED, awaiting Brett's sign-off. What changed:

- **Adam train step** (`tests/TestMobilenetV2TrainPC.lean`): the BN proof tokens `.bnPerChannelF` /
  `.bnPerChannelBack` are replaced by **hand-emitted** flat↔NCHW true-batch-norm fragments
  (`bnB`/`bnBackB`, reduce `[0,2,3]`, nf=BS·H·W). `bnB` saves x̂/istd/nf/γb + `[oc]` batch sums;
  `bnBackB` reuses them and **folds dγ/dβ** (the old separate `bnParamGrad` recompute is gone → the
  artifact shrank 383KB→335KB). Per-layer batch mean/var are carried out in passthrough slots
  (`bnLayers`, 20 layers), so the generic `trainAdamSched` running-stats subsystem drives it.
  **This is the proof-token-BN tradeoff the caveat warned about** — BN forward+backward stop being
  proof-rendered; the convs, depthwise, relu6, residual, gap and dense all STAY `pretty`-rendered.
- **Eval forward** (`tests/TestMobilenetV2Fwd.lean`): `bnPC` swapped to true batch-norm; new
  `@mobilenetv2_fwd_eval` (affine BN consuming per-layer running μ/var, forward-order stat sig).
- **SGD train step** (`tests/TestMobilenetV2Train.lean`): `bnPC`/`bnBackPC` swapped to true
  batch-norm (the exact r34 diff) — keeps the `mobilenetv2-verified` SGD exe self-consistent.
- **Spec** (`LeanMlir/VerifiedNets.lean`): `bnChannels := #[16, 64,64,24, …, 128]` (20 layers).

Validation: all 4 files typecheck + render; adam params==returns (289); 40 BN grad names
defined+consumed; zero undefined SSA refs; 246 reshapes element-consistent; stat-in dummies pure
passthrough; 40 stat divides = 20 bnChannels × 2. **GPU (gfx1100, lean4-jax FFI + IREE 3.12, rocm):**
all 4 artifacts iree-compile clean (the `--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`
rocm workaround is needed now that BN reduces `[0,2,3]`); fresh run epoch-1 loss 2.18 → epoch-2 ~1.9,
running-stats val_acc 32.9%. Run recipe: clear `.lake/build/mobilenetv2_adam_ckpt.bin{,.epoch}`, then
`PATH=…/lean4-jax/.venv/bin IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 .lake/build/bin/mobilenetv2-verified-adam
…/mnist-lean4/data` (see `[[running-verified-trainers-locally]]`).

**For r34 + enet** (committed `293cb1c`, `cceea14`): same true `[0,2,3]` batch-norm + running-stats
eval — the SHARED INFRA in `trainAdamSched` (a `<slug>_fwd_eval.mlir`, batch mean/var in PASSTHROUGH
slots keeping the FFI param/return symmetry, EMA `bnMom` 1.0→0.1 into `runningBnStats`, the
`bnChannels : Array Nat` field, empty for LN nets). mnv2 now reuses this subsystem too — only the
mnv2-side render (hand-emitted BN + fwd_eval + bnChannels) was the remaining work, now done. See
`[[verified-adam-net-progress]]` + the r34 commit `293cb1c` for the template.

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
