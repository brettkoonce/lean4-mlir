# rsb_a2_resnet50.md — JAX trainer for RSB-A2 ResNet-50 (new-session handoff)

Build a **literal RSB-A2** ResNet-50 trainer in the Lean→JAX path, smoke-test each piece on
GPU, then (eventually) run it and add the net to the bestiary. RSB-A2 = the timm "ResNet
Strikes Back" 300-epoch recipe (Wightman et al. 2021) → **79.8% top-1** — the canonical modern
ResNet-50 baseline, and an honest one to have in the repo. Written 2026-06-22 as a fresh-session
plan; the decision + short gap summary also live in `planning/resnet50_imagenet.md` (RSB-A2 section).

Goal even if we never burn the ~60–65 hr run: a clean, smoke-tested RSB-A2 trainer + spec in the
bestiary. The run is a bonus.

---

## FIDELITY LEDGER — what's paper-faithful vs what deviates (latest: 2026-07-07)

Single source of truth for how close the R50 trainer is to the published timm recipe. Keep this
section current when faithfulness changes; the phased build history below is frozen at 2026-06-22.
Config lives in `jax/MainResnet50Imagenet.lean`; RandAugment/aug codegen in `jax/Jax/Codegen.lean`.

**Best measured result (RSB-A3 tier):** `rsb-faithful` recipe → **76.66% top-1 / 93.03% top-5**
(ep100), vs paper RSB-A3 **78.1%** = **−1.4 pt**. The 40.8%→76.66% recovery was giving LAMB its
design batch via grad-accum (eff bs2048); the saga is in memory `project_r50_a3_lowval_diagnostic`.

**Recipes** (positional arg to `resnet50-imagenet <recipe>`; `--help` lists them):
| recipe | tier | batch | notes |
|---|---|---|---|
| `default` | RSB-A2 300ep | 512 | full recipe, not yet run to completion |
| `short` | RSB-A3 100ep | 512 | LAMB starved at bs512 (→40.8%); kept as the naive baseline |
| `rsb-faithful` | RSB-A3 100ep | eff 2048 (512×4 grad-accum) | **the 76.66% run**; LAMB's design batch on 4×16GB |
| `true-2048` | RSB-A3 100ep | real 2048 (no accum) | needs ~80GB; removes the Ghost-BN approximation. NOT yet run |
| `a2-accum` | RSB-A2 300ep | eff 2048 (512×4 grad-accum) | A2's counterpart to `rsb-faithful` — the A2 to run on a 4-GPU box |
| `a1` | RSB-A1 600ep | eff 2048 (512×4 grad-accum) | 80.4% target; = `a2-accum` + 600ep / wd0.01 / mixup0.2 |
| `adam-probe` | A3 diagnostic | 512 | AdamW+wd-skip; used to confirm LAMB was the culprit |

### A2 / A1 cost on a 4-GPU box (measured 2026-07-25)

`default` (A2 @ bs512) is the same starved-LAMB regime that gave A3 40.8%, so A2/A1
should be run at effective bs2048. `a2-accum` and `a1` do that. Measured on
4× 4060 Ti, `scripts/step_probe.py` (compute only — excludes the tf.data pipeline):

| config | ms/step | peak | min/epoch | wall clock |
|---|---|---|---|---|
| A2 @ bs512 (`default`) | 358 | 7.41 GiB | 14.9 | 74.7 hr / 300ep |
| A2 @ eff 2048 (`a2-accum`) | 1427 | 7.59 GiB | 14.9 | 74.3 hr / 300ep |
| A1 @ eff 2048 (`a1`) | 1427 | 7.59 GiB | 14.9 | **148.6 hr / 600ep** |

Accumulation is per-epoch free: 4× the step time at 1/4 the steps. Peak is
unchanged from bs512, so **no big-memory card is needed** — unlike `a2-true-2048`
(~160 GB). Re-measured under a 12 GB budget (`MEM_FRACTION=0.548`): identical
7.41/7.59 GiB and 360/1427 ms — no rematerialization pressure, so this fits a
12 GB card (4070 / 3060) with ~1 GiB spare.

**Input-bound caveat.** This doc's own A2 figure is ~458 ms/step on 4× 4060 Ti vs
the 358 measured here — a 28% pipeline tax, far above the ~10% seen on ConvNeXt-T.
RSB-A2 has the heaviest CPU-side pipeline in the repo (geometric RandAugment +
3× repeated-aug + Mixup/CutMix). Faster GPUs do not shrink that floor, so a
faster card converts more of the run to input-bound and realizes less than its
FLOPs ratio.

**4× RTX 4070 estimate** (29.15 TFLOPS / 504 GB/s vs the 4060 Ti's 22.06 / 288 —
1.32× compute, 1.75× bandwidth; spec-derived, not benchmarked):

| tier | compute-bound best case | with the input floor | 
|---|---|---|
| A2 (300ep) | ~56 hr | ~77 hr |
| A1 (600ep) | ~113 hr | ~155 hr |

So A2 ≈ 2.5–3.5 days, A1 ≈ 5–6.5 days.

**Accuracy expectation.** Anchor on the measured −1.4 pt A3 gap (76.66 vs 78.1),
not the paper headline: A2 → **~78.4%** (paper 79.8), A1 → **~79.0%** (paper 80.4),
±0.5. Two risks the A3 validation did *not* cover: repeated augmentation is ON for
A2/A1 (`n0` for A3) and ours is a stream-level approximation of timm's index-level
RASampler; and the −1.4 pt attribution (BN regime / LAMB micro-details / bf16) is a
single data point at one tier.

**Disk.** A1 at 600 epochs with `LEAN_MLIR_CKPT_EVERY=1` writes 600 × 102 MB ≈ 61 GB
of `.bin` weights. Set `LEAN_MLIR_KEEP_BIN`.

**PAPER-FAITHFUL (matches timm):**
- Optimizer LAMB at its design batch (eff/real bs2048), lr 0.008@2048, cosine + 5ep warmup.
- BCE-with-logits over multi-hot mixup/cutmix targets; no label smoothing (RSB subsumes it).
- Aug pack: Mixup 0.1 + CutMix 1.0, RRC(0.08–1.0)+hflip, train@160 / eval@224 crop 0.95.
- `wdExcludeNormBias` (timm no_weight_decay skip-list: BN γ/β + biases). Was an A3-only-branch bug
  in the LAMB path; fixed (see `project_grad_accum_lever`).
- running-BN eval (train running mean/var, not eval-batch).
- **RandAugment — now literal `rand-m6-mstd0.5-inc1`** (closed 2026-07-07, `Jax/Codegen.lean`):
  op set = timm `_RAND_INCREASING_TRANSFORMS` (15 ops, no `Identity`, +`Invert` +`SolarizeAdd`);
  per-op apply prob 0.5; geometric interp BILINEAR (was NEAREST); TranslateX/Y `-Rel`
  (0.45×dim, was absolute 100px). `_RA_INC` handles the increasing magnitude mappings.

**KNOWN DEVIATIONS (deliberate / open):**
- **bf16 matmul+conv** — KEPT for perf (ares/cuDNN ~1.6× faster). Probed ≈ −0.1 pt vs fp32, so
  near-free; the paper is fp32/amp.
- **BN batch regime** — `rsb-faithful` uses Ghost-BN (each grad-accum micro-step normalizes over
  its own 512, not the full 2048) + running stats updated K×/optimizer-step (BN momentum
  compensated in codegen). `true-2048` on a single card instead normalizes over the full 2048 =
  a *larger* BN batch than timm's per-GPU BN (timm was multi-GPU → per-GPU batch 2048/n_gpu).
  Neither is a literal per-GPU-BN match; GhostBN kept for now (decision 2026-07-07).
- **Repeated Augmentation** — correctly OFF for A3 (`n0`); the A2 `default` recipe has it at 3×.

**Residual −1.4 pt attribution (best guess, post-RandAugment-fix):** BN regime (Ghost/K×-update or
single-card big-batch) + LAMB impl micro-details + bf16. The RandAugment op-set/interp deltas that
were on this list are now closed. Quantifying requires the `true-2048` run on an 80GB card.

---

## BUILD STATUS — all phases landed + smoke-tested (updated 2026-06-22)

Phases 1–5 are DONE on `main`, each smoke-tested on the ROCm box (2× 7900 XTX, gfx1100):
- **P1** bottleneck running-BN + stochastic-depth threading; `MainResnet50Imagenet.lean`
  (`resnet50-imagenet` exe). 25,557,032 params (exact torchvision R50), 53 BN layers.
- **P2** `repeatedAug` (tfds `flat_map(repeat K)` + reshuffle); verified exactly K independent copies.
- **P3** `OptimizerKind.lamb` (trust-ratio + decoupled WD); smoke loss decreased smoothly.
- **P4** `LossKind.bce` (BCE-with-logits over multi-hot, timm mean-over-B×C); manual cross-check matched.
- **P5** literal RSB-A2 `resnet50ImagenetConfig` (LAMB lr 5e-3@2048 → 1.25e-3@512, BCE, mixup0.1/
  cutmix1.0, RA m7-mstd0.5-inc1, repeatedAug 3, dropPath0.05, wd0.02, EMA0.9999, 300ep). End-to-end
  smoke passed; `supervise_r50_300ep.sh` + `eval_r50_full50k.py` written.
- Bug fixed along the way: `_aa_posterize` uint8/int32 shift dtype mismatch broke the whole geometric-
  RandAugment dataloader (latent — affected ConvNeXt/ViT too; their geo-RA had never run end-to-end).

**Measured warmup ETA (this ROCm box, 2× 7900 XTX, batch 512 = 2×256, bf16Conv=false):**
steady-state **~1.04 s/step** (3 consecutive 100-step intervals: 104s, 103s, 104s). At
total_steps = 2502/epoch × 300 = **750,600**, the full run is **~216 h ≈ 9 days** (≈43 min/epoch),
training-only. This is FAR above the 60–65 hr plan estimate (which assumed bf16-conv on the CUDA box).
→ Confirms R50 is conv-bound and belongs on the **CUDA box (ares, 6× 4060 Ti, bf16Conv=true)** for the
real burn; this ROCm box is for build + smoke only. (The ~10× gap from ideal also suggests MIOpen
fp32-conv and/or the 3× CPU RandAugment pipeline as ROCm-side bottlenecks — re-measure on CUDA.)

---

## Starting state (what's already landed — read first)

The phase-2 JAX codegen (`jax/Jax/Codegen.lean`) emits idiomatic JAX from a `NetSpec` +
`TrainConfig` (`LeanMlir/Types.lean`). Through 2026-06-22 these are DONE and on `main`:

- **Faithfulness gaps A–D** across the convnet/transformer sweep:
  - **A. running-BN stats** (eval normalizes with training running mean/var, not the eval batch).
    Gated `TrainConfig.runningBN`. Wired for `convBn` + `invres_block` (mnv2), `mbconv_block`
    (enet), and **`basic_block`/`basic_block_down`** (r34). GPU-validated incl. 2-GPU sharding
    (global stats confirmed). **`bottleneck_block` is the ONE un-threaded BN helper** → this plan.
  - **B. exp-decay LR** (`expLRDecayRate`/`expLRDecayEpochs`); **C. classifier dropout** (`dropout`);
    **D. RandAugment mstd0.5/inc1** (`randAugmentMstd`/`randAugmentInc`).
- **Subrun selector**: each imagenet trainer's `main` picks the tier via a positional recipe arg
  (`<exe> full` for short-default nets, `<exe> short` for paper-default nets), writing distinct
  `generated_<net>[_full|_short].py`. (Was the `LEAN_MLIR_FULL/SHORT` env flags, now retired.) Mirror this for R50.
- **Lossless suspend/resume**: `save_train_state`/`load_train_state` (params+opt_state+ema+step into
  one atomic, keep-last-3 `.npz`); `LEAN_MLIR_RESUME` restores it. Supervisor `supervise_vit_80ep.sh`
  uses it (ROCm default + `BACKEND=cuda`).

**Direct templates to copy from:**
- R50 architecture (bottleneck 3/4/6/3): `jax/MainResnet50.lean` (Imagenette).
- ImageNet trainer pattern (tfds, mesh, the `main`): `jax/MainResnetImagenet.lean` (R34, runningBN on).
- running-BN block threading to mirror for bottleneck: the `basic_block` running variant in
  `Codegen.lean` (search `def basic_block(params, x, idx, bn, bn_start, training)`), added in commit
  `8acb866`; the per-net wiring pattern (init/from-buf/to-buf/forward/bnChannels) in `b6be097`.
- smoke-test harness: `jax/scripts/smoke_mnv2_runningbn_gpu.py`, `smoke_vit_resume_gpu.py`.

---

## RSB-A2 target recipe (Wightman et al. 2021)

ResNet-50 (standard bottleneck), 224×224, 300 epochs. **LAMB** optimizer, lr 5e-3 @ batch 2048
(linear-scale to your batch), 5-epoch warmup + cosine, weight decay 0.02, **BCE loss** over
multi-hot targets (NO label smoothing — BCE subsumes it), **Mixup 0.1 + CutMix 1.0**, **RandAugment
m7 / mstd0.5**, **Repeated Augmentation (3×)**, **stochastic depth 0.05**, bf16, test crop ratio 0.95.

---

## Build plan (phased; each phase builds + smoke-tests + commits independently)

Recommended order below. **Repeated Aug (Phase 2) is pipeline-independent** — it can move to the
front if preferred, and it ALSO closes ViT's one remaining gap (faithful DeiT-Ti), so it
double-counts. The rest layer onto a working R50 host, so Phase 1 (skeleton) comes first.

### Phase 1 — working R50/ImageNet skeleton (mechanical, no new features)
Get a real R50 trainer running at ImageNet scale before adding RSB features.
1. **Bottleneck running-BN + stochastic-depth threading** in `Codegen.lean`:
   - `bottleneck_block` / `bottleneck_block_down` call `conv_bn` (3–4 calls) — add running variants
     exactly like `basic_block`: signature `(..., bn, bn_start, training)`, thread the `(out,new)`
     `conv_bn`, return `(x, [n0,n1,n2(,n3)])`. Add the `.bottleneckBlock` arm to `bnChannels`
     (3 BN per block; +1 shortcut BN on EACH stage's first block — R50's stage-1 also changes
     channels 64→256, so all 4 stages get a shortcut, unlike r34's 3. Count check: 1 stem + 16
     blocks·3 + 4 shortcuts = **53**; verify the emitted `_BN_CHANNELS` length).
   - Stochastic depth: thread `drop_key`/`keep_prob` into the bottleneck residual (mirror the
     mbconv/convnext inverted-drop), add the `.bottleneckBlock` arm to the forward SD-dispatch.
2. **`jax/MainResnet50Imagenet.lean`** = `MainResnet50.lean` bottleneck backbone + `.dense 2048 1000`
   head + `.imagenet` + a placeholder recipe (start with the R34 SGD config so it builds NOW) +
   `runningBN := true`. Output `generated_resnet50_imagenet.py`. Add `resnet50-imagenet` to
   `jax/lakefile.lean` (mirror `resnet34-imagenet`).
3. **Smoke test** (template below): 53 BN layers (verify count), trains a few steps, eval(running)
   ≠ eval(batch), grad flows. Commit.

### Phase 2 — Repeated Augmentation (also unblocks DeiT-Ti)
In the tfds pipeline (`build_imagenet_iter` in `Codegen.lean`): insert
`flat_map(lambda x: Dataset.from_tensors(x).repeat(K))` BEFORE `.map(_pp)` so each of the K copies
augments independently, + a shuffle after so copies spread across batches (approximates timm's
index-level RASampler — note in `log()`/doc that it's an approximation). New `TrainConfig` field
`repeatedAug : Nat := 1`. Keep steps_per_epoch fixed (epoch sees 1/K unique imgs ×K, per RSB).
**Smoke/throughput check**: confirm 3 copies are independently augmented AND measure step time —
3× CPU aug load can flip ViT-Ti-scale runs input-bound; R50 is heavier so likely fine, but verify.

### Phase 3 — LAMB optimizer
New `OptimizerKind.lamb` (`Types.lean:294`). Emit a `train_step` LAMB variant in `Codegen.lean`
(mirror the adam variant): per-param `m,v` (Adam moments) → update `r = m̂/(√v̂+ε)`, then the
layer-wise **trust ratio** `min(‖θ‖ / ‖r + wd·θ‖, clip)` scaling the step; opt_state `(m, v, t)`.
Decoupled weight decay. Smoke-test: trains, loss finite/decreasing, trust-ratio finite.

### Phase 4 — BCE loss
New `LossKind.bce` (`Types.lean:271`). In `loss_fn`, emit
`-mean(sum(tgt·logσ(z) + (1-tgt)·log(1-σ(z))))` over multi-hot targets (the mixup/cutmix soft-label
path already produces `[B,NC]` targets; BCE consumes them directly, no label smoothing). Smoke-test:
finite loss on a multi-hot batch, grad flows.

### Phase 5 — assemble + run + bestiary
RSB-A2 `resnet50ImagenetConfig` (LAMB, lr 5e-3 scaled, BCE, mixup0.1/cutmix1.0, RA m7/mstd0.5,
repeatedAug 3, dropPath 0.05, WD 0.02, EMA, 300ep) + the `full`/`short` recipe-arg selector (short =
~30–50ep validation). End-to-end smoke. Supervisor `scripts/supervise_r50_300ep*.sh` (mirror the
6-GPU convnet ones / the lossless-resume ViT one). Eval `scripts/eval_r50_full50k.py`. Then add the
spec to the **bestiary** (`tests/bestiary_timm_report.md` + the bestiary spec list — see how the
other nets are registered) so it's recorded even if the full run waits. Compute ≈ **60–65 hr**
(see `resnet50_imagenet.md`); LAMB @ large batch may shift per-step cost — measure first 400 steps.

---

## Smoke-test harness pattern (reuse for every phase)

The generated `.py` has an `if __name__ == "__main__":` guard, so import it as a module and
exercise `init_params`/`forward`/`train_step` on a synthetic batch — no tfds, no full run:

```python
# venv: /home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3
import importlib.util, jax, jax.numpy as jnp
from jax import random
spec = importlib.util.spec_from_file_location("g", ".lake/build/generated_resnet50_imagenet.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert jax.devices()[0].platform == "gpu"
p = m.init_params(random.PRNGKey(0)); bn = m.init_bn_state()
# ... build opt_state per optimizer, run a few m.train_step(...), assert finite/decreasing,
#     bn buffers move, eval(training=False) != eval(training=True). See smoke_mnv2_runningbn_gpu.py.
```
Generate the `.py` first: `cd jax && lake build resnet50-imagenet && timeout 8 ./.lake/build/bin/resnet50-imagenet >/dev/null 2>&1` (the exe writes the `.py` before spawning python; kill it).

---

## Environment / gotchas

- **venv** (jax 0.10.0): `/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3`. ROCm box =
  2× 7900 XTX (gfx1100); ares = 6× 4060 Ti (CUDA, mask idx 1,5 for PCIe AER).
- **2-GPU on ROCm needs `LD_PRELOAD=/opt/rocm/lib/librccl.so.1`** (TF's bundled NCCL shadows RCCL).
- bf16: `bf16Conv` is ~1.6× FASTER on CUDA/cuDNN but SLOWER on ROCm/MIOpen — set `bf16Conv := false`
  when running R50 on the 7900 XTX box. (R50 is conv-bound → CUDA box is the better home for it.)
- **Exe-cache gotcha**: after a `Codegen.lean` change, `lake build resnet50-imagenet` should rebuild
  the exe; if the generated `.py` looks stale, `rm .lake/build/bin/resnet50-imagenet` and rebuild.
- Adding a new `Layer`/`OptimizerKind`/`LossKind` constructor: the build surfaces every
  non-exhaustive match (Types/Spec/SpecHelpers/Codegen, maybe MlirCodegen) — fix each; JAX codegen
  forward/init often has a `_ =>` catch-all that silently no-ops, so verify the emit, don't trust
  a clean build. (This is how the `convNextStem` + running-BN rollouts went.)
- Smoke tests filter noisy stderr: `grep -viE "cuInit|cuda_platform|wmma|matrix core"`. The
  `wmma … bf16` lines on gfx1100 are benign.

## References
- `planning/resnet50_imagenet.md` — RSB-A2 vs SGD-fallback decision + compute estimate + supervisor/eval scaffolding notes.
- `planning/jax_imagenet_sweep.md` — gaps A–D status, per-net faithfulness, subrun selector.
- Commits `b6be097` (running-BN mnv2 + the per-net wiring template), `8acb866` (basic_block running
  variant — the exact thing to mirror for bottleneck), `72ee51e`/`eb9a0ed` (subrun selector).
