# rsb_a2_resnet50.md — JAX trainer for RSB-A2 ResNet-50 (new-session handoff)

Build a **literal RSB-A2** ResNet-50 trainer in the Lean→JAX path, smoke-test each piece on
GPU, then (eventually) run it and add the net to the bestiary. RSB-A2 = the timm "ResNet
Strikes Back" 300-epoch recipe (Wightman et al. 2021) → **79.8% top-1** — the canonical modern
ResNet-50 baseline, and an honest one to have in the repo. Written 2026-06-22 as a fresh-session
plan; the decision + short gap summary also live in `planning/resnet50_imagenet.md` (RSB-A2 section).

Goal even if we never burn the ~60–65 hr run: a clean, smoke-tested RSB-A2 trainer + spec in the
bestiary. The run is a bonus.

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
- **Subrun selector**: each imagenet trainer's `main` picks validation vs full via env
  (`LEAN_MLIR_FULL=1` for short-default nets, `LEAN_MLIR_SHORT=1` for paper-default nets), writing
  distinct `generated_<net>[_full|_short].py`. Mirror this for R50.
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
repeatedAug 3, dropPath 0.05, WD 0.02, EMA, 300ep) + the `LEAN_MLIR_FULL`/`SHORT` selector (short =
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
