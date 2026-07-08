# MI300X rental program — the runs local hardware structurally can't do

**Strategy** (decided 2026-07-07): mars (7900 XTX, RDNA3) keeps the ROCm/JAX stack
validated for free — locks, generated scripts, the jax-0.10.0 mesh-hang fix. That sunk
cost makes underpriced AMD datacenter cloud usable: MI300X = 192 GB HBM3, CDNA3 bf16
matrix cores (~1.3 PF dense — the RDNA3 "bf16 is a wash" result does NOT apply), rentable
at ~$2/GPU/h because most stacks are CUDA-only. Local validation transfers the SOFTWARE,
not the perf profile (gfx1100 ≠ gfx942 MIOpen paths): **the first rented hour is always a
smoke test.**

Two workload classes benefit:
1. **Memory-bound**: single-forward big-batch (true-2048/A2/bs4096), high-res diffusion
   UNet attention, long-context GPT, 512px segmentation — no remat machinery needed,
   which keeps the generated code teaching-artifact readable.
2. **GEMM-bound**: ViT / transformer training — the isolated ViT-Ti matmuls already
   measured 2.7× bf16 on gfx1100; CDNA is that path's real home.

R50-style conv training is host-pipeline-bound (~4–6 k img/s numpy aug) on EVERY
accelerator we costed (MI300X, TPU v3-8/v5e-8) — faster silicon idles behind the feed.
Fixing the input pipeline is what would unlock paying for faster chips; until then the
estimates below assume the ~4–6 k cap.

## Provider intel (checked 2026-07-07)

| provider | $/GPU/h | verdict |
|---|---|---|
| **RunPod** | **$2.10** | **CHOSEN** — single MI300X, per-second billing, prior account |
| TensorWave | $1.71 | cheapest but sales-agent gated; revisit if program grows |
| Vultr | <$2 | backup marketplace; also has MI355X ($2.59) |
| DigitalOcean | $16/node | 8-GPU node only (=$2/GPU) — wrong shape |
| Azure / Oracle | $$$ | 8-GPU ND-series, enterprise quota — skip |
| Crusoe | ~$2–3 | MI355X (288 GB) fallback if A2 memory probe fails |
| **AMD Developer Cloud** | free credits | **APPLY** — TRC-analog; we're the ideal applicant (open source, ROCm-committed, upstreamed a jax-rocm bug, public "verified training on Instinct" artifact). Could cover the whole program. |

GCP has NO AMD GPUs (NVIDIA + TPU only; AMD = EPYC CPUs). TPU alternative if AMD falls
through: v5e-8 spot ~$2.80/h slice (~$20/run), TRC = free; v3-8 still exists
(us-central1-a/b, europe-west4-a) but is legacy.

### 2026-07-08 pivot: RunPod has NO AMD GPUs — program runs NVIDIA there

**Decision: A100 SXM 80 GB (~$1.39/h community) is the workhorse.** NVIDIA is the
repo's home turf (`jax/requirements-cuda-lock.txt` klawd-validated, bf16 conv 1.6×
measured) — the gfx942 smoke-test layer drops out entirely. RunPod NVIDIA menu:
4090 24 GB $0.34 / A100 80 GB $1.39 / H100 80 GB $2.89 (same VRAM, unfeedable GEMMs —
skip) / H200 141 GB $4.39 / B200 192 GB $5.89.

Per-run card assignments:
- run 0a probe + ViT-S + ViT-B: **A100 SXM** (probe on the workhorse so extrapolations
  are measured on the card the program uses).
- A3 `true-2048` (~80 GB): A100 is EXACTLY at the line — try it (per-second billing
  makes an OOM cost pennies); fallback H200 ~7 h ≈ $31.
- **A2 single-forward (~155–160 GB): no sane RunPod option** (H200 short, B200 ≈
  $140–175 ≈ 3× the MI300X price). The clean-BN A2 headliner RELOCATES to AMD
  elsewhere — Vultr MI300X <$2 or the AMD Dev Cloud application. RunPod fallback if
  impatient: `gradAccumSteps := 2` on A100 (~$35–42, Ghost-BN halved, asterisk kept).
- NVIDIA pod runbook deltas: pip line = `jax/requirements-cuda-lock.txt`; for
  `lake run benchmark` the CUDA default chip is sm_86 (klawd) — set `IREE_CHIP=sm_80`
  on A100 (`IREE_BACKEND` auto-detects cuda via nvidia-smi). The gfx942 items (§run-0,
  §run-0b item 5) apply only when the AMD pod happens.

## The run program (≈ $155–170 total @ $2.10/h)

Sequenced so each run de-risks the next. All bs2048 recipes exist in
`jax/MainResnet50Imagenet.lean` (`a2-true-2048` added 2026-07-07, uncommitted).

| # | run | recipe | est. h | est. $ | target / decision |
|---|---|---|---|---|---|
| 0a | **Imagenette profiling probe** | branch `runpod-probe` | ~1–2 | ~$3 | FIRST pod: RunPod pattern down, R50/ViT step-time+memory on gfx942; no ImageNet needed |
| 0 | smoke test + memory probes | — | ~1 | ~$2 | gate for the big runs (needs ImageNet volume) |
| 0b | exploratory smoke session | see below | ~2–3 | ~$5 | 8k-context GPT working; ViT-B memory/step-time; DDPM@128 probe |
| 1 | A3 @ true bs2048 | `true-2048` | ~7 | ~$15 | ≥78% ⇒ Ghost-BN was the −1.4%; recipe mechanics proven |
| 2 | **A2 @ true bs2048 (headliner)** | `a2-true-2048` | 24–30 | ~$50–63 | **79.8%** paper target |
| 3 | ViT-S/16 DeiT | (to wire) | ~20 | ~$42 | **79.8%** — the matched-scale CNN-vs-ViT TIE (22M/4.6GF vs R50 25.6M/4.1GF) |
| 4 | ViT-B/16 DeiT (+↑384 ft) | (to wire) | ~22 | ~$46 | **81.8%** (+83.1% after 30ep @384 ft) — the scaling headline |

Narrative payoff of 2+3+4: re-stage the 2021 DeiT-vs-RSB debate from one verified
trainer — equal recipe + equal scale = dead heat (79.8 = 79.8); the architecture win is
scaling headroom, not "transformer beats CNN". Blueprint chapter, not benchmark table.

## Run-0 runbook (the $2 hour)

Pod: RunPod ROCm 7.1 image (fine — the locked `jax-rocm7` wheels are FAT wheels bundling
their own hipBLASLt/MIOpen userspace; the image only supplies plumbing, the kernel driver
comes from RunPod's host regardless; mars validated on 7.2).

1. `pip install -r jax/requirements-rocm-lock.txt`
   (pins: jax/jaxlib 0.10.0, jax-rocm7-{pjrt,plugin} 0.9.1.post4)
2. `python -c "import jax; print(jax.devices())"` → expect one gfx942 device
3. `python jax/scripts/smoke_r50_a3_gpu.py` → finite, decreasing loss
4. **Memory probes** (the only genuinely open numbers):
   - fwd/bwd @ bs2048/160px — expect ~80 GB (true-2048 gate)
   - fwd/bwd @ bs2048/224px — expect ~155–160 GB (a2-true-2048 gate; if OOM ⇒ flip
     `gradAccumSteps := 2` (2×1024, Ghost-BN halved) or rent MI355X 288 GB)
   - CAVEAT: MIOpen auto-tunes conv kernels on first encounter per arch — first steps
     are slow and workspace-hungry; run ≥5 steps before reading the high-water mark.
5. Checkpoint/resume roundtrip on-pod: train 2 epochs of something tiny, kill, resume
   via `LEAN_MLIR_RESUME`, confirm loss continuity.

## Run-0a — Imagenette profiling probe (the FIRST pod; bootstraps the RunPod pattern)

**Goal**: get the whole rent→load→benchmark→report loop working with a **1.5 GB dataset
that downloads in minutes** — zero ImageNet-volume dependency. Single GPU. Also produces
the first real MI300X step-time/memory numbers for R50-A3-shaped and ViT-DeiT-shaped
work, which harden every estimate in the program table.

**Division of labor** (the pattern being established):
- **This machine / a coding session**: build + typecheck + push a branch. The pod never
  compiles Lean.
- **The pod (a Claude session on the box)**: clone the branch, run the runbook below,
  report numbers back.

### To build (coding session, branch e.g. `runpod-probe`)

The JAX path ALREADY has Imagenette support — `.imagenette` dataset kind in
`jax/Jax/Codegen.lean` (~241: loads pre-decoded `train.bin`/`val.bin` — no JPEG decode,
no tfds), `download_imagenette.sh` fetches + preprocesses. So the probe is a config +
profiling layer, minimal diff (teaching-artifact ethos):

1. **`imagenette-profile` recipes** — two configs on existing Mains or one new
   `MainProfileImagenette.lean`:
   - R50, A3-shaped: LAMB + BCE, 160px-style, bs512, cosine — but epochs ~5 (enough
     for steady state past MIOpen autotune, NOT a training run).
   - ViT-S/16, DeiT-shaped: AdamW, the DeiT aug pack, bs512, epochs ~5 — the "300-epoch
     approximation": profile the step, extrapolate the schedule.
   - CHECK: what resolution the imagenette preprocessor stores (`preprocess_imagenette.py`
     / Codegen ~243) — the profile res must match what the bins contain.
2. **Profile output** (the deliverable — print, don't plot):
   - steady-state ms/step (exclude warmup/autotune steps; report median of last N)
   - img/s, and `jax.local_devices()[0].memory_stats()` peak bytes
   - extrapolations: full A3 (100 ep) and DeiT (300 ep) wall-clock at BOTH imagenette
     scale and ImageNet scale (×1.28M/9.5k images), printed as "$@ $2.10/h"
3. **Commit the generated `.py` on the branch** (normally `.lake/build` is gitignored) —
   e.g. under `jax/probe/`. This is the key pattern decision: the pod needs NO Lean
   toolchain, just `git clone -b runpod-probe --depth 1` + pip + the download script.
4. Optional stretch: the 8k-context TinyStories regeneration (run-0b item 1) rides the
   same branch — `data/tinystories` upload is ~1 GB.

### Pod-side runbook (instructions for the Claude on the box)

```
git clone -b runpod-probe --depth 1 <repo-url> && cd lean4-jax
pip install -r jax/requirements-cuda-lock.txt        # NVIDIA pod (A100 SXM);
                                                     # rocm lock on an AMD pod
python -c "import jax; print(jax.devices())"        # expect CudaDevice / platform gpu
./download_imagenette.sh                             # ~1.5 GB, minutes
python jax/probe/<r50 probe>.py                      # R50 A3-shaped profile
python jax/probe/<vit probe>.py                      # ViT-S DeiT-shaped profile
```
Report back: device kind, steady-state ms/step + img/s + peak memory for each probe,
the extrapolation lines, and anything that crashed (full traceback). Watch the FIRST
steps being slow — XLA compile + conv autotune (cuDNN or MIOpen), not steady state. If
a probe OOMs, halve batch and note it. Everything is disposable: no checkpoints matter
on this pod.

NB the imagenette bins are pre-decoded arrays — this probe measures GPU compute clean
of the tfds/JPEG pipeline question. The tfds `pipeline_bench.py` needs the ImageNet
volume and belongs to run-0/0b, NOT this pod.

## Run-0b — exploratory smoke session (~2–3 h, ~$5; can share the run-0 pod)

Interactive session; goal is "does it work + what does it cost", not trained models.

1. **8k-context TinyStories** — the star. Mechanics verified 2026-07-07:
   - attention is NAIVE (`Codegen.lean` ~920: full L×L softmax) ⇒ genuinely
     memory-bound ⇒ the honest scaling demo. No flash-attention shortcut hiding in there.
   - data is a flat token stream sampled in chunks (`F32.sampleChunks` over `train.bin`)
     ⇒ `seqLen := 8192` just means longer chunks — NO packing work. Learned pos table
     8k×256 is trivial. Causal mask runs across story boundaries (standard packed-stream
     trade; TinyStories stories are ~200–300 tokens, so an 8k window spans dozens).
   - plan: regenerate at seqLen ∈ {1k, 2k, 4k, 8k}, batch sweep (8k likely lands at
     bs≈8–16 bf16 given ~L²·heads·layers activation residency), record max-fitting bs +
     step time + loss-goes-down at each L. Deliverable: the "context vs memory/step-time"
     table — the loss-vs-context CURVE is a later long run.
2. **ViT-B smoke** — wire the spec first (local spike item): bs512@224, ≥5 steps past
   MIOpen tuning, record memory + step time ⇒ hardens the ~$46 estimate for run 4;
   sanity-check bf16 GEMM throughput vs the 2.7× gfx1100 datapoint.
3. **DDPM@128px probe** (optional third): current UNet spec scaled to 128px with
   attention, few steps, memory + step time ⇒ sizes headliner #2 before designing it.
4. **Pipeline scaling test** — `jax/scripts/pipeline_bench.py` (built 2026-07-07):
   imports `build_imagenet_iter` from a generated trainer (jax pinned to CPU — no GPU
   touched, can run while something trains), drains batches, reports img/s + cpu-sat.
   `--sweep` re-execs across thread counts for the scaling curve; `--eval` isolates
   decode+center-crop from the aug/RA3 stack; `--gpu-step-ms <measured>` prints the
   input-bound verdict. On the pod: run the sweep at batch 512 AND 2048, once against
   train and once `--eval`. Outcomes: cpu-sat→1.0 with img/s short of the GPU ⇒
   genuinely CPU-bound ⇒ the pre-decoded-array fix is worth the evening; low cpu-sat +
   flat scaling ⇒ disk/volume-locality bound ⇒ fix is storage, not code. Either answer
   converts the ~4–6k img/s assumption in every estimate above into a measurement.
5. **`lake run benchmark` — the verified-IREE path on gfx942.** The book's
   training-time estimator (dense/conv/attn probes → per-chapter wall-clock scaled
   from the 7900 XTX reference table). Backend auto-detects rocm; set
   `IREE_CHIP=gfx942` (plumbs to `--iree-rocm-target`, `Types.lean` ~597). This is the
   bigger deal than it looks: **first validation of the PRIMARY Lean→StableHLO→IREE
   path on CDNA**, plus a datacenter column for `BENCHMARK.md` (MI300X vs 7900 XTX vs
   4060 Ti). Setup is the tax: elan + `lake build` of the probe trainers on the pod
   (Mathlib via `lake exe cache get`; pod CPUs are beefy) and **rebuild the FFI `.so`
   against the pip-installed IREE runtime per `IREE_BUILD.md` §4** — the stale-FFI
   lesson, never skip. MNIST/CIFAR probe data auto-downloads (tiny). Note: the rocm
   reduction-vector-distribution workaround (`Types.lean` ~613) applies to ALL rocm
   chips — fine for the smoke; trying the run WITHOUT it on gfx942 (CDNA may not have
   the gfx1100 bug) is a free experiment while there.

**Data logistics** (the real chore): ~150 GB ImageNet → RunPod network volume
(~$0.07/GB/mo ≈ $10/mo — create in a datacenter WITH MI300X capacity, keep for the whole
program, delete after). Upload from home is the slow step; do it once, every pod mounts
it. Community-tier preemption is fine: `.state.npz` suspend/resume (atomic, keeps newest
3, cosine-aware `LEAN_MLIR_START_STEP`) was built for klawd's thermal wall. Set
checkpoint interval ~5 epochs for the long runs.

## Local GPU spike — faithfulness/prep checklist (before renting)

- [ ] **NEXT CODING SESSION: build the run-0a probe branch** (see §Run-0a "To build") —
      recipes, profile output, committed generated `.py`, push `runpod-probe`.

- [ ] `a2-true-2048` render sanity: generated `_a2true2048.py` has lr 0.005, EPOCHS 300,
      `BATCH_SIZE = (2048 // n_devices) * n_devices` (✓ checked at generation
      2026-07-07); confirm WD_MASK applies on the LAMB branch (the rsb-faithful
      WD_MASK bug fix must cover this recipe too).
- [ ] Small-scale mechanics run of the a2 script locally. NB `BATCH_SIZE` auto-scales by
      device count — on 4×16 GB that's 512/GPU @224px = OOM. Need a batch-override env
      (or run on mars single-GPU with tiny synthetic steps) purely to exercise the code
      path: aug pack @224, EMA, wd-mask, checkpoint write.
- [ ] Checkpoint/resume roundtrip INCLUDING the EMA shadow (A2 evals from EMA — a resume
      that drops the shadow silently costs the headline).
- [ ] **Measure host pipeline throughput @224px** with `jax/scripts/pipeline_bench.py`
      (`--sweep`, then `--gpu-step-ms 457` for the known klawd A2 step) on klawd and/or
      mars. Every cost estimate above rides on the ~4–6 k img/s assumption. NB the RA3
      `flat_map` repeats ENCODED bytes (decode lives in `_pp`, after it) — so RA3 saves
      disk reads but NOT decode CPU; if the `--eval` delta shows decode dominates, a
      cheap generated-pipeline win is moving the ×3 repeat after decode. Mixup/cutmix
      are ON-DEVICE (jit) — not a host cost. If it measures ≪4k, budget the fix first.
- [ ] Wire ViT-S/B specs + DeiT-S/B hypers in `MainVitImagenet.lean` (two-line net
      change: `.patchEmbed 3 {384,768} 16 196` + `.transformerEncoder {384,768} {6,12}
      {1536,3072} 12`; lr 5e-4@bs512-scaled per DeiT, dropPath {0.1,0.1}, keep the
      existing DeiT-Ti aug pack). Render + typecheck locally.
- [ ] Decide bs4096 LR policy before anyone asks: linear (0.016) vs sqrt (~0.011) from
      8e-3@2048 — LAMB lore says sqrt. (Only matters if we do the batch-science run.)
- [ ] Upload ImageNet to the network volume (can start before any pod exists).
- [ ] Run-0b prep: parameterize `seqLen` in `demos/MainTinyStories.lean` (or just a
      one-line edit per length), regenerate + typecheck at 8192, and smoke a few steps
      at seqLen 1024 / small bs on mars so the only new variable on the pod is scale.
      TinyStories token stream + tokenizer upload is tiny (~1 GB) — trivial vs ImageNet.
- [ ] gfx942 cross-compile smoke on mars (free, no pod): `IREE_CHIP=gfx942` compile of
      one train-step `.vmfb` — catches "iree-compile rejects gfx942" locally; the
      artifact can't RUN on gfx1100, compile success is the whole test.

## Beyond the program — 192 GB exploration menu (rough order)

1. **DDPM @128–256px with attention** (extends the active `ddpm_v2` arc; headliner #2).
   Attention at 32×32/64×64 feature maps is the classic memory wall; diffusion's host
   input is trivial, so it's the workload that actually USES CDNA FLOPs. No remat needed.
2. **Long-context TinyStories** — `seqLen := 256` today; naive O(L²) attention at 192 GB
   reaches ~4–8k context at modest batch (16–32×). "Loss vs context" curve from a
   verified trainer. Pure GEMM.
3. **bs4096 A3 LAMB study** (~160 GB @160px — fits): the actual LAMB-paper regime that
   used to need a pod. Does A3 hold at 2× design batch? Feeds the faithfulness ledger.
4. **UNet-Pets @512px** — skip connections pin full-res activations; bs 64–128 untouchable
   locally; visibly better demo figures.
5. ViT-L/16: SKIP from-scratch IN-1k (needs the DeiT-III bag: LayerScale, 3-Aug, longer
   schedule — a recipe-porting project, not a rental decision).
6. MI355X (gfx950): only if (a) the @224 probe OOMs, or (b) we want the "locked stack on
   AMD's newest silicon: y/n" finding for ~$20. Check jax-rocm7 wheels ship gfx950
   kernels FIRST.

## Open questions

- AMD Developer Cloud application — submit; program cost could go to $0.
- Host pipeline fix (pre-decoded/pre-resized array cache + real prefetch): the single
  lever that converts every estimate above from pipeline-bound to compute-bound
  (v5e-8/MI300X would finish 300ep runs in ~½ the wall clock). Worth an evening; also
  what a TRC-scale grant would want. **2026-07-08 upgrade — ImageNet-in-RAM:** RunPod
  A100 hosts measured at 2× EPYC 7713 (255 threads) + **1 TB RAM**; full ImageNet as
  pre-decoded uint8 is ~193 GB @224 (~98 GB @160) — the whole dataset fits in host RAM,
  no JPEG/tf.data/network-FS in the hot path at all. Verify the pod's cgroup actually
  allows the allocation before building on it. Related pod facts for the runbook:
  `/workspace` = MooseFS network volume (slow small-file I/O — preprocess ground on it);
  local container disk (`/`) for hot data; RA-flat-map decode caveat moot under this fix.
- ~~Does the generated attention use a fused/flash path or naive O(L²)?~~ ANSWERED
  2026-07-07: naive (full L×L softmax, `Codegen.lean` ~920) — memory-bound as hoped;
  the long-context demo is honest. A flash-attention lowering would itself be a nice
  future codegen story (verified rewrite naive→flash?).
