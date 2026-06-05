# imagenet_sweep.md — bf16 ImageNet sweep on the 4060 Ti box (Lean→JAX)

## Purpose

This is a **decision tool for practitioners running these experiments**: a
quantified map of *what each training knob buys you* — in accuracy and in
wall-clock — on fixed, consumer-grade hardware. The goal is not "we trained
five nets"; it's "here is what turning knob X costs and returns, measured on
the same box, so you can choose where to spend compute instead of guessing."

The knobs quantified (or being quantified) here:

- **Precision** — bf16 vs fp32: speedup *and* the per-op reason (conv vs
  matmul; the depthwise wash). Does mixed precision pay off on consumer NVIDIA?
- **Epochs** — 80 vs 300 (and the 30→90 anchor): what does 1/4 the compute
  cost you in final accuracy? The short-tier runs are baselines, not throwaways.
- **Architecture at fixed budget** — R34 / MNv2 / ENet-B0 / ConvNeXt-T / ViT-Tiny
  on *identical* hardware + recipe: the clean params-vs-accuracy-vs-time
  comparison you rarely get because real-world runs differ in setup.
- **Recipe levers** — EMA, stochastic depth, heavy aug, grad-clip: each with
  measured complexity / leverage (see "Path to paper numbers"). Which knob is
  worth turning? (We even measured the EMA per-step tax: ~4–5%.)
- **Scaling out** — 4→6 consumer GPUs over PCIe: the failure mode (Gen4 AER)
  and the fix (Gen3), so multi-GPU-on-a-budget is documented, not folklore.

The per-run `RESULTS.md` files are the raw data; the synthesis layer (the
combined sweep chart in the TODO) is the "knob → cost/benefit" payoff. Frame
additions as *comparisons*, not just more run logs.

---

Per-architecture ImageNet-1k training on the CUDA box (**ares**, 4× RTX
4060 Ti, `CUDA_VISIBLE_DEVICES=0,2,3,4` — idx1/idx5 excluded, see
`reference_ares_pcie_aer`). All phase-2 Lean→JAX, bf16 mixed precision
(matmul + the heavy convs via `bf16Conv`; norm/softmax/GELU/SE-gate stay
fp32). Batch 256 (4×64) everywhere. Written 2026-06-02.

## TL;DR

- Five architectures characterized on the same 4-GPU bf16 setup. Two run
  to completion (R34, ViT-Tiny); three validated + throughput-measured but
  NOT yet trained to completion (MNv2, ENet-B0, ConvNeXt-T).
- Per-epoch cost measured from steady-state ms/step (incremental, JIT
  amortized out) over ~400 steps each — thermally safe sample, then stop
  (the box has no case fan; long unattended runs need the AER-watchdog
  supervisor + cooling care).
- bf16-vs-fp32 measured for R34 (1.60×) and ViT (1.48×); the conv vs
  matmul split is why they differ. See `reference_bf16_depthwise_4060ti`
  for the per-op microbench (the 3×3 depthwise is a bf16 wash; 1×1s and
  7×7 depthwise win; whole MBConv block ~2×).

## The sweep (measured throughput, batch 256, 4× 4060 Ti, bf16)

| Net            | Params | ms/step | min/epoch | 80ep   | 90ep   | Status |
|----------------|--------|---------|-----------|--------|--------|--------|
| ViT-Tiny       | 5.7M   | 176     | ~7.6      | ~11 hr | —      | **DONE** |
| EfficientNet-B0| 5.3M   | 102     | ~8.9      | ~12 hr | —      | **DONE** |
| MobileNetV2    | 3.5M   | 106     | ~9.1      | —      | ~14 hr | **DONE** |
| ResNet-34      | 21.8M  | 139     | ~11.9     | —      | ~18 hr | **DONE** |
| ConvNeXt-T     | 28.6M  | 185     | ~15.9     | ~21 hr | —      | validated |

(ms/step is per-step wall-clock at steady state; min/epoch = 5004 steps ×
ms/step + ~0.3–0.5 min val. ViT is 2502 steps/epoch at batch 512 — its
176 ms/step is already normalized into the 7.6 min/epoch figure.)

Wait — ViT note: the ViT run used **batch 512** (2502 steps/epoch), the
rest use **batch 256** (5004 steps/epoch). The min/epoch column is the
apples-to-apples number to chart; ms/step is not directly comparable
across the ViT row because of the batch difference.

## Completed runs (real accuracy)

| Net       | Epochs | Precision | Val top-1 | Val top-5 | Weights |
|-----------|--------|-----------|-----------|-----------|---------|
| ResNet-34   | 90     | bf16      | **72.02%**| **90.62%**| `/home/skoonce/r34_imagenet_bf16.bin` |
| ViT-Tiny    | 80     | bf16      | **65.64%**| **87.06%**| `/home/skoonce/vit_tiny_imagenet_bf16.bin` |
| MobileNetV2 | 90     | bf16      | **68.33%**| **88.17%**| `/home/skoonce/mnv2_imagenet_bf16.bin` |
| EfficientNet-B0 | 80 | bf16      | **72.31%**| **90.31%**| `/home/skoonce/enet_b0_imagenet_bf16.bin` |

(All full-50k canonical eval. Per-epoch curves + RESULTS.md in
`jax/runs/{r34,vit_tiny,mnv2,enet_b0}_imagenet_bf16_*/`; R34/ViT/MNv2 also blueprint §6.4/§10.5/§7.2.)

**ENet-B0 is the sweep accuracy leader** — 72.31%, edging R34 (72.02%) at ~¼ the
params (5.3M vs 21.8M). First full 6-GPU run (post-Gen3): 2 AER auto-resumes over
~10.5 hr (epochs 25, 74) — Gen3 makes 6-GPU AER much rarer (vs Gen4's death in
30 s) but NOT zero. Measured 4→6 GPU speedup ~1.17× (data-pipeline-bound).

**PCIe Gen3 fix validated (2026-06-04).** The MNv2 run spanned the BIOS
Gen4→Gen3 change: the Gen4 portion (ep 1–84) threw ~5 AER `BadTLP`
auto-resumes; the Gen3 portion (ep 85–90, ~1 hr sustained load) threw **0**.
Throughput cost ~108→~111 ms/step (**+~3%**, negligible — compute-bound, not
interconnect-bound). ⇒ Gen3 trades ~3%/step for stability and unlocks 6-GPU.

**6-GPU confirmed clean post-Gen3 (2026-06-04, ENet-B0 run).** First all-6-GPU
run since the fix: 0 AER past the threshold that killed every Gen4 6-GPU
attempt in ~20–30 s. Gen3 fixes the 6-GPU case, not just 4.

**…but the 4→6 GPU speedup is only ~1.17×, not the assumed 1.35×.** Measured
on ENet-B0: 4-GPU 107 ms/step (2393 img/s) → 6-GPU 90 ms/step (2800 img/s).
**Cause is NOT bandwidth** (gradient all-reduce is ~2 ms of a ~90 ms step, even
on Gen3) — it's **input starvation**: GPUs sit at ~65% util, waiting on the
tfds CPU-side decode+augment pipeline, which can't feed 6 cards. The light nets
hit this wall first (ENet ~90 ms/step is very data-hungry); smaller per-GPU
batch (64→42 at fixed 256) is a secondary factor. Prediction: **heavier nets
(ConvNeXt) should scale better to 6-GPU** (more compute/step → less starved).
Mitigations if wanted: more data-loader parallelism, caching, or larger
per-GPU batch. Revise 300-epoch 6-GPU estimates to ~1.17–1.25× (net-dependent),
not 1.35×.

## bf16 vs fp32 (measured, matched 4-GPU / same batch)

| Net       | fp32 ms/step | bf16 ms/step | speedup |
|-----------|--------------|--------------|---------|
| ResNet-34 | 223          | 139          | 1.60×   |
| ViT-Tiny  | 260          | 176          | 1.48×   |

ViT's smaller multiple: narrow matmuls (embed 192) + proportionally more
time in fp32 LayerNorm/softmax that bf16 leaves alone. (fp32 ms/step for
MNv2/ENet/ConvNeXt not separately measured — only the bf16 configs were
benchmarked for those three.)

## Per-net recipe notes

- **ResNet-34 / MobileNetV2 / EfficientNet-B0**: SGD+momentum 0.9, peak LR
  0.1, cosine + 5ep warmup, label smoothing 0.1, RRC+flip. wd: 1e-4 (R34),
  4e-5 (MNv2), 1e-5 (ENet — small, protects depthwise/SE). No mixup/cutmix.
- **ViT-Tiny / ConvNeXt-T**: AdamW + **grad-clip 1.0** (mandatory — the
  DeiT LR collapses to chance without it; same insurance on ConvNeXt). LR
  5e-4 (ViT, batch 512) / 4e-4 (ConvNeXt, batch 256, = 4e-3@4096 scaled),
  wd 0.05. ViT runs the full mixup/cutmix/RandAug/erasing suite; ConvNeXt
  validation tier leaves them off.

## Augmentation (what we use, and how it differs from the Imagenette examples)

What `augment := true` expands to depends on the data path:

- **ImageNet (`.imagenet`, tfds):** Inception-style **random-resized-crop**
  (`sample_distorted_bounding_box`: 8–100% of image area at a 3/4–4/3 aspect
  ratio, resized to 224×224) **+ random horizontal flip**, applied in the tfds
  map. Validation: resize + **center-crop** to 224 (no flip).
- **Imagenette (`.imagenette`, in-RAM `.bin`):** `load_imagenette` statically
  **center-crops** 256→224; then per training batch `augment_batch` does a
  **pad-14 random crop** (pad 224→252, crop back — i.e. ±14px translation
  jitter) **+ random horizontal flip**. No scale or aspect distortion.

So the convnet **augmentation delta, Imagenette → ImageNet, is crop style
only**: translation jitter (Imagenette) vs scale+aspect RRC (ImageNet, the
standard ImageNet aug, since object scale varies hugely there). Both flip;
both center-crop at val. (The bigger recipe deltas are elsewhere — optimizer,
LR, schedule, 1000-class head — not the augmentation.)

**Label smoothing 0.1** is in the loss for all five ImageNet recipes.

**Heavy-aug pack — deliberately off for the convnets.** Mixup, CutMix,
RandAugment, and Random Erasing are wired (config flags + a soft-label
train-step for the label-mixing ones) but used **only by ViT-Tiny on
ImageNet**: Mixup α0.8 + CutMix α1.0 (alternating per step), RandAugment M9
(color subset), Random Erasing p0.25 — the DeiT suite. MNv2 / ENet-B0 /
ConvNeXt-T run **base aug only** at the validation tier. ConvNeXt additionally
lacks **stochastic depth + EMA** (not yet wired on the JAX path).

These omissions are the main reason the convnet validation runs will land
under their paper numbers; the 300-epoch + heavy-aug (+ stochastic-depth/EMA
for ConvNeXt) push is the TODO that closes the gap.

## Trainers / build targets

| Net | Lean file | exe | supervisor |
|-----|-----------|-----|------------|
| R34 | `MainResnetImagenet.lean` | `resnet34-imagenet` | `supervise_r34_90ep.sh` |
| ViT | `MainVitImagenet.lean` | `vit-tiny-imagenet` | `supervise_vit_80ep.sh` |
| MNv2| `MainMobilenetV2Imagenet.lean` | `mobilenet-v2-imagenet` | `supervise_mnv2_30ep.sh` |
| ENet| `MainEfficientNetImagenet.lean` | `efficientnet-b0-imagenet` | `supervise_enet_b0_80ep.sh` |
| CNeXt| `MainConvNeXtImagenet.lean` | `convnext-tiny-imagenet` | `supervise_convnext_t_80ep.sh` |

Run pattern: `lake build <exe>` → emit the `.py` (run exe briefly, it
writes `.lake/build/generated_*.py` before spawning python, then dies on
the wrong python — harmless) → launch the supervisor (checkpoint/epoch +
AER-watchdog auto-resume). Canonical eval: `scripts/eval_*_full50k.py`.

## ⚠️ Open caveats

- **LR-warmup stability unverified** for MNv2 / ENet / ConvNeXt — the
  ~400-step samples were all still in early warmup (LR < 2e-3), loss flat
  near ln(1000)≈6.9. Whether each takes off cleanly (esp. SGD LR 0.1 on
  the BN-depthwise nets, and ConvNeXt AdamW) needs a real run. The configs
  note a fallback (drop peak LR to ~0.05) if early collapse shows.
- **Thermal**: no case fan — only run the full sweep with the supervisor
  (auto-resume) and an eye on temps; don't stack concurrent runs.
- **PCIe AER**: idx5 (bus 62) is the worst link, idx3 (bus 42) second.
  Swapping those two cables is the pending hardware fix; until then stay on
  `0,2,3,4` under the watchdog. (BIOS PCIe Gen3 is the fallback fix.)

## Path to paper numbers: what's wired vs. net-new (2026-06-02 scoping)

MNv2 needs nothing more — SGD + base aug is its standard recipe (~71% target).
EfficientNet-B0 and ConvNeXt-T need more to approach paper. The four levers:

| Lever | Status | Complexity | Leverage | Notes |
|-------|--------|-----------|----------|-------|
| Aug pipeline — color (Mixup/CutMix/RandAug-color/Erasing) | **wired** (config flags; ViT uses it) | S — flip flags | high | RandAug is **color-only** (brightness/contrast/etc.). |
| Aug pipeline — geometric RandAug (rotate/shear/translate) | not wired | M | low–med | `tfa` gone on tf2.21; reimplement via `tf.raw_ops.ImageProjectiveTransformV3` (the op tfa wrapped) — feed an 8-elem projective vector per op + magnitude map, add to the RandAug op pool, set fill_mode. The transform-matrix math is the only real work; sampling framework already exists. Runs in the tfds map. Last slice of the aug stack; color RandAug + Mixup/CutMix capture most of the gain. |
| RMSProp | not wired (have SGD+mom, Adam/AdamW) | M | low | EfficientNet-only; optimizer is emitted at ~3–4 parallel sites. SGD/AdamW reproductions hit ~75–76%, so optional. |
| EMA (weight averaging) | **wired** (ImageNet path, gated by `useEMA`) | done | high (~+0.5–1% ENet) | Jitted `ema_update`, decoupled from the 3 optimizers; eval + checkpoints use the EMA tree. v1 limits: ImageNet path only (in-RAM/Imagenette no-ops if set); resume resets live params to EMA (negligible late). |
| Stochastic depth (drop-path) | **wired** (ConvNeXt + ENet MBConv, gated by `dropPath`) | done | medium | Additive RNG in `forward` (default None → drop-free); per-block inverted drop, linear keep schedule over all blocks. ConvNeXt: every block. MBConv: drop guarded to skip-blocks (`ic==oc && stride==1`) inside `mbconv_block`. |

**Critical path done: EMA + stochastic depth both wired, for ConvNeXt *and*
ENet.** Both trainers are set up for 80ep with EMA + SD on (ConvNeXt dropPath
0.1, ENet 0.2). Remaining, both optional: geometric RandAug (low–med) and
RMSProp (ENet-only). ConvNeXt uses AdamW and needs neither.

**Common prerequisite — RNG threading in `forward`.** Today `forward(params, x)`
takes no RNG. Stochastic depth needs per-block drop masks, so the signature has
to become `forward(params, x, rng, training)`, threaded through every
residual-block helper and through `value_and_grad`/jit. This is the single
change that most affects the codegen's shape — scope it first, since SD depends
on it. (EMA does **not** need the rng — it's an optimizer-side shadow copy — so
EMA and the rng-threading can land independently.)

**ConvNeXt coverage.** ConvNeXt already has AdamW (✓) and LayerScale (✓, in the
ported arch). So once EMA + stochastic depth land and the aug flags are flipped,
ConvNeXt is **fully equipped** for a faithful 300ep run — RMSProp is irrelevant
to it. The only remaining ConvNeXt gap is cosmetic (convBn-stem-with-ReLU vs
conv+LN), not an accuracy lever.

## TODO

- [ ] **Actual accuracy results** for MobileNetV2 (90ep), EfficientNet-B0
      (80ep), ConvNeXt-T (80ep) — fill the completed-runs table with real
      val top-1/top-5 once each is trained to completion (full-50k eval).
- [ ] Per-epoch validation curves for the three pending nets (RESULTS.md +
      pgfplots, paralleling the R34/ViT `jax/runs/*/` treatment).
- [ ] **Sweep chart**: once the three pending results land, build a
      combined chart from this doc's numbers — e.g. params-vs-top1 scatter
      and/or min-per-epoch bar, native pgfplots (see the R34 §6.4 / ViT
      §10.5 curves for the style; `figures/log_to_pgfplots.py` may help).
- [ ] Verify MNv2/ENet/ConvNeXt LR stability past warmup (above caveat);
      record the working peak LR per net.
- [ ] fp32 ms/step for MNv2/ENet/ConvNeXt if a full bf16-vs-fp32 speedup
      table across all five is wanted.
- [ ] 300-epoch push for ENet/ConvNeXt to approach paper numbers — gated on
      EMA + stochastic depth (+ flipping the aug flags); see "Path to paper
      numbers" above for the per-lever scope and the RNG-in-`forward`
      prerequisite. Current configs are the 80ep validation tier.

## Run queue (decided 2026-06-04)

Agreed sequence. **Gen3 BIOS fix goes right after MNv2-90 finishes**, before
everything else — every later run wants the stability (no resume babysitting)
and the ~1.35× from 6 GPUs, and the 300-epoch runs (1–2.5 days each, unattended)
must not be exposed to host-reset risk. The two 80-epoch convnet runs stay in
as the **anchor baseline** (short-tier apples-to-apples datapoint) AND the
6-GPU+Gen3 stability shakedown before committing to the multi-day 300s.

1. [x] **MNv2-90** — DONE (4-GPU): 68.33% top-1 / 88.17% top-5 full-50k.
2. [x] **BIOS → PCIe Gen3** — DONE. AER eliminated; ~3% step-time cost. 6-GPU confirmed clean.
   - [x] 6-GPU multiplier measured (ENet): **~1.17×**, not 1.35× — input-pipeline-bound (see above).
   - [x] Clean 6-GPU run holds (0 AER past the Gen4 failure threshold).
3. [x] **ENet-B0 80ep** @ 6-GPU — DONE: 72.31% top-1 / 90.31% top-5. ~8min/epoch, 1.17x vs 4-GPU, 2 AER resumes.
4. [ ] **ConvNeXt-T 80ep** @ 6-GPU — anchor + shakedown. Heavier → should scale better to 6-GPU than ENet did.
5. [ ] **ViT-Tiny 300ep** @ 6-GPU — fastest of the 300s + riskiest recipe (LR-collapse history), front-loaded.
6. [ ] **ENet-B0 300ep** @ 6-GPU — full recipe (heavy aug on).
7. [ ] **ConvNeXt-T 300ep** @ 6-GPU — heaviest; Gen3 stability is a prerequisite.

**6-GPU 300ep estimates — REVISED to measured ~1.17× (light nets) / up to ~1.25× (heavy):**
soft until each net's own 6-GPU rate is measured. ViT ~31 hr, ENet ~37 hr (now
~10.7 hr/80ep × 300/80), ConvNeXt ~50–55 hr (expected to scale better, less
input-bound). **Confirmed risk:** at 6-GPU the tfds CPU data pipeline is the
bottleneck for light nets — ENet sat at ~65% GPU util (input-starved), giving
only 1.17× from +50% GPUs. Heavy aug (300ep recipes turn it on) is also CPU-side
and will compound this. Mitigation before the 300s if speed matters: more
data-loader workers / caching / larger per-GPU batch.
