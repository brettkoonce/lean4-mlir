# Measurements removed from code comments (2026-09-30)

The comment-numbers rule (`planning/rubric_review.md`, decision 6; enforced by
`scripts/gates/comment_numbers.py`) took measured results, timings, memory sizes, speedups and
metrics out of code comments, because nothing re-checks them there. Where a run directory,
planning document or generator holds a number, the comment now points at it. This file keeps
the rest, so nothing is lost.

Each line below is verbatim as it stood at `bee5ab0d`, with that commit's line number, grouped by
the file it was in. None of it was re-measured, and much of it describes code, hardware or data
that has since changed. Read it as history, not as a current result. A line is listed whenever
its text no longer appears in the file, so some entries were reworded rather than removed.

`scripts/gates/comment_numbers.py` owns the patterns; this file was generated from `git diff
bee5ab0d` against them.

## `Bestiary/Inception.lean`

```
   61  is the same; the total param count lands within ~20% of the paper's
```

## `Bestiary/ResNet.lean`

```
   43  Lean → StableHLO → JAX pipeline to **76.66 % top-1 / 93.03 % top-5** on
```

## `Bestiary/ShuffleNet.lean`

```
   57  | ShuffleNet 0.5× (g=3) | 120 / 240 / 480          | ~1.0M  | 43.2%     |
   58  | ShuffleNet 1.0× (g=3) | 240 / 480 / 960          | ~2.4M  | 32.6%     |
   59  | ShuffleNet 1.5× (g=3) | 360 / 720 / 1440         | ~3.4M  | 31.3%     |
   60  | ShuffleNet 2.0× (g=3) | 480 / 960 / 1920         | ~5.4M  | 29.1%     |
```

## `Bestiary/WRN.lean`

```
   11  widening) on CIFAR-10/100 at half the training time and a quarter
   75  to 32-64-128 channels. ~2.2M params, 95.4% on CIFAR-10. -/
```

## `LeanMlir.lean`

```
  195  lake build ProofsMinimal    # the linear on-ramp above, ~1 min
```

## `LeanMlir/F32Array.lean`

```
  120  batch 32, against a ~310 ms step — so the C round trip buys nothing measurable, and keeping the
  183  -- (256 × 1280) it measured 150.07 ms of a 281 ms step (32×1280: 18.74 ms; `dropScales` at
  184  -- 9 sites × 256: 1.83 ms; `F32.const` on the same 327,680 floats: 0.073 ms). A host-side
  197  -- `lean_f32_dropout_fill`, which is where the 150 ms above went.
```

## `LeanMlir/IreeRuntime.lean`

```
  138  pushed 79-123 times per epoch. Measured on the MNIST MLP, **73% of an eval
  139  step was the parameter push** (0.6 ms of 0.8 — compute is 0.1).
```

## `LeanMlir/MlirCodegen.lean`

```
 4795  -- VisDrone is ~44% car / ~21% pedestrian, and the unweighted head collapses
```

## `LeanMlir/Proofs/Architectures/ChannelLN.lean`

```
   42  -- transposes measure free (Δ 0.00 ms on 16.1 ms of whole-net LN).
```

## `LeanMlir/Proofs/Architectures/Depthwise.lean`

```
   13  ~10× cheaper because you avoid the `O(ic · oc)` cross-channel sum.
```

## `LeanMlir/Proofs/Certificates/IbpConvScorecard/Basic.lean`

```
   58  79/100, 73/100, 47/100, 13/100 at
```

## `LeanMlir/Proofs/Certificates/IbpConvScorecard/Net.lean`

```
   30  the fixed first 100 test images: 79/100, 73/100,
   31  47/100, 13/100 at ε = 1, 2, 4, 8/255;
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/Instance.lean`

```
  129  -- 49→8→10 bias-free ReLU MLP trained on pooled MNIST (test acc ≈0.898);
  130  -- weights rounded to /128 rationals (quantized test acc ≈0.898). Test image
  229  --   212→88.9 and the certified radius grows 0.0463→0.1106 (2.4×).
  232  --   bound provably sits within 24%/26% of the per-layer optimum.
  292  (2.4× the Frobenius radius) leaves the prediction fixed. -/
  308  `[7.452, 9.2]` — the Gram bound is provably ≤ 1.235× optimal. -/
  381  certified lower bounds ℓ₁·ℓ₂ = 57.38, provably within 11.2% of the
  392  /-- **Schatten-8 trained certificate**: radius ≈ 0.1541 (3.3× Frobenius,
  393  1.4× Schatten-4; the true-σ ceiling for the product method is 0.171). -/
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/Scorecard.lean`

```
   17  * **unconstrained** — the committed /128 net (`W1t`/`W2t`, q-acc 0.898),
   19  **1/100 certified** at ε (measured, see below);
   22  /256-rationalized (`W1s`/`W2s`, q-acc 0.870), Schatten-8 product
   23  L = 19.76: **34/100 certified** at the same ε (measured).
   26  bites (the σ ≤ 4 cap keeps 87.0% test accuracy vs 89.8% unconstrained).
   44  (100 steps, 4 restarts) leaves uncon 69/100, capped 72/100
  315  -- § Per-image certificates, capped net (8/100 at ε = 1/10)
  543  -- § Per-image certificates, unconstrained net (1/100 at the same ε)
  646  `unconCerts_certified`). The dataset counts (34/100 capped,
  647  1/100 unconstrained) are exact-rational measurements recorded in the
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardCrown.lean`

```
 3019  /-- **The CROWN-IBP L∞ scorecard, spectrally-capped σ≤2 net (`mlpSF`)** — MEASURED 93/100 @ 1/255, 93/100 @ 2/255, 92/100 @ 4/255, 81/100 @ 8/255 (IBP box: 92/88/69/24; PGD-L∞ bracket 93/93/92/88). -/
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardCrownUncon.lean`

```
 3684  /-- **The CROWN-IBP L∞ scorecard, unconstrained net (`mlpTF`)** — MEASURED 94/100 @ 1/255, 92/100 @ 2/255, 76/100 @ 4/255, 15/100 @ 8/255 (IBP box: 87/42/2/0; PGD-L∞ bracket 95/92/85/36). -/
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardFull.lean`

```
   10  * **spectrally capped** (σ ≤ 2 projected SGD, q-acc 0.924, Schatten-8
   11  L = 4.95): **92/100 certified at ε = 0.1** (L2-PGD leaves 93/100
   13  **72/100 at ε = 0.3** (PGD: 92/100);
   14  * **unconstrained** (q-acc 0.951, L = 29.85): 76/100 at ε = 0.1
   15  (PGD: 94/100), collapsing to **2/100 at ε = 0.3** (PGD: 86/100) —
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardFullNets.lean`

```
    7  q-acc 0.924) and unconstrained (`TF`: 12 epochs, q-acc 0.951)
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardIBP.lean`

```
    9  **92/100**, **88/100**, **69/100**, **24/100** predictions robust (PGD-L∞ bracket: 93, 93, 92, 88).
 1674  /-- **The IBP L∞ scorecard, spectrally-capped σ≤2 net (`mlpSF`)** — MEASURED 92/100 @ 1/255, 88/100 @ 2/255, 69/100 @ 4/255, 24/100 @ 8/255 (PGD-L∞ bracket 93/93/92/88). -/
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardIBPUncon.lean`

```
    9  **87/100**, **42/100**, **2/100**, **0/100** predictions robust (PGD-L∞ bracket: 95, 92, 85, 36).
 1502  /-- **The IBP L∞ scorecard, unconstrained net (`mlpTF`)** — MEASURED 87/100 @ 1/255, 42/100 @ 2/255, 2/100 @ 4/255, 0/100 @ 8/255 (PGD-L∞ bracket 95/92/85/36). -/
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardSDP.lean`

```
   14  count from **34/100 to 69/100** — no retraining, no new data, just a
   21  72/100 — the cert ≤ TRUE ≤ PGD sandwich is nearly closed.
 1480  -- § Per-image certificates (8/100 at ε = 1/10)
 2081  /-- **The LipSDP scorecard** — MEASURED 69/100, vs 34/100 under the
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardSDPFull.lean`

```
    9  per-pair LipSDP constants lift the counts from **92→93/100 @ ε=0.1**
   10  (PGD bracket 93) and **72→91/100 @ ε=0.3** (PGD 92) — no
 2935  92→93/100 @ ε=0.1 (PGD 93) and 72→91/100 @ ε=0.3
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardSDPFullUncon.lean`

```
    7  per-pair LipSDP constants lift the counts from **76→91/100 @ ε=0.1**
    8  (PGD bracket 94) and **2→77/100 @ ε=0.3** (PGD 86) — no
 3071  76→91/100 @ ε=0.1 (PGD 94) and 2→77/100 @ ε=0.3
```

## `LeanMlir/Proofs/Certificates/LipschitzCert/ScorecardSDPUncon.lean`

```
   14  count from **1/100 to 63/100** — no retraining, no new data, just a
   21  69/100 — the cert ≤ TRUE ≤ PGD sandwich is nearly closed.
 1465  -- § Per-image certificates (8/100 at ε = 1/10)
 2089  /-- **The LipSDP scorecard** — MEASURED 63/100, vs 1/100 under the
```

## `LeanMlir/Proofs/Certificates/Smoothing/CPScorecard.lean`

```
   23  -- ── MNIST-MLP: 99/100 with a kernel-checked tail bound (σ=0.5, of which 99 correctly classified) ──
  648  -- ── MNIST-CNN: 100/100 with a kernel-checked tail bound (σ=0.5, of which 100 correctly classified) ──
 1279  -- ── CIFAR-CNN: 80/100 with a kernel-checked tail bound (σ=0.5, of which 60 correctly classified) ──
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecChunk1.lean`

```
    7  own module because (a) one whole-grid kernel evaluation retains ~15 GB of
    8  kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT
   10  processes cap the worst chunk at ~5 GB. -/
   17  out in pending-mvar instance synthesis (>10 min). Untrusted input —
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecChunk2.lean`

```
    7  own module because (a) one whole-grid kernel evaluation retains ~15 GB of
    8  kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT
   10  processes cap the worst chunk at ~5 GB. -/
   17  out in pending-mvar instance synthesis (>10 min). Untrusted input —
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecChunk3.lean`

```
    7  own module because (a) one whole-grid kernel evaluation retains ~15 GB of
    8  kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT
   10  processes cap the worst chunk at ~5 GB. -/
   17  out in pending-mvar instance synthesis (>10 min). Untrusted input —
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecChunk4.lean`

```
    7  own module because (a) one whole-grid kernel evaluation retains ~15 GB of
    8  kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT
   10  processes cap the worst chunk at ~5 GB. -/
   17  out in pending-mvar instance synthesis (>10 min). Untrusted input —
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecChunk5.lean`

```
    7  own module because (a) one whole-grid kernel evaluation retains ~15 GB of
    8  kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT
   10  processes cap the worst chunk at ~5 GB. -/
   17  out in pending-mvar instance synthesis (>10 min). Untrusted input —
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecChunk6.lean`

```
    7  own module because (a) one whole-grid kernel evaluation retains ~15 GB of
    8  kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT
   10  processes cap the worst chunk at ~5 GB. -/
   17  out in pending-mvar instance synthesis (>10 min). Untrusted input —
```

## `LeanMlir/Proofs/Certificates/Smoothing/DecScorecard.lean`

```
   14  ~2 min total); each per-image check is then a single O(index) list lookup —
   17  stays ~2 GB where full-literal lookups accumulated 11.3 GB — kernel memory is
   21  retains ~15 GB of kernel-cache bignums and OOMs 16 GB CI runners, and memory
   23  processes cap the worst chunk at ~5 GB. Per image, m is the LARGEST grid index with
```

## `LeanMlir/Proofs/Certificates/Smoothing/PhiBounds.lean`

```
   31  whole-grid evaluation peaks at 15 GB of retained kernel-cache bignums (an
   32  OOM on 16 GB CI runners); per-declaration chunks are freed in between.
  295  evaluation — 15 GB peak at 3300 panels, an OOM on 16 GB CI runners — be
```

## `LeanMlir/Proofs/Codegen/CnnArtifacts.lean`

```
  116  -- NO SPEEDUP, by design — bf16 measures 0.87× across cifar8's conv stack. These
  228  --   bn_sgd   gradient norm-rel 3.8e-5 vs the control's 1.9e-5 (2.0×), spread 11/38 ⊂ the control's 12
  263  -- comparison behind *"head width barely matters — 7.1× the params, accuracy within a point; the
```

## `LeanMlir/Proofs/Codegen/ConvNeXtRender.lean`

```
  254  **free** (Δ 0.00 ms on 16.1 ms of whole-net LN — XLA folds a transpose into the consumer's layout).
 1077  -- emitted under the `convnext` slug would collide with the artifacts the 84.41% Imagenette run,
```

## `LeanMlir/Proofs/Codegen/ConvNeXtRenderB.lean`

```
  762  -- This carve-out is NOT a fixed tax: it cost MobileNetV2 nothing (1.92× verified
  763  -- against a 1.94× JAX reference) and EfficientNet-B0 almost everything (1.09×). Which one ConvNeXt
  766  -- SINGLE-DEVICE IS THE ARM THAT MEANS ANYTHING. MobileNetV2 measures 1.92× on one GPU
  767  -- and 1.37× on four from the SAME GRAPH — the loss is the shim feed first and the f32 all-reduce
  845  -- green — ViT's stem wgrad runs 0.19× — so the render existing says nothing about the wall clock.
  921  -- render peaks at **9.53 GiB of the plugin's 11.68 default**, so B needs no accumulation render
  999  -- holding 12.8% of the random init at epoch 66, scoring **0.00% top-1** while the live weights
 1000  -- scored 70.48%. An 80-epoch Imagenette run is 2.4 τ, i.e. inside that regime.
 1121  -- ConvNeXt's reference number IS the EMA shadow's — **75.93%**, against a live best of 76.28% — so
```

## `LeanMlir/Proofs/Codegen/EfficientNetRender/Basic.lean`

```
 1039  -- Evidence for the tuned lr: runs/efficientnet_verified_crop_gpu1.log, val accuracy 40.63% after
 1040  -- epoch 1, 87.65% after epoch 80, peak 87.81% at epoch 79.
 1388  -- Imagenette trains to a known 87.58% in 80 epochs (`RESULTS.md`), with a per-epoch trajectory to
 1435  -- 10-class pair that the 88.20% Imagenette run, the prefix audit and `fwd-tie efficientnet
 1457  -- The EfficientNet reference's **72.31%** is an RMSProp number. ρ = μ = 0.9,
 1475  -- its 72.31% is the EMA shadow's number.
 1503  -- feed + f32 all-reduce), not a statement about the emit — MobileNetV2 is 1.92× on one GPU and
 1504  -- 1.37× on four, same graph. A 1-GPU pair is the measurement that isolates the renderer.
 1534  -- number is the single-device 1.09× (1.10× on the bare device), and that is the one that says what
 1687  -- `convnextin_adamdpwxclipdrop`, and what the 72.31% reference pair needs to be reachable through
 1724  -- With flat-activation NHWC↔NCHW relayouts in the graph B0's bf16 ratio is 1.10×, because that
```

## `LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean`

```
 1304  -- cannot: **how much of the bf16 win does the f32 all-reduce eat?** `adamdp64bf16` measured 1.37×
 1305  -- at 4 replicas while MNv4 — which renders no DP variant at all — measured 1.88× at 1. Those two
 1326  -- The depthwise convs are ~13% of MNv2's step and bf16 is a mild LOSS on them in isolation
 1327  -- (0.86× at MNv2's own layers — cuDNN has a better f32 depthwise kernel on Ada). The win comes
 1345  -- RMSProp is the ONLY gap between this net and the JAX reference's **68.33%** (everything else —
 1438  -- is what stops these overwriting the 10-class pair the 86.73% Imagenette run and the prefix check
```

## `LeanMlir/Proofs/Codegen/MobileNetV4RenderB.lean`

```
 1229  -- Target: `historical/RESULTS.md`'s 84.58%, the baseline path's number for this block table.
```

## `LeanMlir/Proofs/Codegen/RenderKit.lean`

```
  165  -- effect concentrates at low LR, i.e. in the cosine endgame. The first RSB-A3 R50 run (77.43%)
```

## `LeanMlir/Proofs/Codegen/ResNet34RenderB.lean`

```
 1439  -- the numeric tie, as a 0.28% loss disagreement against an otherwise bit-identical forward.
 1524  --   * the step bench (`resnet34-adam-bench`) — no cost, despite 1.68× the emitted ops, because
 1636  -- `cfg.batchSize := 256`. Batch is worth ~1.8× img/s on this net and bs256 fits on a 7900 XTX;
 1654  -- Why this batch: bs256 measures **1.78× img/s** over bs32 single-device, most of it
 1655  -- amortising the ~272 MB `[θ|m|v]` host↔device round trip over 8× the images — and that transfer is
```

## `LeanMlir/Proofs/Codegen/ResNet50RenderB.lean`

```
  621  -- suffix beside this `Bool` would be two writers for one fact on the artifact the 77.43% run
  648  -- `wdExcludeNormBias := true`; the live artifact has zero `%wdz`. So 77.43% was reached while
 1242  -- 1.41× is a comparable number rather than a differently-configured one.
 1322  -- BCE-with-logits and a 160/224 resolution split; LAMB at bs512 gives 40.8% against 78.1%, so the
 1379  -- gives **40.8% against 78.1%**, so the batch is not a detail). At `q = 5`, i.e. A3's 160² train
 1401  -- so the 77.43% run decayed all 161 parameters — BN γ, BN β and every bias included — at wd = 0.02.
 1404  -- renders are new artifacts beside the old ones rather than in place of them: the 77.43% result
 1434  -- delta on: the 77.43% result belongs to the graph that produced it, and re-pointing a slug at a
 1452  -- It does NOT restate the committed run: 77.91% is fp32 and stays fp32. This prices the tier's
 1453  -- other precision, and the pricing is what it is for — 191.6 → 121.6 ms/step on four cards,
 1545  -- the artifact the 77.43% run trained on.
 1553  -- This is the name whose graph produced 77.43%, and it must not move.
 1630  -- 79.8%-target run. Quote them the way A3's deltas are quoted, never as "RSB-A2 reproduced".
 1736  -- default of `memory_fraction = 0.75`, which on a 16 GB 4060 Ti reserves **11.68 GiB** and leaves
 1737  -- ~4.3 GiB of the card unreachable. A "% of budget" figure taken against that number is against
 1741  -- XLA's own log line is the measurement: *"XLA backend allocating 15.11GiB … for BFCAllocator"*
 1747  --     8×64   fp32                   6.18 G      53 %                 41 %
 1748  --     8×64   fp32 + EMA + sd        6.42 G      55 %                 42 %
 1749  --     8×64   bf16                   4.32 G      37 %                 29 %
 1750  --     4×128  fp32                  11.33 G      97 %                 75 %
 1751  --     4×128  fp32 + EMA + sd       11.90 G     over                79 %
 1752  --     4×128  bf16 + EMA + sd        8.09 G      69 %                 54 %
```

## `LeanMlir/Proofs/Codegen/StableHLO/Basic.lean`

```
   95  -- **f8-TYPED result**, convert back — an f32 result is 1.17× where the f8 result is 3.43×, the
  213  -- for CORRECTNESS but **not for SPEED**: on ViT's own MLP chain the f32-result shape is 1.18×
  214  -- over f32 and the bf16-result shape is **1.60×**. The f32 result makes the gemm write twice the
  283  -- bf16 peer of `denseRow` — the six per-block matmuls (Q/K/V/O/fc1/fc2) that are 90 % of a
  337  -- (an f32 result makes the gemm write twice the bytes, worth ~1.2× on a real chain). ViT's dot ops take the **bf16-typed
  970  -- The **bf16** input-VJP peers. These are where the money is: the backward is ~60% of the conv
 1690  -- `dot_general` a bf16-TYPED result (worth 1.18× → 1.60×), so the hardware rounds the output
```

## `LeanMlir/Proofs/Codegen/StableHLO/Pretty.lean`

```
   59  reshape pair) but is exactly the relayout the bracket exists to remove. Measured: 2.434 GB of
   60  transposes and 84.45 ms/step keyed by width, **0.122 GB and 68.28 ms** keyed by name.
   70  after it on the residual chain goes with it — 0.223 GB of relayout against 0.122 (measured,
 1023  -- WHY THIS EXISTS, and why it is worth 2.3× on EfficientNet-B0.
 1031  -- Measured on one 4060 Ti, node-granularity nsys: B0 bf16 @64 spent **72.61 ms/step,
 1032  -- 54.6% of all GPU time**, in those relayouts (10.486 GB) against its JAX reference's 0.75 ms.
 1033  -- ConvNeXt-T 53.2%. MobileNetV2 and ViT, which end up with no relayouts, are FASTER than their
 3357  -- the bytes and take a worse epilogue. Measured on ViT's own MLP chain: f32-result 1.18×,
 3358  -- bf16-result **1.60×**.
```

## `LeanMlir/Proofs/Codegen/ViTRenderB.lean`

```
  336  -- And the backward is where the money is — it measures at ~60 % of a conv step and the
  337  -- forward-only arm at 1.09×. A render that flipped only `vBlockFwdB` would look wired and buy
  482  -- **0.52×** — nearly twice as SLOW as f32 — and an `nsys` profile says why in one line: the
  484  -- `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_*` at ~30 ms,
  485  -- where the f32 arm's same op lowers to `conv2d_grouped_direct_kernel<float>` at ~5.8 ms.
  640  -- **THE ISOLATED-MATMUL MEASUREMENT PREDICTS 1.03×.** ViT-Tiny's own 387 matmuls, timed three ways
  641  -- at B = 32: f32 26.9 ms, bf16 in THIS emit shape 26.2 ms (1.03×), bf16 with activations staying
  642  -- bf16 BETWEEN ops 15.7 ms (1.71×). ViT's matmuls are skinny (contracting dim 192, or 768 at the
  644  -- about what the tensor cores save. That is NOT true of the convnets — ConvNeXt keeps 64 % of its
  668  --     (bf16Conv := false) (bf16ConvW := false)   24.39 ms   1.23x   ← what is rendered above
  669  --     (bf16Conv := true)  (bf16ConvW := false)   24.64 ms   1.21x   ← stem forward: ~free, no gain
  670  --     (bf16Conv := false) (bf16ConvW := true)    57.22 ms   0.52x   ← THE WEIGHT GRADIENT ALONE
  671  --     (bf16Conv := true)  (bf16ConvW := true)    57.29 ms   0.52x
  672  --                              f32 control       29.89 ms
  675  -- within noise either way; the weight gradient is +32.8 ms on a 29.89 ms step.
  708  -- RENDERER number is the single-device bare-device **1.46×**, and that is what the emit is
  712  -- gradient is 0.19× its f32 peer and the replica axis does not change that.
  797  -- this net's stem weight gradient measures **0.19×** its f32 peer — a 209×209 window
  831  -- four times too small, and it is **slower** per epoch (322 h against 270 fp32, 228 against 155
  842  -- **11.68 GiB**, which is the CUDA plugin's BFC `memory_fraction = 0.75` DEFAULT and not the
  844  -- **15.11 GiB** (XLA's own log line).
  848  -- number produced at global 128 is not comparable to DeiT-B's 81.8% even in principle. 128×4 is
  857  --   | `adamdp128x4wxclipdrop`   | 13.99 G |   93 %   |   1298.92      | `RESOURCE_EXHAUSTED` |
  858  --   | `…dropbf16`               | 12.61 G |   83 %   |    746.25      | ✅ runs (10.88 G there) |
  862  -- allocate 11.96GiB"*. So an accumulation loop is not a ViT-B fact; the limit is an unset
  865  -- **And the un-rematerialised graph really is 20.39 GiB** — XLA says so in its own words when
  866  -- compiled against the default budget (*"Can't reduce memory use below 10.16GiB … down from
  867  -- 20.39GiB originally"*). 13.99 is what rematerialisation buys, which is why a peak must be quoted
  892  -- region (86.6 M floats, ~0.35 GB per replica) on top of the bf16 twin's 12.61 GiB peak at
 1013  -- broken kernel deterministically. The variable drops that solver family, at ~7% throughput.
 1046  -- room (3.2 GiB peak per device at bs128×4, 20% of the card).
 1057  -- global 512 is **18 steps/epoch** — 1,480 updates over 80 epochs against the 23,600 the 71.31%
 1073  -- forward that the 71.31% run, the prefix audit and every `fwd-tie vit` invocation depend on.
 1198  -- `vit_adam_train_step.mlir` (which the 71.31% 80-epoch run, `vit-adam-tie` and `vit-dp-check` all
 1205  -- it shows a shadow still holding 12.8% of the random init at 3.1 τ, scoring **0.00% top-1**
 1206  -- while the live weights scored 70.48%, and an 80-epoch Imagenette run is 23,600
 1209  -- Read this net's smoke as a DELTA, never an absolute. ViT's 80-epoch Imagenette result (71.31%)
```

## `LeanMlir/Proofs/Float/MlpFloatBridge.lean`

```
  135  97.8% run — He init already exceeds the prettier `1/32` in its tails).
  548  real 12-epoch run is ≤ 1.6·10⁻⁵, 600× inside the `1/100` hypothesis. -/
  641  92.89% of the MNIST test set (`scripts/demos/mnist_e4m3_demo.py`). fp32 ≈ exact-ℝ
```

## `LeanMlir/Proofs/Foundation/ListDot.lean`

```
   14  ~15 ms/element `Neg`-elaboration tax);
```

## `LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTie.lean`

```
   61  --   step and the kernel re-derives the whole chain by unfolding (17 s / 6 GB on Lean 4.32.2,
   62  --   6 min / 48 GB on 4.34.0). The `rw`s hand it syntactic rewrites: 3 s / 3 GB on 4.34.0.
```

## `LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean`

```
   53  --   this module took ~18 min on Lean 4.32.2 and did not check on 4.34.0 (kernel timeout). As
```

## `LeanMlir/Proofs/Nets/MobileNet/MobileNetV4FullB.lean`

```
  824  -- this same chain elaborates for ~9 minutes and then dies in the KERNEL with a deterministic
```

## `LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean`

```
   60  --   cost ~45 s and 8 GB under `maxRecDepth 800000`.
```

## `LeanMlir/Proofs/Nets/ResNet/ResNet50FullB.lean`

```
  271  -- And `q = 5` IS the 160-px net -- `resnet50in160_*`, where the quoted 76.66% comes from.
```

## `LeanMlir/Proofs/Nets/Small/MlpCanonical.lean`

```
   27  vacuous — L ≈ 39 ⇒ 0% certified — which is why randomized smoothing, which DOES run on
```

## `LeanMlir/Proofs/Training/Trained/LinearDescent.lean`

```
   39  /-- Trained linear weights (input×class, entries `k/128`), test acc 0.874. -/
```

## `LeanMlir/Proofs/Training/Trained/MlpWitness.lean`

```
   14  `LipschitzCert.Instance` (test acc 89.8%) instantiates the conditional VJP
```

## `LeanMlir/ReferenceNets.lean`

```
  101  (mIoU 0.633 against 0.741 with skips, same schedule).
```

## `LeanMlir/Spec.lean`

```
  336  -- uses BN; original GoogLeNet predates BN but the delta is ~1%.
```

## `LeanMlir/SpecHelpers.lean`

```
  163  with pos/neg means −1.549/−1.803: the head had real signal (AUC 0.742) but
```

## `LeanMlir/SyncBnCheck.lean`

```
   48  whose committed DP shape is too large to gate (ResNet-50's 1×256 reference peaks at 94–95 %
```

## `LeanMlir/Types.lean`

```
  399  BraTS at 0.02% of CE's gradient once p₃ ≈ 2e-5
  425  the current prediction. At `p_t → 1` (confident background — 97% of
  594  the class argmax collapsed onto the two most frequent classes (car 44% +
  595  pedestrian 21% of encoded positives), leaving 5/10 classes never predicted
  714  → ~2× cheaper per step). imagenet (tfds) path only. -/
  735  data-loading-bound (e.g. ImageNet: ~75s/epoch rebuilding the tfds val
```

## `LeanMlir/Verified/NetsCore.lean`

```
 1475  At 100 epochs this spec reaches 76.68% top-1 on the verified path against 76.57% for the JAX
```

## `LeanMlir/Verified/Spec.lean`

```
   39  RAM against a 188 GB box, so it cannot be. Batches arrive over a pipe from **that net's own**
```

## `LeanMlir/Verified/Train.lean`

```
  500  -- gradients 5.8× larger (|g|max 0.85 → 4.94), which drops `wd·θ/g` to ~2e-6 and below f32
 1203  -- The demo nets are the MOST transfer-bound in the set (the dense probe at **75%**, against R34's
 1204  -- 55%), and residency measured **3.1×** on cifar8-bn. These loops are what a reader sits and
 1725  -- alone a ConvNeXt eval measured ~100 s/epoch at 91% util on GPU 0 while GPUs 1-3 sat at 0%. The
 1934  -- steps, and at R34 that is 260 MB each way per step that stops crossing PCIe (55% of a bs32
 2072  -- set (5.3 GiB) per epoch → OOM after ~30 epochs on a 188 GB box.
 2147  -- round-robin read then runs the whole job at its pace — 690 → 1,250 s/epoch on the 350-epoch
 2150  -- waits out its startup (~30 s, i.e. ~0.4% at E=10). A zero-downtime variant — spawn in the
 2188  -- the median exactly on the boundary: that is how §9.6's ViT row got 159 ms/step where the
 2189  -- steady state is 375 (2.4×). Benchmarks want PROBE_WARM=200 with MAX_STEPS=600.
 2212  -- regions — a fresh `F32.concat` per step would cost two whole-blob host memcpys (272 MB/step
 2225  -- pipe's buffer is 64 KB (this box caps `pipe-max-size` at 1 MB, still 0.6% of a batch), so the
 2226  -- producer fills its buffer, blocks in `write()`, and sleeps through the entire compute — 258%
 2232  -- step is **377 ms = 158 read + 219 rest**, so `max(158, 219)` = **219** and the read hides
 2233  -- COMPLETELY behind compute. Prefetched: **224 ms/step, 1.68×** (30 epochs 15.7 h → 9.3 h),
 2234  -- 5 ms off that ceiling. Bit-identity gated by `tests/prefetch_tie.sh`.
 2240  -- producer while the other n−1 sit blocked in `write()` with 64 KB buffered — 0.08% of a batch —
 2243  -- MEASURED at global depth 1, ViT/ImageNet 4×bs128, `SHIM_WORKERS=8`: the box ran **70% IDLE**
 2244  -- (22 of 32 cores) at 783 ms/step against a 249 ms synthetic floor, with the eight producers
 2246  -- as R34 without prefetch ("258% CPU on a 32-core box"). A zero-cost producer through the SAME
 2247  -- pipes at the SAME depth ran 248 ms, so the plumbing and the 308 MB of transport are not the
 2248  -- problem: 5 ms of the step. Capacity is not the problem either — making each producer 5.3×
 2249  -- faster (`SHIM_DETERMINISM=0`) moved the step 0%.
 2355  -- (and slicing [theta|m|v] back out afterwards) would cost two 272 MB host
 2369  -- init away only as `decay^t`: the reference MEASURED a shadow still holding 12.8% init at
 2370  -- 3.1 tau, scoring 0.00% top-1 while the live weights scored 70.48%. An 80-epoch Imagenette
 2447  -- `Task.Priority.default` (the pool), NOT `.dedicated`, and it is worth **12 ms/step**
 2453  -- — 150,120 of them over a 30-epoch run. Pooled lands 5 ms above the 219 ms synth floor.
 2469  -- until the kernel is under pressure — 79–136 MB/step, the box full inside one epoch,
 2470  -- then continuous direct reclaim and a mean step 2.5× the median
 2475  -- keeps allocating — at a 220 ms step cadence it changes nothing (measured).
 2602  -- because the A3 result (77.43%) is quoted as beating its JAX reference.
 2669  -- One 272 MB copy per EPOCH (for eval + checkpoint), not per step. Under
 2677  -- feature: ConvNeXt's 75.93% IS the shadow's number. The shadow is region 4, so it starts at
```

## `apps/ablation/MainAblation.lean`

```
  470  -- bare-recipe ViT-Tiny lands at ~72% Imagenette while CNNs hit ~88%.
  514  -- cutmix already wins (77.1%), this tests whether the WD bump that
```

## `apps/ablation/MainCifar8WideBf16Ablation.lean`

```
   11  3-epoch warmup + cosine, 40 epochs, bs 128). bf16 reaches the FORWARD, the input-VJP and the weight gradients — 23/23 convolutions — because the batched family is the one the 27 bf16 ops were built for. Expect no speedup (0.87× at these shapes); this is a numerics result.
```

## `apps/ablation/MainCifar8WideBnBf16Ablation.lean`

```
   18  Expect no speedup — bf16 measures 0.87× at cifar8's conv shapes. This is a stability and
```

## `apps/ablation/MainResnet34Ablation.lean`

```
   14  sequence, which is right at 40 epochs on CIFAR; here an arm is ~80 minutes, so eight of them in
   68  -- measurement of the rate rather than of the recipe — and what says whether `bare`'s 84.43%
```

## `apps/cifar/MainCifarSmooth.lean`

```
    7  is astronomically loose (global L = 942K, cert 0% at every radius). Randomized smoothing is
```

## `apps/imagenette/MainConvNeXtBImagenet.lean`

```
   28  single-device peer and the default here; it peaks at 9.53 GiB of the plugin's 11.68 default arena
```

## `apps/imagenette/MainConvNeXtImagenet.lean`

```
   34  -- **2.6x-10.2x wider** — 0.2041 vs 0.02 at the 4x4 stem, 0.2020 vs 0.02 at the 7x7 depthwise. Two
```

## `apps/imagenette/MainEfficientNetImagenet.lean`

```
   25  The JAX reference the chapter prints is the 350-epoch RMSProp `full` run (77.15% / 93.30%,
```

## `apps/imagenette/MainEfficientNetVerified.lean`

```
   24  on XLA/CUDA: `387/3925 = 9.859873%` on every epoch, byte identical, which is
```

## `apps/imagenette/MainEfficientNetVerifiedAdam.lean`

```
   81  -- this net's REFERENCE recipe (`efficientNetB0ImagenetConfig`) — and its 72.31% is the shadow's
```

## `apps/imagenette/MainMobileNetV2Imagenet.lean`

```
   26  reference number this net is measured against (**71.90% / 90.41%**, §6.5, `85daffbc`) is the
```

## `apps/imagenette/MainMobilenetV4VerifiedAdam.lean`

```
    8  80 epochs, bs32, AdamW, target **84.58%** — the JAX-baseline path's number for this block table. The
   37  The target is `RESULTS.md`'s **84.58%**, which is the JAX-baseline path's number for this
```

## `apps/imagenette/MainResnet34Imagenet.lean`

```
   35  30-epoch validation subrun is the opt-in (`LEAN_MLIR_EPOCHS`). ~27.9 h on four CUDA cards at
   36  the measured 18.6 min/epoch. 5-epoch warmup matches the reference; cosine as everywhere here.
```

## `apps/imagenette/MainResnet34VerifiedAdam.lean`

```
   48  * **`adam256`** — bs256, single device, worth **1.78×** img/s over bs32.
```

## `apps/imagenette/MainResnet50Imagenet.lean`

```
   30  composed A3 artifact (`lambaccdp8x64bce`) is specified for; the 4-GPU@160 probe measured 240 ms/step, so 100 epochs is ~33 h.
   41  That is the intended way to take a look before committing the full ~33 h. It is NOT the same
```

## `apps/imagenette/MainViTBImagenet.lean`

```
   23  11.68 GiB is NOT what "the BFC allocator gets on a 16 GB card" — it is what it gets at the CUDA
   25  `ffi/pjrt_ffi.c` passes. `LEAN_MLIR_MEM_FRACTION=0.97` gives **15.11 GiB**, and
   26  `vitbin_adamdp128x4wxclipdrop` then executes on four cards at **13.99 GiB, 93 %** of it. The
   28  *"Out of memory while trying to allocate 11.96GiB"*.
   30  | variant                        | per-dev | global | peak    | 11.68 GiB | 15.11 GiB |
   32  | `adamdp128x4wxclipdrop`        |     128 |  **512** | 13.99 G | OOM      | ✅ 93 %   |
   33  | `adamdp128x4wxclipdropbf16`    |     128 |  **512** | 12.61 G | ✅ 93 %   | ✅ 83 %   |
   36  2,502 steps/epoch at global 512 against 10,009 at 128 — **291 h fp32 and 178 h bf16** for 300
  104  DEFAULT arena: the graph peaks at 13.99 GiB and PJRT's CUDA plugin hands out 11.68 unless
  106  unhelpful — `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 11.96GiB` reads as
  109  The bf16 twin DOES fit at the default (10.88 GiB, measured), so the check exempts it rather
```

## `apps/imagenette/MainViTImagenet.lean`

```
   67  -- a shadow that averages ~4× faster than its reference's and reports it as the pair.
```

## `apps/imagenette/MainViTSImagenet.lean`

```
   25  probed (40 steps on real ImageNet, four cards): **531 ms/step fp32, 323 bf16**, i.e.
   26  113 h and 71 h for the 300-epoch schedule. `runs/2026-08-27-vitb-global512/`.
   46  peaks at **10.27 GiB**, which fits the plugin's default 11.68 arena at 88 %; ViT-B's is 13.99 and
   48  headroom (88 % → 68 %). It is not free: on ConvNeXt-S and -B the same 0.97 makes both bf16 arms die
   50  at the default, because a 97 % BFC pool starves what lives OUTSIDE it (device-to-host staging, NCCL,
```

## `apps/mnist/MainMnistCnnVerified.lean`

```
   29  table (84.6% of wall clock is parameter round-trip), so the two lowerers
```

## `apps/mnist/MainMnistLinearVerified.lean`

```
   43  Both produce an identical 12-epoch trajectory (final 9210/10000); XLA is ~2.3×
```

## `apps/mnist/MainMnistMlpSmooth.lean`

```
   11  Where the MLP's three-layer spectral-norm product gave a *vacuous* cert (L = 39, 0% certified),
```

## `demos/MainBratsPredict.lean`

```
   92  -- would instead punch black speckle through the brain: ~1.6% of brain
```

## `demos/MainDiffusion2d.lean`

```
    9  trains in seconds on 18,178 params rather than 7 h on 3M.
  252  -- 100 % off-support). Kept because that measurement is the evidence.
```

## `demos/MainNqsIsing.lean`

```
  189  same loops cost ~1.5 µs per pushed float: 11 s per step for the MLP at N = 64. -/
```

## `demos/MainPongDqn.lean`

```
   29  steps and the forwards hold their parameters, 24 → 14.6 ms per pixel update on
```

## `demos/MainTttEnv.lean`

```
   70  -- above 4×4 the root is the max over its openings, each solved full-window (3.5 min
```

## `demos/MainUnetBratsR34.lean`

```
   16  72% top-1, trained by this stack on ImageNet. Nothing here is downloaded.
   27  The from-scratch `unetBrats` result (mIoU 0.736, 3 epochs) is a
  120  a ~15% analytic-vs-finite-difference gap (measured by the FD probe) that
```

## `demos/MainUnetBratsTrain.lean`

```
   19  tumour is on the order of 1% of pixels, and the thin classes *are* the
   42  IoU + region Dice (WT/TC/ET) every epoch. Expect val mIoU ≈ 0.73,
   43  WT Dice ≈ 0.90 at 3 epochs.
   68  Measured, `dice` and `dicece` are the two BEST arms (mIoU 0.736 / 0.734)
   70  (0.640, and it over-paints 1.5–2.4×). Equalizing the loss shares is an
  111  β = 0    → all ones = plain CE            (mIoU 0.728)
  112  β = 0.5  → `unetBratsClassWeightsSqrt`    (mIoU 0.709)
  113  β = 1    → `unetBratsClassWeights`        (mIoU 0.640, over-paints)
  132  -- ~3.5 min on one 4060 Ti and the eval is a forward pass over 2,569 val
  182  -- where every other arm is monotone). It over-paints by 1.5–2.4×.
```

## `demos/MainYolov1NeuDet448.lean`

```
    8  arm's) and an epoch override. It scored mAP 0.0000 on VisDrone: seventy 20-px
```

## `demos/MainYolov1NeuDetFpn.lean`

```
   58  -- Measured routing at 448 px, 24 / 64 px thresholds: P3 0.0% / P4 1.9% / P5 98.1%
   66  -- P5 carries 98% of the boxes at every aspect from 64 px to the full frame, and
   67  -- three anchors fit that poorly: mean best wh-IoU 0.52, recall@0.5 0.52 (k=9
   71  -- slots per GT box, is 99.9% on NEU against VisDrone's 88%.
```

## `demos/MainYolov1VisdroneFpn.lean`

```
   10  Two backbones, selected by `FPN_BACKBONE`: `r50` (default, RSB-A3 77.2% top-1)
   55  car 44.1% and pedestrian 21.2% of positives, and the unweighted e12 head
   70  top-1 overall  67.58% → 67.32%     top-3 overall  92.88% → 92.78%
  191  -- needed. Targets the measured failure: objectness had AUC 0.742 but every
  210  score 77.2%), most likely a scale folded against running statistics kept in the
  229  R50 is the default because it is both the better base — RSB-A3, 77.2% top-1
  230  against R34's ~74% — and the only one that still has a loadable checkpoint:
  268  hundreds of epochs and wants the loss trajectory, not 100 × 86 MB of
```

## `demos/probes/MainGradFdProbe.lean`

```
    6  Probing `r34UnetBrats` turned up a systematic ~15% gap between the analytic
   26  best-established path in the repo; if `mlp` shows a 15% gap then the probe
```

## `demos/probes/MainMnistDdpmScore.lean`

```
   10  Chapter 3's `cnnVerified` (`LeanMlir/Verified/NetsCore.lean`), 98.75 % at ten
```

## `jax/Jax/Codegen.lean`

```
  466  -- no-op when upsampling (measured: 0.3106 vs 0.3107 at 0.70×), and RandomResizedCrop's area
  496  -- FLAT from 0.7× to 11.7× downscale. Without it the error grows with the ratio — 0.31 at 1.0×,
  497  -- 2.54 at 3.9×, 13.73 at 11.7× — i.e. it concentrates on the biggest images in the set.
 2901  --     29.91% sorted -> 70.15% shuffled, same weights and recipe.
 3551  bs256 this is ~154 MB/batch against a ~670 ms step = ~230 MB/s, and a pipe does GB/s, so the
 3626  -- 2 processes 1.71x, 4 processes 2.36x on a 32-core box — so two clear the ~1,940 img/s a
 3633  -- handle at a time holds the ViT job at 567 ms/step against a 249 ms floor at BOTH producer
 3635  -- takes it to 287 ms/step (1.98x) — a consumer change, with the producers untouched.
 3651  -- lands on producer 195 % N). Every producer reads all 6.3 GiB of raw records (cheap from cache)
```

## `jax/MainConvNeXt.lean`

```
   15  verified path the paper's 1e-6 lands 81.78% where ones landed 85.45% on the
```

## `jax/MainConvNeXtBImagenet.lean`

```
   84  4× 16 GB: one-shot bs512 peaks at 11.52 of 11.68 GiB and runs 747 ms/step;
   85  4×128 peaks at 6.19 GiB and runs **681 ms/step**. Under memory pressure XLA
   87  Also beats bs256 per epoch (28.4 min vs 32.9). -/
```

## `jax/MainConvNeXtImagenet.lean`

```
    9  bf16 incl. bf16 conv: the depthwise-7×7 is 2.32× faster in bf16 and the
   57  no-RandAugment 80ep run that hit 75.93%. The blueprint named this as the
```

## `jax/MainConvNeXtSImagenet.lean`

```
   89  Measured on 4× 16 GB: bs512 in one shot fits at 11.26 of 11.68 GiB (481
   90  ms/step) but leaves only 0.42 GiB — and the probe does not model the tf.data
   91  prefetch buffers that also live on device. The 2×256 accumulation costs 8%
   92  (521 ms/step) and drops peak to 6.91 GiB, which is the version to actually
   93  run. Larger batch is a win per epoch either way: 21.7 min vs 21.9 at bs256. -/
```

## `jax/MainEfficientNet.lean`

```
   12  -- swish there too; without this line the reference is not, a deviation worth 51% of logit range —
```

## `jax/MainEfficientNetImagenet.lean`

```
   21  -- The Imagenette twin carries this line too; the deviation measures at 51% of logit range — five
```

## `jax/MainMobilenetV4Imagenet.lean`

```
   91  -- eval every 5 ep (per-epoch 50k-img val wastes ~1.75h over 100ep)
```

## `jax/MainResnet50Imagenet.lean`

```
   43  CUDA box (ares), where bf16 conv on cuDNN tensor cores is ~1.6× faster
   44  (measured: 458→ vs 737 ms/step on 4× 4060 Ti, A2@224). On ROCm/MIOpen bf16
   74  -- CUDA/cuDNN: bf16 conv ~1.6× faster (R50 is conv-bound, ares is its home); slower-but-correct on ROCm
  120  @160 / test @224 (crop 0.95)** — the resolution split is ~2× faster/step, so A3
  121  is ~6× cheaper than the 300-ep A2 (~10-11 hr on ares vs ~60-65 hr).
  153  ~2x cheaper per step (FixRes); this recipe pays that back. Do not read a 2018-vs-A3
  178  /-- Optimizer-regime probe (diagnosing the ~41% RSB-A3 result). Same A3 recipe
  191  at 1/4 its intended batch (40.8% on A3). BN stats are per-micro-batch
  232  `LEAN_MLIR_RESUME` (per-N-epoch `.state.npz`) makes the ~24 h run spot-safe.
  244  gave A3 **40.8%** instead of 78.1%: LAMB is a large-batch optimizer and bs512
  246  **76.66%**, so A2 should be run the same way.
  252  (measured 7.41 GiB on 4× 16 GB @224). -/
```

## `jax/MainResnetImagenet.lean`

```
   36  -- full paper recipe (4-GPU bf16 run, ~18 hr clean)
   45  -- bf16 mixed precision (incl. bf16 conv): a CUDA/cuDNN recipe — 1.60x faster
   46  -- than fp32 on the 4060 Ti box, reaching 74.16% top-1 / 91.92% top-5 over the
   47  -- full 50k val (the 72.0% in jax/runs/r34_imagenet_bf16_90ep/RESULTS.md is the
```

## `jax/MainVitImagenet.lean`

```
   29  The 80-epoch grad-clip-only ancestor of this recipe reached 65.6% top-1;
   78  at loss 7.4637, timm 14.28 at loss 7.1597 — 3.1x better conditioned, and a
```

## `jax/MainVitSImagenet.lean`

```
   69  /-- 80-epoch validation tier — the schedule ViT-Ti was actually run at (65.6%),
```

## `lakefile.lean`

```
  325  when someone ran it. Linking is what makes the exes expensive (~149 MB
  455  -- The first CONVOLUTIONAL graph on the XLA ladder — where IREE's ~1%-of-peak
  492  /-- 80ep, bs32, AdamW, target 84.58%. XLA/PJRT only — no
  632  `scripts/supervise.sh vits-default-g512-4gpu` is the job: 528 → 319 ms/step measured,
  633  113 → 71 h for 300 epochs. Renders, ties and STEPS; NOTHING has been trained on it. -/
  786  falls short (an 8.7% false-positive floor). It also checks
  810  -- FORWARD-only bf16, and NO speedup by design (§5.3: 0.87× at cifar8's shapes) — the arms
  929  -- the conv-aware spectral product is 942K-loose (cert 0%). Same forward-only procedure, any depth.
 1579  1×128 with its DP step rendered at run time too (the 1×256 reference peaks at 94–95 % of the
 1591  0.94% for a net really at ~4.4%, because it can only ever be right on labels 0..9. The file
 1740  1.68× the emitted ops (10014 vs 5971) because `pretty` has no CSE and the batched backward ops
 1750  -- segmentation batch (mAP@0.5 0.0001 vs 0.1167).
 1990  -- tiered by time budget. `lake run mnist` (~30 min) / `lake run cifar` (~1 hr);
 2099  IREE lowerer selected instead of XLA. ~30 min (XLA is 2.3-4.3x faster here). -/
 2117  80-epoch AdamW at 224². **~37 h end-to-end** (9.5 + 5.4 + 6.2 + 13.3 + 2.3,
 2125  (86.24% / 89.71%, medians of five), both off the XLA path. **`imagenette` is the official set**; this
 2135  -- XLA is the default because it is where every quoted number comes from, it is ~4.6×
 2142  -- Why you'd reach for these: XLA is **4.6× IREE** on EfficientNet — 80 epochs in
 2143  -- 1 h 35 m against 7 h 50 m — and multi-GPU is reachable ONLY
 2145  -- entry point outright. Re-measure per net rather than assuming 4.6×; it is one
 2167  binaries rather than six. Wide costs about 1.6x the wall-clock per epoch over the narrow pair
 2186  | ResNet-34 | 89.99 (mean of five seeds) | 45 min | `runs/2026-09-12-r34-ablation-fp32-seeds/` |
 2187  | ResNet-50 | 89.71 (median of five) | 74 min | `runs/2026-08-31-imagenette-n3/` |
 2188  | MobileNetV2 | 89.25 | 36 min | ch 6 |
 2189  | MobileNetV4-Conv-M | 86.24 (median of five) | 33 min | `runs/2026-08-31-imagenette-n3/` |
 2190  | EfficientNet-B0 | 89.96 | 39 min | ch 7 |
 2191  | ConvNeXt-T | 82.27 | 75 min | `runs/2026-09-13-convnext-imagenette-ls1e-6-resident/` |
 2192  | ViT-Tiny | 68.74 | 24 min | ch 9 |
 2403  -- on the reference 7900 XTX, XLA is 2.2× IREE on the conv anchor, 4.7× on the dense one and **8.6×
 2404  -- on the attn one** (and 4.6× on EfficientNet). A single blended factor would be wrong in both
 2416  -- (9.5h / 5.4h / 6.2h / 13.3h) and the IREE ViT row is measured here (7.8h warm —
 2417  -- the 2.3h figure elsewhere is the JAX bf16 path, not this verified trainer). The
 2418  -- IREE rows EXCLUDE the one-time IREE compile (~10–15 min/arch, CPU-bound,
 2421  -- own anchors (every factor reads ~1.0×).
 2424  --   * ch3 (MNIST CNN) reads 23764 ms/epoch; re-measured on the same card, same
 2425  --     basis (real data + eval, steady state) it is **17659** — the row is ~1.35×
 2430  --     one session on the reference card, i.e. 0.82-0.93×. Unlike the XLA anchors,
 2433  --     card being slow; the conv (1.01-1.02×) and attn (1.01×) anchors reproduce.
 2454  arithmetic: the param share of a step measures **33.5%** for the conv probe
 2455  (`cifar8-bn`, 32²) against **59.4%** for ResNet-34, **46.7%** for EfficientNet and **84.6%**
 2462  **0.56×** and dense **1.33×** (idle card) — a 2.4× spread that is not noise but two different
 2464  **89m** (66.7 s marginal epoch × 80, `runs/r34_pool3s2_80ep_aug04.log`), i.e. **2.5×
 2465  optimistic**. The bracket's transport end predicts **84m** — 1.06× low.
 2468  like-for-like ratio is **5333/3780 = 1.41×**, above *both* probe factors. The cause is
 2470  loader **by design**, and per-epoch host overhead measures **6.3%** of a 1-GPU
 2473  answer lies in. It takes ch5 from 2.5× wrong to 1.06× wrong; that is the whole claim.
 2514  **0.3% out**. So `scripts/sweeps/marginal_epoch.sh` × epochs is trustworthy at
 2521  ±6% per-run spread documented on `probeConvRefMsXla`; the conv-family ones (ch3, ch4)
 2522  are the affected pair. Treat them as ±6%, not as exact. -/
 2525  -- IREE 535ms × 12   | XLA 239ms × 12
 2527  -- IREE 3200ms × 12  | XLA 676ms × 12
 2529  -- IREE 23764ms × 10 | XLA 4103ms × 10  84.6% param round trip
 2531  -- 40 ep × 6 ARMS, approximated as the BN arm ×6 (the 3 no-BN arms are cheaper) — the same approximation the ref column makes, kept so the two stay comparable      -- IREE 8490ms×40×6  | XLA 3698ms×40×6
 2533  -- 6.31 s/epoch × 240 = 1514 on `cifar8w-bn-ablation`, which
 2537  -- (`runs/2026-08-26-cifar8w-6arm-timing/`: 374 s for the 3 no-BN arms at 3.12 s/epoch + 757 s
 2538  -- for the 3 BN arms at 6.31). So the BN-arm×6 approximation OVERSHOOTS by 34% here — the
 2547  -- IREE 9.5h  | XLA 1h03m (the PAPER net). 59.4% param round trip
 2549  -- IREE 5.4h  | XLA 1h25m measured on the net with its 52 conv biases
 2551  -- IREE 6.2h  | XLA 1h34m  46.7% param round trip
 2553  -- IREE 13.3h | XLA 1h54m01s (the channel-LN net, 6841s)
 2555  -- IREE 7.8h (1185ms/step × 295 × 80, warm steady-state) | XLA 0.97h = MEASURED 80-epoch wall 3491s
 2559  per-chapter cross-vendor ratio spans 2.4× and no single probe factor fits it — see
 2568  the dataset's real step count, eval skipped — so the on-reference factor reads ~1.0×
 2574  cold-cache / GC-blip outliers that make a 40-step mean swing ±10%+).
 2577  own factor. (The 2.3h ViT figure elsewhere is the JAX bf16 path, not this
 2578  verified-IREE trainer, which is ~7.8h here.) -/
 2584  the IREE anchors these read 4.66× (dense) and 2.19× (conv), which is the whole reason
 2596  `MIOPEN_DEBUG_CONV_GEMM=0` is **not needed**, and setting it costs ~7% (attn probe 136 vs
 2597  128 ms/step median; marginal epoch 46.5 s vs 43.5 s). Why it fired once is unexplained —
 2620  / 3528 / 3565 / 3733 / 3774 / 3778 / 3792 / 3865 ms/epoch — a ±6% band with no pattern.
 2621  So a single sample can read 0.94× against its own anchor and look
 2622  like a regression when nothing changed. The dense probe is stable to ±1.5% (601 / 605 /
 2623  607 / 610 / 610 / 610 / 615 / 619). Read an on-reference factor of 0.94-1.06× as
 2629  MIOpen override (123/125/126/127/128/129/132/137 — ±5%). Against IREE's 1173 that is
 2630  **9.2×**, the largest cross-lowerer gap of the three families and the reason ViT cannot
 2632  is a ~7% regression, not a fix; see the note above.) -/
 2638  -- measured directly on all five Imagenette nets, spans **0.585 → 1.411 — a 2.4× range**:
 2642  -- a step is parameter transport (33.5% for the conv probe against 59.4% for R34), and the
 2685  baseline, which the 2.4× cross-vendor spread (see `probeDenseRefMsCuda`) says is the best a
 2699  **The conv proxy for a transformer measured ~3.5× LOW**, which is why the attn family
 2700  exists at all: on a 4060 Ti the three factors came out dense 4.82× / conv
 2701  3.54× / **attn 11.98×**, so scaling ViT as conv estimated 7.9 h against a measured ~28 h.
 2704  instance of the same principle, measured on one card across lowerers: XLA-vs-IREE is **2.24× for
 2705  conv but 9.2× for attn**. Attention and convolution do not track each other, whether you change
 2708  **The 3.5×-low proxy figure is an IREE fact, not a 4060 Ti fact.** Measured on
 2710  reads dense **1.29×** / conv **0.54×** / attn **0.70×** — i.e. it BEATS the reference 7900
 2711  XTX on both conv and attn, and the conv proxy for attn would have been off by only 1.3×
 2712  rather than 3.5×. Against the IREE column's 4.82/3.54/11.98 for this same card that is a
 2713  3.7× / 6.6× / **17×** improvement, which is a statement about IREE's CUDA backend rather
 2716  `*proxy` row is a mild approximation, not the 3.5× trap it is on IREE. -/
 2808  assumption. The method is validated at **0.3%** (ch9's wall extrapolates to 3480 s from a
 2813  ~6.3% of a 1-GPU epoch. Excluding it is most of why the transport-bound estimate came in 6% low
 2906  --   3-epoch probe would pay ~10-15 min of iree-compile per net.
 3055  All nine XLA references are measured; ch.9's is a real 80-epoch run (3491 s, val 71.31%),
 3056  which confirms the marginal-epoch extrapolation to within 0.3%. See `benchTable`.
```

## `scripts/audit_census/Dump.lean`

```
    5  `lake env lean --run scripts/audit_census/Dump.lean <modules.txt> <decls.tsv>` (~2 min, ~7 GB).
```

## `tests/BlueprintCheckDecls.lean`

```
   19  1 h 5 m build, since checkdecls runs last.
```

## `tests/TestArgmaxN.lean`

```
   57  -- The reference's headline is quoted as "72.02% top-1 / 90.62% top-5". `rankOf` counts strictly-greater logits, so the label is in
```

## `tests/TestChannelLN.lean`

```
  265  -- 1-3 ms — a resolution comparable to the quantity being measured. Time `inner` invokes per
```

## `tests/TestCifar8DpCheck.lean`

```
   28  0.0007–0.0034 across identical invocations — 7× to 34× over the 1e-4 threshold, and *red every
```

## `tests/TestConvBiasZero.lean`

```
  529  -- cancellation leaves a residue on ~93% of coordinates. Two runs disagree on which coordinates,
```

## `tests/TestEfficientNetAdamTie.lean`

```
   18  1.68× on R34 costs nothing after XLA optimisation). The cotangents are also composed differently —
```

## `tests/TestEfficientNetTrain.lean`

```
   26  `runs/efficientnet_verified_crop_gpu1.log` descends 40.6% → **87.81%** over 80 epochs, matching
   27  README's 87.58%. Leave the number alone. A `tests/` writer that re-rendered it as **mean**-CE at
```

## `tests/TestImagenetSyncBnCheck.lean`

```
   25  reference is 14.14 / 14.35 GiB (bf16 / f32), 94–95 % of even the raised arena. Its DP step is
   36  compiled peak at 256 is 6.2 / 5.8 / 6.7 GiB for R34 / MNv2 / B0 in bf16, inside the default
```

## `tests/TestLabelDecode.lean`

```
   15  **1..988**, with **193 of 256 (75.4%) above 255**.
```

## `tests/TestR34SyncBnCheck.lean`

```
   47  2e-4 off in the statistics after 36 layers and 15 % off in `m'` on this gate, with the sensitivity
```

## `tests/TestR50GradCheck.lean`

```
   75  So the honest sentence is: **tier 2 pins the gradient's magnitude to ~0.1% at the head and stage
   76  4, loosening to ~17% at the stem**, and tier 1 pins its structure to ~6e-5 everywhere including the
   98  the shortcut is only ~4% of `‖m'‖²` at s1b0.)
  101  tie, which at the stem is 6.9× and at stage 4 is ~700×.
  358  -- runs on the SAME seeded base point, `bC` spreads **2.75× under CE and 10.5× under BCE**, so
  364  -- **1.1×** across the same three runs, because it takes one site's collapse to move the minimum
  491  -- The check below therefore reads the 10th-percentile control, not `bC`: `bC` spreads 2.75× (CE)
  492  -- and 10.5× (BCE) over three runs on the same seeded base point, the quantile 1.1×.
```

## `tests/TestResnet34AdamBench.lean`

```
    6  The batched render (the committed `verified_mlir/resnet34_adam_train_step.mlir`) is **1.68× the
   22  (~272 MB each way) over PCIe because parameters are host-resident. That cost is
  122  -- ── compile both (timed — B is 1.68× the ops, so this is a dev-loop cost worth naming) ────
```

## `tests/TestResnet34BatchCheck.lean`

```
   45  * `LEAN_MLIR_MEM_FRACTION=0.97` — the bs256 step wants one 11.50 GiB allocation and the plugin's
   46  default BFC pool is 11.68 GiB, already part-consumed by the two bs32 runs, so the gate dies with
```

## `tests/TestRmsTie.lean`

```
   11  JAX's 68.33%, and one of two for EfficientNet's 72.31%. This is their numeric gate, built the
```

## `tests/TestSDPA.lean`

```
  101  ~0.5 over seeds 0–19), so f32's own ~1.5e-5 finite-difference floor is 1% of it and the check
```

## `tests/TestShufflePairing.lean`

```
   12  entire existence and could only learn the marginal target distribution: mAP@0.5
```

## `tests/TestSoftTargetTie.lean`

```
  134  -- ~0.15% of coordinates by sub-print-precision amounts, which is its documented ill-conditioned
```

## `tests/TestUibLayoutTie.lean`

```
   16  Imagenette demo — the one `historical/RESULTS.md`'s 84.58% belongs to, NOT faithful Conv-M). All four
```

## `tests/TestViTDpCheck.lean`

```
   48  variable costs ~7%. Diagnosis and a 20-line JAX reproducer:
```
