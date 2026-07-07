# paper_faithfulness.md — the canonical fidelity ledger (all ImageNet trainers)

**THE single source of truth for how close each Lean→JAX ImageNet trainer is to its
published paper recipe.** Audited hostile-referee style (aim to *reject* the "faithful"
claim; no charitable rounding). Keep this current when a recipe/codegen change moves
faithfulness. Last full audit: **2026-07-07** (fresh per-net audit, code as ground truth).

- Config lives in `jax/Main<Net>Imagenet.lean`; aug/optimizer/BN codegen in `jax/Jax/Codegen.lean`.
- R50 has a deeper writeup in `planning/rsb_a2_resnet50.md` → "FIDELITY LEDGER".
- Run logistics / supervisors / the older per-net audit: `planning/jax_imagenet_sweep.md`
  (its "Paper-faithfulness deltas" section is SUPERSEDED by this doc — kept for run history).
- **Scope:** the 6 full-ImageNet paper-target trainers below. The newer variants
  (EfficientNetV2-S, MobileNetV3-L, MobileNetV4-M) are Imagenette-scale 80ep architecture
  demos, not accuracy reproductions — their faithfulness is param/arch level (bestiary report).

## Scoreboard

| Net | Measured (this repo) | Paper | Gap | Run status | Headline referee objection |
|-----|----------------------|-------|-----|------------|----------------------------|
| **ResNet-50** (RSB-A3) | 76.66 / 93.03 | 78.1 | −1.44 | ✅ done (`rsb-faithful`) | BN regime (Ghost-BN + K×/step) + bf16; `true-2048` unrun |
| **ResNet-34** | 72.1 / 90.7 | 73.3 | −1.2 | ✅ done (90ep) | bf16-conv + no PCA/color aug; otherwise clean |
| **ConvNeXt-T** | 75.93 (80ep, old) | 82.1 | −6.2 | ⚠ stale; full unrun | default 80ep≠300; full aug+wd-mask+LR now faithful, un-re-measured |
| **ViT-Ti / DeiT-Ti** | 70.28 / 90.05 (300ep) | 72.2 | −1.9 | ✅ done (300ep) | RepeatedAug 3×→1× deferred (largest remaining lever) |
| **EfficientNet-B0** | none stable | 77.1 | — | ❌ no stable run | default LR now paper 0.016 (erodes in ε1e-3 stack); 80ep≠350; no run |
| **MobileNetV2** | 68.77 / 88.53 | 72.0 | −3.23 | ✅ done (90ep)† | 90ep≠~350; unscaled LR (LS→0 landed; number pre-dates it) |

Only measured top-1s are stated as fact; "unrun" = the faithful config exists but has not
produced a verified number. **No net currently reaches its paper number**, but four of six are
within ~2pt (R34 −1.2, R50 −1.44, ViT −1.9, and MNv2 −3.23 pre-fixes). The two real outliers are
ConvNeXt (−6.2, but that number predates the full aug pack + today's fixes — un-re-measured) and
ENet-B0 (no stable run). Every gap is schedule-tier or config, not a model bug.

† Measured numbers pre-date the 2026-07-07 config fixes (ConvNeXt wd-mask/LR, ENet LR, MNv2 LS→0);
no number here has been re-measured under the new configs yet — a re-run is needed to reflect them.

## Cross-cutting

**RandAugment now literal timm (landed 2026-07-07, `Jax/Codegen.lean`):** op set = timm's
15-op `_RAND_INCREASING_TRANSFORMS` (inc1: +SolarizeAdd +Invert, no Identity), per-op apply
prob 0.5, geometric interp BILINEAR, TranslateX/Y `-Rel` (0.45×dim). **Affects & correct for:**
R50 (m6), ConvNeXt-T (m9), ViT/DeiT-Ti (m9). **N/A to:** R34 + MNv2 (crop/flip only),
ENet-B0 (uses AutoAugment, whose `_AA_POLICY` was separately spot-checked faithful).

**Gaps A–D (all DONE, 2026-06-21):** A running-BN eval stats — confirmed `runningBN:=true` on
the 3 BN nets (R34, ENet-B0, MNv2) + R50; N/A to the LN nets (ConvNeXt, ViT). B exp-decay LR,
C classifier dropout, D RandAugment mstd/inc — all wired.

## Fixes applied (2026-07-07) — faithfulness-closing

All four audit-flagged fixes below are LANDED (config edits in `Main<Net>Imagenet.lean`; built +
regenerated + py_compiled). They change training behavior — the affected nets' prior measured
numbers were taken under the OLD values.

1. **ConvNeXt-T `wdExcludeNormBias := true`** ✅ — WD 0.05 no longer decays LayerNorm γ/β, biases,
   or LayerScale γ (1e-6). Codegen mask is `p.ndim <= 1 → 0` so it correctly catches all three.
   Emitted trainer now has `_wd_mask`/`WD_MASK`.
2. **EfficientNet-B0 default LR 0.045 → 0.016** ✅ — the EfficientNet reference base LR at bs256
   (= 0.256@bs4096). NB our RMSProp ε1e-3 stack *erodes* at 0.016 (memory
   `project_enet_lr_instability`); `_full` keeps the stability-adjusted 0.01. The old 0.045 was an
   arbitrary MobileNet-style value that diverges.
3. **MobileNetV2 `labelSmoothing := 0.1 → 0.0`** ✅ — matches Sandler 2018 (used none).
4. **ConvNeXt-T LR 4e-4 → 2.5e-4** ✅ (strict linear scale of 4e-3@bs4096) **+ stale docstring
   fixed** (stem is faithful patchify+LN, not convBn/ReLU; mixup/cutmix are on).

---

## Per-net detail

### ResNet-50 (RSB-A3)
**Result:** `rsb-faithful` recipe → 76.66 / 93.03 vs paper A3 78.1 (−1.44). Full detail +
recipe table (`default`/`short`/`rsb-faithful`/`true-2048`/`adam-probe`) in
`planning/rsb_a2_resnet50.md`.
- **Faithful:** LAMB @ design batch (eff bs2048 via grad-accum), lr 0.008@2048, cosine+5ep
  warmup, BCE over multi-hot mixup/cutmix, `wdExcludeNormBias` (timm no_weight_decay), running-BN,
  RandAugment now literal `rand-m6-mstd0.5-inc1`, train@160/eval@224 crop 0.95.
- **Deviations:** bf16 matmul+conv (kept, ≈−0.1pt); BN regime — Ghost-BN (each accum micro-step
  normalizes over its own 512) + running stats updated K×/step; `true-2048` (real single-forward
  bs2048, needs ~80GB) added but **not yet run**.
- **Residual −1.44pt:** BN regime + LAMB impl micro-details + bf16 (RandAugment deltas now closed).

### ResNet-34
**Result:** ~72.1 / 90.7 (90ep, bf16) vs paper/torchvision ~73.3 (−1.2). Faithful-modernized
classic ResNet, not the literal 2015 step-decay recipe.
- **Faithful:** basic blocks [3,4,6,3], 7×7/2 stem, exact torchvision arch; SGD+mom 0.9;
  LR 0.1@bs256 (correctly unscaled — bs256 is canonical); 90 epochs; WD 1e-4 on all params
  (classic decay-everything); RandomResizedCrop+hflip only; no SD/dropout/EMA; train=eval 224,
  test crop 0.875 BICUBIC; running-BN eval. RandAugment change N/A (crop/flip net).
- **Deviations:** cosine+5ep warmup vs original step-decay/no-warmup (deliberate); label
  smoothing 0.1 vs 0.0 (deliberate); no PCA-lighting/color-jitter (RRC+flip is the torchvision
  substitute — small content deviation); bf16 incl. conv vs fp32.
- **Residual −1.2pt:** bf16-conv precision + missing color/PCA aug (LS+cosine should help, not
  hurt); no arch/optimizer/LR bug found.

### ConvNeXt-T
**Result:** last measured 75.93 (old 80ep, RandAugment-off) vs paper 82.1. Arch faithful; full
aug/regularizer pack now present but the paper 300ep tier is only the `full` recipe arg, and
neither 300ep nor the full-aug 80ep has been re-measured.
- **Faithful:** patchify 4×4/s4 → channel-LN stem (no BN/ReLU — code correct, docstring stale);
  DW7→chLN(1e-6)→1×1 expand4→GELU→1×1→**LayerScale 1e-6**→residual; depths (3,3,9,3) widths
  (96,192,384,768); AdamW β(0.9,0.999) eps1e-8 decoupled; WD 0.05; **20-epoch warmup** (exact);
  cosine; LS 0.1; Mixup 0.8 + CutMix 1.0 + RandomErasing 0.25; drop-path 0.1 ramp; EMA 0.9999;
  **RandAugment m9-mstd0.5-inc1** (today's fix is exactly this net's recipe — correct); no BN
  (gap A N/A); eval crop 0.875; bf16+conv, LN/GELU fp32.
- **Deviations:** **default = 80ep, not 300** (biggest lever, only the `full` recipe arg); grad-clip 1.0
  (paper uses none); mixup/cutmix alternate-per-step vs switch-prob; cosine floor 0 vs 1e-6 (negligible).
  FIXED 2026-07-07: `wdExcludeNormBias` now on (was decaying LN/bias/LayerScale); LR 4e-4→2.5e-4 (linear).
- **Residual −6.2pt:** now schedule tier (80 vs 300) + un-re-measured full aug/wd-mask/LR pack;
  the two config deviations (wd-mask, LR) are closed but not yet reflected in a measured number.

### ViT-Ti / DeiT-Ti
**Result:** **300ep DeiT-Ti run completed → 70.28 / 90.05** (ROCm 2×7900 XTX), vs paper 72.2
(−1.9); the 65.6 is the older 80ep grad-clip-only ancestor. (The 2026-06-21 doc's "unrun" was
stale — blueprint §ViT records the finished run.)
- **Faithful:** patch16/embed192/3-head/12-block, MLP×4, pre-norm LN, GELU, CLS+pos-embed
  (WD-excluded), LN-only (no BN); AdamW β(0.9,0.999) eps1e-8, WD 0.05 with `wdExcludeNormBias`;
  LR 5e-4@bs512 (= DeiT 5e-4×bs/512), 5ep warmup→cosine, grad-clip 1.0; 300 epochs; LS 0.1;
  Mixup 0.8 + CutMix 1.0 + RandomErasing 0.25; **RandAugment m9** (today's fix moves it faithful);
  stochastic depth 0.1 per-branch ramp; EMA 0.99996; test crop 0.875 BICUBIC, 224/224.
- **Deviations:** **RepeatedAugmentation 3×→1× DEFERRED** (single largest gap; codegen supports a
  stream-level `flat_map(repeat K)` *approximation* of timm's index-level RASampler, not exact);
  bf16 matmul vs DeiT fp16-AMP (deliberate); the faithful 300ep run has not produced a verified
  top-1 yet (65.6 is pre-flip 80ep).
- **Residual −1.9pt:** dominated by RepeatedAug 3× deferral + aug-fidelity drift (RRC/Random-Erase
  are from-scratch tf.data; RandAugment now timm-literal as of 2026-07-07) + minor knobs (bs512 vs
  1024, grad-clip, Random-Erase-to-zero). Every recipe axis is code-faithful.

### EfficientNet-B0
**Result:** paper 77.1; **no stable completed run**. Recipe is the most paper-complete in the
repo (all four ENet signature pieces faithful) but the *default* tier ships the divergent LR.
- **Faithful:** exact B0 MBConv stage table (t/c/n/s/k), SE on every block, Swish/SiLU, 1280 head,
  mult 1.0; RMSProp ρ0.9/μ0.9/**ε1e-3** (matches timm, not 1e-8); exp-decay ×0.97/2.4ep, 5ep
  warmup; WD 1e-5; LS 0.1; classifier dropout 0.2; drop-connect 0.2; EMA 0.9999; running-BN;
  res 224; **AutoAugment** (`_AA_POLICY` spot-checked = canonical ImageNet AA, all 25 sub-policies).
  Today's RandAugment change N/A (AA net).
- **Deviations:** default LR now **0.016** (EfficientNet base LR @bs256, FIXED 2026-07-07 from the
  arbitrary divergent 0.045) — but our RMSProp ε1e-3 stack *erodes* at 0.016 (memory
  `project_enet_lr_instability`), so `_full` keeps a stable 0.01; **default 80ep vs paper 350** (AA
  under-trains at 80); bs256 vs paper 4096; bf16+conv vs fp32; 5ep warmup (timm addition); eval
  resizes straight to 224 (no 256→center-crop-224, crop_padding=32 — minor top-1 leak).
- **Blocker:** not recipe-completeness — schedule length + LR fragility; no stable 350ep@≤0.01 run
  has confirmed 77.1.

### MobileNetV2
**Result:** 68.77 / 88.53 (90ep RMSProp, offline full-50k EMA eval) vs paper 72.0 (−3.23).
- **Faithful:** exact paper `[t,c,n,s]` table (17 inverted-residual blocks, ReLU6, linear
  bottleneck, mult 1.0); RMSProp ρ0.9/μ0.9/**ε1.0** (paper/timm value); WD 4e-5; exp-decay
  ×0.98/epoch; minimal aug (random-crop+hflip only, no AA/RA/mixup/cutmix/erase — correct);
  no stochastic depth; no in-training EMA (paper-faithful); running-BN; 224/224. RandAugment N/A.
- **Deviations:** label smoothing **FIXED 2026-07-07 → 0.0** (matches paper; the 68.77 number
  predates this and used 0.1); **default 90ep vs paper ~300–400** (measured 68.77 is the 90ep undertrained point;
  350ep `_full` untested); reported metric uses **offline post-hoc EMA weight-average** not in the
  paper or the trainer (`useEMA=false`) — apples-to-oranges eval; warmup 5ep added; LR 0.045
  unscaled at bs256 (effectively under-scaled → undertrains further); dropout 0.2 (defensible);
  bf16+conv.
- **Residual −3.23pt:** short 90ep schedule dominates + LS 0.1 mismatch + unscaled LR; arch/
  optimizer/WD/aug are faithful — the gap is training-budget/regularization, not model.
