# recipe_fidelity_diffs.md — per-net recipe deviations, code as ground truth

Companion to `planning/paper_faithfulness.md`. That doc is the **scoreboard** (measured
vs paper, one row per net). This one is the **diff list**: every place the emitted
trainer departs from its reference recipe, why, and how much it is likely worth.

**Method.** Audited by emitting each trainer (`.lake/build/generated_*.py`) and reading
the Python, not by reading the Lean config or trusting prior notes. Where a reference
value could be checked against upstream source it was; those are marked **[verified]**.
Where the reference is from the paper/recipe string and could not be checked from this
box, it is marked **[unverified]** — treat those as claims to confirm, not facts.

Audited 2026-07-27. Nets: R50 RSB-A3, ConvNeXt-T, ViT-Ti, EfficientNet-B0, MobileNetV2,
MobileNetV4-Conv-M.

---

## Cross-cutting (affects several nets)

### C1. Resize is not antialiased — **~0.2 pt, MEASURED (⚠ on two nets, and now we know why)**

⚠⚠ **SCOPE, added 2026-08-14 — the table below covers ViT-Ti and ConvNeXt-T and could not have
covered anything else.** `eval_preproc_ab.py` called `m.forward(params, x)`, which is the emitted
signature for a **LayerNorm** net; every BatchNorm net's forward is
`forward(params, x, bn, training, …)` and raised `TypeError`. So R50, MobileNetV2, EfficientNet
and MNv4 were never measured here — the coverage was decided by an arity mismatch rather than by a
choice, and neither this doc nor the script said so. ▶ Both that and a crop_pct bug that hit
**only RSB-A3** are fixed; see `planning/resize_eval_reconciliation.md` §2, which also shows this
number is the one with an instrument behind it where blueprint §5.8's ~2.6 pt is not.
⭐ The `timm` arm's PIL equivalence is now MEASURED rather than assumed (flat at ~0.30/255 from
0.7× to 11.7× downscale), so the ruler below is a real one.
⚠ But the aliasing penalty is **not uniform in image size** — `antialias=False` tracks PIL at 0.31
with no downscale and 13.73 at 11.7× — so "0.2 pt" is an average over a skewed distribution, not a
per-image constant.

Every `tf.image.resize` in the emitter runs with the default `antialias=False`, on
both the train RandomResizedCrop and the eval centre-crop. timm/PIL antialiases both.
Confirmed absent in all six trainers (`grep -c antialias` = 0).

Measured on trained checkpoints, full 50k (`jax/scripts/eval_preproc_ab.py`):

| | ViT-Ti | ConvNeXt-T |
|---|---|---|
| ours (aliased both sides) | 65.642 | 78.144 |
| eval antialiased only | 65.398 | 77.918 |
| full timm protocol | 65.436 | 77.940 |

Two conclusions. The crop-then-resize vs resize-then-crop **ordering is worth ~0.02–0.04
pt** — noise, and it costs a full `decode_jpeg` instead of `decode_and_crop_jpeg`.
Antialiasing is the entire effect. And **fixing eval alone makes the number worse**,
because the model is trained under the aliased resampler; a genuine fix means
`antialias=True` on both sides plus a retrain. The matched-antialiased arm has never
been run, so its sign is unknown — the 0.2 measured here is a *mismatch* penalty, not a
prediction of what matched-AA would score.

Practical read: this is a shared ~0.2 pt term inside every gap in the scoreboard, and it
is the cheapest thing to *report* (run the A/B script per checkpoint) and among the most
expensive to *fix* (retrain everything).

### C2. BN momentum is 0.99 everywhere — correct for some nets, not others

The emitter hard-codes running-BN decay 0.99 (`_bn(..., momentum=0.99)`), compensated
for grad-accum where used (R50 K=4 → 0.997491, MNv4 K=8 → 0.998744). Whether that is
faithful is **per-net**, which the existing notes do not distinguish:

| net | reference | ours | verdict |
|---|---|---|---|
| EfficientNet-B0 | TF impl / paper: BN momentum 0.99 | 0.99 | ✅ correct |
| MobileNetV2 | TF-slim: 0.997 **[unverified]** | 0.99 | ~ minor |
| R50 (timm) | PyTorch default 0.1 → decay **0.9** | 0.99 | ❌ 10× slow |
| MobileNetV4 (timm) | PyTorch default → decay **0.9** | 0.99 | ❌ 10× slow |

Averaging window is 1/(1−decay): 100 steps at PyTorch's default vs 1000 here. Low impact
at the end of a cosine schedule (weights barely move, stats converge either way), but it
is a real difference during the phase where weights are moving, and it compounds with
the Ghost-BN item on R50.

### C3. Mixup/CutMix switching is deterministic

`if _global_step % 2 == 0: mixup else: cutmix`. timm draws `switch_prob=0.5` randomly per
batch. Same 50/50 expectation, zero variance. Affects R50, ConvNeXt-T, ViT-Ti, MNv4.
Minor. The pairing itself (`jnp.flip(x, 0)`) **is** timm's default `mode='batch'` — not a
deviation, despite looking like one.

### C4. LR schedule granularity — **not** a deviation

Checked because it is a classic mismatch: every trainer schedules per **optimizer step**
(`_global_step`), matching timm's `--sched-on-updates`. All six verified. No action.

---

## ResNet-50 — RSB-A3 (`rsb-faithful`, the 76.66% run)

Reference: timm `a3` — LAMB, lr 8e-3 @ bs2048, wd 0.02, 100ep, 5ep warmup, cosine
on updates, BCE, mixup 0.1 / cutmix 1.0, `rand-m6-mstd0.5-inc1`, no repeated-aug (`n0`),
sd 0.0, dropout 0.0, ls 0.0, train@160 / eval@224 crop 0.95. Paper 78.1; ours 76.66 (−1.44).

**Faithful [verified against timm/optim/lamb.py]:** betas (0.9, 0.999), eps 1e-6, bias
correction, `trust_clip=False`. The bias-correction algebra is *identical* — timm's
`m̂/(√v/√bc2 + ε)` is the same expression as the emitted `m̂/(√v̂ + ε)`. Weight decay is
added to the update *before* the trust-ratio norms, as in timm.

**Faithful (config-level):** per-step cosine, RRC area (0.08,1.0) aspect (3/4,4/3), no
random erasing, ls 0, sd 0, EMA off, repeated-aug off, crop_pct 0.95, 160/224 split.

### D1. Missing gradient clipping — **NEW [verified]**

timm's `Lamb.__init__` defaults `max_grad_norm=1.0` and clips the global grad norm
*inside the optimizer* every step. The emitted trainer has no clipping (`gradClipNorm`
unset → 0); `grep -c "gn = jnp.sqrt"` = 0.

**Likely modest**, for a specific reason: LAMB's update is approximately scale-invariant
to gradient scaling — Adam normalisation divides by √v, and the trust ratio renormalises
again — so a uniform rescale largely cancels. The residual is second-order: the clip
factor varies per step, so `m` and `v` accumulate differently-scaled gradients. Real, but
I would not expect it to carry the −1.44 alone.

### D2. Trust ratio applied to zero-weight-decay params — **NEW [verified], the best candidate**

timm guards layer adaptation with `if weight_decay != 0 or group['always_adapt']:`. The
`no_weight_decay` group (BN γ/β, biases) therefore gets a **plain Adam step, trust = 1**.
The emitted `_lamb` applies `trust = ‖p‖/‖r‖` to every parameter, including the ones
`WD_MASK` zeroes.

Direction is clear and one-sided. For zero-initialised 1-D params (BN β, biases) ‖p‖
starts tiny while ‖r‖ ≈ √C, so trust collapses to ~0.01–0.1 against timm's 1.0 — **BN β
and biases are under-trained by 1–2 orders of magnitude early in training**. BN γ is
unaffected (init 1.0 ⇒ ‖p‖≈‖r‖≈√C ⇒ trust ≈ 1).

Fix is a one-line guard: reuse `WD_MASK` to force `trust = 1` where the mask is 0, which
reproduces timm's group semantics exactly.

### D3. Known/documented (unchanged)

- **Ghost-BN** — each grad-accum micro-step normalises over its own 512, not the full
  2048, and running stats update K×/optimizer-step. `true-2048` would remove it but needs
  ~80 GB and is unrun.
- **bf16 matmul + conv** — kept for speed, probed ≈ −0.1 pt.
- **C2** BN momentum 0.99 vs timm's 0.9.

### D4. `short` conflates two problems

The plain `short` A3 recipe does **not** set `wdExcludeNormBias` — only `rsb-faithful`
does. So the 40.8% naive baseline is starved-LAMB *and* decaying BN γ/β + biases. Fine
for a deliberately-naive baseline, but it is not a clean one-variable control.

---

## ConvNeXt-T

Reference: AdamW lr 4e-3 @ bs4096 (→ 2.5e-4 @ 256), wd 0.05, 300ep, 20ep warmup, cosine,
ls 0.1, mixup 0.8 / cutmix 1.0, RandAugment (9, 0.5), random erasing 0.25, sd 0.1,
EMA 0.9999, LayerScale 1e-6. Paper 82.1; ours 81.10 (−1.0).

**Faithful:** architecture incl. LayerScale 1e-6 and the faithful patchify stem
(4×4 s4 conv → channel-LN, no BN/ReLU), AdamW + wd-mask, LR scaling, 20ep warmup, the
full aug pack, sd 0.1, EMA. No BN anywhere, so C2 is N/A.

### D5. No final LayerNorm between GAP and head — **NEW**

The paper has `GAP → LayerNorm → Linear`. The emitted forward is `global_avg_pool(x)` →
head directly; verified in the emitted Python. Carried into the new ConvNeXt-S/B specs
for consistency. Small but free to fix, and it is a genuine architecture difference
rather than a recipe one.

### D6. The ledger's attribution is now partly falsified

`paper_faithfulness.md` attributes the −1.0 to "EMA-eval/test-crop territory". C1 measures
test-crop at **0.2**, so ~0.8 of that −1.0 is **unattributed**, not explained. Worth
re-opening rather than leaving as-is.

Also: C3 (deterministic mixup switching), C1 (antialias).

---

## ViT-Ti / DeiT-Ti

Reference: AdamW lr 5e-4 × bs/512, wd 0.05, 300ep, 5ep warmup, cosine, ls 0.1,
mixup 0.8 / cutmix 1.0, `rand-m9-mstd0.5-inc1`, random erasing 0.25, sd 0.1,
repeated-aug 3×, EMA 0.99996. Paper 72.2; ours 70.28 (−1.9).

**Faithful:** architecture, optimizer, schedule, wd-mask (norm/bias/pos-embed/CLS), the
full DeiT aug pack, sd 0.1 with linear ramp, EMA 0.99996. Repeated augmentation **closed
2026-07-26** — it needed only `repeatedAug := 3`, since RSB-A2 had already built the
`flat_map` in the shared `build_imagenet_iter`.

### D7. Weight init — **NEW, largest remaining lever** (shipped as `deit-init`)

Transformer Linears (QKV, attn-out, MLP fc1/fc2, head) went through the generic
Xavier-uniform `emitDenseInit`; timm uses `trunc_normal_(std=0.02)`. Xavier scales as
1/√dim against a fixed 0.02, so the error is worst at the smallest model — **3.6× too
wide at Ti**, 2.6× at S, 1.8× at B. The patch-embed conv had the opposite problem, **6.5×
too narrow**, dividing by the output fan (dim·p·p) instead of the input fan (ic·p·p).
CLS token and pos-embed were already correct at 0.02, so this was an inconsistency inside
one file.

Measured on ViT-B — every group now lands on target under `vitInit := true`:

| | patch | cls | pos | QKV | head |
|---|---|---|---|---|---|
| default (Xavier) | 0.0032 | 0.0193 | 0.0199 | 0.0361 | 0.0336 |
| `deit-init` | **0.0208** | 0.0193 | 0.0199 | **0.0200** | **0.0200** |
| timm target | 0.0208 | 0.0200 | 0.0200 | 0.0200 | 0.0200 |

Init grad-norm (ViT-B, batch 32, pre-clip): Xavier **44.09** at loss 7.4637 vs timm
**14.28** at loss 7.1597 — 3.1× better conditioned, and much nearer ln(1000)=6.908.

### D8. grad-clip 1.0 is labelled "DeiT default" — **unverified**

The config comment claims `gradClipNorm := 1.0` is DeiT's default. Not checked against
DeiT's `main.py` from this box. It matters: if DeiT does not clip, the clip is itself a
deviation, and D7 suggests it may have been compensating for the init. Both grad norms
still exceed the threshold by 10×+, so D7 does **not** show the clip is removable — that
needs a clip-off arm.

Also: C3, C1. Still an approximation: tfds cannot do timm's index-level RASampler, so the
3 repeated-aug copies are spread by an 8192-element reshuffle.

---

## EfficientNet-B0 — the closest to its paper (−0.3)

Reference: RMSProp ρ 0.9 / μ 0.9 / **ε 1e-3**, lr 0.256 @ bs4096 (→ 0.016 @ 256), wd 1e-5,
exp-decay ×0.97 / 2.4 epochs, 5ep warmup, dropout 0.2, drop-connect 0.2, AutoAugment,
ls 0.1, EMA 0.9999, BN momentum 0.99, 350ep. Paper 77.1; ours 76.80 (−0.3).

**Faithful [verified in emitted code]:** RMSProp with **ε inside the sqrt**
(`g / jnp.sqrt(s + EPS)`, the TF form, not PyTorch's `sqrt(s) + eps`) and mean-square
initialised to **1.0** rather than 0. BN momentum 0.99 matches TF. Exp-decay schedule,
dropout 0.2, drop-connect 0.2 as a linear ramp over blocks (which is what the official
impl does), AutoAugment, ls 0.1, EMA with the BN buffers EMA-shadowed too (`ema_bn`),
running-BN eval.

This is the reference implementation in the repo — it is the existence proof that the
pipeline can reach paper numbers, which is the strongest evidence the residual gaps
elsewhere are per-net recipe details rather than anything systemic.

Remaining: C1 (antialias) is essentially the whole −0.3 budget.

---

## MobileNetV2

Reference: RMSProp ρ 0.9 / μ 0.9 / **ε 1.0**, lr 0.045, ×0.98 per epoch, wd 4e-5,
dropout 0.2, crop+flip only, no label smoothing, ~300–400ep. Paper 72.0; ours 68.77
at **90ep** (−3.23).

**Faithful:** optimizer form and constants (same TF-RMSProp fix as ENet), exp-decay 0.98/
epoch, wd 4e-5, dropout 0.2, ReLU6, crop/flip-only aug with no mixup/RA/erasing/EMA/sd
(all correctly absent — the paper uses none), running-BN eval, ls 0.

### D9. The gap is schedule, not fidelity

90 epochs against the paper's ~300–400. That is the whole story; the `full` recipe at
350ep exists and is unrun. Nothing here needs fixing before that run happens — the −3.23
is not evidence of a recipe defect.

### D10. BN momentum — see C2, `[unverified]`

TF-slim MobileNetV2 uses 0.997 (needs confirming); we use 0.99.

Also: C1. Footnote in the scoreboard says this number **pre-dates** the 2026-07-07
`labelSmoothing → 0` fix and was never re-measured.

---

## MobileNetV4-Conv-M

Reference: AdamW, lr 4e-3 @ bs4096, wd 0.1, dropout 0.2, RandAugment M=15, sd 0.075,
500ep, EMA. Ours: an explicitly **reduced-regularisation 100-epoch** tier — wd 0.05,
dropout 0.1, RA M=9 — with a `full` recipe restoring the paper values.

**Faithful:** effective batch 4096 via `gradAccumSteps := 8` (so the paper LR is used at
its design batch), AdamW + wd-mask, cosine + 5ep warmup, ls 0.1, EMA 0.9999 with
`ema_bn`, running-BN eval.

### D11. `default` is deliberately not paper-faithful

Every reduction is commented as such. This is a confidence run, not a reproduction — it
should not appear in a "measured vs paper" comparison without that caveat.

### D12. The `full` recipe's own caveats — one is **stale**

`full` carries two `NB:` warnings. Re-checked both:

- *"dropPath not yet wired into UIB — verify before trusting"* — **STALE, it is wired.**
  The emitted `uib_block` takes `drop_key`/`keep_prob` and applies inverted stochastic
  depth (`bernoulli(drop_key, keep_prob); x = x * keep / keep_prob`). Safe to delete the
  caveat.
- *"codegen clamps M to 0–10 — verify scale"* — **still open, and the description is
  wrong.** There is no clamp; the emitter computes `_M = randAugmentM / 10.0`, so the
  paper's M=15 yields `_M = 1.5`, i.e. magnitudes driven 1.5× past the top of the
  normalised range the op tables are built against. Needs checking before the `full` run,
  since it silently over-drives every geometric/colour op rather than clamping.

Also: C2 (BN momentum 0.99 vs timm 0.9), C3, C1.

---

## Priority

Ranked by (likely points) ÷ (cost to test):

1. **D2** — R50 LAMB trust ratio on zero-wd params. One-line guard, clear direction,
   shares an A3 arm (~10–11 hr) with D1.
2. **D7** — ViT init. Already shipped as `deit-init`, never run. An 80ep ViT-Ti A/B is
   ~6 hr per arm.
3. **D12** — MNv4 RandAugment M scale. Free to check, and it gates the 500ep `full` run.
4. **D5** — ConvNeXt final LayerNorm. Small emitter change; would need a re-run to value.
5. **D6** — re-open the ConvNeXt −1.0 attribution; ~0.8 pt is currently unexplained.
6. **C1** — report the timm-protocol number alongside each result (minutes per
   checkpoint). Actually *fixing* it means retraining the sweep; not obviously worth it
   at ~0.2 pt against gaps of 1–4 pt.
