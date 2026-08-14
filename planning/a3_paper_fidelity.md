# a3_paper_fidelity.md — every way the verified RSB-A3 run is NOT the paper, and what it would take

**Written 2026-08-06, while leg 2 is in flight.** Companion to `next_session_a3_the_run.md` (which
is now spent: its §1 probe is measured, its §2 conf exists and has run twice, its §5 steps 1–4 are
done or running). This file is the **fidelity ledger** — the thing to quote a final number against.

▶ **Read §0 before quoting any number. Read §5 before deciding what to fix.**

---

## 0. WHAT THE RUN IS

`resnet50in160_lambaccdp8x64bce`, 4 replicas × bs64 × k=8 = **effective batch 2048**, train@160 /
eval@224, LAMB × BCE-with-logits × gradient accumulation, 100-epoch cosine + 5-epoch warmup at
lr 0.008, wd 0.02.

| | |
|---|---|
| leg 1 | epochs 0–29, 2026-08-06 03:47→14:07, **10h20m**, 1 attempt, 0 cooldowns, 0 crashes |
| leg 2 | epochs 30–99, started 14:17, ~24 h |
| measured | 240 ms/step real, 222 synth, ~54 s/epoch eval ⇒ **20.6 min/epoch** |
| ep25 | **50.49% / 75.40%** vs JAX's 40.02% and 37.98% at identical lr 0.007157 |

⭐ The LR schedule is **verified identical** to the reference: a 100-epoch cosine with 5-epoch
warmup predicts 0.007157 at ep25, JAX's ep25 milestone logged `lr=0.007157`, and every epoch since
has matched the closed form to six decimals. The accuracy difference is not a schedule artifact.

---

## 1. ✅ WHAT **IS** FAITHFUL — verified 2026-08-06, not assumed

This half of the ledger matters as much as the other. Each was checked against the reference
config or the generated shim, not against the recipe matrix.

| ingredient | value | evidence |
|---|---|---|
| optimizer | LAMB | `r50-lamb-tie` |
| loss | BCE-with-logits | `r50-bce-tie` |
| effective batch | 2048 | 4×64×k8, announced per run |
| base LR | **0.008** @ bs2048 | `jax/MainResnet50Imagenet.lean:125`; ⚠ the driver's default is 0.001 — see §2.6 |
| weight decay | 0.02 | (the VALUE matches; the EXCLUSION does not — §2.1) |
| schedule | cosine, 5-epoch warmup, 100 ep | matched to 6 dp against JAX ep25 |
| mixup / cutmix | α **0.1** / α **1.0**, both on | shim defaults 323–324; ref `:58–61` |
| RandAugment | n=2, **m=6.0**, mstd=0.5, **inc1** | shim `_randaugment(img,2,6.0,0.5)`, `_RA_INC=True` |
| interpolation | BICUBIC | shim 236, 250 |
| test crop ratio | **0.95** (RSB-A3) | shim 244 |
| resolution split | train@160 / eval@224 | `EVAL RES SPLIT` banner, d0 76800 / 150528 |
| BN eps | 1e-5 | render arg `"1.0e-05"` |
| epochs | 100 | `resnet50ImagenetConfig.epochs`, raised 30→100 on 2026-08-06 |

---

## 2. ⛔ THE DELTAS

Ordered by expected impact. **Quote a final number against this list, never as "RSB-A3
reproduced"** — `ResNet50RenderB.lean:990` says the same thing at the render site.

### 2.1 ✅ CLOSED 2026-08-14 — `wdExcludeNormBias` (was ⛔⛔, the largest delta)

**What:** timm's `no_weight_decay` skip-list excludes all 1-D params (BN γ, BN β, every bias) from
decay, decaying only ≥2-D weight matrices. We decay everything, at wd = 0.02.

**Evidence:** the live artifact has **zero** `%wdz` occurrences. `ResNet50RenderB.lean:990` states
it. The reference has `wdExcludeNormBias := true` (`jax/MainResnet50Imagenet.lean:127`).

**Why it should matter:** BN γ directly scales each layer's output, so decaying it shrinks the
network's realised function. Decay on *pre-BN conv weights* is renormalised away by BN and acts
only as an effective-LR control — decay on γ/β does not. The effect concentrates at low LR, i.e.
exactly the endgame leg 2 is entering.

⭐ **The mechanism already exists on the verified path — just not for R50.** ConvNeXt and ViT both
implement it by emitting a second decay constant `%wdz` (zero) and routing excluded params to it:
*"121 of 180 params take `%wdz`, not `%wd`"* (`ConvNeXtRender.lean:746`, `ViTRender.lean:678`).

**How to get there:** port that two-constant pattern into `ResNet50RenderB`. The R50 parameter
layout is already fully enumerated (161 θ), and the predicate is purely structural — 1-D ⇒ `%wdz`.
⚠ Read `ConvNeXtRender.lean:1092` first: gating this flag is what surfaced a
"silently-wrong-hyperparameter shape" there, so the port has a known trap.
**Cost:** moderate. Render + re-gate + a tie against the 1-replica peer.

✅ **DONE 2026-08-14.** `optOne` took a trailing `wdName` (defaulted to `"%wd"`, so every committed
R34/R50 artifact re-renders byte-identically), `r34WdDecays` is the plain rank test, `wdzConst`
emits the zero, and `r34AdamVariant` gained a trailing `wx`. Two new artifacts:
`resnet50in160_lambaccdp8x64wxbce` (4 replicas) and `lambacc8x64wxbce` (its 1-replica gate peer).
⭐ **The split is 54 decayed / 107 excluded of 161**, and both halves decompose structurally: 53
conv weights + 1 dense weight decayed; 53 BN layers × 2 (γ, β) + 1 dense bias excluded.
⚠ The `wx` renders are NEW artifacts beside the old ones, not replacements — the 77.43% result
belongs to the graph that produced it, and re-pointing that slug would make a finished 34-hour run
unreproducible. ⚠ NOT YET RUN: this changes the trajectory, so it is comparable to A3 only by a
fresh run.

### 2.2 ⛔ Ghost-BN normalises over **64**; the reference normalises over **512**

**What:** an 8× difference in the BN statistics group.

**Evidence, both sides.** Ours: all 162 `all_reduce` ops in the artifact are on *parameter*-shaped
tensors (`64x3x7x7` etc.) and the `%armean` blocks are `divide %arsums, %arns` — gradient
averaging. **No collective touches activation statistics**, so each replica normalises over its own
64. Reference: `bm = jnp.mean(x, axis=(0,2,3))` reduces over axis 0, the batch axis, which is
mesh-sharded (`NamedSharding(mesh, P(None,'batch'))`, generated script 1310) — so XLA inserts the
all-reduce and the group is the **full 512 micro-batch**. Corroborated by the config's own words,
*"Ghost-BN, Hoffer et al. 2017 — benign at micro=512"* (`MainResnet50Imagenet.lean:117`).

**Sign of the effect is genuinely unknown.** Hoffer et al. is usually cited for smaller ghost
batches *improving* generalization at large effective batch, which would make our 64 a benefit —
but 64 is far below what the reference tuned at, and it is one of the few live hypotheses for our
ep25 lead.

**How to get there**, cheapest first:
1. **All-reduce the batch statistics across replicas** → group 256. A render change only, no memory
   cost. ⚠ But it adds a collective per BN layer per step — 53 of them — and this run's whole
   throughput story is that the allreduce term is 56 ms. Measure before adopting.
2. **bs128/replica with k=4** → group 128, preserving effective batch 2048. Needs a new render
   (`lambaccdp4x128bce`); the batch is baked as `%x: tensor<64x76800xf32>`. ~5–7 GB/device, fits.
3. **Both** → 4×128 synced = **512, the exact match.**

⚠ A true single-forward bs2048 is not reachable here and that is confirmed on the reference side,
not estimated: *"needs a big-memory card (~80 GB; a 48 GB card is borderline/OOM)"*
(`MainResnet50Imagenet.lean:131`).

### 2.3 ✅ CLOSED 2026-08-14 — BN running-stat momentum vs accumulation (was ⛔)

**What:** we apply the BN running-stat EMA once per *micro-batch* at m = 0.01 (momentum 0.99), with
k = 8 micro-batches per optimizer step. So per optimizer step our stats decay by 0.99⁸ ≈ **0.923**
where the reference's decay by 0.99.

**Evidence:** ours `F32.ema runningBnStats batchBn 0.01` (`VerifiedTrain.lean:1574`). The reference
compensates explicitly: `momentum=0.997491`, with the generated script's own comment — *"BN
momentum compensated for gradient accumulation (K=4): per-micro momentum = 0.99**(1/K) → K updates
compose to ~one 0.99/step update"*.

**Effect: EVAL-ONLY.** Running stats never touch a gradient; the train forward uses batch stats. Our
running estimates are ~8× fresher and correspondingly noisier per optimizer step.

⚠ **This is the same shape as a bug already paid for once.** `VerifiedTrain.lean:1567` records the
2026-08-04 correction of this very constant from 0.1 to 0.01, noting it *"depressed every reported
top-1 without touching a single gradient, and it bit hardest early"*. The accumulation
compensation is the second half of that fix and was not made.

⭐ **Cheapest fix in this document — a one-line driver change, no re-render, no re-gate:**
`(if bnFirst then 1.0 else 1.0 - Float.pow 0.99 (1.0 / accK.toFloat))` — at k=8 that is m ≈
0.001256. ⚠ Do **not** apply mid-run: it changes reported eval, so leg 2's curve would develop a
discontinuity at the point of change.

✅ **DONE 2026-08-14**, exactly as prescribed: `bnMom := if accOn then 1 − 0.99^(1/k) else 0.01`.
At k = 1 that is exactly 0.01, so every non-accumulating run is bit-identical across the change —
which is why the guard is `accOn` rather than a version. ⚠ The fp8 peer (`trainAdamSchedE4M3`)
keeps a bare 0.01 and that is correct, not an oversight: it implements no accumulation, so k = 1.

▶ **Note the direction.** This delta plausibly makes our reported top-1 *understated*, which is
worth stating explicitly given that we are currently ahead of the reference.

### 2.4 fp32, not bf16

The reference ran bf16 matmul+conv. Ours is fp32 throughout. Numerics aside, this is most of the
throughput gap: we sustain ~1,067 img/s against the reference's ~2,133 (600 s/epoch over 1.28 M
images) — **2×**, from bf16 plus the larger micro-batch. Relevant to any future 34 h run, and it is
the lever if a budget ever gets tight.

### 2.5 The 0.08% dropped tail

`G2_STEPS=5000` against ImageNet's 5,004 micro-batches/epoch, because 5004 = 2²·3²·139 is not
divisible by k=8. Drops 4 micro-batches = **1,024 images/epoch**. Real, tiny, and unavoidable
without changing k (which would change the effective batch — see the conf's note).

### 2.6 ⚠ Not a delta, but the near-miss worth recording

The driver's `baseLR` defaults to **0.001** — AdamW's rate, correct for this net's *other*
artifacts, and 8× below A3's. Nothing gated it; it descends and reports a number either way. Found
2026-08-06 and now carried by the conf with a `PRECHECK` refusal. Same family as §2.3 and the
2026-08-04 momentum bug: **a hyperparameter with no output and no gate.**

### 2.7 Minor / accepted

* **mixup λ is drawn from numpy's `Generator`, not `jax.random`** — agreement with the reference is
  distributional, never per-step. Announced by the run. Not fixable without reimplementing the RNG.
* **eval denominators differ**: ours 49,920 (195×256), the reference's 49,664 (97×512), against
  ImageNet's 50,000. Both drop a tail; ours drops less. ≤0.7%, but it means the two top-1 figures
  are over slightly different denominators.
  ✅ **OURS CLOSED 2026-08-14 — the verified path now scores all 50,000**, which is timm's
  denominator. The shim batches the VAL split with `drop_remainder=training` (train keeps `True`;
  its batch is baked into the graph), the drain terminates on the closed pipe instead of a
  hardcoded `nB := 195`, and `readShimBatchPartial` accepts the short final batch that creates.
  ⭐ **No MLIR changed**: `F32.sliceImagesPad` already zero-padded a short tail to the eval graph's
  baked width and the loop already scored real rows only, so the forward still sees a full batch.
  Safe because eval normalises per example everywhere (running-stat BN, or LayerNorm).
  ⚠ `LEAN_MLIR_EVAL_BATCHSTATS=1` is the exception — it scores through `@_fwd` with BATCH stats,
  where the zero pad WOULD shift the real rows — so the drain drops the tail under that flag and
  announces the denominator it used.
  ⚠⚠ **Every ImageNet top-1 quoted before this date, including the 77.43%, is over 49,920.** The
  drain now announces which denominator it used, because nothing else in the output did.

---

## 3. VERIFICATION DEBT (distinct from recipe fidelity)

### 3.1 ⛔⛔ `r50-gradcheck` tier 1 B does not survive BCE
Pre-existing, localised by a 2×2 to the **loss alone**: ‖g‖ is 197.5 under CE and 2.18 under BCE,
and tier 1 B measures a cancellation residual, so it degrades against a 90× smaller gradient. The
gate genuinely *cannot decide* at BCE — its passing sites (0.000757) sit above the weakest
deliberate control violation (0.000545), so no tolerance separates the populations.
▶ **Do NOT raise `R50_GC_EXACT_U`.** Either build a conditioning-robust tier 1 B (normalise per-site
by ‖g_γ‖·‖γ‖, or run the identity at a scaled loss), or record explicitly that tier 1 B is CE-only.
The composition passes tier 1 A (0.000017) and tier 2 (rel ≤ 0.230 vs 0.30, control live at 0.997).
⚠ Also affects `adam64bce` and `lamb64bce`.

### 3.2 `r50_dp_render_tie.py` — the composed pair is not wired in
`lambacc8x64bce` / `lambaccdp8x64bce` exist and are not in the (1-replica, 4-replica) tie.

### 3.3 The `bce` axis is still a loose `Bool`
`bce` is a trailing `Bool` on `resnet50TrainStepFaithfulB` while `lamb`/`accum` are constructors of
`R34Opt`, and the variant string is `r34AdamVariant` **plus** a hand-passed `vSuffix := "bce"` — so
the name and the flag can disagree with nothing noticing, and the artifact this run depends on ends
in `bce`. Mitigated by `#guard`s on the produced names, not closed.

---

## 4. HOW TO QUOTE THE RESULT

▶ **"RSB-A3 recipe, verified path, with the §2 deltas"** — never "RSB-A3 reproduced".

The reference on this same box: **76.66% / 93.03%** (Jul-4) and **77.22% / 93.34%** (Jul-9→10
rerun). Paper RSB-A3 is 78.1%.

⚠ **Both reference runs back-load hard** — Jul-9→10 went 37.98% @ep25 → 55.81% @ep50 → 71.62% @ep75
→ 77.22% @ep100, gaining **20 points in its final quarter** as the cosine annealed. An ep25 or ep50
lead therefore predicts very little about ep100, and §2.1 is a delta whose cost is expected
precisely in that endgame. Do not extrapolate leg 2 from its midpoint.

⚠ Only ep25/50/75/100 milestones survive for the reference's early curve, plus a single ep6 point
(11.17% / 27.03%) from the Jul-4 run — its supervisor truncated the runlog on every resume, so
per-epoch data exists only for epochs 76–100 (`epochs_final_attempt_76-100.tsv`).

---

## 4b. ✅ GEOMETRIC AUGMENTATION vs timm — MEASURED 2026-08-14, and it corrected the worry

The standing concern was that our RandAugment geometry, written in TensorFlow
(`tf.raw_ops.ImageProjectiveTransformV3`), used a different angle/matrix convention from timm's,
which calls PIL. timm's geometric ops are thin PIL wrappers, so PIL **is** the reference here.
`scripts/geo_aug_pil_diff.py` diffs them:

| op | mean \|Δ\| of 255 (smooth image) |
|---|---|
| ShearX / ShearY | 0.20 – 0.48 |
| TranslateX | 0.04 – 0.19 |
| Rotate (±30°, ±21°, 7.5°) | 0.07 – 0.20 |

**The geometry matches.** Maxima land on a deliberately-planted hard edge, i.e. sub-pixel
resampling, not a different transform. Our rotate agrees with PIL's DEFAULT centre; forcing PIL to
the `((w−1)/2, (h−1)/2)` centre our formula nominally names makes it *much worse*, so TF's and
PIL's pixel-coordinate conventions differ by half a pixel and our offset already compensates.

⚠⚠ **THE METHOD IS THE FINDING, because the first run of this test said the opposite.** On RANDOM
NOISE the same comparison reports 85–93% of pixels differing at mean \|Δ\| ≈ 10 — which reads as
broken geometry and is not. On uncorrelated noise a half-pixel resampling difference makes every
pixel disagree by an arbitrary amount, so noise cannot separate "wrong transform" from "same
transform, different resampler". ▶ **Never validate a resampler on noise.** A smooth image with one
hard edge separates them: a wrong transform shows large INTERIOR error, a resampler shows error
only where the gradient is steep.

### ✅ …and then timm was installed (1.0.28), which found a REAL one

`scripts/randaug_timm_diff.py` evaluates our `_aa_*` magnitude mappings — read out of the generated
shim by regex, not restated — against timm's own `level_to_arg` functions at every integer
magnitude. **Two of six were wrong, in the same way, since they were written:**

| | timm @ m=7 | ours (was) | |
|---|---|---|---|
| **Posterize** | keep **2** MSBs | keep **1** | ⛔ a whole bit, at RSB's own magnitude |
| **Solarize** | 77 | 76 | 1 threshold unit of 256 — invisible |

⭐ **One root cause: which side of the subtraction the truncation happens on.** timm builds each
`inc` mapping by negating the already-INTEGER decreasing one — `4 - int((m/10)·4)` — where we wrote
`int(4 − (m/10)·4)`. That lands one step lower at **7 of the 11** integer magnitudes. Fixed
2026-08-14; the gate has an anti-vacuity control (it fires on the pre-fix shim, 8/11 magnitudes).

⚠⚠ **Nothing could have caught this.** `shim_wiring_gate.py` compares the shim to the reference by
augmentation PARTITION — *which* ops are on — and never by the ARGUMENT each op is called with. The
partition was right the whole time. ▶ A gate on "is the feature enabled" is not a gate on "is the
feature correct", and this repo now has one of each.

### ✅ …and the interpolation, MEASURED 2026-08-14 — the answer is KEEP BILINEAR

The plan was to flip the geometric ops to BICUBIC and eat a full re-run. **Measured, that makes
fidelity ~60× worse.** Rotate 21°, our shim's exact call including its clip:

| image | TF-BILINEAR → PIL-BICUBIC | TF-BICUBIC → PIL-BICUBIC | PIL-BILINEAR → PIL-BICUBIC |
|---|---|---|---|
| pure smooth | **0.56** | 32.15 | *0.39* |
| smooth + noise | **2.53** | 29.47 | *2.41* |

⭐⭐ **The right-hand column is the whole argument.** It is the GENUINE bilinear-vs-bicubic gap —
PIL against itself, the most that switching interpolation could ever be worth. It is 0.39–2.41 of
255. **Our TF-BILINEAR is already sitting at that floor** (0.56 / 2.53), i.e. as close to timm's
bicubic as a correct bilinear implementation can get. There is nothing left to win.

⚠⚠ And TF's BICUBIC is not the thing to win it with. `ImageProjectiveTransformV3` *accepts* a
`BICUBIC` attr and genuinely computes something different — but that something agrees with neither
PIL kernel, sitting ~29–32 from both. It is a third resampler. Asking for it would have moved us
from 0.56 to 32.

▶ **The general shape, and it is worth carrying:** "our X is bilinear, theirs is bicubic" sounds
like a fidelity gap and is really a bound on one. Before paying for a change, measure what the
change could possibly buy — here, the reference's own two kernels differ by less than our current
error bar, so the gap was never the thing to fix. The two magnitude mappings above, which cost
nothing to check, were worth 7 of 11 magnitudes.

⚠ Still true and still not matched: we are BILINEAR where timm is BICUBIC. That is a real
difference of ~0.4–2.4/255 on the geometric ops only, and it is now a MEASURED, BOUNDED one rather
than an open question. Closing it properly would need a hand-written Catmull-Rom affine sampler in
TF, not a flag.

### (superseded) The interpolation as a third difference We always use BILINEAR for the
geometric ops. timm resolves the resample mode from the MODEL's data config:
`resolve_data_config` for resnet50 returns `interpolation='bicubic'`, and `transforms_factory` then
sets `aa_params['interpolation'] = str_to_pil_interp('bicubic')` — so **timm's geometric
RandAugment runs BICUBIC**. (Only with interpolation unset or `'random'` does it use
`_RANDOM_INTERPOLATION`, a per-call BILINEAR/BICUBIC choice — also not what we do.) The shim's own
comment claimed BILINEAR *was* timm's default; that comment was wrong and is corrected.
▶ **Not changed**, because unlike the two above it is a behaviour change for every net rather than
a transcription bug — it needs a decision and ideally a probe, not a patch.
⚠ resnet50's resolved `crop_pct` is **0.95**, which is what we already use — checked at the same
time, so the eval crop is confirmed rather than assumed.
⚠ Also uncovered by this test: op ORDER and per-op probability, the `_RA_INC` magnitude mappings,
and every photometric op. This is about geometry alone.

## 5. RECOMMENDED ORDER

1. ⭐ **§2.3, the BN momentum compensation.** One line, no re-render, no re-gate, and it is
   eval-only so it cannot corrupt a trajectory. Highest value per unit risk in this document.
   *Do it after leg 2 finishes, then re-evaluate the final checkpoint to quantify it.*
2. ⭐ **§2.1, `wdExcludeNormBias`.** The largest recipe delta, with a working two-renderer
   precedent to copy. This is the one most likely to move the final number.
3. **§3.1, the gradcheck gap.** Blocks nothing, but it is the honest verification debt and it is
   what the artifact's claim ceiling rests on.
4. **§2.2, the BN group.** Measure first (an OOM ladder at 128/256/512 per replica on the synth
   path, minutes). Prefer bs128/k4 over a batch-stat collective until the collective's cost is
   measured against the 56 ms allreduce term.
5. **§2.4, bf16.** Chase only if throughput becomes the constraint; it is a 2× lever, not a
   fidelity one, and it changes numerics.
6. §3.2 / §3.3 — hygiene, any time.

⚠ **Nothing in §2 may be applied to the run in flight.** Every one of them changes either the
trajectory or the reported eval, and a mid-cosine change makes the whole run unquotable.

---

## 6. ✅ RESULT — 2026-08-07 14:01, 100/100 epochs

# **77.43% top-1 / 93.60% top-5**

against the JAX reference's **77.22 / 93.34** (Jul-9→10 rerun) and **76.66 / 93.03** (Jul-4), and
paper RSB-A3's **78.1%**. Peak was ep99 at 77.49; ep100 is the quotable figure.

⚠ **This is the RSB-A3 recipe on the verified path WITH the §2 deltas — not "RSB-A3 reproduced."**
It beats this box's own JAX reference by +0.21 top-1 / +0.26 top-5 while *missing*
`wdExcludeNormBias` (§2.1) and running an 8×-smaller BN group (§2.2), and §2.3's uncompensated BN
momentum plausibly makes the eval slightly **understated**. That the verified path came out ahead
anyway is the interesting result, and it is not explained by anything in this document.

**Shape of the race** — we led early and the reference closed almost all of it during the anneal,
then the gap stabilised:

| | ep25 | ep50 | ep75 | ep79 | ep90 | ep100 |
|---|---|---|---|---|---|---|
| Δ top-1 vs rerun | **+10.5** | **+5.5** | +1.66 | +0.55 | +0.29 | **+0.21** |

**Run quality:** leg1 10.33 h + leg2 23.73 h = **34.06 h compute** (34.22 h end-to-end), **one
attempt per leg, zero thermal cooldowns, zero crashes, zero AER events**, peak ~53 °C against a
78 °C trip. Against the 40 h bar with ~6 h to spare. The reference took ~16h55m for the same 100
epochs — the 2× is bf16 plus the larger micro-batch (§2.4).



| epoch | ours top-1 | ours top-5 | JAX Jul-4 | JAX rerun |
|---|---|---|---|---|
| 5 | 21.73 | 45.44 | — | — |
| 6 | — | — | 11.17 | — |
| 10 | 33.17 | 59.14 | — | — |
| 15 | 45.36 | 71.52 | — | — |
| 20 | 47.98 | 73.18 | — | — |
| 25 | **50.49** | **75.40** | **40.02** | **37.98** |
| 30 | 50.67 | 76.34 | — | — |
| 31 | 53.09 | 78.06 | — | — |
| 40 | 55.56 | 79.62 | — | — |
| 50 | **62.10** | **84.33** | **56.60** | **55.81** |
| 60 | 66.97 | 87.59 | — | — |
| 70 | 71.85 | 90.61 | — | — |
| 75 | **73.28** | **91.65** | **71.81** | **71.62** |
| 80 | 74.99 | 92.35 | — | 74.30 / 91.91 |
| 90 | 76.97 | 93.45 | — | 76.68 / 93.07 |
| 95 | 77.41 | 93.50 | — | 77.15 / 93.31 |
| 98 | 77.41 | 93.57 | — | 77.16 / 93.32 |
| 99 | 77.49 | 93.61 | — | 77.21 / 93.33 |
| **100** | **77.43** | **93.60** | **76.66 / 93.03** | **77.22 / 93.34** |

**Wall clock:** leg 1 10h20m + leg 2 ~24 h ≈ **34 h**, against a 40 h bar, with zero thermal
cooldowns observed across leg 1's 10 hours (peak ~53 °C vs the 78 °C trip).
