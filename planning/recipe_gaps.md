# recipe_gaps.md — closing the distance between the verified trainers and the JAX references

**Written 2026-08-02, at the end of the session that scaffolded all five ImageNet trainers
(handoff §2p).** Goal of **v1: catch JAX.** Not the papers — the JAX references in this repo, which
are themselves short of the papers in places and whose numbers we actually have.

Every claim here is from the configs and run results in `jax/`, or from measurements taken on ares
(6 × RTX 4060 Ti, 4 used) on 2026-08-01/02. Where something is a projection it says so.

---

## 0. ▶ THE HEADLINE: ResNet-34 is ALREADY AT FEATURE PARITY. v1 for R34 is a RUN, not a build.

This was not obvious and it changes the plan. `jax/MainResnetImagenet.lean`'s recipe is:

| reference | verified path | status |
|---|---|---|
| SGD + momentum 0.9, **coupled** L2 wd 1e-4 | heavy-ball via `momVNextF (μ := wd, v := θ)` then `sgdParamF`; emitted constant is `dense<0.000100>` | ✅ **exact** |
| cosine decay + 5-epoch warmup | `trainAdamSched`'s cosine + warmup | ✅ |
| label smoothing 0.1 | `alphaOverK nClasses` — derived, `-0.000100` at K=1000 | ✅ |
| batch 256 | 4 × 64 = global 256, 5004 steps/epoch | ✅ **exact** |
| 90 epochs | `VerifiedConfig.epochs` | ✅ |
| **bf16 / bf16Conv** | fp32 | ❌ **the only gap** |

§2k built the heavy-ball path *specifically* to match this reference and gated it
(`r34-mom-tie`: `v' = g + wd·θ` at 7.3e-8, with the Nesterov control missing by 515,403×). The
work is done. **bf16 is a throughput lever, not an accuracy one** — bf16 with fp32 accumulate is
normally accuracy-neutral, so an fp32 run should be comparable to the reference's **72.02%**
directly, just slower.

**So the first thing to do on this whole document is start the R34/ImageNet run.** It is the only
one of the five where "catch JAX" requires zero new code, and it is the pair §2k built the shim for.

---

## 1. The targets — what JAX actually got

From `jax/runs/*/RESULTS.md`. All bf16, all on this box's GPUs.

| net | JAX top-1 | epochs | notes |
|---|---|---|---|
| **ResNet-34** | **72.02%** | 90 | the cleanest pair — see §0 |
| **EfficientNet-B0** | **72.31%** | 80 | RMSProp + exp decay + EMA |
| **MobileNetV2** | **68.33%** | 90 | RMSProp + exp decay, no label smoothing |
| **ConvNeXt-T** | **75.93%** | 80 | **EMA-evaluated**; live best was 76.28% at ep48 |
| **ViT-Tiny** | **65.64%** | 80 | the 300-epoch DeiT-Ti config targets ~72% and has not been run to completion |

⚠ **ConvNeXt's number is the EMA shadow's**, not the raw weights'. So EMA is not a nicety there —
it is part of how that number was produced, and a verified run without it is not comparing like
with like.

⚠ **ViT's 65.64% is the 80-epoch tier.** Do not compare a verified ViT run against DeiT-Ti's 72.0%;
compare it against 65.64% at matched epochs, or run 300 and compare against a JAX 300-epoch run
that does not exist yet.

---

## 2. The gap matrix

✅ = the verified path has it · ❌ = missing · — = the reference does not use it either

| feature | R34 | ViT | ConvNeXt | EfficientNet | mnv2 | verified path |
|---|---|---|---|---|---|---|
| optimizer | SGD+mom | AdamW | AdamW | **RMSProp** | **RMSProp** | AdamW / SGD / Nesterov / heavy-ball / **RMSProp** |
| | ✅ | ✅ | ✅ | ✅ **render** | ✅ **render** | ⚠ the two RMSProp nets still owe the DRIVER — see §3 Tier C |
| weight decay | 1e-4 coupled ✅ | 0.05 ✅ | 0.05 ✅ | 1e-5 ❌ | 4e-5 ❌ | decoupled in AdamW, coupled in heavy-ball |
| `wdExcludeNormBias` | — | ✅→❌ | ✅→❌ | — | — | ❌ |
| LR schedule | cosine ✅ | cosine ✅ | cosine ✅ | **exp 0.97** ❌ | **exp 0.98** ❌ | cosine + warmup only |
| warmup | 5 ✅ | 5 ✅ | 20 ✅ | 5 ✅ | 5 ✅ | driver arg |
| label smoothing | 0.1 ✅ | 0.1 ✅ | 0.1 ✅ | 0.1 ✅ | 0.0 ✅ | derived from `nClasses` |
| grad clip | — | 1.0 ❌ | 1.0 ❌ | — | — | ❌ |
| **mixup / cutmix** | — | 0.8/1.0 ❌ | 0.8/1.0 ❌ | — | — | wire ✅, **producer ❌** |
| RandAugment (geo) | — | ✅ | ✅ | — | — | ✅ **free via shim** |
| random erasing | — | ✅ | ✅ | — | — | ✅ **free via shim** |
| repeated aug | — | 3 ✅ | — | — | — | ✅ **free via shim** |
| stochastic depth | — | 0.1 ❌ | 0.1 ❌ | 0.2 ❌ | — | ❌ |
| EMA | — | ✅→❌ | ✅→❌ | ✅→❌ | — | ❌ |
| **bf16 / bf16Conv** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| batch / epochs | ✅ | ✅ | ✅ | ✅ | ✅ | all matched at 4 replicas |

**Distance to parity, per net:**

| net | gaps | verdict |
|---|---|---|
| **R34** | bf16 | **at parity — run it** |
| **mnv2** | ~~RMSProp~~ ✅, **ms-init 1.0 + exp decay**, bf16 | **two driver lines away** (was "one optimizer") |
| **EfficientNet** | ~~RMSProp~~ ✅, **ms-init 1.0 + exp decay**, dropPath, EMA, bf16 | the same two driver lines, + two more |
| **ConvNeXt** | mixup, cutmix, dropPath, EMA, grad clip, wdExclude, bf16 | the aug/regulariser pack |
| **ViT** | same as ConvNeXt | same |

⚠ **"RMSProp" split into a render half and a driver half, and only the render half is done**
(2026-08-02). The renders are certified — see §3 Tier D — but a driver that initialises the
mean-square slot to **0** instead of **1.0** runs a *different and much larger first step*, and
this optimizer is not bias-corrected, so there is nothing to absorb it. **Do not read the ✅ in the
matrix above as "mnv2 can be run against 68.33% today."**

---

## 3. Where each gap LIVES — this is what determines cost

The useful axis is not "how important" but "which layer does it touch", because the layers have
wildly different costs in this repo.

### Tier A — the pipeline. **Already free, already flowing.**
RandAugment (full geometric, m9/mstd0.5/inc1), random erasing, repeated augmentation, RRC + hflip.

These live inside `build_imagenet_iter`, which `generateShim` reuses **verbatim** (§2k). Turning
them on in the JAX config moves both paths at once, and the verified side owns no augmentation code
at all. **Nothing to build. Verify by flipping the config and re-running `SHIM_HASH`.**

### Tier B — the producer (shim-side Python). **Wire is done; the mixing is not.**
**mixup, cutmix.**

Two things already landed that make this the next-cheapest real item:
* **wire v2** carries `float32[batch·nClasses]` soft targets, gated bit-identical against v1
  (commit `4755317`);
* **the renders are AFFINE in `%onehot`** — measured, `lake build soft-target-tie`, ViT 492× /
  ConvNeXt 309× separation — so a mixed target yields the mixed gradient **with no render change**.

What is left is filling those slots. ⚠ **This is the one place where a second definition of
something is unavoidable**: the JAX reference applies `_mixup`/`_cutmix` in the *train step* with
`jax.random`, not in `tf.data`, so a shim-side implementation is a genuinely new copy of the mixing
rule. It is much smaller and more self-contained than the augmentation pipeline (a Beta draw, a
convex combination, a box), but it is a second writer and the doc should say so rather than pretend
otherwise. The alternative — no mixup on the verified path — is worse.

### Tier C — the driver (Lean, no render change).
**EMA, exponential LR decay.**

* **exponential LR decay** is nearly free: `lrt` is computed host-side in `VerifiedTrain.lean:815`
  and passed in as `%lr`. Add a schedule kind; ~10 lines. Unblocks the *schedule* half of
  EfficientNet and mnv2.
* ⚠ **the RMSProp mean-square init (1.0, not 0)** — added here 2026-08-02 when the render landed.
  It is ~one line in the driver's optimizer-state setup, and it is a **correctness** item, not a
  tuning one: TF's RMSProp is not bias-corrected, so `s = 0` makes the first step
  `g/√((1−ρ)g²+ε)` instead of `g/√(ρ + (1−ρ)g²+ε)` — much larger, and nothing downstream absorbs
  it. Together with exponential decay this is ALL that stands between mnv2 and its 68.33%.
* **EMA** needs a shadow buffer and an update per step, plus eval reading the shadow. Also ~a day.
  ⚠ There is a known EMA warmup failure mode recorded in this project's memory — a shadow
  initialised at random init and evaluated too early scores at chance. Whatever lands must gate
  that: EMA accuracy must **track then exceed** the raw weights, never start at 0.1%.

### Tier D — the render (a new proven `SHlo` op family; §4's "ten sites" each).
**RMSProp, `wdExcludeNormBias`.**

* ~~**RMSProp** is the single highest-value render item~~ ✅ **DONE 2026-08-02** — and **this
  section's cost estimate was wrong by a factor of the whole tier.** It is filed here as "a new
  proven `SHlo` op family, §4's ten sites each". Measured, it is **ONE op**, because three of the
  four steps are already-certified ops given their RMSProp reading — the same discovery §2k made
  for heavy-ball, one optimizer over:

  | reference | verified path |
  |---|---|
  | `grads = g + WD*p` | `momVNextF` at `(μ := wd, v := θ)` — `momVNext_as_coupled_l2` |
  | `sq = RHO*s + (1-RHO)*g*g` | **`adamVNextF` at `β₂ := ρ` IS this** — `rmsSqNext_eq_adamVNext`, by `rfl` |
  | `buf = MOMENTUM*b + g/sqrt(sq+EPS)` | ⛔ `rmsBufNextF` — the only new op |
  | `params = p - lr*buf` | `sgdParamF` |

  `[θ|m|v]` is reused with `m` = buffer, `v` = mean-square, so the **signature is byte-identical to
  each net's AdamW render apart from the entry name** and no driver or interface change is implied.
  Certified on both nets by `rms-tie` (①②③ ≤ 1.1e-6) with the textbook-ε control missing by
  **365,412× (mnv2) / 774,497× (enet)**. Six artifacts; gate 1 held at 0 diff lines.

  ⚠ **The ε placement is not cosmetic and the two nets sit on opposite sides of it.** The reference
  is **TF-flavoured** — ε inside the sqrt, mean-square initialised to 1.0. `timm` ships `RMSpropTF`
  for exactly this. `Proofs.rmsBufNext_eps_placement_at_zero` makes it a theorem: the textbook form
  steps `1/√ε` vs `1/ε`, i.e. **31.6× at EfficientNet's ε = 1e-3 and 1× at MobileNetV2's ε = 1.0**.
  Measured, the controls fire 4× harder on enet, so **a green mnv2 gate does not license the enet
  render** — which is why `rms-tie` takes a net argument and both were run.

  ▶ **What is left is the DRIVER, and it moved to Tier C**: the mean-square slot must init to
  **1.0, not 0**, and the LR schedule must be exponential. Neither is a render change. Until both
  land these are correct renders of the right optimizer, **not a matched pair**.
* **`wdExcludeNormBias`** is cheap: the AdamW render already emits a per-parameter tail, so 1-D
  parameters just get the no-decay tail. The risk is picking the wrong predicate — timm excludes
  norm γ/β, biases, pos-embed, CLS **and** ConvNeXt's LayerScale γ, all of which are 1-D.

### Tier E — the render, structurally.
**Gradient clipping, stochastic depth.**

* **grad clip** needs the **global** norm across every parameter gradient, then a rescale — a
  cross-parameter reduction the render has never had. It cannot be done in the driver: under
  residency the gradients never leave the device, and even on the copying path the driver receives
  θ′, not g. ⚠ It also must happen *before* the optimizer update, so it cannot be bolted onto the
  tail. Non-trivial, and it is a hard blocker for ViT's 5e-4 LR — the reference config calls grad
  clip *"the unlock"* for that rate.
* **stochastic depth** is architectural: a per-block Bernoulli mask plus its VJP, and a train/eval
  divergence. This is a new layer family, not a knob.

### Tier F — render + a new theorem.
**bf16 / bf16Conv.** 4-6 sessions, needs `conv_close_mixed`, and `planning/bf16_renderer.md`
already has the ladder, the measured per-op-class numbers and the two refuted emit strategies.
⚠ Measured there: **bf16 depthwise conv is 0.50× — twice as SLOW** — so mnv2/EfficientNet/ConvNeXt
capture far less of it than R34 does. It is a throughput item; treat it as orthogonal to catching
JAX on accuracy.

---

## 4. The plan

### v1.0 — **run R34.** Zero build.
`resnet34-imagenet-verified-xla`, `momdp64`, 4 GPUs, residency on, 90 epochs against the
reference's **72.02%**. ~48 h train-only at the measured 386 ms/step × 5004 × 90. This is the pair
the whole §2k/§2p line of work was pointed at, and nothing blocks it.

### v1.1 — the driver tier (Tier C). Unblocks two nets' schedules and ConvNeXt's headline metric.
Exponential LR decay, then EMA. Both are Lean-side, both are gateable the usual way (an existing
run must be byte-identical with the feature off).

### v1.2 — ~~**RMSProp** (Tier D)~~ ✅ **RENDER DONE 2026-08-02**, driver half outstanding.
The op, both renderers, six artifacts and a two-net numeric gate landed in one session (it was one
op, not a family — see Tier D). **What is left is two driver lines** — mean-square init 1.0 and
exponential decay — after which mnv2 is at parity and can be run against **68.33%**; EfficientNet
needs dropPath + EMA on top for **72.31%**.

### v1.3 — mixup + cutmix (Tier B). The wire is already there. Closes the largest remaining
accuracy gap for ViT and ConvNeXt.

### v1.4 — `wdExcludeNormBias`, then grad clip. Grad clip is the gate on ViT's LR.

### v2 — stochastic depth, bf16.

---

## 5. Wall-clock budget, so the runs can be planned

Measured ms/step on 4 GPUs with residency (40-step probes, train-only, eval NOT counted):

| net | ms/step | steps/epoch | reference epochs | **train-only** |
|---|---|---|---|---|
| R34 | 386 | 5004 | 90 | **48 h** |
| ConvNeXt | 270 | 10009 | 80 | **60 h** |
| mnv2 | 294 | 5004 | 90 | **37 h** |
| EfficientNet | 310 | 5004 | 80 | **34 h** |
| ViT | 424 | 2502 | 80 / 300 | **24 h / 88 h** |

**All five at reference epochs ≈ 200 h ≈ 8.5 days of solid 4-GPU time**, plus eval, plus the ViT
300-epoch tier if wanted. Sequence them; the box has 6 cards but every render here is baked at 4
replicas.

⚠ These are projections from 40-step probes, and ConvNeXt's is the least trustworthy — its
10,009 steps/epoch come from `cBS` still being private (global 128, not 256). Threading `cBS`
would halve its step count and is worth doing before a 60-hour run.

---

## 6. What is already unblocked, and should not be re-litigated

* **Soft targets need no render change.** Measured, not argued (`soft-target-tie`).
* **Wire v2 works and is inert when off.** Gated bit-identical, with a refusal control.
* **All five param counts match their reference exactly** — 21,797,672 · 5,717,416 · 28,587,592 ·
  5,288,548 · 3,504,872. This is the precondition that makes any accuracy comparison meaningful,
  and it is the check that caught R34 being the wrong net entirely (§2k).
* **Sharding is proven on the BN nets** (`shard-check`, N-replica) and the collective is bit-exact
  on the LayerNorm ones.
* **Residency is bit-identical on all five** at 4 replicas.
* **RMSProp renders exist and are certified on both nets** (2026-08-02) — `rms-tie
  [mobilenetv2|efficientnet]`, both controls firing. Do not re-derive the op: three of its four
  steps are existing certified ops, and `Proofs/Codegen/RmsPropStep.lean` says which.
* **An estimate in THIS document was wrong by a tier** — Tier D's "a new op family, ten sites
  each" for RMSProp was one op. The transferable check is the one §2k used and this repeated:
  before scoping a new optimizer, enumerate the reference's update line by line against the
  existing `SHlo` ops **at their other readings** (`momVNextF` at `(μ:=wd, v:=θ)` is a coupled L2;
  `adamVNextF` at `β₂:=ρ` is a running mean-square). Two of four steps hid there.

---

## 7. The honest caveats

**Catching JAX on accuracy does not move the verification tier.** §5's claim ceiling applies to
every number this document is about: the proof-carrying tier stops at Imagenette, and what the
ImageNet runs inherit is *provenance* — the artifacts are `pretty(provenGraph)` off the same
renderers — plus whatever the pair agreement shows. "One architecture, two independent lowerings,
agreeing" is the strongest honest phrasing, and if IREE is deprecated it becomes "a verified
rendering and an independent JAX implementation, agreeing."

**Some gaps may not need closing to catch the number.** EMA, mixup and stochastic depth are worth
between a fraction of a point and a couple of points each, and they interact. The disciplined
approach is: close a gap, run, measure the delta, and record it — which incidentally produces an
ablation table nobody has for a verified trainer.

**⚠ Watch for the K-constant bug in anything new.** Five copies of one defect were found across
four nets in a single session (R34's cotangent, ConvNeXt's loss, EfficientNet's loss, mnv2's
cotangent *and* loss): a label-smoothing constant hardcoded at the K=10 value. It survives because
the obvious check greps the cotangent's negative spelling `-0.010000` and the loss copy is
positive. **Any new emitted constant that depends on `nClasses`, `alpha` or the batch must be
derived and gated by a byte-identical re-render at the old values** — which is exactly how a wrong
α was caught mid-edit on mnv2.
