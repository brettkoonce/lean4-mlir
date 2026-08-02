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
| | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ **render + driver, 2026-08-02** |
| weight decay | 1e-4 coupled ✅ | 0.05 ✅ | 0.05 ✅ | 1e-5 ✅* | 4e-5 ✅* | decoupled in AdamW, coupled in heavy-ball **and RMSProp** |
| `wdExcludeNormBias` | — | ✅→❌ | ✅→❌ | — | — | ❌ |
| LR schedule | cosine ✅ | cosine ✅ | cosine ✅ | **exp 0.97** ✅ | **exp 0.98** ✅ | cosine **or exponential** + warmup |
| warmup | 5 ✅ | 5 ✅ | 20 ✅ | 5 ✅ | 5 ✅ | driver arg |
| label smoothing | 0.1 ✅ | 0.1 ✅ | 0.1 ✅ | 0.1 ✅ | 0.0 ✅ | derived from `nClasses` |
| grad clip | — | 1.0 ❌ | 1.0 ❌ | — | — | ❌ |
| **mixup / cutmix** | — | 0.8/1.0 ❌ | 0.8/1.0 ❌ | — | — | wire ✅, **producer ❌** |
| RandAugment (geo) | — | ✅ | ✅ | — | — | ✅ **free via shim** |
| random erasing | — | ✅ | ✅ | — | — | ✅ **free via shim** |
| repeated aug | — | 3 ✅ | — | — | — | ✅ **free via shim** |
| stochastic depth | — | 0.1 ❌ | 0.1 ❌ | 0.2 ❌ | — | ❌ |
| EMA | — | ✅→❌ | ✅→**✅** | ✅→❌ | — | ✅ **ConvNeXt 2026-08-02**; ViT + enet owed (enet also needs `ema_bn`) |
| **bf16 / bf16Conv** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| batch / epochs | ✅ | ✅ | ✅ | ✅ | ✅ | all matched at 4 replicas |

**Distance to parity, per net:**

| net | gaps | verdict |
|---|---|---|
| **R34** | bf16 | **at parity — run it** |
| **mnv2** | ~~RMSProp~~ ✅, ~~ms-init 1.0 + exp decay~~ ✅, bf16 | **at parity — run it** |
| **EfficientNet** | ~~RMSProp~~ ✅, ~~ms-init 1.0 + exp decay~~ ✅, dropPath, EMA, bf16 | two regularisers short |
| **ConvNeXt** | mixup, cutmix, dropPath, ~~EMA~~ ✅, grad clip, wdExclude, bf16 | the aug/regulariser pack, minus EMA |
| **ViT** | same as ConvNeXt | same |

✅ **BOTH HALVES OF RMSProp ARE DONE as of 2026-08-02** — the render (v1.2, §3 Tier D) and now the
driver: the mean-square slot initialises to **1.0** and the exponential schedule is threaded
(§3 Tier C). `\*` on the weight-decay row means "on the `rms*` variants", which is where the
reference's coupled 4e-5 / 1e-5 lives; the AdamW variants of those two nets are unchanged.

⚠ **`\*` also on "at parity" for mnv2: it means every FEATURE matches, not that the number has been
reproduced.** The 68.33% run has not been done. What has been measured is that the optimizer is the
reference's, that it descends (§3 Tier C), and that its DP render passes a numeric gate at four
replicas.

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

* ✅ **exponential LR decay — DONE 2026-08-02.** It was as cheap as scoped: `expDecayRate` /
  `expDecayEpochs` as trailing optional arguments on `trainAdamSched`, defaulting to 0.0 = cosine,
  so every existing call site is untouched. The formula is transcribed from the one
  `jax/Jax/Codegen.lean` **emits** for these two references rather than from the prose:
  `lr = LR · rate^((_ep − warmup)/decayEpochs)` with `_ep = _global_step / steps_per_epoch`.
  ⚠ **`_global_step` there is 0-based at the point the LR is computed** — its own warmup branch
  reads `(_global_step + 1)/warmup_steps` — so the epoch is `(gstep − 1)/nb`, not `gstep/nb`. One
  step of offset is invisible in a 5004-step epoch, which is why it had to be read off the
  generator. Gated as a **known answer**: the six printed per-epoch LRs across the warmup boundary
  match the reference formula recomputed in Python to ≤3.5e-7, i.e. to the rounding of the
  driver's own six-decimal print.
* ✅ **the RMSProp mean-square init (1.0, not 0) — DONE 2026-08-02**, and the correctness argument
  below is now also a measurement. TF's RMSProp is not bias-corrected, so `s = 0` makes the first
  step `gw/√((1−ρ)gw²+ε)` instead of `gw/√(ρ + (1−ρ)gw²+ε)` — much larger, with nothing downstream
  to absorb it. **Measured as a single-variable control**: pre-change vs post-change driver on the
  same mnv2 RMSProp render, same LR, entirely inside warmup so the schedule is identical, moves
  **20,929,404 of 26,840,184 bytes** of `[θ|m|v]` — 78%. It is one line and it is not cosmetic.
  ⚠ It lives in the DRIVER because it is the initial value of a graph *input*; the step graph
  never sees step 0. `Proofs.rmsBufNext` was correct either way.
* **EMA** — ⚠ **SCOPED 2026-08-02, `planning/ema.md`, and the "~a day, driver-only" estimate here
  is STALE BY ONE COMMIT: §2d.3.** It was written before device residency, and under residency θ
  never reaches the host between epochs, so a host-side per-step shadow reintroduces the 260 MB
  round trip §2d.3 removed. *Where the update runs* is now the whole design question. Three findings:
  * ✅ **zero new ops.** `adamMNext β₁ m g = β₁·m + (1−β₁)·g` **IS** the EMA update at
    `(β₁ := d, m := ema, g := θ')`, and `adamMNextF` takes β₁/(1−β₁) **by SSA name**, so it carries
    the reference's *time-varying* decay as runtime scalars. Third time this check has paid
    (§2k, v1.2, here).
  * ▶ **recommended design: in-graph, a 4th `[θ|m|v|ema]` region** — the resident shim's "n tensors
    in, n out, counts must agree" contract supports it *by construction*, so residency costs
    nothing. ⚠ It changes the CHECKPOINT FORMAT by 33%, and checkpoints carry no header — a size
    guard is mandatory.
  * ⚠⚠ **the warmup correction is REQUIRED at our scale, not optional.** The reference already hit
    this and measured it: decay 0.9999 at 31.2k steps left **12.8% of the random init** in the
    shadow and scored **0.00% top-1** while the live weights scored 70.48%. An 80-epoch Imagenette
    run is 23,600 steps = 2.4τ, i.e. **squarely inside that regime** — `d = min(decay, (1+t)/(10+t))`
    from the start, or the run scores at chance and looks like a broken feature.
  ✅ **BUILT FOR CONVNEXT 2026-08-02** (`LEAN_MLIR_VARIANT=ema`, design A). Zero new ops as
  predicted; 727 in / 725 out; gate 1 at 0 diff lines; **`decay = 0` ⇒ shadow BIT-IDENTICAL to the
  live weights** (0 of 27,826,282); the ratio known-answer at **8.7e-7** against a
  no-warmup-correction control at **0.90**; the checkpoint size guard fires rc=1; residency holds
  the 4th region free (720 = 4×180 tensors). And the gate the memory asks for **passes** — the
  shadow TRACKS THEN EXCEEDS from epoch 1: **48.25/57.35/63.03/67.77%** against the live weights'
  39.95/46.75/56.23/58.29%, never near chance. ⚠ Still owed: the DP peer, ViT/EfficientNet
  (the latter needs `ema_bn`, which §5c measures as nearly free), and a long run.

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

  ▶ ~~**What is left is the DRIVER, and it moved to Tier C**~~ ✅ **BOTH LANDED 2026-08-02** — see
  Tier C. The mean-square slot inits to 1.0 and the exponential schedule is threaded, each with its
  own control, and the AdamW path is bit-identical across the change (0 bytes of 638,904 on
  cifar8-bn, against a cross-process floor of 0 under `scripts/det_shim.sh`). **It descends on both
  nets** — §4 v1.2 has the runs.

  ▶ **The DP renders are compiled and gated too**, which was the third thing this section owed.
  `mnv2in_rmsdp64` and `enetin_rmsdp64` now compile and execute at **4 replicas** and pass the
  duplicated-batch identity: forward **BIT-EXACT** (34,112 / 42,016 BN statistics), buffer norm-rel
  **7.8e-7 / 8.1e-7**, against sum-not-mean controls that fire at **2.22 / 2.39** with rc=1 — six
  orders of separation.

  ⚠ **And a limitation of an existing gate, found here and worth knowing before reusing it:
  `shard-check` CANNOT gate a nonlinear-tail optimizer.** Its known answer is
  `DP([A|B]) = mean(single(A), single(B))`, which requires the gated slot to be **linear in the
  gradient** — true of AdamW's `m` at `m = 0` (`m' = (1−β₁)·g`, which is exactly why that harness
  gates `m` and not `θ'`) and **false of RMSProp's buffer**,
  `b' = μ·b + gw/√(ρ·s + (1−ρ)·gw² + ε)`. The duplicated-batch identity in `*-dp-check` is
  optimizer-AGNOSTIC — both sides receive the identical gradient, so any tail must agree — so that
  is the construction that transfers, and the two `dp-check` harnesses were generalised
  (`DP_NET` / `DP_VARIANT{,_DP}` / `DP_REPLICAS` / `DP_BATCH`, the `TestShardCheck` shape) rather
  than a third one being written. Each still reproduces its committed 2-replica result with no
  arguments, which is the gate on the generalisation itself.
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
* ~~**stochastic depth** is architectural: a per-block Bernoulli mask plus its VJP, and a train/eval
  divergence. This is a new layer family, not a knob.~~ ⚠ **SCOPED 2026-08-02 —
  `planning/stochastic_depth.md`, and this tiering is wrong in BOTH directions.** The math is
  smaller than Tier E implies: the reference's drop-path is a **per-example scale**
  (`branch · keep / keep_prob`, mask `(B,1,…,1)`), not a new layer family, and its VJP is **itself**
  — one `selectMidB`-shaped constructor. What actually costs is the plumbing: it is the first
  **per-step random graph INPUT** in the repo, which touches the driver, the shim's shard mask and
  every arity-sensitive harness on three nets (26 train steps + 8 forwards).
  Two findings from that doc worth having here:
  * the mask must be **host-drawn and passed in**, never `stablehlo.rng` — in-graph randomness would
    void every bit-exactness and known-answer gate in the repo at once;
  * ⚠ **the duplicated-batch `*-dp-check` gates are structurally BLIND to a mis-sharded mask** (both
    replicas get the same rows, so sharded and replicated are indistinguishable), and `shard-check`
    — the gate that exists for exactly that hole — **cannot be used on EfficientNet**, because its
    construction needs linearity in the gradient and that net wants stochastic depth *and* RMSProp.
    That is an open design question, and it is a prerequisite, not a follow-up.

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

### v1.1 — the driver tier (Tier C). ✅ **HALF DONE 2026-08-02 — exponential LR decay landed; EMA is what remains.**
Both are Lean-side and both are gateable the usual way, which is exactly how the schedule was
gated: an existing run must be **bit-identical** with the feature off (measured — 0 bytes of
638,904 on cifar8-bn AdamW, against a cross-process floor of 0 under `scripts/det_shim.sh`; note
the floor is NOT free on CUDA, §2d.3's Finding 1 is ROCm-specific). **EMA is now the only Tier C
item left**, and it is the one ConvNeXt's headline metric depends on — its 75.93% is the EMA
shadow's, not the raw weights'.

### v1.2 — ~~**RMSProp** (Tier D)~~ ✅ **DONE 2026-08-02, BOTH HALVES.**
The op, both renderers, six artifacts and a two-net numeric gate landed in one session (it was one
op, not a family — see Tier D); the driver half — mean-square init 1.0, exponential decay, the DP
renders compiled and gated at 4 replicas — landed in the next. **mnv2 is now at feature parity**
and is the second net after R34 where v1 is a run rather than a build; EfficientNet still needs
dropPath + EMA for **72.31%**.

**It descends on both nets** — ⛔ but the runs were **killed mid-flight** (sustained 4-GPU load
destabilises this box), so these are descent evidence and NOT accuracy numbers:

| Imagenette, bs32, 224² no-crop | stopped at | best val |
|---|---|---|
| **mnv2 `rms`** | ep 60 of 80 | **76.48%** |
| mnv2 `adam` (same-box control) | ep 61 of 80 | 82.42% |
| **EfficientNet `rms`** | ep 50 of 80 | **85.50%** |
| EfficientNet `adam` (same-box control) | ep 50 of 80 | 80.36% |

⚠ **Do not read these against 68.33% / 72.31%** — different dataset, batch, epoch count and
augmentation, with the peak LR batch-scaled from the reference's 256. And do not read them against
the handoff's §0b 80-epoch table either: that was a different box at 256² **with** random crop.
The AdamW columns exist so the RMSProp ones have a same-box, same-augmentation peer at all.

✅ **Residency is gated on the `rms` variant too** — bit-identical on **0 of 26,840,184** (mnv2)
and **0 of 48,244,296** (enet) bytes against bit-exact floors, both controls firing, and
`scripts/residency_gate_all.sh` carries the two rows so they run by default. ⚠ At **fault mode 2**:
**RMSProp is contractive** — the 1-ULP fault lands at 2 bytes of 26.8M — which makes it the third
counterexample to "AdamW ⇒ chaotic ⇒ mode 1" and the first where the property tracks the optimizer
rather than the net.

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
* **RMSProp's DRIVER half is done too** (2026-08-02) — mean-square init 1.0, exponential decay,
  the DP renders compiled and gated at 4 replicas, descent on both nets. The reference's peak LR
  and decay live in ONE place, `RmsSchedule` in `LeanMlir/VerifiedNets.lean`, read by all four
  entry points; the *emitted* half (ρ/μ/ε/wd) stays in `Proofs.StableHLO.RmsHyper`. They are
  deliberately in different modules — a learning rate that became a graph constant is the
  `RenderCifar8Sgd02` / EfficientNet-16× failure this repo has already paid for twice.
* ⚠ **`shard-check` cannot gate an optimizer whose tail is nonlinear in the gradient** — its known
  answer is `DP([A|B]) = mean(single(A), single(B))`. Use the duplicated-batch identity in
  `*-dp-check`, which is optimizer-agnostic. Both dp-check harnesses now take
  `DP_NET`/`DP_VARIANT{,_DP}`/`DP_REPLICAS`/`DP_BATCH` and still reproduce their committed
  2-replica results with no arguments.
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
