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
| `wdExcludeNormBias` | — | ✅→**✅** | ✅→**✅** | — | — | ✅ **BOTH nets, 2026-08-02** — `wx` variant, `lake build wdx-tie {vit,convnext}` |
| LR schedule | cosine ✅ | cosine ✅ | cosine ✅ | **exp 0.97** ✅ | **exp 0.98** ✅ | cosine **or exponential** + warmup |
| warmup | 5 ✅ | 5 ✅ | 20 ✅ | 5 ✅ | 5 ✅ | driver arg |
| label smoothing | 0.1 ✅ | 0.1 ✅ | 0.1 ✅ | 0.1 ✅ | 0.0 ✅ | derived from `nClasses` |
| grad clip | — | 1.0 ✅ | 1.0 ✅ | — | — | ✅ **BOTH nets, 2026-08-02** — `clip` variant, `lake build clip-tie {vit,convnext}` |
| **mixup / cutmix** | — | 0.8/1.0 ⚠ | 0.8/1.0 ⚠ | — | — | producer ✅ + shim ✅, but **OFF on the default path**: a mixed target cannot ride wire v1's int32 labels, so the driver passes `SHIM_MIX=off` and prints that it did. `SHIM_SOFT=1` turns it on. ⚠ λ is numpy's, not `jax.random`'s — agreement is **permanently distribution-only** |
| RandAugment (geo) | — | ✅ | ✅ | — | — | ✅ **WIRED 2026-08-02** (§0.9) — `VerifiedNet.shimScript`, no fallback; `shim_wiring_gate.py` gate ② confirms call sites match each config |
| random erasing | — | ✅ | ✅ | — | — | ✅ **WIRED 2026-08-02** (§0.9) |
| repeated aug | — | 3 ✅ | — | — | — | ✅ **WIRED 2026-08-02** (§0.9) |
| **AutoAugment** | — | — | — | ✅ | — | ✅ **WIRED 2026-08-02** (§0.9) — and control C1 measures why a definition census gets this wrong: ViT/ConvNeXt *define* `_autoaugment` and never call it |
| **classifier dropout** | — | — | — | **0.2 ✅** | — | ✅ **DONE 2026-08-03** — `dropoutB` op + cert, `adamdo`/`emarms64dropdo` renders, `lake build dropout-tie` (gate A bit-exact 40/40 + gate W). ⚠ It is NOT the stochastic-depth row below: same op (`layerScale`), **different mask RANK** — per ELEMENT here, per EXAMPLE there |
| stochastic depth | — | 0.1 ✅ | 0.1 ✅ | 0.2 ✅ | — | ✅ **ALL THREE, both scales, 2026-08-02/03** — EfficientNet (§0.4), ConvNeXt (§0.10) and ViT (§0.11), the last two on the batched chain the §0.2 ▶2/▶3 move built. DP mask-shard gate run on each (§0.6) |
| EMA | — | ✅ | ✅ | ✅ | — | ✅ **all three, 2026-08-02**, and at BOTH scales — `vitin_ema128`/`emadp128x4`, `convnextin_ema`/`emadp`, `efficientnetin_emarms64`/`emarmsdp64`. DP peers gated at 4 replicas, every region bit-exact (§0.7). ⚠ Never GATE on the shadow — it is θ's low-pass filter (§3) |
| **bf16 / bf16Conv** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| batch / epochs | ✅ | ✅ | ✅ | ✅ | ✅ | all matched at 4 replicas |

⚠ **A ✅ on this matrix means the RENDER exists — check it is the render a RUN would load.** Until
2026-08-02 every ViT/ConvNeXt ✅ above was true of the single-device artifact and false of the
data-parallel one, which is the artifact an ImageNet run actually loads: the DP renders were three
features behind (`wd` 500× off, no `wx`, no clip). Closed by `vitin_adamdp128x4wxclip` /
`convnextin_adamdpwxclip` (handoff §0.5). The check that found it was listing what each artifact BAKES,
not reading this table.

**Distance to parity, per net:**

| net | gaps | verdict |
|---|---|---|
| **R34** | bf16 | **at parity — run it** |
| **mnv2** | ~~RMSProp~~ ✅, ~~ms-init 1.0 + exp decay~~ ✅, bf16 | **at parity — run it** |
| **EfficientNet** | ~~RMSProp~~ ✅, ~~EMA~~ ✅, ~~dropPath~~ ✅, ~~classifier dropout~~ ✅ (2026-08-03), bf16 | **at parity — run it.** `efficientnetin_emarms64dropdo` is the whole recipe in one artifact |
| **ConvNeXt** | ~~dropPath~~ ✅, ~~EMA~~ ✅, ~~grad clip~~ ✅, ~~wdExclude~~ ✅, ~~aug pack~~ ✅, **mixup/cutmix off by default**, bf16 | **at parity except mixup**, which is wired but needs `SHIM_SOFT=1` |
| **ViT** | same as ConvNeXt | same |

⚠⚠ **EVERY REMAINING ROW IS A RUN OR A THROUGHPUT ITEM, NOT A RENDER GAP** — as of 2026-08-03 the
last unlisted render gap (EfficientNet's classifier dropout) closed. What is left on this matrix is
**bf16** (throughput, and `bf16_renderer.md` measured bf16 depthwise conv at 0.50× — *slower* — on
the depthwise nets), **mixup's default-off wiring**, and **the runs themselves**. ⚠ Read that as
"the code is there", NOT as "the numbers are reproduced": §0.1 says this box cannot do long runs,
and no net has an ImageNet pair run on the verified path.

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

### Tier A — the pipeline. ✅ **AVAILABLE *AND WIRED*, 2026-08-02** (it was wrong as state for months)
RandAugment (full geometric, m9/mstd0.5/inc1), random erasing, repeated augmentation, RRC + hflip.

These live inside `build_imagenet_iter`, which `generateShim` reuses **verbatim** (§2k). Turning
them on in the JAX config moves both paths at once, and the verified side owns no augmentation code
at all. ~~**Nothing to build.**~~

> ⚠⚠ **CORRECTED 2026-08-02. The mechanism is free; the WIRING is not, and this tier's ✅ marks in
> §2 were reading as state when they were only a capability.**
>
> `spawnShim` defaults to **`generated_resnet34_imagenet_shim.py` for every net**
> (`VerifiedTrain.lean:216`), `SHIM_SCRIPT` is set **nowhere** in any job conf, script or doc, and
> R34's is the **only generated shim on disk**. Grepped inside that file: `_autoaugment`,
> `_randaugment`, `_random_erase` and repeated aug are **not defined and not called**; it applies
> RRC + hflip only, because R34's config sets just `augment := true`.
>
> So **a verified EfficientNet / ViT / ConvNeXt ImageNet run as documented streams R34's
> augmentation** — not AutoAugment (enet), not RandAugment m9/mstd0.5/inc1 + random erasing +
> repeated aug ×3 (ViT). `generateShim` *does* honour all of them (`useAutoAugment` at
> `Codegen.lean:335`, `randomErasing` :368, `repeatedAug` :425) — nothing is missing but the
> generation and the per-net default.
>
> ✅ **CLOSED THE SAME DAY — handoff §0.9.** `VerifiedNet.shimScript` (no default, no fallback —
> an empty one refuses at spawn), `scripts/gen_shims.sh` as the one writer, and
> `scripts/shim_wiring_gate.py` holding it closed. **Measured**: ViT / EfficientNet / ConvNeXt each
> stream a different train digest from R34's now; **mnv2 ≡ R34 to the bit**, which is a prediction
> (its config *is* R34's crop/flip) and is the inertness half this plan did not anticipate; all
> five validation streams are identical. Two things it turned up: **a definition is not a call
> site** (`_autoaugment` is *defined* in ViT's and ConvNeXt's shims and never called, so a
> definition census mis-reads 2 of 5 nets), and **the right shim breaks the default path** —
> ViT/ConvNeXt bake `SHIM_MIX=both`, which wire v1 cannot carry, so the driver passes `off` and
> announces it. So the ✅ marks in §2 now read as state for this tier.

### Tier B — the producer (shim-side Python). ✅ **DONE 2026-08-02 — wire AND mixing.**
**mixup, cutmix.**

Two things had already landed, and they are why this was the next-cheapest real item:
* **wire v2** carries `float32[batch·nClasses]` soft targets, gated bit-identical against v1
  (commit `4755317`);
* **the renders are AFFINE in `%onehot`** — measured, `lake build soft-target-tie`, ViT 492× /
  ConvNeXt 309× separation — so a mixed target yields the mixed gradient **with no render change**.

✅ **The producer now fills those slots.** `generateShim`'s `_mix` is a line-for-line numpy
transcription of the reference's `_mixup`/`_cutmix`, selected by `SHIM_MIX=off|mixup|cutmix|both`
(`both` alternates per step, as the reference does), with the config's own alphas as defaults and
`SHIM_MIXUP_ALPHA` / `SHIM_CUTMIX_ALPHA` as overrides. **No render change, no new op, and no Lean
change** — `IO.Process.spawn`'s `env` array EXTENDS the inherited environment, so `SHIM_MIX` set on
the trainer's command line reaches the shim child. That was *checked, not assumed*.

Two refusals, both at startup: `SHIM_MIX` without `SHIM_NCLASSES` (a mixed label is a distribution;
int32 cannot carry one) and an unknown mode. Loud beats a run whose log says mixup while it trains
on hard labels.

⚠ **This is the one place where a second definition of something is unavoidable**: the JAX reference
applies `_mixup`/`_cutmix` in the *train step* with `jax.random`, not in `tf.data`, so a shim-side
implementation is a genuinely new copy of the mixing rule. It is much smaller and more
self-contained than the augmentation pipeline (a Beta draw, a convex combination, a box), but it is
a second writer and the doc should say so rather than pretend otherwise. The alternative — no mixup
on the verified path — is worse.

⚠⚠ **AND THE COST OF THAT SECOND DEFINITION IS NOW PRECISE, WHICH IT WAS NOT BEFORE.** The reference
draws `jax.random.beta(fold_in(PRNGKey(seed), step), α, α)`; the shim draws from numpy's
`Generator`. Both are Beta(α, α); **they are not the same numbers, and no seeding makes them so.**
So a verified-vs-JAX pair under mixup agrees **in distribution, not per step** — strictly weaker
than the augmentation pipeline's byte-identity, which is reused verbatim and therefore cannot drift
at all. Never quote the two as the same kind of agreement. It is also why the gates below are
known-answer and determinism gates rather than a cross-path byte comparison: **that comparison does
not exist to be made.**

#### The gates — `scripts/mixup_gate.py`, all three green with controls firing

| gate | result |
|---|---|
| **1. inert when off** | v1 `c375ad0f…`, v2 `f3a4b2a0…` — **byte-identical to the pre-mixing shim**, measured on the pre-change generated script at the same config |
| **2. determinism** | same seed ⇒ same digest on all three modes; **different seed ⇒ DIFFERENT digest** on all three |
| **3. known answer** | the mixed stream vs the OFF stream at one seed: `t' = λ·t + (1−λ)·flip(t)` and `x' = λ·x + (1−λ)·flip(x)` **BIT-EXACT**, on images and labels alike |
| 3b. cutmix structure | the pasted region recovered **from the pixels** is a RECTANGLE, and λ equals `1 − area/(H·W)` to 1e-6 — i.e. the label follows the CLIPPED box, not the drawn λ |
| ⚠ control A | unmixed images against a mixed target — **rejected**, 1.15 |
| ⚠ control B | λ wrong by 1% — **rejected**, 7.5e-3 |

**Control A is the one that earns its keep**: a mixed label with unmixed pixels compiles, streams,
trains and descends — it is simply a worse objective — so a gate that checked only the target would
pass it. Hence the images are gated too.

**Three findings from building it:**

1. ⚠⚠ **RECOVER A CONSTANT BY READING IT, NOT BY FITTING IT.** The first λ recovery solved
   `tm = λ·t + (1−λ)·flip(t)` by least squares in float64 and cast to float32. That reconstructs λ
   to about a ULP — fine for a tolerance check, and **useless feeding a bit-exact assertion**: mixup
   batch 0 passed and batch 1 failed at 1.5e-08, a phantom defect in a correct producer. `t` is
   one-hot, so λ is *literally stored* at `tm[i, y_i]` for any row whose partner carries a different
   label. Reading it is exact and needs no tolerance — and it gates a real property for free, since
   λ is a per-STEP scalar and every identified row must agree.
2. ⚠ **A GATE THAT ONLY PRINTS CANNOT FAIL.** Gate 1 first printed the two digests with "compare
   against the doc" — unfalsifiable, the `vit-dp-check` defect in its purest form. It now checks
   against baselines measured on the pre-change shim, and **SKIPS LOUDLY** at any other
   batch/batch-count rather than silently comparing incomparable numbers.
3. ⚠ **"Inert when off" is a CROSS-VERSION claim and needs a recorded constant.** That the new
   shim's two modes agree with each other is trivial — one calls the other's code path. What
   matters is that the off path is byte-for-byte the shim as it was *before mixing existed*, and
   nothing computable at run time substitutes for having measured that.

#### ✅ The end-to-end smoke, after the fix — mnv2/ImageNet, 1 GPU, 6 steps, fresh checkpoint each

The trio was run TWICE, which is what gives the floors rather than a single before/after pair
(§2j: *one sample is not a measurement*):

| run | step 0 | step 1 | step 2 | epoch 1 |
|---|---|---|---|---|
| **A** v1 hard labels, trio 1 | 7.019591 | 7.017135 | 7.069839 | 7.057056 |
| **A** v1 hard labels, trio 2 | 7.019591 | 7.017126 | 7.069849 | 7.057081 |
| **B** v2 one-hot, trio 1 | 7.019591 | 7.017126 | 7.069456 | 7.056798 |
| **B** v2 one-hot, trio 2 | 7.020040 | 7.016245 | 7.068997 | 7.056218 |
| **C** v2 **MIXED** (`SHIM_MIX=both`) | **7.029974** | 7.026065 | 7.063550 | 7.052764 |

* **C is ~1.0e-2 above the unmixed band, ≈20× the B-vs-B run-to-run floor of 4.5e-4** — and it is
  *higher*, which is what a higher-entropy target must do. So the mixed stream reaches the trainer
  and changes the objective, which is what the smoke existed to establish.
* **A-vs-B sits INSIDE that floor** (0 to 4.5e-4, against a B-vs-B floor of 4.5e-4) — the two wires
  agree as well as one wire agrees with itself, §2f-bis's shape again.
* ⚠ **2 samples per config, and NOT run under `det_shim.sh`** — my own miss, since the 2026-08-02
  finding says the det shim is mandatory for any cross-process numeric comparison on CUDA. Quote
  the C separation as ≈20× a measured floor, not as a bound on anything. The *stream*-level v1/v2
  bit-identity is wire v2's own gate (`SHIM_HASH`, and re-confirmed here at the trainer's exact
  seed and batch: byte-identical pre-mixing vs current on both wires).

#### ⛔⛔ AND THE END-TO-END SMOKE FOUND A DEFECT ALL THREE GATES MISSED — 2026-08-02

The smoke was nearly skipped as confirmatory. It was not: `SHIM_MIX=both` **killed the trainer
before its first step.**

`SHIM_MIX` is an ordinary environment variable, so **every shim the driver spawns inherits it — and
it spawns two**: the train stream at `nclasses = K`, and the **validation drain at `nclasses = 0`**
(wire v1, hard labels, because eval scores against a label, not a distribution). Gating the mixing
on the variable alone made the "needs `SHIM_NCLASSES>0`" refusal fire on the *val* shim, which died
before writing its preamble. The trainer then reported

```
uncaught exception: imagenet shim closed the pipe after 0 of 16 bytes (did it crash? …)
```

**Silencing the refusal would have been the wrong fix.** Mixup/CutMix are TRAIN-time augmentations —
the reference applies them inside the train loop only, and a mixed validation target would score the
net against a convex combination of two labels, which is not the metric. So the rule is *mix the
train split, never the eval one*, and `_MIX_ON = (_MIX_MODE != 'off') and training` is it.

> ⚠⚠ **A GATE THAT EXERCISES ONE SPLIT CANNOT SEE A SPLIT-DEPENDENT DEFECT.** Gates 1-3 were green
> throughout, and they always will be: every one of them drives the **train** split. Nothing
> producer-side would have found this. It is §5's duplicated-batch hole in a third place — *a gate
> whose input makes the failure impossible cannot test for it* — and it is the argument for running
> the cheap end-to-end check even when the component gates are green.

`mixup_gate.py` grew **gate 1b** for it: at every mode, the validation split must (a) survive
`SHIM_MIX` at wire v1, the driver's own spawn, and (b) hash **identically to the unmixed** validation
stream.

⚠ Two ImageNet-smoke traps found on the way, both worth knowing before the next one:
* the step cap for `trainAdamSched` is **`LEAN_MLIR_G2_STEPS`**, not `LEAN_MLIR_MAX_STEPS` — the
  latter means *"time a step window then exit"* and does not bound the run (a smoke set with it ran
  past step 300);
* the **28 GB validation drain is unconditional** (195 × 256 images) and `LEAN_MLIR_SKIP_EVAL` does
  NOT skip it — it gates the eval *pass*, not `loadData`'s drain. Budget ~4-5 min per config.
* ⚠ all configs share slug+variant, so **delete `.lake/build/<slug>_<variant>_ckpt_xla.bin{,.epoch}`
  between them** or run B silently resumes run A (§4).

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
  `mobilenetv2in_rmsdp64` and `efficientnetin_rmsdp64` now compile and execute at **4 replicas** and pass the
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

* ~~**grad clip** needs the **global** norm across every parameter gradient, then a rescale — a
  cross-parameter reduction the render has never had.~~ ✅ **DONE 2026-08-02, and THIS TIERING WAS
  WRONG BY A TIER** — the second time an estimate in this document was (RMSProp's Tier D "op family"
  was one op). It is **two ops and no new proof machinery**: the "cross-parameter reduction" reads
  like a shared DAG node where `SHlo` is a tree, and that dissolves because **`SHlo` is
  single-OUTPUT, not single-INPUT** and every gradient is already an `.operand` leaf, so the 200-way
  fold is an ordinary tree. `planning/grad_clip.md`. The original note follows.
  **grad clip** needs the **global** norm across every parameter gradient, then a rescale — a
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

⚠⚠ **READ THIS BEFORE PLANNING OFF §4: THE RUNS BELOW ARE NOT AVAILABLE ON THIS BOX.** Brett,
2026-08-02: *"no long runs this box will crash"* — sustained multi-GPU load destabilises ares. §5's
~203 h budget is a budget for hardware we do not currently have. So the ORDERING below still holds
as a statement of what closes each gap, but the *sequencing* in practice is: **do the build-and-gate
half of every item now** (certified render + numeric gate + a 3-4 epoch descent smoke, all
single-GPU and all minutes), and bank the reference-epoch runs. v1.0 is the one item that is
*purely* a run, which is why it is the one item currently blocked outright.

### v1.0 — **run R34.** Zero build. ⛔ **BLOCKED: hardware, not code.**
`resnet34-imagenet-verified`, `momdp64`, 4 GPUs, residency on, 90 epochs against the
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

### v1.2b — **EMA. ✅ ALL THREE NETS, 2026-08-02 — ConvNeXt (+ DP peer), EfficientNet, ViT.**
`planning/ema.md` — scoped, then built. **Zero new `SHlo` ops**: `adamMNext β₁ m g = β₁·m + (1−β₁)·g`
IS the reference's `ema_update` at `(β₁ := d, m := ema, g := θ')`. The blob gains a 4th
`[θ|m|v|ema]` region; residency supports that **by construction** (its "n in, n out, counts must
agree" contract), so it cost no C change.

**ViT was the last of the three and the cheapest, as predicted** — LayerNorm, so no `ema_bn`;
807 in / 805 out, zero new ops, and the driver needed nothing. Gates: `decay = 0` ⇒ shadow
**BIT-IDENTICAL** (0 of 5,526,346), ratio known answer **5.96e-5** against a no-warmup-correction
control at **150,995×**, residency at 4 regions **0 of 88,421,536 bytes**, and the shadow **exceeds
from epoch 1** (42.37/50.65/54.34/56.61% vs live 40.64/42.93/48.41/51.64%).

⚠⚠ **A GATE PORTED BETWEEN NETS INHERITS THE SOURCE NET'S CONDITIONING.** The ratio gate is a
difference of two nearly-equal f32 numbers; at ViT's step 1 the warmup LR is ~2e-7, which put
`ema − θ` at roughly ONE ULP of θ and made the gate read 1.96e-2 with its control only 455× away —
*with nothing wrong with the render*. ConvNeXt never saw it because its baseLR/warmup give a 50×
larger step 1. Fixed by conditioning the instrument (`LEAN_MLIR_BASE_LR_U`, the `r34-mom-tie` move).

⚠ **All EMA renders are on the IMAGENETTE slugs.** No `*in_ema*` artifact exists, so none of this
is in an ImageNet trainer yet — see §4's note on the scale gap.

⚠ **The warmup-corrected decay `d = min(decay, (1+t)/(10+t))` is required at our scale**, not
optional — an 80-epoch Imagenette run is 2.4 τ at decay 0.9999, inside the regime where the
reference measured a shadow holding 12.8% of the random init and scoring **0.00% top-1**.

### ⚠⚠ v1.2c — **THE SCALE GAP: every feature since RMSProp is on the IMAGENETTE slug only.**
Discovered 2026-08-02 by listing the artifacts rather than reasoning about them:

| feature | Imagenette slug | ImageNet slug (`*in_*`) |
|---|---|---|
| RMSProp | ✅ `efficientnet_rms`, `mobilenetv2_rms` | ✅ `efficientnetin_rms64`/`rmsdp64`, `mobilenetv2in_rms64`/`rmsdp64` |
| **EMA** | ✅ `convnext_ema{,dp}`, `efficientnet_emarms`, `vit_ema` | ❌ **none** |
| **stochastic depth** | ⚠ `efficientnet_adamsd` (render only) | ❌ **none** |

So `efficientnetin`'s trainer is still AdamW-or-RMSProp with no EMA and no dropPath — i.e. **the ImageNet
trainers do not yet carry the features their references need**, and the 72.31% pair is not
reachable through them today. RMSProp is the only feature that was carried to both scales.

**It is cheap**: every renderer takes `nClasses`/`B`/`slug` as ordinary parameters (§2p's finding
for ViT — *"no renderer change, three `#eval`s"*), so each missing variant is one more `#eval` plus
its drift-guard line. Do this BEFORE any v1.x reference-epoch run on enet or ConvNeXt, or the run
measures a net that is missing the regularisers its target number depends on.

### v1.3 — ✅ **mixup + cutmix (Tier B) — the PRODUCER landed 2026-08-02.** See Tier B above for
the gates, the controls and the three findings. What is left on this item is a run, not a build.

Producer-side Python only: **no render change and no new op**, because the renders are AFFINE in
`%onehot` (measured — `lake build soft-target-tie`, ViT 492× / ConvNeXt 309× separation) and shim
wire v2 already carries `float32[batch·nClasses]` soft targets, gated bit-identical against v1. So
it is a Beta draw, a convex combination and a box, and every gate is minutes — which is what makes
it the right item for a box that cannot do long runs.

Gates to write with it: **inert when off** (the `SHIM_HASH` byte-identity that gated wire v2);
**determinism** (same seed ⇒ same hash, AND different seeds ⇒ different hashes — the control that
stops a pipeline silently ignoring its seed from looking correct, which is the failure §2k's shim
work actually hit); and a **known answer on the mixed labels** — for a chosen λ and permutation the
target must be exactly `λ·y_a + (1−λ)·y_b`.

⚠ Keep saying that this is the one place a SECOND DEFINITION is unavoidable (Tier B above has the
argument): the reference mixes in the *train step* with `jax.random`, not in `tf.data`.

### v1.4 — ✅ **`wdExcludeNormBias` — BOTH nets (2026-08-02).** Then grad clip.

`lake build wdx-tie`. The reference rule (`jax/Jax/Codegen.lean`'s `_wd_mask`) decays only ≥2-D
weight matrices: every 1-D param (biases, LayerNorm γ/β, the CLS token) and the **positional
embedding** are excluded. Run over the reference's own `init_params`, that is **74 decayed / 126
excluded of 200**, every mask leaf uniform — so the decision is per-TENSOR.

**It needed no new op, no interface change and no driver change**, which is what made it an
evening: `adamWParamF` already takes `wd` as a runtime OPERAND NAME (the `%lr` shape, for the `%lr`
reason), so "exclude" is binding that operand to a zero constant. 126 of 200 operand strings move;
arity, types and regions do not. `vitWdDecays` derives the mask from `vitParamSig`, the same list
that names the sites, and `#guard`s pin the 74/126 against the reference's own count.

⚠ It keys the positional embedding by NAME where the reference keys it by SHAPE. Deliberate: the
reference walks an unnamed pytree, while a shape test here would also exclude any *other* param
that happened to be 197×192. And the rule reads the RANK, which is the one thing that survives the
layout difference — the render carries `Wfc1` as `[192,768]` where the reference has `(768,192)`.

| gate | result |
|---|---|
| gate 1 | every committed artifact **byte-identical**; only the two new `wx` paths appear |
| ① decayed params | θ' **BIT-EXACT** between `adam` and `adamwx`, 74/74 |
| ② excluded params | `θ'_wx − θ'_adam` = `lr·wd·θ` to **0.49 ULPs of θ'**, 126/126 |
| ③ `m'`, `v'` | **bit-exact on all 11,052,692** moment coordinates — decoupled decay touches neither |
| ④ `%loss` | bit-exact — a forward-only output cannot see the optimizer tail |
| ⚠ control `invert` | swap every `%wd`↔`%wdz` → **fires**, 200 params misclassified, rc=1 |
| ⚠ control `swap1` | flip ONE param each way → **fires on exactly 2**, rc=1 |

**▶ `swap1` is the control that justifies the design.** It leaves the counts at 74/126, so a gate
checking *how many* params moved passes it. The gate instead recovers, per parameter, which bucket
it EMPIRICALLY falls in and requires that partition to equal `vitWdDecays`' **name for name** —
which is the only thing that catches a mask excluding the wrong 126. That failure is otherwise
silent in the arity, the types and the prefix audit (§2e's slot rule).

`scripts/perturb_wd_mask.py` builds both controls; `wdx-tie --cand <path>` drives them.

#### ⛔ AND IT TURNED UP A SEPARATE 500× GAP: the ViT renders bake the WRONG weight decay

Found while conditioning the gate, not by reading the configs. `vitAdamConsts` baked
`%wd = 1e-4` for **every** ViT render — that is `vitTinyConfig`'s (Imagenette) value, but
**`vitTinyImagenetConfig.weightDecay := 0.05`**, the DeiT one. So an ImageNet ViT render was
training at **1/500th of its reference's decay**. It is the `RenderCifar8Sgd02` / EfficientNet-16×
shape (§2a-quater): a silently wrong hyperparameter in a committed artifact that compiles, runs and
descends.

`wdStr` is a parameter now (default unchanged, so gate 1 stayed free) and **`vitin_adam128wx`
renders at 0.05** — both halves of the reference's decay recipe, the magnitude and the mask.

⚠ **The same 500× gap is on ConvNeXt** (`convnextTinyImagenetConfig.weightDecay := 0.05` against a
baked 1e-4), and `convnextin_adamwx` renders at 0.05 for the same reason `vitin_adam128wx` does.

⚠ **`vitin_adam128`, `vitin_adamdp128x4`, `convnextin_adam` and `convnextin_adamdp` are STILL at 1e-4 and
were NOT touched.** Changing them
is a separate call with its own blast radius (the DP peer, the residency-gate row, the committed
`vit-dp-check` numbers). **Owed**, and it should be done before any ViT/ImageNet pair run — a
matched pair at the wrong decay is not a matched pair.

#### ✅ ConvNeXt followed the same day — **59 decayed / 121 excluded of 180**

The recipe transferred, and the prediction held: it is the **plain rank test**, no name carve-out.
Its generated reference sets `_WD_POS_SHAPE = None` (ConvNeXt has no positional parameter), so
ViT's `nm != "pos"` has no analogue — carrying it over would have transcribed a rule this net does
not have. Checked in the generated file, not assumed. What that leaves excluded: every LN γ/β,
every conv bias, and **LayerScale γ** — 1-D, so excluded structurally rather than as a special case.

| gate | ViT | ConvNeXt |
|---|---|---|
| params (decayed / excluded) | 200 (74 / 126) | 180 (**59 / 121**) |
| ① decayed θ' bit-exact | 74/74 | **59/59** |
| ② excluded offset = `lr·wd·θ` | 0.49 ULPs | **0.48 ULPs** |
| ③ `m'`,`v'` bit-exact | 11,052,692 | **55,652,564** |
| ④ `%loss` bit-exact | ✅ | ✅ |
| ⚠ `invert` / `swap1` controls | fire on 200 / 2 | fire on **180 / 2** |

**ONE harness for both** (`wdx-tie <net>`), per `rms-tie`/`shard-check`: a second copy is the
double-writer disease one level down, in code. Everything per-net comes from the renderer's own
signature list and mask predicate. ⚠ And a green ViT run does not license ConvNeXt — the two rules
genuinely differ, which is the `rms-tie` ε-placement lesson one knob over.

#### ⚠ And ConvNeXt's entry name is DERIVED, where ViT's is passed — the same change, two behaviours

`convNextAdamTrainStepFaithful` builds `funcName` from `{slug}_{cnxAdamVariant …}`, so the flag has
to reach the *variant* as well as the render. It did not, and the artifact
`convnext_adamwx_train_step.mlir` came out declaring `@convnext_adam_train_step` — an entry that
disagrees with its own path. **The shim refused the call outright** rather than running the wrong
graph (§2b-quater's entry check earning its keep a second time), so it surfaced as
`mlp train step failed` at the first invoke instead of as a silent wrong answer. `#guard`s on the
`wx` spellings pin it now. ViT never showed this because it takes `funcName` explicitly — *the same
edit behaves differently on the two renderers, and only running both found it.*

### v1.4b — ✅ **grad clip: DONE, BOTH NETS, 2026-08-02.** ⚠ **SCOPED 2026-08-02, `planning/grad_clip.md`, and the Tier E entry below is WRONG BY A TIER.**

`planning/grad_clip.md` §11 is the record: **two `SHlo` ops, no new proof machinery, no driver
change**, four artifacts (`vit_adamclip`, `vitin_adam128wxclip`, `convnext_adamclip`,
`convnextin_adamwxclip`), and `lake build clip-tie` green on both nets with **all six controls firing**.
The gate is the CONSTANCY of `m'_clip/m'_adam` across parameters (**1.12 / 1.15 ULPs**), because
that is the only property a per-parameter clip gets wrong — and its control fires at **7.6M /
30.4M ULPs** while passing every other gate. ⛔ Owed: the DP artifact and its numeric gate.

**The scoping below was written before the code and is kept for the method**; `xla_pjrt_handoff.md` §0.2 is the earlier sketch and this
paragraph supersedes both. Measured, it is **four small ops in the parameter-shape family, no new
proof machinery, and no driver change** — Tier D, not Tier E. Tier E's framing (*"a cross-parameter
reduction the render has never had"*) rests on the norm being a shared DAG node where `SHlo` is a
tree; that dissolves because **`SHlo` is single-OUTPUT, not single-INPUT** (`sub`/`addV`/`matmulF`
are already binary) and every gradient is already an `.operand` leaf, so the 200-way fold is an
ordinary tree with 200 leaves and nothing is recomputed. **No carve-out is needed on the norm.**
Three other corrections in that doc: the reference already measured the init grad norm at
**14.28–44.09 against a threshold of 1.0**, so the clipping regime is the default and the
*un*clipped one is what needs conditioning; §0.2's "breaks four gates" is **one** (`shard-check` —
the RMSProp-nonlinearity finding one optimizer over); and the shipping ViT/ConvNeXt variant is
`wx` **composed with** clip, which is the `emarms` composition trap.

`recipe_gaps` calls it *"the unlock for the 5e-4 LR"* on ViT. Used by **ViT 1.0** and **ConvNeXt
1.0**; EfficientNet sets it to **0.0 deliberately** (its own comment: the TF-RMSProp fix removed the
blow-up it compensated for) — do not add it there.

**It is the first feature here whose SHAPE the kit does not have.** The reference takes a GLOBAL L2
norm across all parameters, then scales every gradient by `min(1, clip/(gn+1e-6))`. Measured, the
kit already has the reduce-to-rank-0 shape (`lnBetaGrad`), rank-0 broadcast-multiply (200× per ViT
render, inside `adamWParamF`), `sqrt` and `minimum`; what it lacks is a **scale-by-a-RUNTIME-scalar**
op — `scaleF`/`scaleB` look like they take one but emit `constant dense<…>`, i.e. they bake a
literal.

⚠ The real question is not the op count: the norm is **one scalar consumed by all N sites**, a
shared DAG node, where `SHlo` is a single-output tree. Decide certified-vs-carve-out first.
⚠ And two ordering traps: under DP the clip must go **after** the `all_reduce` (clipping per-replica
partial gradients is a different function), and it **breaks every gate that recovers `g` from `m'`**
— `r34-mom-tie`, `rms-tie`, `wdx-tie`, `shard-check`. §0.2 has the detail and the gate design.

### v1.5a — ✅ **THE SHIM WIRING — DONE 2026-08-02.** Handoff §0.9 is the record

**Every net streamed R34's augmentation**; each now streams its own, with no fallback anywhere in
the path. It cost what it was scoped at — wiring, not a build: one field, one generator script, one
gate, no render change and no new op.

⚠ **This plan's proposed gate was half of the right one.** It said: `SHIM_HASH` twice per net
(determinism), plus a DIFFERENT digest between two nets whose configs differ. Both landed — but the
*other* direction turned out to be the sharper check: **two nets whose configs AGREE must produce
the SAME digest**, and mnv2 vs R34 is exactly that pair (mnv2 sets `useAutoAugment := false` and
nothing else, so its shim differs from R34's in one comment line). Discrimination without an
inertness peer cannot tell "the streams differ" from "the streams are noisy". Same shape as every
A-vs-A determinism floor in this repo, in the data path.

⚠ And the digest gate would **not** have caught the pre-fix defect on its own: five nets pointed at
one file hash identically, which is indistinguishable from five identical configs. What discriminates
is the **partition** — each net's call sites against its own config's flags — with the digest as the
measured consequence. Gate the partition, not the count, one layer over again.

### v1.5b — ▶ **THE BATCHED-INDEX MOVE, then stochastic depth on ViT/ConvNeXt**

⚠ **This tiering used to read "ConvNeXt/ViT need the §2b batched-index move first" as a footnote on
the stochastic-depth row. It is not a footnote, it is the whole cost.** Measured, distinct AST forms
per renderer:

| renderer | batched | per-example |
|---|---|---|
| EfficientNet · MobileNetV2 · ResNet-34 | **28 · 21 · 18** | 1 each |
| **ViT** | 4 | **14** |
| **ConvNeXt** | 2 | **8** |

Every net that HAS stochastic depth is in the batched world; these two are not.

⚠⚠ **The requirement is STRUCTURAL.** The mask is per-EXAMPLE (`sⱼ` for example `j`); in a
per-example-indexed AST a node denotes ONE example (`Vec n`) and `pretty B` lifts it — the node
cannot see `j`. §4's descriptor rule is exactly that: `den` is `batchMap N (denOp op)`, one FIXED
function across the batch. ⚠ And **the emit would typecheck** — `pretty B` already emits
`tensor<Bxn>`, so a `broadcast_in_dim %mask, dims = [0]` + multiply compiles, trains and descends
with no faithful `den` behind it. That is `swishBack`/`selectPos`'s shape, and those needed their
own constructors holding the whole-batch `x`.

So: 14 forms on ViT, 8 on ConvNeXt, each a batched peer + cert + emit tie. §2b did this for R34 and
is the largest single thread in the handoff. ⚠ The COUNTS are measured; the per-form effort is not —
cost it before committing to a session count.

**Once it lands SD is nearly free**: `dropPathB`, its cert (`layerScale_has_vjp` verbatim), the
driver and the DP mask-shard gate all exist already.

### v1.5 — ✅ **stochastic depth: DONE on EfficientNet single-device, 2026-08-02.**
`planning/stochastic_depth.md`. One `SHlo` op whose **VJP is `layerScale_has_vjp` verbatim** (no new
certificate; the backward emits the same constructor), a host-drawn per-example scale as an ordinary
non-resident input, and `LEAN_MLIR_VARIANT=adamdrop` trains. The exact gate, under the det shim:
at keep = 1 the op is the identity, so `adamdrop` must train what `adam` trains — **0 of 4,020,358**
against a **0** floor, control at norm-rel **1.89**.

⚠ **Two gates still owed**, and they cover the op's INTERIOR rather than its endpoints: the
known-answer tie (`branch · scale` at a chosen scale, host-computed) and the all-zero-mask control
(a zero scale on one site must kill that branch, which pins the site to the residual branch). ~1 h,
harness work, no driver change.

⚠ **ConvNeXt and ViT need the §2b batched-index move FIRST** — they render at the per-example index,
where a per-example mask is §4's descriptor trap. That move is chapter-sized and worth doing on its
own merits; it is not a stochastic-depth prerequisite to be rushed.

⚠ §5b's DP question is **deferred, not solved**: the duplicated-batch gates are blind to a
mis-sharded mask and `shard-check` needs linearity in the gradient, which RMSProp does not have.
Single-device defers it exactly as ConvNeXt-single-device would have.

### v2 — **bf16.** The ONLY gap left on R34 and mnv2. ×1.76 measured on ares, and residency doubled
what it is worth (1.21× → 1.56×, because transport was masking the arithmetic). Needs
`conv_close_mixed`; `planning/bf16_renderer.md`; 4-6 sessions.

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
