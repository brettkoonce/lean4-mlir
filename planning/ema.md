# ema.md — the EMA weight shadow for the verified trainers

> ## ✅ BUILT FOR CONVNEXT, SINGLE-DEVICE, 2026-08-02 — design A, as recommended below
>
> `verified_mlir/convnext_ema_train_step.mlir`, `LEAN_MLIR_VARIANT=ema`. **Zero new `SHlo` ops**, as
> §3 predicted. Interface arithmetic closes exactly: **727 in / 725 out** = 545/543 + 180 (the
> shadow region) + 2 (`%emad`/`%oemad`).
>
> | gate | result |
> |---|---|
> | gate 1 — every committed artifact re-renders | **0 lines of diff** (writers FORCED via `lake env lean`) |
> | ⭐ **`decay = 0` ⇒ shadow ≡ live weights** | **BIT-IDENTICAL, 0 of 27,826,282 params differ** |
> | ⭐ **known answer**, `(ema@.05 − θ)/(ema@.9999 − θ) = d_A/d_B = 0.5` | **8.7e-7** at well-conditioned coords; residue tracks f32 cancellation decade-for-decade |
> | **control** — no warmup correction (`d` = 0.9999 not 0.1) | **0.90**, i.e. 252× the tie. This gate tests the correction DIRECTLY |
> | **checkpoint size guard** on a forged 3-region file | throws, rc=1, with the fix in the message |
> | **residency**, 4 regions | ✅ free, as predicted — `holds 720 parameter tensors (424.6 MB)` = 4×180 |
> | ⭐ **the shadow TRACKS THEN EXCEEDS**, 4 epochs | 48.25/57.35/63.03/**67.77%** vs live 39.95/46.75/56.23/**58.29%** — never near chance |
> | train loss vs the AdamW peer | 2.83/2.05/1.62/1.35 vs 2.89/2.05/1.61/1.37 — the EMA op does not perturb the optimizer |
>
> ### ✅ AND THE DP PEER, same day — `convnext_emadp_train_step.mlir`, 2 replicas
>
> One `#eval` (both `replicas` and `ema` were already renderer parameters). 727 in / 725 out — the
> collective adds no arguments — 180 collectives, 0 MALFORMED. `convnext-dp-check` generalised the
> way the mnv2/enet peers were (`DP_VARIANT{,_DP}`/`DP_REPLICAS`, plus 4-region blob support), and
> it still reproduces its committed AdamW result with no arguments.
>
> | | result |
> |---|---|
> | duplicated-batch identity, 2 replicas | **shadow region BIT-EXACT 27,826,282/27,826,282**; `%loss` bit-exact; gradient norm-rel **9.1e-9**; all 111,305,133 floats |
> | sum-not-mean control | **0.942685**, rc=1 |
>
> ⚠ **The collective and the shadow do not interact** — `all_reduce` is on the gradient, upstream of
> the AdamW triple; the EMA reads θ', the triple's output. So the gate is not circular: what it
> checks is that the 4th region is threaded identically on both paths, which an arity check cannot
> see (both renders *have* the region; the question is whether it carries the same values).
>
> ### ⚠⚠ AND A GATING LESSON THE CONTROL HANDED OVER: NEVER GATE ON THE SHADOW
>
> The sum-not-mean control moves the regions by wildly different amounts:
>
> | region | control moves it by |
> |---|---|
> | `m` (the gradient) | **0.942685** |
> | θ | 1.95e-4 |
> | **`ema` (the shadow)** | **1.00e-4** — sitting exactly ON a 1e-4 gate |
>
> §3's standing rule is *gate the gradient, never θ*, because Adam's update is scale-free and θ
> lands near 1e-4 whether or not anything is wrong. **The shadow is θ's low-pass filter, so it is
> that failure one level worse** — ~9,400× attenuated against the gradient here, and it would have
> waved a 2× gradient error through by landing at the gate boundary. Any future EMA gate on any net
> must read `m`. The harness does; this is a note for whoever adds the next one.
>
> ### ✅ AND EFFICIENTNET, 2026-08-02 — `emarms`, the reference's ACTUAL recipe
>
> `verified_mlir/efficientnet_emarms_train_step.mlir` — **RMSProp + exp decay + EMA**, which with
> stochastic depth is `efficientNetB0ImagenetConfig` entire. 742 → **957 in / 955 out** (+213 shadow
> +2 scalars); gate 1 at 0 diff lines; **`decay = 0` ⇒ shadow BIT-IDENTICAL, 0 of 4,020,358**.
> The EMA op is emitted at the CALL SITE, so one insertion serves both `enetAdamOne` and
> `enetRmsOne` — a copy in each would be the double-writer disease across an optimizer axis that
> already exists.
>
> **§5c's `ema_bn` is real, and now measured rather than quoted.** 3 epochs, Imagenette 224:
>
> | | epoch 1 | 2 | 3 |
> |---|---|---|---|
> | live weights (`rms`) | 38.50% | 53.30% | 64.99% |
> | **EMA shadow + EMA-lagged BN stats** | **44.13%** | **61.35%** | **69.25%** |
> | EMA shadow + LIVE BN stats *(control, `LEAN_MLIR_EMA_BN=0`)* | 37.04% | 44.03% | 56.87% |
>
> The shadow **tracks then exceeds** from epoch 1 (+5.6/+8.1/+4.3), and pairing it with live stats
> costs **7.1/17.3/12.4 points** — worse than not using EMA at all at epochs 2-3. ⚠ Scope the
> reference's wording honestly though: it says this pairing *"blows up early eval"*, and at this
> scale it **degrades** rather than collapses. The mismatch is real and large; "blows up" belongs to
> a more extreme regime than Imagenette-3-epochs.
>
> ⚠⚠ **A BUG CAUGHT BEFORE IT SHIPPED, and it is a naming interaction worth remembering.** The
> RMSProp predicate was `variant.startsWith "rms"` — and the RMSProp+EMA variant is **`emarms`**,
> which does not start with `"rms"`. A prefix test silently classifies it as non-RMSProp, so the
> mean-square would have initialised to **0 instead of 1.0** — precisely the defect the RMSProp
> driver work exists to fix, reintroduced through a variant name. Both predicates are substring
> tests now. **When one string encodes two independent axes, prefix tests stop working and fail
> quietly.**
>
> ### ✅ AND VIT, 2026-08-02 — the third and last EMA net. **The scorecard is 3 of 3.**
>
> `verified_mlir/vit_ema_train_step.mlir`, `LEAN_MLIR_VARIANT=ema`. Zero new ops, fourth time.
> **807 in / 805 out** = 605/603 + 200 (the shadow) + 2 (`%emad`/`%oemad`). The cheapest of the
> three exactly as scoped: LayerNorm ⇒ `hasBn` is false ⇒ the driver's whole `ema_bn` arm is
> skipped rather than special-cased, so it is the parameter shadow alone. **The driver needed
> nothing** — `nRegions`/`nScalars` and the size guard were already generic.
>
> | gate | result |
> |---|---|
> | gate 1 — all 12 committed `vit*`/`vitin*` artifacts re-render | **0 lines of diff** (writers FORCED via `lake env lean`) |
> | ⭐ **`decay = 0` ⇒ shadow ≡ live weights** | **BIT-IDENTICAL, 0 of 5,526,346**, and non-vacuous (`m` non-zero on all 5,526,346, so a real step happened) |
> | ⭐ **ratio known answer** `(ema@.05−θ)/(ema@.9999−θ) = d_A/d_B = 0.5` | **5.96e-5** norm-rel |
> | **control** — no warmup correction (predicted 0.050005) | **8.999**, i.e. **150,995×** the tie |
> | **control 0** — θ and `m` bit-identical across the two decays | ✅ required, else the ratio compares two different θ trajectories |
> | **checkpoint size guard** on a forged 3-region file | throws, rc=1, with the fix in the message |
> | **residency**, 4 regions | **0 of 88,421,536 bytes**, floor 0, both controls fire (init 39,687,902 · staleness 54,590,705). Banner: `holds 800 parameter tensors` = 4×200 |
> | ⭐ **shadow TRACKS THEN EXCEEDS**, 4 epochs | 42.37/50.65/54.34/**56.61%** vs live 40.64/42.93/48.41/**51.64%** — exceeds from **epoch 1**, never near chance (10%) |
> | train loss vs the AdamW peer | **IDENTICAL to all six decimals, all four epochs** |
>
> **▶ The loss row is stronger than it looks and it is what makes the accuracy pair a CONTROL.**
> ConvNeXt's peer losses were close but not equal (2.83/2.05/1.62/1.35 vs 2.89/2.05/1.61/1.37);
> ViT's are *identical* — 2.102756 / 1.791624 / 1.658023 / 1.578948 on both sides — across two
> different HLO programs (805 outputs vs 603). So the shadow-vs-live comparison is one θ trajectory
> read two ways, not two runs that merely look similar, and the EMA op provably perturbs nothing.
>
> ### ⚠⚠ THE RATIO GATE NEEDED ITS INSTRUMENT CONDITIONED FIRST, and that is the reusable part
>
> The gate measures `ema₁ − θ₁ = d₀·(θ₀ − θ₁)`, i.e. a difference of two nearly-equal f32 numbers.
> At ViT's **step 1** the warmup LR is `3e-4/(5·295) ≈ 2e-7`, which puts `ema − θ` at roughly **one
> ULP of θ** — so the first run of this gate read **1.96e-2** and its control only **455×** away.
> Nothing was wrong with the render; the instrument could not resolve the quantity. Re-run at
> `LEAN_MLIR_BASE_LR_U=100000` (a knob added here, the `r34-mom-tie` move) the conditioning goes
> **1.4e-5 → 3.6e-2** median and the same gate reads **5.96e-5** against a **150,995×** control.
>
> ⚠ ConvNeXt did not hit this because its baseLR is 1e-3 with a 3-epoch warmup against ViT's 3e-4
> with 5 — a **50× larger** step 1. **A gate ported between nets inherits the source net's
> conditioning, not the target's.** §2j's rule in a fourth place: check the instrument can resolve
> the thing you are measuring before reading the answer off it.
>
> **The residue is cancellation, not a wrong formula, and it is measured rather than asserted:**
> bucketed by `|ema−θ|/|θ|` the median ratio is **0.500000000 in every bucket** while the max
> deviation tracks conditioning decade for decade — 2.06e-6 / 1.60e-5 / 8.25e-5. A wrong formula
> would be uniform across buckets (§2f-bis: fp conditioning is LOCAL, a different function GLOBAL).
>
> ⚠ **An `ema` run on Imagenette is a GATE VEHICLE, not a matched pair.** `vitTinyConfig` (this
> trainer's reference) sets **no EMA at all**; 0.99996 is `vitTinyImagenetConfig`'s DeiT value, and
> it is carried as the knob's default so a future ImageNet pair wants no edit. Do not describe the
> 4-epoch numbers as a reference comparison — and read them as a delta, since ViT's 80-epoch
> Imagenette result (71.31%) is the weakest of the five.
>
> ⚠ **The predicate check, run rather than assumed** (the `emarms` lesson): all seven ViT AdamW
> spellings read `rmsprop=F, emaOn=F` ⇒ 3 regions; `ema`/`emadp` read `F/T` ⇒ 4; and `emarms` still
> reads BOTH. ViT is AdamW-only so there is no second axis today — if one is ever added, make both
> predicates substring tests first.
>
> ⚠ **Still not done**: ViT's and EfficientNet's DP peers (`emadp` — `vitAdamVariant 32 2 true`
> names it, nothing renders it), and any long run. The 3-4 epoch smokes are descent evidence, not
> accuracy numbers.
>
> The scoping below is kept as written; the measurement calibrates against it.

---

## 0. The original scope (2026-08-02)

**Written 2026-08-02**, as the peer of `planning/stochastic_depth.md` so the two can be compared
before either is built. `planning/recipe_gaps.md` files EMA as **Tier C**, *"needs a shadow buffer
and an update per step, plus eval reading the shadow. Also ~a day."*

**That estimate is stale, and by exactly one commit: §2d.3.** It was written before device-resident
parameters landed. Under residency θ never reaches the host between epochs, so the "shadow buffer
updated per step" it describes would reintroduce the 260 MB per-step round trip that §2d.3 removed —
55% of an R34 bs32 step. EMA is no longer a driver-only item; **where the update runs is now the
whole design question.**

Nothing here is built.

---

## 1. What the reference actually does — read, not assumed

`jax/Jax/Codegen.lean:2439-2461`, plus its call sites:

```python
EMA_DECAY = 0.9999                                    # ENet-B0; ViT/DeiT uses 0.99996
def ema_update(ema, params, step):
    d = jnp.minimum(EMA_DECAY, (1.0 + step) / (10.0 + step))
    return jax.tree.map(lambda e, p: d * e + (1.0 - d) * p, ema, params)

ema_params = params        # the shadow starts AT the weights (fresh or resumed)
ema_bn     = bn_state      # ...and so does a shadow of the BN running buffers
...
    ema_params = ema_update(ema_params, params, _global_step)
    ema_bn     = ema_update(ema_bn,     bn_state, _global_step)
```

Five facts, each load-bearing:

1. **The update is `d·ema + (1−d)·θ`** — an exponential moving average of the *weights*, decoupled
   from the optimizer. The reference says so in as many words: *"Kept decoupled from the optimizer
   so the three train_step variants above stay untouched."*
2. ⚠ **The decay is TIME-VARYING**, `d = min(decay, (1+t)/(10+t))` — TF's
   `ExponentialMovingAverage(decay, num_updates)`. It ramps from 0.09 at `t=0` to the nominal decay
   once `t ≫ 10`. This is not a nicety; see §2.
3. **Eval AND checkpoints read the shadow**, not the live weights (`evalArgs`, `params_to_file` at
   `:2829/:2904/:2911`).
4. **The BN running buffers get their own shadow** (`ema_bn`), because *"eval pairs EMA weights with
   EMA-lagged BN stats, avoiding the weights/stats mismatch that blows up early eval"*. Applies to
   EfficientNet (49 BN layers); ConvNeXt and ViT are LayerNorm and have none.
5. **The shadow is part of the resume state** (`save_train_state` carries `ema_params`/`ema_bn`
   alongside the optimizer moments) — *"Adam m/v and the EMA shadow survive, not just the weights."*

Which nets: **ViT (0.99996), ConvNeXt, EfficientNet (0.9999)**. R34 and mnv2 do not use it.

---

## 2. ⚠⚠ THE WARMUP CORRECTION IS REQUIRED AT OUR SCALE, NOT OPTIONAL

This project's memory carries an EMA warmup bug. **The reference has already hit it, measured it,
and fixed it**, and its writeup is worth more than the memory note:

> Measured on MNv4-Conv-M 100ep (2026-07-31): decay 0.9999 = τ 10k steps, but gradAccum 8 leaves
> only 312 optimizer steps/epoch = 31.2k total = 3.1 τ. At epoch 66 the shadow still held **12.8%
> init** and scored **0.00% top-1**, while the LIVE weights scored **70.48%** on full 50k. Eval and
> `.bin` checkpoints both read the EMA, so the run looked like a total failure when only the shadow
> was broken.

The mechanism: the shadow starts at `params`, which at `t=0` *is* the random init, and plain decay
removes that init only as `decay^t`. Init and trained weights are not linearly connected — there is
a loss barrier between them — so a blend of the two is **not degraded, it is at chance.**

**▶ Run the arithmetic for the runs we would actually do, and it lands inside the failure regime:**

| config | optimizer steps | τ at decay 0.9999 | residual init weight |
|---|---|---|---|
| Imagenette, 80 ep × 295 steps | 23,600 | 10,000 | `0.9999^23600` = **9.4%** |
| the reference's failed run | 31,200 | 10,000 | 12.8% → **0.00% top-1** |
| ENet-B0 ImageNet 350 ep | ~1.75M | 10,000 | 175 τ ⇒ **0** |

**So a verified Imagenette EMA run WITHOUT the warmup correction would score at or near chance, and
it would look like a broken feature rather than a broken shadow.** Any implementation must carry
`d = min(decay, (1+t)/(10+t))` from the start, and the gate below must be written to catch exactly
this shape.

---

## 3. The op inventory — `adamMNextF` **is** the EMA. Zero new ops.

The §2k / v1.2 check — enumerate the reference's update against existing ops *at their other
readings* — run a third time, and it pays a third time:

```lean
adamMNext β₁ m g = fun i => β₁ * m i + (1 - β₁) * g i      -- AdamStep.lean:26
ema_update:        d * ema + (1 - d) * θ'
```

They are the **same function**. `adamMNextF` at `(β₁ := d, m := ema, g := θ')` is the EMA update,
and `adamMNextF_faithful` (`StableHLO.lean:2104`) already closes the denotation side by `rfl`.

And critically, the signature is `adamMNextF (mName b1Name ob1Name : String)` — β₁ and (1−β₁) enter
**by SSA NAME, not as literals**. So they can be *runtime scalars*, which is exactly what the
reference's time-varying `d` needs. The `%lr`/`%bc1`/`%bc2` tail already establishes the pattern.

> **Render-side cost: no new `SHlo` constructor, no new `den`, no new faithfulness theorem, no new
> VJP.** One extra `pretty` call per parameter in the tail, consuming two new runtime scalars.

---

## 4. The three designs, and which one to take

| | where the update runs | render change | residency | trust |
|---|---|---|---|---|
| **A. in-graph** | `adamMNextF` per param, in the train step | ⚠ yes — a 4th blob region | ✅ **free** | ✅ certified arithmetic |
| **B. shim-side** | a device axpy over the retained buffers in `pjrt_ffi.c` | none | ✅ preserved | ⚠ trusted C carve-out |
| **C. host-side** | `F32.ema` in the driver, as `Train.lean:942` already does for the unverified path | none | ⛔ **destroys it** | ✅ |

**C is disqualified by §2d.3.** It needs θ on the host every step. At decay 0.9999 you cannot batch
it to once per epoch either — that is a different filter, not an approximation of the same one.

**▶ Recommended: A, and the reason is structural rather than aesthetic.** The resident shim's
contract is *"inputs `[res_in, res_in+n)` and outputs `[res_out, res_out+n)` are the same tensors one
step apart, and it refuses unless the element counts agree tensor for tensor"* (§2d.3). A fourth
parameter region that is both an input and an output **satisfies that contract by construction** —
`nResident` goes from `3 × P` to `4 × P` and nothing in the C layer changes. Residency supports
this design for free, and it is the only one of the three that keeps the arithmetic inside
`pretty(provenGraph)` where the rest of the optimizer lives.

B is the fallback if the blob-layout change proves too broad — but note it puts an arithmetic update
in the trusted shim, and the mitigating argument ("the shadow is eval-only, so a wrong EMA cannot
corrupt trained parameters") is *also* what makes it invisible to the `[θ|m|v]` residency gate. It
would need its own gate either way, so B buys less than it looks.

---

## 5. What it actually costs: the blob's fourth region

**5a. The layout.** `[θ|m|v]` becomes `[θ|m|v|ema]`, and the scalar tail `[lr, bc1, bc2]` becomes
`[lr, bc1, bc2, d, 1−d]`. Concretely in `VerifiedTrain.lean`:

| today | becomes |
|---|---|
| `nResident := (3 * net.paramShapes.size)` (`:726`) | `4 *` |
| `mvBytes := 3 * net.nParams * 4` (`:768`) | `4 *` |
| `scalarSlots ← F32.const 3 0.0` (`:839`) | `F32.const 5` |
| `F32.write3 pbuf (3*nParams) lrt bc1 bc2` | a 5-slot write at `4*nParams` |
| `adamShapes := packShapes (paramShapes ++ ×3 ++ #[[],[],[]] ++ bnStats)` | `×4`, `#[[],[],[],[],[]]` |
| eval reads `thetamv.extract 0 pBytes` | reads the **ema** region instead |

**5b. ⚠ THE CHECKPOINT FORMAT CHANGES, and this is the sharpest trap here.** Checkpoints are the raw
`[θ|m|v]` blob with no header and no fingerprint. A 3-region file read by a 4-region driver does not
fail — it **misaligns every parameter** and resumes silent garbage. §4 already records that "a
checkpoint OUTLIVES the artifact it was trained on, and nothing notices"; this makes that worse,
because the size changes by exactly 33% and nothing checks it. **A size guard is mandatory**, and it
is two lines: `expected = 4 * nParams * 4`, refuse otherwise with the fix in the message.

**5c. The BN shadow is nearly free, and that asymmetry is worth knowing.** `runningBnStats` already
lives on the host, already crosses per step (`F32.blit` in, `extract` out), and is already EMA'd
there with `F32.ema` at momentum 0.1. `ema_bn` is a second `F32.ema` call on a small buffer — no
residency implication at all. So **EfficientNet's extra requirement costs almost nothing**, while
the param shadow is the whole job. That inverts the naive reading that the BN net is the hard one.

**5d. Blast radius.** Smaller than stochastic depth's, and differently shaped: it touches the
**driver and the blob** rather than the graph *bodies*. Train steps for the three nets gain a region
(the same ~26 artifacts as stochastic depth minus the enet-RMSProp ones if EMA is not wired there);
**the forwards do not change at all** — eval simply reads a different slice of the same blob. The
prefix audit is untouched, which is the opposite of stochastic depth's situation.

---

## 6. Gates

| gate | what it establishes |
|---|---|
| **`useEMA = false` re-renders every artifact byte-identically** | the threading is inert — gate 1, strong form |
| ⭐ **`decay = 0` ⇒ the shadow is BIT-IDENTICAL to the live weights** | `d = min(0, ·) = 0 ⇒ ema' = θ'`. An exact endpoint, free, and it pins the wiring: a shadow reading the wrong slot fails immediately |
| **known answer over 2 steps**: `ema₁ = d₀·θ₀ + (1−d₀)·θ₁` with `d₀ = min(decay, 1/10)`, host-computed | the update AND the warmup correction — the `rms-tie` / `r34-mom-tie` construction |
| **control**: drop the warmup correction (plain `decay`) | the known-answer gate fires; verify red |
| ⭐ **the §2 failure shape, directly**: EMA val must **track then exceed** the live weights, never start near chance | the memory's bug and the reference's measured 0.00%. ⚠ A run too short to distinguish them proves nothing — size it against τ |
| **checkpoint size guard fires** on a 3-region file | §5b |
| **residency gate** at `nResident = 4·P` | the 4th region rides the resident path bit-identically |
| ⚠ **eval-shadow gate** | the `[θ|m|v]` gate is blind to eval-only state — this is `eval_residency_gate.sh`'s hold-mode problem exactly, and its answer (compare reported accuracy epoch by epoch, refuse as VACUOUS if it never moves) transfers |

---

## 7. Cost, and how it compares to stochastic depth

| | **EMA** | **stochastic depth** |
|---|---|---|
| new `SHlo` ops | **0** (`adamMNextF` is the update) | 1 (`selectMidB`-shaped) |
| new certs / VJPs | **0** | 1 (trivial — VJP is itself) |
| graph BODIES change | no — a tail addition | **yes**, on every residual branch |
| forwards change | **no** | yes (to save the prefix audit) |
| prefix audit | untouched | ⚠ needs the ones-mask design to survive |
| driver | ⚠ **blob layout + checkpoint format** | ⚠ new random input class + seeding |
| shim / C | none | ⚠ shard mask |
| DP gate | existing ones work | ⚠ **no working construction for EfficientNet** |
| known-answer gate | ⭐ exact endpoint at `decay = 0` | needs a host-computed reference |
| **what it buys** | ConvNeXt's headline **is** the EMA number (75.93%) — without it that pair is not comparable at all | a regulariser; ConvNeXt is visibly overfitting, so measurable at Imagenette scale |

**▶ EMA is the better first move, and by a clear margin.** It has no open design question (stochastic
depth's DP sharding gate has one, and it is a prerequisite); it needs no new proof obligation at all;
it has a *free exact* gate at `decay = 0`; and it is the one of the two that a reference number
actually depends on — ConvNeXt's 75.93% and EfficientNet's 72.31% are both EMA-evaluated, so no
amount of stochastic depth makes those pairs comparable while EMA is missing.

**Suggested slice: ConvNeXt, single-device, design A.** LayerNorm ⇒ no `ema_bn` to carry, so it is
the param shadow alone; four artifacts; and the `decay = 0` gate plus the 2-step known answer can
both be written before the feature, which is §2d.3's phase-3 move that paid off there.

⚠ **The one thing that will bite**: §5b's checkpoint format. Write the size guard first, and move
every existing `<slug>_*_ckpt_xla.bin` aside before the first run.


---

## ✅ THE DP PEERS ARE GATED (2026-08-02) — and "never gate on the shadow" now has three nets

`convnext_emadp` was gated when it landed; `vitin_emadp128x4` and `enetin_emarmsdp64` shipped in
v1.2c as a carry-forward and were never checked. Both now are, at **4 replicas**, every region
bit-exact on the duplicated-batch identity — **22,869,669** floats on ViT and **21,196,213** on
EfficientNet (the latter including the 4th region *and* 49 BN layers). Sum-not-mean controls fire at
**2.9647 / 2.3915**. No new renders were needed; the single-device peers already existed at the
matching per-device batch.

⚠ Both `dp-check` harnesses had to learn the **4-region blob and 5-slot scalar tail** — only
ConvNeXt's knew it. The predicate is a **substring**, not `startsWith`, because EfficientNet's
recipe is `emarms` and that is the exact axis where a prefix test already failed here.

### ⚠⚠ The rule, re-measured on two more nets and SHARPER than when it was written

*Never gate on the EMA shadow.* In the same control runs:

| | gradient `m` | θ | **shadow** | ratio |
|---|---|---|---|---|
| ViT | 2.9647 | 3.73e-4 | **1.94e-4** | **15,000×** |
| EfficientNet | 2.3915 | 6.94e-4 | **5.42e-4** | **4,400×** |
| ConvNeXt (2026-08-02, earlier) | 0.94 | 1.95e-4 | **1.00e-4** | 9,400× |

On all three the shadow is the most damped region — **below θ**, which §3 already says never to
gate. ConvNeXt's landed exactly ON a 1e-4 gate; ViT's at 1.94× it, i.e. it would have *barely*
fired. A gate on the shadow is not merely weaker than one on the gradient, it is weaker than the one
this repo already forbids. Both harnesses REPORT the shadow — a mis-threaded 4th region must be
visible — and neither lets it decide.
