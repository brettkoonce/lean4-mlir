# next_session_rsb_a3.md — LAMB + BCE + accumulation, and then: can RSB-A3 actually be run?

> ⚠⚠ **SUPERSEDED 2026-08-06 by `planning/next_session_a3_the_run.md`.** Everything this file plans
> is BUILT and GATED: the composition renders, compiles, runs and passes both accumulation gates;
> the 160/224 data path works end to end. **This file is now the RECORD of how that happened** — the
> measurements (§4), the three-hardcoded-224s finding (§2), the `accOn` defect (§3), the gate
> generalisation and the BCE/gradcheck gap (§1.4c). Read it for *why* things are the way they are.
> ▶ **For what to DO next, read `next_session_a3_the_run.md`.**
>
> ⚠ Sections still marked ⛔/⚠ below were resolved in place and carry a dated ✅ where they were —
> trust the dated annotations over the original prose, which is left intact so the reasoning that
> turned out wrong (e.g. §3's "gradcheck runs as-is", §2's located-one-layer-too-high warning) stays
> legible rather than being quietly corrected.

**Written 2026-08-05 (late).** The brief for the next session. Supersedes §4 of
`next_session_pipeline_then_r50.md`, which is now the *history* of how the pieces got built; this
file is the *plan* for composing them.

▶ **The frame: every ingredient of RSB-A3 now exists and is individually certified. None of them
are composed.** The session's question is not "can we build LAMB" — it is "does the composition
type-check, does it still pass the gates, and is the run affordable on this box".

---

## 0. WHERE IT STANDS — what is built, what each gate says

| RSB-A3 ingredient | state | the gate that says so |
|---|---|---|
| **the R50 gradient** | ✅ certified | `r50-gradcheck` — 53 conv + 33 BN homogeneity identities ≤ 6.1e-5, adjoint probe per block |
| **gradient accumulation** | ✅ certified | `r50-accum-tie` (machinery, bit-exact freeze) + `r50-accum-shard-tie` (different micro-batches, rel ≤ 1e-6) |
| **LAMB** | ✅ certified | `r50-lamb-tie` — closed form in two `v̂` regimes, three wrong-LAMB controls |
| **BCE-with-logits** | ✅ certified | `r50-bce-tie` — exact closed form at `Wd = 0`, wrong-divisor control misses by 999× |
| **train @160 / eval @224** | ✅ **RUNS end to end** (2026-08-06) | `LEAN_MLIR_RES=160`: trains at 76,800 off the A3 shim, evals at 150,528 off `@resnet50in160_fwd_eval`; 169 ms/step. §2 |
| **mixup / cutmix** | ✅ shipping | `soft-target-tie`, wire v2, `SHIM_MIX=both` |
| **`wdExcludeNormBias`** | ⛔ absent | — |
| **the composition** | ✅ **RENDERED + COMPILES** (2026-08-06) | `resnet50in160_lambaccdp8x64bce_train_step` — 755 outputs, 4 replicas; `.lambAccum k` in `R34Opt`. §1.4 |

Artifacts on disk (all `verified_mlir/`):
`resnet50in_{adam64,adamdp64,adam64bce,lamb64,lamb64bce,acc4x64,accdp4x64,accdp8x64}_train_step`,
`resnet50in_{fwd,fwd_eval}`, `resnet50in160_{lamb64bce_train_step,fwd,fwd_eval}`.

---

## 1. ⭐ THE COMPOSITION IS A TYPING PROBLEM, NOT A SEMANTICS PROBLEM

`R34Opt` is a flat enumeration:

```lean
inductive R34Opt | adamw | heavyBall | lamb | adamwAccum (k : Nat)
```

so `lamb` and `adamwAccum k` are mutually exclusive **by the shape of the type**, not because
anything about the algorithms conflicts. ⭐⭐ **The accumulation trick works verbatim for LAMB, and
this is the load-bearing observation for the whole session:**

* the accumulator is `Gt = akeep·G + g`, upstream of any optimizer — it does not care who consumes it;
* an accumulate micro-batch needs `m' = m`, `v' = v`, `θ' = θ`. The first two come from
  `%b1 = %b2 = 1`, `%ob1 = %ob2 = 0` (already how `.adamwAccum` does it), and **θ' = θ comes from
  `lr = 0` for LAMB too**, because LAMB's parameter step is `sgdParamF θ lr (trust·r)` — at `lr = 0`
  that is `θ − 0·(…) = θ` exactly, with no decay term left running (LAMB's wd is inside `r`, which
  the zero multiplies away).
* ⚠ `lambDirF` reads `%b1..%ob2` as well, so on an accumulate micro-batch it computes a `r` built
  from `β₁·m` rather than the real moment. **That is harmless and must be said out loud**: `r` feeds
  only `lambScaleF → sgdParamF`, and `lr = 0` discards it. Nothing stateful is written.

▶ **So the work is to make accumulation ORTHOGONAL in the type**, not to re-derive it.

### 1.1 The shape to move to

```lean
inductive R34Base | adamw | heavyBall | lamb
structure R34Opt where
  base   : R34Base
  accumK : Option Nat := none        -- `some k` ⇒ the fourth region + %aup/%akeep
  bce    : Bool := false             -- ⚠ SEE BELOW — it is currently a renderer flag, not here
```

⚠ **Cost it before doing it.** `R34Opt` is matched on in `optOne`, `optConstsB`, `r34AdamVariant`,
both renderers' `optLabel`, and every `#eval` call site. That is ~8 sites plus every artifact's
`#eval`. The cheap alternative is `| lambAccum (k : Nat)` — one more constructor, no call-site
churn, and the accumulation logic duplicated between two arms. ▶ **Prefer the structure.** The
duplication is exactly the double-writer failure this repo keeps paying for, and there is already a
third axis (`bce`) sitting in a different place for no reason.

### 1.2 ⚠ `bce` is currently in the wrong place, and it will bite

`bce` is a trailing `Bool` on `resnet50TrainStepFaithfulB`, while `lamb`/`accum` are constructors of
`R34Opt`. Three axes, two mechanisms. The variant name is assembled from `r34AdamVariant` *plus* a
hand-passed `vSuffix := "bce"`, i.e. the name and the flag can disagree and nothing notices.
▶ **Fold `bce` into the same structure and derive the whole variant string from it**, so
`r34AdamVariant` is the single source and `#guard`s can pin it — the way `k` is already pinned.

### 1.4 ✅ WHAT WAS ACTUALLY DONE (2026-08-06) — and why NOT the structure of §1.1

▶ **Shipped: `| lambAccum (k : Nat)`, one constructor — but with the accumulate/apply mechanism
FACTORED OUT into `accumScalarConsts k`, emitted by both accumulation arms.**

⭐ §1.1 preferred the `{base, accumK, bce}` structure, and its stated reason was that a second
constructor would leave *"the accumulation logic duplicated between two arms"*. **Factoring the
block out buys exactly that property** — there is now ONE writer of the `%aup` arithmetic and the
`1/k` folding — at a fraction of the cost §1.1 itself said to count first (~8 match sites plus every
`#eval`, against a 13-artifact byte-identity net). ⚠ The extraction was verified inert the way that
section demanded: **all 27 committed R50/R34 artifacts re-rendered BYTE-IDENTICAL.**

⚠ **What the structure would still buy, and is therefore still open:** §1.2's point that `bce` is a
renderer flag while `lamb`/`accum` are constructors, so the variant string is assembled from
`r34AdamVariant` *plus* a hand-passed `vSuffix := "bce"` and **the name and the flag can still
disagree with nothing noticing**. That hazard is live for exactly the artifact just rendered, whose
name ends in `bce`. It is mitigated only by `#guard`s on the produced names, not closed.

⚠ `optOne`'s `.lambAccum` arm does duplicate LAMB's four-op tail with `gt` substituted for `gr`.
That is a genuine second copy, and the honest reason it was not abstracted is that the two differ
only in one operand — a shared helper parameterised by "which gradient" would be a function of one
argument used twice. ▶ If a third accumulating optimizer ever lands, abstract then.

### 1.4a ✅ IT RUNS — 4 GPUs, 2026-08-06

```
▸ TRAIN RES: 160×160 (slug resnet50in160, d0 76800, shim …_short_shim.py)
▸ GRADIENT ACCUMULATION: k = 8, 4th blob region [θ|m|v|G].
  Micro-batch 256 x 8 = EFFECTIVE BATCH 2048. 40 micro-batches/epoch = 5 updates/epoch
[pjrt_ffi] compiled …_lambaccdp8x64bce_train_step.mlir (755 outputs, 4 replicas) in 11090 ms
  step 0/40: loss=0.834864     ← identical to 6 dp across the accumulate phase:
  step 1/40: loss=0.834864        θ frozen EXACTLY at lr = 0, which is §1's whole argument
  step 2/40: loss=0.834864
Epoch 1/30: loss=0.821412        ← and it descends once the applies fire
```

⭐ **755 outputs is the four-region count**, and the driver supplied four — so §3's predicted first
failure (`startsWith "acc"` packing THREE regions into a FOUR-region graph, "755 outputs, caller
supplied 594 destinations") did **not** occur, because the predicate was fixed first.

### 1.4b ✅✅ AND IT IS CERTIFIED — `r50-accum-tie` on the LAMB accumulator, 2026-08-06

§3's generalisation landed: the peer is a knob (`R50_ACC_PEER`) and the resolution is a knob
(`R50_ACC_RES`), so the gate runs on the composition.

```
R50_ACC_RES=160 R50_ACC_VARIANT=lambacc4x64bce R50_ACC_PEER=lamb64bce

  ① accumulate micro-batches leave [θ|m|v] frozen   76671096/76671096 BIT-EXACT
  ② after 4 micro-steps == one lamb64bce step
      θ'  rel 0.000000   bit-exact 25556975/25557032
      m'  rel 0.000000   bit-exact 25554748/25557032
      v'  rel 0.000000   bit-exact 25557030/25557032
  ⟂ WRONG-k control: θ' rel 0.000043 — misses by 533x the tie
  ✅ CERTIFIED
```

⭐ **This is §1's `lr = 0` argument, measured.** ① is 100% bit-exact across every accumulate
micro-batch, which is the claim that LAMB's parameter step `sgdParamF θ lr (trust·r)` freezes θ
*exactly* at `lr = 0` — including the `r` built from `β₁·m` that §1 flagged as harmless, since it
feeds only the frozen step. ⭐ The LAMB tie is TIGHTER than AdamW's original (θ' 25,556,975
bit-exact against 25,551,446) — the trust ratio is computed from θ and `r` rather than accumulated,
so it contributes no extra rounding.

⚠ The wrong-k control is LIVE but narrower than AdamW's (533× the tie against 1228×), and the reason
is structural rather than a weakness: LAMB's trust ratio RENORMALISES the step, so a `(k−1)/k`
gradient error is partly absorbed by `‖θ‖/‖r‖` before it reaches θ. ▶ Worth knowing if the tolerance
is ever tightened — this control has less headroom on LAMB than the AdamW number suggests.

### ✅✅ AND THE OTHER HALF — `r50-accum-shard-tie` on the composition, 4 GPUs

`r50-accum-tie`'s deliberate blind spot is the DUPLICATED batch: it cannot see whether genuinely
DIFFERENT micro-batches combine correctly. The shard tie is that half, and it now runs on the
composition against the freshly-rendered `lambdp64bce` peer:

```
R50_ACC_RES=160 R50_ACC_VARIANT=lambacc4x64bce R50_ACC_PEER=lambdp64bce   (4 GPUs)

  θ'  rel 0.000000   bit-exact 25556945/25557032
  m'  rel 0.000000   bit-exact 25553898/25557032
  v'  rel 0.000000   bit-exact 25557032/25557032      ← 100%
  ⟂ DUPLICATED-batch control  θ' rel 0.000158 — misses by 2137x the tie
  ✅ CERTIFIED
```

⭐ **`acc(x₁,x₂,x₃,x₄) == lambdp([x₁|x₂|x₃|x₄])`** — a serial accumulator with a folded `1/k` on one
side and a collective `all_reduce` on the other, reaching the same answer through two different
mechanisms. ▶ **Together with §1.4b this closes BOTH halves for the composed optimizer**: the
accumulate/apply switching AND the combination of different micro-batches. That is §5 step 5's
"the point at which lamb+bce+accum is DONE and defensible".

### 1.4c ⛔⛔ `r50-gradcheck`'s TIER 1 B DOES NOT TRANSFER TO BCE — a PRE-EXISTING gap, found 2026-08-06

Pointing the gradcheck at the composition (§3 said it "runs as-is") made it **fail**, and the
failure is not the composition's. Tier 1 B is `⟨g_γ,γ⟩+⟨g_β,β⟩ = 0`, exact in ℝ:

| res | loss | optimizer | accum | tier 1 B worst \|cos∠\| | verdict |
|---|---|---|---|---|---|
| 224 | CE | AdamW | — | 0.000084 | ✅ passes |
| 224 | CE | **LAMB** | — | **0.000061** | ✅ passes |
| 224 | **BCE** | LAMB | — | **0.000757** | ❌ fails |
| 160 | BCE | LAMB | — | 0.000574 | ❌ fails |
| 160 | BCE | LAMB | **k = 4** | 0.000571 | ❌ fails |

⭐ **ONE VARIABLE, ISOLATED.** Holding resolution, optimizer and accumulation fixed, CE → BCE moves
tier 1 B by **12×**. Everything else is exonerated by its own row:
* **LAMB** — 0.000061, *better* than AdamW's 0.000084, and passes outright;
* **resolution** — 160 (0.000574) is BETTER than 224 (0.000757), so the 160 render is not the cause;
* **accumulation** — 0.000571 WITH against 0.000574 WITHOUT, i.e. indistinguishable, and the
  accumulating render is marginally the better of the two.

▶ **The mechanism is in the run header: `‖g‖` is 197.5 under CE and 2.18 under BCE — a 90× smaller
gradient** (base loss 7.82 vs 0.852). Tier 1 B measures a CANCELLATION residual, so at fp32 it grows
relative to a gradient 90× smaller. This is conditioning, not a wrong derivative.

⚠⚠ **AND THE GATE'S OWN CALIBRATION SAYS IT CANNOT DECIDE.** The CE baseline separates two
populations: passing sites worst **0.000084**, the 21 deliberately non-homogeneous CONTROL sites
weakest **0.000545**, tolerance 0.000300 sitting between them. Under BCE the passing sites reach
**0.000757 — ABOVE the weakest genuine violation in that calibration.** The populations OVERLAP, so
no tolerance separates them: tier 1 B can neither certify nor condemn a BCE artifact. §6's rule
applies exactly — *a control that cannot fire is not a control, and finding that out is a result.*
▶ **Do NOT "fix" this by raising `R50_GC_EXACT_U` until it goes green.** That would convert a gate
that cannot discriminate into one that reports success, which is strictly worse than the failure.

✅ **What DOES still hold for the composed artifact**, and it is not nothing:
* **TIER 1 A** (conv-kernel homogeneity) passes — worst 0.000017 over 53 kernels;
* **TIER 2** — the direct finite-difference fit — **PASSES on the composed artifact**: all 22 groups
  agree to `rel ≤ 0.230` against a tolerance of 0.30, with the doubled-gradient control live at
  0.997. That is the check that actually compares the recovered gradient to `(L₊−L₋)/2`, and it is
  loss-agnostic;
* the composed artifact's backward is the SAME backward as `lamb64bce`'s — only the optimizer tail
  differs — and `r50-accum-tie` ties the two numerically at rel 0 (§1.4b).

⚠ **This gap is OLDER than the composition and belongs to the BCE renders**: `r50-gradcheck`
defaulted to `adam64` (CE) and had never been pointed at `adam64bce` or `lamb64bce`. `r50-bce-tie`
certified the BCE LOSS in closed form; nothing had gradchecked a BCE artifact until now. ▶ Owed: a
tier 1 B formulation that is conditioning-robust at small ‖g‖, or an explicit record that tier 1 B
is CE-only and BCE rests on tier 1 A + tier 2.

### 1.5 ⛔⛔ k = 8 DOES NOT DIVIDE THE EPOCH — found by running it, 2026-08-06

The driver refuses outright:

> `5004 micro-batches per epoch is not divisible by k = 8: the last cycle of every epoch would
> apply 4 micro-batches' gradient still divided by 8.`

**5004 = 2²·3²·139**, so `k ∈ {2, 3, 4, 6, 9, 12, 18, 36}` divides it and **8 does not** — and 8 is
exactly what effective batch 2048 requires at `gbs = 256`. ⚠ This is a property of ImageNet's
1,281,167 against a 256 global batch, not of this render, and it would have bitten the real run on
its first epoch boundary.

▶ **The fix is `LEAN_MLIR_G2_STEPS=5000`** (= 8 × 625): it drops **4 micro-batches = 1,024 images
per epoch, 0.08%**, which is far below any effect on the result and is the standard "drop the
remainder" the reference's own `drop_remainder` already applies to val. ⚠ Do NOT instead pick a
`k` that divides 5004 — `k = 6` would give effective batch 1536 and `k = 9` 2304, i.e. it would
silently change the recipe's batch to make the arithmetic tidy.

### 1.3 The target artifacts

| artifact | what it is |
|---|---|
| `resnet50in160_lambacc8x64bce_train_step` | ⭐ **the recipe**: LAMB + BCE + k=8, 4 replicas × bs64 ⇒ effective **2048**, at 160 |
| `resnet50in160_lambacc4x64bce_train_step` | single-device peer at k=4, so the accum gates can run |
| `resnet50in160_fwd`, `resnet50in160_fwd_eval` | ✅ already rendered (train @160, eval @224) |

---

## 2. THE DATA PATH AT 160 — ✅ TRAIN SIDE BUILT 2026-08-06, eval side is all that remains

1. ✅ **A 160 `VerifiedNetSpec`** — `resnet50Imagenet160Verified` in `LeanMlir/VerifiedNets.lean`,
   `d0 = 76800`. ⭐ `layers` is **shared by construction** with the 224 spec rather than re-typed, so
   `toSpecs` is identical *derivationally*, and a `#guard` pins that equality plus 25,557,032 params
   / 161 tensors / `d0 == 76800`. `TestR50Contract` passes unchanged.
2. ✅ **A 160 shim** — `generated_resnet50_imagenet_short_shim.py`, from the **`short`** recipe
   (`gen_shims.sh` grew a RECIPE column; the six pre-existing shims are byte-size unchanged).
   ⭐ **§2's open question is ANSWERED, in TWO layers, and only one of them was already right:**
   * ✅ the **tf.data pipeline** applies `trainRes` only inside `_imagenet_decode_random_crop_flip`
     (train), while eval goes through `_imagenet_decode_center_crop` at `_IMG_SIZE = 224`. One shim,
     A3's split — `_TRAIN_SIZE = 160` on the train resize, `0.950000` centre-crop on eval.
   * ⛔ the **batch-wrapper around it did NOT**, and this is §2.2's warning landing one layer lower
     than it was written. `flat = 3 * _IMG_SIZE * _IMG_SIZE` framed the wire, and `H = W = _IMG_SIZE`
     framed CutMix, both hardcoded — so the pipeline handed over REAL 160 pixels and the wrapper
     described them as 224. Fixed 2026-08-06 (`_RES = _TRAIN_SIZE if training else _IMG_SIZE`).

   ⭐⭐ **AND THIS IS THE SECOND INSTANCE OF §6's FIRST BULLET.** A resolution parameterisation is
   not done when it elaborates: this one passed a clean `lake build`, every `#guard`, the contract
   gate, `shim_wiring_gate.py` end to end, AND an XLA compile of both 160 artifacts — and still
   could not read one batch. ▶ The failure was loud, which is the only reason it cost minutes:
   `cannot reshape array of size 4915200 into shape (64,3,224,224)` — and **4,915,200 = 64 × 3 ×
   160² is the tell**, real 160 data with the wrong width asserted over it, followed by
   "pipe closed after 19660800 of 38535168 bytes". ⚠ Neither number is derivable from any static
   check in the repo, because both sides of the mismatch are *runtime* framings of the same buffer.
3. ⚠ **THE DRIVER FED ONE IMAGE SIZE — from THREE separate hardcoded literals, not one.** This is
   the item the original brief called "the piece most likely to be underestimated", and it was:
   the estimate assumed a single `net.d0` threading through. What was actually there:

   | site | what it sizes | fix |
   |---|---|---|
   | `loadData` `.imagenet`, `let flat := 3*224*224` returned as `trainPix` | the TRAIN stream width, taken from the VAL drain's literal | split into `evalFlat` (stays 224) + `trainPix := net.d0` |
   | `spawnShimSharded … "train" gbs (3*224*224)` | what the shim is TOLD to emit | `net.d0` |
   | the read loop's `let flat := 3*224*224` | what the driver READS off the pipe | `net.d0` |

   ⭐⭐ **All three describe the SAME buffer, and each had to agree with the render independently —
   so fixing them one at a time surfaced the identical refusal three times.** ▶ The reusable form:
   *a shape constant that appears N times is N defects, and the count is not visible from the one
   you are looking at.* `grep -n '3 \* 224 \* 224'` was worth more than reading the call chain.
   ⚠ The fourth occurrence is `evalFlat` and it is CORRECT — it is a property of the val split, not
   of the net. Do not "consistency-fix" it.

   ▶ **INERT for all six incumbents, and that is PROVEN, not asserted**: `VerifiedNets.lean`'s
   closing `#guard` block pins `net.d0 == 3*224*224` for every 224 ImageNet net, so each `net.d0`
   substitutes equal for equal there. ⚠⚠ That block FIRES if an `.imagenet` net is ever added at a
   non-224 train resolution — which is the signal that `evalD0` is required before its eval loop can
   be trusted. Do not relax it.

   ✅ **THE EVAL SIDE — `evalD0`, DONE 2026-08-06.** ⭐ **The split was already in the ARTIFACTS**,
   which is what made this small: `resnet50in160_fwd_eval.mlir` declares `tensor<256x150528xf32>`
   (224² eval) while `resnet50in160_fwd.mlir` declares `tensor<64x76800xf32>` (160² train). The
   renderer had been emitting A3's split all along; only the driver's plumbing assumed one width.

   ▶ **Read off the artifact, exactly like `evalBs` already was** — `fwdRenderedShape` returns
   `(batch, d0)` from **one parse of one `%x: tensor<BxWxf32>` declaration**, because two parsers of
   the same text could disagree about the same tensor. Four consumers now size from it:

   | consumer | was | now |
   |---|---|---|
   | the ImageNet **val drain** (`loadData`) | `3*224*224` literal | `evalD0` (passed in) |
   | the eval invoke's `xShape` | `net.xShape evalBs` — bakes `net.d0` | `packXShape #[evalBs, evalD0]` |
   | the eval **slicer** | `… evalBs d0 nEval` | `… evalBs evalD0 nEval` |
   | `evalBs` | already off the artifact | unchanged, now same parse |

   ⚠ `evalD0` is computed **above** `loadData` — that ordering is load-bearing, since the val drain
   allocates and reads at the eval width. ⚠ The slicer was the dangerous one: slicing a 224-drained
   buffer at the train width strides wrongly from the second row on and would have scored a
   plausible-looking accuracy off garbage, **silently**. ▶ Inert for 224 nets by construction — their
   eval render declares `net.d0`, so `fwdRenderedShape` returns `(bs, d0)`.
   ▶ The run now ANNOUNCES `EVAL RES SPLIT: train d0 76800, eval d0 150528` whenever the two differ,
   because nothing else in the log would reveal it. The `LEAN_MLIR_SKIP_EVAL` refusal is retired.
   ▶ **Until it lands the driver REFUSES**: `LEAN_MLIR_RES=160` without `LEAN_MLIR_SKIP_EVAL=1`
   throws, rather than reading a 150,528-float val image into a 76,800-float graph. Refuse-don't-
   fall-back, the same design as `spawnShim`.

⚠ **The original plan here said "smoke at 160-eval-160 FIRST, it needs no driver change". It does
need one** — `_IMG_SIZE = 224` is a hardcoded literal in the generator, so 160/160 needs a
`Jax/Codegen.lean` change. ▶ **Train @160 with `LEAN_MLIR_SKIP_EVAL=1` buys the same separation for
free** and is what was actually run: it proves "the 160 render trains off the 160 data path" without
touching either the driver or the generator, leaving the split as the single remaining unknown.

### 2.1 ✅ THE 160 TRAIN PATH RUNS — end to end, 2026-08-06

`LEAN_MLIR_RES=160 LEAN_MLIR_VARIANT=lamb64bce SHIM_WORKERS=8 LEAN_MLIR_SKIP_EVAL=1`, 1 GPU:

```
▸ TRAIN RES: 160×160 (slug resnet50in160, d0 76800, shim generated_resnet50_imagenet_short_shim.py)
  imagenet shim: … train split, batch 64, 76800 floats/img …, wire v2 soft targets [64x1000]
  imagenet shim: 8 sharded producers (round-robin over batches)
  step 0/20018: loss=0.823093        ← BCE-with-logits, and it descends
  PROBE: 169 ms/step
```

⭐ **RandAugment does NOT stall the producer here: 169 ms real vs 166 ms synth, a 1.8% gap.** The
concern was live — A3's shim adds RandAugment m6 where the 224 shim is RRC+hflip only, which is the
augmentation `scripts/jobs/enet-imagenet-4gpu.conf` blames for EfficientNet's 10× stall — and it was
worth checking rather than assuming. At 8 producers the read is fully hidden.

⚠⚠ **BUT THAT IS A 1-GPU RESULT AND IT DOES NOT TRANSFER UNEXAMINED.** 64 img / 0.169 s = **379
img/s**; a 4-replica step needs 256 img / 0.225 s = **~1,140 img/s, 3× more**. EfficientNet reached
~1,260 img/s on 8 producers with the heavier AutoAugment, so it is plausible — but **plausible is
not measured**, and this is precisely the axis that produced a 10× surprise once already. ▶ Re-run
real-vs-synth at 4 replicas before quoting a 4-GPU wall-clock, and raise `SHIM_WORKERS` if the gap
opens. ⚠ The startup transient is misleading: GPU sat at 0% with all 8 producers alive for the
first minute or so, which looks exactly like a producer stall and is not one.

### 2.2 ⭐ What this exposed in `scripts/shim_wiring_gate.py` — a hole that could not fire until now

`read_config_flags` matched only `def X : TrainConfig where`, so it **silently could not read any
config written as a structure update** (`:= { Base with ... }`) — the form 20+ configs across the
Main files use, *including every `*ConfigShort`*. Unreachable until a verified net ported a
non-`default` recipe, which is exactly what the 160 net is. Now it follows `{ Base with ... }`
chains and resolves inheritance at the **field** level, never on the derived features: `randaugment`
is `useRandAugment ∧ randAugmentGeometric`, so a config overriding only one of the two is mis-read
by any coarser merge. ▶ It demonstrates itself — `resnet50in160` reads `repeated_aug OFF` against
`resnet50in`'s `ON`, which is A3's `repeatedAug := 1` arriving through the chain.

---

## 3. THE GATES — what must be re-run, and what has to be GENERALISED

Every existing gate hardcodes a peer artifact. Composition moves the peers.

| gate | change needed |
|---|---|
| `r50-gradcheck` | ⚠ **NOT "as-is" — needed `R50_GC_RES` AND 4-region packing** (4 param regions, 5 scalars, `%aup=1`/`%akeep=0`, and `g = 10k·m'` not `10·m'`). Done. ⛔⛔ **And this row's claim "tier 1's identities are loss-agnostic and hold for BCE" is FALSE** — tier 1 B loses all discrimination under BCE (§1.4c). Tier 1 A + tier 2 pass on the composition |
| `r50-accum-tie` | ✅ **DONE** — `R50_ACC_PEER` + `R50_ACC_RES`; run on the composition, CERTIFIED (§1.4b). 224/`adam64` regression unchanged |
| `r50-accum-shard-tie` | ✅ **DONE** — same two knobs, `lambdp64bce` peer rendered, run on the composition on 4 GPUs: CERTIFIED, control 2137× the tie (§1.4b) |
| `r50-lamb-tie`, `r50-bce-tie` | ✅ cover their own piece and do not need the composition |
| `r50_dp_render_tie.py` | add the composed (1-replica, 4-replica) pair |
| `TestVariantPredicates` | ⚠ add `lambacc8x64bce` and friends. The marker `acc` is a PREFIX test and `lambacc…` does not start with it — **this WILL misfire** and the fix (a substring test) has to be checked against every concatenation, per that file's own rule |

⚠⚠ **The `accOn` prefix test is the first thing that breaks.** `variant.startsWith "acc"` is false
for `lambacc8x64bce`, so the driver would pack THREE regions into a FOUR-region graph. The G4 gate
refuses (it did exactly this during the accumulation work — *"755 outputs, caller supplied 594
destinations"*), so it fails loudly rather than silently. Still: fix the predicate and pin the
counterfactual before rendering anything.

---

## 4. ▶▶ CAN IT ACTUALLY BE RUN? — the question this session should ANSWER, early

✅ **MEASURED 2026-08-05. The answer is YES at 100 epochs — at 160, and only at 160.** The standing
open item since §3.3 is closed; the verdict and its bar are §4.0.2. Three probes,
`LEAN_MLIR_MAX_STEPS=120` (median of steps 9..120), `PJRT_FFI_RESIDENT=1`, prefetch ON,
`SHIM_WORKERS=8`, GPUs 0,2,3,4:

| config | artifact | ms/step |
|---|---|---|
| 4 replicas × bs64 (gbs 256) | `resnet50in_adamdp64_train_step` | **376** |
| 1 replica × bs64 | `resnet50in_adam64_train_step` | **317** |
| 4 replicas, `LEAN_MLIR_BENCH_SYNTH=1` (no shim at all) | `resnet50in_adamdp64_train_step` | **367** |

⭐ **THE THIRD ROW IS THE ONE THAT LICENSES THE OTHER TWO.** `scripts/jobs/enet-imagenet-4gpu.conf`
records EfficientNet at **2,061 ms/step with `SHIM_WORKERS=1` against 203 with 8** — a 10× that was
entirely the data producer and "read exactly like a correct one". So R50's numbers are worthless
until the same question is asked of them. Synth 367 vs real 376 is a **2.4% gap**: the pipeline is
fully hidden behind compute at `SHIM_WORKERS=8` and **these are compute numbers, not producer
numbers.** ▶ R50's shim does RRC+hflip, not EfficientNet's per-image AutoAugment/RandAugment, which
is exactly the fast/slow split that config identifies across the five nets.

⚠ The probe command in the original draft of this section OMITTED `PJRT_FFI_RESIDENT=1`, which is the
setting `VerifiedTrain.lean:1277` records as having let "a 16 h benchmark and a 26 h production
config diverge for a week". Both numbers above are resident. Use the driver header's env block.

### 4.0 ⭐ THE TWO PROBES DECOMPOSE THE STEP — which is why the 160 answer is a FLOOR, not a guess

Each device does bs64 of compute in **both** rows, so the difference is the collective:

* **317 ms** — per-device compute + optimizer + host patching
* **59 ms** — 4-way allreduce + sync (`376 − 317`), **resolution-independent**: it moves 25.6 M
  gradients whatever the crop is

So with `X` = the resolution-independent part *inside* the 317 (the AdamW update over 25.6 M params,
the blob patching):

```
160 step  =  (317 − X)·(160/224)²  +  X  +  59   =   220.7 + 0.49·X   ms
```

▶▶ **`X ≥ 0`, so the 160 step is ≥ 221 ms** — and the naive FLOPs-ratio number (`376 × 0.51 = 192`)
is wrong because it scales the allreduce, which does not scale. ⚠ 221 is a FLOOR in a second way
too: at 160 the last stage is 5×5 rather than 7×7, so occupancy drops and the real number lands
above the FLOPs ratio, not on it.

### 4.0a ✅ THE 160 STEP IS NO LONGER A BOUND — MEASURED 2026-08-06

The floor below was replaced by a measurement the same session, once §2.1's spec existed. Matched
pair, **one GPU, synth, same optimizer (`lamb64bce`), same batch — only resolution differs**:

| res | artifact | ms/step |
|---|---|---|
| 224 | `resnet50in_lamb64bce_train_step` | **317** |
| 160 | `resnet50in160_lamb64bce_train_step` | **166** |

* ratio **0.524** vs the FLOPs ratio `(160/224)² = 0.510` — measured slightly ABOVE, which is the
  5×5-vs-7×7 occupancy prediction confirming rather than a discrepancy
* ⭐ the 224 row (317, LAMB+BCE, synth) lands **exactly** on the independently measured 224 row in
  the table above (317, AdamW+CE, real data). Two optimizers, two loss functions, synth vs real,
  same number — LAMB+BCE costs the same per step as AdamW+CE, and it re-confirms the data is hidden

▶▶ **4-GPU @160 = 166 (measured compute) + 59 (measured allreduce) ≈ 225 ms/step**, both terms
measured rather than modelled. That implies `X ≈ 9 ms` in the formula below — the single-digit
optimizer cost predicted there. **100 epochs = 500,400 × 0.225 s = 31.3 h**, inside the 40 h bar
with ~22% headroom.

✅ **CONFIRMED ON REAL DATA (§2.2): 169 ms/step at 160, 1 GPU** — 1.8% above the synth floor, so the
compute term is not a synth artefact. Using it instead: `169 + 59 = 228 ms` ⇒ **31.7 h**, which is
the number to plan against.

⚠ The 59 ms is carried over from the 224 pair because gradient volume is resolution-independent
(25.6 M params either way). It is the one term not yet measured AT 160, and it cannot be until a
4-replica 160 artifact exists (§5 step 4). Treat 225 as measured-compute + measured-collective, not
as an end-to-end 4-GPU number.

### 4.0.1 The arithmetic, with the measured numbers

* 1,281,167 ÷ 256 = **5,004 invokes/epoch** (the driver prints `step 0/5004` — 5,004, not 5,005)
* × 100 epochs = **500,400 invokes**, independent of `k` — an accumulate micro-batch runs the same
  fwd+bwd and the same optimizer arithmetic, `lr = 0` only discards the result (§1)

| run | ms/step | 100 ep | 30 ep |
|---|---|---|---|
| @224, 4×bs64 | 376 (measured) | **52.3 h** | 15.7 h |
| @160, 4×bs64 | **225** (166 measured + 59 measured; §4.0.05) | **31.3 h** | 9.4 h |

### 4.0.2 ▶▶ THE VERDICT, against the OPERATOR's bar and not this document's

⚠ This section originally set its own bar at "under ~200 ms ⇒ 28 h". **That bar was never the
operator's** — asked directly on 2026-08-06, the answer was **40 h upper limit**. Re-decided there:

* 40 h ÷ 500,400 invokes = **288 ms/step allowed**
* the 160 floor is 221 ms, so the margin is ~30%. Breaching 288 would need `X ≈ 137 ms` of the
  single-GPU 317 to be resolution-independent, and the AdamW update over 25.6 M params is
  single-digit ms, not 137

| 160 step | 100 ep | vs 40 h |
|---|---|---|
| 221 (floor) | 30.7 h | ✅ |
| 240 | 33.4 h | ✅ |
| 263 | 36.6 h | ✅ |
| 288 | 40.0 h | the bar itself |

▶▶ **A3 at 100 epochs FITS a 40 h budget — but ONLY at 160.** 224 is 52.3 h and stays out at every
bar discussed. ⚠⚠ **So §2's data path is no longer optional work that can slip to another session:
it is the ONLY thing standing between here and an affordable A3**, and it is the section with no
render-side answer — specifically §2.3, the `VerifiedTrain.lean` change giving the eval invoke its
own `evalD0`.

### 4.0.3 What else that leaves

* ⭐ **30 epochs at 160 is ~9–10 h** — a genuine overnight run at the same validation tier
  R34/ImageNet ran, so it is directly comparable. Worth doing FIRST regardless: it is the cheapest
  thing that proves the 160 data path end to end before a 31 h run rests on it.
* **bf16** is the lever the original draft named as the only route to 100 epochs. At a 40 h bar it
  is **no longer required** — fp32 at 160 fits. Still unmeasured, and now an optimisation rather
  than a precondition.
* ⭐ **THE DUTY CYCLE IS ALREADY BUILT, on the verified path.** `scripts/supervise.sh` is the one
  supervisor engine (it replaced the `jax/scripts/supervise_*` copy-paste family); a job is a config
  in `scripts/jobs/`. It carries fixed-schedule rests (`REST_EPOCHS`/`REST_SECS`), **temperature-
  driven rests (`TEMP_MAX`/`TEMP_RESUME`)**, a stall guard, the AER watchdog, and crash-resume off
  the trainer's own `.bin.epoch`. ⚠ There is **no `r50-*.conf` yet** — cnx/enet/mnv2/r34/vit exist.
  Writing one is the deliverable that makes a 31 h run startable, and `enet-imagenet-4gpu.conf` is
  the model to copy (its `PRECHECK` refuses to launch if `PJRT_FFI_RESIDENT`/`SHIM_WORKERS` are
  missing — exactly the two silent-10× knobs).
* ⚠ **Duty overhead has historically been ZERO on this box** — across every past 4-GPU supervise run
  (`/tmp/supervise_r34-imagenet-4gpu*`, 30 ep at `TEMP_MAX=78`) not one `🌡`/`💤`/`cooldown` line
  fired. ▶ Do NOT budget that forward: R50's step is 376 ms against R34's 224, i.e. materially
  denser and hotter. Keep `TEMP_MAX` armed and let it govern rather than adding a fixed rest.
* ⚠ **The 59 ms allreduce is running UNREGISTERED.** Both probes logged
  `ncclCommRegister ... unhandled cuda error ... Cuda failure 500 'named symbol not found'`, so NCCL
  fell back off user-buffer registration. Non-fatal, and both numbers are honest for this box as
  configured — but the 59 ms is the one term that might come down without touching the math.
* ⚠ The 160 numbers stay a FLOOR until the 160 render is probed directly, which needs §2.1's spec
  (the synth path `LEAN_MLIR_BENCH_SYNTH` can probe it with **no data path at all** — that is the
  cheap way to convert this bound into a measurement, and `resnet50in_lamb64bce` /
  `resnet50in160_lamb64bce` are a matched pair differing only in resolution).

⚠ **Ask before starting any of it.** This box crashes on long runs and 9 h is not a probe either.

✅ **The 4-GPU setup itself smoke-tested clean** on this run: zero `BadTLP|Hardware Error|AER:` in
`journalctl -k` across both probes, all cards released to 2 MiB. The watchdog used is
`jax/scripts/aer_watchdog_4gpu.sh`'s loop retargeted at the Lean XLA driver, stopping on the
`PROBE:` line. ⚠ Devices **0,2,3,4** — idx1 (bus 02) and idx5 (bus 62) are the two cards that threw
BadTLP, confirmed still the right exclusion by `nvidia-smi --query-gpu=index,pci.bus_id`.

### 4.1 What the run would and would not license

⚠ Even a completed A3 run is **not** the reference's 78.1%: this is AdamW-family LAMB at
`wdExcludeNormBias = false` unless §1's third axis lands, at whatever `k` the box affords, with the
BN regime being Ghost-BN over micro-batches rather than a genuine bs2048 forward. ▶ Write the delta
list BEFORE the run and quote the number against it — `rsb_a2_resnet50.md` records LAMB at bs512
giving **40.8% against 78.1%**, which is what a recipe delta is worth at this batch.

---

## 5. THE ORDER I WOULD DO IT IN

1. ✅ **DONE 2026-08-05 — R50's ms/step measured: 376 (4×bs64) / 317 (1×bs64) / 367 (synth).** §4's
   affordability question is ANSWERED against the operator's real 40 h bar: **A3 at 100 epochs FITS,
   at ~31–37 h, but ONLY at 160** (224 is 52.3 h). ⚠ That promotes §2's data path from "probably
   another session" to **the critical path** — see §4.0.2.
2. **Fix `accOn`** to a substring test + pin the counterfactuals in `TestVariantPredicates`. *~20 min.*
3. **Refactor `R34Opt` into `{base, accumK, bce}`** and derive the variant string from it, with
   `#guard`s on the round trip. Re-render everything and require **byte-identity on all 13 existing
   artifacts** — that is the whole safety net for this refactor. *~2 h.*
4. **Render `lambacc4x64bce` (1 replica) and `lambaccdp8x64bce` (4 replicas)** at q=7 first.
   Compile-check both. *⚠ At q=7, not 160 — separate the composition from the resolution.*
5. ✅ **DONE 2026-08-06.** Both accumulation gates generalised (`R50_ACC_PEER` + `R50_ACC_RES`) and
   **run on the LAMB accumulator: both CERTIFY** (§1.4b) — the machinery at rel 0 with a 533×
   control, and the different-micro-batch identity at rel 0 with a 2137× control. Both 224
   regressions unchanged. ▶ **"lamb+bce+accum" is DONE and defensible.**
   ⚠ With ONE documented exception: `r50-gradcheck`'s **tier 1 B does not transfer to BCE** and the
   gate cannot discriminate there (§1.4c) — a PRE-EXISTING gap in the BCE renders, not the
   composition's, with tier 1 A and tier 2 passing on the composed artifact.
6. ✅ **DONE 2026-08-06 — the 160 data path, BOTH sides** (§2). Train @160 off the A3 shim and eval
   @224 off `@resnet50in160_fwd_eval`, verified end to end at 1 GPU; the 224 path re-verified
   unregressed and provably inert. ⚠ What is NOT done: a **4-replica** 160 render, so the
   producer-throughput question of §2.1 and the 59 ms allreduce term are both still 1-GPU
   extrapolations.
7. **A3 itself** — ✅ 1's answer is IN and **100 epochs at 160 fits the 40 h bar** (§4.0.2), so the
   epoch count is no longer a delta. ⚠ Still requires: step 6's data path (it is the gate), an
   `r50-*.conf` for `scripts/supervise.sh`, a 30-epoch shakeout first, and asking. Quote the result
   against §4.1's delta list, which is unchanged and still substantial.

⚠ Steps 1–5 are a session. Step 6 is probably another. **Do not let step 7 pull steps 3–5 into
being done quickly** — the composition is the deliverable that survives whether or not the run
happens, and it is the one with a byte-identity safety net.

---

## 6. THINGS THIS REPO LEARNED THAT APPLY DIRECTLY HERE

* ⭐ **A resolution/shape parameterisation is not done when it elaborates — it is done when it
  COMPILES.** `gapBackBatched` kept a baked `h := 7` through a clean Lean build and only XLA saw it
  (`tensor<64x51200>` used where `tensor<64x100352>` was defined), because the literal is a graph
  *argument*, not a type index. Compile every new `q`.
* ⭐ **A control that can catch the HARNESS is worth more than one that can only catch the render.**
  `r50-bce-tie`'s ⟂① requires the CE peer to match softmax-CE's own closed form, and it caught the
  test computing plain CE where the render emits smoothed CE. Build at least one of those per gate.
* ⭐ **A control that cannot fire is not a control, and finding that out is a result.** `r50-lamb-tie`
  runs two `v̂` regimes because at `v̂ ≈ 1` LAMB's `ε = 1e-6` makes the two ε placements agree to 5e-7
  — below the harness's floor. It asserts the control is DEAD in one regime and LIVE in the other.
* ⚠ **Marker collisions live in CONCATENATIONS, not in single names.** `lambacc…` breaking a
  `startsWith "acc"` test is the fourth instance; `TestVariantPredicates` exists for it.
* ⚠ **Estimates in this repo have come in wrong in both directions.** LAMB was estimated at "2–3
  ops" and measured at 2; the R50 render was estimated at 3–5 sessions and took 1. ▶ Cost the
  `R34Opt` refactor by counting match sites before starting it, not after.
