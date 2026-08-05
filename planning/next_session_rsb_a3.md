# next_session_rsb_a3.md — LAMB + BCE + accumulation, and then: can RSB-A3 actually be run?

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
| **train @160 / eval @224** | ⚠ renders, no data path | q=5 compiles; `resnet50in160_*` exist |
| **mixup / cutmix** | ✅ shipping | `soft-target-tie`, wire v2, `SHIM_MIX=both` |
| **`wdExcludeNormBias`** | ⛔ absent | — |
| **the composition** | ⛔⛔ **does not exist** | — |

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

### 1.3 The target artifacts

| artifact | what it is |
|---|---|
| `resnet50in160_lambacc8x64bce_train_step` | ⭐ **the recipe**: LAMB + BCE + k=8, 4 replicas × bs64 ⇒ effective **2048**, at 160 |
| `resnet50in160_lambacc4x64bce_train_step` | single-device peer at k=4, so the accum gates can run |
| `resnet50in160_fwd`, `resnet50in160_fwd_eval` | ✅ already rendered (train @160, eval @224) |

---

## 2. ⛔ THE DATA PATH AT 160 — the piece with no render-side answer

The 160 artifacts have **no way to be fed**. `resnet50ImagenetVerified` is a 224 net
(`d0 = 3·224·224 = 150528`) and the tfds shim produces 224 crops.

1. **A 160 `VerifiedNetSpec`** — `resnet50Imagenet160Verified`, `imageH := 160`, `imageW := 160`,
   `slug := "resnet50in160"`, its own `shimScript`. Everything else (161 tensors, `bnChannels`) is
   resolution-independent, so `TestR50Contract` should pass unchanged — ⚠ **run it and confirm**,
   because that is the cheapest possible check that the 160 spec is the same net.
2. **A 160 shim.** `jax/MainResnet50Imagenet.lean`'s config already carries `trainRes := 160` and
   `testCropRatio := 0.95`; `scripts/gen_shims.sh` is where it becomes a `.py`. ⚠ Check whether the
   generator honours `trainRes` for the TRAIN split only — A3 is 160 train / 224 eval, so a shim
   that resizes both is the wrong shim and would read as a working run.
3. ⚠⚠ **THE DRIVER FEEDS ONE IMAGE SIZE.** `net.d0` is used for the train invoke *and* the eval
   invoke. Train @160 / eval @224 needs the eval path to use its own `d0`. That is a real
   `VerifiedTrain.lean` change and it is the piece most likely to be underestimated — look at
   `evalBs` (already read off the artifact) for the pattern, and add `evalD0` beside it.

▶ **Do (1)+(2) and check the smoke at 160-eval-160 FIRST** (i.e. eval also at 160, which needs no
driver change), then add the 224 eval. A 160/160 run is not A3, but it separates "the 160 render
trains" from "the split works", and one of those is much likelier to break.

---

## 3. THE GATES — what must be re-run, and what has to be GENERALISED

Every existing gate hardcodes a peer artifact. Composition moves the peers.

| gate | change needed |
|---|---|
| `r50-gradcheck` | ⭐ **runs as-is** via `R50_GC_VARIANT` — point it at the composed artifact. ⚠ Tier 1's identities are loss-agnostic and hold for BCE; tier 2 pins the scale. **Run it on the 160 render too** — the resolution changes the numbers, not the argument |
| `r50-accum-tie` | ⛔ its peer is hardcoded `adam64`. For a LAMB accumulator the peer must be `lamb64` (or `lamb64bce`). Parameterise the peer |
| `r50-accum-shard-tie` | ⛔ same: its peer is `adamdp64`. Needs a `lambdp64` peer rendered |
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

⛔⛔ **R50's ms/step is STILL UNMEASURED.** It has been the standing open item since §3.3 and every
wall-clock number for A3 in this repo is an extrapolation off R34's 224 ms.

⭐ **Measure it FIRST, before any composition work.** It is one probe:

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_VARIANT=adamdp64 \
  LEAN_MLIR_REPLICAS=4 LEAN_MLIR_MAX_STEPS=120 SHIM_WORKERS=8 \
  .lake/build/bin/resnet50-imagenet-verified-xla data
```

and again with the 160 render once it has a data path. Then the arithmetic is:

* 1,281,167 images/epoch ÷ 256 per invoke = **5,005 invokes/epoch**
* × 100 epochs = **500,500 invokes**, independent of `k` (accumulation changes *when* the optimizer
  fires, not how many forward/backwards run)
* at 160px the FLOPs are `(160/224)² = 0.51×` the 224 cost

▶ **A3 is affordable iff the 160 step is under ~200 ms**: 500,500 × 0.2 s ≈ **28 hours**. At 400 ms
it is 56 h and the answer is "not on this box without bf16".

⚠ **Ask before starting it.** This box crashes on long runs and 28 h is not a probe.

### 4.1 What the run would and would not license

⚠ Even a completed A3 run is **not** the reference's 78.1%: this is AdamW-family LAMB at
`wdExcludeNormBias = false` unless §1's third axis lands, at whatever `k` the box affords, with the
BN regime being Ghost-BN over micro-batches rather than a genuine bs2048 forward. ▶ Write the delta
list BEFORE the run and quote the number against it — `rsb_a2_resnet50.md` records LAMB at bs512
giving **40.8% against 78.1%**, which is what a recipe delta is worth at this batch.

---

## 5. THE ORDER I WOULD DO IT IN

1. ⭐ **Measure R50's ms/step at 224** (one 120-step probe, 4 GPUs). Decide affordability before
   building anything. *~20 min.*
2. **Fix `accOn`** to a substring test + pin the counterfactuals in `TestVariantPredicates`. *~20 min.*
3. **Refactor `R34Opt` into `{base, accumK, bce}`** and derive the variant string from it, with
   `#guard`s on the round trip. Re-render everything and require **byte-identity on all 13 existing
   artifacts** — that is the whole safety net for this refactor. *~2 h.*
4. **Render `lambacc4x64bce` (1 replica) and `lambaccdp8x64bce` (4 replicas)** at q=7 first.
   Compile-check both. *⚠ At q=7, not 160 — separate the composition from the resolution.*
5. **Generalise `r50-accum-tie` / `r50-accum-shard-tie`'s peer** and run them on the LAMB
   accumulator. Run `r50-gradcheck` on the composed artifact. *~1.5 h.* ▶ **This is the point at
   which "lamb+bce+accum" is DONE and defensible, independent of whether A3 ever runs.**
6. **The 160 data path** (§2), 160/160 smoke first, then the 224 eval split.
7. **A3 itself** — only after 1's answer says it fits, and only after asking.

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
