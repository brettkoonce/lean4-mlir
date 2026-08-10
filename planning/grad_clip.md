# grad_clip.md — global-norm gradient clipping, scoped before building

> **✅ BUILT AND GATED 2026-08-02, the same day.** Everything below §0 is the scoping as written
> *before* the code; **§11 at the bottom is what actually happened**, including the three things
> this document got wrong and the two defects the build turned up. Read §11 first if you want the
> state; read §0-§9 for how it was scoped.

**Written 2026-08-02, before any code.** The v1.4b item of `planning/recipe_gaps.md`, sketched in
`xla_pjrt_handoff.md` §0.2. Same discipline as `planning/ema.md` and `planning/stochastic_depth.md`:
read the reference, grep the kit, check every proposed op against an existing reading, and re-tier
the `recipe_gaps` entry from measurement rather than from the estimate that is already in it.

Everything below with a file:line is measured today, not remembered.

---

## 0. ▶ THE HEADLINE: `recipe_gaps` Tier E is WRONG BY A TIER, and for a checkable reason

Tier E files grad clip as *"the render, structurally … a cross-parameter reduction the render has
never had"*, and §0.2 names the real worry: **the global norm is one scalar consumed by all 200
sites, and `SHlo` is a single-output expression tree.** That framing is what makes it look
structural.

**It dissolves on inspection: `SHlo` is single-OUTPUT, not single-INPUT.** `sub`, `addV`, `matmulF`
and `batched2` are already binary (`StableHLO.lean:205, 287, 519, 3224`). So a 200-way fold of
200 sum-of-squares subtrees into one scalar **is an ordinary `SHlo` tree** — and it is a *small*
one, because every gradient the fold consumes is already an `.operand` leaf (`vitAdamOne` builds
`.operand gAvg z` at `ViTRender.lean:502`, and the un-fused gradient SSAs come out of `vitBackAll`
one per parameter). Nothing is recomputed; the fold has 200 leaves, not 200 backward passes.

So **the norm needs no carve-out at all**, which was the open question. The clip lands in the same
place RMSProp did: *"filed as Tier D, measured it is ONE op"* — here, filed as Tier E, measured it
is **four small ops in the parameter-shape family and no new proof machinery**.

⚠ What stays outside a theorem is much narrower than a carve-out: the renderer must pair the
**SSA name** `pretty` returned for the norm tree with the `.operand` leaf every `clipScaleF`
consumes. That is the same obligation class as every `xName`/`x` pairing in the kit
(`StableHLO.lean:846` says so in as many words), **not** the `emitGradAllReduce` / `%loss` class,
which is emitted text with no AST node behind it. Every line grad clip emits is `pretty` of a node.

---

## 1. The reference

`jax/Jax/Codegen.lean:2262-2265`, emitted verbatim into every trainer whose config sets it, placed
**after `value_and_grad` and before weight decay + the optimizer**:

```python
gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, CLIP / (gn + 1e-6)), grads)
```

**Who sets it** (grepped, not recalled): `MainVitImagenet.lean:45` and
`MainConvNeXtImagenet.lean:74`, both `1.0`; also the S/B tiers of each.
`MainEfficientNetImagenet.lean:63` sets **0.0 deliberately** — *"the TF-RMSProp fix (ε-inside-sqrt +
ms-init-1.0) removes the blow-up this was compensating for"*. **Do not add it there.**
R34 and mnv2 do not use it.

⚠ **The Imagenette configs set NOTHING** — `vitTinyConfig` / `convnextTinyConfig` have no
`gradClipNorm`, so it defaults to 0.0. An Imagenette clip render is therefore a **GATE VEHICLE, not
a matched pair**, exactly as `ema` is (§0.4). Carry the ImageNet value as the knob's default so the
`*in_*` renders need no edit, and never quote a 4-epoch Imagenette clip run as a reference
comparison.

### ⚠ A measurement the reference already made, and it conditions our gates for free

`MainVitImagenet.lean:89-93` — pre-clip **global** grad norm at init, ViT-B, batch 32:
**44.09 (Xavier init) / 14.28 (timm init)**, against a threshold of **1.0**. Its own conclusion:
*"both norms still exceed the threshold by 10x+."*

So **the clipping regime is the DEFAULT regime at init, not an edge case.** §0.2 worried that
`LEAN_MLIR_BASE_LR_U` cannot condition this gate because the norm is a property of the data — true,
and it turns out not to matter in the direction that was feared. The regime that needs conditioning
is the *un*clipped one, and §7 drives it with the threshold rather than with the data.

---

## 2. What the kit already has — measured 2026-08-02

| piece | evidence |
|---|---|
| **a rank-0 result from a reduce** | `lnBetaGrad : SHlo n → SHlo 1` emits a real `reduce … -> tensor<f32>` (`StableHLO.lean:5834`). `SHlo 1` denoting a rank-0 `tensor<f32>` is established, not new |
| **the parameter-shape emit world** | `adamMNextF`/`adamVNextF`/`adamWParamF`/`sgdParamF`/`rmsBufNextF` all carry `ds : List Nat` and emit at `ty ds` at arbitrary rank, ignoring the batch (`StableHLO.lean:4400-4455`). This is the sub-world grad clip lives in |
| **broadcast a RUNTIME rank-0 scalar to a param shape, then multiply** | `broadcast_in_dim {lrN}, dims = [] : (tensor<f32>) -> {T}` — emitted **200× per ViT render** inside `adamWParamF`. The exact shape `scaleByScalarF` needs, already in the file |
| `stablehlo.sqrt` | 200× in `vit_adam_train_step.mlir`, 158× in `mobilenetv2_adam_train_step.mlir` |
| `stablehlo.minimum` | **35×** in `mobilenetv2_adam_train_step.mlir` (relu6). New to ViT/ConvNeXt's renders, not new to the repo |
| **`%zero` / `%one` as rank-0 `tensor<f32>`** | already in both AdamW preambles (`ViTRender.lean:474-475`, `ConvNeXtRender.lean:110`) — the fold seed is free |
| **the insertion point** | `vitBackAll` returns `gradNames` (all 200) and `convNextBackAll` returns `gradMap`, both **before** the optimizer loop (`ViTRender.lean:627`) — every gradient SSA coexists exactly where the clip goes |
| **no rank-0 PARAMETERS** | `vitParamSig` (`ViTRender.lean:385`) and ConvNeXt's `allParams` (`ConvNeXtRender.lean:310`) are rank ≥ 1 throughout, so `dims = List.range ds.length` is never empty |

### ⚠ The one thing that looks usable and is not

**`scaleF`/`scaleB` do NOT take a runtime scalar.** They carry `sStr : String` beside the ℝ, which
reads like an operand name, and they emit `stablehlo.constant dense<{sStr}>` — a baked literal
(`StableHLO.lean:527, 727`). Checked in the emitter, because the constructor signature reads the
other way. Nothing else in the kit scales a tensor by a runtime scalar outside a fused optimizer op.

**And `addV` cannot fold the scalars.** It emits at `ty [B,n]` (`StableHLO.lean:4122`) — the batched
*activation* world — so `addV` at `n = 1` is `tensor<Bx1xf32>`, not `tensor<f32>`. The parameter
world and the activation world are separate emit families keyed on `ds : List Nat` vs `n : Nat`, and
grad clip is entirely in the first.

---

## 3. The op inventory — four constructors, all in the `ds : List Nat` family

Each is `StableHLO.lean`'s ten sites (constructor, `den`, the `rfl` faithfulness theorem, `Raw`,
`skel`, `Tok`, `toToks`, `emitTok`, plus `parseStack` and the `parse_toToks` case in
`StableHLOParse.lean` — §4 of the handoff). None is a `BatchableOp` descriptor: these are
parameter-space, not per-example.

**1. `gradSumSqF {n} (ds : List Nat) : SHlo n → SHlo 1`** — `den = fun _ => ∑ i, (den e i)^2`

```mlir
%z  = stablehlo.constant dense<0.0> : tensor<f32>
%sq = stablehlo.multiply {r}, {r} : {ty ds}
%o  = stablehlo.reduce(%sq init: %z) applies stablehlo.add across dimensions = [0,…,rank−1]
        : ({ty ds}, tensor<f32>) -> tensor<f32>
```

**2. `addScalarF : SHlo 1 → SHlo 1 → SHlo 1`** — `den = fun _ => den a 0 + den b 0`

```mlir
%o = stablehlo.add {a}, {b} : tensor<f32>
```

**3. `gradClipFacF (clipStr epsStr : String) (c ε : ℝ) : SHlo 1 → SHlo 1`** —
`den = fun _ => min 1 (c / (Real.sqrt (den e 0) + ε))`

```mlir
%gn  = stablehlo.sqrt {r} : tensor<f32>
%eps = stablehlo.constant dense<{epsStr}> : tensor<f32>
%d   = stablehlo.add %gn, %eps : tensor<f32>
%c   = stablehlo.constant dense<{clipStr}> : tensor<f32>
%rat = stablehlo.divide %c, %d : tensor<f32>
%one = stablehlo.constant dense<1.0> : tensor<f32>
%o   = stablehlo.minimum %one, %rat : tensor<f32>
```

**4. `clipScaleF {n} (ds : List Nat) : SHlo 1 → SHlo n → SHlo n`** —
`den = fun i => den f 0 * den e i`

```mlir
%b = stablehlo.broadcast_in_dim {f}, dims = [] : (tensor<f32>) -> {ty ds}
%o = stablehlo.multiply {b}, {r} : {ty ds}
```

⚠⚠ **`clipScaleF` takes the factor as a CHILD, not as a `facName : String` + `fac : ℝ` field
pair, and that is the whole design.** The name+value form is what `%lr`/`%wd` use, and it would
work — but it puts an ℝ inside the constructor whose agreement with the norm is *assumed*. As a
child, `den` is exactly `factor · g` with no ℝ of its own, and the pairing obligation collapses onto
an ordinary `.operand` leaf. **Cost of the child form is zero in the emit**, because the leaf is
`.operand normSSA` — `pretty` prints nothing for a leaf, it just returns the name.

⚠ **Do NOT give `clipScaleF` the norm SUBTREE as its child.** `pretty` has no CSE (§4), so 200
copies of a 400-line tree is ~80,000 lines. Emit the tree once, take its SSA name, and hand every
site an `.operand` leaf at that name. This is `resnetFwdGraph`'s *"tree-safe via operand leaves"*
trick (`StableHLO.lean:2682`) used for its other purpose.

**Alternative worth 10 sites**: fuse 1 and 2 into
`gradSumSqAccF {n} (ds) : SHlo 1 → SHlo n → SHlo 1` (`den = acc + ∑ g²`), making the fold a
linear chain by construction — 3 constructors instead of 4. Rejected as the default because
`gradSumSqF`'s denotation is then exactly `‖g‖²`, which is the thing the faithfulness theorem wants
to say, and `addScalarF` is trivially reusable by any future scalar fold. Cheap to reverse either
way.

### The `Proofs` side

One small module, `Proofs/Codegen/GradClip.lean`, in the shape of `RmsPropStep.lean` (179 lines,
one op, 13 theorems) or `DropPath.lean` (196):

* `gradSumSq : Vec n → ℝ := ∑ i, (g i)^2` and `clipFactor c ε (s : ℝ) := min 1 (c / (√s + ε))`.
  ⚠ Grep `Proofs/` for an existing squared-norm first — `LipschitzCert.lean:75` has `‖v‖² = Σ (vᵢ)²`
  on `EuclideanSpace`, which is a *lemma over a different carrier*, not a reusable `Vec n` def. That
  check is exactly the one that has collapsed a scoped family to zero four times; it does not
  collapse here, but it is two minutes.
* four `@[simp] … _faithful : den (.X …) = <spec> := rfl` theorems.
* the useful **known-answer lemma**: `clipFactor c ε s = 1` when `√s + ε ≤ c` — this is what makes
  §7's below-threshold control a *bit-exactness* claim rather than a tolerance, since `g * 1.0` is
  exact in IEEE-754. Same shape as `dropPath_ones_id`, and `dropPathB`'s gate leaned on it the same
  way.
* the whole-net statement instantiates the `.operand` leaf **at the norm tree's `den`**, the way
  `dropPathB_back_faithful` instantiates its cotangent — so the pairing is exact *in the theorem*
  and the renderer keeps using placeholder-zero leaves like every other render. ⚠ Do not try to make
  the renderer store `den normTree` in the leaf: `ℝ` is noncomputable and the renderer is a `#eval`.
  The render is value-independent by design (§5 of the handoff) and this does not change that.

---

## 4. ⚠ The DP restructure — the collective must move, and only when clip is on

`emitGradAllReduce` is called **inside** `vitAdamOne` (`ViTRender.lean:501`), i.e. **per parameter,
after the point where the clip factor would have to exist.** Same in `ConvNeXtRender.lean:601`,
`EfficientNetRender.lean:968/1000`, `MobileNetV2RenderB.lean:497/528`, `ResNet34RenderB.lean:247`.

Clipping before the `all_reduce` makes every replica clip its own **partial** gradient — a different
function that compiles, trains, descends, and that no structural check sees. So at `replicas > 1`
the loop must split:

1. per parameter: `emitGradAllReduce` → collect `gAvg` names (emits **nothing** at `replicas ≤ 1`)
2. once: the norm tree over those 200 leaves → `normSSA`
3. per parameter: `clipScaleF` on `.operand gAvg` with `.operand normSSA` → the AdamW triple

⚠⚠ **Gate the restructure on `clip`, not on `replicas`.** Splitting the loop unconditionally moves
the all_reduce text relative to the adam blocks and shifts the fresh-name counter, so every
committed `*dp*` artifact would re-render non-byte-identically **with the feature off** — gate 1's
strong form lost for nothing. Take the `ema := false` route instead (`ViTRender.lean:524`): at
`clip = false` **no `pretty` call happens**, the counter does not move, and all committed artifacts
re-render byte-identically. That is gate 1, free.

---

## 5. ⚠ What it breaks — §0.2's list is too long by three, and short by one

§0.2 says clipping *"breaks every gate that recovers `g` from `m'`"* and names `r34-mom-tie`,
`rms-tie`, `wdx-tie`, `shard-check`. Audited against what those harnesses actually run:

| gate | net / variant | verdict |
|---|---|---|
| `r34-mom-tie` | R34, `mom` | **unaffected** — R34 sets no clip and gets none |
| `rms-tie` | mnv2 / enet, `rms` | **unaffected** — neither net gets a clip; enet's 0.0 is deliberate |
| `wdx-tie` | ViT / ConvNeXt, `adam` vs `adamwx` | **argued unaffected, and worth measuring**: ① compares decayed θ′ between two renders that would carry the *same* clip, so the common factor cancels; ② is `θ'_wx − θ'_adam = lr·wd·θ`, which is independent of the gradient path entirely. ⚠ It only holds if **both** sides clip identically — running `wx`+`clip` against plain `adam` would break it |
| `shard-check` | ViT / ConvNeXt | ⛔ **genuinely breaks.** Its known answer needs the gated slot **linear in the gradient**; with clipping `m' = (1−β₁)·min(1, c/‖g‖)·g`, and ‖g‖ differs between the DP run and each single-device run. This is the RMSProp-buffer finding one optimizer over. **Run it on the clip-off variant**, or use the duplicated-batch `*-dp-check` identity, which is clip-agnostic (both sides get the identical gradient, so any tail must agree) |

**And one §0.2 missed**: the ViT/ConvNeXt ImageNet reference sets `gradClipNorm` **and**
`wdExcludeNormBias`, so the shipping variant is `wx` + `clip` composed, not clip alone. Whatever
harness gates `wx` must be run on a variant that carries both, or the composition is untested — the
`emarms` lesson, where a feature was fine alone and wrong composed.

---

## 6. Variant naming — and the concatenation check, not the marker check

Markers in play today: `adam`, `ema`, `dp`, `rms`, `drop`, `wx`, a batch number, `x{replicas}`.
Adding `clip`:

* **check every CONCATENATION, not every marker.** That is the `sd` ⊂ `rmsdp` finding — a
  stochastic-depth predicate fired on every RMSProp DP variant because two *other* markers, meeting,
  spelled it. `tests/TestVariantPredicates.lean` runs all spellings × axes; the clip marker goes in
  there before the render, not after.
* **trailing, like `wx`** — `ema` must keep leading, because `trainAdamSched` keys the 4-region
  `[θ|m|v|ema]` blob off `variant.startsWith "ema"`.
* ⚠ **ConvNeXt DERIVES its entry name from the variant** (`cnxAdamVariant`, `ConvNeXtRender.lean:639`)
  where ViT takes `funcName` explicitly. The `wx` thread shipped a ConvNeXt artifact whose declared
  entry disagreed with its own path because the flag reached the renderer and not the variant. The
  shim refused the call rather than running the wrong graph — but **only running both nets found
  it**, so run both, and add the `#guard`s pinning the new spellings.

**No driver change and no driver predicate**, for `wx`'s reason: with `clipStr` baked as a literal,
arity, types and regions are all unchanged. ⚠ Bake it as a **renderer parameter** with `wdStr`'s
treatment (default = the ImageNet value, `#guard`s on the spellings, and a byte-identical re-render
at the old value as the gate) — a hyperparameter that becomes a graph constant with no parameter
behind it is the `RenderCifar8Sgd02` / EfficientNet-16× / ViT-500× failure this repo has now paid
for three times.

---

## 7. The gates

**Gate 1 — inertness.** Every committed `vit*`/`vitin*`/`convnext*`/`convnextin*` artifact re-renders
**byte-identical**; only the new `clip` paths appear. Free if §4's `clip = false` short-circuit
holds. ⚠ FORCE the writers with `lake env lean` — a plain `lake build` can leave the `#eval`s unrun
and the empty diff is then vacuous (§2n).

**Gate 2 — the known answer, `r34-mom-tie`/`wdx-tie` construction.** At `m = v = 0`,
`m' = (1−β₁)·g_clipped`. Recover `g_i` per parameter from the **unclipped** render's `m'`, and
require the clipped render's `m'` to be `(1−β₁)·fac·g_i`.

⚠ **Read the factor, do not fit it — and do not recompute it on the host.** `gn` is an f32
tree-reduction of ~5.7M squares on device; a host f64 sum of the *recovered* f32 gradients will not
reproduce it, and the first version of this gate will look like a defect in a correct render. That
is the mixup λ finding (*recover a constant by READING it*) meeting the `wdx-tie` tolerance finding
(*a tolerance must be in the unit the instrument has*). Two ways to read it, in order of preference:

* ⭐ **the ratio is the gate.** `fac` is ONE scalar shared by all 200 parameters, so
  `m'_clip[j] / m'_unclip[j]` must be the **same constant at every one of the 5.7M coordinates**.
  That needs no host norm at all, and it is the check that distinguishes a *global* norm from a
  *per-parameter* one — a per-param clip gives 200 different ratios and passes any single-parameter
  comparison. **Gate the constancy across parameters, not the presence of scaling**: `wdx-tie`'s
  *gate the partition, not the count*, one feature over.
* then compare that measured constant against the host's f64 prediction, and **state the residual in
  ULPs**, not as an absolute bound.

**Gate 3 — the below-threshold control, and it is BIT-EXACT by argument.** Render at a `c` large
enough that `min(1, c/(gn+ε))` is exactly `1.0`; `g * 1.0` is exact in IEEE-754, so the clipped
render's outputs must be **bit-identical** to the unclipped render's — a bit-exactness claim, not a
tolerance. This is `dropPath`'s *"bit-exact is the bar by argument, not by luck"* in a second place.

**Gate 4 — the above-threshold control.** Render at a small `c` (or just use `c = 1.0`: the
reference measured the init norm at 14.28–44.09, so **the real threshold already clips by 10×+** and
no instrument trick is needed). Require the ratio to match the prediction and to be **< 1**.

⚠⚠ **Gates 3 and 4 together are what make gate 2 non-vacuous.** A harness that only ever runs in one
regime cannot tell a global clip from no clip at all — *a gate whose input makes the intervention
inert cannot test the intervention*, which is the misplaced-drop-site finding, and here it is
reachable by accident because `min(1, ·)` is the identity on one whole side of its domain.

**Controls that must be verified to FIRE**, in the `perturb_wd_mask.py` / `misplace_drop_sites.py`
style — a script that edits the committed artifact, so SSA names, order, types and line count are
untouched and no structural check moves:

| control | what it must catch |
|---|---|
| **per-parameter norm** — replace the shared `normSSA` at each site with that parameter's own `gradSumSqF` | the global-vs-local distinction, i.e. the actual semantics of the feature. Gate 2's ratio-constancy is the only check that sees it |
| **clip before the collective** (DP only) | §4's ordering. ⚠ Needs 2 GPUs and asymmetric data — with the duplicated batch both orderings agree exactly, the `*-dp-check` blindness a fourth time |
| **`sqrt` dropped** (clip on ‖g‖² instead of ‖g‖) | an off-by-a-square that trains and descends |
| **ε outside** — `c/gn + 1e-6` instead of `c/(gn+1e-6)` | the `rms-tie` ε-placement lesson; expect it to be small at these norms and say so |

⚠ **Run everything under `scripts/det_shim.sh`** and measure the A-vs-A floor first. The det shim is
mandatory for any cross-graph numeric comparison on CUDA; §2d.3's *"the floor IS bit-exact across
processes"* is ROCm-specific and has twice been the difference between a green feature and a phantom
defect.

---

## 8. Cost, in this repo's units

| piece | size | anchor |
|---|---|---|
| 4 `SHlo` ops × 10 sites | the mechanical half | `bnPerChannelF` is the template to grep |
| `Proofs/Codegen/GradClip.lean` | ~150-200 lines | `RmsPropStep.lean` 179, `DropPath.lean` 196 |
| ViT renderer: the loop split + the clip knob | ~70 functional lines | ViT's EMA peer was 69 |
| ConvNeXt renderer: same, plus the variant/entry-name fix | ~70 | and it is where the `wx` thread's defect lived |
| `clip-tie` harness + 4 perturbation controls | the real work | `wdx-tie` is the shape: ONE harness, `clip-tie <net>` |
| driver | **nothing** | arity, types and regions unchanged |

**Re-tier `recipe_gaps.md`: Tier E → Tier D**, with the note that the "structural" framing was about
a DAG problem that `SHlo`'s binary constructors already solve.

---

## 9. Before writing any code — the 20-minute checks

1. `grep` `Proofs/` for a `Vec n` squared-norm. (Expected: none reusable; `LipschitzCert.lean:75` is
   over `EuclideanSpace`.)
2. Confirm `stablehlo.minimum` and a rank-0 `stablehlo.divide` compile in a throwaway module — the
   spellings are in the committed artifacts (35× / everywhere) but a rank-0 `minimum` is new here.
3. Add the clip marker to `tests/TestVariantPredicates.lean` and check **every concatenation**
   against every existing predicate *before* the render exists.
4. `cat .lake/build/<slug>_<variant>_ckpt_xla.bin.epoch` before any smoke, and clear it between
   configs sharing slug+variant — a second run silently resumes the first (§4).
5. ⚠ Ask before anything long. This box cannot sustain multi-GPU load; the deliverable here is a
   certified render plus numeric gates, all single-GPU and all minutes.


---

## 11. ✅ WHAT ACTUALLY HAPPENED — built, gated and measured 2026-08-02

`lake build Proofs Certs Codegen` **3,913** green (3,912 + `GradClip.lean`) · **115** artifacts, one
writer each · `verified_mlir/` **0 lines of diff** · drift-guard coverage **68/99 → 72/103** with
`render_guard_baseline.txt` **unchanged** · `TestVariantPredicates` **39 spellings** · all 17 new
declarations **3-axiom clean**.

### The result

**Two `SHlo` ops, not four; no new proof machinery; no driver change.** `Proofs/Codegen/GradClip.lean`
(3 defs, 9 theorems) + `gradSumSqAccF` / `clipScaleF` + 5 faithfulness theorems in `StableHLO.lean`.
Four committed artifacts: `vit_adamclip`, `vitin_adam128wxclip`, `convnext_adamclip`,
`convnextin_adamwxclip`. The Tier E → Tier D re-tiering held.

### The gates — `lake build clip-tie`, both nets, ⚠ **under `scripts/det_shim.sh`**

| gate | ViT (200 params) | ConvNeXt (180) |
|---|---|---|
| ⓪ clip ACTIVE | factor **0.038283**, ‖g‖ = 26.12 | **0.022005**, ‖g‖ = 45.44 |
| ① the factor is ONE SHARED SCALAR | **1.120 ULPs** of spread | **1.149 ULPs** |
| ② = `min(1, c/(‖g‖+ε))` | **0.105 ppm** | **0.0070 ppm** |
| ③ `%loss` bit-exact | ✅ | ✅ |
| ④ inert above the threshold | **16,579,039/16,579,039** bit-identical | **83,478,847/83,478,847** |

**All six controls fire, rc=1:**

| control | ViT | ConvNeXt | fires |
|---|---|---|---|
| `perparam` — per-parameter norm | **7,601,258 ULPs** vs an 8-ULP bar | **30,372,642** | ① |
| `nosqrt` — clip on ‖g‖² | 961,717 ppm vs a 5 ppm bar | 977,995 ppm | ② |
| `epsout` — `c/gn + ε` | **26.28 ppm** (predicted `ε/fac` = 26.1) | **45.46** (45.5) | ② |

⚠ `perparam` passes ⓪, ③ **and** ④ — it is a working clip, just not a global one. `nosqrt` passes
①, because ‖g‖² is still one shared scalar. The gates are independent by construction.

⚠ The ⓪ reading confirms the reference's own measurement: at init the global norm is **26-45× the
threshold**, so clipping is the regime a real run starts in, not an edge case.

### ⚠ THE DP ORDERING, CLOSED STRUCTURALLY (and its numeric gate is OWED)

Rendered at `replicas := 2`: the last of **200** `all_reduce`s is at line 8435, the first
sum-of-squares reduce at 8444, and all 200 folds consume `%armean*` — the AVERAGED gradients. **200
collectives, not 400**, so `preAvg` correctly suppressed the per-parameter duplicate. ⛔ **Owed: a
committed DP clip artifact and its numeric gate** (2 GPUs, the `*-dp-check` construction —
`shard-check` cannot do it, the clip makes `m'` nonlinear in `g`). It is deliberately not committed
because the DP ImageNet renders already carry §0.3's 500× `wd` gap, and stacking a second variant on
a known-wrong one is not a matched pair either.

### ⚠ THREE THINGS THIS DOCUMENT GOT WRONG

1. ⚠⚠ **THE FOUR-OP SPLIT WAS UNBUILDABLE, AND FOR A REASON NOTHING HERE ANTICIPATED.** §3 proposed
   `addScalarF : SHlo 1 → SHlo 1 → SHlo 1` and `gradClipFacF : SHlo 1 → SHlo 1`. **A constructor
   with NO `{n : Nat}` binder is a shape this AST does not otherwise have**, and adding two of them
   made **nine unrelated `simp only [… den …]` proofs elsewhere in `StableHLO.lean` die with a
   `whnf` timeout** — `den` is a ~200-case dependent match and fully-index-fixed arms make
   unfolding it markedly more expensive. **4× the heartbeat budget did not fix it.** Fusing to two
   parametric ops (`gradSumSqAccF` carries the fold; `clipScaleF` computes the factor inline)
   cleared it on the first try. ▶ **Rule: a new `SHlo` constructor must be parametric in `n`.** The
   fused form is also cheaper — 20 sites instead of 40 — so the constraint pushed toward the better
   design.
   ⚠ A smaller cousin of the same rule: **`den` must never APPLY a recursive `den` call to an
   index.** `den a 0` was doing that; `Proofs.scalarOf` exists to keep the new arms the same shape
   as the other 200.
2. **§5 said `wdx-tie` was "argued unaffected, worth measuring".** Not measured — the shipping
   variants are `wx` ++ `clip` and `wdx-tie` runs on `adam`/`adamwx`, so it is untouched today. The
   argument still stands and is still unmeasured; say so rather than quoting it as checked.
3. **§7's ② was scoped as "state the residual in ULPs".** It is a **relative** quantity with no
   natural ULP unit, and printing it at 6 decimals made a correct render and an ε-placement error
   both read `0.000026` vs `0.000000` — two numbers whose RATIO is the whole question, rendered as
   one digit. It reports in **ppm** now, which is what made the 5 ppm bar calibratable from
   measurement on both nets instead of picked round.

### ⚠ TWO DEFECTS THE BUILD TURNED UP, both of which generalise

1. ⚠⚠ **A BOOL-DERIVED VARIANT NAME CANNOT DISTINGUISH TWO RENDERS THAT DIFFER ONLY IN A BAKED
   CONSTANT.** The below-threshold render was first *committed* at `clipStr := "1e9"`. On ConvNeXt,
   whose entry name is DERIVED (`cnxAdamVariant`'s `clip : Bool`), both thresholds spelled
   `adamclip` — so `convnext_adamcliphi_train_step.mlir` came out declaring
   `@convnext_adamclip_train_step`, an entry disagreeing with its own path. ViT takes `funcName`
   explicitly and hid it. **That is §0.4's derived-vs-explicit finding meeting §2a-quater's
   silent-hyperparameter one**, and the fix is better than the bug: the below-threshold render is
   **generated** by `scripts/perturb_clip.py hi`, not committed, because an artifact baking a
   threshold no config sets is exactly a silent-hyperparameter artifact.
2. ⚠⚠ **THE DET SHIM IS MANDATORY HERE AND I DID NOT USE IT AT FIRST — the third recorded instance.**
   Without `scripts/det_shim.sh`, ConvNeXt's gate ④ read **137,229 of 83,478,847 outputs differing
   at a 24,149-ULP floor**, and ①'s spread (12,696 ULPs) sat *below* that floor, i.e. the gate was
   reading noise. The floor was carried by **exactly ONE parameter of 180** — `d0W`, the even-kernel
   2×2/s2 downsample weight gradient — and it **moved between runs** (24,169 → 24,149). Under the
   det shim: **83,478,847/83,478,847 bit-identical, ① at 1.15 ULPs.** *Nothing about the render
   changed.* ⚠ **ViT is clean either way**, which is precisely how the trap stays hidden — a gate
   developed on ViT and ported to ConvNeXt inherits ViT's conditioning (handoff §0.4 finding 1, now
   in a **fourth** place). Gate ④ now refuses with the det-shim recipe rather than reporting a
   phantom defect, and it is deliberately run FIRST because it is also ①'s floor.

### The transferable one-liner

**Measure the floor with an instrument that has the intervention's STRUCTURE but not its EFFECT.**
The `hi` render is the clip block with the factor pinned at exactly 1.0 — same ops, same schedule
pressure, mathematically the identity — so it answers a question the gate cannot ask of itself:
*does the feature's mere presence move the number?* On ConvNeXt it did, and that is the only reason
①'s reading was interpretable at all.
