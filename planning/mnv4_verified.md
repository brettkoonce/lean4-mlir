# mnv4_verified.md — MobileNetV4 as a verified net

**Scoped 2026-08-07 by reading the source, not by estimate.** Every file/line reference below was
checked. Companions: `planning/mnv4_imagenet.md` (the **JAX reference** side — recipe, timm decode,
the Conv-S/Conv-M distinction), `planning/rsb_a3_r50_verified.md` (the closest precedent: a new
block form brought onto the verified path).

Status: **PHASE 2 (backward) DONE 2026-08-09 — see §3i.** The train step renders, compiles, and its
gradient ties `jax.grad` of the reference; zero new SHlo ops were required. What remains is the
driver/spec plumbing and the 80-epoch run.

Status: **PHASES 0–3 (forward) DONE 2026-08-08.** `.uib` and `.fusedMbConvNB` are in `VLayer`, the
whole Conv-S forward chain renders, `@mnv4_fwd` compiles, and the forward tie against the JAX
reference RAN. Block order is **pinned at 1.8e-6**. One blocker left, decided below.
**The MNv2 tie also ran (2026-08-08): 6.08e-06 with both conventions matched — see §3d, which
found a second, larger split (the BN world) and a hole in the artifact guard.**

---

## ✅ THE DESCRIPTOR IS BUILT AND THE MNv4 FORWARD TIE PASSES CLEAN (2026-08-08)

**`@mnv4_fwd` now ties the UNPATCHED reference at `1.423e-06`.** The diagnostic columns swapped,
which is the confirmation that matters:

| reference variant | before | after |
|---|---|---|
| **as-is (`padding='SAME'` stem)** | 6.164e-02 | **1.423e-06** ✅ |
| stem patched to symmetric (1,1) | 1.788e-06 | 6.164e-02 |

Nothing about the reference moved; 84.58% stands. **Phase 3-forward is now closed with no known
deviation.** What was built, and why it cost far less than §3b estimated, is §3e. What remains is
§4's Phase 2 + Phase 3-rest, unblocked and unstarted.

---

## ▶ THE ORIGINAL PLAN (kept — §3e records what actually happened)

**DECISION (Brett, 2026-08-08): build the asymmetric-pad conv + its VJP.** §3b's option 1. The
render is the side that changes; the reference and its 84.58% stay put.

**Why this and not the other two:** MobileNetV4 is a TF-origin port, and this repo's stated rule for
that family is that asymmetric `'SAME'` *is* the reference (`generated_mobilenet_v4.py`'s `conv2d`:
*"MobileNetV2/EfficientNet … are TF-origin ports, where asymmetric 'SAME' IS the reference. Do not
'fix' those"*). Changing the reference would move the net and void the number; documenting the
deviation would leave the verified render as a net nobody has a result for.

**What to build, in order:**

1. **Padding in `VLayer`.** It currently has **zero** occurrences of `Padding` — the verified layer
   language cannot express the distinction at all, which is why this was invisible per-net rather
   than a transcription slip. The baseline `Layer` already carries `pad : Padding` (`.same`/`.valid`,
   `Types.lean:20`), and both MNv2 and MNv4 specs declare `.convBn 3 32 3 2 .same`.
2. **An asymmetric-pad strided conv descriptor** in `Proofs/Codegen/StableHLO.lean`. `convStrided`
   emits `[[1,1],[1,1]]`; XLA `'SAME'` on 3×3/s2 at an even input is `[[0,1],[0,1]]`. ⚠ The
   "stride-2 SAME conv" comment at `:400` means *same output size*, NOT XLA SAME — do not read it
   as prior art for this.
3. **Its VJP.** The backward has to place the same asymmetry; a symmetric backward against an
   asymmetric forward is a silent wrong-gradient, and the existing gradcheck harnesses would be the
   place to catch it.
4. **Re-run `scripts/mnv4_forward_tie.py`.** Success is the as-is column reaching ~1.8e-6 — the
   number the symmetric-patched column already achieves, which is what the rest of the net is
   worth once the stem stops disagreeing.

⚠ **Adding an SHlo op is 10 sites across 2 files**, and a bare `lake build` misses the parser
roundtrip because it lives under `Certs` (memory: `adding-an-shlo-op`). Budget for that.

⚠⚠ **THIS IS NOT AN MNv4 BUG — IT IS REPO-WIDE, AND MNv2 IS WORSE.** See §3c. Fixing the descriptor
without then re-checking MobileNetV2 and EfficientNet leaves the same defect in two *shipping*
nets, one of which carries a quoted Imagenette result.

### ⭐ UPDATE 2026-08-08 — the MNv2 tie ran. Read §3d. Two things changed:

1. **The descriptor has ≥2 measured consumers, not 1.** MNv2 ties at **6.08e-06** once padding AND
   BN are matched, so its structure is sound; padding is a real 5-site deviation worth **27.7% of
   logit range** in the batch-BN world its trainer uses. Build the descriptor once, re-run both ties.
2. ⛔ **A HOLE IN THE ARTIFACT GUARD.** `mobilenetv2_fwd.mlir` is *per-example* BN while
   `mobilenetv2_adam_train_step.mlir` — what trained the 89.35% — is *batch* BN. They are not
   byte-prefixes, and `scripts/regen_verified_mlir.sh check` goes **green** because it only ever
   pairs a forward with the **SGD** train step, never the Adam one. **That applies to all five
   nets**, and the Adam artifacts are the ones every quoted verified number trains on.
   ✅ The 89.35% itself is CLEAN — traced, §3d(c): the per-example forward is compiled but never
   invoked, because eval goes through `_fwd_eval`. But that is a runtime `if`, not an invariant
   anything checks. **Close the guard hole; the number is fine.**

---

---

## 0. TL;DR

**The block is nearly free; the pre-DW position is the one genuinely new thing.**

MNv4's Universal Inverted Bottleneck is `optional pre-DW → expand 1×1 → optional post-DW →
project 1×1`, every conv BN-followed, with `k = 0` meaning "no depthwise". `VLayer` already renders
and certifies **every one of those pieces** — `invertedResidual` is expand → DW3×3 → project, and
`mbConvSE` is expand → DW k×k → SE → project. What no existing verified block has is a **depthwise
before the expand**.

So this is the R50 shape of problem: *"the architecture is nearly free and the recipe is not"* —
except here the recipe is free too (Imagenette AdamW, same as every other net in that tier), and
what costs is the proof chain.

---

## 1. WHERE IT STANDS — measured

| piece | state | evidence |
|---|---|---|
| `uib` in the **baseline** `Layer` language | ✅ exists | `LeanMlir/Types.lean:39` — `uib (ic oc expand stride) (preDWk postDWk)` |
| baseline param count | ✅ exists | `LeanMlir/Spec.lean:98` — preDW + expand + postDW + proj, each with BN |
| baseline emitter | ✅ exists | `LeanMlir/MlirCodegen.lean`, `SpecHelpers.lean` |
| **UIB VJP oracle** | ✅ **passes at 1.13e-05** | `vjp-oracle-uib` (`lakefile.lean:2486`), `tests/vjp_oracle/README.md` |
| the net, JAX side | ✅ two of them | `jax/MainMobilenetV4.lean` (Conv-S-sized, 10-class Imagenette demo, **4.1M**) and `jax/MainMobilenetV4Imagenet.lean` (faithful Conv-M, ~9.7M, 1000-class) |
| Imagenette accuracy | ✅ **84.58%** @ 80ep | `RESULTS.md` — and per `MainMobilenetV4Imagenet.lean:4-7` this number belongs to the **Conv-S-sized demo**, not Conv-M |
| `uib` in **`VLayer`** | ⛔ **absent** | `LeanMlir/VerifiedSpec.lean:27-113` — 21 constructors, no `uib` |
| verified render | ⛔ absent | no `verified_mlir/mnv4*` |
| verified spec / app | ⛔ absent | no `mobilenetv4Verified` in `VerifiedNets.lean` |

⚠ **Two different nets share the name.** The 84.58% is the *Conv-S-sized* block table. Faithful
Conv-M is ~9.7M and has no accuracy run at all. **Pick the 4.1M demo** for the Imagenette tier —
it is the one with a number to reproduce. Chasing Conv-M here means having no target.

---

## 2. WHAT IS GENUINELY NEW — the collapse table

Following `rsb_a3_r50_verified.md` §1: state what a first look thinks is new, then what is actually
new.

| looks new | actually | why |
|---|---|---|
| "depthwise conv at k=3 and k=5" | **already rendered** | `mbConvSE ic mid oc r k` takes the depthwise kernel `k` as a parameter (`VerifiedSpec.lean:87`) and is certified for EfficientNet |
| "expand/project pointwise + BN" | **already rendered** | `invertedResidual` is exactly this (`:75`), certified for MobileNetV2 |
| "the residual skip and its stride dispatch" | **already rendered** | `invertedResidual`'s stride argument; same dispatch `residualStage`/`bottleneckStage` use |
| "BN running statistics through a mobile block" | **already threaded** | `bnChannels` drives it for mnv2/enet. ⚠ The *JAX* side needed UIB wiring and got it 2026-07-19 (`MainMobilenetV4Imagenet.lean:91`); the verified side inherits nothing from that |
| ~~"a depthwise BEFORE the expand"~~ | **also already rendered** — corrected 2026-08-07, see below | `MobileNetV2RenderB.lean:196` |
| **"four block types"** | **NEW as dispatch, not as ops** | `k = 0` means "omit that DW", so one constructor renders 4 shapes: ExtraDW (both), IB/MBConv (post only), ConvNeXt (pre only), FFN (neither) |

### ✅ PHASE 0 IS DONE (2026-08-07) — and it retired this doc's one "NEW" row

**Result: no new SHlo op, and no new block position either.** Two readings settled it:

1. **The depthwise VJP is kernel-general, not per-`k`.** `Proofs/Architectures/ConvNeXtClose.lean:15`
   certifies `depthwiseConv2d` via `cnx_render_dw7{W,b}_certified`, annotated *"kernel-general —
   pinned below"*, and the descriptor carries `kH kW` explicitly (`StableHLOParse.lean:257`,
   `depthwiseF w b c h w' kH kW e`). Rendered in anger at 3×3 (MNv2), 5×5 (EfficientNet `mbConvSE`)
   and 7×7 (ConvNeXt). So MNv4's `k ∈ {3,5}` is free.
2. ⭐ **A leading depthwise already exists.** The row above claimed no `VLayer` puts a DW upstream of
   the pointwise. It does: `MobileNetV2RenderB.lean:196` is the `t = 1` inverted residual, and it
   emits `.depthwise (c := ic)` **directly on the block input**, then BN → relu6 → project. ConvNeXt's
   block is the same shape one level up (DW → LN → expand → project). The op is channel-parameterised
   (`c`), so the pre-DW is that same constructor at `c := ic`.

▶ **What is actually new is one composition, not a primitive**: ExtraDW's `DW → expand → DW →
project` puts a depthwise on *both* sides of the pointwise expand. Each adjacency in it —
`DW → pointwise` (MNv2 `t=1`, ConvNeXt) and `pointwise → DW` (MNv2 `t>1`, EfficientNet) — is already
certified; nothing has yet composed both in one block. That is a proof-composition question, which
is Phase 2, and it is a much smaller thing than a missing descriptor.

---

## 3. ⚠ THE TRAP — a wrong `k = 0` dispatch is silent

The four variants come from one primitive by *omitting* convs. A renderer that mis-dispatches —
emitting a post-DW where the table says pre-DW, or emitting both where the table says FFN — produces
**a valid net that trains and descends and is not MobileNetV4**. Every structural check passes: the
op set is unchanged, the channel counts are unchanged, only the *order* moves.

This is the same failure class as two already in the record:

* R50's **stride-on-the-3×3** (`VerifiedSpec.lean:46`) — *"putting it on the first 1×1 compiles,
  trains, descends and is a different net"*.
* R50's **2×2 vs 3×3 stem pool** (`:53`) — *"a deviation at an unchanged type is invisible to arity,
  op counts and the prefix audit alike"*.

▶ So the param-count `#guard` is **not sufficient** as the gate here: pre-DW and post-DW at the same
`k` have the *same parameter count*. The gate must be a forward tie against the JAX reference on the
same weights, not a shape or count check.

---

## 3b. ⭐⭐ THE FORWARD TIE RAN (2026-08-08) — order CONFIRMED, one op's padding left

`scripts/mnv4_forward_tie.py` runs `@mnv4_fwd` (iree-compile + iree-run-module) and the ACTUAL
reference `forward()` from `jax/.lake/build/generated_mobilenet_v4.py` on one shared set of random
weights, and compares logits. Result, batch 2, seed 42:

| reference variant | max \|Δ\| |
|---|---|
| as-is (`padding='SAME'` stem) | **6.164e-02** |
| stem patched to symmetric `(1,1)` | **1.788e-06** |

⭐ **One line moves it from 14% to fp noise**, so the stem's padding is the ENTIRE disagreement.

⭐⭐ **THE PRE/POST-DW ORDER IS PINNED.** §3 says a swap is invisible to counts, shapes, group
widths and the types; 1.8e-6 agreement on random weights across 4.1M params and 52 convolutions is
not survivable by a swapped depthwise. All 14 blocks, all four families, the fused stage, the
swish/relu split, the skips and the head are confirmed against the net that produced 84.58%. This
is the gate the doc said nothing could substitute for, and it holds.

### ⛔ The remaining blocker: there is no asymmetric-padding conv descriptor

The reference stem is `conv_bn(…, stride=(2,2), padding='SAME')`; XLA `'SAME'` on 3×3/s2 at 224
pads **(0,1)**. Every conv descriptor in `Proofs/Codegen/StableHLO.lean` emits **symmetric**
padding — `convStrided` gives `[[1,1],[1,1]]`, and the "stride-2 SAME conv" at `:400` means *same
output size*, not XLA `SAME`. Both give 112×112, which is why nothing until this tie could see it.

▶ Three ways out, and it is a judgement call, not a technical one:

1. **Add an asymmetric-pad strided conv** — a new descriptor AND its VJP (the backward has to
   place the same asymmetry). Real proof-layer work. Correct if the verified path should reproduce
   the reference bit-for-bit.
2. **Change the reference to symmetric.** Defensible on paper-faithfulness grounds — this is the
   `conv2d` change of 2026-08-04 applied one net further — but it changes the net, so **84.58% would
   have to be re-run** before it means anything.
3. **Accept and document the deviation.** Cheapest, and the worst of the three: the verified render
   would then not be the net with the published number, which is precisely the ambiguity the tier
   exists to remove.

⚠ Note the repo's existing position, which cuts BOTH ways (`generated_mobilenet_v4.py`'s `conv2d`):
symmetric was adopted for the ResNet family and *"scoped ON PURPOSE: MobileNetV2/EfficientNet emit
their own … with 'SAME' and are TF-origin ports, where asymmetric 'SAME' IS the reference. Do not
'fix' those."* MNv4 is a TF-origin port, so by that rule the RENDER is the side that is wrong.

⚠ And the same question is open one net over: **MobileNetV2's verified render uses the same
symmetric `convStrided` stem**, and its reference is TF-origin too. Nobody has run this tie for
MNv2. That is worth an hour before trusting the mnv2 Imagenette number as a *tied* result.

---

## 3c. ⚠⚠ THE SAME DEFECT IS IN MOBILENETV2 AND EFFICIENTNET — and MNv2 is worse

`VLayer` cannot express padding, so **every** verified render emits symmetric. Measured over the
committed corpus (`verified_mlir/*_fwd.mlir`, 2026-08-08):

> **▶ THE MNv2 TIE HAS NOW RUN — see §3d. The inference below was right about padding, and it
> missed a second, larger convention split (the BN world). Read §3d before acting on this table.**

| render | stride-2 convs | its reference | exposed? |
|---|---|---|---|
| `resnet34_fwd`, `resnet50in_fwd` | `[[3,3],[3,3]]` | symmetric since 2026-08-04 | ✅ no — both sides agree |
| `convnext_fwd` | `[[0,0],[0,0]]` | — | ✅ no — 2×2/s2, SAME and symmetric coincide |
| `vit_fwd` | none | — | ✅ no |
| **`mobilenetv2_fwd`** | **5 × `[[1,1],[1,1]]`** | spec says `.same` | ⛔ **yes, and at all five** |
| **`efficientnet_fwd`** | 3 × `[[1,1]]`, 2 × `[[2,2]]` | TF-origin | ⛔ **likely** |
| `mnv4_fwd` | `[[1,1],[1,1]]` stem | `.same` | ⛔ measured at 6.16e-2 |

⭐ **MNv2 is exposed at FIVE sites where MNv4 is exposed at one.** MNv4's `uib_block` overrides the
padding with an explicit symmetric `(k-1)//2` on its depthwises — which is exactly why patching the
stem alone took the tie to 1.8e-6. MNv2's `sep_conv` instead calls `depthwise_conv(x, dw,
stride=stride)`, and `depthwise_conv`'s default is **`padding='SAME'`** (`Jax/Codegen.lean:683`).
So MNv2's stem *and* its four strided depthwises all differ from the render.

⚠ **This does NOT invalidate MNv2's §1a tie.** That tie is Lean-level — spec ↔ render — and it still
holds: the render faithfully implements the `VerifiedNetSpec`. What is in question is whether that
spec *is* MobileNetV2-as-published at the strided convs. Both can be true, and only the second is in
doubt. Say it that way; the two claims are easy to blur and the blur favours us, which is the
direction to be careful in.

▶ **MNv2's 89.35% (Imagenette, 80ep, 2026-08-06) is the number this touches**, because MNv2 is one
of the five nets whose results are quoted as tied.

---

## 3d. ⭐⭐ THE MNv2 TIE RAN (2026-08-08) — padding confirmed, AND a second split nobody predicted

`scripts/mnv2_forward_tie.py` (new; sibling of the MNv4 one, same method) runs
`verified_mlir/mobilenetv2_fwd.mlir` through iree-compile + iree-run-module against the ACTUAL
`forward()` from `jax/.lake/build/generated_mobilenet_v2.py`, on one shared set of random weights.
159 inputs, batch 32 (the render pins it, and batch BN forbids shrinking it), 53 reference
entries = 52 (W, γ, β) triples + dense. Two conventions are patched independently, and **crossed**,
because with the wrong BN world every padding row is swamped by the same residual and the table
says nothing:

| | pad as-is (XLA `SAME`) | stem sym | dw sym | **pad both symmetric** |
|---|---|---|---|---|
| **BN batch** `(0,2,3)` — the reference's own | 2.841e-01 | 1.745e-01 | 2.804e-01 | 1.551e-01 |
| **BN per-ex** `(2,3)` — what the render computes | 2.725e-02 | 2.312e-02 | 2.022e-02 | **6.080e-06** |

⭐ **One cell is fp noise and fifteen are not.** Matching *both* conventions ties at **6.08e-06**
across 2.2M params and 53 convolutions. So MNv2's block structure is **confirmed**: expand/DW/project
order, the `t=1` two-entry first block, all 17 blocks' channel widths and strides, the relu6
placements, the skip dispatch, the head conv-BN and the dense orientation. Nothing structural is
wrong. What is wrong is exactly two conventions, and they are separable.

### (a) Padding — the doc's inference was right, and it is 5 sites

`verified_mlir/mobilenetv2_fwd.mlir` has 5 stride-2 convolutions and **all 5 are
`pad = [[1,1],[1,1]]`**: the stem plus the four strided depthwises, exactly as §3c predicted from
reading `depthwise_conv`'s `padding='SAME'` default (`jax/Jax/Codegen.lean:679`). Every strided
depthwise sees an even input (112, 56, 28, 14), so XLA `SAME` pads (0,1) at each.

Measuring the padding deviation **reference-against-itself** (no render involved — this is
"how far is the net that was trained from the net that was published"):

| BN world | `SAME` vs symmetric | as % of logit range |
|---|---|---|
| per-example | max 2.725e-02, mean 6.370e-03 | 3.5% |
| **batch** | max 2.911e-01, mean 4.434e-02 | **27.7%** |

⚠ Batch BN **amplifies** the padding deviation by ~10×, because BN couples the examples and the
edge rows shift the statistics rather than just the edge activations. The trained artifact is the
batch-BN one (below), so 27.7% is the figure that applies to it. On random weights, so read it as
"these are different functions", not as an accuracy delta — but a padding fix is not cosmetic here.

### (b) ⚠⚠ THE BN WORLD — a second split, larger than padding, and NOT predicted by §3c

The reason none of the four padding rows tied on the first run: `mobilenetv2_fwd.mlir` reduces its
BN statistics across `dimensions = [2, 3]` and divides by **12544 = H·W** — *per-example*,
instance-norm shaped. The reference reduces `axis=(0, 2, 3)` over **B·H·W**. Measured across the
MNv2 corpus:

| artifact | BN divisor at the stem | world |
|---|---|---|
| `mobilenetv2_fwd` | `dense<12544.0>` | per-example |
| `mobilenetv2_train_step` (SGD) | `dense<12544.0>` | per-example |
| **`mobilenetv2_adam_train_step`** | **`dense<401408.0>`** | **batch** |
| `mobilenetv2_adamdp_train_step` | `dense<401408.0>` | batch |

▶ **`mobilenetv2_fwd.mlir` is NOT a byte-prefix of `mobilenetv2_adam_train_step.mlir`** (checked
directly with `cmp`). This is the documented two-worlds hazard — memory `r34-bn-two-worlds`, "the
split runs along the WRITER, not the net" — still live for MNv2, one net over from where it was
found.

⛔ **AND THE GUARD DOES NOT CATCH IT.** `scripts/regen_verified_mlir.sh check` reports
*"OK — mobilenetv2_fwd.mlir is a byte-identical 1614-line prefix of mobilenetv2_train_step.mlir"*
and goes green. It only ever checks the forward against the **SGD** train step — the one that
shares its world. No pairing in that script compares a forward to an **Adam** train step, so the
artifact that actually trains every quoted verified number is unaudited for this. That is a hole
in the guard, not a fact about MNv2, and it applies to all five nets.

### (c) What this does to the 89.35%

`runs/mnv2_adam_80ep_aug06.log` (2026-08-06, XLA/PJRT, gfx1100) is the run — epoch 80,
3507/3925 = **89.350318%**. Its header loads **three** artifacts:

```
compiled verified_mlir/mobilenetv2_adam_train_step.mlir   ← batch BN   (trains)
compiled verified_mlir/mobilenetv2_fwd.mlir               ← PER-EXAMPLE BN  (⚠ purpose untraced)
compiled verified_mlir/mobilenetv2_fwd_eval.mlir          ← running stats  (scores val)
```

A **per-example** forward is compiled into the same loop, which is the precise shape of the R34 bug
(*"trained the per-example net and scored it with batch statistics… rel 1.13, not a rounding
difference"*). ✅ **TRACED, and it is clean.** `VerifiedTrain.lean:946` creates `fwdSess`
unconditionally, but `:1617` selects `let evalSess := if useRunning then fwdEvalSess else fwdSess`,
and `useRunning = hasBn && !batchStatEval`. MNv2 has 52 BN layers and the run sets no
`LEAN_MLIR_EVAL_BATCHSTATS` (0 occurrences in the log; the header prints *"eval via
@mobilenetv2_fwd_eval"*). So the per-example artifact was **compiled and never invoked** — 1364 ms
of wasted compile, not a wrong number.

▶ **So the BN split does NOT touch the 89.35%.** That run is batch-BN training + running-stats eval,
internally consistent. What remains against it is padding alone (§3d(a)) — which is also the only
thing left against MNv4's 84.58%. Both nets, same single defect, same fix.

⚠ The guard hole in (b) is still real and still worth closing: nothing in
`regen_verified_mlir.sh check` would have caught the per-example/batch divergence, and the reason
this run is fine is a runtime `if`, not an audited invariant.

⚠ Restating the boundary, because it is easy to blur and the blur favours us: **MNv2's §1a tie is
untouched.** Spec ↔ render still holds — the render faithfully implements the `VerifiedNetSpec`.
What §3d establishes is that the *spec* deviates from MobileNetV2-as-published at 5 padding sites,
and that the corpus carries two incompatible BN spellings of it. Different claims; only the first
was ever proven.

### (d) ~~EfficientNet — structurally exposed, NOT measured~~ ⛔ **REFUTED — see §3f**

This section inferred EfficientNet was exposed at **all five** stride-2 convolutions, by analogy
with MobileNetV2's `depthwise_conv` default. **That was wrong, and the tie (2026-08-08, §3f) says
so:** `mbconv_block` computes `pad = ((ksize-1)//2, (ksize-1)//2)` and passes it EXPLICITLY, so
its depthwises are symmetric and the render's `[[1,1]]`/`[[2,2]]` are already correct. EfficientNet
is exposed at **one** site — the stem — like MNv4, not five like MNv2.

▶ Second time an inference-by-analogy in this doc has been wrong about EfficientNet's padding
(§3c said "likely"). The pattern: **`depthwise_conv`'s default only bites when a caller doesn't
override it**, and mnv2's `sep_conv` is the only caller that doesn't. Read the caller, not the
default.

### ▶ THE ANSWER TO THE QUESTION THAT WAS ASKED

**The asymmetric-pad descriptor has at least TWO measured consumers (MNv4, MNv2) and a third
structurally implicated (EfficientNet).** Build it once, properly, with its VJP — and re-run both
ties afterwards. The BN-world split (b) is a *separate* defect that this fix does not touch and
that is currently larger; it needs its own decision.

---

## 3e. ✅ HOW THE DESCRIPTOR WAS BUILT — it was a decimation PHASE, not new padding

§3b called this *"a new descriptor AND its VJP … real proof-layer work"*. It was not, and the
reason is worth keeping.

⭐ **The identity.** The repo already defines the stride-2 conv as
`flatConvStride2 = decimateFlat ∘ flatConv` — a symmetric stride-1 SAME conv on the `2h×2w` grid,
then keep the **even** positions. XLA `'SAME'` at stride 2 is the *same stride-1 conv* read at the
**odd** positions:

```
  convXlaSame_s2 W b X  =  decimateOddFlat (flatConv W b X)
```

because output `ho` then reads `x[2·ho + 1 + kh − (k−1)/2] = x[2·ho + kh − (k−2)/2]`, and `(k−2)/2`
is exactly XLA's `pad_low` at an even input. **The whole asymmetry is a phase shift in the
decimation** — `flatConv` and every one of its VJPs is reused verbatim.

⭐⭐ **And `decimateOddFlat` already existed**, with its VJP, added for ConvNeXt's 4×4/s4 patchify
stem (`StridedConv.lean` §stride-4). So the new op needed **zero new proof obligations** — only
`vjp_comp` compositions of results already closed.

**Verified before writing any Lean**: the rule is `d = 1` at even inputs, `d = 0` at odd (where XLA
`SAME` *is* symmetric, so the existing op is already right). Checked against
`jax.lax.conv_general_dilated(…, 'SAME')` over `H ∈ {224,112,56,28,14,32,16,9,7,15,33} × k ∈
{3,5,7}` — **33 configs, no mismatches**. Every strided site in mnv2/mnv4/enet is at an even input.

### What landed

| where | what | note |
|---|---|---|
| `Foundation/StridedConv.lean` | `flatConvStride2Xla` + differentiability + input/weight/bias VJPs + 2 ℝ-headline correctness theorems | **7 defs, all `[propext, Classical.choice, Quot.sound]`** |
| `Architectures/Depthwise.lean` | `depthwiseStride2FlatXla` + the same five | **6 defs, 3-axiom clean** |
| `Codegen/StableHLO.lean` | `BatchableOp.convStridedXla` **and** `.depthwiseStridedXla` — each: constructor, `den`, `den_batchOp_*` simp lemma, `batchOpDescr` tag, emit case | **5 sites each, ONE file** |
| `Codegen/MobileNetV4RenderB.lean` | stem `.convStrided` → `.convStridedXla` | one line |
| `tests/TestXlaPadOps.lean` + `scripts/xla_pad_op_check.py` | op-level known-answer guard | see below |

### ✅ The op-level guard — because a whole-net tie says "somewhere", not "which op"

`tests/TestXlaPadOps.lean` emits four one-op modules; `scripts/xla_pad_op_check.py` runs each
through IREE and compares against `jax.lax.conv_general_dilated(…, 'SAME')`:

| probe | vs XLA `SAME` | vs symmetric | verdict |
|---|---|---|---|
| `conv_xla` (k=3) | 7.153e-07 | 5.632e+00 | ✓ |
| `dw_xla_k3` | 2.608e-07 | 6.179e+00 | ✓ |
| `dw_xla_k5` | 7.292e-07 | 5.753e+00 | ✓ (the EfficientNet case — `SAME` pads (1,2), total 3 ≠ 2·((k-1)/2)) |
| `conv_sym` (control) | 7.082e+00 | 8.345e-07 | ✓ symmetric |

⭐ **The control is load-bearing.** It asserts the symmetric token does NOT match `SAME` at this
shape — without it, a shape where the two conventions coincide would make the whole probe vacuous
while printing green. The checker fails on "both columns small" rather than treating it as a pass.
The `#guard`s in the Lean file pin the emitted `pad` strings and the `feature_group_count`, so a
depthwise falling through to the dense-conv emit is caught at `lake env lean`, not at IREE.

⚠ **`renderModule` is NOT the right wrapper for a `batchOp` graph** — it takes `g : SHlo retLen` and
returns `tensor<B × retLen>`, i.e. the per-example convention, while a `batchOp` index is already
`B*n`. It fails to unify AND would declare the wrong return type. Wrap by hand (`wrap` in the test).
The failure mode is a `whnf` heartbeat timeout, which reads like a size problem and is not one.

⚠ **The doc's "10 sites across 2 files" warning did not apply.** That is the cost of a top-level
`SHlo` op, which needs `Raw`/`Tok`/parse constructors and the roundtrip proof. A **`BatchableOp`**
maps onto the generic `batched` Raw, so it needs no parser work at all — `lake build Certs` passes
untouched. Worth remembering: **the two op kinds have very different costs**, and this one was the
cheap kind.

⚠ The emit is deliberately a **distinct tag**, not an alias of `"convStrided"`. The bias-grads in
this file legitimately alias because their emitted text is stride-independent; this one's `pad`
differs, so sharing a Raw would emit the wrong bytes.

### ⛔ WHAT IS STILL NEEDED TO SWITCH MNv2 — and why it is a decision, not a task

Both forward tokens exist and are validated. **MNv2's render is still NOT switched**, and that is
deliberate: its render carries a full backward, so flipping the forward alone would be exactly the
silent wrong-gradient this work exists to prevent. MNv4 was safe to switch because it has no
backward at all yet.

Measured by grepping the two renders for strided ops:

| path | artifact it owns | strided ops needing an Xla peer | state |
|---|---|---|---|
| **batched** (`MobileNetV2RenderB`) | `mobilenetv2_adam_train_step` — **what trained 89.35%** | `convStrided` ✅, `depthwiseStrided` ✅, `depthwiseStridedBackBatched`, `depthwiseStridedWeightGradB`, `convStridedWeightGradB` — plus `convStridedBiasGradB` / `depthwiseStridedBiasGradB` | 2 of 7 done |
| **per-example** (`MobileNetV2Render`) | `mobilenetv2_fwd`, `mobilenetv2_train_step` (SGD) | `depthwiseStridedF`, `depthwiseStridedBack`, `depthwiseStridedWeightSgd`, `depthwiseStridedBiasSgd`, `convStridedWeightSgd`, `convStridedBiasSgd` | 0 of 6 done |

⭐ **The two bias-grads are nearly free, and for a reason worth stating.** A conv's bias gradient is
`Σ_{batch,spatial} dy` — `∂y/∂b = 1` at every output position regardless of which input taps fed
it — so it is padding-independent exactly as the file already documents it to be
stride-independent. Their **emitted bytes do not change at all**; only the `den` must point at
`…Xla_bias_grad_has_vjp`. So the real emit work is **3 ops**, not 5.

⚠⚠ **And the per-example path is the expensive kind.** Those six are top-level `SHlo` ops, not
`BatchableOp`s — so they DO cost the 10-sites-across-2-files with the parser roundtrip under
`Certs`. The batched path stays cheap.

### ⛔⛔ THE ACTUAL BLOCKER IS NOT TECHNICAL: switching MNv2 VOIDS 89.35%

This is the same trade §3b listed as option 2 and rejected for MNv4 — except here it lands on a net
that **already has a trained verified number**. Changing the padding changes the net, so
`runs/mnv2_adam_80ep_aug06.log`'s 89.350318% would become a number for a net that no longer exists
and would have to be re-run (80 epochs). MNv4 had no such cost: its 84.58% belongs to the *JAX
baseline* path, which does not move.

▶ **So this is Brett's call, not a task to grind out.** The options:
1. **Switch and re-run** — 89.35% is re-measured on the faithful net. Costs one 80-epoch run plus
   the 3 emit ops (+6 if the SGD path is also wanted).
2. **Switch the batched path only**, leave the per-example SGD artifacts symmetric and say so — but
   that deepens the two-worlds split §3d(b) already flags as a guard hole. Not recommended.
3. **Document and leave** — the cheapest, and the one §3b already called the worst of three: the
   verified render then isn't the net anyone has a reference number for.

✅ **EfficientNet has now been tied (§3f) and it changes this picture.** It needs `convStridedXla`
only — **one site, no depthwises** — so it is a one-line render change plus the 3 backward peers.
But its *dominant* deviation isn't padding at all: its JAX reference uses **relu** at the stem/head
where the render (correctly) uses **swish**, worth 51% of logit range against padding's 10%. So
EfficientNet's repair is mostly on the **reference** side, and it is independent of this decision.

---

## 3f. ⭐⭐ THE EFFICIENTNET TIE RAN (2026-08-08) — and padding was the SMALLER of two problems

`scripts/enet_forward_tie.py`, `verified_mlir/efficientnet_fwd.mlir` vs the real `forward()` from
`generated_efficientnet_b0.py`. 214 inputs, batch 32, 82 reference entries (49 triples + 33 pairs).
**Three** axes crossed — activation, BN world, stem padding:

| act (stem/head) · BN | stem pad as-is (`SAME`) | stem pad symmetric |
|---|---|---|
| relu · batch | 4.548e-01 | 4.532e-01 |
| relu · per-example | 4.260e-01 | 4.306e-01 |
| **swish · batch** | 6.767e-02 | **1.132e-06** ✅ |
| swish · per-example | 4.424e-02 | 4.418e-02 |

⭐ **One cell in eight is fp noise.** So the render's structure is confirmed — all 16 MBConv blocks,
the SE gates, the k=3/k=5 split, the skip dispatch, the head and the dense — and exactly **two**
conventions separate it from its reference. BN world already agrees (both batch).

### ⚠⚠ The two deviations point in OPPOSITE directions

| deviation | size (ref vs itself) | which side is right |
|---|---|---|
| **stem+head `relu` vs `swish`** | **4.605e-01 max — 51.0% of logit range** | the **RENDER**. EfficientNet-B0 uses SiLU/swish throughout, including stem and head (`EfficientNetRender.lean:722,729`). The JAX reference uses relu **not by design**: the generic `.convBn` layer emitter appends `jax.nn.relu(x)` unconditionally and has no activation parameter |
| **stem padding symmetric vs XLA `SAME`** | 9.140e-02 max — 10.1% of logit range | the **REFERENCE**. TF-origin port, same as mnv2/mnv4 |

▶ **This is the first case in the sweep where the reference is the deviating side**, and it is the
dominant term — 5× the padding effect. A "fix the render" reflex would make things worse here.

### ⛔ What it means for the numbers — and they are BOTH real runs

* `runs/enet_adam_80ep_aug06.log` — the **verified** path, epoch 80: **90.06%**. Swish stem/head,
  symmetric stem padding.
* `RESULTS.md` — the **JAX baseline** table: **87.58%**. Relu stem/head, XLA `SAME` stem padding.

⚠ **These are different nets, so 90.06% is not a reproduction of 87.58% and must not be presented
as one** — the verified path is not "the baseline, verified" for EfficientNet; it is a *different
and more paper-faithful* net that happens to score higher. Nobody has run the baseline recipe on
the render's actual architecture, so there is no matched pair for this net at all. That is a
reporting problem, not a correctness one, and it is worth fixing before either number is quoted as
tying the other.

▶ **Cheapest repair, and it makes the render the reference rather than the other way round:** give
the JAX `.convBn` layer an activation parameter (it has none — the relu is hardcoded), set the
EfficientNet spec's stem/head to swish, and re-run the baseline. That closes the 51% term in the
direction of the paper. The 10% padding term then closes by switching the render's stem to
`convStridedXla` (§3e), which is a one-line change here — EfficientNet has **no** exposed
depthwises, so it needs `convStridedXla` only, plus the 3 backward peers if the train step follows.

### 🔧 Two gotchas worth keeping

1. **`iree-run-module --device=local-task` dies on this module: exit 245, EMPTY stderr, no output
   file, after printing `EXEC @efficientnet_fwd`.** `--device=local-sync` runs the same vmfb fine in
   ~12 MiB. A silent 245 is a device/threading problem, NOT a bad render — do not go hunting in the
   MLIR. (The mnv2/mnv4 ties use `local-task` happily; this net is the first that doesn't.)
2. **Patch order matters when axes overlap textually.** The activation patch anchors on the stem
   line *including* its `padding='SAME')`, so it must run before the padding patch rewrites that
   text. Applied the other way round it fails with "patch site not found", which reads like a
   generator change and isn't.

---

## 3g. ✅ THE BACKWARD OP SET FOR MNv2 IS BUILT AND GRADIENT-CHECKED (2026-08-08)

Brett authorised re-running the nets, so MNv2's switch is on. **The op layer is complete for the
batched (Adam) path** — the one that trains the quoted number. Five backward peers, each 4 sites in
`StableHLO.lean` (constructor, `den`, `skel`, and emit only where the bytes change):

| op | emit | `den` cert |
|---|---|---|
| `convStridedXlaWeightGradB` | new, `pad → [p-1, p+1]` | `flatConvStride2Xla_weight_grad_has_vjp` |
| `depthwiseStridedXlaWeightGradB` | new, `pad → [p-1, p+1]` | `depthwiseStride2Xla_weight_grad_has_vjp` |
| `depthwiseStridedXlaBackBatched` | new, `pad → [p+1, p-1]` ⚠ | `depthwiseStride2FlatXla_has_vjp` |
| `convStridedXlaBiasGradB` | **none — aliases `"convBiasGrad"`** | `flatConvStride2Xla_bias_grad_has_vjp` |
| `depthwiseStridedXlaBiasGradB` | **none — aliases `"depthwiseBiasGrad"`** | `depthwiseStride2Xla_bias_grad_has_vjp` |

⭐ The bias grads needed no emitter at all: `∂y/∂b = 1` at every output position regardless of which
input taps fed it, so `Σ_{batch,spatial} dy` is **padding-independent** for the same reason the file
already documents it to be stride-independent. Only the `den` distinguishes them.

### ⚠⚠ THE GUARD CAUGHT A REAL BUG — and it is the one this whole thread is about

The backward pads were **derived**, not copied from an oracle. I derived `[p-1, p+1]` for all three
by symmetry with each other. That is **right for the two weight grads and WRONG for the input-VJP**,
which shifts the *other* way: `[p+1, p-1]`. The reason is that the input-VJP reverses its kernel
(`stablehlo.reverse`, dims [2,3]), and the reversal flips the sign of the index shift.

▶ The wrong version **type-checked, produced the right shape, and would have trained and
descended.** It was caught only because `scripts/xla_pad_op_check.py` compares against `jax.vjp` of
the actual Xla-padded forward — it reported 2.613e0 against *both* references. A numeric sweep over
(upsample phase, `pad_low`) then pinned the true answer at k=3 **and** k=5.

**This is the exact failure mode §3 named as the reason a `#guard` on shapes cannot be the gate**,
arriving in the backward instead of the forward. It is now pinned by a `#guard` on the emitted pad
string plus the vjp check. Do not "fix" the input-VJP to match its siblings.

Final state — `scripts/xla_pad_op_check.py`, 7 probes:

```
conv_xla       7.153e-07 vs SAME   5.632e+00 vs symmetric   ✓
dw_xla_k3      2.608e-07                6.179e+00           ✓
dw_xla_k5      7.292e-07                5.753e+00           ✓   (k=5: SAME pads (1,2))
conv_sym       7.082e+00                8.345e-07           ✓   control — proves non-vacuity
dw_xla_back    1.192e-07 vs jax.vjp     2.499e+00 vs sym-vjp ✓
conv_xla_wgrad 1.431e-06                1.125e+01           ✓
dw_xla_wgrad   3.338e-06                8.191e+00           ✓
```

### ⚠ BLAST RADIUS — ONE renderer writes SEVEN train steps

`mobilenetv2AdamTrainStepFaithfulB` is the sole writer for every MNv2 train step, so switching the
render moves **all of them at once**. There is no way to switch "just Adam". Measured by reading
the `#eval` writers in `MobileNetV2RenderB.lean` and grepping `runs/` for what each one backs:

| artifact | args | number it backs | re-run? |
|---|---|---|---|
| `mobilenetv2_adam_train_step` | B32, 10cls | **89.35%** (`mnv2_adam_80ep_aug06.log`) | **YES** |
| `mobilenetv2_rms_train_step` | B32, rmsprop | **76.03%** (`mnv2_rms_80ep_aug02.log`) | **YES** |
| `mobilenetv2_adamdp_train_step` | B32, 2 replicas | DP timing/eval probes only (`runs/xla2h/mnv2_dp_*`) | optional |
| `mnv2in_adam64_train_step` | B64, 1000cls | — no run exists | no |
| `mnv2in_adamdp64_train_step` | B64, 4 replicas | — no run exists | no |
| `mnv2in_rms64_train_step` | B64, rmsprop | — no run exists | no |
| `mnv2in_rmsdp64_train_step` | B64, 4 rep, rmsprop | — no run exists | no |

▶ **So it is TWO headline re-runs, not one**: Adam 89.35% *and* RMSProp 76.03%. The DP variant
shares the identical chain, so it is automatically correct once switched — only its probes would
want re-timing, and those are cheap.

✅ **The ImageNet tier costs nothing.** All four `mnv2in` artifacts move, but nobody has trained on
them, so there is no number to void. ⚠ And do **not** confuse this with README's ImageNet
MobileNetV2 **68.33%** — that is the **Tier-4 JAX/baseline** number (`jax/runs/*/RESULTS.md`, the
unverified emitter), not these artifacts. It is untouched by any of this.

⚠ Two older Imagenette numbers also came off `mobilenetv2_adam_train_step` — 86.73%
(`mobilenetv2_xla_80ep_jul29.log`) and 86.81% (`mobilenetv2_verified_crop_gpu0.log`). Both already
predate the BN-world fix, so they were superseded before this; just don't resurrect them.

⚠ **Checkpoints are per-VARIANT and outlive the artifact they trained on** (memory:
`xla-pjrt-thread`). Every saved mnv2 checkpoint becomes invalid at the switch — delete rather than
resume, or the re-run silently continues the old net.

## 3h. ⚠ MNv2 IS SWITCHED IN CODE — BUT THE WHOLE-NET GATE IS NOT GREEN YET (2026-08-08)

**Done, and verified:**
* Two more top-level `SHlo` ops for the per-example chain — `flatConvStridedXlaF`,
  `depthwiseStridedXlaF` — the **expensive** kind: constructor, `den`, faithfulness `rfl`, `Raw`,
  `Tok`, `skel`, `toToks`, `emitTok`, parse case AND the roundtrip proof (10 sites × 2, across
  `StableHLO.lean` + `StableHLOParse.lean`). `lake build Certs` green, so the roundtrip holds.
* `mnv2FwdChain` takes `xlaPad`, passed `true` **only** by the two `fwd_eval` writers.
* `MobileNetV2RenderB` switched: 2 forward + 5 backward ops → their Xla peers.
* **Nine artifacts re-rendered**, exactly the predicted set: 7 train steps + 2 `fwd_eval`. All five
  stride-2 sites now `pad = [[0, 1], [0, 1]]`.
* The SGD pair (`mobilenetv2_fwd`, `mobilenetv2_train_step`) is deliberately **untouched** and
  still symmetric — self-consistent, and its byte-prefix audit still passes.

⚠ **`@mobilenetv2_fwd` and `@mobilenetv2_fwd_eval` are now DIFFERENT NETS on purpose.** Consequence
worth knowing: `LEAN_MLIR_EVAL_BATCHSTATS=1` scores through `@mobilenetv2_fwd`, so under an Adam run
that diagnostic now measures the wrong *architecture*, not just transductively. It was already
labelled non-reportable; it is now cross-net too.

### ✅ THE TIE IS GREEN (2026-08-08) — and finding it exposed a THIRD deviation

`scripts/mnv2_forward_tie.py --eval --diag --tol 1e-3`, on `@mobilenetv2_fwd_eval` with μ=0/var=1
frozen stats, He-init weights, γ=1, β=0:

| variant | **pre-switch artifact (control)** | **switched artifact** |
|---|---|---|
| **stem pad as-is (XLA `SAME`)** | 1.202e+01 | **1.535e-04** ✅ |
| stem symmetric | 1.366e+01 | 1.032e+01 |
| dw symmetric | 1.263e+01 | 1.299e+01 |
| **both symmetric** | **1.373e-04** ✅ | 1.202e+01 |

⭐ The columns swapped, and the **control ties exactly where it should** — that is what makes this
a gate rather than a hopeful number. Relative error 3.13e-06 against logits spanning ±49.

### ⚠⚠ THE THIRD DEVIATION: the render is relu6 at stem/head, the reference is relu

Debugging the probe surfaced a real difference nobody had recorded. `@mobilenetv2_fwd*` applies
**relu6** at the stem and head; `generated_mobilenet_v2.py` applies plain **relu** — because the
generic `.convBn` layer emitter appends `jax.nn.relu(x)` unconditionally and has no activation
parameter. **MobileNetV2 as published is ReLU6 throughout, so the RENDER is right and the
REFERENCE deviates** — the identical defect EfficientNet has with swish (§3f), from the identical
emitter.

⭐⭐ **It is INERT at `--scale 0.1` and that is why every earlier tie missed it**: activations never
reach 6, so relu6 and relu agree pointwise, and the original §3d tie passed at 6.08e-06 with the
mismatch present. He init makes activations large enough to clamp, and it appears immediately.
▶ **A convention difference that is inert on your test inputs is still a difference in the net.**
Small-magnitude probes systematically hide saturating-activation mismatches; this is now the second
net where the `.convBn` emitter's hardcoded relu was invisible until the weights got realistic.

▶ **Consequence for the numbers, same shape as §3f:** MNv2's verified **89.35%** (relu6) and the
JAX baseline **87.09%** (relu) are **different nets**, so neither reproduces the other. The repair
is the same one §3f recommends and it is now owed for two nets: give `.convBn` an activation
parameter.

### ~~⛔ THE BLOCKER~~ — resolved; kept for the debugging record

`scripts/mnv2_forward_tie.py --diag` still targets `@mobilenetv2_fwd`, which is (correctly) still
symmetric — so it cannot see the new net. I added an `--eval` mode to tie `@mobilenetv2_fwd_eval`
by feeding μ=0 / var=1 and freezing the reference's BN to match, which avoids exercising stat-slot
ordering. **It does not work yet, and the control proves it is the PROBE, not the render:**

| attempt | result |
|---|---|
| v1: flat 0.1 weight scale | every padding variant ~1e-7 — **vacuous**. Frozen `var=1` does no renormalisation, so the trunk decays over 52 layers and the logits are pure bias. A green tie measuring nothing. |
| v2: He init, γ=1, β=0 | non-vacuous (padding worth 12.2% of logit range) but **nothing ties** — ~1.7e1 |
| **control: v2 against the PRE-SWITCH artifact** | **also fails, 1.75e1** ⇒ the probe is unsound |

▶ **Do not read anything into v2's numbers.** A probe that fails on a known-good artifact cannot
convict the new one. Candidates for the bug, in order: the eval BN's exact formula (is `var`
pre-ε? std vs variance?); whether the 104 stat inputs really are the last 104 and μ/var-interleaved;
and relu6 saturation under He init. The control gives a clean target — **make the pre-switch
artifact tie first**, then run the same probe on the new one.

⚠ Two process notes this cost real time:
1. A `--eval` flag that **overwrites** `--mlir` silently made the control re-run the *new* artifact
   and "confirm" it. Defaults must yield to explicit arguments, or a control is not a control.
2. Added a **vacuity guard**: the script now exits if the two padding conventions are
   indistinguishable on the chosen weights, instead of printing a green tie. v1 would have passed.

### ▶ BOTH RE-RUNS LAUNCHED (2026-08-08)

```
HIP_VISIBLE_DEVICES=0                        .lake/build/bin/mobilenetv2-verified-adam-xla data
HIP_VISIBLE_DEVICES=1 LEAN_MLIR_VARIANT=rms  .lake/build/bin/mobilenetv2-verified-adam-xla data
  -> runs/mnv2_adam_80ep_xlapad_aug08.log   (replaces 89.35%)
  -> runs/mnv2_rms_80ep_xlapad_aug08.log    (replaces 76.03%)
```
Both cold-started: step-0 loss 2.4999, epoch 1 = 21.07% (adam) / 10.68% (rms). The old adam run's
epoch 1 was 22.01%, so the trajectory is in family.

### 🔧 THREE LAUNCH TRAPS, all of which produced a normal-looking wrong run first

1. **The data arg is the ROOT, not the dataset dir.** `VerifiedTrain.loadData` appends
   `/imagenette` itself (`:504`), so it is `… adam-xla data`, NOT `data/imagenette`. Passing the
   latter fails only at the loader, *after* all three artifacts compile and the full header prints
   — it looks like a data problem when it is an argv problem.
2. ⚠⚠ **A STALE CHECKPOINT MADE THE ADAM RUN "SUCCEED" INSTANTLY.**
   `.lake/build/mobilenetv2_adam_ckpt_xla.bin` (epoch 80, from the 89.35% run on the OLD symmetric
   net) was picked up, and the run printed `▸ resuming from checkpoint at epoch 80` followed by
   `done (trained …)` and exited **zero**. Nothing about that output says "I did nothing", and had
   it not been read closely the next step would have been to quote the old number as the new one.
   The blob is size-guarded but NOT architecture-guarded: same param count, different net, silent
   resume. This is the `xla-pjrt-thread` memory's "checkpoints outlive the artifact" warning
   landing exactly as written — and it was flagged in this very doc a few sections up and still
   missed, because the first search looked in `runs/` instead of `.lake/build/`.
   ▶ Checkpoints live at `.lake/build/<slug>_<variant>_ckpt{_xla}.bin{,.epoch}`. Move both aside.
3. ⚠ **The RMS checkpoint was overwritten before I caught this.** The contaminated first launch
   resumed the Aug-02 rms checkpoint and wrote over it, so the weights behind **76.03% are gone**
   — only the log survives. The number itself is still on record and that net is being re-run
   anyway, so nothing published is lost; but it cannot be resumed or re-scored from weights.
   Everything moved aside carries the `.pre-xlapad-aug08` suffix, following the repo convention.

### ~~⛔ SO THE 80-EPOCH RE-RUNS HAVE NOT BEEN STARTED~~ — superseded, see above

Phase 4's own rule: *"Do not start this before the forward tie is clean. A number off a render that
disagrees with the reference is a number for a different net, and it would look completely normal."*
That applies exactly here. The op-level evidence is strong — all 4 forward and 3 backward ops match
JAX/`jax.vjp` (§3g), and the artifacts carry the right pads at the right 5 sites — but op-level
correctness is not the whole-net gate, and this doc has twice been wrong when it reasoned instead of
measuring. **Fix the probe, get the tie, then launch.**

### ▶ WHAT REMAINS FOR THE MNv2 RE-RUN

1. **Switch `MobileNetV2RenderB`** — stem + 4 strided depthwises forward, and the 5 backward ops.
   ~7 line changes. ⚠ Deliberately NOT done yet: switching the train step while
   `mobilenetv2_fwd_eval.mlir` is still symmetric would leave the trained and scored nets
   disagreeing, which is the §3d(b) defect all over again. Both must land together.
2. **The eval forward — and there are TWO of them**, `mobilenetv2_fwd_eval.mlir` and
   `mnv2in_fwd_eval.mlir`. Both come from the same slug-parameterised chain, so it is one change
   covering both. They are written by the **per-example** `MobileNetV2Render`, whose chain uses
   top-level `SHlo` ops (`depthwiseStridedF` etc.) — the expensive kind, 6 of them with the parser
   roundtrip. ▶ **Recommended instead: render them from `MobileNetV2RenderB` using
   `BatchableOp.bnEval`**, which already exists. That needs no new ops, and it *fixes* §3d(b) by
   putting the eval forwards in the same file and world as the train steps they actually partner.
   Cost is a ~120-line chain and getting the 315-input signature right.
   ⚠ `mnv2in_fwd_eval` must move too even though no `mnv2in` run exists — leaving it symmetric
   would plant the train/score mismatch for whoever runs that tier first, which is precisely how
   §3d(b) happened in the first place.
3. **Verify**: re-run `scripts/mnv2_forward_tie.py`; success is the as-is column at ~1e-6 in the
   batch-BN row (it currently ties only at per-example BN + symmetric pad, 6.08e-06).
4. **Re-run 80 epochs** and replace 89.35%.

⚠ The per-example SGD pair (`mobilenetv2_fwd`, `mobilenetv2_train_step`) stays symmetric under this
plan — self-consistent, so its byte-prefix audit still passes, but it is then a *different net* from
the Adam pair. That is a deliberate, documentable split and it should be written into
`RESULTS.md`, not left implicit.

---

## 3i. ✅ THE BACKWARD IS BUILT AND GRADIENT-TIED (2026-08-09) — zero new ops

**Phase 2's gate is met.** `verified_mlir/mnv4_{fwd,fwd_eval,adam_train_step}.mlir` all render and
`iree-compile`; the train step is 583 inputs / 581 outputs (158 params × θ/m/v + lr/bc1/bc2 + 104
BN stat slots + `%onehot`), and its gradient ties `jax.grad` of the reference.

### ⭐ It needed NO new SHlo ops — every one already ships in another net

§4's Phase 2 expected proof-layer work. There was none. Checked op by op: `selectPosB` (R34's relu
mask), `swishBackB` (EfficientNet), `depthwiseBackBatched` / `depthwiseStridedBackBatched` /
`depthwiseStridedWeightGradB` (mnv2/enet), `convStridedBackBatched` / `convStridedWeightGradB`
(R34/R50/ConvNeXt), `convStridedXlaWeightGradB` (the mnv2 stem, §3g). MNv4's backward is pure
composition — which is §6's claim ("a family from one constructor") landing on the backward too.

### ⭐⭐ ONE block table, five consumers

The 14 rows were being hand-written **four** times (signature, stat slots, forward, backward). They
are now one `List UibSpec`, and `uibFwdDispatch`/`uibBackDispatch` read the same row — so the
forward and the backward cannot disagree about which family a block is, which is §3's trap.
Collapsing the four transcriptions re-rendered **byte-identical** to the hand-written chain, which
is both the check on the refactor and the evidence the table is right.

`@mnv4_fwd`, `@mnv4_fwd_eval` and the train step are all one `mnv4FwdChainB` call switched by
`BnMode`. ⭐ **So §3d(b)'s hole cannot exist here**: `mnv4-train-smoke` asserts the train step
contains the forward module's body *verbatim*, as a string. That is the pairing
`regen_verified_mlir.sh check` fails to form for the other five nets.

### ⚠⚠ THE GRADIENT TIE — and the two controls that stopped it convicting the wrong thing

`scripts/grad_tie.py --net mnv4`. The train step returns updated parameters, not gradients, so the
gradient is recovered **exactly**: AdamW's `m' = β₁·m + (1−β₁)·g` at **m = 0** is `0.1·g`, so
`g = 10·m'`, independent of lr, the bias corrections and the decoupled weight decay. The reference
side is `jax.grad` of the **label-smoothed** CE — *not* the reference file's own `loss_fn`, which is
plain CE (`generated_mobilenet_v4.py:1120`); tying a smoothed backward against an unsmoothed loss
mismatches on every parameter, which is §2b's R34 bug in mirror image.

The raw run reported **116/158 parameters off by ~1e-2**, and two plausible explanations were
measured and **both refuted** before the real one was found:

| hypothesis | how it was killed |
|---|---|
| fp32 conditioning | an **f64 control**: the fp32 reference tracks f64 at ~1e-5, so 1e-2 is 1000× the arithmetic floor |
| the symmetric strided-depthwise backward is wrong (the localisation pointed at the block whose dx it produces, and it had **never** had a known-answer check) | three new probes in `xla_pad_op_check.py` — `dw_sym_back_k3/k5`, `dw_sym_wgrad_k5` — all match `jax.vjp` to 5e-7 |

⭐⭐⭐ **The actual cause: a ReLU net's gradient is DISCONTINUOUS in its forward.** The render and the
reference are two fp32 algorithms agreeing to ~1e-6; wherever a pre-activation sits within that of
zero, `1[x>0]` disagrees and that position's cotangent moves by O(dy). The fingerprint is exact —
**one bad channel per BN** (a β gradient is `Σ_{b,h,w} dy`, so one bad spatial position corrupts
exactly one channel), the same channel in both the γ and β rows, growing with depth as more masks
are crossed. Setting every BN β to +5 (`--nokink`) puts every pre-activation 5σ clear of zero and
the whole net drops to the floor: **0/147 live parameters over 10× the control, worst block 7.7×,
where the raw run was 1000–10000×.**

⚠ Three decades separate passing from the observed failure, so the 10× line is not slack. And it
was **verified to fail**: swapping the expand BN's γ and β gradients — same shapes, same arity,
every structural gate still green — takes it to **21/147 failing, worst 3e4**, and independently
trips the dead-parameter check.

### 🔧 What is worth keeping from this

1. **A gradient tie between two fp32 ReLU nets has a floor set by the forward tie, not by
   arithmetic.** Budget ~1 corrupted channel per BN per 1e-6 of forward disagreement. Any whole-net
   gradient tie in this repo will hit this; `--nokink` is the way through.
2. ⭐ **β gradients are the cotangent trace.** `Σ dy` has no weight contraction in the way, so
   reading the `*bt` rows in forward order localises where a cotangent first goes wrong. The weight
   rows blur the boundary; β does not.
3. **Some β gradients are structurally ZERO** — a BN backward's output has zero per-channel sum and
   a 1×1 conv-back preserves it, so any BN fed through that path has `Σ dy = 0` exactly. Relative
   error on those is 0/0 and they dominate every table unless excluded; they get an absolute check
   instead.
4. ⚠ **Inference-by-analogy was wrong a THIRD time** (§3d(d) and §3c were the first two, both about
   EfficientNet's padding). Here it was "the symmetric strided depthwise backward must be the bug".
   It had genuinely never been probed — that gap was real and is now closed — but it was not the
   cause. **Probe the op before rewriting it.**
5. `--device=local-task` gives **exit 245 with empty stderr** on these modules; `local-sync` runs
   the same vmfb. §3f's gotcha, now handled by an automatic fallback that says which device ran.

### ▶ WHAT PHASE 3-REST STILL OWES

The three artifacts exist and are gated. Not yet done: `mobilenetv4Verified` in `VerifiedNets.lean`,
the `Resnet50AdamCommon`-shaped driver pair, and the `check_render_coverage.py` entry — the
"three `#eval` lines + a 50-line Common" part §4 calls mechanical. Then Phase 4's 80-epoch run
against **84.58%**.

---

## 4. PHASES

### Phase 0 — confirm the op set (do this FIRST, it is the load-bearing assumption)
Read the depthwise descriptor the `mbConvSE` render emits and confirm it is expressible at the
pre-DW position and at every `(k, stride)` the block table uses — `k ∈ {3, 5}`, stride ∈ {1, 2}.
*Gate*: name the descriptor, or name the missing one. Cheap, and it decides whether the rest is
engineering or research.

### Phase 1 — `.uib` in `VLayer`
Add the constructor + its param spec, mirroring `Spec.lean:98`'s arithmetic.
*Gate*: a `#guard` pinning the `VLayer` param count against the baseline `Layer` count for the whole
block table — the same two-list shape as `toSpecs == XLayout.specs`, and for the same reason.

### ✅ Phase 1b — the fused block (added 2026-08-07, MISSED by the original scoping)
`VLayer.fusedMbConvNB` + `fusedMbConvFwdStridedB`. Phase 0 scoped the UIB *block* and not the
*net*: stage 0 is `.fusedMbConv 32 48 4 3 2 1 false` and `fusedMbConv` had never existed on the
verified path (the verified EfficientNet is B0, `mbConvSENB` throughout). ⚠ It uses **swish**, not
relu — a deliberate paper deviation, because both emitters behind the 84.58% do
(`Jax/Codegen.lean:1031`). Deliberately narrower than the baseline constructor: no `nBlocks`, no
`useSE`, so no layout exists that the renderer cannot emit.

### ✅ Phase 3-forward — DONE (2026-08-08)
Built: `uibFwdSkipB` / `uibFwdPreStridedB` / `uibFwdPostStridedB` (three functions, because
`.depthwise` and `.depthwiseStrided` differ in INPUT type and a stride-polymorphic block cannot
typecheck), `fusedMbConvFwdStridedB`, `mnv4FwdChainB`, `mnv4ShapeList`/`mnv4SigList`,
`mnv4FwdFaithfulV`.

*Gates standing*: `uib-layout-tie` (3,737,088 + 43,360, `VLayer.toSpecs` vs baseline
`Layer.nParams`) · `mnv4-fwd-smoke` (32 regular convs / 20 depthwise / 1 swish / 36 relu, the
per-block depthwise histogram, `%zb` binding, signature ↔ layout shape-for-shape, 4,124,426 params
= `RESULTS.md`'s 4.1M) · `iree-compile` accepts the module (204,622-byte vmfb) ·
`scripts/mnv4_forward_tie.py` (§3b).

### ✅ Phase 2 + most of Phase 3-rest — DONE 2026-08-09 (codegen)
See §3i. `@mnv4_fwd_eval` and `@mnv4_adam_train_step` render, compile, and the **whole-net gradient
tie passes**. Zero new SHlo ops were needed. The paragraph below is the original scoping, kept
because its sizing estimate was roughly right (~700 lines) and its gate was the correct one.

### ▶ Phase 2 — the UIB block VJP (proof side) — NOT STARTED, **and now unblocked**
Four shapes from one primitive. The depthwise and pointwise VJPs exist (mnv2/enet); what is new is
the composition with a leading DW.
*Gate*: 3-axiom clean, and the existing `vjp-oracle-uib` (1.13e-05) as the numerical control.
✅ The precondition is met: the forward is final (§3e), so the VJP can be written once against it.
**This is the next thing to build.** Sizing, from the peers: `MobileNetV4RenderB.lean` is 438 lines
of forward; `MobileNetV2RenderB.lean` is 972 total and `ResNet34RenderB.lean` 938, so expect
**~500–700 lines** of backward + train-step render — and MNv4 needs one backward per stride
dispatch, mirroring the three forward variants (`uibFwdSkipB`/`PreStrided`/`PostStrided`) plus the
fused stage. There is **no MNv4 backward of any kind today** (grep: zero hits).

### Phase 3-rest — the eval forward, the train step, the spec, the app — NOT STARTED
`mnv4_{fwd,fwd_eval,adam_train_step}` at `nClasses := 10`, B = 32, plus `mobilenetv4Verified` and an
`Resnet50AdamCommon`-shaped driver pair. ⭐ **The R50/Imagenette work of 2026-08-07 is the template
and it is three `#eval` lines + a 50-line Common** — that part is genuinely mechanical.
⚠ The eval forward needs **running-BN** sites, where everything above is batch BN.
*Gate*: `check_render_coverage.py`; and note that `lake build <exe>` does **not** run the renders
(learned the hard way on R50, then again here — the exe builds green against artifacts that do not
exist, and fails at first invoke).

### Phase 4 — the run
80 epochs, bs32, AdamW — the Imagenette tier's schedule.
*Gate*: **84.58%**, the baseline path's number for this block table.
~~⚠ Do not start this before the forward tie is clean.~~ ✅ **Precondition met 2026-08-08** — the
tie passes against the unpatched reference at 1.423e-06 (§3e). Still gated on Phases 2 and 3-rest,
which is what actually produces a train step to run.

---

## 5. WHAT WOULD MAKE THIS NOT WORTH DOING

* ~~The forward tie cannot be made to pass.~~ **Retired 2026-08-08** — it passes at 1.8e-6 modulo
  one op's padding, so the block order is settled and this risk is gone.
* The asymmetric-pad descriptor turns out not to be batch-invariant — then it is a `BatchableOp`
  question first and an MNv4 question second, and the blast radius (§3c) makes it a repo decision
  rather than a net one.

---

## 7. WHAT THIS SESSION LEARNED THAT IS NOT ABOUT MNv4

Three findings that outlive this net:

1. **`VLayer` cannot express padding.** Zero occurrences of `Padding`. Every verified render is
   symmetric by omission, while the baseline specs declare `.same`. This is the root cause of §3b
   and §3c and it is a gap in the verified layer language, not a transcription error.
2. **Scoping a BLOCK is not scoping a NET.** Phase 0 cleared UIB and missed `fusedMbConv` entirely,
   which is the same failure the doc had already written down as the R50 precedent
   (`BatchableOp.sigmoid`). Enumerate the spec's layer list against `VLayer`'s constructors before
   declaring an op-set verdict.
3. **A gate that can fail falsely is worse than a loose one.** `mnv4-fwd-smoke`'s first version
   reported 42 regular convs instead of 32 because `"feature_group_count = 1"` substring-matches
   `= 160`. It failed on a *correct* render. Parse numerically.

Three more from the MNv2 tie (2026-08-08):

4. ⭐ **A tie with two unmatched conventions attributes NEITHER.** The first MNv2 run patched only
   padding and every one of the four rows came back ~1e-1 — including "both symmetric", which
   *should* have tied. The reflex reading is "the structure is wrong"; the truth was a second
   convention (BN world) swamping the first. **Cross the axes rather than sweeping one.** The 2×4
   grid found the single fp-noise cell immediately; the 1×4 sweep would have sent the next session
   hunting a structural bug that does not exist.
5. **The audit passing is not the artifact being audited.** `regen_verified_mlir.sh check` prints
   *"OK — mobilenetv2_fwd.mlir is a byte-identical 1614-line prefix of mobilenetv2_train_step.mlir"*
   while `mobilenetv2_fwd.mlir` and `mobilenetv2_adam_train_step.mlir` sit in different BN worlds.
   The check is true; it is just not about the artifact that trains anything. ▶ When a guard goes
   green, check **what pairing it actually formed**, not just that it passed.
6. **Reference-vs-reference is a measurement nobody thinks to take.** Quantifying the padding
   deviation needed no render at all — just the reference evaluated twice. It is cheaper than a
   tie, it isolates one convention exactly, and it answers "how far is what we built from what was
   published" directly. It also surfaced that batch BN **amplifies** the deviation ~10× (3.5% →
   27.7% of logit range), which no single-world measurement would have shown.

---

## 6. WHAT THIS BUYS

MNv4 is the **"stop adding new block types"** architecture — four block families expressed by one
parameterised primitive. Every other verified net needed its own block form; this one would
demonstrate that the verified layer language can express a *family* from a single constructor. That
is a different claim from "we verified another net", and it is the interesting one.
