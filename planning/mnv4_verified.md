# mnv4_verified.md — MobileNetV4 as a verified net

**Scoped 2026-08-07 by reading the source, not by estimate.** Every file/line reference below was
checked. Companions: `planning/mnv4_imagenet.md` (the **JAX reference** side — recipe, timm decode,
the Conv-S/Conv-M distinction), `planning/rsb_a3_r50_verified.md` (the closest precedent: a new
block form brought onto the verified path).

Status: **PHASES 0–3 (forward) DONE 2026-08-08.** `.uib` and `.fusedMbConvNB` are in `VLayer`, the
whole Conv-S forward chain renders, `@mnv4_fwd` compiles, and the forward tie against the JAX
reference RAN. Block order is **pinned at 1.8e-6**. One blocker left, decided below.

---

## ▶ START HERE — the next session (2026-08-08)

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

▶ **NOT YET MEASURED.** The MNv2 row above is inference from reading `sep_conv`/`depthwise_conv`,
not a run. `verified_mlir/mobilenetv2_fwd.mlir` exists and `cd jax && lake exe mobilenet-v2`
generates the reference, so `scripts/mnv4_forward_tie.py` needs only its param-grouping adapted
(MNv2's triples are not UIB's). **Do this before or alongside the descriptor work** — it decides
whether the fix has one consumer or three.

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

### Phase 2 — the UIB block VJP (proof side) — NOT STARTED
Four shapes from one primitive. The depthwise and pointwise VJPs exist (mnv2/enet); what is new is
the composition with a leading DW.
*Gate*: 3-axiom clean, and the existing `vjp-oracle-uib` (1.13e-05) as the numerical control.
▶ Do this AFTER the asymmetric-pad descriptor, so the VJP is written once against the final forward.

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
⚠ Do not start this before the forward tie is clean. A number off a render that disagrees with the
reference at the stem is a number for a different net, and it would look completely normal.

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

---

## 6. WHAT THIS BUYS

MNv4 is the **"stop adding new block types"** architecture — four block families expressed by one
parameterised primitive. Every other verified net needed its own block form; this one would
demonstrate that the verified layer language can express a *family* from a single constructor. That
is a different claim from "we verified another net", and it is the interesting one.
