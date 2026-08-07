# mnv4_verified.md — MobileNetV4 as a verified net

**Scoped 2026-08-07 by reading the source, not by estimate.** Every file/line reference below was
checked. Companions: `planning/mnv4_imagenet.md` (the **JAX reference** side — recipe, timm decode,
the Conv-S/Conv-M distinction), `planning/rsb_a3_r50_verified.md` (the closest precedent: a new
block form brought onto the verified path).

Status: **NOT STARTED.** Nothing about verified MNv4 exists.

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

### Phase 2 — the UIB block VJP (proof side)
Four shapes from one primitive. The depthwise and pointwise VJPs exist (mnv2/enet); what is new is
the composition with a leading DW.
*Gate*: 3-axiom clean, and the existing `vjp-oracle-uib` (1.13e-05) as the numerical control.

### Phase 3 — the renders + the spec + the app
`mnv4_{fwd,fwd_eval,adam_train_step}` at `nClasses := 10`, B = 32, plus `mobilenetv4Verified` and an
`Resnet50AdamCommon`-shaped driver pair. ⭐ **The R50/Imagenette work of 2026-08-07 is the template
and it is three `#eval` lines + a 50-line Common** — that part is genuinely mechanical.
*Gate*: `check_render_coverage.py`; and note that `lake build <exe>` does **not** run the renders
(learned the hard way on R50 — the exe builds green against artifacts that do not exist).

### Phase 4 — the run
80 epochs, bs32, AdamW — the Imagenette tier's schedule.
*Gate*: **84.58%**, the baseline path's number for this block table.

---

## 5. WHAT WOULD MAKE THIS NOT WORTH DOING

* Phase 0 finds a missing descriptor that is not batch-invariant — then it is a `BatchableOp`
  question first and an MNv4 question second.
* The forward tie of §3 cannot be made to pass. Without it there is no way to tell a correct UIB
  from a plausible one, and a 84.58% that came from the wrong block order is worse than no number.

---

## 6. WHAT THIS BUYS

MNv4 is the **"stop adding new block types"** architecture — four block families expressed by one
parameterised primitive. Every other verified net needed its own block form; this one would
demonstrate that the verified layer language can express a *family* from a single constructor. That
is a different claim from "we verified another net", and it is the interesting one.
