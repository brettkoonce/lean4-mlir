# bf16 ties by erasure: the step ties reach the artifacts that trained

Written 2026-10-07, after a read-only survey of the bf16 proof surface. Every verified ImageNet
job the book runs trains a bf16 artifact, and every whole-net statement in the Proofs tier is
stated at the f32 twin and says so ("bf16 is outside this statement"). This plan closes that gap
the cheap way, by restating each tie at the bf16 artifact with the identity rounding, and records
why that is the same claim the f32 artifacts already carry. It does not change any artifact, any
run, or the bf16 design.

## ▶ Start here (next session): MobileNetV4, then ConvNeXt and ViT

**State (2026-10-07, end of the second session).** On `wp8fg`, not pushed: the plan (`053baf64`),
§3.1 the 25 erasure lemmas (`40172103`), §3.2 ResNet-34's typed graphs on the flag (`c27825fc`),
§3.3 ResNet-34's step tie / sync tie / loss gradient on the flag (`906bfa44`), step 3 ResNet-50
end to end plus the yaml 4f paragraph (`8b8ae732`), MobileNetV2 end to end (`7de4ff73`: `GradNodesBAt` gained the depthwise nets' kinds — `ConvStridedXlaWTiedBAt`,
`DepthwiseWTiedBAt`, `DepthwiseStridedWTiedBAt`, `DepthwiseStridedXlaWTiedBAt`, their
`_hasGradAt`s and the `Sync`s for all but the XLA-strided depthwise, which stays local to
`MobileNetV2SyncStepTieB` as its f32 form did), and EfficientNet-B0 end to end (the commit after
`7de4ff73`; no shared file touched — its block builders live in `EfficientNetStagesPC`, which took
the flag in place). Both ResNets, MobileNetV2 and B0 now have every tier — graph (MobileNetV2 also
the `do` graph, B0 also the `Drop` graph the `dropdo` artifacts use), sync graph, step tie, sync
tie, loss gradient — stated at `bf16 := true`, the artifacts the book's ImageNet runs train from,
read over ℝ as the f32 ones are. Nothing else changed: no artifact, no run, no book sentence (§3.4's book part is still open;
the yaml part is done through MobileNetV2).

**What the MobileNetV2 step taught** (the recipe below holds; two additions): the binder-threading
regexes over-match the cotangent CHAIN definitions (`mnv2NoExpCotPc … (p : IVWNoExp ic oc)\n (xin`,
the `section … variable` lines, the `*_scaled` lemmas, the `*_hasGradAt_comp` lemmas, the
real-valued `mnv2HeadBDo`) — list every `(bf16 : Bool)` with its enclosing declaration before
building and revert the ones on anything that is not a graph builder, a tie bundle or a capstone;
and a kind with no f32 predicate (`depthwiseStridedXla`) gets its `At` predicate in `GradNodesBAt`
with `_holds` straight from the `_den` lemma and no `_false`. B0's shared pieces
(`DepthwiseStridedWTiedBAt`, `depthwiseStridedWAt_hasGradAt`, `DepthwiseStridedWSyncAt`) are
already in `GradNodesBAt`, so B0 and MobileNetV4 touch no shared file. B0 confirmed the trap and
the cure: its tie files' chain lemmas share the bundles' binder shapes exactly (`(hp : 0 < p.pε)`,
`(hq : p.EpsPos)`, `(γs βs : Vec oc)`, `(Wfc : Mat oc nC)` …), so the binders went in by LINE
NUMBER with a content assertion, and only the name-anchored sites / nodes / uses by regex; the
post-hoc "every `(bf16 : Bool)` with its enclosing declaration" listing then came back clean on
the first pass. MobileNetV4's files are the biggest (`MobileNetV4SyncStepTieB` 1454 lines); use
the same method, and expect its `MobileNetV4FullB{,Do,Drop}` and the UIB block kinds (`convStrided`
at the stem and fused stage, `depthwiseStrided`) — all four predicates already exist.

**The recipe, per net** (what §3.2 + §3.3 did for ResNet-34, step 3 repeated verbatim for
ResNet-50; `git show c27825fc 906bfa44 8b8ae732` are the templates):

1. **Shared predicates first**, in `Foundation/GradNodesBAt.lean`: for each conv kind the net
   emits, `<Kind>WTiedBAt bf16` (copy of the f32 `GradNodesB` predicate with the node on the
   switch `SHlo.<kind>WeightGradBAt bf16 id …`; `_holds` = `rw [Bf16Fold.den_<kind>WeightGradBAt_id]; exact
   <f32 holds>`; `_false … = <f32 predicate> := rfl`), `<kind>WAt_hasGradAt bf16` (from
   `ParamGradNodes`' `<kind>W_hasGradAt` the same way) and `SyncKit.<Kind>WSyncAt bf16` with
   `_of_scaled` (`simp only [den_allReduceMeanF, Bf16Fold.den_<kind>WeightGradBAt_id] at h ⊢; exact h`).
   The erasure lemmas for every kind already exist in `Bf16Erasure` (`denOp_depthwiseAt_id`,
   `den_depthwiseWeightGradBAt_id`, …; the only-mentions report lists the 19 unconsumed ones —
   they are exactly this step's inputs).
2. **Typed graphs** (`*FullB.lean`, `*SyncB.lean`): every builder gets `(bf16 : Bool)` right
   before its input `e` (the sync builders are applied to the replica index after `e`, so no
   trailing default; the single-device family matches for symmetry); each conv site becomes
   `.<kind>At bf16 id …` (`StableHLO.PrecisionSwitch`; MobileNetV4 spells the depthwise with
   `(c := …)` first — keep its argument order); `_faithful` proofs erase the switch in a FIRST
   `simp only […, Bf16Fold.denOp_<kind>At_id]` and run `denOp` in a SECOND — on a symbolic
   `bf16` a single `simp only [denOp, …]` unfolds `denOp` to a stuck `match` and the proof dies
   (or times out in `acLt`); sync `_shard` proofs add `simp only [Bf16Fold.denOp_<kind>At_id] at hcX`
   right after each `den_batchOp_shard`. The whole-net `_faithful` / `_shard` take the binder.
3. **Ties** (`*StepTieB.lean`, `*SyncStepTieB.lean`, `*ParamGrad.lean`): the flag goes in
   place — block bundles take `(bf16 : Bool)` after the weight record `p` (before the inputs),
   capstones after the weights `w` (before `x` / `X`); the `open Proofs.GradNodeB (…_holds)`
   lines name the `At_holds`; conv nodes in the loss bundles become
   `SHlo.<kind>WeightGradBAt bf16 id …` and their proofs call `GradNodeB.<kind>WAt_hasGradAt bf16`.
   Bias, BatchNorm, SE-dense and head-dense nodes carry no flag (no render switches them).
4. **Text ties** (`Codegen/FwdGraphTextTies.lean`): the f32 rows pass `false` before the leaf, and
   a bf16 row per block kind runs the renderer's emitter at `(bf16 := true)` against the graph at
   `true`. Every renderer here has one `bf16 : Bool := false` flag (not ViT's triple).
5. **Consumers**: `SpecVJP.lean` names `mobilenetv2FwdGraphBFull` (286/288) and
   `efficientnetFwdGraphBFull` (367/369) in statements — pass `false` there; grep every bundle
   name outside its file before building (ResNet-50's borrowed `r34StemSyncTiedB` was missed
   once and found by the build). Cross-net code borrows between these three nets: none
   (MobileNetV4's mentions of `mnv2StemGraphB` / `r50StemGraphB` are prose).
6. **After the build**: `python3 scripts/gates/gen_comparator_tier.py` (the capstones and
   `*FwdGraphSyncFull_shard` are DECLS; it regenerates and verifies in seconds),
   `python3 scripts/gates/check_audit_coverage.py`, a scratch `#print axioms` of the new lemmas
   and capstones (expect `[propext, Classical.choice, Quot.sound]`), the docstring form
   `Module.Name` not `dir/File.lean` (doc-gen4 turns the latter into dead links), stage and STOP
   for the commit word. Build lists: the net's own `Nets/<family>/*.lean` importers of the
   changed files plus `Codegen.FwdGraphTextTies` and `SpecVJP`; lake under `flock lake.lock`.

**Per net — what exists and what is new** (f32 pieces in `GradNodesB` / `ParamGradNodes` /
`SyncKit`; "new" = the `At` twin to write in `GradNodesBAt`):

| net | conv kinds in its ties | shared pieces | notes |
|---|---|---|---|
| MobileNetV2 — DONE 2026-10-07 (`7de4ff73`) | `conv` ×6, `convStridedXla` ×1 (stem), `depthwise` ×2, `depthwiseStridedXla` ×1 | all in `GradNodesBAt` now: `ConvStridedXlaWTiedBAt`, `DepthwiseWTiedBAt`, `DepthwiseStridedXlaWTiedBAt` (+ B0\'s `DepthwiseStridedWTiedBAt`), their `_hasGradAt`s and the `Sync`s except the XLA-strided depthwise\'s, which stays `private` in the sync file as `DepthwiseStridedXlaWSyncAt`; what was found: `depthwiseStridedXla` had NO f32 predicate — `MobileNetV2StepTieB.lean:607` states the node inline, `MobileNetV2ParamGrad.lean:299` likewise, and `DepthwiseStridedXlaWSync` is `private` in `MobileNetV2SyncStepTieB.lean:621` with its `_of_scaled` at 631 and a local `den_allReduceMeanF_depthwiseStridedXlaWeightGradB_shard` at 602 (`depthwiseStridedXlaW_hasGradAt` exists in `ParamGradNodes`) | graphs `MobileNetV2FullB` (sites: conv 7, convStridedXla 1, depthwise 2, depthwiseStridedXla 1), `MobileNetV2SyncB` (12 / 2 / 4 / 2); flag `mobilenetv2FwdGraphBFullDo` too (the shipping artifact `rmsdp64wxdols0eps0001bf16` is a `do` one); `MobileNetV2FullPaperEval` stays f32 (eval is f32); ties `MobileNetV2StepTieB` (`mnv2_net_tiedB`), `MobileNetV2SyncStepTieB` (`mnv2_net_syncTiedB`), `MobileNetV2ParamGrad` (`mnv2_net_lossGrad`, 11 nodes); comparator rows `mnv2_net_lossGrad`, `mobilenetv2FwdGraphSyncFull_shard`, `mnv2_net_syncTiedB`; text ties 7 guards |
| EfficientNet-B0 — DONE 2026-10-07 (the commit after `7de4ff73`) | `conv` ×6, `convStridedXla` ×1 (stem), `depthwise` ×2, `depthwiseStrided` ×1; the SE and head denses are f32 (`DenseWTiedB` ×7, no flag) | nothing new: `DepthwiseStridedWTiedBAt` and its `_hasGradAt` / `Sync` landed with MobileNetV2; the block graph builders are `EfficientNetStagesPC`\'s and took the flag there | graphs `EfficientNetFullB0` (+ `…Drop`, the drop-path forward the `drop` artifacts use; `…Eval` / `…EvalDrop` stay f32), `EfficientNetSyncB`; ties `EfficientNetStepTieG` (`efficientnet_net_tiedG`; binders `(xN vN epsStr cotN dN : String) (N : Nat)`), `EfficientNetSyncStepTieG` (`efficientnet_net_syncTiedG`), `EfficientNetParamGrad` (`enet_net_lossGrad`, 13 nodes, 33 `hasGradAt` calls); comparator rows `enet_net_lossGrad`, `efficientnetFwdGraphSyncFull_shard`, `efficientnet_net_syncTiedG` (`efficientnetInputGradBFull_correct` is the input gradient, untouched); text ties 8 guards |
| MobileNetV4-Conv-M (last: biggest) | `conv` ×11, `convStrided` ×2 (stem, fused stage), `depthwise` ×4, `depthwiseStrided` ×1 | nothing new once the two above landed | graphs `MobileNetV4FullB` (+ `…Do`, `…Drop`; `…Eval` stays f32), `MobileNetV4SyncB`; ties `MobileNetV4StepTieB` (`mnv4_net_tiedB`), `MobileNetV4SyncStepTieB` (1454 lines, `mnv4_net_syncTiedB`), `MobileNetV4ParamGrad` (`mnv4_net_lossGrad`, 19 nodes); comparator rows `mnv4FwdGraphBFull_faithful`, `mnv4FwdGraphSyncFull_shard`, `mnv4_net_syncTiedB`, `mnv4_net_lossGrad`; text ties 7 guards (+ 5 at inference, f32) |

Then ConvNeXt (`convStride4` stem, `convStrided` downsamples, `depthwise`; the hand-written GAP
backward stays a carve-out; its fold file is `ConvNeXtFoldGB`) and ViT (`rowDense`, `patchEmbed`;
the renderer's three flags `bf16` / `bf16Conv` / `bf16ConvW` — the shipping variant keeps the
patch embed and head f32, so the typed graph takes the triple), and §3.4's book sentences
(`content.tex` 6567, 8361, 9199, 10566, 11137, 12751, 13397, 19153 — one chapter per commit).

## 1. The census

| layer | state | where |
|---|---|---|
| verified ImageNet jobs training a bf16 artifact | 7 of 7: `r34-default-bf16-4gpu`, `r50-2018-bf16-4gpu`, `r50-a3-wxclip4x128-bf16-4gpu`, `mnv2-default-4gpu` (`rmsdp64wxdols0eps0001bf16`), `enet-default-4gpu` (`emarmsdp64dropdowxeps0001bf16`), `cnx-default-4gpu` (`adamdpwxclipdropbf16`), `vit-default-emabf16-4gpu` | `content.tex` ~18377, `verified_mlir/MANIFEST.md` |
| whole-net step ties, sync ties, `*_net_lossGrad` | f32 only, all seven nets; each file carries a scope sentence | `ResNet34StepTieB.lean:33`, `ResNet50StepTieB.lean:64`, `ViTStepTieGB.lean:17`, `ConvNeXtStepTieGB.lean:16`, `EfficientNetStepTieG.lean:17`, `ResNet34SyncStepTieB.lean:51`, `ResNet34ParamGrad.lean:16` |
| the nine `*GradBBf16` weight-gradient kinds | folded once, for any `rnd` | `Foundation/Bf16GradNodes.lean` |
| six kinds under sharding | 22 theorems; one in the comparator tier | `Foundation/DataParallel/SyncBf16.lean` |
| single-layer and composed error bounds | done; vacuous in absolute terms, ratio 1.52x (R34) / 1.86x (R50) | `Float/ConvMixedFloatBridge.lean`, `Float/ConvMixedComposeBridge.lean` |
| "adds rounding and nothing else" at `rnd = id` | one op, the per-example CIFAR conv | `StableHLO/Basic.lean:2999` `flatConvFBf16_id` |
| batched bf16 kinds with no theorem outside their `den` clause | 11 of 25: `convStridedXlaBf16`, `depthwiseStridedXlaBf16`, `depthwiseStridedBf16`, `convStride4Bf16`, `denseRowBf16`, `patchEmbedBf16`, `matmulFBBf16`, `denseRowBackBf16`, `depthwiseBackBatchedBf16`, `depthwiseStridedBackBatchedBf16`, `depthwiseStridedXlaBackBatchedBf16` | `StableHLO/Basic.lean` |

The book says what the tier says: "tied per operator rather than as a step; so for this run the
claim is 'one architecture, two independent lowerings, agreeing', not 'proven at ImageNet scale'"
(`content.tex:6567`, and the same sentence at 8361, 9199, 10566, 11137, 12751, 13397). That
wording is the 2026-09-30 rubric pass's decision (WP6 scope sentences): disclose the gap rather
than close it. `SyncBf16`'s header says why nobody closed it: "there is no single-device bf16
chain for either net to tie a twin to".

**The mechanism.** Every bf16 constructor carries a binder `rnd : R -> R`, and every renderer
instantiates it with `zrnd := fun r => r` (`StableHLO/Pretty.lean:4767`, via RenderKit's
`XAt bf16 zrnd` switch). So the AST behind every bf16 artifact denotes the f32 computation; bf16
lives only in the constructor tag, which `pretty` turns into bf16 tensor types and `skel` keeps
distinct. No generator ever supplies a real rounding, and there is no `rnd`-threaded chain
anywhere in `Nets/` or `Architectures/`. The `den` of a bf16 kind is its f32 peer's `den` with
`rnd` at the operand reads and (for all but `rowDenseWeightGradBBf16`) at the store
(`Basic.lean:1651`, `:2262`); at `rnd = id` the two are the same real number, and that equality is
stated for exactly one op.

## 2. The claim to make

The f32 ties are exact-arithmetic readings of the f32 text: `den` is over R, and GPU rounding is a
Trusted-row item (`TRUST.md:15`; the book: "the theorems are over R"). A tie at the bf16 artifact
with `rnd = id` is the same object: the bf16 text, read over R, computes the certified step. What
is trusted is that the hardware rounds where the text says and nowhere else, which is the same
trust the f32 claim takes at every op, at 2^-8 instead of 2^-24 at the conv and dense sites. The
Float tier already bounds that deviation as a ratio.

So the statement becomes, per net: the gradient nodes of `<net>in_...bf16_train_step.mlir`, read
over R, are the certified batched gradient at the chain cotangent (the existing `*_net_tiedB`,
`*_net_syncTiedB`, `*_net_lossGrad` with `bf16 := true`), and each bf16 kind equals its f32 peer at
the identity rounding (`Bf16Erasure`). The rounding sites are then named by the kinds, per
operator, as today. The statement does not say anything about the size of the rounding; it never
did at f32 either.

Not changing: the design. The `rnd` binder, the f32 BatchNorm / LayerNorm / head, `zrnd` in the
generators and gate 2 for the emit shape are right and were validated by measurement;
`archive/bf16_dtype_ir.md` refuted a dtype in the IR.

## 3. Tier A: the work

### 3.1 `Foundation/Bf16Erasure.lean`: 25 lemmas, one per batched bf16 kind

`den (X_bf16 id ...) = den (X ...)` for every kind the ImageNet renders emit, named against its
own peer. Template `flatConvFBf16_id` (`simp only [...]; ring`, the bias sits outside the store).

| group | kinds | peer | proof |
|---|---|---|---|
| forward `BatchableOp` (9) | `convBf16`, `convStridedBf16`, `convStridedXlaBf16`, `depthwiseBf16`, `depthwiseStridedBf16`, `depthwiseStridedXlaBf16`, `convStride4Bf16`, `denseRowBf16`, `patchEmbedBf16` | `conv`, `convStrided`, `convStridedXla`, `depthwise`, `depthwiseStrided`, `depthwiseStridedXla`, `convStride4`, `denseRow`, `patchEmbed` | `denOp` equality at `id`; one generic `den_batchOp_congr` lifts it through `.batchOp` |
| forward `SHlo` (1) | `matmulFBBf16` | `matmulFB` | `den` equality at `id` |
| dgrad (6) | `convBackBatchedBf16`, `convStridedBackBatchedBf16`, `depthwiseBackBatchedBf16`, `depthwiseStridedBackBatchedBf16`, `depthwiseStridedXlaBackBatchedBf16`, `denseRowBackBf16` | the `Back`/`BackBatched` peers | `den` equality at `id`; the VJP backward is applied to `fun j => id (dy j)` |
| wgrad (9) | the `Bf16GradNodes` table | `GradNodeB.*_den` peers | corollary: `Bf16GradNodes.*_den` at `id` rewritten with the f32 `_den` |

Traps, from the files: `convStridedXla` and `convStrided` have identical types and emitted shapes
and differ only in `denOp` (`Basic.lean:1664`), so each erasure must name its own peer; a wrong
pairing fails to prove, which is the point. `rowDenseWeightGradBBf16` has no outer rounding
(`Bf16GradNodes` header); `denseRowBf16` rounds the store and adds the bias after it;
`patchEmbedBf16` rounds the patch sum (`Basic.lean:1560`); read each `den` clause, do not pattern-
copy. New `Proofs.*` names: `lake build LeanMlir` at the root before committing (the
`convWindow` collision, `floatclose_precision_agnostic`).

Print all 25 in `tests/AuditAxioms.lean` next to the nine `*_den`s (line ~2079).

### 3.2 The typed graphs take the flag

The graph builders in `Nets/` construct `.batchOp (.conv ...)` directly (`ResNet34FullB.lean:233`);
the renderers choose by `XAt bf16 zrnd` in RenderKit. Give each builder the trailing
`(bf16 : Bool := false)` the renderers already have and make the choice through the same
RenderKit switch, so builder and renderer pick the constructor from one definition. Scope: the
builders behind the trained artifacts' forwards, `*FwdGraphBFull` and `*FwdGraphSyncFull` and
their block builders (ResNet 19 defs, MobileNet 59, EfficientNet 38, ConvNeXt 6, ViT 7 by name;
the `Eval` / `PaperEval` / `W` variants are out).

Then each `*_faithful` is stated for any `bf16` and proved by `cases bf16 <;> simp only [erasure]`
followed by the existing proof, or by one `graph_bf16_den : den (G true ...) = den (G false ...)`
per builder and a `rw`. `FwdGraphTextTies` (60 guards, scope "f32 ... the configuration the typed
graphs describe") gets its `bf16 := true` row: the renderer's block emitter at bf16 against
`pretty` of the typed block graph at bf16, same start state.

ViT's renderer has three flags (`bf16`, `bf16Conv`, `bf16ConvW`, `ViTRenderB.lean:205/493/496`)
and the shipping variant keeps the patch embed and the head in f32; the typed graph takes the same
triple, not one flag. ConvNeXt's hand-written GAP-backward text (`ConvNeXtRenderB.lean:487`) stays
the declared carve-out it is at f32.

### 3.3 The step ties at the bf16 nodes

The tie predicates (`GradNodeB.ConvWTiedB`, `ConvStridedWTiedB`, `ConvStridedXlaWTiedB`,
`DepthwiseWTiedB`, `DepthwiseStridedWTiedB`, and the ViT / ConvNeXt fold statements) name the
f32 constructor. Give each a `bf16` flag selecting the constructor through the RenderKit switch;
`_holds` at `bf16 := true` is `rw [erasure]; exact <f32 holds>`. The seven capstones
(`r34_net_tiedB`, `r50_net_tiedB`, `mnv2_net_tiedB`, `mnv4_net_tiedB`, `efficientnet_net_tiedG`,
`cnx_net_tiedGB`, `vit_net_tiedGB`), the sync ties and the `*_net_lossGrad`s take the flag and
keep their proofs. For the sync ties the all-reduce wraps the node;
`den_allReduceMeanF_convWeightGradBBf16_shard_id` is already the lemma at `id`.

Kinds per net, from the `Bf16GradNodes` table: R34 / R50 `conv`, `convStrided`; MNv2
`convStridedXla` (stem), `depthwise`, `depthwiseStridedXla`; MNv4 `convStridedXla` (stem),
`convStrided` (fused stage), `depthwise`, `depthwiseStrided`; B0 `convStridedXla` (stem),
`depthwise`, `depthwiseStrided`; ConvNeXt `convStride4` (stem), `convStrided` (downsamples),
`depthwise`; ViT `rowDense`, `patchEmbed`. Drop-path and dropout keep their present scope.

The CIFAR batched artifacts (`cifar8wb_bf16*`, `cifar8wb_bn_bf16*`) use the same kinds; the
`Cifar8ParamGrad` / `Cifar8BnParamGrad` statements take the flag for free once 3.1 exists.

### 3.4 Shared files, after the Lean lands

* `formalization.yaml` 4f: the last sentence ("The bf16 twins are NOT the same node ... carry
  their own lemmas") gains "and equal their f32 peers at the identity rounding (`Bf16Erasure`);
  the capstones are stated for either precision". The `main_results` rows for the capstones: one-
  line comments, no new rows unless a new declaration is the headline. Section 6 keeps "that bf16
  doesn't change results", which is the runtime claim.
* `scripts/gates/gen_comparator_tier.py`: the capstones are already `DECLS`; if a statement gains
  a binder, check the comparator config still elaborates (comparator tier and yaml move together).
* The seven tie files, the ParamGrad files and `SyncBf16`'s "No whole-net statement" paragraph:
  the scope sentences say the statement is at either precision and name `Bf16Erasure`.
* The book: `content.tex` 6567, 8361, 9199, 10566, 11137, 12751, 13397 and 19153. The sentence
  becomes: the tie reaches the bf16 render at the step level, reading the text over R; the
  rounding sites are the `*Bf16` kinds, each equal to its f32 peer at the identity rounding; what
  is trusted is the rounding at those sites, as at f32. The "two independent lowerings, agreeing"
  sentence at 6567 stays as the IREE / XLA fact it reports. One chapter per commit.
* Blueprint: regenerate `blueprint/lean_decls` then `blueprint_uses.py --fix --check` if any
  theorem's dependencies moved.
* No artifact changes: nothing is re-rendered, every byte stays. `lake build` the gates that read
  the graph builders before claiming green (stale `lean_exe` gates).

### 3.5 Order and cost

1. `Bf16Erasure.lean` and its AuditAxioms lines. Builds alone; half a session. Done 2026-10-07:
   `Foundation/Bf16Erasure.lean`, the 25 `*_id` lemmas, `den_batchOp_congr` and the two missing
   splits, every one on the three core axioms; 20 of the 25 are `rfl`, the seven conv / depthwise
   forwards go through the bias splits, `denseRowBf16` through `rowBiasFlat`. Registered as a
   `Certs` root (the audit-coverage gate requires it).
2. ResNet-34 end to end (3.2 and 3.3 for `resnet34in_momdp64bf16`: graph flag, faithful, text
   ties, step tie, sync tie, ParamGrad). The template; one session. 3.2 done 2026-10-07: the
   24 `XAt` switches moved out of `RenderKit` into `StableHLO/PrecisionSwitch.lean` (a leaf on
   `Basic`, so `Nets/` can import it without the printer); `Bf16Erasure` gained one lemma per
   switch at `id` for either `bf16` (`denOp_convAt_id`, `den_convBackBatchedAt_id`, …; `cases
   bf16`, `rfl` / the kind's `_id`); `r34{Id,Down,Stem}GraphB`, `resnet34FwdGraphBFull` and the
   four sync twins take `(bf16 : Bool)` right before the input `e` (the sync builders are applied
   to the replica index after `e`, so a trailing default was not available; the B family matches
   for symmetry), and every `_faithful` / `_shard` is stated for either value — the proofs are the
   f32 ones with the switch erased first: `simp only […, Bf16Fold.denOp_convAt_id]` BEFORE the
   `denOp` equations (on a symbolic `bf16` those are stuck on the `if`, and simp otherwise unfolds
   `denOp` to a stuck `match`), and in the sync shard proofs `simp only [Bf16Fold.denOp_convAt_id]
   at hc` right after each `den_batchOp_shard`. `FwdGraphTextTies` checks the stem, identity and
   downsample rows at both values. `ResNet50FullB` / `ResNet50SyncB` (borrowing the R34 stem) and
   `SpecVJP` pass `false` until their own steps. The comparator tier's
   `resnet34FwdGraphSyncFull_shard` statement gains the binder and is regenerated.
   3.3 done 2026-10-07: the flagged tie predicates live in one new file above `Bf16Erasure` and
   `SyncKit`, `Foundation/GradNodesBAt.lean` (`ConvWTiedBAt bf16` / `ConvStridedWTiedBAt bf16`
   with `_holds`, `convWAt_hasGradAt` / `convStridedWAt_hasGradAt`, `ConvWSyncAt bf16` /
   `ConvStridedWSyncAt bf16` with `_of_scaled`; each `= ` its f32 original at `false` by `rfl`),
   so neither `GradNodesB` nor `SyncKit` nor `ParamGradNodes` changes and their cones do not
   rebuild; the sync `_of_scaled` is the f32 lemma under `simp only [den_allReduceMeanF,
   den_convWeightGradBAt_id] at h ⊢`. In the R34 files the flag is in place (no outside users of
   the identity / downsample bundles): `r34{Id,Down,Stem}TiedB`, `r34_net_tiedB`, the sync
   bundles and `r34_net_syncTiedB` (+ `_smoothedCE`), `r34{Id,Down}LossTiedB`,
   `r34StemLossTiedAtB` / `r34StemLossTiedB` and the pool-selector lemmas through them,
   `R34NetLossTiedB`, `r34_net_lossGrad` (+ `_smoothedCE`, `_stemSelect`),
   `r34_net_tied_lossGrad` — the flag sits after the weights, before the input; ResNet-50's
   borrowed stem bundles pass `false` at seven sites until step 3. Bias, BatchNorm and dense
   nodes carry no flag. Comparator regenerated (`r34_net_lossGrad`, `r34_net_syncTiedB`,
   `r50_net_tiedB`'s borrowed stem). The three files' "bf16 is outside this statement"
   paragraphs now state the flag; the book's sentences are §3.4.
3. ResNet-50 (same kinds), then MobileNetV2 / V4 / B0 (the depthwise kinds), ConvNeXt, ViT. Script
   the binder threading (the cone-parametrisation method); compile and fix. ResNet-50 done
   2026-10-07, the R34 recipe verbatim: `r50{Id,Proj,Down}GraphB`, `resnet50FwdGraphBFull`, the
   sync twins (the resolution examples at `q = 7` / `q = 5` generic in the flag), `r50{Id,Proj,
   Down}TiedB`, `r50_net_tiedB`, the sync bundles, `r50_net_syncTiedB` (+ `_smoothedCE`, `_bce`),
   `r50{Id,Proj,Down}LossTiedB`, `R50NetLossTiedB`, `r50_net_lossGrad` (+ `_smoothedCE`, `_bce`,
   `_stemSelect`, `r50_net_tied_lossGrad`) take `bf16`; the seven borrowed R34 stem bundles pass it
   through instead of `false`; `FwdGraphTextTies` checks the R50 stem and three block rows at both
   values. No new shared lemma: R50's kinds are `conv` and `convStrided`, already in
   `GradNodesBAt`. Count trap: the strided bottleneck's `W₁` is a plain conv at the input grid,
   so the plain-conv sites are 8 per file, not 9. The yaml's 4f paragraph and the R34/R50 row
   comments now say "at either precision" (the §3.4 yaml part). MobileNetV2 done 2026-10-07, the
   recipe verbatim: `GradNodesBAt` gained the depthwise nets' four kinds (the XLA-`SAME` stem, the
   depthwise, B0's symmetric strided depthwise and MobileNetV2's XLA-strided one — the last with
   no f32 predicate to be `rfl` to, so its `_holds` is `depthwiseStridedXlaWGradB_den` under the
   erasure, and its collective stays `private` in `MobileNetV2SyncStepTieB`); `mnv2{Stem,NoExp,
   ExpOnly,Resid,Strided,Head}GraphB` (+ `mnv2HeadGraphBDo`), `mobilenetv2FwdGraphBFull{,Do}` and
   the six sync twins take `(bf16 : Bool)` before `e`; `mnv2{Stem,NoExp,Stride1,Stride2,Head}TiedB`,
   `mnv2_net_tiedB`, the sync bundles, `mnv2_net_syncTiedB` (+ `_smoothedCE`), the five
   `*LossTiedB`, `MNV2NetLossTiedB`, `mnv2_net_lossGrad` (+ `_smoothedCE`, `mnv2_net_tied_lossGrad`)
   take it after the weights (the head's after `bd`, since its 1×1 conv switches; the dense does
   not); `SpecVJP` passes `false`; `FwdGraphTextTies` checks the stem, the four block kinds, the
   head and the dropout head at both values (7 → 14 MobileNetV2 guards). The yaml rows and 4f say
   MobileNetV2 too. Trap met: the binder regexes hit the cotangent chain definitions, the
   `section … variable` lines, the `*_scaled` and `*_hasGradAt_comp` lemmas and the real-valued
   `mnv2HeadBDo` — reverted by listing every `(bf16 : Bool)` with its enclosing declaration.
   EfficientNet-B0 done 2026-10-07, no shared lemma: `stemGraphB` / `mb{NoExp,Strided,Resid}GraphB`
   / `headGraphB` (`EfficientNetStagesPC`) and `mbExpGraphB` take `(bf16 : Bool)` before `e`, the
   `mb*GraphW` wrappers, `efficientnetFwdGraphBFull`, `mbResidDropGraphB{,W}`, `headGraphBDo`,
   `efficientnetFwdGraphBFullDrop` (the `dropdo` artifacts' forward; the `Eval*` twins stay f32)
   and the seven sync builders pass it through; `enet{Exp,Strided,NoExp}TiedG` (+ `*At`),
   `enet{Stem,Head}TiedG`, `efficientnet_net_tiedG`, the six sync bundles,
   `efficientnet_net_syncTiedG` (+ `_smoothedCE`), the five `*LossTiedG`, `EnetNetLossTiedG`,
   `enet_net_lossGrad` (+ `_smoothedCE`, `enet_net_tied_lossGrad`) take it after the ε-positivity
   hypotheses (`(hp : 0 < p.pε) (bf16 : Bool)`, `(hεw : w.EpsPos) (bf16 : Bool)`), the head's
   after `bfc` (its 1×1 conv switches; the squeeze-excite and classifier denses do not);
   `SpecVJP` passes `false`; `FwdGraphTextTies` checks the stem, the four block kinds, the
   drop-site block, the head and the dropout head at both values (8 → 16 B0 guards). The yaml
   rows and 4f say B0 too. Binders by line number this time (see the brief at the top).
4. The shared files (3.4), one commit each.

Each commit staged and shown before it is made.

## 4. Tier B, optional: the chain for any `rnd` (ResNet-34 pilot)

Thread `rnd` through the real-valued chain (`r34Pre_k`, `r34IdCotIn`, `r34IdCotC1`, ...) so the
capstone is stated for any `rnd`: the bf16 artifact is the certified chain with `rnd` at exactly
the conv sites, and `rnd := rndP 7` feeds `ConvMixedComposeBridge`. That makes the site census a
theorem rather than gate 2 plus the op-kind histogram. Cost: new chain definitions per net (17
sites on R34, 35 on R50, 38 on MNv4), the forward-seal and backward-chain twins, and the ViT /
ConvNeXt hand-written pieces in the way. Value over tier A: the census as a theorem; the bound it
feeds stays vacuous in absolute terms. Decide after tier A's book sentence is written; pilot on
ResNet-34 only.

## 5. Declined

* A dtype in the IR, or a real rounding in the generators: `zrnd` stays. The emit is right (gate
  2) and `archive/bf16_dtype_ir.md` §0 refuted the dtype route by measurement.
* A non-vacuous absolute bf16 bound: needs probabilistic rounding or a BatchNorm-renormalisation
  argument. Research, not a tie.
* bf16 BatchNorm / LayerNorm / head twins: no net wants them (`bf16_batchnorm.md`, archived).
* The per-example CIFAR `cifar8_bf16*` ("V") artifacts: forward-only bf16 on a family whose
  backward has no bf16 twins; `flatConvFBf16_id` is their erasure already.
