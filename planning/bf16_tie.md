# bf16 ties by erasure: the step ties reach the artifacts that trained

Written 2026-10-07, after a read-only survey of the bf16 proof surface. Every verified ImageNet
job the book runs trains a bf16 artifact, and every whole-net statement in the Proofs tier is
stated at the f32 twin and says so ("bf16 is outside this statement"). This plan closes that gap
the cheap way, by restating each tie at the bf16 artifact with the identity rounding, and records
why that is the same claim the f32 artifacts already carry. It does not change any artifact, any
run, or the bf16 design.

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
   ties, step tie, sync tie, ParamGrad). The template; one session.
3. ResNet-50 (same kinds), then MobileNetV2 / V4 / B0 (the depthwise kinds), ConvNeXt, ViT. Script
   the binder threading (the cone-parametrisation method); compile and fix.
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
