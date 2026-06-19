# Verified-faithful sweep: tying the emitted `.mlir` to the proofs

_Status note, 2026-06-17._ Audit + PoC. Question driving this doc: for each
`Main*Verified*.lean` trainer, is the MLIR it actually compiles **proof-tied**
to the `Proofs/` math, or only validated numerically? And what would close the
gap, net by net?

> **▶ IN PROGRESS (2026-06-19): vit (Vision Transformer) — the LAST Tier-3 net. GROUND-TRUTHED; step 1 (core ops) started.**
> Tier-3 order enet → convnext → vit; enet AND convnext §1a TIE both CLOSED. vit is the only §1a-tie target left.
>
> **GROUND-TRUTH VERDICT (2026-06-19 — the feared LN-granularity gap is NOT a blocker):** the committed
> `vit_train_step.mlir` (ViT-Tiny, depth-12, D=192, 3 heads, MLP 768, 200 params, BS=32) confirms **per-channel
> `tensor<192xf32>` LN** (`%g1_*/%b1_*/%g2_*/%b2_*/%gF/%bF`). BUT `LeanMlir/Proofs/ViTVecLN.lean` is a full
> **"scaling pass" that already models exactly this**: `layerNormVec` (per-token normalize then per-channel affine
> `γ⊙x̂+β`, *"the committed ViTRender LN form"*) + `layerNormVec_has_vjp` + the vector-LN transformer-block VJP +
> a 2-block whole-net VJP (`vitForward2V_has_vjp`) + a **render graph + faithfulness** (`vitFwdGraphV`/`_faithful`)
> + per-channel LN param certs (`vit_render_vecln{gamma,beta}_certified`) AND their chain-pinned forms
> (`vit_render_vecln{1,2,F}{gamma,beta}_chain_certified`) + the chain cotangents (`vitCot{H,Att,Xin,B2out}V`). The
> doc's "scalar-LN whole-net backward" worry refers to the OLD `vitNetBackGraph_faithful` parallel universe — which
> the §1a tie does NOT use (the tie builds its backward from per-op chain cots, like convnext used ConvNeXtChainClose).
> **So the per-channel `[192]` LN that ships IS modeled end-to-end.**
>
> **vit = the convnext situation (full §1 stack to build), at transformer scale.** No `ViTRender`/`ViTFaithfulPoC`/
> `ViTTiePoC` in `Proofs/`; the committed `.mlir` is rendered by `LeanMlir/ViTRender.lean` — a hand-written
> **String-fragment emitter** (*"NOT a single `den(trainStep)` theorem — faithful PER-OP, validated by gradchecks
> and training"*), not `pretty(provenGraph)`. EXISTS (rich): per-channel LN fwd/VJP/graph/param-certs/chain-certs
> (`ViTVecLN`); rowwise-dense certs `vit_render_rowdense{W,b}_certified` (attn Wq/Wk/Wv/Wo, MLP Wfc1/Wfc2, classifier
> Wc); patch-conv **weight**+bias certs `vit_render_patch{W,b}_certified` (vit HAS the patch-weight VJP convnext
> lacked → likely NO even-kernel gap); cls/pos certs `vit_render_{cls,pos}_certified`; forward graph `vitFwdGraphV`
> (2-block → needs depth-12 instantiation via the generic `vitBlockGraphV`). TO BUILD: §1 render (Proofs
> `vitTrainStepFaithfulV`) + §1 fold (`ViTFaithfulPoC`) + §1a tie (`ViTTiePoC`, **two residual fan-ins per block** —
> attn-sublayer + mlp-sublayer) + the new core SHlo param-SGD ops.
>
> **CORE OPS — 5 new ops built (corrected from "2"→"4"→"5"; a 6th `posEmbedSgd` still needed — see the BACKWARD
> banner below), all iree-validated.** ⚠️ **Gotcha caught
> before building a broken backward: enet's batched `denseWeightSgdB`/`denseBiasSgdB` match vit's `rowDense_*_grad`
> only at the `den` level, NOT the EMIT** — their `.batched` emit is 2D (`dot_general [B,a]×[B,c]`, batch-only
> contraction), but `denseRowF`/`denseRowBack` are 3D token-matrix `[B,197,a]`, so the rowdense weight grad must
> contract batch×tokens. So I added 2 MORE ops (cheap via the generic `.batched` path — ctor+den (same as
> denseWeightSgdB, Σ over N)+skel+1 emit case each): **`rowDenseWeightSgd`** (3D, contract `[0,1]x[0,1]`→`[a,c]`) +
> **`rowDenseBiasSgd`** (3D, `reduce[0,1]`→`[c]`; covers per-block biases + vector-LN β + patch-b). STILL reuse the
> 2D `denseBiasSgdB` for **pos** (c=197·192, keeps tokens, reduces batch) + **cls** (c=192 on the row-0-sliced
> `[B,192]` cotangent — a backward-chain slice). The FOUR new ops (vecln γ + patch W committed; the 2 rowdense
> built+suite-validated, `lake build Proofs` green 2274):
> 1. **`veclnGammaSgd`** (committed `e662aef`) — vector-LN γ: recompute x̂ = LN(1,0) per token over D, weight by dy,
>    reduce over the N=BS·197 row axis → `Vec D`; den = `vit_render_veclngamma_certified` verbatim; emit 3D `[B,N,D]`.
> 2. **`patchEmbedWeightSgd`** — the 16×16/s16 patchify conv WEIGHT grad: slice patch tokens [1..N] (drop CLS),
>    reshape→`[B,D,ph,pw]`, dilate interior P-1, valid conv with the saved image → `dW : Kernel4 D ic P P`; den via
>    the local re-spelling `patchEmbedWeightGradFlat` (= `vit_render_patchW_certified`). **vit HAS the patch-weight
>    cert (unlike convnext's gapped 4×4/s4 stem), so it is tied — vit will be the FIRST net with ZERO param gaps
>    (200/200).** Both: `lake build Proofs` green (2274, roundtrip + den-faithfulness hold), iree-compile on gfx1100
>    (vecln 17440 B, patch-W 16552 B), harness `tests/TestVeclnGammaSgd.lean`.
>
> **▶ §1 RENDER — FORWARD half DONE (2026-06-19, `LeanMlir/Proofs/ViTRender.lean`).** The full depth-12 ViT-Tiny
> forward rendered NODE-BY-NODE (the computable ConvNeXt pattern — each `pretty` emits one op with `.operand
> <prevSSA> <zero>`; a composed-graph term is noncomputable, can't `#eval`): patch embed → 12× `vBlockFwd`
> (LN1→Q/K/V dense→per-head SDPA slice/matmul/scale/softmax/matmul/pad summed over 3 heads→out dense→+res→LN2→
> fc1→GELU→fc2→+res, mirroring `vitBlockGraphMHV`) → final vector-LN → CLS-slice → dense head. Renders a 2477-line
> module that **iree-compiles clean on rocm/gfx1100 (136 KB vmfb)**; `den = vitForward` via `vitFwdGraphMHV_faithful`
> at depth-12. Module preamble defines `%one`/`%zero`/`%sc` (the lnRow γ=1/β=0 + reduce-init `tensor<f32>` consts).
> **✅ §1 RENDER — BACKWARD: DONE (2026-06-19). The whole depth-12 ViT-Tiny train step renders as
> `pretty(provenGraph)` and iree-compiles clean on rocm/gfx1100 (366 KB vmfb, 7578 lines, 200 return tensors,
> ZERO render-TODO stubs).** The 6th core op **`posEmbedSgd` landed** (the pos grad reduces ONLY the batch axis,
> KEEPING `[197,192]`, so it emits the 2D `tensor<197x192>` the flat `denseBiasSgd` couldn't; cheap `.batched`
> path like patchEmbedBiasSgd, den = `vit_render_pos_certified`'s identity-Jacobian `pos − lr·dy`; standalone
> iree-validated 10776 B). The full node-by-node backward is in `LeanMlir/Proofs/ViTRender.lean`: `vlnBack`
> (vector-LN back = β-SGD `rowDenseBiasSgd` + γ-SGD `veclnGammaSgd` + `rowScaleF(γ)` then `lnRowBack(γ=1)` for the
> input cot) and `vBlockBack` (reverse of `vBlockFwd`: res₂ fan-out → fc2-back/`geluBack`/fc1-back → LN2-back →
> res₁ fan-in → out-dense-back → **per-head SDPA backward** [`headSliceF`/`matmulF`+`transposeF` backs/`softmaxRowBack`/
> `scaleF`/`headPadF`, summed over 3 heads] → Q/K/V-dense-back summed → LN1-back → res₁ fan-in), then the whole-net
> `vitTrainStepRenderV`: loss cot (`softmax−onehot`, mean folded into lr=0.1/32=0.003125, r34 convention) → head
> back (`dotOut`+`weightSgd`/`biasSgd`) → `clsPadF` → final-LN back → 12× `vBlockBack` → patch-embed back
> (`patchEmbedWeightSgd` w/ `%ximg = reshape %x` preamble + `patchEmbedBiasSgd` + `clsSliceF`→`denseBiasSgdB` for cls
> + `posEmbedSgd` for pos). The SDPA backward was verified against the gradcheck-validated hand-written `mhsaBack`
> (matmul/transpose backs + softmax-back all match). **GOTCHA fixed:** `veclnGammaSgd`'s emit used `{xN}`/`{r}` as 3D
> `[B,N,D]` (it required a 3D arg, validated that way in its harness) but the whole-net thread is FLAT `[B,N*D]` like
> every other op (`lnRowBack`/`rowDenseWeightSgd` reshape internally) — added the two internal reshapes + updated the
> harness to flat args (re-iree-validated 17440 B). `lake build Proofs` + roundtrip theorem green. (Committed
> `vit_train_step.mlir` NOT yet replaced: the new render uses **1D cls `tensor<192>`** vs the committed 2D
> `tensor<1x192>` — a trivial FFI-layout reconcile that's the trainer-swap follow-up, NOT a backward-render blocker.)
> **REMAINING for vit: regen the committed `.mlir` + cls-dim/FFI reconcile (trainer swap), then §1 fold
> `ViTFaithfulPoC` → §1a tie `ViTTiePoC` (two residual fan-ins per block).**
>
> **✅ DONE (2026-06-19): convnext (ConvNeXt-T) §1 — the fold + the full [3,3,9,3] render(provenGraph) train step.**
> Commits `1bcbf70` (per-channel layer-scale γ cert — the one new proof) → `09b8195` (3 new core SHlo ops
> `layerScaleChGammaSgd`/`lnGammaSgd`/`lnBetaSgd` across all 9 sites + den-fold, 3-axiom clean) → `0c0a4e7`
> (`ConvNeXtRender.lean` `convNextTrainStepFaithfulV`: the full train step = pretty(provenGraph), regenerated
> `verified_mlir/convnext_train_step.mlir` BYTE-IDENTICAL sig → drop-in, iree-validated 647884 B, `lake build
> Proofs` green 2273). **178/182 params are render(provenGraph) with den=certified**; 4 documented hand-written
> weight-grad gaps (stem 4×4/s4 + 3 downsample 2×2/s2 — the even-kernel/stride-4 weight grads have no VJP-cert
> op; their bias grads DO use SHlo ops). convnext was the **mnv2 situation** (no §1 render/fold existed; committed
> .mlir was hand-written), but the hard math (whole-net VJP, render-math param certs, chain cots in
> `ConvNeXtChainClose`) already existed. (The "178/182" count above was a miscount — the committed render emits
> **180** return tensors: stem 2 + 18 blocks × 9 + 3 downsamples × 4 + head 4; **176** are den-certified, 4 are the
> even-kernel weight-grad gaps.) Full detail in [[convnext-tie-scope]].
>
> **✅ DONE (2026-06-19): convnext (ConvNeXt-T) §1a TIE CLOSED — the whole [3,3,9,3] train step tied through the real forward.**
> `ConvNeXtTiePoC.lean` (`Proofs.CnxTiePoC`): `cnx_net_tied_certified` threads all **176** SHlo-op params (every param
> EXCEPT the 4 even-kernel weight-grad gaps) through the REAL `convNextTrainStepFaithfulV` forward
> (`cnxStemFwdO`/`cnxBlockFwdO`/`cnxDownFwdO`, `@[irreducible]`) + the loss-driven backward cotangent chain, with the
> **identity-skip fan-in `+ dyOut`** at each of the 18 block merges + the LN-back at each of the 3 downsamples. Per-type
> tie lemmas (`cnx_block_tied` 9-param, `cnx_down_tied` 3-param LN+strided-bias, `cnx_stem_bias_tied`, `cnx_head_tied`
> GAP→LN→dense), each a pure delegation to the §1-fold generics (`Mnv2PoC.depthwise*`/`CifarPoC.conv*`/`CnxPoC.{lnGamma,
> lnBeta,layerScaleChGamma}Sgd_den`/`ResNet34PoC.convStridedB_den`/`Cifar8PoC.dense*`) at the `ConvNeXtChainClose`
> cotangents (`cnxCotP`/`cnxCotE`/`cnxCotN`/`cnxCotD`). **ZERO new core ops, ZERO new bridges** — pure thread + fan-in.
> New content vs mnv2: **GELU mask** (smooth, no kink), **scalar LayerNorm** γ/β (`Vec 1`), **per-channel layer-scale**
> γ (`Vec c`, `chanIdx` broadcast). Wired to lakefile `Proofs` root + `tests/AuditAxioms.lean`; `lake build Proofs`
> green (2274); all 6 capstones 3-axiom clean `[propext, Classical.choice, Quot.sound]`. Den-level tie (no
> renderer/.mlir change, as r34/mnv2). Residual: block backward rendered hand-written (cotangent SSA↔chain-cot per-op
> trust); per-op `pretty` lexing; LN `0<ε` smoothness; ℝ→Float32. **This closes the 11th net; vit (the last Tier-3) is
> the only §1a-tie target left.**
>
> **✅ DONE (2026-06-18): enet (EfficientNet-B0) §1a TIE CLOSED — the whole 262-param train step tied through the real forward.**
> The §1a-tie sweep has now closed **10 nets** (linear, mlp, cnn, cifar, cifar-bn, cifar8, cifar8-bn, r34, mnv2, **enet**).
> **enet §1a TIE landed** (`EfficientNetTiePoC.lean`, commit `9533ec4` = 5 per-block ties; whole-net thread next commit):
> the loss-cotangent den (`efficientnetLossCot_den`) + five per-block-type tie lemmas (`enet_{exp,strided,noexp,stem,head}_tied`,
> covering all 262 params: 9 residual + 2 no-skip via the generic expand tie, 4 strided, b1 no-expand, stem, head) +
> the whole-net thread `efficientnet_net_tied` composing them through the REAL `efficientnetForwardB_full` (block inputs =
> its forward prefixes, the per-block dyOuts threaded top-down by the proven block VJPs `mb{Exp,Resid,Strided,NoExp}W_has_vjp`
> + `headFwdB_has_vjp`; `@[irreducible]` `*TiedAt` wrappers for the heartbeat). The genuinely-new content vs mnv2 — **swish**
> masks, the **SE gate fan-in** (`gateCotB`=`den seReduceB` → `sigBackB` → `rowDenseBackFlat` → `swBackB` → SE dense ops),
> **true batch-norm** backward (`bnBackB`=`bnBatchLA` VJP), strided depthwise back, all **batched** at `N·(c·h·w)` with
> `reassocB` bridging `(oc·h·w)`↔`(oc·(h·w))` — all PROVEN. Each per-block tie is a pure delegation to the §1-fold
> `EnetPoC.*` generics at the chain cotangents; `lake build Proofs` green (2270), all 7 capstones 3-axiom clean + wired to
> AuditAxioms. Residual: block backward rendered hand-written (cotangent SSA↔chain-cot per-op trust); per-op `pretty` lexing;
> swish/true-BN `0<ε` smoothness; ℝ→Float32 — the boundary every fold carries. (Optional refinement: the dense-head total-loss
> fold `Wfc → ∂CE/∂Wfc` — today the head dense ties at the loss cotangent `g` directly.) **Next: convnext, then vit.**
>
> _Earlier-this-session history (Item B emit, renderer, §1 fold) preserved below._
>
> **▶ (earlier 2026-06-18): enet — batched emit ("Item B") + param-SGD DONE; renderer + §1 fold DONE.**
> enet was under way (Tier-3: **enet → convnext → vit**). **DIAGNOSIS CORRECTED:** the blocker was NOT
> "no whole-net backward graph" — the backward MATH already exists (`EfficientNetBackB0`/`ChainClose`:
> `efficientnetFwdGraphB_full_faithful` whole-net forward, `efficientnetForwardB_full_has_vjp` whole-net
> math VJP, den-faithful per-block backward bricks incl. batched bn/conv/depthwise/SE). The **real blocker
> was the deferred batched MLIR emit ("Item B")**: every batched op skel'd to a lossy `.batched (tag,info)`
> whose `emitTok` was a `// render TODO` stub, so neither `efficientnet_fwd.mlir` nor `_train_step.mlir`
> was `render(provenGraph)`; plus there were no batched param-SGD ops. **DONE this session (5 commits
> `af3e037`…`e7b6855`, every op iree-validated on rocm/gfx1100):** enriched `.batched` to carry SSA names
> (+ round-trip repair, theorem holds); filled `emitTok` for ALL forward + backward batched tags (conv/
> depthwise/bnBatch/swish/seBlock + bnBatchBack/conv·depthwise BackBatched/seBackBatched, self-contained
> recompute); added EVERY batched param-SGD op (stem/conv/depthwise weights incl. 5×5, BN γ/β, dense W/b,
> all biases via `bnBetaSgdB` = channel-sum).
> **SCOPE CORRECTION (2026-06-18, found while starting the renderer — the doc UNDER-scoped item 5):** Item B's
> SE work is the **fused** `batchOp seBlock` (fwd) + **fused** `seBackBatched` (bwd), which emits ONLY the SE
> input-cotangent `dx`. But the committed trainer **un-fuses SE** and **trains all 4 SE dense params**
> (`zW1/zb1/zW2/zb2` — 64 of the 262), and `EfficientNetClose` certifies them via the **dense** weight bridge
> (needs the SE internals `s=GAP(x)`, `z=swish(dense W₁ s)` SAVED + internal cotangents). So the fused ops do
> NOT suffice — the renderer needs an **un-fused batched SE** path, which Item B did not build. Minimal fix =
> **ONE new core op `seReduceB`** ✅ DONE (2026-06-18, NOT yet committed): the bwd gate-cotangent
> `dgate[n,c]=Σ_{h,w}(x⊙dy)` (= the batched `broadcastFlat_has_vjp.backward` — the FIRST step of the SE gate
> backward). `seScaleB` turned out **unnecessary**: keep the FUSED `batchOp seBlock` for the forward `out`
> (already iree-valid in `efficientnet_fwd.mlir`) and ADDITIONALLY emit the un-fused gate subnet `s=batchOp gap →
> e1=batchOp dense W₁ → z=swishF → e2=batchOp dense W₂` ONLY to expose `s/e1/z/e2` for the param grads; `dx`
> reuses the fused `seBackBatched`. So the SE param grads chain `seReduceB → sigmoidBack(e2) → denseWeightSgdB/
> denseBiasSgdB (W₂) → denseRowBack(W₂) → swishBack(e1) → denseWeightSgdB/denseBiasSgdB (W₁)` — all existing ops.
> Everything else (stem, 1×1 conv, depthwise 3×3/5×5, true-BN, dense head) uses the Item-B batched ops as-is.
> `seReduceB` added through 4 sites (ctor/`den`/`skel`→`.batched "seReduceB"`/`emitTok`; the generic `.batched`
> Raw/Tok/parseStack covers round-trip, NO new round-trip case); `lake build Proofs` green (2267 jobs, 0 err,
> `StableHLOParse` roundtrip intact); iree-compiles on rocm/gfx1100 (10 KB vmfb).
> **REMAINING:** (4b) ✅ `seReduceB` DONE (committed `cc652bc`) + ✅ `gapBackBatched` DONE (head GAP-back,
> 2nd new op, same 4-site pattern, iree-validated, uncommitted); (5) ✅ **DONE** the 262-param renderer
> `efficientnetTrainStepFaithfulV` (`EfficientNetRender.lean`) — hand-assembled module (NOT `renderModule`)
> at SHlo **N=1** (so pointwise `swishF`/`sigmoidF` emit `[B,c·h·w]`; emit `B`=real batch), 4 fwd + 4 bwd
> block emitters, un-fused SE; `lake build` green; (6) ✅ **DONE** re-emit + iree-validate: the whole 262-param
> `efficientnet_train_step.mlir` (7466 lines) **iree-compiles CLEAN on rocm/gfx1100 (1.68 MB vmfb)**, 0 render-TODO
> stubs, 262 return tensors, func sig BYTE-IDENTICAL to the committed (drop-in for `efficientnetVerified`) — the
> **"full-16 train step" headline landed** (every line = `pretty(verified AST node)`); (7) ✅ **DONE**
> §1 fold `EfficientNetFaithfulPoC` (commit `4791cb0`): 8 generic batched param-op `den=certified` lemmas
> covering EVERY param family (conv/strided-stem/dense W,b + BN γ/β + depthwise stride-1/strided), each
> generic in dims+cotangent; the **Σ_n batch-sum bridge** (the "long pole") was a clean `Finset.sum_congr`
> of the per-example `.correct`, NOT a blocker; BN γ/β delegate to `cifar_bn_render_*_certified` at the
> generic `m=N·(h·w)`. `lake build Proofs` green (2269), all 8 3-axiom clean + wired to AuditAxioms.
> (Optional follow-up: per-block-type capstone bundling = mnv2-style "all 262 params" accounting.) The
> ORIGINAL scoping kept: (BN/dense/conv/SE-dense = ~1-line cert delegations; depthwise/stem
> need the **Σ_n batch-sum bridge** — the proof long pole); (8) §1a tie `EfficientNetTiePoC` (new vs mnv2:
> swish — smooth, no relu6 kink — + the SE gate fan-in via `seReduceB`). Then convnext (same per-block-only backward; 7×7
> depthwise + scalar LN + layer-scale), then vit (reconcile scalar-proven vs per-channel-`[192]`-emitted LN).
>
> _NOTE: the §1 matrix row (§1) + §5 enet bullet below still describe the PRE-correction state in places —_
> _the authoritative current status is this banner. enet's batched-emit core work is the new `StableHLO.lean`
> `.batched` enrichment (`batchOpDescr` helper) + the `emitTok` `.batched` dispatch + the `*SgdB` ctors._
>
> **⏸ Parked (Brett, separate future session — NOT part of this sweep): full mnv2 VJP-witness upgrade**
> (full-net scope task 5 below) — promote `Mnv2Live`'s nonzero-Jacobian seal from the 2-block
> representative to the full 17-block net. Independent of the den-tie; the hardest math; the last open
> mnv2 item.

## 0. The trust chain (what "faithful" means here)

Two universes, today mostly disconnected:

```
PROOFS (Proofs/, exact ℝ)                EXECUTION (what trains on GPU)
  reference forward fn  ── HasVJP ──▶      VerifiedNet.train reads a committed
  proven backward = fderiv-transpose       verified_mlir/<slug>_{fwd,train_step}.mlir
                                           ──iree-compile──▶ .vmfb ──FFI/IREE──▶ Float32
```

Closing the gap is one chain of equalities, split into edges:

- **Edge A — denotation = reference.** The emitted artifact, given a math
  meaning (`den`), equals the proven reference forward *and* its proven backward.
  Pattern exists via the `SHlo` AST + `den` + `*FwdGraph*_faithful` /
  `*BackGraph*_faithful` lemmas.
- **Edge B — text = the proven graph.** The bytes `iree-compile` consumes are
  `pretty/renderModule(provenGraph)` (or provably parse to it), not a parallel
  hand-written string that merely agrees.
- **Edge C — ℝ → Float32.** `FloatBridge.lean` (Tier-1 only today).

Irreducibly trusted after all edges: `iree-compile`, the IREE runtime, the C
FFI, and the per-op `den`/`pretty` op-modelling (the meaning we assign each
emitted StableHLO op) — the same boundary the forward `den` already lives at.

Key structural fact: **none of the nine verified trainers call
`MlirCodegen.lean` at runtime** (that 7551-line, 0-theorem string builder feeds
only the *non*-"Verified" `Main*Train.lean` trainers + emit tests). All nine read
committed `verified_mlir/<slug>_*.mlir`. So "doesn't use MlirCodegen" is true for
all of them — but that is *not* faithfulness. The real split is whether each
committed `.mlir` is `render(provenGraph)` (Edge B) or a hand-written `String`.

## 1. Per-net faithfulness matrix

`_fwd.mlir` = eval module; `_train_step.mlir` = training module. "proof-tied" =
the emitted bytes are `renderModule`/`pretty` of a graph with a `den`-faithfulness
theorem. "parallel" = the faithfulness theorem is about a proof-side graph the
emitter does **not** print (independent hand-written string emitter).

| Net | runtime path | `_fwd` bytes | `_train_step` bytes | proof-side faithful (parallel unless noted) | headline gap |
|---|---|---|---|---|---|
| **mnist-linear** | committed `.mlir` | ✅ `linearFwdModuleV = renderModule(fwdGraph)`, `fwdGraph_faithful` | ✅ **CLOSED** — whole module is `pretty(provenGraph)` via `linTrainStepFaithfulV` (cotangent + `weightSgd`/`biasSgd` AST ops); `den = certified` by `rfl` | + `weightSgd`/`biasSgd` `SHlo` ops, `poc_{weightSgd,biasSgd}_den_eq`, `poc_train_step_tail_certified` | — (tail folded; only per-op `pretty` lexing + ℝ→Float32 remain) |
| **mnist-mlp (1d)** | committed `.mlir` | ✅ `mlpFwdModuleV = renderModule(mlpFwdGraph)`, `mlpFwdGraph_faithful` | ✅ **CLOSED** — `mlpTrainStepFaithfulV`: whole 3-layer train step is `pretty(provenGraph)` (fwd + `dotOut`/`selectPos` backward chain + 6× `weightSgd`/`biasSgd`); each output `den = certified` (`MlpFaithfulPoC`, reusing `mlp_render_*_certified` + `mlpCotOut*_denote`) | `MlpPoC.{cot1,cot0}_den` + `MlpPoC.{W0,W1,W2,b0,b1,b2}_den_certified` | — (no new core ops; same residual as linear) |
| **mnist-cnn (2d)** | committed `.mlir` | ✅ `cnnFwdModuleV = renderModule(cnnFwdGraph)`, `cnnFwdGraph_faithful` | ✅ **CLOSED** — `cnnTrainStepFaithfulV` (CnnRender.lean) renders the whole train step as `pretty(provenGraph)`: forward + backward chain (`dotOut`/`selectPos`/`maxPoolBack`/`convBack`) + 10 param SGD ops (`convWeightSgd`/`convBiasSgd` conv + `weightSgd`/`biasSgd` dense head); each output `den = certified` via `CnnPoC.{cW,cb}{1,2}_den` (conv chain bridges) + `{dW,db}{3,4,5}_den` (M2 dense bridges) | **2 new core ops** `convWeightSgd`/`convBiasSgd` (9 sites each, `roundtrip` extended); committed bytes iree-compile on rocm/gfx1100 (121 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + ℝ→Float32) |
| **cifar (ch5, no-BN)** | committed `.mlir` | ✅ `cifarFwdGraph` rendered | ✅ **CLOSED** — `cifarTrainStepFaithfulV` (CnnRender.lean) renders the whole 2-scale train step (4 conv + 3 dense) as `pretty(provenGraph)`; each of the 14 outputs `den = certified` via `CifarPoC.conv{W,B}_den` (generic, covers all 4 conv layers) + `{dW,db}{5,6,7}_den` (M2 dense bridges) | **NO new core ops** (reuses cnn's `convWeightSgd`/`convBiasSgd`); committed bytes iree-compile on rocm/gfx1100 (186 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + ℝ→Float32) |
| **cifar-bn (ch5)** | committed `.mlir` | ✅ `cifarBnFwdGraph` rendered (BN incl.) | ✅ **CLOSED** — `cifarBnTrainStepFaithfulV` (CnnRender.lean) renders the whole BN train step (22 params) as `pretty(provenGraph)`; conv layers reuse `CifarPoC.conv{W,B}_den`, dense head `CifarPoC.{dW,db}{5,6,7}_den`, per-channel BN γ/β via new `bnGammaSgd`/`bnBetaSgd` ops (`CifarBnPoC.bn{Gamma,Beta}_den` ← `cifar_bn_render_{gamma,beta}_certified`, bridged `oc·h·w↔oc·m` by `reassocFwd`) | **2 new core ops** `bnGammaSgd`/`bnBetaSgd`; committed bytes iree-compile on rocm/gfx1100 (259 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + BN `0<ε` + ℝ→Float32) |
| **cifar8 (8-conv, no-BN)** | committed `.mlir` | ✅ `cifar8FwdGraph` rendered | ✅ **CLOSED** — `cifar8TrainStepFaithfulV` (CnnRender.lean) renders the whole 4-stage train step (8 conv + 3 dense, 22 params) as `pretty(provenGraph)`; conv via `CifarPoC.conv{W,B}_den` (generic), dense via the new generic `Cifar8PoC.dense{W,B}_den` | **NO new core ops** (pure reuse); committed bytes iree-compile on rocm/gfx1100 (271 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + ℝ→Float32) |
| **cifar8-bn** | committed `.mlir` | ✅ `cifar8BnFwdGraph` rendered | ✅ **CLOSED** — `cifar8BnTrainStepFaithfulV` (CnnRender.lean) renders the whole BN train step (8 conv + 8 BN + 3 dense, 38 params) as `pretty(provenGraph)`; **no new ops, NO new proof** — every output's `den` = certified by the existing generics (`CifarPoC.conv{W,B}_den`, `CifarBnPoC.bn{Gamma,Beta}_den`, `Cifar8PoC.dense{W,B}_den`) | committed bytes iree-compile on rocm/gfx1100 (393 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + BN `0<ε` + ℝ→Float32) |
| **r34** | committed `.mlir` | ❌ hand-written (`TestResnet34Fwd`) | ✅ **CLOSED** — `resnet34TrainStepFaithfulV` (ResNet34Render.lean) renders the whole `[3,4,6,3]` train step (146 params) as `pretty(provenGraph)`: 7×7/s2 stem + 16 residual blocks (residual cotangent-sum via `addV` at each skip merge) + GAP + dense; **2 new core ops** `convStridedWeightSgd`/`convStridedBiasSgd` (7×7 stem + 3×3 strided down/proj), den-certified via `mnv2_render_stem_conv{W,b}_certified` (`ResNet34PoC.convStrided{W,B}_den`); 142 other params reuse the cifar conv/BN/dense generics | committed bytes iree-compile on rocm/gfx1100 (537 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin incl. residual fan-in sums + BN `0<ε` + ℝ→Float32) |
| **mnv2** | committed `.mlir` | ❌ hand-written | ✅ **CLOSED (§1 fold, reduced 6-block)** — `mnv2TrainStepFaithfulV` (MobileNetV2Render.lean) renders the whole reduced-6-block train step (82 params) as `pretty(provenGraph)`; each param `den = certified` via `MobileNetV2FaithfulPoC` — **4 new core ops** `depthwise{,Strided}{Weight,Bias}Sgd` (StableHLO.lean, the per-channel `batch_group_count=c` transpose-trick weight + `convBiasSgd`-aliased bias), expand/project conv via `CifarPoC.conv{W,B}_den`, BN via `CifarBnPoC.bn{Gamma,Beta}_den`, dense via `Cifar8PoC.dense{W,B}_den`; committed bytes iree-compile on rocm/gfx1100 (789 KB vmfb), drop-in positional param layout. **FULL 17-block paper net now also DONE**: §1 CLOSE (`mnv2TrainStepFaithfulVPaper`, 210 params, 559 KB vmfb), §1 fold den (`MobileNetV2FaithfulPoCPaper`), and §1a TIE (`mnv2_net_tied_certified`) all landed + 3-axiom clean | **§1a tie DONE** (full 17-block, `MobileNetV2TiePoCPaper`); remaining: trainer swap to the full `.mlir` (task 4) + 2-block→17-block VJP-witness upgrade (task 5, separate §4) |
| **enet** | committed `.mlir` | ◐ hand-written (batched emit "Item B" now BUILT — `pretty(provenGraph)` validates; renderer pending) | ◐ hand-written; all batched ops + param-SGD BUILT + iree-validated (Item B done); renderer pending | `efficientnetFwdGraphB_full_faithful` (full 16 MBConv) + `efficientnetForwardB_full_has_vjp` (whole-net VJP); den-faithful per-block backward bricks | **real blocker was Item B (batched `emitTok` stub) + missing batched param-SGD, NOT "no whole-net backward"** — both DONE (5 commits, 2026-06-18); renderer/fold/tie remain |
| **convnext** | committed `.mlir` | ❌ hand-written | ✅ **CLOSED (§1 render + §1a tie)** — `convNextTrainStepFaithfulV` renders the full [3,3,9,3] train step (**180 params**) as `pretty(provenGraph)`, iree-validated (647884 B); §1a tie `CnxTiePoC.cnx_net_tied_certified` threads all **176** SHlo-op params through the REAL render forward + the loss-driven backward (GELU masks, identity-skip fan-in ×18, downsample LN-back ×3, scalar-LN γ/β + per-channel layer-scale γ); the **4 even-kernel weight grads** (stem 4×4/s4 `psW` + 3 downsample 2×2/s2 `d{0,1,2}W`) are the documented render gap (no even/stride-4 weight-grad VJP op), outside the den-tie — their *bias* grads ARE tied | **§1a TIED** (176/180; 4 even-kernel weight-grad gaps; the §1 fold added 3 ConvNeXt core ops `layerScaleChGammaSgd`/`lnGammaSgd`/`lnBetaSgd`, the tie reuses them — ZERO new ops/bridges) |
| **vit** | committed `.mlir` | ❌ hand-written (`LeanMlir/ViTRender.lean` String emitter, faithful per-op, NOT `pretty(provenGraph)`) | ❌ hand-written String emitter (depth-12, 200 params) | **richest — per-channel `[192]` LN fully modeled**: `ViTVecLN` (`layerNormVec` fwd/VJP + `vitFwdGraphV`/`_faithful` render graph + `vitForward2V_has_vjp` + `vit_render_vecln*_certified` + chain cots `vitCot*V`); `ViTClose` rowdense/patch/cls/pos certs; (old scalar `vitNetBackGraph_faithful` is a separate parallel universe the tie doesn't use) | **convnext situation (full §1 stack to build) — NOT blocked by the LN granularity gap** (per-channel `[192]` is modeled by ViTVecLN). TO BUILD: §1 render + fold + §1a tie + core ops. Core ops: only **`veclnGammaSgd`** is new (rowdense W/b + vecln β reuse enet's `denseWeightSgdB`/`denseBiasSgdB`) |

### What is genuinely proof-tied to emitted bytes today
- **Forward-eval modules of linear, mlp, cnn, cifar (ch5), cifar-bn (ch5), cifar8 (8-conv),
  cifar8-bn** — `<slug>_fwd.mlir` literally *is* `renderModule(provenGraph)` with a `den`-
  faithfulness theorem about that exact text (cifar8/cifar8-bn closed via `cifar8{,Bn}FwdModuleV`
  = `renderModule(cifar8{,Bn}FwdGraph)`, the graphs already audited faithful).
- **Closest train step: linear** — `_train_step.mlir`'s forward+cotangent is
  `pretty(lossCotGraph)` (one composed proven graph, `den = CE-grad`); but the SGD
  tail still consumes the cotangent via `.operand %dy <placeholder>` (SSA-name pin),
  not the `lossCotGraph` node — so even linear isn't *fully* tied (see §1a).

### 1a. Forward graph ⟷ train step: tied, or parallel? (the deepest residual)

The `_fwd` closes above tie the **eval** modules to their proven forward graphs. The
**train steps do NOT reuse those graphs.** Each `*TrainStepFaithfulV` re-renders the
forward as a chain of separate `pretty (.flatConvF …)` / `pretty (denseF …)` nodes, and
every node past the first is fed `.operand <name> <placeholder-zero>` — the SSA *name*
of its predecessor, but a placeholder *value*. So:

- The forward is **not** a single composed proven graph (each per-node `den` is about a
  placeholder operand, never chained into `den(*FwdGraph)`); `*FwdGraph_faithful` is never
  invoked by the train step.
- The param-SGD `den` theorems are `∀ activation, cotangent` — genuinely certified, but for
  *symbolic* operands. That the runtime SSA an op references actually carries the proven
  forward activation / backward cotangent is **trusted by name**, not proven. This is the
  "cotangent-subgraph⇄SHlo pin" residual every fold lists.

So `den(_fwd graph) = forward` (proven, composed) and `den(SGD op) = certified ∀ c`
(proven, per-op) are **two disconnected facts** — a bug wiring the wrong SSA into an SGD op
leaves both green.

| net | train step reuses proven fwd graph? | consumers composed (no SSA-name pin)? | **tie** |
|---|---|---|---|
| **linear** | ✅ forward = `fwdGraph` (nested in `lossCotGraph`, fed directly) | ✅ `weightSgd`/`biasSgd` consume `lossCotGraph` directly | **✅ FULL** |
| **mlp** | ✅ den-composed: real forward threaded; top cotangent `g` pinned to the composed softmax-CE (`mlpLossCot_den`) | ◐ level-2: consumers fed real forward dens at correctly-threaded SSAs; `W₂` folds to `∂CE/∂W₂` | **✅ TIED** |
| **cnn** | ✅ conv+dense den-composed: real conv forward threaded through `ac1`/`ac2`/`pool` (`cnn_conv_tied_certified`), cotangent = softmax-CE of the conv forward (`cnnLossCot_den`), output `W₅` → `∂CE/∂W₅` | ◐ level-2: cotangents at correctly-threaded SSAs (conv backward rendered hand-written, not `SHlo`) | **✅ TIED** |
| **cifar (ch5)** | ✅ conv+dense den-composed: real 2-stage conv forward threaded through `ac1`–`ac4`/`zp1`/`pool2` (`cifar_conv_tied_certified`), cotangent = softmax-CE of the cifar forward (`cifarLossCot_den`), output `W₇` → `∂CE/∂W₇` (`cifar_W7_tied_totalloss`) | ◐ level-2: cotangents at correctly-threaded SSAs (conv backward rendered hand-written, not `SHlo`); the new `cifarChainCotW2` crosses pool₁ (conv₃-back then maxpool₁-back) — the step cnn (one pool) lacked | **✅ TIED** |
| **cifar-bn (ch5)** | ✅ conv+BN+dense den-composed: real cifar-BN forward threaded, cotangent = softmax-CE of the BN forward (`cifarBnLossCot_den`), all 16 conv/BN params tied at the BN backward chain (`cifarBn_convbn_tied_certified`), output `W₇` → `∂CE/∂W₇` (`cifarBn_W7_tied_totalloss`) | ◐ level-2: cotangents at correctly-threaded SSAs (conv/BN backward rendered hand-written); the chain alternates BN-output cot (relu-masked, γ/β) and conv-output cot (BN input-VJP of it) — cifar's chain + a BN-back at every conv | **✅ TIED** |
| **cifar8 (8-conv)** | ✅ conv+dense den-composed: real 4-stage forward threaded, cotangent = softmax-CE of the cifar8 forward (`cifar8LossCot_den`), all 16 conv params tied at the 4-stage backward chain (`cifar8_convs_tied_certified`), output `Wb` → `∂CE/∂Wb` (`cifar8_Wb_tied_totalloss`) | ◐ level-2: cotangents at correctly-threaded SSAs (conv backward rendered hand-written); cifar's chain repeated over 4 stages — all reused constructors (`cnnChainCotW2`/`cnnChainCotW1`/`cifarChainCotW2`), no new chain content | **✅ TIED** |
| **cifar8-bn** | ✅ conv+BN+dense den-composed: real cifar8-BN forward threaded, cotangent = softmax-CE of the BN forward (`cifar8BnLossCot_den`), all 32 conv/BN params tied at the 4-stage BN backward chain (`cifar8Bn_convbn_tied_certified`); dense head via the pre-audited generics | ◐ level-2: cotangents at correctly-threaded SSAs (conv/BN backward rendered hand-written); cifar8's 4-stage chain + a BN-back at every conv — pure reuse, no new content | **✅ TIED** |
| **r34** | ✅ **whole net** den-composed: all 16 residual blocks + stem threaded at the real `resnet34Forward_full_pc` activations, cotangent composed from the loss through dense/GAP-back + the **residual fan-in sum** at every skip (`idBlockCotIn`/`downBlockCotIn`); capstone `r34_net_tied_certified` bundles every block's tie + dense total-loss fold + `r34LossCot_den` | ◐ level-2: cotangents at correctly-threaded SSAs (block backward rendered hand-written, not `SHlo`); the new fan-in-sum constructors add the skip+body cotangent merge cnn/cifar (no residuals) lacked | **✅ TIED** |
| **mnv2** (full 17-block paper) | ✅ **whole net** den-composed: all 17 inverted-residual blocks + stem + conv-bn-relu6 head threaded at the real `mobilenetv2ForwardPaper` activations (single-ε), cotangent composed from the loss through dense/GAP-back + the head + the **residual fan-in `+ dyOut`** at every stride-1 skip (`ivS1SkipCotInAt`); capstone `mnv2_net_tied_certified` bundles all 210 params' ties + dense total-loss fold + `mnv2LossCot_den` | ◐ level-2: cotangents at correctly-threaded SSAs (block backward rendered hand-written, not `SHlo`); relu6 two-kink mask + linear project bottleneck (project-BN cot = `dyOut` directly) are the new content vs r34 | **✅ TIED** |
| **enet** (full 16-MBConv, 262 params) | ✅ **whole net** den-composed: all 16 MBConv blocks + stem + conv-bn-swish head threaded at the real `efficientnetForwardB_full` activations (batched true-BN + SE), cotangent composed from the loss `g` down through the head VJP + every block's VJP (residual fan-in folded into `mbResidW`'s own VJP); capstone `efficientnet_net_tied` bundles all 262 params' ties + the loss-cotangent den (`efficientnetLossCot_den`) | ◐ level-2: cotangents at correctly-threaded SSAs (block backward rendered hand-written, not `SHlo`); **swish** masks + the **SE gate fan-in** (`gateCotB`/`sigBackB`/`rowDenseBackFlat`) + **true-BN** backs (`bnBatchLA` VJP) + strided depthwise, all **batched** — the new content vs mnv2 | **✅ TIED** |
| **convnext** (full [3,3,9,3], 176 tied params) | ✅ **whole net** den-composed: all 18 ConvNeXt blocks + 3 downsamples + GAP→LN→dense head + stem bias threaded at the real `convNextTrainStepFaithfulV` forward activations (`cnxStemFwdO`/`cnxBlockFwdO`/`cnxDownFwdO`), cotangent composed from the loss `g` down through dense (`dense_has_vjp`) + head-LN + GAP (`globalAvgPoolFlat_has_vjp`) + every block's backward, with the **identity-skip fan-in `+ dyOut`** at each of the 18 block merges + the LN-back at each of the 3 downsamples; capstone `cnx_net_tied_certified` bundles all 176 params' ties + the dense total-loss fold + `cnxLossCot_den` | ◐ level-2: cotangents at correctly-threaded SSAs (block backward rendered hand-written, not `SHlo`); **GELU mask** (smooth, no kink — `geluScalarDeriv`) + **scalar LayerNorm** γ/β (`Vec 1`) + **per-channel layer-scale** γ (`Vec c`, `chanIdx` broadcast) are the new content vs mnv2's relu6/per-channel-BN | **✅ TIED** |
| vit | — (Tier-3, last — IN PROGRESS: §1 render/fold/tie to build; per-channel `[192]` LN modeled by `ViTVecLN`, NOT a blocker) | — | **▶ building** |

**The close ("tie them together").** Feed the *proven* cotangent/forward subgraph directly
into each consumer (`weightSgd … (lossCotGraph …)` instead of `.operand %dy …`), so each
output's `den = certified` is **one composed theorem** with the forward = the proven
`*FwdGraph` — no pin. Cost: a shared cotangent feeding K outputs is rendered K times (the
`SHlo` is a *tree*, no DAG sharing); for the 1d nets (linear 2, mlp 6 outputs) that's fine
and iree CSEs the duplicates. The scalable, no-duplication version needs a shared-SSA / DAG
renderer with proven late-binding (the handoff §3 "the real work") — deferred until the 1d
nets validate the pattern.

**Status: linear FULLY TIED** (`linTrainStepFaithfulV` feeds `lossCotGraph` directly into
`weightSgd`/`biasSgd`; `poc_train_step_tail_certified` is now one composed theorem with the
forward = `fwdGraph`, no SSA-name pin; the regenerated `linear_train_step.mlir` iree-compiles
to a **byte-identical-size vmfb** — iree CSEs the duplicated cotangent, so zero runtime cost).

**mnist-mlp tie — DONE (NOT via the DAG renderer).** On closer reading, `MlpFaithfulPoC` already
threads the **real forward activations** into its backward/SGD `den` theorems (`cot1_den` feeds the
real pre-activation `dense W₁ b₁ (relu …)`; `W2_den_certified` feeds the real activation
`relu (dense W₁ b₁ (relu …))`). So mlp's backward is *already* den-composed — the value fields
carry real forward dens, referenced by the correctly-threaded SSAs (the only residual there is the
universal per-op `pretty` lexing trust, same as every net). The **one** open gap is the top loss
cotangent `g`, left `∀ g`. Closing it = a capstone that instantiates the six `*_den_certified` at
`g = softmax(mlpForward x) − onehot` and folds via the existing `mlp_{output,hidden,input}_total_loss_grad`
(the `mlpForward` chain rule) — exactly the `lossCotGraph_isCEgrad` move linear used. **No new ops,
no DAG renderer, no duplication, no renderer change** — just a PoC capstone (the emitted graph
was already correctly threaded; skel erases the placeholder values). **Landed:** `mlpLossCot_den`
(the emitted loss graph denotes `∂CE/∂logits` at the real forward logits), `mlp_W2_tied_totalloss`
(the output weight op denotes `W₂ − lr·∂CE/∂W₂`, the WHOLE-loss gradient, via `mlp_output_total_loss_grad`),
and `mlp_train_step_tied_certified` (all six outputs at the composed cotangent). The deeper total-loss
*fold* for `W₁`/`W₀` (single `pdiv(crossEntropy∘forward)` form) needs the chain-cotangent↔loss-grad-at-
preactivation lemma + the conditional relu-smoothness hyps — deferred; the level-2 composed tie (real
forward dens threaded at correct SSAs) holds for all six now.

**Where the DAG renderer is actually needed: the conv nets.** Unlike mlp, the conv PoCs
(`CifarPoC.convW_den`, etc.) leave the activation `∀`-symbolic — so the conv nets need the *whole
backward chain* composed (real values threaded, like mlp's backward already is) before `g` can be
closed; that is the "compose a whole-net backward graph" step in §2, the bigger lift. A *structural*
tie (consumed graph as an `SHlo` child, à la linear, with no SSA reference at all) on a deep net
would duplicate the forward O(depth²) times — that is the case a shared-SSA/DAG renderer with proven
late-binding solves. So: mlp → capstone (cheap); conv → whole-net-backward composition + (eventually)
the DAG renderer to keep it from blowing up.

**cnn — DONE at the den level (first conv net).** Both the dense head AND the conv layers are now
den-composed. `cnnLossCot_den` pins the top cotangent to `softmax(mnistCnnNoBnForward x) − onehot`;
`cnn_W5_tied_totalloss` folds the dense output to `∂CE/∂W₅`; and `cnn_conv_tied_certified` ties all
four conv kernel/bias ops at the REAL conv forward and the composed cotangent. The Vec↔Tensor3
wrinkle (the forward runs in flat `Vec` space via `flatConv`, the backward/SGD read `Tensor3` via
`conv2d`) is bridged in the statement: each conv activation appears as its `Vec` form (for
`flatConv`/`maxPool`) and `Tensor3.unflatten` of it (for `conv2d`/`convWeightSgd`/`cnnChainCot`) — and
since the `cW*_den` hold for *any* free activation, this lives purely in the statement, not the proof.
No free activations, no symbolic cotangent. **Remaining polish (the stricter structural form, shared
by every conv net):** the emitted conv backward is rendered **hand-written string** (`selMask4`/
`scatter`/`convBack`/`convWGrad`), so the cotangent SSA ↔ `cnnChainCot` correspondence is per-op
trust (the universal residual); re-rendering it as `pretty(SHlo)` + a `den` pin (the cnn analogue of
`MlpPoC.cot{0,1}_den`, crossing `convBack`/`maxPoolBack`) would remove even that. cifar/r34 inherit
this exact pattern — the den-level conv fold is now a worked template.

### cifar (ch5) tie — ✅ DONE (this session)

**Landed exactly as the plan below scoped it.** Three capstones in `CifarFaithfulPoC.lean`
(`namespace Proofs.CifarPoC`), all 3-axiom clean (`[propext, Classical.choice, Quot.sound]`), wired
into `tests/AuditAxioms.lean`: `cifarLossCot_den` (the emitted loss graph denotes
`softmax(cifarCnnForward x) − onehot`; copy of `cnnLossCot_den`), `cifar_W7_tied_totalloss` (the
dense output `W₇` folds to `∂CE/∂W₇` through the whole forward; copy of `cnn_W5_tied_totalloss` via
`mlp_output_total_loss_grad`), and `cifar_conv_tied_certified` (all 4 conv layers den at the real
forward + composed cotangent). **Key structural insight that made it cheap:** `CifarPoC.convW_den`/
`convB_den` are already generic in the cotangent, so the conv tie only needed the backward-chain
cotangents *built* and fed in — no new bridge theorems. Three of the four reuse the cnn chain cots
verbatim (`cnnChainCotW2` for conv₄ at the cifar head dims, `cnnChainCotW1` for conv₃/conv₁); the
**only new def** is `cifarChainCotW2` — conv₂'s cotangent crosses pool₁ at the relu-free conv₃ input,
so it is relu₂-mask on `maxpool₁-back(conv₃-back(W₃, cotW3))` (a conv input-VJP *then* a maxpool
input-VJP — the step cnn's single pool never had; same shape as `cifar8CotBn8`'s maxpool step, no BN).
No renderer/mlir change (den-level tie, as with cnn/mlp). Residual unchanged from cnn: the conv
backward is rendered hand-written, so the cotangent SSA ↔ chain-cot correspondence is the per-op trust
the whole suite carries. **The bonus** (`cifarTrainStepFaithfulV` emitting `SHlo` backward nodes so the
cotangent-subgraph could be pinned as a `den`, removing even the per-op-SSA residual) was NOT pursued
— matching cnn's scope; it stays the polish. **§1a tie-table cifar (ch5) row flipped to ✅ TIED.**

### r34 (ch6) §1a tie — ✅ DONE (this session, the full whole-net thread)

The §1a tie for the full `[3,4,6,3]` ResNet-34, all 3-axiom clean, in `ResNet34TiePoC.lean`:
**`r34_net_tied_certified`** threads `resnet34Forward_full_pc`'s real activations through all 16
residual blocks + stem + GAP/dense, composes the backward cotangent from the loss `g` down through
dense (`dense_has_vjp`) + GAP (`globalAvgPoolFlat_has_vjp`) + the **residual fan-in sum at every one
of the 16 skip merges**, and bundles every block's tie + the dense total-loss fold + the loss-cotangent
denotation. r34's structural novelty vs cnn/cifar (no residuals): the block-INPUT cotangent is
`skip-branch + body-branch` — the new constructors `idBlockCotIn`/`downBlockCotIn` (the cotangent ADD
at each merge). Reusable per-block-type tie lemmas (`r34_idblock_tied` 8 params / `r34_downblock_tied`
12 / `r34_stem_tied` 4, each pure instantiation of the generic `den = certified` lemmas at the
`ResNet34ChainClose` cotangents) are proven once and applied 16×; **no new core ops, no new bridges.**
Engineering gotcha that mattered: the 16-deep forward/backward let-thread blew the heartbeat limit via
eager unfolding — fixed by **irreducible aliases** for the forward steps (`idFwdO`/`downFwdO`/`stemMpO`)
and **`@[irreducible]` on all the `*TiedAt`/`*CotInAt` wrappers**, so the elaborator keeps the thread
opaque (the tie lemmas are generic in the block input, so opacity is harmless). **§1a row flipped to
✅ TIED.** Next tie targets: cifar8-bn, then mnv2/enet/convnext/vit (each with its own §5 blocker).

### cifar8 (8-conv) §1a tie — ✅ DONE (this session)

cifar (ch5)'s tie repeated over **four** conv→conv→pool stages, in `Cifar8TiePoC.lean`, 3-axiom clean.
The 4-stage backward chain reuses **every** existing constructor at the deeper dims: `cnnChainCotW2`
(conv₈, last before pool₄), `cnnChainCotW1` (conv₇/₅/₃/₁, within-stage conv-back), `cifarChainCotW2`
(conv₆/₄/₂, the cross-pool move) — no new constructor. `cifar8_convs_tied_certified` ties all 16 conv
params at the real forward + these chain cots (via `CifarPoC.convW_den`/`convB_den`, generic in the
cotangent); the dense head (3-layer MLP) is covered by the generic `denseW_den`/`denseB_den`; plus
`cifar8LossCot_den` + the `Wb` total-loss fold. Pure reuse, **zero new ops/bridges/constructors**.
**§1a row flipped to ✅ TIED.** Next: cifar8-bn.

### cifar8-bn §1a tie — ✅ DONE (this session)

cifar8's 4-stage chain + a BN-back at every conv (exactly the cifar→cifar-bn step, at 4 stages), in
`Cifar8BnTiePoC.lean`, 3-axiom clean. `cifar8Bn_convbn_tied_certified` ties all 32 conv/BN params at
the real forward + the BN backward chain (BN-output cots `dyBn1–8` relu-masked for γ/β, conv cots
`cotC1–8` = `bnPerChannelTensor3_grad_input` of them for W/b), plus `cifar8BnLossCot_den`; the dense
head is covered by the pre-audited `Cifar8PoC` generics. **Zero new ops/bridges/constructors** — conv
via `CifarPoC.convW_den`/`convB_den`, BN via `CifarBnPoC.bnGamma_den`/`bnBeta_den`. **§1a row flipped to
✅ TIED — the entire cifar family (cifar, cifar-bn, cifar8, cifar8-bn) + cnn/mlp/linear + r34 are now
TIED.** Remaining: the Tier-3 nets mnv2 / enet / convnext / vit, each with its own §5 blocker (mnv2:
reduced-net + 2-block VJP witness; enet/convnext: no whole-net backward graph; vit: scalar-vs-per-channel
LN granularity).

## NEXT SESSION: mnv2 §1a tie — handoff

_Update 2026-06-18: **this handoff under-scoped it** — it presupposed mnv2's §1 fold (every param op
`den = certified` + a `render(provenGraph)` train step) was done, as it was for all 9 prior TIED nets.
It was NOT — the committed `mobilenetv2_train_step.mlir` was hand-written and there were no depthwise SGD
`SHlo` ops. So mnv2 took the full r34-scale effort. **§1 fold now DONE** (3 commits on main): `783dd85`
core ops (4 `depthwise{,Strided}{Weight,Bias}Sgd`), `e8310c9` `MobileNetV2FaithfulPoC` (den=certified),
`fc31ca7` `MobileNetV2Render` (committed mlir = render(provenGraph), iree 789 KB). **What remains is the
genuine §1a tie below — now a real TiePoC on top of the committed den lemmas** (`Mnv2PoC.depthwise*_den`).
Lesson recorded: a "§1a tie" handoff is only valid if the matrix `_train_step` column is already ✅._

_**GOAL CHANGED 2026-06-18 → the FULL 17-block paper net** (Brett: "full net is the goal"). The reduced
6-block §1 fold above is the FOUNDATION (it built the depthwise SGD core ops + validated the pattern), not
the end state; mnv2 is the only reduced net among the 5 "verified" trainers (r34/enet/convnext/vit are full).
Much full-net infra is ALREADY committed + 3-axiom clean: `MobileNetV2FullPaper.lean` has
`mobilenetv2ForwardPaper` (full ℝ-forward, per-channel BN) AND `mobilenetv2FwdGraphPaper` +
`mobilenetv2FwdGraphPaper_faithful` (`den(graph)=forward`), built from per-block graph helpers
`iv{NoExp,ExpOnly,Resid,Strided}GraphW`(+`_faithful`); the param-grad render-math lemmas
(`mnv2_render_depthwise*_certified`, conv/BN/dense generics) are **dimension-generic** so they apply at full
dims (channels up to 320) unchanged; my reduced renderer's per-block emitters (`irFwd`/`irBack`/strided in
`MobileNetV2Render.lean`) are dimension-generic and REUSABLE. So the full net reuses everything; the new work:_

_**Full-net scope (tasks 3–6):**_
1. _**Full §1 CLOSE — ✅ DONE (`eb4de9f`)**: `mnv2TrainStepFaithfulVPaper` (17-block paper SGD train step,
   210 params, render(provenGraph)) appended to `MobileNetV2Render.lean` — reuses the dim-generic emitters +
   2 new variants (`irFwd/irBackNoExp` for the t=1 no-expand b1; `irFwd/irBackNoSkip` for the ic≠oc widenings
   b11/b17); writes `verified_mlir/mobilenetv2_paper_train_step.mlir` (iree 1.22 MB vmfb); reduced renderer
   untouched, both coexist. Original scoping kept for the record:_
   _`mnv2TrainStepFaithfulVPaper` (17-block SGD renderer): reuse the per-block emitters +
   ADD a no-expand variant for b1 (t=1, `IVWNoExp`: depthwise→project, no expand); assemble the
   `[t,c,n,s]` schedule (stem 3→32; stages 16/24/32/64/96/160/320; head 320→1280; dense 1280→10; 17 blocks).
   Crib the schedule from the TEST-ONLY `tests/TestMobilenetV2TrainPC.lean` (which already renders the full
   net for AdamW — forward via `pretty(mobilenetv2FwdGraphFullPC)`, hand-emitted tail). Write the committed
   full SGD `.mlir` + iree-validate. ~214 param tensors (vs reduced 82; +132 = 11 extra blocks ×12)._
2. _**Full §1 fold (den) — ✅ DONE (this session)**: `MobileNetV2FaithfulPoCPaper.lean`. Confirmed the
   17-block paper renderer emits ONLY the twelve param-SGD op types the reduced net already exercises (no
   new op type, no new block-level param op — the no-expand b1 / no-skip b11,b17 variants change the
   forward/backward WIRING, not the param ops). So every one of the 210 params is den=certified by an
   already-audited generic — the **cifar8-bn lesson at full scale, ZERO new ops/lemmas**. Added six
   per-block-type capstones (`mnv2{Stem,NoExp,Stride1,Stride2,Head,Dense}ParamsCertified`), each GENERIC in
   the block dims (so one theorem covers every block of that type) and each conjunct a direct delegation to
   the audited `CifarPoC.conv{W,B}_den` (expand/project/head 1×1) / `Mnv2PoC.depthwise{,Strided}{W,B}_den` /
   `ResNet34PoC.convStrided{W,B}_den` (stem) / `CifarBnPoC.bn{Gamma,Beta}_den` / `Cifar8PoC.dense{W,B}_den`.
   210-param accounting (stem 4 + b1 no-exp 8 + b2..b17 16×12=192 + head 4 + dense 2). Wired to lakefile
   `Proofs` root + `tests/AuditAxioms.lean`; `lake build Proofs` green (2266 jobs); all six capstones
   3-axiom clean `[propext, Classical.choice, Quot.sound]`. Gotcha hit: a literal `-/` inside a header table
   cell (`dim-/cotangent`) prematurely closed the module doc-comment — escaped it._
3. _**Full §1a TIE — ✅ DONE (this session)**: `MobileNetV2TiePoCPaper.lean`. `mnv2_net_tied_certified`
   threads `mobilenetv2ForwardPaper`'s real activations (single-ε per-channel-BN, = the renderer's
   forward) through all 17 blocks + stem + conv-bn-relu6 head + dense, composing the backward cotangent
   from the loss `g` down through dense (`dense_has_vjp`) + GAP (`globalAvgPoolFlat_has_vjp`) + the head +
   every block's backward, with the **residual fan-in `+ dyOut` at each of the ten stride-1 skip merges**.
   THREE per-block-type tie defs+theorems (`ivS1Tied` 12-param — covers all stride-1 skip AND the no-skip
   widenings b11/b17; `ivS2Tied` 12-param strided; `ivNoExpTied` 8-param b1) + stem (4) + conv-bn-relu6
   head (4) + dense fold + loss-cot. Each conjunct delegates to the §1-fold generics
   (`CifarPoC.conv*`/`Mnv2PoC.depthwise{,Strided}*`/`CifarBnPoC.bn*`/`ResNet34PoC.convStrided*`/
   `Cifar8PoC.dense*`) at the `MobileNetV2ChainClose` cotangents (`invresCotPc/Dc/EcS1/EcS2`, `mnv2StemCot`).
   What's new vs r34: the **relu6 two-kink mask** (`if 0<x∧x<6`, the `selectMid` form) at the expand+
   depthwise BN outputs, the **linear project bottleneck** (project-BN cot = `dyOut` directly, no relu6),
   3 block variants + the conv-bn-relu6 head. `@[irreducible]` `FwdO`/`CotInAt`/`TiedAt` wrappers keep the
   17-deep chain opaque (r34 heartbeat lesson; `maxHeartbeats 8000000`, capstone elaborates in ~2s).
   Wired to lakefile + AuditAxioms; `lake build Proofs` green (2267 jobs); all 6 tie capstones 3-axiom
   clean. ZERO new core ops, ZERO new bridges — pure thread + fan-in over the existing generics._
   _Residual: block backward rendered hand-written (cotangent SSA↔chain-cot per-op trust); per-op
   `pretty` lexing; relu6 two-kink + BN `0<ε` smoothness; ℝ→Float32 — the boundary every fold carries._
4. _**Trainer swap + canonical standardization — ✅ DONE (this session, validated on the rocm box)**.
   Decided (Brett): standardize the whole mnv2 stack on the **canonical 210-param net** (b1 is t=1 →
   NO expand conv, torchvision-standard; matches the repo's own `mbConvSE` t=1-skip + the §1 fold/§1a
   tie, which were already 210). Was: the spec/`fwd`/`fwd_eval`/adam carried a redundant b1 32→32
   expand (214), while the SGD `train_step.mlir` was the STALE reduced 6-block (82). Changes:
   `VerifiedSpec.invertedResidual` + `IreeRuntime.MobileNetV2Layout.irBlk` skip the expand when
   `mid==ic` (mirrors `mbConvSE`); `bnChannels` drops b1's expand-BN (53→52 BN); `#guard
   toSpecs==MobileNetV2Layout.specs` holds at **210**. The SGD `mobilenetv2_train_step.mlir` is now
   `mnv2TrainStepFaithfulVPaper` (the PROOF-TIED 210 paper render, func renamed to `@mobilenetv2_train_step`
   via a `funcName` param; reduced render → `mobilenetv2_reduced_train_step.mlir` demo). The hand-emitters
   `TestMobilenetV2Fwd` (`fwd`+`fwd_eval`) and `TestMobilenetV2TrainPC` (adam + `bnLayers` running-stats)
   got the same t=1 no-expand skip across fwd/bwd/param-grad/sgd/sig/adam-moment/bn-stat paths. All FOUR
   committed `.mlir` regenerated at 210 and **iree-compile on gfx1100** (train_step 1.2 MB, fwd 367 KB,
   fwd_eval 241 KB, adam 1.69 MB). **`mobilenetv2-verified` smoke-tested on-box**: both vmfbs compile,
   imagenette loads clean (9469 train/3925 val — NO short read on the 256²-train data; the "short read"
   was only an `LEAN_MLIR_IMAGENETTE_TRAIN` resolution mismatch, NOT mnv2-specific — same shared loader
   as r34), 210 params/2.25M floats, He init, trains without FFI mismatch. `lake build Proofs` green
   (2267) + 3-axiom closure clean (the standardization ALIGNED the proofs (already 210) with the spec).
   `MainMobilenetV2Verified` docstring updated to the full 17-block 210-param net. (Stale
   `mobilenetv2_paper_train_step.mlir` removed — the train step IS the paper render now.)_
5. _**VJP witness upgrade (separate §4 track) — ⏸ DEFERRED to a dedicated future session (Brett, this
   session)**. `Mnv2Live` (`MobileNetV2JacobianSeal`) seals the whole-net nonzero-Jacobian only at a
   **2-block structural representative** (1-ch, 2×2, sealed at input 0); upgrade to the full 17-block net.
   The hardest math (the level-3 nonzero-Jacobian seal at realistic dims), and the LAST open mnv2 item.
   Does NOT block / is independent of everything else mnv2 (§1 fold, §1a tie, and the trainer are all
   den=certified at the real cotangent — they don't rely on the nonzero-Jacobian guarantee). **Brett will
   take this in a separate session at some point — not part of the current §1a-tie sweep.** When picked up:
   start from `Mnv2Live` (the 2-block witness in `MobileNetV2JacobianSeal.lean`) + the [[whole-net-backward-b2]]
   memory (r34 reached full [3,4,6,3] depth at level 3 — the mnv2 analogue is the target); the open frontier
   there is BN-CNN + realistic-dims._

**Goal (reduced-net, ALREADY DONE — kept as the worked foundation):** tie the committed MobileNetV2 train
step to its real forward, the same §1a tie now landed on 9 nets. mnv2 is the first Tier-3 net; it has residual structure (inverted-residual / MBConv blocks),
so **r34 is the closest template** (`ResNet34TiePoC.lean`) — residual fan-in sums + the whole-net
thread with `@[irreducible]` wrappers to dodge the heartbeat blowup.

### The §1a tie recipe (proven on linear/mlp/cnn/cifar{,-bn}/cifar8{,-bn}/r34)
1. The §1 fold already gives every param op `den = certified ∀ cotangent` (generic in the cotangent).
2. Build the **whole-net backward chain cotangents** threading the REAL forward activations, with the
   **fan-in SUM at every residual skip merge** (block-input cot = skip-branch + body-branch).
3. Feed each op its chain cotangent into the generic `*_den` lemma — `exact <generic> … cot …`.
4. `*LossCot_den` (loss graph denotes softmax-CE of the real forward) + the dense head total-loss fold.
5. For deep/residual nets: per-block-type tie lemmas (proven once, applied per block) + the thread as a
   theorem whose `let`-block threads inputs (via the forward) and dyOuts (via fan-in constructors);
   **mark the forward-step + cotangent-in + tie wrappers `@[irreducible]`** so the elaborator keeps the
   deep `let`-chain opaque (without this r34 blew the heartbeat limit at both statement and proof).

### What already exists for mnv2 (a lot — most of the per-block work is done)
- **`MobileNetV2Close.lean`** — the §1-fold den-certified lemmas, **generic in the cotangent**:
  `mnv2_render_depthwiseW_certified` / `…b_certified` (stride-1), `…_strided_certified` (stride-2),
  `mnv2_render_stem_conv{W,b}_certified` (the strided stem — also reused by r34). The expand/project
  1×1 convs are stride-1 regular convs → reuse `CifarPoC.convW_den`/`convB_den`; BN γ/β →
  `CifarBnPoC.bnGamma_den`/`bnBeta_den`; final dense → `Cifar8PoC.denseW_den`/`denseB_den`.
- **`MobileNetV2ChainClose.lean`** — the **per-block cotangents already built**: `invresCotDc` (the MBConv
  block's depthwise-output cotangent) + `mnv2StemCot`, with chain-certified theorems pinning each
  depthwise/stem op to them. This is the mnv2 analogue of `ResNet34ChainClose` — the per-block backward
  is done; what's missing is the cross-block composition (the fan-in sums + the whole-net thread).
- **`MobileNetV2RenderPC.lean`** — `mobilenetv2Forward_full_pc` (the committed **reduced 6-block**
  forward) + `mobilenetv2FwdGraphFullPC_faithful`. This is the forward to thread.
- **`MobileNetV2BackB0.lean`** — the whole-block batched backward graph (the parallel-universe proof;
  reference for the MBConv backward structure incl. the residual fan-in).

### MBConv block structure (what the per-block-type tie lemma must cover)
`expand 1×1 conv→BN→relu6 → depthwise 3×3 conv→BN→relu6 → project 1×1 conv→BN` (no relu6 after project);
a **residual skip-add when stride=1 AND in_ch=out_ch** (else no skip). So the block has 3 convs (expand,
depthwise, project) + 3 BN + 2 relu6. The new bits vs r34: (a) **relu6** masks (two kinks: `0<x ∧ x<6`)
instead of relu — check `MobileNetV2BackB0`/`MobileNetV2Close` for the relu6 backward token/mask;
(b) **depthwise** convs (covered by `mnv2_render_depthwise*_certified`, generic in cotangent);
(c) the skip is **conditional** (only stride-1 same-channel blocks) — the fan-in sum applies to those,
the others are a plain chain (no add).

### The blocker, and the honest decision
- The committed mnv2 trainer + `mobilenetv2Forward_full_pc` are the **reduced 6-block** net (the full
  17-block paper net is `MobileNetV2FullPaper.mobilenetv2ForwardPaper`, NOT committed); the whole-net
  VJP witness is a **2-block representative**.
- **For the §1a tie this is fine:** the §1a tie certifies the *committed train step* against *its* forward
  — and both are the reduced 6-block net. So **tie the reduced net** (it's honest: it ties exactly what
  trains). The "promote to the full 17-block net" + "upgrade the 2-block VJP witness" are **separate §1/§4
  concerns** (what the committed trainer *should be*), not §1a-tie blockers. Flip the §1a row to ✅ TIED
  for the reduced net, and keep the size caveat in the row text (as the existing matrix already notes the
  mnv2 reduced-net trap).

### Concrete plan
1. New file `MobileNetV2TiePoC.lean`, import `MobileNetV2ChainClose` + `MobileNetV2RenderPC` +
   `Cifar8FaithfulPoC`/`CifarBnFaithfulPoC` (for the conv/BN/dense generics).
2. `mbconvTied` (per-block-type tie, def+theorem like r34's `idblockTied`): the block's expand/depthwise/
   project conv W/b + 3 BN γ/β tied at the real block activations + the block chain cotangents (reuse
   `invresCotDc` + the depthwise/conv/BN generics). Two variants: with-skip (stride-1 same-ch) and
   without (strided / channel-change).
3. `mbconvCotIn` (fan-in sum, like r34's `idBlockCotIn`/`downBlockCotIn`): block-input cot = (skip cot if
   present) + project-back→depthwise-back→expand-back of the block-output cot, with relu6 masks + BN-backs.
4. `mnv2StemTied` (reuse `mnv2StemCot`) + the head (conv-bn-relu6 → GAP → dense) + `mnv2LossCot_den`.
5. Whole-net thread `mnv2_net_tied_certified` over the 6 blocks + stem + head, `@[irreducible]` wrappers,
   threading `mobilenetv2Forward_full_pc`. Wire to lakefile + `tests/AuditAxioms.lean`; `lake build Proofs`;
   3-axiom closure; flip the §1a mnv2 row to ✅ TIED (reduced-net, with the size caveat).

Gotcha to expect: relu6's two-kink mask, and the conditional skip — get the with-skip vs no-skip block
types right (only stride-1 same-channel blocks have the `addV` fan-in). The heartbeat/`@[irreducible]`
lesson from r34 applies if the thread is deep.

### cifar-bn (ch5) §1a tie — ✅ DONE (this session)

The cifar (ch5) tie + a BN-back at every conv, in `CifarBnTiePoC.lean`, 3-axiom clean. The cifar-BN
backward chain alternates **BN-output cotangent** `dyBnᵢ` (relu-masked — what the `bnGammaSgd`/`bnBetaSgd`
ops consume) and **conv-output cotangent** `cotCᵢ` (`bnPerChannelTensor3_grad_input` of `dyBnᵢ` — what
the `convWeightSgd`/`convBiasSgd` ops consume); the cross-pool₁ step is cifar's `cifarChainCotW2` move
(conv₃-back then maxpool₁-back) with a BN-back in front. `cifarBn_convbn_tied_certified` ties all 16
conv/BN params at the real forward + these chain cots (conv via `CifarPoC.convW_den`/`convB_den`, BN via
`CifarBnPoC.bnGamma_den`/`bnBeta_den` — all generic in the cotangent, **zero new ops/bridges**); the dense
head + loss-cot (`cifarBnLossCot_den`) + W₇ total-loss fold (`cifarBn_W7_tied_totalloss`) mirror cifar.
**§1a row flipped to ✅ TIED.**

The original concrete plan (kept for the record / as the r34+ template):

### Next session: cifar (ch5) tie — concrete plan

Tie cifar (ch5 CIFAR-CNN, no-BN, 2-scale: `(conv→relu)×2→pool→(conv→relu)×2→pool→(dense→relu)×2→dense`,
14 params: `W₁–W₄` conv + `W₅–W₇` dense + biases). **The cnn tie is the worked template — copy
`CnnFaithfulPoC`'s last three theorems** (`cnnLossCot_den`, `cnn_W5_tied_totalloss`,
`cnn_conv_tied_certified`). Forward: `cifarCnnForward` (CifarCNN.lean:45). Renderer:
`cifarTrainStepFaithfulV` (CnnRender.lean). PoC to extend: `CifarFaithfulPoC.lean` (`namespace Proofs.CifarPoC`).

**Easy — mirror cnn directly:**
1. `cifarLossCot_den` — the emitted cotangent `sub(softmaxDiv(expe(logits)), onehot)` denotes
   `softmax(cifarCnnForward … x) − onehot`. Copy `CnnPoC.cnnLossCot_den` verbatim, swapping
   `mnistCnnNoBnForward`→`cifarCnnForward`. Proof: `funext j; simp only [den, softmax]`.
2. `cifar_W7_tied_totalloss` — the dense output `W₇` folds to `∂CE/∂W₇`. Copy `cnn_W5_tied_totalloss`:
   `rw [CifarPoC.dW7_den …, mlp_output_total_loss_grad W₇ b₇ a₆ label i j]; simp only [cifarCnnForward,
   mnistLinear, Function.comp_apply]` (`a₆` = the dense activation feeding `W₇`).
3. Dense head `W₅`/`W₆`/`b₅`/`b₆`/`b₇` at the composed cotangent: instantiate the existing
   `CifarPoC.dW5/dW6/db5/db6/db7` at `g = softmax(cifarCnnForward x) − onehot` (they already thread
   the real activations from `pool`).

**The real work — cifar differs from cnn here.** `CifarPoC.convW_den`/`convB_den` are *fully generic*:
BOTH the activation `x` AND the cotangent `c` are free (cnn's `cW*_den` had the pinned
`cnnChainCotW1/2` baked in — cifar has **no** such chain). So `cifar_conv_tied_certified` (4 conv layers
`W₁–W₄`) needs, per layer:
- **`x` = the real cifar forward activation** — thread it via the SAME Vec↔Tensor3 bridge as
  `cnn_conv_tied_certified`'s `let`-block (`Tensor3.unflatten` of the flat `flatConv`/`relu`/`maxPool`
  chain). 2-scale: `conv₁,conv₂` at the outer spatial → `pool₁`; `conv₃,conv₄` at the halved spatial → `pool₂`.
- **`c` = the real cifar conv backward cotangent — THIS MUST BE BUILT** (the cifar analogue of
  `cnnChainCotW1/2` in `CnnChainClose`, which cnn got for free). Construct `cifarChainCotW1–4`:
  backward through 2 pools + 4 convs + relus + the dense-head cotangent. Resources: the whole-net
  `cifarCnn_has_vjp_at` (CifarCNN.lean:77) is the correctness reference; the backward building blocks
  are `maxPoolBackFlat`, the conv input-VJP (`conv2d_has_vjp3`), `selectPos` relu masks, and the
  `mlpCotOut`-style dense-head cotangent (already used by `dW5/dW6`). This is the "compose a whole-net
  backward" step (§2.3) — the bulk of the cifar work, and the piece r34/enet/convnext also need.

**Bonus cifar may have that cnn lacked:** check `cifarTrainStepFaithfulV`'s backward — if it already
emits `SHlo` nodes (`.convBack`/`.maxPoolBack`/`.selectPos`, like `cifar8BnTrainStepFaithfulV`) rather
than cnn's hand-written string, then the *stricter* cotangent-subgraph pins (the cifar analogue of
`MlpPoC.cot{0,1}_den`) are also in reach — removing even the per-op-SSA residual. **Verify this first**;
if true, build `cifarChainCotW1–4` as `den`s of those emitted subgraphs directly (a cleaner path than
cnn's, since cnn's backward is hand-written).

Wire: capstones → `tests/AuditAxioms.lean`; `lake build` + `lake env lean tests/AuditAxioms.lean`
(3-axiom closure, all benign); flip the §1a tie-table cifar row to ✅. No renderer/mlir change needed
for the den-level tie (as with cnn/mlp). Commit per-net; keep capstone names short (closure greps
`#print axioms` per line, wraps past ~120 cols).

### Everything else
The `*FwdGraph_faithful` / `*BackGraph_faithful` / `*_chain_certified` theorems
are a **parallel universe**: about proof-side graphs the committed `.mlir`
emitters (`*TrainStepText`, `renderBody`, `vitFwd/vitBack`) do not print and share
no code with. A bug in an emitter leaves every proof green. Agreement is
established only empirically (JAX `value_and_grad` oracle, cross-backend ULP).

### Two extra per-net traps
- **mnv2**: committed trainer is the *reduced 6-block* net (the full 17-block
  proof-render lives only in the non-committed `TestMobilenetV2TrainPC`), and the
  whole-net VJP witness is a 2-block representative — so even the parallel proof
  is representative, not the trained net.
- **vit** (UPDATED 2026-06-19 — the granularity gap is NOT a tie blocker): the OLD whole-net backward
  `vitNetBackGraph_faithful` is proven for *scalar* LN, and the emitted net uses *per-channel* `[192]`
  LN — but the **`ViTVecLN` scaling pass already models the per-channel `[192]` LN end-to-end**
  (`layerNormVec` fwd/VJP, render graph `vitFwdGraphV`/`_faithful`, param certs
  `vit_render_vecln{gamma,beta}_certified`, chain-pinned `vit_render_vecln{1,2,F}*_chain_certified`,
  chain cots `vitCot*V`). The §1a tie builds its backward from those per-op chain cots (like convnext),
  NOT from the scalar `vitNetBackGraph_faithful` — so the scalar/per-channel mismatch in the *parallel
  universe* doesn't block the tie. What remains is the convnext-style full §1 build (render + fold + tie
  + the `veclnGammaSgd` core op), still "not a printed-graph tie" until that lands.

## 2. What's needed, net by net (Edge B/A close-out)

Universal recipe, ordered by how close each net already is:

1. **Route the committed `.mlir` through `pretty(provenGraph)` + prove the printer.**
   Forward-eval of linear/mlp/cnn already does this — generalize to (a) forward-eval
   of cifar8 + all Tier-3 nets and (b) the *train-step* modules. Concretely: delete
   `*TrainStepText`/`renderBody`/`vitFwd/vitBack` and emit `renderModule(provenTrainStepGraph)`,
   or prove the hand-written string parses to the proven `SHlo` AST
   (`StableHLOParse.roundtrip` is the seed).
2. **Fold the grad/SGD/AdamW tail into the proven AST** so the *whole* train step
   is `render(provenGraph)`, not `render(fwd) ++ handwritten(tail)`. Linear is the
   pilot (see §3); finish there, then it's a template.
3. **Build whole-net backward-graph faithfulness where missing.** mlp/vit have it;
   cnn/cifar8/enet/convnext have only per-op/per-block/per-param pieces — compose
   into one `<net>BackGraph_faithful`. r34 has per-param certs + full-net forward,
   so it is the best-positioned conv net to compose end-to-end first.
4. **Per-net deltas:** mnv2 — promote `TestMobilenetV2TrainPC` (full 17-block) to
   the committed trainer and upgrade the VJP witness from 2-block to full; vit —
   either prove whole-net backward for *per-channel* LN (match emitted) or emit
   scalar LN (match proven).
5. **ℝ → Float32** (FloatBridge) on top, per architecture.

**Highest leverage:** finish linear's train-step tail (a few ops) to get one
*fully* `render(provenGraph)` train step end-to-end, then replicate. Second:
compose r34's per-param backward certs + full forward into a whole-net
`render(provenGraph)` train step and route the committed `.mlir` through it —
that converts the strongest parallel-proof net into the first genuinely faithful
conv trainer.

## 3. PoC: mnist-linear, proof-tied — `LeanMlir/Proofs/LinearFaithfulPoC.lean`

Builds clean; all three capstones close under `[propext, Classical.choice,
Quot.sound]` (`lake env lean LeanMlir/Proofs/LinearFaithfulPoC.lean`).

What it establishes:
- `poc_linear_fwd_faithful` — `den(fwdGraph W b x) = mnistLinear W b`.
- `poc_linear_fwd_is_render_of_proven_graph` — the committed `linear_fwd.mlir`
  generator `linearFwdModuleV` *is* `renderModule(fwdGraph)`. Combined with the
  above: the forward-eval bytes are end-to-end proof-tied (text = `render(graph)`
  ∧ `den(graph) = mnistLinear`). **Forward = fully faithful.**
- `poc_train_step_certified` (∀ W b x lr label) — the train step's three
  semantic outputs each denote the certified `fderiv`-derived loss-descent step:
  `%dy = ∂CE/∂logits`, `%W0n = W − lr·∂CE/∂W`, `%b0n = b − lr·(certified bias step)`.
  Bundles `lossCotGraph_isCEgrad` + `linWeightDen_is_loss_descent` +
  `linBiasDen_is_certified` into one named "the train step is the certified step"
  theorem that did not previously exist as a single statement.
- `poc_train_step_tail_certified` — **the tail fold (landed).** The param-grad +
  SGD tail gets a *structural* denotation `tailDenW`/`tailDenB` built from the
  emitted ops (`dot_general → x⊗dy`, `reduce → dy`, `multiply`, `subtract`, B=1),
  proven equal to the certified step — the tail's meaning is now derived from the
  ops it emits, not supplied.

**Wiring landed (step "A").** `LinearFaithfulPoC` is a `Proofs` lakefile root;
the four capstones are in `tests/AuditAxioms.lean` (CI three-axiom closure,
638/638 benign); and `proofs.yml` has a **Verified-render drift guard** that
regenerates `verified_mlir/linear_*` from `StableHLO.lean` and `git diff`s — so
the committed bytes `MainMnistLinearVerified` compiles can't drift from the
certified renderer. (Capstone names kept short: the closure check greps
`#print axioms` per line and Lean wraps long qualified names past ~120 cols.)

Honest residual (documented in the file header):
- **Tail `den` ⇄ MLIR text.** The cotangent prefix is genuinely `pretty(SHlo)`;
  the param-grad + SGD tail is modelled by the values `wGrad/bGrad/sgdW/sgdB`
  (proven = certified here), but that the four emitted ops
  (`dot_general`/`reduce`/`multiply`/`subtract`) compute those values is trusted
  per-op `den` modelling, not derived from a parse — same boundary the forward
  graph already sits at. **Next step:** add `SHlo` (or a small batched-tail) nodes
  for these four ops with a `den`, so the whole module is `pretty(provenGraph)`
  and Edge B closes textually.
- **B = 1.** `wGrad x dy = x ⊗ dy` (per-example); the emitted module
  batch-contracts. Mean-loss cotangent makes the batch sum the mean gradient, but
  a batched denotation is not yet modelled.
- **ℝ → Float32** handled separately (FloatBridge, Tier-1).

### iree validation gate (landed, GPU box)
`scripts/validate_linear_faithful.sh` (needs `iree-compile`; local, not CI —
GitHub's ubuntu runner has no iree/rocm) checks both halves: (a) committed
`verified_mlir/linear_*` == the proven renderer (`git diff` after regenerate),
and (b) those bytes `iree-compile` cleanly. Verified on rocm/gfx1100 (iree
3.12.0): `linear_fwd` → 11.6 KB vmfb, `linear_train_step` → 24.9 KB vmfb. So the
chain for the mnist-linear chapter is: **bytes == proven renderer (drift) →
renderer outputs == certified loss-descent step (`LinearFaithfulPoC`) → iree
accepts the bytes (compile)** — with the tail's per-op `den`⇄text still trusted.

### ✅ DONE — the tail fold landed (core refactor completed)
The plan below was executed: two `SHlo` ops (`weightSgd`, `biasSgd`) added through
all 9 sites (`SHlo`/`den`/`Tok`/`Raw`/`skel`/`toToks`/`emitTok`/`parseStack`/
`parseStack_toToks`); `linTrainStepFaithfulV` renders the whole linear train step
as `pretty` of proven nodes (cotangent shared once, then the two SGD ops);
`poc_{weightSgd,biasSgd}_den_eq` prove `den(op) = certified sgdW/sgdB` by `rfl`.
Verified: `lake build Proofs` (2249 jobs) green; `AuditAxioms` 3-axiom closure
640/640 incl. the new capstones; `roundtrip` theorem still holds (StableHLOParse
rebuilt); committed `verified_mlir/linear_train_step.mlir` regenerated from
`linTrainStepFaithfulV` and `iree-compile`s on rocm/gfx1100 (24928 B vmfb).
So the linear train step is now `render(provenGraph)` with `den = certified` end
to end — the forward and tail on identical footing. Residual: per-op `pretty`
lexing (shared with the whole suite) + ℝ→Float32. **This is the template for
steps 2–3 of §2 on every other net.** The original scoping (for reference):

### To finish linear — this is a CORE REFACTOR, not a mechanical add
(Correcting the earlier "last mechanical step" framing.) The forward is genuinely
"under `den`" because `linearFwdModuleV = renderModule(fwdGraph)` and `fwdGraph`
is an `SHlo` with `den`; the trust is the shared `pretty` printer. To put the tail
on the same footing the tail ops must become `SHlo` nodes printed by that same
`pretty` — which touches the audited core:
1. **New `SHlo` ops** — `outer` (`x⊗dy : SHlo (m*n)`), `affineUpdate` (`θ − lr·g`),
   batch-`reduce`. Each needs a case in: `SHlo`, `den` (math), `Raw`, `skel`,
   `toToks` (emit valid StableHLO — iree-validated), and the
   `StableHLOParse.parseStack_toToks` round-trip induction (+1 uniform case each —
   mechanical, but it does extend the `roundtrip` theorem the audit cites).
2. **Multi-output shared-SSA rendering** — the train step returns two values
   (`%W0n`, `%b0n`) sharing the cotangent `%dy`. `pretty` renders one `SHlo`;
   `renderModuleN` shares `%dy` but feeds it to *string* `emit`s, not `SHlo`
   sub-graphs. Rendering two `SHlo` outputs that reference the cotangent's
   *result* SSA name is the coordination the repo hand-writes the tail to avoid —
   it needs late-binding the `%dy` name into each output graph. This is the real
   work, not the op cases.
3. Then `den(module) = certified` (reuse `sgd*_descends_certified_grad`), switch
   the `#eval` writer + drift guard to the denoted module, iree-compile to confirm
   identical/valid bytes.

Risk: (1)+(2) edit `SHlo`/`den`/`Raw`/`skel`/`toToks`/`parseStack_toToks` — the
core 70-module suite + the round-trip theorem rebuild. Do it on a worktree/branch,
re-verifying `lake build Proofs` + `AuditAxioms` (3-axiom closure) + `roundtrip` +
`iree-compile` at each step. Not a blind in-place edit. This same refactor is the
template for steps 2–3 of §2 on every other net.

## 4. Wiring `MainMnistLinearVerified`, and the chapter-trainer end state

_Edge C (ℝ→Float32) is explicitly deferred to a future pass; the "verified
trainer" bar below is Edge A+B only._

### `MainMnistLinearVerified` already trains on the certified render
Data flow: `MainMnistLinearVerified` → `linearVerified.trainLinear` →
`VerifiedNet.trainLinear` reads `verified_mlir/linear_train_step.mlir`
(`VerifiedTrain.lean:168`), iree-compiles it, invokes `m.linear_train_step`. And
that committed file is written by `linearTrainStepModuleV 128 784 10 "0.00078125"`
(`StableHLO.lean:4167`) — the *same* renderer `poc_linear_train_step_certified`
is about. So the trainer is already pointed at the certified render; the link was
just unenforced/unsurfaced. `LinearFaithfulPoC.lean` now adds a **drift guard**
`#eval` that reads the committed file and asserts byte-equality with
`linearTrainStepModuleV(…)` (prints `OK`). Closed loop: trainer bytes ==
certified render (build-checked) ∧ render outputs == certified math (kernel).

Two ways to make this a first-class, enforced tie:
- **(A) Committed file + guard (decoupled, minimal).** Keep the pre-rendered
  files; promote the drift guard to CI and add the PoC capstones to
  `tests/AuditAxioms.lean` + a `Proofs` root. Runtime stays Mathlib-free. The tie
  is build-level string equality (not in-kernel) + kernel-proven render outputs.
- **(B) Inline render (self-contained, end-state).** `trainLinear` calls
  `linearTrainStepModuleV` at startup and feeds the string straight to
  iree-compile — no committed file, no drift possible, the compiled bytes *are*
  the Lean render. Cost: the runtime (`VerifiedTrain`/`VerifiedNets`) imports
  `Proofs.StableHLO`, pulling Mathlib into the trainer binary (today's committed-
  file design exists precisely to avoid that coupling).

Either way, linear is not *fully* faithful until the tail is folded into the AST
(§3 "To finish linear"); today: forward end-to-end in-kernel, train step
semantically certified per-output with the tail-op `den` trusted + bytes
CI-pinned.

### End state: top-level = one verified trainer per chapter, rest → demos
What's needed:
1. **A `VerifiedTrainer` bundle** making "verified" a *checked property*, not a
   filename: per chapter, bundle (spec `VerifiedNetSpec`, proven renderer
   `render → String`, whole-net fwd+bwd faithfulness theorem, the FFI driver, and
   a regeneration-equality guard). Ideally a structure you can't build without the
   faithfulness witness; minimally a manifest `chapter → trainer → capstone → axiom-audit`.
2. **Render pipeline closed per net** (§2 recipe): every chapter's `_train_step.mlir`
   = `render(provenGraph)` with whole-net fwd+bwd faithful. Status: linear closest;
   mlp/cnn need `*TrainStepText` replaced by a denoted renderer; cifar8/r34/enet/
   convnext need a composed whole-net backward graph; vit has it (scalar-LN).
3. **Per-net blockers to "verified":** mnv2 — promote the full 17-block
   `TestMobilenetV2TrainPC` to the committed trainer + upgrade the VJP witness from
   2-block representative to full; vit — reconcile scalar-proven vs per-channel-emitted LN.
4. **Reorg mechanics:** move the `Main*Train.lean` (MlirCodegen path) + `demos/*` +
   redundant Mains into `demos/`; keep one verified trainer per chapter at top;
   update lakefile `lean_exe` roots, README chapter table, CI, blueprint.
5. **CI faithfulness gate:** a top-level trainer ships as "verified" only when its
   render-=-certified capstone + drift guard are in the audit. Tier honestly during
   transition (`verified` vs `verified-forward` vs `in-progress`) — don't relabel a
   net "verified" before its train-step capstone lands.

## 5. Session handoff — r34 next, then the rest

_State: **8 nets fully folded** (`render(provenGraph)` with every output `den = certified`,
axiom-clean, iree-validated on rocm/gfx1100): **linear, mlp, cnn, cifar (ch5), cifar-bn (ch5),
cifar8 (8-conv), cifar8-bn, r34 (ch6, full [3,4,6,3], 146 params)**. Commits: linear/mlp
`e4d2a46`/`7ed4c2a`; then a prior run — cnn `4d6a07a` (added `convWeightSgd`/`convBiasSgd`),
cifar `31dedf8` (reuse), cifar-bn `805ff04` (added `bnGammaSgd`/`bnBetaSgd`), cifar8 `1957930`
(reuse + generic dense lemmas), cifar8-bn `1441754` (reuse, **zero new proof**), CI scorecard rows
`4dc953a`; r34 this run (added `convStridedWeightSgd`/`convStridedBiasSgd` + ResNet34Render.lean +
ResNet34FaithfulPoC.lean). The core SGD-op kit is now complete + proven generic: `weightSgd`/
`biasSgd` (dense), `convWeightSgd`/`convBiasSgd` (stride-1 conv), `convStridedWeightSgd`/
`convStridedBiasSgd` (stride-2 conv, 7×7 stem + 3×3 down/proj), `bnGammaSgd`/`bnBetaSgd`
(per-channel BN). Blueprint intentionally NOT touched. This section is the recipe + per-net plan
for the rest; **mnv2 is next** (it has a blocker — see its bullet below)._

_**cnn close notes (the conv template — reuse for cifar-bn/r34):** the dense head is a
3-layer MLP, so its cotangents are literally IR `mlpCotOut0/1` and its `den`s close via
the M2 `weight_grad_bridge`/`bias_grad_bridge` (copy `MlpFaithfulPoC`). The conv layers
needed two new core SGD ops, cloned through all 9 sites from `weightSgd`/`biasSgd`:
`convWeightSgd` (`den = flatten(W − lr·conv2d_weight_grad(b,x)·dy)`, emit = the
transpose-trick conv + SGD wrap) and `convBiasSgd` (`den = b − lr·conv2d_bias_grad(W,x)·dy`,
emit = reduce[0,2,3] + SGD). Their `den` reduces by `rfl` to the LHS of the existing
`cnn_render_conv{W,b}{1,2}_chain_certified` (CnnChainClose), so `CnnPoC.{cW,cb}{1,2}_den`
are one-line delegations. Gotcha hit: the `Back.cotangent` dense rows (`dW5`/`db5`) need a
trailing `rfl` after the bridge `rw` to collapse `Back.cotangent.denote dy = dy`._

### The proven recipe (what worked for linear + mlp + cnn)
For chapter net `N` with committed `verified_mlir/N_train_step.mlir`:
1. **Find the proven param-grad certs.** Each net has `N_render_*_certified` /
   `N_layer*_*_grad_bridge` (analogs of `linWeightDen_is_loss_descent`):
   `θ − lr·emit{Weight,Bias}Grad(activation, cotangent) = θ − lr·(certified pdiv
   Jacobian · cotangent)`. These already exist for linear/mlp/cnn/cifar/r34. Reuse them.
2. **Express the train step as `SHlo` nodes.** Forward via `denseF`/`reluF`/`bnF`/
   conv ops (already proof-rendered in `*TrainStepStructured`); backward chain via
   existing `dotOut`/`selectPos`/`convBack`/`maxPoolBack`/`bnBack`; param updates via
   `weightSgd`/`biasSgd` (dense) — **add new SGD ops only for param grads the existing
   ones can't express** (see per-net below).
3. **`*FaithfulPoC.lean`:** for each emitted param op, prove `den(op) = certified`
   by `have step : den(op) = θ − lr·emit*Grad(...) := by simp [den, emit*, Mat.outer,
   Mat.flatten, Equiv.symm_apply_apply, ...]` then `rw [step, N_render_*_certified ...]`.
   Cotangent subgraph lemmas: `den(selectPos p (dotOut W e)) = (chain cotangent).denote`
   (cf. `MlpPoC.cot{1,0}_den`).
4. **`*TrainStepFaithfulV` renderer** (in `*Render.lean`): forward + cotangent chain
   (shared once) + the param SGD ops, all via `pretty`, multi-output. Switch the
   `#eval` that writes `verified_mlir/N_train_step.mlir` to it (move it out of
   `StableHLO.lean` if the renderer lives in `*Render.lean`).
5. **Wire + verify:** add `*FaithfulPoC` to `lakefile.lean` `Proofs` roots + the
   capstones to `tests/AuditAxioms.lean`; `lake build Proofs`; re-run the closure
   (must stay all-benign); `iree-compile` the regenerated `.mlir` (rocm/gfx1100);
   add the net's row to the CI scorecard in `proofs.yml`; update the §1 matrix here.

### Gotchas (learned the hard way)
- **Capstone names must be short.** The closure check greps `#print axioms` output
  per line; Lean wraps past ~120 cols, splitting the benign triple across lines and
  false-failing. Keep `Proofs.<NS>.<name>` short (e.g. `Proofs.MlpPoC.W0_den_certified`).
- **Render is value-independent; `den` needs real values.** `skel` erases `ℝ`/operand
  values, so the renderer passes placeholders (`fun _ => 0`, `lr := 0`) and stays
  computable for `#eval`; the `den` theorems use the real values. A bare `lr : ℝ` field
  is fine to `#eval` only as the placeholder `(0:ℝ)` — don't pass a real `lr` literal to
  the renderer (noncomputable).
- **`den(op) = certified` is usually `rfl`-close** once `wGrad`/`Mat.outer`/`bGrad`
  unfold; if `simp` stalls on the `Mat.flatten ∘ finProdFinEquiv` step use
  `simp [Mat.flatten, Equiv.symm_apply_apply]`.
- **Core `SHlo` extension is contained**: only `den` + `skel` match `SHlo`
  constructors; a new op also needs `Tok`/`Raw` ctors, `toToks`, `emitTok` (the MLIR
  text — copy the committed hand-written op text), and `parseStack` + the one-line
  `parseStack_toToks` case (keeps `roundtrip` true). `weightSgd`/`biasSgd` (in
  `StableHLO.lean`) are the worked template.
- iree-compile lives at `/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-compile`
  (rocm/gfx1100); the drift gate is `scripts/validate_linear_faithful.sh` (generalize
  per net). CI's ubuntu runner can't iree-compile — keep that gate local.

### Per-net plan
- **cnn (2d) — ✅ DONE (this session).** Added the 2 core ops `convWeightSgd`/`convBiasSgd`
  (9 sites each, `roundtrip` extended); `CnnFaithfulPoC.lean` proves all 10 param outputs'
  `den = certified` (conv via `cnn_render_conv{W,b}{1,2}_chain_certified`, dense head via the
  M2 bridges); `cnnTrainStepFaithfulV` (CnnRender.lean) renders the whole step as
  `pretty(provenGraph)` and now writes `verified_mlir/cnn_train_step.mlir` (iree-compiles,
  121 KB vmfb). Capstones in `tests/AuditAxioms.lean` (3-axiom closure, all benign);
  scorecard row flipped to ✅. (`cnnTrainStepText` kept in StableHLO.lean for reference.)
- **cifar (ch5, no-BN) — ✅ DONE (this session).** Reused the cnn conv ops with ZERO new
  core ops: `CifarFaithfulPoC.lean` has generic `conv{W,B}_den` (cover all 4 conv layers)
  + the 3-dense head (`{dW,db}{5,6,7}_den`, M2 bridges + `mlpCotOut0/1`); `cifarTrainStepFaithfulV`
  (CnnRender.lean) renders the whole 2-scale step as `pretty(provenGraph)` and writes
  `verified_mlir/cifar_train_step.mlir` (iree-compiles, 186 KB vmfb). Capstones in the
  3-axiom closure (all benign); scorecard row ✅.
- **cifar-bn (ch5) — ✅ DONE (this session).** Added the 2 core ops `bnGammaSgd`/`bnBetaSgd`
  (per-channel BN scale/shift grads; `den` bridges `oc·h·w↔oc·m` via `reassocFwd`, closed by
  `cifar_bn_render_{gamma,beta}_certified`). Conv layers + dense head reuse the cifar fold.
  `CifarBnFaithfulPoC.lean` + `cifarBnTrainStepFaithfulV` (CnnRender.lean) writes
  `verified_mlir/cifar_bn_train_step.mlir` (iree-compiles, 259 KB vmfb). Capstones in the
  closure (all benign). **`bnPerChannel_grad_{gamma,beta}` moved CifarBnClose→PerChannelBN**
  so `den` can reference them upstream. Unblocks cifar8-bn (same BN ops).
- **cifar8 (8-conv, no-BN) — ✅ DONE (this session).** Pure reuse, ZERO new ops:
  `cifar8TrainStepFaithfulV` (CnnRender.lean, 4 stages, 22 params) writes
  `verified_mlir/cifar8_train_step.mlir` (iree-compiles, 271 KB vmfb); `Cifar8FaithfulPoC.lean`
  adds the generic `dense{W,B}_den` (conv reuses `CifarPoC` generics). Note: the deep do-block
  needs `set_option maxRecDepth 4000 in` (≈70 `pretty` binds).
- **cifar8-bn — ✅ DONE (this session).** Pure reuse, NO new ops AND NO new proof:
  `cifar8BnTrainStepFaithfulV` (CnnRender.lean, 4 stages, 38 params, `maxRecDepth 8000`) writes
  `verified_mlir/cifar8_bn_train_step.mlir` (iree-compiles, 393 KB vmfb); every param's
  `den` = certified by the existing generics (conv `CifarPoC`, BN `CifarBnPoC`, dense `Cifar8PoC`).
  This is the payoff of building generic op-lemmas: a 38-param net folds with zero new theorems.
- **r34 — ✅ DONE (this run).** The concrete plan below was executed. Added **2 new core ops**
  `convStridedWeightSgd`/`convStridedBiasSgd` (clone of `convWeightSgd`/`convBiasSgd`: weight gets
  the full 9-site treatment with the zero-upsample weight-grad emit; **bias `skel` aliases
  `convBiasSgd`** since the bias grad is stride-independent — same `reduce` text, only `den`
  differs). `flatConvStride2_bias_grad_has_vjp` + `conv2d_bias_differentiable` **relocated**
  MobileNetV2Close→StridedConv so the bias op's `den` can reference them upstream (same pattern as
  the per-channel BN grads). `ResNet34FaithfulPoC.lean`: `convStrided{W,B}_den` are one-line
  delegations to `mnv2_render_stem_conv{W,b}_certified` (generic kH/kW → covers the 7×7 stem AND
  every 3×3 strided down/proj); the 142 other params reuse the cifar conv/BN/dense generics (no
  other new theorems — the cifar8-bn lesson). `resnet34TrainStepFaithfulV` (ResNet34Render.lean,
  `maxRecDepth 1000000`, factored into `idFwd`/`downFwd`/`idBackSgd`/`downBackSgd` block helpers
  returning `String` records) renders the whole 146-param step and writes
  `verified_mlir/resnet34_train_step.mlir` (iree-compiles, 537 KB vmfb). Capstones in the 3-axiom
  closure (all benign); scorecard row ✅. Gotcha hit: emitter reduce-regions use literal block-args
  `%sa`/`%sb`/`%sc`/`%ss`, so the stem-bias param had to be renamed `%sb`→`%sbi` (MLIR forbids a
  region block-arg shadowing a func arg; func args are positional in the vmfb so the rename is free).
- **mnv2 — has a blocker.** The committed trainer is the *reduced 6-block* net; the full
  17-block proof-render is `TestMobilenetV2TrainPC` (not committed), and the whole-net VJP
  witness is only a 2-block representative. Promote the full net + upgrade the VJP witness
  BEFORE folding, else the fold certifies a non-representative net.
- **enet — IN PROGRESS (Item B emit + param-SGD DONE, 5 commits, 2026-06-18).** CORRECTED diagnosis: the
  blocker was NOT "no whole-net backward graph" — the backward math exists (`efficientnetFwdGraphB_full_faithful`,
  `efficientnetForwardB_full_has_vjp`, den-faithful per-block backward bricks). The real blocker was the
  **deferred batched MLIR emit (Item B: the `.batched` `emitTok` stub)** + the missing batched param-SGD ops,
  both now done + iree-validated on gfx1100. NEXT: the 262-param renderer `efficientnetTrainStepFaithfulV`,
  then §1 fold (BN/dense/conv ≈ 1-line cert delegations; depthwise/stem need the Σ_n batch-sum bridge) + §1a
  tie (swish + SE gate). See the IN-PROGRESS banner at the top for full detail.
- **convnext — blocker.** Full forward faithful but **no whole-net backward graph** (only per-block
  `cnxBlockBodyBackGraph`). Compose a whole-net backward + per-param certs first, then fold. (NB: the enet
  experience suggests checking whether convnext's true blocker is also the batched/render emit rather than
  the backward math — audit before assuming.)
- **vit — granularity blocker.** Richest (`vitNetBackGraph_faithful` + full
  `vit_render_*_chain_certified`) BUT proven for *scalar* LayerNorm while the emitted net
  uses *per-channel* `[D]` LN. Reconcile (prove per-channel-LN whole-net backward, or emit
  scalar LN) before the fold lands honestly.

### r34 — concrete plan (✅ EXECUTED this run; kept for the record / as the template for mnv2+)

r34 is the [3,4,6,3] ResNet-34 (146 params): a 3×3/s2 stem, 16 residual blocks (each
`conv→BN→relu→conv→BN` + skip-add → relu; downsample blocks add a strided projection skip),
global-average-pool, final dense. It's the **hardest fold** (strided convs + residual
fan-out/in + sheer size), but every math cert already exists. Follow the proven recipe:

**1. Two new core ops** (clone the `convWeightSgd`/`convBiasSgd` 9-site template, but for
   stride-2): `convStridedWeightSgd` and `convStridedBiasSgd`.
   - `den` for the weight op: `flatten W − lr·(flatConvStride2_weight_grad_has_vjp b x).backward (flatten W) dy`
     (the stride-2 analogue; the function lives in `StridedConv.lean:141`). Closes by `rfl`
     to the LHS of the **already-proven** generic `r34_render_downConvW_certified`
     (ResNet34Close.lean:89) — so the PoC theorem is a one-line delegation, exactly like
     `CifarPoC.convW_den`. Bias op ⇄ `flatConvStride2_bias_grad_has_vjp` / `r34_render_downConvb_certified`.
   - `emit`: the strided convWGrad text — reshape→transpose→`convolution(window_strides=[2,2]…)`
     →transpose + SGD wrap. Crib the exact op text from the committed `resnet34_train_step.mlir`
     / the r34 string emitter (`TestResnet34Train`, the `renderBody`), and from the existing
     `flatConvStridedF`/`convStridedBack` `emitTok` cases (the stride-2 conv text is already there).
   - The 3×3/s2 **stem** uses `r34_render_stem_convW/b_certified` (ResNet34Close.lean:41) — check
     whether it's the same strided op or stride-1 (`r34_render_blockConvW` is the stride-1 3×3
     block conv → reuse `convWeightSgd`). Wire the stem to whichever matches.

**2. Reuse everything else** (no new ops): `convWeightSgd`/`convBiasSgd` (stride-1 3×3 block
   convs), `bnGammaSgd`/`bnBetaSgd` (per-channel BN), `weightSgd`/`biasSgd` (final dense),
   and the backward ops `convBack`/`convStridedBack`/`bnPerChannelBack`/`gapBack`. **GAP**: the
   `gapF`/`gapBack` ops exist; the head is `gap → dense`.

**3. The residual wrinkle (the real new work in the renderer).** A skip-add `addV(F(x), skip)`
   sends its output cotangent to BOTH branches; where two paths reconverge the cotangents **sum**.
   So the backward renderer can't be a single linear chain (cnn/cifar were) — at each block input
   the cotangent is `(F-branch backward) + (skip-branch backward)`. Render the per-branch cotangents
   then an explicit `addV`-style sum node for the merge (the forward `addV` op's backward = copy;
   model the sum with the existing add/`.addV` emit). Get this right per block type (identity skip =
   pass-through; downsample skip = strided-proj-conv backward). This is the part to design carefully.

**4. PoC** (`ResNet34FaithfulPoC.lean`): conv via `CifarPoC.conv{W,B}_den` (stride-1) + the 2 new
   strided generics (`convStridedW_den`/`convStridedB_den` ← `r34_render_downConv{W,b}_certified`),
   BN via `CifarBnPoC.bn{Gamma,Beta}_den`, dense via `Cifar8PoC.dense{W,B}_den`. Likely **no other
   new theorems** — the 146 params are all instances of these generics (the cifar8-bn lesson).

**5. Renderer** (`resnet34TrainStepFaithfulV` in a new `ResNet34Render.lean` or CnnRender.lean):
   the biggest yet — 16 blocks, will need `set_option maxRecDepth` well above 8000 and almost
   certainly the build-the-module-inside-`StateM` pattern. Crib the forward block structure from
   `resnet34FwdGraphFullPC` and the committed `resnet34_train_step.mlir`. Switch its `#eval` to
   write `verified_mlir/resnet34_train_step.mlir`; iree-validate; add the scorecard row
   (`r34 (ch6)` → `$(faithful …)`) and flip the §1 matrix row.

Gotchas already learned: `set_option maxRecDepth N in` goes **before** the docstring, not between
docstring and `def`; `let inner : String := go.run' 0` needs the explicit `: String`.

### Commands
```
lake build Proofs                                  # whole suite (≈ rebuilds on core edits)
lake env lean tests/AuditAxioms.lean               # 3-axiom closure (must be all-benign)
lake env lean LeanMlir/Proofs/<Net>Render.lean     # regenerate that net's committed .mlir
IREE=/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-compile
$IREE verified_mlir/<net>_train_step.mlir --iree-hal-target-backends=rocm \
  --iree-rocm-target=gfx1100 --iree-codegen-llvmgpu-use-reduction-vector-distribution=false -o /tmp/x.vmfb
```

