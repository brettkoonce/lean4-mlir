# Verified-faithful sweep: tying the emitted `.mlir` to the proofs

_Status note, 2026-06-17._ Audit + PoC. Question driving this doc: for each
`Main*Verified*.lean` trainer, is the MLIR it actually compiles **proof-tied**
to the `Proofs/` math, or only validated numerically? And what would close the
gap, net by net?

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
| **mnv2** | committed `.mlir` | ❌ hand-written | ✅ **CLOSED (§1 fold, reduced 6-block)** — `mnv2TrainStepFaithfulV` (MobileNetV2Render.lean) renders the whole reduced-6-block train step (82 params) as `pretty(provenGraph)`; each param `den = certified` via `MobileNetV2FaithfulPoC` — **4 new core ops** `depthwise{,Strided}{Weight,Bias}Sgd` (StableHLO.lean, the per-channel `batch_group_count=c` transpose-trick weight + `convBiasSgd`-aliased bias), expand/project conv via `CifarPoC.conv{W,B}_den`, BN via `CifarBnPoC.bn{Gamma,Beta}_den`, dense via `Cifar8PoC.dense{W,B}_den`; committed bytes iree-compile on rocm/gfx1100 (789 KB vmfb), drop-in positional param layout | **§1a tie remains** (commit 3); the reduced-net + 2-block-VJP-witness are separate §1/§4 concerns (committed = reduced 6-block, NOT the full 17-block `mobilenetv2ForwardPaper`) |
| **enet** | committed `.mlir` | ❌ hand-written | ❌ `renderBody` hand-written | `efficientnetFwdGraphB_full_faithful` (full 16 MBConv); backward per-block, **no whole-net** | no whole-net backward; emitted untied |
| **convnext** | committed `.mlir` | ❌ hand-written | ❌ `renderBody` hand-written | `convNextFwdGraphT_faithful` (full [3,3,9,3]); backward per-block, **no whole-net** | no whole-net backward; emitted untied |
| **vit** | committed `.mlir` | ❌ hand-written (`vitFwd`) | ❌ `vitBack` hand-written | **richest**: `vitFwdGraphKMHV_faithful` + whole-net `vitNetBackGraph_faithful` + full per-param `vit_render_*_chain_certified` | emitted untied **+ granularity gap**: whole-net backward proven for *scalar* LN, emitted uses *per-channel* `[192]` LN |

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
| mnv2 / enet / convnext / vit | — (train-step fold WIP) | — | — |

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
1. _**Full §1 CLOSE** — `mnv2TrainStepFaithfulVPaper` (17-block SGD renderer): reuse the per-block emitters +
   ADD a no-expand variant for b1 (t=1, `IVWNoExp`: depthwise→project, no expand); assemble the
   `[t,c,n,s]` schedule (stem 3→32; stages 16/24/32/64/96/160/320; head 320→1280; dense 1280→10; 17 blocks).
   Crib the schedule from the TEST-ONLY `tests/TestMobilenetV2TrainPC.lean` (which already renders the full
   net for AdamW — forward via `pretty(mobilenetv2FwdGraphFullPC)`, hand-emitted tail). Write the committed
   full SGD `.mlir` + iree-validate. ~214 param tensors (vs reduced 82; +132 = 11 extra blocks ×12)._
2. _**Full §1 fold (den)** — `MobileNetV2FaithfulPoCPaper`: den=certified for all 214 params (generic-lemma
   instantiation; the depthwise den lemmas already exist). Wire to lakefile + AuditAxioms._
3. _**Full §1a TIE** — `MobileNetV2TiePoCPaper`: whole-net thread of `mobilenetv2ForwardPaper` (17 blocks),
   per-block-type tie lemmas (no-expand/skip/strided) + fan-in cots + `@[irreducible]` wrappers (the r34
   heartbeat lesson, MORE acute at 17 blocks). Reuse `MobileNetV2ChainClose` cotangents._
4. _**Trainer swap** — point `MainMobilenetV2Verified`/`VerifiedTrain` at the full SGD `.mlir` (currently
   reads the reduced 6-block `mobilenetv2_train_step.mlir`); validate it trains (the map flagged a
   data-loader bug blocking the swap — investigate). Reduced renderer/PoC → demo/stepping-stone._
5. _**VJP witness upgrade (separate §4 track, deferrable)** — `Mnv2Live` (`MobileNetV2JacobianSeal`) seals the
   whole-net nonzero-Jacobian only at a **2-block structural representative** (1-ch, 2×2, sealed at input 0);
   upgrade to the full 17-block net. The hardest math; does NOT block the §1 fold or §1a tie (those are
   den=certified at the real cotangent, independent of the nonzero-Jacobian guarantee)._

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
- **vit**: whole-net backward (`vitNetBackGraph_faithful`) is proven for *scalar*
  LayerNorm; the emitted net uses *per-channel* `[192]` LN. The per-param
  `vit_render_*_chain_certified` cover the per-channel render chain (per param),
  but not as a printed-graph tie.

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
- **enet / convnext — blocker.** Full forward faithful but **no whole-net backward graph**
  (only per-block `mbconvBodyBackGraph`/`cnxBlockBodyBackGraph`). Compose a whole-net
  backward + per-param certs first, then fold.
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

