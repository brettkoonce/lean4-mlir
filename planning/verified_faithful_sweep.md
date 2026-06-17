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
| **cifar8 (8-conv, no-BN)** | committed `.mlir` | ❌ `cifar8FwdText` hand-written | ✅ **CLOSED** — `cifar8TrainStepFaithfulV` (CnnRender.lean) renders the whole 4-stage train step (8 conv + 3 dense, 22 params) as `pretty(provenGraph)`; conv via `CifarPoC.conv{W,B}_den` (generic), dense via the new generic `Cifar8PoC.dense{W,B}_den` | **NO new core ops** (pure reuse); committed bytes iree-compile on rocm/gfx1100 (271 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + ℝ→Float32) |
| **cifar8-bn** | committed `.mlir` | ❌ `cifar8BnFwdText` hand-written | ✅ **CLOSED** — `cifar8BnTrainStepFaithfulV` (CnnRender.lean) renders the whole BN train step (8 conv + 8 BN + 3 dense, 38 params) as `pretty(provenGraph)`; **no new ops, NO new proof** — every output's `den` = certified by the existing generics (`CifarPoC.conv{W,B}_den`, `CifarBnPoC.bn{Gamma,Beta}_den`, `Cifar8PoC.dense{W,B}_den`) | committed bytes iree-compile on rocm/gfx1100 (393 KB vmfb) | — (per-op `pretty` lexing + cotangent-subgraph⇄SHlo pin + BN `0<ε` + ℝ→Float32) |
| **r34** | committed `.mlir` | ❌ hand-written (`TestResnet34Fwd`) | ❌ `renderBody` hand-written (`TestResnet34Train`) | `resnet34FwdGraphFullPC_faithful` (**full 34-layer, 146 params**); backward per-param `r34_render_*_chain_certified` | strongest parallel proofs, zero tie to emitted bytes |
| **mnv2** | committed `.mlir` | ❌ hand-written | ❌ hand-written, **reduced 6-block net** | `mobilenetv2FwdGraphFullPC_faithful` (full 17-block); **whole-net VJP witness only 2-block representative** | double gap: committed net ≠ proven full net; VJP representative |
| **enet** | committed `.mlir` | ❌ hand-written | ❌ `renderBody` hand-written | `efficientnetFwdGraphB_full_faithful` (full 16 MBConv); backward per-block, **no whole-net** | no whole-net backward; emitted untied |
| **convnext** | committed `.mlir` | ❌ hand-written | ❌ `renderBody` hand-written | `convNextFwdGraphT_faithful` (full [3,3,9,3]); backward per-block, **no whole-net** | no whole-net backward; emitted untied |
| **vit** | committed `.mlir` | ❌ hand-written (`vitFwd`) | ❌ `vitBack` hand-written | **richest**: `vitFwdGraphKMHV_faithful` + whole-net `vitNetBackGraph_faithful` + full per-param `vit_render_*_chain_certified` | emitted untied **+ granularity gap**: whole-net backward proven for *scalar* LN, emitted uses *per-channel* `[192]` LN |

### What is genuinely proof-tied to emitted bytes today
- **Forward-eval modules of linear, mlp, cnn** — `<slug>_fwd.mlir` literally *is*
  `renderModule(provenGraph)` with a `den`-faithfulness theorem about that exact text.
- **Closest train step: linear** — `_train_step.mlir`'s forward+cotangent prefix
  is `pretty(lossCotGraph)`; only the param-grad/SGD tail is hand-written.

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

_State: **7 nets fully folded** (`render(provenGraph)` with every output `den = certified`,
axiom-clean, iree-validated on rocm/gfx1100): **linear, mlp, cnn, cifar (ch5), cifar-bn (ch5),
cifar8 (8-conv), cifar8-bn**. Commits: linear/mlp `e4d2a46`/`7ed4c2a`; then this run —
cnn `4d6a07a` (added `convWeightSgd`/`convBiasSgd`), cifar `31dedf8` (reuse), cifar-bn
`805ff04` (added `bnGammaSgd`/`bnBetaSgd`), cifar8 `1957930` (reuse + generic dense lemmas),
cifar8-bn `1441754` (reuse, **zero new proof**), CI scorecard rows `4dc953a`. The core
SGD-op kit is now complete + proven generic: `weightSgd`/`biasSgd` (dense),
`convWeightSgd`/`convBiasSgd` (stride-1 conv), `bnGammaSgd`/`bnBetaSgd` (per-channel BN).
Blueprint intentionally NOT touched. This section is the recipe + per-net plan for the rest;
**r34 is next** (see the dedicated subsection below)._

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
- **r34 — NEXT (see the dedicated plan below).** Strongest proof side: full-depth forward
  faithful (`resnet34FwdGraphFullPC_faithful`, 146 params) + per-layer backward
  `r34_render_*_chain_certified`. Needs **2 new core ops** (strided-conv weight/bias grad) +
  the big residual-block renderer. All the math certs already exist.
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

### Next session: r34 — concrete plan

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

