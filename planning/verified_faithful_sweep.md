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
| **mnist-mlp (1d)** | committed `.mlir` | ✅ `mlpFwdModuleV = renderModule(mlpFwdGraph)`, `mlpFwdGraph_faithful` | ❌ `mlpTrainStepText` hand-written string | `mlpFwdGraph_faithful`, `mlpBackGraph_faithful` (whole-net, smooth-pt) | back graph proven but not the emitted bytes |
| **mnist-cnn (2d)** | committed `.mlir` | ✅ `cnnFwdModuleV = renderModule(cnnFwdGraph)`, `cnnFwdGraph_faithful` | ❌ `cnnTrainStepText` hand-written | `cnnFwdGraph_faithful`; backward per-op only | no whole-net back; emitted bytes untied |
| **cifar8 / cifar8-bn** | committed `.mlir` | ❌ `cifar8FwdText` hand-written | ❌ `cifar8TrainStepText` hand-written | `cifar8{Bn}FwdGraph_faithful` (fwd, full dims); backward = per-param `cifar8_render_*_chain_certified` | Tier-2: fwd proven (parallel), backward per-param |
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

