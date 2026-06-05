# validation_sweep.md — putting "verified codegen" into chapters 3–10

**Purpose.** A handoff plan (for a clean session) to add a *verified-codegen*
section to each architecture chapter of the book: explainer prose + **one
real emitted-MLIR listing of a key block** + the proven-vs-trusted ledger and
the GPU validation number. The verification, the renderers, and the artifacts
already exist (see "What's already built"); this is mostly a **writing pass**
plus ~3 small key-block builds.

Predecessor doc: `planning/verified_codegen.md` (the design + results of the
codegen pipeline itself). This doc is specifically the **book sweep**.

---

## The thesis (why this pattern, vs the monolith)

The production trainer emits the whole train step as one ~7.5k-line hand-SSA'd
string (`generateTrainStep`); the proof↔code link there is a *comment*. The
`IRPrint` pattern is the opposite: a small **parallel pipeline** where the
emitted StableHLO **is** `print(proof-backed IR)` by construction. We do not
retarget or replace the monolith — we stand a *verified* path alongside it
(CompCert did not rewrite GCC). For the book, each chapter shows that its
gradient code is the rendering of a proven derivative, not a description of one.

**Why it's tractable (the lesson worth recording).** This sweep is "easy" only
because the foundations were built in the right order:

- the VJP library is **per-op and generic** (`HasVJP`/`HasVJPAt`, `vjp_comp_at`,
  the `pdiv` lemmas) — proven *before* any codegen, parameterized over dims;
- the **whole-net VJPs** are already proven (`mlp_has_vjp_at`,
  `cnn_has_vjp_at`, `vit_full_has_vjp`, the MobileNet/ConvNeXt/EfficientNet
  capstones), 3-axiom clean;
- so codegen carries **no new proof obligation** — it is render + numerically
  validate. Every architecture's ops were proven once; the sweep reuses them.

That is the whole reason a deep ResNet train step was a focused engineering
build and not a research project. Build the math foundation per-op and generic,
and the codegen + book sweep become mechanical.

---

## The three pieces (the pattern each chapter cites)

1. **Denoted IR + bridge.** `LeanMlir/Proofs/IR.lean`: `Back`/`Fwd`/`Back3`
   ASTs with a denotation `⟦·⟧` into the proofs' own `Vec`/`Tensor3` type, and
   bridge theorems `⟦emitted graph⟧ = proven backward/forward` (e.g.
   `mlp_whole_bridge`, `weight_grad_bridge`, `lossCot_bridge`,
   `conv_back_bridge_1to2`, `bn_back_bridge`, `softmax_back_bridge`, `denote_subst`).
2. **Computable printer.** `LeanMlir/Proofs/IRPrint.lean`: AST mirrors
   (`Hlo`/`HloF`) + renderers (`convOp`, `renderLN/renderLNBack`, `renderDense`,
   `renderGelu`, `renderSoftmax`, `renderResF/renderResFBack`,
   `renderConvWGrad`, `renderBNParamGrad`, `renderGAP`, …) + per-net `*Module`
   functions and the `resTower*`/`resnetTrainStep` generators. The emitted text
   is `print(IR)` by construction.
3. **Execution oracle.** `LeanMlir/Proofs/check_ir_codegen.py`: regenerates the
   `.mlir` from `IRPrint`, compiles with IREE (`llvm-cpu` **and** `rocm`), runs,
   and diffs against an independent NumPy reference. 32 checks today; CPU is the
   gate, GPU (rocm/gfx1100, Radeon RX 7900 XTX) confirms it runs on hardware.

**The ledger (the standard claim, stated honestly in every chapter):**
- *Proven* (Lean, axioms `propext, Classical.choice, Quot.sound`): `⟦IR⟧ = proven math`.
- *By construction*: emitted StableHLO = the rendering of that IR.
- *Trusted* (validated numerically, not proven): the printer faithfully renders
  `⟦·⟧` (the R4 surface — would need a formal StableHLO **text** semantics),
  IREE's lowering, and `float32 ≈ ℝ`. Validated end-to-end to ~1e-7 on GPU.

---

## Editorial rule: one key block per chapter, full net to go deeper

Each chapter shows **one representative block's emitted MLIR** (boxed listing +
the bridge pairing) — *not* the whole net, to avoid overwhelming the reader.
The **full net is the "go deeper" artifact** (in the repo + the oracle), built
where it adds value. Chapter-3 format is the template (see the `mlp_back`
listing + "read it against the graph" pairing).

---

## Per-chapter plan

`✅` = key block already rendered + GPU-validated (oracle name + GPU max-err).
Full-net column: the deeper artifact (built, or an optional later build).

| ch | architecture | key block (in-chapter listing) | status | full-net artifact |
|----|---|---|---|---|
| 3 | MLP | `mlp_back` — dense→relu VJP chain (`dot_general` + `compare/select`) | ✅ 2.4e-7 | `mlp_train_step` ✅ 1.2e-7 |
| 4 | CNN (2D) | `conv_back` — reversed-kernel conv VJP (transpose+reverse+`convolution`) | ✅ 1.9e-6 | `cnn_train_step` ✅ 1.2e-7 |
| 5 | CIFAR + BatchNorm | `bn_back` — 3-term rank-1 BN VJP (reduce/broadcast) | ✅ 6e-8 | (cbr block; CNN/ResNet exemplars cover) |
| 6 | ResNet-34 | `res_back` — residual block backward, the `dx = dF + dadd` fan-in | ✅ 2.4e-7 | `resnet_train_step` ✅ 1.2e-7; `res_tower` (16 blocks) ✅ 2.5e-5 |
| 7 | MobileNetV2 | inverted residual (1×1 → depthwise → 1×1 + residual; relu6) | **build** | optional |
| 8 | EfficientNet | MBConv (depthwise + SE + swish) | **build** | optional |
| 9 | ConvNeXt | ConvNeXt block (7×7 depthwise → LN → 1×1 → gelu → 1×1 → layerScale + residual) | **build** (+ trivial `layerScale` render) | optional |
| 10 | Vision Transformer | `attn_back` (SDPA + 3-way QKV fan-in) or `vit_back` (full block) | ✅ 3e-7 / 6e-7 | `vit` transformer block ✅ |

**To-build blocks: ch 7, 8, 9** — each reuses only already-validated ops
(`dw_fwd/back`, `se_fwd/back`, `swish`, `sigmoid`, `relu6`, `gelu`, BN/LN,
`conv`, `residual`); no new proofs. `layerScale` (ch9) is a trivial `dy ⊙ γ`
render. Estimate ~½ day each (same shape as the ResNet block build).

---

## The section template (4 beats — fill per chapter)

Mirrors the chapter-3 section now drafted in `content.tex`:

1. **What is already proven** — the chapter's forward map + its proven VJP +
   param-grad + loss theorems; "3 axioms, no sorry / native_decide."
2. **The gap, and how we close it** — the comment→theorem move; define the
   block's `Back` graph; state its bridge `⟦emit⟧ = proven`; **insert the boxed
   MLIR listing**; "read it against the graph" pairing each op to its IR node.
3. **Proven versus trusted** — the ledger (bullets), then the validation number
   ("updates N params to within X of an independent NumPy reference on the GPU").
4. **One honest caveat** — smooth-point for the kinked ops (relu/relu6/maxpool/
   BN); representative scale; "the same three pieces carry to the other chapters."

LaTeX notes: portable `[\![\cdot]\!]` brackets (no `stmaryrd`); `listings` with
`breaklines` for the MLIR box (`\scriptsize\ttfamily`, `frame=single`).

---

## What's already built (inventory)

- **Whole nets, end-to-end + GPU-validated:** MLP train step, CNN train step,
  ViT transformer block, ResNet train step (+ 16-block tower). All in
  `IRPrint.lean`, checked by `check_ir_codegen.py`.
- **Every op proof-backed + validated:** dense, relu, conv2d (+weight grad via
  the transpose trick), depthwise conv, maxpool, flat-BatchNorm/LayerNorm,
  softmax, scaled-dot-product attention (dQ/dK/dV), softmax-CE loss,
  gelu/swish/sigmoid/relu6, residual, squeeze-excite, global-avg-pool, GAP.
- **Audit:** `tests/AuditAxioms.lean` — the bridge theorems are in the 3-axiom
  gate (107 decls; CI re-checks the closure).
- **State:** all codegen/oracle pushed (`origin/main` @ `0492995`).
  `blueprint/src/content.tex` has the ch3 section drafted **uncommitted, prose
  only** — the MLIR listing still needs folding in (see Task 0).

Chapter map (`content.tex`): 1 Intro, 2 You Are Here, **3 MLP, 4 CNN,
5 CIFAR+BatchNorm, 6 ResNet-34, 7 MobileNetV2, 8 EfficientNet, 9 ConvNeXt,
10 Vision Transformer**, 11 Bestiary, 12 Getting started, 13 On Verification.

---

## Work plan (for the clean session)

- **Task 0 — finish ch3.** Fold the `mlp_back` listing into the drafted ch3
  section in `content.tex` (currently prose-only); commit. Reference render:
  `planning`-adjacent preview was `/tmp/prev/ch3_combined.tex`.
- **Task 1 — writing pass, chapters with a built block (4, 5, 6, 10).** Add the
  4-beat section + the boxed listing from the existing `.mlir` (regenerate via
  `lake env lean LeanMlir/Proofs/IRPrint.lean`). Mechanical.
- **Task 2 — key-block builds (7, 8, 9).** Render + oracle-validate one block
  each (inverted residual / MBConv / ConvNeXt block); add `layerScale` render.
  Then their sections (Task 1 style).
- **Task 3 — "go deeper" pointers.** Make the full `.mlir` artifacts reachable
  (appendix listing, repo path, or a generated-MLIR dump target) so readers can
  see the whole net.
- **Optional — full-net builds for 7/8/9** if a chapter should stand alone with
  its own assembled net (the ResNet tower is the template).

**Effort:** Task 0 ~30 min; Task 1 ~½ day/chapter (writing); Task 2 ~½ day/block
+ writing. No research risk; no new proofs.

---

## Scope & honest boundaries (carry into every chapter)

- **Representative (Imagenette) scale, not ImageNet-1000.** The proofs cover
  representative concrete instances (and the simpler nets are dim-parameterized);
  the literal full-scale published models under verified codegen are the open
  scaling question the blueprint already flags (the big nets use the phase-2
  JAX trainer for accuracy numbers).
- **Smooth-point conditionality** for relu/relu6/maxpool/BN bridges — the
  equality is allowed to fail only on the measure-zero kink set.
- **R4 (printer faithfulness)** is the irreducible trusted surface: ~one printer
  + IREE + float, validated numerically. Proving it needs a StableHLO *text*
  semantics (out of scope).
- **Whole-net theorems are representative**: e.g. `cnn_has_vjp_at` is a 2-block
  ResNet; the 16-deep tower composes the same proven per-block lemmas (the
  generator does the composition; the Lean side extends by the same
  `vjp_comp_at` pattern). "Verified [net]" = every op/block proven + composed +
  GPU-validated, at representative scale — stated that way, not as one monolithic
  N-deep theorem.

---

## Start here (clean session)

1. Read this doc + `planning/verified_codegen.md`.
2. `lake env lean LeanMlir/Proofs/IRPrint.lean` (writes all `/tmp/*.mlir`);
   `.venv/bin/python LeanMlir/Proofs/check_ir_codegen.py` (confirm all green,
   CPU + GPU).
3. Do **Task 0** (commit ch3 with its listing), then **Task 1** for ch 4/5/6/10,
   then **Task 2** for ch 7/8/9. Commit per chapter; push when asked.
