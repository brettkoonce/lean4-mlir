# validation_sweep.md — putting "verified codegen" into chapters 3–10

**Status (2026-06-06): the chapter 3–10 sweep is complete.** Every
architecture chapter has its *verified-codegen* section in
`blueprint/src/content.tex` (the nine `\section{MLIR: …}` blocks), each with a
real emitted-MLIR listing of its key block, the proven-vs-trusted ledger, and
a GPU-validated number. All blocks are rendered by `IRPrint.lean` and
oracle-checked by `check_ir_codegen.py` (52 checks, CPU gate + ROCm GPU).

This doc now serves two purposes: **(a)** the *record* of that completed sweep
(the original handoff plan, kept below, with statuses updated to ✅), and
**(b)** a *forward sketch* for extending the same verified-codegen pattern to
the rest of the book — the generative / detection / sequence demos and the
deeper formal frontiers (see "Extending verified codegen to the rest of the
book").

Predecessor doc: `planning/verified_codegen.md` (the design + results of the
codegen pipeline itself).

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
   and diffs against an independent NumPy reference. 52 checks today; CPU is the
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
| 7 | MobileNetV2 | inverted residual (1×1 → depthwise → 1×1 + residual; relu6) | ✅ `invres_back` | optional |
| 8 | EfficientNet | MBConv (depthwise + SE + swish) | ✅ `mbconv_back` | optional |
| 9 | ConvNeXt | ConvNeXt block (7×7 depthwise → LN → 1×1 → gelu → 1×1 → layerScale + residual) | ✅ `convnext_back` (+ `layerScale` render) | optional |
| 10 | Vision Transformer | `attn_back` (SDPA + 3-way QKV fan-in) or `vit_back` (full block) | ✅ 3e-7 / 6e-7 | `vit` transformer block ✅ |

**All eight chapters done.** ch 7/8/9 reused only already-validated ops
(`dw_fwd/back`, `se_fwd/back`, `swish`, `sigmoid`, `relu6`, `gelu`, BN/LN,
`conv`, `residual`) with no new proofs; `layerScale` (ch9) was the one trivial
`dy ⊙ γ` render added. Each landed in ~½ day, as estimated. The MLIR sections
in `content.tex` are `MLIR: Depthwise Convolution` (ch7), `MLIR:
Squeeze-and-Excitation` (ch8), `MLIR: Layer Scale` (ch9).

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

- **Whole nets / key blocks, end-to-end + GPU-validated:** MLP train step,
  CNN train step, ResNet train step (+ 16-block tower), ViT transformer block,
  and the ch 7/8/9 blocks — MobileNetV2 inverted residual (`invres_*`),
  EfficientNet MBConv (`mbconv_*`), ConvNeXt block (`convnext_*`). All in
  `IRPrint.lean`, checked by `check_ir_codegen.py` (52 checks).
- **Every op proof-backed + validated:** dense, relu, conv2d (+weight grad via
  the transpose trick), depthwise conv, maxpool, flat-BatchNorm/LayerNorm,
  softmax, scaled-dot-product attention (dQ/dK/dV), softmax-CE loss,
  gelu/swish/sigmoid/relu6, residual, squeeze-excite, global-avg-pool,
  layerScale.
- **Audit:** `tests/AuditAxioms.lean` — the bridge theorems are in the 3-axiom
  gate (114 decls; CI re-checks the closure).
- **Whole-net VJP status (the proofs the codegen renders):** ViT, ConvNeXt
  and EfficientNet are now *unconditional* global `HasVJP` (correct at every
  input, only `0 < ε`); MLP, MNIST-CNN, ResNet and MobileNetV2 carry concrete
  instances discharging every smoothness hypothesis (`MlpConcrete`,
  `Spatial`/`Mini`, `CnnConcrete`, `MobileNetV2Concrete`). See the
  "Whole-network VJPs" section of `LeanMlir/Proofs/README.md`.
- **State:** all codegen, oracle, and the nine ch 2–10 `MLIR:` book sections
  are committed and on `origin/main`. (The ch3 listing has been folded in;
  the original "Task 0" below is done.)

Chapter map (`content.tex`): 1 Intro, 2 You Are Here, **3 MLP, 4 CNN,
5 CIFAR+BatchNorm, 6 ResNet-34, 7 MobileNetV2, 8 EfficientNet, 9 ConvNeXt,
10 Vision Transformer**, 11 Bestiary, 12 Getting started, 13 On Verification.

---

## Work plan — completed ✅

The original handoff tasks are all done, kept here as a record:

- **Task 0 — finish ch3.** ✅ `mlp_back` listing folded into the ch3 section.
- **Task 1 — writing pass (4, 5, 6, 10).** ✅ all four `MLIR:` sections written.
- **Task 2 — key-block builds (7, 8, 9).** ✅ inverted residual / MBConv /
  ConvNeXt block rendered, oracle-validated, and written; `layerScale` added.
- **Task 3 — "go deeper" pointers.** ✅ full `.mlir` artifacts regenerable via
  `lake env lean LeanMlir/Proofs/IRPrint.lean`.

No research risk materialized and no new proofs were needed, exactly as
predicted — the per-op/generic foundation made the sweep mechanical.

---

## Extending verified codegen to the rest of the book

The ch 3–10 sweep covers supervised image classification. The remaining book
material — the generative / detection / sequence **demos** (blueprint §11.2.x:
DDPM, TinyGPT, U-Net, YOLO, autoencoder, GradCAM) — reuses the **same three
pieces** (proven op → `Back`/`Fwd`/`Back3` bridge → `IRPrint` render →
`check_ir_codegen` oracle). Only the op library grows; the pattern is
unchanged. What's reusable vs new, by current proof coverage:
reused — SDPA, LN, gelu, conv, residual, softmax-CE (all proven); new — the
ops with no proof file today (`channelConcat/Split`, `focal`/`mse`/`BCE`,
`adam`, causal mask, upsample).

### New ops, by demo

| demo (ch 11.2.x) | net | new ops to prove + bridge | reuses |
|---|---|---|---|
| **TinyGPT** (LM) | causal transformer | token/pos **embedding lookup** (backward = scatter-add); **causal-masked softmax** (additive −∞ mask; smooth where finite); LM cross-entropy = softmax-CE (done) | SDPA, LN, gelu, dense |
| **U-Net / autoencoder** | encoder–decoder | **transposed / nearest-upsample conv** (input-VJP is a forward conv — same Toeplitz lemma); **channel concat/split** (`channelConcat`↔`channelSplit`: a reindex VJP from the foundation's reindex rule) | conv, BN, relu |
| **DDPM** (diffusion) | U-Net + time embed | **sinusoidal timestep embedding** (smooth, closed-form deriv); **MSE / ε-prediction loss** (trivial linear VJP); weight **EMA** is inference-time (no grad) | U-Net ops above |
| **YOLOv1** (detection) | R34 backbone + head | composite loss: coord **MSE** + objectness/class **BCE** + **focal** term (each a closed-form scalar grad); NMS is inference-only | conv, residual (backbone already proven) |
| **DCGAN / Pix2Pix** | G/D adversarial | **BCE adversarial loss**; transposed conv (above). The alternating two-network step is a training-loop concern, not a new VJP | conv, BN, transposed conv |
| **GradCAM** | — | **none** — GradCAM *is* a backward pass (visualizes `∂score/∂activations`), so it is a direct *consumer* of the proven VJPs. The cleanest "the proofs are the product" demo. |

Each new op is a single `HasVJP`/`HasVJPAt` proof in the existing
`pdiv`/`fderiv` style, then a `Back`-node + bridge + one renderer + one oracle
check — the same ~½-day-per-op shape as the ch 7–9 blocks. No research risk.

### Two structural pieces still trusted (candidates to pull into "proven")

1. **The optimizer.** `weight_grad_bridge` / `lossCot_bridge` prove the
   *gradient* the train step feeds Adam, but the **Adam update map itself**
   (`m,v ← …; θ ← θ − lr·m̂/(√v̂+ε)`) is rendered and numerically validated, not
   proven (`grep adam Proofs/` = 0). It is a smooth elementwise map — a
   self-contained `adamStep θ g m v = …` lemma bridged to the emitted
   `stablehlo` would make the *whole* train step proof-backed, not just
   forward + backward + loss. SGD/momentum are easier warm-ups.
2. **The loss/data boundary.** Augmentation + IDX/JPEG decode stay trusted
   (I/O, out of scope). The loss *heads* (above) are in scope and cheap.

### The deeper formal frontiers (the genuinely hard, unchanged)

- **R4 — printer faithfulness.** That `print(IR)` *is* the StableHLO it claims
  is the one irreducible trusted surface. Closing it needs a formal StableHLO
  **text** semantics (a parser + denotation agreeing with `⟦·⟧`). Highest value,
  highest effort; the natural subject of ch 13 ("On Verification"). Scoping it to
  the *emitted subset* (~20 ops the printer uses) makes it finite — start with
  the already-bridged dense/relu fragment.
- **float32 ≈ ℝ.** The proofs are over `ℝ`; the kernels run `float32`. A per-op
  error-bound layer (interval / relative-error) is the bridge — large but
  standard numerical-analysis territory; the oracle's ~1e-7 agreement is the
  empirical stand-in today.
- **Scale.** Proofs are representative concrete instances + dim-generic per-op
  lemmas; full ImageNet-1000 models under verified codegen is the open scaling
  question (big nets use the phase-2 JAX trainer for accuracy). The per-block
  lemmas compose; the open part is instantiating / generating the N-deep
  assembly, not new math.

### Suggested order (lowest effort × highest leverage first)

1. **Adam bridge** — closes the train-step story end to end; small, self-contained.
2. **U-Net ops** (transposed/upsample conv + channel concat/split) — unlocks
   U-Net, autoencoder, and the DDPM/GAN backbones at once.
3. **Embedding + causal mask** — unlocks TinyGPT; pairs with proven SDPA/LN/gelu.
4. **Loss heads** (MSE, BCE, focal) — unlocks DDPM and YOLO training objectives.
5. **R4 subset semantics** — the ch-13 capstone; begin with the dense/relu fragment.

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

The ch 3–10 sweep is done. To **verify the current state**:

1. Read this doc + `planning/verified_codegen.md`.
2. `lake env lean LeanMlir/Proofs/IRPrint.lean` (writes all `/tmp/*.mlir`);
   `.venv/bin/python LeanMlir/Proofs/check_ir_codegen.py` (confirm all 52 green,
   CPU + GPU); `lake env lean tests/AuditAxioms.lean` (114 decls, 3-axiom).

To **extend to the rest of the book**, work the "Extending verified codegen to
the rest of the book" section in suggested order — start with the **Adam
bridge**, then the **U-Net ops**. Each new op is one proof + bridge + renderer
+ oracle check, then its demo's `MLIR:` section in the same 4-beat template.
Commit per op/chapter; push when asked.
