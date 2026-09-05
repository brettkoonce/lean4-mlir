# Re-spell the TF-origin nets at XLA-SAME padding, tie B0, then audit the blueprint

**Scoped 2026-09-05; steps 0 to 6 all landed the same day (`ec977de`, `0773e20`, `0584ab8`,
the B0 commit, `ee81d36`, and the 3c/docs commit). What is left is step 7 (B0's whole-net
certified tie) and step 8 (the blueprint audit).** Three items in order:
(A) move EfficientNet-B0's and MobileNetV2's Proofs tier from symmetric stride-2 padding to the
XLA-SAME forms the shipped renders and the TF-origin references use, (B) EfficientNet-B0's whole-net
certified backward tie, which must not be built before (A), and (C) an audit of the LaTeX blueprint
against the current Lean. The standing reference for the float budgets is
`planning/float_budget_numbers.md`; the finding that motivates (A) is its section 5, item 2.

**Why (A) is the right call and not a disclosure.** These nets are reproductions of published
recipes, and the reference implementations for MobileNetV2, MobileNetV4 and EfficientNet are
TensorFlow-era: their stride-2 convolutions pad `'SAME'`, which on an even input with a 3x3/s2
kernel is `(0, 1)`, not the PyTorch-style symmetric `(1, 1)`. Using timm as the ruler has exposed
this class of gap before (the r50 stem, the mnv4 stem, mnv2's five sites, the B0 stem, all in
`scripts/convention_audit.py`'s ledger). The renders were fixed on 2026-08-08. The certified chains
were not, so today the theorems about "the deployed forward" for those two nets describe a program
that reads its window one pixel from where the shipped program does. The numbers do not move; the
claim does. A theorem should be about the program we run.

## 1. What is wrong, precisely

The two maps. `flatConvStride2 W b = decimateFlat ∘ flatConv W b` keeps the EVEN positions of a
stride-1 symmetric-pad conv. `flatConvStride2Xla W b = decimateOddFlat ∘ flatConv W b` keeps the
ODD ones, which is exactly XLA `'SAME'` on an even input (`3d9b14d`, verified over 33 configs).
Same output size, same fan-in, same rounding; a different function. Depthwise likewise
(`depthwiseStride2Flat` against `depthwiseStride2FlatXla`).

| net | sites at the wrong phase in the Proofs tier | shipped artifact (XLA-SAME sites / convs) |
|---|---|---|
| EfficientNet-B0 | stem 3x3/s2, 3 to 32, 224 to 112 | `efficientnet_fwd` 1/49, `efficientnet_fwd_eval` 1/49, `efficientnet_adam_train_step` 1/146 |
| MobileNetV2 (the Proofs-tier 6-block reduced net) | stem 3x3/s2, 3 to 16; strided depthwises in b1, b3, b5, b6 (112 to 56, 56 to 28, 28 to 14, 14 to 7) | `mobilenetv2_fwd_eval` 5/52, `mobilenetv2_adam_train_step` 5/155; `mobilenetv2_fwd` 0/52 and `mobilenetv2_train_step` (SGD) 0, symmetric on purpose per `scripts/convention_audit.py` (until step 0) |

**Correction found in step 0 (2026-09-05): the Proofs tier describes neither shipped net.**
`mobilenetv2Forward_full_pc` (`MobileNetV2RenderPC.lean`) is the 6-block reduced MobileNetV2
(stem 3 to 16, head 64 to 128, 82 parameters), not the 17-block paper net the trainers run; its
only artifact is `mobilenetv2_reduced_train_step.mlir`, written by `MobileNetV2Render.lean`,
exempt from both prefix audits and read by no trainer. `efficientnetForwardB`
(`EfficientNetRenderPC.lean`) is a three-block representative B0 with no artifact at all. So the
byte tie step 6 asks for, between `pretty` of a PC graph and `efficientnet_fwd` /
`mobilenetv2_fwd`, cannot exist: the graphs and the artifacts are different nets. What can exist
is a tie of `mobilenetv2FwdGraphFullPC`'s `pretty` against the forward prefix of
`mobilenetv2_reduced_train_step.mlir`, both XLA-SAME after step 0, and nothing for B0. The
re-spelling stands: the Proofs nets are the representatives of the shipped convention, and (A)
makes them read the same phase the shipped nets do. `formalization.yaml` 4d's "the B0 chain
describes a program no shipped artifact has run" should be read the same way: no artifact runs
that three-block net at either phase.

Not affected, and do not touch: ResNet-34 and ResNet-50 (PyTorch-origin; the 7x7/s2 stem pads 3
symmetrically and the 3x3/s2 pool pads 1, both correct), ConvNeXt (4x4/s4 and 2x2/s2 at pad 0),
ViT (own patch-embed definition). MobileNetV4's Proofs (`MobileNetV4BackB0.lean`) are the UIB
block bodies only, whose depthwises are symmetric in render and reference alike; the XLA stem is
not in its Proofs tier at all, so nothing there is at the wrong phase.

**Pulled INTO scope by step 0 (2026-09-05): the 17-block paper MobileNetV2 files.**
`MobileNetV2FullPaper.lean` (forward + graph + faithfulness), `MobileNetV2FaithfulPoCPaper.lean`
(every param-SGD op `den = certified`) and `MobileNetV2TiePoCPaper.lean` (the §1a tie, 210
params) describe `mobilenetv2_train_step.mlir` at the symmetric spelling (`flatConvStride2`,
`depthwiseStridedF`, `convStrided{Weight,Bias}Sgd`, `depthwiseStrided{Weight,Bias}Sgd`). Before
step 0 that artifact was symmetric, so they were right; step 0 moved it, so they now describe the
program of 2026-09-04. Step 3 re-spells them with the others: the forward at
`flatConvStride2Xla` / `depthwiseStride2FlatXla`, the graph at `.flatConvStridedXlaF` /
`.depthwiseStridedXlaF`, and the PoC `_den` lemmas at the five per-example `…Xla…Sgd` /
`…XlaBack` tokens (their `rfl` faithfulness lemmas landed in step 0; the `.correct` fields of
the `…Xla` VJPs give the `pdiv` form). The EfficientNet PoC pair (`EfficientNetTiePoC.lean`,
`EfficientNetFaithfulPoC.lean`) has the same relation to `efficientnet_train_step.mlir` since
2026-08-08 (batched `convStridedWeightSgdB` where the artifact emits `convStridedXlaWeightSgdB`)
and was already in section 4's list.

## 2. What exists already

Foundation. `flatConvStride2Xla` with `_differentiable`, `_has_vjp`, `_has_vjp_correct`,
`_weight_grad_has_vjp`, `_bias_grad_has_vjp` (`Foundation/StridedConv.lean:363-434`).
`depthwiseStride2FlatXla` with the same five (`Architectures/Depthwise.lean:1433-1490`).
`decimateOddFlat` and its VJP, from ConvNeXt's patchify.

Codegen. Per-example tokens `.flatConvStridedXlaF`, `.depthwiseStridedXlaF`; batched
`.convStridedXla`, `.depthwiseStridedXla` with `den_batchOp_convStridedXla` and
`den_batchOp_depthwiseStridedXla` (`Codegen/StableHLO.lean:2355`, `:2370`); the batched
weight-grad and SGD peers `convStridedXlaWeightGradB`, `convStridedXlaWeightSgdB` (`601a900`).
`EfficientNetRender.lean:776` and `MobileNetV2Render.lean:104, :551` already emit them.

Float tier for the odd phase. `Maps.decimateOddBack` and the `flatConvStride4` leaves
(`FloatBudgetEnvLN.lean`, `FloatBudgetEnvBackLN.lean`) show the shape: a selection or a scatter
is envelope-preserving, so the Xla leaves have the SAME numerals as the symmetric ones.

Does not exist. Any hand-written odd-phase backward (`flatConvStride2XlaBack`,
`depthwiseStride2XlaBack`); any float peer (`flatConvStride2XlaF`, `floatClose_`,
`floatBridgesTo_`, `Maps.` for either op, forward or backward); any leaf tie
`flatConvStride2XlaBack_eq_vjp_backward`. `grep -rl Xla LeanMlir/Proofs/Float` is empty.

## 3. The work, in order

Each step has an acceptance criterion. Probe before Lean where a number is involved.

0. **Decide the SGD pair. Done 2026-09-05: moved.** `mobilenetv2_fwd` and `mobilenetv2_train_step`
   were the one symmetric pair left, kept "as a self-consistent different net"
   (`convention_audit.py`, NETS comment). Moving the forward alone would have paired an XLA-SAME
   forward with a symmetric backward, which type-checks, descends and computes a different net's
   gradient, and the per-example SGD render had no XLA-SAME backward tokens to reach for (only
   the batched ones the Adam render uses existed). So step 0 cost five per-example tokens in
   `StableHLO.lean` (`convStridedXla{Weight,Bias}Sgd`, `depthwiseStridedXlaBack`,
   `depthwiseStridedXla{Weight,Bias}Sgd`; constructor, `den`, `rfl` faithfulness, `Raw`, `Tok`,
   `skel`, `toToks`, emit, parse case and roundtrip proof), two `Den` wrappers in
   `Depthwise.lean`, and `MobileNetV2Render.lean` losing its `xlaPad` flag so every artifact it
   writes (`mobilenetv2_fwd`, `_fwd_eval`, `_train_step`, `_reduced_train_step`, the two
   `mobilenetv2in` forwards) is XLA-SAME at all five sites. The three new backward emit arms are
   known-answer checked against `jax.vjp` in `scripts/xla_pad_op_check.py` (rows `*_pe`), since
   they are hand copies of the batched arms and a copy is what drifts. `convention_audit.py`
   audits `_fwd` directly again; `--selftest` reproduces the ledger.

1. **Foundation: the two hand-written odd-phase backwards and their leaf ties. Done 2026-09-05
   (`0773e20`).** `flatConvStride2XlaBack = convFlatBack ∘ decimateOddBack` and
   `depthwiseStride2FlatXlaBack = depthwiseFlatBack ∘ decimateOddBack`, beside their symmetric
   peers, each with `floatBridges_` / `floatBridgesTo_` by one `.comp`; the leaf ties
   `flatConvStride2XlaBack_eq_vjp_backward` / `depthwiseStride2FlatXlaBack_eq_vjp_backward` by the
   same four-line proof as the symmetric ones. No basis probe was needed: the definition is the
   composition, the kernel checks it is the certified VJP, and the emitted arms were
   `jax.vjp`-checked in step 0.

2. **Float leaves. Done 2026-09-05 (`0584ab8`).** `floatClose_` / `floatBridges_` /
   `floatBridgesTo_` for `flatConvStride2Xla` (`Resnet34WholeFloatBridge.lean`) and
   `depthwiseStride2FlatXla` (`EfficientNetWholeFloatBridge.lean`); `FloatModel.flatConvStride2XlaF`
   / `depthwiseStride2FlatXlaF`; `Maps.flatConvStride2Xla`, `Maps.depthwiseStride2FlatXla`,
   `Maps.flatConvStride2XlaBack`, `Maps.depthwiseStride2XlaBack`; `Maps.decimateOddBack` moved
   from the LN file to `FloatBudgetEnvBack.lean`. All verbatim copies of the symmetric leaves,
   since a decimation picks coordinates.

3. **Re-spell the committed forwards.** ⚠ Steps 4 and 5 fold INTO this step per net: a net's
   budget files name its leaves, so the forward cannot move without the backward chain and both
   budget files moving in the same commit, and the acceptance for 4 and 5 (numerals unchanged)
   is checked here.

   **3a. EfficientNet-B0. Done 2026-09-05.** One site, the stem. `stemB` and `stemBEval`,
   the two PC graphs (`.convStridedXla`, faithfulness through `den_batchOp_convStridedXla`),
   `stemB_has_vjp`, `EnetPoC.convStridedWB_den` and `enetStemTied` (now about
   `convStridedXlaWeightSgdB`, the op `efficientnet_train_step.mlir` has emitted since
   2026-08-08), `floatBridges_stemB` / `stemBGen` / `stemBF` / `floatBridgesTo_stemBGen`,
   `efficientnetInputGradB` and `efficientnetInputGradBF` (stem scatter `decimateOddBack`),
   and both budget files. `EfficientNetFullB0.lean` and the 262-param tie follow through
   `stemB`. Both numbers reproduced to the digit. B0's strided depthwises stay symmetric:
   render and reference both pad them `(p,p)`. Collateral: `MobileNetV4BackB0.lean` had
   borrowed `stemB` for MNv4's fused 3x3/s2 stage, which is symmetric in MNv4's render, so that
   stage now has its own `fusedConvB` with the same lemmas; its docstring had also claimed the
   shared backward graph certified B0's stem, which it cannot (no render emits a gradient into
   the image, and no batched XLA input-VJP token exists), and that is now recorded.

   **3b. MobileNetV2, the 6-block reduced cone, the 17-block paper files and the batched Adam
   backward graphs. Done 2026-09-05 (`ee81d36`), by running `scripts/respell_mnv2_xla.py`.**
   All four numerals reproduced (`2.154e3 / 1.444e96`, `4.750e153 / 1.076e152`) and the two
   counts held (116, 136). Four hand-fixes were needed, all one shape — a float skeleton still
   spelling the even-phase scatter after its real side had moved: `mnv2InputGradF` and the
   `hstem` ascription (`MobileNetV2BackFloatBridge.lean`, which failed as a `maxRecDepth`),
   `mnv2GradF` (`MobileNetV2BackFloatBudget.lean`), and `FloatBudgetEnvBackMBConv.lean` —
   NOT on the script's list — where `invresBodyStridedBackPCF` and the bridge and `Maps`
   proofs still reached for the symmetric depthwise; nothing outside MobileNetV2 consumes
   those three, so they flip in place. The fourth was `tests/TestMobilenetV2TrainPC.lean`:
   the script moved its `pretty` tokens but not its hand-emitted weight grads, which left an
   XLA forward against symmetric weight gradients — shape-preserving, so IREE compiles it
   either way and nothing would have caught it. `convWGrad` / `dwWGrad` took an `xla` flag
   copying the emitter's `jax.vjp`-checked arms (`[[0,2],[0,2]]`, the `[p-1,p+1]` shift, NOT
   the `[p+1,p-1]` the reversed-kernel input-VJP takes); five sites carry it. The script
   substitutes the 55-rule name map over the sixteen
   MobileNetV2 files (the two PC graphs, ChainClose, FaithfulPoC, FaithfulPoCPaper, TiePoCPaper,
   FullPaper, FullVJP, BackB0, BackCertifiedTie, WholeBackCertifiedTie, the four Float files,
   and `tests/TestMobilenetV2TrainPC.lean`), adds the four XLA twins of the shared certs to
   `MobileNetV2Close.lean`, the two per-example XLA stem dens to `MobileNetV2FaithfulPoC.lean`,
   `depthwiseStridedXlaBackBatched_faithful` to `EfficientNetBackB0.lean`, and the seven
   `#print axioms` lines to `tests/AuditAxioms.lean`. It refuses to run twice. Then
   `lake build LeanMlir.Proofs.Architectures.MobileNetV2TiePoCPaper
   LeanMlir.Proofs.Float.MobileNetV2FloatBudget LeanMlir.Proofs.Float.MobileNetV2BackFloatBudget
   LeanMlir.Proofs.Foundation.MobileNetV2WholeBackCertifiedTie
   LeanMlir.Proofs.Architectures.MobileNetV2FullVJP LeanMlir.Proofs.Foundation.BackNetFolds` for
   the fast signal, then the section-6 gates. Expect to hand-fix: (i) any `rfl` that fails
   because a term still spells the even phase (grep the failing file for `decimateBack` and
   `flatConvStride2 `; the B0 float backward net needed exactly this), (ii) prose in the
   sixteen files that names ResNet or ConvNeXt next to a now-XLA name (the script's final grep
   lists candidates; ResNet's `cbrStridedPC` is symmetric and must read so), (iii)
   `tests/TestMobilenetV2TrainPC.lean`, which is a retired demo run only by
   `regen_verified_mlir.sh tests`; if it will not compile leave its tokens symmetric and say so
   in its header. **Numerals: `2.154e3 / 1.444e96` and `4.750e153 / 1.076e152` must reproduce,
   `verify_mnv2` 116 and `verify_mnv2_back` 136 unchanged; if one moves, stop and explain.**

   ⛔ **The trap the script is built around: shared lemmas.** `mnv2_render_stem_conv{W,b}_certified`
   are reused by ResNet-34 (`ResNet34Close.lean`, `ResNet34ChainClose.lean`,
   `ResNet34FaithfulPoC.lean`) and `mnv2_render_depthwise{W,b}_strided_certified` by
   EfficientNet-B0 (`EfficientNetClose.lean`), both at symmetric padding, correctly. They must
   NOT flip; the script adds `_xla_` twins and repoints only the MobileNetV2 consumers.
   `ResNet34PoC.convStrided{W,B}_den` likewise stays; MobileNetV2's paper PoC gets
   `Mnv2PoC.convStridedXla{W,B}_den`. The same shape as the MNv4 `stemB` collision in 3a: a
   shared definition is right for one net and wrong for the other the moment the conventions
   diverge, and nothing structural says which.

   **3c. The scalar-BN twin. Done 2026-09-05: FLIPPED, not allow-listed.** `MobileNetV2.lean`'s
   `mobilenetv2Forward_full` (scalar `bnForward`, the reduced 6-block net) with
   `convBnRelu6Strided_has_vjp_at` / `dwBnRelu6Strided_has_vjp_at` / `invresBodyStrided` — 42
   occurrences over six identifiers, each with an `Xla` peer already — and `StableHLO.lean`'s
   `mobilenetv2FwdGraphFull` (5 token sites, scoped to that def's body: the same constructors
   appear in ResNet's, EfficientNet's and MNv4's graphs) with `_faithful`'s simp set at
   `flatConvStridedXlaF_faithful` / `depthwiseStridedXlaF_faithful`. Both green first try, and
   `SpecVJP.lean`'s `rfl` denotation ties followed without an edit (they name the net, not its
   leaves). Flipped rather than allow-listed because the whole MobileNetV2 cone then reads one
   phase and no reader can mistake the stepping stone for the deployed net; the cost was one
   6.5-minute `StableHLO.lean` rebuild. The 2-block generic `mobilenetv2Forward` (yaml headline,
   `Mnv2Live`, the three seals) has a stride-1 stem and was untouched.

   **Acceptance: met, with the scalar twin flipped rather than allow-listed.**
   `lake build Proofs Certs` green (3956 jobs) and
   `grep -rn "flatConvStride2 \|depthwiseStride2Flat " LeanMlir/Proofs --include=*.lean`
   returns only ResNet, ConvNeXt, MobileNetV4, `EfficientNetClose.lean`'s strided-depthwise
   reuse, the shared symmetric certs in `MobileNetV2Close.lean`, and the leaf definitions —
   no MobileNetV2 file at all.

4. **Backward chains and the two backward numbers.** Folded into step 3 per net (see the
   note there). B0's landed with 3a: `7.104e182 / 1.578e182` unchanged. MobileNetV2's is
   part of 3b.

5. **Forward numbers.** Folded into step 3 per net. B0's `2.580e55 / 8.408e210` unchanged with
   3a; MobileNetV2's `2.154e3 / 1.444e96` is part of 3b.

6. **Artifacts, ties, disclosures. Done 2026-09-05.** Every MobileNetV2 artifact was
   re-rendered in step 0 and ties at 5.5e-6 (`scripts/mnv2_forward_tie.py --diag`, per-example
   BN row); B0's artifacts were already XLA-SAME, and nothing in steps 3b/3c writes an artifact
   (the PC render files have no writers), so `verified_mlir/` did not move. The byte tie between
   a PC graph's `pretty` and a shipped artifact cannot exist (section 1's correction: the PC nets
   are the reduced and representative ones); the achievable one, `mobilenetv2FwdGraphFullPC`'s
   `pretty` against the forward prefix of `mobilenetv2_reduced_train_step.mlir`, stays optional
   and was not built. Disclosures closed: `formalization.yaml` 4d's "A PADDING-CONVENTION GAP"
   is now "THE PADDING CONVENTION, CLOSED" and names what stayed symmetric and why; the two
   MobileNetV2 rows lost their flags; `planning/float_budget_numbers.md`'s section 4 row and
   section 5 item 2 record the decision as taken and done, and item 3 as unblocked; both
   MobileNetV2 budget headers gained the sentence B0's got. `convention_audit.py --selftest`
   passes (⚠ it needs `.venv/bin/python` — the system interpreter has a jaxlib-less jax).

7. **EfficientNet-B0's whole-net certified tie**, as scoped in `planning/float_budget_numbers.md`
   section 5 item 3, now against the XLA stem: apex `efficientnetForwardB_has_vjp`, the three
   batched block ties at `bnBatchLA`, the batched leaf ties via `hasVJPMat_to_hasVJP
   (rowwise_has_vjp_mat ..)`, assembly in `MobileNetV2WholeBackCertifiedTie.lean`'s shape with
   opaque blocks, at `N = 1`. Done when the tie compiles, the shape check is used, and the number
   in `EfficientNetBackFloatBudget.lean` is unchanged.

8. **Blueprint audit** (its own commit, after the Lean settles). `blueprint/src/content.tex` is
   15,126 lines with 107 `\lean{}` tags; CI's `blueprint-checkdecls` (`.github/workflows/blueprint.yml`)
   checks that every tagged NAME resolves and nothing else, so stale prose and stale numbers pass.
   Concrete leads, found 2026-09-05:
   * `content.tex:15019` quotes the adjoint-chain budgets "MNIST-MLP (0.8/21), MNIST-CNN
     (0.015/0.64), CIFAR-8 (2.6/4.6)". `formalization.yaml` 4c records that the 2.6 is the probe's
     figure at MEASURED activation magnitudes, which no instantiation of the theorem supplies; the
     theorem's own budget at He magnitudes is at least 1.8e13. Rewrite the sentence to say which
     quantity it is.
   * `content.tex:2891` and `:2935` describe `conv2d` as "SAME padding". The Lean pads
     `(k-1)/2` symmetrically; XLA `'SAME'` differs at even inputs, and after (A) the MobileNetV2
     and EfficientNet chapters (`:6136`, `:7357`) need a sentence saying which convention their
     verified render uses and why (TF-origin reference). Say "symmetric, same-size" for the
     definition and reserve "SAME" for XLA's.
   * The ResNet-34 chapter (`:4686`): check that the stem pool is described as 3x3/s2 (restored
     2026-08-03) and not 2x2; the `MaxPool 2x2` definition at `:3092` is the MNIST/CIFAR pool and
     is fine.
   * `content.tex:14950` mentions the float budgets in one clause. The table from
     `planning/float_budget_numbers.md` section 1 and the one-sentence claim at the end of its
     section 2 belong here, with the cap/fold label and the batch-size qualifiers.
   * Method for the rest: for each of the six net chapters list every claim about padding, pool,
     BatchNorm mode, LayerNorm count and activation, and check it against the current definition;
     for each `\lean{}` tag read the surrounding paragraph against the theorem it names. The
     archived log's corrections are the checklist: 36 BatchNorm sites not 33, 23 LayerNorm not
     22, training-mode BatchNorm HAS a statable (capped) number, the even-kernel backward, the
     head LayerNorm, the 3x3/s2 pool.
   Done when `blueprint-checkdecls` is green, the leads above are fixed, and every chapter's
   convention claims match the Lean.

## 4. Files that name the symmetric stride-2 maps in the B0 / MobileNetV2 cone

From `grep -rl "flatConvStride2\b\|depthwiseStride2Flat\b\|flatConvStride2Back\b\|depthwiseStride2Back\b"`,
restricted to the two nets and the shared leaves. Those marked (leaf) keep the symmetric
definitions and gain Xla peers; the rest change their spelling.

* Codegen: `EfficientNetRenderPC.lean`, `EfficientNetRenderPCEval.lean`, `MobileNetV2RenderPC.lean`,
  `MobileNetV2RenderPCEval.lean`, `StableHLO.lean` (leaf; tokens exist), `EfficientNetRender.lean`
  and `MobileNetV2Render.lean` (already Xla; step 0 touches the SGD branch only).
* Architectures / Foundation: `EfficientNetChainClose.lean`, `EfficientNetClose.lean`,
  `EfficientNetFaithfulPoC.lean`, `EfficientNetTiePoC.lean`, `MobileNetV2ChainClose.lean`,
  `MobileNetV2Close.lean`, `MobileNetV2FaithfulPoC.lean`, `MobileNetV2.lean`,
  `MobileNetV2FullVJP.lean`, `WholeNetForwardTies.lean`, `StridedConv.lean` (leaf),
  `Depthwise.lean` (leaf), `Foundation/MobileNetV2WholeBackCertifiedTie.lean`.
* Float: `EfficientNetWholeFloatBridge.lean`, `EfficientNetWholeBackFloatBridge.lean`,
  `EfficientNetFloatBudget.lean`, `EfficientNetBackFloatBudget.lean`, `EfficientNetBackB0.lean`,
  `MobileNetV2WholeFloatBridge.lean`, `MobileNetV2BackFloatBridge.lean`,
  `MobileNetV2FloatBudget.lean`, `MobileNetV2BackFloatBudget.lean`, `MobileNetV2BackB0.lean`,
  `MobileNetV2BackCertifiedTie.lean`, `StridedConvBackFloatBridge.lean` (leaf),
  `DepthwiseBackFloatBridge.lean` (leaf), `DepthwiseBackCertifiedTie.lean` (leaf),
  `FloatBudgetEnv.lean`, `FloatBudgetEnvMBConv.lean`, `FloatBudgetEnvBack.lean`,
  `FloatBudgetEnvBackMBConv.lean`, `FloatBudgetEnvBackSE.lean` (leaves).
* Out of scope, same class: `MobileNetV4BackB0.lean`, `MobileNetV2FullPaper.lean`,
  `MobileNetV2TiePoCPaper.lean`, `MobileNetV2FaithfulPoCPaper.lean`.
* Do not touch: every `Resnet34*` and ConvNeXt file the grep lists.

## 5. Traps this touches

* A computed dimension in an APPLIED position (`Vec (ic * (2*h) * (2*w))` against
  `Vec (3 * 224 * 224)`) sends the unifier into the net's semantics; give the stage a `def` with
  the type ascribed in the chain's spelling, and state leaf ties in the LEMMA's spelling. Pin
  `(ic := 3) (oc := 32) (h := 112) (w := 112)` on every strided leaf in a statement.
* Transport a `HasVJP` witness or a bridge along an equation by rebuilding the structure, never
  with `▸`: an `Eq.mpr` blocks `.backward` and `.mag` from reducing.
* `Elab.async` makes per-declaration timings order-dependent; measure with it off.
* The `verify_*` passes check a chain against itself. Reproducing the committed numerals after
  the re-spelling proves the fold is unchanged, not that the leaf is the right function; the leaf
  tie against the certified VJP is what proves that.
* The backward pad shift runs opposite to the forward's (`3d9b14d`); probe, do not derive.
* When a docstring justifies a slot with a count or a convention, re-derive it; three of this
  thread's defects hid behind one.

## 6. Process per commit

`lake build Proofs Certs`; `lake env lean tests/AuditAxioms.lean` exit 0 on the three core
axioms; `lake exe docstring-checkrefs`; `python3 scripts/check_audit_coverage.py`; for the
blueprint, `lake exe blueprint-checkdecls blueprint/lean_decls`. One commit per step above, or
per net inside step 3. Stage, then stop and ask before committing.
