# Re-spell the TF-origin nets at XLA-SAME padding, tie B0, then audit the blueprint

**For a fresh session. Scoped 2026-09-05, nothing below has been started.** Three items in order:
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

1. **Foundation: the two hand-written odd-phase backwards and their leaf ties.**
   `flatConvStride2XlaBack W` is `convFlatBack W` after scattering the cotangent to the ODD
   positions, mirroring `flatConvStride2Back = convFlatBack ∘ decimateBack`. Prove
   `flatConvStride2XlaBack_eq_vjp_backward` against `flatConvStride2Xla_has_vjp` the way
   `flatConvStride2Back_eq_vjp_backward` is proved; same for depthwise. ⚠ `3d9b14d`: the
   input-VJP's pad shift runs OPPOSITE to the weight-grad's because the reversed kernel flips the
   index shift; a version derived "by symmetry" type-checks, has the right shape, trains, and is
   wrong. Basis-probe the new backward against the certified `.backward` at k = 3 on a small even
   grid before writing the tie (the method of the archived log, section 3.19). Done when both ties
   compile and the probe agrees exactly.

2. **Float leaves. Done 2026-09-05** (`floatClose_` / `floatBridges_` / `floatBridgesTo_` for
   `flatConvStride2Xla` in `Resnet34WholeFloatBridge.lean` and for `depthwiseStride2FlatXla` in
   `EfficientNetWholeFloatBridge.lean`; the float peers `FloatModel.flatConvStride2XlaF` /
   `depthwiseStride2FlatXlaF`; `Maps.flatConvStride2Xla`, `Maps.depthwiseStride2FlatXla`,
   `Maps.flatConvStride2XlaBack`, `Maps.depthwiseStride2XlaBack`; `Maps.decimateOddBack` moved
   from the LN file to `FloatBudgetEnvBack.lean` so the stride-2 leaves reach it; an `example`
   in `EfficientNetFloatBudget.lean` closes the stem at `b0EvalBridge_maps`'s numerals through
   the XLA leaf, to be retired by step 5). Original text: `floatClose_flatConvStride2Xla` / `floatBridgesTo_` / `Maps.flatConvStride2Xla`
   with the float peer `decimateOddFlat ∘ flatConvF`, and the depthwise peer; backward
   `Maps.flatConvStride2XlaBack` and `Maps.depthwiseStride2XlaBack` as `convBack` after
   `decimateOddBack`. Put them beside the symmetric ones (`Resnet34WholeFloatBridge.lean` /
   `FloatBudgetEnv.lean` for the conv, `FloatBudgetEnvMBConv.lean` for depthwise,
   `FloatBudgetEnvBack*.lean` for the backwards), not in a new file, so no two files carry the
   same name unseen. Done when a compiled `example` closes one site at `b0_eval_chain`'s stem
   numerals with the Xla leaf.

3. **Re-spell the committed forwards.** `stemB` (`Codegen/EfficientNetRenderPC.lean:50-54`) and
   `stemB_has_vjp` (`Architectures/EfficientNetChainClose.lean:174`); the mnv2 stem
   (`Codegen/MobileNetV2RenderPC.lean:131`) and the strided depthwise stage (`:54`); the two
   `*RenderPCEval` twins; the PC graphs' tokens (`.convStrided` to `.convStridedXla`, depthwise
   likewise) and the `_faithful` proofs through the `den_batchOp_*Xla` lemmas; the apexes
   `efficientnetForwardB_has_vjp` and `mobilenetv2PC_has_vjp_at` (its stem witness
   `convStridedBnRelu6PC_has_vjp_at` moves to the Xla VJP); `WholeNetForwardTies.lean`,
   `*ChainClose.lean`, `*Close.lean`, `*FaithfulPoC.lean`, `*TiePoC.lean` wherever they name the
   stem. The shape checks `efficientnetForwardB_eq_chain` and `mobilenetv2Forward_full_pc_eq_chain`
   must stay `rfl`. Section 4 has the file list. Done when `lake build Proofs Certs` is green and
   `grep -rn "flatConvStride2 \|depthwiseStride2Flat " LeanMlir/Proofs --include=*.lean` returns
   only ResNet, ConvNeXt, MobileNetV4, the paper-MobileNetV2 files, and the leaf definitions.

4. **Backward chains and the two backward numbers.** `efficientnetInputGradB`'s stem
   (`EfficientNetWholeBackFloatBridge.lean:45`, `flatConvStride2Back`) and `mnv2InputGrad`'s stem
   plus four `depthwiseStride2Back` sites (`MobileNetV2BackFloatBridge.lean:133` and the strided
   bodies); the `Maps` chains in `EfficientNetBackFloatBudget.lean` and
   `MobileNetV2BackFloatBudget.lean` at the new leaves; `mnv2InputGrad_eq_mobilenetv2_vjp` re-tied
   with the Xla leaf ties. ⚠ Expect **7.104e182 / 1.578e182** and **4.750e153 / 1.076e152** to
   reproduce to the digit (`verify_b0_back` 138 and `verify_mnv2_back` 136 unchanged, since the
   probe's fan-ins and roundings are the same). If a numeral moves, stop and explain before
   committing. Done when both files compile with the committed numerals and the mnv2 tie is green.

5. **Forward numbers.** `b0EvalForward` and `mnv2EvalForward` at the Xla leaves. Expect
   **2.580e55 / 8.408e210** and **2.154e3 / 1.444e96** unchanged, `verify_b0` 96 and `verify_mnv2`
   116 unchanged. Same rule if anything moves.

6. **Artifacts, ties, disclosures.** Re-render `mobilenetv2_fwd.mlir` (and the SGD train step if
   step 0 moves it); add or run a byte tie between `pretty` of each PC graph and its artifact
   (`efficientnet_fwd`, `efficientnet_fwd_eval`, `mobilenetv2_fwd`, `mobilenetv2_fwd_eval`) so the
   `_faithful` theorem is about the file that ships; `convention_audit.py --selftest`;
   `scripts/mnv2_forward_tie.py` on shared weights (CPU is enough; use the pinned jax venv). Then
   remove the "A PADDING-CONVENTION GAP" paragraph from `formalization.yaml` 4d and the four flags
   on the B0 / MobileNetV2 `main_results` comments; in `planning/float_budget_numbers.md` mark
   section 5 item 2 (a) done and update the section 4 row. Update the four budget files' headers
   where they say "the deployed forward". Done when every artifact for the two nets has the same
   pad profile as its JAX reference and the docs say so.

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
