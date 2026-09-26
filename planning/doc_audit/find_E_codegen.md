# Slice E — LeanMlir/Proofs/Codegen/ (25 files, ~27.9k lines)

**Coverage.** Read fully: StableHLOParse, StableHLOLex, LambTriple, SyncBnSites, FwdGraphTextTies (docstrings + guards), every module docstring in the slice, and every declaration docstring + its statement/signature (extracted mechanically, 1,200 pairs in StableHLO.lean, 3,800 lines of pairs elsewhere). Skimmed: the bodies of the big renderers (StableHLOPretty `emitTok` arms, IRPrint string bodies, ResNet/MNv2/MNv4/ENet/ViT/ConvNeXt traversals), which I only opened to check a specific claim. Every backticked identifier in the slice was checked against the repo's declarations. Escape hatches: `grep sorry|admit|native_decide|implemented_by|@[extern]|^axiom` over the slice finds none. I did not run `tests/AuditAxioms.lean`.

---

### LeanMlir/Proofs/Codegen/StableHLOParse.lean:5-29, 216-220 — module docstring + `roundtrip`

**Kind:** overclaim
**Says:** (module) "`StableHLO.lean` closes the **semantic** half of R4: `den (emit g) = fderiv`. … What this buys: the *structure* of the emitted graph (which op, which operands, which shapes) is now **proven** recoverable — it leaves the trusted surface." (`roundtrip`) "The emitted op-graph (skeleton) of any `SHlo` is a faithful, recoverable serialization … The op structure / shapes / SSA names leave the trusted surface; only the per-op `Tok ↔ StableHLO-text` lexing stays audited."
**Actually states:** `parse (toToks (skel a)) = some (skel a)`. That is a round trip between two Lean data structures, `Raw` and `List Tok`, and it holds for any injective postorder encoding. It says nothing about `pretty` or the emitted text. The text comes from `serializeToks`/`emitTok`, which decide how many operands each op pops, the order they appear in the MLIR line, which names are printed, and which shapes are printed. None of that is constrained by `roundtrip`. So operand order, arity and shapes in the *text* are still trusted, in the same place as the lexical syntax. Also, no theorem of the form `den (emit g) = fderiv` exists. `den` faithfulness is stated against named ℝ functions (`flatConv`, `adamWParam`, …), mostly by `rfl`. The module docstring also names `parse_skel`, which does not exist (the lemma is `parse_toToks`).
**Fix:** "`roundtrip`: the postorder token encoding of a skeleton is invertible (`parse (toToks (skel a)) = some (skel a)`). It is a statement about `toToks`, not about the text: which operands each op's emitted line reads, in what order, and with what types is decided by `emitTok` and remains trusted along with the lexical syntax." Replace "`den (emit g) = fderiv`" with "`den g` equals the named ℝ reference (`*_faithful`)", and replace `parse_skel` with `parse_toToks`.

### LeanMlir/Proofs/Codegen/StableHLOPretty.lean:21-22 (module) and :4646-4650 (`pretty`)

**Kind:** overclaim
**Says:** (module) "**Trusted residue:** the text's lexical conformance to the StableHLO spec is checked by execution (`iree-compile` / PJRT) and by the `StableHLOParse` round-trip, not proved." (`pretty`) "The emitter shares ONE structured form with the parser, so the round-trip `parse (toToks (skel a)) = skel a` … is about the very tokens this prints — the printer can't structurally drift."
**Actually states:** `roundtrip` never looks at text, so it checks no property of the text's lexical conformance. `pretty` = `serializeToks B (toToks (skel g))`. Each `emitTok` arm is free to pop the wrong number of names, print operands in any order, or fall through to `"// MALFORMED …"`, and `pretty` returns `"%MALFORMED"` on a bad stack. The round trip cannot rule out any of these. (The docstring also drops the `some` in the round-trip statement.)
**Fix:** module: "…is checked by execution (`iree-compile` / PJRT) and by the byte-diff guards on committed artifacts; the `StableHLOParse` round-trip covers only the token encoding, not the text." `pretty`: "…so the tokens it prints are exactly `toToks (skel g)`, whose encoding `StableHLOParse.roundtrip` shows is invertible; how each token becomes text (`emitTok`) is trusted."

### LeanMlir/Proofs/Codegen/StableHLO.lean:3176, 3633, 3667, 3681, 3695, 3720, 3734 — `bnPerChannelF_faithful`, `depthwiseF_faithful`, `swishF_faithful`, `sigmoidF_faithful`, `geluF_faithful`, `softmaxRowF_faithful`, `matmulF_faithful`

**Kind:** overclaim
**Says:** "(`rfl`, so kept out of the axiom audit — `roundtrip` covers it structurally.)"
**Actually states:** Each is `den (.xF …) = <ℝ function> (den e) := rfl`. `roundtrip` is about `toToks ∘ skel` and has no bearing on whether this op's emitted text matches this `den` arm, or on its axioms.
**Fix:** "(`rfl` — `den`'s arm is this function by definition.)" Drop the `roundtrip` clause. If axiom coverage matters, add the lemma to `tests/AuditAxioms.lean` instead of justifying its absence.

### LeanMlir/Proofs/Codegen/StableHLO.lean:33, 40-41 — module docstring

**Kind:** overclaim
**Says:** "Every verified artifact in `verified_mlir/` is `pretty` of a term of one typed AST, `SHlo`. … SSA names are annotations `den` ignores, so the rendered program and the denoted one are one object."
**Actually states:** Every train-step artifact also contains hand-written text outside any `SHlo` term. Examples: the report-only `%loss` block (MlpRender:66-79, CnnRender:89-101, …); ConvNeXt's GAP-backward block (`%dgi…%dgapf`, ConvNeXtRender:768-772, ConvNeXtRenderB:587-591); the SGD `%dy` divide (ConvNeXtRender:736); the stem `psW` SGD wrap `sgdOf` (ConvNeXtRender:248, 808); `%bc1/%bc2` passthroughs; signatures and constants. `pretty` is a trusted printer, so "one object" holds only for the term, not for the text.
**Fix:** "Every verified artifact's computational body is built from `pretty` of `SHlo` terms, plus declared text carve-outs (report-only `%loss`, `%bc` passthroughs, ConvNeXt's GAP-backward block, …) listed in each renderer. `pretty` is trusted: the proofs are about the term `den` reads, not about the text."

### LeanMlir/Proofs/Codegen/ConvNeXtRender.lean:4-11, 22, 40-42 (module), 812-819 (`convNextTrainStepFaithfulV`), 920-927 (`convNextAdamTrainStepFaithful`)

**Kind:** overclaim + stale
**Says:** (title) "ConvNeXt-T train step rendered ENTIRELY from the verified AST". "Since 4c leg 3 … this file writes ONE artifact: the SGD-inline `verified_mlir/convnext_train_step.mlir`." "**The two weight-gradient residuals are CLOSED (2026-07-28); all 180 params are now SHlo ops.**" (`convNextTrainStepFaithfulV`) "(except the two documented weight-grad gaps — the stem 4×4/s4 patchify and the even-kernel 2×2/s2 downsample, neither of which has a VJP-cert `SHlo` op)". (adam) "all 180 params are `pretty(AST)` end to end … Still outside the AST here, and unchanged: `%loss`".
**Actually states:** The one artifact this file writes, the SGD step, still contains hand-written text. The stem weight update is `sgdOf nPsW "psW"` (a hand-written `multiply`/`subtract`; line 808, and the inline comment at 790-795 admits "SGD is certified-gradient + hand-written-update there"). The `%dy` divide is hand-written (736), and so is the GAP-backward block (768-772). The AdamW render also carries the GAP-backward block, which ConvNeXtRenderB:523-526 calls "hand-written text on both sides … one of §5's declared non-AST carve-outs". The `convNextTrainStepFaithfulV` docstring's "two gaps with no VJP-cert op" is stale: `.convStride4WeightGrad` and `.convStridedWeightGrad` at 2×2 are both certified now. The count is also stale. The head LN added two parameters, so it is 182 (`allParams` docstring line 540, "180 of the 182"; `convnext_fwd.mlir` takes 183 args). As a result, "180-parameter signature … (181 inputs)" (675-676), "545 in / 543 out … 180 θ" (916-917) and "all 180 params" (22, 920) are all stale; the AdamW artifact has 551 args.
**Fix:** Title: "ConvNeXt-T train step rendered from the verified AST (with declared carve-outs)". Replace the ⭐ line with: "All 182 parameter *gradients* are `SHlo` ops. In the SGD render the stem weight's update is a hand-written `θ − lr·g` (`sgdOf`), and both renders carry the hand-written GAP-backward block and (SGD) the `%dy` divide." In `convNextTrainStepFaithfulV`, replace the parenthetical with the same carve-out list. Update 180/181/545/543 to 182/183/551/549 (check the artifact).

### LeanMlir/Proofs/Codegen/ConvNeXtRender.lean:904 (emitted into 10+ committed artifacts)

**Kind:** stale
**Says:** (emitted MLIR comment) "`// ── timm no_weight_decay (wdExcludeNormBias): 121 of 180 params take %wdz, not %wd ──`"
**Actually states:** The file's own comment at 586-589 says the reference reports "182 tensors: 59 decayed, 123 excluded", and that the count "Was 180/59/121 before 2026-08-30". The line is baked into `convnext_adamwx`, `convnextin_*wx*`, and `convnextbin_*wx*`. ConvNeXt-B has 36 blocks, so its count differs again.
**Fix:** Derive the text from `cnxWdCounts nClasses V` rather than a literal. This re-renders the `wx` artifacts, so it needs sign-off.

### LeanMlir/Proofs/Codegen/MlpRender.lean:20-26 (`mlpTrainStepFaithfulV`); CnnRender.lean:21-31, 110-117, 197-203 (`cnnTrainStepFaithfulV`, `cifarTrainStepFaithfulV`, `cifar8TrainStepFaithfulV`)

**Kind:** overclaim
**Says:** "are all `pretty` of denoted `SHlo` nodes — so every emitted line is `pretty(provenNode)`" / "Every emitted line is `pretty(provenNode)`". The committed `mlp_train_step.mlir` opens with the banner "every line is pretty(verified AST node)".
**Actually states:** Each of these renderers appends a 12-line hand-written report-only `%loss` block (MlpRender:66-79; CnnRender:89-101 and peers). The artifact itself marks it "`%loss below is REPORT-ONLY (logging), NOT pretty(AST node)`", so the banner contradicts the file it heads. The AdamW docstrings in the same file (CnnRender:404-407) state this carve-out correctly.
**Fix:** "Every line that feeds a returned parameter is `pretty` of a denoted node; the appended report-only `%loss` block is hand-written and feeds nothing." Apply the same wording to the emitted banner.

### LeanMlir/Proofs/Codegen/MobileNetV4RenderB.lean:9-17 — module docstring, the UIB diagram

**Kind:** stale (prior overclaim group 4, MNv4 parity)
**Says:**
```
  optional pre-DW (k×k)  → BN → relu      -- takes the block's stride
  ...
  optional post-DW (k×k) → BN → relu      -- at stride 1 if a pre-DW already consumed it
```
**Actually states:** Since the timm-parity fix (90e4af7e), the pre-DW is BN only (no relu) and never takes the stride. The post-DW (`dw_mid`) carries it: `uibFwdStridedB` (268-312) runs the pre-DW and the expand at `2h` and applies `.depthwiseStridedAt` on the post-DW. The same docstring says so 40 lines later (line 55: "the pre-DW and the project are BN only"; line 41: "the stride rides the post-DW"). The diagram at the top, which is what a reader sees first, describes the pre-parity net.
**Fix:**
```
  optional pre-DW (k×k)  → BN             -- stride 1, NO activation (timm dw_start)
  expand 1×1 (ic → mid)  → BN → relu
  optional post-DW (k×k) → BN → relu      -- carries the block's stride (timm dw_mid)
```
Also line 31 cites `cnx_render_dw7*_certified`, which no longer exists (retired in 31f76507/18c31142). Cite `depthwiseBack_faithful` / `depthwiseFlatHasVJP` (kernel-generic) instead. Line 491, "Block dispatch is forced by the table and checked by the types", contradicts line 62 ("the four families are `if`s that the compiler cannot check"). Keep the latter.

### LeanMlir/Proofs/Codegen/EfficientNetRenderPC.lean:7-9, 21-27 — module docstring

**Kind:** overclaim + stale
**Says:** "MNV2/r34 get away with a batch-1 `den` because their per-channel BN reduces `[2,3]` (per-example, separable); EfficientNet's does not." … "We prove the FORWARD half — `den (graph) = forward` — for every block form B0 has: the stride-2 stem …, an MBConv1 …, an MBConv6 … **stride-2** downsample, an MBConv6 … **identity residual** skip, and the 1×1 conv-bn-swish head".
**Actually states:** MobileNetV2 and ResNet-34 both render batch BN at `N := B` now (MobileNetV2RenderB:28-30, ResNet34RenderB:11-19), so the first sentence describes retired renderers. B0 also has the stride-1 expand **no-skip** block (b9/b16, `eFwdNoSkip`). Its graph and faithfulness (`mbExpGraphB` / `mbExpGraphB_faithful`) live in `Nets/EfficientNet/EfficientNetFullB0.lean:143-158`, not in this file, which proves five forms. The Eval twin (EfficientNetRenderPCEval:5-8, "the five block graphs") has the same gap.
**Fix:** "Every net's batch-BN render reduces over `[0,2,3]`, which couples the batch, so these graphs live at `N·(c·h·w)`." And: "This file proves four of B0's five block forms plus the stem and head; the stride-1 no-skip expand block is `mbExpGraphB_faithful` in `EfficientNetFullB0.lean`."

### LeanMlir/Proofs/Codegen/ResNet34RenderB.lean:236-246 — `resnet34FwdEvalFaithfulV`

**Kind:** stale
**Says:** "the eval partner of a **batch**-statistic train step … `resnet34_adam_train_step.mlir`, which is still a hand-written render in `TestResnet34Train.lean` (since retired). So the eval forward is now certified while the train step it partners is not; that asymmetry is the remaining §2a work".
**Actually states:** This file is "The sole writer of every ResNet-34 artifact" (module docstring, line 7), including `resnet34_adam_train_step.mlir` via `resnet34AdamTrainStepFaithfulB`. `tests/TestResnet34Train.lean` no longer exists.
**Fix:** "…the eval partner of `resnet34AdamTrainStepFaithfulB` (this file), whose returned batch mean/var the driver EMAs into exactly these slots."

### LeanMlir/Proofs/Codegen/ViTRenderB.lean:262-265 — `vitFwdRenderB`

**Kind:** stale
**Says:** "Not written to `verified_mlir/`: it exists to be TIED against the committed per-example artifact … The writer lands with the swap, not before."
**Actually states:** The swap landed, and this function writes `vit_fwd.mlir` (line 995), `vit_drop_fwd`, `vitin_fwd`, `vitsin_fwd`, `vitsin_drop_fwd` and `vitbin_fwd` (659-1163), as the module docstring says ("the sole writer of every ViT artifact but one").
**Fix:** "`@vit_fwd` and its ImageNet/size/SD peers, rendered from the batched chain; the writer of every committed ViT forward. `vit-fwd-b-tie` renders the per-example chain against these bytes."

### LeanMlir/Proofs/Codegen/ViTRender.lean:4-15 — module docstring

**Kind:** stale
**Says:** "# ViT-Tiny train step rendered from the verified AST (the §1 render) — FORWARD portion … This file is the FORWARD half of the §1 train-step render; the backward-cotangent chain … + the param-SGD tail … follow."
**Actually states:** The file holds the whole backward (`vBlockBack`, `vitBackAll`), `vitTrainStepRenderV` and `vitAdamTrainStepFaithful`, and it writes `vit_train_step.mlir` (line 876). Per ViTRenderB's module docstring, it is now the writer of that one SGD artifact only.
**Fix:** "ViT-Tiny/S/B per-example render: the forward chain, the shared backward traversal `vitBackAll`, and the AdamW tail `vitAdamTrainStepFaithful` (which `ViTRenderB` reuses). It writes only `vit_train_step.mlir`, the SGD-inline step that `ViTStepTie` is stated at."

### LeanMlir/Proofs/Codegen/IRPrint.lean:418-421 — `convBackModule`

**Kind:** overclaim
**Says:** "Conv input-gradient backward `IR.convBackDenote W` as `@conv_back` … Denotes the proven conv input-VJP (`conv_back_bridge_1to2`)."
**Actually states:** `convBackModule (B ic oc H Wd kH kW)` is shape-generic, but `conv_back_bridge_1to2` is stated only at `W : Kernel4 2 1 3 3` on a 4×4 map (IR.lean:293). The general result, `convBackDenote_eq_input_grad_formula` (IR.lean:279), needs odd `kH`, `kW`, a hypothesis the docstring drops.
**Fix:** "Denotes the proven conv input-VJP for odd `kH`, `kW` (`IR.convBackDenote_eq_input_grad_formula`; `conv_back_bridge_1to2` is its 1→2-channel 3×3 instance)."

### LeanMlir/Proofs/Codegen/IRPrint.lean:1-24 — module docstring

**Kind:** stale + process-narrative
**Says:** "# Phase 0 of `planning/archive/verified_codegen.md` — `Back → StableHLO` printer … So `mlpHlo` below mirrors `IR.emitMlpBack` … (Phase 1: feed the output to IREE.)"
**Actually states:** The file is 1,883 lines. It covers linear/MLP/CNN train steps, BN/LN, softmax, SDPA, activations, SE, a ViT block, a ResNet train step, MBConv, the MobileNetV2 inverted residual and a ConvNeXt block (section banners "Phase 3 …"). Its `#eval`s write `/tmp/*.mlir` on every elaboration. The training engine is PJRT/XLA, not IREE. The section note at 486 ("the conv weight-gradient + a full CNN train step is the next step") is followed by `cnnTrainStepModule` at 554.
**Fix:** "Legacy hand-written `Hlo`/`String` printers that mirror `IR.lean`'s `Back`/`Fwd` graphs, one module per layer family. The text is trusted; each docstring names the `IR` bridge its op sequence mirrors. The verified renderers are `StableHLOPretty`/`*Render*.lean`; the `#eval`s at the bottom write scratch modules to `/tmp`." Drop the Phase labels and the "next step" note.

### LeanMlir/Proofs/Codegen/StableHLO.lean:1829-1831 (comment in `den`) and StableHLOPretty.lean:230-231 (comment in `Raw`)

**Kind:** stale
**Says:** "`gradSumSqF` collapses one parameter's gradient … `addScalarF` folds those across parameters; `gradClipFacF` roots the total and forms `min(1, c/(√s+ε))`; `clipScaleF` multiplies…" / "`gradClipFacF` keeps only its two literal strings …; `clipScaleF`/`addScalarF` are BINARY."
**Actually states:** None of `gradSumSqF`, `addScalarF`, `gradClipFacF` exists. StableHLO.lean:1421-1428 records that the four-op split was RETRACTED. The ops are `gradSumSqAccF` (binary, accumulating) and `clipScaleF`, which forms the factor itself.
**Fix:** "`gradSumSqAccF acc g` adds `∑g²` to the running rank-0 total; `clipScaleF s g` forms `min(1, c/(√s+ε))` from the total `s` and scales `g`." In Pretty: "`clipScaleF` keeps its two literal strings; `gradSumSqAccF`/`clipScaleF`/`lambScaleF` are BINARY."

### LeanMlir/Proofs/Codegen/StableHLO.lean:1784-1786 — `den`

**Kind:** stale
**Says:** "**AST denotation `⟦·⟧ₐ`** — our reading of each StableHLO op's spec, over `ℝ`, per-example, in primitive terms".
**Actually states:** `den` also has batch-coupled arms (`bnBatchF`, the `*GradB` batch sums, `bnSync*`) and a cross-replica arm (`allReduceMeanF`, the mean over `R` replicas). The file's own theorems (`den_rowDenseBiasGradB_at_one`, `den_posEmbedGradB_at_one`) exist because those arms are NOT per-example.
**Fix:** "…over `ℝ`, in primitive terms: per-example for the per-example constructors, and at the batched index `N·n` (`batchOp`, `*B`, `*GradB`) or across replicas (`allReduceMeanF`) for the rest."

### LeanMlir/Proofs/Codegen/StableHLO.lean:3423-3428 — `adamWParamF_faithful`

**Kind:** overclaim
**Says:** "The emitted 26-op block denotes exactly `Proofs.adamWParam` … — the theorem that moves the optimizer from a trusted hand-written emitter (`ViTRender.emitAdamV`, which only *claimed* to be op-for-op `adamWParam`) into the proven kit."
**Actually states:** `den (.adamWParamF …) = adamWParam … (den e) := rfl`, true because `den`'s arm is defined as `adamWParam`. The 26 emitted lines are still produced by a hand-written `emitTok` arm, so the correspondence between that text and `adamWParam` has the same trusted status as before. What moved is only that the graph node now has a `den`.
**Fix:** "The node's denotation is `Proofs.adamWParam` (by definition of `den`), so ties stated over `den` can use it; the 26-op text `emitTok` prints for it is trusted, like every op's."

### LeanMlir/Proofs/Codegen/ResNet50RenderB.lean:40-46 — module docstring §"There is no incumbent…"

**Kind:** stale
**Says:** "R50 has no such artifact, so that license does not exist here and **must not be implied**. The substitutes are the layer-level VJP oracle … and a keep-1 known-answer check".
**Actually states:** R50 now has a Proofs tier stated at this render: `Nets/ResNet/ResNet50FullB.lean`, `ResNet50StepTieB.lean` and `ResNet50SyncStepTieB.lean` import this file. The module docstring never mentions them, while every peer renderer's module docstring names its Proofs tier.
**Fix:** Keep the "no hand-written incumbent" caveat and add: "The Proofs tier stated at these bytes is `ResNet50FullB` (T1/T2) and `ResNet50StepTieB` / `ResNet50SyncStepTieB` (T3 §1a)."

### LeanMlir/Proofs/Codegen/EfficientNetRender.lean:599-600 — `eBackNoExp`

**Kind:** stale (draft text left in a docstring)
**Says:** "8 params (Wd bd gd btd zW1 zb1 zW2 zb2 ... wait, 4 dw + 4 SE + 4 proj = 12)."
**Actually states:** The no-expand block has 12 parameters (depthwise W/b/γ/β, four SE, project W/b/γ/β), per `enetSig`'s "b1 no-exp(12)".
**Fix:** "**No-expand MBConv backward** (b1): project back → SE back → depthwise back → dx. 12 params: depthwise W b γ β, SE W₁ b₁ W₂ b₂, project W b γ β."

### LeanMlir/Proofs/Codegen/CnnRender.lean:634-639 — `cifar8AdamTrainStepFaithfulB`

**Kind:** stale
**Says:** "Both families denote the SAME proven VJP — `StableHLO.lean` l.2016 vs l.2200 are `(conv2dHasVJP3 W b).backward v …` … and l.2990 records why it is free".
**Actually states:** Those lines are now a bf16 comment (2016), an unrelated `batchSlice` term (2200) and the `mlpFwdGraph` docstring (2990). The cited `den` arms are at 2076 (`.convBack`) and 2268 (`.convBackBatched`), and "conv is linear, so this is a global VJP" is at 3079 (`convBack_faithful`).
**Fix:** Cite declarations, not lines: "`den`'s `.convBack` and `.convBackBatched` arms …; `convBack_faithful` records why…".

### LeanMlir/Proofs/Codegen/StableHLO.lean:1264 — comment on the un-fused gradients

**Kind:** stale
**Says:** "`den (xSgd …) = θ − lr · den (xGrad …)` is `rfl`, see the `_sgd_eq` theorems."
**Actually states:** No `*_sgd_eq` theorem exists. They are `weightSgd_eq_grad`, `rowDenseWeightSgd_eq_grad`, … (`*Sgd_eq_grad`, line 3205 onward).
**Fix:** "…see the `*Sgd_eq_grad` theorems."

### LeanMlir/Proofs/Codegen/RenderKit.lean:135-140 — `r34WdDecays`

**Kind:** stale / process-narrative
**Says:** "⚠⚠ **This is `a3_paper_fidelity.md` §2.1, open since the A3 run.** The live A3 artifact has ZERO `%wdz` occurrences against ConvNeXt's 123 — so the 77.43% run decayed BN γ/β…".
**Actually states:** The `wx` R50 renders exist and exclude correctly. For example, `resnet50in160_lambaccdp8x64wxclipbce_train_step.mlir` has 109 `%wdz`. The "open" status is out of date, and a run history does not belong in the docstring of a one-line predicate.
**Fix:** "timm's `no_weight_decay` rule: every rank-1 parameter (BN γ/β, biases) is excluded. Identical to `cnxWdDecays` because the rule is timm's, not the net's." Move the A3 history to the commit log or planning doc.

### LeanMlir/Proofs/Codegen/FwdGraphTextTies.lean:17-19, 30-31 — module docstring

**Kind:** overclaim (mild)
**Says:** "for every block kind, the renderer's block emitter and `pretty` of the T2 block graph … print the same bytes … Covered: ResNet-34, ResNet-50, MobileNetV2, MobileNetV4-Conv-M and EfficientNet-B0 — every block kind, stem and head."
**Actually states:** Each check is a `#guard` at one concrete instance: batch 2, one channel/spatial choice per kind (for example `idFwdB 2 64 56 …`, `bnkStridedFwdB 2 512 256 1024 14 …`). MNv4 checks all 21 table rows. None of these is a statement for all shapes.
**Fix:** "…print the same bytes, checked by `#guard` at batch 2 on one representative shape per block kind (every row of the MNv4 table)."

### LeanMlir/Proofs/Codegen/StableHLOLex.lean:3-71 — module docstring

**Kind:** process-narrative (+ minor stale)
**Says:** "## Honest scope of the remaining lexer (corrects the planning doc)" … "~90 `Tok` constructors" … "## Status: deliberate STOP, not work-in-progress (decided 2026-06-27)".
**Actually states:** The file holds one def and one theorem (`parseNat_toString`). `Tok` now has 98 constructors. The rest is planning history.
**Fix:** "Decimal `Nat ⟷ String` round trip (`parseNat_toString`), the numeric piece a text lexer for `pretty`'s output would need. No lexer is built; the text/token correspondence is trusted and guarded by byte-diffing committed artifacts." Move the scoping analysis to `planning/archive/tier23_…`.

### Process narrative in module/declaration docstrings (grouped)

**Kind:** process-narrative
These module docstrings are organised around history ("4c leg 3 (2026-09-07)", "§2a-quater", "Checked 2026-07-29 and retired", "handoff §0.2 ▶2", "Until 2026-09-24 …", "(Corrected 2026-09-06: this docstring and four below said 146 unconditionally)"):
- ConvNeXtRender.lean:4-43, ConvNeXtRenderB.lean:3-34 (the title says "the forward" but the body says the file writes forward, backward and AdamW), ViTRenderB.lean:3-45, MobileNetV2RenderB.lean:5-52, ResNet34RenderB.lean:5-34 (present-tense contrast with the retired `ResNet34Render.lean` and a "numeric tie against the hand-written artifact" that no longer exists), ResNet50RenderB.lean:5, LambTriple.lean:12-15 ("The audit's 'LAMB has NO faithfulness theorem' was wrong…") and :58, StableHLO.lean:3957-3972 (`biasName`: an 80-epoch measurement).
- Declaration docstrings citing results that are gone: EfficientNetRender.lean:1060-1063 ("matching README's 87.58%"; README no longer has that number) and ResNet50RenderB.lean:505-508 ("keeps 89.86% (`runs/r50_imagenette_adam_80ep.log`)"; that log does not exist).
**Fix:** In each, state what the file writes, its Proofs tier, and its declared carve-outs. Move leg/§/date history to commit messages.

### Small inconsistencies

**Kind:** stale
- ConvNeXtRenderB.lean:29 says `TestBatchedEmitTie.lean` pins "31 forms". ViTRenderB.lean:42 says "all 47". One of them is stale. The test prints `cases.length`, so cite that rather than a number.
- StableHLO.lean:419-420 (constructor comment): "`maxPoolBack` … Conditional (no-ties) like the ReLU kink." This is prior overclaim group 1 in miniature. The ReLU kink is a measure-zero set of pre-activations, but the max-pool sits after ReLU, where tied zeros in a window are routine. Suggested wording: "Conditional on `MaxPool2Smooth` (all window entries distinct), which fails whenever a window holds two post-ReLU zeros."

---

### Leads for the correctness pass (not documentation findings)
- ViTRender.lean:579. `vitTrainStepRenderV`'s return-type list ends `ty [10]` for the classifier bias while the argument signature uses `ty [nClasses]`. At `nClasses ≠ 10` the declared result type would not match the returned value. Only the 10-class SGD artifact is written today.

---

## Overclaims (fix before anyone reads the results again)

1. StableHLOParse module + `roundtrip` docstring: "structure / operands / shapes / SSA names leave the trusted surface". `roundtrip` is token-level and says nothing about emitted text. Also cites a nonexistent `den (emit g) = fderiv` and `parse_skel`.
2. StableHLOPretty module ("lexical conformance … checked … by the `StableHLOParse` round-trip") and `pretty` ("the printer can't structurally drift"). The round trip does not touch `emitTok`.
3. StableHLO.lean, 7 `*_faithful` docstrings: "`roundtrip` covers it structurally".
4. StableHLO module: "Every verified artifact in `verified_mlir/` is `pretty` of a term … the rendered program and the denoted one are one object". Hand-written carve-outs are in every train step.
5. ConvNeXtRender: "rendered ENTIRELY from the verified AST" / "all 180 params are now SHlo ops". The SGD artifact has a hand-written stem update, `%dy` divide and GAP-backward, and the count is 182.
6. MlpRender / CnnRender (4 renderers) and the `mlp_train_step.mlir` banner: "every emitted line is `pretty(provenNode)`". There is a hand-written `%loss` block.
7. EfficientNetRenderPC: "for every block form B0 has". The no-skip expand form is proved elsewhere (`EfficientNetFullB0.mbExpGraphB_faithful`).
8. IRPrint `convBackModule`: a shape-generic module cited to a fixed-shape (`Kernel4 2 1 3 3`, 4×4) bridge, dropping the odd-kernel hypothesis of the general lemma.
9. `adamWParamF_faithful`: "moves the optimizer … into the proven kit". It is `rfl` on `den`'s definition, and the 26-line text is still a hand-written `emitTok` arm.
10. FwdGraphTextTies: "for every block kind … print the same bytes". These are `#guard`s at one instance per kind (mild).
