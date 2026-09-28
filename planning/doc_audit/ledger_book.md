# ledger_book.md — api_docs_followups §4 (the book pass), one row per changed site

Working tree at `db99bdee` (+ this pass). Line numbers are content.tex at `db99bdee`. Recounts,
run 2026-09-28:

* `grep -c '^#print axioms' tests/AuditAxioms.lean` → 1,766 (Heavy 62). The book quotes no
  number (l.16964 says "that file is the live count"); the followups' 1,610 is stale but unused.
* comparator: 13 + 39 + 35 = 87 (`python3 -c "…len(json.load(open('tests/comparator/<cfg>.json'))['theorem_names'])"`).
* README.md carries no "87.58" (B0 rows read 89.96 Imagenette / 76.88 ImageNet) — item dropped.
* "4× accumulation" for ViT: no such phrase in content.tex, README.md or LeanMlir.lean (the ViT
  recipe says $4 \times 128$ replicas) — item dropped.
* B0 "prose still says 10 classes": no such sentence in the EfficientNet chapter (8166–9310); the
  step-tie block never named a class count. `efficientnet_net_tiedG … {nCls : Nat}` binds it.
  Nothing to change in the book; the §5 row's Lean-docstring sites are not this pass.
* ConvNeXt-S/B (10427–10467) makes no VJP or tie claim; `thm:convnext_whole_back` already says
  `convnextImagenetInputGradB_eq_vjp` is "the 1000-class instance" of ConvNeXt-T. Nothing to change.

## Round 1 — text-only sites

### Theme 1: the pooling condition (§4.A pooling, §4.B.4)

| Site | Old claim | Statement read | New text | Checked by |
|---|---|---|---|---|
| 3665–3667 (ch 3 MLIR caveats) | "Max-pool needs a unique argmax … strict argmax. The tie set is the codegen trust boundary" | `maxpool_back_bridge` (Foundation/IR.lean:322) takes `h_smooth : MaxPool2Smooth x`; `MaxPool2Smooth` (CNN.lean:893) = every 2×2 window's max strictly above every other cell; `mnistCnnNoBnHasVJPAt` has `h_mp` (book l.3222) | smooth-point bridge at `MaxPool2Smooth`; other cells may tie; all-zero post-ReLU window fails; whole-net theorem states it as `h_mp` | read both defs; `maxPool2Smooth_of_pairwise` keeps the old discharge |
| 17120–17128 (On Verification, "The one conditional") | "an argmax tie for max-pool … that measure-zero set" | same; `MaxPool3s2Smooth` (MaxPool3s2.lean:161) for the stem pool; `R34SmoothAtB` carries "stem pool tie-free" (ResNet34ParamGrad.lean:754) | ReLU set has measure zero; pooling's does not (all-zero window); whole-net theorems carry the clause | yaml fidelity (2)–(3) at `db99bdee`, reused |
| 2726, 7444, 17193, 17361 | ReLU / ReLU6 / generic "measure-zero" | — | left as written (the plan: ReLU/ReLU6 may stay) | — |

### Theme 2: per-layer step, loss form, cotangent ties (§4.A l.17101, §4.B.1 prose, §4.B.2)

| Site | Old claim | Statement read | New text | Checked by |
|---|---|---|---|---|
| 17100–17101 | "proves equal to the certified loss-descent step, output by output" | `mlp_train_step_tied_certified` (MlpFold.lean:169): update = `W − lr · pdiv(loss)` for the head, per-layer VJP at the chain cotangent for hidden layers (book l.2359–2364) | "equal to the certified per-layer step: θ − lr·(that layer's certified Jacobian · the chain cotangent)" | read the statement |
| 17108 | "All twelve chapter networks are tied this way" | the `\lean{}` tie capstones in content.tex: linear, mlp, cnn, cifar, cifar8, cifar8Bn, r34, r50, mnv2, mnv4, enet, cnx, vit | "thirteen" | `grep -n '\\lean{[^}]*\(net_tied\|train_step_tied\|train_step_tail\)'` → 13 lines |
| 17109–17111 | "each deep net's rendered block-backward is pinned to that block's certified VJP (vitBlockBackV_eq_transformerBlockV_vjp …)" | `r34IdCotIn_eq_vjp` (ResNet34StepTieB.lean:123), `vitBlockCotInB_eq_vjp` (ViTStepTieGB.lean:531) | "each deep net's chain cotangent is pinned to its block's certified VJP (r34IdCotIn_eq_vjp, vitBlockCotInB_eq_vjp …)" | names grepped |
| 17111 (new sentence) | — | `r34_net_lossGrad` (ResNet34ParamGrad.lean:757) and the six peers; each `hL : HasGradAt L (net) g` except R34's, stated directly at `smoothedBatchLoss`; `*_lossGrad_smoothedCE` per net, R50 also `_bce`; scope per docstrings: one replica, f32, drop-free | "For the seven ImageNet networks one more theorem …: every parameter gradient node is ∂L/∂θ of the whole network, for any L with a gradient at the logits …" | signatures read (§4.B.1 blocks are round 2) |
| 5252–5258 R34 step tie | no cotangent clause | `r34IdCotIn_eq_vjp`, `r34DownCotIn_eq_vjp`; tie docstring (ResNet34StepTieB.lean:410–413): capstone needs neither `0 < ε` nor a kink condition, those enter in the `_eq_vjp` lemmas | clause added | docstring + names |
| 5276–5281 R50 step tie | same | `r50{Id,Proj,Down}CotIn_eq_vjp` (ResNet50StepTieB.lean:137/221/312); docstring l.495–499 | clause added | same |
| 7048–7054 MNv2 step tie | same | `mnv2{NoExp,ExpOnly,Resid,Strided}CotIn_eq_vjp`, `mnv2StemCotC_eq_vjp`, `mnv2HeadCotBlk_eq_vjp` (MobileNetV2StepTieB.lean:156–454); docstring l.692–695 | clause added ("the four block kinds' *CotIn_eq_vjp, stem, head") | same |
| 7071–7075 MNv4 step tie | same | `mnv4{Body,SBody,Skip,Fused,Head}CotIn_eq_vjp` (MobileNetV4StepTieB.lean:723–809); docstring l.943–946 | clause added | same |
| 9705–9710 ConvNeXt, 11738–11741 ViT | same (folded into the Theme 3 rows below) | `cnx{BlockCotIn,DownCotIn}B_eq_vjp`, `cnxHeadDyB_eq_vjp` (ConvNeXtStepTieGB.lean:387/401/412); `vitBlockCotInB_eq_vjp`, `vitCotB2outB_eq_vjp` (ViTStepTieGB.lean:531/546) | clause added | names grepped |
| B0 step tie | no clause added | the B0 chain is built from the block witnesses' `.backward`s directly (EfficientNetStepTieG.lean:408–416); no separate `_eq_vjp` | — | docstring |

### Theme 3: tie scope (§1(a), §1(d)), the quoted bf16/drop renders, "uniquely unconditional"

| Site | Old claim | Statement read | New text | Checked by |
|---|---|---|---|---|
| 8479–8485 B0 step tie | "and every efficientnetin_* artifact"; "the target a binder" | `efficientnet_net_tiedG … {nCls} (w : B0Weights nCls) (hεw : w.EpsPos) … (t)` (EfficientNetStepTieG.lean:417); docstring 408–416: without drop-path or classifier dropout, one replica's `*GradB`, bf16 `*GradBBf16` outside | "the f32 efficientnetin_* steps"; "the target and the number of classes binders"; "at one replica, on the chain without stochastic depth or classifier dropout"; bf16 clause | signature + docstring; artifact list `ls verified_mlir/efficientnetin_*` (bf16: rms64bf16, rmsdp64bf16, emarmsdp64dropdobf16, …eps0001bf16) |
| 9705–9710 ConvNeXt step tie | "and of every convnextin_* artifact" | ConvNeXtStepTieGB.lean:462–473: `*GradB` nodes of `convnext_adam_train_step.mlir` and every f32 `convnextin_*`; bf16 not covered; one replica; drop-free chain, `*drop*` nodes are the same constructors at a chain carrying `dropPathB` sites | as the docstring says, plus the three cotangent lemmas | docstring; `ls verified_mlir/convnextin_*` |
| 11738–11741 ViT step tie | "Every vitin_* artifact renders from this chain" | ViTStepTieGB.lean:582–593: same scope as ConvNeXt (f32 `vitin_*`, bf16 not covered, one replica, drop-free, ViT-Tiny dims) | scope stated; drop and bf16 clauses; two cotangent lemmas | docstring; `ls verified_mlir/vitin_*` |
| 12300 ViT recipe (`vitin_emadp128x4wxclipdropbf16`) | no scope sentence next to the quoted render | as above | "Theorem vit_step_tie is stated on the f32, drop-free chain; this render's gradient nodes are the bf16 *GradBBf16 kinds, tied per operator, at a cotangent chain that carries the drop-path sites" | Bf16GradNodes: nine `*GradBBf16` kinds tied per op (LeanMlir.lean landing page) |
| 10220 ConvNeXt recipe (`convnextin_adamdpwxclipdropbf16`) | same | as above | same sentence for `thm:convnext_step_tie` | same |
| 9993 ConvNeXt MLIR | "Uniquely in this book, every one is unconditional" | B0: `efficientnetForwardBFullHasVJP` global, "Assume the fifty BatchNorm positivities" (book 8427); ViT: `vitForwardKVHasVJP` "only 0 < ε" (ViTDepthK.lean:170); both kink-free | "Every one is unconditional …: ConvNeXt has no kinked operator, and neither do EfficientNet-B0 and the Vision Transformer" | LeanMlir.lean:103 "The last three take only 0 < ε hypotheses" |
| 6073 R34 ImageNet | "The proof-carrying tier stops at Imagenette" | `r34_net_tiedB` names `resnet34in_momdp64` at N = 64 (book 5252); `r34_net_syncTiedB` names it at replicas > 1 (book 5334); the run's render is `momdp64bf16` (book 6028); bf16 renders emit `*GradBBf16` | "reaches this render's f32 form, resnet34in_momdp64 (Theorems step_tie, sync_tie), and stops at its bf16 gradient nodes" | block texts |
| 6118 R34 ImageNet | "the render Theorem sync_tie ties to one device's step at 256" | same | "the data-parallel step Theorem sync_tie ties …, on the render's f32 form" | same |
| 8043 MNv4 side quest | "Theorem mobilenetv4_sync_tie ties that render [emaaccdp8x128wxdowd005bf16] to one device's step at 512" | `mnv4_net_syncTiedB` docstring (MobileNetV4SyncStepTieB.lean:1300–1302): the AdamW update "and, in mnv4in_emaaccdp8x128wxdowd005bf16, the EMA and gradient accumulation" are not stated; f32 `*GradB` nodes; `mnv4in_adamdp64` is the f32 DP render (MobileNetV4StepTieB.lean names it) | "ties the f32 data-parallel render (mnv4in_adamdp64) …; this render's gradient nodes are the bf16 *GradBBf16 kinds, tied per operator, and its EMA, gradient accumulation and classifier dropout sit outside" | docstring; `ls verified_mlir/mnv4in_*` (no f32 ema/acc/do render) |
| 17080–17082 headline | "exact reverse-mode derivative over ℝ, up to one printer, one lowerer, and floating point" | the bullets above it (17062–17077) list the formal StableHLO semantics as trusted; tie scope per the capstone docstrings | "— at a smooth point, at one replica, in f32, on the chain without drop-path — up to one printer, one lowerer, the StableHLO semantics the denotation assumes, and floating point" | LeanMlir.lean:112–116 wording reused |

### Theme 5: stale facts

| Site | Old claim | Statement read | New text | Checked by |
|---|---|---|---|---|
| 369, 354–355, 375 budget table | ch 5 "10 contracts … ResNet-34 per example"; totals 45 / 84 | no per-example ResNet-34 VJP exists: `grep -rn 'def resnet34Forward[A-Za-z0-9]*HasVJP'` → `resnet34ForwardBFullHasVJPAt` only; Nets/ResNet has no non-B file. The row's other nine: `residualHasVJP`, `residualProjHasVJP`, `globalAvgPoolFlatHasVJP`, `flatConvStride2HasVJP`, `flatConvStride2WeightGradHasVJP`, `maxPool3s2HasVJPAt3`, `cnnHasVJPAt` (CNN.lean:1604, stem→pool→rblk→rblkP→GAP), `resnet34ForwardBFullHasVJPAt`, `resnet50ForwardBFullHasVJPAt` | 9 contracts; 44 / 83 | each name grepped. ⚠ steps column untouched in round 1: the seven `*_net_lossGrad` would add one step certificate per ImageNet net (ch 5 +2, ch 6 +2, ch 7–9 +1 each → 37 / 90) if the rule counts them — decision for round 2 |
| 5098 | "This chapter has exactly one theorem" | ch 5 theorems section has ten blocks (5150–5348) | "The residual is the one theorem the idea needs" | block list |
| 2113 | "line for line, the rendering of a theorem" | `mlp_train_step_tied_certified` is one example (`x`, `label : Fin d₃`); the listing is the N = 128 render with the batch-sum weight gradient and α/128 | "stated for one example: Theorem mlp_fold certifies the six updates at N = 1, and the batch sum and the α/N scale are the render's" | statement read; block 2352 says "six update operations" |
| 10656 | softmax Jacobian $p_i(\delta_{ij} - p_i)$ | `pdiv_softmax` (Softmax.lean:53): `pdiv (softmax c) z i j = softmax c z j * ((if i = j then 1 else 0) - softmax c z i)` | $p_j(\delta_{ij} - p_i)$ | statement read |

### Theme 6: forwards of the quoted artifacts (§4.B.3), prose sites

| Site | Old claim | Statement read | New text | Checked by |
|---|---|---|---|---|
| 6691–6693 MNv2 | forward named only by `mobilenetv2FwdGraphBFull_faithful` | `mobilenetv2FwdGraphBFullDo_faithful` (MobileNetV2FullB.lean:453, "at every mask"); `mobilenetv2FwdGraphPaperEval_faithful` (MobileNetV2FullPaperEval, one example, frozen statistics); `FwdGraphTextTies` guards MNv2's blocks, heads with `cd := true`, and the per-example inference forward | clause added | module headers |
| 8028–8031 MNv4 | the eval renders unnamed | `mnv4FwdGraphBFullEval_faithful` (MobileNetV4FullBEval.lean:498, "at every final feature side f"); `FwdGraphTextTies`: MNv4 `.eval` "at both input sizes its evals are rendered at"; `mnv4FwdGraphBFullDo_faithful` (MobileNetV4FullB.lean:996, f32); `ls verified_mlir/mnv4in_*`: the `do` renders are `…wxdowd005bf16` | sentence added | docstrings + artifact list |
| B0 forward block 8427 | — | `efficientnetFwdGraphBFull{,Eval}Drop_faithful` (EfficientNetFullB0Drop.lean:411/443) | round 2 (adds `\lean{}` names) | — |
