# Mathlib-reuse audit — lean4-jax-mlir @ `aed84250` (Lean 4.34.0, Mathlib `v4.34.0`)

Ground truth: the vendored `.lake/packages/mathlib/Mathlib` at tag `v4.34.0` (the audit prompt's
`v4.32.2` is stale — the repo moved to 4.34.0 in `aed84250`). Mathlib paths below are relative to
`.lake/packages/mathlib/Mathlib/`; core paths to `~/.elan/toolchains/leanprover--lean4---v4.34.0/src/lean/`.

**Method.** 11 parallel auditors, one per slice (Foundation ×2, Float, Certificates, Architectures,
Training, Nets ×3, Codegen, non-proof code), plus a cross-repo identical-statement / identical-body scan.
Every `verified` item was restated in a scratch file and closed against the prebuilt oleans with
`lake env lean`. All 156 scratch files were re-run at the end: every cited proof compiles (one
auditor's one-liner did not; its passing sibling is what is quoted below). Nothing in the repo was
edited. Audit run 2026-09-18 on `aed84250`; §10's `VerifiedTrain` line numbers refreshed to `e042f38c` (no proof-layer file changed in between). Scratch-file names cited below (`a5/t2.lean`, …) refer
to that session's scratchpad; the proofs are reproduced inline where they matter.

**Skipped as generated:** the `Certificates/*Scorecard*`, `SmoothingDecChunk*`, `SmoothingNetWitness`,
`LipschitzCertFloat`, the data half of `LipschitzCertInstance`, and `Training/TrainedCnn{Seal,Witness}`,
`TrainedLinearDescent`. Their generators are reported in §5. (`Foundation/IntervalBound.lean`,
`ListDot.lean` and `Codegen/ViTRenderB.lean` mention generation in prose but are hand-written, and were
audited.)

Confidence: **verified** = typechecked · **likely** = matched by grep/diff, not typechecked · **lead** =
suspected, no drop-in located.

---

## Status (2026-09-19, main `716ba7d3`, 5 ahead of origin, + the staged row)

**Landed** — about 15.4k lines out, every pinned theorem name and statement unchanged:

| commit | what |
|---|---|
| `a91c85a0` | §10/§11 defect 1: `ViTGradcheck.parseFloat` rebuilt on `Lean.Syntax.decodeScientificLitVal?`; `TestSgdRenderTie`'s lr guard also rejects non-finite |
| `2fb6574a` | the three demo `parseFloat` copies call the shared `parseFloat?` |
| `dc7150b3` | §9 `StableHLOLex` onto core `Nat.ofDigitChars`; `StableHLOParse.parseStack_toToks` in one line |
| `671d0d5f` | §5 LipschitzCertPairSDP (PSD, Gram), SmoothingNetSemantics, SmoothingMC, SmoothingCP, SmoothingPhiBounds |
| `bbd3e3f0` | FloatSubnormalBridge, MuonGeometry/NewtonSchulz, CrownBound, ResNet34Live{Realistic,PC,2}, the two seal files |
| `998f2c70` | **§0.1 done for Foundation + Architectures**: `pdiv_clm` / `pdiv_of_affine` / `pdiv_of_linear` in `Tensor.lean`; 14 sites (conv / depthwise input-weight-bias, GAP, patch-embed, CLS, BN affine + centered, dense, dense-W, both matmuls); `pdiv_pi_pad_eval` + `pdiv_const_mul_pi_pad_eval` deleted |
| `f4a41380` | **§0.1 done for Nets** (−400): ViTClose `pdiv_rowDense_W`, `pdiv_patchEmbed_W`, the row-dense bias Jacobian, `pdiv_id_add_const`, `pdiv_maskGather_add_const`, both `pdiv_scalarAffine_*`; `ViTVecLN.pdiv_vecLN_beta`; `pdiv_layerScale`, `pdiv_layerScale_gamma`, `pdiv_layerScaleCh_gamma`, the two scalar-LN Jacobians (ConvNeXt, ConvNeXtClose, ConvNeXtFold); both `pdiv_bnPerChannelFlat_*` (CifarBnClose). `pdiv_patchEmbed_{pos,cls,b}` and `pdiv_vecLN_gamma` already delegate to the two helpers, so they stay |
| `7f1093ad` | **§0.3 done + §0.2 for Foundation/Architectures** (−830). §0.3: `relu`, `relu6`, `maxPool2`, `maxPool3s2` linearisations on `Filter.eventually_all` / `hasFDerivAt_pi` (the maxpools' `0 < c,h,w` are now unused, kept as `_hc _hh _hw`). §0.2: 31 `Differentiable` proofs → `fun_prop` (Attention ×15, CNN ×5, Depthwise ×2, StridedConv ×2, BatchNorm ×2, MLP, PerChannelBN, BatchMapVJPAt, `Tensor3.{un,}flatten`); new `@[fun_prop] differentiable_dite_zero` (CNN) for the pad-guarded reads; 14 existing atoms tagged `@[fun_prop]` (dense, relu-at-smooth, bnForward, bnIstdBroadcast, layerNorm, conv2d, flatConv, depthwise, depthwiseFlat, globalAvgPoolFlat, decimate{,Odd}Flat, bnPerChannelTensor3, differentiableAt_pad_eval) |
| `0fe85cc3` | **§0.2 done** (−530): Nets + Training. ViTVecLN ×7, ConvNeXt ×7, ResNet ×13 top-level + 5 inline (B0 `hres_diff` ascriptions dropped), EfficientNet ×10, MobileNet ×5, `crossEntropy_differentiable`, SgdDescentCnn ×7 + 1 inline, SgdDescentMlp ×2 + 3 inline, MlpTrainStep 2 inline. 7 more atoms tagged (`batchMap`, `bnBatchLA`, `seGate`, `seBlockFull`, `mbconvBody`, `relu6_differentiableAt_of_smooth`, `crossEntropy`). Left as they are: `vitForwardKV_differentiable` and `convNextForwardTCh_differentiable` (fun_prop hits a kernel timeout on the recursive stage folds), `chanLNTensor3_diff` (needs an 8-name unfold), `r50IdB_differentiableAt`, and the 3-line-type + 1-line-`.comp` inline haves that `vjp_comp_at` needs typed |
| `54a33add` | **§0.4 done** (−66, and §11 defect 6 closed): `pdiv_elementwise` in `Tensor.lean` (`hasFDerivAt_pi` + `HasDerivAt.comp_hasFDerivAt`); `pdiv_gelu`, `pdiv_swish`, `pdiv_sigmoid` are one-line instances and `pdiv_coordFun` the same argument at one coordinate. `sigmoidScalar` keeps its definition; the bridge `sigmoidScalar_eq_sigmoid` (and `swishScalar_eq_mul_sigmoid`) brings in Mathlib's `Real.sigmoid`, so `sigmoidScalar_diff`, `swishScalar_diff` and `BceLossCot.one_sub_sigmoidScalar` are one line. New: `swishScalarDeriv_eq`, `sigmoidScalarDeriv_eq` — the closed forms the `swishBack`/`sigmoidBack` emitters render, previously only claimed in comments |
| `a9c57a3b` | **§4 near-clones, `SgdDescentCnn` conv2 slot** (−190): new `Conv2Slot` namespace — the drift chain, the four margin lemmas and the segment-Lipschitz gradient stated once for any parameter map `Z` into conv2's pre-activation with per-entry drift `ρ·‖e‖₁`, `ℓ1` drift `(2h)(2w)·ρ·‖e‖₁`, and a fixed Jacobian row `J`. The conv2-kernel (`ρ = a`) and conv2-bias (`ρ = 1`) lemmas are instances (Lipschitz 304 → 25 and 277 → 14); the unpinned `cnn{,b2}_postrelu_close` / `_z3_drift` / `_z4_drift` are gone. Capstones left as they are: they are mostly statement and already delegate to `sgd_descends` + the Lipschitz lemma, so a generic capstone would add lines |
| `046ec333` | **§4 near-clones, `SgdDescentCnn` conv1 slot + the four `gradAt` closed forms** (−1,290; file 9,974 → 8,682): new `Conv1Slot` namespace — the drift chain (`z2_entry_drift`, `z2_l1_drift`, `pool_l1_drift`, `logit_drift`), the relu₂/pool/relu₃/relu₄ margins and the segment-Lipschitz gradient, stated once for any map `Z` into conv1's pre-activation. From conv2's pre-activation on it is `Conv2Slot` at radius `c·kH·kW·w₂·ρ` (`ring` re-associates the radii); relu₁'s margin is `Conv2Slot.margin2_keeps_offkink` at `Z` itself. The Lipschitz lemma folds `J₁`, the frozen relu₁ mask and conv2's taps into one fixed row at the conv2 pre-activation (row mass `(2h)(2w)·c·kH·kW·w₂·ρ`, via `convTap_out_l1` + new `sum_swap_triple_triple`) and calls `Conv2Slot.loss_grad_lipschitz`, which gained a predicate `Q` (`hgrad` only where `Q` holds, `Q` at both segment ends; the conv2 rungs pass `True`). Lipschitz proofs 471 → 15 (kernel) and 459 → 11 (bias). New `gradAt_comp_t3` (chain rule with the flat index split into its triple) takes the four `gradAt` closed forms (conv1/conv2 × kernel/bias) from 73–128-line `calc`s to 10–11 lines; new `convPad_row_l1` / `biasRow_l1` replace the inline row-mass proofs. The unpinned `cnn{,b}1_postrelu{1,2}_close` / `_z3_drift` / `_z4_drift` are gone; every other name and statement is unchanged (`Conv2Slot.loss_grad_lipschitz` is unpinned) |
| `7c379b55` | **§4 `SgdDescentCnn` `grad_close` family** (−248; file 8,682 → 8,434): `cnn_conv2_grad_close` (204-line body) predated `cnn_conv2_cot_close` and re-derived its whole chain inline; it is now `cnn_conv2_cot_close` at the exact input `x₁` + the final `dot_perturbed_close` (≈30 lines), like its bias twin. `cnnConv2GradBudget` already equals that composite with its `let`s unfolded, so no definition changed. The cotangent block (`cnnConv2CotBudget`, `cnnConv2CotMag`, `cnn_conv2_cot_close`, `cnn_conv2_cot_real_abs_le`, `abs_le_of_close`) moved up from the conv1 section to sit before its first user. New `FloatModel.cnnConv2CotMag_nonneg` / `cnnConv2CotBudget_nonneg` replace five hand-built nonnegativity chains (both conv1 `grad_close`s and three float capstones; the conv2-kernel capstone's was ~55 lines) |
| `850dac3b` | **§4 head family, `SgdDescentMlp`** (−322; Mlp 2,187 → 1,925, Cnn 8,434 → 8,374): new `MlpSlot.loss_grad_lipschitz` — the segment-Lipschitz gradient for any map `Z` into a ReLU layer's pre-activation with per-entry drift `σ·‖e‖₁`, `ℓ1` drift `ρ·‖e‖₁` and a fixed row `J` (the `Conv2Slot` shape, same `Q` hook). `mlp_hidden_loss_grad_lipschitz` is `σ = ρ = a` (proof 135 → 10 lines); `mlp_input_loss_grad_lipschitz` takes `Z` = the middle pre-activation, `σ = w₁·a`, `ρ = d₂·w₁·a`, `J` = `xᵢ`·relu₀'s frozen mask·`W₁`'s row, `Q` = relu₀'s signs (231 → 31). New `margin_keeps_offkink_of_drift` (the margin lemma for any per-entry drift) replaces `margin_keeps_offkink`'s and `margin_keeps_offkink_mid`'s proofs and the identical `Conv2Slot.margin2_keeps_offkink` (deleted; its seven callers point at the shared one). New `dense_relu_drift` takes the logit drifts 32 → 2 and 46 → 7; `smul_l1_mass{,_le}` and `dense_input_drift` move up from `SgdDescentCnn`. `mlp_input_logit_drift`'s now-unused `ha` is `_ha`; every other name and statement unchanged |
| `9fa47e55` | **§0.6(a) `CertLayer` adoption, R34/R50 blocks** (−507): new `CertLayer.residualProj` and `CertLayer.reluOut` (CertifiedChain) and four stage layers in `ResNet34BackB0` (`cbReluLayer`, `projLayer`, `cbReluStridedLayer`, `projStridedLayer`). The five block layers (`r34BasicBlockLayer`, `r34DownBlockLayer`, `r50BottleneckLayer`, `r50ProjBlockLayer`, `r50DownBlockLayer`) are composites of those, moved up from `BackNetFolds` / `ResNet50BackNet` into the files that prove the capstones. Each block `_has_vjp_at` is its layer's `.vjp`, each capstone its `.faithful` (29–45-line proofs → 2–3 lines), and the `FullBVJP` block differentiability lemmas its `.diff`. `mnv4ExpandLayer` / `mnv4ProjectLayer` were the same two stage layers; MNv4 now uses `cbReluLayer` / `projLayer`. Gone, unpinned and unused after this: the R34 down-body and R50 body / down-body VJPs and `_faithful`s, and the four `*BodyB_differentiableAt`. The layers' `ok` changes shape (`(A ∧ True) ∧ B` etc.; `projLayer` contributes `True`); no consumer destructured it. R34BackB0 640 → 560, R50BackB0 653 → 400, R50BackNet 339 → 226, BackNetFolds 330 → 272 |
| `45a6eb27` | **§0.6(b) one stage VJP** (−176): new `stage_has_vjp_at` in `Architectures/CNN.lean` — the VJP of `act ∘ norm ∘ lin` at a point, from `lin`'s and `norm`'s global VJPs and `act`'s pointwise one. The 15 hand-built copies (relu and relu6 over global, per-channel and batched BN: `convBnRelu`, `convBnReluStrided`, `convBnReluPC`, `cbrStridedPC`, `bnReluStage`, the four MobileNetV2 relu6 stages and their three per-channel peers, `convBnRelu6StridedPC` / `convStridedBnRelu6PC`, `bnRelu6Stage`) are one call each (14–21 lines → 4–6); the seven batched-stage faithfulness proofs add it to their `simp only` unfold list. Ten stage `_differentiableAt` twins are `fun_prop (disch := assumption)` (two after `unfold` of the XLA strided op). Not the same shape, so left: the EfficientNet swish stages (global `HasVJP`, already 6–10 lines) and MnistCNN's two relu-after-map VJPs (5–6 lines) |
| `4246a624` | **§7 `BnPairTiedB`, MobileNet/EfficientNet family** (−509): new `ResNet34PoCB.BnPairTiedB` (`ResNet34FoldB.lean`) — the γ and β gradient-node clauses of one batched BN layer, at its pre-BN activation and output cotangent — and `bnPairTiedB_holds` (the two `_den` lemmas). The 38 pairs in `MobileNetV4StepTieB` (18), `MobileNetV2StepTieB` (10) and `EfficientNetStepTieG` (10) are folded into their tie `def`s (14 lines → 2) and proved by one `bnPairTiedB_holds` each; the `refine` arities drop to match. EfficientNet's β clauses were stated with γ and the activation at `0` (β's gradient ignores both); the pair states them at the layer's own, which is equally true, and the ties' only consumers are in-file and `AuditAxioms`. The conv-bias β clauses (`bnBetaGradB` at the pre-BN cotangent) are not pairs and stay. Spliced mechanically: regex over whitespace-normalised clauses with balanced-paren atoms, each pair checked field by field (same cotangent, ε, β, layer) before replacing |
| `15c237c8` | **§7 `BnPairTiedB`, ResNet family** (−285): the 17 pairs in `ResNet50StepTieB` (11) and `ResNet34StepTieB` (6, stem included), same splice. All 55 grad-form pairs now go through `BnPairTiedB` |
| `cb36729c` | **§7 fused-from-unfused, EfficientNet** (−119): the eight fused `EnetPoC.*_den` folds (`EfficientNetFold.lean`) are now their un-fused peers (`ResNet34PoCB` / `EnetPoCG`) through `*SgdB_eq_grad` — one `rw` each, `congrArg` for the XLA-`SAME` stem, which has no `_eq_grad` lemma but is `θ − lr·` its gradient node by `rfl`. For that the import runs the other way: `ResNet34FoldB` imports `EfficientNetFold`'s three imports instead of it, and `EfficientNetFold` imports `EfficientNetFoldG`. New `EnetPoC.BnSgdPairTiedB` + `bnSgdPairTiedB_holds` (the fused γ/β pair) fold the 10 pairs in `EfficientNetStepTie`'s tie `def`s, same splice as `BnPairTiedB` |
| `01c6c6f2` | **§6 small-net dense-head folds** (−101): `Cifar8PoC.denseW_den` / `denseB_den` (the emitted `weightSgd` / `biasSgd` of any dense layer, free in activation, weights and cotangent) move unchanged from `Cifar8Fold.lean` to `MlpTrainStep.lean`, the leafiest common ancestor of the small-net folds (46 downstream); the 18 per-layer copies in `MlpFold`, `CnnFold`, `CifarFold` are one-line instances (activation and cotangent by unification). `Cifar8Fold.lean` keeps only its module doc (six files import it) |
| `2a737436` | **§8 ViT/ConvNeXt near-clones** (−691, 20 files). `preLNRes_has_vjp_mat` (`Attention.lean`): the four pre-LN residual sublayer VJPs (scalar/vector LN × attention/MLP) are instances, `rfl`-equal backwards; `patchEmbed_flat_has_vjp.correct` and `pdiv_softmax` on `sum_ite_eq` / `if_congr` (Attention 2,992 → 2,799). The heads = 1 MHSA collapse is `mhsaClean_backward_collapseMH` + the `Fin 1` reindex (121 → 21 lines, 4M heartbeats gone; `qkv_back_fanin`, `sum_one_3d` deleted). The 12 backward-unfold `rfl` lemmas → 6: the four in the tie files move up into `ViTBackB0` under the same names (`_root_.`, types `Expr`-identical; two are AuditAxioms-pinned) and six private copies go. `mlpSublayerBackGraph_faithful` is the MH one at `hm1 := 0`. `vitApexVJP` is `vitForwardKV_has_vjp` itself and `chanLNTensor3_has_vjp` is the term-mode chain, so both `backward_unique` transfers and `chanLNTensor3_vjp_chain` go (the docstrings' "tactic witness sits behind an `Eq.mpr`" was false: `vitNetBackGraph_faithful` already unfolds it by `rfl`). New `StableHLO.patchEmbedF_x_den` (`ViTFwdGraph`) — stage 0 of the five ViT forward-graph faithfulness proofs. Grad bridges on `sum_ite_eq` (ViTClose ×6, ViTVecLN ×2, ViTFoldG, ConvNeXtFold/FoldG, ChannelLN); the four ConvNeXt head folds delegate to the ViT ones (`ConvNeXtFoldG` now imports `ViTFoldG`); `pdiv_layerScale_gamma` from `pdiv_layerScale`. Dead: `hasVJP_backward_det`, `vitCotB2out`, `cnxStemPatchO`, `cnxD11`, the two ChannelLN doc `rfl`s |
| `38a320c6` | **§7 tie clauses, conv / depthwise / dense** (−469): one clause `Prop` per gradient node, each its `_den` lemma's statement with the index bound — `ConvWTiedB`, `ConvBTiedB`, `ConvStrided{W,B}TiedB`, `ConvStridedXlaWTiedB`, `Depthwise{W,B}TiedB`, `DepthwiseStridedWTiedB`, `Dense{W,B}TiedB` (`ResNet34FoldB`, beside `BnPairTiedB`) and the fused `ConvWSgdTiedB`, `DepthwiseWSgdTiedB`, `Dense{W,B}SgdTiedB` (`EfficientNetFold`). 116 clauses in 30 tie `def`s (R34, R50, MNv2, MNv4, EfficientNet G + fused, ConvNeXt GB) are one line each; the tie proofs don't move (`intro idx` unfolds the Prop). Each rewritten `def` checked `rfl`-equal to its HEAD text. Left as written: seven one-off ops (MNv2's `convStridedXlaBias` / `depthwiseStridedXla` pair, the fused ENet `convStridedXla` / `depthwiseStrided` weights, ConvNeXt's stride-4 stem pair, ViT's patch bias) |
| `7f727834` | **§7/§8 tie clauses, row-dense / LN** (−155): the same fold for the ViT and ConvNeXt ties — `ViTPoC.{RowDense{W,B},VecLN{Gamma,Beta}}SgdTied` (`ViTFold`), `ViTPoCGB.{RowDense{W,B},VecLN{Gamma,Beta}}TiedB` (`ViTFoldGB`), `CnxPoC.ChanLN{Gamma,Beta}SgdTied`, `CnxPoCGB.ChanLN{Gamma,Beta}TiedB`. 66 clauses in 12 tie `def`s (ViTStepTie/GB, ConvNeXtStepTie/GB; the ConvNeXt head uses the ViT vector-LN Props), each `def` `rfl`-equal to its HEAD text; the tie proofs don't move. Left as written: ViT's final-LN pair, whose cotangent is an unparenthesised application |
| `feba953c` | **§10 test and program duplicates** (−1,238; §11 defects 2–4 closed): `TestCifar8WideTrain` deleted — `TestCifar8AdamTrain`'s six train steps are one packed-signature `trainStep` + per-optimizer one-liners over `(d1, fname)`, and `main` renders both canonical widths (`cifar8` at 64, `cifar8w` at 512); all 17 renders byte-identical to the two old files, all 14 committed artifacts `iree-compile`. The 14 vjp-oracle nets + `cfg` live once in `LeanMlir/VjpOracleNets.lean` (namespace `VjpOracle`), imported by the 14 phase-3 and 14 phase-2 mains (one call each); `ci_smoke.sh` 14/14. `TestMHSA` gradchecks the shipped `ViTRender.mhsaFwd`/`mhsaBack` (its copy is gone; renders byte-identical); `TestMHSA`/`TestSDPA` call `adjointGradcheck`, which is `adjointGradcheckFixed … []`. The comparators in `TestR50AccumTie`, `TestR50AccumShardTie`, `TestChannelLN`, `TestConvBiasZero` (×3) take the magnitude over both buffers |
| `491f9b31` | **§4 descent-capstone arithmetic** (−123): new `sgd_step_l1_le` (`SgdDescent`) — the `ℓ1` radius `lr·(‖g‖₁ + m·η)` of an inexact step — replaces the 15-line `calc` in `sgd_descent_inexact` and the seven capstones (Linear, Mlp ×2, Cnn ×4); each site keeps its stated `hD` type. New `softmax_seg_drift` (`SgdDescentLinear`) — logit drift `≤ t·δ` with `2δ < 1` gives softmax drift `≤ 2tδ/(1−2δ)` (`softmax_perturb` + `exp_sub_one_le` + the denominator step) — closes the softmax step of the Linear, `MlpSlot` and `Conv2Slot` Lipschitz lemmas (Cnn 26 → 1 lines). Only three softmax copies were left; `Conv1Slot`/`Conv2Slot`/`MlpSlot` had absorbed the other four |
| `b50ba380` | **§4 `finProdFinEquiv` reindexing** (−101): new `sum_finProdFinEquiv` (`SgdDescent`; the same statement as `sum_fin_prod` in `ViTClose`, parked here until the root-file batch) — `∑ k : Fin (m·n), f k` as the row-major double sum. `sum_t3`, `sum_w3`, `sum_abs_k4` are `simp only [sum_finProdFinEquiv]` (+ `rfl` for the non-reducible `w3Idx`/`k4Idx`), `sum_s2` is the lemma itself, `sum_abs_flatten_cols` one line; `sum_abs_kernel_slab_le` (30 lines) is `sum_abs_k4` + `Finset.single_le_sum` (so `sum_abs_k4` moved above it) and Linear's column-mass `hinj` the same argument; `k4Idx_inj` / `t3Idx_inj` are `simpa [and_assoc]`. The four hand-peeled 3-level indices are `t3Idx_surj` and the two 4-level ones `k4Idx_surj` (7 → 2 and 6 → 1 lines) |
| `7639580e` | **§4 seal files** (−90): new `fderiv_ne_zero_of_ray` (`JacobianSeal`) — a readout `ℓ` of `f` with nonzero derivative along `t ↦ x + t • v` at `0` gives `fderiv ℝ f x ≠ 0`; the readout is a plain function (`fun y => y 0` or `fun y => y 0 - y 1`, differentiability by `fun_prop`), since `proj 0 - proj 1` as a `→L` stalls in instance elaboration. The five `*_jacobian_nonzero` seals (MNv2 toy / 224, R34 toy / 224 / full depth) go 14–17 → 4–6 lines. New `hasDerivAt_mul_self_zero` (`t·Q t` has derivative `Q 0` for `Q` continuous at `0`) closes the four `gd_hasDerivAt*` / `g_hasDerivAt` slope arguments (12–13 → 3–5). `winF` moves into the MNv2 toy seal in place of the identical private `win'`; the γ-general `bnForward_chan_diff_γ` / `UDiff_bn_γ` move from `Mnv2RealSeal` into `R34RealSeal` (unpinned; the MNv2 file reaches them through its `open`), and `UDiff_bn` is their γ = 1 case (13 → 2). Deleted, unpinned: `g_full_hasDerivAt` (dead since the full seal went through `fderiv_add_const`), `max_add_r` (= `max_add_add_right`). Left: `MobileNetV2.lean`'s private `win` (240 downstream — root-file batch), `ldS` = `ldSβ · · 20` (§6 live witnesses), `stemS`/`Rr`/`ray` (not instances) |
| `5c53587b` | **§4 dense / conv difference identities** (−176): `smul_l1_mass{,_le}`, `dense_unflatten_diff`, `dense_unflatten_col_drift` move up from `SgdDescentMlp` into `SgdDescentLinear`, whose `dense_unflatten_drift` (41 lines) is now `dense_unflatten_col_drift` + the column-mass bound and whose segment drift uses `smul_l1_mass`. The "difference of two affine evaluations" identities are one `simp only [… ← Finset.sum_sub_distrib …]` each: `dense_unflatten_diff`, `dense_input_drift`'s `hdiff`, `conv2d_kernel_sub` (20 → 2), and new `conv2d_input_sub` (beside `conv2d_kernel_sub` / `conv2d_bias_sub`) for the two 25-line inline copies in `conv2d_input_entry_drift` / `conv2d_input_l1_drift`; `head3_sum_drift`'s `hcoll` and `abs_triple_sum_sub_le` (30 → 4) the same way. `mask_scalar_close` is `sign_stable_of_close` + `if_congr` (its `hex` is now unused: `_hex`), `relu_entry_lipschitz` is `abs_max_sub_max_le_abs` via `max_def_lt`, Linear's 21-line `hcot0` is `M.cotErr_nonneg` |
| `73a0db90` | **§4 Mathlib one-liners + Training's §0.5 idioms** (−119): `JacobianSeal` — both `backward_ne_zero_of_pdiv_ne` are `simpa [h.correct]` (the `sum_eq_single` block was written twice), `sum_smul_basisVec` one `simp`, `fderiv_eq_zero_of_pdiv_all_zero` via `sum_smul_basisVec` + `map_sum` (a `pdiv` row is `fderiv` on the basis vector by definition), `exists_pdiv_ne_of_fderiv_ne` one `simpa … using mt …`; the unpinned `fderiv_basisVec_eq_zero_of_pdiv_row` deleted. `fderiv_apply_eq_sum_grad` on `LinearMap.pi_apply_eq_sum_univ` (15 → 5). New `abs_softmax_sub_oneHot_le_one` (`SgdDescentLinear`, parked until §3 makes FloatBridge's softmax bounds public) replaces five 14-line copies (Linear, Mlp ×2, Cnn ×2). `MlpSlot`'s mask freeze is `if_congr`; nine margin ⇒ off-kink proofs are `abs_pos.mp (h.trans_lt hm)`. Left: eight `sum_const`/`card_univ` rewrites (a line or less each) |
| `96bb507b` | **§3 FloatBridge** (−243): new `FloatModel.rnd_close` (`|a − b| ≤ e`, `|a| ≤ A` ⇒ `|fl(a) − b| ≤ u·A + e` — every rounded `add`/`sub`/`mul`/`div` is `M.rnd` of the exact op) and `abs_rnd_le` (`|fl(x)| ≤ (1+u)|x|`): `mul_close`, `sgd_step_close`, `dense_close_mixed`'s bias add, `softmaxF_close`, `softmax_ce_cot_close` and `ResNet34FloatBridge.add_close` end in one `rnd_close`; the four leaf-magnitude `calc`s in the mixed dots are `abs_rnd_le`. New `mlp_l1_close` — the layer-0 → layer-1 forward prefix (`l0`, `r0`, `ha₁`, `l1`) as one conjunction — replaces 17 lines in each of the five `mlp_*_step_float_close`. `mulErr_mono`, `sgdErr_mono`, `layerBudget_le_of`, `denseMixedBudget_le_of` are `unfold; gcongr`. `pow_one_add_sub_one_le` on `geom_sum_mul`; `pow_gamma_bound` via `(1+u)^k ≤ e^{k·u}` + `exp_sub_one_le` (moved up), so the private `one_sub_mul_pow_le` goes. `relu_close` on `abs_max_sub_max_le_abs`; `relu_abs_le` one line. `softmax_nonneg` / `softmax_le_one` are public and `abs_softmax_sub_oneHot_le_one` moves up from `SgdDescentLinear` (same full name), closing FloatBridge's own copy |
| `a7696b70` | **§3 BN float** (−125): `bnIstd_close` is `bnIstd_close_at` at the floor `V = ε` (25 → 7; it moved below its general form); `FloatModel.bnVar_close` runs `bnMean_close_of` on the float squares plus a `|bnMean fsq − bnVar| ≤ esq` shift, instead of re-deriving the sum-and-divide chain (103 → 41, `bnVarBudget` unchanged). The centered, β-add and mean-divide stages of `bnForward_close_of` / `bnMean_close_of` / `bnVar_close` are `rnd_close`. BnInputBridge's private `abs_sub_le_add` is Mathlib `abs_sub` |
| `03cbb0db` | **§3 `FloatClose` family** (−140): new `FloatClose.of_close` (`FloatComposeBridge`) — a real bound `R`, a fresh-input rounding bound `E` and the error modulus give `FloatClose A (R+E)`; `floatClose_flatConv` (18 → 4), `_dense` (16 → 6), `_gap` (48 → 30), `_depthwise` (20 → 4), `_flatConvMixed` (21 → 4) and `_bn` (30 → 15) are instances. `floatClose_bnRelu` is `(floatClose_bn …).comp (floatClose_relu _)` (44 → 1) and `floatClose_residualBlock` `(floatClose_addResidual M hF).comp (floatClose_relu _)` (28 → 1). Binder-only renames in five pinned statements whose `0 ≤ β` / `0 ≤ A` hypothesis only fed the old `0 ≤ E` step: `_hβ` (flatConv, dense, depthwise, flatConvMixed), `_hA0` (gap) |
| `caa2960a` | **§3 mixed precision** (−65): new `FloatModel.storeBias_close` (`ConvMixedFloatBridge`) — dot → store at `L` → bias add at `M`, both roundings through `rnd_close` / `abs_rnd_le` — makes `conv_close_mixed` and `depthwise_close_mixed` 2 lines each (54 before). The bracket is one definition: `convBrR` / `convBrR_nonneg` / `convBr_eq_convBrR` move up from `ConvMixedComposeBridge` (same full names), `convBr M L n := convBrR M.u L.u n`, `dwBr := convBr` (`rfl`-equal to the HEAD formulas), and the two inline nonnegativity proofs go; `DepthwiseMixedFloatBridge` imports `ConvMixedFloatBridge`. `floatClose_addResidual`'s magnitude is `abs_rnd_le` |
| `49ba248e` | **§6 live witnesses — one generic downsample** (−163): `liveDownW` (+ `_proj_pos`, `_sum_pos`, `_vjp`, `_diff`) moves up from `ResNet34LiveGeneric` into `ResNet34LivePC`, with `liveDownβ := liveDownW · · · WsP2` (from LiveRealistic) and `liveDownPC := liveDownβ · · 20` — three copies of the positivity / VJP / differentiability lemmas become one; `Dom2_liveDownPC` / `liveDownPC_const` are the βp = 20 instances; instance names kept so call sites don't move. Seal side: `ldSβ` and its `_pos` / `_continuous` / `liveDownβ_eq_ldSβ` move up into `ResNet34LiveSeal`, `ldS` is `abbrev ldSβ · · 20`. Each changed definition a pinned statement mentions is `rfl`-equal to its HEAD text |
| `b8124630` | **§6 live witnesses — one generic stem** (−107): `Xs s` and `stemβ s β` in `ResNet34LivePC` with `_bn_pos`, `_inj`, `_maxpool_smooth`, `_vjp`, `_diff`, `Dom2_stemβ`, `Dom2_Xs`, `stemβ_zero`, plus `maxPoolFlat_vjp_of_smooth` / `_diff_of_smooth` for the flatten/unflatten point step written four times. The 16×16 and 112×112 families (`X2`, `stem2`, `stem2_vjp` / `_diff`, `hmp_vjp2`, … and their 224 peers) are one-line instances; their `_bn_pos` / `_inj` / `_maxpool_smooth` / `_conv_eq` / `mp_point_eq*` copies and `X2_inj` / `X224_inj` go. Seal side: generic `Yt` / `Ys`, `stemSβ`, `stemβ_eq_stemSβ` in `ResNet34LiveSeal`; `Y`, `Y224`, `stemS`, `stemS224` are instances. Left: ResNet34Concrete's 1-channel stem (ResNet34.lean has 99 downstream) |
| `2c565fb9` | **§6 live misc** (−16): `ResNet34LivePC.flatConv_diag_id` takes a bias (`b`, `hb`) and absorbs `MobileNetV2SealRealistic`'s `flatConv_diag_id_b`; LivePC's `relu_const_pos` goes for the identical `Proofs.relu_const_pos` in ResNet34.lean |
| `f48d9726` | **§6/§7 exact duplicates** (−39): `r50StemGraphB` is `r34StemGraphB` and its pinned `_faithful` `r34StemGraphB_faithful` (the docstring's "differs by one bias string" was stale — both use `biasName false "" oc`); `r34Trunk_3463` is `r50Trunk_3463` (unpinned `r34Stage` goes); MobileNetV2's `convBn'_has_vjp` / `_differentiable` go for CNN.lean's identical `convBn_*` (6 call sites); the pinned WholeBack stem `convStridedBnRelu6PC_has_vjp_at` / `_differentiableAt` delegate to FullVJP's `convBnRelu6StridedPC_*` (the file imports FullVJP; 3 downstream). Left: `r50StageFirst` / `r50StageDown` (pinned `r50Stage_faithful` names the first), `reluMaskB` → `reluMaskBack` (0 lines saved), and the root-file ones — `conv2d_1x1'`, `sum_flatten8`, `bnForward_const_eq` / `flatConv_zero` vs MobileNetV2's copies (siblings in the import graph) |
| `1da2619f` | **§6 `MnistCNN` `Mini` ≡ `Spatial`** (−71): new namespace `TwoChan` holds the shared `T0`, `X`, `b1`, `b2`, `W3`…`b5`, `κ` and two clause Props `Conv1Mix W1` / `Conv2Mix W2`; the chain (`conv1_pos` … `dense3_pos`, `cnn_has_vjp_at`) is stated once over any kernels satisfying them. Each tier keeps its kernels, `conv1_eq` (now `: Conv1Mix W1`), a `conv2_mix`, a one-line `*Cnn_has_vjp_at` and its pinned `*_has_vjp_correct`, whose text is unchanged but whose constants now resolve to the identical `TwoChan.*` (nothing else named the 24 per-namespace copies) |
| `611b8469` | **§9 `den_batchOp`** (−131): one `@[simp] den_batchOp : den (.batchOp op e) = batchMap N (denOp op) (den e)` + `attribute [simp] denOp` replace the 32 per-op `den_batchOp_<op>` `rfl` lemmas. Their warnings (the XLA-`SAME` conv/depthwise ≠ the symmetric ones, inference BN's batch independence, maxPool3s2 ≠ maxPool) are comments on `denOp`'s arms. The ~15 consumer `simp only` lists in the R34/R50/MNv2/MNv4/ENet `FullB` and render files read `den_batchOp, denOp`; the `_eq_reluF`/`_eq_relu6F`/`_eq_swishF` lemmas they combine with are pre-rewrites (`↓`), and `mnv4StemGraphB_faithful` unfolds its graph first. The pinned `_eq_*` and `_per_example` lemmas stay |
| `c7b534af` | **§9 one-liners** (−90): `patchEmbedFlat` / `patchEmbedBackFlat` are `abbrev`s of Attention's `patchEmbed_flat` / `patchEmbed_input_grad_formula` (−47; `ViTBackB0.patchEmbedBackFlat_eq_backward` goes, `patchEmbedBackGraph_faithful` is `rfl`); the stale "so `StableHLO` needn't import `Attention`" docstrings and a docstring naming a nonexistent `maxPool3s2_ne_maxPool_descr` fixed (§11 defect 5). `batchSlice_batchMap` / `batchMap_pointwise` are `Mat.unflatten_flatten` / `Mat.flatten_unflatten` terms, `lookupEntry` is `List.lookup`. `adamVNext_nonneg`, `adam_denom_pos`, `clipDenom_pos`, `lambDenom_pos` (= `clipDenom_pos`), `rmsSqNext_nonneg` (= `adamVNext_nonneg`) one term each; `lambScale_not_shared` is `norm_num [lambTrust, gradSumSq]` (12 → 2); `keepProb_last` via `Nat.cast_sub` |
| `cb52f18b` | **§9 one `zrnd`** (−151): the renderers' identity rounding for the bf16/fp8 ops is one documented `StableHLO.zrnd`; the 48 `let zrnd := fun r => r` in CnnRender, EfficientNetRender, the MNv2/MNv4/R34/R50 `RenderB`s, their repeated "placeholder rounding" comment blocks and the private `zrndB` / `vzrnd` go (use sites unchanged but the two renames). All 249 `verified_mlir/` artifacts re-render byte-identically |
| `f6dc58c9` | **§9 one AdamW tail** (−59): `StableHLO.prettyAdamW` (the `adamMNextF`/`adamVNextF`/`adamWParamF` triple on a gradient name, with the `wdName`/`wd` note that sat in ViTRender) and `adamWConsts wdStr` (β₁/β₂/ε/wd). The seven triples (MNv2 `adamOneM`, MNv4 `adamOne4`, `enetAdamOne`, `convnextAdamOne`, `vitAdamOne`, R34 `optOne`'s `.adamw` and `.adamwAccum`) call it, and the six AdamW constant blocks are `adamWConsts` (`adamConstsM`, `adamConsts4`, `enetAdamConsts` deleted; `vitAdamConsts`, `convnextAdamConsts`, `optConstsB .adamw` delegate). Byte-identical |
| `716ba7d3` | **§5 generator sync** (scripts only): four generators had drifted from their committed output — hand edits made after generation (`732c7750`'s docstring link, the 4.34 `ite_eq_left`/`ite_eq_right` renames, `TrainedLinearDescent`'s scope paragraph) that the next regeneration would have reverted. Each now regenerates its committed file byte-identically, and `lipschitz_cert_scorecard.py`, `_float.py` and `trained_linear_descent.py` resolve paths from the repo root instead of an absolute path into the sibling `lean4-jax` checkout |
| *(staged)* | **§5 generator lemmas** (−2,198 Lean, −122 scripts): G2 `pair_sq_bound_mlp` (`pair_sq_bound` on `denseE W2 ∘ reluE ∘ denseE W1`'s own gap) and `pair_sq_symm` (`LipschitzCertPairSDP`) make every emitted `pairSq*` theorem one term — the 12-line body and the 6-line reverse-order wrapper; G1 `mlp_out_eq` (`LipschitzCertInstance`) makes the 73 emitted `hout` blocks one line. G3: `lipschitz_cert_float.py` no longer emits `coord_abs_le_norm` (Mathlib `PiLp.norm_apply_le`) or `layerBudget_le_of'` (FloatBridge's `FloatModel.layerBudget_le_of`, now public); `trained_linear_descent.py` uses `FloatModel.softmax_le_one` for `sm_le_one`. G4: `sqrt_two_le_rat` (one line, `Real.sqrt_le_iff`) and `certified_at_eps` move from the generated scorecard into `LipschitzCertInstance` under the same full names. Regenerated: the pooled scorecard, its float tier, the four SDP files, `FullImgsA/B`, `SmoothingNetWitness`, `TrainedLinearDescent`; every declaration and statement unchanged. Checked: the gate, `lake build CertsHeavy` + `AuditAxiomsHeavy`, and the two CI-disabled `SDPFull` modules by `lake env lean` (160 s / 15.6 GB and 133 s / 13.4 GB; `Uncon` imports `SDPFull`, so the first needs `-o` to an olean the second can find) |

**Deferred on purpose:** the conv1 `grad_close` pair (both already delegate to `cnn_conv2_cot_close`; what
they share is ~70 lines of margin → off-kink and forward-closeness setup whose generic statement is as
long as the setup); `DataParallel.dpIterate_lockstep` (the `Semiconj` term is not shorter);
the `X_inj` family (1 line each); `sigmoidScalar := Real.sigmoid` as a definition (the bridge lemma
gets the same reuse without moving its 32 consumers). From §8: the vector-LN backward seam (four
pinned statements of one fact, each already a 2-line proof over the pointwise lemma, in sibling files);
`layerScale_grad_gamma` = `diagBack` (`rfl`, but no proof gets shorter); `preLNRes` flat-diff helpers
(one line saved per site, 14 lines of statement).

**Next, in order** (handoff for the next session: chase the remaining duplicates, largest verified first,
one family per commit; the pinned-statement questions wait for the end list below). ⚠ Recent batches landed
at 35–70% of their estimates (signatures stay; some listed sites turn out not to be the shape) — plan on half.
1. **§4 Training near-clones** — done 2026-09-18 (−609 over five commits, against ~600 estimated). Families
   plus the seal files, one commit each (see §4 below for the lemma sketches):
   - ~~softmax segment drift → `softmax_seg_drift`; ℓ1 step radius `hD` ×8 → `sgd_step_l1_le`~~ (`491f9b31`);
   - ~~the `finProdFinEquiv` reindex ×5 → `sum_finProdFinEquiv`, the inline peels → `t3Idx_surj`/`k4Idx_surj`~~
     (`b50ba380`);
   - ~~the `*_jacobian_nonzero` ray argument → `fderiv_ne_zero_of_ray`; the seal-file specialisations~~
     (`7639580e`);
   - ~~the dense-difference identity / `smul_l1_mass` / `dense_unflatten_diff` / `mask_scalar_close` / `hcot0`
     bullet, `abs_triple_sum_sub_le`~~ (`5c53587b`);
   - ~~the Mathlib one-liners and Training's share of the §0.5 idioms~~ (`73a0db90`).
   Left: eight `sum_const`/`card_univ` rewrites (a line or less each).
2. **§3 Float near-clones** — done 2026-09-18 (−573 over four commits, against ~500 estimated). Left: the
   `convWindow3` hoist onto SgdDescentCnn's window block (~35; `convWindow3` is `rfl`-equal to a `convPad`
   lambda, but reusing SgdDescentCnn's lemmas means importing it into ConvMixedFloatBridge — 43 modules then
   rebuild on every SgdDescentCnn edit — or a new shared `convPad` module); `bnVar_nonneg` / `bnIstd_pos`
   into BatchNorm.lean waits for the root-file batch (item 6). FloatBridge has 60 downstream modules; a
   FloatBridge batch gates in under 2 minutes.
3. **§6/§7 remaining duplicates and witnesses** — done 2026-09-19 (−396 over five commits, against ~700
   estimated: the eight `*LossCot_den` are pinned one-liners, `reluMaskB` → `reluMaskBack` saves nothing,
   §0.6(b) had already taken every relu-after-map site but MnistCNN's two 5-line ones, `idBlk2` is not `idBlk`
   at c = 2, and the EfficientNet / ConvNeXt / ViT fold restatements are pinned one-line delegations — a §0.7
   question). For the root-file batch: `conv2d_1x1'`, `sum_flatten8`, `bnForward_const_eq` / `flatConv_zero`
   (ResNet34 and MobileNetV2 can't see each other) and ResNet34Concrete's 1-channel stem.
4. **§9 Codegen** — done 2026-09-19 (−431 over four commits, against ~400 in Lean). Left, with the
   reason: `batchSlice` / the eight `row*Flat` as `Mat.unflatten` / `batchMap` instances (496 use lines,
   and the `simp only [batchSlice, Mat.unflatten]` sites expect the current unfolding); `clsSliceFlat` /
   `clsPadFlat` / `rowSoftmaxFlat` as aliases (downstream `simp` unfolds their bodies; 0 lines saved);
   `DropPath`'s lemmas (already one-liners, stated before `dropout` is); the zero placeholders (the ~700
   local `let z… : Vec (…) := fun _ => 0` pin the SHlo shape indices at each `.operand`; the 15 generic
   private `zV`/`zK`/… would save 15 lines for ~380 touched). Renderer near-clones still open (likely):
   the RMSProp tail ×2, the bf16/fp8 constructor switch (~250 sites → smart constructors), the
   BN-parametrised forward twins (~400), `R34Bn` ≡ `BnMode` and friends. The op-template `let`s (`dg`,
   `convFwd`, …) live in the hand-written `*Text` predecessors — see the end list.
5. **§5 generator lemmas** — done 2026-09-19 (−2,198 Lean over two commits, against ~1,600 emitted
   estimated; ~1,070 of it is in the CI-disabled `SDPFull` pair). Left: the float-side engine lemmas
   (`certified_at_eps_close`, `mlp2_float_close_uniform`) stay in the generated `LipschitzCertFloat` —
   moving them needs a hand-written module between FloatBridge and the scorecard, for 0 lines;
   `sm_pos` / `sm_sum` have no upstream peer and are emitted once. Still hardcoding the sibling
   `lean4-jax/` path: `lipschitz_cert_witness_s8.py`, `_power_iter.py`, `_rationalize.py` (writes a
   snippet into an old scratchpad), `trained_cnn_{witness,seal}.py`, `mnv2_forward_tie.py`,
   `xla_pad_op_check.py`.
6. **Root-file batches, together** (each is a full ~7.5-min `Certs` rebuild): §1 `Tensor.lean` (~600:
   `pdivMat_rowIndep_perRow_at`, `pdivMat_colIndep`, the Kronecker `correct` fields, the `finProdFinEquiv`
   reindexes, `vjp_comp` from `vjp_comp_at`); the §0.5 Finset idiom sweep (~250 sites); the §0.2 leftovers —
   tag `flatConvStride2Xla_differentiable` / `depthwiseStride2FlatXla_differentiable` `@[fun_prop]`, after
   which three of the §0.6(b) stage twins drop their `unfold`; the §8 backward-uniqueness copies
   (`hasVJPMat_backward_det{,'}` in ViTBackB0, `HasVJP.backward_unique` in ConvNeXtBackCertifiedTie, and
   the EvenKernelConvBack / BatchMapVJPAt / ResNet34BackCertifiedTieB / StableHLO ones) → one per structure.
7. **§0.6(c)** MobileNetV2 / EfficientNet bodies and whole-net folds onto `CertLayer`, as MobileNetV4 was
   (~400, likely). ⚠ Leave the hand-unrolled whole-net apex chains alone (kernel timeouts, `opaqueA*`).

**At the end — pinned statements** (user decisions; add new ones here as they turn up):
- The 12 fused BN pairs written into the statements of the pinned `cifar8Bn_convbn_tied_certified` /
  `cifarBn_convbn_tied_certified` (~116 via a per-example `BnSgdPairTied`; drafted and reverted 2026-09-18,
  it compiled). The same three CIFAR tie theorems (and `Cifar8StepTie`'s) also state 40 per-example conv
  W/b clauses (`convWeightSgd` / `convBiasSgd`) in their statements (~200 via clause Props).
- §0.7 keep-or-retire: the restatements only `tests/AuditAxioms.lean` consumes (~1,800), including
  `vitNetBackGraph_faithful`'s second proof and the 1-head scalar-LN chain (§8) and the SGD-wrapped
  `*_render_*_certified` restatements (§7, `HasVJP.sgd_certified`).
- The fused EfficientNet tie theorems derived from `EfficientNetStepTieG`'s (~60; needs a file move,
  since `StepTieG` imports `StepTie`).
- The 2-block vector-LN ViT (`vitForward2V_has_vjp`, `vitFwdGraphMHV{,_faithful}`) as the depth-k one at
  k = 2: both sit upstream of `ViTDepthK`, so deriving them means moving pinned theorems into it (~100).
- The hand-written `*TrainStepText` / `*FwdText*` emitters in `StableHLO.lean` (~1,110 lines: mlp, cnn,
  cifar, cifarBn, cifar8, cifar8Bn), "kept for reference" since the `*FaithfulV` renders replaced them
  (CnnRender's comments). Callers: `tests/TestCifar8AdamTrain`, `IreeRuntime` (`mlpTrainStepText`),
  `CnnTrainStep` (`cnnTrainStepText`). Retire, or keep and hoist their op-template `let`s (~300).

**How the batches were run** (worked; keep doing it):
- *Keep every statement verbatim.* Replace only a proof body — for a `have h_pdiv` inside a long VJP
  proof, just that `have` — so the rest of the proof, `formalization.yaml`, `AuditAxioms` and the
  docstring gate never move. Every site in `998f2c70` did.
- *Scratch first, per site.* Restate the exact statement (with its local context as binders) in a scratch
  file importing the built module, prove it there, then splice. The oleans are only read, so helpers can
  draft sites in parallel; splice bottom-up so reported line numbers stay valid; don't build while a
  helper is still reading oleans.
- *Rebuild cost decides batching.* Downstream module counts: `Tensor` 301, `MLP` 331, `BatchNorm` 248,
  `CNN` 242, `Attention` 200, `StableHLO` 193, `FloatBridge` 102, `LipschitzCert` 35, the leaf files
  1–8. A root-file batch took 8 min for `lake build Certs` (4038 jobs); leaf batches ~2 min.
  `Nets/ConvNeXt/ConvNeXt.lean` is imported by `StableHLO`, so a batch touching it costs ~7 min.
- *Gate* = `lake build Certs` + `lake build` + `lake env lean tests/AuditAxioms.lean` (every
  `#print axioms` emits a verdict, all ⊆ {propext, Classical.choice, Quot.sound}) +
  `lake exe docstring-checkrefs` + `scripts/check_audit_coverage.py` + `scripts/check_render_coverage.py`.
- Traps met: `simp_all` inside a proof that sits under `set x := … with hx` rewrites with `hx` — use
  `simp only [..., @eq_comm _ idx_in]` instead; `rw [pdiv_of_linear]` needs `f` passed explicitly when
  the side goals are lambdas; plain `mul_left_comm` can time out in `acLt` (fix the scalar:
  `mul_left_comm a`); `split_ifs` also splits a right-hand-side padding `dite` — prefer `by_cases`.
- `pdiv_of_affine` wants `fun v => f v + c`: a site written `fun v k => f v k + c k` takes
  `show … = fun v => (fun k => …) + c from rfl` first (when the constant comes first,
  `funext …; exact add_comm _ _`). After the rewrite, destructure a flat index with
  `obtain ⟨⟨r, k⟩, rfl⟩ := finProdFinEquiv.surjective idx` before `simp [Prod.ext_iff]` — otherwise
  simp turns `finProdFinEquiv.symm idx` into `divNat`/`modNat` and case hypotheses stop matching.
- `fun_prop`: an `@[fun_prop]` tag only reaches downstream files after its home module is rebuilt,
  so a standalone `lake env lean` of a downstream file fails until then — simulate with
  `attribute [fun_prop] X` in the scratch file instead. It has no rule for `dite` (hence
  `differentiable_dite_zero`) nor for `/` with a nonzero side condition (`simp only [div_eq_mul_inv]`
  first, then `fun_prop (disch := intro z; positivity)`). Hypotheses like `0 < ε` on tagged atoms
  go through `fun_prop (disch := assumption)`. Don't tag stage-level composites (`cbsB_`, `seB_`,
  `projB_differentiable`, …): `fun_prop` then times out in `whnf`; tag the atoms and unfold the stage.
- A scratch file that imports a downstream module sees that module's Mathlib imports, not the target
  file's: draft a lemma for a root file (`Tensor.lean`) against an import of that file only, or the
  splice can fail on a missing Mathlib module (`HasDerivAt.comp_hasFDerivAt` needs `Deriv.Comp`).
- Near-clone slots: generate the generic lemma mechanically from the kernel text (regex the conv map
  into `Z`, the input bound `a` into `ρ`), then hand-edit the seams; instances pass the map explicitly
  (`(fun v' => …)`), bias instances pass `(ρ := 1)` explicitly and convert with `simpa only [one_mul]`.
  In a `by simpa … using` body, continuation lines must be indented past the tactic column.
- A generic lemma whose hypothesis holds only at some points: add a predicate `Q` with `hQv`/`hQt` at the
  two points used, not a second lemma (`Conv2Slot.loss_grad_lipschitz`; callers without the side
  condition pass `Q := fun _ => True` and `trivial`). A point-dependent row that is frozen along the
  segment becomes a fixed row by evaluating it at the base point.
- Radii that differ only by association (`c·kH·kW·(w₂·ρ)·D` vs `c·kH·kW·(w₂·(ρ·D))`): pass the implicit
  `ρ` explicitly, then `lt_of_eq_of_lt (by ring) (hm k)` for hypotheses and `.trans_eq (by ring)` for
  conclusions — `ring` also closes the Lipschitz constants' quotients.
- Swapping two index triples: `Fintype.sum_prod_type` + one `Finset.sum_comm` (`sum_swap_triple_triple`);
  a `Finset.sum_product'` rewrite gets stuck on the `AddCommMonoid ?m` instance.
- Never break a line right after `using` in a `by` nested inside a term: the continuation closes the
  tactic block and the error is a misleading "Function expected". Keep `(fun v e => by simpa … using h)`
  on one line. When a hand-formatted statement stops parsing, diff it against the generated draft that
  compiled, with whitespace and paren spacing normalised — that finds the lost `)` directly.
- `SgdDescentCnn` is a leaf: `lake build Certs` ≈ 35 s, a standalone `lake env lean` of the file ≈ 20 s.
- Four forked helpers drafting one file group each (scratch-only, `final_<File>.lean` with primed
  names) and a mechanical splice from those files worked well — ~55 sites in one pass.
- ⚠ **Fold only inside `def` bodies.** Before splicing a clause pattern, find each site's enclosing
  declaration: a pattern written into a pinned theorem's statement is a statement change
  (`Cifar{8,}BnStepTie`, caught after the splice and reverted).
- Placing a generic lemma: from the import graph, take the common ancestor of its consumers with the fewest
  downstream modules — `MlpTrainStep` (46) for the small-net folds, `CNN.lean` for `stage_has_vjp_at`. When
  the natural home is downstream of a consumer, invert the import instead (`ResNet34FoldB` swapped
  `EfficientNetFold` for its three imports; check no module relied on the transitive path first).
- Clause-level splices (`BnPairTiedB`): regex over whitespace-normalised text with balanced-paren atoms
  (strip parens around bare identifiers first), check every field of the pair agrees (cotangent, ε, β,
  layer) before replacing, then re-count each `refine ⟨?_, …⟩` from the bullets that follow it.
- `Certs` rebuild: a `CNN.lean` edit ≈ 7.5 min; leaf batches 5–60 s.
- Clause Props for tie `def`s: state each as its `_den` lemma's conclusion under `∀ idx`, no `_holds`
  lemma — `intro idx; exact …_den … idx` already proves it (`intro` unfolds the def), so no proof moves.
  Fold with strict templates (repeated placeholders = backreferences, named args cross-checked), then
  prove each rewritten `def` equal to its HEAD text: re-elaborate the old text as `<name>_old` against the
  new module and `example : @D = @D_old := rfl` (a mutated copy fails, so the check bites).
- Program-code dedups: dump every render the old files produce (replace `#eval main` with writes to a
  scratch dir), refactor, dump again, `diff -r`. `iree-compile` is not on this repo's `.venv` PATH —
  it lives in `../lean4-jax/.venv/bin` (so `regen_verified_mlir.sh`'s smoke loop prints FAILED here).
- Helpers draft whole-file copies (flat names, outside any `LeanMlir/` dir) and compile them with
  `lake env lean`; a copy imports its upstream, not itself, so names don't clash. A downstream copy that
  needs a changed upstream declaration either carries it in a marked TEMP block (primed name), or is
  checked against a mirror olean root (symlinks to `.lake/build/lib/lean` with the rebuilt module swapped
  in). Splice, then diff every declaration's signature (text up to the top-level `:=`, comments
  stripped, per namespace) against `git show HEAD:` before building.
- A tactic-built witness that opens with `unfold f` still reduces its `.backward` by `rfl`; a term-mode
  twin + `backward_unique` transfer is unnecessary (§8 deleted two).
- Moving a pinned lemma up a file: declare it `theorem _root_.Proofs.X` inside the target namespace and
  check the type is `Expr`-identical to the old olean's with a `run_meta` before deleting the original.
- `simp [sum_fin_prod …]` turns `finProdFinEquiv.symm` into `divNat`/`modNat` and stalls — `rw` it first,
  then `simp`. A bare `eq_comm` in `simp` can time out in `acLt` (pass `@eq_comm _ j i`), and
  `rw [@eq_comm _ j i]` under an `ite` fails on the `Decidable` motive — use `simp only`.
- A generic simp lemma next to a specific one on the same head (`den_batchOp` vs
  `den_batchOp_relu_eq_reluF`): in one `simp only` the generic fires first. Mark the specific one `↓`
  (pre); if the same call also unfolds the graph def, the node only appears after the pre pass — `unfold`
  the def first.
- `StableHLO.lean` alone compiles in ~5.5 min; a gate for a `StableHLO` batch is ~13 min. The renderers
  rewrite all 249 `verified_mlir/` artifacts on rebuild, so `git status verified_mlir/` after the gate is
  the byte-identity check.
- `pkill -f <pattern>` inside a Bash call whose own command line contains the pattern kills that call.
- Regenerate into scratch BEFORE editing a generator: 4 of 9 had drifted from their committed text. A
  copy with `D` / `OUT` / `SRC` / `ROOT` / `OUTDIR` rewritten by `sed` runs any of them into scratch;
  `lipschitz_cert_pair_sdp.py` takes ~8 min, `_pair_sdp_full.py` ~11.5 min (it re-runs
  `_scorecard_full.py` on import), the rest under a minute. Then edit, regenerate, and compare every
  output's declaration names and statements with the committed file before copying it in.
- The generated heavy tier is checked locally, never in CI: `lake build CertsHeavy`,
  `lake env lean tests/AuditAxiomsHeavy.lean`, and `lake env lean` on each `SDPFull` module (in no lib).
  `SDPFullUncon` imports `SDPFull`: write its olean with `-o` and put it where `lake env` looks (a
  second root prepended to `LEAN_PATH` shadows the whole `LeanMlir` package); remove it afterwards.
- `CertLayer` capstones: make the named `_has_vjp_at` def the composite layer's `.vjp` at the same
  point and the capstone its `.faithful`; the hand-written graph defs stay and are `rfl`-equal to the
  composite's `.graph`. Consumers that `rw [← capstone]` never unfold the VJP def, so they don't move.
  `comp` conjoins `ok` as `L₁.ok x ∧ L₂.ok (L₁.fwd x)` and a globally certified stage contributes
  `True`, so hypotheses go in as `⟨⟨h_s1, trivial⟩, h_out⟩`. Left-nest three stages
  (`(L₁.comp L₂).comp L₃`) so `.fwd` is `f₃ ∘ (f₂ ∘ f₁)`, the association the stated types use.

---

## 0. Cross-cutting: one Mathlib construction, many files

### 0.1 — Jacobians of affine maps, derived term by term (~2,900 lines)

**Replace with:** `ContinuousLinearMap.fderiv` (`Analysis/Calculus/FDeriv/Linear.lean:79`) +
`LinearMap.toContinuousLinearMap` (`Topology/Algebra/Module/FiniteDimension.lean:307`) + `fderiv_add_const`
**Why it matches:** every conv / depthwise / patch-embed / GAP / CLS / BN-affine / dense / row-dense
Jacobian is `pdiv` of `v ↦ L v + c` with `L` linear; the repo re-derives it by pushing `pdiv_add`,
`pdiv_const`, `pdiv_finset_sum` ×3, `pdiv_mul`, `pdiv_reindex` through the sum and discharging a
`DifferentiableAt` goal at every level. On a finite-dimensional space `fderiv (L · + c) = L`, so
`pdiv f x i j = L (basisVec i) j`.
**How** (verified, `a5/t2.lean`; the same lemma was found independently by three auditors as
`pdiv_clm` / `pdiv_of_affine` / `pdiv_linear`), in `Foundation/Tensor.lean`:
```lean
import Mathlib.Topology.Algebra.Module.FiniteDimension
theorem pdiv_of_affine (f : Vec m → Vec n) (c : Vec n)
    (hadd : ∀ u v, f (u + v) = f u + f v) (hsmul : ∀ (a : ℝ) v, f (a • v) = a • f v) (x i j) :
    pdiv (fun v => f v + c) x i j = f (basisVec i) j := by
  let L : Vec m →ₗ[ℝ] Vec n := ⟨⟨f, hadd⟩, hsmul⟩
  have hf : (fun v => f v + c) = fun v => (LinearMap.toContinuousLinearMap L) v + c := rfl
  unfold pdiv; rw [hf, fderiv_add_const, ContinuousLinearMap.fderiv]; rfl
```
At each site, `hadd`/`hsmul` are `funext k; simp only [defs, ← Finset.sum_add_distrib]; … split_ifs <;> ring`.

| Site | Now → after | Conf. |
|---|---|---|
| `Architectures/CNN.lean:343–651` `conv2d_has_vjp3` Step 1 | ~310 → ~22 | verified |
| `Architectures/CNN.lean:1176–1531` `conv2d_weight_grad_has_vjp` Step 1 | ~355 → ~20 | verified |
| `Architectures/CNN.lean:1642–1751` `conv2d_bias_grad_has_vjp.correct` (whole) | ~110 → 12 | verified |
| `Architectures/CNN.lean:2558–2663` `pdiv_globalAvgPoolFlat` (whole) | ~105 → 18 | verified |
| `Architectures/Depthwise.lean:141–393` `depthwise_has_vjp3` Step 1 | ~253 → ~22 | verified |
| `Architectures/Depthwise.lean:772–1089` `depthwise_weight_grad_has_vjp3` | ~318 → ~22 | likely |
| `Architectures/Depthwise.lean:1134–1242` `depthwise_bias_grad_has_vjp.correct` | ~108 → 12 | verified |
| `Architectures/Attention.lean:2905–3408` `patchEmbed_flat_has_vjp` Step 1 | ~504 → ~27 | verified |
| `Architectures/Attention.lean:2660–2712` `cls_slice_flat_has_vjp.correct` | ~52 → 7 | verified |
| `Architectures/BatchNorm.lean:268` `pdiv_bnAffine` / `:351` `pdiv_bnCentered` | 86 → 13 | verified |
| `Foundation/Tensor.lean:716` / `:824` `pdivMat_matmul_left_const` / `_right_const` | 199 → ~24 | verified |
| `Foundation/MLP.lean:29` `pdiv_dense` / `:99` `pdiv_dense_W` | 167 → ~25 | verified |
| `Nets/ViT/ViTClose.lean:63` `pdiv_rowDense_W` / `:604` `pdiv_patchEmbed_W` | ~311 → ~53 | verified |
| `ViTClose.lean:453/463/287/317`, `ViTVecLN.lean:754`, `ConvNeXtClose.lean:149/181`, `ConvNeXtFold.lean:34`, `Small/CifarBnClose.lean:55` | ~220 → ~30 | verified |
| `ConvNeXt.lean:63` `pdiv_layerScale`, `ViTClose.lean:485/525/872`, `ViTVecLN.lean:733` | — | likely |

Knock-on: `pdiv_pi_pad_eval` (CNN:212) and `pdiv_const_mul_pi_pad_eval` (CNN:257) become dead (~70).
**Consumers:** every `*_has_vjp*` keeps its statement. **Confidence:** verified (lemma + 14 sites).

### 0.2 — Differentiability side conditions assembled by hand (~650 lines; 100+ sites)

**Replace with:** Mathlib's `fun_prop` tactic, after tagging the repo's atom lemmas `@[fun_prop]`
(`flatConv_differentiable`, `depthwiseFlat_differentiable`, strided variants, `bnForward_differentiable`,
`bnPerChannelTensor3_differentiable`, `bnBatchLA_differentiable`, `batchMap_differentiable`,
`globalAvgPoolFlat_differentiable`, `dense_differentiable`, `swish_diff`, `sigmoid_diff`,
`broadcastFlat_differentiable`, `relu6/relu_differentiableAt_of_smooth`; only 6 repo lemmas carry the
attribute today).
**Why it matches:** each twin is a hand-written `.comp` chain / `differentiable_pi` peel / `reindexCLM …
differentiableAt` block.
**How:** `unfold <body>; fun_prop (disch := assumption)`. Verified on `Tensor3.flatten_differentiable`,
`unflatten_differentiable` (Tensor:1573/1588), `dense_differentiable` (MLP:228, 29 → 1),
`rowwisePerRow_flat_differentiable` (PerChannelBN:143, 30 → 1), `transformerMlp_flat_diff` (Attention:1980,
~57), `layerNorm_per_token_flat_diff` (Attention:170), `rowwise_flat_diff` / `layerNormVec_diff`
(ViTVecLN:88/40), `rowLNVecFlat_gamma/beta_diffAt` (ConvNeXtChannelLN:261/285), `mbconvBody`,
`mbconvResidual`, `seGate`, `seBlockFull`, `mbResidFwdB`, `invresBody_differentiableAt`,
`batchMap_differentiable` (EfficientNetChainClose:40). 63 twins / 652 lines in MobileNet+EfficientNet
alone; ~46 more in ResNet/Small/ConvNeXt/ViT/Architectures; 40 `reindexCLM … differentiab…` and 34
`DifferentiableAt.fun_sum` sites repo-wide.
**Caveat:** `fun_prop` does not see through the dependent `if` in the conv pad-eval
(`conv2d_differentiable`, `depthwise_differentiable`); those should call `differentiableAt_pad_eval`
(CNN:195) instead (~30 lines).
**Confidence:** verified (representatives), likely (bulk).

### 0.3 — Local linearisation at a smooth point via a hand-built `inf'` radius (~310 lines)

**Replace with:** `Filter.eventually_all` (`Order/Filter/Finite.lean:247`) + `ContinuousAt.eventually_lt` /
`eventually_lt_nhds` / `eventually_gt_nhds` (`Topology/Order/OrderClosed.lean:691`, `:229`) +
`HasFDerivAt.congr_of_eventuallyEq`, with `hasFDerivAt_pi` (`Analysis/Calculus/FDeriv/Prod.lean:396`) to split
coordinates.
**Why it matches:** each proof builds `r := univ.inf' (gap …)`, proves `r > 0`, bounds `|y k − x k| ≤ ‖y − x‖`
and case-splits — i.e. re-derives "finitely many strict inequalities persist on a neighbourhood".

| Site | Now → after |
|---|---|
| `Foundation/MLP.lean:349` `relu_hasFDerivAt` | 43 → ~13 |
| `Nets/MobileNet/MobileNetV2.lean:74` `relu6_hasFDerivAt` | 107 → ~20 |
| `Architectures/CNN.lean:2072–2201` `maxPool2_flat_hasFDerivAt` | ~130 → 28 |
| `Architectures/MaxPool3s2.lean:305–432` `maxPool3s2_flat_hasFDerivAt` | ~128 → 32 |

**Consumers:** 2 / 4 / 5 / 4; statements unchanged (the maxpool versions no longer need `0 < c/h/w`).
**Confidence:** verified (all four).

### 0.4 — Elementwise Jacobians and the logistic function

**Replace with:** one `pdiv_elementwise (φ) (hφ : ∀ k, DifferentiableAt ℝ φ (x k)) : pdiv (fun y k => φ (y k)) x i j
= if i = j then deriv φ (x i) else 0`, 12 lines over `hasFDerivAt_pi`, `hasFDerivAt_apply`
(`FDeriv/Prod.lean:392`) and `HasDerivAt.comp_hasFDerivAt`; and `Real.sigmoid`
(`Analysis/SpecialFunctions/Sigmoid.lean:63`), `differentiable_sigmoid` (:193), `Real.hasDerivAt_sigmoid` (:138),
`Real.deriv_sigmoid` (:144), `Real.sigmoid_neg` (:108).
**Why it matches:** `Architectures/LayerNorm.lean:222` `pdiv_gelu`, `:349` `pdiv_swish`,
`Nets/EfficientNet/EfficientNet.lean:68` `pdiv_sigmoid` are three verbatim ~30-line copies of one proof;
`Foundation/Tensor.lean:215` `pdiv_coordFun` is the same at one coordinate. `sigmoidScalar`
(EfficientNet:49) is `Real.sigmoid` (`funext; simp [Real.sigmoid, one_div]`), and `swishScalar x = x *
Real.sigmoid x`.
**How:** `pdiv_gelu n x i j := pdiv_elementwise geluScalar x (fun _ => geluScalar_diff _) i j` (same for
swish, sigmoid); `abbrev sigmoidScalar := Real.sigmoid`; `swishScalar_diff := differentiable_id.mul
differentiable_sigmoid`; `Foundation/BceLossCot.lean:101` `one_sub_sigmoidScalar` → `Real.sigmoid_neg`.
Bonus: the σ(1−σ) closed form that `Codegen/StableHLO.lean:6945` and the LayerNorm docstring say is
"only checked empirically" becomes `simp [sigmoidScalarDeriv, Real.deriv_sigmoid]`.
**Consumers:** 8 / 2 / 3 (pdivs); 32 `sigmoidScalar`. **Lines saved:** ~120. **Confidence:** verified.
(Mathlib v4.34 has no `tanh` derivative, GELU, softplus, or softmax — those stay.)

### 0.5 — Finset / `Fin` plumbing idioms (repo-wide counts)

| Idiom in the repo | Library replacement | Sites | Conf. |
|---|---|---|---|
| `calc … Finset.sum_le_sum …; rw [sum_const, card_univ, card_fin, nsmul_eq_mul]` | `(Finset.sum_le_card_nsmul _ _ _ fun i _ => h i).trans_eq (by simp)` (`Algebra/Order/BigOperators/Group/Finset.lean`) | 44 (Training 31, Float 13) | verified |
| `Fintype.sum_equiv finProdFinEquiv.symm …` + nested-sum split | `rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type]` (`Algebra/BigOperators/Group/Finset/Defs.lean:750`); hoist `ViTClose.sum_fin_prod` / `BackwardMaps.sum_flat3` into `Tensor.lean` (Mathlib has no `Fin (m*n)` sum lemma) | ~30 (Tensor ×5, CNN ×4, Depthwise ×2, SgdDescentCnn ×6, ViT ×8, …) | verified |
| `Finset.sum_eq_single` + `ite_eq_left/right` + `absurd (Finset.mem_univ _)` | `simp [ite_and, Finset.sum_ite_eq, Finset.sum_ite_eq', Finset.sum_ite_irrel]` (`…/Finset/Piecewise.lean:141,153`, `Defs.lean:574`) | 85 | verified (Tensor ×7, CNN, ViT ×6) |
| `finProdFinEquiv.injective heq` / `Prod.mk.inj` case bashes | `simp [EmbeddingLike.apply_eq_iff_eq, Prod.ext_iff]` + `finProdFinEquiv.surjective` destructuring | 65 in 16 files | verified (Tensor, Attention, BackwardMaps, ResNet34) |
| `have := hm k; rw [h0, abs_zero] at this; exact absurd …` | `abs_pos.mp (hc.trans_lt (hm k))` | 20 | verified |
| 5-line `by_cases … ite_eq_left` mask freeze | core `if_congr (hstab _).2 rfl rfl` | 17 | verified |
| `|x| ≤ √y` / `a ≤ b` from squares via `√(a²) ≤ √(b²)` | `Real.abs_le_sqrt`, `sq_le_sq₀`, `abs_le_of_sq_le_sq` | 9 | verified |
| hand-evaluated literal `√256 = 16`, `√4 = 2` | `norm_num` extension `Mathlib/Tactic/NormNum/RealSqrt.lean` | 7 | verified |

### 0.6 — The repo's own `CertLayer.comp` is not used where it applies (~1,800 lines)

`Foundation/CertifiedChain.lean:64` `CertLayer` + `.comp` already bundle fwd/ok/diff/vjp/graph/faithful,
and MobileNetV4, EfficientNetBackNet, ResNet-50 and ViT use it; its docstring says it replaces "the
argument every `<body>BackBatchedGraph_faithful` in the repo writes out by hand". Still hand-written:
- `Nets/ResNet/ResNet34BackB0.lean:164–642` (12 decls), `ResNet50BackB0.lean:79–661` (16),
  `ResNet50BackNet.lean:50–156` → two combinators (`CertLayer.reluOut`, `CertLayer.residualProjL`, ~25
  lines) + four stage layers; each block becomes one line. `.fwd`, `.graph`, `.faithful` and the
  backward are `rfl`-equal to the current ones (verified, `a8/t19.lean`), so `StepTieB`/`FullBVJP`
  consumers keep working. ~950.
- The act∘norm∘lin stage VJP is written 17+ times (MobileNetV2:221/259/662/708;
  MobileNetV2BackCertifiedTie:39/71/103/214; MobileNetV2FullVJP:54; MobileNetV2WholeBackCertifiedTie:59;
  EfficientNet:239/262; MobileNetV2BackB0:90; ResNet34BackB0:86; EfficientNetChainClose:135/153;
  CNN:869; ResNet34:249; CifarCNN:735) → one 7-line `stage_has_vjp_at`; `rfl`-equal for 5 checked
  (verified). ~450.
- MobileNetV2 / EfficientNet bodies and whole-net folds (MobileNetV2:353/416/763,
  MobileNetV2FullVJP:364, MobileNetV2FullBVJP:318, EfficientNetFullB0:381, EfficientNetChainClose:224–352)
  → port exactly as MobileNetV4 was (likely, ~400).

### 0.7 — Restatements consumed only by `tests/AuditAxioms.lean` (a keep-or-retire decision)

These re-state a generic certificate at one instance and have no Lean consumer besides the axiom
audit (verified by grep): 58 `*_chain_certified` (ViTChainClose ×18, ViTVecLN ×6,
ConvNeXtChainClose ×11, MobileNetV2ChainClose ×9, Cifar8Close ×10, CnnChainClose ×4); `EfficientNetClose.lean`
(whole file); `ResNet34Close.lean:41–107`; `ResNet50FoldB.lean:74–263` (`r50*GradsCertified`;
`ResNet50StepTieB` calls `ResNet34PoCB` directly); `Cifar8Close.lean:106–240`; `MobileNetV2FoldPaper.lean`
(its own header says the artifact "NO LONGER EXISTS"); `LipschitzCert.lean:211`
`smoothing_certified_radius` (docstring: "can never be instantiated"); `SmoothingGaussian.lean:269`
`gaussian_np_shift` (the n = 0 case of `pi_gaussian_np_shift`). ~1,800 lines if retired. Not a Mathlib
question — whether the audit file *should* pin them is yours (`MlpCanonical.lean` is a deliberate audit
surface; these may be too).

---

## 1. `LeanMlir/Proofs/Foundation/`

### Tensor.lean:928 — `pdivMat_rowIndep` (+ `BatchMapVJPAt.lean:37` `pdivMat_rowIndep_at`, `PerChannelBN.lean:32` `pdivMat_rowIndep_perRow`)

**Replace with:** one `pdivMat_rowIndep_perRow_at (g : Fin m → Vec n → Vec p) (A) (h : ∀ r, DifferentiableAt ℝ (g r) (A r))` (34 lines) in `Tensor.lean`; the three become corollaries
**Why it matches:** three ~90-line copies of the same `rowProj`/`h_coord`/`h_swap`/`h_basis` proof; `rowwise_has_vjp_mat` (Tensor:1024, 35 refs) = `rowwisePerRow_has_vjp_mat` (PerChannelBN:120) by `rfl`
**How:** `pdivMat_rowIndep := by rw [pdivMat_rowIndep_perRow_at (fun _ => g) A (fun _ => h_g_diff _)]; split_ifs with h <;> simp [h]`; also drops `BatchMapVJPAt`'s import of `EfficientNetChainClose`
**Consumers:** 4 / 1 / 1   **Lines saved:** ~217
**Confidence:** verified

### Tensor.lean:1078 / :1230 — `pdivMat_colIndep` (146 lines), `colSlabwise_has_vjp_mat.correct` (70)

**Replace with:** `finProdFinEquiv.surjective` + `simp [Prod.ext_iff, eq_comm]`; `Equiv.sum_comp` + `Fintype.sum_prod_type`
**How:** `obtain ⟨⟨h, j'⟩, rfl⟩ := finProdFinEquiv.surjective jj; simp_rw [← Equiv.sum_comp finProdFinEquiv _, Fintype.sum_prod_type, pdivMat_colIndep g hg_diff]; simp [hg.correct]`
**Consumers:** 2 / 1   **Lines saved:** ~160   **Confidence:** verified

### Tensor.lean:246 — `pdiv_finset_sum` (29-line `Finset.induction_on`)

**Replace with:** `fderiv_fun_sum` (`Analysis/Calculus/FDeriv/Add.lean:489`)
**How:** `unfold pdiv; rw [show (fun y k => ∑ s ∈ S, f s y k) = fun y => ∑ s ∈ S, f s y from by funext y k; simp [Finset.sum_apply], fderiv_fun_sum hdiff]; simp [Finset.sum_apply]` (drops the needless `[DecidableEq α]`)
**Consumers:** 41 lines / 13 files   **Lines saved:** ~20   **Confidence:** verified

### Tensor.lean:175 — `pdiv_comp` (key step `hv_decomp` + `map_sum`/`map_smul`)

**Replace with:** `LinearMap.pi_apply_eq_sum_univ` (`LinearAlgebra/Pi.lean:359`)
**How:** `have h := LinearMap.pi_apply_eq_sum_univ (fderiv ℝ g (f x)).toLinearMap (fderiv ℝ f x (basisVec i)); simp [pdiv, fderiv_comp x hg hf, h, Finset.sum_apply]`
**Consumers:** 195 lines / 13 files   **Lines saved:** ~13   **Confidence:** verified

### Tensor.lean:215 — `pdiv_coordFun` (26 lines) → `pdiv_elementwise` / `hasFDerivAt_pi` (§0.4). ~19. verified.

### Tensor.lean:110 / 121 / 131 — `pdiv_id`, `pdiv_const`, `pdiv_reindex`

**Replace with:** simp set `fderiv_id`, `fderiv_const`, `ContinuousLinearMap.fderiv`
**How:** `simp [pdiv, eq_comm]` / `simp [pdiv]` / keep the `reindexCLM` rewrite then `rw [ContinuousLinearMap.fderiv]; simp [eq_comm]`
**Lines saved:** ~14   **Confidence:** verified

### Tensor.lean:523 / 1918 / 1347 / 1404 — `pdivMat_id`, `pdiv3_id`, `pdivMat_scalarScale`, `pdivMat_transpose`

**Replace with:** `EmbeddingLike.apply_eq_iff_eq` + `Prod.mk.injEq` under `simp`, instead of 10–20-line `finProdFinEquiv.injective` case bashes
**How:** `simp [pdivMat, Mat.flatten_unflatten, pdiv_id]`; `simp [pdiv3, Tensor3.flatten_unflatten, pdiv_id, and_assoc]`; `rw [pdiv_reindex]; simp [and_comm]`
**Lines saved:** ~107   **Confidence:** verified

### Tensor.lean:624, 1029, 1444, 1467, 1491, 1510, 1944 — seven `correct` fields collapsing Kronecker sums by hand

**Replace with:** `simp [ite_and]` (`Finset.sum_ite_eq`, `Fintype.sum_ite_eq`)
**How:** `simp_rw [pdivMat_id]; simp [ite_and]` (resp. `pdivMat_scalarScale`, `pdivMat_transpose`, `pdiv3_id`; `simp [mul_comm]` for `matmul_right`; `simp [hg.correct]` for `rowwise`)
**Lines saved:** ~100   **Confidence:** verified

### Tensor.lean:485, 677, 1658, 1831, 1889 — `finProdFinEquiv` sum reindexing ×5 (1831/1889 are verbatim 26-line copies)

**Replace with:** `rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type]` (twice for triple sums)   **Lines saved:** ~100   **Confidence:** verified

### Tensor.lean:287 / 385 / 1694 — `vjp_comp`, `vjp_comp_at`, `vjp3_comp`

Derive the global compositions from the pointwise peers (`correct x := (vjp_comp_at … ).correct`); the 25-line triple-sum `calc` in `vjp3_comp` is repeated verbatim in `vjp3_comp_at`. `backward` unchanged (`rfl`). ~40. verified.

### Tensor.lean:93 — `reindexCLM` (hand-built structure literal)

**Replace with:** `ContinuousLinearMap.pi fun k => ContinuousLinearMap.proj (σ k)` (`Topology/Algebra/Module/ContinuousLinearMap/PiProd.lean:208`) — `rfl`-equal; keep the name (101 consumer lines unchanged). ~4. verified.

### Tensor.lean:419–438 / 1540–1566 — flatten/unflatten inverse lemmas (low value)

`Mat.flatten = (Equiv.curry (Fin m) (Fin n) ℝ).symm.trans (finProdFinEquiv.arrowCongr (Equiv.refl ℝ))` by `rfl` (Tensor3: nested). The four inverses become `Equiv.symm_apply_apply` / `apply_symm_apply` (`Logic/Equiv/Defs.lean:455`). Keep the defs (hundreds of `show`/`unfold` sites rely on the unfolded shape). ~14. verified.

**Base definitions — checked, keep (not a finding).** `Mat.mulVec`, `Mat.mul`, `Mat.transpose`, `Mat.outer`, `dense`, `reindexCLM`, `Mat.flatten`/`Tensor3.flatten` are all `rfl`-equal to `Matrix.mulVec`, `*`, `Matrix.transpose`, `Matrix.vecMulVec`, `Matrix.vecMul x W + b`, `ContinuousLinearMap.pi`, `Equiv.curry`-based equivs; `basisVec i = Pi.single i 1` is provable but not `rfl` (dite vs ite). A swap would move ~57 `unfold`/`simp only [Mat.X]` sites onto `dotProduct`/`Matrix.of` and put a `Matrix` import in the root file, for no line savings. `Codegen/MatBridge.lean` already bridges (its 4 proofs can be `:= rfl`; 0 Lean consumers).

### MLP.lean:29 / :99 — `pdiv_dense` (65) / `pdiv_dense_W` (102) → §0.1. ~138. verified.
### MLP.lean:228 — `dense_differentiable` (29) → `unfold dense; fun_prop` (§0.2). Same statement also at `Architectures/Attention.lean:111` `dense_diff` and `:2728` `dense_input_diff`. ~26 + ~20. verified.
### MLP.lean:349 — `relu_hasFDerivAt` (43) → §0.3. ~29. verified.
### MLP.lean:335 / 446 / 478 — `reluLinearPart_apply`, `relu_codegen_matches_canonical`, `relu_has_vjp_at.correct` → `ContinuousLinearMap.pi_apply` + `split_ifs`; `simp_rw [pdiv_relu …]; simp`. ~21. verified.

### MLP.lean:433 / 531 + SpecVJP.lean ×12 — canonical `HasVJP` witness written 15 times

`backward x dy i := ∑ j, pdiv F x i j * dy j; correct := rfl` (SpecVJP:157, 278, 332, 521, 580, 711, 791, 851, 998, 1054, 1116, 1172; `tests/AuditMutation.lean`) → one `HasVJP.canonical f` in Tensor.lean; each witness is `rfl`-equal (4 checked). ~44. verified/likely.

### MLP.lean:501 — `oneHot c l` = `basisVec l` by `rfl` (217 uses). 0 lines, but it unlocks `basisVec_apply` and the Kronecker lemmas for one-hot targets. verified.

### BatchMapVJPAt.lean:176 — `batchMap_has_vjp_at.correct` (31) → `Equiv.sum_comp` + `Fintype.sum_prod_type` + `Finset.sum_ite_irrel` + `Finset.sum_ite_eq`. ~26. verified.

### MuonGeometry.lean:323 — `conj_diag_pow`

**Replace with:** `Units.conj_pow` (`Algebra/Group/Semiconj/Units.lean:98`) + `Matrix.diagonal_pow` (`Data/Matrix/Basic.lean:180`)
**How:** `have := Units.conj_pow ⟨W, Wᵀ, mul_eq_one_comm.mp hWtW, hWtW⟩ (Matrix.diagonal d) k; simp only [Units.val_mk, Units.inv_mk, Matrix.diagonal_pow] at this; exact this`
**Consumers:** 8   **Lines saved:** ~9   **Confidence:** verified

### MuonNewtonSchulz.lean:62 / 101 — `hdiag_smul`, `hsum3.hd` → `Matrix.diagonal_smul` / `Matrix.diagonal_add` (`Data/Matrix/Diagonal.lean:102/95`). ~6. verified.
Near-clone: the `U·diag·Vᵀ` collapse ×6 and SVD transpose ×3 (MuonGeometry 167/361/385/388/394, MuonNewtonSchulz 70/79/82) → `svd_transpose` + `conj_diag_mul`. ~35. verified. (Mathlib v4.34 has no SVD.)

### DataParallel.lean:295 — `dpIterate_lockstep` → `Function.Semiconj.iterate_right` (`Logic/Function/Iterate.lean:111`). 5 consumers. ~7. verified.
Near-clone: lifted-scalar-loss sum ×3 (`BceLossCot.lean:160–173`, `SmoothedLossCot.lean:95–106`, `DataParallel.lean:114–125`; and `DataParallel.lean:98–103` duplicates `:114–119`) → `pdiv_lift_sum`. ~20. verified.

### BceLossCot.lean:101 — `one_sub_sigmoidScalar` → `Real.sigmoid_neg` (§0.4). ~7. verified.

### CrownBound.lean:339 / 367 / 217–223 — `getD_replicate_zero`, `getD_map_getD`, `crownRow_dot`

`by simp` (`List.getElem?_getD_replicate_default_eq`, `Data/List/GetD.lean:64`); `List.getD_map` (:43); `crownRow a W1 = Matrix.vecMul a (Matrix.of W1)` by `rfl`, proof body → `Matrix.dotProduct_mulVec` (`Data/Matrix/Mul.lean:761`; keep the statement, 166 generated refs). ~16. verified.

### IR.lean:175 — `kRev` = core `Fin.rev` (`Init/Data/Fin/Basic.lean:382`). 8 consumers. verified (equality), likely (drop-in).
### IntervalBoundConv.lean:168 — `relu_apply_eq_max` → `(max_def_lt' (x i) 0).symm` (`Order/Basic.lean:402`). Move it next to `relu` in MLP.lean — §3 and §4 need it. verified.

### IntervalBound / IntervalBoundConv / CrownBound — interval boxes are `Set.Icc`, box-soundness is `Set.MapsTo`

`IBP.InBox lo hi u ↔ u ∈ Set.Icc lo hi` (`Pi.le_def`); a generic `BoxSoundG f Flo Fhi := ∀ lo hi, Set.MapsTo f (Icc lo hi) (Icc (Flo lo hi) (Fhi lo hi))` has `.comp` = `Set.MapsTo.comp` (`Data/Set/Function.lean`) and replaces `BoxSound`/`BoxSound3`/`BoxSound3V`, their three `.comp` lemmas (IntervalBoundConv 66/70/78/92/446/451) and two capstones (:402, :511). The `hbox` block ×4 (IntervalBound:228, IntervalBoundConv:410/519, CrownBound:89) → one helper / `Set.mem_Icc_iff_abs_le` (`Algebra/Order/Interval/Set/Group.lean:111`). Sign-split ×8 and uniform-collapse ×6 → two scalar lemmas. ~100. verified. ⚠ the capstones are cited by generated scorecards (`ibp2_certified_at_eps` 60 refs): change the generators in the same commit. `InBoxE` stays (`EuclideanSpace` has no Pi order).

### PerChannelBN.lean / StridedConv.lean — six hand-rolled reindex `HasVJP` instances and their index maps

`reassocFwd/Back_has_vjp` (371/383), `bnchwFwd/Back_has_vjp` (575/587), `decimateFlat_has_vjp`, `decimateOddFlat_has_vjp` (StridedConv 61/226) are `rfl`-equal to one generic `reindexVJP σ`; the four `*_has_vjp_backward_eq` (398/412/599/613) → one `reindexVJP_backward_of_inv`. `reassocFwdIdx`/`reassocBackIdx` (269/277) = `finCongr (Nat.mul_assoc oc h w).symm` / `finCongr (Nat.mul_assoc …)` (`Logic/Equiv/Fin/Basic.lean`); `bnchwFwdIdx`/`bnchwBackIdx` (531/539) = an `Equiv` built from `finProdFinEquiv`/`prodCongr`/`prodAssoc`/`prodComm` (`rfl`), so their round trips (545/551, 286/292) are `Equiv.apply_symm_apply`. `ConvNeXtChannelLN.lean:57/70` `_val` lemmas (~20) disappear. ~130. verified.

### BackwardMaps.lean:137 — `decimateOddIdx_injective` (+ `Nets/ResNet/ResNet34.lean:503` `decimateIdx_injective`)

**Replace with:** `EmbeddingLike.apply_eq_iff_eq` (`Data/FunLike/Equiv.lean:187`) + `Prod.mk.injEq` / `Fin.mk.injEq` under `simp only … at heq`, then `Fin.ext (by omega)` ×2. 20 + 26 lines → ~6 each. ~32. verified. (`decimateOddIdx_injective` has 0 proof consumers.)

**Foundation — clean:** OpaquePrefix (widths differ per slot; plain defs needed for `rfl`), CertifiedChain, ListDot (`decide +kernel` needs computable `List ℤ`), UpstreamDraft (**none** of its 19 drafts is in Mathlib v4.34.0 — keep), DataParallelNode, BackNetFolds, ConvLossFold, Bf16GradNodes, EvenKernelConvBack.

---

## 2. `LeanMlir/Proofs/Architectures/`

(The big wins — §0.1 affine Jacobians, §0.3 maxpool linearisation, §0.4 gelu/swish — are above.)

### CNN.lean:2715 — `max_close`
**Replace with:** `abs_max_sub_max_le_max` (`Algebra/Order/Group/MinMax.lean:83`)
**How:** `(abs_max_sub_max_le_max a b c d).trans (max_le h1 h2)`
**Consumers:** 3 + AuditAxioms (keep the name as a one-liner)   **Lines saved:** 16   **Confidence:** verified

### CNN.lean:2759 — `abs_max_le` → `abs_max_le_max_abs_abs.trans (max_le ha hb)` (`Algebra/Order/Group/Abs.lean:162`). 3. verified.
### CNN.lean:2303–2328 — triple `Finset.sum_eq_single` in `maxPool2_codegen_matches_canonical` → `simp [ite_and, Finset.sum_ite_eq']`. ~24. verified.
### CNN.lean:86 — `Kernel4.flatten_unflatten` (20 lines) → `funext k; simp only [Kernel4.flatten, Kernel4.unflatten, Prod.mk.eta, Equiv.apply_symm_apply]`. 22 consumers. ~15. verified.

### Attention.lean:2366 / 2384 — `transformerTower` (hand `Nat.rec`) + `transformerTower_flat_diff`
**Replace with:** `Nat.iterate` / `Function.iterate_succ'` (`Logic/Function/Iterate.lean:171`), `Differentiable.iterate` (`Analysis/Calculus/FDeriv/Comp.lean:217`), `Function.Semiconj.iterate_right` (:111)
**How:** keep the def, add `transformerTower … k = (transformerBlock …)^[k]` (`induction k; rfl; rw [Function.iterate_succ', ← ih]; rfl`), and a 6-line generic `flat_diff_iterate`
**Consumers:** 18   **Lines saved:** ~40   **Confidence:** verified (keep the def: the `succ` cases rely on the `Nat.rec` defeq)

### Attention.lean:477 — `rowSoftmax_flat_diff` re-proves `softmax_diff` (:117) per row → compose with `reindexCLM`, as `layerNorm_per_token_flat_diff` does. 10 consumers. ~45. verified.
### Attention.lean:150 — `layerNorm_diff` = `bnForward_differentiable D ε γ β hε`; the `bnNormalize`-differentiable block is also at BatchNorm.lean:736 and :766 → one `bnNormalize_differentiable`. ~35. verified.
### Attention.lean — flat-diff-of-composition ×13 (754, 847, 1057, 1854, 1906, 2018, 2031, 2072, 2141, 2232, 2315, 2421, 2524; +46 elsewhere) → `flat_diff_comp` (`simpa [Function.comp_def, Mat.unflatten_flatten] using hG.comp hF`). ~100. verified (lemma), likely (sites).
### Attention.lean:1400 `h_lift_basis`; :542 → `Finset.sum_ite_irrel`; :292 `pdiv_softmax` Kronecker → `simp`. ~45. verified.
### Attention.lean:2778 / 2849 — `patchEmbed_flat`, `patchEmbed_input_grad_formula` are duplicated verbatim in `Codegen/StableHLO.lean` (see §9).

### BatchNorm.lean:50 / 327 — `bnMean` = `𝔼 i, x i` (`Finset.expect`, `Algebra/BigOperators/Expect.lean`), `bnCentered` = `Fintype.balance` (`Algebra/BigOperators/Balance.lean`); the hand "Σ(xₖ − μ) = 0" at :606–611 is `Fintype.sum_balance`. Keep the defs (97/30 refs), add the bridges. ~6. verified.

### Near-clones
- `DepthwiseBackCertifiedTie.lean:82–142` ≡ `Foundation/IR.lean:207–270` `convBackDenote_eq_input_grad_formula` after IR's per-channel `sum_congr` (same `sum_subset`/`sum_bij'`/five `omega` cases) → one slab lemma. ~55. likely.
- `h_var_nonneg` ×2 (BatchNorm:418/463) — see §3 `bnVar_nonneg`.

**Architectures — clean:** SE, Residual, ChannelLNBack. (Softmax/LSE: not in Mathlib v4.34 — genuine.)

---

## 3. `LeanMlir/Proofs/Float/`

### BnInputBridge.lean:47 — private `abs_sub_le_add : |a − b| ≤ |a| + |b|`
**Replace with:** `abs_sub` (`Algebra/Order/Group/Abs.lean`; additive form of `mabs_div`, :79)
**How:** delete; rename 5 uses   **Lines saved:** 3   **Confidence:** verified

### FloatBridge.lean:479 / 645 — `relu_close` (19-line 4-way split), `relu_abs_le` (+ `Training/SgdDescentMlp.lean:54` `relu_entry_lipschitz`, the same fact a third time)
**Replace with:** `abs_max_sub_max_le_abs` (`Algebra/Order/Group/MinMax.lean:93`), via `relu_apply_eq_max` (moved to MLP.lean)
**How:** `rw [relu_apply_eq_max, relu_apply_eq_max]; exact (abs_max_sub_max_le_abs _ _ _).trans (hx i)`
**Consumers:** 36 / 30 / 22   **Lines saved:** ~50   **Confidence:** verified

### FloatBridge.lean:554 / 574 / 114 — `one_sub_mul_pow_le` (15), `pow_gamma_bound`, `pow_one_add_sub_one_le` (13): (1+u)^k bounds by two separate inductions
**Replace with:** `Real.add_one_le_exp`, `Real.exp_nat_mul`, `pow_le_pow_left₀`, and the repo's `exp_sub_one_le` (Mathlib's `Real.exp_bound_div_one_sub_of_interval`, `Analysis/Complex/Exponential.lean:617`, needs `0 ≤ x`, so keep the repo lemma)
**How:** `pow_gamma_bound := (sub_le_sub_right (pow_le_exp u hu k) 1).trans (exp_sub_one_le hk)` with `pow_le_exp := by rw [Real.exp_nat_mul]; exact pow_le_pow_left₀ …`
**Lines saved:** ~28   **Confidence:** verified

### FloatBridge.lean:939 / 953 / 626 / 2069 — `mulErr_mono`, `sgdErr_mono`, `layerBudget_le_of`, `denseMixedBudget_le_of` (13/18/17/28 lines of `mul_le_mul`)
**Replace with:** Mathlib's `gcongr` tactic
**How:** `unfold sgdErr; have : 0 ≤ u' := hu.trans huu; gcongr`
**Consumers:** 2 / 4 / 8 / 3   **Lines saved:** ~60   **Confidence:** verified

### FloatSubnormalBridge.lean:147 — `bnSqrt_normal` (13) → `Real.le_sqrt'` (`Analysis/Real/Sqrt.lean:243`). ~9. verified.
### FloatSubnormalBridge.lean:167 — `istd_ge_minNormal` → `le_one_div` (`Algebra/Order/Field/Basic.lean:55`). ~5. verified.

### Near-clones (Float)
- `ConvMixedFloatBridge.lean:136` `conv_close_mixed` ≡ `DepthwiseMixedFloatBridge.lean:83` `depthwise_close_mixed` (54-line bodies, line for line) → one `FloatModel.storeBias_close`. ~65. verified.
- `ConvMixedFloatBridge.lean:42–94` + `ConvMixedComposeBridge.lean:46–82`: nine `convWindow3`-family decls are `Training/SgdDescentCnn.lean`'s `convWindow`/`convKernelMat`/`sum_w3`/`conv2d_eq_dense`/`abs_convPad_le`/`convPad_close` (`rfl` or one line); the docstring at :58 records a rename-on-collision. Hoist SgdDescentCnn's window block into a small shared module. ~55. verified.
- One mixed-precision bracket defined 3× (`convBr` ConvMixedFloat:103, `convBrR` ConvMixedCompose:160, `dwBr` DepthwiseMixed:56; `rfl`-equal), inline 4× in FloatBridge, nonnegativity re-proved 5×. ~30. verified.
- Four `FloatClose` instances with one template (FloatComposeBridge:65/96, DepthwiseFloatBridge:174, ConvMixedComposeBridge:302; `floatClose_gap` :159) → 7-line `FloatClose.of_close`. ~70. verified.
- `floatClose_bnRelu` (FloatComposeBridge:349, 44) = `(floatClose_bn …).comp (floatClose_relu _)`; `floatClose_residualBlock` (:248, 28) = `(floatClose_addResidual M hF).comp (floatClose_relu _)`. ~68. verified.
- `bnVar_close` (BnFloatBridge:366, 103) re-proves `bnMean_close_of` (:255) on squares; `bnIstd_close` (:82) = `bnIstd_close_at`. ~77. verified.
- The rounding step `|M.rnd a − b| ≤ u(|b|+e)+e` ×6 (BnFloatBridge 183/380, ResNet34FloatBridge:41 `add_close`, FloatBridge 390/876/906) → `FloatModel.rnd_close`; `|L.rnd x| ≤ (1+L.u)|x|` ×6 → `FloatModel.abs_rnd_le`. ~40. verified.
- MLP layer-0→1 forward chain copied into five step capstones (FloatBridge 1069/1154/1223/1292/1375) → `mlp_l1_close`. ~70. verified (lemma).
- `ConvMixedComposeBridge.lean:441` `floatClose_r50_stages_mixed` body is literally `floatClose_r34_stages hblk`. 7. verified.
- **`bnVar_nonneg` / `bnIstd_pos`**: named at BnFloatBridge:70 but re-derived inline 6× (BatchNorm:418/463, ResNet34:446/471, MobileNetV2:1037/1122) and `bnIstd_pos` exists twice (ResNet34:443, MobileNetV2:1115) → hoist both into `Architectures/BatchNorm.lean`; `bnIstd_pos := by unfold bnIstd; positivity`. ~50. verified.
- `softmax_nonneg` / `softmax_le_one` (FloatBridge:1536/1541) are **private**, so `|softmax − oneHot| ≤ 1` is re-proved 7× (FloatBridge:1836, SgdDescentLinear:305, SgdDescentMlp:1488/1930, SgdDescentCnn:2209/5521/5623) → make public + `abs_softmax_sub_oneHot_le_one`. ~75. verified.

**Float — not findings:** `FloatClose` vs `LipschitzWith` (compares two different maps with a non-linear modulus on a domain — no Mathlib analogue); `exp_sub_one_le` (holds for all x < 1). Clean: Bf16Fold, E4M3Fold, Binary32Instance (already uses `Int.zpow_log_le_self`, `abs_sub_round`).

---

## 4. `LeanMlir/Proofs/Training/`

### SgdDescent.lean:40 — `fderiv_apply_eq_sum_grad` → `LinearMap.pi_apply_eq_sum_univ` (`LinearAlgebra/Pi.lean:359`): `rw [← ContinuousLinearMap.coe_coe, LinearMap.pi_apply_eq_sum_univ]; simp [gradAt, basisVec_eq_ite]`. ~12. verified.

### JacobianSeal.lean:50 / 71 — `sum_smul_basisVec`, `fderiv_eq_zero_of_pdiv_all_zero`
**Replace with:** Mathlib's `Pi.single` simp set (via a bridge `basisVec i = Pi.single i 1`), and `(Pi.basisFun ℝ (Fin m)).ext` + `ContinuousLinearMap.coe_injective`
**How:** `simp [basisVec_eq_single]`; `refine ContinuousLinearMap.coe_injective ((Pi.basisFun ℝ (Fin m)).ext fun i => ?_); funext j; simpa [pdiv, basisVec_eq_single] using hall i j` — this also retires `fderiv_basisVec_eq_zero_of_pdiv_row`
**Lines saved:** ~25   **Confidence:** verified

### MobileNetV2JacobianSealFull.lean:213–234 → `fderiv_add_const` + `mnv2Live_jacobian_nonzero` (`fwdFull = fwd + 45`). ~18. verified.
### ResNet34LiveRealisticSeal.lean:45 — `max_add_r` = `max_add_add_right`. 4. verified.
### SgdDescentCnn.lean:3630 — `abs_triple_sum_sub_le` (30) → `simp only [← Finset.sum_sub_distrib]` + three nested `Finset.abs_sum_le_sum_abs`. 5 consumers. ~25. verified.
### SgdDescentLinear.lean:102 — `hinj` → `Finset.sum_le_sum_of_injOn` (`Algebra/Order/BigOperators/Group/Finset.lean:218`). ~6. verified.
### abs_pos ×20, if_congr ×17, sum_le_card_nsmul ×31 — see §0.5.

### Near-clones (Training)
- **SgdDescentCnn.lean:6993–10252 re-proves :858–6977 with the bias in place of the kernel** (the file header, lines 51–56, says so: "the kernel arguments verbatim with … `a·D` radii replaced by the bare `D` and `a² ↦ 1`"). After normalising, `cnn_conv2_loss_grad_lipschitz` (2748) vs its bias twin (7482) differ in 192 of ~330 lines; `conv1` (5960 vs 8644) in 210 of ~525; the `cnn1_*`/`cnnb1_*` drift/margin families (4213–4652 vs 7964–8392) likewise. State each layer's lemmas once over an abstract pre-activation map `z : Vec P → Vec _` with drift constant ρ and row mass. **~1,300–1,800 lines. likely** (diff evidence, refactor not typechecked).
- Head drift/margin family in 4 slots (SgdDescentCnn 2452–2652, 4357–4641, 7133–7318, 8107–8386; ~700 lines) + 3 Mlp copies → three generic lemmas (~25 lines). `cnn_conv2_logit_drift` (40) closes in 2 lines. ~600. verified.
- Softmax segment drift ×7 (Linear 163; Mlp 453/1096; Cnn 2946/6240/7662/8914) → `softmax_seg_drift`. ~150. verified.
- ℓ1 step radius `hD` ×8 (SgdDescent 150; Linear 233; Mlp 594/1306; Cnn 3282/6658/7924/9294) → `sgd_step_l1_le`. ~105. verified.
- `finProdFinEquiv` sum reindex ×5 (`sum_t3` Cnn:116, `sum_w3` :531, `sum_s2` :1398, `sum_abs_k4` :492, `sum_abs_flatten_cols` Mlp:89) → one `sum_finProdFinEquiv`; `sum_abs_kernel_slab_le` (:463, 30) → `sum_abs_k4` + `Finset.single_le_sum`; 7 inline `surjective` peels → `t3Idx_surj`/`k4Idx_surj`. ~110. verified.
- Dense-difference identity ×5, `smul_l1_mass` inline ×7, `dense_unflatten_diff` ×2, `mask_scalar_close` (Cnn:1911) = `sign_stable_of_close` + `if_congr`, `hcot0` (Linear:376, 21) = `FloatModel.cotErr_nonneg`. ~150. verified.
- `*_jacobian_nonzero` ray argument ×6 (MNv2JacobianSeal:250, MNv2JacobianSealFull:222, MNv2SealRealistic:343, R34LiveSeal:531, R34LiveRealisticSeal:425, `Nets/ResNet/ResNet34LiveFull.lean:328`) → `fderiv_ne_zero_of_ray` (8 lines over `HasFDerivAt.comp_hasDerivAt`, `HasDerivAt.unique`); slope of `t·Q t` ×4 → `hasDerivAt_mul_of_continuousAt` (Mathlib v4.34 has no Carathéodory form). ~130. verified.
- Seal-file specialisations: `bnForward_chan_diff` = `_γ` version at γ = 1; `winF` = private `win'` = `Nets/MobileNet/MobileNetV2.lean:1165 win` (three copies); `ldS`/`stemS`/`Rr`/`ray` = the realistic `…β`/`…224` families at βp = 20. ~90. verified/likely.

Clean: SgdDescentCifar, TrainedMlpWitness. `descent_segment` is not Mathlib's MVT (ℓ∞/ℓ1 pairing is the opposite of the sup-norm one).

---

## 5. `LeanMlir/Proofs/Certificates/` (hand-written) and the generator scripts

### LipschitzCert.lean:76 — `euclid_norm_sq`
**Replace with:** `EuclideanSpace.real_norm_sq_eq` (`Analysis/InnerProductSpace/PiL2.lean:156`) — same statement
**How:** delete; rename the uses   **Consumers:** 12 hits / 5 files   **Lines saved:** 4   **Confidence:** verified

### LipschitzCert.lean:41 / 48 / 62 — `LipschitzL2`, `.comp`, `clm_lipschitzL2`: a real-constant `LipschitzWith`
`LipschitzL2 L f ↔ LipschitzWith ⟨L,hL⟩ f` is `(lipschitzWith_iff_norm_sub_le …).symm` (`Analysis/Normed/Group/Uniform.lean:459`); `clm_lipschitzL2 A := lipschitzWith_iff_norm_sub_le.1 A.lipschitzWith` (`Analysis/Normed/Operator/NNNorm.lean:122`); `.comp` via `LipschitzWith.comp` (`Topology/EMetricSpace/Lipschitz.lean:225`). Keep the def (ℝ literals throughout the generated corpus); re-prove the two lemmas. ~6. verified.

### LipschitzCertPairSDP.lean:66 — `quad_form_nonneg_of_ldl` (22-line double-sum expansion)
**Replace with:** `Matrix.PosSemidef.diagonal` → `.mul_mul_conjTranspose_same` → `.dotProduct_mulVec_nonneg` (`LinearAlgebra/Matrix/PosDef.lean:63/322/306`)
**How:** needs `Mathlib.LinearAlgebra.Matrix.PosDef` + `Mathlib.Algebra.Order.Star.Real`; 6-line proof, same statement
**Consumers:** 1   **Lines saved:** ~16   **Confidence:** verified

### √-sandwich ×7 (LipschitzCertInstance 68/95/386/412/656, LipschitzCert 113, LipschitzCertPairSDP 302) → `(sq_le_sq₀ (norm_nonneg _) hb).1 hsq` (`Algebra/Order/GroupWithZero/Basic.lean:759`) / `abs_le_of_sq_le_sq` (`Algebra/Order/Ring/Abs.lean:131`). ~25. verified.

### SmoothingGaussian.lean — integrability boilerplate
- "integrable × bounded" ×5 (283, 474, 560, 581, 590) → `MeasureTheory.Integrable.mul_bdd` / `bdd_mul` (`MeasureTheory/Function/L1Space/Integrable.lean:1068–1076`). ~40. verified.
- [0,1]-bounded ×3 (557, 574, 886) → `Integrable.of_mem_Icc` (…/Integrable.lean:156). ~10. verified.
- :427 `pi_gaussian_integral_eval` → `MeasureTheory.integral_comp_eval` (`MeasureTheory/Integral/Pi.lean:144`); same for `SmoothingMC.lean:60` `hmean`. ~14. verified.
- :452 `htrans` → `MeasurePreserving.integral_comp'` along `(measurePreserving_piFinSuccAbove _ 0).symm` (no measurability needed; drops `hmF`/`hmW`/`hz0`). ~12. verified.
- :71 `stdGaussian_Ioo_pos` + the open-pos instance → `Measure.AbsolutelyContinuous.isOpenPosMeasure` (`MeasureTheory/Measure/OpenPos.lean:71`) on `ProbabilityTheory.gaussianReal_absolutelyContinuous'` (`Probability/Distributions/Gaussian/Real.lean:259`). ~12. verified.

### SmoothingNetSemantics.lean:166 — `stdGaussian.instIsOpenPosMeasure` → `Continuous.isOpenPosMeasure_map` (`MeasureTheory/Measure/OpenPos.lean:145`). ~14. verified.
### SmoothingNetSemantics.lean:128 — `isOpen_strictRegion` (21) → `simp only [Set.ofPred_forall]; exact isOpen_iInter_of_finite fun j => isOpen_iInter_of_finite fun _ => isOpen_lt (hf j) (hf c)`. ~17. verified.
### SmoothingMC.lean:32 — `iIndepFun_eval_pi` → `ProbabilityTheory.iIndepFun_pi` (`Probability/Independence/Basic.lean:888`). ~13. verified.
### SmoothingCP.lean:88 / 98 — `hitCount_mono`, `hitCount_le` → `Set.indicator_le_indicator_of_subset`; `Finset.sum_le_card_nsmul` + `Set.indicator_le_self'`. ~10. verified.
### SmoothingPhiBounds.lean:85 — `hreal` → `ENNReal.toReal_le_of_le_ofReal`. ~5. verified.

### Near-clones (Certificates)
- `LipschitzCertInstance.lean:42/303/566` — Frobenius / Schatten-4 / Schatten-8 Lipschitz lemmas share one skeleton (the file's :643 comment says "identical tail"); `gram`'s `hTz` re-derives `sum_sq_matTvec_eq` inline → `denseE_lipschitzL2_of_sq` + `_of_transpose`; 212 → ~90 lines, signatures unchanged. verified.
- "q ≤ p_y ⇒ certified" ×3 (SmoothingMC:198, SmoothingCP:432/510) → `smoothing_certified_of_le`; `hpA` ×3 / `hcount` ×2. ~45. verified.
- `LipschitzCertPairSDP.lean:207` `hEB` = `sum_sq_matTvec_eq`, after which private `sum_comm3` (:36) is dead. ~17. verified.

### Generators — the generated files re-emit proof material that belongs in one hand-written lemma
- **G2** — `scripts/lipschitz_cert_pair_sdp.py` and `…_pair_sdp_full.py` emit the ~10-line `pairSq*` body 148× → hand-written `pair_sq_bound_mlp` (2 lines, in LipschitzCertPairSDP.lean); each emitted theorem becomes one term. **~1,300 emitted lines.** verified against `pairSqC_0_1`.
- **G1** — the 5-line `hout` block (`mlp x jj = ∑ k, W2 jj k * max (hpre k) 0`) is emitted 72× by `lipschitz_cert_scorecard.py` (9), `_scorecard_full.py` (21), `_pair_sdp.py` (16), `_pair_sdp_full.py` (16), `smoothing_net_witness_gen.py` (10), `_rationalize.py` (1) and `trained_cnn_seal.py` → hand-written `mlp_out_eq` next to `mlp_gap_eq`. ~290 emitted lines. verified.
- **G3** — `scripts/lipschitz_cert_float.py` emits `coord_abs_le_norm` (= `PiLp.norm_apply_le`, verified) and `layerBudget_le_of'`, whose docstring calls it a "public copy" of private `Float/FloatBridge.lean:626 layerBudget_le_of` → make that public and reference it.
- **G4** — `lipschitz_cert_scorecard.py` emits `sqrt_two_le_rat` (→ `Real.sqrt_le_iff.2 ⟨by norm_num, by norm_num⟩`) and data-free engine lemmas (`certified_at_eps`, `certified_at_eps_close`, `mlp2_float_close_uniform`) that belong in `LipschitzCert.lean`.
- `scripts/trained_linear_descent.py:189–205` emits generic softmax facts (`sm_pos`, `sm_sum`, `sm_le_one`) — point at the §3 public softmax lemmas.
- Clean generators: `ibp_conv_scorecard.py`, `lipschitz_cert_scorecard_ibp.py`, `crown_ibp_scorecard.py`, `smooth_scorecard_gen.py`, `smooth_dec_scorecard_gen.py`; `trained_cnn_{witness,seal}.py` row splits are deliberate (kernel heartbeats).

**Lead:** `stdNormalQuantile` (SmoothingGaussian:67) + ~150-line inverse package — Mathlib v4.34 has no CDF quantile; an `OrderIso ℝ ≃o Ioo 0 1` route exists but changes the `sSup ∅ = 0` junk value MC/CP rely on. **Not in Mathlib v4.34 (checked):** Φ monotonicity/symmetry/inverse, Gaussian tilt, Cameron–Martin, Neyman–Pearson, Clopper–Pearson, an iid-indicator binomial law (Hoeffding and `Real.sum_le_exp_of_nonneg` are already used).

---

## 6. `LeanMlir/Proofs/Nets/ResNet/` + `Nets/Small/`

### ResNet34LiveRealistic.lean:32 — `sqrt_lt_param` → `(Real.sqrt_lt n.cast_nonneg hβ).2 h` (`Analysis/Real/Sqrt.lean:232`). 9 consumers. verified.
### ResNet34LivePC.lean:230 / 191 — `sqrt_lt_20`, `sqrt512_lt_30` → `(Real.sqrt_lt' (by norm_num)).2 (by linarith)` (:235). 14 consumers. verified.
### ResNet34.lean:489 — `bnForward_lb`, `habs` step → `Real.abs_le_sqrt` (:246); 3 lines with `gcongr`. 8 consumers. verified.
### ResNet34Live2.lean:49 — `maxPool2_chan_lt` → `max_lt_max (max_lt_max …) (max_lt_max …)` (`Order/MinMax.lean:113`). verified.
### CifarBnClose.lean:112 — `sum_channel_fibre` (14) → `rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type]; simp`. verified.
### ResNet34.lean:751 / LivePC:169 / LiveRealistic:192 — `X_inj`, `X2_inj`, `X224_inj` → `Nat.cast_injective.comp Fin.val_injective`. verified.

### Near-clones (ResNet/Small)
- **§0.6** — R34/R50 batched block capstones hand-write `CertLayer.comp` (~950, verified for the R34 identity block).
- `reluAfter_has_vjp_at` — one generic "relu after a map with a global VJP" is `rfl`-equal to `MnistCNN:35/55`, `ResNet34:249`, `ResNet34BackCertifiedTie:351` (verified); same shape at `CifarCNN:735`, `ResNet34BackB0:86`, `CNN:869`, and inline copies. ~150.
- **Small/ dense-head folds**: `Small/Cifar8Fold.lean:28/40` `denseW_den`/`denseB_den` are fully generic; 18 per-net copies (`MlpFold:64–147`, `CnnFold:121–236`, `CifarFold:69–180`) are instances. ~290. verified.
- `*LossCot_den` ×7 (MlpFold:160, CnnFold:249, CifarFold:221, CifarBnStepTie:30, Cifar8StepTie:28, Cifar8BnStepTie:25, ConvNeXtStepTie:329) → one `softmaxCELossCot_den`. ~80. verified.
- Live witnesses: `liveDownPC` / `liveDownβ` / `liveFwd224` are `liveDownW`/`liveFwdW` instances (`rfl`); `stem2` / `stem224` are `stemβ 16 30` / `stemβ 112 160` (`rfl`); BN-positivity ×6; `idBlk2` = `idBlk` at c = 2; unit-kernel identity conv ×4; no-tie maxpool ×5; 2×2 maxpool `_vec` peel ×18. ~430. mostly verified.
- Exact duplicates: `r50StemGraphB` (ResNet50FullB:405) = `r34StemGraphB` (ResNet34FullB:246) (`rfl`); `r50StageFirst`/`r50StageDown`/`r50Trunk_3463` = `r34Stage`/`r34Trunk_3463` (`rfl`); `bnForward_const_eq`/`flatConv_zero` (ResNet34:545/555) = MobileNetV2:924/935; `reluMaskB` (ResNet34StepTieB:82, 26 uses) = `reluMaskBack (pre · > 0)`; `relu_const_pos` ×2 = `relu_id_of_pos`. ~110. verified.
- `MnistCNN.lean` `Mini` (458–613) ≡ `Spatial` (628–767) apart from `conv1_eq`/`conv2_eq` (1×1 vs centre-3×3). ~90. likely.
- `EnetPoC.*B_den` (fused) = `rw [convWeightSgdB_eq_grad, ResNet34PoCB.convWGradB_den]`. ~50. verified.
- StepTieB tie Props (`r34{Id,Down,Stem}TiedB`, `r50{Id,Proj,Down}TiedB`) are conjunctions of the same conv-W/b + BN-γ/β shapes → `ConvBnStageTiedB` / `BnPairTiedB` (§7). ~500. likely.
- Hand-unrolled whole-net VJP chains (MnistCNN:106, CifarCNN:77/441/807/1030, R34/R50 FullBVJP) — lead only: the files record kernel timeouts that forced `opaqueA*`/`rNNPreK`.

---

## 7. `LeanMlir/Proofs/Nets/MobileNet/` + `Nets/EfficientNet/`

(Mathlib items — `relu6_hasFDerivAt`, `sigmoidScalar`, `pdiv_sigmoid`, the 63 `fun_prop` twins — are in §0.)

### MobileNetV2.lean:1029 — `Mnv2Live.bn13_window` (42) → `ResNet34.lean:461 bnXhat_sq_le` (hoist to BatchNorm.lean) + `abs_lt_of_sq_lt_sq'`; 6 lines. ~35. verified.
### MobileNetV2.lean:186 — `pdiv_relu6` (+ MLP `pdiv_relu`) → one `pdiv_of_hasFDerivAt_mask`. ~22. verified.
### MobileNetV2.lean:1418 — `sum_flatten8` is the (2,2,2) instance of the general `Tensor3.sum_flatten` (same proof). verified.

### Near-clones (MobileNet/EfficientNet)
- **BN γ/β conjunct pair** — two ~7-line conjuncts proven by `ResNet34PoCB.bnGammaGradB_den`/`bnBetaGradB_den`, repeated 48× here (MobileNetV4StepTieB 18, MobileNetV2StepTieB 10, EfficientNetStepTieG 10, EfficientNetStepTie 10) and ~29× elsewhere (ResNet50StepTieB 11, Cifar8BnStepTie 8, ResNet34StepTieB 6, CifarBnStepTie 4) → `def BnPairTiedB … : Prop` + 2-line `bnPairTiedB_holds`. **~1,200. verified** (instance check on `mnv2Stride1TiedB`).
- **Fused-SGD files re-prove the un-fused ones** — `StableHLO.lean:3400–3441` `*SgdB_eq_grad` (`rfl`, `@[simp]`) already says fused = θ − lr·unfused, yet `EfficientNetFold.lean` (8 lemmas, 190 lines) duplicates `EfficientNetFoldG.lean` (each is `rw [depthwiseWeightSgdB_eq_grad, EnetPoCG.depthwiseWGradB_den]`), and `EfficientNetStepTie`'s block ties re-prove `EfficientNetStepTieG`'s (`enet_exp_tied` from `enet_exp_tiedG` in 12 lines). ~340–640. verified (4 lemmas + 1 tie).
- SGD-wrapped `.correct` restatements (`*_render_*_certified`): MobileNetV2Close ×9, EfficientNetClose ×6, MobileNetV2ChainClose ×9 (+23 elsewhere) → one `HasVJP.sgd_certified`. ~250. verified. (Most are AuditAxioms-only — §0.7.)
- Exact duplicates: `convBn'_has_vjp`/`_differentiable` (MobileNetV2:297/308) = `CNN.lean:905/917` (`rfl`); `convStridedBnRelu6PC_has_vjp_at` (MobileNetV2WholeBackCertifiedTie:59/84) = `convBnRelu6StridedPC_has_vjp_at` (MobileNetV2FullVJP:54/77) (`rfl`); `conv2d_1x1'` (MobileNetV2:1205) = `MnistCNN.lean:393`; `relu6MaskB` (MobileNetV2StepTieB:83) = `reluMaskBack (0 < pre i ∧ pre i < 6)`; `EfficientNetFoldG.lean:63/76/89/101` restate `ResNet34PoCB.*_den` (so do ConvNeXtFoldGB ×9, ViTFold ×2). ~130. verified.
- Whole-net straight-chain apex per depth (MNv2 ×2, MNv4, R34, ConvNeXt) — lead (kernel-timeout history).

---

## 8. `LeanMlir/Proofs/Nets/ViT/` + `Nets/ConvNeXt/`

(Affine-Jacobian and `fun_prop` items are in §0.1/§0.2.)

### ViTClose.lean:504 / 549 / 815 / 896 / 195 / 215, ViTVecLN.lean:791/815, ConvNeXtFold.lean:53, ConvNeXtFoldG.lean:68, ViTFoldG.lean:142 — grad bridges collapsing sums by `sum_eq_single`
**Replace with:** `Finset.sum_ite_eq` / `sum_ite_eq'` / `sum_ite_irrel` under `simp` (e.g. `vit_render_pos_certified := by simp [pdiv_patchEmbed_pos]`; `posEmbedGrad_den` 15 → 1)
**Lines saved:** ~200   **Confidence:** verified

### ConvNeXt.lean:54 — `layerScale γ` = Pi multiplication `(γ * ·)`; `layerScale_grad_gamma` (ConvNeXtClose:96) and `Foundation/BackwardMaps.lean:46 diagBack` are the same function (all `rfl`). `pdiv_layerScale_gamma` = `pdiv_layerScale` with arguments swapped. ~35. verified.
### ConvNeXtChannelLN.lean:99 — `transposeFlat_diff` = `transpose_flat_diff` (Attention:85). verified.

### Near-clones (ViT/ConvNeXt)
- Pre-LN residual sublayer ×4 (ViTVecLN:122–365 vector-LN; Attention.lean:2084–2356 scalar-LN, verbatim apart from the LN) → `preLNRes ln F := biPathMat id (F ∘ rowwise ln)` + two 7-line lemmas. ~370. verified.
- heads = 1 MHSA backward collapse (ViTBackB0:317/330/386, ~180 lines at `maxHeartbeats 4M`) re-proves the general-heads one (:521/537/584) → a 3-line `mhsaBackCollapsed = mhsaBackCollapsedMH N 1`. ~170. verified.
- `vitNetBackGraph_faithful` proved twice (ViTBackB0:2095 bespoke induction ~100 lines vs ViTBackNet:355 `_via_fold`, identical statement; the ViTBackB0 trio is cited only in docstrings + AuditAxioms). ~155. verified.
- 2-block vector-LN net = depth-k net at k = 2 (ViTVecLN:411, ViTMultiHead:472; graphs `rfl`-equal incl. SSA prefixes). ~115. verified.
- Backward-uniqueness lemma, 7 copies of 3 statements (ConvNeXtBackCertifiedTie:128, EvenKernelConvBack:177, ViTBackB0:237/243/253, BatchMapVJPAt:238, ResNet34BackCertifiedTieB:80, StableHLO:4428) → one `*_backward_unique_of_eq` per structure in Tensor.lean. ~30. verified.
- Vector-LN backward seam, 4 statements of one fact (ViTWholeBackCertifiedTie:59 and ConvNeXtWholeBackCertifiedTie:187 are literally the same statement). ~30. verified.
- Term-mode apex twins (`vitApexVJP` vs `vitForwardKV_has_vjp`; `chanLNTensor3_vjp_chain` vs `chanLNTensor3_has_vjp`) → define the witness by the term. ~60. verified.
- Six backward-unfold `rfl` lemmas written 12× (ViTBackB0, ViTMhsaBackCertifiedTie, ViTVecLNBackCertifiedTie). ~70. likely.
- ViT ↔ ConvNeXt fold aliases (`veclnGammaGrad_den` = `headLnGammaGrad_den`, `headWGradB_den`, `headBGradB_den`, `rowDenseBiasGradB_den_lnbeta` = `headLnBetaGradB_den`) — identical statements in ViTFoldG/GB and ConvNeXtFoldG/GB.
- Dead (0 consumers anywhere): `hasVJP_backward_det` (ViTBackB0:253), `vitCotB2out` (ViTChainClose:135), `cnxStemPatchO` (ConvNeXtStepTie:341), `cnxD11` (ConvNeXtWholeBackCertifiedTie:403), two doc `rfl`s (ConvNeXtChannelLN:226/230).
- Leads: depth-12/18 whole-net step ties unrolled by hand (ViTStepTie:513/712, ViTStepTieGB:509, ConvNeXtStepTie:504, ConvNeXtStepTieGB:476, ~500) → recursive tower tie; the 1-head scalar-LN chain (54 AuditAxioms-only declarations) superseded by vector-LN — keep-or-retire.

---

## 9. `LeanMlir/Proofs/Codegen/`

### StableHLOLex.lean:83–178 — the whole decimal codec (`digitVal`, `dstep`, `toDigitsCore_suffix`, `foldl_dstep_toDigitsCore`, `lt_ten_pow_succ`, `toString_toList`, `parseNat`, `parseNat_toString`, …)
**Replace with:** core `Nat.ofDigitChars` + `Nat.ofDigitChars_ten_toDigits` (`Init/Data/Nat/ToString.lean:289/337`), `Nat.toList_repr` (:238); or `Nat.toNat?_repr : (Nat.repr n).toNat? = some n` (`Std/Data/String/ToNat.lean:320`)
**Why it matches:** the file's docstring says there is "no off-the-shelf `(toString n).toNat? = some n`" — on 4.34 there is. `Nat.ofDigitChars` is the same `List Char` Horner fold and still evaluates under `decide`.
**How:** `def parseNat (s : String) := Nat.ofDigitChars 10 s.toList 0`; `parseNat_toString := by simp [parseNat, Nat.ofDigitChars_ten_toDigits]`; delete the other 8 declarations and the "why a from-scratch codec" section
**Consumers:** 1 (`tests/AuditAxioms.lean:680`)   **Lines saved:** ~85   **Confidence:** verified

### StableHLOParse.lean:199–309 — `parseStack_toToks`, ~100 identical case arms → `induction r <;> intro ts st <;> simp only [toToks, List.append_assoc, *] <;> rfl`. ~105. verified.

### StableHLO.lean:1499 / 1555 / 1590 / 1677 / 1682 — `rowSoftmaxFlat`, `patchEmbedFlat`, `patchEmbedBackFlat`, `clsSliceFlat`, `clsPadFlat`
**Replace with:** `Proofs.patchEmbed_flat`, `patchEmbed_input_grad_formula`, `cls_slice_flat`, `(cls_slice_flat_has_vjp N D).backward`, `Mat.flatten (rowSoftmax (Mat.unflatten ·))` (Architectures/Attention.lean)
**Why it matches:** the comment says these are "LOCAL re-spelling… so StableHLO needn't import Attention", but StableHLO imports `Foundation.IR`, which imports `Architectures.Attention`; all five are `rfl`-equal. The tie lemmas `ViTBackB0.patchEmbedBackFlat_eq_backward` (:2046), `ViTFwdGraph.rowSoftmaxFlat_flat` (:404) and `tests/TestSoftmaxRow`'s become unnecessary.
**Consumers:** 94 refs / 11 files (`abbrev` keeps them)   **Lines saved:** ~100   **Confidence:** verified

### StableHLO.lean:2407–2574 — 32 `@[simp] theorem den_batchOp_<op> … := rfl` → one `@[simp] den_batchOp : den (.batchOp op e) = batchMap N (denOp op) (den e) := rfl` + `denOp`'s equation lemmas (restated `stemGraphB_faithful`, `mbResidGraphB_faithful` close). ~150 uses / ~20 files. ~110. verified.
### StableHLO.lean:113 / 118 / 2468 — `batchSlice` = `Mat.unflatten` (`rfl`; `abbrev` keeps 736 sites); `batchSlice_batchMap` → `congrFun (Mat.unflatten_flatten _) n`; `batchMap_pointwise` → `Mat.flatten_unflatten`. The 8 `row*Flat` defs (1499–1548, 1706–1712) are `batchMap`/`batchMapAux` instances (`rfl`). ~37. verified.
### StableHLO.lean:5529 — `lookupEntry` = core `List.lookup`. verified.
### AdamStep:62 / GradClip:91 / Lamb:114 — `adam_denom_pos`, `clipDenom_pos`, `lambDenom_pos` (same statement ×3) → `add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε`. verified.
### Lamb.lean:162 — `lambScale_not_shared` (hand `√4 = 2`) → `norm_num [lambTrust, gradSumSq]` via the `RealSqrt` extension. ~11. verified.
### RmsPropStep.lean:120 — `rmsSqNext_nonneg` = `adamVNext_nonneg …` (`rmsSqNext = adamVNext` by `rfl`). verified.
### DropPath.lean — `dropPath_ones_id`, `_zeros_zero`, `_vjp_is_self`, `_has_vjp_correct` are the `dropout_*` lemmas (`dropPath = dropout ∘ dropScale` by `rfl`; the two `*_has_vjp_correct` have 0 consumers); `keepProb_last` → `Nat.cast_sub`. ~25. verified.
### Zero placeholders — `zK/zD/zV/zM/zT`, `zVB/…`, `zVv/…`, `zVb/…` (17 defs) + ~20 in tests + 48 `let zrnd := fun r => r` → Pi `0` (`Pi.instZero`) and `id` (computable; `skel` erases them, bytes unchanged). verified.

### Near-clones (renderers; all must stay byte-identical — enforced by `#guard`s and the CI diff, not by `rfl`)
- AdamW tail ×6 (`adamOneM` MNv2RenderB:552, `adamOne4` MNv4RenderB:999, `enetAdamOne` EfficientNetRender:1098, `convnextAdamOne` ConvNeXtRender:867, `vitAdamOne` ViTRender:598, `ResNet34RenderB.optOne .adamw` :610) → one `adamTail`; AdamW constants md5-identical ×3 (MNv2:600, MNv4:1013, ENet:1150) = `optConstsB .adamw`; RMSProp tail ×2. ~180. likely.
- Op-template `let`s re-declared per renderer (`dg` ×16, `convFwd` ×8, `convBack` ×7 — md5-identical at StableHLO 9780/9973/10262/10484 + CnnRender:280 — `convWGrad` ×7, `selMask2` ×8, `reduce0` ×8, `dense` ×9, `sgd` ×10) → top-level helpers. ~300. likely.
- bf16/fp8 constructor switch written out at ~250 sites → reducible smart constructors (`convP`, `denseRowP`, …). ~150. likely.
- BN-parametrised forward-graph twins (EfficientNetRenderPC ↔ …PCEval; MobileNetV2RenderPC ↔ …PCEval ↔ `StableHLO.mobilenetv2FwdGraphFull`; ResNet34RenderPC ↔ `StableHLO.resnetFwdGraph`). ~400. likely.
- `R34Bn` (ResNet34RenderB:81) ≡ `StableHLO.BnMode`; `PGrad` ≡ `PGradM` ≡ `PGradV4`; four `*DropSig`; `vlnFwd(B)` ≡ `headLnFwdSite(B)`; tensor-type string helpers ×4 families (⚠ `IRPrint.tt`/`MlirCodegen.tensorTy` print `tensor<xf32>` for `[]`); `fmt6`/`fmt12`. ~90. likely.
- Lead: `IRPrint.lean` (1.9k lines, no theorems, superseded by `StableHLO.pretty`, cited only in prose).

Clean: MatBridge (its proofs can be `rfl`; 0 consumers), AdamRender, SgdMomentumStep, LambTriple, MlpRender, ResNet34RenderPC, ResNet50RenderB.

---

## 10. Non-proof Lean (`LeanMlir/*.lean`, `tests/`, `jax/`)

### GradcheckHelpers.lean:17–52 — `parseFloat` (+ `pow10`, `digitsToNat`, `splitAtChar`)
**Replace with:** core `Lean.Syntax.decodeScientificLitVal?` (`Init/Meta/Defs.lean:1008`) + `Float.ofScientific`, with `String.toNat?` for bare integers
**Why it matches:** same token grammar; bit-identical to Lean's own literal elaboration on 7/7 values, where the hand parser is 1 ulp off on 5 (double rounding). **And it fixes a defect** (re-verified): `parseFloat "abc" = 5451.0`, `"nan" = 6752.0`, `"inf" = 6374.0` — but `tests/TestSgdRenderTie.lean:67–72` relies on "yields 0.0 on anything it cannot read" to reject a mistyped learning rate, so that guard never fires.
**How:** `def parseFloat? (tok) : Option Float := … match Lean.Syntax.decodeScientificLitVal? body with | some (m, s, e) => some (Float.ofScientific m s e) | none => body.toNat?.map Nat.toFloat`; `parseFloat tok := (parseFloat? tok).getD 0.0`
**Consumers:** 1 + TestSgdRenderTie; demo copies at `demos/MainPlantLeaf.lean:217`, `demos/MainAraslSigns.lean:135` (these also panic on `"-0.5"` and return `+0.5`), `demos/probes/MainSegLossProbe.lean:38`   **Lines saved:** ~45   **Confidence:** verified

### tests/DocstringCheckRefs.lean:159 / tests/TestYolov1Mutex.lean:29 — `containsSubstr` / `hasSubstr` → core `String.contains` (`Init/Data/String/Search.lean:300`); 45 inline `(s.splitOn x).length > 1` (incl. `VerifiedTrain.lean:311–336`). verified.
### tests/DocstringCheckRefs.lean:224 — `leanFiles` → `System.FilePath.walkDir` (`Init/System/IO.lean:1181`); same file set on LeanMlir (262), tests (120), jax/Jax (2), demos (44). verified.
### tests/DocstringCheckRefs.lean:196 — `endsWithComponents` body → `List.isSuffixOf` (proved equal). verified.

### Near-clones (program code)
- **vjp oracle** — `vjpCfg` has 28 byte-identical copies and the 14 test nets are defined twice (`tests/vjp_oracle/phase3/*.lean` and `jax/tests/vjp_oracle/phase2/*.lean`). The oracle is only valid if both sides train the *same* `NetSpec`; today that holds by copy-paste. One `LeanMlir/VjpOracleNets.lean` importable from both packages (`jax/lakefile.lean:9` requires the root). ~380. verified.
- `tests/TestCifar8WideTrain.lean` (652) is `TestCifar8AdamTrain.lean` (693) at `D1 = 512` with a different slug (~25 helpers byte-identical). ~600. verified.
- `tests/TestMHSA.lean:41/73` copy `LeanMlir/ViTRender.lean:162/190` — the test checks a copy, not the emitter that ships; `TestMHSA`/`TestSDPA` re-implement `adjointGradcheck`, which is itself `adjointGradcheckFixed … []`. ~165. verified.
- LE byte codecs inlined (SpecHelpers:188, VerifiedTrain:783–897, Train:587–610; demo copies) → `packXShape`, `F32.readLabel`, `pushF32LE`, `pushU32`. ~35. verified.
- Buffer-compare helpers ×7 in tests; 11 inline `iree-compile` blocks missed by the `Types.compileCheck` lift; `mkLabels` ×4. ~120.

---

## 11. Incidental defects found by the audit

1. **`ViTGradcheck.parseFloat` returns garbage on non-numeric input**, so `TestSgdRenderTie`'s mistyped-lr guard is inert (§10). Re-verified by `#eval`.
2. ~~**Four test comparators take the magnitude over one buffer only** (`TestR50AccumTie`, `TestR50AccumShardTie`, `TestChannelLN`, `TestConvBiasZero`) — the exact bug `TestDropPathTie`'s docstring records producing a false "logits did not move".~~ Closed (§10 row).
3. ~~**The vjp oracle's two sides agree only by copy-paste** (§10).~~ Closed: `LeanMlir/VjpOracleNets.lean`.
4. ~~**`TestMHSA` validates a copy of the MHSA emitter**, not `LeanMlir/ViTRender.lean`.~~ Closed.
5. **Stale justifications in docstrings:** `StableHLOLex` ("no off-the-shelf `toNat?` round-trip"), `StableHLO` ("so StableHLO needn't import Attention"), `TestYolov1Mutex` ("core lacks containsSubstr"), `VerifiedTrain.lean:1201` + three demos ("no `String.toFloat?`" → integer-unit env vars).
6. ~~**An unproven claim that is one line away:** σ′ = σ(1−σ) (`sigmoidScalarDeriv`, `StableHLO.lean:6945`, the swish closed form) follows from `Real.deriv_sigmoid`.~~ Closed: `sigmoidScalarDeriv_eq`, `swishScalarDeriv_eq`.

---

## Summary

About 6,900 declarations were enumerated, and roughly 5,400 of them sit in the proof layer; the 1,480 program-code definitions were skimmed, and about 80 were read closely. Mathlib has **no drop-in replacement for the base layer** (`Vec`/`Mat`/`pdiv`/`basisVec`/flatten), which is `rfl`-compatible with Mathlib already and costs more to swap than it saves. It also has none of `UpstreamDraft.lean`'s 19 CDF lemmas and no softmax, GELU, tanh derivative, SVD, CDF quantile, Cameron–Martin, Neyman–Pearson or Clopper–Pearson, so those are genuine.

The duplication is instead in proofs. **About 160 declarations or blocks can be replaced by a named Mathlib/core lemma or tactic.** The largest single one is `ContinuousLinearMap.fderiv` through a 5-line `pdiv_of_affine`, which removes ~2,900 lines of hand-derived affine Jacobians. After that come `fun_prop` for ~100 differentiability side conditions and `Filter.eventually_all` for the four kink/maxpool linearisations. Then there are the `Finset.sum_le_card_nsmul`, `Equiv.sum_comp`, `sum_ite_eq` and `Prod.ext_iff` idioms at ~250 sites, `Real.sigmoid`, core `Nat.ofDigitChars`, the PSD/`Integrable.mul_bdd`/`integral_comp_eval` measure-theory lemmas, and `gcongr`.

**About 60 near-clone families** account for most of the rest:
- the kernel/bias slots of `SgdDescentCnn`
- nets not using the repo's own `CertLayer.comp`
- the BN γ/β tie pair repeated 77×
- fused-SGD files re-proving the un-fused ones
- 148 + 72 proof blocks emitted by two certificate generators

The auditors' estimates total **~18,000–20,000 removable hand-written lines**, about 10% of the ~183k hand-written lines (129k proof layer + 54k program/test code); **~12,000 of those rest on typechecked evidence**. Another ~1,600 lines come out of generated output once the generators call the shared lemmas, and ~1,800 more are restatements that are consumed only by `tests/AuditAxioms.lean`, a keep-or-retire decision for you.

No file examined was entirely clean of findings except these:
- Foundation: OpaquePrefix, CertifiedChain, UpstreamDraft, ListDot, DataParallelNode, BackNetFolds, ConvLossFold, Bf16GradNodes
- Float: Bf16Fold, E4M3Fold, Binary32Instance
- Architectures: SE, Residual, ChannelLNBack
- SgdDescentCifar
- a few thin per-net delegation files

Landing note: `Tensor.lean` is a root file with 423 downstream modules, so batch the Foundation edits and gate them with `lake build Certs`, not bare `lake build`.
