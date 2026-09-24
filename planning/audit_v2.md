# audit_v2.md — reuse v2 + clarity / abstraction boundaries

Audit run 2026-09-23 on `proof-cleanup` @ `51aa502f` (Lean 4.34.0, Mathlib `v4.34.0`), oleans freshly
built. Seven parallel auditors, one per slice (Foundation+Float, Architectures+Training, Certificates,
Codegen, Nets R34/R50/Small/ConvNeXt, Nets MNv2/MNv4/ENet/ViT, cross-cutting + non-proof Lean), each
running two lenses over its files:

- **Part A — reuse v2**: the 2026-09-18 Mathlib-reuse prompt again, minus everything
  `mathlib_reuse_audit.md` landed or rejected; v1's still-present *likely*/*lead* items resolved.
- **Part B — clarity**: could a Lean + ML reader new to the repo find where a concept lives and which
  file to edit? Misplaced ownership, files with several jobs, leaky abstractions, one concept with
  several spellings, suffix schemes, stale entry points, import direction, empty indirection.

"verified" = typechecked in a scratch file against the HEAD oleans (scratch dirs listed in §9).
Nothing in the repo was edited by the audit. Pinned-name costs were counted per item.

**Headline.** Part A is small now — v1 got the Mathlib wins; what remains is ~1.1k proof lines of
in-repo factoring plus ~1.5k lines of program-code dedup. Part B is where v2 lives: **the repo's shared
kits sit in whichever file first needed them** (ResNet-34, EfficientNet, StableHLO, a Training
descent file), and that single cause drives most of the wrong-way imports, the rebuild fan-out, and
the "where is X?" problem. The fix is mostly *moves with names kept* — no pinned statement changes.

## Status (2026-09-23, landed on main)

| commit | what landed |
|---|---|
| `a745b694` | §0.2 — `sigmoidScalarDeriv_eq` restored; it and `swishScalarDeriv_eq` pinned. §1 — the verified import drops (`LipschitzCert` off `Tensor`; 7 renderers off `LeanMlir.ViTRender`; `SgdDescentCnn` off `MobileNetV2Close`; ENet whole-net tie; `PairSDP`). ⚠ `ViTFold` → `Cifar8Fold` was NOT free — it now imports `MlpTrainStep` (the lemmas' home) instead |
| `d3688f85` | §4 unpinned dead code (−894): ResNet34 strided family + helpers, the three `*Structured` renderers, `mobilenetv2FwdGraph`, LinearTrainStep's render scaffold, 14 small decls, `Cifar8Fold.lean` (hub) |
| `4530a0df` | §2.3 root batch 1/2 (moves, names kept): relu6 → MLP, sigmoid/SE gate → SE, layerScale → LayerNorm, softmax Jacobian → new `Softmax.lean`, multi-head slab kit + `HasVJPMat3` → Attention. StableHLO imports no net; IR imports Softmax only. ⚠ `pdivMat_transpose`/`scalarScale` + the matmul/scale/transpose VJPs must STAY in Tensor (the architecture-free comparator tier cites them) — the first cut took them and the comparator caught it |
| `134894bc` | §6 root batch 2/2: `pdiv_of_hasFDerivAt_mask` (relu + relu6), `relu_apply_eq_max`/`relu_nonneg`/`relu_entry_lipschitz` in MLP, `mlp_has_vjp_at` on `vjp_comp_diff_at`, `pdiv_lift_sum`, `bnMean_eq_expect`; three more ResNet34 imports gone. Gated incl. the local comparator (3 tiers okay). `sum_finProdFinEquiv₃` skipped (≈0 lines) |
| `62cca9ef` | §3 docs: the Proofs/README chain table + suffix legend + tier glossary, new `Codegen/README.md`, Certificates family map, lakefile claims, ENet headline, the all-reduce headers, history-first module docs → contents, 21 file:line cites → names, 8 dangling refs |
| `bcb18000` | §0.1 — MBConv SE width `ic/4` everywhere via `Spec.mbConvSeMid`; `Train.lean` sizes from the buffer and throws if `totalParams` disagrees |
| GradNodesB commit | §2.4 — `Foundation/GradNodesB.lean`: `ResNet34FoldB` + `EfficientNetFoldG` + `MobileNetV2FoldPaperG` (all three deleted) + `CnxPoCGB.psWGradB_den`, names kept; the 13 forwarders deleted (ConvNeXtFoldGB 10, EfficientNetFoldG 3) and their callers + pins moved to the canonical names (AuditAxioms 1,597 → 1,584). `Bf16GradNodes` imports `GradNodesB` + `ViTClose` (not `ConvNeXtFoldGB`). ⚠ the ViT row-dense / vector-LN nodes stay in `ViTFoldGB` — their per-example bridges are in `Nets/ViT/ViTClose`/`ViTVecLN`; moving them needs those bridges in Architectures first |
| BatchedBackLinks commit | §2.5 part 1 — the batched kit leaves the EfficientNet files, names kept. `BatchMapVJPAt` gains `batchMap_has_vjp` / `reindex_has_vjp` / `bnBatchLA_has_vjp` / `flatConv_has_vjp` (still StableHLO-only); new `Foundation/BatchedStages` (`cbsB` `stemB` `dwbsB` `dwbsSB` `seB` `projB` + their VJPs, from `EfficientNetRenderPC` / `EfficientNetChainClose`); new `Foundation/BatchedBackLinks` (`residualBackGraph`, the batched backward `_faithful` + stage backward graphs from `EfficientNetBackB0`, `EnetTiePoC`'s cotangent steps, `ResNet34TieB`'s `reluMaskB`/`cStridedInB`/`bnInB`/`mpInB`/`rowB`/`unrowB`). ResNet-34's and MNv4's T3 ties no longer import the EfficientNet tie; `EfficientNetStepTieG` no longer imports ResNet-34's. Gated incl. the local comparator (3 tiers) |
| StageLayers commit | §2.5 part 2 — names kept. `ResNet34BackCertifiedTie` → `Architectures/ConvBackCertifiedTie` (imports `BatchedStages` + `BackwardMaps` only). New `Foundation/BatchedStageLayers`: MNv2's relu6 stages + `projLayer` and R34's relu / strided-relu / strided-projection stages, each with `_at` VJP, backward graph and `CertLayer` — `ResNet34BackB0` stops importing `MobileNetV2BackB0`, which stops importing `EfficientNetBackB0`. `CertLayer.comp_ok_of` → `CertifiedChain`; `r34PoolLayer` + `R34PoolSmoothAt` → `HeadLayers` (which drops `EfficientNetChainClose` for `BatchedStages`); the batched pool backward (`maxPool3s2FlatBackB`, `den_maxPool3s2BackB_eq_flatBackB`) → `BackwardMaps`; `R34FullBSeal`'s nine stage lemmas (`projB_zero_const`, `sealProj*`, `cbReluStridedB_eq`, four `*_continuous`) → `BatchSealKit`. Also yaml: `softmaxCE_grad`'s file (stale since `4530a0df`) |
| SyncKit commit | §2.5 part 3 — the sync-BN kit, names kept. New `Foundation/DataParallelSyncKit`: `ResNet34SyncB`'s index cast / non-BN shard lemmas / BN sync site, `ResNet34SyncStepTieB`'s per-op homogeneity, shard, P4, per-parameter `*Sync` + `_of_scaled` and divisor lemmas, and all of `MBConvSyncTieB` (deleted). MNv2's, MNv4's and ENet's sync forward twins and MNv2's / ENet's step twins now import the kit, not ResNet-34's sync files. The P4 proof written 7× more → one `shard_sum` tactic macro in `DataParallelSync` (8 uses) |
| SgdNodes commit | §2.5 part 4, closes §2.5 — names kept. New `Foundation/SgdNodes`: the per-example fused-SGD node lemmas (`Cifar8PoC.denseW/B_den` out of `MlpTrainStep`, `CifarPoC.convW/B_den` out of `CifarFold`, all of `CifarBnFold` — deleted); `ViTFold` / `ConvNeXtStepTie` drop `MlpTrainStep`. New `Foundation/IndexCast` (`castIdx`, `den_castIdx`, `laAssoc`); `den_cast` deleted (`den_reassocS`/`den_unassocS` go through `den_castIdx`); ConvNeXt's two renderers' private casts are `castIdx` — artifacts byte-identical. The Vec-level `reassocB` / `reassocFwd` / `reassocBack` stay (functions, not graph transports); `IndexCast`'s header maps all of them |
| ConvIndex commit | §2.1 — names kept. `SgdDescentCnn`'s index plumbing + 2×2 max-pool window facts + `sum_s2` → `Architectures/ConvIndex` (imports `Architectures.CNN` only); conv-as-dense + float conv forward (`convPad`…`flatConvF_close`) → `Float/ConvFloat` (on `ConvIndex` + `FloatBridge`), `convPad` itself into `ConvIndex`; `FloatClose` + `.comp` / `.of_close` + the `relu` / `id` / `iterate` instances → `Float/FloatClose` (on `FloatBridge` alone); `add_close` → `FloatBridge` beside `mul_close`. `ResNet34FloatBridge` imports `ConvFloat`, not Training, so the float tier no longer imports Training except `Binary32Instance` (a real use: it instantiates the linear descent at binary32). The padded read: `convWindow3` and `IBP.convTap` are now `convPad` (with `convWindow` / `dwWindow` already) — one definition; `convTapQ` stays, the computable ℚ checker. ⚠ NOT done: the `SgdDescentCnnFloat` split — the float budgets are woven into every rung's capstone, not a contiguous block, so a real/float cut does not exist in dependency order |
| Batched commit | §2.2 + §2.6 part 1. §2.2: `batchMap`/`batchSlice`/`batchMapAux`/`bnBatchLA`/`batchMap_pointwise` → `Foundation/Batched` (on `PerChannelBN`; namespace `StableHLO` kept); `BatchMapVJPAt` imports `Batched` only; root-file comment `CifarBnFold` → `SgdNodes`. §2.6 Foundation: `BackNetFolds` deleted (`cnxBlockChLayer` → `ConvNeXtBackB0`; MNv4 imports `ResNet50BackB0` directly); `EvenKernelConvBack` → `Architectures/` (⚠ not Nets/ConvNeXt as proposed: ViT cites it too, it is an op fact); `SpecVJP` → `Proofs/SpecVJP.lean` (the apex); `PerChannelBN`/`StridedConv` → `Architectures/`; `crossEntropy_differentiable` → `Architectures/Softmax` (⚠ renamed `StableHLO.` → `Proofs.`: its old namespace was `LinearTrainStep`'s; one pin updated). §2.6 Certificates: new `Certificates/DenseEuclid` (the dense/ReLU engine + `CertifiedAt`, which `lipschitz_cert_scorecard.py` no longer emits — regenerated diff = the 9 removed lines); `IntervalBound` and `LipschitzCertPairSDP` import the engine, not the trained weights; `mlpT_logit_continuous` → beside `mlpT` (the witness generator emits the import); new `Certificates/GaussianQuantile` (Φ/Φ⁻¹ API from `SmoothingGaussian` + `SmoothingMC`, both `IsOpenPosMeasure` instances from `SmoothingNetSemantics` — the 1-D one now above its use, so `stdGaussian_Ioo_pos` is one line). `UpstreamDraft` (Mathlib-PR staging) deliberately not wired in |
| §8 deletions commit | §8 owner decisions executed. `ResNet34.lean` deleted (`resnet34_has_vjp_at` + `chainComp`/`ChainData`/`chain_vjp_diff_at`; LeanMlir.lean now links `resnet34ForwardB_full_has_vjp_at`). IBP flat-`Vec` residue out of `IntervalBoundConv` (`BoxSound`/`.comp`, `boxSound_id`, `denseLoV/HiV(_uniform)`, `flatten_reluT`, `unflatten_*_const`, `CertifiedAtLinfV(.mono)`; ⚠ flat `InBox` STAYS — `BoxSound3V`'s output box uses it). The three `*_correct` wrappers. Rank-3 kit out of `Tensor` (`vjp3_comp(_at)`, `HasVJP3.toHasVJPAt3`, `pdiv3_add`, `pdiv3_id`, `identity3_has_vjp`, `biPath3(_has_vjp)`, `pdiv_clm`, `HasVJPMat.backward_unique`, and `pdiv3_comp` — dead once `vjp3_comp_at` went) + their THREE blueprint nodes; ⚠ that made §9.2 stop citing ch 3, so ch 3 lost its double border in Figure C.1 and `\depgraphCitesCnn` is no longer generated — caption rewritten. MNv2 legacy chain: `IVPos`/`IVNoExpPos` → `MobileNetV2FullPaper`; `MobileNetV2FullVJP`, `MobileNetV2BackCertifiedTie`, `MobileNetV2WholeBackCertifiedTie` deleted (TieB now imports `ConvBackCertifiedTie` directly). `tests/Audit{Bridge,Mutation,Probes,Sanity}.lean` deleted; `AUDIT_REPORT*.md` → `planning/archive/audit_reports/` (the yaml cites them). `lean_lib «Codegen»` → `«Reference»`. AuditAxioms 1,584 → 1,573. Gated incl. AuditAxiomsHeavy 62/62, blueprint PDF, local comparator (3 tiers) |
| §2.6 Codegen-moves commit | §2.6 part 2 — `Proofs/Codegen/` keeps only renderers + IR. `AdamStep`/`SgdMomentumStep`/`RmsPropStep`/`GradClip`/`Lamb` → `Training/Optim/`, `DropPath` → `Training/` (a regulariser, not an optimizer); `MatBridge` → `Foundation/` (kept: opt-in interop); `MobileNetV2RenderPC(Eval)` → `Nets/MobileNet/MobileNetV2StagesPC(Eval)`; `UibSpec` + `mnv4Blocks` → new `Nets/MobileNet/MobileNetV4Spec` (MNv4's proof chain stops importing the 9-artifact renderer); the `#eval` writers of `MlpRender`/`CnnRender` → leaf `MlpArtifacts`/`CnnArtifacts` (lakefile roots, regen list and drift guard follow; artifacts byte-identical after re-running them). ⚠ `verified_mlir/MANIFEST.md` is STALE since `5a4e7778` (2026-08-28: 231 vs 248 artifacts, ungated) — not regenerated here. `EfficientNetRenderPC(Eval)` stay: they hold typed graphs |
| §2.6 one-job commit | §2.6 part 3 — one job per file. `PerChannelBN`'s sync-BN SHARDING (19 decls: `batchShard`, `bnShardEquiv`, the per-shard statistics and every shard = global identity) → `DataParallelSync`; the sync-BN op and its at-own-stats facts STAY (`StableHLO`'s `den` reads them). The 14 op facts in `BatchSeal` → their op files AND out of the namespace (user call: `Proofs.BatchSeal.x` → `Proofs.x`; unpinned): BN bounds/`bnMean_pair`/`bnIstd_*` → `BatchNorm`, `relu_continuous` → `MLP`, residual continuity → `Residual`, pool shift/nonneg/continuity → `MaxPool3s2`, `globalAvgPool_shift` → `CNN`, `batchMap_continuous` → `Batched`. `ViTVecLN`'s vector-LN op + VJP + per-token lift + γ/β gradients (incl. pinned `vit_render_vecln*_certified`, names kept) and `pdiv_id_add_const` / `pdiv_maskGather_add_const` (out of `ViTClose`) → `Architectures/LayerNorm` — ⭐ `ConvNeXtChannelLN` stops importing the ViT chain. Tensor's SDPA item was already done by `4530a0df` (what remains is comparator-cited) |
| Certs-cone commit | §2.6 part 4 — `Certs` no longer builds the trainer or the FFI: it reaches 3 program modules, all import-free data (`VerifiedSpec`, `ParamLayouts`, `VerifiedNetsCore`), was 7 incl. `VerifiedTrain` + `IreeRuntime`. `VerifiedSpec` import-free (gains `VerifiedData`; `toNet` + the 10 IO forwarders → `VerifiedTrain` / `VerifiedAttack` / `VerifiedSmoothing`); the XLayout tables → new `ParamLayouts` (`IreeRuntime` re-exports); the specs → `VerifiedNetsCore` (git mv; `SpecVJP` imports it), `VerifiedNets` = Core + the program modules (apps unchanged). `VerifiedTrain`'s four jobs split on its banners: driver (+ fp8) stays, `VerifiedPgdGen` (the four hand-typed PGD-step StableHLO generators, now documented as unverified), `VerifiedAttack`, `VerifiedSmoothing`; `compileVmfb` / `loadData` / `gen*PgdStep` lose `private`. Book: one path (`LeanMlir/VerifiedNetsCore.lean`, ch 1 "the spec lives in …"). ⚠ `fwdRenderedBatch` (private, `VerifiedTrain`) has no caller — dead |
| edges commit | §2.6 part 5, closes §2.6 — ⭐ no Foundation file imports a net or a certificate (was 13/29 at the audit), and `StableHLO` imports no net. The chapter 3–4 graph ties (`cnnFwdGraph_faithful`, `cifar{,8,8Bn}FwdGraph_faithful`, `cnnBackGraph_faithful` + `maxPoolFlat_has_vjp_at'`) → new leaf `Nets/Small/ChapterGraphTies` (graphs stay in `StableHLO`; names kept). `CnnTrainStep` → `Architectures/ConvGrad`, `CifarBnClose` → `Architectures/PerChannelBNGrad`, `ViTClose` → `Architectures/TokenParamGrad` (git mv, names kept, now on Architectures imports only). `IntervalBound` + `CrownBound` → `Certificates/` (both declare into `LipschitzCertDemo`; the two scorecard generators re-run, diff = the 7 path lines). `SmoothedLossCot`'s `LinearTrainStep` import was dead. `relu_id_of_pos` (MnistCNN) → `MLP`. ⚠ Left: `Architectures/ChannelLNBack` → `Nets/ConvNeXt/ConvNeXtChannelLN` (needs the channel-LN op itself in Architectures) |
| seals fun_prop commit | §6 seals' continuity, −489 lines net (est. ~250). The op files tag their continuity facts `@[fun_prop]` (`relu`, `relu6`, `residual(Proj)`, `batchMap`, `maxPool3s2Flat`, `bnRowLA`, `rayX`) and gain seven atoms (`bnBatchLA`, `flatConv`, `flatConvStride2(Xla)`, `depthwiseFlat`, `depthwiseStride2Flat(Xla)`, `swish`) + `bnIstd_continuous` (the old `bnIstd_cont` took an index it never read, so `fun_prop` could not infer it — deleted). Each seal's `Rr_continuous` (and MNv4's `Q0_continuous`) is now one `fun_prop (disch := exact one_pos)` after unfolding the carrier to its atoms; the hand-composed `cn*` / `Z*_continuous` / block-continuity chains (R34 17, R50 22, MNv2 51, MNv4 38) and the four `R34FullBSeal` stage lemmas are gone. ⚠ Recipe: unfold to the ATOMS, never tag a block lemma — `fun_prop` then unifies at literal widths and times out (numeral-shape trap); MNv2/MNv4's interleaved `Z`/`A` defs need `repeat (first | unfold …)` |
| DenseEuclid commit | §6 dense bounds, −133 lines (est. ~150), statements + 5 pins unchanged. `denseE_lipschitzL2_of_sq` (every upper bound's L2 tail from a raw-sum `‖Wd‖² ≤ B²‖d‖²`), `quad_le_of_sq_matvec` / `quad_le_of_frob` (`⟨y, My⟩ ≤ c‖y‖²`), `sq_le_of_gram_quad` (the Gram step `⟨y, Gy⟩ ≤ c‖y‖² ⇒ ‖Wd‖² ≤ c‖d‖²`); Frobenius, Schatten-4 and Schatten-8 are each one term on them, and the Schatten-4 proof reuses `sum_sq_matTvec_eq` instead of re-deriving it |
| smoothing commit | §6 smoothing, −59 lines (est. ~85), statements + pins unchanged. `SmoothingGaussian`: the integrable-×-bounded proofs → `Integrable.mul_bdd` (one local `hWbdd` for the three weighted ones), the bounded ones → `Integrable.of_mem_Icc`, `htrans` → `MeasurePreserving.integral_comp'` (drops three measurability `have`s), `pi_gaussian_integral_eval` (unpinned) → Mathlib's `integral_comp_eval`. New `smoothing_certified_of_le` (any `q ≤ p` certifies radius `σ·Φ⁻¹(q)`), `smoothProb_eq_real` + `measurableSet_smoothRegion` (the indicator bridge), `hitCount_setOf` (CP) — MC, CP ×2 and NetSemantics use them. Gated incl. the local comparator (3 tiers) |
| ENet VJP commit | §6 ENet MBConv VJPs written twice, −52 lines. `EfficientNetChainClose` holds the one term-mode chain per shape: `mbStridedFwdB_has_vjp` (was tactic-built) and `mbExpFwdB` + `_differentiable` + `_has_vjp` (moved from `EfficientNetFullB0`, names kept); `mbResidFwdB_has_vjp` takes its body from `mbExpFwdB_has_vjp`. `BackB0`'s `mbBodyB_has_vjp` / `mbDownBodyB_has_vjp` (unpinned, uncited) deleted — the two body graph ties now state `mbExpFwdB_has_vjp` / `mbStridedFwdB_has_vjp`. Gated incl. the local comparator (3 tiers) |
| swish commit | §6 `BatchSealKit`'s second swish derivative, −29 lines. `swishD` (the quotient-rule formula) and `one_add_exp_pos` deleted; `hasDerivAt_swishScalar` (now one line, from `swishScalar_diff`), `swishScalarDeriv_pos` (was `swishD_pos`) and `swishScalar_lt` move to `Architectures/LayerNorm` beside `swishScalar`, proved on `Real.sigmoid_pos` / `_lt_one` / `_monotone`; `swishGap` and MNv4's seal use `swishScalarDeriv`. ⚠ namespace `Proofs.BatchSeal` → `Proofs` (unpinned). Gated incl. the local comparator (3 tiers) |
| reindex commit | §6 reindex VJPs — one Jacobian proof, not ~30 lines out (net +4). `reindexVJP` (+ `_backward_of_inv`) moves from `PerChannelBN` to `Foundation/Tensor` beside `pdiv_reindex`, with a bridge `reindexVJP_backward` to the masked-sum spelling; `reindex_has_vjp` keeps that spelling as its backward and takes `correct` from `reindexVJP`; ⭐ `StridedConv` stops importing `PerChannelBN` (it was only there for `reindexVJP`). ⚠ The audit's premise was wrong: `reindex_has_vjp` / `broadcastFlat_has_vjp` can't be `reindexVJP` aliases — their backwards are spelled as the IR's scatter ops denote, and `bnBatchLABack_faithful`, the SE-gate graph ties (`EfficientNetBackB0`) and MNv4's whole-net tie match them by `rfl` (as aliases MNv4's hits max recursion depth). `broadcastFlat_has_vjp` unchanged. Gated incl. the local comparator (3 tiers) |
| small-collapses commit | §6 small collapses, −47 lines (est. ~80). New `sum_finProdFinEquiv₃` (Tensor): `sum_flat3` (unpinned) deleted, `sum_w3` / `Tensor3.sum_flatten` one-liners, and the three hand-rolled triple re-indexings in `Depthwise` / `CNN` (the audit's "Depthwise / CNN GAP collapses") call it; `sum_s2` kept (pinned, already `sum_finProdFinEquiv`). IBP conv uniform boxes: `ite_sign_lo/hi`, `convTap_uniform_*` by `split_ifs`, `convLo/Hi_uniform` one `simp` each (names kept). `euclid_norm_sq` := `EuclideanSpace.real_norm_sq_eq` (pinned, name kept). `layerNormVec_has_vjp`'s `h3` by `fun_prop`. `mulVec_headPadMat` on `sum_finProdFinEquiv` + one `simp`. `mhSlab` → `headSliceMat` (one spelling; ≈ +2 lines). ⛔ Skipped: GramQ `rowDotQ`/`castM` ≡ IBP `sumQ`/`castV` (needs a shared ℚ module both import, generated certificates cite the names, ~3 lines); dense `denseLo/Hi_uniform` (`IntervalBound` can't see `ite_sign_*` without a heavy import edge, ~1 line each). Gated incl. the local comparator (3 tiers) |
| RenderKit commit | §5 part 1, −131 lines. New `Codegen/RenderKit.lean`: `PGrad` (from `ResNet34RenderB`; MNv2's `PGradM` / MNv4's `PGradV4` gone), `adamOne` (was `adamOneM` / `adamOne4` / `enetAdamOne`), `rmsOne` (was `rmsOneM` ≡ `enetRmsOne`), `adamOneEma` (was `vitAdamOne` ≡ `convnextAdamOne`). R34's multi-optimizer `optOne` stays. ⭐ 198 of 248 artifacts re-written by the build, `git diff verified_mlir/` empty; `regen_verified_mlir.sh check` all OK. |
| packedTrainSig commit | §5 part 2, −51 lines. RenderKit's `packedTrainSig` (the `[θ|m|v|G|E]` argument regions + `%lr/%bc1/%bc2` [+ `%aup/%akeep`] [+ `%emad/%oemad`]) and `packedTrainRetTys` (results in the same order) replace the hand-built lists in R34, R50, MNv2, MNv4, ENet, ViT, ConvNeXt — 7 signatures and 10 result-type lists (incl. the inner `retTys` of ViT / ConvNeXt / ENet). R50's `emaSuf := "ema"` (the `%sge` collision) is a parameter. CnnRender's hand-listed small nets untouched. 198 artifacts re-written, byte-identical; regen check OK. |
| dropMaskSig commit | §5 part 3, net +1 line — one spelling, not a line win. RenderKit's `dropMaskSig B sd idxs` (`, %dp<i>: tensor<Bxf32>` per site, `""` when off) is the body of `vitDropSig` / `cnxDropSig` / `enetDropSig` / `r50DropSig`, each now a one-liner over its own site list; the per-net names stay (8 call sites + prose cite them). 198 artifacts byte-identical; regen check OK. |
| bf16 smart-ctor commit | §5 part 4 — 24 precision-switched constructors `XAt bf16 rnd …` in RenderKit (generated from `StableHLO`'s own signatures; each `XBf16` is `X` with `rnd` first), and all 248 `if bf16 then .XBf16 … else .X …` switches in 9 renderers rewritten to them by a checker that only rewrites when the else-branch is exactly the then-branch minus `Bf16` and the rounding argument (recursing through `.batchOp`). −16.3 KB of renderer source; lines ≈ flat (+34: most switches were single long lines, RenderKit +190). ⚠ The byte gate caught a rewriter bug: CnnRender spells the ctors qualified (`SHlo.convBackBatchedBf16`) and the first pass dropped their f32 branch — 8 cifar8b artifacts changed; fixed and re-verified. 238 artifacts re-written, byte-identical; regen check OK. |
| fwdSigParts commit | §5 part 5, −321 lines. `MlirCodegen.emitForwardEvalSig` was a 335-line copy of `emitForwardSig` (identical but for comments) plus the per-BN `%bn_mean/%bn_var` suffix; both are now thin wrappers over `fwdSigParts : NetSpec → Nat → String × List Nat`. Gate (reference codegen has no `verified_mlir/` artifact): a per-module harness renders `generate` / `generateEval` / `generateForwardCam` for every `NetSpec` constant in the 45 Bestiary modules + `VjpOracleNets` — 216 specs, 51 MB — before and after: byte-identical. `regen_jax_generated.sh check` OK. |
| paramSlots commit | §5 part 6 (NetSpec layout, first half), −431 lines. `Spec.Layer.paramSlots : Layer → Option (List ParamSlot)` — shape + init role (`he fanIn` / `normal σ` / `const v` / `zeroSeeded`) per tensor, for the 22 trainable layer kinds — is now the one writer that `NetSpec.paramShapes`, `heInitLayer` (a fold: random draws consume a seed, constants don't) and `Layer.nParams` read; the old `nParams` keeps only the untrained Bestiary arms as `nParamsUntrained` (69 of 216 specs' displayed totals come from those, e.g. DenseNet / Swin / Mamba — unchanged). Gate: harness over all 216 NetSpecs — `paramShapes`, `totalParams`, `heInitParams` size + hash (164 specs; the F32 extern loaded via a scratch `.so` of `F32Array.c` + `ffi/f32_helpers.c`; 52 too large to allocate) and `generateTrainStep` — byte-identical before/after. ⚠ Still hand-spelled: the MLIR signature emitters (`fwdSigParts`, `emitTrainStepSig`, `emitForwardCamSig`) — they carry SSA names (`%W{pidx}`/`%b`/`%g`/`%bt` per group) and thread the activation shape, so they need a naming role + group index on the slot, and `MlirCodegen` can only reach it through `Spec` (SpecHelpers imports MlirCodegen) |
| signature-emitters commit | §5 part 6 (NetSpec layout, second half), −1,092 lines. `ParamSlot` gains its signature name (`W` / `b` / `g` / `bt`; a `W` opens a group sharing one index). `emitTrainStepSig`'s three ~300-line walks (θ, `m_`, `v_`, plus the lockstep return-type lists and a dead `curShape` walk) are one `trainSigArgs pfx` over the slots; `fwdSigParts` is a per-layer `fwdLayerArgs` plus a shape-only walk. Kept by hand: `.fpnDetect`'s line layout, `.tokenPositionEmbed`'s one-line pair, and `.layerNorm` / `.convNextStem` emitting nothing (the reference codegen has no lowering for them). ⚠ Found and preserved verbatim: the FORWARD signature drops a block layer's params when the activation isn't rank 4 (`muZeroGoPredictionPolicy`: a residual stack on a flat input) while the TRAIN signature keeps them — the two signatures disagree for such specs. Gate: the harness widened to all 77 NetSpec modules (Bestiary, VjpOracleNets, apps/, demos/, tests/) — 281 specs, 190 MB of `generate` / `generateEval` / `generateForwardCam` / `generateTrainStep` + shapes / totals / init hashes — byte-identical. Not done: `emitTrainStepBody`'s optimizer-update walk, which is interleaved with the update code |
| `544e61f9` | §5 part 8, −384 lines. `emitTrainStepBody`'s optimizer updates walk `Layer.paramSlots` (the signature's order) instead of a 13-family dispatch over forward records; `ParamSlot` gains `decay` (matrices/kernels yes; vectors and the two positional embeddings no); update temporaries are `{nm}{p}`. `emitConvBnAdam/Momentum` + 23 never-read `FwdRec` fields deleted. Gate: 281 specs × 6 optimizer variants equal up to a bijective SSA renaming (`alpha.py`), except outputs already invalid: 35 transformer specs' final-LN update was `tensor<0xf32>` (now `[d]`), 5 specs (VAE decoders, MuZero heads) returned fewer tensors than declared |
| `7e97edae` | §5 decisions (user). The clip norm sums EVERY trained tensor's gradient (was conv2d/dense/fpnDetect/conv-BN only; 78 specs' clip variants move, the three YOLO demos that clip are byte-identical). `inputChannels` recognises every conv block as a first layer, `inputFlatDim` derives from it: MuZero's prediction heads take 256×19×19 (were flat 361) and lower with no undefined names. ⚠ The audit's "residual stack on a flat input" was a MISREADING — the input was never flat, the first-layer table just missed blocks. The forward signature's rank-4 guard is now a `panic!` (fires on no spec) |
| `d9b15213` | §5 reference NetSpecs: `LeanMlir/ReferenceNets` (Reference lib) holds the Imagenette ResNet-34 and ConvNeXt-T-GELU; 8 copies (ablation, baselines ×2, GradCAM ×2, InspectConvNeXt, TestResnetResidual) gone, all byte-identical incl. names; the ConvNeXt baseline keeps its `ConvNeXt-T` prefix via `{ … with name := }`. UNet-Pets left: its other copy is the Bestiary gallery entry (own `main`, not importable) |
| `3f72358a` | §5 `TestR34SyncBnCheck` on `SyncBnCheck.run` (308 → 42 lines); 1×64 rendered at run time = the committed `adam64` bytes. Ran on 2 GPUs: ✓ CONFIRMED (stats 1.8e-4 vs CONTROL 3.4e-3, first layer 0). Dropped the never-gated REORDER columns |
| `d9fed7a2` | v1 §10: 79 `(s.splitOn p).length > 1 / == 1` → `s.contains p` / `!…` (incl. the variant predicates + their #guards; `== 2` exactly-once guards kept); `containsSubstr` ×2 / `hasSubstr` gone; `leanFiles` → `walkDir`; suffix tests → `List.isSuffixOf` |
| `6323a691` | v1 §10: new import-free `LeanMlir/LEBytes` (`pushU32LE`, `pushF32LE`); 7 int32 + 5 f32 LE writer copies and 6 `mkLabels` (→ `VerifiedTrain.mkLabels`, true int32) folded; old = new on 3,009 ints / 2,011 floats / every caller's labels. ⚠ `ParamLayouts` (Certs cone) now imports `LEBytes` — lakefile comment says four data modules |
| `bce4cb1a` | v1 §10: the four hand-rolled `iree-compile` smokes → `compileCheck` / `tryCompile` (the two representative train steps now land in `.lake/build/`, not `/tmp`); `findIreeCompile` (`.venv` first, then PATH) → `Types`, used by Train + 3 copies. ⚠ `compileCheck` / `tryCompile` / `compileVmfb` still use the PATH compiler. All four smokes compile under iree-compile 3.12 (llvm-cpu; binary at `../lean4-jax/.venv/bin`) |
| `91737514` | §5 `emitTok`, −411 lines: `emitContract` renders a convolution / dot at f32, bf16 or fp8 (operand converts, low-typed op, convert back — or f32 result for the accumulator dot); `lowOf tag`; 24 `.batched` families match `"x" \| "xBf16" \| "xF8"`, 27 twin cases gone; `emitFlatConv` / `dotInOp` / `emitMatmul` for the top-level pairs. Gate: old `emitTok` copied into a scratch module vs new on every tag × B∈{1,3} × 2 stacks = 560 cases, 0 mismatches (corrupted control caught); regen proofs → `verified_mlir/` unchanged; emit ties; Certs/CertsHeavy; comparator 3 tiers; AuditAxioms 1,573 |
| `db38ec97` | §5 `emitTrainStepBody` 3,094 → 2,456 lines: `emitTrainConstants`, `emitTrainLoss` (all five loss branches + seed → `(text, gradSSA, gradShape)`), `emitOptimizerUpdates` split out. Gate: main harness byte-identical + 281 specs × 15 loss paths hash-identical |

Each gated: `lake build Certs LeanMlir Apps`, AuditAxioms 1,597/1,597, `docstring-checkrefs`,
`verified_mlir/` byte-clean; `a745b694` also `CertsHeavy` + AuditAxiomsHeavy 62/62 and the three
tie test exes; `d3688f85` also `check_render_coverage.py`.

**Auditor claims that were wrong** (checked before editing): `transformerTower_has_vjp_mat` and
`vit_full_has_vjp` exist (§3.3's "names that don't exist" list is otherwise right);
`VerifiedConfig.lr` IS display-only (the Adam path takes `trainAdamSched`'s `baseLR`) — only the
wording was off; `ViTFold`'s import (above).

**Still open:** see "▶ Next session" below — it is the current list.

## ▶ Next session — start here

State (2026-09-24, night): `proof-cleanup` is 10 commits ahead of `origin/main` (`3e654e95`), NOT
pushed — `544e61f9` … `db38ec97` + this planning commit. Done: §0–§4, §6, §8, and §5 (rows above).
⭐ No Foundation file imports a net or a certificate; `StableHLO` imports no net; `Certs` reaches
four pure-data program modules; the renderers share `Codegen/RenderKit`; the reference codegen's
parameter layout — shapes, init, counts, both signatures AND the optimizer updates — is one table,
`Spec.Layer.paramSlots`; `emitTok` renders every precision of a contraction through
`emitContract`. AuditAxioms: 1,573.

**The gate, every commit** (the user approves each commit; commit ≠ push):
`lake build Certs LeanMlir Apps CertsHeavy Reference`; `tests/AuditAxioms.lean` elaborates with
every `#print axioms` giving a 3-axiom verdict (1,573 today — the certs.yml recipe, which JOINS
wrapped lines: a plain grep undercounts) and `AuditAxiomsHeavy.lean` 62/62; `lake exe
docstring-checkrefs`; `python3 scripts/gen_comparator_tier.py --check`;
`scripts/check_audit_coverage.py` + `scripts/check_render_coverage.py`;
`scripts/regen_verified_mlir.sh check` when a renderer or writer moves; regenerate
`blueprint/lean_decls` from content.tex's `\lean{}` names, then `lake exe blueprint-checkdecls
blueprint/lean_decls blueprint/lean_deps` + `scripts/blueprint_uses.py --check` (`--fix`, then
re-run `scripts/blueprint_depgraph_tikz.py`, when an edge moves); every `formalization.yaml`
`declaration`/`file` pair still matches; `git status verified_mlir/` clean; and
`tests/comparator/run.sh` (~4 min, three tiers) whenever a comparator-cited name, a root file or a
tie moves. A book change gets the current-vs-proposed preview on :8765 (tailscale 100.76.1.97)
before the commit.

**Gates for program code (2026-09-24, scratch — `/tmp` is not durable; rebuild from here):**
- *Reference codegen harness*: per module of the 77 that define a `NetSpec` (Bestiary, `VjpOracleNets`,
  apps/, demos/, tests/), a generated `#eval!` file dumping, per constant, `paramShapes`,
  `totalParams`, `heInitParams` size+hash (F32 extern from a `.so` of
  `.lake/build/ir/LeanMlir/F32Array.c` + `ffi/f32_helpers.c`, `--load-dynlib`), `generateTrainStep`
  under Adam / momentum / Muon / Shampoo / clip+headLR / momentum+clip+wd0, and
  `generate` / `generateEval` / `generateForwardCam` — ~0.9 GB, 6-way parallel, a few minutes. A
  second pass hashes `generateTrainStep` under 15 loss paths. `alpha.py` compares two dumps
  section by section: identical / equal up to a bijective SSA renaming / different.
- *Printer harness*: copy the old `emitTok` (+ the private `sWGradGeom`) verbatim into a scratch
  module importing the new `StableHLO`, render every `.batched` / `.batched2` tag (arity read off
  the patterns) and the top-level pairs with both, compare `run` results. One `StableHLO` build +
  ~25 s, instead of the ~12-min writer rebuild; the full regen is then the final check only.

**Traps met (all rounds):**
- `git grep` misses untracked new files — use `grep -r` for renames.
- Moving a decl out of a file whose namespace it inherited RENAMES it; AuditAxioms catches pinned
  ones, checkdecls the blueprint ones.
- ⚠ Cutting an import (or moving a file) silently removes TRANSITIVE names from every importer:
  budget one fix-up build per move.
- Generated files change only via their generator. Before a move that touches a generated file's
  imports, RUN its generator on the untouched tree and confirm it reproduces the committed bytes.
- Dropping blueprint nodes can flip Figure C.1's double borders (the `EVERY` rule) and delete a
  caption macro (`\depgraphCites*`) — build the PDF.
- `fun_prop` at the nets' literal widths: tag the ATOMS, unfold down to them, never tag a block
  lemma; interleaved defs need `repeat (first | unfold …)`.
- `scripts/blueprint_depgraph_tikz.py` treats `argv[1]` as the OUTDIR — `--help` wrote a stray
  `--help/` directory (untracked, in the repo root; delete it).
- `rw` does not match `h ▸ e` against `castIdx h e` — use `refine (lemma …).trans ?_`.
- A `HasVJP` whose backward is spelled the way the IR op denotes cannot be swapped for an
  equal-but-differently-spelled witness (the reindex row). Dedupe `correct`, keep the spelling.
- Scripted renderer rewrites: constructors appear both as `.XBf16` and qualified `SHlo.XBf16`.
- A signature emitter's shape guard can hide params — and can hide a WRONG FIRST-LAYER TABLE
  (the `7e97edae` row).
- An anonymous constructor `⟨…⟩` does not fill defaulted structure fields — adding a field to
  `ParamSlot` means spelling it at every site.
- A defaulted parameter BEFORE an explicit function parameter swallows a trailing `fun` positionally
  (`emitContract`'s `lowResult`): put the function first.
- Python regexes over Lean source: a docstring-optional prefix `(/--…-/)?` with a lazy body can
  match from the top of the file (one ate 256 KB) — anchor on line indices instead.
- `lake` rebuilds on content hashes: `touch` rebuilds nothing, so it cannot time a cycle.

### 1. §5 — what is left (judgement, not dedup)
- `emitTrainStepBody`'s forward and backward walks (still 2,456 lines together): they share ~25
  mutable locals through `records` (`FwdRec`), so a split is a design change.
- `SHlo` constructors grouped by delivery increment → by family; printer → `StableHLOPretty.lean`
  (§3.2 of proof_cleanup parked it for build time).
- Emitted forward ↔ proven T2 graph linked only by prose → one `#guard` per net (lead).
- `compileCheck` / `tryCompile` / `compileVmfb` run the PATH `iree-compile`, Train / GradCAM / the
  DDPM sampler / the UNet test prefer `.venv` (`findIreeCompile`) — one rule, if wanted.
- UNet-Pets: the Bestiary gallery and `TestUnetForward` each spell it (gallery has its own `main`).

### 2. Found in §6 — open
- GramQ `rowDotQ`/`castM` ≡ IBP `sumQ`/`castV`, and the dense uniform boxes — skipped (reasons in
  the small-collapses row).

### 3. Found on the way — open, small
- `Architectures/ChannelLNBack` → `Nets/ConvNeXt/ConvNeXtChannelLN`: the last Architectures → Nets
  edge; needs ConvNeXt's channel-LN op (`chanLNTensor3`, `chanLNRows`) moved into Architectures.
- `verified_mlir/MANIFEST.md` is stale since `5a4e7778` (2026-08-28): 231 vs 248 artifacts,
  ungated; regenerating puts four `cifar8b_*` under an "(unknown)" slug — fix the generator's
  slug table, then regenerate.
- `fwdRenderedBatch` (private, `VerifiedTrain`) has no caller — dead.

### 4. Small (from round 1)
History-first module docs (`ResNet50BackB0`, `MobileNetV4BackB0`, `EfficientNetStepTie`,
`EfficientNetBackB0`); file:line citations into `jax/` and `VerifiedTrain`; the four `@[simp]` lemmas
with no named use (`win3RowInv_val`, `win3ColInv_val` in MaxPool3s2, `zk_apply`, `dzk_apply` in
BatchSealKit — need a build without them); the `.train`-arm hazard in `r34FwdChain` /
`r50FwdChain` / `mnv2FwdChain`.

---

## 0. Check first — two correctness items the audit surfaced

1. **`NetSpec.totalParams` vs `paramShapes` on SE nets** (already in memory since 2026-09-11, still
   unverified). Measured: ENet-B0 4,020,358 vs 7,155,658; V2-S 20.2M vs 38.2M; SE-off specs agree.
   `Train.lean:697` sets `nP := spec.totalParams` and slices p/m/v at `3·nP`, so `efficientnet-train`
   / `efficientnetv2-train` (reference path) read misaligned slices. Root cause is §5's six copies of
   the layer layout. Decide which SE width is canonical (Spec.lean:65 says `ic/4`; graph + `paramShapes`
   use `mid/4`) — then a short CPU probe of one train step settles it. No GPU needed to confirm.
2. **v1 §11 defect 6 regressed.** `sigmoidScalarDeriv_eq` (σ' = σ(1−σ), the fact behind the emitted
   `sigmoidBack` text, StableHLO.lean:6682) was deleted by the dead-code sweep `fb00989c` as unused and
   unpinned. `swishScalarDeriv_eq` is exposed the same way. Restore (3 lines, verified):
   `simp [sigmoidScalarDeriv, sigmoidScalar_eq_sigmoid, Real.deriv_sigmoid]`, and **pin both** in
   `tests/AuditAxioms.lean` so a sweep can't take them again. Lesson for future sweeps: a closed-form
   lemma that justifies emitted text is a consumer even with no Lean user.

---

## 1. Free wins — dead or wrong-way imports (all verified by compiling without them)

| import to drop | effect |
|---|---|
| `Certificates/LipschitzCert.lean:1` → `Foundation.Tensor` (replace with `Mathlib.Tactic.Positivity.Finset`) | ⭐ 33 certificate modules (CROWN/IBP/SDP/smoothing, ~1,000 s serial) stop rebuilding on every `Tensor.lean` edit |
| `import LeanMlir.ViTRender` in 7 `Proofs/Codegen/*Render*` (+ `LeanMlir.Types` in ConvNeXtRender) | all 121 artifacts byte-identical without them; "proofs never import the program side" becomes true for `Proofs` |
| `Training/SgdDescentCnn.lean:2` → `Nets.MobileNet.MobileNetV2Close` (needs `Nets.Small.CnnTrainStep` + `Foundation.StridedConv` instead) | cuts the Float tier's cone (see §2.1) |
| `Nets.ResNet.ResNet34` from `Foundation/BackwardMaps`, `Training/BatchSealKit`, `Foundation/SpecVJP` | move `relu_nonneg` into BatchSealKit first; ResNet34.lean then has only AuditAxioms as importer. BackwardMaps is a lead: 48 modules reach ResNet34 only through it |
| MNv2 legacy chain: `MobileNetV2FullBVJP`→`FullVJP`, `FullVJP`→`BackCertifiedTie`, `WholeBackCertifiedTieB`→`WholeBackCertifiedTie`; `MobileNetV2FoldPaperG`'s four fold imports (needs only `EfficientNetFoldG`) | move `IVPos`/`IVNoExpPos` beside `IVW` first; the retired per-example files detach into a leaf |
| `EfficientNetFullWholeBackCertifiedTie` → `ConvNeXtBackCertifiedTie`; `ViTFold` → `Small/Cifar8Fold` (⚠ NOT free as reported: it uses `Cifar8PoC.denseW/B_den` — import `Small/MlpTrainStep`, their home, instead); `ResNet34Fold` → `Cifar8Fold` (lead) | sideways family edges |
| `LipschitzCertPairSDP.lean:2–3` (`Matrix.PosDef`, `Order.Star.Real`, leftovers from a retired lemma) | tidiness |
| `Architectures/Attention.lean:5` → `SE` (lead) | — |

Gate: `lake build Certs LeanMlir` + the regen drift check. One commit.

---

## 2. Misplaced owners — the structural theme (moves, names kept)

Every row keeps full declaration names (namespaces stay), so AuditAxioms, the book, the yaml and the
comparator tier are untouched unless noted. "Fan-out" = modules rebuilt by the move.

### 2.1 The float / conv vocabulary out of `Training/SgdDescentCnn` — two auditors, independently
`FloatClose` (the float tier's one closeness form) sits on a **39-module / ~37k-line** import cone
through `SgdDescentCnn` (6.8k lines) and `MobileNetV2Close`. It and 8 core lemmas compile on
`FloatBridge` alone (verified). Cause: the conv index vocabulary (`t3Idx` 121 uses, `k4Idx`, `w3Idx`),
the padded conv read, the float conv forward (`convF`, `flatConvF_close`, …) and max-pool facts all
live in the descent file — so `Float/ResNet34FloatBridge` imports Training, and the padded read is
defined **five times** (`convPad`, `IBP.convTap`, `convWindow3`, `convWindow`, `dwWindow`; two proved
`rfl`-equal). `convTap` also means two different things in two namespaces.
- `Float/FloatClose.lean` (imports FloatBridge only): `FloatClose` + `.comp/.of_close/…`; `add_close`
  into FloatBridge beside `mul_close`.
- `Architectures/ConvIndex.lean`: index defs + sum lemmas, `convPad`/`convWindow`/kernel tap, conv
  Jacobian lemmas, max-pool window facts; `IBP.convTap := convPad`, `convWindow3` in terms of it.
- `Float/ConvFloat.lean`: SgdDescentCnn :353–866.
Payoff: a padding change is one edit, not five; Float stops depending on Training; closes v1's
deferred "hoist the window block".

### 2.2 Batched math out of the IR file
`batchMap`/`batchMapAux`/`batchSlice`/`bnBatchLA` (2,229 uses; 23 files use them and never touch
`den`/`SHlo`) live in `Codegen/StableHLO.lean`; `Foundation/BatchMapVJPAt` and
`Training/BatchSealKit` import the code generator only for them. → `Foundation/Batched.lean`
(imports PerChannelBN; compiles standalone, verified). 3 import edits.

### 2.3 Primitive ops out of net files → StableHLO/IR stop importing nets
StableHLO imports CifarCNN, MobileNetV2, EfficientNet, ConvNeXt — for `relu6` (22 uses), `sigmoid` /
`broadcastFlat` / `seGate` / `seBlockFull` (34), `layerScale` (15). IR imports EfficientNet for
`sigmoid`. An edit to MobileNetV2.lean rebuilds ~152 modules incl. every ViT/ConvNeXt file.
- `relu6` (+ linearisation/VJP) → next to `relu` in MLP.lean
- `sigmoid`, `broadcastFlat`, SE gate → `Architectures/SE.lean`
- gelu/swish/tanh/sigmoid → one `Architectures/Activations.lean` (today: four homes + a second swish
  derivative in BatchSealKit, §6)
- `layerNormVec`, `layerScale` → LayerNorm.lean (judgement — ConvNeXt co-owns)
- softmax Jacobian + `softmaxCE_grad` (defined in MLP.lean, differentiated inside the 2.9k-line
  `Attention.lean`; IR imports Attention only for these) → `Architectures/Softmax.lean`
- the cifar chapter-graph `_faithful` theorems → a leaf (proof_cleanup §3.2's unfinished item)
Root batch: one ~210–330-module rebuild. Comparator ChallengeArch/SolutionArch need the new import.

### 2.4 Gradient-node lemmas get a home (model: `Foundation/Bf16GradNodes.lean`)
Of 140 `den (SHlo.<op> …)` node lemmas, 129 live in `Nets/`; 26 op kinds are proved in more than one
file. Namespaces carry the first user's name (`ResNet34PoCB.convWGradB_den` serves 6 nets); two
Foundation modules import net files for them. **11 forwarding copies** (ConvNeXtFoldGB ×10 incl.
2 statement-identical pairs under different names; EfficientNetFoldG ×4) + one duplicated body.
→ `GradNodesB.lean` beside Bf16GradNodes holding the ~25 generic f32 lemmas. Keeping full names moves
0 pins; deleting the forwarders retires **10 + 4 audit-only pins** (owner call, §8).

### 2.5 Shared tie / stage / sync kits out of ResNet-34 and EfficientNet files
| today | what's generic | proposed |
|---|---|---|
| `ResNet34FoldB` (ns `ResNet34PoCB`) | whole file: 8 `*GradB_den`, `BnPairTiedB`, 10 clause Props (3 ResNet never uses); ~340 refs, 17 files, 5 families | merge into §2.4 kit (compiles on StableHLO + CnnTrainStep + CifarBnClose) |
| `EfficientNetStepTie` (fused-SGD ENet tie, "PoC… DONE") | `reassocB` (324 refs / 13 files), `cInB` (90/11), `bnBackB`, `dInB`, `gapInB`, … — ResNet-34's T3 imports the 712-line ENet tie for them | `Foundation/BatchedBackLinks.lean`, with `ResNet34StepTieB`'s `reluMaskB`/`cStridedInB`/`bnInB`/`rowB`/`unrowB` (the other half of the same vocabulary) |
| `EfficientNetChainClose` | `batchMap_has_vjp` (62 refs), `bnBatchLA_has_vjp`, `reindex_has_vjp`, `flatConv_has_vjp` | `Foundation/BatchMapVJPAt` / CNN.lean |
| `EfficientNetBackB0` ("Spike" header) | `bnBatchLABack_faithful` (7 files), `residualBackGraph`, per-op batched `_faithful` | same BatchedBackLinks leaf |
| `ResNet34BackCertifiedTie` | all 5 leaf ties (conv/strided/XLA/dense/GAP); its depthwise twin is already in Architectures | `git mv` → `Architectures/ConvBackCertifiedTie.lean` |
| `ResNet34SyncB` + `ResNet34SyncStepTieB` §1/2/4/5/7 + `MBConvSyncTieB` (in `Nets/EfficientNet/`) | `castIdx`/`laAssoc`, IsHomog/shard lemmas, P4 collectives (the 7-line proof appears 8×) | beside `Foundation/DataParallelSync`; one generic Σ-collective lemma |
| `ResNet34FullBVJP:190–230` | `CertLayer.comp_ok_of` (37 uses), `r34PoolLayer` (net-agnostic pool) | CertifiedChain / HeadLayers — ⚠ certlayer_nets §4.2 step 1 |
| `MobileNetV2BackB0` `projLayer`, `ResNet34BackB0` `cbReluLayer(Strided)` | stage CertLayers used by 4 nets; import chain ENet→MNv2→R34→R50→BackNetFolds→MNv4 | grow HeadLayers into StageLayers — ⚠ certlayer_nets |
| `ResNet34FullBSeal` | `projB_zero_const`, `sealProj*`, `*_continuous` used by R50/MNv2/MNv4 seals | BatchSealKit |
| `EfficientNetRenderPC` (a Codegen "render" file that renders nothing) | batched stages `cbsB`/`dwbsB`/`projB`/`stemB`/`seB` consumed by R34/R50/MNv2/MNv4 | `Foundation/BatchedStages` |
| small nets | per-example fold kit over 4 files / 3 namespaces; `Cifar8PoC` is declared inside MlpTrainStep.lean:285 | one small-net kit file |

The `c·h·w ↔ c·(h·w)` seam has **five spellings** (`castIdx`+`laAssoc`, `den_cast` ≡ `den_castIdx`,
private `reassoc`/`reassocB` in two renderers, `EnetTiePoC.reassocB` on Vec, `reassocFwd/Back`); six
sync files each re-explain it. One leaf module; delete `den_cast`.

### 2.6 Other misplacements
- **Foundation isn't a layer**: 13 of 29 files import Nets/Certificates/Codegen. Most edges disappear
  with §2.3–2.5. Remaining moves: `BackNetFolds.lean` is one audit-only def + an import barrel → move
  `cnxBlockChLayer` to ConvNeXtBackB0, delete the file; `EvenKernelConvBack` → Nets/ConvNeXt;
  `SpecVJP` (most-downstream module in the tree, 12 net imports, no proof importer) → out of Foundation;
  `crossEntropy_differentiable` out of `Small/LinearTrainStep`; `denseE`/`reluE` out of the trained-
  weights file (so `Foundation/IntervalBound` stops importing `LipschitzCertInstance`);
  `PerChannelBN`/`StridedConv` are ops → Architectures.
- **`Certs` builds the trainer and FFI**: SpecVJP → VerifiedNets → VerifiedSpec → VerifiedTrain (4.9k)
  → IreeRuntime. Make VerifiedSpec import-free (DSL + `VerifiedData` + XLayout tables), move its 10 IO
  forwarders into VerifiedTrain, `VerifiedNetsCore` for SpecVJP.
- **Certificates engine**: `CertifiedAt` is defined in a *generated* file; the dense engine is
  interleaved with trained data in `LipschitzCertInstance`; `mlp_gap_eq` sits in the SDP file; the
  smoothing chain imports one net's weights for one continuity lemma → `Certificates/DenseEuclid.lean`
  (one generator edit to stop emitting `CertifiedAt`). Φ/Φ⁻¹ API scattered over 3 files + an
  `IsOpenPosMeasure` instance built downstream of the file that needs it → `GaussianQuantile.lean`.
- **`Proofs/Codegen/` non-codegen**: `AdamStep`, `SgdMomentumStep`, `RmsPropStep`, `GradClip`, `Lamb`,
  `DropPath` (ℝ specs, ~0 `SHlo`) → `Training/Optim/`; `MatBridge` (0 importers) → Foundation;
  `MobileNetV2RenderPC(Eval)` → Nets; `UibSpec`/`mnv4Blocks` → `MobileNetV4Spec` (a proof file
  imports the 9-artifact renderer for them); CnnRender/MlpRender artifact writers → leaf modules.
- **Two jobs per file**: `SgdDescentCnn` (real + ~2.7k float; header documents one; 19 "Increment
  N / Item A" headings; `Conv2Slot` opened twice) → split `SgdDescentCnnFloat`. `PerChannelBN` +513
  lines of sync-BN sharding → DataParallelSync (stops a 222-module rebuild). `Tensor.lean` ~230 lines
  of SDPA calculus → Attention. `BatchSeal` namespace holds 14 BN/continuity/pool layer facts → their
  op files. `ViTVecLN` five jobs. `VerifiedTrain` four jobs (driver, PGD/spectral, three hand-typed
  StableHLO generators, smoothing) → split on its banners.

---

## 3. Documentation — cheapest high-payoff tier (prose only)

### 3.1 One legend (README), replacing `Proofs/README.md:73–76`
The README's stage list (BackB0 → ChainClose → Render → Close → Fold/StepTie → Seal) is the 2026-06
chain. **There is now one canonical conv-net chain**, followed by R34/R50/MNv2/MNv4:

`BackB0 → FullB → FullBVJP (…PosB/…SmoothAtB) → FullBSeal → StepTieB → BackChains + WholeBackCertifiedTieB → SyncB / SyncStepTieB`

| tier | R34 | R50 | MNv2 | MNv4 | ENet-B0 | ConvNeXt | ViT |
|---|---|---|---|---|---|---|---|
| block back graphs | BackB0 | BackB0 | BackB0 | BackB0 | BackB0 + BackNet | BackB0 (per-example) | BackB0 + BackNet |
| T1 fwd + T2 graph | FullB | FullB | FullB | FullB | FullB0 | FullT | DepthK (+VecLN, MultiHead, FwdGraph) |
| T1 VJP | FullBVJP | FullBVJP | FullBVJP | FullBVJP | in FullB0 | in FullT | in DepthK |
| seal | FullBSeal | FullBSeal | FullBSeal | FullBSeal | — | — | — |
| T3 un-fused batched fold | FoldB | (R34's) | FoldPaperG | (others') | FoldG | FoldGB | FoldGB |
| T3 tie | StepTieB | StepTieB | StepTieB | StepTieB | StepTieG | StepTieGB | StepTieGB |
| T6 whole-net back tie | **BackCertifiedTieB** | WholeBackCertifiedTieB | WholeBackCertifiedTieB | WholeBackCertifiedTieB | **FullWholeBackCertifiedTie** | WholeBackCertifiedTieB | WholeBackCertifiedTieB |
| data-parallel | SyncB / SyncStepTieB | same | same | same | SyncB / **SyncStepTieG** | — | — |

Suffixes as they are used today: `B` batched index (+ batch BN); `G` tied at the un-fused `*GradB`
node; `GB` both — but `B`, `G`, `GB`, `PaperG` all name the *same* tier (each records which axis
differed from that net's historical baseline). `B0` means three things: "block backward" in `*BackB0`
(fossil of the 2026-06-14 ENet-B0 spike, `21b71428`), the model in `EfficientNetFullB0`, "prefix 0"
in `mnv2PreB0`. `PC` per-channel per-example BN; `Eval` frozen stats; `Paper` MNv2's [t,c,n,s] table;
`Full` paper depth; `T` ConvNeXt-T; `V`/`MH`/`K` vector-LN / multi-head / depth-k; `Xla` SAME padding.
Namespaces ≠ file names (`ResNet34FoldB`↔`ResNet34PoCB`, `StepTieB`↔`ResNet34TieB`,
`ConvNeXtStepTieGB`↔`CnxTiePoCGB`, …) — add a column. Declaration suffixes: `_faithful` (239),
`_den` (105), `_bridge` (33), `_eq_vjp` (43) all mean "this denotes that" at different granularity;
`_certified`, `_tied{,B,G,GB}`, `…Tied*` clause Props. **T1/T2/T3/T6** are used 130× in 28 files and
the yaml but defined only in `planning/archive/` — add the glossary. Also the BN-ε positivity bundles'
four spellings. Renames are **not** proposed (≈350 AuditAxioms lines, ≈225 comparator refs); two
optional cheap ones: `ResNet34BackCertifiedTieB` → `…WholeBackCertifiedTieB` (14 mentions),
`mobilenetv2PaperPC_has_vjp_at` → `mnv2B_full_has_vjp_at` (6 refs, 1 audit line).

### 3.2 Missing maps
- `Codegen/README.md`: file roles; one row per net — renderer → chain → artifacts → T2 graph → T3 tie
  → T6 tie (MobileNetV2RenderB's "Proofs tier" paragraph is the model); the `SHlo` suffix legend
  (`F` 48 = forward *and* optimizer ops; `B` 64 vs `Batched` 13 = two spellings of batched;
  `Faithful`/`FaithfulV`/`FaithfulB` don't track which chain renders; `RenderPC` means per-channel BN
  for MNv2, batch BN for ENet).
- `Certificates/README.md`: the family table (ℝ theorem ← checker ← generated data ← generator, ten
  rows, drafted in the certificates report), the one-paragraph pattern, the naming legend
  (`S`/`T`, `C`/`U`, `SC`/`SU`, `Uncon`, `F`, `e8`, `ImgsA–D`; `LipschitzCertDemo` also covers IBP/CROWN),
  and the exceptions (Instance's hand-merged data; SDPFull built by no lib).
- Non-proof world: a 15-line map of the two pipelines (reference `Types → Spec → SpecHelpers →
  MlirCodegen → Train`, unverified; verified `VerifiedSpec → VerifiedNets → VerifiedTrain` loading
  `verified_mlir/`). `lean_lib «Codegen»` builds only the reference path and nothing named Codegen in
  Proofs; no CI uses it → rename `Reference` or retire.
- Layering: the "by content, not import order" rule lives only in `planning/cleanup_backlog.md` §10;
  the README's "Dependency graph" predates Codegen/ and Nets/. One paragraph + a generated graph.
- Float tier map (FloatModel → per-op `*_close` → FloatClose/`.comp` → mixed precision → `rndP` →
  `FaithfulFloatModel`) exists only in `formalization.yaml:267–275` → FloatBridge's header.

### 3.3 Wrong statements in entry points (≈50, each a one-line fix; full lists in the slice notes)
Worst first:
- `lakefile.lean:27` "`lake build LeanMlir` type-checks the whole repo" — 91 of 259 modules.
  `:34` "never the codegen" — false for Proofs and Certs. README vs lakefile vs measured disagree on
  every count (Certs roots 201/195/193, modules 235/229/224+8, MlirCodegen 7.5k vs 10.4k).
- `LeanMlir.lean:127`, `Proofs/README.md:326` cite `efficientnet_net_tied` as ENet's headline; the
  book and yaml cite `efficientnet_net_tiedG` (the one covering the quoted accuracy).
- `LinearTrainStep.lean:19–32` — the README's **first** "Start here" file says `crossEntropy`
  differentiability "doesn't exist yet"; the same file proves it.
- `MaxPool3s2.lean:53` "nothing downstream references these" (18 files do).
- "All-reduce is emitted text outside the AST" in DataParallel.lean:7 and 4 Fold/Tie headers —
  it's an AST node since 2026-09-07. Two different things are called "piece 3".
- `ViTBackNet:47–52/179–185` "no conv net folds to logits / R50's stem is blocked" — false since
  `6778f8c9`; MobileNetV4FullB(VJP) "cannot be one CertLayer" (⚠ certlayer_nets).
- 14 `MlirCodegen` line-number citations in Architectures, all stale → cite function names. Same for
  StableHLO line cites (`StableHLOLex:46`, `ResNet34BackB0:24`, `MobileNetV2BackB0:22`, …).
- Names that don't exist: `rblk`, `emitMlpHlo`, `mnv2DownBodyB`, `transformerTower_has_vjp_mat`,
  `vit_full_has_vjp(_correct)`, `EfficientNetBackFloatBudget.lean`, `b0_full_back_chain`,
  `mobilenetv2ForwardPaper_eq_slots`, `imagenet_specs_drift_from_twins` (a memory file name).
  `SmoothingNetSemantics:17` credits the wrong lemma. `MobileNetV2RenderB:60` says `PGrad` is private.
- History-first headers: StableHLO ("Stage A, closes R4 for Chapter 1"), IR ("Phase 0a spike"),
  Tensor ("post-foundation-flip"), CertifiedChain (opens with "⛔ CORRECTION"), FloatBridge ("first
  bite … future work", both done), EfficientNetBackB0 ("Spike"), EfficientNetStepTie ("PoC… DONE /
  Remaining"), ResNet50BackB0 ("⚠⚠ WHAT §8 GOT WRONG"), MobileNetV4BackB0 ("▶ When a session lands").
  Scale: 181 planning-section refs in 30 Nets files (26 of 31 cited plan paths archived), 39 dated
  notes, 378 ⭐ / 48 ⛔ markers. Rule: module doc = table of contents; history → planning/archive.
- `VerifiedTrain` "lr baked in / display only" (it schedules lr), "VAL split IS preloaded" (streamed);
  `IreeRuntime` "FFI bindings for IREE" (PJRT default); `MlirCodegen` "Supports MLPs and CNNs" (50
  constructors); `genCifarPgdStep` called "the proven input-VJP" (hand-typed text, no tie).
- ConvMixedFloatBridge has no `/-!` (doc-gen4 shows no module doc).

---

## 4. Dead code (0 consumers; ✱ = pinned, owner call)

| where | what | lines |
|---|---|---|
| `Nets/ResNet/ResNet34.lean:140–351` | per-example strided family, `bnForward_injective`, `decimate*_injective`, … — rest exists only for audit-only `resnet34_has_vjp_at`✱ | ~190 |
| `Proofs/Codegen/MlpRender`, `CnnRender` | `mlp/cnn/cifarTrainStepStructured` + their op-template `let`s | ~325 |
| `Codegen/StableHLO.lean:4068–4150` | `mobilenetv2FwdGraph` + `_faithful` (unpinned) | ~80 |
| `Small/LinearTrainStep.lean:139–217` | render scaffold `TrainOut`/`TrainStepModule`/`renderModuleN`/… | ~80 |
| Architectures/Training | 13 decls (list in arch notes: `mhsa_proj_c_*`, `maxPool2_eq_argmax_value`, `residual*_has_vjp_at_correct`, `t3Idx_inj`, …) | ~105 |
| `Foundation/Tensor.lean` | rank-3 kit `vjp3_comp(_at)`, `pdiv3_add`, `biPath3(_has_vjp)`, `pdiv3_id`, `identity3_has_vjp` (book presents 2 as framework — content.tex:11000/11023); `pdiv_clm`, `HasVJPMat.backward_unique` | ~120 |
| Certificates IBP | flat-`Vec` residue: `BoxSound(.comp)`, `boxSound_id`, `denseLoV/HiV(_uniform)`, `unflatten_*_const`, `flatten_reluT`✱, `CertifiedAtLinfV`✱ | ~60, 3✱ |
| non-proof | `LeanMlir/MnistData.lean` (0 importers); IreeRuntime `Mlp/Cnn/CifarLayout` + FFI `mlpTrainStep`/`trainStepPacked`/`trainStepF32`; VerifiedSpec's 10 forwarders | — |
| misc | `vitBlockBack`, MNv4 seal `comp_ok`, `Ah2_continuous`, `reluMaskB_shard`, `chanOf`, `Cifar8Fold.lean` (docstring-only hub), `BackNetFolds.lean` (§2.6); `SgdDescentCnn`'s 19 single-use margin instances (14✱, ~420 lines) | — |
| hazard | `.train` arm of `r34FwdChain`/`r50FwdChain`/`mnv2FwdChain` — only called with `.eval`, and `.train` emits the per-example-BN defect; `R34Bn` is a second mode enum whose `.train` means something else → drop the parameter, merge `bnSite`≡`bnSiteP` into `bnEvalSite` | 0 artifact bytes |

`tests/Audit{Bridge,Probes,Sanity,Mutation}.lean` + `tests/AUDIT_REPORT*.md` (~10.4k lines): an
external audit's leftovers, no CI job; AuditBridge's two theorems now exist in MLP.lean — decide keep/move.

---

## 5. Program-code dedup (Part A, non-proof) — byte-identity gated

| item | evidence | size |
|---|---|---|
| `MlirCodegen.emitForwardEvalSig` (335 lines) = `emitForwardSig` + BN suffix → `fwdSigParts` | **verified**: byte-equal on all 210 elaborating specs | ~330 |
| NetSpec layer layout written 6× (`nParams`, `paramShapes`, `heInitLayer`, 3 sig emitters); `.uib` 18 arms / 3 files; `seMid` spelled 20× → `Layer.paramSlots` + `Layer.outShape` (FPN's `fpnDetectParamShapes` already does it) | copies disagree → §0.1 | ~400–600 |
| 258 `if bf16 then .XBf16 zrnd … else .X …` in 9 renderers → ~20 `@[reducible]` smart ctors | **verified** `rfl`; bytes unchanged by construction | ~250–350 |
| per-param optimizer step ×7 (`rmsOneM`≡`enetRmsOne` token-identical; `vitAdamOne`≡`convnextAdamOne`), `PGrad` record ×3 → `optOne` + `.rmsprop` arm + `emaSuffix` param, in a `Codegen/RenderKit.lean` (R34's file is today's de-facto kit: R50 calls `r34AdamVariant` 64×) | likely; regen diff | ~60 + ~120 doc |
| packed `[θ|m|v]` signature ×8 → `packedTrainSig` (the driver's positional contract, one writer) | likely | ~80 |
| `dropMaskSig` ×4, vector-LN site ×4, tensor-type strings (3 helpers, 103 inline; `tensorTy []` emits invalid `tensor<xf32>`), `fmtFixed`, `fmt` ×6 in demos, `foldl (·*·) 1` → `List.prod` ×28 | verified/likely | ~100 |
| `tests/TestR34SyncBnCheck.lean` never ported to `SyncBnCheck.run` (others are 25-line `Cfg`s) | lead | ~280 |
| reference NetSpecs: `resnet34` ×5 copies (4 byte-identical), ConvNeXt-T ×3, UNet-Pets ×3 → a `ReferenceNets` module | verified identity | — |
| `emitTok` per-arm/per-dtype text (74 `stablehlo.convolution(` blocks; `TestBatchedEmitTie` exists to keep 47 pairs in sync) | lead — `fresh` order fixes SSA numbering | several hundred |
| `emitTrainStepBody` is one 3,416-line def; printer has two encodings (98 typed vs 118 descriptor) | judgement, large | — |
| v1 §10 still open: `containsSubstr`/`hasSubstr` (`String.contains` works in 4.34), `walkDir`, `isSuffixOf`, LE codec, `mkLabels` ×6, 8 inline `iree-compile` | — | — |
| `SHlo` constructors grouped by delivery increment (conv ctors at 13 places) → regroup by family; split printer → `StableHLOPretty.lean` | judgement (§3.2 of proof_cleanup parked it for build time) | — |
| emitted forward ↔ proven T2 graph linked only by prose → one `#guard` per net (`pretty` of the T2 graph = the chain's forward code) | lead | 6 guards |

---

## 6. Proof reuse v2 (Part A, proof side) — ~1.1k lines, all verified unless marked

**v1 items verified then, never landed** (the certificate/Training files missed the landing batches):
- `LipschitzCertInstance` Frobenius / Schatten-4 / Schatten-8 → 3 shared lemmas (`denseE_lipschitzL2_of_sq`,
  `sq_le_of_gram_quad`, `quad_le_of_frob`); statements + 5 pins unchanged — **~150**
- `bnMean = 𝔼 i, x i` bridge (`Fintype.expect_eq_sum_div_card`, `Fintype.expect_equiv`,
  `Finset.expect_product`); the sync-BN commits hand-rolled it 3× since — `bnMean_shard` 18→4,
  `bnMean_pair` 13→5, Σ(x−μ)=0 2 lines — ~30
- `SmoothingGaussian` integrability → `Integrable.mul_bdd`, `integral_comp_eval` — ~45
- smoothing "q ≤ p ⇒ certified" ×3 + indicator bridge ×3 → `smoothing_certified_of_le`, `smoothProb_eq_real` — ~40
- relu6 linearisation copies relu's → `pdiv_of_hasFDerivAt_mask` (MLP) — ~30
- `CifarBnClose.sum_channel_fibre` → `Equiv.sum_comp` + `Fintype.sum_prod_type` — ~15
- `relu_apply_eq_max` still in IntervalBoundConv; move beside `relu` in MLP.lean, then `relu_close`,
  `relu_entry_lipschitz` (a third copy), `relu_continuous`, `relu_nonneg` collapse — ~15
- `euclid_norm_sq` = `EuclideanSpace.real_norm_sq_eq` (rfl), √-sandwiches → `sq_le_sq₀`, IBP
  uniform-box collapse ×6 → `ite_sign_lo/hi` (keep names; generated files cite them) — ~60

**New since v1:**
- ⭐ Seals' continuity lemmas by hand (R34 27, R50 24, MNv2 51, MNv4 38) → tag BatchSealKit atoms
  `@[fun_prop]` + 3 new atoms; one `fun_prop` covers R34's 14 blocks at literal widths in <1 s — **~250**
- `pdiv_lift_sum`: `softCE_grad` 45→17, `bceLogits_grad` −12 — ~40
- `mlp_has_vjp_at` onto `vjp_comp_diff_at` (backward `rfl`-equal) 44→16; MLP `pdiv_dense_b`,
  `dense_*_grad_correct`, `pdiv_relu` — ~60 (root batch)
- EfficientNet MBConv VJPs written twice under two names (`mbStridedFwdB`≡`mbDownBodyB`,
  `mbExpFwdB`≡`mbBodyB`, rfl) — ~50
- BatchSealKit's second swish derivative → LayerNorm's kit + `Real.sigmoid_*` — ~30
- `reindex_has_vjp` / `broadcastFlat_has_vjp` → `reindexVJP` (one `rfl` site needs a bridge) — ~30
- `sum_finProdFinEquiv₃` (3 sites + `sum_s2`), `mhSlab`≡`headSliceMat`, `stdGaussian_Ioo_pos` one-liner
  once the instance moves up, GramQ `rowDotQ`/`castM` ≡ `IBP.sumQ`/`castV` (one ℚ-check kit),
  small collapses (Depthwise, CNN GAP, `mulVec_headPadMat`, ViTVecLN `fun_prop`), 5 small near-clones — ~80
- `pdiv_finset_sum`'s unused `[DecidableEq α]` — 0 lines, pinned + comparator: owner call

**Resolved and killed**: interval boxes as `Set.Icc`/`Set.MapsTo` (every `.comp` is already one term;
`InBoxE` lives on `EuclideanSpace`, no Pi order); quantile / Gaussian CDF (absent from Mathlib);
binomial via `ProbabilityTheory.binomial` (no iid-sum law; no lines saved); operator-norm swap.
Confirmed absent from Mathlib v4.34: softmax, `Real.tanh` derivative, GELU, window max-pool,
Finset variance identities, `log b (b^k·r)`.

---

## 7. Suggested order (each a gated batch; the user approves every commit)

1. **§0** — restore + pin the sigmoid/swish closed forms; probe `totalParams` (CPU, one step).
2. **§1 dead imports** — one commit, biggest rebuild win per line.
3. **§3 docs** — legend + three READMEs + the ≈50 wrong statements. Docstring edits in leaf files
   are free; batch the root-file ones (Tensor, MLP, StableHLO, IR, BatchNorm) with step 5.
4. **§4 dead code** (unpinned first; ✱ items as one AskUserQuestion like census session 4).
5. **Root batch** — §2.3 primitives + §6 MLP/Tensor items (`relu_apply_eq_max`, `pdiv_lift_sum`,
   `mlp_has_vjp_at`, `sum_finProdFinEquiv₃`, relu6) + Tensor SDPA → Attention + root docstrings. One
   ~330-module rebuild instead of five.
6. **Leaf moves** in dependency order: §2.2 Batched → §2.1 FloatClose/ConvIndex/ConvFloat → §2.4
   GradNodesB → §2.5 BatchedBackLinks / sync kit / ConvBackCertifiedTie / seal pieces → §2.6.
   Coordinate the CertLayer rows with `certlayer_nets.md`.
7. **§6 remaining proof reuse** (certificates, seals `fun_prop`, ENet twins) — can interleave with 6.
8. **§5 program code** — gate on `regen_verified_mlir.sh check` + the 210-spec harness; RenderKit
   first, then smart ctors, then the NetSpec layout (after §0.1's decision).

Rough size: dead code ~1.0k + proof reuse ~1.1k + program dedup ~1.5k lines out; the clarity gain is
mostly from §2 + §3, which delete little but move ~40 files' worth of vocabulary to where a reader
would look.

## 8. Owner decisions (pinned / statement-changing)
- delete the 10 + 4 forwarding grad-node aliases (14 audit-only pins) — §2.4
- `SgdDescentCnn`'s 19 single-use margin instances (14 pins, ~420 lines) — §4
- `resnet34_has_vjp_at` (1 pin; keeps ~160 lines of ResNet34.lean alive) — §4
- IBP residue pins `flatten_reluT`, `CertifiedAtLinfV` + 1 — §4
- rank-3 kit retire vs document (2 blueprint nodes) — §4
- `pdiv_finset_sum` binder (comparator restates it) — §6
- MNv2 per-channel legacy files: keep as detached leaf or retire — §1
- `tests/Audit*` leftovers — §4
- `lean_lib «Codegen»` rename/retire — §3.2
- optional renames: `ResNet34BackCertifiedTieB`, `mobilenetv2PaperPC_has_vjp_at`, `*BackB0`→`*BackBlock` (127 mentions)
- which SE width is canonical — §0.1

## 9. Evidence
Scratch (session `930c08a6`, `/tmp/claude-1000/…/scratchpad/`): `foundation_float/t1–t6.lean`,
`arch_training/A1–A6.lean` + `rdeps.py`, `certificates/a1–a3.lean` + import probes,
`codegen/t1–t4.lean` + `Batched.lean`, `nets_resnet_small_convnext/*.lean`,
`nets_mobilenet_enet_vit/a1–a7.lean` + `*_noimp.lean`, `crosscut/g.json` (import graph),
`decls.json` (9,687 decls), `tc/*.lean` (incl. the 210-spec `emitForwardEvalSig` harness, `Tp2.lean`
for §0.1). `/tmp` is not durable — re-derive from this doc if the scratch is gone.
Rubrics (re-runnable) and the per-slice notes: `planning/audit_v2_rubrics/`.
