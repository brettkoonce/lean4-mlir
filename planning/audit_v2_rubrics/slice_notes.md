## certificates (report delivered inline; REPORT.md write blocked)
A1 LipschitzCertInstance:43/339/598 Frob/S4/S8 → 3 shared lemmas (a1.lean) ~150 lines, pins unchanged. verified (v1 open)
A2 SmoothingGaussian integrability → Integrable.mul_bdd, integral_comp_eval ~45 verified
A3 stdGaussian_Ioo_pos → IsOpen.measure_pos w/ IsOpenPosMeasure instance moved up from NetSemantics:136 ~5
A4 euclid_norm_sq → EuclideanSpace.real_norm_sq_eq (rfl), 9 uses, 1 pin
A5 smoothing_certified_of_le + smoothProb_eq_real, MC/CP/NetSem ~40 verified (a2.lean)
A6 sqrt sandwich → sq_le_sq₀ ~10
A7 GramQ rowDotQ/castM = IBP.sumQ/castV rfl → one ℚ kit
killed: interval boxes vs Set.Icc
B1 ⭐ LipschitzCert.lean:1 unused import Foundation.Tensor → 33 cert modules (~1000 s) rebuild per Tensor edit; verified compiles without
B2 engine module Certificates/DenseEuclid.lean (denseE/reluE/CertifiedAt(now in GENERATED file)/mlp_gap_eq…); Foundation/IntervalBound imports trained-weights Instance
B3 4 CertifiedAt* predicates, 2 box vocabs; dead flat-Vec IBP residue ~60 lines, 3 pins (1649/1654/1665)
B4 Φ/Φ⁻¹ API scattered → Certificates/GaussianQuantile.lean
B5 README family map table (theorem←checker←data←generator), 6 different bridges
B6 stale: lakefile CertsHeavy claims SDPFull (in no lib); Instance header; NetSemantics:17 attribution; PairSDP dead imports
B7 naming legend (S/T, C/U, SC/SU, Uncon, F, e8, ImgsA–D; LipschitzCertDemo ns covers IBP/CROWN)
B8 smoothed-prob integral inline 36× → scoped notation
B9 SDP generator scripts copied; load_images ×9 → scripts/lipsdp_common.py
## arch_training (inline; scratch in arch_training/)
A1 bnMean = 𝔼 bridge (Fintype.expect_eq_sum_div_card, expect_equiv, expect_product); bnMean_shard 18→4, bnMean_pair 13→5, BN:705 Σ(x−μ)=0 2 lines; ~30; v1 verified never landed; sync-BN re-derived 3×
A2 BatchSealKit:1165 second swish derivative → LayerNorm swish kit + Real.sigmoid_*; ~30
A3 relu_apply_eq_max still in IntervalBoundConv:151; relu_continuous/relu_close/relu_entry_lipschitz re-derived; ~10 (root batch)
A4 small: sum_s2=sum_finProdFinEquiv, max4_sub_abs_le, hasDerivAt_mul_self_zero, bnForward_lb, bnIstd_le_one ~25
A5 Depthwise:401 / CNN:1418 collapses; padded conv read 5 defs (convPad, IBP.convTap, convWindow3, convWindow, dwWindow) rfl-equal
B1 ⭐ conv/pool index vocab + float conv forward live in Training/SgdDescentCnn → Float/ResNet34FloatBridge imports Training; propose Architectures/ConvIndex.lean + Float/ConvFloatBridge.lean
B2 SgdDescentCnn 6,847 lines real+float, header documents one; "Increment N" headings; Conv2Slot opened twice; 19 single-use margin instances (14 pinned) ~420 lines
B3 imports: ChannelLNBack (Architectures) imports Nets.ConvNeXt → move to Nets/ConvNeXt; SgdDescentCnn imports MobileNetV2Close unused (verified); Attention imports SE unused (lead)
B4 softmax Jacobian/CE grad inside Attention.lean → Architectures/Softmax.lean; IR imports Attention only for these
B5 stale docs: MaxPool3s2:53 "nothing downstream references" (18 do); 14 MlirCodegen line-number cites all wrong → cite fn names; Attention TOC; BatchNorm header omits sync half; README:60
B6 14 layer facts in BatchSeal namespace → BatchNorm / Continuity leaf
B7 activations 4 homes → Architectures/Activations.lean
B8 13 dead decls ~105 lines (list in report); forwarders dense_diff, rowSoftmax_has_vjp_mat', sum_s2
B9 descent naming legend (cnn1_ vs cnn_conv1_)
B10 depthwise*SgdDen in layer file (lead, low value)
## foundation_float (inline; scratch foundation_float/t1–t6)
A1 pdiv_lift_sum: softCE_grad 45→17, bceLogits ~12; ~40 verified
A2 mlp_has_vjp_at onto vjp_comp_diff_at 44→16, backward rfl-equal; ~28
A3 MLP pdiv_dense_b/dense_*_grad_correct/pdiv_relu ~30 (root, batch)
A4 IBP uniform-box collapse ×6 → ite_sign_lo/hi ~45 (keep names; generated cite)
A5 EfficientNetChainClose reindex_has_vjp ≡ PerChannelBN reindexVJP ~12
A6 sum_finProdFinEquiv₃ (sum_flat3, Tensor3.sum_flatten, sum_w3; sum_s2) ~12
A7 pdiv_finset_sum unused DecidableEq (pinned + comparator; user call)
A8 relu_apply_eq_max = (max_def_lt' _ 0).symm, move to MLP (dup of arch A3)
B1 ⭐ FloatClose defined atop 39-module/37k-line cone through SgdDescentCnn+MobileNetV2Close; compiles on FloatBridge alone (t3) → Float/FloatClose.lean + Float/ConvFloat.lean (same as arch B1)
B2 ⭐ Foundation not a layer: 13/29 import Nets/Certs/Codegen (SpecVJP imports 12 nets, IR imports EfficientNet, IntervalBound imports LipschitzCertInstance…); batchMap_has_vjp in EfficientNetChainClose; crossEntropy_differentiable in LinearTrainStep
B3 Tensor.lean: ~230 lines SDPA calculus → Attention; stale "§ The mean the collective computes" header; history-first docstring → TOC
B4 rank-3 kit (vjp3_comp etc.) 0 proof consumers; pdiv_clm, HasVJPMat.backward_unique 0 uses; MlirCodegen comments cite biPath3 wrongly → document or retire (~120, 2 blueprint nodes)
B5 BackNetFolds.lean = one audit-only def + import barrel → delete file
B6 PerChannelBN carries ~300 lines sync-BN sharding → DataParallelSync / SyncBnShard.lean (stops 222-module rebuild)
B7 lossGrad vs gradAt vs raw Fin 1 lift (123×) → legend or canonical
B8 stale headers: FloatBridge ("future work" done; no tier map), DataParallel "outside AST", two "piece 3", IR "Phase 0a spike", BatchMapVJPAt, CertifiedChain CORRECTION-first, SpecVJP, ConvMixedFloatBridge no /-!, Bf16Fold not audited
B9 softmax def in MLP, derivative in Attention (dup of arch B4) → Softmax.lean
## nets_resnet_small_convnext (inline; scratch nets_resnet_small_convnext/)
A1 ⭐ seals' *_continuous by hand → tag BatchSealKit atoms @[fun_prop] + 3 atoms; Zp4 all 14 blocks <1s verified; ~120 slice / ~250 across 4 seals
A2 ConvNeXtFoldGB 10 pinned alias fold lemmas (+4 EfficientNetFoldG) ~140; 10(+4) pins audit-only — user call
A3 den_cast ≡ den_castIdx
A4 CifarBnClose sum_channel_fibre → Equiv.sum_comp + Fintype.sum_prod_type (v1 verified, never landed) ~15; chanOf dead
A5 relu_nonneg / relu_continuous via relu_apply_eq_max (dup)
A6 transposeFlat_diff alias; r50StemGraphB forwarder (1 pin)
B1 ⭐ suite-wide kits live in ResNet-34 files/namespaces (FoldB=ResNet34PoCB ~340 refs/17 files/5 families; StepTieB helpers; SyncStepTieB 25 decls; SyncB castIdx; BackCertifiedTie leaf ties; FullBSeal pieces; small-net Cifar8PoC declared in MlpTrainStep) → move files keep namespaces
B2 ⭐ suffix scheme table (B, G, GB, B0 fossil, namespace≠file); README:73–76 legend wrong → replace with table
B3 unused imports of Nets/ResNet/ResNet34 in BackwardMaps, BatchSealKit, SpecVJP (verified); SmoothedLossCot→LinearTrainStep for crossEntropy_differentiable only
B4 stale headers: LinearTrainStep (first Start-here file!) says lemma "doesn't exist yet"; ResNet34.lean; ConvNeXt; "per-example only" ConvNeXt claims; 181 planning-section refs, 26/31 paths archived; T-tier glossary missing
B5 c·h·w seam 5 spellings (castIdx/laAssoc, den_cast, reassoc/reassocB private, EnetTiePoC.reassocB, reassocFwd) → one leaf
B6 dead: ResNet34.lean ~190 of 351 lines; LinearTrainStep render scaffold ~80; resnet34_has_vjp_at audit-only (census)
B7 CertLayer.comp_ok_of + r34PoolLayer in ResNet34FullBVJP → CertifiedChain / HeadLayers (overlaps certlayer §4.2)
B8 Cifar8Fold.lean docstring-only hub → delete
## codegen (inline; scratch codegen/)
A1 ⭐ 258 `if bf16 then .XBf16 zrnd … else .X` switches in 9 renderers → ~20 smart ctors (rfl), ~250–350 lines, bytes unchanged
A2 per-param opt step ×7 (adamOneM/adamOne4/enetAdamOne/rmsOneM≡enetRmsOne/vitAdamOne≡convnextAdamOne) + PGrad×3 → optOne + .rmsprop + emaSuffix; ~60+120 docstring (likely; regen diff)
A3 packed [θ|m|v] signature ×8 → packedTrainSig ~80 likely
A4 emitTok per-arm/dtype text copies (74 convolution blocks) — lead, SSA-order risk
A5 dropMaskSig ×4, tyOf, fmtFixed, vlnFwd site ×4, bnSite≡bnSiteP ~40
A6 foldl (*) 1 → List.prod 28 sites
A7 MatBridge 4 proofs rfl, 0 importers
B1 ⭐ batchMap/bnBatchLA/batchSlice (2,229 uses) live in StableHLO.lean → Foundation/Batched.lean keep namespace; verified compiles standalone; BatchMapVJPAt/BatchSealKit import StableHLO only for it
B2 ⭐ no renderer kit; R34 file is de facto kit (R34Opt, r34AdamVariant ×64 from R50) → Codegen/RenderKit.lean; MNv4 mnv4Blocks table the model
B3 Codegen holds non-codegen: optimizer specs (AdamStep, Sgd, RmsProp, GradClip, Lamb, DropPath) → Training/Optim; *RenderPC render nothing; EfficientNetRenderPC owns generic batched stages cbsB/projB used by R34/MNv; MatBridge; proof files import renderers (MobileNetV4BackB0→MobileNetV4RenderB for UibSpec)
B4 StableHLO.lean header "Stage A/closes R4 Ch1"; 17 § for 9.6k lines; ctors grouped by increment; dead mobilenetv2FwdGraph ~80; six sync-BN R=1 lemmas rfl over DataParallelSync.syncStats (6 pins kept); printer split
B5 emitted forward vs proven T2 graph linked only by prose → #guard per net (lead)
B6 SHlo suffix legend (F, B, Batched, Sgd/SgdB, Faithful/FaithfulV/FaithfulB…; RenderPC means different things)
B7 StableHLO imports 4 nets for relu6/sigmoid/layerScale; six dead `import LeanMlir.ViTRender` in renderers (verified)
B8 no Codegen README; Proofs/README :70/:74/:81 stale; 7 stale docstrings
B9 dead hazardous .train arm in r34/r50/mnv2 FwdChain; R34Bn second enum → bnEvalSite
B10 dead *Structured renderers ~325 lines
B11 printer two encodings (typed 98 vs .batched 118) — judgement, large
## crosscut (inline; scratch crosscut/ g.json decls.json tc/)
A1 ⭐ emitForwardEvalSig (335 lines) = emitForwardSig + BN suffix → fwdSigParts; verified byte-equal on 210 specs ~330
A2 11 forwarding grad-node lemma copies + denseBGradB_den dup body + 2 statement-identical pairs under different names (veclnGamma=headLnGamma, rowDenseBias_lnbeta=headLnBeta) ~140
A3 NetSpec param layout written 6× (nParams, paramShapes, heInitLayer, 3 sigs); seMid spelled 20×; copies DISAGREE (D1) → Layer.paramSlots/outShape ~400–600 likely
A4 fmt Float ×6 in demos; A5 tensor-type strings 3 helpers + 103 inline, tensorTy [] gives invalid tensor<xf32>; A6 TestR34SyncBnCheck not ported to SyncBnCheck.run ~280; A7 reference NetSpecs resnet34 ×5 copies
v1 §10 still open: containsSubstr/hasSubstr (String.contains works in 4.34), walkDir, isSuffixOf, LE codec, mkLabels ×6, 8 inline iree-compile
B1 ⭐ StableHLO imports 5 nets for relu6/sigmoid/SE/layerScale + cifar graph theorems; MobileNetV2 edit rebuilds ~152 → relu6→MLP, sigmoid/SE→SE.lean, layerScale→Arch
B2 ⭐ 129 of 140 den-node lemmas in Nets/, 26 kinds in >1 file → Codegen/GradNodesB.lean (Bf16GradNodes is the model)
B3 ⭐ SpecVJP (in Foundation) pulls VerifiedTrain+IreeRuntime FFI into Certs; lakefile "never the codegen" false
B4 7(8) dead ViTRender imports verified byte-identical 121 artifacts; program-side ViTRender.lean no shipping consumer
B5 suffix legend table (B/G/GB/PC/Eval/Paper/Full/T; B0 3 meanings; _faithful/_den/_bridge/_eq_vjp); optional *BackB0→*BackBlock (127 mentions)
B6 stale: lakefile "LeanMlir type-checks whole repo" (91/259); counts disagree README vs lakefile; EfficientNet headline cites efficientnet_net_tied not _tiedG; MlirCodegen "MLPs and CNNs"; VerifiedTrain lr/val claims; MobileNetV2FullPaper name; tests/Audit* leftovers; genCifarPgdStep "proven"
B7 no map of non-proof world (two pipelines); lean_lib Codegen misnamed/unused
B8 emitTrainStepBody single 3,416-line def; VerifiedTrain 4 jobs → split
B9 optimizer specs → Training/Optim (dup codegen B3)
B10 layering undocumented; Foundation→Architectures edges; net-family cycles; Nets/Common
B11 dead: MnistData.lean, IreeRuntime Mlp/Cnn/CifarLayout + 3 FFI bindings, mobilenetv2FwdGraph, VerifiedSpec forwarders
D1 ⚠ totalParams vs paramShapes on mbConv+SE (ENet-B0 4.02M vs 7.16M); Train.lean:697 slices by totalParams → efficientnet-train/v2 path misaligned?
## nets_mobilenet_enet_vit (inline; scratch nets_mobilenet_enet_vit/)
A1 ENet MBConv VJP twice under two names (mbStridedFwdB=mbDownBodyB, mbExpFwdB=mbBodyB, rfl) ~50
A2 BatchSealKit swish (dup of arch A2)
A3 ⚠ sigmoidScalarDeriv_eq deleted by fb00989c sweep → v1 defect 6 REGRESSED; restore via Real.deriv_sigmoid + pin; swishScalarDeriv_eq also unpinned/exposed
A4 broadcastFlat_has_vjp / reindex_has_vjp → reindexVJP (rfl site at EfficientNetBackB0:397)
A5 relu6 linearisation copies relu's (v1 §7 verified never landed) ~30
A6 mhSlab = headSliceMat rfl
A7 small: mulVec_headPadMat, ViTVecLN h3 fun_prop; dead vitBlockBack, comp_ok, Ah2_continuous; rf dup MNv2/MNv4 seals
B1 ⭐ generic batched VJP code in three ENet files (ChainClose batchMap_has_vjp 62 refs; EnetTiePoC reassocB 324 refs/cInB 90 in the fused-SGD tie; EfficientNetBackB0 generic *_faithful) → BatchMapVJPAt + Foundation/BatchedBackLinks
B2 relu6/sigmoid/SE/broadcastFlat/layerNormVec/layerScale in net files (dup crosscut B1)
B3 ⭐ canonical conv-net chain exists (R34/R50/MNv2/MNv4): BackB0→FullB→FullBVJP→FullBSeal→StepTieB→BackChains+WholeBackCertifiedTieB→SyncB/SyncStepTieB; ENet/ViT deviate; tier table; legend not renames (~350 audit, 225 comparator refs)
B4 MNv2 retired per-example files wired in via dead imports (verified)
B5 ~15 false docstrings (all-reduce "outside AST" ×4, ViTBackNet "no conv net folds to logits" false since 6778f8c9, EfficientNetFullWhole cites nonexistent; dangling names); 39 dated notes, 378 ⭐
B6 stage CertLayer vocab scattered (projLayer in MNv2BackB0, cbReluLayer R34BackB0) → StageLayers (overlaps certlayer)
B7 sync twin kit split R34/ENet; P4 collective proof ×8
B8 BN-ε positivity bundles 4 spellings
B9 ViTVecLN five jobs
B10 generic fused folds with mnv2 names
B11 rename mobilenetv2PaperPC_has_vjp_at → mnv2B_full_has_vjp_at (6 refs)
