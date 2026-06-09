import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.ConvNeXt
import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.MnistCNN
import LeanMlir.Proofs.CifarCNN
import LeanMlir.Proofs.IR
import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.StableHLOParse
import LeanMlir.Proofs.StridedConv
import LeanMlir.Proofs.ResNet34
import LeanMlir.Proofs.PerChannelBN
import LeanMlir.Proofs.LinearTrainStep
import LeanMlir.Proofs.MlpTrainStep
import LeanMlir.Proofs.CnnTrainStep
import LeanMlir.Proofs.CifarBnClose
import LeanMlir.Proofs.CnnChainClose
import LeanMlir.Proofs.MobileNetV2Close
import LeanMlir.Proofs.MobileNetV2RenderPC
import LeanMlir.Proofs.MobileNetV2ChainClose
import LeanMlir.Proofs.ConvLossFold
import LeanMlir.Proofs.EfficientNetClose
import LeanMlir.Proofs.EfficientNetRenderPC
import LeanMlir.Proofs.EfficientNetChainClose
import LeanMlir.Proofs.EfficientNetFullB0
import LeanMlir.Proofs.ResNet34Close
import LeanMlir.Proofs.ResNet34RenderPC
import LeanMlir.Proofs.ResNet34ChainClose
import LeanMlir.Proofs.ConvNeXtClose
import LeanMlir.Proofs.ConvNeXtChainClose
import LeanMlir.Proofs.ViTFwdGraph
import LeanMlir.Proofs.ViTClose
import LeanMlir.Proofs.ViTChainClose
import LeanMlir.Proofs.ViTVecLN

open Proofs

-- Foundation
#print axioms pdiv_id
#print axioms pdiv_const
#print axioms pdiv_reindex
#print axioms pdiv_add
#print axioms pdiv_mul
#print axioms pdiv_comp
#print axioms pdiv_finset_sum
#print axioms pdivMat_comp
#print axioms pdivMat_matmul_left_const
#print axioms pdivMat_matmul_right_const
#print axioms pdivMat_rowIndep
#print axioms pdivMat_colIndep
#print axioms pdivMat_scalarScale
#print axioms pdivMat_transpose

-- MLP
#print axioms pdiv_dense
#print axioms pdiv_dense_W
#print axioms pdiv_dense_b
#print axioms pdiv_relu
#print axioms dense_weight_grad_correct
#print axioms dense_bias_grad_correct
#print axioms relu_has_vjp_correct
#print axioms mlp_has_vjp_correct

-- CNN
#print axioms maxPool2_has_vjp3_correct
#print axioms conv2d_has_vjp3
#print axioms conv2d_has_vjp3_correct

-- Depthwise
#print axioms depthwise_has_vjp3_correct
#print axioms depthwiseStride2Flat_has_vjp_correct

-- BatchNorm
#print axioms pdiv_bnAffine
#print axioms pdiv_bnCentered
#print axioms pdiv_bnIstdBroadcast
#print axioms pdiv_bnNormalize
#print axioms bn_input_grad_correct

-- Residual
#print axioms residual_has_vjp_correct
#print axioms residualProj_has_vjp_correct

-- SE
#print axioms seBlock_has_vjp_correct

-- LayerNorm / GELU / Swish
#print axioms pdiv_gelu
#print axioms gelu_has_vjp_correct
#print axioms swish_has_vjp_correct
#print axioms layerNorm_has_vjp_correct

-- Attention (apex)
#print axioms pdiv_softmax
#print axioms softmaxCE_grad
#print axioms sdpa_back_Q_correct
#print axioms sdpa_back_K_correct
#print axioms sdpa_back_V_correct
#print axioms mhsa_has_vjp_mat_correct
#print axioms transformerBlock_has_vjp_mat_correct
#print axioms transformerTower_has_vjp_mat
#print axioms vit_body_has_vjp_mat
#print axioms vit_full_has_vjp
#print axioms vit_full_has_vjp_correct

-- Codegen smooth-point bridge theorems (MLP.lean)
#print axioms relu_codegen_matches_canonical
#print axioms relu_canonical_diagonal

-- Codegen smooth-point bridge theorems (CNN.lean / MaxPool2)
#print axioms pdiv3_maxPool2_smooth
#print axioms maxPool2_codegen_matches_canonical

-- HasVJPAt pointwise framework (E.5) — kills `correct := rfl` escape
-- at the three kinked operators (relu, mlp, maxPool2) at smooth points.
#print axioms relu_has_vjp_at
#print axioms mlp_has_vjp_at
#print axioms mnistLinear_has_vjp_correct
#print axioms maxPool2_has_vjp_at3

-- Capstone: end-to-end ResNet-style CNN whole-network VJP (CNN analogue
-- of vit_full_has_vjp) + global-average-pool VJP it depends on.
#print axioms globalAvgPoolFlat_has_vjp
#print axioms globalAvgPoolFlat_has_vjp_correct
#print axioms cnn_has_vjp_at
#print axioms cnn_has_vjp_at_correct

-- Whole-network VJPs for the depthwise/SE/LN-based architectures
-- (each a representative concrete net; see per-file docstrings for the
-- fixed structural choices). New activations: relu6 (MNv2, two-sided kink),
-- sigmoid (EfficientNet SE gate), layerScale (ConvNeXt).
#print axioms relu6_has_vjp_at
#print axioms mobilenetv2_has_vjp_at_correct
-- MobileNetV2: concrete whole-network instance, every ReLU6 smoothness
-- hypothesis discharged (unconditional, degenerate constant-activation
-- witness via bnForward_const).
#print axioms MobileNetV2Concrete.mnv2Concrete_has_vjp_correct
#print axioms layerScale_has_vjp_correct
#print axioms convnext_has_vjp_at_correct
-- ConvNeXt promoted to an UNCONDITIONAL global VJP (all-smooth: LN+GELU, no
-- kinks) — joins vit_full as a whole-net VJP holding at every input, not a
-- fixed point. Only the four `0 < ε` LayerNorm conditions.
#print axioms convnext_has_vjp
#print axioms convnext_has_vjp_correct
#print axioms sigmoid_has_vjp
#print axioms efficientnet_has_vjp_at_correct
-- EfficientNet promoted to an UNCONDITIONAL global VJP (all-smooth: swish +
-- sigmoid SE gate + conv/bn, no kinks) — joins vit_full and convnext.
-- Only the `0 < ε` batch-norm conditions.
#print axioms efficientnet_has_vjp
#print axioms efficientnet_has_vjp_correct

-- Chapter-4 MNIST 2D CNN (no BN): the conditional whole-network capstone,
-- and the concrete tiny instance with every smoothness hypothesis
-- discharged (unconditional — closes the gap that the *_at apexes are
-- never instantiated).
#print axioms mnistCnnNoBn_has_vjp_at_correct
#print axioms Micro.mnistMicroCnn_has_vjp_correct
-- Tier-1 discharged instance: multi-channel (2→2 mixing conv), four pool
-- windows per channel, 10-class head — every smoothness hypothesis
-- discharged via the reusable structural lemmas (no `native_decide`).
#print axioms maxPool2Smooth_of_injective
#print axioms Mini.miniCnn_has_vjp_correct
-- Tier-2: same CNN with genuine 3×3 SAME-padding convolutions
-- (center-structured, via conv2d_center3x3), every hypothesis discharged.
#print axioms conv2d_center3x3
#print axioms Spatial.spatialCnn_has_vjp_correct
-- Chapter-3 MLP: concrete whole-network instance, every ReLU smoothness
-- hypothesis discharged (unconditional) — the simplest kinked capstone.
#print axioms MlpConcrete.mlpConcrete_has_vjp_correct
-- ResNet-style CNN *with* BN: concrete whole-network instance, every
-- smoothness hypothesis discharged (injective stem for maxpool no-ties +
-- exact-istd BN positivity; resblocks via γ=0). The last conditional
-- capstone instantiated.
#print axioms CnnConcrete.cnnConcrete_has_vjp_correct

-- Denoted StableHLO-subset IR (Phase 0a/0b spike, planning/typed_ir.md):
-- the emitted backward graph denotes the proven HasVJP.backward.
#print axioms IR.dense_back_bridge
#print axioms IR.relu_back_bridge
-- Phase 2: the emitted transposed-convolution graph denotes the proven
-- conv input-VJP (the reversed-kernel identity, by expansion at the
-- Spatial instance's conv shapes).
#print axioms IR.conv_back_bridge_1to2
#print axioms IR.conv_back_bridge_2to2
-- Phase 2: the emitted tile-compare-select graph denotes the canonical
-- maxpool backward at smooth points (no argmax ties).
#print axioms IR.maxpool_back_bridge
-- Phase 1 smooth activations: emitted `dy ⊙ act'(x)` graph denotes the
-- proven diagonal-Jacobian backward.
#print axioms IR.gelu_back_bridge
#print axioms IR.swish_back_bridge
#print axioms IR.sigmoid_back_bridge
-- BatchNorm: the emitted reduce+broadcast+elementwise graph denotes the
-- proven consolidated 3-term rank-1 backward (+ the affine γ·dy half).
#print axioms IR.bn_affine_back_bridge
#print axioms IR.bn_normalize_back_bridge
#print axioms IR.bn_back_bridge
#print axioms IR.layernorm_back_bridge
#print axioms IR.softmax_back_bridge
-- Phase 3: IR-level chain rule + an end-to-end composite bridge.
#print axioms IR.denote_subst
#print axioms IR.twoDense_back_bridge
#print axioms IR.se_back_bridge
-- Tensor3 IR: conv/maxpool lifted into a composable backward graph + chain rule.
#print axioms IR.denote_subst3
#print axioms IR.maxpool3_node_bridge
#print axioms IR.conv3_node_bridge_1to2
#print axioms IR.conv_compose3
-- Flatten bridge: flattened Back3 graph denotes the proven flattened-layer Vec backward.
#print axioms IR.maxpool_flatten_bridge
#print axioms IR.conv_flatten_bridge_1to2
-- HasVJPAt smooth-point variants + a real dense→relu block via vjp_comp_at.
#print axioms IR.relu_at_bridge
#print axioms IR.dense_at_bridge
#print axioms IR.denseRelu_at_bridge
-- Final assembly: the emitted whole-MLP backward graph denotes the proven
-- whole-network VJP (mlp_has_vjp_at), per-op _at bridges chained via denote_subst.
#print axioms IR.mlp_whole_bridge
-- Parameter gradients (train-step pieces): the emitted weight/bias backward
-- (outer-product `dot_general` / batch `reduce`) computes the certified
-- cotangent-contracted Jacobians wrt W and b at the actual chain cotangent.
#print axioms IR.weight_grad_bridge
#print axioms IR.bias_grad_bridge
#print axioms IR.mlp_layer1_weight_grad_bridge
-- Forward IR (Phase 2): the emitted forward graph denotes the proven forward
-- map mlpForward, and its sub-graphs denote the pre-activations the backward
-- reads — so the whole train-step module (fwd + back + grads) is proof-backed.
#print axioms IR.denote_subst_fwd
#print axioms IR.mlp_fwd_bridge
#print axioms IR.mlp_fwd_preact0
#print axioms IR.mlp_fwd_preact1
-- Loss cotangent (rest of Phase 4): the emitted softmax−onehot loss head
-- denotes the proven softmax-CE gradient ∂L/∂logits, so the cotangent fed to
-- the backward is proof-backed too (not supplied) — full train step closed.
#print axioms IR.lossCot_bridge

-- R4 printer-faithfulness, Stage A (Chapter 2): the StableHLO-subset AST's
-- denotation `den` matches the proven math for every piece of the linear
-- train step — forward logits, dense input-VJP, softmax-CE cotangent (to the
-- proven ∂CE/∂logits), and the weight/bias parameter Jacobians.
#print axioms StableHLO.fwdGraph_faithful
#print axioms StableHLO.backGraph_faithful
#print axioms StableHLO.softmaxDiv_expe_faithful
#print axioms StableHLO.lossCotGraph_faithful
#print axioms StableHLO.lossCotGraph_isCEgrad
#print axioms StableHLO.wGrad_isWeightJacobian
#print axioms StableHLO.bGrad_isBiasJacobian
-- SGD update proven (not trusted) for plain SGD on the linear net.
#print axioms StableHLO.sgdW_descends_certified_grad
#print axioms StableHLO.sgdB_descends_certified_grad
-- M1: the linear SGD step bundled to the certified closed-form softmax-CE
-- gradient (softmax − onehot) contracted with the certified ∂logits/∂θ Jacobian.
#print axioms StableHLO.lossCot_eq_softmax_sub_onehot
#print axioms StableHLO.sgdW_descends_softmaxCE_grad
#print axioms StableHLO.sgdB_descends_softmaxCE_grad
-- M1 chain-rule fold: the SGD step is literally θ − lr·∂Loss/∂θ.
#print axioms StableHLO.crossEntropy_differentiable
#print axioms StableHLO.denseWeightMap_differentiable
#print axioms StableHLO.lossWeightGrad_eq_sum
#print axioms StableHLO.sgdW_descends_loss_gradient
-- M1 rendering half: the multi-output `renderModuleN`/`denN` train-step module —
-- each rendered parameter output denotes the certified SGD step.
#print axioms StableHLO.linWeightDen_is_loss_descent
#print axioms StableHLO.linBiasDen_is_certified
-- M2: the MLP per-layer parameter-gradient assembly (layer-0 cotangent + the
-- weight/bias bridges completing all three layers; Crux A).
#print axioms IR.mlp_layer0_weight_grad_bridge
#print axioms IR.mlp_layer0_bias_grad_bridge
#print axioms IR.mlp_layer1_bias_grad_bridge
#print axioms IR.mlp_layer2_weight_grad_bridge
-- M3: the CNN convolution parameter-gradient bridges (kernel grad = correlation).
#print axioms conv_weight_grad_bridge
#print axioms conv_bias_grad_bridge
-- CNN render close: the rendered conv weight/bias SGD outputs denote θ − lr·certified.
#print axioms cnn_render_convW_certified
#print axioms cnn_render_convb_certified
-- Chain: the composed cotangent subgraphs reduce to the explicit relu'⊙Wᵀ·… backprop
-- formulas (denote_subst). Fold: the output-layer total-loss gradient (unconditional).
#print axioms IR.mlpCotOut1_denote
#print axioms IR.mlpCotOut0_denote
#print axioms IR.mlp_output_total_loss_grad
-- The conditional hidden-layer folds: total loss gradient wrt W₁ / W₀ (the chain runs
-- back through one / two ReLU kinks — discharged at a smooth point).
#print axioms IR.mlp_hidden_total_loss_grad
#print axioms IR.mlp_input_total_loss_grad
-- Whole-net capstone: every weight layer's total-loss gradient at once (one statement).
#print axioms IR.mlp_whole_net_weight_grads
-- Render close: the rendered MLP train step's six param outputs (W₂',W₁',W₀',b₂',b₁',b₀')
-- denote θ − lr·(certified per-layer gradient). The denotation side of MlpRender.
#print axioms IR.mlp_render_W2_certified
#print axioms IR.mlp_render_W1_certified
#print axioms IR.mlp_render_W0_certified
#print axioms IR.mlp_render_b2_certified
#print axioms IR.mlp_render_b1_certified
#print axioms IR.mlp_render_b0_certified
-- R4 Stage A, Chapter 3 (MLP): ReLU forward (maximum) + backward (select),
-- whole-MLP forward (= mlpForward) and backward chain (= mlp_has_vjp_at, the
-- conditional whole-network VJP at a smooth point).
#print axioms StableHLO.reluF_faithful
#print axioms StableHLO.selectPos_faithful
#print axioms StableHLO.mlpFwdGraph_faithful
#print axioms StableHLO.mlpBackGraph_faithful
-- R4 Stage A, Chapter 4 (CNN): conv + maxpool forward ops; the whole MNIST-CNN
-- forward graph denotes the proven mnistCnnNoBnForward.
#print axioms StableHLO.flatConvF_faithful
#print axioms StableHLO.maxPoolF_faithful
#print axioms StableHLO.cnnFwdGraph_faithful
#print axioms StableHLO.convBack_faithful
#print axioms StableHLO.maxPoolBack_faithful
-- A2c: the whole-chain CNN backward graph denotes the proven conditional
-- whole-network VJP mnistCnnNoBn_has_vjp_at.backward (MLP-analog of
-- mlpBackGraph_faithful).
#print axioms StableHLO.cnnBackGraph_faithful

-- Chapter-5 CIFAR-10 2D CNN (no BN): the conditional whole-network VJP capstone
-- (two conv→conv→pool stages, channel changes, two maxpools) and the verified
-- forward-graph faithfulness it is rendered through. The Chapter-5 peers of
-- mnistCnnNoBn_has_vjp_at_correct and StableHLO.cnnFwdGraph_faithful. Trained on
-- GPU through the proof-rendered StableHLO by the `cifar-verified` exe.
#print axioms cifarCnn_has_vjp_at_correct
#print axioms StableHLO.cifarFwdGraph_faithful
-- Concrete tiny CIFAR instance: every smoothness hypothesis discharged
-- (unconditional) — including the Chapter-5-specific SECOND-pool no-tie
-- condition (first-pool output positionally injective: the four 2×2 window
-- maxima 6,8,14,16 are distinct, via Nat.cast_max + omega). Non-vacuity witness.
#print axioms Tiny.cifarTinyCnn_has_vjp_correct

-- Chapter-5 CIFAR **BatchNorm** variant: per-example BN (bnForward, reduce over
-- the feature vec, scalar γ/β) inserted after each conv. The whole-network VJP
-- (conv→BN→relu blocks via convBnRelu_has_vjp_at), the verified BN forward-graph
-- faithfulness, and the typed BN input-VJP op (the consolidated O(N) three-term
-- gradient = pdiv-Jacobian, bn_input_grad_correct). GPU-trained by `cifar-bn-verified`.
#print axioms cifarCnnBn_has_vjp_at_correct
#print axioms StableHLO.cifarBnFwdGraph_faithful
#print axioms StableHLO.bnBack_faithful

-- Chapter-6 ResNet-style net: the verified forward-graph faithfulness for the
-- structure the proven whole-network VJP `cnn_has_vjp_at` already covers
-- (stem → maxpool → identity block → projection block → GAP → dense). The
-- residual skips render as `addV` fan-ins (the block-input subtree reused in both
-- operands, tree-safe); the head is global-average-pool (`gapF`). Its whole-net
-- VJP `cnn_has_vjp_at_correct` (+ the unconditional CnnConcrete instance) is
-- audited above; this is the rendered-forward peer (cf. cifarBnFwdGraph_faithful).
#print axioms StableHLO.resnetFwdGraph_faithful

-- Chapter-6 ResNet **Milestone B** (toward real ResNet-34): stride-2 SAME
-- convolution — the downsampling op that gates the jump from the ch6-A ResNet-
-- style net to 34 layers. Key identity: conv_stride2 = decimate2 ∘ (stride-1
-- conv), so the input-VJP reuses the proven conv2d_has_vjp3 (via vjp_comp + the
-- decimation reindex-VJP) rather than re-deriving it with stride arithmetic.
#print axioms flatConvStride2_has_vjp_correct
-- ...and its weight-VJP (the kernel grad for training a strided block), likewise
-- by vjp_comp reusing the proven conv2d_weight_grad_has_vjp.
#print axioms flatConvStride2_weight_grad_has_vjp_correct
-- Deep-block chain: a list of same-type residual blocks composes to a single
-- VJP (the 16-block depth of ResNet-34, as a List.length rather than per-block
-- boilerplate). Generic enabler, proven by induction chaining vjp_comp.
#print axioms vjp_chain_correct
-- Strided downsampling block (conv stride-2 → BN → relu), the workhorse opening
-- each ResNet-34 stage: its conditional VJP, by vjp_comp_at reusing
-- flatConvStride2_has_vjp + the proven BN/ReLU VJPs.
#print axioms convBnReluStrided_has_vjp_at_correct
-- Strided residual-PROJECTION block (relu(proj(x)+F(x)), both first-conv and 1×1
-- skip stride-2) — the downsampling residual block at each ResNet-34 stage start.
-- Via residualProj_has_vjp_at over the strided proj + strided body.
#print axioms rblkPStrided_has_vjp_at_correct
-- Stage assembly: the conditional (_at) deep-block chain (vjp_chain_at, the
-- identity-block run of a stage) and a full ResNet stage = downsample block then
-- that chain (resStage). The composition machinery for the 4-stage whole net.
#print axioms vjp_chain_at_correct
#print axioms resStage_has_vjp_at_correct
-- THE WHOLE-NETWORK ResNet-34 VJP: dense ∘ GAP ∘ stage₄ ∘ stage₃ ∘ stage₂ ∘
-- stage₁ ∘ maxpool ∘ stem, each stage = (identity-block chain) ∘ downsample, the
-- 16 basic blocks as the idsᵢ lists. The conditional whole-net VJP (its HasVJPAt
-- .correct field is the ℝ-carrying pdiv-Jacobian), folded from the verified
-- vjp_comp_at / vjp_chain_at — the 34-layer structural analogue of cnn_has_vjp_at.
#print axioms resnet34_has_vjp_at
-- B7: the UNCONDITIONAL concrete instance — `resnet34_has_vjp_at` instantiated at
-- 1ch/32×32 with the verified components (strided identity stem, 3 strided
-- downsamplers, 16 zero-weight identity blocks, GAP, dense), every smoothness/no-tie
-- hypothesis discharged. Makes the verified ResNet-34 non-vacuous; the ResNet-34
-- peer of CnnConcrete.cnnConcrete_has_vjp_correct.
#print axioms ResNet34Concrete.resnet34Concrete_has_vjp_correct
-- B8: per-channel BatchNorm — the real-ResNet BN (each channel-slice normalized with
-- its own γ_c/β_c, γ/β : Vec oc). Block-diagonal VJP: each channel runs its own
-- bn_has_vjp, cross-channel blocks vanish (pdivMat_rowIndep_perRow, a per-row
-- generalization of rowwise_has_vjp_mat). The genuinely-new "parallel/blockwise VJP".
#print axioms bnPerChannelFlat_has_vjp_correct
-- B8a': the RENDERABLE per-channel BN backward — the per-example three-term
-- bn_grad_input run on each channel-slice, proven faithful (= pdiv-Jacobian) under
-- 0<ε. The closed form a per-channel bnBack op / renderLNBack-per-channel emits.
#print axioms bnPerChannel_grad_input_correct
-- B9 entry: per-channel BN on the network's Tensor3 (oc*h)*w activation layout —
-- bnPerChannelFlat (Mat-split oc*(h*w)) conjugated by the layout-bridge reindex
-- reassocFwd/reassocBack (a pure product re-association, VJP via pdiv_reindex like
-- decimateFlat). The plug-in op that wires per-channel BN into ResNet-34.
#print axioms bnPerChannelTensor3_has_vjp_correct
-- B9 entry (cont.): the RENDERABLE per-channel BN backward in the Tensor3 layout —
-- the bridge reindexes are permutations, so the vjp_comp backward collapses to
-- reassocBack ∘ bnPerChannel_grad_input ∘ reassocFwd. The closed form bnPerChannelBack
-- emits (per-channel renderLNBack over the spatial axis); faithful = pdiv-Jacobian.
#print axioms bnPerChannelTensor3_grad_input_correct
-- ch8 E5 (EfficientNet BATCH-norm): batch-norm per channel on the [N,C,H,W] layout —
-- bnPerChannelFlat with m = N·h·w (the per-channel group enlarged from one example's
-- spatial cells to the WHOLE BATCH), conjugated by the [N,C,H,W]↔[C,N·H·W] TRANSPOSE
-- bridge (bnchwFwd/bnchwBack, a permutation reindex like reassoc). The renderable
-- batch-norm backward = bnchwBack ∘ bnPerChannel_grad_input(m=N·h·w) ∘ bnchwFwd; faithful
-- = pdiv-Jacobian of bnBatchTensor4 (batch-coupled, block-diagonal across channels, 0<ε).
-- This is what makes EfficientNet genuinely all-swish (instance-norm degenerated the GAP).
#print axioms bnBatchTensor4_grad_input_correct
-- B8b: the per-channel BN SHlo op pair backward-faithfulness — the rendered
-- bnPerChannelBack op (4-D reshape, per-channel reduce over [2,3], rank-1 γ dims=[1])
-- denotes the proven block-diagonal BN input-VJP (= pdiv-Jacobian of bnPerChannelTensor3,
-- under 0<ε). The per-channel peer of bnBack_faithful. (bnPerChannelF_faithful is rfl
-- ⇒ covered structurally by roundtrip, not separately audited.)
#print axioms StableHLO.bnPerChannelBack_faithful
-- R4 syntactic core: the emitted op-graph is a faithful serialization
-- (parse (toToks (skel a)) = some (skel a)). (The underlying `parse_toToks`
-- lemma is even cleaner — `[propext]` only, no ℝ — but the exact-triple gate
-- wants all three, so the ℝ-carrying headline `roundtrip` is the audited one.)
#print axioms StableHLO.roundtrip
-- CIFAR-BN render CLOSE — the per-channel BN scale/shift parameter-gradient bridges
-- (the last params of the CIFAR-BN train step). γ/β enter BN affinely (y = γ·x̂ + β,
-- x̂ independent of both), so the rendered per-channel reduces dγ_c = Σ_s dy·x̂ and
-- dβ_c = Σ_s dy equal the certified pdiv-Jacobian of per-channel BN (as a function of
-- γ resp. β) contracted with the cotangent — via pdiv_reindex/pdiv_mul/pdiv_const over a
-- channel-gather. The affine BN analogue of bias_grad_bridge; no 0<ε (ε only enters the
-- constant x̂). Together with bnPerChannelTensor3_grad_input_correct (BN input grad) and
-- the conv/dense bridges, every CIFAR-BN train-step parameter output is now certified.
#print axioms bnPerChannel_grad_gamma_correct
#print axioms bnPerChannel_grad_beta_correct
#print axioms cifar_bn_render_gamma_certified
#print axioms cifar_bn_render_beta_certified
-- CNN conv-close UPGRADE — the conv param closes pinned to the ACTUAL backward-chain
-- cotangent (not a generic c). The chain from dy: dense-head flat Back chain
-- (cnnDenseHeadCot, the mlpCotOut mechanism) → maxpool-back (Back3 node via flatDenote,
-- crossing the flatten boundary) → relu mask → conv2-back (Back3 node via flatDenote) →
-- relu mask. Instantiates cnn_render_conv{W,b}_certified at cnnChainCotW2/W1 — so each
-- conv θ output denotes θ − lr·(certified ∂conv/∂θ · the cotangent the chain delivers).
-- The conv analogue of mlpCotOut0/1; pins the cotangent (the further "= ∂loss/∂θ" fold is
-- the separate pdiv G = Back.denote step).
#print axioms cnnDenseHeadCot_denote
#print axioms cnn_render_convW2_chain_certified
#print axioms cnn_render_convb2_chain_certified
#print axioms cnn_render_convW1_chain_certified
#print axioms cnn_render_convb1_chain_certified
-- MobileNetV2 CLOSE (planning/mobilenetv2_close.md Item C) — the "free close" generic in the
-- cotangent: every MobileNetV2 train-step parameter output denotes θ − lr·(certified Jacobian ·
-- cotangent). Three genuinely-new bridge families (the 1×1 conv W/b, BN γ/β, and dense W/b
-- families reuse the M3/CIFAR-BN/M2 bridges verbatim at the MobileNetV2 shapes, so are covered by
-- cnn_render_conv{W,b}_certified / cifar_bn_render_{gamma,beta}_certified / weight_grad_bridge):
--   • depthwise (stride-1) W/b — .correct of the proven depthwise_weight_grad_has_vjp3 /
--     depthwise_bias_grad_has_vjp (per-channel transpose trick / spatial reduce), SGD-wrapped.
--   • stem strided 3×3 conv W/b — flatConvStride2_weight_grad_has_vjp (ch6) + a new strided-conv
--     bias VJP (decimate ∘ conv2d-in-b).
--   • strided depthwise W/b (4 of 6 blocks downsample) — new depthwiseStride2 weight/bias VJPs,
--     the decimate ∘ stride-1 recipe (vjp_comp of a proven stride-1 depthwise VJP + decimateFlat).
#print axioms mnv2_depthwise_weight_grad_bridge
#print axioms mnv2_depthwise_bias_grad_bridge
#print axioms mnv2_render_depthwiseW_certified
#print axioms mnv2_render_depthwiseb_certified
#print axioms mnv2_render_stem_convW_certified
#print axioms mnv2_render_stem_convb_certified
#print axioms mnv2_render_depthwiseW_strided_certified
#print axioms mnv2_render_depthwiseb_strided_certified
-- MobileNetV2 RENDER (planning/mobilenetv2_close.md Item A) — the PER-CHANNEL-BN typed SHlo
-- forward graph at the full ch7 render dims (3×224² → 7×7×64): strided stem → 6 inverted-residual
-- blocks (4 stride-2 downsampling via depthwiseStridedF, 2 stride-1 with an addV skip) → conv-bn-
-- relu6 head → GAP → dense. Per-channel BN (bnPerChannelF, γ/β : Vec c) at every BN site, so it
-- matches the operational render's BN flavor (StableHLO's prior mobilenetv2FwdGraphFull used SCALAR
-- bnF, tied to the scalar mobilenetv2Forward_full — a different function than the render). The
-- faithfulness `den (graph) = mobilenetv2Forward_full_pc` is the "text = render of a proven graph"
-- forward half at the render's per-channel BN; prerequisite for the structured render (Item B).
#print axioms StableHLO.mobilenetv2FwdGraphFullPC_faithful
-- MobileNetV2 cotangent-chain CLOSE (Item D) — the inverted-residual analogue of CnnChainClose/
-- ResNet34ChainClose. The Item C conv/depthwise bridges pinned to the cotangent the backward chain
-- delivers: project→depthwise→expand composes the rendered backward denotations — relu6 two-sided-kink
-- mask (selectMid, if 0<x<6), per-channel BN input-VJP (bnPerChannelTensor3_grad_input), 1×1 conv
-- input-VJP (conv2d_has_vjp3 via flatten), depthwise input-VJP (depthwiseFlat / depthwiseStride2Flat
-- _has_vjp) — into invresCotPc/Dc/EcS1/EcS2. The linear bottleneck (no relu6 after the addV) makes the
-- project-BN output cotangent dyOut directly; the stride-2 blocks carry the expand-side cotangent at
-- 2h×2w (the _s2 split). Each conv/depthwise θ output denotes θ − lr·(certified ∂/∂θ · the-actual-chain-
-- cotangent). Pins the cotangent; the = ∂loss/∂θ fold stays separate, as for the CNN. 3-axiom clean.
#print axioms invres_render_projW_chain_certified
#print axioms invres_render_projb_chain_certified
#print axioms invres_render_dwW_s1_chain_certified
#print axioms invres_render_dwb_s1_chain_certified
#print axioms invres_render_dwW_s2_chain_certified
#print axioms invres_render_dwb_s2_chain_certified
#print axioms invres_render_expW_s1_chain_certified
#print axioms invres_render_expW_s2_chain_certified
#print axioms mnv2_stem_render_convW_chain_certified
-- THE COTANGENT PASS / = ∂loss/∂θ FOLD — what ties the certified per-layer Jacobian (Item C/D) to
-- genuine gradient descent on the LOSS. The single gradient of the whole loss wrt a conv/depthwise
-- param equals that certified Jacobian contracted with ∂loss/∂(layer output) — pdiv_comp (chain rule)
-- on G ∘ (the θ-weight-map), at a smooth point (the downstream loss G differentiable at the layer
-- output, the relu6/BN smoothness bundle). The conv/depthwise analogue of mlp_hidden_total_loss_grad;
-- generic in G, so one theorem covers every conv (CNN/CIFAR/MobileNetV2/r34) and one every depthwise.
-- The inner factor pdiv G (layer output) IS the cotangent Item D renders — composing closes the loop
-- θⁿ = θ − lr·∂loss/∂θ. 3-axiom clean.
#print axioms conv_total_loss_grad_fold
#print axioms conv_bias_total_loss_grad_fold
#print axioms depthwise_total_loss_grad_fold
#print axioms depthwise_bias_total_loss_grad_fold
-- EfficientNet-B0 CLOSE (Item C) — another FREE close: every param family reuses an existing bridge.
-- The two new structures introduce no new PARAM-gradient bridge: batch-norm γ/β = per-channel BN γ/β
-- over the merged batch+spatial axis (bnBatchTensor4 = bnchwBack∘bnPerChannelFlat oc (N·h·w)∘bnchwFwd,
-- γ/β affine), and squeeze-excite's squeeze/excite are dense (→ M2 bridges). These pin the 5×5 depthwise
-- (the kernel no prior net used) and record the batch-norm-as-per-channel-at-N·h·w reuse. 3-axiom clean.
#print axioms enet_render_dw5W_certified
#print axioms enet_render_dw5b_certified
#print axioms enet_render_dw5W_strided_certified
#print axioms enet_render_dw5b_strided_certified
#print axioms enet_render_bngamma_certified
#print axioms enet_render_bnbeta_certified
-- ResNet-34 CLOSE (Item C) — a FREE close: every r34 param family certified by an existing bridge.
-- r34 uses only regular convs (3×3 + the 7×7 stem), per-channel BN, relu, maxpool, residual, dense —
-- no depthwise/relu6, and maxpool/relu/add/GAP carry no params. So NO new VJP; these six theorems pin
-- the generic strided/regular conv W/b bridges to r34's exact kernels, confirming the 7×7 stem and the
-- 3×3 strided projection (the shapes no prior net exercised) are covered. The per-channel BN γ/β
-- (cifar_bn_render_*) and dense (M2 weight/bias_grad_bridge) families are verbatim reuse, already
-- audited. 3-axiom clean by inheritance from the MobileNetV2/CNN/CIFAR-BN bridges.
#print axioms r34_render_stem_convW_certified
#print axioms r34_render_stem_convb_certified
#print axioms r34_render_blockConvW_certified
#print axioms r34_render_blockConvb_certified
#print axioms r34_render_downConvW_certified
#print axioms r34_render_downConvb_certified
-- ResNet-34 RENDER (Item A) — the PER-CHANNEL-BN typed SHlo forward graph matching the render
-- (StableHLO's resnetFwdGraph is scalar-bnF + representative; this is per-channel + full depth).
-- Per-block faithfulness (idBlockGraphPC_faithful / downBlockGraphPC_faithful: each basic block's
-- token tree denotes its per-channel rblkPC / rblkPStridedPC forward; the residual addV reuses the
-- block-input subtree) chains into the whole-net resnet34FwdGraphFullPC_faithful (7×7 stem → maxpool
-- → [3,4,6,3] blocks → GAP → dense, 146 params). The "text = render of a proven graph" forward half
-- at the render's per-channel BN; prerequisite for the r34 structured render (Item B).
#print axioms StableHLO.idBlockGraphPC_faithful
#print axioms StableHLO.downBlockGraphPC_faithful
#print axioms StableHLO.resnet34FwdGraphFullPC_faithful
-- ResNet-34 cotangent-chain CLOSE (Item D) — the CnnChainClose analogue: the Item C conv bridges
-- pinned to the cotangent the actual backward chain delivers. The chain through a basic block composes
-- the rendered backward denotations — the block-output relu mask (selectPos, if a>0), the per-channel
-- BN input-VJP (bnPerChannelTensor3_grad_input, = bnPerChannelBack's denotation), and the conv input-VJP
-- (conv2d_has_vjp3 via the flatten bridge, = convBack's denotation) — into idBlockCotC2/idBlockCotC1;
-- the downsample block adds the strided main/projection convs (idBlockCotC1 / idBlockCotC2 with γp), the
-- stem crosses the maxpool select_and_scatter (maxPoolBackFlat). Each conv θ output then denotes
-- θ − lr·(certified ∂conv/∂θ · the-actual-chain-cotangent). Pins the cotangent; the further = ∂loss/∂θ
-- fold (composing through all 16 blocks) stays separate, as for the CNN. 3-axiom clean.
#print axioms idBlock_render_convW2_chain_certified
#print axioms idBlock_render_convb2_chain_certified
#print axioms idBlock_render_convW1_chain_certified
#print axioms idBlock_render_convb1_chain_certified
#print axioms downBlock_render_convW1_chain_certified
#print axioms downBlock_render_convb1_chain_certified
#print axioms downBlock_render_convWp_chain_certified
#print axioms downBlock_render_convbp_chain_certified
#print axioms stem_render_convW_chain_certified
#print axioms stem_render_convb_chain_certified
-- EfficientNet-B0 RENDER (Item A) — the BATCHED typed SHlo forward graph matching the render.
-- EfficientNet's render emits TRUE batch-norm (reduce [0,2,3], batch-coupled), so unlike MNV2/r34
-- the graph lives at the batched index N·(c·h·w): every batch-separable op is `batchMap N` of the
-- proven per-example op (`SHlo.batchOp`/`BatchableOp`), the pointwise ops reuse their existing tokens,
-- and true batch-norm is `SHlo.bnBatchF` (= the proven `bnBatchTensor4`). Per-block faithfulness
-- (each block's token tree denotes its batched ℝ-forward; SE = `batchMap N seBlockFull`, residual =
-- `addV`) chains into the whole-net `efficientnetFwdGraphB_faithful`. The "text = render of a proven
-- forward graph" half for EfficientNet at the render's genuine (batch-coupled) BN flavor.
#print axioms StableHLO.stemGraphB_faithful
#print axioms StableHLO.mbNoExpGraphB_faithful
#print axioms StableHLO.mbStridedGraphB_faithful
#print axioms StableHLO.mbResidGraphB_faithful
#print axioms StableHLO.headGraphB_faithful
#print axioms StableHLO.efficientnetFwdGraphB_faithful
-- EfficientNet-B0 cotangent-chain CLOSE (Item D), batched backward primitives. The backward math at
-- the batched index: `batchMap_has_vjp` is the block-diagonal VJP lift — a batch-separable op's
-- gradient is the proven per-example VJP applied per example (reuses `rowwise_has_vjp_mat` +
-- `hasVJPMat_to_hasVJP`); this is `seBlockFull_has_vjp` / the conv-depthwise-dense VJPs lifted to the
-- batch. `bnBatchLA_has_vjp` is the one batch-coupled op — the proven `bnBatchTensor4` VJP reindex-
-- conjugated to the network's flat index. (`reindex_has_vjp` is a reusable generic reindex VJP.)
#print axioms batchMap_has_vjp
#print axioms batchMap_differentiable
#print axioms reindex_has_vjp
#print axioms bnBatchLA_has_vjp
#print axioms bnBatchLA_differentiable
-- Per-block batched gradients (the per-block VJP, the user-requested deliverable): each MBConv block
-- type + head, composing the batched stage VJPs via `vjp_comp` (SE = `batchMap N seBlockFull`, true-BN
-- = `bnBatchLA`, swish pointwise, residual via `residual_has_vjp`). Then the whole-subnet capstone —
-- the batched, true-batch-norm + SE analogue of `efficientnet_has_vjp`. Both forward (Item A) and
-- backward now proven for the block decomposition we scale.
#print axioms mbNoExpFwdB_has_vjp
#print axioms mbStridedFwdB_has_vjp
#print axioms mbResidFwdB_has_vjp
#print axioms headFwdB_has_vjp
#print axioms efficientnetForwardB_has_vjp
-- FULL EfficientNet-B0 (all 16 MBConv blocks, real [t,c,n,s,k] spec): the batched forward
-- graph at full depth denotes the full ℝ-forward. Scales the representative via the generic
-- per-block graph/faithful machinery (+ the 4th block shape mbExp: expand+stride1+no-residual).
#print axioms StableHLO.mbExpGraphB_faithful
#print axioms StableHLO.efficientnetFwdGraphB_full_faithful
#print axioms efficientnetForwardB_full_has_vjp
-- ConvNeXt RENDER (planning/convnext_close.md Item A) — the representative 2-block forward graph.
-- The DELIBERATE CONTRAST to EfficientNet: ConvNeXt's normalization is LayerNorm, which is per-example
-- separable, so the graph lives at a plain batch-1 index (no batched token layer, no `batchMap`/`bnBatchF`).
-- The representative `convNextFwdGraph` (stem 1×1 patchify → scalar-LN → block×2 → GAP → head-LN → dense;
-- tokens flatConvF×5, depthwiseF×2 [7×7], bnF×4 [scalar LN = bnForward over c·h·w], geluF×2, layerScaleF×2,
-- gapF, denseF, addV×2 residual) denotes the proven `convNextForward` (`convnext_has_vjp`, audited above).
-- Scalar LN matches the operational render reducing dim `[1]` per example — faithful at batch-1, as for
-- MNV2/r34. The "text = render of a proven forward graph" forward half (Item A) for ConvNeXt.
#print axioms StableHLO.convNextFwdGraph_faithful
-- ConvNeXt CLOSE (planning/convnext_close.md Item C) — mostly reuse, two genuinely-new param families.
-- The 1×1 convs (stem/expand/project) and dense head reuse M3/M2 verbatim; the 7×7 depthwise (the kernel
-- size no prior net used; stride-1 — ConvNeXt blocks keep resolution) pins the generic depthwise bridges.
-- New: layer-scale γ — layerScale is symmetric in (γ,x), so the param Jacobian is the diagonal x_iδ_ij
-- (pdiv_layerScale_gamma, the mirror of pdiv_layerScale), giving the rendered dγ = x⊙dy. And scalar-LN
-- γ/β — layerNormForward has SCALAR γ,β (BatchNorm.lean left bn_grad_gamma/beta as definitions-only:
-- "scalar params don't fit the pdiv framework"); the Vec-1 embedding brings them inside (LN affine in
-- the params, the CifarBnClose recipe with the constant channel map Fin n → Fin 1), certifying the
-- rendered whole-n reduces dγ = Σ dy·x̂, dβ = Σ dy. Affine ⇒ no 0<ε. 3-axiom clean.
#print axioms cnx_render_dw7W_certified
#print axioms cnx_render_dw7b_certified
#print axioms pdiv_layerScale_gamma
#print axioms layerScale_gamma_grad_bridge
#print axioms cnx_render_lsgamma_certified
#print axioms cnx_lnGamma_grad_bridge
#print axioms cnx_lnBeta_grad_bridge
#print axioms cnx_render_lngamma_certified
#print axioms cnx_render_lnbeta_certified
-- ConvNeXt cotangent-chain CLOSE (planning/convnext_close.md Item D) — the MobileNetV2ChainClose/
-- ResNet34ChainClose analogue: the Item C bridges pinned to the cotangent the ACTUAL backward chain
-- delivers through a ConvNeXt block. The chain composes the rendered backward denotations —
-- layer-scale back (= layerScale γls on the cotangent, the symmetric-diagonal trick the Item B render
-- uses), project/expand 1×1 conv-back (conv2d_has_vjp3), the GELU mask (geluScalarDeriv at the saved
-- pre-GELU activation), the scalar-LN input-VJP (bn_grad_input = bnBack's denotation), the 7×7
-- depthwise-back — with the residual addV passing dyOut straight through (no post-add activation,
-- no stride split: one set of cotangents covers every block). Unlike MNV2/r34, the ConvNeXt-signature
-- param families are pinned too: layer-scale γ (cotangent = dyOut, the exact passthrough) and the
-- block scalar-LN γ/β at cnxCotN (through ls-back → proj-back → GELU mask → exp-back). Batch-1,
-- pure-Lean — no batched-VJP machinery (the EfficientNet contrast). 3-axiom clean.
#print axioms cnx_render_lsgamma_chain_certified
#print axioms cnx_render_projW_chain_certified
#print axioms cnx_render_projb_chain_certified
#print axioms cnx_render_expW_chain_certified
#print axioms cnx_render_expb_chain_certified
#print axioms cnx_render_lngamma_chain_certified
#print axioms cnx_render_lnbeta_chain_certified
#print axioms cnx_render_dw7W_chain_certified
#print axioms cnx_render_dw7b_chain_certified
#print axioms cnx_stem_render_convW_chain_certified
#print axioms cnx_stem_render_convb_chain_certified
-- ViT RENDER (planning/vit_close.md Item A) — the representative distinct-param 2-block ViT.
-- The proven transformerTower/vit_full share ONE param tuple across blocks; the close needs distinct
-- per-block params, so `vitForward2` (patchEmbed → block₁ → block₂ → final per-token LN → CLS slice →
-- dense head) composes `transformerBlock_has_vjp_mat` twice with the patch-embed/final-LN/classifier
-- witnesses — an UNCONDITIONAL whole-net VJP (only 0 < ε; softmax/GELU/LN are kink-free), joining
-- vit_full/convnext. The forward graph `vitFwdGraph` spells the block at heads = 1 over the ch10 token
-- vocabulary (patchEmbedF, lnRowF, denseRowF ×6, matmulF ×2 [Q·Kᵀ, P·V], transposeF, scaleF [1/√d],
-- softmaxRowF, geluF, addV ×2 residual, clsSliceF, denseF head); `vitFwdGraph_faithful` proves
-- den vitFwdGraph = vitForward2 at heads = 1 — the per-head slice/concat plumbing collapses via
-- `mhsa_layer_one_head` (SDPA = three matmuls + a row-softmax). The attention analogue of
-- `convNextFwdGraph_faithful`: "text = render of a proven forward graph", Item A for ch10.
#print axioms vitForward2_has_vjp
#print axioms vitForward2_has_vjp_correct
#print axioms mhsa_layer_one_head
#print axioms StableHLO.vitFwdGraph_faithful
-- ViT CLOSE (planning/vit_close.md Item C) — the param close, two genuinely-new bridge families.
-- Per-token dense W/b (the M2 outer-product bridge row-lifted): every row of [N,a] through the same
-- W:[a,c], so dW = Σ_tokens xᵣ⊗dyᵣ (one dot_general contracting the token axis) and db = Σ_tokens dyᵣ
-- are the certified Jacobian contractions — covers Wq/Wk/Wv/Wo, Wfc1/Wfc2 + biases at every block.
-- Row-lifted scalar-LN γ/β (the ConvNeXtClose Vec-1 embedding over N token rows; affine in the params
-- ⇒ no 0<ε): dγ = Σ_r Σ_k dY·x̂ᵣ, dβ = Σ Σ dY — covers all five LN sites. pos_embed: the Jacobian of
-- patchEmbed_flat in pos is the IDENTITY (broadcast-add) ⇒ dPos = dy. cls_token: a row-0 masked
-- gather ⇒ dCls = the row-0 slice of the embed cotangent (clsSliceF's shape). The classifier head is
-- verbatim M2 weight/bias_grad_bridge reuse (audited above). Patch conv Wp/bp: patchEmbed_flat is
-- LINEAR in the kernel with CONSTANT pad-guarded image-read coefficients (the mirror of the
-- input-grad case — no pad-eval calculus), so dWp = Σ_patches read·dy_(p+1,·) (the dilate-dy/valid-
-- conv form, CLS row excluded) and dbp = Σ_patches dy_(p+1,·) are the certified contractions.
-- EVERY representative-ViT train-step param family is now certified.
#print axioms pdiv_rowDense_W
#print axioms vit_rowDenseW_grad_bridge
#print axioms vit_rowDenseb_grad_bridge
#print axioms vit_render_rowdenseW_certified
#print axioms vit_render_rowdenseb_certified
#print axioms pdiv_rowLN_gamma
#print axioms pdiv_rowLN_beta
#print axioms vit_rowlnGamma_grad_bridge
#print axioms vit_rowlnBeta_grad_bridge
#print axioms vit_render_rowlngamma_certified
#print axioms vit_render_rowlnbeta_certified
#print axioms pdiv_patchEmbed_pos
#print axioms vit_render_pos_certified
#print axioms pdiv_patchEmbed_cls
#print axioms vit_render_cls_certified
#print axioms pdiv_patchEmbed_W
#print axioms vit_patchW_grad_bridge
#print axioms vit_render_patchW_certified
#print axioms pdiv_patchEmbed_b
#print axioms vit_patchb_grad_bridge
#print axioms vit_render_patchb_certified
-- ViT cotangent-chain CLOSE (planning/vit_close.md Item D) — the ConvNeXtChainClose analogue: the
-- Item C bridges pinned to the cotangent the ACTUAL backward chain delivers through the attention
-- block. The chain composes the rendered backward denotations — per-token dense input-VJPs
-- (rowDenseBackFlat), the GELU mask, the rowwise scalar-LN input-VJP (rowLNBackFlat = rowwise
-- bn_grad_input), the row-softmax backward at the saved pre-softmax scores, and the SDPA matmuls
-- spelled with the forward matMulFlat/transposeFlat on cotangents. THE SUBSTANTIVE TIES
-- (vitCotD{Q,K,V}_eq_sdpa_back_{Q,K,V}): at the pinned saved activations the matmul-spelled chain
-- segments ARE the proven closed forms sdpa_back_{Q,K,V} — the rendered attention backward is
-- pinned to the audited SDPA suite. New structural wrinkle vs all prior nets: the THREE-WAY fan-in
-- at LN1's output (the Q/K/V dense-backs sum, vitCotLn1). Residual fan-ins at both sublayers
-- (vitCotH, vitCotXin); classifier-back scattered to row 0 (vitCotFl); the embed params (pos, cls,
-- patch W/b) pinned at the block-1 input cotangent. Batch-1, pure-Lean. 3-axiom clean.
#print axioms vitCotDP_eq_sdpa_dWeights
#print axioms vitCotDS_eq_sdpa_dScaled
#print axioms vitCotDQ_eq_sdpa_back_Q
#print axioms vitCotDK_eq_sdpa_back_K
#print axioms vitCotDV_eq_sdpa_back_V
#print axioms vit_render_Wfc2_chain_certified
#print axioms vit_render_bfc2_chain_certified
#print axioms vit_render_Wfc1_chain_certified
#print axioms vit_render_bfc1_chain_certified
#print axioms vit_render_ln2gamma_chain_certified
#print axioms vit_render_ln2beta_chain_certified
#print axioms vit_render_Wo_chain_certified
#print axioms vit_render_bo_chain_certified
#print axioms vit_render_Wq_chain_certified
#print axioms vit_render_Wk_chain_certified
#print axioms vit_render_Wv_chain_certified
#print axioms vit_render_ln1gamma_chain_certified
#print axioms vit_render_ln1beta_chain_certified
#print axioms vit_render_lnFgamma_chain_certified
#print axioms vit_render_pos_chain_certified
#print axioms vit_render_cls_chain_certified
#print axioms vit_render_patchW_chain_certified
#print axioms vit_render_patchb_chain_certified
-- ViT SCALING PASS: vector-[D] LayerNorm (planning/vit_close.md scaling item 1) — the close lifted
-- from the proof's scalar LN gamma/beta to the committed production render's per-channel vector
-- form (ViTRender: scalar-LN(1,0) followed by per-channel scale + bias). layerNormVec's VJP
-- composes layerNorm_has_vjp(1,0) + layerScale_has_vjp + the bias translation; the vector-LN
-- sublayers/block re-run the biPathMat/vjpMat_comp recipe (transformerBlockV_has_vjp_mat);
-- vitForward2V is the distinct-param 2-block net at vector LN with an UNCONDITIONAL whole-net VJP
-- (only 0 < eps). The graph spells each LN site lnRowF(1,0) -> rowScaleF gamma -> rowBiasF beta
-- (two new broadcast tokens, 9-site lockstep; rowScaleF is its own input-VJP, rowBiasF passes the
-- cotangent through) and vitFwdGraphV_faithful proves den = vitForward2V at heads = 1. The
-- per-channel param grads d-gamma_k = Sum_tokens dy*xhat (KEEPING the channel axis — ViTRender's
-- reduce) and d-beta_k = Sum_tokens dy are certified via the masked-gather Jacobian recipe.
#print axioms layerNormVec_has_vjp
#print axioms transformerBlockV_has_vjp_mat
#print axioms vitForward2V_has_vjp
#print axioms vitForward2V_has_vjp_correct
#print axioms StableHLO.vitFwdGraphV_faithful
#print axioms pdiv_vecLN_gamma
#print axioms pdiv_vecLN_beta
#print axioms vit_veclnGamma_grad_bridge
#print axioms vit_veclnBeta_grad_bridge
#print axioms vit_render_veclngamma_certified
#print axioms vit_render_veclnbeta_certified
-- Vector-LN render upgrade + chain pins: TestViTTrainPC.lean now renders the production ViTRender
-- LN form (each site lnRowF(1,0) -> rowScaleF -> rowBiasF; backward = rowScaleF on the cotangent
-- then lnRowBack at gamma=1; per-channel dgamma/dbeta reduces off the SAVED normalize output) —
-- iree-compile OK + gfx1100 ref-only smoke 40/40. The vector-LN chain cots decompose each LN
-- input-VJP the same way (vitCot{H,Att,Xin,B2out}V); the dense/SDPA chain segments and ties are
-- LN-form-agnostic and hold verbatim. The per-channel gamma/beta pins at the actual chain
-- cotangents (the three-way fan-in at LN1, the MLP chain at LN2, the row-0 scatter at the final LN):
#print axioms vit_render_vecln1gamma_chain_certified
#print axioms vit_render_vecln1beta_chain_certified
#print axioms vit_render_vecln2gamma_chain_certified
#print axioms vit_render_vecln2beta_chain_certified
#print axioms vit_render_veclnFgamma_chain_certified
#print axioms vit_render_veclnFbeta_chain_certified
