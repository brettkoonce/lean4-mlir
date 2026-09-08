import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Training.JacobianSeal
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Architectures.Residual
import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.SE
import LeanMlir.Proofs.Architectures.LayerNorm
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Architectures.MobileNetV2
import LeanMlir.Proofs.Training.MobileNetV2JacobianSeal
import LeanMlir.Proofs.Architectures.ConvNeXt
import LeanMlir.Proofs.Architectures.EfficientNet
import LeanMlir.Proofs.Architectures.MnistCNN
import LeanMlir.Proofs.Architectures.CifarCNN
import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Codegen.StableHLOParse
import LeanMlir.Proofs.Codegen.StableHLOLex
import LeanMlir.Proofs.Foundation.StridedConv
import LeanMlir.Proofs.Foundation.ResNet34
import LeanMlir.Proofs.Foundation.ResNet34LivePC
import LeanMlir.Proofs.Training.ResNet34LiveSeal
import LeanMlir.Proofs.Foundation.ResNet34LiveFull
import LeanMlir.Proofs.Training.MobileNetV2JacobianSealFull
import LeanMlir.Proofs.Foundation.ResNet34LiveRealistic
import LeanMlir.Proofs.Training.ResNet34LiveRealisticSeal
import LeanMlir.Proofs.Foundation.ResNet34LiveGeneric
import LeanMlir.Proofs.Training.MobileNetV2SealRealistic
import LeanMlir.Proofs.Foundation.PerChannelBN
import LeanMlir.Proofs.Foundation.LinearTrainStep
import LeanMlir.Proofs.Foundation.MlpTrainStep
import LeanMlir.Proofs.Foundation.CnnTrainStep
import LeanMlir.Proofs.Architectures.CifarBnClose
import LeanMlir.Proofs.Foundation.CnnChainClose
import LeanMlir.Proofs.Architectures.Cifar8Close
import LeanMlir.Proofs.Architectures.MobileNetV2Close
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC
import LeanMlir.Proofs.Architectures.MobileNetV2ChainClose
import LeanMlir.Proofs.Foundation.ConvLossFold
import LeanMlir.Proofs.Architectures.EfficientNetClose
import LeanMlir.Proofs.Codegen.EfficientNetRenderPC
import LeanMlir.Proofs.Architectures.EfficientNetChainClose
import LeanMlir.Proofs.Architectures.EfficientNetFullB0
import LeanMlir.Proofs.Architectures.EfficientNetFullB0Eval
import LeanMlir.Proofs.Foundation.ResNet34Close
import LeanMlir.Proofs.Codegen.ResNet34RenderPC
import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoC
import LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoC
import LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoCPaper
import LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoC
import LeanMlir.Proofs.Architectures.EfficientNetTiePoC
import LeanMlir.Proofs.Architectures.ConvNeXtClose
import LeanMlir.Proofs.Architectures.ConvNeXtChainClose
import LeanMlir.Proofs.Architectures.ViTFwdGraph
import LeanMlir.Proofs.Architectures.ViTClose
import LeanMlir.Proofs.Architectures.ViTChainClose
import LeanMlir.Proofs.Architectures.ViTVecLN
import LeanMlir.Proofs.Architectures.ViTMultiHead
import LeanMlir.Proofs.Architectures.ViTMultiHeadChain
import LeanMlir.Proofs.Architectures.ViTDepthK
import LeanMlir.Proofs.Architectures.MobileNetV2FullPaper
import LeanMlir.Proofs.Architectures.MobileNetV2FullVJP
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Float.FloatSubnormalBridge
import LeanMlir.Proofs.Training.SgdDescent
import LeanMlir.Proofs.Training.SgdDescentLinear
import LeanMlir.Proofs.Training.SgdDescentCnn
import LeanMlir.Proofs.Training.SgdDescentCifar
import LeanMlir.Proofs.Float.BnFloatBridge
import LeanMlir.Proofs.Float.Resnet34FloatBridge
import LeanMlir.Proofs.Codegen.BnInputBridge
import LeanMlir.Proofs.Codegen.Resnet34BlockBridge
import LeanMlir.Proofs.Float.FloatComposeBridge
import LeanMlir.Proofs.Float.ConvMixedComposeBridge
import LeanMlir.Proofs.Float.DepthwiseMixedFloatBridge
import LeanMlir.Proofs.Float.DepthwiseFloatBridge
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPCEval
import LeanMlir.Proofs.Architectures.MobileNetV2FullPaperEval
import LeanMlir.Proofs.Codegen.EfficientNetRenderPCEval
import LeanMlir.Proofs.Foundation.BatchMapVJPAt
import LeanMlir.Proofs.Architectures.ResNet34FullB
import LeanMlir.Proofs.Architectures.ResNet34FullBVJP
import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoCB
import LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoCG
import LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCG
import LeanMlir.Proofs.Architectures.ViTFaithfulPoCG
import LeanMlir.Proofs.Architectures.ViTFaithfulPoCGB
import LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCGB
import LeanMlir.Proofs.Foundation.Bf16GradNodes
import LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoCPaperG
import LeanMlir.Proofs.Architectures.MobileNetV2FullB
import LeanMlir.Proofs.Architectures.MobileNetV2FullBVJP
import LeanMlir.Proofs.Foundation.MobileNetV2TiePoCB
import LeanMlir.Proofs.Architectures.EfficientNetTiePoCG
import LeanMlir.Proofs.Architectures.ConvNeXtTiePoCGB
import LeanMlir.Proofs.Architectures.ViTTiePoCGB
import LeanMlir.Proofs.Foundation.DataParallel
import LeanMlir.Proofs.Foundation.DataParallelNode
import LeanMlir.Proofs.Codegen.LambTriple
import LeanMlir.Proofs.Foundation.BceLossCot
import LeanMlir.Proofs.Architectures.ResNet50FullBVJP
import LeanMlir.Proofs.Foundation.ResNet50TiePoCB
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Foundation.ResNet34TiePoCB
import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Architectures.ChannelLNBack
import LeanMlir.Proofs.Foundation.ResNetBackChains
import LeanMlir.Proofs.Foundation.MobileNetBackChains
import LeanMlir.Proofs.Foundation.EfficientNetBackChains
import LeanMlir.Proofs.Foundation.ConvNeXtBackChains
import LeanMlir.Proofs.Foundation.ViTBackChains
import LeanMlir.Proofs.Float.LinBackFloatBridge
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie
import LeanMlir.Proofs.Foundation.MobileNetV2WholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.MobileNetV2PaperWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.EfficientNetWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.EfficientNetFullWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTieB
import LeanMlir.Proofs.Foundation.MobileNetV2WholeBackCertifiedTieB
import LeanMlir.Proofs.Foundation.Resnet50WholeBackCertifiedTieB
import LeanMlir.Proofs.Foundation.EvenKernelConvBack
import LeanMlir.Proofs.Foundation.ConvNeXtWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.ViTWholeBackCertifiedTie
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Architectures.ConvNeXtBackCertifiedTie
import LeanMlir.Proofs.Architectures.MobileNetV2BackCertifiedTie
import LeanMlir.Proofs.Architectures.EfficientNetBackCertifiedTie
import LeanMlir.Proofs.Architectures.ViTMhsaBackCertifiedTie
import LeanMlir.Proofs.Training.SgdDescentMlp
import LeanMlir.Proofs.Codegen.AdamStep
import LeanMlir.Proofs.Codegen.AdamRender
import LeanMlir.Proofs.Architectures.EfficientNetBackB0
import LeanMlir.Proofs.Architectures.MobileNetV2BackB0
import LeanMlir.Proofs.Foundation.ResNet34BackB0
import LeanMlir.Proofs.Architectures.ConvNeXtBackB0
import LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoC
import LeanMlir.Proofs.Architectures.ConvNeXtTiePoC
import LeanMlir.Proofs.Architectures.ViTBackB0
import LeanMlir.Proofs.Foundation.ViTBackNet
import LeanMlir.Proofs.Foundation.ResNet50BackNet
import LeanMlir.Proofs.Foundation.BackNetFolds
import LeanMlir.Proofs.Foundation.MobileNetV4BackB0
import LeanMlir.Proofs.Architectures.MobileNetV4FullBVJP
import LeanMlir.Proofs.Foundation.MobileNetV4FaithfulPoCB
import LeanMlir.Proofs.Foundation.MobileNetV4TiePoCB
import LeanMlir.Proofs.Foundation.MobileNetV4WholeBackCertifiedTieB
import LeanMlir.Proofs.Foundation.EfficientNetBackNet
import LeanMlir.Proofs.Foundation.LinearFaithfulPoC
import LeanMlir.Proofs.Float.E4M3FaithfulPoC
import LeanMlir.Proofs.Foundation.MlpFaithfulPoC
import LeanMlir.Proofs.Foundation.CnnFaithfulPoC
import LeanMlir.Proofs.Architectures.CifarFaithfulPoC
import LeanMlir.Proofs.Architectures.CifarBnFaithfulPoC
import LeanMlir.Proofs.Architectures.CifarBnTiePoC
import LeanMlir.Proofs.Architectures.Cifar8FaithfulPoC
import LeanMlir.Proofs.Architectures.Cifar8TiePoC
import LeanMlir.Proofs.Architectures.Cifar8BnTiePoC
import LeanMlir.Proofs.Architectures.ViTFaithfulPoC
import LeanMlir.Proofs.Architectures.ViTTiePoC
import LeanMlir.Proofs.Certificates.LipschitzCert
import LeanMlir.Proofs.Certificates.SmoothingGaussian
import LeanMlir.Proofs.Certificates.LipschitzCertInstance
import LeanMlir.Proofs.Training.TrainedMlpWitness
import LeanMlir.Proofs.Training.TrainedCnnWitness
import LeanMlir.Proofs.Training.TrainedCnnSeal
import LeanMlir.Proofs.Certificates.LipschitzCertScorecard
import LeanMlir.Proofs.Certificates.LipschitzCertPairSDP
import LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDP
import LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDPUncon
import LeanMlir.Proofs.Certificates.LipschitzCertFloat
import LeanMlir.Proofs.Foundation.ListDot
import LeanMlir.Proofs.Foundation.IntervalBound
import LeanMlir.Proofs.Foundation.IntervalBoundConv
import LeanMlir.Proofs.Foundation.CrownBound
import LeanMlir.Proofs.Certificates.SmoothingMC
import LeanMlir.Proofs.Certificates.SmoothingCP
import LeanMlir.Proofs.Certificates.SmoothingCPScorecard
import LeanMlir.Proofs.Certificates.SmoothingPhiBounds
import LeanMlir.Proofs.Certificates.SmoothingDecScorecard
import LeanMlir.Proofs.Certificates.SmoothingNetSemantics
import LeanMlir.Proofs.Certificates.SmoothingNetWitness
import LeanMlir.Proofs.Foundation.UpstreamDraft
import LeanMlir.Proofs.Float.Binary32Instance
import LeanMlir.Proofs.Training.TrainedLinearDescent
import LeanMlir.Proofs.Foundation.MuonGeometry
import LeanMlir.Proofs.Foundation.MuonNewtonSchulz
import LeanMlir.Proofs.Foundation.SpecVJP
import LeanMlir.Proofs.Foundation.MlpCanonical

open Proofs

-- One `#print axioms` per certified theorem; CI (certs.yml) fails on any axiom beyond
-- propext / Classical.choice / Quot.sound. The narrative each section carried — what the tier
-- found, when, and why — is planning/archive/audit_axioms_log.md, under the same headers.

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

-- Nonzero-Jacobian seal (JacobianSeal.lean, planning/archive/whole_network_backward.md Item B)
#print axioms HasVJP.backward_ne_zero_of_pdiv_ne
#print axioms sum_smul_basisVec
#print axioms fderiv_eq_zero_of_pdiv_all_zero
#print axioms exists_pdiv_ne_of_fderiv_ne
#print axioms HasVJP.backward_nontrivial_of_fderiv_ne
#print axioms mnistLinear_backward_nontrivial
-- The pointwise (HasVJPAt) seal variants — the kinked witnesses are HasVJPAt, not HasVJP.
#print axioms HasVJPAt.backward_ne_zero_of_pdiv_ne
#print axioms HasVJPAt.backward_nontrivial_of_fderiv_ne
-- Item B2 discharged at the live MobileNetV2 witness (MobileNetV2JacobianSeal.lean)
#print axioms Mnv2Live.mnv2Live_jacobian_nonzero
#print axioms Mnv2Live.mnv2Live_backward_nontrivial

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

-- HasVJPAt pointwise framework (E.5)
#print axioms relu_has_vjp_at
#print axioms mlp_has_vjp_at
#print axioms mnistLinear_has_vjp_correct
#print axioms maxPool2_has_vjp_at3

-- Capstone: end-to-end ResNet-style CNN whole-network VJP + the global-average-pool VJP it depends on
#print axioms globalAvgPoolFlat_has_vjp
#print axioms globalAvgPoolFlat_has_vjp_correct
#print axioms cnn_has_vjp_at
#print axioms cnn_has_vjp_at_correct

-- Whole-network VJPs for the depthwise/SE/LN-based architectures
#print axioms relu6_has_vjp_at
#print axioms mobilenetv2_has_vjp_at_correct
-- MobileNetV2: concrete whole-network instance, every ReLU6 smoothness hypothesis discharged
#print axioms MobileNetV2Concrete.mnv2Concrete_has_vjp_correct
-- MobileNetV2: the LIVE counterpart
#print axioms Mnv2Live.bn13_window
#print axioms Mnv2Live.mnv2Live_has_vjp_correct
-- ...and the live witness is non-degenerate
#print axioms Mnv2Live.chSum_convX
#print axioms Mnv2Live.mnv2Live_forward_nonconstant
#print axioms layerScale_has_vjp_correct
#print axioms convnext_has_vjp_at_correct
-- ConvNeXt promoted to an UNCONDITIONAL global VJP (all-smooth ops)
#print axioms convnext_has_vjp
#print axioms convnext_has_vjp_correct
#print axioms sigmoid_has_vjp
#print axioms efficientnet_has_vjp_at_correct
-- EfficientNet promoted to an UNCONDITIONAL global VJP (all-smooth ops)
#print axioms efficientnet_has_vjp
#print axioms efficientnet_has_vjp_correct

-- Chapter-4 MNIST 2D CNN (no BN)
#print axioms mnistCnnNoBn_has_vjp_at_correct
#print axioms Micro.mnistMicroCnn_has_vjp_correct
-- Tier-1 discharged instance
#print axioms maxPool2Smooth_of_injective
#print axioms Mini.miniCnn_has_vjp_correct
-- Tier-2: same CNN with genuine 3×3 SAME-padding convolutions
#print axioms conv2d_center3x3
#print axioms Spatial.spatialCnn_has_vjp_correct
-- Chapter-3 MLP: concrete whole-network instance, every ReLU smoothness hypothesis discharged
#print axioms MlpConcrete.mlpConcrete_has_vjp_correct
-- ResNet-style CNN *with* BN
#print axioms CnnConcrete.cnnConcrete_has_vjp_correct

-- Denoted StableHLO-subset IR (Phase 0a/0b spike, planning/archive/typed_ir.md)
#print axioms IR.dense_back_bridge
#print axioms IR.relu_back_bridge
-- Phase 2: the emitted transposed-convolution graph denotes the proven conv input-VJP
#print axioms IR.conv_back_bridge_1to2
#print axioms IR.conv_back_bridge_2to2
-- The GENERAL conv-adjoint reindex (all dims, odd kernels)
#print axioms IR.convBackDenote_eq_input_grad_formula
-- Phase 2: the emitted tile-compare-select graph denotes the canonical maxpool backward
#print axioms IR.maxpool_back_bridge
-- Phase 1 smooth activations
#print axioms IR.gelu_back_bridge
#print axioms IR.swish_back_bridge
#print axioms IR.sigmoid_back_bridge
-- BatchNorm: the emitted reduce+broadcast+elementwise graph denotes the proven 3-term backward
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
-- Final assembly: the emitted whole-MLP backward graph denotes the proven whole-network VJP
#print axioms IR.mlp_whole_bridge
-- Parameter gradients (train-step pieces)
#print axioms IR.weight_grad_bridge
#print axioms IR.bias_grad_bridge
#print axioms IR.mlp_layer1_weight_grad_bridge
-- Forward IR (Phase 2)
#print axioms IR.denote_subst_fwd
#print axioms IR.mlp_fwd_bridge
#print axioms IR.mlp_fwd_preact0
#print axioms IR.mlp_fwd_preact1
-- Loss cotangent (rest of Phase 4)
#print axioms IR.lossCot_bridge

-- R4 printer-faithfulness, Stage A (Chapter 2)
#print axioms StableHLO.fwdGraph_faithful
#print axioms StableHLO.backGraph_faithful
#print axioms StableHLO.softmaxDiv_expe_faithful
#print axioms StableHLO.lossCotGraph_faithful
#print axioms StableHLO.lossCotGraph_isCEgrad
#print axioms StableHLO.wGrad_isWeightJacobian
#print axioms StableHLO.bGrad_isBiasJacobian
-- SGD update proven (not trusted) for plain SGD on the linear net.
#print axioms StableHLO.sgdW_isCertifiedGradStep
#print axioms StableHLO.sgdB_isCertifiedGradStep
-- M1: the linear SGD step bundled to the certified closed-form softmax-CE gradient
#print axioms StableHLO.lossCot_eq_softmax_sub_onehot
#print axioms StableHLO.sgdW_descends_softmaxCE_grad
#print axioms StableHLO.sgdB_descends_softmaxCE_grad
-- M1 chain-rule fold: the SGD step is literally θ − lr·∂Loss/∂θ.
#print axioms StableHLO.crossEntropy_differentiable
#print axioms StableHLO.denseWeightMap_differentiable
#print axioms StableHLO.lossWeightGrad_eq_sum
#print axioms StableHLO.sgdW_descends_loss_gradient
-- M1 rendering half
#print axioms StableHLO.linWeightDen_is_loss_descent
#print axioms StableHLO.linBiasDen_is_certified
-- PoC capstones (LinearFaithfulPoC.lean)
#print axioms LinPoC.poc_fwd_faithful
#print axioms LinPoC.poc_fwd_is_render
#print axioms LinPoC.poc_train_step_certified
-- Tail fold closed
#print axioms LinPoC.poc_weightSgd_den_eq
#print axioms LinPoC.poc_biasSgd_den_eq
#print axioms LinPoC.poc_train_step_tail_certified
-- mnist-MLP fully folded
#print axioms MlpPoC.cot1_den
#print axioms MlpPoC.cot0_den
#print axioms MlpPoC.W2_den_certified
#print axioms MlpPoC.W1_den_certified
#print axioms MlpPoC.W0_den_certified
#print axioms MlpPoC.b2_den_certified
#print axioms MlpPoC.b1_den_certified
#print axioms MlpPoC.b0_den_certified
-- mnist-mlp FULLY TIED
#print axioms MlpPoC.mlpLossCot_den
#print axioms MlpPoC.mlp_W2_tied_totalloss
#print axioms MlpPoC.mlp_train_step_tied_certified
-- mnist-CNN fully folded
#print axioms CnnPoC.cW1_den
#print axioms CnnPoC.cb1_den
#print axioms CnnPoC.cW2_den
#print axioms CnnPoC.cb2_den
#print axioms CnnPoC.dW3_den
#print axioms CnnPoC.db3_den
#print axioms CnnPoC.dW4_den
#print axioms CnnPoC.db4_den
#print axioms CnnPoC.dW5_den
#print axioms CnnPoC.db5_den
-- mnist-cnn dense-head TIE
#print axioms CnnPoC.cnnLossCot_den
#print axioms CnnPoC.cnn_W5_tied_totalloss
-- mnist-cnn CONV fold
#print axioms CnnPoC.cnn_conv_tied_certified
-- ch5-CIFAR fully folded (no-BN, 2-scale)
#print axioms CifarPoC.convW_den
#print axioms CifarPoC.convB_den
#print axioms CifarPoC.dW5_den
#print axioms CifarPoC.db5_den
#print axioms CifarPoC.dW6_den
#print axioms CifarPoC.db6_den
#print axioms CifarPoC.dW7_den
#print axioms CifarPoC.db7_den
-- ch5-CIFAR §1a TIE
#print axioms CifarPoC.cifarLossCot_den
#print axioms CifarPoC.cifar_W7_tied_totalloss
#print axioms CifarPoC.cifar_conv_tied_certified
-- ch5-CIFAR-BN fully folded
#print axioms CifarBnPoC.bnGamma_den
#print axioms CifarBnPoC.bnBeta_den
-- ch5-CIFAR-BN §1a TIE
#print axioms CifarBnPoC.cifarBnLossCot_den
#print axioms CifarBnPoC.cifarBn_W7_tied_totalloss
#print axioms CifarBnPoC.cifarBn_convbn_tied_certified
-- deeper 8-conv cifar8 fully folded
#print axioms Cifar8PoC.denseW_den
#print axioms Cifar8PoC.denseB_den
-- ch5-cifar8 §1a TIE
#print axioms Cifar8PoC.cifar8LossCot_den
#print axioms Cifar8PoC.cifar8_Wb_tied_totalloss
#print axioms Cifar8PoC.cifar8_convs_tied_certified
-- ch5-cifar8-bn §1a TIE
#print axioms Cifar8BnPoC.cifar8BnLossCot_den
#print axioms Cifar8BnPoC.cifar8Bn_convbn_tied_certified
-- ch6-ResNet-34 fully folded (full [3,4,6,3], 146 params)
#print axioms ResNet34PoC.convStridedW_den
#print axioms ResNet34PoC.convStridedB_den
-- ch7-MobileNetV2 §1 fold (depthwise half)
#print axioms Mnv2PoC.depthwiseW_den
#print axioms Mnv2PoC.depthwiseB_den
#print axioms Mnv2PoC.depthwiseStridedW_den
#print axioms Mnv2PoC.depthwiseStridedB_den
-- The XLA-SAME per-example stem dens (2026-09-05)
#print axioms Mnv2PoC.convStridedXlaW_den
#print axioms Mnv2PoC.convStridedXlaB_den
-- ch7-MobileNetV2 FULL 17-block paper §1 fold (den)
#print axioms Mnv2PaperPoC.mnv2StemParamsCertified
#print axioms Mnv2PaperPoC.mnv2NoExpParamsCertified
#print axioms Mnv2PaperPoC.mnv2Stride1ParamsCertified
#print axioms Mnv2PaperPoC.mnv2Stride2ParamsCertified
#print axioms Mnv2PaperPoC.mnv2HeadParamsCertified
#print axioms Mnv2PaperPoC.mnv2DenseParamsCertified
-- ch8-EfficientNet-B0 §1 fold (den)
#print axioms EnetPoC.convWB_den
#print axioms EnetPoC.convStridedWB_den
#print axioms EnetPoC.denseWB_den
#print axioms EnetPoC.denseBB_den
#print axioms EnetPoC.bnGammaB_den
#print axioms EnetPoC.bnBetaB_den
#print axioms EnetPoC.depthwiseWB_den
#print axioms EnetPoC.depthwiseStridedWB_den
-- ch8-EfficientNet-B0 §1a TIE
#print axioms EnetTiePoC.efficientnetLossCot_den
#print axioms EnetTiePoC.enet_exp_tied
#print axioms EnetTiePoC.enet_strided_tied
#print axioms EnetTiePoC.enet_noexp_tied
#print axioms EnetTiePoC.enet_stem_tied
#print axioms EnetTiePoC.enet_head_tied
#print axioms EnetTiePoC.efficientnet_net_tied
-- M2: the MLP per-layer parameter-gradient assembly (Crux A)
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
#print axioms IR.mlpCotOut1_denote
#print axioms IR.mlpCotOut0_denote
#print axioms IR.mlp_output_total_loss_grad
-- The conditional hidden-layer folds
#print axioms IR.mlp_hidden_total_loss_grad
#print axioms IR.mlp_input_total_loss_grad
-- Whole-net capstone: every weight layer's total-loss gradient at once (one statement).
#print axioms IR.mlp_whole_net_weight_grads
-- Render close: the rendered MLP train step's six param outputs (W₂',W₁',W₀',b₂',b₁',b₀')
#print axioms IR.mlp_render_W2_certified
#print axioms IR.mlp_render_W1_certified
#print axioms IR.mlp_render_W0_certified
#print axioms IR.mlp_render_b2_certified
#print axioms IR.mlp_render_b1_certified
#print axioms IR.mlp_render_b0_certified
-- R4 Stage A, Chapter 3 (MLP)
#print axioms StableHLO.reluF_faithful
#print axioms StableHLO.selectPos_faithful
-- The batched-index move for ViT/ConvNeXt (§0.2 2)
#print axioms StableHLO.den_batchOp_lnRow_eq_lnRowF
#print axioms StableHLO.den_batchOp_gelu_eq_geluF
-- ViT increment 1 (§0.2 3)
#print axioms StableHLO.den_batchOp_denseRow_eq_denseRowF
-- The saved-activation backwards, which CANNOT be descriptors
#print axioms StableHLO.den_lnRowBackB_per_example
-- The two statements the EMIT tie structurally cannot make (both forms render identical)
#print axioms StableHLO.den_batchOp_softmaxDiv_per_example
#print axioms StableHLO.den_rowDenseBiasGradB_at_one
-- ViT's version of that trap
#print axioms StableHLO.den_batchOp_clsSlice_per_example
-- ViT increment 2
#print axioms StableHLO.den_matmulFB_per_example
#print axioms StableHLO.den_softmaxRowBackB_per_example
-- The batch sum that is invisible at N = 1
#print axioms StableHLO.den_posEmbedGradB_at_one
-- CLASSIFIER DROPOUT (`recipe_gaps.md` gap C)
#print axioms StableHLO.dropoutB_faithful
#print axioms StableHLO.dropoutB_back_faithful
#print axioms StableHLO.den_dropoutB_of_dropScale
#print axioms Proofs.dropPath_scales_uniformly
#print axioms Proofs.dropout_eq_reference
#print axioms Proofs.dropout_ones_id
#print axioms Proofs.dropout_vjp_is_self
#print axioms StableHLO.mlpFwdGraph_faithful
#print axioms StableHLO.mlpBackGraph_faithful
-- R4 Stage A, Chapter 4 (CNN)
#print axioms StableHLO.flatConvF_faithful
#print axioms StableHLO.maxPoolF_faithful
#print axioms StableHLO.cnnFwdGraph_faithful
#print axioms StableHLO.convBack_faithful
#print axioms StableHLO.maxPoolBack_faithful
-- He et al.'s 3×3/s2 STEM POOL (`planning/archive/rsb_a3_r50_verified.md` §4b)
#print axioms win3RowInv_first_dup
#print axioms win3ColInv_first_dup
#print axioms win3Row_mem_le_two
-- the argmax: `Finset.sup'` over `Fin 3 × Fin 3`, which is why window size stopped mattering
#print axioms maxPool3s2_eq_at_max
#print axioms maxPool3s2_eq_argmax_value
-- the local linearisation and the smooth-point VJP
#print axioms maxPool3s2_flat_hasFDerivAt
#print axioms pdiv3_maxPool3s2_smooth
#print axioms maxPool3s2_has_vjp_at3
#print axioms maxPool3s2Flat_differentiableAt
#print axioms maxPool3s2Flat_has_vjp_at
-- the DISCHARGE lemma for the smoothness hypothesis
#print axioms maxPool3s2Smooth_of_injective
-- and the float side (`floatBridges_maxPool`'s peer)
#print axioms floatClose_maxPool3s2
#print axioms floatBridges_maxPool3s2
-- and the codegen that denotes it
#print axioms StableHLO.maxPool3s2F_faithful
#print axioms StableHLO.maxPool3s2Back_faithful
-- A2c: the whole-chain CNN backward graph denotes the proven conditional whole-network VJP
#print axioms StableHLO.cnnBackGraph_faithful

-- Chapter-5 CIFAR-10 2D CNN (no BN)
#print axioms cifarCnn_has_vjp_at_correct
#print axioms StableHLO.cifarFwdGraph_faithful
-- Concrete tiny CIFAR instance
#print axioms Tiny.cifarTinyCnn_has_vjp_correct

-- Chapter-5 CIFAR **BatchNorm** variant
#print axioms cifarCnnBn_has_vjp_at_correct
#print axioms StableHLO.cifarBnFwdGraph_faithful
#print axioms StableHLO.bnBack_faithful

-- Deeper 8-conv CIFAR (the pedagogical BN-acceleration demo)
#print axioms cifarCnn8_has_vjp_at_correct
#print axioms cifarCnnBn8_has_vjp_at_correct
-- ...and the rendered-FORWARD peer
#print axioms StableHLO.cifar8FwdGraph_faithful
#print axioms StableHLO.cifar8BnFwdGraph_faithful

-- Chapter-6 ResNet-style net
#print axioms StableHLO.resnetFwdGraph_faithful

-- Chapter-6 ResNet **Milestone B** (toward real ResNet-34)
#print axioms flatConvStride2_has_vjp_correct
-- ...and its weight-VJP (the kernel grad for training a strided block)
#print axioms flatConvStride2_weight_grad_has_vjp_correct
-- Deep-block chain
#print axioms vjp_chain_correct
-- Strided downsampling block (conv stride-2 → BN → relu)
#print axioms convBnReluStrided_has_vjp_at_correct
-- Strided residual-PROJECTION block (relu(proj(x)+F(x)), the stage-start downsampling block)
#print axioms rblkPStrided_has_vjp_at_correct
-- Stage assembly: the conditional (_at)
#print axioms vjp_chain_at_correct
#print axioms resStage_has_vjp_at_correct
-- THE WHOLE-NETWORK ResNet-34 VJP
#print axioms resnet34_has_vjp_at
-- B7: the UNCONDITIONAL concrete instance
#print axioms ResNet34Concrete.resnet34Concrete_has_vjp_correct
-- The first NON-DEGENERATE ResNet-34 whole-net backward witness (Item A, level 2)
#print axioms ResNet34LivePC.liveFwd2_has_vjp_correct
#print axioms ResNet34LivePC.liveFwd2_nonconstant
-- Item A level 3: the nonzero-Jacobian SEAL for the live ResNet-34 (ResNet34LiveSeal.lean)
#print axioms ResNet34LiveSeal.liveFwd2_jacobian_nonzero
#print axioms ResNet34LiveSeal.liveFwd2_backward_nontrivial
-- Item A FULL DEPTH (ResNet34LiveFull.lean)
#print axioms ResNet34LiveFull.liveFwd2Full_has_vjp_correct
#print axioms ResNet34LiveFull.liveFwd2Full_nonconstant
#print axioms ResNet34LiveFull.liveFwd2Full_jacobian_nonzero
#print axioms ResNet34LiveFull.liveFwd2Full_backward_nontrivial
-- Item B2 FULL DEPTH (MobileNetV2JacobianSealFull.lean)
#print axioms Mnv2Live.fwdFull_has_vjp_correct
#print axioms Mnv2Live.fwdFull_nonconstant
#print axioms Mnv2Live.fwdFull_jacobian_nonzero
#print axioms Mnv2Live.fwdFull_backward_nontrivial
-- Item D (ResNet34LiveRealistic.lean)
#print axioms ResNet34LiveRealistic.liveFwd224_has_vjp_correct
#print axioms ResNet34LiveRealistic.liveFwd224_nonconstant
-- Item D WEIGHT-GENERICITY (ResNet34LiveGeneric.lean)
#print axioms liveFwdW_has_vjp_correct
#print axioms liveFwdW_mixing_has_vjp_correct
-- Item D LEVEL 3 (ResNet34LiveRealisticSeal.lean)
#print axioms R34RealSeal.liveFwd224_jacobian_nonzero
#print axioms R34RealSeal.liveFwd224_backward_nontrivial
-- Item D LEVEL 3 for MobileNetV2 (MobileNetV2SealRealistic.lean)
#print axioms Mnv2RealSeal.fwdR_jacobian_nonzero
#print axioms Mnv2RealSeal.fwdR_backward_nontrivial
-- B8: per-channel BatchNorm
#print axioms bnPerChannelFlat_has_vjp_correct
-- B8a': the RENDERABLE per-channel BN backward
#print axioms bnPerChannel_grad_input_correct
-- B9 entry: per-channel BN on the network's Tensor3 (oc*h)
#print axioms bnPerChannelTensor3_has_vjp_correct
-- B9 entry (cont.)
#print axioms bnPerChannelTensor3_grad_input_correct
-- ch8 E5 (EfficientNet BATCH-norm)
#print axioms bnBatchTensor4_grad_input_correct
-- B8b: the per-channel BN SHlo op pair backward-faithfulness
#print axioms StableHLO.bnPerChannelBack_faithful
-- R4 syntactic core
#print axioms StableHLO.roundtrip
-- R4 syntactic LEXER numeric keystone
#print axioms StableHLO.parseNat_toString
-- CIFAR-BN render CLOSE
#print axioms bnPerChannel_grad_gamma_correct
#print axioms bnPerChannel_grad_beta_correct
#print axioms cifar_bn_render_gamma_certified
#print axioms cifar_bn_render_beta_certified
-- CNN conv-close UPGRADE
#print axioms cnnDenseHeadCot_denote
#print axioms cnn_render_convW2_chain_certified
#print axioms cnn_render_convb2_chain_certified
#print axioms cnn_render_convW1_chain_certified
#print axioms cnn_render_convb1_chain_certified
-- DEEPER 8-conv CIFAR (cifar8)
#print axioms cifar8DenseHeadCot_denote
#print axioms cifar8_render_denseWb_chain_certified
#print axioms cifar8_render_densebb_chain_certified
#print axioms cifar8_render_denseW9_chain_certified
#print axioms cifar8_render_denseb9_chain_certified
#print axioms cifar8_render_convW8_chain_certified
#print axioms cifar8_render_convb8_chain_certified
#print axioms cifar8_render_bn8gamma_chain_certified
#print axioms cifar8_render_bn8beta_chain_certified
#print axioms cifar8_render_convW7_chain_certified
#print axioms cifar8_render_convb7_chain_certified
-- MobileNetV2 CLOSE (planning/archive/mobilenetv2_close.md Item C)
#print axioms mnv2_depthwise_weight_grad_bridge
#print axioms mnv2_depthwise_bias_grad_bridge
#print axioms mnv2_render_depthwiseW_certified
#print axioms mnv2_render_depthwiseb_certified
#print axioms mnv2_render_stem_convW_certified
#print axioms mnv2_render_stem_convb_certified
#print axioms mnv2_render_depthwiseW_strided_certified
#print axioms mnv2_render_depthwiseb_strided_certified
-- Their XLA-SAME twins (2026-09-05)
#print axioms mnv2_render_stem_convW_xla_certified
#print axioms mnv2_render_stem_convb_xla_certified
#print axioms mnv2_render_depthwiseW_strided_xla_certified
#print axioms mnv2_render_depthwiseb_strided_xla_certified
-- MobileNetV2 RENDER (planning/archive/mobilenetv2_close.md Item A)
#print axioms StableHLO.mobilenetv2FwdGraphFullPC_faithful
-- MobileNetV2 cotangent-chain CLOSE (Item D)
#print axioms invres_render_projW_chain_certified
#print axioms invres_render_projb_chain_certified
#print axioms invres_render_dwW_s1_chain_certified
#print axioms invres_render_dwb_s1_chain_certified
#print axioms invres_render_dwW_s2_chain_certified
#print axioms invres_render_dwb_s2_chain_certified
#print axioms invres_render_expW_s1_chain_certified
#print axioms invres_render_expW_s2_chain_certified
#print axioms mnv2_stem_render_convW_chain_certified
-- THE COTANGENT PASS / = ∂loss/∂θ FOLD
#print axioms conv_total_loss_grad_fold
#print axioms conv_bias_total_loss_grad_fold
#print axioms depthwise_total_loss_grad_fold
#print axioms depthwise_bias_total_loss_grad_fold
-- EfficientNet-B0 CLOSE (Item C)
#print axioms enet_render_dw5W_certified
#print axioms enet_render_dw5b_certified
#print axioms enet_render_dw5W_strided_certified
#print axioms enet_render_dw5b_strided_certified
#print axioms enet_render_bngamma_certified
#print axioms enet_render_bnbeta_certified
-- ResNet-34 CLOSE (Item C)
#print axioms r34_render_stem_convW_certified
#print axioms r34_render_stem_convb_certified
#print axioms r34_render_blockConvW_certified
#print axioms r34_render_blockConvb_certified
#print axioms r34_render_downConvW_certified
#print axioms r34_render_downConvb_certified
-- ResNet-34 RENDER (Item A)
#print axioms StableHLO.idBlockGraphPC_faithful
#print axioms StableHLO.downBlockGraphPC_faithful
#print axioms StableHLO.resnet34FwdGraphFullPC_faithful
-- ResNet-34 cotangent-chain CLOSE (Item D)
#print axioms StableHLO.stemGraphB_faithful
#print axioms StableHLO.mbNoExpGraphB_faithful
#print axioms StableHLO.mbStridedGraphB_faithful
#print axioms StableHLO.mbResidGraphB_faithful
#print axioms StableHLO.headGraphB_faithful
#print axioms StableHLO.efficientnetFwdGraphB_faithful
-- EfficientNet-B0 cotangent-chain CLOSE (Item D)
#print axioms batchMap_has_vjp
#print axioms batchMap_differentiable
#print axioms reindex_has_vjp
#print axioms bnBatchLA_has_vjp
#print axioms bnBatchLA_differentiable
-- Per-block batched gradients (the per-block VJP, the user-requested deliverable)
#print axioms mbNoExpFwdB_has_vjp
#print axioms mbStridedFwdB_has_vjp
#print axioms mbResidFwdB_has_vjp
#print axioms headFwdB_has_vjp
#print axioms efficientnetForwardB_has_vjp
-- FULL EfficientNet-B0 (all 16 MBConv blocks, real [t,c,n,s,k] spec)
#print axioms StableHLO.mbExpGraphB_faithful
#print axioms StableHLO.efficientnetFwdGraphB_full_faithful
#print axioms efficientnetForwardB_full_has_vjp
-- The nested↔∘-chain bridge + correctness on the nested forward itself
#print axioms efficientnetForwardB_full_eq_chain
#print axioms efficientnetForwardB_full_has_vjp_correct
-- ConvNeXt RENDER (planning/archive/convnext_close.md Item A)
#print axioms StableHLO.convNextFwdGraph_faithful
-- ConvNeXt CLOSE (planning/archive/convnext_close.md Item C)
#print axioms cnx_render_dw7W_certified
#print axioms cnx_render_dw7b_certified
#print axioms pdiv_layerScale_gamma
#print axioms layerScale_gamma_grad_bridge
#print axioms cnx_render_lsgamma_certified
#print axioms cnx_lnGamma_grad_bridge
#print axioms cnx_lnBeta_grad_bridge
#print axioms cnx_render_lngamma_certified
#print axioms cnx_render_lnbeta_certified
-- ConvNeXt §1 fold START
#print axioms Proofs.CnxPoC.pdiv_layerScaleCh_gamma
#print axioms Proofs.CnxPoC.cnx_render_lsgammaCh_certified
-- ConvNeXt §1 fold
#print axioms Proofs.CnxPoC.layerScaleChGammaSgd_den
#print axioms Proofs.CnxPoC.lnGammaSgd_den
#print axioms Proofs.CnxPoC.lnBetaSgd_den
-- The CHANNEL-LN γ/β param certs (ConvNeXtChannelLN)
#print axioms Proofs.chanRowsIdxInv_chanRowsIdx
#print axioms Proofs.chanRowsIdx_chanRowsIdxInv
#print axioms Proofs.pdiv_reindexOut_contract
#print axioms Proofs.chanLN_gamma_contract
#print axioms Proofs.chanLN_beta_contract
#print axioms Proofs.cnx_render_chlngamma_certified
#print axioms Proofs.cnx_render_chlnbeta_certified
#print axioms Proofs.CnxPoC.chanLnGammaSgd_den
#print axioms Proofs.CnxPoC.chanLnBetaSgd_den
-- ConvNeXt cotangent-chain CLOSE (planning/archive/convnext_close.md Item D)
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
-- ch9-ConvNeXt-T FULL [3,3,9,3] §1a TIE
#print axioms Proofs.CnxTiePoC.cnx_block_ch_tied
#print axioms Proofs.CnxTiePoC.cnx_down_ch_tied
#print axioms Proofs.CnxTiePoC.cnx_stem_ch_tied
#print axioms Proofs.CnxTiePoC.cnx_head_ch_tied
#print axioms Proofs.CnxTiePoC.cnxLossCot_den
#print axioms Proofs.CnxTiePoC.cnx_net_tied_certified
-- ViT RENDER (planning/archive/vit_close.md Item A)
#print axioms vitForward2_has_vjp
#print axioms vitForward2_has_vjp_correct
#print axioms mhsa_layer_one_head
#print axioms StableHLO.vitFwdGraph_faithful
-- ViT CLOSE (planning/archive/vit_close.md Item C)
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
-- ViT cotangent-chain CLOSE (planning/archive/vit_close.md Item D)
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
-- ViT SCALING PASS
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
-- Vector-LN render upgrade + chain pins
#print axioms vit_render_vecln1gamma_chain_certified
#print axioms vit_render_vecln1beta_chain_certified
#print axioms vit_render_vecln2gamma_chain_certified
#print axioms vit_render_vecln2beta_chain_certified
#print axioms vit_render_veclnFgamma_chain_certified
#print axioms vit_render_veclnFbeta_chain_certified

-- ViT scaling pass: multi-head (ViTMultiHead.lean)
#print axioms sum_headPadMat_apply
#print axioms mhsa_layer_spelled
#print axioms vitBlockSpelledMH_eq
#print axioms vitBlockSpelledMHV_eq
#print axioms StableHLO.den_headsSumG
#print axioms StableHLO.vitFwdGraphMH_faithful
#print axioms StableHLO.vitFwdGraphMHV_faithful

-- ViT scaling pass: depth-k (ViTDepthK.lean)
#print axioms vitBodyKVFlat_eq_flatten
#print axioms vitBodyKVFlat_has_vjp
#print axioms vitForwardKV_two_eq
#print axioms vitForwardKV_has_vjp
#print axioms vitForwardKV_has_vjp_correct
-- Production capstone
#print axioms vitTiny_has_vjp_correct
#print axioms StableHLO.vitBodyGraphKMHV_den
#print axioms StableHLO.vitFwdGraphKMHV_faithful

-- Paper-spec full MobileNetV2 (MobileNetV2FullPaper.lean)
#print axioms StableHLO.ivNoExpGraphW_faithful
#print axioms StableHLO.ivExpOnlyGraphW_faithful
#print axioms StableHLO.ivResidGraphW_faithful
#print axioms StableHLO.ivStridedGraphW_faithful
#print axioms StableHLO.mobilenetv2FwdGraphPaper_faithful

-- ...and its whole-net VJP at all 17 (MobileNetV2FullVJP.lean)
#print axioms ivNoExpW_has_vjp_at
#print axioms ivExpOnlyW_has_vjp_at
#print axioms ivResidW_has_vjp_at
#print axioms ivStridedW_has_vjp_at
#print axioms convBnRelu6StridedPC_has_vjp_at
#print axioms mobilenetv2ForwardPaper_eq_chain
#print axioms mobilenetv2_full_has_vjp_at
#print axioms mobilenetv2_full_has_vjp_at_correct

-- Full ConvNeXt-T [3,3,9,3] (ConvNeXtFullT.lean)
#print axioms decimateOddFlat_has_vjp
#print axioms flatConvStride4_has_vjp
#print axioms convNextStageChK_has_vjp
#print axioms cnxDownChW_has_vjp
#print axioms convNextForwardTCh_has_vjp
-- The nested↔∘-chain bridge
#print axioms convNextForwardTCh_eq_chain
#print axioms convNextForwardTCh_has_vjp_correct
-- The channel-LN GRAPH + faithfulness (rung E's apex)
#print axioms StableHLO.chanLNGraph_faithful
#print axioms StableHLO.cnxBlockChGraphW_faithful
#print axioms StableHLO.cnxStageChGraphK_den
#print axioms StableHLO.cnxDownChGraphW_faithful
#print axioms StableHLO.convNextFwdGraphTCh_faithful
-- §2n: the SCALAR-LN twin of this chain

-- ℝ→Float32 bridge, Tier 1 (FloatBridge.lean)
#print axioms FloatModel.dot_close
#print axioms FloatModel.dot_close_linear
-- P2 (TreeReduceBridge.lean, planning/archive/adjoint_chain.md)
#print axioms FloatModel.dotMixed
#print axioms FloatModel.dot_close_mixed
#print axioms FloatModel.dot_close_mixed_uniform
#print axioms FloatModel.dotMixed_exact_leaf
-- §1c threaded through the dense layer
#print axioms FloatModel.denseMixed
#print axioms FloatModel.dense_close_mixed
#print axioms FloatModel.dense_close
#print axioms FloatModel.dense_close_fresh
#print axioms FloatModel.relu_close
-- Subnormal-floor closure (FloatSubnormalBridge.lean, planning §2)
#print axioms Proofs.FaithfulFloatModel.toFloatModel
#print axioms Proofs.FaithfulFloatModel.err_of_normal
#print axioms Proofs.FaithfulFloatModel.exactFaithful
-- The stays-normal invariant for BN/LN
#print axioms Proofs.bnDenom_normal
#print axioms Proofs.bnSqrt_normal
#print axioms Proofs.istd_ge_minNormal
#print axioms Proofs.subFloor_total_negligible
-- MaxPool exact-in-float (CNN.lean, planning §1b-A)
#print axioms max_close
#print axioms maxPool2_close
#print axioms maxPoolFlat_close
-- Conv forward rounding budget (SgdDescentCnn.lean, planning §1b-A)
#print axioms sum_w3
#print axioms conv2d_eq_dense
#print axioms convPad_close
#print axioms FloatModel.convF
#print axioms FloatModel.convF_close
-- Whole-net capstone (SgdDescentCnn.lean, planning §1b-A)
#print axioms FloatModel.flatConvF
#print axioms FloatModel.flatConvF_close
#print axioms FloatModel.mnistCnnNoBnForwardF
#print axioms FloatModel.cnn_float_close
-- Chapter-5 no-BN CIFAR CNN (CifarFloatBridge.lean)
#print axioms rsqrt_lipschitz
#print axioms bnVar_nonneg
#print axioms bnIstd_close
-- bnIstd_close at the OPERATING POINT
#print axioms bnIstd_close_at
-- BN float tail (BnFloatBridge.lean)
#print axioms FloatModel.bnForwardF
#print axioms FloatModel.bnMean_close
#print axioms FloatModel.bnVar_close
#print axioms FloatModel.bnForward_close_of
#print axioms FloatModel.bnForward_close
-- ResNet-34 structural float ops (Resnet34FloatBridge.lean)
#print axioms FloatModel.add_close
#print axioms FloatModel.reluAdd_close
#print axioms FloatModel.flatConvStride2F_close
#print axioms FloatModel.bnPerChannelFlat_close_of
#print axioms FloatModel.gapFlat_close
-- Real-BN input-sensitivity (BnInputBridge.lean)
#print axioms bnMean_input_close
#print axioms bnVar_input_close
#print axioms bnIstd_input_close
#print axioms bnForward_input_close
-- First assembled ResNet block step (Resnet34BlockBridge.lean)
#print axioms FloatModel.bnRelu_close
-- Whole-net certificate backbone (FloatComposeBridge.lean)
#print axioms FloatClose.comp
#print axioms floatClose_relu
#print axioms floatClose_flatConv
#print axioms floatClose_reluConv
-- The residual combinator (AdjointChainResidual.lean)
#print axioms floatClose_maxPool
#print axioms floatClose_cifarStage
-- The r34 wraps that let the fold RUN on a real block
#print axioms floatClose_residualBlock
#print axioms floatClose_resBlock
#print axioms floatClose_bnRelu
-- THE FINAL FOLD: floatClose_id + floatClose_iterate
#print axioms floatClose_id
#print axioms floatClose_iterate
#print axioms floatClose_r34_stages
-- planning/archive/floatbridge_enet_vit.md §1a–§1d (EfficientNet float bridge finished)
#print axioms floatClose_addResidual
-- §1b: the remaining FloatClose instances, all wraps of existing closeness
#print axioms FloatModel.bnStep_close
#print axioms floatClose_bn
#print axioms floatClose_dense
#print axioms globalAvgPoolFlat_eq_bnMean
#print axioms floatClose_gap
-- §1c: depthwise conv
#print axioms depthwiseConv2d_eq_dense
#print axioms FloatModel.depthwiseConv2dF_close
#print axioms FloatModel.depthwiseFlatF_close
#print axioms floatClose_depthwise
-- §1d: the SE gate net + MBConv fold (pure assembly)
#print axioms FloatClose.cod_nonneg
#print axioms FloatClose.modulus_zero_nonneg
#print axioms FloatBridges.comp
#print axioms floatClose_residual
#print axioms FloatBridges.residual
#print axioms floatBridges_flatConv
#print axioms floatBridges_dense
#print axioms floatBridges_depthwise
-- The backward whole-net FOLD
#print axioms Proofs.floatClose_reluMaskBack
#print axioms Proofs.floatBridges_linBack
#print axioms Proofs.mlpInputGrad_floatBridges
-- 1d smooth-activation backward (diagonal Jacobian)
#print axioms Proofs.floatClose_diagBack
#print axioms Proofs.floatBridges_diagBack
-- Strided-conv backward (r34 down-blocks + stem)
#print axioms Proofs.decimateBack_eq_vjp
-- stride-4 (ConvNeXt 4×4/s4 patchify)
#print axioms Proofs.decimateOddIdx_injective
-- FloatBridges binds the float map existentially (constrains nothing); r34_floatBridgesTo names the net and carries the budget
#print axioms Proofs.FloatBridgesTo.comp
#print axioms Proofs.FloatBridgesTo.fresh_le
#print axioms Proofs.FloatBridgesTo.residual
#print axioms Proofs.floatBridgesTo_cifarStage
#print axioms Proofs.floatBridgesTo_resBlock
-- The CIFAR-8 chain tie
#print axioms Proofs.FloatModel.bnMean_close_of
#print axioms Proofs.FloatModel.bnMean_num_le
-- The SECOND ImageNet-scale whole-net float number: the MobileNetV2 inference forward (mnv2EvalBridge)
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphFullPCEval_faithful
-- The mnv2 block bridges are generic in the NORMALISATION too (`*Gen`)
#print axioms Proofs.mobilenetv2ForwardPaperEval
#print axioms Proofs.StableHLO.ivNoExpGraphEvalW_faithful
#print axioms Proofs.StableHLO.ivExpOnlyGraphEvalW_faithful
#print axioms Proofs.StableHLO.ivResidGraphEvalW_faithful
#print axioms Proofs.StableHLO.ivStridedGraphEvalW_faithful
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphPaperEval_faithful
-- The EfficientNet-B0 INFERENCE forward and its graph
#print axioms Proofs.efficientnetForwardBEval
#print axioms Proofs.StableHLO.stemGraphBEval_faithful
#print axioms Proofs.StableHLO.mbNoExpGraphBEval_faithful
#print axioms Proofs.StableHLO.mbStridedGraphBEval_faithful
#print axioms Proofs.StableHLO.mbResidGraphBEval_faithful
#print axioms Proofs.StableHLO.headGraphBEval_faithful
#print axioms Proofs.StableHLO.efficientnetFwdGraphBEval_faithful
-- The B0 stage and block bridges are generic in the NORMALISATION (`*BGen`)
#print axioms Proofs.mbExpFwdBEval
#print axioms Proofs.StableHLO.mbExpGraphBEval_faithful
#print axioms Proofs.StableHLO.mbNoExpGraphEvalW_faithful
#print axioms Proofs.StableHLO.mbStridedGraphEvalW_faithful
#print axioms Proofs.StableHLO.mbResidGraphEvalW_faithful
#print axioms Proofs.StableHLO.mbExpGraphEvalW_faithful
#print axioms Proofs.efficientnetForwardB_fullEval
#print axioms Proofs.StableHLO.efficientnetFwdGraphB_fullEval_faithful
-- The FOURTH ImageNet-scale whole-net float statement, ConvNeXt-T's forward (cnxBridge), a different kind of statement
#print axioms Proofs.efficientnetForwardB_eq_chain
-- §B integrity tie (the r34 identity block)
#print axioms Proofs.convFlatBack_eq_vjp_backward
#print axioms Proofs.rblkPC_has_vjp_at
#print axioms Proofs.r34IdBlockBack_eq_rblkPC_vjp
-- §B integrity tie (the r34 DOWNSAMPLE block)
#print axioms Proofs.flatConvStride2Back_eq_vjp_backward
-- Its XLA-SAME peer (the TF-origin B0 / MobileNetV2 stems): the same conv leaf + decimateOddBack rfl.
#print axioms Proofs.flatConvStride2XlaBack_eq_vjp_backward
#print axioms Proofs.rblkPStridedPC_has_vjp_at
#print axioms Proofs.r34DownBlockBack_eq_rblkPStridedPC_vjp
-- §B DEPTHWISE adjoint gate (shared prereq for convnext/mnv2/enet)
#print axioms Proofs.depthwiseConv2d_dwReverse_eq_input_grad_formula
#print axioms Proofs.depthwiseFlatBack_eq_vjp_backward
#print axioms Proofs.depthwiseStride2FlatBack_eq_vjp_backward
-- Its XLA-SAME peer (MobileNetV2's four strided depthwises, B0's downsample depthwise).
#print axioms Proofs.depthwiseStride2FlatXlaBack_eq_vjp_backward
-- §B integrity tie (convnext)
#print axioms Proofs.cnxBlockBodyBack_eq_convNextBlockBody_vjp
#print axioms Proofs.cnxBlockBack_eq_convNextBlock_vjp
-- §2n §B at ConvNeXt's REAL channel LayerNorm
#print axioms Proofs.HasVJP.backward_unique
#print axioms Proofs.bn_grad_input_eq_vjp_backward
#print axioms Proofs.transposeFlat_has_vjp_backward_eq
#print axioms Proofs.rowLNVecFlat_has_vjp_backward_eq
#print axioms Proofs.chanLNTensor3Back_eq_chanLN_vjp
#print axioms Proofs.cnxBodyWithChanLNBack_eq_vjp
#print axioms Proofs.cnxBlockChBack_eq_vjp
-- §B integrity tie (mnv2)
#print axioms Proofs.invresBodyPC_has_vjp_at
#print axioms Proofs.invresBodyBackPC_eq_invresBodyPC_vjp
#print axioms Proofs.invresBodyStridedPC_has_vjp_at
#print axioms Proofs.invresBodyStridedBackPC_eq_invresBodyStridedPC_vjp
-- §B integrity tie (efficientnet)
#print axioms Proofs.mbconvBodyBack_eq_mbconvBody_vjp
-- §B integrity tie (vit MHSA — the sdpa adjoint)
#print axioms Proofs.projBack_core_coord
#print axioms Proofs.woback_unflatten
#print axioms Proofs.mhsaBackFlat_eq_mhsa_vjp
-- §B vit attn-SUBLAYER reconciliation (grounds the MHSA leaf in the block)
#print axioms Proofs.transformerAttnSublayer_backward_decomp
#print axioms Proofs.transformerAttnSublayerBack_flat_decomp
-- §B THE FULL vitBlockBack TIE (the per-token-LN enrichment that closes the structural gap above)
#print axioms Proofs.perRowFlatPR_LN_back
#print axioms Proofs.transformerMlp_back_flat_eq_perRowFlatPR
#print axioms Proofs.transformerMlpSublayer_backward_decomp
#print axioms Proofs.transformerBlock_backward_unfold_gen
#print axioms Proofs.attnSubFlatTie
#print axioms Proofs.mlpSubFlatTie
#print axioms Proofs.vitBlockBackPR_eq_transformerBlock_vjp
-- §B endpoint leaf ties
#print axioms Proofs.dense_transpose_eq_vjp_backward
#print axioms Proofs.gapBack_eq_vjp_backward
#print axioms Proofs.maxPoolFlatBack_eq_vjp_backward
-- He et al.'s 3×3/s2 stem pool's BACKWARD
#print axioms Proofs.maxPool3s2FlatBack_eq_vjp_backward
#print axioms Proofs.maxPool3s2Flat_has_vjp_at_vec
-- THE WHOLE-NET CERTIFIED TIE (ResNet-34, r34InputGrad_eq_resnet34_vjp)
#print axioms Proofs.cbrStridedPC_has_vjp_at
#print axioms Proofs.cbrStridedPC_differentiableAt
#print axioms Proofs.cbrStridedPCBack_eq_vjp_backward
#print axioms Proofs.r34InputGrad_eq_resnet34_vjp
-- AND THE SHAPE CHECK — the last net to get one (2026-09-04)
#print axioms Proofs.resnet34Forward_full_pc_eq_chain
-- THE SAME TIE FOR THE WHOLE MOBILENETV2 (MobileNetV2WholeBackCertifiedTie.lean)
#print axioms Proofs.convStridedBnRelu6PC_has_vjp_at
#print axioms Proofs.convStridedBnRelu6PC_differentiableAt
#print axioms Proofs.convStridedBnRelu6PCBack_eq_vjp_backward
#print axioms Proofs.convBnRelu6PCBack_eq_vjp_backward
#print axioms Proofs.residualBack_eq_vjp_backward
#print axioms Proofs.mobilenetv2PC_has_vjp_at
#print axioms Proofs.mobilenetv2Forward_full_pc_eq_chain
#print axioms Proofs.mnv2InputGrad_eq_mobilenetv2_vjp
-- AND THE SAME TIE AT THE PAPER DEPTH — all seventeen bottlenecks (MobileNetV2PaperWholeBackCertifiedTie.lean)
#print axioms Proofs.mobilenetv2PaperPC_has_vjp_at
#print axioms Proofs.mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp
#print axioms Proofs.mobilenetv2ForwardPaper_eq_slots
-- AND FOR THE WHOLE EFFICIENTNET-B0 (EfficientNetWholeBackCertifiedTie.lean)
#print axioms Proofs.efficientnetB_has_vjp
#print axioms Proofs.stemBBack_eq_vjp_backward
#print axioms Proofs.headFwdBBack_eq_vjp_backward
#print axioms Proofs.efficientnetInputGradB_eq_efficientnetForwardB_vjp
#print axioms Proofs.efficientnetForwardB_has_vjp_committed
-- AND AT THE PAPER DEPTH — all 16 MBConv blocks (EfficientNetFullWholeBackCertifiedTie.lean)
#print axioms Proofs.efficientnetB_full_has_vjp
#print axioms Proofs.efficientnetInputGradB_full_eq_efficientnetB_full_vjp
#print axioms Proofs.efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp
#print axioms Proofs.efficientnetInputGradB_full_correct
-- AND AT BATCH BATCH-NORM (Resnet34BackCertifiedTieB.lean, MobileNetV2WholeBackCertifiedTieB.lean)
#print axioms Proofs.HasVJPAt.backward_unique
#print axioms Proofs.maxPool3s2FlatBackB_eq_vjp_backward
#print axioms Proofs.cbReluStridedBBack_eq_vjp_backward
#print axioms Proofs.r34StemBBack_eq_vjp_backward
#print axioms Proofs.r34HeadBBack_eq_vjp_backward
#print axioms Proofs.r34B_full_has_vjp_at
#print axioms Proofs.r34InputGradB_eq_r34B_full_vjp
#print axioms Proofs.r34InputGradB_correct
#print axioms Proofs.resnet34ForwardB_full_eq_slots
#print axioms Proofs.mnv2StemBBack_eq_vjp_backward
#print axioms Proofs.cbrBBack_eq_vjp_backward
#print axioms Proofs.mnv2InputGradB_eq_mobilenetv2B_full_vjp
#print axioms Proofs.mnv2InputGradB_correct
#print axioms Proofs.mobilenetv2ForwardB_full_eq_slots
-- AND RESNET-50's T6 (Resnet50WholeBackCertifiedTieB.lean)
#print axioms Proofs.r50InputGradB_eq_r34B_full_vjp
#print axioms Proofs.r50InputGradB_correct
#print axioms Proofs.resnet50ForwardB_full_eq_slots
-- THE EVEN-KERNEL CONV BACKWARD (EvenKernelConvBack.lean)
#print axioms Proofs.padOdd
#print axioms Proofs.padOdd_abs_le
#print axioms Proofs.conv2d_padOdd_eq
#print axioms Proofs.flatConv_padOdd_eq
#print axioms Proofs.HasVJP.backward_unique_of_eq
#print axioms Proofs.convFlatBack_padOdd_eq_vjp_backward
#print axioms Proofs.flatConvStride2Back_padOdd_eq_vjp_backward
#print axioms Proofs.flatConvStride4Back_padOdd_eq_vjp_backward
-- ConvNeXt-T's whole-net backward tie, the two pieces that landed
#print axioms Proofs.cnxDownChBack_eq_vjp
#print axioms Proofs.cnxBlockChBackAt
#print axioms Proofs.cnxStageChKBack
#print axioms Proofs.cnxStageChKBack_eq_vjp
#print axioms Proofs.rowLNVecFlat_has_vjp_backward_eq_fun
#print axioms Proofs.cnxSavedA10
#print axioms Proofs.convNextForwardTCh_vjp_chain
#print axioms Proofs.convnextInputGrad_eq_convNextForwardTCh_vjp
-- ViT-Tiny's whole-net backward tie
#print axioms Proofs.vitBlockBackV
#print axioms Proofs.vitBlockBackVAt
#print axioms Proofs.vitTowerBackK
#print axioms Proofs.vitInputGradK
#print axioms Proofs.rowLNVecFlatBack_eq_vecLN_vjp
#print axioms Proofs.attnSubFlatTieV
#print axioms Proofs.mlpSubFlatTieV
#print axioms Proofs.vitBlockBackV_eq_transformerBlockV_vjp
#print axioms Proofs.vitBlockBackVAt_eq_vjp
#print axioms Proofs.vitHeadBack_eq_classifier_vjp
#print axioms Proofs.vitFinalLNBack_eq_vjp
#print axioms Proofs.vitPatchEmbedBack_eq_vjp
#print axioms Proofs.vitTowerBackK_eq_vjp
#print axioms Proofs.vitForwardKV_eq_chain
#print axioms Proofs.vitInputGradK_eq_vitApexVJP
#print axioms Proofs.vitInputGradK_eq_vitForwardKV_vjp
#print axioms Proofs.vitInputGradK_correct
#print axioms Proofs.vitTinyInputGrad_eq_vitTiny_vjp
-- ViT WHOLE-NET FORWARD (forward peer of vit_grad_floatBridges)
#print axioms FloatModel.softmaxF_close_at
#print axioms FloatModel.smErr_nonneg
#print axioms FloatModel.softmax_abs_le_one
-- ViT TRANSFORMER-BLOCK FOLD (planning/archive/floatbridge_enet_vit.md §2)
#print axioms FloatModel.dotSgd_step_close
#print axioms FloatModel.sumSgd_step_close
#print axioms sum_s2
#print axioms convWeightGrad_eq_dot
#print axioms convBiasGrad_eq_sum
#print axioms FloatModel.cnn_convW_step_float_close
#print axioms FloatModel.cnn_convb_step_float_close
-- Item C — the numeric conv-weight-step capstone (SgdDescentCnn.lean)
#print axioms FloatModel.mnist_cnn_convW_step_float_budget
-- Item C, bias peer (SgdDescentCnn.lean)
#print axioms FloatModel.mnist_cnn_convb_step_float_budget
-- CIFAR-8 last-conv SGD descent (SgdDescentCifar.lean)
#print axioms Proofs.cifarCnn8Forward_factor
#print axioms Proofs.cifar8_lastConv_sgd_descends
#print axioms FloatModel.pow_one_add_sub_one_le
#print axioms FloatModel.linear_float_close
#print axioms FloatModel.mlp_float_close
-- The numeric rung
#print axioms FloatModel.pow_gamma_bound
#print axioms FloatModel.dense_abs_le
#print axioms FloatModel.denseErr_le_uniform
#print axioms FloatModel.mlp_float_close_uniform
#print axioms FloatModel.mnist_mlp_float_budget
-- The gradient half, first rung
#print axioms FloatModel.mul_close
#print axioms FloatModel.sgd_step_close
#print axioms FloatModel.reluMask_close
#print axioms FloatModel.cot_step_close
#print axioms FloatModel.mlp_w2_step_float_close
#print axioms FloatModel.mlp_b2_step_float_close
#print axioms FloatModel.mlp_w1_step_float_close
-- Completion: the remaining param entries (b₁, W₀/b₀)
#print axioms FloatModel.mlp_b1_step_float_close
#print axioms FloatModel.mlp_w0_step_float_close
#print axioms FloatModel.mlp_b0_step_float_close
#print axioms FloatModel.mnist_w2_step_float_budget
-- The loss head — the LAST Tier-1 float hypothesis discharged
#print axioms FloatModel.sum_close
#print axioms FloatModel.softmax_perturb
#print axioms FloatModel.softmaxF_close
#print axioms FloatModel.softmax_ce_cot_close
#print axioms FloatModel.mnist_cot_budget
-- §3c (planning/archive/floatbridge_quantization.md)
#print axioms FloatModel.argmax_preserved
#print axioms FloatModel.denseMixedBudget
#print axioms FloatModel.dense_close_mixed_uniform_budget
#print axioms FloatModel.denseMixedBudget_le_of
#print axioms u_e4m3
#print axioms FloatModel.linear_e4m3_logit_budget
#print axioms FloatModel.linear_e4m3_argmax_preserved
-- §3b (planning/archive/floatbridge_quantization.md)
#print axioms QuantPoC.dequant_factors
#print axioms QuantPoC.e4m3_render_faithful
-- Inexact-gradient descent over ℝ (SgdDescent.lean)
#print axioms fderiv_apply_eq_sum_grad
#print axioms descent_segment
#print axioms sgd_descent_inexact
#print axioms sgd_descends
-- The smoothness hypothesis discharged for the Chapter-2 net
#print axioms gradAt_eq_pdiv
#print axioms linear_loss_gradAt
#print axioms dense_unflatten_drift
#print axioms linear_loss_grad_lipschitz
#print axioms linear_sgd_descends
-- Item D / G1 — the η-composition, the "two halves finally meet"
#print axioms FloatModel.linearFloatGrad
#print axioms linearFloatGrad_apply
#print axioms linear_grad_close
#print axioms linear_float_sgd_descends
-- The smoothness hypothesis discharged through the Chapter-3 MLP
#print axioms relu_entry_lipschitz
#print axioms sign_stable_of_close
#print axioms sum_abs_flatten_cols
#print axioms dense_unflatten_diff
#print axioms dense_unflatten_col_drift
#print axioms dense_unflatten_drift_sum
#print axioms margin_keeps_offkink
#print axioms margin_keeps_offkink_mid
#print axioms ce_dense_input_grad
#print axioms ce_head_relu_input_grad
#print axioms ce_head2_input_grad
#print axioms mlp_hidden_loss_differentiableAt
#print axioms mlp_hidden_loss_gradAt
#print axioms mlp_hidden_logit_drift
#print axioms mlp_hidden_loss_grad_lipschitz
#print axioms mlp_hidden_sgd_descends
#print axioms mlp_output_sgd_descends
-- Output-layer η-composition (planning §1a/§4, G1 for the MLP)
#print axioms mlp_output_float_sgd_descends
-- Hidden-layer float-backward grad-close (planning §1a/§4, the joint-step engine)
#print axioms FloatModel.cotErr_nonneg
#print axioms mlp_w1_grad_close
-- Hidden-layer η-composition (planning §3 descent, Step 1)
#print axioms FloatModel.mlpHiddenFloatGrad
#print axioms mlpHiddenFloatGrad_apply
#print axioms mlp_hidden_loss_gradAt_reluMask
#print axioms mlp_hidden_float_sgd_descends
#print axioms mlp_input_loss_differentiableAt
#print axioms mlp_input_loss_gradAt
#print axioms mlp_input_logit_drift
#print axioms mlp_input_loss_grad_lipschitz
#print axioms mlp_input_sgd_descends
-- Input-layer η-composition (planning §3 descent, Step 2)
#print axioms reluMask_dense_transpose_eq
#print axioms FloatModel.mlpInputFloatGrad
#print axioms mlpInputFloatGrad_apply
#print axioms mlp_input_loss_gradAt_reluMask
#print axioms mlp_w0_grad_close
#print axioms mlp_input_float_sgd_descends
-- The descent program reaches the Chapter-4 CNN (SgdDescentCnn.lean)
#print axioms max4_sub_abs_le
#print axioms max4_sub_abs_le_sum
#print axioms flatten_t3Idx
#print axioms sum_t3
#print axioms sum_window_cells
#print axioms maxPoolFlat_apply
#print axioms maxPoolFlat_entry_lipschitz
#print axioms maxPoolFlat_l1_contract
#print axioms ne_of_gap_of_close
#print axioms lt_of_lt_gap_of_close
#print axioms MaxPool2MarginQ.smooth_of_close
#print axioms MaxPool2MarginQ.smooth
#print axioms MaxPool2MarginQ.isArgmax_iff
#print axioms MaxPool2MarginQ.pdiv3_eq
-- Float-bridge §3 (CNN descent, Increment 1 keystone): the pool selector is an indicator pass-through in float
#print axioms MaxPool2MarginQ.poolBack_close
#print axioms conv2d_eq_convPad
#print axioms abs_convPad_le
#print axioms k4Idx_inj
#print axioms sum_abs_kernel_slab_le
#print axioms sum_abs_k4
#print axioms conv2d_kernel_sub
#print axioms conv2d_kernel_drift
#print axioms conv2d_kernel_drift_total
#print axioms conv2d_kernel_drift_sum
-- The conv2-layer rung, assembled (SgdDescentCnn.lean)
#print axioms ce_head3_input_grad
#print axioms pool_relu_input_grad
#print axioms conv2d_weight_pdiv
#print axioms conv2d_weight_pdiv_row_l1
#print axioms cnn_conv2_loss_differentiableAt
#print axioms cnn_conv2_loss_gradAt
-- Float-bridge §3 (CNN descent, Increment 1 keystone): the certified conv-2 gradient in dense/reluMask form
#print axioms dense_transpose_eq
#print axioms head3_cot_reluMask
#print axioms cnn_conv2_loss_gradAt_reluMask
-- Float-bridge §3 (CNN descent, Increment 2)
#print axioms t3Idx_surj
#print axioms mask_scalar_close
#print axioms FloatModel.dot_perturbed_close
#print axioms FloatModel.cnnConv2FloatGrad
#print axioms FloatModel.cnnConv2FloatGrad_apply
#print axioms FloatModel.cnnConv2GradBudget
#print axioms cnn_conv2_grad_close
#print axioms cnn_pool_l1_drift
#print axioms cnn_conv2_logit_drift
#print axioms cnn_margin2_keeps_offkink
#print axioms cnn_margin3_keeps_offkink
#print axioms cnn_margin4_keeps_offkink
#print axioms head3_sum_drift
#print axioms cnn_conv2_loss_grad_lipschitz
#print axioms cnn_conv2_sgd_descends
-- Float-bridge §3 (CNN descent, Increment 3)
#print axioms flatten_k4Idx
#print axioms k4Idx_surj
#print axioms cnn_conv2_float_sgd_descends
-- The conv1 rung (SgdDescentCnn.lean)
#print axioms sum_pinned_le
#print axioms abs_convTap_expand
#print axioms convTap_out_l1
#print axioms convTap_in_l1
#print axioms conv2d_input_pdiv3
#print axioms conv2d_flat_input_pdiv
#print axioms conv2d_input_entry_drift
#print axioms conv2d_input_l1_drift
#print axioms cnn1_z2_entry_drift
#print axioms cnn1_pool_l1_drift
#print axioms cnn1_logit_drift
#print axioms cnn1_margin1_keeps_offkink
#print axioms cnn1_margin2_keeps_offkink
#print axioms cnn1_margin3_keeps_offkink
#print axioms cnn1_margin4_keeps_offkink
#print axioms cnn1_pool_head_input_grad
#print axioms cnn_conv1_loss_differentiableAt
#print axioms cnn_conv1_loss_gradAt
-- Float-bridge §3 (CNN descent, Increment 4 keystone)
#print axioms cnn_conv1_loss_gradAt_reluMask
#print axioms cnn_conv1_loss_grad_lipschitz
#print axioms cnn_conv1_sgd_descends
-- Float-bridge §3 (CNN descent, Increment 4)
#print axioms convTap_abs_le
#print axioms FloatModel.cnnConv2CotBudget
#print axioms FloatModel.cnnConv2CotMag
#print axioms cnn_conv2_cot_close
#print axioms cnn_conv2_cot_real_abs_le
#print axioms abs_le_of_close
#print axioms convTap_back_close
#print axioms convTap_back_abs_le
#print axioms FloatModel.cnnConv1FloatGrad
#print axioms FloatModel.cnnConv1FloatGrad_apply
#print axioms FloatModel.cnnConv1GradBudget
#print axioms cnn_conv1_grad_close
#print axioms cnn_conv1_float_sgd_descends
-- The conv BIAS rungs (SgdDescentCnn.lean)
#print axioms conv2d_bias_sub
#print axioms conv2d_flat_bias_drift_total
#print axioms conv2d_flat_bias_drift_sum
#print axioms conv2d_bias_pdiv
#print axioms cnnb2_pool_l1_drift
#print axioms cnnb2_logit_drift
#print axioms cnnb2_margin2_keeps_offkink
#print axioms cnnb2_margin3_keeps_offkink
#print axioms cnnb2_margin4_keeps_offkink
#print axioms cnn_conv2_bias_loss_differentiableAt
#print axioms cnn_conv2_bias_loss_gradAt
#print axioms cnn_conv2_bias_loss_grad_lipschitz
#print axioms cnn_conv2_bias_sgd_descends
#print axioms cnnb1_z2_entry_drift
#print axioms cnnb1_pool_l1_drift
#print axioms cnnb1_logit_drift
#print axioms cnnb1_margin1_keeps_offkink
#print axioms cnnb1_margin2_keeps_offkink
#print axioms cnnb1_margin3_keeps_offkink
#print axioms cnnb1_margin4_keeps_offkink
#print axioms cnn_conv1_bias_loss_differentiableAt
#print axioms cnn_conv1_bias_loss_gradAt
#print axioms cnn_conv1_bias_loss_grad_lipschitz
#print axioms cnn_conv1_bias_sgd_descends
-- Float-bridge §3 (CNN descent, Increment 5)
#print axioms FloatModel.sum_perturbed_close
#print axioms FloatModel.cnnConv2BiasFloatGrad
#print axioms cnn_conv2_bias_loss_gradAt_reluMask
#print axioms FloatModel.cnnConv2BiasGradBudget
#print axioms cnn_conv2_bias_grad_close
#print axioms cnn_conv2_bias_float_sgd_descends
#print axioms FloatModel.cnnConv1BiasFloatGrad
#print axioms cnn_conv1_bias_loss_gradAt_reluMask
#print axioms FloatModel.cnnConv1BiasGradBudget
#print axioms cnn_conv1_bias_grad_close
#print axioms cnn_conv1_bias_float_sgd_descends
-- Adam/AdamW optimizer step over ℝ (Phase 3a, vit_train_to_vit_verified.md)
#print axioms adamVNext_nonneg
#print axioms adam_denom_pos
#print axioms adamWParam_apply
#print axioms adamWParam_wd_zero
-- Phase 3b: AdamW render-close (den-level faithfulness)
#print axioms Proofs.adamWParam_eq_scalar
#print axioms StableHLO.adamW_certified_grad
#print axioms StableHLO.adamB_certified_grad

-- EfficientNet backward-graph faithfulness (den-level)
#print axioms StableHLO.residualBackGraph_faithful
#print axioms StableHLO.residual_dense_backGraph_faithful
#print axioms StableHLO.seBlockBackGraph_faithful
#print axioms StableHLO.se_dense_backGraph_faithful
#print axioms StableHLO.gapBack_faithful
#print axioms StableHLO.broadcastBack_faithful
#print axioms StableHLO.seGate_backGraph_faithful
#print axioms StableHLO.seBlockFull_backGraph_faithful
#print axioms StableHLO.bnBack_faithful_fn
#print axioms StableHLO.convBnSwishBackGraph_faithful
#print axioms StableHLO.dwBnSwishBackGraph_faithful
#print axioms StableHLO.convBnBackGraph_faithful
#print axioms StableHLO.seGateBackGraphE_faithful
#print axioms StableHLO.seBlockFullBackGraphE_faithful
#print axioms StableHLO.mbconvBodyBackGraph_faithful
#print axioms StableHLO.mbconvResidual_backGraph_faithful
-- ConvNeXt backward-graph faithfulness (den-level)
#print axioms StableHLO.cnxBlockBodyBackGraph_faithful
#print axioms StableHLO.cnxResidBlockBackGraph_faithful
-- §2o Part A (2026-07-31)
#print axioms Proofs.rowLNBack_affine_eq
#print axioms StableHLO.chanLNBackGraph_faithful
#print axioms StableHLO.chanLNBackGraph_eq_vjp
#print axioms StableHLO.cnxBlockBodyChBackGraph_faithful
#print axioms StableHLO.cnxResidBlockChBackGraph_faithful
#print axioms StableHLO.cnxDownChBackGraph_faithful
#print axioms StableHLO.bnBatchBack_faithful
#print axioms StableHLO.convBackBatched_faithful
#print axioms StableHLO.depthwiseBackBatched_faithful
#print axioms StableHLO.bnBatchLA_back_conj
#print axioms StableHLO.bnBatchLABack_faithful
#print axioms StableHLO.seBackBatched_faithful
-- Batched MBConv stage backward graphs (the bn wrapper lets these compose).
#print axioms StableHLO.cbsBackBatchedGraph_faithful
#print axioms StableHLO.dwbsBackBatchedGraph_faithful
#print axioms StableHLO.projBackBatchedGraph_faithful
-- Capstone: the whole batched MBConv residual block backward graph.
#print axioms StableHLO.mbBodyBackBatchedGraph_faithful
#print axioms StableHLO.mbResidBlockBackBatchedGraph_faithful

-- EfficientNet DOWNSAMPLE MBConv body backward-graph faithfulness
#print axioms StableHLO.depthwiseStridedBackBatched_faithful
-- Its XLA-SAME peer, the token MobileNetV2's Adam render emits at its four strided depthwises.
#print axioms StableHLO.depthwiseStridedXlaBackBatched_faithful
-- The PER-EXAMPLE XLA-SAME backward tokens MobileNetV2's SGD train step emits (MobileNetV2Render.lean, 2026-09-05)
#print axioms StableHLO.depthwiseStridedXlaBack_faithful
#print axioms StableHLO.depthwiseStridedXlaWeightSgd_faithful
#print axioms StableHLO.depthwiseStridedXlaBiasSgd_faithful
#print axioms StableHLO.convStridedXlaWeightSgd_faithful
#print axioms StableHLO.convStridedXlaBiasSgd_faithful
-- Batched strided depthwise → bn → swish stage backward graph.
#print axioms StableHLO.dwbsSBackBatchedGraph_faithful
-- Capstone: the batched EfficientNet downsample MBConv body backward graph.
#print axioms StableHLO.mbDownBodyBackBatchedGraph_faithful

-- MobileNetV2 backward-graph faithfulness (den-level)
#print axioms StableHLO.cbrBackBatchedGraph_faithful
#print axioms StableHLO.dwbrBackBatchedGraph_faithful
-- The SE-less inverted-residual body backward graph (projB ∘ dwbrB ∘ cbrB).
#print axioms StableHLO.mnv2BodyBackBatchedGraph_faithful
-- Capstone: the whole batched MobileNetV2 inverted-residual block backward graph.
#print axioms StableHLO.mnv2ResidBlockBackBatchedGraph_faithful
-- DOWNSAMPLE (stride-2)
#print axioms StableHLO.dwbrBstridedBackBatchedGraph_faithful
-- Capstone: the batched MobileNetV2 downsample inverted-residual body backward graph.
#print axioms StableHLO.mnv2DownBodyBackBatchedGraph_faithful

-- ResNet-34 backward-graph faithfulness (den-level)
#print axioms StableHLO.cbReluBackBatchedGraph_faithful
-- The basic-block body backward graph (projB ∘ cbReluB).
#print axioms StableHLO.r34BodyBackBatchedGraph_faithful
-- Capstone: the whole batched ResNet-34 identity basic block backward graph
#print axioms StableHLO.r34BasicBlockBackBatchedGraph_faithful

-- ResNet-34 DOWNSAMPLE/STRIDED basic block backward-graph faithfulness
#print axioms StableHLO.convStridedBackBatched_faithful
-- Capstone: the whole batched ResNet-34 downsample basic block backward graph
#print axioms StableHLO.r34DownBlockBackBatchedGraph_faithful

-- ViT whole-block backward-graph faithfulness (den-level, per-token Mat-VJP)
#print axioms StableHLO.mhsa_backward_collapse
#print axioms StableHLO.mhsaBackGraph_faithful
#print axioms StableHLO.mlpSublayerBackGraph_faithful
#print axioms StableHLO.attnSublayerBackGraph_faithful
#print axioms StableHLO.transformerBlockBackGraph_faithful

-- ViT whole-block backward-graph faithfulness, lifted to GENERAL MULTI-HEAD
#print axioms StableHLO.mhsa_backward_collapseMH
#print axioms StableHLO.mhsaBackGraphMH_faithful
#print axioms StableHLO.mlpSublayerBackGraph_faithfulMH
#print axioms StableHLO.attnSublayerBackGraphMH_faithful
#print axioms StableHLO.transformerBlockBackGraphMH_faithful

-- ViT whole-block backward-graph faithfulness at the FULL PRODUCTION config
#print axioms StableHLO.rowVecLNBack_eq_backward
#print axioms StableHLO.mlpSublayerVBackGraph_faithfulMH
#print axioms StableHLO.attnSublayerVBackGraphMH_faithful
#print axioms StableHLO.transformerBlockVBackGraphMH_faithful

-- ViT WHOLE-NET backward-graph faithfulness
#print axioms StableHLO.classifierBackGraph_faithful
#print axioms StableHLO.finalLNBackGraph_faithful
#print axioms StableHLO.transformerBlockVBackGraphMHP_faithful
#print axioms StableHLO.vitBodyBackGraphKMHV_den
#print axioms StableHLO.patchEmbedBack_faithful
#print axioms StableHLO.patchEmbedBackGraph_faithful
#print axioms StableHLO.vitNetBackGraph_faithful

-- ViT folded onto the net-agnostic `CertLayer` machinery (ViTBackNet.lean, 2026-08-10)
#print axioms StableHLO.vitBlockVLayer
#print axioms StableHLO.vitTrunkV_eq_chain
#print axioms StableHLO.vitTrunkV_fwd
#print axioms StableHLO.vitTrunkV_graph
#print axioms StableHLO.vitTrunkV_faithful
#print axioms StableHLO.vitTinyTrunk_is_shipped

-- THE WHOLE NET AS ONE `CertLayer`
#print axioms StableHLO.vitPatchEmbedLayer
#print axioms StableHLO.vitFinalLNLayer
#print axioms StableHLO.vitClassifierLayer
#print axioms StableHLO.vitNetLayer_fwd
#print axioms StableHLO.vitNetLayer_graph
#print axioms StableHLO.vitNetLayer_ok
#print axioms StableHLO.vitNetBackGraph_faithful_via_fold

-- THE MACHINERY FIRST — `CertLayer.comp`, the single composition proof every fold routes through
#print axioms StableHLO.CertLayer.id'
#print axioms StableHLO.CertLayer.comp
#print axioms StableHLO.CertLayer.residual
#print axioms StableHLO.CertLayer.chain
#print axioms StableHLO.CertLayer.chain_fwd
#print axioms StableHLO.CertLayer.chain_faithful

-- R50 — the three bottleneck forms (identity / stride-1 projection / strided projection)
#print axioms StableHLO.r50BottleneckLayer
#print axioms StableHLO.r50ProjBlockLayer
#print axioms StableHLO.r50DownBlockLayer
#print axioms StableHLO.r50Stage_faithful
#print axioms StableHLO.r50Trunk_faithful
#print axioms StableHLO.r50Trunk_3463
#print axioms StableHLO.r50DownBlockOfRow

-- r34 / mnv2 / enet / convnext
#print axioms StableHLO.enetMBConvLayer
#print axioms StableHLO.cnxBlockLayer
#print axioms StableHLO.cnxBlockChLayer
#print axioms StableHLO.r34BasicBlockLayer
#print axioms StableHLO.r34DownBlockLayer
#print axioms StableHLO.mnv2ResidBlockLayer
#print axioms StableHLO.enetChain_faithful
#print axioms StableHLO.r34Trunk_3463
#print axioms StableHLO.r34DownBlockOfRow

-- MNv4 — the four UIB families COLLAPSED into one body
#print axioms StableHLO.dwbReluBackBatchedGraph_faithful
#print axioms StableHLO.dwbReluBstridedBackBatchedGraph_faithful
#print axioms StableHLO.mnv4UibSkipBlock_faithful
#print axioms StableHLO.mnv4UibPreStridedBody_faithful
#print axioms StableHLO.mnv4UibPostStridedBody_faithful
#print axioms StableHLO.stemBackBatchedGraph_faithful
#print axioms StableHLO.mnv4FusedStage_faithful
#print axioms StableHLO.mnv4Head_faithful
#print axioms StableHLO.mnv4BodyOfRow_faithful
#print axioms StableHLO.mnv4PreStridedBodyOfRow_faithful

-- MNv4's NET level (T1, T2)
#print axioms StableHLO.mobilenetv4ForwardB_full_has_vjp_at
#print axioms StableHLO.mobilenetv4ForwardB_full_has_vjp_at_correct
#print axioms StableHLO.mnv4FwdGraphB_full_faithful
#print axioms StableHLO.mnv4ExtraDWBodyGraphB_faithful
#print axioms StableHLO.mnv4ConvNeXtBodyGraphB_faithful
#print axioms StableHLO.mnv4FfnBodyGraphB_faithful
#print axioms StableHLO.mnv4PreStridedGraphB_faithful

-- MNv4's T3 §1 fold
#print axioms Mnv4PoCB.mnv4BnGradsCertified
#print axioms Mnv4PoCB.mnv4StemGradsCertified
#print axioms Mnv4PoCB.mnv4FusedGradsCertified
#print axioms Mnv4PoCB.mnv4ExtraDWGradsCertified
#print axioms Mnv4PoCB.mnv4ConvNeXtGradsCertified
#print axioms Mnv4PoCB.mnv4FfnGradsCertified
#print axioms Mnv4PoCB.mnv4PreStridedGradsCertified
#print axioms Mnv4PoCB.mnv4HeadGradsCertified

-- MNv4's T3 §1a tie
#print axioms Mnv4TieB.mnv4_extradw_tiedB
#print axioms Mnv4TieB.mnv4_convnext_tiedB
#print axioms Mnv4TieB.mnv4_ffn_tiedB
#print axioms Mnv4TieB.mnv4_prestrided_tiedB
#print axioms Mnv4TieB.mnv4_stem_tiedB
#print axioms Mnv4TieB.mnv4_fused_tiedB
#print axioms Mnv4TieB.mnv4_head_tiedB
#print axioms Mnv4TieB.mnv4_net_tiedB
#print axioms Mnv4TieB.mnv4_lossCot_is_smoothedCE_grad

-- MNv4's T6 -- the certified whole-net input gradient
#print axioms Proofs.mnv4StemBBack_eq_vjp_backward
#print axioms Proofs.cbReluBBack_eq_vjp_backward
#print axioms Proofs.mnv4B_full_has_vjp_at
#print axioms Proofs.mnv4InputGradB_eq_mnv4B_full_vjp
#print axioms Proofs.mnv4InputGradB_correct
#print axioms Proofs.mobilenetv4ForwardB_full_eq_slots

-- EfficientNet — §8e's VJP-without-backward-graph holes, closed
#print axioms StableHLO.enetMbExp_faithful
#print axioms StableHLO.enetMbNoExp_faithful
#print axioms StableHLO.enetMbStrided_faithful
#print axioms StableHLO.enetHead_faithful
#print axioms StableHLO.mbStridedFwdBackBatchedGraph_faithful
#print axioms StableHLO.mbExpFwdBackBatchedGraph_faithful
#print axioms StableHLO.mbNoExpBackBatchedGraph_faithful
#print axioms StableHLO.headBackBatchedGraph_faithful
#print axioms StableHLO.enetTrunk
-- ViT-Tiny §1 FOLD (ViTFaithfulPoC)
#print axioms Proofs.ViTPoC.veclnGammaSgd_den
#print axioms Proofs.ViTPoC.rowDenseWeightSgd_den
#print axioms Proofs.ViTPoC.rowDenseBiasSgd_den
#print axioms Proofs.ViTPoC.rowDenseBiasSgd_den_lnbeta
#print axioms Proofs.ViTPoC.patchEmbedWeightSgd_den
#print axioms Proofs.ViTPoC.patchEmbedBiasSgd_den
#print axioms Proofs.ViTPoC.posEmbedSgd_den
#print axioms Proofs.ViTPoC.headW_den
#print axioms Proofs.ViTPoC.headB_den
-- ViT-Tiny §1a TIE — per-block (ViTTiePoC)
#print axioms Proofs.ViTTiePoC.vit_block_tiedV
-- ViT-Tiny §1a TIE — whole-net thread (2-block vector-LN representative)
#print axioms Proofs.ViTTiePoC.vit_block_tiedAtV
#print axioms Proofs.ViTTiePoC.vit_net_tiedV
-- ViT-Tiny §1a TIE — MULTI-HEAD promotion (ViTMultiHeadChain + ViTTiePoC)
#print axioms Proofs.vitCotDQmh_eq
#print axioms Proofs.vitCotDKmh_eq
#print axioms Proofs.vitCotDVmh_eq
-- The multi-head per-block tie (vit_block_tiedMHV)
#print axioms Proofs.ViTTiePoC.vit_block_tiedMHV
#print axioms Proofs.ViTTiePoC.vit_block_tiedAtMHV
#print axioms Proofs.ViTTiePoC.vit_net_tiedMHV
-- ViT-Tiny §1a TIE — the ALL-200-PARAMS capstone (vit_net_tied_certified)
#print axioms Proofs.ViTTiePoC.vit_cls_den
#print axioms Proofs.ViTTiePoC.vit_finalLN_tied
#print axioms Proofs.ViTTiePoC.vit_head_tied
#print axioms Proofs.ViTTiePoC.vit_embed_tied
#print axioms Proofs.ViTTiePoC.vit_net_tied_certified

-- Robustness certificate (planning/archive/robustness_ladder.md, the cert side of cert ≤ TRUE ≤ PGD)
#print axioms Proofs.lipschitz_margin_certified_radius
#print axioms Proofs.logit_gap_stable
#print axioms Proofs.coord_pair_bound
#print axioms Proofs.euclid_norm_sq
#print axioms Proofs.LipschitzL2.comp
#print axioms Proofs.clm_lipschitzL2

-- Randomized-smoothing certified radius (Cohen–Rosenfeld–Kolter 2019)
#print axioms Proofs.smoothing_certified_radius
#print axioms Proofs.smoothed_margin_certified_radius

-- The smoothing radius at the REAL Gaussian quantile (SmoothingGaussian.lean, G1)
#print axioms Proofs.smoothing_certified_radius_probit
#print axioms Proofs.stdNormalCDF_strictMono
#print axioms Proofs.stdNormalCDF_neg
#print axioms Proofs.stdNormalQuantile_monotoneOn
#print axioms Proofs.stdNormalQuantile_anti
#print axioms Proofs.smoothing_certified_radius_gaussian

-- G2, the 1-D Neyman–Pearson core (smoothing_gaussian_lemma.md)
#print axioms Proofs.stdNormalCDF_quantile
#print axioms Proofs.gaussian_np_shift

-- G3, dimension reduction (smoothing_gaussian_lemma.md)
#print axioms Proofs.integral_gaussianReal_shift_eq
#print axioms Proofs.pi_gaussian_shift_eq
#print axioms Proofs.pi_gaussian_np_shift
#print axioms Proofs.stdGaussian_np_shift

-- G4, assembly — the Cohen radius with NOTHING left on the smoothing side
#print axioms Proofs.stdNormalQuantile_cdf
#print axioms Proofs.stdNormalCDF_mem_Ioo
#print axioms Proofs.smoothing_probit_lipschitz
#print axioms Proofs.smoothing_certified_radius_cohen
#print axioms Proofs.smoothing_certified_radius_classifier
-- ...and the MONTE-CARLO tie (SmoothingMC.lean)
#print axioms Proofs.iIndepFun_eval_pi
#print axioms Proofs.mc_mean_lower_bound
#print axioms Proofs.stdNormalQuantile_of_nonpos
#print axioms Proofs.smoothing_mc_certified

-- ...and the EXACT Clopper-Pearson tie (SmoothingCP.lean, 2026-07-12)
#print axioms Proofs.pi_hitCount_eq_binomial
#print axioms Proofs.pi_hitCount_tail_real
#print axioms Proofs.binomTail_le_of_lt_cpLower
#print axioms Proofs.cp_coverage
#print axioms Proofs.smoothing_cp_certified

-- ...and the SOLVED form (the per-image scorecard shape, 2026-07-12)
#print axioms Proofs.binomTail_monotoneOn
#print axioms Proofs.le_cpLower_of_tail_le
#print axioms Proofs.smoothing_cp_certified_solved
#print axioms Proofs.binomTail_check_99of100
#print axioms Proofs.binomTail_check_999of1000

-- ...and the KERNEL ENGINE for driver-scale tail checks (the ListDot recipe)
#print axioms Proofs.binomTailNum_eq
#print axioms Proofs.binomTail_eq_kernel
#print axioms Proofs.binomTail_le_of_kernel_check
#print axioms Proofs.binomTail_check_9900of10112
#print axioms Proofs.binomTail_check_5500of10112

-- ...and the SCORECARD (SmoothingCPScorecard.lean, generated)
#print axioms Proofs.smoothCpMlp_certified
#print axioms Proofs.smoothCpCnn_certified
#print axioms Proofs.smoothCpCifar_certified

-- ...and certified DECIMAL quantile bounds (SmoothingPhiBounds.lean)
#print axioms Proofs.stdNormalCDF_panel
#print axioms Proofs.stdNormalCDF_le_phiGridUB
#print axioms Proofs.le_stdNormalQuantile_of_grid
#print axioms Proofs.stdNormalQuantile_ge_of_09
#print axioms Proofs.stdNormalQuantile_ge_of_9952
#print axioms Proofs.smooth_cp_mlp_i1_radius_dec

-- ...and the DECIMAL-radius SCORECARD (the prefix-scan corpus pass)
#print axioms Proofs.phiScanRev_getD
#print axioms Proofs.phiScanRevFrom_append
#print axioms Proofs.le_stdNormalQuantile_of_scan
#print axioms Proofs.smooth_radius_dec
#print axioms Proofs.smoothDecMlp_certified
#print axioms Proofs.smoothDecCnn_certified
#print axioms Proofs.smoothDecCifar_certified

-- ...and the NET-SEMANTICS closure (SmoothingNetSemantics.lean)
#print axioms Proofs.measurable_argmaxNet
#print axioms Proofs.argmaxNet_smoothProb_mem_Ioo
#print axioms Proofs.smoothing_cp_certified_net
#print axioms Proofs.LipschitzCertDemo.mlpT_logit_continuous
#print axioms Proofs.LipschitzCertDemo.netW_strict
#print axioms Proofs.LipschitzCertDemo.smoothing_cp_certified_mlpT
#print axioms Proofs.LipschitzCertDemo.smooth_cp_mlpT_demo

-- ...and the two-sided quantile packaging (SmoothingGaussian.lean, 2026-07-12)
#print axioms Proofs.stdNormalQuantile_strictMonoOn
#print axioms Proofs.stdNormalQuantile_surjOn
#print axioms Proofs.stdNormalQuantile_continuousAt
#print axioms Proofs.stdNormalQuantile_continuousOn

-- The Mathlib upstreaming drafts (UpstreamDraft.lean, a Certs root)
#print axioms MathlibUpstream.strictMono_cdf_iff
#print axioms MathlibUpstream.continuous_cdf_iff
#print axioms MathlibUpstream.cdf_mem_Ioo
#print axioms MathlibUpstream.cdf_gaussianReal_mem_Ioo
#print axioms MathlibUpstream.cdf_gaussianReal_neg
#print axioms MathlibUpstream.cdf_gaussianReal_sub_const

-- ...and the Tsuzuku certificate INSTANTIATED (LipschitzCertInstance.lean)
#print axioms Proofs.LipschitzCertDemo.denseE_lipschitzL2
#print axioms Proofs.LipschitzCertDemo.reluE_lipschitzL2
#print axioms Proofs.LipschitzCertDemo.linear_demo_certified
#print axioms Proofs.LipschitzCertDemo.linear_radius_pos
#print axioms Proofs.LipschitzCertDemo.mlp_lip
#print axioms Proofs.LipschitzCertDemo.mlp_demo_certified
#print axioms Proofs.LipschitzCertDemo.mlp_radius_pos
#print axioms Proofs.LipschitzCertDemo.W1t_lip
#print axioms Proofs.LipschitzCertDemo.W2t_lip
#print axioms Proofs.LipschitzCertDemo.mlpT_lip
#print axioms Proofs.LipschitzCertDemo.xt_margin
#print axioms Proofs.LipschitzCertDemo.trained_radius_pos
#print axioms Proofs.LipschitzCertDemo.trained_demo_certified
-- ...tightened by the power-iteration certificate (certified two-sided spectral sandwich)
#print axioms Proofs.LipschitzCertDemo.sum_sq_matvec_le
#print axioms Proofs.LipschitzCertDemo.denseE_lipschitzL2_gram
#print axioms Proofs.LipschitzCertDemo.lipschitzL2_lower_euclid
#print axioms Proofs.LipschitzCertDemo.G1t_eq
#print axioms Proofs.LipschitzCertDemo.G2t_eq
#print axioms Proofs.LipschitzCertDemo.W1t_lip_gram
#print axioms Proofs.LipschitzCertDemo.W2t_lip_gram
#print axioms Proofs.LipschitzCertDemo.mlpT_lip_gram
#print axioms Proofs.LipschitzCertDemo.trained_radius_gram_pos
#print axioms Proofs.LipschitzCertDemo.trained_demo_certified_gram
#print axioms Proofs.LipschitzCertDemo.W1t_lip_lower
#print axioms Proofs.LipschitzCertDemo.W2t_lip_lower
-- ...iterated once more (Schatten-8)
#print axioms Proofs.LipschitzCertDemo.sum_sq_matTvec_eq
#print axioms Proofs.LipschitzCertDemo.denseE_lipschitzL2_gram2
#print axioms Proofs.LipschitzCertDemo.H1t_eq
#print axioms Proofs.LipschitzCertDemo.H2t_eq
#print axioms Proofs.LipschitzCertDemo.W1t_lip_gram2
#print axioms Proofs.LipschitzCertDemo.W2t_lip_gram2
#print axioms Proofs.LipschitzCertDemo.mlpT_lip_gram2
#print axioms Proofs.LipschitzCertDemo.trained_radius_gram2_pos
#print axioms Proofs.LipschitzCertDemo.trained_demo_certified_gram2

-- CERTIFIED-ACCURACY SCORECARD (LipschitzCertScorecard.lean, post_audit_roadmap §1)
#print axioms Proofs.LipschitzCertDemo.sqrt_two_le_rat
#print axioms Proofs.LipschitzCertDemo.certified_at_eps
#print axioms Proofs.LipschitzCertDemo.G1s_eq
#print axioms Proofs.LipschitzCertDemo.G2s_eq
#print axioms Proofs.LipschitzCertDemo.H1s_eq
#print axioms Proofs.LipschitzCertDemo.H2s_eq
#print axioms Proofs.LipschitzCertDemo.W1s_lip_gram2
#print axioms Proofs.LipschitzCertDemo.W2s_lip_gram2
#print axioms Proofs.LipschitzCertDemo.mlpS_lip_gram2
#print axioms Proofs.LipschitzCertDemo.marginC0
#print axioms Proofs.LipschitzCertDemo.certifiedC0
#print axioms Proofs.LipschitzCertDemo.certifiedC3
#print axioms Proofs.LipschitzCertDemo.certifiedC10
#print axioms Proofs.LipschitzCertDemo.certifiedC13
#print axioms Proofs.LipschitzCertDemo.certifiedC14
#print axioms Proofs.LipschitzCertDemo.certifiedC17
#print axioms Proofs.LipschitzCertDemo.certifiedC25
#print axioms Proofs.LipschitzCertDemo.marginU82
#print axioms Proofs.LipschitzCertDemo.certifiedU82
-- the mechanized aggregate
#print axioms Proofs.LipschitzCertDemo.cappedCerts_idx
#print axioms Proofs.LipschitzCertDemo.unconCerts_idx
#print axioms Proofs.LipschitzCertDemo.cappedCerts_certified
#print axioms Proofs.LipschitzCertDemo.unconCerts_certified
#print axioms Proofs.LipschitzCertDemo.scorecard
#print axioms Proofs.LipschitzCertDemo.scorecard_counts

-- Per-pair LipSDP tightening (LipschitzCertPairSDP.lean + the generated instances)
#print axioms Proofs.LipschitzCertDemo.relu_slope_restricted
#print axioms Proofs.LipschitzCertDemo.quad_form_nonneg_of_ldl
#print axioms Proofs.LipschitzCertDemo.lipsdp_slack_of_cert
#print axioms Proofs.LipschitzCertDemo.pair_sq_bound
#print axioms Proofs.LipschitzCertDemo.mlp_gap_eq
#print axioms Proofs.LipschitzCertDemo.certified_at_eps_pair
#print axioms Proofs.LipschitzCertDemo.hS01C
#print axioms Proofs.LipschitzCertDemo.pairSqC_0_1
#print axioms Proofs.LipschitzCertDemo.pairSqC_7_0
#print axioms Proofs.LipschitzCertDemo.pairSqU_0_1
#print axioms Proofs.LipschitzCertDemo.certifiedSC0
#print axioms Proofs.LipschitzCertDemo.certifiedSC4
#print axioms Proofs.LipschitzCertDemo.certifiedSC9
#print axioms Proofs.LipschitzCertDemo.certifiedSU0
#print axioms Proofs.LipschitzCertDemo.certifiedSU4
#print axioms Proofs.LipschitzCertDemo.certifiedSU11
#print axioms Proofs.LipschitzCertDemo.sdpCappedCerts_certified
#print axioms Proofs.LipschitzCertDemo.sdpUnconCerts_certified
#print axioms Proofs.LipschitzCertDemo.scorecard_sdp
#print axioms Proofs.LipschitzCertDemo.scorecard_sdp_uncon

-- The certificate × float bridge (LipschitzCertFloat.lean, 2026-07 audit gap #1)
#print axioms Proofs.FloatModel.mlp2_float_close_uniform
#print axioms Proofs.LipschitzCertDemo.certified_at_eps_close
#print axioms Proofs.LipschitzCertDemo.capped_B_le
#print axioms Proofs.LipschitzCertDemo.real_tie
#print axioms Proofs.LipschitzCertDemo.certifiedFloat_of_margin
#print axioms Proofs.LipschitzCertDemo.certifiedC0_float
#print axioms Proofs.LipschitzCertDemo.float_scorecard_count

-- The kernel-dotZ list engine (ListDot.lean)
#print axioms Proofs.dotZ_comm
#print axioms Proofs.sum_getD_mul
#print axioms Proofs.sum_getD_div
#print axioms Proofs.sum_getD_abs
#print axioms Proofs.sum_getD_abs_div
#print axioms Proofs.LipschitzCertDemo.denseLo_le
#print axioms Proofs.LipschitzCertDemo.le_denseHi
#print axioms Proofs.LipschitzCertDemo.relu_box
#print axioms Proofs.LipschitzCertDemo.denseLo_uniform
#print axioms Proofs.LipschitzCertDemo.denseLo2_eval
#print axioms Proofs.LipschitzCertDemo.denseHi2_eval
-- The dense tier's certificate is stated on a BRACKET, not on interval arithmetic
#print axioms Proofs.LipschitzCertDemo.BoxSoundE.comp
#print axioms Proofs.LipschitzCertDemo.denseE_boxSound
#print axioms Proofs.LipschitzCertDemo.reluE_boxSound
#print axioms Proofs.LipschitzCertDemo.certified_of_boxSound
#print axioms Proofs.LipschitzCertDemo.mlp2_boxSound
#print axioms Proofs.LipschitzCertDemo.ibp2_certified_at_eps

-- CROWN (Foundation/CrownBound.lean)
#print axioms Proofs.LipschitzCertDemo.certified_of_marginPos
#print axioms Proofs.LipschitzCertDemo.relu_lower_envelope
#print axioms Proofs.LipschitzCertDemo.relu_upper_envelope
#print axioms Proofs.LipschitzCertDemo.reluLB_dead
#print axioms Proofs.LipschitzCertDemo.reluLB_active
#print axioms Proofs.LipschitzCertDemo.reluLB_unstable_pos
#print axioms Proofs.LipschitzCertDemo.reluLB_unstable_neg
#print axioms Proofs.LipschitzCertDemo.crownRow_dot
#print axioms Proofs.LipschitzCertDemo.linf_lower_bound
#print axioms Proofs.LipschitzCertDemo.crown_margin_ge
#print axioms Proofs.LipschitzCertDemo.crown2_certified_at_eps

-- IBP PAST THE TWO-LAYER DENSE WALL (Foundation/IntervalBoundConv.lean)
#print axioms Proofs.IBP.BoxSound.comp
#print axioms Proofs.IBP.BoxSound3.comp
#print axioms Proofs.IBP.BoxSound3V.comp3
#print axioms Proofs.IBP.denseV_boxSound
#print axioms Proofs.IBP.denseT_boxSound3V
#print axioms Proofs.IBP.reluV_boxSound
#print axioms Proofs.IBP.reluT_boxSound3
#print axioms Proofs.IBP.flatten_reluT
#print axioms Proofs.IBP.conv2d_boxSound3
#print axioms Proofs.IBP.flatConv_boxSound
#print axioms Proofs.IBP.maxPool2_boxSound3
#print axioms Proofs.IBP.maxPoolFlat_boxSound
-- the conv peer of denseLo_uniform
#print axioms Proofs.IBP.convLo_uniform
#print axioms Proofs.IBP.convHi_uniform
-- DEPTH, concretely
#print axioms Proofs.IBP.deepNet_boxSound
-- capstones (flat and tensor shape) + radius monotonicity
#print axioms Proofs.IBP.ibp_certified_of_boxSound
#print axioms Proofs.IBP.ibp3_certified_of_boxSound
#print axioms Proofs.IBP.CertifiedAtLinf3.mono
#print axioms Proofs.IBP.CertifiedAtLinfV.mono


-- THE IEEE AXIOMS, DISCHARGED (Binary32Instance.lean, post_audit_roadmap §2)
#print axioms Proofs.rndP_err
#print axioms Proofs.binary32_e4m3_argmax_preserved
#print axioms Proofs.binary32_e4m3_argmax_small
#print axioms Proofs.binary32_linear_sgd_descends_concrete

-- DESCENT AT TRAINED WEIGHTS (TrainedLinearDescent.lean, post_audit_roadmap §3)
#print axioms Proofs.TrainedLinearDescent.hz_lbl_le
#print axioms Proofs.TrainedLinearDescent.sm_lbl_le_half
#print axioms Proofs.TrainedLinearDescent.gradL1_le
#print axioms Proofs.TrainedLinearDescent.gradSq_lower
#print axioms Proofs.TrainedLinearDescent.hdelta
#print axioms Proofs.TrainedLinearDescent.heta
#print axioms Proofs.TrainedLinearDescent.trained_linear_sgd_descends_concrete
#print axioms Proofs.TrainedLinearDescent.trained_linear_sgd_strictly_descends

-- TRAINED-WEIGHT whole-net VJP witness (TrainedMlpWitness.lean)
#print axioms Proofs.TrainedMlp.preact_eq
#print axioms Proofs.TrainedMlp.preact_ne
#print axioms Proofs.TrainedMlp.trainedMlp_has_vjp_correct
#print axioms Proofs.TrainedMlp.pdiv_fwd
#print axioms Proofs.TrainedMlp.pdiv_fwd_val
#print axioms Proofs.TrainedMlp.trainedMlp_backward_nontrivial
#print axioms Proofs.TrainedMlp.trainedMlp_jacobian_nonzero
#print axioms Proofs.TrainedMlp.trainedMlp_not_constant

-- TRAINED-WEIGHT whole-net VJP witness, CNN rung (TrainedCnnWitness.lean, post_audit gap #3)
#print axioms Proofs.TrainedCnn.conv1_eq
#print axioms Proofs.TrainedCnn.conv2_eq
#print axioms Proofs.TrainedCnn.r2_smooth
#print axioms Proofs.TrainedCnn.pooled_eq
#print axioms Proofs.TrainedCnn.d3_ne
#print axioms Proofs.TrainedCnn.d4_ne
#print axioms Proofs.TrainedCnn.trainedCnn_has_vjp_correct

-- Level-3 seal for the CNN witness (TrainedCnnSeal.lean)
#print axioms Proofs.TrainedCnn.S2
#print axioms Proofs.TrainedCnn.S1
#print axioms Proofs.TrainedCnn.pdiv_fwd_entry
#print axioms Proofs.TrainedCnn.pdiv_fwd_entry_ne
#print axioms Proofs.TrainedCnn.trainedCnn_backward_nontrivial
#print axioms Proofs.TrainedCnn.trainedCnn_jacobian_nonzero
#print axioms Proofs.TrainedCnn.trainedCnn_not_constant

-- Muon geometry (planning/archive/muon_geometry.md)
#print axioms Proofs.MuonGeometry.steepest_l2_bound
#print axioms Proofs.MuonGeometry.steepest_l2_attained
#print axioms Proofs.MuonGeometry.steepest_linf_bound
#print axioms Proofs.MuonGeometry.steepest_linf_attained
#print axioms Proofs.MuonGeometry.muon_polar_achieves_nuclear
-- L3 upper bound (von Neumann's trace inequality)
#print axioms Proofs.MuonGeometry.muon_polar_is_max
#print axioms Proofs.MuonGeometry.muon_polar_steepest
#print axioms Proofs.MuonGeometry.svd_of_isUnit
#print axioms Proofs.MuonGeometry.muon_polar_achieves_nuclear_of_isUnit
-- L5 the jewel: single-step Shampoo (GGᵀ)
#print axioms Proofs.MuonGeometry.conj_diag_pow
#print axioms Proofs.MuonGeometry.shampoo_eq_muon
#print axioms Proofs.MuonGeometry.shampoo_eq_muon_of_isUnit
-- L6 manifold view
#print axioms Proofs.MuonGeometry.muon_polar_orthogonal
#print axioms Proofs.MuonGeometry.muon_polar_nearest_orthogonal
-- Newton–Schulz P1 (planning/archive/muon_ns_convergence.md)
#print axioms Proofs.MuonNewtonSchulz.nsStep_spectral
#print axioms Proofs.MuonNewtonSchulz.nsStep_iterate_spectral
-- Newton–Schulz P2 (the scalar engine)
#print axioms Proofs.MuonNewtonSchulz.scalar_iterate_tendsto_one
#print axioms Proofs.MuonNewtonSchulz.gCubic_eq_nsScalar
#print axioms Proofs.MuonNewtonSchulz.gCubic_iterate_tendsto_one
#print axioms Proofs.MuonNewtonSchulz.q5Scalar_eq_nsScalar
#print axioms Proofs.MuonNewtonSchulz.q5Scalar_iterate_tendsto_one
-- Newton–Schulz P3 (CLOSES THE LOOP)
#print axioms Proofs.MuonNewtonSchulz.nsStep_iterate_tendsto_polar
#print axioms Proofs.MuonNewtonSchulz.nsStep_cubic_iterate_tendsto_polar
#print axioms Proofs.MuonNewtonSchulz.nsStep_q5_iterate_tendsto_polar
-- Newton–Schulz P4 (the HONEST tier)
#print axioms Proofs.MuonNewtonSchulz.qScalar_one_lt_one
#print axioms Proofs.MuonNewtonSchulz.qScalar_half_gt_one
#print axioms Proofs.MuonNewtonSchulz.qScalar_not_le_one
#print axioms Proofs.MuonNewtonSchulz.qScalar_iterate_band_half

-- Spec→math ties (SpecVJP.lean, rungs B/C/E)
#print axioms linearVerified_denote_eq
#print axioms linearVerified_has_vjp_correct
#print axioms linearVerified_fwd_faithful
#print axioms linearVerified_lossCot_isCEgrad
#print axioms mlpVerified_denote_eq
#print axioms mlpVerified_has_vjp_correct
#print axioms mlpVerified_fwd_faithful
#print axioms mlpVerified_back_faithful
#print axioms cnnVerified_denote_eq
#print axioms cnnVerified_fwd_faithful
#print axioms cifarVerified_denote_eq
#print axioms cifarVerified_fwd_faithful
#print axioms cifarBnVerified_denote_eq
#print axioms cifarBnVerified_fwd_faithful
#print axioms mobilenetv2Rep_denote_eq
#print axioms mobilenetv2Rep_fwd_faithful
-- mnv2 FULL-paper tie (the committed 21-entry spec ↔ mobilenetv2ForwardPaper)
#print axioms mobilenetv2Verified_denote_eq
#print axioms mobilenetv2Verified_fwd_faithful
#print axioms efficientnetRep_denote_eq
#print axioms convnextRep_denote_eq
#print axioms convnextRep_fwd_faithful
#print axioms vitRep_denote_eq
#print axioms r34Rep_denote_eq
-- FULL committed-spec ties (unified weight bundles, 2026-07-07)
#print axioms resnet34Verified_denote_eq
#print axioms resnet34Verified_fwd_faithful
#print axioms efficientnetVerified_denote_eq
#print axioms efficientnetVerified_fwd_faithful
#print axioms convnextVerified_denote_eq
#print axioms convnextVerified_fwd_faithful
#print axioms vitVerified_denote_eq
#print axioms vitVerified_has_vjp
#print axioms vitVerified_fwd_faithful

-- The CANONICAL MNIST MLP surface (MlpCanonical.lean, 2026-07-07)
#print axioms Proofs.MlpCanonical.has_vjp_at
#print axioms Proofs.MlpCanonical.has_vjp_correct
#print axioms Proofs.MlpCanonical.output_float_sgd_descends
#print axioms Proofs.MlpCanonical.hidden_float_sgd_descends
#print axioms Proofs.MlpCanonical.input_float_sgd_descends
#print axioms Proofs.MlpCanonical.w1_grad_close
#print axioms Proofs.MlpCanonical.w0_grad_close
#print axioms Proofs.MlpCanonical.inputGrad_floatBridges
#print axioms Proofs.MlpCanonical.train_step_tied_certified

-- The bf16-MIXED conv, composed (ConvMixedComposeBridge.lean, 2026-08-24)
#print axioms Proofs.convFanS_le
#print axioms Proofs.conv2d_sub_abs_le
#print axioms Proofs.FloatModel.convMixed_close_prop
#print axioms Proofs.FloatModel.flatConvMixed_close
#print axioms Proofs.floatClose_flatConvMixed
#print axioms Proofs.floatClose_reluConvMixed
#print axioms Proofs.floatClose_convMixed_twice
#print axioms Proofs.floatClose_r50_stages_mixed
#print axioms Proofs.convMixedBudget_affine
#print axioms Proofs.layerBudget_affine

-- The bf16-mixed DEPTHWISE (DepthwiseMixedFloatBridge.lean, 2026-08-24)
#print axioms Proofs.depthwiseConv2d_eq_dw_dot
#print axioms Proofs.FloatModel.depthwise_close_mixed


-- RESNET-34 AT TRUE BATCH BN — T1-forward and T2 (ResNet34FullB.lean, 2026-09-06)
#print axioms Proofs.resnet34ForwardB_full
#print axioms Proofs.StableHLO.r34IdGraphB_faithful
#print axioms Proofs.StableHLO.r34DownGraphB_faithful
#print axioms Proofs.StableHLO.r34StemGraphB_faithful
#print axioms Proofs.StableHLO.r34HeadGraphB_faithful
#print axioms Proofs.StableHLO.resnet34FwdGraphB_full_faithful

-- `batchMap` AT A POINT (BatchMapVJPAt.lean, 2026-09-06)
#print axioms Proofs.pdivMat_rowIndep_at
#print axioms Proofs.batchMap_differentiableAt
#print axioms Proofs.pdiv_batchMap_at
#print axioms Proofs.batchMap_has_vjp_at

-- RESNET-34 AT TRUE BATCH BN — T1's VJP half (ResNet34FullBVJP.lean, 2026-09-06)
#print axioms Proofs.r34IdB_has_vjp_at
#print axioms Proofs.r34DownB_has_vjp_at
#print axioms Proofs.r34StemB_has_vjp_at
#print axioms Proofs.r34HeadB_has_vjp
#print axioms Proofs.resnet34ForwardB_full_has_vjp_at
#print axioms Proofs.resnet34ForwardB_full_eq_chain
#print axioms Proofs.resnet34ForwardB_full_has_vjp_at_correct

-- RESNET-34 AT TRUE BATCH BN — T3's §1 fold, UN-FUSED (ResNet34FaithfulPoCB.lean, 2026-09-06)
#print axioms Proofs.ResNet34PoCB.convWGradB_den
#print axioms Proofs.ResNet34PoCB.convBGradB_den
#print axioms Proofs.ResNet34PoCB.convStridedWGradB_den
#print axioms Proofs.ResNet34PoCB.convStridedBGradB_den
#print axioms Proofs.ResNet34PoCB.bnGammaGradB_den
#print axioms Proofs.ResNet34PoCB.bnBetaGradB_den
#print axioms Proofs.ResNet34PoCB.denseWGradB_den
#print axioms Proofs.ResNet34PoCB.denseBGradB_den

-- 4b.1 EfficientNet-B0
#print axioms Proofs.EnetPoCG.convWGradB_den
#print axioms Proofs.EnetPoCG.bnGammaGradB_den
#print axioms Proofs.EnetPoCG.bnBetaGradB_den
#print axioms Proofs.EnetPoCG.denseWGradB_den
#print axioms Proofs.EnetPoCG.denseBGradB_den
#print axioms Proofs.EnetPoCG.convStridedXlaWGradB_den
#print axioms Proofs.EnetPoCG.depthwiseWGradB_den
#print axioms Proofs.EnetPoCG.depthwiseStridedWGradB_den

-- 4b.2 ConvNeXt-T
#print axioms Proofs.CnxPoCG.layerScaleChGammaGrad_den
#print axioms Proofs.CnxPoCG.convWGrad_den
#print axioms Proofs.CnxPoCG.convBGrad_den
#print axioms Proofs.CnxPoCG.depthwiseWGrad_den
#print axioms Proofs.CnxPoCG.depthwiseBGrad_den
#print axioms Proofs.CnxPoCG.convStridedWGrad_den
#print axioms Proofs.CnxPoCG.convStridedBGrad_den
#print axioms Proofs.CnxPoCG.psWGrad_den
#print axioms Proofs.CnxPoCG.chanLnGammaGrad_den
#print axioms Proofs.CnxPoCG.chanLnBetaGrad_den
#print axioms Proofs.CnxPoCG.headLnGammaGrad_den
#print axioms Proofs.CnxPoCG.headLnBetaGrad_den
#print axioms Proofs.CnxPoCG.headWGrad_den
#print axioms Proofs.CnxPoCG.headBGrad_den

-- 4b.3 ViT-Tiny — vit_adam_train_step and vitin_adamdp128x4wxclipdrop
#print axioms Proofs.ViTPoCG.veclnGammaGrad_den
#print axioms Proofs.ViTPoCG.rowDenseBiasGrad_den_lnbeta
#print axioms Proofs.ViTPoCG.rowDenseWeightGrad_den
#print axioms Proofs.ViTPoCG.rowDenseBiasGrad_den
#print axioms Proofs.ViTPoCG.patchEmbedWeightGrad_den
#print axioms Proofs.ViTPoCG.patchEmbedBiasGrad_den
#print axioms Proofs.ViTPoCG.posEmbedGrad_den
#print axioms Proofs.ViTPoCG.clsGrad_den
#print axioms Proofs.ViTPoCG.headWGrad_den
#print axioms Proofs.ViTPoCG.headBGrad_den

-- 4c leg 4 ViT-Tiny
#print axioms Proofs.ViTPoCGB.veclnGammaGradB_den
#print axioms Proofs.ViTPoCGB.rowDenseBiasGradB_den_lnbeta
#print axioms Proofs.ViTPoCGB.rowDenseWeightGradB_den
#print axioms Proofs.ViTPoCGB.rowDenseBiasGradB_den
#print axioms Proofs.ViTPoCGB.patchEmbedWeightGradB_den
#print axioms Proofs.ViTPoCGB.patchEmbedBiasGradB_den
#print axioms Proofs.ViTPoCGB.posEmbedGradB_den
#print axioms Proofs.ViTPoCGB.clsGrad_denB
#print axioms Proofs.ViTPoCGB.headWGradB_den
#print axioms Proofs.ViTPoCGB.headBGradB_den

-- 4c leg 3 ConvNeXt-T
#print axioms Proofs.CnxPoCGB.layerScaleChGammaGradB_den
#print axioms Proofs.CnxPoCGB.convWGradB_den
#print axioms Proofs.CnxPoCGB.convBGradB_den
#print axioms Proofs.CnxPoCGB.depthwiseWGradB_den
#print axioms Proofs.CnxPoCGB.depthwiseBGradB_den
#print axioms Proofs.CnxPoCGB.convStridedWGradB_den
#print axioms Proofs.CnxPoCGB.convStridedBGradB_den
#print axioms Proofs.CnxPoCGB.psWGradB_den
#print axioms Proofs.CnxPoCGB.chanLnGammaGradB_den
#print axioms Proofs.CnxPoCGB.chanLnBetaGradB_den
#print axioms Proofs.CnxPoCGB.headLnGammaGradB_den
#print axioms Proofs.CnxPoCGB.headLnBetaGradB_den
#print axioms Proofs.CnxPoCGB.headWGradB_den
#print axioms Proofs.CnxPoCGB.headBGradB_den
-- The bf16 gradient nodes, folded ONCE for every net (Bf16GradNodes.lean)
#print axioms Proofs.Bf16PoC.convWGradBBf16_den
#print axioms Proofs.Bf16PoC.convStridedWGradBBf16_den
#print axioms Proofs.Bf16PoC.convStridedXlaWGradBBf16_den
#print axioms Proofs.Bf16PoC.convStride4WGradBBf16_den
#print axioms Proofs.Bf16PoC.depthwiseWGradBBf16_den
#print axioms Proofs.Bf16PoC.depthwiseStridedWGradBBf16_den
#print axioms Proofs.Bf16PoC.depthwiseStridedXlaWGradBBf16_den
#print axioms Proofs.Bf16PoC.rowDenseWGradBBf16_den
#print axioms Proofs.Bf16PoC.patchEmbedWGradBBf16_den

-- 4b.4 MobileNetV2 at 17 blocks
#print axioms Proofs.Mnv2PaperPoCG.convStridedXlaBGradB_den
#print axioms Proofs.Mnv2PaperPoCG.depthwiseBGradB_den
#print axioms Proofs.Mnv2PaperPoCG.depthwiseStridedXlaWGradB_den
#print axioms Proofs.Mnv2PaperPoCG.depthwiseStridedXlaBGradB_den
#print axioms Proofs.Mnv2PaperPoCG.mnv2StemGradsCertified
#print axioms Proofs.Mnv2PaperPoCG.mnv2NoExpGradsCertified
#print axioms Proofs.Mnv2PaperPoCG.mnv2Stride1GradsCertified
#print axioms Proofs.Mnv2PaperPoCG.mnv2Stride2GradsCertified
#print axioms Proofs.Mnv2PaperPoCG.mnv2HeadDenseGradsCertified

-- 4.2a: THE LABEL-SMOOTHED LOSS COTANGENT, AT A GENERAL TARGET (SmoothedLossCot.lean, 2026-09-06)
#print axioms Proofs.softCE_oneHot
#print axioms Proofs.softCE_grad
#print axioms Proofs.smoothTarget_sum
#print axioms Proofs.smoothedCE_grad
#print axioms Proofs.smoothedLossCotGraph_den
#print axioms Proofs.smoothedLossCotGraph_row

-- 4.2a: RESNET-34'S T3 §1a TIE AT BATCH BN, UN-FUSED (ResNet34TiePoCB.lean, 2026-09-06)
#print axioms Proofs.ResNet34TieB.bnInB_eq_bnBackB
#print axioms Proofs.ResNet34TieB.r34IdCotIn_eq_vjp
#print axioms Proofs.ResNet34TieB.r34DownCotIn_eq_vjp
#print axioms Proofs.ResNet34TieB.r34_idblock_tiedB
#print axioms Proofs.ResNet34TieB.r34_downblock_tiedB
#print axioms Proofs.ResNet34TieB.r34_stem_tiedB
#print axioms Proofs.ResNet34TieB.r34_head_tiedB
#print axioms Proofs.ResNet34TieB.r34_net_tiedB
#print axioms Proofs.ResNet34TieB.r34_lossCot_is_smoothedCE_grad

-- 4.2 leg 1: MOBILENETV2 AT TRUE BATCH BN — T1-forward and T2 (MobileNetV2FullB.lean, 2026-09-06)
#print axioms Proofs.mobilenetv2ForwardB_full
#print axioms Proofs.StableHLO.mnv2StemGraphB_faithful
#print axioms Proofs.StableHLO.mnv2NoExpGraphB_faithful
#print axioms Proofs.StableHLO.mnv2ExpOnlyGraphB_faithful
#print axioms Proofs.StableHLO.mnv2ResidGraphB_faithful
#print axioms Proofs.StableHLO.mnv2StridedGraphB_faithful
#print axioms Proofs.StableHLO.mnv2HeadGraphB_faithful
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphB_full_faithful

-- 4.2 leg 1: MOBILENETV2 AT TRUE BATCH BN — T1's VJP half (MobileNetV2FullBVJP.lean, 2026-09-06)
#print axioms Proofs.mnv2StemB_has_vjp_at
#print axioms Proofs.mnv2NoExpB_has_vjp_at
#print axioms Proofs.mnv2ExpOnlyB_has_vjp_at
#print axioms Proofs.mnv2ResidB_has_vjp_at
#print axioms Proofs.mnv2StridedB_has_vjp_at
#print axioms Proofs.mnv2HeadB_has_vjp_at
#print axioms Proofs.mobilenetv2ForwardB_full_has_vjp_at
#print axioms Proofs.mobilenetv2ForwardB_full_eq_chain
#print axioms Proofs.mobilenetv2ForwardB_full_has_vjp_at_correct

-- 4.2c: MOBILENETV2'S T3 §1a TIE AT BATCH BN, UN-FUSED (MobileNetV2TiePoCB.lean, 2026-09-06)
#print axioms Proofs.MobileNetV2TieB.mnv2NoExpBackGraph_faithful
#print axioms Proofs.MobileNetV2TieB.mnv2NoExpCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2ExpOnlyCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2ResidCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2StridedCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2_stem_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_noexp_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_stride1_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_stride2_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_head_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_net_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_lossCot_is_smoothedCE_grad

-- 4b's CAPSTONE RE-POINTING, EFFICIENTNET-B0 (EfficientNetTiePoCG.lean, 2026-09-06)
#print axioms Proofs.EnetTiePoCG.enet_exp_tiedG
#print axioms Proofs.EnetTiePoCG.enet_strided_tiedG
#print axioms Proofs.EnetTiePoCG.enet_noexp_tiedG
#print axioms Proofs.EnetTiePoCG.enet_stem_tiedG
#print axioms Proofs.EnetTiePoCG.enet_head_tiedG
#print axioms Proofs.EnetTiePoCG.efficientnet_net_tiedG

-- 4b's CAPSTONE RE-POINTING, CONVNEXT-T (ConvNeXtTiePoCGB.lean, 2026-09-07)
#print axioms Proofs.smoothedLossCotGraphDiv_den
#print axioms Proofs.smoothedLossCotGraphDiv_row
#print axioms Proofs.CnxTiePoCGB.cnx_block_ch_tiedGB
#print axioms Proofs.CnxTiePoCGB.cnx_down_ch_tiedGB
#print axioms Proofs.CnxTiePoCGB.cnx_stem_ch_tiedGB
#print axioms Proofs.CnxTiePoCGB.cnx_head_ch_tiedGB
#print axioms Proofs.CnxTiePoCGB.cnx_net_tiedGB

-- 4b's CAPSTONE RE-POINTING, ViT-TINY (ViTTiePoCGB.lean, 2026-09-07) — FIVE OF FIVE
#print axioms Proofs.ViTTiePoCGB.vit_block_tiedGB
#print axioms Proofs.ViTTiePoCGB.vit_finalLN_tiedGB
#print axioms Proofs.ViTTiePoCGB.vit_head_tiedGB
#print axioms Proofs.ViTTiePoCGB.vit_embed_tiedGB
#print axioms Proofs.ViTTiePoCGB.vit_net_tiedGB

-- 4d PIECE 1: DATA PARALLELISM -- WHAT FUNCTION A *dp* RUN MINIMISED (DataParallel.lean, 2026-09-06)
#print axioms Proofs.pdiv_const_smul
#print axioms Proofs.meanLoss_differentiableAt
#print axioms Proofs.lossGrad_meanLoss
#print axioms Proofs.dpMeanGrad_eq_grad_meanLoss
#print axioms Proofs.meanLoss_shard
#print axioms Proofs.dpMeanGrad_eq_globalBatchGrad_of_perExample
#print axioms Proofs.dpMeanGrad_eq_globalBatchGrad_contiguous
#print axioms Proofs.dpToyShard_eq_batch
#print axioms Proofs.lossGrad_smul_coord
#print axioms Proofs.lossGrad_bnToyLoss
#print axioms Proofs.dpMeanGrad_ne_globalBatchGrad
#print axioms Proofs.dpStep_const
#print axioms Proofs.dpIterate_lockstep
#print axioms Proofs.dpSingleStep_eq_meanLoss_step
#print axioms Proofs.dpIterate_eq_meanLossTrain

-- 4d PIECE 2: THE COLLECTIVE AS AN AST NODE (StableHLO.allReduceMeanF, DataParallelNode.lean, 2026-09-07)
#print axioms Proofs.StableHLO.den_allReduceMeanF
#print axioms Proofs.StableHLO.roundtrip
#print axioms Proofs.den_allReduceMeanF_eq_dpMean
#print axioms Proofs.skel_allReduceMeanF_of_spmd
#print axioms Proofs.den_allReduceMeanF_eq_lossGrad_meanLoss
#print axioms Proofs.den_allReduceMeanF_convWeightGradB
#print axioms Proofs.adamW_at_allReduceMeanF

-- RESNET-50's TWO PREREQUISITES: THE LAMB TRIPLE AND BCE'S COTANGENT (2026-09-06)
#print axioms Proofs.lambStep
#print axioms Proofs.lambScale_zero_weight
#print axioms Proofs.StableHLO.lamb_triple_faithful
#print axioms Proofs.StableHLO.lamb_triple_faithful_committed
#print axioms Proofs.StableHLO.lamb_triple_faithful_excluded
#print axioms Proofs.softplus_hasDerivAt
#print axioms Proofs.softplus_neg
#print axioms Proofs.log_sigmoidScalar
#print axioms Proofs.one_sub_sigmoidScalar
#print axioms Proofs.bceLogits_eq_logSigmoid
#print axioms Proofs.pdiv_coordFun
#print axioms Proofs.bceLogits_grad
#print axioms Proofs.bceLossCotGraph_den
#print axioms Proofs.bceLossCotGraph_row
#print axioms Proofs.bceLossCotGraph_row_committed

-- §3.5(a)+(b): RESNET-50's T1 AND T2 AT BATCH BATCHNORM (ResNet50FullB{,VJP}.lean, 2026-09-06)
#print axioms Proofs.resnet50ForwardB_full
#print axioms Proofs.r50IdB_has_vjp_at
#print axioms Proofs.r50IdB_differentiableAt
#print axioms Proofs.r50ProjB_has_vjp_at
#print axioms Proofs.r50ProjB_differentiableAt
#print axioms Proofs.r50DownB_has_vjp_at
#print axioms Proofs.r50DownB_differentiableAt
#print axioms Proofs.resnet50ForwardB_full_has_vjp_at
#print axioms Proofs.resnet50ForwardB_full_eq_chain
#print axioms Proofs.resnet50ForwardB_full_has_vjp_at_correct

-- T2, the typed forward graph (ResNet-50)
#print axioms Proofs.StableHLO.r50IdGraphB_faithful
#print axioms Proofs.StableHLO.r50ProjGraphB_faithful
#print axioms Proofs.StableHLO.r50DownGraphB_faithful
#print axioms Proofs.StableHLO.r50StemGraphB_faithful
#print axioms Proofs.StableHLO.resnet50FwdGraphB_full_faithful

-- §3.5(c): RESNET-50's T3 -- THE FOLD AND THE TIE (ResNet50{Faithful,Tie}PoCB.lean, 2026-09-06)
#print axioms Proofs.ResNet50PoCB.r50BnGradsCertified
#print axioms Proofs.ResNet50PoCB.r50StemGradsCertified
#print axioms Proofs.ResNet50PoCB.r50IdGradsCertified
#print axioms Proofs.ResNet50PoCB.r50ProjGradsCertified
#print axioms Proofs.ResNet50PoCB.r50DownGradsCertified
#print axioms Proofs.ResNet50PoCB.r50HeadGradsCertified
#print axioms Proofs.ResNet50TieB.r50IdCotIn_eq_vjp
#print axioms Proofs.ResNet50TieB.r50ProjCotIn_eq_vjp
#print axioms Proofs.ResNet50TieB.r50DownCotIn_eq_vjp
#print axioms Proofs.ResNet50TieB.r50_idblock_tiedB
#print axioms Proofs.ResNet50TieB.r50_projblock_tiedB
#print axioms Proofs.ResNet50TieB.r50_downblock_tiedB
#print axioms Proofs.ResNet50TieB.r50_net_tiedB
#print axioms Proofs.ResNet50TieB.r50_lossCot_is_smoothedCE_grad
#print axioms Proofs.ResNet50TieB.r50_lossCot_is_bce_grad
