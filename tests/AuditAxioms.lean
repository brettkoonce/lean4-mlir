import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Training.JacobianSeal
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Architectures.Residual
import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.SE
import LeanMlir.Proofs.Architectures.LayerNorm
import LeanMlir.Proofs.Architectures.GeluErfGaussian
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNet
import LeanMlir.Proofs.Nets.Small.MnistCNN
import LeanMlir.Proofs.Nets.Small.CifarCNN
import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Codegen.StableHLO.Basic
import LeanMlir.Proofs.Nets.Small.ChapterGraphTies
import LeanMlir.Proofs.Codegen.StableHLO.Parse
import LeanMlir.Proofs.Architectures.StridedConv
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBSeal
import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBSeal
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBSeal
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBSeal
import LeanMlir.Proofs.Architectures.PerChannelBN
import LeanMlir.Proofs.Nets.Small.LinearTrainStep
import LeanMlir.Proofs.Nets.Small.MlpTrainStep
import LeanMlir.Proofs.Architectures.ConvGrad
import LeanMlir.Proofs.Architectures.PerChannelBNGrad
import LeanMlir.Proofs.Nets.Small.CnnChainClose
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPC
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStagesPC
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Eval
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Drop
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBEval
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBDrop
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFold
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTie
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChainClose
import LeanMlir.Proofs.Nets.ViT.ViTFwdGraph
import LeanMlir.Proofs.Architectures.TokenParamGrad
import LeanMlir.Proofs.Nets.ViT.ViTChainClose
import LeanMlir.Proofs.Nets.ViT.ViTVecLN
import LeanMlir.Proofs.Nets.ViT.ViTMultiHead
import LeanMlir.Proofs.Nets.ViT.ViTMultiHeadChain
import LeanMlir.Proofs.Nets.ViT.ViTDepthK
import LeanMlir.Proofs.Nets.ViT.ViTFwdDrop
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaper
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Float.MlpFloatBridge
import LeanMlir.Proofs.Training.SgdDescent.Basic
import LeanMlir.Proofs.Training.SgdDescent.Linear
import LeanMlir.Proofs.Training.SgdDescent.Cnn
import LeanMlir.Proofs.Training.SgdDescent.CnnFloat
import LeanMlir.Proofs.Training.SgdDescent.Cifar
import LeanMlir.Proofs.Float.BnFloatBridge
import LeanMlir.Proofs.Float.ResNet34FloatBridge
import LeanMlir.Proofs.Float.BnInputBridge
import LeanMlir.Proofs.Float.FloatComposeBridge
import LeanMlir.Proofs.Float.ConvMixedComposeBridge
import LeanMlir.Proofs.Float.DepthwiseMixedFloatBridge
import LeanMlir.Proofs.Float.DepthwiseFloatBridge
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPCEval
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaperEval
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStagesPCEval
import LeanMlir.Proofs.Foundation.BatchMapVJPAt
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullB
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP
import LeanMlir.Proofs.Foundation.GradNodesB
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldG
import LeanMlir.Proofs.Nets.ViT.ViTFoldG
import LeanMlir.Proofs.Nets.ViT.ViTFoldGB
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFoldGB
import LeanMlir.Proofs.Foundation.Bf16GradNodes
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StepTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTieG
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB
import LeanMlir.Proofs.Nets.ViT.ViTStepTieGB
import LeanMlir.Proofs.Foundation.DataParallel.Basic
import LeanMlir.Proofs.Foundation.DataParallel.Node
import LeanMlir.Proofs.Foundation.DataParallel.Sync
import LeanMlir.Proofs.Foundation.DataParallel.SyncBf16
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncB
import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB
import LeanMlir.Proofs.Foundation.DataParallel.SyncKit
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncStepTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncStepTieG
import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncStepTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncStepTieB
import LeanMlir.Proofs.Codegen.LambTriple
import LeanMlir.Proofs.Foundation.BceLossCot
import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Architectures.ChannelLNBack
import LeanMlir.Proofs.Nets.ResNet.ResNetBackChains
import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackChains
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackChains
import LeanMlir.Proofs.Nets.ViT.ViTBackChains
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2WholeBackCertifiedTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet50WholeBackCertifiedTieB
import LeanMlir.Proofs.Architectures.EvenKernelConvBack
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTieB
import LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTieB
import LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackCertifiedTie
import LeanMlir.Proofs.Nets.ViT.ViTMhsaBackCertifiedTie
import LeanMlir.Proofs.Training.SgdDescent.MlpBias
import LeanMlir.Proofs.Training.Optim.AdamStep
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackB0
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2BackB0
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFold
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTie
import LeanMlir.Proofs.Nets.ViT.ViTBackB0
import LeanMlir.Proofs.Nets.ViT.ViTBackNet
import LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackB0
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4BackB0
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBVJP
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4StepTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4WholeBackCertifiedTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetBackNet
import LeanMlir.Proofs.Nets.Small.LinearFold
import LeanMlir.Proofs.Float.E4M3Fold
import LeanMlir.Proofs.Float.Bf16Fold
import LeanMlir.Proofs.Nets.ResNet.ResNet34ParamGrad
import LeanMlir.Proofs.Nets.ResNet.ResNet50ParamGrad
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2ParamGrad
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4ParamGrad
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetParamGrad
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtParamGrad
import LeanMlir.Proofs.Nets.ViT.ViTParamGrad
import LeanMlir.Proofs.Nets.Small.MlpFold
import LeanMlir.Proofs.Nets.Small.CnnFold
import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Nets.Small.Cifar8StepTie
import LeanMlir.Proofs.Nets.Small.Cifar8BnStepTie
import LeanMlir.Proofs.Nets.Small.Cifar8StepTieG
import LeanMlir.Proofs.Nets.Small.Cifar8BnStepTieG
import LeanMlir.Proofs.Nets.Small.LinearParamGrad
import LeanMlir.Proofs.Nets.Small.MlpParamGrad
import LeanMlir.Proofs.Nets.Small.CnnParamGrad
import LeanMlir.Proofs.Nets.Small.CifarParamGrad
import LeanMlir.Proofs.Nets.Small.Cifar8ParamGrad
import LeanMlir.Proofs.Nets.Small.Cifar8BnParamGrad
import LeanMlir.Proofs.Nets.ViT.ViTFold
import LeanMlir.Proofs.Nets.ViT.ViTStepTie
import LeanMlir.Proofs.Certificates.LipschitzCert.Basic
import LeanMlir.Proofs.Certificates.Smoothing.Gaussian
import LeanMlir.Proofs.Certificates.LipschitzCert.Instance
import LeanMlir.Proofs.Training.Trained.MlpWitness
import LeanMlir.Proofs.Training.Trained.CnnWitness
import LeanMlir.Proofs.Training.Trained.CnnSeal
import LeanMlir.Proofs.Training.Trained.CnnDescent
import LeanMlir.Proofs.Training.Trained.CnnDescentConv1
import LeanMlir.Proofs.Certificates.LipschitzCert.Scorecard
import LeanMlir.Proofs.Certificates.LipschitzCert.PairSDP
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardSDP
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardSDPUncon
import LeanMlir.Proofs.Certificates.LipschitzCert.Float
import LeanMlir.Proofs.Foundation.ListDot
import LeanMlir.Proofs.Certificates.IntervalBound
import LeanMlir.Proofs.Certificates.IntervalBoundConv.Basic
import LeanMlir.Proofs.Certificates.CrownBound
import LeanMlir.Proofs.Certificates.Smoothing.MC
import LeanMlir.Proofs.Certificates.Smoothing.CP
import LeanMlir.Proofs.Certificates.Smoothing.CPScorecard
import LeanMlir.Proofs.Certificates.Smoothing.PhiBounds
import LeanMlir.Proofs.Certificates.Smoothing.DecScorecard
import LeanMlir.Proofs.Certificates.Smoothing.NetSemantics
import LeanMlir.Proofs.Certificates.Smoothing.NetWitness
import LeanMlir.Proofs.Foundation.UpstreamDraft
import LeanMlir.Proofs.Float.Binary32Instance
import LeanMlir.Proofs.Training.Trained.LinearDescent
import LeanMlir.Proofs.Foundation.Muon.Geometry
import LeanMlir.Proofs.Foundation.Muon.NewtonSchulz
import LeanMlir.Proofs.SpecVJP
import LeanMlir.Proofs.Nets.Small.MlpCanonical
import LeanMlir.Proofs.Foundation.SgdNodes

open Proofs

-- One `#print axioms` per certified theorem; CI (certs.yml) fails on any axiom beyond
-- propext / Classical.choice / Quot.sound.

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
#print axioms denseWeightGrad_correct
#print axioms denseBiasGrad_correct
#print axioms reluHasVJP_correct
#print axioms mlpHasVJP_correct
-- The POINTWISE (HasVJPAt) variants. They are also in the comparator suite
-- (tests/comparator/config-arch.json), and they are the three the book singles out as the ones
-- whose `.correct` field is a real proof rather than `rfl`, i.e. exactly where the kink escape is
-- closed.
#print axioms reluHasVJPAt_correct
#print axioms mlpHasVJPAt_correct

-- Nonzero-Jacobian seal (JacobianSeal.lean)
#print axioms sum_smul_basisVec
#print axioms fderiv_eq_zero_of_pdiv_all_zero
#print axioms exists_pdiv_ne_of_fderiv_ne
-- The pointwise (HasVJPAt) seal variants — the kinked witnesses are HasVJPAt, not HasVJP.
#print axioms HasVJPAt.backward_ne_zero_of_pdiv_ne
#print axioms HasVJPAt.backward_nontrivial_of_fderiv_ne

-- CNN
#print axioms maxPool2HasVJP3_correct
#print axioms maxPool2HasVJPAt3_correct   -- pointwise variant; see the note above
#print axioms conv2dHasVJP3
#print axioms conv2dHasVJP3_correct

-- Depthwise
#print axioms depthwiseHasVJP3_correct

-- BatchNorm
#print axioms pdiv_bnAffine
#print axioms pdiv_bnCentered
#print axioms pdiv_bnIstdBroadcast
#print axioms pdiv_bnNormalize
#print axioms bn_input_grad_correct

-- Residual
#print axioms residualHasVJP_correct
#print axioms residualProjHasVJP_correct

-- SE
#print axioms seBlockHasVJP_correct

-- LayerNorm / GELU / Swish
#print axioms pdiv_gelu
#print axioms geluHasVJP_correct
-- either GELU (`GeluForm`): what the ViT and ConvNeXt chains are stated over
#print axioms GeluForm.pdiv_map
#print axioms GeluForm.hasVJP_correct
-- exact GELU x·Φ(x): the VJP, the erfc closed forms (no Lean consumer; keep pinned), Φ = the standard normal cdf
#print axioms pdiv_geluErf
#print axioms geluErfHasVJP_correct
#print axioms geluErfScalar_eq_erfc
#print axioms geluErfScalarDeriv_eq_erfc
#print axioms gaussPhi_eq_cdf
-- closed form the emitted swishBack text computes (no Lean consumer; keep pinned)
#print axioms swishScalarDeriv_eq
#print axioms layerNormHasVJP_correct

-- Attention (apex)
#print axioms pdiv_softmax
#print axioms softmaxCE_grad
#print axioms sdpaBackQ_correct
#print axioms sdpaBackK_correct
#print axioms sdpaBackV_correct
#print axioms mhsaHasVJPMat_correct
#print axioms transformerBlockHasVJPMat_correct
#print axioms transformerTowerHasVJPMat
#print axioms vitBodyHasVJPMat
#print axioms vitFullHasVJP
#print axioms vitFullHasVJP_correct

-- Codegen smooth-point bridge theorems (MLP.lean)
#print axioms relu_codegen_matches_canonical

-- Codegen smooth-point bridge theorems (CNN.lean / MaxPool2)
#print axioms pdiv3_maxPool2_smooth
#print axioms maxPool2_codegen_matches_canonical

-- HasVJPAt pointwise framework
#print axioms reluHasVJPAt
#print axioms mlpHasVJPAt
#print axioms mnistLinearHasVJP_correct
#print axioms maxPool2HasVJPAt3

-- Capstone: end-to-end ResNet-style CNN whole-network VJP + the global-average-pool VJP it depends on
#print axioms globalAvgPoolFlatHasVJP
#print axioms globalAvgPoolFlatHasVJP_correct
#print axioms cnnHasVJPAt
#print axioms cnnHasVJPAt_correct

-- Whole-network VJPs for the depthwise/SE/LN-based architectures
#print axioms relu6HasVJPAt
#print axioms mobilenetv2HasVJPAt_correct
#print axioms convnextHasVJPAt_correct
-- ConvNeXt promoted to an UNCONDITIONAL global VJP (all-smooth ops)
#print axioms convnextHasVJP
#print axioms convnextHasVJP_correct
#print axioms sigmoidHasVJP
-- closed form the emitted sigmoidBack text computes (no Lean consumer; keep pinned)
#print axioms sigmoidScalarDeriv_eq
#print axioms efficientnetHasVJPAt_correct
-- EfficientNet promoted to an UNCONDITIONAL global VJP (all-smooth ops)
#print axioms efficientnetHasVJP
#print axioms efficientnetHasVJP_correct

-- Chapter-4 MNIST 2D CNN (no BN)
#print axioms mnistCnnNoBnHasVJPAt_correct
-- the reusable MaxPool2Smooth discharge
#print axioms maxPool2Smooth_of_injective
#print axioms maxPool2Smooth_of_pairwise
-- Chapter-3 MLP: concrete whole-network instance, every ReLU smoothness hypothesis discharged
#print axioms MlpConcrete.mlpConcreteHasVJPAt
-- ResNet-style CNN *with* BN
#print axioms CnnConcrete.cnnConcreteHasVJPAt

-- Denoted StableHLO-subset IR
#print axioms IR.dense_back_bridge
#print axioms IR.relu_back_bridge
-- The emitted transposed-convolution graph denotes the proven conv input-VJP
#print axioms IR.conv_back_bridge
-- The GENERAL conv-adjoint reindex (all dims, odd kernels)
#print axioms IR.convBackDenote_eq_input_grad_formula
-- The emitted select_and_scatter graph denotes the canonical maxpool backward;
-- its den routes to the first argmax, which off a tie is the argmax
#print axioms IR.maxpool_back_bridge
#print axioms IR.maxPool2Argmax_eq_iff_isArgmax
#print axioms IR.maxPoolBackDenote_eq_of_smooth
#print axioms windowArgmax_first
-- Smooth activations
#print axioms IR.gelu_back_bridge
#print axioms IR.geluErf_back_bridge
#print axioms IR.swish_back_bridge
#print axioms IR.sigmoid_back_bridge
-- BatchNorm: the emitted reduce+broadcast+elementwise graph denotes the proven 3-term backward
#print axioms IR.bn_back_bridge
#print axioms IR.layernorm_back_bridge
#print axioms IR.softmax_back_bridge
-- IR-level chain rule + an end-to-end composite bridge.
#print axioms IR.denote_subst
#print axioms IR.se_back_bridge
-- Tensor3 IR: conv/maxpool lifted into a composable backward graph + chain rule.
#print axioms IR.denote_subst3
#print axioms IR.maxpool3_node_bridge
#print axioms IR.conv3_node_bridge
#print axioms IR.conv_compose3
-- Flatten bridge: flattened Back3 graph denotes the proven flattened-layer Vec backward.
#print axioms IR.maxpool_flatten_bridge
#print axioms IR.conv_flatten_bridge
-- HasVJPAt smooth-point variants + a real dense→relu block via vjpCompAt.
#print axioms IR.relu_at_bridge
#print axioms IR.dense_at_bridge
-- Final assembly: the emitted whole-MLP backward graph denotes the proven whole-network VJP
#print axioms IR.mlp_whole_bridge
-- Parameter gradients (train-step pieces)
#print axioms IR.weight_grad_bridge
#print axioms IR.bias_grad_bridge
#print axioms IR.mlp_layer1_weight_grad_bridge
-- Forward IR
#print axioms IR.mlp_fwd_bridge
#print axioms IR.mlp_fwd_preact0
#print axioms IR.mlp_fwd_preact1
-- Loss cotangent
#print axioms IR.lossCot_bridge

-- Printer-faithfulness (Chapter 2)
#print axioms StableHLO.fwdGraph_faithful
#print axioms StableHLO.lossCotGraph_faithful
#print axioms StableHLO.lossCotGraph_isCEgrad
#print axioms StableHLO.wGrad_isWeightJacobian
#print axioms StableHLO.bGrad_isBiasJacobian
-- SGD update proven (not trusted) for plain SGD on the linear net.
#print axioms StableHLO.sgdW_isCertifiedGradStep
#print axioms StableHLO.sgdB_isCertifiedGradStep
-- The linear SGD step bundled to the certified closed-form softmax-CE gradient
#print axioms StableHLO.lossCot_eq_softmax_sub_onehot
#print axioms StableHLO.sgdW_descends_softmaxCE_grad
#print axioms StableHLO.sgdB_descends_softmaxCE_grad
-- Chain-rule fold: the SGD step is literally θ − lr·∂Loss/∂θ.
#print axioms Proofs.crossEntropy_differentiable
#print axioms StableHLO.denseWeightMap_differentiable
#print axioms StableHLO.lossWeightGrad_eq_sum
#print axioms StableHLO.sgdW_descends_loss_gradient
-- Rendering half
#print axioms StableHLO.linWeightDen_is_loss_descent
#print axioms StableHLO.linBiasDen_is_certified
-- MNIST-linear capstones (LinearFold.lean)
#print axioms LinFold.poc_fwd_is_render
-- Tail fold closed
#print axioms LinFold.poc_weightSgd_den_eq
#print axioms LinFold.poc_biasSgd_den_eq
#print axioms LinFold.poc_train_step_tail_certified
-- mnist-MLP fully folded
#print axioms MlpFold.cot1_den
#print axioms MlpFold.cot0_den
#print axioms MlpFold.W2_den_certified
#print axioms MlpFold.W1_den_certified
#print axioms MlpFold.W0_den_certified
#print axioms MlpFold.b2_den_certified
#print axioms MlpFold.b1_den_certified
#print axioms MlpFold.b0_den_certified
-- mnist-mlp FULLY TIED
#print axioms MlpFold.mlpLossCot_den
#print axioms MlpFold.mlp_W2_tied_totalloss
#print axioms MlpFold.mlp_train_step_tied_certified
-- mnist-CNN fully folded
#print axioms CnnFold.cW1_den
#print axioms CnnFold.cb1_den
#print axioms CnnFold.cW2_den
#print axioms CnnFold.cb2_den
#print axioms CnnFold.dW5_den
-- mnist-cnn dense-head TIE
#print axioms CnnFold.cnnLossCot_den
#print axioms CnnFold.cnn_W5_tied_totalloss
-- mnist-cnn CONV fold
#print axioms CnnFold.cnn_train_step_tied_certified
-- ch5-CIFAR fully folded (no-BN, 2-scale)
#print axioms SgdNode.convW_den
#print axioms SgdNode.convB_den
#print axioms CifarFold.dW7_den
-- ch5-CIFAR TIE
#print axioms CifarFold.cifarLossCot_den
#print axioms CifarFold.cifar_W7_tied_totalloss
#print axioms CifarFold.cifar_train_step_tied_certified
-- ch5-CIFAR-BN fully folded
#print axioms SgdNode.bnGamma_den
#print axioms SgdNode.bnBeta_den
-- deeper 8-conv cifar8 fully folded
#print axioms SgdNode.denseW_den
#print axioms SgdNode.denseB_den
-- ch5-cifar8 TIE
#print axioms Cifar8Tie.cifar8_train_step_tied_certified
-- ch5-cifar8-bn TIE
#print axioms Cifar8BnTie.cifar8Bn_train_step_tied_certified
#print axioms Cifar8TieG.cifar8_train_step_tiedG
#print axioms Cifar8BnTieG.cifar8Bn_train_step_tiedG
#print axioms GradNode.convWGrad_den
#print axioms GradNode.convBGrad_den
#print axioms GradNode.bnGammaGrad_den
#print axioms GradNode.bnBetaGrad_den
#print axioms GradNode.denseWGrad_den
#print axioms GradNode.denseBGrad_den
-- ch6-ResNet-34 fully folded (full [3,4,6,3], 146 params)
#print axioms SgdNode.convStridedW_den
#print axioms SgdNode.convStridedB_den
-- ch7-MobileNetV2 fold (depthwise half)
#print axioms SgdNode.depthwiseW_den
#print axioms SgdNode.depthwiseB_den
-- ch8-EfficientNet-B0 fold (den)
#print axioms EnetFold.convWB_den
#print axioms EnetFold.convStridedWB_den
#print axioms EnetFold.denseWB_den
#print axioms EnetFold.denseBB_den
#print axioms EnetFold.bnGammaB_den
#print axioms EnetFold.bnBetaB_den
#print axioms EnetFold.depthwiseWB_den
#print axioms EnetFold.depthwiseStridedWB_den
-- ch8-EfficientNet-B0 TIE
#print axioms EnetTie.enet_exp_tied
#print axioms EnetTie.enet_strided_tied
#print axioms EnetTie.enet_noexp_tied
#print axioms EnetTie.enet_stem_tied
#print axioms EnetTie.enet_head_tied
#print axioms EnetTie.efficientnet_net_tied
-- The MLP per-layer parameter-gradient assembly
#print axioms IR.mlp_layer0_weight_grad_bridge
#print axioms IR.mlp_layer0_bias_grad_bridge
#print axioms IR.mlp_layer1_bias_grad_bridge
#print axioms IR.mlp_layer2_weight_grad_bridge
-- The CNN convolution parameter-gradient bridges (kernel grad = correlation).
#print axioms conv_weight_grad_bridge
#print axioms conv_bias_grad_bridge
-- CNN render close: the rendered conv weight/bias SGD outputs denote θ − lr·certified.
#print axioms conv_weight_sgd_certified
#print axioms conv_bias_sgd_certified
-- Chain: the composed cotangent subgraphs reduce to the explicit relu'⊙Wᵀ·… backprop
#print axioms IR.mlpCotOut1_denote
#print axioms IR.mlpCotOut0_denote
-- The conditional hidden-layer folds
#print axioms IR.mlp_hidden_total_loss_grad
#print axioms IR.mlp_input_total_loss_grad
-- Whole-net capstone: every weight layer's total-loss gradient at once (one statement).
#print axioms IR.mlp_whole_net_weight_grads
-- Printer-faithfulness, Chapter 3 (MLP)
#print axioms StableHLO.reluF_faithful
#print axioms StableHLO.selectPos_faithful
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
-- CLASSIFIER DROPOUT
#print axioms StableHLO.dropoutB_faithful
#print axioms StableHLO.dropoutB_back_faithful
#print axioms StableHLO.den_dropoutB_of_dropScale
#print axioms Proofs.dropPath_scales_uniformly
#print axioms Proofs.dropout_eq_reference
#print axioms Proofs.dropout_ones_id
#print axioms Proofs.dropout_vjp_is_self
#print axioms StableHLO.mlpFwdGraph_faithful
#print axioms StableHLO.mlpBackGraph_faithful
-- Printer-faithfulness, Chapter 4 (CNN)
#print axioms StableHLO.flatConvF_faithful
#print axioms StableHLO.maxPoolF_faithful
#print axioms StableHLO.cnnFwdGraph_faithful
#print axioms StableHLO.convBack_faithful
#print axioms StableHLO.maxPoolBack_faithful
-- He et al.'s 3×3/s2 STEM POOL
#print axioms win3RowInv_first_dup
#print axioms win3ColInv_first_dup
#print axioms win3Row_mem_le_two
-- the argmax: `Finset.sup'` over `Fin 3 × Fin 3`, which is why window size stopped mattering
#print axioms maxPool3s2_eq_at_max
-- the local linearisation and the smooth-point VJP
#print axioms maxPool3s2HasVJPAt3
#print axioms maxPool3s2Flat_differentiableAt
#print axioms maxPool3s2FlatHasVJPAt
-- the DISCHARGE lemma for the smoothness hypothesis
#print axioms maxPool3s2Smooth_of_injective
-- and the float side (`floatClose_maxPool`'s peer)
#print axioms floatClose_maxPool3s2
-- and the codegen that denotes it
#print axioms StableHLO.maxPool3s2F_faithful
#print axioms StableHLO.maxPool3s2Back_faithful
-- The whole-chain CNN backward graph denotes the proven conditional whole-network VJP
#print axioms StableHLO.cnnBackGraph_faithful

-- Chapter-5 CIFAR-10 2D CNN (no BN)
#print axioms cifarCnnHasVJPAt
#print axioms StableHLO.cifarFwdGraph_faithful
-- Concrete tiny CIFAR instance
#print axioms Tiny.cifarTinyCnnHasVJP_correct

-- Chapter-5 BatchNorm backward render
#print axioms StableHLO.bnBack_faithful

-- Deeper 8-conv CIFAR (the pedagogical BN-acceleration demo)
#print axioms cifarCnn8HasVJPAt
#print axioms cifarCnnBn8HasVJPAt_correct
-- ...and the rendered-FORWARD peer
#print axioms StableHLO.cifar8FwdGraph_faithful
#print axioms StableHLO.cifar8BnFwdGraph_faithful

-- Chapter-6 ResNet **Milestone B** (toward real ResNet-34)
-- ...and its weight-VJP (the kernel grad for training a strided block)
#print axioms flatConvStride2WeightGradHasVJP_correct
-- ResNet-34's non-degeneracy, ON THE NET THE ARTIFACTS RUN (ResNet34FullBSeal.lean): both
-- levels are stated on `resnet34ForwardBFull` itself — full width, batch BN, 224x224.
#print axioms R34FullBSeal.sealX_nonconstant
#print axioms R34FullBSeal.sealX_jacobian_nonzero
#print axioms R34FullBSeal.sealX_backward_nontrivial
-- ResNet-50's, likewise on `resnet50ForwardBFull` — 48 relu clauses, and `q` a binder, so ONE
-- statement seals both shipped resolutions (224 px at q = 7, 160 px at q = 5).
#print axioms R50FullBSeal.sealX_nonconstant
#print axioms R50FullBSeal.sealX_jacobian_nonzero
#print axioms R50FullBSeal.sealX_backward_nontrivial
-- MobileNetV2's, on `mobilenetv2ForwardBFull` -- all seventeen bottlenecks, batch BN, 224x224,
-- relu6. All 35 kink clauses are weight-only here (every relu6 sits on a BatchNorm output, and
-- the linear bottleneck has no relu after the residual add), and the carrier threads 22
-- BatchNorms because a channel-changing bottleneck has no skip to pass it on.
#print axioms Mnv2FullBSeal.sealX_nonconstant
#print axioms Mnv2FullBSeal.sealX_jacobian_nonzero
#print axioms Mnv2FullBSeal.sealX_backward_nontrivial
-- MobileNetV4-Conv-M's, on `mobilenetv4ForwardBFull` -- all 21 UIB blocks, the fused stage and
-- the two-conv head, batch BN, 224x224. Its 38 kink clauses (counted off the block table by a
-- `#guard`) are weight-only like MobileNetV2's. The witness is grid-constant (one value per example
-- and channel), and the carrier threads 17 BatchNorms: the stem, the fused stage's two, four in
-- each of the three channel-changing rows, and the head's two (the second at 1x1, after the pool).
#print axioms Mnv4FullBSeal.sealX_nonconstant
#print axioms Mnv4FullBSeal.sealX_jacobian_nonzero
#print axioms Mnv4FullBSeal.sealX_backward_nontrivial
-- Per-channel BatchNorm
#print axioms bnPerChannelFlatHasVJP_correct
-- The RENDERABLE per-channel BN backward
#print axioms bnPerChannelGradInput_correct
-- Per-channel BN on the network's Tensor3 (oc*h)
#print axioms bnPerChannelTensor3HasVJP_correct
-- Per-channel BN on the network's Tensor3 (cont.)
#print axioms bnPerChannelTensor3GradInput_correct
-- ch8 EfficientNet BATCH-norm
#print axioms bnBatchTensor4GradInput_correct
-- sync-BN kit
-- the identity that lets replicas exchange the SECOND MOMENT and each recover the variance
#print axioms bnVar_eq_bnMeanSq_sub_sq
-- a statistic of the whole = the mean of the shards' statistics, which is what makes a plain
-- `allReduceMeanF` the right collective. The variance has no such lemma, and cannot.
#print axioms bnMean_shard
#print axioms bnMeanSq_shard
-- the R = 1 anchors: handed the batch's OWN statistics, the sync forward/backward ARE the
-- existing batch forward/backward — so the single-device renders denote the tied function
#print axioms bnEvalForward_at_own_stats
#print axioms bnSyncTensor4_at_own_stats
#print axioms bnSyncGradInput_at_own_stats
#print axioms bnSyncTensor4GradInput_at_own_stats
#print axioms bnSyncXhat_at_own_stats
-- Chan's parallel variance — what the exchange carries instead of E[x²], so that no consumer
-- forms E[x²] − μ² (that formulation drifts 2e-4 over R34's 36 layers in f32)
#print axioms bnVar_shard_chan
#print axioms bnVar_row_shard_chan
-- THE DROP-IN, on actual graph nodes: at R = 1 the sync-BN subgraphs (bnSyncF / bnSyncBack
-- fed by the two-round statistics subgraph / bnSyncDyStatsB) denote exactly what the
-- bnBatchF / bnBatchBack renders denote — so the R = 1 artifacts need not move.
#print axioms StableHLO.den_syncStats_R1
-- Sync-BN on replica r IS the shard-r block of the GLOBAL-BATCH forward and
-- input-VJP, handed the all-reduced statistics. The spec does not move: both right-hand sides
-- are the EXISTING bnBatchTensor4 / bnBatchTensor4GradInput at N := R·N.
#print axioms bnSyncTensor4_shard_eq_global
#print axioms bnSyncTensor4GradInput_shard_eq_global
-- their two halves: pointwise-so-sharding-commutes (no mathematics), and the statistics really
-- being the all-reduced ones (all the mathematics)
#print axioms bnSyncTensor4_batchShard
#print axioms bnSyncTensor4GradInput_batchShard
#print axioms bnMean_row_shard
#print axioms bnMeanSq_row_shard
#print axioms bnMean_pair_row_shard
-- the layout step the bnchwFwd relabel was hiding
#print axioms bnchwFwd_row_batchShard
#print axioms StableHLO.den_bnSyncF_allReduce_R1
#print axioms StableHLO.den_bnSyncBack_allReduce_R1
-- the fifth op: the γ gradient at the handed-in statistics (bnGammaGradB rebuilds x̂ from the
-- shard's own μ/σ², which under sync-BN is the wrong x̂), its R = 1 anchor, and the γ/β row splits
#print axioms bnSyncPerChannelGradGamma_at_own_stats
#print axioms bnSyncPerChannelGradGamma_row_shard
#print axioms bnPerChannelGradBeta_row_shard
#print axioms StableHLO.den_bnSyncGammaGradB_allReduce_R1
-- the handed-back running statistics, read off the packed vector: at R = 1 they ARE
-- bnBatchMeanB / bnBatchVarB
#print axioms StableHLO.den_bnStatsMeanB_allReduce_R1
#print axioms StableHLO.den_bnStatsVarB_allReduce_R1
-- The per-channel BN SHlo op pair backward-faithfulness
#print axioms StableHLO.bnPerChannelBack_faithful
-- Syntactic core
#print axioms StableHLO.roundtrip
-- CIFAR-BN render CLOSE
#print axioms bnPerChannelGradGamma_correct
#print axioms bnPerChannelGradBeta_correct
#print axioms bnPerChannel_gamma_sgd_certified
#print axioms bnPerChannel_beta_sgd_certified
-- CNN conv-close UPGRADE
#print axioms cnn_render_convW2_chain_certified
#print axioms cnn_render_convb2_chain_certified
#print axioms cnn_render_convW1_chain_certified
#print axioms cnn_render_convb1_chain_certified
-- MobileNetV2 CLOSE
#print axioms depthwise_bias_sgd_certified
#print axioms convStride2_weight_sgd_certified
#print axioms convStride2_bias_sgd_certified
-- ResNet-34 cotangent-chain CLOSE
#print axioms StableHLO.stemGraphB_faithful
#print axioms StableHLO.mbNoExpGraphB_faithful
#print axioms StableHLO.mbStridedGraphB_faithful
#print axioms StableHLO.mbResidGraphB_faithful
#print axioms StableHLO.headGraphB_faithful
-- EfficientNet-B0 cotangent-chain CLOSE
#print axioms batchMapHasVJP
#print axioms batchMap_differentiable
#print axioms reindexHasVJP
#print axioms bnBatchLAHasVJP
#print axioms bnBatchLA_differentiable
-- Per-block batched gradients (the per-block VJP, the user-requested deliverable)
#print axioms mbNoExpFwdBHasVJP
#print axioms mbStridedFwdBHasVJP
#print axioms mbResidFwdBHasVJP
#print axioms headFwdBHasVJP
-- FULL EfficientNet-B0 (all 16 MBConv blocks, real [t,c,n,s,k] spec)
#print axioms StableHLO.mbExpGraphB_faithful
#print axioms StableHLO.efficientnetFwdGraphBFull_faithful
#print axioms efficientnetForwardBFullHasVJP
-- The nested↔∘-chain bridge + correctness on the nested forward itself
#print axioms efficientnetForwardBFull_eq_chain
#print axioms efficientnetForwardBFullHasVJP_correct
-- ConvNeXt fold
#print axioms Proofs.CnxFold.pdiv_layerScaleCh_gamma
#print axioms Proofs.CnxFold.cnx_render_lsgammaCh_certified
-- ConvNeXt fold (cont.)
#print axioms Proofs.CnxFold.layerScaleChGammaSgd_den
-- The CHANNEL-LN γ/β param certs (ConvNeXtChannelLN)
#print axioms Proofs.chanRowsIdxInv_chanRowsIdx
#print axioms Proofs.chanRowsIdx_chanRowsIdxInv
#print axioms Proofs.pdiv_reindexOut_contract
#print axioms Proofs.chanLN_gamma_contract
#print axioms Proofs.chanLN_beta_contract
#print axioms Proofs.cnx_render_chlngamma_certified
#print axioms Proofs.cnx_render_chlnbeta_certified
#print axioms Proofs.CnxFold.chanLnGammaSgd_den
#print axioms Proofs.CnxFold.chanLnBetaSgd_den
-- ch9-ConvNeXt-T FULL [3,3,9,3] TIE
#print axioms Proofs.CnxTie.cnx_block_ch_tied
#print axioms Proofs.CnxTie.cnx_down_ch_tied
#print axioms Proofs.CnxTie.cnx_stem_ch_tied
#print axioms Proofs.CnxTie.cnx_head_ch_tied
#print axioms Proofs.CnxTie.cnxLossCot_den
#print axioms Proofs.CnxTie.cnx_net_tied_certified
-- ViT CLOSE
#print axioms pdiv_rowDense_W
#print axioms rowDense_weight_grad_bridge
#print axioms rowDense_bias_grad_bridge
#print axioms rowDense_weight_sgd_certified
#print axioms rowDense_bias_sgd_certified
#print axioms pdiv_patchEmbed_pos
#print axioms posEmbed_sgd_certified
#print axioms pdiv_patchEmbed_cls
#print axioms clsToken_sgd_certified
#print axioms pdiv_patchEmbed_W
#print axioms patchEmbed_weight_grad_bridge
#print axioms patchEmbed_weight_sgd_certified
#print axioms pdiv_patchEmbed_b
#print axioms patchEmbed_bias_grad_bridge
#print axioms patchEmbed_bias_sgd_certified
-- ViT cotangent-chain CLOSE
#print axioms vitCotDP_eq_sdpaDWeights
#print axioms vitCotDS_eq_sdpaDScaled
#print axioms vitCotDQ_eq_sdpaBackQ
#print axioms vitCotDK_eq_sdpaBackK
#print axioms vitCotDV_eq_sdpaBackV
-- ViT SCALING PASS
#print axioms layerNormVecHasVJP
#print axioms transformerBlockVHasVJPMat
#print axioms pdiv_vecLN_gamma
#print axioms pdiv_vecLN_beta
#print axioms layerNormVec_gamma_grad_bridge
#print axioms layerNormVec_beta_grad_bridge
#print axioms layerNormVec_gamma_sgd_certified
#print axioms layerNormVec_beta_sgd_certified

-- ViT scaling pass: multi-head (ViTMultiHead.lean)
#print axioms sum_headPadMat_apply
#print axioms mhsaLayer_spelled
#print axioms vitBlockSpelledMHV_eq
#print axioms StableHLO.den_headsSumG

-- ViT scaling pass: depth-k (ViTDepthK.lean)
#print axioms vitBodyKVFlat_eq_flatten
#print axioms vitBodyKVFlatHasVJP
#print axioms vitForwardKVHasVJP
#print axioms vitForwardKVHasVJP_correct
#print axioms StableHLO.vitBodyGraphKMHV_den
#print axioms StableHLO.vitFwdGraphKMHV_faithful

-- Full ConvNeXt-T [3,3,9,3] (ConvNeXtFullT.lean)
#print axioms decimateOddFlatHasVJP
#print axioms flatConvStride4HasVJP
#print axioms convNextStageChKHasVJP
#print axioms cnxDownChWHasVJP
#print axioms convNextForwardTChHasVJP
-- The nested↔∘-chain bridge
#print axioms convNextForwardTCh_eq_chain
#print axioms convNextForwardTCh_differentiable
#print axioms convNextForwardTChHasVJP_correct
-- The channel-LN GRAPH + faithfulness (the apex)
#print axioms StableHLO.chanLNGraph_faithful
#print axioms StableHLO.cnxBlockChGraphW_faithful
#print axioms StableHLO.cnxStageChGraphK_den
#print axioms StableHLO.cnxDownChGraphW_faithful
#print axioms StableHLO.convNextFwdGraphTCh_faithful
-- The SCALAR-LN twin of this chain

-- ℝ→Float32 bridge (FloatBridge.lean, MlpFloatBridge.lean)
#print axioms FloatModel.dot_close
-- TreeReduceBridge.lean
#print axioms FloatModel.dotMixed
#print axioms FloatModel.dot_close_mixed
#print axioms FloatModel.dot_close_mixed_uniform
#print axioms FloatModel.dotMixed_exact_leaf
-- Threaded through the dense layer
#print axioms FloatModel.denseMixed
#print axioms FloatModel.dense_close_mixed
#print axioms FloatModel.dense_close
#print axioms FloatModel.dense_close_fresh
#print axioms FloatModel.relu_close
-- MaxPool exact-in-float (CNN.lean)
#print axioms maxPoolFlat_close
-- Conv forward rounding budget (Float/ConvFloat.lean)
#print axioms conv2d_eq_dense
#print axioms convPad_close
#print axioms FloatModel.convF
#print axioms FloatModel.convF_close
-- The flat conv's float forward (ConvFloat.lean)
#print axioms FloatModel.flatConvF
#print axioms FloatModel.flatConvF_close
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
-- ResNet-34 structural float ops (ResNet34FloatBridge.lean)
#print axioms FloatModel.add_close
#print axioms FloatModel.gapFlat_close
-- Real-BN input-sensitivity (BnInputBridge.lean)
#print axioms bnMean_input_close
#print axioms bnVar_input_close
#print axioms bnIstd_input_close
#print axioms bnForward_input_close
-- Whole-net certificate backbone (FloatComposeBridge.lean)
#print axioms FloatClose.comp
#print axioms floatClose_relu
#print axioms floatClose_flatConv
-- FloatClose max-pool (FloatComposeBridge.lean)
#print axioms floatClose_maxPool
-- The r34 wrap that lets the fold RUN on a real block
#print axioms floatClose_residualBlock
-- THE FINAL FOLD: floatClose_id + floatClose_iterate
#print axioms floatClose_id
#print axioms floatClose_iterate
-- EfficientNet float bridge
#print axioms floatClose_addResidual
-- The remaining FloatClose instances, all wraps of existing closeness
#print axioms globalAvgPoolFlat_eq_bnMean
#print axioms floatClose_gap
-- Depthwise conv
#print axioms depthwiseConv2d_eq_dense
#print axioms FloatModel.depthwiseConv2dF_close
#print axioms FloatModel.depthwiseFlatF_close
-- Strided-conv backward (r34 down-blocks + stem)
#print axioms Proofs.decimateBack_eq_vjp
-- BN mean at any reduction order (BnFloatBridge.lean)
#print axioms Proofs.FloatModel.bnMean_close_of
-- The mnv2 block bridges are generic in the NORMALISATION too (`*Gen`)
#print axioms Proofs.mobilenetv2ForwardPaperEval
#print axioms Proofs.StableHLO.ivNoExpGraphEvalW_faithful
#print axioms Proofs.StableHLO.ivExpOnlyGraphEvalW_faithful
#print axioms Proofs.StableHLO.ivResidGraphEvalW_faithful
#print axioms Proofs.StableHLO.ivStridedGraphEvalW_faithful
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphPaperEval_faithful
-- The EfficientNet-B0 INFERENCE forward and its graph
#print axioms Proofs.StableHLO.stemGraphBEval_faithful
#print axioms Proofs.StableHLO.mbNoExpGraphBEval_faithful
#print axioms Proofs.StableHLO.mbStridedGraphBEval_faithful
#print axioms Proofs.StableHLO.mbResidGraphBEval_faithful
#print axioms Proofs.StableHLO.headGraphBEval_faithful
-- The B0 stage and block bridges are generic in the NORMALISATION (`*BGen`)
#print axioms Proofs.mbExpFwdBEval
#print axioms Proofs.StableHLO.mbExpGraphBEval_faithful
#print axioms Proofs.StableHLO.mbNoExpGraphEvalW_faithful
#print axioms Proofs.StableHLO.mbStridedGraphEvalW_faithful
#print axioms Proofs.StableHLO.mbResidGraphEvalW_faithful
#print axioms Proofs.StableHLO.mbExpGraphEvalW_faithful
#print axioms Proofs.efficientnetForwardBFullEval
#print axioms Proofs.StableHLO.efficientnetFwdGraphBFullEval_faithful
-- EfficientNet-B0 with stochastic depth + classifier dropout (EfficientNetFullB0Drop.lean)
#print axioms Proofs.efficientnetForwardBFullDrop_none
#print axioms Proofs.efficientnetForwardBFullDrop_ones
#print axioms Proofs.efficientnetForwardBFullEvalDrop_none
#print axioms Proofs.efficientnetForwardBFullEvalDrop_ones
#print axioms Proofs.StableHLO.mbResidDropGraphW_faithful
#print axioms Proofs.StableHLO.headGraphBDo_faithful
#print axioms Proofs.StableHLO.mbResidDropGraphEvalW_faithful
#print axioms Proofs.StableHLO.headGraphBEvalDo_faithful
#print axioms Proofs.StableHLO.efficientnetFwdGraphBFullDrop_faithful
#print axioms Proofs.StableHLO.efficientnetFwdGraphBFullEvalDrop_faithful
-- The MobileNetV4-Conv-M INFERENCE forward and its graph, at any input size
#print axioms Proofs.StableHLO.mnv4BodyGraphBEval_faithful
#print axioms Proofs.StableHLO.mnv4StridedGraphBEval_faithful
#print axioms Proofs.StableHLO.mobilenetv4ForwardBFullEval
#print axioms Proofs.StableHLO.mnv4FwdGraphBFullEval_faithful
-- Classifier dropout (the `%do` form) on MobileNetV2 and MobileNetV4
#print axioms Proofs.mobilenetv2ForwardBFullDo_ones
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphBFullDo_faithful
#print axioms Proofs.StableHLO.mobilenetv4ForwardBFullDo_ones
#print axioms Proofs.StableHLO.mnv4FwdGraphBFullDo_faithful
-- Stochastic depth (the `%dp` sites) on MobileNetV4 (MobileNetV4FullBDrop.lean) and ViT (ViTFwdDrop.lean)
#print axioms Proofs.StableHLO.mnv4SkipDropGraphB_faithful
#print axioms Proofs.StableHLO.mobilenetv4ForwardBFullDrop_sdOnes
#print axioms Proofs.StableHLO.mobilenetv4ForwardBFullDrop_ones
#print axioms Proofs.StableHLO.mnv4FwdGraphBFullDrop_faithful
#print axioms Proofs.blockVDrop_one
#print axioms Proofs.vitBlockSpelledMHVDrop_eq
#print axioms Proofs.vitForwardKVDrop_ones
#print axioms Proofs.vitForwardKVDropB_ones
#print axioms Proofs.StableHLO.vitBlockGraphBDrop_slice
#print axioms Proofs.StableHLO.vitBodyGraphBDrop_slice
#print axioms Proofs.StableHLO.vitFwdGraphBDrop_slice
#print axioms Proofs.StableHLO.vitFwdGraphBDrop_faithful
-- Integrity tie (the r34 identity block)
#print axioms Proofs.convFlatBack_eq_vjp_backward
-- Integrity tie (the r34 DOWNSAMPLE block)
#print axioms Proofs.flatConvStride2Back_eq_vjp_backward
-- Its XLA-SAME peer (the TF-origin B0 / MobileNetV2 stems): the same conv leaf + decimateOddBack rfl.
#print axioms Proofs.flatConvStride2XlaBack_eq_vjp_backward
-- DEPTHWISE adjoint gate (shared prereq for convnext/mnv2/enet)
#print axioms Proofs.depthwiseConv2d_dwReverse_eq_input_grad_formula
#print axioms Proofs.depthwiseFlatBack_eq_vjp_backward
-- Its XLA-SAME peer (MobileNetV2's four strided depthwises, B0's downsample depthwise).
#print axioms Proofs.depthwiseStride2FlatXlaBack_eq_vjp_backward
-- At ConvNeXt's REAL channel LayerNorm
#print axioms Proofs.HasVJP.backward_unique
#print axioms Proofs.bnGradInput_eq_vjp_backward
#print axioms Proofs.transposeFlatHasVJP_backward_eq
#print axioms Proofs.rowLNVecFlatHasVJP_backward_eq
#print axioms Proofs.chanLNTensor3Back_eq_chanLN_vjp
#print axioms Proofs.cnxBodyWithChanLNBack_eq_vjp
#print axioms Proofs.cnxBlockChBack_eq_vjp
-- Integrity tie (vit MHSA — the sdpa adjoint)
#print axioms Proofs.projBack_core_coord
#print axioms Proofs.woback_unflatten
#print axioms Proofs.mhsaBackFlat_eq_mhsa_vjp
-- The MLP-sublayer per-token leaf
#print axioms Proofs.transformerMlp_back_flat_eq_perRowFlatPR
-- Endpoint leaf ties
#print axioms Proofs.dense_transpose_eq_vjp_backward
#print axioms Proofs.gapBack_eq_vjp_backward
-- He et al.'s 3×3/s2 stem pool's BACKWARD
#print axioms Proofs.maxPool3s2FlatBack_eq_vjp_backward
#print axioms Proofs.maxPool3s2FlatHasVJPAtVec
-- the two spellings of that scatter are one map: the render's `.maxPool3s2BackB` node = the chain's
#print axioms Proofs.maxPool3s2BackFlat_eq_flatBack
#print axioms Proofs.den_maxPool3s2BackB_eq_flatBackB
-- the generic 21-stage apex the batched MobileNetV2 tie instantiates (MobileNetV2WholeBackCertifiedTieB.lean)
#print axioms Proofs.mobilenetv2PaperPCHasVJPAt
-- AND FOR THE WHOLE EFFICIENTNET-B0 (EfficientNetWholeBackCertifiedTie.lean)
#print axioms Proofs.stemBBack_eq_vjp_backward
#print axioms Proofs.headFwdBBack_eq_vjp_backward
-- AND AT THE PAPER DEPTH — all 16 MBConv blocks (EfficientNetFullWholeBackCertifiedTie.lean)
#print axioms Proofs.efficientnetBFullHasVJP
#print axioms Proofs.efficientnetInputGradBFull_eq_efficientnetB_full_vjp
#print axioms Proofs.efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp
#print axioms Proofs.efficientnetInputGradBFull_correct
-- AND AT BATCH BATCH-NORM (ResNet34BackCertifiedTieB.lean, MobileNetV2WholeBackCertifiedTieB.lean)
#print axioms Proofs.HasVJPAt.backward_unique
#print axioms Proofs.cbReluStridedBBack_eq_vjp_backward
#print axioms Proofs.r34HeadBBack_eq_vjp_backward
#print axioms Proofs.r34BFullHasVJPAt
#print axioms Proofs.r34InputGradB_eq_r34B_full_vjp
#print axioms Proofs.r34InputGradB_correct
#print axioms Proofs.resnet34ForwardBFull_eq_slots
#print axioms Proofs.mnv2StemBBack_eq_vjp_backward
#print axioms Proofs.cbrBBack_eq_vjp_backward
#print axioms Proofs.mnv2InputGradB_eq_mobilenetv2B_full_vjp
#print axioms Proofs.mnv2InputGradB_correct
#print axioms Proofs.mobilenetv2ForwardBFull_eq_slots
-- AND RESNET-50's T6 (ResNet50WholeBackCertifiedTieB.lean)
#print axioms Proofs.r50InputGradB_eq_r34B_full_vjp
#print axioms Proofs.r50InputGradB_correct
#print axioms Proofs.resnet50ForwardBFull_eq_slots
-- THE EVEN-KERNEL CONV BACKWARD (EvenKernelConvBack.lean)
#print axioms Proofs.padOdd
#print axioms Proofs.conv2d_padOdd_eq
#print axioms Proofs.flatConv_padOdd_eq
#print axioms Proofs.HasVJP.backward_unique_of_eq
#print axioms Proofs.convFlatBack_padOdd_eq_vjp_backward
#print axioms Proofs.flatConvStride2Back_padOdd_eq_vjp_backward
#print axioms Proofs.flatConvStride4Back_padOdd_eq_vjp_backward
-- ConvNeXt-T's whole-net backward tie, two pieces
#print axioms Proofs.cnxDownChBack_eq_vjp
#print axioms Proofs.cnxBlockChBackAt
#print axioms Proofs.cnxStageChKBack
#print axioms Proofs.cnxStageChKBack_eq_vjp
#print axioms Proofs.rowLNVecFlatHasVJP_backward_eq_fun
#print axioms Proofs.cnxSavedA10
#print axioms Proofs.convNextForwardTChVjpChain
#print axioms Proofs.convnextInputGrad_eq_convNextForwardTCh_vjp
-- ConvNeXt WHOLE-NET BACKWARD AT A BATCH — T6 at the shipped index and the ImageNet head (ConvNeXtWholeBackCertifiedTieB.lean)
#print axioms Proofs.convnextInputGradB
#print axioms Proofs.cnxSavedB10
#print axioms Proofs.cnxChanLNBAt
#print axioms Proofs.cnxDownBAt
#print axioms Proofs.cnxStemBackB_eq_vjp
#print axioms Proofs.cnxChanLNBackB_eq_vjp
#print axioms Proofs.cnxStageBackB_eq_vjp
#print axioms Proofs.cnxDownBackB_eq_vjp
#print axioms Proofs.cnxGapBackB_eq_vjp
#print axioms Proofs.cnxLNhBackB_eq_vjp
#print axioms Proofs.cnxDenseBackB_eq_vjp
#print axioms Proofs.vjpCompDiffAt_fst_backward
#print axioms Proofs.convNextForwardTChBHasVJPAt
#print axioms Proofs.convnextInputGradB_eq_convNextForwardTChB_vjp
#print axioms Proofs.convNextForwardTChB_eq_chain
#print axioms Proofs.convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp
#print axioms Proofs.convnextInputGradB_correct
#print axioms Proofs.convnextImagenetInputGradB_eq_vjp
-- ViT-Tiny's whole-net backward tie
#print axioms Proofs.vitBlockBackV
#print axioms Proofs.vitBlockBackVAt
#print axioms Proofs.vitTowerBackK
#print axioms Proofs.vitInputGradK
#print axioms Proofs.rowLNVecFlatBack_eq_vecLN_vjp
#print axioms Proofs.attnSubFlat_tie_v
#print axioms Proofs.mlpSubFlat_tie_v
#print axioms Proofs.vitBlockBackV_eq_transformerBlockV_vjp
#print axioms Proofs.vitBlockBackVAt_eq_vjp
#print axioms Proofs.vitHeadBack_eq_classifier_vjp
#print axioms Proofs.vitFinalLNBack_eq_vjp
#print axioms Proofs.vitTowerBackK_eq_vjp
#print axioms Proofs.vitForwardKV_eq_chain
#print axioms Proofs.vitInputGradK_eq_vitApexVJP
#print axioms Proofs.vitInputGradK_eq_vitForwardKV_vjp
#print axioms Proofs.vitInputGradK_correct
#print axioms Proofs.vitTinyInputGrad_eq_vitTiny_vjp
#print axioms Proofs.vitTinyHasVJP_correct
-- ViT WHOLE-NET BACKWARD AT A BATCH — T6 at the shipped index (ViTWholeBackCertifiedTieB.lean)
#print axioms Proofs.vitTowerBackB_eq_vjp
#print axioms Proofs.vitLNBackB_eq_vjp
#print axioms Proofs.vitHeadBackB_eq_vjp
#print axioms Proofs.vitKVBHasVJPAt
#print axioms Proofs.vitInputGradKB_eq_vitKVB_vjp
#print axioms Proofs.vitForwardKVB_eq_chain
#print axioms Proofs.vitForwardKV_differentiable
#print axioms Proofs.vitInputGradKB_eq_batchMap_vitForwardKV_vjp
#print axioms Proofs.vitInputGradKB_correct
#print axioms Proofs.vitTinyInputGradB_eq_vitTiny_vjp
-- ViT TRANSFORMER-BLOCK FOLD
#print axioms convWeightGrad_eq_dot
#print axioms convBiasGrad_eq_sum
-- CIFAR-8 last-conv SGD descent (SgdDescent/Cifar.lean)
#print axioms Proofs.cifarCnn8Forward_factor
#print axioms Proofs.cifar8_lastConv_sgd_descends
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
-- The loss head — the LAST float hypothesis discharged
#print axioms FloatModel.sum_close
#print axioms FloatModel.softmax_perturb
#print axioms FloatModel.softmaxF_close
#print axioms FloatModel.softmax_ce_cot_close
#print axioms FloatModel.mnist_cot_budget
-- floatbridge quantization
#print axioms FloatModel.argmax_preserved
#print axioms FloatModel.denseMixedBudget
#print axioms FloatModel.dense_close_mixed_uniform_budget
#print axioms FloatModel.denseMixedBudget_le_of
#print axioms uE4M3
#print axioms FloatModel.linear_e4m3_logit_budget
#print axioms FloatModel.linear_e4m3_argmax_preserved
-- floatbridge quantization (cont.)
#print axioms QuantFold.dequant_factors
#print axioms QuantFold.e4m3_render_faithful
-- The bf16-mixed render-tie, its companion (Bf16Fold.lean)
#print axioms Proofs.Bf16Fold.bf16_render_faithful
#print axioms Proofs.Bf16Fold.bf16_render_faithful_emit
#print axioms Proofs.Bf16Fold.bf16_render_faithful_depth2
-- The loss gradient in a parameter, from the gradient at its op's output (ParamGrad.lean)
#print axioms Proofs.addConstHasVJPAt
#print axioms Proofs.constAddHasVJPAt
#print axioms Proofs.hasGradAt_iff
#print axioms Proofs.HasGradAt.comp
#print axioms Proofs.HasGradAt.congr_left
#print axioms Proofs.HasGradAt.congr_of_eventuallyEq
#print axioms Proofs.HasGradAt.param
#print axioms Proofs.HasGradAt.param_batchMap
-- The batched smoothed loss, and its gradient is the emitted cotangent (SmoothedBatchLoss.lean)
#print axioms Proofs.rowSumLoss_pdiv
#print axioms Proofs.smoothedBatchLoss_pdiv
#print axioms Proofs.smoothedBatchLoss_grad
-- The batched BCE-with-logits loss, and its gradient is the emitted cotangent (BceBatchLoss.lean)
#print axioms Proofs.bceBatchLoss_pdiv
#print axioms Proofs.bceBatchLoss_grad
-- Each batched parameter gradient node is a loss derivative (ParamGradNodes.lean)
#print axioms Proofs.GradNodeB.convW_hasGradAt
#print axioms Proofs.GradNodeB.convB_hasGradAt
#print axioms Proofs.GradNodeB.convStridedW_hasGradAt
#print axioms Proofs.GradNodeB.convStridedB_hasGradAt
#print axioms Proofs.GradNodeB.denseW_hasGradAt
#print axioms Proofs.GradNodeB.denseB_hasGradAt
#print axioms Proofs.GradNodeB.bnGamma_hasGradAt
#print axioms Proofs.GradNodeB.bnBeta_hasGradAt
#print axioms Proofs.GradNodeB.cStridedInB_eq_batchMapBackward
#print axioms Proofs.GradNodeB.dInB_eq_batchMapBackward
#print axioms Proofs.GradNodeB.convStridedXlaW_hasGradAt
#print axioms Proofs.GradNodeB.convStridedXlaB_hasGradAt
#print axioms Proofs.GradNodeB.depthwiseW_hasGradAt
#print axioms Proofs.GradNodeB.depthwiseB_hasGradAt
#print axioms Proofs.GradNodeB.depthwiseStridedXlaW_hasGradAt
#print axioms Proofs.GradNodeB.depthwiseStridedXlaB_hasGradAt
#print axioms Proofs.GradNodeB.depthwiseStridedW_hasGradAt
#print axioms Proofs.GradNodeB.dStridedInB_eq_batchMapBackward
#print axioms Proofs.GradNodeB.hasGradAt_bnBatchLA
#print axioms Proofs.GradNodeB.hasGradAt_relu
#print axioms Proofs.GradNodeB.hasGradAt_conv
#print axioms Proofs.GradNodeB.hasGradAt_convStrided
#print axioms Proofs.GradNodeB.hasGradAt_depthwise
#print axioms Proofs.GradNodeB.hasGradAt_depthwiseStrided
-- ResNet-34: every parameter gradient node IS the batched smoothed loss's derivative (ResNet34ParamGrad.lean)
#print axioms Proofs.ResNet34TieB.r34_idblock_lossTiedB
#print axioms Proofs.ResNet34TieB.r34_downblock_lossTiedB
#print axioms Proofs.ResNet34TieB.r34_stem_lossTiedB
#print axioms Proofs.ResNet34TieB.r34_head_lossTiedB
#print axioms Proofs.ResNet34TieB.r34_factor_a0
#print axioms Proofs.ResNet34TieB.r34_net_lossGrad
#print axioms Proofs.ResNet34TieB.r34_net_lossGrad_smoothedCE
#print axioms Proofs.ResNet34TieB.r34_net_tied_lossGrad
-- ResNet-50: every parameter gradient node IS the loss's derivative, for both shipped losses (ResNet50ParamGrad.lean)
#print axioms Proofs.ResNet50TieB.r50_idblock_lossTiedB
#print axioms Proofs.ResNet50TieB.r50_projblock_lossTiedB
#print axioms Proofs.ResNet50TieB.r50_downblock_lossTiedB
#print axioms Proofs.ResNet50TieB.r50_factor_s1b0
#print axioms Proofs.ResNet50TieB.r50_net_lossGrad
#print axioms Proofs.ResNet50TieB.r50_net_lossGrad_smoothedCE
#print axioms Proofs.ResNet50TieB.r50_net_lossGrad_bce
#print axioms Proofs.ResNet50TieB.r50_net_tied_lossGrad
-- MobileNetV2: every parameter gradient node IS the loss's derivative (MobileNetV2ParamGrad.lean)
#print axioms Proofs.MobileNetV2TieB.mnv2_stem_lossTiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_noexp_lossTiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_stride1_lossTiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_resid_lossTiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_stride2_lossTiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_head_lossTiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_factor_b1
#print axioms Proofs.MobileNetV2TieB.mnv2_net_lossGrad
#print axioms Proofs.MobileNetV2TieB.mnv2_net_lossGrad_smoothedCE
#print axioms Proofs.MobileNetV2TieB.mnv2_net_tied_lossGrad
-- MobileNetV4-Conv-M: every parameter gradient node IS the loss's derivative (MobileNetV4ParamGrad.lean)
#print axioms Proofs.Mnv4TieB.mnv4_stem_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_fused_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_head_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_extradw_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_convnext_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_ffn_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_strided_lossTiedB
#print axioms Proofs.Mnv4TieB.mnv4_factor_stem
#print axioms Proofs.Mnv4TieB.mnv4_factor_b21
#print axioms Proofs.Mnv4TieB.mnv4_net_lossGrad
#print axioms Proofs.Mnv4TieB.mnv4_net_lossGrad_smoothedCE
#print axioms Proofs.Mnv4TieB.mnv4_net_tied_lossGrad
-- EfficientNet-B0: every parameter gradient node IS the loss's derivative (EfficientNetParamGrad.lean)
#print axioms Proofs.GradNodeB.biasBeta_hasGradAt
#print axioms Proofs.GradNodeB.hasGradAt_swish
#print axioms Proofs.GradNodeB.hasGradAt_bnBackB
#print axioms Proofs.BackLinks.seGateMulBHasVJP
#print axioms Proofs.EnetTieG.seB_eq_gateMul
#print axioms Proofs.EnetTieG.enet_se_lossTiedB
#print axioms Proofs.EnetTieG.enet_exp_lossTiedG
#print axioms Proofs.EnetTieG.enet_resid_lossTiedG
#print axioms Proofs.EnetTieG.enet_strided_lossTiedG
#print axioms Proofs.EnetTieG.enet_noexp_lossTiedG
#print axioms Proofs.EnetTieG.enet_stem_lossTiedG
#print axioms Proofs.EnetTieG.enet_head_lossTiedG
#print axioms Proofs.EnetTieG.enet_factor_b1
#print axioms Proofs.EnetTieG.enet_net_lossGrad
#print axioms Proofs.EnetTieG.enet_net_lossGrad_smoothedCE
#print axioms Proofs.EnetTieG.enet_net_tied_lossGrad
-- ConvNeXt-T: every parameter gradient node IS the loss's derivative (ConvNeXtParamGrad.lean)
#print axioms Proofs.hasGradAt_linLoss
#print axioms Proofs.HasGradAt.param_batchMap_through
#print axioms Proofs.smoothedBatchLossDiv_grad
#print axioms Proofs.GradNodeB.pdiv_bias_of_split
#print axioms Proofs.CnxTieGB.cnxBlk_hasGradAt
#print axioms Proofs.CnxTieGB.cnx_block_lossTiedGB
#print axioms Proofs.CnxTieGB.cnx_down_lossTiedGB
#print axioms Proofs.CnxTieGB.cnx_stem_lossTiedGB
#print axioms Proofs.CnxTieGB.cnx_head_lossTiedGB
#print axioms Proofs.CnxTieGB.cnx_factor_b1
#print axioms Proofs.CnxTieGB.cnx_logitsB_eq
#print axioms Proofs.CnxTieGB.cnx_net_lossGrad
#print axioms Proofs.CnxTieGB.cnxNetB_eq_convNextForwardTCh
#print axioms Proofs.CnxTieGB.cnx_net_lossGrad_smoothedCE
#print axioms Proofs.CnxTieGB.cnx_net_tied_lossGrad
-- ViT-Tiny: every parameter gradient node IS the loss's derivative (ViTParamGrad.lean)
#print axioms Proofs.pdivMat_colIndepH
#print axioms Proofs.colSlabwiseHasVJPMatH
#print axioms Proofs.ViTTieGB.attnCoreQHasVJPMat
#print axioms Proofs.ViTTieGB.attnCoreKHasVJPMat
#print axioms Proofs.ViTTieGB.attnCoreVHasVJPMat
#print axioms Proofs.ViTTieGB.attnCoreQ_backward
#print axioms Proofs.ViTTieGB.vitMlpSub_hasGradAt
#print axioms Proofs.ViTTieGB.vitPostQ_hasGradAt
#print axioms Proofs.ViTTieGB.vitPostL1_hasGradAt
#print axioms Proofs.ViTTieGB.vitPostL2_hasGradAt
#print axioms Proofs.ViTTieGB.vitPostF1_hasGradAt
#print axioms Proofs.ViTTieGB.vit_fwd_Wq
#print axioms Proofs.ViTTieGB.vit_block_lossTiedGB
#print axioms Proofs.ViTTieGB.vit_head_lossTiedGB
#print axioms Proofs.ViTTieGB.vit_embed_lossTiedGB
#print axioms Proofs.ViTTieGB.vit_factor_b1
#print axioms Proofs.ViTTieGB.vit_logitsB_eq
#print axioms Proofs.ViTTieGB.vit_net_lossGrad
#print axioms Proofs.ViTTieGB.fwdO_eq_blockVFlat
#print axioms Proofs.ViTTieGB.vitNetB_eq_vitForwardKV
#print axioms Proofs.ViTTieGB.vit_net_lossGrad_smoothedCE
#print axioms Proofs.ViTTieGB.vit_net_tied_lossGrad
-- The chapter nets: every parameter gradient node IS the loss's derivative, pools up to twins
-- (SmallParamGrad.lean and the Nets/Small *ParamGrad files)
#print axioms Proofs.SmallParamGrad.hasGradAt_crossEntropy
#print axioms Proofs.SmallParamGrad.convW_hasGradAt
#print axioms Proofs.SmallParamGrad.convB_hasGradAt
#print axioms Proofs.SmallParamGrad.denseW_hasGradAt
#print axioms Proofs.SmallParamGrad.denseB_hasGradAt
#print axioms Proofs.SmallParamGrad.convWeightSgd_eq_grad
#print axioms Proofs.SmallParamGrad.weightSgd_eq_grad
#print axioms Proofs.SmallParamGrad.hasGradAt_conv
#print axioms Proofs.SmallParamGrad.poolGatherFlat_eq_sel
#print axioms Proofs.SmallParamGrad.hasGradAt_gatherRelu
#print axioms Proofs.SmallParamGrad.maxPool_relu_eventuallyEq_sel
#print axioms Proofs.SmallParamGrad.maxpool_flatDenote_eq_selScatter
#print axioms Proofs.LinFold.linear_net_lossGrad
#print axioms Proofs.LinFold.linear_net_lossGrad_CE
#print axioms Proofs.MlpFold.mlp_net_lossGrad
#print axioms Proofs.MlpFold.mlp_net_lossGrad_CE
#print axioms Proofs.CnnFold.cnnPoolTwin_of_convPatchEq2
#print axioms Proofs.CnnFold.cnn_net_lossGrad
#print axioms Proofs.CnnFold.cnn_net_lossGrad_CE
#print axioms Proofs.CnnFold.cnnChainCotW2_eq_sel
#print axioms Proofs.CifarFold.cifar_net_lossGrad
#print axioms Proofs.CifarFold.cifar_net_lossGrad_CE
#print axioms Proofs.CifarFold.cifarChainCotW2_eq_sel
#print axioms Proofs.Cifar8TieG.cifar8_net_lossGrad
#print axioms Proofs.Cifar8TieG.cifar8_net_lossGrad_CE
#print axioms Proofs.Cifar8BnTieG.bnGamma_hasGradAt
#print axioms Proofs.Cifar8BnTieG.bnBeta_hasGradAt
#print axioms Proofs.Cifar8BnTieG.hasGradAt_bnPC
#print axioms Proofs.Cifar8BnTieG.cifar8Bn_net_lossGrad
#print axioms Proofs.Cifar8BnTieG.cifar8Bn_net_lossGrad_CE
-- Inexact-gradient descent over ℝ (SgdDescent/Basic.lean)
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
-- The η-composition, the "two halves finally meet"
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
-- Output-layer η-composition (for the MLP)
#print axioms mlp_output_float_sgd_descends
-- Hidden-layer float-backward grad-close (the joint-step engine)
#print axioms FloatModel.cotErr_nonneg
#print axioms mlp_w1_grad_close
-- Hidden-layer η-composition (descent)
#print axioms FloatModel.mlpHiddenFloatGrad
#print axioms mlpHiddenFloatGrad_apply
#print axioms mlp_hidden_loss_gradAt_reluMask
#print axioms mlp_hidden_float_sgd_descends
#print axioms mlp_input_loss_differentiableAt
#print axioms mlp_input_loss_gradAt
#print axioms mlp_input_loss_grad_lipschitz
#print axioms mlp_input_sgd_descends
-- Input-layer η-composition (descent)
#print axioms reluMask_dense_transpose_eq
#print axioms FloatModel.mlpInputFloatGrad
#print axioms mlpInputFloatGrad_apply
#print axioms mlp_input_loss_gradAt_reluMask
#print axioms mlp_w0_grad_close
#print axioms mlp_input_float_sgd_descends
-- Dense-bias descent (SgdDescent/MlpBias.lean)
#print axioms gradAt_bias_eq_pdiv
#print axioms linear_bias_sgd_descends
#print axioms mlp_hidden_bias_loss_grad_lipschitz
#print axioms mlp_hidden_bias_sgd_descends
#print axioms mlp_input_bias_loss_grad_lipschitz
#print axioms mlp_input_bias_sgd_descends
-- The descent program reaches the Chapter-4 CNN (SgdDescent/Cnn.lean)
#print axioms flatten_t3Idx
#print axioms sum_t3
#print axioms sum_window_cells
#print axioms lt_of_lt_gap_of_close
#print axioms MaxPool2MarginQ.smooth_of_close
#print axioms MaxPool2MarginQ.smooth
#print axioms MaxPool2MarginQ.isArgmax_iff
-- Float-bridge CNN descent keystone: the pool selector is an indicator pass-through in float
#print axioms MaxPool2MarginQ.poolBack_close
-- The pool margin up to twins, and the pool as a fixed gather at a twin tie (ConvIndex, WindowMax)
#print axioms windowSmoothUpTo_of_margin
#print axioms WindowMarginUpTo.mono
#print axioms windowMarginUpTo_of_cert
#print axioms windowMax_eq_windowGather
#print axioms poolGatherFlat_l1_contract
#print axioms pdiv_poolGatherFlat
#print axioms maxPoolFlat_eq_poolGatherFlat
#print axioms MaxPool2MarginQ.to_marginQUpTo
#print axioms MaxPool2MarginQ.to_marginQUpTo_flat
#print axioms conv2d_eq_convPad
#print axioms abs_convPad_le
#print axioms sum_abs_kernel_slab_le
#print axioms sum_abs_k4
#print axioms conv2d_kernel_sub
#print axioms conv2d_kernel_drift
#print axioms conv2d_kernel_drift_total
#print axioms conv2d_kernel_drift_sum
-- The conv2-layer rung, assembled (SgdDescent/Cnn.lean)
#print axioms ce_head3_input_grad
#print axioms pool_relu_input_grad
#print axioms conv2d_weight_pdiv
#print axioms cnn_conv2_loss_gradAt
-- Float-bridge CNN descent keystone: the certified conv-2 gradient in dense/reluMask form
#print axioms dense_transpose_eq
#print axioms head3_cot_reluMask
#print axioms cnn_conv2_loss_gradAt_reluMask
-- Float-bridge CNN descent
#print axioms t3Idx_surj
#print axioms mask_scalar_close
#print axioms FloatModel.dot_perturbed_close
#print axioms FloatModel.cnnConv2FloatGrad
#print axioms FloatModel.cnnConv2FloatGrad_apply
#print axioms FloatModel.cnnConv2GradBudget
#print axioms cnn_conv2_grad_close
#print axioms head3_sum_drift
#print axioms conv2d_eq_of_convPatchEq
#print axioms convPatchEq_relu_conv
#print axioms gather_relu_input_grad
#print axioms Conv2Slot.maxPool_relu_eventuallyEq_gather
#print axioms Conv2Slot.marginUpTo_seg
#print axioms Conv2Slot.gather_grad_lipschitz
#print axioms Conv2Slot.sgd_descends
#print axioms Conv2Slot.marginUpTo_strict
#print axioms Conv2Slot.gradAt_abs_le
#print axioms cnn_conv2_sgd_descends
#print axioms cnn_conv2_exact_sgd_descends
-- Float-bridge CNN descent (cont.)
#print axioms flatten_k4Idx
#print axioms k4Idx_surj
#print axioms cnn_conv2_float_sgd_descends
-- The conv1 rung (SgdDescent/Cnn.lean)
#print axioms abs_convTap_expand
#print axioms convTap_out_l1
#print axioms conv2d_input_pdiv3
#print axioms conv2d_flat_input_pdiv
#print axioms conv2d_input_entry_drift
#print axioms conv2d_input_l1_drift
#print axioms cnn1_pool_head_input_grad
#print axioms cnn_conv1_loss_gradAt
-- Float-bridge CNN descent keystone
#print axioms cnn_conv1_loss_gradAt_reluMask
#print axioms Conv1Slot.sgd_descends
#print axioms Conv1Slot.gradAt_abs_le
#print axioms cnn_conv1_sgd_descends
#print axioms ConvPatchEq2.symm
#print axioms ConvPatchEq2.trans
#print axioms cnn_conv1_exact_sgd_descends
-- Float-bridge CNN descent (cont.)
#print axioms convTap_abs_le
#print axioms FloatModel.cnnConv2CotBudget
#print axioms FloatModel.cnnConv2CotMag
#print axioms cnn_conv2_cot_close
#print axioms cnn_conv2_cot_real_abs_le
#print axioms abs_le_of_close
#print axioms convTap_back_close
#print axioms convTap_back_abs_le
#print axioms FloatModel.cnnConv1CotF
#print axioms cnnConv1CotR
#print axioms FloatModel.cnnConv1CotBudget
#print axioms cnn_conv1_cot_close
#print axioms FloatModel.cnnConv1FloatGrad
#print axioms FloatModel.cnnConv1FloatGrad_apply
#print axioms FloatModel.cnnConv1GradBudget
#print axioms cnn_conv1_grad_close
#print axioms cnn_conv1_float_sgd_descends
-- The conv BIAS rungs (SgdDescent/Cnn.lean)
#print axioms conv2d_bias_sub
#print axioms conv2d_flat_bias_drift_total
#print axioms conv2d_flat_bias_drift_sum
#print axioms conv2d_bias_pdiv
#print axioms cnn_conv2_bias_loss_gradAt
#print axioms cnn_conv2_bias_sgd_descends
#print axioms cnn_conv2_bias_exact_sgd_descends
#print axioms cnn_conv1_bias_loss_gradAt
#print axioms cnn_conv1_bias_sgd_descends
#print axioms cnn_conv1_bias_exact_sgd_descends
-- Float-bridge CNN descent (cont.)
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
-- Adam/AdamW optimizer step over ℝ
#print axioms adamVNext_nonneg
#print axioms adam_denom_pos

-- EfficientNet backward-graph faithfulness (den-level)
#print axioms StableHLO.residualBackGraph_faithful
#print axioms StableHLO.seBlockBackGraph_faithful
#print axioms StableHLO.bnBack_faithful_fn
-- The channel-LN backward graphs
#print axioms Proofs.rowLNBack_affine_eq
#print axioms StableHLO.chanLNBackGraph_faithful
#print axioms StableHLO.chanLNBackGraph_eq_vjp
#print axioms StableHLO.cnxBlockBodyChBackGraph_faithful
#print axioms StableHLO.cnxDownChBackGraph_faithful
#print axioms StableHLO.bnBatchBack_faithful
#print axioms StableHLO.convBackBatched_faithful
#print axioms StableHLO.depthwiseBackBatched_faithful
#print axioms StableHLO.bnBatchLA_back_conj
#print axioms StableHLO.bnBatchLABack_faithful
-- the render's `.bnBatchBack` node and the ties' `.bnBatchLABack` node denote one map (up to reassocB)
#print axioms BackLinks.den_bnBatchLABack_eq_bnBatchBack
#print axioms BackLinks.bnBackB_eq_den_bnBatchBack
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
-- Capstone: the whole batched ResNet-34 identity basic block backward graph
#print axioms StableHLO.r34BasicBlockBackBatchedGraph_faithful

-- ResNet-34 DOWNSAMPLE/STRIDED basic block backward-graph faithfulness
#print axioms StableHLO.convStridedBackBatched_faithful
-- Capstone: the whole batched ResNet-34 downsample basic block backward graph
#print axioms StableHLO.r34DownBlockBackBatchedGraph_faithful

-- ViT whole-block backward-graph faithfulness, lifted to GENERAL MULTI-HEAD
#print axioms StableHLO.mhsa_backward_collapseMH
#print axioms StableHLO.mhsaBackGraphMH_faithful

-- ViT whole-block backward-graph faithfulness at the FULL PRODUCTION config
#print axioms StableHLO.rowVecLNBack_eq_backward
#print axioms StableHLO.mlpSublayerVBackGraph_faithfulMH
#print axioms StableHLO.attnSublayerVBackGraphMH_faithful
#print axioms StableHLO.transformerBlockVBackGraphMH_faithful

-- ViT WHOLE-NET backward-graph faithfulness
#print axioms StableHLO.classifierBackGraph_faithful
#print axioms StableHLO.finalLNBackGraph_faithful
#print axioms StableHLO.transformerBlockVBackGraphMHP_faithful
#print axioms StableHLO.patchEmbedBackGraph_faithful
#print axioms StableHLO.vitNetBackGraph_faithful

-- ViT folded onto the net-agnostic `CertLayer` machinery (ViTBackNet.lean)
#print axioms StableHLO.vitTrunkV_fwd
#print axioms StableHLO.vitTrunkV_graph

-- THE WHOLE NET AS ONE `CertLayer`
#print axioms StableHLO.vitPatchEmbedLayer
#print axioms StableHLO.vitFinalLNLayer
#print axioms StableHLO.vitClassifierLayer
#print axioms StableHLO.vitNetLayer_fwd
#print axioms StableHLO.vitNetLayer_graph
#print axioms StableHLO.vitNetLayer_ok

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

-- r34 / mnv2 / enet / convnext
#print axioms StableHLO.cnxBlockChLayer
#print axioms StableHLO.r34BasicBlockLayer
#print axioms StableHLO.r34DownBlockLayer

-- MNv4 — the four UIB families COLLAPSED into one body

-- MNv4's NET level (T1, T2)
#print axioms StableHLO.mobilenetv4ForwardBFullHasVJPAt
#print axioms StableHLO.mobilenetv4ForwardBFullHasVJPAt_correct
#print axioms StableHLO.mobilenetv4ForwardBFull_differentiableAt
#print axioms StableHLO.mnv4FwdGraphBFull_faithful
#print axioms StableHLO.mnv4ExtraDWBodyGraphB_faithful
#print axioms StableHLO.mnv4ConvNeXtBodyGraphB_faithful
#print axioms StableHLO.mnv4FfnBodyGraphB_faithful
#print axioms StableHLO.mnv4StridedGraphB_faithful

-- MNv4's T3 tie
#print axioms Mnv4TieB.mnv4_extradw_tiedB
#print axioms Mnv4TieB.mnv4_convnext_tiedB
#print axioms Mnv4TieB.mnv4_ffn_tiedB
#print axioms Mnv4TieB.mnv4_strided_tiedB
#print axioms Mnv4TieB.mnv4_stem_tiedB
#print axioms Mnv4TieB.mnv4_fused_tiedB
#print axioms Mnv4TieB.mnv4_head_tiedB
#print axioms Mnv4TieB.mnv4BodyCotIn_eq_vjp
#print axioms Mnv4TieB.mnv4SBodyCotIn_eq_vjp
#print axioms Mnv4TieB.mnv4SkipCotIn_eq_vjp
#print axioms Mnv4TieB.mnv4FusedCotIn_eq_vjp
#print axioms Mnv4TieB.mnv4HeadCotIn_eq_vjp
#print axioms Mnv4TieB.mnv4_net_tiedB
#print axioms Mnv4TieB.mnv4_lossCot_is_smoothedCE_grad

-- MNv4's T6 -- the certified whole-net input gradient
#print axioms Proofs.mnv4StemBBack_eq_vjp_backward
#print axioms Proofs.cbReluBBack_eq_vjp_backward
#print axioms Proofs.mnv4Hc2BBack_eq_vjp_backward
#print axioms Proofs.mnv4ClsBBack_eq_vjp_backward
#print axioms Proofs.mnv4BFullHasVJPAt
#print axioms Proofs.mnv4InputGradB_eq_mnv4B_full_vjp
#print axioms Proofs.mnv4InputGradB_correct
#print axioms Proofs.mobilenetv4ForwardBFull_eq_slots

-- EfficientNet — the VJP-without-backward-graph holes, closed
#print axioms StableHLO.mbNoExpBackBatchedGraph_faithful
#print axioms StableHLO.headBackBatchedGraph_faithful
-- ViT-Tiny FOLD (ViTFold)
#print axioms Proofs.SgdNode.veclnGammaSgd_den
#print axioms Proofs.ViTFold.rowDenseWeightSgd_den
#print axioms Proofs.ViTFold.rowDenseBiasSgd_den
#print axioms Proofs.SgdNode.rowDenseBiasSgd_den_lnbeta
#print axioms Proofs.ViTFold.patchEmbedWeightSgd_den
#print axioms Proofs.ViTFold.patchEmbedBiasSgd_den
#print axioms Proofs.ViTFold.posEmbedSgd_den
-- ViT-Tiny TIE — MULTI-HEAD promotion (ViTMultiHeadChain + ViTStepTie)
#print axioms Proofs.vitCotDQmh_eq
#print axioms Proofs.vitCotDKmh_eq
#print axioms Proofs.vitCotDVmh_eq
-- The multi-head per-block tie (vit_block_tiedMHV)
#print axioms Proofs.ViTTie.vit_block_tiedMHV
#print axioms Proofs.ViTTie.vit_block_tiedAtMHV
-- ViT-Tiny TIE — the ALL-200-PARAMS capstone (vit_net_tied_certified)
#print axioms Proofs.ViTTie.vit_cls_den
#print axioms Proofs.ViTTie.vit_finalLN_tied
#print axioms Proofs.ViTTie.vit_head_tied
#print axioms Proofs.ViTTie.vit_embed_tied
#print axioms Proofs.ViTTie.vit_net_tied_certified

-- Robustness certificate (the cert side of cert ≤ TRUE ≤ PGD)
#print axioms Proofs.lipschitz_margin_certified_radius
#print axioms Proofs.logit_gap_stable
#print axioms Proofs.coord_pair_bound
#print axioms Proofs.lipschitzL2_iff_lipschitzWith
#print axioms Proofs.LipschitzL2.mono
#print axioms Proofs.LipschitzL2.comp
#print axioms Proofs.clm_lipschitzL2

-- Randomized-smoothing certified radius (Cohen–Rosenfeld–Kolter 2019)
#print axioms Proofs.smoothed_margin_certified_radius

-- The smoothing radius at the REAL Gaussian quantile (Smoothing/Gaussian.lean)
#print axioms Proofs.smoothing_certified_radius_probit
#print axioms Proofs.stdNormalCDF_strictMono
#print axioms Proofs.stdNormalCDF_neg
#print axioms Proofs.stdNormalQuantile_monotoneOn
#print axioms Proofs.stdNormalQuantile_anti
#print axioms Proofs.smoothing_certified_radius_gaussian

-- The 1-D Neyman–Pearson core
#print axioms Proofs.stdNormalCDF_quantile

-- Dimension reduction
#print axioms MathlibUpstream.integral_gaussianReal_comp_add_const
#print axioms Proofs.pi_gaussian_shift_eq
#print axioms Proofs.pi_gaussian_np_shift
#print axioms Proofs.stdGaussian_np_shift

-- Assembly — the Cohen radius with NOTHING left on the smoothing side
#print axioms Proofs.stdNormalQuantile_cdf
#print axioms Proofs.stdNormalCDF_mem_Ioo
#print axioms Proofs.smoothing_probit_lipschitz
#print axioms Proofs.smoothing_certified_radius_cohen
#print axioms Proofs.smoothing_certified_radius_classifier
-- ...and the MONTE-CARLO tie (Smoothing/MC.lean)
#print axioms Proofs.mc_mean_lower_bound
#print axioms Proofs.stdNormalQuantile_of_nonpos
#print axioms Proofs.smoothing_mc_certified

-- ...and the EXACT Clopper-Pearson tie (Smoothing/CP.lean)
#print axioms Proofs.pi_hitCount_eq_binomial
#print axioms Proofs.pi_hitCount_tail_real
#print axioms Proofs.binomTail_le_of_lt_cpLower
#print axioms Proofs.cp_coverage
#print axioms Proofs.smoothing_cp_certified

-- ...and the SOLVED form (the per-image scorecard shape)
#print axioms Proofs.binomTail_monotoneOn
#print axioms Proofs.le_cpLower_of_tail_le
#print axioms Proofs.smoothing_cp_certified_solved

-- ...and the KERNEL ENGINE for driver-scale tail checks (the ListDot recipe)
#print axioms Proofs.binomTailNum_eq
#print axioms Proofs.binomTailGo_eq
#print axioms Proofs.binomTailNumFast_eq
#print axioms Proofs.binomTail_eq_kernel
#print axioms Proofs.binomTail_le_of_kernel_check

-- ...and the SCORECARD (Smoothing/CPScorecard.lean, generated)
#print axioms Proofs.smoothCpMlp_tail_le
#print axioms Proofs.smoothCpCnn_tail_le
#print axioms Proofs.smoothCpCifar_tail_le

-- ...and certified DECIMAL quantile bounds (Smoothing/PhiBounds.lean)
#print axioms Proofs.stdNormalCDF_panel
#print axioms Proofs.stdNormalCDF_le_phiGridUB
#print axioms Proofs.le_stdNormalQuantile_of_grid

-- ...and the DECIMAL-radius SCORECARD (the prefix-scan corpus pass)
#print axioms Proofs.phiScanRev_getD
#print axioms Proofs.phiScanRevFrom_append
#print axioms Proofs.le_stdNormalQuantile_of_scan
#print axioms Proofs.smooth_radius_dec
#print axioms Proofs.smoothDecMlp_radius_le
#print axioms Proofs.smoothDecCnn_radius_le
#print axioms Proofs.smoothDecCifar_radius_le

-- ...and the NET-SEMANTICS closure (Smoothing/NetSemantics.lean)
#print axioms Proofs.measurable_argmaxNet
#print axioms Proofs.argmaxNet_smoothProb_mem_Ioo
#print axioms Proofs.smoothing_cp_certified_net
#print axioms Proofs.Robustness.mlpT_logit_continuous
#print axioms Proofs.Robustness.netW_strict
#print axioms Proofs.Robustness.smoothing_cp_certified_mlpT
#print axioms Proofs.Robustness.smooth_cp_mlpT_demo

-- ...and the two-sided quantile packaging (Smoothing/Gaussian.lean)
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

-- ...and the Tsuzuku certificate INSTANTIATED (LipschitzCert/Instance.lean)
#print axioms Proofs.Robustness.denseE_lipschitzL2
#print axioms Proofs.Robustness.reluE_lipschitzL2
#print axioms Proofs.Robustness.linear_demo_certified
#print axioms Proofs.Robustness.linear_radius_pos
#print axioms Proofs.Robustness.mlp_lip
#print axioms Proofs.Robustness.mlp_demo_certified
#print axioms Proofs.Robustness.mlp_radius_pos
#print axioms Proofs.Robustness.W1t_lip
#print axioms Proofs.Robustness.W2t_lip
#print axioms Proofs.Robustness.mlpT_lip
#print axioms Proofs.Robustness.xt_margin
#print axioms Proofs.Robustness.trained_radius_pos
#print axioms Proofs.Robustness.trained_demo_certified
-- ...tightened by the power-iteration certificate (certified two-sided spectral sandwich)
#print axioms Proofs.Robustness.sum_sq_matvec_le
#print axioms Proofs.Robustness.denseE_lipschitzL2_gram
#print axioms Proofs.Robustness.lipschitzL2_lower_euclid
#print axioms Proofs.Robustness.G1t_eq
#print axioms Proofs.Robustness.G2t_eq
#print axioms Proofs.Robustness.W1t_lip_gram
#print axioms Proofs.Robustness.W2t_lip_gram
#print axioms Proofs.Robustness.mlpT_lip_gram
#print axioms Proofs.Robustness.trained_radius_gram_pos
#print axioms Proofs.Robustness.trained_demo_certified_gram
#print axioms Proofs.Robustness.W1t_lip_lower
#print axioms Proofs.Robustness.W2t_lip_lower
-- ...iterated once more (Schatten-8)
#print axioms Proofs.Robustness.sum_sq_matTvec_eq
#print axioms Proofs.Robustness.denseE_lipschitzL2_gram2
#print axioms Proofs.Robustness.H1t_eq
#print axioms Proofs.Robustness.H2t_eq
#print axioms Proofs.Robustness.W1t_lip_gram2
#print axioms Proofs.Robustness.W2t_lip_gram2
#print axioms Proofs.Robustness.mlpT_lip_gram2
#print axioms Proofs.Robustness.trained_radius_gram2_pos
#print axioms Proofs.Robustness.trained_demo_certified_gram2

-- CERTIFIED-ACCURACY SCORECARD (LipschitzCert/Scorecard.lean)
#print axioms Proofs.Robustness.sqrt_two_le_rat
#print axioms Proofs.Robustness.certified_at_eps
#print axioms Proofs.Robustness.G1s_eq
#print axioms Proofs.Robustness.G2s_eq
#print axioms Proofs.Robustness.H1s_eq
#print axioms Proofs.Robustness.H2s_eq
#print axioms Proofs.Robustness.W1s_lip_gram2
#print axioms Proofs.Robustness.W2s_lip_gram2
#print axioms Proofs.Robustness.mlpS_lip_gram2
#print axioms Proofs.Robustness.marginC0
#print axioms Proofs.Robustness.certifiedC0
#print axioms Proofs.Robustness.certifiedC3
#print axioms Proofs.Robustness.certifiedC10
#print axioms Proofs.Robustness.certifiedC13
#print axioms Proofs.Robustness.certifiedC14
#print axioms Proofs.Robustness.certifiedC17
#print axioms Proofs.Robustness.certifiedC25
#print axioms Proofs.Robustness.marginU82
#print axioms Proofs.Robustness.certifiedU82
-- the mechanized aggregate
#print axioms Proofs.Robustness.cappedCerts_certified
#print axioms Proofs.Robustness.unconCerts_certified
#print axioms Proofs.Robustness.scorecard

-- Per-pair LipSDP tightening (LipschitzCert/PairSDP.lean + the generated instances)
#print axioms Proofs.Robustness.relu_slope_restricted
#print axioms Proofs.Robustness.pair_sq_bound
#print axioms Proofs.Robustness.mlp_gap_eq
#print axioms Proofs.Robustness.certified_at_eps_pair
#print axioms Proofs.Robustness.hS01C
#print axioms Proofs.Robustness.pairSqC_0_1
#print axioms Proofs.Robustness.pairSqC_7_0
#print axioms Proofs.Robustness.pairSqU_0_1
#print axioms Proofs.Robustness.certifiedSC0
#print axioms Proofs.Robustness.certifiedSC4
#print axioms Proofs.Robustness.certifiedSC9
#print axioms Proofs.Robustness.certifiedSU0
#print axioms Proofs.Robustness.certifiedSU4
#print axioms Proofs.Robustness.certifiedSU11
#print axioms Proofs.Robustness.sdpCappedCerts_certified
#print axioms Proofs.Robustness.sdpUnconCerts_certified
#print axioms Proofs.Robustness.scorecard_sdp
#print axioms Proofs.Robustness.scorecard_sdp_uncon

-- The certificate × float bridge (LipschitzCert/Float.lean)
#print axioms Proofs.FloatModel.mlp2_float_close_uniform
#print axioms Proofs.Robustness.certified_at_eps_close
#print axioms Proofs.Robustness.capped_B_le
#print axioms Proofs.Robustness.real_tie
#print axioms Proofs.Robustness.certifiedFloat_of_margin
#print axioms Proofs.Robustness.certifiedC0_float

-- The kernel-dotZ list engine (ListDot.lean)
#print axioms Proofs.dotZ_comm
#print axioms Proofs.sum_getD_mul
#print axioms Proofs.sum_getD_div
#print axioms Proofs.sum_getD_abs
#print axioms Proofs.sum_getD_abs_div
#print axioms Proofs.Robustness.denseLo_le
#print axioms Proofs.Robustness.le_denseHi
#print axioms Proofs.Robustness.relu_box
#print axioms Proofs.Robustness.denseLo_uniform
#print axioms Proofs.Robustness.denseLo2_eval
#print axioms Proofs.Robustness.denseHi2_eval
-- The dense tier's certificate is stated on a BRACKET, not on interval arithmetic
#print axioms Proofs.Robustness.BoxSoundE.comp
#print axioms Proofs.Robustness.denseE_boxSound
#print axioms Proofs.Robustness.reluE_boxSound
#print axioms Proofs.Robustness.certified_of_boxSound
#print axioms Proofs.Robustness.mlp2_boxSound
#print axioms Proofs.Robustness.ibp2_certified_at_eps

-- CROWN (Certificates/CrownBound.lean)
#print axioms Proofs.Robustness.certified_of_marginPos
#print axioms Proofs.Robustness.relu_lower_envelope
#print axioms Proofs.Robustness.relu_upper_envelope
#print axioms Proofs.Robustness.reluLB_dead
#print axioms Proofs.Robustness.reluLB_active
#print axioms Proofs.Robustness.reluLB_unstable_pos
#print axioms Proofs.Robustness.reluLB_unstable_neg
#print axioms Proofs.Robustness.crownRow_dot
#print axioms Proofs.Robustness.linf_lower_bound
#print axioms Proofs.Robustness.crown_margin_ge
#print axioms Proofs.Robustness.crown2_certified_at_eps

-- IBP PAST THE TWO-LAYER DENSE WALL (Certificates/IntervalBoundConv/Basic.lean)
#print axioms Proofs.IBP.BoxSound3.comp
#print axioms Proofs.IBP.BoxSound3V.comp3
#print axioms Proofs.IBP.denseT_boxSound3V
#print axioms Proofs.IBP.reluT_boxSound3
#print axioms Proofs.IBP.conv2d_boxSound3
#print axioms Proofs.IBP.maxPool2_boxSound3
-- the conv peer of denseLo_uniform
#print axioms Proofs.IBP.convLo_uniform
#print axioms Proofs.IBP.convHi_uniform
-- DEPTH, concretely
#print axioms Proofs.IBP.deepNet_boxSound
-- capstone + radius monotonicity
#print axioms Proofs.IBP.ibp3_certified_of_boxSound
#print axioms Proofs.IBP.CertifiedAtLinf3.mono

-- THE IEEE AXIOMS, DISCHARGED (Binary32Instance.lean)
#print axioms Proofs.rndP_err
#print axioms Proofs.binary32_e4m3_argmax_preserved
#print axioms Proofs.binary32_e4m3_argmax_small
#print axioms Proofs.binary32_linear_sgd_descends_concrete

-- DESCENT AT TRAINED WEIGHTS (Trained/LinearDescent.lean)
#print axioms Proofs.TrainedLinearDescent.hz_lbl_le
#print axioms Proofs.TrainedLinearDescent.sm_lbl_le_half
#print axioms Proofs.TrainedLinearDescent.gradL1_le
#print axioms Proofs.TrainedLinearDescent.gradSq_lower
#print axioms Proofs.TrainedLinearDescent.hdelta
#print axioms Proofs.TrainedLinearDescent.heta
#print axioms Proofs.TrainedLinearDescent.trained_linear_sgd_descends_concrete
#print axioms Proofs.TrainedLinearDescent.trained_linear_sgd_strictly_descends

-- TRAINED-WEIGHT whole-net VJP witness (Trained/MlpWitness.lean)
#print axioms Proofs.TrainedMlp.preact_eq
#print axioms Proofs.TrainedMlp.preact_ne
#print axioms Proofs.TrainedMlp.pdiv_fwd
#print axioms Proofs.TrainedMlp.pdiv_fwd_val
#print axioms Proofs.TrainedMlp.trainedMlp_backward_nontrivial
#print axioms Proofs.TrainedMlp.trainedMlp_jacobian_nonzero
#print axioms Proofs.TrainedMlp.trainedMlp_not_constant

-- TRAINED-WEIGHT whole-net VJP witness, CNN rung (Trained/CnnWitness.lean)
#print axioms Proofs.TrainedCnn.conv1_eq
#print axioms Proofs.TrainedCnn.conv2_eq
#print axioms Proofs.TrainedCnn.r2_smooth
#print axioms Proofs.TrainedCnn.pooled_eq
#print axioms Proofs.TrainedCnn.d3_ne
#print axioms Proofs.TrainedCnn.d4_ne
#print axioms Proofs.TrainedCnn.trainedCnnHasVJP_correct

-- CNN descent at trained weights through tied pool windows (Trained/CnnDescent.lean)
#print axioms Proofs.TrainedCnnDescent.pool_margin
#print axioms Proofs.TrainedCnnDescent.x1_eq
#print axioms Proofs.TrainedCnnDescent.trained_cnn_conv2_sgd_descends_concrete
#print axioms Proofs.TrainedCnnDescent.trained_cnn_conv2_bias_sgd_descends_concrete
-- CNN conv1 descent at trained weights through two-layer twins (Trained/CnnDescentConv1.lean)
#print axioms Proofs.TrainedCnnDescentConv1.pool_margin
#print axioms Proofs.TrainedCnnDescentConv1.x1_fun
#print axioms Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_sgd_descends_concrete
#print axioms Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_bias_sgd_descends_concrete
-- Level-3 seal for the CNN witness (Trained/CnnSeal.lean)
#print axioms Proofs.TrainedCnn.S2
#print axioms Proofs.TrainedCnn.S1
#print axioms Proofs.TrainedCnn.pdiv_fwd_entry
#print axioms Proofs.TrainedCnn.pdiv_fwd_entry_ne
#print axioms Proofs.TrainedCnn.trainedCnn_backward_nontrivial
#print axioms Proofs.TrainedCnn.trainedCnn_jacobian_nonzero
#print axioms Proofs.TrainedCnn.trainedCnn_not_constant

-- Muon geometry
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
-- Newton–Schulz
#print axioms Proofs.MuonNewtonSchulz.nsStep_spectral
#print axioms Proofs.MuonNewtonSchulz.nsStep_iterate_spectral
-- Newton–Schulz (the scalar engine)
#print axioms Proofs.MuonNewtonSchulz.scalar_iterate_tendsto_one
#print axioms Proofs.MuonNewtonSchulz.gCubic_eq_nsScalar
#print axioms Proofs.MuonNewtonSchulz.gCubic_iterate_tendsto_one
#print axioms Proofs.MuonNewtonSchulz.q5Scalar_eq_nsScalar
#print axioms Proofs.MuonNewtonSchulz.q5Scalar_iterate_tendsto_one
-- Newton–Schulz (CLOSES THE LOOP)
#print axioms Proofs.MuonNewtonSchulz.nsStep_iterate_tendsto_polar
#print axioms Proofs.MuonNewtonSchulz.nsStep_cubic_iterate_tendsto_polar
#print axioms Proofs.MuonNewtonSchulz.nsStep_q5_iterate_tendsto_polar
-- Newton–Schulz (the HONEST tier)
#print axioms Proofs.MuonNewtonSchulz.qScalar_one_lt_one
#print axioms Proofs.MuonNewtonSchulz.qScalar_half_gt_one
#print axioms Proofs.MuonNewtonSchulz.qScalar_not_le_one
#print axioms Proofs.MuonNewtonSchulz.qScalar_iterate_band_half

-- Spec→math ties (SpecVJP.lean)
#print axioms linearVerified_denote_eq
#print axioms linearVerified_fwd_faithful
#print axioms linearVerified_lossCot_isCEgrad
#print axioms mlpVerified_denote_eq
#print axioms mlpVerified_fwd_faithful
#print axioms mlpVerified_back_faithful
#print axioms cnnVerified_denote_eq
#print axioms cnnVerified_fwd_faithful
#print axioms cifarVerified_denote_eq
#print axioms cifarVerified_fwd_faithful
-- mnv2 committed-spec tie, at batch BN — the net every shipped MobileNetV2 artifact runs
#print axioms mobilenetv2VerifiedB_denote_eq
#print axioms mobilenetv2VerifiedB_fwd_faithful
-- FULL committed-spec ties (unified weight bundles); r34's at batch BN, the net every shipped artifact runs
#print axioms resnet34VerifiedB_denote_eq
#print axioms resnet34VerifiedB_fwd_faithful
#print axioms efficientnetVerified_denote_eq
#print axioms efficientnetVerified_fwd_faithful
#print axioms convnextVerified_denote_eq
#print axioms convnextVerified_fwd_faithful
#print axioms vitVerified_denote_eq
#print axioms vitVerifiedHasVJP
#print axioms vitVerified_fwd_faithful

-- The CANONICAL MNIST MLP surface (MlpCanonical.lean)
#print axioms Proofs.MlpCanonical.hasVJPAt
#print axioms Proofs.MlpCanonical.output_float_sgd_descends
#print axioms Proofs.MlpCanonical.hidden_float_sgd_descends
#print axioms Proofs.MlpCanonical.input_float_sgd_descends
#print axioms Proofs.MlpCanonical.w1_grad_close
#print axioms Proofs.MlpCanonical.w0_grad_close
#print axioms Proofs.MlpCanonical.train_step_tied_certified

-- The bf16-MIXED conv, composed (ConvMixedComposeBridge.lean)
#print axioms Proofs.convFanS_le
#print axioms Proofs.conv2d_sub_abs_le
#print axioms Proofs.FloatModel.convMixed_close_prop
#print axioms Proofs.FloatModel.flatConvMixed_close

-- The bf16-mixed DEPTHWISE (DepthwiseMixedFloatBridge.lean)
#print axioms Proofs.depthwiseConv2d_eq_dw_dot
#print axioms Proofs.FloatModel.depthwise_close_mixed

-- RESNET-34 AT TRUE BATCH BN — T1-forward and T2 (ResNet34FullB.lean)
#print axioms Proofs.resnet34ForwardBFull
#print axioms Proofs.StableHLO.r34IdGraphB_faithful
#print axioms Proofs.StableHLO.r34DownGraphB_faithful
#print axioms Proofs.StableHLO.r34StemGraphB_faithful
#print axioms Proofs.StableHLO.r34HeadGraphB_faithful
#print axioms Proofs.StableHLO.resnet34FwdGraphBFull_faithful

-- `batchMap` AT A POINT (BatchMapVJPAt.lean)
#print axioms Proofs.pdivMat_rowIndep_at
#print axioms Proofs.batchMap_differentiableAt
#print axioms Proofs.pdiv_batchMap_at
#print axioms Proofs.batchMapHasVJPAt
#print axioms Proofs.batchMapAux_eq_batchMapHasVJPAt
#print axioms Proofs.batchMap_eq_batchMapHasVJPAt
#print axioms Proofs.batchMap_comp
#print axioms Proofs.HasVJPAt.backward_unique_of_eq

-- RESNET-34 AT TRUE BATCH BN — T1's VJP half (ResNet34FullBVJP.lean)
#print axioms Proofs.r34IdBHasVJPAt
#print axioms Proofs.r34DownBHasVJPAt
#print axioms Proofs.r34StemBHasVJPAt
#print axioms Proofs.stemReluPoolLayer
#print axioms Proofs.maxPool3s2Flat_relu_eventuallyEq
#print axioms Proofs.maxPool3s2FlatBackB_eq_reindex
#print axioms Proofs.gatherReluHasVJPAt
#print axioms Proofs.ResNet34TieB.r34StemPool_param_germ
#print axioms Proofs.ResNet34TieB.r34StemGCg_hasGradAt
#print axioms Proofs.stemPoolRelu_param_eventuallyEq_select
#print axioms Proofs.maxPool3s2LocalReindexB_isSelect
#print axioms Proofs.ResNet34TieB.r34StemPool_param_germ_select
#print axioms Proofs.ResNet34TieB.r34StemGCg_hasGradAt_at
#print axioms Proofs.ResNet34TieB.r34_stem_lossTiedB_select
#print axioms Proofs.ResNet34TieB.r34Stem_select_grads_eq
#print axioms Proofs.ResNet34TieB.r34StemLossTiedB.select
#print axioms Proofs.ResNet34TieB.r34_net_lossGrad_stemSelect
#print axioms Proofs.ResNet50TieB.r50_net_lossGrad_stemSelect
#print axioms Proofs.ResNet34TieB.r34LossSmoothAtB_of_smoothAtB
#print axioms Proofs.ResNet50TieB.r50LossSmoothAtB_of_smoothAtB
#print axioms Proofs.BatchSeal.bnBatchLA_bcell_eq_of_eq
#print axioms Proofs.r34StemLayer
#print axioms Proofs.r34HeadLayer
#print axioms Proofs.r34HeadBHasVJP
#print axioms Proofs.r34NetLayer
#print axioms Proofs.r34NetLayer_fwd_apply
#print axioms Proofs.r34SmoothAtB_ok
#print axioms Proofs.resnet34ForwardBFullHasVJPAt
#print axioms Proofs.resnet34ForwardBFull_eq_chain
#print axioms Proofs.resnet34ForwardBFullHasVJPAt_correct
#print axioms Proofs.resnet34ForwardBFull_differentiableAt

-- RESNET-34 AT TRUE BATCH BN — T3's fold, UN-FUSED (Foundation/GradNodesB.lean)
#print axioms Proofs.GradNodeB.convWGradB_den
#print axioms Proofs.GradNodeB.convBGradB_den
#print axioms Proofs.GradNodeB.convStridedWGradB_den
#print axioms Proofs.GradNodeB.convStridedBGradB_den
#print axioms Proofs.GradNodeB.bnGammaGradB_den
#print axioms Proofs.GradNodeB.bnBetaGradB_den
#print axioms Proofs.GradNodeB.denseWGradB_den
#print axioms Proofs.GradNodeB.denseBGradB_den

-- EfficientNet-B0
#print axioms Proofs.GradNodeB.denseBGradB_den
#print axioms Proofs.GradNodeB.convStridedXlaWGradB_den
#print axioms Proofs.GradNodeB.depthwiseWGradB_den
#print axioms Proofs.GradNodeB.depthwiseStridedWGradB_den

-- ConvNeXt-T
#print axioms Proofs.CnxFoldG.layerScaleChGammaGrad_den
#print axioms Proofs.CnxFoldG.chanLnGammaGrad_den
#print axioms Proofs.CnxFoldG.chanLnBetaGrad_den

-- ViT-Tiny — vit_adam_train_step and vitin_adamdp128x4wxclipdrop
#print axioms Proofs.ViTFoldG.posEmbedGrad_den
#print axioms Proofs.ViTFoldG.clsGrad_den

-- ViT-Tiny
#print axioms Proofs.GradNodeB.veclnGammaGradB_den
#print axioms Proofs.GradNodeB.rowDenseBiasGradB_den_lnbeta
#print axioms Proofs.ViTFoldGB.rowDenseWeightGradB_den
#print axioms Proofs.ViTFoldGB.rowDenseBiasGradB_den
#print axioms Proofs.ViTFoldGB.patchEmbedWeightGradB_den
#print axioms Proofs.ViTFoldGB.patchEmbedBiasGradB_den
#print axioms Proofs.ViTFoldGB.posEmbedGradB_den
#print axioms Proofs.ViTFoldGB.clsGrad_denB
#print axioms Proofs.GradNodeB.headWGradB_den
#print axioms Proofs.GradNodeB.headBGradB_den

-- ConvNeXt-T
#print axioms Proofs.CnxFoldGB.layerScaleChGammaGradB_den
#print axioms Proofs.GradNodeB.psWGradB_den
#print axioms Proofs.CnxFoldGB.chanLnGammaGradB_den
#print axioms Proofs.CnxFoldGB.chanLnBetaGradB_den
-- The bf16 gradient nodes, folded ONCE for every net (Bf16GradNodes.lean)
#print axioms Proofs.Bf16Fold.convWGradBBf16_den
#print axioms Proofs.Bf16Fold.convStridedWGradBBf16_den
#print axioms Proofs.Bf16Fold.convStridedXlaWGradBBf16_den
#print axioms Proofs.Bf16Fold.convStride4WGradBBf16_den
#print axioms Proofs.Bf16Fold.depthwiseWGradBBf16_den
#print axioms Proofs.Bf16Fold.depthwiseStridedWGradBBf16_den
#print axioms Proofs.Bf16Fold.depthwiseStridedXlaWGradBBf16_den
#print axioms Proofs.Bf16Fold.rowDenseWGradBBf16_den
#print axioms Proofs.Bf16Fold.patchEmbedWGradBBf16_den

-- MobileNetV2 at 17 blocks
#print axioms Proofs.GradNodeB.convStridedXlaBGradB_den
#print axioms Proofs.GradNodeB.depthwiseBGradB_den
#print axioms Proofs.GradNodeB.depthwiseStridedXlaWGradB_den
#print axioms Proofs.GradNodeB.depthwiseStridedXlaBGradB_den

-- THE LABEL-SMOOTHED LOSS COTANGENT, AT A GENERAL TARGET (SmoothedLossCot.lean)
#print axioms Proofs.softCE_oneHot
#print axioms Proofs.softCE_grad
#print axioms Proofs.smoothTarget_sum
#print axioms Proofs.smoothedCE_grad
#print axioms Proofs.smoothedLossCotGraph_den
#print axioms Proofs.smoothedLossCotGraph_row

-- RESNET-34'S T3 TIE AT BATCH BN, UN-FUSED (ResNet34StepTieB.lean)
#print axioms Proofs.BackLinks.bnInB_eq_bnBackB
#print axioms Proofs.BackLinks.bnInB_eq_den_bnBatchBack
#print axioms Proofs.ResNet34TieB.r34IdCotIn_eq_vjp
#print axioms Proofs.ResNet34TieB.r34DownCotIn_eq_vjp
#print axioms Proofs.ResNet34TieB.r34_idblock_tiedB
#print axioms Proofs.ResNet34TieB.r34_downblock_tiedB
#print axioms Proofs.ResNet34TieB.r34_stem_tiedB
#print axioms Proofs.ResNet34TieB.r34_head_tiedB
#print axioms Proofs.ResNet34TieB.r34_net_tiedB
#print axioms Proofs.ResNet34TieB.r34_lossCot_is_smoothedCE_grad

-- MOBILENETV2 AT TRUE BATCH BN — T1-forward and T2 (MobileNetV2FullB.lean)
#print axioms Proofs.mobilenetv2ForwardBFull
#print axioms Proofs.StableHLO.mnv2StemGraphB_faithful
#print axioms Proofs.StableHLO.mnv2NoExpGraphB_faithful
#print axioms Proofs.StableHLO.mnv2ExpOnlyGraphB_faithful
#print axioms Proofs.StableHLO.mnv2ResidGraphB_faithful
#print axioms Proofs.StableHLO.mnv2StridedGraphB_faithful
#print axioms Proofs.StableHLO.mnv2HeadGraphB_faithful
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphBFull_faithful

-- MOBILENETV2 AT TRUE BATCH BN — T1's VJP half (MobileNetV2FullBVJP.lean)
#print axioms Proofs.mnv2StemBHasVJPAt
#print axioms Proofs.mnv2NoExpBHasVJPAt
#print axioms Proofs.mnv2ExpOnlyBHasVJPAt
#print axioms Proofs.mnv2ResidBHasVJPAt
#print axioms Proofs.mnv2StridedBHasVJPAt
#print axioms Proofs.mobilenetv2ForwardBFullHasVJPAt
#print axioms Proofs.mobilenetv2ForwardBFull_eq_chain
#print axioms Proofs.mobilenetv2ForwardBFullHasVJPAt_correct
#print axioms Proofs.mobilenetv2ForwardBFull_differentiableAt

-- MOBILENETV2'S T3 TIE AT BATCH BN, UN-FUSED (MobileNetV2StepTieB.lean)
#print axioms Proofs.MobileNetV2TieB.mnv2NoExpBackGraph_faithful
#print axioms Proofs.MobileNetV2TieB.mnv2NoExpCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2ExpOnlyCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2ResidCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2StridedCotIn_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2HeadCotBlk_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2StemCotC_eq_vjp
#print axioms Proofs.MobileNetV2TieB.mnv2_stem_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_noexp_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_stride1_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_stride2_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_head_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_net_tiedB
#print axioms Proofs.MobileNetV2TieB.mnv2_lossCot_is_smoothedCE_grad

-- CAPSTONE RE-POINTING, EFFICIENTNET-B0 (EfficientNetStepTieG.lean)
#print axioms Proofs.EnetTieG.convBBetaTiedB_holds
#print axioms Proofs.GradNodeB.convWTiedB_holds
#print axioms Proofs.GradNodeB.convBTiedB_holds
#print axioms Proofs.GradNodeB.convStridedWTiedB_holds
#print axioms Proofs.GradNodeB.convStridedBTiedB_holds
#print axioms Proofs.GradNodeB.convStridedXlaWTiedB_holds
#print axioms Proofs.GradNodeB.depthwiseWTiedB_holds
#print axioms Proofs.GradNodeB.depthwiseBTiedB_holds
#print axioms Proofs.GradNodeB.depthwiseStridedWTiedB_holds
#print axioms Proofs.GradNodeB.denseWTiedB_holds
#print axioms Proofs.GradNodeB.denseBTiedB_holds
#print axioms Proofs.ViTFoldGB.rowDenseWTiedB_holds
#print axioms Proofs.ViTFoldGB.rowDenseBTiedB_holds
#print axioms Proofs.GradNodeB.vecLNGammaTiedB_holds
#print axioms Proofs.GradNodeB.vecLNBetaTiedB_holds
#print axioms Proofs.CnxFoldGB.chanLNGammaTiedB_holds
#print axioms Proofs.CnxFoldGB.chanLNBetaTiedB_holds
#print axioms Proofs.EnetFold.convWSgdTiedB_holds
#print axioms Proofs.EnetFold.depthwiseWSgdTiedB_holds
#print axioms Proofs.EnetFold.denseWSgdTiedB_holds
#print axioms Proofs.EnetFold.denseBSgdTiedB_holds
#print axioms Proofs.ViTFold.rowDenseWSgdTied_holds
#print axioms Proofs.ViTFold.rowDenseBSgdTied_holds
#print axioms Proofs.SgdNode.vecLNGammaSgdTied_holds
#print axioms Proofs.SgdNode.vecLNBetaSgdTied_holds
#print axioms Proofs.CnxFold.chanLNGammaSgdTied_holds
#print axioms Proofs.CnxFold.chanLNBetaSgdTied_holds
#print axioms Proofs.convWSgdTied_holds
#print axioms Proofs.convBSgdTied_holds
#print axioms Proofs.denseWSgdTied_holds
#print axioms Proofs.denseBSgdTied_holds
#print axioms Proofs.EnetTie.convBBetaSgdTied_holds
#print axioms Proofs.EnetTieG.enet_exp_tiedG
#print axioms Proofs.EnetTieG.enet_strided_tiedG
#print axioms Proofs.EnetTieG.enet_noexp_tiedG
#print axioms Proofs.EnetTieG.enet_stem_tiedG
#print axioms Proofs.EnetTieG.enet_head_tiedG
#print axioms Proofs.EnetTieG.efficientnet_net_tiedG

-- CAPSTONE RE-POINTING, CONVNEXT-T (ConvNeXtStepTieGB.lean)
#print axioms Proofs.smoothedLossCotGraphDiv_den
#print axioms Proofs.smoothedLossCotGraphDiv_row
#print axioms Proofs.CnxTieGB.cnx_block_ch_tiedGB
#print axioms Proofs.CnxTieGB.cnx_down_ch_tiedGB
#print axioms Proofs.CnxTieGB.cnx_stem_ch_tiedGB
#print axioms Proofs.CnxTieGB.cnx_head_ch_tiedGB
#print axioms Proofs.CnxTieGB.cnxBlockCotInChAt_eq_vjp
#print axioms Proofs.CnxTieGB.cnxDownCotInChAt_eq_vjp
#print axioms Proofs.CnxTieGB.cnxHeadHasVJP
#print axioms Proofs.CnxTieGB.cnxHeadDyXheadChN_eq_vjp
#print axioms Proofs.CnxTieGB.cnxBlockCotInB_eq_vjp
#print axioms Proofs.CnxTieGB.cnxDownCotInB_eq_vjp
#print axioms Proofs.CnxTieGB.cnxHeadDyB_eq_vjp
#print axioms Proofs.CnxTieGB.cnx_net_tiedGB

-- CAPSTONE RE-POINTING, ViT-TINY (ViTStepTieGB.lean)
#print axioms Proofs.ViTTieGB.vit_block_tiedGB
#print axioms Proofs.ViTTieGB.vit_finalLN_tiedGB
#print axioms Proofs.ViTTieGB.vit_head_tiedGB
#print axioms Proofs.ViTTieGB.vit_embed_tiedGB
-- per-example cotangent ties (ViTStepTie.lean)
#print axioms Proofs.rowDenseBackFlat_eq_perRowFlat
#print axioms Proofs.vitCotDQmh_eq_core
#print axioms Proofs.vitCotDKmh_eq_core
#print axioms Proofs.vitCotDVmh_eq_core
#print axioms Proofs.vitCotLn2_eq_perRowFlatPR
#print axioms Proofs.vitCotXin_eq_blockBack
#print axioms Proofs.vitBlockCotInAtMHV_eq_vjp
#print axioms Proofs.vitHeadHasVJP
#print axioms Proofs.vitCotTowerOutV_eq_vjp
#print axioms Proofs.ViTTieGB.vitBlockCotInB_eq_vjp
#print axioms Proofs.ViTTieGB.vitCotTowerOutB_eq_vjp
#print axioms Proofs.ViTTieGB.vit_net_tiedGB

-- DATA PARALLELISM -- WHAT FUNCTION A *dp* RUN MINIMISED (DataParallel/Basic.lean)
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

-- THE COLLECTIVE AS AN AST NODE (StableHLO.allReduceMeanF, DataParallel/Node.lean)
#print axioms Proofs.StableHLO.den_allReduceMeanF
#print axioms Proofs.den_allReduceMeanF_eq_dpMean
#print axioms Proofs.skel_allReduceMeanF_of_spmd
#print axioms Proofs.den_allReduceMeanF_eq_lossGrad_meanLoss
#print axioms Proofs.den_allReduceMeanF_convWeightGradB
#print axioms Proofs.adamW_at_allReduceMeanF

-- SYNCHRONISED BATCHNORM -- THE DP STEP IS THE GLOBAL-BATCH STEP
-- (DataParallel/Basic.lean §sync + DataParallel/Sync.lean)
-- At the ℝ level: the positive twin of dpMeanGrad_ne_globalBatchGrad
#print axioms Proofs.dpMean_shardSum
#print axioms Proofs.dpSyncGrad_eq_globalBatchGrad
#print axioms Proofs.dpSyncGrad_eq_globalBatchGrad_contiguous
-- Sharding commutes with every per-example lift (definitional)
#print axioms Proofs.batchSlice_batchShard
#print axioms Proofs.batchShard_batchMap
#print axioms Proofs.batchShard_batchMapAux
-- the statistics subgraph denotes the global [μ ‖ σ²]; shard identities on the graph at ANY R --
-- the BN case of the per-net chain induction
#print axioms Proofs.den_syncStats_left
#print axioms Proofs.den_syncStats_right
#print axioms Proofs.mulR_nhw_ne_zero
#print axioms Proofs.den_bnSyncF_allReduce
#print axioms Proofs.den_bnSyncBack_allReduce
#print axioms Proofs.den_allReduceMeanF_bnSyncGammaGradB
-- the handed-back statistics under DP are the GLOBAL batch's own (bnBatchMeanB/VarB at N := R·N)
#print axioms Proofs.den_bnStatsMeanB_allReduce
#print axioms Proofs.den_bnStatsVarB_allReduce
-- At the node: the parameter collective is 1/R of the batch-R·N gradient node
#print axioms Proofs.den_allReduceMeanF_convWeightGradB_shard
#print axioms Proofs.den_allReduceMeanF_bnBetaGradB_shard
-- the divisor step (divConstB N on a replica vs divConstB (R·N) on one device)
#print axioms Proofs.HasVJP.backward_smul

-- SYNC-BN AT bf16: EVERY NODE SHARDS EXACTLY BUT THE CONV WEIGHT GRADIENT
-- (DataParallel/SyncBf16.lean)
-- forward convs and input-VJPs: replica r's node is shard r of the same node at batch R·N
#print axioms Proofs.den_batchOp_shard_node
#print axioms Proofs.den_convBf16_shard
#print axioms Proofs.den_convStridedBf16_shard
#print axioms Proofs.den_convBackBatchedBf16_shard
#print axioms Proofs.den_convStridedBackBatchedBf16_shard
-- weight gradients: each replica rounds its own partial sum; the batch-R·N node rounds their sum once
#print axioms Proofs.den_convWeightGradBBf16_eq_rnd
#print axioms Proofs.den_convStridedWeightGradBBf16_eq_rnd
#print axioms Proofs.den_allReduceMeanF_convWeightGradBBf16_shard
#print axioms Proofs.den_convWeightGradBBf16_global_split
#print axioms Proofs.den_allReduceMeanF_convWeightGradBBf16_sub_global
#print axioms Proofs.den_allReduceMeanF_convWeightGradBBf16_shard_id
#print axioms Proofs.den_allReduceMeanF_convStridedWeightGradBBf16_shard
#print axioms Proofs.den_convStridedWeightGradBBf16_global_split
#print axioms Proofs.den_allReduceMeanF_convStridedWeightGradBBf16_sub_global
-- the divisor step through rnd: rndP commutes with powers of two (R = 4 on every ImageNet run)
#print axioms Proofs.int_log_zpow_mul
#print axioms Proofs.int_log_abs_zpow_mul
#print axioms Proofs.rndP_zpow_mul
#print axioms Proofs.convBackBatchedBf16_smul
#print axioms Proofs.convStridedBackBatchedBf16_smul
#print axioms Proofs.convWeightGradBBf16_smul
#print axioms Proofs.convStridedWeightGradBBf16_smul

-- SYNC-BN AT RESNET-34: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (ResNet34SyncB.lean + ResNet34SyncStepTieB.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of resnet34ForwardBFull (R*N)
#print axioms Proofs.StableHLO.den_castIdx
#print axioms Proofs.StableHLO.batchShard_castIdx
#print axioms Proofs.StableHLO.den_bnSyncSiteLA
#print axioms Proofs.StableHLO.r34IdGraphSync_shard
#print axioms Proofs.StableHLO.r34DownGraphSync_shard
#print axioms Proofs.StableHLO.r34StemGraphSync_shard
#print axioms Proofs.StableHLO.resnet34FwdGraphSyncFull_shard
-- T3 twin: the single-device chain is homogeneous in its cotangent
#print axioms Proofs.SyncKit.bnGradInput_smul
#print axioms Proofs.SyncKit.bnInB_smul
#print axioms Proofs.SyncKit.maxPool3s2BackFlat_smul
#print axioms Proofs.ResNet34SyncTieB.r34IdCotIn_smul
#print axioms Proofs.ResNet34SyncTieB.r34DownCotIn_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#print axioms Proofs.SyncKit.bnSyncInB_shard
#print axioms Proofs.ResNet34SyncTieB.r34IdSyncCotIn_shard
#print axioms Proofs.ResNet34SyncTieB.r34DownSyncCotIn_shard
#print axioms Proofs.ResNet34SyncTieB.r34StemSyncCotC_shard
#print axioms Proofs.ResNet34SyncTieB.r34HeadCotBlk_shard
-- ...the strided-conv and dense collectives, and the divisor
#print axioms Proofs.SyncKit.den_allReduceMeanF_convStridedWeightGradB_shard
#print axioms Proofs.SyncKit.den_allReduceMeanF_denseWeightGradB_shard
#print axioms Proofs.SyncKit.den_allReduceMeanF_denseBiasGradB_shard
#print axioms Proofs.SyncKit.replicaLossCot_eq
-- the capstone: every all-reduced parameter gradient IS the single-device node at R·N
#print axioms Proofs.ResNet34SyncTieB.r34_net_syncTiedB
#print axioms Proofs.ResNet34SyncTieB.r34_net_syncTiedB_smoothedCE

-- SYNC-BN: THE MBCONV PIECES MOBILENETV2 AND EFFICIENTNET-B0 SHARE
-- (MBConvSyncTieB.lean)
#print axioms Proofs.HasVJP3.backward_smul
#print axioms Proofs.SyncKit.depthwiseWeightGradB_smul
#print axioms Proofs.SyncKit.depthwiseStridedWeightGradB_smul
#print axioms Proofs.SyncKit.convStridedXlaWeightGradB_smul
#print axioms Proofs.SyncKit.rowDenseBackFlat_smul
#print axioms Proofs.SyncKit.rowDenseBackFlat_shard
#print axioms Proofs.SyncKit.den_allReduceMeanF_depthwiseWeightGradB_shard
#print axioms Proofs.SyncKit.den_allReduceMeanF_depthwiseStridedWeightGradB_shard
#print axioms Proofs.SyncKit.den_allReduceMeanF_convStridedXlaWeightGradB_shard
#print axioms Proofs.SyncKit.depthwiseWSync_of_scaled
#print axioms Proofs.SyncKit.depthwiseStridedWSync_of_scaled
#print axioms Proofs.SyncKit.convStridedXlaWSync_of_scaled

-- SYNC-BN AT MOBILENETV2: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (MobileNetV2SyncB.lean + MobileNetV2SyncStepTieB.lean)
#print axioms Proofs.StableHLO.den_relu6_shard
#print axioms Proofs.StableHLO.mnv2StemGraphSync_shard
#print axioms Proofs.StableHLO.mnv2NoExpGraphSync_shard
#print axioms Proofs.StableHLO.mnv2ExpOnlyGraphSync_shard
#print axioms Proofs.StableHLO.mnv2ResidGraphSync_shard
#print axioms Proofs.StableHLO.mnv2StridedGraphSync_shard
#print axioms Proofs.StableHLO.mnv2HeadGraphSync_shard
#print axioms Proofs.StableHLO.mobilenetv2FwdGraphSyncFull_shard
#print axioms Proofs.SyncKit.relu6MaskB_smul
#print axioms Proofs.SyncKit.depthwiseStridedXlaWeightGradB_smul
#print axioms Proofs.MobileNetV2SyncTieB.mnv2NoExpCotIn_smul
#print axioms Proofs.MobileNetV2SyncTieB.mnv2ResidCotIn_smul
#print axioms Proofs.MobileNetV2SyncTieB.mnv2StridedCotIn_smul
#print axioms Proofs.MobileNetV2SyncTieB.mnv2HeadCotBlk_smul
#print axioms Proofs.MobileNetV2SyncTieB.mnv2NoExpSyncCotIn_shard
#print axioms Proofs.MobileNetV2SyncTieB.mnv2SyncCotInBody_shard
#print axioms Proofs.MobileNetV2SyncTieB.mnv2ResidSyncCotIn_shard
#print axioms Proofs.MobileNetV2SyncTieB.mnv2StridedSyncCotIn_shard
#print axioms Proofs.MobileNetV2SyncTieB.mnv2StemSyncCotC_shard
#print axioms Proofs.MobileNetV2SyncTieB.mnv2HeadSyncCotBlk_shard
#print axioms Proofs.MobileNetV2SyncTieB.den_allReduceMeanF_depthwiseStridedXlaWeightGradB_shard
#print axioms Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB
#print axioms Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB_smoothedCE

-- SYNC-BN AT EFFICIENTNET-B0: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (EfficientNetSyncB.lean + EfficientNetSyncStepTieG.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of efficientnetForwardBFull (R*N)
#print axioms Proofs.StableHLO.den_swishF_shard
#print axioms Proofs.StableHLO.den_addV_shard
#print axioms Proofs.StableHLO.mbNoExpGraphSync_shard
#print axioms Proofs.StableHLO.mbBodyGraphSync_shard
#print axioms Proofs.StableHLO.mbResidGraphSync_shard
#print axioms Proofs.StableHLO.mbStridedGraphSync_shard
#print axioms Proofs.StableHLO.stemGraphSync_shard
#print axioms Proofs.StableHLO.headGraphSync_shard
#print axioms Proofs.StableHLO.efficientnetFwdGraphSyncFull_shard
-- T3 twin: each block's input cotangent IS its certified VJP (T3 threads `.backward`)
#print axioms Proofs.EnetSyncTieG.xCotIn_eq_vjp
#print axioms Proofs.EnetSyncTieG.rCotIn_eq_vjp
#print axioms Proofs.EnetSyncTieG.sCotIn_eq_vjp
#print axioms Proofs.EnetSyncTieG.nCotIn_eq_vjp
#print axioms Proofs.EnetSyncTieG.hdCotIn_eq_vjp
-- ...the single-device chain is homogeneous in its cotangent
#print axioms Proofs.BackLinks.gateCotB_smul
#print axioms Proofs.EnetSyncTieG.tCotDc_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#print axioms Proofs.SyncKit.gateCotB_shard
#print axioms Proofs.SyncKit.seInB_shard
#print axioms Proofs.SyncKit.bnSyncInB_shard_bnBackB
#print axioms Proofs.EnetSyncTieG.tsCotDc_shard
#print axioms Proofs.EnetSyncTieG.xsCotIn_scaled
#print axioms Proofs.EnetSyncTieG.rsCotIn_scaled
#print axioms Proofs.EnetSyncTieG.ssCotIn_scaled
#print axioms Proofs.EnetSyncTieG.nsCotIn_scaled
#print axioms Proofs.EnetSyncTieG.hdsCotIn_scaled
-- the per-block ties and the capstone: every all-reduced parameter gradient IS the node at R·N
#print axioms Proofs.EnetSyncTieG.tail_syncTiedG
#print axioms Proofs.EnetSyncTieG.exp_syncTiedG
#print axioms Proofs.EnetSyncTieG.strided_syncTiedG
#print axioms Proofs.EnetSyncTieG.noExp_syncTiedG
#print axioms Proofs.EnetSyncTieG.stem_syncTiedG
#print axioms Proofs.EnetSyncTieG.head_syncTiedG
#print axioms Proofs.EnetSyncTieG.efficientnet_net_syncTiedG
#print axioms Proofs.EnetSyncTieG.efficientnet_net_syncTiedG_smoothedCE

-- SYNC-BN AT RESNET-50: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (ResNet50SyncB.lean + ResNet50SyncStepTieB.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of resnet50ForwardBFull (R*N) q
#print axioms Proofs.StableHLO.den_addVB_shard_comm
#print axioms Proofs.StableHLO.r50IdGraphSync_shard
#print axioms Proofs.StableHLO.r50ProjGraphSync_shard
#print axioms Proofs.StableHLO.r50DownGraphSync_shard
#print axioms Proofs.StableHLO.resnet50FwdGraphSyncFull_shard
-- T3 twin: the single-device chain is homogeneous in its cotangent
#print axioms Proofs.ResNet50SyncTieB.r50IdCotIn_smul
#print axioms Proofs.ResNet50SyncTieB.r50ProjCotIn_smul
#print axioms Proofs.ResNet50SyncTieB.r50DownCotIn_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#print axioms Proofs.ResNet50SyncTieB.r50IdSyncCotIn_shard
#print axioms Proofs.ResNet50SyncTieB.r50ProjSyncCotIn_shard
#print axioms Proofs.ResNet50SyncTieB.r50DownSyncCotIn_shard
#print axioms Proofs.ResNet50SyncTieB.r50IdSyncCotIn_scaled
#print axioms Proofs.ResNet50SyncTieB.r50ProjSyncCotIn_scaled
#print axioms Proofs.ResNet50SyncTieB.r50DownSyncCotIn_scaled
-- the per-block ties and the capstone (the cotangent a binder), then both losses discharged
#print axioms Proofs.ResNet50SyncTieB.r50_idblock_syncTiedB
#print axioms Proofs.ResNet50SyncTieB.r50_projblock_syncTiedB
#print axioms Proofs.ResNet50SyncTieB.r50_downblock_syncTiedB
#print axioms Proofs.ResNet50SyncTieB.r50_net_syncTiedB
#print axioms Proofs.ResNet50SyncTieB.replicaBceLossCot_eq
#print axioms Proofs.ResNet50SyncTieB.r50_net_syncTiedB_smoothedCE
#print axioms Proofs.ResNet50SyncTieB.r50_net_syncTiedB_bce

-- SYNC-BN AT MOBILENETV4: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (MobileNetV4SyncB.lean + MobileNetV4SyncStepTieB.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of mobilenetv4ForwardBFull (R*N)
#print axioms Proofs.StableHLO.den_castIdx_shard
#print axioms Proofs.StableHLO.mnv4StemGraphSync_shard
#print axioms Proofs.StableHLO.mnv4FusedGraphSync_shard
#print axioms Proofs.StableHLO.mnv4ExtraDWBodyGraphSync_shard
#print axioms Proofs.StableHLO.mnv4ConvNeXtBodyGraphSync_shard
#print axioms Proofs.StableHLO.mnv4FfnBodyGraphSync_shard
#print axioms Proofs.StableHLO.mnv4StridedGraphSync_shard
#print axioms Proofs.StableHLO.mnv4SkipGraphSync_shard
#print axioms Proofs.StableHLO.mnv4HeadGraphSync_shard
#print axioms Proofs.StableHLO.mnv4FwdGraphSyncFull_shard
-- T3 twin: the single-device chain is homogeneous in its cotangent
#print axioms Proofs.MobileNetV4SyncTieB.mnv4BodyCotIn_smul
#print axioms Proofs.MobileNetV4SyncTieB.mnv4SBodyCotIn_smul
#print axioms Proofs.MobileNetV4SyncTieB.mnv4StemCotC_smul
#print axioms Proofs.MobileNetV4SyncTieB.mnv4FusedCotIn_smul
#print axioms Proofs.MobileNetV4SyncTieB.mnv4HeadCotIn_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#print axioms Proofs.MobileNetV4SyncTieB.mnv4BodySyncCotIn_shard
#print axioms Proofs.MobileNetV4SyncTieB.mnv4SBodySyncCotIn_shard
#print axioms Proofs.MobileNetV4SyncTieB.mnv4StemSyncCotC_shard
#print axioms Proofs.MobileNetV4SyncTieB.mnv4FusedSyncCotIn_shard
#print axioms Proofs.MobileNetV4SyncTieB.mnv4HeadSyncCotIn_shard
#print axioms Proofs.MobileNetV4SyncTieB.mnv4BodySyncCotIn_scaled
#print axioms Proofs.MobileNetV4SyncTieB.mnv4SkipSyncCotIn_scaled
#print axioms Proofs.MobileNetV4SyncTieB.mnv4SBodySyncCotIn_scaled
#print axioms Proofs.MobileNetV4SyncTieB.mnv4FusedSyncCotIn_scaled
#print axioms Proofs.MobileNetV4SyncTieB.mnv4HeadSyncCotIn_scaled
-- the per-block ties and the capstone (the cotangent a binder), then smoothed CE discharged
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_extradw_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_convnext_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_ffn_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_strided_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_stem_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_fused_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_head_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB
#print axioms Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB_smoothedCE

-- RESNET-50's TWO PREREQUISITES: THE LAMB TRIPLE AND BCE'S COTANGENT
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

-- RESNET-50's T1 AND T2 AT BATCH BATCHNORM (ResNet50FullB{,VJP}.lean)
#print axioms Proofs.resnet50ForwardBFull
#print axioms Proofs.r50IdBHasVJPAt
#print axioms Proofs.r50ProjBHasVJPAt
#print axioms Proofs.r50DownBHasVJPAt
#print axioms Proofs.r50NetLayer
#print axioms Proofs.r50NetLayer_fwd_apply
#print axioms Proofs.resnet50ForwardBFullHasVJPAt
#print axioms Proofs.resnet50ForwardBFull_eq_chain
#print axioms Proofs.resnet50ForwardBFullHasVJPAt_correct
#print axioms Proofs.resnet50ForwardBFull_differentiableAt

-- T2, the typed forward graph (ResNet-50)
#print axioms Proofs.StableHLO.r50IdGraphB_faithful
#print axioms Proofs.StableHLO.r50ProjGraphB_faithful
#print axioms Proofs.StableHLO.r50DownGraphB_faithful
#print axioms Proofs.StableHLO.r50StemGraphB_faithful
#print axioms Proofs.StableHLO.resnet50FwdGraphBFull_faithful

-- RESNET-50's T3 -- THE FOLD AND THE TIE (ResNet50StepTieB.lean)
#print axioms Proofs.ResNet50TieB.r50IdCotIn_eq_vjp
#print axioms Proofs.ResNet50TieB.r50ProjCotIn_eq_vjp
#print axioms Proofs.ResNet50TieB.r50DownCotIn_eq_vjp
#print axioms Proofs.ResNet50TieB.r50_idblock_tiedB
#print axioms Proofs.ResNet50TieB.r50_projblock_tiedB
#print axioms Proofs.ResNet50TieB.r50_downblock_tiedB
#print axioms Proofs.ResNet50TieB.r50_net_tiedB
#print axioms Proofs.ResNet50TieB.r50_lossCot_is_smoothedCE_grad
#print axioms Proofs.ResNet50TieB.r50_lossCot_is_bce_grad
