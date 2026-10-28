import tests.AxiomAudit
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
import LeanMlir.Proofs.Foundation.DropSites
import LeanMlir.Proofs.Foundation.Batched.Indexed
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
import LeanMlir.Proofs.Foundation.Bf16Erasure
import LeanMlir.Proofs.Foundation.GradNodesBAt
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StepTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTieG
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtDropBlock
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB
import LeanMlir.Proofs.Nets.ViT.ViTDropBlock
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
import LeanMlir.Proofs.Nets.Small.LinearFoldB
import LeanMlir.Proofs.Nets.Small.MlpFoldB
import LeanMlir.Proofs.Nets.Small.CnnFoldB
import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Nets.Small.Cifar8StepTie
import LeanMlir.Proofs.Nets.Small.Cifar8BnStepTie
import LeanMlir.Proofs.Nets.Small.Cifar8StepTieG
import LeanMlir.Proofs.Nets.Small.Cifar8BnStepTieG
import LeanMlir.Proofs.Nets.Small.Cifar8BbnStepTieB
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

-- One `#audit_axioms` per certified theorem (tests/AxiomAudit.lean): it errors on any axiom
-- beyond propext / Classical.choice / Quot.sound, so this file elaborating IS the audit.

-- Foundation
#audit_axioms pdiv_id
#audit_axioms pdiv_const
#audit_axioms pdiv_reindex
#audit_axioms pdiv_add
#audit_axioms pdiv_mul
#audit_axioms pdiv_comp
#audit_axioms pdiv_finset_sum
#audit_axioms pdivMat_comp
#audit_axioms pdivMat_matmul_left_const
#audit_axioms pdivMat_matmul_right_const
#audit_axioms pdivMat_rowIndep
#audit_axioms pdivMat_colIndep
#audit_axioms pdivMat_scalarScale
#audit_axioms pdivMat_transpose

-- MLP
#audit_axioms pdiv_dense
#audit_axioms pdiv_dense_W
#audit_axioms pdiv_dense_b
#audit_axioms pdiv_relu
#audit_axioms denseWeightGrad_correct
#audit_axioms denseBiasGrad_correct
#audit_axioms reluHasVJP_correct
#audit_axioms mlpHasVJP_correct
-- The POINTWISE (HasVJPAt) variants. They are also in the comparator suite
-- (tests/comparator/config-arch.json), and they are the three the book singles out as the ones
-- whose `.correct` field is a real proof rather than `rfl`, i.e. exactly where the kink escape is
-- closed.
#audit_axioms reluHasVJPAt_correct
#audit_axioms mlpHasVJPAt_correct

-- Nonzero-Jacobian seal (JacobianSeal.lean)
#audit_axioms sum_smul_basisVec
#audit_axioms fderiv_eq_zero_of_pdiv_all_zero
#audit_axioms exists_pdiv_ne_of_fderiv_ne
-- The pointwise (HasVJPAt) seal variants — the kinked witnesses are HasVJPAt, not HasVJP.
#audit_axioms HasVJPAt.backward_ne_zero_of_pdiv_ne
#audit_axioms HasVJPAt.backward_nontrivial_of_fderiv_ne

-- CNN
#audit_axioms maxPool2HasVJP3_correct
#audit_axioms maxPool2HasVJPAt3_correct   -- pointwise variant; see the note above
#audit_axioms conv2dHasVJP3
#audit_axioms conv2dHasVJP3_correct

-- Depthwise
#audit_axioms depthwiseHasVJP3_correct

-- BatchNorm
#audit_axioms pdiv_bnAffine
#audit_axioms pdiv_bnCentered
#audit_axioms pdiv_bnIstdBroadcast
#audit_axioms pdiv_bnNormalize
#audit_axioms bn_input_grad_correct

-- Residual
#audit_axioms residualHasVJP_correct
#audit_axioms residualProjHasVJP_correct

-- SE
#audit_axioms seBlockHasVJP_correct

-- LayerNorm / GELU / Swish
#audit_axioms pdiv_gelu
#audit_axioms geluHasVJP_correct
-- either GELU (`GeluForm`): what the ViT and ConvNeXt chains are stated over
#audit_axioms GeluForm.pdiv_map
#audit_axioms GeluForm.hasVJP_correct
-- exact GELU x·Φ(x): the VJP, the erfc closed forms (no Lean consumer; keep pinned), Φ = the standard normal cdf
#audit_axioms pdiv_geluErf
#audit_axioms geluErfHasVJP_correct
#audit_axioms geluErfScalar_eq_erfc
#audit_axioms geluErfScalarDeriv_eq_erfc
#audit_axioms gaussPhi_eq_cdf
-- closed form the emitted swishBack text computes (no Lean consumer; keep pinned)
#audit_axioms swishScalarDeriv_eq
#audit_axioms layerNormHasVJP_correct

-- Attention (apex)
#audit_axioms pdiv_softmax
#audit_axioms softmaxCE_grad
#audit_axioms sdpaBackQ_correct
#audit_axioms sdpaBackK_correct
#audit_axioms sdpaBackV_correct
#audit_axioms mhsaHasVJPMat_correct
#audit_axioms transformerBlockHasVJPMat_correct
#audit_axioms transformerTowerHasVJPMat
#audit_axioms vitBodyHasVJPMat
#audit_axioms vitFullHasVJP
#audit_axioms vitFullHasVJP_correct

-- Codegen smooth-point bridge theorems (MLP.lean)
#audit_axioms relu_codegen_matches_canonical

-- Codegen smooth-point bridge theorems (CNN.lean / MaxPool2)
#audit_axioms pdiv3_maxPool2_smooth
#audit_axioms maxPool2_codegen_matches_canonical

-- HasVJPAt pointwise framework
#audit_axioms reluHasVJPAt
#audit_axioms mlpHasVJPAt
#audit_axioms mnistLinearHasVJP_correct
#audit_axioms maxPool2HasVJPAt3

-- Capstone: end-to-end ResNet-style CNN whole-network VJP + the global-average-pool VJP it depends on
#audit_axioms globalAvgPoolFlatHasVJP
#audit_axioms globalAvgPoolFlatHasVJP_correct
#audit_axioms cnnHasVJPAt
#audit_axioms cnnHasVJPAt_correct

-- Whole-network VJPs for the depthwise/SE/LN-based architectures
#audit_axioms relu6HasVJPAt
#audit_axioms mobilenetv2HasVJPAt_correct
#audit_axioms convnextHasVJPAt_correct
-- ConvNeXt promoted to an UNCONDITIONAL global VJP (all-smooth ops)
#audit_axioms convnextHasVJP
#audit_axioms convnextHasVJP_correct
#audit_axioms sigmoidHasVJP
-- closed form the emitted sigmoidBack text computes (no Lean consumer; keep pinned)
#audit_axioms sigmoidScalarDeriv_eq
#audit_axioms efficientnetHasVJPAt_correct
-- EfficientNet promoted to an UNCONDITIONAL global VJP (all-smooth ops)
#audit_axioms efficientnetHasVJP
#audit_axioms efficientnetHasVJP_correct

-- Chapter-4 MNIST 2D CNN (no BN)
#audit_axioms mnistCnnNoBnHasVJPAt_correct
-- the reusable MaxPool2Smooth discharge
#audit_axioms maxPool2Smooth_of_injective
#audit_axioms maxPool2Smooth_of_pairwise
-- Chapter-3 MLP: concrete whole-network instance, every ReLU smoothness hypothesis discharged
#audit_axioms MlpConcrete.mlpConcreteHasVJPAt
-- ResNet-style CNN *with* BN
#audit_axioms CnnConcrete.cnnConcreteHasVJPAt

-- Denoted StableHLO-subset IR
#audit_axioms IR.dense_back_bridge
#audit_axioms IR.relu_back_bridge
-- The emitted transposed-convolution graph denotes the proven conv input-VJP
#audit_axioms IR.conv_back_bridge
-- The GENERAL conv-adjoint reindex (all dims, odd kernels)
#audit_axioms IR.convBackDenote_eq_input_grad_formula
-- The emitted select_and_scatter graph denotes the canonical maxpool backward;
-- its den routes to the first argmax, which off a tie is the argmax
#audit_axioms IR.maxpool_back_bridge
#audit_axioms IR.maxPool2Argmax_eq_iff_isArgmax
#audit_axioms IR.maxPoolBackDenote_eq_of_smooth
#audit_axioms windowArgmax_first
-- Smooth activations
#audit_axioms IR.gelu_back_bridge
#audit_axioms IR.geluErf_back_bridge
#audit_axioms IR.swish_back_bridge
#audit_axioms IR.sigmoid_back_bridge
-- BatchNorm: the emitted reduce+broadcast+elementwise graph denotes the proven 3-term backward
#audit_axioms IR.bn_back_bridge
#audit_axioms IR.layernorm_back_bridge
#audit_axioms IR.softmax_back_bridge
-- IR-level chain rule + an end-to-end composite bridge.
#audit_axioms IR.denote_subst
#audit_axioms IR.se_back_bridge
-- Tensor3 IR: conv/maxpool lifted into a composable backward graph + chain rule.
#audit_axioms IR.denote_subst3
#audit_axioms IR.maxpool3_node_bridge
#audit_axioms IR.conv3_node_bridge
#audit_axioms IR.conv_compose3
-- Flatten bridge: flattened Back3 graph denotes the proven flattened-layer Vec backward.
#audit_axioms IR.maxpool_flatten_bridge
#audit_axioms IR.conv_flatten_bridge
-- HasVJPAt smooth-point variants + a real dense→relu block via vjpCompAt.
#audit_axioms IR.relu_at_bridge
#audit_axioms IR.dense_at_bridge
-- Final assembly: the emitted whole-MLP backward graph denotes the proven whole-network VJP
#audit_axioms IR.mlp_whole_bridge
-- Parameter gradients (train-step pieces)
#audit_axioms IR.weight_grad_bridge
#audit_axioms IR.bias_grad_bridge
#audit_axioms IR.mlp_layer1_weight_grad_bridge
-- Forward IR
#audit_axioms IR.mlp_fwd_bridge
#audit_axioms IR.mlp_fwd_preact0
#audit_axioms IR.mlp_fwd_preact1
-- Loss cotangent
#audit_axioms IR.lossCot_bridge

-- Printer-faithfulness (Chapter 2)
#audit_axioms StableHLO.fwdGraph_faithful
#audit_axioms StableHLO.lossCotGraph_faithful
#audit_axioms StableHLO.lossCotGraph_isCEgrad
#audit_axioms StableHLO.wGrad_isWeightJacobian
#audit_axioms StableHLO.bGrad_isBiasJacobian
-- SGD update proven (not trusted) for plain SGD on the linear net.
#audit_axioms StableHLO.sgdW_isCertifiedGradStep
#audit_axioms StableHLO.sgdB_isCertifiedGradStep
-- The linear SGD step bundled to the certified closed-form softmax-CE gradient
#audit_axioms StableHLO.lossCot_eq_softmax_sub_onehot
#audit_axioms StableHLO.sgdW_descends_softmaxCE_grad
#audit_axioms StableHLO.sgdB_descends_softmaxCE_grad
-- Chain-rule fold: the SGD step is literally θ − lr·∂Loss/∂θ.
#audit_axioms Proofs.crossEntropy_differentiable
#audit_axioms StableHLO.denseWeightMap_differentiable
#audit_axioms StableHLO.lossWeightGrad_eq_sum
#audit_axioms StableHLO.sgdW_descends_loss_gradient
-- Rendering half
#audit_axioms StableHLO.linWeightDen_is_loss_descent
#audit_axioms StableHLO.linBiasDen_is_certified
-- MNIST-linear capstones (LinearFold.lean)
#audit_axioms LinFold.poc_fwd_is_render
-- Tail fold closed
#audit_axioms LinFold.poc_weightSgd_den_eq
#audit_axioms LinFold.poc_biasSgd_den_eq
#audit_axioms LinFold.poc_train_step_tail_certified
-- MNIST-linear at batch N (LinearFoldB.lean)
#audit_axioms LinFoldB.lossCotB_slice
#audit_axioms LinFoldB.lin_train_step_tiedB
#audit_axioms LinFoldB.lin_weight_meanlossB
-- mnist-MLP fully folded
#audit_axioms MlpFold.cot1_den
#audit_axioms MlpFold.cot0_den
#audit_axioms MlpFold.W2_den_certified
#audit_axioms MlpFold.W1_den_certified
#audit_axioms MlpFold.W0_den_certified
#audit_axioms MlpFold.b2_den_certified
#audit_axioms MlpFold.b1_den_certified
#audit_axioms MlpFold.b0_den_certified
-- mnist-mlp FULLY TIED
#audit_axioms MlpFold.mlpLossCot_den
#audit_axioms MlpFold.mlp_W2_tied_totalloss
#audit_axioms MlpFold.mlp_train_step_tied_certified
-- mnist-mlp at batch N (MlpFoldB.lean)
#audit_axioms MlpFoldB.cot1B_slice
#audit_axioms MlpFoldB.cot0B_slice
#audit_axioms MlpFoldB.mlp_train_step_tiedB
-- mnist-CNN fully folded
#audit_axioms CnnFold.cW1_den
#audit_axioms CnnFold.cb1_den
#audit_axioms CnnFold.cW2_den
#audit_axioms CnnFold.cb2_den
#audit_axioms CnnFold.dW5_den
-- mnist-cnn dense-head TIE
#audit_axioms CnnFold.cnnLossCot_den
#audit_axioms CnnFold.cnn_W5_tied_totalloss
-- mnist-cnn CONV fold
#audit_axioms CnnFold.cnn_train_step_tied_certified
-- mnist-cnn at batch N (CnnFoldB.lean): the chain pinned to the emitted subgraphs
#audit_axioms CnnFoldB.dhc2B_slice
#audit_axioms CnnFoldB.dhc1B_slice
#audit_axioms CnnFoldB.w5B_tied
#audit_axioms CnnFoldB.cnn_train_step_tiedB
-- ch5-CIFAR fully folded (no-BN, 2-scale)
#audit_axioms SgdNode.convW_den
#audit_axioms SgdNode.convB_den
#audit_axioms CifarFold.dW7_den
-- ch5-CIFAR TIE
#audit_axioms CifarFold.cifarLossCot_den
#audit_axioms CifarFold.cifar_W7_tied_totalloss
#audit_axioms CifarFold.cifar_train_step_tied_certified
-- ch5-CIFAR-BN fully folded
#audit_axioms SgdNode.bnGamma_den
#audit_axioms SgdNode.bnBeta_den
-- deeper 8-conv cifar8 fully folded
#audit_axioms SgdNode.denseW_den
#audit_axioms SgdNode.denseB_den
-- ch5-cifar8 TIE
#audit_axioms Cifar8Tie.cifar8_train_step_tied_certified
-- ch5-cifar8-bn TIE
#audit_axioms Cifar8BnTie.cifar8Bn_train_step_tied_certified
#audit_axioms Cifar8TieG.cifar8_train_step_tiedG
#audit_axioms Cifar8BnTieG.cifar8Bn_train_step_tiedG
#audit_axioms Cifar8BbnTieB.cifar8Bbn_train_step_tiedB
#audit_axioms GradNode.convWGrad_den
#audit_axioms GradNode.convBGrad_den
#audit_axioms GradNode.bnGammaGrad_den
#audit_axioms GradNode.bnBetaGrad_den
#audit_axioms GradNode.denseWGrad_den
#audit_axioms GradNode.denseBGrad_den
-- ch6-ResNet-34 fully folded (full [3,4,6,3], 146 params)
#audit_axioms SgdNode.convStridedW_den
#audit_axioms SgdNode.convStridedB_den
-- ch7-MobileNetV2 fold (depthwise half)
#audit_axioms SgdNode.depthwiseW_den
#audit_axioms SgdNode.depthwiseB_den
-- ch8-EfficientNet-B0 fold (den)
#audit_axioms EnetFold.convWB_den
#audit_axioms EnetFold.convStridedWB_den
#audit_axioms EnetFold.denseWB_den
#audit_axioms EnetFold.denseBB_den
#audit_axioms EnetFold.bnGammaB_den
#audit_axioms EnetFold.bnBetaB_den
#audit_axioms EnetFold.depthwiseWB_den
#audit_axioms EnetFold.depthwiseStridedXlaWB_den
-- ch8-EfficientNet-B0 TIE
#audit_axioms EnetTie.enet_exp_tied
#audit_axioms EnetTie.enet_strided_tied
#audit_axioms EnetTie.enet_noexp_tied
#audit_axioms EnetTie.enet_stem_tied
#audit_axioms EnetTie.enet_head_tied
#audit_axioms EnetTie.efficientnet_net_tied
-- The MLP per-layer parameter-gradient assembly
#audit_axioms IR.mlp_layer0_weight_grad_bridge
#audit_axioms IR.mlp_layer0_bias_grad_bridge
#audit_axioms IR.mlp_layer1_bias_grad_bridge
#audit_axioms IR.mlp_layer2_weight_grad_bridge
-- The CNN convolution parameter-gradient bridges (kernel grad = correlation).
#audit_axioms conv_weight_grad_bridge
#audit_axioms conv_bias_grad_bridge
-- CNN render close: the rendered conv weight/bias SGD outputs denote θ − lr·certified.
#audit_axioms conv_weight_sgd_certified
#audit_axioms conv_bias_sgd_certified
-- Chain: the composed cotangent subgraphs reduce to the explicit relu'⊙Wᵀ·… backprop
#audit_axioms IR.mlpCotOut1_denote
#audit_axioms IR.mlpCotOut0_denote
-- The conditional hidden-layer folds
#audit_axioms IR.mlp_hidden_total_loss_grad
#audit_axioms IR.mlp_input_total_loss_grad
-- Whole-net capstone: every weight layer's total-loss gradient at once (one statement).
#audit_axioms IR.mlp_whole_net_weight_grads
-- Printer-faithfulness, Chapter 3 (MLP)
#audit_axioms StableHLO.reluF_faithful
#audit_axioms StableHLO.selectPos_faithful
-- The saved-activation backwards, which CANNOT be descriptors
#audit_axioms StableHLO.den_lnRowBackB_per_example
-- The two statements the EMIT tie structurally cannot make (both forms render identical)
#audit_axioms StableHLO.den_batchOp_softmaxDiv_per_example
#audit_axioms StableHLO.den_rowDenseBiasGradB_at_one
-- ViT's version of that trap
#audit_axioms StableHLO.den_batchOp_clsSlice_per_example
-- ViT increment 2
#audit_axioms StableHLO.den_matmulFB_per_example
#audit_axioms StableHLO.den_softmaxRowBackB_per_example
-- The max-shifted softmax (2026-10-14): the shift is exact, row by row and per example
#audit_axioms StableHLO.softmax_sub_const
#audit_axioms StableHLO.rowSoftmaxShiftFlat_eq
#audit_axioms StableHLO.rowSoftmaxBackShiftFlat_eq
#audit_axioms StableHLO.den_softmaxRowBackShiftB_eq
-- The batch sum that is invisible at N = 1
#audit_axioms StableHLO.den_posEmbedGradB_at_one
-- CLASSIFIER DROPOUT
#audit_axioms StableHLO.dropoutB_faithful
#audit_axioms StableHLO.dropoutB_back_faithful
#audit_axioms StableHLO.den_dropoutB_of_dropScale
#audit_axioms Proofs.dropPath_scales_uniformly
#audit_axioms Proofs.dropout_eq_reference
#audit_axioms Proofs.dropout_ones_id
#audit_axioms Proofs.dropout_vjp_is_self
#audit_axioms StableHLO.mlpFwdGraph_faithful
#audit_axioms StableHLO.mlpBackGraph_faithful
-- Printer-faithfulness, Chapter 4 (CNN)
#audit_axioms StableHLO.flatConvF_faithful
#audit_axioms StableHLO.maxPoolF_faithful
#audit_axioms StableHLO.cnnFwdGraph_faithful
#audit_axioms StableHLO.convBack_faithful
#audit_axioms StableHLO.maxPoolBack_faithful
-- He et al.'s 3×3/s2 STEM POOL
#audit_axioms win3RowInv_first_dup
#audit_axioms win3ColInv_first_dup
#audit_axioms win3Row_mem_le_two
-- the argmax: `Finset.sup'` over `Fin 3 × Fin 3`, which is why window size stopped mattering
#audit_axioms maxPool3s2_eq_at_max
-- the local linearisation and the smooth-point VJP
#audit_axioms maxPool3s2HasVJPAt3
#audit_axioms maxPool3s2Flat_differentiableAt
#audit_axioms maxPool3s2FlatHasVJPAt
-- the DISCHARGE lemma for the smoothness hypothesis
#audit_axioms maxPool3s2Smooth_of_injective
-- and the float side (`floatClose_maxPool`'s peer)
#audit_axioms floatClose_maxPool3s2
-- and the codegen that denotes it
#audit_axioms StableHLO.maxPool3s2F_faithful
#audit_axioms StableHLO.maxPool3s2Back_faithful
-- The whole-chain CNN backward graph denotes the proven conditional whole-network VJP
#audit_axioms StableHLO.cnnBackGraph_faithful

-- Chapter-5 CIFAR-10 2D CNN (no BN)
#audit_axioms cifarCnnHasVJPAt
#audit_axioms StableHLO.cifarFwdGraph_faithful
-- Concrete tiny CIFAR instance
#audit_axioms Tiny.cifarTinyCnnHasVJP_correct

-- Chapter-5 BatchNorm backward render
#audit_axioms StableHLO.bnBack_faithful

-- Deeper 8-conv CIFAR (the pedagogical BN-acceleration demo)
#audit_axioms cifarCnn8HasVJPAt
#audit_axioms cifarCnnBn8HasVJPAt_correct
-- ...and the rendered-FORWARD peer
#audit_axioms StableHLO.cifar8FwdGraph_faithful
#audit_axioms StableHLO.cifar8BnFwdGraph_faithful

-- Chapter-6 ResNet **Milestone B** (toward real ResNet-34)
-- ...and its weight-VJP (the kernel grad for training a strided block)
#audit_axioms flatConvStride2WeightGradHasVJP_correct
-- ResNet-34's non-degeneracy, ON THE NET THE ARTIFACTS RUN (ResNet34FullBSeal.lean): both
-- levels are stated on `resnet34ForwardBFull` itself — full width, batch BN, 224x224.
#audit_axioms R34FullBSeal.sealX_nonconstant
#audit_axioms R34FullBSeal.sealX_jacobian_nonzero
#audit_axioms R34FullBSeal.sealX_backward_nontrivial
-- ResNet-50's, likewise on `resnet50ForwardBFull` — 48 relu clauses, and `q` a binder, so ONE
-- statement seals both shipped resolutions (224 px at q = 7, 160 px at q = 5).
#audit_axioms R50FullBSeal.sealX_nonconstant
#audit_axioms R50FullBSeal.sealX_jacobian_nonzero
#audit_axioms R50FullBSeal.sealX_backward_nontrivial
-- MobileNetV2's, on `mobilenetv2ForwardBFull` -- all seventeen bottlenecks, batch BN, 224x224,
-- relu6. All 35 kink clauses are weight-only here (every relu6 sits on a BatchNorm output, and
-- the linear bottleneck has no relu after the residual add), and the carrier threads 22
-- BatchNorms because a channel-changing bottleneck has no skip to pass it on.
#audit_axioms Mnv2FullBSeal.sealX_nonconstant
#audit_axioms Mnv2FullBSeal.sealX_jacobian_nonzero
#audit_axioms Mnv2FullBSeal.sealX_backward_nontrivial
-- MobileNetV4-Conv-M's, on `mobilenetv4ForwardBFull` -- all 21 UIB blocks, the fused stage and
-- the two-conv head, batch BN, 224x224. Its 38 kink clauses (counted off the block table by a
-- `#guard`) are weight-only like MobileNetV2's. The witness is grid-constant (one value per example
-- and channel), and the carrier threads 17 BatchNorms: the stem, the fused stage's two, four in
-- each of the three channel-changing rows, and the head's two (the second at 1x1, after the pool).
#audit_axioms Mnv4FullBSeal.sealX_nonconstant
#audit_axioms Mnv4FullBSeal.sealX_jacobian_nonzero
#audit_axioms Mnv4FullBSeal.sealX_backward_nontrivial
-- Per-channel BatchNorm
#audit_axioms bnPerChannelFlatHasVJP_correct
-- The RENDERABLE per-channel BN backward
#audit_axioms bnPerChannelGradInput_correct
-- Per-channel BN on the network's Tensor3 (oc*h)
#audit_axioms bnPerChannelTensor3HasVJP_correct
-- Per-channel BN on the network's Tensor3 (cont.)
#audit_axioms bnPerChannelTensor3GradInput_correct
-- ch8 EfficientNet BATCH-norm
#audit_axioms bnBatchTensor4GradInput_correct
-- sync-BN kit
-- the identity that lets replicas exchange the SECOND MOMENT and each recover the variance
#audit_axioms bnVar_eq_bnMeanSq_sub_sq
-- a statistic of the whole = the mean of the shards' statistics, which is what makes a plain
-- `allReduceMeanF` the right collective. The variance has no such lemma, and cannot.
#audit_axioms bnMean_shard
#audit_axioms bnMeanSq_shard
-- the R = 1 anchors: handed the batch's OWN statistics, the sync forward/backward ARE the
-- existing batch forward/backward — so the single-device renders denote the tied function
#audit_axioms bnEvalForward_at_own_stats
#audit_axioms bnSyncTensor4_at_own_stats
#audit_axioms bnSyncGradInput_at_own_stats
#audit_axioms bnSyncTensor4GradInput_at_own_stats
#audit_axioms bnSyncXhat_at_own_stats
-- Chan's parallel variance — what the exchange carries instead of E[x²], so that no consumer
-- forms E[x²] − μ² (that formulation drifts 2e-4 over R34's 36 layers in f32)
#audit_axioms bnVar_shard_chan
#audit_axioms bnVar_row_shard_chan
-- THE DROP-IN, on actual graph nodes: at R = 1 the sync-BN subgraphs (bnSyncF / bnSyncBack
-- fed by the two-round statistics subgraph / bnSyncDyStatsB) denote exactly what the
-- bnBatchF / bnBatchBack renders denote — so the R = 1 artifacts need not move.
#audit_axioms StableHLO.den_syncStats_R1
-- Sync-BN on replica r IS the shard-r block of the GLOBAL-BATCH forward and
-- input-VJP, handed the all-reduced statistics. The spec does not move: both right-hand sides
-- are the EXISTING bnBatchTensor4 / bnBatchTensor4GradInput at N := R·N.
#audit_axioms bnSyncTensor4_shard_eq_global
#audit_axioms bnSyncTensor4GradInput_shard_eq_global
-- their two halves: pointwise-so-sharding-commutes (no mathematics), and the statistics really
-- being the all-reduced ones (all the mathematics)
#audit_axioms bnSyncTensor4_batchShard
#audit_axioms bnSyncTensor4GradInput_batchShard
#audit_axioms bnMean_row_shard
#audit_axioms bnMeanSq_row_shard
#audit_axioms bnMean_pair_row_shard
-- the layout step the bnchwFwd relabel was hiding
#audit_axioms bnchwFwd_row_batchShard
#audit_axioms StableHLO.den_bnSyncF_allReduce_R1
#audit_axioms StableHLO.den_bnSyncBack_allReduce_R1
-- the fifth op: the γ gradient at the handed-in statistics (bnGammaGradB rebuilds x̂ from the
-- shard's own μ/σ², which under sync-BN is the wrong x̂), its R = 1 anchor, and the γ/β row splits
#audit_axioms bnSyncPerChannelGradGamma_at_own_stats
#audit_axioms bnSyncPerChannelGradGamma_row_shard
#audit_axioms bnPerChannelGradBeta_row_shard
#audit_axioms StableHLO.den_bnSyncGammaGradB_allReduce_R1
-- the handed-back running statistics, read off the packed vector: at R = 1 they ARE
-- bnBatchMeanB / bnBatchVarB
#audit_axioms StableHLO.den_bnStatsMeanB_allReduce_R1
#audit_axioms StableHLO.den_bnStatsVarB_allReduce_R1
-- The per-channel BN SHlo op pair backward-faithfulness
#audit_axioms StableHLO.bnPerChannelBack_faithful
-- Syntactic core
#audit_axioms StableHLO.roundtrip
-- CIFAR-BN render CLOSE
#audit_axioms bnPerChannelGradGamma_correct
#audit_axioms bnPerChannelGradBeta_correct
#audit_axioms bnPerChannel_gamma_sgd_certified
#audit_axioms bnPerChannel_beta_sgd_certified
-- CNN conv-close UPGRADE
#audit_axioms cnn_render_convW2_chain_certified
#audit_axioms cnn_render_convb2_chain_certified
#audit_axioms cnn_render_convW1_chain_certified
#audit_axioms cnn_render_convb1_chain_certified
-- MobileNetV2 CLOSE
#audit_axioms depthwise_bias_sgd_certified
#audit_axioms convStride2_weight_sgd_certified
#audit_axioms convStride2_bias_sgd_certified
-- ResNet-34 cotangent-chain CLOSE
#audit_axioms StableHLO.stemGraphB_faithful
#audit_axioms StableHLO.mbNoExpGraphB_faithful
#audit_axioms StableHLO.mbStridedGraphB_faithful
#audit_axioms StableHLO.mbResidGraphB_faithful
#audit_axioms StableHLO.headGraphB_faithful
-- EfficientNet-B0 cotangent-chain CLOSE
#audit_axioms batchMapHasVJP
#audit_axioms batchMap_differentiable
#audit_axioms reindexHasVJP
#audit_axioms bnBatchLAHasVJP
#audit_axioms bnBatchLA_differentiable
-- Per-block batched gradients (the per-block VJP, the user-requested deliverable)
#audit_axioms mbNoExpFwdBHasVJP
#audit_axioms mbStridedFwdBHasVJP
#audit_axioms mbResidFwdBHasVJP
#audit_axioms headFwdBHasVJP
-- FULL EfficientNet-B0 (all 16 MBConv blocks, real [t,c,n,s,k] spec)
#audit_axioms StableHLO.mbExpGraphB_faithful
#audit_axioms StableHLO.efficientnetFwdGraphBFull_faithful
#audit_axioms efficientnetForwardBFullHasVJP
-- The nested↔∘-chain bridge + correctness on the nested forward itself
#audit_axioms efficientnetForwardBFull_eq_chain
#audit_axioms efficientnetForwardBFullHasVJP_correct
-- ConvNeXt fold
#audit_axioms Proofs.CnxFold.pdiv_layerScaleCh_gamma
#audit_axioms Proofs.CnxFold.cnx_render_lsgammaCh_certified
-- ConvNeXt fold (cont.)
#audit_axioms Proofs.CnxFold.layerScaleChGammaSgd_den
-- The CHANNEL-LN γ/β param certs (ConvNeXtChannelLN)
#audit_axioms Proofs.chanRowsIdxInv_chanRowsIdx
#audit_axioms Proofs.chanRowsIdx_chanRowsIdxInv
#audit_axioms Proofs.pdiv_reindexOut_contract
#audit_axioms Proofs.chanLN_gamma_contract
#audit_axioms Proofs.chanLN_beta_contract
#audit_axioms Proofs.cnx_render_chlngamma_certified
#audit_axioms Proofs.cnx_render_chlnbeta_certified
#audit_axioms Proofs.CnxFold.chanLnGammaSgd_den
#audit_axioms Proofs.CnxFold.chanLnBetaSgd_den
-- ch9-ConvNeXt-T FULL [3,3,9,3] TIE
#audit_axioms Proofs.CnxTie.cnx_block_ch_tied
#audit_axioms Proofs.CnxTie.cnx_down_ch_tied
#audit_axioms Proofs.CnxTie.cnx_stem_ch_tied
#audit_axioms Proofs.CnxTie.cnx_head_ch_tied
#audit_axioms Proofs.CnxTie.cnxLossCot_den
#audit_axioms Proofs.CnxTie.cnx_net_tied_certified
-- ViT CLOSE
#audit_axioms pdiv_rowDense_W
#audit_axioms rowDense_weight_grad_bridge
#audit_axioms rowDense_bias_grad_bridge
#audit_axioms rowDense_weight_sgd_certified
#audit_axioms rowDense_bias_sgd_certified
#audit_axioms pdiv_patchEmbed_pos
#audit_axioms posEmbed_sgd_certified
#audit_axioms pdiv_patchEmbed_cls
#audit_axioms clsToken_sgd_certified
#audit_axioms pdiv_patchEmbed_W
#audit_axioms patchEmbed_weight_grad_bridge
#audit_axioms patchEmbed_weight_sgd_certified
#audit_axioms pdiv_patchEmbed_b
#audit_axioms patchEmbed_bias_grad_bridge
#audit_axioms patchEmbed_bias_sgd_certified
-- ViT cotangent-chain CLOSE
#audit_axioms vitCotDP_eq_sdpaDWeights
#audit_axioms vitCotDS_eq_sdpaDScaled
#audit_axioms vitCotDQ_eq_sdpaBackQ
#audit_axioms vitCotDK_eq_sdpaBackK
#audit_axioms vitCotDV_eq_sdpaBackV
-- ViT SCALING PASS
#audit_axioms layerNormVecHasVJP
#audit_axioms transformerBlockVHasVJPMat
#audit_axioms pdiv_vecLN_gamma
#audit_axioms pdiv_vecLN_beta
#audit_axioms layerNormVec_gamma_grad_bridge
#audit_axioms layerNormVec_beta_grad_bridge
#audit_axioms layerNormVec_gamma_sgd_certified
#audit_axioms layerNormVec_beta_sgd_certified

-- ViT scaling pass: multi-head (ViTMultiHead.lean)
#audit_axioms sum_headPadMat_apply
#audit_axioms mhsaLayer_spelled
#audit_axioms vitBlockSpelledMHV_eq
#audit_axioms StableHLO.den_headsSumG

-- ViT scaling pass: depth-k (ViTDepthK.lean)
#audit_axioms vitBodyKVFlat_eq_flatten
#audit_axioms vitBodyKVFlatHasVJP
#audit_axioms vitForwardKVHasVJP
#audit_axioms vitForwardKVHasVJP_correct
#audit_axioms StableHLO.vitBodyGraphKMHV_den
#audit_axioms StableHLO.vitFwdGraphKMHV_faithful

-- Full ConvNeXt-T [3,3,9,3] (ConvNeXtFullT.lean)
#audit_axioms decimateOddFlatHasVJP
#audit_axioms flatConvStride4HasVJP
#audit_axioms convNextStageChKHasVJP
#audit_axioms cnxDownChWHasVJP
#audit_axioms convNextForwardTChHasVJP
-- The nested↔∘-chain bridge
#audit_axioms convNextForwardTCh_eq_chain
#audit_axioms convNextForwardTCh_differentiable
#audit_axioms convNextForwardTChHasVJP_correct
-- The channel-LN GRAPH + faithfulness (the apex)
#audit_axioms StableHLO.chanLNGraph_faithful
#audit_axioms StableHLO.cnxBlockChGraphW_faithful
#audit_axioms StableHLO.cnxStageChGraphK_den
#audit_axioms StableHLO.cnxDownChGraphW_faithful
#audit_axioms StableHLO.convNextFwdGraphTCh_faithful
-- The SCALAR-LN twin of this chain

-- ℝ→Float32 bridge (FloatBridge.lean, MlpFloatBridge.lean)
#audit_axioms FloatModel.dot_close
-- TreeReduceBridge.lean
#audit_axioms FloatModel.dotMixed
#audit_axioms FloatModel.dot_close_mixed
#audit_axioms FloatModel.dot_close_mixed_uniform
#audit_axioms FloatModel.dotMixed_exact_leaf
-- Threaded through the dense layer
#audit_axioms FloatModel.denseMixed
#audit_axioms FloatModel.dense_close_mixed
#audit_axioms FloatModel.dense_close
#audit_axioms FloatModel.dense_close_fresh
#audit_axioms FloatModel.relu_close
-- MaxPool exact-in-float (CNN.lean)
#audit_axioms maxPoolFlat_close
-- Conv forward rounding budget (Float/ConvFloat.lean)
#audit_axioms conv2d_eq_dense
#audit_axioms convPad_close
#audit_axioms FloatModel.convF
#audit_axioms FloatModel.convF_close
-- The flat conv's float forward (ConvFloat.lean)
#audit_axioms FloatModel.flatConvF
#audit_axioms FloatModel.flatConvF_close
-- Chapter-5 no-BN CIFAR CNN (CifarFloatBridge.lean)
#audit_axioms rsqrt_lipschitz
#audit_axioms bnVar_nonneg
#audit_axioms bnIstd_close
-- bnIstd_close at the OPERATING POINT
#audit_axioms bnIstd_close_at
-- BN float tail (BnFloatBridge.lean)
#audit_axioms FloatModel.bnForwardF
#audit_axioms FloatModel.bnMean_close
#audit_axioms FloatModel.bnVar_close
#audit_axioms FloatModel.bnForward_close_of
#audit_axioms FloatModel.bnForward_close
-- ResNet-34 structural float ops (ResNet34FloatBridge.lean)
#audit_axioms FloatModel.add_close
#audit_axioms FloatModel.gapFlat_close
-- Real-BN input-sensitivity (BnInputBridge.lean)
#audit_axioms bnMean_input_close
#audit_axioms bnVar_input_close
#audit_axioms bnIstd_input_close
#audit_axioms bnForward_input_close
-- Whole-net certificate backbone (FloatComposeBridge.lean)
#audit_axioms FloatClose.comp
#audit_axioms floatClose_relu
#audit_axioms floatClose_flatConv
-- FloatClose max-pool (FloatComposeBridge.lean)
#audit_axioms floatClose_maxPool
-- The r34 wrap that lets the fold RUN on a real block
#audit_axioms floatClose_residualBlock
-- THE FINAL FOLD: floatClose_id + floatClose_iterate
#audit_axioms floatClose_id
#audit_axioms floatClose_iterate
-- EfficientNet float bridge
#audit_axioms floatClose_addResidual
-- The remaining FloatClose instances, all wraps of existing closeness
#audit_axioms globalAvgPoolFlat_eq_bnMean
#audit_axioms floatClose_gap
-- Depthwise conv
#audit_axioms depthwiseConv2d_eq_dense
#audit_axioms FloatModel.depthwiseConv2dF_close
#audit_axioms FloatModel.depthwiseFlatF_close
-- Strided-conv backward (r34 down-blocks + stem)
#audit_axioms Proofs.decimateBack_eq_vjp
-- BN mean at any reduction order (BnFloatBridge.lean)
#audit_axioms Proofs.FloatModel.bnMean_close_of
-- The mnv2 block bridges are generic in the NORMALISATION too (`*Gen`)
#audit_axioms Proofs.mobilenetv2ForwardPaperEval
#audit_axioms Proofs.StableHLO.ivNoExpGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.ivExpOnlyGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.ivResidGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.ivStridedGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.mobilenetv2FwdGraphPaperEval_faithful
-- The EfficientNet-B0 INFERENCE forward and its graph
#audit_axioms Proofs.StableHLO.stemGraphBEval_faithful
#audit_axioms Proofs.StableHLO.mbNoExpGraphBEval_faithful
#audit_axioms Proofs.StableHLO.mbStridedGraphBEval_faithful
#audit_axioms Proofs.StableHLO.mbResidGraphBEval_faithful
#audit_axioms Proofs.StableHLO.headGraphBEval_faithful
-- The B0 stage and block bridges are generic in the NORMALISATION (`*BGen`)
#audit_axioms Proofs.mbExpFwdBEval
#audit_axioms Proofs.StableHLO.mbExpGraphBEval_faithful
#audit_axioms Proofs.StableHLO.mbNoExpGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.mbStridedGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.mbResidGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.mbExpGraphEvalW_faithful
#audit_axioms Proofs.efficientnetForwardBFullEval
#audit_axioms Proofs.StableHLO.efficientnetFwdGraphBFullEval_faithful
-- EfficientNet-B0 with stochastic depth + classifier dropout (EfficientNetFullB0Drop.lean)
#audit_axioms Proofs.efficientnetForwardBFullDrop_none
#audit_axioms Proofs.efficientnetForwardBFullDrop_ones
#audit_axioms Proofs.efficientnetForwardBFullEvalDrop_none
#audit_axioms Proofs.efficientnetForwardBFullEvalDrop_ones
#audit_axioms Proofs.StableHLO.mbResidDropGraphW_faithful
#audit_axioms Proofs.StableHLO.headGraphBDo_faithful
#audit_axioms Proofs.StableHLO.mbResidDropGraphEvalW_faithful
#audit_axioms Proofs.StableHLO.headGraphBEvalDo_faithful
#audit_axioms Proofs.StableHLO.efficientnetFwdGraphBFullDrop_faithful
#audit_axioms Proofs.StableHLO.efficientnetFwdGraphBFullEvalDrop_faithful
#audit_axioms Proofs.mbResidDropW_differentiable
#audit_axioms Proofs.mbResidDropWHasVJP
#audit_axioms Proofs.headDoFwdB_differentiable
#audit_axioms Proofs.headDoFwdBHasVJP
-- Optional drop sites: the site as an Option, its graph node, VJP, homogeneity and shard (Foundation/DropSites.lean)
#audit_axioms Proofs.StableHLO.den_dropPathOptG
#audit_axioms Proofs.StableHLO.den_dropoutOptG
#audit_axioms Proofs.dropoutOptHasVJP
#audit_axioms Proofs.dropPathOptHasVJP
#audit_axioms Proofs.dropoutOpt_differentiable
#audit_axioms Proofs.dropoutOpt_smul
#audit_axioms Proofs.dropoutOpt_shard
#audit_axioms Proofs.dropPathOpt_smul
#audit_axioms Proofs.dropPathOpt_shard
#audit_axioms Proofs.dropPath_shard
#audit_axioms Proofs.dropPathOptFam_shard
#audit_axioms Proofs.dropPathOptFam_scaled
-- The indexed lift: a different per-example map at every example, and one example's drop site
-- (Foundation/Batched/Indexed.lean)
#audit_axioms Proofs.StableHLO.batchSlice_batchMapIdx
#audit_axioms Proofs.StableHLO.batchSlice_batchMapAuxIdx
#audit_axioms Proofs.StableHLO.batchMapIdx_comp
#audit_axioms Proofs.batchMapIdx_differentiableAt
#audit_axioms Proofs.pdiv_batchMapIdx_at
#audit_axioms Proofs.batchMapIdxHasVJPAt
#audit_axioms Proofs.batchMapIdxHasVJP
#audit_axioms Proofs.batchMapAuxIdx_eq_batchMapIdxHasVJPAt
#audit_axioms Proofs.HasGradAt.param_batchMapIdx
#audit_axioms Proofs.HasGradAt.param_batchMapIdx_through
#audit_axioms Proofs.dropScalarOptHasVJP
#audit_axioms Proofs.dropScalarOpt_smul
#audit_axioms Proofs.batchSlice_dropPathOpt
#audit_axioms Proofs.siteResHasVJP
#audit_axioms Proofs.StableHLO.batchShard_batchMapIdx
#audit_axioms Proofs.StableHLO.batchShard_batchMapAuxIdx
#audit_axioms Proofs.batchMapIdx_smul
#audit_axioms Proofs.batchMapAuxIdx_smul
-- One ViT block with its two drop sites, per example (Nets/ViT/ViTDropBlock.lean)
#audit_axioms Proofs.ViTTieGB.fwdOD_none
#audit_axioms Proofs.ViTTieGB.cotInD_none
#audit_axioms Proofs.ViTTieGB.vitAttnBr_back
#audit_axioms Proofs.ViTTieGB.vitMlpBr_back
#audit_axioms Proofs.ViTTieGB.fwdOD_eq_sites
#audit_axioms Proofs.BlockParamsV.fwdODHasVJP
#audit_axioms Proofs.ViTTieGB.vitCotXinV_attn
#audit_axioms Proofs.ViTTieGB.vitCotHVD_eq
#audit_axioms Proofs.ViTTieGB.cotInD_eq_vjp
-- One ConvNeXt block with its drop site, per example (Nets/ConvNeXt/ConvNeXtDropBlock.lean)
#audit_axioms Proofs.CnxTieGB.fwdOD_none
#audit_axioms Proofs.CnxTieGB.cotInD_none
#audit_axioms Proofs.CnxTieGB.cotInD_eq_cotIn
#audit_axioms Proofs.CnxTieGB.bodyF_back
#audit_axioms Proofs.CnxTie.CnxTieBlk.fwdODHasVJP
#audit_axioms Proofs.CnxTieGB.cotInD_eq_vjp
-- The MobileNetV4-Conv-M INFERENCE forward and its graph, at any input size
#audit_axioms Proofs.StableHLO.mnv4BodyGraphBEval_faithful
#audit_axioms Proofs.StableHLO.mnv4StridedGraphBEval_faithful
#audit_axioms Proofs.StableHLO.mobilenetv4ForwardBFullEval
#audit_axioms Proofs.StableHLO.mnv4FwdGraphBFullEval_faithful
-- Classifier dropout (the `%do` form) on MobileNetV2 and MobileNetV4
#audit_axioms Proofs.mobilenetv2ForwardBFullDo_ones
#audit_axioms Proofs.StableHLO.mobilenetv2FwdGraphBFullDo_faithful
#audit_axioms Proofs.MobileNetV2TieB.mobilenetv2ForwardBFullDoOpt_eq_chain
#audit_axioms Proofs.MobileNetV2TieB.mnv2HeadBDoOptHasVJPAt
#audit_axioms Proofs.StableHLO.mobilenetv4ForwardBFullDo_ones
#audit_axioms Proofs.StableHLO.mnv4FwdGraphBFullDo_faithful
-- Stochastic depth (the `%dp` sites) on MobileNetV4 (MobileNetV4FullBDrop.lean) and ViT (ViTFwdDrop.lean)
#audit_axioms Proofs.StableHLO.mnv4SkipDropGraphB_faithful
#audit_axioms Proofs.StableHLO.mobilenetv4ForwardBFullDrop_sdOnes
#audit_axioms Proofs.StableHLO.mobilenetv4ForwardBFullDrop_ones
#audit_axioms Proofs.StableHLO.mnv4FwdGraphBFullDrop_faithful
#audit_axioms Proofs.blockVDrop_one
#audit_axioms Proofs.vitBlockSpelledMHVDrop_eq
#audit_axioms Proofs.vitForwardKVDrop_ones
#audit_axioms Proofs.vitForwardKVDropB_ones
#audit_axioms Proofs.StableHLO.vitBlockGraphBDrop_slice
#audit_axioms Proofs.StableHLO.vitBodyGraphBDrop_slice
#audit_axioms Proofs.StableHLO.vitFwdGraphBDrop_slice
#audit_axioms Proofs.StableHLO.vitFwdGraphBDrop_faithful
-- Integrity tie (the r34 identity block)
#audit_axioms Proofs.convFlatBack_eq_vjp_backward
-- Integrity tie (the r34 DOWNSAMPLE block)
#audit_axioms Proofs.flatConvStride2Back_eq_vjp_backward
-- Its XLA-SAME peer (the TF-origin B0 / MobileNetV2 stems): the same conv leaf + decimateOddBack rfl.
#audit_axioms Proofs.flatConvStride2XlaBack_eq_vjp_backward
-- DEPTHWISE adjoint gate (shared prereq for convnext/mnv2/enet)
#audit_axioms Proofs.depthwiseConv2d_dwReverse_eq_input_grad_formula
#audit_axioms Proofs.depthwiseFlatBack_eq_vjp_backward
-- Its XLA-SAME peer (MobileNetV2's four strided depthwises, B0's downsample depthwise).
#audit_axioms Proofs.depthwiseStride2FlatXlaBack_eq_vjp_backward
-- At ConvNeXt's REAL channel LayerNorm
#audit_axioms Proofs.HasVJP.backward_unique
#audit_axioms Proofs.bnGradInput_eq_vjp_backward
#audit_axioms Proofs.transposeFlatHasVJP_backward_eq
#audit_axioms Proofs.rowLNVecFlatHasVJP_backward_eq
#audit_axioms Proofs.chanLNTensor3Back_eq_chanLN_vjp
#audit_axioms Proofs.cnxBodyWithChanLNBack_eq_vjp
#audit_axioms Proofs.cnxBlockChBack_eq_vjp
-- Integrity tie (vit MHSA — the sdpa adjoint)
#audit_axioms Proofs.projBack_core_coord
#audit_axioms Proofs.woback_unflatten
#audit_axioms Proofs.mhsaBackFlat_eq_mhsa_vjp
-- The MLP-sublayer per-token leaf
#audit_axioms Proofs.transformerMlp_back_flat_eq_perRowFlatPR
-- Endpoint leaf ties
#audit_axioms Proofs.dense_transpose_eq_vjp_backward
#audit_axioms Proofs.gapBack_eq_vjp_backward
-- He et al.'s 3×3/s2 stem pool's BACKWARD
#audit_axioms Proofs.maxPool3s2FlatBack_eq_vjp_backward
#audit_axioms Proofs.maxPool3s2FlatHasVJPAtVec
-- the two spellings of that scatter are one map: the render's `.maxPool3s2BackB` node = the chain's
#audit_axioms Proofs.maxPool3s2BackFlat_eq_flatBack
#audit_axioms Proofs.den_maxPool3s2BackB_eq_flatBackB
-- the generic 21-stage apex the batched MobileNetV2 tie instantiates (MobileNetV2WholeBackCertifiedTieB.lean)
#audit_axioms Proofs.mobilenetv2PaperPCHasVJPAt
-- AND FOR THE WHOLE EFFICIENTNET-B0 (EfficientNetWholeBackCertifiedTie.lean)
#audit_axioms Proofs.stemBBack_eq_vjp_backward
#audit_axioms Proofs.headFwdBBack_eq_vjp_backward
-- AND AT THE PAPER DEPTH — all 16 MBConv blocks (EfficientNetFullWholeBackCertifiedTie.lean)
#audit_axioms Proofs.efficientnetBFullHasVJP
#audit_axioms Proofs.efficientnetInputGradBFull_eq_efficientnetB_full_vjp
#audit_axioms Proofs.efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp
#audit_axioms Proofs.efficientnetInputGradBFull_correct
-- AND AT BATCH BATCH-NORM (ResNet34BackCertifiedTieB.lean, MobileNetV2WholeBackCertifiedTieB.lean)
#audit_axioms Proofs.HasVJPAt.backward_unique
#audit_axioms Proofs.cbReluStridedBBack_eq_vjp_backward
#audit_axioms Proofs.r34HeadBBack_eq_vjp_backward
#audit_axioms Proofs.r34BFullHasVJPAt
#audit_axioms Proofs.r34InputGradB_eq_r34B_full_vjp
#audit_axioms Proofs.r34InputGradB_correct
#audit_axioms Proofs.resnet34ForwardBFull_eq_slots
#audit_axioms Proofs.mnv2StemBBack_eq_vjp_backward
#audit_axioms Proofs.cbrBBack_eq_vjp_backward
#audit_axioms Proofs.mnv2InputGradB_eq_mobilenetv2B_full_vjp
#audit_axioms Proofs.mnv2InputGradB_correct
#audit_axioms Proofs.mobilenetv2ForwardBFull_eq_slots
-- AND RESNET-50's T6 (ResNet50WholeBackCertifiedTieB.lean)
#audit_axioms Proofs.r50InputGradB_eq_r34B_full_vjp
#audit_axioms Proofs.r50InputGradB_correct
#audit_axioms Proofs.resnet50ForwardBFull_eq_slots
-- THE EVEN-KERNEL CONV BACKWARD (EvenKernelConvBack.lean)
#audit_axioms Proofs.padOdd
#audit_axioms Proofs.conv2d_padOdd_eq
#audit_axioms Proofs.flatConv_padOdd_eq
#audit_axioms Proofs.HasVJP.backward_unique_of_eq
#audit_axioms Proofs.convFlatBack_padOdd_eq_vjp_backward
#audit_axioms Proofs.flatConvStride2Back_padOdd_eq_vjp_backward
#audit_axioms Proofs.flatConvStride4Back_padOdd_eq_vjp_backward
-- ConvNeXt-T's whole-net backward tie, two pieces
#audit_axioms Proofs.cnxDownChBack_eq_vjp
#audit_axioms Proofs.cnxBlockChBackAt
#audit_axioms Proofs.cnxStageChKBack
#audit_axioms Proofs.cnxStageChKBack_eq_vjp
#audit_axioms Proofs.rowLNVecFlatHasVJP_backward_eq_fun
#audit_axioms Proofs.cnxSavedA10
#audit_axioms Proofs.convNextForwardTChVjpChain
#audit_axioms Proofs.convnextInputGrad_eq_convNextForwardTCh_vjp
-- ConvNeXt WHOLE-NET BACKWARD AT A BATCH — T6 at the shipped index and the ImageNet head (ConvNeXtWholeBackCertifiedTieB.lean)
#audit_axioms Proofs.convnextInputGradB
#audit_axioms Proofs.cnxSavedB10
#audit_axioms Proofs.cnxChanLNBAt
#audit_axioms Proofs.cnxDownBAt
#audit_axioms Proofs.cnxStemBackB_eq_vjp
#audit_axioms Proofs.cnxChanLNBackB_eq_vjp
#audit_axioms Proofs.cnxStageBackB_eq_vjp
#audit_axioms Proofs.cnxDownBackB_eq_vjp
#audit_axioms Proofs.cnxGapBackB_eq_vjp
#audit_axioms Proofs.cnxLNhBackB_eq_vjp
#audit_axioms Proofs.cnxDenseBackB_eq_vjp
#audit_axioms Proofs.vjpCompDiffAt_fst_backward
#audit_axioms Proofs.convNextForwardTChBHasVJPAt
#audit_axioms Proofs.convnextInputGradB_eq_convNextForwardTChB_vjp
#audit_axioms Proofs.convNextForwardTChB_eq_chain
#audit_axioms Proofs.convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp
#audit_axioms Proofs.convnextInputGradB_correct
#audit_axioms Proofs.convnextImagenetInputGradB_eq_vjp
-- ViT-Tiny's whole-net backward tie
#audit_axioms Proofs.vitBlockBackV
#audit_axioms Proofs.vitBlockBackVAt
#audit_axioms Proofs.vitTowerBackK
#audit_axioms Proofs.vitInputGradK
#audit_axioms Proofs.rowLNVecFlatBack_eq_vecLN_vjp
#audit_axioms Proofs.attnSubFlat_tie_v
#audit_axioms Proofs.mlpSubFlat_tie_v
#audit_axioms Proofs.vitBlockBackV_eq_transformerBlockV_vjp
#audit_axioms Proofs.vitBlockBackVAt_eq_vjp
#audit_axioms Proofs.vitHeadBack_eq_classifier_vjp
#audit_axioms Proofs.vitFinalLNBack_eq_vjp
#audit_axioms Proofs.vitTowerBackK_eq_vjp
#audit_axioms Proofs.vitForwardKV_eq_chain
#audit_axioms Proofs.vitInputGradK_eq_vitApexVJP
#audit_axioms Proofs.vitInputGradK_eq_vitForwardKV_vjp
#audit_axioms Proofs.vitInputGradK_correct
#audit_axioms Proofs.vitTinyInputGrad_eq_vitTiny_vjp
#audit_axioms Proofs.vitTinyHasVJP_correct
-- ViT WHOLE-NET BACKWARD AT A BATCH — T6 at the shipped index (ViTWholeBackCertifiedTieB.lean)
#audit_axioms Proofs.vitTowerBackB_eq_vjp
#audit_axioms Proofs.vitLNBackB_eq_vjp
#audit_axioms Proofs.vitHeadBackB_eq_vjp
#audit_axioms Proofs.vitKVBHasVJPAt
#audit_axioms Proofs.vitInputGradKB_eq_vitKVB_vjp
#audit_axioms Proofs.vitForwardKVB_eq_chain
#audit_axioms Proofs.vitForwardKV_differentiable
#audit_axioms Proofs.vitInputGradKB_eq_batchMap_vitForwardKV_vjp
#audit_axioms Proofs.vitInputGradKB_correct
#audit_axioms Proofs.vitTinyInputGradB_eq_vitTiny_vjp
-- ViT TRANSFORMER-BLOCK FOLD
#audit_axioms convWeightGrad_eq_dot
#audit_axioms convBiasGrad_eq_sum
-- CIFAR-8 last-conv SGD descent (SgdDescent/Cifar.lean)
#audit_axioms Proofs.cifarCnn8Forward_factor
#audit_axioms Proofs.cifar8_lastConv_sgd_descends
#audit_axioms FloatModel.mlp_float_close
-- The numeric rung
#audit_axioms FloatModel.pow_gamma_bound
#audit_axioms FloatModel.dense_abs_le
#audit_axioms FloatModel.denseErr_le_uniform
#audit_axioms FloatModel.mlp_float_close_uniform
#audit_axioms FloatModel.mnist_mlp_float_budget
-- The gradient half, first rung
#audit_axioms FloatModel.mul_close
#audit_axioms FloatModel.sgd_step_close
#audit_axioms FloatModel.reluMask_close
#audit_axioms FloatModel.cot_step_close
#audit_axioms FloatModel.mlp_w2_step_float_close
#audit_axioms FloatModel.mlp_b2_step_float_close
#audit_axioms FloatModel.mlp_w1_step_float_close
-- Completion: the remaining param entries (b₁, W₀/b₀)
#audit_axioms FloatModel.mlp_b1_step_float_close
#audit_axioms FloatModel.mlp_w0_step_float_close
#audit_axioms FloatModel.mlp_b0_step_float_close
#audit_axioms FloatModel.mnist_w2_step_float_budget
-- The loss head — the LAST float hypothesis discharged
#audit_axioms FloatModel.sum_close
#audit_axioms FloatModel.softmax_perturb
#audit_axioms FloatModel.softmaxF_close
#audit_axioms FloatModel.softmax_ce_cot_close
#audit_axioms FloatModel.mnist_cot_budget
-- floatbridge quantization
#audit_axioms FloatModel.argmax_preserved
#audit_axioms FloatModel.denseMixedBudget
#audit_axioms FloatModel.dense_close_mixed_uniform_budget
#audit_axioms FloatModel.denseMixedBudget_le_of
#audit_axioms uE4M3
#audit_axioms FloatModel.linear_e4m3_logit_budget
#audit_axioms FloatModel.linear_e4m3_argmax_preserved
-- floatbridge quantization (cont.)
#audit_axioms QuantFold.dequant_factors
#audit_axioms QuantFold.e4m3_render_faithful
-- The bf16-mixed render-tie, its companion (Bf16Fold.lean)
#audit_axioms Proofs.Bf16Fold.bf16_render_faithful
#audit_axioms Proofs.Bf16Fold.bf16_render_faithful_emit
#audit_axioms Proofs.Bf16Fold.bf16_render_faithful_depth2
-- The loss gradient in a parameter, from the gradient at its op's output (ParamGrad.lean)
#audit_axioms Proofs.addConstHasVJPAt
#audit_axioms Proofs.constAddHasVJPAt
#audit_axioms Proofs.hasGradAt_iff
#audit_axioms Proofs.HasGradAt.comp
#audit_axioms Proofs.HasGradAt.congr_left
#audit_axioms Proofs.HasGradAt.congr_of_eventuallyEq
#audit_axioms Proofs.HasGradAt.param
#audit_axioms Proofs.HasGradAt.param_batchMap
-- The batched smoothed loss, and its gradient is the emitted cotangent (SmoothedBatchLoss.lean)
#audit_axioms Proofs.rowSumLoss_pdiv
#audit_axioms Proofs.smoothedBatchLoss_pdiv
#audit_axioms Proofs.smoothedBatchLoss_grad
-- The batched BCE-with-logits loss, and its gradient is the emitted cotangent (BceBatchLoss.lean)
#audit_axioms Proofs.bceBatchLoss_pdiv
#audit_axioms Proofs.bceBatchLoss_grad
-- Each batched parameter gradient node is a loss derivative (ParamGradNodes.lean)
#audit_axioms Proofs.GradNodeB.convW_hasGradAt
#audit_axioms Proofs.GradNodeB.convB_hasGradAt
#audit_axioms Proofs.GradNodeB.convStridedW_hasGradAt
#audit_axioms Proofs.GradNodeB.convStridedB_hasGradAt
#audit_axioms Proofs.GradNodeB.denseW_hasGradAt
#audit_axioms Proofs.GradNodeB.denseB_hasGradAt
#audit_axioms Proofs.GradNodeB.bnGamma_hasGradAt
#audit_axioms Proofs.GradNodeB.bnBeta_hasGradAt
#audit_axioms Proofs.GradNodeB.cStridedInB_eq_batchMapBackward
#audit_axioms Proofs.GradNodeB.dInB_eq_batchMapBackward
#audit_axioms Proofs.GradNodeB.convStridedXlaW_hasGradAt
#audit_axioms Proofs.GradNodeB.convStridedXlaB_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseW_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseB_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseStridedXlaW_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseStridedXlaB_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseStridedW_hasGradAt
#audit_axioms Proofs.GradNodeB.dStridedInB_eq_batchMapBackward
#audit_axioms Proofs.GradNodeB.hasGradAt_bnBatchLA
#audit_axioms Proofs.GradNodeB.hasGradAt_relu
#audit_axioms Proofs.GradNodeB.hasGradAt_conv
#audit_axioms Proofs.GradNodeB.hasGradAt_convStrided
#audit_axioms Proofs.GradNodeB.hasGradAt_depthwise
#audit_axioms Proofs.GradNodeB.hasGradAt_depthwiseStrided
-- ResNet-34: every parameter gradient node IS the batched smoothed loss's derivative (ResNet34ParamGrad.lean)
#audit_axioms Proofs.ResNet34TieB.r34_idblock_lossTiedB
#audit_axioms Proofs.ResNet34TieB.r34_downblock_lossTiedB
#audit_axioms Proofs.ResNet34TieB.r34_stem_lossTiedB
#audit_axioms Proofs.ResNet34TieB.r34_head_lossTiedB
#audit_axioms Proofs.ResNet34TieB.r34_factor_a0
#audit_axioms Proofs.ResNet34TieB.r34_net_lossGrad
#audit_axioms Proofs.ResNet34TieB.r34_net_lossGrad_smoothedCE
#audit_axioms Proofs.ResNet34TieB.r34_net_tied_lossGrad
-- ResNet-50: every parameter gradient node IS the loss's derivative, for both shipped losses (ResNet50ParamGrad.lean)
#audit_axioms Proofs.ResNet50TieB.r50_idblock_lossTiedB
#audit_axioms Proofs.ResNet50TieB.r50_projblock_lossTiedB
#audit_axioms Proofs.ResNet50TieB.r50_downblock_lossTiedB
#audit_axioms Proofs.ResNet50TieB.r50_factor_s1b0
#audit_axioms Proofs.ResNet50TieB.r50_net_lossGrad
#audit_axioms Proofs.ResNet50TieB.r50_net_lossGrad_smoothedCE
#audit_axioms Proofs.ResNet50TieB.r50_net_lossGrad_bce
#audit_axioms Proofs.ResNet50TieB.r50_net_tied_lossGrad
-- MobileNetV2: every parameter gradient node IS the loss's derivative (MobileNetV2ParamGrad.lean)
#audit_axioms Proofs.MobileNetV2TieB.mnv2_stem_lossTiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_noexp_lossTiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_stride1_lossTiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_resid_lossTiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_stride2_lossTiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_head_lossTiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_factor_b1
#audit_axioms Proofs.MobileNetV2TieB.mnv2_net_lossGrad
#audit_axioms Proofs.MobileNetV2TieB.mnv2_net_lossGrad_smoothedCE
#audit_axioms Proofs.MobileNetV2TieB.mnv2_net_tied_lossGrad
-- MobileNetV4-Conv-M: every parameter gradient node IS the loss's derivative (MobileNetV4ParamGrad.lean)
#audit_axioms Proofs.Mnv4TieB.mnv4_stem_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_fused_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_head_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_extradw_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_convnext_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_ffn_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_strided_lossTiedB
#audit_axioms Proofs.Mnv4TieB.mnv4_factor_stem
#audit_axioms Proofs.Mnv4TieB.mnv4_factor_b21
#audit_axioms Proofs.Mnv4TieB.mnv4_net_lossGrad
#audit_axioms Proofs.Mnv4TieB.mnv4_net_lossGrad_smoothedCE
#audit_axioms Proofs.Mnv4TieB.mnv4_net_tied_lossGrad
-- EfficientNet-B0: every parameter gradient node IS the loss's derivative (EfficientNetParamGrad.lean)
#audit_axioms Proofs.GradNodeB.biasBeta_hasGradAt
#audit_axioms Proofs.GradNodeB.hasGradAt_swish
#audit_axioms Proofs.GradNodeB.hasGradAt_bnBackB
#audit_axioms Proofs.BackLinks.seGateMulBHasVJP
#audit_axioms Proofs.EnetTieG.seB_eq_gateMul
#audit_axioms Proofs.EnetTieG.enet_se_lossTiedB
#audit_axioms Proofs.EnetTieG.enet_exp_lossTiedG
#audit_axioms Proofs.EnetTieG.enet_resid_lossTiedG
#audit_axioms Proofs.EnetTieG.enet_strided_lossTiedG
#audit_axioms Proofs.EnetTieG.enet_noexp_lossTiedG
#audit_axioms Proofs.EnetTieG.enet_stem_lossTiedG
#audit_axioms Proofs.EnetTieG.enet_head_lossTiedG
#audit_axioms Proofs.EnetTieG.enet_factor_b1
#audit_axioms Proofs.EnetTieG.enet_net_lossGrad
#audit_axioms Proofs.EnetTieG.enet_net_lossGrad_smoothedCE
#audit_axioms Proofs.EnetTieG.enet_net_tied_lossGrad
-- ConvNeXt-T: every parameter gradient node IS the loss's derivative (ConvNeXtParamGrad.lean)
#audit_axioms Proofs.hasGradAt_linLoss
#audit_axioms Proofs.HasGradAt.param_batchMap_through
#audit_axioms Proofs.smoothedBatchLossDiv_grad
#audit_axioms Proofs.GradNodeB.pdiv_bias_of_split
#audit_axioms Proofs.CnxTieGB.cnxBlk_hasGradAt
#audit_axioms Proofs.CnxTieGB.cnx_block_lossTiedGB
#audit_axioms Proofs.CnxTieGB.cnx_down_lossTiedGB
#audit_axioms Proofs.CnxTieGB.cnx_stem_lossTiedGB
#audit_axioms Proofs.CnxTieGB.cnx_head_lossTiedGB
#audit_axioms Proofs.CnxTieGB.cnx_factor_b1
#audit_axioms Proofs.CnxTieGB.cnx_logitsB_eq
#audit_axioms Proofs.CnxTieGB.cnx_net_lossGrad
#audit_axioms Proofs.CnxTieGB.cnxNetB_eq_convNextForwardTCh
#audit_axioms Proofs.CnxTieGB.cnx_net_lossGrad_smoothedCE
#audit_axioms Proofs.CnxTieGB.cnx_net_tied_lossGrad
-- ViT-Tiny: every parameter gradient node IS the loss's derivative (ViTParamGrad.lean)
#audit_axioms Proofs.pdivMat_colIndepH
#audit_axioms Proofs.colSlabwiseHasVJPMatH
#audit_axioms Proofs.ViTTieGB.attnCoreQHasVJPMat
#audit_axioms Proofs.ViTTieGB.attnCoreKHasVJPMat
#audit_axioms Proofs.ViTTieGB.attnCoreVHasVJPMat
#audit_axioms Proofs.ViTTieGB.attnCoreQ_backward
#audit_axioms Proofs.ViTTieGB.vitMlpSite_hasGradAt
#audit_axioms Proofs.ViTTieGB.hasGradAt_constAdd_site
#audit_axioms Proofs.ViTTieGB.vitPostF2_hasGradAt
#audit_axioms Proofs.ViTTieGB.vitPostQ_hasGradAt
#audit_axioms Proofs.ViTTieGB.vitPostL1_hasGradAt
#audit_axioms Proofs.ViTTieGB.vitPostL2_hasGradAt
#audit_axioms Proofs.ViTTieGB.vitPostF1_hasGradAt
#audit_axioms Proofs.ViTTieGB.vit_fwd_Wq
#audit_axioms Proofs.ViTTieGB.vit_block_lossTiedGB
#audit_axioms Proofs.ViTTieGB.vit_head_lossTiedGB
#audit_axioms Proofs.ViTTieGB.vit_embed_lossTiedGB
#audit_axioms Proofs.ViTTieGB.vit_factor_b1
#audit_axioms Proofs.ViTTieGB.vit_logitsB_eq
#audit_axioms Proofs.ViTTieGB.vit_net_lossGrad
#audit_axioms Proofs.ViTTieGB.fwdO_eq_blockVFlat
#audit_axioms Proofs.ViTTieGB.vitNetB_eq_vitForwardKV
#audit_axioms Proofs.ViTTieGB.vit_net_lossGrad_smoothedCE
#audit_axioms Proofs.ViTTieGB.vit_net_tied_lossGrad
-- The chapter nets: every parameter gradient node IS the loss's derivative, pools up to twins
-- (SmallParamGrad.lean and the Nets/Small *ParamGrad files)
#audit_axioms Proofs.SmallParamGrad.hasGradAt_crossEntropy
#audit_axioms Proofs.SmallParamGrad.convW_hasGradAt
#audit_axioms Proofs.SmallParamGrad.convB_hasGradAt
#audit_axioms Proofs.SmallParamGrad.denseW_hasGradAt
#audit_axioms Proofs.SmallParamGrad.denseB_hasGradAt
#audit_axioms Proofs.SmallParamGrad.convWeightSgd_eq_grad
#audit_axioms Proofs.SmallParamGrad.weightSgd_eq_grad
#audit_axioms Proofs.SmallParamGrad.hasGradAt_conv
#audit_axioms Proofs.SmallParamGrad.poolGatherFlat_eq_sel
#audit_axioms Proofs.SmallParamGrad.hasGradAt_gatherRelu
#audit_axioms Proofs.SmallParamGrad.maxPool_relu_eventuallyEq_sel
#audit_axioms Proofs.SmallParamGrad.maxpool_flatDenote_eq_selScatter
#audit_axioms Proofs.LinFold.linear_net_lossGrad
#audit_axioms Proofs.LinFold.linear_net_lossGrad_CE
#audit_axioms Proofs.MlpFold.mlp_net_lossGrad
#audit_axioms Proofs.MlpFold.mlp_net_lossGrad_CE
#audit_axioms Proofs.CnnFold.cnnPoolTwin_of_convPatchEq2
#audit_axioms Proofs.CnnFold.cnn_net_lossGrad
#audit_axioms Proofs.CnnFold.cnn_net_lossGrad_CE
#audit_axioms Proofs.CnnFold.cnnChainCotW2_eq_sel
#audit_axioms Proofs.CifarFold.cifar_net_lossGrad
#audit_axioms Proofs.CifarFold.cifar_net_lossGrad_CE
#audit_axioms Proofs.CifarFold.cifarChainCotW2_eq_sel
#audit_axioms Proofs.Cifar8TieG.cifar8_net_lossGrad
#audit_axioms Proofs.Cifar8TieG.cifar8_net_lossGrad_CE
#audit_axioms Proofs.Cifar8BnTieG.bnGamma_hasGradAt
#audit_axioms Proofs.Cifar8BnTieG.bnBeta_hasGradAt
#audit_axioms Proofs.Cifar8BnTieG.hasGradAt_bnPC
#audit_axioms Proofs.Cifar8BnTieG.cifar8Bn_net_lossGrad
#audit_axioms Proofs.Cifar8BnTieG.cifar8Bn_net_lossGrad_CE
-- Inexact-gradient descent over ℝ (SgdDescent/Basic.lean)
#audit_axioms fderiv_apply_eq_sum_grad
#audit_axioms descent_segment
#audit_axioms sgd_descent_inexact
#audit_axioms sgd_descends
-- The smoothness hypothesis discharged for the Chapter-2 net
#audit_axioms gradAt_eq_pdiv
#audit_axioms linear_loss_gradAt
#audit_axioms dense_unflatten_drift
#audit_axioms linear_loss_grad_lipschitz
#audit_axioms linear_sgd_descends
-- The η-composition, the "two halves finally meet"
#audit_axioms FloatModel.linearFloatGrad
#audit_axioms linearFloatGrad_apply
#audit_axioms linear_grad_close
#audit_axioms linear_float_sgd_descends
-- The smoothness hypothesis discharged through the Chapter-3 MLP
#audit_axioms relu_entry_lipschitz
#audit_axioms sign_stable_of_close
#audit_axioms sum_abs_flatten_cols
#audit_axioms dense_unflatten_diff
#audit_axioms dense_unflatten_col_drift
#audit_axioms dense_unflatten_drift_sum
#audit_axioms margin_keeps_offkink
#audit_axioms margin_keeps_offkink_mid
#audit_axioms ce_dense_input_grad
#audit_axioms ce_head_relu_input_grad
#audit_axioms ce_head2_input_grad
#audit_axioms mlp_hidden_loss_differentiableAt
#audit_axioms mlp_hidden_loss_gradAt
#audit_axioms mlp_hidden_logit_drift
#audit_axioms mlp_hidden_loss_grad_lipschitz
#audit_axioms mlp_hidden_sgd_descends
-- Output-layer η-composition (for the MLP)
#audit_axioms mlp_output_float_sgd_descends
-- Hidden-layer float-backward grad-close (the joint-step engine)
#audit_axioms FloatModel.cotErr_nonneg
#audit_axioms mlp_w1_grad_close
-- Hidden-layer η-composition (descent)
#audit_axioms FloatModel.mlpHiddenFloatGrad
#audit_axioms mlpHiddenFloatGrad_apply
#audit_axioms mlp_hidden_loss_gradAt_reluMask
#audit_axioms mlp_hidden_float_sgd_descends
#audit_axioms mlp_input_loss_differentiableAt
#audit_axioms mlp_input_loss_gradAt
#audit_axioms mlp_input_loss_grad_lipschitz
#audit_axioms mlp_input_sgd_descends
-- Input-layer η-composition (descent)
#audit_axioms reluMask_dense_transpose_eq
#audit_axioms FloatModel.mlpInputFloatGrad
#audit_axioms mlpInputFloatGrad_apply
#audit_axioms mlp_input_loss_gradAt_reluMask
#audit_axioms mlp_w0_grad_close
#audit_axioms mlp_input_float_sgd_descends
-- Dense-bias descent (SgdDescent/MlpBias.lean)
#audit_axioms gradAt_bias_eq_pdiv
#audit_axioms linear_bias_sgd_descends
#audit_axioms mlp_hidden_bias_loss_grad_lipschitz
#audit_axioms mlp_hidden_bias_sgd_descends
#audit_axioms mlp_input_bias_loss_grad_lipschitz
#audit_axioms mlp_input_bias_sgd_descends
-- The descent program reaches the Chapter-4 CNN (SgdDescent/Cnn.lean)
#audit_axioms flatten_t3Idx
#audit_axioms sum_t3
#audit_axioms sum_window_cells
#audit_axioms lt_of_lt_gap_of_close
#audit_axioms MaxPool2MarginQ.smooth_of_close
#audit_axioms MaxPool2MarginQ.smooth
#audit_axioms MaxPool2MarginQ.isArgmax_iff
-- Float-bridge CNN descent keystone: the pool selector is an indicator pass-through in float
#audit_axioms MaxPool2MarginQ.poolBack_close
-- The pool margin up to twins, and the pool as a fixed gather at a twin tie (ConvIndex, WindowMax)
#audit_axioms windowSmoothUpTo_of_margin
#audit_axioms WindowMarginUpTo.mono
#audit_axioms windowMarginUpTo_of_cert
#audit_axioms windowMax_eq_windowGather
#audit_axioms poolGatherFlat_l1_contract
#audit_axioms pdiv_poolGatherFlat
#audit_axioms maxPoolFlat_eq_poolGatherFlat
#audit_axioms MaxPool2MarginQ.to_marginQUpTo
#audit_axioms MaxPool2MarginQ.to_marginQUpTo_flat
#audit_axioms conv2d_eq_convPad
#audit_axioms abs_convPad_le
#audit_axioms sum_abs_kernel_slab_le
#audit_axioms sum_abs_k4
#audit_axioms conv2d_kernel_sub
#audit_axioms conv2d_kernel_drift
#audit_axioms conv2d_kernel_drift_total
#audit_axioms conv2d_kernel_drift_sum
-- The conv2-layer rung, assembled (SgdDescent/Cnn.lean)
#audit_axioms ce_head3_input_grad
#audit_axioms pool_relu_input_grad
#audit_axioms conv2d_weight_pdiv
#audit_axioms cnn_conv2_loss_gradAt
-- Float-bridge CNN descent keystone: the certified conv-2 gradient in dense/reluMask form
#audit_axioms dense_transpose_eq
#audit_axioms head3_cot_reluMask
#audit_axioms cnn_conv2_loss_gradAt_reluMask
-- Float-bridge CNN descent
#audit_axioms t3Idx_surj
#audit_axioms mask_scalar_close
#audit_axioms FloatModel.dot_perturbed_close
#audit_axioms FloatModel.cnnConv2FloatGrad
#audit_axioms FloatModel.cnnConv2FloatGrad_apply
#audit_axioms FloatModel.cnnConv2GradBudget
#audit_axioms cnn_conv2_grad_close
#audit_axioms head3_sum_drift
#audit_axioms conv2d_eq_of_convPatchEq
#audit_axioms convPatchEq_relu_conv
#audit_axioms gather_relu_input_grad
#audit_axioms Conv2Slot.maxPool_relu_eventuallyEq_gather
#audit_axioms Conv2Slot.marginUpTo_seg
#audit_axioms Conv2Slot.gather_grad_lipschitz
#audit_axioms Conv2Slot.sgd_descends
#audit_axioms Conv2Slot.marginUpTo_strict
#audit_axioms Conv2Slot.gradAt_abs_le
#audit_axioms cnn_conv2_sgd_descends
#audit_axioms cnn_conv2_exact_sgd_descends
-- Float-bridge CNN descent (cont.)
#audit_axioms flatten_k4Idx
#audit_axioms k4Idx_surj
#audit_axioms cnn_conv2_float_sgd_descends
-- The conv1 rung (SgdDescent/Cnn.lean)
#audit_axioms abs_convTap_expand
#audit_axioms convTap_out_l1
#audit_axioms conv2d_input_pdiv3
#audit_axioms conv2d_flat_input_pdiv
#audit_axioms conv2d_input_entry_drift
#audit_axioms conv2d_input_l1_drift
#audit_axioms cnn1_pool_head_input_grad
#audit_axioms cnn_conv1_loss_gradAt
-- Float-bridge CNN descent keystone
#audit_axioms cnn_conv1_loss_gradAt_reluMask
#audit_axioms Conv1Slot.sgd_descends
#audit_axioms Conv1Slot.gradAt_abs_le
#audit_axioms cnn_conv1_sgd_descends
#audit_axioms ConvPatchEq2.symm
#audit_axioms ConvPatchEq2.trans
#audit_axioms cnn_conv1_exact_sgd_descends
-- Float-bridge CNN descent (cont.)
#audit_axioms convTap_abs_le
#audit_axioms FloatModel.cnnConv2CotBudget
#audit_axioms FloatModel.cnnConv2CotMag
#audit_axioms cnn_conv2_cot_close
#audit_axioms cnn_conv2_cot_real_abs_le
#audit_axioms abs_le_of_close
#audit_axioms convTap_back_close
#audit_axioms convTap_back_abs_le
#audit_axioms FloatModel.cnnConv1CotF
#audit_axioms cnnConv1CotR
#audit_axioms FloatModel.cnnConv1CotBudget
#audit_axioms cnn_conv1_cot_close
#audit_axioms FloatModel.cnnConv1FloatGrad
#audit_axioms FloatModel.cnnConv1FloatGrad_apply
#audit_axioms FloatModel.cnnConv1GradBudget
#audit_axioms cnn_conv1_grad_close
#audit_axioms cnn_conv1_float_sgd_descends
-- The conv BIAS rungs (SgdDescent/Cnn.lean)
#audit_axioms conv2d_bias_sub
#audit_axioms conv2d_flat_bias_drift_total
#audit_axioms conv2d_flat_bias_drift_sum
#audit_axioms conv2d_bias_pdiv
#audit_axioms cnn_conv2_bias_loss_gradAt
#audit_axioms cnn_conv2_bias_sgd_descends
#audit_axioms cnn_conv2_bias_exact_sgd_descends
#audit_axioms cnn_conv1_bias_loss_gradAt
#audit_axioms cnn_conv1_bias_sgd_descends
#audit_axioms cnn_conv1_bias_exact_sgd_descends
-- Float-bridge CNN descent (cont.)
#audit_axioms FloatModel.sum_perturbed_close
#audit_axioms FloatModel.cnnConv2BiasFloatGrad
#audit_axioms cnn_conv2_bias_loss_gradAt_reluMask
#audit_axioms FloatModel.cnnConv2BiasGradBudget
#audit_axioms cnn_conv2_bias_grad_close
#audit_axioms cnn_conv2_bias_float_sgd_descends
#audit_axioms FloatModel.cnnConv1BiasFloatGrad
#audit_axioms cnn_conv1_bias_loss_gradAt_reluMask
#audit_axioms FloatModel.cnnConv1BiasGradBudget
#audit_axioms cnn_conv1_bias_grad_close
#audit_axioms cnn_conv1_bias_float_sgd_descends
-- Adam/AdamW optimizer step over ℝ
#audit_axioms adamVNext_nonneg
#audit_axioms adam_denom_pos

-- EfficientNet backward-graph faithfulness (den-level)
#audit_axioms StableHLO.residualBackGraph_faithful
#audit_axioms StableHLO.seBlockBackGraph_faithful
#audit_axioms StableHLO.bnBack_faithful_fn
-- The channel-LN backward graphs
#audit_axioms Proofs.rowLNBack_affine_eq
#audit_axioms StableHLO.chanLNBackGraph_faithful
#audit_axioms StableHLO.chanLNBackGraph_eq_vjp
#audit_axioms StableHLO.cnxBlockBodyChBackGraph_faithful
#audit_axioms StableHLO.cnxDownChBackGraph_faithful
#audit_axioms StableHLO.bnBatchBack_faithful
#audit_axioms StableHLO.convBackBatched_faithful
#audit_axioms StableHLO.depthwiseBackBatched_faithful
#audit_axioms StableHLO.bnBatchLA_back_conj
#audit_axioms StableHLO.bnBatchLABack_faithful
-- the render's `.bnBatchBack` node and the ties' `.bnBatchLABack` node denote one map (up to reassocB)
#audit_axioms BackLinks.den_bnBatchLABack_eq_bnBatchBack
#audit_axioms BackLinks.bnBackB_eq_den_bnBatchBack
#audit_axioms StableHLO.seBackBatched_faithful
-- Batched MBConv stage backward graphs (the bn wrapper lets these compose).
#audit_axioms StableHLO.cbsBackBatchedGraph_faithful
#audit_axioms StableHLO.dwbsBackBatchedGraph_faithful
#audit_axioms StableHLO.projBackBatchedGraph_faithful
-- Capstone: the whole batched MBConv residual block backward graph.
#audit_axioms StableHLO.mbBodyBackBatchedGraph_faithful
#audit_axioms StableHLO.mbResidBlockBackBatchedGraph_faithful

-- EfficientNet DOWNSAMPLE MBConv body backward-graph faithfulness
#audit_axioms StableHLO.depthwiseStridedBackBatched_faithful
-- Its XLA-SAME peer, the token MobileNetV2's Adam render emits at its four strided depthwises.
#audit_axioms StableHLO.depthwiseStridedXlaBackBatched_faithful
-- Batched strided depthwise → bn → swish stage backward graph.
#audit_axioms StableHLO.dwbsSBackBatchedGraph_faithful
-- Capstone: the batched EfficientNet downsample MBConv body backward graph.
#audit_axioms StableHLO.mbDownBodyBackBatchedGraph_faithful

-- MobileNetV2 backward-graph faithfulness (den-level)
#audit_axioms StableHLO.cbrBackBatchedGraph_faithful
#audit_axioms StableHLO.dwbrBackBatchedGraph_faithful
-- The SE-less inverted-residual body backward graph (projB ∘ dwbrB ∘ cbrB).
#audit_axioms StableHLO.mnv2BodyBackBatchedGraph_faithful
-- Capstone: the whole batched MobileNetV2 inverted-residual block backward graph.
#audit_axioms StableHLO.mnv2ResidBlockBackBatchedGraph_faithful
-- DOWNSAMPLE (stride-2)
#audit_axioms StableHLO.dwbrBstridedBackBatchedGraph_faithful
-- Capstone: the batched MobileNetV2 downsample inverted-residual body backward graph.
#audit_axioms StableHLO.mnv2DownBodyBackBatchedGraph_faithful

-- ResNet-34 backward-graph faithfulness (den-level)
#audit_axioms StableHLO.cbReluBackBatchedGraph_faithful
-- Capstone: the whole batched ResNet-34 identity basic block backward graph
#audit_axioms StableHLO.r34BasicBlockBackBatchedGraph_faithful

-- ResNet-34 DOWNSAMPLE/STRIDED basic block backward-graph faithfulness
#audit_axioms StableHLO.convStridedBackBatched_faithful
-- Capstone: the whole batched ResNet-34 downsample basic block backward graph
#audit_axioms StableHLO.r34DownBlockBackBatchedGraph_faithful

-- ViT whole-block backward-graph faithfulness, lifted to GENERAL MULTI-HEAD
#audit_axioms StableHLO.mhsa_backward_collapseMH
#audit_axioms StableHLO.mhsaBackGraphMH_faithful

-- ViT whole-block backward-graph faithfulness at the FULL PRODUCTION config
#audit_axioms StableHLO.rowVecLNBack_eq_backward
#audit_axioms StableHLO.mlpSublayerVBackGraph_faithfulMH
#audit_axioms StableHLO.attnSublayerVBackGraphMH_faithful
#audit_axioms StableHLO.transformerBlockVBackGraphMH_faithful

-- ViT WHOLE-NET backward-graph faithfulness
#audit_axioms StableHLO.classifierBackGraph_faithful
#audit_axioms StableHLO.finalLNBackGraph_faithful
#audit_axioms StableHLO.transformerBlockVBackGraphMHP_faithful
#audit_axioms StableHLO.patchEmbedBackGraph_faithful
#audit_axioms StableHLO.vitNetBackGraph_faithful

-- ViT folded onto the net-agnostic `CertLayer` machinery (ViTBackNet.lean)
#audit_axioms StableHLO.vitTrunkV_fwd
#audit_axioms StableHLO.vitTrunkV_graph

-- THE WHOLE NET AS ONE `CertLayer`
#audit_axioms StableHLO.vitPatchEmbedLayer
#audit_axioms StableHLO.vitFinalLNLayer
#audit_axioms StableHLO.vitClassifierLayer
#audit_axioms StableHLO.vitNetLayer_fwd
#audit_axioms StableHLO.vitNetLayer_graph
#audit_axioms StableHLO.vitNetLayer_ok

-- THE MACHINERY FIRST — `CertLayer.comp`, the single composition proof every fold routes through
#audit_axioms StableHLO.CertLayer.id'
#audit_axioms StableHLO.CertLayer.comp
#audit_axioms StableHLO.CertLayer.residual
#audit_axioms StableHLO.CertLayer.chain
#audit_axioms StableHLO.CertLayer.chain_fwd
#audit_axioms StableHLO.CertLayer.chain_faithful

-- R50 — the three bottleneck forms (identity / stride-1 projection / strided projection)
#audit_axioms StableHLO.r50BottleneckLayer
#audit_axioms StableHLO.r50ProjBlockLayer
#audit_axioms StableHLO.r50DownBlockLayer

-- r34 / mnv2 / enet / convnext
#audit_axioms StableHLO.cnxBlockChLayer
#audit_axioms StableHLO.r34BasicBlockLayer
#audit_axioms StableHLO.r34DownBlockLayer

-- MNv4 — the four UIB families COLLAPSED into one body

-- MNv4's NET level (T1, T2)
#audit_axioms StableHLO.mobilenetv4ForwardBFullHasVJPAt
#audit_axioms StableHLO.mobilenetv4ForwardBFullHasVJPAt_correct
#audit_axioms StableHLO.mobilenetv4ForwardBFull_differentiableAt
#audit_axioms StableHLO.mnv4FwdGraphBFull_faithful
#audit_axioms StableHLO.mnv4ExtraDWBodyGraphB_faithful
#audit_axioms StableHLO.mnv4ConvNeXtBodyGraphB_faithful
#audit_axioms StableHLO.mnv4FfnBodyGraphB_faithful
#audit_axioms StableHLO.mnv4StridedGraphB_faithful

-- MNv4's T3 tie
#audit_axioms Mnv4TieB.mnv4_extradw_tiedB
#audit_axioms Mnv4TieB.mnv4_convnext_tiedB
#audit_axioms Mnv4TieB.mnv4_ffn_tiedB
#audit_axioms Mnv4TieB.mnv4_strided_tiedB
#audit_axioms Mnv4TieB.mnv4_stem_tiedB
#audit_axioms Mnv4TieB.mnv4_fused_tiedB
#audit_axioms Mnv4TieB.mnv4_head_tiedB
#audit_axioms Mnv4TieB.mnv4BodyCotIn_eq_vjp
#audit_axioms Mnv4TieB.mnv4SBodyCotIn_eq_vjp
#audit_axioms Mnv4TieB.mnv4SkipCotIn_eq_vjp
#audit_axioms Mnv4TieB.mnv4FusedCotIn_eq_vjp
#audit_axioms Mnv4TieB.mnv4HeadCotIn_eq_vjp
#audit_axioms Mnv4TieB.mnv4_net_tiedB
#audit_axioms Mnv4TieB.mnv4_lossCot_is_smoothedCE_grad

-- MNv4's T6 -- the certified whole-net input gradient
#audit_axioms Proofs.mnv4StemBBack_eq_vjp_backward
#audit_axioms Proofs.cbReluBBack_eq_vjp_backward
#audit_axioms Proofs.mnv4Hc2BBack_eq_vjp_backward
#audit_axioms Proofs.mnv4ClsBBack_eq_vjp_backward
#audit_axioms Proofs.mnv4BFullHasVJPAt
#audit_axioms Proofs.mnv4InputGradB_eq_mnv4B_full_vjp
#audit_axioms Proofs.mnv4InputGradB_correct
#audit_axioms Proofs.mobilenetv4ForwardBFull_eq_slots

-- EfficientNet — the VJP-without-backward-graph holes, closed
#audit_axioms StableHLO.mbNoExpBackBatchedGraph_faithful
#audit_axioms StableHLO.headBackBatchedGraph_faithful
-- ViT-Tiny FOLD (ViTFold)
#audit_axioms Proofs.SgdNode.veclnGammaSgd_den
#audit_axioms Proofs.ViTFold.rowDenseWeightSgd_den
#audit_axioms Proofs.ViTFold.rowDenseBiasSgd_den
#audit_axioms Proofs.SgdNode.rowDenseBiasSgd_den_lnbeta
#audit_axioms Proofs.ViTFold.patchEmbedWeightSgd_den
#audit_axioms Proofs.ViTFold.patchEmbedBiasSgd_den
#audit_axioms Proofs.ViTFold.posEmbedSgd_den
-- ViT-Tiny TIE — MULTI-HEAD promotion (ViTMultiHeadChain + ViTStepTie)
#audit_axioms Proofs.vitCotDQmh_eq
#audit_axioms Proofs.vitCotDKmh_eq
#audit_axioms Proofs.vitCotDVmh_eq
-- The multi-head per-block tie (vit_block_tiedMHV)
#audit_axioms Proofs.ViTTie.vit_block_tiedMHV
#audit_axioms Proofs.ViTTie.vit_block_tiedAtMHV
-- ViT-Tiny TIE — the ALL-200-PARAMS capstone (vit_net_tied_certified)
#audit_axioms Proofs.ViTTie.vit_cls_den
#audit_axioms Proofs.ViTTie.vit_finalLN_tied
#audit_axioms Proofs.ViTTie.vit_head_tied
#audit_axioms Proofs.ViTTie.vit_embed_tied
#audit_axioms Proofs.ViTTie.vit_net_tied_certified

-- Robustness certificate (the cert side of cert ≤ TRUE ≤ PGD)
#audit_axioms Proofs.lipschitz_margin_certified_radius
#audit_axioms Proofs.logit_gap_stable
#audit_axioms Proofs.coord_pair_bound
#audit_axioms Proofs.lipschitzL2_iff_lipschitzWith
#audit_axioms Proofs.LipschitzL2.mono
#audit_axioms Proofs.LipschitzL2.comp
#audit_axioms Proofs.clm_lipschitzL2

-- Randomized-smoothing certified radius (Cohen–Rosenfeld–Kolter 2019)
#audit_axioms Proofs.smoothed_margin_certified_radius

-- The smoothing radius at the REAL Gaussian quantile (Smoothing/Gaussian.lean)
#audit_axioms Proofs.smoothing_certified_radius_probit
#audit_axioms Proofs.stdNormalCDF_strictMono
#audit_axioms Proofs.stdNormalCDF_neg
#audit_axioms Proofs.stdNormalQuantile_monotoneOn
#audit_axioms Proofs.stdNormalQuantile_anti
#audit_axioms Proofs.smoothing_certified_radius_gaussian

-- The 1-D Neyman–Pearson core
#audit_axioms Proofs.stdNormalCDF_quantile

-- Dimension reduction
#audit_axioms MathlibUpstream.integral_gaussianReal_comp_add_const
#audit_axioms Proofs.pi_gaussian_shift_eq
#audit_axioms Proofs.pi_gaussian_np_shift
#audit_axioms Proofs.stdGaussian_np_shift

-- Assembly — the Cohen radius with NOTHING left on the smoothing side
#audit_axioms Proofs.stdNormalQuantile_cdf
#audit_axioms Proofs.stdNormalCDF_mem_Ioo
#audit_axioms Proofs.smoothing_probit_lipschitz
#audit_axioms Proofs.smoothing_certified_radius_cohen
#audit_axioms Proofs.smoothing_certified_radius_classifier
-- ...and the MONTE-CARLO tie (Smoothing/MC.lean)
#audit_axioms Proofs.mc_mean_lower_bound
#audit_axioms Proofs.stdNormalQuantile_of_nonpos
#audit_axioms Proofs.smoothing_mc_certified

-- ...and the EXACT Clopper-Pearson tie (Smoothing/CP.lean)
#audit_axioms Proofs.pi_hitCount_eq_binomial
#audit_axioms Proofs.pi_hitCount_tail_real
#audit_axioms Proofs.binomTail_le_of_lt_cpLower
#audit_axioms Proofs.cp_coverage
#audit_axioms Proofs.smoothing_cp_certified

-- ...and the SOLVED form (the per-image scorecard shape)
#audit_axioms Proofs.binomTail_monotoneOn
#audit_axioms Proofs.le_cpLower_of_tail_le
#audit_axioms Proofs.smoothing_cp_certified_solved

-- ...and the KERNEL ENGINE for driver-scale tail checks (the ListDot recipe)
#audit_axioms Proofs.binomTailNum_eq
#audit_axioms Proofs.binomTailGo_eq
#audit_axioms Proofs.binomTailNumFast_eq
#audit_axioms Proofs.binomTail_eq_kernel
#audit_axioms Proofs.binomTail_le_of_kernel_check

-- ...and the SCORECARD (Smoothing/CPScorecard.lean, generated)
#audit_axioms Proofs.smoothCpMlp_tail_le
#audit_axioms Proofs.smoothCpCnn_tail_le
#audit_axioms Proofs.smoothCpCifar_tail_le

-- ...and certified DECIMAL quantile bounds (Smoothing/PhiBounds.lean)
#audit_axioms Proofs.stdNormalCDF_panel
#audit_axioms Proofs.stdNormalCDF_le_phiGridUB
#audit_axioms Proofs.le_stdNormalQuantile_of_grid

-- ...and the DECIMAL-radius SCORECARD (the prefix-scan corpus pass)
#audit_axioms Proofs.phiScanRev_getD
#audit_axioms Proofs.phiScanRevFrom_append
#audit_axioms Proofs.le_stdNormalQuantile_of_scan
#audit_axioms Proofs.smooth_radius_dec
#audit_axioms Proofs.smoothDecMlp_radius_le
#audit_axioms Proofs.smoothDecCnn_radius_le
#audit_axioms Proofs.smoothDecCifar_radius_le

-- ...and the NET-SEMANTICS closure (Smoothing/NetSemantics.lean)
#audit_axioms Proofs.measurable_argmaxNet
#audit_axioms Proofs.argmaxNet_smoothProb_mem_Ioo
#audit_axioms Proofs.smoothing_cp_certified_net
#audit_axioms Proofs.Robustness.mlpT_logit_continuous
#audit_axioms Proofs.Robustness.netW_strict
#audit_axioms Proofs.Robustness.smoothing_cp_certified_mlpT
#audit_axioms Proofs.Robustness.smooth_cp_mlpT_demo

-- ...and the two-sided quantile packaging (Smoothing/Gaussian.lean)
#audit_axioms Proofs.stdNormalQuantile_strictMonoOn
#audit_axioms Proofs.stdNormalQuantile_surjOn
#audit_axioms Proofs.stdNormalQuantile_continuousAt
#audit_axioms Proofs.stdNormalQuantile_continuousOn

-- The Mathlib upstreaming drafts (UpstreamDraft.lean, a Certs root)
#audit_axioms MathlibUpstream.strictMono_cdf_iff
#audit_axioms MathlibUpstream.continuous_cdf_iff
#audit_axioms MathlibUpstream.cdf_mem_Ioo
#audit_axioms MathlibUpstream.cdf_gaussianReal_mem_Ioo
#audit_axioms MathlibUpstream.cdf_gaussianReal_neg
#audit_axioms MathlibUpstream.cdf_gaussianReal_sub_const

-- ...and the Tsuzuku certificate INSTANTIATED (LipschitzCert/Instance.lean)
#audit_axioms Proofs.Robustness.denseE_lipschitzL2
#audit_axioms Proofs.Robustness.reluE_lipschitzL2
#audit_axioms Proofs.Robustness.linear_demo_certified
#audit_axioms Proofs.Robustness.linear_radius_pos
#audit_axioms Proofs.Robustness.mlp_lip
#audit_axioms Proofs.Robustness.mlp_demo_certified
#audit_axioms Proofs.Robustness.mlp_radius_pos
#audit_axioms Proofs.Robustness.W1t_lip
#audit_axioms Proofs.Robustness.W2t_lip
#audit_axioms Proofs.Robustness.mlpT_lip
#audit_axioms Proofs.Robustness.xt_margin
#audit_axioms Proofs.Robustness.trained_radius_pos
#audit_axioms Proofs.Robustness.trained_demo_certified
-- ...tightened by the power-iteration certificate (certified two-sided spectral sandwich)
#audit_axioms Proofs.Robustness.sum_sq_matvec_le
#audit_axioms Proofs.Robustness.denseE_lipschitzL2_gram
#audit_axioms Proofs.Robustness.lipschitzL2_lower_euclid
#audit_axioms Proofs.Robustness.G1t_eq
#audit_axioms Proofs.Robustness.G2t_eq
#audit_axioms Proofs.Robustness.W1t_lip_gram
#audit_axioms Proofs.Robustness.W2t_lip_gram
#audit_axioms Proofs.Robustness.mlpT_lip_gram
#audit_axioms Proofs.Robustness.trained_radius_gram_pos
#audit_axioms Proofs.Robustness.trained_demo_certified_gram
#audit_axioms Proofs.Robustness.W1t_lip_lower
#audit_axioms Proofs.Robustness.W2t_lip_lower
-- ...iterated once more (Schatten-8)
#audit_axioms Proofs.Robustness.sum_sq_matTvec_eq
#audit_axioms Proofs.Robustness.denseE_lipschitzL2_gram2
#audit_axioms Proofs.Robustness.H1t_eq
#audit_axioms Proofs.Robustness.H2t_eq
#audit_axioms Proofs.Robustness.W1t_lip_gram2
#audit_axioms Proofs.Robustness.W2t_lip_gram2
#audit_axioms Proofs.Robustness.mlpT_lip_gram2
#audit_axioms Proofs.Robustness.trained_radius_gram2_pos
#audit_axioms Proofs.Robustness.trained_demo_certified_gram2

-- CERTIFIED-ACCURACY SCORECARD (LipschitzCert/Scorecard.lean)
#audit_axioms Proofs.Robustness.sqrt_two_le_rat
#audit_axioms Proofs.Robustness.certified_at_eps
#audit_axioms Proofs.Robustness.G1s_eq
#audit_axioms Proofs.Robustness.G2s_eq
#audit_axioms Proofs.Robustness.H1s_eq
#audit_axioms Proofs.Robustness.H2s_eq
#audit_axioms Proofs.Robustness.W1s_lip_gram2
#audit_axioms Proofs.Robustness.W2s_lip_gram2
#audit_axioms Proofs.Robustness.mlpS_lip_gram2
#audit_axioms Proofs.Robustness.marginC0
#audit_axioms Proofs.Robustness.certifiedC0
#audit_axioms Proofs.Robustness.certifiedC3
#audit_axioms Proofs.Robustness.certifiedC10
#audit_axioms Proofs.Robustness.certifiedC13
#audit_axioms Proofs.Robustness.certifiedC14
#audit_axioms Proofs.Robustness.certifiedC17
#audit_axioms Proofs.Robustness.certifiedC25
#audit_axioms Proofs.Robustness.marginU82
#audit_axioms Proofs.Robustness.certifiedU82
-- the mechanized aggregate
#audit_axioms Proofs.Robustness.cappedCerts_certified
#audit_axioms Proofs.Robustness.unconCerts_certified
#audit_axioms Proofs.Robustness.scorecard

-- Per-pair LipSDP tightening (LipschitzCert/PairSDP.lean + the generated instances)
#audit_axioms Proofs.Robustness.relu_slope_restricted
#audit_axioms Proofs.Robustness.pair_sq_bound
#audit_axioms Proofs.Robustness.mlp_gap_eq
#audit_axioms Proofs.Robustness.certified_at_eps_pair
#audit_axioms Proofs.Robustness.hS01C
#audit_axioms Proofs.Robustness.pairSqC_0_1
#audit_axioms Proofs.Robustness.pairSqC_7_0
#audit_axioms Proofs.Robustness.pairSqU_0_1
#audit_axioms Proofs.Robustness.certifiedSC0
#audit_axioms Proofs.Robustness.certifiedSC4
#audit_axioms Proofs.Robustness.certifiedSC9
#audit_axioms Proofs.Robustness.certifiedSU0
#audit_axioms Proofs.Robustness.certifiedSU4
#audit_axioms Proofs.Robustness.certifiedSU11
#audit_axioms Proofs.Robustness.sdpCappedCerts_certified
#audit_axioms Proofs.Robustness.sdpUnconCerts_certified
#audit_axioms Proofs.Robustness.scorecard_sdp
#audit_axioms Proofs.Robustness.scorecard_sdp_uncon

-- The certificate × float bridge (LipschitzCert/Float.lean)
#audit_axioms Proofs.FloatModel.mlp2_float_close_uniform
#audit_axioms Proofs.Robustness.certified_at_eps_close
#audit_axioms Proofs.Robustness.capped_B_le
#audit_axioms Proofs.Robustness.real_tie
#audit_axioms Proofs.Robustness.certifiedFloat_of_margin
#audit_axioms Proofs.Robustness.certifiedC0_float

-- The kernel-dotZ list engine (ListDot.lean)
#audit_axioms Proofs.dotZ_comm
#audit_axioms Proofs.sum_getD_mul
#audit_axioms Proofs.sum_getD_div
#audit_axioms Proofs.sum_getD_abs
#audit_axioms Proofs.sum_getD_abs_div
#audit_axioms Proofs.Robustness.denseLo_le
#audit_axioms Proofs.Robustness.le_denseHi
#audit_axioms Proofs.Robustness.relu_box
#audit_axioms Proofs.Robustness.denseLo_uniform
#audit_axioms Proofs.Robustness.denseLo2_eval
#audit_axioms Proofs.Robustness.denseHi2_eval
-- The dense tier's certificate is stated on a BRACKET, not on interval arithmetic
#audit_axioms Proofs.Robustness.BoxSoundE.comp
#audit_axioms Proofs.Robustness.denseE_boxSound
#audit_axioms Proofs.Robustness.reluE_boxSound
#audit_axioms Proofs.Robustness.certified_of_boxSound
#audit_axioms Proofs.Robustness.mlp2_boxSound
#audit_axioms Proofs.Robustness.ibp2_certified_at_eps

-- CROWN (Certificates/CrownBound.lean)
#audit_axioms Proofs.Robustness.certified_of_marginPos
#audit_axioms Proofs.Robustness.relu_lower_envelope
#audit_axioms Proofs.Robustness.relu_upper_envelope
#audit_axioms Proofs.Robustness.reluLB_dead
#audit_axioms Proofs.Robustness.reluLB_active
#audit_axioms Proofs.Robustness.reluLB_unstable_pos
#audit_axioms Proofs.Robustness.reluLB_unstable_neg
#audit_axioms Proofs.Robustness.crownRow_dot
#audit_axioms Proofs.Robustness.linf_lower_bound
#audit_axioms Proofs.Robustness.crown_margin_ge
#audit_axioms Proofs.Robustness.crown2_certified_at_eps

-- IBP PAST THE TWO-LAYER DENSE WALL (Certificates/IntervalBoundConv/Basic.lean)
#audit_axioms Proofs.IBP.BoxSound3.comp
#audit_axioms Proofs.IBP.BoxSound3V.comp3
#audit_axioms Proofs.IBP.denseT_boxSound3V
#audit_axioms Proofs.IBP.reluT_boxSound3
#audit_axioms Proofs.IBP.conv2d_boxSound3
#audit_axioms Proofs.IBP.maxPool2_boxSound3
-- the conv peer of denseLo_uniform
#audit_axioms Proofs.IBP.convLo_uniform
#audit_axioms Proofs.IBP.convHi_uniform
-- DEPTH, concretely
#audit_axioms Proofs.IBP.deepNet_boxSound
-- capstone + radius monotonicity
#audit_axioms Proofs.IBP.ibp3_certified_of_boxSound
#audit_axioms Proofs.IBP.CertifiedAtLinf3.mono

-- THE IEEE AXIOMS, DISCHARGED (Binary32Instance.lean)
#audit_axioms Proofs.rndP_err
#audit_axioms Proofs.binary32_e4m3_argmax_preserved
#audit_axioms Proofs.binary32_e4m3_argmax_small
#audit_axioms Proofs.binary32_linear_sgd_descends_concrete

-- DESCENT AT TRAINED WEIGHTS (Trained/LinearDescent.lean)
#audit_axioms Proofs.TrainedLinearDescent.hz_lbl_le
#audit_axioms Proofs.TrainedLinearDescent.sm_lbl_le_half
#audit_axioms Proofs.TrainedLinearDescent.gradL1_le
#audit_axioms Proofs.TrainedLinearDescent.gradSq_lower
#audit_axioms Proofs.TrainedLinearDescent.hdelta
#audit_axioms Proofs.TrainedLinearDescent.heta
#audit_axioms Proofs.TrainedLinearDescent.trained_linear_sgd_descends_concrete
#audit_axioms Proofs.TrainedLinearDescent.trained_linear_sgd_strictly_descends

-- TRAINED-WEIGHT whole-net VJP witness (Trained/MlpWitness.lean)
#audit_axioms Proofs.TrainedMlp.preact_eq
#audit_axioms Proofs.TrainedMlp.preact_ne
#audit_axioms Proofs.TrainedMlp.pdiv_fwd
#audit_axioms Proofs.TrainedMlp.pdiv_fwd_val
#audit_axioms Proofs.TrainedMlp.trainedMlp_backward_nontrivial
#audit_axioms Proofs.TrainedMlp.trainedMlp_jacobian_nonzero
#audit_axioms Proofs.TrainedMlp.trainedMlp_not_constant

-- TRAINED-WEIGHT whole-net VJP witness, CNN rung (Trained/CnnWitness.lean)
#audit_axioms Proofs.TrainedCnn.conv1_eq
#audit_axioms Proofs.TrainedCnn.conv2_eq
#audit_axioms Proofs.TrainedCnn.r2_smooth
#audit_axioms Proofs.TrainedCnn.pooled_eq
#audit_axioms Proofs.TrainedCnn.d3_ne
#audit_axioms Proofs.TrainedCnn.d4_ne
#audit_axioms Proofs.TrainedCnn.trainedCnnHasVJP_correct

-- CNN descent at trained weights through tied pool windows (Trained/CnnDescent.lean)
#audit_axioms Proofs.TrainedCnnDescent.pool_margin
#audit_axioms Proofs.TrainedCnnDescent.x1_eq
#audit_axioms Proofs.TrainedCnnDescent.trained_cnn_conv2_sgd_descends_concrete
#audit_axioms Proofs.TrainedCnnDescent.trained_cnn_conv2_bias_sgd_descends_concrete
-- CNN conv1 descent at trained weights through two-layer twins (Trained/CnnDescentConv1.lean)
#audit_axioms Proofs.TrainedCnnDescentConv1.pool_margin
#audit_axioms Proofs.TrainedCnnDescentConv1.x1_fun
#audit_axioms Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_sgd_descends_concrete
#audit_axioms Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_bias_sgd_descends_concrete
-- Level-3 seal for the CNN witness (Trained/CnnSeal.lean)
#audit_axioms Proofs.TrainedCnn.S2
#audit_axioms Proofs.TrainedCnn.S1
#audit_axioms Proofs.TrainedCnn.pdiv_fwd_entry
#audit_axioms Proofs.TrainedCnn.pdiv_fwd_entry_ne
#audit_axioms Proofs.TrainedCnn.trainedCnn_backward_nontrivial
#audit_axioms Proofs.TrainedCnn.trainedCnn_jacobian_nonzero
#audit_axioms Proofs.TrainedCnn.trainedCnn_not_constant

-- Muon geometry
#audit_axioms Proofs.MuonGeometry.steepest_l2_bound
#audit_axioms Proofs.MuonGeometry.steepest_l2_attained
#audit_axioms Proofs.MuonGeometry.steepest_linf_bound
#audit_axioms Proofs.MuonGeometry.steepest_linf_attained
#audit_axioms Proofs.MuonGeometry.muon_polar_achieves_nuclear
-- L3 upper bound (von Neumann's trace inequality)
#audit_axioms Proofs.MuonGeometry.muon_polar_is_max
#audit_axioms Proofs.MuonGeometry.muon_polar_steepest
#audit_axioms Proofs.MuonGeometry.svd_of_isUnit
#audit_axioms Proofs.MuonGeometry.muon_polar_achieves_nuclear_of_isUnit
-- L5 the jewel: single-step Shampoo (GGᵀ)
#audit_axioms Proofs.MuonGeometry.conj_diag_pow
#audit_axioms Proofs.MuonGeometry.shampoo_eq_muon
#audit_axioms Proofs.MuonGeometry.shampoo_eq_muon_of_isUnit
-- L6 manifold view
#audit_axioms Proofs.MuonGeometry.muon_polar_orthogonal
#audit_axioms Proofs.MuonGeometry.muon_polar_nearest_orthogonal
-- Newton–Schulz
#audit_axioms Proofs.MuonNewtonSchulz.nsStep_spectral
#audit_axioms Proofs.MuonNewtonSchulz.nsStep_iterate_spectral
-- Newton–Schulz (the scalar engine)
#audit_axioms Proofs.MuonNewtonSchulz.scalar_iterate_tendsto_one
#audit_axioms Proofs.MuonNewtonSchulz.gCubic_eq_nsScalar
#audit_axioms Proofs.MuonNewtonSchulz.gCubic_iterate_tendsto_one
#audit_axioms Proofs.MuonNewtonSchulz.q5Scalar_eq_nsScalar
#audit_axioms Proofs.MuonNewtonSchulz.q5Scalar_iterate_tendsto_one
-- Newton–Schulz (CLOSES THE LOOP)
#audit_axioms Proofs.MuonNewtonSchulz.nsStep_iterate_tendsto_polar
#audit_axioms Proofs.MuonNewtonSchulz.nsStep_cubic_iterate_tendsto_polar
#audit_axioms Proofs.MuonNewtonSchulz.nsStep_q5_iterate_tendsto_polar
-- Newton–Schulz (the HONEST tier)
#audit_axioms Proofs.MuonNewtonSchulz.qScalar_one_lt_one
#audit_axioms Proofs.MuonNewtonSchulz.qScalar_half_gt_one
#audit_axioms Proofs.MuonNewtonSchulz.qScalar_not_le_one
#audit_axioms Proofs.MuonNewtonSchulz.qScalar_iterate_not_tendsto_one
#audit_axioms Proofs.MuonNewtonSchulz.qScalar_not_hconv
#audit_axioms Proofs.MuonNewtonSchulz.qScalar_iterate_band_half

-- Spec→math ties (SpecVJP.lean)
#audit_axioms linearVerified_denote_eq
#audit_axioms linearVerified_fwd_faithful
#audit_axioms linearVerified_lossCot_isCEgrad
#audit_axioms mlpVerified_denote_eq
#audit_axioms mlpVerified_fwd_faithful
#audit_axioms mlpVerified_back_faithful
#audit_axioms cnnVerified_denote_eq
#audit_axioms cnnVerified_fwd_faithful
#audit_axioms cifarVerified_denote_eq
#audit_axioms cifarVerified_fwd_faithful
-- mnv2 committed-spec tie, at batch BN — the net every shipped MobileNetV2 artifact runs
#audit_axioms mobilenetv2VerifiedB_denote_eq
#audit_axioms mobilenetv2VerifiedB_fwd_faithful
-- FULL committed-spec ties (unified weight bundles); r34's at batch BN, the net every shipped artifact runs
#audit_axioms resnet34VerifiedB_denote_eq
#audit_axioms resnet34VerifiedB_fwd_faithful
#audit_axioms efficientnetVerified_denote_eq
#audit_axioms efficientnetVerified_fwd_faithful
#audit_axioms convnextVerified_denote_eq
#audit_axioms convnextVerified_fwd_faithful
#audit_axioms vitVerified_denote_eq
#audit_axioms vitVerifiedHasVJP
#audit_axioms vitVerified_fwd_faithful

-- The CANONICAL MNIST MLP surface (MlpCanonical.lean)
#audit_axioms Proofs.MlpCanonical.hasVJPAt
#audit_axioms Proofs.MlpCanonical.output_float_sgd_descends
#audit_axioms Proofs.MlpCanonical.hidden_float_sgd_descends
#audit_axioms Proofs.MlpCanonical.input_float_sgd_descends
#audit_axioms Proofs.MlpCanonical.w1_grad_close
#audit_axioms Proofs.MlpCanonical.w0_grad_close
#audit_axioms Proofs.MlpCanonical.train_step_tied_certified

-- The bf16-MIXED conv, composed (ConvMixedComposeBridge.lean)
#audit_axioms Proofs.convFanS_le
#audit_axioms Proofs.conv2d_sub_abs_le
#audit_axioms Proofs.FloatModel.convMixed_close_prop
#audit_axioms Proofs.FloatModel.flatConvMixed_close

-- The bf16-mixed DEPTHWISE (DepthwiseMixedFloatBridge.lean)
#audit_axioms Proofs.depthwiseConv2d_eq_dw_dot
#audit_axioms Proofs.FloatModel.depthwise_close_mixed

-- RESNET-34 AT TRUE BATCH BN — T1-forward and T2 (ResNet34FullB.lean)
#audit_axioms Proofs.resnet34ForwardBFull
#audit_axioms Proofs.StableHLO.r34IdGraphB_faithful
#audit_axioms Proofs.StableHLO.r34DownGraphB_faithful
#audit_axioms Proofs.StableHLO.r34StemGraphB_faithful
#audit_axioms Proofs.StableHLO.r34HeadGraphB_faithful
#audit_axioms Proofs.StableHLO.resnet34FwdGraphBFull_faithful

-- `batchMap` AT A POINT (BatchMapVJPAt.lean)
#audit_axioms Proofs.pdivMat_rowIndep_at
#audit_axioms Proofs.batchMap_differentiableAt
#audit_axioms Proofs.pdiv_batchMap_at
#audit_axioms Proofs.batchMapHasVJPAt
#audit_axioms Proofs.batchMapAux_eq_batchMapHasVJPAt
#audit_axioms Proofs.batchMap_eq_batchMapHasVJPAt
#audit_axioms Proofs.batchMap_comp
#audit_axioms Proofs.HasVJPAt.backward_unique_of_eq

-- RESNET-34 AT TRUE BATCH BN — T1's VJP half (ResNet34FullBVJP.lean)
#audit_axioms Proofs.r34IdBHasVJPAt
#audit_axioms Proofs.r34DownBHasVJPAt
#audit_axioms Proofs.r34StemBHasVJPAt
#audit_axioms Proofs.stemReluPoolLayer
#audit_axioms Proofs.maxPool3s2Flat_relu_eventuallyEq
#audit_axioms Proofs.maxPool3s2FlatBackB_eq_reindex
#audit_axioms Proofs.gatherReluHasVJPAt
#audit_axioms Proofs.ResNet34TieB.r34StemPool_param_germ
#audit_axioms Proofs.ResNet34TieB.r34StemGCg_hasGradAt
#audit_axioms Proofs.stemPoolRelu_param_eventuallyEq_select
#audit_axioms Proofs.maxPool3s2LocalReindexB_isSelect
#audit_axioms Proofs.ResNet34TieB.r34StemPool_param_germ_select
#audit_axioms Proofs.ResNet34TieB.r34StemGCg_hasGradAt_at
#audit_axioms Proofs.ResNet34TieB.r34_stem_lossTiedB_select
#audit_axioms Proofs.ResNet34TieB.r34Stem_select_grads_eq
#audit_axioms Proofs.ResNet34TieB.r34StemLossTiedB.select
#audit_axioms Proofs.ResNet34TieB.r34_net_lossGrad_stemSelect
#audit_axioms Proofs.ResNet50TieB.r50_net_lossGrad_stemSelect
#audit_axioms Proofs.ResNet34TieB.r34LossSmoothAtB_of_smoothAtB
#audit_axioms Proofs.ResNet50TieB.r50LossSmoothAtB_of_smoothAtB
#audit_axioms Proofs.BatchSeal.bnBatchLA_bcell_eq_of_eq
#audit_axioms Proofs.r34StemLayer
#audit_axioms Proofs.r34HeadLayer
#audit_axioms Proofs.r34HeadBHasVJP
#audit_axioms Proofs.r34NetLayer
#audit_axioms Proofs.r34NetLayer_fwd_apply
#audit_axioms Proofs.r34SmoothAtB_ok
#audit_axioms Proofs.resnet34ForwardBFullHasVJPAt
#audit_axioms Proofs.resnet34ForwardBFull_eq_chain
#audit_axioms Proofs.resnet34ForwardBFullHasVJPAt_correct
#audit_axioms Proofs.resnet34ForwardBFull_differentiableAt

-- RESNET-34 AT TRUE BATCH BN — T3's fold, UN-FUSED (Foundation/GradNodesB.lean)
#audit_axioms Proofs.GradNodeB.convWGradB_den
#audit_axioms Proofs.GradNodeB.convBGradB_den
#audit_axioms Proofs.GradNodeB.convStridedWGradB_den
#audit_axioms Proofs.GradNodeB.convStridedBGradB_den
#audit_axioms Proofs.GradNodeB.bnGammaGradB_den
#audit_axioms Proofs.GradNodeB.bnBetaGradB_den
#audit_axioms Proofs.GradNodeB.denseWGradB_den
#audit_axioms Proofs.GradNodeB.denseBGradB_den

-- EfficientNet-B0
#audit_axioms Proofs.GradNodeB.denseBGradB_den
#audit_axioms Proofs.GradNodeB.convStridedXlaWGradB_den
#audit_axioms Proofs.GradNodeB.depthwiseWGradB_den
#audit_axioms Proofs.GradNodeB.depthwiseStridedWGradB_den

-- ConvNeXt-T
#audit_axioms Proofs.CnxFoldG.layerScaleChGammaGrad_den
#audit_axioms Proofs.CnxFoldG.chanLnGammaGrad_den
#audit_axioms Proofs.CnxFoldG.chanLnBetaGrad_den

-- ViT-Tiny — vit_adam_train_step and vitin_adamdp128x4wxclipdrop
#audit_axioms Proofs.ViTFoldG.posEmbedGrad_den
#audit_axioms Proofs.ViTFoldG.clsGrad_den

-- ViT-Tiny
#audit_axioms Proofs.GradNodeB.veclnGammaGradB_den
#audit_axioms Proofs.GradNodeB.rowDenseBiasGradB_den_lnbeta
#audit_axioms Proofs.ViTFoldGB.rowDenseWeightGradB_den
#audit_axioms Proofs.ViTFoldGB.rowDenseBiasGradB_den
#audit_axioms Proofs.ViTFoldGB.patchEmbedWeightGradB_den
#audit_axioms Proofs.ViTFoldGB.patchEmbedBiasGradB_den
#audit_axioms Proofs.ViTFoldGB.posEmbedGradB_den
#audit_axioms Proofs.ViTFoldGB.clsGrad_denB
#audit_axioms Proofs.GradNodeB.headWGradB_den
#audit_axioms Proofs.GradNodeB.headBGradB_den

-- ConvNeXt-T
#audit_axioms Proofs.CnxFoldGB.layerScaleChGammaGradB_den
#audit_axioms Proofs.GradNodeB.psWGradB_den
#audit_axioms Proofs.CnxFoldGB.chanLnGammaGradB_den
#audit_axioms Proofs.CnxFoldGB.chanLnBetaGradB_den
-- The bf16 gradient nodes, folded ONCE for every net (Bf16GradNodes.lean)
#audit_axioms Proofs.Bf16Fold.convWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.convStridedWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.convStridedXlaWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.convStride4WGradBBf16_den
#audit_axioms Proofs.Bf16Fold.depthwiseWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.depthwiseStridedWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.depthwiseStridedXlaWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.rowDenseWGradBBf16_den
#audit_axioms Proofs.Bf16Fold.patchEmbedWGradBBf16_den
-- Erasure: every batched bf16 kind is its f32 peer at the identity rounding (Bf16Erasure.lean)
#audit_axioms Proofs.Bf16Fold.flatConvStride2_bias_split
#audit_axioms Proofs.Bf16Fold.depthwiseStride2FlatXla_bias_split
#audit_axioms Proofs.Bf16Fold.den_batchOp_congr
#audit_axioms Proofs.Bf16Fold.convBf16_id
#audit_axioms Proofs.Bf16Fold.convStridedBf16_id
#audit_axioms Proofs.Bf16Fold.convStridedXlaBf16_id
#audit_axioms Proofs.Bf16Fold.convStride4Bf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseStridedBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseStridedXlaBf16_id
#audit_axioms Proofs.Bf16Fold.denseRowBf16_id
#audit_axioms Proofs.Bf16Fold.patchEmbedBf16_id
#audit_axioms Proofs.Bf16Fold.denseRowBackBf16_id
#audit_axioms Proofs.Bf16Fold.matmulFBBf16_id
#audit_axioms Proofs.Bf16Fold.convBackBatchedBf16_id
#audit_axioms Proofs.Bf16Fold.convStridedBackBatchedBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseBackBatchedBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseStridedBackBatchedBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseStridedXlaBackBatchedBf16_id
#audit_axioms Proofs.Bf16Fold.convWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.convStridedWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.convStridedXlaWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.convStride4WeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseStridedWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.depthwiseStridedXlaWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.rowDenseWeightGradBBf16_id
#audit_axioms Proofs.Bf16Fold.patchEmbedWeightGradBBf16_id
-- …and the renderers' switches at `id`, either `bf16` (StableHLO/PrecisionSwitch.lean)
#audit_axioms Proofs.Bf16Fold.denOp_convAt_id
#audit_axioms Proofs.Bf16Fold.denOp_convStridedAt_id
#audit_axioms Proofs.Bf16Fold.denOp_convStridedXlaAt_id
#audit_axioms Proofs.Bf16Fold.denOp_convStride4At_id
#audit_axioms Proofs.Bf16Fold.denOp_depthwiseAt_id
#audit_axioms Proofs.Bf16Fold.denOp_depthwiseStridedAt_id
#audit_axioms Proofs.Bf16Fold.denOp_depthwiseStridedXlaAt_id
#audit_axioms Proofs.Bf16Fold.denOp_denseRowAt_id
#audit_axioms Proofs.Bf16Fold.denOp_patchEmbedAt_id
#audit_axioms Proofs.Bf16Fold.denOp_denseRowBackAt_id
#audit_axioms Proofs.Bf16Fold.den_flatConvFAt_id
#audit_axioms Proofs.Bf16Fold.den_matmulFBAt_id
#audit_axioms Proofs.Bf16Fold.den_convBackBatchedAt_id
#audit_axioms Proofs.Bf16Fold.den_convStridedBackBatchedAt_id
#audit_axioms Proofs.Bf16Fold.den_depthwiseBackBatchedAt_id
#audit_axioms Proofs.Bf16Fold.den_depthwiseStridedBackBatchedAt_id
#audit_axioms Proofs.Bf16Fold.den_depthwiseStridedXlaBackBatchedAt_id
#audit_axioms Proofs.Bf16Fold.den_convWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_convStridedWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_convStridedXlaWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_convStride4WeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_depthwiseWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_depthwiseStridedWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_depthwiseStridedXlaWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_rowDenseWeightGradBAt_id
#audit_axioms Proofs.Bf16Fold.den_patchEmbedWeightGradBAt_id
-- The tie predicates on the switch, either bf16 (GradNodesBAt.lean): node, loss derivative, collective
#audit_axioms Proofs.GradNodeB.convWTiedBAt_holds
#audit_axioms Proofs.GradNodeB.convStridedWTiedBAt_holds
#audit_axioms Proofs.GradNodeB.convWAt_hasGradAt
#audit_axioms Proofs.GradNodeB.convStridedWAt_hasGradAt
#audit_axioms Proofs.SyncKit.convWSyncAt_of_scaled
#audit_axioms Proofs.SyncKit.convStridedWSyncAt_of_scaled
-- …the depthwise nets' kinds: XLA-SAME stem, depthwise, symmetric and XLA-SAME strided depthwise
#audit_axioms Proofs.GradNodeB.convStridedXlaWTiedBAt_holds
#audit_axioms Proofs.GradNodeB.depthwiseWTiedBAt_holds
#audit_axioms Proofs.GradNodeB.depthwiseStridedWTiedBAt_holds
#audit_axioms Proofs.GradNodeB.depthwiseStridedXlaWTiedBAt_holds
#audit_axioms Proofs.GradNodeB.convStridedXlaWAt_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseWAt_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseStridedWAt_hasGradAt
#audit_axioms Proofs.GradNodeB.depthwiseStridedXlaWAt_hasGradAt
#audit_axioms Proofs.SyncKit.convStridedXlaWSyncAt_of_scaled
#audit_axioms Proofs.SyncKit.depthwiseWSyncAt_of_scaled
#audit_axioms Proofs.SyncKit.depthwiseStridedWSyncAt_of_scaled

-- MobileNetV2 at 17 blocks
#audit_axioms Proofs.GradNodeB.convStridedXlaBGradB_den
#audit_axioms Proofs.GradNodeB.depthwiseBGradB_den
#audit_axioms Proofs.GradNodeB.depthwiseStridedXlaWGradB_den
#audit_axioms Proofs.GradNodeB.depthwiseStridedXlaBGradB_den

-- THE LABEL-SMOOTHED LOSS COTANGENT, AT A GENERAL TARGET (SmoothedLossCot.lean)
#audit_axioms Proofs.softCE_oneHot
#audit_axioms Proofs.softCE_grad
#audit_axioms Proofs.smoothTarget_sum
#audit_axioms Proofs.smoothedCE_grad
#audit_axioms Proofs.smoothedLossCotGraph_den
#audit_axioms Proofs.smoothedLossCotGraph_row

-- RESNET-34'S T3 TIE AT BATCH BN, UN-FUSED (ResNet34StepTieB.lean)
#audit_axioms Proofs.BackLinks.bnInB_eq_bnBackB
#audit_axioms Proofs.BackLinks.bnInB_eq_den_bnBatchBack
#audit_axioms Proofs.ResNet34TieB.r34IdCotIn_eq_vjp
#audit_axioms Proofs.ResNet34TieB.r34DownCotIn_eq_vjp
#audit_axioms Proofs.ResNet34TieB.r34_idblock_tiedB
#audit_axioms Proofs.ResNet34TieB.r34_downblock_tiedB
#audit_axioms Proofs.ResNet34TieB.r34_stem_tiedB
#audit_axioms Proofs.ResNet34TieB.r34_head_tiedB
#audit_axioms Proofs.ResNet34TieB.r34_net_tiedB
#audit_axioms Proofs.ResNet34TieB.r34_lossCot_is_smoothedCE_grad

-- MOBILENETV2 AT TRUE BATCH BN — T1-forward and T2 (MobileNetV2FullB.lean)
#audit_axioms Proofs.mobilenetv2ForwardBFull
#audit_axioms Proofs.StableHLO.mnv2StemGraphB_faithful
#audit_axioms Proofs.StableHLO.mnv2NoExpGraphB_faithful
#audit_axioms Proofs.StableHLO.mnv2ExpOnlyGraphB_faithful
#audit_axioms Proofs.StableHLO.mnv2ResidGraphB_faithful
#audit_axioms Proofs.StableHLO.mnv2StridedGraphB_faithful
#audit_axioms Proofs.StableHLO.mnv2HeadGraphB_faithful
#audit_axioms Proofs.StableHLO.mobilenetv2FwdGraphBFull_faithful

-- MOBILENETV2 AT TRUE BATCH BN — T1's VJP half (MobileNetV2FullBVJP.lean)
#audit_axioms Proofs.mnv2StemBHasVJPAt
#audit_axioms Proofs.mnv2NoExpBHasVJPAt
#audit_axioms Proofs.mnv2ExpOnlyBHasVJPAt
#audit_axioms Proofs.mnv2ResidBHasVJPAt
#audit_axioms Proofs.mnv2StridedBHasVJPAt
#audit_axioms Proofs.mobilenetv2ForwardBFullHasVJPAt
#audit_axioms Proofs.mobilenetv2ForwardBFull_eq_chain
#audit_axioms Proofs.mobilenetv2ForwardBFullHasVJPAt_correct
#audit_axioms Proofs.mobilenetv2ForwardBFull_differentiableAt

-- MOBILENETV2'S T3 TIE AT BATCH BN, UN-FUSED (MobileNetV2StepTieB.lean)
#audit_axioms Proofs.MobileNetV2TieB.mnv2NoExpBackGraph_faithful
#audit_axioms Proofs.MobileNetV2TieB.mnv2NoExpCotIn_eq_vjp
#audit_axioms Proofs.MobileNetV2TieB.mnv2ExpOnlyCotIn_eq_vjp
#audit_axioms Proofs.MobileNetV2TieB.mnv2ResidCotIn_eq_vjp
#audit_axioms Proofs.MobileNetV2TieB.mnv2StridedCotIn_eq_vjp
#audit_axioms Proofs.MobileNetV2TieB.mnv2HeadCotBlk_eq_vjp
#audit_axioms Proofs.MobileNetV2TieB.mnv2StemCotC_eq_vjp
#audit_axioms Proofs.MobileNetV2TieB.mnv2_stem_tiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_noexp_tiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_stride1_tiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_stride2_tiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_head_tiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_net_tiedB
#audit_axioms Proofs.MobileNetV2TieB.mnv2_lossCot_is_smoothedCE_grad

-- CAPSTONE RE-POINTING, EFFICIENTNET-B0 (EfficientNetStepTieG.lean)
#audit_axioms Proofs.EnetTieG.convBBetaTiedB_holds
#audit_axioms Proofs.GradNodeB.convWTiedB_holds
#audit_axioms Proofs.GradNodeB.convBTiedB_holds
#audit_axioms Proofs.GradNodeB.convStridedWTiedB_holds
#audit_axioms Proofs.GradNodeB.convStridedBTiedB_holds
#audit_axioms Proofs.GradNodeB.convStridedXlaWTiedB_holds
#audit_axioms Proofs.GradNodeB.depthwiseWTiedB_holds
#audit_axioms Proofs.GradNodeB.depthwiseBTiedB_holds
#audit_axioms Proofs.GradNodeB.depthwiseStridedWTiedB_holds
#audit_axioms Proofs.GradNodeB.denseWTiedB_holds
#audit_axioms Proofs.GradNodeB.denseBTiedB_holds
#audit_axioms Proofs.ViTFoldGB.rowDenseWTiedB_holds
#audit_axioms Proofs.ViTFoldGB.rowDenseWTiedBAt_holds
#audit_axioms Proofs.ViTFoldGB.rowDenseBTiedB_holds
#audit_axioms Proofs.GradNodeB.vecLNGammaTiedB_holds
#audit_axioms Proofs.GradNodeB.vecLNBetaTiedB_holds
#audit_axioms Proofs.CnxFoldGB.chanLNGammaTiedB_holds
#audit_axioms Proofs.CnxFoldGB.chanLNBetaTiedB_holds
#audit_axioms Proofs.EnetFold.convWSgdTiedB_holds
#audit_axioms Proofs.EnetFold.depthwiseWSgdTiedB_holds
#audit_axioms Proofs.EnetFold.denseWSgdTiedB_holds
#audit_axioms Proofs.EnetFold.denseBSgdTiedB_holds
#audit_axioms Proofs.ViTFold.rowDenseWSgdTied_holds
#audit_axioms Proofs.ViTFold.rowDenseBSgdTied_holds
#audit_axioms Proofs.SgdNode.vecLNGammaSgdTied_holds
#audit_axioms Proofs.SgdNode.vecLNBetaSgdTied_holds
#audit_axioms Proofs.CnxFold.chanLNGammaSgdTied_holds
#audit_axioms Proofs.CnxFold.chanLNBetaSgdTied_holds
#audit_axioms Proofs.convWSgdTied_holds
#audit_axioms Proofs.convBSgdTied_holds
#audit_axioms Proofs.denseWSgdTied_holds
#audit_axioms Proofs.denseBSgdTied_holds
#audit_axioms Proofs.EnetTie.convBBetaSgdTied_holds
#audit_axioms Proofs.EnetTieG.enet_exp_tiedG
#audit_axioms Proofs.EnetTieG.enet_strided_tiedG
#audit_axioms Proofs.EnetTieG.enet_noexp_tiedG
#audit_axioms Proofs.EnetTieG.enet_stem_tiedG
#audit_axioms Proofs.EnetTieG.enet_head_tiedG
#audit_axioms Proofs.EnetTieG.efficientnet_net_tiedG

-- CAPSTONE RE-POINTING, CONVNEXT-T (ConvNeXtStepTieGB.lean)
#audit_axioms Proofs.smoothedLossCotGraphDiv_den
#audit_axioms Proofs.smoothedLossCotGraphDiv_row
#audit_axioms Proofs.CnxTieGB.cnx_block_ch_tiedGB
#audit_axioms Proofs.CnxTieGB.cnx_down_ch_tiedGB
#audit_axioms Proofs.CnxTieGB.cnx_stem_ch_tiedGB
#audit_axioms Proofs.CnxTieGB.cnx_head_ch_tiedGB
#audit_axioms Proofs.CnxTieGB.cnxBlockCotInChAt_eq_vjp
#audit_axioms Proofs.CnxTieGB.cnxDownCotInChAt_eq_vjp
#audit_axioms Proofs.CnxTieGB.cnxHeadHasVJP
#audit_axioms Proofs.CnxTieGB.cnxHeadDyXheadChN_eq_vjp
#audit_axioms Proofs.CnxTieGB.cnxBlockCotInB_eq_vjp
#audit_axioms Proofs.CnxTieGB.cnxDownCotInB_eq_vjp
#audit_axioms Proofs.CnxTieGB.cnxHeadDyB_eq_vjp
#audit_axioms Proofs.CnxTieGB.cnx_net_tiedGB

-- CAPSTONE RE-POINTING, ViT-TINY (ViTStepTieGB.lean)
#audit_axioms Proofs.ViTTieGB.vit_block_tiedGB
#audit_axioms Proofs.ViTTieGB.vit_finalLN_tiedGB
#audit_axioms Proofs.ViTTieGB.vit_head_tiedGB
#audit_axioms Proofs.ViTTieGB.vit_embed_tiedGB
-- per-example cotangent ties (ViTStepTie.lean)
#audit_axioms Proofs.rowDenseBackFlat_eq_perRowFlat
#audit_axioms Proofs.vitCotDQmh_eq_core
#audit_axioms Proofs.vitCotDKmh_eq_core
#audit_axioms Proofs.vitCotDVmh_eq_core
#audit_axioms Proofs.vitCotLn2_eq_perRowFlatPR
#audit_axioms Proofs.vitCotXin_eq_blockBack
#audit_axioms Proofs.vitBlockCotInAtMHV_eq_vjp
#audit_axioms Proofs.vitHeadHasVJP
#audit_axioms Proofs.vitCotTowerOutV_eq_vjp
#audit_axioms Proofs.ViTTieGB.vitBlockCotInB_eq_vjp
#audit_axioms Proofs.ViTTieGB.vitCotTowerOutB_eq_vjp
#audit_axioms Proofs.ViTTieGB.vit_net_tiedGB

-- DATA PARALLELISM -- WHAT FUNCTION A *dp* RUN MINIMISED (DataParallel/Basic.lean)
#audit_axioms Proofs.pdiv_const_smul
#audit_axioms Proofs.meanLoss_differentiableAt
#audit_axioms Proofs.lossGrad_meanLoss
#audit_axioms Proofs.dpMeanGrad_eq_grad_meanLoss
#audit_axioms Proofs.meanLoss_shard
#audit_axioms Proofs.dpMeanGrad_eq_globalBatchGrad_of_perExample
#audit_axioms Proofs.dpMeanGrad_eq_globalBatchGrad_contiguous
#audit_axioms Proofs.dpToyShard_eq_batch
#audit_axioms Proofs.lossGrad_smul_coord
#audit_axioms Proofs.lossGrad_bnToyLoss
#audit_axioms Proofs.dpMeanGrad_ne_globalBatchGrad
#audit_axioms Proofs.dpStep_const
#audit_axioms Proofs.dpIterate_lockstep
#audit_axioms Proofs.dpSingleStep_eq_meanLoss_step
#audit_axioms Proofs.dpIterate_eq_meanLossTrain

-- THE COLLECTIVE AS AN AST NODE (StableHLO.allReduceMeanF, DataParallel/Node.lean)
#audit_axioms Proofs.StableHLO.den_allReduceMeanF
#audit_axioms Proofs.den_allReduceMeanF_eq_dpMean
#audit_axioms Proofs.skel_allReduceMeanF_of_spmd
#audit_axioms Proofs.den_allReduceMeanF_eq_lossGrad_meanLoss
#audit_axioms Proofs.den_allReduceMeanF_convWeightGradB
#audit_axioms Proofs.adamW_at_allReduceMeanF

-- SYNCHRONISED BATCHNORM -- THE DP STEP IS THE GLOBAL-BATCH STEP
-- (DataParallel/Basic.lean §sync + DataParallel/Sync.lean)
-- At the ℝ level: the positive twin of dpMeanGrad_ne_globalBatchGrad
#audit_axioms Proofs.dpMean_shardSum
#audit_axioms Proofs.dpSyncGrad_eq_globalBatchGrad
#audit_axioms Proofs.dpSyncGrad_eq_globalBatchGrad_contiguous
-- Sharding commutes with every per-example lift (definitional)
#audit_axioms Proofs.batchSlice_batchShard
#audit_axioms Proofs.batchShard_batchMap
#audit_axioms Proofs.batchShard_batchMapAux
-- the statistics subgraph denotes the global [μ ‖ σ²]; shard identities on the graph at ANY R --
-- the BN case of the per-net chain induction
#audit_axioms Proofs.den_syncStats_left
#audit_axioms Proofs.den_syncStats_right
#audit_axioms Proofs.mulR_nhw_ne_zero
#audit_axioms Proofs.den_bnSyncF_allReduce
#audit_axioms Proofs.den_bnSyncBack_allReduce
#audit_axioms Proofs.den_allReduceMeanF_bnSyncGammaGradB
-- the handed-back statistics under DP are the GLOBAL batch's own (bnBatchMeanB/VarB at N := R·N)
#audit_axioms Proofs.den_bnStatsMeanB_allReduce
#audit_axioms Proofs.den_bnStatsVarB_allReduce
-- At the node: the parameter collective is 1/R of the batch-R·N gradient node
#audit_axioms Proofs.den_allReduceMeanF_convWeightGradB_shard
#audit_axioms Proofs.den_allReduceMeanF_bnBetaGradB_shard
-- the divisor step (divConstB N on a replica vs divConstB (R·N) on one device)
#audit_axioms Proofs.HasVJP.backward_smul

-- SYNC-BN AT bf16: EVERY NODE SHARDS EXACTLY BUT THE CONV WEIGHT GRADIENT
-- (DataParallel/SyncBf16.lean)
-- forward convs and input-VJPs: replica r's node is shard r of the same node at batch R·N
#audit_axioms Proofs.den_batchOp_shard_node
#audit_axioms Proofs.den_convBf16_shard
#audit_axioms Proofs.den_convStridedBf16_shard
#audit_axioms Proofs.den_convBackBatchedBf16_shard
#audit_axioms Proofs.den_convStridedBackBatchedBf16_shard
-- weight gradients: each replica rounds its own partial sum; the batch-R·N node rounds their sum once
#audit_axioms Proofs.den_convWeightGradBBf16_eq_rnd
#audit_axioms Proofs.den_convStridedWeightGradBBf16_eq_rnd
#audit_axioms Proofs.den_allReduceMeanF_convWeightGradBBf16_shard
#audit_axioms Proofs.den_convWeightGradBBf16_global_split
#audit_axioms Proofs.den_allReduceMeanF_convWeightGradBBf16_sub_global
#audit_axioms Proofs.den_allReduceMeanF_convWeightGradBBf16_shard_id
#audit_axioms Proofs.den_allReduceMeanF_convStridedWeightGradBBf16_shard
#audit_axioms Proofs.den_convStridedWeightGradBBf16_global_split
#audit_axioms Proofs.den_allReduceMeanF_convStridedWeightGradBBf16_sub_global
-- the divisor step through rnd: rndP commutes with powers of two (R = 4 on every ImageNet run)
#audit_axioms Proofs.int_log_zpow_mul
#audit_axioms Proofs.int_log_abs_zpow_mul
#audit_axioms Proofs.rndP_zpow_mul
#audit_axioms Proofs.convBackBatchedBf16_smul
#audit_axioms Proofs.convStridedBackBatchedBf16_smul
#audit_axioms Proofs.convWeightGradBBf16_smul
#audit_axioms Proofs.convStridedWeightGradBBf16_smul

-- SYNC-BN AT RESNET-34: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (ResNet34SyncB.lean + ResNet34SyncStepTieB.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of resnet34ForwardBFull (R*N)
#audit_axioms Proofs.StableHLO.den_castIdx
#audit_axioms Proofs.StableHLO.batchShard_castIdx
#audit_axioms Proofs.StableHLO.den_bnSyncSiteLA
#audit_axioms Proofs.StableHLO.r34IdGraphSync_shard
#audit_axioms Proofs.StableHLO.r34DownGraphSync_shard
#audit_axioms Proofs.StableHLO.r34StemGraphSync_shard
#audit_axioms Proofs.StableHLO.resnet34FwdGraphSyncFull_shard
-- T3 twin: the single-device chain is homogeneous in its cotangent
#audit_axioms Proofs.SyncKit.bnGradInput_smul
#audit_axioms Proofs.SyncKit.bnInB_smul
#audit_axioms Proofs.SyncKit.maxPool3s2BackFlat_smul
#audit_axioms Proofs.ResNet34SyncTieB.r34IdCotIn_smul
#audit_axioms Proofs.ResNet34SyncTieB.r34DownCotIn_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#audit_axioms Proofs.SyncKit.bnSyncInB_shard
#audit_axioms Proofs.ResNet34SyncTieB.r34IdSyncCotIn_shard
#audit_axioms Proofs.ResNet34SyncTieB.r34DownSyncCotIn_shard
#audit_axioms Proofs.ResNet34SyncTieB.r34StemSyncCotC_shard
#audit_axioms Proofs.ResNet34SyncTieB.r34HeadCotBlk_shard
-- ...the strided-conv and dense collectives, and the divisor
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_convStridedWeightGradB_shard
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_denseWeightGradB_shard
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_denseBiasGradB_shard
#audit_axioms Proofs.SyncKit.replicaLossCot_eq
-- the capstone: every all-reduced parameter gradient IS the single-device node at R·N
#audit_axioms Proofs.ResNet34SyncTieB.r34_net_syncTiedB
#audit_axioms Proofs.ResNet34SyncTieB.r34_net_syncTiedB_smoothedCE

-- SYNC-BN: THE MBCONV PIECES MOBILENETV2 AND EFFICIENTNET-B0 SHARE
-- (MBConvSyncTieB.lean)
#audit_axioms Proofs.HasVJP3.backward_smul
#audit_axioms Proofs.SyncKit.depthwiseWeightGradB_smul
#audit_axioms Proofs.SyncKit.depthwiseStridedWeightGradB_smul
#audit_axioms Proofs.SyncKit.convStridedXlaWeightGradB_smul
#audit_axioms Proofs.SyncKit.rowDenseBackFlat_smul
#audit_axioms Proofs.SyncKit.rowDenseBackFlat_shard
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_depthwiseWeightGradB_shard
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_depthwiseStridedWeightGradB_shard
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_convStridedXlaWeightGradB_shard
#audit_axioms Proofs.SyncKit.depthwiseWSync_of_scaled
#audit_axioms Proofs.SyncKit.depthwiseStridedWSync_of_scaled
#audit_axioms Proofs.SyncKit.convStridedXlaWSync_of_scaled

-- SYNC-BN AT MOBILENETV2: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (MobileNetV2SyncB.lean + MobileNetV2SyncStepTieB.lean)
#audit_axioms Proofs.StableHLO.den_relu6_shard
#audit_axioms Proofs.StableHLO.mnv2StemGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv2NoExpGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv2ExpOnlyGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv2ResidGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv2StridedGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv2HeadGraphSync_shard
#audit_axioms Proofs.StableHLO.mobilenetv2FwdGraphSyncFull_shard
#audit_axioms Proofs.StableHLO.mobilenetv2FwdGraphSyncFullDo_shard
#audit_axioms Proofs.SyncKit.relu6MaskB_smul
#audit_axioms Proofs.SyncKit.depthwiseStridedXlaWeightGradB_smul
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2NoExpCotIn_smul
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2ResidCotIn_smul
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2StridedCotIn_smul
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2HeadCotBlk_smul
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2NoExpSyncCotIn_shard
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2SyncCotInBody_shard
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2ResidSyncCotIn_shard
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2StridedSyncCotIn_shard
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2StemSyncCotC_shard
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2HeadSyncCotBlk_shard
#audit_axioms Proofs.SyncKit.den_allReduceMeanF_depthwiseStridedXlaWeightGradB_shard
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB
#audit_axioms Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB_smoothedCE

-- SYNC-BN AT EFFICIENTNET-B0: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (EfficientNetSyncB.lean + EfficientNetSyncStepTieG.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of efficientnetForwardBFull (R*N)
#audit_axioms Proofs.StableHLO.den_swishF_shard
#audit_axioms Proofs.StableHLO.den_addV_shard
#audit_axioms Proofs.StableHLO.mbNoExpGraphSync_shard
#audit_axioms Proofs.StableHLO.mbBodyGraphSync_shard
#audit_axioms Proofs.StableHLO.mbResidGraphSync_shard
#audit_axioms Proofs.StableHLO.mbStridedGraphSync_shard
#audit_axioms Proofs.StableHLO.stemGraphSync_shard
#audit_axioms Proofs.StableHLO.headGraphSync_shard
#audit_axioms Proofs.StableHLO.efficientnetFwdGraphSyncFull_shard
#audit_axioms Proofs.StableHLO.mbResidDropGraphSync_shard
#audit_axioms Proofs.StableHLO.headGraphSyncDo_shard
#audit_axioms Proofs.StableHLO.efficientnetFwdGraphSyncFullDrop_shard
-- T3 twin: each block's input cotangent IS its certified VJP (T3 threads `.backward`)
#audit_axioms Proofs.EnetSyncTieG.xCotIn_eq_vjp
#audit_axioms Proofs.EnetSyncTieG.rCotIn_eq_vjp
#audit_axioms Proofs.EnetSyncTieG.sCotIn_eq_vjp
#audit_axioms Proofs.EnetSyncTieG.nCotIn_eq_vjp
#audit_axioms Proofs.EnetSyncTieG.hdCotIn_eq_vjp
-- ...the single-device chain is homogeneous in its cotangent
#audit_axioms Proofs.BackLinks.gateCotB_smul
#audit_axioms Proofs.EnetSyncTieG.tCotDc_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#audit_axioms Proofs.SyncKit.gateCotB_shard
#audit_axioms Proofs.SyncKit.seInB_shard
#audit_axioms Proofs.SyncKit.bnSyncInB_shard_bnBackB
#audit_axioms Proofs.EnetSyncTieG.tsCotDc_shard
#audit_axioms Proofs.EnetSyncTieG.xsCotIn_scaled
#audit_axioms Proofs.EnetSyncTieG.rsCotIn_scaled
#audit_axioms Proofs.EnetSyncTieG.ssCotIn_scaled
#audit_axioms Proofs.EnetSyncTieG.nsCotIn_scaled
#audit_axioms Proofs.EnetSyncTieG.hdsCotIn_scaled
-- the per-block ties and the capstone: every all-reduced parameter gradient IS the node at R·N
#audit_axioms Proofs.EnetSyncTieG.tail_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.exp_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.strided_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.noExp_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.stem_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.head_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.efficientnet_net_syncTiedG
#audit_axioms Proofs.EnetSyncTieG.efficientnet_net_syncTiedG_smoothedCE

-- SYNC-BN AT RESNET-50: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (ResNet50SyncB.lean + ResNet50SyncStepTieB.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of resnet50ForwardBFull (R*N) q
#audit_axioms Proofs.StableHLO.den_addVB_shard_comm
#audit_axioms Proofs.StableHLO.r50IdGraphSync_shard
#audit_axioms Proofs.StableHLO.r50ProjGraphSync_shard
#audit_axioms Proofs.StableHLO.r50DownGraphSync_shard
#audit_axioms Proofs.StableHLO.resnet50FwdGraphSyncFull_shard
-- T3 twin: the single-device chain is homogeneous in its cotangent
#audit_axioms Proofs.ResNet50SyncTieB.r50IdCotIn_smul
#audit_axioms Proofs.ResNet50SyncTieB.r50ProjCotIn_smul
#audit_axioms Proofs.ResNet50SyncTieB.r50DownCotIn_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#audit_axioms Proofs.ResNet50SyncTieB.r50IdSyncCotIn_shard
#audit_axioms Proofs.ResNet50SyncTieB.r50ProjSyncCotIn_shard
#audit_axioms Proofs.ResNet50SyncTieB.r50DownSyncCotIn_shard
#audit_axioms Proofs.ResNet50SyncTieB.r50IdSyncCotIn_scaled
#audit_axioms Proofs.ResNet50SyncTieB.r50ProjSyncCotIn_scaled
#audit_axioms Proofs.ResNet50SyncTieB.r50DownSyncCotIn_scaled
-- the per-block ties and the capstone (the cotangent a binder), then both losses discharged
#audit_axioms Proofs.ResNet50SyncTieB.r50_idblock_syncTiedB
#audit_axioms Proofs.ResNet50SyncTieB.r50_projblock_syncTiedB
#audit_axioms Proofs.ResNet50SyncTieB.r50_downblock_syncTiedB
#audit_axioms Proofs.ResNet50SyncTieB.r50_net_syncTiedB
#audit_axioms Proofs.ResNet50SyncTieB.replicaBceLossCot_eq
#audit_axioms Proofs.ResNet50SyncTieB.r50_net_syncTiedB_smoothedCE
#audit_axioms Proofs.ResNet50SyncTieB.r50_net_syncTiedB_bce

-- SYNC-BN AT MOBILENETV4: THE SYNC-BN DP RENDER IS THE SINGLE-DEVICE NET AT R·N
-- (MobileNetV4SyncB.lean + MobileNetV4SyncStepTieB.lean)
-- T2 twin: replica r's sync-BN forward graph denotes shard r of mobilenetv4ForwardBFull (R*N)
#audit_axioms Proofs.StableHLO.den_castIdx_shard
#audit_axioms Proofs.StableHLO.mnv4StemGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4FusedGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4ExtraDWBodyGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4ConvNeXtBodyGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4FfnBodyGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4StridedGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4SkipGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4HeadGraphSync_shard
#audit_axioms Proofs.StableHLO.mnv4FwdGraphSyncFull_shard
-- T3 twin: the single-device chain is homogeneous in its cotangent
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4BodyCotIn_smul
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4SBodyCotIn_smul
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4StemCotC_smul
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4FusedCotIn_smul
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4HeadCotIn_smul
-- ...each replica's sync-BN backward chain is the shard of the single-device one
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4BodySyncCotIn_shard
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4SBodySyncCotIn_shard
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4StemSyncCotC_shard
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4FusedSyncCotIn_shard
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4HeadSyncCotIn_shard
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4BodySyncCotIn_scaled
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4SkipSyncCotIn_scaled
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4SBodySyncCotIn_scaled
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4FusedSyncCotIn_scaled
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4HeadSyncCotIn_scaled
-- the per-block ties and the capstone (the cotangent a binder), then smoothed CE discharged
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_extradw_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_convnext_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_ffn_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_strided_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_stem_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_fused_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_head_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB
#audit_axioms Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB_smoothedCE

-- RESNET-50's TWO PREREQUISITES: THE LAMB TRIPLE AND BCE'S COTANGENT
#audit_axioms Proofs.lambStep
#audit_axioms Proofs.lambScale_zero_weight
#audit_axioms Proofs.StableHLO.lamb_triple_faithful
#audit_axioms Proofs.StableHLO.lamb_triple_faithful_committed
#audit_axioms Proofs.StableHLO.lamb_triple_faithful_excluded
#audit_axioms Proofs.softplus_hasDerivAt
#audit_axioms Proofs.softplus_neg
#audit_axioms Proofs.log_sigmoidScalar
#audit_axioms Proofs.one_sub_sigmoidScalar
#audit_axioms Proofs.bceLogits_eq_logSigmoid
#audit_axioms Proofs.pdiv_coordFun
#audit_axioms Proofs.bceLogits_grad
#audit_axioms Proofs.bceLossCotGraph_den
#audit_axioms Proofs.bceLossCotGraph_row
#audit_axioms Proofs.bceLossCotGraph_row_committed

-- RESNET-50's T1 AND T2 AT BATCH BATCHNORM (ResNet50FullB{,VJP}.lean)
#audit_axioms Proofs.resnet50ForwardBFull
#audit_axioms Proofs.r50IdBHasVJPAt
#audit_axioms Proofs.r50ProjBHasVJPAt
#audit_axioms Proofs.r50DownBHasVJPAt
#audit_axioms Proofs.r50NetLayer
#audit_axioms Proofs.r50NetLayer_fwd_apply
#audit_axioms Proofs.resnet50ForwardBFullHasVJPAt
#audit_axioms Proofs.resnet50ForwardBFull_eq_chain
#audit_axioms Proofs.resnet50ForwardBFullHasVJPAt_correct
#audit_axioms Proofs.resnet50ForwardBFull_differentiableAt

-- T2, the typed forward graph (ResNet-50)
#audit_axioms Proofs.StableHLO.r50IdGraphB_faithful
#audit_axioms Proofs.StableHLO.r50ProjGraphB_faithful
#audit_axioms Proofs.StableHLO.r50DownGraphB_faithful
#audit_axioms Proofs.StableHLO.r50StemGraphB_faithful
#audit_axioms Proofs.StableHLO.resnet50FwdGraphBFull_faithful

-- RESNET-50's T3 -- THE FOLD AND THE TIE (ResNet50StepTieB.lean)
#audit_axioms Proofs.ResNet50TieB.r50IdCotIn_eq_vjp
#audit_axioms Proofs.ResNet50TieB.r50ProjCotIn_eq_vjp
#audit_axioms Proofs.ResNet50TieB.r50DownCotIn_eq_vjp
#audit_axioms Proofs.ResNet50TieB.r50_idblock_tiedB
#audit_axioms Proofs.ResNet50TieB.r50_projblock_tiedB
#audit_axioms Proofs.ResNet50TieB.r50_downblock_tiedB
#audit_axioms Proofs.ResNet50TieB.r50_net_tiedB
#audit_axioms Proofs.ResNet50TieB.r50_lossCot_is_smoothedCE_grad
#audit_axioms Proofs.ResNet50TieB.r50_lossCot_is_bce_grad
