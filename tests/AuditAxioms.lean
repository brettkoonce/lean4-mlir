import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.JacobianSeal
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.MobileNetV2JacobianSeal
import LeanMlir.Proofs.ConvNeXt
import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.MnistCNN
import LeanMlir.Proofs.CifarCNN
import LeanMlir.Proofs.IR
import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.StableHLOParse
import LeanMlir.Proofs.StridedConv
import LeanMlir.Proofs.ResNet34
import LeanMlir.Proofs.ResNet34LivePC
import LeanMlir.Proofs.ResNet34LiveSeal
import LeanMlir.Proofs.ResNet34LiveFull
import LeanMlir.Proofs.MobileNetV2JacobianSealFull
import LeanMlir.Proofs.ResNet34LiveRealistic
import LeanMlir.Proofs.ResNet34LiveRealisticSeal
import LeanMlir.Proofs.MobileNetV2SealRealistic
import LeanMlir.Proofs.PerChannelBN
import LeanMlir.Proofs.LinearTrainStep
import LeanMlir.Proofs.MlpTrainStep
import LeanMlir.Proofs.CnnTrainStep
import LeanMlir.Proofs.CifarBnClose
import LeanMlir.Proofs.CnnChainClose
import LeanMlir.Proofs.Cifar8Close
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
import LeanMlir.Proofs.ResNet34FaithfulPoC
import LeanMlir.Proofs.ResNet34TiePoC
import LeanMlir.Proofs.MobileNetV2FaithfulPoC
import LeanMlir.Proofs.MobileNetV2FaithfulPoCPaper
import LeanMlir.Proofs.MobileNetV2TiePoCPaper
import LeanMlir.Proofs.EfficientNetFaithfulPoC
import LeanMlir.Proofs.EfficientNetTiePoC
import LeanMlir.Proofs.ConvNeXtClose
import LeanMlir.Proofs.ConvNeXtChainClose
import LeanMlir.Proofs.ViTFwdGraph
import LeanMlir.Proofs.ViTClose
import LeanMlir.Proofs.ViTChainClose
import LeanMlir.Proofs.ViTVecLN
import LeanMlir.Proofs.ViTMultiHead
import LeanMlir.Proofs.ViTMultiHeadChain
import LeanMlir.Proofs.ViTDepthK
import LeanMlir.Proofs.MobileNetV2FullPaper
import LeanMlir.Proofs.ConvNeXtFullT
import LeanMlir.Proofs.FloatBridge
import LeanMlir.Proofs.SgdDescent
import LeanMlir.Proofs.SgdDescentLinear
import LeanMlir.Proofs.SgdDescentCnn
import LeanMlir.Proofs.CifarFloatBridge
import LeanMlir.Proofs.BnFloatBridge
import LeanMlir.Proofs.SgdDescentMlp
import LeanMlir.Proofs.AdamStep
import LeanMlir.Proofs.AdamRender
import LeanMlir.Proofs.EfficientNetBackB0
import LeanMlir.Proofs.MobileNetV2BackB0
import LeanMlir.Proofs.ResNet34BackB0
import LeanMlir.Proofs.ConvNeXtBackB0
import LeanMlir.Proofs.ConvNeXtFaithfulPoC
import LeanMlir.Proofs.ConvNeXtTiePoC
import LeanMlir.Proofs.ViTBackB0
import LeanMlir.Proofs.LinearFaithfulPoC
import LeanMlir.Proofs.E4M3FaithfulPoC
import LeanMlir.Proofs.MlpFaithfulPoC
import LeanMlir.Proofs.CnnFaithfulPoC
import LeanMlir.Proofs.CifarFaithfulPoC
import LeanMlir.Proofs.CifarBnFaithfulPoC
import LeanMlir.Proofs.CifarBnTiePoC
import LeanMlir.Proofs.Cifar8FaithfulPoC
import LeanMlir.Proofs.Cifar8TiePoC
import LeanMlir.Proofs.Cifar8BnTiePoC
import LeanMlir.Proofs.ViTFaithfulPoC
import LeanMlir.Proofs.ViTTiePoC

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

-- Nonzero-Jacobian seal (JacobianSeal.lean, planning/whole_network_backward.md Item B):
-- the generic level-3 bridge. backward_ne_zero_of_pdiv_ne — one nonzero Jacobian entry
-- ⇒ the proven backward is not the zero map (the basis cotangent collapses the correctness
-- sum). fderiv_eq_zero_of_pdiv_all_zero / exists_pdiv_ne_of_fderiv_ne — the fderiv form
-- (all entries zero ⇔ fderiv = 0, via the standard-basis decomposition sum_smul_basisVec).
-- mnistLinear_backward_nontrivial — the linear-classifier demo (Jacobian = W). Upgrades a
-- witness from "forward ≠ const" to "the backward here is non-trivial"; the deep kinked
-- witnesses (Mnv2Live / a future ResNet34Live) discharge the pdiv ≠ 0 premise (Item B2).
#print axioms HasVJP.backward_ne_zero_of_pdiv_ne
#print axioms sum_smul_basisVec
#print axioms fderiv_eq_zero_of_pdiv_all_zero
#print axioms exists_pdiv_ne_of_fderiv_ne
#print axioms HasVJP.backward_nontrivial_of_fderiv_ne
#print axioms mnistLinear_backward_nontrivial
-- The pointwise (HasVJPAt) seal variants — the kinked witnesses are HasVJPAt, not HasVJP.
#print axioms HasVJPAt.backward_ne_zero_of_pdiv_ne
#print axioms HasVJPAt.backward_nontrivial_of_fderiv_ne
-- Item B2 discharged at the live MobileNetV2 witness (MobileNetV2JacobianSeal.lean): the
-- whole-net Jacobian is genuinely nonzero at the input 0 (the product-rule cross-term
-- vanishes there, so no BN-variance derivative is needed), hence the proven backward is
-- not the zero map. Upgrades Mnv2Live from level 2 (forward ≠ const) to level 3.
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
-- MobileNetV2: the LIVE counterpart — same ReLU6 bundle discharged on a
-- NONZERO, non-collapsed net via the `γ=1,β=3,n≤8` window (`bn13_window`),
-- not a constant collapse. (Nonzero-Jacobian seal is the documented residual.)
#print axioms Mnv2Live.bn13_window
#print axioms Mnv2Live.mnv2Live_has_vjp_correct
-- ...and the live witness is non-degenerate: its forward is non-constant
-- (`forward X ≠ forward 0`), so the Jacobian is not identically zero — the
-- formal seal that distinguishes it from the constant-output `MobileNetV2Concrete`.
#print axioms Mnv2Live.chSum_convX
#print axioms Mnv2Live.mnv2Live_forward_nonconstant
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
-- PoC capstones (LinearFaithfulPoC.lean): the mnist-linear trainer's render is
-- the certified loss-descent step — forward end-to-end tied, train step certified,
-- and the param-grad/SGD tail given a structural denotation proven certified.
-- See planning/verified_faithful_sweep.md; the byte tie (committed .mlir == render)
-- is the CI "Verified-render drift guard" step in proofs.yml.
#print axioms LinPoC.poc_fwd_faithful
#print axioms LinPoC.poc_fwd_is_render
#print axioms LinPoC.poc_train_step_certified
-- Tail fold closed: the emitted weightSgd/biasSgd SHlo nodes (what
-- linTrainStepFaithfulV prints) denote the certified loss-descent step.
#print axioms LinPoC.poc_weightSgd_den_eq
#print axioms LinPoC.poc_biasSgd_den_eq
#print axioms LinPoC.poc_train_step_tail_certified
-- mnist-MLP fully folded: backward-chain cotangent subgraphs denote mlpCotOut*,
-- and the 6 emitted weightSgd/biasSgd ops (what mlpTrainStepFaithfulV prints)
-- denote the certified per-layer loss-descent step (MlpFaithfulPoC.lean).
#print axioms MlpPoC.cot1_den
#print axioms MlpPoC.cot0_den
#print axioms MlpPoC.W2_den_certified
#print axioms MlpPoC.W1_den_certified
#print axioms MlpPoC.W0_den_certified
#print axioms MlpPoC.b2_den_certified
#print axioms MlpPoC.b1_den_certified
#print axioms MlpPoC.b0_den_certified
-- mnist-mlp FULLY TIED: the top loss cotangent pinned to the composed softmax-CE gradient of the
-- forward (mlpLossCot_den), so all 6 outputs denote the certified step driven by the real loss
-- (output weight W₂ folded to the WHOLE-loss gradient ∂CE/∂W₂).
#print axioms MlpPoC.mlpLossCot_den
#print axioms MlpPoC.mlp_W2_tied_totalloss
#print axioms MlpPoC.mlp_train_step_tied_certified
-- mnist-CNN fully folded: the 10 emitted param ops (what cnnTrainStepFaithfulV
-- prints) denote the certified per-param loss-descent step (CnnFaithfulPoC.lean) —
-- conv layers via the new convWeightSgd/convBiasSgd ops + chain bridges, dense head
-- via weightSgd/biasSgd + the M2 dense bridges.
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
-- mnist-cnn dense-head TIE: the top loss cotangent pinned to the composed softmax-CE gradient of
-- the CONV forward (cnnLossCot_den), and the dense output weight W₅ folded to ∂CE/∂W₅ through the
-- whole conv+dense forward (cnn_W5_tied_totalloss). Conv layers W₁/W₂ remain (conv backward chain).
#print axioms CnnPoC.cnnLossCot_den
#print axioms CnnPoC.cnn_W5_tied_totalloss
-- mnist-cnn CONV fold: all four conv kernel/bias ops, at the REAL conv forward (ac1/ac2/hc2 = the
-- actual conv₁/relu/conv₂/relu outputs) and the composed softmax-CE cotangent, denote the certified
-- step. With the dense-head tie above, the whole cnn train step is den-composed (no free acts).
#print axioms CnnPoC.cnn_conv_tied_certified
-- ch5-CIFAR fully folded (no-BN, 2-scale): the generic conv ops cover all 4 conv
-- layers (convW_den/convB_den) and the 3-dense head's 6 outpus denote the certified
-- step (CifarFaithfulPoC.lean) — reuses the cnn conv ops, no new core ops.
#print axioms CifarPoC.convW_den
#print axioms CifarPoC.convB_den
#print axioms CifarPoC.dW5_den
#print axioms CifarPoC.db5_den
#print axioms CifarPoC.dW6_den
#print axioms CifarPoC.db6_den
#print axioms CifarPoC.dW7_den
#print axioms CifarPoC.db7_den
-- ch5-CIFAR §1a TIE: the emitted loss-cotangent graph denotes softmax-CE of the cifar
-- forward (cifarLossCot_den); the dense output W₇ folds to ∂CE/∂W₇ through the whole
-- forward (cifar_W7_tied_totalloss); and all 4 conv layers are tied at the real backward-
-- chain cotangent (cifar_conv_tied_certified) — the cnn tie scaled to 2 conv stages, the
-- new cifarChainCotW2 crossing pool₁ (conv₃-back then maxpool₁-back).
#print axioms CifarPoC.cifarLossCot_den
#print axioms CifarPoC.cifar_W7_tied_totalloss
#print axioms CifarPoC.cifar_conv_tied_certified
-- ch5-CIFAR-BN fully folded: conv layers + dense head reuse the cifar fold; the per-
-- channel BN γ/β ops (bnGammaSgd/bnBetaSgd) denote the certified step (CifarBnFaithfulPoC).
#print axioms CifarBnPoC.bnGamma_den
#print axioms CifarBnPoC.bnBeta_den
-- ch5-CIFAR-BN §1a TIE: all 16 conv+BN params tied at the real forward + the BN backward chain
-- (BN-output cots relu-masked for γ/β; conv cots = BN-back of them); loss-cot + dense W₇ total-loss
-- fold. The cifar tie + a BN-back at every conv (CifarBnTiePoC.lean).
#print axioms CifarBnPoC.cifarBnLossCot_den
#print axioms CifarBnPoC.cifarBn_W7_tied_totalloss
#print axioms CifarBnPoC.cifarBn_convbn_tied_certified
-- deeper 8-conv cifar8 fully folded: conv layers reuse CifarPoC generics, the 3 dense
-- layers via the generic denseW_den/denseB_den (Cifar8FaithfulPoC.lean).
#print axioms Cifar8PoC.denseW_den
#print axioms Cifar8PoC.denseB_den
-- ch5-cifar8 §1a TIE: all 16 conv params tied at the real 4-stage forward + the backward chain
-- (cifar's chain repeated — cnnChainCotW2/cnnChainCotW1/cifarChainCotW2 reused, no new constructor);
-- loss-cot + dense Wb total-loss fold (Cifar8TiePoC.lean).
#print axioms Cifar8PoC.cifar8LossCot_den
#print axioms Cifar8PoC.cifar8_Wb_tied_totalloss
#print axioms Cifar8PoC.cifar8_convs_tied_certified
-- ch5-cifar8-bn §1a TIE: cifar8's 4-stage chain + a BN-back at every conv; all 32 conv+BN params
-- tied at the real forward (BN-output cots relu-masked for γ/β, conv cots = BN-back); loss-cot.
-- Pure reuse of CifarPoC/CifarBnPoC generics — zero new ops/bridges/constructors (Cifar8BnTiePoC.lean).
#print axioms Cifar8BnPoC.cifar8BnLossCot_den
#print axioms Cifar8BnPoC.cifar8Bn_convbn_tied_certified
-- ch6-ResNet-34 fully folded (full [3,4,6,3], 146 params): the 2 new strided-conv SGD ops
-- (convStrided{Weight,Bias}Sgd) for the 7×7 stem + 3×3 downsample/projection convs denote the
-- certified step; the 142 other params reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics.
#print axioms ResNet34PoC.convStridedW_den
#print axioms ResNet34PoC.convStridedB_den
-- ch7-MobileNetV2 §1 fold (depthwise half): the 4 new depthwise SGD ops denote the certified step.
-- Stride-2 (b1/b3/b5/b6) one-line via the flat strided VJP; stride-1 (b2/b4) weight via the flat
-- bridge (hasVJP3_to_hasVJP.correct, the 3-index→flat reindex), bias via the spatial reduce.
#print axioms Mnv2PoC.depthwiseW_den
#print axioms Mnv2PoC.depthwiseB_den
#print axioms Mnv2PoC.depthwiseStridedW_den
#print axioms Mnv2PoC.depthwiseStridedB_den
-- ch7-MobileNetV2 FULL 17-block paper §1 fold (den): all 210 params of the paper train step
-- (mnv2TrainStepFaithfulVPaper) den-certified, one capstone per block-type param profile. ZERO
-- new ops/lemmas — every conjunct delegates to the audited generics (cifar8-bn lesson at scale).
#print axioms Mnv2PaperPoC.mnv2StemParamsCertified
#print axioms Mnv2PaperPoC.mnv2NoExpParamsCertified
#print axioms Mnv2PaperPoC.mnv2Stride1ParamsCertified
#print axioms Mnv2PaperPoC.mnv2Stride2ParamsCertified
#print axioms Mnv2PaperPoC.mnv2HeadParamsCertified
#print axioms Mnv2PaperPoC.mnv2DenseParamsCertified
-- ch8-EfficientNet-B0 §1 fold (den): every batched param-SGD op type denotes the certified Σ_n
-- batched gradient (the batch-sum bridge). Generic in dims+cotangent; covers all 262 params.
#print axioms EnetPoC.convWB_den
#print axioms EnetPoC.convStridedWB_den
#print axioms EnetPoC.denseWB_den
#print axioms EnetPoC.denseBB_den
#print axioms EnetPoC.bnGammaB_den
#print axioms EnetPoC.bnBetaB_den
#print axioms EnetPoC.depthwiseWB_den
#print axioms EnetPoC.depthwiseStridedWB_den
-- ch8-EfficientNet-B0 §1a TIE: the loss-cotangent den (top of the chain) + the five per-block-type tie
-- lemmas covering all 262 params — each param-SGD op denotes the certified batched Σ_n loss-descent
-- step at the REAL loss-driven backward cotangent (swish masks, the SE gate fan-in via
-- seReduceB→sigmoid→denseRow→swish, true batch-norm backs, strided depthwise) — then the WHOLE-NET
-- 16-block thread `efficientnet_net_tied`: composes all 262 params through the real
-- efficientnetForwardB_full, the per-block dyOuts threaded top-down by the proven block VJPs.
#print axioms EnetTiePoC.efficientnetLossCot_den
#print axioms EnetTiePoC.enet_exp_tied
#print axioms EnetTiePoC.enet_strided_tied
#print axioms EnetTiePoC.enet_noexp_tied
#print axioms EnetTiePoC.enet_stem_tied
#print axioms EnetTiePoC.enet_head_tied
#print axioms EnetTiePoC.efficientnet_net_tied
-- ch7-MobileNetV2 FULL 17-block paper §1a TIE: the whole 210-param train step den-composed
-- forward→loss→backward through the REAL mobilenetv2ForwardPaper + the residual-fan-in cotangent
-- chain. Per-block-type tie lemmas applied across all 17 blocks + stem + conv-bn-relu6 head + dense.
#print axioms Mnv2TiePoC.mnv2_ivS1_tied
#print axioms Mnv2TiePoC.mnv2_ivS2_tied
#print axioms Mnv2TiePoC.mnv2_ivNoExp_tied
#print axioms Mnv2TiePoC.mnv2_stem_tied
#print axioms Mnv2TiePoC.mnv2_head_tied
#print axioms Mnv2TiePoC.mnv2_net_tied_certified
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

-- Deeper 8-conv CIFAR (the pedagogical BN-acceleration demo, FOUR conv→conv→pool
-- stages [16,16,32,32], four maxpools): the conditional whole-network VJP capstones
-- for both ±BN halves (twelve ReLU kinks + four maxpools; BN adds 0<εᵢ ×8). GPU-trained
-- by the `cifar8-verified` / `cifar8-bn-verified` exes through the proof-side VJP.
#print axioms cifarCnn8_has_vjp_at_correct
#print axioms cifarCnnBn8_has_vjp_at_correct
-- ...and the rendered-FORWARD peer: the deeper 8-conv CIFAR-CNN forward graph (four
-- conv→conv→pool stages [16,16,32,32] then 3-dense head) denotes the proven
-- cifarCnn8{Bn}Forward, by chaining the per-op faithfulness lemmas (the cifarBnFwdGraph
-- recipe extended by two conv stages). The deeper peer of cifar{,Bn}FwdGraph_faithful.
#print axioms StableHLO.cifar8FwdGraph_faithful
#print axioms StableHLO.cifar8BnFwdGraph_faithful

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
-- The first NON-DEGENERATE ResNet-34 whole-net backward witness (Item A, level 2):
-- a 2-channel real ResNet skeleton (strided stem + maxpool + 3 strided downsamples +
-- GAP + dense), nonzero weights, every smoothness/no-tie hypothesis discharged, with
-- forward X ≠ forward 0 (the channel-order invariant Dom2 threaded through the net).
-- Retires the "degenerate constant-output witness" caveat for ResNet-34.
#print axioms ResNet34LivePC.liveFwd2_has_vjp_correct
#print axioms ResNet34LivePC.liveFwd2_nonconstant
-- Item A level 3: the nonzero-Jacobian SEAL for the live ResNet-34 (ResNet34LiveSeal.lean).
-- Unlike Mnv2Live (globally smooth, sealed at input 0), the maxpool binds off-witness, so
-- the seal is at a channel-symmetric base Y where the channel-difference carrier vanishes;
-- the BN channel-diff identity then makes every istd-derivative cross-term carry a factor t
-- (drops at 0), so fderiv ℝ liveFwd2 Y ≠ 0 needs no BN-variance derivative. Upgrades the
-- live ResNet-34 from level 2 (forward ≠ const) to level 3 (backward not the zero map).
#print axioms ResNet34LiveSeal.liveFwd2_jacobian_nonzero
#print axioms ResNet34LiveSeal.liveFwd2_backward_nontrivial
-- Item A FULL DEPTH (ResNet34LiveFull.lean): the real [3,4,6,3] = 16-block live ResNet-34,
-- both level 2 (nonconstant) and level 3 (seal). The 13 identity blocks have a zeroed body
-- (relu(x+1) = x+1 on nonneg activations) and wash out through each downsample's BN
-- (bn(z+c) = bn(z)), so liveFwd2Full = liveFwd2 + 2 and the seal/derivative reuse the
-- empty-chain machinery verbatim. Full-depth, non-degenerate, nonzero-Jacobian-sealed.
#print axioms ResNet34LiveFull.liveFwd2Full_has_vjp_correct
#print axioms ResNet34LiveFull.liveFwd2Full_nonconstant
#print axioms ResNet34LiveFull.liveFwd2Full_jacobian_nonzero
#print axioms ResNet34LiveFull.liveFwd2Full_backward_nontrivial
-- Item B2 FULL DEPTH (MobileNetV2JacobianSealFull.lean): the real 17-block live
-- MobileNetV2, both level 2 (nonconstant) and level 3 (seal). The 15 identity skip
-- blocks have a zeroed body and NO final relu (linear bottleneck), so ivId a = a + 3
-- for every input; the chain shifts by +45, which GAP + the identity head pass, so
-- fwdFull = fwd + 45 and the seal/derivative reuse MobileNetV2JacobianSeal's Qq /
-- g_hasDerivAt. The whole-net VJP is genuinely composed through all 17 block backwards.
#print axioms Mnv2Live.fwdFull_has_vjp_correct
#print axioms Mnv2Live.fwdFull_nonconstant
#print axioms Mnv2Live.fwdFull_jacobian_nonzero
#print axioms Mnv2Live.fwdFull_backward_nontrivial
-- Item D (ResNet34LiveRealistic.lean): the live ResNet-34 whole-net backward at real
-- ImageNet 224×224 spatial resolution — the genuine 5-halving pyramid (224→112→56→28→14→7,
-- the conv1→maxpool→layer2→layer3→layer4 downsampling skeleton). β-parametric downsample
-- (β=64>√1568) + stem (β=160>√25088) keep every ReLU/maxpool smooth at n up to 2·112·112=25088;
-- forward X224≠forward 0 (level 2). Confirms the witness machinery used no hidden small-n cap.
#print axioms ResNet34LiveRealistic.liveFwd224_has_vjp_correct
#print axioms ResNet34LiveRealistic.liveFwd224_nonconstant
-- Item D LEVEL 3 (ResNet34LiveRealisticSeal.lean): the nonzero-Jacobian SEAL at 224×224.
-- A uniform channel-0 perturbation (vs the toy's single coordinate) makes channel0 = channel1 + δ
-- everywhere, so the 7×7 GAP of a uniform diff is δ and maxpool(ch0)=maxpool(ch1)+δ holds for ALL t
-- (max(a+δ,b+δ)=max(a,b)+δ) — eliminating the toy's eventual-selection topology. The UDiff invariant
-- threads like Dom2 (each BN ×istd via bnForward_chan_diff), so the output diff along the ray is
-- t·Rr (Rr = 4 positive istds) ⇒ fderiv ℝ liveFwd224 Y ≠ 0 ⇒ backward not the zero map. Full depth
-- realistic-spatial seal, n up to 25088, maxpool no-tie via the decreasing per-channel-injective base.
#print axioms R34RealSeal.liveFwd224_jacobian_nonzero
#print axioms R34RealSeal.liveFwd224_backward_nontrivial
-- Item D LEVEL 3 for MobileNetV2 (MobileNetV2SealRealistic.lean): the nonzero-Jacobian SEAL
-- at 224×224. ReLU6 is a BOUNDED window (0,6), so (unlike ResNet's β-grows-with-√n positivity
-- route) γ is SCALED DOWN — γ=1/128 keeps bn ∈ (3−|γ|√n, 3+|γ|√n) ⊆ (0,6) at n=2·112·112=25088.
-- The 1×1 channel-map weights are dimension-independent and reused. Uniform-perturbation UDiff
-- seal: the asymmetric stem turns a uniform input t into the channel difference −t, each BN
-- multiplies it by γ·istd, so the output diff is −t·Rr (Rr = four positive γ·istds), g'(0)=−Rr 0 ≠ 0.
#print axioms Mnv2RealSeal.fwdR_jacobian_nonzero
#print axioms Mnv2RealSeal.fwdR_backward_nontrivial
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
-- DEEPER 8-conv CIFAR (cifar8) close — the backward peer of cifar8{,Bn}FwdGraph_faithful:
-- each conv W/b, BN γ/β, and dense W/b output pinned to the cotangent the ACTUAL 4-stage
-- backward chain delivers. The chain reuses CnnChainClose's pieces (dense-head flat Back
-- chain cnnDenseHeadCot → maxpool-back Back3 via flatDenote → relu mask → conv-back Back3)
-- + per-channel BN-back (bnPerChannelTensor3_grad_input, 0<ε), the MNV2 stride-1 recipe,
-- through two more conv→conv→pool stages. Each θ output denotes θ − lr·(certified ∂/∂θ ·
-- the-actual-chain cotangent), via cnn_render_conv{W,b}_certified / weight_grad_bridge /
-- bias_grad_bridge / cifar_bn_render_{gamma,beta}_certified instantiated at the cifar8
-- cotangents (cifar8DenseHeadCot / cifar8CotBn8 / cifar8CotConv8 / cifar8CotBn7 / cifar8CotConv7).
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
-- ch6-ResNet-34 §1a TIE — the train-step ops tied to the REAL forward + the loss-driven backward
-- chain. Per-block-type tie lemmas (all params of an identity/downsample/stem block den = certified
-- at the ResNet34ChainClose cotangents); the residual fan-in SUM constructors (idBlockCotIn/
-- downBlockCotIn — r34's structural novelty, the skip+body cotangent add at each merge); the loss-cot
-- pin + the dense head total-loss fold. The cnn/cifar tie scaled to the residual net.
#print axioms ResNet34PoC.r34_idblock_tied
#print axioms ResNet34PoC.r34_downblock_tied
#print axioms ResNet34PoC.r34_stem_tied
#print axioms ResNet34PoC.r34LossCot_den
#print axioms ResNet34PoC.r34_dense_tied_totalloss
#print axioms ResNet34PoC.r34_dense_bias_den
-- THE WHOLE-NET CAPSTONE: resnet34Forward_full_pc threaded through all 16 residual blocks + stem +
-- dense, backward cotangents composed from the loss (dense/GAP-back + the residual fan-in sums at
-- every skip), every block tied at its real input + threaded cotangent. The full §1a ✅ TIED.
#print axioms ResNet34PoC.r34_net_tied_certified
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
-- The nested↔∘-chain bridge + correctness on the nested forward itself — the form-gap
-- this file shipped with, closed by the ConvNeXt-T rw-shaped-proof recipe
-- (equation-lemma rw + comp_apply peeling, syntactic close; simp/rfl would make the
-- kernel reduce the block bodies and time out).
#print axioms efficientnetForwardB_full_eq_chain
#print axioms efficientnetForwardB_full_has_vjp_correct
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
-- ConvNeXt §1 fold START — the per-channel layer-scale γ gradient cert (ConvNeXtFaithfulPoC). The
-- committed ConvNeXt net trains PER-CHANNEL layer-scale γ : Vec c (the layerScaleChF forward, which
-- broadcasts γ over c·h·w via chanIdx), so its γ-grad is the per-channel reduce dγ_c = Σ_{chanIdx=c} x·dy
-- — NOT the per-element Vec n of cnx_render_lsgamma_certified. The one genuinely-new proof obligation for
-- the ConvNeXt tie (depthwise/conv/dense/scalar-LN grads all reuse existing certs); the den target of the
-- pending layerScaleChGammaSgd core op. pdiv via the chanIdx reindex (pdiv_mul/pdiv_reindex/pdiv_const).
#print axioms Proofs.CnxPoC.pdiv_layerScaleCh_gamma
#print axioms Proofs.CnxPoC.cnx_render_lsgammaCh_certified
-- ConvNeXt §1 fold — the three new core SHlo param-SGD ops' `den` = the certified loss-descent step.
-- layerScaleChGammaSgd (per-channel layer-scale γ, the lsGradCh reduce + SGD), lnGammaSgd/lnBetaSgd
-- (scalar LN γ/β, the lnParamGrad reduces + SGD, output SHlo 1 ≅ tensor<1xf32>). One-line delegations
-- to the certs above / ConvNeXtClose. The ops build clean in StableHLO (roundtrip intact) + iree-compile.
#print axioms Proofs.CnxPoC.layerScaleChGammaSgd_den
#print axioms Proofs.CnxPoC.lnGammaSgd_den
#print axioms Proofs.CnxPoC.lnBetaSgd_den
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
-- ch9-ConvNeXt-T FULL [3,3,9,3] §1a TIE — the whole train step den-composed forward→loss→backward
-- through the REAL committed render forward (cnxStemFwdO/cnxBlockFwdO/cnxDownFwdO) + the loss-driven
-- cotangent chain: GELU masks (smooth, no kink), the residual fan-in `+ dyOut` at each of the 18
-- identity-skip merges, the LN-back at each of the 3 downsamples, scalar-LN γ/β + per-channel
-- layer-scale γ. Per-block / down / head / stem-bias ties applied across all 18 blocks. The 4
-- even-kernel weight grads (stem 4×4/s4 + 3 downsample 2×2/s2) are the documented render gap, outside
-- this den-tie. ZERO new ops/bridges — pure thread + fan-in over the §1-fold generics.
#print axioms Proofs.CnxTiePoC.cnx_block_tied
#print axioms Proofs.CnxTiePoC.cnx_down_tied
#print axioms Proofs.CnxTiePoC.cnx_stem_bias_tied
#print axioms Proofs.CnxTiePoC.cnx_head_tied
#print axioms Proofs.CnxTiePoC.cnxLossCot_den
#print axioms Proofs.CnxTiePoC.cnx_net_tied_certified
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

-- ============ ViT scaling pass: multi-head (ViTMultiHead.lean) ============
-- The MHSA math was always general in heads; this closes RENDERING + faithfulness at heads > 1.
-- Two new tokens headSliceF/headPadF (per-head column slice + pad-scatter; row-major layout makes
-- head h's columns the contiguous block [h*d,(h+1)*d) — slice/pad on the feature axis, the
-- clsSliceF/clsPadF templates). The concat is the PAD-SUM: Sum_h pad_h(SDPA_h) stays at the one
-- index N*(heads*d), dodging the (N*a)+(N*b) Nat-cast trap of a binary concat token; the pad is
-- simultaneously the slice's VJP, so the backward reuses the same pair on cotangents.
-- mhsa_layer_spelled ties mhsa_layer N heads d DIRECTLY (not a 1-head specialization) to the
-- per-head slice -> matmul-spelled SDPA -> pad form; vitFwdGraphMH(V)_faithful prove the heads =
-- hm1+1 forward graphs (scalar AND production vector-LN form) denote vitForward2(V).
#print axioms sum_headPadMat_apply
#print axioms mhsa_layer_spelled
#print axioms vitBlockSpelledMH_eq
#print axioms vitBlockSpelledMHV_eq
#print axioms StableHLO.den_headsSumG
#print axioms StableHLO.vitFwdGraphMH_faithful
#print axioms StableHLO.vitFwdGraphMHV_faithful

-- ============ ViT scaling pass: depth-k (ViTDepthK.lean) ============
-- General-depth distinct-param tower at the production form (vector-LN + multi-head). The proven
-- transformerTower shares ONE param tuple; vitForwardKV takes ps : Fin k -> BlockParamsV (the
-- 16-field per-block structure) and folds blocks head-first. The whole-net VJP holds at EVERY
-- depth with only 0 < eps (vjp_comp gluing the bridged transformerBlockV_has_vjp_mat inductively
-- — the tower induction at distinct params). At k = 2 it IS vitForward2V definitionally
-- (vitForwardKV_two_eq, rfl). The token-level fold vitBodyGraphKMHV (per-block SSA prefixes
-- b{i}_) denotes the Mat fold by induction chaining vitBlockGraphMHV_den_aux per block;
-- vitFwdGraphKMHV_faithful is the depth-general apex. Render: TestViTTrainPC now data-drives
-- DEPTH = 12 blocks (200 params, the ViT-Tiny count) — iree-compile OK + gfx1100 smoke 200/200.
#print axioms vitBodyKVFlat_eq_flatten
#print axioms vitBodyKVFlat_has_vjp
#print axioms vitForwardKV_two_eq
#print axioms vitForwardKV_has_vjp
#print axioms vitForwardKV_has_vjp_correct
-- Production capstone: vitForwardKV_has_vjp_correct instantiated at the real ViT-Tiny spec
-- (3×224² image, 16×16 patches → N=196 + CLS, D=192=3×64, MLP 768, 12 DISTINCT-param blocks,
-- 10 classes — MainVitTrain.lean's `vitTiny`). The full-architecture whole-net backward, the
-- ViT peer of convNextForwardT_has_vjp (18-block) / efficientnetForwardB_full_has_vjp (16-block).
#print axioms vitTiny_has_vjp_correct
#print axioms StableHLO.vitBodyGraphKMHV_den
#print axioms StableHLO.vitFwdGraphKMHV_faithful

-- ============ Paper-spec full MobileNetV2 (MobileNetV2FullPaper.lean) ============
-- The reduced ch7 close certified the repo's 6-block trainer; this scales the per-channel stage
-- machinery over the REAL [t,c,n,s] table — 17 bottlenecks at 224^2 (4 stride-2 downsamples,
-- 10 identity skips, 2 stage-first s=1 widenings, and the t=1 NO-EXPAND first block, the one
-- genuinely-new shape) + stem 3->32 + head 320->1280. The EfficientNetFullB0 enumeration recipe;
-- forward + graph + faithfulness (relu6 is kinked, so the whole-net input-VJP stays
-- pointwise-only — the repo standard for relu-family nets, same as full ResNet-34). The
-- MobileNetV2Close/ChainClose param bridges are dim-polymorphic and cover the paper shapes.
#print axioms StableHLO.ivNoExpGraphW_faithful
#print axioms StableHLO.ivExpOnlyGraphW_faithful
#print axioms StableHLO.ivResidGraphW_faithful
#print axioms StableHLO.ivStridedGraphW_faithful
#print axioms StableHLO.mobilenetv2FwdGraphPaper_faithful

-- ============ Full ConvNeXt-T [3,3,9,3] (ConvNeXtFullT.lean) ============
-- The last flagship without a full-architecture close. Stage depth-k (CnxBlockParams +
-- Fin k induction, convNextBlock_has_vjp as the chain step), LN + 2x2/s2 downsample
-- boundaries (existing stride-2 VJPs), and the NEW 4x4/s4 patchify stem
-- (flatConvStride4 = decimate . decimateOdd . stride-1 SAME conv — the LEFT-ALIGNED
-- pad-0 window x[4i..4i+3] of the paper's Conv2d(4, s=4) and the committed render —
-- + the flatConvStride4F token). Per-channel layer-scale (γls : Vec c, the paper's
-- form, via layerScaleChF/chanIdx). GELU/LN/conv smooth => the whole-net VJP is
-- GLOBAL at every depth (only the LN positivities) — ConvNeXt-T joins
-- efficientnetForwardB_full/vitForwardKV. Same scalar-LN representation caveat
-- as the representative; faithful channel-LN stays the optional follow-up.
#print axioms decimateOddFlat_has_vjp
#print axioms flatConvStride4_has_vjp
#print axioms convNextStageK_has_vjp
#print axioms cnxDownW_has_vjp
#print axioms convNextForwardT_has_vjp
-- The nested↔∘-chain bridge: provable ONLY with the rw-shaped proof (equation-lemma
-- rw + comp_apply peeling, syntactic close) — a simp/rfl close makes the kernel
-- iota-unroll the recursive stage folds (no defeq cache) and deterministically
-- time out. Closes the form-gap EfficientNet-B0 still carries.
#print axioms convNextForwardT_eq_chain
#print axioms convNextForwardT_has_vjp_correct
#print axioms StableHLO.cnxBlockGraphW_faithful
#print axioms StableHLO.cnxStageGraphK_den
#print axioms StableHLO.cnxDownGraphW_faithful
#print axioms StableHLO.convNextFwdGraphT_faithful
-- Committed-render config (no stem-LN — the committed convnext_train_step.mlir
-- omits the paper's stem-LN, 180 params): the graph tests/TestConvNeXtTTrainPC.lean
-- proof-renders at the committed signature. CAPSTONE: two-sided GPU parity vs the
-- committed trainer — 180/180 outputs, 179 bit-identical, worst rel-diff 1.05e-5.
#print axioms convNextForwardTC_has_vjp
#print axioms convNextForwardTC_eq_chain
#print axioms convNextForwardTC_has_vjp_correct
#print axioms StableHLO.convNextFwdGraphTC_faithful

-- ℝ→Float32 bridge, Tier 1 (FloatBridge.lean): standard-model rounding
-- (hypothesis-style `FloatModel`, NO project axioms — binary32 satisfies the
-- interface with u = 2⁻²⁴ on the normal range). Forward error bounds for the
-- toy nets: compounded dot/dense budgets (association-independent, so IREE
-- reduction reassociation is covered), ReLU exact-in-float 1-Lipschitz
-- pass-through, and the linear/MLP forward-extraction capstones.
#print axioms FloatModel.dot_close
#print axioms FloatModel.dot_close_linear
-- §1c (planning/floatbridge_quantization.md): the two-roundoff generalization
-- of the dot budget — a leaf precision u_leaf (FloatModel L, e.g. bf16 2⁻⁸ /
-- fp8-E4M3 2⁻⁴ on the matmul inputs) and an accumulate precision u_acc (M.u,
-- fp32 2⁻²⁴). dot_close_mixed: the leaf contributes only a FLAT per-leaf term
-- (2·u_leaf + u_leaf²)·Σ|xy| while the fan-in amplification rides entirely on
-- the accumulate γ-factor ((1+u_acc)^(n+1)−1) — the reason bf16-mixed is
-- non-vacuous where pure bf16 is not (the 1/u fan-in wall sits at u_acc, not at
-- the leaf). dot_close_mixed_uniform folds it to a single Σ|xy| factor (the
-- directly-instantiable, shipped-artifact form); dotMixed_exact_leaf shows
-- u_leaf = 0 collapses it back to dot_close (a genuine generalization).
#print axioms FloatModel.dotMixed
#print axioms FloatModel.dot_close_mixed
#print axioms FloatModel.dot_close_mixed_uniform
#print axioms FloatModel.dotMixed_exact_leaf
-- §1c threaded through the dense layer: denseMixed (leaf precision L on the
-- matmul, accumulate M on the bias add) is the deployed bf16-mixed dense layer;
-- dense_close_mixed = the leaf precision enters only via the flat dotMixed term,
-- the accumulate rides the bias add + fan-in γ. bf16/fp8 dense fall out by L.u.
#print axioms FloatModel.denseMixed
#print axioms FloatModel.dense_close_mixed
#print axioms FloatModel.dense_close
#print axioms FloatModel.dense_close_fresh
#print axioms FloatModel.relu_close
-- MaxPool exact-in-float (CNN.lean, planning §1b-A): max is compare-and-select,
-- so it rounds nothing — inherited input error e passes through with no
-- rounding term and no amplification (the max-peer of relu_close). max_close is
-- the scalar core; maxPool2_close lifts it to the 2×2 window; maxPoolFlat_close
-- is the Vec-space form the MNIST-CNN forward composes. The one genuinely-new
-- forward-budget fact the CNN rounding side needs beyond dense/relu.
#print axioms max_close
#print axioms maxPool2_close
#print axioms maxPoolFlat_close
-- Conv forward rounding budget (SgdDescentCnn.lean, planning §1b-A): conv =
-- dense at the conv fan-in. conv2d_eq_dense makes "conv = dense-with-sharing"
-- exact — each output coordinate is Proofs.dense of the kernel slab against the
-- flattened window (sum_w3 collapses the triple window sum to one fan-in sum).
-- convF is the float conv (M.dense on the window); convF_close is then
-- dense_close at the fan-in ic·kH·kW (the Higham γ rides that fan-in), with
-- convPad_close passing the inherited input error through the padded reads.
-- This is the conv half of cnn_float_close; combine with relu_close +
-- maxPoolFlat_close + the dense head for the whole-net forward budget.
#print axioms sum_w3
#print axioms conv2d_eq_dense
#print axioms convPad_close
#print axioms FloatModel.convF
#print axioms FloatModel.convF_close
-- Whole-net capstone (SgdDescentCnn.lean, planning §1b-A): flatConvF is the
-- Vec-space float conv; flatConvF_close gives the uniform conv-fan-in
-- layerBudget (conv threads exactly like a dense layer). cnn_float_close is the
-- binary32 forward-error bound for the whole Chapter-4 MNIST CNN — the
-- mlp_float_close_uniform nest extended to conv→relu→conv→relu→maxpool→3×dense,
-- with relu/maxpool exact-in-float passing error through unamplified. The
-- forward chain "binary32 → certified proximity" now closed for the CNN too.
#print axioms FloatModel.flatConvF
#print axioms FloatModel.flatConvF_close
#print axioms FloatModel.mnistCnnNoBnForwardF
#print axioms FloatModel.cnn_float_close
-- Chapter-5 no-BN CIFAR CNN (CifarFloatBridge.lean): cnn_float_close scaled to
-- the deeper net (4 conv in two conv→conv→pool stages + 3-dense head) — the
-- binary32 forward-error bound for cifarCnnForward, same layerBudget machinery,
-- zero new numerical primitives. First pass of the MNIST→CIFAR bridge step (BN
-- deferred). cifar_float_close is the CIFAR peer of cnn_float_close.
#print axioms FloatModel.cifarCnnForwardF
#print axioms FloatModel.cifar_float_close
-- CIFAR conv SGD-step rounding budgets (CifarFloatBridge.lean): the cotangent-
-- generic cnn_convW/convb_step_float_close instantiated at the two committed
-- CIFAR spatial scales — 32×32 (fan-in 1024, conv₁/conv₂) and 16×16 (256,
-- conv₃/conv₄). Each rounded conv weight/bias SGD entry is within an explicit
-- (a·g)/150 / (a·g)/2000 (+10⁻⁷) of the certified step. CIFAR peers of
-- mnist_cnn_convW_step_float_budget; the dense head reuses the MLP step closes.
#print axioms FloatModel.cifar_stage1_convW_step_float_budget
#print axioms FloatModel.cifar_stage1_convb_step_float_budget
#print axioms FloatModel.cifar_stage2_convW_step_float_budget
#print axioms FloatModel.cifar_stage2_convb_step_float_budget
-- BN float keystone (BnFloatBridge.lean): the new numerical op BatchNorm adds is
-- the inverse-stddev 1/√(σ²+ε), uncovered by the relative-error model. rsqrt_lipschitz
-- bounds 1/√ on [ε,∞) (constant 1/(2ε√ε), via 1/√a−1/√b = (b−a)/((√a+√b)√a√b));
-- bnIstd_close composes a supplied rsqrt accuracy `ers` with the variance rounding
-- `evar` ⇒ |fistd − bnIstd| ≤ ers/√ε + evar/(2ε√ε). The BN analog of the exp/eexp
-- softmax handoff; the mean/var Higham budgets + normalize chain are the mechanical tail.
#print axioms rsqrt_lipschitz
#print axioms bnVar_nonneg
#print axioms bnIstd_close
-- BN float tail (BnFloatBridge.lean): the full per-example bnForward closeness,
-- composed from the keystone. bnMean_close / bnVar_close are the mean/variance
-- Higham reductions (sum_close fan-in + per-term mul_close + division rounding);
-- bnForward_close_of is the normalize chain (sub + 2 mul + add); bnForward_close
-- assembles them — mean discharged by bnMean_close, istd by the rsqrt keystone,
-- normalize by bnForward_close_of, only the variance budget supplied (bnVar_close).
#print axioms FloatModel.bnForwardF
#print axioms FloatModel.bnMean_close
#print axioms FloatModel.bnVar_close
#print axioms FloatModel.bnForward_close_of
#print axioms FloatModel.bnForward_close
-- Conv gradient-step rounding (planning §1b-B): the conv weight gradient is a
-- spatial correlation (a dot over the h·w positions), the bias gradient a
-- spatial sum — so both rounded SGD steps reduce to the generic step closes.
-- dotSgd_step_close / sumSgd_step_close (FloatBridge.lean): dot_close / sum_close
-- feeding sgd_step_close — the reusable cores. convWeightGrad_eq_dot /
-- convBiasGrad_eq_sum re-express the certified conv gradient (conv2d_weight_pdiv)
-- as the flat spatial dot/sum (sum_s2 collapses the (hi,wi) grid).
-- cnn_convW_step_float_close / cnn_convb_step_float_close: the rounded conv
-- weight/bias updates within sgdErr of the real step, the dot/sum Higham γ at
-- fan-in h·w as the gradient-error slot.
#print axioms FloatModel.dotSgd_step_close
#print axioms FloatModel.sumSgd_step_close
#print axioms sum_s2
#print axioms convWeightGrad_eq_dot
#print axioms convBiasGrad_eq_sum
#print axioms FloatModel.cnn_convW_step_float_close
#print axioms FloatModel.cnn_convb_step_float_close
-- Item C — the numeric conv-weight-step capstone (SgdDescentCnn.lean) at the
-- committed Chapter-4 dims (conv2 32→32, 3×3, 28×28 → fan-in 784), u ≤ 2⁻²⁴,
-- lr = 1/10, |W| ≤ 3/5: every rounded conv2 weight SGD entry is within
-- (a·g)/250 + 10⁻⁷ of the certified step (a = conv-input activation bound,
-- g = conv cotangent bound, both a-posteriori / measured). The 0.4% rate is
-- lr·γ₇₈₅ — the gradient's Higham error at learning-rate scale.
#print axioms FloatModel.mnist_cnn_convW_step_float_budget
#print axioms FloatModel.pow_one_add_sub_one_le
#print axioms FloatModel.linear_float_close
#print axioms FloatModel.mlp_float_close
-- The numeric rung: the γ-form (classical γₖ = k·u/(1−k·u) division bound —
-- plain rational arithmetic at concrete u, no big-power evaluation), the
-- uniform-magnitude closed-form budgets, and the capstone at the committed
-- MainMnistMlpTrain dims (784→512→512→10): any binary32-accuracy model
-- (u ≤ 2⁻²⁴), |W| ≤ 3/5 (TRAINED magnitudes — measured max|W| = 0.52 on a
-- real 97.8% run; He init already exceeds 1/32 in its tails), |b|,|x| ≤ 1 ⇒
-- every rounded logit within 5100 of the exact-ℝ logit (worst-case logits
-- ≈4.5e7 ⇒ ≈1e-4 relative; measured drift 1.6e-5 — scripts/margin_probe.py).
#print axioms FloatModel.pow_gamma_bound
#print axioms FloatModel.dense_abs_le
#print axioms FloatModel.denseErr_le_uniform
#print axioms FloatModel.mlp_float_close_uniform
#print axioms FloatModel.mnist_mlp_float_budget
-- The gradient half, first rung: rounded backward ops (3-rounding param
-- update fl(θ − fl(lr·fl(a·c))), exact-select ReLU mask) + the float-side
-- kink condition — reluMask_close needs a QUANTITATIVE margin ez < |zᵢ|
-- (forward rounding must not flip a ReLU), the float analogue of the
-- suite's x k ≠ 0 hypotheses. cot_step_close reuses dense (b = 0) for the
-- transposed matvec, so the whole backward chain rides the forward
-- machinery. Capstones: the rounded W₂/b₂/W₁ SGD entries are within
-- explicit sgdErr budgets of θ − lr·(aᵢ·cⱼ) — the emitWeightGrad/
-- emitBiasGrad entries that mlp_render_{W2,W1,b2}_certified prove equal
-- to the pdiv-Jacobian contractions. Output cotangent g̃ ≈ g is a
-- hypothesis (softmax−onehot head awaits an exp accuracy axiom).
#print axioms FloatModel.mul_close
#print axioms FloatModel.sgd_step_close
#print axioms FloatModel.reluMask_close
#print axioms FloatModel.cot_step_close
#print axioms FloatModel.mlp_w2_step_float_close
#print axioms FloatModel.mlp_b2_step_float_close
#print axioms FloatModel.mlp_w1_step_float_close
-- Completion: the remaining param entries (b₁ = cotangent SGD; W₀/b₀ cross
-- BOTH masks, so both quantitative margins appear) — all six rounded MLP
-- train-step param entries now budgeted against their certified real steps,
-- the float mirror of mlp_render_{W2,W1,W0,b2,b1,b0}_certified. Plus the
-- numeric gradient capstone at the committed dims: u ≤ 2⁻²⁴, lr = 1/10,
-- |W| ≤ 3/5 (trained magnitudes), |b|,|x| ≤ 1, |g| ≤ 1, exact cotangent ⇒
-- every rounded W₂ SGD entry within 5/4 of the certified step (≈1.2 =
-- lr·E₁·|g|, the forward budget at lr scale; fresh backward rounding ≈2e-3;
-- measured deviation 7.5e-9 — the a-posteriori case in numbers).
#print axioms FloatModel.mlp_b1_step_float_close
#print axioms FloatModel.mlp_w0_step_float_close
#print axioms FloatModel.mlp_b0_step_float_close
#print axioms FloatModel.mnist_w2_step_float_budget
-- The loss head — the LAST Tier-1 float hypothesis discharged. fexp is
-- hypothesis-supplied (GPU exp has no IEEE spec; |fexp t − exp t| ≤
-- eexp·exp t is the constant vjp_oracle validates). softmax_perturb is the
-- elementary ratio sandwich (logit error δ moves softmax by ≤ e^(2δ)−1, no
-- MVT); softmaxF_close budgets the rounded exp/sum/div head at the same
-- logits; softmax_ce_cot_close combines both + the final rounded subtract
-- against the CERTIFIED gradient softmax−onehot (softmaxCE_grad). Numeric
-- mnist_cot_budget (n = 10, eexp ≤ 1e-6, logits within δ = 1/100): the
-- rounded cotangent is within 21/1000 of certified — nearly all of it the
-- e^(2δ)−1 ≈ 2δ math-perturbation term; head rounding < 4e-6. The δ
-- hypothesis is a-posteriori-style: the worst-case forward logit budget
-- (≈5100) makes e^(2δ)−1 vacuous — the formal hand-off point from worst-case
-- to measured-error analysis.
#print axioms FloatModel.sum_close
#print axioms FloatModel.softmax_perturb
#print axioms FloatModel.softmaxF_close
#print axioms FloatModel.softmax_ce_cot_close
#print axioms FloatModel.mnist_cot_budget
-- §3c (planning/floatbridge_quantization.md): the E4M3 (fp8) argmax-preservation
-- statement — the honest end-to-end accuracy claim that exists ONLY because
-- MNIST-linear is depth-1 (the single-matmul leaf bound IS the end-to-end bound,
-- no vacuous depth compounding). argmax_preserved: a B-accurate logit
-- perturbation cannot flip the prediction on any input whose strict top-1 margin
-- exceeds 2B (B is a hypothesis, so it holds for the proven worst-case bound AND
-- the demo's measured a-posteriori drift alike). denseMixedBudget /
-- dense_close_mixed_uniform_budget / denseMixedBudget_le_of: a single uniform
-- per-logit B over all outputs from dense_close_mixed (the layerBudget_le_of
-- analogue, keeping the fan-in power abstract). linear_e4m3_logit_budget: at the
-- committed 784→n dims, u_leaf ≤ 2⁻⁴, u_acc ≤ 2⁻²⁴, |x| ≤ 1, |W| ≤ 3/5, every
-- E4M3-mixed logit is within 61 of the exact-ℝ logit (worst-case; the leaf 12.5%
-- dominates, the fp32 fan-in γ₇₈₅ ≈ 5e-5 is negligible). linear_e4m3_argmax_preserved:
-- the capstone — margin > 122 ⟹ provably same prediction. Empirically (the demo,
-- measured B = 0.38) that region is 92.89% of the MNIST test set.
#print axioms FloatModel.argmax_preserved
#print axioms FloatModel.denseMixedBudget
#print axioms FloatModel.dense_close_mixed_uniform_budget
#print axioms FloatModel.denseMixedBudget_le_of
#print axioms u_e4m3
#print axioms FloatModel.linear_e4m3_logit_budget
#print axioms FloatModel.linear_e4m3_argmax_preserved
-- §3b (planning/floatbridge_quantization.md): the E4M3 (fp8) STRUCTURAL render-tie
-- (E4M3FaithfulPoC.lean) — correctness-of-implementation, NO accuracy claim. The
-- deployed fp8 kernel is block-scaled with fp32 accumulate: int weight code (per-output
-- column scale sWⱼ), int activation code (per-tensor sx), fp32 accumulate, one per-output
-- dequant sx·sWⱼ, fp32 bias. dequant_factors: the per-output scale factors out of the
-- accumulate ((sx·sWⱼ)·∑ q q = ∑ (sx q)(sWⱼ q)) — the arithmetic that makes "int matmul
-- then dequant" = "dequant then matmul", i.e. why fp32 accumulate is the faithful choice.
-- e4m3_render_faithful: the emitted graph (built ONLY from den-faithful ops operand/dotIn/
-- layerScaleF/addBcast — zero new SHlo constructors) denotes the intended dequant-first
-- algorithm quantLinear, for ANY quantizer q (E4M3 is one instance; q left abstract).
#print axioms QuantPoC.dequant_factors
#print axioms QuantPoC.e4m3_render_faithful
-- Inexact-gradient descent over ℝ (SgdDescent.lean): the keystone that
-- turns the FloatBridge budgets into a TRAINING statement. descent_segment
-- is the MVT-form descent lemma (segment-local differentiability +
-- coordinatewise gradient drift ⇒ f(x+d) ≤ f(x) + ⟨d,∇f⟩ + C·D²);
-- sgd_descent_inexact gives the explicit three-term bound (full descent −
-- oracle tax − curvature tax) for x − lr·ĝ with ‖ĝ−∇f‖∞ ≤ η — the η the
-- per-entry float budgets supply; sgd_descends: if each tax is ≤ a quarter
-- of the descent, the inexact step decreases the loss by ≥ lr·‖∇f‖₂²/2.
-- Discharging smoothness for the concrete nets is future work.
#print axioms fderiv_apply_eq_sum_grad
#print axioms descent_segment
#print axioms sgd_descent_inexact
#print axioms sgd_descends
-- The smoothness hypothesis discharged for the Chapter-2 net
-- (SgdDescentLinear.lean): linear_loss_gradAt re-derives the certified
-- ∂L/∂W_{ij} = xᵢ·(softmaxⱼ − onehotⱼ) through gradAt;
-- dense_unflatten_drift (logits move ≤ a·‖d‖₁); linear_loss_grad_lipschitz
-- gives the EXPLICIT segment-Lipschitz constant 2a²/(1−2aD) under 2aD < 1 —
-- the softmax ratio sandwich + the γ-form, no Hessian, no MVT — and
-- linear_sgd_descends: one inexact SGD step on the MNIST-linear classifier
-- provably decreases the cross-entropy loss by ≥ lr·‖∇L‖₂²/2. Smoothness is
-- PROVEN here, not assumed; the remaining hypotheses are checkable
-- arithmetic (oracle η from the float budgets, small-step, dominance).
#print axioms gradAt_eq_pdiv
#print axioms linear_loss_gradAt
#print axioms dense_unflatten_drift
#print axioms linear_loss_grad_lipschitz
#print axioms linear_sgd_descends
-- Item D / G1 — the η-composition, the "two halves finally meet"
-- (SgdDescentLinear.lean): the descent side (linear_sgd_descends) and the
-- rounding side (FloatBridge's cotErr/mulErr head budget) fused into one
-- statement. linearFloatGrad is the ACTUAL binary32 gradient the rendered
-- trainer computes (float forward logits → rounded softmax−onehot cotangent
-- → one rounded multiply by the exact input); linear_grad_close proves it is
-- within mulErr u a 1 0 (cotErr …) of the certified ∂L/∂W (softmax_ce_cot_close
-- for the head, mul_close for the input multiply with an exact left operand);
-- linear_float_sgd_descends discharges linear_sgd_descends' abstract η with
-- THAT proven budget — so "one binary32 SGD step on MNIST-linear provably
-- decreases the cross-entropy loss" holds with NO abstract gradient-accuracy
-- parameter. Depth-1 ⇒ no per-layer η-threading; the chain binary32 →
-- proximity → smoothness → descent is closed end-to-end for one net. The only
-- residue is the documented FloatModel → kernel trust boundary (exp accuracy
-- eexp, a-posteriori logit drift δ) + checkable arithmetic (small-step,
-- dominance).
#print axioms FloatModel.linearFloatGrad
#print axioms linearFloatGrad_apply
#print axioms linear_grad_close
#print axioms linear_float_sgd_descends
-- The smoothness hypothesis discharged through the Chapter-3 MLP
-- (SgdDescentMlp.lean), layer by layer. The key is the MARGIN hypothesis
-- (the step's ℓ1 radius keeps every ReLU pre-activation away from its
-- kink): the masks then FREEZE along the whole segment
-- (sign_stable_of_close / margin_keeps_offkink), and the frozen-mask loss
-- gets the same elementary treatment as the linear net — logit drift
-- (1-Lipschitz ReLU, column-tiled ℓ1 mass), softmax ratio sandwich, γ-form.
-- Closed-form gradients (mlp_hidden/input_loss_gradAt) collapse the
-- conditional folds to the explicit relu'⊙Wᵀ backprop forms; the explicit
-- segment-Lipschitz constants are 2·d₃·w₂²·a²/(1−2w₂aD) for the hidden
-- layer and 2·d₃·d₂²·w₁²·w₂²·a²/(1−2w₂d₂w₁aD) for the input layer; the
-- capstones mlp_{output,hidden,input}_sgd_descends prove one inexact SGD
-- step on EACH weight layer decreases the loss by ≥ lr·‖∇L‖₂²/2 (output
-- layer = the linear theorem at the hidden activation, margin-free).
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
-- Output-layer η-composition (planning §1a/§4, G1 for the MLP): the output rung
-- with NO abstract gradient-accuracy parameter — the actual binary32 output-layer
-- gradient M.linearFloatGrad W₂ b₂ a₁, its accuracy η = mulErr u a 1 0 (cotErr …)
-- proven (linear_grad_close), is fed into the output-layer descent. The output
-- layer IS linear_float_sgd_descends at the hidden activation a₁ (margin-free).
-- Hidden/input rungs still take abstract η (the joint-step float-backward
-- grad-close under the margins is left open).
#print axioms mlp_output_float_sgd_descends
-- Hidden-layer float-backward grad-close (planning §1a/§4, the joint-step engine):
-- with a₀ frozen exact, the binary32 W₁ gradient fl(a₀ᵢ·c̃₁ⱼ) — float layer-1
-- cotangent c̃₁ = mask(z̃₁, W₂ᵀ·c̃₂) from the float softmax−onehot head — is within
-- mulErr … 0 (layerBudget … (cotErr …)) of the certified a₀ᵢ·mask(z₁, W₂ᵀ·(softmax−onehot))ⱼ
-- (= mlp_hidden_loss_gradAt). Three reusable closes: softmax_ce_cot_close (head),
-- cot_step_close (masked W₂ᵀ contraction, under the margin E₁ < |z₁ⱼ|), mul_close
-- (input multiply, exact a₀ operand ea=0). FloatModel.cotErr_nonneg is the reusable
-- cot_step_close precondition (factored from linear_float_sgd_descends).
#print axioms FloatModel.cotErr_nonneg
#print axioms mlp_w1_grad_close
#print axioms mlp_input_loss_differentiableAt
#print axioms mlp_input_loss_gradAt
#print axioms mlp_input_logit_drift
#print axioms mlp_input_loss_grad_lipschitz
#print axioms mlp_input_sgd_descends
-- The descent program reaches the Chapter-4 CNN (SgdDescentCnn.lean):
-- the three genuinely-new ingredient families beyond the MLP. (1) The
-- POOL SELECTION MARGIN: MaxPool2MarginQ δ (pairwise window gaps > 2δ)
-- is the quantitative form of MaxPool2Smooth — a δ-perturbation can
-- neither tie nor reorder a window, so the argmax (isArgmax_iff), the
-- smoothness (smooth_of_close), and the pool's entire pdiv3 routing
-- pattern (pdiv3_eq) FREEZE, exactly as the ReLU margins freeze the
-- masks. (2) The pool passes drift through unamplified: 1-Lipschitz per
-- entry (max4_sub_abs_le) and ℓ1-contractive across entries — the 2×2
-- stride-2 windows partition the input (sum_window_cells). (3) Conv is a
-- dense layer with weight sharing: affine in the kernel
-- (conv2d_kernel_sub), per-entry drift ≤ a·(slab ℓ1) ≤ a·‖e‖₁, and the
-- ℓ1 drift picks up the spatial multiplicity (h·w)·a·‖e‖₁
-- (conv2d_kernel_drift_sum). The dense head below the pool is already
-- covered by the MLP descent theorems at the pooled activation.
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
#print axioms conv2d_eq_convPad
#print axioms abs_convPad_le
#print axioms k4Idx_inj
#print axioms sum_abs_kernel_slab_le
#print axioms sum_abs_k4
#print axioms conv2d_kernel_sub
#print axioms conv2d_kernel_drift
#print axioms conv2d_kernel_drift_total
#print axioms conv2d_kernel_drift_sum
-- The conv2-layer rung, assembled (SgdDescentCnn.lean). The gradient
-- closed form chains the EXISTING fold (conv_total_loss_grad_fold,
-- generic in G) with the pool-collapsed head gradient: pdiv through the
-- relu picks up the mask (pool_relu_input_grad), the pool's pdiv3
-- collapses to the single argmax term at a smooth point, and the
-- 3-dense head above the pool is one pdiv_comp hop on ce_head2
-- (ce_head3_input_grad). The conv weight Jacobian is extracted from the
-- certified VJP by contracting .correct against a basis vector
-- (conv2d_weight_pdiv) — its closed form is POINT-FREE (conv is affine
-- in the kernel), so along a step segment only the head gradient moves.
-- Under the four margins at the step radius — relu₂ (a·D), pool
-- selection (MaxPool2MarginQ (a·D), POST-relu), relu₃, relu₄ — every
-- mask and the pool's routing pattern freeze
-- (cnn_margin{2,3,4}_keeps_offkink, cnn_postrelu_close_seg), the
-- difference collapses to the softmax drift
-- (cnn_conv2_loss_grad_lipschitz, constant explicit with the
-- weight-sharing multiplicity ((2h)·(2w))² — vs the MLP's width
-- factors), and one inexact SGD step on the second conv kernel provably
-- decreases the cross-entropy loss by ≥ lr·‖∇L‖₂²/2
-- (cnn_conv2_sgd_descends). The bias rungs follow below.
#print axioms ce_head3_input_grad
#print axioms pool_relu_input_grad
#print axioms conv2d_weight_pdiv
#print axioms conv2d_weight_pdiv_row_l1
#print axioms cnn_conv2_loss_differentiableAt
#print axioms cnn_conv2_loss_gradAt
#print axioms cnn_pool_l1_drift
#print axioms cnn_conv2_logit_drift
#print axioms cnn_margin2_keeps_offkink
#print axioms cnn_margin3_keeps_offkink
#print axioms cnn_margin4_keeps_offkink
#print axioms head3_sum_drift
#print axioms cnn_conv2_loss_grad_lipschitz
#print axioms cnn_conv2_sgd_descends
-- The conv1 rung (SgdDescentCnn.lean): the deepest descent statement.
-- The new mathematics is conv AS A FUNCTION OF ITS INPUT: conv is
-- linear there, and its Jacobian entry is a single kernel tap
-- (convTap, the input-side peer of convPad), extracted point-free from
-- the certified input-VJP (conv2d_has_vjp3) by contracting .correct
-- against a basis cotangent (conv2d_input_pdiv3 /
-- conv2d_flat_input_pdiv). The ℓ1 operator factor of a conv crossing
-- is LOCALITY, not a spatial count: each input entry feeds at most
-- oc·kH·kW outputs and each output reads at most ic·kH·kW inputs
-- (convTap_out_l1 / convTap_in_l1, via the kernel-offset indicator
-- expansion abs_convTap_expand and the pinned-sum bound sum_pinned_le);
-- the same locality bounds the drift (conv2d_input_entry_drift,
-- conv2d_input_l1_drift via abs_convPad_sub_expand). The conv1 chain
-- crosses relu₁, conv2-as-input, relu₂, the pool, and the 3-dense head:
-- FIVE margins at the step radius freeze every mask and the pool
-- routing (cnn1_margin{1,2,3,4}_keeps_offkink,
-- cnn1_postrelu2_close_seg), the head gradient collapses through the
-- point-free taps (cnn1_pool_head_input_grad, cnn_conv1_loss_gradAt),
-- the segment-Lipschitz constant is explicit with BOTH multiplicities
-- — conv1 weight sharing ((2h)·(2w))² and conv2 locality (c·kH·kW)²·w₂²
-- (cnn_conv1_loss_grad_lipschitz) — and one inexact SGD step on the
-- first conv kernel provably decreases the loss by ≥ lr·‖∇L‖₂²/2
-- (cnn_conv1_sgd_descends). Every conv kernel of the Chapter-4 CNN now
-- has a proven descent statement; the bias rungs follow below.
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
#print axioms cnn_conv1_loss_grad_lipschitz
#print axioms cnn_conv1_sgd_descends
-- The conv BIAS rungs (SgdDescentCnn.lean): conv2d is affine in its
-- bias with the simplest possible Jacobian — a Kronecker channel
-- indicator (conv2d_bias_pdiv, extracted from the certified bias VJP
-- conv2d_bias_grad_has_vjp by contracting .correct against a basis
-- vector, exactly as conv2d_weight_pdiv). The per-entry drift is
-- exactly |e o| (conv2d_bias_sub — no input bound a, no kernel mass);
-- the ℓ1 drift picks up the spatial multiplicity h·w
-- (conv2d_flat_bias_drift_sum — one bias entry feeds a whole channel).
-- Everything downstream is the kernel-rung argument verbatim with the
-- conv stage's a·D radii replaced by the bare D: four margins for the
-- conv2 bias (cnnb2_margin{2,3,4}_keeps_offkink + the pool), five for
-- the conv1 bias (cnnb1_margin{1,2,3,4}_keeps_offkink + the pool), the
-- head gradients reused verbatim (pool_relu_input_grad /
-- cnn1_pool_head_input_grad), the segment-Lipschitz constants the
-- kernel constants with a² ↦ 1, and one inexact SGD step on either
-- conv bias provably decreases the loss by ≥ lr·‖∇L‖₂²/2
-- (cnn_conv2_bias_sgd_descends / cnn_conv1_bias_sgd_descends). With
-- these, EVERY parameter of the Chapter-4 CNN — both conv kernels,
-- both conv biases, and the dense head — has a proven descent
-- statement.
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
-- Adam/AdamW optimizer step over ℝ (Phase 3a, vit_train_to_vit_verified.md): the
-- emitted-update spec (adamWParam_apply), denominator well-definedness, and the
-- second-moment nonneg invariant. Faithfulness/well-definedness only — no descent.
#print axioms adamVNext_nonneg
#print axioms adam_denom_pos
#print axioms adamWParam_apply
#print axioms adamWParam_wd_zero
-- Phase 3b: AdamW render-close (den-level faithfulness) — the emitted weight/bias
-- update = adamWScalar of the certified ∂/∂θ Jacobian · denoted softmax-CE cotangent
-- (the sgdW_descends_certified_grad analogue, optimizer = AdamW). Faithfulness only.
#print axioms Proofs.adamWParam_eq_scalar
#print axioms StableHLO.adamW_certified_grad
#print axioms StableHLO.adamB_certified_grad

-- EfficientNet backward-graph faithfulness (den-level): fan-in bricks, per-op
-- backward ops, the whole per-example MBConv residual block, and the batched
-- per-stage backward primitives (true-batch-norm + batched conv/depthwise).
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
-- ConvNeXt backward-graph faithfulness (den-level), per-example (batch-1): the
-- block-body backward graph, the identity/residual block capstone, and the
-- LN+2×2/s2 downsample capstone. LayerNorm is per-example separable (= bnForward
-- on the feature axis), so no batched machinery; the LN backward routes through
-- bnBack_faithful_fn (layerNorm_has_vjp ≡ bn_has_vjp).
#print axioms StableHLO.cnxBlockBodyBackGraph_faithful
#print axioms StableHLO.cnxResidBlockBackGraph_faithful
#print axioms StableHLO.cnxDownBackGraph_faithful
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

-- EfficientNet DOWNSAMPLE MBConv body backward-graph faithfulness: the stride-2
-- peer of the identity-residual MBConv above. The downsample block changes
-- spatial/channels so it has NO residual skip — the "block" is the body alone:
-- projB ∘ seB ∘ dwbsSB ∘ cbsB, where the depthwise stage is STRIDED (dwbsSB).
-- Built on the NEW stride-2 batched-depthwise VJP `depthwiseStridedBackBatched`
-- (the stride-2 / depthwise analog of `convStridedBackBatched`). EfficientNet uses
-- swish (a global VJP), so this stays in the clean global HasVJP/vjp_comp form.
#print axioms StableHLO.depthwiseStridedBackBatched_faithful
-- Batched strided depthwise → bn → swish stage backward graph.
#print axioms StableHLO.dwbsSBackBatchedGraph_faithful
-- Capstone: the batched EfficientNet downsample MBConv body backward graph.
#print axioms StableHLO.mbDownBodyBackBatchedGraph_faithful

-- MobileNetV2 backward-graph faithfulness (den-level): the relu6 (_at) peer of the
-- EfficientNet block above. relu6's TWO-SIDED kink gives only a pointwise VJP
-- (relu6_has_vjp_at), so the stages/body/capstone are _at-form with the relu6
-- smoothness hypotheses threaded; the per-op relu6 back token is `.selectMid`. The
-- block is the MBConv body MINUS squeeze-excite, with the same linear-bottleneck projB.
-- Batched relu6 stage backward graphs (conv/depthwise → bn → relu6).
#print axioms StableHLO.cbrBackBatchedGraph_faithful
#print axioms StableHLO.dwbrBackBatchedGraph_faithful
-- The SE-less inverted-residual body backward graph (projB ∘ dwbrB ∘ cbrB).
#print axioms StableHLO.mnv2BodyBackBatchedGraph_faithful
-- Capstone: the whole batched MobileNetV2 inverted-residual block backward graph.
#print axioms StableHLO.mnv2ResidBlockBackBatchedGraph_faithful
-- DOWNSAMPLE (stride-2) inverted-residual peer of the block above: the depthwise is
-- STRIDED (dwbrBstrided) and there is NO residual skip (spatial/channels change),
-- so the "block" is the body alone: projB ∘ dwbrBstrided ∘ cbrB. Built on the
-- stride-2 batched-depthwise VJP `depthwiseStridedBackBatched`; the relu6 back token
-- stays `.selectMid` (_at-form, smoothness threaded). The relu6 peer of the
-- EfficientNet `mbDownBodyBackBatchedGraph_faithful`.
-- Batched strided depthwise → bn → relu6 stage backward graph.
#print axioms StableHLO.dwbrBstridedBackBatchedGraph_faithful
-- Capstone: the batched MobileNetV2 downsample inverted-residual body backward graph.
#print axioms StableHLO.mnv2DownBodyBackBatchedGraph_faithful

-- ResNet-34 backward-graph faithfulness (den-level): the relu (_at) peer of the two
-- blocks above, with the structural twist of an OUTER post-residual relu. The body
-- is conv-bn ∘ conv-bn-relu and the block is `relu ∘ residual(F)`; relu's one-sided
-- kink gives only a pointwise VJP (relu_has_vjp_at), so the stage/body/capstone are
-- _at-form with TWO relu smoothness families threaded (body mid-relu + outer relu).
-- The per-op relu back token is `.selectPos`.
-- Batched conv-bn-relu stage backward graph.
#print axioms StableHLO.cbReluBackBatchedGraph_faithful
-- The basic-block body backward graph (projB ∘ cbReluB).
#print axioms StableHLO.r34BodyBackBatchedGraph_faithful
-- Capstone: the whole batched ResNet-34 identity basic block backward graph
-- (residual fan-in + OUTER relu).
#print axioms StableHLO.r34BasicBlockBackBatchedGraph_faithful

-- ResNet-34 DOWNSAMPLE/STRIDED basic block backward-graph faithfulness: the
-- `relu ∘ residualProj(proj, F_s)` peer of the identity block above, with a STRIDED
-- conv1 in the body + a STRIDED conv-bn projection skip (both backward paths
-- nontrivial). Built on the NEW stride-2 batched-conv VJP `convStridedBackBatched`
-- (the stride-2 analog of `convBackBatched`).
#print axioms StableHLO.convStridedBackBatched_faithful
-- Capstone: the whole batched ResNet-34 downsample basic block backward graph
-- (PROJECTED-residual fan-in [body + strided-proj skip] + OUTER relu).
#print axioms StableHLO.r34DownBlockBackBatchedGraph_faithful

-- ViT whole-block backward-graph faithfulness (den-level, per-token Mat-VJP,
-- heads = 1): the transformer peer of the conv nets' *BackB0 capstones. The MHSA
-- backward is collapsed at heads = 1 to the plain three-way dense fan-in over the
-- proven sdpa_back_{Q,K,V} (mhsa_backward_collapse: tied to mhsa_has_vjp_mat by
-- VJP determinism + the qkv-stack / column-slab one-head collapse + the inner-d ↔
-- 1*d sdpa-back width bridges), expressed in ViTChainClose's vitCotD{Q,K,V} render
-- forms (mhsaBackGraph_faithful). The MLP + attention sublayer graphs are residual
-- fan-ins of an LN-back over the body back (mlpSublayerBackGraph_faithful,
-- attnSublayerBackGraph_faithful), assembled into the whole block via the
-- vjpMat_comp wiring block.backward A dY = attn.backward A (mlp.backward (attn A) dY).
#print axioms StableHLO.mhsa_backward_collapse
#print axioms StableHLO.mhsaBackGraph_faithful
#print axioms StableHLO.mlpSublayerBackGraph_faithful
#print axioms StableHLO.attnSublayerBackGraph_faithful
#print axioms StableHLO.transformerBlockBackGraph_faithful

-- ViT whole-block backward-graph faithfulness, lifted to GENERAL MULTI-HEAD
-- (heads = hm1 + 1, ≥1; subsumes the production heads = 3). The MHSA backward
-- collapses to the per-head fan-in: each head slices the dense Q/K/V projections +
-- the Wo-back cotangent to its d-columns, runs the proven sdpa_back_{Q,K,V} at
-- d_head, pads back to D and sums over heads, with the qkv-stack dense-back fanning
-- into denseRowBack Wq/Wk/Wv (mhsa_backward_collapseMH ↔ mhsa_has_vjp_mat by VJP
-- determinism + the general-heads colSlab/qkv split). The render graph mirrors this
-- per head via headSliceF/headPadF + headsSumG (mhsaBackGraphMH_faithful); the
-- sublayer/block wrappers are head-agnostic at (hm1+1)*d (transformerBlockBackGraphMH_faithful
-- = the multi-head capstone: den graph = transformerBlock_has_vjp_mat.backward, scalar LN).
#print axioms StableHLO.mhsa_backward_collapseMH
#print axioms StableHLO.mhsaBackGraphMH_faithful
#print axioms StableHLO.mlpSublayerBackGraph_faithfulMH
#print axioms StableHLO.attnSublayerBackGraphMH_faithful
#print axioms StableHLO.transformerBlockBackGraphMH_faithful

-- ViT whole-block backward-graph faithfulness at the FULL PRODUCTION config:
-- general MULTI-HEAD (heads = hm1 + 1) + VECTOR-[D] LayerNorm γ/β per LN site
-- (the committed verified_mlir/vit_train_step.mlir ViT-Tiny form). The MHSA + MLP-body
-- backwards are LN-agnostic and REUSED verbatim from the MH scalar capstone; the only
-- new piece is the vec-LN LN-back fragment: layerNormVec = (+βv) ∘ layerScale γv ∘ LN(1,0),
-- whose input-VJP collapses (bias backward = id) to the normalize-only (γ=1) backward of
-- the rowwise-layerScale-γv cotangent — rendered as lnRowBack(γ=1) ∘ rowScaleF γv
-- (rowVecLNBack_eq_backward, the crux bridge). Sublayers + whole block then mirror the
-- scalar/MH templates, re-targeting the _has_vjp_mat to the …V… (vec-LN) versions
-- (transformerBlockVBackGraphMH_faithful = the production-parity capstone:
-- den graph = transformerBlockV_has_vjp_mat.backward, vector LN + general heads).
#print axioms StableHLO.rowVecLNBack_eq_backward
#print axioms StableHLO.mlpSublayerVBackGraph_faithfulMH
#print axioms StableHLO.attnSublayerVBackGraphMH_faithful
#print axioms StableHLO.transformerBlockVBackGraphMH_faithful

-- ViT WHOLE-NET backward-graph faithfulness — the depth-k, multi-head, vector-LN
-- production capstone. The backward analogue of vitFwdGraphKMHV_faithful: a reverse-
-- composed backward graph (classifier-back → final-vec-LN-back → depth-k tower-back
-- reverse fold → patchEmbed-back) whose denotation IS the proven whole-net VJP
-- vitForwardKV_has_vjp.backward (ViTDepthK), at every input image + cotangent, every
-- depth k. Stage 1 (classifierBackGraph_faithful): clsPadF (dotOut Wcls dy) =
-- classifier_flat_has_vjp.backward. Stage 2 (finalLNBackGraph_faithful): the vec-LN
-- LN-back fragment over N+1 tokens, bridged. Stage 3 (vitBodyBackGraphKMHV_den): the
-- depth-k reverse fold of transformerBlockVBackGraphMHP, by induction on k chaining the
-- bundled per-block faithful — the backward analogue of vitBodyGraphKMHV_den. Stage 4
-- (patchEmbedBack_faithful + patchEmbedBackGraph_faithful): the new patchEmbedBack SHlo
-- token (strided-patchify conv input-VJP) = patchEmbed_flat_has_vjp.backward, routed
-- through the generic `batched` Raw/Tok tag. Stage 5 (vitNetBackGraph_faithful): the
-- whole-net capstone, composing the four stages by the three vjp_comp backward rules.
#print axioms StableHLO.classifierBackGraph_faithful
#print axioms StableHLO.finalLNBackGraph_faithful
#print axioms StableHLO.transformerBlockVBackGraphMHP_faithful
#print axioms StableHLO.vitBodyBackGraphKMHV_den
#print axioms StableHLO.patchEmbedBack_faithful
#print axioms StableHLO.patchEmbedBackGraph_faithful
#print axioms StableHLO.vitNetBackGraph_faithful
-- ViT-Tiny §1 FOLD (ViTFaithfulPoC) — each emitted param-SGD op `den` = the certified loss-descent
-- step, for every parameter family of the depth-12 ViT-Tiny train step. veclnGammaSgd (vector-[D] LN
-- γ, the Σ_tokens dy·x̂ reduce) → vit_render_veclngamma_certified; rowDenseWeightSgd/rowDenseBiasSgd
-- (per-token attn/MLP dense W/b) → vit_render_rowdense{W,b}_certified, and the SAME bias op against the
-- vector-LN β forward (vit_render_veclnbeta_certified); patchEmbedWeightSgd/patchEmbedBiasSgd (the
-- 16×16/s16 patchify conv W/b — vit HAS the patch-weight VJP, so NO even-kernel gap) →
-- vit_render_patch{W,b}_certified; posEmbedSgd (identity pos-Jacobian) → vit_render_pos_certified; the
-- classifier head reuses Cifar8PoC.dense{W,B}_den. One-/few-line delegations; the ops iree-compile
-- (whole train step 366 KB vmfb). Covers every param family → vit is the first net with ZERO param gaps.
#print axioms Proofs.ViTPoC.veclnGammaSgd_den
#print axioms Proofs.ViTPoC.rowDenseWeightSgd_den
#print axioms Proofs.ViTPoC.rowDenseBiasSgd_den
#print axioms Proofs.ViTPoC.rowDenseBiasSgd_den_lnbeta
#print axioms Proofs.ViTPoC.patchEmbedWeightSgd_den
#print axioms Proofs.ViTPoC.patchEmbedBiasSgd_den
#print axioms Proofs.ViTPoC.posEmbedSgd_den
#print axioms Proofs.ViTPoC.headW_den
#print axioms Proofs.ViTPoC.headB_den
-- ViT-Tiny §1a TIE — per-block (ViTTiePoC). Every one of a vector-LN transformer block's 16 params,
-- fed the cotangent the REAL backward chain delivers at its site, den=certified (θ - lr·certified·chain-cot).
-- The new content vs every prior net: TWO residual fan-ins per block (MLP residual vitCotHV, attention
-- residual vitCotXinV) + the three-way fan-in at LN₁ (Q/K/V dense-backs SUM in vitCotLn1) + the per-head
-- SDPA backward pinned to the audited sdpa_back_{Q,K,V} (vitCotD{Q,K,V}). Pure thread + fan-in: every
-- conjunct delegates to a §1-fold ViTPoC.*_den generic at the chain cotangent — ZERO new ops/bridges.
-- Single-head vector-LN representative (heads=1); the multi-head/depth-12 thread (per-head headSlice/
-- headPad backs summed) is the remaining step, the analogue of mnv2's reduced→full.
#print axioms Proofs.ViTTiePoC.vit_block_tiedV
-- ViT-Tiny §1a TIE — whole-net thread (2-block vector-LN representative). vit_block_tiedAtV: the
-- per-block tie at the block INPUT, recomputing the 11 saved activations from xin (the vitBlockSpelledV
-- let-chain) — the vit peer of cnxBlockTiedAt. vit_net_tiedV: BOTH blocks of the 2-block ViT tied
-- through the REAL forward (ib1 → vitBlockFwdOV → ib2 → b2out) + the inter-block cotangent fan-in
-- (block 2's vitBlockCotInAtV = the attention-residual fan-in vitCotXinV is block 1's dyOut; the
-- final-LN input-VJP of the classifier-back vitCotB2outV is block 2's dyOut) — the convnext
-- cnx_net_tied_certified pattern at the single-head representative. Single-head/2-block; the
-- multi-head/depth-12 promotion (per-head headSlice/headPad backs summed) is the remaining step.
#print axioms Proofs.ViTTiePoC.vit_block_tiedAtV
#print axioms Proofs.ViTTiePoC.vit_net_tiedV
-- ViT-Tiny §1a TIE — MULTI-HEAD promotion (ViTMultiHeadChain + ViTTiePoC). The committed render is
-- multi-head (heads=3, d_head=64 → D=192); only the SDPA-internal backward dAtt → dQ/dK/dV changes
-- (the out-proj Wo, LN₂, the MLP are head-agnostic). vitCotD{Q,K,V}mh_eq: the rendered slice → per-head
-- SDPA-back → pad chain IS Mat.flatten (Σ_h headPadMat h (sdpa_back_{Q,K,V} d_head Q_h K_h V_h dOut_h))
-- — the concat of the audited per-head SDPA backwards (each via the single-head pin
-- vitCotD{Q,K,V}_eq_sdpa_back_{Q,K,V} at d_head, slid through the slice/pad/flatten-sum bridges).
#print axioms Proofs.vitCotDQmh_eq
#print axioms Proofs.vitCotDKmh_eq
#print axioms Proofs.vitCotDVmh_eq
-- The multi-head per-block tie (vit_block_tiedMHV = vit_block_tiedV with the SDPA cots swapped to the
-- …mh cots; the 16 conjuncts delegate to the head-agnostic ViTPoC.*_den generics unchanged) and the
-- depth-12 whole-net thread (vit_net_tiedMHV: 12 blocks tied through the real multi-head forward
-- ib1 → … → ib12 → b12out + the loss-driven backward, the convnext 18-block cnx_net_tied_certified
-- pattern at the committed ViT-Tiny config @vit_net_tiedMHV 196 3 64 768 10). @[irreducible] wrappers.
#print axioms Proofs.ViTTiePoC.vit_block_tiedMHV
#print axioms Proofs.ViTTiePoC.vit_block_tiedAtMHV
#print axioms Proofs.ViTTiePoC.vit_net_tiedMHV
-- ViT-Tiny §1a TIE — the ALL-200-PARAMS capstone (vit_net_tied_certified, committed ViT-Tiny config:
-- 3 heads, d_head=64, D=192, N=196, mlpDim=768, 10 classes, 16×16 patches). Threads the REAL forward
-- (patchEmbed → 12 multi-head vector-LN blocks → final vector-LN → CLS-slice → dense head) + the
-- loss-driven backward, and bundles EVERY param op den=certified: the 12 blocks' 192 params, the
-- final-LN γ/β (vitFinalLNTied), the classifier Wcls/bcls (vitHeadTied), and the patch-embed
-- wConv/bConv/cls/pos (vitEmbedTied; cls via vit_cls_den — its row-0 batch slice IS cls_token_grad).
-- 200/200 — the FIRST net with zero param gaps (vit has the patch-weight VJP cert). The vit peer of
-- convnext's cnx_net_tied_certified; this CLOSES the last Tier-3 §1a tie.
#print axioms Proofs.ViTTiePoC.vit_cls_den
#print axioms Proofs.ViTTiePoC.vit_finalLN_tied
#print axioms Proofs.ViTTiePoC.vit_head_tied
#print axioms Proofs.ViTTiePoC.vit_embed_tied
#print axioms Proofs.ViTTiePoC.vit_net_tied_certified
