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
import LeanMlir.Proofs.IR

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
#print axioms layerScale_has_vjp_correct
#print axioms convnext_has_vjp_at_correct
#print axioms sigmoid_has_vjp
#print axioms efficientnet_has_vjp_at_correct

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
