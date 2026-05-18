import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention

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

-- Codegen smooth-point bridge theorems (MLP.lean)
#print axioms relu_codegen_matches_canonical
#print axioms relu_canonical_diagonal
