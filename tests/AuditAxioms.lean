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
