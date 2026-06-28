import Lake
open Lake DSL

package «lean4-mlir» where
  version := v!"0.6.1"
  buildType := .release

-- doc-gen4 is a conditional dependency only activated when the CI
-- (or a local user) passes `-Kenv=dev`. Without that flag, this
-- require is inert, so normal `lake build` invocations don't pull it
-- in. Standard pattern used by Mathlib, PrimeNumberTheoremAnd,
-- Carleson, FLT, etc.
--
-- CRITICAL: Mathlib must be required LAST so its version constraints
-- for shared transitive deps (plausible, etc.) take precedence over
-- doc-gen4's. Otherwise `lake exe cache get` can't find matching
-- Mathlib archives and the build fails.
meta if get_config? env = some "dev" then
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "main"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.31.0"

-- The everything-root: codegen + IREE FFI + trainers' shared modules +
-- the proof suite, all pulled in transitively via `LeanMlir.lean`.
-- `lake build LeanMlir` type-checks the whole repo.
lean_lib «LeanMlir» where
  roots := #[`LeanMlir]

-- Scoped targets, so CI and contributors can build one slice without the
-- rest. `LeanMlir` above is all-or-nothing; these split it along the same
-- seam the codebase already has (the proof suite imports only Mathlib —
-- never the codegen — so `Proofs` skips Types/Spec/MlirCodegen/Train).

/-- **`lake build Proofs`** — the VJP proof suite only. Roots are the apex
    modules (same set `LeanMlir.lean` aggregates); their transitive imports
    cover every proof file. Default target: a bare `lake build` now
    type-checks the formalization instead of doing nothing. -/
@[default_target]
lean_lib «Proofs» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Attention, `LeanMlir.Proofs.CNN,
             `LeanMlir.Proofs.Depthwise, `LeanMlir.Proofs.MobileNetV2,
             `LeanMlir.Proofs.ConvNeXt, `LeanMlir.Proofs.EfficientNet,
             -- Chapter-4 MNIST 2D CNN (no BN): conditional whole-net VJP
             -- + a concrete instance with every smoothness hyp discharged.
             `LeanMlir.Proofs.MnistCNN,
             -- Nonzero-Jacobian seal (planning/whole_network_backward.md Item B): the
             -- generic "one nonzero Jacobian entry ⇒ non-trivial backward" bridge.
             `LeanMlir.Proofs.JacobianSeal,
             -- Item B2: the seal discharged at the live MobileNetV2 witness —
             -- `fderiv ℝ forward 0 ≠ 0` ⇒ non-trivial whole-net backward (level 3).
             `LeanMlir.Proofs.MobileNetV2JacobianSeal,
             -- Chapter-6 ResNet Milestone B: stride-2 SAME convolution (the hard
             -- new downsampling op) = decimate ∘ stride-1 conv, with its input-VJP.
             `LeanMlir.Proofs.StridedConv,
             -- Chapter-6 ResNet Milestone B: the deep-block chain (a list of
             -- same-type residual blocks composes to one VJP) — 16-block depth.
             `LeanMlir.Proofs.ResNet34,
             -- Chapter-6 ResNet Milestone B8: per-channel BatchNorm (block-diagonal
             -- VJP via a per-row generalization of `rowwise_has_vjp_mat`).
             `LeanMlir.Proofs.PerChannelBN,
             -- Limit-D strengthening: the 224×224 live ResNet-34 whole-net VJP with
             -- the three downsample projection convs generalized to ARBITRARY kernels
             -- (the β-positivity discharge is weight-independent). 3-axiom clean.
             `LeanMlir.Proofs.ResNet34LiveGeneric,
             -- opt-in Mathlib.Matrix interop; not imported by the suite,
             -- listed here so CI keeps it green.
             `LeanMlir.Proofs.MatBridge,
             -- denoted StableHLO-subset IR (Phase 0a/0b spike); bridges the
             -- emitted backward graph to the proven HasVJP.backward.
             `LeanMlir.Proofs.IR,
             -- R4 printer-faithfulness Stage A (ch 2): StableHLO-subset AST +
             -- denotation `den` proven to match the linear train-step math.
             `LeanMlir.Proofs.StableHLO,
             -- R4 syntactic core: op-graph serialization round-trip
             -- (parse (toToks (skel a)) = a).
             `LeanMlir.Proofs.StableHLOParse,
             -- R4 syntactic LEXER numeric keystone: decimal Nat⟷String
             -- round-trip (parseNat (toString n) = n), the load-bearing
             -- first rung of text→token faithfulness.
             `LeanMlir.Proofs.StableHLOLex,
             -- M1 (planning/verified_train_step.md): the linear train step bundled
             -- into one SGD-on-certified-softmax-CE-gradient theorem.
             `LeanMlir.Proofs.LinearTrainStep,
             -- M2: the MLP per-layer parameter-gradient assembly (Crux A).
             `LeanMlir.Proofs.MlpTrainStep,
             -- M3: the CNN convolution parameter-gradient bridges.
             `LeanMlir.Proofs.CnnTrainStep,
             -- MLP render half: the train-step text as a name-threaded render of the
             -- proven forward graphs (multi-intermediate generalization).
             `LeanMlir.Proofs.MlpRender,
             -- CNN render half: the CNN train-step text rendered from `cnnFwdGraph`,
             -- with flat→NCHW reshape glue bridging the conv param-grad tail.
             `LeanMlir.Proofs.CnnRender,
             -- ResNet-34 render half: the full [3,4,6,3] train step (146 params) rendered
             -- from the verified AST — stem/16 residual blocks/GAP/dense, residual cotangent-
             -- sums at the skip merges; regenerates verified_mlir/resnet34_train_step.mlir.
             `LeanMlir.Proofs.ResNet34Render,
             -- CIFAR-BN close: the per-channel BN scale/shift (dγ, dβ) param-grad
             -- bridges — the affine BN analogue of `bias_grad_bridge`.
             `LeanMlir.Proofs.CifarBnClose,
             -- CNN conv-close upgrade: the conv param closes pinned to the actual
             -- backward-chain cotangent (Back3 maxpool/conv via flatDenote + relu masks).
             `LeanMlir.Proofs.CnnChainClose,
             -- Deeper (8-conv) CIFAR-CNN close: cifar8{,Bn}FwdGraph_faithful's backward
             -- peer — each conv W/b, BN γ/β, dense W/b output pinned to the actual 4-stage
             -- backward-chain cotangent (the CnnChainClose recipe + BN, two more pool stages).
             `LeanMlir.Proofs.Cifar8Close,
             -- MobileNetV2 close (Item C): the depthwise (stride-1/2) + strided-conv
             -- parameter-gradient bridges — every MobileNetV2 train-step param output
             -- certified θ − lr·(certified Jacobian · cotangent).
             `LeanMlir.Proofs.MobileNetV2Close,
             -- MobileNetV2 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph
             -- (matches the operational render's BN flavor) + faithfulness to the
             -- per-channel ℝ-forward. Prerequisite for the structured render (Item B).
             `LeanMlir.Proofs.MobileNetV2RenderPC,
             -- MobileNetV2 cotangent-chain close (Item D): the Item C conv/depthwise bridges
             -- pinned to the inverted-residual backward chain (relu6 kink + depthwise + stride-2).
             `LeanMlir.Proofs.MobileNetV2ChainClose,
             -- The cotangent pass / = ∂loss/∂θ fold: the certified per-layer conv/depthwise
             -- Jacobian contracted with ∂loss/∂(layer output) IS the total loss gradient (pdiv_comp
             -- at a smooth point). The conv analogue of mlp_hidden_total_loss_grad; program-wide.
             `LeanMlir.Proofs.ConvLossFold,
             -- EfficientNet-B0 close (Item C): a FREE close — every param family reuses an
             -- existing bridge (5×5 depthwise pinned; batch-norm γ/β = per-channel BN at m=N·h·w;
             -- SE squeeze/excite are dense → M2). No new VJP.
             `LeanMlir.Proofs.EfficientNetClose,
             -- ResNet-34 close (Item C): a FREE close — every r34 param family certified
             -- by an existing bridge (the 7×7 stem + 3×3 strided projection pinned to the
             -- generic strided conv W/b bridges; no new VJP).
             `LeanMlir.Proofs.ResNet34Close,
             -- ResNet-34 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph (full
             -- 16-block [3,4,6,3] net, 7×7 stem, maxpool) + per-block + whole-net faithfulness.
             `LeanMlir.Proofs.ResNet34RenderPC,
             -- ResNet-34 cotangent-chain close (Item D): the Item C conv bridges pinned to the
             -- cotangent the backward chain delivers (id/downsample block + maxpool-back stem).
             `LeanMlir.Proofs.ResNet34ChainClose,
             -- ConvNeXt close (Item C): mostly reuse (7×7 depthwise pinned to the generic
             -- bridges) + the two genuinely-new families — layer-scale γ (dγ = x⊙dy) and
             -- scalar-LN γ/β (the Vec-1 embedding bridging bn_grad_gamma/beta).
             `LeanMlir.Proofs.ConvNeXtClose,
             -- ViT close (Item A): the distinct-param 2-block ViT forward (vitForward2 +
             -- whole-net VJP) and the heads=1 token forward graph + faithfulness
             -- (den vitFwdGraph = vitForward2 via mhsa_layer_one_head).
             `LeanMlir.Proofs.ViTFwdGraph,
             -- ViT close (Item C): the per-token dense W/b family (row-lifted M2
             -- outer product), row-lifted scalar-LN γ/β, pos-embed identity, CLS
             -- masked-gather — every representative-ViT param family except the
             -- patch conv certified.
             `LeanMlir.Proofs.ViTClose,
             -- ViT cotangent-chain close (Item D): the Item C bridges pinned to the
             -- attention-block backward chain (SDPA matmul chain = the proven
             -- sdpa_back_{Q,K,V} closed forms; the Q/K/V three-way fan-in at LN1).
             `LeanMlir.Proofs.ViTChainClose,
             -- ViT scaling pass (vector-[D] LN): layerNormVec block + vitForward2V
             -- whole-net VJP + the rowScaleF/rowBiasF token graph + faithfulness +
             -- the per-channel gamma/beta param bridges.
             `LeanMlir.Proofs.ViTVecLN,
             -- ViT scaling pass (multi-head + depth-k): headSliceF/headPadF tokens,
             -- mhsa at general heads, then the distinct-param depth-k tower
             -- (vitForwardKV). ViTDepthK imports ViTMultiHead, covering both.
             `LeanMlir.Proofs.ViTDepthK,
             -- ViT multi-head backward cotangents: the per-head SDPA backward the real
             -- chain delivers at the Q/K/V dense outputs (Σ_h pad ∘ vitCotD{Q,K,V}(d_head)
             -- ∘ slice), pinned to the audited sdpa_back_{Q,K,V} (vitCotD{Q,K,V}mh_eq).
             -- The multi-head/depth-12 tie's substantive build (mnv2 reduced→full).
             `LeanMlir.Proofs.ViTMultiHeadChain,
             -- EfficientNet-B0 at full depth (16 distinct MBConv blocks, true BN+SE):
             -- batched forward graph + whole-net VJP. Imports the EfficientNet
             -- RenderPC + ChainClose modules, covering all three.
             `LeanMlir.Proofs.EfficientNetFullB0,
             -- Full ConvNeXt-T [3,3,9,3]: forward graph + faithfulness + whole-net
             -- VJP. Imports ConvNeXtChainClose, covering both.
             `LeanMlir.Proofs.ConvNeXtFullT,
             -- Paper-spec full MobileNetV2 (all 17 [t,c,n,s] bottlenecks): forward
             -- graph + faithfulness.
             `LeanMlir.Proofs.MobileNetV2FullPaper,
             -- ℝ→Float32 bridge, Tier 1: standard-model rounding (hypothesis-style,
             -- no axioms) + forward error bounds for the linear/MLP nets
             -- (dot/dense budgets, ReLU exact-in-float Lipschitz pass-through).
             `LeanMlir.Proofs.FloatBridge,
             -- Subnormal-floor closure (planning §2): the honest FaithfulFloatModel
             -- (relative bound on normals + the gradual-underflow absolute floor),
             -- FloatModel = its η→0 face, the BN/LN denominator stays-normal
             -- invariant (rsqrt keystone never underflows), and the residual floor
             -- proved globally negligible. Converts FloatBridge's subnormal caveat
             -- into lemmas.
             `LeanMlir.Proofs.FloatSubnormalBridge,
             -- Inexact-gradient descent over ℝ (MVT form): an η-accurate gradient
             -- oracle + segment smoothness ⇒ the SGD step still decreases the loss,
             -- with an explicit decrease. The keystone the FloatBridge budgets
             -- plug into ("close" ⇒ "still trains").
             `LeanMlir.Proofs.SgdDescent,
             -- The smoothness hypothesis DISCHARGED for the Chapter-2 linear
             -- softmax-CE loss: explicit segment-Lipschitz constant 2a²/(1−2aD)
             -- via the softmax ratio sandwich (no Hessian), and the capstone —
             -- one inexact SGD step provably decreases the cross-entropy loss.
             `LeanMlir.Proofs.SgdDescentLinear,
             -- The smoothness hypothesis discharged through the Chapter-3
             -- MLP: under quantitative ReLU margins (the step cannot flip a
             -- mask sign) the loss-of-one-layer maps get explicit
             -- segment-Lipschitz constants, and one inexact SGD step on each
             -- weight layer provably decreases the cross-entropy loss.
             `LeanMlir.Proofs.SgdDescentMlp,
             -- The descent program reaches the Chapter-4 CNN: quantitative
             -- max-pool selection margins (the argmax freezes along the step
             -- segment), pool 1-Lipschitz/ℓ1-contraction, conv kernel drift.
             `LeanMlir.Proofs.SgdDescentCnn,
             -- CIFAR-8 last-conv SGD descent: the first non-MNIST provable descent. CIFAR-8's tail
             -- (conv W₈ → relu → maxpool → 3 denses) IS cnn_conv2's architecture, so descent at the
             -- last conv (earlier 7 layers frozen) is an instance — non-vacuous lr. Full-depth descent
             -- stays open (the per-layer operator-norm product in hsmall compounds to vacuity).
             `LeanMlir.Proofs.SgdDescentCifar,
             -- ℝ→Float32 forward rounding budget for the no-BN CIFAR CNN
             -- (cnn_float_close scaled to 4 conv + 2 maxpool + 3 dense).
             `LeanMlir.Proofs.CifarFloatBridge,
             -- BN float keystone: 1/√ Lipschitz on [ε,∞) + the inverse-stddev
             -- rounding budget (rsqrt accuracy + variance error, ε-floor).
             `LeanMlir.Proofs.BnFloatBridge,
             -- residual additive fan-in float closeness (add_close / reluAdd_close)
             -- — the new structural op toward the ResNet-34 float bridge.
             `LeanMlir.Proofs.Resnet34FloatBridge,
             -- real-BN input-sensitivity (mean/var/istd/forward Lipschitz) — the
             -- per-block composition enabler (the float BN's input is perturbed).
             `LeanMlir.Proofs.BnInputBridge,
             -- first assembled ResNet block step: relu(BN(·)) at a perturbed BN
             -- input = rounding (bnForward_close_of) + input-shift (bnForward_input_close).
             `LeanMlir.Proofs.Resnet34BlockBridge,
             -- whole-net certificate backbone: FloatClose composes (moduli ∘, magnitudes
             -- thread) — the whole net is the fold of per-op budgets.
             `LeanMlir.Proofs.FloatComposeBridge,
             -- Eval-mode BN/LN as a fixed affine (planning §4): running-stats at
             -- eval = a·x+b with a=γ/√(σ²+ε), b=β−γμ/√(σ²+ε) precomputed offline
             -- (no batch reduce, no runtime rsqrt). bnEvalAffine_fold proves the
             -- eval-BN formula IS that affine; floatClose_bnEval bridges it (one
             -- mul + one add, no fan-in γ) — the deployed-forward win.
             `LeanMlir.Proofs.BnEvalFloatBridge,
             -- EfficientNet float bridge step 1: Swish/sigmoid (bounded, rounding
             -- closeness, σ is ¼-Lipschitz) — the shared smooth-activation transcendental.
             `LeanMlir.Proofs.EnetFloatBridge,
             -- ViT float bridge apex: transitively imports ViTFloatBridge (LN/GELU/MLP) and
             -- ViTAttentionFloatBridge (sdpa_close + input-sensitivity); adds the transformer
             -- block fold + projections + single/multi-head (block-diagonal & full-d MHA).
             `LeanMlir.Proofs.ViTBlockFloatBridge,
             -- A1 forward float-bridge capstones (planning/tier23…): the deeper 8-conv
             -- no-BN CIFAR via the FloatBridges.comp existential path (cifar8_floatBridges).
             `LeanMlir.Proofs.Cifar8FloatBridge,
             -- The BatchNorm FloatBridges keystone: flat/global BN (floatBridges_bn,
             -- discharges the EfficientNet MBConv hbnE/D/P) + the per-channel block-diagonal
             -- lift via FloatClose.perRowIdx (floatBridges_bnPerChannelFlat) + the network
             -- Tensor3-layout conjugation by the reassoc permutations
             -- (floatBridges_bnPerChannelTensor3). The "do-it-once" BN infra A1/A3 share.
             `LeanMlir.Proofs.BnPerChannelFloatBridge,
             -- A1 BatchNorm CIFAR forward capstone: cifarCnnBnForward float-bridges, the
             -- four per-channel BNs supplied as FloatBridges (each discharged by
             -- floatBridges_bnPerChannelTensor3). The BN-net peer of cifar8_floatBridges.
             `LeanMlir.Proofs.CifarBnFloatBridge,
             -- A3 "other side" keystone (planning/tier23… A3): the BatchNorm BACKWARD
             -- float closeness — param grads (bnBetaGrad_close / bnGammaGrad_close) + the
             -- genuinely-new three-term input gradient (bnGradInput_close) + the reusable
             -- reduction_close / sub_close' helpers. The shared backward op every deep
             -- net's gradient folds (r34/mnv2/enet/convnext LN/vit LN).
             `LeanMlir.Proofs.BnBackFloatBridge,
             -- A3 backward fold: the linear input-VJP (dx = Wᵀ·dy = bias-free dense over the
             -- transpose, reuses floatBridges_dense) + the exact ReLU-back selectPos mask
             -- (floatBridges_reluMaskBack) compose via FloatBridges.comp into a whole-net
             -- backward gradient bridge (mlpInputGrad_floatBridges) — the backward peer of
             -- cifar8_floatBridges.
             `LeanMlir.Proofs.LinBackFloatBridge,
             -- A3 CNN backward (planning/a3_backward_deepnet_assembly.md 1a/1b + cifar8 witness):
             -- the maxpool-back select_and_scatter (floatBridges_maxPoolBack, exact masked gather,
             -- modulus id) + the conv input-VJP as a reversed-kernel flatConv (floatBridges_convBack,
             -- reuses floatBridges_flatConv) compose into the whole 8-conv CIFAR input-gradient VJP
             -- (cifar8_grad_floatBridges) — the backward peer of cifar8_floatBridges.
             `LeanMlir.Proofs.CnnBackFloatBridge,
             -- A3 1c: BatchNorm BACKWARD as a composable FloatClose MAP over the cotangent
             -- (floatClose_bnBack / floatBridges_bnBack) — wraps the bnGradInput_close keystone
             -- with the real map's magnitude (bn_grad_input_abs_le) + Lipschitz-in-dy modulus
             -- (bn_grad_input_diff_abs_le, the real BN-back is linear in dy). The shared BN/LN
             -- backward op cifarBn/r34/convnext/vit gradients fold via .comp.
             `LeanMlir.Proofs.BnBackComposeBridge,
             -- A3: per-channel BatchNorm BACKWARD float-bridge (floatBridges_bnPerChannelBack) —
             -- the block-diagonal FloatClose.perRowIdx lift of floatClose_bnBack conjugated by the
             -- reassoc layout gathers, bridging the certified bnPerChannelTensor3_grad_input. The
             -- backward peer of floatBridges_bnPerChannelTensor3; discharges cifarBn_grad's BN hyps.
             `LeanMlir.Proofs.BnPerChannelBackFloatBridge,
             -- A3 r34 backward: the residual identity-block input-VJP (floatBridges_r34IdBlockBack) —
             -- relu(F(x)+x) backward = residual bF ∘ reluMaskBack, the residual-skip backward reusing
             -- FloatBridges.residual (NO new combinator; the rounded skip-add is the backward's too),
             -- bF = convFlatBack∘bnBack∘reluMaskBack∘convFlatBack∘bnBack. The dominant r34 block.
             `LeanMlir.Proofs.Resnet34BackFloatBridge,
             -- A3 strided-conv backward (r34 down-blocks + stem): flatConvStride2 = decimateFlat ∘
             -- flatConv, so its input-VJP = convFlatBack ∘ decimateBack (zero-upsample scatter then
             -- reversed-kernel conv). floatBridges_flatConvStride2Back via floatBridges_decimateBack
             -- (the scatter, exact in float / magnitude-nonincreasing by decimateIdx_injective —
             -- decimateBack IS the certified decimateFlat VJP, decimateBack_eq_vjp) .comp convBack.
             `LeanMlir.Proofs.StridedConvBackFloatBridge,
             -- A3 r34 downsample-block backward: relu(proj+body) reversed — the two-branch fan-in
             -- (FloatBridges.biPathSum, the general f(x)+g(x) rounded sum, of which the identity
             -- block's f(x)+x is the g=id case) of the projection backward and the strided body
             -- backward, both using flatConvStride2Back. Completes the r34 block set.
             `LeanMlir.Proofs.Resnet34DownBackFloatBridge,
             -- A3 r34 WHOLE-NET backward (the first Imagenette whole-net backward): gapBack (GAP
             -- VJP = broadcast÷(h·w)) + the [3,4,6,3] .comp fold (r34_grad_floatBridges) — concrete
             -- stem/GAP/maxpool/dense endpoints, the 16 blocks supplied as FloatBridges (discharged
             -- by floatBridges_r34IdBlockBack/DownBlockBack). The exact reverse of resnet34Forward.
             `LeanMlir.Proofs.Resnet34WholeBackFloatBridge,
             -- A3 r34 WHOLE-NET FORWARD (the forward peer of r34_grad_floatBridges): the missing
             -- forward op-bridges floatBridges_flatConvStride2 (stem, = flatConv read at decimateIdx)
             -- + floatBridges_gap (wraps floatClose_gap) + the [3,4,6,3] .comp fold (r34_floatBridges)
             -- — concrete stem/maxpool/GAP/dense endpoints, stem BN + 16 blocks supplied as
             -- FloatBridges. Closes the forward/backward whole-net asymmetry.
             `LeanMlir.Proofs.Resnet34WholeFloatBridge,
             -- §B integrity tie: the r34 IDENTITY-BLOCK backward float bridge targets the CERTIFIED
             -- VJP. Same-vocabulary (per-channel BN, non-batched) target rblkPC_has_vjp_at — built
             -- here, mirrors resblock_has_vjp_at — + the conv-leaf tie (convFlatBack_eq_vjp_backward,
             -- via IR.convBackDenote_eq_input_grad_formula) ⇒ r34IdBlockBack(pinned) = its .backward.
             -- b1-free (no batched↔non-batched reconciliation).
             `LeanMlir.Proofs.Resnet34BackCertifiedTie,
             -- A3 §1e depthwise backward (mnv2/enet/convnext blocker): the depthwise input-VJP is a
             -- forward depthwise conv at the spatially-reversed kernel (dwReverse, FREE reuse of
             -- floatBridges_depthwise — the depthwise twin of convBack); strided variant =
             -- depthwiseFlatBack ∘ decimateBack. floatBridges_depthwiseBack/depthwiseStride2Back.
             `LeanMlir.Proofs.DepthwiseBackFloatBridge,
             -- A3 §1e Squeeze-Excite backward (the architecturally-distinctive product-rule op):
             -- seBack(dy) = (g⊙dy) + gateBack(x⊙dy), the two-branch multiplicative fan-in
             -- (FloatBridges.biPathSum of two diagBacks) with the gate sub-net backward gateBack
             -- supplied/assembled. floatBridges_seBack + floatBridges_seGateBack (gapBack∘linBack∘
             -- swishBack∘linBack∘sigmoidBack∘broadcastBack) + floatBridges_broadcastBack (the new
             -- spatial-reduce op, via the BN-back reduction_close machinery).
             `LeanMlir.Proofs.SEBackFloatBridge,
             -- A3 Part 2 per-net backward assembly (consumes §1e). MobileNetV2: the inverted-residual
             -- body backward (expandBack∘depthwiseBack∘projectBack, depthwiseFlatBack concrete — mnv2
             -- has NO SE) + strided variant; mnv2_grad_floatBridges = whole-net fold (reverse of
             -- mobilenetv2Forward_full_pc), concrete stem/head/GAP/dense, 6 blocks supplied.
             `LeanMlir.Proofs.MobileNetV2BackFloatBridge,
             -- MobileNetV2 WHOLE-NET FORWARD (forward peer of mnv2_grad_floatBridges): the ch7 6-block
             -- per-channel render mobilenetv2Forward_full_pc. New op-bridge floatBridges_relu6 (relu6 =
             -- min(max(·,0),6) exact in float + 1-Lipschitz, mirror of floatClose_relu via the mathlib
             -- clamp lemmas); strided depthwise reuses floatBridges_depthwiseStride2Flat. mnv2Forward =
             -- the ∘ skeleton (concrete stem/head/GAP/dense, stem/head BNs + 6 invres blocks supplied);
             -- named block bridges floatBridges_invresBody{,Strided}PC discharge them (no SE).
             `LeanMlir.Proofs.MobileNetV2WholeFloatBridge,
             -- EfficientNet MBConv body backward (whole-net is batched → per-example body, peer of
             -- floatBridges_mbconvBody): expandBack∘depthwiseBack∘seBack∘projectBack — BOTH §1e ops
             -- (depthwiseFlatBack concrete + seBack supplied) land here. + the additive-skip variant.
             `LeanMlir.Proofs.EfficientNetBackFloatBridge,
             -- EfficientNet WHOLE-NET FORWARD (forward peer of efficientnetForwardB_has_vjp): stated
             -- on the ACTUAL batched efficientnetForwardB (stem→MBConv1→MBConv6-strided→MBConv6-resid
             -- →head). Each batch-separable op is FloatBridges.batchMap of its op-bridge, swish is
             -- block-diagonal at the batched index, and the 10 true-batch-norms (bnBatchLA, batch-
             -- coupled) are supplied as FloatBridges facts. New op-bridge: floatBridges_depthwiseStride2Flat
             -- (mbStrided downsample, = depthwise read at decimateIdx, peer of floatBridges_flatConvStride2).
             `LeanMlir.Proofs.EfficientNetWholeFloatBridge,
             -- EfficientNet WHOLE-NET BACKWARD (backward peer of efficientnetForwardB_floatBridges, the
             -- last entry in the 5-net × {fwd,bwd} matrix): efficientnet_grad_floatBridges = the batched
             -- .comp fold (reverse of head∘mbResid∘mbStrided∘mbNoExp∘stem) with concrete batchMap-lifted
             -- conv/GAP/dense endpoints and supplied BN/swish/block backs; + the no-exp/strided block-back
             -- bridges (mbNoExpBodyBack/mbStridedBodyBack) so every supplied block back is dischargeable.
             `LeanMlir.Proofs.EfficientNetWholeBackFloatBridge,
             -- ConvNeXt-T backward (per-example): block body backward (depthwiseBack∘lnBack∘convBack∘
             -- geluBack∘convBack∘layerScaleBack) + residual block + downsample (lnBack∘stride2Back);
             -- convnext_grad_floatBridges = whole-net [3,3,9,3] fold, concrete GAP/dense, stem/stages/
             -- downsamples supplied.
             `LeanMlir.Proofs.ConvNeXtBackFloatBridge,
             -- ConvNeXt-T WHOLE-NET FORWARD (forward peer of convnext_grad_floatBridges): the [3,3,9,3]
             -- fold of convNextForwardT. Two new op-bridges: floatBridges_layerScale (γ⊙x = diagBack γ
             -- definitionally, γ exact ⇒ es=0) + floatBridges_flatConvStride4 (the 4×4/s4 patchify stem
             -- = flatConv read at decimateOddIdx∘decimateIdx, two-decimation cousin of stride2). Named
             -- bridges floatBridges_convNextBlock (residual body) + floatBridges_convNextStageK (the
             -- depth-k stage fold, induction) + floatBridges_cnxDownW; convnextForward ∘-skeleton with
             -- stem-conv/GAP/dense concrete, stem/head LN + 4 stages + 3 downsamples supplied.
             `LeanMlir.Proofs.ConvNeXtWholeFloatBridge,
             -- The skeleton↔real-net forward ties (item #5, cosmetic polish): each whole-net forward
             -- bridge is stated on a fresh skeleton (r34Forward/mnv2Forward/convnextForward) with abstract
             -- blocks; these rfl lemmas plug the concrete blocks (idFwd/downFwd, invresBody*PC,
             -- convNextStageK/cnxDownW) into the slots ⇒ each skeleton = THE committed real ℝ-forward def.
             `LeanMlir.Proofs.WholeNetForwardTies,
             -- §B shared prerequisite: the DEPTHWISE adjoint gate (the depthwise twin of
             -- IR.convBackDenote_eq_input_grad_formula) — depthwiseConv2d (dwReverse W) 0 =
             -- depthwiseConv2d_input_grad_formula W, all dims/odd kernels, via Finset.sum_bij' on the
             -- pad supports (no Σ co) — plus the flat + strided depthwise leaf ties (depthwiseFlatBack
             -- = certified depthwise input-VJP). Unblocks the convnext/mnv2/enet §B ties.
             `LeanMlir.Proofs.DepthwiseBackCertifiedTie,
             -- §B integrity tie (convnext): cnxBlockBodyBack(pinned LN/gelu/layerScale backs) = the
             -- certified convNextBlockBody_has_vjp.backward — depthwise gate + 1×1 conv leaves + rfl;
             -- plus the residual-wrapped block tie. b1-free.
             `LeanMlir.Proofs.ConvNeXtBackCertifiedTie,
             -- §B integrity tie (mnv2): build the per-channel-BN certified body VJP invresBodyPC_has_vjp_at
             -- (fresh, like r34's rblkPC) then tie invresBodyBackPC (+ strided) — relu6 masks pinned to the
             -- 0<preact<6 clamp-window signs, BN backs to bnPerChannelTensor3_has_vjp, depthwise via the
             -- gate. b1-free.
             `LeanMlir.Proofs.MobileNetV2BackCertifiedTie,
             -- §B integrity tie (efficientnet): mbconvBodyBack(pinned bn/swish/SE backs) = the certified
             -- mbconvBody_has_vjp.backward — SE back pinned to seBlockFull_has_vjp, swish to swish_has_vjp,
             -- depthwise via the gate. Certified per-example body VJP already exists (global bnForward).
             `LeanMlir.Proofs.EfficientNetBackCertifiedTie,
             -- §B integrity tie (vit MHSA — the sdpa adjoint): mhsaBackFlat (Q/K/V pinned to the actual
             -- dense projections at the saved input X) = the certified mhsa_has_vjp_mat.backward,
             -- flattened. Via ViTBackB0's mhsa_backward_collapseMH (certified Mat backward = per-head
             -- merged sum) + the projBack_core_coord/woback_unflatten coordinate match (dense Wᵀ = mulVec,
             -- Σk over h·dh reindexes to Σh Σj, separate projBacks regroup via sum_add_distrib).
             `LeanMlir.Proofs.ViTMhsaBackCertifiedTie,
             -- A3 §1g loss-head cotangent seed: lift softmax_ce_cot_close to a FloatBridges seed
             -- (z ↦ softmax(z)−onehot, the CE input-gradient; bounded by 1+cotErr(0) since softmax∈[0,1])
             -- so any <net>_grad .comp it = the whole "logits → input-gradient" backward "from the loss".
             `LeanMlir.Proofs.LossHeadCotFloatBridge,
             -- A3 §1f softmax-Jacobian backward (the vit/attention crux): the row-coupled VJP
             -- diag(p)−p·pᵀ, softmaxBack p dy i = pᵢ(dyᵢ−⟨p,dy⟩). Linear in dy (modulus = magnitude at
             -- Cdy:=e, like bnGradInput); float threads mul_close/reduction_close/sub_close' with the
             -- softmax weights supplied within smErr. floatClose_softmaxBack / floatBridges_softmaxBack.
             `LeanMlir.Proofs.SoftmaxBackFloatBridge,
             -- A3 §1f Mat-space SDPA BACKWARD assembly (the vit-crux capstone): the backward peer of
             -- sdpa_close. Certified sdpa_back_{Q,K,V} = three matmuls + softmaxBack + a 1/√d scale;
             -- floats reuse attnScore_close (dw), attnDot_close (dQ/dK/dV at perturbed weights),
             -- softmaxBack_close/sub_abs_le (the row VJP), mul_close (scale). sdpaBack{V,Q,K}_close.
             `LeanMlir.Proofs.SdpaBackFloatBridge,
             -- A3 §1f FULL multi-head self-attention backward assembly: the backward peer of
             -- floatBridges_mhProjAttnFull. dY ↦ dX = WoBack → 3 sdpa cores → Q/K/V projBacks + fan-in.
             -- The projBacks are FREE (per-token linBack); the cores (floatBridges_core{V,Q,K}) lift the
             -- flattened mhsaSdpaBack* to FloatClose (linear-in-cotangent). floatBridges_mhsaBack.
             `LeanMlir.Proofs.MhsaBackFloatBridge,
             -- A3 §1f the ViT PATCH-EMBED backward (the last whole-net endpoint): the certified
             -- patchEmbed_input_grad_formula (a transposed-conv / guarded patch-scatter triple-sum, linear
             -- in the cotangent) float-bridges via dot_close (fan-in D) + nested reduction_close (the
             -- kw/kh/p sums). floatBridges_patchEmbedBack discharges vit_grad_floatBridges's hPatch.
             `LeanMlir.Proofs.PatchEmbedBackFloatBridge,
             -- ViT WHOLE-NET FORWARD (forward peer of vit_grad_floatBridges): vit_full reversed =
             -- classifier ∘ perRowFlat finalLN ∘ tower blocks ∘ patchEmbed. The encoder tower REUSES
             -- towerBack (its head-first fold IS the forward order) + floatBridges_towerBack; the LN
             -- rides FloatBridges.perRow; the head (dense ∘ cls-slice) is concrete with the one new
             -- op-bridge floatBridges_clsSlice (the cls-slice gather, peer of clsScatter, exact). The
             -- per-row LN / blocks / patch-embed supplied as FloatBridges (blocks via floatBridges_vitBlock).
             `LeanMlir.Proofs.ViTWholeFloatBridge,
             -- ViT PATCH-EMBED FORWARD (the last vit forward endpoint, peer of floatBridges_patchEmbedBack):
             -- patchEmbed_flat = pos_embed + (cls_token | b_conv + ∑c∑kh∑kw W·guarded-img), affine in the
             -- image (constants cancel in the diff). M.patchEmbedF rounds the leaf mul (mul_close), the 3
             -- c/kh/kw sums (nested reduction_close) and the 2 constant adds (add_close). floatBridges_
             -- patchEmbed discharges vit_floatBridges's hPatch ⇒ vit_floatBridges_concrete (fully concrete).
             `LeanMlir.Proofs.PatchEmbedFloatBridge,
             -- The optimizer rung beyond SGD: the ℝ Adam/AdamW step mirroring
             -- the emitted update (Phase 3a of vit_train_to_vit_verified.md).
             -- Faithfulness target + denominator well-definedness; NO descent
             -- claim (Adam isn't monotone).
             `LeanMlir.Proofs.AdamStep,
             -- Phase 3b: the AdamW render-close — emitted weight/bias update =
             -- adamWScalar of the certified gradient (sgdW_descends_certified_grad
             -- analogue, optimizer swapped for AdamW).
             `LeanMlir.Proofs.AdamRender,
             -- Stage 2 of the live ResNet-34 (Item A2): the channel-order invariant
             -- kit (maxpool/BN/ReLU preserve strict pointwise channel domination —
             -- the non-vacuity carrier). Build-checked; not yet a live witness, so
             -- also NOT in the AuditAxioms headline set.
             `LeanMlir.Proofs.ResNet34Live2,
             -- Item A: the first NON-DEGENERATE ResNet-34 whole-net backward witness
             -- (level 2) — 2-channel stem + maxpool + 3 strided downsamples + GAP +
             -- dense, every smoothness hypothesis discharged, forward X ≠ forward 0
             -- via the channel-order invariant. In the AuditAxioms headline set.
             `LeanMlir.Proofs.ResNet34LivePC,
             -- Item A level 3: the nonzero-Jacobian SEAL for the live ResNet-34
             -- witness (fderiv ℝ liveFwd2 Y ≠ 0 ⇒ backward not the zero map). Sealed
             -- at a channel-symmetric base Y via the BN channel-difference identity
             -- (carrier vanishes ⇒ no BN-variance derivative needed). The ResNet peer
             -- of MobileNetV2JacobianSeal. In the AuditAxioms headline set.
             `LeanMlir.Proofs.ResNet34LiveSeal,
             -- Item A FULL DEPTH: the real [3,4,6,3] (16-block) live ResNet-34, level-3
             -- sealed. The 13 identity blocks (zeroed body ⇒ relu(x+1)=x+1) wash out
             -- through the downsamples' BN (bn(z+c)=bn(z)), so the full net = the
             -- empty-chain witness + 2 and the seal reduces to ResNet34LiveSeal's.
             `LeanMlir.Proofs.ResNet34LiveFull,
             -- MobileNetV2 FULL DEPTH: the real 17-block live MobileNetV2, level-3
             -- sealed. 15 identity skip blocks (zeroed body ⇒ ivId a = a+3, no relu —
             -- linear bottleneck) shift by +45; GAP + identity head pass it, so the
             -- full net = the 2-block witness + 45 and the seal reduces to
             -- MobileNetV2JacobianSeal's Qq / g_hasDerivAt. VJP composed through all 17.
             `LeanMlir.Proofs.MobileNetV2JacobianSealFull,
             -- Item D (realistic dims): the live ResNet-34 whole-net backward at real
             -- ImageNet 224×224 spatial resolution (the genuine 5-halving pyramid
             -- 224→112→56→28→14→7). β-parametric downsample (β=64>√1568) + stem
             -- (β=160>√25088); every smoothness/no-tie hyp discharged at n up to 25088,
             -- forward X≠0 (level 2). Confirms no discharge secretly used a small n.
             `LeanMlir.Proofs.ResNet34LiveRealistic,
             -- Item D level 3: the nonzero-Jacobian SEAL at 224×224. A uniform channel-0
             -- perturbation makes channel0 = channel1 + δ everywhere, so 7×7 GAP of a
             -- uniform diff = δ and maxpool(ch0)=maxpool(ch1)+δ for ALL t (max(a+δ,b+δ)=
             -- max(a,b)+δ) — no eventual-selection topology. UDiff invariant threaded like
             -- Dom2; output diff = t·Rr (4 positive istds), g'(0)=Rr 0 ≠ 0.
             `LeanMlir.Proofs.ResNet34LiveRealisticSeal,
             -- Item D level 3 for MobileNetV2: the nonzero-Jacobian SEAL at 224×224. ReLU6
             -- is a BOUNDED window (0,6), so unlike ResNet's β-grows route, γ is SCALED DOWN
             -- (γ=1/128 ⇒ |γ|√n < 3 keeps bn∈(0,6) at n=2·112·112). The 1×1 weights are
             -- dimension-independent and reused. Uniform-perturbation UDiff seal: the
             -- asymmetric stem turns input t into channel-diff −t, each BN ×γ·istd, so the
             -- output diff is −t·Rr (4 positive γ·istds), g'(0)=−Rr 0 ≠ 0.
             `LeanMlir.Proofs.MobileNetV2SealRealistic,
             -- Backward-graph faithfulness (den-level): fan-in bricks
             -- (residual/SE), per-op backward ops (gap/broadcast/true-batch-norm/
             -- batched conv+depthwise), the whole per-example MBConv block, and
             -- the batched-stage backward primitives.
             `LeanMlir.Proofs.EfficientNetBackB0,
             -- MobileNetV2 backward-graph faithfulness (den-level): the batched
             -- relu6 conv/depthwise stages (selectMid kink), the SE-less inverted-
             -- residual body, and the whole-block capstone — the relu6 (_at)
             -- peer of EfficientNetBackB0.
             `LeanMlir.Proofs.MobileNetV2BackB0,
             -- ResNet-34 backward-graph faithfulness (den-level): the batched
             -- conv-bn-relu stage (selectPos one-sided kink), the basic-block
             -- body (conv-bn ∘ conv-bn-relu), and the identity-block capstone —
             -- relu (_at) with an OUTER post-residual relu (the extra factor
             -- vs the MBConv/inverted-residual blocks).
             `LeanMlir.Proofs.ResNet34BackB0,
             -- ConvNeXt backward-graph faithfulness (den-level): the per-example
             -- (batch-1) peer of EfficientNetBackB0. LayerNorm is per-example
             -- separable, so no batched machinery — the block-body backward graph
             -- (depthwise → LN → expand → gelu → project → layerScale) + identity-skip
             -- residual capstone, plus the LN+2×2/s2 downsample capstone. GELU is a
             -- global VJP, so everything stays in the clean global HasVJP form.
             `LeanMlir.Proofs.ConvNeXtBackB0,
             -- ViT whole-block backward-graph faithfulness (den-level, heads = 1):
             -- the per-token Mat-VJP peer of the conv nets' *BackB0 capstones. MLP +
             -- attention sublayer backward graphs (residual fan-in + LN-back), with
             -- the MHSA backward collapsed at heads = 1 to the plain three-way dense
             -- fan-in over the proven sdpa_back_{Q,K,V} (tied to mhsa_has_vjp_mat by
             -- VJP determinism), assembled into the whole transformerBlock VJP.
             `LeanMlir.Proofs.ViTBackB0,
             -- PoC: the mnist-linear train step proof-tied to the certified
             -- loss-descent SGD step (the renderer `MainMnistLinearVerified`
             -- trains on), incl. the param-grad/SGD "tail fold". Template for
             -- making each chapter's verified trainer faithful — see
             -- planning/verified_faithful_sweep.md.
             `LeanMlir.Proofs.LinearFaithfulPoC,
             -- E4M3 (fp8) render-tie (planning §3b): the emitted block-scaled
             -- int-matmul graph denotes the intended dequant-first algorithm
             -- (the per-output dequant scale factors out of the fp32 accumulate),
             -- via existing den-faithful ops only — E4M3FaithfulPoC.lean.
             `LeanMlir.Proofs.E4M3FaithfulPoC,
             -- bf16-mixed render-tie (planning §5, the symmetric gap): the emitted
             -- bf16-leaf/fp32-accumulate linear graph denotes the rounded-operand
             -- linear (no scale to factor — simpler than the E4M3 twin). Unlike fp8,
             -- this graph lowers on CUDA. Bf16FaithfulPoC.lean.
             `LeanMlir.Proofs.Bf16FaithfulPoC,
             -- mnist-MLP peer: the whole 3-layer MLP train step folded into the
             -- verified AST (forward + backward chain + 6 weightSgd/biasSgd), each
             -- output's den proven = certified via mlp_render_*_certified.
             `LeanMlir.Proofs.MlpFaithfulPoC,
             -- mnist-CNN peer: the conv train step folded into the verified AST via
             -- the new convWeightSgd/convBiasSgd ops (conv layers) + weightSgd/biasSgd
             -- (dense head); each of the 10 outputs' den proven = certified via the
             -- conv chain bridges + the M2 dense bridges (CnnFaithfulPoC.lean).
             `LeanMlir.Proofs.CnnFaithfulPoC,
             -- ch5-CIFAR peer (no-BN, deeper 2-scale net): reuses the cnn conv ops +
             -- dense bridges (NO new core ops) — generic convW/convB_den cover all 4
             -- conv layers, the 3-dense head via the M2 bridges (CifarFaithfulPoC.lean).
             `LeanMlir.Proofs.CifarFaithfulPoC,
             -- ch5-CIFAR-BN peer (per-channel BatchNorm): reuses the cnn conv ops + the
             -- cifar dense head; the new bnGammaSgd/bnBetaSgd ops carry the per-channel
             -- γ/β grads, den-certified via cifar_bn_render_{gamma,beta}_certified
             -- (CifarBnFaithfulPoC.lean).
             `LeanMlir.Proofs.CifarBnFaithfulPoC,
             -- ch5-CIFAR-BN §1a TIE: conv+BN tied through the real forward + the BN backward chain
             -- (BN-output cots relu-masked for γ/β, conv cots via BN-back) — CifarBnTiePoC.lean.
             `LeanMlir.Proofs.CifarBnTiePoC,
             -- deeper 8-conv cifar8 (no-BN): pure reuse — conv via CifarPoC generics,
             -- dense via the new generic denseW/denseB_den (Cifar8FaithfulPoC.lean).
             `LeanMlir.Proofs.Cifar8FaithfulPoC,
             -- ch5-cifar8 §1a TIE: 8-conv chain tied through the real forward — cifar's chain
             -- repeated over 4 stages, all reused constructors (Cifar8TiePoC.lean).
             `LeanMlir.Proofs.Cifar8TiePoC,
             -- ch5-cifar8-bn §1a TIE: cifar8's chain + a BN-back at every conv; all 32 conv+BN
             -- params tied (Cifar8BnTiePoC.lean).
             `LeanMlir.Proofs.Cifar8BnTiePoC,
             -- ch6-ResNet-34 (full [3,4,6,3], 146 params): the 2 new strided-conv SGD ops
             -- (convStrided{Weight,Bias}Sgd) for the 7×7 stem + 3×3 downsample/projection
             -- convs den-certified via mnv2_render_stem_conv{W,b}_certified; the 142 other
             -- params reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics (ResNet34FaithfulPoC.lean).
             `LeanMlir.Proofs.ResNet34FaithfulPoC,
             -- ch6-ResNet-34 §1a TIE: per-block-type tie lemmas (identity/downsample/stem) at the
             -- real forward + ResNet34ChainClose cotangents, the residual fan-in SUM constructors
             -- (idBlockCotIn/downBlockCotIn), loss-cot + dense fold (ResNet34TiePoC.lean).
             `LeanMlir.Proofs.ResNet34TiePoC,
             -- ch7-MobileNetV2 §1 fold (depthwise half): the 4 new depthwise SGD ops
             -- (depthwise{,Strided}{Weight,Bias}Sgd) den-certified via the mnv2_render_depthwise*
             -- bridges; expand/project/BN/dense reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics
             -- (MobileNetV2FaithfulPoC.lean).
             `LeanMlir.Proofs.MobileNetV2FaithfulPoC,
             -- ch7-MobileNetV2 §1 CLOSE (render): the reduced 6-block train step rendered ENTIRELY
             -- as pretty(provenGraph) — every line pretty of a verified SHlo node, the depthwise
             -- param updates via the new depthwise SGD ops; writes verified_mlir/mobilenetv2_train_step.mlir
             -- (MobileNetV2Render.lean, the peer of ResNet34Render.lean).
             `LeanMlir.Proofs.MobileNetV2Render,
             -- ch7-MobileNetV2 FULL 17-block paper §1 fold (den): every one of the 210 params of
             -- mnv2TrainStepFaithfulVPaper denotes the certified step — ZERO new ops/lemmas, the
             -- cifar8-bn lesson at full scale. Six per-block-type capstones (stem/no-exp/stride-1/
             -- stride-2/head/dense), each delegating to the audited CifarPoC/CifarBnPoC/Cifar8PoC/
             -- Mnv2PoC/ResNet34PoC generics (MobileNetV2FaithfulPoCPaper.lean).
             `LeanMlir.Proofs.MobileNetV2FaithfulPoCPaper,
             -- ch7-MobileNetV2 FULL 17-block paper §1a TIE: the whole 210-param train step tied
             -- through the REAL mobilenetv2ForwardPaper + the loss-driven backward chain (relu6
             -- two-kink masks, residual fan-in at every stride-1 skip). Per-block-type tie lemmas
             -- (no-exp/stride-1/stride-2/stem/head) applied across all 17 blocks via @[irreducible]
             -- FwdO/CotInAt/TiedAt wrappers (the r34 heartbeat lesson) (MobileNetV2TiePoCPaper.lean).
             `LeanMlir.Proofs.MobileNetV2TiePoCPaper,
             -- ch8-EfficientNet-B0 full-16 (262-param) train step rendered as pretty(provenGraph)
             -- at the batched index (N=1, emit B = batch); un-fused SE for the SE param grads
             -- (EfficientNetRender.lean); writes verified_mlir/efficientnet_train_step.mlir.
             `LeanMlir.Proofs.EfficientNetRender,
             -- ch8-EfficientNet-B0 §1 fold (den): every batched param-SGD op type denotes the
             -- certified Σ_n batched gradient — conv/strided-stem/dense W,b + BN γ/β + depthwise
             -- (the Σ_n batch-sum bridge = Finset.sum_congr of the per-example .correct)
             -- (EfficientNetFaithfulPoC.lean).
             `LeanMlir.Proofs.EfficientNetFaithfulPoC,
             -- ch8-EfficientNet-B0 §1a TIE (IN PROGRESS): pins each param cotangent to the actual
             -- loss-driven backward chain. Landed: the loss-cotangent den (batched softmaxRowF − onehot);
             -- the whole-net thread (swish/SE-gate/true-BN chain-cot constructors) is the remaining
             -- dedicated effort (EfficientNetTiePoC.lean).
             `LeanMlir.Proofs.EfficientNetTiePoC,
             -- ch9-ConvNeXt-T §1 fold (started): the per-channel layer-scale γ gradient cert —
             -- the one genuinely-new proof obligation (Vec c via the chanIdx broadcast, vs the
             -- per-element Vec n cnx_render_lsgamma_certified); the den target of the pending
             -- layerScaleChGammaSgd core op (ConvNeXtFaithfulPoC.lean).
             `LeanMlir.Proofs.ConvNeXtFaithfulPoC,
             -- ch9-ConvNeXt-T §1 RENDER: the full [3,3,9,3] train step rendered as pretty(provenGraph)
             -- (fwd + bwd-cotangent chain + param-SGD via the new ops); writes
             -- verified_mlir/convnext_train_step.mlir. 2 documented hand-written gaps (the stem 4×4/s4
             -- + downsample 2×2/s2 weight grads — no even/stride-4 weight-grad VJP yet) (ConvNeXtRender.lean).
             `LeanMlir.Proofs.ConvNeXtRender,
             -- ch9-ConvNeXt-T §1a TIE: the whole [3,3,9,3] train step tied through the REAL forward —
             -- 18 blocks + 3 downsamples + GAP→LN→dense head + stem bias den-composed
             -- forward→loss→backward (GELU masks, identity-skip fan-in, downsample LN-back); the 4
             -- even-kernel weight grads are the documented render gap (ConvNeXtTiePoC.lean).
             `LeanMlir.Proofs.ConvNeXtTiePoC,
             -- ch10-ViT-Tiny §1 RENDER: the full depth-12 train step rendered as pretty(provenGraph)
             -- (fwd + per-head SDPA backward chain + 200-param SGD via the 6 new ops); iree-validated
             -- (LeanMlir/Proofs/ViTRender.lean). NO param gap — vit has the patch-weight VJP cert.
             `LeanMlir.Proofs.ViTRender,
             -- ch10-ViT-Tiny §1 FOLD: each emitted param-SGD op den=certified — vecln γ/β, rowwise
             -- dense W/b, patch conv W/b, pos (one-line delegations to ViTVecLN/ViTClose certs); the
             -- head reuses Cifar8PoC.dense{W,B}_den, cls reuses denseBiasSgdB (ViTFaithfulPoC.lean).
             `LeanMlir.Proofs.ViTFaithfulPoC,
             -- ch10-ViT-Tiny §1a TIE (per-block): every one of a vector-LN transformer block's 16 params,
             -- fed the cotangent the REAL backward chain delivers (vitCot* — two residual fan-ins + the
             -- three-way LN₁ fan-in + the SDPA backs), den=certified. Single-head representative; the
             -- multi-head/depth-12 thread is the remaining step (mnv2 reduced→full) (ViTTiePoC.lean).
             `LeanMlir.Proofs.ViTTiePoC]

/-- **`lake build ProofsMinimal`** — the suite's "hello world": the smallest
    end-to-end story (the Linear classifier), both halves — faithfulness
    (`LinearFaithfulPoC`: emitted train-step = certified math) and descent
    (`SgdDescentLinear`: that step decreases the loss). Their transitive closure is
    exactly the minimum working set (LinearTrainStep + the shared StableHLO/Tensor/
    FloatBridge/IR foundation), nothing per-net beyond Linear. Point a newcomer here
    before the full `Proofs` target. See `LeanMlir/Proofs/README.md` (Start here) and
    `planning/proofs_minimal_set.md`. -/
lean_lib «ProofsMinimal» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.LinearFaithfulPoC, `LeanMlir.Proofs.SgdDescentLinear]

/-- **`lake build TrustedBridge`** — the named ℝ→Float32 trust bridge. Kept SEPARATE from
    `Proofs` on purpose: it carries the explicit IEEE rounding axioms (`ieeeRnd`/`ieeeRnd_err`,
    the one thing Lean can't prove about opaque hardware `Float`) and discharges concrete
    binary32/fp8 corollaries from them. CI builds this, then `tests/AuditTrustedBridge.lean`
    bounds its axiom footprint to exactly the 3-axiom triple + those two named axioms (no
    `sorryAx`, no other axiom) — so the zero-axiom `Proofs` invariant stays untouched. -/
lean_lib «TrustedBridge» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Binary32Instance]

/-- **`lake build Codegen`** — the Lean→MLIR codegen + spec core, no proofs.
    The half that actually emits StableHLO and runs on device. -/
lean_lib «Codegen» where
  srcDir := "."
  roots := #[`LeanMlir.MlirCodegen, `LeanMlir.Train, `LeanMlir.Spec,
             `LeanMlir.SpecHelpers, `LeanMlir.Types, `LeanMlir.IreeRuntime,
             `LeanMlir.Ddpm, `LeanMlir.Cam, `LeanMlir.F32Array]

-- IREE FFI shim: Lean ↔ C bridge for libiree_ffi.so (see ffi/).
target ireeLeanFfiO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "iree_lean_ffi.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "iree_lean_ffi.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString,
                    "-I", (pkg.dir / "ffi").toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

-- F32 ByteArray helpers (He init, argmax, data loading — all in C for speed).
target f32HelpersO pkg : System.FilePath := do
  let oFile := pkg.buildDir / "ffi" / "f32_helpers.o"
  let srcJob ← inputTextFile <| pkg.dir / "ffi" / "f32_helpers.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString]
  let traceArgs := #["-fPIC", "-O2"]
  buildO oFile srcJob weakArgs traceArgs

extern_lib libireeffi pkg := do
  let shimO ← fetch <| pkg.target ``ireeLeanFfiO
  let f32O  ← fetch <| pkg.target ``f32HelpersO
  buildStaticLib (pkg.staticLibDir / nameToStaticLib "ireeffi") #[shimO, f32O]

-- ═══════════════════════════════════════════════════════════════════
-- Phase 3 trainers (Lean → MLIR → IREE → GPU)
-- ═══════════════════════════════════════════════════════════════════

private def ireeLink : Array String :=
  #["-L", "./ffi", "-liree_ffi", "-Wl,-rpath,./ffi", "-Wl,--allow-shlib-undefined"]

lean_exe «resnet34-train» where
  root := `apps.baselines.MainResnetTrain
  moreLinkArgs := ireeLink

lean_exe «resnet50-train» where
  root := `apps.baselines.MainResnet50Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v2-train» where
  root := `apps.baselines.MainMobilenetV2Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v3-train» where
  root := `apps.baselines.MainMobilenetV3Train
  moreLinkArgs := ireeLink

lean_exe «mobilenet-v4-train» where
  root := `apps.baselines.MainMobilenetV4Train
  moreLinkArgs := ireeLink

lean_exe «efficientnet-train» where
  root := `apps.baselines.MainEfficientNetTrain
  moreLinkArgs := ireeLink

lean_exe «efficientnet-v2-train» where
  root := `apps.baselines.MainEfficientNetV2Train
  moreLinkArgs := ireeLink

lean_exe «convnext-tiny-train» where
  root := `apps.baselines.MainConvNeXtTrain
  moreLinkArgs := ireeLink

lean_exe «vit-tiny-train» where
  root := `apps.baselines.MainVitTrain
  moreLinkArgs := ireeLink

-- Muon (Newton–Schulz polar projection) on the 2D weights, AdamW on the rest.
-- Same ViT-Tiny + recipe as vit-tiny-train → a compute-matched A/B. See planning/muon.md.
lean_exe «vit-tiny-muon-train» where
  root := `apps.baselines.MainVitMuonTrain
  moreLinkArgs := ireeLink

lean_exe «ablation» where
  root := `apps.ablation.MainAblation
  moreLinkArgs := ireeLink

lean_exe «vgg-train» where
  root := `apps.baselines.MainVggTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-cnn-train» where
  root := `apps.baselines.MainMnistCnnTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-bn-train» where
  root := `apps.baselines.MainCifarCnnBnTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-mlp-train» where
  root := `apps.baselines.MainMnistMlpTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-linear-train» where
  root := `apps.baselines.MainMnistLinearTrain
  moreLinkArgs := ireeLink

-- Trains MNIST-linear on the VERIFIED-rendered StableHLO
-- (`verified_mlir/`, = Proofs.StableHLO.linearTrainStepModuleV) through the
-- real Lean/IREE FFI. See MainMnistLinearVerified.lean.
lean_exe «mnist-linear-verified» where
  root := `apps.mnist.MainMnistLinearVerified
  moreLinkArgs := ireeLink

-- Chapter 2 (low precision): fp8 (E4M3) training on the SAME verified StableHLO —
-- fp32 master, per-column W / per-tensor x projected to the E4M3 grid, fp32 accumulate.
-- See MainMnistLinearE4M3Verified.lean + LeanMlir/E4M3Quant.lean (§3b/§3c sit on this).
lean_exe «mnist-linear-e4m3-verified» where
  root := `apps.mnist.MainMnistLinearE4M3Verified
  moreLinkArgs := ireeLink

-- Chapter 3: trains the MNIST MLP on the VERIFIED-rendered StableHLO
-- (verified_mlir/mlp_train_step.mlir = Proofs.StableHLO.mlpTrainStepText).
lean_exe «mnist-mlp-verified» where
  root := `apps.mnist.MainMnistMlpVerified
  moreLinkArgs := ireeLink

-- Chapter 3 (low precision): fp8 (E4M3) MLP training on the SAME verified StableHLO.
-- fp32 master, per-column weight quant + per-tensor input, fp32 accumulate.
-- fp8 weights+input, fp32 intermediates. See MainMnistMlpE4M3Verified.lean.
lean_exe «mnist-mlp-e4m3-verified» where
  root := `apps.mnist.MainMnistMlpE4M3Verified
  moreLinkArgs := ireeLink

-- Chapter 4: trains the MNIST CNN on the VERIFIED-rendered StableHLO
-- (verified_mlir/cnn_train_step.mlir = Proofs.StableHLO.cnnTrainStepText).
lean_exe «mnist-cnn-verified» where
  root := `apps.mnist.MainMnistCnnVerified
  moreLinkArgs := ireeLink

-- Chapter 4 (low precision): fp8 (E4M3) CNN training on the SAME verified StableHLO.
-- fp32 master, conv per-channel / dense per-column weight quant + per-tensor input,
-- fp32 accumulate. fp8 weights+input, fp32 intermediates. See MainMnistCnnE4M3Verified.lean.
lean_exe «mnist-cnn-e4m3-verified» where
  root := `apps.mnist.MainMnistCnnE4M3Verified
  moreLinkArgs := ireeLink

-- Chapter 5: trains the CIFAR-10 CNN (no BN) on the VERIFIED-rendered StableHLO
-- (verified_mlir/cifar_train_step.mlir = Proofs.StableHLO.cifarTrainStepText).
lean_exe «cifar-verified» where
  root := `apps.cifar.MainCifarVerified
  moreLinkArgs := ireeLink

-- Chapter 5 (low precision): fp8 (E4M3) CIFAR-10 training on the SAME verified StableHLO.
-- fp32 master, conv per-channel / dense per-column weight quant + per-tensor input,
-- fp32 accumulate. fp8 weights+input, fp32 intermediates. See MainCifarE4M3Verified.lean.
lean_exe «cifar-e4m3-verified» where
  root := `apps.cifar.MainCifarE4M3Verified
  moreLinkArgs := ireeLink

-- Chapter 5 (BatchNorm): trains the CIFAR-10 CNN + per-example BN on the
-- VERIFIED-rendered StableHLO (Proofs.StableHLO.cifarBnTrainStepText).
lean_exe «cifar-bn-verified» where
  root := `apps.cifar.MainCifarBnVerified
  moreLinkArgs := ireeLink

-- Deeper 8-conv CIFAR-10 CNN (no BN; [16,16,32,32], 4 pools) on the VERIFIED-rendered
-- StableHLO (verified_mlir/cifar8_train_step.mlir = Proofs.StableHLO.cifar8TrainStepText).
lean_exe «cifar8-verified» where
  root := `apps.cifar.MainCifar8Verified
  moreLinkArgs := ireeLink

-- Deeper 8-conv CIFAR-10 CNN + per-channel BN on the VERIFIED-rendered StableHLO
-- (Proofs.StableHLO.cifar8BnTrainStepText). The pedagogical BN-acceleration demo.
lean_exe «cifar8-bn-verified» where
  root := `apps.cifar.MainCifar8BnVerified
  moreLinkArgs := ireeLink

-- cifar8 (no BN) Adam peer: the proof-rendered fwd/bwd/param-grads with the SGD update
-- swapped for AdamW (ViTRender.emitAdamV) + packed [θ|m|v] + runtime lr/bc threading via
-- trainAdamSched. Render: tests/TestCifar8AdamTrain.lean. BN/noBN × SGD/Adam ablation.
lean_exe «cifar8-verified-adam» where
  root := `apps.cifar.MainCifar8VerifiedAdam
  moreLinkArgs := ireeLink

-- cifar8 + per-channel BN Adam peer (38 params incl. 8× BN γ/β). Same as above with BN.
-- Render: tests/TestCifar8AdamTrain.lean.
lean_exe «cifar8-bn-verified-adam» where
  root := `apps.cifar.MainCifar8BnVerifiedAdam
  moreLinkArgs := ireeLink

-- cifar8 Nesterov-momentum SGD peers (v←μv+∇, θ←θ−lr(μv+∇), μ=0.9): same proof-rendered body +
-- emitMomentum, driven by trainAdamSched variant "mom" (reuses [θ|m|v] packing + cosine+warmup lr).
-- Render: tests/TestCifar8AdamTrain.lean. Completes the optimizer ablation (SGD/momentum/Adam).
lean_exe «cifar8-verified-momentum» where
  root := `apps.cifar.MainCifar8VerifiedMomentum
  moreLinkArgs := ireeLink

-- fp8 (E4M3) optimizer sweep on the cifar8 CNN: the SGD / Nesterov-momentum / Adam
-- demos run through the E4M3 host-quant path (fp8 weights+input, fp32 accumulate,
-- fp32 master). Same verified train-step MLIR as their fp32 peers.
lean_exe «cifar8-e4m3-verified» where
  root := `apps.cifar.MainCifar8E4M3Verified
  moreLinkArgs := ireeLink

lean_exe «cifar8-e4m3-verified-momentum» where
  root := `apps.cifar.MainCifar8E4M3VerifiedMomentum
  moreLinkArgs := ireeLink

lean_exe «cifar8-e4m3-verified-adam» where
  root := `apps.cifar.MainCifar8E4M3VerifiedAdam
  moreLinkArgs := ireeLink

lean_exe «cifar8-bn-verified-momentum» where
  root := `apps.cifar.MainCifar8BnVerifiedMomentum
  moreLinkArgs := ireeLink

-- cifar8 plain-SGD CONTROL on the momentum/Adam pipeline (trainAdamSched variant "sgd": same
-- per-epoch shuffle + hflip + cosine-warmup, update θ←θ−lr·∇). Makes the SGD/momentum/Adam
-- comparison differ ONLY in the optimizer. Render: tests/TestCifar8AdamTrain.lean.
lean_exe «cifar8-verified-sgdsched» where
  root := `apps.cifar.MainCifar8VerifiedSgdSched
  moreLinkArgs := ireeLink

lean_exe «cifar8-bn-verified-sgdsched» where
  root := `apps.cifar.MainCifar8BnVerifiedSgdSched
  moreLinkArgs := ireeLink

-- Wide-head (MNIST-style 2×512 dense, d1=512) cifar8 optimizer ablation: each exe runs SGD /
-- momentum / AdamW in sequence on the controlled pipeline. Render: tests/TestCifar8WideTrain.lean.
lean_exe «cifar8w-ablation» where
  root := `apps.ablation.MainCifar8WideAblation
  moreLinkArgs := ireeLink

lean_exe «cifar8w-bn-ablation» where
  root := `apps.ablation.MainCifar8WideBnAblation
  moreLinkArgs := ireeLink

-- ch6 B9: real ResNet-34 ([3,4,6,3], per-channel BN, strided downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestResnet34{Train,Fwd}.lean); 146 params.
lean_exe «resnet34-verified» where
  root := `apps.imagenette.MainResnet34Verified
  moreLinkArgs := ireeLink

-- r34 peer of mnv2/enet-verified-adam: the proof-rendered train step (per-channel BN + strided
-- downsamples) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime
-- lr/bc threading via trainAdamSched. Recipe matches the reference (lr 1e-3, wd 1e-4, cosine+warmup
-- 3, label-smoothing 0.1). Render: tests/TestResnet34Train.lean.
lean_exe «resnet34-verified-adam» where
  root := `apps.imagenette.MainResnet34VerifiedAdam
  moreLinkArgs := ireeLink

-- ch7 C4: small MobileNetV2 (inverted-residual blocks: depthwise conv + relu6 +
-- per-channel BN) trained on VERIFIED-rendered StableHLO
-- (tests/TestMobilenetV2{Train,Fwd}.lean); 30 params.
lean_exe «mobilenetv2-verified» where
  root := `apps.imagenette.MainMobilenetV2Verified
  moreLinkArgs := ireeLink

-- mnv2 peer of vit-verified-adam: the proof-rendered train step with the SGD update swapped for
-- AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime lr/bc threading via trainAdamSched. Recipe
-- matches mobilenet-v2-train (lr 1e-3, wd 1e-4, cosine+warmup 3, label-smoothing 0.1). Loss-curve
-- parity; batch-BN eval (running-stats BN deferred). Render: tests/TestMobilenetV2TrainPC.lean.
lean_exe «mobilenetv2-verified-adam» where
  root := `apps.imagenette.MainMobilenetV2VerifiedAdam
  moreLinkArgs := ireeLink

-- ch8 E4/E5/E6: EfficientNet-B0 (faithful [t,c,n,s,k] config — 16 MBConv layers,
-- inverted-residual + squeeze-excite + swish + BATCH norm, 3×3/5×5 depthwise) trained
-- on VERIFIED-rendered StableHLO (tests/TestEfficientNet{Train,Fwd}.lean); 262 params.
lean_exe «efficientnet-verified» where
  root := `apps.imagenette.MainEfficientNetVerified
  moreLinkArgs := ireeLink

-- enet peer of mnv2-verified-adam: the proof-rendered train step (all-swish + squeeze-excite +
-- batch-norm) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime
-- lr/bc threading via trainAdamSched. Recipe matches efficientnet-train (lr 1e-3, wd 1e-4,
-- cosine+warmup 3, label-smoothing 0.1). Render: tests/TestEfficientNetTrain.lean.
lean_exe «efficientnet-verified-adam» where
  root := `apps.imagenette.MainEfficientNetVerifiedAdam
  moreLinkArgs := ireeLink

-- Chapter 9: ConvNeXt-T (Liu et al. 2022 — patchify stem + [3,3,9,3] depthwise-7×7
-- blocks with LN + GELU + layerScale + 3 between-stage downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestConvNeXt{Train,Fwd}.lean); 180 params.
lean_exe «convnext-verified» where
  root := `apps.imagenette.MainConvNeXtVerified
  moreLinkArgs := ireeLink

-- convnext peer of r34-verified-adam: the proof-rendered train step (all-smooth — LayerNorm +
-- GELU + layerScale, no BN) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed
-- θ|m|v + runtime lr/bc threading via trainAdamSched. Recipe matches the reference (lr 1e-3, wd
-- 1e-4, cosine+warmup 3, label-smoothing 0.1). Render: tests/TestConvNeXtTrain.lean.
lean_exe «convnext-verified-adam» where
  root := `apps.imagenette.MainConvNeXtVerifiedAdam
  moreLinkArgs := ireeLink

lean_exe «vit-verified» where
  root := `apps.imagenette.MainViTVerified
  moreLinkArgs := ireeLink

-- Phase 3c: ViT-Tiny with the VERIFIED-rendered AdamW step (packed θ|m|v threading
-- through the generic FFI; ViTRender.vitTrainStepModuleAdamPacked / trainAdamPacked).
lean_exe «vit-verified-adam» where
  root := `apps.imagenette.MainViTVerifiedAdam
  moreLinkArgs := ireeLink

lean_exe «cifar-cnn-train» where
  root := `apps.baselines.MainCifarCnnTrain
  moreLinkArgs := ireeLink

lean_exe «autoencoder-pets-train» where
  root := `demos.MainAutoencoderPetsTrain
  moreLinkArgs := ireeLink

lean_exe «unet-pets-train» where
  root := `demos.MainUnetPetsTrain
  moreLinkArgs := ireeLink

lean_exe «pets-predict» where
  root := `demos.MainPetsPredict
  moreLinkArgs := ireeLink

lean_exe «gradcam» where
  root := `demos.MainGradCAM
  moreLinkArgs := ireeLink

lean_exe «bigram-shakespeare» where
  root := `demos.MainBigramShakespeare
  moreLinkArgs := ireeLink

lean_exe «tinygpt-shakespeare» where
  root := `demos.MainTinyGptShakespeare
  moreLinkArgs := ireeLink

lean_exe «mnist-ddpm-train» where
  root := `demos.MainMnistDdpmTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-ddpm-sample» where
  root := `demos.MainMnistDdpmSample
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-train» where
  root := `demos.MainCifarDdpmTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-sample» where
  root := `demos.MainCifarDdpmSample
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-attn-train» where
  root := `demos.MainCifarDdpmAttnTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-attn-sample» where
  root := `demos.MainCifarDdpmAttnSample
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-sincos-train» where
  root := `demos.MainCifarDdpmSincosTrain
  moreLinkArgs := ireeLink

lean_exe «cifar-ddpm-sincos-sample» where
  root := `demos.MainCifarDdpmSincosSample
  moreLinkArgs := ireeLink

-- YOLOv1 cat/dog head detector on Oxford-IIIT Pets (2×2 mosaic, R34 backbone
-- bootstrap, focal objectness). See planning/yolo_final.md.
lean_exe «yolov1-pets-train-bootstrap» where
  root := `demos.MainYolov1PetsTrainBootstrap
  moreLinkArgs := ireeLink

-- Inference dump (logits + images + IDs) for scripts/yolo_render.py.
lean_exe «yolov1-pets-infer» where
  root := `demos.MainYolov1PetsInfer
  moreLinkArgs := ireeLink

-- ═══════════════════════════════════════════════════════════════════
-- VJP oracle — one binary per axiom under test.
-- Trainers live in tests/vjp_oracle/phase3/ so the root isn't crowded
-- with test-only files. See tests/vjp_oracle/README.md.
-- ═══════════════════════════════════════════════════════════════════

lean_exe «vjp-oracle-dense» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDense
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-dense-relu» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDenseRelu
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-conv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConv
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-convbn» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConvBn
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-conv-pool» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleConvPool
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-residual» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleResidual
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-depthwise» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleDepthwise
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-attention» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleAttention
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-mbconv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleMbConv
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-global-avg-pool» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleGlobalAvgPool
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-bottleneck» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleBottleneck
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-mbconv-v3» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleMbConvV3
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-fused-mbconv» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleFusedMb
  moreLinkArgs := ireeLink

lean_exe «vjp-oracle-uib» where
  root := `tests.vjp_oracle.phase3.MainVjpOracleUib
  moreLinkArgs := ireeLink

-- ═══════════════════════════════════════════════════════════════════
-- Tests + benchmarks
-- ═══════════════════════════════════════════════════════════════════

lean_exe «test-forward» where
  root := `tests.TestForward
  moreLinkArgs := ireeLink

lean_exe «test-iree» where
  root := `tests.TestIreeRuntime
  moreLinkArgs := ireeLink

lean_exe «test-train» where
  root := `tests.TestTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-iree-load» where
  root := `tests.TestIreeLoad
  moreLinkArgs := ireeLink

lean_exe «test-f32» where
  root := `tests.TestF32
  moreLinkArgs := ireeLink

lean_exe «bench-resnet» where
  root := `tests.BenchResnet
  moreLinkArgs := ireeLink

lean_exe «test-resnet-fwd» where
  root := `tests.TestResnetForward

lean_exe «test-unet-forward» where
  root := `tests.TestUnetForward

lean_exe «test-ddpm-train-emit» where
  root := `tests.TestDdpmTrainEmit
  moreLinkArgs := ireeLink

lean_exe «test-convnext-fwd-emit» where
  root := `tests.TestConvNextForwardEmit
  moreLinkArgs := ireeLink

lean_exe «test-convnext-train-emit» where
  root := `tests.TestConvNextTrainEmit
  moreLinkArgs := ireeLink

lean_exe «test-focal-emit» where
  root := `tests.TestFocalEmit
  moreLinkArgs := ireeLink

-- YOLOv1 Phase 1 tests (planning/yolo_final.md decisions D10-D11).
lean_exe «test-yolov1-emit» where
  root := `tests.TestYolov1Emit
  moreLinkArgs := ireeLink

lean_exe «test-yolov1-mutex» where
  root := `tests.TestYolov1Mutex
  moreLinkArgs := ireeLink

lean_exe «test-yolov1-train-step» where
  root := `tests.TestYolov1TrainStep
  moreLinkArgs := ireeLink

lean_exe «test-randaugment» where
  root := `tests.TestRandAugment
  moreLinkArgs := ireeLink

lean_exe «test-cam» where
  root := `tests.TestCam
  moreLinkArgs := ireeLink

lean_exe «inspect-convnext-bn» where
  root := `tests.InspectConvNeXtBN
  moreLinkArgs := ireeLink

lean_exe «inspect-convnext» where
  root := `demos.MainInspectConvNeXt
  moreLinkArgs := ireeLink

lean_exe «test-convnext-train-step» where
  root := `tests.TestConvNextTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-convnext-bn-train-step» where
  root := `tests.TestConvNextBnTrainStep
  moreLinkArgs := ireeLink

lean_exe «test-convnext-ablation-smoke» where
  root := `tests.TestConvNextAblationSmoke
  moreLinkArgs := ireeLink

lean_exe «test-convnext-tiny-emit» where
  root := `tests.TestConvNextTinyEmit

lean_exe «test-resnet-residual» where
  root := `tests.TestResnetResidual

lean_exe «test-spec-helpers» where
  root := `tests.TestSpecHelpers

lean_exe «test-smoke-trainers» where
  root := `tests.TestSmokeTrainers

lean_exe «test-codegen-ts» where
  root := `tests.TestCodegenTrainStep

-- Dischargeability sanity check: 11 examples confirming every
-- Differentiable hypothesis the proofs propagate is satisfiable for
-- the architecture functions (dense, softmax, layerNorm, the flat
-- transformer pieces, mhsa_layer_flat). If any goes vacuous on a
-- refactor, this will fail at build time.
lean_exe «test-diff-sanity» where
  root := `tests.TestDifferentiableSanity

-- ════════════════════════════════════════════════════════════════
-- Bestiary: architecture-only NetSpec examples (print, no training)
-- ════════════════════════════════════════════════════════════════

lean_exe «bestiary-alphazero» where
  root := `Bestiary.AlphaZero

lean_exe «bestiary-highway» where
  root := `Bestiary.Highway

lean_exe «bestiary-densenet» where
  root := `Bestiary.DenseNet

lean_exe «bestiary-vgg» where
  root := `Bestiary.VGG

lean_exe «bestiary-resnet» where
  root := `Bestiary.ResNet

lean_exe «bestiary-wrn» where
  root := `Bestiary.WRN

lean_exe «bestiary-mamba» where
  root := `Bestiary.Mamba

lean_exe «bestiary-swin» where
  root := `Bestiary.SwinT

lean_exe «bestiary-unet» where
  root := `Bestiary.UNet

lean_exe «bestiary-detr» where
  root := `Bestiary.DETR

lean_exe «bestiary-yolo» where
  root := `Bestiary.YOLO

lean_exe «bestiary-shufflenet» where
  root := `Bestiary.ShuffleNet

lean_exe «bestiary-evoformer» where
  root := `Bestiary.Evoformer

lean_exe «bestiary-muzero» where
  root := `Bestiary.MuZero

lean_exe «bestiary-mobilevit» where
  root := `Bestiary.MobileViT

lean_exe «bestiary-wavenet» where
  root := `Bestiary.WaveNet

lean_exe «bestiary-nerf» where
  root := `Bestiary.NeRF

lean_exe «bestiary-clip» where
  root := `Bestiary.CLIP

lean_exe «bestiary-squeezenet» where
  root := `Bestiary.SqueezeNet

lean_exe «bestiary-lenet» where
  root := `Bestiary.LeNet

lean_exe «bestiary-inception» where
  root := `Bestiary.Inception

lean_exe «bestiary-xception» where
  root := `Bestiary.Xception

lean_exe «bestiary-alexnet» where
  root := `Bestiary.AlexNet

lean_exe «bestiary-bert» where
  root := `Bestiary.BERT

lean_exe «bestiary-shufflenetv2» where
  root := `Bestiary.ShuffleNetV2

lean_exe «bestiary-gpt» where
  root := `Bestiary.GPT

lean_exe «bestiary-diffusion» where
  root := `Bestiary.Diffusion

lean_exe «bestiary-sam» where
  root := `Bestiary.SAM

lean_exe «bestiary-whisper» where
  root := `Bestiary.Whisper

lean_exe «bestiary-llava» where
  root := `Bestiary.LLaVA

lean_exe «bestiary-stable-diffusion» where
  root := `Bestiary.StableDiffusion

lean_exe «bestiary-segformer» where
  root := `Bestiary.SegFormer

lean_exe «bestiary-vae» where
  root := `Bestiary.VAE

lean_exe «bestiary-deeplab» where
  root := `Bestiary.DeepLabV3Plus

lean_exe «bestiary-maskrcnn» where
  root := `Bestiary.MaskRCNN

lean_exe «bestiary-dcgan» where
  root := `Bestiary.DCGAN

lean_exe «bestiary-cyclegan» where
  root := `Bestiary.CycleGAN

lean_exe «bestiary-alphago» where
  root := `Bestiary.AlphaGo

lean_exe «bestiary-pix2pix» where
  root := `Bestiary.Pix2Pix

lean_exe «bestiary-nystromformer» where
  root := `Bestiary.Nystromformer

lean_exe «bestiary-qanet» where
  root := `Bestiary.QANet

require checkdecls from git "https://github.com/PatrickMassot/checkdecls.git"

-- ═══════════════════════════════════════════════════════════════════════
-- Demo groups: one command that builds + runs a curated chunk of trainers,
-- tiered by time budget. `lake run mnist` (~30 min) / `lake run cifar` (~1 hr);
-- anything bigger is a deliberate single-model run (see run.sh). Backend
-- auto-detects (cuda if `nvidia-smi` is present, else rocm) but `IREE_BACKEND`
-- overrides; GPU honors `LEAN_DEMO_GPU` (default 0). Each trainer streams live
-- and tees to `<name>.log` via run.sh.
-- ═══════════════════════════════════════════════════════════════════════

/-- cuda when an NVIDIA GPU is visible (`nvidia-smi -L` succeeds), else rocm. -/
private def detectBackend : IO String := do
  try
    let o ← IO.Process.output { cmd := "nvidia-smi", args := #["-L"] }
    pure (if o.exitCode == 0 then "cuda" else "rocm")
  catch _ => pure "rocm"

/-- Build then run each named trainer in sequence (streaming) via `run.sh`. -/
private def runDemoGroup (names : List String) : IO UInt32 := do
  let backend ← match ← IO.getEnv "IREE_BACKEND" with
    | some b => pure b
    | none   => detectBackend
  let gpu := (← IO.getEnv "LEAN_DEMO_GPU").getD "0"
  -- The trainers shell out to `iree-compile`; put the project venv on PATH so
  -- `lake run` works without pre-activating it (the usual one-click footgun).
  let venvBin := (← IO.currentDir) / ".venv" / "bin"
  let runEnv ← do
    if ← System.FilePath.pathExists (venvBin / "iree-compile") then
      pure #[("PATH", some s!"{venvBin}:{(← IO.getEnv "PATH").getD ""}")]
    else pure #[]
  for n in names do
    IO.println s!"\n━━━ {n}: build ━━━"
    let bp ← IO.Process.spawn { cmd := "lake", args := #["build", n] }
    if (← bp.wait) != 0 then
      IO.eprintln s!"build failed: {n}"
      return 1
    IO.println s!"━━━ {n}: run (gpu {gpu}, {backend}) ━━━"
    let rp ← IO.Process.spawn { cmd := "./run.sh", args := #[n, gpu, backend], env := runEnv }
    let _ ← rp.wait
  return 0

/-- `lake run mnist` — the verified-MNIST demos (linear/MLP/CNN), ~30 min. -/
script mnist do
  runDemoGroup ["mnist-linear-verified", "mnist-mlp-verified", "mnist-cnn-verified"]

/-- `lake run cifar` — the ch.5 verified cifar8 variants: SGD/momentum/adam ×
    bn/no-bn, ~1 hr. -/
script cifar do
  runDemoGroup ["cifar8-verified", "cifar8-bn-verified",
                "cifar8-verified-momentum", "cifar8-bn-verified-momentum",
                "cifar8-verified-adam", "cifar8-bn-verified-adam"]

/-- `lake run imagenette` — the Part-I verified Imagenette trainers (the rest of
    the chapters: ResNet-34, MobileNetV2, EfficientNet-B0, ConvNeXt-T, ViT-Tiny),
    80-epoch AdamW at 224². **~37 h end-to-end** (9.5 + 5.4 + 6.2 + 13.3 + 2.3,
    single 7900 XTX — per the ViT-chapter results table) — a real time
    investment, not a quick demo. -/
script imagenette do
  runDemoGroup ["resnet34-verified-adam", "mobilenetv2-verified-adam",
                "efficientnet-verified-adam", "convnext-verified-adam",
                "vit-verified-adam"]

-- ═══════════════════════════════════════════════════════════════════════
-- `lake run download` — fetch the core datasets the verified trainers + the
-- benchmark need. Each entry pairs a download script with a sentinel file that
-- proves the dataset is already on disk, so re-running is a fast no-op. Used both
-- by `lake run download` and by `lake run benchmark` (which auto-downloads any
-- missing dataset as its first step, instead of soft-failing on imagenette).
-- ═══════════════════════════════════════════════════════════════════════

/-- The core datasets: `(label, download script, sentinel that exists once it's
    downloaded)`. MNIST + CIFAR feed the benchmark's dense/conv probes; Imagenette
    feeds the ViT/attn probe and the `lake run imagenette` tier. -/
def coreDatasets : List (String × String × String) :=
  [ ("MNIST",      "download_mnist.sh",      "data/train-images-idx3-ubyte"),
    ("CIFAR-10",   "download_cifar.sh",      "data/cifar-10/data_batch_1.bin"),
    ("Imagenette", "download_imagenette.sh", "data/imagenette/train.bin") ]

/-- Run a dataset's download script (via `bash`, so the exec bit doesn't matter)
    if its sentinel file is missing. Returns `false` on a download failure. -/
def ensureDataset (label sh sentinel : String) : IO Bool := do
  if ← System.FilePath.pathExists sentinel then
    IO.println s!"  ✓ {label} present ({sentinel})"
    return true
  IO.println s!"  ▸ {label} missing — running ./{sh} …"
  let rp ← IO.Process.spawn { cmd := "bash", args := #[sh] }
  if (← rp.wait) != 0 then
    IO.eprintln s!"    ✗ download failed: ./{sh}"
    return false
  pure (← System.FilePath.pathExists sentinel)

/-- Download any missing core dataset. Returns `false` if any download failed. -/
def ensureCoreData : IO Bool := do
  let mut ok := true
  for (label, sh, sentinel) in coreDatasets do
    if !(← ensureDataset label sh sentinel) then ok := false
  pure ok

/-- `lake run download` — fetch the core datasets (MNIST, CIFAR-10, Imagenette)
    that the verified trainers and `lake run benchmark` need, downloading only the
    ones not already on disk. Imagenette additionally needs `python3` + Pillow for
    the binary preprocessing step (see ./download_imagenette.sh). -/
script download do
  IO.println "━━━ lake run download ━━━ core datasets: MNIST, CIFAR-10, Imagenette"
  if ← ensureCoreData then
    IO.println "\n  ✓ all core datasets present."
    return 0
  else
    IO.eprintln "\n  ✗ one or more downloads failed (see above)."
    return 1

-- ═══════════════════════════════════════════════════════════════════════
-- `lake run benchmark` — estimate the book's training time on YOUR gpu.
-- Probes two fast verified nets for a few epochs (the only thing that runs),
-- reads steady-state ms/epoch from the trainer's own per-epoch print, and
-- scales the reference per-chapter wall-clock by the measured hardware factor.
-- Backend auto-detects (cuda if nvidia-smi present, else rocm) — works on
-- either vendor out of the box. A dense factor (MNIST-MLP) and a conv factor
-- (CIFAR-8-BN) scale the dense- vs conv-dominated chapters independently.
--
-- REFERENCE NUMBERS below are per-chapter *training* wall-clock on a single AMD
-- 7900 XTX (gfx1100, ROCm 7.2), verified-IREE path. The MNIST/CIFAR rows and all
-- three probe anchors (dense/conv/attn) were MEASURED directly from these verified
-- trainers (steady-state ms/{epoch,step} × the trainer's epoch/step count); the
-- R34/MNv2/ENet/ConvNeXt Imagenette rows are the verified-adam tier runs (9.5h /
-- 5.4h / 6.2h / 13.3h) and the ViT row is measured here (7.8h warm — the 2.3h
-- figure elsewhere is the JAX bf16 path, not this verified trainer). All EXCLUDE the
-- one-time IREE compile (~10–15 min/arch, CPU-bound, ~hardware-independent).
-- Re-running this benchmark on a 7900 XTX reproduces all three anchors (every
-- factor reads ~1.0×); regenerate the other Imagenette rows from a clean full run.
-- ═══════════════════════════════════════════════════════════════════════

structure BenchItem where
  chapter : String
  family  : String          -- "dense" | "conv"
  refSec  : Nat             -- reference training wall-clock (s) on the 7900 XTX
  tier    : String          -- "" | "mnist" | "cifar" | "imagenette"

def benchTable : List BenchItem :=
  [ { chapter := "2  MNIST linear", family := "dense", refSec := 6,     tier := "mnist" },      -- 535ms × 12
    { chapter := "3  MNIST MLP",    family := "dense", refSec := 38,    tier := "mnist" },      -- 3200ms × 12
    { chapter := "4  MNIST CNN",    family := "conv",  refSec := 238,   tier := "mnist" },      -- 23764ms × 10
    { chapter := "5  CIFAR x6",     family := "conv",  refSec := 2038,  tier := "cifar" },      -- 8490ms × 40 × 6
    { chapter := "6  ResNet-34",    family := "conv",  refSec := 34200, tier := "imagenette" }, -- 9.5h
    { chapter := "7  MobileNetV2",  family := "conv",  refSec := 19440, tier := "imagenette" }, -- 5.4h
    { chapter := "8  EfficientNet", family := "conv",  refSec := 22320, tier := "imagenette" }, -- 6.2h
    { chapter := "9  ConvNeXt",     family := "conv",  refSec := 47880, tier := "imagenette" }, -- 13.3h
    { chapter := "10 ViT",          family := "attn",  refSec := 27966, tier := "imagenette" } ]-- 7.8h (1185ms/step × 295 × 80, warm steady-state)

/-- Steady-state ms/epoch on the reference 7900 XTX for the two anchors, measured by
    the synthetic-input probe (`LEAN_MLIR_BENCH_SYNTH`): one constant batch reused at
    the dataset's real step count, eval skipped — so the on-reference factor reads ~1.0×
    and no dataset download is needed. -/
def probeDenseRefMs : Nat := 3030   -- mnist-mlp-verified  (784→512→512→10)
def probeConvRefMs  : Nat := 8020   -- cifar8-bn-verified  (8-conv + BN, 512 head)
/-- ms/STEP on the reference 7900 XTX for the `attn` anchor — synthetic-input probe of
    vit-verified-adam, reported as the median of a 100-step window (robust to the
    cold-cache / GC-blip outliers that made the old 40-step mean swing ±10%+).
    Step-based, not per-epoch: a ViT epoch is too slow to probe, and ViT's
    matmul/attention cost scales unlike conv across GPUs — so transformers get their
    own factor. (The 2.3h ViT figure elsewhere is the JAX bf16 path, not this
    verified-IREE trainer, which is ~7.8h here.) -/
def probeAttnRefMs : Nat := 1173

/-- Scale a chapter's reference seconds by the measured per-family factor. `aMs` is
    the attn ms/step probe (0 when no imagenette → attn falls back to the conv
    factor, which is known ~3.5x low for ViT). -/
def yourSecOf (it : BenchItem) (dMs cMs aMs : Nat) : Nat :=
  if it.family == "dense" then it.refSec * dMs / probeDenseRefMs
  else if it.family == "attn" then
    if aMs == 0 then it.refSec * cMs / probeConvRefMs        -- fallback: conv proxy
    else it.refSec * aMs / probeAttnRefMs
  else it.refSec * cMs / probeConvRefMs

/-- Human duration from whole seconds: `45s` / `12m` / `9.5h`. -/
def fmtDur (sec : Nat) : String :=
  if sec < 90 then s!"{sec}s"
  else if sec < 5400 then s!"{(sec + 30) / 60}m"
  else let t := (sec * 10 + 1800) / 3600; s!"{t / 10}.{t % 10}h"

/-- `num/den` as a 2-decimal multiplier string, e.g. `1.24`. -/
def fmtFactor (num den : Nat) : String :=
  if den == 0 then "?" else
  let h := num * 100 / den
  s!"{h / 100}.{(h % 100) / 10}{h % 10}"

/-- Right-pad to a fixed width for column alignment. -/
def padR (s : String) (n : Nat) : String :=
  if s.length < n then s ++ String.ofList (List.replicate (n - s.length) ' ') else s

/-- Pull the steady-state (last) `(<n>ms)` epoch timing out of a trainer's stdout. -/
def lastEpochMs (out : String) : Option Nat :=
  let eps := (out.splitOn "\n").filter fun l =>
    (l.splitOn "epoch ").length > 1 && (l.splitOn "ms)").length > 1
  match eps.getLast? with
  | none => none
  | some line =>
    match (line.splitOn "(").getLast? with
    | none => none
    | some s => match s.splitOn "ms)" with
                | h :: _ => h.toNat?
                | []     => none

/-- Pull `<n> ms/step` out of a `PROBE:` line (the LEAN_MLIR_MAX_STEPS path). -/
def probeMsStep (out : String) : Option Nat :=
  match ((out.splitOn "\n").filter fun l => (l.splitOn "PROBE:").length > 1).getLast? with
  | none => none
  | some line => match (line.splitOn "PROBE: ").getLast? with
    | none => none
    | some s => match s.splitOn " ms/step" with
                | h :: _ => h.toNat?
                | []     => none

/-- First run of consecutive digits in `s` as a Nat (tolerates surrounding text). -/
def firstNat (s : String) : Option Nat :=
  let ds := (s.toList.dropWhile (fun c => !c.isDigit)).takeWhile (·.isDigit)
  if ds.isEmpty then none else (String.ofList ds).toNat?

/-- Best-effort GPU utilization % (rocm-smi / nvidia-smi); none if the tool is absent. -/
def gpuBusyPct (backend : String) : IO (Option Nat) := do
  try
    if backend == "cuda" then
      let o ← IO.Process.output { cmd := "nvidia-smi", args := #["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"] }
      pure (firstNat o.stdout)
    else
      let o ← IO.Process.output { cmd := "rocm-smi", args := #["--showuse"] }
      pure (((o.stdout.splitOn "\n").find? (fun l => (l.splitOn "use (%)").length > 1)).bind firstNat)
  catch _ => pure none

/-- Build + run one probe net and return its steady-state timing. With `stepProbe`
    set (`attn` anchor) it caps at N steps and reads ms/step; otherwise it runs 3
    epochs and reads ms/epoch. -/
def runProbe (bin family : String) (refMs : Nat) (backend gpu : String)
    (runEnv : Array (String × Option String)) (stepProbe : Option Nat := none) : IO (Option Nat) := do
  let what := match stepProbe with | some n => s!"{n} steps" | none => "3 epochs"
  IO.println s!"\n  ▸ probing {bin} ({family}) — build + {what}…"
  let bp ← IO.Process.spawn { cmd := "lake", args := #["build", bin] }
  if (← bp.wait) != 0 then
    IO.eprintln s!"    build failed: {bin}"
    return none
  let vis := if backend == "cuda" then "CUDA_VISIBLE_DEVICES" else "HIP_VISIBLE_DEVICES"
  let capEnv := match stepProbe with
    | some n => #[("LEAN_MLIR_MAX_STEPS", some (toString n))]
    | none   => #[("LEAN_MLIR_MAX_EPOCHS", some "3")]
  let env := runEnv ++ capEnv ++ #[("IREE_BACKEND", some backend), (vis, some gpu),
                                    ("LEAN_MLIR_BENCH_SYNTH", some "1")]
  let o ← IO.Process.output { cmd := s!".lake/build/bin/{bin}", args := #["data"], env := env }
  let parsed := match stepProbe with | some _ => probeMsStep o.stdout | none => lastEpochMs o.stdout
  match parsed with
  | none =>
      IO.eprintln s!"    no timing for {bin} (data present? exit {o.exitCode})"
      pure none
  | some ms =>
      let unit := match stepProbe with | some _ => "ms/step" | none => "ms/epoch"
      IO.println s!"    {ms} {unit}   [ref {refMs}]   → {fmtFactor ms refMs}× the 7900 XTX"
      pure (some ms)

/-- `lake run benchmark` — probe this GPU, print a per-chapter training-time estimate. -/
script benchmark do
  let backend ← match ← IO.getEnv "IREE_BACKEND" with
    | some b => pure b
    | none   => detectBackend
  let gpu := (← IO.getEnv "LEAN_DEMO_GPU").getD "0"
  let venvBin := (← IO.currentDir) / ".venv" / "bin"
  let runEnv ← do
    if ← System.FilePath.pathExists (venvBin / "iree-compile") then
      pure #[("PATH", some s!"{venvBin}:{(← IO.getEnv "PATH").getD ""}")]
    else pure #[]
  IO.println "━━━ lake run benchmark ━━━ verified-NN training throughput on your GPU"
  IO.println s!"  backend: {backend}   gpu: {gpu}   (synthetic-input probes — no dataset needed)"
  -- Pre-flight: a busy GPU inflates every probe. Warn if the card isn't idle.
  match ← gpuBusyPct backend with
  | some u => IO.println (if u > 20 then
      s!"  ⚠ GPU is {u}% busy — close other GPU jobs first or the estimate will inflate."
      else s!"  GPU idle ({u}%).")
  | none   => pure ()
  -- Synthetic input (LEAN_MLIR_BENCH_SYNTH, set in runProbe): one constant batch reused
  -- at the dataset's real step count, so no MNIST/CIFAR/Imagenette download is required.
  let denseMs ← runProbe "mnist-mlp-verified" "dense" probeDenseRefMs backend gpu runEnv
  let convMs  ← runProbe "cifar8-bn-verified" "conv"  probeConvRefMs  backend gpu runEnv
  let attnMs ← runProbe "vit-verified-adam" "attn" probeAttnRefMs backend gpu runEnv (stepProbe := some 100)
  match denseMs, convMs with
  | some dMs, some cMs =>
    let aMs := attnMs.getD 0
    IO.println "\n  ESTIMATED training time on YOUR gpu  (ref = single AMD 7900 XTX):\n"
    let rule := "  " ++ String.ofList (List.replicate 47 '-')
    IO.println s!"  {padR "Chapter" 18}{padR "family" 8}{padR "ref(7900 XTX)" 15}your gpu"
    IO.println rule
    let mut yourTotal := 0
    let mut refTotal := 0
    for it in benchTable do
      let yourSec := yourSecOf it dMs cMs aMs
      yourTotal := yourTotal + yourSec
      refTotal := refTotal + it.refSec
      let flag := if it.family == "attn" && aMs == 0 then " *proxy" else ""
      IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR (fmtDur it.refSec) 15}{fmtDur yourSec}{flag}"
    IO.println rule
    IO.println s!"  {padR "Full Part-1 training" 26}{padR (fmtDur refTotal) 15}{fmtDur yourTotal}"
    IO.println "\n  `lake run` tiers on your gpu (training time):"
    for (tier, label) in [("mnist", "lake run mnist"), ("cifar", "lake run cifar"),
                          ("imagenette", "lake run imagenette")] do
      let items := benchTable.filter (·.tier == tier)
      let refS := (items.map (·.refSec)).foldl (· + ·) 0
      let yourS := (items.map (fun it => yourSecOf it dMs cMs aMs)).foldl (· + ·) 0
      IO.println s!"    {padR label 22}{padR (fmtDur refS) 9}→  {fmtDur yourS}"
    IO.println "\n  * dense/conv/attn rows each use their own probe (mnist-mlp, cifar8-bn, ViT);"
    IO.println "    other 224² convnets are extrapolated from the 32² conv probe — order-of-"
    IO.println "    magnitude. A `*proxy` ViT row means no imagenette, so it borrowed the conv"
    IO.println "    factor (~3.5× low). Training time only; first run adds ~10–15 min/arch compile."
    return 0
  | _, _ =>
    IO.eprintln "\n  probe failed — need data (`lake run download`) and the IREE venv from"
    IO.eprintln "  Track 2. No estimate produced."
    return 1
