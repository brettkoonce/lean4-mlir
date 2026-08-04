import Lake
open Lake DSL

package «lean4-mlir» where
  version := v!"0.6.3"
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
  "https://github.com/leanprover-community/mathlib4" @ "v4.32.2"

-- The everything-root: codegen + IREE FFI + trainers' shared modules +
-- the proof suite, all pulled in transitively via `LeanMlir.lean`.
-- `lake build LeanMlir` type-checks the whole repo.
lean_lib «LeanMlir» where
  roots := #[`LeanMlir]

-- Scoped targets, so CI and contributors can build one slice without the
-- rest. `LeanMlir` above is all-or-nothing; these split it along the same
-- seam the codebase already has (the proof suite imports only Mathlib —
-- never the codegen — so `Proofs` skips Types/Spec/MlirCodegen/Train).

/-- **`lake build Proofs`** — the fast per-push slice: the IR/render layer
    every demo's import cone actually reaches (StableHLO/IR + the per-net
    op/VJP modules the proven renderers are built on) plus every renderer the
    CI drift guard re-elaborates. ~20 roots, builds in minutes. The
    certificate corpus lives in `Certs` below and is checked by its own
    workflow (.github/workflows/certs.yml: proof-path pushes + nightly cron),
    so demo/book/engine pushes stop paying the multi-hour corpus tail.
    Split per planning/repo_shape_deletion_audit.md §4 (2026-07-06). -/
@[default_target]
lean_lib «Proofs» where
  srcDir := "."
  roots := #[-- the demo-cone IR/render layer
             `LeanMlir.Proofs.Architectures.Attention, `LeanMlir.Proofs.Architectures.CNN,
             `LeanMlir.Proofs.Architectures.Depthwise, `LeanMlir.Proofs.Architectures.MobileNetV2,
             `LeanMlir.Proofs.Architectures.ConvNeXt, `LeanMlir.Proofs.Architectures.EfficientNet,
             `LeanMlir.Proofs.Architectures.MnistCNN, `LeanMlir.Proofs.Foundation.StridedConv,
             `LeanMlir.Proofs.Foundation.PerChannelBN, `LeanMlir.Proofs.Foundation.IR,
             `LeanMlir.Proofs.Codegen.StableHLO,
             -- the renderers the verified-render drift guard re-elaborates
             `LeanMlir.Proofs.Codegen.MlpRender, `LeanMlir.Proofs.Codegen.CnnRender,
             `LeanMlir.Proofs.Codegen.ResNet34Render, `LeanMlir.Proofs.Codegen.ResNet34RenderB,
             `LeanMlir.Proofs.Codegen.AdamRender,
             `LeanMlir.Proofs.Codegen.MobileNetV2Render,
             `LeanMlir.Proofs.Codegen.MobileNetV2RenderB,
             `LeanMlir.Proofs.Codegen.EfficientNetRender,
             `LeanMlir.Proofs.Codegen.ConvNeXtRender,
             -- ⚠ A ROOT, and it has to be: nothing imports a leaf renderer, so on a fresh
             -- runner this is the ONLY way its `.olean` — and therefore the six stochastic-
             -- depth artifacts its `#eval`s write — ever get built (scripts/check_render_coverage.py).
             `LeanMlir.Proofs.Codegen.ConvNeXtRenderB,
             `LeanMlir.Proofs.Codegen.ViTRender, `LeanMlir.Proofs.Codegen.ViTRenderB]

/-- **`lake build Certs`** — the certificate corpus (155 files, ~87k lines:
    FloatBridges, ties, seals, descent, Lipschitz/LipSDP, Muon, …): the VJP
    proof suite's apex modules; their transitive imports cover every proof
    file (they subsume the `Proofs` roots above, so building `Certs` builds
    everything the axiom audit needs). Built + 3-axiom-audited by
    .github/workflows/certs.yml, NOT by the per-push proofs workflow. -/
lean_lib «Certs» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Architectures.Attention, `LeanMlir.Proofs.Architectures.CNN,
             `LeanMlir.Proofs.Architectures.Depthwise, `LeanMlir.Proofs.Architectures.MobileNetV2,
             `LeanMlir.Proofs.Architectures.ConvNeXt, `LeanMlir.Proofs.Architectures.EfficientNet,
             -- Chapter-4 MNIST 2D CNN (no BN): conditional whole-net VJP
             -- + a concrete instance with every smoothness hyp discharged.
             `LeanMlir.Proofs.Architectures.MnistCNN,
             -- Nonzero-Jacobian seal (planning/whole_network_backward.md Item B): the
             -- generic "one nonzero Jacobian entry ⇒ non-trivial backward" bridge.
             `LeanMlir.Proofs.Training.JacobianSeal,
             -- Item B2: the seal discharged at the live MobileNetV2 witness —
             -- `fderiv ℝ forward 0 ≠ 0` ⇒ non-trivial whole-net backward (level 3).
             `LeanMlir.Proofs.Training.MobileNetV2JacobianSeal,
             -- Chapter-6 ResNet Milestone B: stride-2 SAME convolution (the hard
             -- new downsampling op) = decimate ∘ stride-1 conv, with its input-VJP.
             `LeanMlir.Proofs.Foundation.StridedConv,
             -- Chapter-6 ResNet Milestone B: the deep-block chain (a list of
             -- same-type residual blocks composes to one VJP) — 16-block depth.
             `LeanMlir.Proofs.Foundation.ResNet34,
             -- Chapter-6 ResNet Milestone B8: per-channel BatchNorm (block-diagonal
             -- VJP via a per-row generalization of `rowwise_has_vjp_mat`).
             `LeanMlir.Proofs.Foundation.PerChannelBN,
             -- Limit-D strengthening: the 224×224 live ResNet-34 whole-net VJP with
             -- the three downsample projection convs generalized to ARBITRARY kernels
             -- (the β-positivity discharge is weight-independent). 3-axiom clean.
             `LeanMlir.Proofs.Foundation.ResNet34LiveGeneric,
             -- opt-in Mathlib.Matrix interop; not imported by the suite,
             -- listed here so CI keeps it green.
             `LeanMlir.Proofs.Codegen.MatBridge,
             -- denoted StableHLO-subset IR (Phase 0a/0b spike); bridges the
             -- emitted backward graph to the proven HasVJP.backward.
             `LeanMlir.Proofs.Foundation.IR,
             -- R4 printer-faithfulness Stage A (ch 2): StableHLO-subset AST +
             -- denotation `den` proven to match the linear train-step math.
             `LeanMlir.Proofs.Codegen.StableHLO,
             -- R4 syntactic core: op-graph serialization round-trip
             -- (parse (toToks (skel a)) = a).
             `LeanMlir.Proofs.Codegen.StableHLOParse,
             -- R4 syntactic LEXER numeric keystone: decimal Nat⟷String
             -- round-trip (parseNat (toString n) = n), the load-bearing
             -- first rung of text→token faithfulness.
             `LeanMlir.Proofs.Codegen.StableHLOLex,
             -- M1 (planning/verified_train_step.md): the linear train step bundled
             -- into one SGD-on-certified-softmax-CE-gradient theorem.
             `LeanMlir.Proofs.Foundation.LinearTrainStep,
             -- M2: the MLP per-layer parameter-gradient assembly (Crux A).
             `LeanMlir.Proofs.Foundation.MlpTrainStep,
             -- M3: the CNN convolution parameter-gradient bridges.
             `LeanMlir.Proofs.Foundation.CnnTrainStep,
             -- MLP render half: the train-step text as a name-threaded render of the
             -- proven forward graphs (multi-intermediate generalization).
             `LeanMlir.Proofs.Codegen.MlpRender,
             -- CNN render half: the CNN train-step text rendered from `cnnFwdGraph`,
             -- with flat→NCHW reshape glue bridging the conv param-grad tail.
             `LeanMlir.Proofs.Codegen.CnnRender,
             -- ResNet-34 render half: the full [3,4,6,3] train step (146 params) rendered
             -- from the verified AST — stem/16 residual blocks/GAP/dense, residual cotangent-
             -- sums at the skip merges; regenerates verified_mlir/resnet34_train_step.mlir.
             `LeanMlir.Proofs.Codegen.ResNet34Render,
             -- CIFAR-BN close: the per-channel BN scale/shift (dγ, dβ) param-grad
             -- bridges — the affine BN analogue of `bias_grad_bridge`.
             `LeanMlir.Proofs.Architectures.CifarBnClose,
             -- CNN conv-close upgrade: the conv param closes pinned to the actual
             -- backward-chain cotangent (Back3 maxpool/conv via flatDenote + relu masks).
             `LeanMlir.Proofs.Foundation.CnnChainClose,
             -- Deeper (8-conv) CIFAR-CNN close: cifar8{,Bn}FwdGraph_faithful's backward
             -- peer — each conv W/b, BN γ/β, dense W/b output pinned to the actual 4-stage
             -- backward-chain cotangent (the CnnChainClose recipe + BN, two more pool stages).
             `LeanMlir.Proofs.Architectures.Cifar8Close,
             -- MobileNetV2 close (Item C): the depthwise (stride-1/2) + strided-conv
             -- parameter-gradient bridges — every MobileNetV2 train-step param output
             -- certified θ − lr·(certified Jacobian · cotangent).
             `LeanMlir.Proofs.Architectures.MobileNetV2Close,
             -- MobileNetV2 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph
             -- (matches the operational render's BN flavor) + faithfulness to the
             -- per-channel ℝ-forward. Prerequisite for the structured render (Item B).
             `LeanMlir.Proofs.Codegen.MobileNetV2RenderPC,
             -- MobileNetV2 cotangent-chain close (Item D): the Item C conv/depthwise bridges
             -- pinned to the inverted-residual backward chain (relu6 kink + depthwise + stride-2).
             `LeanMlir.Proofs.Architectures.MobileNetV2ChainClose,
             -- The cotangent pass / = ∂loss/∂θ fold: the certified per-layer conv/depthwise
             -- Jacobian contracted with ∂loss/∂(layer output) IS the total loss gradient (pdiv_comp
             -- at a smooth point). The conv analogue of mlp_hidden_total_loss_grad; program-wide.
             `LeanMlir.Proofs.Foundation.ConvLossFold,
             -- EfficientNet-B0 close (Item C): a FREE close — every param family reuses an
             -- existing bridge (5×5 depthwise pinned; batch-norm γ/β = per-channel BN at m=N·h·w;
             -- SE squeeze/excite are dense → M2). No new VJP.
             `LeanMlir.Proofs.Architectures.EfficientNetClose,
             -- ResNet-34 close (Item C): a FREE close — every r34 param family certified
             -- by an existing bridge (the 7×7 stem + 3×3 strided projection pinned to the
             -- generic strided conv W/b bridges; no new VJP).
             `LeanMlir.Proofs.Foundation.ResNet34Close,
             -- ResNet-34 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph (full
             -- 16-block [3,4,6,3] net, 7×7 stem, maxpool) + per-block + whole-net faithfulness.
             `LeanMlir.Proofs.Codegen.ResNet34RenderPC,
             -- ResNet-34 cotangent-chain close (Item D): the Item C conv bridges pinned to the
             -- cotangent the backward chain delivers (id/downsample block + maxpool-back stem).
             `LeanMlir.Proofs.Foundation.ResNet34ChainClose,
             -- ConvNeXt close (Item C): mostly reuse (7×7 depthwise pinned to the generic
             -- bridges) + the two genuinely-new families — layer-scale γ (dγ = x⊙dy) and
             -- scalar-LN γ/β (the Vec-1 embedding bridging bn_grad_gamma/beta).
             `LeanMlir.Proofs.Architectures.ConvNeXtClose,
             -- ViT close (Item A): the distinct-param 2-block ViT forward (vitForward2 +
             -- whole-net VJP) and the heads=1 token forward graph + faithfulness
             -- (den vitFwdGraph = vitForward2 via mhsa_layer_one_head).
             `LeanMlir.Proofs.Architectures.ViTFwdGraph,
             -- ViT close (Item C): the per-token dense W/b family (row-lifted M2
             -- outer product), row-lifted scalar-LN γ/β, pos-embed identity, CLS
             -- masked-gather — every representative-ViT param family except the
             -- patch conv certified.
             `LeanMlir.Proofs.Architectures.ViTClose,
             -- ViT cotangent-chain close (Item D): the Item C bridges pinned to the
             -- attention-block backward chain (SDPA matmul chain = the proven
             -- sdpa_back_{Q,K,V} closed forms; the Q/K/V three-way fan-in at LN1).
             `LeanMlir.Proofs.Architectures.ViTChainClose,
             -- ViT scaling pass (vector-[D] LN): layerNormVec block + vitForward2V
             -- whole-net VJP + the rowScaleF/rowBiasF token graph + faithfulness +
             -- the per-channel gamma/beta param bridges.
             `LeanMlir.Proofs.Architectures.ViTVecLN,
             -- ViT scaling pass (multi-head + depth-k): headSliceF/headPadF tokens,
             -- mhsa at general heads, then the distinct-param depth-k tower
             -- (vitForwardKV). ViTDepthK imports ViTMultiHead, covering both.
             `LeanMlir.Proofs.Architectures.ViTDepthK,
             -- ViT multi-head backward cotangents: the per-head SDPA backward the real
             -- chain delivers at the Q/K/V dense outputs (Σ_h pad ∘ vitCotD{Q,K,V}(d_head)
             -- ∘ slice), pinned to the audited sdpa_back_{Q,K,V} (vitCotD{Q,K,V}mh_eq).
             -- The multi-head/depth-12 tie's substantive build (mnv2 reduced→full).
             `LeanMlir.Proofs.Architectures.ViTMultiHeadChain,
             -- EfficientNet-B0 at full depth (16 distinct MBConv blocks, true BN+SE):
             -- batched forward graph + whole-net VJP. Imports the EfficientNet
             -- RenderPC + ChainClose modules, covering all three.
             `LeanMlir.Proofs.Architectures.EfficientNetFullB0,
             -- Full ConvNeXt-T [3,3,9,3]: forward graph + faithfulness + whole-net
             -- VJP. Imports ConvNeXtChainClose, covering both.
             `LeanMlir.Proofs.Architectures.ConvNeXtFullT,
             -- Paper-spec full MobileNetV2 (all 17 [t,c,n,s] bottlenecks): forward
             -- graph + faithfulness.
             `LeanMlir.Proofs.Architectures.MobileNetV2FullPaper,
             -- ℝ→Float32 bridge, Tier 1: standard-model rounding (hypothesis-style,
             -- no axioms) + forward error bounds for the linear/MLP nets
             -- (dot/dense budgets, ReLU exact-in-float Lipschitz pass-through).
             `LeanMlir.Proofs.Float.FloatBridge,
             -- Subnormal-floor closure (planning §2): the honest FaithfulFloatModel
             -- (relative bound on normals + the gradual-underflow absolute floor),
             -- FloatModel = its η→0 face, the BN/LN denominator stays-normal
             -- invariant (rsqrt keystone never underflows), and the residual floor
             -- proved globally negligible. Converts FloatBridge's subnormal caveat
             -- into lemmas.
             `LeanMlir.Proofs.Float.FloatSubnormalBridge,
             -- Inexact-gradient descent over ℝ (MVT form): an η-accurate gradient
             -- oracle + segment smoothness ⇒ the SGD step still decreases the loss,
             -- with an explicit decrease. The keystone the FloatBridge budgets
             -- plug into ("close" ⇒ "still trains").
             `LeanMlir.Proofs.Training.SgdDescent,
             -- The smoothness hypothesis DISCHARGED for the Chapter-2 linear
             -- softmax-CE loss: explicit segment-Lipschitz constant 2a²/(1−2aD)
             -- via the softmax ratio sandwich (no Hessian), and the capstone —
             -- one inexact SGD step provably decreases the cross-entropy loss.
             `LeanMlir.Proofs.Training.SgdDescentLinear,
             -- The smoothness hypothesis discharged through the Chapter-3
             -- MLP: under quantitative ReLU margins (the step cannot flip a
             -- mask sign) the loss-of-one-layer maps get explicit
             -- segment-Lipschitz constants, and one inexact SGD step on each
             -- weight layer provably decreases the cross-entropy loss.
             `LeanMlir.Proofs.Training.SgdDescentMlp,
             -- The descent program reaches the Chapter-4 CNN: quantitative
             -- max-pool selection margins (the argmax freezes along the step
             -- segment), pool 1-Lipschitz/ℓ1-contraction, conv kernel drift.
             `LeanMlir.Proofs.Training.SgdDescentCnn,
             -- CIFAR-8 last-conv SGD descent: the first non-MNIST provable descent. CIFAR-8's tail
             -- (conv W₈ → relu → maxpool → 3 denses) IS cnn_conv2's architecture, so descent at the
             -- last conv (earlier 7 layers frozen) is an instance — non-vacuous lr. Full-depth descent
             -- stays open (the per-layer operator-norm product in hsmall compounds to vacuity).
             `LeanMlir.Proofs.Training.SgdDescentCifar,
             -- ℝ→Float32 forward rounding budget for the no-BN CIFAR CNN
             -- (cnn_float_close scaled to 4 conv + 2 maxpool + 3 dense).
             `LeanMlir.Proofs.Float.CifarFloatBridge,
             -- BN float keystone: 1/√ Lipschitz on [ε,∞) + the inverse-stddev
             -- rounding budget (rsqrt accuracy + variance error, ε-floor).
             `LeanMlir.Proofs.Float.BnFloatBridge,
             -- residual additive fan-in float closeness (add_close / reluAdd_close)
             -- — the new structural op toward the ResNet-34 float bridge.
             `LeanMlir.Proofs.Float.Resnet34FloatBridge,
             -- real-BN input-sensitivity (mean/var/istd/forward Lipschitz) — the
             -- per-block composition enabler (the float BN's input is perturbed).
             `LeanMlir.Proofs.Codegen.BnInputBridge,
             -- first assembled ResNet block step: relu(BN(·)) at a perturbed BN
             -- input = rounding (bnForward_close_of) + input-shift (bnForward_input_close).
             `LeanMlir.Proofs.Codegen.Resnet34BlockBridge,
             -- whole-net certificate backbone: FloatClose composes (moduli ∘, magnitudes
             -- thread) — the whole net is the fold of per-op budgets.
             `LeanMlir.Proofs.Float.FloatComposeBridge,
             -- Eval-mode BN/LN as a fixed affine (planning §4): running-stats at
             -- eval = a·x+b with a=γ/√(σ²+ε), b=β−γμ/√(σ²+ε) precomputed offline
             -- (no batch reduce, no runtime rsqrt). bnEvalAffine_fold proves the
             -- eval-BN formula IS that affine; floatClose_bnEval bridges it (one
             -- mul + one add, no fan-in γ) — the deployed-forward win.
             `LeanMlir.Proofs.Float.BnEvalFloatBridge,
             -- EfficientNet float bridge step 1: Swish/sigmoid (bounded, rounding
             -- closeness, σ is ¼-Lipschitz) — the shared smooth-activation transcendental.
             `LeanMlir.Proofs.Float.EnetFloatBridge,
             -- ViT float bridge apex: transitively imports ViTFloatBridge (LN/GELU/MLP) and
             -- ViTAttentionFloatBridge (sdpa_close + input-sensitivity); adds the transformer
             -- block fold + projections + single/multi-head (block-diagonal & full-d MHA).
             `LeanMlir.Proofs.Float.ViTBlockFloatBridge,
             -- A1 forward float-bridge capstones (planning/tier23…): the deeper 8-conv
             -- no-BN CIFAR via the FloatBridges.comp existential path (cifar8_floatBridges).
             `LeanMlir.Proofs.Float.Cifar8FloatBridge,
             -- The BatchNorm FloatBridges keystone: flat/global BN (floatBridges_bn,
             -- discharges the EfficientNet MBConv hbnE/D/P) + the per-channel block-diagonal
             -- lift via FloatClose.perRowIdx (floatBridges_bnPerChannelFlat) + the network
             -- Tensor3-layout conjugation by the reassoc permutations
             -- (floatBridges_bnPerChannelTensor3). The "do-it-once" BN infra A1/A3 share.
             `LeanMlir.Proofs.Float.BnPerChannelFloatBridge,
             -- A1 BatchNorm CIFAR forward capstone: cifarCnnBnForward float-bridges, the
             -- four per-channel BNs supplied as FloatBridges (each discharged by
             -- floatBridges_bnPerChannelTensor3). The BN-net peer of cifar8_floatBridges.
             `LeanMlir.Proofs.Float.CifarBnFloatBridge,
             -- A3 "other side" keystone (planning/tier23… A3): the BatchNorm BACKWARD
             -- float closeness — param grads (bnBetaGrad_close / bnGammaGrad_close) + the
             -- genuinely-new three-term input gradient (bnGradInput_close) + the reusable
             -- reduction_close / sub_close' helpers. The shared backward op every deep
             -- net's gradient folds (r34/mnv2/enet/convnext LN/vit LN).
             `LeanMlir.Proofs.Float.BnBackFloatBridge,
             -- A3 backward fold: the linear input-VJP (dx = Wᵀ·dy = bias-free dense over the
             -- transpose, reuses floatBridges_dense) + the exact ReLU-back selectPos mask
             -- (floatBridges_reluMaskBack) compose via FloatBridges.comp into a whole-net
             -- backward gradient bridge (mlpInputGrad_floatBridges) — the backward peer of
             -- cifar8_floatBridges.
             `LeanMlir.Proofs.Float.LinBackFloatBridge,
             -- A3 CNN backward (planning/a3_backward_deepnet_assembly.md 1a/1b + cifar8 witness):
             -- the maxpool-back select_and_scatter (floatBridges_maxPoolBack, exact masked gather,
             -- modulus id) + the conv input-VJP as a reversed-kernel flatConv (floatBridges_convBack,
             -- reuses floatBridges_flatConv) compose into the whole 8-conv CIFAR input-gradient VJP
             -- (cifar8_grad_floatBridges) — the backward peer of cifar8_floatBridges.
             `LeanMlir.Proofs.Float.CnnBackFloatBridge,
             -- A3 1c: BatchNorm BACKWARD as a composable FloatClose MAP over the cotangent
             -- (floatClose_bnBack / floatBridges_bnBack) — wraps the bnGradInput_close keystone
             -- with the real map's magnitude (bn_grad_input_abs_le) + Lipschitz-in-dy modulus
             -- (bn_grad_input_diff_abs_le, the real BN-back is linear in dy). The shared BN/LN
             -- backward op cifarBn/r34/convnext/vit gradients fold via .comp.
             `LeanMlir.Proofs.Codegen.BnBackComposeBridge,
             -- A3: per-channel BatchNorm BACKWARD float-bridge (floatBridges_bnPerChannelBack) —
             -- the block-diagonal FloatClose.perRowIdx lift of floatClose_bnBack conjugated by the
             -- reassoc layout gathers, bridging the certified bnPerChannelTensor3_grad_input. The
             -- backward peer of floatBridges_bnPerChannelTensor3; discharges cifarBn_grad's BN hyps.
             `LeanMlir.Proofs.Float.BnPerChannelBackFloatBridge,
             -- A3 r34 backward: the residual identity-block input-VJP (floatBridges_r34IdBlockBack) —
             -- relu(F(x)+x) backward = residual bF ∘ reluMaskBack, the residual-skip backward reusing
             -- FloatBridges.residual (NO new combinator; the rounded skip-add is the backward's too),
             -- bF = convFlatBack∘bnBack∘reluMaskBack∘convFlatBack∘bnBack. The dominant r34 block.
             `LeanMlir.Proofs.Float.Resnet34BackFloatBridge,
             -- A3 strided-conv backward (r34 down-blocks + stem): flatConvStride2 = decimateFlat ∘
             -- flatConv, so its input-VJP = convFlatBack ∘ decimateBack (zero-upsample scatter then
             -- reversed-kernel conv). floatBridges_flatConvStride2Back via floatBridges_decimateBack
             -- (the scatter, exact in float / magnitude-nonincreasing by decimateIdx_injective —
             -- decimateBack IS the certified decimateFlat VJP, decimateBack_eq_vjp) .comp convBack.
             `LeanMlir.Proofs.Float.StridedConvBackFloatBridge,
             -- A3 r34 downsample-block backward: relu(proj+body) reversed — the two-branch fan-in
             -- (FloatBridges.biPathSum, the general f(x)+g(x) rounded sum, of which the identity
             -- block's f(x)+x is the g=id case) of the projection backward and the strided body
             -- backward, both using flatConvStride2Back. Completes the r34 block set.
             `LeanMlir.Proofs.Float.Resnet34DownBackFloatBridge,
             -- A3 r34 WHOLE-NET backward (the first Imagenette whole-net backward): gapBack (GAP
             -- VJP = broadcast÷(h·w)) + the [3,4,6,3] .comp fold (r34_grad_floatBridges) — concrete
             -- stem/GAP/maxpool/dense endpoints, the 16 blocks supplied as FloatBridges (discharged
             -- by floatBridges_r34IdBlockBack/DownBlockBack). The exact reverse of resnet34Forward.
             `LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridge,
             -- A3 r34 WHOLE-NET FORWARD (the forward peer of r34_grad_floatBridges): the missing
             -- forward op-bridges floatBridges_flatConvStride2 (stem, = flatConv read at decimateIdx)
             -- + floatBridges_gap (wraps floatClose_gap) + the [3,4,6,3] .comp fold (r34_floatBridges)
             -- — concrete stem/maxpool/GAP/dense endpoints, stem BN + 16 blocks supplied as
             -- FloatBridges. Closes the forward/backward whole-net asymmetry.
             `LeanMlir.Proofs.Float.Resnet34WholeFloatBridge,
             -- §B integrity tie: the r34 IDENTITY-BLOCK backward float bridge targets the CERTIFIED
             -- VJP. Same-vocabulary (per-channel BN, non-batched) target rblkPC_has_vjp_at — built
             -- here, mirrors resblock_has_vjp_at — + the conv-leaf tie (convFlatBack_eq_vjp_backward,
             -- via IR.convBackDenote_eq_input_grad_formula) ⇒ r34IdBlockBack(pinned) = its .backward.
             -- b1-free (no batched↔non-batched reconciliation).
             `LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie,
             -- A3 §1e depthwise backward (mnv2/enet/convnext blocker): the depthwise input-VJP is a
             -- forward depthwise conv at the spatially-reversed kernel (dwReverse, FREE reuse of
             -- floatBridges_depthwise — the depthwise twin of convBack); strided variant =
             -- depthwiseFlatBack ∘ decimateBack. floatBridges_depthwiseBack/depthwiseStride2Back.
             `LeanMlir.Proofs.Float.DepthwiseBackFloatBridge,
             -- A3 §1e Squeeze-Excite backward (the architecturally-distinctive product-rule op):
             -- seBack(dy) = (g⊙dy) + gateBack(x⊙dy), the two-branch multiplicative fan-in
             -- (FloatBridges.biPathSum of two diagBacks) with the gate sub-net backward gateBack
             -- supplied/assembled. floatBridges_seBack + floatBridges_seGateBack (gapBack∘linBack∘
             -- swishBack∘linBack∘sigmoidBack∘broadcastBack) + floatBridges_broadcastBack (the new
             -- spatial-reduce op, via the BN-back reduction_close machinery).
             `LeanMlir.Proofs.Float.SEBackFloatBridge,
             -- A3 Part 2 per-net backward assembly (consumes §1e). MobileNetV2: the inverted-residual
             -- body backward (expandBack∘depthwiseBack∘projectBack, depthwiseFlatBack concrete — mnv2
             -- has NO SE) + strided variant; mnv2_grad_floatBridges = whole-net fold (reverse of
             -- mobilenetv2Forward_full_pc), concrete stem/head/GAP/dense, 6 blocks supplied.
             `LeanMlir.Proofs.Float.MobileNetV2BackFloatBridge,
             -- MobileNetV2 WHOLE-NET FORWARD (forward peer of mnv2_grad_floatBridges): the ch7 6-block
             -- per-channel render mobilenetv2Forward_full_pc. New op-bridge floatBridges_relu6 (relu6 =
             -- min(max(·,0),6) exact in float + 1-Lipschitz, mirror of floatClose_relu via the mathlib
             -- clamp lemmas); strided depthwise reuses floatBridges_depthwiseStride2Flat. mnv2Forward =
             -- the ∘ skeleton (concrete stem/head/GAP/dense, stem/head BNs + 6 invres blocks supplied);
             -- named block bridges floatBridges_invresBody{,Strided}PC discharge them (no SE).
             `LeanMlir.Proofs.Float.MobileNetV2WholeFloatBridge,
             -- EfficientNet MBConv body backward (whole-net is batched → per-example body, peer of
             -- floatBridges_mbconvBody): expandBack∘depthwiseBack∘seBack∘projectBack — BOTH §1e ops
             -- (depthwiseFlatBack concrete + seBack supplied) land here. + the additive-skip variant.
             `LeanMlir.Proofs.Float.EfficientNetBackFloatBridge,
             -- EfficientNet WHOLE-NET FORWARD (forward peer of efficientnetForwardB_has_vjp): stated
             -- on the ACTUAL batched efficientnetForwardB (stem→MBConv1→MBConv6-strided→MBConv6-resid
             -- →head). Each batch-separable op is FloatBridges.batchMap of its op-bridge, swish is
             -- block-diagonal at the batched index, and the 10 true-batch-norms (bnBatchLA, batch-
             -- coupled) are supplied as FloatBridges facts. New op-bridge: floatBridges_depthwiseStride2Flat
             -- (mbStrided downsample, = depthwise read at decimateIdx, peer of floatBridges_flatConvStride2).
             `LeanMlir.Proofs.Float.EfficientNetWholeFloatBridge,
             -- EfficientNet WHOLE-NET BACKWARD (backward peer of efficientnetForwardB_floatBridges, the
             -- last entry in the 5-net × {fwd,bwd} matrix): efficientnet_grad_floatBridges = the batched
             -- .comp fold (reverse of head∘mbResid∘mbStrided∘mbNoExp∘stem) with concrete batchMap-lifted
             -- conv/GAP/dense endpoints and supplied BN/swish/block backs; + the no-exp/strided block-back
             -- bridges (mbNoExpBodyBack/mbStridedBodyBack) so every supplied block back is dischargeable.
             `LeanMlir.Proofs.Float.EfficientNetWholeBackFloatBridge,
             -- §2n: ConvNeXt's REAL channel LayerNorm (the §2m flip) in float, fwd AND bwd. Route A's
             -- conjugation is four gathers around one row map, so floatBridges_chanLNTensor3 is
             -- floatBridges_bnPerChannelTensor3's blueprint with a transpose inserted. Three new op
             -- bridges: floatBridges_transposeFlat (the transpose IS a gather, by rfl), floatBridges_
             -- biasAdd (the +β token), floatBridges_layerNormVec (= supplied LN(1,0) ∘ diagBack γ ∘ +β,
             -- rfl). Backward: rowLNVecFlatBack (perRowIdx of bn_grad_input ∘ diagBack γ) conjugated
             -- the same way — it discharges the LN-abstract lnB hyps ConvNeXtBackFloatBridge already has.
             `LeanMlir.Proofs.Float.ChannelLNFloatBridge,
             -- ConvNeXt-T backward (per-example): block body backward (depthwiseBack∘lnBack∘convBack∘
             -- geluBack∘convBack∘layerScaleBack) + residual block + downsample (lnBack∘stride2Back);
             -- convnext_grad_floatBridges = whole-net [3,3,9,3] fold, concrete GAP/dense, stem/stages/
             -- downsamples supplied.
             `LeanMlir.Proofs.Float.ConvNeXtBackFloatBridge,
             -- ConvNeXt-T WHOLE-NET FORWARD (forward peer of convnext_grad_floatBridges): the [3,3,9,3]
             -- fold of convNextForwardTCh. Two new op-bridges: floatBridges_layerScale (γ⊙x = diagBack γ
             -- definitionally, γ exact ⇒ es=0) + floatBridges_flatConvStride4 (the 4×4/s4 patchify stem
             -- = flatConv read at decimateOddIdx∘decimateIdx, two-decimation cousin of stride2). Named
             -- bridges floatBridges_cnxBlockWith (LN-abstract body; floatBridges_convNextBlock = its ch9
             -- instantiation) + floatBridges_convNextStageChK (depth-k fold) + floatBridges_cnxDownChW;
             -- convnextForward ∘-skeleton with
             -- stem-conv/GAP/dense concrete, stem/head LN + 4 stages + 3 downsamples supplied.
             `LeanMlir.Proofs.Float.ConvNeXtWholeFloatBridge,
             -- The skeleton↔real-net forward ties (item #5, cosmetic polish): each whole-net forward
             -- bridge is stated on a fresh skeleton (r34Forward/mnv2Forward/convnextForward) with abstract
             -- blocks; these rfl lemmas plug the concrete blocks (idFwd/downFwd, invresBody*PC,
             -- convNextStageChK/cnxDownChW) into the slots ⇒ each skeleton = THE committed real ℝ-forward def.
             `LeanMlir.Proofs.Foundation.WholeNetForwardTies,
             -- §B shared prerequisite: the DEPTHWISE adjoint gate (the depthwise twin of
             -- IR.convBackDenote_eq_input_grad_formula) — depthwiseConv2d (dwReverse W) 0 =
             -- depthwiseConv2d_input_grad_formula W, all dims/odd kernels, via Finset.sum_bij' on the
             -- pad supports (no Σ co) — plus the flat + strided depthwise leaf ties (depthwiseFlatBack
             -- = certified depthwise input-VJP). Unblocks the convnext/mnv2/enet §B ties.
             `LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie,
             -- §B integrity tie (convnext): cnxBlockBodyBack(pinned LN/gelu/layerScale backs) = the
             -- certified convNextBlockBody_has_vjp.backward — depthwise gate + 1×1 conv leaves + rfl;
             -- plus the residual-wrapped block tie. b1-free.
             `LeanMlir.Proofs.Architectures.ConvNeXtBackCertifiedTie,
             -- §B integrity tie (mnv2): build the per-channel-BN certified body VJP invresBodyPC_has_vjp_at
             -- (fresh, like r34's rblkPC) then tie invresBodyBackPC (+ strided) — relu6 masks pinned to the
             -- 0<preact<6 clamp-window signs, BN backs to bnPerChannelTensor3_has_vjp, depthwise via the
             -- gate. b1-free.
             `LeanMlir.Proofs.Architectures.MobileNetV2BackCertifiedTie,
             -- §B integrity tie (efficientnet): mbconvBodyBack(pinned bn/swish/SE backs) = the certified
             -- mbconvBody_has_vjp.backward — SE back pinned to seBlockFull_has_vjp, swish to swish_has_vjp,
             -- depthwise via the gate. Certified per-example body VJP already exists (global bnForward).
             `LeanMlir.Proofs.Architectures.EfficientNetBackCertifiedTie,
             -- §B integrity tie (vit MHSA — the sdpa adjoint): mhsaBackFlat (Q/K/V pinned to the actual
             -- dense projections at the saved input X) = the certified mhsa_has_vjp_mat.backward,
             -- flattened. Via ViTBackB0's mhsa_backward_collapseMH (certified Mat backward = per-head
             -- merged sum) + the projBack_core_coord/woback_unflatten coordinate match (dense Wᵀ = mulVec,
             -- Σk over h·dh reindexes to Σh Σj, separate projBacks regroup via sum_add_distrib).
             `LeanMlir.Proofs.Architectures.ViTMhsaBackCertifiedTie,
             -- A3 §1g loss-head cotangent seed: lift softmax_ce_cot_close to a FloatBridges seed
             -- (z ↦ softmax(z)−onehot, the CE input-gradient; bounded by 1+cotErr(0) since softmax∈[0,1])
             -- so any <net>_grad .comp it = the whole "logits → input-gradient" backward "from the loss".
             `LeanMlir.Proofs.Float.LossHeadCotFloatBridge,
             -- A3 §1f softmax-Jacobian backward (the vit/attention crux): the row-coupled VJP
             -- diag(p)−p·pᵀ, softmaxBack p dy i = pᵢ(dyᵢ−⟨p,dy⟩). Linear in dy (modulus = magnitude at
             -- Cdy:=e, like bnGradInput); float threads mul_close/reduction_close/sub_close' with the
             -- softmax weights supplied within smErr. floatClose_softmaxBack / floatBridges_softmaxBack.
             `LeanMlir.Proofs.Float.SoftmaxBackFloatBridge,
             -- A3 §1f Mat-space SDPA BACKWARD assembly (the vit-crux capstone): the backward peer of
             -- sdpa_close. Certified sdpa_back_{Q,K,V} = three matmuls + softmaxBack + a 1/√d scale;
             -- floats reuse attnScore_close (dw), attnDot_close (dQ/dK/dV at perturbed weights),
             -- softmaxBack_close/sub_abs_le (the row VJP), mul_close (scale). sdpaBack{V,Q,K}_close.
             `LeanMlir.Proofs.Float.SdpaBackFloatBridge,
             -- A3 §1f FULL multi-head self-attention backward assembly: the backward peer of
             -- floatBridges_mhProjAttnFull. dY ↦ dX = WoBack → 3 sdpa cores → Q/K/V projBacks + fan-in.
             -- The projBacks are FREE (per-token linBack); the cores (floatBridges_core{V,Q,K}) lift the
             -- flattened mhsaSdpaBack* to FloatClose (linear-in-cotangent). floatBridges_mhsaBack.
             `LeanMlir.Proofs.Float.MhsaBackFloatBridge,
             -- A3 §1f the ViT PATCH-EMBED backward (the last whole-net endpoint): the certified
             -- patchEmbed_input_grad_formula (a transposed-conv / guarded patch-scatter triple-sum, linear
             -- in the cotangent) float-bridges via dot_close (fan-in D) + nested reduction_close (the
             -- kw/kh/p sums). floatBridges_patchEmbedBack discharges vit_grad_floatBridges's hPatch.
             `LeanMlir.Proofs.Float.PatchEmbedBackFloatBridge,
             -- ViT WHOLE-NET FORWARD (forward peer of vit_grad_floatBridges): vit_full reversed =
             -- classifier ∘ perRowFlat finalLN ∘ tower blocks ∘ patchEmbed. The encoder tower REUSES
             -- towerBack (its head-first fold IS the forward order) + floatBridges_towerBack; the LN
             -- rides FloatBridges.perRow; the head (dense ∘ cls-slice) is concrete with the one new
             -- op-bridge floatBridges_clsSlice (the cls-slice gather, peer of clsScatter, exact). The
             -- per-row LN / blocks / patch-embed supplied as FloatBridges (blocks via floatBridges_vitBlock).
             `LeanMlir.Proofs.Float.ViTWholeFloatBridge,
             -- ViT PATCH-EMBED FORWARD (the last vit forward endpoint, peer of floatBridges_patchEmbedBack):
             -- patchEmbed_flat = pos_embed + (cls_token | b_conv + ∑c∑kh∑kw W·guarded-img), affine in the
             -- image (constants cancel in the diff). M.patchEmbedF rounds the leaf mul (mul_close), the 3
             -- c/kh/kw sums (nested reduction_close) and the 2 constant adds (add_close). floatBridges_
             -- patchEmbed discharges vit_floatBridges's hPatch ⇒ vit_floatBridges_concrete (fully concrete).
             `LeanMlir.Proofs.Float.PatchEmbedFloatBridge,
             -- The optimizer rung beyond SGD: the ℝ Adam/AdamW step mirroring
             -- the emitted update (Phase 3a of vit_train_to_vit_verified.md).
             -- Faithfulness target + denominator well-definedness; NO descent
             -- claim (Adam isn't monotone).
             `LeanMlir.Proofs.Codegen.AdamStep,
             -- The SGD / Nesterov peers of AdamStep (§2i). Same claim ceiling:
             -- faithfulness, NOT descent — nothing here claims Nesterov descends.
             `LeanMlir.Proofs.Codegen.SgdMomentumStep,
             -- Phase 3b: the AdamW render-close — emitted weight/bias update =
             -- adamWScalar of the certified gradient (sgdW_descends_certified_grad
             -- analogue, optimizer swapped for AdamW).
             `LeanMlir.Proofs.Codegen.AdamRender,
             -- Stage 2 of the live ResNet-34 (Item A2): the channel-order invariant
             -- kit (maxpool/BN/ReLU preserve strict pointwise channel domination —
             -- the non-vacuity carrier). Build-checked; not yet a live witness, so
             -- also NOT in the AuditAxioms headline set.
             `LeanMlir.Proofs.Foundation.ResNet34Live2,
             -- Item A: the first NON-DEGENERATE ResNet-34 whole-net backward witness
             -- (level 2) — 2-channel stem + maxpool + 3 strided downsamples + GAP +
             -- dense, every smoothness hypothesis discharged, forward X ≠ forward 0
             -- via the channel-order invariant. In the AuditAxioms headline set.
             `LeanMlir.Proofs.Foundation.ResNet34LivePC,
             -- Item A level 3: the nonzero-Jacobian SEAL for the live ResNet-34
             -- witness (fderiv ℝ liveFwd2 Y ≠ 0 ⇒ backward not the zero map). Sealed
             -- at a channel-symmetric base Y via the BN channel-difference identity
             -- (carrier vanishes ⇒ no BN-variance derivative needed). The ResNet peer
             -- of MobileNetV2JacobianSeal. In the AuditAxioms headline set.
             `LeanMlir.Proofs.Training.ResNet34LiveSeal,
             -- Item A FULL DEPTH: the real [3,4,6,3] (16-block) live ResNet-34, level-3
             -- sealed. The 13 identity blocks (zeroed body ⇒ relu(x+1)=x+1) wash out
             -- through the downsamples' BN (bn(z+c)=bn(z)), so the full net = the
             -- empty-chain witness + 2 and the seal reduces to ResNet34LiveSeal's.
             `LeanMlir.Proofs.Foundation.ResNet34LiveFull,
             -- MobileNetV2 FULL DEPTH: the real 17-block live MobileNetV2, level-3
             -- sealed. 15 identity skip blocks (zeroed body ⇒ ivId a = a+3, no relu —
             -- linear bottleneck) shift by +45; GAP + identity head pass it, so the
             -- full net = the 2-block witness + 45 and the seal reduces to
             -- MobileNetV2JacobianSeal's Qq / g_hasDerivAt. VJP composed through all 17.
             `LeanMlir.Proofs.Training.MobileNetV2JacobianSealFull,
             -- Item D (realistic dims): the live ResNet-34 whole-net backward at real
             -- ImageNet 224×224 spatial resolution (the genuine 5-halving pyramid
             -- 224→112→56→28→14→7). β-parametric downsample (β=64>√1568) + stem
             -- (β=160>√25088); every smoothness/no-tie hyp discharged at n up to 25088,
             -- forward X≠0 (level 2). Confirms no discharge secretly used a small n.
             `LeanMlir.Proofs.Foundation.ResNet34LiveRealistic,
             -- Item D level 3: the nonzero-Jacobian SEAL at 224×224. A uniform channel-0
             -- perturbation makes channel0 = channel1 + δ everywhere, so 7×7 GAP of a
             -- uniform diff = δ and maxpool(ch0)=maxpool(ch1)+δ for ALL t (max(a+δ,b+δ)=
             -- max(a,b)+δ) — no eventual-selection topology. UDiff invariant threaded like
             -- Dom2; output diff = t·Rr (4 positive istds), g'(0)=Rr 0 ≠ 0.
             `LeanMlir.Proofs.Training.ResNet34LiveRealisticSeal,
             -- Item D level 3 for MobileNetV2: the nonzero-Jacobian SEAL at 224×224. ReLU6
             -- is a BOUNDED window (0,6), so unlike ResNet's β-grows route, γ is SCALED DOWN
             -- (γ=1/128 ⇒ |γ|√n < 3 keeps bn∈(0,6) at n=2·112·112). The 1×1 weights are
             -- dimension-independent and reused. Uniform-perturbation UDiff seal: the
             -- asymmetric stem turns input t into channel-diff −t, each BN ×γ·istd, so the
             -- output diff is −t·Rr (4 positive γ·istds), g'(0)=−Rr 0 ≠ 0.
             `LeanMlir.Proofs.Training.MobileNetV2SealRealistic,
             -- Backward-graph faithfulness (den-level): fan-in bricks
             -- (residual/SE), per-op backward ops (gap/broadcast/true-batch-norm/
             -- batched conv+depthwise), the whole per-example MBConv block, and
             -- the batched-stage backward primitives.
             `LeanMlir.Proofs.Architectures.EfficientNetBackB0,
             -- MobileNetV2 backward-graph faithfulness (den-level): the batched
             -- relu6 conv/depthwise stages (selectMid kink), the SE-less inverted-
             -- residual body, and the whole-block capstone — the relu6 (_at)
             -- peer of EfficientNetBackB0.
             `LeanMlir.Proofs.Architectures.MobileNetV2BackB0,
             -- ResNet-34 backward-graph faithfulness (den-level): the batched
             -- conv-bn-relu stage (selectPos one-sided kink), the basic-block
             -- body (conv-bn ∘ conv-bn-relu), and the identity-block capstone —
             -- relu (_at) with an OUTER post-residual relu (the extra factor
             -- vs the MBConv/inverted-residual blocks).
             `LeanMlir.Proofs.Foundation.ResNet34BackB0,
             -- ConvNeXt backward-graph faithfulness (den-level): the per-example
             -- (batch-1) peer of EfficientNetBackB0. LayerNorm is per-example
             -- separable, so no batched machinery — the block-body backward graph
             -- (depthwise → LN → expand → gelu → project → layerScale) + identity-skip
             -- residual capstone, plus the LN+2×2/s2 downsample capstone. GELU is a
             -- global VJP, so everything stays in the clean global HasVJP form.
             `LeanMlir.Proofs.Architectures.ConvNeXtBackB0,
             -- ViT whole-block backward-graph faithfulness (den-level, heads = 1):
             -- the per-token Mat-VJP peer of the conv nets' *BackB0 capstones. MLP +
             -- attention sublayer backward graphs (residual fan-in + LN-back), with
             -- the MHSA backward collapsed at heads = 1 to the plain three-way dense
             -- fan-in over the proven sdpa_back_{Q,K,V} (tied to mhsa_has_vjp_mat by
             -- VJP determinism), assembled into the whole transformerBlock VJP.
             `LeanMlir.Proofs.Architectures.ViTBackB0,
             -- PoC: the mnist-linear train step proof-tied to the certified
             -- loss-descent SGD step (the renderer `MainMnistLinearVerified`
             -- trains on), incl. the param-grad/SGD "tail fold". Template for
             -- making each chapter's verified trainer faithful — see
             -- planning/verified_faithful_sweep.md.
             `LeanMlir.Proofs.Foundation.LinearFaithfulPoC,
             -- E4M3 (fp8) render-tie (planning §3b): the emitted block-scaled
             -- int-matmul graph denotes the intended dequant-first algorithm
             -- (the per-output dequant scale factors out of the fp32 accumulate),
             -- via existing den-faithful ops only — E4M3FaithfulPoC.lean.
             `LeanMlir.Proofs.Float.E4M3FaithfulPoC,
             -- bf16-mixed render-tie (planning §5, the symmetric gap): the emitted
             -- bf16-leaf/fp32-accumulate linear graph denotes the rounded-operand
             -- linear (no scale to factor — simpler than the E4M3 twin). Unlike fp8,
             -- this graph lowers on CUDA. Bf16FaithfulPoC.lean.
             `LeanMlir.Proofs.Float.Bf16FaithfulPoC,
             -- mnist-MLP peer: the whole 3-layer MLP train step folded into the
             -- verified AST (forward + backward chain + 6 weightSgd/biasSgd), each
             -- output's den proven = certified via mlp_render_*_certified.
             `LeanMlir.Proofs.Foundation.MlpFaithfulPoC,
             -- mnist-CNN peer: the conv train step folded into the verified AST via
             -- the new convWeightSgd/convBiasSgd ops (conv layers) + weightSgd/biasSgd
             -- (dense head); each of the 10 outputs' den proven = certified via the
             -- conv chain bridges + the M2 dense bridges (CnnFaithfulPoC.lean).
             `LeanMlir.Proofs.Foundation.CnnFaithfulPoC,
             -- ch5-CIFAR peer (no-BN, deeper 2-scale net): reuses the cnn conv ops +
             -- dense bridges (NO new core ops) — generic convW/convB_den cover all 4
             -- conv layers, the 3-dense head via the M2 bridges (CifarFaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.CifarFaithfulPoC,
             -- ch5-CIFAR-BN peer (per-channel BatchNorm): reuses the cnn conv ops + the
             -- cifar dense head; the new bnGammaSgd/bnBetaSgd ops carry the per-channel
             -- γ/β grads, den-certified via cifar_bn_render_{gamma,beta}_certified
             -- (CifarBnFaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.CifarBnFaithfulPoC,
             -- ch5-CIFAR-BN §1a TIE: conv+BN tied through the real forward + the BN backward chain
             -- (BN-output cots relu-masked for γ/β, conv cots via BN-back) — CifarBnTiePoC.lean.
             `LeanMlir.Proofs.Architectures.CifarBnTiePoC,
             -- deeper 8-conv cifar8 (no-BN): pure reuse — conv via CifarPoC generics,
             -- dense via the new generic denseW/denseB_den (Cifar8FaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.Cifar8FaithfulPoC,
             -- ch5-cifar8 §1a TIE: 8-conv chain tied through the real forward — cifar's chain
             -- repeated over 4 stages, all reused constructors (Cifar8TiePoC.lean).
             `LeanMlir.Proofs.Architectures.Cifar8TiePoC,
             -- ch5-cifar8-bn §1a TIE: cifar8's chain + a BN-back at every conv; all 32 conv+BN
             -- params tied (Cifar8BnTiePoC.lean).
             `LeanMlir.Proofs.Architectures.Cifar8BnTiePoC,
             -- ch6-ResNet-34 (full [3,4,6,3], 146 params): the 2 new strided-conv SGD ops
             -- (convStrided{Weight,Bias}Sgd) for the 7×7 stem + 3×3 downsample/projection
             -- convs den-certified via mnv2_render_stem_conv{W,b}_certified; the 142 other
             -- params reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics (ResNet34FaithfulPoC.lean).
             `LeanMlir.Proofs.Foundation.ResNet34FaithfulPoC,
             -- ch6-ResNet-34 §1a TIE: per-block-type tie lemmas (identity/downsample/stem) at the
             -- real forward + ResNet34ChainClose cotangents, the residual fan-in SUM constructors
             -- (idBlockCotIn/downBlockCotIn), loss-cot + dense fold (ResNet34TiePoC.lean).
             `LeanMlir.Proofs.Foundation.ResNet34TiePoC,
             -- ch7-MobileNetV2 §1 fold (depthwise half): the 4 new depthwise SGD ops
             -- (depthwise{,Strided}{Weight,Bias}Sgd) den-certified via the mnv2_render_depthwise*
             -- bridges; expand/project/BN/dense reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics
             -- (MobileNetV2FaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoC,
             -- ch7-MobileNetV2 §1 CLOSE (render): the reduced 6-block train step rendered ENTIRELY
             -- as pretty(provenGraph) — every line pretty of a verified SHlo node, the depthwise
             -- param updates via the new depthwise SGD ops; writes verified_mlir/mobilenetv2_train_step.mlir
             -- (MobileNetV2Render.lean, the peer of ResNet34Render.lean).
             `LeanMlir.Proofs.Codegen.MobileNetV2Render,
             -- ch7-MobileNetV2 FULL 17-block paper §1 fold (den): every one of the 210 params of
             -- mnv2TrainStepFaithfulVPaper denotes the certified step — ZERO new ops/lemmas, the
             -- cifar8-bn lesson at full scale. Six per-block-type capstones (stem/no-exp/stride-1/
             -- stride-2/head/dense), each delegating to the audited CifarPoC/CifarBnPoC/Cifar8PoC/
             -- Mnv2PoC/ResNet34PoC generics (MobileNetV2FaithfulPoCPaper.lean).
             `LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoCPaper,
             -- ch7-MobileNetV2 FULL 17-block paper §1a TIE: the whole 210-param train step tied
             -- through the REAL mobilenetv2ForwardPaper + the loss-driven backward chain (relu6
             -- two-kink masks, residual fan-in at every stride-1 skip). Per-block-type tie lemmas
             -- (no-exp/stride-1/stride-2/stem/head) applied across all 17 blocks via @[irreducible]
             -- FwdO/CotInAt/TiedAt wrappers (the r34 heartbeat lesson) (MobileNetV2TiePoCPaper.lean).
             `LeanMlir.Proofs.Architectures.MobileNetV2TiePoCPaper,
             -- ch8-EfficientNet-B0 full-16 (262-param) train step rendered as pretty(provenGraph)
             -- at the batched index (N=1, emit B = batch); un-fused SE for the SE param grads
             -- (EfficientNetRender.lean); writes verified_mlir/efficientnet_train_step.mlir.
             `LeanMlir.Proofs.Codegen.EfficientNetRender,
             -- ch8-EfficientNet-B0 §1 fold (den): every batched param-SGD op type denotes the
             -- certified Σ_n batched gradient — conv/strided-stem/dense W,b + BN γ/β + depthwise
             -- (the Σ_n batch-sum bridge = Finset.sum_congr of the per-example .correct)
             -- (EfficientNetFaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoC,
             -- ch8-EfficientNet-B0 §1a TIE (IN PROGRESS): pins each param cotangent to the actual
             -- loss-driven backward chain. Landed: the loss-cotangent den (batched softmaxRowF − onehot);
             -- the whole-net thread (swish/SE-gate/true-BN chain-cot constructors) is the remaining
             -- dedicated effort (EfficientNetTiePoC.lean).
             `LeanMlir.Proofs.Architectures.EfficientNetTiePoC,
             -- ch9-ConvNeXt-T §1 fold (started): the per-channel layer-scale γ gradient cert —
             -- the one genuinely-new proof obligation (Vec c via the chanIdx broadcast, vs the
             -- per-element Vec n cnx_render_lsgamma_certified); the den target of the pending
             -- layerScaleChGammaSgd core op (ConvNeXtFaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoC,
             -- ch9-ConvNeXt-T §1 RENDER: the full [3,3,9,3] train step rendered as pretty(provenGraph)
             -- (fwd + bwd-cotangent chain + param-SGD via the new ops); writes
             -- verified_mlir/convnext_train_step.mlir. 2 documented hand-written gaps (the stem 4×4/s4
             -- + downsample 2×2/s2 weight grads — no even/stride-4 weight-grad VJP yet) (ConvNeXtRender.lean).
             `LeanMlir.Proofs.Codegen.ConvNeXtRender,
             -- ch9-ConvNeXt-T §1b BATCHED: the same chain at N := B, plus the STOCHASTIC-DEPTH
             -- renders (18 per-block residual-branch masks) — the only ConvNeXt artifacts from
             -- that chain, since the drop-free batched render is tied but not swapped
             -- (ConvNeXtRenderB.lean).
             `LeanMlir.Proofs.Codegen.ConvNeXtRenderB,
             -- ch9-ConvNeXt-T §1a TIE: the whole [3,3,9,3] train step tied through the REAL forward —
             -- 18 blocks + 3 downsamples + GAP→LN→dense head + stem bias den-composed
             -- forward→loss→backward (GELU masks, identity-skip fan-in, downsample LN-back); the 4
             -- even-kernel weight grads are the documented render gap (ConvNeXtTiePoC.lean).
             `LeanMlir.Proofs.Architectures.ConvNeXtTiePoC,
             -- ch10-ViT-Tiny §1 RENDER: the full depth-12 train step rendered as pretty(provenGraph)
             -- (fwd + per-head SDPA backward chain + 200-param SGD via the 6 new ops); iree-validated
             -- (LeanMlir/Proofs/Codegen/ViTRender.lean). NO param gap — vit has the patch-weight VJP cert.
             `LeanMlir.Proofs.Codegen.ViTRender,
             -- ch10-ViT-Tiny §1b BATCHED: the same forward at N := B — the last net to make the
             -- move, and the only one where a per-EXAMPLE stochastic-depth mask is not yet
             -- expressible. Writes no artifact; `vit-fwd-b-tie` gates it (ViTRenderB.lean).
             `LeanMlir.Proofs.Codegen.ViTRenderB,
             -- ch10-ViT-Tiny §1 FOLD: each emitted param-SGD op den=certified — vecln γ/β, rowwise
             -- dense W/b, patch conv W/b, pos (one-line delegations to ViTVecLN/ViTClose certs); the
             -- head reuses Cifar8PoC.dense{W,B}_den, cls reuses denseBiasSgdB (ViTFaithfulPoC.lean).
             `LeanMlir.Proofs.Architectures.ViTFaithfulPoC,
             -- ch10-ViT-Tiny §1a TIE (per-block): every one of a vector-LN transformer block's 16 params,
             -- fed the cotangent the REAL backward chain delivers (vitCot* — two residual fan-ins + the
             -- three-way LN₁ fan-in + the SDPA backs), den=certified. Single-head representative; the
             -- multi-head/depth-12 thread is the remaining step (mnv2 reduced→full) (ViTTiePoC.lean).
             `LeanMlir.Proofs.Architectures.ViTTiePoC,
             -- Robustness certificate (planning/robustness_ladder.md): the Lipschitz-margin
             -- certified radius (Tsuzuku et al. 2018) — if the logit map is L-Lipschitz in L2
             -- and the margin is m, every ‖δ‖₂ < m/(√2·L) leaves the argmax fixed (proof, vs
             -- the PGD attack's one-attack upper bound). The cert side of cert ≤ TRUE ≤ PGD.
             `LeanMlir.Proofs.Certificates.LipschitzCert,
             -- Mathlib upstreaming drafts (planning/mathlib_upstream_drafts/): the
             -- PR1 generic-cdf lemmas (strictMono_cdf_iff ⟺ IsOpenPosMeasure,
             -- continuous_cdf_iff ⟺ NoAtoms, cdf_pos/lt_one/mem_Ioo) + PR2
             -- gaussianReal instantiations (+ symmetry, mean-shift), kept
             -- compiling on the pin while the Mathlib PRs are in flight.
             `LeanMlir.Proofs.Foundation.UpstreamDraft,
             -- The real Gaussian probit (planning/smoothing_gaussian_lemma.md, G1): the
             -- smoothing radius instantiated at the TRUE standard-normal quantile —
             -- stdNormalCDF strict-mono + symmetry, quantile MonotoneOn (0,1) + odd-about-½,
             -- capstone smoothing_certified_radius_gaussian with only the Neyman–Pearson
             -- (1/σ)-Lipschitz core (hg) left as a hypothesis.
             `LeanMlir.Proofs.Certificates.SmoothingGaussian,
             -- The Monte-Carlo tie (the smoothing chain's LAST honest gap):
             -- Hoeffding over the sample product measure (Mathlib subgaussian
             -- machinery) ⇒ with prob ≥ 1−exp(−2Nt²) the reported radius
             -- σ·Φ⁻¹(p̂−t) is genuinely certified — Cohen's CERTIFY end to end.
             `LeanMlir.Proofs.Certificates.SmoothingMC,
             -- The exact Clopper-Pearson tie (the arithmetic CERTIFY deploys):
             -- the count of successes over Measure.pi IS binomial (the lemma
             -- Mathlib lacks; piFinSuccAbove induction + Pascal), the CP
             -- lower bound covers with prob >= 1-alpha (sInf trick: no tail
             -- monotonicity needed), composed => smoothing_cp_certified.
             `LeanMlir.Proofs.Certificates.SmoothingCP,
             -- The smoothing CP SCORECARD (generated: scripts/smooth_scorecard_gen.py
             -- from the fixed-protocol driver runs, run_smooth_scorecard.sh):
             -- first-100 test images x {MNIST-MLP, MNIST-CNN, CIFAR-CNN},
             -- sigma=0.5, n=10112, alpha=1/1000 -- 279 per-image kernel tail
             -- checks (decide +kernel) + per-net aggregates. Light enough for
             -- Certs (no norm_num megaterms, pure kernel bignum arithmetic).
             `LeanMlir.Proofs.Certificates.SmoothingCPScorecard,
             -- Certified DECIMAL quantile bounds (the float-Phi^-1 gap):
             -- upper-Riemann panels of the Gaussian density with a kernel-
             -- computable rational pdf bound (32-term Taylor exp + pi_gt_d20
             -- + ceiling-rounding) => one decide-check certifies
             -- m*h <= Phi^-1(q0); demos Phi^-1(0.9) >= 1.27, and the
             -- scorecard MLP-img1 radius >= 1.27 in decimals.
             `LeanMlir.Proofs.Certificates.SmoothingPhiBounds,
             -- The DECIMAL-radius scorecard (generated:
             -- scripts/smooth_dec_scorecard_gen.py, same fixed-protocol runs
             -- and per-image q0 as SmoothingCPScorecard): the prefix scan
             -- phiScanRev kernel-evaluated ONCE over the whole h=1/1000 grid
             -- (3300 panels, ~2 min), then all 279 per-image decimal radii
             -- m/2000 <= sigma*Phi^-1(q0) are O(index) list lookups.
             `LeanMlir.Proofs.Certificates.SmoothingDecScorecard,
             -- The NET-SEMANTICS closure (the chain's last informality):
             -- argmaxNet classifier + measurability from logits, strict
             -- decision regions open, stdGaussian IsOpenPosMeasure, and the
             -- hp-interiority discharge from per-class strict-argmax
             -- witnesses => smoothing_cp_certified_net (CERTIFY for a
             -- concrete net's argmax, no abstract-classifier hypotheses).
             `LeanMlir.Proofs.Certificates.SmoothingNetSemantics,
             -- ...INSTANTIATED (generated: scripts/smoothing_net_witness_gen.py)
             -- for mlpT, the trained /128 pooled-MNIST MLP: ten in-kernel
             -- strict-argmax witnesses discharge hp; capstone
             -- smoothing_cp_certified_mlpT + deployed-scale demo.
             `LeanMlir.Proofs.Certificates.SmoothingNetWitness,
             -- Muon geometry (planning/muon_geometry.md): the optimizer as steepest descent under
             -- a norm. SGD = Euclidean (Cauchy-Schwarz), sign/Adam = L∞→L¹, Muon = operator→nuclear
             -- with the polar factor UVᵀ realizing the nuclear norm (achievability, given an SVD).
             `LeanMlir.Proofs.Foundation.MuonGeometry,
             -- Newton–Schulz convergence (planning/muon_ns_convergence.md): the Muon matmul iteration
             -- aX + b(XXᵀ)X + c(XXᵀ)²X actually COMPUTES the polar factor UVᵀ. P1 = the spectral-step
             -- lemma: a step is the scalar map φ(t)=at+bt³+ct⁵ applied per singular value (U,V carried
             -- through), so matrix convergence to UVᵀ reduces to scalar convergence φ^[k](σᵢ)→1.
             `LeanMlir.Proofs.Foundation.MuonNewtonSchulz,
             -- The robustness certificate INSTANTIATED (the 2026-07 audit's #1 gap):
             -- certified Frobenius Lipschitz constants (denseE_lipschitzL2 — ‖W‖₂ ≤ ‖W‖_F,
             -- no power-iteration estimate in the trust path), the hand-picked linear +
             -- dense→ReLU→dense demos, and the TRAINED tier: a /128-rationalized 49→8→10
             -- pooled-MNIST MLP with in-kernel margin and provably positive certified radius.
             `LeanMlir.Proofs.Certificates.LipschitzCertInstance,
             -- The trained-weight whole-net VJP witness (MLP rung): the same trained
             -- /128-rationalized net instantiates HasVJPAt at a REAL input — ReLU
             -- smoothness inherited from training (exact nonzero pre-activations),
             -- not engineered; level-3 sealed via an explicit Jacobian entry.
             `LeanMlir.Proofs.Training.TrainedMlpWitness,
             -- Certified-accuracy scorecard (post_audit_roadmap §1): the one-input
             -- certificate scaled to a dataset-level claim — over the FIXED first-100
             -- MNIST test subset at FIXED ε = 1/10 (pooled L2), 34/100 certified on a
             -- spectrally-capped (σ≤4 projected-SGD) /256 net vs 1/100 on the
             -- unconstrained net; per-image in-kernel margins, honest lower-bound
             -- aggregate. Same theorem, same ε — training decides if the cert bites.
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecard,
             -- Per-pair LipSDP certificates (the tighter-Lipschitz-constant pass):
             -- LipSDP-Neuron (Fazlyab 2019) for one hidden layer, PSD witnessed by
             -- exact rational LDLᵀ (kernel-checkable, no √, no eigensolver) — lifts
             -- the SAME scorecard (same nets, same images, same ε) from 34→69/100
             -- capped and 1→63/100 unconstrained; PGD bracket 72/69, sandwich
             -- nearly closed. Core lemmas + the two generated instance files.
             `LeanMlir.Proofs.Certificates.LipschitzCertPairSDP,
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDP,
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDPUncon,
             -- The kernel-dotZ list engine + IBP interval-soundness cores: the
             -- small, reusable halves of the full-input scorecard work. The
             -- GENERATED full-input instance files (30k+ lines of weight/image
             -- data each) live in the separate `CertsHeavy` lib below — they
             -- OOM'd/priced out the shared 4-core runners, and heavy corpus
             -- tails must not break the core workflows (certs.yml + blueprint
             -- both build `Certs`).
             `LeanMlir.Proofs.Foundation.ListDot,
             `LeanMlir.Proofs.Foundation.IntervalBound,
             -- IBP past the two-layer dense wall: a COMPOSITIONAL interval engine
             -- (`BoxSound`/`BoxSound3`/`BoxSound3V` + `.comp`, so depth is just `∘`)
             -- with conv2d / maxPool2 / dense / relu transformers proved sound, the
             -- conv uniform-box collapse, and tensor- and flat-space capstones.
             -- `ibp2_certified_at_eps` only ever covered `dense ∘ relu ∘ dense`;
             -- this is what lets a certificate reach a convolution at all. Engine
             -- only (no generated data) — the instance lives in `CertsHeavy`.
             `LeanMlir.Proofs.Foundation.IntervalBoundConv,
             -- CROWN: never concretize in the middle. Carry a LINEAR FUNCTION OF
             -- THE INPUT backward (each unstable ReLU relaxed by a linear
             -- envelope) and concretize ONCE, so the cancellation between rows of
             -- W1 survives in the composite row `A` instead of dying to IBP's
             -- per-row ‖·‖₁. Certified on the MARGIN (`certified_of_marginPos`),
             -- not on a logit box: separating two independently-bounded logits
             -- discards exactly the correlation this buys. The upper envelope
             -- takes any `s` with `u ≤ s*(u-l)`, so the slope may be ROUNDED to a
             -- /2^k grid — measured k=8 costs zero images, which is what keeps
             -- the rationals at weight scale (planning/crown_ibp.md §5.5).
             -- Engine only (no generated data).
             `LeanMlir.Proofs.Foundation.CrownBound,
             -- The binary32/fp8-E4M3 hardware models, CONSTRUCTED (post_audit_roadmap §2):
             -- rndP p = round-to-nearest on the unbounded-exponent p-bit grid, standard
             -- model |rndP p x − x| ≤ 2⁻¹⁻ᵖ|x| PROVED (rndP_err) — the former
             -- ieeeRnd/ieeeRnd_err axioms discharged, so the concrete argmax-preservation
             -- and binary32-SGD-descent capstones now live in the ordinary zero-axiom
             -- closure (this was the quarantined TrustedBridge lib, no longer needed).
             `LeanMlir.Proofs.Float.Binary32Instance,
             -- Descent at TRAINED weights (post_audit_roadmap §3): one binary32 SGD
             -- step on the trained /128 pooled-MNIST linear classifier provably
             -- decreases the real CE loss — the misclassified-witness trick makes the
             -- descent window rational-checkable with zero exp evaluations (z_lbl ≤
             -- z_pred exact ⇒ softmax_lbl ≤ 1/2 ⇒ Σ|∇| ≤ 2Σx, Σ∇² ≥ Σx²/4). Retires
             -- the W=0 degeneracy caveat of binary32_linear_sgd_descends_concrete.
             `LeanMlir.Proofs.Training.TrainedLinearDescent,
             -- Trained-weight whole-net VJP witness, CNN rung (post_audit gap #3):
             -- the Chapter-3 mnistCnnNoBn conditional whole-net VJP instantiated at
             -- TRAINED /128-rationalized weights + a REAL test digit — all five
             -- smoothness hypotheses (conv1/conv2 ReLU kinks, maxpool no-tie,
             -- dense3/dense4 kinks) discharged by exact in-kernel rationals. The
             -- no-tie condition is trained in (pool-tie margin regularizer), the
             -- h_mp analogue of the scorecard's spectral cap.
             `LeanMlir.Proofs.Training.TrainedCnnWitness,
             -- Level-3 seal for the trained CNN witness: one whole-net Jacobian
             -- entry (∂logit₇/∂pixel(0,2) = −326103939411/2³⁵ ≈ −9.49) computed in
             -- closed form — pdiv_comp peeling with exact backward-cotangent
             -- tables, the max-pool argmax routing decided per position, the conv
             -- input-VJPs via conv2d_input_grad_formula through HasVJPAt.correct.
             -- Yields backward_nontrivial / jacobian_nonzero / not_constant, the
             -- full TrainedMlpWitness theorem set at the conv rung.
             `LeanMlir.Proofs.Training.TrainedCnnSeal,
             -- The robustness certificate composed with the float bridge (2026-07
             -- audit gap #1): the scorecard's per-image Tsuzuku certificates ×
             -- the 2-layer FloatBridge budget (γ-form, B ≤ 5.96e-3 at the capped
             -- net's exact magnitudes, input quantization included) ⇒ 33/34
             -- ℝ-certified images are certified for the FLOAT-EVALUATED net,
             -- ∀ rounding models at binary32 accuracy (M.u ≤ u32).
             `LeanMlir.Proofs.Certificates.LipschitzCertFloat,
             -- Depth-linear float composition (planning/adjoint_chain.md): the
             -- telescoping chain bound (chain_adjointClose), the residual-carry
             -- combinator (chain2_adjointClose), the saturation-aware GELU
             -- Lipschitz gain (|gelu′| ≤ 3/2), and P2's balanced-tree reduction
             -- bound (tree_close / dot_tree_close — n·u → log₂n·u under the
             -- quarantined order-balanced hypothesis). Audit-leaf modules; roots
             -- so `lake build Proofs` builds their oleans for the axiom gate.
             `LeanMlir.Proofs.Codegen.AdjointChainBridge,
             `LeanMlir.Proofs.Architectures.AdjointChainResidual,
             `LeanMlir.Proofs.Certificates.GeluLipschitz,
             `LeanMlir.Proofs.Codegen.TreeReduceBridge,
             `LeanMlir.Proofs.Architectures.Cifar8ChainCert,
             -- Spec→math ties (rungs B/C/E): the one proof file that imports the
             -- trainer side (VerifiedNets), so the `denote spec.layers = <proven
             -- forward> := rfl` ties break when a spec drifts. It sat OUTSIDE
             -- every build target and silently rotted when mobilenetv2Verified
             -- was promoted 6→17 blocks (fixed 2026-07-07: mnv2 keeps the
             -- representative 6-block rung AND gains the full-paper 17-block
             -- B/C/E tie, denoteMobilenetPaper) — a root here so CI
             -- re-elaborates it.
             `LeanMlir.Proofs.Foundation.SpecVJP,
             -- The canonical-MLP surface (784→512→512→10): the generic MLP chain
             -- instantiated at the ch2 reference dims (see MlpCanonical.lean).
             `LeanMlir.Proofs.Foundation.MlpCanonical,
             -- The two batched (`N := B`) AdamW renderers. NOTHING imports
             -- either — they are leaf `#eval` writers — so the "Certs subsumes
             -- Proofs" claim above was false for exactly these two and
             -- `lake build Certs` never produced their oleans, failing any job
             -- that needs them with "object file ... does not exist". They are
             -- also the SOLE writers of the artifacts
             -- `resnet34-verified-adam{,-xla}` and `mobilenetv2-verified-adam`
             -- actually train on, so their oleans must exist wherever the
             -- corpus is built. Guarded by scripts/check_render_coverage.py.
             `LeanMlir.Proofs.Codegen.ResNet34RenderB,
             `LeanMlir.Proofs.Codegen.MobileNetV2RenderB]

/-- **`lake build CertsHeavy`** — the GENERATED full-input certificate
    instances (784-dim scorecard + per-pair LipSDP + IBP L∞: ~90k lines of
    weight/image data and per-image theorems across 8 files). Split out of
    `Certs` 2026-07-12: these are data-heavy tails (the linarith PSD goals
    carry ~230-digit LDLᵀ fractions) that OOM'd the shared 4-core runners and
    took certs.yml + blueprint.yml down with them — long-running corpus work
    gets its OWN workflow (.github/workflows/certs-heavy.yml: weekly cron +
    on-demand + pushes touching these files) so it can never break the core.
    Results (all 3-axiom, audited by tests/AuditAxiomsHeavy.lean): L2 capped
    σ≤2 92/100 @ ε=0.1 → LipSDP 93/100 = the PGD bound (sandwich closed);
    IBP pixel-L∞ 92/88/69/24 per 100 at ε = 1/2/4/8 /255 (PGD 93/93/92/88). -/
lean_lib «CertsHeavy» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Certificates.LipschitzCertScorecardFull,
             -- The per-pair LipSDP files (LipschitzCertScorecardSDPFull{,Uncon})
             -- are DISABLED here for now: their linarith PSD witnesses carry
             -- ~230-digit LDLᵀ fractions and OOM every free-tier runner config
             -- (4 attempts, incl. 1-thread + 10G swap). They remain in the repo,
             -- kernel-verified locally (93/100 @ ε=0.1 = PGD, sandwich closed);
             -- re-enable path: planning/certs_heavy_psd_memory.md (small-
             -- coefficient DD-split witnesses, or a self-hosted runner).
             -- 2026-07-25: capping the per-image exhibits (30,884 → 7,241 lines)
             -- did NOT change this — re-measured 16.0/16.6 GB at 1 thread, i.e.
             -- the peak is ONE `hS*` goal, not the file size. Still out.
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecardIBP,
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecardIBPUncon,
             -- The CROWN instance (engine: Proofs.Foundation.CrownBound, in
             -- `Certs`) on the SAME nets/subset/ε grid as the IBP tier above, so
             -- it is a new COLUMN in that table rather than a new experiment:
             -- 93/93/92/81 (capped) and 94/92/76/15 (unconstrained) per 100,
             -- against IBP's 92/88/69/24 and 87/42/2/0. Imports the IBP files to
             -- reuse their committed hpre/absr data; the only new kernel facts
             -- are one `absSumZ (combZ …)` per (image, class) — the kernel FORMS
             -- the CROWN row from the committed weight rows, so A's 784 entries
             -- are never emitted. Generated by scripts/crown_ibp_scorecard.py.
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecardCrown,
             `LeanMlir.Proofs.Certificates.LipschitzCertScorecardCrownUncon,
             -- The CONVOLUTIONAL IBP instance (engine:
             -- Proofs.Foundation.IntervalBoundConv, in `Certs`): conv → relu →
             -- max-pool → dense head at k/256 trained weights, per-image pixel-L∞
             -- certificates on the first 40 MNIST test images. Generated by
             -- scripts/ibp_conv_scorecard.py; data-heavy (one propagated box per
             -- image, ~38 GB peak elaboration), so it lives in the heavy tier with
             -- the other generated corpora.
             `LeanMlir.Proofs.Certificates.IbpConvScorecard]

/-- **`lake build ProofsMinimal`** — the suite's "hello world": the smallest
    end-to-end story (the Linear classifier), both halves — faithfulness
    (`LinearFaithfulPoC`: emitted train-step = certified math) and descent
    (`SgdDescentLinear`: that step decreases the loss). Their transitive closure is
    exactly the minimum working set (LinearTrainStep + the shared StableHLO/Tensor/
    FloatBridge/IR foundation), nothing per-net beyond Linear. Point a newcomer here
    before the full `Proofs` target. See `LeanMlir/Proofs/README.md` (Start here). -/
lean_lib «ProofsMinimal» where
  srcDir := "."
  roots := #[`LeanMlir.Proofs.Foundation.LinearFaithfulPoC, `LeanMlir.Proofs.Training.SgdDescentLinear]

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
  #["-L", "./ffi", "-liree_ffi", "-ldl", "-Wl,-rpath,./ffi",
    "-Wl,--allow-shlib-undefined"]

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

lean_exe «vit-tiny-shampoo-train» where
  root := `apps.baselines.MainVitShampooTrain
  moreLinkArgs := ireeLink

/-- `lake exe blueprint-checkdecls blueprint/lean_decls` — the split-aware
    blueprint declaration check (checkdecls minus the `CertsHeavy` lib, whose
    oleans the blueprint workflow deliberately does not build). -/
lean_exe «blueprint-checkdecls» where
  root := `tests.BlueprintCheckDecls
  supportInterpreter := true

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

lean_exe «mnist-mlp-shampoo-train» where
  root := `apps.baselines.MainMnistMlpShampooTrain
  moreLinkArgs := ireeLink

lean_exe «mnist-linear-train» where
  root := `apps.baselines.MainMnistLinearTrain
  moreLinkArgs := ireeLink

-- Trains MNIST-linear on the VERIFIED-rendered StableHLO
-- (`verified_mlir/`, = Proofs.StableHLO.linearTrainStepModuleV) through the
-- real Lean/IREE FFI. See MainMnistLinearVerified.lean.
/-- Shared body of the verified linear trainer, imported by BOTH the IREE and
    XLA executables so their config cannot drift (which would invalidate the G2
    comparison). Needs its own lib target: `apps/` modules are otherwise only
    reachable as executable roots, and an import of one would not get built. -/
lean_lib «LinearVerifiedCommon» where
  srcDir := "."
  roots := #[`apps.mnist.LinearVerifiedCommon]

lean_exe «mnist-linear-verified» where
  root := `apps.mnist.MainMnistLinearVerified
  moreLinkArgs := ireeLink

-- ═══════════════════════════════════════════════════════════════════
-- XLA/PJRT backend (planning/xla_pjrt_ladder.md)
--
-- Same Lean root, same verified_mlir/*.mlir, same §1a ties — the ONLY change is
-- which trusted lowerer consumes the emitted StableHLO. `libpjrt_ffi.so` exports
-- the identical C surface as `libiree_ffi.so`, so nothing above the shim moves.
--
-- Build the shim first (it is not built by lake — it only needs libc + dlopen):
--   gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
-- ═══════════════════════════════════════════════════════════════════

private def xlaLink : Array String :=
  #["-L", "./ffi", "-lpjrt_ffi", "-ldl", "-Wl,-rpath,./ffi",
    "-Wl,--allow-shlib-undefined"]

/-- Rung 0 of the XLA ladder: the Chapter-2 linear classifier, trained on the
    verified-rendered StableHLO through XLA instead of IREE. Compare against
    `mnist-linear-verified` (gate G2 — same params, not just same forward). -/
lean_exe «mnist-linear-verified-xla» where
  root := `apps.mnist.MainMnistLinearVerifiedXla
  moreLinkArgs := xlaLink

-- Phase-3 PGD adversarial attack on the verified linear net (planning/robustness.md):
-- the attack's input gradient is the proven dx=(softmax-onehot)·Wᵀ VJP, run via IREE.
lean_exe «mnist-linear-pgd» where
  root := `apps.mnist.MainMnistLinearPgd
  moreLinkArgs := ireeLink

-- Phase-3 PGD attack on the verified MLP (planning/robustness.md): input gradient =
-- the proven mlpInputGrad VJP; certificate = the loose product of layer spectral norms.
lean_exe «mnist-mlp-pgd» where
  root := `apps.mnist.MainMnistMlpPgd
  moreLinkArgs := ireeLink
  moreLinkArgs := ireeLink

-- Phase-3 PGD attack on the verified CNN (planning/robustness_ladder.md, the conv rung):
-- input gradient = the proven conv/maxpool input-VJP; certificate = the conv-aware product.
lean_exe «mnist-cnn-pgd» where
  root := `apps.mnist.MainMnistCnnPgd
  moreLinkArgs := ireeLink

-- Spectral-norm-constrained MLP training (planning/robustness_ladder.md, the gap-shrinking
-- lever): projected SGD onto ‖Wᵢ‖₂ ≤ c shrinks the global L = ∏‖Wᵢ‖₂, turning the vacuous
-- product certificate non-vacuous — the empirical face of lipschitz_margin_certified_radius.
lean_exe «mnist-mlp-spectral» where
  root := `apps.mnist.MainMnistMlpSpectral
  moreLinkArgs := ireeLink

-- Spectral-norm-constrained CNN training (planning/robustness_ladder.md): the conv sibling —
-- caps the dense ‖Wᵢ‖₂ and the conv tap-sum bound; a 5-layer product + loose conv-norm make
-- certifying the conv net harder than the MLP (tighter c, more clean cost).
lean_exe «mnist-cnn-spectral» where
  root := `apps.mnist.MainMnistCnnSpectral
  moreLinkArgs := ireeLink

-- Phase-3 PGD attack on the verified CIFAR-10 CNN (planning/robustness_ladder.md, the deeper
-- conv rung): input gradient = the proven 4-conv/2-pool input-VJP (genCifarPgdStep); cert = the
-- 7-layer conv-aware product. Reuses the generic attackPgdConvNet driver.
lean_exe «cifar-pgd» where
  root := `apps.cifar.MainCifarPgd
  moreLinkArgs := ireeLink

-- Spectral-norm-constrained CIFAR-10 CNN training (planning/robustness_ladder.md): the 7-layer
-- product compounds the loose conv bound harder still — tightest caps, smallest certified radii.
lean_exe «cifar-spectral» where
  root := `apps.cifar.MainCifarSpectral
  moreLinkArgs := ireeLink

-- Phase-3 PGD attack on the verified CIFAR-10 CNN + (instance) BatchNorm: genCifarBnPgdStep runs
-- the proven input-VJP through 4 instance-norm layers (the BN grad-input 3-term formula). Cert
-- N/A (instance-norm Lipschitz is data-dependent) — the attack rung only.
lean_exe «cifar-bn-pgd» where
  root := `apps.cifar.MainCifarBnPgd
  moreLinkArgs := ireeLink

-- Randomized-smoothing certificate (planning/robustness_ladder.md §3, Cohen 2019): the
-- DEPTH-INDEPENDENT cert. Forward-only Monte-Carlo over the proof-rendered fwd (no kernel, no
-- input-VJP) — sample noisy copies, Clopper-Pearson lower-bound p_A, radius = σ·Φ⁻¹(p_A). Base
-- net trained with matched Gaussian augmentation. Non-vacuous where the spectral product is hopeless.
lean_exe «mnist-mlp-smooth» where
  root := `apps.mnist.MainMnistMlpSmooth
  moreLinkArgs := ireeLink

lean_exe «mnist-cnn-smooth» where
  root := `apps.mnist.MainMnistCnnSmooth
  moreLinkArgs := ireeLink

-- The deep-net payoff: smoothing certifies a non-vacuous L2 radius on the 7-layer CIFAR CNN where
-- the conv-aware spectral product was 942K-loose (cert 0%). Same forward-only procedure, any depth.
lean_exe «cifar-smooth» where
  root := `apps.cifar.MainCifarSmooth
  moreLinkArgs := ireeLink

-- Chapter 2 (low precision): fp8 (E4M3) training on the SAME verified StableHLO —
-- fp32 master, per-column W / per-tensor x projected to the E4M3 grid, fp32 accumulate.
-- See MainMnistLinearE4M3Verified.lean + LeanMlir/E4M3Quant.lean (§3b/§3c sit on this).
lean_exe «mnist-linear-e4m3-verified» where
  root := `apps.mnist.MainMnistLinearE4M3Verified
  moreLinkArgs := ireeLink

-- Chapter 3: trains the MNIST MLP on the VERIFIED-rendered StableHLO
-- (verified_mlir/mlp_train_step.mlir = Proofs.StableHLO.mlpTrainStepText).
/-- Shared body of the verified MLP trainer — imported by BOTH the IREE and XLA
    executables so their config and He-init seed cannot drift. -/
lean_lib «MlpVerifiedCommon» where
  srcDir := "."
  roots := #[`apps.mnist.MlpVerifiedCommon]

lean_exe «mnist-mlp-verified» where
  root := `apps.mnist.MainMnistMlpVerified
  moreLinkArgs := ireeLink

/-- Rung 1 of the XLA ladder (`planning/xla_pjrt_ladder.md`): depth + multiple
    param tensors via the packed-params path, and the first rung with He init.
    Compare against `mnist-mlp-verified` for gate G2. -/
lean_exe «mnist-mlp-verified-xla» where
  root := `apps.mnist.MainMnistMlpVerifiedXla
  moreLinkArgs := xlaLink

-- Width-parametric MNIST MLP: `mnist-mlp-grid <d₁> <d₂> [epochs]` renders + trains
-- the 784→d₁→d₂→10 MLP on the faithful verified StableHLO (the size-sweep demo).
lean_exe «mnist-mlp-grid» where
  root := `apps.mnist.MainMnistMlpGrid
  moreLinkArgs := ireeLink

-- FC-width-parametric MNIST CNN: `mnist-cnn-grid <fc-width> [epochs]` holds the conv
-- stack at 32 channels and sweeps the dense head (…→d→d→10) on the faithful StableHLO.
lean_exe «mnist-cnn-grid» where
  root := `apps.mnist.MainMnistCnnGrid
  moreLinkArgs := ireeLink

-- FC-head-parametric cifar8-BN (AdamW): `cifar8-bn-grid <fc-width> [epochs]` holds the
-- 8-conv [16,16,32,32] backbone and sweeps the dense head (128→d→d→10) on the verified
-- renders (tests/TestCifar8AdamTrain.lean), trained via trainAdamSched "adam".
lean_exe «cifar8-bn-grid» where
  root := `apps.cifar.MainCifar8BnGrid
  moreLinkArgs := ireeLink

-- Chapter 3 (low precision): fp8 (E4M3) MLP training on the SAME verified StableHLO.
-- fp32 master, per-column weight quant + per-tensor input, fp32 accumulate.
-- fp8 weights+input, fp32 intermediates. See MainMnistMlpE4M3Verified.lean.
lean_exe «mnist-mlp-e4m3-verified» where
  root := `apps.mnist.MainMnistMlpE4M3Verified
  moreLinkArgs := ireeLink

-- Chapter 4: trains the MNIST CNN on the VERIFIED-rendered StableHLO
-- (verified_mlir/cnn_train_step.mlir = Proofs.StableHLO.cnnTrainStepText).
/-- Shared body of the verified CNN trainer — imported by BOTH the IREE and XLA
    executables so their config and He-init seed cannot drift. -/
lean_lib «CnnVerifiedCommon» where
  srcDir := "."
  roots := #[`apps.mnist.CnnVerifiedCommon]

lean_exe «mnist-cnn-verified» where
  root := `apps.mnist.MainMnistCnnVerified
  moreLinkArgs := ireeLink

/-- The first CONVOLUTIONAL graph on the XLA ladder — where IREE's ~1%-of-peak
    conv codegen actually bites, unlike the dense-only rungs 0-1. -/
lean_exe «mnist-cnn-verified-xla» where
  root := `apps.mnist.MainMnistCnnVerifiedXla
  moreLinkArgs := xlaLink

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
/-- Shared body of the verified CIFAR-8-BN (plain-SGD) trainer — imported by BOTH the
    IREE and XLA executables so their config and He-init seed cannot drift. This net is
    the **conv anchor** for `lake run benchmark` and `lake run benchmark-xla`, which is
    why the XLA peer exists: both must probe the same net (handoff §2j). -/
lean_lib «Cifar8BnCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8BnCommon]

lean_exe «cifar8-bn-verified» where
  root := `apps.cifar.MainCifar8BnVerified
  moreLinkArgs := ireeLink

/-- The XLA/PJRT peer of `cifar8-bn-verified` — the conv anchor for
    `lake run benchmark-xla`. Adds no rung: `VerifiedNet.train` rides
    `iree_ffi_invoke_f32`, which `ffi/pjrt_ffi.c` already implements. -/
lean_exe «cifar8-bn-verified-xla» where
  root := `apps.cifar.MainCifar8BnVerifiedXla
  moreLinkArgs := xlaLink

-- cifar8 (no BN) Adam peer: the proof-rendered fwd/bwd/param-grads with the SGD update
-- swapped for AdamW (ViTRender.emitAdamV) + packed [θ|m|v] + runtime lr/bc threading via
-- trainAdamSched. Render: tests/TestCifar8AdamTrain.lean. BN/noBN × SGD/Adam ablation.
lean_lib «Cifar8AdamCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8AdamCommon]

lean_exe «cifar8-verified-adam» where
  root := `apps.cifar.MainCifar8VerifiedAdam
  moreLinkArgs := ireeLink

/-- The no-BN control for rung 2 — same driver/hyperparameters as
    `cifar8-bn-verified-adam-xla`, differing only by BatchNorm. -/
lean_exe «cifar8-verified-adam-xla» where
  root := `apps.cifar.MainCifar8VerifiedAdamXla
  moreLinkArgs := xlaLink

-- cifar8 + per-channel BN Adam peer (38 params incl. 8× BN γ/β). Same as above with BN.
-- Render: tests/TestCifar8AdamTrain.lean.
/-- Shared body of the CIFAR-8 BN + AdamW trainer — imported by BOTH the IREE and
    XLA executables so their schedule, seed, and hyperparameters cannot drift. -/
lean_lib «Cifar8BnAdamCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8BnAdamCommon]

lean_exe «cifar8-bn-verified-adam» where
  root := `apps.cifar.MainCifar8BnVerifiedAdam
  moreLinkArgs := ireeLink

/-- Rung 2 of the XLA ladder (`planning/xla_pjrt_ladder.md`): Adam moments and
    runtime lr/bc₁/bc₂ scalars, i.e. the first RANK-0 tensor inputs. -/
lean_exe «cifar8-bn-verified-adam-xla» where
  root := `apps.cifar.MainCifar8BnVerifiedAdamXla
  moreLinkArgs := xlaLink

-- cifar8 Nesterov-momentum SGD peers (v←μv+∇, θ←θ−lr(μv+∇), μ=0.9): same proof-rendered body +
-- emitMomentum, driven by trainAdamSched variant "mom" (reuses [θ|m|v] packing + cosine+warmup lr).
-- Render: tests/TestCifar8AdamTrain.lean. Completes the optimizer ablation (SGD/momentum/Adam).
-- Shared body of the no-BN Nesterov-momentum arm of the six-way optimizer ablation: one module for the IREE and
-- XLA executables, so `epochs`, `batchSize`, the seed and the learning rate cannot drift
-- between the two backends (§2h). Lake needs it as its own lib to build it for both roots.
lean_lib «Cifar8MomCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8MomCommon]

lean_exe «cifar8-verified-momentum» where
  root := `apps.cifar.MainCifar8VerifiedMomentum
  moreLinkArgs := ireeLink

/-- XLA/PJRT peer of `cifar8-verified-momentum` — the no-BN Nesterov-momentum arm of the six-way optimizer ablation. Shares its body via
    `apps/cifar/Cifar8MomCommon.lean`, so the two backends cannot drift on epochs,
    batch size, seed or learning rate (§2h). Completes `lake run cifar-xla` to all six. -/
lean_exe «cifar8-verified-momentum-xla» where
  root := `apps.cifar.MainCifar8VerifiedMomentumXla
  moreLinkArgs := xlaLink

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

-- Shared body of the BN Nesterov-momentum arm of the six-way optimizer ablation: one module for the IREE and
-- XLA executables, so `epochs`, `batchSize`, the seed and the learning rate cannot drift
-- between the two backends (§2h). Lake needs it as its own lib to build it for both roots.
lean_lib «Cifar8BnMomCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8BnMomCommon]

lean_exe «cifar8-bn-verified-momentum» where
  root := `apps.cifar.MainCifar8BnVerifiedMomentum
  moreLinkArgs := ireeLink

/-- XLA/PJRT peer of `cifar8-bn-verified-momentum` — the BN Nesterov-momentum arm. Shares its body via
    `apps/cifar/Cifar8BnMomCommon.lean`, so the two backends cannot drift on epochs,
    batch size, seed or learning rate (§2h). Completes `lake run cifar-xla` to all six. -/
lean_exe «cifar8-bn-verified-momentum-xla» where
  root := `apps.cifar.MainCifar8BnVerifiedMomentumXla
  moreLinkArgs := xlaLink

-- cifar8 plain-SGD CONTROL on the momentum/Adam pipeline (trainAdamSched variant "sgd": same
-- per-epoch shuffle + hflip + cosine-warmup, update θ←θ−lr·∇). Makes the SGD/momentum/Adam
-- comparison differ ONLY in the optimizer. Render: tests/TestCifar8AdamTrain.lean.
-- Shared body of the no-BN plain-SGD arm of the six-way optimizer ablation: one module for the IREE and
-- XLA executables, so `epochs`, `batchSize`, the seed and the learning rate cannot drift
-- between the two backends (§2h). Lake needs it as its own lib to build it for both roots.
lean_lib «Cifar8SgdSchedCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8SgdSchedCommon]

lean_exe «cifar8-verified-sgdsched» where
  root := `apps.cifar.MainCifar8VerifiedSgdSched
  moreLinkArgs := ireeLink

/-- XLA/PJRT peer of `cifar8-verified-sgdsched` — the no-BN plain-SGD arm (SGD through the SAME shuffle/hflip/cosine pipeline, so the
    optimizer is the only free variable). Shares its body via
    `apps/cifar/Cifar8SgdSchedCommon.lean`, so the two backends cannot drift on epochs,
    batch size, seed or learning rate (§2h). Completes `lake run cifar-xla` to all six. -/
lean_exe «cifar8-verified-sgdsched-xla» where
  root := `apps.cifar.MainCifar8VerifiedSgdSchedXla
  moreLinkArgs := xlaLink

-- Shared body of the BN plain-SGD arm of the six-way optimizer ablation: one module for the IREE and
-- XLA executables, so `epochs`, `batchSize`, the seed and the learning rate cannot drift
-- between the two backends (§2h). Lake needs it as its own lib to build it for both roots.
lean_lib «Cifar8BnSgdSchedCommon» where
  srcDir := "."
  roots := #[`apps.cifar.Cifar8BnSgdSchedCommon]

lean_exe «cifar8-bn-verified-sgdsched» where
  root := `apps.cifar.MainCifar8BnVerifiedSgdSched
  moreLinkArgs := ireeLink

/-- XLA/PJRT peer of `cifar8-bn-verified-sgdsched` — the BN plain-SGD arm. Shares its body via
    `apps/cifar/Cifar8BnSgdSchedCommon.lean`, so the two backends cannot drift on epochs,
    batch size, seed or learning rate (§2h). Completes `lake run cifar-xla` to all six. -/
lean_exe «cifar8-bn-verified-sgdsched-xla» where
  root := `apps.cifar.MainCifar8BnVerifiedSgdSchedXla
  moreLinkArgs := xlaLink

-- Wide-head (MNIST-style 2×512 dense, d1=512) cifar8 optimizer ablation: each exe runs SGD /
-- momentum / AdamW in sequence on the controlled pipeline. Render: tests/TestCifar8WideTrain.lean.
lean_exe «cifar8w-ablation» where
  root := `apps.ablation.MainCifar8WideAblation
  moreLinkArgs := ireeLink

lean_exe «cifar8w-bn-ablation» where
  root := `apps.ablation.MainCifar8WideBnAblation
  moreLinkArgs := ireeLink

-- ch6 B9: real ResNet-34 ([3,4,6,3], per-channel BN, strided downsamples) trained on
-- VERIFIED-rendered StableHLO; 146 params. Train step AND eval forward both come from
-- LeanMlir/Proofs/Codegen/ResNet34Render.lean (pretty(provenGraph)); regenerate with
-- scripts/regen_verified_mlir.sh.
lean_exe «resnet34-verified» where
  root := `apps.imagenette.MainResnet34Verified
  moreLinkArgs := ireeLink

-- r34 peer of mnv2/enet-verified-adam: the proof-rendered train step (per-channel BN + strided
-- downsamples) with the SGD update swapped for AdamW (ViTRender.emitAdamV) + packed θ|m|v + runtime
-- lr/bc threading via trainAdamSched. Recipe matches the reference (lr 1e-3, wd 1e-4, cosine+warmup
-- 3, label-smoothing 0.1). Render: tests/TestResnet34Train.lean.
/-- Shared body of the verified R34 + AdamW Imagenette trainer — imported by BOTH
    the IREE and XLA executables so their schedule and seed cannot drift. -/
lean_lib «Resnet34AdamCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.Resnet34AdamCommon]

lean_exe «resnet34-verified-adam» where
  root := `apps.imagenette.MainResnet34VerifiedAdam
  moreLinkArgs := ireeLink

/-- Rung 3 of the XLA ladder (`planning/xla_pjrt_ladder.md`): full scale at 224²,
    BN running stats, and the regime the 20-40x measurements came from. -/
lean_exe «resnet34-verified-adam-xla» where
  root := `apps.imagenette.MainResnet34VerifiedAdamXla
  moreLinkArgs := xlaLink

/-- Shared body of the ResNet-34 / **full ImageNet-1k** trainer (handoff §2k). Its own `lean_lib`
    for the same reason `Resnet34AdamCommon` has one: lake needs a module for a root shared by an
    executable, and a `Common` without one silently fails to build for the second consumer. -/
lean_lib «Resnet34ImagenetCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.Resnet34ImagenetCommon]

/-- **ResNet-34 on full 1000-class ImageNet** — the scale/reference tier. Same certified renderer at
    `nClasses := 1000, B := 256`, heavy-ball + coupled L2 (the `jax/MainResnetImagenet.lean` recipe),
    fed by the generated tfds shim so both paths see identical augmented batches.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). ⚠ It used to
    say `lake exe resnet34-imagenet default --shim` — R34's, for every net, which is exactly
    how every net came to stream R34's augmentation.
    ⚠ Does NOT move the verification tier — proofs stop at Imagenette (§2k). -/
lean_exe «resnet34-imagenet-verified-xla» where
  root := `apps.imagenette.MainResnet34ImagenetXla
  moreLinkArgs := xlaLink

/-- Shared body of the ViT-Tiny / **full ImageNet-1k** trainer (handoff §2p). Its own `lean_lib`
    for the same reason `Resnet34ImagenetCommon` has one. -/
lean_lib «ViTImagenetCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.ViTImagenetCommon]

/-- **ViT-Tiny on full 1000-class ImageNet** — the ViT peer of the R34 scale tier. Same certified
    renderer at `nClasses := 1000, bs := 128`; at four replicas that is global batch 512, the
    reference's (`jax/MainVitImagenet.lean`). Fed by the generated tfds shim.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). ⚠ It used to
    say `lake exe resnet34-imagenet default --shim` — R34's, for every net, which is exactly
    how every net came to stream R34's augmentation.
    ⚠ Set `SHIM_WORKERS=2` — one producer cannot feed a 4×128 ViT step (§2p).
    ⚠ Does NOT move the verification tier, and is NOT the DeiT recipe (§2p). -/
lean_exe «vit-imagenet-verified-xla» where
  root := `apps.imagenette.MainViTImagenetXla
  moreLinkArgs := xlaLink

/-- Shared body of the ConvNeXt-T / **full ImageNet-1k** trainer (handoff §2p). -/
lean_lib «ConvNeXtImagenetCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.ConvNeXtImagenetCommon]

/-- **ConvNeXt-T on full 1000-class ImageNet**. Same certified renderer at `nClasses := 1000`;
    batch stays 32 per device (`cBS` is still private), so four replicas is global 128 and 10,009
    steps/epoch — more optimizer steps than the reference's 5,004 at batch 256.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). ⚠ It used to
    say `lake exe resnet34-imagenet default --shim` — R34's, for every net, which is exactly
    how every net came to stream R34's augmentation.
    ⚠ Does NOT move the verification tier, and is NOT the ConvNeXt paper recipe (§2p). -/
lean_exe «convnext-imagenet-verified-xla» where
  root := `apps.imagenette.MainConvNeXtImagenetXla
  moreLinkArgs := xlaLink

/-- Shared body of the EfficientNet-B0 / **full ImageNet-1k** trainer (handoff §2p). -/
lean_lib «EfficientNetImagenetCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.EfficientNetImagenetCommon]

/-- **EfficientNet-B0 on full 1000-class ImageNet**. Same certified renderer at `nClasses := 1000,
    B := 64`; four replicas is global 256, the reference's batch. The first ImageNet net here with
    BatchNorm, so it has a `_fwd_eval` peer and a running-stat region.

    Needs this net's OWN shim emitted first: `scripts/gen_shims.sh` (all five). ⚠ It used to
    say `lake exe resnet34-imagenet default --shim` — R34's, for every net, which is exactly
    how every net came to stream R34's augmentation.
    ⚠ Optimizer does NOT match the reference (RMSProp there, AdamW here) — §2p. -/
lean_exe «efficientnet-imagenet-verified-xla» where
  root := `apps.imagenette.MainEfficientNetImagenetXla
  moreLinkArgs := xlaLink

/-- Shared body of the MobileNetV2 / **full ImageNet-1k** trainer (handoff §2p). -/
lean_lib «MobileNetV2ImagenetCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.MobileNetV2ImagenetCommon]

/-- **MobileNetV2 on full 1000-class ImageNet** — the fifth scale-tier trainer. `nClasses := 1000,
    B := 64`; four replicas is global 256, the reference's batch. Batch-BN, so it has a `_fwd_eval`
    peer and a running-stat region.

    ⚠ Optimizer does NOT match the reference (RMSProp there, AdamW here) — §2p. -/
lean_exe «mobilenetv2-imagenet-verified-xla» where
  root := `apps.imagenette.MainMobileNetV2ImagenetXla
  moreLinkArgs := xlaLink

/-- Migration guard for the §2a `_fwd` move: feeds two renders of `@<slug>_fwd` (or, with
    `--eval`, `@<slug>_fwd_eval`) the same θ and x and compares logits. The two emitters differ
    textually by construction, so a numeric tie is the only meaningful check. XLA-linked — it
    compiles the module in-process, so this is seconds rather than the multi-minute 224²
    `iree-compile`.

        .lake/build/bin/fwd-tie <slug> [--eval] [<pathA> [<pathB>]]

    Replaces `resnet34-fwd-tie`, which was this harness with one net hardcoded; `fwd-tie resnet34`
    is the same check. Unlike it, this one DELETES its `.vmfb` before every compile (§4) — without
    that, a second run with a different candidate silently reuses the first candidate's binary and
    reports a perfect match, which is exactly what running a negative control looks like. -/
lean_exe «fwd-tie» where
  root := `tests.TestFwdTie
  moreLinkArgs := xlaLink

/-- §0.2 ▶2, the batched-index move: the ConvNeXt forward rendered at `N := B` must emit the
    committed `verified_mlir/convnext_fwd.mlir` BYTE FOR BYTE. No GPU — it is a string compare, so
    it belongs in every pre-commit sweep rather than behind a device.

    ⚠ Pair it with `lake env lean tests/TestBatchedEmitTie.lean`: that file pins each of the 31
    batched forms against its per-example peer individually, so it localises a failure this
    whole-net diff can only report. -/
lean_exe «convnext-fwd-b-tie» where
  root := `tests.TestConvNeXtFwdBTie

/-- **ViT's batched-index forward, byte-tied against the committed artifact** (handoff §0.2 ▶3).

    The peer of `convnext-fwd-b-tie`, and the bar is STRICTER: ConvNeXt's batched chain differs from
    its per-example one on 78 conv-VJP lines (two emitters for one VJP that were never tied to each
    other), so its train-step tie carries an allowance. ViT uses one emitter per op, so this is
    exact byte-identity with no allowance — anything else is a defect, not a known divergence.

        lake build vit-fwd-b-tie && .lake/build/bin/vit-fwd-b-tie -/
lean_exe «vit-fwd-b-tie» where
  root := `tests.TestViTFwdBTie

/-- §2m ConvNeXt step 1: can the existing ops spell a **channel** LayerNorm?

    ConvNeXt's render normalises with `.bnF` (`bnForward` over the whole `C·H·W` map, scalar γ/β)
    where the reference is `channel_layer_norm` (`H·W` statistics, each over `C`, per-channel
    affine) — 21 of its 22 sites are on the wrong axis (§2m). Route A says the fix needs **no new
    op**: channel-LN is ViT's row-LN under a transpose. This settles that on device before 21 sites
    are restructured around it, and it carries its own control — the incumbent `.bnF` chain must
    NOT match, or gate 2 is measuring something both paths satisfy.

        lake build channel-ln && HIP_VISIBLE_DEVICES=0 .lake/build/bin/channel-ln -/
lean_exe «channel-ln» where
  root := `tests.TestChannelLN
  moreLinkArgs := xlaLink

/-- §2l step 1: does the emitter spell a **1×1 strided** conv, and does it compute the right one?

    The paper's ResNet-34 option-B shortcut is a 1×1 stride-2 projection where `downFwdB` builds a
    3×3 one (§2k). This sizes that change before any of it is made: renders the four strided-conv
    ops at `k = 1` (with the committed `k = 3` alongside as the control), `iree-compile`s both, then
    drives each op on device against the **closed form** `den` implies — `flatConvStride2` is
    `decimateFlat ∘ flatConv` and `decimateIdx` reads the even positions, so at `k = 1` all four
    ops are writable in one line each. The `dx` odd-position zeros are the load-bearing check: they
    distinguish `decimate ∘ conv` from a conv that read the wrong pixel, and no norm would show it.

        lake build strided-1x1 && HIP_VISIBLE_DEVICES=0 .lake/build/bin/strided-1x1 -/
lean_exe «strided-1x1» where
  root := `tests.TestStrided1x1
  moreLinkArgs := xlaLink

/-- §2l step B: are the R34 conv biases inert? §2l argues dropping them is layout-only because
    every conv is BN-followed and BN removes the bias — this MEASURES it. One step of the committed
    AdamW render from `m = v = 0`, where `m' = (1−β₁)·g` recovers the gradient exactly (§2k), then
    reads the 36 conv-bias slots. The DENSE bias is the control: same shape, same zero init, no BN
    after it, so it must move — otherwise the reading is "the harness sees zeros".

        lake build conv-bias-zero && HIP_VISIBLE_DEVICES=0 .lake/build/bin/conv-bias-zero -/
lean_exe «conv-bias-zero» where
  root := `tests.TestConvBiasZero
  moreLinkArgs := xlaLink

/-- §2k's owed numeric gate: is `resnet34_mom_train_step` really heavy-ball with COUPLED L2?

    A cross-render known answer. AdamW's stored `m' = 0.1·g` at `m = v = 0` recovers the gradient
    exactly, so on the same (θ, x, onehot) the momentum render must satisfy `v' = g + wd·θ` and
    `θ' = θ − lr·v'`. The controls are the point: the repo's `momParamF` is NESTEROV, which at
    `v = 0` differs by exactly 1.9×, and the harness requires that prediction to MISS — a gate that
    cannot separate the two optimizers would pass the one §2k warned about.

        lake build r34-mom-tie && HIP_VISIBLE_DEVICES=0 .lake/build/bin/r34-mom-tie -/
lean_exe «r34-mom-tie» where
  root := `tests.TestMomTie
  moreLinkArgs := xlaLink

/-- **recipe_gaps v1.2's gate — the RMSProp render, numerically certified.** The `r34-mom-tie`
    construction (§2k) pointed at MobileNetV2's RMSProp tail: recover the gradient from the AdamW
    render's `m'` at `m = v = 0`, then require the RMSProp render to satisfy `s' = (1-rho)*gw^2`,
    `b' = gw/sqrt(s'+eps)` and `theta' = theta - lr*b'` where `gw = g + wd*theta`.

    Two controls it requires to FIRE: the TEXTBOOK epsilon placement `gw/(sqrt(s')+eps)`, and the
    decay dropped. The first is the whole point — TensorFlow's RMSProp puts epsilon INSIDE the
    root and that is a different optimizer.

        lake build rms-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/rms-tie -/
lean_exe «rms-tie» where
  root := `tests.TestRmsTie
  moreLinkArgs := xlaLink

/-- **Stochastic depth — the two gates that cover the op's INTERIOR** (`stochastic_depth.md` §7).
    Everything gated when the feature landed pins an ENDPOINT: `dropPath = 0` re-renders every
    artifact byte-identically, keep = 1 is bit-identical to AdamW, and `TestDropPathRamp` pins the
    keep ramp across the driver/renderer seam. Neither says what a scale strictly between those
    endpoints does, and neither can — every existing tie compares the render against a peer built
    from the SAME constants.

    **A — the known answer**: drive `dropPathB` through the same `pretty` emitter and compare
    against a host-computed `s[j]·x[j,i]`. Bit-exact is the bar, not tolerance: an f32 product is
    exact in the f64 the host multiplies in, so any difference is a different function.
    **B — the all-zero-mask control**: `s ⊙ (branch + x)` compiles, trains and descends, and no
    structural check distinguishes it from the correct `s ⊙ branch + x`. A zeroed site separates
    them — it must leave the block an IDENTITY, not annihilate the signal.

    ⚠ Run gate B under `scripts/det_shim.sh`: it compares two different HLO programs and the
    committed compile options autotune on CUDA (§2d.3 Finding 1 is ROCm-specific). The harness
    measures its own A-vs-A floor first and degrades B1 to a bound if the floor is not bit-exact.

        lake build droppath-tie
        scripts/det_shim.sh /tmp/detshim
        LD_LIBRARY_PATH=/tmp/detshim CUDA_VISIBLE_DEVICES=0 .lake/build/bin/droppath-tie
        .lake/build/bin/droppath-tie --op --break     # gate A: a 1% wrong scale is caught
        .lake/build/bin/droppath-tie --net --cand <misplaced.mlir>   # gate B goes red, rc=1 -/
lean_exe «droppath-tie» where
  root := `tests.TestDropPathTie
  moreLinkArgs := xlaLink

/-- ▶ **Classifier dropout's two gates** (`recipe_gaps.md` gap C) — the ones its endpoint checks
    structurally cannot make. Gate A: the mask multiplies PER ELEMENT, against a host-computed
    answer, with the per-EXAMPLE mask (i.e. stochastic depth on the classifier) as the control.
    Gate W: the classifier WEIGHT GRADIENT reads the dropped activation, not the pooled one — the
    ConvNeXt LayerScale-γ defect (handoff §0.10) one net over, and invisible to every ones-mask
    gate because there the two activations are the same buffer.

        lake build dropout-tie
        CUDA_VISIBLE_DEVICES=0 .lake/build/bin/dropout-tie
        .lake/build/bin/dropout-tie --op --break     # gate A is falsifiable
        .lake/build/bin/dropout-tie --net            # gate W only — NO GPU, milliseconds
        scripts/fault_dropout_wgrad.py verified_mlir/efficientnet_adamdo_train_step.mlir /tmp/f.mlir
        .lake/build/bin/dropout-tie --net /tmp/f.mlir              # goes red, rc=1 -/
lean_exe «dropout-tie» where
  root := `tests.TestDropoutTie
  moreLinkArgs := xlaLink

/-- **recipe_gaps v1.4's gate — `wdExcludeNormBias`, the timm/DeiT `no_weight_decay` render.**
    `vit_adamwx` is `vit_adam` with decoupled decay switched off for the 126 params timm excludes
    (every 1-D param plus the positional embedding). The change moves NO arity, NO type and NO
    region — only which constant feeds `%wd` at 126 of 200 sites — so every structural check
    passes on both renders and only a numeric known answer can tell them apart:

      adam:  θ' = θ − lr·( m̂/(√v̂+ε) + wd·θ )
      wx:    θ' = θ − lr·( m̂/(√v̂+ε) + wd·msk·θ )

    ▶ THE PARTITION IS THE CONTROL. It does not check that 74 params match and 126 differ — a
    count is satisfied by any 74. It recovers per parameter which bucket that param EMPIRICALLY
    falls in and requires the partition to equal `vitWdDecays`' name for name, which is what
    catches a mask that excluded the wrong 126 (silent in arity, types and the prefix audit).
    m'/v' bit-exact on all 400 moment regions is the other half: AdamW's decay is DECOUPLED, so
    reaching a moment would mean coupled L2 — a different optimizer.

        lake build wdx-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/wdx-tie -/
lean_exe «wdx-tie» where
  root := `tests.TestWdExcludeTie
  moreLinkArgs := xlaLink

/-- v1.4b: **global-norm gradient clipping**, ONE harness for ViT and ConvNeXt (`clip-tie <net>`).
    The reference's `g * min(1, CLIP/(‖g‖+1e-6))` with ‖g‖ taken across EVERY parameter.

    At `m = v = 0` the moment slot recovers the factor exactly — `m' = (1−β₁)·g` — so
    `m'_clip/m'_adam` is ONE number at all ~5.5M coordinates. **That constancy is the gate**, and
    it is the only property a per-parameter clip gets wrong: a per-parameter clip scales, never
    amplifies, and is the identity below the threshold, so it satisfies every other check here.
    `scripts/perturb_clip.py perparam` builds it and it must fire.

    ⚠ Needs the below-threshold render, which is GENERATED rather than committed (an artifact
    baking a threshold no config sets is a silent hyperparameter — handoff §2a-quater):

        lake build clip-tie
        python3 scripts/perturb_clip.py verified_mlir/vit_adamclip_train_step.mlir \
          .lake/build/clip_hi_vit.mlir hi
        CUDA_VISIBLE_DEVICES=0 .lake/build/bin/clip-tie vit -/
lean_exe «clip-tie» where
  root := `tests.TestGradClipTie
  moreLinkArgs := xlaLink

/-- `stochastic_depth.md` §5b, the gate that document left open: **the drop mask is SHARDED, not
    replicated.** The mask is per-EXAMPLE and rides in the PARAMETER blob, where the DP shim's rule
    ("x and the labels shard, everything between replicates") copied it to every replica.

    ⚠ The duplicated-batch `*-dp-check` gates are structurally blind to this — same rows on both
    replicas means sharded and replicated agree bit-exact — and `shard-check` needs the gated slot
    linear in the gradient, false for the RMSProp variant this net wants. So: duplicate the DATA,
    make only the MASK asymmetric, and **swap the halves**. A sharded mask is swap-invariant TO THE
    BIT (f32 addition is commutative, so the 2-replica mean is order-free); a replicated one is not.
    Optimizer-agnostic — it compares two runs of the same graph, never a device answer to a host one.

        lake build drop-shard-check && scripts/det_shim.sh /tmp/detshim
        CUDA_VISIBLE_DEVICES=0,1 PJRT_REPLICAS=2 LD_LIBRARY_PATH=/tmp/detshim \
          .lake/build/bin/drop-shard-check
        PJRT_DP_NO_MASK_SHARD=1 … .lake/build/bin/drop-shard-check    # the control, rc=1 -/
lean_exe «drop-shard-check» where
  root := `tests.TestDropShardCheck
  moreLinkArgs := xlaLink

/-- §2i: the cifar8 optimizer-render tie for ALL THREE variants — `cifar8-opt-tie <adam|sgd|mom>`.
    Gates the RECOVERED GRADIENT, never θ': a train step returns θ' = θ − lr·g and θ' is dominated
    by θ, the same input on both sides, so at lr 1e-3 a wholly wrong gradient still looks like a
    match (§2a-quinquies). Each variant's gradient is exactly recoverable from its own outputs —
    adam from m', sgd from θ', mom from v'. Also gates the m/v PASSTHROUGH slots bit-exactly, since
    a tail that silently dropped a moment would still yield a plausible θ'. Deletes its .vmfb before
    every compile (§4), unlike `cifar8-adam-tie`. -/
lean_exe «cifar8-opt-tie» where
  root := `tests.TestCifar8OptTie
  moreLinkArgs := xlaLink

/-- §2a-ter guard: one AdamW step through two renders of `@cifar8_adam_train_step`, same packed
    `[θ|m|v|lr|bc1|bc2]`, every returned float compared. ⚠ Compares θ' among other things and does
    NOT delete its `.vmfb`; prefer `cifar8-opt-tie`, which recovers the gradient and deletes. -/
lean_exe «cifar8-adam-tie» where
  root := `tests.TestCifar8AdamTie
  moreLinkArgs := xlaLink

/-- §2b step-5 guard: one AdamW step through two renders of `@resnet34_adam_train_step` — the
    hand-written emitter vs the batched `pretty(provenGraph)` — same packed
    `[θ|m|v|lr,bc1,bc2|bn stats]`, every returned float compared. Numeric and not textual on
    purpose: the two are the same function but not the same graph. -/
lean_exe «resnet34-adam-tie» where
  root := `tests.TestResnet34AdamTie
  moreLinkArgs := xlaLink

/-- §2a-quinquies guard: one SGD step through two renders of `@<slug>_train_step` — the `tests/`
    emitter vs `pretty(provenGraph)` — on one shared θ, comparing the recovered GRADIENT
    `(θ − θ')/lr` rather than θ' (θ' is dominated by the shared θ, so it hides a wrong gradient).
    The lr is per side because the two emitters do not always agree on it. Run BEFORE deleting a
    `tests/` emitter: afterwards the comparison no longer exists. -/
lean_exe «sgd-render-tie» where
  root := `tests.TestSgdRenderTie
  moreLinkArgs := xlaLink

/-- ViT AdamW step-3 gate: one AdamW step through two renders of `@vit_adam_train_step` — the
    hand-written emitter the driver writes at startup vs `pretty(provenGraph)` — same packed
    `[θ|m|v|lr,bc1,bc2]`, every returned float compared. ViT has no BN, so there is no forward-only
    region: the gate is the gradient AND `%loss` (the only direct read of the forward).

    **ireeLink, not xlaLink** — unlike the R34/cifar8 ties. `vit-verified-adam` is an IREE binary,
    so the ViT AdamW graph has only ever run under IREE; on XLA/PJRT it dies in the patch-embed
    weight-grad convolution with `miopenStatusUnknownError` (this box is MIOpen-conv-weak). A tie
    must run on the backend the trainer actually uses anyway. -/
lean_exe «vit-adam-tie» where
  root := `tests.TestViTAdamTie
  moreLinkArgs := ireeLink

/-- EfficientNet-B0 AdamW step-3 gate: one AdamW step through two renders of
    `@efficientnet_adam_train_step` — the hand-written emitter in `tests/TestEfficientNetTrain.lean`
    vs `pretty(provenGraph)` — same packed `[θ|m|v|lr,bc1,bc2|bn stats]`, every returned float
    compared.

    **Stronger than `vit-adam-tie`, because EfficientNet has BatchNorm.** The 98 returned batch
    statistics (μ/σ² of all 49 BN inputs) depend on the forward alone, so the `bnstat` region pins
    the whole forward chain BIT-EXACTLY and separates a forward disagreement from a backward one in
    one run. `%loss` is still gated, but as a cross-check rather than the only forward evidence.

    **ireeLink, not xlaLink**, for `vit-adam-tie`'s reason: `efficientnet-verified-adam` is an IREE
    binary, and a tie must run on the backend the trainer actually uses. -/
lean_exe «efficientnet-adam-tie» where
  root := `tests.TestEfficientNetAdamTie
  moreLinkArgs := ireeLink

/-- MobileNetV2 AdamW gate (§2f, the last net on the scorecard): one AdamW step through two renders
    of `@mobilenetv2_adam_train_step` — the hand-written emitter in `tests/TestMobilenetV2TrainPC.lean`
    against `Proofs/Codegen/MobileNetV2RenderB.lean`'s `pretty(provenGraph)` — comparing all
    returned floats per region. 52 BN layers give a `bnstat` region that pins the forward
    bit-exactly, and the gate covers SPREAD as well as magnitude (§2f-bis). IREE-linked, because
    `mobilenetv2-verified-adam` is. Deletes its `.vmfb` before every compile (§2e's false-PASS
    trap). -/
lean_exe «mobilenetv2-adam-tie» where
  root := `tests.TestMobilenetV2AdamTie
  moreLinkArgs := ireeLink

/-- ConvNeXt-T AdamW gate (§2f): one AdamW step through two renders of `@convnext_adam_train_step`
    — the hand-written emitter in `tests/TestConvNeXtTrain.lean` vs `pretty(provenGraph)` — same
    packed `[θ|m|v|lr,bc1,bc2]`, every returned float compared.

    **ViT-grade, not EfficientNet-grade**: ConvNeXt has no BatchNorm, so there is no `bnstat`
    forward-only region and `%loss` is the only direct read of the forward — which is why it is
    gated rather than reported. ireeLink, because `convnext-verified-adam` is an IREE binary. -/
lean_exe «convnext-adam-tie» where
  root := `tests.TestConvNeXtAdamTie
  moreLinkArgs := ireeLink

/-- §2d.1 gate on the bs256 re-render: feed it 8 identical copies of one bs32 batch. Batch-BN
    statistics and the mean-CE cotangent are then exactly the bs32 render's, so all 68M returned
    floats must AGREE — an exact known-answer check, not a tolerance argument. -/
lean_exe «resnet34-batch-check» where
  root := `tests.TestResnet34BatchCheck
  moreLinkArgs := xlaLink

/-- ViT DP gate: ViT has no BN, so giving both replicas the SAME batch makes `all_reduce(add)/2`
    the identity — the data-parallel step must reproduce the single-device one exactly. Needs two
    GPUs and the XLA backend. -/
lean_exe «vit-dp-check» where
  root := `tests.TestViTDpCheck
  moreLinkArgs := xlaLink

/-- Soft-target gate: the committed renders are AFFINE in their `%onehot` input, so a mixed
    target gives the mixed gradient and mixup/cutmix need **no new cotangent**. Measures
    `grad(λ·y_a + (1−λ)·y_b) == λ·grad(y_a) + (1−λ)·grad(y_b)` on the committed bytes, gating
    `m` (never θ', which is nonlinear in the gradient under AdamW), against a control that runs
    every time and a vacuity refusal. Retires §2p's claim that a `softLabelCE` render was needed. -/
lean_exe «soft-target-tie» where
  root := `tests.TestSoftTargetTie
  moreLinkArgs := xlaLink

/-- EfficientNet DP gate. Giving both replicas the SAME batch makes `all_reduce(add)/2` the
    identity, so the data-parallel step must reproduce the single-device one exactly. BatchNorm does
    not spoil this: BN normalises per replica, and both replicas' groups are the same 32 examples,
    so their statistics are identical by construction. (The §10.3b caveat that blocked this gate for
    R34 is about SPLITTING a batch — 2×32 really is not 1×64 — not duplicating one.)

    Stronger than `vit-dp-check`: EfficientNet returns 98 BN batch statistics, so it has a
    forward-only region that must come back BIT-EXACT. Needs two GPUs and the XLA backend. -/
lean_exe «efficientnet-dp-check» where
  root := `tests.TestEfficientNetDpCheck
  moreLinkArgs := xlaLink

/-- The mnv2 peer of `efficientnet-dp-check`, gated by the same EXACT duplicated-batch identity:
    both replicas get the same 32 examples, so their BN groups are identical by construction,
    `all_reduce(add)/2 = (g+g)/2 = g`, and the DP step must reproduce the single-device one.

    mnv2 returns 104 BN batch statistics (52 layers), so it has a forward-only `bnstat` region that
    must come back BIT-EXACT. Needs two GPUs and the XLA backend — collectives exist only on the
    PJRT path, which is why `mobilenetv2-verified-adam-xla` (§2h) had to come first. -/
lean_exe «mobilenetv2-dp-check» where
  root := `tests.TestMobilenetV2DpCheck
  moreLinkArgs := xlaLink

/-- The ConvNeXt peer, gated by the same EXACT duplicated-batch identity — and the one net that
    needs no BatchNorm caveat to justify it: LayerNorm reduces within an example, never across the
    batch, so nothing couples the replicas at all.

    The flip side is that ConvNeXt returns no batch statistics, so there is no `bnstat` region and
    `%loss` is the whole of the forward evidence — this harness gates it as well as the gradient,
    the same split `convnext-adam-tie` uses. It is also the first execution anywhere here of a
    RANK-0 `all_reduce` (the 44 scalar LayerNorm γ/β). Needs two GPUs and the XLA backend —
    collectives exist only on the PJRT path, which is why `convnext-verified-adam-xla` (§2h) had to
    come first. -/
lean_exe «convnext-dp-check» where
  root := `tests.TestConvNeXtDpCheck
  moreLinkArgs := xlaLink

/-- The SHARDING gate — what `convnext-dp-check` cannot see. That gate hands both replicas the same
    rows, so a shard-offset bug leaves the halves identical and it still passes bit-exact; it
    establishes "the collective averages correctly", not "the replicas saw different data".

    This one gives the replicas DIFFERENT data and checks `DP([xA|xB]) == mean(single(xA),
    single(xB))`, with a built-in control that `DP vs single(xA)` — what a broken shard would
    return — is a far larger number. Gates the first Adam moment with `m = 0` on input, because
    `m' = (1-β₁)·g` is exactly linear in the gradient while θ' and v' are not. Needs two GPUs. -/
lean_exe «convnext-shard-check» where
  root := `tests.TestConvNeXtShardCheck
  moreLinkArgs := xlaLink

/-- `shard-check <convnext|efficientnet|mobilenetv2> [<dpPath>]` — the asymmetric-batch SHARDING
    gate for every net with a DP render, generalised from `convnext-shard-check` (handoff §5's
    "still open" item). The `*-dp-check` gates hand both replicas the SAME rows, so they are
    structurally blind to a shard-offset bug; this one gives them different data and checks
    `DP([xA|xB]) == mean(single(xA), single(xB))`. Needs two GPUs and the XLA backend. -/
lean_exe «shard-check» where
  root := `tests.TestShardCheck
  moreLinkArgs := xlaLink

/-- §2e-bis step-time bench: 1 GPU (bs 32) vs 2 GPUs (global 64) on the same certified net,
    compiled in ONE process and interleaved A,B,A,B so drift hits both equally, min statistic,
    SYNTHETIC inputs so the data loader is out of it (§3's data-bound trap). Reports ms/image and
    the Amdahl-implied non-parallelisable share of a step, which is the measured argument for
    device-resident parameters. Needs two GPUs and the XLA backend. -/
lean_exe «efficientnet-dp-bench» where
  root := `tests.TestEfficientNetDpBench
  moreLinkArgs := xlaLink

/-- §2b-quater gate: the collective's SEMANTICS, checked where they can be. cifar8 has no BN, so
    2×128 + all_reduce must equal 1×256 to fp rounding. Needs two GPUs and the XLA backend. -/
lean_exe «cifar8-dp-check» where
  root := `tests.TestCifar8DpCheck
  moreLinkArgs := xlaLink

/-- §2b tail: step-time bench for the same two renders the tie compares. The batched render is
    1.68× the emitted ops (10014 vs 5971) because `pretty` has no CSE and the batched backward ops
    are self-contained recomputes; the open question is whether XLA's own CSE collapses that. Both
    are compiled in one process and their steps interleaved, so the comparison is drift-free. -/
lean_exe «resnet34-adam-bench» where
  root := `tests.TestResnet34AdamBench
  moreLinkArgs := xlaLink

-- ch7 C4: small MobileNetV2 (inverted-residual blocks: depthwise conv + relu6 +
-- per-channel BN) trained on VERIFIED-rendered StableHLO
-- (tests/TestMobilenetV2{Train,Fwd}.lean); 30 params.
lean_exe «mobilenetv2-verified» where
  root := `apps.imagenette.MainMobilenetV2Verified
  moreLinkArgs := ireeLink

-- mnv2 peer of vit-verified-adam: the proof-rendered train step with the gradients un-fused and
-- handed to the proven AdamW triple + packed θ|m|v + runtime lr/bc threading via trainAdamSched.
-- Recipe matches mobilenet-v2-train (lr 1e-3, wd 1e-4, cosine+warmup 3, label-smoothing 0.1).
-- Render: LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean (pretty(provenGraph) since 2026-07-28).
/-- Shared body of the verified mnv2 + AdamW Imagenette trainer — imported by BOTH the IREE and
    XLA executables so their schedule and seed cannot drift. -/
lean_lib «MobilenetV2AdamCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.MobilenetV2AdamCommon]

lean_exe «mobilenetv2-verified-adam» where
  root := `apps.imagenette.MainMobilenetV2VerifiedAdam
  moreLinkArgs := ireeLink

/-- The XLA/PJRT peer of `mobilenetv2-verified-adam` — same program, same certified bytes, other
    trusted lowerer. XLA measured 4.6× IREE on EfficientNet, and this net was IREE-only; it is also
    the prerequisite for ever giving mnv2 a DP path, since collectives live only on the PJRT path.
    Measured clear of ViT's MIOpen blocker before it was written (handoff §2h). -/
lean_exe «mobilenetv2-verified-adam-xla» where
  root := `apps.imagenette.MainMobilenetV2VerifiedAdamXla
  moreLinkArgs := xlaLink

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
lean_lib «EfficientNetAdamCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.EfficientNetAdamCommon]

lean_exe «efficientnet-verified-adam» where
  root := `apps.imagenette.MainEfficientNetVerifiedAdam
  moreLinkArgs := ireeLink

/-- The XLA/PJRT peer of `efficientnet-verified-adam` — same program, same certified bytes, other
    trusted lowerer. It exists for MULTI-GPU: collectives live only on the PJRT path, so the
    `adamdp` variant cannot run under IREE at all (the shim refuses the DP entry point rather than
    silently running single-device). -/
lean_exe «efficientnet-verified-adam-xla» where
  root := `apps.imagenette.MainEfficientNetVerifiedAdamXla
  moreLinkArgs := xlaLink

-- Chapter 9: ConvNeXt-T (Liu et al. 2022 — patchify stem + [3,3,9,3] depthwise-7×7
-- blocks with LN + GELU + layerScale + 3 between-stage downsamples) trained on
-- VERIFIED-rendered StableHLO (tests/TestConvNeXt{Train,Fwd}.lean); 180 params.
lean_exe «convnext-verified» where
  root := `apps.imagenette.MainConvNeXtVerified
  moreLinkArgs := ireeLink

-- Randomized-smoothing certificate on the verified ConvNeXt-T (Imagenette 224²): the deep / real-
-- resolution rung of the depth-INDEPENDENT cert (Cohen 2019). LayerNorm ⇒ per-sample fwd, so the
-- generic smoothCertify driver applies unchanged. σ via SMOOTH_SIGMA_MILLI (split across 2 GPUs).
lean_exe «convnext-smooth» where
  root := `apps.imagenette.MainConvNeXtSmooth
  moreLinkArgs := ireeLink

-- convnext peer of r34-verified-adam: the proof-rendered train step (all-smooth — LayerNorm +
-- GELU + layerScale, no BN) with the gradients un-fused and handed to the proven AdamW triple +
-- packed θ|m|v + runtime lr/bc threading via trainAdamSched. Recipe matches the reference (lr 1e-3,
-- wd 1e-4, cosine+warmup 3, label-smoothing 0.1).
-- Render: LeanMlir/Proofs/Codegen/ConvNeXtRender.lean (pretty(provenGraph) since 2026-07-28).
/-- Shared body of the verified ConvNeXt-T + AdamW Imagenette trainer — imported by BOTH the IREE
    and XLA executables so their schedule and seed cannot drift. -/
lean_lib «ConvNeXtAdamCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.ConvNeXtAdamCommon]

lean_exe «convnext-verified-adam» where
  root := `apps.imagenette.MainConvNeXtVerifiedAdam
  moreLinkArgs := ireeLink

/-- The XLA/PJRT peer of `convnext-verified-adam` — same program, same certified bytes, other
    trusted lowerer. ConvNeXt has NO `replicas` support in its renderer, so DP is a later step, but
    it is unreachable without this binary. Measured clear of ViT's MIOpen blocker before it was
    written: its 4×4/s4 patchify weight gradient runs on this box (handoff §2h). -/
lean_exe «convnext-verified-adam-xla» where
  root := `apps.imagenette.MainConvNeXtVerifiedAdamXla
  moreLinkArgs := xlaLink

lean_exe «vit-verified» where
  root := `apps.imagenette.MainViTVerified
  moreLinkArgs := ireeLink

-- Phase 3c: ViT-Tiny with the VERIFIED-rendered AdamW step (packed θ|m|v threading through the
-- generic FFI). Render: LeanMlir/Proofs/Codegen/ViTRender.lean (pretty(provenGraph) since
-- 2026-07-28 — this driver used to emit it at startup).
/-- Shared body of the verified ViT-Tiny + AdamW Imagenette trainer — imported by BOTH the IREE and
    XLA executables so their schedule and seed cannot drift. ViT's schedule is the odd one out
    (baseLR 3e-4, 5-epoch warmup), which is exactly why it lives in one place. -/
lean_lib «ViTAdamCommon» where
  srcDir := "."
  roots := #[`apps.imagenette.ViTAdamCommon]

lean_exe «vit-verified-adam» where
  root := `apps.imagenette.MainViTVerifiedAdam
  moreLinkArgs := ireeLink

/-- The XLA/PJRT peer of `vit-verified-adam`. ⛔ **It does not run on this box** — every ViT graph
    with a backward dies at EXECUTION in `miopenStatusUnknownError` (the patch-embed weight-grad
    convolution), and it compiles fine, so a green `lake build` is not evidence it works. It exists
    because the plumbing is 30 lines and the DP render is unreachable without it. Run on ares, or
    after rerouting that convolution. Handoff §2h. -/
lean_exe «vit-verified-adam-xla» where
  root := `apps.imagenette.MainViTVerifiedAdamXla
  moreLinkArgs := xlaLink

lean_exe «cifar-cnn-train» where
  root := `apps.baselines.MainCifarCnnTrain
  moreLinkArgs := ireeLink

lean_exe «autoencoder-pets-train» where
  root := `demos.MainAutoencoderPetsTrain
  moreLinkArgs := ireeLink

lean_exe «unet-pets-train» where
  root := `demos.MainUnetPetsTrain
  moreLinkArgs := ireeLink

lean_exe «unet-brats-train» where
  root := `demos.MainUnetBratsTrain
  moreLinkArgs := ireeLink

lean_exe «unet-brats-r34» where
  root := `demos.MainUnetBratsR34
  moreLinkArgs := ireeLink

lean_exe «grad-fd-probe» where
  root := `demos.MainGradFdProbe
  moreLinkArgs := ireeLink

lean_exe «pets-predict» where
  root := `demos.MainPetsPredict
  moreLinkArgs := ireeLink

lean_exe «brats-predict» where
  root := `demos.MainBratsPredict
  moreLinkArgs := ireeLink

lean_exe «gradcam» where
  root := `demos.MainGradCAM
  moreLinkArgs := ireeLink

lean_exe «bigram-shakespeare» where
  root := `demos.MainBigramShakespeare
  moreLinkArgs := ireeLink

lean_exe «flash-probe» where
  root := `demos.MainFlashProbe
  moreLinkArgs := ireeLink

lean_exe «seg-loss-probe» where
  root := `demos.MainSegLossProbe
  moreLinkArgs := ireeLink

-- DIoU box-loss forward probe (detection infra brick #1); FD-checked by
-- scripts/diou_probe_check.py against scripts/diou_grad_check.py.
lean_exe «diou-loss-probe» where
  root := `demos.MainDiouLossProbe
  moreLinkArgs := ireeLink

-- Anchor-YOLO-loss probe (brick #2, A anchors); FD-checked by
-- scripts/anchor_loss_probe_check.py.
lean_exe «anchor-loss-probe» where
  root := `demos.MainAnchorLossProbe
  moreLinkArgs := ireeLink

-- FPN-neck (top-down multi-scale merge) probe (brick #3); FD-checked by
-- scripts/fpn_neck_probe_check.py against scripts/fpn_neck_check.py's oracle.
lean_exe «fpn-neck-probe» where
  root := `demos.MainFpnNeckProbe
  moreLinkArgs := ireeLink

-- FPN multi-scale-loss probe (brick #3, bites 4+6); FD-checked by
-- scripts/fpn_loss_probe_check.py against a numpy Σ-of-per-scale-anchor-loss ref.
lean_exe «fpn-loss-probe» where
  root := `demos.MainFpnLossProbe
  moreLinkArgs := ireeLink

-- Whole-FPN-detector probe (bite 7 de-risk): neck+heads+concat+loss+DAG backward,
-- γ=0 so every grad is FD-checkable; validated by scripts/fpn_detect_probe_check.py.
lean_exe «fpn-detect-probe» where
  root := `demos.MainFpnDetectProbe
  moreLinkArgs := ireeLink

-- Emit-only: dump the r34FpnDet train-step MLIR for eyeball / iree-compile
-- --compile-to=input parse check (planning/yolo_fpn.md bite 7 wiring).
lean_exe «fpn-train-emit» where
  root := `demos.MainFpnTrainEmit
  moreLinkArgs := ireeLink

-- FPN multi-scale detector training + inference (brick #3, bite 8). VisDrone 448,
-- 3-scale (56/28/14) neck, trained via the single-target DDPM FFI path.
lean_exe «yolov1-visdrone-fpn» where
  root := `demos.MainYolov1VisdroneFpn
  moreLinkArgs := ireeLink

lean_exe «tinygpt-shakespeare» where
  root := `demos.MainTinyGptShakespeare
  moreLinkArgs := ireeLink

lean_exe «tinystories» where
  root := `demos.MainTinyStories
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

-- VisDrone single-scale detector at 448 input / 14×14 grid (train + infer).
-- The resolution rung above the 224/7×7 WS-A baseline; planning/yolo_drone.md.
lean_exe «yolov1-visdrone448» where
  root := `demos.MainYolov1VisDrone448
  moreLinkArgs := ireeLink

-- Stride-16 "finer grid" variant: 448 input / 28×28 grid (the different-head hedge).
lean_exe «yolov1-visdrone448s16» where
  root := `demos.MainYolov1VisDrone448S16
  moreLinkArgs := ireeLink

-- Anchor-based detector: 448 / 14×14 grid, A=6 anchors (brick #2, emitAnchorYoloLoss).
lean_exe «yolov1-visdrone-anchor» where
  root := `demos.MainYolov1VisDroneAnchor
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

-- Pins the image/label pairing invariant of `F32.shuffle` on a synthetic
-- dataset where label k is derivable from image k. The FFI used to swap a
-- hardcoded 4 bytes of label per record, which silently mispaired every
-- detection and segmentation batch (mAP@0.5 0.0001 vs 0.1167 after the fix).
-- Hermetic — no data files, no GPU. See planning/post_shuffle_fix.md §3.
lean_exe «test-shuffle-pairing» where
  root := `tests.TestShufflePairing
  moreLinkArgs := ireeLink

-- Checks every DatasetIO's declared `trainPixels` / `labelBytesPerRecord`
-- against what its C loader actually allocates. Skips absent datasets, so it
-- is a pre-flight check rather than a CI job — run it whenever a dataset or
-- its preprocessing script changes.
lean_exe «test-dataset-record-sizes» where
  root := `tests.TestDatasetRecordSizes
  moreLinkArgs := ireeLink

lean_exe «test-unet-forward» where
  root := `tests.TestUnetForward

lean_exe «test-yolov1-mutex» where
  root := `tests.TestYolov1Mutex
  moreLinkArgs := ireeLink

lean_exe «inspect-convnext» where
  root := `demos.MainInspectConvNeXt
  moreLinkArgs := ireeLink

lean_exe «test-resnet-residual» where
  root := `tests.TestResnetResidual

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

/-- `ffi/libpjrt_ffi.so` is **not** a lake target — it is the gcc one-liner documented above
    `xlaLink`. Build it when it is missing or older than its source, so `lake run <group>-xla`
    works from a fresh clone instead of failing at link time, and so an edited shim cannot be
    silently run stale (the `.vmfb`-cache hazard's cousin — see planning/xla_pjrt_handoff.md §4). -/
private def ensurePjrtShim : IO Bool := do
  let src : System.FilePath := "ffi/pjrt_ffi.c"
  let so  : System.FilePath := "ffi/libpjrt_ffi.so"
  let stale ← if !(← so.pathExists) then pure true else do
    let a ← src.metadata; let b ← so.metadata
    pure (b.modified.sec < a.modified.sec)
  if !stale then
    IO.println s!"  ✓ {so} up to date"
    return true
  IO.println s!"  ▸ building {so} (gcc -fPIC -O2 -shared {src} -ldl)"
  let r ← IO.Process.output
    { cmd := "gcc", args := #["-fPIC", "-O2", "-shared", src.toString, "-ldl", "-o", so.toString] }
  if r.exitCode != 0 then
    IO.eprintln s!"    ✗ shim build failed:\n{r.stderr.take 2000}"
    return false
  IO.println s!"    ✓ built {so}"
  return true

/-- The XLA path `dlopen`s a PJRT plugin at run time (`$PJRT_PLUGIN`, else the path compiled
    into the shim). Nothing links against it, so a missing plugin surfaces as a `dlopen` error
    at the first step rather than at build time — say so up front. -/
private def notePjrtPlugin : IO Unit := do
  match ← IO.getEnv "PJRT_PLUGIN" with
  | some p =>
    if ← System.FilePath.pathExists p then IO.println s!"  ✓ PJRT_PLUGIN={p}"
    else IO.println s!"  ⚠ PJRT_PLUGIN={p} does NOT exist — the first step will fail in dlopen"
  | none =>
    IO.println "  ▸ PJRT_PLUGIN unset — falling back to the path compiled into ffi/pjrt_ffi.c \
(a jax rocm/cuda plugin .so). If the run dies in dlopen, set PJRT_PLUGIN to your plugin."

/-- Build then run each named trainer in sequence (streaming) via `run.sh`.

    `xla := true` selects the PJRT peers: it builds the shim first and reports the plugin, and
    it does NOT need the venv, because those binaries compile in-process through PJRT instead of
    shelling out to `iree-compile`. One loop serves both so the two paths cannot drift. -/
private def runDemoGroup (names : List String) (xla : Bool := false) : IO UInt32 := do
  let backend ← match ← IO.getEnv "IREE_BACKEND" with
    | some b => pure b
    | none   => detectBackend
  let gpu := (← IO.getEnv "LEAN_DEMO_GPU").getD "0"
  if xla then
    IO.println "━━━ XLA/PJRT backend ━━━"
    if !(← ensurePjrtShim) then return 1
    notePjrtPlugin
  -- The IREE trainers shell out to `iree-compile`; put the project venv on PATH so
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

/-- `lake run cifar` — the ch.4 verified cifar8 variants: SGD/momentum/adam ×
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
-- The XLA/PJRT peers of the three groups above. Same nets, same certified
-- artifacts, same schedules and seeds — the ONLY difference is which trusted
-- lowerer consumes the emitted StableHLO, which is the whole point of the
-- second backend (planning/xla_pjrt_handoff.md §1).
--
-- Why you'd reach for these: XLA is **4.6× IREE** on EfficientNet — 80 epochs in
-- 1 h 35 m against 7 h 50 m (§2e-quinquies) — and multi-GPU is reachable ONLY
-- here, since collectives exist on the PJRT path and the IREE shim refuses a DP
-- entry point outright. Re-measure per net rather than assuming 4.6×; it is one
-- net's number, on a depthwise-convolution-heavy net.
--
-- ⚠ Two coverage gaps, both named rather than papered over — see each docstring.
-- ═══════════════════════════════════════════════════════════════════════

/-- `lake run mnist-xla` — the XLA peers of `lake run mnist`, and an EXACT mirror:
    all three verified MNIST demos (linear/MLP/CNN) have a `-xla` target. -/
script «mnist-xla» do
  runDemoGroup ["mnist-linear-verified-xla", "mnist-mlp-verified-xla",
                "mnist-cnn-verified-xla"] (xla := true)

/-- `lake run cifar-xla` — the XLA peers of `lake run cifar`, and an EXACT mirror since
    2026-07-30: all six of the Chapter-4 optimizer ablation (SGD / Nesterov-momentum / AdamW
    × BN / no-BN) now have `-xla` targets. Each shares one body with its IREE peer
    (`apps/cifar/Cifar8*Common.lean`), so the two backends cannot drift on epochs, batch size,
    seed or learning rate — which is what makes a cross-backend difference attributable to the
    lowerer rather than to the run. Same order as `lake run cifar`. -/
script «cifar-xla» do
  runDemoGroup ["cifar8-verified-sgdsched-xla", "cifar8-bn-verified-sgdsched-xla",
                "cifar8-verified-momentum-xla", "cifar8-bn-verified-momentum-xla",
                "cifar8-verified-adam-xla", "cifar8-bn-verified-adam-xla"] (xla := true)

/-- `lake run imagenette-xla` — the XLA peers of `lake run imagenette`, and **all five as of
    2026-07-30**.

    ViT was excluded here from 2026-07-28 to 2026-07-30 by measurement, not omission: the graph
    compiled but died at *execution* in the patch-embed weight-gradient convolution with
    `miopenStatusUnknownError` (diagnosed in
    `upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`: a fused interior-dilated pad+conv
    selects MIOpen's no-workspace `GemmFwdRest` solver, whose `MIOpenIm2d2Col.cpp` fails to build
    under HIPRTC — it uses the OpenCL builtin `get_global_id`).

    ⚠ **It now runs, with no workaround, and the failure does not reproduce.** Measured
    2026-07-30: the im2col error fired on the session's first ViT/XLA execution and never again
    across 11 runs, including the byte-identical invocation that had just failed. So this target
    is included on the strength of it working repeatedly — but treat a recurrence as possible,
    and see `probeAttnRefMsXla` for the escape hatch (`MIOPEN_DEBUG_CONV_GEMM=0`, which is a ~7%
    regression rather than a fix).

    It is gated, not merely running: the first three step losses agree with the IREE peer to
    **3e-6** from identical fresh init, it descends (39.7 → 46.7 → **49.6%** over 3 epochs), and
    `vit-dp-check` now passes **bit-exact on all 16,579,041 floats** against a sum-not-mean control
    that fires at 0.996 — so ViT is the fifth working data-parallel net.

    Per-epoch, all measured on this card: mnv2 58.0 s, ConvNeXt 84.5 s, EfficientNet 71.1 s,
    ViT **43.5 s** (marginal, `(T₃−T₁)/2`). ⚠ ViT is the one net here **without** an 80-epoch run
    on its certified bytes — the other four have one. -/
script «imagenette-xla» do
  runDemoGroup ["resnet34-verified-adam-xla", "mobilenetv2-verified-adam-xla",
                "efficientnet-verified-adam-xla", "convnext-verified-adam-xla",
                "vit-verified-adam-xla"] (xla := true)

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
-- `lake run benchmark` (IREE) and `lake run benchmark-xla` (XLA/PJRT) —
-- estimate the book's training time on YOUR gpu.
-- Probes two fast verified nets for a few epochs (the only thing that runs),
-- reads steady-state ms/epoch from the trainer's own per-epoch print, and
-- scales the reference per-chapter wall-clock by the measured hardware factor.
-- Backend auto-detects (cuda if nvidia-smi present, else rocm) — works on
-- either vendor out of the box. A dense factor (MNIST-MLP) and a conv factor
-- (CIFAR-8-BN) scale the dense- vs conv-dominated chapters independently.
--
-- ⚠ **EACH LOWERER HAS ITS OWN REFERENCE COLUMN, AND THAT IS NOT OPTIONAL.**
-- The two backends are not within noise of each other on the same card: measured
-- on the reference 7900 XTX, XLA is 2.2× IREE on the conv anchor, 4.7× on the
-- dense one and **8.6× on the attn one** (and 4.6× on EfficientNet, handoff
-- §2e-quinquies). A single blended factor would be wrong in both directions by
-- nearly 4×, which is why the split is per family AND per lowerer. So dividing an XLA
-- probe by an IREE anchor conflates *your GPU vs a 7900 XTX* with *XLA vs IREE*,
-- and reports a training estimate several times too fast with no warning. That is
-- why `BenchItem` carries `refSecXla` and why there are `probe*RefMsXla` constants:
-- a `BenchRef` bundles one lowerer's anchors so a probe can only ever be divided by
-- a reference measured on the same path. See planning/xla_pjrt_handoff.md §2j.
--
-- REFERENCE NUMBERS below are per-chapter *training* wall-clock on a single AMD
-- 7900 XTX (gfx1100, ROCm 7.2). The MNIST/CIFAR rows and all three IREE probe
-- anchors (dense/conv/attn) were MEASURED directly from these verified trainers
-- (steady-state ms/{epoch,step} × the trainer's epoch/step count); the
-- R34/MNv2/ENet/ConvNeXt IREE Imagenette rows are the verified-adam tier runs
-- (9.5h / 5.4h / 6.2h / 13.3h) and the IREE ViT row is measured here (7.8h warm —
-- the 2.3h figure elsewhere is the JAX bf16 path, not this verified trainer). The
-- IREE rows EXCLUDE the one-time IREE compile (~10–15 min/arch, CPU-bound,
-- ~hardware-independent); the XLA rows need no such carve-out, since XLA compiles
-- in-process in seconds. Re-running either benchmark on a 7900 XTX reproduces its
-- own anchors (every factor reads ~1.0×).
--
-- ⚠ Two known staleness caveats in the IREE column, both MEASURED 2026-07-30 and
-- left as-is rather than silently changed:
--   * ch3 (MNIST CNN) reads 23764 ms/epoch; re-measured on the same card, same
--     basis (real data + eval, steady state) it is **17659** — the row is ~1.35×
--     pessimistic. ch1/ch2/ch4 reproduce (535→539, 3200→3032, 8490→8782).
--   * the IREE Imagenette rows predate the 2026-07-28 codegen swaps, so they were
--     measured on RETIRED hand-written renders (handoff §0b). The XLA Imagenette
--     rows are the current certified bytes.
--   * the IREE DENSE anchor (3030) also reads high: measured 2485 / 2815 / 2819 in
--     one session on the reference card, i.e. 0.82-0.93×. Unlike the XLA anchors,
--     which are medians of 8-10 samples, the IREE ones are single historical
--     samples. So on IREE read a low dense factor as anchor noise, not as your
--     card being slow; the conv (1.01-1.02×) and attn (1.01×) anchors reproduce.
-- Correcting any of these means re-anchoring the IREE column from medians, which is
-- a deliberate separate change — it moves published per-chapter estimates.
-- ═══════════════════════════════════════════════════════════════════════

structure BenchItem where
  chapter : String
  family  : String          -- "dense" | "conv" | "attn"
  refSec  : Nat             -- IREE reference training wall-clock (s) on the 7900 XTX
  /-- XLA/PJRT reference wall-clock (s) on the same card. `none` = this chapter has no
      measured XLA reference, in which case `benchmark-xla` prints the row as `n/a` and
      leaves it out of the totals rather than borrowing the IREE number (which would be
      the §2j mismatched-baseline trap). **Every chapter is now measured**; the mechanism
      is kept because it is what makes an unmeasured row honest rather than invented, and
      ch.9 needed it until the MIOpen workaround landed on 2026-07-30. -/
  refSecXla : Option Nat
  tier    : String          -- "" | "mnist" | "cifar" | "imagenette"
  /-- ⚠⚠ **This chapter's BOTTLENECK MIX is not the one its probe measures**, so a single
      hardware factor cannot describe it and the row prints a RANGE instead.

      The `family` axis says which *ops* a chapter runs. It does not say what the chapter is
      *limited by*, and on this path most of them are limited by the parameter round trip, not by
      arithmetic: §2d.3 measured the param share of a step at **33.5%** for the conv probe
      (`cifar8-bn`, 32²) against **59.4%** for ResNet-34, **46.7%** for EfficientNet and **84.6%**
      for the MNIST CNN. A card whose transport:compute ratio differs from the reference's — e.g.
      PCIe Gen3 x8 against a 7900 XTX — therefore gets a *different* factor for each, and scaling a
      transport-bound chapter by a compute-bound probe is the §2j mismatched-baseline trap one axis
      over: same lowerer, same card, wrong bottleneck.

      ⭐ **MEASURED, 2026-08-04 on ares (RTX 4060 Ti, PCIe Gen3 x8):** the probes read conv
      **0.56×** and dense **1.33×** (idle card) — a 2.4× spread that is not noise but two different
      bottlenecks. The old single number predicted ch5 at **35m**; the real 80-epoch run came in at
      **89m** (66.7 s marginal epoch × 80, `runs/r34_pool3s2_80ep_aug04.log`), i.e. **2.5×
      optimistic**. The bracket's transport end predicts **84m** — 1.06× low.

      ⚠⚠ **SO THE BRACKET DOES NOT CONTAIN THE TRUTH, AND MUST NOT BE SOLD AS A BOUND.** R34's real
      like-for-like ratio is **5333/3780 = 1.41×**, above *both* probe factors. The cause is
      structural and known: the probes run `LEAN_MLIR_BENCH_SYNTH=1`, so they exclude the data
      loader **by design**, and §2e-ter measured per-epoch host overhead at **6.3%** of a 1-GPU
      epoch — the size of the miss. A bracket over two synthetic probes cannot reach a real run's
      loader term. Read it as *"the compute/transport estimate spans this"*, not as an interval the
      answer lies in. It takes ch5 from 2.5× wrong to 1.06× wrong; that is the whole claim.

      On the reference card both factors read ~1.0, the range collapses, and nothing about the
      published column changes.

      ⚠ Do NOT "fix" this by re-pointing these rows at the dense probe. That is one measured
      chapter, and one sample is not a measurement — the same error this repo has already paid for
      three times (the retracted conv-probe thermal story, the retracted `MIOPEN_DEBUG_CONV_GEMM`
      fix, the retracted pinned-d2h mechanism). The honest fix is a 224² conv probe with its own
      anchor measured on the reference card; until someone has that card, a bracket is the most
      the evidence supports. -/
  transportSensitive : Bool := false
  /-- **The CUDA (RTX 4060 Ti) reference wall-clock**, measured 2026-08-04, not scaled from
      anything. ch5 is today's real 80-epoch run (`runs/r34_pool3s2_80ep_aug04.log`, 66.7 s
      marginal epoch × 80); ch6-9 are eval-inclusive marginal epochs × 80; ch1-4 are 3 steady-state
      epochs × their own epoch counts. `benchmark-xla` picks this column on a CUDA backend, so on a
      4060 Ti every factor reads ~1.00 and the estimate IS the measurement. -/
  refSecCuda : Option Nat := none
  /-- **Direct mode** (`BENCH_DIRECT=1`): the `-xla` trainer that IS this chapter, and the epoch
      count it trains for. With both set, the chapter can be measured on THIS box instead of
      scaled from the reference card — which is the only way a row that is not its own probe can
      be accurate, since no hardware factor transfers across a bottleneck change. -/
  probeXla : String := ""
  epochs   : Nat := 0
  /-- Direct mode: steps/epoch, set ONLY for `trainAdamSched` nets (the five Imagenette ones).
      ⚠ Those trainers print `Epoch N/80: loss=…` with **no ms**, so `lastEpochMs` cannot read them
      and a 3-epoch probe returns nothing — which is exactly how the first version of direct mode
      failed, silently, with exit 0 on all five. They report ms/**step** through the
      `LEAN_MLIR_MAX_STEPS` PROBE line instead, which is why `runProbe` already carries a
      `stepProbe` parameter for the attn probe. A non-zero value here selects that path.
      ⭐ It is also strictly better: the step probe warms 8 and times 9..40 (seconds, not epochs),
      and §2d.3 records that it `return ()`s BEFORE any checkpoint write, so it cannot leave a
      marker. ⚠ It excludes the EVAL pass, so these rows are TRAIN-ONLY — see the footnote. -/
  stepsPerEpoch : Nat := 0

/-- The XLA MNIST/CIFAR rows were measured 2026-07-30 on the reference 7900 XTX with the
    SAME construction as the IREE ones — steady-state ms/epoch (real data + eval, last of
    3 epochs) × the trainer's own epoch count, so the two columns are directly comparable.
    **All five XLA Imagenette rows are now measured 80-epoch single-GPU runs on the current
    certified bytes** — ch5-8 from handoff §0b (`runs/<net>_xla_80ep_jul29.log`) and ch9 from
    `runs/vit_xla_80ep_jul30.log` (wall **3491 s**, epoch marker 80).

    ⚠ ch9 is also the **validation of the marginal-epoch method** the other rows lean on: before
    the run, this row held 3480 s extrapolated from a 43.5 s `(T₃−T₁)/2` measurement × 80. The real
    wall came in at 3491 s — **0.3% out**. So `scripts/marginal_epoch.sh` × epochs is trustworthy at
    this scale, which is worth knowing because it is far cheaper than an 80-epoch run.
    ch4 mirrors the IREE row's
    approximation — the BN arm's cost × 6 — so that the two columns stay comparable, even
    though the 3 no-BN arms are cheaper.

    ⚠ The XLA MNIST/CIFAR rows are each a SINGLE steady-state sample, so they inherit the
    ±6% per-run spread documented on `probeConvRefMsXla`; the conv-family ones (ch3, ch4)
    are the affected pair. Treat them as ±6%, not as exact. -/
def benchTable : List BenchItem :=
  [ { chapter := "1  MNIST linear", family := "dense", refSec := 6,     refSecXla := some 3,    tier := "mnist",
      refSecCuda := some 3, probeXla := "mnist-linear-verified-xla", epochs := 12 },      -- IREE 535ms × 12   | XLA 239ms × 12
    { chapter := "2  MNIST MLP",    family := "dense", refSec := 38,    refSecXla := some 8,    tier := "mnist",
      refSecCuda := some 11, probeXla := "mnist-mlp-verified-xla", epochs := 12 },      -- IREE 3200ms × 12  | XLA 676ms × 12
    { chapter := "3  MNIST CNN",    family := "conv",  refSec := 238,   refSecXla := some 41,   tier := "mnist",
      transportSensitive := true, refSecCuda := some 49, probeXla := "mnist-cnn-verified-xla", epochs := 10 },                                                                     -- IREE 23764ms × 10 | XLA 4103ms × 10  ⚠ 84.6% param round trip (§2d.3)
    { chapter := "4  CIFAR x6",     family := "conv",  refSec := 2038,  refSecXla := some 888,  tier := "cifar",
      refSecCuda := some 540, probeXla := "cifar8-bn-verified-xla", epochs := 240 },   -- ⚠ 40 ep × 6 ARMS, approximated as the BN arm ×6 (the 3 no-BN arms are cheaper) — the same approximation the ref column makes, kept so the two stay comparable      -- IREE 8490ms×40×6  | XLA 3698ms×40×6
    { chapter := "5  ResNet-34",    family := "conv",  refSec := 34200, refSecXla := some 3780, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 5333, probeXla := "resnet34-verified-adam-xla", epochs := 80, stepsPerEpoch := 295 },                                                                     -- IREE 9.5h  | XLA 1h03m ⚠ was 4260 (1h11m) = the RETIRED 3×3-projection net; §2l re-ran the PAPER net at 1h03m and even wrote "8 minutes faster", but this table never got it. 59.4% param round trip (§2d.3)
    { chapter := "6  MobileNetV2",  family := "conv",  refSec := 19440, refSecXla := some 5100, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 2986, probeXla := "mobilenetv2-verified-adam-xla", epochs := 80, stepsPerEpoch := 295 },                                                                     -- IREE 5.4h  | XLA 1h25m ⚠ measured on the PRE-§2m net (52 conv biases not yet dropped)
    { chapter := "7  EfficientNet", family := "conv",  refSec := 22320, refSecXla := some 5640, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 3760, probeXla := "efficientnet-verified-adam-xla", epochs := 80, stepsPerEpoch := 295 },                                                                     -- IREE 6.2h  | XLA 1h34m  46.7% param round trip (§2d.3)
    { chapter := "8  ConvNeXt",     family := "conv",  refSec := 47880, refSecXla := some 6841, tier := "imagenette",
      transportSensitive := true, refSecCuda := some 8080, probeXla := "convnext-verified-adam-xla", epochs := 80, stepsPerEpoch := 295 },                                                                     -- IREE 13.3h | XLA 1h54m01s ⚠ was 6960 (1h56m) = the retired SCALAR-LN net; §2o Part B re-ran the channel-LN net at 6841s
    { chapter := "9  ViT",          family := "attn",  refSec := 27966, refSecXla := some 3491, tier := "imagenette",
      refSecCuda := some 2560, probeXla := "vit-verified-adam-xla", epochs := 80, stepsPerEpoch := 295 } ]-- IREE 7.8h (1185ms/step × 295 × 80, warm steady-state) | XLA 0.97h = MEASURED 80-epoch wall 3491s

/-- **This chapter's reference wall-clock, for the column in play.** Three columns now, not two:
    IREE, XLA-on-ROCm (7900 XTX) and XLA-on-CUDA (4060 Ti). The vendor split exists because the
    per-chapter cross-vendor ratio spans 2.4× and no single probe factor fits it — see
    `probeDenseRefMsCuda`. -/
def BenchItem.refFor (it : BenchItem) (col : String) : Option Nat :=
  if col == "iree" then some it.refSec
  else if col == "cuda" then it.refSecCuda
  else it.refSecXla

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

/-- The XLA/PJRT anchors, measured 2026-07-30 on the same reference 7900 XTX, with the
    same synthetic-input probe and the same "last of 3 epochs" steady-state rule as the
    IREE ones above — the only difference is which `.so` the probe binary linked. Against
    the IREE anchors these read 4.66× (dense) and 2.19× (conv), which is the whole reason
    a shared reference column would be wrong (§2j).

    **The attn anchor exists as of 2026-07-30 — `vit-verified-adam-xla` runs on this box,
    and it needs no workaround.** That reverses the state recorded from 2026-07-28: the
    graph used to die at *execution* in the patch-embed weight-gradient convolution
    (a fused interior-dilated pad+conv selects MIOpen's no-workspace `GemmFwdRest` solver,
    whose `MIOpenIm2d2Col.cpp` fails to build under HIPRTC — it uses the OpenCL builtin
    `get_global_id`; see `upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`).

    ⚠ **It is BATCH-DEPENDENT, and this anchor is the bs32 number.** Measured 2026-07-30:
    * **bs32** — the fault fired on the session's FIRST ViT/XLA execution and then never again
      across 11 runs, *including the byte-identical invocation that had just failed*. So here
      `MIOPEN_DEBUG_CONV_GEMM=0` is **not needed**, and setting it costs ~7% (attn probe 136 vs
      128 ms/step median; marginal epoch 46.5 s vs 43.5 s). Why it fired once is unexplained —
      the MIOpen on-disk cache shows no writes in that window, so cache population is not it.
    * **bs64** — the fault fires **reliably** and the variable is **REQUIRED** (see
      `vit_adamdp64_train_step` in `ViTRender.lean`).

    The mechanism, from these logs plus `upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`:
    XLA fuses the interior-dilated `pad` into the patch-embed weight-gradient convolution as
    `rhs_dilation = 16` rather than materialising a 209×209 filter, and requests that conv with a
    **zero-byte workspace** (`provided ptr: 0 size: 0`). That confines MIOpen to no-workspace
    solvers; the one it lands on is `GemmFwdRest`, whose kernel source `MIOpenIm2d2Col.cpp` uses
    the **OpenCL** builtins `get_global_id`/`get_global_size` while MIOpen JIT-compiles it through
    **HIPRTC**, where those identifiers do not exist ⇒ code-object build failure ⇒
    `miopenStatusUnknownError`. A MIOpen packaging bug, not anything about the graph. The im2col
    workspace it wanted is **linear in batch** — 6,422,528 bytes at bs32, exactly **2×** that
    (12,845,056) at bs64 — which is why the batch moves the outcome. The variable removes that
    solver family from consideration, so the broken kernel is never built.

    The render is gated independently of any of this: the first three step losses agree with
    the IREE peer to **3e-6** from identical fresh init, and `vit-dp-check` passes bit-exact
    on all 16,579,041 floats against a control that fires at 0.996.

    All three are **medians over repeated runs, not single samples**, because the conv probe has
    real run-to-run spread on this card: ten runs of the same binary gave 3449 / 3473 / 3482
    / 3528 / 3565 / 3733 / 3774 / 3778 / 3792 / 3865 ms/epoch — a ±6% band with no pattern
    (an earlier reading of it as context-dependent, benchmark-run vs standalone, was refuted
    by the 11th sample). So a single sample can read 0.94× against its own anchor and look
    like a regression when nothing changed. The dense probe is stable to ±1.5% (601 / 605 /
    607 / 610 / 610 / 610 / 615 / 619). Read an on-reference factor of 0.94-1.06× as
    agreement, not as signal — and if you re-anchor, use a median of several runs, not the
    one number in front of you. -/
def probeDenseRefMsXla : Nat := 610    -- mnist-mlp-verified-xla   (vs 3030 on IREE); median of 8
def probeConvRefMsXla  : Nat := 3650   -- cifar8-bn-verified-xla   (vs 8020 on IREE); median of 10
/-- ms/STEP, `vit-verified-adam-xla`, median of 8 in the DEFAULT configuration, i.e. with no
    MIOpen override (123/125/126/127/128/129/132/137 — ±5%). Against IREE's 1173 that is
    **9.2×**, the largest cross-lowerer gap of the three families and the reason ViT cannot
    share the conv factor. (With `MIOPEN_DEBUG_CONV_GEMM=0` the median is 136 — that variable
    is a ~7% regression, not a fix; see the note above.) -/
def probeAttnRefMsXla : Nat := 128

-- ═══ THE CUDA REFERENCE COLUMN — RTX 4060 Ti, measured 2026-08-04 on an idle ares ═══
--
-- ⚠⚠ WHY A SECOND COLUMN RATHER THAN A BETTER FACTOR. The per-chapter 4060Ti/7900XTX ratio,
-- measured directly on all five Imagenette nets, spans **0.585 → 1.411 — a 2.4× range**:
--   ch5 R34 1.411 · ch6 mnv2 0.585 · ch7 enet 0.667 · ch8 ConvNeXt 1.181 · ch9 ViT 0.733
-- against probe factors of conv 0.56, dense 1.33, attn 0.74. **No single factor fits them**, so
-- cross-vendor scaling cannot be made accurate at any coefficient — the nets differ in how much of
-- a step is parameter transport (§2d.3: 33.5% for the conv probe against 59.4% for R34), and the
-- two vendors differ in exactly that ratio. Measuring each vendor is the fix; scaling is the
-- fallback for a box with no datasets, not the answer.
def probeDenseRefMsCuda : Nat := 814    -- mnist-mlp-verified-xla,  idle, 3 real epochs
def probeConvRefMsCuda  : Nat := 2049   -- cifar8-bn-verified-xla,  idle, 3 real epochs
def probeAttnRefMsCuda  : Nat := 95     -- vit-verified-adam-xla,   idle, MAX_STEPS=100

/-- One lowerer's complete probe configuration: which binaries to probe and which anchors
    to divide by. Bundling them is the point — `yourSecOf` takes a `BenchRef`, so a probe
    measured on one path cannot be divided by the other path's anchor, which is the §2j
    trap made unrepresentable rather than merely documented. -/
structure BenchRef where
  /-- Lowerer name for the table header — the label whose absence was §2j's complaint. -/
  lowerer    : String
  xla        : Bool
  /-- Which reference column to read: "iree" | "rocm" | "cuda". -/
  col        : String := "rocm"
  /-- The card that column was measured on, for the table header. -/
  card       : String := "7900 XTX"
  denseProbe : String
  convProbe  : String
  /-- `""` = this lowerer has no runnable attn probe. -/
  attnProbe  : String
  denseRefMs : Nat
  convRefMs  : Nat
  attnRefMs  : Nat

def ireeRef : BenchRef :=
  { lowerer := "IREE", xla := false, col := "iree", card := "7900 XTX"
    denseProbe := "mnist-mlp-verified", convProbe := "cifar8-bn-verified"
    attnProbe := "vit-verified-adam"
    denseRefMs := probeDenseRefMs, convRefMs := probeConvRefMs, attnRefMs := probeAttnRefMs }

/-- XLA on ROCm — scaled from the 7900 XTX column. -/
def xlaRefRocm : BenchRef :=
  { lowerer := "XLA/PJRT", xla := true, col := "rocm", card := "7900 XTX"
    denseProbe := "mnist-mlp-verified-xla", convProbe := "cifar8-bn-verified-xla"
    attnProbe := "vit-verified-adam-xla"
    denseRefMs := probeDenseRefMsXla, convRefMs := probeConvRefMsXla
    attnRefMs := probeAttnRefMsXla }

/-- XLA on CUDA — the 4060 Ti column, measured rather than scaled. On that card every factor reads
    ~1.00, so the estimate is the measurement; on another NVIDIA card it scales from a same-vendor
    baseline, which the 2.4× cross-vendor spread (see `probeDenseRefMsCuda`) says is the best a
    scaled number can do. -/
def xlaRefCuda : BenchRef :=
  { lowerer := "XLA/PJRT", xla := true, col := "cuda", card := "RTX 4060 Ti"
    denseProbe := "mnist-mlp-verified-xla", convProbe := "cifar8-bn-verified-xla"
    attnProbe := "vit-verified-adam-xla"
    denseRefMs := probeDenseRefMsCuda, convRefMs := probeConvRefMsCuda
    attnRefMs := probeAttnRefMsCuda }

/-- Scale a chapter's reference seconds by the measured per-family factor, against `ref`'s
    OWN anchors. `aMs` is the attn ms/step probe (0 when there is no attn probe or it
    failed → attn falls back to the conv factor). `none` when this chapter has no reference
    on `ref`'s lowerer.

    **The conv proxy for a transformer measured ~3.5× LOW**, which is why the attn family
    exists at all (`7e0e6a1`): on a 4060 Ti the three factors came out dense 4.82× / conv
    3.54× / **attn 11.98×**, so scaling ViT as conv estimated 7.9 h against a measured ~28 h.
    That is the 7900 XTX → 4060 Ti (ROCm → CUDA) divergence specifically, not a constant —
    on the reference card the proxy is harmless because every factor reads ~1.0, which is
    exactly why it went unnoticed. An independent instance of the same principle, measured
    2026-07-30 on one card across lowerers: XLA-vs-IREE is **2.24× for conv but 9.2× for
    attn**. Attention and convolution do not track each other, whether you change the GPU or
    the lowerer.

    ⚠ **The 3.5×-low proxy figure is an IREE fact, not a 4060 Ti fact.** Measured 2026-08-01 on
    ares (6× 4060 Ti, CUDA 12.9, jax 0.10.2 CUDA PJRT plugin), the SAME card on the XLA path
    reads dense **1.29×** / conv **0.54×** / attn **0.70×** — i.e. it BEATS the reference 7900
    XTX on both conv and attn, and the conv proxy for attn would have been off by only 1.3×
    rather than 3.5×. Against the IREE column's 4.82/3.54/11.98 for this same card that is a
    3.7× / 6.6× / **17×** improvement, which is a statement about IREE's CUDA backend rather
    than about the GPU. Consequence for anyone re-anchoring: the attn family is still worth
    keeping (it costs one probe and it is what makes ViT honest on IREE), but on XLA/CUDA a
    `*proxy` row is a mild approximation, not the 3.5× trap it is on IREE. -/
def yourSecOf (ref : BenchRef) (it : BenchItem) (dMs cMs aMs : Nat) : Option Nat :=
  (it.refFor ref.col).map fun refSec =>
    if it.family == "dense" then refSec * dMs / ref.denseRefMs
    else if it.family == "attn" then
      if aMs == 0 then refSec * cMs / ref.convRefMs            -- fallback: conv proxy
      else refSec * aMs / ref.attnRefMs
    else refSec * cMs / ref.convRefMs

/-- **The estimate as a BRACKET**, `(lo, hi)`, for the reason on `BenchItem.transportSensitive`.

    A chapter whose bottleneck mix its probe does not share gets both candidate factors — its own
    family's (compute-leaning) and the dense probe's (transport-leaning) — and the row prints the
    interval between them. Everything else returns a degenerate `(x, x)` and prints exactly as
    before.

    ⭐ **On the reference card the two factors both read ~1.0, so every bracket collapses and the
    published column is unchanged.** The bracket only opens on a card whose transport:compute ratio
    differs from the reference's — which is precisely when a single factor stops being meaningful.
    Measured on ares 2026-08-04: ch5's bracket is 40m–91m and the real run landed at **89m**. -/
def yourSecRange (ref : BenchRef) (it : BenchItem) (dMs cMs aMs : Nat) : Option (Nat × Nat) :=
  (yourSecOf ref it dMs cMs aMs).map fun base =>
    if it.transportSensitive then
      let transport := (it.refFor ref.col).getD 0 * dMs / ref.denseRefMs
      (Nat.min base transport, Nat.max base transport)
    else (base, base)


/-- Human duration from whole seconds: `45s` / `12m` / `9.5h`. -/
def fmtDur (sec : Nat) : String :=
  if sec < 90 then s!"{sec}s"
  else if sec < 5400 then s!"{(sec + 30) / 60}m"
  else let t := (sec * 10 + 1800) / 3600; s!"{t / 10}.{t % 10}h"

/-- Render a bracket: a single duration when it is degenerate, `lo-hi` when it is not. -/
def fmtRange (lo hi : Nat) : String :=
  if lo == hi then fmtDur lo else s!"{fmtDur lo}-{fmtDur hi}"


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

/-- **Direct mode's probe: measure the chapter's OWN trainer on THIS box, 3 real epochs.**

    `runProbe` answers *"how does my card compare to a 7900 XTX"* and then multiplies a reference.
    That is exact for a chapter which IS its own probe (ch1/2 are the dense probe, ch4 the conv one,
    ch9 the attn one) and an extrapolation for every other — which is the whole reason ch3 and ch5-8
    print a bracket. This answers the other question directly: run the chapter's real trainer and
    multiply by its own epoch count. No reference card, no hardware factor, no bottleneck
    assumption. §2j validated the method at **0.3%** (ch9's wall was extrapolated at 3480 s from a
    marginal-epoch measurement; the real 80-epoch run landed at 3491 s).

    ⚠⚠ **REAL data, not `LEAN_MLIR_BENCH_SYNTH`** — deliberately. The synthetic path exists to take
    the loader out of a *comparison*; here the loader is part of the answer, and §2e-ter measured it
    at ~6.3% of a 1-GPU epoch. Excluding it is most of why even the bracket's transport end came in
    6% low against the measured ResNet-34 run.

    ⚠⚠ **AND IT MUST NOT TOUCH A CHECKPOINT**, in either direction. `trainAdamSched` checkpoints per
    epoch, so a naive 3-epoch probe would (a) RESUME an existing checkpoint — measuring a warm
    restart, or nothing at all if the marker is at the epoch budget (§4's silent no-op) — and
    (b) leave its own marker at 3 for the next real run to resume from. The caller stashes and
    restores; this function only runs. -/
def runDirectProbe (it : BenchItem) (backend gpu : String)
    (runEnv : Array (String × Option String)) : IO (Option Nat) := do
  IO.println s!"\n  ▸ measuring {it.chapter} directly — {it.probeXla}, 3 real epochs…"
  let bp ← IO.Process.spawn { cmd := "lake", args := #["build", it.probeXla] }
  if (← bp.wait) != 0 then
    IO.eprintln s!"    build failed: {it.probeXla}"
    return none
  let vis := if backend == "cuda" then "CUDA_VISIBLE_DEVICES" else "HIP_VISIBLE_DEVICES"
  let stepMode := it.stepsPerEpoch > 0
  -- ⚠ MAX_STEPS warms 8 and times 9..n, so anything ≤ 8 fires NOTHING and caps nothing (§0.12).
  let capEnv := if stepMode then #[("LEAN_MLIR_MAX_STEPS", some "40")]
                            else #[("LEAN_MLIR_MAX_EPOCHS", some "3")]
  let env := runEnv ++ capEnv ++ #[("IREE_BACKEND", some backend), (vis, some gpu)]
  let o ← IO.Process.output { cmd := s!".lake/build/bin/{it.probeXla}", args := #["data"], env := env }
  match (if stepMode then probeMsStep o.stdout else lastEpochMs o.stdout) with
  | none =>
      IO.eprintln s!"    no timing for {it.probeXla} (dataset present? exit {o.exitCode})"
      pure none
  | some ms =>
      if stepMode then
        let total := ms * it.stepsPerEpoch * it.epochs / 1000
        IO.println s!"    {ms} ms/step × {it.stepsPerEpoch} steps × {it.epochs} ep = {fmtDur total}   (measured here; TRAIN-ONLY)"
        pure (some total)
      else
        let total := ms * it.epochs / 1000
        IO.println s!"    {ms} ms/epoch × {it.epochs} ep = {fmtDur total}   (measured here, incl. eval)"
        pure (some total)

/-- Build + run one probe net and return its steady-state timing. With `stepProbe`
    set (`attn` anchor) it caps at N steps and reads ms/step; otherwise it runs 3
    epochs and reads ms/epoch. -/
def runProbe (bin family : String) (refMs : Nat) (card backend gpu : String)
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
      IO.println s!"    {ms} {unit}   [ref {refMs}]   → {fmtFactor ms refMs}× the {card}"
      pure (some ms)

/-- The shared body of `lake run benchmark` and `lake run benchmark-xla`. One printer, two
    `BenchRef`s, so the two commands cannot drift on the probe recipe, the steady-state rule
    or the table layout — and, more to the point, so neither can end up dividing by the other
    lowerer's anchors (handoff §2j).

    Rows with no reference on this lowerer print `n/a` and are excluded from the totals and
    from the tier subtotals; the footer says how many chapters were covered, so a short total
    can never read as a whole-Part-1 number. -/
def runBenchmark (ref : BenchRef) : IO UInt32 := do
  let backend ← match ← IO.getEnv "IREE_BACKEND" with
    | some b => pure b
    | none   => detectBackend
  let gpu := (← IO.getEnv "LEAN_DEMO_GPU").getD "0"
  -- The XLA binaries compile in-process through PJRT, so unlike the IREE ones they need
  -- neither the venv on PATH nor `iree-compile` — but they DO need the shim built and a
  -- resolvable plugin, the same two guards `runDemoGroup (xla := true)` applies.
  let runEnv ← if ref.xla then do
      IO.println "━━━ XLA/PJRT backend ━━━"
      if !(← ensurePjrtShim) then return 1
      notePjrtPlugin
      pure #[]
    else do
      let venvBin := (← IO.currentDir) / ".venv" / "bin"
      if ← System.FilePath.pathExists (venvBin / "iree-compile") then
        pure #[("PATH", some s!"{venvBin}:{(← IO.getEnv "PATH").getD ""}")]
      else pure #[]
  let cmdName := if ref.xla then "benchmark-xla" else "benchmark"
  -- ▶ DIRECT MODE: measure every chapter's own trainer here instead of scaling a foreign
  --   reference. See `runDirectProbe`. XLA only — the IREE path has no in-process compile, so a
  --   3-epoch probe would pay ~10-15 min of iree-compile per net.
  let direct := ref.xla && ((← IO.getEnv "BENCH_DIRECT").getD "" != "")
  IO.println s!"━━━ lake run {cmdName} ━━━ verified-NN training throughput on your GPU"
  IO.println s!"  lowerer: {ref.lowerer}   backend: {backend}   gpu: {gpu}   (synthetic-input probes — no dataset needed)"
  -- Pre-flight: a busy GPU inflates every probe. Warn if the card isn't idle.
  match ← gpuBusyPct backend with
  | some u => IO.println (if u > 20 then
      s!"  ⚠ GPU is {u}% busy — close other GPU jobs first or the estimate will inflate."
      else s!"  GPU idle ({u}%).")
  | none   => pure ()
  -- Synthetic input (LEAN_MLIR_BENCH_SYNTH, set in runProbe): one constant batch reused
  -- at the dataset's real step count, so no MNIST/CIFAR/Imagenette download is required.
  if direct then
    -- ⚠⚠ A 3-epoch probe on a checkpointing trainer is DESTRUCTIVE IN BOTH DIRECTIONS (§4): it
    --   resumes whatever marker is on disk — measuring a warm restart, or NOTHING if the marker is
    --   at the epoch budget, which exits `done` with rc=0 and no timing — and it leaves its own
    --   marker at 3 for the next real run to resume from. So: refuse if a trainer is live, stash
    --   every XLA checkpoint, measure, restore. Stash rather than delete, because these are
    --   someone's training state.
    let ps ← IO.Process.output { cmd := "bash", args := #["-c", "ps -eo comm | grep -c verified || true"] }
    if (ps.stdout.trim.toNat?.getD 0) > 0 then
      IO.eprintln "  ⛔ a *-verified* trainer is RUNNING. Direct mode stashes checkpoints and would \
corrupt it (and contend for the GPU). Stop it first."
      return 1
    let stash := ".lake/build/benchdirect-stash"
    IO.println s!"\n  ▶ DIRECT MODE — measuring each chapter's own trainer on THIS box (3 real epochs each)."
    IO.println s!"    checkpoints stashed to {stash}/ and restored afterwards (§4: a probe must not resume or leave one)."
    _ ← IO.Process.output { cmd := "bash", args := #["-c",
      s!"mkdir -p {stash} && mv .lake/build/*_ckpt_xla.bin* {stash}/ 2>/dev/null; true"] }
    let mut rows : Array (BenchItem × Option Nat) := #[]
    for it in benchTable do
      if it.probeXla.isEmpty || it.epochs == 0 then
        rows := rows.push (it, none)
      else
        rows := rows.push (it, ← runDirectProbe it backend gpu runEnv)
    -- the probes' own markers go; the stash comes back
    _ ← IO.Process.output { cmd := "bash", args := #["-c",
      s!"rm -f .lake/build/*_ckpt_xla.bin*; mv {stash}/* .lake/build/ 2>/dev/null; rmdir {stash} 2>/dev/null; true"] }
    IO.println s!"\n  MEASURED training time on THIS box ({ref.lowerer}, real data, current certified bytes):\n"
    let rule := "  " ++ String.ofList (List.replicate 47 '-')
    IO.println s!"  {padR "Chapter" 18}{padR "family" 8}{padR s!"ref({ref.card})" 18}measured here"
    IO.println rule
    let mut tot := 0
    for (it, m) in rows do
      let refCol := match it.refFor ref.col with | some r => fmtDur r | none => "n/a"
      match m with
      | some sec => tot := tot + sec
                    IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR refCol 18}{fmtDur sec}"
      | none     => IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR refCol 18}— (probe failed)"
    IO.println rule
    IO.println s!"  {padR "Full Part-1 training" 30}{padR "" 18}{fmtDur tot}"
    IO.println "\n  * every number is 3 steady-state epochs of that chapter's OWN trainer × its own"
    IO.println "    epoch count — no reference card, no hardware factor, no bottleneck assumption."
    IO.println "    §2j validated the method at 0.3% (ch9 extrapolated 3480s; real run 3491s)."
    IO.println "  * real data, so the ~6.3% per-epoch loader term (§2e-ter) is included — unlike the"
    IO.println "    scaled mode, whose probes are synthetic by design."
    IO.println "  * current certified bytes, so a re-render cannot silently invalidate it."
    IO.println "  * ⚠ the five Imagenette rows are TRAIN-ONLY. Their trainers (trainAdamSched) print no"
    IO.println "    per-epoch ms, so they are measured as median ms/step × 295 steps × 80 ep via the"
    IO.println "    MAX_STEPS probe — which returns before the eval pass. Eval adds ~5% on R34 and"
    IO.println "    ~8% on ConvNeXt (§2h). Add that back before comparing to a wall clock."
    IO.println "  * PJRT_FFI_RESIDENT is not set, so this is the COPYING path (§2d.3: residency is"
    IO.println "    2.03× on ResNet-34 bs32). Set it to measure the other one."
    return 0
  let denseMs ← runProbe ref.denseProbe "dense" ref.denseRefMs ref.card backend gpu runEnv
  let convMs  ← runProbe ref.convProbe  "conv"  ref.convRefMs  ref.card backend gpu runEnv
  let attnMs ← if ref.attnProbe.isEmpty then do
      IO.println s!"\n  ▸ attn probe SKIPPED — no runnable ViT probe on {ref.lowerer}. \
ch.9 has no {ref.lowerer} reference and prints n/a."
      pure none
    else runProbe ref.attnProbe "attn" ref.attnRefMs ref.card backend gpu runEnv (stepProbe := some 100)
  match denseMs, convMs with
  | some dMs, some cMs =>
    let aMs := attnMs.getD 0
    IO.println s!"\n  ESTIMATED training time on YOUR gpu  (ref = {ref.card}, {ref.lowerer}):\n"
    let rule := "  " ++ String.ofList (List.replicate 47 '-')
    IO.println s!"  {padR "Chapter" 18}{padR "family" 8}{padR s!"ref({ref.card})" 18}your gpu"
    IO.println rule
    let mut yourLoTotal := 0
    let mut yourHiTotal := 0
    let mut refTotal := 0
    let mut covered := 0
    let mut anyBracket := false
    for it in benchTable do
      match it.refFor ref.col, yourSecRange ref it dMs cMs aMs with
      | some refSec, some (lo, hi) =>
        yourLoTotal := yourLoTotal + lo
        yourHiTotal := yourHiTotal + hi
        refTotal := refTotal + refSec
        covered := covered + 1
        if lo != hi then anyBracket := true
        let flag := if it.family == "attn" && aMs == 0 then " *proxy" else ""
        -- ⚠ Single number, deliberately. A bracket wide enough to hold the real cross-vendor
        --   spread (0.585-1.411, measured) would tell a first-time user nothing they could plan
        --   with — and with a per-vendor reference column the on-vendor factors read ~1.00 anyway,
        --   so the honest number is simply the measured one. `yourSecOf` is the estimate; the
        --   transport-leaning `hi` is kept only for the off-vendor note below.
        IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR (fmtDur refSec) 18}{fmtDur lo}{flag}"
      | _, _ =>
        IO.println s!"  {padR it.chapter 18}{padR it.family 8}{padR "n/a" 18}n/a  (no {ref.lowerer} reference)"
    IO.println rule
    let totalLabel := if covered == benchTable.length then "Full Part-1 training"
                      else s!"Part-1 training ({covered} of {benchTable.length} ch.)"
    IO.println s!"  {padR totalLabel 30}{padR (fmtDur refTotal) 18}{fmtDur yourLoTotal}"
    IO.println "\n  `lake run` tiers on your gpu (training time):"
    for (tier, label) in [("mnist", "lake run mnist"), ("cifar", "lake run cifar"),
                          ("imagenette", "lake run imagenette")] do
      let items := benchTable.filter (·.tier == tier)
      let refS := (items.filterMap (·.refFor ref.col)).foldl (· + ·) 0
      let rs := items.filterMap (fun it => yourSecRange ref it dMs cMs aMs)
      let yourLo := (rs.map (·.1)).foldl (· + ·) 0
      let yourHi := (rs.map (·.2)).foldl (· + ·) 0
      let miss := items.length - (items.filterMap (·.refFor ref.col)).length
      let note := if miss == 0 then "" else s!"   ({miss} ch. n/a)"
      -- All three demo groups have an `-xla` peer, so name the command the user would
      -- actually run on this path rather than its IREE sibling.
      let suffix := if ref.xla then "-xla" else ""
      IO.println s!"    {padR (label ++ suffix) 26}{padR (fmtDur refS) 9}→  {fmtDur yourLo}{note}"
    IO.println s!"\n  * probes: {ref.denseProbe} / {ref.convProbe}\
{if ref.attnProbe.isEmpty then " / (no attn probe)" else s!" / {ref.attnProbe}"}"
    IO.println s!"  * ref column measured on {ref.card}; on that card the factors read ~1.00 and"
    IO.println "    the estimate is the measurement. On other cards it is scaled — accurate within a"
    IO.println "    vendor, approximate across one (the cross-vendor per-chapter spread is 2.4×)."
    IO.println "  * Imagenette rows are TRAIN-ONLY; eval adds ~5-10%. Training time only."
    if ref.xla then
      IO.println "  * BENCH_DIRECT=1 measures YOUR box instead of scaling (needs the datasets, ~5 min)."
    else
      IO.println "  * first run adds ~10-15 min/arch IREE compile."
    return 0
  | _, _ =>
    if ref.xla then
      IO.eprintln "\n  probe failed — check that ffi/libpjrt_ffi.so built and $PJRT_PLUGIN resolves"
      IO.eprintln "  (reported above). No estimate produced."
    else
      IO.eprintln "\n  probe failed — need data (`lake run download`) and the IREE venv from"
      IO.eprintln "  Track 2. No estimate produced."
    return 1

/-- `lake run benchmark` — probe this GPU on the **IREE** path, print a per-chapter
    training-time estimate. The XLA peer is `lake run benchmark-xla`; the two scale from
    separate reference columns measured on separate lowerers and are not interchangeable
    (handoff §2j). -/
script benchmark do
  runBenchmark ireeRef

/-- `lake run benchmark-xla` — the XLA/PJRT peer of `lake run benchmark`.

    Same probe recipe, same steady-state rule, same printer — the only difference is which
    lowerer the probe binaries linked and which reference column they scale from. Both of
    those move together inside `xlaRef`, which is what stops the §2j trap: an XLA probe
    divided by IREE's anchors would report Part-1 training ~2-5× too fast, silently.

    **All 3 probes and all 9 chapters as of 2026-07-30.** It was 2-and-8 until then, because
    `vit-verified-adam-xla` did not execute on this box; it now does, with no workaround, though
    the MIOpen failure it used to hit is non-deterministic rather than fixed — see
    `probeAttnRefMsXla`. The `n/a` machinery is retained on purpose: it is what would keep an
    unmeasured row honest, and it is what ch.9 needed for two days.

    All nine XLA references are measured; ch.9's is a real 80-epoch run as of 2026-07-30
    (3491 s, val 71.31%), which also confirmed the marginal-epoch extrapolation it replaced to
    within 0.3%. See `benchTable`.

    Needs no venv: the XLA binaries compile in-process rather than shelling out to
    `iree-compile`. It does build `ffi/libpjrt_ffi.so` if missing/stale and report whether
    `$PJRT_PLUGIN` resolves, exactly as `lake run {mnist,cifar,imagenette}-xla` do. -/
script «benchmark-xla» do
  -- ▶ ONE PATH PER VENDOR. The reference column is measured on that vendor's card, so on a
  --   4060 Ti (CUDA) or a 7900 XTX (ROCm) the factors read ~1.00 and the table is the measurement.
  let be ← match ← IO.getEnv "IREE_BACKEND" with | some b => pure b | none => detectBackend
  runBenchmark (if be == "cuda" then xlaRefCuda else xlaRefRocm)
