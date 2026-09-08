# lakefile roots — the note behind every root

Archived 2026-09-08 (`planning/cleanup_backlog.md` §6). Until this date each entry of the
`Proofs`, `Certs` and `CertsHeavy` root arrays in `lakefile.lean` carried a comment saying what
the module was and why it was a root — a per-module essay that the module's own `/-!` header
duplicates in most cases. The arrays are bare lists now; the comments are here verbatim, in
array order, under the module they preceded. A paragraph reads as it was written at the date it
names; the module header is the current description.

## `lean_lib «Proofs»`

### `LeanMlir.Proofs.Codegen.MlpRender`

the renderers the verified-render drift guard re-elaborates

### `LeanMlir.Proofs.Codegen.ResNet50RenderB`

R50 phase 2: the bottleneck train-step renderer. SOLE writer of
verified_mlir/resnet50in_*_train_step.mlir, so its olean must exist wherever the
corpus is built (same reason the two RenderB entries below carry).

### `LeanMlir.Proofs.Codegen.ConvNeXtRenderB`

⚠ A ROOT, and it has to be: nothing imports a leaf renderer, so on a fresh
runner this is the ONLY way its `.olean` — and therefore the six stochastic-
depth artifacts its `#eval`s write — ever get built (scripts/check_render_coverage.py).

## `lean_lib «Certs»`

### `LeanMlir.Proofs.Architectures.MnistCNN`

Chapter-4 MNIST 2D CNN (no BN): conditional whole-net VJP
+ a concrete instance with every smoothness hyp discharged.

### `LeanMlir.Proofs.Training.JacobianSeal`

Nonzero-Jacobian seal (planning/archive/whole_network_backward.md Item B): the
generic "one nonzero Jacobian entry ⇒ non-trivial backward" bridge.

### `LeanMlir.Proofs.Training.MobileNetV2JacobianSeal`

Item B2: the seal discharged at the live MobileNetV2 witness —
`fderiv ℝ forward 0 ≠ 0` ⇒ non-trivial whole-net backward (level 3).

### `LeanMlir.Proofs.Foundation.StridedConv`

Chapter-6 ResNet Milestone B: stride-2 SAME convolution (the hard
new downsampling op) = decimate ∘ stride-1 conv, with its input-VJP.

### `LeanMlir.Proofs.Foundation.ResNet34`

Chapter-6 ResNet Milestone B: the deep-block chain (a list of
same-type residual blocks composes to one VJP) — 16-block depth.

### `LeanMlir.Proofs.Foundation.PerChannelBN`

Chapter-6 ResNet Milestone B8: per-channel BatchNorm (block-diagonal
VJP via a per-row generalization of `rowwise_has_vjp_mat`).

### `LeanMlir.Proofs.Foundation.ResNet34LiveGeneric`

Limit-D strengthening: the 224×224 live ResNet-34 whole-net VJP with
the three downsample projection convs generalized to ARBITRARY kernels
(the β-positivity discharge is weight-independent). 3-axiom clean.

### `LeanMlir.Proofs.Codegen.MatBridge`

opt-in Mathlib.Matrix interop; not imported by the suite,
listed here so CI keeps it green.

### `LeanMlir.Proofs.Foundation.IR`

denoted StableHLO-subset IR (Phase 0a/0b spike); bridges the
emitted backward graph to the proven HasVJP.backward.

### `LeanMlir.Proofs.Codegen.StableHLO`

R4 printer-faithfulness Stage A (ch 2): StableHLO-subset AST +
denotation `den` proven to match the linear train-step math.

### `LeanMlir.Proofs.Codegen.StableHLOParse`

R4 syntactic core: op-graph serialization round-trip
(parse (toToks (skel a)) = a).

### `LeanMlir.Proofs.Codegen.StableHLOLex`

R4 syntactic LEXER numeric keystone: decimal Nat⟷String
round-trip (parseNat (toString n) = n), the load-bearing
first rung of text→token faithfulness.

### `LeanMlir.Proofs.Foundation.LinearTrainStep`

M1 (planning/archive/verified_train_step.md): the linear train step bundled
into one SGD-on-certified-softmax-CE-gradient theorem.

### `LeanMlir.Proofs.Foundation.MlpTrainStep`

M2: the MLP per-layer parameter-gradient assembly (Crux A).

### `LeanMlir.Proofs.Foundation.CnnTrainStep`

M3: the CNN convolution parameter-gradient bridges.

### `LeanMlir.Proofs.Codegen.MlpRender`

MLP render half: the train-step text as a name-threaded render of the
proven forward graphs (multi-intermediate generalization).

### `LeanMlir.Proofs.Codegen.CnnRender`

CNN render half: the CNN train-step text rendered from `cnnFwdGraph`,
with flat→NCHW reshape glue bridging the conv param-grad tail.

### `LeanMlir.Proofs.Architectures.CifarBnClose`

⛔ `LeanMlir.Proofs.Codegen.ResNet34Render` was here until 2026-09-06, when 4c leg 1
retired it: the per-example renderer was the last writer in the suite emitting a
train step at per-example BatchNorm, so `resnet34_fwd` could not be a prefix of both
it and the batch-BN Adam step. Everything the inference forward needs — plus
`bnSite`/`R34Bn`, which ResNet-50 shares — moved into ResNet34RenderB, which is now
the sole writer of every ResNet-34 artifact. `planning/archive/renderer_convergence.md`.
CIFAR-BN close: the per-channel BN scale/shift (dγ, dβ) param-grad
bridges — the affine BN analogue of `bias_grad_bridge`.

### `LeanMlir.Proofs.Foundation.CnnChainClose`

CNN conv-close upgrade: the conv param closes pinned to the actual
backward-chain cotangent (Back3 maxpool/conv via flatDenote + relu masks).

### `LeanMlir.Proofs.Architectures.Cifar8Close`

Deeper (8-conv) CIFAR-CNN close: cifar8{,Bn}FwdGraph_faithful's backward
peer — each conv W/b, BN γ/β, dense W/b output pinned to the actual 4-stage
backward-chain cotangent (the CnnChainClose recipe + BN, two more pool stages).

### `LeanMlir.Proofs.Architectures.MobileNetV2Close`

MobileNetV2 close (Item C): the depthwise (stride-1/2) + strided-conv
parameter-gradient bridges — every MobileNetV2 train-step param output
certified θ − lr·(certified Jacobian · cotangent).

### `LeanMlir.Proofs.Codegen.MobileNetV2RenderPC`

MobileNetV2 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph
(matches the operational render's BN flavor) + faithfulness to the
per-channel ℝ-forward. Prerequisite for the structured render (Item B).

### `LeanMlir.Proofs.Architectures.MobileNetV2ChainClose`

MobileNetV2 cotangent-chain close (Item D): the Item C conv/depthwise bridges
pinned to the inverted-residual backward chain (relu6 kink + depthwise + stride-2).

### `LeanMlir.Proofs.Foundation.ConvLossFold`

The cotangent pass / = ∂loss/∂θ fold: the certified per-layer conv/depthwise
Jacobian contracted with ∂loss/∂(layer output) IS the total loss gradient (pdiv_comp
at a smooth point). The conv analogue of mlp_hidden_total_loss_grad; program-wide.

### `LeanMlir.Proofs.Architectures.EfficientNetClose`

EfficientNet-B0 close (Item C): a FREE close — every param family reuses an
existing bridge (5×5 depthwise pinned; batch-norm γ/β = per-channel BN at m=N·h·w;
SE squeeze/excite are dense → M2). No new VJP.

### `LeanMlir.Proofs.Foundation.ResNet34Close`

ResNet-34 close (Item C): a FREE close — every r34 param family certified
by an existing bridge (the 7×7 stem + 3×3 strided projection pinned to the
generic strided conv W/b bridges; no new VJP).

### `LeanMlir.Proofs.Codegen.ResNet34RenderPC`

ResNet-34 render (Item A): the PER-CHANNEL-BN typed SHlo forward graph (full
16-block [3,4,6,3] net, 7×7 stem, maxpool) + per-block + whole-net faithfulness.

### `LeanMlir.Proofs.Architectures.ConvNeXtClose`

ConvNeXt close (Item C): mostly reuse (7×7 depthwise pinned to the generic
bridges) + the two genuinely-new families — layer-scale γ (dγ = x⊙dy) and
scalar-LN γ/β (the Vec-1 embedding bridging bn_grad_gamma/beta).

### `LeanMlir.Proofs.Architectures.ViTFwdGraph`

ViT close (Item A): the distinct-param 2-block ViT forward (vitForward2 +
whole-net VJP) and the heads=1 token forward graph + faithfulness
(den vitFwdGraph = vitForward2 via mhsa_layer_one_head).

### `LeanMlir.Proofs.Architectures.ViTClose`

ViT close (Item C): the per-token dense W/b family (row-lifted M2
outer product), row-lifted scalar-LN γ/β, pos-embed identity, CLS
masked-gather — every representative-ViT param family except the
patch conv certified.

### `LeanMlir.Proofs.Architectures.ViTChainClose`

ViT cotangent-chain close (Item D): the Item C bridges pinned to the
attention-block backward chain (SDPA matmul chain = the proven
sdpa_back_{Q,K,V} closed forms; the Q/K/V three-way fan-in at LN1).

### `LeanMlir.Proofs.Architectures.ViTVecLN`

ViT scaling pass (vector-[D] LN): layerNormVec block + vitForward2V
whole-net VJP + the rowScaleF/rowBiasF token graph + faithfulness +
the per-channel gamma/beta param bridges.

### `LeanMlir.Proofs.Architectures.ViTDepthK`

ViT scaling pass (multi-head + depth-k): headSliceF/headPadF tokens,
mhsa at general heads, then the distinct-param depth-k tower
(vitForwardKV). ViTDepthK imports ViTMultiHead, covering both.

### `LeanMlir.Proofs.Architectures.ViTMultiHeadChain`

ViT multi-head backward cotangents: the per-head SDPA backward the real
chain delivers at the Q/K/V dense outputs (Σ_h pad ∘ vitCotD{Q,K,V}(d_head)
∘ slice), pinned to the audited sdpa_back_{Q,K,V} (vitCotD{Q,K,V}mh_eq).
The multi-head/depth-12 tie's substantive build (mnv2 reduced→full).

### `LeanMlir.Proofs.Architectures.EfficientNetFullB0`

EfficientNet-B0 at full depth (16 distinct MBConv blocks, true BN+SE):
batched forward graph + whole-net VJP. Imports the EfficientNet
RenderPC + ChainClose modules, covering all three.

### `LeanMlir.Proofs.Architectures.EfficientNetFullB0Eval`

Its INFERENCE twin: the same 16-block ladder at frozen statistics (49 BN sites,
one shared ε), the fourth block shape at eval (mbExpFwdBEval / mbExpGraphBEval,
which the 3-block eval render has no instance of), the typed graph and its
faithfulness. The rung EfficientNetFullFloatBudget's number ends at
(b0Full_float_logits_le_committed); the typed form of efficientnet_fwd_eval.

### `LeanMlir.Proofs.Architectures.ConvNeXtFullT`

Full ConvNeXt-T [3,3,9,3]: forward graph + faithfulness + whole-net
VJP. Imports ConvNeXtChainClose, covering both.

### `LeanMlir.Proofs.Architectures.MobileNetV2FullPaper`

Paper-spec full MobileNetV2 (all 17 [t,c,n,s] bottlenecks): forward
graph + faithfulness.

### `LeanMlir.Proofs.Architectures.MobileNetV2FullVJP`

...and its whole-net input-VJP at all 17, folded over the same weight
bundles. Pointwise (`_at`) — relu6 is kinked, so unlike EfficientNet-B0's
swish the global form is unavailable. Imports MobileNetV2BackCertifiedTie
for the per-block body VJPs it delegates to.

### `LeanMlir.Proofs.Float.FloatBridge`

ℝ→Float32 bridge, Tier 1: standard-model rounding (hypothesis-style,
no axioms) + forward error bounds for the linear/MLP nets
(dot/dense budgets, ReLU exact-in-float Lipschitz pass-through).

### `LeanMlir.Proofs.Float.FloatSubnormalBridge`

Subnormal-floor closure (planning §2): the honest FaithfulFloatModel
(relative bound on normals + the gradual-underflow absolute floor),
FloatModel = its η→0 face, the BN/LN denominator stays-normal
invariant (rsqrt keystone never underflows), and the residual floor
proved globally negligible. Converts FloatBridge's subnormal caveat
into lemmas.

### `LeanMlir.Proofs.Training.SgdDescent`

Inexact-gradient descent over ℝ (MVT form): an η-accurate gradient
oracle + segment smoothness ⇒ the SGD step still decreases the loss,
with an explicit decrease. The keystone the FloatBridge budgets
plug into ("close" ⇒ "still trains").

### `LeanMlir.Proofs.Training.SgdDescentLinear`

The smoothness hypothesis DISCHARGED for the Chapter-2 linear
softmax-CE loss: explicit segment-Lipschitz constant 2a²/(1−2aD)
via the softmax ratio sandwich (no Hessian), and the capstone —
one inexact SGD step provably decreases the cross-entropy loss.

### `LeanMlir.Proofs.Training.SgdDescentMlp`

The smoothness hypothesis discharged through the Chapter-3
MLP: under quantitative ReLU margins (the step cannot flip a
mask sign) the loss-of-one-layer maps get explicit
segment-Lipschitz constants, and one inexact SGD step on each
weight layer provably decreases the cross-entropy loss.

### `LeanMlir.Proofs.Training.SgdDescentCnn`

The descent program reaches the Chapter-4 CNN: quantitative
max-pool selection margins (the argmax freezes along the step
segment), pool 1-Lipschitz/ℓ1-contraction, conv kernel drift.

### `LeanMlir.Proofs.Training.SgdDescentCifar`

CIFAR-8 last-conv SGD descent: the first non-MNIST provable descent. CIFAR-8's tail
(conv W₈ → relu → maxpool → 3 denses) IS cnn_conv2's architecture, so descent at the
last conv (earlier 7 layers frozen) is an instance — non-vacuous lr. Full-depth descent
stays open (the per-layer operator-norm product in hsmall compounds to vacuity).

### `LeanMlir.Proofs.Float.BnFloatBridge`

BN float keystone: 1/√ Lipschitz on [ε,∞) + the inverse-stddev
rounding budget (rsqrt accuracy + variance error, ε-floor).

### `LeanMlir.Proofs.Float.Resnet34FloatBridge`

residual additive fan-in float closeness (add_close / reluAdd_close)
— the new structural op toward the ResNet-34 float bridge.

### `LeanMlir.Proofs.Float.BnInputBridge`

real-BN input-sensitivity (mean/var/istd/forward Lipschitz) — the
per-block composition enabler (the float BN's input is perturbed).

### `LeanMlir.Proofs.Float.Resnet34BlockBridge`

first assembled ResNet block step: relu(BN(·)) at a perturbed BN
input = rounding (bnForward_close_of) + input-shift (bnForward_input_close).

### `LeanMlir.Proofs.Float.FloatComposeBridge`

whole-net certificate backbone: FloatClose composes (moduli ∘, magnitudes
thread) — the whole net is the fold of per-op budgets.

### `LeanMlir.Proofs.Architectures.MobileNetV2FullPaperEval`

⭐ The PAPER net's inference twin: the same 17-block [t,c,n,s] ladder at frozen
statistics (52 BN sites, one shared eps), its typed graph and faithfulness. The
rung MobileNetV2PaperFloatBudget's number ends at
(mnv2Paper_float_logits_le_committed). ⭐ Its SSA names are bnSiteP's — %stnmu,
%b{k}enmu/%b{k}dnmu/%b{k}pnmu, %hnmu — so the typed graph diffs against
mobilenetv2_fwd_eval line for line, where the six-block eval graph's %mue1 matches
no artifact and the 17-block TRAINING graph's %b17gp is the render's %gp17.

### `LeanMlir.Proofs.Codegen.EfficientNetRenderPCEval`

The B0 INFERENCE forward graph + whole-net faithfulness (the eval twin of
efficientnetFwdGraphB_faithful). bnBatchLA is the ONE op in the B0 render that is
not batchMap N of a per-example op — it reduces across examples; at frozen stats
the `bnEval` descriptor denotes batchMap N (bnPerChannelEvalTensor3 …) by rfl, so
the eval forward is per-example throughout. The BN mode a whole-net B0 float
number can be stated at (the training one's modulus is quadratic in the window).

### `LeanMlir.Proofs.Foundation.BatchMapVJPAt`

⭐⭐ ResNet-34's whole-net forward and typed graph at TRUE BATCH BN (T1-forward, T2).
ResNet34RenderPC states the same ladder at PER-EXAMPLE BN, which is resnet34_fwd's
world and the Imagenette SGD trainer's, but NOT the Adam/momentum steps' — those
reduce [0,2,3], and they are where the quoted ImageNet accuracies come from
(formalization.yaml 4e). Nothing about the blocks is new: ResNet34BackB0 already
carries the batched stages, their _at VJPs and their backward graphs at bnBatchLA.
⚠ Symmetric padding at all seven stride-2 sites (.convStrided, NOT .convStridedXla)
and a 3x3/s2 stem pool; both are invisible to the types.
⚠ N stays a variable — T1/T2 carry no numerals, so the batch is pinned only at T4/T5.
⭐ `batchMap` at a POINT — the pointwise peer of batchMap_has_vjp, and the one lemma
between ResNet34FullB and r34's whole-net VJP. EfficientNet never needed it (swish
is smooth and B0's stem has no pool); r34's stem is batchMap N (maxPool3s2Flat),
and a max-pool has no derivative at a tie. ⭐ pdivMat_rowIndep's global
`Differentiable` weakens to differentiability at each ROW with no change to the
argument — every use of it in that proof is already at a row.

### `LeanMlir.Proofs.Architectures.ResNet34FullBVJP`

⭐⭐ …and its whole-net input-VJP (T1's VJP half). Delegation only: the two batched
block VJPs are ResNet34BackB0's, and the one thing that did not exist is
batchMap_has_vjp_at, for the stem pool. ⛔ Each block carries TWO relu clauses —
the body's mid-relu AND the post-residual OUTER relu, ResNet's structural
difference from MobileNetV2/EfficientNet, whose residual add IS the block output.
⭐ The head takes no hypothesis (GAP and dense are smooth batchMaps).

### `LeanMlir.Proofs.Foundation.ResNet34FaithfulPoCB`

⭐ T3's §1 fold at batch BN, and it is UN-FUSED. Every batched r34 train step —
sgd, the Adam family, mom256 and the data-parallel peers — emits the RAW gradient
(*GradB) and hands it to an optimizer tail; the fused theta - lr*grad op only
appears where the optimizer is SGD-inline, which EfficientNet's is and r34's is
not. ⛔ Every den=certified lemma in the repo before this one is at the fused form.
⭐ The un-fused statement covers every optimizer variant at once, and it is
EfficientNetFaithfulPoC's proofs minus the `congr 1`/`congrArg (lr * .)` peeling —
the *SgdB_eq_grad family already said the fusion is rfl.
⚠ SYMMETRIC padding: convStridedWeightGradB / flatConvStride2, not B0's Xla peers.

### `LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoCG`

⭐ 4b: the SAME fold for the other four nets, one file each, so that every net's
T3 is stated at the node its Adam/RMSProp/LAMB artifact actually emits rather than
at the SGD-inline fused op the book does not name. All four are delegations plus a
handful of new op kinds; the arithmetic is the *Sgd_eq_grad rfl read once.
⭐ Five of B0's eight op kinds are ResNet-34's at the same generality, so 4b.1 is
three new lemmas: the XLA-SAME stem and the two depthwise weights
(EfficientNetFaithfulPoCG.lean).

### `LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCG`

⭐ ConvNeXt's twelve gradient nodes. psW (the 4x4/s4 patchify stem) had no fused
peer to begin with — a declared §5 carve-out — and 4b makes that shape the norm.
⚠ SYMMETRIC padding at the three 2x2/s2 downsamples (ConvNeXtFaithfulPoCG.lean).

### `LeanMlir.Proofs.Architectures.ViTFaithfulPoCG`

⭐ ViT's ten gradient nodes at the VECTOR LayerNorm the shipped vitForwardKV runs.
rowDenseBiasGrad appears twice against two different certified Jacobians (a dense
bias and an LN beta are the same reduce), as it does in the fused file
(ViTFaithfulPoCG.lean).

### `LeanMlir.Proofs.Architectures.ViTFaithfulPoCGB`

⭐ 4c leg 4: the SAME ten nodes at the BATCHED traversal (vitBackAllB), which is
what every committed ViT artifact renders from after the leg. Measured 2026-09-07:
all nineteen drop-free artifacts re-render byte-identically off it, so this is a
statement about the AST the bytes are pretty of and not about different bytes.
⭐⭐ clsGrad_denB is the one the per-example fold could not make — the CLS token is
a shared [192] vector, so its gradient sums over the batch, and the per-example
render emits denseBiasGradB at N := 1 because pretty B lifted outside the AST.
Same emitted text, different function; den_rowDenseBiasGradB_at_one is the trap.
⚠ headBGradB_den is per-example at batchSlice n: biasGradB is the identity on its
operand and its batch reduce is emitted text, the per-example carve-out unchanged
(ViTFaithfulPoCGB.lean).

### `LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoCGB`

⭐ 4c leg 3: ConvNeXt's fourteen nodes at the BATCHED traversal. Owed BEFORE the swap: every
convnextin_* and *drop* artifact had rendered from that traversal since it existed,
with a fold only at the per-example constructors. The 22 channel-LN sites take
batchMap N (chanLNRows …) and batchSlice_batchMap peels the lift per example
(ConvNeXtFaithfulPoCGB.lean).

### `LeanMlir.Proofs.Foundation.Bf16GradNodes`

The bf16 gradient nodes, folded once for every net: nine *GradBBf16 kinds, den =
rnd outside the batch sum of the certified VJP at rounded operands (rowDense keeps
its f32-typed result and has no outer rounding). (Bf16GradNodes.lean)

### `LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoCPaperG`

⭐ MobileNetV2's twelve, at the BATCHED index — this net's two renders do not
overlap (the per-example one is SGD-inline only, the batched one AdamW-only), so
its Adam/RMSProp artifacts have no fused op to un-fuse and the fold goes straight
to *GradB. ⛔ Two header corrections to MobileNetV2FaithfulPoCPaper fall out: the
artifact it names did not exist even then (the writer passed mobilenetv2_train_step,
itself retired by 4c leg 2), and
the shipped parameter count is 158, not 210 — 210 is the convBias := true census
and both renders default to false (MobileNetV2FaithfulPoCPaperG.lean).

### `LeanMlir.Proofs.Foundation.SmoothedLossCot`

⭐ 4.2a: the SHARED label-smoothed loss cotangent, at a GENERAL target. Every T3 tie
in the repo pinned its top-of-chain cotangent to softmax - oneHot, the gradient of
plain CE at a hard label; the batched ImageNet renders compose
(softmax - t + alpha*t - alpha/K)/B from six kit ops, with t the graph INPUT (a soft
vector under mixup). softCE is CE against a target DISTRIBUTION (its gradient needs
no hypothesis on t at all), smoothTarget is label smoothing as a map on targets, and
smoothedCE_grad says the emitted expression IS that loss's gradient — not an
approximation of it (SmoothedLossCot.lean).

### `LeanMlir.Proofs.Foundation.ResNet34TiePoCB`

⭐⭐ 4.2a: ResNet-34's T3 §1a TIE at batch BN, un-fused and batched — the last piece
of r34's T3. Every parameter gradient node at the cotangent the emitted chain
delivers, threaded from the smoothed loss through the certified head backward and
the sixteen certified block backwards over resnet34ForwardB_full's own prefixes.
⭐⭐ The block cotangents are NOT derived here: 4.1d's r34IdB_has_vjp_at /
r34DownB_has_vjp_at ARE the certified block backwards, and
r34{BasicBlock,DownBlock}BackBatchedGraph_faithful already proves the emitted fan-in
denotes them — so r34{Id,Down}CotIn_eq_vjp close by rfl and add_comm, and the whole
file is 763 lines against the per-example tie's 615 with three times the content.
⭐ N is a binder and the capstone needs NO smoothness hypothesis: the folds are
forall-cot statements at explicitly constructed cotangents, and the kink conditions
enter only in the two _eq_vjp lemmas.
⛔ The census is 110 parameters, not the 146 the retired per-example tie named: both r34 renders
run convBias := false and the conv biases are zeroBiasPrelude's zero constants.

### `LeanMlir.Proofs.Architectures.MobileNetV2FullB`

⭐ 4.2 leg 1: MobileNetV2's T1-forward and T2 at batch BN — the second net whose
Proofs tier is re-stated in the world its Adam/RMSProp artifacts train in. Pure
enumeration: MobileNetV2BackB0 already carries the batched relu6 stages (cbrB,
dwbrB, dwbrBstrided) and projB with their _at VJPs and backward graphs at
bnBatchLA; what was missing is the level above. ⭐ IVW/IVWNoExp are REUSED from
MobileNetV2FullPaper — a weight bundle knows no BatchNorm world — and only the
top-level record is new, because it is generic in nCls where MNV2PaperWeights is
pinned at 10. ⚠ XLA-SAME padding at all five stride-2 sites (the stem conv and the
four strided depthwises), and NO stem pool, which is why this net needs no
batchMap_has_vjp_at. ⚠ Bias operands are the render's default convBias := false
names (%zb{c}); the census is 158 parameters, not 210.

### `LeanMlir.Proofs.Architectures.MobileNetV2FullBVJP`

⭐⭐ …and its whole-net input-VJP (T1's VJP half). Delegation only: mnv2BodyB and
mnv2DownBodyB ARE the two body shapes, residual_has_vjp_at wraps the first for the
ten skip blocks, and bnRelu6Stage_has_vjp_at is generic in the inner op so the
stride-2 stem is the same construction as every stride-1 stage. ⭐ Where r34 needed
a new Foundation lemma (batchMap_has_vjp_at, for its stem pool) this net needs
none: MobileNetV2 has no pool and its GAP/dense are smooth. ⛔ Two kink clauses per
bottleneck, both INSIDE the body (expand relu6, depthwise relu6) — the linear
bottleneck has no activation after project, so the residual add adds nothing; 35
relu6 sites in 19 binders. ⚠ Unlike r34's, the head is NOT hypothesis-free.
⭐ IVPos / IVNoExpPos are reused from MobileNetV2FullVJP — a BN epsilon's
positivity does not know which axis the norm reduces; only the smoothness bundles
need batched peers. ⭐ mnv2BodyB's family was generalised from one channel count to
ic/oc there (b11 and b17 are stride-1 bodies with ic ≠ oc, which the residual-only
statement could not express); mnv2DownBodyB already had that shape.

### `LeanMlir.Proofs.Foundation.MobileNetV2TiePoCB`

⭐⭐ 4.2c: MobileNetV2's T3 §1a TIE at batch BN, un-fused and batched — the last piece
of mnv2's T3, and the second net whose train step is tied at the artifact that
trains. Every parameter gradient node at the cotangent the emitted chain delivers,
threaded from the smoothed loss through the head's own four nodes and the seventeen
certified block backwards over mobilenetv2ForwardB_full's own prefixes.
⭐⭐ The block cotangents are NOT derived: 4.2b's mnv2{ExpOnly,Resid,Strided,NoExp}B
_has_vjp_at ARE the certified block backwards and MobileNetV2BackB0's
*BackBatchedGraph_faithful family already proves the emitted subgraphs denote them,
so three of the four _eq_vjp lemmas close by rfl (the residual one needs Eq.trans,
since mnv2ResidB unfolds to residual_has_vjp_at at an abbreviation).
⭐ ONE tie bundle covers twelve of the seventeen blocks: a skip changes only the dx
handed to the previous block, never a parameter cotangent.
⭐ N is a binder and the capstone needs NO smoothness hypothesis; the kink and
positivity conditions enter only in the four _eq_vjp lemmas.
⛔ The census is 158 parameters, not the 210 slots stated: convBias := false, so the
52 bias nodes are not emitted. ⛔ ONE REPLICA — the all-reduce is text outside the AST.

### `LeanMlir.Proofs.Architectures.EfficientNetTiePoCG`

⭐ 4b's capstone re-pointing, EfficientNet-B0: the 262-parameter §1a tie restated
at the RAW gradient nodes (*GradB, what every non-SGD-inline step emits) and at the
SHARED SMOOTHED loss cotangent at a general target. The fused file pins g to
softmax - oneHot, the gradient of plain CE at a hard label, which no ImageNet
artifact computes. ⭐ No new mathematics: every cotangent chain, activation and
Jacobian witness is EfficientNetTiePoC's, and each conjunct is that file's proof
with the `theta - lr *` peeling dropped — the fusion is rfl. The lr/wN/bN/gN/lrStr
binders go with the wrapper. ⭐ The head takes `g` as a BINDER where the fused one
computes it internally; that is the whole of the loss axis, since the per-block
ties were already forall-cot (EfficientNetTiePoCG.lean).

### `LeanMlir.Proofs.Architectures.ConvNeXtTiePoCGB`

⭐⭐ 4b's capstone for ConvNeXt-T, THREE axes at once: the un-fused *GradB nodes,
the smoothed loss at a general target, and the BATCHED index (the per-example
ConvNeXtTiePoC was at a single image with the batch outside the AST). Every
activation is batchMap N of the fused file's prefix and every cotangent is
batchMapAux N of its chain — honest for this net because no ConvNeXt op couples
examples. N and nC are binders; no smoothness hypothesis (GELU). The loss chain is
smoothedLossCotGraphDiv, the softmaxDiv∘expe spelling at the plain N·K width that
ConvNeXtRenderB and ViTRenderB emit (SmoothedLossCot.lean) (ConvNeXtTiePoCGB.lean).

### `LeanMlir.Proofs.Architectures.ViTTiePoCGB`

⭐⭐ 4b's capstone for ViT-Tiny — the set closes at FIVE OF FIVE. ConvNeXtTiePoCGB's
three-axis transformation applied to ViTTiePoC: the *GradB nodes every vitin_*
artifact emits, the smoothed loss at a general target (smoothedLossCotGraphDiv,
reused), and N a binder via batchMap/batchMapAux of the per-example prefixes and
chain. ⭐ The CLS token's gradient is tied with the batch sum INSIDE den
(ViTPoCGB.clsGrad_denB at the real embed cotangent), which the per-example
capstone could state only at N = 1. No *BackBatchedGraph_faithful family needed
(ViTTiePoCGB.lean).

### `LeanMlir.Proofs.Foundation.DataParallel`

⭐⭐ 4d piece 1: DATA PARALLELISM at the R-level — what function a *dp* run
actually minimised. Every *dp* artifact all-reduces each parameter gradient and
divides by R as emitted TEXT outside the SHlo AST, so every tie in the repo
(r34_net_tiedB, mnv2_net_tiedB, efficientnet_net_tiedG) is stated at the
per-replica node and disclaims the collective. dpMeanGrad_eq_grad_meanLoss names
that collective as a gradient: (1/R) sum_r g_r IS grad of (1/R) sum_r L_r, for
ANY per-replica losses, coupled or not.
⭐⭐ And then it SPLITS the two worlds. With no batch coupling (the replica loss is
a mean over its own slice) meanLoss_shard says the mean of the R replica losses is
literally the mean over the global R*N batch — so the DP step is the single-device
step at batch R*N, at ANY sharding (the equiv is a binder; the contiguous cut is
one instance). ⛔ With batch coupling that is FALSE and dpMeanGrad_ne_globalBatchGrad
is the two-replica witness: a training-mode BatchNorm reads a NONLINEAR function of
its own slice's statistics, and no all-reduce repairs it, because nothing
all-reduces mu/var. That is why N in the batch-BN tiers is the PER-CARD batch.
⭐⭐ dpIterate_lockstep + dpIterate_eq_meanLossTrain: n steps of R replicas ARE n
steps of ordinary single-device training on the mean loss — the property
VerifiedTrain.lean relies on when it checkpoints from replica 0. The shared start
is a HYPOTHESIS; the driver establishing it is calling logic (4d piece 3).
⚠ pdiv_const_smul belongs in Tensor.lean and is here because that file is the root
of the corpus and a definition added to it rebuilds all of Certs.

### `LeanMlir.Proofs.Foundation.DataParallelNode`

⭐⭐ 4d piece 2: the collective as an AST node. allReduceMeanF's den is dpMean of
the per-replica gradient nodes, its skel reads replica 0 (SPMD), its emit is the
old text verbatim so no *dp* artifact moved, and the parser round-trip has its
case. This leaf composes it with piece 1 (the gradient of the MEAN loss), with
4b's fold (convWeightGradB shown) and with the AdamW tail — the "den (tail
(allReduceMeanF …)) = adamW (dpMean …)" statement §4d asked for
(DataParallelNode.lean).

### `LeanMlir.Proofs.Codegen.LambTriple`

⭐ ResNet-50's two prerequisites (§3.5), both leaf files for the same reason
DataParallel is one: their natural homes (Lamb.lean, StableHLO.lean) have 315+
downstream modules apiece and these have none.
⭐ THE LAMB TRIPLE, assembled — the peer of adamW_triple_faithful, and the last
thing between LAMB and a ResNet-50 T3 at resnet50in160_lambaccdp8x64bce.
⛔ The audit's "LAMB has NO faithfulness theorem" named the wrong cause:
lambDirF_faithful and lambScaleF_faithful have said the emitted ops denote lambDir
and lambScale since LAMB landed, both by rfl; only the (theta', m', v') assembly
was missing. ⭐ The scalar child is a BINDER, and the two shipped instantiations
are corollaries: the committed one is gradSumSqAccF seeded at %lzero over theta
ALONE (that single-leaf fold is the entire difference from the global-norm clip),
and the excluded one is %lzero itself -- timm's no_weight_decay group, which is NOT
layer-adapted, and lamb_triple_faithful_excluded says the emitted step is then
exactly theta - lr*r at trust 1 (LambTriple.lean).

### `LeanMlir.Proofs.Foundation.BceLossCot`

⭐ BCE-WITH-LOGITS' COTANGENT — SmoothedLossCot's twin at RSB-A2/A3's loss, and
the second R50 prerequisite. ResNet50RenderB's bce := true path emits three ops
(sigmoidB -> subB -> divConstB, i.e. (sigma(z) - t)/(B*K)) where softmax-CE emits
five, and nothing said that chain was a loss's gradient.
⭐⭐ bceLogits_eq_logSigmoid is what keeps bceLogits_grad from being circular:
softplus(z) - t*z IS -[t log sigma(z) + (1-t) log(1 - sigma(z))], so the function
is binary cross-entropy rather than whatever has the wanted derivative. The stable
softplus form is the renderer's own %loss spelling.
⭐ NO hypothesis on the target, where softmax-CE's gradient needs sum t = 1: BCE is
per-class and separable, which is the point under mixup. ⚠⚠ The divisor is B*K,
not B -- timm's BinaryCrossEntropy is reduction='mean' over B x C, and at K = 1000
the two differ by 1000x on the effective step (BceLossCot.lean).

### `LeanMlir.Proofs.Architectures.ResNet50FullB`

⭐⭐ §3.5(a): RESNET-50's T1 at batch BatchNorm — the first net-level tier this net
has ever had. ⭐ The one net where T1 matches the trained world from the start:
ResNet50RenderB has always been the sole renderer, so bnBatchLA is the world of
resnet50_fwd and of every train step, and there is no BN-world port to do later.
⭐⭐ Pure enumeration: ResNet50BackB0 already carries all three batched bottleneck
forms with their _at VJPs and backward graphs, and the STEM AND HEAD ARE
RESNET-34's (r34StemB / r34HeadB are generic in their widths), so r34StemB_has_vjp_at
and r34HeadB_has_vjp apply verbatim and nothing new was needed one tier down --
unlike r34's own T1, which needed batchMap_has_vjp_at for exactly that stem pool.
⭐ ONE weight record (R50ProjW) serves both projection forms; they differ only in
which convs are strided, which is a property of the forward.
⭐⭐ q IS A BINDER: ResNet-50 ships at TWO resolutions (resnet50in_fwd at 224 = 32*7
and resnet50in160_fwd at 160 = 32*5, the net the quoted 76.66% trains), so one
statement covers both. ⚠ Every resolution is an explicit nest of 2 * (...), never
8 * q: those are equal Nats and NOT definitionally equal terms at a variable q.
⛔ THREE kink clauses per bottleneck (two interior relus + the post-residual one),
where r34's basic block has two; 48 clauses in 16 bundles. ⛔ 0 < q is a real
hypothesis, where r34's literal ladder needed none (the stem pool's output grid).
⚠⚠ v1.5 stride placement (on the 3x3, not the leading 1x1) and SYMMETRIC padding
at all five stride-2 sites; neither is visible to the types
(ResNet50FullB.lean, ResNet50FullBVJP.lean).

### `LeanMlir.Proofs.Architectures.MobileNetV4FullB`

⭐⭐ MOBILENETV4-CONV-M's T1 and T2 at batch BatchNorm -- the last net in
planning/archive/proofs_tier_to_paper_nets.md §2's table with nothing at the net level.
⭐⭐ The whole trunk is ONE CertLayer: 24 stages composed with CertLayer.comp and
CertLayer.residual, so .fwd IS the forward, .ok IS the ~60-clause smoothness
hypothesis (assembled stage by stage rather than written down), .vjp IS the
whole-trunk HasVJPAt and .faithful IS the whole-trunk BACKWARD-graph faithfulness.
The apex therefore takes TWO hypotheses where ResNet-50's takes 33, and needs no
r50Pre_k prefix chain at all.
⚠⚠ The STEM sits outside that chain and cannot be inside it: no render emits a
gradient into %x, so there is no convStridedXlaBackBatched token and hence no
backward graph for a CertLayer to be faithful to. EfficientNet-B0's enetTrunk takes
its stem as a parameter for exactly this reason.
⚠ The 21 block rows are NAMED constants pinned to mnv4Blocks by #guard, not list
indices: UibParams (mnv4Blocks[3]!) in a type forces whnf through List.get! at
every use. ⚠ Rows 4/5/10, 12/18 and 15/19/20 are shape-identical, so what pins
their identity is the SSA NAMES the T2 graph reads off the row, not the types.
⚠ TWO padding phases in one net: XLA-SAME at the stem, symmetric everywhere else.
⚠⚠ No accuracy is quoted for Conv-M -- no Imagenette run, no verified ImageNet
run; what pins these tiers to the reference is the 2026-09-07 tie pair
(MobileNetV4FullB.lean, MobileNetV4FullBVJP.lean).

### `LeanMlir.Proofs.Foundation.MobileNetV4FaithfulPoCB`

⭐⭐ MOBILENETV4's T3 §1 fold: every parameter GRADIENT node the batched train step
emits denotes the certified gradient, by block profile. ZERO new fp32 op-kind
lemmas -- MNv4's nine kinds are ResNet-34's, EfficientNet-B0's and its own dense
pair, drawn from three files and not one. 3 + 6 + 13x12 + 4x9 + 4x6 + 8 = 233,
the artifact's signature minus %x, and every slot exercised (bias-free by
construction, so no convBias census to over-count).
⛔⛔ An ABSENT depthwise gets NO conjunct: the ConvNeXt and FFN profiles are not
ExtraDW with a spare slot -- the render emits no token there, and a conjunct would
be the den of a node the artifact does not have.
⚠⚠ TWO padding phases: the stem is convStridedXlaWeightGradB (XLA-SAME, B0's op)
and the fused stage is convStridedWeightGradB (symmetric, ResNet's). Identical
types, identical emitted shapes, different certificates.
bf16: the five *GradBBf16 kinds MNv4 emits are folded in Bf16GradNodes.lean.
(MobileNetV4FaithfulPoCB.lean)

### `LeanMlir.Proofs.Foundation.MobileNetV4TiePoCB`

⭐⭐ MOBILENETV4's T3 §1a TIE: every gradient node at the cotangent the render's own
backward chain delivers, driven by a loss cotangent g at the logits -- all 233.
⭐ The UIB bottleneck is LINEAR (no activation after the project BN, none after the
skip add), so dyOut reaches the project BN's gamma/beta UNMASKED -- shorter than
ResNet's chain, whose residual carries a relu mask there.
⭐⭐ ONE cotangent chain serves all three stride-1 profiles: mnv4CotEn DISPATCHES on
s.postDWk exactly as mnv4PostDWSlot does and off the same row, so a ConvNeXt-like
row's expand BN reads the project conv's input-VJP directly.
⚠⚠ Every definition and theorem is GENERIC IN THE ROW (or in its widths): MNv4's
resolutions are literals, and stating any of this at them lets `den` run and the
kernel give up -- the same failure MobileNetV4FullB.lean records four times. The
capstone instantiates at the 21 concrete rows, which is application and is free.
⛔ g is a BINDER; mnv4_lossCot_is_smoothedCE_grad instantiates it at the
label-smoothed softmax chain (softmaxRow at m := 1 -- ResNet's spelling, NOT
ConvNeXt's expe-then-softmaxDiv; the two take different lemmas and nothing in the
types tells them apart) (MobileNetV4TiePoCB.lean).

### `LeanMlir.Proofs.Foundation.MobileNetV4WholeBackCertifiedTieB`

⭐⭐ MOBILENETV4's T6 (mnv4_proofs_tier.md §Session 3, ~3 s) -- the last tier this
net can have, and with it Conv-M is certified from its R forward through its typed
graph, its 233-parameter train-step tie and now its whole-net input gradient.
⭐ NOT ONE new float leaf: MNv4's stem is EfficientNet-B0's (3x3/s2 at XLA-SAME,
decimateOddBack scatter), its two head convs are plain 1x1s, and its GAP-and-dense
tail is ResNet-34's r34HeadB verbatim. What is new is the two stage ties, the
twenty-six-stage apex and the tie itself.
⭐⭐ The chain is stated TO THE IMAGE and its last node is ONE STEP PAST the
artifact: no render emits a gradient into %x, so MNv4's committed backward ends at
the stem conv's WEIGHT gradient, whose operand mnv4StemCotN already ties. B0 makes
the same choice at the identical stem; the file header says which.
⚠⚠ MEASURED: peeling ONE CertLayer.comp at MNv4's literal resolutions is a kernel
deterministic timeout by rfl, by simp only [.., Function.comp_apply], and with
Mathlib's Function.comp_assoc in the simp set. The same peel through a generic
rfl-at-variables lemma (certLayer_comp_fwd_apply) is 2 s for all 26 stages.
⚠ Twenty-six stages, not eighteen: 21 UIB blocks plus a fused stage, and TWO head
convs before the pool where MobileNetV2 has one and ResNet-34 none. The blocks stay
OPAQUE and there is no backward_unique step (§4.2d's measured reason); the shape
check is what replaces it, and it names every block by its table row -- which is
what pins rows 4/5/10, 12/18 and 15/19/20 apart
(MobileNetV4WholeBackCertifiedTieB.lean).

### `LeanMlir.Proofs.Foundation.ResNet50FaithfulPoCB`

⭐⭐ §3.5(c): RESNET-50's T3 — the §1 fold and the §1a tie at batch BatchNorm.
⭐⭐ ZERO new op-kind lemmas: ResNet34FaithfulPoCB's six are statements about OP
KINDS at full generality, and the bottleneck's third conv is one more instance of
the first. The fold file is an ENUMERATION of the artifact's op table by block
profile. ⛔ ResNet-50 emits NO conv bias gradient at all -- ResNet50RenderB has no
convBias flag -- so its table is six op kinds where r34's is eight, and every slot
the tie states is exercised by the bytes (161 of 161).
⭐⭐ THE LOSS COTANGENT IS A BINDER, and for this net it HAD to be: R50 ships BOTH
losses, the label-smoothed softmax chain on bce := false artifacts and BCE's
three-op chain on bce := true ones (including resnet50in160_lambaccdp8x64bce, where
the 76.66% comes from). r50_lossCot_is_smoothedCE_grad and r50_lossCot_is_bce_grad
instantiate it; neither is privileged. That is 4b's "g as a BINDER" made necessary.
⭐ The STEM and HEAD tie bundles are ResNet-34's reused verbatim, as the stem and
head forwards were in T1.
⚠ One add_comm per projection form (the render emits addVB(body, projection) where
residualProj adds proj + body); the identity block needs none.

### `LeanMlir.Proofs.Foundation.BackwardMaps`

The per-op ℝ backward maps the certified backward ties are stated about
(reluMaskBack, diagBack, the perRow* lifts, convFlatBack, maxPoolFlatBack, the
decimateBack/decimateOddBack scatters and the strided/depthwise backwards built
on them, maxPool3s2FlatBack with its VJP at a Vec point). One leaf, no float
content; moved out of the *FloatBridge files 2026-09-08 (float_second_pass.md).

### `LeanMlir.Proofs.Architectures.ChannelLNBack`

The channel-LayerNorm backward (rowLNVecFlatBack, chanLNTensor3Back) — ConvNeXt's
and ViT's LN input-VJP, the ℝ map chanLNTensor3Back_eq_chanLN_vjp is about.

### `LeanMlir.Proofs.Foundation.ResNetBackChains`

The ResNet-34/50 backward chains (r34IdBlockBack, r34DownBlockBack, r34InputGrad,
r34InputGradB, r50InputGradB) and the batched 3×3/s2 pool backward maxPool3s2FlatBackB
— the ℝ maps the three ResNet certified ties are stated about (no float content).

### `LeanMlir.Proofs.Foundation.MobileNetBackChains`

The MobileNetV2/V4 backward chains (invresBodyBackPC, invresBodyStridedBackPC,
mnv2InputGrad, mnv2InputGradB, mnv4InputGradB) — the ℝ maps the four MobileNet
certified ties are stated about (no float content).

### `LeanMlir.Proofs.Foundation.EfficientNetBackChains`

The EfficientNet-B0 backward chains (mbconvBodyBack, efficientnetInputGradB,
efficientnetInputGradB_full) — the ℝ maps the three B0 certified ties are stated
about (no float content).

### `LeanMlir.Proofs.Foundation.ConvNeXtBackChains`

The ConvNeXt-T backward chains (cnxBlockBodyBack, cnxDownBack, convnextInputGrad) —
the ℝ maps the two ConvNeXt certified ties are stated about (no float content).

### `LeanMlir.Proofs.Float.LinBackFloatBridge`

A3 backward fold: the linear input-VJP (dx = Wᵀ·dy = bias-free dense over the
transpose, reuses floatBridges_dense) + the exact ReLU-back selectPos mask
(floatBridges_reluMaskBack) compose via FloatBridges.comp into a whole-net
backward gradient bridge (mlpInputGrad_floatBridges) — the backward peer of
cifar8_floatBridges.

### `LeanMlir.Proofs.Float.ConvMixedComposeBridge`

⭐⭐ The bf16-MIXED float bridges (`planning/archive/bf16_renderer.md` §9.3, §11, §12.3), and
both are APEXES: `ConvMixedComposeBridge` transitively imports `ConvMixedFloatBridge`
and `DepthwiseMixedFloatBridge` imports `DepthwiseFloatBridge`, so these two lines
cover all four. `conv_close_mixed` is `dot_close_mixed_uniform` instantiated at
fan-in `ic·kH·kW` (a conv output IS a dot product over its flattened receptive
field); the compose bridge is what makes the mixed conv `FloatClose`, i.e. usable by
`floatClose_relu`/`_bn`/`_residualBlock`/`.comp` — the fold that says bf16 costs
1.86× the f32 certificate at R50's 53 conv layers.
⚠⚠ THESE WERE AUDITED BY `tests/AuditAxioms.lean` WITHOUT BEING ROOTS OF EITHER LIB,
which is precisely the gap `scripts/check_audit_coverage.py` exists to catch: locally
it hides behind stale dev `.olean`s and the axiom gate looks green, while a fresh CI
runner never builds the object at all. It bit at `5f27766^` and it bit again here.

### `LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie`

§B integrity tie: the r34 IDENTITY-BLOCK backward float bridge targets the CERTIFIED
VJP. Same-vocabulary (per-channel BN, non-batched) target rblkPC_has_vjp_at — built
here, mirrors resblock_has_vjp_at — + the conv-leaf tie (convFlatBack_eq_vjp_backward,
via IR.convBackDenote_eq_input_grad_formula) ⇒ r34IdBlockBack(pinned) = its .backward.
b1-free (no batched↔non-batched reconciliation).

### `LeanMlir.Proofs.Foundation.MobileNetV2WholeBackCertifiedTie`

⭐⭐ The same tie for the WHOLE MobileNetV2: mnv2InputGrad(pinned) =
(mobilenetv2PC_has_vjp_at ...).backward, plus the piece r34's file does NOT have —
mobilenetv2Forward_full_pc_eq_chain, a rfl saying the ten-stage chain the apex is
instantiated at IS the committed forward. That is the shape check §3.10's wrong
pool slipped past. ⭐ No drift found here: the mnv2 backward number is unchanged.

### `LeanMlir.Proofs.Foundation.MobileNetV2PaperWholeBackCertifiedTie`

⭐⭐ And the SAME TIE at MobileNetV2's PAPER depth — all seventeen bottlenecks
(proofs_tier_to_paper_nets 3.2c). The six-block file above is the ch7
representative; this is the net `mobilenetv2ForwardPaper` is. The blocks stay
opaque, so the composition is checked between variables and the file costs ~3 s;
what depth 17 forced is `mnv2OpaqueA0 … A17`, prefix defs for the running
activations, because the nested-application hypotheses of the 6-block statement
are quadratic in the writing. ⛔ `mobilenetv2ForwardPaper_eq_slots` cannot be a
one-step `rfl` — the kernel times out at 3 min; it peels through
`mobilenetv2ForwardPaper_eq_chain` and then unfolds the prefixes by name.

### `LeanMlir.Proofs.Foundation.EfficientNetWholeBackCertifiedTie`

⭐⭐ And for the whole EfficientNet-B0, the fourth net to get one:
efficientnetInputGradB(pinned) = (efficientnetB_has_vjp ...).backward, at every
batch size, with efficientnetForwardB_eq_chain as the shape check. It had to wait
for the XLA-SAME re-spelling — at the symmetric stem it would have certified a
program no shipped B0 artifact runs. ⭐ The b0 backward number is unchanged.

### `LeanMlir.Proofs.Foundation.EfficientNetFullWholeBackCertifiedTie`

⭐⭐ And at the PAPER depth, all 16 MBConv blocks, one step further than the
representative's: efficientnetInputGradB_full(pinned) = the generic 18-stage
apex's backward (blocks opaque), then instantiated at the concrete mb*W blocks
and carried to efficientnetForwardB_full_has_vjp by HasVJP.backward_unique —
two witnesses for one map have one backward, so the tactic-built whole-net
witness is never unfolded (the representative's file stopped at a ▸-transported
`_committed` the kernel could not reduce through). Then _correct reads it through
efficientnetForwardB_full_has_vjp_correct, whose proof is the shape check
efficientnetForwardB_full_eq_chain: the chain IS the Jacobian-transpose of the
committed nested-application forward. Every batch size; no smooth point.

### `LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTieB`

⭐⭐ T6 AT BATCH BATCH-NORM, for the two nets whose Proofs tier was per-example
(proofs_tier_to_paper_nets §4.2, the last real statement in that section's port —
T4/T5 there are float budgets and that thread is closed). r34InputGradB and
mnv2InputGradB are the reverses of resnet34ForwardB_full and
mobilenetv2ForwardB_full, the forwards the shipped trainers run; each is tied to
its net's certified whole-net VJP with the blocks OPAQUE, plus a *_eq_slots shape
check saying those stages ARE the committed forward.
⭐ ResNet-34's pool tie is `rfl` — it closes the one seam §4.2a left open (the
3x3/s2 pool backward was threaded as the emitted den and not identified with the
certified VJP) — and it is rfl only because 4.1c built batchMap_has_vjp_at field
by field and maxPool3s2Flat_has_vjp_at_vec did the same one tier down.
⭐⭐ MobileNetV2 defines NO apex and NO prefix defs: mobilenetv2PaperPC_has_vjp_at
is generic in every dimension and every stage, so the batched net instantiates the
per-example file's own 21-stage chain.
⛔ Neither goes B0's extra step (concrete blocks, then backward_unique): that is a
KERNEL deterministic timeout at six minutes here, and the reason is the KINK, not
the depth — B0's witnesses are global HasVJP and carry no point, where these are
HasVJPAt at prefixes spelled two different ways.

### `LeanMlir.Proofs.Foundation.Resnet50WholeBackCertifiedTieB`

⭐⭐ AND RESNET-50's T6 (proofs_tier_to_paper_nets §3.5(e)) — the last statement
that net was missing which says anything; (d)'s two float budgets are the vacuous
half. ⭐⭐ Almost all of it is ResNet-34's, reused rather than rewritten:
resnet50ForwardB_full is r34HeadB ∘ [3,4,6,3] bottlenecks ∘ r34StemB, so §4.2d's
two endpoint ties, its batched 3x3/s2 pool tie, its batchMapAux float lift AND
r34B_full_has_vjp_at itself (the generic 18-stage apex — [3,4,6,3] is sixteen
blocks for both nets) all apply at R50's widths. This file adds the sixteen
bottleneck slots, the tie, its pdiv reading and the shape check.
⚠ q is a BINDER, so one statement covers resnet50in_fwd (224 px) and
resnet50in160_fwd (160 px, the net the quoted 76.66% trains), every dimension is
an explicit 2 * (…) nest rather than 8 * q, and 0 < q is a real hypothesis where
ResNet-34 needed none (the stem pool's grid).
⭐ ~3 s, against ResNet-34's ~60 s for the same shape against the same apex.

### `LeanMlir.Proofs.Foundation.EvenKernelConvBack`

⛔⛔ `convFlatBack` is NOT the adjoint at an EVEN kernel: conv2d pads by pH=(kH-1)/2
and the reversed-kernel forward conv is the adjoint only when kH-1-pH = pH, i.e.
only for odd kH. ConvNeXt's 4x4/s4 patchify stem and three 2x2/s2 downsamples are
the repo's only even kernels (ViT's 16x16 patch embed does not use conv2d). The
EMITTED backward is correct — StableHLO's .convStridedBack already pads
asymmetrically [[kH-1-pH, pH]] — so nothing trained is affected; the float tier is
the third spelling of that map and never got the fix. padOdd is the repair: an
even-kernel conv IS an odd-kernel conv on the kernel zero-extended at (+1,+1), so
the existing odd leaf tie does all the work and no new float machinery is needed.

### `LeanMlir.Proofs.Foundation.ConvNeXtWholeBackCertifiedTie`

ConvNeXt-T's whole-net backward tie: the stage-boundary downsample tie and the
depth-k STAGE FOLD (planning §3.18's "one real proof"), plus the eleven named saved
activations. ⛔ The assembly is not here — see the file header and planning §3.19.

### `LeanMlir.Proofs.Foundation.ViTBackChains`

ViT-Tiny's whole-net backward, tier T6 (proofs_tier_to_paper_nets 3.4a), in three
modules. ⛔ The gap it closes is NOT the head count the audit recorded
(ViTMhsaBackCertifiedTie is general in h) — it is the LayerNorm FORM: the block tie
was at the retired SCALAR gamma/beta and the shipped vitForwardKV runs
transformerBlockV at vector [D]. Foundation/ViTBackChains names the chain
(vitBlockBackV / vitTowerBackK / vitInputGradK) with the per-token LN slots as
ConvNeXt's rowLNVecFlatBack, whose header already said it is "literally ViT's
per-token LN"; ViTVecLNBackCertifiedTie re-states the block tie there (both sublayer
decompositions and the block unfold stay rfl); ViTWholeBackCertifiedTie folds the
depth-k tower head-first and closes the apex through a TERM-mode vjp_comp chain +
HasVJP.backward_unique, since vitForwardKV_has_vjp opens with `unfold`.

### `LeanMlir.Proofs.Foundation.Resnet50BlocksCertified`

R50 phase 1 (planning/archive/next_session_pipeline_then_r50.md §3.1): the THREE bottleneck
blocks' certified VJPs. bblkPC (identity, 12 blocks), bblkPStridedPC (strided
projection, stages 2/3/4 block 0) and — the one with NO R34 analogue —
bblkPProjPC, the STRIDE-1 projection that R50's stage 1 block 0 needs because it
goes 64→256 without changing resolution. ⚠ rblkPStridedPC cannot serve there: the
halving is in its TYPE. ⚠ The stride sits on the 3×3 (v1.5/torchvision), not the
leading 1×1. Zero new foundation — every underlying lemma was already generic in
{ic oc h w kH kW}, so the third conv is one more vjp_comp_at link.

### `LeanMlir.Proofs.Architectures.DepthwiseBackCertifiedTie`

§B shared prerequisite: the DEPTHWISE adjoint gate (the depthwise twin of
IR.convBackDenote_eq_input_grad_formula) — depthwiseConv2d (dwReverse W) 0 =
depthwiseConv2d_input_grad_formula W, all dims/odd kernels, via Finset.sum_bij' on the
pad supports (no Σ co) — plus the flat + strided depthwise leaf ties (depthwiseFlatBack
= certified depthwise input-VJP). Unblocks the convnext/mnv2/enet §B ties.

### `LeanMlir.Proofs.Architectures.ConvNeXtBackCertifiedTie`

§B integrity tie (convnext): cnxBlockBodyBack(pinned LN/gelu/layerScale backs) = the
certified convNextBlockBody_has_vjp.backward — depthwise gate + 1×1 conv leaves + rfl;
plus the residual-wrapped block tie. b1-free.

### `LeanMlir.Proofs.Architectures.MobileNetV2BackCertifiedTie`

§B integrity tie (mnv2): build the per-channel-BN certified body VJP invresBodyPC_has_vjp_at
(fresh, like r34's rblkPC) then tie invresBodyBackPC (+ strided) — relu6 masks pinned to the
0<preact<6 clamp-window signs, BN backs to bnPerChannelTensor3_has_vjp, depthwise via the
gate. b1-free.

### `LeanMlir.Proofs.Architectures.EfficientNetBackCertifiedTie`

§B integrity tie (efficientnet): mbconvBodyBack(pinned bn/swish/SE backs) = the certified
mbconvBody_has_vjp.backward — SE back pinned to seBlockFull_has_vjp, swish to swish_has_vjp,
depthwise via the gate. Certified per-example body VJP already exists (global bnForward).

### `LeanMlir.Proofs.Architectures.ViTMhsaBackCertifiedTie`

§B integrity tie (vit MHSA — the sdpa adjoint): mhsaBackFlat (Q/K/V pinned to the actual
dense projections at the saved input X) = the certified mhsa_has_vjp_mat.backward,
flattened. Via ViTBackB0's mhsa_backward_collapseMH (certified Mat backward = per-head
merged sum) + the projBack_core_coord/woback_unflatten coordinate match (dense Wᵀ = mulVec,
Σk over h·dh reindexes to Σh Σj, separate projBacks regroup via sum_add_distrib).

### `LeanMlir.Proofs.Codegen.AdamStep`

The optimizer rung beyond SGD: the ℝ Adam/AdamW step mirroring
the emitted update (Phase 3a of vit_train_to_vit_verified.md).
Faithfulness target + denominator well-definedness; NO descent
claim (Adam isn't monotone).

### `LeanMlir.Proofs.Codegen.SgdMomentumStep`

The SGD / Nesterov peers of AdamStep (§2i). Same claim ceiling:
faithfulness, NOT descent — nothing here claims Nesterov descends.

### `LeanMlir.Proofs.Codegen.AdamRender`

Phase 3b: the AdamW render-close — emitted weight/bias update =
adamWScalar of the certified gradient (sgdW_isCertifiedGradStep
analogue, optimizer swapped for AdamW).

### `LeanMlir.Proofs.Foundation.ResNet34Live2`

Stage 2 of the live ResNet-34 (Item A2): the channel-order invariant
kit (maxpool/BN/ReLU preserve strict pointwise channel domination —
the non-vacuity carrier). Build-checked; not yet a live witness, so
also NOT in the AuditAxioms headline set.

### `LeanMlir.Proofs.Foundation.ResNet34LivePC`

Item A: the first NON-DEGENERATE ResNet-34 whole-net backward witness
(level 2) — 2-channel stem + maxpool + 3 strided downsamples + GAP +
dense, every smoothness hypothesis discharged, forward X ≠ forward 0
via the channel-order invariant. In the AuditAxioms headline set.

### `LeanMlir.Proofs.Training.ResNet34LiveSeal`

Item A level 3: the nonzero-Jacobian SEAL for the live ResNet-34
witness (fderiv ℝ liveFwd2 Y ≠ 0 ⇒ backward not the zero map). Sealed
at a channel-symmetric base Y via the BN channel-difference identity
(carrier vanishes ⇒ no BN-variance derivative needed). The ResNet peer
of MobileNetV2JacobianSeal. In the AuditAxioms headline set.

### `LeanMlir.Proofs.Foundation.ResNet34LiveFull`

Item A FULL DEPTH: the real [3,4,6,3] (16-block) live ResNet-34, level-3
sealed. The 13 identity blocks (zeroed body ⇒ relu(x+1)=x+1) wash out
through the downsamples' BN (bn(z+c)=bn(z)), so the full net = the
empty-chain witness + 2 and the seal reduces to ResNet34LiveSeal's.

### `LeanMlir.Proofs.Training.MobileNetV2JacobianSealFull`

MobileNetV2 FULL DEPTH: the real 17-block live MobileNetV2, level-3
sealed. 15 identity skip blocks (zeroed body ⇒ ivId a = a+3, no relu —
linear bottleneck) shift by +45; GAP + identity head pass it, so the
full net = the 2-block witness + 45 and the seal reduces to
MobileNetV2JacobianSeal's Qq / g_hasDerivAt. VJP composed through all 17.

### `LeanMlir.Proofs.Foundation.ResNet34LiveRealistic`

Item D (realistic dims): the live ResNet-34 whole-net backward at real
ImageNet 224×224 spatial resolution (the genuine 5-halving pyramid
224→112→56→28→14→7). β-parametric downsample (β=64>√1568) + stem
(β=160>√25088); every smoothness/no-tie hyp discharged at n up to 25088,
forward X≠0 (level 2). Confirms no discharge secretly used a small n.

### `LeanMlir.Proofs.Training.ResNet34LiveRealisticSeal`

Item D level 3: the nonzero-Jacobian SEAL at 224×224. A uniform channel-0
perturbation makes channel0 = channel1 + δ everywhere, so 7×7 GAP of a
uniform diff = δ and maxpool(ch0)=maxpool(ch1)+δ for ALL t (max(a+δ,b+δ)=
max(a,b)+δ) — no eventual-selection topology. UDiff invariant threaded like
Dom2; output diff = t·Rr (4 positive istds), g'(0)=Rr 0 ≠ 0.

### `LeanMlir.Proofs.Training.MobileNetV2SealRealistic`

Item D level 3 for MobileNetV2: the nonzero-Jacobian SEAL at 224×224. ReLU6
is a BOUNDED window (0,6), so unlike ResNet's β-grows route, γ is SCALED DOWN
(γ=1/128 ⇒ |γ|√n < 3 keeps bn∈(0,6) at n=2·112·112). The 1×1 weights are
dimension-independent and reused. Uniform-perturbation UDiff seal: the
asymmetric stem turns input t into channel-diff −t, each BN ×γ·istd, so the
output diff is −t·Rr (4 positive γ·istds), g'(0)=−Rr 0 ≠ 0.

### `LeanMlir.Proofs.Architectures.EfficientNetBackB0`

Backward-graph faithfulness (den-level): fan-in bricks
(residual/SE), per-op backward ops (gap/broadcast/true-batch-norm/
batched conv+depthwise), the whole per-example MBConv block, and
the batched-stage backward primitives.

### `LeanMlir.Proofs.Architectures.MobileNetV2BackB0`

MobileNetV2 backward-graph faithfulness (den-level): the batched
relu6 conv/depthwise stages (selectMid kink), the SE-less inverted-
residual body, and the whole-block capstone — the relu6 (_at)
peer of EfficientNetBackB0.

### `LeanMlir.Proofs.Foundation.ResNet34BackB0`

ResNet-34 backward-graph faithfulness (den-level): the batched
conv-bn-relu stage (selectPos one-sided kink), the basic-block
body (conv-bn ∘ conv-bn-relu), and the identity-block capstone —
relu (_at) with an OUTER post-residual relu (the extra factor
vs the MBConv/inverted-residual blocks).

### `LeanMlir.Proofs.Foundation.ResNet50BackB0`

ResNet-50 backward-graph faithfulness (den-level): the 3-conv
bottleneck body and all THREE block capstones — identity, the
stride-1 projection (stage 1 block 0, no R34 analogue) and the
strided projection. Reuses R34's/EfficientNet's batched stages
verbatim; the bottleneck's extra conv is one more vjp_comp_at link.
⚠ Resnet50BlocksCertified is the PER-CHANNEL phase 1; this is the
batched world the render actually emits, and needed its own.

### `LeanMlir.Proofs.Foundation.CertifiedChain`

⭐ Net-agnostic FOLD machinery: `CertLayer` packages a layer with its VJP,
its backward graph and the proof that the graph denotes the VJP, and
`CertLayer.comp` proves once the chaining argument every *BackB0 body
faithfulness lemma writes out by hand. Smoothness preconditions conjoin at
the right activations, so an `_at` (relu) chain threads its own hypotheses.

### `LeanMlir.Proofs.Foundation.ResNet50BackNet`

R50's blocks as CertLayers + stages + the four-stage trunk. The first
net-level backward fold in the repo; every other *BackB0 stops at a block.

### `LeanMlir.Proofs.Foundation.BackNetFolds`

The other four conv nets folded: CertLayer instances for r34, mnv2,
enet and convnext. enet/convnext are globally smooth (ok = True);
r34/mnv2 are `_at` (relu / relu6 kinks). No new proof per net.

### `LeanMlir.Proofs.Foundation.MobileNetV4BackB0`

MobileNetV4's batched UIB backward: the depthwise-bn-RELU stage (the
one stage the repo lacked — it had relu6 and swish), the four stage
CertLayers, and ⭐ the FOUR FAMILIES COLLAPSED into one body via
CertLayer.id' in the absent-depthwise slots. No case split.

### `LeanMlir.Proofs.Foundation.EfficientNetBackNet`

EfficientNet's four §8e holes closed: mbExp / mbNoExp / mbStrided /
head as CertLayer comp-chains. Every stage was already certified with a
backward graph; what was missing was the COMPOSITION.

### `LeanMlir.Proofs.Foundation.ViTBackNet`

ViT folded onto the same machinery — the LAST net onto `CertLayer` and the
only one whose fold covers stem-to-head. ⭐ `vitTrunkV_graph` proves the
generic chain reproduces ViTBackB0's hand-written depth-k tower TERM FOR
TERM, so that bespoke induction is derived rather than kept in parallel.
GELU/LayerNorm are smooth ⇒ `ok = True` at every depth (enet/convnext tier).

### `LeanMlir.Proofs.Architectures.ConvNeXtBackB0`

ConvNeXt backward-graph faithfulness (den-level): the per-example
(batch-1) peer of EfficientNetBackB0. LayerNorm is per-example
separable, so no batched machinery — the block-body backward graph
(depthwise → LN → expand → gelu → project → layerScale) + identity-skip
residual capstone, plus the LN+2×2/s2 downsample capstone. GELU is a
global VJP, so everything stays in the clean global HasVJP form.

### `LeanMlir.Proofs.Architectures.ViTBackB0`

ViT whole-block backward-graph faithfulness (den-level, heads = 1):
the per-token Mat-VJP peer of the conv nets' *BackB0 capstones. MLP +
attention sublayer backward graphs (residual fan-in + LN-back), with
the MHSA backward collapsed at heads = 1 to the plain three-way dense
fan-in over the proven sdpa_back_{Q,K,V} (tied to mhsa_has_vjp_mat by
VJP determinism), assembled into the whole transformerBlock VJP.

### `LeanMlir.Proofs.Foundation.LinearFaithfulPoC`

PoC: the mnist-linear train step proof-tied to the certified
loss-descent SGD step (the renderer `MainMnistLinearVerified`
trains on), incl. the param-grad/SGD "tail fold". Template for
making each chapter's verified trainer faithful — see
planning/archive/verified_faithful_sweep.md.

### `LeanMlir.Proofs.Float.E4M3FaithfulPoC`

E4M3 (fp8) render-tie (planning §3b): the emitted block-scaled
int-matmul graph denotes the intended dequant-first algorithm
(the per-output dequant scale factors out of the fp32 accumulate),
via existing den-faithful ops only — E4M3FaithfulPoC.lean.

### `LeanMlir.Proofs.Float.Bf16FaithfulPoC`

bf16-mixed render-tie (planning §5, the symmetric gap): the emitted
bf16-leaf/fp32-accumulate linear graph denotes the rounded-operand
linear (no scale to factor — simpler than the E4M3 twin). Unlike fp8,
this graph lowers on CUDA. Bf16FaithfulPoC.lean.

### `LeanMlir.Proofs.Foundation.MlpFaithfulPoC`

mnist-MLP peer: the whole 3-layer MLP train step folded into the
verified AST (forward + backward chain + 6 weightSgd/biasSgd), each
output's den proven = certified via mlp_render_*_certified.

### `LeanMlir.Proofs.Foundation.CnnFaithfulPoC`

mnist-CNN peer: the conv train step folded into the verified AST via
the new convWeightSgd/convBiasSgd ops (conv layers) + weightSgd/biasSgd
(dense head); each of the 10 outputs' den proven = certified via the
conv chain bridges + the M2 dense bridges (CnnFaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.CifarFaithfulPoC`

ch5-CIFAR peer (no-BN, deeper 2-scale net): reuses the cnn conv ops +
dense bridges (NO new core ops) — generic convW/convB_den cover all 4
conv layers, the 3-dense head via the M2 bridges (CifarFaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.CifarBnFaithfulPoC`

ch5-CIFAR-BN peer (per-channel BatchNorm): reuses the cnn conv ops + the
cifar dense head; the new bnGammaSgd/bnBetaSgd ops carry the per-channel
γ/β grads, den-certified via cifar_bn_render_{gamma,beta}_certified
(CifarBnFaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.CifarBnTiePoC`

ch5-CIFAR-BN §1a TIE: conv+BN tied through the real forward + the BN backward chain
(BN-output cots relu-masked for γ/β, conv cots via BN-back) — CifarBnTiePoC.lean.

### `LeanMlir.Proofs.Architectures.Cifar8FaithfulPoC`

deeper 8-conv cifar8 (no-BN): pure reuse — conv via CifarPoC generics,
dense via the new generic denseW/denseB_den (Cifar8FaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.Cifar8TiePoC`

ch5-cifar8 §1a TIE: 8-conv chain tied through the real forward — cifar's chain
repeated over 4 stages, all reused constructors (Cifar8TiePoC.lean).

### `LeanMlir.Proofs.Architectures.Cifar8BnTiePoC`

ch5-cifar8-bn §1a TIE: cifar8's chain + a BN-back at every conv; all 32 conv+BN
params tied (Cifar8BnTiePoC.lean).

### `LeanMlir.Proofs.Foundation.ResNet34FaithfulPoC`

ch6-ResNet-34 (full [3,4,6,3], 146 params): the 2 new strided-conv SGD ops
(convStrided{Weight,Bias}Sgd) for the 7×7 stem + 3×3 downsample/projection
convs den-certified via mnv2_render_stem_conv{W,b}_certified; the 142 other
params reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics (ResNet34FaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoC`

ch7-MobileNetV2 §1 fold (depthwise half): the 4 new depthwise SGD ops
(depthwise{,Strided}{Weight,Bias}Sgd) den-certified via the mnv2_render_depthwise*
bridges; expand/project/BN/dense reuse the CifarPoC/CifarBnPoC/Cifar8PoC generics
(MobileNetV2FaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.MobileNetV2FaithfulPoCPaper`

⛔ MobileNetV2Render.lean is RETIRED (4c leg 2, 2026-09-06), with
verified_mlir/mobilenetv2_train_step.mlir and mobilenetv2_reduced_train_step.mlir.
Its per-example forward chain moved into MobileNetV2RenderB, which the EVAL forward
still needs; everything else it held was about artifacts that no longer exist.
ch7-MobileNetV2 FULL 17-block paper §1 fold (den): every one of the 210 params of
mnv2TrainStepFaithfulVPaper denotes the certified step — ZERO new ops/lemmas, the
cifar8-bn lesson at full scale. Six per-block-type capstones (stem/no-exp/stride-1/
stride-2/head/dense), each delegating to the audited CifarPoC/CifarBnPoC/Cifar8PoC/
Mnv2PoC/ResNet34PoC generics (MobileNetV2FaithfulPoCPaper.lean).

### `LeanMlir.Proofs.Codegen.EfficientNetRender`

ch8-EfficientNet-B0 full-16 (262-param) train step rendered as pretty(provenGraph)
at the batched index (N=1, emit B = batch); un-fused SE for the SE param grads
(EfficientNetRender.lean); writes verified_mlir/efficientnet_train_step.mlir.

### `LeanMlir.Proofs.Architectures.EfficientNetFaithfulPoC`

ch8-EfficientNet-B0 §1 fold (den): every batched param-SGD op type denotes the
certified Σ_n batched gradient — conv/strided-stem/dense W,b + BN γ/β + depthwise
(the Σ_n batch-sum bridge = Finset.sum_congr of the per-example .correct)
(EfficientNetFaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.EfficientNetTiePoC`

ch8-EfficientNet-B0 §1a TIE (IN PROGRESS): pins each param cotangent to the actual
loss-driven backward chain. Landed: the loss-cotangent den (batched softmaxRowF − onehot);
the whole-net thread (swish/SE-gate/true-BN chain-cot constructors) is the remaining
dedicated effort (EfficientNetTiePoC.lean).

### `LeanMlir.Proofs.Architectures.ConvNeXtFaithfulPoC`

ch9-ConvNeXt-T §1 fold (started): the per-channel layer-scale γ gradient cert —
the one genuinely-new proof obligation (Vec c via the chanIdx broadcast, vs the
per-element Vec n cnx_render_lsgamma_certified); the den target of the pending
layerScaleChGammaSgd core op (ConvNeXtFaithfulPoC.lean).

### `LeanMlir.Proofs.Codegen.ConvNeXtRender`

ch9-ConvNeXt-T §1 RENDER: the full [3,3,9,3] train step rendered as pretty(provenGraph)
(fwd + bwd-cotangent chain + param-SGD via the new ops); writes
verified_mlir/convnext_train_step.mlir. 2 documented hand-written gaps (the stem 4×4/s4
+ downsample 2×2/s2 weight grads — no even/stride-4 weight-grad VJP yet) (ConvNeXtRender.lean).

### `LeanMlir.Proofs.Codegen.ConvNeXtRenderB`

ch9-ConvNeXt-T §1b BATCHED: the same chain at N := B, plus the STOCHASTIC-DEPTH
renders (18 per-block residual-branch masks) — the only ConvNeXt artifacts from
that chain, since the drop-free batched render is tied but not swapped
(ConvNeXtRenderB.lean).

### `LeanMlir.Proofs.Architectures.ConvNeXtTiePoC`

ch9-ConvNeXt-T §1a TIE: the whole [3,3,9,3] train step tied through the REAL forward —
18 blocks + 3 downsamples + GAP→LN→dense head + stem bias den-composed
forward→loss→backward (GELU masks, identity-skip fan-in, downsample LN-back); the 4
even-kernel weight grads are the documented render gap (ConvNeXtTiePoC.lean).

### `LeanMlir.Proofs.Codegen.ViTRender`

ch10-ViT-Tiny §1 RENDER: the full depth-12 train step rendered as pretty(provenGraph)
(fwd + per-head SDPA backward chain + 200-param SGD via the 6 new ops); iree-validated
(LeanMlir/Proofs/Codegen/ViTRender.lean). NO param gap — vit has the patch-weight VJP cert.

### `LeanMlir.Proofs.Codegen.ViTRenderB`

ch10-ViT-Tiny §1b BATCHED: the same forward at N := B — the last net to make the
move, and the only one where a per-EXAMPLE stochastic-depth mask is not yet
expressible. Writes no artifact; `vit-fwd-b-tie` gates it (ViTRenderB.lean).

### `LeanMlir.Proofs.Architectures.ViTFaithfulPoC`

ch10-ViT-Tiny §1 FOLD: each emitted param-SGD op den=certified — vecln γ/β, rowwise
dense W/b, patch conv W/b, pos (one-line delegations to ViTVecLN/ViTClose certs); the
head reuses Cifar8PoC.dense{W,B}_den, cls reuses denseBiasSgdB (ViTFaithfulPoC.lean).

### `LeanMlir.Proofs.Architectures.ViTTiePoC`

ch10-ViT-Tiny §1a TIE (per-block): every one of a vector-LN transformer block's 16 params,
fed the cotangent the REAL backward chain delivers (vitCot* — two residual fan-ins + the
three-way LN₁ fan-in + the SDPA backs), den=certified. Single-head representative; the
multi-head/depth-12 thread is the remaining step (mnv2 reduced→full) (ViTTiePoC.lean).

### `LeanMlir.Proofs.Certificates.LipschitzCert`

Robustness certificate (planning/archive/robustness_ladder.md): the Lipschitz-margin
certified radius (Tsuzuku et al. 2018) — if the logit map is L-Lipschitz in L2
and the margin is m, every ‖δ‖₂ < m/(√2·L) leaves the argmax fixed (proof, vs
the PGD attack's one-attack upper bound). The cert side of cert ≤ TRUE ≤ PGD.

### `LeanMlir.Proofs.Foundation.UpstreamDraft`

Mathlib upstreaming drafts (planning/mathlib_upstream_drafts/): the
PR1 generic-cdf lemmas (strictMono_cdf_iff ⟺ IsOpenPosMeasure,
continuous_cdf_iff ⟺ NoAtoms, cdf_pos/lt_one/mem_Ioo) + PR2
gaussianReal instantiations (+ symmetry, mean-shift), kept
compiling on the pin while the Mathlib PRs are in flight.

### `LeanMlir.Proofs.Certificates.SmoothingGaussian`

The real Gaussian probit (planning/archive/smoothing_gaussian_lemma.md, G1): the
smoothing radius instantiated at the TRUE standard-normal quantile —
stdNormalCDF strict-mono + symmetry, quantile MonotoneOn (0,1) + odd-about-½,
capstone smoothing_certified_radius_gaussian with only the Neyman–Pearson
(1/σ)-Lipschitz core (hg) left as a hypothesis.

### `LeanMlir.Proofs.Certificates.SmoothingMC`

The Monte-Carlo tie (the smoothing chain's LAST honest gap):
Hoeffding over the sample product measure (Mathlib subgaussian
machinery) ⇒ with prob ≥ 1−exp(−2Nt²) the reported radius
σ·Φ⁻¹(p̂−t) is genuinely certified — Cohen's CERTIFY end to end.

### `LeanMlir.Proofs.Certificates.SmoothingCP`

The exact Clopper-Pearson tie (the arithmetic CERTIFY deploys):
the count of successes over Measure.pi IS binomial (the lemma
Mathlib lacks; piFinSuccAbove induction + Pascal), the CP
lower bound covers with prob >= 1-alpha (sInf trick: no tail
monotonicity needed), composed => smoothing_cp_certified.

### `LeanMlir.Proofs.Certificates.SmoothingCPScorecard`

The smoothing CP SCORECARD (generated: scripts/smooth_scorecard_gen.py
from the fixed-protocol driver runs, run_smooth_scorecard.sh):
first-100 test images x {MNIST-MLP, MNIST-CNN, CIFAR-CNN},
sigma=0.5, n=10112, alpha=1/1000 -- 279 per-image kernel tail
checks (decide +kernel) + per-net aggregates. Light enough for
Certs (no norm_num megaterms, pure kernel bignum arithmetic).

### `LeanMlir.Proofs.Certificates.SmoothingPhiBounds`

Certified DECIMAL quantile bounds (the float-Phi^-1 gap):
upper-Riemann panels of the Gaussian density with a kernel-
computable rational pdf bound (32-term Taylor exp + pi_gt_d20
+ ceiling-rounding) => one decide-check certifies
m*h <= Phi^-1(q0); demos Phi^-1(0.9) >= 1.27, and the
scorecard MLP-img1 radius >= 1.27 in decimals.

### `LeanMlir.Proofs.Certificates.SmoothingDecScorecard`

The DECIMAL-radius scorecard (generated:
scripts/smooth_dec_scorecard_gen.py, same fixed-protocol runs
and per-image q0 as SmoothingCPScorecard): the prefix scan
phiScanRev kernel-evaluated ONCE over the whole h=1/1000 grid
(3300 panels, ~2 min), then all 279 per-image decimal radii
m/2000 <= sigma*Phi^-1(q0) are O(index) list lookups.

### `LeanMlir.Proofs.Certificates.SmoothingNetSemantics`

The NET-SEMANTICS closure (the chain's last informality):
argmaxNet classifier + measurability from logits, strict
decision regions open, stdGaussian IsOpenPosMeasure, and the
hp-interiority discharge from per-class strict-argmax
witnesses => smoothing_cp_certified_net (CERTIFY for a
concrete net's argmax, no abstract-classifier hypotheses).

### `LeanMlir.Proofs.Certificates.SmoothingNetWitness`

...INSTANTIATED (generated: scripts/smoothing_net_witness_gen.py)
for mlpT, the trained /128 pooled-MNIST MLP: ten in-kernel
strict-argmax witnesses discharge hp; capstone
smoothing_cp_certified_mlpT + deployed-scale demo.

### `LeanMlir.Proofs.Foundation.MuonGeometry`

Muon geometry (planning/archive/muon_geometry.md): the optimizer as steepest descent under
a norm. SGD = Euclidean (Cauchy-Schwarz), sign/Adam = L∞→L¹, Muon = operator→nuclear
with the polar factor UVᵀ realizing the nuclear norm (achievability, given an SVD).

### `LeanMlir.Proofs.Foundation.MuonNewtonSchulz`

Newton–Schulz convergence (planning/archive/muon_ns_convergence.md): the Muon matmul iteration
aX + b(XXᵀ)X + c(XXᵀ)²X actually COMPUTES the polar factor UVᵀ. P1 = the spectral-step
lemma: a step is the scalar map φ(t)=at+bt³+ct⁵ applied per singular value (U,V carried
through), so matrix convergence to UVᵀ reduces to scalar convergence φ^[k](σᵢ)→1.

### `LeanMlir.Proofs.Certificates.LipschitzCertInstance`

The robustness certificate INSTANTIATED (the 2026-07 audit's #1 gap):
certified Frobenius Lipschitz constants (denseE_lipschitzL2 — ‖W‖₂ ≤ ‖W‖_F,
no power-iteration estimate in the trust path), the hand-picked linear +
dense→ReLU→dense demos, and the TRAINED tier: a /128-rationalized 49→8→10
pooled-MNIST MLP with in-kernel margin and provably positive certified radius.

### `LeanMlir.Proofs.Training.TrainedMlpWitness`

The trained-weight whole-net VJP witness (MLP rung): the same trained
/128-rationalized net instantiates HasVJPAt at a REAL input — ReLU
smoothness inherited from training (exact nonzero pre-activations),
not engineered; level-3 sealed via an explicit Jacobian entry.

### `LeanMlir.Proofs.Certificates.LipschitzCertScorecard`

Certified-accuracy scorecard (post_audit_roadmap §1): the one-input
certificate scaled to a dataset-level claim — over the FIXED first-100
MNIST test subset at FIXED ε = 1/10 (pooled L2), 34/100 certified on a
spectrally-capped (σ≤4 projected-SGD) /256 net vs 1/100 on the
unconstrained net; per-image in-kernel margins, honest lower-bound
aggregate. Same theorem, same ε — training decides if the cert bites.

### `LeanMlir.Proofs.Certificates.LipschitzCertPairSDP`

Per-pair LipSDP certificates (the tighter-Lipschitz-constant pass):
LipSDP-Neuron (Fazlyab 2019) for one hidden layer, PSD witnessed by
exact rational LDLᵀ (kernel-checkable, no √, no eigensolver) — lifts
the SAME scorecard (same nets, same images, same ε) from 34→69/100
capped and 1→63/100 unconstrained; PGD bracket 72/69, sandwich
nearly closed. Core lemmas + the two generated instance files.

### `LeanMlir.Proofs.Foundation.ListDot`

The kernel-dotZ list engine + IBP interval-soundness cores: the
small, reusable halves of the full-input scorecard work. The
GENERATED full-input instance files (30k+ lines of weight/image
data each) live in the separate `CertsHeavy` lib below — they
OOM'd/priced out the shared 4-core runners, and heavy corpus
tails must not break the core workflows (certs.yml + blueprint
both build `Certs`).

### `LeanMlir.Proofs.Foundation.IntervalBoundConv`

IBP past the two-layer dense wall: a COMPOSITIONAL interval engine
(`BoxSound`/`BoxSound3`/`BoxSound3V` + `.comp`, so depth is just `∘`)
with conv2d / maxPool2 / dense / relu transformers proved sound, the
conv uniform-box collapse, and tensor- and flat-space capstones.
`ibp2_certified_at_eps` only ever covered `dense ∘ relu ∘ dense`;
this is what lets a certificate reach a convolution at all. Engine
only (no generated data) — the instance lives in `CertsHeavy`.

### `LeanMlir.Proofs.Foundation.CrownBound`

CROWN: never concretize in the middle. Carry a LINEAR FUNCTION OF
THE INPUT backward (each unstable ReLU relaxed by a linear
envelope) and concretize ONCE, so the cancellation between rows of
W1 survives in the composite row `A` instead of dying to IBP's
per-row ‖·‖₁. Certified on the MARGIN (`certified_of_marginPos`),
not on a logit box: separating two independently-bounded logits
discards exactly the correlation this buys. The upper envelope
takes any `s` with `u ≤ s*(u-l)`, so the slope may be ROUNDED to a
/2^k grid — measured k=8 costs zero images, which is what keeps
the rationals at weight scale (planning/archive/crown_ibp.md §5.5).
Engine only (no generated data).

### `LeanMlir.Proofs.Float.Binary32Instance`

The binary32/fp8-E4M3 hardware models, CONSTRUCTED (post_audit_roadmap §2):
rndP p = round-to-nearest on the unbounded-exponent p-bit grid, standard
model |rndP p x − x| ≤ 2⁻¹⁻ᵖ|x| PROVED (rndP_err) — the former
ieeeRnd/ieeeRnd_err axioms discharged, so the concrete argmax-preservation
and binary32-SGD-descent capstones now live in the ordinary zero-axiom
closure (this was the quarantined TrustedBridge lib, no longer needed).

### `LeanMlir.Proofs.Training.TrainedLinearDescent`

Descent at TRAINED weights (post_audit_roadmap §3): one binary32 SGD
step on the trained /128 pooled-MNIST linear classifier provably
decreases the real CE loss — the misclassified-witness trick makes the
descent window rational-checkable with zero exp evaluations (z_lbl ≤
z_pred exact ⇒ softmax_lbl ≤ 1/2 ⇒ Σ|∇| ≤ 2Σx, Σ∇² ≥ Σx²/4). Retires
the W=0 degeneracy caveat of binary32_linear_sgd_descends_concrete.

### `LeanMlir.Proofs.Training.TrainedCnnWitness`

Trained-weight whole-net VJP witness, CNN rung (post_audit gap #3):
the Chapter-3 mnistCnnNoBn conditional whole-net VJP instantiated at
TRAINED /128-rationalized weights + a REAL test digit — all five
smoothness hypotheses (conv1/conv2 ReLU kinks, maxpool no-tie,
dense3/dense4 kinks) discharged by exact in-kernel rationals. The
no-tie condition is trained in (pool-tie margin regularizer), the
h_mp analogue of the scorecard's spectral cap.

### `LeanMlir.Proofs.Training.TrainedCnnSeal`

Level-3 seal for the trained CNN witness: one whole-net Jacobian
entry (∂logit₇/∂pixel(0,2) = −326103939411/2³⁵ ≈ −9.49) computed in
closed form — pdiv_comp peeling with exact backward-cotangent
tables, the max-pool argmax routing decided per position, the conv
input-VJPs via conv2d_input_grad_formula through HasVJPAt.correct.
Yields backward_nontrivial / jacobian_nonzero / not_constant, the
full TrainedMlpWitness theorem set at the conv rung.

### `LeanMlir.Proofs.Certificates.LipschitzCertFloat`

The robustness certificate composed with the float bridge (2026-07
audit gap #1): the scorecard's per-image Tsuzuku certificates ×
the 2-layer FloatBridge budget (γ-form, B ≤ 5.96e-3 at the capped
net's exact magnitudes, input quantization included) ⇒ 33/34
ℝ-certified images are certified for the FLOAT-EVALUATED net,
∀ rounding models at binary32 accuracy (M.u ≤ u32).

### `LeanMlir.Proofs.Foundation.SpecVJP`

Spec→math ties (rungs B/C/E): the one proof file that imports the
trainer side (VerifiedNets), so the `denote spec.layers = <proven
forward> := rfl` ties break when a spec drifts. It sat OUTSIDE
every build target and silently rotted when mobilenetv2Verified
was promoted 6→17 blocks (fixed 2026-07-07: mnv2 keeps the
representative 6-block rung AND gains the full-paper 17-block
B/C/E tie, denoteMobilenetPaper) — a root here so CI
re-elaborates it.

### `LeanMlir.Proofs.Foundation.MlpCanonical`

The canonical-MLP surface (784→512→512→10): the generic MLP chain
instantiated at the ch2 reference dims (see MlpCanonical.lean).

### `LeanMlir.Proofs.Codegen.ResNet34RenderB`

The two batched (`N := B`) AdamW renderers. NOTHING imports
either — they are leaf `#eval` writers — so the "Certs subsumes
Proofs" claim above was false for exactly these two and
`lake build Certs` never produced their oleans, failing any job
that needs them with "object file ... does not exist". They are
also the SOLE writers of the artifacts
`resnet34-verified-adam{,-xla}` and `mobilenetv2-verified-adam`
actually train on, so their oleans must exist wherever the
corpus is built. Guarded by scripts/check_render_coverage.py.

### `LeanMlir.Proofs.Codegen.MobileNetV4RenderB`

MNv4 joined 2026-08-09: three artifacts, zero importers — exactly the leaf-writer
shape `scripts/check_render_coverage.py` exists to catch, and it did.

## `lean_lib «CertsHeavy»`

### `LeanMlir.Proofs.Certificates.LipschitzCertScorecardIBP`

The per-pair LipSDP files (LipschitzCertScorecardSDPFull{,Uncon})
are DISABLED here for now: their linarith PSD witnesses carry
~230-digit LDLᵀ fractions and OOM every free-tier runner config
(4 attempts, incl. 1-thread + 10G swap). They remain in the repo,
kernel-verified locally (93/100 @ ε=0.1 = PGD, sandwich closed);
re-enable path: planning/archive/certs_heavy_psd_memory.md (small-
coefficient DD-split witnesses, or a self-hosted runner).
2026-07-25: capping the per-image exhibits (30,884 → 7,241 lines)
did NOT change this — re-measured 16.0/16.6 GB at 1 thread, i.e.
the peak is ONE `hS*` goal, not the file size. Still out.

### `LeanMlir.Proofs.Certificates.LipschitzCertScorecardCrown`

The CROWN instance (engine: Proofs.Foundation.CrownBound, in
`Certs`) on the SAME nets/subset/ε grid as the IBP tier above, so
it is a new COLUMN in that table rather than a new experiment:
93/93/92/81 (capped) and 94/92/76/15 (unconstrained) per 100,
against IBP's 92/88/69/24 and 87/42/2/0. Imports the IBP files to
reuse their committed hpre/absr data; the only new kernel facts
are one `absSumZ (combZ …)` per (image, class) — the kernel FORMS
the CROWN row from the committed weight rows, so A's 784 entries
are never emitted. Generated by scripts/crown_ibp_scorecard.py.

### `LeanMlir.Proofs.Certificates.IbpConvScorecard`

The CONVOLUTIONAL IBP instance (engine:
Proofs.Foundation.IntervalBoundConv, in `Certs`): conv → relu →
max-pool → dense head at k/256 trained weights, per-image pixel-L∞
certificates on the first 40 MNIST test images. Generated by
scripts/ibp_conv_scorecard.py; data-heavy (one propagated box per
image, ~38 GB peak elaboration), so it lives in the heavy tier with
the other generated corpora.

