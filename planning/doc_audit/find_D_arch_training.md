# Slice D — `LeanMlir/Proofs/Architectures/` + `LeanMlir/Proofs/Training/` (HEAD 290187b1)

**Coverage.** Read in full: ChannelLNBack, DepthwiseBackCertifiedTie, ConvBackCertifiedTie, ConvGrad,
PerChannelBNGrad, EvenKernelConvBack, Residual, Softmax, ChannelLN, ConvIndex, SE, TokenParamGrad,
StridedConv, MaxPool3s2, LayerNorm, PerChannelBN, Depthwise, BatchNorm; Optim/{AdamStep, SgdMomentumStep,
RmsPropStep, Lamb, GradClip}, JacobianSeal, SgdDescent, SgdDescentCifar, TrainedMlpWitness, DropPath.
Every docstring + full signature (proof bodies skimmed): CNN, Attention, SgdDescentLinear, SgdDescentMlp,
SgdDescentCnn, BatchSealKit. Generated (TrainedLinearDescent, TrainedCnnWitness, TrainedCnnSeal): judged
via `scripts/certs/{trained_linear_descent,trained_cnn_witness,trained_cnn_seal}.py` emitted text + output.
**Trust-escape check:** `grep sorry|^axiom |admit|native_decide|implemented_by|@[extern` over both trees —
zero hits (only prose mentions of "no sorry"). The "no sorry" docstring claims (SE, Attention, Residual)
hold; "3-axiom clean" claims (StridedConv, PerChannelBN, Depthwise, TokenParamGrad) are consistent with
that but were not re-run (`#print axioms` not executed — read-only brief).

Ranked most misleading first.

---

### LeanMlir/Proofs/Training/SgdDescentCnn.lean:20 (module) + :2312 `cnn_conv2_sgd_descends` + Training/SgdDescentCifar.lean:12,100 — pool margin on the POST-ReLU tensor

**Kind:** overclaim
**Says:** module: "`MaxPool2Smooth` (pairwise-distinct window cells) is the qualitative off-the-kink condition; descent needs its quantitative form `MaxPool2MarginQ δ`"; "EVERY parameter of the Chapter-3 CNN … now has a proven descent statement." SgdDescentCifar: "descent at the LAST conv layer reaches CIFAR-8 **for free**, with the SAME non-vacuous admissible `lr` as MNIST"; docstring of `cifar8_lastConv_sgd_descends`: "the admissible `lr` is the same non-vacuous MNIST regime."
**Actually states:** every CNN/CIFAR descent capstone carries `hmq : MaxPool2MarginQ (a * stepRadius …) (Tensor3.unflatten (relu … (conv2d W₂ b₂ x₁)))` — all four cells of every 2×2 window of the **post-ReLU** tensor must differ pairwise by more than `2δ ≥ 0`. Any window with two non-positive pre-activations has two post-ReLU zeros (`|0−0| = 0`), so the hypothesis is false there; with 32 channels × 14² windows at MNIST shape this is essentially every trained point. `TrainedCnnWitness`'s own header admits it ("ReLU zeros collide, so this needs ≤ 1 negative conv2 pre-activation per window — trained in via a pool-tie margin regularizer"). Pairwise-distinctness is also strictly stronger than off-the-kink (only a tie *at the max* is a kink). Nothing proves the MNIST `lr` regime non-vacuous.
**Fix:** module: "`MaxPool2Smooth` (pairwise-distinct window cells) is a sufficient, not necessary, off-the-kink condition; on the post-ReLU tensor it additionally requires at most one non-positive pre-activation per 2×2 window, which ordinary training does not produce." Replace "EVERY parameter … has a proven descent statement" with "every parameter has a single-layer, single-example descent statement conditional on these margins". SgdDescentCifar: drop "non-vacuous"; say "the same hypotheses as the MNIST lemma, at the frozen features `x₁`, including the post-ReLU pool margin."

### LeanMlir/Proofs/Training/SgdDescentLinear.lean:308 `linear_float_sgd_descends`, SgdDescentMlp.lean:794/1005/1367, SgdDescentCnn.lean:2454/4445/5722/6150, generator scripts/certs/trained_linear_descent.py:107–136,332 — "one binary32 SGD step"

**Kind:** overclaim
**Says:** "**One binary32 SGD step on the MNIST-linear classifier provably decreases the cross-entropy loss — with NO abstract gradient-accuracy parameter.**"; `linearFloatGrad` / `mlpHiddenFloatGrad` / `cnnConv2FloatGrad`: "exactly as the rendered trainer computes it"; generated `TrainedLinearDescent`: "one binary32 SGD step on a TRAINED … classifier … provably decreases the real cross-entropy loss … the whole statement is axiom-free."
**Actually states:** the update is real arithmetic — conclusion is `crossEntropy … (Mat.flatten W - lr • M.linearFloatGrad …) ≤ …` with `-`/`•` over ℝ; only the *gradient* is float-modelled. The loss is one example `(x, label)`, while the rendered trainers are batch-mean (B = 128) XLA graphs. `hδ` (logit drift) and `hfexp` remain hypotheses; the generated concrete instance sets `fexp := Real.exp`, `eexp := 0`, i.e. an exact exponential. `FloatModel.*FloatGrad` is a FloatModel expression, not a proven denotation of any rendered trainer.
**Fix:** "One SGD step whose gradient is the FloatModel binary32 gradient (update applied in ℝ, single example) provably decreases that example's cross-entropy loss; the gradient's accuracy is proven, not assumed." Replace "exactly as the rendered trainer computes it" with "the FloatModel transcription of the per-example gradient". Generator: add "with exact `exp` (`eexp = 0`) and the update taken in ℝ."

### LeanMlir/Proofs/Architectures/Attention.lean:2057 `vitFull` / :2095 `vitFullHasVJP` / :2194 `vitFullHasVJP_correct`, :1505 (stacking note)

**Kind:** overclaim
**Says:** "**vitFull** — full ViT forward from flattened image pixels to logits"; "**vitFull VJP — the grand finale.**"; "the full ViT's backward equals the `pdiv`-contracted Jacobian"; stacking note: "the theorem generalizes trivially to per-block parameters … mechanical once the single-shared-param case is proved."
**Actually states:** `vitFull` runs `vitBody kBlocks …` = `transformerTower`, which is `Nat.rec` over ONE shared parameter tuple `(Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)` for every block, with LayerNorm `layerNormForward … (γ β : ℝ)` — scalar affine, not the per-channel `[D]` LN the ViT artifacts use. The per-block generalization is not proved anywhere in this file.
**Fix:** "`vitFull` — a weight-tied ViT (all `kBlocks` blocks share one parameter tuple, scalar-affine LayerNorm) from pixels to logits." Stacking note: "Per-block parameters are not formalized here."

### LeanMlir/Proofs/Architectures/CNN.lean:887 `MaxPool2Smooth`; :1314 capstone note (`cnnHasVJPAt`)

**Kind:** overclaim
**Says:** "every 2×2 window of `x` has pairwise-distinct values (so a unique strict argmax). The natural domain on which `maxPool2` is differentiable." Capstone: "The bundled smoothness hypotheses (`h_stem`, `h_mp`, …) are the family of every ReLU + max-pool site's smooth-point condition."
**Actually states:** pairwise distinctness is sufficient, not necessary (differentiability only needs a unique maximiser). `cnnHasVJPAt_correct`'s `h_mp` is `MaxPool2Smooth (Tensor3.unflatten (cbr … x))` on the post-ReLU stem output, which fails whenever two cells of one window are ≤ 0 pre-ReLU.
**Fix:** "A sufficient condition for differentiability of `maxPool2` (stronger than a unique maximiser). After a ReLU it also forbids two dead cells in one window."

### LeanMlir/Proofs/Architectures/ConvGrad.lean:27,43,67,80 — conv "render" closes

**Kind:** overclaim
**Says:** module: the conv bias/weight gradients "(the transpose-trick …) are the certified Jacobian"; `cnn_render_convW_certified`: "`Wⁿ = W − lr·(transpose-trick kernel grad)` denotes, at the flattened kernel, `W − lr·(certified ∂conv/∂kernel · cotangent)`"; section comment: "each rendered conv SGD output equals `θ − lr·(certified conv Jacobian · …)`".
**Actually states:** `conv_weight_grad_bridge` / `conv_bias_grad_bridge` are literally `(conv2dWeightGradHasVJP b x).correct v c idx`; the `cnn_render_*_certified` theorems are those rewrites under `v idx - lr * _`. No rendered text, `den`, or transpose-trick formula appears in any statement (prior-audit group 2).
**Fix:** "`conv_weight_grad_bridge` restates the `.correct` field of `conv2dWeightGradHasVJP`; the link from the rendered `convWGrad` op to this witness is made by that op's `den`, not here." Retitle the two `*_render_*` theorems "SGD form of the bridge".

### LeanMlir/Proofs/Architectures/LayerNorm.lean:94–116 ("Why this isn't a new chapter"); Attention.lean:1709

**Kind:** overclaim
**Says:** "**InstanceNorm** (which is what the ResNet code actually uses) … All four normalization variants share one `HasVJP` instance." Attention: "**Rank-1 correction to diagonal** (softmax, BN, LN, IN, GN)".
**Actually states:** only `bnHasVJP` exists (LN = BN by `rfl`). RMSNorm is a different function with no VJP here; GroupNorm/InstanceNorm are not formalized. The ResNet trainers use batch BN (`bnBatchTensor4`, `PerChannelBN.lean`), not InstanceNorm.
**Fix:** "BN and LN share one `HasVJP` (`layerNormForward` is `bnForward`). RMSNorm, GroupNorm and InstanceNorm would need their own proofs and are not formalized here."

### LeanMlir/Proofs/Architectures/Residual.lean:49 `residualHasVJP`

**Kind:** overclaim
**Says:** "the gradient floor is `dy` itself, so it can never get smaller than the loss gradient at this layer."
**Actually states:** `backward = f.back(x, dy) + dy`; the sum can cancel (e.g. `f.back = −dy` gives 0). No lower bound is proved.
**Fix:** "The skip contributes `dy` unchanged, so the gradient does not have to pass through `f`'s Jacobian to reach the input."

---

### LeanMlir/Proofs/Architectures/BatchNorm.lean:73 `bnMeanSq`, :102 `bnMeanSq_shard`, :359 `bnSyncGradInput`; PerChannelBN.lean:472 `bnSyncTensor4`

**Kind:** stale
**Says:** `bnMeanSq`: "The quantity a SYNCHRONISED BatchNorm reduces across replicas — never the variance … `R` replicas exchange `μ` and `E[x²]`". `bnMeanSq_shard`: "which is the whole reason sync-BN reduces `E[x²]` rather than the variance. ⛔ The variance has NO such lemma, and cannot". `bnSyncTensor4`: "It takes the SECOND MOMENT … That is not a convenience: `E[x²]` … survives an `allReduceMeanF`, whereas the variance … does not." `bnSyncGradInput`: "one collective per direction suffices."
**Actually states:** the emitted sync-BN forward now exchanges `[μ ‖ σ²]` in two rounds via Chan's parallel variance (`StableHLO.lean:1143–1160`: `bnBatchMeanB` → all-reduce, `bnBatchVarAtB x μ` → all-reduce, `bnPackB`). The comment there says the `E[x²]` exchange was the "first cut" and was removed. `bnVar_shard_chan` (same file, :145) is exactly the variance shard lemma these docstrings say "cannot" exist. `bnSyncTensor4` keeps an `m2` argument only because `den` feeds it `σ² + μ²`.
**Fix:** `bnMeanSq`: "Second moment. `bnSyncTensor4`'s statistics argument; the emitted sync op supplies it as `σ² + μ²` from a Chan exchange (`bnVar_shard_chan`)." Drop the "never the variance / cannot" sentences. `bnSyncTensor4`: "Stated at `μ` and the second moment `m2`; the render supplies `m2 := σ² + μ²`." `bnSyncGradInput`: drop "one collective per direction suffices."

### LeanMlir/Proofs/Architectures/MaxPool3s2.lean:10–22 (module), :176 `maxPool3s2Smooth_of_injective`, :44

**Kind:** stale
**Says:** "⚠ The repo's JAX references emit `reduce_window(…, 'SAME')` … Paper-faithfulness is the goal, so … the JAX `max_pool2d` helper moves to match. ⚠ Until both land and are re-run, verified and JAX disagree at the stem pool." `maxPool3s2Smooth_of_injective`: "No whole-net witness discharges it yet: the R34 Live/Seal witnesses pool 2×2 and use the `MnistCNN` lemma." Also "`maxPool2`'s `codegen_matches_canonical`".
**Actually states:** `jax/Jax/Codegen.lean:668–685` `max_pool2d` already pads symmetrically `(p, p)`. `BatchSealKit.ctConv_pool_smooth` (Training/BatchSealKit.lean:800) discharges `MaxPool3s2Smooth` through this lemma for the full-width seals. The theorem is `maxPool2_codegen_matches_canonical` (CNN.lean:1144).
**Fix:** "The JAX reference's `max_pool2d` uses the same symmetric padding." Replace the "No whole-net witness …" sentence with "Used by `BatchSeal.ctConv_pool_smooth` for the full-width seals." Rename the reference to `maxPool2_codegen_matches_canonical`.

### LeanMlir/Proofs/Architectures/Attention.lean:665,721,986,1691 — pre-rename `sdpa_back_*` names

**Kind:** stale
**Says:** "Packages `sdpa_back_{Q, K, V}_correct` into a single `HasVJPMat3`"; "(we already proved `sdpa_back_{Q,K,V}_correct`)"; "column-stacks the three `sdpa_back_*` outputs"; "Scaled dot-product attention backwards `sdpa_back_{Q,K,V}`".
**Actually states:** the declarations are `sdpaBackQ/K/V` and `sdpaBackQ_correct`/`sdpaBackK_correct`/`sdpaBackV_correct`. No `sdpa_back_*` declaration exists. (Lead outside this slice: the same stale spelling appears in Nets/ViT/{ViTMultiHeadChain, ViTBackB0, ViTChainClose, ViTMhsaBackCertifiedTie}.lean.)
**Fix:** replace with `sdpaBackQ_correct` / `sdpaBackK_correct` / `sdpaBackV_correct` (and `sdpaBackQ/K/V` for the maps).

### LeanMlir/Proofs/Architectures/Attention.lean:1588–1607, :1679–1694 — ViT boundary and "what we've proved"

**Kind:** stale
**Says:** "The patch-embedding and classifier-head steps … don't fit in the uniform `HasVJPMat` frame. We mark them as future work; closing this would require a unified rank-polymorphic VJP framework." "Softmax cross-entropy loss gradient (`MLP.lean`)"; "Standalone softmax VJP (this file)".
**Actually states:** the same file proves `patchEmbedFlatHasVJP`, `classifierFlatHasVJP` and `vitFullHasVJP` (via `HasVJPMat.toHasVJP`). `softmaxHasVJP` and `softmaxCE_grad` are in `Softmax.lean`.
**Fix:** "The boundary pieces are bridged below with `HasVJPMat.toHasVJP` (§ Bridging ranks)." "Softmax VJP and softmax-CE gradient (`Softmax.lean`)."

### LeanMlir/Proofs/Training/SgdDescentMlp.lean:806–811 `mlp_output_float_sgd_descends`

**Kind:** stale
**Says:** "The hidden/input rungs (`mlp_{hidden,input}_sgd_descends`) still take an abstract `η` … needs a per-layer float-backward grad-close (a `mlp_w{1,0}_grad_close`) … left open."
**Actually states:** `mlp_w1_grad_close`, `mlp_w0_grad_close`, `mlp_hidden_float_sgd_descends` and `mlp_input_float_sgd_descends` are all proved later in the same file.
**Fix:** delete the paragraph, or write "The hidden and input rungs are `mlp_hidden_float_sgd_descends` / `mlp_input_float_sgd_descends`."

### LeanMlir/Proofs/Training/SgdDescent.lean:25–29 (module)

**Kind:** stale
**Says:** "Discharging its hypotheses for the concrete MNIST nets (actual Lipschitz constants for the MLP loss) is future work".
**Actually states:** `SgdDescentLinear`, `SgdDescentMlp` and `SgdDescentCnn` discharge the smoothness hypothesis per layer, and `TrainedLinearDescent` gives a concrete instance.
**Fix:** "The per-net discharges are in `SgdDescentLinear`, `SgdDescentMlp` and `SgdDescentCnn`."

### LeanMlir/Proofs/Architectures/LayerNorm.lean:57 `layerNormForward`; :238–257 taxonomy

**Kind:** stale
**Says:** "in LN, `gamma` and `beta` are per-feature (not per-channel), so they're full vectors." Taxonomy: "GELU | `Phi(x_i) + x_i * phi(x_i)`"; "For the book, we show the template once (ReLU, in `MLP.lean`) and assert that GELU follows the same pattern."
**Actually states:** `layerNormForward (n) (ε) (γ β : ℝ)` takes scalars (the vector-affine form is `layerNormVec`). `geluScalar` is the tanh approximation, whose derivative is `geluScalarDeriv_eq`, not `Φ + xφ`. `geluHasVJP` is proved in this file, not asserted.
**Fix:** "`γ`, `β` are scalars here; the per-feature `[D]` affine is `layerNormVec` below." Taxonomy row: "GELU (tanh approx.) | `geluScalarDeriv_eq`". Replace "assert that GELU follows" with "`geluHasVJP` / `swishHasVJP` instantiate it."

### LeanMlir/Proofs/Architectures/SE.lean:50–60 (module), :111–145 ("What's actually inside `gate`")

**Kind:** stale
**Says:** "All foundational definitions and proofs live in `Tensor.lean` … we sketch the concrete gate from MobileNetV3 in a final commentary section"; "you'd need to add `globalAvgPoolHasVJP` (linear, easy) and `broadcastHasVJP` … That's a few hours of mechanical work."
**Actually states:** the file defines and proves `sigmoidHasVJP`, `broadcastFlatHasVJP`, `seGate`, `seGateHasVJP`, `seBlockFull`, `seBlockFullHasVJP`. `globalAvgPoolFlatHasVJP` is in CNN.lean.
**Fix:** module: "…plus the concrete gate `seGate` (GAP → dense → swish → dense → sigmoid → broadcast) and its VJP `seGateHasVJP`." Delete the "you'd need to add … few hours" paragraph.

### LeanMlir/Proofs/Architectures/CNN.lean:1251–1280 (summary), :749 `maxPool2HasVJP3`, :1158 `maxPool2HasVJPAt3`, :1200 walkthrough; Residual.lean:151

**Kind:** stale
**Says:** summary: "`conv2d`, `maxPool2` — forward operations (black-box forward)"; "`maxPool2HasVJP3` — input-path VJP for maxPool2 (argmax-routing subgradient convention)"; "`conv2dInputGradFormula`, `conv2dBiasGradFormula` — … (numerically verified to equal the VJP's backward)". `maxPool2HasVJPAt3`: "(future) `cnnHasVJPAt3`". Walkthrough: "a uniform `HasVJP`-style composition would need a type family. … we instead trace the backward pass step-by-step". Residual: "the end-to-end CNN VJP (`cnnHasVJPAt`, future)".
**Actually states:** both forwards are concrete defs. `maxPool2HasVJP3`'s backward is the canonical pdiv3 witness, zero at ties (its own docstring). `conv2dInputGradFormula` is the backward by definition, and no theorem equates `conv2dBiasGradFormula` to anything. `cnnHasVJPAt` / `cnnHasVJPAt_correct` exist in CNN.lean (:1314 ff.), composed in flat `Vec` space.
**Fix:** summary: "concrete forwards"; "canonical (pdiv-derived, zero-at-tie) witness"; "`conv2dBiasGradFormula` — the closed form, not linked by a theorem". Replace "(future) `cnnHasVJPAt3`" and "(`cnnHasVJPAt`, future)" with `cnnHasVJPAt`. Walkthrough: "`cnnHasVJPAt` below composes this in flat `Vec` space; this trace is the per-layer reading."

### LeanMlir/Proofs/Architectures/Depthwise.lean:22–31 (module), :461 `depthwiseConv2dBiasGradFormula`

**Kind:** stale
**Says:** "So we don't re-derive the VJPs from scratch. We state them as the 'channel-restricted' versions of `conv2dInputGrad` / `conv2dWeightGrad` from `CNN.lean`." Bias formula: "(documented, numerically verified, expected to equal `depthwiseConv2dBiasGrad` up to fp precision)".
**Actually states:** `depthwiseHasVJP3`, `depthwiseWeightGradHasVJP3` and `depthwiseBiasGradHasVJP` are proved directly by `pdiv_of_affine`, not derived from the regular-conv VJPs. The bias formula is an ℝ definition with no theorem relating it to `depthwiseConv2dBiasGrad`, so "up to fp precision" does not apply.
**Fix:** "The VJPs are proved directly (`pdiv_of_affine`); they have the same shape as the regular-conv ones with the `Σ c` removed." Bias formula: "Closed form `db[c] = Σ dy[c,·,·]`; no theorem ties it to `depthwiseConv2dBiasGrad` (the backward of `depthwiseBiasGradHasVJP` is this sum by construction)." Same fix for CNN.lean:715 `conv2dBiasGradFormula`.

### LeanMlir/Proofs/Architectures/ConvGrad.lean:5

**Kind:** stale
**Says:** "The MNIST CNN (`conv → relu → maxpool → conv → relu → maxpool → dense → … → dense`)".
**Actually states:** `cnnTrainStepFaithfulV` (Codegen/CnnRender.lean:32) and `mnistCnnNoBnForward` have one pool: conv → relu → conv → relu → maxpool → dense → relu → dense → relu → dense (the file's own section comment at :57 says this).
**Fix:** use the one-pool chain.

### LeanMlir/Proofs/Architectures/BatchNorm.lean:401–417, :284–289, :789

**Kind:** stale
**Says:** parameter-gradient note: "`bnGradGamma` and `bnGradBeta` … don't fit our `pdiv` / `HasVJP` framework cleanly … We state these as the *definitions*". Derivation: "`∂σ²/∂xᵢ = (2/N) · (xᵢ − μ) · (1 − 1/N) ≈ (2/N) · (xᵢ − μ)`". `bnForward_abs_sub_le`: "`bnForward_lb`'s symmetric form".
**Actually states:** `PerChannelBNGrad.bnPerChannelGradGamma_correct` / `…Beta_correct` prove the per-channel γ/β gradients against `pdiv`. `bnVarDeriv_basisVec` proves `∂σ²/∂xᵢ = 2(xᵢ − μ)/n` exactly, with no `(1 − 1/N)` factor and no approximation. `bnForward_lb` does not exist.
**Fix:** note: "Their `pdiv` correctness is `bnPerChannelGradGamma_correct`/`…Beta_correct` (`PerChannelBNGrad`)." Derivation: "`∂σ²/∂xᵢ = (2/N)(xᵢ − μ)` exactly (`bnVarDeriv_basisVec`; the `−1/N` term cancels against `Σ(xₖ − μ) = 0`)." Drop the `bnForward_lb` clause.

### LeanMlir/Proofs/Training/Optim/AdamStep.lean:8–9 (module)

**Kind:** stale
**Says:** "so the later faithfulness theorem (`den (adamGraph) = adamWStep …`) is a structural match."
**Actually states:** no `adamGraph` exists. The faithfulness theorem is `adamW_triple_faithful` (cited in Codegen/CnnRender.lean:352, ResNet34RenderB.lean:529).
**Fix:** "so `adamW_triple_faithful` (`StableHLO`) is a structural match."

### LeanMlir/Proofs/Training/Optim/Lamb.lean:34

**Kind:** stale
**Says:** "`lambDir_wd_inside` below states the second as an inequality rather than as prose."
**Actually states:** `lambDir_wd_inside` is an equation: `lambDir … wd … i = lambDir … 0 … i + wd * θ i`.
**Fix:** "…states the second as an identity (the direction moves by exactly `wd·θ` before the trust ratio)."

### LeanMlir/Proofs/Training/Optim/RmsPropStep.lean:93–96 `rmsSqNext_nonneg`

**Kind:** stale
**Says:** "The reference starts `s` at **1.0** … the FIRST step is damped (`g/√(1−ρ+…)`)".
**Actually states:** with `s = 1`, `rmsSqNext ρ 1 g = ρ + (1−ρ)g²`, so the first step is `g/√(ρ + (1−ρ)g² + ε)`. At ρ = 0.9, `√(1−ρ)` would *amplify* by ≈3.2×, which contradicts "damped".
**Fix:** "`g/√(ρ + (1−ρ)g² + ε)`".

### LeanMlir/Proofs/Architectures/PerChannelBNGrad.lean:8; ConvBackCertifiedTie.lean:75; TokenParamGrad.lean:386

**Kind:** stale
**Says:** "the BN analogue of `IR.bias_grad_bridge` / `conv_bias_grad`"; `dense_transpose_eq_vjp_backward`: "conv is linear so the activation `x` is ignored"; TokenParamGrad trailing comment: "the row-lifted scalar-LN γ/β (§ B)".
**Actually states:** no `conv_bias_grad` exists (it is `conv_bias_grad_bridge`, ConvGrad.lean:46). The theorem is about `dense`. TokenParamGrad has no §B, and ViT's LN γ/β are the vector `vit_vecln{Gamma,Beta}_grad_bridge` (LayerNorm.lean).
**Fix:** `conv_bias_grad_bridge`; "dense is linear in its input"; "the vector-LN γ/β (`vit_render_vecln{gamma,beta}_certified`, LayerNorm.lean)".

### LeanMlir/Proofs/Architectures/EvenKernelConvBack.lean:21

**Kind:** stale
**Says:** "`StableHLO.lean`'s `.convStridedBack` emitter … says so in as many words — *'The symmetric `[[p,p],[p,p]]` this emitted AGREES …'*".
**Actually states:** the quoted comment is in `Codegen/StableHLOPretty.lean:3744`, not `StableHLO.lean`.
**Fix:** "`StableHLOPretty.lean`'s `.convStridedBack` emitter …".

### scripts/certs/trained_linear_descent.py:114 (→ Training/TrainedLinearDescent.lean module)

**Kind:** stale
**Says:** "`binary32_linear_sgd_descends_concrete` (the suite's only concrete descent instance) holds at the degenerate `W = 0` net".
**Actually states:** the generated file itself adds `trained_linear_sgd_descends_concrete`, a second concrete instance.
**Fix:** "`binary32_linear_sgd_descends_concrete` (Float/Binary32Instance.lean) holds only at the degenerate `W = 0` net."

### LeanMlir/Proofs/Training/SgdDescentCnn.lean:5722 `cnn_conv2_bias_float_sgd_descends`, :6150 `cnn_conv1_bias_float_sgd_descends`; JacobianSeal.lean:50,72; DropPath.lean:320

**Kind:** copied
**Says:** both float bias rungs open "**One inexact SGD step on the CNN's … conv BIAS provably decreases …**" (the abstract-η wording). Two different JacobianSeal theorems (`exists_pdiv_ne_of_fderiv_ne`, `HasVJPAt.backward_nontrivial_of_fderiv_ne`) are both headed "**The seal in `fderiv` form.**" `keepProb`: "`keep_i = 1 − dropPath · i / (totalDrop − 1)`".
**Actually states:** the bias rungs use the FloatModel gradient, with no abstract `η`. The first JacobianSeal theorem only produces a nonzero Jacobian entry. `keepProb`'s parameter is `dropRate`.
**Fix:** "One SGD step with the FloatModel binary32 bias gradient …"; retitle `exists_pdiv_ne_of_fderiv_ne` "Nonzero `fderiv` ⇒ a nonzero Jacobian entry"; "`1 − dropRate · i / …`".

### LeanMlir/Proofs/Architectures/LayerNorm.lean:13 (module)

**Kind:** missing
**Says:** "# LayerNorm & GELU … Two quick chapters".
**Actually states:** the file also holds Swish (`swishHasVJP`, `swishScalarDeriv_eq`), `layerScale`, the vector LN `layerNormVec`/`layerNormVecHasVJP`/`layerNormVecPerTokenHasVJPMat`, and the ViT LN γ/β bridges `vit_render_vecln{gamma,beta}_certified` — the definitions ChannelLN, TokenParamGrad and ViTFold import it for.
**Fix:** add one paragraph listing those sections and their consumers (ChannelLN, ViT).

### Process narrative (grouped — each module docstring cites plan sections, phases, dates or run results)

**Kind:** process-narrative
**Says (representative):**
- ChannelLN.lean:8–38 — "`convnextVerified`'s LN was `bnForward` …", "Settled on device before any of this was written (`lake build channel-ln`) … rel 0.82 … Δ 0.00 ms", "§2k's own sin in a new place".
- ChannelLNBack.lean:16 — "Moved here from the float bridge … on 2026-09-08".
- EvenKernelConvBack.lean:21–32 — "for the third time (`planning/archive/float_budget_numbers_log.md` §3.10 …)".
- StridedConv.lean:3–7 — "Chapter 5 Milestone B, the hard new op … the Chapter-5 handoff (`planning/archive/verified_r34.md` §3.6)".
- PerChannelBN.lean:3,135,300,352 — "Milestone B8", "B9 entry", "B8b's `den` target".
- MaxPool3s2.lean:119–127 (comment) — whnf-timeout war story.
- Attention.lean:406–421, 702–709, 750–766, 805, 1139–1146, 1192, 1756, 1809–1823 — "Earlier drafts … `axiom sdpaHasVJP`", "Phase 3, Apr 2026", "Phase 6 (Apr 2026)", "Phase 8 … (was an axiom)", "kernel-cost lesson, 2026-07".
- Softmax.lean:48,163; BatchNorm.lean:477,527,607; LayerNorm.lean:127,141,214 — "Proved (was an axiom)", "planning/archive/VJP.md follow-up C/E".
- AdamStep.lean:4 "(Phase 3a)"; GradClip.lean:3 "the ViT / ConvNeXt recipe's last v1.4 piece"; Lamb.lean:4,12 "the one item `rsb_a3_r50_verified.md` §2.3 ESTIMATED", "(which reached **76.66% top-1 @ ep100**)".
- DropPath.lean:197–200, 387–391 — "the fourth time … (§2k heavy-ball, recipe_gaps v1.2 RMSProp, the EMA shadow, here)", "the fifth time".
- SgdDescentCnn.lean:91,305,606,640,904,1183,1470,2456,4028,4447 — "Item A capstone", "Increment 1–4 keystone/capstone"; SgdDescentLinear.lean:310 / SgdDescentMlp.lean:796 "Item D / G1"; SgdDescentCifar.lean:4–7 "(A2 probe)".
- BatchSealKit.lean:8–10 "`planning/full_width_seals.md` §3 … have until now been exhibited on 2-channel proxies".
- Generators: trained_cnn_witness.py:418 "(the 2026-07 audit's gap #3)"; trained_linear_descent.py:107 "(post_audit_roadmap §3)".
**Actually states:** none of these labels, dates, measurements or run accuracies is part of any statement. They date as the plans move; several already point at archived plans.
**Fix:** strip the phase/item/increment/milestone labels, dates, run numbers and "was an axiom" histories from docstrings (keep them in commit messages). Keep the mathematical rationale, e.g. EvenKernelConvBack's "`convFlatBack` is the adjoint only at odd kernels; `padOdd` repairs even ones".

---

## Overclaims (fix before the published results are read again)

1. **SgdDescentCnn / SgdDescentCifar** — post-ReLU `MaxPool2MarginQ` is called the "off-the-kink condition". In practice it rules out any window with two dead ReLUs, and the MNIST `lr` regime is called "non-vacuous" with nothing proving it. "EVERY parameter … has a proven descent statement" hides that each statement covers one layer and one example under those margins.
2. **"One binary32 SGD step … NO abstract gradient-accuracy parameter"** (SgdDescentLinear/Mlp/Cnn float rungs, generated TrainedLinearDescent) — the update is in ℝ, the loss is one example, `hδ` is assumed, and the concrete instance uses exact `exp`. "Exactly as the rendered trainer computes it" is a FloatModel transcription, not a tie to the B = 128 XLA trainer.
3. **Attention `vitFull` / `vitFullHasVJP_correct`** — "full ViT", but every block shares one parameter tuple and LN is scalar-affine. The per-block generalization is called "trivial / mechanical" and is unproved.
4. **CNN `MaxPool2Smooth`** — called "the natural domain on which `maxPool2` is differentiable", but it is strictly stronger than that. The capstone's `h_mp` sits on the post-ReLU tensor.
5. **ConvGrad `cnn_render_conv{W,b}_certified` / conv bridges** — "rendered … denotes certified", but the statements are `.correct` projections with no rendered text (group 2).
6. **LayerNorm taxonomy** — "all four normalization variants share one `HasVJP` instance". Only BN = LN is formalized, and "InstanceNorm (which is what the ResNet code actually uses)" is false.
7. **Residual `residualHasVJP`** — "the gradient floor is `dy` … can never get smaller than the loss gradient" is mathematically false; the terms can cancel.

Prior-audit groups not found in this slice's prose: 3 (round-trip), 4 (MNv4/timm), 5 (B0 10-class; "every shipped artifact"), 6 (CotIn_eq_vjp).
