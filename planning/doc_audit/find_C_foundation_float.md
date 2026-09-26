# Slice C — `LeanMlir/Proofs/Foundation/`, `LeanMlir/Proofs/Float/`, `LeanMlir/Proofs/SpecVJP.lean`

**Coverage.** I read every module docstring and every declaration docstring, each next to its statement (up to `:=`), in all 48 files of Foundation/ and Float/ plus `Proofs/SpecVJP.lean` (the only `.lean` directly in `Proofs/`). That is about 15.4k lines. I read proof bodies only where a claim depended on them. Every backticked identifier in the slice's prose was checked against the declarations repo-wide. The naming pass (7dbe9c92) left **no** stale old-style names in this slice. The only unresolved names are `r34_float_close`, `CifarFloatBridge.lean`, `rowwise` and `ResNet34StepTieB.mpInB`, each reported below. `sorry` / `axiom ` / `admit` / `native_decide` / `implemented_by` / `@[extern]`: I grepped all three paths and found none, and repo-wide there are no `axiom` declarations. So the "zero axioms / no sorry" claims in Binary32Instance, FloatSubnormalBridge, CertifiedChain, Tensor and IR hold. One exception: Bf16Fold's citation of `tests/AuditAxioms.lean` is wrong (see below).

Ranked most misleading first.

---

### LeanMlir/Proofs/Float/FloatSubnormalBridge.lean:3 — module docstring (and `subFloor_total_negligible`, :166)

**Kind:** overclaim
**Says:** "the subnormal floor, closed as a lemma (not a caveat)" … "This file closes that gap … by proving activations **stay normal** so the clean relative model genuinely applies" … "binary32 RN instantiates it with `u = 2⁻²⁴`, `η = 2⁻¹⁵⁰` … now *stated*, not hidden" … "`bnDenom_normal` / … — this is *why* LN/BN keep activations O(1)" … "the floor cannot move any closeness bound". At :166, "**The subnormal floor cannot move any bound.** … below every nonzero closeness budget that appears."
**Actually states:**
- The only `FaithfulFloatModel` constructed is `exactFaithful` (`rnd = id`). No binary32 instance exists; grep finds `FaithfulFloatModel` outside this file only in a FloatBridge docstring.
- `bnDenom_normal` / `bnSqrt_normal` / `istd_ge_minNormal` prove that the BN *denominator*, its square root and `istd` are `≥ minNormal`. Nothing is proved about activations.
- `subFloor_total_negligible` is the arithmetic `(n:ℝ) ≤ 2⁶⁴ → n·2⁻¹⁵⁰ ≤ 2⁻⁸⁶`. No theorem connects it to any budget.
- None of these lemmas is used outside the file.
**Fix:** "Defines the subnormal-honest rounding model `FaithfulFloatModel` (relative bound on the normal range, absolute floor `η` everywhere) and shows `FloatModel` is its `η = 0` face. Proves the BN/LN denominator `var+ε`, its root, and (under `√(var+ε) ≤ minNormal⁻¹`) `istd` lie in the normal range, and that `n ≤ 2⁶⁴` floor terms sum to `≤ 2⁻⁸⁶`. No binary32 instance is constructed and no existing budget is re-derived under this model; activations staying normal is not proved."

### LeanMlir/Proofs/Float/FloatBridge.lean:222 — `dot_close` (also :73 `dot`, :1268 `sum`, :1277 `sum_close`, module :26–30)

**Kind:** overclaim
**Says:** "valid for every association of the sum, not just the left fold `dot` fixes". At :73, "The bound below is association-independent, so the choice is immaterial". At :1277, "association-independent". The module says "so the statement survives a backend that reassociates (IREE tiles reductions)".
**Actually states:** `|M.dot x y - ∑ i, x i * y i| ≤ ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i * y i|`, where `M.dot` is the left fold defined at :74. No other association is stated. `BnFloatBridge.bnMean_close_of` (:190) says so itself: "a number stated through that instance is about a program we do not ship", and it adds an order-parametric `fsum` form to cover other orders. ("IREE" is also stale; the engine is XLA/PJRT.)
**Fix:** `dot_close` → "Rounded dot product forward error for the left-fold `M.dot`. (The classical bound holds for every summation order, but only the left fold is formalized here; `FloatModel.bnMean_close_of` shows the order-parametric form.)" Make the same edit at :73, :1268, :1277 and in the module's "Order-robustness" bullet (and replace "IREE" with "XLA").

### LeanMlir/Proofs/Float/FloatClose.lean:25, :69 · FloatComposeBridge.lean:9, :22, :48, :163, :196 · DepthwiseFloatBridge.lean:173 · ConvMixedComposeBridge.lean:252 — "whole-net float certificate"

**Kind:** overclaim (and stale: `r34_float_close` does not exist)
**Says:**
- `FloatClose.comp`: "the whole-net certificate backbone".
- `floatClose_iterate`: "THE FINAL FOLD … This is r34's within-stage depth …; the whole net is these iterates `.comp`-joined with the stem / downsamples / GAP / dense. The depth-generic whole-net certificate".
- `floatClose_r34_stages`: "The full `r34_float_close` is these `.comp` the stem / strided downsamples / GAP / dense".
- `floatClose_dense`: "The SE excite/reduce denses and the classifier head are this instance; the ViT MLP denses reuse it too".
- `floatClose_bn`: "The BN-before-swish steps in EfficientNet's MBConv … are this instance".
- `floatClose_depthwise`: "the MBConv depthwise stage folds through `.comp` like any other conv".
- `floatClose_flatConvMixed`: "This is the ONLY thing the whole-net bf16 bound needed".
**Actually states:** per-op `FloatClose` instances and a composition lemma only. `r34_float_close` is not declared anywhere. No `FloatClose` name is used outside `Proofs/Float/`: `floatClose_dense` and `floatClose_flatConvMixed` have 0 uses, and no whole-net fold exists for any net. `BackwardMaps.lean:16` records that this tier's whole-net budgets "were found vacuous and deleted".
**Fix:** Retitle FloatComposeBridge "per-op `FloatClose` instances for the conv-net op set". In each docstring, drop the "whole-net certificate" / "FINAL FOLD" / "is this instance" / "ONLY thing … needed" sentences. Replace them with "a whole-net bound would be `.comp` of these; none is assembled in the repo". Delete the `r34_float_close` reference.

### LeanMlir/Proofs/Foundation/MuonNewtonSchulz.lean:243 — `nsStep_iterate_tendsto_polar` (also module :11–13; MuonGeometry.lean:445 `muon_polar_orthogonal`, :199 `muon_polar_steepest`)

**Kind:** overclaim
**Says:** "**This closes the loop: the thing the hardware computes is the thing the theory says is optimal.**" The module says "What remains is that Muon's matmul iteration computes it: `X ↦ aX + b(XXᵀ)X + c(XXᵀ)²X` converges to `UVᵀ`." MuonGeometry says "the implementation's Newton–Schulz iteration is the retraction that computes this projection" and "This is *why* Muon's `den = UVᵀ` Newton–Schulz update is steepest descent".
**Actually states:** convergence holds under `hconv : ∀ t₀ ∈ (0,1], g^[k] t₀ → 1`, instantiated only for the cubic `(3/2,−1/2,0)` and the principled quintic `(15/8,−5/4,3/8)`. The same file proves that Muon's shipped coefficients `(3.4445,−4.7750,2.0315)` violate the hypothesis (`qScalar_not_le_one`, `qScalar_one_lt_one`) and only band (`qScalar_iterate_band_half`).
**Fix:** ":: For any coefficients whose scalar map drives `(0,1]` to `1` — proved here for the classic cubic and Higham's quintic, NOT for Muon's tuned quintic, which bands (`qScalar_not_le_one`) — the matmul iterate converges to the polar factor." Make the matching edits in the module and in the two MuonGeometry docstrings ("an iteration with convergent coefficients computes this projection; Muon's tuned quintic approximates it to a band").

### LeanMlir/Proofs/Float/ConvMixedComposeBridge.lean:33 — module docstring vs `convMixedGain_factor` (:297)

**Kind:** overclaim (self-contradiction)
**Says:** at :33, "the §9.3 separation that makes this non-vacuous at R50's n = 4608". At :310 the same file says: "⚠⚠ **AND BOTH BOUNDS ARE VACUOUS IN ABSOLUTE TERMS** … `gain^53` is astronomical for the f32 bound and the bf16 one alike."
**Actually states:** `convMixedBudget` is affine in `E` with slope `n·w·(1+ε)`. No non-vacuity statement exists.
**Fix:** at :33, "…the §9.3 separation: the fan-in amplification rides `uacc`, so the per-layer relative factor stays `1 + O(u)`; composed depth-first the bound is still vacuous in absolute terms (see `convMixedGain_factor`)."

### LeanMlir/Proofs/Foundation/MuonGeometry.lean:300 — `muon_polar_achieves_nuclear_of_isUnit`

**Kind:** overclaim + stale
**Says:** "**Muon's update is the steepest ascent in operator-norm geometry — unconditionally, for any invertible `G`.** … (Von Neumann's trace inequality — that `Σσᵢ` is the *max* … — is the next layer.)"
**Actually states:** `∃ U V s, UᵀU = 1 ∧ VᵀV = 1 ∧ (∀ i, 0 ≤ s i) ∧ G = U·diag s·Vᵀ ∧ fInner G (U*Vᵀ) = ∑ s i`. This is achievability only, not maximality. The "next layer" (`muon_polar_is_max`, :146) is already in the file. The same stale "next layer" appears at :120 (`muon_polar_achieves_nuclear`).
**Fix:** "**For invertible `G`, the constructed SVD's polar factor `UVᵀ` attains the nuclear norm** `⟨G, UVᵀ⟩_F = Σ sᵢ`. With `muon_polar_is_max` this makes `UVᵀ` the operator-norm steepest-ascent direction." Drop "next layer" at :120 and :306.

### LeanMlir/Proofs/SpecVJP.lean:421 — `efficientnetVerifiedHasVJP`

**Kind:** stale (wrong)
**Says:** "canonical `pdiv` witness (swish/SE are smooth but relu6 clamps; the per-block differentiability lemmas live in `EfficientNetFullB0.lean`)."
**Actually states:** `HasVJP.canonical _`. EfficientNet-B0's forward has no relu6: `EfficientNetFullB0.lean` contains only `swish` activations and grep finds no `relu6` in it. A genuine global witness `efficientnetForwardBFullHasVJP` (hypothesis `w.EpsPos`) exists at `EfficientNetFullB0.lean:371`.
**Fix:** "canonical `pdiv` witness. B0 is all-smooth (swish, SE sigmoid, batch BN); the real whole-net VJP is `efficientnetForwardBFullHasVJP` (on the ∘-chain form, `w.EpsPos`)." Or use that witness as ViT's rung does.

### LeanMlir/Proofs/Float/FloatBridge.lean:1326 — `softmaxF` (also :1644 `mnist_cot_budget`)

**Kind:** overclaim
**Says:** "GPU `exp` has no IEEE spec; its accuracy constant is exactly what the repo's `vjp_oracle` harness validates empirically." At :1644: "the constant is what `vjp_oracle` validates".
**Actually states:** `fexp` / `eexp` are free hypotheses. `tests/vjp_oracle/README.md`: the harness "Diffs step-2 loss" between two training pipelines. It measures no `exp` relative-error constant.
**Fix:** "`fexp` is hypothesis-supplied (GPU `exp` has no IEEE spec); `eexp` is an assumed relative accuracy, not measured by any harness in the repo."

### LeanMlir/Proofs/Foundation/DataParallel.lean:3 — module docstring, "What is NOT claimed"

**Kind:** stale
**Says:** at :42–44, "⚠ **Nothing here is about the emitted `all_reduce`.** `den (allReduceMeanF R g) = …` is §4d piece 2 … and it waits on 4c's batched chains. Until it lands a tie composes with these lemmas only through the reader." At :46, "⚠ **BatchNorm statistics are per replica.** Nothing all-reduces μ/var".
**Actually states:** the same header (:8) already says piece 2 landed (`SHlo.allReduceMeanF`, `DataParallelNode.lean`). Sync-BN all-reduces μ/σ² (`DataParallelSync.lean`, `den_syncStats_left/right`), and this file proves the sync-BN positive result `dpSyncGrad_eq_globalBatchGrad` (:278).
**Fix:** replace the first bullet with "The `SHlo` collective and its `den` are `DataParallelNode.lean`." Replace the second with "Under per-replica BN (the non-sync renders) nothing all-reduces μ/var, which is why `dpMeanGrad_ne_globalBatchGrad` bites; sync-BN renders all-reduce them and `dpSyncGrad_eq_globalBatchGrad` applies."

### LeanMlir/Proofs/Foundation/SmoothedLossCot.lean:5 — module docstring

**Kind:** stale
**Says:** "Every whole-net T3 tie in the repo pins its top-of-chain cotangent to `softmax(logits) − oneHot label` … it is not what the batched ImageNet renders emit."
**Actually states:** 12 step-tie files now consume `smoothedLossCotGraph`. They are the ResNet34/50, MobileNetV2/V4, EfficientNet, ConvNeXt and ViT `*StepTie*B/G` files and their `*Sync*` twins.
**Fix:** "The per-example SGD-inline renders' ties pin the cotangent to `softmax − oneHot`; the batched renders emit the six-op smoothed chain below, and their step ties state it through `smoothedLossCotGraph_row`."

### LeanMlir/Proofs/Float/Binary32Instance.lean:94 — `binary32_e4m3_budget_small`

**Kind:** overclaim
**Says:** "This makes precise why the deployed net (errors not aligned, activations far below the `m·w·a` ceiling) needs only the measured `0.38` drift, not `61`: the bound scales with realized fan-in, not the worst-case 784."
**Actually states:** `denseMixedBudget binary32.u fp8E4M3.u 4 (3/5) 1 1 ≤ 1/2`, a numeric bound at input width 4. Nothing about the deployed net or its measured drift.
**Fix:** "The same worst-case budget at input width `m = 4` is `≤ 1/2`: the budget is linear in fan-in. (Why the deployed 784-wide net drifts only 0.38 is an empirical observation, not proved here.)"

### LeanMlir/Proofs/SpecVJP.lean:151, :327, :378, :421, :477 — canonical-witness "carries the math"

**Kind:** overclaim
**Says:** "**The spec carries the math.** The CNN spec's denotation … has a VJP" (and the same headline for MobileNetV2, ResNet-34, B0, ConvNeXt).
**Actually states:** `:= HasVJP.canonical _`. `Tensor.lean:282` says the canonical witness "Exists for every `f`" and its `correct` is `rfl`, so these defs certify nothing about the spec.
**Fix:** "**The spec's denotation, with the canonical (definitional) witness** — exists for every function and adds no content. The certified pointwise VJP is `<net>HasVJPAt` (cited)." Keep the headline "carries the math" only for the linear, MLP-`At` and ViT rungs, which use real witnesses.

### LeanMlir/Proofs/Foundation/IR.lean:827 — `lossCot_bridge`

**Kind:** overclaim + stale
**Says:** "So the cotangent fed to the backward is itself proof-backed … the whole train step `forward → loss → backward → grads` is proof-backed end to end, and only the SGD arithmetic (and printer/IREE/float) stays trusted."
**Actually states:** `emitLossCot c logits label j = pdiv (fun z _ => crossEntropy c z label) logits j 0`, one equation about the loss head. No end-to-end statement is made here, and IREE is no longer the engine.
**Fix:** "The emitted softmax−onehot vector is `∂(crossEntropy)/∂logits` (`softmaxCE_grad`), so the backward's cotangent leaf is proof-backed, not supplied."

### LeanMlir/Proofs/Foundation/CertifiedChain.lean:6 — module docstring

**Kind:** overclaim
**Says:** "Every net's whole backward is built this way now: `r34NetLayer` / `r50NetLayer` …, the MobileNet and EfficientNet chains, and ViT's tower".
**Actually states:** `CertLayer` is used in ResNet, MobileNet, ViT and ConvNeXt files. No file under `Nets/EfficientNet/` (or anywhere else for B0) uses `CertLayer`.
**Fix:** "The ResNet, MobileNet, ConvNeXt-block and ViT backwards are built this way …; EfficientNet-B0's is still open-coded (`EfficientNetChainClose`)."

### LeanMlir/Proofs/Float/BnFloatBridge.lean:69 — `bnIstd_close_at`

**Kind:** overclaim
**Says:** "…(empirically ~10⁷× on the CIFAR-BN probe…). The non-vacuous BN certificate."
**Actually states:** a conditional bound `|fistd fvarε − bnIstd| ≤ ers/√V + evar/(2V√V)` under the hypotheses `V ≤ fvarε`, `V ≤ σ²+ε`, `hrs` and `hclose`. It is not a certificate of anything shipped, and no instantiation exists in the repo.
**Fix:** "BN inverse-stddev budget at an operating-point variance floor `V` (a-posteriori): `ers/√V + evar/(2V√V)`, far tighter than the `ε`-floor form when `σ² ≫ ε`."

### LeanMlir/Proofs/Float/BnFloatBridge.lean:4 — module docstring (and `bnForward_close`, :311)

**Kind:** stale
**Says:** "The no-BN CIFAR bridge (`CifarFloatBridge.lean`) reuses…" and "(mean/var rounding + the normalize-stage products remain the mechanical tail)". At :317: "The only supplied input is `fvarε`'s closeness to `σ²+ε` (`hvar`) — the variance Higham reduction, the one remaining mechanical piece."
**Actually states:** `CifarFloatBridge.lean` does not exist. This file already proves `bnMean_close`, `bnMean_close_of`, `bnVar_close` and `bnForward_close_of`.
**Fix:** drop the CifarFloatBridge sentence. Module: "…composed with the Higham mean/variance budgets (`bnMean_close`, `bnVar_close`) and the normalize chain (`bnForward_close_of`) in this file." `bnForward_close`: "…`hvar` is taken as a hypothesis; `bnVar_close` supplies the variance budget in the rounded-sum form."

### LeanMlir/Proofs/Foundation/BatchMapVJPAt.lean:3 — module docstring

**Kind:** stale
**Says:** "`EfficientNetChainClose.lean` lifts … the GLOBAL form: `batchMapHasVJP` takes `HasVJP f`…". Also "the one thing standing between `ResNet34FullB.lean` and T1", and "⚠ The r34 stem's instance lives with r34's VJP, not here — `maxPool3s2FlatHasVJPAtVec` is in the `Float` tier and this is a `Foundation` file."
**Actually states:** `batchMapHasVJP` is declared in this file (:161). `maxPool3s2FlatHasVJPAtVec` lives in `Foundation/BackwardMaps.lean:246`. R34's T1 has landed.
**Fix:** "The global lift `batchMapHasVJP` (below) takes `HasVJP f`…; `batchMapHasVJPAt` is its pointwise peer, which r34's stem pool needs because `maxPool3s2FlatHasVJPAtVec` (`BackwardMaps`) is `_at` by nature." Drop the T1 sentence.

### LeanMlir/Proofs/Foundation/MuonNewtonSchulz.lean:6 and MuonGeometry.lean:8 — module docstrings

**Kind:** stale + process-narrative
**Says:** "**This file is P1 — the spectral-step lemma**" and "The downstream scalar analysis (P2) and the matrix-continuity assembly (P3) build on these two lemmas." MuonGeometry has "**L4 (this layer)**", "**L5 (this layer)**" and "The only remaining layer is the singular `G` case".
**Actually states:** MuonNewtonSchulz contains P1–P4: `scalar_iterate_tendsto_one`, `gCubic`/`q5Scalar` convergence, `nsStep_iterate_tendsto_polar`, and the tuned-quintic negative results. The layer and phase numbers refer to a retired plan.
**Fix:** MuonNewtonSchulz: "Newton–Schulz iteration for the polar factor: the spectral-step reduction to a scalar map, a monotone convergence engine, convergence for the cubic and Higham's quintic, the lift to matrices, and why Muon's tuned quintic only bands." MuonGeometry: name the results without "L4/L5 (this layer)"; state the singular-`G` gap as a limitation of the `_of_isUnit` forms.

### LeanMlir/Proofs/Float/FloatBridge.lean:928 — `mlp_w2_step_float_close`

**Kind:** stale
**Says:** "Takes the output cotangent `gt ≈ g` as a hypothesis (the softmax−onehot head needs an `exp` accuracy axiom — future rung)."
**Actually states:** the same file proves `softmax_ce_cot_close` (:1609) and `mnist_cot_budget`, with a hypothesis `hfexp` (not an axiom) discharging exactly that `gt ≈ g`.
**Fix:** "Takes the output cotangent `gt ≈ g` as a hypothesis; `softmax_ce_cot_close` discharges it with `eg := cotErr u eexp δ n`."

### LeanMlir/Proofs/Float/FloatSubnormalBridge.lean:5 — module docstring, first paragraph

**Kind:** stale
**Says:** "`FloatBridge.lean`'s `FloatModel` … (its docstring flags the subnormal absolute-error term as future work; …)".
**Actually states:** FloatBridge's module now says "the subnormal absolute-error term is `FloatSubnormalBridge`'s". It flags nothing as future work.
**Fix:** "…true for IEEE-754 binary32 round-to-nearest only on the normal range (FloatBridge's module notes this and points here)."

### LeanMlir/Proofs/Float/ResNet34FloatBridge.lean:6 — module docstring (and ConvFloat.lean:9)

**Kind:** stale
**Says:** "This file holds the two that are not one line elsewhere: **residual skip** `relu(F(x) + skip(x))` — a two-operand `add_close` …; **global-avg-pool**". ConvFloat.lean:9 says "(ResNet-34's strided convs through `ResNet34FloatBridge`)".
**Actually states:** the file declares only `gapFlatF`, `gapFlat_close` and `globalAvgPoolFlat_eq_bnMean`. `add_close` is in `FloatBridge.lean`, the residual `FloatClose` instances are in `FloatComposeBridge`, and there is no strided-conv lemma here.
**Fix:** "The float global-average-pool (`gapFlatF`, `gapFlat_close`) and its reading as a per-channel `bnMean`." Drop the ConvFloat parenthetical.

### LeanMlir/Proofs/Foundation/BatchedBackLinks.lean:69 — `bnBatchBack_faithful`

**Kind:** stale + process-narrative
**Says:** "The first batched-backward primitive … The `bnBatchLA` layout-reindex wrapper to the network's `N·(oc·h·w)` index is a thin remaining layer."
**Actually states:** `bnBatchLABack_faithful` (:214, same file) is that wrapper. "First", "second brick" (:92) and "fourth (and last)" (:226) are order-of-work notes.
**Fix:** "…via the three-term `bnBatchTensor4GradInput`. The network-layout wrapper is `bnBatchLABack_faithful`." Drop the ordinal remarks at :92 and :226.

### LeanMlir/Proofs/Foundation/BackwardMaps.lean:284 — `maxPool3s2FlatBackB` (and :63 `perRowFlatPR`)

**Kind:** stale
**Says:** "(`ResNet34StepTieB.mpInB` is the `maxPool3s2BackFlat` one)". At :63: "The flat analogue of `rowwise` (`Tensor.lean`)".
**Actually states:** `mpInB` is declared in namespace `Proofs.ResNet34TieB` (`BatchedBackLinks.lean:485`); no `ResNet34StepTieB.mpInB` exists. No declaration named `rowwise` exists (Tensor.lean has `rowwiseHasVJPMat`).
**Fix:** `ResNet34TieB.mpInB`; `rowwiseHasVJPMat`.

### LeanMlir/Proofs/SpecVJP.lean:18 — module docstring

**Kind:** stale
**Says:** "# Spec → math (the verification tie), Rung 1: the linear classifier … This file is the first rung of connecting a readable `VerifiedNetSpec` to the actual **math** … on the simplest net".
**Actually states:** the file carries rungs 1–4 and rungs B/C/E for the linear classifier, MLP, CNN, CIFAR, MobileNetV2, ResNet-34, EfficientNet-B0, ConvNeXt-T and ViT-Tiny.
**Fix:** "# Spec → math: each committed `VerifiedNetSpec` denotes its proven forward. For every shipped net's layer list: `denote` = the proven forward by `rfl` (drift-sensitive), a VJP witness for it, and the forward graph's faithfulness composed with that tie."

### LeanMlir/Proofs/Float/Bf16Fold.lean:36 — module docstring

**Kind:** stale
**Says:** "All theorems kernel-close under `[propext, Classical.choice, Quot.sound]` ([`tests/AuditAxioms.lean`](…))".
**Actually states:** `tests/AuditAxioms.lean` neither imports `Bf16Fold` nor prints its theorems (grep for `bf16_render_faithful` / `Bf16Fold` finds 0 hits). The theorems are `rfl` and are axiom-free in fact, but the cited audit does not cover them.
**Fix:** add the file to `tests/AuditAxioms.lean`, or cite whatever check does cover it.

### LeanMlir/Proofs/Float/FloatBridge.lean:11 — module docstring (tier list)

**Kind:** stale
**Says:** "`Binary32Instance` / `RndP` (binary32, bf16, E4M3 as instances)".
**Actually states:** only `binary32` and `fp8E4M3` are constructed `FloatModel`s. bf16 appears only as an abstract `rnd`/`L` or as `rndP 7` in DataParallelSyncBf16; no bf16 `FloatModel` exists.
**Fix:** "(binary32 and E4M3 as named `FloatModel`s; `rndP` is the grid operator behind them and behind the bf16 sharding lemmas)".

### LeanMlir/Proofs/Float/Binary32Instance.lean:58 — `binary32`

**Kind:** overclaim (minor)
**Says:** "…at unit roundoff `u32 = 2⁻²⁴` — the bound is tight for the grid."
**Actually states:** `gridModel 23 u32 _`. No tightness statement exists; the sup of `|rnd x − x|/|x|` on the grid is `u/(1+u) < u`, so the bound is not attained.
**Fix:** "…at unit roundoff `u32 = 2⁻²⁴` (the standard-model constant for this grid)."

### LeanMlir/Proofs/Foundation/MLP.lean:65 — `mnistLinearHasVJP_correct`

**Kind:** overclaim (group 2, minor)
**Says:** "Whole-model VJP contract for the linear classifier — the degenerate simplest case of the per-architecture `*HasVJP_correct` capstones".
**Actually states:** `:= (denseHasVJP W b).correct x dy i`, the `.correct` projection restated.
**Fix:** "`denseHasVJP`'s `.correct` field, restated for `mnistLinear`, so it can be cited by name."

---

### Proof narrative in declaration docstrings

### LeanMlir/Proofs/Float/DepthwiseMixedFloatBridge.lean:35 — `dwSlice`

**Kind:** proof-narrative / process-narrative
**Says:** "⚠⚠ **`dwWindow`, `dwKernelMat` and `depthwiseConv2d_eq_dense` are REUSED … The first draft of this file defined its own `dwWindow` … and `lake build LeanMlir` refused the import … broken for three commits … ▶ When a `dw*`/`conv*` helper seems to be missing, grep before defining".
**Actually states:** `dwSlice W ch : Vec (kH*kW)`, one channel's flattened filter.
**Fix:** "Channel `ch`'s flattened filter as a `Vec (kH·kW)` — `dwKernelMat`'s single column." Move the build history to the commit message.

### LeanMlir/Proofs/Float/ConvMixedComposeBridge.lean:297 — `convMixedGain_factor`

**Kind:** proof-narrative / overclaim
**Says:** about 20 lines of commentary: "`(1.012043/1.000275)^d` … 1.52× at R34's 36 … Under a factor of two on the certificate, for a 1.41×/1.55× speedup … the f32 whole-net bridges the repo carried until 2026-09-08 had exactly the same factor …".
**Actually states:** an algebraic identity: `convMixedGain uacc uleaf n w = n·w·(1 + (br + uleaf(1+br) + uacc(1+uleaf)(1+br)))`.
**Fix:** "The per-layer gain factors as `n·w·(1+ε)` with `ε = br + u_leaf(1+br) + u_acc(1+u_leaf)(1+br)`: bf16-mixed changes only the `1+ε` factor relative to f32, not the `n·w` growth rate." Move the numeric illustration to a comment and drop the history.

### LeanMlir/Proofs/Float/Bf16Fold.lean:123 — `bf16Depth2`

**Kind:** proof-narrative
**Says:** "(Writing `rnd ∘ …` here instead would round twice and the tie below would fail — it did, first try.)"
**Fix:** delete the parenthetical, or move it to a comment.

### LeanMlir/Proofs/Foundation/IR.lean:268 — `convBackDenote_eq_input_grad_formula`

**Kind:** proof-narrative / process-narrative
**Says:** "This is the reversed-kernel ⇒ correlation-adjoint reindex that `conv_back_bridge_{1to2,2to2}` previously asserted only at two toy 4×4 shapes … Proof: per output coordinate … The single load-bearing leaf for the §B certified-VJP tie". There is a similar "no longer the brute-force `fin_cases` expansion" at :288.
**Fix:** "For odd `kH`, `kW` and all dimensions, the reversed-kernel forward conv `conv2d (reverseSwap W) 0` equals the conv input-gradient `conv2dInputGradFormula W`. Every conv net's `convFlatBack` routes through this." Move the proof sketch into the proof.

---

### Process narrative in module or declaration docstrings (grouped)

**Kind:** process-narrative. Fix for all of them: delete the dated or phase clauses and keep the mathematical sentence.
- **BackwardMaps.lean:16** — "Until 2026-09-08 these lived inside the `Proofs/Float/*FloatBridge` files…". **:210** — "Found 2026-08 because the per-example r34 chain (retired 2026-09-19) had been written as the reverse of the 2×2 pool…". **:323** — "Until this lemma the only bridge was…".
- **OpaquePrefix.lean:10** — "Until 2026-09-08 ResNet-34, EfficientNet-B0, MobileNetV2 and MobileNetV4 each carried a private copy…".
- **BatchedStages.lean:39** `stemB` — "the shipped render has emitted `convStridedXla` there since 2026-08-08 … (re-spelled 2026-09-05, …)".
- **DataParallelNode.lean:14** — "Since 2026-09-07 the collective is `SHlo.allReduceMeanF` … every committed `*dp*` artifact re-rendered byte-identically"; **:7–12** — the piece-1/2/3 framing.
- **ConvMixedComposeBridge.lean:17** — "(… `Resnet34WholeFloatBridge`, was deleted with the whole-net budgets on 2026-09-08)".
- **SpecVJP.lean:439** `denoteConvnextT` — the "RESTORED 2026-08-30 … §2m/§2n had deleted it … §2k's own sin" paragraphs. Also **:288** (mobilenet "retired on 2026-09-19") and **:529** `vitVerified_denote_eq` ("Retires the rep tie's … caveats").
- **MLP.lean:83, :99** — "theorem (Phase 7) … promotes the previous vacuous `rfl`"; **:313** `mlpHasVJPAt` — "Replaces the vacuous `mlpHasVJP.correct := rfl`" (both defs still coexist). **Tensor.lean:557, :836** — "No longer an axiom", "(Phase 8, Tensor-level)".
- **BceLossCot.lean:13** — "Nothing said that chain is a loss's gradient — `…/proofs_tier_to_paper_nets.md` §3.5 lists … This is that item."
- **Bf16GradNodes.lean:11** — "'The bf16 twins consume the same node' was written in three fold headers and is false."
- **Binary32Instance.lean:11–26** — "Historically … Those axioms are now DISCHARGED (post_audit_roadmap §2)", "the 2026-06 audit's gaps 2 and 3". Keep the mathematical content (constructed `rndP` grids; named-model corollaries).
- **E4M3Fold.lean:3 / Bf16Fold.lean:3** — "PoC … (planning §3b)", "(planning §5)"; **Bf16Fold.lean:100** — "That op now exists … so this section closes the gap the header left open".
- **CertifiedChain.lean:174** — "(MobileNetV4's T2 took three minutes without these, seconds with them)"; **:40** — "(the §3 trap, one level up)" is a dangling reference.

---

### Missing documentation where it matters

### LeanMlir/Proofs/Foundation/Tensor.lean:277 — `structure HasVJP`

**Kind:** missing
**Says:** (no docstring; a `-- § VJP Framework` banner only)
**Actually states:** `backward : Vec m → Vec n → Vec m`, `correct : ∀ x dy i, backward x dy i = ∑ j, pdiv f x i j * dy j`. This is the central contract every `*HasVJP_correct` in the repo projects from.
**Fix:** "/-- A vector–Jacobian product for `f`: a backward map with the proof that at every `x` it contracts `f`'s Jacobian (`pdiv`) against the cotangent. `correct` is the whole content; `HasVJP.canonical` inhabits it for every `f`, so a witness is informative only when its `backward` is a hand-written formula tied to it. -/" The same applies to `HasVJPAt` (:400), which has only the preceding `/-!` block.

### LeanMlir/Proofs/Foundation/MLP.lean:20 — `dense`, `relu` (:130), `softmax` (:264), `oneHot` (:269), `crossEntropy` (:272), `mlpForward` (:291), `relu6` (:400)

**Kind:** missing
**Says:** (no docstrings)
**Actually states:** these are the core forward definitions every proof tier is about. `dense W b x j = ∑ i, x i * W i j + b j` uses the non-obvious row-vector convention: `W : Mat m n` maps `Vec m → Vec n`, so `W i j` is input `i` → output `j`.
**Fix:** add a one-liner to each; for `dense`: "/-- Affine layer `x ↦ xW + b` in the row-vector convention: `W i j` connects input `i` to output `j`. -/"

---

## Overclaims — fix before the published results are read again

1. **FloatSubnormalBridge.lean:3 / :166.** The prose claims the subnormal gap is closed, that activations are proved to stay normal, that binary32 is instantiated, and that the floor "cannot move any bound". What is proved: three denominator lemmas and one arithmetic inequality. No binary32 instance, no link to any budget.
2. **FloatBridge.lean:222 / :73 / :1268 / :1277 / module :26.** "Valid for every association", yet `dot_close` / `sum_close` state only the left fold, and `bnMean_close_of` says so itself.
3. **FloatClose.lean:25, :69 · FloatComposeBridge.lean:9, :22, :48, :163, :196 · DepthwiseFloatBridge.lean:173 · ConvMixedComposeBridge.lean:252.** "Whole-net float certificate", "THE FINAL FOLD", "is this instance", "ONLY thing the whole-net bf16 bound needed". No whole-net fold exists, `r34_float_close` is undeclared, and several of these instances have zero uses.
4. **MuonNewtonSchulz.lean:243 (+ module :11) · MuonGeometry.lean:199, :445.** "The thing the hardware computes is the thing the theory says is optimal." Proved only for the cubic and Higham's quintic; the same file proves Muon's tuned quintic fails the hypothesis.
5. **ConvMixedComposeBridge.lean:33.** "Non-vacuous at R50's n = 4608", contradicted by the same file's "BOTH BOUNDS ARE VACUOUS".
6. **MuonGeometry.lean:300.** "Muon's update is the steepest ascent … unconditionally", while the statement proves achievability only.
7. **FloatBridge.lean:1326, :1644.** The `exp` accuracy constant is "what `vjp_oracle` validates"; `vjp_oracle` diffs step-2 loss and measures no `exp` constant.
8. **Binary32Instance.lean:94.** "Makes precise why the deployed net needs only 0.38", but the theorem is a numeric bound at `m = 4`.
9. **SpecVJP.lean:151, :327, :378, :421, :477.** "The committed spec carries the math" for `HasVJP.canonical` witnesses, which exist for every function.
10. **IR.lean:827.** "The whole train step … is proof-backed end to end", from a one-equation loss-head bridge (and still names IREE).
11. **CertifiedChain.lean:6.** "Every net's whole backward is built this way", but EfficientNet-B0 does not use `CertLayer`.
12. **BnFloatBridge.lean:69.** "The non-vacuous BN certificate" for a conditional bound that nothing instantiates.
13. **Binary32Instance.lean:58.** "The bound is tight for the grid": unproved, and not attained.
14. **MLP.lean:65.** A `.correct` projection called a "whole-model VJP contract … capstone".
