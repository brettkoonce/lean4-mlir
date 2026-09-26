# Slice A — published documents: findings

**Coverage.** Read fully: README.md, CHANGELOG.md (v0.7.1 and v0.7.0 closely; older entries skimmed as history, where old names such as `swish_has_vjp_correct` are correct for their date and are not reported), LeanMlir/README.md, LeanMlir/NAMING.md, apps/README.md. blueprint/src/content.tex lines 1–8750: every `\lean{}`-cited theorem block (94 names) read and compared with the Lean statement; prose read around every theorem section (ch 1–7), the ch 1/2 MLIR and training-step sections, the ch 4 BN derivation, and the ResNet/MobileNet ImageNet sections. Other prose (recipe tables, run transcripts, TikZ) was skimmed by grep for claim words.
**Mechanical checks.** All 94 `\lean{}` names and all `\leandocref{}` targets in the slice resolve to live declarations. Every project identifier in `\texttt{}` in the slice exists (the non-hits are Mathlib names, MLIR ops, file names or variables). No old snake-case `*_has_vjp*` / `_diff` declaration names remain in the slice; the `hf_diff`/`hg_diff` hits are hypothesis names that match the Lean. Every NAMING.md example exists (`aczTFe4_0_12` and `hrelTFe4_0_12` are in `Certificates/LipschitzCertScorecardCrownUncon.lean`). README's "zero project axioms": grep for `axiom`, `sorry`, `admit`, `native_decide`, `implemented_by` and `@[extern]` in `LeanMlir/Proofs` found hits only in comments and docstrings. The comparator's `run.sh` also checks the axiom closure `[propext, Quot.sound, Classical.choice]`.

---

### blueprint/src/content.tex:8384 — Thm "Emitted train step is the certified step" (EfficientNet-B0; `EnetTiePoCG.efficientnet_net_tiedG`)

**Kind:** overclaim
**Says:** "the 262 parameter gradients of the batched EfficientNet-B0 step (`efficientnet_adam_train_step.mlir` and every `efficientnetin_*` artifact) are tied … The AdamW, RMSProp, EMA and data-parallel tails all consume that node." The sync twin (8408) also says "all 49 BatchNorms synchronised … equals the single-device node", with no scope.
**Actually states:** `efficientnet_net_tiedG (…) (N : Nat) (w : B0Weights) … (x : Vec (N * (3*224*224))) (t : Vec (N * (1 * 10)))`. The head is fixed at **10 classes**, and so is the VJP (`efficientnetForwardBFullHasVJP_correct … (dy : Vec (N * 10))`) and the sync twin (`g : Vec ((R*N) * 10)`). Every `efficientnetin_*` render is 1000-class (`tensor<1280x1000xf32>` in `efficientnetin_emarmsdp64dropdobf16_train_step.mlir`). The sync file's own docstring says it holds "without drop-path and dropout", and the shipped ImageNet run is the `…dropdobf16` render. None of the ImageNet B0 artifacts is covered by this theorem.
**Fix:** "…tied for the 10-class head (`efficientnet_adam_train_step.mlir` and the Imagenette renders). The `efficientnetin_*` ImageNet renders use a 1000-class head that this statement does not cover, and the data-parallel twin excludes drop-path and dropout." Alternatively, generalise `B0Weights` over `nCls`, as ResNet-34, ResNet-50 and MobileNetV2 already are.

### blueprint/src/content.tex:5212–5244 — Thms "ResNet-34 / ResNet-50 at batch N has the VJP" (`resnet34ForwardBFullHasVJPAt_correct`, `resnet50ForwardBFullHasVJPAt_correct`)

**Kind:** overclaim
**Says:** "Assume the eighteen smooth-point bundles: no ReLU pre-activation of the stem, of any residual block, or of the pooled features sits at zero for the batch. Prove the backward … equals the pdiv-contracted Jacobian of the forward at every input, cotangent and index." For ResNet-50: "the eighteen bottleneck bundles".
**Actually states:** `(hq : R34PosB w) (x) (hx : R34SmoothAtB N w x)`. `R34SmoothAtB` has fields `stem`, **`pool : R34PoolSmoothAt …`** and 16 block clauses. `R34PoolSmoothAt` is `∀ r, MaxPool3s2Smooth (…cbReluStridedB … x…)`: every 3×3 stem-pool window of the **post-ReLU** activation must have pairwise-distinct values. No ReLU acts on "the pooled features" (the docstring says "The head has none"). `R34PosB` (every ε > 0) is also dropped. `R50SmoothAtB` has the same `pool` field, so its "eighteen bottleneck bundles" are stem + pool + 16 blocks. After a ReLU, zeros repeat within a window whenever two or more of its entries were negative, so the pool hypothesis fails on ordinary inputs. The prose never mentions it, and "at every input" contradicts the pointwise statement.
**Fix:** "Assume every BatchNorm ε > 0 and, at this input, the eighteen smooth-point bundles: no ReLU pre-activation of the stem or of any block sits at zero, and every 3×3 window of the post-ReLU stem-pool input has pairwise-distinct entries (`MaxPool3s2Smooth`), a condition that ties from repeated zeros can violate. Prove that at this input, for every cotangent and index, the backward equals the pdiv-contracted Jacobian." Use the same wording for ResNet-50: "stem, stem-pool no-tie, sixteen bottleneck bundles".

### blueprint/src/content.tex:1535 (`LinPoC.poc_train_step_tail_certified`), 2389 (`MlpPoC.mlp_train_step_tied_certified`), 2147–2150, 2518–2520 — the small-net folds versus the committed batch-128 renders

**Kind:** overclaim
**Says:** (1535) "the two update operations of `linear_train_step.mlir` … denote W'ᵢⱼ = Wᵢⱼ − α·pdiv(L)…". (2147) "The constant 0.00078125 is α/N = 0.1/128 … Everything else in the listing is, line for line, the rendering of a theorem proved earlier in this chapter." (2518) "The six parameter outputs are the proved ones — `MlpFold` shows each denotes the certified gradient-descent step."
**Actually states:** Both theorems take a single example (`x : Vec m`, `label : Fin n`) and a real `lr`. `LinearFold`'s own docstring lists the residual: "**Single example (B = 1):** `wGrad x dy = x ⊗ dy`; the emitted module batch-contracts." `MlpFold` says "B=1 (the emitted module batch-contracts; `den` is per-example)". The committed `linear_train_step.mlir` takes `tensor<128x784xf32>`, contracts over the batch axis and scales by the literal `0.00078125`. No theorem covers the batch contraction, the batch sum or the α/N folding.
**Fix:** at 1535: "Prove, for one example (x, ℓ), that the two update operations denote … The committed render runs the same graph at batch 128, contracting the batch axis and scaling by α/128; that the batch sum is the mean-loss gradient is the standard identity, not part of this theorem." At 2150, replace "line for line, the rendering of a theorem" with "line for line the batch-128 rendering of graphs whose per-example meaning is proved earlier in this chapter". Scope 2518 the same way.

### blueprint/src/content.tex:5298–5310 — Thm "Emitted ResNet-50 step is the certified step" (`ResNet50TieB.r50_net_tiedB`)

**Kind:** stale
**Says:** "instantiated at the smoothed cross-entropy and at BCE-with-logits over N·K (`r50_lossCot_is_bce_grad`) — the loss of the 76.66% `resnet50in160_lambaccdp8x64bce` run."
**Actually states:** The theorem holds and the BCE corollary exists, but 76.66% appears nowhere else in the book. The ResNet-50 run the book reports (§ r50_pjrt, 6333–6348; README; CHANGELOG v0.7.1) is 77.98% on `resnet50in160_lambaccdp8x64wxclipbcebf16`, a bf16 render with weight-decay exclusion and clipping. The theorem block therefore cites a retired run and the wrong artifact.
**Fix:** "…instantiated at BCE-with-logits over N·K (`r50_lossCot_is_bce_grad`), the loss every `resnet50in160_*bce*` render trains, including the A3 run of §\ref{sec:r50_pjrt}." If the bf16 gradient nodes are covered, cite that tie (`Bf16GradNodes`); if they are not, say so.

### blueprint/src/content.tex:675–683 — ch 1, "The correctness field is the whole point"

**Kind:** overclaim
**Says:** "The backward function is the kernel that ships into the codegen, and the proof is the guarantee that what it computes matches what its name claims. Every later chapter proves a closed-form HasVJP for a layer family … codegen reads off the `backward` field and emits a single fused StableHLO kernel."
**Actually states:** `HasVJP.canonical f` exists for every `f` (`backward := ∑ pdiv f x i j * dy j`, `correct := rfl`, Tensor.lean:285). `reluHasVJP`, `mlpHasVJP` and `maxPool2HasVJP3` are canonical witnesses; the book says so at 2339 and 3240. Their `backward` is noncomputable `fderiv` and cannot be "read off" by codegen. The emitted backward is hand-written IR (`selectPos`, `maxPoolBack`, …) and is tied to the math by separate bridge theorems, which for the kinked operators hold only at smooth points. "You don't get the record without the proof" is true but vacuous: the canonical witness is always available.
**Fix:** "For smooth layers (dense, conv, BN, softmax, …) the record's `backward` is a closed form, and the emitted backward is proved to denote it. For kinked operators (ReLU, ReLU6, max-pool) the global record is the canonical pdiv witness, which says nothing about the kink. The emitted subgradient is tied to the derivative by a pointwise bridge that holds where no pre-activation sits at a kink and no pooling window ties."

### blueprint/src/content.tex:3246–3262 (`mnistCnnNoBnHasVJPAt_correct`) and 3695–3697 (ch 3 MLIR caveat)

**Kind:** overclaim
**Says:** "every 2 × 2 window the max pool sees has a strict maximum [`h_mp`]". At 3695: "Max-pool needs a unique argmax. The bridge holds only where every 2×2 window has a strict argmax."
**Actually states:** `h_mp : MaxPool2Smooth (…)` requires all four entries of each window to be **pairwise distinct** (CNN.lean:890: `∀ … ab ≠ ab' → x … ab ≠ x … ab'`). That is strictly stronger than a unique maximum. The pool reads post-ReLU activations, so any window with two or more clipped zeros (for example `[5, 0, 0, 0]`, which has a strict argmax) violates it.
**Fix:** "every 2×2 window the max pool sees has four pairwise-distinct entries (stronger than a strict maximum; post-ReLU windows with two or more zeros violate it)." Mirror the wording in the 3695 caveat.

### blueprint/src/content.tex:2389–2397 — Thm "Emitted train step is the certified step" (MLP; `MlpPoC.mlp_train_step_tied_certified`)

**Kind:** overclaim
**Says:** "…six update operations … and each denotes θ − α ∂L/∂θ: the output layer's against the gradient of the whole loss, the two hidden layers' against the certified per-layer VJP at the chain cotangent."
**Actually states:** Only the W₂ conjunct is stated against `pdiv` of the whole loss. The four hidden-layer conjuncts equal `θ − lr·∑ pdiv(layer)·(mlpCotOut{1,0}).denote g`, where the cotangent is the emitted `selectPos`/`dotOut` chain. The theorem has no ReLU smooth-point hypothesis, so it does not state that this cotangent equals ∂L/∂(pre-activation). For the hidden layers it is not θ − α ∂L/∂θ.
**Fix:** "…each denotes a gradient step: the output layer's against the gradient of the whole loss, and the hidden layers' against the certified per-layer Jacobian contracted with the cotangent that the emitted ReLU-masked backward chain delivers. At smooth points (Theorem~\ref{thm:mlpHasVJPAt}) that cotangent is the loss gradient."

### blueprint/src/content.tex:1592–1595 — Thm "SGD descends" (`linear_float_sgd_descends`)

**Kind:** overclaim
**Says:** "The same holds with the binary32 gradient in place of g (`linear_float_sgd_descends`), its η the rounding budget of §precision rather than an assumption."
**Actually states:** The theorem is over an abstract `M : FloatModel`, not IEEE binary32 itself. η is derived, but from two new hypotheses that the statement assumes: `hfexp : ∀ t, |fexp t − exp t| ≤ eexp·exp t` (the GPU exp's accuracy) and `hδ : ∀ k', |M.dense W b x k' − dense W b x k'| ≤ δ` (an a-posteriori bound on logit drift), plus `hρ1`. The book's own caveat at 2765 says the float layer is "conditional on two measured constants".
**Fix:** "…with the rounded gradient of an abstract rounding model in place of g (`linear_float_sgd_descends`). Its η is computed from the rounding budget of §precision, given an exp accuracy `eexp` and a logit-drift bound `δ` that are measured, not proved."

### README.md:100–102 — "The proofs", the comparator

**Kind:** overclaim
**Says:** "`tests/comparator/run.sh` re-runs Lean's kernel typechecker over the headline theorems independently, and `tests/comparator/Challenge.lean` imports Mathlib and nothing else, so those can be read and checked without reading a line of this project."
**Actually states:** `Challenge.lean` holds the 13 architecture-free calculus theorems. The headline theorems (the whole-net VJPs, step ties, sync twins and seals) are in `ChallengeArch.lean` (39) and `ChallengeTier.lean` (35). Both import `LeanMlir.Proofs.*` for their statements' vocabulary, so reading them means reading the project's definitions.
**Fix:** "`tests/comparator/run.sh` re-checks the headline theorems with Lean's kernel, independently of the elaborator. Its 13 calculus statements (`Challenge.lean`) import only Mathlib; the architecture and tie statements import the project's definitions, which a reader must still trust as the intended networks."

### README.md:80–86 — "The proofs"

**Kind:** overclaim
**Says:** "Every layer's backward is proven to be the Jacobian-transpose of its forward over the exact reals … composed up to whole-network VJPs for ResNet-34, MobileNetV2, EfficientNet-B0, ConvNeXt-T and ViT-Tiny … For every chapter net the committed train-step render in `verified_mlir/` is tied … each emitted parameter-update node denotes the certified descent step, and the tiers train on exactly those bytes."
**Actually states:** (a) The kinked operators are proved only at smooth points (see the ReLU/max-pool findings above). (b) Whole-net VJPs also exist for ResNet-50 and MobileNetV4 (`resnet50ForwardBFullHasVJPAt_correct`, `mobilenetv4ForwardBFullHasVJPAt_correct`), which this list omits. (c) The EfficientNet-B0 tie is pinned at 10 classes, so the tier-4 `efficientnetin_*` renders it trains are not covered (first finding). (d) The ties state that gradient nodes denote the certified gradient (θ − α·∇ for SGD). Descent theorems exist only for the small nets under step-size hypotheses, so "descent step" suggests a guarantee that the ties do not give.
**Fix:** "Every layer's backward is proven to be the Jacobian-transpose of its forward over the reals, everywhere for smooth layers and at smooth points for ReLU, ReLU6 and max-pool. These compose to whole-network VJPs for ResNet-34/50, MobileNetV2/V4, EfficientNet-B0, ConvNeXt-T and ViT-Tiny. For the chapter nets' committed renders, each emitted gradient node denotes the certified gradient (B0's tie is at the 10-class head)…"

### blueprint/src/content.tex:3855–3940 — ch 4 BN derivation versus the CIFAR renders

**Kind:** overclaim
**Says:** "You also jiggle the batch mean μ … the gradient of one sample depends on every sample's centered value …" (3860–3905) and "The value of the formal proof is that the Lean kernel mechanically verifies we are computing the right thing at every training step." (3938)
**Actually states:** The CIFAR "BatchNorm" that ch 4 trains (`.bnPerChannel`, `bnPerChannelTensor3`) normalises each channel over the **h·w spatial cells of one example**. `cifar8w_bn_mom_train_step.mlir` reduces over dimensions `[2, 3]` of `tensor<128x16x32x32>` and never over the batch axis. The theorems are over an abstract n-vector, so they are correct, but here n is spatial positions, not samples. The kernel checks the theorems once at build time over ℝ; it does not verify anything at a training step.
**Fix:** Add one sentence where the derivation starts: "In this chapter's CIFAR nets the n entries are the h·w positions of one channel of one image. The statistics are per example, and batch statistics arrive with Chapter~\ref{chap:residual}'s batched renders." At 3938, write "…is that the backward the render emits is proved, once and over ℝ, to be this expression."

### blueprint/src/content.tex:83–84 — Introduction

**Kind:** overclaim
**Says:** "…training ResNets on a single GPU, with machine-checked proofs that every gradient is correct."
**Actually states:** The proofs are over ℝ. The kinked-operator bridges hold at smooth points. Per-op text printing and the lowerer are trusted, which README line 85 says openly.
**Fix:** "…with machine-checked proofs that every gradient the network emits is, over the reals and away from the kinks, the true derivative."

### CHANGELOG.md:89–90 — v0.7.0

**Kind:** wrong
**Says:** "…and trains RSB-A3 on the verified path to **78.26%**, ahead of its JAX reference."
**Actually states:** In the book (6269, 6381) and in CHANGELOG v0.7.1 (line 11), 78.26 is the **JAX reference**; the verified path's A3 number is 77.98. The v0.7.0 entry swaps the columns and claims a win that the published table shows as −0.28.
**Fix:** "…and the RSB-A3 recipe reaches **78.26%** on the JAX reference (the verified path's run follows in v0.7.1)."

### blueprint/src/content.tex:5125–5127 and 450–451 — ch 5 "Why the proof is a one-liner"; Roadmap

**Kind:** stale
**Says:** "That is Theorem~\ref{thm:residualHasVJP} below, and its proof is one line. This chapter has exactly one theorem, and that theorem is mechanical." Roadmap: "Target: ResNet-34 … Adds one theorem on top of CNN+BN."
**Actually states:** Chapter 5's theorem section has 11 blocks: residual, the R34 and R50 whole-net VJPs, the R50 backward chain, two step ties, two seals and two sync twins. The chapter table at 406 counts 10 contracts, 3 witnesses and 5 steps.
**Fix:** "The residual VJP is one line; that is Theorem~\ref{thm:residualHasVJP}. The rest of this chapter's theorems compose it, with no new calculus, into the full ResNet-34 and ResNet-50 and tie them to the committed renders." In the Roadmap: "Adds one new VJP, the residual fan-in, on top of CNN+BN."

### blueprint/src/content.tex:6103–6105 — ch 5 ResNet-34 verified path

**Kind:** stale
**Says:** "The proof-carrying tier stops at Imagenette, so the claim is 'one architecture, two independent lowerings, agreeing', not 'proven at ImageNet scale'."
**Actually states:** Theorem~\ref{thm:resnet34_step_tie} (5277) names `resnet34in_momdp64` (the ImageNet render) at N = 64. The sync twin (5355) ties the data-parallel ImageNet step, and the seals are on the full-width 224² net. The proof tier now reaches the ImageNet renders.
**Fix:** "The ties reach this render (Theorems~\ref{thm:resnet34_step_tie} and \ref{thm:resnet34_sync_tie}); what the proofs do not reach is the float arithmetic and the lowerer, so on that side the evidence is two independent lowerings, agreeing."

### README.md:53–54 — The tour, tiers 3 and 4

**Kind:** stale
**Says:** Tier 3 "R34 89.50"; tier 4 "R34 74.06 … on the verified path". The paragraph opens "The numbers are the book's".
**Actually states:** The book's canonical R34 Imagenette number is 89.99 ± 0.32 over five seeds (5088, 5574, 6670, 8629; 89.50 appears only as one curve point at 7591). The book's R34 ImageNet verified number is 74.17% / 91.89% (6090, 6127).
**Fix:** "R34 89.99" (or "89.99 ± 0.32, five seeds") and "R34 74.17".

### blueprint/src/content.tex:406 — Theorem-budget table, ch 5 row

**Kind:** stale
**Says:** "…whole-net `cnn`; ResNet-34 per example, **full-depth ResNet-34 and ResNet-50** at batch statistics…"
**Actually states:** `Proofs/Nets/ResNet/` has no per-example ResNet-34 whole-net VJP. Every ResNet-34 file is batched (`*B`), and CHANGELOG v0.7.1 says "the superseded tiers — the per-example ResNet-34 and MobileNetV2 … retire."
**Fix:** Drop "ResNet-34 per example" and re-derive the contract count for the row.

### blueprint/src/content.tex:6347–6362, 6381–6386, 6409 — ResNet-50 verified path

**Kind:** stale
**Says:** The 2018 cell is 77.07 with BN group "64 (per replica)", and "What it does not carry is the reference's BatchNorm group: no collective touches the batch statistics…" `\globalbntodo` sits at 6409.
**Actually states:** Commit 82445a97 ran the 2018 recipe on the committed sync-BN render `resnet50in_momdp64bf16` (BN over 256) to 77.160 / 93.430, and its message cites 3af1cc9c for A3 at the reference's group (78.330). The commit message says the content.tex edits at these lines are left to the author. Flagged so the table and the "no collective" sentence are not published as they stand. README tier 4 (R50 77.98) moves with the table.
**Fix:** Update the two verified cells, the BN-group row and the Δ line to the sync-BN runs; delete the "no collective touches the batch statistics" sentence and `\globalbntodo`.

### LeanMlir/README.md:16 — module table

**Kind:** missing
**Says:** "`Blackjack`, `Cam`, `Ddpm` | support for the Chapter 10 demos"
**Actually states:** `LeanMlir/Pong.lean` (the Pong game `pong-dqn` and `pong-env` use) and `LeanMlir/FloatFmt.lean` (shared `fmt`) were added in 290187b1 and appear in no row.
**Fix:** "`Blackjack`, `Pong`, `Cam`, `Ddpm`, `FloatFmt` | support for the Chapter 10 demos".

### apps/README.md:79 — `tools/` row

**Kind:** stale
**Says:** "`score-checkpoint`: rescore a saved ConvNeXt or ViT checkpoint"
**Actually states:** `apps/tools/MainScoreCheckpoint.lean`: "▶ Every net scores. The LayerNorm nets (ConvNeXt, ViT) carry their whole eval state in the blob; the BN nets read their running statistics from the `<ckpt>.bn` companion…"
**Fix:** "`score-checkpoint`: re-score any verified net's saved checkpoint (BN nets read the `.bn` companion)".

---

## Overclaims (fix before the published results are read again)

1. content.tex:8384 (and 8408; README:83): the B0 step tie claims "every `efficientnetin_*` artifact". The theorem is pinned at 10 classes, and the ImageNet renders are 1000-class and, for the shipped run, drop-path/dropout/bf16.
2. content.tex:5212–5244: the R34/R50 whole-net VJPs describe the stem-pool no-tie hypothesis as a ReLU clause on "pooled features". It is a pairwise-distinct `MaxPool3s2Smooth` on post-ReLU activations, and ε > 0 is also dropped. "At every input" overstates a pointwise theorem.
3. content.tex:1535, 2147–2150, 2389, 2518: the linear/MLP folds are per-example (B = 1), but the prose says the batch-128 renders are "line for line the rendering of a theorem".
4. content.tex:675–683: "Every later chapter proves a closed-form HasVJP … codegen reads off the backward field". The kinked witnesses are `HasVJP.canonical`; the emitted backward is tied by separate pointwise bridges.
5. content.tex:3246, 3695: "strict maximum / unique argmax" understates `MaxPool2Smooth` (pairwise distinct), which post-ReLU zero ties violate.
6. content.tex:2389: the MLP fold says every update is θ − α ∂L/∂θ. The hidden layers are stated only at the emitted chain cotangent, with no smoothness link to ∂L.
7. content.tex:1592: `linear_float_sgd_descends` is described as η "rather than an assumption". It assumes an exp accuracy and a logit-drift bound, and works over an abstract `FloatModel`.
8. README.md:100–102: says the headline theorems can be checked "without reading a line of this project". Only the 13 calculus statements import Mathlib alone.
9. README.md:80–86: says every layer's backward is proven (no smooth-point caveat) and every chapter render is tied to a "descent step". Omits R50/MNv4, and B0 at ImageNet is not covered.
10. content.tex:3855–3940: the ch 4 BN derivation is about batch statistics, but the CIFAR renders normalise per example over spatial cells. It also says "the Lean kernel … verifies … at every training step".
11. content.tex:83: "machine-checked proofs that every gradient is correct".
12. CHANGELOG.md:89: v0.7.0 credits the verified path with 78.26% "ahead of its JAX reference". That is the reference's number; the verified run was 77.98.
