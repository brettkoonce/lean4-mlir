# Slice B: blueprint/src/content.tex 8751–17500 + formalization.yaml

**Coverage.** I read these closely: content.tex 8751–11680 (the end of EfficientNet, the ConvNeXt chapter, the ViT theorems) and 16446–17446 (Getting started, On Verification), plus all of formalization.yaml. I skimmed 11680–12420 (ViT example, MLIR and ImageNet), the Bestiary (12419–15993) by grepping it for claim words and reading the hits, and Data availability. I checked all 70 `\lean{}` cites in the slice: every one resolves, and the 22 `Layer.*` cites are constructors in LeanMlir/Types.lean. I also grepped every identifier-shaped `\texttt{}` name against the repo's declarations and found none left over from the old naming. For "no sorry / no axioms" I grepped `sorry|admit`, `^axiom`, `native_decide`, `implemented_by` and `@[extern` over LeanMlir/ and tests/. The only hits were in comments, so the zero-sorry claim holds as text. I did not run `#print axioms`.

---

### blueprint/src/content.tex:16990 — On Verification §"Verified code generation", closing sentence of "Proven versus trusted"

**Kind:** overclaim
**Says:** "So the gradient the GPU computes is, by a machine-checked theorem, the network's exact reverse-mode derivative over $\mathbb{R}$, up to one printer, one lowerer, and floating point. … the unproven surface here is a single printer, tested end to end."
**Actually states:** The list directly above (16980–16988) names three trusted items: a formal StableHLO semantics (which "does not yet exist"), the lowerer, and rounding. The sentence also leaves out hypotheses that sit inside the statements. The ReLU, ReLU6 and max-pool bridges hold only at smooth points (`relu_back_bridge` takes `h_smooth : ∀ k, x k ≠ 0`; `MaxPool2Smooth` and `MaxPool3s2Smooth` require all window entries pairwise distinct). Text lexing is not proved (`StableHLO.roundtrip` is `parse (toToks (skel a)) = some (skel a)`, a token-level result; StableHLOLex.lean proves only `parseNat_toString`). The step ties are one-replica, stated at drop-free chains (`cnx_net_tiedGB` and `vit_net_tiedGB` docstrings say so), and bf16 artifacts emit different `*GradBBf16` nodes. The theorems are about `den` of a Lean AST, not about text or hardware.
**Fix:** "So at every smooth point, the denotation of the graph the printer walks is the network's exact reverse-mode derivative over ℝ, by a machine-checked theorem. The trusted surface around that is the printer (tokens to text: the token round-trip is proved, lexing is not), a StableHLO semantics that does not yet exist formally, the lowerer, and floating point. Drop-path and bf16 variants are covered at their gradient nodes but not by a whole-step tie."

### blueprint/src/content.tex:16811 (also 10583, 12394) — On Verification opener; ViT "What's actually proved"; "What Part 1 has established"

**Kind:** overclaim
**Says:** (16811) "Every VJP correctness theorem is machine-checked … and that covers dense layers, convolution, batch normalization, … self-attention. If it builds, it's correct." (10583–10589) "every architectural piece used by any Part-1 network has a machine-checked backward. Once we land Theorem vitTinyHasVJP_correct, the sentence 'the gradient computed by this trainer is the mathematically correct one' … is a theorem the Lean kernel verifies on every build." (12394) "Every layer primitive shipped in Part 1's trainers … has a machine-checked backward pass."
**Actually states:** `vitTinyHasVJP_correct` is the `.correct` projection of `vitForwardKVHasVJP` (ViTDepthK.lean:256). It is a statement over ℝ about a Lean witness at 10 classes and one example. It says nothing about "the gradient computed by this trainer", which is Float32 or bf16 on a GPU, runs through the printer and lowerer, and at ImageNet scale uses drop-path and bf16 artifacts the ties do not name. For the kinked primitives shipped in Part 1 (ReLU, ReLU6, `.maxPool`), the backward that matches the emitted code is proved only under smooth-point hypotheses. The global `reluHasVJP`, `mlpHasVJP` and `maxPool2HasVJP3` are `HasVJP.canonical` / pdiv-sum witnesses whose `correct` holds by `rfl`.
**Fix:** (10583) "…every architectural piece used by any Part-1 network has a machine-checked backward over ℝ, unconditionally for the smooth ones and at smooth points for ReLU, ReLU6 and max-pool. Theorem vitTinyHasVJP_correct makes 'the composed backward is the Jacobian transpose' a kernel-checked theorem for ViT-Tiny; what ties that to the bytes the trainer runs is the step tie and the trusted printer and lowerer (Appendix app:verification)." (16815) Drop "If it builds, it's correct" or replace it with "If it builds, every stated equation holds under its stated hypotheses."

### blueprint/src/content.tex:9616 — Theorem thm:convnext_step_tie (`Proofs.CnxTiePoCGB.cnx_net_tiedGB`); also 9653 thm:vit_step_tie; formalization.yaml:166

**Kind:** overclaim
**Says:** (9619–9621) "each raw gradient node of convnext_adam_train_step.mlir and of every convnextin_* artifact denotes the certified batch gradient." (9653) "Every vitin_* artifact renders from this chain." (yaml, cnx_net_tiedGB comment) "ConvNeXt-T's 182 parameters, every shipped ConvNeXt artifact."
**Actually states:** The `cnx_net_tiedGB` docstring (ConvNeXtStepTieGB.lean:364–368) says "⛔ ONE REPLICA … ⛔ Stated at the drop-free chain; the `*drop*` artifacts' … cotangent chain carries the `dropPathB` sites, which this thread does not name." ConvNeXtFoldGB.lean:45 adds that "the bf16 artifacts … emit `*GradBBf16` constructors, not these nodes." `vit_net_tiedGB` (ViTStepTieGB.lean:369–371) carries the same "ONE REPLICA … stated at the drop-free chain" caveat. The runs the book reports are `convnextin_adamdpwxclipdropbf16` (10132) and `vitin_emadp128x4wxclipdropbf16` (12198). Both are data-parallel, drop-path and bf16, which is the combination the whole-step theorem does not cover.
**Fix:** "…each raw gradient node of convnext_adam_train_step.mlir and of the fp32, drop-free convnextin_* artifacts denotes the certified batch gradient on one replica. The data-parallel renders compose this with allReduceMeanF. The drop-path and bf16 artifacts, including the one the ImageNet run used, are covered only at their gradient nodes (the ∀-cotangent folds and Bf16GradNodes.lean), not by this whole-step tie." Make the same change for ViT and for the yaml comment ("every drop-free fp32 ConvNeXt artifact, one replica").

### blueprint/src/content.tex:17030 — On Verification "The one conditional"; formalization.yaml:221–226 (fidelity 2–3)

**Kind:** overclaim
**Says:** "For the kinked operators, which are ReLU, ReLU6 and max-pool, it holds only at a smooth point, where no pre-activation sits exactly on the kink (… an argmax tie for max-pool). The equality is permitted to fail precisely on that measure-zero set, and nowhere else." The yaml says "…no pooling tie … the measure-zero non-smooth points are (2)."
**Actually states:** `MaxPool2Smooth` (CNN.lean:890) and `MaxPool3s2Smooth` (MaxPool3s2.lean:169) require every pair of window entries to be distinct, not just "no tie at the argmax." Where max-pool follows ReLU (the ResNet stem), windows with two or more exact zeros are routine, so the hypothesis fails on a set that is far from negligible in practice. It is measure-zero only in pre-ReLU input space.
**Fix:** "…for max-pool, it holds when every window's entries are pairwise distinct. Before the pool that set is measure-zero. After a ReLU it is not: two zeros in one window, which is common, falls outside the theorem."

### formalization.yaml:225–226 — fidelity 3

**Kind:** wrong
**Says:** "The global instances (`reluHasVJP`, `mlpHasVJP`, `maxPool2HasVJP3`) are codegen-shaped witnesses whose `correct` closes by `rfl`."
**Actually states:** `reluHasVJP n := HasVJP.canonical _` (MLP.lean:222) and `mlpHasVJP := HasVJP.canonical _` (MLP.lean:311), and `maxPool2HasVJP3`'s backward is the `∑ pdiv3 …` contraction (CNN.lean:781–785). These are the canonical fderiv-shaped witnesses, not codegen-shaped ones. That is why `correct` is `rfl`: it is a tautology. The codegen shape (mask or argmax select) meets them only through `relu_codegen_matches_canonical` / `maxPool2_codegen_matches_canonical`, at smooth points.
**Fix:** "The global instances … are the canonical pdiv-derived witnesses (`HasVJP.canonical`), whose `correct` is `rfl` by definition. The emitted compare/select backward equals them only at smooth points (2)."

### blueprint/src/content.tex:16960 and 16982 — "A computable printer" and "Proven versus trusted"

**Kind:** overclaim
**Says:** (16960–16963) "The tier where the printed text is the denoted graph's own printout, with the parse-back proved (`StableHLO.roundtrip`)…" (16982–16984) "…the emitted text lexes and parses back to the proven op-graph (StableHLOLex.lean, StableHLOParse.lean)."
**Actually states:** `theorem roundtrip (a : SHlo k) : parse (toToks (skel a)) = some (skel a)` (StableHLOParse.lean:220) is about the token list, not the text. The StableHLOLex.lean header says the lexical edge, `parse (lex (pretty g)) = some (skel g)`, is "the remaining trusted edge". The file proves only the numeric keystone `parseNat_toString`.
**Fix:** "…with the token-level parse-back proved (`StableHLO.roundtrip`: the op skeleton is recovered from its token stream). Lexing the emitted text back into tokens is not yet proved; StableHLOLex.lean holds only its decimal-number keystone."

### blueprint/src/content.tex:11488 — Theorem thm:transformerTowerHasVJPMat (also thm:vitBodyHasVJPMat at 11518 and thm:transformerBlockHasVJPMat)

**Kind:** overclaim
**Says:** "\Prove{} HasVJPMat of the k-block tower, for every k --- ViT-Tiny/Base (k = 12) and Large (k = 24) are instances." At 11518: "the full ViT transformer backbone finalLN ∘ transformerTower is one HasVJPMat." The Block, Tower and Body statements assume nothing.
**Actually states:** `transformerTower` (Attention.lean:1522) is `Nat.rec` of one block with a single shared parameter tuple, and the LayerNorm affines are scalars (`γ1 β1 γ2 β2 : ℝ`). The file itself says "we use a single shared parameter tuple across blocks (a mild simplification; in practice every block has its own weights)". No real ViT is an instance. All three defs also take `hε : 0 < ε`. The yaml (alignment row 398) gets this right: "the weight-shared scalar-LN tower".
**Fix:** "\Assume{} ε > 0. \Prove{} HasVJPMat of the k-fold iterate of one block, with one parameter tuple shared across blocks and scalar LayerNorm affines, for every k. Real ViTs have distinct per-block weights and vector affines; that network is Theorem thm:vitForwardKVHasVJP." Add the same ε assumption to the Block and Body statements.

### blueprint/src/content.tex:9362 — ConvNeXt "Run it first" (also 9816–9820, 10488 for ViT)

**Kind:** stale / overclaim
**Says:** "every LayerNorm in it backpropagates by Theorem thm:layerNormHasVJP." At 9819: "Theorem thm:layerNormHasVJP wants ε > 0 at each [of the 23 LayerNorms]." For ViT at 10488: "every LayerNorm by Chapter chap:layernorm's."
**Actually states:** `layerNormHasVJP (n) (ε γ β : ℝ) (hε)` is LayerNorm with scalar γ and β, which is `bnForward` definitionally (LayerNorm.lean:75–93). The ConvNeXt that trained uses channel LayerNorm with a `Vec c` affine (`chanLNTensor3`, witness `chanLNTensor3HasVJP`, ChannelLN.lean:134). ViT uses `layerNormVecHasVJP` (LayerNorm.lean:437). Neither is the cited theorem.
**Fix:** "every LayerNorm in it backpropagates by its channel-LN VJP (`chanLNTensor3HasVJP`, the per-position vector-affine form of Theorem thm:layerNormHasVJP)". Either state thm:layerNormHasVJP's scalar affine in its statement, or cite the vector and channel forms.

### blueprint/src/content.tex:11906 — "MLIR: Attention", "The gap and how we close it"

**Kind:** overclaim
**Says:** "The emitted graph denotes `transformerAttnSublayerHasVJPMat`'s backward."
**Actually states:** Nothing outside Attention.lean references `transformerAttnSublayerHasVJPMat` (grep), and it is the scalar-LN, `Mat`-level witness. No theorem connects any emitted graph to it. The rendered block backward is tied to `vitBlockBackV_eq_transformerBlockV_vjp` (ViTVecLNBackCertifiedTie.lean:142) and the step tie `vit_net_tiedGB`.
**Fix:** "The emitted graph's block backward is pinned to the vector-LN block VJP by `vitBlockBackV_eq_transformerBlockV_vjp`; `transformerAttnSublayerHasVJPMat` is the scalar-LN statement of the same sublayer."

### blueprint/src/content.tex:16522 — Getting started, Track 1 Level 3 (comparator split)

**Kind:** overclaim
**Says:** "Thirty-nine more live in ChallengeArch.lean …, because each is a statement about a specific network, and 'ResNet-34's rendered backward equals its Fréchet derivative' cannot be phrased without ResNet-34 in scope. Once the fderiv pin and the structural rules are checked …, the architecture theorems are applications of them, and what you must additionally trust is the forward functions and nothing else."
**Actually states:** ChallengeArch.lean contains no ResNet-34 statement; the ResNet-34 tie `chk_r34InputGradB_eq_r34B_full_vjp` is in ChallengeTier.lean. Its whole-network rows (`chk_cnnHasVJPAt_correct`, `chk_convnextHasVJP_correct`, `chk_efficientnetHasVJP_correct`, …) are two-block toy nets. They are `.correct` projections of Lean witnesses, not statements about a rendered backward, and several carry `MaxPool2Smooth` or relu smoothness hypotheses. The connection to what runs is the ChallengeTier ties plus the trusted printer and lowerer.
**Fix:** "…each is a statement about a specific (small, two-block) network's Lean backward. What you must additionally trust for these is the forward definitions and the stated hypotheses. The link from those backwards to the rendered program is ChallengeTier.lean's ties, and past that, the printer and the lowerer."

### blueprint/src/content.tex:16504, 16529–16537, 16898–16920 — comparator and audit counts

**Kind:** stale
**Says:** "re-runs Lean's kernel typechecker independently over 73 theorems"; "The last 21 are ChallengeTier.lean"; "With the four that ChallengeArch.lean already holds, these are exactly the declarations formalization.yaml puts forward"; "three smooth-point pointwise variants (reluHasVJPAt_correct, mlpHasVJPAt_correct, maxPool2HasVJPAt3_correct)"; "tests/AuditAxioms.lean prints the axiom closure of 1,374 declarations"; "What the 73 add…"
**Actually states:** Challenge.lean has 13 theorems, ChallengeArch.lean 39 and ChallengeTier.lean 35, for 87. That matches the 87 in the config*.json `theorem_names` and in formalization.yaml:69 and :370. Of the yaml's 39 main_results, 35 are checked in config-tier, 3 in config-arch and 1 (`pdiv_comp`) in config.json, so "four in ChallengeArch" is wrong. ChallengeArch has seven `*HasVJPAt*_correct` pointwise variants, not three (it adds cnn, mobilenetv2, convnext and efficientnet). tests/AuditAxioms.lean has 1,611 `#print axioms` lines, and tests/AuditAxiomsHeavy.lean has 76 more.
**Fix:** Change 73 to 87 and 21 to 35. Say "with the three ChallengeArch.lean and one Challenge.lean already hold". List the seven pointwise variants or say "the smooth-point `*HasVJPAt*_correct` variants". For AuditAxioms, write "about 1,600 declarations (plus AuditAxiomsHeavy)", or follow the prose's own advice at 16878 and say "the live count".

### blueprint/src/content.tex:10403, 10549, 10559, 10565 — ViT chapter theorem counts

**Kind:** stale
**Says:** "the sixteen matrix-machinery lemmas"; "Why this chapter has thirty-seven theorems"; "(Matrix-level machinery, sixteen theorems)"; "(Attention proofs, twenty-one theorems)".
**Actually states:** §Matrix-level machinery (10598–10983) has 14 theorem/definition environments. §Attention proofs (10984–11660) has 25. That makes 39.
**Fix:** "fourteen", "twenty-five", "thirty-nine". Alternatively, drop the numerals from the section title.

### blueprint/src/content.tex:10568 — "Why this chapter has thirty-seven theorems"

**Kind:** wrong
**Says:** "Row-wise softmax … has the well-known closed-form Jacobian \(p_i(\delta_{ij} - p_i)\)"
**Actually states:** `pdiv_softmax`: `pdiv (softmax c) z i j = softmax c z j * ((if i = j then 1 else 0) - softmax c z i)`, i.e. \(p_j(\delta_{ij} - p_i)\). Theorem ax:pdiv_softmax at 10998 prints it correctly.
**Fix:** "\(p_j(\delta_{ij} - p_i)\)".

### blueprint/src/content.tex:17012 — "From operators to whole training steps"

**Kind:** overclaim
**Says:** "…that file is the printout of one graph whose denotation a faithfulness theorem proves equal to the certified loss-descent step, output by output."
**Actually states:** The faithfulness and tie theorems equate each output to the certified gradient, or to the optimizer step at that gradient. Descent, meaning that the step decreases the loss, is proved only for the shallow nets. §Finite precision (17293–17295, 17407) says so: "closes only for the shallow nets … all five deep nets stay closeness-only."
**Fix:** "…proves equal to the certified training step (the optimizer applied to the certified gradient), output by output."

### blueprint/src/content.tex:9905 — "MLIR: Layer Scale"

**Kind:** wrong
**Says:** "Uniquely in this book, every one is unconditional (the only hypothesis anywhere is LayerNorm's ε > 0)."
**Actually states:** ViT's chain is equally unconditional apart from ε > 0 (`vitForwardKVHasVJP_correct`, and the chapter says so itself at 10499–10504).
**Fix:** "As in ViT (Chapter chap:attention), every one is unconditional…"

### blueprint/src/content.tex:16668 — Track 4 "This tier needs Python"

**Kind:** overclaim
**Says:** "…so the augmentation the verified trainer consumes is provably the augmentation its JAX reference trains on."
**Actually states:** No theorem covers the shim. It is generated from the same `TrainConfig`, which makes the match hold by construction. In a book where "proved" means Lean-checked, "provably" reads as a theorem.
**Fix:** "…is by construction the augmentation its JAX reference trains on."

### blueprint/src/content.tex:14369, 14373 — Bestiary, DQN on blackjack

**Kind:** overclaim (minor)
**Says:** "The ceiling of the training curve is a theorem about the rules, not a number somebody measured." Also: "Every arm scored against the theorem."
**Actually states:** The optimum comes from value iteration run in code. No Lean theorem states it.
**Fix:** "…is computed exactly from the rules by value iteration, not measured." Also: "Every arm scored against the exact optimum."

### blueprint/src/content.tex:10386 — ViT chapter opener

**Kind:** stale (minor)
**Says:** "Each one slotted into the VerifiedNetSpec type and earned its own `hasVJP` theorem"
**Actually states:** No declaration is named `hasVJP`. The per-layer witnesses are `noncomputable def`s named `…HasVJP` (e.g. `layerNormHasVJP`, `geluHasVJP`), each with a `…HasVJP_correct` theorem.
**Fix:** "…and earned its own `HasVJP` witness".

---

## Leads outside this slice (Lean docstrings, for the Lean-file auditors)
- LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtFoldGB.lean:52 says "`convnextin_adamdpwxclipdrop`, whose accuracy the book quotes". The book (content.tex:10132) says the run used `convnextin_adamdpwxclipdropbf16`.
- LeanMlir/Proofs/Nets/ViT/ViTWholeBackCertifiedTieB.lean:283 (`vitTinyInputGradB_eq_vitTiny_vjp`) says "tier T6 at the paper net and the shipped index … 128 or 512 per device in the shipped `vitin_*` artifacts", but the statement is pinned at `nClasses = 10`, while the vitin_* artifacts have 1000 classes. This is overclaim group 5.

## Overclaims (fix before anyone reads the published results again)
1. content.tex:16990: "the gradient the GPU computes is, by a machine-checked theorem, the exact derivative … up to one printer, one lowerer, and floating point." It drops the smooth-point hypotheses, the unproved lexing and StableHLO semantics, the drop/bf16/one-replica scope, and the fact that the object is a Lean AST.
2. content.tex:16811 / 10583 / 12394: "If it builds, it's correct"; "the gradient computed by this trainer is the mathematically correct one … is a theorem"; "every layer primitive shipped … has a machine-checked backward". The kinked ops are proved only at smooth points, the result is over ℝ at the Lean level, and it is a `.correct` projection.
3. content.tex:9619 / 9653 and formalization.yaml:166: the step ties cover "every convnextin_*" / "every vitin_*" / "every shipped ConvNeXt artifact". Both ties are drop-free and one-replica, bf16 artifacts emit other nodes, and the runs the book reports are dp+drop+bf16.
4. content.tex:17030 and formalization.yaml:221–226: the max-pool condition is called an "argmax tie" on a "measure-zero set, and nowhere else". The hypothesis actually requires all window entries distinct, which fails routinely after ReLU.
5. formalization.yaml:225: the global `reluHasVJP` / `mlpHasVJP` / `maxPool2HasVJP3` are called "codegen-shaped". They are the canonical pdiv witnesses.
6. content.tex:16960 / 16982: "the emitted text lexes and parses back to the proven op-graph". `roundtrip` is token-level and lexing is unproved.
7. content.tex:11488 / 11518: the weight-shared, scalar-LN `transformerTower` is presented with "ViT-Tiny/Base and Large are instances" and as "the full ViT transformer backbone", and the ε > 0 hypothesis is dropped.
8. content.tex:9362 / 9819 / 10488: every trained LayerNorm is said to backpropagate "by Theorem layerNormHasVJP". That theorem is the scalar-affine LN; the nets use the channel and vector LN witnesses.
9. content.tex:11906: "The emitted graph denotes transformerAttnSublayerHasVJPMat's backward." No theorem links any emitted graph to that witness.
10. content.tex:16522–16527: ChallengeArch is said to phrase "ResNet-34's rendered backward", leaving you to trust "the forward functions and nothing else". ChallengeArch holds no ResNet-34 statement, and its rows are toy nets' Lean witnesses.
11. content.tex:17012: "the certified loss-descent step". Descent is proved only for the shallow nets.
12. content.tex:16668: "provably the augmentation". It holds by construction.
13. content.tex:14369: the blackjack "theorem". It is computed by value iteration.
