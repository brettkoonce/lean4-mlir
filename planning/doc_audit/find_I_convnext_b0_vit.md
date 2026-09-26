# Slice I — ConvNeXt / EfficientNet / ViT (`LeanMlir/Proofs/Nets/{ConvNeXt,EfficientNet,ViT}/`)

**Coverage.** I read every module docstring and every declaration docstring in all 45 files. I compared each docstring to its declaration's signature. I read these files in full, proofs included: ConvNeXt.lean, ConvNeXtFullT, ConvNeXtChannelLN, ChainClose, Fold, FoldG, FoldGB, BackB0, BackCertifiedTie, BackChains, ConvNeXtStepTie and ConvNeXtStepTieGB. For the WholeBack ties, all EfficientNet files and all ViT files, I read the module docs, the declaration docs and the capstone statements, but skimmed the bodies of mechanical proofs.

- **Artifacts checked.** I opened the `verified_mlir/*.mlir` signatures to check each claim about artifact shapes: the `%x`, `%onehot` and `%psW` widths.
- **Axiom claims.** I checked the "3-axiom-clean" claims by grep only. The slice has zero hits for `sorry`, `admit`, `axiom `, `native_decide`, `implemented_by` and `@[extern]`. That is not proof of the claims, because I did not run `#print axioms`.

Ranked most misleading first.

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean:519 — `convnextImagenetInputGradB_eq_vjp` (and module doc lines 6–9)

**Kind:** overclaim
**Says:**
- Declaration: "`nC = 1000`, the class count of every `convnextin_*` / `convnextsin_*` / `convnextbin_*` artifact, at a variable batch `B` — 64 or 128 per device in those runs … The dims are the paper's (`3×224²`, `[3,3,9,3]` at `96→192→384→768`), so this is the whole statement at the artifact and not an instance of it."
- Module: "Every shipped ConvNeXt artifact runs a batch — `convnext_adam_train_step` and the `convnextin_*` / `convnextsin_*` / `convnextbin_*` families — and its batched T3 tie … states every activation as …"

**Actually states:** The theorem is about `w : CnxTWeightsCh 1000`. That is ConvNeXt-**T** (`[3,3,9,3]`, 96-wide stem, `convNextForwardTCh`), without drop-path and in f32.
- `convnextsin_*` is ConvNeXt-S (`[3,3,27,3]`, `cnxSmall`).
- `convnextbin_*` is ConvNeXt-B. Its `%psW` is `tensor<128x3x4x4>`, not 96.
- `ConvNeXtStepTieGB` says so itself: "S and B are other nets."
- The per-device batch in `convnextsin_*`/`convnextbin_*` is 32 (`%x: tensor<32x150528>`), not 64 or 128.
- The drop-path artifacts, including the book-quoted `convnextin_adamdpwxclipdrop`, have a different forward (the `dropPathB` sites).

**Fix:** "`nC = 1000`, the class count of the `convnextin_*` (ConvNeXt-T) artifacts, at a variable batch `B`. It covers the drop-free f32 forward `convNextForwardTCh`; the drop-path (`*drop*`) and ConvNeXt-S/B (`convnextsin_*`, `convnextbin_*`) artifacts are different functions and are not covered." Make the same correction to module lines 6–9.

---

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetStepTieG.lean:11–16 — module doc; `efficientnet_net_tiedG` (:402–411)

**Kind:** overclaim
**Says:** "Every conjunct is at the RAW gradient node (`*GradB`), which is what `efficientnet_adam_train_step.mlir` and every ImageNet artifact emit … One statement therefore covers AdamW, RMSProp, EMA, the clipped and drop-path variants and their data-parallel and bf16 twins, because they all consume this node."

**Actually states:** The capstone falls short of that claim in three ways.
- **Class count.** `efficientnet_net_tiedG (… (w : B0Weights) … (t : Vec (N * (1 * 10))))` is pinned at 10 classes: `B0Weights.fcW : Mat 1280 10`. Every `efficientnetin_*` artifact is 1000-class (`%onehot: tensor<64x1000xf32>`), so no ImageNet artifact is covered.
- **Drop-path and dropout.** The threaded forward is `efficientnetForwardBFull`, which has no drop-path and no dropout. The sibling file `EfficientNetSyncStepTieG.lean:57–60` says so: "T3 states the chain without stochastic depth or classifier dropout".
- **bf16.** The bf16 twins emit `*GradBBf16` nodes. `EfficientNetSyncStepTieG.lean:60` says "The `bf16` DP variants emit different gradient nodes and are not covered."

**Fix:** "Every conjunct is at the raw `*GradB` node that the f32 Adam/RMSProp/EMA renders emit. The statement is at `B0Weights`' 10-class head (Imagenette) and on the chain without drop-path or dropout. The 1000-class `efficientnetin_*` artifacts, the drop/dropout variants and the bf16 twins (`*GradBBf16`) are not covered."

---

### LeanMlir/Proofs/Nets/ViT/ViTWholeBackCertifiedTieB.lean:290–296 — `vitTinyInputGradB_eq_vitTiny_vjp` (and module doc lines 7–9)

**Kind:** overclaim
**Says:** "ViT-Tiny's BATCHED whole-net backward tie — tier T6 at the paper net and the shipped index. … 10 classes), at a variable batch `B` — 128 or 512 per device in the shipped `vitin_*` artifacts, and neither number appears here."

**Actually states:**
- The capstone is instantiated at 10 classes. The `vitin_*` artifacts whose batch sizes it quotes are 1000-class (`vitin_emadp128x4wxclipdropbf16`: `%onehot: tensor<128x1000xf32>`).
- The per-device sizes in the `vitin_*` family are 128 and 256 (`vitin_adamdp256x2…`). 512 is a global batch.
- The class-generic `vitInputGradKB_eq_batchMap_vitForwardKV_vjp` does cover `nC = 1000`, but only for the drop-free f32 forward.

**Fix:** "at Imagenette's 10 classes and a variable batch `B`. The 1000-class `vitin_*` forward is the generic `vitInputGradKB_eq_batchMap_vitForwardKV_vjp` at `nClasses := 1000`, drop-free. Drop-path artifacts are not covered."

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTieGB.lean:211–219 — `cnx_net_tiedGB`; ViT/ViTStepTieGB.lean:359–368 — `vit_net_tiedGB`; ViT/ViTStepTie.lean:13–15 — module

**Kind:** overclaim
**Says:**
- ConvNeXt: "All 182 parameters, at the nodes `convnext_adam_train_step.mlir` and every `convnextin_*` train step emit."
- ViT: "at the nodes `vit_adam_train_step.mlir` and every `vitin_*` artifact emit."
- ViTStepTie: "Its batched peer … — the chain every `vitin_*` accuracy comes from — is `ViTTiePoCGB.vit_net_tiedGB`."
- Module docs, ConvNeXtStepTieGB:13–15 and ViTStepTieGB:13–14: "which is what … every `convnextin_*` [`vitin_*`] artifact emit".

**Actually states:**
- **bf16.** The `*bf16` artifacts emit `*GradBBf16` nodes. `ConvNeXtFoldGB.lean:45–48` and `ViTFoldGB.lean:44–47,61–62` say so. The artifact `ViTFoldGB` says the book quotes is `vitin_emadp128x4wxclipdropbf16`, which is bf16.
- **Drop-path.** Both capstones are stated "at the drop-free chain". That artifact is also drop-path, so it is excluded twice over. `cnx_net_tiedGB`'s own ⛔ note concedes drop-path but not bf16.

**Fix:** "at the `*GradB` nodes every f32 `convnextin_*` [`vitin_*`] train step emits; the bf16 artifacts emit `*GradBBf16` (see `Foundation/Bf16GradNodes.lean`), and the `*drop*` artifacts' cotangent chains carry `dropPathB` sites not stated here." In ViTStepTie:14, replace "the chain every `vitin_*` accuracy comes from" with "the drop-free f32 chain the `vitin_*` artifacts share".

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTie.lean:9–17, 62–70, 552–569 — module doc and `cnx_net_tied_certified` (ViT peer: ViTStepTie.lean:326–334 `vit_net_tied_certified`)

**Kind:** overclaim
**Says:** "`verified_mlir/convnext_train_step.mlir`, measured, not assumed … so the whole 18-block train step is den-composed forward → loss → backward, no free activations, no symbolic cotangent". The coverage note adds: "181 of them at the full `θ − lr·(certified ∂Loss/∂θ)` step".

**Actually states:**
- **One image, not the artifact's batch.** The theorem takes one image `x : Vec (3*224*224)` and a hard label `label : Fin 10`. The artifact computes a batch-mean step at `%x: tensor<32x150528xf32>`. `ConvNeXtStepTieGB.lean:10–12` says this capstone "was at a single image with the batch outside the AST".
- **No tie to the certified VJP.** Each conjunct is `θ − lr·Σ pdiv(layer)·cot` at the hand-threaded cotangents `w.bK.cotIn` (`cnxBlockCotInChAt`, `cnxDownCotInChAt`, `cnxHeadDyXheadCh`). No theorem in the ConvNeXt or ViT trees says these chain cotangents equal the certified VJP of the downstream loss. There is no `cnx…CotIn_eq_vjp` or `vit…CotIn_eq_vjp`; grep finds `vitBlockCotInAtMHV` and `cnxBlockCotInChAt` only in the StepTie files. MobileNetV2 (`mnv2*CotIn_eq_vjp`), ResNet-34 (`r34IdCotIn_eq_vjp`) and B0-sync (`xCotIn_eq_vjp`, …) do have such theorems. So "certified ∂Loss/∂θ" is not what is stated.
- **Same wording in ViT.** `vit_net_tied_certified` ("EVERY param op `den`otes the certified loss-descent step … 200/200") is also per-example at a hard label against a batch-32 artifact.

**Fix:**
- Module: "tied **per example at one hard-labelled image** (the artifact's batch of 32 and its mean lie outside this statement; the batched form is `ConvNeXtStepTieGB`)".
- Coverage: "181 at `θ − lr·(certified layer Jacobian · chain cotangent)`, where the chain cotangent is the hand-composed reverse `cnxBlockCotInChAt` … (not separately proved equal to the loss VJP)".
- StepTieGB:215: drop "certified" from "every block's certified cotangent chain". Make the same edits in ViTStepTie/ViTStepTieGB.

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTie.lean:47–52 — module doc (apex paragraph); ViT/ViTWholeBackCertifiedTie.lean:29–31 — module doc

**Kind:** wrong
**Says:**
- ConvNeXt: "Its only hypotheses are the 23 LayerNorm positivities, so unlike every other whole-net backward tie in this repo it carries no smoothness side-condition."
- ViT: "so like ConvNeXt-T and unlike ResNet-34 / MobileNetV2 / EfficientNet-B0 this is a `HasVJP` and not a smooth-point `HasVJPAt`".

**Actually states:**
- `vitInputGradK_eq_vitForwardKV_vjp` (ViTWholeBackCertifiedTie:158) takes only `0 < ε`.
- `efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp` (EfficientNetFullWholeBackCertifiedTie:183) is against the global `HasVJP` `efficientnetForwardBFullHasVJP` under `w.EpsPos` only. Its own module says "there is no smooth point: swish and the SE sigmoid are differentiable everywhere".
- `ViTBackNet.lean:47` states the correct grouping: "ViT joins enet (swish) and convnext (gelu) in the unconditional tier".

**Fix:**
- ConvNeXt: "…so, like ViT's and EfficientNet-B0's and unlike the ReLU nets' (ResNet-34/50, MobileNetV2, MNv4), it carries no smoothness side-condition."
- ViT: "like ConvNeXt-T and EfficientNet-B0, and unlike ResNet-34 / MobileNetV2, …".

---

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetFullWholeBackCertifiedTie.lean:6–13, 34–36 — module doc

**Kind:** overclaim + stale
**Says:**
- Title: "`efficientnetInputGradBFull` IS the certified whole-net PAPER EfficientNet-B0 gradient".
- "`EfficientNetWholeBackCertifiedTie.lean` closed this for the three-block representative … That is what the representative's file could not do — it stopped at a `▸`-transported `_committed` witness".

**Actually states:**
- The ties are at `w : B0Weights`, whose classifier is `Mat 1280 10`. The paper B0 head is 1000-way.
- `EfficientNetWholeBackCertifiedTie.lean` now holds only the two endpoint ties, `stemBBack_eq_vjp_backward` and `headFwdBBack_eq_vjp_backward`, and describes itself that way. No three-block representative tie or `_committed` witness exists there.

**Fix:**
- Title: "…IS the certified whole-net EfficientNet-B0 gradient (paper `[t,c,n,s,k]` trunk, 10-class head)".
- Replace the history with: "The stem and head endpoint ties are `EfficientNetWholeBackCertifiedTie.lean`'s; the sixteen blocks enter opaque."

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtFullT.lean:23–24, 58–62, 563 — module doc and `convNextFwdGraphTCh_faithful`

**Kind:** stale
**Says:**
- Lines 23–24: "the whole-net VJP is GLOBAL (unconditional except the 22 LN positivities)".
- Lines 58–62: "…the pre-§2m net had **18 + 3 + 1 head**. So this forward has a stem LN and no head LN."
- Line 563: "Same `rw` chain as the scalar apex, with `chanLNGraph_faithful` where the `bnF`s were."

**Actually states:**
- `convNextForwardTChHasVJP` takes 23 positivities, including `hhε : 0 < w.hε`.
- `convNextForwardTCh` has a head LN (`rowLNVecFlat 1 768 w.hε w.hγ w.hβ`).
- The scalar apex was deleted (lines 28–35 say so).

**Fix:**
- "unconditional except the 23 LN positivities (1 stem + 18 block + 3 downsample + head)".
- Delete the "no head LN" sentence.
- Line 563: "One `rw` per stage, `chanLNGraph_faithful` at each spatial LN and `headLNGraph_faithful` at the head."

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtBackB0.lean:197–202 — module doc

**Kind:** stale
**Says:** "ConvNeXt's whole verified stack is **per-example / batch-1** — LayerNorm here is the per-example separable `layerNormForward` (= `bnForward` on the feature axis), so NONE of EfficientNet's `batchMap`/`bnBatchLA` batched machinery is needed".

**Actually states:**
- Every declaration in the file is over `chanLNTensor3`, not `layerNormForward`.
- The stack has batched ties that do use `batchMap`/`batchMapAux`: `ConvNeXtStepTieGB.cnx_net_tiedGB` and `ConvNeXtWholeBackCertifiedTieB.convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp`.

**Fix:** "This file's graphs are per-example: ConvNeXt's channel LayerNorm (`chanLNTensor3`) is per-example separable, so the batched lifts (`ConvNeXtStepTieGB`, `ConvNeXtWholeBackCertifiedTieB`) are plain `batchMap`s of these."

Also fix line 311 (`cnxBlockBodyChBackGraph_faithful`), which says "Same proof as the scalar peer with `chanLNBackGraph_eq_vjp` where `bnBack_faithful_fn` was". The scalar peer no longer exists; describe the proof instead.

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtChainClose.lean:13–26, 125–126 — module doc and `cnxCotN`

**Kind:** stale
**Says:**
- Module: "…back through `layerScale → project → gelu → expand` to the LN output, where the scalar-LN `γ/β` grads read it", and "(layer-scale `γ`, scalar-LN `γ/β`)".
- `cnxCotN`: "the cotangent the scalar-LN `γ/β` grads contract with".

**Actually states:** Every consumer is channel-LN. `ConvNeXtStepTie.cnxBlockChTied` feeds `cnxCotN` to `CnxPoC.ChanLNGammaSgdTied`/`ChanLNBetaSgdTied` at `Vec c` γ/β. The scalar-LN ConvNeXt net was retired (ConvNeXtFullT:28–35).

**Fix:** Replace "scalar-LN" with "channel-LN (`chanLNTensor3`, `Vec c` γ/β)" in all three places.

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXt.lean:34–42 (module) and 58–65 (`convNextBlockBody`)

**Kind:** stale
**Says:** "…true ConvNeXt LayerNorm is per-spatial-position over the channel axis … This is the same representation simplification the audit flagged for the LN family; a faithful channel-LN-over-NCHW lift is a follow-up."

**Actually states:** The channel-LN lift exists and is what ships: `chanLNTensor3` (`Architectures/ChannelLN`) and `cnxBlockChW` / `convNextForwardTCh` (ConvNeXtFullT.lean). This file is kept as the scalar-LN representative for the comparator (ConvNeXtFullT:54–56).

**Fix:** "…normalizes over the whole flattened vector with scalar `γ, β`. The shipped net's per-position channel LayerNorm is `chanLNTensor3`, and the full ConvNeXt-T at it is `convNextForwardTCh` (`ConvNeXtFullT.lean`). This two-block scalar-LN net is kept as the ch9 representative the comparator checks."

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtFoldGB.lean:7–9, 19–21, 58–62 — module doc (ViT peer: ViTFoldGB.lean:5–8)

**Kind:** stale
**Says:**
- "`ConvNeXtFoldG.lean` folds the fourteen gradient nodes of the PER-EXAMPLE traversal".
- "4c leg 3 moves the drop-free writers onto this chain; this file lands first".
- Honest residual: "ConvNeXt's capstone (`ConvNeXtStepTie.lean`, 182 params) is at the per-example SGD-inline `convnext_train_step.mlir` … Re-pointing it at these nodes with `SmoothedLossCot` is 4b's ConvNeXt capstone, which this file is the prerequisite for."

**Actually states:**
- `ConvNeXtFoldG.lean` contains three lemmas. The other node kinds are folded in `Foundation/GradNodesB` and elsewhere.
- Leg 3 has landed. `ConvNeXtRenderB.lean:716–722` says the drop-free writers were moved, and `ConvNeXtFoldG`'s own header says "Every Adam artifact of this net renders from the batched chain".
- The re-pointed capstone exists as `ConvNeXtStepTieGB.cnx_net_tiedGB`.
- The same stale pattern appears in ViTFoldGB:5: "`ViTFoldG.lean` folds the ten gradient nodes", where FoldG has two lemmas.

**Fix:**
- "`ConvNeXtFoldG.lean` holds the three per-example lemmas specific to this net (layer-scale γ, channel-LN γ/β); the rest are shared (`GradNodesB`)."
- Replace the leg-3 sentence with the present fact: "every Adam artifact renders from `convNextBackAllB`".
- Replace the last residual bullet with: "The §1a tie at these nodes is `ConvNeXtStepTieGB.cnx_net_tiedGB`."
- Correct ViTFoldGB:5 the same way.

---

### LeanMlir/Proofs/Nets/ViT/ViTWholeBackCertifiedTie.lean:5–7 — module doc

**Kind:** stale
**Says:** "The ViT peer of `r34InputGrad_eq_resnet34_vjp`, `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp`, `convnextInputGrad_eq_convNextForwardTCh_vjp` and `efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp`".

**Actually states:** Neither `r34InputGrad_eq_resnet34_vjp` nor `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp` exists as a declaration; I grepped `(def|theorem|lemma) <name>` over `LeanMlir/`. The same file's line 107 calls the per-example ResNet-34 tie "retired". The live peers are `r34InputGradB_eq_r34B_full_vjp` and `mnv2InputGradB_eq_mobilenetv2B_full_vjp`.

**Fix:** "The ViT peer of `convnextInputGrad_eq_convNextForwardTCh_vjp` and `efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp` (per-example); the batched family is `r34InputGradB_eq_r34B_full_vjp`, `mnv2InputGradB_eq_mobilenetv2B_full_vjp`, …"

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean:15–17 and ViT/ViTWholeBackCertifiedTieB.lean:13–14 — module docs

**Kind:** copied
**Says:**
- ConvNeXt: "ConvNeXt was the last net without a batched whole-net tie … with this the `*InputGradB_eq_*_vjp` family covers all seven nets."
- ViT: "ViT was the one net without a batched whole-net tie; with this the `*InputGradB_eq_*_vjp` family covers all seven nets."

**Actually states:** Both files claim to be the one that completed the family, so at least one sentence is false. These are dated status notes that nothing keeps current.

**Fix:** Delete both sentences, or keep one neutral line in each: "The batched `*InputGradB_eq_*_vjp` peers of the other nets are …".

---

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtFullT.lean:387–388 — `convNextForwardTCh_eq_chain`

**Kind:** copied
**Says:** "The nested↔chain bridge (see `convNextForwardTCh_eq_chain` for why the proof shape matters — a `simp`/`rfl` proof of this statement dies in the kernel on the recursive stage folds)."

**Actually states:** The docstring sits on `convNextForwardTCh_eq_chain` itself, so it points at itself. It was adapted from `efficientnetForwardBFull_eq_chain`, which cites this lemma.

**Fix:** "The nested↔chain bridge: `convNextForwardTCh w x` equals the twelve-factor `∘` chain `convNextForwardTChHasVJP` is stated on." Move the kernel note into a comment in the proof.

---

### Minor stale names and references (one finding, several sites)

**Kind:** stale

1. **ViT/ViTDepthK.lean:24–25.** Says "Depth-12 ViT-Tiny shapes are now a config change away (the production capstone needs only the P=16/D=192/heads=3 instantiation of these)." That capstone already exists in the same file: `vitTinyHasVJP_correct` (:256). **Fix:** "The ViT-Tiny instantiation is `vitTinyHasVJP_correct`."
2. **ViT/ViTVecLN.lean:234.** `vitCotB2outV` "Cot at block 2's output". The net is depth 12; this is the cotangent at the tower output (the final-LN input). **Fix:** "Cot at the last block's output (the final-LN input)".
3. **EfficientNet/EfficientNet.lean.**
   - Line 41 says "Like `convBnRelu`". No such declaration exists.
   - Lines 24–26 say `efficientnetHasVJPAt` is "built by `vjpCompAt`". It is the global `efficientnetHasVJP` restricted by `.toHasVJPAt` (:292–297).
4. **ConvNeXt/ConvNeXtFullT.lean:257–258.** Refers to `rowLNVecFlat`'s "`_diff`". Since the naming pass the lemma is `rowLNVecFlat_differentiable`.
5. **ConvNeXt/ConvNeXtWholeBackCertifiedTie.lean.**
   - Line 54 calls `convNextForwardTCh_eq_chain` "the `rfl` saying…". It is proved by `rw` (ConvNeXtFullT:403–406).
   - Line 71 names `cnxTk`. No declaration has that name.
6. **ConvNeXt/ConvNeXtBackChains.lean:218–219.** Says "`lnB` the LayerNorm back (= BN-back)". The ties fill `lnB` with `chanLNTensor3Back`, which is not BN-back.

---

### Process narrative in module and declaration docs (grouped)

**Kind:** process-narrative
**Says (representative):**
- ConvNeXtFullT:28–35, 50–62, 228–240, and the `convNextForwardTChHasVJP` docstring at 290–293 ("⚠ The count read `22 … no head LN` until 2026-09-04 … the third place that stale number had been copied to").
- ConvNeXtWholeBackCertifiedTie:3–100: §3.18/§3.10 history, "17 s and 6 GB … on Lean 4.32.2, 6 min and 48 GB on 4.34.0".
- ConvNeXtWholeBackCertifiedTieB item 4 ("~18 min on Lean 4.32.2").
- ConvNeXtStepTie:34–40 "Superseded scope note".
- ViTFoldGB:10–14 "Measured 2026-09-07".
- EfficientNetBackNet:6–11, 26–38 (the §8e sweep post-mortem).
- EfficientNetFullB0Eval:10–12 ("Built 2026-09-05 … the budget was deleted 2026-09-08").
- ConvNeXtBackB0 `chanLNBackGraph_eq_vjp` (:276–281, "the reason landing §B first was worth doing").

**Actually states:** None of this describes the declarations; it is commit history and elaboration-cost logs.

**Fix:**
- Keep one line per module on what it proves and what feeds it.
- Move build-cost and proof-shape notes (the "two spellings of one numeral" rule, the `rw`-not-`simp` kernel note) into comments inside the proofs they govern.
- Move dates and § history to commit messages.

---

## Overclaims (fix before anyone reads the published results again)

1. **ConvNeXtWholeBackCertifiedTieB:519 `convnextImagenetInputGradB_eq_vjp` (and module lines 6–9).**
   - It claims to cover `convnextsin_*` (ConvNeXt-S) and `convnextbin_*` (ConvNeXt-B, 128-wide), and calls itself "the whole statement at the artifact".
   - It is ConvNeXt-T, without drop-path, in f32.
   - The quoted batch sizes are wrong for S/B.
2. **EfficientNetStepTieG:11–16 (module) and `efficientnet_net_tiedG`.**
   - It claims "every ImageNet artifact" and says it covers the drop-path and bf16 twins.
   - The capstone is pinned at `B0Weights`' 10-class head and has no drop-path or dropout.
   - The bf16 artifacts emit different node kinds; the sibling SyncStepTieG says so.
3. **ViTWholeBackCertifiedTieB:290 `vitTinyInputGradB_eq_vitTiny_vjp`.** It is framed as "the shipped index" of the 1000-class `vitin_*` artifacts but is stated at 10 classes.
4. **`cnx_net_tiedGB`, `vit_net_tiedGB` (and module docs), ViTStepTie:14.** "Every `convnextin_*` / `vitin_*` artifact emit[s]" these `*GradB` nodes is false: bf16 artifacts emit `*GradBBf16`, including the ViT artifact the book quotes. For ViT, "the chain every `vitin_*` accuracy comes from" is false for that same drop+bf16 artifact.
5. **ConvNeXtStepTie `cnx_net_tied_certified` and ViTStepTie `vit_net_tied_certified` (and the StepTieGB peers' "certified cotangent chain").**
   - The statements are per example at a hard label, while the artifacts are batch-32 means.
   - "Certified ∂Loss/∂θ" is asserted, but ConvNeXt and ViT have no `*CotIn_eq_vjp` theorem tying the hand-threaded chain cotangents to the certified VJP.
6. **EfficientNetFullWholeBackCertifiedTie title.** It says "PAPER EfficientNet-B0", but the head is 10-class.
7. **ConvNeXtWholeBackCertifiedTie:51–52 and ViTWholeBackCertifiedTie:29–31 (wrong comparative claims).** ConvNeXt says it is the only unconditional whole-net tie. ViT says B0's is `HasVJPAt`. Both are false: ViT's and B0's are both global `HasVJP` under only the ε-positivity hypotheses.
