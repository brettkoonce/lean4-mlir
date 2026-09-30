# Slice N1 — Nets: ResNet, Small, ConvNeXt: rubric review 2026-09-30

Scope: `LeanMlir/Proofs/Nets/{ResNet,Small,ConvNeXt}/*.lean` minus `*ParamGrad.lean`, at `55ad3a5a`.
Code after 2026-09-26 in the slice: `88ad803e` (ConvNeXt `*CotIn_eq_vjp`), `a4c268d9` (small-CNN
dense heads), `37f8ab12` (MaxPool smoothness), and the docs/private sweeps. All three are read here.

## Verdicts
| angle | verdict | findings |
|---|---|---|
| correctness | request_changes | 3 |
| reuse | request_changes | 2 |
| scope | request_changes | 1 |
| attribution | request_changes | 1 |
| api-design | approve | 0 |
| generality | request_changes | 2 |
| placement | request_changes | 1 |
| naming | request_changes | 1 (carried) |
| documentation | request_changes | 3 |
| proof-quality | request_changes | 1 |

No false theorem was found. Every capstone was checked against its signature, and every seal against its
existential. The correctness findings are about which function or artifact a true statement is about.

## Findings

### correctness

- **N1-corr-1** `LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTie.lean:207–223` (`cnxStemChTied`),
  `ConvNeXtStepTieGB.lean:226–247` (`cnxStemChTiedGB`), and the capstones that thread them
  (`cnx_net_tied_certified:632`, `cnx_net_tiedGB:474`, binder `xstem`). **Problem:** the ConvNeXt
  stem-bias clause certifies the wrong function. It is stated as
  `pdiv (fun b' => Tensor3.flatten (conv2d Wst b' xstem)) psb`, a stride-1 4×4 convolution applied
  to a free phantom input `xstem : Tensor3 3 56 56`. The net's stem is `flatConvStride4 Wst psb x`
  applied to the real `3×224×224` image. The clause is true: a conv's bias Jacobian does not depend
  on the input or the stride, so the value is right. But the "certified per-layer Jacobian" named
  in the statement is the Jacobian of a map the network never computes, and no lemma equates it
  with the stride-4 stem's. So "all 182 parameters tied" is, for `psb`, a tie to a stand-in. The
  module text justifies this as "the same modelling the mnv2/r34 stems use" (`ConvNeXtStepTie.lean:204`),
  and that is false. `r34StemTiedB` (`ResNet34StepTieB.lean:346`) uses
  `GradNodeB.ConvStridedBTiedB` at `flatConvStride2 Ws b' x` of the real input.
  `mnv2StemTiedB` (`MobileNetV2StepTieB.lean:486`) uses `flatConvStride2Xla Ws b' x`.
  **Fix:** state the stem-bias clause at `fun b' => flatConvStride4 Wst b' x` (per example) or at
  `batchSlice … x n` (batched), and drop the `xstem` binder from `cnxStemChTied`,
  `cnxStemChTiedAt`, `cnxStemChTiedGB`, `cnxStemChTiedGBAt`, `cnx_net_tied_certified` and
  `cnx_net_tiedGB`. The proof needs one bridge: `pdiv` in `b` of `flatConvStride4 W b x` equals `pdiv`
  in `b` of `conv2d W b y` at the output resolution. Both are the bias broadcast; prove it once
  next to `flatConvStride4WeightGradHasVJP`, or as a `convStride4BiasGradB_den` in `GradNodesB`
  beside `convStridedBGradB_den`. Delete the "same modelling" sentence.
  **Evidence:** `grep -rn "fun b'.*flatConvStride4" LeanMlir/Proofs` returns nothing, while the stride-2
  peer exists (`SgdNodes.lean:263`, `ConvGrad.lean:148`). `ConvNeXtFoldGB.lean:19` lists the stem bias
  under the stride-1 `convBGradB_den`.
  **Cost:** the two capstones lose one binder. They are pinned in `tests/AuditAxioms.lean`, and
  `cnx_net_tiedGB` is cited at `content.tex:9999`, but those references are by name, so the name
  stays and no consumer outside the two files passes `xstem`. About +30/−15 lines.
  **Gate:** `lake build Certs`, the audit, docstring-checkrefs. **Size:** S–M.

- **N1-corr-2** `LeanMlir/Proofs/Nets/Small/Cifar8StepTie.lean:90` (`cifar8_train_step_tied_certified`),
  `Cifar8BnStepTie.lean:59` (`cifar8Bn_train_step_tied_certified`), and in the same way
  `CifarFold.lean:181` and `CnnFold.lean:229`. **Problem:** the Chapter-4 step ties are about
  artifacts other than the ones the chapter's results were trained on. Every clause is at a fused
  SGD op (`convWeightSgd`, `weightSgd`, …; `SgdNodes.lean:142–191`), so the ties cover only the
  SGD-inline `…V` renders. The book says `thm:cifar8_step_tie` covers `cifar8_train_step.mlir`
  (`content.tex:4294–4305`), and no trainer runs that file: `MainCifar8Verified` was deleted in
  cleanup §12, and the only references left are `CnnArtifacts.lean:94` (writer),
  `tests/TestCifar8AdamTrain.lean:7` (a comment) and `convention_audit.py:214`. The results the
  book reports for cifar8 come from the packed wide-head arms:
  `cifar8w{,_bn}_{sgd,mom,adam}` (`c8wPacked`, `CnnArtifacts.lean:270`) emit unfused
  `SHlo.convWeightGrad` (`CnnRender.lean:510–534`), and `cifar8wb*` (`c8wbPacked`) emit the batched
  `*GradB` family. No step tie covers any of them.
  **Fix:** add a gradient-node step tie for the packed families, the way ViT, ConvNeXt and B0 have
  `*StepTieG`/`*GB`. For `cifar8wb*` the node lemmas already exist (`GradNodeB.convWGradB_den`,
  `denseWGradB_den`, `GradNodesB.lean:48/177`), so a `Cifar8StepTieGB.lean` at the batched chain
  is instantiation. Until then, the Lean docstrings (which name no artifact) and
  `content.tex:4294–4327` should say which artifact is tied and that the wide ablation arms are not.
  **Cost:** a new file of about 250 lines, with a new audit block and a new blueprint node, or a
  prose-only fix. **Size:** M (tie), S (prose).

- **N1-corr-3** `LeanMlir/Proofs/Nets/ResNet/ResNet34StepTieB.lean:23–33, 419`,
  `ResNet50StepTieB.lean:21–24, 57, 491–501`, `ResNet50WholeBackCertifiedTieB.lean:40`,
  `ResNet50FullB.lean:9, 266`. **Problem:** the ImageNet runs the book reports are bf16:
  `resnet34in_momdp64bf16` (`content.tex:6125`), and `resnet50in_momdp64bf16` and
  `resnet50in160_lambaccdp4x128wxclipbcebf16` (`content.tex:6395–6396`). No whole-net tie covers them.
  `Foundation/DataParallel/SyncBf16.lean` states "No whole-net statement … for either net", and the
  bf16 nodes (`*GradBBf16`) are separate op kinds. The ResNet capstones are correct about what they
  name (f32 `resnet34in_momdp64`, `resnet50in160_lambaccdp8x64bce`). But the ResNet-34 files never
  mention bf16. `ResNet50StepTieB` uses the retired f32 76.66% run as its only example, and
  `ResNet50FullB.lean:266` still says "where the quoted 76.66% comes from". The sync twin
  `ResNet50SyncStepTieB.lean:58` is the only ResNet file that says "the bf16 conv twins are not
  this". ConvNeXt states it in both tie files (`ConvNeXtStepTieGB.lean:17–18, 463–464`).
  **Fix:** add one scope sentence to the module docstrings of `ResNet34StepTieB`,
  `ResNet34SyncStepTieB`, `ResNet34BackCertifiedTieB`, `ResNet50StepTieB` and
  `ResNet50WholeBackCertifiedTieB`. The sentence names the bf16 artifacts behind the book's numbers
  and says they are outside the statement, pointing at `SyncBf16.lean`. Replace the 76.66% example
  with the current f32 twin of the A3 render (`resnet50in160_lambaccdp8x64wxclipbce`), or with none.
  **Evidence:** `grep -n -i bf16 Nets/ResNet/ResNet34*.lean` (non-ParamGrad) returns nothing.
  **Cost:** docstrings only. **Size:** S.

### reuse

- **N1-reuse-1** `LeanMlir/Proofs/Nets/ResNet/ResNet34FullBSeal.lean:801`,
  `ResNet50FullBSeal.lean:920` (`sealX_continuous`). **Problem:** both are one-line wrappers,
  `:= rayX_continuous _ _`, and nothing uses them. The token `sealX_continuous` occurs only at its two
  definitions across all `.lean/.tex/.yaml/.yml/.py` files. The api-design audit §7.3 flagged
  R34's copy for re-proving `rayX_continuous`. It now aliases it, which is still a wrapper.
  **Fix:** delete both. **Cost:** no pins (0 audit, 0 tex, 0 yaml, 0 comparator); −2 lines.
  **Gate:** `lake build Certs`. **Size:** S.

- **N1-reuse-2** `LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:110–174`
  (`r34BFullHasVJPAt`, `r34BFullHasVJPAt_backward`), next to
  `Nets/MobileNet/MobileNetV4WholeBackCertifiedTieB.lean:230` (`mnv4BFullHasVJPAt`, which says it is
  "the same construction at eighteen stages; MobileNetV4 needs its own because Conv-M's ladder is
  longer, not because anything differs") and MobileNetV2's apex. **Problem:** the opaque n-stage
  `vjpCompDiffAt` apex is written out per net, and it is net-agnostic (every dimension a variable).
  ResNet-50 imports ResNet-34's copy. **Fix:** see N1-place-1 for the move. A single
  length-indexed chain is not proposed, because the heterogeneous `Vec sₖ` types and the
  kernel-depth trap the build notes record make it risky. **Size:** see N1-place-1.

### scope

- **N1-scope-1** `LeanMlir/Proofs/Nets/ResNet/ResNet34FullBSeal.lean:803` (`r34StemB_continuous`).
  **Problem:** dead. Its only occurrence anywhere is its definition, and `Rr_continuous:811`
  unfolds `r34StemB` itself rather than using it. **Fix:** delete it. **Cost:** no pins; −5 lines.
  **Size:** S.

### attribution

- **N1-attr-1** Module docstrings of `ResNet34FullB.lean`, `ResNet50FullB.lean`,
  `ConvNeXtFullT.lean` and `ConvNeXt.lean`, and of `ResNet50StepTieB.lean` for the recipe.
  **Problem:** the architectures these files formalise are not credited. ResNet (He, Zhang, Ren
  and Sun, "Deep Residual Learning for Image Recognition", CVPR 2016, arXiv:1512.03385) appears only
  as an inline "He et al.'s 3×3/s2 max-pool" in declaration docstrings (`ResNet34FullB.lean:141`,
  `ResNet50FullB.lean:391`, `ResNet34StepTieB.lean:39`), never in a module docstring. ConvNeXt (Liu
  et al., "A ConvNet for the 2020s", CVPR 2022, arXiv:2201.03545) is never named; the files say only
  "the paper's `forward`" (`ConvNeXtFullT.lean:16, 215, 322, 444`) and follow its block, stem, head-LN
  and layer-scale design line for line. ResNet v1.5 (the stride on the 3×3) is credited only to
  "torchvision" (`ResNet50FullB.lean:37–44`). The R50 loss and optimizer ties follow the
  "ResNet strikes back" A3 recipe (Wightman, Touvron and Jégou 2021, arXiv:2110.00476: BCE over
  `N·K`, LAMB) and LAMB itself (You et al. 2019, arXiv:1904.00962). Those papers are credited in
  `content.tex:6313/6581` and `LeanMlir/Types.lean:439`, but not in the Lean files that state their
  loss (`ResNet50StepTieB.lean:586–590`).
  **Fix:** one reference line in each of those module docstrings. **Cost:** docstrings only.
  **Size:** S.

### generality

- **N1-gen-1** `LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTieGB.lean:474` (`cnx_net_tiedGB`),
  `ConvNeXtWholeBackCertifiedTieB.lean:531` (`convnextImagenetInputGradB_eq_vjp`), and
  `CnxTieWeights`/`CnxTWeightsCh`. **Problem:** both ConvNeXt ties are written at ConvNeXt-T's
  literal `[3,3,9,3]`/`96…768`. The repo also ships ConvNeXt-S and ConvNeXt-B renders
  (`convnextsin_*`, `convnextbin_*`, 18 artifacts) from one renderer, which is already generic
  over `CnxDims` (`api_design_audit.md` §5.2: `{depths, dims}` renders T byte for byte). The
  docstrings admit the gap honestly ("S and B are other nets"). But the proof is one stage lemma per
  depth (`convNextStageChK k`), so generalising the depths and dims is a special case the file
  could prove in general. **Fix:** state the weight record and the capstones over the stage depths
  `d₁…d₄` and the widths, with T, S and B as `#guard`ed instances. The literal-width `rfl` trap
  (cleanup_backlog §11 ⚠) argues for keeping the leaf ties at variable dims as they are now.
  **Cost:** L. It rewrites both tie files and `ConvNeXtFullT`, and touches the pins on
  `cnx_net_tiedGB` and `convnextImagenetInputGradB_eq_vjp` (tex 2, audit). **Size:** L.

- **N1-gen-2** `LeanMlir/Proofs/Nets/Small/{CnnFold,CifarFold,Cifar8StepTie,Cifar8BnStepTie,MlpFold}.lean`.
  **Problem:** every ImageNet net now has a whole-net parameter-gradient theorem
  (`r34_net_lossGrad`, `r50_net_lossGrad`, `cnx_net_lossGrad`, the MNv2, MNv4, B0 and ViT peers)
  saying each node at the chain cotangent is ∂L/∂θ. The five chapter nets still stop at "that `c`
  equals the loss gradient at a layer below the output is not stated" (for example
  `Cifar8StepTie.lean:82–84`). They are the simplest nets for the general `Foundation/ParamGrad`
  calculus (`pdiv_param_chain`, `HasGradAt`), and they have no BN coupling across examples.
  **Fix:** add `cnn_net_lossGrad` and `cifar_net_lossGrad` (then `cifar8` and `cifar8Bn`) over
  `Foundation/ParamGrad`, and replace the "not stated" sentences. **Size:** M each; `cnn` first.

### placement

- **N1-place-1** `LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:110–200`
  (`r34BFullHasVJPAt`, `r34BFullHasVJPAt_backward`). **Problem:** a net-agnostic combinator lives in
  the ResNet-34 file under a ResNet-34 name. The file itself says so ("Dimension-generic and
  parametric in every component"), and `ResNet50WholeBackCertifiedTieB.lean:17–21` says "It is
  ResNet-34's only by where it was written". R50 imports the whole ResNet-34 tie file to get it.
  Its inputs `opaqueA0…A16` already live in `Foundation/OpaquePrefix.lean`. **Fix:** move both
  declarations to `Foundation/OpaquePrefix.lean` under a length name (for example `vjpChain18At`,
  `vjpChain18At_backward`). Update the R34 and R50 tie files and the MNv2/MNv4 docstrings that cite
  it. MNv2's and MNv4's own apexes can follow as `vjpChainNAt` (MobileNet slice). **Cost:** 1
  AuditAxioms line; 0 tex/yaml/comparator; 4 Lean files; ±0 net lines. **Gate:** `lake build Certs`,
  the audit, docstring-checkrefs. **Size:** S.

### naming

- **N1-name-1** Namespaces `Proofs.CnnPoC`, `CifarPoC`, `Cifar8PoC`, `Cifar8BnPoC`, `MlpPoC`, `LinPoC`,
  `CnxPoC`, `CnxPoCG`, `CnxPoCGB`, `CnxTiePoC`, `CnxTiePoCGB`, and the module headers
  `/-! # PoC: …` of `LinearFold`, `CnnFold`, `CifarFold`, `MlpFold`, `Cifar8StepTie` and
  `Cifar8BnStepTie`. **Problem:** the shipped capstones sit in "proof-of-concept" namespaces while
  their ResNet peers are `ResNet34TieB`/`ResNet50TieB`. The file rename is done; the namespace step
  is open. **Fix:** as `cleanup_backlog.md` §8 step 4. **Cost:** tree-wide 188 AuditAxioms lines, 26
  tex, 10 certs.yml, 3 yaml, 3 comparator (`grep -c PoC`). (carried: `cleanup_backlog.md` §8, "the
  optional fourth step, the 30 `*PoC*` namespaces, is still open")

### documentation

- **N1-doc-1** `blueprint/src/content.tex:5263–5279` (`thm:resnet50_whole_back`, citing
  `r50InputGradB_eq_r34B_full_vjp` and `r50InputGradB_correct`). **Problem:** it overclaims. The book
  says the committed `r50InputGradB` "equals Σ_j pdiv(resnet50ForwardBFull) x i j · dy_j for every
  batch, cotangent and pixel". The Lean statement (`ResNet50WholeBackCertifiedTieB.lean:146–213`) is
  weaker in two ways. It is at sixteen opaque blocks `b1…b16` with caller-supplied
  `HasVJPDiffAt` witnesses, and its right-hand side is `pdiv (r34HeadB ∘ b16 ∘ … ∘ r34StemB)`. The
  identification with `resnet50ForwardBFull` is a separate shape check (`_eq_slots`), and the two
  are never composed, because the build notes record a kernel timeout. It also assumes `h_stem`,
  `h_pool` and `0 < q`, which the book entry omits. **Fix:** reword the book entry to the Lean
  statement. Name the opaque blocks and the shape check, and list the hypotheses. **Size:** S.

- **N1-doc-2** `LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTie.lean:201–205` and
  `ConvNeXtStepTieGB.lean:219–223`. **Problem:** "the carried `W`/`x` are generic — the same
  modelling the mnv2/r34 stems use" is false (see N1-corr-1). **Fix:** it goes with N1-corr-1.
  **Size:** S.

- **N1-doc-3** `LeanMlir/Proofs/Nets/ResNet/ResNet50FullB.lean:266` and the other R50 sites in
  N1-corr-3. **Problem:** stale. They anchor on the retired 76.66% run
  (`resnet50in160_lambaccdp8x64bce`). **Fix:** it goes with N1-corr-3 (decision D3 in
  `doc_honesty_pass.md` deferred only the book's R50 section, not these docstrings). **Size:** S.

### proof-quality

- **N1-pq-1** 28 `show`/`change` steps with no comment. `ResNet34FullBSeal.lean:168, 187, 190, 204,
  225, 250, 257, 269, 277`; `ResNet50FullBSeal.lean:201, 236, 239, 263, 266, 298, 305, 312, 323, 330,
  337, 349, 356, 364`; `ConvNeXtStepTieGB.lean:361`; `ConvNeXtWholeBackCertifiedTie.lean:91, 144,
  148`; `MnistCNN.lean:374, 380, 408, 442`. **Problem:** most of them unfold the `@[reducible]`
  block defs (`r34IdB = relu ∘ residual …`, `r34StemB = batchMap pool ∘ cbReluStridedB`, the R50
  peers) or a `R*SmoothAt` field by defeq. Every seal proof relies on that definitional shape and
  never says so, even though `r34IdB`/`r34DownB`/`r34StemB`/`r50*B` have no `_def`/`_apply` lemma
  (`grep "theorem r34IdB_"` finds only `_nonneg` and ParamGrad's `_hasGradAt_comp`).
  **Fix:** add `r34IdB_def`, `r34DownB_def`, `r34StemB_def` and the R50 peers in `ResNet34FullB`
  and `ResNet50FullB`, and `rw` with them. Where a `show` states the rewritten goal on purpose,
  a one-line comment is enough. For `MnistCNN`, `change` → `Fin.sum_univ_four` after
  `simp only [bnMean]`. **Size:** M.

## Checked, not findings

- **Seals.** `R34FullBSeal.sealX_backward_nontrivial` and `R50FullBSeal.sealX_backward_nontrivial`:
  a concrete witness (`sealW`, `sealX 0`) discharges every hypothesis of
  `resnet{34,50}ForwardBFullHasVJPAt` (`seal_pos`, `seal_smooth`, including the pool no-tie). The
  conclusion is an existential of a nonzero backward entry, derived from `fderiv ≠ 0` through
  `backward_nontrivial_of_fderiv_ne`. Both are on the full-width `*ForwardBFull`, at nCls generic
  with `0 < nCls`, and R50's at `0 < q ≤ 7`. Nothing is moved into hypotheses. Not vacuous.
- **Sync twins.** `r34_net_syncTiedB`/`r50_net_syncTiedB` take the scaled-shard `hgs` as a
  hypothesis, and it is discharged by `*_smoothedCE`/`*_bce` via `replicaLossCot_eq`. The replicas'
  saved activations are shards of the global forward, and `*FwdGraphSyncFull_shard` is the
  (uncomposed) forward half. Both points are stated in "What is NOT claimed". Sync-BN at R > 1 is the
  statement's subject, not per-replica BN; the counterexample `dpMeanGrad_ne_globalBatchGrad` is cited.
- **Class count and width.** R34/R50/ConvNeXt-GB are generic in `nCls`/`nC`.
  `convnextImagenetInputGradB_eq_vjp` is at 1000. The per-example `cnx_net_tied_certified` is at
  `CnxTieWeights 10`, matching its 10-class `convnext_train_step.mlir`. R50's `q` covers 160 and 224
  (`#guard`/`example`s at `ResNet50FullB.lean:255–268`).
- **Drop-path, EMA and wx/clip.** ConvNeXt's tie files state that the drop chain and the optimizer
  tails are outside the statement. R50's sync file does too. All of these are `∀ cot` node lemmas,
  which is correct.
- **ConvNeXt `*CotIn_eq_vjp`** (09-24 gap). Landed: `cnxBlockCotInChAt_eq_vjp`,
  `cnxDownCotInChAt_eq_vjp`, `cnxHeadDyXheadChN_eq_vjp` and their batched `…B_eq_vjp`
  (`ConvNeXtStepTieGB.lean:324+`). The `show` at :361 is the only proof-quality blemish.
- **Small-CNN dense heads** (09-24 gap). Landed in `a4c268d9`: 10/14/22/38 conjuncts, with the
  `DenseWSgdTied`/`DenseBSgdTied` clauses at `g`, `mlpCotOut1` and `mlpCotOut0`. The docstrings now
  say which layers the loss-gradient reading covers.
- **Step-tie shape.** Every capstone (R34/R50/ConvNeXt/small) is an ∧-bundle of `∀ cot` fold
  lemmas instantiated at `let`-bound chain cotangents. Standing alone, it is exactly as strong as
  those lemmas. All files say so ("the folds are `∀ cot` statements instantiated at explicitly
  constructed cotangents"). The cotangent-is-VJP half is the separate `*CotIn_eq_vjp`; the ∂L/∂θ
  half is ParamGrad's. This is a deliberate split, so not a reuse finding.
- **`r34InputGradB_correct` / `r50InputGradB_correct` Lean docstrings** are accurate about the
  opaque blocks and the hypotheses (only the book overclaims; N1-doc-1).
- **`CnxTieWeights` vs `CnxTWeightsCh`** (api-design §7.3). Bridged by
  `CnxTieWeights.forward_eq_convNextForwardTCh` (`ConvNeXtStepTie.lean:594`). Unused, but it is the
  requested bridge.
- **`MlpCanonical.lean`.** Eight `noncomputable def` specialisations whose only consumer is the
  audit, by the file's stated design ("checkable canonical surface"). Below the bar.
- **`ConvNeXtFoldGB`.** audit_v2's "10 pinned alias fold lemmas" are gone; there are 3 lemmas + 2 clauses.
- **Long spans** (census: `cifar8Bn_train_step_tied_certified` 139, `cnx_net_tied_certified` 116, …)
  are statement bodies (`let` chains + conjuncts) with one-line proofs, not long proofs.

## Gaps for the humans

- No gate checks that the artifact a step tie names is one a trainer loads, or that the artifacts
  behind the book's reported numbers have a tie. N1-corr-2 and N1-corr-3 are both that.
  `check_target_names.sh` could grow a "tied artifact has a consumer" check.
- No gate flags a tie whose certified function takes a free input that is not in the forward
  chain (the `xstem` shape of N1-corr-1).
- The rubric's reuse line "∧-bundles of existing lemmas" fits every step-tie capstone here
  literally. For this repo the bundle is the deliverable (it pins the chain cotangents), so it was
  not reported. The rubric needs a local exemption line.
