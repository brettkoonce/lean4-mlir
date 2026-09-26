# Slice G — LeanMlir/Proofs/Nets/MobileNet/ (MobileNetV2 + MobileNetV4)

**Coverage.** Read in full: MobileNetV4Spec, MobileNetV2 (.lean), MobileNetV2Fold, StagesPC, StagesPCEval,
FullPaper, Close, MobileNetBackChains, MobileNetV2BackB0, MobileNetV2FullB, MobileNetV2FullBVJP,
MobileNetV2StepTieB, MobileNetV2WholeBackCertifiedTieB, MobileNetV4BackB0, the MobileNetV4FullB header
and every docstring in it. Read the docstrings plus the statements of the main theorems in:
MobileNetV2FullBSeal (the start and the end in full), MobileNetV2FullPaperEval, MobileNetV2SyncB,
MobileNetV2SyncStepTieB, MobileNetV4FullBVJP, MobileNetV4FullBSeal, MobileNetV4StepTieB, MobileNetV4SyncB,
MobileNetV4SyncStepTieB, MobileNetV4WholeBackCertifiedTieB. Skimmed: the `_smul`/`_shard` helper bodies.
Checked against the artifacts: `verified_mlir/` signatures (argument counts, `%x` shapes, SSA names) and the timm 1.0.28 Conv-M layout.
Checked for `sorry`, `axiom`, `admit`, `native_decide`, `implemented_by` and `@[extern]` in all 25 files: none occur, so the
"3-axiom clean" prose in MobileNetV2.lean and StagesPCEval is consistent with the source. I did not run `#print axioms`.

---

### LeanMlir/Proofs/Nets/MobileNet/MobileNetV4StepTieB.lean:25 (module doc), :801 `mnv4_net_tiedB`; also MobileNetV4WholeBackCertifiedTieB.lean:8

**Kind:** overclaim
**Says:** module: "⭐⭐ **And every block's `*CotIn_eq_vjp` is its block layer's `.faithful`, not a new derivation.** … This file composes certified VJPs; it does not re-prove them." Capstone: "Threading … down through the certified head backward, the 21 certified UIB block backwards and the fused stage, every parameter GRADIENT node … denotes the certified batched `Σ_n` gradient." WholeBack header: "MobileNetV4-Conv-M is certified from its ℝ forward (T1) through its typed graph (T2), its 233-parameter train-step tie (T3) and now its whole-net input gradient."
**Actually states:** no MobileNetV4 `*CotIn_eq_vjp` lemma exists anywhere in the repo. The only occurrence of `eq_vjp` in these files is this sentence; the other hits are the WholeBack endpoint ties. `mnv4BodyCotIn`, `mnv4SBodyCotIn`, `mnv4HeadCotIn` and `mnv4FusedCotIn` are used only in StepTieB and SyncStepTieB. No theorem equates any of them with `(mnv4BodyOfRow …).vjp … .backward` or with the head's or the fused stage's VJP. `mnv4_net_tiedB` ties each gradient node to the certified Σₙ gradient at a *constructed* cotangent. Nothing proves that the constructed cotangent chain is the certified block backward, which MobileNetV2 does with its four `*CotIn_eq_vjp` lemmas (the gap is group 6 of the prior audit).
**Fix:** module: "Each block's input cotangent is built from the same per-op backwards the block `CertLayer`s certify. ⚠ No `*CotIn_eq_vjp` is stated for MobileNetV4, so it is not proved here that the constructed chain equals the blocks' certified VJP backwards." Capstone: "…down through the constructed head, UIB-block and fused-stage backward chains (the render's node-for-node spelling), every parameter gradient node denotes the certified batched Σₙ gradient at that chain's cotangent." Drop "certified" before "UIB block backwards" and "head backward". In the WholeBack header, say that T3 ties the parameter nodes at the chain's cotangent and does not identify the chain with the certified VJP.

### MobileNetV2WholeBackCertifiedTieB.lean:356 `mnv2InputGradB_correct`; MobileNetV4WholeBackCertifiedTieB.lean:530 `mnv4InputGradB_correct`

**Kind:** overclaim
**Says:** "⭐⭐ **The batched chain IS the `pdiv`-contracted Jacobian of the twenty-one-stage net** — at every batch size, every input, every loss cotangent and every input pixel." The MNv4 version adds "every class count".
**Actually states:** the equation holds only for `x` satisfying `h_stem : MNV2StemSmoothAtB … x` and `h_head : MNV2HeadSmoothAtB …` (the MNv4 version needs `Mnv4StemSmoothAtB` and the head's relu clauses). It also needs seventeen (MNv4: twenty-two) *supplied* `HasVJPDiffAt` block witnesses. The block backwards in `mnv2InputGradB` are those witnesses' `.backward`s, and the blocks `b1 … b17` are free variables. The statement is about an arbitrary chain of certified stages, not the concrete MobileNet blocks. The file explains that last point in its header, but the theorem docstring drops it along with the smoothness hypotheses.
**Fix:** "At every batch size and loss cotangent, and at every input where the stem and head relu6 (MNv4: relu) clauses hold: the chain, with its seventeen block slots filled by any certified block VJPs at the running activations, IS the `pdiv`-contracted Jacobian of the twenty-one-stage composition. `mobilenetv2ForwardBFull_eq_slots` identifies the stages with the committed forward. The blocks themselves stay opaque (see header)."

### Per-replica claims written before sync-BN — MobileNetV2StepTieB.lean:34–38; MobileNetV2WholeBackCertifiedTieB.lean:50–52; MobileNetV4StepTieB.lean:76–77 and :817–818; MobileNetV4WholeBackCertifiedTieB.lean (header "What this does NOT reach", and :408–410); MobileNetV4FullBVJP.lean:47–48

**Kind:** stale (overclaim in effect)
**Says:** e.g. MNv4 StepTieB: "⛔ **One replica.** Under `mnv4in_adamdp64*` every node named here feeds `allReduceMeanF` (`DataParallelNode.lean`, §4d); this is the per-replica gradient." MNv2 StepTieB: "Every statement below is at the PER-REPLICA gradient node; `DataParallelNode.lean` composes it with the mean and the tail." MNv4 FullBVJP and WholeBack: "`N` is a binder … and the artifacts' `N` is the PER-REPLICA batch (`DataParallel.lean`, §4d)."
**Actually states:** since the sync-BN switch, every data-parallel artifact normalises over the global batch. The Sync* twins in this same directory say so: "replica `r`'s forward graph denotes shard `r` of `mobilenetv4ForwardBFull (R * N)`". A replica's gradient node is therefore *not* the single-device node at the per-replica `N`: its BatchNorm backward is `bnSyncInB`, not `bnInB`. The single-device theorems describe a DP artifact only at `N := R·N`, through `mnv2_net_syncTiedB` / `mnv4_net_syncTiedB`. Read as written, these sentences claim the theorems describe what each replica computes, and that was true only of the retired per-replica-BN renders.
**Fix:** "On the data-parallel (sync-BN) artifacts this theorem applies at `N := R·N`; the per-replica statement is `MobileNetV{2,4}SyncStepTieB`'s `mnv*_net_syncTiedB`, whose right-hand side is this theorem's node at the global batch." Replace "the artifacts' `N` is the PER-REPLICA batch" with "on DP artifacts, instantiate at the global batch `R·N` (see the Sync twin)".

### MobileNetV4FullB.lean header "artifacts" row and "MNv4 ships one resolution"; :807 `mnv4FwdGraphBFull`; MobileNetV4FullBSeal.lean:945 `sealX_backward_nontrivial`; MobileNetV2FullBSeal.lean:1040 `sealX_backward_nontrivial`

**Kind:** overclaim
**Says:** FullB table: "artifacts | `mnv4_fwd`, `mnv4_fwd_eval`, `mnv4_adam_train_step`, and the five `mnv4in*` ImageNet twins". FullB: "Unlike R50 there is no `q` binder — MNv4 ships one resolution." `mnv4FwdGraphBFull`: "the typed graph diffs against `mnv4_fwd.mlir` and its five ImageNet twins name for name". MNv4 seal: "`mobilenetv4ForwardBFull`, the forward every MobileNetV4 artifact runs". MNv2 seal: "`mobilenetv2ForwardBFull`, the forward every MobileNetV2 artifact runs".
**Actually states:** `mobilenetv4ForwardBFull` is the *training*-BatchNorm forward at 224×224 with no dropout. `verified_mlir/` holds nine `mnv4in*` files, not five. `mnv4_fwd_eval` and `mnv4in_fwd_eval` take frozen statistics (388 inputs, e.g. `%stnmu`), and no MNv4 eval forward is stated anywhere. `mnv4in_fwd_eval_s256` takes `%x: tensor<64x196608>`, which is 3·256·256. `mnv4in_emaacc{,dp}8x128wxdowd005bf16` take a classifier dropout mask `%do: tensor<128x1280>`. On the MobileNetV2 side, `mobilenetv2{,in}_fwd_eval*` run `mobilenetv2ForwardPaperEval`, not `mobilenetv2ForwardBFull`.
**Fix:** FullB: "artifacts | `mnv4_fwd`, `mnv4_adam_train_step`, and the 224×224 `mnv4in` train steps and forward. ⚠ Not covered: the frozen-statistics evals (`mnv4{,in}_fwd_eval`), the 256×256 eval `mnv4in_fwd_eval_s256`, and the dropout variants `mnv4in_emaacc*`." Remove "MNv4 ships one resolution", or change it to "MNv4 trains at one resolution". In both seals, "the training-BatchNorm forward every MobileNetV2/V4 *train step* runs (at N = 2 here)".

### MobileNetV4FullBVJP.lean:9–11 (module doc) — MNv4-vs-timm parity

**Kind:** stale
**Says:** "⚠⚠ **No accuracy is quoted for this net.** Conv-M has no Imagenette run and no verified ImageNet run; what pins the artifact to the reference's function is the pair of ties re-run 2026-09-07 (forward `max |Δ| = 3.770e-06`, gradient inside the reference's own fp32 floor)."
**Actually states:** the net changed on 2026-09-24, when it was brought to timm parity (stride on the post-DW, BN-only pre-DW, a relu stage 0, the pool before `conv_head`, a symmetric stem). The 2026-09-07 numbers pin the *pre-parity* function. MobileNetV4FullB.lean's header names the current gates: `scripts/parity/mnv4_timm_parity.py`, `mnv4_forward_tie.py` and `grad_tie.py --net mnv4`.
**Fix:** "No accuracy is quoted for this net. The statements are about timm's `mobilenetv4_conv_medium` (`planning/mnv4_timm_parity.md`). The artifacts are pinned to it by `scripts/parity/mnv4_timm_parity.py`, `mnv4_forward_tie.py` and `grad_tie.py --net mnv4`." Drop the dated numbers.

### MobileNetV4FullB.lean header ("~60 relu clauses"), :354; MobileNetV4FullBVJP.lean:26 and :94 (`Mnv4SmoothAt`)

**Kind:** stale / wrong
**Says:** "~60 relu clauses across the net, none of them written here"; `Mnv4SmoothAt`: "Roughly sixty clauses in total, none of which had to be written down." FullB:351–352: "`Mnv4SmoothAt` binds one `.ok` per group (seven)".
**Actually states:** the net has 38 relu sites. `MobileNetV4FullBSeal` counts them with a `#guard` (`1 + 1 + Σ(1 + [postDWk≠0]) + 2 = 38`), and its header says "**38** relu sites in all". `Mnv4SmoothAt` has **eight** fields (the stem plus seven groups), as FullBVJP's own section title "Eight fields, not thirty-five" and FullB's earlier "eight fields rather than two" say.
**Fix:** "38 relu clauses (the seal's `#guard` counts them), none of them written here"; "binds one `.ok` per group plus the stem's clause (eight fields)".

### MobileNetV4BackB0.lean:463–467; MobileNetV4FullB.lean header ("Rows 4/5/10, 12/18 and 15/19/20 are shape-identical") and :163 `Mnv4BWeights`; MobileNetV4WholeBackCertifiedTieB.lean:717–719 `mobilenetv4ForwardBFull_eq_slots`

**Kind:** stale (wrong rows)
**Says:** BackB0: "rows 4, 5 and 10 are all `160 → 160, expand 4`, and 4/10 share `k = 3,3`. Their records are therefore the *same type*". FullB, Mnv4BWeights and WholeBack: "shape-identical rows (4/5/10, 12/18, 15/19/20)".
**Actually states:** in `mnv4Blocks`, row 10 is `⟨"10",160,160,4,3,0,14,false⟩` (a ConvNeXt-like row with `postDWk = 0`). Rows 4 and 5 are `(3,3)`, so row 10 does not share `k = 3,3` with row 4. `UibParams s` types `Wd : DepthwiseKernel _ s.postDWk s.postDWk`, so row 10's record does not have row 4's field shapes. The shape-identical sets are {4, 5, 7}, {8, 10}, {12, 18}, {13, 14} and {15, 19, 20}. These row lists read like a leftover from Conv-S.
**Fix:** "Rows 4/5/7, 8/10, 12/18, 13/14 and 15/19/20 are shape-identical (same `ic, oc, expand, preDWk, postDWk, h`)". Fix all four places.

### MobileNetV2FullPaperEval.lean:5–9, :19–21 (module doc) and :301 `mobilenetv2FwdGraphPaperEval_faithful`

**Kind:** stale
**Says:** "The eval twin of `MobileNetV2FullPaper.lean`. That file states the seventeen-block `[t,c,n,s]` net at TRAINING BatchNorm, the world its VJP and its typed graph live in". "The seventeen-block TRAINING graph names its parameters `%b17gp`/`%b17btp`, where the render emits `%gp17`/`%btp17`." Theorem: "the training twin's recipe with `bnPerChannelEvalF_faithful` in place of `bnPerChannelF_faithful`."
**Actually states:** MobileNetV2FullPaper.lean now holds only the `IVW`/`IVWNoExp` records and positivity bundles. Its own header says the forward, graph and faithfulness were "retired on 2026-09-20". The training twin is `MobileNetV2FullB.lean`, at `bnBatchLA`/`.bnBatchF`, and its graph writes `%b17pg`/`%b17pbt`. `mobilenetv2_adam_train_step.mlir` emits exactly those names (`%b1pg %b1pbt …`), so the training graph *matches* its render. Only the eval render uses `%gp17`.
**Fix:** "The inference twin of `MobileNetV2FullB.lean`, the batch-BN training forward. It reuses `MobileNetV2FullPaper.lean`'s weight records." "The training graph (`MobileNetV2FullB`) names parameters `%b{k}{e,d,p}{W,g,bt}` as the train-step render does; the eval render uses `%gp{k}`/`%btp{k}`, and this graph follows it." Drop the `bnPerChannelF_faithful` comparison.

### MobileNetV2FullB.lean:9–15, :27–29, :64–65 (module doc) and :86 `MNV2BWeights`

**Kind:** stale
**Says:** "`MobileNetV2FullPaper.lean` states this net's whole-net ℝ forward and typed graph at **per-example** BatchNorm (`bnPerChannelTensor3`, reduce `[2,3]`)." "`MobileNetV2BackB0.lean` already carries the batched relu6 stages (`cbrB`, `dwbrB`, `dwbrBstrided`, and `projB` from `BatchedStages`)". "it is pinned only where a `Maps` envelope turns a width into a rational (T4/T5)." `MNV2BWeights`: "the lesson `MobileNetV2FullPaperEval.lean` and B0's eval twin both paid for is that the head's envelope depends on the fan-in".
**Actually states:** FullPaper.lean has no forward and no graph (both retired 2026-09-20). `cbrB`, `dwbrB` and `dwbrBstrided` are defined in `Foundation/BatchedStageLayers.lean`, and `projB` in `Foundation/BatchedStages.lean`. There is no MobileNetV2 T4/T5 or `Maps` envelope in the tree: the float tier was cut. The "envelope" lesson refers to that deleted budget tier.
**Fix:** "`MobileNetV2FullPaper.lean` holds the weight records this net reuses." "The batched relu6 stages (`cbrB`, `dwbrB`, `dwbrBstrided` in `BatchedStageLayers`, `projB` in `BatchedStages`) and their `_at` VJPs are one tier down (`MobileNetV2BackB0.lean` composes them)." Delete the T4/T5 sentence. `MNV2BWeights`: "Generic in `nCls`, so one record covers the 10- and 1000-class artifacts."

### MobileNetV2StepTieB.lean:656–660 `mnv2_net_tiedB`; :20–24 (module doc)

**Kind:** overclaim
**Says:** "Those enter only in the four `*CotIn_eq_vjp` lemmas, which say the constructed chain IS the certified whole-net backward". Module: "instantiated at `b3`, `b5`, `b6`, `b8`–`b13`, `b15`, `b16` (skip) and `b11`, `b17` (no skip)".
**Actually states:** the four lemmas (`mnv2NoExpCotIn_eq_vjp`, `mnv2ExpOnlyCotIn_eq_vjp`, `mnv2ResidCotIn_eq_vjp` and `mnv2StridedCotIn_eq_vjp`) cover the *block* backwards only. No lemma ties the head chain (`mnv2HeadCotBlk`) or the stem cotangents to `mnv2HeadBHasVJPAt` / `mnv2StemBHasVJPAt`, and nothing composes the four into a whole-net identity. The skip list "b8–b13" includes b11, which is the no-skip widening.
**Fix:** "…the four block-level `*CotIn_eq_vjp` lemmas, which say each block's constructed backward IS its certified block VJP backward. The head and stem segments of the chain are not separately tied to a VJP." For the skip list: "`b3, b5, b6, b8, b9, b10, b12, b13, b15, b16` (skip)".

### MobileNetV2FullBSeal.lean:20–21 (module doc) and :67–69 (§1 comment)

**Kind:** stale / wrong
**Says:** "`β = 3` wherever a relu6 follows, `β = 0` at the eleven linear-bottleneck projections, which no activation follows".
**Actually states:** every one of the seventeen bottlenecks has a project BN, and `sealIVW`, `sealResW` and `sealNoExpW` all set `pβ := kv _ 0`. The 52 BN sites minus the 35 relu6 sites leave 17.
**Fix:** "`β = 0` at the seventeen linear-bottleneck projections".

### MobileNetV4FullBVJP.lean:171 `mobilenetv4ForwardBFullHasVJPAt_correct`

**Kind:** overclaim (prior-audit group 2)
**Says:** "⭐ And it IS the `pdiv`-contracted Jacobian of the whole net, at every batch size and both shipped class counts. The reading that says the object above is the gradient rather than merely a function of the right type."
**Actually states:** the proof is `(mobilenetv4ForwardBFullHasVJPAt N w x hx).correct dy i`, which projects the witness's own `.correct` field. It certifies nothing beyond the `HasVJPAt` value. It also holds only under `hx : Mnv4SmoothAt N w x`, which the docstring omits.
**Fix:** "The `.correct` field of `mobilenetv4ForwardBFullHasVJPAt`, restated: at a point satisfying `Mnv4SmoothAt`, its backward is the `pdiv`-contracted Jacobian of `mobilenetv4ForwardBFull`."

### MobileNetV2.lean:3–31 (module doc) and :367 `mobilenetv2HasVJPAt_correct`

**Kind:** overclaim
**Says:** "Builds a representative MobileNetV2 forward and proves its end-to-end vector–Jacobian product correct"; the bullets describe "1×1 conv → bn → relu6"; the theorem: "the full MobileNetV2 backward equals the `pdiv`-contracted Jacobian".
**Actually states:** the "bn" is `bnForward n ε γ β` with **scalar** `γ β : ℝ`, a single normalisation over the whole flattened tensor rather than per-channel BN. The net has two blocks with no strides and no head conv. The theorem holds only at points satisfying five `≠ 0 ∧ ≠ 6` families and is the witness's `.correct` projection.
**Fix:** "a two-block MobileNetV2-shaped toy (whole-tensor BN with scalar γ/β, stride 1 throughout)…"; theorem: "the two-block representative's backward equals the `pdiv`-contracted Jacobian at a point where all five relu6 families are away from 0 and 6 (the witness's `.correct` field)."

### MobileNetV2Close.lean:1–44 (module doc)

**Kind:** stale / process-narrative
**Says:** "`planning/archive/mobilenetv2_close.md` Item C … every MobileNetV2 train-step parameter output denotes `θ − lr·(certified Jacobian · cotangent)`. The MobileNetV2 train step (`TestMobilenetV2Train.lean` (retired 2026-09-20)) has these parameter families, and each is now certified by the bridge in the right column". Table row: "depthwise W stride 1 (`dW`, blocks b2,b4)".
**Actually states:** the SGD train step it certifies is gone (`tests/TestMobilenetV2Train.lean` does not exist). The file now serves as a library of generic depthwise and strided bias/weight SGD bridges, used by `MobileNetV2Fold` (and through it `ConvNeXtStepTie`) and by `ResNet34Fold`. In the shipped paper table, b2 and b4 are the *strided* blocks.
**Fix:** rewrite as "Generic SGD bridges for the stride-1 depthwise bias (`mnv2_render_depthwiseb_certified`) and the stride-2 conv weight/bias (`mnv2_render_stem_conv{W,b}_certified`), each `θ − lr·(certified ∂/∂θ · c)` for a free cotangent `c`. Used by `MobileNetV2Fold.lean` and `ResNet34Fold.lean`." Drop the retired-step table.

### MobileNetV2BackB0.lean:30–45 (module doc "Structure")

**Kind:** stale / process-narrative
**Says:** "* `cbrB` / `dwbrB` — batched conv/depthwise → bn → **relu6** stages … with `_at` differentiability + VJP and backward-graph faithfulness (`cbrBackBatchedGraph` + `…_faithful`). * `cbrLayer` / `dwbrLayer` / `dwbrStridedLayer` / `projLayer` — the four stages as `CertLayer`s." Also: "The project stage and the residual fan-in are reused VERBATIM from the EfficientNet file"; "⚠ This family takes `ic` and `oc` SEPARATELY (2026-09-06). It was written for the residual block and pinned them equal…"
**Actually states:** none of `cbrB`, `dwbrB`, `cbrBackBatchedGraph`, `cbrLayer`, `dwbrLayer`, `dwbrStridedLayer` or `projLayer` is defined here. They live in `Foundation/BatchedStageLayers.lean`, with `projB` in `Foundation/BatchedStages.lean` and `residualBackGraph` in `Foundation/BatchedBackLinks.lean`. The file defines only the two body `CertLayer`s, their VJP/diff/graph wrappers and the residual capstone.
**Fix:** "Uses the batched relu6 stages and their `CertLayer`s from `Foundation/BatchedStageLayers` (`cbrLayer`, `dwbrLayer`, `dwbrStridedLayer`) and `projLayer`; this file composes them into `mnv2BodyLayer` / `mnv2DownBodyLayer`…". Drop the dated `ic`/`oc` history.

### MobileNetV2SyncStepTieB.lean:984; MobileNetV4SyncStepTieB.lean:1298

**Kind:** overclaim
**Says:** "so this and it together say the DP step's update is the certified gradient of the global-batch step."
**Actually states:** both capstones are about the all-reduced **gradient** nodes. The optimizer tail (RMSProp/AdamW, EMA, gradient accumulation in `mnv4in_emaaccdp8x128…`) is not stated. Separately, for MNv4 the chain's certified-ness rests on the missing `*CotIn_eq_vjp` (first finding).
**Fix:** "…say every all-reduced gradient the DP render emits is the global-batch step's gradient node."

### MobileNetV4BackB0.lean:8–10, :42, :220–222

**Kind:** stale
**Says:** module: "the depthwise-relu stages, the UIB body, the skip block, the stride-2 form, the fused stage and the head, each a `CertLayer`"; the comment at :220: "The UIB **expand** … and **project** … stages are `ResNet34BackB0`'s `cbReluLayer` and `projLayer`." The fused-stage note says "the strided conv-bn-relu stage is ResNet's `cbReluStridedLayer`".
**Actually states:** no skip-block `CertLayer` is defined here; callers apply `CertLayer.residual`. `cbReluLayer`, `cbReluStridedLayer` and `projLayer` are defined in `Foundation/BatchedStageLayers.lean`, not ResNet34BackB0.
**Fix:** "…the UIB body (the skip is `CertLayer.residual` at the call site)…"; "…are `Foundation/BatchedStageLayers`'s `cbReluLayer` and `projLayer`."

### MobileNetV4FullBSeal.lean:62–64 (§1 comment)

**Kind:** stale (minor)
**Says:** "`β = 160` at every BatchNorm a relu follows; `β = 0` at the projections"
**Actually states:** `sealP` also sets `bq2 := kv s.ic 160` at the BN-only pre-DW, where no relu follows.
**Fix:** "`β = 160` at every BatchNorm except the projections' (`β = 0`)".

### Process narrative (module and declaration docstrings)

**Kind:** process-narrative
**Says / where:**
- MobileNetV4Spec.lean:29–34 `mnv4Blocks`: "Before it existed the same rows were hand-written FOUR times, and §3/§7.2's whole point is…"
- MobileNetV4BackB0.lean:260–261, :296–297, :314–315: "Until 2026-09-24 these rows were PRE-strided…", "Until 2026-09-24 this stage was swish…", "Until 2026-09-24 `conv_head` ran at 7×7…"; the comments at :403–412 ("REWRITTEN FOR CONV-M (2026-08-14). `ed5a797` swapped…") and :424–426 ("A docstring once named Conv-S's families here … for four weeks").
- MobileNetV4FullB.lean header: "this cost a day to establish", "planning/archive/mnv4_proofs_tier.md is the plan; ResNet-50 closed the same two tiers on 2026-09-06", "the rerun on timm's net is queued" (a status note); `mnv4StemB` (:261) "Until 2026-09-24 this was the XLA-`SAME` phase".
- MobileNetV4StepTieB.lean:42 "Until 2026-09-24 the stem was XLA-`SAME`"; :54 "records four separate kernel blow-ups". FullB says "Three separate blow-ups" and WholeBack says "six blow-ups across sessions 1–2", so the three counts disagree.
- MobileNetV4WholeBackCertifiedTieB.lean:8–11: "Tier **T6** of `planning/archive/mnv4_proofs_tier.md` §Session 3 … T4 and T5 are float budgets and `planning/archive/float_budget_numbers.md` closed that thread by user decision on 2026-09-05."
- MobileNetV2FullB.lean:17–23: "⛔ MobileNetV2's two renderers did not overlap … ⭐ Since 4c leg 2 (2026-09-06) it is the ONLY renderer…".
- MobileNetV2FullBVJP.lean:7, :412–414: "the second piece of `formalization.yaml` 4e's port"; "(the per-example fold it was the batched peer of was retired 2026-09-19)". FullPaper.lean:298–300 dates the same retirement 2026-09-20.
- MobileNetV2StepTieB.lean:8–12, :34–37: "`GradNodesB` (4b.4)", "With 4.2b it completes MobileNetV2's T3", "since 4d piece 2 (2026-09-07) … until then `emitGradAllReduce`…".
- MobileNetV2WholeBackCertifiedTieB.lean:8–12, :61: "retired 2026-09-19 with `MobileNetV2PaperWholeBackCertifiedTie.lean`", "(moved here 2026-09-19 from the retired per-example tie…)".
- MobileNetV2FullPaper.lean:298–300 and MobileNetV2FullPaperEval.lean:10–12: retirement and budget-deletion dates.
- MobileNetV2FullBSeal.lean:11–12: "until now that was exhibited only on a per-example, two-block, 2-channel proxy, deleted when this file landed."
**Fix:** delete the dated history and plan or section numbers; keep the mathematical point (e.g. "the post-DW carries the stride (timm `dw_mid`)", "stage 0 is relu"). Move the before/after notes and the timing measurements to commit messages or a planning doc. Replace status notes ("rerun queued") with nothing, or with a pointer to where results live.

---

## Overclaims (fix before anyone reads the published results again)

1. **MobileNetV4StepTieB module and `mnv4_net_tiedB`; WholeBack header:** claim that `*CotIn_eq_vjp` lemmas and "certified UIB block backwards" exist. No MobileNetV4 `*CotIn_eq_vjp` exists, so the T3 chain is never identified with the certified VJPs.
2. **`mnv2InputGradB_correct` / `mnv4InputGradB_correct`:** "every input" drops the stem and head smoothness hypotheses and the supplied block-witness hypotheses, and the blocks are opaque variables.
3. **Single-device ties described as "the per-replica gradient" and `N` as "the PER-REPLICA batch"** (MNv2 StepTieB and WholeBack; MNv4 StepTieB, WholeBack and FullBVJP): false since sync-BN. On DP artifacts they apply only at `N := R·N`, through the Sync twins.
4. **Artifact coverage:** the MNv4 FullB artifacts row counts "five" `mnv4in` twins where there are nine. It lists the frozen-statistics `mnv4_fwd_eval` under the training-BN forward, ignores the s256 eval and the dropout (`%do`) variants, and says "MNv4 ships one resolution". Both seals say "the forward every MobileNetV2/V4 artifact runs", which the eval artifacts do not.
5. **`mobilenetv4ForwardBFullHasVJPAt_correct`:** a pure `.correct` projection presented as the reading that makes the backward "the gradient". It also omits `Mnv4SmoothAt`.
6. **MobileNetV2StepTieB capstone:** "the four `*CotIn_eq_vjp` lemmas … say the constructed chain IS the certified whole-net backward". They cover blocks only, not the head or stem segments.
7. **MobileNetV2.lean:** the "representative MobileNetV2" uses scalar-γ whole-tensor BN, and `mobilenetv2HasVJPAt_correct` is called "the full MobileNetV2 backward".
8. **Sync capstones (MNv2, MNv4):** "the DP step's update is the certified gradient". The theorems are about gradients; the optimizer and EMA tail is not stated.
