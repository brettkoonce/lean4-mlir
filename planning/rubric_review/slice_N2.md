# Slice N2 — Nets: MobileNet (V2+V4), EfficientNet, ViT: rubric review 2026-09-30

Scope: `LeanMlir/Proofs/Nets/{MobileNet,EfficientNet,ViT}/`, except `*ParamGrad.lean`. The audit
tree is at `55ad3a5a`. Weight went to `EfficientNetFullB0Drop.lean` and `MobileNetV4FullBEval.lean`
(both new), to `EfficientNetSyncStepTieG.lean` and `MobileNetV4FullBSeal.lean`, and to the rest of
`git diff --stat 373059db HEAD`. Nothing was edited and nothing was typechecked; every finding was
checked against the source, the artifacts in `verified_mlir/` or timm 1.0.28 (`.venv-timm`).

## Verdicts

| angle | verdict | findings |
|---|---|---|
| correctness | request_changes | 2 |
| reuse | request_changes | 4 |
| scope | request_changes | 1 |
| attribution | request_changes | 3 |
| api-design | approve | 0 |
| generality | request_changes | 1 |
| placement | request_changes | 2 |
| naming | request_changes | 1 |
| documentation | request_changes | 10 |
| proof-quality | request_changes | 4 |

## Findings

### correctness

- **N2-corr-1** `LeanMlir/Proofs/Nets/ViT/ViTDepthK.lean:239-274` — `vitTinyHasVJP_correct`, and
  the `vitForwardKVHasVJP_correct` it instantiates (:213-233), are the witness's own `.correct`
  field restated.
  - **Why that is empty:** `HasVJP` is a `Subsingleton` (`Foundation/Tensor.lean:282`) and
    `HasVJP.canonical` (:286) inhabits it for every `f`. So "`(vitForwardKVHasVJP …).backward x dy i
    = ∑ pdiv (vitForwardKV …) x i j * dy j`" holds for any map at all.
  - **What the docstring claims instead:** "the production capstone … non-degenerate by
    construction … a full-spec, real-architecture whole-network backward". The book makes it the
    chapter's destination (content.tex ~10854, ~11044, ~12056).
  - **Same shape elsewhere in this slice:**
    - `efficientnetForwardBFullHasVJP_correct` (`EfficientNetFullB0.lean:452-456`, "Public
      correctness theorem");
    - `mobilenetv2ForwardBFullHasVJPAt_correct` (`MobileNetV2FullBVJP.lean:513-518`, "Public
      correctness theorem"; `HasVJPAt`, Tensor.lean:437, has no differentiability field, so it is
      the same shape).
    - MNv4's twin was already reworded honestly and is the template for the others:
      `MobileNetV4FullBVJP.lean:173`, "The `.correct` field of … restated".
  - **Where the real tie is:** the non-vacuous content is `vitInputGradK_correct` /
    `vitTinyInputGrad{,B}_eq_vitTiny_vjp`, where a hand-written chain equals the backward. For B0
    it is `efficientnetInputGradBFull_eq_efficientnetForwardB_full_vjp`.
  - **Fix:** restate `vitTinyHasVJP_correct` as the ViT-Tiny instance of `vitInputGradK_correct`
    (LHS `vitInputGradK 3 224 224 16 196 768 3 64 nCls 12 … x dy i`). The name and blueprint label
    stay, and the proof is one term. Reword the B0 and MNv2 docstrings to MNv4's form. Also
    `ViTBackNet.lean:27-30`, see N2-doc-10.
  - **Evidence:** ViTDepthK.lean:233, whose proof is `(vitForwardKVHasVJP …).correct x dy i`;
    Tensor.lean:270-290, whose docstring says exactly this.
  - **Cost:** `vitTinyHasVJP_correct` pins are AuditAxioms 1, content.tex 4, formalization.yaml 2,
    `gen_comparator_tier.py` 1 (the Challenge statement changes). The docstring-only rewordings
    cost 0.
  - **Size:** S.
  - (carried: `doc_audit/find_B_blueprint2.md` entry 1, for the book prose, and
    `find_G_mobilenet.md:97`, whose MNv4 site is fixed. The Lean docstrings of the ViT, B0 and MNv2
    sites were not touched.)

- **N2-corr-2** `LeanMlir/Proofs/Nets/EfficientNet/EfficientNetFullB0Eval.lean:15-27`
  (`efficientnetFwdGraphBFullEval_faithful`); also `EfficientNetFullB0Drop.lean`
  (`efficientnetFwdGraphBFullEvalDrop_faithful`).
  - **Problem:** B0's inference forward is the graph behind every reported B0 accuracy, and it has
    no text tie to its artifacts: `efficientnet{,in}_fwd_eval*.mlir` and
    `efficientnet_{drop,do}_fwd_eval.mlir`.
  - **The module docstring lists four differences from the artifact:**
    - a bias slot per conv that the render folds away;
    - `%smu`/`%svar` against `%stnmu`/`%stnvar`;
    - `zWa…` against `zW1…`;
    - `%Wfc` against `%Wd`.
  - The docstring calls the renaming "a separate, cosmetic pass". So "`efficientnet_fwd_eval.mlir`
    is THIS net" holds at `den` only. MNv2 (`29bb72f7`) and MNv4 (`8211e3ca`) now have
    `FwdGraphTextTies` guards on their eval forwards; B0 is the last net without one.
  - **Fix:**
    - Rename the SSA names in the `EfficientNetRender.PCEval` builders, `mbResidDropGraphBEval` and
      `headGraphBEvalDo` to the render's.
    - Drop the bias slots, or use `biasName false`.
    - Add a B0 `.eval` `#guard` section to `LeanMlir/Proofs/Codegen/FwdGraphTextTies.lean` like
      MNv4's (:292).
    - Gate: build `FwdGraphTextTies`, then `regen_verified_mlir.sh check`.
  - **Evidence:** `grep -n -i "efficientnet\|enet" LeanMlir/Proofs/Codegen/FwdGraphTextTies.lean`
    hits training-mode sections only (:346, :381, :411).
  - **Cost:** 0 pins; about +60/−20 lines; low risk, since names do not enter `den`.
  - **Size:** M.

### reuse

- **N2-reuse-1** `EfficientNetSyncB.lean:55,62` (`den_swishF_shard`, `den_addV_shard`),
  `MobileNetV2SyncB.lean:62` (`den_relu6_shard`) and `MobileNetV4SyncB.lean:81`
  (`den_castIdx_shard`).
  - **Problem:** these are generic replica-shard lemmas that say nothing about any one net. Their
    siblings `den_relu_shard` and `den_addVB_shard` are already in
    `Foundation/DataParallel/SyncKit.lean:44,51`.
  - **Fix:** move all four into SyncKit with no renames and adjust the imports. Gate: build the
    three `*SyncB` files and their `*SyncStepTie*` consumers.
  - **Cost:** 0 pins; about ±40 lines.
  - **Size:** S.

- **N2-reuse-2** `MobileNetV4FullBEval.lean:424` — `mnv4SkipGraphBEval_faithful` duplicates
  `private theorem mnv4SkipGraphB_faithful` (`MobileNetV4FullB.lean:599`), which FullBEval imports.
  - **Problem:** statement and proof are the same (`simp only [mnv4SkipGraphB, den_addVB, hb,
    residual]`).
  - **Fix:** make the FullB lemma public, delete the Eval copy, and repoint the `rw` sites in
    `mnv4FwdGraphBFullEval_faithful`.
  - **Cost:** 0 pins; about −6 lines.
  - **Size:** S.

- **N2-reuse-3** `MobileNetV4FullBEval.lean:56` — `mnv4ProjBEval` is, binder for binder,
  `Proofs.projBEval` (`Codegen/EfficientNetRender/PCEval.lean:57`): `batchMap N
  (bnPerChannelEvalTensor3 …) ∘ batchMap N (flatConv W b)`.
  - **Fix:** move `projBEval` to `Foundation/Batched/Stages.lean` beside its training twin `projB`
    (:71), delete `mnv4ProjBEval`, and repoint its uses and the B0 Eval/Drop uses.
  - **Cost:** 0 pins; about 8 lines; check imports in the two B0 files.
  - **Size:** S.

- **N2-reuse-4** `ViTMhsaBackCertifiedTie.lean:120` — `dense_transpose_eq_mulVec : dense (Wᵀ) 0 =
  Mat.mulVec W` restates `dense_transpose_eq_vjp_backward`
  (`Architectures/ConvBackCertifiedTie.lean:79`, whose RHS `(denseHasVJP W b).backward x` is
  `Mat.mulVec W` by `rfl`, `Foundation/MLP.lean:56`). The proofs are the same `simp only …;
  sum_congr … mul_comm`.
  - **Two more copies inline:** `rowDenseBackFlat_eq_perRowFlat` (`ViTStepTieGB.lean:353-360`) and
    `vitCotLn2_eq_perRowFlatPR` (:380-392).
  - **Fix:**
    - Keep one lemma, `dense_transpose_eq_mulVec`, in `Foundation/BackwardMaps.lean`.
    - Derive `dense_transpose_eq_vjp_backward` from it in one line.
    - Reprove the two ViTStepTieGB lemmas with it.
  - **Cost:** 1 AuditAxioms pin; about −15 lines.
  - **Size:** S.

### scope

- **N2-scope-1** `LeanMlir/Proofs/Nets/ViT/` (line 0) and `MobileNetV4FullB.lean:56`.
  - **Problem:** the drop-path forwards of two nets have no forward statement. B0 has
    `efficientnetFwdGraphBFull{,Eval}Drop_faithful`; MNv2/MNv4 have only the `%do` forwards.
    - **ViT:** `vit_drop_fwd`, `vitin_drop_fwd` and `vitsin_drop_fwd` (24 `%dp*` mask inputs,
      written by `ViTRenderB.vitFwdRenderB (sd := true)`) and every `*drop*` train step are
      unstated. The book's quoted ViT artifact, `vitin_emadp128x4wxclipdropbf16`, is one of them.
    - **MNv4:** the paper-tier `mnv4in_acc{,dp}8x128wxdropdowd01bf16` (content.tex:8361) carry
      stochastic depth on the 18 skip rows. No MNv4 drop-path forward exists, and FullB's "Not
      this graph" artifact list (:56) omits both.
  - **Honest so far:** the ViT prose states the exclusion (ViTStepTieGB:54-56,
    ViTWholeBackCertifiedTieB:293), so this is coverage, not overclaim.
  - **Fix:** add `ViTFwdDrop.lean` and an MNv4 drop section on the `EfficientNetFullB0Drop`
    template: a `dropPathB` site per residual branch, `*FwdGraph*Drop_faithful` against a forward
    taking the mask vectors. Meanwhile, list the two MNv4 artifacts in the FullB:56 row.
  - **Cost:** new files (~300 lines each), plus a yaml/book clause.
  - **Size:** M per net; S for the list.

### attribution

- **N2-attr-1** `MobileNetV2.lean:3`, `MobileNetV2FullB.lean:3`, `MobileNetV4Spec.lean:1`,
  `MobileNetV4BackB0.lean` (module), `MobileNetV4FullB.lean:3`.
  - **Problem:** no MobileNet Lean file credits the papers.
    - MobileNetV2 is Sandler et al., CVPR 2018 (arXiv:1801.04381): inverted residual, linear
      bottleneck, and the `[t,c,n,s]` table FullB transcribes.
    - MobileNetV4 is Qin et al. 2024 (arXiv:2404.10518): UIB and its ExtraDW/ConvNeXt-like/FFN/IB
      families.
    - timm is credited in 11 MNv4 files; the book credits Sandler (content.tex:6696).
  - **Fix:** one reference line per spec-level module docstring.
  - **Evidence:** `grep -rn -E "Sandler|Qin|1801.04381|2404.10518" LeanMlir/Proofs` returns
    nothing.
  - **Size:** S.

- **N2-attr-2** `EfficientNet.lean:3`, `EfficientNetFullB0.lean:3`, `EfficientNetFullB0Drop.lean:5`,
  `EfficientNetFullWholeBackCertifiedTie.lean:6`.
  - **Problem:** these files say "paper B0" and transcribe Table 1 (FullB0:10-22), but name no
    paper:
    - EfficientNet: Tan & Le, ICML 2019 (arXiv:1905.11946);
    - squeeze-excitation: Hu et al. 2018, ratio 0.25 of the block input;
    - stochastic depth, which FullB0Drop implements: Huang et al. 2016 (arXiv:1603.09382). Only
      `LeanMlir/Types.lean:705` credits it.
    - The book credits Tan & Le (content.tex:8559).
  - **Fix:** a reference line in FullB0 (Tan & Le Table 1; MBConv after Sandler 2018; SE after Hu
    2018, r = ic/4) and in FullB0Drop (Huang 2016; dropout, Srivastava 2014).
  - **Size:** S.

- **N2-attr-3** `ViTDepthK.lean:1-25`, `ViTVecLN.lean:1-17`, `ViTMultiHead.lean:1-24`.
  - **Problem:** no ViT file credits a source.
    - `vitForwardKV` is Dosovitskiy et al. 2021 (arXiv:2010.11929): patchify, CLS, learned position
      embedding, pre-LN, CLS-slice head.
    - The instantiated config (D 192, 3 heads, 12 blocks, MLP 768, P 16) is DeiT-Ti, Touvron et al.
      2021 (arXiv:2012.12877).
    - Multi-head SDPA with the 1/√d scale is Vaswani et al. 2017.
    - LayerNorm is Ba et al. 2016.
    - The book credits both ViT papers (content.tex:9596, 12184).
  - **Fix:** a reference line in each file, and one line each on the two deliberate differences
    from timm DeiT: tanh-approximate GELU and LN ε = 1e-5 (timm uses erf and 1e-6). Both are
    confirmed in `verified_mlir/vitin_adam128_train_step.mlir`.
  - **Size:** S.

### generality

- **N2-gen-1** `ViTWholeBackCertifiedTie.lean:205-226` — `vitTinyInputGrad_eq_vitTiny_vjp` is still
  pinned at 10 classes (`Wcls : Mat (3 * 64) 10`, `bcls : Vec 10`, `… 3 64 10 12`), and its
  docstring says "Imagenette's 10 classes".
  - **Problem:** its batched peer and `vitTinyHasVJP_correct` bind `{nCls}` since `c5d74c10`, which
    did not touch this one.
  - **Fix:** add `{nCls : Nat}`, replace the four `10`s, and reword the docstring.
  - **Cost:** 1 AuditAxioms pin, name unchanged.
  - **Size:** S.

### placement

- **N2-place-1** `EfficientNetFullB0Drop.lean:40-58, 244-262` — eight generic optional-site
  wrappers live in a net file and never mention B0:
  - `dropPathOpt`, `dropoutOpt` and their `_ones` lemmas;
  - `StableHLO.dropPathOptG`, `dropoutOptG`, `den_dropPathOptG`, `den_dropoutOptG`.
  - **Why it matters:** N2-scope-1's ViT/MNv4 drop files will need them.
  - **Fix:** move the ℝ side to `Training/DropPath.lean` and the graph side beside `dropPathB` /
    `dropoutB`.
  - **Cost:** 0 pins; about ±50 lines.
  - **Size:** S.

- **N2-place-2** `ViTStepTieGB.lean:341-528` — nine per-example lemmas and defs sit in the batched
  step-tie file under `namespace Proofs.ViTTiePoCGB`. None of them mentions a batch:
  `rowDenseBackFlat_eq_perRowFlat`, `vitCotD{Q,K,V}mh_eq_core`, `vitCotLn2_eq_perRowFlatPR`,
  `vitCotXin_eq_blockBack`, `vitBlockCotInAtMHV_eq_vjp`, `vitHeadHasVJP`, `vitCotB2outV_eq_vjp`.
  - **Problem:** the per-example capstone `vit_net_tied_certified` cites two of them
    (`ViTStepTie.lean:334`) from a file downstream of it.
  - **Fix:** move the nine to the end of `ViTWholeBackCertifiedTie.lean` (namespace `Proofs`),
    beside `vitFinalLNBack_eq_vjp`, and keep only the `*B_eq_vjp` pair in ViTStepTieGB.
  - **Cost:** 9 AuditAxioms lines renamespaced (2133-2141); `ViTParamGrad.lean` opens
    `ViTTiePoCGB`.
  - **Size:** S.

### naming

- **N2-name-1** `ViTVecLN.lean:235` — `vitCotB2outV` means "cotangent at block 2's output", a
  leftover from the two-block prototype. Its docstring now says "the last block's output"; at depth
  12 that is `b12out`. The same applies to `vitCotB2outV_eq_vjp` and `vitCotB2outB_eq_vjp`
  (`ViTStepTieGB.lean:518, 546`).
  - **Fix:** rename to `vitCotTowerOutV` / `…_eq_vjp` / `vitCotTowerOutB_eq_vjp` across every
    consumer, with no alias.
  - **Cost:** about 25 occurrences in 6 files; AuditAxioms 2; content.tex 1.
  - **Size:** S.

### documentation

- **N2-doc-1** `EfficientNetStepTieG.lean:43-48` ("One replica") says the gradient nodes of
  `efficientnetin_emarmsdp64` are this file's per-replica node, composed by `DataParallel.Node`.
  - **Problem:** every B0 data-parallel artifact runs sync-BN. All 11 `efficientnet*dp*` files
    carry 361 `all_reduce`s, and their headers say "BatchNorm is SYNCHRONISED". So the per-replica
    composition applies to no B0 artifact; this is the per-replica-identity trap.
  - **Fix:** "Every B0 data-parallel render synchronises BatchNorm; its all-reduced gradients are
    `efficientnet_net_syncTiedG` (this file's node at N := R·N)". Drop the `DataParallel.Node`
    clause.
  - The same sentence at ResNet34StepTieB:32 is outside this slice.
  - **Size:** S.

- **N2-doc-2** `EfficientNetChainClose.lean:3-23` — the module docstring is stale.
  - **Problem:** it says the file proves "the `batchMap` VJP first, then `bnBatchLA`, then the
    per-block chains". The file now holds only five per-block `*FwdBHasVJP` defs and their
    `_differentiable` lemmas, none of them documented.
  - **Fix:** retitle it "EfficientNet's per-block batched VJPs", list the five, and give each a
    one-line docstring.
  - **Size:** S.

- **N2-doc-3** `EfficientNetFullB0.lean:239` — `efficientnetForwardBFull`, the B0 spec every B0
  capstone is stated against, has only a `--` banner.
  - **Problem:** neither it nor the module docstring states the padding. The stem is XLA-SAME and
    the four strided depthwises (b2/b4/b6/b12) are symmetric, a hybrid that is neither timm
    `efficientnet_b0` nor `tf_efficientnet_b0`. This is the known OPEN deviation
    (`imagenet_parity.md` §4 G3). No file claims TF parity; only BackChains:12 and StepTieG:40-41
    describe it.
  - **Fix:** add a docstring: "stem 3×3/s2 at the XLA-SAME phase; strided depthwises pad
    symmetrically, not TF SAME; SE width ic/4; batch BN; no drop (see `EfficientNetFullB0Drop`)".
    Add the padding sentence to the module header.
  - **Size:** S.

- **N2-doc-4** `MobileNetV4Spec.lean:33-37` and `MobileNetV4BackB0.lean:419-423` — both say the
  `#guard`s "pin" the timm reading of all 21 `(ic, oc, expand, preDWk, postDWk, h, stride2)`.
  - **Problem:** the guards (BackB0:425-450) pin only the family sequence and counts. A 3↔5 kernel
    swap or an expand 4↔6 passes them all. The table itself is correct: this audit re-walked timm
    1.0.28's `mobilenetv4_conv_medium`, and all 21 rows, the stem, the fused stage and the head
    match.
  - **Fix:** one `#guard` with the 21 timm tuples written out literally, `mnv4Blocks.map (fun s =>
    (s.ic, s.oc, s.expand, s.preDWk, s.postDWk, s.h, s.stride2)) = [...]`, or reword to "the guards
    pin the family pattern".
  - **Size:** S.

- **N2-doc-5** `MobileNetV2SyncStepTieB.lean` (module and `mnv2_net_syncTiedB`, ~:956-972) and
  `MobileNetV2SyncB.lean` (module).
  - **Problem:** they say "every all-reduced gradient the DP render emits …" and mention neither
    f32 nor bf16. Four of the eight MNv2 DP artifacts are bf16 (`mobilenetv2in_adamdp64bf16`,
    `rmsdp64bf16`, `rmsdp64wxdols0bf16`, `rmsdp64wxdols0eps0001bf16`), and bf16 weight gradients
    are not sharding-invariant.
  - **Fix:** copy MNv4's scope sentence (`MobileNetV4SyncStepTieB.lean:79`) and name the four f32
    DP artifacts.
  - **Size:** S.

- **N2-doc-6** `MobileNetV4SyncStepTieB.lean:1300-1302` (`mnv4_net_syncTiedB`).
  - **Problem:** the one artifact it names, `mnv4in_emaaccdp8x128wxdowd005bf16`, is bf16 and has
    `%do` dropout, and both are outside the statement. The "not stated" list names only EMA and
    accumulation.
  - **Fix:** name `mnv4in_adamdp64` instead, or add "bf16 nodes, `%do`" to the list.
  - **Size:** S.

- **N2-doc-7** `MobileNetV4FullBSeal.lean:13` — "discharges every clause with genuinely nonzero
  weights".
  - **Problem:** `sealP` zeroes every kernel of the 18 skip rows (:60-66: "the eighteen skipped
    rows pass zeros").
  - **Fix:** "…centre-tap kernels on the stem, the fused stage, the three channel-changing rows and
    the head; the eighteen skip rows zeroed (each block the exact identity)".
  - **Size:** S.

- **N2-doc-8** `MobileNetV4FullBEval.lean:3` ("at any resolution"), :11, :108 ("every input size").
  - **Problem:** the input side is `2*(2*(2*(2*(2*f))))` (:248), so only multiples of 32 are
    covered.
  - **Fix:** "at every input side 32·f".
  - **Size:** S.

- **N2-doc-9** `MobileNetV4FullBEval.lean` (`Mnv4BWeightsEval` docstring) and
  `MobileNetV2FullPaperEval.lean:26-40` (`IVWEval`).
  - **Problem:** the records carry a free conv-bias field per site (`be`, `bz`, `pre/post.b`,
    `h1b`, `hb`, …). The graphs spell each one as the `%zb{c}` zero operand while `den` reads the
    field (FullBEval:305-317). So the artifact is the `b = 0` instance.
  - **Why the existing justification does not apply:** FullB's justification ("∀-quantified; bias
    = 0 is one instance"; batch BN cancels a bias) does not transfer to eval BN, where a bias does
    not cancel. The Eval docstring counts "the 233 parameters … plus μ, σ²" and omits the 77 extra
    bias vectors.
  - **Fix:** state in both records' docstrings and headers that the artifacts are the zero-bias
    instance. The stronger fix is to drop the fields from the eval records and pass `0`.
  - **Cost:** docstring 0 pins; the record change touches `mobilenetv4ForwardBFullEval` (1
    AuditAxioms pin) and `FwdGraphTextTies`' `mnv4UibEvalW0`.
  - **Size:** S (doc) / M (record).

- **N2-doc-10** `ViTBackNet.lean:27-30` — "including that the fold's VJP is the shipped
  `vitForwardKVHasVJP`, not merely another VJP of the same map".
  - **Problem:** VJP witnesses of one map are unique (the `Subsingleton`, `Tensor.lean:282`), so
    the distinction the sentence draws does not exist. The content of `vitNetBackGraph_faithful` is
    that the hand-written `vitNetBackGraph` denotes the backward.
  - **Fix:** "…so `vitNetBackGraph` denotes `(vitForwardKVHasVJP …).backward` (VJP witnesses of
    one map are unique; `HasVJPAt.backward_unique_of_eq` along `vitNetLayer_fwd`)".
  - **Size:** S.

### proof-quality

- **N2-pq-1** `EfficientNetSyncStepTieG.lean` ~320-409 (`xCotIn_eq_vjp`, `rCotIn_eq_vjp`,
  `sCotIn_eq_vjp`, `nCotIn_eq_vjp`, `hdCotIn_eq_vjp`).
  - **Problem:** four proofs open with `have hc : <whole block backward> := rfl`, 10-15 lines each.
    They rely, without comment, on the tactic-built `mb*FwdBHasVJP` witness unfolding
    definitionally. `hdCotIn_eq_vjp` also has a bare `show cInB N Wh bh (den …) = _`.
  - **Fix:** one comment naming the defeq (the witness is `vjpComp` of the stage VJPs). Better,
    derive `hc` from `vjpComp_backward` (Tensor.lean:541) or factor one `mbBlockBackward_eq`.
    Avoid the `simp only [rfl lemmas]` kernel trap.
  - **Size:** S (comments) / M (factor).

- **N2-pq-2** Uncommented `show`s restating goals through definitional unfolding:
  - `MobileNetV4FullBSeal.lean:906-917` (`head_eq_dense`): a 10-line literal `show`, then `rfl`;
  - `MobileNetV4StepTieB.lean:769` (`mnv4SkipCotIn_eq_vjp`, new in `64db78c2`): relies on
    `CertLayer.residual`'s backward unfolding;
  - `MobileNetV4FullBSeal.lean:235` (`resid_id`).
  - **Fix:**
    - `head_eq_dense`: `simp only [sealW]` plus the `cbReluB_eq` rewrites, or a one-line comment.
    - The other two: `CertLayer.residual` fwd/backward apply lemmas in
      `Foundation/CertifiedChain.lean`, then `rw`.
  - **Size:** S.

- **N2-pq-3** `ViTMhsaBackCertifiedTie.lean:88-101` (`mhsaBackFlat_eq_mhsa_vjp`).
  - **Problem:** three `← Equiv.sum_comp (finProdFinEquiv …) (fun k => <restated summand>)`, then
    `Fintype.sum_prod_type` ×3. `sum_finProdFinEquiv` (`Foundation/Tensor.lean:582`) does this
    with no motive.
  - **Fix:** `rw [sum_finProdFinEquiv, sum_finProdFinEquiv, sum_finProdFinEquiv]`, which removes
    about 12 lines.
  - **Size:** S.
  - (carried: `proof_cleanup_audits/audit_effnet_vit.md`, the `ViTMhsaBackCertifiedTie` half; the
    `ViTBackB0` half landed.)

- **N2-pq-4** `ViTStepTieGB.lean:383` (`vitCotLn2_eq_perRowFlatPR`, new code).
  - **Problem:** a two-line uncommented `show Mat.flatten (fun i => Mat.mulVec W1 …) = _` that
    relies on how `vitCotLn2`/`vitCotM1` unfold (`ViTChainClose.lean:63-66`).
  - **Fix:** `unfold vitCotLn2 vitCotM1 rowDenseBackFlat`. With N2-reuse-4 the rest collapses to
    `dense_transpose_eq_mulVec`.
  - **Size:** S.

## Checked, not findings

- **MNv4 against timm 1.0.28 `mobilenetv4_conv_medium`**, re-walked here:
  - All 21 `mnv4Blocks` rows match: expand = `pw_exp.out/in`, `dw_start`/`dw_mid` kernels, stride
    on `dw_mid`.
  - The rest matches too: stem 3→32 s2 symmetric; fused `EdgeResidual` 32→128 3×3 s2 → 48;
    pre-DW BN-only; head 1×1 256→960 BN-relu → GAP → `conv_head` 960→1280 BN-relu → Linear;
    9,715,512 parameters at 1000 classes, equal to FullB's census.
  - The D1–D5 fixes hold in `MobileNetV4Spec` and `MobileNetV4FullB`.
- **Class count:** `B0Weights nCls` is threaded through every B0 statement, with no `Mat 1280 10` /
  `Vec 10` left in the slice. MNv2/MNv4 train ties, the sync ties and both FullB records are
  generic. ViT's `vit_net_tiedGB`, `vitTinyInputGradB_eq_vitTiny_vjp` and `vitTinyHasVJP_correct`
  bind `nCls`. ViTStepTie's `10`s are right, because `vit_train_step.mlir` is 10-class.
- **SE width:** r = ic/4 in the FullB0 table matches the render (b1 r8, b9 r20, b16 r48).
- **B0 padding:** no Lean file claims TF or timm parity. The hybrid is described at BackChains:12
  and StepTieG:40 (see N2-doc-3 for the defining file).
- **FullB0Drop matches the renderer:**
  - drop sites at blocks b3 b5 b7 b8 b10 b11 b13 b14 b15 (`enetDropIdxs`), on the branch before
    the skip add;
  - a per-example scale with 1/keep folded in;
  - width-1280 dropout between GAP and dense;
  - `_none`/`_ones` lemmas exact.
- **B0 sync capstone** (`efficientnet_net_syncTiedG`):
  - 213 = 3 + 10 + 15·13 + 5 conjuncts;
  - the cotangent hypothesis is discharged by `_smoothedCE`, so no content is moved into a
    hypothesis;
  - bf16 and drop are excluded in the text.
- **ViT architecture:**
  - D 192 = 3×64, 12 distinct blocks, MLP 768, P 16 → 196 + CLS, pre-LN, CLS-slice head,
    `sdpaScale = 1/√d_head`, tanh GELU and LN ε 1e-5 — all match the artifacts.
  - The weight-shared `vitFullHasVJP` is never called "full ViT" in the ViT directory.
- **Tie scope already stated:** ViT's bf16/drop-path/one-replica scope is stated at ViTStepTieGB
  :13-18/:53-56, ViTFoldGB :43-46 and ViTWholeBackCertifiedTieB:293, so the 09-24 "every shipped
  artifact" finding has landed.
- **Cotangent lemmas are real ties, not canonical-witness restatements:**
  - MNv4 `*CotIn_eq_vjp` (`64db78c2`);
  - MNv2 `mnv2StemCotC_eq_vjp` / `mnv2HeadCotBlk_eq_vjp` (`8eacd2da`);
  - ViT `vitBlockCotInAtMHV_eq_vjp` / `*B_eq_vjp`.
  - Each is a render-spelled chain equal to the backward.
- **MNv4 seal counts:** 38 relu clauses, 17 carrier BatchNorms, and rows 1/3/11 are the only
  channel changers.
- **Long proofs** (`mnv4_net_syncTiedB` 133 lines, B0 tiedG/tied/syncTiedG 61-84) are commented
  enumerations of 17-23 blocks; they don't factor further.
- **Earlier items:**
  - landed: `EnTail`, the `*_back_eq`/`_smul` moves, `relu6MaskB` → BackLinks, `sum_heads_3d`,
    the `ViTTieWeights` bundling;
  - declined: `*PoC` namespaces (api_design_audit.md:63), seal-helper publicity (§9).
- **Process narrative:** no markers or dates remain in the slice.

## Gaps for the humans

- No gate checks that each `verified_mlir/*_fwd_eval*.mlir` has a `FwdGraphTextTies` section. That
  is how B0's eval (N2-corr-2) slipped through.
- No gate checks that each shipped `*drop*` / `%do` forward has a `*Drop_faithful` / `*Do_faithful`
  statement. ViT and MNv4 drop-path, and ConvNeXt outside N2, slipped through (N2-scope-1).
- No lint flags a `*_correct` theorem whose proof is `(<witness>).correct …` while its docstring
  says "capstone" or "public correctness theorem" (N2-corr-1). `grep -rnE ":=\s*\(.*HasVJP.*\)\.correct"`
  over `*_correct` theorems would list them repo-wide.
- No gate reads an artifact header's "BatchNorm is SYNCHRONISED" line against a per-replica
  `DataParallel.Node` docstring (N2-doc-1).
- The MNv4 table's numeric agreement with timm is guarded only by the out-of-CI
  `scripts/parity/mnv4_timm_parity.py` (N2-doc-4).
