# Slice P — param-level loss-gradient cluster: rubric review 2026-09-30

Files: `Foundation/{ParamGrad,ParamGradNodes,SmoothedBatchLoss,BceBatchLoss}.lean`,
`Nets/{ResNet/ResNet34,ResNet/ResNet50,MobileNet/MobileNetV2,MobileNet/MobileNetV4,EfficientNet/EfficientNet,ConvNeXt/ConvNeXt,ViT/ViT}ParamGrad.lean`,
`Training/SgdDescent/MlpBias.lean` (all under `LeanMlir/Proofs/`; commits 4e8d9c90..df2a20d2, f7dd0419).
This is the first audit of any of this code.

## Verdicts
| angle | verdict | findings |
|---|---|---|
| correctness | block | 3 (P-C-1 blocks) |
| reuse | request_changes | 4 |
| scope | request_changes | 1 |
| attribution | request_changes | 1 |
| api-design | request_changes | 2 |
| generality | request_changes | 2 |
| placement | request_changes | 2 |
| naming | request_changes | 1 |
| documentation | request_changes | 2 |
| proof-quality | request_changes | 2 |

**Answer to the caller's key question.** The capstones state the right object: every tied node,
at the cotangent chain the step tie threads, equals `pdiv` of the whole-net loss in that
parameter. The loss is `smoothedBatchLoss N nCls α B t` (or `bceBatchLoss N nCls t`, the mean over
B×K), with α, B, the target, N and nCls all binders, so the shipped values are covered. Widths are
the shipped training resolutions: 224 literal for R34/MNv2/MNv4/B0/ConvNeXt/ViT, and R50 is generic
in `q`, so its 160 px A3 run is covered. The loss content is not hidden in hypotheses: `hL` is
discharged by `*_smoothedCE` / `_bce`. There are three faults:
- **P-C-1.** The ResNet-34/50 capstones need a stem-pool no-tie hypothesis that fails on every
  real batch.
- **P-C-2.** "Is ∂L/∂θ" is stated as a `pdiv` equation, and `pdiv` takes a junk value where the
  function is not differentiable.
- **P-C-3.** The ViT and ConvNeXt "whole net" is a local definition. No batched theorem ties it to
  the rendered forward.

The factoring is uneven. MNv4, B0, ConvNeXt and ViT use the shared kit. R34, R50 and MNv2 were
written before the kit's stage pull-backs landed and still re-derive them (57 defs, 53 theorems).

## Findings

### correctness

- **P-C-1** `LeanMlir/Proofs/Nets/ResNet/ResNet34ParamGrad.lean:757` (`r34_net_lossGrad`, uses
  `hx.pool` at :876) and `ResNet50ParamGrad.lean:1033` (`r50_net_lossGrad`, `hx.pool` at :1096).
  - **The problem.** Both capstones take `R34SmoothAtB` / `R50SmoothAtB`. Their `pool` field
    (`ResNet34FullBVJP.lean:289`) is `StemPoolSmoothAt … (cbReluStridedB …)`
    (`Foundation/HeadLayers.lean:61`): `MaxPool3s2Smooth` of the **post-ReLU** stem activation.
    `MaxPool3s2Smooth` (`Architectures/MaxPool3s2.lean:161`) requires each window's maximum to be
    strictly above every other cell. A 3×3 window whose nine pre-ReLU values are all ≤ 0 has
    maximum 0 attained nine times.
  - **How often it fails.** At init with independent signs that is 2⁻⁹ per window. A 128-image
    batch has about 128·64·56·56 ≈ 25.7M windows, so about 50k fail. Trained stems have dead
    channels and flat image regions, which make it worse. The hypothesis fails on essentially every
    real ImageNet batch, so neither capstone, nor `r50_net_lossGrad_smoothedCE` / `_bce`, applies
    to any real training step.
  - **The statement is still true there.** In such a window every input is strictly negative
    before the ReLU, so the loss is locally constant in it. The emitted `maxPool3s2BackB` routes to
    one zero cell, and `reluMaskB` then kills it, so both sides are 0. The hypothesis is simply too
    strong.
  - **Secondary instance.** The JAX reference zero-inits R50's bn3 γ (`jax/Jax/Codegen.lean:1605`,
    `:1610`, `:1614`). At step 0 the outer-ReLU input of every identity bottleneck is then
    `v + 0` with `v ≥ 0` post-ReLU, so `hout` fails at initialisation too.

  **Fix:**
  1. Add `maxPool3s2Flat_relu_comm`: `maxPool3s2Flat (relu x) = relu (maxPool3s2Flat x)`. ReLU is
     monotone and the windows clamp to real cells.
  2. Define a stem predicate on the **pre-ReLU** BN output: no entry is 0, and each window's
     maximum is attained at one cell.
  3. Prove the composite `batchMap pool ∘ relu` has the emitted `reluMaskB ∘ maxPool3s2BackB` as
     its VJP there. Windows with maximum ≤ 0 give 0 on both sides.
  4. Replace the `stem`/`pool` fields of `R34SmoothAtB` / `R50SmoothAtB` with it, and re-prove
     `r34StemGC_hasGradAt` and the whole-net VJPs that consume the fields.
  5. Gate with `lake build CertsHeavy` plus `tests/AuditAxioms.lean`.

  **Evidence:** `ResNet34ParamGrad.lean:413-433` pulls back through
  `stemPoolLayer … hpool`, where `hpool : StemPoolSmoothAt N h w (cbReluStridedB …)`. The
  post-ReLU problem was first raised for the whole-net VJP in
  `planning/doc_audit/find_A_published.md:19`. `planning/doc_honesty_pass.md` §1(f) only reworded
  the prose. The new capstones inherit the fault.

  **Cost:** `R34SmoothAtB` / `R50SmoothAtB` are shared with `resnet34/50ForwardBFullHasVJPAt` and
  the full-width seals. Pins: `r34_net_lossGrad`, `r50_net_lossGrad*` (AuditAxioms:1119,
  1125-1127). About +250 lines of new pool lemmas. Risk: moderate, because the stem proofs change
  shape. **Size:** L.
  (partly carried: `doc_audit/find_A_published.md` §19, for the VJP)

- **P-C-2** `LeanMlir/Proofs/Foundation/ParamGrad.lean:97` (`HasGradAt.pdiv_param`), `:107`
  (`pdiv_param_batchMap`), and every `*_eq_pdiv` in `ParamGradNodes.lean`, the
  `*LossTiedB` / `*LossTiedGB` bundles, and all seven `*_net_lossGrad`.
  - **The problem.** The conclusions are equations `den node = pdiv (fun θ => Φ (upd θ)) θ₀ i 0`.
    `pdiv` is `fderiv ℝ f x (basisVec i) j` (`Foundation/Tensor.lean:97-99`), which is 0 wherever
    `f` is not differentiable.
  - **Why that is weaker than advertised.** The statement does not say the loss is differentiable
    in θ at θ₀. It says "node = the junk-or-real partial". A reader cannot get "the node **is** the
    gradient" (book `thm:resnet34_loss_grad` …) from the statement alone. Differentiability is
    proved inside every proof (`hG.1.comp θ hl`) and then thrown away.

  **Fix:** make `HasGradAt.pdiv_param` / `pdiv_param_batchMap` /
  `pdiv_param_batchMap_through` conclude `HasGradAt (fun θ' => G (layer θ')) θ (fun i => …)`.
  `HasGradAt` already packages `DifferentiableAt`. Restate each node lemma and bundle clause as
  `HasGradAt (fun θ => Φ (upd θ)) θ₀ (fun i => den node (idx i))`. The dense W index is
  `finProdFinEquiv (i, j)`, so it needs one reindex. The cheaper alternative is to add a
  `DifferentiableAt ℝ (fun θ => Φ (upd θ)) θ₀` conjunct per slot. Gate with `lake build Certs` and
  AuditAxioms.

  **Evidence:** `ParamGrad.lean:100` states only the `pdiv` equation, while its proof uses `hG.1`
  and `hl` (:101-102). **Cost:**
  - The theorem names can stay.
  - The bundle defs change shape: 7 nets, about 40 `*LossTiedB` defs and their proofs.
  - Book text: "is ∂L/∂θ" becomes literally true.

  **Size:** M–L.

- **P-C-3** `LeanMlir/Proofs/Nets/ViT/ViTParamGrad.lean:1345` (`vitNetB`) and
  `ConvNeXt/ConvNeXtParamGrad.lean:659` (`cnxNetB`).
  - **The problem.** Both capstones are stated against a forward defined in the same file as a
    `batchMap` composition of the tie's `BlockParamsV.fwdO` / `CnxTieBlk.fwdO`.
  - **ViT.** No theorem equates `vitNetB` with `vitForwardKV`
    (`Nets/ViT/ViTDepthK.lean:151`), the forward whose graph `vitFwdGraphKMHV_faithful` (:350)
    ties to the render. `grep -rln vitBlockFwdOMHV` finds only the two StepTie files.
  - **ConvNeXt.** There is a per-example bridge, `CnxTieWeights.forward_eq_convNextForwardTCh`
    (`ConvNeXtStepTie.lean:594`), but no batched corollary for `cnxNetB`.
  - **Contrast.** The other five nets state their capstones against the canonical sealed
    `*ForwardBFull`. For ViT, "the WHOLE net" in the docstring and in book `thm:vit_loss_grad` is
    faithful only by definition.

  **Fix:**
  - ViT: add `vitNetB_eq_vitForwardKV : vitNetB N ε w img = batchMap N (vitForwardKV 3 224 224 16 196 768 3 64 nC 12 w.Wc w.bc w.cls w.pos ε ![w.b1, …, w.b12] w.γF w.βF w.Wcls w.bcls) img`.
    Prove it per block with `vitBlockSpelledMHV_eq`, as `ViTDepthK.lean:306-325` already does for
    the graph.
  - ConvNeXt: add `cnxNetB_eq : cnxNetB N ε w x = batchMap N (convNextForwardTCh (w.toCh ε)) x`
    from `cnx_logitsB_eq`, `batchMap_comp` and `forward_eq_convNextForwardTCh`.
  - Cite both in the capstone docstrings. Gate with `lake build CertsHeavy`.

  **Cost:** 2 new theorems, no renames, plus 2 AuditAxioms lines. **Size:** S (ConvNeXt) / M (ViT).

### reuse

- **P-R-1** `ResNet34ParamGrad.lean:43-445`, `ResNet50ParamGrad.lean:49-710`,
  `MobileNetV2ParamGrad.lean:52-705` (per-block sections).
  - **The duplication.** These three files re-derive, stage by stage, what
    `Foundation/ParamGradNodes.lean:65-133` provides as one-line pull-backs: `hasGradAt_bnBatchLA`,
    `hasGradAt_relu`, `hasGradAt_conv`, `hasGradAt_convStrided`, `hasGradAt_depthwise`,
    `hasGradAt_depthwiseStrided`.
  - **Scale.** They carry 57 `*G*` loss-at-activation defs (R34 15, R50 23, MNv2 19) and 53
    `*G*_hasGradAt` theorems (13 / 25 / 15). Each theorem is one `HasGradAt.comp … .of_eq
    (bnInB_eq_bnBackB …)`, re-spelled 6 / 11 / 10 times. MNv4, B0, ConvNeXt and ViT have zero such
    defs.
  - **Example.** `r34StemGC_hasGradAt` plus `r34_stem_lossTiedB` (`ResNet34ParamGrad.lean:413-481`,
    about 60 lines) against `mnv4_stem_lossTiedB` (`MobileNetV4ParamGrad.lean:147-161`: `have hN :=
    hasGradAt_relu _ hs hGn; have hC := hasGradAt_bnBatchLA … hN; exact ⟨…⟩`).

  **Fix:**
  1. Add `hasGradAt_relu6` (relu6 → `relu6MaskB`, via `relu6HasVJPAt`) to ParamGradNodes.
  2. Rewrite each R34/R50/MNv2 `*_lossTiedB` proof in the MNv4 style.
  3. Delete the `r34IdG*` / `r34DownG*` / `r34StemG*` / `r50*G*` / `mnv2*G*` defs and theorems.

  **Evidence:** `grep -cE '^noncomputable def [a-z0-9]+(Id|Down|Proj|Stem|Head|NoExp|Body|SBody)G[A-Z]'`
  per file gives 15/23/19/0/0/0/0. `grep -c "bnInB_eq_bnBackB N"` gives R34 6, R50 11, MNv2 10.

  **Cost:** none of the deleted names is pinned (AuditAxioms:1113-1137 pins only `*_lossTiedB`,
  one `*_factor_*`, and `*_net_lossGrad*`). About −700 to −1000 lines. Low risk. **Size:** M.

- **P-R-2** `MobileNetV4ParamGrad.lean:720` (`certLayer_hasGradAt_comp`) and `:735`
  (`mnv4_residual_body_hasGradAt`).
  - **The problem.** Both are general lemmas, about any `CertLayer` and any residual body, parked
    in a net file.
  - **The re-inlining.** Other nets re-inline the same argument against their own CertLayers:
    - `r34IdB_hasGradAt_comp` / `r34DownB_hasGradAt_comp` (`ResNet34ParamGrad.lean:725-747`,
      `StableHLO.r34BasicBlockLayer … .diff`).
    - The stem-pool step (:428-433, `stemPoolLayer .diff/.vjp/.faithful` by hand).
    - `r50IdB/ProjB/DownB_hasGradAt_comp` (`ResNet50ParamGrad.lean:718-752`).
    - `mnv2*_hasGradAt_comp` (`MobileNetV2ParamGrad.lean:711-754`).
  - **The residual-body step** ("skip is a constant, gradient at the body output is still `dyOut`")
    is re-proved as `r34IdGN2_hasGradAt` (:88), `r50IdGN3_hasGradAt` (:102), the body of
    `mnv2_resid_lossTiedB` (:408), and B0's `enet_resid_lossTiedG`.

  **Fix:**
  1. Move `certLayer_hasGradAt_comp` to ParamGradNodes as `CertLayer.hasGradAt_comp`. Add a variant
     whose cotangent is `den (L.graph x e)`, via `L.faithful`.
  2. Move `mnv4_residual_body_hasGradAt` there as `HasGradAt.residual_body`.
  3. Use both at the sites above, adding `import …Foundation.CertifiedChain` if it is not already
     transitive.

  **Cost:** neither name is pinned. About −120 lines. **Size:** S–M.

- **P-R-3** `ViT/ViTParamGrad.lean:746,752` (`rowVecLN_gamma_differentiable`,
  `rowVecLN_beta_differentiable`) duplicate `ConvNeXt/ConvNeXtParamGrad.lean:52,56`
  (`rowLNVecFlat_gamma_differentiable`, `_beta_differentiable`). ViT's lambda
  `fun θ => Mat.flatten (fun r => layerNormVec D ε θ β (Mat.unflatten x r))` is
  `fun θ => rowLNVecFlat tk D ε θ β x` by definition (`Architectures/ChannelLN.lean:73`).

  **Fix:** move the ConvNeXt pair to `Architectures/ChannelLN.lean` next to `rowLNVecFlat` and
  delete ViT's two. At the four ViT call sites, pass `rowLNVecFlat_gamma_differentiable …` (defeq),
  or state the ViT LN stage as `rowLNVecFlat`. **Cost:** unpinned, −12 lines. **Size:** S.

- **P-R-4** `Foundation/SmoothedBatchLoss.lean:117-123` and `Foundation/BceBatchLoss.lean:66-72`
  carry the same `hidx` / `hrow` blocks verbatim. `smoothedBatchLoss_pdiv` (:92-105) and
  `smoothedBatchLossDiv_grad` (:141-157) repeat the same `hℓ` + `rowSumLoss_pdiv` +
  `pdiv_const_smul` block. `smoothedBatchLoss` and `smoothedBatchLossDiv` differ only in how the
  target row is read.

  **Fix:**
  - Extract `unrowB_rowIdx` (the `Fin.cast` index identity) and `targetRow_rowB_eq_logitRow` (the
    `hrow` identity) as public lemmas in SmoothedBatchLoss, and use them in both `*_grad`.
  - Define `smoothedBatchLoss N K α B t := smoothedBatchLossDiv N K α B (fun J => targetRow' …)`
    at the reindexed target, and derive `smoothedBatchLoss_pdiv` from one shared per-row lemma.

  **Cost:** pins `smoothedBatchLoss_pdiv` / `_grad`, `bceBatchLoss_pdiv` / `_grad` keep their
  names. About −30 lines. **Size:** S.

### scope

- **P-S-1** Two declarations are dead:
  - `Foundation/ParamGrad.lean:38` `addConstHasVJPAt_backward`.
  - `Foundation/ParamGradNodes.lean:361` `bnBatchLA_apply_perm`.

  **Fix:** delete `addConstHasVJPAt_backward`. Either delete `bnBatchLA_apply_perm` or make it
  used (P-PQ-2). **Evidence:** `grep -rw` over the tree finds only each definition, with no pins,
  book or yaml hits. **Cost:** −8 lines. **Size:** S.

### attribution

- **P-A-1** `Foundation/SmoothedBatchLoss.lean:1-12` defines "the batched label-smoothed loss", and
  the technique is uncredited here, in `SmoothedLossCot.lean`, and in the book (`grep -i szegedy`
  finds only BatchNorm and GoogLeNet). **Fix:** one sentence in the module docstring: label
  smoothing as in Szegedy et al. 2016, "Rethinking the Inception Architecture for Computer Vision",
  §7. BCE is already credited to timm/RSB in `BceBatchLoss` and `BceLossCot`. **Size:** S.

### api-design

- **P-API-1** `Foundation/ParamGrad.lean:67` `HasGradAt`.
  - **The problem.** It is a bare `def … : Prop := DifferentiableAt ℝ G x ∧ ∀ j, pdiv G x j 0 =
    dy j`, used through `.1`/`.2` and anonymous constructors, which works only by unfolding the def.
  - **Sites:** ParamGrad.lean:75, 77, 102, 103, 135; the `hL` constructors in all eight
    `*_smoothedCE` / `_bce` / `r34_net_lossGrad`.
  - **Missing API:** there are no named projections and no `iff` lemma.

  **Fix:** make it a `structure HasGradAt … : Prop` with fields `differentiableAt` and `pdiv_eq`,
  or keep the def and add `HasGradAt.differentiableAt`, `HasGradAt.pdiv_eq`, `hasGradAt_iff`, then
  switch the consumers. **Cost:** book `def:hasgradat` wording is unchanged. About 12 consumer
  edits. **Size:** S.

- **P-API-2** The cotangent chain is spelled twice per net: once in the step tie's statement and
  once in the loss-gradient statement.
  - **R34 example:** `ResNet34StepTieB.lean:426-446` against `ResNet34ParamGrad.lean:760-781`.
  - **The others:** R50 `R50NetLossTiedB` (:966-986) against `r50_net_tiedB`, and likewise for
    MNv2, MNv4, B0, ConvNeXt and ViT.
  - **Why it matters.** Docstrings and the book say the nodes are loss derivatives "at the same
    cotangent" the tie threads, but that sameness is a textual coincidence. Editing one chain
    leaves both theorems green, and the claim silently stops holding.

  **Fix:** per net, one def of the chain (for example `r34NetCots N w x g`, a record of the 17
  cotangents) that both `r34_net_tiedB` and `r34_net_lossGrad` take their lets from. Or add a
  corollary per net combining the two into "tie RHS = loss derivative" per slot. **Cost:** the
  statements of 14 pinned capstones are restated with the same names. About ±0 lines. **Size:** M.

### generality

- **P-G-1** `ResNet34ParamGrad.lean:757` `r34_net_lossGrad`.
  - **The problem.** It is the only one of the seven hard-wired to `smoothedBatchLoss`. The other
    six take any `L` with `hL : HasGradAt L (net w x) g`, and a `*_smoothedCE` corollary
    discharges it.
  - **Naming consequence.** `r34_net_lossGrad` means "smoothed" while `r50_net_lossGrad` means
    "any L".
  - **Stale doc.** `planning/api_docs_followups.md:109` already claims all seven are any-L.

  **Fix:** split into `R34NetLossTiedB` (def), `r34_net_lossGrad (hL)` and
  `r34_net_lossGrad_smoothedCE`, as R50 does (`ResNet50ParamGrad.lean:966-1136`). **Cost:** pins
  AuditAxioms:1119 (add the `_smoothedCE` line) and content.tex:5312/5334 (`\lean{}` gains the
  corollary). About +15 lines. **Size:** S.

- **P-G-2** `ViT/ViTParamGrad.lean:63-153` generalises the attention kit's column-slab machinery
  and leaves the special case with its own proof.
  - **The new material:** `colSlabApplyH`, `pdivMat_colIndepH`, `colSlabwiseHasVJPMatH`,
    `colSlabApplyH_flat_differentiable`.
  - **What it generalises:** `Architectures/Attention.lean:73,88,144` (`colSlabApply`,
    `pdivMat_colIndep`, `colSlabwiseHasVJPMat`). `colSlabApply g` unfolds to the same body as
    `colSlabApplyH (fun _ => g)`, yet the special case keeps its own ~50-line proof.

  **Fix:** move the four H declarations into Attention.lean. Prove `pdivMat_colIndep` as
  `pdivMat_colIndepH (fun _ => g) (fun _ => h_g_diff)` and `colSlabwiseHasVJPMat` as the H version
  at a constant family.
  - **Trap:** keep `colSlabwiseHasVJPMat.backward` definitionally identical. Its 17 consumers
    include `ViTBackB0.lean` graph ties that may `rfl`-match it (the IR-spelled trap). Gate with
    `lake build CertsHeavy`.
  - **Also move** `attnCore*` / `attnCore{Q,K,V}HasVJPMat` (:191-300), which are generic in
    `Np1 heads d`.

  **Cost:** Attention.lean downstream rebuild. Pins `pdivMat_colIndepH`,
  `colSlabwiseHasVJPMatH`, `attnCore*HasVJPMat` and `attnCoreQ_backward` (AuditAxioms) change
  namespace from `Proofs.ViTTiePoCGB`. About −45 lines. **Size:** M.

### placement

- **P-PL-1** General helpers sit in per-net files:
  - `hasGradAt_cast` (`MobileNetV4ParamGrad.lean:112`).
  - `hasGradAt_linLoss_constAdd` (`ViTParamGrad.lean:316`).
  - `lb_batchMap_congr` (`ViTParamGrad.lean:803`).
  - `rowDense_weight_differentiable` / `rowDense_bias_differentiable` (`ViTParamGrad.lean:734,740`).
  - `chanLNTensor3_gamma/beta_differentiable` and `rowLNVecFlat_*_differentiable`
    (`ConvNeXtParamGrad.lean:52-70`).
  - `seGateMulB` / `seGateMulBHasVJP` (`EfficientNetParamGrad.lean:54,61`), generic in `N c h w`.

  **Fix:**
  - `HasGradAt` helpers go to `Foundation/ParamGrad.lean`.
  - The row-dense/LN differentiability lemmas go to `Architectures/ChannelLN.lean` or LayerNorm
    (see P-R-3).
  - The SE gate product goes to `Architectures/SE.lean`.
  - Attention goes as in P-G-2, and CertLayer as in P-R-2.

  **Cost:** only `seGateMulBHasVJP` is pinned (AuditAxioms, namespace `Proofs.EnetTiePoCG`).
  About ±0 lines. **Size:** M.

- **P-PL-2** `Foundation/ParamGrad.lean:142` `batchSlice_batchMapAux` is a pure `batchSlice` /
  `batchMapAux` identity. It belongs next to `batchSlice_batchMap` in
  `Foundation/Batched/Basic.lean:36`, which ParamGrad already imports. **Fix:** move it, keeping
  the name. **Cost:** unpinned. **Size:** S.

### naming

- **P-N-1** Three names break the conventions:
  - `certLayer_hasGradAt_comp` (`MobileNetV4ParamGrad.lean:720`) takes a `CertLayer` as its first
    explicit argument. It should be `CertLayer.hasGradAt_comp` for dot notation.
  - `mnv4_residual_body_hasGradAt` (:735) carries a net prefix on a net-independent statement.
  - `lb_batchMap_congr` (`ViTParamGrad.lean:803`) is named after the local variable `Lb`. It should
    be `batchMap_congr_apply`, or go away in favour of
    `congrArg (fun f => Lb (batchMap N f X)) (funext h)`.

  **Fix:** rename along with the moves in P-R-2 and P-PL-1. **Cost:** unpinned. **Size:** S.

### documentation

- **P-D-1** The ResNet capstone docstrings and book blocks omit the post-ReLU failure caveat:
  - `ResNet34ParamGrad.lean:29-30, 754-755` and `ResNet50ParamGrad.lean:32-33, 1029-1030` say
    "the stem pool tie-free at the real activations".
  - `blueprint/src/content.tex:5322, 5373` say "every stem-pool window's maximum attained at one
    cell".
  - Neither says the condition is on the post-ReLU activation and fails in any window of two or
    more dead ReLUs. That wording is what `planning/doc_honesty_pass.md` §1(f) required for the
    whole-net VJP sites.

  **Fix:** until P-C-1 lands, add that clause at all four code sites and both book blocks. Gate
  with `scripts/book/book_xrefs.py`. **Size:** S.

- **P-D-2** The R34/R50/MNv2/MNv4 module and capstone docstrings present `L` as the loss "the
  trainer minimises" (R34 :10-11) or "the artifacts ship" (R50 :11, MNv2 :11, MNv4 :10), with no
  scope.
  - **The gap.** The book's quoted R34/R50 ImageNet runs are bf16, sync-BN, 4×128 (§5.7/§5.8),
    whose nodes are the `*GradBBf16` kinds, across 4 replicas.
  - **The contrast.** B0, ConvNeXt and ViT do state the scope ("Drop-path and the bf16 nodes are
    outside this statement", `EfficientNetParamGrad.lean:37-38`, `ConvNeXtParamGrad.lean:36-37`,
    `ViTParamGrad.lean:44-45`).

  **Fix:** add the same sentence to the four files. Say which nodes are covered (f32 nodes, one
  replica), and that sync-BN is reached by composition at `N := R·N`. **Size:** S.

### proof-quality

- **P-PQ-1** `ViT/ViTParamGrad.lean:880-1110` `vit_block_lossTiedGB` (233 lines) and its statement
  `vitBlockLossTiedGB` (:817-878).
  - **The repetition.** The proof is 16 near-identical bullets, each
    `(…TiedB_holds).trans (HasGradAt.pdiv_param_batchMap_through pre per post cot …).trans
    (congrArg … (lb_batchMap_congr … ).trans (hΦ _).symm)`.
  - **The long spellings.** Every bullet spells `blkSaves ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 x` (16 args) and `cLn1/cQ/… ε p.γ1 … p.Wfc2` (17 args) out in
    full.
  - **Same shape** in `cnx_block_lossTiedGB` (`ConvNeXtParamGrad.lean:248`, 108 lines).

  **Fix:**
  1. Add `BlockParamsV.saves ε x` and `BlockParamsV.cot{Ln1,Q,K,V,H,Ln2,M1} ε` abbrevs next to
     `BlockParamsV.fwdO` / `cotIn` (`ViTStepTie.lean:299`).
  2. Add one local lemma `vit_node_lossTied` taking `(pre per post cot)`, the fwd-factor lemma, the
     parameter-differentiability fact and the node tie, and closing one bullet.
  3. Each bullet becomes about 3 lines, and the proof goes from about 233 to about 80 lines.

  **Cost:** unpinned apart from the theorem itself, whose statement is unchanged. **Size:** M.

- **P-PQ-2** Two kinds of undocumented definitional-equality reliance:
  - **`ParamGradNodes.lean:403, 416`** (`bnGamma_eq_pdiv`, `bnBeta_eq_pdiv`). These close with
    `exact (bnLA_param_pdiv (fun θ => bnPerChannelFlat …) … hG c).symm`. That matches
    `bnBatchLA N oc h w ε θ β v` against `fun J => F θ (bnLAPerm N oc h w J)` by unfolding
    `bnBatchLA`. The equation is stated as `bnBatchLA_apply_perm` (:361), which is never used.
  - **Uncommented `change`/`show` on defeq reshapes:** `ParamGrad.lean:76, 101`
    (`change pdiv (G ∘ f) …`), `ParamGradNodes.lean:468`, `MobileNetV4ParamGrad.lean:118`,
    `MlpBias.lean:60, 171, 291`.

  **Fix:** state `bnBatchLA_apply_perm` in `funext` form, `bnBatchLA … = fun J => … (bnLAPerm J)`,
  and `rw` with it in both proofs. Add a one-line comment at each `change`/`show`. For
  ParamGrad:76/101, "`pdiv_comp` is stated on `G ∘ f`" suffices. **Size:** S.

## Checked, not findings
- **Target and loss hypotheses.**
  - `smoothedBatchLoss` / `smoothedBatchLossDiv`: α, B, target, N and nCls are all binders, and
    the emitted cotangent is proved to be its full gradient (`smoothedBatchLoss_grad`, `…Div_grad`).
  - `ht` (targets sum to 1) holds for one-hot, smoothed, and mixup/cutmix targets.
  - `bceBatchLoss` is the mean over N·K with no target hypothesis, matching timm's
    `BinaryCrossEntropy`.
- **Resolution and classes.** Widths and resolution match the shipped training runs: 224 literal
  for R34/MNv2/MNv4/B0/ConvNeXt/ViT, and R50 is generic in `q` (the A3 run is at 160). `nCls` is a
  binder in all seven. ViT is at ViT-Tiny dims, the same as its tie. The ViT-S/B side-quest renders
  are outside both the tie and the capstone.
- **Smoothness hypotheses beyond the pool.**
  - The relu/relu6 clauses in R34, R50 (except step 0, see P-C-1), MNv2 and MNv4 are on
    pre-activations (BN outputs, residual sums), so they hold generically on real data.
  - B0, ConvNeXt and ViT correctly need none (swish, sigmoid, GELU and LN with `0 < ε`).
- **The two ConvNeXt/ViT special nodes.** The ConvNeXt stem bias over a free `xstem`
  (`pdiv_bias_of_split`) and the ViT/ConvNeXt classifier-bias batch-sum statements are sound as
  stated.
- **`pdiv_param_batchMap_through`.** Its per-example `hcot` hypotheses are discharged in-file
  (`cnxBlk_hasGradAt`, `vitPost*_hasGradAt`), not left to the capstone caller.
- **Mathlib reuse.** Mathlib's `HasGradientAt` needs an inner-product space, and `Vec m` has the
  sup norm, so the local `HasGradAt` is not a Mathlib duplicate (its API shape is P-API-1).
- **MlpBias.** It reuses `sgd_descends` and `MlpSlot.loss_grad_lipschitz`. Its h1/h2/margin
  hypotheses have the same form as the weight rungs and are satisfiable for small lr and η. It is
  on the roadmap through `SgdDescent/Cnn.lean` (imports it; the conv-bias descents) and
  AuditAxioms:1254-1260.
- **Wiring.** All seven capstones are pinned in `tests/AuditAxioms.lean:1113-1199` and have book
  theorem blocks (content.tex:5312, 5363, 7169, 7225, 8709, 10028, 12153). The `\uses` edges are
  CI-checked.
- **Namespaces** follow each net's tie namespace (`ResNet34TieB`, `CnxTiePoCGB`, …), which is
  consistent.
- **Known traps.** The `r34_factor_*` proofs are `rw [16 × Pre*_apply]; rfl`, standalone by design
  because inline they hit kernel deep recursion at literal widths (as documented). This is not a
  proof-quality finding.

## Gaps for the humans
- **No formalization.yaml row or comparator challenge** exists for any `*_net_lossGrad`
  (`grep -c lossGrad formalization.yaml scripts/gates/gen_comparator_tier.py` gives 0/0). The
  strongest per-net claim in the book is not guarded against statement drift. Only two step ties
  (`r50_net_tiedB`, `cnx_net_tiedGB`) have rows.
- **No check that hypotheses can be met on real data.** A JAX probe evaluating
  `R34SmoothAtB`-style predicates on one real batch from a checkpoint would have caught P-C-1
  mechanically.
- **No check that the tie and loss-gradient cotangent chains agree** (P-API-2).
- **Process slip.** During this audit I ran one read-only `git -C <main checkout> show HEAD:…` to
  diff a file, against the protocol's "no git in main". Nothing in that checkout changed. One
  `lake env lean` typecheck of a scratch copy was also run there. It showed that ParamGradNodes'
  `SmoothedBatchLoss` import supplies `denseWeightMap_differentiable` transitively, so no
  unused-import finding is filed.
