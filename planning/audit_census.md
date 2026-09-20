# Audit-only census and stale-citation sweep — 2026-09-19 @ `de4db64b`

Passes 1 and 2 of the next-session block in `planning/mathlib_reuse_audit.md`. The survey sections
below are as written on `de4db64b`; the Status section records what has landed since.

## Status (2026-09-19, end of session 2): census cuts 2a–2e and the docstring gate landed, pushed

Everything below is on `origin/main` at `281b6beb`. Each commit passed the full gate
(`scripts/audit_census/gate.sh`: Certs, default build, the LeanMlir lib, AuditAxioms verdicts,
docstring-checkrefs, both coverage scripts, `verified_mlir/` byte-identical):

| commit | what | files | lines | pins |
|---|---|---:|---:|---:|
| `ca9ea890` | finding 1 fixed: `MlpPoC.mlp_train_step_tied_certified` states all six parameter ops (the b₁/b₀ conjuncts, from `MlpPoC.b{1,0}_den_certified`) | 1 | +28 −2 | 0 |
| `3722300c` | 2a, ViT/ConvNeXt: VC.A, VC.B, VC.C, VC.E, VC.G, VC.F1. `ConvNeXtClose.lean` emptied and deleted | 25 | +116 −1,397 | 45 |
| `7f9abb5e` | 2b, ResNet: R.B1, R.C, R.D, R.E, R.F, plus the dead weight wiring, the `ResNet34Concrete` defs, `flatConvStride2XlaF(_close)`. `ResNet50BackNet.lean` deleted | 21 | +89 −1,152 | 24 |
| `333add66` | 2c, MobileNet/EfficientNet: M.1, M.3–M.7, B.6, M.10. `MobileNetV4FoldB.lean` and `EfficientNetBackCertifiedTie.lean` deleted | 41 | +169 −2,829 | 69 |
| `458ba0f2` | 2c follow-up: the three `mnv2_render_*_xla_certified` pins, the `MobileNetV2Concrete` defs, 22 emptied AuditAxioms headers | 6 | +35 −147 | 3 |
| `4220ac74` | 2d, small nets: S.1 (the `Micro`/`TwoChan`/`Mini`/`Spatial` namespaces), S.2, S.4, S.5, S.7, S.9 | 13 | +50 −1,176 | 35 |
| `96e7d3b0` | 2e, generic: A.1 (5 of 20), A.3, A.4, A.6, A.7, B.4, B.5, B.7, B.8, X.1. `ConvLossFold.lean`, `AdamRender.lean` deleted | 25 | +55 −1,398 | 37 |
| `281b6beb` | the docstring gate (Pass 2): resolver + markers `Render Live Tied Back Fwd`, stale citations fixed | 38 | +142 −81 | 0 |

The cuts total +542 −8,101 lines. AuditAxioms went from 1,643 to 1,430 verdicts, 213 pins
fewer. Group sizes differ from the survey tables because the cuts also re-pointed prose and
deleted what the fixed point orphaned.

**Notes on what landed:**
- **2c follow-up.** Kept `flatConvStride2Xla_weight_grad_has_vjp_correct`, generic API the fixed
  point would have taken. Still a candidate: `mnv2_render_depthwiseW_certified`. It has been
  audit-only since 2a and is the only user of the pinned `mnv2_depthwise_weight_grad_bridge`.
- **2d.** `CnnConcrete` built fine without the shared `_proof_1`. `MnistCNN`'s header and
  Proofs/README.md now name `TrainedCnn` as the MNIST-CNN witness. `JacobianSeal` keeps only the
  `HasVJPAt` seal.
- **2e.** X.1 also took the dead `*Rep_has_vjp` defs, `mobilenetv2FwdGraphFull(_faithful)`,
  `denoteMobilenet` and `mobilenetv2Forward_full`. `denoteMobilenet` shares its matcher with the
  other `denote*` functions, so the fixed point never reached it. B.4's wording was changed in
  `scripts/lipschitz_cert_pair_sdp_full.py` and in both generated SDPFull files, together.
- **A.1b, held (15 pins).** `IRPrint.lean` and `check_ir_codegen.py`, the execution oracle the
  book describes, name these as the proofs behind what they print. No StableHLO twin can stand
  in for them. They are:
  - `dense_at_bridge`, `relu_at_bridge`, `mlp_fwd_bridge`, `mlp_fwd_preact1`;
  - `se_back_bridge`, `softmax_back_bridge`, `gelu_back_bridge`, `swish_back_bridge`,
    `sigmoid_back_bridge`;
  - `denote_subst3`, `maxpool3_node_bridge`, `conv3_node_bridge_1to2`, and `conv_compose3`
    (the only user of `denote_subst3`);
  - `maxpool_flatten_bridge`, `conv_flatten_bridge_1to2`.
- **The docstring gate.** `tests/DocstringCheckRefs.lean` now resolves `private` declarations,
  `File.decl`, module and namespace names, and a scanned file's basename. With the new markers it
  checks 2,303 citations, up from 1,314.
  - Every stale row of the Pass 2 table is fixed, except two that were not stale:
    `convDownWGrad`/`patchifyWGrad` are private defs of `tests/TestConvNeXtStrided.lean`.
  - History mentions of deleted modules now cite the file (`MobileNetV2Render.lean`). The label of
    a deleted test is plain text (RenderCifar8Sgd02).
  - The baseline lost two uncited entries and gained nine, each with a reason: five real
    declarations outside the environment (`IRPrint.lean`, `tests/TestSDPA.lean`),
    `GemmFwdRest` (a MIOpen solver), two binders, and `convStridedXlaBackBatched` (cited as
    absent).
- **One survey claim was wrong.** It said `mobilenetv2ForwardPaper_eq_chain` was stale. It is
  declared, at `MobileNetV2FullVJP.lean:542`.

## Next session: correctness findings 2–11 and the decision groups

Checked against `281b6beb`. The full text of each finding is in Pass 1 below.

**Correctness findings.**

| # | Finding | State at `281b6beb` | Fix |
|---|---|---|---|
| 2 | The r34 and MNv2 spec rungs (`resnet34Verified_fwd_faithful`, `mobilenetv2Verified_fwd_faithful`) are stated at per-example forwards no artifact renders | fixed: `denoteR34FullB` / `denoteMobilenetB` map the SAME committed layer lists to `resnet34ForwardB_full` / `mobilenetv2ForwardB_full` at every `N`; `resnet34VerifiedB_denote_eq` and `mobilenetv2VerifiedB_denote_eq` are `rfl`, `*VerifiedB_fwd_faithful` compose the batched T2 apexes; all in SpecVJP, pinned. The per-example rungs stay until R.A / M.2 are ruled on | R.A and M.2 are now ordinary cuts |
| 3 | BN node seam: the RenderB files emit `.bnBatchBack` (44 sites) and never `.bnBatchLABack`, but the T3 ties state the BN cotangent at `.bnBatchLABack`'s `den` | fixed: `EnetTiePoC.den_bnBatchLABack_eq_bnBatchBack` (the two `den`s are one map up to the associativity relabelling `reassocB`; the scatters collapse because `Fin.cast` is a bijection), `bnBackB_eq_den_bnBatchBack` (the certified cotangent every batched tie threads is the emitted node's `den`), `ResNet34TieB.bnInB_eq_den_bnBatchBack`; all pinned | — |
| 4 | Stem-pool seam: `maxPool3s2BackFlat` (`ResNet34StepTieB`) vs `maxPool3s2FlatBack` (whole-back tie) | fixed: `maxPool3s2BackFlat_eq_flatBack` (the two scatters are one map, no smoothness needed — `sum_flat3` re-indexes one into the other) and `den_maxPool3s2BackB_eq_flatBackB` (the emitted batched node denotes `maxPool3s2FlatBackB`), both in `ResNetBackChains.lean`, pinned | — |
| 5 | The book cites the wrong theorem in five places (4295; 6242–6249 and 6748; 391; 16279; 16306), and the IRPrint passages overclaim "by construction" | fixed, five commits `2b88d7bb`..`f6da580d` (front matter, ch 3, ch 4, ch 6, appendix C). Two of the survey's readings were wrong: `cifarCnn8_has_vjp_at` exists (a def, CifarCNN.lean:441) — the issue was the no-BN VJP cited in the BN section; and an fp8 graph IS emitted (`cifar8b_fp8_adam_train_step.mlir`, `.convF8`), it is `E4M3Fold`'s linear graph that is not. Also fixed: a `finSum` that never existed (l.259) and `matmul_left_const` in `fig:spines`. Ch 6's spec-rung sentence changes again with finding 2 | — |
| 6 | `vit_net_tiedGB` / `cnx_net_tiedGB` certify every shipped AdamW artifact but are cited nowhere outside Lean | open | decide where to cite: the ImageNet sections, the certs.yml table, the yaml |
| 7 | certs.yml / READMEs / yaml | certs.yml and the READMEs fixed in the sweep: the r34 row cites `ResNet34PoCB.convStrided{W,B}GradB_den`, the two even-kernel clauses are gone, Certificates/README names its eight hand-written files | yaml fixed in the follow-up commit (:99 → `vitTiny_has_vjp_correct`, the seals paragraph, 4b + the smoothing row, :227, :370). certs-heavy.yml:27 stays: it is a history comment describing the pre-2026-07 filter that matched nothing |
| 8 | Tests and roots on retired artifacts | `ConvLossFold` deleted in 2e; the regen-script line dropped in the sweep; `TestResnet34Train`, `TestResnet34TrainPC` and `ResNet50BlocksCertified` deleted with R.A | `TestMobilenetV2Train` deleted with M.2; `TestMobilenetV2TrainPC` kept (a live `iree-compile` smoke of the committed AdamW bytes) | — |
| 9 | `mathlib_reuse_audit.md` calls `vit_net_tied_certified` the 2-block representative | fixed in the sweep | — |
| 10 | Names and docstrings claiming more than they state | 2 of 7 resolved by cuts; the other 5 fixed in the sweep (`smooth_cp_mlpT_demo` through its generator; `linearTrainStepModuleV` → `linTrainStepFaithfulV` at six sites) | — |
| 11 | `StableHLO.roundtrip` pinned twice | fixed in the sweep (1,429 pins) | — |

**Decision groups.** The recommendation column is from the end of session 2; the user has not
ruled on any of them yet.

| Group | Size | The decision | Recommendation |
|---|---:|---|---|
| R.A per-example r34 tier | 393 / 6 | CUT 2026-09-19 (the whole per-example chain: 37 declarations, 15 pins; `ResNet34RenderPC.lean`, `ResNet50BlocksCertified.lean`, `tests/TestResnet34Train{,PC}.lean` deleted; `ResNet34BackCertifiedTie` keeps its five leaf ties) | done |
| M.2 per-example MNv2 paper tier | 276 / 4 | CUT 2026-09-20 (85 declarations, 18 pins incl. the 2c leftover `mnv2_render_depthwiseW_certified`; `MobileNetV2PaperWholeBackCertifiedTie.lean` deleted after its generic apex moved into the batched tie; `tests/TestMobilenetV2Train.lean` deleted; `IVW`/`IVWNoExp`/`IVPos`/`IVNoExpPos` stay for the batched tier) | done |
| S.3 dense-head restatements | now 10 pins | the MLP pair left the group once `ca9ea890` used it; certs.yml's per-op column cites `CnnPoC.db5_den` and `CifarPoC.db7_den` | cut; re-point the two certs.yml cells |
| S.6 MLP layer-bridge restatements | 106 / 6 | reverses `18c31142`; part of the IR tier | keep; decide with A.1b |
| S.8 `MlpCanonical` aliases | 31 / 8 | an audit surface by design (fixed point 1,510 / 35) | keep; fix its header (finding 10) |
| B.1 FloatClose demos | 158 / 7 | reverses the earlier float review's "every `floatClose_*` stays"; the whole-net consumers were deleted 2026-09-08 | cut (the float-budget work closed 2026-09-05) |
| B.1b FloatClose op instances | 127 / 4 | yaml 4d cites them collectively | keep |
| B.2 Lipschitz demo ladder (generated) | 75 / 9 | loses the Frobenius → Schatten-4 → Schatten-8 comparison; generator change | keep |
| B.3 smoothing demos + Hoeffding tier | 116 / 8 | ~1,900 kernel panels of CI | cut, except keep `smoothing_mc_certified` |
| B.9 scorecard bookkeeping (generated) | 22 / 4 | `scorecard_counts` calls itself legacy; generator change | cut |
| X.3 `.correct` projections | 28 / 4 | a repo-wide convention, everywhere or nowhere | keep |
| A.1b IR bridges | 15 pins | the book's IR tier and `check_ir_codegen.py` | keep the IR tier; fix IRPrint's "by construction" |

**Suggested order.**
1. One sweep commit: findings 7 (without the yaml), 8's regen-script line, 9, 10 and 11. Then
   the yaml fixes as their own commit, following the yaml style rules.
2. Finding 5 in the book, one chapter per commit.
3. The proof work: 4, then 3, then 2. After that, R.A and M.2 are ordinary cuts.
4. The decision-group cuts the user approves, one commit per family.

**Also found, not in any group: unpinned dead code.** `size.py` never counts a declaration with
no users at all. There are about 25 in MobileNet/EfficientNet, dead before the census started:
- `sigmoidScalarDeriv_eq`, `sigmoid_has_vjp_correct`, `mbconvResidual_has_vjp_at`;
- the four `enet*Layer` defs;
- `mnv4Family*`, `mnv4Stage14`, `mnv4BlockLadder`, `mnv4*DWSlot_*`, `mnv4UibSkipBlockOfKs`;
- MobileNetV2.lean's strided infrastructure: `invresBodyStrided` and its lemmas,
  `convBnRelu6Strided_differentiableAt`, `dwBnRelu6Strided_*`. The last user of
  `invresBodyStrided` went with `mobilenetv2Forward_full` in 2e.
- `Mnv2Live.bn1_devSum_scale`, `Mnv2Live.invresBody₂_eq`, `Mnv2Live.fwdFull_differentiable`,
  `Mnv2RealSeal.fwdR_has_vjp_correct`.

A sweep needs a query for "unpinned, no env users, no token users", excluding auto-generated
`recOn`/`casesOn`/`inst*`. The earlier item-1 query did this per directory.

**The per-family recipe that worked** (all in `scripts/audit_census/`):
1. **Census.** Full `lake build Certs` first (declaration ranges come from the oleans), then
   `run.sh` into a fresh `CENSUS_DIR`.
2. **Dry run.** `retire.py groups.json` checks book / yaml / certs.yml / README citations,
   qualifier-aware. A `keep` list holds generic API out of the fixed point, and declarations
   nested in another's range (structure fields) are never cut.
3. **Cut.** Re-point the real citations, then `retire.py --apply`. It cuts by range, plus any
   `set_option … in` above, and drops the `#print axioms` lines.
4. **`orphans.py`** — the exact-graph pass. Common short names (`X`, `fwd`, `stem`) collide with
   tokens elsewhere, so the union graph stops the fixed point early; `ResNet34Concrete`'s defs
   and the weight wiring surfaced here.
5. **`mentions.py`** lists every leftover mention to re-point or cut.
6. **`emptysec.py`** finds emptied section headers.
7. **Delete emptied files.** Importers take the file's imports, drop its lakefile root, and `rm`
   its `.lake/build` products.
8. **`gate.sh`**, then record the state (`git stash create` + `git update-ref refs/census/<name>`)
   so several gated families can become one commit each: `git diff A B | git apply --cached`.

**Traps met.**
- The default target (`Proofs`) does not build `LeanMlir.lean`, so removing an import there left a
  stale `LeanMlir.olean` that `docstring-checkrefs` failed on. `gate.sh` now builds the LeanMlir lib.
- A bare `Dir/File.lean` in a moved docstring fails the file-link check; write the markdown link.
- Quote shell heredocs (`<<'EOF'`) when the text contains backticks. Don't chain an edit script
  after `git grep -c … &&`: a zero count exits 1 and silently skips the edit.
- `scripts/vjp_graph_sweep.py`'s ratchet was already failing at `HEAD` (18 batched holes against a
  ledger of `{efficientnetForwardB}`), and 2c removed `efficientnetForwardB` itself. The script is
  not in CI; the ledger needs re-deriving, not tuning.
- `retire.py` strips a `set_option … in` above a cut, but not an `open … in`. A dangling one at
  the end of SpecVJP broke the build ("unexpected end of input"). Grep for them after `--apply`.
- Lean shares `match_N` / `_proof_N` auxiliaries across declarations, so the environment graph
  shows false users: `CnnConcrete` → `Micro`, and every `denote*` → `denoteMobilenet`. Check the
  real users by hand before believing a declaration is live.
- `size.py` cuts seeds regardless of their users. Check a seed's users outside the cut set before
  applying.
- `gate.sh` does not build CertsHeavy. After touching IntervalBound(Conv) or `Certificates/`,
  build the direct importers (`IbpConvScorecardNet`).
- `emptysec.py` only scans `LeanMlir/`, and silently checks nothing when run outside the repo
  root. Empty section comments in `tests/AuditAxioms.lean` need their own pass: a `-- ` line
  that sat directly above a `#print` line before the cut and no longer does.
- A census run's line ranges are stale after any edit. A second cut in the same session means a
  rebuild and a fresh `run.sh` first; otherwise remove it by hand, as `denoteMobilenet` was.

## Pass 1: pinned declarations with no consumer but the audit

**Method.** The use graph is the union of two sources, so every error leans toward "used":
- The elaborated environment's constant references: `scripts/audit_census/Dump.lean`, 36,571
  project constants, 15,421 user-written declarations.
- A comment-stripped identifier-token match over all 610 tracked `.lean` files. It catches `#guard`,
  `#eval` and `example` uses, tests, apps, demos, `IRPrint` and the SDPFull modules.

The pins are the 1,643 `#print axioms` lines in `tests/AuditAxioms.lean`, naming 1,642 distinct
theorems: `StableHLO.roundtrip` is printed twice, at :652 and :2236 (the second from `d6af4213`).
**494 pins have no user outside the audit file.**

Six parallel read-only classifiers, one per family, read the code, the book, `certs.yml`,
`formalization.yaml`, the READMEs and `verified_mlir/`. Each gave every pin one verdict: APEX,
CITED, REPRESENTATIVE, SUPERSEDED or RESTATEMENT.

Groups were sized with `scripts/audit_census/size.py`:
- cut the seed pins;
- follow unpinned orphans to a fixed point;
- list any other pin that loses its last user, and follow it only where the classifier put it in the
  chain.

**Blind spots.** Treat every size as a lower bound, and build-check (`lake build Certs`) before
cutting:
- A `@[simp]` `rfl` lemma used via `dsimp` leaves no constant and no token.
- Lean's shared `_proof_N` auxiliaries link unrelated declarations (`CnnConcrete` ↔ `Micro`).
- A short name that collides with a test declaration looks used.
- Book citations split across TikZ lines or written as brace expansions
  (`mlp\_\{w2,w1,…\}\_step\_float\_close`) were missed by the name match. The classifiers caught
  them, and each group below accounts for them.

**Result.** 35 groups are recommended: superseded, representative, or restatement, with no citation
that can't be re-pointed. Cut jointly they come to **5,211 lines, 347 declarations and 225 pins,
across 66 files**. Ten pins the cut would orphan are kept: generic API, or book-cited.
By directory: MobileNet 918 lines, Codegen 884, Foundation 604, ViT 527, EfficientNet 525,
ResNet 509, Training 503, ConvNeXt 370, Float 171, Small 113, Certificates 72, Architectures 15.

Eleven more groups, about 1,470 lines and 72 pins with pins held, are decisions. Each reverses a
recorded call, touches the book, or needs new work first.

### Recommended groups

| id | group | lines | pins | superseded by / why | re-point first | keep / risk |
|---|---|---:|---:|---|---|---|
| VC.B | ViT T3 ladder (vit_net_tiedV/MHV) | 294 | 4 | the 2-block single-head rung and the generic-dims 12-block thread of the ViT fused-SGD tie ("the ladder, not the result" per its own header); `vit_net_tied_certified` (depth 12, 200 params, at `vit_train_step.mlir`) and `vit_net_tiedGB` carry it | ViTStepTie header bullets (l.14-22); the `vit_net_tiedMHV`/`MHV2` docstrings; AuditAxioms comments near 1596-1608 | keep `vit_block_tied(At)MHV`, `vitBlockFwdOMHV`, `vitBlockCotInAtMHV`, `vitCotB2outV` (capstone and GB use them) |
| VC.A | ConvNeXt scalar-LN leftovers | 285 | 13 | scalar-LN / per-element layer-scale ConvNeXt leftovers; superseded by the channel-LN `cnxResidBlockChBackGraph_faithful`, `cnxBlockChBack_eq_vjp`, `CnxPoC.chanLn{Gamma,Beta}Sgd_den`, `CnxPoC.cnx_render_lsgammaCh_certified` | ConvNeXtFold header + channel-LN section; ConvNeXtClose header table; ConvNeXtBackB0 §1; `cnxBlockChLayer` docstring; three AuditAxioms comments; `tests/TestConvNeXtTrainPC.lean` (unbuilt smoke) | keep book-cited `convNextBlock_has_vjp`, `layerScale_has_vjp`, `convnext_has_vjp(_correct)`; afterwards `SHlo.lnGammaSgd`/`lnBetaSgd` are dead IR (follow-up) |
| VC.C | per-example un-fused G tier | 183 | 19 | per-example un-fused ViT/ConvNeXt folds: no tie consumes them and the adam branch they fold writes no committed file since 4c legs 3-4; superseded by `ViTPoCGB`/`CnxPoCGB.*GradB_den` through the GB ties | the "per-example peer" columns of ViTFoldGB (l.21-36) and ConvNeXtFoldGB (l.27-33); ViTFoldG / ConvNeXtFoldG headers; AuditAxioms "4b.2 / 4b.3" comments | keep the five G lemmas the GB folds use: `ViTPoCG.{posEmbedGrad,clsGrad}_den`, `CnxPoCG.{layerScaleChGammaGrad,chanLnGammaGrad,chanLnBetaGrad}_den` |
| VC.E | ViT scalar rowLN bridges | 100 | 4 | scalar rowwise-LN bridges left by `cf0e1e39`; superseded by `vit_vecln{Gamma,Beta}_grad_bridge` | none | - |
| VC.G | ViT CertLayer restatements | 40 | 3 | CertLayer restatements; `vitTinyTrunk_is_shipped` is misnamed (ViTBackNet: "not tied to the committed artifact") | ViTBackNet header | - |
| VC.F1 | ViT rfl endpoint leaves | 16 | 2 | `rfl` endpoint leaves with no consumer | items 1-2 of the ViTWholeBackCertifiedTie(B) headers | - |
| R.B1 | R34 batched rfl/one-line restatements | 61 | 3 | one-line / `rfl` restatements the whole-net tie inlines | ResNet50WholeBackCertifiedTieB header ("maxPool3s2FlatBackB and its rfl tie") | - |
| R.C | ResNet34.lean Milestone-B scaffolding | 301 | 6 | ResNet34.lean Milestone-B scaffolding: the 1-channel, constant-output `ResNet34Concrete` (JacobianSeal calls it vacuous) and five `.correct` projections; the Live seals carry non-degeneracy | JacobianSeal.lean:8; Proofs/README.md:156-162 | the Live files use `bnForward_lb`, `bnForward_injective`, `relu_const_pos`, `flatConvStride2_eq_zero` (not in the cut); for `rblkPStrided`/`convBnReluStrided` pin the def instead |
| R.D | R34 Live-family trim | 114 | 5 | Live-family `.correct` projections and the 32x32 empty-chains seal (LiveFull beats it on depth, R34RealSeal on shape) | formalization.yaml:354 (drop "liveFwd2_jacobian_nonzero (empty chains)") | keep `liveFwd2_nonconstant` (LiveFull uses it) |
| R.E | CertLayer trunk/row wiring (r34/r50) | 107 | 6 | CertLayer trunk / row wiring no artifact renders: `r34Trunk_3463 := r50Trunk_3463`, row wrappers superseded by `R34BWeights`/`R50BWeights` | MobileNetV4BackB0:796-800; ViTBackNet:50/207/388; ResNet34FullB:22; ResNet50FullB:7/21 | the `*Layer` defs stay (FullBVJP) |
| R.F | R34 per-op float leftovers | 75 | 4 | per-example-BN R34 float bridges whose consumers the 2026-09-08 float chop deleted | FloatComposeBridge:20-22/238; both R34 float-bridge headers (they promise an `r34_float_close` that does not exist) | keep `add_close`, `bnStep_close`, `gapFlat_close` |
| M.1 | MNv2 six-block per-example tier | 697 | 8 | the six-block reduced MobileNetV2 per-example tier: its render writes only /tmp; superseded by `mnv2InputGradB_correct`, `mobilenetv2ForwardB_full_eq_slots`, `mobilenetv2FwdGraphB_full_faithful`, `mobilenetv2FwdGraphPaperEval_faithful`, `mnv2*CotIn_eq_vjp` | formalization.yaml:227; comments in ConvNeXtWholeBackCertifiedTie (x2), EfficientNetWholeBackCertifiedTie, ResNet34BackCertifiedTie, EfficientNetChainClose, the MobileNetBackChains module doc | keep `depthwiseStride2FlatXlaBack_eq_vjp_backward` (generic) and the RenderPC/RenderPCEval stage abbreviations (FullPaperEval and a test import them) |
| M.3 | EfficientNet 3-block representative | 418 | 9 | the 3-block EfficientNet representative (`efficientnetForwardB_has_vjp_committed` is misnamed: its "committed" forward is the 3-block one); superseded by the full-B0 T2, eval T2 and T6 | comments only; `scripts/vjp_graph_sweep.py`'s `EXPECTED_BATCHED_HOLES = {"efficientnetForwardB"}` | keep `stemBBack_eq_vjp_backward`, `headFwdBBack_eq_vjp_backward` and the stage abbreviations |
| M.4 | *GradsCertified bundles (MNv2/MNv4) | 357 | 13 | `*GradsCertified` bundles: conjunctions of pinned leaf folds; `mnv2_net_tiedB` / `mnv4_net_tiedB` call the leaves directly | MNv2 / MNv4 fold-file headers | `ResNet34PoCB.denseBGradB_den` loses its last user (keep: op fold) |
| M.5 | EfficientNet per-example scalar-BN spike | 162 | 7 | the per-example scalar-BN EfficientNet backward spike; superseded by the batched `mb*BackBatchedGraph_faithful` and `seBackBatched` | comments in ConvNeXtBackB0, EfficientNetBackChains | keep `residualBackGraph_faithful`, `seBlockBackGraph_faithful`, `bnBatchBack_faithful`; `backGraph_faithful`, `seGate_backGraph_faithful`, `mbconvBodyBackGraph_faithful` lose their last user and stay pinned |
| M.6 | CertLayer .faithful projections (MNv4/ENet) | 202 | 14 | CertLayer `.faithful` projections (MNv4 BackB0 x7, EfficientNet x5) and two `*Layer` wrappers nothing composes | the archived audit log narrates `mnv4UibSkipBlock_faithful` | keep `CertLayer.chain_faithful`, `mbResidBlockBackBatchedGraph_faithful` |
| M.7 | aliases and stale statements (MNv2/ENet) | 76 | 6 | an alias (`EnetPoCG.bnGammaGradB_den`), two `rfl` renames, two `.correct` one-liners, and `efficientnetLossCot_den` (stated at tokens the render no longer emits) | teach `vjp_graph_sweep.py` the two renames | keep `mbDownBodyBackBatchedGraph_faithful` |
| B.6 | MNv2 per-example XLA-SAME fused tokens (+M.9) | 87 | 10 | MNv2 per-example XLA-SAME fused-SGD tokens and their `Mnv2PoC` folds: the only emitter (`MobileNetV2Render.lean`) was retired in `32d657bb`; superseded by `EnetPoCG.convStridedXlaWGradB_den`, `Mnv2PaperPoCG.*Xla*GradB_den`, `depthwiseStridedXlaBackBatched_faithful` | ResNet34BackCertifiedTie:195; AuditAxioms:1429 comment | four are `@[simp]` rfl lemmas: build-check; keep `Mnv2PoC.depthwise{W,B}_den` (ConvNeXtStepTie uses them); follow-up: the five SHlo constructors and their parse arms |
| M.10 | MNv2 toy witnesses | 24 | 2 | toy MobileNetV2 witnesses: the degenerate `MobileNetV2Concrete`; the 2x2 `mnv2Live` seal, carried at 224 (`fwdR`) and depth 17 (`fwdFull`) | Proofs/README.md:160-170 (point at Mnv2Live) | - |
| S.1 | toy MNIST-CNN witnesses (Micro/Mini/Spatial) | 78 | 4 | toy MNIST-CNN witnesses; `TrainedCnn` is a trained discharge of the same `mnistCnnNoBn_has_vjp_at` | Proofs/README.md:161; MnistCNN header; CifarCNN.lean:232; tests/AUDIT_REPORT.md:343 | measured 78 lines; the real span is ~400 (a shared `_proof_1` and a short-name collision in tests/comparator hide it); keep `maxPool2Smooth_of_injective` |
| S.2 | SgdDescentCnn drift restatements | 239 | 10 | SgdDescentCnn drift lemmas, each a one-line `Conv{1,2}Slot.*` instance kept by `046ec333`/`a9c57a3b`'s "every pinned name unchanged" | none | - |
| S.4 | toy ResNet forward graph | 72 | 1 | the toy ResNet forward graph; `resnetFwdGraph` has no renderer; superseded by `resnet34FwdGraphB_full_faithful` | ResNet34RenderPC:5/18/193; StableHLO:1424 and the "peer of `resnetFwdGraph`" docstrings at 3930/3980/4126 | - |
| S.5 | orphaned SgdDescent helpers | 127 | 8 | helpers orphaned by the SgdDescent refactors (`b50ba380`, `a9c57a3b`, `850dac3b`, `0c2410d6`, `31f76507`) | SgdDescentCnn prose at :26, :906, :2347-2349, :2483-2485, :3396 | - |
| S.7 | ViT float orphans | 35 | 3 | float lemmas extracted for the ViT attention bridge the float chop deleted | AuditAxioms:1129 header (it also names the deleted `vit_grad_floatBridges`) | - |
| S.9 | small-net tail | 128 | 9 | `LinPoC.poc_train_step_certified` (superseded by `poc_train_step_tail_certified`), `poc_fwd_faithful`, `dot_close_linear`, `linear_float_close`, the global-HasVJP seal pair, `mlp_output_sgd_descends` | LinearFold header | - |
| A.1 | uncited IR bridges with a StableHLO twin | 194 | 20 | typed-IR bridges with a StableHLO-level twin (listed per pin in the classifier table) plus the unused `Fwd` IR | IRPrint.lean docstrings (14-21, 81-84, 157-163, 203-205, 293; string at 249); `check_ir_codegen.py`; MlpTrainStep:19/45/68; FloatBridge:948/1012/1110; BackwardMaps:43; CnnTrainStep:13; AuditAxioms 343-378; book 15899-15900 names `Fwd` | keep book-cited `conv_back_bridge_1to2`; `Back.add` becomes a dead constructor |
| A.3 | flat-Vec IBP path | 74 | 5 | the flat-`Vec` IBP path nothing uses; the dense scorecards use IntervalBound's E-path, the conv scorecard the tensor path | IntervalBound:72-73/173-174; IntervalBoundConv header | keep `flatten_reluT`, `relu_apply_eq_max` |
| A.4 | ConvLossFold (whole file) | 70 | 4 | ConvLossFold, the whole file: one-line `pdiv_comp` instances whose last consumer moved to `gradAt_comp_t3` (`046ec333`) | SgdDescentCnn:2 (import) and :1411; lakefile:108 (Certs root); LeanMlir.lean:27; AuditAxioms 680-684 | SgdDescentCnn may reach something through ConvLossFold's import of MobileNetV2Close; `lake build Certs` decides |
| A.6 | even-phase strided depthwise map tie | 22 | 1 | even-phase strided depthwise map tie built "for mnv2", which uses the XLA twin | DepthwiseBackCertifiedTie 20-22; BackwardMaps 200-214; AuditAxioms:977 (all three also misstate B0's padding) | - |
| A.7 | float-chop orphans | 21 | 2 | float-chop orphans | EvenKernelConvBack header | - |
| B.4 | lipsdp_slack_of_cert (unused LDL route) | 72 | 2 | the entrywise LDL route no SDP scorecard uses | PairSDP:137; SDPFull headers; `scripts/lipschitz_cert_pair_sdp_full.py` (14, 210, 257) | - |
| B.5 | AdamRender Phase-3b spec | 63 | 5 | AdamRender.lean, the Phase-3b spec (whole file); superseded by `adamWParamF_faithful` / `adamW_triple_faithful` | lakefile roots at 63 and 199; RmsPropStep:168 | - |
| B.7 | bnMean_num_le | 36 | 1 | numeral form for the deleted budget files (`0de261e8`) | none | - |
| B.8 | rfl restatements (den_batchOp_*_eq_*, patchEmbedBack) | 27 | 4 | `rfl` restatements (`den_batchOp` + `denOp` already give them) | none | `@[simp]` rfl: build-check |
| X.1 | SpecVJP Rep rungs (all nets) | 322 | 8 | SpecVJP `Rep` rungs for all five nets (and their dead `denote*Rep` / `*Rep_has_vjp`); superseded by the `*Verified` rungs | SpecVJP header | keep book/yaml-cited `vit_full` |

### Decision groups

| id | group | lines | pins | what | why it is a decision |
|---|---|---:|---:|---|---|
| R.A | per-example ResNet-34 tier | 393 | 6 | the per-example ResNet-34 tier (spec rung E, T6, block ties, the 2x2-pool leaf) outlived its renderer (`resnet34_train_step.mlir` retired 2026-09-06) | it holds the only spec->math tie for `resnet34Verified`: port a batched rung first (`denoteR34FullB` on `resnet34FwdGraphB_full_faithful`, template `efficientnetVerified_fwd_faithful`) or accept the gap; formalization.yaml:227; frees ResNet34RenderPC.lean's remainder (266 lines) and tests/TestResnet34TrainPC; ResNet34BackCertifiedTie also hosts leaf ties nine files import, so cut the r34-specific declarations only |
| M.2 | MNv2 per-example paper tier | 276 | 4 | the MobileNetV2 per-example "paper" (training-BN) tier; its artifact was retired 2026-09-06 | same spec-rung question as R.A (`mobilenetv2Verified_fwd_faithful`); the fixed point would take the book-cited `mobilenetv2_full_has_vjp_at` (book 6243-6249, 6748 describe the batch-BN run with the per-example names) |
| S.3 | dense-head per-layer den restatements | 135 | 12 | per-layer dense-head folds, each a one-liner of `Cifar8PoC.denseW_den`/`denseB_den` | decide the MLP tie first (finding 1): if b1/b0 join it, `MlpPoC.b{0,1}_den_certified` stop being restatements (group drops to 10 pins); certs.yml:214-216 cite three of them |
| S.6 | MLP layer-bridge restatements | 106 | 6 | MLP layer-bridge restatements (`weight_grad_bridge` / `bias_grad_bridge` at `mlpCotOut*`) | reverses `18c31142`'s call that the bridges "are the statements" |
| S.8 | MlpCanonical aliases | 31 | 8 | `MlpCanonical`'s eight Prop-valued aliases | an audit surface by design; its fixed point is 1,510 lines / 35 pins because it is the only Lean consumer of 27 book- and certs-cited pins; 7 "Canonical surface" banners point at it (4 emitted by generators) |
| B.1 | FloatClose demos orphaned by the float chop | 158 | 7 | FloatClose demos and aliases whose whole-net consumers the float chop deleted (fixed point 518 lines / 20 pins) | reverses float_second_pass's "every `floatClose_*` result stays"; yaml 4d l.219-222 |
| B.1b | FloatClose op instances (yaml 4d, collectively) | 127 | 4 | FloatClose op instances left without a consumer | cited collectively by yaml 4d; yaml:220 names `floatClose_residual` (an alias of `floatClose_addResidual`) |
| B.2 | Lipschitz demo ladder (generated sections) | 75 | 9 | the Lipschitz demo ladder: two toys and two certificates strictly implied by `trained_demo_certified_gram2` (generated sections) | loses the in-kernel Frobenius -> Schatten-4 -> Schatten-8 comparison; generator change |
| B.3 | smoothing demos + Hoeffding tier | 116 | 8 | smoothing demos and the Hoeffding tier (`smooth_cp_mlp_i1_radius_dec` is beaten by `smooth_dec_mlp_i1`; ~1,900 kernel panels of CI) | `smoothing_mc_certified` is a result in its own right, superseded only in use; keep `binomTail_check_5500of10112` if a deep-tail regime test is wanted |
| B.9 | generated scorecard bookkeeping | 22 | 4 | generated scorecard bookkeeping (`scorecard_counts` calls itself legacy) | generator change |
| X.3 | .correct projections (one-liners) | 28 | 4 | `X_has_vjp_correct := (X_has_vjp ...).correct` projections (R.C and R.D hold more) | a repo-wide convention: pin the def instead, everywhere or nowhere |

### Checked and not candidates

- **The per-net `*_lossCot_is_*_grad` one-liners** (r34, r50 ×2, mnv2, mnv4; 104 lines). They are
  the only users of `smoothedLossCotGraph_row` / `bceLossCotGraph_row_committed`, so they are the
  link from each tie's loss binder `g` to the actual loss.
- **`Bf16GradNodes`' nine `*GradBBf16_den`.** These are the only certificates of the gradient nodes
  in the committed bf16 artifacts. All nine constructors are emitted by the shipped renders.
- **The six book-cited IR bridges** (`dense_back`, `relu_back`, `conv_back_2to2`, `maxpool_back`,
  `layernorm_back`, `mlp_whole`). Each has a StableHLO twin, but retiring them means rewriting the
  IR-tier teaching sections: ch3 2566-2600, ch4 3462-3515, ch5 4664-4712, appendix 15895-15935 and
  16030-16080. Two cheap ones take one book line each: `layernorm_back_bridge` is literally
  `bn_back_bridge`, and `conv_back_bridge_2to2` is an instance of
  `convBackDenote_eq_input_grad_formula`.
- **The per-example ViT / ConvNeXt / EfficientNet fused-SGD ties and their folds.**
  `vit_train_step`, `convnext_train_step` and `efficientnet_train_step` are still committed.
  `cnx_net_tied_certified` also keeps `Mnv2PoC.depthwise{W,B}_den` and
  `ResNet34PoC.convStrided{W,B}_den` alive.
- **The MNIST float / descent tier and the SgdDescentCnn float capstones.** The book's §Finite
  precision names them. The 2026-09-05 closure covered only the deep-net whole-net budgets.
- **`maxPool3s2Back_faithful` and `bnBatchBack_faithful`.** Each is the only certificate of a node
  the renders emit (findings 3-4).
- **Other results in their own right:** MuonGeometry / MuonNewtonSchulz (yaml headline 4), the
  DataParallel pieces, `smoothing_cp_certified`, and the Smoothing CP / Dec scorecards. The two
  scorecards are two halves of one certificate over the same rows, not two methods.
- **The T6 `∑pdiv` readings** (`*InputGradB_correct`, `vitInputGradK(B)_correct`, seven nets).
  These are all-or-none: the yaml gives `efficientnetInputGradB_full_correct` as the T6 exemplar.

### Correctness findings (independent of any retirement)

1. ✅ *Fixed in `ca9ea890`.* **The MLP tie covered 4 of its 6 parameters.** `MlpPoC.mlp_train_step_tied_certified` (book 15961;
   certs.yml:214 "§1a tied") states W₂, b₂, W₁ and W₀ (`weightSgd "%x"`) but no b₁ or b₀ op. Its
   docstring (MlpFold.lean:159) says "all six". The missing pieces exist and are unused:
   `MlpPoC.b{0,1}_den_certified`. Fixing it changes a pinned statement, and
   `MlpCanonical.train_step_tied_certified` inherits it.
2. **Two spec rungs sit at forwards no artifact renders.**
   - `resnet34Verified_fwd_faithful` (SpecVJP:633) is stated at the per-example
     `resnet34FwdGraphFullPC`.
   - `mobilenetv2Verified_fwd_faithful` (book 6249) is stated at the per-example training-BN chain.
   - The trainers these specs drive run batch BN. EfficientNet already has the batched template
     (`efficientnetVerified_fwd_faithful` → `denoteEfficientnetB0 N`). Decision groups R.A and M.2
     wait on this.
3. **BN node seam.** Every batched render emits `.bnBatchBack`: 54 sites across EfficientNet,
   MNv2, MNv4, R34 and R50, and none emits `.bnBatchLABack`. The T3 ties state the BN input
   cotangent at `den (SHlo.bnBatchLABack …)`. Both print the same text, but their `den`s differ by a
   reindex. The only certificate of the node actually in the AST is `bnBatchBack_faithful`, which is
   audit-only. This is byte identity without tier identity again. Fix: state the tie at the emitted
   node, or prove the two `den`s equal.
4. **Stem-pool seam.** `ResNet34StepTieB.mpInB` and the `.maxPool3s2BackB` den use
   `maxPool3s2BackFlat`. The batched whole-back tie uses `maxPool3s2FlatBack`. No lemma equates
   them. `maxPool3s2Back_faithful`, stated at a per-example constructor, is the only bridge.
5. **The book cites the wrong theorems in five places.**
   - 4295-4298 cite the no-BN `cifarCnn8_has_vjp_at` as the gradient of the BN nets; the BN one,
     `cifarCnnBn8_has_vjp_at_correct`, is uncited.
   - 6242-6249 and 6748 name per-example-BN MobileNetV2 theorems while describing the batch-BN
     `mobilenetv2_adam_train_step` run.
   - 391 still lists a "depth-2 (×2 LN forms)" ViT.
   - 16279 refers to an R34 binary32 certificate deleted on 2026-09-08.
   - 16306 calls E4M3 "emitted"; its graph never is.
   - The IRPrint passages (2566-2600, 15906-15911) say the printout matches the IR "by
     construction". `IRPrint.lean` imports nothing and copies `Back` by comment only; only the
     StableHLO tier has the proven printer.
6. **The canonical ties are cited nowhere outside Lean.** `vit_net_tiedGB` and `cnx_net_tiedGB`
   certify every shipped AdamW artifact, including the book-quoted ImageNet ones. The book,
   certs.yml and yaml cite only the per-example SGD ties.
7. **certs.yml, READMEs, yaml:**
   - Proofs/README.md:315-316 cites `r34_net_tied_certified` and `mnv2_net_tied_certified`,
     which don't exist; they are now `*_net_tiedB`.
   - certs.yml:219 (r34 row) cites the retired-artifact `ResNet34PoC.convStrided{W,B}_den`.
   - certs.yml:209 and README:329 still count "4 even-kernel gaps"; ConvNeXt is at 182/182.
   - Certificates/README says every file is generated. Seven are hand-written.
   - formalization.yaml:
     - :227 gives the per-example R34 and MNv2 T6 ties.
     - :98-99 describes `vit_full` as vector-LN depth 12; it is the weight-shared scalar-LN
       tower, and `vitTiny_has_vjp_correct` fits the description.
     - :196-199 says the Live seals are "at 224x224"; both are at 32×32, and no R34 seal has
       both full depth and 224.
     - :204 and :377 say the smoothing estimation gap "is not formalized"; SmoothingMC / CP /
       NetSemantics close it.
     - :370 and certs-heavy.yml:27 give `LeanMlir/Proofs/LipschitzCertScorecard*`; the files are
       under `Certificates/`.
8. **Tests and roots on retired artifacts.**
   - `tests/TestResnet34Train.lean` smoke-tests the retired `resnet34_train_step.mlir` and names a
     deleted writer. It is still in `scripts/regen_verified_mlir.sh:582`, so that step throws.
   - `tests/TestResnet34TrainPC.lean` targets the same artifact.
   - `ResNet50BlocksCertified.lean` (394 lines) is a lake root with no importer and no pins.
     `ResNet50RenderB.lean:13` says it carries the certified VJPs.
   - `ConvLossFold` is a Certs root that SgdDescentCnn imports and does not use.
9. **Planning-doc correction.** `mathlib_reuse_audit.md`'s `1aafefd1` row calls
   `vit_net_tied_certified` "ViTStepTie's 2-block representative tie". It is the depth-12,
   200-parameter capstone at `vit_train_step.mlir`. The 2-block representative is `vit_net_tiedV`,
   which is why that one survived (group VC.B).
10. **Names and docstrings that claim more than they state.**
    - `efficientnetForwardB_has_vjp_committed` is about the 3-block representative.
    - `vitTinyTrunk_is_shipped` is not tied to an artifact.
    - `smooth_cp_mlpT_demo` uses the 784-dim MLP's count for the 49-dim `mlpT`.
    - `mlpVerified_back_faithful` says its graph is in `mlp_train_step.mlir`; it isn't, and no
      renderer prints `mlpBackGraph` or `cnnBackGraph`.
    - `MlpCanonical`'s header says every MNIST MLP path uses it; nothing in Lean does.
    - Four places name `linearTrainStepModuleV` as the writer of `linear_train_step.mlir`; the
      writer is `linTrainStepFaithfulV`.
    - `maxPool3s2Smooth_of_injective` names consumers that don't exist. The R34 Live witnesses use
      the 2×2 pool, not the shipped 3×3/s2.
11. **The duplicate `roundtrip` pin:** delete AuditAxioms:2236.

## Pass 2: stale docstring citations

**Baseline.** The gate checks 1,472 of the 18,158 backticked names in 504 files: the ones carrying a
`projectMarkers` substring. It is green. A scratch copy that checks every name with the same resolver
(`scripts/audit_census/AllRefs.lean`, before its fixes) finds 3,336 unresolved (1,526 names).
Filtered back to today's markers it finds 0, which matches the gate.

**The resolver, not the marker list, is what blocks widening.** Four citation forms are correct but
can never resolve:

| form | names | e.g. | why it fails today |
|---|---:|---|---|
| a `private` declaration | 131 | `lnFwdSite`, `paperSig` | the suffix index skips internal names, and `_private.*` is internal |
| `File.decl`, where `decl` is declared in `File` | 59 | `ResNet34FoldB.denseWGradB_den` | the resolver knows namespaces, not modules |
| a module name | 115 | `ViTRenderB`, `ResNet34BackB0` | not a declaration |
| a namespace | 23 | `ResNet34PoCB`, `Mnv2Live` | not a declaration (`env.isNamespace` doesn't see imported namespaces here) |

With all four resolved (exact lookups, no heuristics), the universe drops to 1,200 names. Two of the
11 baseline entries then resolve (`convWGrad_faithful`, `kernel_faithfulness_probe`).

**Markers.** The doc's candidates `_certified`, `_den` and `_eq` surface 9 names today and 0 real
ones after the resolver fix. `_den` also matches `_dense` (JAX names). Project-shaped markers do
better: `Render`, `Back`, `Fwd`, `Tied`/`Tie`, `Live`, `Close`, `Cot`, `Grad`, `Graph`, `Layer`,
`PoC`, `Seal`, `Fold`, `Cert`, `Chain` and `Spec` surface 61 names at 124 sites:
- about 20 stale (table below);
- about 12 deliberate history ("the retired `MobileNetV2Render`", "replaces `X` as the writer");
- about 30 vocabulary: MNv4 table labels `Cot{Pc,Dn,…}`, MIOpen `GemmFwdRest`, test-module names,
  JAX names, and the lib names `Certs` / `CertsHeavy`.

A history check agrees. Of the unresolved names that were declared once in git history, 30 are
declared nowhere now (47 sites). Read in context, the real stale ones are the table's.

The field-access rule (`foo_has_vjp.backward` resolves when `foo_has_vjp` does) accepts any
resolving prefix. Narrowing it to "a value prefix, not a type" catches only 7 citations, mostly
deliberate "no `String.toFloat?`" statements. `FloatClose.batchMapAux` slips past both versions,
because `FloatClose` is a def.

**Stale citations** (current-tense references to things that no longer exist, or under a wrong name):

| cited | where | now |
|---|---|---|
| `ResNet34RenderB.adamOne` | ConvNeXtRender:851, EfficientNetRender:1065, ViTRender:597 | `ResNet34RenderB.optOne` |
| `cnxBlockTiedAt` | ViTStepTie:160 | `cnxBlockChTiedAt` |
| `resnet34FwdFaithfulV` (present tense) | ResNet34RenderB:208 | retired with `ResNet34Render.lean` (2026-09-06) |
| `MobileNetV2Render` as a live peer | ConvNeXtRender:16, ViTRender:6, tests/TestConvBiasZero:325 | retired in `32d657bb`; `MobileNetV2RenderB` |
| `MobileNetV2Render.paperSig` | MobileNetV2RenderB:466 | private `paperSig` in MobileNetV2RenderB |
| `ResNet34Render.bnSite` / `.R34Bn` | MobileNetV2RenderB:1027, StableHLO:1789 | `bnSite` / `R34Bn` in ResNet34RenderB |
| `MobileNetV2Close`/`ChainClose` | SpecVJP:497 | `MobileNetV2ChainClose` deleted in `31f76507` |
| "a future `ResNet34Live`" | JacobianSeal:20/79/143 | the Live family exists (LivePC, LiveFull, LiveRealistic + seals) |
| `ResNet34Live.liveDown`, Stage-1 `liveDown`/`liveFwd` | ResNet34LivePC:8/16/61, ResNet34Live2:7-8 | `ResNet34Live.lean` removed in `ab543660`; `liveDownW` / `liveDownPC` |
| `lipschitzL2_comp` | LipschitzCert:17 | `LipschitzL2.comp` |
| `Proofs.clipScaleF_id_below` | tests/TestGradClipTie | `Proofs.StableHLO.clipScaleF_id_below` |
| `IR.maxPool2` | IRPrint | `Proofs.maxPool2` (CNN.lean) |
| `resnet34Forward_full_pc_eval` | EfficientNetRenderPCEval:38, MobileNetV2RenderPCEval:26 | deleted in `0de261e8` |
| `bnBatch` (as the emitted op) | PerChannelBN:515, EfficientNetBackB0:531 | the renders emit `.bnBatchBack` |
| `cnxGls` | ConvNeXtFullT:135 | deleted in `7dd8c978` |
| the `irBack` docstring | MobileNetV2RenderB:1156 | its function went in `32d657bb`; the docstring now sits on `irSig` |
| `adamParams`, `adamCot`, `trainStepAdamSched`, `bnPC`, `bnBackPC` | tests/TestMobilenetV2TrainPC:26/104/128/263 | removed 2026-07-28 |
| "the committed `convDownWGrad`/`patchifyWGrad` formulation" | tests/TestConvNeXtTTrainPC:160/174 | removed in `283bb2dc` |
| "`mobilenetv2-verified-adam` is an `ireeLink` binary" | tests/TestMobilenetV2AdamTie:41 | `ireeLink` retired in `64c5d2c7` |
| `FloatClose.batchMapAux` | ResNet50WholeBackCertifiedTieB:18 | no such lemma; `StableHLO.batchMapAux` exists. It passes today because the field-access rule accepts any resolving prefix |

Outside the gate's scan, and so never checked: Proofs/README.md:315-316 (finding 7), the
AuditAxioms section comments naming deleted files (`AdjointChainResidual.lean`, "the CIFAR-8 chain
tie", `MobileNetV2Render.lean`, `vit_grad_floatBridges`).

**Proposed gate change.** Land the four resolver fixes in `tests/DocstringCheckRefs.lean`. Fix the
stale rows above, and reword the history mentions as file paths or plain text. Then add markers in
order of signal: `Render`, `Live`, `Tied` first (mostly stale or history), then `Back` and `Fwd`
(about a dozen vocabulary entries to baseline, each with a `# why`). Tests, apps, `IRPrint` and the
lakefile are outside the environment, so citations of their declarations stay unresolvable; baseline
them or teach the resolver to accept a tracked `tests/X.lean` for `X`.

## Pin lists

Every group's pins as cut (names without `Proofs.` / `Proofs.StableHLO.`). Unpinned declarations the
fixed point also takes are in `size.py`'s output.

- **VC.B** — `ViTTiePoC.vit_block_tiedAtV`, `ViTTiePoC.vit_block_tiedV`, `ViTTiePoC.vit_net_tiedMHV`, `ViTTiePoC.vit_net_tiedV`
- **VC.A** — `CnxPoC.lnBetaSgd_den`, `CnxPoC.lnGammaSgd_den`, `cnxBlockBodyBackGraph_faithful`, `cnxBlockLayer`, `cnxResidBlockBackGraph_faithful`, `cnxBlockBack_eq_convNextBlock_vjp`, `cnxBlockBodyBack_eq_convNextBlockBody_vjp`, `cnx_lnBeta_grad_bridge`, `cnx_lnGamma_grad_bridge`, `cnx_render_lnbeta_certified`, `cnx_render_lngamma_certified`, `layerScale_gamma_grad_bridge`, `pdiv_layerScale_gamma`
- **VC.C** — `CnxPoCG.convBGrad_den`, `CnxPoCG.convStridedBGrad_den`, `CnxPoCG.convStridedWGrad_den`, `CnxPoCG.convWGrad_den`, `CnxPoCG.depthwiseBGrad_den`, `CnxPoCG.depthwiseWGrad_den`, `CnxPoCG.headBGrad_den`, `CnxPoCG.headLnBetaGrad_den`, `CnxPoCG.headLnGammaGrad_den`, `CnxPoCG.headWGrad_den`, `CnxPoCG.psWGrad_den`, `ViTPoCG.headBGrad_den`, `ViTPoCG.headWGrad_den`, `ViTPoCG.patchEmbedBiasGrad_den`, `ViTPoCG.patchEmbedWeightGrad_den`, `ViTPoCG.rowDenseBiasGrad_den`, `ViTPoCG.rowDenseBiasGrad_den_lnbeta`, `ViTPoCG.rowDenseWeightGrad_den`, `ViTPoCG.veclnGammaGrad_den`
- **VC.E** — `pdiv_rowLN_beta`, `pdiv_rowLN_gamma`, `vit_rowlnBeta_grad_bridge`, `vit_rowlnGamma_grad_bridge`
- **VC.G** — `vitTinyTrunk_is_shipped`, `vitTrunkV_eq_chain`, `vitTrunkV_faithful`
- **VC.F1** — `vitEmbedBackB_eq_vjp`, `vitPatchEmbedBack_eq_vjp`
- **R.B1** — `r34BodyBackBatchedGraph_faithful`, `maxPool3s2FlatBackB_eq_vjp_backward`, `r34StemBBack_eq_vjp_backward`
- **R.C** — `ResNet34Concrete.resnet34Concrete_has_vjp_correct`, `convBnReluStrided_has_vjp_at_correct`, `rblkPStrided_has_vjp_at_correct`, `resStage_has_vjp_at_correct`, `vjp_chain_at_correct`, `vjp_chain_correct`
- **R.D** — `ResNet34LiveFull.liveFwd2Full_has_vjp_correct`, `ResNet34LivePC.liveFwd2_has_vjp_correct`, `ResNet34LiveRealistic.liveFwd224_has_vjp_correct`, `ResNet34LiveSeal.liveFwd2_backward_nontrivial`, `ResNet34LiveSeal.liveFwd2_jacobian_nonzero`
- **R.E** — `r34DownBlockOfRow`, `r34Trunk_3463`, `r50DownBlockOfRow`, `r50Stage_faithful`, `r50Trunk_3463`, `r50Trunk_faithful`
- **R.F** — `FloatModel.bnPerChannelFlat_close_of`, `FloatModel.bnRelu_close`, `FloatModel.flatConvStride2F_close`, `FloatModel.reluAdd_close`
- **M.1** — `mobilenetv2FwdGraphFullPCEval_faithful`, `mobilenetv2FwdGraphFullPC_faithful`, `invresBodyBackPC_eq_invresBodyPC_vjp`, `invresBodyStridedBackPC_eq_invresBodyStridedPC_vjp`, `mnv2InputGrad_eq_mobilenetv2_vjp`, `mobilenetv2Forward_full_pc_eq_chain`, `mobilenetv2PC_has_vjp_at`, `residualBack_eq_vjp_backward`
- **M.3** — `efficientnetFwdGraphBEval_faithful`, `efficientnetFwdGraphB_faithful`, `enetTrunk`, `efficientnetB_has_vjp`, `efficientnetForwardBEval`, `efficientnetForwardB_eq_chain`, `efficientnetForwardB_has_vjp`, `efficientnetForwardB_has_vjp_committed`, `efficientnetInputGradB_eq_efficientnetForwardB_vjp`
- **M.4** — `Mnv2PaperPoCG.mnv2HeadDenseGradsCertified`, `Mnv2PaperPoCG.mnv2NoExpGradsCertified`, `Mnv2PaperPoCG.mnv2StemGradsCertified`, `Mnv2PaperPoCG.mnv2Stride1GradsCertified`, `Mnv2PaperPoCG.mnv2Stride2GradsCertified`, `Mnv4PoCB.mnv4BnGradsCertified`, `Mnv4PoCB.mnv4ConvNeXtGradsCertified`, `Mnv4PoCB.mnv4ExtraDWGradsCertified`, `Mnv4PoCB.mnv4FfnGradsCertified`, `Mnv4PoCB.mnv4FusedGradsCertified`, `Mnv4PoCB.mnv4HeadGradsCertified`, `Mnv4PoCB.mnv4PreStridedGradsCertified`, `Mnv4PoCB.mnv4StemGradsCertified`
- **M.5** — `broadcastBack_faithful`, `gapBack_faithful`, `mbconvResidual_backGraph_faithful`, `residual_dense_backGraph_faithful`, `seBlockFull_backGraph_faithful`, `se_dense_backGraph_faithful`, `mbconvBodyBack_eq_mbconvBody_vjp`
- **M.6** — `enetChain_faithful`, `enetHead_faithful`, `enetMBConvLayer`, `enetMbExp_faithful`, `enetMbNoExp_faithful`, `enetMbStrided_faithful`, `mnv2ResidBlockLayer`, `mnv4BodyOfRow_faithful`, `mnv4FusedStage_faithful`, `mnv4Head_faithful`, `mnv4PreStridedBodyOfRow_faithful`, `mnv4UibPostStridedBody_faithful`, `mnv4UibPreStridedBody_faithful`, `mnv4UibSkipBlock_faithful`
- **M.7** — `EnetPoCG.bnGammaGradB_den`, `EnetTiePoC.efficientnetLossCot_den`, `Mnv2Live.fwdFull_has_vjp_correct`, `Mnv2Live.mnv2Live_has_vjp_correct`, `mbExpFwdBackBatchedGraph_faithful`, `mbStridedFwdBackBatchedGraph_faithful`
- **B.6** — `Mnv2PoC.convStridedXlaB_den`, `Mnv2PoC.convStridedXlaW_den`, `Mnv2PoC.depthwiseStridedB_den`, `Mnv2PoC.depthwiseStridedW_den`, `convStridedXlaBiasSgd_faithful`, `convStridedXlaWeightSgd_faithful`, `depthwiseStridedXlaBack_faithful`, `depthwiseStridedXlaBiasSgd_faithful`, `depthwiseStridedXlaWeightSgd_faithful`, `mnv2_render_stem_convb_xla_certified`
- **M.10** — `Mnv2Live.mnv2Live_backward_nontrivial`, `MobileNetV2Concrete.mnv2Concrete_has_vjp_correct`
- **S.1** — `Micro.mnistMicroCnn_has_vjp_correct`, `Mini.miniCnn_has_vjp_correct`, `Spatial.spatialCnn_has_vjp_correct`, `conv2d_center3x3`
- **S.2** — `cnn1_logit_drift`, `cnn1_pool_l1_drift`, `cnn1_z2_entry_drift`, `cnn_conv2_logit_drift`, `cnn_pool_l1_drift`, `cnnb1_logit_drift`, `cnnb1_pool_l1_drift`, `cnnb1_z2_entry_drift`, `cnnb2_logit_drift`, `cnnb2_pool_l1_drift`
- **S.4** — `resnetFwdGraph_faithful`
- **S.5** — `MaxPool2MarginQ.pdiv3_eq`, `cnnDenseHeadCot_denote`, `conv2d_weight_pdiv_row_l1`, `convTap_in_l1`, `k4Idx_inj`, `maxPoolFlat_entry_lipschitz`, `mlp_input_logit_drift`, `sum_pinned_le`
- **S.7** — `FloatModel.smErr_nonneg`, `FloatModel.softmaxF_close_at`, `FloatModel.softmax_abs_le_one`
- **S.9** — `FloatModel.dot_close_linear`, `FloatModel.linear_float_close`, `FloatModel.pow_one_add_sub_one_le`, `HasVJP.backward_ne_zero_of_pdiv_ne`, `HasVJP.backward_nontrivial_of_fderiv_ne`, `LinPoC.poc_fwd_faithful`, `LinPoC.poc_train_step_certified`, `mlp_output_sgd_descends`, `mnistLinear_backward_nontrivial`
- **A.1** — `IR.bn_affine_back_bridge`, `IR.bn_normalize_back_bridge`, `IR.conv3_node_bridge_1to2`, `IR.conv_compose3`, `IR.conv_flatten_bridge_1to2`, `IR.denote_subst3`, `IR.denote_subst_fwd`, `IR.denseRelu_at_bridge`, `IR.dense_at_bridge`, `IR.gelu_back_bridge`, `IR.maxpool3_node_bridge`, `IR.maxpool_flatten_bridge`, `IR.mlp_fwd_bridge`, `IR.mlp_fwd_preact1`, `IR.relu_at_bridge`, `IR.se_back_bridge`, `IR.sigmoid_back_bridge`, `IR.softmax_back_bridge`, `IR.swish_back_bridge`, `IR.twoDense_back_bridge`
- **A.3** — `IBP.denseV_boxSound`, `IBP.flatConv_boxSound`, `IBP.ibp_certified_of_boxSound`, `IBP.maxPoolFlat_boxSound`, `IBP.reluV_boxSound`
- **A.4** — `conv_bias_total_loss_grad_fold`, `conv_total_loss_grad_fold`, `depthwise_bias_total_loss_grad_fold`, `depthwise_total_loss_grad_fold`
- **A.6** — `depthwiseStride2FlatBack_eq_vjp_backward`
- **A.7** — `decimateOddIdx_injective`, `padOdd_abs_le`
- **B.4** — `LipschitzCertDemo.lipsdp_slack_of_cert`, `LipschitzCertDemo.quad_form_nonneg_of_ldl`
- **B.5** — `adamB_certified_grad`, `adamW_certified_grad`, `adamWParam_apply`, `adamWParam_eq_scalar`, `adamWParam_wd_zero`
- **B.7** — `FloatModel.bnMean_num_le`
- **B.8** — `den_batchOp_denseRow_eq_denseRowF`, `den_batchOp_gelu_eq_geluF`, `den_batchOp_lnRow_eq_lnRowF`, `patchEmbedBack_faithful`
- **X.1** — `convNextFwdGraph_faithful`, `convnextRep_denote_eq`, `convnextRep_fwd_faithful`, `efficientnetRep_denote_eq`, `mobilenetv2Rep_denote_eq`, `mobilenetv2Rep_fwd_faithful`, `r34Rep_denote_eq`, `vitRep_denote_eq`
- **R.A** — `maxPoolFlatBack_eq_vjp_backward`, `r34DownBlockBack_eq_rblkPStridedPC_vjp`, `r34IdBlockBack_eq_rblkPC_vjp`, `r34InputGrad_eq_resnet34_vjp`, `resnet34Forward_full_pc_eq_chain`, `resnet34Verified_fwd_faithful`
- **M.2** — `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp`, `mobilenetv2ForwardPaper_eq_slots`, `mobilenetv2_full_has_vjp_at_correct`, `mobilenetv2Verified_fwd_faithful`
- **S.3** — `CifarPoC.dW5_den`, `CifarPoC.dW6_den`, `CifarPoC.db5_den`, `CifarPoC.db6_den`, `CifarPoC.db7_den`, `CnnPoC.dW3_den`, `CnnPoC.dW4_den`, `CnnPoC.db3_den`, `CnnPoC.db4_den`, `CnnPoC.db5_den`, `MlpPoC.b0_den_certified`, `MlpPoC.b1_den_certified`
- **S.6** — `IR.mlp_layer0_bias_grad_bridge`, `IR.mlp_layer0_weight_grad_bridge`, `IR.mlp_layer1_bias_grad_bridge`, `IR.mlp_layer1_weight_grad_bridge`, `IR.mlp_layer2_weight_grad_bridge`, `IR.mlp_whole_net_weight_grads`
- **S.8** — `MlpCanonical.has_vjp_at`, `MlpCanonical.has_vjp_correct`, `MlpCanonical.hidden_float_sgd_descends`, `MlpCanonical.input_float_sgd_descends`, `MlpCanonical.output_float_sgd_descends`, `MlpCanonical.train_step_tied_certified`, `MlpCanonical.w0_grad_close`, `MlpCanonical.w1_grad_close`
- **B.1** — `floatClose_bnRelu`, `floatClose_cifarStage`, `floatClose_convMixed_twice`, `floatClose_r50_stages_mixed`, `floatClose_reluConv`, `floatClose_reluConvMixed`, `floatClose_resBlock`
- **B.1b** — `floatClose_dense`, `floatClose_gap`, `floatClose_maxPool3s2`, `floatClose_residual`
- **B.2** — `LipschitzCertDemo.linear_demo_certified`, `LipschitzCertDemo.linear_radius_pos`, `LipschitzCertDemo.mlp_demo_certified`, `LipschitzCertDemo.mlp_radius_pos`, `LipschitzCertDemo.trained_demo_certified`, `LipschitzCertDemo.trained_demo_certified_gram`, `LipschitzCertDemo.trained_radius_gram_pos`, `LipschitzCertDemo.trained_radius_pos`, `clm_lipschitzL2`
- **B.3** — `binomTail_check_5500of10112`, `binomTail_check_9900of10112`, `binomTail_check_999of1000`, `binomTail_check_99of100`, `iIndepFun_eval_pi`, `smooth_cp_mlp_i1_radius_dec`, `smoothing_mc_certified`, `stdNormalQuantile_ge_of_09`
- **B.9** — `LipschitzCertDemo.cappedCerts_idx`, `LipschitzCertDemo.float_scorecard_count`, `LipschitzCertDemo.scorecard_counts`, `LipschitzCertDemo.unconCerts_idx`
- **X.3** — `depthwiseStride2Flat_has_vjp_correct`, `flatConvStride2_has_vjp_correct`, `layerScale_has_vjp_correct`, `swish_has_vjp_correct`

## Re-running

```
scripts/audit_census/run.sh refs      # needs a current `lake build Certs` (+ CertsHeavy, Codegen, ProofsMinimal)
CENSUS_DIR=/tmp/audit_census python3 scripts/audit_census/size.py groups.json sized.json
```

`run.sh` writes `modules.txt`, `decls.tsv`, `graph.pkl`, `audit_only.txt` and `allmisses.tsv` to
`$CENSUS_DIR`: about 2.5 min for the census plus 1.5 min for the citations, at 7 GB. Only
lib-reachable modules are dumped: `IRPrint`'s olean is from an older toolchain and fails to load.
After each cut, diff `audit_only.txt` against the previous run, as the fixed-point rule requires.
The retirement loop (`retire.py`, `orphans.py`, `mentions.py`, `emptysec.py`, `gate.sh`) is in the
Status section's recipe.
