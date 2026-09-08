# The float tier, second pass — the ℝ chains out of `Float/`, the float twins deleted

**Opened 2026-09-08 at the end of the cleanup session that landed step 1 (`0de261e`, branch
`cleanup/2026-09-08-unify-and-float-chop`); ✅ COMPLETED the same day in seven commits — see §2b for
what each step did and where it deviated from §3. §2 is the inventory it acted on, computed from the
import graph, not guessed.**

## 0. What step 1 did, and what it left

Step 1 deleted everything in `Proofs/Float/` that no kept module could reach: the 13 whole-net
budgets, their `Maps.*` envelopes, the forward float chains, the adjoint-chain tier and the ten
probe scripts (−31.7k lines, Certs 4003 → 3966). The decision behind it: the numbers were vacuous
(1e16–1e287 against logits of 10), thread closed 2026-09-05, "less is more".

**49 `Float/` modules survived (18.6k lines), and most of them survived for one reason only:** the
T6 certified backward ties — the theorems that DID earn their keep, by finding a reversed pool, a
stale head-LN slot, ViT's scalar affines and an even-kernel non-adjoint — are stated about ℝ chains
(`r34InputGradB`, `mnv2InputGrad`, `vitInputGradK`, …) that are *defined inside* the `*FloatBridge`
files, next to their float twins (`r34InputGradBF`) and the `floatClose_*` / `floatBridgesTo_*`
threads that nothing consumes any more. This pass separates the two: the ℝ definitions move to
where the ties are, the float side goes.

## 1. The criterion

A surviving `Float/` module is one of three things:

* **model core — keep as is.** `FloatBridge` (2197: `FloatModel`, the per-op `*_close` lemmas,
  `mlp_float_close_uniform`, the MNIST/CNN step bounds), `Binary32Instance` (313), the bf16-mixed
  and fp8 files (`ConvMixedFloatBridge`, `ConvMixedComposeBridge`, `DepthwiseFloatBridge`,
  `DepthwiseMixedFloatBridge`, `Bf16Fold`, `E4M3Fold`), `FloatSubnormalBridge`.
  Consumers: `Training/SgdDescentLinear`, `Training/TrainedLinearDescent`,
  `Certificates/LipschitzCertFloat`, `Foundation/MlpCanonical`; the book's "Finite precision"
  section argues from exactly these. ⚠ `FloatComposeBridge` (776) is in this bucket by import
  (the mixed bridges pull it in) but is the `FloatClose.comp` / `FloatBridgesTo` scaffolding —
  decide whether the mixed bridges need it or only import it.
* **a bridge file whose ℝ definitions a kept non-Float file uses — surgery.** Move the ℝ names
  in §2's table to a Foundation (or Architectures) file beside their tie, re-point the imports,
  delete the rest of the file.
* **a bridge file reachable only through other Float imports — delete** once the file that
  imported it stops existing. ⚠ Chase the ℝ dependencies first: a moved chain composes per-op
  `*Back` maps that may be defined in one of these (§3).

The rule that bit in step 1 and applies again: **a leaf theorem file has no importers by nature**,
so "no importer" is not "dead". Every file in the third bucket must be checked for theorems that
are about a shipped artifact (the `*FullB0Eval` / `*FullPaperEval` lesson) before it goes.

## 2. The inventory — what the kept non-Float files actually use, per surviving module

Computed 2026-09-08 by scanning every non-Float module's text for the declarations of every Float
module in its transitive import closure. "ℝ" = real-side names (move), "float" = float-side names
(these are the consumers to re-examine).

| module | lines | ℝ names the ties use (MOVE) | float names a kept file uses |
|---|---|---|---|
| `StridedConvBackFloatBridge` | 331 | `decimateBack`, `decimateBack_eq_vjp`, `decimateOddBack`, `decimateOddBack_eq_vjp`, `flatConvStride2Back`, `flatConvStride2XlaBack`, `flatConvStride4Back` | — |
| `CnnBackFloatBridge` | 344 | `convFlatBack`, `maxPoolFlatBack` | `floatBridges_convBack` (check who) |
| `DepthwiseBackFloatBridge` | 156 | `depthwiseFlatBack`, `depthwiseStride2FlatBack`, `depthwiseStride2FlatXlaBack`, `dwReverse` | — |
| `MaxPool3s2BackFloatBridge` | 366 | `maxPool3s2Flat_has_vjp_at_vec`, `maxPool3s2Flat_differentiableAt_vec` — ⚠ VJP lemmas, used by **`ResNet34FullBVJP`** (T1); belong in `Architectures/MaxPool3s2.lean` | — |
| `LinBackFloatBridge` | 198 | `diagBack`, `reluMaskBack` | `mlpInputGrad_floatBridges` ← `MlpCanonical` (tier-1 MNIST float — keep that half) |
| `ChannelLNFloatBridge` | 406 | `chanLNTensor3Back`, `rowLNVecFlatBack` (← `ConvNeXtStepTie`, `ViTVecLNBackCertifiedTie`) | `floatBridges_chanLNTensor3Back` (check who) |
| `MhsaBackFloatBridge` | 649 | `mhsaBackFlat`, `coreQFlat`/`coreKFlat`/`coreVFlat`, `clsScatter`, `vitBlockBack`, `vitBlockBackPR` | `floatBridges_mhsaBack`, `floatBridges_vitBlockBackPR` (check who) |
| `SdpaBackFloatBridge` | 649 | `mhSlab`, `mhsaSdpaBackQ`/`K`/`V` | — |
| `ViTBlockFloatBridge` | 1116 | `perRowFlat`, `perRowFlat_apply`, `perRowFlatPR`, `perRowFlatPR_apply`, `perRowFlatPR_comp`, `perRowIdxFlat` | — |
| `ViTWholeBackFloatBridge` | 175 | `vitBlockBackV`, `vitBlockBackVAt`, `vitTowerBackK`, `vitSavedPE`, `vitSavedBody`, `vitInputGradK` | — |
| `ConvNeXtBackFloatBridge` | 445 | `cnxBlockBodyBack`, `cnxDownBack`, `convnextInputGrad` | `floatBridges_cnxBlockBack` (check who) |
| `EfficientNetBackFloatBridge` | 199 | `mbconvBodyBack` | — |
| `EfficientNetWholeBackFloatBridge` | 227 | `efficientnetInputGradB` | `efficientnet_grad_floatBridges` (check who) |
| `EfficientNetFullWholeBackFloatBridge` | 170 | `efficientnetInputGradB_full` | — |
| `MobileNetV2BackFloatBridge` | 269 | `invresBodyBackPC`, `invresBodyStridedBackPC`, `mnv2InputGrad` | — |
| `Resnet34BackFloatBridge` / `Resnet34DownBackFloatBridge` | 67 / 129 | `r34IdBlockBack` / `r34DownBlockBack` | — |
| `Resnet34WholeBackFloatBridge` | 290 | `gapBack`, `r34InputGrad` | — |
| `Resnet34WholeBackFloatBridgeB` | 276 | `r34InputGradB`, `maxPool3s2FlatBackB`, `batchMapAux` (+ its `_apply`) | — |
| `MobileNetV2WholeBackFloatBridgeB` / `MobileNetV4WholeBackFloatBridgeB` / `Resnet50WholeBackFloatBridgeB` | 192 / 268 / 180 | `mnv2InputGradB` / `mnv4InputGradB` / `r50InputGradB` | — |
| `BnBackFloatBridge` | 348 | `bnGradInputBudget` (a bound, float-side by content) | `bnGradInputF`, `bnGradInput_close` ← `Codegen/BnBackComposeBridge` |
| `BnFloatBridge` | 496 | `bnForward_close_of`, `bnNormBudget`, `bnVar_nonneg`, `rsqrt_lipschitz` | `bnForwardF`, `bnForward_close`, `bnIstd_close` ← `Codegen/BnInputBridge` |
| `ResNet34FloatBridge` | 201 | — | `reluAdd_close` ← `Codegen/ResNet34BlockBridge` |
| `ViTFloatBridge` | 233 | — | `floatClose_gelu` ← `Certificates/GeluLipschitz` |
| `EnetFloatBridge` | 591 | — | `floatBridges_mbconvBody` (check who — a block tie citing it?) |
| `FloatBudgetEnv` / `FloatBudgetEnvBack` | 630 / 1130 | none of their own (the matches are `Maps.*` twins of real names) | — → **delete outright** |
| `FloatComposeBridge` | 776 | `cod_nonneg`, `residual` (real-side helpers) | `FloatBridges`, `FloatBridgesTo`, `FloatClose`, `floatClose_bn` — the scaffolding; see §1 |

Reachable only through other Float imports (bucket three, 17 files): `BnBatchFloatBridge` (383),
`BnEvalRuntimeFloatBridge` (221), `BnPerChannelFloatBridge` (259), `BnPerChannelBackFloatBridge`
(81), `BnXhatFloatBridge` (502), `PatchEmbedBackFloatBridge` (308), `Resnet34WholeFloatBridge`
(514), `SEBackFloatBridge` (234), `SoftmaxBackFloatBridge` (186), `ViTAttentionFloatBridge` (439),
and the seven model-core files. ⚠ Several of the first ten define per-op ℝ backward maps the
chains above compose (`seBack*`, `softmaxRowBack*`, `patchEmbedBack*`, the BN input-gradient
formulas) — §3.

Three Codegen files are float-side and only Float files consume them: `BnInputBridge`,
`ResNet34BlockBridge` (imported by `FloatComposeBridge`), `BnBackComposeBridge`. They go with the
float side unless the model core needs them.

## 2b. Progress

* **Step 1 DONE 2026-09-08** (staged on `cleanup/2026-09-08-unify-and-float-chop`). Two leaves, not
  one: `Foundation/BackwardMaps.lean` (the generic per-op maps — `reluMaskBack`, `diagBack`, the
  `perRow*` lifts, `maxPoolFlatBack`, `convFlatBack`, `decimateBack`/`decimateOddBack` with their
  `_eq_vjp` ties and `decimateOddIdx_injective`, the three strided-conv and three depthwise backwards,
  `maxPool3s2FlatBack` + `sum_flat3` + its tie + the `Vec`-point VJP pair) and
  `Architectures/ChannelLNBack.lean` (`rowLNVecFlatBack`, `chanLNTensor3Back`). The split is import
  hygiene: one leaf would have pulled seven ConvNeXt/ViT modules into every ResNet tie's closure.
  ⚠ The doc's "put the pool VJP lemmas in `Architectures/MaxPool3s2.lean`" was dropped — that file
  has 172 downstream modules and the addition would have rebuilt them all; the leaf rebuilds nothing.
  The `perRow*` lifts came out of `ViTBlockFloatBridge` now (not at step 6) because
  `rowLNVecFlatBack` needs `perRowIdxFlat`. The `FloatClose` ingredients (`reluMaskBack_abs_le`,
  `decimateBack_eq_filter`, the 3×3/s2 fibre count `maxPool3s2Back_mask_sum_abs_le`, …) stayed with
  the float side and die with it at step 7. Re-pointed: `DepthwiseBackCertifiedTie`, `ResNet34FullBVJP`,
  `ConvNeXtStepTie`, `ResNet34BackCertifiedTie` (its strided import only).
* **Step 2 DONE 2026-09-08.** `Nets/ResNet/ResNetBackChains.lean` holds `r34IdBlockBack`,
  `r34DownBlockBack`, `r34InputGrad`, `maxPool3s2FlatBackB`, `r34InputGradB`, `r50InputGradB`; `gapBack`
  went into `BackwardMaps.lean` (every conv net's head endpoint; `flatChannel` was already in that
  closure). The batched pool backward sits with the chains rather than in `BackwardMaps` because it
  needs `StableHLO.batchMapAux`, and the generic leaf does not import `StableHLO.lean`. The three
  ResNet ties compiled against the leaf with no proof change. ⚠ `Resnet34WholeBackFloatBridgeB` and
  `Resnet50WholeBackFloatBridgeB` were never Certs roots — they reached the build only through the
  two ties — so re-pointing the ties orphaned them (the audit's six prints of their float twins
  went unknown). Deleted on the spot, prints removed: −538 lines net for the step. The three
  per-example ResNet bridge files stay for now; they are roots and MobileNetV2/B0/SE's bridges
  still import `Resnet34WholeBackFloatBridge` for the float side of `gapBack`.
* **Step 3 DONE 2026-09-08.** `Nets/MobileNet/MobileNetBackChains.lean` holds `invresBodyBackPC`,
  `invresBodyStridedBackPC`, `mnv2InputGrad`, `mnv2InputGradB`, `mnv4InputGradB`; the four MobileNet
  ties re-pointed with no proof change. Same orphaning as step 2: `MobileNetV2WholeBackFloatBridgeB`
  and `MobileNetV4WholeBackFloatBridgeB` were reachable only through their ties, so both are deleted
  with their two audit prints. `MobileNetV2BackFloatBridge` (a root) keeps its float side until step 7.
* **Step 4 DONE 2026-09-08.** `Nets/EfficientNet/EfficientNetBackChains.lean` holds `mbconvBodyBack`,
  `efficientnetInputGradB`, `efficientnetInputGradB_full`; the three B0 ties re-pointed with no proof
  change. All four B0/SE bridge files are Certs roots, so none was orphaned; they keep their float side
  until step 7. Left float-side on purpose: `mbNoExpBodyBack` / `mbStridedBodyBack` (the b1/b2
  dischargers of the float capstone, tied to nothing) and every SE map in `SEBackFloatBridge`
  (`broadcastBackFlat` + its `rfl` tie, `seGateInputGrad`, `seInputGrad`) — the B0 ties take the SE
  backward as a supplied slot pinned to `seBlockFull_has_vjp`, and `EfficientNetBackB0` denotes the
  emitted `broadcastBack` straight from `broadcastFlat_has_vjp`, so nothing kept names them. Their
  audit prints (`broadcastBackFlat_eq_vjp` included) go with the file at step 7.
* **Step 5 DONE 2026-09-08.** `Nets/ConvNeXt/ConvNeXtBackChains.lean` holds `cnxBlockBodyBack`,
  `cnxDownBack`, `convnextInputGrad` (BackwardMaps names only — the LN slots are supplied, and the ties
  fill them with `chanLNTensor3Back` from `ChannelLNBack.lean`, which the block tie now imports
  directly). Both ConvNeXt ties re-pointed with no proof change; `ConvNeXtBackFloatBridge` is a root and
  keeps its float side (both the scalar-LN and the channel-LN folds) until step 7.
* **Step 6 DONE 2026-09-08.** `Nets/ViT/ViTBackChains.lean` holds the multi-head sdpa wrap
  (`mhSlab`, `mhsaSdpaBackQ`/`K`/`V` — on `Attention.lean`'s certified `sdpa_back_*`, no Float
  dependency), the flattened cores, `mhsaBackFlat`, the block backward in its three spellings
  (`vitBlockBack`, `vitBlockBackPR`, `vitBlockBackV`), `vitBlockBackVAt`, `vitTowerBackK`, `clsScatter`,
  the two saved prefixes and `vitInputGradK`. `ViTWholeBackFloatBridge` was ℝ from top to bottom, so it
  is deleted outright and its Certs root re-pointed at the leaf. `MhsaBackFloatBridge` and
  `SdpaBackFloatBridge` keep their float side (roots, and `PatchEmbedBackFloatBridge` imports the
  former). The three ViT ties re-pointed with no proof change. Left float-side: `towerBack` (the list
  fold only the float story used — `vitTowerBackK` is its own recursion), `vitGradFlat` (the pre-vector-LN
  skeleton) and everything in `SoftmaxBackFloatBridge` / `PatchEmbedBackFloatBridge` (the ties denote
  softmax and patch-embed straight from `Attention.lean`'s VJPs).
* **Step 7 DONE 2026-09-08 — the deletions.** Dead-by-closure, computed from the import graph with
  the lakefile roots and the audit's imports set aside: 31 `Float/` files, `Codegen/BnBackComposeBridge`,
  and the three saturation files (`Certificates/GeluLipschitz`, `Architectures/GeluSaturation`,
  `Architectures/SwishSaturation` — their only consumers were the budgets; git has them). Also
  `scripts/respell_mnv2_xla.py` (a one-off that named itself deletable). `Float/` is 13 files, 5.6k lines:
  the model core plus the r34 forward chain the bf16-mixed compose bridge builds on —
  `FloatComposeBridge` (its `floatClose_r34_stages` / `floatClose_bn` / … are what
  `ConvMixedComposeBridge` composes) and, through it, `ResNet34FloatBridge`, `BnFloatBridge`,
  `Codegen/ResNet34BlockBridge`, `Codegen/BnInputBridge`; and `LinBackFloatBridge` for
  `MlpCanonical`'s tier-1 MNIST backward. Audit 2007 → 1758 prints (161 by name, 88 more the checker
  found under bare / `Maps.` spellings), lakefile −177 lines of roots and their comments, yaml 4d and
  the book's "Finite precision" sentence rewritten, the docstring gate green.
* **THE PASS IS COMPLETE.** Remaining prose that names a deleted file is historical narrative inside
  `tests/AuditAxioms.lean` comment blocks that also describe live theorems, and `planning/`.
* **Bucket three, checked at step 1:** no kept non-test file uses any ℝ name from `SEBackFloatBridge`,
  `SoftmaxBackFloatBridge`, `PatchEmbedBackFloatBridge`, the five `Bn*FloatBridge` or
  `Resnet34WholeFloatBridge` — `seBack*`, `softmaxRowBack*`, `patchEmbedBack*` are consumed only by the
  Float net chains, so they move with their nets (SE at step 4, softmax/patch-embed at step 6), not
  into the generic leaf.

## 3. Order of work

Net by net, each a commit, each with `git diff verified_mlir/` empty (no artifact can move):

1. **The shared per-op ℝ backward maps first**, one Foundation leaf (say
   `Foundation/BackwardMaps.lean`): the strided/depthwise/conv/pool/LN/dense backward maps and
   their `_eq_vjp` leaf ties from `StridedConvBackFloatBridge`, `CnnBackFloatBridge`,
   `DepthwiseBackFloatBridge`, `LinBackFloatBridge`, `ChannelLNFloatBridge`, plus the two
   `MaxPool3s2` VJP lemmas to `Architectures/MaxPool3s2.lean`. Chase each definition's own
   dependencies (the BN input-gradient formula, `softmaxRowBack`, `seBack`, `patchEmbedBack`) into
   the same leaf. This is the step that decides how much of bucket three dies.
2. **ResNet-34 / ResNet-50** (per-example and batched chains, `r34IdBlockBack`,
   `r34DownBlockBack`, `gapBack`, `r34InputGrad{,B}`, `maxPool3s2FlatBackB`, `batchMapAux` lifts,
   `r50InputGradB`) → beside `ResNet34BackCertifiedTie{,B}`.
3. **MobileNetV2 / MobileNetV4** (`invresBody*BackPC`, `mnv2InputGrad{,B}`, `mnv4InputGradB`).
4. **EfficientNet-B0** (`mbconvBodyBack`, `efficientnetInputGradB{,_full}`).
5. **ConvNeXt-T** (`cnxBlockBodyBack`, `cnxDownBack`, `convnextInputGrad`).
6. **ViT-Tiny** (`mhsaBackFlat` and the cores, `mhSlab`, `perRow*`, `vitBlockBack*`,
   `vitTowerBackK`, `vitInputGradK`) — the largest, and the one net with no batched T6.
7. Then delete: every emptied `*FloatBridge` file, `FloatBudgetEnv{,Back}`, the bucket-three
   files whose ℝ content moved, the three Codegen float bridges if nothing kept needs them, and
   the `floatBridgesTo_*` / `FloatClose` scaffolding in `FloatComposeBridge` if the mixed bridges
   do not use it.

Expected: ~14k of the 18.6k lines go; `Float/` ends at roughly a dozen files.

## 4. What else moves with it

* **The book.** `content.tex` "Finite precision" still says *"`FloatBridge.lean` and the per-net
  `*FloatBridge` files budget every operator of every network, forward and backward, from the
  MNIST linear classifier to ViT-Tiny"*. After this pass only `FloatBridge.lean`'s per-op bounds
  exist; rewrite that sentence. No `\lean{}` tag is affected (zero on the float tier).
* **`formalization.yaml` 4d** already says the ℝ chains are still inside `*FloatBridge` files and
  names this pass; delete that sentence when it is done. No row names a moved definition.
* **Docstrings.** ~40 kept files cite `formalization.yaml` §4d or a `*FloatBridge` file by name;
  `docstring-checkrefs` catches identifiers, not file names — grep `FloatBridge` in
  `LeanMlir/Proofs/{Foundation,Architectures,Codegen}` afterwards.
* **Decide, do not assume:** `Certificates/GeluLipschitz` (uses `floatClose_gelu`) →
  `Architectures/GeluSaturation` → whose consumer was the deleted ViT budget; `SwishSaturation`
  likewise. Possibly dead now; check importers before keeping.
* `planning/archive/float_budget_numbers.md` and `planning/archive/float_budget_numbers_log.md` carry the
  DELETED banner; nothing else in planning/ needs to move for this pass.

## 5. Gates, per commit

`lake build Certs` (no corpus rebuild expected — nothing here touches `Tensor.lean` or
`StableHLO.lean`; a Foundation leaf that the ties import rebuilds the ties only),
`lake env lean tests/AuditAxioms.lean` (3-axiom clean; remove the prints of deleted lemmas, keep the
prints of moved ones under their new names), `lake exe docstring-checkrefs`,
`python3 scripts/check_audit_coverage.py`, `python3 scripts/check_render_coverage.py`,
`git diff verified_mlir/` empty. ⚠ `lake build -j N` is not a valid flag in this toolchain, and an
unbounded rebuild of many heavy modules at once has been memory-killed once on this box; the
Float-side rebuilds here are small.
