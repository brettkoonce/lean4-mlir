# Slice F — Foundation + Float: rubric review 2026-09-30

Scope: `LeanMlir/Proofs/Foundation/**`, `LeanMlir/Proofs/Float/**`, `LeanMlir/Proofs/SpecVJP.lean`,
excluding `ParamGrad.lean`, `ParamGradNodes.lean`, `SmoothedBatchLoss.lean`, `BceBatchLoss.lean`.
Tree `rubric-review` @ 55ad3a5a. Weighted toward `git diff 373059db HEAD` (Tensor.lean +237, the
float chop that deleted `FloatSubnormalBridge.lean` / `ResNet34BlockBridge.lean`, BackLinks +115,
SyncKit +77, SpecVJP −86). Three claims typechecked with `lake env lean` against the main
checkout's oleans (scratch `scratchpad/F/x.lean`): `FloatModel.u_nonneg` is derivable from `err`;
`sum_channel_fiber` closes in two lines on `sum_finProdFinEquiv`; `rndP 1 (5/4) = 3/2` and
`rndP 1 (-5/4) = -1`.

## Verdicts

| angle | verdict | findings |
|---|---|---|
| correctness | approve | 0 (float tier re-checked: see "Checked") |
| reuse | request_changes | 3 |
| scope | approve | 0 |
| attribution | request_changes | 3 |
| api-design | request_changes | 3 (+1 carried) |
| generality | request_changes | 4 |
| placement | request_changes | 3 |
| naming | approve | 0 (one rename folded into F-pl-3) |
| documentation | request_changes | 5 |
| proof-quality | request_changes | 1 (+2 carried) |

## Findings

### reuse

- **F-re-1** `LeanMlir/Proofs/Foundation/Tensor.lean:596` `sum_channel_fiber` — 13-line proof
  re-derives `sum_finProdFinEquiv` (same file, :582) by hand (`Equiv.sum_comp` +
  `Fintype.sum_prod_type` + a `hpull` helper + `Finset.sum_ite_eq`). **Fix:** body
  `rw [sum_finProdFinEquiv]; simp [Finset.sum_ite_irrel]`. **Evidence:** typechecked in scratch.
  **Cost:** 0 pins; 2 consumers (PerChannelBNGrad:89/105) unchanged; −11 lines; root file (Tensor,
  ~423 downstream) so batch with another Tensor edit. **Size:** S.
- **F-re-2** `LeanMlir/Proofs/Foundation/Batched/BackLinks.lean:515` `den_bnBatchLABack_eq_bnBackB`
  (new) — restates `bnBatchLABack_faithful` (:215) up to delta of `bnBackB` (:339, `:=
  (bnBatchLAHasVJP …).backward`); `bnInB_eq_bnBackB` (:475) is the same fact at an `.operand` leaf.
  Three names, one equation. **Fix:** move `bnBackB` above `bnBatchLABack_faithful` and state the
  latter with `bnBackB …` on the right; delete `den_bnBatchLABack_eq_bnBackB` and repoint its 7
  uses (BackLinks ×4, EfficientNetSyncStepTieG) at `bnBatchLABack_faithful`. **Cost:** 0 pins; ~8
  sites; `bnBatchLABack_faithful`'s statement changes only by folding a def (callers that `rw` with it
  keep working if they `unfold bnBackB` or already see `bnBackB`). **Size:** S.
- **F-re-3** `LeanMlir/Proofs/Float/FloatComposeBridge.lean:139` `floatClose_residual` — body is
  `floatClose_addResidual M hF` (:103); the docstring says the statements are defeq (`residual F`
  vs `fun v j => F v j + v j`). **Fix:** keep one: restate `floatClose_addResidual` with `residual F`
  (the `Residual.lean` API spelling) and delete `floatClose_residual`, or the reverse; `floatClose_residualBlock`
  (:128) follows. **Cost:** AuditAxioms 2 lines, formalization.yaml:335 one mention; −10 lines.
  **Size:** S.

### attribution

- **F-at-1** `LeanMlir/Proofs/Foundation/DataParallel/Sync.lean:19,170,180,508` — "Chan's
  `σ²_r + (μ_r − μ)²`" / "Chan's parallel variance" (`bnVar_row_shard_chan`) names the source but
  never cites it; same in `SyncKit.lean:60` and (other slice) `Architectures/BatchNorm.lean:142`
  `bnVar_shard_chan`. **Fix:** module docstring of `DataParallel/Sync.lean`: "Chan, Golub & LeVeque,
  *Updating formulae and a pairwise algorithm for computing sample variances* (Stanford
  STAN-CS-79-773, 1979; *Am. Stat.* 37, 1983)". **Evidence:** `grep -rn -i "LeVeque\|Golub"` empty
  repo-wide. **Size:** S.
- **F-at-2** `LeanMlir/Proofs/Foundation/Muon/Geometry.lean:1–50` (module) — the "every optimizer is
  steepest descent under a norm" table (SGD / sign / Muon / Shampoo rows) and `shampoo_eq_muon` follow
  Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology* (arXiv:2409.20325) — which
  `formalization.yaml:224` already names as the literature dependency, but the code does not. Muon
  itself (Jordan et al. 2024) is mentioned only as "(Jordan 2024)" in a `NewtonSchulz.lean:313`
  docstring; Shampoo (Gupta, Koren & Singer, ICML 2018) is uncredited. **Fix:** add a References
  paragraph to both Muon module docstrings: Bernstein–Newhouse 2024, Jordan et al. 2024 (Muon blog
  post URL), Gupta–Koren–Singer 2018, and Higham, *Functions of Matrices* (2008) ch. 8 for the
  Newton–Schulz / "Higham's quintic" material. **Size:** S.
- **F-at-3** `LeanMlir/Proofs/Foundation/IntervalBoundConv.lean:1–45` (module),
  `IntervalBoundConvQ.lean` — interval bound propagation is the central method and uncredited
  (`grep -rn -i "gowal\|mirman"` empty). **Fix:** cite Gowal et al., *On the Effectiveness of
  Interval Bound Propagation for Training Verifiably Robust Models* (arXiv:1810.12715, 2018) and
  Mirman, Gehr & Vechev (ICML 2018) in the module docstring. **Size:** S.

### api-design

- **F-api-1** `LeanMlir/Proofs/Float/FloatBridge.lean:53` `FloatModel.u_nonneg` — a structure field
  its own `err` field already forces (`err 1 : |rnd 1 − 1| ≤ u`). Redundant data every constructor
  must supply. **Fix:** delete the field; add `theorem FloatModel.u_nonneg (M : FloatModel) : 0 ≤ M.u
  := (abs_nonneg _).trans (by simpa using M.err 1)` under the same name, so the 112 `M.u_nonneg` uses
  are untouched; drop `u_nonneg :=` from `exactModel` (FloatBridge:1200) and `gridModel`
  (Binary32Instance:47). **Evidence:** typechecked in scratch. **Cost:** 0 pins; 3 lines.
  **Size:** S.
- **F-api-2** `LeanMlir/Proofs/Foundation/Batched/BackLinks.lean:397–413` `cInB`, `dInB`,
  `dStridedInB` — each docstring asserts "= `den convBackBatched`" (resp. depthwise, strided) but no
  lemma states it; consumers get there by `show … ; rfl` (see F-pq-1). **Fix:** add
  `theorem den_convBackBatched_eq_cInB : den (.convBackBatched wN W b e) = cInB N W b (den e) := rfl`
  and the `dInB` / `dStridedInB` peers beside the defs. **Cost:** +12 lines, no pins. **Size:** S.
- **F-api-3** `LeanMlir/Proofs/Float/Binary32Instance.lean:122` `FloatModel.rnd_zero` — the
  characteristic `rnd 0 = 0` of every `FloatModel` is not `@[simp]` (while `rndP_zero` is) and lives
  in the concrete-step section of Binary32Instance. **Fix:** move to FloatBridge.lean beside
  `FloatModel` (with F-pl-1) and tag `@[simp]`. **Cost:** 0 pins; 2 consumers in-file. **Size:** S.
- (carried: `planning/api_design_audit.md` §2.5) `pdiv3` / `pdivMat` peers of
  `pdiv_eq_of_hasFDerivAt` (Tensor.lean:102) never landed; `Architectures/CNN.lean:1086` and
  `MaxPool3s2.lean:356` still `unfold pdiv3` first.

### generality

- **F-gen-1** `LeanMlir/Proofs/Foundation/IR.lean:282,291,538,586` `conv_back_bridge_1to2`,
  `conv_back_bridge_2to2`, `conv3_node_bridge_1to2`, `conv_flatten_bridge_1to2` — fixed at
  `Kernel4 2 1 3 3` / `2 2 3 3` and 4×4 (a "Spatial instance" nothing defines any more), while the
  general `convBackDenote_eq_input_grad_formula` (:268, any `ic oc h w`, odd `kH kW`) is right above
  and each body is a one-line instance. **Fix:** replace the four by `conv_back_bridge`,
  `conv3_node_bridge`, `conv_flatten_bridge` over `{ic oc h w kH kW}` with the two oddness hypotheses
  (the proofs are unchanged with the `by decide`s turned into the hypotheses). **Cost:** AuditAxioms
  4 lines (:330/331/350/354); `LeanMlir/Proofs/Codegen/IRPrint.lean` (standalone `#eval` script, 5
  mentions). **Size:** S.
- **F-gen-2** `LeanMlir/Proofs/Float/FloatComposeBridge.lean:33` `floatClose_flatConv (_hβ : 0 ≤ β)`
  and `:67` `floatClose_gap (_hA0 : 0 ≤ A)` — hypotheses the proofs never use (silenced by `_`, kept
  after the 2026-09 `of_close` refactor removed their last use). **Fix:** drop both binders.
  **Cost:** AuditAxioms lines are name-only; 1 caller of `floatClose_flatConv` (ConvFloat docstring
  mention only), `floatClose_gap` none. **Size:** S.
- **F-gen-3** `LeanMlir/Proofs/Foundation/DataParallel/SyncBf16.lean:267,283,296,312`
  `int_log_two_pow_mul`, `int_log_abs_two_pow_mul`, `rndP_two_pow_mul`, `rndP_mul_four` — base 2 and
  `k : ℕ` only; the natural statements are `Int.log b (b ^ z * r) = z + Int.log b r` (`1 < b`,
  `0 < r`, `z : ℤ`; absent from Mathlib's `Data/Int/Log.lean`) and `rndP p (2 ^ z * x) = 2 ^ z *
  rndP p x` for `z : ℤ`, which also covers the collective's `1/R` scaling. `rndP_mul_four` is the
  `k = 2` instance with no consumer beyond AuditAxioms. **Fix:** generalise as stated and move them
  (F-pl-3); drop `rndP_mul_four` or keep it as a one-line `rndP_two_pow_mul p 2`. **Cost:**
  AuditAxioms 4 lines (:2217–2220), no Lean consumers outside the file. **Size:** S.
- **F-gen-4** `LeanMlir/Proofs/Foundation/Muon/Geometry.lean:111–470` — every Muon/Shampoo statement
  is at `Matrix (Fin n) (Fin n) ℝ`; Muon is applied to rectangular weight matrices and von Neumann's
  inequality, the polar factor and the nearest-semi-orthogonal statement all hold for
  `Matrix (Fin m) (Fin n) ℝ` (U semi-orthogonal). The restriction is disclosed (module "Scope",
  `formalization.yaml:92` "square matrices") but it is below the natural level. **Fix:** restate
  `muon_polar_is_max`, `muon_polar_achieves_nuclear`, `muon_polar_nearest_orthogonal` over
  `Fin m × Fin n` with `U : Matrix (Fin m) (Fin r)`, `Uᵀ U = 1`; keep the `_of_isUnit` square
  corollaries. **Cost:** yaml row + comparator tier pin `shampoo_eq_muon` (statement change).
  **Size:** L.

### placement

- **F-pl-1** `LeanMlir/Proofs/Float/Binary32Instance.lean:122,128` `FloatModel.rnd_zero`,
  `FloatModel.dot_right_zero` — general `FloatModel` facts (any `M`) filed in the concrete binary32
  file. **Fix:** move both to `Float/FloatBridge.lean` after `FloatModel.dot_succ`. **Cost:** 0 pins;
  consumers are in Binary32Instance, which imports FloatBridge. **Size:** S.
- **F-pl-2** `LeanMlir/Proofs/Foundation/OpaquePrefix.lean:145` `HasVJPAt.correct_of_backward_eq` —
  a `HasVJPAt` lemma unrelated to the opaque-prefix defs. **Fix:** move to Tensor.lean beside
  `HasVJPAt.backward_unique` in the next Tensor batch (root-file cost; 5 consumers all import Tensor).
  **Size:** S.
- **F-pl-3** `LeanMlir/Proofs/Foundation/DataParallel/SyncBf16.lean:267–316` — the `Int.log` shift
  lemmas and `rndP_two_pow_mul` are `rndP`'s own API sitting in a data-parallel file. **Fix:** move
  (generalised per F-gen-3) to `Float/RndP.lean` (already a Mathlib-only leaf that SyncBf16 imports);
  name the `Int.log` one `Int.log_zpow_mul` in `namespace Int` (and add it to
  `UpstreamDraft`/`planning/mathlib_upstream_drafts/` if upstreaming). **Cost:** AuditAxioms 4 lines
  (namespace for the `Int` one). **Size:** S.

### documentation

- **F-doc-1** `LeanMlir/Proofs/Float/FloatBridge.lean:23` and `Binary32Instance.lean:27` — both say
  the subnormal model lives in `FloatSubnormalBridge` ("the subnormal absolute-error term is
  `FloatSubnormalBridge`'s", "`FloatSubnormalBridge` states a model with the latter"); the file was
  deleted in the 2026-09-28 cut. Same stale row in `TRUST.md:28` (`FaithfulFloatModel`).
  **Fix:** "the subnormal floor is outside the model (no subnormal-aware model is stated)"; delete
  the TRUST row. docstring-checkrefs misses it (a module name, not a declaration). **Size:** S.
- **F-doc-2** `LeanMlir/Proofs/Float/RndP.lean:17` — "This is IEEE round-to-nearest minus overflow
  and subnormals". `rndP` rounds through Mathlib's `round` (`if 2·fract x < 1 then ⌊x⌋ else ⌈x⌉`),
  i.e. ties toward +∞: `rndP 1 (5/4) = 3/2` where roundTiesToEven gives `1`, and `rndP 1 (−5/4) =
  −1` (not odd; not roundTiesToAway either). No theorem depends on the tie rule (`rndP_err` and
  `rndP_two_pow_mul` hold for any), so only the prose is wrong. **Fix:** "round to nearest on the
  grid, ties toward +∞ (Mathlib `round`); IEEE's ties-to-even differs only at ties, where the error
  bound is the same". Also Binary32Instance.lean:11 ("round-to-nearest"), fine as is.
  **Evidence:** scratch proof. **Size:** S.
- **F-doc-3** `LeanMlir/Proofs/Foundation/IR.lean:150–164, 278, 288, 343–347, 536, 582` — the conv
  section comment says the reversed-kernel identity is proved only "at the concrete shapes the
  Spatial instance uses … The general-shape proof is the remaining Phase-2 item", and that CNN.lean
  "only asserts [it] in prose"; the general proof (`convBackDenote_eq_input_grad_formula`, :268)
  is in the same file and the CNN.lean prose is gone. :343–347 says BN/LN and softmax "need a
  `reduce`/`broadcast` IR extension … the remaining smooth layers", but `bn_back_bridge` (:399),
  `softmax_back_bridge` (:439) and `se_back_bridge` (:465) are below it. "Spatial instance" names
  nothing in the tree. **Fix:** rewrite both banners to describe what the section proves; drop
  "Spatial instance" with F-gen-1. **Size:** S.
- **F-doc-4** `LeanMlir/Proofs/SpecVJP.lean:9` (module title "each committed `VerifiedNetSpec`
  denotes its proven forward"), :257 and :313 docstrings ("the batch-statistics net every shipped
  MobileNetV2 / ResNet-34 artifact runs", "The typed graph the shipped … artifacts are printed
  from") — nine of the 33 `VerifiedNetSpec`s in `Verified/NetsCore.lean` are tied, all at 10 classes
  (`.dense 512 10`, `R34BWeights 10`, …); the ImageNet specs (`resnet34ImagenetVerified`,
  `mobilenetv2ImagenetVerified`, `efficientnetImagenetVerified`, …), ResNet-50 and MobileNetV4 have
  no spec→math tie here, and the `*in*` artifacts are 1000-class. Outside the slice the same
  overclaim is in `formalization.yaml:485` ("committed-spec ties for the 5 ImageNet nets in
  SpecVJP") and `LeanMlir/Proofs/README.md:20` ("each net's whole-model VJP is stated at the spec's
  denotation" — false since 2.7 removed the six canonical `*VerifiedHasVJP`). **Fix:** title "nine
  committed specs (the MNIST/CIFAR/Imagenette ones) denote their proven forwards"; the two docstrings
  "the Imagenette artifact"; yaml → "Imagenette-spec ties in SpecVJP; the ImageNet specs are not
  tied"; README row → "…and the linear, MLP and ViT-Tiny VJPs are stated at the spec's denotation".
  **Size:** S.
- **F-doc-5** `blueprint/src/content.tex:2545–2552` presents `mlpVerifiedHasVJP` (SpecVJP.lean:82,
  `= mlpHasVJP = HasVJP.canonical`) as "Again the VJP is proved for the denotation of the spec's own
  layers field" — the canonical witness exists for every function, so nothing is proved. The code
  docstring is honest; the book is the overclaim, and it is the only reason `mlpVerifiedHasVJP`
  survived api_design_audit §2.7 ("the blueprint lists them"). **Fix:** cite `mlpVerifiedHasVJPAt`
  (the `vjpCompAt` fold, with its `h0 h1` smoothness hypotheses) in the book; then delete
  `mlpVerifiedHasVJP` (consumers: `apps/mnist/MainMnistMlpVerified.lean`, `Nets/Small/MlpCanonical.lean`
  docstring). **Cost:** book text + 2 consumers, 0 AuditAxioms/yaml/comparator pins. **Size:** S.

### proof-quality

- **F-pq-1** `LeanMlir/Proofs/Foundation/Batched/BackLinks.lean:521–565` `cbsB_back_eq`,
  `dwbsB_back_eq`, `dwbsSB_back_eq`, `projB_back_eq` (new) — four copies of the same 6-line proof,
  each an uncommented `show cInB … (den (SHlo.bnBatchLABack …)) = _` followed by a closing `rfl`
  that rests on `den (.convBackBatched …)` unfolding to `cInB` definitionally. **Fix:** with F-api-2's
  `den_*_eq_*InB` lemmas the proofs are `rw [← hg, den_convBackBatched_eq_cInB,
  den_bnBatchLABack_eq_bnBackB …]` with no `show`/`rfl`; or prove all four directly from
  `cbsBHasVJP`/`vjpComp_backward` by `simp only` without the graph detour. **Cost:** 0 pins.
  **Size:** S.
- (carried: `planning/proof_cleanup_audits/reaudit_2026_09_24_open_findings.md` F6)
  `FloatBridge.lean:937` `softmaxF_close`, 133 lines, `div_eq_iff … div_mul_cancel₀` chain.
- (carried: same doc, F11) `Binary32Instance.lean:153` `binary32_linear_sgd_descends_concrete`,
  98 lines, `show`-dense.

## Checked, not findings

- **Float tier vacuity.** `FloatBridgesTo`/`FloatBridges` are gone (only strings in
  `tests/DocstringCheckRefs.lean` / `scripts/audit_census/AllRefs.lean`). `FloatClose A B f fF L`
  takes an explicit modulus `L` (no ∃); it is trivially true only for `A < 0` with `m > 0`, and no
  instance fixes `A`. `FloatModel` is honest: `exactModel` and the constructed `rndP` grids
  (`binary32`, `fp8E4M3`) inhabit it, `rndP_err` is proved from Mathlib, and the docs say the
  hardware-matches-grid step is trusted (TRUST.md:27) — no overclaim beyond F-doc-1/2.
  `binary32_e4m3_argmax_preserved`'s margin 122 is attainable (logit range ±471.4 at the stated
  bounds). `binary32_linear_sgd_descends_concrete` passes `Real.exp` as `fexp`, exact at `z = 0`,
  so "actual float gradient" holds there.
- The canonical-witness comparator challenges (`chk_reluHasVJP_correct`, `chk_mlpHasVJP_correct`,
  `chk_maxPool2HasVJP3_correct`) are content-free but labelled "by definition" in the challenge and in
  `formalization.yaml:284`; not an overclaim.
- Tensor.lean §2 landing (api_design_audit 2.1–2.5): `@[ext]` + `Subsingleton` on the five witness
  structures, `canonical` peers, `congr` + `@[simp] congr_backward`, `*_backward` rfl peels,
  `flatten_finProdFinEquiv` simp lemmas, `@[fun_prop]` Mat set, `IsHomog`, `backward_smul/add` — all
  correct, sensibly annotated; `CertLayer.ext` sound (vjp is pinned by `Subsingleton`, `ok`/`graph`
  are compared). Not bundling `backwardₗ` was the landed choice; not re-reported.
- `softmax_pos/nonneg/le_one`, `sum_softmax` (MLP.lean:279) — Mathlib has no softmax; `[NeZero c]`
  on `sum_softmax` is needed. `IndexCast.reassoc/unassoc` are the SHlo casts; their `den` lemmas live
  in ConvNeXtChannelLN because `reassocFwd` is an Architectures def (acceptable).
- `UpstreamDraft.lean` declarations match `planning/mathlib_upstream_drafts/PR{1,2}` one for one; the
  new "GaussianQuantile derives its Φ facts" line is true (it is the only importer).
- `DenseWSgdTied` / `DenseBSgdTied` (SgdNodes, new) are consumed by the four small-CNN step ties
  (Cifar8StepTie:174–176, …). `relu6MaskB` / `reluMaskB` two-mask duplication (one `_smul` each) is
  too small to factor.
- FloatClose kit (FloatClose.lean + FloatComposeBridge.lean) has no consumer outside `Float/` and
  AuditAxioms, but is on the roadmap through `formalization.yaml:335` and survived an explicit
  2026-09-28 cut (`planning/api_docs_followups.md:172`); scope clean.
- Attribution present and adequate: Higham *Accuracy and Stability* §2.2/§3.1 (FloatBridge,
  Binary32Instance), von Neumann trace inequality (Muon/Geometry), He et al. stem pool.
- OpaquePrefix's 25 `opaqueA*` defs: dependent widths rule out one indexed def; consolidated earlier.

## Gaps for the humans

- docstring-checkrefs resolves backticked declaration names only; backticked module/file names
  (`FloatSubnormalBridge`, F-doc-1) and prose section references (F-doc-3) go stale silently. A check
  that every backticked `CamelCase` token matching a `Proofs/**/X.lean` basename pattern still exists
  would have caught F-doc-1 the day the file was deleted.
- `formalization.yaml` / `LeanMlir/Proofs/README.md` / the book are not re-read when SpecVJP loses
  declarations (F-doc-4/5): the §2.7 deletion updated SpecVJP's own docstring only.
- Lean's unused-variables linter is silenced by `_`-prefixed hypotheses (F-gen-2); a lint for
  `_h…` binders in `theorem` signatures would list them.
