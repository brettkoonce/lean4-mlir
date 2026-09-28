# api_docs_followups.md — what the API-docs pass left for later

The API-docs pass (branch `api-docs-pass`, eight `docs(api):` commits 4bbb86f3..39039d89) fixed
every docstring doc-gen4 renders against `doc_audit/find_C..J_*.md`; per-site detail is in
`doc_audit/ledger_{C..J}.md`. It changed docstrings and comments only: no statements, no emitted
text, no markdown, no LaTeX. This file lists what it deliberately left, grouped by what the fix
needs. `doc_honesty_pass.md` is still the parent plan (its §0 protocol applies to every item).

## ▶ Start here

**§4.A and §4.B landed 2026-09-28** (round 1 `69609e00`: the text-only sites; round 2: the new
theorem blocks, the regenerated `\uses` lines and dep-graph figures). Every site, the statement
read against it and how each noun and number was checked is in `doc_audit/ledger_book.md`. Four §4
items turned out moot (README "87.58", ViT "4× accumulation", B0 "prose says 10 classes",
ConvNeXt-S/B); two more bf16-render overclaims were found and fixed (R34 ImageNet's sync-tie
sentence, the MNv4 side quest's). §1–§3 and §7 were done earlier; §5's rows have all landed; §6 is
down to report-or-reword leads. **Next: the cross-reference pass** (`book_xrefs.md`, ~220
claim-bearing refs), then the §6 leads if wanted.

## 1. Comments that still say something false (not rendered, but read by maintainers)

- `sdpa_back_{Q,K,V}` spellings (now `sdpaBack{Q,K,V}`): ViTMhsaBackCertifiedTie.lean:91,
  ViTMultiHeadChain.lean:149; grep the rest of `LeanMlir/` for `sdpa_back_`.
- ~~MaxPool3s2.lean:115 — "the r34 forward float chain (`floatClose_r34_stages` …)"~~ done: the
  comment names `floatClose_maxPool3s2`; `floatClose_r34_stages` is cut.
- ConvGrad.lean:~55 — cites a nonexistent `mlp_render_W*_certified`, says "rendered … denotes".
- TokenParamGrad.lean:~386 — cites a "§ B" that does not exist; "EVERY parameter … certified".
- EvenKernelConvBack.lean — section header says four leaf ties; the section has three.
- BatchNorm.lean:~401 — the γ/β "definitions" note is a plain `/- -/` block (audit D entry 19).
- MnistCNN.lean:274 — `--` banner flagged by audit H.
- LeanMlir.lean:62 — import comment (audit J).
- The remaining ⭐/⛔/⚠ markers in `--` comments (plan D4 said a full sweep of 136 files; this
  pass swept docstrings only). Decide whether comments get the same treatment.

## 2. Text inside emitted artifacts (needs emitter change + regeneration)

- The "every line is pretty(verified AST node)" banners in `mlp_train_step.mlir`,
  `cnn_train_step.mlir` and the other train steps that carry the report-only `%loss` block. The
  Lean docstrings now say "every line that feeds a returned parameter"; the artifacts still say
  "every line". Change the emitter strings, regenerate, `gen_mlir_manifest --check`.
- ConvNeXt `"121 of 180 params take %wdz"` (it is 182) — plan D2: generator fix, regenerate the
  18 ConvNeXt artifacts + MANIFEST in one commit.
- Stale printed strings in top-level code (string literals, so not touched): R50 `blurb`
  "LAYOUT SKELETON"; "proven … VJP via IREE" in VerifiedAttack; the planning ref in
  `spawnValStream`; `ViTRender.emitAdamVDP`'s advice to scale the batch to keep the BN tie
  (predates sync-BN).

## 3. Docs outside doc-gen4 (markdown, Python, lakefile)

- `LeanMlir/README.md` — module list missing `Pong`, `FloatFmt` (plan §3).
- `LeanMlir/Proofs/Codegen/README.md` — probably repeats "every line pretty" and the round-trip
  claim that the Codegen docstrings no longer make.
- `LeanMlir/Proofs/Certificates/README.md` — not read against the fixed docstrings.
- Python module docstrings of the certificate generators still cite planning docs
  (`trained_linear_descent.py`, `trained_cnn_witness.py`, others in `scripts/certs/`).
- `lakefile.lean` `lowererLink` docstring is mostly history.
- `tests/TestConvNeXtTrain.lean` says the retired AdamW tie was "179 of 180 bit-exact"; the old
  ConvNeXtRender docstring said "spread 0/180". The number was dropped from the docstring;
  the test's comment still disagrees with history.

## 4. The book pass (`content.tex`, README, yaml, CHANGELOG)

All of `doc_honesty_pass.md` §1 except the landing-page items (a)(b)(c)(j), which landed here —
reuse the landing page's wording (LeanMlir.lean) so the book and the API home page agree. Plus
what this pass found:

- ~~Dense-head overclaim~~ fixed with the §5 dense-head row: the four blueprint statements now
  match the capstones (all parameters).
- The ViT artifact the book quotes, `vitin_emadp128x4wxclipdropbf16`, is bf16 and drop-path:
  neither `vit_net_tiedGB` nor the batched input-gradient tie covers it. Same for the quoted
  ConvNeXt drop artifact.
- Check published text for: "ConvNeXt is the only unconditional whole-net tie"; ViT/B0 described
  as `HasVJPAt`; `convnextImagenetInputGradB_eq_vjp` covering ConvNeXt-S/B (it is T only);
  "4× accumulation" for ViT (it is 4 replicas × 128).
- README's "87.58" for EfficientNet no longer exists in any log (the tuned-lr log reads 87.65
  after epoch 80, peak 87.81 at 79).
- Comparator count is 87 (13 + 39 + 35), AuditAxioms 1,610 + Heavy 62.
- R50 section and content.tex:5309 stay with the separate R50 book pass (D3).

### 4.A What is still in `content.tex` (grep, 2026-09-28 at `db99bdee`) — **all landed** (`69609e00` + round 2)

Already fixed by earlier commits, do not redo: the round-trip wording (§1(e)), "codegen-shaped"
(§1(g)), the comparator 73/21 counts and the AuditAxioms count (§1(j)). Still present — line
numbers drift, re-grep:

- Pooling "measure-zero" / "argmax tie" (§1(f)): l.2726, 3664, 7444, 17124, 17193, 17361. Reuse
  `formalization.yaml` fidelity (2)–(3) as of `db99bdee`: the condition is `MaxPool2Smooth` /
  `MaxPool3s2Smooth` (each window's max attained at one cell); an all-zero post-ReLU window fails
  it, so for pooling it is no measure-zero corner. ReLU/ReLU6 "measure-zero" can stay.
- Theorem-budget table, ch 5 row (l.369): "ResNet-34 per example" does not exist; re-derive the
  row's count and the table total by its own counting rule (content.tex ~350–392) first.
- "exactly one theorem" (ch 5, l.5098); "stops at Imagenette" (l.6073).
- "the certified loss-descent step" (l.17101): state the per-layer form, and now the loss form
  (B1 below makes "∂L/∂θ" true for the seven ImageNet nets).
- LayerNorm citations (§1(i)): 11 `layerNormHasVJP` sites (l.9451–11716); ConvNeXt →
  `chanLNTensor3HasVJP`, ViT → `layerNormVecHasVJP`.
- Tie scope (§1(a)), headline sentence (§1(d)), batch / "line for line" (§1(h), l.2113),
  softmax Jacobian index (§1(l)), and the §4 bullets above (drop artifacts outside the ties,
  "only unconditional tie", ConvNeXt-S/B, ViT "4× accumulation").
- A2: B0 at 1000 classes — `efficientnet_net_tiedG` binds `nCls`; the prose still says 10.

### 4.B Statements landed since the book was last synced — **all cited now** (round 2: B1 a block per net + `def:hasgradat`/`thm:hasGradAt_comp`; B2 a clause per tie; B3 the B0 drop forwards on the forward block, MNv2/MNv4 in prose; B4 with the 4.A pooling rewrite)

Each wants a theorem block (`\lean{…}`, `\uses` regenerated by `blueprint_uses.py --fix` after
`lean_decls` is refreshed) or a clause in an existing one, plus a sentence where the chapter
states its tie. Decide per item whether it earns a block or a clause.

1. **Parameter-level loss gradients, all seven ImageNet nets** — every parameter gradient node IS
   ∂L/∂θ of the whole net, for any `L` with gradient `g` at the logits: `r34_net_lossGrad`,
   `r50_net_lossGrad` (+ `_smoothedCE`, `_bce`), `mnv2_net_lossGrad`, `mnv4_net_lossGrad`,
   `enet_net_lossGrad`, `cnx_net_lossGrad`, `vit_net_lossGrad` (each + `_smoothedCE`). Shared:
   `Foundation/ParamGrad` (`HasGradAt`), `smoothedBatchLoss_grad` (the emitted cotangent IS the
   smoothed loss's gradient), `Foundation/ParamGradNodes`. Chapters 5–9; the natural home is
   where each chapter states its train-step tie.
2. **Cotangent ties** — the chain cotangents the step ties thread are certified VJP backwards:
   MNv4 `mnv4{Body,SBody,Skip,Fused,Head}CotIn_eq_vjp`; ConvNeXt `cnx{BlockCotIn,DownCotIn,
   HeadDy}B_eq_vjp`; ViT `vitBlockCotInB_eq_vjp`, `vitCotB2outB_eq_vjp`; MNv2's two ends
   `mnv2HeadCotBlk_eq_vjp`, `mnv2StemCotC_eq_vjp`. Probably a clause each, subsumed in prose by B1.
3. **Forwards the book's quoted artifacts use** — MNv4 frozen-statistics eval at any input size
   (`mnv4FwdGraphBFullEval_faithful`, covers `_fwd_eval` and `_s256`); the `%do` dropout
   forwards (`mnv4FwdGraphBFullDo_faithful`, `mobilenetv2FwdGraphBFullDo_faithful`, f32 only —
   the `%do` artifacts are bf16); B0 stochastic depth + classifier dropout
   (`efficientnetFwdGraphBFull{,Eval}Drop_faithful`, f32, so it reaches the artifacts); MNv2's
   eval forward text tie (`FwdGraphTextTies`).
4. **MaxPool condition** — `MaxPool2Smooth` / `MaxPool3s2Smooth` replaced "all window cells
   distinct" (`maxPool2Smooth_of_pairwise` keeps the old discharge). Goes with the 4.A pooling
   rewrite.

Already in the book, not B: dense-bias descent (l.2383), small-CNN dense heads, ViT `nCls`.

## 5. Prose waiting on the D1 Lean changes (left verbatim on purpose)

Each row's landing rewrites these sites to the new statement.

| D1 row | Sites left verbatim |
|---|---|
| small-CNN dense heads | **landed** (Lean + the four fold files + CnnRender + blueprint thm:cnn_fold / cifar_fold / cifar8_step_tie / cifar8bn_step_tie). Bias descent landed too (`f7dd0419`, `SgdDescent/MlpBias`) |
| nCls for B0 | EfficientNetStepTieG (module + head), EfficientNetFullWholeBackCertifiedTie title, B0 Sync files, VerifiedNetsCore:909 claim ceiling, LeanMlir.lean "the ImageNet head its artifacts run" |
| nCls for ViT (+ `ty [10]`) | **landed**: both ViT-Tiny capstones bind `nCls`, `ty [nClasses]`, blueprint thm:vitTinyHasVJP_correct / thm:vit_whole_back and the §9 prose. SpecVJP's `Vec 10` is the Imagenette spec's own head, kept |
| MaxPool live-cell predicate | **landed** (2×2 and the 3×3/s2 stem pool): CNN.lean `MaxPool2Smooth` docstring, SgdDescentCnn module pool margin, TrainedCnnWitness `h_mp` bullet (generator), StableHLO/Basic `maxPoolBack` comments, blueprint thm:cnn_sgd_descends. "Measure-zero" for post-ReLU pooling fixed in `formalization.yaml` fidelity (2)–(3); the book's sites stay with §4 |
| `*CotIn_eq_vjp` MNv4 | **landed**: StepTieB module paragraph + `mnv4_net_tiedB` docstring name the five lemmas; WholeBackCertifiedTieB header has the train-step paragraph |
| `*CotIn_eq_vjp` ConvNeXt | **landed**: "certified loss-descent step" / "certified ∂Loss/∂θ" → `θ − lr·(certified per-layer Jacobian · chain cotangent)` + the cotangent lemmas, in ConvNeXtStepTie (module, block, `cnx_net_tied_certified`), `cnx_net_tiedGB`, ConvNeXtRender, ConvNeXtFold section header |
| `*CotIn_eq_vjp` ViT | **landed**: `vit_net_tied_certified`, `vit_net_tiedGB` and the ViTFold header state the per-layer form and name `vitBlockCotInAtMHV_eq_vjp` / `vitCotB2outV_eq_vjp` (batched `*B_eq_vjp`) |

Also open: slice I dropped eight unchecked "3-axiom-clean" claims; the audit is green
(1610/1610), so they can come back if wanted — cite `tests/AuditAxioms.lean` as a repo link.

## 6. Leads for a correctness pass (no false theorem found)

Still open (2026-09-28): the float-tier items, SpecVJP's canonical witnesses, the float descent
rungs' scope, and the certificates' `hp` — all report-or-reword, none a Lean gap.


- ~~`vit-fwd-b-tie` compares `vitFwdRenderB "vit_fwd"` against `vit_fwd.mlir`, which
  `vitFwdRenderB` itself writes~~ **done** `66e90873`: it ties the per-example chain to the
  committed bytes; both fwd-b-ties run in `proofs.yml`.
- ~~`MlirCodegen` ignores `pad := .valid` on `conv2d`/`convBn` and always applies ReLU after
  `convBn`~~ **guarded** `ca5eb513`: `MlirCodegen.checkSupported` refuses both; no committed
  artifact comes from this path. Implement when a net first needs either.
- ~~MobileNetV2FullPaperEval states the eval forward per example; the artifact is batched~~
  **done** `29bb72f7`: stem, head and each block kind text-guarded, and the module reduces only
  over `[2, 3]`.
- ~~No forward statement covers the MNv4 evals or the `%do` dropout variants~~ **done**
  `8211e3ca` (MNv4 eval at any size, MNv4/MNv2 `%do` f32 forwards) and `2d361928` (B0's
  `*do*`/`*drop*` forwards, `EfficientNetFullB0Drop`).
- ~~MNv2 stem and head segments of the train-step chain have no VJP tie~~ **done** `8eacd2da`
  (`mnv2HeadCotBlk_eq_vjp`, `mnv2StemCotC_eq_vjp`).
- ~~No parameter-level `pdiv (loss ∘ net)` statement~~ **done for all seven nets**:
  `*_net_lossGrad` (R34 `106a15b1`, R50 `c3bd872b`, MNv2 `72d284b9`, MNv4 `d91f281e`, B0
  `01576a15`, ConvNeXt `8e59f946`, ViT `df2a20d2`) — every parameter gradient node is ∂L/∂θ for
  any `L` with gradient `g` at the logits, with a smoothed-CE corollary per net (R50 also BCE).
- ~~No whole-net `FloatClose` fold; `floatClose_dense`, `_flatConvMixed`, `_depthwise`,
  `_r34_stages` unused, `floatClose_bn` used by no net proof; FloatSubnormalBridge lemmas unused,
  no binary32 `FaithfulFloatModel`~~ **cut** (2026-09-28): the five wraps, `FloatSubnormalBridge.lean`
  and `ResNet34BlockBridge.lean` (`bnStep_close`, orphaned with `floatClose_bn`) are gone; the kit
  (`FloatClose`, `.comp`, `floatClose_flatConv` / pools / skips / `iterate`) stays as the audit's
  per-op instances. The appendix's subnormal parenthetical and "block and stage chains" rewritten.
- Seven SpecVJP `*VerifiedHasVJP` and `mlpHasVJP` are `HasVJP.canonical` (say nothing about the
  net); `bnIstd_close_at` applied only at V = ε.
- ~~`tests/AuditAxioms.lean` does not print the four `Bf16Fold` theorems~~ **done** `a8d6bb78`.
- Float descent rungs: one example, update in ℝ, only the gradient in float;
  TrainedLinearDescent uses exact exp (`fexp := Real.exp`).
- Certificates: `hp` (every class's smoothed probability strictly inside (0,1) everywhere) is
  strong. ~~The pooled scorecard header's "σ ≤ 2 → 66% test acc" is unchecked prose~~ **dropped**
  (generator + `Scorecard.lean`): no run at the pooled scale ever used cap 2; the only cap sweep
  (`runs/spectral_mlp_phase3.log`, full MLP) has σ ≤ 2 at 96.19% and σ ≤ 1 at 64.38%.
- ~~`trainAdamPacked` has no callers and runs IREE only; PGD attacks and `smoothCertify` call
  iree-compile directly~~ **done**: `trainAdamPacked` removed in `d60b56c8`; PGD and
  `smoothCertify` open their graphs through `mkSession` (`671257d3`), which also fixed their
  loss-slot arity on the `mlp`/`cnn` renders. IREE stays as the differential oracle only.

## 7. Stale facts the process-history sweep surfaced — done 2026-09-27

Fixed against the code: the per-replica-BN statements (cifar8 DP check, CnnRender,
`efficientnet-dp-check`); `TestR34DpShard` (compares θ', not `m`; `resnet34-syncbn-check` exists);
NetsCore's MNv4 note (the 100-epoch pair) and the `evalD0` guard block; Train's val-drain notes;
MNv2's variant list; jax Codegen's one-hot note; lakefile's MNv4 recipe, bracket and strided-1x1
docstrings; the ViT Imagenette header; EfficientNet's BN-group note. Emitted text: planning labels,
dates and glyphs out of the MLIR banners (188 artifacts) and the Python shims (74 files; comment
lines only in both).

Checked and left as written: "Nothing has been trained" on ViT-S/B and ConvNeXt-B (only probes
exist); B0's 262 proof-side parameters (213 tensors + 49 conv biases); "One harness, three nets"
in `TestShardCheck` (three rows); the MIOpen/gfx1100 notes (ROCm path, still a target). Not
touched: the `IREE_BACKEND=rocm` run lines in several `apps/imagenette` headers.
