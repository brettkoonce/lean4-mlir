# api_docs_followups.md — what the API-docs pass left for later

The API-docs pass (branch `api-docs-pass`, eight `docs(api):` commits 4bbb86f3..39039d89) fixed
every docstring doc-gen4 renders against `doc_audit/find_C..J_*.md`; per-site detail is in
`doc_audit/ledger_{C..J}.md`. It changed docstrings and comments only: no statements, no emitted
text, no markdown, no LaTeX. This file lists what it deliberately left, grouped by what the fix
needs. `doc_honesty_pass.md` is still the parent plan (its §0 protocol applies to every item).

## ▶ Start here

**Status 2026-09-27:** §1–§3 done on branch `docs-followups` (`0a7eeea7`, then the banner
pass: 174 artifacts say which lines are hand-written and name the right optimizer). Left in §1:
nothing (the D4 marker/date/citation sweep of 255 files is done, comments only; §7 lists the
stale facts it surfaced). Left in §3: the `179 of 180` ConvNeXt
test note, which is a dated measurement and consistent with itself.

Pick a section. §1–§3 are small and mechanical; §4 is the book pass (the plan's §1, minus what
this pass already did on the landing page); §5 are Lean changes decided in D1; §6 are leads for a
correctness pass, not doc work.

## 1. Comments that still say something false (not rendered, but read by maintainers)

- `sdpa_back_{Q,K,V}` spellings (now `sdpaBack{Q,K,V}`): ViTMhsaBackCertifiedTie.lean:91,
  ViTMultiHeadChain.lean:149; grep the rest of `LeanMlir/` for `sdpa_back_`.
- MaxPool3s2.lean:115 — "the r34 forward float chain (`floatClose_r34_stages` …)"; no such chain
  exists (`floatClose_r34_stages` has no users and iterates one block at one width).
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

## 5. Prose waiting on the D1 Lean changes (left verbatim on purpose)

Each row's landing rewrites these sites to the new statement.

| D1 row | Sites left verbatim |
|---|---|
| small-CNN dense heads | **landed** (Lean + the four fold files + CnnRender + blueprint thm:cnn_fold / cifar_fold / cifar8_step_tie / cifar8bn_step_tie). Still open: SgdDescentCnn "weights and biases via the MLP rungs" needs bias descent (MLP rungs omit bias columns) |
| nCls for B0 | EfficientNetStepTieG (module + head), EfficientNetFullWholeBackCertifiedTie title, B0 Sync files, VerifiedNetsCore:909 claim ceiling, LeanMlir.lean "the ImageNet head its artifacts run" |
| nCls for ViT (+ `ty [10]`) | **landed**: both ViT-Tiny capstones bind `nCls`, `ty [nClasses]`, blueprint thm:vitTinyHasVJP_correct / thm:vit_whole_back and the §9 prose. SpecVJP's `Vec 10` is the Imagenette spec's own head, kept |
| MaxPool live-cell predicate | **landed for 2×2**: CNN.lean `MaxPool2Smooth` docstring, SgdDescentCnn module pool margin, TrainedCnnWitness `h_mp` bullet (generator), StableHLO/Basic `maxPoolBack` comments, blueprint thm:cnn_sgd_descends. Open: `MaxPool3s2Smooth` (all-pairs), and "measure-zero" for post-ReLU pooling (all-zero windows) |
| `*CotIn_eq_vjp` MNv4 | MobileNetV4StepTieB:25 (paragraph kept whole, markers included), `mnv4_net_tiedB` first paragraph; MobileNetV4WholeBackCertifiedTieB header (the "Tier T6" paragraph was deleted, not reworded — write the new one at landing) |
| `*CotIn_eq_vjp` ConvNeXt, ViT | `cnx_net_tied_certified`, `cnx_net_tiedGB`, `vit_net_tied_certified`, `vit_net_tiedGB` "certified ∂L" wording; ConvNeXtRender/RenderB "certified loss-descent step" |

Also open: slice I dropped eight unchecked "3-axiom-clean" claims; the audit is green
(1610/1610), so they can come back if wanted — cite `tests/AuditAxioms.lean` as a repo link.

## 6. Leads for a correctness pass (no false theorem found)

- `vit-fwd-b-tie` compares `vitFwdRenderB "vit_fwd"` against `vit_fwd.mlir`, which
  `vitFwdRenderB` itself writes — the gate only catches non-determinism.
  (`convnext-fwd-b-tie` still compares against the other chain.)
- `MlirCodegen` ignores `pad := .valid` on `conv2d`/`convBn` and always applies ReLU after
  `convBn` (`convBnAct` is read only by the JAX emitter).
- MobileNetV2FullPaperEval states the eval forward per example; `mobilenetv2_fwd_eval.mlir` is
  batched (32×150528). Check how "diff line for line" is text-tied.
- No forward statement covers: `mnv4{,in}_fwd_eval` (frozen stats), `mnv4in_fwd_eval_s256`, the
  `%do` dropout variants (`mnv4in_emaacc*`, `mobilenetv2in_rmsdp64wxdols0*`).
- MNv2 stem and head segments of the train-step chain (`mnv2HeadCotBlk`, `mnv2StemCotN/C`) have
  no VJP tie; only the four block-level `*CotIn_eq_vjp` exist.
- No parameter-level `pdiv (loss ∘ net)` statement for R34/R50; the capstones reach the certified
  block backwards only through `*CotIn_eq_vjp` under `R34/R50*SmoothAt` + positive ε.
- No whole-net `FloatClose` fold; `floatClose_dense`, `_flatConvMixed`, `_depthwise`,
  `_r34_stages` unused, `floatClose_bn` used by no net proof; FloatSubnormalBridge lemmas unused,
  no binary32 `FaithfulFloatModel`.
- Seven SpecVJP `*VerifiedHasVJP` and `mlpHasVJP` are `HasVJP.canonical` (say nothing about the
  net); `bnIstd_close_at` applied only at V = ε.
- `tests/AuditAxioms.lean` does not print the four `Bf16Fold` theorems.
- Float descent rungs: one example, update in ℝ, only the gradient in float;
  TrainedLinearDescent uses exact exp (`fexp := Real.exp`).
- Certificates: `hp` (every class's smoothed probability strictly inside (0,1) everywhere) is
  strong; the pooled scorecard header's "σ ≤ 2 → 66% test acc" is unchecked prose.
- `trainAdamPacked` has no callers and runs IREE only; PGD attacks and `smoothCertify` call
  iree-compile directly (PJRT is the engine).

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
