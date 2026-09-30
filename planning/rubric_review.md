# rubric_review.md — TauCeti rubrics over the whole tree, as farmable work packages

Run 2026-09-30 on branch `rubric-review` @ `55ad3a5a`, read-only. The ten TauCeti review rubrics
(`TauCetiProject/TauCetiReview@dcc918ab`, vendored in `rubric_review/rubrics/`) were applied as a
whole-tree audit. The adaptation is `rubric_review/rubric_common.md`:
- "roadmap" = yaml / book / AuditAxioms / lib roots;
- the repo's own vacuity shapes;
- what CI already gates;
- every earlier audit, so their landed or rejected items are not re-reported.

Eight slice auditors ran it. Their reports (`rubric_review/slice_*.md`) are the detail behind every
ID below: file:line, the fix, evidence, pin cost and size. The top items were re-checked by hand
before this plan was written (marked ✔).

**167 findings.** They show no false theorem. What they do show:

- **Hypotheses the shipped data can never satisfy (block).**
  - The max-pool smoothness predicates, stated on post-ReLU activations or on equal cells, fail on
    real inputs. The ResNet-34/50 loss-gradient capstones and whole-net VJPs never apply to a real
    ImageNet batch.
  - No CNN descent rung has a satisfiable MNIST instance: 99.5% of test images contain a
    constant-patch window. ✔ reproduced.
  - This is the 09-24 correctness audit's #1 item. It was fixed in prose only and has since spread
    to the new capstones.
- **One unsound "certified" number.**
  - The `mnist-*-pgd` demos print a "certified-robust acc" from a power-iteration L, which is a
    *lower* bound on ‖W‖₂.
  - The docs tie that print to `lipschitz_margin_certified_radius`. ✔
- **One silent wrong function in the reference codegen.** `MlirCodegen.unsupported` lets through
  the 22 layer kinds that have no emitter. They emit a comment and no params, so a shape-preserving
  block trains as the identity. ✔
- **One IR op whose `den` ≠ its text.** `biasGradB` has an identity `den` but prints a batch reduce.
  It is a documented carve-out, but the batch Σ sits outside the AST in 4 capstones. ✔
- **Attribution.** This is the first pass ever run on it, and credit is almost entirely absent from
  the Lean tree. 34 of 266 proof files cite any paper, and no optimizer except LAMB credits its
  source.
- **The rest** is doc overclaim residue, reuse and factoring in the unaudited `*ParamGrad` cluster
  (~−1k lines), placement, and naming.

## Decisions (user, 2026-09-30)

1. **WP3: make the demo's radius a real upper bound**, not a relabel.
2. **Comment-only re-renders of committed artifacts are approved** (G-doc-2's 31 ConvNeXt banners,
   G-doc-3's 8 bf16 DP paths).
3. **Citations: list first, web link only for the first pass.** The list is
   `rubric_review/citations.md`, one verified link per work plus the files that should cite it.
   WP7 applies it; a richer format (bib keys, titles) can come later.
4. **Delete the orphans and everything only they use:** A-scope-1's float-budget cluster, the
   X-sco-1 stale test scripts, `StableHLO/Lex.lean` (G-scope-1), the dead declarations in P-S-1 /
   N1-scope-1 / A-reuse-1 / X-reu-2, and their AuditAxioms lines, lakefile roots and README rows.
5. **Renames are approved:** G-name-1, N1-name-1, P-N-1, N2-name-1, C-nam-1, C-nam-2.
6. **No numbers that nothing re-checks in comments or docstrings** (new rule; WP6c). Code comments
   do not carry:
   - measured results (accuracy %, "N/100 certified", timings, speedups);
   - counts of the codebase ("185 roots", "~250 lines", "87 call sites");
   - `file:line` or commit hashes in prose.

   Those rot silently: the lakefile's counts, the 69/100 residue and the stale paths are all this
   class. A comment names the source instead: the run directory, the theorem (which
   docstring-checkrefs resolves), or the generator. Numbers that ARE the specification (224, ε, the
   0.1 smoothing, widths) stay. So does a count the adjacent statement proves ("8 witnesses" beside
   a theorem stating `length = 8`). Measured numbers live in run READMEs and the book, whose ledgers
   are audited. At HEAD: 72 "N/100", 126 `xx.x%`, 23 codebase counts, 8 `file:line` in Lean
   comments. The commit-hash hits need a manual pass; the census regex also matches hex constants.

## How to farm this

Each work package (WP) below is sized to be handed to one agent. Its brief is: "read
`planning/rubric_review.md` §WPn and the cited `slice_*.md` entries; re-check each finding against
the code before acting (audit claims were wrong often enough before; see api_design_audit.md §0);
implement; gate; stage; stop."

Standing rules (from memory and earlier threads, all binding):
- **No compatibility layer.** A rename or delete repoints every consumer: AuditAxioms, comparator
  `DECLS`/`MODULES`, `\lean{}` + `blueprint_uses.py --check`, yaml, docstrings.
- **Stage then STOP.** No commit without the user's word, and no push without a separate one.
  Fast-forward only.
- The AlphaZero session committed (`086beab0`, the 5×5 board). The *in-flight* findings are now
  ordinary findings: re-check each against `086beab0`. One addition, **X-cor-4**:
  `TicTacToe.solverEntry` (`opaque … (arena : @& ByteArray) … : UInt8`, a pure extern) mutates the
  borrowed arena as a memo cache. That is outside Lean's FFI contract, since the compiler may share
  or reuse a borrowed value. Make it an `IO`/`ST` extern, or have the C side own the cache behind
  an opaque handle.
- Root files (`Foundation/Tensor.lean`, `Codegen/StableHLO/Basic.lean`, `Architectures/LayerNorm`,
  `BatchNorm`) rebuild most of `Certs`. Batch their edits, one WP at a time.

**Standard gate:**
- `lake build Certs CertsHeavy LeanMlir Apps Reference TestSupport` (bare `lake build` skips Certs).
- `lake env lean tests/AuditAxioms.lean`.
- `gen_comparator_tier.py --check`.
- `name_lint.py`.
- `import_audit.py implied`.
- `lake exe docstring-checkrefs`.
- `blueprint-checkdecls` then `blueprint_uses.py --check`.
- `check_target_names.sh`.
- `scripts/regen_verified_mlir.sh check` with an empty `git diff verified_mlir/`.
- `scripts/book/book_xrefs.py` for book edits.

## Verdict matrix

| slice | corr | reuse | scope | attr | api | gen | place | name | doc | pq | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P ParamGrad (never audited) | **block** 3 | 4 | 1 | 1 | 2 | 2 | 2 | 1 | 2 | 2 | 20 |
| A Architectures + Training | **block** 1 | 1 | 1 | 2 | ✓ | 2 | 3 | ✓ | 2 | 3 | 15 |
| F Foundation + Float | ✓ | 3 | ✓ | 3 | 3 | 4 | 3 | ✓ | 5 | 1 | 22 |
| C Certificates | 1 | block¹ 2 | ✓ | 1 | ✓ | ✓ | 3 | 2 | 4 | 1 | 14 |
| G Codegen (proof tier) | 4 | 5 | 1 | 1 | 1 | ✓ | 2 | 1 | 3 | 1 | 19 |
| N1 ResNet / Small / ConvNeXt | 3 | 2 | 1 | 1 | ✓ | 2 | 1 | 1 | 3 | 1 | 15 |
| N2 MobileNet / B0 / ViT | 2 | 4 | 1 | 3 | ✓ | 1 | 2 | 1 | 10 | 4 | 28 |
| X program code, Bestiary, cross-cutting | 6 | 6 | 3 | 6 | ✓ | ✓ | 6 | 1 | 6 | n/a | 34 |

✓ = approve. ¹ The rubric blocks any declaration an existing one directly replaces. The one here is
`CrownBound.mlp2_apply`, a one-line fix.

## Tier 1 — statements that don't reach real data (do first)

### WP1 — Max-pool smoothness at the data the nets see · L · 1 agent, owns `R34SmoothAtB`
Findings: **P-C-1** ✔, **A-corr-1** ✔ (a)+(b), P-D-1 (the interim prose), plus the book/yaml
"irreducible boundary" sentences (content.tex:17990–17996, yaml:287–289).
- **Two root causes, one theme.**
  - *Post-ReLU pool.* `MaxPool{2,3s2}Smooth` is asked of `relu z`, so a window of dead cells ties
    at 0. The statement is true there, and the hypothesis is just too strong.
  - *Equal cells.* A constant input patch makes all four conv2 cells equal for every weight, so no
    δ-margin exists.
- **Fix, part 1 (M).** Add `maxPool3s2Flat_relu_comm` / the 2×2 peer. Add a pre-ReLU predicate:
  either a unique argmax or all < 0. Prove `pool ∘ relu` has the emitted `reluMask ∘ poolBack` as
  its VJP there. Then restate `R34SmoothAtB` / `R50SmoothAtB` `stem`+`pool` fields and
  `cnnHasVJPAt`'s `h_mp`.
- **Fix, part 2 (L).** Weaken the descent rungs' `hmq` to allow cells that are equal *as functions
  of the moving parameter* along the step segment, discharged from patch equality.
- **Acceptance.** Build a concrete satisfiable instance: a generator over real MNIST at the trained
  weights (the `TrainedLinearDescent` pattern), plus a JAX probe that evaluates the R34 predicate on
  one real batch.
- **Also.** P-C-1 notes R50's zero-init bn3 γ breaks `hout` at step 0. Scope-sentence it.
- **Blocks.** WP2's R34/R50 edits and WP8a (same files). Do the P-D-1 prose caveat immediately
  if WP1 won't start soon.

### WP2 — Loss-gradient capstones say what the book says · M–L · 1 agent, after WP1
Findings: **P-C-2** (conclude `HasGradAt` in θ, not a `pdiv` equation that is junk off
differentiability), **P-C-3** (`vitNetB = batchMap vitForwardKV`, `cnxNetB` batched bridge),
**P-API-1** (`HasGradAt` projections / structure), **P-API-2** (one shared cotangent chain per net
for the step tie and the loss gradient; today "the same cotangent" is a textual coincidence),
**P-G-1** (`r34_net_lossGrad` any-L + `_smoothedCE`, like the other six; also fix
`api_docs_followups.md:109`).
- Pins: the 14 capstones keep their names, and their statements change.
- Add yaml rows plus comparator challenges for `*_net_lossGrad` (a gap: none exist).

### WP3 — "Certified" means proved · S · 1 agent
Findings: **C-cor-1** ✔. Relabel the demo's print as *estimated*, drop "sound" from
`specNormConvTapSum`, and fix the Basic.lean / Attack.lean prose. Or make the number sound:
Frobenius/Gram upper bound, or power iteration × proven slack.
- Also **C-doc-1/2/3** (the 69/100, 92/100 and "99/100 / 80/100 certified" residue in yaml,
  lakefile, certs-heavy.yml summary and AuditAxiomsHeavy; the wrong CertsHeavy description).
- Also **C-nam-1** (`smoothCp*_certified` / `smoothDec*_certified` prove only tail arithmetic;
  rename, through the generators).
- Also **X-cor-2** (Attack.lean `label % 256`).
- **Decided: make it sound.** Replace `specNormW` / `specNormConvTapSum`'s power-iteration value
  with a proven upper bound: the Frobenius/Gram route `DenseEuclid` already proves, or a
  Schatten-2k bound `tr((WᵀW)^k)^{1/2k}`, computed in the host.
  - Prefer the same bound the certificate tier proves, so the printed radius IS
    `lipschitz_margin_certified_radius` at an `L` whose `LipschitzL2` has a theorem.
  - For the conv tap-sum, each tap's norm must be an upper bound (Frobenius per tap is the cheap
    sound choice).
  - Keep the power-iteration value only as a printed "(estimate: …)" beside it, for the tightness
    gap. The tighter per-layer bound can be a follow-up.
  - Gate: the `mnist-*-pgd` smokes. The certified accuracy may drop; say so in the demo README and
    the book.

### WP4 — Codegen: no silent wrong function, `den` = text · M · 1 agent
Findings: **X-cor-1** ✔ (`unsupported` must refuse every layer with no emitter; +25 lines),
**G-corr-1** ✔ (`biasGradB : SHlo (N*n) → SHlo n` with `denseBiasGradB`'s `den`; text unchanged;
drop the external Σ from 4 capstones; root file), **G-corr-4** (pad formulas outside the faithful
kernel domain → `// MALFORMED`), **G-corr-2** + **G-reuse-2** (a computable loss-cotangent graph,
printed via one RenderKit helper, so the T3 start is the printed text), **G-corr-3** (ViT /
ConvNeXt-T forward text guards; fixes book content.tex:12046).
- Gate: byte-identical artifacts throughout.

### WP5 — Per-net capstone reach · M each · parallelisable by net
- **N1-corr-1** ✔ ConvNeXt stem bias is tied on a free `xstem`. State it at `flatConvStride4`.
  The docstring's "same modelling as mnv2/r34" is false.
- **N1-corr-2**: the chapter-4 step ties name a trainer-less artifact. Either write
  `Cifar8StepTieGB` for the `cifar8w*` arms, or fix the prose.
- **N2-corr-2**: B0 is the last net with no eval `FwdGraphTextTies`.
- **N2-scope-1**: no drop-path forward statement for ViT / MNv4. B0's `EfficientNetFullB0Drop` is
  the template.
- **N1-gen-1**: ConvNeXt ties are at T only, though S and B ship. Size L.
- **N1-gen-2**: the five small nets have no param-level loss gradient.
- **N2-gen-1**: a ViT per-example tie is still at 10 classes.

## Tier 2 — prose and credit (cheap, broad, one agent each)

### WP6 — Doc-honesty residue · S×~40 · 1 agent (book + docstrings)
The book and docstrings still claim more than the statements (grouped; the IDs carry the wording):
- **HasVJP `_correct` presented as content:** **N2-corr-1** ✔ (`vitTinyHasVJP_correct`
  "production capstone"; B0/MNv2 "Public correctness theorem") and **F-doc-5** (book
  `mlpVerifiedHasVJP` "the VJP is proved"). This is the 09-24 item #2, still live in these sites.
- **Scope sentences:**
  - **P-D-2** (R34/R50/MNv2/MNv4 loss-gradient files: f32 nodes, one replica).
  - **N1-corr-3** / **N1-doc-3** (the book's ResNet runs are bf16; R50 docstrings cite the retired
    76.66% run).
  - **N2-doc-1** (B0 DP artifacts are sync-BN, not per-replica: the per-replica-identity trap).
  - **N2-doc-5/6/9**.
- **Book statement drift:**
  - **N1-doc-1** (R50 input-grad theorem over opaque blocks, with hypotheses omitted).
  - **G-doc-1** ("lexes and parses back", and `Lex.lean` builds no lexer).
  - **F-doc-4** (SpecVJP ties 9 of 33 specs, not "each"; yaml:485 is false).
- **Stale references and facts:**
  - **F-doc-1** ✔ (`FloatSubnormalBridge`, deleted, in 3 places incl. TRUST.md).
  - **F-doc-2** ✔ (`rndP` rounds ties up, not to even).
  - **F-doc-3**, **G-doc-3** (8 artifacts cite a moved path), **N2-doc-2/3/4/7/8/10**,
    **N1-doc-2**, **A-doc-1/2**, **X-doc-1…5** (lakefile counts all stale; certs.yml bumped the
    wrong way), **C-doc-4**.
- **Artifact banner:** **G-doc-2** (31 ConvNeXt artifacts say "gradients + optimizer are
  pretty(AST node)" over a hand-written GAP-backward block).
- **Bestiary prose vs spec:** **X-bes-1** (AlphaZero 40 blocks + a 21.8M "identity" layer),
  **X-bes-2** (~12 param claims off), **X-bes-3** (flatten fan-in; `validate` stops at `.flatten`),
  **X-bes-4**.
- **Approved:** G-doc-2 and G-doc-3 re-render comment lines in committed artifacts (31 + 8
  files). Take G-doc-2 option (a) (the banner names the carve-out). Option (b) is a WP4 follow-up.
- **WP6c: the numbers sweep (decision 6).** Remove or repoint every unchecked number in Lean
  comments and docstrings, the lakefile, `formalization.yaml` comments and workflow step summaries.
  Then add a lint (`scripts/gates/comment_numbers.py`) with an allowlist for spec constants and
  statement-backed counts, and wire it into proofs.yml. This subsumes X-doc-1 and C-doc-1, and
  generated banners are fixed in their generators.

### WP7 — Attribution pass · S · 1 agent; applies `rubric_review/citations.md`
No earlier audit covered this. **First pass (decided): a web link only.** Add a `## References`
line in the module docstring of the file that *defines* the method: author–year plus the verified
link from `citations.md`. Derived files point at the defining file rather than repeating the link.
A richer format (titles, book cite keys) is a later pass.
- **Findings:**
  - Optimizers and regularisers (**A-attr-1**): Adam/AdamW, RMSProp, momentum/Nesterov, clipping,
    stochastic depth/dropout, the descent lemma.
  - Ops (**A-attr-2**): ResNet, BN, LN, GELU, Swish, layer-scale, attention, SE, depthwise,
    ConvNeXt.
  - Chan variance, Muon/Shampoo, IBP (**F-at-1/2/3**).
  - IBP, CROWN-IBP (**C-att-1**).
  - He, Liu, RSB, LAMB (**N1-attr-1**).
  - Sandler, Qin, Tan & Le, Hu, Huang, Dosovitskiy, Touvron, Vaswani (**N2-attr-1/2/3**, **G-attr-1**).
  - Label smoothing, Szegedy 2016 (**P-A-1**).
  - Program side:
    - An acknowledgements section for timm / JAX / OpenXLA / IREE / Mathlib / doc-gen4
      (**X-att-1**).
    - alpha-zero-general author and URL (**X-att-2**, *in-flight*).
    - `Verified/NetsCore.lean` (**X-att-3**).
    - DDPM / Boltzmann / NQS etc. (**X-att-4/5**).
    - Bestiary full citations (**X-att-6**).
- `formalization.yaml` references gain IBP/CROWN rows.
- Batch the root-file docstrings (BatchNorm, LayerNorm) with WP9's root edits.

## Tier 3 — reuse, factoring, placement (farm in parallel by file cluster)

### WP8 — Proof factoring
Farm the sub-packages to separate agents. They touch disjoint files, except where noted.
- **8a ParamGrad cluster** (after WP1/WP2), ~−1k lines:
  - **P-R-1** (R34/R50/MNv2 re-derive `ParamGradNodes`' stage pull-backs: 57 defs, 53 theorems,
    none pinned; −700…−1000).
  - **P-R-2** (`CertLayer.hasGradAt_comp` / `HasGradAt.residual_body` into ParamGradNodes).
  - **P-R-3**, **P-R-4**.
  - **P-PQ-1** (`vit_block_lossTiedGB` 233 lines → one local lemma).
  - **P-PQ-2**, **P-S-1**.
- **8b CNN descent**, ~−750:
  - **A-pq-1** (`Conv{1,2}Slot.sgd_descends`; deletes 16 margin wrappers; subsumes audit_v2 §8's
    owner item).
  - **A-pq-2** (`cnn_conv1_cot_close`).
  - **A-scope-1** (the ~330-line MNIST float-budget cluster has no roadmap consumer: delete?).
  - Coordinate with WP1 part 2: same file.
- **8c MaxPool unification**, ~−400 (**A-gen-2**, L): one window-max over an index family, with
  2×2 and 3×3/s2 as instances. ⚠ Graph ties `rfl`-match `maxPoolBack` (the IR-spelled trap). Keep
  the instance names as `abbrev`s and check the T2 ties. Best done *with* WP1 part 1.
- **8d Nets**: **N1-reuse-1/2**, **N2-reuse-1…4**, **N2-pq-1…4**, **N1-pq-1** (28 undocumented
  `show`/`change`; add `_def` lemmas).
- **8e Foundation/Float/Certificates**:
  - **F-re-1/2/3**, **F-pq-1** (the `den_convBackBatched_eq_cInB` peers).
  - **C-reuse-1** ✔ (typechecked), **C-reuse-2** (−40), **C-pq-1**.
  - **A-reuse-1** (six `.correct` restatements with only an AA line).
  - **A-pq-3** (13 `show … decimate` → `HasVJP.decimate`; check the `rfl` ties).
- **8f Renderers**: **G-reuse-1/3/4/5** (the `%loss` block ×9, wd rank test, vector-LN site ×4,
  sync-BN banner ×5 into RenderKit; byte-identical).
- **8g Program code**:
  - **X-reu-1…6** (LE readers ×7, a dead `emitChannelSplitGrad`, float parsers, BraTS scoring,
    `NetSpec`s in Main files, "compile if IREE" ×3).
  - **X-pla-4/5** (the classifier kit copied ×4, the LM kit ×3; −450).
  - **X-cor-3** (`roundE4M3` ties away from zero vs the numpy oracle's ties-to-even).
  - **X-bes-5**.

### WP9 — Placement, API and naming
- **Kit homes:**
  - **P-PL-1/2** (general helpers in per-net ParamGrad files).
  - **P-G-2** (`colSlabApplyH` into Attention; the special case derived from it).
  - **N1-place-1** (`r34BFullHasVJPAt` → `OpaquePrefix`).
  - **N2-place-1/2**, **F-pl-1/2/3**.
  - **A-place-2** (general conv/float lemmas out of `SgdDescent/Cnn.lean`).
  - **C-plc-1** (the float-composition engine lives only in a generator template).
  - **C-plc-2/3** (Gaussian material → upstream drafts; the drafts point at a stale path).
- **File splits (carried):** **A-place-1** (`SgdDescent/Cnn.lean` ℝ vs float), **A-place-3**
  (`Activations.lean`, audit_v2 §2.3, never created).
- **Codegen:**
  - **G-place-1** + **G-api-1** (one `OptRecipe` plus one `optOne` in RenderKit; MNv4 stops
    importing ResNet34RenderB).
  - **G-place-2** (`EfficientNetRender/PC*` → `Nets/EfficientNet/`).
- **Program:** **X-pla-1/2/3**, **X-nam-1**.
- **API:** **F-api-1** (`FloatModel.u_nonneg` field → theorem, 112 uses unchanged), **F-api-2/3**.
- **Naming** (approved 2026-09-30):
  - **G-name-1**: 33 String renderers named `*Faithful{V,B}`.
  - **N1-name-1**: the `*PoC` namespaces (carried from cleanup_backlog §8).
  - **P-N-1**, **N2-name-1**.
  - **C-nam-2**: the `LipschitzCertDemo` namespace (carried from api_design_audit §6.5; ~181 AA
    lines plus 9 generators; expensive).

### WP10 — Generality
- **A-gen-1**: unused `_hc _hh _hw` on the pool derivatives, forwarded through 87 call sites;
  `bnMean_shard`; `mask_scalar_close`.
- **F-gen-1**: four 2-channel 4×4 conv bridges in IR.lean are instances of the general lemma right
  above them.
- **F-gen-2/3**.
- **F-gen-4**: Muon geometry is square-only. Size L; ask.

### WP11 — Scope / dead surface · S
- **G-scope-1**: delete `StableHLO/Lex.lean`; its target was abandoned. Pairs with G-doc-1.
- **X-sco-1**: 21 `tests/*.lean` scripts run by nothing. Delete the 18 stale ones and wire or
  keep the `#guard` gates.
- **X-sco-2**.
- **X-sco-3**: certs.yml's Bestiary guard doesn't trigger on `Bestiary/**` or `Spec.lean`.
- **N1-scope-1**, plus the dead declarations in P-S-1 / N1 (`sealX_continuous`,
  `r34StemB_continuous`) and A-reuse-1.

## New gates the audit asks for (humans' call)

1. **Hypothesis satisfiability on shipped data.** A probe that evaluates each capstone's smoothness
   bundle on a real batch at trained weights. It would have caught WP1 twice.
2. **"Only AuditAxioms mentions it" report.** This is how A-scope-1 and A-reuse-1 hid.
3. **A tie ↔ loss-gradient cotangent-chain equality check.** Moot if WP2's shared chain lands.
4. **Prose "N/100 certified" lint** against the aggregate theorems' lengths (WP3).
5. **Module-path references in docstrings.** docstring-checkrefs resolves declaration names only,
   which is how F-doc-1 and G-doc-3 survived.
6. **CI coverage:**
   - `IRPrint.lean` (the book prints its output) and the SDPFull files, which no job builds.
   - Only 4 of ~11 certificate generators are `--check`ed.
   - Upstream drafts ↔ `Foundation/UpstreamDraft.lean` sync.
7. **`pretty B` vs the batched terms' `N`**, and `allReduceMeanF`'s `ds.prod = n`: both are
   mechanical `#guard`s.
8. **Generate the lakefile's hand-kept counts or delete them** (X-doc-1).

## Suggested order

1. **Tier 1: WP1 → WP2 on one thread.** In parallel, run WP3, WP4 and WP5 on separate agents; they
   touch disjoint files.
2. **Tier 2: WP6 and WP7** with one agent each, anytime. Book edits serialise with each other.
3. **Tier 3: the WP8 sub-packages** in parallel by cluster, with WP9 and WP10 folded into whichever
   sub-package owns the file. WP8a waits on WP1/WP2.

All decisions are taken (see "Decisions" at the top) except WP10's F-gen-4 (Muon rectangular,
size L), which stays parked until someone needs a rectangular Muon statement.

## Rubric fit notes

- **Roadmap.** TauCeti's "roadmap" has no direct analogue here. Mapping it to yaml / book /
  AuditAxioms / lib roots made an abandoned file (`Lex.lean`) formally "on roadmap". G applied the
  "satellite whose target never got closer" test instead, and future runs should too.
- **api-design mostly approves.** That is because the 09-26 api-design audit *was* this rubric. The
  remaining api items are in the post-09-26 ParamGrad code.
- **Blocks in reuse.** The rubric's "block on any direct replacement" fires on one-line private
  duplicates (`CrownBound.mlp2_apply`). Read "block" there as "delete it", not as severity.
