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

## ▶ Start here (next session)

**State at 2026-10-01.** Everything through `1dcbcb58` is pushed to `origin/main`, and CI is
green on all eight workflows. Work from a branch off `main` (the local `rubric-review` branch is
at the same commit). Landed, in order:
- `5ff18875`: this plan.
- `0069e30d` / `9bebaed8`: WP1 parts 1 and 2A.
- `5c0deb1a`: WP3.
- `3f20d71d`: WP2.
- `84ee2bd1`: WP7.
- `dc9fe1ea`: WP4.
- `994b1eb6`: the Foundations closing section.
- `bee5ab0d`: WP5.
- `2e759820`: WP6 and WP6c (the `comment_numbers.py` lint).
- `1dcbcb58`: the small tier (WP10, WP11, X-cor-3, the `module_refs.py` gate, the
  `audit_only_mentions.py` report).

Parked:
- WP1 part 2B, the MNIST descent rungs. Do it with WP8b/A-pq-1; the design is under WP1.
- The B0/ConvNeXt/ViT combined corollaries (WP2) and the ViT/MNv4 drop-path forwards (WP5),
  for the CPU box.
- N1-gen-1 (ConvNeXt S/B), per the user.

**Next (user, 2026-10-01): WP8f and WP8g, in parallel on separate agents** (§WP8 below; the
detail is in `slice_G.md` and `slice_X.md`).
- **WP8f, renderers.** G-reuse-1/3/4/5 go into RenderKit: the `%loss` block (×9), the wd rank
  test, the vector-LN site (×4) and the sync-BN banner (×5).
  - Gate: `git diff verified_mlir/` empty after the build. Every artifact must be byte-identical.
- **WP8g, program code.**
  - X-reu-1…6: the LE readers ×7, the dead `emitChannelSplitGrad`, the float parsers, BraTS
    scoring, `NetSpec`s in Main files, and "compile if IREE" ×3.
  - X-pla-4/5: the classifier kit copied ×4 and the LM kit ×3.
  - X-bes-5.
  - X-cor-3 is already done (`1dcbcb58`).
  - Gate: `lake build Apps` plus the affected demos' smokes (ask before any GPU run over a
    minute), and `check_target_names.sh`.
- The two touch disjoint files. Re-check each finding against HEAD first: the audit ran at
  `55ad3a5a`.
- Standing rules and the standard gate are under "How to farm this". The gate now also
  includes `scripts/gates/comment_numbers.py` and `scripts/gates/module_refs.py`.

After WP8f/g: WP8a (the ParamGrad cluster, about −1k lines), then WP1 part 2B together with WP8b.

Also owed, from the user:
- Triage the `audit_only_mentions.py` list (158 at first run; some are capstones that want a
  yaml or book citation rather than deletion).
- Eventually, re-run the B0 350-epoch ImageNet pair on the i/16 drop-path config. The book notes
  the gap.

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

**Status 2026-09-30: part 1 done for the ResNet stem.**
- **What landed:**
  - `MaxPool3s2SmoothOrDead` is the new op-level predicate.
  - `StemPoolSmoothAt` keeps its name and argument and gets the weaker body: every window smooth,
    or entirely zero.
  - `stemReluPoolLayer` replaces `stemPoolLayer`. It certifies conv-BN-ReLU and the pool as one
    layer, via the germ lemma `maxPool3s2Flat_relu_eventuallyEq` (near a smooth-or-dead
    pre-activation, `pool ∘ relu` is the argmax gather) and `maxPool3s2FlatBackB_eq_reindex` (the
    render's scatter is the gather's adjoint).
  - Unchanged: the forward and the backward graph. Every pinned name is kept.
  - Also dropped:
    - The positivity binders `0 < oc`, `0 < h`, `0 < w` on the stem declarations.
    - `0 < q` on the whole R50 chain (`r50NetLayer` → `resnet50ForwardBFullHasVJPAt` →
      `r50_net_lossGrad` → `r50InputGradB_eq_r34B_full_vjp`). It fed only those binders.
  - Also fixed: the R34 whole-net VJP book block, which omitted the pool condition.
- **Measured** by `scripts/probes/stem_pool_smooth_probe.py`, on the R50-A3 epoch-100 checkpoint,
  four 128-image val batches at 160 px, float64:
  - 13.58% of stem windows are dead. The old predicate rejected those, so P-C-1 was far worse
    than the auditor's estimate; the new one allows them.
  - 0.89% are positive ties, in 4 of 4 batches, and **every one** is two cells reading identical
    7×7×3 input patches (flat image regions). No pre-ReLU value is exactly 0.
- **What that means:**
  - *Input VJP / input-grad ties:* at such a batch the net has no derivative in the image, so
    those theorems' failure there is correct. Their hypothesis cannot be met on these batches.
  - *Loss gradient in θ (`*_net_lossGrad`):* the tied cells are the same function of the stem's
    weights, so the loss IS differentiable in θ. The capstones still don't reach a real step,
    because `StemPoolSmoothAt` is stated on activations. The R34/R50 ParamGrad module docs now say
    so, and name the probe.
- **Part 2A done (ResNet loss gradients).**
  - The pool condition is now stated on the parameters:
    - `MaxPool3s2SmoothUpTo T` allows ties only between T-related positions.
    - `StemPoolTwinAt` is its per-example version, used at `StemConvTwin` (two cells read
      identical zero-padded 7×7×3 patches, stated as equal conv output for every kernel).
    - `BatchSeal.bnBatchLA_bcell_eq_of_eq` shows BN keeps such cells equal.
  - The proof works in θ, not in activations:
    - `stemPoolRelu_param_eventuallyEq`: along any parameter family keeping twins equal, the
      pooled ReLU is the fixed argmax gather near θ₀.
    - `pdiv_congr_of_eventuallyEq` moves each stem node's `pdiv` onto the gather model.
    - The gather model's VJP (`gatherReluHasVJPAt`) needs no pool hypothesis and is the render's
      scatter.
  - `r34_net_lossGrad`, `r50_net_lossGrad` and their corollaries take `R34LossSmoothAtB` /
    `R50LossSmoothAtB`: the same relu clauses with the twin pool clause, implied by the
    input-VJP bundles.
  - The probe's loss-gradient stem clauses **hold** on every real batch tried: 160 px ×4 and
    224 px ×2. The 224 px batches have 13.87% dead windows and 1.37% twin ties, all identical
    patches. The input-VJP clauses still fail there, correctly.
  - **Not yet checked on data:** the block bundles' relu clauses (`≠ 0` at BN outputs and residual
    sums: generically true, not measured) and R50's zero-init bn3 γ at step 0. That needs a
    whole-net float64 forward.
- **Part 2B (next).** One notion serves both remaining cases: cells equal *as functions of the
  moving parameter* on a neighbourhood. This covers:
  - the ResNet stem's identical-patch ties, for the θ-gradients: restate the stem-parameter pull-back
    as a germ in θ, not an activation-level `HasGradAt`;
  - A-corr-1(b), the MNIST descent rungs' constant-patch windows.

  Check the render's scatter first: it picks one of the tied cells, and routing to either gives
  the same θ-gradient, because identical patches make the conv weight-grad and BN γ/β terms
  agree. That needs a lemma.
- **Part 2B design (MNIST descent rungs; not started).**
  - **Twins.** A conv2 output pair is twinned when their conv2 input patches (3×3×c of `x₁`) are
    identical, so they are equal at every `W₂, b₂`. For the conv1 rungs the pair needs identical
    6×6 patches of `x₀`, the two-layer receptive field. That is what a constant background patch
    gives, and it is the 99.5% failure.
  - **Predicate.** `MaxPool2MarginQUpTo δ T`: each window is either all negative (`hm2`'s margin
    then keeps it dead along the step), or its max beats every non-twin cell by `2δ`.
  - **The obstacle.** `SgdDescent/Cnn.lean` gets the loss gradient at *every point of the step
    segment* through the pool's activation-space VJP (`smooth_of_close` →
    `maxPoolFlat_differentiableAt`, `poolBack_close`). With twins that VJP does not exist, so
    part 2A's germ-at-θ₀ trick is not enough.
  - **The route:**
    1. Prove `L θ = L_gather θ` on the WHOLE segment. Twins are equal everywhere; strict margins
       hold on the segment by the existing closeness lemmas; dead windows stay dead.
    2. Run the descent argument on `L_gather`, the chain with the pool replaced by the fixed
       gather: linear, norm ≤ 1, each input read by at most one output for disjoint 2×2
       windows, so the backward bounds carry over.
    3. Transfer: the endpoints and the gradient at θ₀ agree.
  - **Cost.** Step 2 re-plumbs the pool-specific drift and backward bounds through a
    6,283-line file. Do it together with A-pq-1 (`Conv{1,2}Slot.sgd_descends`), which rewrites the
    same proofs; one slot-level lemma over a generic "pool-or-gather" stage is the natural shape.
    Size L.
  - **Acceptance.** A concrete satisfiable MNIST instance through a generator, plus a probe like
    the stem's on the MNIST test set.
- **Deliberately not done:** A-corr-1(a) for the 2×2 pool. `cnnHasVJPAt` / `mnistCnnNoBnHasVJPAt`
  state `HasVJPAt` existence plus its `.correct` field, which a canonical witness satisfies at
  every point (the 09-24 "`_correct` says nothing" item, WP6). Weakening their `h_mp` changes
  nothing checkable; the MNIST work with content is part 2.

### WP2 — Loss-gradient capstones say what the book says · M–L · 1 agent, after WP1
Findings: **P-C-2** (conclude `HasGradAt` in θ, not a `pdiv` equation that is junk off
differentiability), **P-C-3** (`vitNetB = batchMap vitForwardKV`, `cnxNetB` batched bridge),
**P-API-1** (`HasGradAt` projections / structure), **P-API-2** (one shared cotangent chain per net
for the step tie and the loss gradient; today "the same cotangent" is a textual coincidence),
**P-G-1** (`r34_net_lossGrad` any-L + `_smoothedCE`, like the other six; also fix
`api_docs_followups.md:109`).
- Pins: the 14 capstones keep their names, and their statements change.
- Add yaml rows plus comparator challenges for `*_net_lossGrad` (a gap: none exist).

**Status 2026-09-30: done except P-API-2 for three nets (committed `3f20d71d`).**
- **P-API-1:** `HasGradAt` is a `structure` with fields `differentiableAt` / `pdiv_eq`, plus
  `hasGradAt_iff`, `.congr_left` and `.congr_of_eventuallyEq`. The foundation files use the
  field names.
- **P-C-2:** `HasGradAt.param` / `param_batchMap` / `param_batchMap_through` conclude
  `HasGradAt` in θ; the `pdiv_param*` equation forms are gone.
  - Every `GradNodeB.*_eq_pdiv` node lemma is now `*_hasGradAt`, concluding
    `HasGradAt F θ (den node)`.
  - Every `*LossTiedB` clause in the seven ParamGrad files is `HasGradAt …`. Two transformer
    passes did this; the scripts are not kept, because the diff is the record.
  - The capstones keep their names, and the book's `def:hasgradat` says the differentiability is
    part of the claim.
- **P-G-1:** `R34NetLossTiedB` (def), `r34_net_lossGrad` (any `L`, `hL`) and
  `r34_net_lossGrad_smoothedCE`, as R50 does. The book block and AuditAxioms are updated.
- **P-C-3:** `cnxNetB_eq_convNextForwardTCh` and `vitNetB_eq_vitForwardKV` (with
  `fwdO_eq_blockVFlat`). Both are cited in the module docs, the defs' docstrings, the book blocks
  and AuditAxioms.
- **yaml + comparator:** seven `*_net_lossGrad` rows, plus DECLS/MODULES in
  `gen_comparator_tier.py`.
- **P-API-2 (user: combined corollary):** `r34/r50/mnv2/mnv4_net_tied_lossGrad` state the tie and
  the loss gradient per slot at ONE chain of lets. They are generated by
  `rubric_review/wp2_combo_gen.py` from the tie theorem and the `*NetLossTiedB` def.
- **Parked: `enet/cnx/vit_net_tied_lossGrad`.** These ties spell the activations as their own lets
  (`a16`, `ib18`); the loss side spells them `enetPreB16 N w x` / `cnxPreB17 N ε w x`. The final
  `exact` must prove the two spellings equal through the whole cotangent chain:
  - Measured on this box: every variant ran past 30 min (heartbeats off), or hit max recursion
    depth at the default limit, even with the tie's names substituted into the loss conjuncts
    (`pre_re`).
  - Next attempt:
    1. First prove one-line bridges `cnxPreB17 N ε w x = <tie spelling>` (from the `*_apply`
       lemmas), and `rw`/`simp only` them into `hl` before `obtain`, so the final `exact` compares
       identical terms.
    2. Or add a per-slot heartbeat cap to find which slots are expensive (the `_tmp_cnx3`
       experiment, stopped unfinished).
  - Run on the dedicated CPU box; ViT needs `groups` (the tie's final-LN and head bundles pair
    with one loss bundle).

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

**Status 2026-09-30: done (committed `5c0deb1a`).**
- `Verified/Attack.lean`: `specNormW` / `specNormGet` / `specNormConvTapSum` are gone.
  - `denseLip` / `convLip` return a `LipPair`:
    - `bound` is `denseE_lipschitzL2_gram2`'s Schatten-8 `B = (Σ H²)^{1/8}` over the output-side
      Gram, times `roundingSlack`;
    - `est` is the power-iteration value, from the same Gram.
  - The conv factor is a tap-sum of per-tap Schatten-8 bounds. The tap-sum step is stated in the
    docstring and has no Lean theorem.
  - Every certified print uses `bound`, with `est` printed beside it. `certProduct` is shared by
    the two conv drivers.
  - Checked against numpy SVD on random matrices: bound = the Schatten-8 norm × 1.000001, and est
    ≤ σ₁.
- `projectSpectral`:
  - convs are capped on `convLip.bound`, the certificate's own quantity;
  - denses stay on the matrix-free estimate, since a full Gram every few steps costs too much. So
    a projected dense layer's certified `Lᵢ` sits above `c`. The spectral docstrings (Attack and
    the three apps) say so, and the false `L ≤ cᵏ` is gone.
- X-cor-2: `oneHotBatchPad` reads the label through `F32.readLabel`. The unused `oneHotBatch` is
  deleted.
- Prose:
  - `LipschitzCert/Basic.lean`'s module doc and `clm_lipschitzL2`, and `DenseEuclid`'s
    `denseE_lipschitzL2` docstring.
  - Attack's module doc, which now states the remaining gap: the certificate is for the
    real-arithmetic net at the float-printed margin, and logit rounding is not budgeted.
  - The CNN / CIFAR / MLP app headers. The stale `IREE_BACKEND=rocm` run lines in the two
    spectral apps are gone too.
- C-doc-1/3: measured counts are out of `formalization.yaml`, the CertsHeavy lakefile docstring,
  `certs-heavy.yml` (header + step summary) and `tests/AuditAxiomsHeavy.lean`; each names its
  source instead. The CertsHeavy description now lists its real roots.
- C-doc-2 + C-nam-1: in both smoothing generators the header says "side-condition checks, not
  certificates of the driver nets", and the banners say "with a kernel-checked tail bound".
  - Renames: `smoothCp<Net>_certified` → `smoothCp<Net>_tail_le` and `smoothDec<Net>_certified` →
    `smoothDec<Net>_radius_le`.
  - Regenerated from the archived CSVs; AuditAxioms repointed.
- Quoted numbers: the book and READMEs never quote the PGD demos' certified accuracies. Only the
  tracked pre-fix logs do (`runs/pgd_*_phase3.log`, `runs/spectral_*_phase3.log`), and they are
  left as records of the old code.
- Follow-up (not done): a general Schatten-2^m theorem, the iterated Gram, would tighten the bound
  toward σ₁, which matters most for the single-layer linear demo.

### WP4 — Codegen: no silent wrong function, `den` = text · M · 1 agent
Findings: **X-cor-1** ✔ (`unsupported` must refuse every layer with no emitter; +25 lines),
**G-corr-1** ✔ (`biasGradB : SHlo (N*n) → SHlo n` with `denseBiasGradB`'s `den`; text unchanged;
drop the external Σ from 4 capstones; root file), **G-corr-4** (pad formulas outside the faithful
kernel domain → `// MALFORMED`), **G-corr-2** + **G-reuse-2** (a computable loss-cotangent graph,
printed via one RenderKit helper, so the T3 start is the printed text), **G-corr-3** (ViT /
ConvNeXt-T forward text guards; fixes book content.tex:12046).
- Gate: byte-identical artifacts throughout.

**Status 2026-09-30: done (staged); every artifact byte-identical.**
- **X-cor-1.** `MlirCodegen.noEmitter` names the constructors neither walk lowers, and
  `unsupported` refuses a spec containing one. There are 24, not 22: `layerNorm` and
  `convNextStem` are lowered only by the JAX emitter, and this path would also have skipped
  them. `#guard`s sit beside it. The module header drops its stale "28 of 50" counts. The book's
  Bestiary preamble and three Bestiary printouts say "no emitter". (The UNet printout is stale
  for another reason and is left for WP6.)
- **G-corr-4.** `Pretty.kernelOutsideDomain`, checked in `serializeToks`, replaces an
  XLA-SAME stride-2 op outside odd `k ≥ 3`, or a stride-4 op outside `k ≥ 3`, with a
  `// MALFORMED` line and a `%MALFORMED` result. Re-checked: at `k ∈ {1, 2}` the pad floors at 0
  and reads the wrong phase. Even `k` on the symmetric stride-1 arms changes the output shape
  (loud), so it is not guarded. The domains are stated in `Basic`'s constructor comments and
  `flatConvStride2Xla`'s docstring.
- **G-corr-1.** `biasGradB : SHlo (N*n) → SHlo n`, with `denseBiasGradB`'s Σ as its `den`.
  `headBGradB_den` (name kept) is stated at the summed node. The ViT/ConvNeXt ParamGrad clauses
  and head step ties drop the external Σ, and the fold files drop the carve-out.
- **G-corr-2 + G-reuse-2.** The three loss-cotangent graphs are defined as a computable tail
  (`smoothedCotTail`, `bceCotTail`) applied to their softmax/sigmoid head. `RenderKit.smoothedCotB`
  and `bceCotB` print that tail, and the seven renderers call them after printing the head, whose
  name the `%loss` report needs. The printed cotangent is therefore `pretty` of the capstones'
  graph. MNv2's α = 0 path (sub, divide) is neither graph and stays as two calls.
- **G-corr-3, ConvNeXt half.** The per-example emitters are public as `cnxFwdBlock`,
  `cnxFwdDown`, `cnxLnFwdSite` and `cnxHeadLnFwdSite`. `convNextFwdGraphTCh`'s blocks, downsamples
  and stem use the artifact's parameter names (strings only, so `den` is untouched).
  `FwdGraphTextTies` guards block, downsample, stem and head. A probe confirmed the guards are not
  vacuous: the block text is 6.6 kB, and a wrong prefix fails.
- **G-corr-3, ViT half: the proposed fix does not hold.** `vitBlockGraphMHV` shares non-leaf
  subterms (LN1 → Q/K/V; Q/K/V → every head; the first residual → LN2 and the second residual).
  `pretty` shares nothing, so the graph's text repeats them, and the render's slice order differs
  from the graph's postorder. A text tie needs a sharing printer or a reordered render (artifact
  change). `FwdGraphTextTies`' coverage paragraph, `vBlockFwd`'s docstring and the book's
  `thm:vitFwdGraphKMHV_faithful` now say so; the book no longer says "the rendered ViT is the
  tower above".

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

**Status 2026-09-30: four done (staged), three parked.**
- **N2-gen-1 done.** `vitTinyInputGrad_eq_vitTiny_vjp` binds `{nCls}`.
- **N1-corr-1 done.** The stem-bias clause of `cnxStemChTied`, `cnxStemChTiedGB` and the
  ParamGrad stem is at the real `flatConvStride4 Wst b' x`; `xstem` is gone from every capstone.
  The node carries an input it never reads (stated at zero). The bridge is
  `GradNodeB.pdiv_flatConvStride4_bias_eq_conv2d` (ParamGradNodes), which ConvNeXtParamGrad now
  reuses. The "same modelling as mnv2/r34" sentence is gone; the comparator tier is regenerated.
- **N2-corr-2 done.** B0's inference graphs use the render's names, and two actual TEXT bugs are
  fixed:
  - `.swishF` and `.addV` at the batched index printed `tensor<2x802816>` and `tensor<2x150528>`;
    they are now `.batchOp .swish` / `.addVB`, as the train graphs spell them.
  - `FwdGraphTextTies` guards stem, the four block shapes and head at `.eval`.
- **N1-corr-2 done, at the headline arms (user's choice).** The finding named `cifar8wb`, but the
  book's chapter-4 runs train `cifar8w{,_bn}_*`: `cifar8{Bn,Adam}TrainStepFaithfulV` at
  `opt := some _`, which emit per-example UN-FUSED `*Grad` nodes.
  - New `GradNode` section in SgdNodes: `conv{W,B}Grad_den`, `bn{Gamma,Beta}Grad_den`,
    `dense{W,B}Grad_den`, the `*GradTied` clauses and their `_holds`.
  - New `Cifar8StepTieG` / `Cifar8BnStepTieG` (`cifar8{,Bn}_train_step_tiedG`) state the fused
    ties' chain node for node, with lakefile roots, audit pins and book `thm:cifar8_step_tieG`.
  - The fused ties' docs say they tie no trained artifact.
- **Parked: N2-scope-1** (ViT and MNv4 drop-path forwards).
  - ViT has no batched typed forward graph, and its drop sites are batched ops.
  - MNv4 needs drop-carrying copies of five group graphs in the file whose `simp only` spelling
    already dies in the kernel.
  - Both are size L and belong on the CPU box. MNv4FullB's "not this graph" row now lists
    `mnv4in_acc{,dp}8x128wxdropdowd01bf16`.
- **Parked: N1-gen-2** (small-net loss gradients). `cnnHasVJPAt` / `cifarCnn8HasVJPAt` take the
  activation-level pool hypothesis real MNIST fails on 99.5% of images. Stating capstones on it
  would recreate WP1's defect, so do it with WP1 part 2B.
- **Parked: N1-gen-1** (ConvNeXt S/B), per the user.
- Noted, not done: `EfficientNetSyncB`'s train-mode sync graphs still use the old SE names
  (`zWa…`), and nothing text-checks them.
- Local only: `leanblueprint web` fails in plasTeX's imager (a missing temp PNG) after writing
  `lean_decls`; the PDF build is fine.

### Small tier (user, 2026-09-30) — done, staged
- **WP11:**
  - `StableHLO/Lex.lean` deleted (G-scope-1), along with the 18 stale `tests/*.lean` smokes and
    render scripts. X-sco-1 was wrong about `TestConvNeXtBlock`: it is compile-only, with no
    gradcheck.
  - The five `#guard` scripts now run in certs.yml as a "#guard scripts" step.
  - `LossKind.floatTargetMse` throws, and the book's DDPM line no longer names it.
  - certs.yml's Bestiary guard also triggers on `Bestiary/**` and `LeanMlir/Spec.lean`.
  - Deleted as dead: `r34StemB_continuous`, both `sealX_continuous`, `addConstHasVJPAt_backward`,
    `bnBatchLA_apply_perm`, the six A-reuse-1 `.correct` restatements, the A-scope-1 MNIST-CNN
    float-budget cluster, and `dotSgd_step_close` / `sumSgd_step_close`, which only that cluster
    used.
- **WP10:**
  - The pool positivity binders are gone everywhere they cascaded (the whole-net CNN VJPs, the
    SgdDescent rungs, the comparator arch tier, the CNN seal and witness generators).
    `bnMean_shard`'s and `mask_scalar_close`'s unused binders are gone too.
  - IR.lean's four fixed-shape conv bridges are three general ones.
  - F-gen-2's unused binders are dropped.
  - SyncBf16's base-2 lemmas are over `z : ℤ` and any base `b > 1`, and `rndP_mul_four` is
    dropped. The F-pl-3 move is skipped because it would put them in a root file.
- **X-cor-3:** `roundE4M3` rounds half to even, as numpy and OCP E4M3 do. 11 `#guard`s are
  checked against the oracle, and the old code fails two of them.
- **TestDropPathRamp:** its two formulas match every renderer and reference. What was wrong was
  the claim about timm: `efficientnet_b0` / `tf_efficientnet_b0` both ramp `i/16`, so `i/15` is
  only the JAX reference's default. The comment is fixed in the test, NetsCore, the jax driver and
  the book's printed config.
- **`ibp_conv_scorecard.py`:** fixed and given a `--check`; it reproduces the committed files, and
  its counts are in `runs/2026-09-30-cert-scorecards/`.
- **Gates:**
  - #5 is `scripts/gates/module_refs.py`, in targets.yml. Every `…/X.lean` path and every dotted
    module name in a Lean file must resolve. It found and fixed five stale citations plus three
    `Proofs.Lamb` → `Optim.Lamb`.
  - #2 is `scripts/gates/audit_only_mentions.py`, a report only. It lists declarations reachable
    from no root but AuditAxioms: 158 at first run, several of them capstone-shaped (e.g.
    `r34/r50/mnv2/mnv4_net_tied_lossGrad`), so the gap may be yaml/book citation rather than dead
    code. Triage that list before deleting from it.
  - #7 is deferred. `allReduceMeanF`'s `ds.prod = n` holds by construction at the only printer
    entry, `prettyAllReduceMean`. `pretty B` vs `N` needs every batched `skel` descriptor to carry
    `N`: a Pretty.lean change plus a lowerer-wide regen.
- **Banner:** `convnext_train_step.mlir` (SGD) now names its GAP-backward block. That is its
  only artifact change.
- **Open from this pass:**
  - The book's B0 ImageNet section prints `dropPathOverN := true` (i/16), but the 350-epoch
    runs it reports (JAX 77.15, verified 76.878) predate that flag and trained at i/15.
    `runs/2026-09-12-enet-verified-350ep` logs keeps 0.973 → 0.813. The book now says so (user: re-run on the i/16 config eventually).
  - The MNv4 reference ramps i/20 over its 21 UIB blocks. timm's `mobilenetv4_conv_medium` ramps
    i/23 over 23. Imagenette B0's i/15 is not timm's either.
  - `apps/imagenette/MainMobilenetV4VerifiedAdam.lean` called the net Conv-S and said "no
    composed-backward theorem yet"; both are fixed.

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

**Status 2026-09-30: done (staged).** Every WP6 finding was re-checked against HEAD. N1-doc-2
was already fixed by WP5; the rest are done.
- **HasVJP `_correct` as content.**
  - `vitTinyHasVJP_correct` is now the ViT-Tiny instance of `vitInputGradK_correct`: the
    hand-written chain equals the `pdiv` contraction. It moved to `ViTWholeBackCertifiedTie`,
    since keeping it in ViTDepthK would be an import cycle; yaml, AuditAxioms and the comparator
    tier are repointed.
  - The B0/MNv2/ViT-KV `_correct` docstrings take MNv4's form.
  - In the book, `def:hasvjp` now says a witness exists for every `f`, so its content is what `B`
    is. The R34/R50/MNv2/MNv4/B0/ViT-KV whole-net blocks now say "the backward composed from the
    per-layer backwards (`…HasVJPAt`)", as the MLP/CNN blocks already did.
  - `mlpVerifiedHasVJP` (the canonical witness) is deleted; the book prints `mlpVerifiedHasVJPAt`.
- **Scope sentences.**
  - R34/R50 tie files and the four ParamGrad files say the nodes are f32 on one replica and that
    the bf16 renders behind the book's numbers are outside the statement.
  - R50 anchors on `resnet50in160_lambaccdp8x64wxclipbce`, not the retired 76.66% run.
  - B0/R34/R50 "one replica" paragraphs point at the sync-BN tie; every DP artifact of those nets
    was checked to be sync-BN.
  - MNv2 sync names its four f32 DP artifacts; `mnv4_net_syncTiedB` names `mnv4in_adamdp64`.
- **Book drift.**
  - `thm:resnet50_whole_back` states its hypotheses and the opaque-block fold, and cites
    `resnet50ForwardBFull_eq_slots`.
  - The parse-back paragraph says `roundtrip` is about tokens, and the "single printer" sentence
    adds the declared hand-written lines.
  - SpecVJP says nine Imagenette-and-smaller specs are tied; yaml and Proofs/README agree.
- **Artifacts (approved comment-only re-renders).**
  - G-doc-2 was 34 files, not 31: 18 single-replica plus 16 DP ConvNeXt train steps carry
    `%dgapf`. The audit's 31 counted 13 EfficientNet files, which have no GAP-backward block.
  - `trainStepHandNote` gains a `hand` argument, and ConvNeXt's banners (both branches) name the
    GAP-backward block.
  - G-doc-3's 8 bf16 DP artifacts cite `Foundation/DataParallel/SyncBf16.lean`.
  - Diff: 41 files, comment lines only.
- **N2-doc-4.** A literal `#guard` over all 21 `(ic, oc, expand, preDWk, postDWk, h, stride2)`
  tuples, re-extracted from timm 1.0.28. It is not vacuous: a flipped kernel fails it.
- **Bestiary.**
  - AlphaZero chess is now Silver 2018's net: 19 blocks and a conv policy head (golden count
    69.4M → 23.3M).
  - The GPT decoders and CLIP text encoders set `causalMask`.
  - AlexNet/YOLOv1 say their FC fan-in is pinned to the paper's.
  - The README's new-entry recipe calls `summarize` (what the golden test parses).
  - `tests/bestiary_timm_report.md` is regenerated, and `verify_bestiary_timm.py`'s `tests`
    package import is fixed.
- **WP6c.**
  - `scripts/gates/comment_numbers.py` scans Lean comments and docstrings, the yaml's comments
    and the workflows' step summaries for N/100, %, codebase counts, `file:line` and hashes.
  - Allowed occurrences sit in `comment_numbers_allow.tsv`, 93 rows, each with a reason (spec,
    statement, citation, example).
  - The sweep went from 401 hits to 0.
  - It is wired into `targets.yml`, not `proofs.yml`: proofs.yml is path-filtered, and this scan
    covers apps/, demos/, jax/, the yaml and the workflows.
  - Generated scorecards were fixed in their generators and regenerated, comment lines only; the
    counts now say which generator prints them.
- **Follow-ups decided by the user (2026-09-30), done:**
  - *Numbers with no home.* The certificate scorecard generators were re-run into
    `runs/2026-09-30-cert-scorecards/`; its README tables every dataset count, and every emitted
    file came back byte-identical. The Certificates README points there.
    `historical/comment_measurements.md`, generated from `git diff bee5ab0d`, keeps verbatim every
    other comment line the sweep removed.
  - *Lint extended.* `comment_numbers.py` also matches timings (ms, s/epoch, min, h), memory
    sizes (GiB, MB), decimal speedups and metrics (mIoU / IoU / Dice / mAP / top-1 / acc). A
    second four-agent sweep cleared the new hits; the allowlist has 242 rows.
  - *B0 / ViT banners.* Both AdamW train-step banners (single-replica and DP) use
    `trainStepHandNote`. ViT names its `%ximg` input reshape (`vitInputReshapeNote`). Every B0 and
    ViT AdamW train step was re-rendered, comment lines only.
  - *Stale Imagenette numbers in comments.* The user's call: they were out of date; the
    comments point at the logs, and nothing more is needed.
- **Open (found by WP6, not done):**
  - The lakefile benchTable cited `runs/<net>_xla_80ep_jul29.log` / `vit_xla_80ep_jul30.log`,
    which were never tracked.
  - `historical/lipschitz_cert_rationalize.py` still prints the old accuracy lines into its
    snippets. Harmless, because `Instance.lean` is hand-merged.

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

**Status 2026-09-30: done (committed `84ee2bd1`).**
- Every Lean file in `citations.md`'s "Cite in" column (80 files, including the root files) ends
  its module docstring with `## References`: one bullet per work, author–year, title, `<link>`.
  - A file named for several works lists them all.
  - Derived files carry the link themselves rather than pointing at the defining file. The
    column already picked the files, and a pointer would be a module path that
    docstring-checkrefs cannot resolve.
  - The inline author-year credits already in these docstrings stay.
- Non-Lean sites:
  - `README.md` gains an Acknowledgements section (X-att-1): Lean 4, Mathlib, StableHLO,
    XLA/PJRT, IREE, JAX, timm, torchvision, leanblueprint, plasTeX, doc-gen4.
  - `TRUST.md` links XLA/PJRT, IREE and timm.
  - `formalization.yaml` `sources` gains Gowal (IBP), CROWN, CROWN-IBP and Clopper–Pearson; the
    authors were checked against the arXiv API.
  - alpha-zero-general is linked in `demos/README.md` and the book's tic-tac-toe paragraph
    (X-att-2). PUCT is credited to AlphaGo Zero in `ffi/f32_helpers.c`.
- Not done:
  - Delattre 2023 (optional). Only if the Gram-iteration idea came from there.
  - The Chan 1983 companion. The 1982 COMPSTAT paper is the one cited.
  - The book's missing inline credits (the "—" cells in the Book column). That is a later pass.
- **TinyStories demo deleted (user, 2026-09-30).** Gone: `demos/MainTinyStories.lean`, its
  lakefile target, the download/preprocess scripts, `scripts/demos/tinystories_decode.py`,
  `blueprint/src/figures/tinystories/` (the book never referenced it), the demos/README section,
  and the README and home-page mentions. Nothing else becomes an orphan: RoPE, flash attention,
  the id-gather embedding and the token-stream helpers all serve TinyGPT.

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
  - **Status 2026-10-01: done (staged); every finding re-checked at `9eed9240`; 0 `verified_mlir/`
    changes.** Net about −885 lines over the ParamGrad files and their kit. No pinned statement
    changed.
    - P-R-1: `ParamGradNodes` gains `hasGradAt_relu6`, `hasGradAt_addConst` and
      `hasGradAt_constAdd` (the skip held fixed on either side). Every R34/R50/MNv2 `*_lossTiedB`
      proof is now the MNv4 chain of one-line pull-backs. The `r34IdG*` / `r34DownG*` /
      `r50{Id,Proj,Down}G*` / `mnv2{Stem,NoExp,Body,SBody,Head}G*` defs and their `_hasGradAt`
      theorems are deleted, as are the four `*ProjOut` / `*BodyOut` abbrevs. MNv2 keeps its
      XLA-`SAME` strided depthwise stage as a local `hasGradAt_depthwiseStridedXla`, because
      `dStridedXlaInB` lives in `MobileNetV2StepTieB`. The head goes through dense and GAP by
      `HasGradAt.comp` inline. The R34 stem keeps `r34StemGNg` / `r34StemGCg`: the argmax-gather
      model is not a kit stage, and AuditAxioms pins `r34StemGCg_hasGradAt`.
    - P-R-2, reshaped: `CertLayer.hasGradAt_comp` and `HasGradAt.residual_body` moved from MNv4
      into `ParamGradNodes` (which now imports `CertifiedChain`). The residual-body step at the
      MNv2 and B0 skip blocks and at the R34/R50 identity blocks is `hasGradAt_addConst`.
      Dropped: the audit's R34/R50/MNv2 `*_hasGradAt_comp` sites are not `CertLayer` re-inlines.
      They compose the bundle-level `HasVJPAt`s (`r34IdBHasVJPAt`, …), not `L.vjp`, so the
      `CertLayer` lemma would need a `Subsingleton` bridge and save nothing. No `den (L.graph …)`
      variant was added, because nothing would use it. With WP9's P-PL-1, MNv4's `hasGradAt_cast`
      moved to `ParamGradNodes` too.
    - P-R-3: the ConvNeXt pair is now `rowLNVecFlat_{gamma,beta}_differentiable` in
      `Architectures/ChannelLN.lean`. ViT's `rowVecLN_*` are deleted, and its six call sites use
      the moved pair by defeq.
    - P-R-4, partly: `rowSumLoss_pdiv_smul`, `rowIdx_cast` and `rowB_row_eq_logitRow` in
      `SmoothedBatchLoss` replace the `hℓ` + `pdiv_const_smul` block (three copies) and the
      `hidx` / `hrow` blocks (two copies). Dropped: redefining `smoothedBatchLoss` through
      `smoothedBatchLossDiv`, since that changes a pinned def's body.
    - P-PQ-1: one private `vit_node_lossTied` (`param_batchMap_through`, `congr_left` along the
      slot's factoring, `of_eq` at the node) closes each of the sixteen bullets. The `*TiedB_holds`
      ascriptions that spelled `blkSaves` / `c*` with their 16–17 arguments are gone, because the
      node's denotation fixes them. No `BlockParamsV` abbrevs were needed (they would have gone in
      `ViTStepTie.lean`). The proof drops from about 233 to about 180 lines (each bullet still
      names its pre/per/post). Left alone: `cnx_block_lossTiedGB`, which already rewrites `Φ`
      up front and needs no `congr_left`, so the lemma buys nothing there.
    - P-PQ-2: one-line comments at the `change`/`show` sites (`ParamGrad` ×2, `ParamGradNodes` ×2,
      `MlpBias` ×2; the third `MlpBias` site already had one). The `bnBatchLA_apply_perm` half is
      moot, because the small tier deleted it.
    - P-S-1: already done in `1dcbcb58`.
    - Gates: `lake build Certs CertsHeavy LeanMlir Apps Reference TestSupport`, comparator
      `--check`, name lint, import audit, blueprint checkdecls (lean_decls regenerated from
      content.tex, unchanged) + `blueprint_uses.py --check`, target names, comment numbers, module
      refs, `regen_verified_mlir.sh check` with an empty `git diff verified_mlir/`.
      `audit_only_mentions.py`: no WP8a declaration on its list. Red only from other in-flight WPs:
      AuditAxioms (`floatClose_residual`) and docstring-checkrefs (`WindowMax.lean`).
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
  - **Status 2026-10-01: done (staged), except two lines outside Codegen; every artifact
    byte-identical** (`regen_verified_mlir.sh proofs` then `check`, empty `git diff verified_mlir/`).
    Codegen net about −200 lines.
    - G-reuse-1, reshaped: the audit's nine copies are three blocks. Seven batched renders spell
      the same smoothed CE, now `reportSmoothedCeLoss`; R50's BCE is `reportBceLoss`. The
      per-example renders print plain CE, not smoothed CE: `reportCeLossOfLogits` (MLP, MNIST CNN;
      MlpRender now imports RenderKit) and `reportCeLossOfSm` (the four CIFAR copies; one keeps
      its own banner text through the `banner` argument).
    - G-reuse-3: `r34WdDecays` → `rankWdDecays`, `r34WdName` → `wdNameBy (decays := rankWdDecays)`;
      `cnxWdDecays` is deleted (its rationale is a `--` comment above `cnxWdCounts`), so ConvNeXt,
      `cnxWdCounts`, its `#guard`s and `tests/TestWdExcludeTie.lean`'s mask use `rankWdDecays`;
      ViT calls `wdNameBy … vitWdDecays`.
    - G-reuse-4: `vecLnSite` / `vecLnSiteB` replace `vlnFwd`, `vlnFwdB` and `headLnFwdSiteB`;
      `cnxHeadLnFwdSite` stays as the ε-fixed ConvNeXt site (FwdGraphTextTies and the ConvNeXt tie
      docstrings cite it), and the channel-LN sites `cnxLnFwdSite` / `lnFwdSiteB` reuse the same
      three ops between their transposes. The backward pairs are not one shape (ViT's `vlnBack`
      carries the SGD/Adam update), so they stay.
    - G-reuse-5: `syncBnBanner tieThm fwdThm dir` plus the two bf16 notes, `syncBnBf16WgradNote`
      (R34, R50) and `syncBnBf16TwinsNote` (MNv2, MNv4, B0). The drop-path and accumulation notes
      differ per net and stay with their callers.
- **8g Program code**:
  - **X-reu-1…6** (LE readers ×7, a dead `emitChannelSplitGrad`, float parsers, BraTS scoring,
    `NetSpec`s in Main files, "compile if IREE" ×3).
  - **X-pla-4/5** (the classifier kit copied ×4, the LM kit ×3; −450).
  - ~~X-cor-3~~ done in `1dcbcb58`.
  - **X-bes-5**.
  - **Status 2026-10-01: done (staged) except the LM kit and X-bes-5; 0 `verified_mlir/`
    changes.** Program code net about −400 lines, two new import-free modules
    (`LeanMlir/CliArgs.lean`, `LeanMlir/SmallClassifier.lean`). Every finding was re-checked at
    `1dcbcb58`.
    - X-reu-1: `LEBytes` gains `readU32LE` / `readU64LE` (byte offset) and `pushF64LE`;
      `F32.readLabel` and `TTT.readU64` are their record-index forms. The copies in the shim
      preamble and row counts (`Verified/Train`), Bigram, BratsEval, AlphaZero, GradcheckHelpers'
      `.npy` reader and `.bin` writer, NQS's `pushF64` and RsBands' packed index are gone. NQS's
      `pushU64` stays: it is over `UInt64`, and `pushU64LE` takes `Nat`, which boxes a
      configuration with the top bit set. Diffusion2d's `floatsToBytes` is already one `pushF32LE`
      fold.
    - X-reu-2: the helper is deleted, not called. The `.unetUp` arm needs other SSA names, and
      the two `dwConvAttrBlock*` go too.
    - X-reu-3 (with X-pla-2's module): `CliArgs.parseFloat?` / `parseFloat` / `kv` / `parseArg` /
      `natArg` / `floatArg` replace `ViTGradcheck.parseFloat?` and the Pong, AlphaZero, NQS,
      RsBands and GwDetect parsers, the four `parseArg`s and the three `kv`/`natArg` lambda sets.
      A token now rounds as the Lean literal does. The old hand parsers could be 1 ulp off where
      they multiplied by `10^e` (`3e-4`); every value the demos' docs show parses identically.
      X-nam-1's namespace rename stays with WP9.
    - X-reu-4: `SegMetrics.regionCounts` / `regionDice` in `Train.lean` serve the trainer's val
      line and `brats-eval`, and `DatasetKind.segRegions` exposes the region table.
      `ReferenceNets.bratsNetOf` resolves `net=` / `ctx=` for `brats-predict` and `brats-eval`.
      The new helpers agree with the deleted ones on a set of hand confusion matrices (a scratch
      `#guard`).
    - X-reu-5: `ReferenceNets.cifar8wOf`, `tinyDdpmUnet`, `r34FpnDet` and `r50FpnDet` (which now
      carries the R50 bootstrap and 2×2-pool rationale); PlantLeaf is `{ resnet34 with … }`. A
      scratch file checked `reprStr` old = new on 23 instantiations (every name, tower and size
      the demos use), so every `generate*` output is unchanged.
    - X-reu-6: `NetSpec.compileArtifact` (iree-compile on IREE, a no-op on XLA) replaces the DDPM
      trainer's and sampler's copies and GradCAM's shell-out. GradCAM uses `F32.argmaxN`, BratsPredict
      and the DDPM sampler use `Cam.writePPM`, and NQS uses `FloatFmt.fmt` and `Ddpm.piF`. The four
      classifier `fmt`s are not `FloatFmt.fmt`; they moved into `SmallClassifier` unchanged.
    - X-pla-4: `SmallClassifier` (`fmt`, `xs`, `permutation`, `gather`, `scoreBatches`,
      `scoreSet`) serves the four classifier demos. PlantLeaf scores through `scoreBatches` with its
      own last-image-padded gather, and RsBands' augmenting gather is `gatherChips`. The detectors'
      four `inferDump`s share `NetSpec.evalLogits`. The per-knob env readers and their checks
      differ per demo, so they stay. **LM kit dropped:** TinyStories is gone (`84ee2bd1`), and the
      only copies left are `loadVocab` / `reverseVocab`, twice. The two `sampleToken`s differ on
      purpose (top-k/top-p), and `readFloats` takes different arguments in each.
    - X-pla-5, reshaped: `F32.unpackAdam` replaces the three-slice unpack at all 10 sites. No
      `AdamState` / `adamStep`: the sites call four different FFI step variants, and they read
      `p` alone between steps. The two `lean_ttt_*` externs moved from the AlphaZero Main into
      `TicTacToe.lean`; the six `lean_mcts_*` stay in the demo.
    - **X-bes-5 parked (needs the user):** the token-embedding half changes six entries' modelled
      parameter counts. That means the golden table, each entry's docstring and Notes printout,
      and the book's bestiary prose. Each model also needs its own call (BERT's segment table,
      LLaMA's RoPE `posEmb := false`, CLIP's 77 positions). The `resNetBody` half would replace
      the per-entry layer lists the book prints, and DeepLab's last stage is stride 1 on purpose.
    - Gates: `lake build Certs CertsHeavy LeanMlir Reference TestSupport Apps`, `lake build` of
      the 31 touched exes, `check_target_names.sh`, `comment_numbers.py`, `module_refs.py`,
      `import_audit.py implied` (dropped E4M3Quant's now-implied `LEBytes` import),
      `docstring-checkrefs`, `name_lint.py`, empty `git diff verified_mlir/`. CPU smokes:
      `label-check`, `argmax-check`, `test-dataset-record-sizes`, `ttt-env`, `fpn-train-emit`,
      `seg-loss-probe` (`g=` / `w=` through `CliArgs`), `sgd-render-tie`'s bad-lr guard, and
      `brats-eval` / `brats-predict`'s spec resolution. The GPU smokes are owed (user launches).

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
   bundle on a real batch at trained weights. It would have caught WP1 twice. The first one exists:
   `scripts/probes/stem_pool_smooth_probe.py` (the ResNet stem). Still to cover: the MNIST CNN
   pool, and every relu clause of the block bundles.
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
