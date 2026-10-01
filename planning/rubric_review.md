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

**State at 2026-10-02 (morning).** `origin/main` is at `1dcbcb58` (CI green). Local branch
`wp8fg` carries eleven unpushed commits on top of it (the six below, then the overnight work as
five commits: `bc5dbff9` WP9 codegen, `c4fdca7b` WP9 foundation/float/certificates, `8f4edf6e`
capstones (WP2, WP5 drop, WP9 nets), `90cdc235` pool + descent (WP1-2B, WP8b, stem, small-net loss
gradients, den = text, SgdDescent split), `a93e0701` PoC namespaces + the shared wiring):
- Committed, unpushed: `bf067c5b` WP8f, `9eed9240` WP8g (GPU smokes passed, one card),
  `27d4059a` WP8a, `494be406` WP8c, `ed34d90e` WP8d, `bb2d22a2` WP8e.
- Overnight, by package (each has its status under its own WP; the claims files in the
  session scratchpad map files to agents):
  - WP1 part 2B + WP8b (agent A): twin-tolerant pool margin, slot lemmas, MNIST probe 10000/10000;
    the ResNet-stem selector half (agent J); conv2-bias / conv1 / conv1-bias concrete instances
    (agent K).
  - WP2: `enet/cnx/vit_net_tied_lossGrad` (agent B).
  - WP4 follow-up: `maxPoolBack`'s `den` routes to the first maximal cell, as the printed
    `select_and_scatter` does (agent L).
  - WP5: ViT / MNv4 drop-path forwards (agent C) and the MNv4 drop-row text guard (agent G); the
    six small nets' loss gradients (agent H).
  - WP9: codegen half (agent D: `*Text` renderer names, `OptRecipe`, PC files moved),
    Foundation/Float/Certificates (agent E, incl. `LipschitzCertDemo` → `Robustness`), nets
    placement + renames (agent F), SgdDescent split + Activations.lean + A-pq-2 (agent I), the
    `*PoC` namespaces (agent M).
  - Full gate green on the combined tree; 0 `verified_mlir/` changes.
- Next: fast-forward `main` to `wp8fg` and push (the user's call).

Still open:
- User's calls: N1-reuse-2 (renaming `r34BFullHasVJPAt` changes the pinned capstone's comparator
  text), X-bes-5 (Bestiary parameter counts), N1-gen-1 (ConvNeXt S/B).
- `audit_only_mentions.py` triage: first pass applied (every cite / delete / unpin row); the keep
  and `?` rows to revisit are in planning/rubric_review/audit_only_triage.md.
- Parked with recipes: F-pl-2 (root file `Tensor.lean`), N2-place-1, F-gen-4.
- Owed (small): the backward through the ViT / MNv4 drop sites; H's items 3–5 under WP5;
  `MlirCodegen`'s emitted GELU comment strings name `LayerNorm.lean`; `CnnRender.CifarOpt` is not
  folded into `OptRecipe`.
- Runs: the B0 350-epoch pair on the i/16 drop-path config; plant-leaf and gradcam smokes need
  data / checkpoints not on this box.

## Overnight plan (2026-10-01 → 10-02)

The user asked for WP1 part 2B with WP8b, WP2's and WP5's parked items, and WP9, run as a plan
rather than all at once. Everything stays on branch `wp8fg`, staged and uncommitted; each package
writes its status under its own WP heading, and the morning split into commits goes by the file
claims, as WP8a/c/d/e did. Long elaboration is allowed tonight (the box is otherwise idle); a single
attempt past about two hours is parked with a recipe.

- **Phase 1 (parallel, disjoint files):**
  - **A: WP1 part 2B + WP8b.** The MNIST descent rungs, via the gather route under WP1 (twins,
    `MaxPool2MarginQUpTo`, whole-segment `L = L_gather`, descent on `L_gather`, transfer), with
    A-pq-1's slot-level lemma as the shape, and the acceptance generator and probe. Base: WP8c's
    `WindowSmoothUpTo`. A-pq-2 if it falls out.
  - **B: WP2's parked corollaries.** `enet/cnx/vit_net_tied_lossGrad`, bridges first (the recipe
    under WP2).
  - **C: WP5 N2-scope-1.** ViT and MNv4 drop-path forward statements; B0's
    `EfficientNetFullB0Drop` is the template.
  - **D: WP9, codegen half.** G-name-1 (the `*Faithful{V,B}` String renderers), G-place-1 +
    G-api-1 (one `OptRecipe` / `optOne` in RenderKit), G-place-2 (`EfficientNetRender/PC*` to
    `Nets/EfficientNet/`). Byte-identical artifacts.
- **Phase 2 (after phase 1):**
  - WP5 N1-gen-2, the small nets' loss gradients, which needs A.
  - WP9's proof half: the kit homes (P-PL-2, P-G-2, N1-place-1 is the user's, N2-place-1/2,
    F-pl-1/2/3, A-place-2, C-plc-1/2/3), the file splits (A-place-1, A-place-3), the renames
    (N1-name-1, P-N-1, N2-name-1, C-nam-2 last), and F-api-1/2/3.
- **Not tonight:** N1-reuse-2 and X-bes-5 (the user's calls), N1-gen-1, F-gen-4, landing or
  pushing anything.

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

**Status 2026-10-02 (overnight, agent A): part 2B done for the MNIST rungs (staged); 0
`verified_mlir/` changes.** Every finding re-checked at `bb2d22a2`.
- **Predicate.** `WindowMarginUpTo δ T` (WindowMax, the quantitative `WindowSmoothUpTo`, on the
  PRE-activation: dead = all cells `≤ 0`), its 2×2 abbrev `MaxPool2MarginQUpTo` (ConvIndex),
  `WindowMarginUpTo.mono`, `windowMarginUpTo_of_cert` (one designated cell per window, `T` an
  equivalence) and `MaxPool2MarginQ.to_marginQUpTo{,_flat}` (the old post-ReLU margin implies it).
  Twins: `ConvPatchEq` (identical zero-padded patches; `conv2d_eq_of_convPatchEq`, for every kernel
  AND bias), `ConvPatchEq2` (two-conv receptive fields, `convPatchEq_relu_conv`), `.symm`,
  `.trans`, `.of_zero`.
- **Route, as designed but per point, not per segment.** The gather is `poolGatherFlat σ`
  (`windowGather` in WindowMax; CLM, `pdiv_poolGatherFlat`, `poolGatherFlat_l1_contract`,
  `maxPoolFlat_eq_poolGatherFlat`). `Conv2Slot.maxPool_relu_eventuallyEq_gather`: if at `p` every
  window is strictly negative or `σ`'s cell is strictly above its non-twins, the relu'd pool IS the
  gather on a neighbourhood of `p` (finitely many strict inequalities, twins equal everywhere).
  `Conv2Slot.marginUpTo_seg` gives that strict form at every segment point from the `2ρD` margin,
  `σ` = the base argmax. So `L =ᶠ L_gather` near each segment point, and differentiability and
  `gradAt` transfer pointwise (`EventuallyEq.fderiv_eq`); no separate endpoint step.
- **A-pq-1 (WP8b).** `Conv2Slot.sgd_descends` / `Conv1Slot.sgd_descends`, generic in the parameter
  map `Z`, `ρ`, a Jacobian row `J` (the slot computes the gradient itself through
  `gather_loss_gradAt`; no `hgrad` hypothesis) and `Q`. The drift lemmas take the pool stage `S`
  with an ℓ1-contraction hypothesis; `Conv2Slot.gather_grad_lipschitz` replaces the maxPool
  `loss_grad_lipschitz` (the selector is constant, no argmax freeze). Each of the four real rungs
  is one application. Deleted: the 16 margin wrappers, the four `cnn_*_loss_grad_lipschitz`, the
  four `cnn_*_loss_differentiableAt` (MaxPool2Smooth forms, AuditAxioms-only after the change),
  both `*Slot.loss_grad_lipschitz`, `postrelu{,2}_close_seg` and their 22 AuditAxioms lines.
  Cnn.lean 5934 → 5443 lines including the additions below.
- **Statements changed (names kept):** `cnn_conv2_sgd_descends`, `cnn_conv2_bias_sgd_descends`,
  `cifar8_lastConv_sgd_descends` take `T`, `hT : T p q → ConvPatchEq kH kW x₁ p q`, and
  `hmq : MaxPool2MarginQUpTo δ T (conv2d W b x₁)`; `cnn_conv1_sgd_descends`,
  `cnn_conv1_bias_sgd_descends` the same with `ConvPatchEq2 kH kW x₀`. Book (thm:cnn_sgd_descends
  prose + a twins paragraph), yaml §3 sentence, AuditAxioms repointed.
- **Binary32 rungs keep `MaxPool2MarginQ` (statements unchanged), deliberately.** Their pool
  backward (`MaxPool2IsArgmax` in `cnnConv*FloatGrad`) routes the
  cotangent to EVERY tied cell, so at a twin tie the float gradient is not the loss gradient (a
  4-way tie counts the window four times). They feed the real rungs through
  `to_marginQUpTo_flat`. Correction (coordinator, re-checked): the rendered trainers do NOT use
  the EQ-mask. `verified_mlir/cnn_train_step.mlir` and the other pool backwards print
  `select_and_scatter` with a `GE` select (`Pretty.lean`), which routes each window's cotangent to
  ONE cell, so at a twin tie the emitted conv gradient is the loss gradient (either twin gives the
  same θ-gradient). The gap is between the binary32 proof model (`MaxPool2IsArgmax`, every tied
  cell) and the artifact, at ties only. The CNN.lean `maxPool2HasVJP3` docstring still describes
  the old tile-compare-select emitter and its "PyTorch/JAX semantics" (both pick one cell): fixed in phase 2 (tag I, under WP9).
- **Acceptance.**
  - Probe scripts/probes/mnist_pool_twin_probe.py (CPU, float64), trained `cnnVerified` dump
    `.lake/build/cnn_verified_params.bin` (on this box, not in git; first 3489130 floats,
    verified by its test accuracy). Full test set: old `MaxPool2MarginQ` 0 images, twin-free
    (binary32) 0, `MaxPool2MarginQUpTo` with relu clauses 10000/10000 for both the conv2 and the
    conv1 twin families; every positive tie is a twin. Numbers in the probe's output, not in Lean.
  - Concrete instance: scripts/certs/trained_cnn_descent.py → `Trained/CnnDescent.lean`,
    `trained_cnn_conv2_sgd_descends_concrete` through the new corollary
    `cnn_conv2_exact_sgd_descends` (exact gradient, every hypothesis at the explicit radius
    `lr·cnnConv2GradBound`, via `Conv2Slot.gradAt_abs_le` and Cauchy–Schwarz). Reduced 12×12
    2-channel net (bias-free conv1, so a blank patch matches the padding), trained without a
    pool regularizer, MNIST test image #0, five live twin-tied windows, `lr = 2⁻⁴²`. Byte-identical
    on regeneration; elaborates in about 1.5 min. The decrease is not shown positive (no gradient
    lower bound).
- **Parked.** (1) The ResNet-stem half of part 2B (stem θ-gradients as a germ; the render's
  scatter routes ties to one cell, needs the identical-patch weight-grad lemma): done (tag J,
  status below). (2) A-pq-2
  (`cnn_conv1_cot_close`): done in phase 2 (tag I, under WP9). (3) Concrete instances of the conv1 / bias rungs
  (the corollary pattern of `cnn_conv2_exact_sgd_descends` repeats; conv1 needs a `ConvPatchEq2`
  certificate per window): done (tag K, status below). (4) A yaml row for the new concrete
  theorem (needs a gen_comparator_tier DECLS entry): done (tag K).

**Status 2026-10-02 (overnight, agent J): the ResNet-stem half of part 2B done (staged); 0
`verified_mlir/` changes, no pinned statement changed.** Re-checked at `bb2d22a2`.
- **The germ in θ was already there.** Part 2A states the stem pull-back as a germ in θ
  (`r34StemPool_param_germ` over each one-slot family, `r34StemGCg_hasGradAt` on the gather model,
  `congr_of_eventuallyEq` onto the real pooled net); the `HasGradAt` it takes is the loss after the
  pool, which is differentiable. So "restate as a germ in θ" needed no change. What was missing
  is the scatter: the nodes are read at `maxPool3s2BackB`'s denotation, which routes each window to
  `maxPool3s2LocalReindexB`'s cell (a `Classical.choose` argmax), while the emitted
  `select_and_scatter` (GE) routes to the first maximal cell. At a twin tie the two can differ.
- **Landed.**
  - HeadLayers: `IsMaxPool3s2SelectB` (each output reads a maximal cell of its own window),
    `maxPool3s2LocalReindexB_isSelect`, and `stemPoolRelu_param_eventuallyEq_select` (the 2A
    germ for ANY selector; twins equal at every θ make the choice irrelevant).
    `stemPoolRelu_param_eventuallyEq` is now its one-line instance.
  - ResNet34ParamGrad: `r34StemPool_param_germ_select`, `r34StemCotNAt` / `r34StemCotCAt` (stem
    cotangents routed along σ), `r34StemCotN_eq_at` / `r34StemCotC_eq_at`, `r34StemGCg_hasGradAt_at`,
    `r34StemLossTiedAtB` (the four-node bundle at given cotangents; `r34StemLossTiedB` is now
    it at the render's cotangents, same meaning), `r34_stem_lossTiedB_select`. The 2A theorems
    (`r34StemPool_param_germ`, `r34StemGCg_hasGradAt`, `r34_stem_lossTiedB`) keep names and
    statements and become instances at the argmax gather.
  - **The lemma:** `r34Stem_select_grads_eq`: for any selector σ and any pool cotangent, the four
    stem node dens (conv weight / bias, BN γ / β) at the σ-routed cotangents equal those at
    `r34StemCotC` / `r34StemCotN`. Proved by uniqueness, not by computing the patch sums: both
    are gradients of `linLoss dy` after the pooled stem at the same point. `r34StemLossTiedB.select`
    transfers the capstones' stem bundle to any selector.
  - Net level: `r34_net_lossGrad_stemSelect`, `r50_net_lossGrad_stemSelect`: under the
    capstones' own hypotheses, the stem's four nodes at the cotangent routed along any selector
    (the emitted scatter among them) are the loss gradients in the stem parameters.
  - Prose: both ParamGrad module docs; blueprint thm:resnet34_loss_grad /
    thm:resnet50_loss_grad (`\lean{}` + one sentence each) and the "one conditional" paragraph;
    AuditAxioms pins for the nine new declarations.
- **Not formalized:** that the printed `select_and_scatter` picks a maximal cell is the
  StableHLO semantics of a GE select; Lean has no model of the printed op, so the selector
  hypothesis is where the artifact enters (same standing as every other `den`).
- Line deltas: HeadLayers +71/−21, ResNet34ParamGrad +261/−46, ResNet50ParamGrad +46/−2.

**Status 2026-10-02 (overnight, agent K): parked (3) and (4) done (staged); 0 `verified_mlir/`
changes, no pinned statement changed.**
- **Corollaries (SgdDescent.Cnn).** `cnn_conv2_bias_exact_sgd_descends`,
  `cnn_conv1_exact_sgd_descends`, `cnn_conv1_bias_exact_sgd_descends`, each with its named radius
  (`cnnConv2BiasGradBound`, `cnnConv1GradBound`, `cnnConv1BiasGradBound`). They share three private
  helpers (`stepRadius_exact_le`, `gradAt_l1_le_card`, `curvature_exact_le`), and
  `cnn_conv2_exact_sgd_descends`'s proof now uses them too (statement unchanged). The conv1 gradient
  bound is the new `Conv1Slot.gradAt_abs_le`. To get it, `Conv1Slot.sgd_descends`'s two inline
  facts became private lemmas, `z2_row_l1` and `z2_pdiv`. Also new: `ConvPatchEq2.symm` / `.trans`,
  the equivalence `windowMarginUpTo_of_cert` needs. They belong in ConvIndex, but tag I holds that
  file, so they sit in Cnn.lean for now.
- **Instances.** Both come from scripts/certs/trained_cnn_descent.py, which now trains two nets
  with the same seed-0 loop.
  - `Trained.CnnDescent` adds `trained_cnn_conv2_bias_sgd_descends_concrete` at `lr = 2⁻³⁶`, on
    the same net and image. Its radius sits inside the kernel rung's, so `c2_margin` /
    `pool_margin` carry over by monotonicity.
  - The conv1 rungs can't be satisfied on that net: their relu₁ margin needs every conv1
    pre-activation nonzero, and a bias-free conv1 is exactly 0 on a blank patch. So the new
    `Trained.CnnDescentConv1` uses net B (trained conv1 bias, init 0.1), on test image #0 again.
    Two live windows tie through two-layer twins (`tw_*`, 5×5 blank fields with in-bounds outer
    reads, at column 1 rows 6–9). Results: `trained_cnn_conv1_sgd_descends_concrete` at
    `2⁻⁴⁸`, `trained_cnn_conv1_bias_sgd_descends_concrete` at `2⁻⁴⁵`.
  - Both files regenerate byte-identically. The generator refactor left the conv2-kernel half of
    CnnDescent.lean byte-identical. Each file elaborates in about 1.5–2 min.
- **yaml.** Four rows (one per concrete theorem) after the `TrainedLinearDescent` row, the house
  style for trained-weight concrete theorems. The §4 prose sentence names both modules. Both
  modules are added to gen_comparator_tier DECLS + MODULES, and the lakefile has the Certs root.

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
- **Landed 2026-10-01 (overnight, staged): `enet/cnx/vit_net_tied_lossGrad`**, one per
  ParamGrad file, at the smoothed loss like R34's (the B0/ConvNeXt/ViT ties fix `g` to the
  smoothed cotangent), each pinned in AuditAxioms beside its `*_smoothedCE`. Generated by
  `wp2_combo_gen.py cnx|enet|vit`; the drivers are now kept in the script.
  - The bridges did it. The proof proves `cnxPreX N ε w x = ibK` one stage at a time from the
    `*_apply` lemmas (`enetPreBk_apply`, `vitPreE_apply`, …) and the logits bridge from
    `cnx_logitsB_eq` / `enet_forward_eq_head` / `vit_logitsB_eq`, then `rw`s them into the
    unfolded loss hypothesis, so the final `exact` compares the tie's let names on both sides.
    ConvNeXt and ViT then close at once.
  - B0 needed one more step. `EnetNetLossTiedG`'s chain passes ε-positivity proofs to the
    `*HasVJP` witnesses, and the def abstracted them (`EnetNetLossTiedG._proof_*`). Matching the
    chain against the tie's `hεw.bk.e` spelling then exceeded `maxRecDepth` (it passes at 8192).
    Instead, `extract_lets` merges the tie's and the loss side's lets into the goal's, and the 16
    loss-side cotangent lets that cannot merge are equated with the goal's one step at a time
    (`remerge`), so no option is raised.
  - ViT pairs the tie's final-LN and classifier conjuncts with the loss side's one head conjunct
    (`groups`).

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

**Status 2026-10-02 (overnight, agent L): G-corr-1's class at the pool backwards, done (staged);
0 `verified_mlir/` changes, no pinned statement changed.** Re-checked first: `maxPoolBack`'s `den`
(`maxPoolBackFlat`, = `IR.maxPoolBackDenote` flattened) tested `MaxPool2IsArgmax`, so a k-way
positive tie counted the window k times, while the printed `select_and_scatter` (`GE` select,
NCHW, window `1×1×2×2`) keeps the current pick while it is `≥` the next cell and so routes to ONE
cell, the first maximum in row-major window order. `maxPool3s2BackB` (and `maxPool3s2Back`) already
routed to one cell, but a `Classical.choose` one, so at a tie den and text could name different
cells. No other pool backward exists in `SHlo`.
- **Fix, at the root of both.** `windowArgmax` (WindowMax) is now the least maximal offset in
  row-major order (`windowMaxOffsets`, `min'` over the flat index `a·k + b`), with
  `windowArgmax_max` re-proved and a new `windowArgmax_first` (every earlier offset is strictly
  below). `maxPool2Argmax`, `maxPool3s2Argmax`, both local reindexes, and so the 3×3/s2 `den`,
  are therefore the printed op's choice with no other edit (the clamped first-window offset repeats
  the real cell after it, so the first maximal offset names the first maximal position the padded
  op visits). `maxPoolBackDenote` and `maxPoolBackFlat` route by
  `maxPool2Argmax … = (winRowMod, winColMod)`, the same spelling in both, so every `rfl` graph tie
  between the render's `den` and the `Back3` chains still closes untouched.
- **(a) Step ties.** Bridge: `IR.maxPool2Argmax_eq_iff_isArgmax` (→ at every point, ← under
  `MaxPool2Smooth`) and `IR.maxPoolBackDenote_eq_of_smooth`. Re-proved through it:
  `maxpool_back_bridge`, `maxpool3_node_bridge`, `maxpool_flatten_bridge`, `maxPoolBack_faithful`,
  and ChapterGraphTies' `maxPoolFlatHasVJPAt'` (now `maxPoolBack_faithful` applied). Nothing else
  broke: every step tie, seal, the ResNet stem layer and J's selector lemmas only use
  `windowArgmax_max`.
- **(b) H's owed items 1 and 2, closed exactly (not only off ties).**
  `SmallParamGrad.maxpool_flatDenote_eq_selScatter`: the `Back3` maxpool node is `selScatter` along
  `poolSelIdx (maxPool2Argmax x)` at every point. Hence `CnnPoC.cnnChainCotW2_eq_sel` and
  `CifarPoC.cifarChainCotW2_eq_sel`: the step ties' chains ARE the capstones' `*Sel` chains at
  `σ = maxPool2Argmax`, whose `PoolSelDom` clause `poolSelDom_argmax` already discharges. Cifar8 and
  Cifar8Bn reuse these chains / the scatter lemma.
- **(c) Prose.** WindowMax header, IR section comment + `maxPoolBackDenote` / bridge docstrings
  (and the stale `MaxPool2IsArgmax` decidability comment, deleted), `Basic`'s constructor comments
  and the two flat backwards' docstrings, the six step-tie "Scope" bullets, the Cnn/Cifar ParamGrad
  module docs, the R34/R50 ParamGrad stem paragraphs and `r34_net_lossGrad_stemSelect`, blueprint's
  `thm:cnn_loss_grad` follow-up paragraph, Proofs/README trust item (b). AuditAxioms pins for the six
  new theorems.
- **Not changed:** the binary32 rungs' `MaxPool2IsArgmax` routing (`cnnConv*FloatGrad`, agent K's
  files) and `maxPool2HasVJPAt3`'s backward (smooth-only, read by CnnSeal); blueprint
  `thm:cnn_sgd_descends`'s "routes to every tied cell" is about that float model and stays true.
- Line deltas: WindowMax +42/−10, IR +48/−20, Basic +25/−13, SmallParamGrad +36, CnnParamGrad
  +17/−5, CifarParamGrad +12/−1, ChapterGraphTies +2/−3, six step-tie docs about +4/−3 each,
  ResNet34ParamGrad +6/−7, ResNet50ParamGrad +3/−3.

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

**Status 2026-09-30: four done (staged), three parked; N2-scope-1 done 2026-10-01.**
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
- **N2-scope-1 done (2026-10-01, overnight), staged.** Both nets have a drop-path forward
  statement on the `EfficientNetFullB0Drop` template; each file elaborates in seconds.
  - **MNv4:** new `MobileNetV4FullBDrop`. `mnv4SkipDrop` / `mnv4SkipDropGraphB` (generic, one
    `_faithful`) put a `dropPathB` at `dpName k` on each of the 18 skip rows, in
    `mnv4DropSites` order. Five drop-carrying group graphs and
    `mnv4FwdGraphBFullDrop_faithful` sit over the classifier-dropout head, all by outside-in
    `rw`, with no `simp only` over `den`. `mobilenetv4ForwardBFullDrop_sdOnes` (to
    `…Do`) and `_ones` (to `mobilenetv4ForwardBFull`) are stated against the `*_fwd_apply`
    expansions. The masks are non-optional, like MNv4's `Do`: both artifacts are `dropdo`.
    MobileNetV4FullB is touched in three places: `mnv4StemB_graph_faithful` and
    `mnv4HeadDo_graph_faithful` lose `private`, and its artifacts row points at the new file.
  - **ViT, decision: a batched graph.** A per-example graph with the mask as an input cannot
    exist: a per-example node cannot see the example index, which is why the drop artifacts are
    written by `ViTRenderB`. So new `ViTFwdDrop` states:
    - `vitBlockGraphBDrop` / `vitBodyGraphBDrop` / `vitFwdGraphBDrop`, at the render's tokens and
      names (`b<i>_`, `%dp<2i>` / `%dp<2i+1>`, `%wConv`, `%Wc`), any depth;
    - `vitFwdGraphBDrop_slice`: example `t` of the graph is the per-example
      `vitForwardKVDrop` at example `t`'s mask entries;
    - `vitFwdGraphBDrop_faithful`: the graph denotes `vitForwardKVDropB`;
    - `vitForwardKVDrop_ones` / `vitForwardKVDropB_ones`: at all-ones masks the forward is
      `vitForwardKV`, per example and `batchMap`ped.
    The spelled block ties to `blockVDrop` as `vitBlockSpelledMHV_eq` does. Like
    `vitFwdGraphKMHV`, the graph is not text-tied (shared intermediates).
  - Pins: 12 AuditAxioms lines, two lakefile roots, and book clauses in
    `thm:vitFwdGraphKMHV_faithful` and `thm:mobilenetv4FullHasVJP`.
  - Owed:
    - ~~a FwdGraphTextTies guard for the MNv4 drop skip row~~ done 2026-10-01: all 21 rows at
      their `mnv4DropSite` print as `mnv4SkipDropGraphB`, and the drop graph's site numbering is
      `mnv4DropSites` (no name changes needed);
    - the backward through the drop sites, for both nets, as for B0.
- **N1-gen-2 done (2026-10-02, overnight, agent H), staged; 0 `verified_mlir/` changes.** All six
  chapter nets have a `*_net_lossGrad`, per example, at the un-fused `*Grad` nodes, for any loss
  `L` with gradient `g` at the logits, plus a `_CE` corollary at the emitted loss cotangent:
  `linear_net_lossGrad`, `mlp_net_lossGrad`, `cnn_net_lossGrad`, `cifar_net_lossGrad`,
  `cifar8_net_lossGrad`, `cifar8Bn_net_lossGrad` (bundles `LinNetLossTied` … `Cifar8BnNetLossTied`).
  Each file elaborates in seconds.
  - **Kit** (new `Nets/Small/SmallParamGrad`, the per-example peer of ParamGradNodes): node lemmas
    (`convW/convB/denseW/denseB_hasGradAt`; BN `bnGamma/bnBeta_hasGradAt` in the BN file), the
    `*Sgd = θ − lr·*Grad` bridges, stage pull-backs (`hasGradAt_dense/relu/conv`), the pool at a
    fixed selection (`poolSelIdx`, `PoolSelDom`, `selScatter`, `hasGradAt_gatherRelu`), the abbrev
    `MaxPool2SmoothUpTo` (the 2×2 `WindowSmoothUpTo`, pre-activation, dead = all `≤ 0`;
    `MaxPool2MarginQUpTo` implies it) and the germ `maxPool_relu_eventuallyEq_sel`.
  - **Pool clause in θ, as WP1 2A/2B.** Twins are semantic: cells equal at every weight upstream
    of that pool (`CnnPoolTwin`, `CifarPoolTwin2`, `Cifar8PoolTwin2/3/4`, `Cifar8BnPoolTwin1–4`);
    `cnnPoolTwin_of_convPatchEq2` gives them from identical two-layer receptive fields (the
    probe's conv1 family), and the CIFAR nets' first pool reuses `CnnPoolTwin`. A stage-`s`
    parameter's germ rewrites pools `s…4` outermost first at the true pre-activations.
  - **Which cotangent: a finding.** At a live tie the loss gradient routes each window's cotangent
    to ONE maximal cell (the gather's adjoint at a selection `σ`, any `σ` naming a maximum: the
    rendered `select_and_scatter` GE choice is one). The step ties' chains read the 2×2 pool
    backward as `maxPoolBackDenote` (the `den` of `maxPoolBack`), which routes to EVERY maximal
    cell, so at a positive twin tie those chain cotangents are NOT the loss gradient (a k-way tie
    counts the window k times). The capstones therefore state the chain with new constructors
    `cnnChainCotW2Sel` / `cifarChainCotW2Sel` (scatter at `σ`) and reuse `cnnChainCotW1`; they
    agree with the step ties' chains except at a live tie (remark, not a lemma). The step-tie files'
    "not stated" sentences (CnnFold, CifarFold, Cifar8StepTie(G), Cifar8BnStepTie(G), MlpFold) now
    point at the capstones and say this. The same gap is the binary32 one WP1 noted, here at
    the ℝ `den` level: `maxPoolBack`'s `den` is not the artifact's `select_and_scatter` at a tie.
  - **Pins:** 27 AuditAxioms `#print` lines, 7 Certs roots, six yaml rows + gen_comparator_tier DECLS /
    MODULES (tier regenerated, 48 theorems), five book blocks (`thm:linear_loss_grad`,
    `thm:mlp_loss_grad`, `thm:cnn_loss_grad`, `thm:cifar_loss_grad`, `thm:cifar8_loss_grad`),
    `\uses` by `blueprint_uses.py --fix`, dep-graph figures ch1–4 regenerated.
  - **Hypotheses:** odd kernels (the rendered conv backward is the conv VJP there); BN `ε > 0`.
  - Cifar8BnParamGrad's statements were generated by a scratch script (not committed); the file is
    hand-maintainable.
  - Owed: (1) and (2) are closed by agent L (WP4 status, 2026-10-02): `maxPoolBack`'s `den` now
    routes to the first maximal cell, as the printed op does, and `cnnChainCotW2_eq_sel` /
    `cifarChainCotW2_eq_sel` make the step ties' chains the capstones' at that selection, ties
    included; (3) `*_net_tied_lossGrad` combined statements (the seven nets have them); (4) a
    `_of_convPatchEq2` lemma for the CIFAR nets' deeper pools and the BN first pool; (5)
    `SmallParamGrad.maxPool_relu_eventuallyEq_sel` repeats the filter argument of
    `Conv2Slot.maxPool_relu_eventuallyEq_gather` (still in SgdDescent/Cnn, which the kit does not
    import); derive it from that lemma once it moves to a light module.
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
  - **Status 2026-10-02: A-pq-1 done with WP1 part 2B (see WP1's status); A-scope-1 was done in
    `1dcbcb58`; A-pq-2 parked, then done in phase 2 (tag I, under WP9).**
- **8c MaxPool unification**, ~−400 (**A-gen-2**, L): one window-max over an index family, with
  2×2 and 3×3/s2 as instances. ⚠ Graph ties `rfl`-match `maxPoolBack` (the IR-spelled trap). Keep
  the instance names as `abbrev`s and check the T2 ties. Best done *with* WP1 part 1.
  - **Status 2026-10-01: landed (staged) as structure, not as the line cut; 0 `verified_mlir/`
    changes.** Re-checked at HEAD: the two developments were still parallel. Net about +65 lines
    (new `Architectures/WindowMax.lean` +327, MaxPool3s2 −222, CNN −40), not −400.
    - `windowMax r s` is the max over the product window `{(r hi a, s wi b)}`; one copy of the
      positions-form predicates (`WindowSmooth`, `…OrDead`, `…UpTo`, the pairwise/injective
      discharges), the argmax, the local linearisation, `pdiv3`, the sum-form VJP, the flat
      bridge, closeness, magnitude, shift and nonnegativity.
    - 3×3/s2: `maxPool3s2`, `MaxPool3s2Smooth{,OrDead,UpTo}`, `maxPool3s2Argmax` and
      `maxPool3s2LocalReindex` are `abbrev`s of the generic; every pinned theorem keeps its name
      and statement as a one-line instance. Three names stay `def`s with their old bodies, because
      downstream proofs unfold them and the files are not ours: `maxPool3s2Flat` (BatchSealKit's
      `simp only`), `maxPool3s2HasVJPAt3` / `maxPool3s2FlatHasVJPAt` (the simp sets in
      BackwardMaps and `maxPool3s2Back_faithful`; the backward is spelled out, `correct` is the
      generic's). `maxPool3s2Flat_continuous` stays: `fun_prop` in ResNet34FullBSeal needs it.
      Unused instance lemmas deleted (`le_maxPool3s2`, `maxPool3s2_attained`, `…_abs_le`,
      `…_close`).
    - 2×2: `maxPool2` keeps its four-way `max` definition (the IR graph ties, `MaxPool2IsArgmax`'s
      lookup backward and the generated CnnSeal/CnnWitness read it); `maxPool2_eq_windowMax` is
      the bridge and `windowSmooth_of_maxPool2Smooth` turns the offsets form into the positions
      form. `maxPool2Argmax` / `maxPool2LocalReindex` are `abbrev`s, `maxPool2_flat_hasFDerivAt`
      and `maxPoolFlat_{close,abs_le}` come from the generic; `maxPool2_eq_at_max`,
      `maxPool2_abs_le` and `abs_max_le` deleted. `MaxPool2Smooth`, `pdiv3_maxPool2_smooth`'s
      decoded form and the codegen collapse stay 2×2-specific.
    - Why not −400: the 3×3 file was already compact (`Finset.sup'`), the 2×2 spelling cannot move,
      and the pinned instance names stay as shims. What it buys: one proof of the analytic core,
      and a generic `WindowSmoothUpTo` that WP1 part 2B's 2×2 twin predicate can instantiate.
    - `max_close` and `maxPool2_close` are now reached only from AuditAxioms
      (`audit_only_mentions.py`): `maxPoolFlat_close` goes through the generic. They are pinned,
      so they stay.
- **8d Nets**: **N1-reuse-1/2**, **N2-reuse-1…4**, **N2-pq-1…4**, **N1-pq-1** (28 undocumented
  `show`/`change`; add `_def` lemmas).
  - **Status 2026-10-01: done (staged) except N1-reuse-2; nothing renamed, no pin moved.** Every finding was re-checked at `9eed9240`.
    - N1-pq-1: `r34IdB_apply` / `r34DownB_apply` / `r34StemB_apply` (ResNet34FullB),
      `r50IdB_apply` / `r50ProjB_apply` / `r50DownB_apply` (ResNet50FullB) and
      `residualProj_apply` (Residual, beside `residual_apply`). The two seals' block, stem and
      `hout` shows are `rw`s with them; the `hmid`/`hm1`/`hm2` shows are gone too: the
      zero-conv and constant-BN lemmas take the seal weights' fields as explicit arguments,
      so they match the clause unreduced. ConvNeXt: `unfold cnxDownCotInChAt`,
      `rw [cnxDownBack, Function.comp_apply]`, and `cnxStageChKBack.eq_2` plus
      `cnxBlockChBackAt` in the stage fold. MnistCNN's four (`Fin (1*(2*1)*(2*1))` restated at
      `Fin 4`, `cbr`/`rblk` read applied) are definitional on purpose and carry a comment.
    - N2-reuse-1: `den_swishF_shard`, `den_addV_shard`, `den_relu6_shard`, `den_castIdx_shard`
      and R50's `den_addVB_shard_comm` (same class, not in the finding) moved into SyncKit,
      names and namespace unchanged, so the AuditAxioms lines stand.
    - N2-reuse-2: `mnv4SkipGraphB_faithful` is public; the Eval copy is deleted.
    - N2-reuse-3: `projBEval` moved from PCEval to `Batched.Stages` beside `projB` (same full
      name, so B0's Eval/Drop `unfold`s stand); `mnv4ProjBEval` deleted, its uses repointed.
    - N2-reuse-4: `dense_transpose_eq_mulVec` lives in `BackwardMaps`;
      `dense_transpose_eq_vjp_backward` is it in one line. `rowDenseBackFlat_eq_perRowFlat` is
      `rw` + `rfl`, and `vitCotLn2_eq_perRowFlatPR` (N2-pq-4) rewrites with it and the new
      `vitCotM1_apply` (ViTChainClose; `vitCotG` is private), so its `show` is gone.
    - N2-pq-2: `resid_id` and `head_eq_dense`'s first `show` are `rw`s (`CertLayer.residual_fwd`
      + `residual_apply`; `mobilenetv4ForwardBFull`); its second `show` reads the `sealW` fields
      off for `cbReluB_eq` and is commented. `mnv4SkipCotIn_eq_vjp` rewrites with the new
      `CertLayer.residual_graph`.
    - N2-pq-3: `rw [sum_finProdFinEquiv (m := h) (n := dh)]` ×3 replaces the three motive-spelled
      `← Equiv.sum_comp` and the `sum_prod_type`s.
    - Already done at HEAD: N1-reuse-1 (both `sealX_continuous` deleted in the small tier).
    - N2-pq-1 (after WP8e released the file, on top of its `hdCotIn_eq_vjp` edit): the four
      `have hc … := rfl` blocks go. `xCotIn` / `sCotIn` / `nCotIn` are `simp only [mb*FwdBHasVJP,
      vjpComp_backward]` then the stage `*_back_eq` rewrites; `rCotIn` keeps a three-line `hc`
      (commented: the witness is `residualHasVJP` of the expand body's) and then does the same.
      `mbNoExpFwdBHasVJP` (EfficientNetChainClose) is now term-mode like its two siblings: the
      tactic `unfold` left an `id` cast that `vjpComp_backward` cannot see through.
    - **N1-reuse-2 / N1-place-1 parked (user):** a length name (`vjpChain18At`) changes the
      text of the R34 capstone statement the comparator pins (`ChallengeTier` spells
      `Proofs.r34BFullHasVJPAt`); moving it to OpaquePrefix under its ResNet-34 name only half
      fixes it. N2-place-1 skipped: the ℝ wrappers' home `Training/DropPath` and the graph side's
      `StableHLO/Basic` are both upstream of all of Certs. N2-place-2 is WP8a's
      (`ViTParamGrad` opens `ViTTiePoCGB`).
- **8e Foundation/Float/Certificates**:
  - **F-re-1/2/3**, **F-pq-1** (the `den_convBackBatched_eq_cInB` peers).
  - **C-reuse-1** ✔ (typechecked), **C-reuse-2** (−40), **C-pq-1**.
  - **A-reuse-1** (six `.correct` restatements with only an AA line).
  - **A-pq-3** (13 `show … decimate` → `HasVJP.decimate`; check the `rfl` ties).
  - **Status 2026-10-01: done (staged); A-reuse-1 was already gone in `1dcbcb58`.** Every
    finding was re-checked at HEAD. Net about −125 lines; no `verified_mlir/` changes.
    - F-re-1: `sum_channel_fiber` is `rw [sum_finProdFinEquiv]; simp [Finset.sum_ite_irrel]`.
      Tensor.lean is a root file, but StridedConv (A-pq-3) already rebuilds everything under
      `StableHLO/Basic`, so it rode the same rebuild.
    - F-re-2, reshaped: `bnBatchLABack_faithful` is an AuditAxioms pin, so its statement stays.
      `den_bnBatchLABack_eq_bnBackB` is deleted and its five uses (four in BackLinks,
      `hdCotIn_eq_vjp` in EfficientNetSyncStepTieG) rewrite with `bnBatchLABack_faithful`.
      `bnInB_eq_bnBackB` stays: it is pinned, and it is the `.operand`-leaf form that the
      ParamGrad files and SyncKit read.
    - F-api-2 + F-pq-1: `den_convBackBatched_eq_cInB`, `den_depthwiseBackBatched_eq_dInB` and
      `den_depthwiseStridedBackBatched_eq_dStridedInB` (all `rfl`) sit beside the defs. The four
      `*_back_eq` and `hdCotIn_eq_vjp` lose their `show`, and each `rw`s through the graph node
      by node. The closing `rfl` only folds `bnBackB` / `swBackB`, and the section docstring now
      says so.
    - F-re-3: `floatClose_residual` is deleted, and `floatClose_addResidual` (unchanged, used by
      `floatClose_residualBlock`) says it serves the `residual F` spelling by definition. Its
      AuditAxioms line is gone and the yaml's 4d sentence names `floatClose_addResidual`.
    - C-reuse-1: `CrownBound.mlp2_apply` is deleted, and `crown2_certified_at_eps` uses
      `mlp_out_eq W1 W2 (fun _ => rfl)`.
    - C-reuse-2: `stdNormalQuantile_anti` moved below `stdNormalCDF_quantile` and is now the
      3-line injectivity proof. `stdNormalCDF_sSup_lt_eq_sInf_gt` is deleted, and the name and
      statement are unchanged.
    - C-pq-1, reshaped: no shared Gaussian lemma. The set-level lemma the audit proposed
      (`{ω | q ω ≤ p_y} ⊆ {certified}`) typechecks, but applying it in `smoothing_cp_certified`
      times out in `whnf`, unifying the lemma's set with the goal's through the `set`
      abbreviations. Instead the three capstones drop their spelled-out `hsub` set and the calc,
      and compose as `(coverage bound).trans (measureReal_mono fun ω hω … => …)` over the
      existing `smoothing_certified_of_le`. Statements are unchanged.
    - A-pq-3: `HasVJP.decimate` / `HasVJP.decimateOdd` (vjpComp with the (odd) decimation VJP)
      are in StridedConv. The 13 `show … from vjpComp …` sites are now `hf_vjp.decimate hf_diff`
      (or `decimateOdd`), and so are the stride-4 input and weight VJPs' inner steps. The `rfl`
      ties (`den_*` arms in `StableHLO/Basic`, ConvBack / DepthwiseBack / EvenKernel leaf ties)
      still close with the one extra delta step.
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
  - **Status 2026-10-01 (codegen half, with G-name-1): done (staged); every artifact
    byte-identical** (`regen_verified_mlir.sh proofs` rewrote 274 files, then `check`; empty
    `git diff verified_mlir/`). Every finding was re-checked against HEAD after WP8f.
    - G-place-1 + G-api-1: `R34Opt` and `StableHLO.OptKind` are one `OptRecipe` in RenderKit
      (`adamw | rmsprop | heavyBall | sgd | lamb | adamwAccum k | lambAccum k`), with
      `OptRecipe.label` and `OptRecipe.slug` (the optimizer marker four variant functions had
      spelled separately). `optOne` gains the `.rmsprop` arm and an `emaSuf` argument (`"e"`; the
      ResNet family's `"ema"` through `optAllParams`), and `adamOne` / `rmsOne` / `adamOneEma` are
      deleted: MobileNetV2, EfficientNet (whose hand-written EMA line goes too), ViT and ConvNeXt
      call `optOne`. The rest of ResNet34RenderB's optimizer stage moved unchanged (`optAllParams`,
      `optConstsB`, which takes the net's `RmsHyper` for `.rmsprop`, `accumScalarConsts`, the clip
      constants, `optWd*`, `wdVariantMark`, `lsVariantMark`, `wdNameExcludes`). MobileNetV4 imports
      RenderKit + SyncBnSites, its `mnv4OptLabel` is `OptRecipe.label`, and its `none` tail is
      `optAllParams .adamw`; R50's label match is `opt.label`. MobileNetV2 / EfficientNet take any
      recipe but the accumulating two (no `G` region); ResNet-34 refuses `.rmsprop` (no RmsHyper).
      Not merged: `CnnRender.CifarOpt`, the per-example CIFAR family's own selector.
    - G-place-2: `git mv` to `Nets/EfficientNet/EfficientNetStagesPC{,Eval}.lean`; importers,
      the Certs root, `LeanMlir.lean`, AuditAxioms and the Codegen README follow. No renames.
    - G-name-1: the 33 renderers are `<net><Kind>Text` (the kit's existing `maxPool3s2FwdText`,
      `allReduceMeanText`, `mnv4RowGraphText` spelling, thing-first like the VJP names). `B` stays
      only where a per-example peer exists (`cifar8AdamTrainStepBText`, `cifar8BnTrainStepBText`,
      `convNextAdamTrainStepBText`, `vitAdamTrainStepBText`); `resnet34FwdFaithfulB` →
      `resnet34FwdText`, `mnv4FwdFaithfulV` → `mnv4FwdText`. The rule is in the Codegen README.
      Repointed: tests, apps, docstrings, the lakefile comments, the book's two `\texttt` names,
      `scripts/parity/resnet_timm_parity.py`, ffi/jax.yml comments and the three
      `runs/2026-08-27-r50-a2-a1-ema-fifth-region/render_ghostbn*.lean` recipes; planning docs keep
      the old names as history.
- **Program:** **X-pla-1/2/3**, **X-nam-1**.
- **API:** **F-api-1** (`FloatModel.u_nonneg` field → theorem, 112 uses unchanged), **F-api-2/3**.
- **Naming** (approved 2026-09-30):
  - **G-name-1**: 33 String renderers named `*Faithful{V,B}`.
  - **N1-name-1**: the `*PoC` namespaces (carried from cleanup_backlog §8).
  - **P-N-1**, **N2-name-1**.
  - **C-nam-2**: the `LipschitzCertDemo` namespace (carried from api_design_audit §6.5; ~181 AA
    lines plus 9 generators; expensive).
  - **Status 2026-10-01 (proof half, tag E: F-api, F-pl, C-plc, C-nam-2): done (staged) except
    F-pl-2; 0 `verified_mlir/` changes.** Every finding was re-checked against HEAD (after WP8e).
    - F-api-1: `FloatModel.u_nonneg` is a theorem under the same name (from `err 1`); the field and
      the two `u_nonneg :=` lines go, every `M.u_nonneg` use is untouched.
    - F-api-2 was already done by WP8e (`den_convBackBatched_eq_cInB` and peers).
    - F-api-3 + F-pl-1: `FloatModel.rnd_zero` (now `@[simp]`) and `FloatModel.dot_right_zero`
      move from Binary32Instance into FloatBridge beside `dot_succ`.
    - F-pl-2 parked: its home is Tensor.lean, a root file; it rides the next Tensor batch.
    - F-pl-3: done after all. `Float/RndP.lean` is not a root file (two importers), so
      `int_log_zpow_mul`, `int_log_abs_zpow_mul` and `rndP_zpow_mul` move there from SyncBf16 with
      their names kept (no `Int.` namespace: a project lemma in Mathlib's namespace breaks on the
      bump that adds it).
    - C-plc-1: `FloatModel.mlp2F` / `mlp2_float_close_uniform` live in MlpFloatBridge and
      `Robustness.certified_at_eps_close` in DenseEuclid; `lipschitz_cert_float.py` emits only the
      capped instance (`--check` passes); the Certificates README says so and drops its file count.
    - C-plc-2: `Foundation/IntervalBoundConv{,Q}.lean` → `Certificates/IntervalBoundConv/{Basic,Q}.lean`
      (`git mv`); importers, `ibp_conv_scorecard.py` and the `Net.lean` it emits, the Certs root,
      AuditAxioms(Heavy), certs-heavy.yml, the yaml note and the README follow. `IBP.` unchanged.
    - C-plc-3: the stdGaussian full-support instance, the likelihood ratio, the 1-D Cameron–Martin
      formula and the halfspace-mass lemma move to UpstreamDraft (`MathlibUpstream.`
      `instIsOpenPosMeasureStdGaussian`, `gaussianPDFReal_add_mean`,
      `integral_gaussianReal_comp_add_const`, `integral_indicator_Iic_eq_cdf`), generalised to any
      mean and variance (the Cameron–Martin one also drops its measurability binder), mirrored in
      PR1 / PR2 and a new `PR3_GaussianMultivariate.lean`; a section-by-section diff of the drafts
      against UpstreamDraft is identical. C-doc-4b (the stale path, the `haveI` drift) was already
      fixed at HEAD.
    - C-nam-2: `Proofs.LipschitzCertDemo` → `Proofs.Robustness` everywhere (engines, scorecards,
      AuditAxioms 106 + Heavy 75 lines, the yaml row, the comparator DECLS and regenerated tier
      files, 9 generators). The two generators with `--check` (`lipschitz_cert_float`,
      `smoothing_net_witness_gen`) pass; the other seven have no `--check` and retrain nets, so they
      got the same token substitution as their outputs instead of a rerun.
- **Status, proof half (P-PL-1/2, P-G-2, N2-place-1/2, P-N-1, N2-name-1; 2026-10-01 overnight):**
  - P-G-2 landed: `colSlabApplyH`, `pdivMat_colIndepH`, `colSlabwiseHasVJPMatH`,
    `colSlabApplyH_flat_differentiable` and `sdpa{Q,K,V}_flat_differentiable` live in
    Architectures/Attention (namespace `Proofs`); `pdivMat_colIndep`, `colSlabwiseHasVJPMat.correct`
    and `colSlabApply_flat_differentiable` are their constant-family instances, and
    `colSlabwiseHasVJPMat.backward` keeps its spelling. `attnCore*` stays in ViTParamGrad: it reads
    ViTMultiHead's `headSliceMat` / `headPadMat` and the `*_backward` ties read ViTBackChains'
    `coreQFlat`, both downstream of Attention.
  - P-PL-1 rest landed: `rowDense_{weight,bias}_differentiable` → Attention (beside
    `dense_per_token_flat_differentiable`); `chanLNTensor3_{gamma,beta}_differentiable` →
    ChannelLN; `seGateMulB` / `seGateMulBHasVJP` / `seGateMulB_differentiable` → Batched/BackLinks
    beside `gateCotB` (not SE.lean: the VJP's backward IS `gateCotB`, which sits downstream of SE);
    `hasGradAt_linLoss_constAdd` deleted, its four uses are `hasGradAt_constAdd` at `linLoss dy`.
  - P-PL-2 landed: `batchSlice_batchMapAux` beside `batchSlice_batchMap` in Batched/Basic (now
    `StableHLO.batchSlice_batchMapAux`).
  - P-N-1: the two MNv4 names were WP8a's; `lb_batchMap_congr` is deleted for the audit's
    `congrArg (fun f => Lb (batchMap N f X)) (funext …)`.
  - N2-place-2 landed, reshaped: the nine per-example lemmas go to the end of ViTStepTie (namespace
    `Proofs`), not ViTWholeBackCertifiedTie, which imports neither `vitBlockCotInAtMHV` (ViTStepTie)
    nor `vitCotD{Q,K,V}mh` (ViTMultiHeadChain); ViTStepTie now imports ViTWholeBackCertifiedTie, so
    `vit_net_tied_certified`'s docstring cites lemmas in its own file. ViTStepTieGB keeps the two
    `*B_eq_vjp` lifts.
  - N2-name-1 landed: `vitCotTowerOutV`, `vitCotTowerOutV_eq_vjp`, `vitCotTowerOutB_eq_vjp`; the
    comparator tier text of `vit_net_tied_certified` changes by that name.
  - N2-place-1 parked: the graph half goes beside `dropPathB` in StableHLO/Basic, which the codegen
    half holds tonight, and the motive went away (the ViT / MNv4 drop files use their own
    `blockVDrop` / `mnv4SkipDrop`). Recipe: move `dropPathOpt` / `dropoutOpt` + `_ones` to
    Training/DropPath and the four `*OptG` defs/dens beside `den_dropPathB`, one batched root edit.
- **Status 2026-10-02 (phase 2, tag I: A-place-1/2/3, A-pq-2, the maxpool-backward prose):
  done (staged); 0 `verified_mlir/` changes; no declaration renamed.** Re-checked against the
  staged tree after WP1 part 2B (agent A's rewrite), not the audit's line numbers.
  - A-place-1: `SgdDescent/Cnn.lean` (5443 lines) splits into the ℝ rungs (`Cnn`, 2856) and the
    FloatModel rungs (new `SgdDescent.CnnFloat`, importing `Cnn`): `cnnConv{1,2}{,Bias}FloatGrad`,
    the budgets, `cnn_conv2_cot_close` / `cnn_conv2_cot_real_abs_le`, every `*_grad_close`,
    `convTap_back_close`, `mask_scalar_close` and the four `*_float_sgd_descends`. The reluMask
    restatements (`cnn_conv{1,2}{,_bias}_loss_gradAt_reluMask`, `head3_cot_reluMask`) are ℝ and
    stay in `Cnn`. Certs root, `LeanMlir.lean`, AuditAxioms import, README and the blueprint's
    file list follow. Every moved declaration keeps its name and namespace, and every kit home is
    reachable from `Cnn`, so importers of `Cnn` see the same ℝ names.
  - A-place-2: `ConvGrad` (now importing `ConvIndex`) takes the conv's kernel / input / bias
    Jacobians and drifts (`conv2d_kernel_*`, `conv2d_weight_pdiv`, `convPadWin` / `cotWin`,
    `convWeightGrad_eq_dot`, `convBiasGrad_eq_sum`, `convTap` and its four lemmas,
    `convTap_back_abs_le`, `conv2d_input_pdiv3`, `conv2d_flat_input_pdiv`,
    `conv2d_input_{entry,l1}_drift`, `conv2d_bias_*`, `conv2d_flat_bias_drift_*`). `ConvIndex`
    takes `t3Idx_def`, `MaxPool2MarginQ.poolBack_close`, WP1's twin relations (`ConvPatchEq`,
    `ConvPatchEq2` and their lemmas) and, from `ConvFloat`, the ℝ conv index facts the drifts read
    (`conv2d_eq_convPad`, `abs_convPad_le`, `k4Idx` and its lemmas, `sum_abs_k4`,
    `sum_abs_kernel_slab_le`). `FloatBridge` takes `abs_le_of_close`,
    `FloatModel.dot_perturbed_close` and `FloatModel.sum_perturbed_close`. Reshaped:
    `mask_scalar_close` stays (in `CnnFloat`): its proof reads `sign_stable_of_close`
    (`SgdDescent.Mlp`), downstream of `Float/`.
  - A-place-3: new `Architectures/Activations` (GELU with `differentiable_tanh` /
    `hasDerivAt_tanh`, Swish, sigmoid, the activation-taxonomy note); `LayerNorm` imports it and
    keeps LN, `layerScale` and the vector LN; `SE` keeps the gate. Comments naming the old homes
    (StableHLO/Basic, Attention, SE, the comment-number allow rows) follow. Not repointed:
    `MlirCodegen.lean`'s emitted `// … LayerNorm.lean: pdiv_gelu` comment strings (generic-walk
    output text).
  - A-pq-2: `cnn_conv1_cot_close` is the conv-1-output cotangent closeness both conv-1 rungs
    shared verbatim, stated over two new named cotangents (`FloatModel.cnnConv1CotF`, the float
    one, and `cnnConv1CotR`, the reluMask one) and `FloatModel.cnnConv1CotBudget`;
    `cnnConv1FloatGrad` / `cnnConv1BiasFloatGrad` and the two conv-1 budgets are spelled through
    them (bodies changed, defeq; statements of the pinned theorems unchanged). Net about −15
    lines, not −150: the 45-line margin binder block repeats in the new lemma. The audit's
    `FloatModel.offkink_of_margin` dropped: the four `hz*` discharges per rung differ in their
    nested `layerBudget_nonneg` chains, which a one-line lemma would not shorten.
  - Maxpool backward prose: `maxPool2HasVJP3`'s docstring (and the accessor, the smooth-point
    section, `maxPool2_codegen_matches_canonical`, `maxPool2HasVJPAt3`), IR's
    `maxPoolBackDenote` / `maxpool_back_bridge` and the blueprint's CNN chapter now say the
    renders emit `select_and_scatter` with a `GE` select (one cell per window, as PyTorch and
    JAX); the generic `MlirCodegen` walk's EQ-mask tile-compare-select is named as such. Proofs
    README: the trust-boundary bullet no longer calls the EQ-mask "PyTorch/JAX semantics", and
    residual (b) of the verified path names the scatter.
  - Gates: `lake build Certs CertsHeavy LeanMlir Apps Reference TestSupport` (3656 jobs),
    AuditAxioms (no sorryAx, no errors), comparator `--check`, name lint, import audit implied,
    docstring-checkrefs, blueprint checkdecls (lean_decls regenerated from content.tex) +
    `blueprint_uses.py --check`, target names, comment numbers, module refs, audit coverage,
    `regen_verified_mlir.sh check` with an empty `git diff verified_mlir/`, book_xrefs.
- **Status 2026-10-02 (N1-name-1, tag M): done (staged); 0 `verified_mlir/` changes; no
  declaration's own name changed, only its namespace.** Scope: every `*PoC*` namespace at HEAD
  (23), the audit's list plus the per-net tie / ParamGrad ones (`*TiePoC*`, `Cifar8{,Bn}PoC{,G}`)
  and the two float folds (`Bf16PoC`, `QuantPoC`): all are production capstones, so the same
  reasoning applies. Convention `<net><Fold|Tie><suffix>`: keep the short net prefix, replace
  `PoC` with what the file is (`Fold` in a `*Fold*` file, `Tie` in a `*StepTie*` file, the
  spelling `ResNet34TieB` / `Mnv4TieB` / `EnetSyncTieG` already use), keep the `G` / `B` / `GB`
  suffix; a `*ParamGrad` file and `Bf16GradNodes` keep sharing their capstone's namespace. Map:
  `CnnPoC`→`CnnFold`, `CifarPoC`→`CifarFold`, `MlpPoC`→`MlpFold`, `LinPoC`→`LinFold`,
  `CnxPoC{,G,GB}`→`CnxFold{,G,GB}`, `EnetPoC`→`EnetFold`, `ViTPoC{,G,GB}`→`ViTFold{,G,GB}`,
  `Bf16PoC`→`Bf16Fold`, `QuantPoC`→`QuantFold`, `Cifar8PoC{,G}`→`Cifar8Tie{,G}`,
  `Cifar8BnPoC{,G}`→`Cifar8BnTie{,G}`, `CnxTiePoC`→`CnxTie`, `CnxTiePoCGB`→`CnxTieGB`,
  `EnetTiePoC{,G}`→`EnetTie{,G}`, `ViTTiePoC`→`ViTTie`, `ViTTiePoCGB`→`ViTTieGB`. One scripted
  word-boundary rewrite (777 hits in 52 files: Lean sources, AuditAxioms, the comparator DECLS +
  regenerated tier files, the yaml, content.tex, certs.yml, `LeanMlir.lean`, tests); the
  regenerated tier files differ from the rewrite only in line reflow. By hand: the stale
  qualifiers naming namespaces that no longer exist (`CifarBnPoC`, `ResNet34PoC`, `Mnv2PoC`,
  `ResNet34PoCB`, and CnnArtifacts' `Cifar8PoC` / `CifarPoC` comment) repoint to `SgdNode` / `GradNodeB`,
  where the cited lemmas live (one of them the blueprint's `\leandocref` for
  `bnSgdPairTied_holds`); the six `/-! # PoC:` headers, two "PoC" prose sites in StableHLO/Basic,
  certs.yml's comment, AuditAxioms' two section comments and the book's `ProofsMinimal` line say
  what the thing is; the Proofs README states the convention. No compatibility aliases. Left:
  `jax/scripts/poc_running_bn.py` (a JAX proof-of-concept script, correctly named) and the
  lowercase `poc_*` declaration names in `LinFold` (declaration names, not namespaces; renaming
  them moves a `\lean{}` pin, a candidate for a later naming pass). No generator under
  `scripts/` emitted a `PoC` name besides gen_comparator_tier.py. Gates: `lake build Certs
  CertsHeavy LeanMlir Apps Reference TestSupport` (3664 jobs), AuditAxioms (no sorryAx, no
  errors), comparator `--check`, name lint, import audit implied, docstring-checkrefs, blueprint
  checkdecls (lean_decls regenerated from content.tex: the 29 renamed pins) + `blueprint_uses.py
  --check`, target names, comment numbers, module refs, audit coverage, `regen_verified_mlir.sh
  check` with an empty `git diff verified_mlir/`, book_xrefs.

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
