# MobileNetV4-Conv-M: the Proofs tier, T1 → T6

**Scoped 2026-09-07 by reading the source, not the earlier planning rows.** This is the last net
in `planning/archive/proofs_tier_to_paper_nets.md`'s table with nothing at the net level, and the last
package on that thread. ▶ START at §0 — it corrects the row that scoped this — then §2's order of
work. Every claim below names the file it was read from; where it says "check", nothing was
read and the first session must.

---

## ✅ PROGRESS, 2026-09-07

| session | state |
|---|---|
| **0(a)** record fixes | ✅ DONE (`015dcec`). The stale `## Scope`, `mnv4Stage14`'s Conv-S family order (with a new `#guard` deriving its argument order from `mnv4Blocks`), and every Conv-S count in the render's docstrings — params, block split, stat slots, all three artifacts' arities — re-derived from the committed artifact. `grad_tie.py` 158/104 → 233/154. Four yaml rows for the block tier. |
| **0(b)** the owed ties | ✅ DONE (`b9cc8d5`) and **both PASS**. Forward `max \|Δ\| = 3.770e-06`; gradient 0 of 232 live parameters outside the reference's own relu-discontinuity floor, in raw and `--nokink` mode. ⭐ Block ORDER is now pinned by measurement. ⛔ Two setup findings recorded in `mnv4_convm_ties_todo.md`: `--device=local-task` SEGFAULTS on `@mnv4_fwd` (empty stderr, `local-sync` runs the same vmfb in 1 s) and both scripts hard-coded a nonexistent `iree-compile`. |
| **1** T1 + T2 | ✅ DONE. `Nets/MobileNet/MobileNetV4FullB.lean` (886 lines) + `MobileNetV4FullBVJP.lean` (188). Both build in ~2 s each. |
| **2** T3 | ✅ DONE. `Nets/MobileNet/MobileNetV4FaithfulPoCB.lean` (the §1 fold, 400 lines) + `MobileNetV4TiePoCB.lean` (the §1a tie, 1334). |
| **3** T6 | ✅ DONE. `Float/MobileNetV4WholeBackFloatBridgeB.lean` (268 lines) + `Nets/MobileNet/MobileNetV4WholeBackCertifiedTieB.lean` (910). Both build in ~3 s each. **The net is now closed on every tier that says anything.** |
| **4** the number | open — a GPU decision, see §2. Independent of T6; nothing in sessions 1–3 waited on it. |

### What session 1 actually cost — ⛔ and the ONE finding worth carrying to any future net

⛔⛔⛔ **THE HEADLINE, and it contradicts what §2 below recommends: a net whose resolutions are
LITERALS cannot afford the proof idioms a net with a resolution BINDER can.** ResNet-50's `q` keeps
`den` and every width-indexed `rfl` STUCK; MNv4's 224/112/56/28/14/7 let them RUN, into terms with
hundreds of thousands of elements. **Four separate blow-ups in this session trace to exactly that
one cause**, and each looked like a different problem:

1. **A graph builder pinned to literal widths kernel-times-out.** `mnv4StemGraphB` at
   `ic := 3, oc := 32, h := 112` makes `den_batchOp_convStridedXla`'s `rfl` a claim about
   150528-element tensors and the kernel tries to REDUCE it — 67 seconds, then failure. Stated at
   binders it is 2 seconds, and instantiating at the net's literals is free (applying a proven
   lemma, not proving one). ▶ `mnv2StemGraphB` and `r50StemGraphB` are generic for this reason;
   it reads as a stylistic habit and is not.
2. **`simp only [CertLayer.comp]` rewrites to the full structure literal** — `fwd`, `ok`, `diff`,
   `vjp`, `graph` AND `faithful` — and only then projects `.fwd`. Three `rfl` projection lemmas
   (`comp_fwd`, `residual_fwd`, `id'_fwd`) fix it; they belong in `Foundation/CertifiedChain.lean`
   and are parked in this leaf with a note, per the root-file rule.
3. ⛔⛔ **The whole trunk CANNOT be one `CertLayer`, and this is the expensive one.**
   `fused.comp (res28.comp (… .comp head))` elaborates fine and reads beautifully. But every later
   statement must peel `CertLayer.comp` to reach `.fwd`, and at literal resolutions **all four
   spellings of that peel fail**: `rfl`, `simp only [CertLayer.comp_fwd]`, inside the T2 capstone,
   and as a standalone `mnv4NetLayer_fwd_apply` lemma — each ~10 minutes of elaboration followed
   by a `(kernel) deterministic timeout`. ⚠ The groups' OWN five-stage `comp` chains are fine; it
   is composing the compositions, under something that can start unfolding, that is not.
   ▶ **The fix is seven named prefixes** (`mnv4Pre0` … `mnv4Pre6`) — R50's shape at seven stages
   instead of eighteen.
4. **The T2 capstone must be `rw`, not `simp only`.** The identical seven rewrites: `simp only`
   elaborates ~9 minutes and dies in the kernel, `rw` takes seconds, because `simp only` rebuilds
   the whole term at each step and `rw` works outside-in.

⭐ **What the `CertLayer` route still bought, which is a lot.** `Mnv4SmoothAt` binds **eight**
hypotheses — the stem's kink clause and one `.ok` per group — where ResNet-50's apex binds **33**,
and MNv4 binds **no `0 < ε` hypothesis at all** because those live inside `UibParams` and
`Mnv4BWeights`. The ~60 relu clauses are still assembled by `comp` at each stage's own input and
never written down. ⚠ But the plan's "the apex takes two hypotheses" was optimistic: it is eight.

⭐ `2 * 28` and `56` unify across every stride join with no transport, and `CertLayer.residual`
applies at each of the eighteen skip rows without one either, because `s.oc` and `s.ic` reduce to
the same literal there. Both were free.

⚠ **Two more, smaller:**

* **A `let` chain does not stop a term from doubling.** Eighteen skips each need their input
  subtree twice; `let`-threading looks like it fixes that and does not, because
  `simp only [<the def>]` ZETA-EXPANDS the lets. What works is a folded combinator
  (`mnv4SkipGraphB`) with its own one-step faithfulness lemma, so `den e` occurs once.
* **The graph builders are row-generic and so are their faithfulness lemmas** — one theorem serves
  all thirteen ExtraDW rows, with `s.preDWk ≠ 0` / `s.postDWk ≠ 0` discharged by `decide` at each
  concrete row, so what selects a builder is the TABLE. ⛔ A row-generic SKIP builder is impossible
  (`.addVB` needs `s.oc` and `s.ic` to be the same type, which they are not at a variable row), so
  the body builder stops before the add.

### Session 2 (T3) — what it cost

⭐⭐ **The fold needed ZERO new fp32 op-kind lemmas.** MNv4's nine kinds come from THREE files —
`ResNet34PoCB`'s six, `EnetPoCG`'s three depthwise/XLA ones — which is the sharpest instance yet of
4b's "op kinds are shared far more than the file names suggest". 3 + 6 + 13×12 + 4×9 + 4×6 + 8 =
**233**, every slot exercised (bias-free by construction, so no `convBias` census to over-count).

⭐ **bf16 corrected a sentence three other nets carry.** "The bf16 twins consume the same node" is
FALSE: a bf16 render emits its own `*GradBBf16` whose `den` rounds the operands going in and rounds
the result ONCE, outside `Σ_n`. MNv4 emits five such kinds; three are ConvNeXt's, and
`depthwiseStridedWGradBBf16_den` + `convStridedXlaWGradBBf16_den` did not exist anywhere and are
four lines each here.

⭐⭐ **The tie is SHORTER than ResNet-50's per block, for a structural reason.** The UIB bottleneck
is LINEAR — no activation after the project BatchNorm and none after the skip add — so `dyOut`
reaches the project BN's γ/β **unmasked**, where R50's `r50IdCotA` must first pass the
post-residual relu's mask. And ONE cotangent chain serves all three stride-1 profiles because
`mnv4CotEn` DISPATCHES on `s.postDWk` exactly as `mnv4PostDWSlot` does, off the same row.

⚠⚠ **Everything is generic in the ROW, and that is the same lesson as session 1**: stating any of
it at MNv4's literal resolutions lets `den` run and the kernel give up. The capstone instantiates
at the 21 concrete rows, which is application and is free — `mnv4_net_tiedB` proves in seconds.

⛔ Two structural splits forced by `s.oc` vs `s.ic` at a row binder, both mirroring T2's:
`mnv4BodyCotIn` stops before the skip add (`mnv4SkipCotIn` applies it at the concrete row), and
the pre-strided chain is a near-copy rather than an instantiation — the leading depthwise's op and
the `2h` input run through every type.

⚠ The pre-strided profile needed its own chain but the head did NOT need its own tail:
`r34HeadCotBlk` and `r34HeadTiedB` are ResNet-34's GAP-and-dense pair, reused verbatim at 1280.

⚠ `mnv4PreStridedBodyOfRow` (+ `_faithful`) was added to `MobileNetV4BackB0.lean` as
`mnv4BodyOfRow`'s sibling — the row-typed section's missing third member. There is deliberately no
post-strided twin: Conv-M has no such row, so a row-typed wrapper for that arm would have no
possible argument.

✅ **Gates:** `lake build Certs` 3999 green, `git diff verified_mlir/` EMPTY, all twelve MNv4
declarations 3-axiom clean in `AuditAxioms`, `check_audit_coverage.py` green,
`docstring-checkrefs` 1681 citations resolve. ⭐ And the T2 graph was checked against the committed
bytes: all **247** SSA names it writes appear in `mnv4_fwd.mlir`, covering all **233** of its
declared parameters — nothing missing in either direction.

### Session 3 (T6) — what it cost

⛔⛔⛔ **THE HEADLINE, and it is session 1's lesson at its sharpest: peeling ONE `CertLayer.comp`
to reach `.fwd` at MNv4's literal resolutions is a kernel deterministic timeout in every
spelling that does the peel HERE, and 2 seconds in the one that applies a lemma proved between
VARIABLES.** Measured on the head stage, ~60 s each to give up:

| spelling | verdict |
|---|---|
| `rfl` | ⛔ kernel deterministic timeout |
| `simp only [<the def>, CertLayer.comp_fwd, Function.comp_apply]` | ⛔ timeout — the `comp_apply` step is what does it |
| `simp only [..., Function.comp_assoc]` (Mathlib's) | ⛔ timeout, at every group of three or more |
| `simp only [<the def>, CertLayer.comp_fwd]` with the RHS LEFT-nested | ✅ 2 s |
| `simp only [<the def>, certLayer_comp_fwd_apply]`, the generic `rfl`-at-variables lemma | ✅ 2 s, all 26 stages |

▶ So the shape check is built compositionally out of six generic projection lemmas
(`certLayer_comp_fwd_apply`, the three layer `_fwd_apply`s, `r34HeadB_apply`, `mnv4Chain_apply`),
five per-group expansions, and a group-granularity `rfl` — seven `rw`s in the capstone and not one
peel discharged at a literal width. ⭐ That is the same shape as T2's five group-faithfulness
proofs, and it is the general answer for this net: **prove it where the terms are variables, then
apply.**

⭐⭐ **NOT ONE new float leaf.** MNv4's stem is EfficientNet-B0's (XLA-`SAME` 3×3/s2, so
`floatBridgesTo_flatConvStride2XlaBack` and the `decimateOddBack` scatter), its two head convs are
plain 1×1s (`floatBridgesTo_convBack`), and its GAP-and-dense tail is ResNet-34's
(`floatBridgesTo_gapBack`, `floatBridgesTo_linBack`). Every concrete endpoint is `batchMap N` of a
per-example leaf, so the whole float file is `FloatBridgesTo.batchMap` and `.comp`. What is new is
the two stage TIES — `mnv4StemBBack_eq_vjp_backward` (relu + XLA, one token from MobileNetV2's
relu6, ResNet-34's symmetric and B0's swish) and `cbReluBBack_eq_vjp_backward` (the stride-1 peer,
applied twice at the head) — each `rw` of a conv leaf tie and then `rfl`.

⚠⚠ **Two elaboration traps that cost a 9-minute build each, both of them silent.**
1. `mnv4B_full_has_vjp_at` is `HasVJPAt`, so its `.backward` takes only the cotangent. Writing
   `.backward x` (B0's spelling, whose apex is a GLOBAL `HasVJP`) is a type error 700 lines deep.
2. **A missing `rw` does not fail — it GRINDS.** Omitting `r34HeadBBack_eq_vjp_backward` from the
   tie's rewrite list left the GAP-and-dense tail untied, and the closing `rfl` spent 2 000 000
   heartbeats on `isDefEq` before giving up. With it, the whole file is 3 s. ⭐ Read a timeout in
   a `rfl` that closes a tie as "one endpoint is not rewritten yet", not as "this is too big".
3. ⚠ And a third, cheaper: `rw [mnv4StemBBack_eq_vjp_backward (by decide) (by decide) … x h_stem]`
   is a `whnf` timeout because unifying `Vec (N * (3 * 224 * 224))` against
   `Vec (?N * (?ic * (2 * ?h) * (2 * ?w)))` asks the elaborator to invert a multiplication. Pin
   `(N := N) (h := 112) (w := 112)`, as B0's tie does and R50's (whose `2 * (…)` nests unify
   syntactically) does not have to.

⭐ **Twenty-six stages, and the head is why.** stem, the fused stage, b1…b21, the two head convs,
and ResNet-34's GAP-and-dense tail. R50/B0/MNv2 all sit at eighteen-to-twenty-one; MNv4 needed its
own apex, and only because Conv-M's ladder is longer — the construction is `r34B_full_has_vjp_at`'s
line for line.

⭐ **The shape check is where the block TABLE reaches T6.** Every one of the twenty-six slots names
its row (`mnv4Row4` vs `mnv4Row5` vs `mnv4Row10`), which is the only thing that tells the
shape-identical rows apart — the same job T2's SSA names do in the graph.

⛔ **Stated TO THE IMAGE**, per §(2) below: the chain's last node is one
`flatConvStride2Xla_has_vjp.backward` past anything the artifact computes, because no render emits
a gradient into `%x`. The file header says so in as many words.

✅ **Gates:** `lake build Certs` 4003 green, `git diff verified_mlir/` EMPTY, all seven new
declarations 3-axiom clean in `AuditAxioms`, `check_audit_coverage.py` and
`check_render_coverage.py` green, `docstring-checkrefs` 1689 citations resolve. ⚠ And
`MobileNetV4BackB0.lean`'s `## Scope` — the paragraph §0 caught being stale once — was updated
again in the same commit, this time to point at the four net-level files that now exist.

---

**The one-paragraph version, 2026-09-07 — ✅ CLOSED.** `Nets/MobileNet/MobileNetV4BackB0.lean` was
complete at the block and stage level and is now consumed by four net-level files: T1 and T2 in
`Nets/MobileNet/MobileNetV4FullB.lean` + `MobileNetV4FullBVJP.lean`, T3 in
`Nets/MobileNet/MobileNetV4FaithfulPoCB.lean` + `MobileNetV4TiePoCB.lean`, T6 in
`Nets/MobileNet/MobileNetV4WholeBackCertifiedTieB.lean` + `Float/MobileNetV4WholeBackFloatBridgeB.lean`.
T4 and T5 are float budgets and that thread is closed. ⚠ What MNv4 still does not have, and no
other net lacks, is a quoted accuracy: Conv-M has no Imagenette run and no verified ImageNet run,
so every tier is stated at artifacts no number comes from until a GPU run — the user's call, §2's
session 4 — is made. Every new file's header says so.

▶ **Everything below this line is the ORIGINAL SCOPE**, kept as the record of what was predicted.
Where a session found it wrong, the PROGRESS section above is what happened.

## 0. What the scoping row got wrong, and the standing facts

⛔⛔ **§3.6 of `proofs_tier_to_paper_nets.md` said the first session must build "the three
pieces `MobileNetV4BackB0.lean`'s own header names as missing — the fused stage's VJP and backward
graph, the head's, and the strided UIB body assembled from the stages already there." All three
exist.** `mnv4FusedStage` / `mnv4FusedStage_faithful` (swish, globally certified,
`stemBackBatchedGraph_faithful` underneath), `mnv4Head` / `mnv4Head_faithful` (conv-bn-relu →
GAP → dense), `mnv4UibPreStridedBody` / `mnv4UibPostStridedBody` with their `_faithful`s — all in
that file, all in `tests/AuditAxioms.lean` (five prints, 3-axiom clean). What is stale is the
file's OWN `## Scope` paragraph ("⚠ Not built: the fused stage … and the head"), which
`planning/archive/mnv4_verified.md` §8d records closing the same day and nobody edited. The row copied
the header. ▶ **Seventh instance of §5's rule** ("an audit's named gap can be the wrong one"),
and the cheapest: the declaration list was one `grep` away. Session 0 fixes the header.

**Standing facts, each read from the file named.**

| fact | where |
|---|---|
| ONE chain: `mnv4FwdChainB` is the traversal `@mnv4_fwd`, `@mnv4_fwd_eval` (`BnMode`) and the train step all use; 4c never applied to this net | `MobileNetV4RenderB.lean` (its `Mnv4FwdRec` docstring) |
| Batch BatchNorm throughout (`bnBatchF` token at `mode := .train`, width `N·h·w`); the artifacts' `N` is per replica | `mnv4Bn`; `DataParallel.lean` |
| stem: 3×3/s2 conv **XLA-SAME** (`.convStridedXla`, 3→32, 224→112) → BN → **relu**; the only XLA site in the net | `mnv4FwdChainB`; `scripts/convention_audit.py` (`mnv4 … fixed (convStridedXla)`) |
| fused stage (stage 0): 3×3/s2 conv **symmetric** 32→128, 112→56 → BN → **swish** → 1×1 128→48 → BN, no skip | `fusedMbConvFwdStridedB`; `MobileNetV4BackB0.lean` §"THE FUSED STAGE" |
| 21 UIB blocks, **relu** (not relu6), all convs bias-free (`%zb{c}`): 13 ExtraDW / 4 ConvNeXt-like / 4 FFN / **0 IB**; 18 skip (`ic = oc`, stride 1) + 3 pre-strided (blocks 1, 3, 11) + **0 post-strided** | `mnv4Blocks`; the `#guard`s at the bottom of `MobileNetV4BackB0.lean` (Conv-M, timm 1.0.28) |
| ladder 56 → 28 → 14 → 7 / 48 → 80 → 160 → 256; expand ratios 2/4/6; kernels 3/5 | `mnv4Blocks` |
| head: 1×1 256→960 → BN → relu → 1×1 960→1280 → BN → relu → GAP(7) → dense 1280→K — **TWO** head convs | `mnv4FwdChainB` (`%h1W`, `%hW`); `MobileNetV4BackB0.mnv4Head` models ONE conv stage |
| census: **233** parameter slots at K = 10 (`mnv4ShapeList`, no `convBias` flag, so 233 of 233 are exercised), 154 running-stat slots; 8,447,322 scalars at K = 10, 9,715,512 at K = 1000 | `#guard (mnv4ShapeList 10).length == 233`; `mnv4_convm_ties_todo.md` |
| artifacts (9, one writer): `mnv4_fwd`, `mnv4_fwd_eval`, `mnv4_adam_train_step` (B 32, K 10); `mnv4in_fwd`, `mnv4in_fwd_eval`, `mnv4in_adam64`, `mnv4in_adam64bf16`, `mnv4in_adamdp64` (4 × 64), `mnv4in_adamdp64bf16` (K 1000) | `verified_mlir/`, `proofs.yml`'s `MobileNetV4RenderB` step |
| gradient constructors the backward emits: `bnGammaGradB`/`bnBetaGradB` ×16 sites, `convWeightGradB` ×9, `depthwiseWeightGradB` ×3, `depthwiseStridedWeightGradB` ×2, `convStridedWeightGradB` ×1 (fused, symmetric), `convStridedXlaWeightGradB` ×1 (stem), `denseWeightGradB`/`denseBiasGradB`; each conv/depthwise kind has a `*Bf16` twin on the bf16 path | `MobileNetV4RenderB.lean` (grep, 2026-09-07) |
| backward ops: `bnBatchBack`, `selectPos` (relu), `swishBackB` (fused stage only), `convBackBatched`, `depthwiseBackBatched`, `depthwiseStridedBackBatched`, `convStridedBackBatched` (+ bf16 twins); **no `dx` into `%x`** — the stem has no input-VJP token, as B0's XLA stem has none | same |
| the all-reduce is `allReduceMeanF` (4d.2); `mnv4-dp-check` fp32 + bf16 and `shard-check` all green with controls, 2026-08-27, 4 GPUs (the DP render is 4-replica only) | `runs/2026-08-27-mnv4-dp-shard-gates/README.md` |
| ⚠ **empirical ties are OWED for Conv-M**: the forward tie (1.423e-6) and gradient tie (0/147) were at the Conv-S table; `mnv4_forward_tie.py`'s reference `.py` is stale and `grad_tie.py`'s `NETS["mnv4"]` still says `nparams=158, nstats=104` (must be 233 / 154) | `planning/archive/mnv4_convm_ties_todo.md` (2026-08-14, still open) |
| ⚠ **no quoted number**: Conv-S Imagenette 87.36% (superseded spec, kept in `RESULTS.md` as such); Conv-M has no Imagenette run; the verified ImageNet port (`mobilenetv4-imagenet-verified`, `scripts/jobs/mnv4-default-4gpu.conf`) has not run; the JAX Conv-M reference is 75.51% (outside the repo, `/home/skoonce/mnv4_convm_100ep`) | `RESULTS.md` §MobileNetV4-Conv-S; `VerifiedNets.lean` ("no Imagenette accuracy run of its own yet"); memory |
| yaml: **zero** rows for `MobileNetV4BackB0.lean` (the audit has five prints) | `grep MobileNetV4BackB0 formalization.yaml` |

## 1. What exists one tier down — read `MobileNetV4BackB0.lean` by its declaration list

| piece | declaration | certified how |
|---|---|---|
| depthwise-bn-relu stage, stride 1 and 2 | `dwbReluB`, `dwbReluBstrided`, `*_has_vjp_at`, `*BackBatchedGraph_faithful` | `bnReluStage_has_vjp_at` at `depthwiseFlat` / `depthwiseStride2Flat` — one instantiation each |
| the four stage `CertLayer`s | `mnv4DWReluLayer`, `mnv4DWReluStridedLayer`, `mnv4ExpandLayer` (= `cbReluB`), `mnv4ProjectLayer` (= `projB`, `ok = True`) | r34's stages, reused |
| the family collapse | `mnv4UibBody preDW expand postDW project`; `mnv4UibSkipBlock` = `CertLayer.residual` of it; `mnv4Family{IB,ConvNeXt,FFN}` | `id'` in an absent slot; both depthwise positions are shape-preserving |
| the stride-2 forms | `mnv4UibPreStridedBody` (blocks 1, 3, 11 in Conv-M), `mnv4UibPostStridedBody` (no Conv-M row) + `_faithful` | the stride is in the TYPE, so `id'` cannot collapse these |
| the fused stage | `fusedConvB` (symmetric strided conv-bn-swish), `stemBackBatchedGraph_faithful`, `mnv4FusedConvLayer`, `mnv4FusedStage`, `_faithful` | global `HasVJP` — swish is smooth, `ok = True` |
| the head | `mnv4GapLayer`, `mnv4DenseLayer` (both `faithful := rfl`), `mnv4Head headConv gap cls`, `_faithful` | ⚠ ONE conv stage; the render has two — compose `mnv4ExpandLayer` twice |
| table-driven dispatch | `mnv4PreDWSlot` / `mnv4PostDWSlot` (`if k = 0 then id'`), four `@[simp]` lemmas, `mnv4UibSkipBlockOfKs` | the proof runs the render's `k = 0` rule off `mnv4Blocks` |
| row-typed weights | `structure UibParams (s : UibSpec)` — every width a projection of `s`; `mnv4BodyOfRow N s p`, `_faithful` | a record whose widths disagree with its row cannot be constructed |
| the ladder | `mnv4BlockLadder` (blk1 · blk2 · blk3 · mid14 · blk11 · tail) — the six-slot shape fits Conv-M (48/80/160/256, reductions at 1, 3, 11) | type-level check on the table |
| the table guards | 21 rows, family sequence, 13/4/4/0, 18/3/0, the `h` and `stride2` lists, `ic ≠ oc ⇔ stride2` | derived from timm, NOT re-read off the table |

⚠ Two docstrings in that file are Conv-S: `mnv4Stage14` ("blocks 4–10 … ExtraDW (4, 5, 10),
ConvNeXt (6, 8), IB (7), FFN (9)") and the `mnv4UibSkipBlock` comment ("11 of MNv4's 14 blocks").
The `#guard`s below them are Conv-M. Fix or delete `mnv4Stage14` in session 0; it is a type-level
demo with no consumer.

⚠ **What the row typing does NOT pin, in the file's own words**: rows 4, 5 and 10 are all
`160 → 160, expand 4` with 4/10 sharing `k = 3, 3`, so their records are the same TYPE and swapping
their weights typechecks. Identity between shape-identical rows is pinned by NAMES — `%u4qW` vs
`%u10qW` — which is exactly what T2 (the graph at the render's tokens) and T3 (the tie at the
emitted nodes) state. That is the point of those tiers here, not a formality.

## 2. The order of work

Four sessions, each one commit, each with a ResNet-50 file to mirror. ResNet-50 is the precedent
for a reason: it too had only a batched renderer, only block-level proofs, and no per-example
legacy — §3.5a–d of `proofs_tier_to_paper_nets.md` closed T1/T2/T3/T6 in four commits on
2026-09-06/07, and the write-ups there are the most recent record of what each tier costs.

### Session 0 — housekeeping and the owed measurement (half a session; the measurement is a GPU ask)

(a) **Fix the record.** Rewrite `MobileNetV4BackB0.lean`'s `## Scope` paragraph to say what §1
above says; fix or delete `mnv4Stage14`; correct the "11 of 14" comment. Add the yaml rows the
audit already prints — `mnv4BodyOfRow_faithful` (the row-typed block theorem),
`mnv4FusedStage_faithful`, `mnv4Head_faithful` — under a comment that names the collapse and the
Conv-M counts. Fix §3.6 of `proofs_tier_to_paper_nets.md` to point here and say why its (0) row was
wrong. No proof work; ~1 hour.

(b) ⚠ **ASK, then run the two owed ties** (`planning/archive/mnv4_convm_ties_todo.md` §1–2b has the
commands and the IREE pairing that actually works on this box — the repo `.venv` has no `iree` and
the scripts' default paths do not exist). Update `grad_tie.py`'s `nparams`/`nstats` to 233 / 154
in the same change. These are seconds-to-minutes on one GPU, but the box rule stands. ▶ They are
what pins block ORDER: a pre/post-DW swap is invisible to every type, count and `#guard` (the
render's own header says so), and every Lean tier below certifies the render AS SHIPPED — if the
shipped order were wrong, T1–T6 would be true about the wrong net. Do (b) before quoting any tier
as "at the paper net"; it does not block writing them.

### Session 1 — T1 (forward + whole-net VJP) and T2 (typed forward graph) — mirror `ResNet50FullB.lean` + `ResNet50FullBVJP.lean` (§3.5b)

**The forward.** `Nets/MobileNet/MobileNetV4FullB.lean`: a weights record `Mnv4BWeights nCls` —
stem (`sW : Kernel4 32 3 3 3`, `sγ sβ : Vec 32`, `sε`), fused (`fW : Kernel4 128 32 3 3` + BN,
`fpW : Kernel4 48 128 1 1` + BN), twenty-one row-typed fields `b1 : UibParams mnv4Row1` …
`b21 : UibParams mnv4Row21`, head (`h1W : Kernel4 960 256 1 1` + BN, `hW : Kernel4 1280 960 1 1` +
BN, `Wd : Mat 1280 nCls`, `bd`). ⭐ `UibParams` already exists and is STRONGER than R50's
`R50IdW`/`R50ProjW`: a record that disagrees with its row is unconstructible, so the forward needs
no side conditions on widths. ⚠ **Name the rows.** Define `mnv4Row1 : UibSpec := ⟨"1", 48, 80, 4,
3, 5, 28, true⟩` … `mnv4Row21` as top-level constants and `#guard mnv4Blocks = [mnv4Row1, …,
mnv4Row21]`; do NOT write `UibParams (mnv4Blocks[0]!)` in a type — the list index in a type forces
`whnf` through `List.get!` at every use and the elaborator will not thank you.

`mobilenetv4ForwardB_full (N : Nat) {nCls} (w : Mnv4BWeights nCls) : Vec (N * (3*224*224)) → Vec
(N * nCls)` as `head ∘ b21 ∘ … ∘ b1 ∘ fused ∘ stem`. Two routes for the block forwards, and the
recommendation is the second:

* (i) R50's: explicit per-block forward functions (`uibSkipB`, `uibPreStridedB`) over the record,
  then a hand-written apex for the VJP. R50 went this way because its `CertLayer` trunk
  (`ResNet50BackNet.lean`) predated the tie files' needs.
⛔⛔ **BOTH ROUTES BELOW ARE SUPERSEDED — read the PROGRESS section at the top of this file
instead.** Route (ii) is right *per resolution group* and wrong *for the whole trunk*: composing
the groups into one `mnv4NetLayer` elaborates and then makes every later use unpayable, because
peeling `CertLayer.comp` to reach `.fwd` at MNv4's literal resolutions costs ten minutes and a
kernel timeout in all four spellings tried. What shipped is five `CertLayer` groups joined by
SEVEN NAMED PREFIXES — route (i)'s shape at the top, route (ii)'s inside each group. The text
below is kept as the record of what was predicted.

* (ii) ⭐ **the `CertLayer` route**: `mnv4NetLayer N w : CertLayer (N*(48*56*56)) (N*nCls)` :=
  `mnv4FusedStage … |>.comp (residual (mnv4BodyOfRow N mnv4Row2 w.b2)) |>.comp … |>.comp
  (mnv4Head …)` — the 18 skip rows as `CertLayer.residual (mnv4BodyOfRow …)` (the `oc = ic` `#guard`
  is `rfl` at each literal row), the 3 strided rows as `mnv4UibPreStridedBody` at their slots.
  Its `.fwd` IS the forward from the fused stage down, its `.vjp` the whole-trunk `HasVJPAt` under
  `.ok`, and its `.faithful` the whole-trunk backward-graph faithfulness — for free, no apex, no
  `r50Pre_k` chain. The stem sits OUTSIDE it (see the next paragraph), composed by `vjp_comp_at`.

⚠⚠ **The stem cannot be a `CertLayer`, and this is B0's situation exactly.** `CertLayer` demands a
backward graph, and no render emits a gradient into `%x` — there is no `convStridedXlaBackBatched`
token. B0's `enetTrunk` takes its stem as a PARAMETER for this reason and B0's stem is
"un-graph-certified" (`MobileNetV4BackB0.lean`'s fused-stage docstring records it). So: the stem's
VJP is `bnReluStage_has_vjp_at N (flatConvStride2Xla sW z) …` (one instantiation, the shape
`dwbReluB` took; `flatConvStride2Xla_has_vjp` and `_differentiable` exist from the re-spell
thread), and `mobilenetv4ForwardB_full_has_vjp_at` is `vjp_comp_at` of that and
`(mnv4NetLayer …).vjp`. ⚠ The stem's kink clause (`∀ k, bnBatchLA … ≠ 0`) and the trunk's `.ok`
are the whole hypothesis; write `Mnv4SmoothAt N w x` as a structure bundling them, R50's
`R50IdSmoothAt` style, so the capstone binds one bundle. ⚠ `0 < ε` for each of the 16 + 4 BN sites
is inside `UibParams` already (`hq he hd hz`); the stem's, fused's and head's need fields.

⚠ **Two head convs.** `mnv4Head headConv gap cls` takes one conv stage; the render has
`%h1W` (256→960) then `%hW` (960→1280). Compose `mnv4ExpandLayer` twice —
`(mnv4ExpandLayer … h1W …).comp (mnv4Head N (mnv4ExpandLayer … hW …) gap cls)` — or generalise
`mnv4Head` to two. Either way the SSA names in T2 must be `%h1c/%h1n/%h1r` and `%hc/%hn/%hr`
(`Mnv4FwdRec`).

**The graph (T2).** `mnv4FwdGraphB_full N epsStr w : SHlo (N * nCls)` at `mnv4FwdChainB`'s tokens,
built per row and chained, with `mobilenetv4FwdGraphB_full_faithful : den graph = forward`.
Mirror `r50IdGraphB` / `r50IdGraphB_faithful` per block kind. The token sequence per stride-1 row
(`uibFwdSkipB`, read it): `[batchOp depthwise → bnBatchF → batchOp relu]` if `preDWk > 0`, then
`batchOp conv → bnBatchF → batchOp relu`, then `[batchOp depthwise → bnBatchF → batchOp relu]` if
`postDWk > 0`, then `batchOp conv → bnBatchF → addVB(·, xin)`. ⚠ The graph builder must dispatch on
`s.preDWk = 0` exactly as `mnv4PreDWSlot` does — no tokens for an absent depthwise — and the
`UibParams` record's degenerate `DepthwiseKernel s.ic 0 0` is simply not read on that branch. The
strided rows use `.depthwiseStrided` at the pre slot (`uibFwdPreStridedB`); the fused stage
`.convStrided` (symmetric) → `bnBatchF` → `batchOp swish` → `.conv` → `bnBatchF`
(`fusedMbConvFwdStridedB`); the stem `.convStridedXla` → `bnBatchF` → relu. ⚠ Check `bnBatchF`'s
`den` is `bnBatchLA` at `(oc, h, w)` before assuming — `mnv4Bn` passes `(h := h) (w := h)`.

**Acceptance.** `#print axioms` 3-axiom on `mobilenetv4ForwardB_full_has_vjp_at` and
`mobilenetv4FwdGraphB_full_faithful`; a `_eq_chain`-style shape check that the record's field
order is `mnv4ShapeList`'s; the committed `mnv4_fwd.mlir` signature is `%x` + 233 (K = 10) and
every name the graph writes appears in it (R50's §3.5b did this by reading the artifact — do the
same); yaml rows; `git diff verified_mlir/` EMPTY (this is proof work; no artifact may move).
**Cost:** R50's T1 + T2 was one session (465 + 482 lines) with a hand-written apex; the `CertLayer`
route removes the apex and the sixteen `r50Pre_k` names but adds the stem composition. One session.

### Session 2 — T3: the fold at the gradient nodes, then the tie — mirror `ResNet50FaithfulPoCB.lean` + `ResNet50TiePoCB.lean` (§3.5c)

**The fold — no new op lemma.** Every gradient kind the backward emits has its batched `den`
lemma already: `ResNet34PoCB.convWGradB_den`, `bnGammaGradB_den`, `bnBetaGradB_den`,
`denseWGradB_den`, `denseBGradB_den`, `convStridedWGradB_den` (the fused stage, symmetric);
`EnetPoCG.depthwiseWGradB_den`, `depthwiseStridedWGradB_den` (symmetric strided depthwise — B0's
op, and MNv4's UIB strides are symmetric too), `convStridedXlaWGradB_den` (the stem — B0's XLA
stem op, identical). So `Nets/MobileNet/MobileNetV4FaithfulPoCB.lean` is per-block-PROFILE capstones
in R50's shape (`r50IdGradsCertified` …): `mnv4UibGradsCertified` for the skip row at its two
`k`s (the conjunct list must dispatch on `k = 0` — no conjunct for an absent depthwise, exactly as
the render emits none), `mnv4UibPreStridedGradsCertified`, `mnv4FusedGradsCertified`,
`mnv4StemGradsCertified`, `mnv4HeadGradsCertified` (two convs + dense).

⭐ **And the bf16 nodes, stated — this is where the "bring bf16 in line" item lands for MNv4.**
`mnv4in_adam64bf16` / `adamdp64bf16` emit `convWeightGradBBf16`, `depthwiseWeightGradBBf16`,
`depthwiseStridedWeightGradBBf16`, `convStridedWeightGradBBf16` and `convStridedXlaWeightGradBBf16`.
Three of those lemmas exist in `ConvNeXtFaithfulPoCGB.lean` (`convWGradBBf16_den`,
`depthwiseWGradBBf16_den`, `convStridedWGradBBf16_den`) and are generic — cite them. Two are new
and four lines each on the same template: `depthwiseStridedWGradBBf16_den` and
`convStridedXlaWGradBBf16_den` (`simp only [den]; congr 1; apply Finset.sum_congr rfl; intro n _;
exact <the strided/Xla VJP>.correct …` at rounded slices — read the two `den` arms in
`StableHLO.lean` first; the strided depthwise one is `Tensor3.flatten (…backward W (unflatten …))`
shaped like `EnetPoCG.depthwiseWGradB_den`'s). ⛔ Do not write "the bf16 twins consume the same
node" anywhere; that sentence is what §4c-quater found loose in three other folds.

**The tie.** `Nets/MobileNet/MobileNetV4TiePoCB.lean`: per-block tie defs with the cotangent chain
built from the CERTIFIED block VJPs — `mnv4BodyOfRow_faithful` IS the `den graph = vjp.backward`
fact the `*CotIn_eq_vjp` lemmas of 4.2a/4.2c/§3.5c re-state, so the cross-block chain composes
certified VJPs rather than re-deriving. ⭐ `g` is a BINDER (4b's rule); `N` and `nCls` are binders;
no `q` (MNv4 ships one resolution). ⚠ **Check which loss chain the render emits before choosing
the loss lemma**: `mobilenetv4AdamTrainStepFaithfulB` is modelled on `ResNet34RenderB`, so expect
`softmaxRow` at `N·(1·K)` and `smoothedLossCotGraph` with `rowB`/`unrowB` — but READ the lines; if
it is `expe → softmaxDiv` at `N·K` it is `smoothedLossCotGraphDiv` (§4b.6). The capstone
`mnv4_net_tiedB (N) {nCls} … (w : Mnv4BWeights nCls) (x) (t)` threads the forward prefixes and the
top-down cotangents through 24 stages; R50's `r50_net_tiedB` at 18 is the template line for line.
Then `mnv4_lossCot_is_smoothedCE_grad`.

**Census and acceptance.** 233 conjunct slots at K = 10 and every one exercised — bias-free by
construction (`%zb`, no `convBias` flag), which is R50's "161 of 161" situation and not r34's
"146 states, 110 exercises". Write the conjunct count in the file header FROM the artifact
(`grep -c` the parameter names in `mnv4_adam_train_step.mlir`), never from a docstring. Gates as
session 1's, plus the tie's `#print axioms`. **Cost:** R50's T3 was one session (fold + 895-line
tie). MNv4's is the same shape with 21 row-blocks instead of 16; budget one session, possibly a
long one — the bf16 lemmas are the only new mathematics and they are four lines each.

### Session 3 — T6: the certified backward tie ▶ **REWRITTEN 2026-09-07 after sessions 1–2; the original text is corrected in three places**

**Mirror `Resnet50WholeBackCertifiedTieB.lean` + `Float/Resnet50WholeBackFloatBridgeB.lean`
(§3.5d). Read `EfficientNetFullWholeBackCertifiedTie.lean`'s header first — it answers this
section's one open question, see (2).**

R50's T6 was four declarations and ~3 s: a float chain naming the input-gradient term
(`r50InputGradB`, `r50InputGradBF`, `r50_grad_floatBridgesToB`), a tie of that chain to the
certified whole-net VJP with the blocks OPAQUE (`r50InputGradB_eq_r34B_full_vjp`), its `pdiv`
reading (`r50InputGradB_correct`) and the `_eq_slots` shape check. MNv4 is the kinked kind (relu),
so §5's pricing applies: generic tie + `_eq_slots`, and ⛔ do NOT price B0's `backward_unique`
step — that lemma is about the global `HasVJP`, and MNv4's whole-net witness is `HasVJPAt`.

#### ⛔⛔ (1) What the original said about the apex is WRONG, because T1 did not go that way

It read: *"If T1 took the `CertLayer` route, the apex is `(mnv4NetLayer …).vjp` composed with the
stem's VJP."* **There is no `mnv4NetLayer`.** Composing the trunk into one `CertLayer` elaborates
and then makes every later use unpayable — all four spellings of peeling `CertLayer.comp` to reach
`.fwd` cost ten minutes and a kernel timeout at MNv4's literal resolutions. T1 ships **seven named
prefixes** instead (`mnv4Pre0 … mnv4Pre6`, one per resolution group), and
`mobilenetv4ForwardB_full_has_vjp_at` is a bottom-up `have` chain of seven `vjp_comp_at`s over
them, binding an eight-field `Mnv4SmoothAt`. ▶ **That is the apex T6 composes with** — and it is
ResNet-50's own shape at seven stages, so §3.6's "needs a wider peer" note and the
fourth-consumer argument for a generic apex are both retired for this net, just not for the reason
the original gave.

⚠ T3's tie also ships a FINER chain — `mnv4Blk0 … mnv4Blk21`, one prefix per block
(`MobileNetV4TiePoCB.lean`). T6 wants the block granularity for its opaque slots, so use those,
not `mnv4Pre_k`.

#### ⭐ (2) The open design question is ANSWERED: follow B0, state it to the IMAGE

The question was whether MNv4's float chain should end at the stem-conv INPUT (the mathematical
input gradient) or at the stem-BN cotangent (what the artifact actually has), given that no render
emits a gradient into `%x`. **B0 has the identical stem and states it to the image**:
`EfficientNetFullWholeBackCertifiedTie.lean`'s apex is `head ∘ b16 ∘ … ∘ b1 ∘ stem`, so
`efficientnetInputGradB_full` runs through the stem. ▶ Do the same, and say plainly in the header
that the last node is **one `flatConvStride2Xla_has_vjp.backward` past anything the artifact
computes** — the artifact's backward ends at the stem conv's WEIGHT gradient, whose operand is the
stem-BN cotangent, which `mnv4StemTiedB` already ties. Either choice is honest if the header says
which; the wrong move is to state one and describe the other.

#### ⚠⚠ (3) Apply session 1 and 2's ONE lesson before writing a line

**A net whose resolutions are LITERALS cannot afford the proof idioms a net with a resolution
BINDER can.** ResNet-50's `q` keeps `den` and every width-indexed `rfl` stuck; MNv4's
224/112/56/28/14/7 let them run into terms with hundreds of thousands of elements. That cause
produced **six** distinct blow-ups across sessions 1–2, each of which looked like a different
problem. So, for T6:

* state every float leaf and every tie lemma **generic in its widths** (or in the `UibSpec` row),
  and instantiate at the concrete rows only in the capstone — instantiating a proven lemma is free;
* prefer `rw` to `simp only` in anything whose goal mentions `den` at concrete widths;
* never let `CertLayer.comp` be peeled under a `den`; and
* if a term needs a subtree twice, hide the duplication behind a folded combinator with its own
  one-step lemma (`mnv4SkipGraphB` / `mnv4SkipCotIn` are the two precedents).

#### What exists for T6 now, and what is genuinely MNv4's own

| piece | state |
|---|---|
| whole-net forward + `HasVJPAt` + `pdiv` reading | ✅ `MobileNetV4FullBVJP.lean` |
| per-block certified VJPs, opaque slots | ✅ `mnv4BodyOfRow` / `mnv4PreStridedBodyOfRow` and their `_faithful`s |
| per-block forward prefixes | ✅ `mnv4Blk0 … mnv4Blk21` (`MobileNetV4TiePoCB.lean`) |
| the stem-BN cotangent the artifact's backward really ends at | ✅ `mnv4StemCotN` / `mnv4StemCotC`, tied |
| XLA strided conv float leaves | ✅ from the re-spell thread |
| batched BN float leaves | ✅ `bnBatchFloatBridge`, from 4.1 |
| **MNv4's own endpoints** | ⛔ the stem/head float leaves and their ties. R50 got these free because its stem and head ARE `r34StemB`/`r34HeadB`; MNv4's stem is conv-bn-relu at XLA-SAME with no pool, and its head is TWO convs before GAP. ⚠ But the GAP-and-dense tail IS ResNet-34's — T3 reused `r34HeadCotBlk`/`r34HeadTiedB` verbatim — so only the two 1×1 conv-BN-relu stages in front of it are new. |

**Files:** `Float/MobileNetV4WholeBackFloatBridgeB.lean` mirrors `Resnet50WholeBackFloatBridgeB.lean`,
with `EfficientNetFullWholeBackFloatBridge.lean` as the shape for the depthwise-heavy float chain;
`Nets/MobileNet/MobileNetV4WholeBackCertifiedTieB.lean` is the tie.
**Cost:** one session. R50's was four declarations; MNv4's own endpoints make it more like B0's.

### ⛔ T4 / T5 — do not write

`planning/archive/float_budget_numbers.md` closed the float-budget thread as vacuous by user decision on
2026-09-05. Expect a fold (relu has no clamp, so the window is r34-sized) at batch BN with two
depthwises per block — a probe row in `scripts/float_budget_envelope.py` is cheap and says what
the number would be; the Lean is not to be written.

### Session 4 — the number (a DECISION, GPU, ask first)

MNv4 is the one net whose tiers, once written, are stated at artifacts NO quoted accuracy comes
from. Two ways to change that, both runs: (a) the Conv-M Imagenette run (`mobilenetv4-verified-
adam`, 80 ep, bs 32, one GPU — Conv-S took ~63.5 h single-card by the DP README's figure, Conv-M
is 2× the parameters); (b) the verified ImageNet port (`mobilenetv4-imagenet-verified` with
`scripts/jobs/mnv4-default-4gpu.conf`, 4 GPUs, roughly a quarter of 63.5 h at Imagenette scale
— ImageNet scale is the 100-epoch reference's ~15 h × the verified-vs-JAX ratio; measure one epoch
first). (b) is the one the tiers would be quoted against (the ImageNet artifact, comparable to the
75.51% reference). ⚠ ASK before either; the box has crashed on long runs. Nothing in sessions 1–3
depends on it; every tier's header must say "no accuracy is quoted for this net yet" until it
lands.

## 3. Traps — inherited ones first, then MNv4's own

Inherited, all previously paid for (`proofs_tier_to_paper_nets.md` §5): `lake build Certs`, never
bare `lake build`; `lake env lean tests/AuditAxioms.lean`, `docstring-checkrefs`,
`check_audit_coverage.py` per commit; a new module needs a `lakefile.lean` `Certs` root line, an
`AuditAxioms` block and yaml rows with one-line comments (no emoji in the yaml); write a `2 * (…)`
nest, never `8 * h`; `Env`-as-structure and a bottom-up `have` chain; a kinked net's T6 stops at
opaque blocks; a row that says "X has no theorem" must NAME the theorem it looked for; grep the
other planning docs for a blocker before pricing on it; the box rule on GPU runs.

MNv4's own:

* ⛔ **relu, not relu6.** `MobileNetV2RenderB` and `MobileNetV2BackB0` sit one file over with
  relu6 (`.selectMid`, two kink clauses); MNv4 is `.selectPos`, one clause, everywhere except the
  fused stage (swish, none). `MobileNetV4BackB0` verified the swap fails; keep it that way.
* ⛔ **`k = 0` means NO tokens and NO conjunct**, and the `UibParams` record still carries a
  degenerate `DepthwiseKernel c 0 0` for that slot. Every T2 graph builder and every T3 profile
  must branch on `s.preDWk`/`s.postDWk` exactly as `mnv4PreDWSlot` does, off the SAME `mnv4Blocks`
  row. A conjunct stated for an absent depthwise is a `den` of a node the artifact does not have.
* ⛔ **Three pre-strided blocks, zero post-strided** in Conv-M. `mnv4UibPostStridedBody` is
  certified and unexercised; a green corpus is not coverage of that arm, and Conv-S (which had one)
  is no longer a shipped net.
* ⚠ **Padding phase per site**: the STEM is XLA-SAME (`flatConvStride2Xla`, `convStridedXla*`),
  the fused stage and all three strided depthwises are SYMMETRIC (`flatConvStride2`,
  `depthwiseStride2Flat`). One net, two phases, both correct; `scripts/convention_audit.py`
  reads them. Do not "tidy" one to match the other.
* ⚠ **Two head convs, one `mnv4Head`.** Compose, and name the SSAs as `Mnv4FwdRec` does.
* ⚠ **The fused stage is the only smooth stage** (`ok = True`, global `HasVJP`); every UIB stage
  and the stem and head convs are `_at`. So `(trunk).ok` is a ~60-clause conjunction (2 relu
  sites per ExtraDW/ConvNeXt/IB-shaped body, 1 per FFN, plus the head's two) threaded through 21
  activations — `CertLayer.comp` generates it; do not write it by hand.
* ⚠ **`mnv4Blocks[i]!` in a type** — name the rows (§2, session 1).
* ⚠ **Shape-identical rows are name-pinned only** (rows 4/5/10; 12/18; 15/19/20). T2's tokens
  carry `%u{p}…` and that is the pin. Say so in the T1 header rather than claiming the record
  pins block identity.
* ⚠ **The stale-header pattern.** `MobileNetV4BackB0.lean`'s `## Scope` was wrong for four weeks
  and was copied into a planning row. When a session lands a piece, edit the file header that
  said it was missing, in the same commit.
* ⚠ **Root-file edits**: none are expected — every op kind exists. If one turns out to be needed,
  park the lemma in the leaf with a note (§5's root-file cost rule); a `StableHLO.lean` edit
  rebuilds the corpus (~40 min, §4d.2).
* ⚠ **No artifact may move** in any of sessions 1–3. `git diff verified_mlir/` empty is a gate
  for every commit; the nine MNv4 artifacts are diffed in CI's `MobileNetV4RenderB` step already.

## 4. Files, by session

| session | new | edited |
|---|---|---|
| 0 | — | `Nets/MobileNet/MobileNetV4BackB0.lean` (header, `mnv4Stage14`), `formalization.yaml` (three rows), `planning/archive/proofs_tier_to_paper_nets.md` §3.6, `scripts/grad_tie.py` (233/154) with the tie runs |
| 1 | `Nets/MobileNet/MobileNetV4FullB.lean` (rows, record, forward, `mnv4NetLayer`, graph, `_faithful`), `Nets/MobileNet/MobileNetV4FullBVJP.lean` (stem VJP, `Mnv4SmoothAt`, `mobilenetv4ForwardB_full_has_vjp_at`) | `lakefile.lean`, `tests/AuditAxioms.lean`, `formalization.yaml` (rows + a status section), this doc |
| 2 | `Nets/MobileNet/MobileNetV4FaithfulPoCB.lean` (profiles + the two new bf16 lemmas), `Nets/MobileNet/MobileNetV4TiePoCB.lean` (`mnv4_net_tiedB`, the loss corollary) | same |
| 3 | `Float/MobileNetV4WholeBackFloatBridgeB.lean`, `Nets/MobileNet/MobileNetV4WholeBackCertifiedTieB.lean` | same; `MobileNetV4RenderB.lean`'s `#eval` block comment ("do not train off these") if the DP tie README has not already retired it |

Done when `proofs_tier_to_paper_nets.md` §2's MobileNetV4 row reads ✓ ✓ ✓ ✗ ✗ ✓ with T4/T5
marked "declined, thread closed" like the other nets' — and the header of every new file says
whether an accuracy is quoted for this net.
