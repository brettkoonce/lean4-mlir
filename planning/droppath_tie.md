# The mask chain: step ties through drop-path and classifier dropout

Written 2026-10-07, at the close of planning/bf16_tie.md (tier A done: every whole-net step, sync
and loss-gradient tie is stated at either precision). The table and site counts are checked
against the job confs, the manifest and the renderers; §1 is the first net landed and the recipe
the rest follow.

## ▶ Start here (next session): the indexed lift for ViT and ConvNeXt

**State (2026-10-07).** The two batched-chain nets are done on `wp8fg` (not pushed):
MobileNetV2's classifier dropout (`a442268c`) and EfficientNet-B0's nine drop sites + dropout (the
commit after it). `mnv2_net_tiedB` / `_syncTiedB` / `mnv2_net_lossGrad` take `cd`, and
`efficientnet_net_tiedG` / `efficientnet_net_syncTiedG` / `enet_net_lossGrad` take `sd : Option
(Fin 9 → Vec N)` and `cd : Option (Vec (N * 1280))` — `none` the statements exactly as they were,
`some` the `*do*` / `dropdo` chains the book's runs trained — so `mnv2-default-4gpu` is reached at
its gradient nodes with nothing of its render outside and `enet-default-4gpu` with only its EMA
tail outside. §1 and §2 record what the two steps took, including the two mask-SPELLING traps
(§2) that apply to every net from here on.

**First, the book (one small commit, preview on :8766).** The Lean moved; these sentences have
not: MNv2 — `thm:mobilenetv2_step_tie` / `_loss_grad` texts (~7636–7696, "at either precision as
for ResNet-34" → and at its classifier-dropout site, the loss of `mobilenetv2ForwardBFullDoOpt`),
`thm:mobilenetv2_sync_tie` (~7801–7826), the run paragraph's "its classifier dropout sits outside
that statement" (~8363); B0 — the forward sentence stays (~9138), the step-tie text "on the chain
without stochastic depth or classifier dropout; the optimizer tails … outside" (~9198 → at the
renderer's two sites, the EMA tail outside), the sync-tie and loss-gradient texts, the run
paragraph (~9683); the trust summary's "on the chain without drop-path" (18760, 18802 → "without
drop-path" now names ViT-Tiny and ConvNeXt-T only). Then blueprint `lean_decls` regenerated +
`blueprint_uses.py --check`, `docstring-checkrefs`.

**Then the thread's one piece of new Foundation: the indexed lift.** ViT's and ConvNeXt's ties
are lifts of ONE per-example map — `batchMap N f` for the forwards, `batchMapAux N f saved` for the
cotangents (`Foundation/Batched/Basic.lean`), `HasGradAt.param_batchMap_through`
(`Foundation/ParamGrad.lean`) for the loss gradient, `batchShard_batchMap` / `_batchMapAux`
(`DataParallel/Sync.lean`) for sharding — and a drop mask makes example `t`'s block a DIFFERENT
map (`blockVDrop … (sdA i t) (sdM i t)`). What exists on the masked side: ViT's forward
`vitForwardKVDropB` is defined pointwise (`fun x idx => let p := …; … (fun i => (sdA i p.1, sdM i
p.1)) …`), `vitFwdGraphBDrop_faithful` proves the typed graph denotes it, and
`batchSlice_den_dropPathB` is the per-example reading of the node; ConvNeXt has no masked forward
at all (and no typed batched forward, bf16_tie.md's note (a)).

1. Read, in this order: `Batched/Basic.lean` (the two lifts and their `batchSlice_*` peels),
   `ParamGrad.lean` (`param_batchMap_through` — what the loss capstone threads through a lift),
   `DataParallel/Sync.lean` §`batchShard_batchMap`, `Nets/ViT/ViTFwdDrop.lean` §2,
   `ViTStepTieGB.lean` (`vit_net_tiedGB`: the `ib_k := batchMap N (w.bk.fwdO …)` and `dy_k :=
   batchMapAux N (w.bk.cotIn …) ib_k dy_{k+1}` lets), `ViTParamGrad.lean` (`vit_net_lossGrad`'s
   `param_batchMap_through` calls), and the two drop sites in `Codegen/ViTRenderB.lean` (forward
   163 / 181, backward 341–377: `dropPathB` on the branch cotangent, the skip raw).
2. Decide the lift. (a) **`batchMapIdx N (f : Fin N → Vec a → Vec b)`** (and `batchMapAuxIdx`),
   with `batchSlice_batchMapIdx`, differentiability, a `HasVJP`, `param_batchMapIdx_through` and
   `batchShard_batchMapIdx` (shard `r`'s index map is `fun n => f (finProdFinEquiv (r, n))`) — the
   honest shape, `batchMap N f = batchMapIdx N (fun _ => f)` by `rfl`, so the drop-free ties are
   the constant-index instance; or (b) **the mask rides in `batchMapAux`'s saved slot**: the
   example's two scalars as a `Vec 2` beside its saved activation, no new lift — cheaper if
   `batchMapAux`'s own `HasGradAt` through-lemma exists, otherwise (a) is less work. Check what
   `batchMapAux` has before choosing; write the chosen lift's lemmas in a leaf file
   (`Foundation/Batched/Indexed.lean`, imported by the ViT and ConvNeXt ties only).
3. Pilot on ONE ViT block: `vitBlockTiedGB` (`ViTStepTieGB`) with `(sdA sdM : Vec N)` for the
   block, the cotangent chain's `cotIn` through `dropPath` at `mA` / `mM`, then the capstone with
   `sd : Option (Fin 12 → Vec N × Vec N)`-shaped sites (24 scalars per example), `vit_net_lossGrad`
   through the indexed `_through`. ConvNeXt after (18 sites, `cnxBlockChTiedGB`, the
   `@[irreducible]` stem untouched). Spell every mask `fun f => f i` (§2).
4. Same discipline: stage per net, stop for the commit word; comparator + yaml + book move with
   the Lean.

## 1. MobileNetV2 classifier dropout — DONE 2026-10-07

**What landed.**
* `Foundation/DropSites.lean` (new, a leaf on `DataParallel.Sync`): `dropPathOpt` / `dropoutOpt`
  and the `*OptG` graph nodes MOVED here from `EfficientNetFullB0Drop` (which now imports it),
  plus what a tie needs: `dropoutOptHasVJP` / `dropPathOptHasVJP` with the backward written as
  the `backward` FIELD (`fun _ dy => dropoutOpt cd dy`, `correct` by cases) so
  `(…).backward x dy` unfolds to the op at a SYMBOLIC site — a `cases`-built witness would not,
  and `mnv2HeadCotBlk_eq_vjp`'s closing `rfl` needs it; `_differentiable` (cases: `id`,
  `layerScale_differentiable`); `dropoutOpt_smul` (`IsHomog`); `dropoutOpt_shard`
  (`cd.map (batchShard … · r)` on `batchShard … X r` is `batchShard … (dropoutOpt cd X) r`,
  `cases cd <;> rfl` — `batchShard_zipWith` is `rfl`).
* `MobileNetV2StepTieB`: the head at the site, `mnv2HeadBDoOpt … cd` (`@[reducible]`, `∘` with
  `dropoutOpt cd` between the dense and the GAP; `_none` = `mnv2HeadB`, `_some` = `mnv2HeadBDo`,
  both `rfl`), `mobilenetv2ForwardBFullDoOpt` in the NESTED form (`_none` = `mobilenetv2ForwardBFull`
  by `rfl`, `_eq_chain` the same `rw [mnv2PreB*_apply]` proof) and `mnv2HeadBDoOptHasVJPAt` (one
  more `vjpCompAt`) — all in the tie file, not `FullB` / `FullBVJP`, so the forward files' cone
  (Seal, WholeBack, SpecVJP) does not rebuild. `mnv2HeadCotGapIn N Wd cd g := dropoutOpt cd
  (rowDenseBackFlat … g)`; the five head chain defs, `mnv2HeadTiedB` (`a := dropoutOpt cd (gap …)`,
  the dense weight node at the DROPPED activation) and the capstone take `cd` after `bf16`;
  `mnv2_lossCot_is_smoothedCE_grad` reads the logits at `DoOpt`. Proofs unchanged except
  `mnv2HeadCotBlk_eq_vjp`'s statement (now against `mnv2HeadBDoOptHasVJPAt`; same `calc`).
* `MobileNetV2ParamGrad`: `mnv2HeadLossTiedB` / `mnv2_head_lossTiedB` at `mnv2HeadBDoOpt` with one
  extra `HasGradAt.comp (f := dropoutOpt cd)` between the dense and the GAP pull-backs; the 18
  `mnv2Suf*` and 19 `mnv2_factor_*` take `cd` (bodies by regex `mnv2Suf\w+ N w ` → `… N w cd `,
  `mobilenetv2ForwardBFull N { w with … } x` → `…DoOpt N { … } cd x`); `h17`'s differentiability
  chain gains `((dropoutOpt_differentiable cd) _).comp _`.
* `MobileNetV2SyncStepTieB`: the capstone takes the GLOBAL mask `cd : Option (Vec ((R * N) * 1280))`
  and replica `r`'s chain runs at `cd.map (batchShard R N 1280 · r)` — what the DP render's
  per-replica `%do` input is; `mnv2HeadCotHr_smul` adds `dropoutOpt_smul`, `mnv2HeadCotHr_shard`
  adds `dropoutOpt_shard`; `mnv2HeadSyncTiedB`'s `DenseSync` at `a := dropoutOpt cd (gap …)`.
  Every other step is the f32/bf16 one.
* `MobileNetV2SyncB`: `mnv2HeadGraphSyncDo` / `mobilenetv2FwdGraphSyncFullDo` (replica `r`'s
  `dropoutB` at its own mask `ms r`) and their `_shard` (`hm : ∀ r, ms r = batchShard … M r`; the
  dropout step `rw [den_dropoutB, hg r, hm r]; rfl` after `unfold mnv2HeadBDo`) — the forward
  half the sync tie's "What is NOT claimed" cites.
* yaml rows (4f paragraph, the MNv2 loss-gradient / sync comments), comparator tier regenerated
  (the two MNv2 DECLS gain the binder), `tests/AuditAxioms.lean` (DropSites + the new MNv2
  lemmas; all core axioms), `docstring-checkrefs`, `check_audit_coverage.py`, `import_audit.py
  implied` for the new file (⚠ that gate is RED at HEAD on 33 pre-existing implied imports, most
  of them the bf16 thread's `GradNodesBAt` / `Bf16Erasure` lines — a separate cleanup).

**What the step taught.**
* Three builds, zero proof failures: every proof that was `rfl` or a `_holds` instance at the
  drop-free chain stayed so, because the site is one more `layerScale` whose VJP is itself. The
  only proof CONTENT is in `DropSites`.
* The `Option` binder, not a separate `*Do*` theorem: `none` is the old statement by `rfl`, so
  nothing is restated and the comparator row moves by one binder.
* The forward files' `mnv2HeadBDo` / `mobilenetv2ForwardBFullDo` (at a bare mask) stay as the
  typed graph's `_faithful` targets; the ties' `DoOpt` forms meet them at `_some … := rfl`.
* Threading by exact-string replacement with a count assertion on every edit (not regex over the
  file) — the first ParamGrad pass tripped on a conjunct the combined theorem indents
  differently, and the assertion caught it before the build did.

## 2. EfficientNet-B0 — stochastic depth and classifier dropout — DONE 2026-10-07

**What landed.** The two sites as `Option` binders on the three capstones, after `bf16`:
`sd : Option (Fin 9 → Vec N)` (the nine skip-carrying blocks' per-example scales, site `i` of
block `b3 b5 b7 b8 b10 b11 b13 b14 b15` being `sd.map fun f => f i`) and `cd : Option (Vec (N *
1280))`; the sync capstone takes the global masks (`Fin 9 → Vec (R * N)`, `Vec ((R * N) * 1280)`)
and replica `r` runs at `exampleShard` / `batchShard` of them.
* `Foundation/DropSites.lean`: `exampleShard` (`batchShard`'s cut at width one, for `Vec N`
  masks), `dropPathOpt_smul`, `dropPathOpt_shard` (`cases sd`; the `some` case is one `simp only`
  with `Equiv.symm_apply_apply` at `batchShard`'s index), `dropPath_shard`, and the replica family
  `dropPathOptFam` with `_shard` / `_scaled` — the invariant the sync tie passes through a site.
* `EfficientNetFullB0Drop.lean` (the forward-with-sites file, now imported by the three ties):
  `mbResidDropWHasVJP` = `residualHasVJP` over `vjpComp (mbExpW …) (dropPathOpt …)`, so its
  backward is `branch.back x (dropPathOpt s dy) + dy` by definition (`eBack`'s rule: the branch sees
  the dropped cotangent, the skip the raw one); `headDoFwdBHasVJP` with `dropoutOptHasVJP` between
  the GAP's and the dense's; the two `_differentiable`s.
* `EfficientNetStepTieG`: `enetHeadTiedG` at the dropped activation; the capstone's prefixes at
  `mbResidDropW … (sd.map fun f => f i)`, its chain at the drop VJPs, the nine residual conjuncts at
  `dropPathOpt N (c·h·w) (site i) dy`, the head at `cd`. Proof: the same 18 `exact`s.
* `EfficientNetParamGrad`: `enet_resid_lossTiedG` at the site (`hasGradAt_addConst` then
  `HasGradAt.comp_global` through `dropPathOptHasVJP`); `enetResidW_hasGradAt_comp` at
  `mbResidDropW`; `enetHeadLossTiedG` with one more `HasGradAt.comp (f := dropoutOpt cd)`; the
  prefixes `enetPreB3 … enetPreB16`, the suffixes and the 18 factor lemmas carry `sd` (`cd`);
  the capstones read the loss at `efficientnetForwardBFullDrop`.
* `EfficientNetSyncStepTieG`: `rCotIn` / `rsCotIn` feed the branch `dropPathOpt … dy` /
  `dropPathOptFam …`; `rCotIn_eq_vjp` against `mbResidDropWHasVJP` (the `hc` is still `rfl`);
  `hdCotIn_eq_vjp` against `headDoFwdBHasVJP` by a `rfl`-stated `have` plus `cbsB_back_eq` (a
  `simp only [headDoFwdBHasVJP]` leaves an `id`-wrapped term the reducible type check rejects);
  the head's shard / smul add `dropoutOpt_shard` / `_smul`; the nine residual conjuncts at
  `dropPathOptFam` / `dropPathOpt`, their invariant `dropPathOptFam_scaled`.
* `EfficientNetSyncB`: `mbResidDropGraphSync`, `headGraphSyncDo`,
  `efficientnetFwdGraphSyncFullDrop` and their `_shard`s (stated at rendered sites, the replicas'
  masks the shards of the global ones) — the forward half the sync tie cites.
* yaml rows and 4f, comparator tier regenerated, `tests/AuditAxioms.lean`, the gates as for §1.

**What the step taught — two elaboration traps, both in how a mask is SPELLED.**
* ⛔ **Never `(· i)` in a statement.** A dot-lambda's binder is hygienic, so the tie theorem's and
  the combined theorem's `let` chains differ by a binder name, `enet_net_tied_lossGrad`'s
  `extract_lets at htie` no longer merges the tie's lets into its own, and the closing `exact`
  pairs 18 conjuncts by unfolding both chains — 12 s and past `maxRecDepth` even at 100000. With
  `fun f => f i` the lets merge (verified on a two-line scratch) and the original proof is back
  as it was.
* ⛔ **In a proof, a mask passed to a lemma whose batch is implicit must carry its binder type**
  (`sd.map fun f : Fin 9 → Vec N => f i`): `Option.map`'s function is elaborated before `sd`, an
  untyped lambda is postponed, and the step's expected-type unification then meets a
  metavariable under the binder, fails, and unfolds the whole residual block before giving up
  (`trace.Meta.isDefEq`: 24k `match` failures on `sd`). Naming `N`, naming `G`, or spelling
  `Option.map` explicitly do not help; `have h := …; exact h` does, and so does the annotation.
* The diagnosis method that worked: cut the proof after each `have` into a scratch copy and
  `lake env lean` it (5 s green vs 16 s red), then `trace.Meta.isDefEq` at a small heartbeat
  budget to read which term the unifier descends into.
* Zero proof-content changes beyond `DropSites`: as for MNv2, every `rfl` and `_holds` instance
  stayed `rfl`; the VJP witnesses' backwards are the chain links by definition.


**The gap.** Every step tie, sync tie and `*_net_lossGrad` is stated on the chain WITHOUT the
training masks. The book's seven jobs (`content.tex` job table, ~18370; each conf's
`LEAN_MLIR_VARIANT` names the artifact) train these:

| job → artifact | drop-path sites | classifier dropout | other outside the tie |
|---|---|---|---|
| `r34-default-bf16-4gpu` → `resnet34in_momdp64bf16` | — | — | — (fully reached) |
| `r50-2018-bf16-4gpu` → `resnet50in_momdp64bf16` | — | — | — (fully reached) |
| `r50-a3-wxclip4x128-bf16-4gpu` → `resnet50in160_lambaccdp4x128wxclipbcebf16` | — (A3 sets `dropPath := 0.0`, `ResNet50RenderB`; the `*drop*` R50 renders are A2's, no job) | — | accumulation |
| `mnv2-default-4gpu` → `mobilenetv2in_rmsdp64wxdols0eps0001bf16` | — | 1 (`%do`, per element, before the dense) — DONE, §1 | — |
| `enet-default-4gpu` → `efficientnetin_emarmsdp64dropdowxeps0001bf16` | 9 (the skip-carrying MBConvs of 16) — DONE, §2 | 1 — DONE, §2 | EMA |
| `cnx-default-4gpu` → `convnextin_adamdpwxclipdroperfbf16` | 18 (one per block) | — | — |
| `vit-default-emabf16-4gpu` → `vitin_emadp128x4wxclipdropeps0000001erfbf16` | 24 (two per block) | — | EMA |

Side quest, not a book job: MobileNetV4-Conv-M `mnv4in_emaaccdp8x128wxdowd005bf16` (dropout;
EMA, accumulation) and its `*wxdropdowd01bf16` axis sibling (drop-path + dropout).

EMA and accumulation are optimizer tails that consume the gradient nodes; the ties already say
they're outside, and that's a different kind of statement. So the masks are the last gap at the
gradient nodes for four of the seven jobs, and for ConvNeXt and ViT they are the only one — every
`convnextin_*bf16` and `vitin_*bf16` artifact in the manifest carries `drop`, so tier A's "at
either precision" for those two nets is stated at a chain no bf16 artifact has.

**Why it should be tractable.**
* Both masks are a diagonal scaling and the VJP is the op itself at the same mask:
  `Proofs.dropPath_vjp_is_self`, `dropout_vjp_is_self` (Training/DropPath.lean);
  `dropout_of_dropScale` says drop-path IS dropout at a lifted (per-example-constant) mask. Every
  renderer's backward emits it that way and cites the lemma (`ConvNeXtRenderB`, `ViTRenderB`,
  `ResNet50RenderB`, `EfficientNetRender/Basic`, `MobileNetV2RenderB`, `MobileNetV4RenderB`).
* The masked FORWARDS are already proved for three nets, with the sites as `Option` binders —
  `none` = no node = the drop-free artifact, `some s` = the `dropPathB` / `dropoutB` node
  (`dropPathOptG` / `dropoutOptG`, EfficientNetFullB0Drop.lean): `efficientnetFwdGraphBFullDrop`
  (`sd : Option (Fin 9 → Vec N)`, `cd : Option (Vec (N * 1280))`), `mnv4FwdGraphBFullDrop` and the
  `Do` heads (MNv2, MNv4), and ViT's `vitFwdGraphBDrop_faithful` (ViTFwdDrop.lean, masks
  `sdA sdM : Fin k → Vec B`, per-example `blockVDrop`). ConvNeXt and ResNet-50 have no typed drop
  forward graph (ConvNeXt has no typed batched forward at all — bf16_tie.md, ConvNeXt note (a)).
* Where the mask lands in the chain decides the cost. The CNN ties are BATCHED chains (BN), so a
  site is one batched node, `dropPath N n s` / `dropout m`, dropped into the chain where the
  renderer puts it; the node predicates (`ConvWTiedBAt … x … dy`) do not change, only the `x` (the
  dropped activation — gate W below) and `dy` they are fed. ConvNeXt's and ViT's ties are
  `batchMap` / `batchMapAux` lifts of a UNIFORM per-example function (Foundation/Batched/Basic.lean
  has only those two; `HasGradAt.param_batchMap_through` likewise), and a per-example mask makes
  it a different function per example. `vitForwardKVDropB` dodges this by being defined pointwise
  (`fun x idx => let p := …`), which the backward cannot do under the ParamGrad lift. So the LN
  nets need either an indexed lift (`batchMapIdx` + its `_through`) or the mask carried in
  `batchMapAux`'s saved slot (`Vec (N * s)`, per-example-sliced) beside the activation. Design
  this before touching ViT; it is the one piece of new Foundation in the thread.

**First session.**
1. Read: `ConvNeXtStepTieGB` / `ViTStepTieGB` scope paragraphs (both name the gap),
   `Training/DropPath.lean`, `ViTFwdDrop.lean`, `EfficientNetFullB0Drop.lean`, and how each
   renderer places the backward drop (site order matters — planning/archive/stochastic_depth.md
   §7b: the ones-mask gate is blind to placement; `tests/TestDropPathTie.lean` gate B is the
   numeric control).
2. Statement shape: the sites as `Option` binders, as the forward graphs already have them — the
   existing drop-free tie is the `none` instance verbatim, the drop artifact's is `some s`;
   nothing restated. `dropPath_ones_id` is the separate eval statement (the drop artifact's
   forward at a ones mask is the drop-free function), not how the drop-free tie is recovered.
3. Order, cheapest whole-net win first: **MobileNetV2 classifier dropout** (DONE, §1: one batched
   site, the job's only gap — `mnv2-default-4gpu` closed; the dense weight node reads the dropped
   buffer, which is `tests/TestDropoutTie.lean` gate W as a theorem), then **B0** (DONE, §2: nine
   drop sites + the dropout, batched chain), then the lift design, then **ViT** (masked forward in
   hand, 24 sites), then **ConvNeXt** (18 sites, no typed forward, `@[irreducible]` stem). MNv4
   follows B0's recipe if the side quest is ever written up.
4. Same discipline as bf16_tie: stage per net, stop for the commit word; comparator tier + yaml +
   book sentences move with the Lean (the book's "classifier dropout sits outside that statement"
   / "the drop-path chain" sentences, one chapter per net).

## Background

* planning/archive/stochastic_depth.md — the August design of the drop-path render (host-drawn
  mask inputs, the depth ramp, why the RNG is not in the graph, the gates and their blind spots).
* tests/TestDropPathTie.lean (`lake build droppath-tie`; gate A: the scale is exactly the supplied
  per-example scalar, gate B: the site is on the residual branch) and tests/TestDropoutTie.lean
  (gate A: per element, gate W: `∂L/∂W` reads the dropped activation) — the numeric gates whose
  content the mask chain states.
* planning/bf16_tie.md — the thread that closed the precision gap; its recipe (flag on the
  statement, erase in a first `simp only`, binders by line number) is the template for threading
  a new binder through a net's tie files.
