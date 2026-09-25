# Proof-quality re-audit: Codegen files changed since 2026-09-22

This is a static read of the tree at `28fc373f` on branch `proof-cleanup`. Nothing was compiled or edited. The rubric is `proofqualityaudit.md`. Prior findings are in `planning/proof_cleanup_audits/audit_codegen_certs.md` §A–B and `planning/proof_cleanup.md` §1(g)/§3.2. The rule "never name `den` in a simp set" is respected: every suggestion below uses `denStep`/`denStepApp` or named `den_*` lemmas.

## Headline facts

- **There are no theorems in 14 of the 15 files in scope.** That includes `StableHLOPretty`, `RenderKit`, `FwdGraphTextTies`, `CnnArtifacts`, `MlpArtifacts` and every `*Render*.lean`. The one hit for `ConvNeXtRender.lean:267` is prose inside a docstring. All proofs are in `StableHLO.lean`: 197 theorems, 157 of them `rfl`, none longer than 23 lines.
- For these files the rubric's findings come down to three things: the option lines, `open Classical`, and a few `show`/bare-`simp` sites in `StableHLO.lean`. Question 4 (repetition) mostly concerns data and `#guard`s, not proof scripts.

---

## Ranked findings (payoff / effort)

| # | finding | effort | payoff |
|---|---|---|---|
| 1 | Two `maxRecDepth 4000000` lines are attached to a **`structure`**. The decl each was meant for has been compiling **without** a bump. This is proof by accident that the 4M values are not needed. | trivial | high: it licenses stripping the other ~30 |
| 2 | Strip-and-compile the other 4M/1M `maxRecDepth` lines (table Q1) | small | medium (hygiene; 4M turns a runaway into a stack overflow) |
| 3 | `open Classical in` at `StableHLO.lean:1455` is dead since §1(r) | trivial (root file: batch it) | low |
| 4 | Two `show`s in `StableHLO.lean` rely on undocumented defeq (open since the first audit) | trivial | low–medium |
| 5 | Nine bare `simp [...]` calls in `StableHLO.lean` close per-example and `N = 1` facts | small | low–medium (bump fragility) |
| 6 | Sync-BN `R = 1` lemmas: one graph spelled 6× and one step proved 3× | small–medium | low |
| 7 | `StableHLOPretty` still carries `#eval` artifact writers, 3 stray `/tmp` writers, a stale "regenerate" comment, and unused `DecidableEq` | small | low–medium (build hygiene) |
| 8 | `emitTok`'s `maxHeartbeats 1000000` was measured before emitTok lost 411 lines | trivial (one measurement) | low |
| 9 | 48 `#guard`s test string literals rather than code | trivial | low |
| 10 | Repeated zero-weight argument packs in `FwdGraphTextTies` | trivial | cosmetic |

---

## Q1. The 43 `set_option maxRecDepth` lines in the render files

### Two lines prove the point: they guard a `structure`, not the decl they were written for

### LeanMlir/Proofs/Codegen/ResNet50RenderB.lean:496 — `R50FwdRecB` (a structure)
**Smell:** heartbeats-class (misattached, dead)
**Current:**
```
set_option maxRecDepth 4000000 in
/-- Everything the whole-net render needs out of ONE forward traversal … -/
structure R50FwdRecB where
  code : String
  …
```
**Why it breaks:** `git show 289c929c` (2026-08-10) inserted the docstring and structure between this option and `resnet50TrainStepFaithfulB`, which the option originally guarded. Since then the R50 train step (`ResNet50RenderB.lean:614`: 64 `←` binds, the largest do-block in the file, with a 4-region optimizer and every recipe flag) has built unbumped at the default `maxRecDepth` 512. That includes every CI run and every artifact regeneration.
**Suggested:** delete the line.

### LeanMlir/Proofs/Codegen/EfficientNetRender.lean:728 — `ENetStemFwdB` (a 5-field structure)
**Smell:** heartbeats-class (misattached, dead)
**Current:**
```
set_option maxRecDepth 4000000 in
/-- The stem's saved SSA names: conv, BN, BN stats (`""` at one replica), swish output. -/
structure ENetStemFwdB where
  code : String
  c : String
  …
```
**Why it breaks:** Before today's `7331d8aa` this line guarded `enetFwdChain` (the full B0 forward). The commit extracted the stem into `ENetStemFwdB`/`enetStemFwdB` above it, so `enetFwdChain` (`EfficientNetRender.lean:783`: 20 binds, a 16-block chain with an `if`, and three 17-way `++` chains) now elaborates unbumped. `FwdGraphTextTies` imports the file, so building it compiles the chain.
**Suggested:** delete the line.

**Rule these two teach:** an `… in` separated from its target by a docstring or a banner drifts onto whatever gets inserted between them. `CnnRender.lean:613` and `:871` already have a blank line and a section banner between the option and the def. Keep each option directly above its target's `/--`.

### The table

Size calibration from unbumped decls that compile today: `resnet50TrainStepFaithfulB` (64 binds), `enetFwdChain` (20), `optOne` (R34 :555, 33), `cifarTrainStepFaithfulV` (CnnRender :118, 42). In the table, "binds" counts `←` inside the decl.

| file:line | value | guards | shape | guess |
|---|---|---|---|---|
| ResNet50RenderB:496 | 4M | `R50FwdRecB` (structure) | — | **dead (certain)** |
| EfficientNetRender:728 | 4M | `ENetStemFwdB` (structure) | — | **dead (certain)** |
| ResNet34RenderB:183 | 1M | `r34FwdChain` | 22 binds, 16-block chain | likely dead (same shape as `enetFwdChain`) |
| ResNet34RenderB:237 | 1M | `resnet34FwdEvalFaithfulV` | 0 binds, string `++` | likely dead |
| ResNet34RenderB:1309 | 4M | `r34FwdChainB` | 18 binds | likely dead |
| ResNet34RenderB:1344 | 4M | `resnet34FwdFaithfulB` | 0 binds, `.run'` + 7-way `++` | likely dead |
| ResNet34RenderB:1364 | 4M | `resnet34AdamTrainStepFaithfulB` | 61 binds | likely dead (R50's 64-bind peer is unbumped) |
| ResNet50RenderB:552 | 4M | `r50FwdChainB` | 18 binds | likely dead |
| ResNet50RenderB:1140 | 4M | `r50FwdChain` (eval) | 22 binds | likely dead |
| ResNet50RenderB:1183 | 4M | `resnet50FwdFaithfulV` | 0 binds | likely dead |
| ResNet50RenderB:1204 | 4M | `resnet50FwdEvalFaithfulV` | 0 binds | likely dead |
| MobileNetV2RenderB:567 | 4M | `mnv2FwdChainB` | 19 binds | likely dead |
| MobileNetV2RenderB:624 | 4M | `mobilenetv2FwdFaithfulB` | 0 binds | likely dead |
| MobileNetV2RenderB:654 | 4M | `mobilenetv2AdamTrainStepFaithfulB` | 72 binds | likely dead at 4M; if not, ≤16000 |
| MobileNetV2RenderB:1158 | 4M | `mnv2FwdChain` (eval) | 25 binds | likely dead |
| MobileNetV2RenderB:1226 | 4M | `mnv2FwdEvalFaithfulV` | 0 binds | likely dead |
| MobileNetV4RenderB:599 | 4M | `mnv4FwdEvalFaithfulV` | 0 binds | likely dead |
| MobileNetV4RenderB:936 | 4M | `mobilenetv4AdamTrainStepFaithfulB` | 45 binds | likely dead |
| EfficientNetRender:865 | 4M | `efficientnetFwdFaithfulV` | 0 binds | likely dead |
| EfficientNetRender:880 | 4M | `efficientnetFwdEvalFaithfulV` | 0 binds | likely dead |
| EfficientNetRender:900 | 4M | `enetBackAll` | 42 binds | likely dead; ≤8000 at worst (its ConvNeXt peer is 8000) |
| EfficientNetRender:1059 | 4M | `efficientnetTrainStepFaithfulV` | 1 bind (a wrapper) | likely dead |
| EfficientNetRender:1160 | 4M | `efficientnetAdamTrainStepFaithful` | 5 binds + `for`/`let mut` loops | uncertain; if needed, ≤8000 (ConvNeXtRender:913 is the same shape at 8000) |
| ConvNeXtRender:630 | 8000 | `convNextFwdChain` | 7 binds | likely dead |
| ConvNeXtRender:675 | 8000 | `convNextFwdFaithfulV` | 0 binds | likely dead |
| ConvNeXtRender:705 | 8000 | `convNextBackAll` | 23 binds | likely dead |
| ConvNeXtRender:815 | 8000 | `convNextTrainStepFaithfulV` | `Id.run do`, 0 binds | likely dead |
| ConvNeXtRender:913 | 8000 | `convNextAdamTrainStepFaithful` | 3 `for` loops, `let mut`, 14-line `s!` `++` | **possibly needed** (the `for`/`mut` desugaring nests joins) |
| ConvNeXtRenderB:250 | 8000 | `convNextFwdChainB` | 7 binds | likely dead |
| ConvNeXtRenderB:308 | 8000 | `convNextFwdRenderB` | 0 binds | likely dead |
| ConvNeXtRenderB:509 | 8000 | `convNextBackAllB` | 23 binds | likely dead |
| ConvNeXtRenderB:634 | 8000 | `convNextAdamTrainStepFaithfulB` | **one function call** | dead (almost certain) |
| ViTRenderB:104 | 8000 | `vBlockFwdB` | 23 binds | likely dead (values look chosen; lower confidence) |
| ViTRenderB:214 | 8000 | `vitFwd12B` | 5 binds | likely dead |
| ViTRenderB:264 | 8000 | `vitFwdRenderB` | 0 binds, string `++` | likely dead |
| ViTRenderB:337 | 8000 | `vBlockBackB` | 41 binds | likely dead |
| ViTRenderB:481 | 16000 | `vitBackAllB` | 19 binds | likely dead |
| ViTRenderB:592 | 16000 | `vitAdamTrainStepFaithfulB` | **one function call** | dead (almost certain) |
| CnnRender:196 | 4000 | `cifar8TrainStepFaithfulV` | 70 binds, 96 lets, flat | possibly needed (modest value) |
| CnnRender:392 | 8000 | `cifar8AdamTrainStepFaithfulV` | 94 binds, flat | possibly needed |
| CnnRender:613 | 8000 | `cifar8AdamTrainStepFaithfulB` (banner in between) | 94 binds | possibly needed; move the line down |
| CnnRender:871 | 8000 | `cifar8BnTrainStepFaithfulV` (blank line in between) | 183 binds | likely needed |
| CnnRender:1258 | 8000 | `cifar8BnTrainStepFaithfulB` | 142 binds | likely needed |

Summary: 2 are certainly dead, ~31 are likely dead, 2 are uncertain, and ~5 (CnnRender, with 70–183-statement flat do-blocks) are plausibly needed at their modest values.

**Protocol for the follow-up (per file, one compile each):**
1. Delete every `maxRecDepth` line in the file.
2. Run `lake env lean` on a scratchpad copy. Watch out: these files carry `#eval IO.FS.writeFile "verified_mlir/…"`. A standalone compile **rewrites the committed artifacts**, which must come back byte-identical, so check `git status verified_mlir` after each run. An alternative is to compile the copy from a scratch cwd, so that the relative `verified_mlir/` paths miss the repo.
3. Where "maximum recursion depth has been reached" names a decl, re-add `8000` directly above that decl's `/--` and double it until the decl compiles.

`maxRecDepth` is a limit, not a cost, so this is hygiene and not speed. The old audit's structural fix (a stage-table `foldlM` instead of 16 unrolled `let fᵢ ←`) is only worth doing if a strip-test shows a real need.

---

## Q2. LeanMlir/Proofs/Codegen/StableHLOPretty.lean:1363 — `emitTok`

**Smell:** heartbeats (5× the default, documented)
**Current:**
```
-- Compiling this one 99-arm def needs ~2× the default budget (more under `trace.profiler`, which
-- trips 400000); 5× leaves room for new arms. Nothing else in the file needs a bump.
set_option maxHeartbeats 1000000 in
def emitTok (B : Nat) : Tok → List String → StateM EmitS (String × List String)
```
**What is new:**
- The "~2×" figure comes from §1(g) (`bb38c790`). Since then `375fff18` (bf16 smart constructors) and `91737514` (bf16/fp8 twins collapsed into `emitContract`, **−411 lines**) have shrunk the body; it now spans L1368–4498, about 3,130 lines. The arm count is still 99 (counted with `^  | `), so the comment's arm count holds, but the budget figure is stale.
- The payoff also shrank. After `9da1a817` the def lives in `StableHLOPretty`, so the 83 semantic-only importers of `StableHLO` no longer wait on it.

**Suggested:** one measurement. Replace the option with `count_heartbeats in` (Mathlib) and set the value to 2× the measured figure, or delete it if the measurement is under 200k. Do not size it under `trace.profiler` (§0 trap). The structural fix, which removes the option for good, is unchanged from the old audit: per-family arm bodies as top-level defs (`emitBnTok`, `emitOptTok`, …), with the dispatcher kept. `emitContract`, `emitFlatConv` and `emitMatmul` (L1299–1356) already show the pattern.

---

## Q3. LeanMlir/Proofs/Codegen/StableHLO.lean:1455 — `open Classical in` on `maxPoolBackFlat`

**Smell:** dead scope (the §1(r) leftover)
**Current:**
```
open Classical in
noncomputable def maxPoolBackFlat (c h w : Nat)
    (xv : Vec (c*(2*h)*(2*w))) (dyv : Vec (c*h*w)) : Vec (c*(2*h)*(2*w)) :=
  fun idx => …
    if MaxPool2IsArgmax (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w)) q.1 q.2 p.2
    then … else 0
```
**Why:** The only `if` it covers is on `MaxPool2IsArgmax`. Since §1(r) that has `noncomputable instance MaxPool2IsArgmax.decidable` at `Architectures/CNN.lean:905`, and that instance's docstring says it exists "so an `if MaxPool2IsArgmax …` elaborates without `open Classical`". `StableHLO` imports `Architectures.CNN`. `Classical.propDecidable` has **low** priority, so even with the `open` in place the elaborator already picks the CNN instance. Removing it therefore leaves the elaborated term unchanged, and `ChapterGraphTies.lean:100`'s `… = maxPoolBackFlat c h w v := rfl` is unaffected. The other decidable `if` nearby (`maxPool3s2BackFlat`, `Fin` equality) is outside the `in`. This is the last `Classical` in `Proofs/Codegen`.
**Suggested:** delete L1455. `StableHLO` is a root file with 241 dependents, so batch the change with items 4–6 (§0 rule).

---

## Q4. Repetition

### FwdGraphTextTies.lean — nothing to macro-ize in *proofs*; it has none
All 26 checks are `#guard textOf (emitter …) (·.code) == prettyText 2 (graph …)`. The repetition is in **data**:
- `(fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)`, a zero conv+BN slot (W b ε γ β), appears about 25 times: L71–72, L104–105, L139–140, L160–162, L198–199, L203–205, L213–216, L224–225, L231–260 and L269–271.
- The zero weight records `r34IdW0`/`r34DownW0`/`r50IdW0`/`r50ProjW0`/`mnv2IVW0`/`mnv2NoExpW0` (L60–135) are the same anonymous constructor with 10/15/20 zero fields.

**Suggested (cosmetic, trivial):** give the weight structures an `Inhabited`/`Zero` instance (`default`), or have the graph builders take the stem/head slots as the existing records. A term macro cannot stand in for five positional arguments, so the slot repetition is only fixable at the builders' signatures, and that is not worth a statement change. **No finding against proof quality.** One note: `mnv4RowGraphText` (L180–194) returns `""` for a row with no T2 graph, and the L208 `.all` guard makes a new post-strided row fail loudly. That is good.

### Render files — no theorems, so no proof repetition
The only repeated *checking* pattern is the literal-only `#guard`s (item 9).

### StableHLO.lean:2515–2650 — sync-BN `R = 1` lemmas
**Smell:** repetition
**Current:** The `R = 1` statistics graph
```
.bnPackB (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))
  (.allReduceMeanF 1 Nat.one_pos t' ds' (fun _ => .bnBatchVarAtB x
    (.allReduceMeanF 1 Nat.one_pos t ds (fun _ => .bnBatchMeanB x))))
```
is spelled out in **six** statements: `den_syncStats_R1` (twice), `den_bnSyncF_allReduce_R1`, `den_bnSyncBack_allReduce_R1`, `den_bnSyncGammaGradB_allReduce_R1`, `den_bnStatsMeanB_allReduce_R1` and `den_bnStatsVarB_allReduce_R1`. The step `by funext c; rw [bnVar_eq_bnMeanSq_sub_sq _ hm]; ring` (`bnVar + μ·μ = bnMeanSq`) is proved three times: L2563, L2617, and `hm2c` at L2590–2595. The `have key : ∀ m2, m2 = … → … := by intro m2 hm2; rw [hm2]; exact …` generalisation appears twice (L2558–2563, L2611–2617).
**Suggested:**
- Extract `theorem bnVar_add_mean_mul_mean (hm : m ≠ 0) (v) : bnVar m v + bnMean m v * bnMean m v = bnMeanSq m v` and put it in the BatchNorm leaf next to `bnVar_eq_bnMeanSq_sub_sq`, not in the root.
- Replace each `key` with `rw [show (fun c => … + … * …) = fun c => bnMeanSq … from funext fun c => bnVar_add_mean_mul_mean hm _]; exact …_at_own_stats …`.
- `Foundation/DataParallelSync.lean:486` already defines `syncStats R hR t t' ds ds' x`, and the six graphs are `syncStats 1 Nat.one_pos … (fun _ => x)`. Moving that def up into `StableHLO` would shorten six statements, but it is a **spelling change**: `rw` consumers matching the literal graph would miss a non-reducible def. Grep the consumers (`SyncBnSites`, the `*SyncB` twins) first. Medium effort, low payoff; optional.

---

## Q5. Other rubric findings, low-hanging first

### LeanMlir/Proofs/Codegen/StableHLO.lean:3188, 3639 — `bnBack_faithful`, `bnPerChannelBack_faithful`
**Smell:** undocumented-defeq (carried over from the first audit, still open)
**Current:**
```
  show bnGradInput n ε γ x (den e) i = _
  exact bn_input_grad_correct n ε γ β hε x (den e) i
```
and
```
  show bnPerChannelTensor3GradInput oc h w ε γ x (den e) i = _
  exact bnPerChannelTensor3GradInput_correct oc h w ε hε γ β x (den e) i
```
**Why it breaks:** The proof relies on the `den (.bnBack …)` arm reducing to exactly that helper. If the arm moves into a `BatchableOp` descriptor, which is the old audit's A.2 direction and how the batched BN already works, the `show` fails with a defeq error far from its cause. These are the only two `show`s in the file.
**Suggested:** add `@[simp] theorem den_bnBack … : den (.bnBack gN xN es ε γ x e) = bnGradInput n ε γ x (den e) := rfl` and its per-channel twin, then write `rw [den_bnBack]; exact …`. Batch this with item 3.

### LeanMlir/Proofs/Codegen/StableHLO.lean:2432, 2519, 2685, 2699, 2726, 2741, 2760, 2971, 2992
**Smell:** fragile-simpa (bare `simp [...]` closing through the default simp set)
**Current (e.g. L2426–2432, L2695–2699):**
```
theorem den_batchOp_softmaxDiv_per_example … :
    den (.batchOp (N := N) (.softmaxDiv (n := n)) e) (finProdFinEquiv (k, j))
      = batchSlice N n (den e) k j / ∑ i, batchSlice N n (den e) k i := by
  simp [batchMap, batchSlice]
…
theorem den_lnRowBackB_per_example … := by
  simp [den_lnRowBackB, batchMapAux]
```
**Why it breaks:** Each closes a `finProdFinEquiv`/`batchSlice` index goal using whatever Mathlib's default set contains (`Equiv.symm_apply_apply`, `Fin.sum_univ_one`, `Finset.sum_fin_eq_sum_range`, …). These are exactly the "sentinel" statements whose docstrings say they are the ONLY check that separates two functions that print the same bytes. If they turn red on a bump, the cause will be a simp-set change, not a semantic one.
**Suggested:** run `simp?` once at each site and paste the `simp only [...]`. Most will read `simp only [den_X, batchMapAux, batchSlice, Equiv.symm_apply_apply]` (plus `Fin.sum_univ_one`/`Fin.isValue` in the `_at_one` pair). Batch this with item 3.

### LeanMlir/Proofs/Codegen/StableHLOPretty.lean:4853–4968 — `#eval` writers and `/tmp` writers
**Smell:** layering (an old-audit A.2 leftover, now inconsistent with the repo's own convention)
**Current:**
```
#eval IO.FS.writeFile "/tmp/linear_fwd_v.mlir" …          -- ×3, nobody reads these
…
-- FFI on GPU). Regenerate with `lake env lean LeanMlir/Proofs/Codegen/StableHLO.lean`.
#eval (do IO.FS.createDirAll "verified_mlir"; IO.FS.writeFile "verified_mlir/linear_fwd.mlir" … )  -- 10 artifacts
```
**Why:** `64eb7943` moved the Mlp/Cnn writers to the leaf modules `MlpArtifacts`/`CnnArtifacts` ("Nothing imports this file … building them never rewrites an artifact"). `StableHLOPretty` has 40 direct importers and still writes 10 committed artifacts plus 3 stray `/tmp` files every time it elaborates. The regenerate comment still names `StableHLO.lean`, while `.github/workflows/proofs.yml:134` and `scripts/regen_verified_mlir.sh:560` use `StableHLOPretty`.
**Suggested:** move L4853–end to a leaf `Codegen/ChapterArtifacts.lean`, delete the three `/tmp` writers, and point `proofs.yml:133–141` and `regen_verified_mlir.sh:560` at the leaf. Per the memory note "Render guard on a new artifact", keep the drift guard on the same artifact list.

### LeanMlir/Proofs/Codegen/StableHLOPretty.lean:281, 889 — `deriving DecidableEq` on `Raw` / `Tok`
**Smell:** compile-time (unused instance; carried over, still unmeasured)
**Current:** `deriving DecidableEq, Repr, Inhabited` (Raw, L281) and `deriving DecidableEq, Repr` (Tok, L889).
**Why:** A re-grep over `LeanMlir`, `tests` and `apps` for `==`, `decide` or `DecidableEq` on `Raw`/`Tok`/`skel`/`toToks` found zero users. After the split these are the costliest derived instances left on the printer's serial path.
**Suggested:** drop `DecidableEq` from both and compare `-Dtrace.profiler=true` timings before and after (§0 method).

### Literal-only `#guard`s (48 across 6 render files)
**Smell:** a check that cannot fail when the code changes (flagged as repetition)
**Current (ResNet50RenderB.lean:1279–1285, the same pattern at 1541–1542, 1564–1565, 1874–1880, 1898–1905, 1933–1937):**
```
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.R34Opt.heavyBall == "momdp64"
#guard !"momdp64bf16".contains "do"
#guard !"momdp64bf16".contains "acc"
```
Counts: ResNet50RenderB 24, ViTRenderB 8, MobileNetV4RenderB 6, EfficientNetRender 4, ResNet34RenderB 3, MobileNetV2RenderB 3.
**Why:** The property lines test a hand-typed copy of the string. They stay green only because a *separate* guard pins function to literal, which is two guards and two copies of each name. The comments say the consuming side is pinned again in `tests/TestVariantPredicates.lean`.
**Suggested:** fold each group into one guard over the function's output, for example `#guard let v := r34AdamVariant 64 4 .heavyBall false false false "" true; v == "momdp64bf16" && !v.contains "do" && !v.contains "acc" && !v.startsWith "ema"`.

### Checked and clean
- `RenderKit.lean` (343 lines): defs only (`PGrad`, `adamOne`, `rmsOne`, `adamOneEma`, `packedTrainSig`, `packedTrainRetTys`, `dropMaskSig`, `bnEvalSite`). No options, no proofs.
- `MlpArtifacts.lean` / `CnnArtifacts.lean`: leaf `#eval` writers, as intended.
- `StableHLO.lean`: `maxHeartbeats` count is 0 (the file-wide 4M went in §1(g)). No `change`, no `simp only [… den …]`, no `native_decide`. `mlpBackGraph_faithful` (L3035) uses `denStep`/`denStepApp` and closes with `rfl`, the §1(g) shape.

---

## Patterns worth more than any single finding

1. **Options that drift onto the wrong declaration.** A `set_option … in` separated from its target by a docstring or banner silently moved onto a `structure` twice (R50 on 2026-08-10, EfficientNet today). Nobody noticed, because the real target compiled fine without it. This shows the render-file 4M values were never needed, and it tells the follow-up to strip and compile per file rather than bisect per line. Rule: put the option directly above the target's `/--`, and never above a banner.
2. **Defeq through `den`'s arms instead of named `den_*` lemmas.** The two `show`s, and the per-example sentinels that close by a bare `simp`, rest on how an arm unfolds. The file already has ~130 `rfl` node lemmas, so name them. This is also what keeps the descriptor refactor (old audit A.2) cheap.
3. **Artifact I/O in imported modules.** `CnnArtifacts`/`MlpArtifacts` set the convention. `StableHLOPretty` and the render files (whose `#eval` writers also trap anyone who strip-tests them) have not followed it yet.
