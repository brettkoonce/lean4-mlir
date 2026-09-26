# api_design_audit.md — public surface: extensionality, characteristic API, exposure

Started 2026-09-26 from a read-only API-design audit of the hand-written files under `LeanMlir/`
(tree at 373059db). Rubric: over-exposed helpers, general infrastructure hidden as `private` or
stranded in a net file, definitions without the lemmas that let consumers avoid unfolding them,
structures whose free data defeats extensionality, `@[simp]` quality, and compatibility-only
surface. Every proposed lemma marked "compiles" was checked with `lake env lean` against the built
oleans; every "no consumer" claim was grepped over `LeanMlir/`, `apps/`, `demos/`, `tests/`, `jax/`,
`blueprint/`, `formalization.yaml` and `scripts/`.

Counts at 373059db: 118 hand-written structures, no `@[ext]` anywhere (grep hits for `@[ext` are
`@[extern]`), 184 `@[simp]`, no `@[grind]`, 265 `private`.

## 0. Rules for this thread

* No compatibility layer. A deleted or renamed declaration is not done until every consumer that
  names it is repointed: `tests/AuditAxioms.lean`, `tests/comparator/` and
  `scripts/gen_comparator_tier.py` (`DECLS`, `MODULES`), `blueprint/` `\lean{}` names (then
  `scripts/blueprint_uses.py --check` against freshly regenerated `lean_decls`),
  `formalization.yaml`, docstring citations (the docstring-checkrefs gate), and generators whose
  output names it (fix the generator and regenerate, never the output).
* `private` breaks a `#print axioms` line in `tests/AuditAxioms.lean`. Drop the line when the
  public capstone already covers the declaration transitively.
* Root files are batched. `Foundation/Tensor.lean`, `Foundation/MLP.lean` and
  `Codegen/StableHLO/Basic.lean` each rebuild most of `Certs`; §2 collects every edit to Tensor and
  MLP, and §3 every edit to StableHLO/Basic, so each corpus rebuild is paid once.
* Gate every batch with `lake build Certs` (bare `lake build` skips it) plus
  `lake build LeanMlir Apps CertsHeavy Reference`, and record the exit status in §1.
* A behaviour change on the training path (§4) needs the affected configs listed and a render diff
  (`scripts/regen_verified_mlir.sh` + the proofs.yml diff list) showing no committed artifact moved,
  or naming the ones that did.
* No new names beyond what a section states; naming follows `LeanMlir/NAMING.md`.

## 1. Done

| commit | change |
|---|---|
| 433b433a | §2 root batch: 2.1–2.4, 2.6, 2.7 as tabled, 2.5 in part. Deviations below |
| 502c0232 | §3: 3.2 as a guard (3.1 not needed), 3.3 minimal form. Deviations below |
| 4fa1bbce | §4.1 + §4.2 as tabled; §4.3 not started. Notes below |
| (staged) | §5.2–5.4, §5.5 in part (5.1 next). Notes below |

§2 deviations:

* No `Unique` instance on the witnesses (2.1): it would make `default` a witness nobody wrote.
  `Subsingleton` plus an explicit `canonical` gives the same uniqueness.
* `.congr` added for `HasVJP` and `HasVJPAt` only (2.2): they are the only two built by a cast.
  The four `mb*WHasVJP` needed no transport at all: the maps are definitionally equal, so the
  `unfold …; exact` became a term. MNv4's `castLayer` stays where it is: it is a real reindex
  layer with its own VJP and graph, not a workaround for a missing `.congr`.
* 2.3 as tabled, without the bundled `backwardₗ`: `IsHomog`, `IsHomog.comp` and
  `HasVJP.backward_smul` moved to Tensor.lean under the same names; `backward_add` added for
  `HasVJP`/`HasVJPAt`; `HasVJPAt.backward_smul` added; `SyncKit.hasVJP3_backward_smul` is now
  `HasVJP3.backward_smul`.
* `unflatten_apply` is not `@[simp]` (2.4): making it simp rewrites every bare `simp` in the corpus,
  and the sites that want it already name it. The two `flatten_finProdFinEquiv` simp lemmas and
  the `Mat` `@[fun_prop]` set landed; 17 of the 18 `unfold …; fun_prop` sites now call bare
  `fun_prop` (Attention:829 multiplies two variable matrices and keeps its unfold).
* `sum_t3`, `unflatten_t3Idx`, `flatten_t3Idx` stay (2.4): they are `t3Idx`'s own API, and
  consumers rewrite in `t3Idx` form. `w3Idx` (= `t3Idx`) and `sum_w3` (= `sum_t3`), `sum_s2`
  (= `sum_finProdFinEquiv`) and `sum_heads_3d` are gone.
* The `pdiv` introduction is `pdiv_eq_of_hasFDerivAt`, not `HasFDerivAt.pdiv_eq` (2.5):
  `HasFDerivAt` unfolds to `HasFDerivAtFilter`, so dot notation cannot find it.
* `gradAt` and the `fderiv`/`pdiv` lemmas stay in `JacobianSeal` / `SgdDescent.Basic` (2.5):
  they are those files' subject, both import only `Tensor` and Mathlib, and placement is out of
  scope here. `sum_smul_basisVec` is stated over `basisVec`, not `Pi.single`, so it is not a
  literal copy of `pi_eq_sum_univ`.
* `mlpVerifiedHasVJP` and `linearVerifiedHasVJP` stay (2.7): the blueprint lists them, and
  `linearVerifiedHasVJP` is `denseHasVJP`, not the canonical witness. The six canonical-only
  `*VerifiedHasVJP` defs are gone.
* `lipschitz_cert_witness_s8.py` had drifted from its output (one docstring line); the hand edit
  is ported, and both generators touched (`trained_linear_descent.py`,
  `lipschitz_cert_witness_s8.py`) reproduce their files byte for byte.

§3 deviations:

* 3.1's count was wrong in the other direction: 167 of 215 `SHlo` constructors have no named
  `den_<ctor>` equation, not 42. They reach `simp` through the `denStep` proc (and most have a
  `*_faithful` theorem in VJP terms). Adding 167 `rfl` equations is not the fix — the file header
  already records that fixed-index arms are expensive by `rfl`.
* 3.2 therefore keeps `denStep`/`denStepApp` in the default simp set (one step on a constructor
  is a normal form, the proc form of per-constructor equations) and fixes the actual defect:
  `denUnfold?` now fires only when the argument is an `SHlo` constructor up to `whnfR`, so a graph
  built by a `def` (`denseF`, `fwdGraph`, …) stays folded. No proof in the corpus relied on the
  see-through (full rebuild, zero failures).
* 3.3 minimal form: `@[simp]` dropped from `relu6F_faithful` (dead: `den_relu6F` always won) and
  from `den_dropPathB_ones` / `den_dropoutB_ones` (never fired). Making `reluF_faithful` /
  `relu6F_faithful` the simp normal form was not done: the proc fires on `.reluF` too, so the
  named form would compete with it. `max_zero_eq` → `max_def_lt'`; `patchEmbedBackFlat` deleted.

§4 notes:

* One resolution, `TrainConfig.lossKindFor cfg ds`, beside `DatasetKind` in Types.lean; it reads
  `DatasetKind.pixelLabels`, which a `#guard` beside `datasetIO` pins to the label-record sizes.
  `lossKind : Option LossKind := none`; `useYolov1` is gone. `compileVmfbs` takes the dataset
  (`ds := .imagenette`) instead of `useSeg`; the LM demos (TinyGPT, TinyStories) set
  `lossKind := some .perPixelCE` in their configs instead of passing `useSeg := true`.
* The FPN detectors now resolve to `.yolov1Masked` at `compileVmfbs` too (they got `.classCE` there
  before), so the YOLOv1 modifier checks apply to them. The codegen flag is
  `.yolov1Masked && fpnScales.isEmpty`, the rule `runTraining` already used, so their emitted train
  step is byte-identical (checked: passing the flag through changes only the header comment, to a
  wrong single-grid description). `test-yolov1-mutex` gained C7 for the derived case.
* `useAdam` is gone: 57 `:= true` writers are `optimizer := .adam`, the `:= false` overrides are
  `optimizer := .sgd` (they sit in `{ base with … }` updates, so deleting them would change the
  optimizer). No config set both fields. `effOpt` is gone; `compileVmfbs` rejects `.rmsprop` /
  `.lamb`. `generateTrainStep`'s own `useAdam` parameter is codegen API and stays.
* Checks: `jax/generated/` drift guard (74 artifacts re-emitted, identical), `verified_mlir/` full
  regen (empty diff), `test-yolov1-mutex` C1–C7, every touched exe built, the full gate.

§5 notes (5.2–5.5):

* 5.4 as tabled: the write-only fields are gone from `MBFwd`, `MNV2Fwd`, `BFwd`, `R34Fwd`,
  `R34FwdRecB`, `ENetFwd`, `Mnv4FwdRec`, `UibFwdB`, `BSaves`, `FwdSaves`; the trap comments now
  say the record has no pre-dropout name to read. `verified_mlir/` regen: empty diff.
* 5.2: `CnxDims` is `Vector Nat 4` × 2 (`#v[…]`); indices stay `[i]!`.
* 5.3: `bnChannels` is `VerifiedNetSpec.bnChannels`, read off `layers` by `VLayer.bnWidths`
  (BN γ entries of `toSpecs`, LN-bearing layers excluded) under a new `runningBN` flag; checked
  equal to all 12 hand lists before they were deleted, and empty for the per-example-BN CIFAR-8
  nets (flag off). The R50 kind-1 guards became the definition and are gone; MNv4's conv-output
  guard stays as the independent second route. `dropKeeps` is `dropKeepsAt dropRate` over
  `dropSites`/`dropDenom`; the keeps are bit-identical to the old literals for all 15 specs
  (compared as `Float.toBits`), and the five apps' `LEAN_MLIR_DROP_RATE_U` overrides call
  `dropKeepsAt` instead of inverting with per-net constants. `nClasses` is read off the head
  `.dense` (every spec ends in one). `inC` stays a field: dense-first nets carry `inC·H·W`, not
  `inC`. `VerifiedConfig.lr` is gone (the banner said "lr baked into the render").
* 5.5: `ViTConfig` goes with §8.3. `ValCarry.rows` stays (two adjacent writers on the
  streamed-eval hot path; not worth the risk). `RmsSchedule.warmup` is not a finding: the MNv2
  and B0 Imagenette demos train AdamW by default, and the literal `3` their apps pass is AdamW's
  warmup. `RmsSchedule` is read only by the `rms`-named descent-check variants (not a recipe), which
  share that `3`; the ImageNet apps pass `if rms then sched.warmup else 5`. Mirroring that in the
  two Imagenette apps would only move the `rms` variants (3 → 5), no published number. Open, low.

## 2. Root batch: Foundation/Tensor.lean + Foundation/MLP.lean

### 2.1 Extensionality for the VJP witnesses

`HasVJP` (Tensor.lean:270), `HasVJPAt` (:396), `HasVJPMat` (:628), `HasVJP3` (:988),
`HasVJPAt3` (:1004), `HasVJPMat3` (Architectures/Attention.lean:177). `correct` fixes `backward` at
every input, so each type has at most one element; none has `@[ext]`, `Subsingleton` or `Unique`.
Uniqueness lemmas exist unevenly (`backward_unique{,_of_eq}` for HasVJP/HasVJPAt, `_of_eq` only for
HasVJPMat, nothing for the rank-3 and Mat3 forms), and consumers swap witnesses by hand about 30
times (`rw [HasVJP.backward_unique …]` at EfficientNetSyncStepTieG:407/428/450/471,
ConvNeXtWholeBackCertifiedTie:532, ChapterGraphTies:139, ViTBackB0:195, …).

* Add `@[ext]` (from `backward` equality) and `instance : Subsingleton` for all six. All compile:
  `cases a; cases b; congr; funext …; simp_all` (the Mat3 one ends `ext i j <;> simp_all`).
* Add `canonical` for the five that lack it. No `Unique` instance: it would make `default` a
  witness nobody wrote, and a witness is meant to carry a formula proved equal to the contraction.
* Re-derive `backward_unique` as `by rw [Subsingleton.elim h₁ h₂]`; keep it only as the `rw` form.
* Then `@[ext] CertLayer.ext` from `fwd`, `ok`, `graph` equality (Foundation/CertifiedChain.lean:54;
  compiles once `HasVJPAt` is a subsingleton). `CertLayer.graph` is not free data: it is the emitted
  program, compared term-for-term (`vitTrunkV_graph`, ViTBackNet.lean:119).

### 2.2 Cast-free transport

Witnesses built by `▸`, `by rw …; exact` or `by unfold …; exact` leave `backward` behind an
`Eq.mpr` that `rw` and `simp` cannot peel: `batchMapHasVJP` (BatchMapVJPAt:157), `bnBatchLAHasVJP`
(:198), `bnHasVJP` (BatchNorm:729), `resnet34ForwardBFullHasVJPAt` (ResNet34FullBVJP:488),
`resnet50ForwardBFullHasVJPAt` (ResNet50FullBVJP:420), the three `sealVJP`s, the four `mb*WHasVJP`
(EfficientNetFullB0:203–236). MNv4 hand-built `castLayer` (MobileNetV4BackB0:321) for the same reason.

* `def HasVJP.congr (h : f = g) (hf : HasVJP f) : HasVJP g` and
  `@[simp] HasVJP.congr_backward : (hf.congr h).backward = hf.backward := rfl` (compile), with peers
  for the other five; rebuild the listed witnesses with `.congr`.
* `CertLayer.cast` to replace `castLayer`.

### 2.3 Backward linearity in its home

`HasVJP.backward_smul` (~30 uses in 8 files) lives in `Foundation/DataParallel/Sync.lean:697`; the
rank-3 peer is the non-dot `hasVJP3_backward_smul` (SyncKit.lean:504); nothing has `backward_add`.
Move/add `backward_add` and `backward_smul` for every witness structure in Tensor.lean, or bundle
`HasVJP.backwardₗ hf x : Vec n →ₗ[ℝ] Vec m`. `IsHomog` (Sync.lean:686) then becomes `map_smul`.

### 2.4 flatten / unflatten

* Evaluation is not simp-reachable: plain `simp` makes no progress on
  `Mat.flatten A (finProdFinEquiv (i,j)) = A i j` or the unflatten/Tensor3 forms. Consumer
  unfold sites: `Mat.flatten` 79, `Mat.unflatten` 63, `Tensor3.flatten` 47, `Tensor3.unflatten` 40.
  Add `@[simp] Mat.flatten_finProdFinEquiv`, `@[simp] Tensor3.flatten_finProdFinEquiv` (compile);
  tag `Mat.unflatten_apply`, `Tensor3.unflatten_apply` `@[simp]`.
* `Mat.flatten`/`unflatten`/`transpose`/`mul` have no `@[fun_prop]` (Tensor3's do): 18
  `unfold …; fun_prop` sites (Attention:208–243, PerChannelBN:53, LayerNorm:456, ChannelLN:235/240, …).
  Add the differentiability lemmas; Attention.lean:205–225's helpers collapse.
* Delete the local restatements: `sum_t3`, `sum_s2`, `unflatten_t3Idx` (ConvIndex:55/67/304),
  `w3Idx`, `sum_w3` (ConvFloat:95/102; re-key `convWindow_w3`/`convKernelMat_w3` on `t3Idx`),
  `private sum_heads_3d` (ViTBackB0:209; add `sum_finProdFinEquiv₃'` or inline two lines).

### 2.5 Bridge peels and pdiv

* `HasVJP3.toHasVJP`, `HasVJPAt3.toHasVJPAt`, `HasVJPMat.toHasVJP`, `rowwiseHasVJPMat` have no
  backward lemmas: 19 unfold sites (BackLinks:105–234, ConvBackCertifiedTie:34, …) and one
  argument pasted three times (GradNodesB:370, SgdNodes:185, Bf16GradNodes:125). Add
  `*_backward` rfl peels and `HasVJP3.flatten_backward` (compile).
* `HasFDerivAt.pdiv_eq (h : HasFDerivAt F F' x) : pdiv F x i j = F' (basisVec i) j` (compiles), with
  `pdiv3`/`pdivMat` peers; 7 `unfold pdiv; rw [h.fderiv]` sites.
* Move `gradAt`, `fderiv_eq_zero_of_pdiv_all_zero`, `exists_pdiv_ne_of_fderiv_ne`
  (Training/JacobianSeal.lean:44/55, SgdDescent/Basic.lean:36) here. Replace `sum_smul_basisVec`
  with `(pi_eq_sum_univ v).symm` and `fderiv_apply_eq_sum_grad` with `LinearMap.pi_apply_eq_sum_univ`.

### 2.6 softmax (MLP.lean)

Only `softmax_apply` exists; `softmax_nonneg`/`softmax_le_one` sit in `namespace FloatModel`
(FloatBridge:818/823) without using `M`. Add `softmax_pos`, `softmax_le_one`, `sum_softmax`
beside the def; delete the FloatModel copies (LinearDescent.lean is generated: fix its generator).

### 2.7 Wrappers this batch makes redundant

* SpecVJP.lean:88/143/259/318/369/414/465, the seven `*VerifiedHasVJP` defs: each is
  `HasVJP.canonical _`, no Lean consumer outside the file. Delete, with `mlpVerifiedHasVJP_correct`
  and `linearVerifiedHasVJP_correct` (AuditAxioms-only); fix the NetsCore:133 docstring.
* `.correct` restatements with no consumer: `flatConvStride2XlaHasVJP_correct`,
  `flatConvStride2XlaWeightGradHasVJP_correct` (StridedConv:380/404), `dropPathHasVJP_correct`,
  `dropoutHasVJP_correct` (DropPath:138/271). `trainedMlpHasVJP_correct` (MlpWitness:89) is
  AuditAxioms plus its generator (`scripts/certs/lipschitz_cert_witness_s8.py:219`) only.
  The other ~40 `X_correct` are consumed by `tests/comparator/` and stay.

## 3. Root batch: Codegen/StableHLO/Basic.lean

### 3.1 Missing `den` equations

42 of 215 `SHlo` constructors have no `den_*` lemma and are reachable only through `denStep`:
`sub` (55 construction sites; `den_subB` exists for the batched twin), `gapBack`, `gapBackBatched`,
`seBackBatched`, `seReduceB`, `convBackBatched`, `convStridedBackBatched`, `broadcastBack`,
`bnBatchLABack`, `biasGradB`, `weightGradB`, `patchEmbedBack`, the `depthwise*` back/grad/sgd
family, the `*Bf16`/`*F8` grads, `layerScaleChGammaGradB`, `veclnGammaGradB`, `matmulFBBf16`,
`rowDenseWeightGradB(Bf16)`, `patchEmbedWeightGradB(Bf16)`. Add each as `@[simp] … := rfl` with
the `den` arm verbatim as the right-hand side.

### 3.2 denStep out of the default simp set

`denStep`/`denStepApp` (Basic.lean:2334) are `dsimproc`s, so they are in the default simp set, and
`denUnfold?` unfolds at default transparency: bare `simp` walks through `denseF` and `fwdGraph`
into raw sums, so `denseF_faithful` and `fwdGraph_faithful` never fire. All 104 explicit uses
already name them. Declare with `dsimproc_decl`; fire only when the argument's head is an `SHlo`
constructor after `whnfR`. Lands after 3.1. Expect bare-`simp` sites that relied on the default
proc to need `denStep` named; `lake build Certs` finds them.

### 3.3 One simp normal form per constructor

* `@[simp] relu6F_faithful` (:2900) and `@[simp] den_relu6F` (:2387) share a left-hand side;
  `den_relu6F` always wins, so simp cannot prove `relu6F_faithful`'s own statement.
  ReLU/ReLU6 normalise to the raw lambda while gap, BN and sigmoid normalise to the named function.
  Make the `*_faithful` form the simp lemma for both and drop `@[simp]` from `den_reluF`/`den_relu6F`.
* `den_dropPathB_ones` (:2942), `den_dropoutB_ones` (:2963): never fire (`dropPath_ones_id` and
  `dropout_ones_id` win); drop `@[simp]`.
* `private max_zero_eq` (:2867) is `max_def_lt' a 0`; delete.
* `abbrev patchEmbedBackFlat := @patchEmbedInputGradFormula` (:1537): alias, one use (:2250). Delete.

## 4. Training-path configuration (program side, no proof rebuild)

### 4.1 One loss-kind resolution

`TrainConfig.useYolov1` (Types.lean:774) is kept "for back-compat"; its only `:= true` writer is
`tests/TestYolov1Mutex.lean:25`. `lossKind := .classCE` doubles as "derive me", and the derivation
is written three times: `compileVmfbs` (Train.lean:119), `runTraining` (:534), `train` (:1294).
The `compileVmfbs` copy has no `.detection` arm, so `MainYolov1NeuDetFpn:305` and
`MainYolov1VisdroneFpn:479` validate against `.classCE` while `runTraining` dispatches
`.yolov1Masked`; the emitted loss is right only because `fpnScales` routing comes first
(MlirCodegen:4883). No caller can request `.classCE` explicitly while a soft-label aug is on.

Delete `useYolov1`; `lossKind : Option LossKind := none`; one `TrainConfig.effLossKind cfg ds`
called at all three sites; `compileVmfbs` takes the resolved kind instead of `useSeg`.

### 4.2 One optimizer selector

`useAdam` (Types.lean:523) and `optimizer` (:528) both select the optimizer. JAX resolves the pair
in `private effOpt` (jax/Jax/Codegen.lean:2509); the reference path reads only `useAdam`
(Train.lean:209). Today only `jax/` mains set `optimizer`, so no run trains the wrong optimizer,
but `{optimizer := .adam}` on the reference path would train SGD without a word.
Drop `useAdam` (57 `:= true` writers → `optimizer := .adam`); `compileVmfbs` reads `optimizer` and
throws on kinds it does not lower; delete `effOpt`.

### 4.3 Flag + parameter pairs (low)

`useMixup`/`mixupAlpha` and the same shape for cutmix, knn, erasing, SWA/SWAG, TTA, EMA, focal,
`bf16Conv`, randAugment, expLR: two configs differing only in an off feature's parameter denote the
same run. `Option` payloads. Decide per field; not required for the thread to close.

## 5. Free data in structures

### 5.1 UibParams (MobileNetV4BackB0.lean:471) → Mnv4BWeights

At rows with `preDWk = 0` (9, 15, 19, 20) or `postDWk = 0` (8, 9, 10, 15, 16, 19, 20, 21) the BN
fields `bq eq_ hq gq bq2` / `bd ed hd gd bd2` are read by nothing (forward, backward, graph,
render): 21,420 free reals and 12 positivity proofs about unused ε's. Checked: two records unequal
only there give the same `mnv4BodyOfRow`.

* `DWBnParams c k` for `(W, b, ε, hε, γ, β)`; `DWSlot c : Nat → Type | 0 => PUnit | k+1 => DWBnParams c (k+1)`;
  fields `pre : DWSlot s.ic s.preDWk`, `post : DWSlot (s.ic * s.expand) s.postDWk`.
* Characteristic lemmas: `mnv4PreDWSlot_of_eq_zero`, `_of_ne_zero`, the `mnv4PostDWSlot` pair,
  `mnv4BodyOfRow_fwd_apply`. They replace 11 unfold sites (MobileNetV4FullB:492/520/542/574,
  MobileNetV4SyncB:215/265/305/373, MobileNetV4FullBSeal:219/281/287/303).

### 5.2 CnxDims (Codegen/ConvNeXtRender.lean:85)

`depths`/`dims : Array Nat` have unconstrained length. The render reads stages 0–3; `cnxDropTotal`
(:150), `cnxModelName` and the derived `DecidableEq` read the whole array.
`{depths := #[3,3,9,3,7], dims := cnxTiny.dims}` renders ConvNeXt-T byte for byte and declares 25
drop sites instead of 18. `Vector Nat 4`, literals `#v[…]`, `Fin 4` indices.

### 5.3 VerifiedNetSpec (Verified/Spec.lean)

* `bnChannels` (:289): hand-written, nothing ties it to `layers` (NetsCore:574 says so); R50 and
  MNv4 have a derived `#guard`, R34/MNv2/B0 do not. `{resnet34Verified with bnChannels := #[]}`
  has identical `toSpecs` and threads no running-stat slots instead of 36. Replace with
  `runningBN : Bool` + a per-constructor `VLayer.bnWidths`; derive the list. (A kind filter over
  `toSpecs` is wrong: it counts ConvNeXt/ViT LayerNorm γ, and the cifar8 nets carry BN γ with
  per-example BN and an empty list on purpose.)
* `dropKeeps` (:299): stores the rate-baked ramp; five apps invert it with hand-copied constants
  (MainEfficientNetVerifiedAdam:108 `*15/0.2`, MainConvNeXtVerifiedAdam:131 `*17/0.1`,
  MainViTVerifiedAdam:148 `*11/0.1`, MainConvNeXtSImagenet:109 `/0.4`, MainConvNeXtBImagenet:91
  `/0.5`). Store sites/denominator/rate and derive; drop the copy on `VerifiedNet`.
* `nClasses`, `inC` (:282/288): derive from `toSpecs`, or one generic `#guard` over every spec.
* `VerifiedConfig.lr` (Verified/Train.lean:143): display-only, never set, banner always prints 0.1. Delete.

### 5.4 Write-only fields in Codegen records

Found by scanning every projection used outside each structure's namespace across the 45 modules
that import Codegen, then re-grepped.

| structure | unread fields |
|---|---|
| `MBFwd` (MobileNetV2RenderB:968) | `ec en er dc dn dr pc` |
| `MNV2Fwd` (:1150) | `stc stn str blocks hc hn hr gap` |
| `BFwd` (ResNet34RenderB:46) | `xin a c1 n1 r1 c2 cp` |
| `R34Fwd` (:163) | `stc stn str stp blocks gap` |
| `R34FwdRecB` (:1201) | `stp` |
| `ENetFwd` (EfficientNetRender/Basic:688) | `hr gap` (comment :937: must not be read, use `cin`) |
| `Mnv4FwdRec` (MobileNetV4RenderB:361) | `h1r hr` (comment :945: `fwd.cin`, not `fwd.hr`) |
| `UibFwdB` (:179) | `qn` (always equals `qr`) |
| `BSaves` (Codegen/ViTRender:145) | `q k v` |
| `FwdSaves` (:207) | `embed fln` |

Drop them, and fix the stale docstrings claiming the train step reads them
(MobileNetV2RenderB:1147, ResNet34RenderB:160). Render bytes must not move.

### 5.5 Top-level structures

* `ViTConfig.dh`, `.scale` (LeanMlir/ViTRender.lean): must be `d/h` and `1/√dh`; derive (or see 8.3).
* `ValCarry.rows` (Verified/Train.lean:876): equals `lbl.size / 4`, maintained by hand at :907/:914. Derive.
* `RmsSchedule.warmup`/`.staircase` (NetsCore:27): the Imagenette apps pass a literal warmup of 3
  while `enetRmsSchedule.warmup = 5`. Read the field, or document it as ImageNet-only.

## 6. Certificates and Training API

### 6.1 Optimizer steps (Training/Optim/)

All nine probes fail without unfolding (`sgdParam lr θ g i = θ i - lr * g i`,
`adamMNext … i = …`, `0 ≤ gradSumSq g`, `0 < clipFactor c ε s`, the `lambStep`/`momStep`
projections, …); only `rmsBufNext_apply` exists.

* `@[simp]` `sgdParam_apply`, `momVNext_apply`, `momParam_apply`, `adamMNext_apply`,
  `adamVNext_apply`, `adamWParam_apply`, `clipScale_apply`; projection lemmas for `lambStep`,
  `adamWStep`, `momStep`, `rmsPropStep`; `lambTrust_of_pos`, `gradSumSq_nonneg`, `clipFactor_pos`,
  `clipScale_eq_smul` (then `clipScale_one`/`_zero` are `one_smul`/`zero_smul`).
* Define `lambDir` (Lamb.lean:70) from `adamMNext`/`adamVNext` instead of re-inlining them.
* Delete `rmsSqNext` (= `adamVNext`, RmsPropStep:49) with `_eq_adamVNext`/`_nonneg`,
  `rmsBufScalar`/`rmsBufNext_eq_scalar` (:170/174, unused), `lambDenom_pos` (Lamb:116, pure
  forward); one `0 < √x + ε` lemma instead of three (`adam_denom_pos`, `clipDenom_pos`, `lambDenom_pos`).
  The render already emits `adamVNextF` for the RMSProp slot (StableHLO/Basic:3463).

### 6.2 CertifiedAt and the L2 capstones

`certified_at_eps`, `certified_at_eps_pair` (DenseEuclid:90/257), `lipschitz_margin_certified_radius`
(LipschitzCert/Basic:136) return the body of `CertifiedAt` instead of `CertifiedAt`, so every
generated per-image theorem is stated unfolded (Scorecard:342); `CertifiedAt` has no `.mono` while
`CertifiedAtLinf.mono` exists and is used. Add `CertifiedAt.mono`; restate the capstones; regenerate
the scorecards from their generators.

### 6.3 LipschitzL2 (LipschitzCert/Basic.lean:40)

It is the right-hand side of Mathlib's `lipschitzWith_iff_norm_sub_le`; the docstring's reason for
a separate def ("`LipschitzWith` is ℝ≥0∞-valued") is wrong (K is ℝ≥0). `.comp` is
`LipschitzWith.comp`, `clm_lipschitzL2` is `ContinuousLinearMap.lipschitzWith`, `euclid_norm_sq` is
an alias of `EuclideanSpace.real_norm_sq_eq`. `LipschitzL2` appears in the pinned statement at
`tests/comparator/ChallengeTier.lean:1190`, so restating it changes a challenge statement.
Minimum: `lipschitzL2_iff_lipschitzWith`, `LipschitzL2.mono`, `.comp`/`clm_lipschitzL2` derived
from Mathlib, `euclid_norm_sq` deleted. Decide in-thread whether to go further.

### 6.4 Gaussian and binomial

* GaussianQuantile.lean:21–232 re-proves `Foundation/UpstreamDraft.lean` (`MathlibUpstream`) line
  for line: the `IsOpenPosMeasure` instance (draft :136), `stdNormalCDF_strictMono` (:145),
  `_neg` (:177), `_pos`/`_lt_one`/`_mem_Ioo` (:157–170), the left-limit block (`leftLim_cdf`, :97).
  Nothing imports the draft. Import it and derive.
* `stdNormalCDF` (:39) is unfolded four times and the interval split is proved twice
  (GaussianQuantile:59, PhiBounds:52). Add `stdNormalCDF_eq_real`, `stdNormalCDF_sub`,
  `continuous_stdNormalCDF` (the last from `MathlibUpstream.continuous_cdf_gaussianReal`).
* Smoothing/CP.lean:54/82/120: `binomTail` and `pi_hitCount_eq_binomial` never mention Mathlib's
  `ProbabilityTheory.binomial`. Add `binomTail_eq_binomial`, `map_hitCount_pi` (upstream candidate).
* Smoothing/Gaussian.lean:54/109: `gaussianPDFReal_shift` and the 1-D Cameron–Martin identity
  `integral_gaussianReal_shift_eq` move to UpstreamDraft PR2, generalised to variance `v`;
  `integral_indicator_Iic_gaussianReal` (:62) generalises to any probability measure.

### 6.5 Smaller

* `sum_swap_12_3` (SgdDescent/Cnn:2602) is `Finset.sum_comm_cycle`.
* `one_add_u_le_pow` (private, FloatBridge:119) is `le_self_pow₀`; `div_one_sub_mono` (:481) and
  `sum_channel_fiber` (PerChannelBNGrad:72) are general and go public in Foundation.
* `dropout (_N _n : Nat)` (DropPath:221): phantom arguments; remove.
* `@[simp] clipScale_zero`, `dropPath_zeros_zero`, `dropout_zeros_zero`: right-hand side `0`, not `fun _ => 0`.
* `LipschitzCertDemo` namespace (DenseEuclid:13) is kept only so old citations resolve; choose a
  namespace on its merits.

## 7. Relocations and duplicated infrastructure

### 7.1 Forward stage graphs

Foundation has the backward stage graphs with faithfulness (`StageLayers.lean`, `BackLinks.lean`)
but no forward ones, so every net's block proof unfolds `projB` (21× in 9 files), `cbReluB` (15× in
6), `cbReluStridedB`, `projStridedB`, `cbrB`, `dwbrB`, `cbsB`, `dwbsB`, `residual`, `biPath`
(ResNet34FullB:214/240/258, ResNet50FullB:315/350/384, MobileNetV2FullB:220/241/278,
EfficientNetFullB0:167, ConvNeXtFullT:475, …). Add `*GraphB` builders with `*GraphB_faithful`, and
`residual_eq_add` in vector form.

### 7.2 Lemmas stranded in net files

Each is stated over Foundation/Architectures operations only and used only in its own file.
* EfficientNetSyncStepTieG:341–567 (17): `den_bnBatchLABack_eq_bnBackB`, `cbsB_back_eq`,
  `dwbsB_back_eq`, `dwbsSB_back_eq`, `projB_back_eq` → BackLinks; the `*_smul`/`*_shard` family →
  SyncKit.
* MobileNetV2SyncStepTieB:75/81/208 and `relu6MaskB` (MobileNetV2StepTieB:88) → SyncKit, beside `reluMaskB`.
* ViTBackB0:52/63/74 (`rowDenseBackFlat_eq_backward`, `rowLNBackFlat_eq_backward`,
  `geluFlat_eq_backward`) → the Attention/LayerNorm architecture files.
* `@[fun_prop] globalAvgPoolFlat_continuous` (MobileNetV4FullBSeal:423) → Architectures/CNN.

### 7.3 Net-level duplication

* ConvNeXt: `CnxTieBlk`/`CnxTieDown`/`CnxTieWeights` + `cnxBlockFwdChO` (ConvNeXtStepTie:305/443)
  beside `CnxBlockParamsCh`/`CnxDownParamsCh`/`CnxTWeightsCh` + `cnxBlockChW` (ConvNeXtFullT:106),
  no bridge; the bridge holds by `rfl` (checked). Bind the FullT records with a shared-ε hypothesis
  and drop the Tie records; failing that, add `cnxBlockFwdChO_eq_cnxBlockChW` and the whole-net
  equation to `convNextForwardTCh`. (ViT already bridges via `vitBlockSpelledMHV_eq`.)
* `EnTail` (EfficientNetSyncStepTieG:82) restates 12 fields of `MBW`/`MBWNoExp`
  (EfficientNetFullB0:33/55). Define it in EfficientNetFullB0 and `extends EnTail`.
* `B0Weights` (EfficientNetFullB0:73) pins `fcW : Mat 1280 10`; every other bundle and its own eval
  twin are generic in `nCls`. (Also in the 2026-09-24 correctness audit.)
* The seal files' `Rr_continuous` (ResNet34FullBSeal:817, ResNet50FullBSeal:929) unfold eight
  reducible block defs that `fun_prop` sees through (checked); R34's `sealX_continuous:803`
  re-proves `rayX_continuous`.

### 7.4 Codegen records and helpers declared several times → RenderKit / IndexCast

* `BBackB` (ResNet34RenderB:268) = `MBBackB` (MobileNetV2RenderB:60) = `private UibBackB`
  (MobileNetV4RenderB:599): one `BlockBack {code dx ps}`.
* `MNV2StemFwdB` (:505) = `Mnv4StemFwdB` (:388) = `ENetStemFwdB` (EfficientNetRender/Basic:722):
  one `StemFwdB`.
* The BN μ/var slot pair is spelled four times (`private bnStatSig` MobileNetV2RenderB:427,
  `private bnStatSig4` MobileNetV4RenderB:111, ResNet34RenderB:143, ResNet50RenderB:103–106), a
  driver contract where a misaligned slot fails silently: one `bnStatSlots nm c`.
* `private vitAdamConsts` (ViTRender:657), `private convnextAdamConsts` (ConvNeXtRender:884)
  duplicate `RenderKit.wdzConst` except for comment text: give `wdzConst` a comment argument.
* 15 private zero placeholders (`zV`, `zVb`, `zVB`, `zVv`, …; ViTRender:68, ViTRenderB:71,
  ConvNeXtRender:106, ConvNeXtRenderB:47) are `(0 : Vec n)` etc.: write `0`.
* `private reassoc`/`unassoc` (ConvNeXtRender:265), `private reassocB` (ConvNeXtRenderB:44): their
  `den` lemmas (`den_reassocS`/`den_unassocS`, ConvNeXtChannelLN:55/61) restate the cast with `▸`
  because they cannot name them. Move to `Foundation/IndexCast.lean`, public, and restate.

## 8. Dead and compatibility-only surface (top-level and Verified/)

### 8.1 Delete

* ParamLayouts.lean: `MlpLayout`, `CnnLayout`, `CifarLayout`, and every per-net
  `paramShapes`/`nParams`/`shapesBA`/`xShape` (only `specs`, `packShapes`, `packXShape` are read);
  fix the stale "IreeRuntime re-exports ParamLayouts" docstrings (ParamLayouts:8, IreeRuntime:4–5).
* IreeRuntime.lean:48/61/71/81: `LowererSession.mlpForward`, `mlpTrainStep`, `trainStepPacked`,
  `trainStepF32`, with their C bodies (ffi/iree_lean_ffi.c:120–345).
* F32Array.lean:213/376/509 `dropLoss`, `hflipNCHW`, `yoloHflip`; Cam.lean:94 `writeHeatmapPPM`;
  `VerifiedNet.trainAdamPacked` (Verified/Train.lean:1315, ~90 lines).
* `vitDropFwdBanner` (Codegen/ViTRenderB:628), unused.

### 8.2 Forwarders and double names

* Ten `VerifiedNetSpec.X := s.toNet.X` (Verified/Attack:789–819, Smoothing:327–332,
  Verified/Train:3334–3347); 137 other sites write `.toNet.X`. Delete; 13 app sites use `.toNet`.
* `NetSpec.bnLayers` ≡ `MlirCodegen.collectBnLayers` (SpecHelpers:28): define `NetSpec.bnLayers`
  in MlirCodegen, delete `collectBnLayers`. `sanitizedName` ≡ `sanitize ∘ name` (:67): Train.lean:24
  calls `sanitize` directly; pick one.

### 8.3 LeanMlir/ViTRender.lean

Test-only (its importers are LeanMlir.lean and 9 tests), sitting in the API root.
`vitParamSig` is defined twice with different types (ViTRender:455 and Codegen/ViTRender:489);
`vitTinyConfig` takes an unused `_depth`; `vitFwdModule`/`vitTrainStepModule` (:461/473) have no
callers. Move to `tests/` and drop from LeanMlir.lean.

## 9. Make private (last, after §7 moves)

Bulk sweep; each name checked against every consumer in §0 first.

* Nets: 1,000 of 2,537 public declarations appear only in their own file at 373059db (rerun the
  same-file-use scan when this section starts; §7 moves change the list). Bulk: seal steps (`sc_*`, `pc*`, `ed*`,
  `Z*`/`A*`, `margin*`), sync per-link `_smul`/`_shard`/`*SyncCot*`/`_scaled`. By file:
  EfficientNetSyncStepTieG 133, MobileNetV2 seal 101, MobileNetV4 seal 88, MobileNetV4 sync step
  tie 82, MobileNetV2 sync step tie 77, ResNet50 sync step tie 66, ResNet34 sync step tie 44,
  ResNet50 seal 36, ResNet34 seal 27, ViTBackB0 24, CifarCNN 23. Keep each file's headline theorems.
* SgdDescent/Cnn.lean: 131 of 139 (all of `Conv2Slot`/`Conv1Slot`, `convPadWin`/`cotWin` and their
  `_apply`, the `cnn*_keeps_offkink`/`_close_seg` instantiations, the loss-slice defs). Keep the
  `cnn_conv{1,2}{,_bias}_{,float_}sgd_descends` family, `cnn_conv2_grad_close`, `convTap`, `t3Idx_def`.
  The docstring at :975–983 belongs to a grad-close theorem but sits on `mask_scalar_close`.
* Certificates/Training: CrownBound:331–367 (`getD_replicate_zero` and `getD_map_getD` are
  `simp`/`List.getD_map`: delete), DenseEuclid:42/150/161/169, GaussianQuantile:91–117, PhiBounds,
  CP, NetSemantics:40–122, BatchSealKit (37 of 88), Mlp, Cifar.
* Architectures: `sdpaQChain_eq`, `sdpaKChain_eq`, `mhsaLiftCCLM_apply`, `mhsaEmbedC_eq`,
  `mhsaEmbedC_hasFDerivAt`, `mhsaG_comp_embed`, `mhsaEmbedC_at_proj`, `pdivMat_mhsaG_split_chain`,
  `transformer{Attn,Mlp}Sublayer_inner_flat_differentiable` (Attention); `maxPool2_argmax_unique`,
  `maxPool2Argmax_eq_of_isArgmax` (CNN:973/991); `bnSyncTensor4GradInput_at_own_stats'` (Sync:344).
* StableHLO/Pretty.lean printer internals: `lowOf`, `emitContract`, `liftPointwise2`, `dotInOp`,
  `emitFlatConv`, `emitMatmul`, `lookupEntry`, `lookupShape`, `lookupEntryM`, `lookupShapeM`,
  `noteEntry`, `noteShapeOf`, `noteTokShapes`, `batchOpDescr`, `serializeToks`. Public, they let a
  renderer emit text without going through `pretty`.
* Verified/Train.lean shim and eval plumbing: `wilson95`, `resolveShimScript`, `shimMixDefault`,
  `ShimCfg`, `ShimProc`, `spawnShim`, `readShimBatch`, `spawnShimSharded`, `readShimBatchRR`,
  `EvalRows`, `ValCarry`, `pullValRows`, `reapValStream`, `evalScore`, `writeBinAtomic`,
  `VerifiedNet.printBlurb`. `VerifiedConfig.bnEmaWeight`'s docstring claims a gate on it; add the
  gate or drop the claim.
* Same-file helpers: `MlirCodegen.inputChannels`, `Layer.inChannels`, `F32.dropoutFill`, the
  GradcheckHelpers and E4M3 helpers, `Ddpm.sBias`, the Pong and Blackjack internals.

## 10. Not findings (checked)

* The 165 `den_*` rfl simp lemmas are well oriented; `#lint simpNF simpComm` is clean on
  Certificates and Training.
* Weight bundles (R34/R50/MNv2/B0/ConvNeXt/ViT) and FloatModel, FaithfulFloatModel, ViTTinyWeights
  are extensional; no proof compares two values, so `@[ext]` on them unblocks nothing.
* The Codegen name records other than §5.2/§5.4 are plain String/Nat records, extensional.
* `ConvNeXtRenderB`'s `bEPS`/`bSpats`/`bTiny`/… are deliberate `#guard`ed restatements so the byte
  tie tests the render, not shared constants.
* `IRPrint.lean` is a standalone `#eval` script nothing imports.
* Pre-BN conv biases are read (so field-wise ext holds) but batch BN cancels them; MNv4 binds them to
  zero. An invariance lemma is possible if ever needed; not required.
* SpecVJP's `rfl` ties on `layers` are deliberately drift-sensitive.
