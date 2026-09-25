# Proof-quality audit: Nets/ResNet, Nets/Small, Nets/ConvNeXt

Scope: `LeanMlir/Proofs/Nets/{ResNet,Small,ConvNeXt}/*.lean` (51 files, 19,390 lines; none generated).
The audit was static reading plus grep/awk only. Nothing was compiled. Where a finding predicts a
compile-time effect, it says "predicted" and names the measurement to take.

Scan of the scope:

| signal | sites |
|---|---|
| `set_option maxHeartbeats` | 9 (2 file-wide, unscoped) — 1.0M ×4, 1.6M ×3, 2.0M ×2, **16M ×2** |
| `set_option maxRecDepth` | 12 (2 file-wide) — 4k, 8k, 16k, 32k, 100k ×2, 400k ×2, 800k ×4 |
| `change` | 2 (MnistCNN.lean:378, 384; both at a concrete 4-vector, harmless) |
| `show` in tactic position | ~45; the suspect ones are listed below |
| `decide` | only kernel-parity side conditions (`2*((7-1)/2)+1 = 7`); not a finding |

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean

### ResNet34BackCertifiedTieB.lean:225-321 — `r34InputGradB_eq_r34B_full_vjp` (and ResNet50WholeBackCertifiedTieB.lean:55-56 / 154-155)

**Smell:** heartbeats (maxRecDepth 800000 = 1600× default; maxHeartbeats 1.0M and 2.0M = 5× and 10×)
**Current:**
```lean
set_option maxRecDepth 800000 in
set_option maxHeartbeats 1000000 in
theorem r34InputGradB_eq_r34B_full_vjp ... := by
  unfold r34InputGradB
  rw [cbReluStridedBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      r34HeadBBack_eq_vjp_backward Wd bd (opaqueA16 ...)]
  rfl
```
The apex it closes against (`:145-218`) is a 17-deep `let p1 := vjpCompDiffAt …; … p17.fst`.
**Why it breaks:** the closing `rfl` has the kernel unfold 17 nested `vjpCompDiffAt`/`vjpCompAt`
records and match `PProd.fst` projections through the `let` chain. The file's own docstring
(`:18-20`) says this is "~60 s, an order more than the other four nets' ties", and CI measures
the module at 102 s, the slowest hand-written module in scope. ConvNeXt hit the same wall and
fixed it. `ConvNeXtWholeBackCertifiedTieB.lean:398` adds a peel lemma, and `:442-444` uses it.
Its header (`:40-47`) records the measured result: ~18 min on 4.32.2 and a kernel timeout on
4.34.0 became "checks in seconds".
```lean
theorem vjpCompDiffAt_fst_backward ... :
    (vjpCompDiffAt f g x hf hg).fst.backward dy = hf.fst.backward (hg.fst.backward dy) := rfl
...
  funext dy
  repeat rw [Function.comp_apply]
  rw [convNextForwardTChBHasVJPAt]
  repeat rw [vjpCompDiffAt_fst_backward]
```
ResNet-34 and ResNet-50 did not get that fix.
**Suggested:**
1. Restate `r34BFullHasVJPAt` as one nested term with no `let`s (each `p_k` is used once, so
   the nested form stays linear in size).
2. Close both ties with `funext dy; repeat rw [Function.comp_apply]; rw [r34BFullHasVJPAt];
   repeat rw [vjpCompDiffAt_fst_backward]`. ⛔ Do not use `simp only`: the peel lemma is `rfl`,
   so simp would record nothing for the kernel to replay (the established 48 GB hazard). This is
   one of the places where `rw` is right and `simp` is wrong.
3. Drop both `set_option`s and measure. Predicted: the module falls from 102 s to the ~10 s range
   that ConvNeXt's batched tie reached.

`r34InputGradB_correct` (`:324-421`) carries the same two bumps only because it rewrites with
the tie and then re-elaborates the apex term. The bumps should go once the tie is fixed.

### ResNet34BackCertifiedTieB.lean:145-218 — `r34BFullHasVJPAt`, and the parallel prefix systems

**Smell:** repetition
**Current:** two separate constructions of one VJP:
- `resnet34ForwardBFullHasVJPAt` (ResNet34FullBVJP.lean:253-336) is built over `r34Pre0 … r34Pre16`
  (FullBVJP:187-238), 17 point-free `def`s, plus 17 `r34PreK_apply` lemmas (`:376-443`), each
  `rw [r34PreK, Function.comp_apply]`.
- `r34BFullHasVJPAt` is built over `Foundation/OpaquePrefix.lean`'s `opaqueA0 … opaqueA16`.

ResNet-50 duplicates the first system (`r50Pre0 … r50Pre16` + 17 `_apply` lemmas,
ResNet50FullBVJP.lean:174-415). That makes 68 declarations whose only job is to be prefixes.
**Why it breaks:** the docstring (`:46-53`) says instantiating the generic tie at the concrete
blocks is "a kernel deterministic timeout at six minutes". The reason it gives is "the witnesses
a caller has are at `r34Pre{k-1} N w x` — sixteen defeq checks between two sixteen-deep nested
applications spelled through different definition chains". The mismatch exists because two
prefix vocabularies exist. As long as it does, a `resnet34ForwardBFull_eq_slots` shape check is
needed, with its unrestricted `simp only [r34Pre16, …, r34Pre0]` (`:460`; ResNet-50 at WholeBackCertifiedTieB:289).
**Suggested:** define `r34PreK N w := opaqueA_K (r34StemB N 56 56 w.sW …) (r34IdB N 56 56 w.a0) … (block_K)`.
Build `resnet34ForwardBFullHasVJPAt` as `r34BFullHasVJPAt` at those slots (term mode).
The kernel's per-level check then compares identical spellings. The 34 `_apply` lemmas become
equation lemmas of `opaqueA_K` (one `rw [opaqueA_K]`), and `_eq_slots` collapses to `rfl` or a
`rw` chain. Measure first: the prediction is that the 6-minute timeout goes away because its
recorded cause does.

---

## LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTie.lean, ConvNeXtStepTieGB.lean

### ConvNeXtStepTie.lean:455-608 — `cnx_net_tied_certified`, and ConvNeXtStepTieGB.lean:362-518 — `cnx_net_tiedGB`

**Smell:** heartbeats (maxHeartbeats **16,000,000 = 80×**, maxRecDepth 400,000) + long-proof
**Current:** about 160 loose parameter binders (`aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 … aW18 …`), a
~50-deep `let` telescope, and a proof that only instantiates universally quantified per-block
lemmas:
```lean
  intro ib1 ib2 ib3 ibD0 ... dyO1 dyStem
  refine ⟨cnx_stem_ch_tiedAt ..., ?_, ?_, ... ?_⟩
  · exact cnx_block_ch_tiedAt xN wN bN gN epsStr lrStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 lr
  ...  (×24)
```
**Why it breaks:** every conjunct is an instance of a `∀ xin dyOut` lemma. The proof therefore
does no mathematical work. The 80× budget is spent on elaborating the statement: the let
telescope, and the numeral-shape implicit arguments of ~160 binders at literal `96*56*56` /
`384*14*14`. This is the compile-time smell the brief warns about (statement at the net's
numerals, not variables). It also ignores ConvNeXt's own abstractions:
- `CnxTWeightsCh` (the weights record ConvNeXtWholeBackCertifiedTie uses) exists.
- The depth-`k` stage is already a `Fin k →` family with an induction-proved tie:
  `cnxStageChKBack_eq_vjp` (ConvNeXtWholeBackCertifiedTie.lean:168-181, `| 0, … => rfl | k+1, … =>`).
**Suggested:** restate both capstones over `(w : CnxTWeightsCh nC)`. Replace the 18 unrolled
blocks with a stage lemma proved by induction on `k`, following `cnxStageChKBack_eq_vjp`:
`cnxStage_tied : ∀ k (ps : Fin k → CnxBlockParamsCh …) xin dyOut, CnxStageTied k ps xin dyOut`.
The capstone becomes stem ∧ 4 stages ∧ 3 downs ∧ head. The `@[irreducible]` forward aliases
(`cnxStemFwdO`, `cnxBlockFwdChO`, …; 15 in StepTie, 6 in StepTieGB) exist to hide the unrolled
chain from dimension inference (`:299-301`), so they can go too. Then drop both `set_option`s and
measure.

### ConvNeXtStepTieGB.lean:342-352 (and 15 `@[irreducible]` sites in ConvNeXtStepTie.lean) — `cnxHeadChTiedGBAt` + `cnx_head_ch_tiedGBAt`

**Smell:** undocumented-defeq / non-mainstream idiom
**Current:**
```lean
@[irreducible] def cnxHeadChTiedGBAt ... : Prop := cnxHeadChTiedGB N xN ...
theorem cnx_head_ch_tiedGBAt ... := by
  unfold cnxHeadChTiedGBAt
  exact cnx_head_ch_tiedGB N xN ...
```
**Why it breaks:** each wrapper is a hand-made copy of what Mathlib's `irreducible_def`
generates (`Mathlib/Tactic/IrreducibleDef.lean`). `unfold` on an `@[irreducible]` def is also a
reducibility override that a future `unfold` behaviour change could break.
**Suggested:** `irreducible_def cnxHeadChTiedGBAt … := …`. Use its generated
`cnxHeadChTiedGBAt_def` with `rw`. Better, remove the wrappers entirely (previous finding).

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34StepTieB.lean, ResNet50StepTieB.lean, ResNet34SyncStepTieB.lean, ResNet50SyncStepTieB.lean

### ResNet34StepTieB.lean:502-584 — `r34_net_tiedB`; ResNet50StepTieB.lean:516; ResNet34SyncStepTieB.lean:951; ResNet50SyncStepTieB.lean:895

**Smell:** heartbeats (1.6M = 8×, at four capstones)
**Current:** a statement with 17-35 `let`s followed by
```lean
  intro g dyE1 dyE0 dyD4 ... cotPool
  exact ⟨r34_stem_tiedB N 56 56 ..., r34_idblock_tiedB N 56 56 ... (r34Pre0 N w x) dyA0, ...⟩
```
**Why it breaks:** same shape as ConvNeXt: `r34_idblock_tiedB` is `∀ xin dyOut`, so the term is
18 instantiations. `r50_net_tiedB` has `q` as a binder and still needs 1.6M, so numerals are not
the whole cost. The let telescope is the common factor.
**Suggested:** name the cotangent chain as `def`s (`r34CotE1 N w x g := r34HeadCotBlk …`,
`r34CotE0 := r34IdCotIn … (r34CotE1 …)`, …), as the forward already is (`r34PreK`). The statement
then has no `let`s. Drop the bumps and measure. The same named cotangents serve the sync
capstone (`ResNet34SyncStepTieB.lean:968-1082`), which currently restates all 17 lets at
`R * N`.

### ResNet34FoldB.lean:57-160 vs :218-320 — `convWGradB_den` / `ConvWTiedB` (and the 218 delegation sites)

**Smell:** repetition
**Current:** every `*TiedB` Prop def restates its `_den` lemma's conclusion verbatim, and every
block tie then proves it by delegation:
```lean
def ConvWTiedB ... : Prop := ∀ idx, den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx = ∑ n, ∑ j, pdiv ...
theorem convWGradB_den ... (idx) : den (SHlo.convWeightGradB ...) idx = ∑ n, ∑ j, pdiv ...   -- same text
...
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.W₁ xin p.b₁ cotC1 o
```
`grep -E "· intro (idx|o|i j|j|i);? +exact [A-Za-z0-9_.]+_den"` finds **218 sites in 13 files**:
R34StepTieB 18, R50StepTieB 22, Cifar8StepTie 16, Cifar8BnStepTie 16, CifarFold 8,
ConvNeXtStepTie 8, ConvNeXtStepTieGB 10, and more outside scope.
**Why it breaks:** two copies of every gradient formula must be kept in sync by hand. When the
`den` of a `*GradB` constructor changes, both texts change.
**Suggested:** state each lemma as the Prop. For example,
`theorem convWGradB_den … : ConvWTiedB N h w xN cotN b x W cot := fun idx => by simp only [den]; …`.
Block ties become terms, `⟨convWGradB_den …, convBGradB_den …, bnPairTiedB_holds …, …⟩`, and the
`unfold r34IdTiedB; intro r1 c1 …; refine ⟨?_,…⟩` scaffolding
(`ResNet34StepTieB.lean:358-371`, `:405-422`, `:439-451`; ResNet50StepTieB.lean:487-507; and
their sync twins) goes away.

### ResNet34SyncStepTieB.lean:58-278 — the `_smul` family (§1 "Homogeneity")

**Smell:** repetition / non-mainstream idiom
**Current:** 33 `_smul` lemmas in this file, 23 in ResNet50SyncStepTieB, **151 repo-wide in 12
files**. Every statement uses the spelled-out scalar action:
```lean
theorem r34IdCotC2_smul ... :
    r34IdCotC2 N h w p xin (fun i => s * dy i) = fun i => s * r34IdCotC2 N h w p xin dy i := by
  unfold r34IdCotC2; rw [r34IdCotA_smul, bnInB_smul]
```
**Why it breaks:** `fun i => s * dy i` is `s • dy` (`Pi.smul_apply`, `smul_eq_mul`) with none of
Mathlib's `smul` API attached. The linearity of each chain link is therefore re-proved by hand,
link by link, once per net, and every new net (the MobileNet, EfficientNet and bf16 sync files)
repeats it.
**Suggested:** add a predicate and its composition lemmas to the leaf that already holds
`HasVJP.backward_smul`, `Foundation/DataParallelSync.lean` (2 direct importers):
```lean
def CotHomog {a b} (f : Vec a → Vec b) : Prop := ∀ (s : ℝ) v, f (s • v) = s • f v
theorem CotHomog.comp (hg : CotHomog g) (hf : CotHomog f) : CotHomog (g ∘ f)
theorem CotHomog.add ... ; theorem HasVJP.cotHomog (hf : HasVJP f) x : CotHomog (hf.backward x)
```
Each `r34*Cot*_smul` is then `(… ).comp (…)` or a one-line `CotHomog` term. Bundling the
backward maps as `Vec a →ₗ[ℝ] Vec b` would give `map_smul` for free; that is the fully
mainstream option, but it is more invasive.

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34FullBVJP.lean, ResNet50FullBVJP.lean

### ResNet34FullBVJP.lean:49 and ResNet50FullBVJP.lean:52 — file-wide `set_option maxHeartbeats 1000000`

**Smell:** heartbeats (5×, unscoped)
**Current:** `set_option maxHeartbeats 1000000`, with no `in`, directly under `namespace Proofs`.
**Why it breaks:** it raises the budget for every declaration in both files, including 17
`rw [r34PreK, Function.comp_apply]` one-liners. It hides which declaration (presumably the apex)
actually needs it, and any future slow proof added to these files is silently absorbed.
**Suggested:** delete the file-wide option. Put `set_option maxHeartbeats N in` on the one
declaration that fails without it. That is probably `resnet34ForwardBFullHasVJPAt`
(`:253-336`), and the next finding may remove the need altogether.

### ResNet34FullBVJP.lean:253-336 — `resnet34ForwardBFullHasVJPAt` (same shape: ResNet50FullBVJP.lean:241, CifarCNN.lean:77/441/803, ConvNeXtFullT.lean:101/214/294)

**Smell:** long-proof / non-mainstream idiom (data built in tactic mode)
**Current:**
```lean
noncomputable def resnet34ForwardBFullHasVJPAt ... : HasVJPAt (...) x := by
  have dS : DifferentiableAt ℝ (r34Pre0 N w) x := ...
  have vS : HasVJPAt (r34Pre0 N w) x := ...
  have e1 : HasVJPAt (r34Pre1 N w) x := vjpCompAt _ _ x dS d1 vS (...)
  ... (84 lines, 16 × {d_k, e_k, f_k})
```
**Why it breaks:** `HasVJPAt` is data (it has a `backward` field). A `have` of data elaborates to
`letFun`, and `.backward` then does not reduce. ConvNeXtWholeBackCertifiedTie.lean:89-94
documents the cost: "`convNextForwardTChHasVJP` is a tactic proof, so its eleven `have`s are
`letFun` and its `.backward` does not reduce; the whole-net `rfl` against it returned no result
at `maxHeartbeats 8000000`, twice, ~8 min each." That is why ConvNeXt carries a second,
term-mode copy (`cnxV0 … cnxV11`, `cnxV1_backward … cnxV11_backward`, `convNextForwardTChVjpChain`,
`:264-520`) and why ResNet carries `r34BFullHasVJPAt`. Every net pays for a second apex.
**Suggested:** follow the mainstream convention: data in term mode, `have` only for `Prop`s.
Write the apex as one nested `vjpCompAt`/`vjpCompDiffAt` term, or as the generic apex at
the slots (see the `r34BFullHasVJPAt` finding). Keep the differentiability facts as `Prop`
side terms. The duplicate term-mode chains can then be deleted after measurement.

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34FullBSeal.lean, ResNet50FullBSeal.lean

### ResNet34FullBSeal.lean:647-686 — `seal_differentiableAt` (ResNet50FullBSeal.lean:719, 65 lines; MobileNetV2/V4 FullBSeal have a third and fourth copy)

**Smell:** repetition / long-proof
**Current:**
```lean
theorem seal_differentiableAt (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (resnet34ForwardBFull 2 (sealW nCls)) (sealX t) := by
  rw [show resnet34ForwardBFull 2 (sealW nCls) = ... from funext (resnet34ForwardBFull_eq_chain 2 (sealW nCls))]
  have f0 : DifferentiableAt ℝ (r34Pre0 2 (sealW nCls)) (sealX t) := ...
  have f1 := (r34IdB_differentiableAt 2 56 56 (sealW nCls).a0 (seal_id_pos 64) _ (sc_a0 nCls t)).comp (sealX t) f0
  ... f16
```
**Why it breaks:** this is line for line the `f_k` half of the apex's own proof. The apex builds
the same differentiability internally (FullBVJP:292-336) but does not export it, so all four
seals rebuild it at the net's numerals.
**Suggested:** export `resnet34ForwardBFull_differentiableAt` from ResNet34FullBVJP.lean with
the apex's hypotheses, or have the apex return a `PProd (HasVJPAt …) (DifferentiableAt …)` as
`vjpCompDiffAt` already does. `seal_differentiableAt` becomes one application. Do the same in R50,
MNv2 and MNv4.

### ResNet34FullBSeal.lean:382-622 and :934-973 — `nn0 … nn15`, `pc0 … pc16`, `sc_a0 … sc_e1`, `ed0 … ed16`, `cn0 … cn13`

**Smell:** repetition
**Current:** 77 lemmas, one per slot per fact. Each one instantiates a variable-shape lemma, for
example
```lean
theorem nn9 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre9 2 (sealW nCls) (sealX t) k := by
  intro k; rw [r34Pre9_apply]; exact r34IdB_nonneg 2 14 14 256 _ _ k
theorem pc9 ... := by rw [r34Pre9_apply]; exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn8 nCls t)
theorem ed9 ... := by rw [pc9]; exact eDiff_shift _ _ 1 (ed8 nCls t)
theorem cn9 ... := (r34IdB_continuous 2 14 14 256 (sealW nCls).c0 one_pos one_pos).comp (cn8 nCls)
```
The comm of the two files' name lists shows ~70 of these names shared with ResNet50FullBSeal.lean.
**Why it breaks:** these are not slow (they correctly instantiate at the numerals rather than
prove there), but ~150 statements across two files spell `2 (sealW nCls) (sealX t)` at a
concrete index.
**Suggested:** one identity-slot step lemma at variable shapes in the seal file (or in
`Training/BatchSealKit.lean`, which has one direct importer, so it is a leaf):
```lean
theorem sealIdSlot (N h w c : Nat) (hn : 0 < N * (h * w)) {d : Fin c → ℝ} (v : Vec (N * (c * h * w)))
    (hv : ∀ k, 0 ≤ v k) (he : EDiff d v) :
    r34IdB N h w (sealIdW c) v = (fun k => v k + 1) ∧ (∀ k, 0 ≤ r34IdB N h w (sealIdW c) v k)
      ∧ EDiff d (r34IdB N h w (sealIdW c) v) ∧ R34IdSmoothAt N h w (sealIdW c) v
```
Thread it once per slot, which cuts four lemmas per identity slot to one. Replace `cn0 … cn13`
with one general `r34Pre_continuous : (∀ positivity bundle) → Continuous (r34PreK N w)` in
ResNet34FullBVJP.lean. That lemma is true for all weights, not only the seal's.

### ResNet34FullBSeal.lean:138-214 — `projB_zero_const`, `sealIdB_eq`, `sealDnB_eq` (ResNet50FullBSeal.lean:190-330 repeats the skeleton 3×)

**Smell:** undocumented-defeq + repetition
**Current:**
```lean
  funext k
  show StableHLO.bnBatchLA N oc h w ε γ β (StableHLO.batchMap N (flatConv W b) u) k = bb   -- :144
...
    show _ + v k = v k + 1                                                                   -- :174
  show relu (N * (c * h * w)) (residual _ v) k = v k + 1                                   -- :178
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [hv i]), hres k]
```
The R50 file has 11 more such `show`s (`:199, 203, 238, 241, 265, 268, 300, 307, 314, 325`).
**Why it breaks:** each `show` depends on the unstated defeqs `projB = bnBatchLA ∘ batchMap flatConv`,
`r34IdB = relu ∘ residual body`, and `residual F v = F v + v`. Reordering `residual`'s summands, or
adding a reassociation cast to `r34IdB`, breaks every `show` silently, and none carries a comment.
**Suggested:** add `@[simp] theorem r34IdB_apply`, `r50IdB_apply`, `projB_apply` (each is `rfl`)
next to their defs, and `rw` with them instead of `show`. Then factor the five copies of "a
residual block with a constant body on a nonnegative input is a shift" into one lemma in
`Training/BatchSealKit.lean`:
`relu_residual_const {n} (F) (v) (c) (hF : F v = fun _ => c) (hpos : ∀ k, 0 < v k + c) : relu n (residual F v) = fun k => v k + c`.

---

## LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean, ConvNeXtWholeBackCertifiedTie.lean

### ConvNeXtWholeBackCertifiedTieB.lean:205-320 — the batched leaf ties `cnxStemBackB_eq_vjp`, `cnxChanLNBackB_eq_vjp`, `cnxStageBackB_eq_vjp`, `cnxDownBackB_eq_vjp`, `cnxLNhBackB_eq_vjp`, `cnxDenseBackB_eq_vjp`

**Smell:** undocumented-defeq + repetition (6 here, 3 more in ViTWholeBackCertifiedTieB.lean)
**Current:**
```lean
  funext dy idx
  show chanLNTensor3Back c h w ε γ (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [chanLNTensor3Back_eq_chanLN_vjp (β := β) ε hε γ]
  rfl
```
**Why it breaks:** every copy hard-codes the internal index layout of `batchMapAux` and of
`batchMapHasVJPAt`'s `backward` field (`Foundation/BatchMapVJPAt.lean:83-90`). Changing
`finProdFinEquiv` to `Fin.divNat`/`modNat`, or `Mat.unflatten` to `batchSlice`, breaks nine
proofs in two files.
**Suggested:** state the row-lift once, in the leaf `Foundation/BatchMapVJPAt.lean` (3 direct
importers):
```lean
theorem batchMapAux_eq_batchMapHasVJPAt_backward {N a b : Nat} (f : Vec a → Vec b)
    (g : Vec a → Vec b → Vec a) (v : Vec (N * a))
    (hf : ∀ r, HasVJPAt f (Mat.unflatten v r)) (hd : ∀ r, DifferentiableAt ℝ f (Mat.unflatten v r))
    (hg : ∀ r, g (Mat.unflatten v r) = (hf r).backward) :
    StableHLO.batchMapAux N g v = (batchMapHasVJPAt f v hf hd).backward := by
  funext dy idx; exact congrFun (congrFun (hg _) _) _
```
Add a `batchMap` twin for point-independent backwards (the stem and dense cases). Each leaf tie
becomes `batchMapAux_eq_… _ _ _ _ _ (fun r => chanLNTensor3Back_eq_chanLN_vjp …)`.

### ConvNeXtWholeBackCertifiedTieB.lean:64 and ConvNeXtWholeBackCertifiedTie.lean:491 — file-wide `set_option maxRecDepth 100000`

**Smell:** heartbeats (maxRecDepth 196×, unscoped)
**Current:** `set_option maxRecDepth 100000` with no `in`.
**Why it breaks:** both headers say the numeral-spelled leaves that needed this were moved to
variable dimensions (`cnxChanLNBackB_eq_vjp` at `:220-222`: "stated at the literal `96 56 56` the
closing `rfl` recurses past maxRecDepth 100000; at variables … closes at once"), and the ties now
peel by `rw`. The file-wide bump may be vestigial. As written, it hides any new numeral-shape
`rfl` that is added to these files.
**Suggested:** remove it. Re-add `set_option maxRecDepth … in` only on the declarations that
fail (at most the `cnxT*`/`cnxSavedB*` iotas).

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34.lean (+ Foundation/BackwardMaps.lean import)

### ResNet34.lean:74 — `vjpCompDiffAt`, and ConvNeXtWholeBackCertifiedTieB.lean:398 — `vjpCompDiffAt_fst_backward`

**Smell:** repetition / layering (hidden shared lemma in per-net files)
**Current:** the generic two-stage composition (no ResNet content) is defined in a ResNet file,
and its peel lemma is defined in a ConvNeXt file. `vjpCompDiffAt` is used by 6 files
(ViT, MobileNetV2, MobileNetV4, EfficientNet, ResNet ties); `_fst_backward` is used only by
ConvNeXt. Related: `Foundation/BackwardMaps.lean:2` imports `Nets.ResNet.ResNet34`. A grep of
every declaration name in ResNet34.lean and MnistCNN.lean shows BackwardMaps uses only
`decimateIdx`/`decimateFlat`, which live in `Foundation/StridedConv.lean`. So a Foundation file
depends on a per-net file, and every BackwardMaps consumer (9 direct importers) rebuilds when
ResNet34.lean changes.
**Why it breaks:** nets that need the peel (ResNet-34 and ResNet-50, first finding) cannot find
it without importing ConvNeXt. Any edit to ResNet34.lean rebuilds the Foundation tier above it.
**Suggested:** move `vjpCompDiffAt` and `vjpCompDiffAt_fst_backward` into
`Foundation/OpaquePrefix.lean`. It imports only `Foundation/Tensor` (where `vjpCompAt` is) and
has 4 direct importers, so it is a leaf. Change BackwardMaps' import to
`Foundation.StridedConv` (plus whatever `lake build Certs` proves missing). This is not a root
file, so it is not a 300-module rebuild. The same smell appears in `ResNet34BackCertifiedTie.lean:1`
importing `Nets.Small.CifarCNN`, and in ConvNeXt files importing `ResNet34Fold` /
`ResNet34BackCertifiedTie`: shared leaf lemmas sit in per-net files.

---

## LeanMlir/Proofs/Nets/Small/*.lean

### CnnFold.lean:210, CifarFold.lean:189, Cifar8StepTie.lean:72, Cifar8BnStepTie.lean:47 — `*_conv_tied_certified`

**Smell:** heartbeats (maxRecDepth 4000 / 8000 / 16000 / 32000)
**Current:** the budget doubles as the let telescope grows. There are 12 `let`s at 4000
(`cnn_conv_tied_certified`), 23 at 8000 (`cifar_conv_tied_certified`), and 42 at 16000
(`cifar8_convs_tied_certified`). Every proof is
```lean
  intro xv cc1 r1 r1t cc2 ... cotC1
  refine ⟨?_, ?_, ... ⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₁ x W₁ cotC1 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₁ x b₁ cotC1 lr o
  ...
```
**Why it breaks:** `maxRecDepth` rises with the length of the `let` telescope, not with the
proof: `convW_den`/`convB_den` hold for every activation and cotangent. The next deeper net needs
another doubling.
**Suggested:** replace the telescope with named activation `def`s (as ResNet does with `r34PreK`)
or a record-valued forward, e.g. `cifar8Acts … : Cifar8Acts` with fields `cc1 r1 …`. Add the
`ConvW/BSgdTied` pair lemma `convPair_tied : ConvWSgdTied … ∧ ConvBSgdTied …` (in
`CnnChainClose.lean`, where both Props are defined). Each conv then costs one conjunct term.
Drop the bumps.

### CifarCNN.lean:441-690 — `cifarCnn8HasVJPAt` (and `cifarCnnBn8HasVJPAt` :803-1026, `cifarCnnHasVJPAt` :77-198)

**Smell:** long-proof + repetition
**Current:** the hypotheses restate the whole forward prefix inline, so each is quadratic in
depth (`hf8`, `hfa` are ~14 lines each of nested `relu ∘ flatConv … maxPoolFlat …`). The body
repeats this pool step 4× per net (20 sites in the file, plus 2 in MnistCNN):
```lean
  have hpt1 : Tensor3.flatten (Tensor3.unflatten z1 : Tensor3 c1 ...) = z1 := Tensor3.flatten_unflatten z1
  have mp1_v : HasVJPAt (maxPoolFlat c1 ...) z1 := by
    rw [← hpt1]; exact maxPoolFlatHasVJPAt _ hp1
  have mp1_d : DifferentiableAt ℝ (maxPoolFlat c1 ...) z1 := by
    rw [← hpt1]; exact maxPoolFlat_differentiableAt _ hp1 hc1 (by omega) (by omega)
```
**Why it breaks:** a raw-point pool VJP `maxPoolFlatHasVJPAt'` already exists
(Codegen/StableHLO.lean:4152), but CifarCNN does not import it and re-derives it with
`rw [← hpt]` 20 times. The inlined hypotheses break on any change to the forward's spelling.
**Suggested:** add `maxPoolFlat_differentiableAt'` (raw point `v`, `MaxPool2Smooth (Tensor3.unflatten v)`)
and move `maxPoolFlatHasVJPAt'` out of the root file into `Nets/Small/MnistCNN.lean`, which
CifarCNN already imports (5 direct importers). Name the prefixes (`cifar8PreK`), as
ResNet34FullBVJP does, so each `hf_k` is `∀ k, flatConv … (cifar8Pre_{k-1} … x) k ≠ 0`. Write the
three apexes in term mode (see the tactic-mode data finding).

### MlpFold.lean:132, CnnFold.lean:148, CifarFold.lean:121, Cifar8StepTie.lean:28, Cifar8BnStepTie.lean:25, ConvNeXtStepTie.lean:293 — `*LossCot_den`

**Smell:** repetition (6 copies)
**Current:** six lemmas, each restating a net's whole binder list (up to 22 weights) to prove
```lean
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN <forward …>)))
          (.operand ohN (oneHot K label))) = fun j => softmax K <forward …> j - oneHot K label j := by
  funext j; simp only [den, softmax]
```
The forward term is irrelevant to the proof. `cnxLossCot_den` already has the generic shape
(`logits : Vec 10`), but hard-codes `K = 10`.
**Suggested:** one lemma
`softmaxCELossCot_den {K} (nlogN ohN : String) (logits : Vec K) (label : Fin K) : … := by funext j; simp only [den, softmax]`
in `Nets/Small/LinearTrainStep.lean`, next to `lossCot_eq_softmax_sub_onehot` (it imports only
Codegen.StableHLO). The six per-net lemmas become instances or disappear. The same applies to
the dense-head `*_tied_totalloss` family (MlpFold ~:150, CnnFold ~:180, CifarFold ~:170,
Cifar8StepTie :55, ConvNeXtStepTie :280). Make ConvNeXt's head-input-generic form generic in `K`
and instantiate it. The current CnnFold/CifarFold copies pass the entire nested forward as an
explicit `rw` argument (CifarFold.lean:177-184), which breaks on any forward refactor.

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34BackB0.lean

### ResNet34BackB0.lean:149 and :323 — `cbReluBackBatchedGraph_faithful`, `cbReluStridedBackBatchedGraph_faithful`

**Smell:** fragile-simpa (definitional `simp only`)
**Current:**
```lean
  simp only [cbReluBHasVJPAt, bnReluStageHasVJPAt, stageHasVJPAt, vjpCompAt,
    HasVJP.toHasVJPAt, Function.comp_apply]
```
**Why it breaks:** every lemma in the set is a definition unfolding, so this is a pure `dsimp`
and the kernel re-derives the equality by unfolding. At these variable-shape leaves it is cheap
today. It is the same construct the project measured at 48 GB in a whole-net tie, and it breaks
if any of the four VJP builders is refactored from a structure literal to a transported term
(the `▸` issue noted at BatchMapVJPAt.lean:80-82).
**Suggested:** add a `vjpCompAt_backward : (vjpCompAt f g x … hf hg).backward dy = hf.backward (hg.backward dy) := rfl`
peel lemma, the `HasVJPAt` twin of `vjpCompDiffAt_fst_backward`, in the file that defines
`vjpCompAt`. Note that Foundation/Tensor.lean is a root (~420-module rebuild), so park the
lemma in `Foundation/OpaquePrefix.lean` with the other peel. Then close with
`rw [cbReluBHasVJPAt, vjpCompAt_backward, …]` or plain `rfl`.

---

## LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean, ResNet34FullBVJP.lean (statement size)

### ResNet34BackCertifiedTieB.lean:330-421 — `r34InputGradB_correct` (R50 twin :158-238; ResNet34FullBVJP.lean:459-504 `_correct`)

**Smell:** repetition
**Current:** each `_correct` corollary copies the tie's 16 `PProd (HasVJPAt …) (DifferentiableAt …)`
binders (≈32 lines), or the apex's 35 bundle binders, before proving
`rw [congrFun (tie …) dy]; exact (apex …).correct dy i`.
**Why it breaks:** there are two copies of a 30-50-line binder list per net. A hypothesis added
to the apex has to be added in 3 places.
**Suggested:** one generic leaf lemma (in `Foundation/OpaquePrefix.lean`):
`HasVJPAt.correct_of_backward_eq (hv : HasVJPAt f x) (h : G = hv.backward) (dy i) : G dy i = ∑ j, pdiv f x i j * dy j := h ▸ hv.correct dy i`.
Bundle the per-block smoothness into one `structure R34BSmoothAt N w x : Prop` (fields `sa0 … se1`),
as `R34IdPos` already does per block. The corollaries then drop to 3-5 lines each.

---

## Recurring patterns (ranked by payoff)

1. **Statements hold at a let telescope or the net's numerals; the proof only instantiates `∀`-lemmas.
   The budget bump pays for elaboration, not mathematics.** Sites: 11 `set_option`s at whole-net
   capstones and ties. Per module: ConvNeXtStepTie and StepTieGB at 16M/400k, R34/R50 StepTieB and
   R34/R50 SyncStepTieB at 1.6M, CnnFold/CifarFold/Cifar8StepTie/Cifar8BnStepTie at 4k→32k
   `maxRecDepth` (doubling with let depth), R34/R50 whole-back ties at 800k/1-2M. The common fix:
   name the activations and cotangents as `def`s or records instead of `let`s; take weights as a
   structure (`CnxTWeightsCh`, which exists); use `Fin k`-indexed stage recursion
   (`cnxStageChKBack_eq_vjp` already proves the ConvNeXt stage by induction on `k`). For VJP
   chains, peel with `rw [vjpCompDiffAt_fst_backward]` rather than `rfl` (ConvNeXt measured
   18 min → seconds; ResNet has not been ported).
2. **Two copies of each whole-net apex, because the first was built in tactic mode.** `HasVJP`/`HasVJPAt`
   data is assembled with `have` chains (letFun) at 8 sites: R34/R50 FullBVJP, CifarCNN ×3,
   ConvNeXtFullT ×3. Every whole-back tie then carries a second, term-mode apex
   (`r34BFullHasVJPAt` + `opaqueA*` alongside `r34Pre*` + `r34Pre*_apply`; `cnxV0…11` + `cnxV1_backward…11`),
   and every seal re-derives the hidden differentiability (`seal_differentiableAt` ×4 nets). That is ≈150
   declarations across ResNet/ConvNeXt/MNv2/MNv4. The fix: data in term mode; one prefix
   vocabulary (define `r34PreK` as `opaqueA_K` at the slots); export
   `*_differentiableAt` next to each apex.
3. **Per-slot and per-node restatement of one generic fact.** Sites: 218 `intro idx; exact *_den … idx`
   delegations to `_den` lemmas whose statements duplicate the `*TiedB` defs (13 files); 151 `_smul`
   homogeneity lemmas with hand-spelled `fun i => s * v i` (12 files); ~150 seal slot lemmas
   (`nn/pc/sc/ed/cn`, R34+R50); 9 batched leaf ties that each `show` the `batchMapAux` index
   layout (ConvNeXt ×6, ViT ×3); 6 `*LossCot_den` and 5 `*_tied_totalloss` copies; 20
   `rw [← hpt]; exact maxPoolFlat_*` sites. Each has a named generic replacement above, placed in
   a leaf: `Foundation/OpaquePrefix.lean`, `Foundation/BatchMapVJPAt.lean`,
   `Foundation/DataParallelSync.lean`, `Training/BatchSealKit.lean`, `Nets/Small/LinearTrainStep.lean`
   or `Nets/Small/MnistCNN.lean`. None goes in Tensor/StableHLO/Lamb.
