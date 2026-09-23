# Proof-quality audit: `Nets/EfficientNet/*` + `Nets/ViT/*` (~12.8k lines)

Static read only. Nothing compiled, nothing edited. No file in scope is marked generated. Toolchain v4.34.0.
All Mathlib and core names suggested below were grepped and exist (`NeZero.mul` instance,
`Algebra/GroupWithZero/Basic.lean:92`; core simprocs `Fin.reduceEq` / `Fin.reduceNe`,
`Lean/Meta/Tactic/Simp/BuiltinSimprocs/Fin.lean:104,106`; `LinearMap.comp`, `map_smul`). Mathlib has
**no** `IsLinearMap.comp`, so any linearity combinator has to be bundled `→ₗ[ℝ]` or written locally.

Project hazards respected: I do **not** recommend replacing any whole-net `rw` peel with
`simp only [rfl-lemmas]`. Where a finding touches a whole-net tie, the fix is to change *what is
stated*, meaning variable shapes, bundles or a fold, and never to switch the tactic.

Findings are ordered by payoff (compile time first), grouped by file.

---

## ViT

### LeanMlir/Proofs/Nets/ViT/ViTStepTieGB.lean:375 — `vit_net_tiedGB` (and ViTStepTie.lean:283 `vit_net_tied_certified`)

**Smell:** heartbeats (16,000,000 = 80× default, + `maxRecDepth 400000`), repetition, work at numeral shapes
**Current:**
```lean
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 400000 in
theorem vit_net_tiedGB (N : Nat) {nC : Nat} ...
    -- block 1
    (lnG1_1 lnB1_1 lnG2_1 lnB2_1 : Vec 192) (mWq_1 mWk_1 mWv_1 mWo_1 : Mat 192 192) ...
    ... (12 × 16 = 192 block binders)
    let ib2 : Vec (N * (197 * 192)) := batchMap N (vitBlockFwdOMHV (Np1 := 197) (heads := 3) (d := 64) ε lnG1_1 ...) ib1
    ... 28 lets ...
    vitBlockTiedGBAt N (Np1 := 197) (heads := 3) (d := 64) ... ib1 dy1 ∧ ... (15 conjuncts) := by
  intro ib1 ... dyEmbed
  refine ⟨?_, ...⟩
  · exact vit_block_tiedGBAt N (Np1 := 197) ... ib1 dy1
  ...
```
**Why it breaks:** Each conjunct is an instance of `vit_block_tiedGBAt`, which is quantified over
**every** `xin` and `dyOut`. The threading therefore carries no proof content. The 80× heartbeat budget
goes entirely on elaborating and zeta-unifying 28 `let`s and 192 binders at the literals
197/192/768/3/64. That is the "whole-net statement at the net's numerals" hazard. The same file
imports `ViTDepthK`, which already has `structure BlockParamsV` (ViTDepthK.lean:38) and a depth-`k`
fold `vitBodyKV` over `Fin k → BlockParamsV`. Neither capstone uses them. Both files repeat the
192 binders verbatim: 107 and 104 lines, the two longest ViT declarations.
**Suggested:** State the capstone at variable depth and dims, then instantiate:
```lean
/-- block inputs / output cotangents of the depth-k chain, by recursion on k -/
noncomputable def vitIbB  (N Np1 heads d mlpDim ε) : (k : Nat) → (Fin k → BlockParamsV (heads*d) mlpDim) → Vec (N*(Np1*(heads*d))) → Fin k → Vec (N*(Np1*(heads*d)))
noncomputable def vitDyB  ... -- the dual fold from the top cotangent
theorem vit_body_tiedGB (N k) (ps : Fin k → BlockParamsV (heads*d) mlpDim) (x dyTop) :
    ∀ i : Fin k, vitBlockTiedGBAt N ... (ps i) (vitIbB ... k ps x i) (vitDyB ... k ps x dyTop i) :=
  fun i => vit_block_tiedGBAt N ... (ps i) _ _
```
Then `vit_net_tiedGB` is `⟨vit_body_tiedGB .., vit_finalLN_tiedGB .., vit_head_tiedGB .., vit_embed_tiedGB ..⟩`,
and the ViT-Tiny instance is a one-line specialisation at `k := 12`. Both `set_option`s should
become unnecessary. Put `vitIbB`/`vitDyB` beside `vitBodyKV` in ViTDepthK.lean, which is a leaf.

### LeanMlir/Proofs/Nets/ViT/ViTStepTieGB.lean:83–170 — `blkSaves`, `cAtt`, `cQ`, `cK`, `cV`, `cLn1`, `cH`, `cLn2`, `cM1`, `vitBlockTiedGB`, `vit_block_tiedGB(At)`

**Smell:** repetition
**Current:** Each of the 12 declarations restates the same 17-parameter block binder list:
```lean
noncomputable def cQ {Np1 heads d mlpDim : Nat} (ε : ℝ)
    (γ1 β1 γ2 β2 : Vec (heads * d)) (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (Wfc1 : Mat (heads * d) mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim (heads * d))
    (xin dyOut : Vec (Np1 * (heads * d))) : Vec (Np1 * (heads * d)) :=
  let s := blkSaves ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo Wfc1 bfc1 xin
  vitCotDQmh Np1 heads d s.q s.k s.v (vitCotAttV ε γ2 Wo Wfc1 Wfc2 s.h s.m1 dyOut)
```
**Why it breaks:** Adding or reordering a block parameter means editing 12 signatures and every call
site, and every call site passes 16 positional arguments. `BlockParamsV` already exists for exactly
this ("bundled so depth-`k` signatures stay sane", ViTDepthK.lean:36).
**Suggested:** Take `(p : BlockParamsV (heads * d) mlpDim)` everywhere, and use `variable {Np1 heads d mlpDim : Nat} (ε : ℝ) (p : BlockParamsV ..)`
in a `section`. `blkSaves ε p xin` / `cQ ε p xin dyOut`. This is also the prerequisite for the
capstone finding above. ViTStepTie.lean's `vitBlockTiedMHV` / `vitBlockTiedAtMHV` (lines 43–184) have
the same unbundled shape.

### LeanMlir/Proofs/Nets/ViT/ViTBackB0.lean:275 — `mhsaClean_backward_collapseMH`

**Smell:** heartbeats (4,000,000 = 20×), long-proof (154 lines), undocumented-defeq, repetition
**Current:**
```lean
set_option maxHeartbeats 4000000 in
theorem mhsaClean_backward_collapseMH ... := by
  funext r c
  show (rowwise_has_vjp_mat (dense_has_vjp (mhsa_qkv_W heads d Wq Wk Wv) ...)).backward X
        ((colSlabwise_has_vjp_mat ...).backward ... ) r c = _        -- 13 lines, no comment
  show Mat.mulVec (mhsa_qkv_W heads d Wq Wk Wv) (fun kj => ...) c = _   -- 6 lines, no comment
  ...
  have hdz : (fun kj => ...) = (fun kj => let p := ...; if q.1 = 0 then sdpa_back_Q ... else ...)  -- 20-line if-form (#1)
    ...
    rw [show (mhsa_g_has_vjp_mat N d).backward ... = (let p := ...; if ... sdpa_back_Q ... ) from rfl]  -- if-form (#2)
    rw [hproj0, hproj1, hproj2, hdY0]
  trans (Mat.mulVec (mhsa_qkv_W heads d Wq Wk Wv) (fun kj => let p := ...; if ... ) c)   -- if-form (#3)
```
**Why it breaks:** The two leading `show`s force the kernel to check that the `vjpMat_comp` bundle
`.backward` reduces structurally to a 13-line term, and nothing explains that reliance. Any change to
`vjpMat_comp`, `rowwise_has_vjp_mat` or `colSlabwise_has_vjp_mat` field order breaks them silently.
The same 20-line `if q.1 = 0 then sdpa_back_Q … else if … sdpa_back_K … else sdpa_back_V …` term is
written out three times. `hproj0/1/2` (lines 347–361) are one lemma at `c = 0, 1, 2`.
**Suggested:** (1) Name the if-form once:
`noncomputable def qkvSlabBack (Qg Kg Vg dA : Fin heads → Mat N d) : Mat N (heads * (3 * d)) := fun r kj => …`,
and restate `qkv_back_fanin_MH` over it. Then `hdz`'s RHS, the `from rfl` rewrite and the `trans`
target are each `qkvSlabBack …`. (2) Replace `hproj0/1/2` with one lemma
`mhsa_proj_c_headSlice (c : Fin 3) : mhsa_proj_c c (fun r j => M0 r (finProdFinEquiv (h, j))) = fun r j => dense (![Wq,Wk,Wv] c) (![bq,bk,bv] c) (X r) (finProdFinEquiv (h, j))`,
proved by `fin_cases c <;> …`. (3) Replace the two `show`s with a named unfolding lemma
`mhsaClean_backward_apply` (statement = the second `show`, proof `rfl`), placed right after
`mhsaClean`, or at minimum add a comment. The heartbeat bump most likely comes from `show` and
`trans` elaborating three copies of the if-term, so it should shrink or disappear.

### LeanMlir/Proofs/Nets/ViT/ViTBackB0.lean:575 — `mhsaBackGraphMH_faithful`

**Smell:** heartbeats (2,000,000 = 10×), long-proof (113 lines), repetition, undocumented-defeq
**Current:**
```lean
  show (∑ h : Fin (hm1 + 1), den (SHlo.addV (SHlo.addV (SHlo.denseRowBack "%Wq" Wq ...) ...) ...) j) = _   -- 11 lines
  simp only [den_addV]
  have hQbr : ∀ h, den (SHlo.denseRowBack "%Wq" Wq (SHlo.headPadF h (sdpaBackQGraph ...))) = ... := ...
  have hKbr : ... (same, K) ...
  have hVbr : ... (same, V) ...
  rw [show (∑ h, ((den (...Wq...) j + den (...Wk...) j) + den (...Wv...) j))
        = ∑ h, ((Mat.flatten (...) j + ...) + ...)
      from by apply Finset.sum_congr rfl; intro h _; rw [hQbr h, hKbr h, hVbr h]]   -- 18 lines restating both sides
  unfold mhsaBackCollapsedMH
  unfold Mat.flatten
  rfl
```
**Why it breaks:** The `rw [show … from by …]` restates both sides of an 18-line sum only to push
three rewrites under a binder. Any change to `sdpaBack?Graph`'s argument order must be mirrored in
four places. `hQbr/hKbr/hVbr` differ only in the projection.
**Suggested:** `refine (Finset.sum_congr rfl fun h _ => ?_).trans ?_` and then `rw [hQbr h, hKbr h, hVbr h]` in
the first goal. This drops the restated sum (the mainstream way to rewrite under `∑`).
`simp_rw [hQbr, hKbr, hVbr]` is **not** advised, because `simp_rw` under binders here would unfold `den`.
Merge `hQbr/hKbr/hVbr` via the `![·,·,·]`-indexed helper from the previous finding.

### LeanMlir/Proofs/Nets/ViT/ViTBackB0.lean:227 — `qkv_back_fanin_MH`

**Smell:** heartbeats (1,600,000 = 8×), brittle-chain
**Current:**
```lean
set_option maxHeartbeats 1600000 in
private lemma qkv_back_fanin_MH ... := by
  unfold Mat.mulVec
  rw [sum_heads_3d]
  apply Finset.sum_congr rfl; intro h _
  rw [Fin.sum_univ_three]
  simp only [Equiv.symm_apply_apply, mhsa_qkv_W_eq0, mhsa_qkv_W_eq1, mhsa_qkv_W_eq2,
    show (1 : Fin 3) ≠ (0 : Fin 3) from by decide,
    show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
    show (2 : Fin 3) ≠ (1 : Fin 3) from by decide, ite_true, ite_false]
```
**Why it breaks:** The three inline `show … from by decide` facts do by hand what core's `Fin.reduceEq`
simproc does, and `ite_true`/`ite_false` are the pre-`reduceIte` names. Both are Mathlib-bump churn.
The statement is at variable `heads d`, so 8× heartbeats means `simp only` is fighting the
`let p := finProdFinEquiv.symm kj` binders (zeta plus `Equiv.symm_apply_apply` under `let`).
**Suggested:** `simp only [Equiv.symm_apply_apply, mhsa_qkv_W_eq0, mhsa_qkv_W_eq1, mhsa_qkv_W_eq2, Fin.reduceEq, reduceIte]`.
If the heartbeats persist, state `qkv_back_fanin_MH` over the named `qkvSlabBack` from the finding
above (no `let` in the statement) and try without the bump.

### LeanMlir/Proofs/Nets/ViT/ViTBackB0.lean:215 `sum_heads_3d`, :492 `mulVec_headPadMat`; ViTMhsaBackCertifiedTie.lean:92–103 `mhsaBackFlat_eq_mhsa_vjp`

**Smell:** repetition / brittle-chain
**Current:**
```lean
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin heads × Fin (3*d) ≃ Fin (heads * (3*d))) f]
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro h _
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 3 × Fin d ≃ Fin (3*d)) (fun kk => f (finProdFinEquiv (h, kk)))]
  rw [Fintype.sum_prod_type]
```
In ViTMhsaBackCertifiedTie.lean:92–103 this appears three times, each with a 3-line explicit motive.
**Why it breaks:** The repo already has exactly this reindex as `sum_finProdFinEquiv`
(Foundation/Tensor.lean:428, "every flatten/unflatten sum in the suite reduces to this split"). The
hand-rolled copies pin `Equiv.sum_comp` and `Fintype.sum_prod_type` argument order and restate the
summand as a motive. There are 6 sites in scope.
**Suggested:** `sum_heads_3d := by rw [sum_finProdFinEquiv]; exact Finset.sum_congr rfl fun h _ => sum_finProdFinEquiv _`.
In `mulVec_headPadMat` and `mhsaBackFlat_eq_mhsa_vjp`, use `rw [sum_finProdFinEquiv]` (no motive
needed). Nothing new is added to Tensor.lean, so there is no rebuild cost.

### LeanMlir/Proofs/Nets/ViT/ViTVecLNBackCertifiedTie.lean:97–99 and :127–129 — `attnSubFlatTieV`, `mlpSubFlatTieV`

**Smell:** undocumented-defeq, repetition
**Current:**
```lean
  have hw : Mat.unflatten w (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = w idx := by
    show w (finProdFinEquiv ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2)) = w idx
    rw [Prod.mk.eta, Equiv.apply_symm_apply]
```
**Why it breaks:** The `show` depends on how `Mat.unflatten` unfolds, and the identical block appears
twice (with `hv`). The lemma already exists: `Mat.flatten_unflatten` (Tensor.lean:450).
**Suggested:** `have hw := congrFun (Mat.flatten_unflatten w) idx`, or rewrite with it directly.
Both sites then lose the `show`.

### LeanMlir/Proofs/Nets/ViT/ViTFoldGB.lean:284–329 (`RowDenseWTiedB` …) vs :78–150 (`*_den`); same in ViTFold.lean:145–181 vs :31–140

**Smell:** repetition
**Current:** Each `*TiedB` Prop is its `_den` lemma's statement copied verbatim, for example:
```lean
theorem rowDenseWeightGradB_den ... : den (SHlo.rowDenseWeightGradB ...) (finProdFinEquiv (i, j)) = ∑ n, ∑ o, pdiv (...) ... := ...
def RowDenseWTiedB ... : Prop := ∀ (i : Fin a) (j : Fin c), den (SHlo.rowDenseWeightGradB ...) (finProdFinEquiv (i, j)) = ∑ n, ∑ o, pdiv (...) ...
```
Then every block tie proves each clause with `intro i j; exact …_den … i j` (16× in `vit_block_tiedGB`).
**Why it breaks:** Both copies must be kept in sync by hand, 4 pairs in ViTFoldGB and 4 in ViTFold.
**Suggested:** Define the Prop first and state `theorem rowDenseWeightGradB_den … : RowDenseWTiedB N tk xN cotN bb x W dy`.
The `intro i j; exact …` lines in `vit_block_tiedGB` then collapse to the lemma term.

### LeanMlir/Proofs/Nets/ViT/ViTStepTie.lean:195 `vit_cls_den` and ViTFoldGB.lean:222 `clsGrad_denB`

**Smell:** fragile-simpa, work at numeral shapes
**Current:**
```lean
set_option linter.unusedSimpArgs false in
theorem vit_cls_den (clsN lrStr cotN : String) (Wc : Kernel4 192 3 16 16) ... :=
  ... simp only [den, batchSlice, clsSliceFlat, cls_token_grad]; rw [Fin.sum_univ_one]; rfl
```
```lean
    simp [batchSlice, batchMap, clsSliceFlat, Equiv.symm_apply_apply]      -- unrestricted, at D = 192
```
**Why it breaks:** Silencing `unusedSimpArgs` hides which of the four unfoldings still fire. A
renamed or removed definition will not show up as an unused-arg warning, so the `rfl` that follows
fails with an opaque message. The unrestricted `simp` at literal 192/197 can pick up any new
`@[simp]` numeral lemma. The docstring (ViTFoldGB.lean:215–218) says the numerals are forced because
`Vec (N * (1 * D))` only reduces at a literal `D`.
**Suggested:** Prove both at variable `D` with an explicit cast: `(Nat.one_mul D) ▸` / `Fin.cast (by simp)`
on the CLS operand, or restate `clsSliceFlat`'s result type as `Vec D` via a `Nat.one_mul` rewrite
lemma. Then instantiate. At minimum, drop the linter switch and prune the `simp only` list to the
args that fire.

---

## EfficientNet

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetFullWholeBackCertifiedTie.lean:109 and :186 — `efficientnetInputGradB_full_eq_efficientnetB_full_vjp`, `…_eq_efficientnetForwardB_full_vjp`

**Smell:** heartbeats (4,000,000 + `maxRecDepth 800000`, the largest recDepth in scope), work at numeral shapes
**Current:**
```lean
set_option maxRecDepth 800000 in
set_option maxHeartbeats 4000000 in
theorem efficientnetInputGradB_full_eq_efficientnetB_full_vjp (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) ... (Wh : Kernel4 1280 320 1 1) ...
    (b1 : Vec (N * (32 * 112 * 112)) → Vec (N * (16 * 112 * 112))) ... :
    efficientnetInputGradB_full N Ws Wh Wfc (...) ... (hb16.backward (opaqueA15 (stemB N (h := 112) (w := 112) ...) b1 ... b15 x))
      = (efficientnetB_full_has_vjp ...).backward x := by
  unfold efficientnetInputGradB_full
  rw [stemBBack_eq_vjp_backward (N := N) (h := 112) (w := 112) (by decide) (by decide) ..., headFwdBBack_eq_vjp_backward ...]
  rfl
```
**Why it breaks:** The closing `rfl` makes the kernel unfold the 17-deep `vjp_comp` chain of
`efficientnetB_full_has_vjp` against `efficientnetInputGradB_full`. It does so at `Vec (N * (32*112*112))`,
`Vec (N * (1280*7*7))` and so on. Most of the 800k recursion depth is `Nat` literal arithmetic in
those types (`32*112*112 = 401408`) during defeq checks. `efficientnetB_full_has_vjp` (line 63) is
already width-generic (`s0 … s18`). Only `efficientnetInputGradB_full`
(EfficientNetBackChains.lean:27) fixes the literals.
**Suggested:** Generalise `efficientnetInputGradB_full` to width binders
`{c0 … c17 h0 … : Nat}`, or add a width-generic twin `b0InputGradChain`. Prove the tie there with the
same `unfold; rw; rfl` and instantiate at B0's widths. That follows the established "state at
variable shapes, instantiate" rule. The second theorem (line 186) is then an instantiation plus
`backward_unique` and should need neither option. In the meantime, replace `(by decide)` for
`0 < 112` with `(by norm_num)` or `Nat.succ_pos _`. It is cheap either way, but `decide` on `Nat.lt`
literals goes through `Nat.decLt` in the kernel.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetFullB0.lean:372 — `efficientnetForwardB_full_has_vjp`

**Smell:** repetition, known kernel trap (`vjp_comp _ _` elaborating to a concrete composed chain), recDepth bump (20000)
**Current:**
```lean
set_option maxRecDepth 20000 in
noncomputable def efficientnetForwardB_full_has_vjp (N : Nat) (w : B0Weights) (hsε ...) ... :
    HasVJP (headFwdB N ... ∘ mbExpW N 7 7 w.b16 ∘ ... ∘ stemB N ...) := by
  have dS := ...; have vS := ...; ... (36 haves)
  have e1 := vjp_comp _ _ dS d1 vS v1;            have f1 := d1.comp dS
  ...
  exact vjp_comp _ _ f16 dH e16 vH
```
**Why it breaks:** This is the same 18-stage apex as `efficientnetB_full_has_vjp`
(FullWholeBackCertifiedTie.lean:63), rewritten at the concrete blocks. Each `_` elaborates to the
concrete composed prefix at numeral widths, which is the known kernel trap. Because the two witnesses
are syntactically different, FullWholeBackCertifiedTie.lean:245 needs
`.trans (funext fun dy => HasVJP.backward_unique _ _ x dy)` to bridge them.
**Suggested:** Move `efficientnetB_full_has_vjp` into EfficientNetFullB0.lean. The only extra dependency
it needs is `vjp_comp`. Define
`efficientnetForwardB_full_has_vjp … := efficientnetB_full_has_vjp (stemB …) (mbNoExpW N 112 112 w.b1) … dS d1 … vS v1 … vH`,
which is one term with no `_`. The bridge in FullWholeBackCertifiedTie then becomes `rfl`, or is
unnecessary, and the `maxRecDepth` goes.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetSyncStepTieG.lean:1364 — `efficientnet_net_syncTiedG`; EfficientNetStepTieG.lean:481 `efficientnet_net_tiedG`; EfficientNetStepTie.lean:624 `efficientnet_net_tied`

**Smell:** heartbeats (4,000,000 + `maxRecDepth 100000`, 3 sites), long-proof (174 / 102 / 100 lines), numeral shapes
**Current (SyncStepTieG):**
```lean
set_option maxHeartbeats 4000000 in
set_option maxRecDepth 100000 in
theorem efficientnet_net_syncTiedG ... :
    let a0 : Vec ((R * N) * (32 * 112 * 112)) := stemB (R * N) (h := 112) (w := 112) ...
    ... 55 lets ...
    stemSyncTiedG ... ∧ noExpSyncTiedG ... ∧ ... (18 conjuncts) := by
  intro a0 ... e0
  have h112 : 0 < 112 := by norm_num
  ...
  have s16 := hdsCotIn_scaled ...
  ... have s0 := ...
  exact ⟨stem_syncTiedG ..., ..., head_syncTiedG ...⟩
```
**Why it breaks:** The single-device capstones have the same structure as ViT's: every conjunct is a
∀-instance, and StepTieG.lean:436 even notes the `@[irreducible]` `*At` wrappers were introduced
"to keep the capstone opaque". The budget is still 20×, so the cost is not in the conjuncts. It is
elaborating 37–55 `let`s whose types carry literal widths and zeta-unifying them against each lemma
application. The sync capstone does carry content, because `s16 … s0` thread the scaled-shard
invariant, but that thread is a list fold over 16 blocks.
**Suggested:** First, try removing both options on each site. The `*At` wrappers already hide the
block bodies, so the bumps may be stale. For a durable fix, bundle the per-stage widths as a
`B0Stage` list, or at least give `stemSyncTiedG` / `expSyncTiedG` … `@[irreducible]` `*At` twins
like the single-device file. Also state the `let` chain via `opaqueA0 … opaqueA16`
(Foundation/OpaquePrefix.lean), which already exist for this purpose, so the statement never exposes
`mbResidW N 14 14 …` nested 16 deep. Combining this with the ε-bundle finding below removes 49 binders
from each of the three theorems.

### EfficientNet capstones — the 49 `0 < ε` hypotheses (7 statements)

**Smell:** repetition (redundant hypothesis lists)
**Current:** This 17-line binder block is repeated in StepTie.lean:631, StepTieG.lean:489,
SyncStepTieG.lean:1384, FullB0.lean (2×, including :377) and FullWholeBackCertifiedTie.lean:193, :253:
```lean
    (hsε : 0 < w.sε)
    (hb1d : 0 < w.b1.dε) (hb1p : 0 < w.b1.pε)
    (hb2e : 0 < w.b2.eε) (hb2d : 0 < w.b2.dε) (hb2p : 0 < w.b2.pε)
    ... (hb16p : 0 < w.b16.pε) (hhε : 0 < w.hε)
```
It is also re-passed positionally (49 args) at FullWholeBackCertifiedTie.lean:240, 303, 304.
**Why it breaks:** Adding a BN to B0 means editing 7 statements and every call. The positional
49-arg calls are unreadable, and transposing two `hbKd`/`hbKp` args is caught only by type mismatch.
**Suggested:** Add `structure B0Weights.EpsPos (w : B0Weights) : Prop where (s : 0 < w.sε) (b1d : 0 < w.b1.dε) …`
next to `B0Weights`. Take `(hε : w.EpsPos)` and use `hε.b16e` inside the proofs. Mainstream Mathlib
bundles side conditions this way, as with `IsProbabilityMeasure`-style structures.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetSyncStepTieG.lean:341–388 and :479–489 — `cbsB_back_eq`, `dwbsB_back_eq`, `dwbsSB_back_eq`, `projB_back_eq`, `hdCotIn_eq_vjp`

**Smell:** repetition, undocumented-defeq (5 `show`s)
**Current:** the same 6-line proof, 5 times:
```lean
  have hg := cbsBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
  rw [den_operand] at hg
  rw [← hg]
  show cInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
  rfl
```
**Why it breaks:** The `show` depends on `den (cbsBackBatchedGraph …)` reducing by `den`'s equations
to `cInB N W b (den (bnBatchLABack …))`. Any change in how the graph is built in EfficientNetBackB0.lean
(adding a reshape node, reordering) breaks all five with "type mismatch" and no pointer to the cause.
**Suggested:** In EfficientNetBackB0.lean, where each graph is defined, add a structural lemma per
graph, for example `den_cbsBackBatchedGraph : den (cbsBackBatchedGraph W b ε γ x e) = cInB N W b (den (SHlo.bnBatchLABack …))`,
proved by `rfl` next to the def. Then each `*_back_eq` is
`by rw [← …_faithful _ _ _ _ _ _ x (.operand "" dy), den_operand, den_cbsBackBatchedGraph, den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]`.
No `show` is needed, and the defeq lives beside the definition it depends on.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetSyncStepTieG.lean:579–692, 754–1111 — 25 `*_smul` + 30 `*_shard` chain lemmas

**Smell:** repetition (55 lemmas in this file; ~160 of the same shape across 5 nets)
**Current:**
```lean
theorem tCotZ_smul (N h w : Nat) {mid oc rd : Nat} (t : EnTail mid oc rd) (hp : 0 < t.pε)
    (dc : Vec (N * (mid * h * w))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    tCotZ N h w t hp dc (fun i => s * dy i) = fun i => s * tCotZ N h w t hp dc dy i := by
  unfold tCotZ; rw [tCotE2_smul, rowDenseBackFlat_smul]

theorem tsCotZ_shard (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc rd : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (t : EnTail mid oc rd) (hp : 0 < t.pε)
    (DC : ...) (dys : ...) (DY : ...) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    tsCotZ R hR N h w t DC dys r = batchShard R N rd (tCotZ (R * N) h w t hp DC DY) r := by
  unfold tsCotZ
  rw [tsCotE2_shard R hR N h w hN hh hw t hp DC dys DY hdys r, rowDenseBackFlat_shard]
  rfl
```
Per-file counts of `theorem …_smul` / `…_shard`: ResNet34SyncStepTieB 33/26, MobileNetV2SyncStepTieB 26/26,
MobileNetV4SyncStepTieB 27/27, ResNet50SyncStepTieB 23/23, EfficientNetSyncStepTieG 25/29.
**Why it breaks:** Each lemma restates 12–16 binders to say "compose the previous link's property
with this link's". Every new link in a block means two more lemmas with full signatures. The sibling
ResNet34/MobileNet files already use `variable` sections (their `…_shard (r : Fin R) :` signatures);
this file does not, so it is the most verbose of the five.
**Suggested:** (a) Cheap now: open a `section` with
`variable (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc rd : Nat} [NeZero N] [NeZero h] [NeZero w] (t : EnTail mid oc rd) (hp : 0 < t.pε) (DC …) (dys …) (DY …) (hdys …)`
as the sibling files do, which cuts roughly 300 lines. (b) Structural, for all five nets: make
homogeneity compositional. Mathlib has no `IsLinearMap.comp`, so either bundle each link's backward as
`Vec m →ₗ[ℝ] Vec n` (then every `*_smul` is `map_smul` and a chain is `LinearMap.comp`), or add in
Foundation/DataParallelSync.lean, a leaf with 2 direct importers beside `HasVJP.backward_smul`:
```lean
def IsHomog {m n} (f : Vec m → Vec n) : Prop := ∀ s dy, f (fun i => s * dy i) = fun i => s * f dy i
theorem IsHomog.comp {f g} (hf : IsHomog f) (hg : IsHomog g) : IsHomog (f ∘ g)
theorem HasVJP.isHomog (hf : HasVJP f) x : IsHomog (hf.backward x)
```
Similarly, `IsShardwise F f := ∀ X DY r, F (shard X r) (shard DY r) = shard (f X DY) r` with `.comp`.
Each chain node's property is then a one-term `.comp` expression with no restated signature.

### EfficientNetSyncStepTieG.lean (≈25 lemmas) and EfficientNetSyncB.lean:204–316 — `(hN : 0 < N) (hh : 0 < h) (hw : 0 < w)` threaded only for `nhw_ne_zero`

**Smell:** repetition, redundant hypotheses
**Current:**
```lean
  bnSyncInB_shard_bnBackB R hR N mid (2 * h) (2 * w)
    (nhw_ne_zero hN (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    (nhw_ne_zero (Nat.mul_pos hR hN) (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw)) ...
...
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num   -- ×5, in both whole-net theorems
```
**Why it breaks:** Each of the three positivity hypotheses exists only to feed `nhw_ne_zero`
(ResNet34SyncB.lean:143). They propagate through every shard lemma signature and force `norm_num`
calls at literals.
**Suggested:** Use `[NeZero N] [NeZero h] [NeZero w]` (and `[NeZero R]` in place of `hR`). Mathlib's
`NeZero.mul` instance (Algebra/GroupWithZero/Basic.lean:92) then provides `NeZero (N * (h * w))`,
`NeZero (2 * h)` and `NeZero ((R*N) * …)` by instance search. `nhw_ne_zero` becomes `NeZero.ne _`,
and the literal cases resolve by the `NeZero (n+1)` instance, so the `have h112 …` lines go.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetStepTieG.lean:57–341 (and EfficientNetStepTie.lean:180–470) — `enetExpTiedG`, `enetStridedTiedG`, `enetNoExpTiedG`

**Smell:** repetition (the MBConv tail chain is written 7 times)
**Current:** Each def repeats the same 12-`let` forward tail (`dn, dr, s, e1, z, e2, se, pc`), the
10-`let` backward tail (`cotPbn … cotDc`) and the 9 tail conjuncts (depthwise BN pair, SE dense ×4,
project conv + BN pair). The only differences are the front stage and whether `h` is `2*h`. The
matching proofs repeat 10 identical `· intro …; exact EnetPoCG.…_den …` lines. StepTie.lean has the
same 3 copies for the fused form. SyncStepTieG.lean:83–179 already factored the tail as
`structure EnTail` + `tDn … tCotDc`, and its docstring (lines 19–23, 106–107) says those defs
"unfold to `enetExpTiedG`'s `let`s … by `rfl`". That is a defeq bridge between two copies of the same
chain.
**Suggested:** Move `EnTail`, `tailOf`, `tailOfNoExp` and the `t*`/`x*`/`s*`/`n*` chain defs out of
SyncStepTieG.lean into a new leaf `Nets/EfficientNet/EfficientNetChainDefs.lean`, imported by StepTie,
StepTieG and SyncStepTieG. Define `enetTailTiedG … (t : EnTail mid oc rd) hp dc dy : Prop` once, with
the 9 conjuncts over `tCot*`, and prove it once. Then
`enetExpTiedG := ConvWTiedB … ∧ BnBetaTiedB … ∧ BnPairTiedB … ∧ DepthwiseWTiedB … ∧ enetTailTiedG …`.
The Sync file's `rfl` bridge disappears because both files name the same defs. The StepTieG docstring
(lines 25–31: "the fusion is `rfl` … each conjunct's proof is the fused file's with the wrapper
peeling dropped") suggests the fused StepTie ties could also be derived from the G ties by one
`*SgdB_eq_grad` rewrite per node kind, rather than re-threading the whole chain.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetStepTieG.lean:93–97 (10 sites in file, 1 in EfficientNetFoldG.lean)

**Smell:** repetition
**Current:** the BN-β clause is written inline, not named, while every other parameter kind has a Prop
(`ConvWTiedB`, `BnPairTiedB`, `DenseWTiedB`, …):
```lean
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w) (.operand cotN (reassocB N mid h w cotEc))) o
          = ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εe (fun _ => 0) β' (fun _ => 0))
                   be o j * bnchwFwd N mid h w (reassocB N mid h w cotEc) j)
```
**Why it breaks:** Five lines × 10 copies, each with explicit implicit-arg annotations
`(N := N) (oc := …) (h := …) (w := …)` that must track any signature change of `bnBetaGradB`.
**Suggested:** In EfficientNetFoldG.lean (a leaf), add
`def BnBetaTiedB (N oc h w : Nat) (cotN : String) (ε : ℝ) (b : Vec oc) (cot : Vec (N*(oc*h*w))) : Prop := ∀ o, …`
and restate `bnBetaGradB_den : BnBetaTiedB …` (see the ViTFoldGB Tied/_den finding: the same pattern
fixes both).

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetBackB0.lean:300–398 — `convBackBatched_faithful`, `convStridedBackBatched_faithful`, `depthwiseStridedBackBatched_faithful`, `depthwiseStridedXlaBackBatched_faithful`, `depthwiseBackBatched_faithful`

**Smell:** fragile-simpa (unfold-then-`rfl`), repetition
**Current:** 5 copies of
```lean
  funext idx
  simp only [den, batchMap, batchMap_has_vjp, flatConv_has_vjp, hasVJPMat_to_hasVJP,
    rowwise_has_vjp_mat, hasVJP3_to_hasVJP, conv2d_has_vjp3]
  rfl
```
**Why it breaks:** The proof unfolds five structure-building definitions by name and closes by `rfl`.
Renaming any of `hasVJPMat_to_hasVJP`, `rowwise_has_vjp_mat` or `hasVJP3_to_hasVJP`, or changing how
`batchMap_has_vjp` transports, breaks all five. The unfold lists also already differ between copies
(some list `conv2d_has_vjp3`, some do not), which shows they were tuned by trial. These are per-op,
not whole-net, so `simp only` is not the 48 GB hazard here. The fragility is the terseness.
**Suggested:** In EfficientNetChainClose.lean (where `batchMap_has_vjp` is defined), add the apply
lemma once:
`theorem batchMap_has_vjp_backward (f hf df v dy) : (batchMap_has_vjp f hf df).backward v dy = fun idx => hf.backward (batchSlice … v n) (batchSlice … dy n) k` (with `(n,k) := finProdFinEquiv.symm idx`), proved once.
Each `*_faithful` then becomes `funext idx; rw [batchMap_has_vjp_backward]; rfl`, or `simp only [den, batchMap_has_vjp_backward]`.

---

## Recurring patterns (worth more than any single finding)

1. **Whole-net capstones stated at the net's literal widths, with 4M–16M heartbeat and 100k–800k recDepth
   bumps (8 sites):** ViTStepTieGB:375 (16M/400k), ViTStepTie:283 (16M/400k),
   FullWholeBackCertifiedTie:109 and :186 (4M/800k), SyncStepTieG:1364, StepTieG:481, StepTie:624
   (4M/100k each), FullB0:372 (recDepth 20k). In most of them every conjunct is an instance of a
   ∀-lemma, so the budget pays for elaborating literal-width `let` chains, not for proof. Fix by
   stating at variable widths and depth, using existing assets (`BlockParamsV`/`vitBodyKV`,
   `efficientnetB_full_has_vjp`, `opaqueA*`), and instantiate.
2. **Unbundled parameters and hypotheses:** 49 `0 < ε` binders × 7 EfficientNet statements (plus 49-arg
   positional calls); 192 ViT block binders × 2 capstones while `BlockParamsV` sits unused; 17-binder
   block signatures × 12 ViT defs; `hN hh hw` threaded through ≈25 lemmas only for `nhw_ne_zero`
   (fix: `[NeZero _]` + `NeZero.mul`); EfficientNetSyncStepTieG restating 12–16 binders per lemma where
   sibling nets use `variable`. The mainstream fixes are `structure … : Prop` bundles, parameter
   structures and `variable` sections.
3. **Copy-pasted per-node chains:** 55 `unfold X; rw [prev, link]; rfl` `_smul`/`_shard` lemmas in
   EfficientNetSyncStepTieG (~160 across the five sync nets); the MBConv tail chain written 7 times
   across StepTie/StepTieG/SyncStepTieG; `Tied` Props restating their `_den` lemma verbatim (ViTFold,
   ViTFoldGB, EfficientNetFold, plus 10 inline BN-β copies); 5× graph-back `show`+`rfl` in
   SyncStepTieG; 5× `simp only [defs]; rfl` in BackB0; 6× hand-rolled `finProdFinEquiv` reindex that
   `sum_finProdFinEquiv` already provides. Fix with one generic combinator per pattern (`IsHomog.comp` /
   `IsShardwise.comp` or `→ₗ[ℝ]` bundling in Foundation/DataParallelSync.lean; `EnTail` moved to a leaf;
   Prop-first `_den` statements).
