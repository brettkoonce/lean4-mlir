# Re-audit: sync/DP files + the longest hand-written proofs outside Training/

Static read only. Nothing was edited, built or profiled. Line numbers are at `proof-cleanup` HEAD
(`28fc373f`). Every "−N lines" figure below is counted from the source; none is measured.

Scope: `Foundation/DataParallelSync{,Kit,Bf16}.lean`, the five `*SyncStepTie*` nets, and the 20 long
declarations the caller named. Files with no findings are listed at the end.

**Ranked by payoff/effort**

| # | effort | finding | where | payoff |
|---|---|---|---|---|
| 1 | trivial | `_correct` proofs re-spell the whole apex witness | R34/R50/MNv2/MNv4 WholeBack `_correct` | −~45 lines, 4 witness copies gone, statements unchanged |
| 2 | trivial | `bnSync_of_scaled` takes a redundant `hM` | Kit:395, 71 call sites | −~80 call-site tokens/lines, 5 files |
| 3 | trivial | MNv4 skip∘body scaled composition written 19× | MNv4SyncStepTieB:1307–1373 | −~38 lines |
| 4 | small | apex writes each block's argument list twice (scaled + tie) | all 5 sync apexes | −~100 lines; one arg list per block |
| 5 | small | R50/MNv2/MNv4 whole-back ties close with `rfl` through the concrete apex (the §0 trap shape) | R50WB:150, MNv2WB:317, MNv4WB:389 | robustness: same shape as R34's §1(f) peel |
| 6 | small | `PProd (HasVJPAt f x) (DifferentiableAt ℝ f x)` spelled 257× | 4 WholeBack files + apex defs | about half of each binder line; ⚠ tier regen |
| 7 | medium | R34 + EffNet sync apexes are not on the free-`G`/`hgs` + named-Prop shape the other three use | R34Sync:560, EffSync:1391 | one shape for five nets; ⚠ tier regen |
| 8 | medium | `mhsaClean_backward_collapseMH`: the if-form is spelled 3×, 3 copied `hproj` blocks, 2 undocumented `show`s | ViTBackB0:276 | 144 → ~40 lines |
| 9 | small | `mhsaBackGraphMH_faithful`: 12-line `show` + 20-line `rw [show … from by sum_congr]` | ViTBackB0:570 | −~30 lines |
| 10 | medium | ConvNeXt/ViT ties take ~190 loose binders; the weight structures already exist, unused | ConvNeXtStepTie:461, …GB:381, ViTStepTie:292, …GB:390 | statements shrink by ~150 lines each; ⚠ ε-sharing, tier |
| 11 | trivial | undocumented `show` / `rw [show … from rfl]` | DPSync:316, EffSync:354/366/380/391/492, Kit:412 | small |
| 12 | trivial | Cifar8 tie: 24 mechanical conjunct instantiations | Cifar8BnStepTie:52 | low |

---

### LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:444 — `r34InputGradB_correct` (also R50WholeBack:156, MNv2WholeBack:323, MNv4WholeBack:395)

**Smell:** repetition
**Current:** each `_correct` restates the `_eq` lemma's full apex witness in its proof (R50WholeBackCertifiedTieB.lean:240–250):
```lean
  rw [congrFun (r50InputGradB_eq_r34B_full_vjp N q hq0 Ws bs εs hεs γs βs Wd bd
    b1 b2 … b16 x h_stem h_pool hb1 … hb16) dy]
  exact (r34BFullHasVJPAt (r34StemB N (2 * (2 * (2 * q))) …) b1 … b16
          (r34HeadB N q q Wd bd) x
          ⟨r34StemBHasVJPAt N … (by norm_num) (by omega) (by omega) x h_stem h_pool,
            r34StemB_differentiableAt N … x h_stem h_pool⟩
          hb1 … hb16
          ⟨(r34HeadBHasVJP N q q Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N q q Wd bd) _⟩).correct dy i
```
MNv4WholeBackCertifiedTieB.lean:515–531 does the same with a 16-line witness. So each witness is written twice per net: once in the `_eq` statement and again in the `_correct` proof.
**Why it breaks:** the two copies must agree term for term. If a witness argument changes (for example the `(by norm_num)` side goals, or R50's `(by omega)`s when `q` changes shape), every `_correct` has to be edited by hand, and the `exact` fails with a large unification error that does not point at the argument that changed.
**Suggested:** add one lemma and let unification find the witness from the `_eq` lemma's type. Put it in `Foundation/OpaquePrefix.lean`, which all four files import, and not in `Tensor.lean`: §0 says a root-file lemma rebuilds ~420 modules.
```lean
theorem HasVJPAt.correct_of_backward_eq {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (hf : HasVJPAt f x) {B : Vec n → Vec m} (h : B = hf.backward) (dy : Vec n) (i : Fin m) :
    B dy i = ∑ j : Fin n, pdiv f x i j * dy j := by
  rw [h]; exact hf.correct dy i
```
Each proof then becomes `HasVJPAt.correct_of_backward_eq _ (r50InputGradB_eq_r34B_full_vjp N q hq0 … hb16) dy i`, and no statement changes. EfficientNet already avoids the copy (EfficientNetFullWholeBackCertifiedTie.lean:258–260 uses a named `_correct`), so it is the model.

### LeanMlir/Proofs/Foundation/DataParallelSyncKit.lean:395 — `bnSync_of_scaled`

**Smell:** repetition (redundant hypothesis)
**Current:**
```lean
theorem bnSync_of_scaled (R : Nat) (hR : 0 < R) (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (hM : (R * N) * (h * w) ≠ 0) …
```
Every caller computes both side conditions from the same three facts:
```lean
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw          -- ResNet34SyncStepTieB.lean:425–426
```
There are 71 `nhw_ne_zero (Nat.mul_pos hR hN) …` occurrences across the five sync ties (R34 9, R50 15, MNv2 16, MNv4 20, EffNet 11), plus inline pairs such as ResNet50SyncStepTieB.lean:763–764.
**Why it breaks:** `hM` follows from `hm` and `hR`, so it is redundant. It also puts the `N * (h * w)` bracketing into 71 call sites. If `BnSync` is ever re-stated at `(R*N)*h*w`, every one of those sites has to change.
**Suggested:** add a sibling lemma and keep the old one for the three stems that already have `hm`:
```lean
theorem bnSync_of_scaled_pos (R : Nat) (hR : 0 < R) (N oc h w : Nat) (hN : 0 < N) (hh : 0 < h)
    (hw : 0 < w) (tg tb vN epsStr cotN : String) (ε : ℝ) (V …) (cots …) (COT …) (hc …) :
    BnSync R hR N oc h w tg tb vN epsStr cotN ε V cots COT :=
  bnSync_of_scaled R hR N oc h w (nhw_ne_zero hN hh hw)
    (nhw_ne_zero (Nat.mul_pos hR hN) hh hw) tg tb vN epsStr cotN ε V cots COT hc
```
With it, the `have hm/hM` preambles go from every `*_syncTiedB` block lemma.

### LeanMlir/Proofs/Nets/MobileNet/MobileNetV4SyncStepTieB.lean:1294 — `mnv4_net_syncTiedB`

**Smell:** repetition
**Current:** the same two-lemma nesting appears 19 times (lines 1307–1373):
```lean
  have s20 := mnv4SkipSyncCotIn_scaled _ e21 _ dy21
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row21 (by decide) w.b21 (mnv4Blk20 (R * N) w X)
      e21 dy21 s21) s21
```
**Why it breaks:** this is not fragile, just repeated. The `(by decide)` for `0 < mnv4RowK.h` is repeated 42 times across the scaled chain and the tuple.
**Suggested:** add a lemma next to `mnv4SkipSyncCotIn_scaled` (:797):
```lean
theorem mnv4SkipBodySyncCotIn_scaled (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * s.h * s.h)))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N _ (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    mnv4SkipCotIn (mnv4BodySyncCotIn R hR N s p XIN dys r) (dys r)
      = batchShard R N _ (fun i => (R : ℝ) *
          mnv4SkipCotIn (mnv4BodyCotIn (R * N) s p XIN DY) DY i) r :=
  mnv4SkipSyncCotIn_scaled _ dys _ DY (mnv4BodySyncCotIn_scaled R hR N hN s hh p XIN dys DY hdys) hdys r
```
Each of the 19 `have`s drops to one line. This composes with #4.

### Cross-net: the five sync apexes — `r34_net_syncTiedB` (ResNet34SyncStepTieB.lean:560), `r50_net_syncTiedB` (ResNet50SyncStepTieB.lean:892), `mnv2_net_syncTiedB` (MobileNetV2SyncStepTieB.lean:1002), `mnv4_net_syncTiedB` (MobileNetV4SyncStepTieB.lean:1294), `efficientnet_net_syncTiedG` (EfficientNetSyncStepTieG.lean:1391)

**Smell:** long-proof / repetition
**Current:** every apex is two passes over the same blocks with the same arguments. From R34 (:637–673):
```lean
  have sE0 := r34IdSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.e1 (r34Pre15 (R * N) w X) eE1 dyE1 sE1
  …
  exact ⟨…,
    r34_idblock_syncTiedB R hR N 7 7 hN h7 h7 "s4b1" xN cotN vN epsStr w.e1 (r34Pre15 (R * N) w X) eE1 dyE1 sE1,
```
`R hR N h w hN hh hw p XIN dys DY hdys` is typed once for `_scaled` and again for `_syncTiedB`. Across the five apexes that is ~92 blocks, each written twice. The proofs do carry structural comments ("the divisor", "the scaled-shard invariant, block by block"), and their length is mostly the statement's `let` chains, so this is repetition, not an uncommented long proof.
**Why it breaks:** a new block argument (for example a `hq`-style positivity fact, as R50 needed) has to be added in both places and kept in the same order. The two lists are ~40 lines apart, and nothing checks that they agree except the final `exact`.
**Suggested:** give each block kind one "step" lemma that returns both results, stated beside the existing pair:
```lean
theorem r34_idblock_syncStepB (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : R34IdW c)
    (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
    (DY : Vec ((R * N) * (c * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (c * h * w) (fun i => (R : ℝ) * DY i) r) :
    r34IdSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY
    ∧ ∀ r, r34IdSyncCotIn R hR N h w p XIN dys r
        = batchShard R N (c * h * w) (fun i => (R : ℝ) * r34IdCotIn (R * N) h w p XIN DY i) r :=
  ⟨r34_idblock_syncTiedB … hdys, r34IdSyncCotIn_scaled … hdys⟩
```
Each apex body becomes `obtain ⟨tE1, sE0⟩ := r34_idblock_syncStepB … sE1` per block, then `exact ⟨tStem, tA0, …⟩`. The same applies to the R50 id/down/proj, MNv2 stride1/stride2/noexp/body, MNv4 body/sbody/fused and EffNet rs/ss/xs/ns kinds. This keeps the §0 rule: nothing is unfolded, and every lemma is still proved at variable shapes.

### LeanMlir/Proofs/Nets/ResNet/ResNet50WholeBackCertifiedTieB.lean:60 — `r50InputGradB_eq_r34B_full_vjp` (also MobileNetV2WholeBackCertifiedTieB.lean:211, MobileNetV4WholeBackCertifiedTieB.lean:251)

**Smell:** undocumented-defeq (the §0 trap shape)
**Current:**
```lean
  unfold r50InputGradB
  rw [cbReluStridedBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      r34HeadBBack_eq_vjp_backward Wd bd (opaqueA16 …)]
  rfl                                                            -- :150
```
MNv2 (:313–317) and MNv4 (:378–389) have the same shape. Their closing `rfl` goes through a 20-/25-deep `vjpCompDiffAt` chain at the literal widths (112, 7).

R34's twin (ResNet34BackCertifiedTieB.lean:436–441) was fixed in §1(f) to peel through a lemma proved at variable stages:
```lean
  funext dy
  rw [r34BFullHasVJPAt_backward, r34StemBHasVJPAt_backward]
  repeat rw [Function.comp_apply]
  rfl
```
**Why it breaks:** this is the "closing `rfl` through the concrete chain" that §0 forbids. The ties compile today only because the blocks are opaque. R34's identical `rfl` cost 43 s / 8.2 GB and hit `maxRecDepth` at its stem. If a stem VJP becomes `vjpCompAt`-built, as R34's is, or the apex gains a stage, the kernel has to re-derive the chain at the net's numerals.
**Suggested:** R50 uses R34's apex, so `r34BFullHasVJPAt_backward` and `r34StemBHasVJPAt_backward` apply unchanged: copy R34's four closing lines into R50 (trivial). For MNv2/MNv4, add `mobilenetv2PaperPCHasVJPAt_backward` / `mnv4BFullHasVJPAt_backward`, each `rfl` at variable `{s0 … s26}` exactly like ResNet34BackCertifiedTieB.lean:259, and `rw` with them. Size the effort after #6, which makes those two statements short.

**On the name.** `r50InputGradB_eq_r34B_full_vjp` is not a copy-paste name. Its right-hand side really is `r34BFullHasVJPAt` (:138), which is the generic 18-stage apex; the module doc (:20–23) says so ("ResNet-34's only by where it was written"). The misnomer, if there is one, is the apex's own name. It does mark a real divergence, though: R50's tie was written from R34's pre-§1(f) proof and did not receive the peel. A second, harmless drift: `r34InputGradB` takes the block backwards `hb16 … hb1` (:401–416), while `r50InputGradB` takes them `hb1 … hb16` (:120–135).

### LeanMlir/Proofs/Nets/ResNet/ResNet34BackCertifiedTieB.lean:345 / :444, ResNet50WholeBackCertifiedTieB.lean:60 / :156, MobileNetV2WholeBackCertifiedTieB.lean:211 / :323, MobileNetV4WholeBackCertifiedTieB.lean:251 / :395

**Smell:** long-proof (statement) / repetition
**Current:** these eight "long proofs" are 90–140 lines of binders and 4–12 lines of proof. Each has, for example (R50:84–115, repeated verbatim at :180–211):
```lean
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ b1 (opaqueA0 (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) Ws bs εs γs βs) x)))
```
`PProd (HasVJPAt` occurs 257 times across these files: MNv4 70, R34 68, MNv2 55, R50 32, plus MNv2/MNv4 FullBVJP.
**Why it breaks:** the `opaqueA_k (stem …) b1 … b_k x` point is written twice per binder. A change to a stem's arguments means about 2 × 16–21 × 2 edits per net, and the eq/_correct pairs have to stay identical by hand.
**Suggested:** add `abbrev HasVJPDiffAt {m n} (f : Vec m → Vec n) (x : Vec m) : Type := PProd (HasVJPAt f x) (DifferentiableAt ℝ f x)` in `OpaquePrefix.lean`. It is an `abbrev`, so `.fst`/`.snd` and anonymous constructors still work and the statements stay defeq. That halves every binder. Going further, a `variable` block shared by each eq/_correct pair would remove the second copy, but Lean 4 `variable`s re-elaborate per declaration, so check the elaboration cost. ⚠ `r34InputGradB_eq_r34B_full_vjp` is in the comparator tier (`gen_comparator_tier.py:64`): regenerate and run the local comparator, as §1(o) did.

### LeanMlir/Proofs/Nets/ResNet/ResNet34SyncStepTieB.lean:560 — `r34_net_syncTiedB`; EfficientNetSyncStepTieG.lean:1391 — `efficientnet_net_syncTiedG`

**Smell:** repetition (divergent shape across siblings)
**Current:** R34 and EffNet put the smoothed-CE loss chain in the statement and derive the divisor inside the proof:
```lean
  have sG : ∀ r, g r = batchShard R N nCls (fun i => (R : ℝ) * G i) r :=
    fun r => replicaLossCot_eq R N nCls hR α B aStr negAK bStr logN ohN _ T r   -- R34:634–635
```
R50 (:796 `r50NetSyncTiedB`), MNv2 (:914) and MNv4 (:1175) instead state a named Prop with free `G`/`gs` and hypothesis `hgs`, and add a one-term `_smoothedCE` corollary (plus R50's `_bce`). R50's apex reuses R34's stem and head ties verbatim (:781–785), so the two proofs could be the same.
**Why it breaks:** R34 and EffNet each re-state the loss and cotangent graph inside their 70–115-line statements. Adding a loss, as R50's `_bce` did, would mean copying the whole statement.
**Suggested:** port R34 and EffNet to the §1(o) shape: `r34NetSyncTiedB … G gs : Prop`, `r34_net_syncTiedB … (hgs)`, and `r34_net_syncTiedB_smoothedCE := r34_net_syncTiedB … (fun r => replicaLossCot_eq …)`. The proof bodies shrink by the `sG` line and the loss `let`s. ⚠ Both are in the comparator tier (`gen_comparator_tier.py:54,58`), in formalization.yaml, and in content.tex (`\lean{}` at :5355/:8399). This is medium effort because of the tier regen and the blueprint entry, not because of the proof. Aside: the EffNet docstring (:1385) still says "the same fifty `0 < ε` hypotheses", but the binder has been `hεw : w.EpsPos` since §1(o).

### LeanMlir/Proofs/Nets/ViT/ViTBackB0.lean:276 — `mhsaClean_backward_collapseMH`

**Smell:** long-proof / undocumented-defeq / repetition
**Current:** two bare `show`s restate the goal through the `mhsaClean` witness's `vjpComp`/`rowwise`/`colSlabwise` structure (:283–306):
```lean
  funext r c
  show (rowwiseHasVJPMat (denseHasVJP (mhsaQkvW heads d Wq Wk Wv) …)).backward X
        ((colSlabwiseHasVJPMat (mhsaGHasVJPMat N d) …).backward … ) r c = _
  show Mat.mulVec (mhsaQkvW heads d Wq Wk Wv) (fun kj => (colSlabwiseHasVJPMat …).backward …) c = _
```
Then the three-way `if q.1 = 0 then sdpaBackQ … else if … sdpaBackK … else sdpaBackV …` term is written out in full three times: in the statement of `hdz` (:322–342), inside `rw [show … from rfl]` (:361–382) and as the `trans` target (:386–406). A fourth near-copy is the argument list of `qkv_back_fanin_MH` (:410–425). `hproj0`/`hproj1`/`hproj2` (:347–358) are the same proof with the index changed.
**Why it breaks:** both `show`s rely on how `mhsaClean`, `rowwiseHasVJPMat` and `colSlabwiseHasVJPMat` currently build their `.backward` definitionally, and no comment says so. A change to any of those builders (for example a `vjpComp_backward` restatement like §1(m)) breaks this proof at a 20-line `show` with no pointer to the cause. The three copies of the if-form must stay character-identical.
**Suggested:** extract three lemmas, each proved at variable arguments:
```lean
/-- The per-slab selector `mhsaG`'s backward reads. -/
noncomputable def sdpaBackSel (N d : Nat) (q : Fin 3) (Q K V dA : Mat N d) : Mat N d :=
  if q = 0 then sdpaBackQ N d Q K V dA else if q = 1 then sdpaBackK N d Q K V dA
  else sdpaBackV N d Q K V dA

theorem mhsaG_backward_eq_sel (N d : Nat) (M : Mat N (3 * d)) (dY : Mat N d) (r : Fin N)
    (j : Fin (3 * d)) :
    (mhsaGHasVJPMat N d).backward M dY r j
      = sdpaBackSel N d (finProdFinEquiv.symm j).1 (mhsaProjC 0 M) (mhsaProjC 1 M)
          (mhsaProjC 2 M) dY r (finProdFinEquiv.symm j).2 := rfl

theorem mhsaProjC_slab (q : Fin 3) (h : Fin heads) (X : Mat N (heads * d)) :
    mhsaProjC q (fun r' j_in => dense (mhsaQkvW heads d Wq Wk Wv) (mhsaQkvB heads d bq bk bv)
        (X r') (finProdFinEquiv (h, j_in)))
      = fun r' j => dense (![Wq, Wk, Wv] q) (![bq, bk, bv] q) (X r') (finProdFinEquiv (h, j))
```
The last one replaces `hproj0/1/2` with one `fin_cases q` proof. With these, `hdz` is `funext kj; rw [hslab, mhsaG_backward_eq_sel]; simp only [mhsaProjC_slab]`, and the `trans` target is `sdpaBackSel …` written once. Replace the two leading `show`s with `rw` on a named `mhsaClean_backward_apply` lemma (`rfl` at variables), or at least comment which builder's defeq each `show` depends on.

### LeanMlir/Proofs/Nets/ViT/ViTBackB0.lean:570 — `mhsaBackGraphMH_faithful`

**Smell:** long-proof / repetition
**Current:** after `funext j`, a 12-line `show (∑ h, den (SHlo.addV …) j) = _` (:620–632) restates the goal. Then `hQbr`/`hKbr`/`hVbr` are stated (:634–669) and applied through a 20-line
```lean
  rw [show (∑ h : Fin (hm1 + 1), ((den (…Wq…) j + den (…Wk…) j) + den (…Wv…) j))
      = ∑ h : Fin (hm1 + 1), ((Mat.flatten (…) j + …) + …)
      from by apply Finset.sum_congr rfl; intro h _; rw [hQbr h, hKbr h, hVbr h]]
```
(:670–690), which spells both sides of the sum out once more.
**Why it breaks:** the `show` depends on `den_headsSumG` leaving the sum in exactly that `addV` nesting, and the `rw [show …]` needs every `_` in that restatement to unify with the saved-activation terms. Any change to `mhsaBackGraphMH`'s operand order breaks both.
**Suggested:** drop the `show` and the `rw [show …]`. After `simp only [den_addV]`, use `simp only [hQbr, hKbr, hVbr]`: the three `have`s are `∀ h` equations, which `simp` rewrites under the `∑ h` binder. Then `unfold mhsaBackCollapsedMH Mat.flatten; rfl` as now. If `simp` does not fire because the operands are only defeq to the goal's, use `refine Finset.sum_congr rfl fun h _ => ?_` followed by `rw [hQbr h, hKbr h, hVbr h]`, which still removes the 20-line restatement. The `show` alone deserves a comment if it stays.

### LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtStepTie.lean:461 — `cnx_net_tied_certified`; ConvNeXtStepTieGB.lean:381 — `cnx_net_tiedGB`; ViT/ViTStepTie.lean:292 — `vit_net_tied_certified`; ViT/ViTStepTieGB.lean:390 — `vit_net_tiedGB`

**Smell:** long-proof (statement) / repetition
**Current:** ~200 loose per-block binders per statement, for example:
```lean
    (aW1 : DepthwiseKernel 96 7 7) (aB1 : Vec 96) (nG1 nB1 : Vec 96) (eW1 : Kernel4 384 96 1 1) (eB1 : Vec 384) (pW1 : Kernel4 96 384 1 1) (pB1 sL1 : Vec 96)
```
Each proof is 25 `exact cnx_block_ch_tiedAt … aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 lr` lines (ConvNeXtStepTie.lean:593–617), and ViT does the same with 16 per-block names at :380–392. The structures for these parameters already exist: `CnxBlockParamsCh` / `CnxDownParamsCh` / `CnxTWeightsCh` (ConvNeXtFullT.lean:123/194/241) and `BlockParamsV` (ViTDepthK.lean:38). The ties do not use them, whereas R34/R50/MNv2/MNv4/EffNet all tie over `…Weights` records.
**Why it breaks:** any per-block parameter change has to be made in 18 (or 12) binder lines, 18 `let`s, 18 conjuncts and 18 `exact`s, in each of two files.
**Suggested:** re-state over `w : CnxTWeightsCh nC` / a `ViTWeights` record, the way §1(o)/(t) did for MNv2/R34/R50. ⚠ Two things to check first. (1) The ConvNeXt ties use ONE `ε` for every block and downsample (`cnxBlockFwdChO ε …`), while `CnxBlockParamsCh` carries a per-block `εn`. Either add a tie-local record without `εn`, or state `∀ i, (w.blk i).εn = ε`. (2) `cnx_net_tiedGB` and `vit_net_tied_certified` are in the comparator tier (`gen_comparator_tier.py:47–48`). Medium effort, and the largest line saving in this audit.

### LeanMlir/Proofs/Foundation/DataParallelSync.lean:316 — `bnSyncTensor4GradInput_apply`

**Smell:** undocumented-defeq
**Current:**
```lean
  have hx : bnchwFwd N oc h w x (bnchwBackIdx N oc h w t) = x t := by
    rw [bnchwFwd_apply, bnchwFwdIdx_bnchwBackIdx]
  have hd : bnchwFwd N oc h w dy (bnchwBackIdx N oc h w t) = dy t := by
    show dy (bnchwFwdIdx N oc h w (bnchwBackIdx N oc h w t)) = dy t
    rw [bnchwFwdIdx_bnchwBackIdx]
```
**Why it breaks:** `hd` has the same shape as `hx`, but it unfolds `bnchwFwd` by `show` where `hx` uses the `bnchwFwd_apply` lemma that §1(m) added. If `bnchwFwd`'s body changes, `hd` breaks and `hx` does not.
**Suggested:** `hd` gets the same proof as `hx`: `rw [bnchwFwd_apply, bnchwFwdIdx_bnchwBackIdx]`. Better, both become one `have hfb : ∀ v, bnchwFwd N oc h w v (bnchwBackIdx N oc h w t) = v t`.

### LeanMlir/Proofs/Nets/EfficientNet/EfficientNetSyncStepTieG.lean:354 (and :366, :380, :391, :492) — `cbsB_back_eq`, `dwbsB_back_eq`, `dwbsSB_back_eq`, `projB_back_eq`, `hdCotIn_eq_vjp`

**Smell:** undocumented-defeq / repetition
**Current:** the same five-line tail appears five times:
```lean
  have hg := cbsBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
  rw [den_operand] at hg
  rw [← hg]
  show cInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
  rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
  rfl
```
**Why it breaks:** each `show` relies on `den` of the stage graph's outer node reducing to the named input-VJP applied to `den` of its BN-back operand. That is one `den` constructor step, which §1(g) moved behind `denStep`/`denStepApp`. The section header (:335–339) describes the idea but not the defeq.
**Suggested:** `simp only [denStep, denStepApp, den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]` should give the same one-constructor unfold without naming the result. §0 says to profile first. Otherwise, a one-line comment per `show`: "`den` of the graph's outer `convBackB` node is `cInB` of its operand's `den` — one constructor step".

### LeanMlir/Proofs/Foundation/DataParallelSyncKit.lean:412 — `bnSync_of_scaled` (γ half)

**Smell:** undocumented-defeq
**Current:**
```lean
    rw [show (bnPerChannelGradGamma oc ((R * N) * (h * w)) ε
          (bnchwFwd (R * N) oc h w (reassocB (R * N) oc h w V))
          (bnchwFwd (R * N) oc h w (reassocB (R * N) oc h w (fun i => (R : ℝ) * COT i))) k)
        = den (SHlo.bnGammaGradB vN epsStr ε (reassocB (R * N) oc h w V)
            (.operand cotN (fun i => (R : ℝ) * reassocB (R * N) oc h w COT i))) k from rfl,
```
**Why it breaks:** this folds the `den` of `bnGammaGradB` backwards by defeq, including the fact that `reassocB` commutes with the pointwise scaling. That second fact is only true definitionally because of `reassocB`'s current body.
**Suggested:** name the two facts, `den_bnGammaGradB_operand` (the `den` step) and `reassocB_smul : reassocB … (fun i => s * COT i) = fun i => s * reassocB … COT i` (which is also what :420 uses), and `rw` with them. Low priority: it is one site.

### LeanMlir/Proofs/Nets/Small/Cifar8BnStepTie.lean:52 — `cifar8Bn_convbn_tied_certified`

**Smell:** long-proof (low)
**Current:** a 110-line statement and a 24-bullet proof of the form `intro idx; exact CifarPoC.convW_den … cotC1 lr idx` / `convB_den` / `bnSgdPairTied_holds`, three per conv layer.
**Why it breaks:** it does not break easily. Each conjunct holds for any cotangent, so this is instantiation, not reasoning.
**Suggested:** optionally add `convBnSgdTriple_holds` (ConvW ∧ ConvB ∧ BnPair for one layer) and state the tie over it, which gives an 8-line proof. Otherwise leave it: the conjuncts carry per-layer comments.

---

## Not findings

* **The five sync apexes' bodies.** Once #4 is set aside, each is ~35–60 lines of `have sK := …_scaled` and one `exact ⟨…⟩`, with a comment per phase. The rest of the "110–152 lines" is the statement's `let` chains, which are the spec.
* **`DataParallelSync.lean:339` / `:384`.** Numbered, commented steps (P1b → P2b → P2a + anchor). Genuinely irreducible.
* **`IsShardwise` stays parked, and this re-read agrees.** There are 98 `fun r => by rw [X_shard …, X_smul]` sites. A generic `scaled_of_shard (hF : IsHomog F) (hsh : …) hdys` would still take the same `_shard` arguments at each site, so it saves about 0 lines per site. #2 and #4 are where the repetition actually is.
* **`r34BFullHasVJPAt` / `_backward`, `mnv4Chain_apply`.** The unrolled, fixed-arity VJP chains are deliberate (opaque stages, `rfl` only between variables). A heterogeneous-list apex would need dependent types that the §0 kernel rules make risky.
* **`DataParallelSyncBf16.lean`.** Its `rw [show … by ring]` at :305 and the `show … from` sites at :350/:365 are proved in place, not defeq.
* **`mnv2PreB*_apply`.** Stays, per §3.6.

## Recurring patterns (worth more than any single fix)

1. **One argument list, written twice.** The pair appears as scaled lemma + tie lemma in the five sync apexes (#4), `_eq` witness + `_correct` witness in the four WholeBack files (#1), `hbK` binder blocks in each eq/_correct pair (#6), and the three copies of the if-form in ViTBackB0 (#8). The fix is the same each time: a lemma whose type carries the shared term, so the second use is found by unification rather than re-typed.
2. **Siblings that missed a fix.** R50/MNv2/MNv4 did not get R34's §1(f) peel (#5). R34/EffNet sync did not get the §1(o) free-`G` shape (#7). ConvNeXt/ViT did not get the weight-record port that R34/R50/MNv2/MNv4/EffNet got (#10). A per-fix checklist of "which sibling nets have this shape" would have caught all three.
3. **Side conditions re-derived at every call site** (`nhw_ne_zero` ×71, `(by decide)` ×42 for MNv4 row heights, `h56 : 0 < 56 := by norm_num` preambles). Fold them into the kit lemma's hypotheses (#2), or into one named fact per net.
