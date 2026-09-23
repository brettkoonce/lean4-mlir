# Proof-quality audit — `LeanMlir/Proofs/Foundation/*.lean`, `LeanMlir/Proofs/Float/*.lean`

Scope: 41 hand-written files, 14,241 lines (none of them is in `gen.txt`). Static reading only, nothing
compiled. Every Mathlib name suggested below was grepped in `.lake/packages/mathlib/Mathlib` or is
already used elsewhere in this repo; where a replacement is untested it says so.

**Headline scan results:**
- `set_option maxHeartbeats` / `maxRecDepth`: **0** in scope.
- `change`: **0**.
- `native_decide`: **0**.
- `decide`: 2 uses, both at `IR.lean:300,309`. They check Nat oddness side conditions, so they are not findings.
- Tactic-position `show` (the rubric's "`show` used as `change`"): **41** sites.
- `nlinarith`: **52** calls, 28 of them in `FloatBridge.lean`.

So the heartbeat and `change` smells don't occur here. The findings are about compile cost, missing API lemmas, and
repetition.

---

## LeanMlir/Proofs/Foundation/DataParallelSyncBf16.lean  (36 s, slowest module in scope)

### DataParallelSyncBf16.lean:2 — `import LeanMlir.Proofs.Float.Binary32Instance`

**Smell:** compile-time (dependency / critical path)
**Current:**
```lean
import LeanMlir.Proofs.Foundation.DataParallelSync
import LeanMlir.Proofs.Float.Binary32Instance
```
The only thing used from the second import is `rndP` (`Binary32Instance.lean:51`). It is a 4-line
def over `Int.log`/`round`, plus `rndP_zero`.
**Why it breaks:** `Binary32Instance` imports `Float.FloatBridge`, the 1,891-line, 26 s module, and
`Training.SgdDescentLinear`. As a result, a data-parallel sharding file sits behind the whole float tier
on the build's critical path, and any edit to FloatBridge rebuilds it. The part of this file that
actually uses `rndP` (§ "The divisor step", lines 266–315: `int_log_two_pow_mul`,
`int_log_abs_two_pow_mul`, `rndP_two_pow_mul`, `rndP_mul_four`) is pure ℝ/`Int.log` and needs no
float tier either.
**Suggested:** Create a new leaf `LeanMlir/Proofs/Float/RndP.lean` that imports only `Mathlib.Data.Int.Log` plus Real. Move into it:
- `rndP` and `rndP_zero`;
- `rndP_err`, which the docstring says is Mathlib-only;
- lines 266–315 of this file.

Then `Binary32Instance` and this file both import `RndP`. This does not by itself explain the 36 s
spent inside the module. It does let the file start as soon as `DataParallelSync` is built.

### DataParallelSyncBf16.lean:155–260, 318–367 — plain/strided twin lemmas

**Smell:** repetition
**Current:** Six pairs of theorems have proofs that are identical word for word, apart from the constructor name:
- `den_convBackBatchedBf16_shard` / `den_convStridedBackBatchedBf16_shard` (101–102 vs 113–114);
- `…_eq_rnd` (126 vs 134);
- `convWGradShardSum` / `convStridedWGradShardSum` (139 vs 146);
- `…_shard` (163–167 vs 225–229);
- `…_global_split` (175–183 vs 238–244);
- `…_sub_global` (197–198 vs 259–260);
- `…_smul` (325–326 vs 336–337, 346–352 vs 361–367).

For example, both `_global_split`s are:
```lean
  rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN]      -- / den_convStridedWeightGradBBf16_eq_rnd
  congr 1
  simp only [convWGradShardSum, den]                  -- / convStridedWGradShardSum
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]
```
**Why it breaks:** Each new bf16 weight-gradient kind means copying all five lemmas again. The file
header already names a third kind (`convStridedXla…`), and `Bf16GradNodes.lean` has nine. A fix to one
copy (such as the `rw`-not-`simp` note at 181) has to be repeated by hand in every other copy.
**Suggested:** State the four weight-gradient facts once, about the ℝ-level batched weight gradient
of an arbitrary per-example VJP family:
```lean
noncomputable def batchWGrad {N a b P} (F : Vec a → Vec P → Vec b)   -- weight ↦ output, per example
    (H : ∀ x, HasVJP (F x)) (w : Vec P) (x : Vec (N * a)) (dy : Vec (N * b)) : Vec P :=
  fun idx => ∑ n, (H (batchSlice N a x n)).backward w (batchSlice N b dy n) idx
theorem batchWGrad_shard_sum … : batchWGrad F H w X DY idx = ∑ r, batchWGrad F H w (batchShard … X r) (batchShard … DY r) idx
theorem batchWGrad_smul … : batchWGrad F H w x (s • dy) = s • batchWGrad F H w x dy
```
Each SHlo kind then gets one `den_… = rnd (batchWGrad …) := rfl` bridge, and the shard, split, sub
and smul statements become one-liners per kind. A good home is this file, or `Bf16GradNodes.lean`,
which is also a leaf.

### DataParallelSyncBf16.lean:163–167, 225–229 — `rw …; rfl` closing across two defs

**Smell:** undocumented-defeq
**Current:**
```lean
  simp only [den_allReduceMeanF]
  congr 1
  apply Finset.sum_congr rfl; intro r _
  rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN, hdy]
  rfl
```
**Why it breaks:** The closing `rfl` has to see through two things:
- `convWGradShardSum` must unfold;
- `batchShard R N _ (fun i => rnd (X i)) r` must be recognised as `fun i => rnd (batchShard R N _ X r i)`.

This works because `batchShard` currently unfolds to the same lambda as the other side. If
`batchShard` becomes a structure projection or gets `@[irreducible]`, the proof fails with an opaque
`rfl` error. `DataParallelSync.lean:98` already has the lemma for this step, so the proof can name it.
**Suggested:** `rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN, hdy, convWGradShardSum,
batchShard_map (fun x => rnd x) X r]`. `batchShard_map` is the lemma at `DataParallelSync.lean:97–98`;
use its real name. This is untested. If the rewrite hits the dependent-type motive problem noted at
line 181, keep the `rfl` but add a one-line comment naming the two unfoldings it relies on. Also measure before changing anything: put `#count_heartbeats in` (Mathlib
`Util/CountHeartbeats.lean`) in front of `den_convWeightGradBBf16_global_split` and the two `_shard`
lemmas. Nothing in the file is statically heavy, so the 36 s has to be located by measurement.

### DataParallelSyncBf16.lean:312–315 — `rndP_mul_four`

**Smell:** fragile-simpa (the `norm_num at this; exact this` form)
**Current:**
```lean
  have := rndP_two_pow_mul p 2 x
  norm_num at this
  exact this
```
**Why it breaks:** This depends on `norm_num` normalising `(2:ℝ)^2` to exactly `4` and leaving
everything else alone. A `norm_num` simp-set change that also rewrites `rndP p (4 * x)` (for example,
commuting it) makes `exact` fail.
**Suggested:** `by simpa [show (2:ℝ) ^ 2 = 4 by norm_num] using rndP_two_pow_mul p 2 x`, or
`(by norm_num : (4:ℝ) = 2 ^ 2) ▸ rndP_two_pow_mul p 2 x`.

---

## LeanMlir/Proofs/Float/FloatBridge.lean  (26 s)

### FloatBridge.lean:159, 1489, 1491, 1522, 1525, 1675 (+ CrownBound.lean:267) — `nlinarith` on goals `linarith` closes

**Smell:** compile-time
**Current:**
```lean
    have h5 : u * u + 2 * u ≤ C * (1 + u) - 1 := by nlinarith                    -- 159
  have hN_ub : fexp (z k) ≤ (1 + eexp) * Real.exp (z k) := by
    nlinarith [abs_le.mp (hfexp (z k))]                                         -- 1489 (and 1491)
  have hS_lb : … := by
    have := abs_le.mp hS_err; nlinarith                                          -- 1522 (and 1525)
    rw [div_le_iff₀ (by linarith)]
    nlinarith                                                                    -- 1675
```
**Why it breaks:** In each case the needed fact is linear in the monomials once products are
expanded:
- line 159 is `h4 : (1+u)*(1+u) ≤ C*(1+u)` with the terms rearranged;
- lines 1489/1491 are `abs_le.mp` of `|f - e| ≤ ε·e`, split into two linear facts, with target `(1±ε)·e`;
- lines 1522/1525 follow the same pattern with `ρ·S`.

`linarith` already multiplies out polynomial products (`Mathlib/Tactic/Linarith/Parsing.lean`,
`Sum.mul`, line 93) and splits conjunctions (`Preprocessing.lean:44`). `nlinarith` also runs its
product preprocessing, which adds the pairwise products of every hypothesis and hint. Inside
`softmaxF_close`, line 1522 has about 15 comparison hypotheses in context, so that is roughly 100
extra atoms handed to the simplex. That makes it the most likely single hot spot in this file.
**Suggested:** Replace these calls with `linarith [...]` using the same hint terms. Where a real product is needed
(`hQs` 1582–1586, `hκρ` 1569–1575), use `nlinarith only [hQlb, hs0, hs1, hκ0]`, a form Mathlib uses
(`Analysis/Normed/Ring/Units.lean:56`), so the context is not multiplied. This is untested because no
builds were allowed. Treat it as try-`linarith`-first: do it one site at a time and look at the
`#count_heartbeats in` delta.

### FloatBridge.lean:1473–1604 — `softmaxF_close` (126-line proof)

**Smell:** long-proof
**Current:** There are three sections, marked "numerator sandwich", "denominator sandwich" and "pre-rounding quotient
sandwich". The first two contain the same argument four times, each time as `nlinarith [abs_le.mp h]`
turning `|a − b| ≤ ε·b` into `(1−ε)b ≤ a ≤ (1+ε)b` (1488–1491 and 1520–1525). The third derives
`(1+ε)/(1−ρ) = 1+κ` by `rw [div_eq_iff hne, add_mul, one_mul, div_mul_cancel₀ _ hne]; ring`
(1546–1549), a five-lemma chain whose job `field_simp` does.
**Why it breaks:** The 1548 chain relies on the exact shape after `simp only [smKappa]`. If Mathlib
renames `div_mul_cancel₀` (it has changed argument order before), it breaks. The four sandwich copies
also bloat the `nlinarith` context described above.
**Suggested:** Extract the following into FloatBridge, a leaf for these purposes:
```lean
theorem rel_sandwich {a b ε : ℝ} (h : |a - b| ≤ ε * b) :
    (1 - ε) * b ≤ a ∧ a ≤ (1 + ε) * b := by
  have := abs_le.mp h; constructor <;> linarith
theorem div_rel_close {p q P Q ε ρ : ℝ} … -- the hQub/hQlb pair: p/q vs P/Q given the two sandwiches
```
Also replace 1547–1549 with `simp only [smKappa]; field_simp; ring`. The proof then reads as
bound numerator, bound denominator, `div_rel_close`, `rnd_close`, in about 40 lines.

### FloatBridge.lean:924–1223 — `mlp_{w2,w1,b1,w0,b0}_step_float_close`

**Smell:** repetition (plus 6 unused `have`s)
**Current:** `w1`, `b1`, `w0` and `b0` each rebuild the same layer-1 cotangent facts word for word:
```lean
  have hcot := fun j' =>
    M.cot_step_close W₂ _ _ gt g hw₂ hG0 heg hW₂ hG hg l1 hmargin₁ j'
  have hc₁mag : ∀ j', |reluMask … (Proofs.dense (fun j'' i' => W₂ i' j'') (fun _ => 0) g) j'| ≤
      layerAct d₃ w₂ 0 G := fun j' =>
    (reluMask_abs_le _ _ j').trans (dense_abs_le hG0 (fun j'' i' => hW₂ i' j'') (fun _ => by simp) hG j')
```
`w0` and `b0` also share the `hcot`/`hc₁mag`/`hcot0`/`hc₀mag` blocks word for word (1122–1141 = 1189–1207).
Dead `have`s, meaning the name is bound and never referenced:
- `hE₀0` at 947, 1015, 1066, 1118 and 1186;
- `hC₁0` at 1014.
**Why it breaks:** Each block restates a 4–5 line reluMask/dense term by hand. Any change to
`reluMask` or `dense`'s argument order has to be made in about 12 places. The dead `have`s cost
elaboration time and would be flagged by Batteries' `unusedHavesSuffices` env-linter
(`Batteries/Tactic/Lint/Misc.lean:193`).
**Suggested:** Mirror what `mlp_l1_close` (893) already does for the forward pass:
```lean
theorem mlp_cot1_close … : (∀ j', |ct₁ j' - c₁ j'| ≤ layerBudget M.u d₃ w₂ 0 G eg) ∧
    (∀ j', |c₁ j'| ≤ layerAct d₃ w₂ 0 G)
theorem mlp_cot0_close … : (∀ j', |ct₀ j' - c₀ j'| ≤ …) ∧ (∀ j', |c₀ j'| ≤ layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G))
```
Each of the four step theorems then becomes `obtain` + `mul_close` + `sgd_step_close`, about 10 lines.
Delete the dead `have`s.

### FloatBridge.lean:139, 142, 184, 397, 764, 768, 794, 1498 (+ BnFloatBridge.lean:213) — hand-rolled `|a| ≤ |a−b| + |b|`

**Smell:** brittle-chain (small, but repeated about 10 times)
**Current:**
```lean
    have h1 : |st| ≤ |st - S| + |S| := by simpa using abs_sub_le st S 0
  …
    have h := abs_sub_le xt x 0
    simp only [sub_zero] at h
    linarith
```
**Why it breaks:** Both forms depend on `simp` normalising `st - 0` and `0 - S` into the right shape
(`sub_zero`, `zero_sub`, `abs_neg`). If the simp set changes, `simpa` either leaves a residue or
closes the goal too early.
**Suggested:** `linarith [abs_sub_abs_le_abs_sub st S]`. This Mathlib lemma is the additive form of
`mabs_div_mabs_le_mabs_div`, and the repo already uses it at `ConvMixedFloatBridge.lean:148` and
`Training/SgdDescent.lean:129`. The linarith step that follows usually absorbs it, so each site
shrinks to one hint.

### FloatBridge.lean:131–211 — `step_bound` / `dense_step_bound`

**Smell:** repetition
**Current:** Both proofs build `htri` the same way, including the `h3 : … = (…) + (…) := by ring;
rw [h3]; exact abs_add_le _ _` sub-step (150–154 and 191–195). Both also derive `|x| ≤ C·A` from
`|x − S| ≤ (C−1)A` (139–140 and 184–185).
**Why it breaks:** This is ordinary duplication. The `rw [h3]` depends on `ring` producing a
syntactically identical right-hand side.
**Suggested:** `htri` is `abs_sub_le r (st + pt) (S + p)` followed by
`abs_add_le (st - S) (pt - p)` after `ring_nf`. Or write it directly:
`calc |r - (S+p)| = |(r - (st+pt)) + ((st - S) + (pt - p))| := by ring_nf
_ ≤ _ := (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)`.

---

## LeanMlir/Proofs/Float/Binary32Instance.lean

### Binary32Instance.lean:140, 174, 311 — `#print axioms` in library code

**Smell:** compile-time / non-mainstream
**Current:** `#print axioms binary32_linear_sgd_descends_concrete` (and two more)
**Why it breaks:** Each one prints on every build of this module and of every downstream module that
replays it in CI logs. Mathlib keeps `#print`/`#check`/`#eval` out of library files. The repo already
has the proper place for this check: `tests/AuditAxioms.lean`, cited at `IR.lean:34`.
**Suggested:** Move the three checks into `tests/AuditAxioms.lean`.

### Binary32Instance.lean:212–310 — `binary32_linear_sgd_descends_concrete` (101 lines)

**Smell:** repetition + undocumented-defeq + non-terminal simp
**Current:**
```lean
    show (∑ i, x0 i * W0 i j) + b0 j = 0                 -- 222: unfolds `dense`
  …
  have hSabs : … = 1 := by
    rw [← finProdFinEquiv.sum_comp (fun idx => |gradAt … idx|)]
    rw [Fintype.sum_prod_type]
    simp_rw [key]
    simp [Fin.sum_univ_two, x0, oneHot, lbl]              -- 248: non-terminal
    norm_num
  have hSsq : … = 1 / 2 := by   -- same 5 lines, `^ 2` instead of `|·|`
  …
  have hη  : … ≤ 1/500 := by simp only [binary32_u, FloatModel.mulErr, FloatModel.cotErr,
      FloatModel.smErr, FloatModel.smKappa, FloatModel.smRho]; norm_num [u32]
  have hη0 : 0 ≤ … := by  -- identical simp only list
```
**Why it breaks:**
- The `show` depends on `dense` unfolding to exactly that sum.
- Lines 248–249 are a non-terminal bare `simp` followed by `norm_num`. Mathlib's `linter.flexible` flags this, because what `norm_num` receives depends on the whole simp set.
- `hSabs` and `hSsq` differ only in the summand.
**Suggested:**
- Replace the `show` with `simp [dense, W0, b0]`.
- Merge 248–249 into one terminal `norm_num [Fin.sum_univ_two, x0, oneHot, lbl]`.
- Factor a local `have hsum : ∀ φ : ℝ → ℝ, ∑ idx, φ (gradAt … idx) = ∑ i, ∑ j, φ (x0 i * (1/2 - oneHot 2 lbl j))` and instantiate it at `|·|` and `(·^2)`.
- Compute `η` once (`have hηv : η = … := by simp only […]; norm_num [u32]`) and derive both bounds from it.

---

## LeanMlir/Proofs/Foundation/Tensor.lean  (root file: 423 downstream modules — batch these with the next root edit)

### Tensor.lean:437–447, 978–990 — `Mat.flatten`/`Tensor3.flatten` have no `_apply` API; round-trips not `@[simp]`

**Smell:** undocumented-defeq (repo-wide pattern)
**Current:**
```lean
noncomputable def flatten {m n : Nat} (A : Mat m n) : Vec (m * n) :=
  fun k => let p := finProdFinEquiv.symm k; A p.1 p.2
theorem unflatten_flatten … := by funext i j; simp [unflatten, flatten]   -- not @[simp]
…
    funext; simp [Mat.flatten, Mat.mul, Mat.unflatten, mul_add, Finset.sum_add_distrib])   -- 680
    funext; simp [Mat.flatten, Mat.mul, Mat.unflatten, Finset.mul_sum, mul_left_comm])]    -- 682
  by_cases h : l = j <;> simp [Mat.flatten, Mat.mul, Mat.unflatten, h, Prod.ext_iff]      -- 683
```
**Why it breaks:** Every consumer unfolds the definitions by name. Repo-wide there are **112**
`simp`/`unfold`/`rw` sites naming `Mat.(un)flatten` and **153** naming `Tensor3.(un)flatten`, 18 of
them in this scope. The `let` in both bodies means each unfold leaves a `let`/zeta-redex that simp has
to clean up. The round-trips `Mat.unflatten_flatten` and `flatten_unflatten` are cited explicitly
58 times because they are not `@[simp]`.
**Suggested:**
- Drop the `let`s: `fun k => A (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2`.
- Add `theorem flatten_apply (A) (k) : flatten A k = A (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2 := rfl` and `unflatten_apply (v i j) : unflatten v i j = v (finProdFinEquiv (i, j)) := rfl`, and the same for `Tensor3`.
- Mark the round-trips `@[simp]`.

⚠ These are rfl lemmas. In whole-net VJP ties use them with `rw`, never `simp only`, because of the
48 GB dsimp-replay hazard. Adding `@[simp]` also changes what every bare `simp` downstream does, so
land it on a branch and let CI (`lake build Certs`) show the fallout.

### Tensor.lean:404–408 — `vjp_comp` has no `backward` lemma

**Smell:** undocumented-defeq
**Current:** The consumers `show` their way through the `where`-built field:
```lean
  show (reassocFwd_has_vjp oc h w).backward x                       -- PerChannelBN.lean:366
        ((bnPerChannelFlat_has_vjp oc (h * w) ε hε γ β).backward (reassocFwd oc h w x)
          ((reassocBack_has_vjp oc h w).backward …)) = _
  show (bnchwFwd_has_vjp N oc h w).backward x …                     -- PerChannelBN.lean:751
```
`Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean:398–401` had to restate the `_at` version
locally as `vjp_comp_diff_at_fst_backward` so it could be `rw`'d. The docstring there explains why.
**Why it breaks:** The `show` restates a 3–4 line nested `.backward` term by hand. It depends on the
unfolding order of two nested `vjp_comp` structures.
**Suggested:** Add next to `vjp_comp` and `vjp_comp_at`:
```lean
theorem vjp_comp_backward … : (vjp_comp f g hf' hg' hf hg).backward x dy
    = hf.backward x (hg.backward (f x) dy) := rfl
theorem vjp_comp_at_backward … : (vjp_comp_at f g x hf' hg' hf hg).backward dy
    = hf.backward (hg.backward dy) := rfl
```
Use them with `rw` in the ties. The ConvNeXt local copy then becomes a one-line corollary.

---

## LeanMlir/Proofs/Foundation/PerChannelBN.lean

### PerChannelBN.lean:167–195, 418–441 — index relabelings as `let`-defs with hand-proved inverses

**Smell:** brittle-chain / non-mainstream
**Current:**
```lean
noncomputable def reassocBackIdx (oc h w : Nat) (t : Fin (oc * h * w)) : Fin (oc * (h * w)) :=
  let chw := finProdFinEquiv.symm t
  let ch := finProdFinEquiv.symm chw.1
  finProdFinEquiv (ch.1, finProdFinEquiv (ch.2, chw.2))
theorem reassocFwdIdx_reassocBackIdx … := by
  unfold reassocFwdIdx reassocBackIdx
  simp only [Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
```
The same 3-line proof appears four times (188, 194, 434, 440). The consumers then need `show` to see
through `bnchwFwd` (544, 872, 875) and pass `hστ`/`hτσ` inverse pairs to
`reindexVJP_backward_of_inv` (144).
**Why it breaks:** Each proof depends on the exact nesting of `Prod.mk` left behind after `unfold` +
zeta. `Prod.mk.eta` has to fire twice, in that order.
**Suggested:** Build each pair as one `Equiv` and read both directions off it, so the inverse lemmas
are just `Equiv.apply_symm_apply`/`Equiv.symm_apply_apply`:
- `bnchwIdxEquiv : Fin (oc*(N*(h*w))) ≃ Fin (N*(oc*(h*w)))` built from `finProdFinEquiv`, `Equiv.prodCongr` and `Equiv.prodAssoc`, the same pieces as `bnShardEquiv` at 572.
- `reassoc` preserves the value. `finProdFinEquiv (x, y) = y + n·x` (Mathlib `Logic/Equiv/Fin/Basic.lean:332`), so `((c,hi),wi)` and `(c,(hi,wi))` both encode to `wi + w·hi + w·h·c`. That makes `reassocFwdIdx = Fin.cast (Nat.mul_assoc oc h w).symm`. Prove this with `Fin.ext` + `simp [finProdFinEquiv_apply_val]` + `ring`, and state the round-trips through `Fin.cast`.

Add `bnchwFwd_apply : bnchwFwd N oc h w x t = x (bnchwFwdIdx N oc h w t) := rfl` to replace the
three `show`s.

---

## LeanMlir/Proofs/Foundation/DataParallel.lean

### DataParallel.lean:141–147 and 276–284 — same 7-step `rw` chain behind a `show`

**Smell:** brittle-chain + repetition + undocumented-defeq
**Current:**
```lean
  show (1 / (R : ℝ)) * ∑ r : Fin R, ((1 / (N : ℝ)) * ∑ n : Fin N, ℓ (e (r, n)) θ)
       = (1 / ((R * N : Nat) : ℝ)) * ∑ k : Fin (R * N), ℓ k θ
  rw [← Finset.mul_sum, ← mul_assoc, div_mul_div_comm, one_mul, Nat.cast_mul,
      ← Equiv.sum_comp e (fun k => ℓ k θ), Fintype.sum_prod_type]
```
The same chain appears at 283–284 for `dpSyncGrad_eq_globalBatchGrad`. The file has six `show`s that
unfold its own `lossGrad`/`meanLoss`/`dpMean`/`dpSingleStep`: 110, 144, 207, 257, 282, 351.
**Why it breaks:** `div_mul_div_comm` and `one_mul` have to match the exact bracket shape produced by
`← mul_assoc`. Any change to how `meanLoss` associates `1/M * Σ` breaks both copies.
**Suggested:** Add rfl lemmas `meanLoss_apply`, `dpMean_apply` and `lossGrad_apply` in this file, which
is not a root file. Factor out the scalar fact:
```lean
theorem mean_mean_equiv {R N : ℕ} (e : Fin R × Fin N ≃ Fin (R * N)) (f : Fin (R * N) → ℝ) :
    (1 / (R : ℝ)) * ∑ r, ((1 / (N : ℝ)) * ∑ n, f (e (r, n))) = (1 / ((R * N : ℕ) : ℝ)) * ∑ k, f k := by
  rw [← Equiv.sum_comp e, Fintype.sum_prod_type, Finset.mul_sum, Finset.mul_sum]; push_cast
  refine Finset.sum_congr rfl fun r _ => ?_; rw [Finset.mul_sum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun n _ => ?_; ring
```
Both theorems then become `funext _; simp only [meanLoss_apply]; exact mean_mean_equiv e _`. The
`mean_mean_equiv` proof is untested.

---

## LeanMlir/Proofs/Foundation/Bf16GradNodes.lean

### Bf16GradNodes.lean:57–63, 76–82, 95–101, 113–119, 131–138, 151–157, 170–176 (+194, 218) — nine copies of one proof

**Smell:** repetition
**Current:**
```lean
  simp only [den]
  congr 1
  apply Finset.sum_congr rfl
  intro n _
  exact (flatConvStride2_weight_grad_has_vjp b (fun j => rnd (batchSlice …))).correct
    (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx
```
**Why it breaks:** Nine hand-copied bodies, each restating two `batchSlice` lambdas. When a tenth
bf16 weight-gradient kind is added, the whole block is copied again. That has already happened
three times in this file for depthwise, strided and XLA.
**Suggested:** Add to this leaf file:
```lean
theorem rnd_sum_backward {N a P : ℕ} (rnd : ℝ → ℝ) (F : Fin N → Vec P → Vec a)
    (H : ∀ n, HasVJP (F n)) (w : Vec P) (c : Fin N → Vec a) (idx : Fin P) :
    rnd (∑ n, (H n).backward w (c n) idx) = rnd (∑ n, ∑ j, pdiv (F n) w idx j * c n j) :=
  congrArg rnd (Finset.sum_congr rfl fun n _ => (H n).correct w (c n) idx)
```
Each theorem then becomes `by simp only [den]; exact rnd_sum_backward rnd _ (fun n => …_has_vjp …) _ _ idx`.

---

## LeanMlir/Proofs/Foundation/EvenKernelConvBack.lean

### EvenKernelConvBack.lean:95–118 — `conv2d_padOdd_eq` restates `conv2d`'s summand twice

**Smell:** brittle-chain / undocumented-defeq
**Current:**
```lean
  have hrow : (∑ kw : Fin (kW + 1), padOdd W o c 0 kw *
      (let pH := (kH + 1 - 1) / 2
       let pW := (kW + 1 - 1) / 2
       let hh := (0 : Fin (kH + 1)).val + hi.val
       … if hpad : … then x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩ else 0)) = 0 := by
    refine Finset.sum_eq_zero (fun kw _ => ?_)
    rw [padOdd_zero_row, zero_mul]
  rw [hrow, zero_add]
  … have hcol : padOdd W o c kh.succ 0 * (let pH := … (the same 7 lines)) = 0 := by
    rw [padOdd_zero_col, zero_mul]
```
**Why it breaks:** The `have` statements copy `conv2d`'s body, including its `let`s, character for
character. Any edit to `conv2d` in `Architectures/CNN.lean`, even renaming a `let` or reordering the
guard, breaks the `rw [hrow]` match with a "motive is not type correct" or "did not find instance"
error. That failure is far away from the actual change.
**Suggested:** `padOdd_zero_row` and `padOdd_zero_col` are already `@[simp]` rfl lemmas (68–72), so let
simp remove the zero rows instead of stating them:
`simp only [Fin.sum_univ_succ, padOdd_zero_row, padOdd_zero_col, zero_mul, Finset.sum_const_zero, zero_add]`
This replaces lines 95–118 and the two `have`s disappear.

---

## LeanMlir/Proofs/Foundation/MuonGeometry.lean

### MuonGeometry.lean:170–172, 219, 344–415 (`shampoo_eq_muon`, 70 lines), 431, 455, 458 — reassociate-by-`show` chains

**Smell:** brittle-chain
**Current:** There are six instances of:
```lean
    rw [show (W * Matrix.diagonal a * Wᵀ) * (W * Matrix.diagonal b * Wᵀ)
          = W * (Matrix.diagonal a * (Wᵀ * W) * Matrix.diagonal b) * Wᵀ from by
            simp only [Matrix.mul_assoc],
       hW, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
```
There are 10 of them, at 170, 172, 219, 360, 385, 391, 407, 431, 455 and 458. `hpt4`/`hpt2` (366–377) repeat
the same four-line `sqrt` setup.
**Why it breaks:** Each `show` writes out a parenthesisation by hand just so a later rewrite can match
it. If a single `Matrix` lemma changes which bracket shape it produces, the proof breaks.
**Suggested:** The Mathlib idiom is to normalise with `Matrix.mul_assoc` and provide cancel lemmas in the same normal form:
```lean
have hU' : ∀ X, Uᵀ * (U * X) = X := fun X => by rw [← Matrix.mul_assoc, hU, Matrix.one_mul]
have hV' : ∀ X, Vᵀ * (V * X) = X := …
simp only [Matrix.mul_assoc, hU', hV', Matrix.diagonal_mul_diagonal_assoc]  -- or diagonal_mul_diagonal after one ← mul_assoc
```
That replaces each `rw [show … from by simp only [Matrix.mul_assoc], …]`. Move the `sqrt` setup into
`have hd2 : ∀ i, d i * d i * s i = 1` and derive `hpt4`/`hpt2` from it with `ring_nf`/`nlinarith only`.
Separately, `hMle` at 196 (`nlinarith [hMsq]`) is `le_of_sq_le_sq (by rwa [one_pow]) zero_le_one`
(Mathlib `Algebra/Order/Ring/Abs.lean:134`).

---

## LeanMlir/Proofs/Foundation/CrownBound.lean  (14 s)

### CrownBound.lean:108–109, 267, 251

**Smell:** compile-time + undocumented-defeq
**Current:**
```lean
  · rw [max_eq_right hz]; nlinarith                                  -- 108
  · rw [max_eq_left (le_of_not_ge hz)]; nlinarith [le_of_not_ge hz]  -- 109
  nlinarith [this]                                                   -- 267 (goal linear in `this`)
  show denseE W2 (reluE (denseE W1 x')) c = _                        -- 251 (unfolds ∘)
```
**Why it breaks:**
- Line 267: `this : a*z + c ≤ (W2 y t − W2 j t) * max z 0` and the goal is its expansion, which is linear once products are expanded.
- Lines 108–109 are Mathlib lemmas.
- Line 251 depends on `Function.comp` reducing when `show` elaborates it.

**Suggested:**
- Line 267: `linarith [this]`.
- Line 108: `exact mul_nonpos_of_nonneg_of_nonpos h0 hz`.
- Line 109: `exact mul_le_of_le_one_left (le_of_not_ge hz) h1`. Both names were grepped: `Algebra/Order/GroupWithZero/Basic.lean:69,361`.
- Line 251: `rw [Function.comp_apply, Function.comp_apply]`.

---

## Recurring patterns (fix the pattern, fix every site)

1. **Missing `_apply` / field API lemmas, so defs get unfolded by `show` or `simp [defName]`.** About 41 tactic `show`s in scope, plus roughly 265 repo-wide `simp`/`unfold` sites that name `Mat.flatten`/`unflatten`/`Tensor3.flatten`/`unflatten` directly. Specific gaps:
   - `Mat.flatten_apply`, `Mat.unflatten_apply`, `Tensor3.flatten_apply`, and `@[simp]` on the round-trips (Tensor.lean);
   - `vjp_comp_backward` and `vjp_comp_at_backward` (Tensor.lean);
   - `bnchwFwd_apply` (PerChannelBN);
   - `meanLoss_apply`, `dpMean_apply`, `lossGrad_apply` (DataParallel).

   The mainstream fix is a `rfl` lemma per projection, used with `rw` in ties because of the dsimp hazard. The Tensor.lean part costs a 423-module rebuild, so do it in one batch.

2. **Copy-paste twins instead of a lemma over the varying piece.** Six plain/strided pairs in DataParallelSyncBf16, nine identical bodies in Bf16GradNodes, four MLP step proofs sharing about 30 lines each, and two DataParallel `rw` chains. That is about **25 declarations** that shrink to about 5 lemmas plus one-line instances. Every new bf16 or strided kind currently copies about 40 lines. `batchWGrad_*`, `rnd_sum_backward`, `mlp_cot{1,0}_close` and `mean_mean_equiv` are the extraction targets. All go in leaf files.

3. **`nlinarith` where `linarith` suffices, run over large contexts.** There are 52 `nlinarith` calls in scope, 28 in FloatBridge. At least 8 of them are on goals that are linear once products are expanded (FloatBridge 159, 1489, 1491, 1522, 1525, 1675; CrownBound 267; MuonGeometry 196). With no hints, `nlinarith` multiplies every pair of hypotheses in context. `linarith` already expands products (`Linarith/Parsing.lean:93`), and `nlinarith only […]` keeps the product set small. This is the most likely static contributor to FloatBridge's 26 s. Confirm each site with `#count_heartbeats in`.

Smaller recurring items:
- The `rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]` chain becomes `rw [Fin.sum_const, nsmul_eq_mul]`. There are 5 sites in scope: DataParallelSync:112 (as `simp only`), SmoothedLossCot:129, BnFloatBridge:210 and 299, BnInputBridge:84. Repo-wide there are 14. `Fin.sum_const` is used in Mathlib, for example `NumberTheory/Height/NumberField.lean:154`.
- `simpa using abs_sub_le a b 0` becomes `linarith [abs_sub_abs_le_abs_sub a b]`, about 10 sites.
- `let` inside the bodies of `Mat.flatten`, `Tensor3.flatten`, `reassoc*Idx` and `bnchw*Idx`: every unfold then carries zeta-redexes.
- `#print axioms` ×3 in Binary32Instance belongs in `tests/AuditAxioms.lean`.
