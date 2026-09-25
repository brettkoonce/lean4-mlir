# Proof-quality re-audit: the 19 files added since 2026-09-22

Static read only. Nothing was compiled or edited. Every "closes with X" below is a prediction until
someone measures it (planning/proof_cleanup.md §0). Paths are relative to `LeanMlir/Proofs/`.
Toolchain v4.34.0.

**Mechanical scan of the 19 files:** 0 `set_option` (no heartbeat or recDepth bumps), 0 `change`,
0 `sorry`/`native_decide`, 2 `nlinarith`, 7 unscoped `simp` calls (all of them terminal), 17
`show` / `rw [show …]`.

**Clean. No findings:** `Foundation/Batched.lean`, `Foundation/BatchedStages.lean` (all
term-mode), `Foundation/GramQ.lean`, `Foundation/HeadLayers.lean` (its one `rfl` after `rw` is
explained in the docstring), `Foundation/IndexCast.lean`, `Foundation/SgdNodes.lean`
(`BnSgdPairTied` restating the two `_den` statements is deliberate packaging),
`Nets/MobileNet/MobileNetV4Spec.lean` (data only).
**Nearly clean, one trivial item each:** `Float/FloatClose.lean`, `Float/RndP.lean`.

The list below is ranked by payoff divided by effort, not grouped by file. Each item carries an
effort tag.

---

### 1. Foundation/BatchedBackLinks.lean:186-207 and :73-75: `bnBatchLA_back_conj`, `bnBatchBack_faithful`

**Smell:** repetition, undocumented-defeq · **Effort:** small
**Current:**
```lean
-- :73
  show bnBatchTensor4GradInput N oc h w ε γ x (den e) i = _
  rw [bnBatchTensor4GradInput_correct N oc h w ε hε γ β x (den e) i,
      ← bnBatchTensor4HasVJP_correct N oc h w ε hε γ β x (den e) i]
-- :194-205  (an 8-line `hb` that restates the whole LHS twice, then)
    funext i
    rw [bnBatchTensor4GradInput_correct N oc h w ε hε γ β,
        ← bnBatchTensor4HasVJP_correct N oc h w ε hε γ β]
  rw [hb]
```
**Why it breaks:** both sites prove "the renderable BN input-grad equals the certified VJP
backward" the same way: they go out to the `pdiv` sum and back (`grad_input_correct`, then
`← hasVJP_correct`). `PerChannelBN.lean:640` proves `grad_input_correct` by going through the
backward in the first place, so the fact is derived three times. The `show` at :73 relies on the
`den` of `bnBatchBack` unfolding definitionally, and no comment says so.
**Suggested:** park one lemma in this leaf. Per §0, not in PerChannelBN.
```lean
theorem bnBatchTensor4GradInput_eq_backward (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ β : Vec oc) (x dy : Vec (N * (oc * (h * w)))) :
    bnBatchTensor4GradInput N oc h w ε γ x dy
      = (bnBatchTensor4HasVJP N oc h w ε hε γ β).backward x dy :=
  funext fun i => by rw [bnBatchTensor4GradInput_correct N oc h w ε hε γ β,
                         ← bnBatchTensor4HasVJP_correct N oc h w ε hε γ β]
```
With it, `hb` goes away (`rw [bnBatchTensor4GradInput_eq_backward (hε := hε) (β := β)]`). At
:73 the proof is the lemma, applied after the `den` unfold. Keep the `show`, add a one-line
comment that it rests on `den (.bnBatchBack …)` being the grad-input definitionally, or open with
`simp only [denStepApp]` as the file's siblings do.

### 2. Architectures/Softmax.lean:181-192: `softmaxCE_grad` re-derives `crossEntropy_differentiable`

**Smell:** repetition · **Effort:** trivial
**Current:**
```lean
  have h_log_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => Real.log (softmax (c' + 1) z label)) :=
    fun z => (h_softmax_label_diff z).log (h_softmax_pos z).ne'
  have h_ce_pi_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => fun _ : Fin 1 => crossEntropy (c' + 1) z label) := by
    rw [differentiable_pi]
    intro _
    simp only [crossEntropy_def]
    exact h_log_diff.neg
```
**Why it breaks:** `crossEntropy_differentiable` at :233, in the same file, proves exactly this.
Its own docstring calls itself "the standalone form of the differentiability infrastructure inside
`softmaxCE_grad`". When one copy changes, the other goes stale.
**Suggested:** move `crossEntropy_differentiable` above `softmaxCE_grad` and write
`have h_ce_pi_diff := differentiable_pi.2 fun _ => crossEntropy_differentiable (c' + 1) label`.
Then `h_log_diff` is dead, and `h_softmax_pos` is needed only at `logits` (for `hp_ne`).

### 3. Architectures/Softmax.lean:63-67, :195-199, :219-222 + Architectures/BatchNorm.lean:574-581: "pdiv is the fderiv of the j-th coordinate"

**Smell:** repetition, undocumented-defeq · **Effort:** small
**Current:** the same three steps appear four times: `unfold pdiv`,
`rw [fderiv_apply (hdiff) j]`, `rfl`. Two of the copies are wrapped in a `show` with no comment
(`Softmax.lean:221`, `BatchNorm.lean:576`):
```lean
    show _ = fderiv ℝ (softmax (c' + 1)) logits (basisVec j) label
    rw [fderiv_apply ((softmax_differentiable (c' + 1)) logits) label]; rfl]
```
**Why it breaks:** every copy relies on `pdiv f x i j` being `fderiv ℝ f x (basisVec i) j`
definitionally, and on the closing `rfl` evaluating a `ContinuousLinearMap.proj`/`comp`
application. If `fderiv_apply` changes form, four proofs break at once.
**Suggested:** park this lemma in Softmax.lean (a leaf) and move it later as part of a Tensor batch:
```lean
theorem pdiv_eq_fderiv_coord {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (hf : DifferentiableAt ℝ f x) (i : Fin m) (j : Fin n) :
    pdiv f x i j = fderiv ℝ (fun y => f y j) x (basisVec i) := by
  unfold pdiv; rw [fderiv_apply hf j]; rfl
```
:195-199 is the same lemma at `(j, 0)` with `f := fun z _ => crossEntropy …`. :219-222 is the
same lemma used right-to-left.

### 4. Certificates/DenseEuclid.lean:113-121 and :241-249: "x² ≤ y², y ≥ 0 ⇒ x ≤ y" done by square roots

**Smell:** brittle-chain, repetition · **Effort:** trivial
**Current:**
```lean
    rw [h1, h2, ← sq_abs (max (u i) 0 - max (w i) 0), ← sq_abs (u i - w i)]
    exact pow_le_pow_left₀ (abs_nonneg _) habs 2
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _),
       one_mul] at *
```
and, for the lower bound, a 5-line `calc` through `Real.sqrt_sq` / `Real.sqrt_le_sqrt`.
**Why it breaks:** `rwa … at *` rewrites every hypothesis in context, so adding a new hypothesis
can change what it closes. `pow_le_pow_left₀` has been renamed twice upstream. The same file
already does this step properly at :51 and :154 (`abs_le_of_sq_le_sq'`).
**Suggested:** `rw [h1, h2]; exact sq_le_sq.2 habs` (Mathlib `Algebra/Order/Ring/Abs.lean:119`).
Then close with `rw [one_mul]; exact (abs_le_of_sq_le_sq' hsq (norm_nonneg _)).2`. At :241-249,
`h1 := (abs_le_of_sq_le_sq' e (norm_nonneg _)).2`.

### 5. Architectures/ConvIndex.lean: five one-line fixes

**Smell:** repetition, fragile-simpa, undocumented-defeq · **Effort:** trivial (all five)
- **:34-37 `max4_sub_abs_le_sum`:** `nlinarith [abs_nonneg …×4]` → `linarith [...]`. Each goal is
  `|a-a'| ≤ |a-a'|+|b-b'|+|c-c'|+|d-d'|`, which is linear in the four atoms.
- **:23-27 `max4_sub_abs_le`:** the body repeats `CNN.lean:1440 max_close`, which the file
  imports. It collapses to `max_close (max_close h1 h2) (max_close h3 h4)`, the same shape
  `maxPool2_close` (CNN.lean:1452) uses.
- **:62-64 `flatten_t3Idx`:** `unfold Tensor3.flatten t3Idx; simp` is an unscoped terminal simp.
  It becomes `by rw [← unflatten_t3Idx, Tensor3.unflatten_flatten]`, using the rfl lemma at :55
  and `Tensor.lean:941`, which is the same move `batchSlice_batchMap` makes.
- **:209 `ne_of_gap_of_close`:** `have heq' : ya - yb = 0 := by rw [heq]; ring` is dead, because
  `linarith` already uses `heq : ya = yb`. Delete the line.
- **:258-263 `isArgmax_iff`:** `hxw` and `hyw` are the same fact for `x` and for `y`. Make it one
  `have hw : ∀ z : Tensor3 c (2*h) (2*w), z ci (winRowInv (winRow hi) …) … = z ci hi wi`.
  Separately, :159's undocumented `show Tensor3.flatten (maxPool2 …) … = _` is `unfold maxPoolFlat`.

### 6. Foundation/BatchedBackLinks.lean:378-387 and :354-365

**Smell:** undocumented-defeq, repetition · **Effort:** trivial
**Current:**
```lean
  show (bnBatchLAHasVJP N oc h w ε hε γ β).backward x (den (.operand "" dy)) = _
  rw [← bnBatchLABack_faithful "" "" "" ε γ β hε x (.operand "" dy),
      den_bnBatchLABack_eq_bnBatchBack]
  rfl
```
**Why it breaks:** the `show` and the closing `rfl` both rely on `den (.operand _ v) ≡ v` (a
structural-recursion unfold) and on `bnBackB` unfolding. The twin at :475-481
(`bnInB_eq_den_bnBatchBack`) is already term-mode.
**Suggested:** use the term
`(bnBatchLABack_faithful "" "" "" ε γ β hε x (.operand "" dy)).symm.trans
  (den_bnBatchLABack_eq_bnBatchBack "" "" "" ε γ x (.operand "" dy))`. It has the same defeq
load, but the load now sits in one elaboration check rather than in a `show` plus an `rfl`. At
:354-365, `h1` and `h2` are the same `Fin.cast` iff, once at `H.symm` and once at `H`. State
`∀ {p q} (H : p = q) a b, a = Fin.cast H b ↔ Fin.cast H.symm a = b` once
(`simp only [Fin.ext_iff, Fin.val_cast]`) and instantiate it twice.

### 7. Certificates/GaussianQuantile.lean:215-222 `stdNormalCDF_pos`; :196

**Smell:** repetition · **Effort:** trivial
**Current:** an 8-line proof that goes interval-mass → `toReal_pos` → `cdf_eq_real`. It repeats the
measure reasoning already done inside `stdNormalCDF_strictMono` (:56-59).
**Suggested:** `(cdf_nonneg (μ := gaussianReal 0 1) (s - 1)).trans_lt (stdNormalCDF_strictMono (sub_one_lt s))`.
The lemma is Mathlib `Probability/CDF.lean:64`. At :196,
`rw [show stdNormalCDF q = … from hll.symm]` is `rw [← hll]`. At :123-126, `hBbdd` is the mirror
image of `stdNormalCDF_sublevel_bddAbove` and can be a named `…_superlevel_bddBelow (hp : 0 < p)`
(low payoff).

### 8. Float/ConvFloat.lean:127, :132, :144, :240/:242/:256

**Smell:** fragile-simpa, undocumented-defeq, repetition · **Effort:** trivial
- :127 and :132 use unscoped terminal `simp [convWindow, w3Idx, Equiv.symm_apply_apply]`. Make
  them `simp only`, with the set `flatten_k4Idx` already uses at :53.
- :144 `conv2d_eq_dense` restates the whole goal with `show` so that `Proofs.dense` unfolds, and
  there is no comment. Use `unfold Proofs.dense` (the repo has no `dense_apply`; parking one here
  also works).
- `huf_e`/`huf_a`/`huf`: `intro c i j; simp only [Tensor3.unflatten]; exact hvte _` is the term
  `fun _ _ _ => hvte _`, since `Tensor3.unflatten_apply` is `rfl`. The same
  `simp only [Tensor3.unflatten]; exact` idiom appears 11 times across the repo.

### 9. Architectures/ChannelLN.lean:250-279: `chanLN_gamma_contract` / `chanLN_beta_contract`

**Smell:** undocumented-defeq, repetition · **Effort:** small
**Current (both, differing only in which argument varies):**
```lean
  rw [show (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x)
        = (fun γ' : Vec c => fun j : Fin (c * h * w) =>
            (fun v : Vec c => rowLNVecFlat (h * w) c ε v β (chanLNRows c h w x)) γ'
              (chanRowsPerm c h w j)) from rfl,
      pdiv_reindexOut_contract _ γ (…) (chanRowsPerm c h w) k cot]
  rfl
```
**Why it breaks:** the `from rfl` relies on `reassocBack ∘ transposeFlat ∘ … ∘ reassocFwd`
reducing to a single `chanRowsPerm` reindex. That spans four wrappers and two `finProdFinEquiv`
round-trips, and it is stated twice. The closing `rfl` is a second defeq,
`cot (σ.symm o) ≡ chanLNRows c h w cot o`. If either definition is refactored, both lemmas fail
with nothing pointing at the cause.
**Suggested:** name the two defeqs once, with docstrings:
`chanLNTensor3_apply : chanLNTensor3 c h w ε γ β x j = rowLNVecFlat (h*w) c ε γ β (chanLNRows c h w x) (chanRowsPerm c h w j) := rfl`
and `chanLNRows_apply : chanLNRows c h w v o = v ((chanRowsPerm c h w).symm o) := rfl`. Rewrite
both contracts with them. Also :230 has an unscoped terminal `simp [pdiv_reindex …]` (low).

### 10. Nets/Small/ChapterGraphTies.lean:215-224: `maxPoolFlatHasVJPAt'` belongs in CNN.lean

**Smell:** repetition · **Effort:** medium
**Current:** a raw-point max-pool VJP defined in a leaf. Its transport is re-derived inline in 7
places, `CNN.lean:1630-1635` and `CifarCNN.lean:153, 175, 605, 626, 647, 668`:
```lean
  have hpt1 : Tensor3.flatten (Tensor3.unflatten zmp1 : Tensor3 c1 …) = zmp1 :=
    Tensor3.flatten_unflatten zmp1
  have mp1_v : HasVJPAt (maxPoolFlat c1 (2*h) (2*w)) zmp1 := by
    rw [← hpt1]; exact maxPoolFlatHasVJPAt _ h_mp1
  have mp1_d : … := by rw [← hpt1]; exact maxPoolFlat_differentiableAt _ h_mp1 hc1 (by omega) (by omega)
```
**Why it breaks:** there are seven copies of a `rw [← hpt]` transport of a data-carrying witness.
Precedent for the fix exists: the 3×3/s2 pool already has `maxPool3s2FlatHasVJPAtVec` /
`…_differentiableAt_vec` (`HeadLayers.lean:73-74`).
**Suggested:** add `maxPoolFlatHasVJPAtVec` (this def) and `maxPoolFlat_differentiableAt_vec`
to CNN.lean and use them at the 7 sites. ⚠ `.backward` spelling changes at those sites, and graph
ties rfl-match backward spelling (memory: IR-spelled backwards). Run every consumer of
`mnistCnnNoBnHasVJPAt` and the CIFAR `HasVJPAt`s. After the move, `cnnBackGraph_faithful`'s
`backward_unique` detour may become a plain `rfl`.

### 11. The `mul_assoc` cast spelled out rather than `la_assoc`

**Smell:** repetition · **Effort:** small (breadth), low risk
**Current:** `congrArg (N * ·) (Nat.mul_assoc oc h w)` appears 19 times in
`Foundation/BatchedBackLinks.lean` (e.g. :188-201, :335, :352, :355-364, :383, :480), twice in
`Foundation/Batched.lean:63-65`, 5 times in `Training/BatchSealKit.lean` and twice in
`BatchMapVJPAt.lean`. `Foundation/IndexCast.lean:28` names it `la_assoc`.
**Why it breaks:** a restatement drifts (`(N * ·)` vs `congrArg (HMul.hMul N)`) and stops matching
a `rw` pattern. It is also 60 characters of noise per occurrence in statements that consumers
read.
**Suggested:** move `la_assoc` down into `Batched.lean`, which only imports PerChannelBN, and
replace the spelled-out casts. Proof arguments are defeq by proof irrelevance, so keyed `rw`
patterns in consumers still unify. Batch this with the next Batched-root rebuild.

### 12. Foundation/BatchedStageLayers.lean: smoothness hypothesis spelled 20×

**Smell:** repetition · **Effort:** small
**Current:** `∀ k, bnBatchLA N oc h w ε γ β (batchMap N (op) x) k ≠ 0 ∧ … ≠ 6` appears at
:61-62, :74-75, :83-84, :92-93, :101-102, :110-111, :122-123, :131-132, :155-156, :176-177,
:202-203, :222-223, :234-235, :247-248, and the `≠ 0` relu form appears 6 more times.
**Suggested:** `def BnRelu6Smooth N op ε γ β x : Prop` and `BnReluSmooth`, mirroring
`R34PoolSmoothAt` (`HeadLayers.lean:61`). Hypotheses are passed positionally, so consumers keep
working by defeq. A related pattern: the four relu/relu6 `*BackBatchedGraph_faithful` proofs
(:152-163, :173-184, :199-210, :337-347) are one proof with the names changed. A generic
`bnActStage_faithful`, taking the op's back-graph faithfulness as a hypothesis, would retire them.
That is medium effort.

### 13. Foundation/BatchedBackLinks.lean:84-173: possibly dead simp-set entries

**Smell:** fragile-simpa · **Effort:** trivial to test
`convBackBatched_faithful` and `depthwiseBackBatched_faithful` add
`flatConvHasVJP, HasVJP3.toHasVJP, conv2dHasVJP3` (respectively the depthwise ones) to the
simp set that their strided siblings (:117, :140, :155) close without, before the same closing
`rfl`. Those entries are probably dead, because the kernel-side `rfl` does the unfolding. Strip
them and compile, as §1(m) found 9 dead peels. The explanatory comment is also pasted 3×.

### 14. Float/RndP.lean:49-56: `hs_eq` zpow rewrite chain

**Smell:** brittle-chain · **Effort:** small (needs one compile)
**Current:**
```lean
      rw [hs, show (1 / 2 : ℝ) = (2 : ℝ) ^ (-1 : ℤ) by norm_num,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
          ← zpow_natCast (2 : ℝ) (p + 1), ← zpow_neg,
          ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      congr 1
      push_cast
      ring
```
**Why it breaks:** this is a 5-step chain through `zpow_add₀`/`zpow_neg`/`zpow_natCast`, and the
order of each rewrite depends on how the previous one left the goal.
**Suggested:** `rw [hs, zpow_sub₀ two_ne_zero, zpow_natCast]; field_simp; ring`.

### 15. Foundation/IntervalBoundConvQ.lean: twin cast proofs, unscoped simps

**Smell:** repetition, fragile-simpa · **Effort:** small, low payoff
`convLoQ_cast` and `convHiQ_cast` (:89-107) are the same proof, and so are `denseTLoQ_cast` and
`denseTHiQ_cast` (:144-160). Each ends in `by_cases … <;> simp [h1, …]`, and together with
:73, :116 and :130 that makes 9 unscoped terminal simps. The cheap part is `hbox`/`hboxm`
(:206-209): `funext a c d; simp [castT]` → `simp only [castT, Rat.cast_add]` (resp.
`Rat.cast_sub`). Leave the rest unless the file is reopened. The generated scorecards depend on
it, and it is fast.

### 16. Float/FloatClose.lean:80

**Smell:** fragile-simpa · **Effort:** trivial
`| zero => simpa using floatClose_id A` becomes `exact floatClose_id A`, because
`Function.iterate_zero` is `rfl`.

---

## Reviewed and not reported

- `Nets/Small/ChapterGraphTies.lean:263` `cnnBackGraph_faithful` uses `simp only` over the
  whole-net definitions, followed by `rfl`. That is the pattern §0 warns against, but on the
  small MNIST-CNN it was already measured bump-free in §1(g). Leave it. Item 10 may make it
  simpler.
- `Certificates/DenseEuclid.lean:190` `nlinarith`: the step divides by `S > 0`, so it really is
  nonlinear. Keep it.
- `Architectures/Softmax.lean:110`: the `show` that restates the goal carries a comment. It could
  be `simp only [softmax_apply, ← hS_def]`, but the gain is small.
- `Training/BatchSealKit.lean:118` `row_batchMap` restates `batchSlice_batchMap` (the two are
  defeq). It is a vocabulary alias, so it is fine.

## Recurring patterns (fix the pattern, not the instance)

1. **A local re-derivation of a lemma that already exists nearby.** `crossEntropy_differentiable`
   (same file), `max_close` (imported), `abs_le_of_sq_le_sq'` (same file, two lines above),
   `Tensor3.unflatten_flatten`, `cdf_nonneg`, and the BN grad-input ↔ backward identity, which is
   derived three times. Before writing a `have` longer than three lines, grep for its conclusion.
   Items 1–5, 7.
2. **Twin proofs with the names swapped.** γ/β (ChannelLN), Lo/Hi (IBP-Q), x/y and h1/h2 (ConvIndex,
   BatchedBackLinks), relu/relu6/strided stage faithfulness (BatchedStageLayers). Each pair is
   one lemma over the thing that varies.
3. **Defeq across wrappers written inline with `show` or `rw [show … from rfl]`.** `den` of a
   node, `chanLNTensor3` as a permutation, `Proofs.dense`, `pdiv` as a coordinate fderiv. Name each
   defeq once as an `rfl` lemma with a docstring, and rewrite with it. A refactor then breaks one
   named lemma rather than many anonymous `show`s. Items 1, 3, 6, 8, 9.
