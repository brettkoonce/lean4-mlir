# Proof-quality audit — `Proofs/Architectures/*.lean`, `Proofs/Training/*.lean` (hand-written)

Scope: 18 hand-written files (TrainedCnnSeal / TrainedCnnWitness / TrainedLinearDescent skipped as
generated). Static reading + grep only; nothing compiled. Toolchain v4.34.0; every Mathlib name
suggested below was grepped in `.lake/packages/mathlib/Mathlib`.

Headline numbers behind the findings:

| signal | count | where |
|---|---|---|
| full loss closure `crossEntropy … (fun v' => …)` written out in statements | 154 / 96 / 32 / 16 | SgdDescentCnn / Mlp / Linear / Cifar |
| `unflatten_flatten` restatements of margin hyps (`hm2'`, `hm3'` …) | 27 | SgdDescent*.lean |
| hand `mul_le_mul_of_nonneg_*` / `mul_nonneg` terms | 57 / 55 | SgdDescent*.lean (Cnn 38 / 38) |
| `gcongr` uses | 0 | SgdDescent*.lean |
| `Finset.abs_sum_le_sum_abs` nests | 28 | SgdDescentCnn (lemma `abs_triple_sum_sub_le` exists at :3349, used by few) |
| `M.dense_close … |>.trans (M.denseErr_le_uniform …)` pairs | 16 | SgdDescentCnn 11, Mlp 5 |
| `simpa [hf] using …` (unrestricted simp on the full statement) | 14 | Linear 2, Mlp 4, Cnn 8 |
| `open … Classical` at file scope | 6 | Attention, BatchNorm, CNN, LayerNorm, DepthwiseBackCertifiedTie, SgdDescentCnn |
| `set_option maxHeartbeats` | 1 (1 000 000, 5×) | SgdDescentCifar.lean:84 |

Theorem shape in SgdDescentCnn (statement vs. proof lines): `cnn_conv1_float_sgd_descends`
185 / 85, `cnn_conv2_float_sgd_descends` 151 / 60, `cnn_conv2_bias_float_sgd_descends` 128 / 81,
`cnn_conv2_sgd_descends` 116 / 88, `cnn_conv1_sgd_descends` 106 / 87. **The statements are longer
than the proofs.** This is the main compile-time lead for the 95 s file (see finding 1).

---

## LeanMlir/Proofs/Training/SgdDescentCnn.lean (95 s, worst file in scope)

### SgdDescentCnn.lean:2899 — `cnn_conv2_sgd_descends` (and every `*_sgd_descends` / `*_float_sgd_descends` rung: :3103, :5032, :5225, :5794, :6305, :6680, :7162; SgdDescentMlp.lean:409, :712, :1062, :1452; SgdDescentLinear.lean; SgdDescentCifar.lean:95)

**Smell:** repetition (compile time)
**Current:** the loss-of-parameter closure and the step radius are spelled out in full in every hypothesis:
```lean
    (hm2 : ∀ k, a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η)) < …)
    (hmq …same closure…) (hm3 …same…) (hm4 …same…) (hsmall …same…)
    (h1 …same closure twice…) (h2 …same closure three times…) :
    crossEntropy … ≤ … - lr * (∑ idx, gradAt (fun v' => …same…) …) / 2 := by
  set f : Vec (c * c * kH * kW) → ℝ := fun v' => … with hf
  …
  have hm2' : ∀ k, … < |Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₂)) b₂ x₁) k| :=
    fun k => by rw [Kernel4.unflatten_flatten]; exact hm2 k
  -- hmq', hm3', hm4' identical
```
The closure appears 154 times in this file (96 in Mlp, 32 in Linear, 16 in Cifar). Each copy is a
dependent-typed term over `Vec (c * (2*h) * (2*w))` that the elaborator must re-elaborate and
unify; the proof then immediately `set`s it back to `f`. The 27 `hmX'` restatements exist only
because the conclusion is stated at `Kernel4.unflatten (Kernel4.flatten W₂)` while the
hypotheses are at `W₂`.
**Why it breaks:** any change to the forward (an extra `relu`, a rename of `maxPoolFlat`) must be
made at 150+ sites per file, and elaboration cost scales with it. This is the most likely
single contributor to the 95 s build (statement elaboration + `linarith`/`simpa` over terms of
this size), ahead of any individual tactic.
**Suggested:** name the objects once, next to the rung:
```lean
/-- The conv-2-kernel loss map. -/
noncomputable def cnnConv2KernelLoss (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w)) (W₃ b₃ W₄ b₄ W₅ b₅)
    (label : Fin nC) : Vec (c * c * kH * kW) → ℝ :=
  fun v' => crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
    (dense W₃ b₃ (maxPoolFlat c h w (relu _ (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
    label

/-- The ℓ1 radius of an inexact SGD step (already the RHS of `sgd_step_l1_le`). -/
noncomputable def stepRadius {m} (f : Vec m → ℝ) (x : Vec m) (lr η : ℝ) : ℝ :=
  lr * ((∑ i, |gradAt f x i|) + m * η)
```
and state `hm2 : ∀ k, a * stepRadius L (flatten W₂) lr η < |…|`, conclusion
`L (flatten W₂ - lr • gh) ≤ L (flatten W₂) - lr * (∑ i, gradAt L (flatten W₂) i ^ 2) / 2`.
The `set f`, the four `hmX'` restatements and the closing `simpa [hf]` all disappear. Blast radius
is small: the only Lean consumers are SgdDescentCifar.lean and tests/AuditAxioms.lean (names
unchanged, so blueprint `\uses` / checkdecls are unaffected).

### SgdDescentCnn.lean:2536 — `Conv2Slot.loss_grad_lipschitz` (294 lines)

**Smell:** long-proof + brittle-chain
**Current:** the Lipschitz constant's sub-expression
`w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D))))))` is written
out 17 times inside the proof (hδ0, hden, hzdrift, hΔ0, hM0, the three-step `hfinal` calc, each
copy twice), and each non-negativity is a hand-nested term:
```lean
  have hδ0 : (0:ℝ) ≤ w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))))) :=
    mul_nonneg hw₅ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) hρD0)))))
  …
    have h3 : w₅ * (… (ρ * ∑ idx, |d idx|) …) ≤ w₅ * (… (ρ * D) …) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hd hρ) (Nat.cast_nonneg _)) hw₃)
          (Nat.cast_nonneg _)) hw₄) (Nat.cast_nonneg _)) hw₅
```
**Why it breaks:** the nesting mirrors the exact association of the product; reassociating the
constant, or a change to the implicit-argument signature of `mul_le_mul_of_nonneg_left` (these
order lemmas have been reshuffled in Mathlib's ordered-algebra refactors), breaks every copy. The 17 copies are also
elaboration cost.
**Suggested:** `set δ := w₅ * (… (ρ * D) …) with hδ` at the top, then `hδ0 := by positivity`
(Mathlib's `positivity` consults `0 ≤ atom` hypotheses via `compareHyp`, so `hw₅ hw₄ hw₃ hρ hD0`
suffice), `h3 := by gcongr` (the main goal `∑|d| ≤ D` closes by assumption `hd`, the side goals by
the positivity discharger), `hM0 := by positivity`. Split the endgame into a named lemma:
```lean
theorem Conv2Slot.frozen_term_le … : |J * (mask * (if argmax then head(v+td) else 0)) - J * (mask * (if argmax then head v else 0))|
  ≤ |J| * (d₃ * (w₃ * (d₄ * (w₄ * (nC * (w₅ * Δ))))))
```
(the per-term `by_cases hA` block at :2786–:2812).

### SgdDescentCnn.lean:2336, :2368, :2713, :2731, :3717 and SgdDescentMlp / SgdDescentLinear — monotonicity chains

**Smell:** brittle-chain (repetition: 57 `mul_le_mul_*`, 55 `mul_nonneg` terms; 0 `gcongr`)
**Current (:2368, `margin4_keeps_offkink`):**
```lean
  have h2 : w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * ∑ idx, |(t • e) idx|)))) ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (ρ * D)))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) hρ)
        (Nat.cast_nonneg _)) hw₃) (Nat.cast_nonneg _)) hw₄
```
**Why it breaks:** the term encodes the tree shape of the expression; any reassociation in the
statement (e.g. adopting the `stepRadius` def of finding 1) breaks each site.
**Suggested:** `by gcongr; exact smul_l1_mass_le e ht0 ht1 he` at every such site; the
non-negativity terms become `by positivity`. This is the most mechanical, highest-count cleanup
in the scope.

### SgdDescentCnn.lean:2776–2784, :797–:806, :3533–:3546, :3597–:3610, :3955–:3960, :4751–:4754 — triple `abs_sum_le_sum_abs`

**Smell:** repetition
**Current (:3597, `conv2d_input_l1_drift`):**
```lean
    calc |∑ c, ∑ kh, ∑ kw, W o c kh kw * (…)|
        ≤ ∑ c, |∑ kh, ∑ kw, …| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c, ∑ kh, |∑ kw, …| := Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c, ∑ kh, ∑ kw, |…| :=
          Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ => Finset.abs_sum_le_sum_abs _ _
```
The same three-level nest (with the big summand restated at each step) occurs at six sites; the
file already proves `abs_triple_sum_sub_le` (:3349), but only after the first four sites need it
(`loss_grad_lipschitz` at :2776 inlines exactly that lemma's proof).
**Suggested:** move `abs_triple_sum_sub_le` above `Conv2Slot` and add its un-subtracted peer
```lean
theorem abs_triple_sum_le {α β γ} [Fintype α] [Fintype β] [Fintype γ] (f : α → β → γ → ℝ) :
    |∑ a, ∑ b, ∑ c, f a b c| ≤ ∑ a, ∑ b, ∑ c, |f a b c|
```
Each calc then collapses to `(abs_triple_sum_le _).trans (by gcongr with c _ kh _ kw _; …)`.

### SgdDescentCnn.lean:3089 (+ :3087, :5210/:5212, :5928/:5930, :6441/:6443; SgdDescentMlp.lean:483/485, :806/808; SgdDescentLinear.lean:251/253)

**Smell:** fragile-simpa (compile time)
**Current:**
```lean
  have hmain := sgd_descends f (Kernel4.flatten W₂) gh hlr hη hC0 hgh (…) (fun t ht idx => by
      have h := cnn_conv2_loss_grad_lipschitz … ; simpa [hf] using h) h1 h2
  simpa [hf] using hmain
```
**Why it breaks:** the full default simp set runs over both sides of a statement that is ~100
lines of `crossEntropy/dense/relu/maxPoolFlat/conv2d` terms, only to β-reduce `f`. It is
expensive, and a new `@[simp]` lemma about `dense`/`relu`/`smul` in Foundation (a root file) can
rewrite the two sides differently and turn this red.
**Suggested:** `exact hmain` (the goal is `hmain`'s type up to β and `set`'s let-unfolding) or
`simpa only [hf] using hmain`. With finding 1 applied the `set`/`simpa` pair disappears.

### SgdDescentCnn.lean:3261, :5424, :6814, :7317; SgdDescentMlp.lean:1165, :1638 — `hη0` budget non-negativity

**Smell:** repetition + brittle-chain
**Current (:5424):**
```lean
  have hη0 : 0 ≤ M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ … eexp := by
    simp only [FloatModel.cnnConv1GradBudget]
    have hγ : ∀ m : ℕ, (0:ℝ) ≤ (1 + M.u) ^ (m + 1) - 1 := …
    have hebacknn : (0:ℝ) ≤ … 12-line restatement of a sub-budget … :=
      add_nonneg (mul_nonneg (hγ _) (mul_nonneg (hn _) (mul_nonneg hw₂ (add_nonneg hCPnn he2nn)))) …
    exact add_nonneg (mul_nonneg (hγ _) (mul_nonneg (hn _) (mul_nonneg ha (add_nonneg …)))) …
```
Only `cnnConv2CotBudget_nonneg` (:1719) exists as a lemma; the four GradBudgets have their
non-negativity proved inline at the (single) use site, each against the unfolded shape of the def.
**Why it breaks:** editing a budget def's association breaks the inline term; the proof lives far
from the def.
**Suggested:** four lemmas beside the defs (:1659, :4659, :6574, :6974), e.g.
```lean
theorem FloatModel.cnnConv2GradBudget_nonneg (M : FloatModel) … (ha : 0 ≤ a) (hw₂ …) … :
    0 ≤ M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
  have hγ := sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]) : 1 ≤ (1 + M.u) ^ _)
  have := M.cnnConv2CotBudget_nonneg …; have := FloatModel.cnnConv2CotMag_nonneg …
  unfold FloatModel.cnnConv2GradBudget; positivity
```
(`positivity` picks up `hγ` and the two budget facts by `compareHyp`).

### SgdDescentCnn.lean:1881–:1893, :742–:769 (`FloatModel.cnn_float_close`); SgdDescentMlp.lean:1333–:1363

**Smell:** repetition
**Current (:1881):**
```lean
  have hE3close : ∀ l, |Z3F l - Z3 l| ≤ E3 := fun l =>
    (M.dense_close W₃ b₃ PF PR E2 E2nn hPool l).trans
      (M.denseErr_le_uniform hw₃ E2nn hW₃ hb₃ hMpool l)
  have hRelu3 := fun l => relu_close _ _ _ hE3close l
  have hE4close : ∀ q, |Z4F q - Z4 q| ≤ E4 := fun q =>
    (M.dense_close W₄ b₄ (relu d₃ Z3F) (relu d₃ Z3) E3 E3nn hRelu3 q).trans
      (M.denseErr_le_uniform hw₄ E3nn hW₄ hb₄ hM3 q)
  …
```
16 `dense_close … .trans (denseErr_le_uniform …)` pairs; the dense-head forward-close chain is
proved twice in this file alone (`cnn_float_close` and `cnn_conv2_cot_close`).
**Suggested:** one leaf lemma (SgdDescentMlp.lean or a new Float leaf; **not** Foundation/Tensor):
```lean
theorem FloatModel.dense_close_layer (M : FloatModel) {m n} (W : Mat m n) (b : Vec n) {xt xa : Vec m}
    {w β A E : ℝ} (hw : 0 ≤ w) (hE : 0 ≤ E) (hW : ∀ i j, |W i j| ≤ w) (hb : ∀ j, |b j| ≤ β)
    (hA : ∀ i, |xa i| ≤ A) (hx : ∀ i, |xt i - xa i| ≤ E) :
    ∀ j, |M.dense W b xt j - dense W b xa j| ≤ FloatModel.layerBudget M.u m w β A E
```
and a `relu ∘ dense` corollary carrying the magnitude `layerAct` alongside.

### SgdDescentCnn.lean:1207 & :1271 — `mnist_cnn_convW_step_float_budget` / `…_convb_…`

**Smell:** repetition
**Current:** both proofs contain the identical Higham block:
```lean
  have hk1 : ((28 * 28 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((28 * 28 + 1 : ℕ) : ℝ) * u32 / (1 - ((28 * 28 + 1 : ℕ) : ℝ) * u32) ≤ 47/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (28 * 28 + 1) - 1 ≤ 47/1000000 := M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (28 * 28 + 1) - 1 := sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
```
**Why it breaks:** two copies of three `norm_num [u32]` evaluations at a concrete numeral; a change
of `u32`'s definition must be tracked twice.
**Suggested:** `theorem FloatModel.gamma785_le (hMu : M.u ≤ u32) : 0 ≤ (1 + M.u) ^ 785 - 1 ∧ (1 + M.u) ^ 785 - 1 ≤ 47/1000000`
once, used by both.

### SgdDescentCnn.lean:62 (and Attention.lean:55, BatchNorm.lean:41, CNN.lean:39, LayerNorm.lean:49, DepthwiseBackCertifiedTie.lean:27)

**Smell:** undocumented-defeq (instance-level)
**Current:** `open StableHLO Classical` / `open Finset BigOperators Classical` at file scope.
**Why it breaks:** every `if p then …` elaborated in these files may pick
`Classical.propDecidable` instead of the real `Decidable` instance (`Fin.decEq`, `Nat.decLt`,
`Real.decidableLT`). Lemmas like `ite_eq_left`, `Finset.sum_ite_eq'`, `if_congr` are then applied
across mismatched instances; that works today by unification luck and fails with "motive is not
type correct" / `Decidable` instance mismatch when a def moves to a file without `open Classical`.
Mathlib's own style is `open scoped Classical` (or per-declaration `open Classical in`) for exactly
this reason.
`open BigOperators` is a no-op in current Mathlib (the `∑` notation is global).
**Suggested:** `open scoped Classical in` on the specific declarations that need a classical `if`
(e.g. the `MaxPool2IsArgmax` indicators), or a `DecidablePred` instance for the Prop; drop
`BigOperators`.

### SgdDescentCnn.lean:3314–:3346 — `sum_swap_12_3`, `sum_swap_pair_pair` (and `sum_window_cells` :156)

**Smell:** brittle-chain (Finset.sum shuffling)
**Current:** hand `calc` chains of `Finset.sum_congr rfl fun _ _ => Finset.sum_comm`; the third
lemma `sum_swap_triple_triple` (:3339) already shows the robust idiom:
```lean
  calc _ = ∑ p : α × β × γ, ∑ q : δ × ε × ζ, … := by simp only [Fintype.sum_prod_type]
    _ = ∑ q, ∑ p, … := Finset.sum_comm
    _ = _ := by simp only [Fintype.sum_prod_type]
```
**Suggested:** give the first two the same three-line product-type proof (or
`Finset.sum_comm' `-free `simp only [Fintype.sum_prod_type]; exact Finset.sum_comm`).
`sum_window_cells` (46 lines) is `Fintype.sum_equiv` along
`(Equiv.prodProdProdComm _ _ _ _).trans ((winRowEquiv h).prodCongr (winColEquiv w))` after
`simp only [← Fintype.sum_prod_type']`.

---

## LeanMlir/Proofs/Training/SgdDescentCifar.lean

### SgdDescentCifar.lean:84 — `cifar8_lastConv_sgd_descends`

**Smell:** heartbeats (5×: `maxHeartbeats 1000000`)
**Current:**
```lean
set_option maxHeartbeats 1000000 in
theorem cifar8_lastConv_sgd_descends … := by
  have hfac : ∀ v', crossEntropy nClasses (cifarCnn8Forward … (Kernel4.unflatten v') … image) label
        = crossEntropy nClasses (dense Wb bb (… (conv2d (Kernel4.unflatten v') b₈ x₁) …)) label := by
    intro v'
    rw [cifarCnn8Forward_factor]
    simp only [Function.comp_apply, cifar8Head, flatConv, hx₁]
  rw [hfac (Kernel4.flatten W₈ - lr • gh), hfac (Kernel4.flatten W₈)]
  exact cnn_conv2_sgd_descends W₈ b₈ x₁ W₉ b₉ Wa ba Wb bb label gh hc4 hh hw … h1 h2
```
**Why it breaks:** the proof is three steps; the budget is spent elsewhere. Candidates: elaborating
the 16 copies of the conv2 closure in the statement against `Vec (c4 * c4 * kH * kW)`; the
`simp only [… hx₁]` rewriting `x₁` to `Tensor3.unflatten (cifar8Prefix7 … image)` (whose
indices are `2*(2*(2*(2*h)))`) and closing by `rfl` on a Tensor3 of that shape; the two `rw [hfac …]`
motive checks over the whole goal. None of these is measured.
**Suggested:** bisect with Mathlib's `count_heartbeats in` (Mathlib/Util/CountHeartbeats.lean) on
(a) the statement alone (`theorem … := by sorry`), (b) `hfac`. Then: pull `hfac` out as a named
lemma `cifar8_loss_lastConv_eq` stated with `x₁` generalized (`(hx₁ : x₁ = …)` and `subst` rather
than `simp` rewriting the big side), and apply finding 1's `cnnConv2KernelLoss` so the statement
shrinks to ~20 lines. The heartbeat bump should then be removable.

---

## LeanMlir/Proofs/Architectures/CNN.lean + Depthwise.lean (+ Attention.lean patch embed)

### CNN.lean:266–:412 — `conv2dHasVJP3.correct` and Depthwise.lean:236–:370 — `depthwiseHasVJP3.correct`

**Smell:** repetition + long-proof + undocumented-defeq
**Current:** the two proofs are a near line-for-line copy (`diff` shows only the channel index and
the `c = ci ∧` conjunct differ). Each contains:
```lean
    rw [show (let pH := (kH - 1) / 2 … if hpad : … then W co ci ⟨kh_nat, _⟩ ⟨kw_nat, _⟩ * dy co ho wo else 0) =
            (let pH := … if hpad : … then W co ci ⟨kh_nat, _⟩ ⟨kw_nat, _⟩ else 0) * dy co ho wo from by
      by_cases hb : … · simp only [dite_eq_left hb] · simp only [dite_eq_right hb, zero_mul]]
    …
    have h_indicator : ∀ c kh kw, (… if idx_in = finProdFinEquiv (finProdFinEquiv (c, ⟨hh - pH, _⟩), ⟨ww - pW, _⟩) then 1 else 0 …) = … := by
      … have h_inj := finProdFinEquiv.injective h_eq
          have h_inj_pair := Prod.mk.inj h_inj
          have h_inj_inner := finProdFinEquiv.injective h_inj_pair.1 …   -- ~55 lines
    …
      rw [Finset.sum_eq_single ci ?_ ?_]
      rw [Finset.sum_eq_single ⟨hi.val + (kH - 1) / 2 - ho.val, hb.2.1⟩ ?_ ?_]
      rw [Finset.sum_eq_single ⟨wi.val + (kW - 1) / 2 - wo.val, hb.2.2.2⟩ ?_ ?_]
      … · intro hni; exact absurd (Finset.mem_univ _) hni    -- ×3
      … show kw.val = wi.val + (kW - 1) / 2 - wo.val; omega  -- ×6 `show … ; omega`
```
~145 lines each, 290 total.
**Why it breaks:** the 25-line `show (let …) = …` restates the formula's `let`-body verbatim, so
any edit to `conv2dInputGradFormula` (a renamed binder, a different padding expression) must
be mirrored in 2 files × 2 places. `Finset.sum_eq_single` + `absurd (Finset.mem_univ _)` is the
pre-`Fintype` idiom.
**Suggested:** the repo already has the short version in Attention.lean:2123–:2140
(patch embed):
```lean
        refine if_congr ?_ rfl rfl
        simp only [EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq, Fin.ext_iff, @eq_comm _ c_in, …, and_assoc]
      …
    simp only [ite_and, mul_ite, mul_one, mul_zero, Finset.sum_ite_irrel, Finset.sum_const_zero,
      Finset.sum_ite_eq', Finset.mem_univ, ite_true, …, dite_eq_ite]
```
Use `dite_mul` (Mathlib/Algebra/Notation/Defs.lean:115) instead of the `show (let …)` restatement,
`Fintype.sum_eq_single` (Mathlib/Data/Fintype/BigOperators.lean, additive of `prod_eq_single`) for
the collapses, and extract the shared 1-D fact once into CNN.lean (leaf, not Tensor.lean):
```lean
theorem padTap_indicator {h kH : Nat} (hi ho : Fin h) (kh : Fin kH) (P : Fin h → ℝ) :
    (if hpad : (kH-1)/2 ≤ kh.val + ho.val ∧ kh.val + ho.val - (kH-1)/2 < h
       then P ⟨kh.val + ho.val - (kH-1)/2, hpad.2⟩ else 0)
      = if kh.val + ho.val = hi.val + (kH-1)/2 then P hi else 0   -- (with P := indicator at hi)
```
Both `hasVJP3` proofs then become rows × cols applications of it.

---

## LeanMlir/Proofs/Architectures/BatchNorm.lean

### BatchNorm.lean:560 — `pdiv_bnIstdBroadcast` (174 lines)

**Smell:** long-proof + undocumented-defeq
**Current:**
```lean
  have hC_apply : ∀ k y, C k y = y k - bnMean (n' + 1) y := by
    intros k y
    show ((ContinuousLinearMap.proj k : Vec (n' + 1) →L[ℝ] ℝ) - mean_clm) y = _
    rw [sub_apply]
    show y k - mean_clm y = _
    show y k - (((n' + 1 : Nat) : ℝ)⁻¹ • ∑ i', (ContinuousLinearMap.proj i' : …)) y = _
    rw [smul_apply, _root_.sum_apply, smul_eq_mul]
    show y k - ((n' + 1 : Nat) : ℝ)⁻¹ * ∑ i', y i' = _
  …
  rw [show bnIstd (n' + 1) x ε = 1 / Real.sqrt (bnVar (n' + 1) x + ε) from rfl]
  …
    rw [show (∑ k, 2 * (x k - μ) * ((if k = i then 1 else 0) - N⁻¹)) = … from by …]   -- 4 such `rw [show … from by …]`
        · intro h; exact absurd (Finset.mem_univ i) h
```
Nine `show`s, three `… from rfl`/`rfl` closes, three `Finset.sum_eq_single` + `absurd` blocks.
**Why it breaks:** each `show` depends on how `ContinuousLinearMap.sub/smul/sum` application
unfolds definitionally — Mathlib has restructured the `→L` coercion stack several times; `simp`
lemmas (`sub_apply`, `smul_apply`, `coe_sum'`) are the stable API.
**Suggested:** split into two named lemmas —
`bnVar_hasFDerivAt : HasFDerivAt (bnVar n) ((2 / n) • ∑ k, (x k - bnMean n x) • C k) x` and
`sum_sub_bnMean : ∑ k, (x k - bnMean n x) = 0` — and replace the `show` chains with
`simp [C, mean_clm, bnMean, div_eq_inv_mul]`; the Kronecker collapses are
`simp [basisVec_apply, Finset.sum_ite_eq']` (the idiom `pdiv_bnCentered` :513 already uses).
`hN_ne` is `Nat.cast_add_one_ne_zero n'`.

---

## LeanMlir/Proofs/Architectures/Attention.lean (15 s)

### Attention.lean:184 — `pdiv_softmax`; :301 — `softmaxCE_grad`

**Smell:** undocumented-defeq
**Current:**
```lean
    rw [fderiv_apply (softmax_differentiable (c' + 1) z) j]
    rfl
  …
    show Real.exp (z' j) / (∑ k, Real.exp (z' k)) = _          -- unfolds `softmax`
  …
  show Real.exp (z j) * (-(S ^ 2)⁻¹ * Real.exp (z i)) + S⁻¹ * (…) = (Real.exp (z j) / S) * (…)
  …
    show Differentiable ℝ (fun z => -(Real.log (softmax (c' + 1) z label)))   -- unfolds `crossEntropy`
    show HasFDerivAt (fun z => -(Real.log (softmax (c' + 1) z label))) _ logits
    show -((softmax … label)⁻¹ * (… ((if j = label then (1 : ℝ) else 0) - …))) = …   -- unfolds `oneHot`
```
Seven `show`s that unfold `softmax`, `crossEntropy` and `oneHot` (Foundation/MLP.lean:264–:272)
by defeq; none of the three defs has an `_apply`/`_def` lemma.
**Why it breaks:** a change to any of the three defs (e.g. `softmax` via `Real.exp z / ∑` →
`exp z * (∑)⁻¹`, or `oneHot` via `Pi.single`) silently breaks every `show`.
**Suggested:** add `softmax_apply`, `crossEntropy_def`, `oneHot_apply` (in MLP.lean if it is a leaf,
otherwise Attention.lean) and `rw` with them. Also the "pdiv entry = fderiv of the coordinate"
step (`rw [fderiv_apply …]; rfl`) recurs at Attention.lean:194, :329, :352 and BatchNorm.lean:581 —
one lemma:
```lean
theorem pdiv_eq_fderiv_coord {m n} (f : Vec m → Vec n) (x : Vec m) (hf : DifferentiableAt ℝ f x) (i j) :
    pdiv f x i j = fderiv ℝ (fun y => f y j) x (basisVec i)
```

### Attention.lean:1042 — `pdivMat_mhsaG_split`, :1087 `mhsaGHasVJPMat`, :1134–:1210 `mhsaQkvW/b` + six `@[simp]` lemmas

**Smell:** repetition
**Current:** the three-way `if c = 0 then … else if c = 1 then … else …` over `Fin 3` is written
four times (two defs, one backward, one theorem statement), and needs six `@[simp]` lemmas
(`mhsaQkvW_eq0/1/2`, `mhsaQkvB_eq0/1/2`), each proved by
```lean
  unfold mhsaQkvW
  simp [Equiv.symm_apply_apply, show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
        show (2 : Fin 3) ≠ (1 : Fin 3) from by decide]
```
**Why it breaks:** the `show … from by decide` arguments are redundant (Lean core's `Fin.reduceEq`
/ `Fin.reduceNe` simprocs, Lean/Meta/Tactic/Simp/BuiltinSimprocs/Fin.lean:104–106, decide literal
`Fin` equalities inside `simp`), and the if-chain forces the `by_cases hc0 / hc1 / fin_cases` dance
in every consumer.
**Suggested:** select with a vector literal: `![Wq, Wk, Wv] q.1 k (finProdFinEquiv (p.1, q.2))`.
`Matrix.cons_val_zero/one/two` (Mathlib/Data/Fin/VecNotation.lean:271–:274) are `simp` lemmas, so
the six `@[simp]` lemmas become one (or none), and consumers `fin_cases c <;> simp`.

### Attention.lean:567, :620, :640, :718, :741 — `sdpa_*_chain_eq`, `sdpa_back_{Q,K,V}_correct`

**Smell:** undocumented-defeq (compile time)
**Current:**
```lean
  rw [← (sdpaQChainHasVJP n d K V).correct Q dOut i j]
  unfold sdpaBackQ sdpaDScores sdpaDScaled sdpaDWeights sdpaWeights sdpaQChainHasVJP
  rfl
```
**Why it breaks:** the closing `rfl` asks the kernel to unfold three nested `vjpMatComp`
structures plus `rowSoftmaxHasVJPMat` down to the explicit formula — exactly the
"`rfl` forces whole-chain unfolding" hazard; a change to any `HasVJPMat` builder (even adding a
field) changes what has to be unfolded. Likely a large share of this file's 15 s.
**Suggested:** give `vjpMatComp` a `@[simp] theorem vjpMatComp_backward` (backward of the
composite = inner backward ∘ outer backward) and close with `simp only [vjpMatComp_backward, …]`
or `rw` steps; or define `sdpaBackQ := (sdpaQChainHasVJP n d K V).backward` and prove the
closed form as an `_apply` lemma instead of the other way round.

---

## LeanMlir/Proofs/Architectures/LayerNorm.lean

### LayerNorm.lean:149, :162 — `Real.differentiable_tanh`, `Real.hasDerivAt_tanh`

**Smell:** undocumented-defeq (name collision risk on a Mathlib bump)
**Current:** declared inside `namespace Proofs` as `theorem Real.hasDerivAt_tanh …`, i.e.
`Proofs.Real.hasDerivAt_tanh`; callers write `Real.hasDerivAt_tanh`. The docstring at :159 says
"Mathlib has `Real.differentiable_tanh`" — the pinned Mathlib has neither (grep of
`Analysis/SpecialFunctions` finds only `Real.logDeriv_cosh`).
**Why it breaks:** when Mathlib adds `Real.hasDerivAt_tanh` / `Real.differentiable_tanh` (both are
natural additions next to `Real.hasDerivAt_sinh` in Trigonometric/DerivHyp.lean), every
`Real.hasDerivAt_tanh` reference inside `namespace Proofs` becomes ambiguous, and the
`@[fun_prop]` attribute is registered twice.
**Suggested:** rename to `Proofs.hasDerivAt_tanh` / `Proofs.differentiable_tanh` (no `Real.`
prefix) and fix the docstring; delete when Mathlib gains them.

---

## LeanMlir/Proofs/Training/BatchSealKit.lean

### BatchSealKit.lean:933 — `head_diff_ct`

**Smell:** brittle-chain + undocumented-defeq
**Current:**
```lean
    rw [row_batchMap, row_batchMap]
    rfl                                   -- globalAvgPoolFlat ↔ globalAvgPool ∘ bcell by defeq
  …
  rw [Finset.sum_congr rfl (fun ci _ => show
      globalAvgPool (bcell v 0) ci * (if ci.val = 0 then (1 : ℝ) else 0)
        - globalAvgPool (bcell v 1) ci * (if ci.val = 0 then (1 : ℝ) else 0)
      = δ ci * (if ci.val = 0 then (1 : ℝ) else 0) from by rw [hgap ci]; ring)]
  refine (Finset.sum_eq_single_of_mem c₀ (Finset.mem_univ _) ?_).trans ?_
```
**Why it breaks:** the `rfl` crosses `globalAvgPoolFlat`/`bcell` wrappers; the restated summand
duplicates `dense`'s unfolding.
**Suggested:** a lemma `globalAvgPoolFlat_bcell : Mat.unflatten (batchMap … globalAvgPoolFlat v) n = globalAvgPool (bcell v n)`
for the `rfl`, and `simp only [dense, hWd, hbd, add_zero, ← Finset.sum_sub_distrib, hgap, ← sub_mul, add_sub_cancel_left]`
followed by `Fintype.sum_eq_single c₀` for the collapse.

### BatchSealKit.lean:887 — `ctConv_inj`

**Smell:** brittle-chain (minor)
**Current:**
```lean
  have hcomm : (2 * (2 * w)) * (2 * r.val) + 2 * s.val = (2 * (2 * w)) * (2 * r'.val) + 2 * s'.val := by
    rw [Nat.mul_comm (2 * (2 * w)) (2 * r.val), Nat.mul_comm (2 * (2 * w)) (2 * r'.val)]
    exact hnat
```
**Suggested:** `by linarith [hnat]` (linarith normalizes with `ring_nf`, so the non-linear
monomials match as atoms) — or `by ring_nf; ring_nf at hnat; exact hnat`.

---

## LeanMlir/Proofs/Architectures/DepthwiseBackCertifiedTie.lean

### DepthwiseBackCertifiedTie.lean:68 — `depthwiseStride2FlatXlaBack_eq_vjp_backward`

**Smell:** undocumented-defeq (minor)
**Current:**
```lean
  funext dy
  show depthwiseFlatBack (h := 2 * h) (w := 2 * w) W (decimateOddBack c h w dy) = _
  rw [depthwiseFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl
```
**Suggested:** `rw [depthwiseStride2FlatXlaBack, Function.comp_apply]` (or the def's equation
lemma) in place of the `show`, and a one-line comment for the closing `rfl` (it unfolds
`depthwiseStride2FlatXlaHasVJP`'s backward).

---

## Recurring patterns (fix the pattern, fix it everywhere)

1. **Unnamed objects in statements (≈300 sites).** The descent rungs spell out the
   loss-of-parameter closure (154 Cnn + 96 Mlp + 32 Linear + 16 Cifar), the ℓ1 step radius, and the
   Lipschitz constant in every hypothesis and every `have`. Two defs (`cnnConv2KernelLoss` &
   siblings, `stepRadius`) plus `set` for the constant inside proofs would remove the bulk of
   SgdDescentCnn's statement text, the 27 `unflatten_flatten` restatements, the 14
   `simpa [hf]` closes, and are the most plausible lever on the 95 s build and the Cifar
   heartbeat bump. Consumers are few (Cifar, AuditAxioms), so it is cheap to do.

2. **Hand monotonicity/positivity instead of `gcongr`/`positivity` (≈110 sites, 0 `gcongr`
   in SgdDescent*).** Nested `mul_le_mul_of_nonneg_left` (57) and `mul_nonneg` (55) terms, plus
   inline budget non-negativity (6 `hη0` blocks). All mechanically replaceable: `gcongr` for the
   inequalities, `positivity` for non-negativity (it reads `0 ≤ atom` hypotheses). This is the
   Mathlib-bump fragility hot spot of the Training files.

3. **Defeq `show`/`rfl` over the project's own defs, and copy-pasted index algebra.**
   `show` unfolding `softmax`/`crossEntropy`/`oneHot`/CLM application (Attention 7, BatchNorm 9),
   `unfold …; rfl` over composed VJP structures (Attention 5), and the
   `sum_eq_single + absurd mem_univ + finProdFinEquiv.injective` conv indicator proof duplicated
   in CNN.lean and Depthwise.lean (~290 lines; Attention's patch-embed version already shows the
   short `simp only [Prod.mk.injEq, Fin.ext_iff, ite_and, Finset.sum_ite_eq']` idiom). Add
   `_apply` lemmas for the defs, a `vjpMatComp_backward` simp lemma, and one `padTap_indicator`
   lemma.

Also worth one pass: file-scope `open Classical` in 6 files (→ `open scoped Classical in` per
declaration), and the triple `abs_sum_le_sum_abs` nest (6 sites; a lemma already exists at
SgdDescentCnn.lean:3349, declared after the sites that need it).
