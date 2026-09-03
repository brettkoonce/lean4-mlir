import LeanMlir.Proofs.Float.FloatBudgetEnvLN
import LeanMlir.Proofs.Float.ViTBlockFloatBridge

/-! # `FloatBridgesTo.Maps` leaves for the ATTENTION family (ViT)

`FloatBudgetEnv.lean` holds the kit and ResNet-34's leaves, `FloatBudgetEnvMBConv.lean` the
inverted-bottleneck family's, `FloatBudgetEnvLN.lean` the LayerNorm family's. This file holds
the two a transformer needs on top of those: the **row softmax** and **multi-head projected
attention**.

⛔ **Both exist because of one thing: `Real.exp` must never reach a stage numeral.**
`FloatModel.smErr u eexp δ n = u(1+κ) + κ + (Real.exp (2δ) − 1)` is EXPONENTIAL in the
inherited error `δ`, and at ViT-Tiny's magnitudes the perturbation reaching the attention
logits is ~10¹⁰ by block 0. `Real.exp` at that argument has no rational bound the kernel will
check — so the failure is not that the number is large, it is that **there is no number**.
`planning/float_budget_numbers.md` §3.5 measures this: 48 unwritable stage numerals at an
otherwise IDENTICAL magnitude, which is a different failure mode from §0.1's quadratic blow-up
and the first of its kind in this repo.

The two leaves dispose of it in the same way, and neither is a concession:

* ⭐ **`Maps.softmaxRow`** — a softmax output is a probability, so `softmax_abs_le_one` puts the
  real row in `[0,1]` and `softmaxF_close` puts the float row within `smCap = u(1+κ) + κ` of it.
  The window is therefore `1 + smCap` **at every input window**, a rational with no `Real.exp`
  in it, and `FloatBridgesTo.capped` turns that into the modulus `2(1 + smCap)`. The `Real.exp`
  is DELETED rather than bounded. Elsewhere the cap is a weakening (§9); here it is simply the
  right bound, because the underlying `smErr` modulus is not merely loose — it is unstatable.

* ⭐⭐ **`Maps.mhProjAttnFullCap`** — and this one is `floatClose_seScale`'s bug again, one net
  later. `mhpB` (`ViTBlockFloatBridge.lean`) derives attention's WINDOW as
  `|float − real| + |real|`, i.e. `vA_F + attnOutErr`, and `attnOutErr` carries `smErr` — so the
  `Real.exp` is in the **window**, where `capped` cannot reach it (`capped` bounds the modulus
  by `2·mag`; it does not touch `mag`). ⛔ That is why the monolithic bridge
  `floatBridges_mhProjAttnFull` CANNOT carry this number, and no amount of capping fixes it.
  But `FloatClose`'s magnitude clause bounds the FLOAT output directly, and the float output is
  a rounded dot of float softmax weights against float `V`: `|weightsF| ≤ 1 + smCap` and
  `|V_F| ≤ vA_F` give `(1 + γ_{n+1})·n·(1 + smCap)·vA_F`, exp-free. §3.4's lesson verbatim:
  *the error never needed to enter the window at all.*

⚠ The `n` factor in that window is the generic fan-in bound, and `sdpa_abs_le` already proves
the real side is a CONVEX combination (window `vA`, no `n`). The float peer — rounded weights
are still nonnegative and sum to `≤ 1 + smCap` — is NOT proved here, deliberately: it is worth
27 orders on ViT-Tiny and the fold is statable without it (`vit_chain(convex_av=False)` in
`scripts/float_budget_envelope.py`), and §1's non-goal is making the numbers small. If a deeper
transformer ever needs it, that is the lemma to write.

⚠ Why this file and not `FloatBudgetEnv.lean`: a `Maps` lemma names its bridge, and these name
`FloatModel.softmaxF` / `mhProjAttnFullFlat`, neither of which is on that file's import path.
Same reason `FloatBudgetEnvMBConv` and `FloatBudgetEnvLN` are separate. ⚠ It imports
`FloatBudgetEnvLN` (not the bare `FloatBudgetEnv`) for `Maps.perRow`, which the out-projection
needs and which every ViT budget needs anyway — so this file sits at the top of the leaf stack
for a transformer, and inherits `FloatBudgetEnvLN`'s own documented coupling to the MBConv kit.

⭐ The two `example`s at the bottom are not decoration: they close ViT-Tiny's block-0 attention
site and its softmax leaf at the numerals `vit_chain` emits, which is simultaneously the check
that the generator's arithmetic IS these lemmas' (§5's rule — an unexercised `Maps` leaf is the
stale-gates failure mode in proof form).
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The softmax head constant, and the float row's own bound
-- ════════════════════════════════════════════════════════════════

/-- **`κ'` — the float softmax's absolute distance from the real one at the SAME logits.**
    `smErr` at zero logit perturbation, i.e. `smErr u eexp 0 n` with the `Real.exp (2·0) − 1`
    term dropped. Named because it is the whole content of the softmax leaf's window, and
    because keeping it out of `smErr`'s shape is what keeps `Real.exp` out of the numerals. -/
noncomputable def smCap (u eexp : ℝ) (n : ℕ) : ℝ :=
  u * (1 + smKappa u eexp n) + smKappa u eexp n

namespace FloatModel

variable (M : FloatModel)

/-- `0 ≤ smRho` — the public peer of `FloatBridge.lean`'s `private smRho_nonneg`. -/
theorem smRho_nonneg' {eexp : ℝ} {n : ℕ} (heexp : 0 ≤ eexp) : 0 ≤ smRho M.u eexp n :=
  add_nonneg
    (mul_nonneg (sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))) (by linarith)) heexp

/-- `0 ≤ smKappa` under the standing side condition `smRho < 1`. -/
theorem smKappa_nonneg {eexp : ℝ} {n : ℕ} (heexp : 0 ≤ eexp)
    (hρ1 : smRho M.u eexp n < 1) : 0 ≤ smKappa M.u eexp n :=
  div_nonneg (by linarith [M.smRho_nonneg' (eexp := eexp) (n := n) heexp]) (by linarith)

/-- `0 ≤ smCap`. -/
theorem smCap_nonneg {eexp : ℝ} {n : ℕ} (heexp : 0 ≤ eexp)
    (hρ1 : smRho M.u eexp n < 1) : 0 ≤ smCap M.u eexp n := by
  have hκ := M.smKappa_nonneg heexp hρ1
  have hu := M.u_nonneg
  simp only [smCap]
  nlinarith

/-- **The float softmax row is bounded by `1 + smCap`, at ANY logits.** The real row is a
    probability (`softmax_abs_le_one`) and the float row is within `smCap` of it
    (`softmaxF_close`) — so the bound is a CONSTANT, independent of the logit window, and free
    of `Real.exp`. This is the fact the whole file is built on. -/
theorem softmaxF_abs_le (fexp : ℝ → ℝ) {eexp : ℝ} {n : ℕ} (z : Vec n)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1) (k : Fin n) :
    |M.softmaxF fexp z k| ≤ 1 + smCap M.u eexp n := by
  have hclose := M.softmaxF_close fexp z heexp0 heexp1 hfexp hρ1 k
  have hreal := softmax_abs_le_one z k
  have htri := abs_sub_abs_le_abs_sub (M.softmaxF fexp z k) (softmax n z k)
  simp only [smCap]
  linarith

end FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The row softmax as a leaf
-- ════════════════════════════════════════════════════════════════

/-- **The row softmax is `FloatClose` with a CONSTANT window `1 + smCap`.** The magnitude
    clause is `softmax_abs_le_one` on the real side and `softmaxF_abs_le` on the float side —
    neither depends on the input window `A`, which is exactly what makes the softmax a RESET
    in the fold. The error clause is `softmaxF_close_at` verbatim, so the modulus is still the
    honest `smErr` (with its `Real.exp`); it is `FloatBridgesTo.capped` downstream, not this
    lemma, that keeps the exponential out of the numerals. -/
theorem floatClose_softmaxRow (M : FloatModel) (fexp : ℝ → ℝ) {n : Nat} {eexp A : ℝ}
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatClose A (1 + smCap M.u eexp n) (softmax n) (M.softmaxF fexp)
      (fun e => smErr M.u eexp e n) := by
  have hcap := M.smCap_nonneg heexp0 hρ1
  refine ⟨fun v _hv k => ⟨?_, ?_⟩, fun vt va e _hva _hvt hd k => ?_⟩
  · have := softmax_abs_le_one v k; linarith
  · exact M.softmaxF_abs_le fexp v heexp0 heexp1 hfexp hρ1 k
  · exact M.softmaxF_close_at fexp vt va heexp0 heexp1 hfexp hρ1 hd k

/-- The row softmax float-bridges to `M.softmaxF`, with the window constant in the input. -/
noncomputable def floatBridgesTo_softmaxRow (M : FloatModel) (fexp : ℝ → ℝ) {n : Nat}
    {eexp : ℝ} (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridgesTo (softmax n) (M.softmaxF fexp) where
  mag := fun _ => 1 + smCap M.u eexp n
  mod := fun _ e => smErr M.u eexp e n
  close := fun _A _hA =>
    ⟨by have := M.smCap_nonneg heexp0 hρ1; linarith,
     floatClose_softmaxRow M fexp heexp0 heexp1 hfexp hρ1⟩

/-- **A rational bound on `smCap` from a rational bound on `smRho`.** `smKappa` is a quotient,
    so every ViT numeral needs this step: `smRho ≤ rb < 1` gives
    `smKappa ≤ (eexp + rb)/(1 − rb)`, and `smCap` follows by `M.u ≤ u32`. Stated with `u32` on
    the right so the closing inequality is `norm_num`-able. -/
theorem smCap_le (M : FloatModel) {eexp rb c : ℝ} {n : ℕ}
    (heexp0 : 0 ≤ eexp) (hMu : M.u ≤ u32)
    (hrho : smRho M.u eexp n ≤ rb) (hrb1 : rb < 1)
    (hc : u32 * (1 + (eexp + rb) / (1 - rb)) + (eexp + rb) / (1 - rb) ≤ c) :
    smCap M.u eexp n ≤ c := by
  have hu := M.u_nonneg
  have hu32 : (0:ℝ) ≤ u32 := by norm_num [u32]
  have hrho0 := M.smRho_nonneg' (eexp := eexp) (n := n) heexp0
  have hrho1 : smRho M.u eexp n < 1 := lt_of_le_of_lt hrho hrb1
  have hdenb : (0:ℝ) < 1 - rb := by linarith
  have hkap : smKappa M.u eexp n ≤ (eexp + rb) / (1 - rb) := by
    simp only [smKappa]
    exact div_le_div₀ (by linarith) (by linarith) hdenb (by linarith)
  have hkap0 : 0 ≤ smKappa M.u eexp n := M.smKappa_nonneg heexp0 hrho1
  have hkb0 : (0:ℝ) ≤ (eexp + rb) / (1 - rb) := le_trans hkap0 hkap
  simp only [smCap]
  nlinarith

/-- **A rational bound on `smRho`, from a `gamma_num` bound** — the softmax leaf's standing side
    condition `smRho < 1` reduced to one `norm_num`. At ViT-Tiny (`n = 197`, `eexp = 10⁻²`) it is
    `0.0100120 < 1`, satisfied with room — but it IS a hypothesis the whole-net statement must
    carry and disclose, like `DeviceRsqrt`/`DeviceSigmoid`. -/
theorem smRho_le_of (M : FloatModel) {eexp g rb : ℝ} {n : ℕ}
    (hg : (1 + M.u) ^ (n + 1) - 1 ≤ g) (heexp0 : 0 ≤ eexp)
    (hrb : g * (1 + eexp) + eexp ≤ rb) : smRho M.u eexp n ≤ rb := by
  simp only [smRho]
  nlinarith

namespace FloatBridgesTo

/-- **An envelope through the CAPPED row softmax.** ⭐ Note what it does not take: no input
    window bound is used at all, because the softmax's window is constant. The output error is
    `2·Ā'` by `Maps.capped`, and `smErr`'s `Real.exp` never becomes a numeral. -/
theorem Maps.softmaxRow (M : FloatModel) (fexp : ℝ → ℝ) {n : Nat} {eexp : ℝ}
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1)
    {Ā Ē Ā' Ē' : ℝ} (hĀ' : 1 + smCap M.u eexp n ≤ Ā') (hĒ' : 2 * Ā' ≤ Ē') :
    (floatBridgesTo_softmaxRow M fexp heexp0 heexp1 hfexp hρ1).capped.Maps Ā Ē Ā' Ē' :=
  Maps.capped (fun _A _h0 _hle => hĀ') hĒ'

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § Multi-head projected attention with an EXP-FREE window
-- ════════════════════════════════════════════════════════════════

namespace FloatModel

variable (M : FloatModel)

/-- **A rounded dot of two bounded vectors.** `|M.dot x y| ≤ (1+γ_{n+1})·n·X·Y` — the real sum
    bounded termwise, plus `dot_close`'s Higham γ on the same sum. The generic fan-in bound;
    it is what lets attention's output matmul be bounded WITHOUT reference to the error of the
    weights it multiplies. -/
theorem dot_abs_le_of {n : ℕ} (x y : Vec n) {X Y : ℝ}
    (hX0 : 0 ≤ X) (hX : ∀ i, |x i| ≤ X) (hY : ∀ i, |y i| ≤ Y) :
    |M.dot x y| ≤ (1 + ((1 + M.u) ^ (n + 1) - 1)) * ((n : ℝ) * (X * Y)) := by
  have hg0 : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  have hterm : ∀ i, |x i * y i| ≤ X * Y := by
    intro i; rw [abs_mul]; exact mul_le_mul (hX i) (hY i) (abs_nonneg _) hX0
  have hsum : ∑ i, |x i * y i| ≤ (n : ℝ) * (X * Y) := by
    calc ∑ i, |x i * y i| ≤ ∑ _i : Fin n, (X * Y) := Finset.sum_le_sum fun i _ => hterm i
      _ = (n : ℝ) * (X * Y) := by simp [Finset.sum_const, Finset.card_univ]
  have habs : |∑ i, x i * y i| ≤ ∑ i, |x i * y i| := Finset.abs_sum_le_sum_abs _ _
  have hc := M.dot_close x y
  have htri : |M.dot x y|
      ≤ |∑ i, x i * y i| + ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i * y i| := by
    have := abs_sub_abs_le_abs_sub (M.dot x y) (∑ i, x i * y i); linarith
  calc |M.dot x y|
      ≤ |∑ i, x i * y i| + ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i * y i| := htri
    _ ≤ ∑ i, |x i * y i| + ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i * y i| := by linarith
    _ = (1 + ((1 + M.u) ^ (n + 1) - 1)) * ∑ i, |x i * y i| := by ring
    _ ≤ (1 + ((1 + M.u) ^ (n + 1) - 1)) * ((n : ℝ) * (X * Y)) :=
        mul_le_mul_of_nonneg_left hsum (by linarith)

end FloatModel

/-- **The exp-free window for multi-head projected attention.** Compare `mhpB`
    (`ViTBlockFloatBridge.lean`), which is `vA_F + attnOutErr` — the float magnitude derived as
    `|real| + |float − real|`, dragging `attnOutErr`'s `smErr` and therefore `Real.exp` into the
    WINDOW. This is the direct float-side bound instead: the float attention output is a rounded
    dot of float softmax weights (`≤ 1 + smCap`, `softmaxF_abs_le`) against float `V`
    (`≤ layerAct + layerBudget 0`, `projF_abs_le`).

    ⭐ Strictly better AND rational: `floatClose_seScale`'s fix from `planning/
    float_budget_numbers.md` §3.4, one net later — *the gate's error never needed to enter the
    window at all.* -/
noncomputable def mhpBCap (M : FloatModel) (n h dh : Nat) (w' β A eexp : ℝ) : ℝ :=
  (1 + ((1 + M.u) ^ (n + 1) - 1))
    * ((n : ℝ) * ((1 + smCap M.u eexp n)
        * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0)))

/-- **Multi-head projected attention is `FloatClose` at the exp-free window `mhpBCap`.**
    The MODULUS is `mhpL` unchanged — taken verbatim from `floatClose_mhProjAttnFull`, since
    `FloatClose`'s error clause does not mention the window — and it still carries `Real.exp`.
    That is fine and it is the design: `FloatBridgesTo.capped` bounds the modulus by `2·mag`,
    and `mag` is now rational, so no numeral ever meets the exponential.

    ⛔ Capping the ORIGINAL bridge does not achieve this: `capped` replaces `mod`, never `mag`,
    and `mhpB`'s `Real.exp` sits at `δ = attnScaledErr ≈ 3.6·10¹⁰` on ViT-Tiny's block 0 — far
    past `exp_sub_one_le`'s `x < 1`, so it has no rational bound at all. -/
theorem floatClose_mhProjAttnFullCap (M : FloatModel) (fexp : ℝ → ℝ) {h n dh : Nat}
    (Wq Wk Wv : Mat (h * dh) (h * dh)) (bq bk bv : Vec (h * dh))
    {w' β A scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β) :
    FloatClose A (mhpBCap M n h dh w' β A eexp)
      (mhProjAttnFullFlat h n dh Wq Wk Wv bq bk bv)
      (mhProjAttnFullFlatF M fexp h n dh Wq Wk Wv bq bk bv)
      (fun e => mhpL M n h dh w' β A scaleA eexp e) := by
  have hLa0 : 0 ≤ layerAct (h * dh) w' β A := layerAct_nonneg hw' hβ hA
  have hLb00 : 0 ≤ layerBudget M.u (h * dh) w' β A 0 :=
    layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  have hvAF0 : 0 ≤ layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0 := by linarith
  have hcap0 : 0 ≤ smCap M.u eexp n := M.smCap_nonneg heexp0 hρ1
  have hg0 : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  have hn1 : (1:ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  -- the real side never exceeds `layerAct`, and `mhpBCap` dominates that
  have hdom : layerAct (h * dh) w' β A ≤ mhpBCap M n h dh w' β A eexp := by
    have h1 : layerAct (h * dh) w' β A
        ≤ (1 + smCap M.u eexp n)
            * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0) := by nlinarith
    have h2 : (1 + smCap M.u eexp n)
        * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0)
        ≤ (n : ℝ) * ((1 + smCap M.u eexp n)
            * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0)) := by nlinarith
    simp only [mhpBCap]; nlinarith
  refine ⟨fun v hv idx => ?_, (floatClose_mhProjAttnFull M fexp Wq Wk Wv bq bk bv hn hw' hβ hA
    heexp0 heexp1 hfexp hscaleA hρ1 hWq hbq hWk hbk hWv hbv).2⟩
  simp only [mhProjAttnFullFlat_apply, mhProjAttnFullFlatF_apply]
  set ii := (finProdFinEquiv.symm idx).1 with hii
  set jj := (finProdFinEquiv.symm idx).2 with hjj
  set hh := (finProdFinEquiv.symm jj).1 with hhh
  set cc := (finProdFinEquiv.symm jj).2 with hcc
  constructor
  · -- REAL: the convex-combination bound, then `hdom`
    have hVR : ∀ a b, |headSlab hh (projR Wv bv v) a b| ≤ layerAct (h * dh) w' β A :=
      fun a b => projR_abs_le Wv bv hA hWv hbv v hv a (finProdFinEquiv (hh, b))
    exact (sdpa_abs_le hn (headSlab hh (projR Wq bq v)) (headSlab hh (projR Wk bk v))
      (headSlab hh (projR Wv bv v)) hVR _ _).trans hdom
  · -- FLOAT: a rounded dot of bounded softmax weights against the bounded float V
    have hVF : ∀ a b, |headSlab hh (projF M Wv bv v) a b|
        ≤ layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0 :=
      fun a b => projF_abs_le M Wv bv hw' hA hWv hbv v hv a (finProdFinEquiv (hh, b))
    rw [M.sdpaF_eq]
    exact M.dot_abs_le_of _ _ (by linarith)
      (fun k => M.softmaxF_abs_le fexp _ heexp0 heexp1 hfexp hρ1 k)
      (fun k => hVF k cc)

/-- Multi-head projected attention float-bridges with the exp-free window. -/
noncomputable def floatBridgesTo_mhProjAttnFullCap (M : FloatModel) (fexp : ℝ → ℝ)
    {h n dh : Nat} (Wq Wk Wv : Mat (h * dh) (h * dh)) (bq bk bv : Vec (h * dh))
    {w' β scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β) :
    FloatBridgesTo (mhProjAttnFullFlat h n dh Wq Wk Wv bq bk bv)
      (mhProjAttnFullFlatF M fexp h n dh Wq Wk Wv bq bk bv) where
  mag := fun A => mhpBCap M n h dh w' β A eexp
  mod := fun A e => mhpL M n h dh w' β A scaleA eexp e
  close := fun A hA =>
    ⟨by
      have hLa0 : 0 ≤ layerAct (h * dh) w' β A := layerAct_nonneg hw' hβ hA
      have hLb00 : 0 ≤ layerBudget M.u (h * dh) w' β A 0 :=
        layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
      have hcap0 : 0 ≤ smCap M.u eexp n := M.smCap_nonneg heexp0 hρ1
      have hg0 : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
        sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
      have hn0 : (0:ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
      simp only [mhpBCap]; positivity,
     floatClose_mhProjAttnFullCap M fexp Wq Wk Wv bq bk bv hn hw' hβ hA heexp0 heexp1 hfexp
       hscaleA hρ1 hWq hbq hWk hbk hWv hbv⟩

namespace FloatBridgesTo

/-- **An envelope through CAPPED multi-head projected attention.** ⭐ Only the WINDOW is
    numeric: `Maps.capped` takes no modulus inequality, so `mhpL` — with its `Real.exp` — is
    never turned into a numeral. `g1` bounds the projections' Higham γ (fan-in `h·dh`), `g2` the
    output matmul's (fan-in `n`), and `sc` is a rational bound on `smCap`. -/
theorem Maps.mhProjAttnFullCap (M : FloatModel) (fexp : ℝ → ℝ) {h n dh : Nat}
    (Wq Wk Wv : Mat (h * dh) (h * dh)) (bq bk bv : Vec (h * dh))
    {w' β scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β)
    {g1 g2 sc Ā Ē Ā' Ē' : ℝ}
    (hg1 : (1 + M.u) ^ (h * dh + 2) - 1 ≤ g1)
    (hg2 : (1 + M.u) ^ (n + 1) - 1 ≤ g2)
    (hsc0 : 0 ≤ sc) (hsc : smCap M.u eexp n ≤ sc)
    (hĀ' : (1 + g2) * ((n : ℝ) * ((1 + sc)
             * ((1 + g1) * (((h * dh : ℕ) : ℝ) * w' * Ā + β)))) ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    (floatBridgesTo_mhProjAttnFullCap M fexp Wq Wk Wv bq bk bv hn hw' hβ heexp0 heexp1 hfexp
      hscaleA hρ1 hWq hbq hWk hbk hWv hbv).capped.Maps Ā Ē Ā' Ē' := by
  refine Maps.capped (fun A h0 hle => ?_) hĒ'
  show mhpBCap M n h dh w' β A eexp ≤ Ā'
  have hu := M.u_nonneg
  have hcap0 : 0 ≤ smCap M.u eexp n := M.smCap_nonneg heexp0 hρ1
  have hg10 : (0:ℝ) ≤ g1 := le_trans (sub_nonneg.mpr (one_le_pow₀ (by linarith))) hg1
  have hg20 : (0:ℝ) ≤ g2 := le_trans (sub_nonneg.mpr (one_le_pow₀ (by linarith))) hg2
  have hn0 : (0:ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  -- the float V bound, exactly `Maps.dense`'s window step at fan-in `h·dh`
  have hvAF : layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0
      ≤ (1 + g1) * (((h * dh : ℕ) : ℝ) * w' * Ā + β) := by
    have h1 := layerAct_le_num' (m := h * dh) (β := β) hw' hle
    have h2 := layerBudget_le_num' (m := h * dh) M.u_nonneg hw' hβ h0 hle
      (le_refl (0:ℝ)) (le_refl (0:ℝ)) hg1
    simp only [add_zero, mul_zero] at h2
    nlinarith
  have hvAF0 : 0 ≤ layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0 :=
    add_nonneg (layerAct_nonneg hw' hβ h0) (layerBudget_nonneg M.u_nonneg hw' hβ h0 le_rfl)
  have hXY : (1 + smCap M.u eexp n)
        * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0)
      ≤ (1 + sc) * ((1 + g1) * (((h * dh : ℕ) : ℝ) * w' * Ā + β)) := by
    have hR0 : (0:ℝ) ≤ (1 + g1) * (((h * dh : ℕ) : ℝ) * w' * Ā + β) := le_trans hvAF0 hvAF
    nlinarith
  have hstep : (n : ℝ) * ((1 + smCap M.u eexp n)
        * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0))
      ≤ (n : ℝ) * ((1 + sc) * ((1 + g1) * (((h * dh : ℕ) : ℝ) * w' * Ā + β))) :=
    mul_le_mul_of_nonneg_left hXY hn0
  have hinner0 : (0:ℝ) ≤ (n : ℝ) * ((1 + smCap M.u eexp n)
        * (layerAct (h * dh) w' β A + layerBudget M.u (h * dh) w' β A 0)) := by positivity
  simp only [mhpBCap]
  nlinarith

end FloatBridgesTo


-- ════════════════════════════════════════════════════════════════
-- § Worked sites at ViT-Tiny's emitted numerals
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The row softmax leaf, closed at ViT-Tiny's `n = 197`.** Window `1.021`, modulus `2.042`
    — and note both are numerals the kernel evaluates, where the underlying `smErr` modulus is
    `Real.exp` of a perturbation ~10¹⁰ (`scripts/float_budget_envelope.py`, `vit_chain`). The
    input window is arbitrary: a softmax RESETS. -/
example (M : FloatModel) (hMu : M.u ≤ u32) (fexp : ℝ → ℝ)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ (1 / 100) * Real.exp t)
    (hρ : smRho M.u (1 / 100) 197 < 1) (Ā Ē : ℝ) :
    (floatBridgesTo_softmaxRow (n := 197) M fexp (by norm_num) (by norm_num) hfexp
      hρ).capped.Maps Ā Ē (1021 / 10 ^ 3) (2042 / 10 ^ 3) := by
  refine FloatBridgesTo.Maps.softmaxRow M fexp (by norm_num) (by norm_num) hfexp hρ ?_
    (by norm_num)
  have hg := M.gamma_num (k := 198) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32])
    (by norm_num [u32])
  have hrb := smRho_le_of M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6) hg
    (by norm_num) (by norm_num)
  have hc := smCap_le M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6)
    (c := 2022 / 10 ^ 5) (by norm_num) hMu hrb (by norm_num) (by norm_num [u32])
  linarith

/-- ⭐⭐ **ViT-Tiny's block-0 attention site, end to end, at `vit_chain`'s numerals.** Multi-head
    projected attention (`h = 3`, `dh = 64`, `n = 197` tokens) at the capped window, then the
    per-row out-projection `Wo` — the exact pair the block bridge composes
    (`floatBridges_vitBlockMHFull`).

    In: the LayerNorm-affine output `(2.519·10⁵, 5.037·10⁵)`. Out: `(9.147·10¹¹, 1.830·10¹²)`.
    ⛔ The intermediate `6.805·10⁹ / 1.361·10¹⁰` has `budget = 2 × window` exactly — the cap's
    tell (§9). This site is where a ViT number stops being a fold. -/
example (M : FloatModel) (hMu : M.u ≤ u32) (fexp : ℝ → ℝ)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ (1 / 100) * Real.exp t)
    (hρ : smRho M.u (1 / 100) 197 < 1)
    (Wq Wk Wv Wo : Mat (3 * 64) (3 * 64)) (bq bk bv bo : Vec (3 * 64))
    (hWq : ∀ i j, |Wq i j| ≤ 7 / 10) (hbq : ∀ j, |bq j| ≤ 9 / 10)
    (hWk : ∀ i j, |Wk i j| ≤ 7 / 10) (hbk : ∀ j, |bk j| ≤ 9 / 10)
    (hWv : ∀ i j, |Wv i j| ≤ 7 / 10) (hbv : ∀ j, |bv j| ≤ 9 / 10)
    (hWo : ∀ i j, |Wo i j| ≤ 7 / 10) (hbo : ∀ j, |bo j| ≤ 9 / 10)
    (hscaleA : |(1 : ℝ) / Real.sqrt ((64 : ℕ) : ℝ)| ≤ 1 / 8) :
    ((floatBridgesTo_mhProjAttnFullCap (h := 3) (n := 197) (dh := 64) M fexp Wq Wk Wv bq bk bv
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hfexp hscaleA hρ
        hWq hbq hWk hbk hWv hbv).capped.comp
      (FloatBridgesTo.perRow 197
        (floatBridgesTo_dense M Wo bo (by norm_num) (by norm_num) (by norm_num)
          hWo hbo))).Maps
      (2519 * 10 ^ 2) (5037 * 10 ^ 2) (9147 * 10 ^ 8) (1830 * 10 ^ 9) := by
  have hsc : smCap M.u (1 / 100) 197 ≤ 2022 / 10 ^ 5 := by
    have hg := M.gamma_num (k := 198) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32])
      (by norm_num [u32])
    exact smCap_le M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6)
      (c := 2022 / 10 ^ 5) (by norm_num) hMu
      (smRho_le_of M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6) hg
        (by norm_num) (by norm_num)) (by norm_num) (by norm_num [u32])
  refine FloatBridgesTo.Maps.comp (by norm_num)
    (FloatBridgesTo.Maps.mhProjAttnFullCap (h := 3) (n := 197) (dh := 64) M fexp
      Wq Wk Wv bq bk bv (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      hfexp hscaleA hρ hWq hbq hWk hbk hWv hbv
      (g1 := 1157 / 10 ^ 8) (g2 := 1181 / 10 ^ 8) (sc := 2022 / 10 ^ 5)
      (Ā' := 6805 * 10 ^ 6) (Ē' := 1361 * 10 ^ 7)
      (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32])
        (by norm_num [u32]))
      (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32])
        (by norm_num [u32]))
      (by norm_num) hsc (by norm_num) (by norm_num))
    (FloatBridgesTo.Maps.perRow 197
      (FloatBridgesTo.Maps.dense M Wo bo (by norm_num) (by norm_num) (by norm_num) hWo hbo
        (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32])
          (by norm_num [u32]))
        (Ā' := 9147 * 10 ^ 8) (Ē' := 1830 * 10 ^ 9) (by norm_num [u32]) (by norm_num [u32])))

end Proofs
