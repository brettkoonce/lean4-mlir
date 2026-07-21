import LeanMlir.Proofs.Float.FloatComposeBridge
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-! # ℝ→Float32 bridge: depth-linear composition via tail gains (the adjoint chain)

`FloatClose.comp` composes per-op certificates, but the composed modulus
`Lg ∘ Lf` multiplies the inherited error by each layer's *worst-case* gain —
for a dense layer the fan-in row sum `m·w` — so the whole-net budget compounds
`(m·w)^depth` and is empirically vacuous past ~6 layers (measured on gfx1100:
~10⁶× loose at depth 3, ~10⁴⁰× at depth 24, while the actual roundoff stays
flat). The interval fold pays the *product of layer gains*; the actual error
pays the *gain of the composed tail*, which is astronomically smaller.

This file proves the telescoping (hybrid-argument) alternative ONCE, generic
in depth, by induction on the layer list:

    |chainF x − chainR x| ≤ Σᵢ Hᵢ · bᵢ        (`chain_adjointClose`)

where `bᵢ` is layer i's *fresh* rounding budget (its `FloatClose` modulus at
`e = 0` — already proven per-op, e.g. `layerBudget … 0`) and `Hᵢ` is a
windowed Lipschitz gain of the REAL tail `fₙ ∘ … ∘ fᵢ₊₁` after layer i
(`LipOnWindow`). The proof is exact — no linearization: hybrid i and hybrid
i−1 differ by one layer's fresh budget pushed through the real tail, and the
budgets telescope. Soundness therefore rests only on how the `Hᵢ` are
discharged:

- **PROVEN tier:** `tailGains_suffixProd` — take `Hᵢ` = the product of the
  per-layer real gains after i (`lipOnWindow_dense`, `lipOnWindow_relu`, …).
  This recovers exactly the old interval-fold bound, so the theorem strictly
  subsumes the `.comp` chain. But `Hᵢ` is a property of the *composed* tail
  (`‖J_tail‖`-style), not of its factors, so any tighter tail bound slots in
  with no change to the chain proof.
- **MEASURED tier:** supply `Hᵢ` from the adjoint/VJP probe (the backward
  pass evaluated along the trajectory measures exactly these tail gains; see
  `scripts/` probes). The hypotheses are ordinary named arguments, so a
  measured instantiation is quarantined the same way as `esig`/`egelu`:
  the Lean statement stays 3-axiom clean, the number is supplied at the
  application site with its provenance stated.

Honest scope: `LipOnWindow` demands the gain on the whole `|·| ≤ A` window,
not just near the trajectory — the measured-Jacobian proxy underestimates the
window supremum, and a trajectory-tube refinement (radius = accumulated
budget) is the natural v2. The window is uniform across the chain (the
`towerBack`-style dim-preserving fold); heterogeneous stems/heads compose at
the ends via the existing `FloatClose.comp`.
-/

namespace Proofs

open FloatModel

/-- Windowed Lipschitz gain of a real map: on inputs within magnitude `A`, a
    per-coordinate input spread `e` is amplified to at most `H·e` per
    coordinate. This is the tail-gain currency of the adjoint chain — the
    quantity the backward/VJP pass measures along a trajectory. -/
def LipOnWindow {m n : Nat} (A H : ℝ) (f : Vec m → Vec n) : Prop :=
  ∀ (u v : Vec m) (e : ℝ), 0 ≤ e → (∀ k, |u k| ≤ A) → (∀ k, |v k| ≤ A) →
    (∀ k, |u k - v k| ≤ e) → ∀ j, |f u j - f v j| ≤ H * e

/-- One chain layer's local certificate: real map `f`, float map `fF`, both
    hold the `|·| ≤ A` window, and the float map is within the *fresh* budget
    `b` of the real map at every window point (the per-op `*_close` lemma at
    input error `e = 0`). -/
structure LayerCert (m : Nat) (A : ℝ) where
  f : Vec m → Vec m
  fF : Vec m → Vec m
  b : ℝ
  b_nonneg : 0 ≤ b
  window : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |f v j| ≤ A ∧ |fF v j| ≤ A
  fresh : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |fF v j - f v j| ≤ b

/-- The real chain: `chainR [l₁, …, lₙ] = lₙ.f ∘ … ∘ l₁.f` (head applied
    first — the `towerBack` orientation). -/
noncomputable def chainR {m : Nat} {A : ℝ} : List (LayerCert m A) → Vec m → Vec m
  | [] => id
  | l :: ls => chainR ls ∘ l.f

/-- The float chain: same fold over the float maps. -/
noncomputable def chainF {m : Nat} {A : ℝ} : List (LayerCert m A) → Vec m → Vec m
  | [] => id
  | l :: ls => chainF ls ∘ l.fF

/-- Tail-gain hypotheses: `Hᵢ` bounds the windowed Lipschitz gain of the REAL
    suffix after layer i. This is the per-position side condition the
    telescoping proof consumes — dischargeable by worst-case products
    (`tailGains_suffixProd`, the PROVEN face) or supplied from measurement. -/
def TailGains {m : Nat} {A : ℝ} : List (LayerCert m A) → List ℝ → Prop
  | [], [] => True
  | _l :: ls, H :: Hs => LipOnWindow A H (chainR (A := A) ls) ∧ TailGains ls Hs
  | _, _ => False

/-- The adjoint-chain certificate value: `Σᵢ Hᵢ · bᵢ`, folded positionally.
    Depth-LINEAR in the fresh budgets — no gain products between budgets. -/
noncomputable def chainBudget {m : Nat} {A : ℝ} : List (LayerCert m A) → List ℝ → ℝ
  | l :: ls, H :: Hs => H * l.b + chainBudget ls Hs
  | _, _ => 0

/-- **THE ADJOINT CHAIN: depth-linear whole-chain float certificate.** If every
    layer carries a local fresh budget `bᵢ` on the window and every real tail
    has windowed gain `Hᵢ`, the float chain is within `Σᵢ Hᵢ·bᵢ` of the real
    chain — proven once by induction on the layer list, any depth. The
    telescoping is exact (hybrid i vs hybrid i−1 = one fresh budget through
    one real tail); nothing is linearized. -/
theorem chain_adjointClose {m : Nat} {A : ℝ}
    (ls : List (LayerCert m A)) (Hs : List ℝ) (hH : TailGains ls Hs)
    (x : Vec m) (hx : ∀ k, |x k| ≤ A) (j : Fin m) :
    |chainF ls x j - chainR ls x j| ≤ chainBudget ls Hs := by
  induction ls generalizing Hs x with
  | nil =>
    cases Hs with
    | nil => simp [chainF, chainR, chainBudget]
    | cons H Hs => exact (hH : False).elim
  | cons l ls ih =>
    cases Hs with
    | nil => exact (hH : False).elim
    | cons H Hs =>
      obtain ⟨hHl, hHs⟩ := hH
      have hz := l.window x hx
      have h1 : |chainF ls (l.fF x) j - chainR ls (l.fF x) j| ≤ chainBudget ls Hs :=
        ih Hs hHs (l.fF x) (fun k => (hz k).2)
      have h2 : |chainR ls (l.fF x) j - chainR ls (l.f x) j| ≤ H * l.b :=
        hHl (l.fF x) (l.f x) l.b l.b_nonneg (fun k => (hz k).2) (fun k => (hz k).1)
          (l.fresh x hx) j
      calc |chainF (l :: ls) x j - chainR (l :: ls) x j|
          = |chainF ls (l.fF x) j - chainR ls (l.f x) j| := rfl
        _ ≤ |chainF ls (l.fF x) j - chainR ls (l.fF x) j|
            + |chainR ls (l.fF x) j - chainR ls (l.f x) j| := abs_sub_le _ _ _
        _ ≤ chainBudget ls Hs + H * l.b := add_le_add h1 h2
        _ = H * l.b + chainBudget ls Hs := by ring

/-- **The certificate as a decision guarantee: rounding-robust argmax.** If the
    REAL chain's output at `j₀` beats every other coordinate by more than twice
    the adjoint-chain budget, the FLOAT chain has the SAME argmax — the rounded
    net makes the same prediction as the exact net. This is the whole-net float
    certificate's payoff: the `Σᵢ Hᵢ·bᵢ` bound turns a margin into a proof that
    rounding cannot flip the decision. -/
theorem chain_argmaxSafe {m : Nat} {A : ℝ}
    (ls : List (LayerCert m A)) (Hs : List ℝ) (hH : TailGains ls Hs)
    (x : Vec m) (hx : ∀ k, |x k| ≤ A) (j₀ : Fin m)
    (hmargin : ∀ j, j ≠ j₀ →
      2 * chainBudget ls Hs < chainR ls x j₀ - chainR ls x j) :
    ∀ j, j ≠ j₀ → chainF ls x j < chainF ls x j₀ := by
  intro j hj
  have b0 := abs_le.mp (chain_adjointClose ls Hs hH x hx j₀)
  have bj := abs_le.mp (chain_adjointClose ls Hs hH x hx j)
  have := hmargin j hj
  linarith [b0.1, b0.2, bj.1, bj.2]

-- ════════════════════════════════════════════════════════════════
-- § Discharging the tail gains, PROVEN face: worst-case products
--   (recovers the old interval-fold bound — the theorem subsumes it)
-- ════════════════════════════════════════════════════════════════

/-- The identity map has windowed gain 1 (the empty tail). -/
theorem lipOnWindow_id {m : Nat} (A : ℝ) : LipOnWindow A 1 (id : Vec m → Vec m) :=
  fun _u _v _e _he _hu _hv hd j => by simpa using hd j

/-- Windowed gains compose multiplicatively when the inner map holds the
    window — the worst-case (product) face of a tail gain. -/
theorem LipOnWindow.comp {m : Nat} {A g h : ℝ} {f t : Vec m → Vec m}
    (hf : LipOnWindow A g f) (hg0 : 0 ≤ g)
    (hfw : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |f v j| ≤ A)
    (ht : LipOnWindow A h t) :
    LipOnWindow A (h * g) (t ∘ f) := by
  intro u v e he hu hv hd j
  have h2 := ht (f u) (f v) (g * e) (mul_nonneg hg0 he) (hfw u hu) (hfw v hv)
    (fun k => hf u v e he hu hv hd k) j
  calc |t (f u) j - t (f v) j| ≤ h * (g * e) := h2
    _ = h * g * e := by ring

/-- Per-layer real gains give the whole real chain the PRODUCT gain. -/
theorem lipOnWindow_chainR {m : Nat} {A : ℝ}
    {ls : List (LayerCert m A)} {gs : List ℝ}
    (hlg : List.Forall₂ (fun (l : LayerCert m A) (g : ℝ) =>
      0 ≤ g ∧ LipOnWindow A g l.f) ls gs) :
    LipOnWindow A gs.prod (chainR ls) := by
  induction hlg with
  | nil => simpa [chainR] using lipOnWindow_id (m := m) A
  | @cons l g ls gs hlgHead _hlgTail ih =>
    rw [List.prod_cons, mul_comm]
    exact LipOnWindow.comp hlgHead.2 hlgHead.1
      (fun v hv j => (l.window v hv j).1) ih

/-- Suffix products of a gain list: position i gets `∏_{j>i} gⱼ`. -/
def suffixGains : List ℝ → List ℝ
  | [] => []
  | _g :: gs => gs.prod :: suffixGains gs

/-- **The PROVEN discharge:** per-layer real gains yield `TailGains` with the
    suffix products. Instantiating `chain_adjointClose` with these gains is
    exactly the old worst-case interval fold — so the adjoint chain is never
    worse, and any tighter tail gain (measured or proven about the composed
    tail directly) strictly improves it. -/
theorem tailGains_suffixProd {m : Nat} {A : ℝ}
    {ls : List (LayerCert m A)} {gs : List ℝ}
    (hlg : List.Forall₂ (fun (l : LayerCert m A) (g : ℝ) =>
      0 ≤ g ∧ LipOnWindow A g l.f) ls gs) :
    TailGains ls (suffixGains gs) := by
  induction hlg with
  | nil => trivial
  | @cons l g ls gs _hlgHead hlgTail ih =>
    exact ⟨lipOnWindow_chainR hlgTail, ih⟩

-- ════════════════════════════════════════════════════════════════
-- § Per-op gain instances and LayerCert constructors
-- ════════════════════════════════════════════════════════════════

/-- ReLU has windowed gain 1 (`relu_close` is exactly 1-Lipschitz-ness). -/
theorem lipOnWindow_relu {n : Nat} (A : ℝ) : LipOnWindow A 1 (relu n) :=
  fun u v e _he _hu _hv hd i => by simpa using relu_close u v e hd i

/-- Dense has windowed gain `m·w'` (the fan-in row sum — the gain the interval
    fold pays per layer; here it is paid only inside a single tail product). -/
theorem lipOnWindow_dense {m n : Nat} (W : Mat m n) (b : Vec n) {w' A : ℝ}
    (hW : ∀ i j, |W i j| ≤ w') :
    LipOnWindow A (m * w') (Proofs.dense W b) := by
  intro u v e he _hu _hv hd j
  have hstep : Proofs.dense W b u j - Proofs.dense W b v j
      = ∑ i, (u i - v i) * W i j := by
    show ((∑ i, u i * W i j) + b j) - ((∑ i, v i * W i j) + b j)
        = ∑ i, (u i - v i) * W i j
    rw [add_sub_add_right_eq_sub, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hstep]
  calc |∑ i, (u i - v i) * W i j|
      ≤ ∑ i, |(u i - v i) * W i j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin m, e * w' := Finset.sum_le_sum fun i _ => by
        rw [abs_mul]
        exact mul_le_mul (hd i) (hW i j) (abs_nonneg _) he
    _ = m * w' * e := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring

/-- **Softmax has a windowed gain** — the nonlinear perturbation modulus
    `e^{2δ} − 1` (`softmax_perturb`) IS linear-on-a-window: convexity of `exp`
    through the origin gives `e^{2e} − 1 ≤ (e/(2A))·(e^{4A} − 1)` for
    `e ≤ 2A`, and window points never differ by more than `2A`. So softmax
    slots into the adjoint chain with gain `(e^{4A} − 1)/(2A)` — the
    "doesn't fit a local contract" objection dissolves on the window. -/
theorem lipOnWindow_softmax {n : Nat} {A : ℝ} (hA : 0 < A) :
    LipOnWindow A ((Real.exp (4 * A) - 1) / (2 * A)) (softmax n) := by
  intro u v e he hu hv hd j
  have h2A : (0:ℝ) < 2 * A := by linarith
  have hH0 : 0 ≤ (Real.exp (4 * A) - 1) / (2 * A) := by
    apply div_nonneg _ h2A.le
    have : (1:ℝ) ≤ Real.exp (4 * A) := by
      rw [show (1:ℝ) = Real.exp 0 from (Real.exp_zero).symm]
      exact Real.exp_le_exp.mpr (by linarith)
    linarith
  by_cases hcase : e ≤ 2 * A
  · -- convexity through the origin: exp(2e) − 1 ≤ (e/(2A))·(exp(4A) − 1)
    have hsp := softmax_perturb u v hd j
    have ht0 : 0 ≤ e / (2 * A) := div_nonneg he h2A.le
    have hb0 : (0:ℝ) ≤ 1 - e / (2 * A) := by
      have : e / (2 * A) ≤ 1 := (div_le_one h2A).mpr hcase
      linarith
    have hab : e / (2 * A) + (1 - e / (2 * A)) = 1 := by ring
    have hconv := convexOn_exp.2 (Set.mem_univ (4 * A)) (Set.mem_univ 0)
      ht0 hb0 hab
    simp only [smul_eq_mul, mul_zero, add_zero, Real.exp_zero, mul_one] at hconv
    rw [show e / (2 * A) * (4 * A) = 2 * e by
      rw [div_mul_eq_mul_div, div_eq_iff h2A.ne']; ring] at hconv
    have hexp : Real.exp (2 * e) - 1 ≤ e / (2 * A) * (Real.exp (4 * A) - 1) := by
      have hid : e / (2 * A) * Real.exp (4 * A) + (1 - e / (2 * A)) - 1
          = e / (2 * A) * (Real.exp (4 * A) - 1) := by ring
      linarith
    calc |softmax n u j - softmax n v j| ≤ Real.exp (2 * e) - 1 := hsp
      _ ≤ e / (2 * A) * (Real.exp (4 * A) - 1) := hexp
      _ = (Real.exp (4 * A) - 1) / (2 * A) * e := by ring
  · -- window points never differ by more than 2A; the gain line lies above
    rw [not_le] at hcase
    have hd' : ∀ k, |u k - v k| ≤ 2 * A := fun k => by
      calc |u k - v k| ≤ |u k| + |v k| := abs_sub _ _
        _ ≤ A + A := add_le_add (hu k) (hv k)
        _ = 2 * A := by ring
    have hsp := softmax_perturb u v hd' j
    calc |softmax n u j - softmax n v j|
        ≤ Real.exp (2 * (2 * A)) - 1 := hsp
      _ = Real.exp (4 * A) - 1 := by rw [show (2:ℝ) * (2 * A) = 4 * A by ring]
      _ = (Real.exp (4 * A) - 1) / (2 * A) * (2 * A) := by field_simp
      _ ≤ (Real.exp (4 * A) - 1) / (2 * A) * e :=
          mul_le_mul_of_nonneg_left hcase.le hH0

/-- Any `FloatClose A B` layer whose output magnitude fits back in the window
    (`B ≤ A`) is a `LayerCert` with fresh budget `L 0` — every existing per-op
    and per-block bridge instance slots into the chain through this door. -/
noncomputable def LayerCert.of_floatClose {m : Nat} {A B : ℝ}
    {f fF : Vec m → Vec m} {L : ℝ → ℝ}
    (h : FloatClose A B f fF L) (hBA : B ≤ A) (hL0 : 0 ≤ L 0) :
    LayerCert m A where
  f := f
  fF := fF
  b := L 0
  b_nonneg := hL0
  window := fun v hv j => ⟨(h.1 v hv j).1.trans hBA, (h.1 v hv j).2.trans hBA⟩
  fresh := fun v hv j => h.2 v v 0 hv hv (fun k => by simp) j

/-- **Demo instance: a relu∘dense layer as a `LayerCert`.** Fresh budget =
    the proven `layerBudget … 0`; the window absorbs the layer output when
    `layerAct + layerBudget(0) ≤ A` (normalized/bounded regimes — exactly the
    stays-in-window discipline the per-net folds thread by hand). -/
noncomputable def layerCert_reluDense {m : Nat} (M : FloatModel)
    (W : Mat m m) (b : Vec m) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β)
    (hfit : layerAct m w' β A + layerBudget M.u m w' β A 0 ≤ A) :
    LayerCert m A :=
  LayerCert.of_floatClose
    ((floatClose_dense M W b hw' hβ hA hm hW hb).comp
      (floatClose_relu (layerAct m w' β A + layerBudget M.u m w' β A 0)))
    hfit
    (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)

/-- The real relu∘dense layer's windowed gain is the fan-in row sum `m·w'`. -/
theorem lipOnWindow_reluDense {m : Nat} (W : Mat m m) (b : Vec m) {w' A : ℝ}
    (hw' : 0 ≤ w')
    (hdw : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |Proofs.dense W b v j| ≤ A)
    (hW : ∀ i j, |W i j| ≤ w') :
    LipOnWindow A (m * w') (relu m ∘ Proofs.dense W b) := by
  have := LipOnWindow.comp (lipOnWindow_dense W b hW)
    (by positivity) hdw (lipOnWindow_relu (n := m) A)
  simpa using this

end Proofs
