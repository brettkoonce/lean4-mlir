import LeanMlir.Proofs.Codegen.AdjointChainBridge

/-!
# The adjoint chain, v2: heterogeneous windows and dimensions

`AdjointChainBridge.lean`'s `LayerCert m A` carries ONE window `A` that every
layer must map into ITSELF. That forward-invariance is not a modelling choice —
it is the source of the `hfit` side condition, and `hfit` is why the whole tier
has no instance at a real net (`Cifar8ChainCert.lean` §2, `formalization.yaml`
fidelity §4c): unfolded it reads `(m·w'·A + β)·(1+u)^(m+2) ≤ A`, which forces
`m·w' ≤ 1`, while the committed CIFAR-8 net runs `m·w'` at 24 … 114.

The fix is not to satisfy `hfit`. It is to stop asking: give each layer its own
input and output window and *define* the output window to be the propagated
bound. Then `LayerCertH.of_floatClose` carries no side condition at all, and
`layerCertH_reluDense` — the peer of `layerCert_reluDense` — takes no `hfit`.
This is exactly what `FloatClose.comp` has always done by threading `A → B → C`;
the v1 chain regressed to a uniform window only so the induction could run over
a plain `List`. An indexed `LayerChain` restores it.

`FloatClose` is already dimension-heterogeneous (`floatClose_dense` is stated at
`Mat m n`), so indexing the chain by `(dim, window)` at both ends closes the
noted dimension gap in the same move. Both v1 obstacles, one construction.

Nothing here is CIFAR-8-specific; `Cifar8ChainCert` migrates onto it separately.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Why v1 cannot be instantiated (the obstruction, as a theorem)
-- ════════════════════════════════════════════════════════════════

/-- **`hfit` forces `m·w' ≤ 1`.** The v1 window condition of `layerCert_reluDense`
    is satisfiable only when a layer's fan-in times its largest weight magnitude
    is at most one. At He init `w' ≈ 1/√m`, so `m·w' ≈ √m`: violated at every
    width above 1. Recorded here so the v2 design has its reason attached. -/
theorem hfit_forces_tiny_weights (M : FloatModel) {m : Nat} {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 < A)
    (hfit : layerAct m w' β A + layerBudget M.u m w' β A 0 ≤ A) :
    (m : ℝ) * w' ≤ 1 := by
  have hu := M.u_nonneg
  have hG : (1:ℝ) ≤ (1 + M.u) ^ (m + 2) := one_le_pow₀ (by linarith)
  have hkey : ((m : ℝ) * w' * A + β) * (1 + M.u) ^ (m + 2) ≤ A := by
    simp only [layerAct, layerBudget, add_zero, mul_zero] at hfit
    nlinarith [hfit]
  have hmw : 0 ≤ (m : ℝ) * w' := mul_nonneg (Nat.cast_nonneg m) hw'
  nlinarith [hkey, hG, hA, hβ, hmw, mul_nonneg hmw hA.le]

/-- **The committed CIFAR-8 net cannot instantiate v1.** At the widest stage of
    `cifar8Verified` (`conv 32→32`, fan-in `m = 32·9 = 288`, He-init
    `|W| ≤ 1/2` — the measured max is 0.39) the v1 window condition is
    contradictory for every positive window. -/
theorem cifar8_stage_defeats_hfit (M : FloatModel) {A : ℝ} (hA : 0 < A)
    (hfit : layerAct 288 (1/2) 1 A + layerBudget M.u 288 (1/2) 1 A 0 ≤ A) :
    False := by
  have h := hfit_forces_tiny_weights M (by norm_num) (by norm_num) hA hfit
  norm_num at h

-- ════════════════════════════════════════════════════════════════
-- § The heterogeneous layer certificate
-- ════════════════════════════════════════════════════════════════

/-- One chain layer, with SEPARATE input and output windows (and dimensions).
    The `LayerCert m A` peer that does not demand the window be forward-invariant. -/
structure LayerCertH (min mout : Nat) (Ain Aout : ℝ) where
  f : Vec min → Vec mout
  fF : Vec min → Vec mout
  b : ℝ
  b_nonneg : 0 ≤ b
  window : ∀ v, (∀ k, |v k| ≤ Ain) → ∀ j, |f v j| ≤ Aout ∧ |fF v j| ≤ Aout
  fresh : ∀ v, (∀ k, |v k| ≤ Ain) → ∀ j, |fF v j - f v j| ≤ b

/-- **The door, with no side condition.** Any `FloatClose Ain Aout` layer is a
    `LayerCertH` — contrast `LayerCert.of_floatClose`, which additionally needs
    `hBA : B ≤ A`. Deleting that hypothesis is the entire v2 design change. -/
noncomputable def LayerCertH.of_floatClose {min mout : Nat} {Ain Aout : ℝ}
    {f fF : Vec min → Vec mout} {L : ℝ → ℝ}
    (h : FloatClose Ain Aout f fF L) (hL0 : 0 ≤ L 0) :
    LayerCertH min mout Ain Aout where
  f := f
  fF := fF
  b := L 0
  b_nonneg := hL0
  window := fun v hv j => h.1 v hv j
  fresh := fun v hv j => h.2 v v 0 hv hv (fun k => by simp) j

-- ════════════════════════════════════════════════════════════════
-- § The chain
-- ════════════════════════════════════════════════════════════════

/-- A chain of layers, indexed by `(dimension, window)` at each end. Each `cons`
    is free to change both — the v1 `List (LayerCert m A)` is the special case
    where every index coincides. -/
inductive LayerChain : Nat → ℝ → Nat → ℝ → Type where
  | nil {m : Nat} {A : ℝ} : LayerChain m A m A
  | cons {m₀ m₁ m₂ : Nat} {A₀ A₁ A₂ : ℝ}
      (l : LayerCertH m₀ m₁ A₀ A₁) (ls : LayerChain m₁ A₁ m₂ A₂) :
      LayerChain m₀ A₀ m₂ A₂

/-- The real chain (head applied first — the `towerBack` orientation). -/
noncomputable def chainRH : {m₀ m₁ : Nat} → {A₀ A₁ : ℝ} →
    LayerChain m₀ A₀ m₁ A₁ → Vec m₀ → Vec m₁
  | _, _, _, _, .nil => id
  | _, _, _, _, .cons l ls => chainRH ls ∘ l.f

/-- The float chain: the same fold over the float maps. -/
noncomputable def chainFH : {m₀ m₁ : Nat} → {A₀ A₁ : ℝ} →
    LayerChain m₀ A₀ m₁ A₁ → Vec m₀ → Vec m₁
  | _, _, _, _, .nil => id
  | _, _, _, _, .cons l ls => chainFH ls ∘ l.fF

/-- Tail-gain hypotheses: `Hᵢ` bounds the windowed gain of the REAL suffix after
    layer `i`, on that suffix's OWN input window.

    ⚠ No proven discharge exists for this predicate: v1's `tailGains_suffixProd` has
    no heterogeneous-dimension peer (`LipOnWindow.comp` is stated at a single
    dimension), so every `TailGainsH` hypothesis in the repo is supplied by
    measurement, and a measured Jacobian underestimates the window supremum
    (`AdjointChainBridge.lean` §"Honest scope"). -/
def TailGainsH : {m₀ m₁ : Nat} → {A₀ A₁ : ℝ} →
    LayerChain m₀ A₀ m₁ A₁ → List ℝ → Prop
  | _, _, _, _, .nil, [] => True
  | _, _, _, _, @LayerChain.cons _ _ _ _ A₁ _ _ ls, H :: Hs =>
      LipOnWindow A₁ H (chainRH ls) ∧ TailGainsH ls Hs
  | _, _, _, _, _, _ => False

/-- **The last tail gain is at least 1** whenever its window is positive: the empty tail is
    `id`, and `LipOnWindow A H id` at two window points `A·1` and `0` forces `H·A ≥ A`. So
    `chainBudgetH ≥ b_last` for every `TailGainsH`-satisfying gain list — the floor
    `Cifar8ChainCert.lean` §"Magnitude" and `formalization.yaml` §4c quote (1.8e13 at the
    He profile) is not an estimate about the measured `Hᵢ` but a consequence of the
    hypothesis itself. -/
theorem lipOnWindow_id_ge_one {m : Nat} {A H : ℝ} (hA : 0 < A) (hm : 0 < m)
    (h : LipOnWindow A H (id : Vec m → Vec m)) : 1 ≤ H := by
  have := h (fun _ => A) (fun _ => 0) A hA.le (fun _ => by simp [abs_of_pos hA])
    (fun _ => by simp [hA.le]) (fun _ => by simp [abs_of_pos hA]) ⟨0, hm⟩
  simp [abs_of_pos hA] at this
  nlinarith

/-- The adjoint-chain certificate value `Σᵢ Hᵢ·bᵢ` — depth-LINEAR, as in v1. -/
noncomputable def chainBudgetH : {m₀ m₁ : Nat} → {A₀ A₁ : ℝ} →
    LayerChain m₀ A₀ m₁ A₁ → List ℝ → ℝ
  | _, _, _, _, .cons l ls, H :: Hs => H * l.b + chainBudgetH ls Hs
  | _, _, _, _, _, _ => 0

-- ════════════════════════════════════════════════════════════════
-- § The theorems, unchanged in content
-- ════════════════════════════════════════════════════════════════

/-- **The adjoint chain at heterogeneous windows and dimensions.** Same
    telescoping proof as v1's `chain_adjointClose` — hybrid `i` vs hybrid `i−1`
    is one fresh budget through one real tail — now with each layer's output
    window feeding the next layer's input window instead of itself. -/
theorem chain_adjointCloseH : ∀ {m₀ m₁ : Nat} {A₀ A₁ : ℝ}
    (ls : LayerChain m₀ A₀ m₁ A₁) (Hs : List ℝ), TailGainsH ls Hs →
    ∀ (x : Vec m₀), (∀ k, |x k| ≤ A₀) → ∀ (j : Fin m₁),
    |chainFH ls x j - chainRH ls x j| ≤ chainBudgetH ls Hs := by
  intro m₀ m₁ A₀ A₁ ls
  induction ls with
  | nil =>
    intro Hs hH x _ j
    cases Hs with
    | nil => simp [chainFH, chainRH, chainBudgetH]
    | cons H Hs => exact (hH : False).elim
  | cons l ls ih =>
    intro Hs hH x hx j
    cases Hs with
    | nil => exact (hH : False).elim
    | cons H Hs =>
      obtain ⟨hHl, hHs⟩ := hH
      have hz := l.window x hx
      have h1 : |chainFH ls (l.fF x) j - chainRH ls (l.fF x) j| ≤ chainBudgetH ls Hs :=
        ih Hs hHs (l.fF x) (fun k => (hz k).2) j
      have h2 : |chainRH ls (l.fF x) j - chainRH ls (l.f x) j| ≤ H * l.b :=
        hHl (l.fF x) (l.f x) l.b l.b_nonneg (fun k => (hz k).2) (fun k => (hz k).1)
          (l.fresh x hx) j
      calc |chainFH (.cons l ls) x j - chainRH (.cons l ls) x j|
          = |chainFH ls (l.fF x) j - chainRH ls (l.f x) j| := rfl
        _ ≤ |chainFH ls (l.fF x) j - chainRH ls (l.fF x) j|
            + |chainRH ls (l.fF x) j - chainRH ls (l.f x) j| := abs_sub_le _ _ _
        _ ≤ chainBudgetH ls Hs + H * l.b := add_le_add h1 h2
        _ = H * l.b + chainBudgetH ls Hs := by ring

/-- **Rounding cannot flip the decision** — the v2 peer of `chain_argmaxSafe`. -/
theorem chain_argmaxSafeH {m₀ m₁ : Nat} {A₀ A₁ : ℝ}
    (ls : LayerChain m₀ A₀ m₁ A₁) (Hs : List ℝ) (hH : TailGainsH ls Hs)
    (x : Vec m₀) (hx : ∀ k, |x k| ≤ A₀) (j₀ : Fin m₁)
    (hmargin : ∀ j, j ≠ j₀ →
      2 * chainBudgetH ls Hs < chainRH ls x j₀ - chainRH ls x j) :
    ∀ j, j ≠ j₀ → chainFH ls x j < chainFH ls x j₀ := by
  intro j hj
  have b0 := abs_le.mp (chain_adjointCloseH ls Hs hH x hx j₀)
  have bj := abs_le.mp (chain_adjointCloseH ls Hs hH x hx j)
  have hm := hmargin j hj
  linarith [b0.1, b0.2, bj.1, bj.2]

-- ════════════════════════════════════════════════════════════════
-- § The relu∘dense layer, with NO window side condition
-- ════════════════════════════════════════════════════════════════

/-- The propagated output window of a relu∘dense layer: the real activation
    bound plus the layer's own fresh rounding budget. In v1 this had to be
    shown `≤ A`; here it simply IS the next window. -/
noncomputable def reluDenseOut (M : FloatModel) (m : Nat) (w' β Ain : ℝ) : ℝ :=
  layerAct m w' β Ain + layerBudget M.u m w' β Ain 0

theorem reluDenseOut_nonneg (M : FloatModel) {m : Nat} {w' β Ain : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ Ain) : 0 ≤ reluDenseOut M m w' β Ain :=
  add_nonneg (layerAct_nonneg hw' hβ hA)
    (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)

/-- **A relu∘dense layer as a `LayerCertH` — compare `layerCert_reluDense`,
    which needs `hfit`.** The hypothesis list is otherwise identical; the window
    obligation is discharged by construction because the output window is
    defined to be the propagated bound. -/
noncomputable def layerCertH_reluDense {m n : Nat} (M : FloatModel)
    (W : Mat m n) (b : Vec n) {w' β Ain : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ Ain) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β) :
    LayerCertH m n Ain (reluDenseOut M m w' β Ain) :=
  LayerCertH.of_floatClose
    ((floatClose_dense M W b hw' hβ hA hm hW hb).comp
      (floatClose_relu (reluDenseOut M m w' β Ain)))
    (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)

/-- **A conv+relu layer as a `LayerCertH`** (SAME padding, so the spatial dims are
    preserved and only the channel count moves). The output window is the conv's
    propagated bound at its RECEPTIVE-FIELD fan-in `ic·kH·kW` — note that is the
    quantity the budget depends on, not the flat activation width. No `hfit`. -/
noncomputable def layerCertH_reluConv {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β Ain : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ Ain) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    LayerCertH (ic * h * w) (oc * h * w) Ain (reluDenseOut M (ic * kH * kW) w' β Ain) :=
  LayerCertH.of_floatClose
    ((floatClose_flatConv (h := h) (w := w) M W b hw' hβ hA hn hW hb).comp
      (floatClose_relu (reluDenseOut M (ic * kH * kW) w' β Ain)))
    (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)

/-- **Max-pool as a `LayerCertH`** — exact in float (compare-and-select rounds
    nothing) and never grows magnitudes, so it carries the window across unchanged
    with a zero fresh budget. -/
noncomputable def layerCertH_maxPool (c h w : Nat) (A : ℝ) :
    LayerCertH (c * (2 * h) * (2 * w)) (c * h * w) A A :=
  LayerCertH.of_floatClose (floatClose_maxPool (c := c) (h := h) (w := w) A) le_rfl

/-- **A bare dense layer as a `LayerCertH`** — the logits head, no ReLU. -/
noncomputable def layerCertH_dense {m n : Nat} (M : FloatModel)
    (W : Mat m n) (b : Vec n) {w' β Ain : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ Ain) (hm : 0 < m)
    (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β) :
    LayerCertH m n Ain (reluDenseOut M m w' β Ain) :=
  LayerCertH.of_floatClose (floatClose_dense M W b hw' hβ hA hm hW hb)
    (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl)

/-- The propagated window folded along a list of per-stage fan-ins (pools carry the
    window unchanged, so they contribute no entry). -/
noncomputable def windowFold (M : FloatModel) (w' β : ℝ) : List Nat → ℝ → ℝ
  | [],      A => A
  | m :: ms, A => windowFold M w' β ms (reluDenseOut M m w' β A)

theorem windowFold_nonneg (M : FloatModel) {w' β : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) :
    ∀ (ms : List Nat) {A : ℝ}, 0 ≤ A → 0 ≤ windowFold M w' β ms A
  | [],      _, hA => hA
  | _ :: ms, _, hA =>
      windowFold_nonneg M hw' hβ ms (reluDenseOut_nonneg M hw' hβ hA)

-- ════════════════════════════════════════════════════════════════
-- § A tower, built without a window side condition
-- ════════════════════════════════════════════════════════════════

/-- A dense layer carrying its magnitude bounds (the v2 peer of
    `Cifar8ChainCert.BoundedDense`). -/
structure BoundedDenseH (m : Nat) (w' β : ℝ) where
  W : Mat m m
  b : Vec m
  hW : ∀ i j, |W i j| ≤ w'
  hb : ∀ j, |b j| ≤ β

/-- The window after `k` relu∘dense stages — the propagated bound, iterated.
    In v1 this had to collapse to a fixed point (`hfit`); here it is allowed to
    grow, which is what real nets do. -/
noncomputable def reluDenseOutIter (M : FloatModel) (m : Nat) (w' β : ℝ) : Nat → ℝ → ℝ
  | 0,     A => A
  | k + 1, A => reluDenseOutIter M m w' β k (reluDenseOut M m w' β A)

theorem reluDenseOutIter_nonneg (M : FloatModel) {m : Nat} {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) :
    ∀ (k : Nat) {A : ℝ}, 0 ≤ A → 0 ≤ reluDenseOutIter M m w' β k A
  | 0,     _, hA => hA
  | k + 1, _, hA =>
      reluDenseOutIter_nonneg M hw' hβ k (reluDenseOut_nonneg M hw' hβ hA)

/-- **A relu∘dense tower as a `LayerChain`, with no `hfit`.** The v2 peer of
    `Cifar8ChainCert.reluDenseTower`, whose signature carries
    `hfit : layerAct + layerBudget(0) ≤ A` — the hypothesis
    `cifar8_stage_defeats_hfit` shows is contradictory at the committed net.
    Here the windows simply grow along the chain. -/
noncomputable def reluDenseTowerH (M : FloatModel) {m : Nat} {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hm : 0 < m) :
    (ds : List (BoundedDenseH m w' β)) → {A : ℝ} → 0 ≤ A →
      LayerChain m A m (reluDenseOutIter M m w' β ds.length A)
  | [],      _, _  => .nil
  | d :: ds, _, hA =>
      .cons (layerCertH_reluDense M d.W d.b hw' hβ hA hm d.hW d.hb)
            (reluDenseTowerH M hw' hβ hm ds (reluDenseOut_nonneg M hw' hβ hA))

/-- **The v2 chain admits exactly what v1 rejects.** A 12-stage relu∘dense tower
    at the magnitudes of `cifar8Verified`'s widest stage — fan-in `m = 288`,
    `|W| ≤ 1/2`, `|b| ≤ 1`, unit input window — builds with no side condition.
    `cifar8_stage_defeats_hfit` shows the v1 constructor cannot accept a single
    layer at these numbers, for any positive window. -/
noncomputable example (M : FloatModel) (ds : List (BoundedDenseH 288 (1/2) 1))
    (h12 : ds.length = 12) :
    LayerChain 288 1 288 (reluDenseOutIter M 288 (1/2) 1 12 1) := by
  have := reluDenseTowerH M (m := 288) (w' := 1/2) (β := 1)
    (by norm_num) (by norm_num) (by norm_num) ds (A := 1) (by norm_num)
  rwa [h12] at this

end Proofs
