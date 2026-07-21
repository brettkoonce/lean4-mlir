import LeanMlir.Proofs.Codegen.AdjointChainBridge

/-! # The residual combinator: partitioned budgets for parallel-path blocks

Probe §6 (EfficientNet/SE) and §8 (ViT/attention) hit the same wall: a block
with a parallel path (SE gate, attention scores) forces the within-block fold
to push an already-amplified inherited error through a nonlinear modulus
(`σ`, `e^{2δ}−1`), which is vacuous once the error reaches the exponent scale.
The cure is chain granularity FINER than the block: cut at the branch
intermediate (scores/gate logits) and carry the residual stream alongside it
as one widened state vector — parallel paths become ordinary chain stages.

Widened states embed in the uniform-width chain by zero-padding, so the only
genuinely new content is **budget bookkeeping by coordinate part**: the
carried stream is an EXACT pass-through (fresh budget 0) while the computed
branch part pays its own fresh budget, and the two parts meet the tail
through SEPARATE gains. That is `chain2_adjointClose`:

    |chainF x − chainR x| ≤ Σᵢ (H₁ᵢ·b₁ᵢ + H₂ᵢ·b₂ᵢ)

with each stage's output coordinates split by a predicate `P` (part 1 = the
computed branch, part 2 = the carried stream, or any other split), per-part
fresh budgets, and per-part tail gains (`LipOnWindow2`, the block-row-sum
refinement of `LipOnWindow`). Downstream softmax/σ then meets the branch
error through the MEASURED tail gain (softmax Jacobian is L∞-bounded by ½),
never through its own exponential modulus. Same telescoping, proven once by
induction; `LayerCert2.stream` certifies the pass-through part at `b₂ = 0`
definitionally, and single-part stages embed via `LayerCert2.ofLayerCert`.
-/

namespace Proofs

open FloatModel

/-- Two-part windowed Lipschitz gain: an input spread `e₁` on the `P`
    coordinates and `e₂` off `P` amplify to at most `H₁·e₁ + H₂·e₂` per
    output coordinate — the block-row-sum refinement of `LipOnWindow`. -/
def LipOnWindow2 {m n : Nat} (A H₁ H₂ : ℝ) (P : Fin m → Prop)
    (f : Vec m → Vec n) : Prop :=
  ∀ (u v : Vec m) (e₁ e₂ : ℝ), 0 ≤ e₁ → 0 ≤ e₂ →
    (∀ k, |u k| ≤ A) → (∀ k, |v k| ≤ A) →
    (∀ k, P k → |u k - v k| ≤ e₁) → (∀ k, ¬P k → |u k - v k| ≤ e₂) →
    ∀ j, |f u j - f v j| ≤ H₁ * e₁ + H₂ * e₂

/-- Any plain windowed gain is a two-part gain with both parts at `H`. -/
theorem LipOnWindow.toTwoPart {m n : Nat} {A H : ℝ} {f : Vec m → Vec n}
    (h : LipOnWindow A H f) (_hH : 0 ≤ H) (P : Fin m → Prop) :
    LipOnWindow2 A H H P f := by
  intro u v e₁ e₂ he₁ he₂ hu hv hd₁ hd₂ j
  have hd : ∀ k, |u k - v k| ≤ e₁ + e₂ := fun k => by
    by_cases hP : P k
    · exact (hd₁ k hP).trans (by linarith)
    · exact (hd₂ k hP).trans (by linarith)
  calc |f u j - f v j| ≤ H * (e₁ + e₂) :=
        h u v (e₁ + e₂) (by linarith) hu hv hd j
    _ = H * e₁ + H * e₂ := by ring

/-- A chain layer with PARTITIONED fresh budgets: output coordinates split by
    `P` (part 1 = a computed branch, part 2 = e.g. the carried residual
    stream), each part with its own fresh budget. -/
structure LayerCert2 (m : Nat) (A : ℝ) where
  f : Vec m → Vec m
  fF : Vec m → Vec m
  P : Fin m → Prop
  b₁ : ℝ
  b₂ : ℝ
  b₁_nonneg : 0 ≤ b₁
  b₂_nonneg : 0 ≤ b₂
  window : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |f v j| ≤ A ∧ |fF v j| ≤ A
  fresh₁ : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, P j → |fF v j - f v j| ≤ b₁
  fresh₂ : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, ¬P j → |fF v j - f v j| ≤ b₂

/-- The real chain (head applied first). -/
noncomputable def chainR2 {m : Nat} {A : ℝ} : List (LayerCert2 m A) → Vec m → Vec m
  | [] => id
  | l :: ls => chainR2 ls ∘ l.f

/-- The float chain. -/
noncomputable def chainF2 {m : Nat} {A : ℝ} : List (LayerCert2 m A) → Vec m → Vec m
  | [] => id
  | l :: ls => chainF2 ls ∘ l.fF

/-- Per-position tail-gain PAIRS: `(H₁, H₂)` bounds the real suffix's
    two-part gain with respect to the head stage's output partition. -/
def TailGains2 {m : Nat} {A : ℝ} : List (LayerCert2 m A) → List (ℝ × ℝ) → Prop
  | [], [] => True
  | l :: ls, Hp :: Hs =>
      LipOnWindow2 A Hp.1 Hp.2 l.P (chainR2 (A := A) ls) ∧ TailGains2 ls Hs
  | _, _ => False

/-- The partitioned chain budget: `Σᵢ (H₁ᵢ·b₁ᵢ + H₂ᵢ·b₂ᵢ)` — an exact
    pass-through part (`b₂ = 0`) contributes nothing regardless of its gain. -/
noncomputable def chainBudget2 {m : Nat} {A : ℝ} :
    List (LayerCert2 m A) → List (ℝ × ℝ) → ℝ
  | l :: ls, Hp :: Hs => Hp.1 * l.b₁ + Hp.2 * l.b₂ + chainBudget2 ls Hs
  | _, _ => 0

/-- **THE RESIDUAL COMBINATOR: partitioned depth-linear chain certificate.**
    Same exact telescoping as `chain_adjointClose`, with each stage's fresh
    error split by its output partition and pushed through the tail's
    two-part gain. Carried streams ride at zero cost; branch nonlinearities
    (softmax, σ) meet inherited error only through the tail gains. -/
theorem chain2_adjointClose {m : Nat} {A : ℝ}
    (ls : List (LayerCert2 m A)) (Hs : List (ℝ × ℝ)) (hH : TailGains2 ls Hs)
    (x : Vec m) (hx : ∀ k, |x k| ≤ A) (j : Fin m) :
    |chainF2 ls x j - chainR2 ls x j| ≤ chainBudget2 ls Hs := by
  induction ls generalizing Hs x with
  | nil =>
    cases Hs with
    | nil => simp [chainF2, chainR2, chainBudget2]
    | cons Hp Hs => exact (hH : False).elim
  | cons l ls ih =>
    cases Hs with
    | nil => exact (hH : False).elim
    | cons Hp Hs =>
      obtain ⟨hHl, hHs⟩ := hH
      have hz := l.window x hx
      have h1 : |chainF2 ls (l.fF x) j - chainR2 ls (l.fF x) j|
          ≤ chainBudget2 ls Hs := ih Hs hHs (l.fF x) (fun k => (hz k).2)
      have h2 : |chainR2 ls (l.fF x) j - chainR2 ls (l.f x) j|
          ≤ Hp.1 * l.b₁ + Hp.2 * l.b₂ :=
        hHl (l.fF x) (l.f x) l.b₁ l.b₂ l.b₁_nonneg l.b₂_nonneg
          (fun k => (hz k).2) (fun k => (hz k).1)
          (fun k hk => l.fresh₁ x hx k hk) (fun k hk => l.fresh₂ x hx k hk) j
      calc |chainF2 (l :: ls) x j - chainR2 (l :: ls) x j|
          = |chainF2 ls (l.fF x) j - chainR2 ls (l.f x) j| := rfl
        _ ≤ |chainF2 ls (l.fF x) j - chainR2 ls (l.fF x) j|
            + |chainR2 ls (l.fF x) j - chainR2 ls (l.f x) j| := abs_sub_le _ _ _
        _ ≤ chainBudget2 ls Hs + (Hp.1 * l.b₁ + Hp.2 * l.b₂) := add_le_add h1 h2
        _ = Hp.1 * l.b₁ + Hp.2 * l.b₂ + chainBudget2 ls Hs := by ring

/-- Every v1 chain layer is a partitioned layer (everything in part 1) —
    `chain2_adjointClose` strictly generalizes `chain_adjointClose`. -/
noncomputable def LayerCert2.ofLayerCert {m : Nat} {A : ℝ}
    (l : LayerCert m A) : LayerCert2 m A where
  f := l.f
  fF := l.fF
  P := fun _ => True
  b₁ := l.b
  b₂ := 0
  b₁_nonneg := l.b_nonneg
  b₂_nonneg := le_refl 0
  window := l.window
  fresh₁ := fun v hv j _ => l.fresh v hv j
  fresh₂ := fun _v _hv _j hj => absurd trivial hj

/-- **The residual-stream constructor**: a stage whose off-`P` coordinates
    are an EXACT pass-through (float text = real text, e.g. the carried
    stream `h` riding beside freshly computed scores/gate logits) certifies
    with `b₂ = 0` definitionally — the stream costs nothing; only the branch
    part pays, and it pays into the tail gain rather than a nonlinear
    modulus. -/
noncomputable def LayerCert2.stream {m : Nat} {A : ℝ}
    (f fF : Vec m → Vec m) (P : Fin m → Prop) (b₁ : ℝ) (hb₁ : 0 ≤ b₁)
    (hcopy : ∀ v j, ¬P j → fF v j = f v j)
    (hwin : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |f v j| ≤ A ∧ |fF v j| ≤ A)
    (hfresh : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, P j → |fF v j - f v j| ≤ b₁) :
    LayerCert2 m A where
  f := f
  fF := fF
  P := P
  b₁ := b₁
  b₂ := 0
  b₁_nonneg := hb₁
  b₂_nonneg := le_refl 0
  window := hwin
  fresh₁ := hfresh
  fresh₂ := fun v _hv j hj => by rw [hcopy v j hj, sub_self, abs_zero]

end Proofs
