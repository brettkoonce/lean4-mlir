import LeanMlir.Proofs.Float.FloatBridge

/-! # `FloatClose` — the float tier's one closeness form, and how it composes

On inputs within magnitude `A`, the float `fF` is within an error modulus `L e` of the real `f`
(per coordinate, at input error `e`), and both outputs are within `B`, so the next layer's
magnitude precondition holds. `FloatClose.comp` composes two (moduli compose, magnitudes thread),
so a whole-net bound would be `.comp` folded over the layer list; no whole-net `FloatClose` is
assembled in the repo. `of_close` builds one from
a per-op `*_close` budget; `relu`, `id` and `iterate` are the generic instances. The per-op
instances for the conv-net op set are in `FloatComposeBridge`.
-/

namespace Proofs

open FloatModel

/-- A-posteriori-magnitude, proved-error float closeness, built to compose.
    `A` bounds the inputs (both real `va` and float `vt`), `B` both outputs;
    `L` is the input-error → output-error modulus. -/
def FloatClose {m n : Nat} (A B : ℝ) (f fF : Vec m → Vec n) (L : ℝ → ℝ) : Prop :=
  (∀ v, (∀ k, |v k| ≤ A) → ∀ i, |f v i| ≤ B ∧ |fF v i| ≤ B) ∧
  (∀ vt va e, (∀ k, |va k| ≤ A) → (∀ k, |vt k| ≤ A) → (∀ k, |vt k - va k| ≤ e)
      → ∀ i, |fF vt i - f va i| ≤ L e)

/-- **Float-closeness composes.** Magnitudes thread `A → B → C`, error moduli
    compose `Lg ∘ Lf`. -/
theorem FloatClose.comp {m n p : Nat} {A B C : ℝ}
    {f fF : Vec m → Vec n} {g gF : Vec n → Vec p} {Lf Lg : ℝ → ℝ}
    (hf : FloatClose A B f fF Lf) (hg : FloatClose B C g gF Lg) :
    FloatClose A C (g ∘ f) (gF ∘ fF) (Lg ∘ Lf) := by
  obtain ⟨hfm, hfe⟩ := hf
  obtain ⟨hgm, hge⟩ := hg
  refine ⟨?_, ?_⟩
  · intro v hv i
    exact ⟨(hgm (f v) (fun k => (hfm v hv k).1) i).1,
           (hgm (fF v) (fun k => (hfm v hv k).2) i).2⟩
  · intro vt va e hva hvt hd i
    exact hge (fF vt) (f va) (Lf e)
      (fun k => (hfm va hva k).1) (fun k => (hfm vt hvt k).2)
      (fun k => hfe vt va e hva hvt hd k) i

/-- **The standard way to build a `FloatClose` instance.** A real bound `R` on the box, a
    rounding bound `E` at an exactly-represented input, and the error modulus give
    `FloatClose A (R + E)`: the float output is within `E` of the real one, so its magnitude is
    at most `R + E`. Every per-op instance below with a fresh-input rounding term is this. -/
theorem FloatClose.of_close {m n : Nat} {A R E : ℝ} {f fF : Vec m → Vec n} {L : ℝ → ℝ}
    (hreal : ∀ v, (∀ k, |v k| ≤ A) → ∀ i, |f v i| ≤ R)
    (hround : ∀ v, (∀ k, |v k| ≤ A) → ∀ i, |fF v i - f v i| ≤ E)
    (herr : ∀ vt va e, (∀ k, |va k| ≤ A) → (∀ k, |vt k| ≤ A) → (∀ k, |vt k - va k| ≤ e)
      → ∀ i, |fF vt i - f va i| ≤ L e) :
    FloatClose A (R + E) f fF L := by
  refine ⟨fun v hv i => ?_, herr⟩
  have h1 := hreal v hv i; have h2 := hround v hv i
  have h3 := abs_sub_abs_le_abs_sub (fF v i) (f v i)
  exact ⟨by linarith [abs_nonneg (fF v i - f v i)], by linarith⟩

/-- **ReLU is `FloatClose` with modulus `id`** — exact in float (real = float map),
    1-Lipschitz on the inherited error, never grows magnitudes. -/
theorem floatClose_relu {n : Nat} (A : ℝ) :
    FloatClose A A (relu n) (relu n) (fun e => e) := by
  refine ⟨fun v hv i => ⟨(relu_abs_le v i).trans (hv i), (relu_abs_le v i).trans (hv i)⟩,
          fun vt va e _ _ hd i => relu_close vt va e hd i⟩

/-- The identity map is `FloatClose` (modulus `id`). -/
theorem floatClose_id {m : Nat} (A : ℝ) :
    FloatClose A A (id : Vec m → Vec m) (id : Vec m → Vec m) (id : ℝ → ℝ) :=
  ⟨fun _v hv i => ⟨hv i, hv i⟩, fun _vt _va _e _ _ hd i => hd i⟩

/-- **A magnitude-stable block iterated `n` times is `FloatClose`.**
    A dim-preserving block that is `FloatClose A A f fF L` (inputs and outputs
    within the same bound `A`, taken as a hypothesis) composes with itself to any
    depth: `f^[n]` is `FloatClose A A` with modulus `L^[n]`. -/
theorem floatClose_iterate {m : Nat} {A : ℝ} {f fF : Vec m → Vec m} {L : ℝ → ℝ}
    (hf : FloatClose A A f fF L) (n : ℕ) :
    FloatClose A A (f^[n]) (fF^[n]) (L^[n]) := by
  induction n with
  | zero => simpa using floatClose_id A
  | succ k ih =>
      rw [Function.iterate_succ', Function.iterate_succ', Function.iterate_succ']
      exact ih.comp hf

end Proofs
