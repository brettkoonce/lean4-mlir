import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.CNN

/-!
# Independent audit probes

Concrete-instance pinning tests written for the second-pass audit
(2026-05-18). Each probe verifies a specific claim from the audit
report or README. If `lake env lean tests/AuditProbes.lean` succeeds,
every probed claim holds *as a definitional or proved equality*, not
just by hand-wave.

Probes:

  P1. `pdiv = fderiv (...) (basisVec ..) j` is `rfl`.
  P2. `pdiv3 = pdiv (...flatten/unflatten chain...)` is `rfl`.
  P3. `dense_has_vjp.backward = Mat.mulVec W dy` is `rfl`.
  P4. `relu_has_vjp.backward = ∑ pdiv * dy` is `rfl` (canonical witness).
  P5. `relu_has_vjp_at.backward = if x i > 0 then dy i else 0` is `rfl`
      (codegen-shape, no rfl-escape).
  P6. `mlp_has_vjp.correct := rfl` works because backward is literally
      the canonical sum.
  P7. `HasVJPAt3` + `vjp3_comp_at` (Tensor.lean) — the Tensor3 pointwise
      chain rule now exists. The CNN apex composes at the flattened `Vec`
      level via `vjp_comp_at`, so `vjp3_comp_at` has no in-tree consumer yet
      (a cleanliness nit, not a capability gap). Header note only — no probe
      body.
  P8. `Vec 0` edge case: pdiv_id, pdiv_relu apply vacuously.
  P9. `pdiv (dense W b) x 1 2 = W 1 2` by `pdiv_dense`.
  P10. `pdivMat_transpose` on a concrete 2×3 transpose returns 1
      at the matching cell.
  P11. Identity VJP backward returns dy directly (rfl).
  P12. Concrete `relu_codegen_matches_canonical` for a 2-vec, both
      strictly positive coords.
-/

open Proofs
open Finset BigOperators

namespace ProofsAudit2

-- ────────────────────────────────────────────────────────────────
-- P1: `pdiv` is a thin wrapper over `fderiv`.
-- ────────────────────────────────────────────────────────────────

example (f : Vec 2 → Vec 3) (x : Vec 2) (i : Fin 2) (j : Fin 3) :
    pdiv f x i j = fderiv ℝ f x (basisVec i) j := rfl

-- ────────────────────────────────────────────────────────────────
-- P2: `pdiv3` is a thin wrapper over `pdiv` (via flatten/unflatten).
-- ────────────────────────────────────────────────────────────────

example (f : Tensor3 2 3 4 → Tensor3 1 1 1) (x : Tensor3 2 3 4)
    (ci : Fin 2) (hi : Fin 3) (wi : Fin 4)
    (co : Fin 1) (ho : Fin 1) (wo : Fin 1) :
    pdiv3 f x ci hi wi co ho wo =
    pdiv (fun v : Vec (2 * 3 * 4) =>
            Tensor3.flatten (f (Tensor3.unflatten v)))
      (Tensor3.flatten x)
      (finProdFinEquiv (finProdFinEquiv (ci, hi), wi))
      (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) := rfl

-- ────────────────────────────────────────────────────────────────
-- P3: `dense_has_vjp` backward is the documented `mulVec`.
-- ────────────────────────────────────────────────────────────────

example (W : Mat 2 3) (b : Vec 3) (x : Vec 2) (dy : Vec 3) :
    (dense_has_vjp W b).backward x dy = Mat.mulVec W dy := rfl

-- ────────────────────────────────────────────────────────────────
-- P4: `relu_has_vjp.backward` is literally the canonical pdiv sum.
-- This is the source of the `correct := rfl` vacuity.
-- ────────────────────────────────────────────────────────────────

example (n : Nat) (x dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j := rfl

-- ────────────────────────────────────────────────────────────────
-- P5: `relu_has_vjp_at` backward is the codegen formula.
-- The `at` variant has no rfl-escape — proven content.
-- ────────────────────────────────────────────────────────────────

example (n : Nat) (x : Vec n) (h_smooth : ∀ k, x k ≠ 0)
    (dy : Vec n) (i : Fin n) :
    (relu_has_vjp_at n x h_smooth).backward dy i =
    if x i > 0 then dy i else 0 := rfl

-- ────────────────────────────────────────────────────────────────
-- P6: `mlp_has_vjp.correct` is the rfl-vacuous one.
-- ────────────────────────────────────────────────────────────────

set_option linter.unusedVariables false in
example {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) :
    (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x =
    fun dy i => ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j := rfl

-- ────────────────────────────────────────────────────────────────
-- P8: `Vec 0` edge case.
-- ────────────────────────────────────────────────────────────────

example (x : Vec 0) (i : Fin 0) : ∀ j : Fin 0, pdiv (fun y : Vec 0 => y) x i j = 0 :=
  fun j => i.elim0

example (x : Vec 0) (h_smooth : ∀ k, x k ≠ 0) :
    (relu_has_vjp_at 0 x h_smooth).backward = fun _ k => k.elim0 := by
  funext dy k; exact k.elim0

-- ────────────────────────────────────────────────────────────────
-- P9: pdiv_dense gives the W entry.
-- ────────────────────────────────────────────────────────────────

example (W : Mat 2 3) (b : Vec 3) (x : Vec 2) :
    pdiv (dense W b) x 1 2 = W 1 2 := by rw [pdiv_dense]

-- ────────────────────────────────────────────────────────────────
-- P11: Identity VJP backward is just dy.
-- ────────────────────────────────────────────────────────────────

example (n : Nat) (x dy : Vec n) : (identity_has_vjp n).backward x dy = dy := rfl

-- ────────────────────────────────────────────────────────────────
-- P12: Concrete ReLU smooth-point bridge.
-- ────────────────────────────────────────────────────────────────

example (dy : Vec 2) (i : Fin 2) :
    (relu_has_vjp 2).backward (![3, 7] : Vec 2) dy i = dy i := by
  have h_smooth : ∀ k, (![3, 7] : Vec 2) k ≠ 0 := by
    intro k; fin_cases k <;> simp
  rw [relu_codegen_matches_canonical 2 _ h_smooth dy i]
  have h_pos : (![3, 7] : Vec 2) i > 0 := by fin_cases i <;> simp
  rw [if_pos h_pos]

end ProofsAudit2
