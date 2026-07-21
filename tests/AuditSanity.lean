import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Foundation.MLP

/-!
# Audit sanity examples

Concrete pinning of the headline theorems. If a theorem statement
drifts, one of these examples will stop elaborating.
-/

open Proofs
open Finset BigOperators Classical

namespace ProofsAuditSanity

-- pdiv on identity ─────────────────────────────────────────────────
example : pdiv (fun v : Vec 2 => v) ![3, 7] 0 0 = 1 := by
  rw [pdiv_id]; simp

example : pdiv (fun v : Vec 2 => v) ![3, 7] 0 1 = 0 := by
  rw [pdiv_id]
  show (if (0 : Fin 2) = 1 then (1 : ℝ) else 0) = 0
  rw [if_neg (by intro h; exact absurd h (by decide))]

-- pdiv on constant ─────────────────────────────────────────────────
example : pdiv (fun _ : Vec 2 => (![5, 9] : Vec 2)) ![3, 7] 0 0 = 0 := by
  rw [pdiv_const]

-- pdivMat on transpose (concrete 2×3) ──────────────────────────────
example (A : Mat 2 3) : pdivMat Mat.transpose A 0 1 1 0 = 1 := by
  rw [pdivMat_transpose]; simp

example (A : Mat 2 3) : pdivMat Mat.transpose A 0 1 0 1 = 0 := by
  rw [pdivMat_transpose]
  -- Indices don't all match: j = 0 vs k = 1, condition `j = k ∧ i = l` is false.
  rw [if_neg (by intro h; exact absurd h.1 (by decide))]

-- HasVJP on dense gives the documented backward ────────────────────
-- (definitionally — this confirms that for `dense`, `correct` is NOT
-- the vacuous self-equality pattern.)
example (W : Mat 2 3) (b : Vec 3) (x : Vec 2) (dy : Vec 3) :
    (dense_has_vjp W b).backward x dy = Mat.mulVec W dy := rfl

-- pdiv_dense yields the W entry ────────────────────────────────────
example (W : Mat 2 3) (b : Vec 3) (x : Vec 2) :
    pdiv (dense W b) x 1 2 = W 1 2 := by rw [pdiv_dense]

-- ReLU at a smooth point gives the codegen formula ─────────────────
example : (relu_has_vjp 2).backward ![3, -1] ![5, 7] 0 = 5 := by
  show ∑ j : Fin 2, pdiv (relu 2) ![3, -1] 0 j * (![5, 7] : Vec 2) j = 5
  -- Use the bridge: at smooth points this collapses to `if x 0 > 0 then dy 0 else 0`.
  have h_smooth : ∀ k : Fin 2, (![3, -1] : Vec 2) k ≠ 0 := by
    intro k
    fin_cases k
    · show (3 : ℝ) ≠ 0; norm_num
    · show (-1 : ℝ) ≠ 0; norm_num
  -- Inline the bridge proof so this example doesn't depend on AuditBridge.lean.
  simp_rw [pdiv_relu 2 ![3, -1] h_smooth 0]
  rw [Finset.sum_eq_single 0
      (fun j _ hne => by rw [if_neg (Ne.symm hne)]; ring)
      (fun h => absurd (Finset.mem_univ (0 : Fin 2)) h)]
  show (if (0 : Fin 2) = 0 then (if (3 : ℝ) > 0 then (1 : ℝ) else 0) else 0) *
       (![5, 7] : Vec 2) 0 = 5
  rw [if_pos rfl, if_pos (by norm_num : (3 : ℝ) > 0)]
  show (1 : ℝ) * 5 = 5
  ring

-- ReLU at a smooth NEGATIVE coordinate → zero ──────────────────────
example : (relu_has_vjp 2).backward ![3, -1] ![5, 7] 1 = 0 := by
  show ∑ j : Fin 2, pdiv (relu 2) ![3, -1] 1 j * (![5, 7] : Vec 2) j = 0
  have h_smooth : ∀ k : Fin 2, (![3, -1] : Vec 2) k ≠ 0 := by
    intro k
    fin_cases k
    · show (3 : ℝ) ≠ 0; norm_num
    · show (-1 : ℝ) ≠ 0; norm_num
  simp_rw [pdiv_relu 2 ![3, -1] h_smooth 1]
  rw [Finset.sum_eq_single 1
      (fun j _ hne => by rw [if_neg (Ne.symm hne)]; ring)
      (fun h => absurd (Finset.mem_univ (1 : Fin 2)) h)]
  show (if (1 : Fin 2) = 1 then (if (-1 : ℝ) > 0 then (1 : ℝ) else 0) else 0) *
       (![5, 7] : Vec 2) 1 = 0
  rw [if_pos rfl, if_neg (by norm_num : ¬ ((-1 : ℝ) > 0))]
  show (0 : ℝ) * 7 = 0
  ring

-- pdiv ↔ fderiv equality is definitional (pdiv is a thin wrapper) ──
example (f : Vec 2 → Vec 3) (x : Vec 2) (i : Fin 2) (j : Fin 3) :
    pdiv f x i j = fderiv ℝ f x (basisVec i) j := rfl

-- Vec inherits NormedSpace, NormedAddCommGroup (mathlib Pi-instances) ─
noncomputable example : NormedAddCommGroup (Vec 3) := inferInstance
noncomputable example : NormedSpace ℝ (Vec 3) := inferInstance
noncomputable example : NormedAddCommGroup (Mat 2 3) := inferInstance

-- Edge case: Vec 0 — vacuous, but pdiv exists ─────────────────────
example (x : Vec 0) (i : Fin 0) (j : Fin 0) :
    pdiv (fun y : Vec 0 => y) x i j = if i = j then 1 else 0 :=
  pdiv_id x i j

example (x : Vec 0) (i : Fin 0) : True := by
  exact i.elim0

end ProofsAuditSanity
