import LeanMlir.Proofs.MLP

/-!
# Mutation probe — confirms `correct := rfl` pins backward to the canonical sum

When `relu_has_vjp.correct := rfl` succeeds, `backward` is pinned to be
*definitionally* equal to the RHS sum. You cannot replace `backward`
with garbage and have `rfl` still work.

The example below constructs a "garbage" `HasVJP (relu n)` with
`backward := fun _ _ _ => 42` and `correct := rfl`. The `correct` field
should fail to elaborate. We use `#guard_msgs` to assert the failure.
-/

open Proofs Finset BigOperators

/--
error: Type mismatch
  rfl
has type
  ?m.15 = ?m.15
but is expected to have type
  42 = ∑ j, pdiv (relu n) x✝² x✝ j * x✝¹ j
-/
#guard_msgs in
noncomputable example (n : Nat) : HasVJP (relu n) where
  backward _ _ _ := 42
  correct _ _ _ := rfl
