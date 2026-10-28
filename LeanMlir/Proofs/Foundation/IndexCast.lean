module

public import LeanMlir.Proofs.Codegen.StableHLO.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # Index casts — relabelling a graph's index along a proved `Nat` equality

`Nat` multiplication is not definitionally associative, so the network's left-associated
`c·h·w` (and `N·(c·h·w)`) and the BatchNorm / channel-LayerNorm ops' `c·(h·w)` are different
types for the same tensor. `castIdx h e` is the same graph typed at the other index; the emitted
text does not change (`h : n = m`, so after `subst` the `skel` is the same; not stated as a lemma
here), and `den_castIdx` says the denotation is read through `Fin.cast`. `la_assoc` is the
batched seam `N·(c·h·w) = N·(c·(h·w))`.

The same reshuffle on plain vectors is `BackLinks.reassocB` (batched, `Batched.BackLinks`) and
`reassocFwd` / `reassocBack` (per example, `PerChannelBN`); `den_reassocS` / `den_unassocS`
(`ConvNeXtChannelLN`) identify the graph casts `reassoc` / `unassoc` with those.
-/

namespace Proofs.StableHLO

/-- **Relabel an AST value's index along a proved equality.** `h ▸ e`: the same graph, typed at
    `m` instead of `n`. The emitted text does not change: `n = m`, so `skel` is unchanged (by
    `subst`; `skel` itself keeps the index). -/
def castIdx {n m : Nat} (h : n = m) (e : SHlo n) : SHlo m := h ▸ e

theorem den_castIdx {n m : Nat} (h : n = m) (e : SHlo n) :
    den (castIdx h e) = fun i => den e (Fin.cast h.symm i) := by
  subst h; rfl

/-- `c·h·w → c·(h·w)`: the `Nat.mul_assoc` relabelling a channel-last transpose needs. -/
def reassoc {c h w : Nat} (e : SHlo (c * h * w)) : SHlo (c * (h * w)) :=
  castIdx (Nat.mul_assoc c h w) e

/-- `c·(h·w) → c·h·w`, the inverse relabelling of `reassoc`. -/
def unassoc {c h w : Nat} (e : SHlo (c * (h * w))) : SHlo (c * h * w) :=
  castIdx (Nat.mul_assoc c h w).symm e

/-- The `mul_assoc` relabelling under `N * ·` — the seam between the network's left-assoc
    `N·(c·h·w)` and the BatchNorm ops' `N·(c·(h·w))`. -/
theorem la_assoc (N oc h w : Nat) : N * (oc * h * w) = N * (oc * (h * w)) :=
  congrArg (N * ·) (Nat.mul_assoc oc h w)

end Proofs.StableHLO
