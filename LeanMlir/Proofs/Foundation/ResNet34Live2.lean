import LeanMlir.Proofs.Foundation.ResNet34
import LeanMlir.Proofs.JacobianSeal

/-!
# Toward a live ResNet-34 witness — Stage 2: the channel-order invariant kit

`ResNet34Live.lean` (Stage 1) banked `liveDown` — a signal-carrying strided
downsample — but found `liveFwd` is still **constant-output**: a 1-channel net
with BN-before-GAP is necessarily constant (`planning/whole_network_backward.md`
Item A). The escape requires ≥2 channels.

This file banks the **non-vacuity mechanism** for the 2-channel rebuild (Item A2).
The earlier plan was to track a channel *deviation sum* through the maxpool —
hard, because maxpool's argmax is input-dependent. The cleaner carrier proved
here is a **pointwise strict channel-order invariant**:

> maintain `forward(z) channel 1 < forward(z) channel 0` at *every* spatial
> position, through every layer.

Each ResNet layer preserves it, and *strict* order at the output gives
`forward X ≠ forward 0` directly — no deviation bookkeeping, no maxpool-argmax
computation. The maxpool — the doc's "crux" — is just **"max preserves strict
pointwise order"** (`maxPool2_chan_lt`): if channel 0 dominates channel 1
everywhere, the window-maxes inherit the strict domination
(`max ch0 ≥ ch0(argmax ch1) > ch1(argmax ch1) = max ch1`). Scalar BN (γ=1)
preserves it because `bn(z)k₀ − bn(z)k₁ = (z k₀ − z k₁)·istd` with `istd > 0`
(`bnForward_chan_lt`); ReLU preserves it in the kept-positive region
(`relu_chan_lt`); decimate / identity-block `+const` preserve it trivially.

These are reusable, dimension-generic, and 3-axiom clean. The 2-channel layer
rebuild (stem / maxpool / `liveDownPC` / `idBlk` at `c = 2`) and the final
assembly remain (multi-session).
-/

namespace Proofs
namespace ResNet34Live2

-- ════════════════════════════════════════════════════════════════
-- § The maxpool crux: max preserves strict pointwise channel order
-- ════════════════════════════════════════════════════════════════

/-- **Maxpool preserves a strict per-position channel domination.** If channel
    `ci` strictly exceeds channel `cj` at *every* input position, then after the
    2×2 max-pool the same holds at every output position — each window's max for
    `ci` dominates the same window's max for `cj`
    (`max ci ≥ ci(argmax cj) > cj(argmax cj) = max cj`). This is the maxpool
    half of the channel-order invariant, with no argmax tracking. -/
theorem maxPool2_chan_lt {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) (ci cj : Fin c)
    (hlt : ∀ (a : Fin (2 * h)) (b : Fin (2 * w)), x cj a b < x ci a b)
    (hi : Fin h) (wi : Fin w) :
    maxPool2 x cj hi wi < maxPool2 x ci hi wi := by
  simp only [maxPool2]
  refine max_lt (max_lt ?_ ?_) (max_lt ?_ ?_)
  · exact lt_of_lt_of_le (hlt _ _) (le_max_of_le_left (le_max_left _ _))
  · exact lt_of_lt_of_le (hlt _ _) (le_max_of_le_left (le_max_right _ _))
  · exact lt_of_lt_of_le (hlt _ _) (le_max_of_le_right (le_max_left _ _))
  · exact lt_of_lt_of_le (hlt _ _) (le_max_of_le_right (le_max_right _ _))

-- ════════════════════════════════════════════════════════════════
-- § Scalar BN and ReLU preserve the order invariant
-- ════════════════════════════════════════════════════════════════

/-- **Scalar BN (γ=1) preserves strict order between any two coordinates.** It is
    affine in the centered input — `bn(z)k = (z k − μ)·istd + β` — and `istd > 0`,
    so the difference between two coordinates is `(z k₀ − z k₁)·istd`, sign-preserving.
    (Holds for arbitrary index pairs; we apply it at corresponding channel positions.) -/
theorem bnForward_chan_lt {n : Nat} (ε β : ℝ) (hε : 0 < ε) (z : Vec n) (k₀ k₁ : Fin n)
    (h : z k₁ < z k₀) :
    bnForward n ε 1 β z k₁ < bnForward n ε 1 β z k₀ := by
  have hist : 0 < bnIstd n z ε := bnIstd_pos z ε hε
  simp only [bnForward, bnXhat, one_mul]
  have hmul : (z k₁ - bnMean n z) * bnIstd n z ε < (z k₀ - bnMean n z) * bnIstd n z ε :=
    mul_lt_mul_of_pos_right (by linarith) hist
  linarith

/-- ReLU is the identity on a strictly-positive coordinate. -/
theorem relu_pos_eq {n : Nat} (z : Vec n) (k : Fin n) (h : 0 < z k) :
    relu n z k = z k := by
  simp only [relu]; rw [if_pos h]

/-- **ReLU preserves strict order in the kept-positive region** — where the net is
    smooth (the invariant maintains positivity alongside the order). -/
theorem relu_chan_lt {n : Nat} (z : Vec n) (k₀ k₁ : Fin n) (h₁ : 0 < z k₁) (h : z k₁ < z k₀) :
    relu n z k₁ < relu n z k₀ := by
  rw [relu_pos_eq z k₁ h₁, relu_pos_eq z k₀ (lt_trans h₁ h)]; exact h

end ResNet34Live2
end Proofs
