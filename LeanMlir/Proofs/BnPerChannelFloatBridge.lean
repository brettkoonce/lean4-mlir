import LeanMlir.Proofs.ViTBlockFloatBridge
import LeanMlir.Proofs.Foundation.PerChannelBN

/-! # ℝ→Float32 bridge: BatchNorm in `FloatBridges` form (the per-channel keystone)

`floatClose_bn` (`FloatComposeBridge.lean`) certifies **one** BatchNorm forward as
`FloatClose` *given* operating-point data: float mean/inv-stddev `fμ`/`fistdv` close to
the true stats, the centered-deviation bound `D`, and the inv-stddev bound `S`. This
file packages that into the existential **`FloatBridges`** form that whole-net `.comp`
assembly consumes — discharging the two *generic* operating-point facts
(`D = 2A` from `bnMean_abs_le`; `S = 1/√ε` from `bnVar ≥ 0`) so the only remaining
inputs are the *supplied* float-stat accuracy moduli `emean`/`eistd` (exactly as
`fexp`/`fsig`/`fistd` are supplied throughout — `rsqrt`/`exp` have no IEEE spec, so the
float stats are necessarily modelled, not derived).

Three rungs, each consumed by the net forwards:

* `floatBridges_bn` — flat/global BN. Directly discharges the `FloatBridges (bnForward …)`
  hypotheses that `floatBridges_mbconvBody` (EfficientNet) defers (`hbnE/hbnD/hbnP`).
* `floatBridges_bnPerChannelFlat` — the block-diagonal per-channel lift via
  `FloatClose.perRowIdx` (`bnPerChannelFlat = perRowIdxFlat` *definitionally*, both being
  `Mat.flatten ∘ (per-row bnForward) ∘ Mat.unflatten`). Uniform budget across channels
  from uniform `G`/`Bbnd` bounds.
* `floatBridges_bnPerChannelTensor3` — conjugated to the network's Tensor3 activation
  layout by the `reassocFwd`/`reassocBack` permutations (`= gather E` / `gather E.symm`
  for the re-association `Equiv`). The BatchNorm op the CIFAR-BN and ResNet-34 forwards
  actually contain.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The two generic operating-point facts (any magnitude domain `A`)
-- ════════════════════════════════════════════════════════════════

/-- **Generic centered-deviation bound.** On inputs within magnitude `A`, every
    coordinate is within `2A` of the batch mean (`|xⱼ| ≤ A`, `|μ| ≤ A` by
    `bnMean_abs_le`). Discharges `floatClose_bn`'s `hD` with `D := 2A`. -/
theorem bn_centered_le {m : Nat} {A : ℝ} (hm : 0 < m) (v : Vec m)
    (hv : ∀ k, |v k| ≤ A) (j : Fin m) : |v j - bnMean m v| ≤ 2 * A := by
  have hμ : |bnMean m v| ≤ A := bnMean_abs_le v hm hv
  have htri : |v j - bnMean m v| ≤ |v j| + |bnMean m v| := by
    rw [sub_eq_add_neg, ← abs_neg (bnMean m v)]; exact abs_add_le _ _
  linarith [hv j, hμ, htri]

-- ════════════════════════════════════════════════════════════════
-- § Rung 1: flat/global BatchNorm float-bridges
-- ════════════════════════════════════════════════════════════════

/-- **Flat/global BatchNorm float-bridges.** For any input magnitude `A` there is an
    output magnitude and a `FloatClose` certificate (the deployed float BN within an
    explicit modulus of the certified `bnForward`). The centered bound (`2A`) and the
    inv-stddev bound (`1/√ε`) are discharged generically; the supplied obligations are
    just the float-stat accuracy moduli `emean`/`eistd` (which `bnMean_close` and
    `bnVar_close`+`bnIstd_close` discharge at instantiation). This is the form
    `floatBridges_mbconvBody` (EfficientNet) takes as `hbnE`/`hbnD`/`hbnP`. -/
theorem floatBridges_bn {m : Nat} (M : FloatModel) {ε γ β : ℝ}
    (fμ fistdv : Vec m → ℝ) (emean eistd : ℝ → ℝ)
    (hm : 0 < m) (hε : 0 < ε)
    (hmean : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean A)
    (histd : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd A) :
    FloatBridges (bnForward m ε γ β) := by
  intro A hA
  have hfc := floatClose_bn M fμ fistdv hm hε (le_refl |γ|) (le_refl |β|)
    (fun v hv => hmean A hA v hv) (fun v hv => histd A hA v hv)
    (fun v hv j => bn_centered_le hm v hv j) (fun v _ => bnIstd_abs_le v hε)
  exact ⟨_, _, _, hfc.cod_nonneg hA hm, hfc⟩

-- ════════════════════════════════════════════════════════════════
-- § Rung 2: per-channel BatchNorm (Mat-split layout) via perRowIdx
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel BatchNorm float-bridges** (Mat-split flat layout `oc·m`). The
    block-diagonal lift of `floatBridges_bn`: each channel `c` runs its own BN with
    `(γ c, β c)` and supplied stats `fμ c`/`fistdv c`, all sharing a uniform budget
    (bounds `G`/`Bbnd`, moduli `emean`/`eistd`). Because
    `bnPerChannelFlat = perRowIdxFlat oc m (fun c => bnForward m ε (γ c) (β c))`
    definitionally, this is exactly `FloatClose.perRowIdx` of the per-channel
    `floatClose_bn`. -/
theorem floatBridges_bnPerChannelFlat {oc m : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec m → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd : ℝ}
    (hoc : 0 < oc) (hm : 0 < m) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fμ c v - bnMean m v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fistdv c v - bnIstd m v ε| ≤ eistd A) :
    FloatBridges (bnPerChannelFlat oc m ε γ β) := by
  intro A hA
  have hg := fun c : Fin oc => floatClose_bn M (fμ c) (fistdv c) hm hε (hγ c) (hβ c)
    (fun v hv => hmean c A hA v hv) (fun v hv => histd c A hA v hv)
    (fun v hv j => bn_centered_le hm v hv j) (fun v _ => bnIstd_abs_le v hε)
  have hpr := FloatClose.perRowIdx (d := m) oc hg
  exact ⟨_, _, _, hpr.cod_nonneg hA (Nat.mul_pos hoc hm), hpr⟩

-- ════════════════════════════════════════════════════════════════
-- § Rung 3: per-channel BatchNorm in the network Tensor3 layout
-- ════════════════════════════════════════════════════════════════

/-- The Tensor3 `(oc·h)·w` ↔ Mat-split `oc·(h·w)` re-association as an `Equiv` — the
    two index maps are mutual inverses (`reassocFwdIdx_reassocBackIdx`,
    `reassocBackIdx_reassocFwdIdx`), so this is a genuine relabeling. -/
noncomputable def reassocEquiv (oc h w : Nat) : Fin (oc * (h * w)) ≃ Fin (oc * h * w) where
  toFun := reassocFwdIdx oc h w
  invFun := reassocBackIdx oc h w
  left_inv := reassocBackIdx_reassocFwdIdx oc h w
  right_inv := reassocFwdIdx_reassocBackIdx oc h w

/-- **Per-channel BatchNorm (network Tensor3 layout) float-bridges.** Conjugate the
    Mat-split `floatBridges_bnPerChannelFlat` by the layout permutations
    `reassocFwd = gather (reassocEquiv …)` and `reassocBack = gather (reassocEquiv …).symm`
    (each `floatBridges_gather`, modulus `id`, magnitude-stable) via `FloatBridges.comp`.
    This is the BatchNorm op the CIFAR-BN and ResNet-34 forwards actually contain
    (`bnPerChannelTensor3 = reassocBack ∘ bnPerChannelFlat ∘ reassocFwd`). -/
theorem floatBridges_bnPerChannelTensor3 {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (h * w) → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd : ℝ}
    (hoc : 0 < oc) (hhw : 0 < h * w) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (h * w) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (h * w) v ε| ≤ eistd A) :
    FloatBridges (bnPerChannelTensor3 oc h w ε γ β) := by
  have hpc := floatBridges_bnPerChannelFlat M γ β fμ fistdv emean eistd hoc hhw hε hγ hβ
    hmean histd
  exact ((floatBridges_gather (reassocEquiv oc h w)).comp hpc).comp
    (floatBridges_gather (reassocEquiv oc h w).symm)

end Proofs
