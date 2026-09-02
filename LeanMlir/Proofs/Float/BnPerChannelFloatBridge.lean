import LeanMlir.Proofs.Float.ViTBlockFloatBridge
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

-- ════════════════════════════════════════════════════════════════
-- § The same three rungs with the float map AND the budget NAMED
--   (`FloatBridgesTo` migration — the leaf the whole-net NUMBERS consume)
-- ════════════════════════════════════════════════════════════════

/-! The `FloatBridges` rungs above name neither the float map nor the modulus, so a
whole-net fold built on them cannot state a number (`formalization.yaml` fidelity §4d).
These are their `FloatBridgesTo` peers: `mag`/`mod` written out, so
`FloatBridgesTo.Env` can push a numeric envelope through a BatchNorm the way it pushes
one through a conv.

**The two magnitude constants.** `floatClose_bn` is parameterised by the centered bound
`D` (`|xⱼ − μ| ≤ D`) and the inverse-stddev bound `S` (`|istd| ≤ S`). `D := 2A` is
generic and tight (`bn_centered_le`). `S` is the one place a BN budget can be loose or
conditional, so it stays an ARGUMENT here:

* `S := 1/√ε` — the `ε`-floor, discharged unconditionally by `bnIstd_abs_le`
  (`floatBridgesTo_bnPerChannelTensor3_eps`). Closed, no operating-point hypothesis, and
  the window it propagates is `≈ 2·G·A/√ε` per BN — at `ε = 10⁻⁵` a factor of ~632 per
  site, which is where the whole-net interval fold's size comes from.
* `S := 1/√V` at a variance floor `V ≤ σ²+ε` — the operating-point form
  (`bnIstd_close_at`'s floor, `FloatComposeBridge.lean`'s header). Tighter by
  `(σ²/ε)^{1/2}` per site, at the cost of a named hypothesis about activations that
  nothing in the repo closes. Passing `S` rather than fixing it means that variant is one
  argument away, not a re-proof.

The float mean/inv-stddev stay SUPPLIED (`fμ`/`fistdv` with accuracy moduli
`emean`/`eistd`) for the reason `BnFloatBridge.lean`'s header gives: a GPU `rsqrt` has no
IEEE spec, so the float statistics are modelled, exactly as `fexp`/`fsig` are. -/

/-- The float per-example BN as a MAP: the supplied float mean/inv-stddev evaluated at the
    layer's own input, then `bnForwardF`'s rounded normalize chain. The float peer of
    `bnForward` that `floatClose_bn` certifies. -/
noncomputable def bnForwardFV {m : Nat} (M : FloatModel) (γ β : ℝ) (fμ fistdv : Vec m → ℝ) :
    Vec m → Vec m :=
  fun v => M.bnForwardF γ β (fμ v) (fistdv v) v

/-- The BN leaf's output window at input window `A`: the real `|γ·x̂ + β| ≤ G·(2A·S) + Bbnd`
    plus the normalize chain's rounding (`bnNormBudget`). -/
noncomputable def bnLeafMag (u S G Bbnd : ℝ) (emean eistd : ℝ → ℝ) (A : ℝ) : ℝ :=
  G * (2 * A * S) + Bbnd + bnNormBudget u (2 * A) S G Bbnd (emean A) (eistd A)

/-- The BN leaf's error modulus at input window `A`: `bnReluBudget` — the rounding
    (`bnNormBudget`) plus the real BN's input-sensitivity at inherited error `e`. -/
noncomputable def bnLeafMod (u ε S G Bbnd : ℝ) (emean eistd : ℝ → ℝ) (A e : ℝ) : ℝ :=
  bnReluBudget u (2 * A) S G Bbnd (emean A) (eistd A) A e ε

/-- **Flat/global BatchNorm float-bridges TO its float map** — rung 1 with `mag`/`mod`
    written out. `floatBridges_bn` with the existential opened. -/
noncomputable def floatBridgesTo_bn {m : Nat} (M : FloatModel) {ε γ β : ℝ}
    (fμ fistdv : Vec m → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd S : ℝ}
    (hm : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean A)
    (histd : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd A)
    (hS : ∀ v : Vec m, |bnIstd m v ε| ≤ S) :
    FloatBridgesTo (bnForward m ε γ β) (bnForwardFV M γ β fμ fistdv) :=
  ⟨bnLeafMag M.u S G Bbnd emean eistd, bnLeafMod M.u ε S G Bbnd emean eistd,
   fun A hA =>
     have hfc := floatClose_bn M fμ fistdv hm hε hγ hβ
       (fun v hv => hmean A hA v hv) (fun v hv => histd A hA v hv)
       (fun v hv j => bn_centered_le hm v hv j) (fun v _ => hS v)
     ⟨hfc.cod_nonneg hA hm, hfc⟩⟩

/-- The float per-channel BN (Mat-split flat layout): channel `c` runs the float BN with its
    own `(γ c, β c)` and its own supplied statistics. The float peer of `bnPerChannelFlat`. -/
noncomputable def bnPerChannelFlatFV {oc m : Nat} (M : FloatModel) (γ β : Vec oc)
    (fμ fistdv : Fin oc → Vec m → ℝ) : Vec (oc * m) → Vec (oc * m) :=
  perRowIdxFlat oc m (fun c => bnForwardFV M (γ c) (β c) (fμ c) (fistdv c))

/-- **Per-channel BatchNorm float-bridges TO its float map** (Mat-split layout) — rung 2 with
    `mag`/`mod` written out. The block-diagonal `FloatClose.perRowIdx` lift of the per-channel
    `floatClose_bn`, at the uniform budget the shared `G`/`Bbnd`/`emean`/`eistd`/`S` give. -/
noncomputable def floatBridgesTo_bnPerChannelFlat {oc m : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec m → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd S : ℝ}
    (hoc : 0 < oc) (hm : 0 < m) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fμ c v - bnMean m v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fistdv c v - bnIstd m v ε| ≤ eistd A)
    (hS : ∀ v : Vec m, |bnIstd m v ε| ≤ S) :
    FloatBridgesTo (bnPerChannelFlat oc m ε γ β) (bnPerChannelFlatFV M γ β fμ fistdv) :=
  ⟨bnLeafMag M.u S G Bbnd emean eistd, bnLeafMod M.u ε S G Bbnd emean eistd,
   fun A hA =>
     have hg := fun c : Fin oc => floatClose_bn M (fμ c) (fistdv c) hm hε (hγ c) (hβ c)
       (fun v hv => hmean c A hA v hv) (fun v hv => histd c A hA v hv)
       (fun v hv j => bn_centered_le hm v hv j) (fun v _ => hS v)
     have hpr := FloatClose.perRowIdx (d := m) oc hg
     ⟨hpr.cod_nonneg hA (Nat.mul_pos hoc hm), hpr⟩⟩

/-- The float per-channel BN in the network Tensor3 layout — the same `reassoc` conjugation
    as the real op, with the two layout permutations exact in float. -/
noncomputable def bnPerChannelTensor3FV {oc h w : Nat} (M : FloatModel) (γ β : Vec oc)
    (fμ fistdv : Fin oc → Vec (h * w) → ℝ) : Vec (oc * h * w) → Vec (oc * h * w) :=
  reassocBack oc h w ∘ bnPerChannelFlatFV M γ β fμ fistdv ∘ reassocFwd oc h w

/-- ⭐ **Per-channel BatchNorm (network Tensor3 layout) float-bridges TO its float map** —
    rung 3 with `mag`/`mod` written out, and the leaf every ResNet-34 / MobileNetV2 /
    EfficientNet whole-net NUMBER goes through. `floatBridges_bnPerChannelTensor3`'s
    conjugation, repackaged so the composite's `mag`/`mod` are the BN's own (the two gathers
    are magnitude-stable with modulus `id`, so the composition collapses definitionally). -/
noncomputable def floatBridgesTo_bnPerChannelTensor3 {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (h * w) → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd S : ℝ}
    (hoc : 0 < oc) (hhw : 0 < h * w) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (h * w) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (h * w) v ε| ≤ eistd A)
    (hS : ∀ v : Vec (h * w), |bnIstd (h * w) v ε| ≤ S) :
    FloatBridgesTo (bnPerChannelTensor3 oc h w ε γ β) (bnPerChannelTensor3FV M γ β fμ fistdv) :=
  ⟨bnLeafMag M.u S G Bbnd emean eistd, bnLeafMod M.u ε S G Bbnd emean eistd,
   ((floatBridgesTo_gather (reassocEquiv oc h w)).comp
      (floatBridgesTo_bnPerChannelFlat M γ β fμ fistdv emean eistd hoc hhw hε hγ hβ
        hmean histd hS)).comp
     (floatBridgesTo_gather (reassocEquiv oc h w).symm) |>.close⟩

/-- **The `ε`-floor instantiation** — the BN leaf with `S := 1/√ε`, closed unconditionally
    (`bnIstd_abs_le`). The form the whole-net numbers use: no operating-point hypothesis, at
    the cost of a per-site window factor `2G/√ε`. -/
noncomputable def floatBridgesTo_bnPerChannelTensor3_eps {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (h * w) → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd : ℝ}
    (hoc : 0 < oc) (hhw : 0 < h * w) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (h * w) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (h * w), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (h * w) v ε| ≤ eistd A) :
    FloatBridgesTo (bnPerChannelTensor3 oc h w ε γ β) (bnPerChannelTensor3FV M γ β fμ fistdv) :=
  floatBridgesTo_bnPerChannelTensor3 M γ β fμ fistdv emean eistd hoc hhw hε hγ hβ
    hmean histd (fun v => bnIstd_abs_le v hε)

end Proofs
