import LeanMlir.Proofs.Float.ConvFloat
import LeanMlir.Proofs.Float.BnFloatBridge

/-!
# ℝ→Float32 bridge: global average pooling

The float global-average-pool `gapFlatF` and its budget `gapFlat_close`: GAP is a
per-channel mean (`globalAvgPoolFlat_eq_bnMean`), so `gapFlat_close` reduces to
`bnMean_close` on the channel slice (`sum_s2` flattens the spatial double sum).
The other ResNet-34 ops' float lemmas are elsewhere: the residual add is `add_close`
(`FloatBridge`), the conv is `flatConvF_close` (`ConvFloat`), per-channel BN is
`bnForward_close_of` (`BnFloatBridge`), and the `FloatClose` instances are in
`FloatComposeBridge`.
-/

namespace Proofs

namespace FloatModel

variable (M : FloatModel)

-- ════════════════════════════════════════════════════════════════
-- § Residual additive fan-in
-- ════════════════════════════════════════════════════════════════

-- ════════════════════════════════════════════════════════════════
-- § Global average pool  (a per-channel mean)
-- ════════════════════════════════════════════════════════════════

/-- The float global-average-pool: per channel, the float mean of the channel's
    `h·w` spatial slice (rounded sum, rounded `/(h·w)`). The float peer of
    `globalAvgPoolFlat`. -/
noncomputable def gapFlatF {c h w : Nat} (M : FloatModel) (v : Vec (c * h * w)) : Vec c :=
  fun ci => M.div (M.sum (fun s : Fin (h * w) =>
    Tensor3.unflatten v ci (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2)) ((h * w : ℕ) : ℝ)

/-- **Global-average-pool closeness.** GAP is the per-channel spatial mean, so the
    float GAP is within the `bnMean_close` budget of `globalAvgPoolFlat` per
    channel (`sum_s2` flattens the spatial double sum to the `Fin (h·w)` slice the
    mean rounds). -/
theorem gapFlat_close {c h w : Nat} (M : FloatModel) (v : Vec (c * h * w)) {A : ℝ}
    (hhw : 0 < h * w) (hA : ∀ ci hi wi, |Tensor3.unflatten v ci hi wi| ≤ A) (ci : Fin c) :
    |M.gapFlatF v ci - globalAvgPoolFlat c h w v ci| ≤
      M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A := by
  set cs : Vec (h * w) := fun s =>
    Tensor3.unflatten v ci (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2 with hcs
  have hgap : globalAvgPoolFlat c h w v ci = bnMean (h * w) cs := by
    simp only [globalAvgPoolFlat, globalAvgPool, bnMean]
    congr 1
    · rw [sum_s2 cs]
      refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
      simp only [hcs, Equiv.symm_apply_apply]
    · push_cast; ring
  rw [FloatModel.gapFlatF, ← hcs, hgap]
  exact M.bnMean_close cs hhw (fun i => by rw [hcs]; exact hA _ _ _)

end FloatModel

/-- **GAP as a per-channel `bnMean`.** `globalAvgPoolFlat c h w v ci` is the mean of
    channel `ci`'s spatial slice — `bnMean (h·w)` of the `Fin (h·w)`-indexed gather.
    The reduction `gapFlat_close` performs inline, exposed so the float-bridge
    magnitude/input-shift bounds (`bnMean_abs_le` / `bnMean_input_close`) apply. -/
theorem globalAvgPoolFlat_eq_bnMean {c h w : Nat} (v : Vec (c * h * w)) (ci : Fin c) :
    globalAvgPoolFlat c h w v ci
      = bnMean (h * w) (fun s => Tensor3.unflatten v ci
          (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2) := by
  simp only [globalAvgPoolFlat, globalAvgPool, bnMean]
  congr 1
  · rw [sum_s2 _]
    refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
    simp only [Equiv.symm_apply_apply]
  · push_cast; ring

end Proofs
