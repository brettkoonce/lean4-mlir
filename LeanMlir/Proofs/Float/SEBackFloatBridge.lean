import LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridge
import LeanMlir.Proofs.Float.BnBackFloatBridge
import LeanMlir.Proofs.Float.EnetFloatBridge

/-! # ℝ→Float32 bridge for the SQUEEZE-EXCITE backward (mnv2 / efficientnet)

A3 (planning/a3_backward_deepnet_assembly.md §1e): the SE block's input-VJP — the
architecturally-distinctive backward op, the **product rule** `y = x ⊙ gate(x)`.

`SE.lean`/`EfficientNetBackB0.lean` certify the structure: the SE input-gradient at a smooth point is
the two-path fan-in (`seBlockFull_has_vjp`, denoted by `seBlockFullBackGraphE`)

  `seBack(dy) = (gate(x) ⊙ dy)            -- main path: the gate is a stop-gradient multiplier`
            ` + gateBack(x ⊙ dy)          -- gate path: the input flows through the gate sub-net,`
            `                                cotangent x⊙dy (NOT dy — the gate multiplies the main)`

— the same fan-in as a residual, but driven by **multiplication** (so each branch sees a different
cotangent). Each path is a `diagBack` (`LinBackFloatBridge`, the saved-vector pointwise scale): the
main path scales `dy` by the saved gate `g = gate(x)`, the gate path pre-scales `dy` by the saved
input `x` before threading `gateBack`. The two add (rounded) — `FloatBridges.biPathSum`. So the
**combinator** `floatBridges_seBack` is the SE analogue of `floatBridges_r34DownBlockBack`'s
two-branch fan-in, with the gate sub-network's backward `gateBack` supplied as a `FloatBridges` fact.

The gate's backward is then **fully assembled** (`floatBridges_seGateBack`): the gate is
`broadcast ∘ sigmoid ∘ dense W₂ ∘ swish ∘ dense W₁ ∘ GAP`, so (reverse order)

  `gateBack = gapBack ∘ linBack W₁ ∘ swishBack ∘ linBack W₂ ∘ sigmoidBack ∘ broadcastBack`

every op already bridged — `gapBack` (`Resnet34WholeBackFloatBridge`), `linBack`
(`LinBackFloatBridge`), the swish/sigmoid `diagBack`s — except the **broadcast adjoint**
`broadcastBack` (sum-over-spatial of each channel's `h·w` cotangent cells, the VJP of the gate's
expand-to-spatial `broadcastFlat`). That one new reduction op is `floatBridges_broadcastBack`,
bridged by the BN-back reduction machinery (`reduction_close`).
-/

namespace Proofs

open scoped Real

-- ════════════════════════════════════════════════════════════════
-- § The broadcast adjoint: sum-over-spatial (the one new reduction op)
-- ════════════════════════════════════════════════════════════════

/-- **Broadcast backward (sum-over-spatial)** — the certified `broadcastFlat` VJP: each channel `k`
    receives the sum of its `h·w` spatial cotangent cells. `Vec (c·h·w) → Vec c`. The adjoint of the
    SE gate's expand-to-spatial `broadcastFlat`. -/
noncomputable def broadcastBackFlat (c h w : Nat) (dy : Vec (c * h * w)) : Vec c :=
  fun k => ∑ idx : Fin (c * h * w), (if flatChannel c h w idx = k then dy idx else 0)

/-- `broadcastBackFlat` is exactly the certified `broadcastFlat` VJP backward (faithful by
    definition — the VJP backward ignores its primal argument). -/
theorem broadcastBackFlat_eq_vjp (c h w : Nat) (v : Vec c) (dy : Vec (c * h * w)) :
    broadcastBackFlat c h w dy = (broadcastFlat_has_vjp c h w).backward v dy := rfl

/-- **The float broadcast backward** — the rounded reduction `fl(Σ masked dy)` per channel (the
    spatial reduce). The float peer of `broadcastBackFlat`. -/
noncomputable def FloatModel.broadcastBackFlatF {c h w : Nat} (M : FloatModel)
    (dy : Vec (c * h * w)) : Vec c :=
  fun k => M.sum (fun idx => if flatChannel c h w idx = k then dy idx else 0)

/-- **Broadcast backward is `FloatClose`** — a per-channel reduction over the `c·h·w` masked
    cotangent (only channel `k`'s `h·w` cells are nonzero). Float `fl(Σ masked)` is within the
    `reduction_close` fan-in budget of the real `Σ masked`; magnitude rides the `(1+u)^(N+1)` Higham
    factor (`N = c·h·w`). The reduction peer of the BN-back `Σ dy`; the one new op the SE gate
    backward needs. -/
theorem floatClose_broadcastBack {c h w : Nat} (M : FloatModel) (A : ℝ) :
    FloatClose A
      ((((c * h * w : ℕ) : ℝ) * A)
        + ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A))
      (broadcastBackFlat c h w) (M.broadcastBackFlatF)
      (fun e => ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * (A + e))
                + ((c * h * w : ℕ) : ℝ) * e) := by
  have hu := M.u_nonneg
  have hγN0 : 0 ≤ (1 + M.u) ^ (c * h * w + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  refine ⟨fun v hv k => ?_, fun vt va e hva hvt hd k => ?_⟩
  · -- magnitude at v
    set mv : Vec (c * h * w) := fun idx => if flatChannel c h w idx = k then v idx else 0 with hmv
    have hNA0 : 0 ≤ ((c * h * w : ℕ) : ℝ) * A := by
      rcases Nat.eq_zero_or_pos (c * h * w) with h0 | hpos
      · simp [h0]
      · exact mul_nonneg (by positivity) ((abs_nonneg _).trans (hv ⟨0, hpos⟩))
    have hmvb : ∀ idx, |mv idx| ≤ A := by
      intro idx; simp only [hmv]; by_cases hc : flatChannel c h w idx = k
      · simp only [if_pos hc]; exact hv idx
      · simp only [if_neg hc, abs_zero]; exact (abs_nonneg _).trans (hv idx)
    have hsumabs : ∑ idx, |mv idx| ≤ ((c * h * w : ℕ) : ℝ) * A := by
      calc ∑ idx, |mv idx| ≤ ∑ _idx : Fin (c * h * w), A := Finset.sum_le_sum (fun idx _ => hmvb idx)
        _ = ((c * h * w : ℕ) : ℝ) * A := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hrealmag : |broadcastBackFlat c h w v k| ≤ ((c * h * w : ℕ) : ℝ) * A := by
      show |∑ idx, mv idx| ≤ _
      exact (Finset.abs_sum_le_sum_abs _ _).trans hsumabs
    have hround : |M.broadcastBackFlatF v k - broadcastBackFlat c h w v k|
        ≤ ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A) := by
      show |M.sum mv - ∑ idx, mv idx| ≤ _
      exact (M.sum_close mv).trans (mul_le_mul_of_nonneg_left hsumabs hγN0)
    refine ⟨hrealmag.trans (le_add_of_nonneg_right (mul_nonneg hγN0 hNA0)), ?_⟩
    calc |M.broadcastBackFlatF v k|
        ≤ |M.broadcastBackFlatF v k - broadcastBackFlat c h w v k|
          + |broadcastBackFlat c h w v k| := by
            simpa using abs_sub_le (M.broadcastBackFlatF v k) (broadcastBackFlat c h w v k) 0
      _ ≤ ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A)
            + ((c * h * w : ℕ) : ℝ) * A := add_le_add hround hrealmag
      _ = (((c * h * w : ℕ) : ℝ) * A)
            + ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A) := by ring
  · -- error: vt within e of va
    set mvt : Vec (c * h * w) := fun idx => if flatChannel c h w idx = k then vt idx else 0 with hmvt
    set mva : Vec (c * h * w) := fun idx => if flatChannel c h w idx = k then va idx else 0 with hmva
    have hterm : ∀ idx, |mvt idx - mva idx| ≤ e := by
      intro idx; simp only [hmvt, hmva]; by_cases hc : flatChannel c h w idx = k
      · simp only [if_pos hc]; exact hd idx
      · simp only [if_neg hc, sub_zero, abs_zero]; exact (abs_nonneg _).trans (hd idx)
    have hbound : ∀ idx, |mva idx| ≤ A := by
      intro idx; simp only [hmva]; by_cases hc : flatChannel c h w idx = k
      · simp only [if_pos hc]; exact hva idx
      · simp only [if_neg hc, abs_zero]; exact (abs_nonneg _).trans (hva idx)
    show |M.sum mvt - ∑ idx, mva idx| ≤ _
    exact reduction_close M mvt mva hterm hbound

/-- Broadcast backward float-bridges (the spatial reduction in the SE gate's backward). -/
theorem floatBridges_broadcastBack {c h w : Nat} (M : FloatModel) (hc : 0 < c) :
    FloatBridges (broadcastBackFlat c h w) := fun A hA =>
  ⟨_, _, _, (floatClose_broadcastBack M A).cod_nonneg hA hc, floatClose_broadcastBack M A⟩

-- ════════════════════════════════════════════════════════════════
-- § The SE gate's backward, fully assembled
-- ════════════════════════════════════════════════════════════════

/-- The SE gate's input-gradient VJP at a smooth point — the **exact reverse** of the gate
    `broadcastFlat ∘ sigmoid ∘ dense W₂ ∘ swish ∘ dense W₁ ∘ GAP`:

      `gapBack ∘ linBack W₁ ∘ swishBack ∘ linBack W₂ ∘ sigmoidBack ∘ broadcastBack`

    `broadcastBack` sums each channel's cotangent over space, `sigmoidBack`/`swishBack` are the
    `diagBack` saved-derivative scales (`ssig = σ'(saved)` on `Vec c`, `ssw = swish'(saved)` on
    `Vec r`), the two `linBack`s are the reduce/expand denses' transposes, and `gapBack` broadcasts
    back ÷ (h·w). -/
noncomputable def seGateInputGrad {c h w r : Nat}
    (W₁ : Mat c r) (W₂ : Mat r c) (ssig : Vec c) (ssw : Vec r) :
    Vec (c * h * w) → Vec (c * h * w) :=
  gapBack c h w
  ∘ dense (Mat.transpose W₁) (0 : Vec c)
  ∘ diagBack ssw
  ∘ dense (Mat.transpose W₂) (0 : Vec r)
  ∘ diagBack ssig
  ∘ broadcastBackFlat c h w

/-- **The SE gate's backward float-bridges.** One `.comp` chain over the per-op backward bridges:
    `broadcastBack` (the spatial reduce), the sigmoid `diagBack`, `linBack W₂`, the swish `diagBack`,
    `linBack W₁`, and `gapBack`. The saved-derivative vectors `ssig`/`ssw` enter with their float
    closeness (`esig`/`eswish` — the transcendental budgets, since `σ'`/`swish'` have no IEEE spec),
    exactly as the forward `floatClose_seGate` supplies its activations. Discharges the `gateBack`
    hypothesis of `floatBridges_seBack`. -/
theorem floatBridges_seGateBack {c h w r : Nat} (M : FloatModel)
    (W₁ : Mat c r) (W₂ : Mat r c) (ssig fssig : Vec c) (ssw fssw : Vec r)
    {w' Ssig esig Ssw eswish : ℝ} (hw' : 0 ≤ w')
    (hc : 0 < c) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hW₂ : ∀ i j, |W₂ i j| ≤ w')
    (hssig : ∀ i, |ssig i| ≤ Ssig) (hfssig : ∀ i, |fssig i - ssig i| ≤ esig)
    (hssw : ∀ i, |ssw i| ≤ Ssw) (hfssw : ∀ i, |fssw i - ssw i| ≤ eswish) :
    FloatBridges (seGateInputGrad (h := h) (w := w) W₁ W₂ ssig ssw) := by
  unfold seGateInputGrad
  exact (((((floatBridges_broadcastBack M hc).comp
    (floatBridges_diagBack M ssig fssig hc hssig hfssig)).comp
    (floatBridges_linBack M W₂ hw' hc hW₂)).comp
    (floatBridges_diagBack M ssw fssw hr hssw hfssw)).comp
    (floatBridges_linBack M W₁ hw' hr hW₁)).comp
    (floatBridges_gapBack M c h w hc hh hww)

-- ════════════════════════════════════════════════════════════════
-- § The SE block backward: the product-rule two-branch fan-in
-- ════════════════════════════════════════════════════════════════

/-- The SE block input-gradient VJP at a smooth point — the **reverse of `seBlockFull = x ⊙ gate(x)`**
    (`seBlockFull_has_vjp`, denoted by `seBlockFullBackGraphE`): the two-path fan-in

      `seBack(dy) = (g ⊙ dy) + gateBack(x ⊙ dy)`

    — the main path scales `dy` by the saved gate `g = gate(x)` (`diagBack g`), the gate path
    pre-scales `dy` by the saved input `xinp` (`diagBack xinp`) then threads `gateBack`, and the two
    add. The multiplicative cousin of the residual fan-in (`r34DownBlockBack`'s `biPathSum`): both
    branches receive the cotangent, but each sees a different scaling (the product rule). -/
noncomputable def seInputGrad {n : Nat} (g xinp : Vec n) (gateBack : Vec n → Vec n) :
    Vec n → Vec n :=
  fun dy j => diagBack g dy j + (gateBack ∘ diagBack xinp) dy j

/-- **The SE block input-gradient VJP float-bridges.** The product-rule two-branch fan-in
    (`FloatBridges.biPathSum`) of the main path `diagBack g` (saved gate `g`, `|g| ≤ Sg` — for a
    sigmoid gate `Sg = 1+esig`, so the branch can't blow up) and the gate path
    `gateBack ∘ diagBack xinp` (saved input `xinp`, `|xinp| ≤ Sx`). The gate sub-network's backward
    `gateBack` is supplied as a `FloatBridges` fact (discharged by `floatBridges_seGateBack`) —
    exactly as `floatBridges_r34DownBlockBack` supplies its BN-backs. The architecturally-distinctive
    EfficientNet/MobileNet backward op; closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem floatBridges_seBack {n : Nat} (M : FloatModel)
    (g fg xinp fx : Vec n) {Sg eg Sx ex : ℝ} (hn : 0 < n)
    (hg : ∀ i, |g i| ≤ Sg) (hfg : ∀ i, |fg i - g i| ≤ eg)
    (hx : ∀ i, |xinp i| ≤ Sx) (hfx : ∀ i, |fx i - xinp i| ≤ ex)
    {gateBack : Vec n → Vec n} (hgateBack : FloatBridges gateBack) :
    FloatBridges (seInputGrad g xinp gateBack) := by
  unfold seInputGrad
  exact FloatBridges.biPathSum M
    (floatBridges_diagBack M g fg hn hg hfg)
    ((floatBridges_diagBack M xinp fx hn hx hfx).comp hgateBack)

end Proofs
