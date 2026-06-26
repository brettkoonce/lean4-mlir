import LeanMlir.Proofs.StridedConvBackFloatBridge

/-! # ℝ→Float32 bridge for the ResNet-34 DOWNSAMPLE-block backward

The r34 downsample block forward is `rblkPStridedPC = relu ∘ residualProj proj body`:
* `proj = bn_p ∘ flatConvStride2 Wp` — the 3×3 stride-2 projection skip;
* `body = bn₂ ∘ conv₂ ∘ relu ∘ bn₁ ∘ flatConvStride2 W₁` — strided body (first conv stride-2);
* `residualProj proj body x = proj x + body x`, then a final ReLU.

Its input-gradient VJP at a smooth point is `relu(proj+body)` reversed: the ReLU mask, then the
**two-branch fan-in** (BOTH branches receive the cotangent and the results ADD) — `dx = bProj(dy') +
bBody(dy')`. Unlike the identity block (one branch is `id`, handled by `FloatBridges.residual`), here
BOTH branches are non-trivial, so we need the two-branch rounded-sum combinator
`FloatBridges.biPathSum` (the general `f(x) + g(x)`, of which `floatClose_addResidual`'s `f(x) + x`
is the `g = id` case). The branch backwards reuse the strided-conv backward
(`floatBridges_flatConvStride2Back`), the non-strided conv backward, and the supplied per-channel
BN-backs:

* `bProj = flatConvStride2Back Wp ∘ bnBp`
* `bBody = flatConvStride2Back W₁ ∘ bnB₁ ∘ reluMaskBack ∘ convFlatBack W₂ ∘ bnB₂`

With this, every r34 block type (identity + downsample) and the stem op are float-bridged.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The two-branch rounded-sum combinator (`f(x) + g(x)`)
-- ════════════════════════════════════════════════════════════════

/-- **Two-branch rounded sum is `FloatClose`** — the general `f(x) + g(x)` (rounded add), of which
    `floatClose_addResidual`'s `f(x) + x` is the `g = id` case. Both branches `FloatClose A B₁`/
    `A B₂`; the float `fl(fF ⊕ gF)` is within `add_close`'s budget of `f + g`, output magnitude
    `(1+u)(B₁+B₂)`. The residual fan-in where BOTH branches are non-trivial (the downsample skip). -/
theorem floatClose_biPathSum {m n : Nat} (M : FloatModel) {A B₁ B₂ : ℝ}
    {f fF g gF : Vec m → Vec n} {L₁ L₂ : ℝ → ℝ}
    (hf : FloatClose A B₁ f fF L₁) (hg : FloatClose A B₂ g gF L₂) :
    FloatClose A (B₁ + B₂ + M.u * (B₁ + B₂))
      (fun v j => f v j + g v j)
      (fun v j => M.add (fF v j) (gF v j))
      (fun e => M.u * (B₁ + L₁ e + B₂ + L₂ e) + (L₁ e + L₂ e)) := by
  have hu := M.u_nonneg
  obtain ⟨hfm, hfe⟩ := hf
  obtain ⟨hgm, hge⟩ := hg
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · have hb1 : |f v i| ≤ B₁ := (hfm v hv i).1
    have hfb1 : |fF v i| ≤ B₁ := (hfm v hv i).2
    have hb2 : |g v i| ≤ B₂ := (hgm v hv i).1
    have hgb2 : |gF v i| ≤ B₂ := (hgm v hv i).2
    have hB12 : 0 ≤ B₁ + B₂ := by
      have := (abs_nonneg (f v i)).trans hb1; have := (abs_nonneg (g v i)).trans hb2; linarith
    refine ⟨?_, ?_⟩
    · calc |f v i + g v i| ≤ |f v i| + |g v i| := abs_add_le _ _
        _ ≤ B₁ + B₂ := add_le_add hb1 hb2
        _ ≤ B₁ + B₂ + M.u * (B₁ + B₂) := le_add_of_nonneg_right (mul_nonneg hu hB12)
    · have hsum : |fF v i + gF v i| ≤ B₁ + B₂ := (abs_add_le _ _).trans (add_le_add hfb1 hgb2)
      calc |M.add (fF v i) (gF v i)|
          ≤ |M.add (fF v i) (gF v i) - (fF v i + gF v i)| + |fF v i + gF v i| := by
            simpa using abs_sub_le (M.add (fF v i) (gF v i)) (fF v i + gF v i) 0
        _ ≤ M.u * |fF v i + gF v i| + |fF v i + gF v i| := add_le_add (M.err _) le_rfl
        _ ≤ M.u * (B₁ + B₂) + (B₁ + B₂) := add_le_add (mul_le_mul_of_nonneg_left hsum hu) hsum
        _ = B₁ + B₂ + M.u * (B₁ + B₂) := by ring
  · refine (M.add_close (hfe vt va e hva hvt hd i) (hge vt va e hva hvt hd i)).trans ?_
    have hb1 : |f va i| ≤ B₁ := (hfm va hva i).1
    have hb2 : |g va i| ≤ B₂ := (hgm va hva i).1
    have h1 : M.u * (|f va i| + L₁ e + |g va i| + L₂ e) ≤ M.u * (B₁ + L₁ e + B₂ + L₂ e) :=
      mul_le_mul_of_nonneg_left (by linarith) hu
    linarith

/-- **Float-bridging survives the two-branch sum.** If both branches float-bridge, so does
    `fun v j => f v j + g v j` (rounded add). The downsample fan-in in bridge form (the two-branch
    cousin of `FloatBridges.residual`). -/
theorem FloatBridges.biPathSum {m n : Nat} (M : FloatModel) {f g : Vec m → Vec n}
    (hf : FloatBridges f) (hg : FloatBridges g) :
    FloatBridges (fun v j => f v j + g v j) := by
  intro A hA
  obtain ⟨B₁, L₁, fF, hB₁, hfc⟩ := hf A hA
  obtain ⟨B₂, L₂, gF, hB₂, hgc⟩ := hg A hA
  refine ⟨B₁ + B₂ + M.u * (B₁ + B₂), _, _, ?_, floatClose_biPathSum M hfc hgc⟩
  have hu := M.u_nonneg
  have hB12 : 0 ≤ B₁ + B₂ := add_nonneg hB₁ hB₂
  nlinarith [mul_nonneg hu hB12]

-- ════════════════════════════════════════════════════════════════
-- § The downsample-block input-VJP
-- ════════════════════════════════════════════════════════════════

/-- The r34 downsample basic-block input-gradient VJP at a smooth point — the **reverse of
    `rblkPStridedPC`**. `relu(proj(x) + body(x))` backward = the ReLU mask, then the two-branch
    fan-in `bProj(dy') + bBody(dy')` (both branches non-trivial, summed). The strided convs reverse
    via `flatConvStride2Back`; the BN-backs `bnB₁`/`bnB₂`/`bnBp` are the per-channel BatchNorm
    backwards (supplied). -/
noncomputable def r34DownBlockBack {ic oc h w : Nat}
    (W₁ : Kernel4 oc ic 3 3) (W₂ : Kernel4 oc oc 3 3) (Wp : Kernel4 oc ic 3 3)
    (bnB1 bnB2 bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_out m_mid : Fin (oc * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid] :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  (fun dy j =>
      (flatConvStride2Back (h := h) (w := w) Wp ∘ bnBp) dy j
      + (flatConvStride2Back (h := h) (w := w) W₁ ∘ bnB1 ∘ reluMaskBack m_mid
          ∘ convFlatBack (h := h) (w := w) W₂ ∘ bnB2) dy j)
    ∘ reluMaskBack m_out

/-- **The r34 downsample-block input-gradient VJP float-bridges.** One `.comp` chain: the outer ReLU
    mask, then the two-branch fan-in (`FloatBridges.biPathSum`) of the projection backward
    `flatConvStride2Back Wp ∘ bnBp` and the body backward
    `flatConvStride2Back W₁ ∘ bnB₁ ∘ reluMaskBack ∘ convFlatBack W₂ ∘ bnB₂`. The BN-backs are
    supplied as `FloatBridges` facts (discharge with `floatBridges_bnPerChannelBack`). Completes the
    r34 block set (identity + downsample); closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem floatBridges_r34DownBlockBack {ic oc h w : Nat} (M : FloatModel)
    (W₁ : Kernel4 oc ic 3 3) (W₂ : Kernel4 oc oc 3 3) (Wp : Kernel4 oc ic 3 3)
    (bnB1 bnB2 bnBp : Vec (oc * h * w) → Vec (oc * h * w))
    (m_out m_mid : Fin (oc * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    {w₁ w₂ wp : ℝ} (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂) (hwp : 0 ≤ wp)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w₁) (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w₂)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hoc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (hbnB1 : FloatBridges bnB1) (hbnB2 : FloatBridges bnB2) (hbnBp : FloatBridges bnBp) :
    FloatBridges (r34DownBlockBack W₁ W₂ Wp bnB1 bnB2 bnBp m_out m_mid) := by
  unfold r34DownBlockBack
  refine (floatBridges_reluMaskBack m_out).comp (FloatBridges.biPathSum M ?_ ?_)
  · -- bProj = flatConvStride2Back Wp ∘ bnBp
    exact hbnBp.comp (floatBridges_flatConvStride2Back (h := h) (w := w) M Wp hwp (by positivity) hWp)
  · -- bBody = flatConvStride2Back W₁ ∘ bnB₁ ∘ reluMaskBack ∘ convFlatBack W₂ ∘ bnB₂
    exact ((((hbnB2.comp (floatBridges_convBack (h := h) (w := w) M W₂ hw₂ (by positivity) hW₂)).comp
          (floatBridges_reluMaskBack m_mid)).comp hbnB1).comp
          (floatBridges_flatConvStride2Back (h := h) (w := w) M W₁ hw₁ (by positivity) hW₁))

end Proofs
