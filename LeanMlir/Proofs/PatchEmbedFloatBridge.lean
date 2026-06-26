import LeanMlir.Proofs.ViTWholeFloatBridge
import LeanMlir.Proofs.PatchEmbedBackFloatBridge

/-! # ℝ→Float32 bridge: the ViT PATCH-EMBED forward (the last vit forward endpoint)

The forward peer of `floatBridges_patchEmbedBack`. The certified patch embedding `patchEmbed_flat`
(`Attention.lean`) is, at output token `n` channel `d`,

  pos_embed n d + (if n = 0 then cls_token d
                   else b_conv d + ∑c ∑kh ∑kw W_conv[d,c,kh,kw]·(if patch p covers (hh,ww) then img(…) else 0))

an **affine** guarded triple-sum: the constants `pos_embed`/`cls_token`/`b_conv` are exact stored
weights, so in the difference `f(imgt) − f(imga)` they cancel and the sensitivity is purely the linear
conv-dot (`ic·P²·wc·e`), exactly like the backward. The float model rounds each leaf multiply
(`mul_close`), the three `c`/`kh`/`kw` sums (nested `reduction_close`), and the two constant adds
(`add_close`); the in-bounds guard is a fixed index predicate (identical in float and real), so the
closeness threads cleanly. `floatBridges_patchEmbed` discharges the `hPatch` hypothesis of
`vit_floatBridges`, making the whole ViT forward fully concrete — `vit_floatBridges_concrete`, the
forward peer of `vit_grad_floatBridges_concrete`. A3 = closeness at a smooth point (NOT descent).
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The float patch-embed forward + budgets
-- ════════════════════════════════════════════════════════════════

/-- **Float patch-embed forward** — the rounded peer of `patchEmbed_flat`: the outer `pos_embed +` and
    the `b_conv +` are `M.add`, the three patch/offset sums are nested `M.sum`, each leaf multiply is
    `M.mul`. Same `cls_token` slot and same in-bounds guard. -/
noncomputable def FloatModel.patchEmbedF (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv : Vec D)
    (cls_token : Vec D) (pos_embed : Mat (N + 1) D) (img : Vec (ic * H * W)) :
    Vec ((N + 1) * D) :=
  fun idx_out =>
    let n := (finProdFinEquiv.symm idx_out).1
    let d := (finProdFinEquiv.symm idx_out).2
    M.add (pos_embed n d)
      (if n.val = 0 then cls_token d
       else M.add (b_conv d)
         (M.sum fun c : Fin ic => M.sum fun kh : Fin patchSize => M.sum fun kw : Fin patchSize =>
           M.mul (W_conv d c kh kw)
             (let W' := W / patchSize
              let p := n.val - 1
              let h' := p / W'
              let w' := p % W'
              let hh := h' * patchSize + kh.val
              let ww := w' * patchSize + kw.val
              if hpad : hh < H ∧ ww < W then
                img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
              else 0)))

/-- The real conv-dot magnitude bound `ic·P²·wc·A` (the triple-sum leaf is `≤ wc·A`). -/
noncomputable def patchEmbedConvMag (ic patchSize : Nat) (wc A : ℝ) : ℝ :=
  (ic : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * (wc * A)))

/-- The real patch-embed magnitude: `pos_embed + (b_conv | cls_token) + conv-dot`. -/
noncomputable def patchEmbedMag (ic patchSize : Nat) (wc pb A : ℝ) : ℝ :=
  pb + (pb + patchEmbedConvMag ic patchSize wc A)

/-- The nested triple-sum (`c`/`kh`/`kw`) rounding error: the leaf `mulErr` folded through three
    `reduction_close`s. -/
noncomputable def patchEmbedTripleErr (M : FloatModel) (ic patchSize : Nat) (wc A : ℝ) : ℝ :=
  redErr M.u ic ((patchSize : ℝ) * ((patchSize : ℝ) * (wc * A)))
    (redErr M.u patchSize ((patchSize : ℝ) * (wc * A))
      (redErr M.u patchSize (wc * A) (mulErr M.u wc A 0 0)))

/-- The inner-add (`b_conv +`) rounding error level. -/
noncomputable def patchEmbedBranchErr (M : FloatModel) (ic patchSize : Nat) (wc pb A : ℝ) : ℝ :=
  M.u * (pb + patchEmbedConvMag ic patchSize wc A + patchEmbedTripleErr M ic patchSize wc A)
    + patchEmbedTripleErr M ic patchSize wc A

/-- The full patch-embed rounding error: the triple-sum error plus the two constant `add_close`s. -/
noncomputable def patchEmbedRoundErr (M : FloatModel) (ic patchSize : Nat) (wc pb A : ℝ) : ℝ :=
  M.u * (pb + (pb + patchEmbedConvMag ic patchSize wc A)
          + patchEmbedBranchErr M ic patchSize wc pb A)
    + patchEmbedBranchErr M ic patchSize wc pb A

/-- A `redErr` reduction budget is nonnegative when its magnitude/error inputs are. -/
theorem redErr_nonneg (u : ℝ) (n : Nat) {Mr ef : ℝ} (hu : 0 ≤ u) (hMr : 0 ≤ Mr) (hef : 0 ≤ ef) :
    0 ≤ redErr u n Mr ef :=
  add_nonneg (mul_nonneg (sub_nonneg.mpr (one_le_pow₀ (by linarith)))
      (mul_nonneg (Nat.cast_nonneg n) (add_nonneg hMr hef)))
    (mul_nonneg (Nat.cast_nonneg n) hef)

theorem patchEmbedConvMag_nonneg (ic patchSize : Nat) {wc A : ℝ} (hwc0 : 0 ≤ wc) (hA : 0 ≤ A) :
    0 ≤ patchEmbedConvMag ic patchSize wc A := by unfold patchEmbedConvMag; positivity

theorem patchEmbedTripleErr_nonneg (M : FloatModel) (ic patchSize : Nat) {wc A : ℝ}
    (hwc0 : 0 ≤ wc) (hA : 0 ≤ A) : 0 ≤ patchEmbedTripleErr M ic patchSize wc A :=
  redErr_nonneg M.u ic M.u_nonneg (by positivity)
    (redErr_nonneg M.u patchSize M.u_nonneg (by positivity)
      (redErr_nonneg M.u patchSize M.u_nonneg (mul_nonneg hwc0 hA)
        (mulErr_nonneg M.u_nonneg hwc0 hA le_rfl le_rfl)))

theorem patchEmbedBranchErr_nonneg (M : FloatModel) (ic patchSize : Nat) {wc pb A : ℝ}
    (hwc0 : 0 ≤ wc) (hpb0 : 0 ≤ pb) (hA : 0 ≤ A) :
    0 ≤ patchEmbedBranchErr M ic patchSize wc pb A := by
  unfold patchEmbedBranchErr
  have hu := M.u_nonneg
  have h1 := patchEmbedConvMag_nonneg ic patchSize hwc0 hA
  have h2 := patchEmbedTripleErr_nonneg M ic patchSize hwc0 hA
  positivity

-- ════════════════════════════════════════════════════════════════
-- § Real magnitude + sensitivity (the forward is affine in the image)
-- ════════════════════════════════════════════════════════════════

/-- **Real patch-embed magnitude** — `|patchEmbed_flat img idx| ≤ patchEmbedMag`. The `pos_embed`/
    `cls_token`/`b_conv` constants ride their bound `pb`; the conv-dot rides `triple_sum_abs_le`. -/
theorem patchEmbed_flat_abs_le (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D) (pos_embed : Mat (N + 1) D)
    {wc pb A : ℝ} (hwc0 : 0 ≤ wc) (hA : 0 ≤ A)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc) (hpos : ∀ n d, |pos_embed n d| ≤ pb)
    (hcls : ∀ d, |cls_token d| ≤ pb) (hbc : ∀ d, |b_conv d| ≤ pb)
    (img : Vec (ic * H * W)) (himg : ∀ k, |img k| ≤ A) (idx : Fin ((N + 1) * D)) :
    |patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed img idx|
      ≤ patchEmbedMag ic patchSize wc pb A := by
  unfold patchEmbed_flat patchEmbedMag patchEmbedConvMag
  have hconv0 : 0 ≤ (ic : ℝ) * ((patchSize : ℝ) * ((patchSize : ℝ) * (wc * A))) := by positivity
  refine (abs_add_le _ _).trans (add_le_add (hpos _ _) ?_)
  by_cases hn : ((finProdFinEquiv.symm idx).1).val = 0
  · rw [if_pos hn]; exact (hcls _).trans (by linarith)
  · rw [if_neg hn]
    refine (abs_add_le _ _).trans (add_le_add (hbc _) ?_)
    refine triple_sum_abs_le _ (fun c kh kw => ?_)
    rw [abs_mul]
    refine mul_le_mul (hwc _ _ _ _) ?_ (abs_nonneg _) hwc0
    by_cases hpad : (((finProdFinEquiv.symm idx).1).val - 1) / (W / patchSize) * patchSize + kh.val < H
        ∧ (((finProdFinEquiv.symm idx).1).val - 1) % (W / patchSize) * patchSize + kw.val < W
    · simp only [dif_pos hpad]; exact himg _
    · simp only [dif_neg hpad, abs_zero]; exact hA

/-- **Real patch-embed sensitivity** — `|patchEmbed_flat imgt idx − patchEmbed_flat imga idx| ≤
    ic·P²·wc·e`. The constants cancel (the `pos_embed +` and the `cls_token`/`b_conv` are
    image-independent), so only the linear conv-dot survives. -/
theorem patchEmbed_flat_sub_abs_le (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D) (pos_embed : Mat (N + 1) D)
    {wc e : ℝ} (hwc0 : 0 ≤ wc) (he : 0 ≤ e)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc)
    (imgt imga : Vec (ic * H * W)) (hd : ∀ k, |imgt k - imga k| ≤ e) (idx : Fin ((N + 1) * D)) :
    |patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed imgt idx
        - patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed imga idx|
      ≤ patchEmbedConvMag ic patchSize wc e := by
  unfold patchEmbed_flat patchEmbedConvMag
  by_cases hn : ((finProdFinEquiv.symm idx).1).val = 0
  · simp only [if_pos hn, sub_self, abs_zero]; positivity
  · simp only [if_neg hn]
    rw [add_sub_add_left_eq_sub, add_sub_add_left_eq_sub]
    simp only [← Finset.sum_sub_distrib, ← mul_sub]
    refine triple_sum_abs_le _ (fun c kh kw => ?_)
    rw [abs_mul]
    refine mul_le_mul (hwc _ _ _ _) ?_ (abs_nonneg _) hwc0
    by_cases hpad : (((finProdFinEquiv.symm idx).1).val - 1) / (W / patchSize) * patchSize + kh.val < H
        ∧ (((finProdFinEquiv.symm idx).1).val - 1) % (W / patchSize) * patchSize + kw.val < W
    · simp only [dif_pos hpad]; exact hd _
    · simp only [dif_neg hpad, sub_zero, abs_zero]; exact he

-- ════════════════════════════════════════════════════════════════
-- § The rounding closeness (nested reduction_close + the two adds)
-- ════════════════════════════════════════════════════════════════

/-- **Patch-embed forward rounding closeness** — the deployed float forward is within
    `patchEmbedRoundErr` of the certified one: the leaf `mul_close` folded through the three nested
    `reduction_close`s over `kw`/`kh`/`c`, then the two constant `add_close`s (`b_conv +` and
    `pos_embed +`). -/
theorem patchEmbedF_round_close (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D) (pos_embed : Mat (N + 1) D)
    {wc pb A : ℝ} (hwc0 : 0 ≤ wc) (hpb0 : 0 ≤ pb) (hA : 0 ≤ A)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc) (hpos : ∀ n d, |pos_embed n d| ≤ pb)
    (hcls : ∀ d, |cls_token d| ≤ pb) (hbc : ∀ d, |b_conv d| ≤ pb)
    (img : Vec (ic * H * W)) (himg : ∀ k, |img k| ≤ A) (idx : Fin ((N + 1) * D)) :
    |M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed img idx
        - patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed img idx|
      ≤ patchEmbedRoundErr M ic patchSize wc pb A := by
  unfold FloatModel.patchEmbedF patchEmbed_flat
  set n := (finProdFinEquiv.symm idx).1 with hn_def
  set d := (finProdFinEquiv.symm idx).2 with hd_def
  set gR : Fin ic → Fin patchSize → Fin patchSize → ℝ := fun c kh kw =>
    W_conv d c kh kw *
      (if hpad : (n.val - 1) / (W / patchSize) * patchSize + kh.val < H
            ∧ (n.val - 1) % (W / patchSize) * patchSize + kw.val < W then
        img (finProdFinEquiv (finProdFinEquiv (c, ⟨(n.val - 1) / (W / patchSize) * patchSize + kh.val,
              hpad.1⟩), ⟨(n.val - 1) % (W / patchSize) * patchSize + kw.val, hpad.2⟩))
       else 0) with hgR
  set gF : Fin ic → Fin patchSize → Fin patchSize → ℝ := fun c kh kw =>
    M.mul (W_conv d c kh kw)
      (if hpad : (n.val - 1) / (W / patchSize) * patchSize + kh.val < H
            ∧ (n.val - 1) % (W / patchSize) * patchSize + kw.val < W then
        img (finProdFinEquiv (finProdFinEquiv (c, ⟨(n.val - 1) / (W / patchSize) * patchSize + kh.val,
              hpad.1⟩), ⟨(n.val - 1) % (W / patchSize) * patchSize + kw.val, hpad.2⟩))
       else 0) with hgF
  -- the guard value is `≤ A`, so the leaf is mul_close / magnitude `wc·A`
  have hleafmag : ∀ c kh kw, |gR c kh kw| ≤ wc * A := by
    intro c kh kw
    rw [hgR, abs_mul]
    refine mul_le_mul (hwc _ _ _ _) ?_ (abs_nonneg _) hwc0
    split_ifs with hpad
    · exact himg _
    · simpa using hA
  have hleafclose : ∀ c kh kw, |gF c kh kw - gR c kh kw| ≤ mulErr M.u wc A 0 0 := by
    intro c kh kw
    rw [hgF, hgR]
    refine M.mul_close (by simp) (by simp) (hwc _ _ _ _) ?_
    split_ifs with hpad
    · exact himg _
    · simpa using hA
  -- partial real magnitudes (each sum multiplies the bound by `patchSize`)
  have hkwmag : ∀ c kh, |∑ kw, gR c kh kw| ≤ (patchSize : ℝ) * (wc * A) := fun c kh => by
    calc |∑ kw, gR c kh kw| ≤ ∑ kw, |gR c kh kw| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _kw : Fin patchSize, (wc * A) := Finset.sum_le_sum fun kw _ => hleafmag c kh kw
      _ = (patchSize : ℝ) * (wc * A) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hkhmag : ∀ c, |∑ kh, ∑ kw, gR c kh kw| ≤ (patchSize : ℝ) * ((patchSize : ℝ) * (wc * A)) :=
    fun c => by
    calc |∑ kh, ∑ kw, gR c kh kw| ≤ ∑ kh, |∑ kw, gR c kh kw| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _kh : Fin patchSize, ((patchSize : ℝ) * (wc * A)) := Finset.sum_le_sum fun kh _ => hkwmag c kh
      _ = (patchSize : ℝ) * ((patchSize : ℝ) * (wc * A)) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- the triple-sum rounding closeness (three nested reductions)
  have het : |M.sum (fun c => M.sum fun kh => M.sum fun kw => gF c kh kw)
        - ∑ c, ∑ kh, ∑ kw, gR c kh kw| ≤ patchEmbedTripleErr M ic patchSize wc A := by
    unfold patchEmbedTripleErr
    refine reduction_close M _ _ (fun c => ?_) (fun c => hkhmag c)
    refine reduction_close M _ _ (fun kh => ?_) (fun kh => hkwmag c kh)
    exact reduction_close M _ _ (fun kw => hleafclose c kh kw) (fun kw => hleafmag c kh kw)
  -- the real triple-sum magnitude
  have hTRmag : |∑ c, ∑ kh, ∑ kw, gR c kh kw| ≤ patchEmbedConvMag ic patchSize wc A := by
    unfold patchEmbedConvMag
    exact triple_sum_abs_le _ (fun c kh kw => hleafmag c kh kw)
  -- the branch (the `if n = 0`): magnitude and rounding closeness
  have hBRmag : |if n.val = 0 then cls_token d else b_conv d + ∑ c, ∑ kh, ∑ kw, gR c kh kw|
      ≤ pb + patchEmbedConvMag ic patchSize wc A := by
    have hconv0 : 0 ≤ patchEmbedConvMag ic patchSize wc A := by unfold patchEmbedConvMag; positivity
    by_cases hn : n.val = 0
    · rw [if_pos hn]; exact (hcls _).trans (by linarith)
    · rw [if_neg hn]
      exact (abs_add_le _ _).trans (add_le_add (hbc _) hTRmag)
  have hBFclose : |(if n.val = 0 then cls_token d
        else M.add (b_conv d) (M.sum (fun c => M.sum fun kh => M.sum fun kw => gF c kh kw)))
        - (if n.val = 0 then cls_token d else b_conv d + ∑ c, ∑ kh, ∑ kw, gR c kh kw)|
      ≤ patchEmbedBranchErr M ic patchSize wc pb A := by
    have het0 : 0 ≤ patchEmbedTripleErr M ic patchSize wc A := (abs_nonneg _).trans het
    have hconv0 : 0 ≤ patchEmbedConvMag ic patchSize wc A := by unfold patchEmbedConvMag; positivity
    by_cases hn : n.val = 0
    · simp only [if_pos hn, sub_self, abs_zero]
      exact patchEmbedBranchErr_nonneg M ic patchSize hwc0 hpb0 hA
    · rw [if_neg hn, if_neg hn]
      refine (add_close (M := M) (by simp : |b_conv d - b_conv d| ≤ 0) het).trans ?_
      unfold patchEmbedBranchErr
      have hu := M.u_nonneg
      nlinarith [hbc d, abs_nonneg (b_conv d), hTRmag, mul_nonneg hu (abs_nonneg (b_conv d))]
  -- assemble the outer `pos_embed +` add
  refine (add_close (M := M) (by simp : |pos_embed n d - pos_embed n d| ≤ 0) hBFclose).trans ?_
  unfold patchEmbedRoundErr
  have hu := M.u_nonneg
  have hbranch0 : 0 ≤ patchEmbedBranchErr M ic patchSize wc pb A :=
    patchEmbedBranchErr_nonneg M ic patchSize hwc0 hpb0 hA
  nlinarith [hpos n d, abs_nonneg (pos_embed n d), hBRmag, hbranch0,
    mul_nonneg hu (abs_nonneg (pos_embed n d))]

-- ════════════════════════════════════════════════════════════════
-- § The FloatClose + the bridge + the fully-concrete whole-net forward
-- ════════════════════════════════════════════════════════════════

/-- **Patch-embed forward is `FloatClose`** — magnitude `patchEmbedMag + rounding`, modulus the
    rounding budget + the real sensitivity at `e` (the embedding is affine in the image). -/
theorem floatClose_patchEmbed (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D) (pos_embed : Mat (N + 1) D)
    {wc pb A : ℝ} (hwc0 : 0 ≤ wc) (hpb0 : 0 ≤ pb) (hA : 0 ≤ A) (himgpos : 0 < ic * H * W)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc) (hpos : ∀ n d, |pos_embed n d| ≤ pb)
    (hcls : ∀ d, |cls_token d| ≤ pb) (hbc : ∀ d, |b_conv d| ≤ pb) :
    FloatClose A (patchEmbedMag ic patchSize wc pb A + patchEmbedRoundErr M ic patchSize wc pb A)
      (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed)
      (M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed)
      (fun e => patchEmbedRoundErr M ic patchSize wc pb A + patchEmbedConvMag ic patchSize wc e) := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hreal := patchEmbed_flat_abs_le ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 hA hwc hpos hcls hbc v hv idx
    have hround := patchEmbedF_round_close M ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 hpb0 hA hwc hpos hcls hbc v hv idx
    have hB0 : 0 ≤ patchEmbedRoundErr M ic patchSize wc pb A := (abs_nonneg _).trans hround
    refine ⟨le_trans hreal (by linarith), ?_⟩
    calc |M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed v idx|
        ≤ |M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed v idx
            - patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed v idx|
          + |patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed v idx| := by
          simpa using abs_sub_le
            (M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed v idx)
            (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed v idx) 0
      _ ≤ patchEmbedRoundErr M ic patchSize wc pb A + patchEmbedMag ic patchSize wc pb A :=
          add_le_add hround hreal
      _ = patchEmbedMag ic patchSize wc pb A + patchEmbedRoundErr M ic patchSize wc pb A := by ring
  · have he0 : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, himgpos⟩)
    have hround := patchEmbedF_round_close M ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 hpb0 hA hwc hpos hcls hbc vt hvt idx
    have hsens := patchEmbed_flat_sub_abs_le ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 he0 hwc vt va hd idx
    calc |M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed vt idx
            - patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed va idx|
        ≤ |M.patchEmbedF ic H W patchSize N D W_conv b_conv cls_token pos_embed vt idx
            - patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed vt idx|
          + |patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed vt idx
            - patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed va idx| :=
          abs_sub_le _ _ _
      _ ≤ patchEmbedRoundErr M ic patchSize wc pb A + patchEmbedConvMag ic patchSize wc e :=
          add_le_add hround hsens

/-- **The patch-embed forward float-bridges** — discharges the `hPatch` hypothesis of
    `vit_floatBridges`, making the whole ViT forward fully concrete. -/
theorem floatBridges_patchEmbed (M : FloatModel) (ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D) (pos_embed : Mat (N + 1) D)
    {wc pb : ℝ} (hwc0 : 0 ≤ wc) (hpb0 : 0 ≤ pb) (hnd : 0 < (N + 1) * D) (himgpos : 0 < ic * H * W)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc) (hpos : ∀ n d, |pos_embed n d| ≤ pb)
    (hcls : ∀ d, |cls_token d| ≤ pb) (hbc : ∀ d, |b_conv d| ≤ pb) :
    FloatBridges (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed) := by
  intro A hA
  exact ⟨_, _, _,
    (floatClose_patchEmbed M ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 hpb0 hA himgpos hwc hpos hcls hbc).cod_nonneg hA hnd,
    floatClose_patchEmbed M ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 hpb0 hA himgpos hwc hpos hcls hbc⟩

/-- **THE FULLY-CONCRETE ViT WHOLE-NET FORWARD.** `vit_floatBridges` with the patch-embed endpoint
    discharged by `floatBridges_patchEmbed` — EVERY endpoint (patch-embed, cls-slice, head) concrete;
    only the per-block forwards and the final LN supplied (dischargeable by `floatBridges_vitBlock` /
    `floatBridges_bn`), exactly as `r34_floatBridges` supplies its blocks. The deployed float ViT
    forward ≈ the certified ℝ forward, end to end — the forward peer of `vit_grad_floatBridges_concrete`. -/
theorem vit_floatBridges_concrete (M : FloatModel)
    (ic H W patchSize N D nClasses : Nat) (Wcls : Mat D nClasses) (bcls : Vec nClasses)
    (finalLN : Vec D → Vec D) (blocks : List (Vec ((N + 1) * D) → Vec ((N + 1) * D)))
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D) (pos_embed : Mat (N + 1) D)
    {w' β wc pb : ℝ} (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hD : 0 < D) (himgpos : 0 < ic * H * W)
    (hwc0 : 0 ≤ wc) (hpb0 : 0 ≤ pb)
    (hWcls : ∀ i j, |Wcls i j| ≤ w') (hbcls : ∀ j, |bcls j| ≤ β)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc) (hpos : ∀ n d, |pos_embed n d| ≤ pb)
    (hcls : ∀ d, |cls_token d| ≤ pb) (hbc : ∀ d, |b_conv d| ≤ pb)
    (hFinalLN : FloatBridges finalLN) (hblocks : ∀ f ∈ blocks, FloatBridges f) :
    FloatBridges (vitForwardFlat Wcls bcls finalLN blocks
      (patchEmbed_flat ic H W patchSize N D W_conv b_conv cls_token pos_embed)) :=
  vit_floatBridges M Wcls bcls finalLN blocks _ hw' hβ hD hWcls hbcls hFinalLN hblocks
    (floatBridges_patchEmbed M ic H W patchSize N D W_conv b_conv cls_token pos_embed
      hwc0 hpb0 (Nat.mul_pos (Nat.succ_pos N) hD) himgpos hwc hpos hcls hbc)

end Proofs
