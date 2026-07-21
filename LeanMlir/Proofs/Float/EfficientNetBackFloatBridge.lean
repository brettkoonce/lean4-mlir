import LeanMlir.Proofs.Float.DepthwiseBackFloatBridge
import LeanMlir.Proofs.Float.SEBackFloatBridge

/-! # ℝ→Float32 bridge: the EfficientNet MBConv body backward (both §1e ops land here)

A3 (planning/a3_backward_deepnet_assembly.md Part 2): the backward peer of the forward MBConv body
bridge `floatBridges_mbconvBody` (`EnetFloatBridge`). The forward whole-net `efficientnetForwardB` is
**batched** (`Vec (N·…)` via `batchMap`/`batchOp`), so — exactly as the forward bridge stops at the
per-example body — the in-scope deliverable is the per-example **MBConv body backward** (the batched
whole-net needs the separate batched-emit lift, the forward's Item-B stub).

The MBConv body forward is `project∘BN ∘ SE ∘ (swish∘BN∘depthwise) ∘ (swish∘BN∘expand)`; the backward
is its exact reverse

  `mbconvBodyBack = expandBack ∘ depthwiseBack ∘ seBack ∘ projectBack`

— and it is the first block where **both** §1e backward ops land: the depthwise input-VJP
(`depthwiseFlatBack`, concrete) AND the squeeze-excite product-rule backward (`seB`, supplied,
discharged by `floatBridges_seBack`). The expand/project convs reverse through `convFlatBack`; the
swish kink is the saved-derivative `diagBack` (`swBe`/`swBd`, supplied, discharged by
`floatBridges_diagBack`); the three batch-norms enter as the supplied `bnBack` facts (discharged by
`floatBridges_bnBack`) — the same modular split the forward `floatBridges_mbconvBody` uses for its BNs.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The MBConv body backward (depthwiseBack + seBack both land here)
-- ════════════════════════════════════════════════════════════════

/-- The EfficientNet MBConv body input-gradient VJP at a smooth point — the **reverse of
    `mbconvBody = (BN∘conv Wp) ∘ seBlockFull ∘ (swish∘BN∘depthwise Wd) ∘ (swish∘BN∘conv We)`**:

      `expandBack ∘ depthwiseBack ∘ seBack ∘ projectBack`

    `projectBack = convFlatBack Wp ∘ bnBp`; `seBack = seB` (the SE product-rule backward, supplied);
    `depthwiseBack = depthwiseFlatBack Wd ∘ bnBd ∘ swBd`; `expandBack = convFlatBack We ∘ bnBe ∘ swBe`.
    The swish backs `swBe`/`swBd` are the saved-derivative `diagBack`s; the BN-backs and the SE-back
    are supplied (the smooth/SE pieces). -/
noncomputable def mbconvBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBe bnBd swBe swBd : Vec (cmid * h * w) → Vec (cmid * h * w))
    (seB : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  (convFlatBack (h := h) (w := w) We ∘ bnBe ∘ swBe)
  ∘ (depthwiseFlatBack (h := h) (w := w) Wd ∘ bnBd ∘ swBd)
  ∘ seB
  ∘ (convFlatBack (h := h) (w := w) Wp ∘ bnBp)

/-- **The EfficientNet MBConv body backward float-bridges.** One `.comp` chain: the project
    `convFlatBack Wp ∘ bnBp`, the supplied SE product-rule backward `seB` (discharge with
    `floatBridges_seBack` — §1e), the depthwise `depthwiseFlatBack Wd ∘ bnBd ∘ swBd` (the §1e
    depthwise input-VJP, concrete), and the expand `convFlatBack We ∘ bnBe ∘ swBe`. The BN-backs
    (`bnBe/bnBd/bnBp`) and swish-backs (`swBe/swBd`) are supplied (discharged by `floatBridges_bnBack`
    / `floatBridges_diagBack`). The backward peer of `floatBridges_mbconvBody`. -/
theorem floatBridges_mbconvBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBe bnBd swBe swBd : Vec (cmid * h * w) → Vec (cmid * h * w))
    (seB : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w))
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hcmid : 0 < cmid) (hcout : 0 < cout) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hbnBd : FloatBridges bnBd)
    (hswBe : FloatBridges swBe) (hswBd : FloatBridges swBd)
    (hseB : FloatBridges seB) (hbnBp : FloatBridges bnBp) :
    FloatBridges (mbconvBodyBack We Wd Wp bnBe bnBd swBe swBd seB bnBp) := by
  unfold mbconvBodyBack
  exact ((((hbnBp.comp (floatBridges_convBack (h := h) (w := w) M Wp hwp (by positivity) hWp)).comp
    hseB).comp
    ((hswBd.comp hbnBd).comp (floatBridges_depthwiseBack (h := h) (w := w) M Wd hwd
      (by positivity) hWd))).comp
    ((hswBe.comp hbnBe).comp (floatBridges_convBack (h := h) (w := w) M We hwe (by positivity) hWe)))

/-- **The EfficientNet MBConv residual-block backward float-bridges** — the MBConv with the additive
    skip (matching in/out channels): the body backward, then the identity skip contributes the
    cotangent verbatim (`residual (mbconvBodyBack)` = `dy ↦ bodyBack(dy) + dy`). `FloatBridges.residual`
    over the body backward. -/
theorem floatBridges_mbconvResidBack {c h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 c c kHe kWe) (Wd : DepthwiseKernel c kHd kWd) (Wp : Kernel4 c c kHp kWp)
    (bnBe bnBd swBe swBd seB bnBp : Vec (c * h * w) → Vec (c * h * w))
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o cc kh kw, |We o cc kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o cc kh kw, |Wp o cc kh kw| ≤ wp)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hbnBd : FloatBridges bnBd)
    (hswBe : FloatBridges swBe) (hswBd : FloatBridges swBd)
    (hseB : FloatBridges seB) (hbnBp : FloatBridges bnBp) :
    FloatBridges (Proofs.residual (mbconvBodyBack We Wd Wp bnBe bnBd swBe swBd seB bnBp)) :=
  FloatBridges.residual M
    (floatBridges_mbconvBodyBack M We Wd Wp bnBe bnBd swBe swBd seB bnBp hwe hwd hwp hWe hWd hWp
      hc hc hh hw hbnBe hbnBd hswBe hswBd hseB hbnBp)

-- ════════════════════════════════════════════════════════════════
-- § The batched-emit lift (the forward's "Item-B") — batchMap float-bridges
--
-- `efficientnetForwardB` is BATCHED: the whole net is `batchMap N (per-example net)` (each of the N
-- examples runs the per-example net independently — the BNs are eval-mode flat affine at the smooth
-- point, NO cross-batch coupling). So the batched whole-net BACKWARD is `batchMap N (per-example net
-- backward)`. The one missing piece is the `batchMap` `FloatBridges` lift — `batchMap N f` reads
-- example `(decode idx).1`'s input slice, applies `f`, takes output coord `(decode idx).2`, so a
-- per-coordinate bound `≤ A` / perturbation `≤ e` restricts to each example's slice and the per-example
-- `B` / `L e` carry over verbatim. The `Vec(N·a)→Vec(N·b)` (input dim `a` ≠ output dim `b`, since a
-- backward swaps cin↔cout) generalization of `FloatClose.perRow`.
-- ════════════════════════════════════════════════════════════════

/-- `batchMap N f` reads coordinatewise as `f` applied to example `(decode idx).1`'s input slice. -/
theorem batchMap_apply {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a)) (idx : Fin (N * b)) :
    StableHLO.batchMap N f v idx
      = f (fun i : Fin a => v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i)))
          (finProdFinEquiv.symm idx).2 := rfl

/-- **The batched seam — `FloatClose.batchMap`.** A per-example `FloatClose A B f fF L` lifts to the
    batch `FloatClose A B (batchMap N f) (batchMap N fF) L` with the SAME magnitude and modulus: each
    example is an independent copy of `f`. The `Vec(N·a)→Vec(N·b)` generalization of `FloatClose.perRow`
    (so it applies to a backward op, whose input/output dims differ). -/
theorem FloatClose.batchMap {a b : Nat} (N : Nat) {A B : ℝ} {f fF : Vec a → Vec b} {L : ℝ → ℝ}
    (hf : FloatClose A B f fF L) :
    FloatClose A B (StableHLO.batchMap N f) (StableHLO.batchMap N fF) L := by
  obtain ⟨hm, he⟩ := hf
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hslice : ∀ i, |v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ A :=
      fun i => hv (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    rw [batchMap_apply, batchMap_apply]
    exact hm (fun i => v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))) hslice
      (finProdFinEquiv.symm idx).2
  · have hva' : ∀ i, |va (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ A :=
      fun i => hva (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    have hvt' : ∀ i, |vt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ A :=
      fun i => hvt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    have hd' : ∀ i, |vt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
                  - va (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))| ≤ e :=
      fun i => hd (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))
    rw [batchMap_apply, batchMap_apply]
    exact he (fun i => vt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i)))
      (fun i => va (finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))) e hva' hvt' hd'
      (finProdFinEquiv.symm idx).2

/-- **`FloatBridges.batchMap`** — the batched-emit lift in bridge form (the forward's "Item-B"). If the
    per-example map float-bridges, so does the whole batch `batchMap N f`. -/
theorem FloatBridges.batchMap {a b : Nat} (N : Nat) {f : Vec a → Vec b} (hf : FloatBridges f) :
    FloatBridges (StableHLO.batchMap N f) := by
  intro A hA
  obtain ⟨B, L, fF, hB, hfc⟩ := hf A hA
  exact ⟨B, L, StableHLO.batchMap N fF, hB, hfc.batchMap N⟩

/-- **The BATCHED MBConv body backward float-bridges** — `batchMap N` of the per-example
    `mbconvBodyBack`. The repeating unit of the batched whole-net backward; closes the forward's
    Item-B stub for the backward direction (`Vec(N·cout·h·w)→Vec(N·cin·h·w)`). -/
theorem floatBridges_mbconvBatchedBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat} (N : Nat)
    (M : FloatModel) (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBe bnBd swBe swBd : Vec (cmid * h * w) → Vec (cmid * h * w))
    (seB : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w))
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hcmid : 0 < cmid) (hcout : 0 < cout) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hbnBd : FloatBridges bnBd)
    (hswBe : FloatBridges swBe) (hswBd : FloatBridges swBd)
    (hseB : FloatBridges seB) (hbnBp : FloatBridges bnBp) :
    FloatBridges (StableHLO.batchMap N (mbconvBodyBack We Wd Wp bnBe bnBd swBe swBd seB bnBp)) :=
  FloatBridges.batchMap N
    (floatBridges_mbconvBodyBack M We Wd Wp bnBe bnBd swBe swBd seB bnBp hwe hwd hwp hWe hWd hWp
      hcmid hcout hh hw hbnBe hbnBd hswBe hswBd hseB hbnBp)

/-- **The BATCHED MBConv residual-block backward float-bridges** — `batchMap N` of the per-example
    residual MBConv backward (the dominant block: most MBConv stages have the matched-channel skip).
    `FloatBridges.batchMap` over `floatBridges_mbconvResidBack`. The batched whole-net backward is the
    `.comp` chain of these (`batchMap` distributes over the per-example composition). -/
theorem floatBridges_mbconvBatchedResidBack {c h w kHe kWe kHd kWd kHp kWp : Nat} (N : Nat)
    (M : FloatModel) (We : Kernel4 c c kHe kWe) (Wd : DepthwiseKernel c kHd kWd) (Wp : Kernel4 c c kHp kWp)
    (bnBe bnBd swBe swBd seB bnBp : Vec (c * h * w) → Vec (c * h * w))
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o cc kh kw, |We o cc kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o cc kh kw, |Wp o cc kh kw| ≤ wp)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (hbnBe : FloatBridges bnBe) (hbnBd : FloatBridges bnBd)
    (hswBe : FloatBridges swBe) (hswBd : FloatBridges swBd)
    (hseB : FloatBridges seB) (hbnBp : FloatBridges bnBp) :
    FloatBridges (StableHLO.batchMap N
      (Proofs.residual (mbconvBodyBack We Wd Wp bnBe bnBd swBe swBd seB bnBp))) :=
  FloatBridges.batchMap N
    (floatBridges_mbconvResidBack M We Wd Wp bnBe bnBd swBe swBd seB bnBp hwe hwd hwp hWe hWd hWp
      hc hh hw hbnBe hbnBd hswBe hswBd hseB hbnBp)

end Proofs
