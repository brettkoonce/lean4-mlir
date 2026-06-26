import LeanMlir.Proofs.FloatComposeBridge
import LeanMlir.Proofs.Resnet34FloatBridge

/-! # ℝ→Float32 bridge: the WHOLE ResNet-34 FORWARD — the [3,4,6,3] fold

The forward peer of `r34_grad_floatBridges` (`Resnet34WholeBackFloatBridge.lean`). The repo had
the r34 forward float story only at *block* level (`Resnet34FloatBridge.lean`'s per-op `_close`
lemmas) while the backward already folded the whole net; this closes that asymmetry by assembling
the forward `[3,4,6,3]` `.comp` chain in the SAME blueprint the backward uses.

`resnet34Forward_full_pc` is `dense ∘ GAP ∘ [3,4,6,3] blocks ∘ maxpool ∘ stem`. The stem
(`relu ∘ bn ∘ flatConvStride2`), maxpool, GAP, and dense **endpoints are concrete**; the stem BN
and the 16 **blocks are supplied as `FloatBridges` facts** — exactly as `cifarBn_floatBridges`
supplies its BNs and `r34_grad_floatBridges` supplies its block backwards — discharged by the
per-block bridges. The `[3,4,6,3]` stage structure is encoded in the block maps' dimensions
(down-blocks change channels×spatial; identity blocks preserve).

Two forward op-bridges were missing and are built here (each a thin wrap of an existing `_close`):
* `floatBridges_flatConvStride2` — the stem stride-2 conv. `flatConvStride2 = decimateFlat ∘
  flatConv` is the stride-1 conv read at the decimated output coordinate, so its `FloatClose` is
  `floatClose_flatConv` (at the `2h×2w` grid) evaluated at `decimateIdx`.
* `floatBridges_gap` — the GAP squeeze, wrapping the existing `floatClose_gap`.
-/

namespace Proofs

open scoped Real
open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Forward stride-2 conv as a `FloatBridges`  (the stem)
-- ════════════════════════════════════════════════════════════════

/-- **Stride-2 conv is `FloatClose`.** `flatConvStride2 W b = decimateFlat ∘ flatConv` selects the
    stride-1 conv's output at the even (`decimateIdx`) coordinates, so every magnitude/perturbation
    bound is `floatClose_flatConv` (on the `2h×2w` grid) read at `decimateIdx`. Same conv-fan-in
    `layerBudget` (the kernel fan-in `ic·kH·kW` is stride-independent). -/
theorem floatClose_flatConvStride2 {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < ic * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatClose A
      (layerAct (ic * kH * kW) w' β A + layerBudget M.u (ic * kH * kW) w' β A 0)
      (flatConvStride2 (h := h) (w := w) W b) (M.flatConvStride2F (h := h) (w := w) W b)
      (fun e => layerBudget M.u (ic * kH * kW) w' β A e) := by
  obtain ⟨hm, he⟩ := floatClose_flatConv (h := 2 * h) (w := 2 * w) M W b hw' hβ hA hn hW hb
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · exact hm v hv (decimateIdx oc h w i)
  · exact he vt va e hva hvt hd (decimateIdx oc h w i)

/-- **Stride-2 conv float-bridges** (output magnitude `layerAct + layerBudget`). -/
theorem floatBridges_flatConvStride2 {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < ic * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β) :
    FloatBridges (flatConvStride2 (h := h) (w := w) W b) :=
  fun _A hA => ⟨_, _, _,
    add_nonneg (layerAct_nonneg hw' hβ hA) (layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl),
    floatClose_flatConvStride2 M W b hw' hβ hA hn hW hb⟩

-- ════════════════════════════════════════════════════════════════
-- § Forward GAP as a `FloatBridges`  (the squeeze)
-- ════════════════════════════════════════════════════════════════

/-- **GAP float-bridges** — the `FloatBridges` wrap of `floatClose_gap` (output magnitude `A + gb`,
    the per-channel `bnMean` budget). -/
theorem floatBridges_gap {c h w : Nat} (M : FloatModel) (hc : 0 < c) (hhw : 0 < h * w) :
    FloatBridges (globalAvgPoolFlat c h w) :=
  fun _A hA => ⟨_, _, _, (floatClose_gap M hA hhw).cod_nonneg hA hc, floatClose_gap M hA hhw⟩

-- ════════════════════════════════════════════════════════════════
-- § The whole-net forward (the [3,4,6,3] fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole ResNet-34 forward — the structural skeleton of `resnet34Forward_full_pc`:
    `dense ∘ GAP ∘ [3,4,6,3] blocks ∘ maxpool ∘ (relu ∘ bn ∘ stride-2-conv)`. The stem conv/relu,
    maxpool, GAP, and dense endpoints are concrete; the stem BN `bnS` and the 16 blocks
    `a0..e1` are supplied (each `FloatBridges`, discharged by the per-block bridges). The
    `[3,4,6,3]` stage structure is in the block maps' dims (down-blocks change channels×spatial;
    identity blocks preserve). The forward peer of `r34InputGrad`. -/
noncomputable def r34Forward (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (Wd : Mat 512 10) (bd : Vec 10)
    (bnS : Vec (64 * 112 * 112) → Vec (64 * 112 * 112))
    (a0 a1 a2 : Vec (64 * 56 * 56) → Vec (64 * 56 * 56))
    (d2 : Vec (64 * 56 * 56) → Vec (128 * 28 * 28))
    (b0 b1 b2 : Vec (128 * 28 * 28) → Vec (128 * 28 * 28))
    (d3 : Vec (128 * 28 * 28) → Vec (256 * 14 * 14))
    (c0 c1 c2 c3 c4 : Vec (256 * 14 * 14) → Vec (256 * 14 * 14))
    (d4 : Vec (256 * 14 * 14) → Vec (512 * 7 * 7))
    (e0 e1 : Vec (512 * 7 * 7) → Vec (512 * 7 * 7)) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wd bd
  ∘ globalAvgPoolFlat 512 7 7
  ∘ e1 ∘ e0
  ∘ d4
  ∘ c4 ∘ c3 ∘ c2 ∘ c1 ∘ c0
  ∘ d3
  ∘ b2 ∘ b1 ∘ b0
  ∘ d2
  ∘ a2 ∘ a1 ∘ a0
  ∘ maxPoolFlat 64 56 56
  ∘ (relu (64 * 112 * 112) ∘ bnS ∘ flatConvStride2 (h := 112) (w := 112) Ws bs)

set_option maxRecDepth 100000 in
/-- **The whole ResNet-34 forward float-bridges** — the forward peer of `r34_grad_floatBridges`.
    One `.comp` chain over the per-op forward bridges: the concrete stem (`flatConvStride2 ∘ bn ∘
    relu`), `maxPoolFlat`, the 16 supplied blocks, `globalAvgPoolFlat`, and `dense`. The deployed
    float forward of the whole 34-layer net is within an explicit budget of the certified `ℝ`
    forward. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem r34_floatBridges (M : FloatModel)
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (Wd : Mat 512 10) (bd : Vec 10)
    (bnS : Vec (64 * 112 * 112) → Vec (64 * 112 * 112))
    (a0 a1 a2 : Vec (64 * 56 * 56) → Vec (64 * 56 * 56))
    (d2 : Vec (64 * 56 * 56) → Vec (128 * 28 * 28))
    (b0 b1 b2 : Vec (128 * 28 * 28) → Vec (128 * 28 * 28))
    (d3 : Vec (128 * 28 * 28) → Vec (256 * 14 * 14))
    (c0 c1 c2 c3 c4 : Vec (256 * 14 * 14) → Vec (256 * 14 * 14))
    (d4 : Vec (256 * 14 * 14) → Vec (512 * 7 * 7))
    (e0 e1 : Vec (512 * 7 * 7) → Vec (512 * 7 * 7))
    {ws bsβ wd bdβ : ℝ} (hws : 0 ≤ ws) (hbsβ : 0 ≤ bsβ) (hwd : 0 ≤ wd) (hbdβ : 0 ≤ bdβ)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hbs : ∀ o, |bs o| ≤ bsβ)
    (hWd : ∀ i j, |Wd i j| ≤ wd) (hbd : ∀ j, |bd j| ≤ bdβ)
    (hbnS : FloatBridges bnS)
    (ha0 : FloatBridges a0) (ha1 : FloatBridges a1) (ha2 : FloatBridges a2)
    (hd2 : FloatBridges d2)
    (hb0 : FloatBridges b0) (hb1 : FloatBridges b1) (hb2 : FloatBridges b2)
    (hd3 : FloatBridges d3)
    (hc0 : FloatBridges c0) (hc1 : FloatBridges c1) (hc2 : FloatBridges c2)
    (hc3 : FloatBridges c3) (hc4 : FloatBridges c4)
    (hd4 : FloatBridges d4)
    (he0 : FloatBridges e0) (he1 : FloatBridges e1) :
    FloatBridges (r34Forward Ws bs Wd bd bnS a0 a1 a2 d2 b0 b1 b2 d3
      c0 c1 c2 c3 c4 d4 e0 e1) := by
  unfold r34Forward
  have hstem : FloatBridges
      (relu (64 * 112 * 112) ∘ bnS ∘ flatConvStride2 (h := 112) (w := 112) Ws bs) :=
    ((floatBridges_flatConvStride2 (h := 112) (w := 112) M Ws bs hws hbsβ (by norm_num) hWs hbs).comp
      hbnS).comp floatBridges_relu
  have hMP := hstem.comp (floatBridges_maxPool (c := 64) (h := 56) (w := 56))
  have hA := ((hMP.comp ha0).comp ha1).comp ha2
  have hD2 := hA.comp hd2
  have hB := ((hD2.comp hb0).comp hb1).comp hb2
  have hD3 := hB.comp hd3
  have hC := ((((hD3.comp hc0).comp hc1).comp hc2).comp hc3).comp hc4
  have hD4 := hC.comp hd4
  have hE := (hD4.comp he0).comp he1
  have hGAP := hE.comp (floatBridges_gap (c := 512) (h := 7) (w := 7) M (by norm_num) (by norm_num))
  exact hGAP.comp (floatBridges_dense M Wd bd hwd hbdβ (by norm_num) hWd hbd)

end Proofs
