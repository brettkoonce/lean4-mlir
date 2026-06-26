import LeanMlir.Proofs.ConvNeXtFullT
import LeanMlir.Proofs.EnetFloatBridge
import LeanMlir.Proofs.Resnet34WholeFloatBridge
import LeanMlir.Proofs.ViTFloatBridge
import LeanMlir.Proofs.LinBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE ConvNeXt-T FORWARD — the [3,3,9,3] fold

The forward peer of `convnext_grad_floatBridges` (`ConvNeXtBackFloatBridge.lean`). The repo had the
ConvNeXt forward float story only at *op* level; this folds the whole `convNextForwardT` (the [3,3,9,3]
ch9 render) in the SAME blueprint the backward uses.

The ConvNeXt block body is `layerScale ∘ conv(proj) ∘ gelu ∘ conv(expand) ∘ LN ∘ depthwise`, wrapped
in `residual`. Two forward op-bridges were missing and are built here:
* `floatBridges_layerScale` — the per-channel layer scale `x ↦ γ ⊙ x`. `layerScale γ = diagBack γ`
  definitionally (`fun s x i => s i * x i`), and `γ` is an exact stored weight (no transcendental), so
  this is `floatBridges_diagBack` at the supplied-derivative `fγ = γ`, `es = 0`.
* `floatBridges_flatConvStride4` — the 4×4/s4 patchify stem. `flatConvStride4 = decimateFlat ∘
  decimateOddFlat ∘ flatConv` is the stride-1 conv read at the composite `decimateOddIdx ∘ decimateIdx`
  coordinate, so its `FloatClose` is `floatClose_flatConv` (on the `4h×4w` grid) evaluated there — the
  two-decimation cousin of r34's `floatClose_flatConvStride2`.

Then the named bridges discharge the fold: `floatBridges_convNextBlock` (the block, `residual(body)`),
`floatBridges_convNextStageK` (the depth-`k` stage fold — the ConvNeXt analogue of ViT's
`floatBridges_towerBack`, by induction on the stage depth), and `floatBridges_cnxDownW` (the
stage-boundary downsample `flatConvStride2 ∘ LN`). `convnextForward` is the `∘` skeleton of
`convNextForwardT` (concrete stem-conv/GAP/dense; stem-LN, head-LN, 4 stages, 3 downsamples supplied
as `FloatBridges` — exactly as `convnextInputGrad` supplies its `s1B..s4B`/`d1B..d3B`/`lnB*`). The LNs
enter abstractly because `layerNormForward = bnForward` has the rsqrt keystone.
-/

namespace Proofs

open scoped Real
open FloatModel

/-- The identity float-bridges (the `convNextStageK 0` base case). Magnitude/modulus pass through. -/
theorem floatBridges_idVec {m : Nat} : FloatBridges (id : Vec m → Vec m) :=
  fun A hA => ⟨A, _, _, hA, ⟨fun _v hv i => ⟨hv i, hv i⟩, fun _ _ _ _ _ hd i => hd i⟩⟩

-- ════════════════════════════════════════════════════════════════
-- § Forward layer-scale as a `FloatBridges`  (the new ConvNeXt op)
-- ════════════════════════════════════════════════════════════════

/-- **Layer scale float-bridges.** `layerScale γ x = γ ⊙ x` is `diagBack γ` definitionally; `γ` is an
    exact stored weight (no transcendental derivative), so the supplied float multiplier is `γ` itself
    (`es = 0`). The output magnitude is `Sd·A + mulErr` (one rounded multiply per coordinate). -/
theorem floatBridges_layerScale {n : Nat} (M : FloatModel) (γ : Vec n) {Sd : ℝ}
    (hn : 0 < n) (hγ : ∀ i, |γ i| ≤ Sd) :
    FloatBridges (layerScale γ) :=
  floatBridges_diagBack (es := 0) M γ γ hn hγ (fun _ => by simp)

-- ════════════════════════════════════════════════════════════════
-- § Forward stride-4 patchify conv as a `FloatBridges`  (the stem)
-- ════════════════════════════════════════════════════════════════

/-- The float stride-4 patchify conv: decimate (even, then odd) the float stride-1 conv (the float peer
    of `flatConvStride4 = decimateFlat ∘ decimateOddFlat ∘ flatConv`). -/
noncomputable def FloatModel.flatConvStride4F {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) → Vec (oc * h * w) :=
  decimateFlat oc h w ∘ decimateOddFlat oc (2 * h) (2 * w) ∘
    M.flatConvF (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b

/-- **Stride-4 patchify conv is `FloatClose`.** `flatConvStride4 W b = decimateFlat ∘ decimateOddFlat ∘
    flatConv` selects the stride-1 conv's output at the composite `decimateOddIdx ∘ decimateIdx`
    coordinate, so every magnitude/perturbation bound is `floatClose_flatConv` (on the `4h×4w` grid) read
    there. Same conv-fan-in `layerBudget` (the kernel fan-in `ic·kH·kW` is stride-independent). The
    two-decimation cousin of `floatClose_flatConvStride2`. -/
theorem floatClose_flatConvStride4 {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' bb A : ℝ}
    (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hA : 0 ≤ A) (hn : 0 < ic * (2 * (2 * h)) * (2 * (2 * w)))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ bb) :
    FloatClose A
      (layerAct (ic * kH * kW) w' bb A + layerBudget M.u (ic * kH * kW) w' bb A 0)
      (flatConvStride4 (h := h) (w := w) W b) (M.flatConvStride4F (h := h) (w := w) W b)
      (fun e => layerBudget M.u (ic * kH * kW) w' bb A e) := by
  obtain ⟨hm, he⟩ :=
    floatClose_flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) M W b hw' hbb hA hn hW hb
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · exact hm v hv (decimateOddIdx oc (2 * h) (2 * w) (decimateIdx oc h w i))
  · exact he vt va e hva hvt hd (decimateOddIdx oc (2 * h) (2 * w) (decimateIdx oc h w i))

/-- **Stride-4 patchify conv float-bridges** (output magnitude `layerAct + layerBudget`). -/
theorem floatBridges_flatConvStride4 {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) {w' bb : ℝ}
    (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hn : 0 < ic * (2 * (2 * h)) * (2 * (2 * w)))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ bb) :
    FloatBridges (flatConvStride4 (h := h) (w := w) W b) :=
  fun _A hA => ⟨_, _, _,
    add_nonneg (layerAct_nonneg hw' hbb hA) (layerBudget_nonneg M.u_nonneg hw' hbb hA le_rfl),
    floatClose_flatConvStride4 M W b hw' hbb hA hn hW hb⟩

-- ════════════════════════════════════════════════════════════════
-- § The ConvNeXt block forward bridge (peer of floatBridges_cnxBlockBack)
-- ════════════════════════════════════════════════════════════════

/-- **The ConvNeXt block float-bridges** — the forward peer of `floatBridges_cnxBlockBack`.
    `convNextBlock = residual (layerScale ∘ conv(proj) ∘ gelu ∘ conv(expand) ∘ LN ∘ depthwise)`: one
    `.comp` chain over the per-op forward bridges (depthwise, expand conv, GELU, project conv, layer
    scale; the LayerNorm supplied as a `FloatBridges` fact, discharged by `floatBridges_bn`), wrapped by
    `FloatBridges.residual`. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem floatBridges_convNextBlock {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c) (εn γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c) (γls : Vec (c * h * w))
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ w') (hbdw : ∀ ch, |bdw ch| ≤ bb)
    (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ w') (hbex : ∀ o, |bex o| ≤ bb)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ w') (hbpr : ∀ o, |bpr o| ≤ bb)
    (hγls : ∀ i, |γls i| ≤ w')
    (hln : FloatBridges (layerNormForward (c * h * w) εn γn βn)) :
    FloatBridges (convNextBlock Wdw bdw εn γn βn Wex bex Wpr bpr γls) := by
  unfold convNextBlock convNextBlockBody
  have hD := floatBridges_depthwise (h := h) (w := w) M Wdw bdw hw' hbb hc hWdw hbdw
  have hEX := floatBridges_flatConv (h := h) (w := w) M Wex bex hw' hbb hc hWex hbex
  have hGE := floatBridges_gelu (n := cExp * h * w) fgelu hegelu hg
  have hPR := floatBridges_flatConv (h := h) (w := w) M Wpr bpr hw' hbb hcExp hWpr hbpr
  have hLS := floatBridges_layerScale M γls hc hγls
  exact FloatBridges.residual M
    (((((hD.comp hln).comp hEX).comp hGE).comp hPR).comp hLS)

/-- The per-block weight/bias/layer-scale bound bundle (a single `w'` for weights+γls, `bb` for biases). -/
abbrev CnxBlockBounded {c cExp h w kH kW : Nat} (p : CnxBlockParams c cExp h w kH kW)
    (w' bb : ℝ) : Prop :=
  (∀ ch kh kw, |p.Wdw ch kh kw| ≤ w') ∧ (∀ ch, |p.bdw ch| ≤ bb) ∧
  (∀ o cc kh kw, |p.Wex o cc kh kw| ≤ w') ∧ (∀ o, |p.bex o| ≤ bb) ∧
  (∀ o cc kh kw, |p.Wpr o cc kh kw| ≤ w') ∧ (∀ o, |p.bpr o| ≤ bb) ∧
  (∀ ch, |p.γls ch| ≤ w')

/-- **The packaged ConvNeXt block (`cnxBlockW`) float-bridges** — `floatBridges_convNextBlock` fed a
    `CnxBlockParams`; the layer-scale bound rides through the `cnxGls` channel-reindex. -/
theorem floatBridges_cnxBlockW {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    (p : CnxBlockParams c cExp h w kH kW)
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hb : CnxBlockBounded p w' bb)
    (hln : FloatBridges (layerNormForward (c * h * w) p.εn p.γn p.βn)) :
    FloatBridges (cnxBlockW p) := by
  obtain ⟨hWdw, hbdw, hWex, hbex, hWpr, hbpr, hγls⟩ := hb
  unfold cnxBlockW
  exact floatBridges_convNextBlock M fgelu p.Wdw p.bdw p.εn p.γn p.βn p.Wex p.bex p.Wpr p.bpr
    (cnxGls p) hw' hbb hegelu hc hcExp hg hWdw hbdw hWex hbex hWpr hbpr
    (fun i => hγls (StableHLO.chanIdx c h w i)) hln

-- ════════════════════════════════════════════════════════════════
-- § The depth-k stage fold (peer of ViT's floatBridges_towerBack)
-- ════════════════════════════════════════════════════════════════

/-- **The depth-`k` ConvNeXt stage float-bridges** — `convNextStageK k ps` is the head-recursive fold
    of `k` blocks (block `0` first); its bridge is the `.comp` fold of `floatBridges_cnxBlockW`, by
    induction on the stage depth. The ConvNeXt analogue of ViT's `floatBridges_towerBack` (blocks have
    DISTINCT params, so the explicit depth fold, not a uniform iterate). Discharges each [3,3,9,3]
    stage given uniform per-block bounds + per-block LayerNorm bridges. -/
theorem floatBridges_convNextStageK {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParams c cExp h w kH kW),
      (∀ i, CnxBlockBounded (ps i) w' bb) →
      (∀ i, FloatBridges (layerNormForward (c * h * w) (ps i).εn (ps i).γn (ps i).βn)) →
      FloatBridges (convNextStageK k ps)
  | 0, _, _, _ => floatBridges_idVec
  | _ + 1, ps, hb, hln =>
      (floatBridges_cnxBlockW M fgelu (ps 0) hw' hbb hegelu hc hcExp hg (hb 0) (hln 0)).comp
        (floatBridges_convNextStageK M fgelu hw' hbb hegelu hc hcExp hg _
          (fun i => ps i.succ) (fun i => hb i.succ) (fun i => hln i.succ))

-- ════════════════════════════════════════════════════════════════
-- § The stage-boundary downsample bridge (peer of floatBridges_cnxDownBack)
-- ════════════════════════════════════════════════════════════════

/-- **The ConvNeXt downsample float-bridges** — the forward peer of `floatBridges_cnxDownBack`.
    `cnxDownW = flatConvStride2 W ∘ LN`: the supplied LayerNorm then the stride-2 widening conv. -/
theorem floatBridges_cnxDownW {cin cout : Nat} (h w : Nat) (M : FloatModel)
    (p : CnxDownParams cin cout)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hn : 0 < cin * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |p.W o c kh kw| ≤ w') (hb : ∀ o, |p.b o| ≤ bb)
    (hln : FloatBridges (layerNormForward (cin * (2 * h) * (2 * w)) p.ε p.γ p.β)) :
    FloatBridges (cnxDownW h w p) := by
  unfold cnxDownW
  exact hln.comp (floatBridges_flatConvStride2 (h := h) (w := w) M p.W p.b hw' hbb hn hW hb)

-- ════════════════════════════════════════════════════════════════
-- § The whole-net forward (the [3,3,9,3] fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole ConvNeXt-T forward — the structural skeleton of `convNextForwardT`:
    `dense ∘ lnHead ∘ GAP ∘ s4 ∘ d3 ∘ s3 ∘ d2 ∘ s2 ∘ d1 ∘ s1 ∘ lnStem ∘ stride-4-conv`. The stem conv,
    GAP, and dense endpoints are concrete; the stem/head LayerNorms `lnStem`/`lnHead` and the 4 stages
    `s1..s4` + 3 downsamples `d1..d3` are supplied (each `FloatBridges`, discharged by
    `floatBridges_convNextStageK` / `floatBridges_cnxDownW` / `floatBridges_bn`). The `[3,3,9,3]` stage
    structure is in the stage maps' depths; the 96→192→384→768 / 56→28→14→7 schedule in their dims. The
    forward peer of `convnextInputGrad`. -/
noncomputable def convnextForward (sW : Kernel4 96 3 4 4) (sb : Vec 96) (Wd : Mat 768 10) (bd : Vec 10)
    (lnStem : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (lnHead : Vec 768 → Vec 768)
    (s1 : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (d1 : Vec (96 * 56 * 56) → Vec (192 * 28 * 28))
    (s2 : Vec (192 * 28 * 28) → Vec (192 * 28 * 28))
    (d2 : Vec (192 * 28 * 28) → Vec (384 * 14 * 14))
    (s3 : Vec (384 * 14 * 14) → Vec (384 * 14 * 14))
    (d3 : Vec (384 * 14 * 14) → Vec (768 * 7 * 7))
    (s4 : Vec (768 * 7 * 7) → Vec (768 * 7 * 7)) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wd bd
  ∘ lnHead
  ∘ globalAvgPoolFlat 768 7 7
  ∘ s4 ∘ d3 ∘ s3 ∘ d2 ∘ s2 ∘ d1 ∘ s1
  ∘ lnStem
  ∘ flatConvStride4 (h := 56) (w := 56) sW sb

set_option maxRecDepth 100000 in
/-- **The whole ConvNeXt-T forward float-bridges** — the forward peer of `convnext_grad_floatBridges`.
    One `.comp` chain over the per-op forward bridges: the concrete stem (`lnStem ∘ flatConvStride4`),
    the 4 stages, the 3 downsamples, `globalAvgPoolFlat`, the head LayerNorm, and `dense`. The deployed
    float forward of the whole [3,3,9,3] net is within an explicit budget of the certified `ℝ` forward.
    Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem convnext_floatBridges (M : FloatModel)
    (sW : Kernel4 96 3 4 4) (sb : Vec 96) (Wd : Mat 768 10) (bd : Vec 10)
    (lnStem : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (lnHead : Vec 768 → Vec 768)
    (s1 : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (d1 : Vec (96 * 56 * 56) → Vec (192 * 28 * 28))
    (s2 : Vec (192 * 28 * 28) → Vec (192 * 28 * 28))
    (d2 : Vec (192 * 28 * 28) → Vec (384 * 14 * 14))
    (s3 : Vec (384 * 14 * 14) → Vec (384 * 14 * 14))
    (d3 : Vec (384 * 14 * 14) → Vec (768 * 7 * 7))
    (s4 : Vec (768 * 7 * 7) → Vec (768 * 7 * 7))
    {ws bsβ wd bdβ : ℝ} (hws : 0 ≤ ws) (hbsβ : 0 ≤ bsβ) (hwd : 0 ≤ wd) (hbdβ : 0 ≤ bdβ)
    (hsW : ∀ o c kh kw, |sW o c kh kw| ≤ ws) (hsb : ∀ o, |sb o| ≤ bsβ)
    (hWd : ∀ i j, |Wd i j| ≤ wd) (hbd : ∀ j, |bd j| ≤ bdβ)
    (hlnStem : FloatBridges lnStem) (hlnHead : FloatBridges lnHead)
    (hs1 : FloatBridges s1) (hd1 : FloatBridges d1) (hs2 : FloatBridges s2)
    (hd2 : FloatBridges d2) (hs3 : FloatBridges s3) (hd3 : FloatBridges d3)
    (hs4 : FloatBridges s4) :
    FloatBridges (convnextForward sW sb Wd bd lnStem lnHead s1 d1 s2 d2 s3 d3 s4) := by
  unfold convnextForward
  have hstem : FloatBridges (lnStem ∘ flatConvStride4 (h := 56) (w := 56) sW sb) :=
    (floatBridges_flatConvStride4 (h := 56) (w := 56) M sW sb hws hbsβ (by norm_num) hsW hsb).comp
      hlnStem
  have h1 := hstem.comp hs1
  have hD1 := h1.comp hd1
  have h2 := hD1.comp hs2
  have hD2 := h2.comp hd2
  have h3 := hD2.comp hs3
  have hD3 := h3.comp hd3
  have h4 := hD3.comp hs4
  have hGAP := h4.comp (floatBridges_gap (c := 768) (h := 7) (w := 7) M (by norm_num) (by norm_num))
  have hHead := hGAP.comp hlnHead
  exact hHead.comp (floatBridges_dense M Wd bd hwd hbdβ (by norm_num) hWd hbd)

end Proofs
