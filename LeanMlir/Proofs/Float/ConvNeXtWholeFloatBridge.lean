import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Float.EnetFloatBridge
import LeanMlir.Proofs.Float.Resnet34WholeFloatBridge
import LeanMlir.Proofs.Float.ViTFloatBridge
import LeanMlir.Proofs.Float.LinBackFloatBridge
import LeanMlir.Proofs.Float.ChannelLNFloatBridge

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

/-- **The ConvNeXt block body float-bridges, at ANY normalisation.** `cnxBodyWith LN = layerScale ∘
    conv(proj) ∘ gelu ∘ conv(expand) ∘ LN ∘ depthwise`: one `.comp` chain over the per-op forward
    bridges, with `LN` entering only as a `FloatBridges` hypothesis — nothing in the body's float
    story inspects it.

    **This is §2n's keystone in float form.** `cnxBodyWith` is the LN-abstract body §2m built for the
    channel-LN VJP, and `convNextBlockBody … = cnxBodyWith (layerNormForward (c*h*w) εn γn βn) …`
    holds by `rfl`, so ONE theorem serves both the retained ch9 scalar-LN net (below) and the shipped
    channel-LN net (`floatBridges_cnxBlockChW`) — the ch9 instantiation is free. -/
theorem floatBridges_cnxBodyWith {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    (LN : Vec (c * h * w) → Vec (c * h * w))
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c) (γls : Vec (c * h * w))
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ w') (hbdw : ∀ ch, |bdw ch| ≤ bb)
    (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ w') (hbex : ∀ o, |bex o| ≤ bb)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ w') (hbpr : ∀ o, |bpr o| ≤ bb)
    (hγls : ∀ i, |γls i| ≤ w')
    (hln : FloatBridges LN) :
    FloatBridges (cnxBodyWith LN Wdw bdw Wex bex Wpr bpr γls) := by
  unfold cnxBodyWith
  have hD := floatBridges_depthwise (h := h) (w := w) M Wdw bdw hw' hbb hc hWdw hbdw
  have hEX := floatBridges_flatConv (h := h) (w := w) M Wex bex hw' hbb hc hWex hbex
  have hGE := floatBridges_gelu (n := cExp * h * w) fgelu hegelu hg
  have hPR := floatBridges_flatConv (h := h) (w := w) M Wpr bpr hw' hbb hcExp hWpr hbpr
  have hLS := floatBridges_layerScale M γls hc hγls
  exact ((((hD.comp hln).comp hEX).comp hGE).comp hPR).comp hLS

/-- **The ConvNeXt block at any normalisation float-bridges** — `residual (cnxBodyWith LN …)`,
    the additive skip wrapping the LN-abstract body. -/
theorem floatBridges_cnxBlockWith {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    (LN : Vec (c * h * w) → Vec (c * h * w))
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c) (γls : Vec (c * h * w))
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ w') (hbdw : ∀ ch, |bdw ch| ≤ bb)
    (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ w') (hbex : ∀ o, |bex o| ≤ bb)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ w') (hbpr : ∀ o, |bpr o| ≤ bb)
    (hγls : ∀ i, |γls i| ≤ w')
    (hln : FloatBridges LN) :
    FloatBridges (Proofs.residual (cnxBodyWith LN Wdw bdw Wex bex Wpr bpr γls)) :=
  FloatBridges.residual M
    (floatBridges_cnxBodyWith M fgelu LN Wdw bdw Wex bex Wpr bpr γls hw' hbb hegelu hc hcExp hg
      hWdw hbdw hWex hbex hWpr hbpr hγls hln)

/-- **The ConvNeXt block float-bridges** — the forward peer of `floatBridges_cnxBlockBack`.
    `convNextBlock = residual (layerScale ∘ conv(proj) ∘ gelu ∘ conv(expand) ∘ LN ∘ depthwise)`: one
    `.comp` chain over the per-op forward bridges (depthwise, expand conv, GELU, project conv, layer
    scale; the LayerNorm supplied as a `FloatBridges` fact, discharged by `floatBridges_bn`), wrapped by
    `FloatBridges.residual`. Closes under `[propext, Classical.choice, Quot.sound]`.

    Since §2n this is `floatBridges_cnxBlockWith` at the scalar `layerNormForward` — the keystone
    `convNextBlockBody = cnxBodyWith (layerNormForward …)` is `rfl`, so the ch9 net keeps this
    statement verbatim and pays nothing for the generalisation. -/
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
    FloatBridges (convNextBlock Wdw bdw εn γn βn Wex bex Wpr bpr γls) :=
  floatBridges_cnxBlockWith M fgelu (layerNormForward (c * h * w) εn γn βn)
    Wdw bdw Wex bex Wpr bpr γls hw' hbb hegelu hc hcExp hg
    hWdw hbdw hWex hbex hWpr hbpr hγls hln

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
-- § §2n — the SAME three bridges at ConvNeXt's REAL channel LayerNorm
-- ════════════════════════════════════════════════════════════════

/-! Everything above normalises with `layerNormForward`: one mean and one variance over the whole
`c·h·w` map, scalar γ/β. The shipped net (§2m) normalises with `chanLNTensor3`. The three bridges
below are the peers of `floatBridges_cnxBlockW` / `floatBridges_convNextStageK` /
`floatBridges_cnxDownW` at that LN, and they are *mechanical*: the block one is
`floatBridges_cnxBlockWith` (the LN-abstract keystone above) fed `floatBridges_chanLNTensor3`, and
the stage fold is the same induction on depth. The supplied fact is now the PURE-normalise
`FloatBridges (layerNormForward c ε 1 0)` over the `c` channels at one spatial position — the same
rsqrt keystone, at a different reduction width; the γ/β affine has moved out of it and into the
bridge (`floatBridges_layerNormVec`), which is why the two bounds `hγn`/`hβn` appear here. -/

/-- The per-block bound bundle at the channel LN. Against `CnxBlockBounded`: `γn`/`βn` are now
    `Vec c` and must be bounded here (the scalar world passed them inside its LN hypothesis). -/
abbrev CnxBlockChBounded {c cExp h w kH kW : Nat} (p : CnxBlockParamsCh c cExp h w kH kW)
    (w' bb : ℝ) : Prop :=
  (∀ ch kh kw, |p.Wdw ch kh kw| ≤ w') ∧ (∀ ch, |p.bdw ch| ≤ bb) ∧
  (∀ o cc kh kw, |p.Wex o cc kh kw| ≤ w') ∧ (∀ o, |p.bex o| ≤ bb) ∧
  (∀ o cc kh kw, |p.Wpr o cc kh kw| ≤ w') ∧ (∀ o, |p.bpr o| ≤ bb) ∧
  (∀ ch, |p.γls ch| ≤ w') ∧ (∀ ch, |p.γn ch| ≤ w') ∧ (∀ ch, |p.βn ch| ≤ bb)

/-- **The channel-LN ConvNeXt block (`cnxBlockChW`) float-bridges** — `floatBridges_cnxBlockWith`
    at `LN := chanLNTensor3`, fed a `CnxBlockParamsCh`; the layer-scale bound rides through the
    `cnxGlsCh` channel-reindex exactly as `cnxGls` does in the scalar peer. -/
theorem floatBridges_cnxBlockChW {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    (p : CnxBlockParamsCh c cExp h w kH kW)
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hcc : 0 < c) (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hb : CnxBlockChBounded p w' bb)
    (hln : FloatBridges (layerNormForward c p.εn 1 0)) :
    FloatBridges (cnxBlockChW p) := by
  obtain ⟨hWdw, hbdw, hWex, hbex, hWpr, hbpr, hγls, hγn, hβn⟩ := hb
  unfold cnxBlockChW
  exact floatBridges_cnxBlockWith M fgelu (chanLNTensor3 c h w p.εn p.γn p.βn)
    p.Wdw p.bdw p.Wex p.bex p.Wpr p.bpr (cnxGlsCh p) hw' hbb hegelu hc hcExp hg
    hWdw hbdw hWex hbex hWpr hbpr (fun i => hγls (StableHLO.chanIdx c h w i))
    (floatBridges_chanLNTensor3 M p.γn p.βn hcc hγn hβn hln)

/-- **The depth-`k` channel-LN stage float-bridges** — the peer of `floatBridges_convNextStageK`:
    the same head-recursive `.comp` fold, by induction on the stage depth, at `cnxBlockChW`.
    Discharges each [3,3,9,3] stage of the shipped net given uniform per-block bounds + per-block
    pure-normalise bridges. -/
theorem floatBridges_convNextStageChK {c cExp h w kH kW : Nat} (M : FloatModel) (fgelu : ℝ → ℝ)
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hcc : 0 < c) (hc : 0 < c * h * w) (hcExp : 0 < cExp * h * w)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu) :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW),
      (∀ i, CnxBlockChBounded (ps i) w' bb) →
      (∀ i, FloatBridges (layerNormForward c (ps i).εn 1 0)) →
      FloatBridges (convNextStageChK k ps)
  | 0, _, _, _ => floatBridges_idVec
  | _ + 1, ps, hb, hln =>
      (floatBridges_cnxBlockChW M fgelu (ps 0) hw' hbb hegelu hcc hc hcExp hg (hb 0) (hln 0)).comp
        (floatBridges_convNextStageChK M fgelu hw' hbb hegelu hcc hc hcExp hg _
          (fun i => ps i.succ) (fun i => hb i.succ) (fun i => hln i.succ))

/-- The per-downsample bound bundle at the channel LN (conv weight + bias, LN affine). -/
abbrev CnxDownChBounded {cin cout : Nat} (p : CnxDownParamsCh cin cout) (w' bb : ℝ) : Prop :=
  (∀ o c kh kw, |p.W o c kh kw| ≤ w') ∧ (∀ o, |p.b o| ≤ bb) ∧
  (∀ i, |p.γ i| ≤ w') ∧ (∀ i, |p.β i| ≤ bb)

/-- **The channel-LN downsample float-bridges** — `cnxDownChW = flatConvStride2 W ∘ chanLN`: the
    channel LayerNorm then the stride-2 widening conv. The peer of `floatBridges_cnxDownW`; note
    the LN runs at the PRE-downsample width `cin` over the `2h×2w` grid. -/
theorem floatBridges_cnxDownChW {cin cout : Nat} (h w : Nat) (M : FloatModel)
    (p : CnxDownParamsCh cin cout)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hcin : 0 < cin)
    (hn : 0 < cin * (2 * h) * (2 * w))
    (hbd : CnxDownChBounded p w' bb)
    (hln : FloatBridges (layerNormForward cin p.ε 1 0)) :
    FloatBridges (cnxDownChW h w p) := by
  obtain ⟨hW, hb, hγ, hβ⟩ := hbd
  unfold cnxDownChW
  exact (floatBridges_chanLNTensor3 M p.γ p.β hcin hγ hβ hln).comp
    (floatBridges_flatConvStride2 (h := h) (w := w) M p.W p.b hw' hbb hn hW hb)

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

set_option maxRecDepth 100000 in
/-- **THE SHIPPED ConvNeXt-T FORWARD float-bridges** (§2n) — `convnext_floatBridges` instantiated at
    ConvNeXt's REAL channel LayerNorm, with the whole [3,3,9,3] stage/downsample schedule plugged in.

    Two things make this a one-liner rather than a second whole-net proof, and both are §2m's doing:
    the skeleton already takes its LNs abstractly, and the reference's 22 LN sites are
    **1 stem + 18 block + 3 downsample** — no head LN — so the head slot is `lnHead := id`
    (`floatBridges_idVec`). The 22 supplied facts are the pure-normalise `layerNormForward c ε 1 0`
    bridges, one per site, each dischargeable by `floatBridges_bn`; the γ/β affine rides in
    `floatBridges_chanLNTensor3`, which is why the bounds bundles carry `γn`/`βn`.

    The tie from this skeleton to the committed `convNextForwardTCh` is
    `WholeNetForwardTies.convNextForwardTCh_eq_skeleton`. Closes under
    `[propext, Classical.choice, Quot.sound]`. -/
theorem convnextCh_floatBridges (M : FloatModel) (fgelu : ℝ → ℝ) (wts : CnxTWeightsCh)
    {w' bb egelu : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hsW : ∀ o c kh kw, |wts.sW o c kh kw| ≤ w') (hsb : ∀ o, |wts.sb o| ≤ bb)
    (hsγ : ∀ i, |wts.sγ i| ≤ w') (hsβ : ∀ i, |wts.sβ i| ≤ bb)
    (hWd : ∀ i j, |wts.Wd i j| ≤ w') (hbd : ∀ j, |wts.bd j| ≤ bb)
    (hb1 : ∀ i, CnxBlockChBounded (wts.s1 i) w' bb)
    (hb2 : ∀ i, CnxBlockChBounded (wts.s2 i) w' bb)
    (hb3 : ∀ i, CnxBlockChBounded (wts.s3 i) w' bb)
    (hb4 : ∀ i, CnxBlockChBounded (wts.s4 i) w' bb)
    (hbd1 : CnxDownChBounded wts.d1 w' bb) (hbd2 : CnxDownChBounded wts.d2 w' bb)
    (hbd3 : CnxDownChBounded wts.d3 w' bb)
    (hlnS : FloatBridges (layerNormForward 96 wts.sε 1 0))
    (hln1 : ∀ i, FloatBridges (layerNormForward 96 (wts.s1 i).εn 1 0))
    (hlnd1 : FloatBridges (layerNormForward 96 wts.d1.ε 1 0))
    (hln2 : ∀ i, FloatBridges (layerNormForward 192 (wts.s2 i).εn 1 0))
    (hlnd2 : FloatBridges (layerNormForward 192 wts.d2.ε 1 0))
    (hln3 : ∀ i, FloatBridges (layerNormForward 384 (wts.s3 i).εn 1 0))
    (hlnd3 : FloatBridges (layerNormForward 384 wts.d3.ε 1 0))
    (hln4 : ∀ i, FloatBridges (layerNormForward 768 (wts.s4 i).εn 1 0)) :
    FloatBridges (convnextForward wts.sW wts.sb wts.Wd wts.bd
      (chanLNTensor3 96 56 56 wts.sε wts.sγ wts.sβ) id
      (convNextStageChK 3 wts.s1) (cnxDownChW 28 28 wts.d1)
      (convNextStageChK 3 wts.s2) (cnxDownChW 14 14 wts.d2)
      (convNextStageChK 9 wts.s3) (cnxDownChW 7 7 wts.d3)
      (convNextStageChK 3 wts.s4)) :=
  convnext_floatBridges M wts.sW wts.sb wts.Wd wts.bd
    (chanLNTensor3 96 56 56 wts.sε wts.sγ wts.sβ) id
    (convNextStageChK 3 wts.s1) (cnxDownChW 28 28 wts.d1)
    (convNextStageChK 3 wts.s2) (cnxDownChW 14 14 wts.d2)
    (convNextStageChK 9 wts.s3) (cnxDownChW 7 7 wts.d3)
    (convNextStageChK 3 wts.s4)
    hw' hbb hw' hbb hsW hsb hWd hbd
    (floatBridges_chanLNTensor3 M wts.sγ wts.sβ (by norm_num) hsγ hsβ hlnS)
    floatBridges_idVec
    (floatBridges_convNextStageChK M fgelu hw' hbb hegelu (by norm_num) (by norm_num)
      (by norm_num) hg 3 wts.s1 hb1 hln1)
    (floatBridges_cnxDownChW 28 28 M wts.d1 hw' hbb (by norm_num) (by norm_num) hbd1 hlnd1)
    (floatBridges_convNextStageChK M fgelu hw' hbb hegelu (by norm_num) (by norm_num)
      (by norm_num) hg 3 wts.s2 hb2 hln2)
    (floatBridges_cnxDownChW 14 14 M wts.d2 hw' hbb (by norm_num) (by norm_num) hbd2 hlnd2)
    (floatBridges_convNextStageChK M fgelu hw' hbb hegelu (by norm_num) (by norm_num)
      (by norm_num) hg 9 wts.s3 hb3 hln3)
    (floatBridges_cnxDownChW 7 7 M wts.d3 hw' hbb (by norm_num) (by norm_num) hbd3 hlnd3)
    (floatBridges_convNextStageChK M fgelu hw' hbb hegelu (by norm_num) (by norm_num)
      (by norm_num) hg 3 wts.s4 hb4 hln4)

end Proofs
