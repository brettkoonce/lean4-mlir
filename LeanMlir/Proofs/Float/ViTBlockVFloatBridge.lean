import LeanMlir.Proofs.Float.FloatBudgetEnvAttn
import LeanMlir.Proofs.Architectures.ViTDepthK

/-! # The vector-LN transformer block, float-bridged and enveloped

Chunk 2 of ViT-Tiny's number (`planning/float_budget_numbers.md` §3.5.2): the
`FloatBridgesTo` + `Maps` chain for ONE `blockVFlat` — the block `vitForwardKV` actually
composes — closed at real weights, with the structural tie to the committed definition proved.

⛔ **This is a different block from the one the float tier already had.**
`floatBridges_vitBlockMHFull` (`ViTBlockFloatBridge.lean`) is stated about `transformerBlock`'s
SCALAR LayerNorm affines (`γ β : ℝ`); `vitForwardKV` uses `transformerBlockV`, whose affines are
vectors (`γ β : Vec D`) — and the trained checkpoint has vectors. ⚠ Same
`imagenet_specs_drift_from_twins` shape as ConvNeXt's stale head-LN slot (§3.3(b)): the two
statements look interchangeable until something forces them to unify, and what forces it is
needing the tie for a number.

⭐ **Most of the migration turned out to be already done.** §3.5.1 read the gap as "the ViT float
cone is `FloatBridges` everywhere", which is true of the *attention* half but not of the LN half:
`floatBridgesTo_rowLNVecFlat` and `Maps.rowLNVecFlat` already exist (`ChannelLNFloatBridge.lean` /
`FloatBudgetEnvLN.lean`), written for ConvNeXt's head LayerNorm, and `rowLNVecFlat N D ε γ β` IS
ViT's per-token vector LN — `rfl`. Likewise `floatBridgesTo_gelu` and `floatBridgesTo_dense`. So
what this file adds is the ASSEMBLY, not the leaves.

The three structural ties, and only one of them needed a proof:

* `flat_perTokenLN_eq` — the per-token LN's flat form IS `rowLNVecFlat`. `rfl`.
* `flat_transformerMlp_eq` — the MLP's flat form IS `perRowFlat` of its three per-token steps.
  `rfl`.
* `flat_mhsaLayer_eq` — `mhsa_layer`'s flat form is the per-row out-projection after
  `mhProjAttnFullFlat`. ⚠ NOT `rfl`: `mhProjAttnFullFlat` ends in `Mat.flatten` and `perRowFlat`
  begins with `Mat.unflatten`, so the roundtrip needs `Mat.unflatten_flatten`.

⚠ And one arithmetic wrinkle: `biPathMat (fun X => X) G` puts the IDENTITY first
(`M r s + G M r s`) where `Proofs.residual` puts the BODY first (`f v i + v i`). Equal, but by
`add_comm`, not by `rfl` — `flat_biPathMat_id` is where that is discharged. It matters because
`FloatBridgesTo.residual` names the float map `fun v j => M.add (fF v j) (v j)`, body first, and
a bridge whose float map is the other association is a different (if extensionally equal) term.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural ties
-- ════════════════════════════════════════════════════════════════

/-- **A `Mat`-level additive skip against the identity IS `Proofs.residual` at the flat level.**
    ⚠ `biPathMat (fun X => X) G` is `M r s + G M r s` and `residual f` is `f v i + v i`, so this
    is `add_comm` plus the `finProdFinEquiv` roundtrip — not `rfl`. -/
theorem flat_biPathMat_id {a b : Nat} (G : Mat a b → Mat a b) :
    (fun v : Vec (a * b) => Mat.flatten (biPathMat (fun X => X) G (Mat.unflatten v)))
      = Proofs.residual (fun v : Vec (a * b) => Mat.flatten (G (Mat.unflatten v))) := by
  funext v k
  show (Mat.unflatten v) _ _ + G (Mat.unflatten v) _ _
      = G (Mat.unflatten v) _ _ + v k
  rw [add_comm]
  congr 1
  show v (finProdFinEquiv (finProdFinEquiv.symm k)) = v k
  rw [Equiv.apply_symm_apply]

/-- Flattening distributes over `Mat`-level composition (the `unflatten ∘ flatten` roundtrip). -/
theorem flat_comp_mat {a b : Nat} (F G : Mat a b → Mat a b) :
    (fun v : Vec (a * b) => Mat.flatten ((F ∘ G) (Mat.unflatten v)))
      = (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v)))
        ∘ (fun v : Vec (a * b) => Mat.flatten (G (Mat.unflatten v))) := by
  funext v
  simp [Function.comp, Mat.unflatten_flatten]

/-- **The per-token vector LayerNorm's flat form IS `rowLNVecFlat`** — so ConvNeXt's head-LN
    bridge and envelope serve ViT's 25 LN sites unchanged. `rfl`. -/
theorem flat_perTokenLN_eq (N D : Nat) (ε : ℝ) (γ β : Vec D) :
    (fun v : Vec (N * D) => Mat.flatten (fun n => layerNormVec D ε γ β ((Mat.unflatten v) n)))
      = rowLNVecFlat N D ε γ β := rfl

/-- **The MLP's flat form is `perRowFlat` of its three per-token steps.** `rfl`. -/
theorem flat_transformerMlp_eq (N D mlpDim : Nat) (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D) :
    (fun v : Vec (N * D) =>
        Mat.flatten (transformerMlp N D mlpDim Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v)))
      = perRowFlat N D (Proofs.dense Wfc2 bfc2 ∘ gelu mlpDim ∘ Proofs.dense Wfc1 bfc1) := rfl

/-- **`mhsa_layer`'s flat form is the per-row out-projection after `mhProjAttnFullFlat`.**
    ⚠ Not `rfl` — `mhProjAttnFullFlat` ends in `Mat.flatten` and `perRowFlat` opens with
    `Mat.unflatten`. The Q/K/V projections, the per-head `sdpa` and the head concat all match
    definitionally; only the roundtrip needs rewriting. -/
theorem flat_mhsaLayer_eq (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v)))
      = perRowFlat N (heads * d_head) (Proofs.dense Wo bo)
          ∘ mhProjAttnFullFlat heads N d_head Wq Wk Wv bq bk bv := by
  funext v
  show Mat.flatten (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))
      = perRowFlat N (heads * d_head) (Proofs.dense Wo bo)
          (mhProjAttnFullFlat heads N d_head Wq Wk Wv bq bk bv v)
  unfold perRowFlat mhProjAttnFullFlat
  simp only [Mat.unflatten_flatten]
  rfl

/-- **THE BLOCK TIE.** The committed `blockVFlat` — `vitBodyKVFlat`'s per-block step, and the
    block `vitForwardKV` composes `k` of — IS two flat residual sublayers over the bridged
    pieces. Everything downstream is stated on the right-hand side, so the envelope provably
    bounds the net the graph denotes and not a look-alike. -/
theorem blockVFlat_eq (Np1 heads d_head mlpDim : Nat) (ε : ℝ)
    (p : BlockParamsV (heads * d_head) mlpDim) :
    blockVFlat Np1 heads d_head mlpDim ε p
      = Proofs.residual (perRowFlat Np1 (heads * d_head)
            (Proofs.dense p.Wfc2 p.bfc2 ∘ gelu mlpDim ∘ Proofs.dense p.Wfc1 p.bfc1)
          ∘ rowLNVecFlat Np1 (heads * d_head) ε p.γ2 p.β2)
        ∘ Proofs.residual (perRowFlat Np1 (heads * d_head) (Proofs.dense p.Wo p.bo)
            ∘ mhProjAttnFullFlat heads Np1 d_head p.Wq p.Wk p.Wv p.bq p.bk p.bv
            ∘ rowLNVecFlat Np1 (heads * d_head) ε p.γ1 p.β1) := by
  show (fun v : Vec (Np1 * (heads * d_head)) =>
      Mat.flatten (((transformerMlpSublayerV Np1 heads d_head mlpDim ε p.γ2 p.β2
                       p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) ∘
                    (transformerAttnSublayerV Np1 heads d_head ε p.γ1 p.β1
                       p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo))
        (Mat.unflatten v))) = _
  rw [flat_comp_mat]
  congr 1
  · show (fun v : Vec (Np1 * (heads * d_head)) =>
        Mat.flatten (biPathMat (fun X => X)
          ((transformerMlp Np1 (heads * d_head) mlpDim p.Wfc1 p.bfc1 p.Wfc2 p.bfc2) ∘
           (fun X : Mat Np1 (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε p.γ2 p.β2 (X n))) (Mat.unflatten v))) = _
    rw [flat_biPathMat_id, flat_comp_mat, flat_transformerMlp_eq, flat_perTokenLN_eq]
  · show (fun v : Vec (Np1 * (heads * d_head)) =>
        Mat.flatten (biPathMat (fun X => X)
          ((mhsa_layer Np1 heads d_head p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo) ∘
           (fun X : Mat Np1 (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε p.γ1 p.β1 (X n))) (Mat.unflatten v))) = _
    rw [flat_biPathMat_id, flat_comp_mat, flat_mhsaLayer_eq, flat_perTokenLN_eq]
    rfl

-- ════════════════════════════════════════════════════════════════
-- § 2. The block's parameter bounds, and the closed bridge
-- ════════════════════════════════════════════════════════════════

/-- The block's magnitude bounds, split by parameter KIND as the measured checkpoint splits
    them (`scripts/float_budget_envelope.py`, ViT-Tiny's profile): `wa` on the four attention
    kernels, `wm` on the two MLP kernels, `bb` on every bias, `gl` on the LayerNorm γ and `bl`
    on its β. ⚠ ViT-Tiny's kinds are only 2.5× apart — unlike ConvNeXt's 14× — so the split
    here buys tidiness rather than statability (§3.5). -/
structure BlockVBounded {D mlpDim : Nat} (p : BlockParamsV D mlpDim) (wa wm bb gl bl : ℝ) :
    Prop where
  hWq : ∀ i j, |p.Wq i j| ≤ wa
  hWk : ∀ i j, |p.Wk i j| ≤ wa
  hWv : ∀ i j, |p.Wv i j| ≤ wa
  hWo : ∀ i j, |p.Wo i j| ≤ wa
  hbq : ∀ j, |p.bq j| ≤ bb
  hbk : ∀ j, |p.bk j| ≤ bb
  hbv : ∀ j, |p.bv j| ≤ bb
  hbo : ∀ j, |p.bo j| ≤ bb
  hWfc1 : ∀ i j, |p.Wfc1 i j| ≤ wm
  hWfc2 : ∀ i j, |p.Wfc2 i j| ≤ wm
  hbfc1 : ∀ j, |p.bfc1 j| ≤ bb
  hbfc2 : ∀ j, |p.bfc2 j| ≤ bb
  hγ1 : ∀ i, |p.γ1 i| ≤ gl
  hγ2 : ∀ i, |p.γ2 i| ≤ gl
  hβ1 : ∀ i, |p.β1 i| ≤ bl
  hβ2 : ∀ i, |p.β2 i| ≤ bl

/-- **The float block** — `blockVFlat`'s deployed peer, named rather than existentially
    discarded (`formalization.yaml` fidelity §4d). Every stage is the float map its bridge
    names: `rowLNVecFlatF` at the two LayerNorm sites, `mhProjAttnFullFlatF` for attention,
    `M.dense` for the three projections, the supplied `fgelu` coordinatewise, and `M.add` for
    each skip — body first, matching `FloatBridgesTo.residual`. -/
noncomputable def blockVFlatF {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (p : BlockParamsV (heads * d_head) mlpDim)
    (lnF : Vec (heads * d_head) → Vec (heads * d_head)) :
    Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head)) :=
  (fun v j => M.add
      ((perRowFlat Np1 (heads * d_head)
          (M.dense p.Wfc2 p.bfc2 ∘ (fun x i => fgelu (x i)) ∘ M.dense p.Wfc1 p.bfc1)
        ∘ rowLNVecFlatF (s := Np1) M p.γ2 p.β2 lnF) v j) (v j))
    ∘ (fun v j => M.add
      ((perRowFlat Np1 (heads * d_head) (M.dense p.Wo p.bo)
          ∘ mhProjAttnFullFlatF M fexp heads Np1 d_head p.Wq p.Wk p.Wv p.bq p.bk p.bv
          ∘ rowLNVecFlatF (s := Np1) M p.γ1 p.β1 lnF) v j) (v j))

/-- **The vector-LN transformer block float-bridges, at real weights.** Two residual sublayers,
    seven bridged stages, no `FloatBridgesTo` hypothesis left except the device LayerNorm `hln`
    (whose statistics have no IEEE spec — the same standing of `DeviceRsqrt`/`DeviceSigmoid`)
    and the device `exp` accuracy behind `hfexp`/`hρ1`.

    Stated on `blockVFlat_eq`'s right-hand side, so `blockVFlat_eq` transports it to the
    committed `blockVFlat`. -/
noncomputable def floatBridgesTo_blockVFlat {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (ε : ℝ) (p : BlockParamsV (heads * d_head) mlpDim)
    (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    {wa wm bb gl bl egelu scaleA eexp : ℝ}
    (hn : 0 < Np1) (hd : 0 < heads * d_head) (hdff : 0 < mlpDim)
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp Np1 < 1)
    (hb : BlockVBounded p wa wm bb gl bl)
    (hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF) :
    FloatBridgesTo
      (Proofs.residual (perRowFlat Np1 (heads * d_head)
            (Proofs.dense p.Wfc2 p.bfc2 ∘ gelu mlpDim ∘ Proofs.dense p.Wfc1 p.bfc1)
          ∘ rowLNVecFlat Np1 (heads * d_head) ε p.γ2 p.β2)
        ∘ Proofs.residual (perRowFlat Np1 (heads * d_head) (Proofs.dense p.Wo p.bo)
            ∘ mhProjAttnFullFlat heads Np1 d_head p.Wq p.Wk p.Wv p.bq p.bk p.bv
            ∘ rowLNVecFlat Np1 (heads * d_head) ε p.γ1 p.β1))
      (blockVFlatF M fgelu fexp p lnF) :=
  let attnBody :=
    ((floatBridgesTo_rowLNVecFlat (s := Np1) M p.γ1 p.β1 lnF hd hb.hγ1 hb.hβ1 hln).comp
      (floatBridgesTo_mhProjAttnFullCap (h := heads) (n := Np1) (dh := d_head) M fexp
        p.Wq p.Wk p.Wv p.bq p.bk p.bv hn hwa hbb heexp0 heexp1 hfexp hscaleA hρ1
        hb.hWq hb.hbq hb.hWk hb.hbk hb.hWv hb.hbv).capped).comp
      (FloatBridgesTo.perRow Np1 (floatBridgesTo_dense M p.Wo p.bo hwa hbb hd hb.hWo hb.hbo))
  let mlpBody :=
    (floatBridgesTo_rowLNVecFlat (s := Np1) M p.γ2 p.β2 lnF hd hb.hγ2 hb.hβ2 hln).comp
      (FloatBridgesTo.perRow Np1
        (((floatBridgesTo_dense M p.Wfc1 p.bfc1 hwm hbb hd hb.hWfc1 hb.hbfc1).comp
          (floatBridgesTo_gelu (n := mlpDim) fgelu hegelu hg)).comp
          (floatBridgesTo_dense M p.Wfc2 p.bfc2 hwm hbb hdff hb.hWfc2 hb.hbfc2)))
  (attnBody.residual M).comp (mlpBody.residual M)

-- ════════════════════════════════════════════════════════════════
-- § 3. The block's numeric envelope
-- ════════════════════════════════════════════════════════════════

namespace FloatBridgesTo

set_option maxHeartbeats 1000000 in
/-- ⭐ **An envelope through one vector-LN transformer block.** Thirteen numeric stages and two
    skips: `LN₁ (3 stages) → attention (capped) → Wo → skip → LN₂ (3 stages) → fc1 → GELU →
    fc2 → skip`. Twenty-six inequalities, the same shape `Maps.cnxBlockChW` has at eighteen.

    ⛔ The attention stage is `Maps.mhProjAttnFullCap`, so its output error is `2·A4` by the
    triangle inequality and NOT the fold — and because a block's two skips carry that forward,
    every ViT number downstream of here is of that kind (§9). The LayerNorm sites are capped
    too (`Maps.rowLNVecFlat` runs on `floatBridgesTo_bn`, whose `Maps.bnCapped` is the capped
    leaf), so there is no stage in this block at which the fold survives.

    ⚠ Both LayerNorm sites take their OWN `mln`: the device-LN envelope depends on the input
    window, and the two sites see different ones. -/
theorem Maps.blockVFlat {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (ε : ℝ) (p : BlockParamsV (heads * d_head) mlpDim)
    (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    {wa wm bb gl bl egelu scaleA eexp : ℝ}
    (hn : 0 < Np1) (hd : 0 < heads * d_head) (hdff : 0 < mlpDim)
    (hnd : 0 < Np1 * (heads * d_head))
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp Np1 < 1)
    (hb : BlockVBounded p wa wm bb gl bl)
    (hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF)
    {q gd gm gattn sc Ā Ē : ℝ}
    {A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 A6 E6 : ℝ}
    {A7 E7 A8 E8 A9 E9 A10 E10 A11 E11 A12 E12 Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hgl0 : 0 ≤ gl) (hbl0 : 0 ≤ bl) (hsc0 : 0 ≤ sc)
    -- attention sublayer: LN₁, attention, out-projection, skip
    (mln1 : hln.Maps Ā Ē A1 E1)
    (l1gA : gl * A1 + FloatModel.mulErr q gl A1 0 0 ≤ A2)
    (l1gE : FloatModel.mulErr q gl A1 0 0 + gl * E1 ≤ E2)
    (l1bA : A2 + bl + q * (A2 + bl) ≤ A3) (l1bE : q * (A2 + bl) + E2 ≤ E3)
    (hgd : (1 + M.u) ^ (heads * d_head + 2) - 1 ≤ gd)
    (hgattn : (1 + M.u) ^ (Np1 + 1) - 1 ≤ gattn)
    (hscb : smCap M.u eexp Np1 ≤ sc)
    (atA : (1 + gattn) * ((Np1 : ℝ) * ((1 + sc)
             * ((1 + gd) * (((heads * d_head : ℕ) : ℝ) * wa * A3 + bb)))) ≤ A4)
    (atE : 2 * A4 ≤ E4)
    (woA : (1 + gd) * (((heads * d_head : ℕ) : ℝ) * wa * A4 + bb) ≤ A5)
    (woE : gd * (((heads * d_head : ℕ) : ℝ) * wa * (A4 + E4) + bb)
            + ((heads * d_head : ℕ) : ℝ) * wa * E4 ≤ E5)
    (r1A : A5 + Ā + q * (A5 + Ā) ≤ A6) (r1E : q * (A5 + E5 + Ā + Ē) + (E5 + Ē) ≤ E6)
    -- mlp sublayer: LN₂, fc1, GELU, fc2, skip
    (mln2 : hln.Maps A6 E6 A7 E7)
    (l2gA : gl * A7 + FloatModel.mulErr q gl A7 0 0 ≤ A8)
    (l2gE : FloatModel.mulErr q gl A7 0 0 + gl * E7 ≤ E8)
    (l2bA : A8 + bl + q * (A8 + bl) ≤ A9) (l2bE : q * (A8 + bl) + E8 ≤ E9)
    (f1A : (1 + gd) * (((heads * d_head : ℕ) : ℝ) * wm * A9 + bb) ≤ A10)
    (f1E : gd * (((heads * d_head : ℕ) : ℝ) * wm * (A9 + E9) + bb)
            + ((heads * d_head : ℕ) : ℝ) * wm * E9 ≤ E10)
    (geA : A10 + egelu ≤ A11) (geE : egelu + 3 / 2 * E10 ≤ E11)
    (hgm : (1 + M.u) ^ (mlpDim + 2) - 1 ≤ gm)
    (f2A : (1 + gm) * ((mlpDim : ℝ) * wm * A11 + bb) ≤ A12)
    (f2E : gm * ((mlpDim : ℝ) * wm * (A11 + E11) + bb) + (mlpDim : ℝ) * wm * E11 ≤ E12)
    (r2A : A12 + A6 + q * (A12 + A6) ≤ Ā') (r2E : q * (A12 + E12 + A6 + E6) + (E12 + E6) ≤ Ē') :
    (floatBridgesTo_blockVFlat M fgelu fexp ε p lnF hn hd hdff hwa hwm hbb hegelu hg
      heexp0 heexp1 hfexp hscaleA hρ1 hb hln).Maps Ā Ē Ā' Ē' := by
  -- attention sublayer body
  have hln1 := Maps.rowLNVecFlat (s := Np1) M p.γ1 p.β1 lnF hd hb.hγ1 hb.hβ1 hln hq hgl0 hbl0
    mln1 l1gA l1gE l1bA l1bE
  have hattn := Maps.mhProjAttnFullCap (h := heads) (n := Np1) (dh := d_head) M fexp
    p.Wq p.Wk p.Wv p.bq p.bk p.bv hn hwa hbb heexp0 heexp1 hfexp hscaleA hρ1
    hb.hWq hb.hbq hb.hWk hb.hbk hb.hWv hb.hbv (Ā := A3) (Ē := E3) hgd hgattn hsc0 hscb atA atE
  have hwo := Maps.perRow Np1 (Maps.dense M p.Wo p.bo hwa hbb hd hb.hWo hb.hbo hgd woA woE)
  have hA1 := (hln1.comp hnd hattn).comp hnd hwo
  have hAres := Maps.residual M hnd hA1 hq r1A r1E
  -- mlp sublayer body
  have hln2 := Maps.rowLNVecFlat (s := Np1) M p.γ2 p.β2 lnF hd hb.hγ2 hb.hβ2 hln hq hgl0 hbl0
    mln2 l2gA l2gE l2bA l2bE
  have hmlp := Maps.perRow Np1
    (((Maps.dense M p.Wfc1 p.bfc1 hwm hbb hd hb.hWfc1 hb.hbfc1 hgd f1A f1E).comp hdff
        (Maps.gelu (n := mlpDim) fgelu hegelu hg geA geE)).comp hdff
      (Maps.dense M p.Wfc2 p.bfc2 hwm hbb hdff hb.hWfc2 hb.hbfc2 hgm f2A f2E))
  have hM1 := hln2.comp hnd hmlp
  have hMres := Maps.residual M hnd hM1 hq r2A r2E
  exact hAres.comp hnd hMres

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § 5. The depth-`k` fold
-- ════════════════════════════════════════════════════════════════

/-- **Transport a bridge along an equation of the REAL map, keeping `mag`/`mod` definitionally.**
    ⚠ Not `blockVFlat_eq ▸ b`: a `▸` inserts an `Eq.mpr` that blocks `.mag` from reducing, and a
    bridge whose `.mag` does not reduce cannot have a `Maps` attached (§2 — the unifier unfolds
    the whole chain and times out). Rebuilding the structure field-by-field keeps `.mag` and
    `.mod` literally `b`'s, which is why `Maps.ofEq` below is the identity. -/
noncomputable def FloatBridgesTo.ofEq {m n : Nat} {f f' fF : Vec m → Vec n}
    (h : f = f') (b : FloatBridgesTo f' fF) : FloatBridgesTo f fF where
  mag := b.mag
  mod := b.mod
  close := h ▸ b.close

/-- The envelope survives the transport unchanged — `mag`/`mod` are the same terms. -/
theorem FloatBridgesTo.Maps.ofEq {m n : Nat} {f f' fF : Vec m → Vec n} (h : f = f')
    {b : FloatBridgesTo f' fF} {Ā Ē Ā' Ē' : ℝ} (hM : b.Maps Ā Ē Ā' Ē') :
    (FloatBridgesTo.ofEq h b).Maps Ā Ē Ā' Ē' := ⟨hM.mag_le, hM.mod_le⟩

/-- **The block bridge, on the COMMITTED `blockVFlat`.** `floatBridgesTo_blockVFlat` is stated on
    `blockVFlat_eq`'s right-hand side; this is the same data at the definition `vitBodyKVFlat`
    actually folds. -/
noncomputable def floatBridgesTo_blockVFlatC {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (ε : ℝ) (p : BlockParamsV (heads * d_head) mlpDim)
    (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    {wa wm bb gl bl egelu scaleA eexp : ℝ}
    (hn : 0 < Np1) (hd : 0 < heads * d_head) (hdff : 0 < mlpDim)
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp Np1 < 1)
    (hb : BlockVBounded p wa wm bb gl bl)
    (hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF) :
    FloatBridgesTo (blockVFlat Np1 heads d_head mlpDim ε p) (blockVFlatF M fgelu fexp p lnF) :=
  FloatBridgesTo.ofEq (blockVFlat_eq Np1 heads d_head mlpDim ε p)
    (floatBridgesTo_blockVFlat M fgelu fexp ε p lnF hn hd hdff hwa hwm hbb hegelu hg
      heexp0 heexp1 hfexp hscaleA hρ1 hb hln)

/-- The float depth-`k` body — the same head-first recursion as `vitBodyKVFlat`, so it is exactly
    what `.comp` produces at each step. -/
noncomputable def vitBodyKVFlatF {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (lnF : Vec (heads * d_head) → Vec (heads * d_head)) :
    (k : Nat) → (Fin k → BlockParamsV (heads * d_head) mlpDim) →
      (Vec (Np1 * (heads * d_head)) → Vec (Np1 * (heads * d_head)))
  | 0, _ => id
  | _ + 1, ps =>
      vitBodyKVFlatF (Np1 := Np1) M fgelu fexp lnF _ (fun i => ps i.succ)
        ∘ blockVFlatF (Np1 := Np1) M fgelu fexp (ps 0) lnF

/-- **The depth-`k` encoder body float-bridges**, by the same head-first recursion the definition
    uses — block `0` applied FIRST, so the `.comp` puts it on the left of the composition.
    ⚠ That association is the opposite of `floatBridgesTo_convNextStageChK`'s reading in §3.3's
    lesson 2, and it is the definition that decides: check it before writing the chain. -/
noncomputable def floatBridgesTo_vitBodyKVFlat {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (ε : ℝ) (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    {wa wm bb gl bl egelu scaleA eexp : ℝ}
    (hn : 0 < Np1) (hd : 0 < heads * d_head) (hdff : 0 < mlpDim)
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp Np1 < 1)
    (hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim),
      (∀ i, BlockVBounded (ps i) wa wm bb gl bl) →
      FloatBridgesTo (vitBodyKVFlat Np1 heads d_head mlpDim ε k ps)
        (vitBodyKVFlatF (Np1 := Np1) M fgelu fexp lnF k ps)
  | 0, _, _ => floatBridgesTo_idVec
  | _ + 1, ps, hb =>
      (floatBridgesTo_blockVFlatC M fgelu fexp ε (ps 0) lnF hn hd hdff hwa hwm hbb hegelu hg
          heexp0 heexp1 hfexp hscaleA hρ1 (hb 0) hln).comp
        (floatBridgesTo_vitBodyKVFlat M fgelu fexp ε lnF hn hd hdff hwa hwm hbb hegelu hg
          heexp0 heexp1 hfexp hscaleA hρ1 hln _ (fun i => ps i.succ) (fun i => hb i.succ))

namespace FloatBridgesTo

/-- `Maps.blockVFlat` transported to the committed `blockVFlat` — the form the depth-`k` fold
    consumes. One wrapper rather than a second 26-argument statement. -/
theorem Maps.blockVFlatC {Np1 heads d_head mlpDim : Nat} {M : FloatModel}
    {fgelu fexp : ℝ → ℝ} {ε : ℝ} {p : BlockParamsV (heads * d_head) mlpDim}
    {lnF : Vec (heads * d_head) → Vec (heads * d_head)}
    {wa wm bb gl bl egelu scaleA eexp : ℝ}
    {hn : 0 < Np1} {hd : 0 < heads * d_head} {hdff : 0 < mlpDim}
    {hwa : 0 ≤ wa} {hwm : 0 ≤ wm} {hbb : 0 ≤ bb} {hegelu : 0 ≤ egelu}
    {hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu}
    {heexp0 : 0 ≤ eexp} {heexp1 : eexp ≤ 1}
    {hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t}
    {hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA}
    {hρ1 : smRho M.u eexp Np1 < 1}
    {hb : BlockVBounded p wa wm bb gl bl}
    {hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF}
    {Ā Ē Ā' Ē' : ℝ}
    (hM : (floatBridgesTo_blockVFlat M fgelu fexp ε p lnF hn hd hdff hwa hwm hbb hegelu hg
            heexp0 heexp1 hfexp hscaleA hρ1 hb hln).Maps Ā Ē Ā' Ē') :
    (floatBridgesTo_blockVFlatC M fgelu fexp ε p lnF hn hd hdff hwa hwm hbb hegelu hg
      heexp0 heexp1 hfexp hscaleA hρ1 hb hln).Maps Ā Ē Ā' Ē' :=
  Maps.ofEq _ hM

/-- ⭐ **An envelope through the depth-`k` body**, from a per-block one. The caller supplies the
    window/error SEQUENCES `W`/`Er` — `W j` is the certified window after `j` blocks — and one
    `Maps` per block; the fold threads them. ⭐ ConvNeXt has the bridge fold but no envelope fold,
    so its budget file spells every stage out; at ViT's depth 12 that is 12 nested `.comp`s by
    hand, and this replaces them with one application. -/
theorem Maps.vitBodyKVFlat {Np1 heads d_head mlpDim : Nat} (M : FloatModel)
    (fgelu fexp : ℝ → ℝ) (ε : ℝ) (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    {wa wm bb gl bl egelu scaleA eexp : ℝ}
    (hn : 0 < Np1) (hd : 0 < heads * d_head) (hdff : 0 < mlpDim)
    (hnd : 0 < Np1 * (heads * d_head))
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp Np1 < 1)
    (hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
      (hb : ∀ i, BlockVBounded (ps i) wa wm bb gl bl) (W Er : Nat → ℝ),
      (∀ i : Fin k,
        (floatBridgesTo_blockVFlatC M fgelu fexp ε (ps i) lnF hn hd hdff hwa hwm hbb hegelu hg
          heexp0 heexp1 hfexp hscaleA hρ1 (hb i) hln).Maps
          (W i.val) (Er i.val) (W (i.val + 1)) (Er (i.val + 1))) →
      (floatBridgesTo_vitBodyKVFlat M fgelu fexp ε lnF hn hd hdff hwa hwm hbb hegelu hg
        heexp0 heexp1 hfexp hscaleA hρ1 hln k ps hb).Maps (W 0) (Er 0) (W k) (Er k)
  | 0, _, _, _, _, _ => ⟨fun _A _h0 hle => hle, fun _A _E _h0 _hE0 _hle hEle => hEle⟩
  | k + 1, ps, hb, W, Er, hstep =>
      Maps.comp hnd (hstep 0)
        (Maps.vitBodyKVFlat M fgelu fexp ε lnF hn hd hdff hnd hwa hwm hbb hegelu hg
          heexp0 heexp1 hfexp hscaleA hρ1 hln k (fun i => ps i.succ) (fun i => hb i.succ)
          (fun j => W (j + 1)) (fun j => Er (j + 1)) (fun i => hstep i.succ))

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § 7. The whole net
-- ════════════════════════════════════════════════════════════════

/-- **`vitForwardKV` is the four-stage skeleton** the bridges below cover: patch embed → depth-`k`
    body → per-token final LayerNorm → CLS-slice + classifier. `rfl` — the final LN slot is
    `flat_perTokenLN_eq`'s left-hand side at `N + 1` rows. -/
theorem vitForwardKV_eq (ic H W patchSize N mlpDim heads d_head nClasses k : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head)) (ε : ℝ)
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls
      = classifier_flat N (heads * d_head) nClasses Wcls bcls
        ∘ rowLNVecFlat (N + 1) (heads * d_head) ε γF βF
        ∘ vitBodyKVFlat (N + 1) heads d_head mlpDim ε k ps
        ∘ patchEmbed_flat ic H W patchSize N (heads * d_head)
            W_conv b_conv cls_token pos_embed := rfl

/-- The float whole net — every stage the float map its bridge names. -/
noncomputable def vitForwardKVF {ic H W patchSize N mlpDim heads d_head nClasses k : Nat}
    (M : FloatModel) (fgelu fexp : ℝ → ℝ)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head)) (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (M.dense Wcls bcls ∘ cls_slice_flat N (heads * d_head))
    ∘ rowLNVecFlatF (s := N + 1) M γF βF lnF
    ∘ vitBodyKVFlatF (Np1 := N + 1) M fgelu fexp lnF k ps
    ∘ M.patchEmbedF ic H W patchSize N (heads * d_head) W_conv b_conv cls_token pos_embed

/-- ⭐⭐ **THE WHOLE ViT FORWARD FLOAT-BRIDGES, AT REAL WEIGHTS.** Four `.comp`s over the patch
    embed, the depth-`k` body, the final per-token LayerNorm and the classifier head — no
    `FloatBridgesTo` hypothesis left except the device LayerNorm `hln`, whose statistics have no
    IEEE specification (the standing of `DeviceRsqrt`/`DeviceSigmoid` on the other nets), and the
    device `exp`/`gelu` accuracies behind `hfexp`/`hg`.

    ⛔ This is `vitForwardKV` — depth-`k`, **distinct per-block parameters, vector-[D] LayerNorm
    affines, multi-head** — and NOT `vit_full`, which shares one parameter tuple across all
    blocks and carries scalar affines. The trained checkpoint has per-block weights and vector
    affines, so `vit_full` is a different function (`planning/float_budget_numbers.md` §3.5.1).
    `vitFwdGraphKMHV_faithful` denotes this one. -/
noncomputable def floatBridgesTo_vitForwardKV {ic H W patchSize N mlpDim heads d_head nClasses
    k : Nat} (M : FloatModel) (fgelu fexp : ℝ → ℝ) (ε : ℝ)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head)) (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    {wa wm bb gl bl egelu scaleA eexp wc pb wh bh : ℝ}
    (hn : 0 < N + 1) (hd : 0 < heads * d_head) (hdff : 0 < mlpDim)
    (hnd : 0 < (N + 1) * (heads * d_head)) (himg : 0 < ic * H * W)
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hwc0 : 0 ≤ wc) (hpb0 : 0 ≤ pb) (hwh : 0 ≤ wh) (hbh : 0 ≤ bh)
    (hgl : ∀ i, |γF i| ≤ gl) (hbl : ∀ i, |βF i| ≤ bl)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp (N + 1) < 1)
    (hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc) (hpos : ∀ n d, |pos_embed n d| ≤ pb)
    (hcls : ∀ d, |cls_token d| ≤ pb) (hbc : ∀ d, |b_conv d| ≤ pb)
    (hWcls : ∀ i j, |Wcls i j| ≤ wh) (hbcls : ∀ j, |bcls j| ≤ bh)
    (hb : ∀ i, BlockVBounded (ps i) wa wm bb gl bl)
    (hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF) :
    FloatBridgesTo
      (vitForwardKV ic H W patchSize N mlpDim heads d_head nClasses k
        W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
      (vitForwardKVF M fgelu fexp W_conv b_conv cls_token pos_embed ps γF βF lnF Wcls bcls) :=
  FloatBridgesTo.ofEq
    (vitForwardKV_eq ic H W patchSize N mlpDim heads d_head nClasses k
      W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
    ((((floatBridgesTo_patchEmbed M ic H W patchSize N (heads * d_head)
            W_conv b_conv cls_token pos_embed hwc0 hpb0 hnd himg hwc hpos hcls hbc).comp
        (floatBridgesTo_vitBodyKVFlat M fgelu fexp ε lnF hn hd hdff hwa hwm hbb hegelu hg
          heexp0 heexp1 hfexp hscaleA hρ1 hln k ps hb)).comp
        (floatBridgesTo_rowLNVecFlat (s := N + 1) M γF βF lnF hd hgl hbl hln)).comp
        (floatBridgesTo_vitHead M Wcls bcls hwh hbh hd hWcls hbcls))

namespace FloatBridgesTo

/-- The CLS-slice gather passes an envelope through unchanged (it reads one row; no rounding). -/
theorem Maps.clsSlice (N D : Nat) {Ā Ē : ℝ} :
    (floatBridgesTo_clsSlice N D).Maps Ā Ē Ā Ē :=
  ⟨fun _A _h0 hle => hle, fun _A _E _h0 _hE0 _hle hEle => hEle⟩

/-- An envelope through the classifier head — the CLS gather, then the dense. -/
theorem Maps.vitHead (N : Nat) {D nClasses : Nat} (M : FloatModel) (Wcls : Mat D nClasses)
    (bcls : Vec nClasses) {wh bh : ℝ} (hwh : 0 ≤ wh) (hbh : 0 ≤ bh) (hD : 0 < D)
    (hWcls : ∀ i j, |Wcls i j| ≤ wh) (hbcls : ∀ j, |bcls j| ≤ bh)
    {g Ā Ē Ā' Ē' : ℝ}
    (hg : (1 + M.u) ^ (D + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * ((D : ℝ) * wh * Ā + bh) ≤ Ā')
    (hĒ' : g * ((D : ℝ) * wh * (Ā + Ē) + bh) + (D : ℝ) * wh * Ē ≤ Ē') :
    (floatBridgesTo_vitHead (N := N) M Wcls bcls hwh hbh hD hWcls hbcls).Maps Ā Ē Ā' Ē' :=
  (Maps.clsSlice N D).comp hD (Maps.dense M Wcls bcls hwh hbh hD hWcls hbcls hg hĀ' hĒ')

/-- ⭐⭐ **THE WHOLE-NET ENVELOPE.** Four stages: patch embed → depth-`k` body → final per-token
    LayerNorm → classifier head. Everything numeric is in the four supplied `Maps`, so a budget
    file supplies numerals and nothing else. -/
theorem Maps.vitForwardKV {ic H W patchSize N mlpDim heads d_head nClasses k : Nat}
    (M : FloatModel) (fgelu fexp : ℝ → ℝ) (ε : ℝ)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ps : Fin k → BlockParamsV (heads * d_head) mlpDim)
    (γF βF : Vec (heads * d_head)) (lnF : Vec (heads * d_head) → Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    {wa wm bb gl bl egelu scaleA eexp wc pb wh bh : ℝ}
    {hn : 0 < N + 1} {hd : 0 < heads * d_head} {hdff : 0 < mlpDim}
    (hnd : 0 < (N + 1) * (heads * d_head)) {himg : 0 < ic * H * W}
    {hwa : 0 ≤ wa} {hwm : 0 ≤ wm} {hbb : 0 ≤ bb} {hegelu : 0 ≤ egelu}
    {hwc0 : 0 ≤ wc} {hpb0 : 0 ≤ pb} {hwh : 0 ≤ wh} {hbh : 0 ≤ bh}
    {hgl : ∀ i, |γF i| ≤ gl} {hbl : ∀ i, |βF i| ≤ bl}
    {hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu}
    {heexp0 : 0 ≤ eexp} {heexp1 : eexp ≤ 1}
    {hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t}
    {hscaleA : |(1 : ℝ) / Real.sqrt (d_head : ℝ)| ≤ scaleA}
    {hρ1 : smRho M.u eexp (N + 1) < 1}
    {hwc : ∀ d c kh kw, |W_conv d c kh kw| ≤ wc} {hpos : ∀ n d, |pos_embed n d| ≤ pb}
    {hcls : ∀ d, |cls_token d| ≤ pb} {hbc : ∀ d, |b_conv d| ≤ pb}
    {hWcls : ∀ i j, |Wcls i j| ≤ wh} {hbcls : ∀ j, |bcls j| ≤ bh}
    {hb : ∀ i, BlockVBounded (ps i) wa wm bb gl bl}
    {hln : FloatBridgesTo (layerNormForward (heads * d_head) ε 1 0) lnF}
    {Ā Ē A1 E1 A2 E2 A3 E3 Ā' Ē' : ℝ}
    (mPatch : (floatBridgesTo_patchEmbed M ic H W patchSize N (heads * d_head)
        W_conv b_conv cls_token pos_embed hwc0 hpb0 hnd himg hwc hpos hcls hbc).Maps Ā Ē A1 E1)
    (mBody : (floatBridgesTo_vitBodyKVFlat M fgelu fexp ε lnF hn hd hdff hwa hwm hbb hegelu hg
        heexp0 heexp1 hfexp hscaleA hρ1 hln k ps hb).Maps A1 E1 A2 E2)
    (mLN : (floatBridgesTo_rowLNVecFlat (s := N + 1) M γF βF lnF hd hgl hbl hln).Maps
        A2 E2 A3 E3)
    (mHead : (floatBridgesTo_vitHead (N := N) M Wcls bcls hwh hbh hd hWcls hbcls).Maps
        A3 E3 Ā' Ē') :
    (floatBridgesTo_vitForwardKV M fgelu fexp ε W_conv b_conv cls_token pos_embed ps γF βF lnF
      Wcls bcls hn hd hdff hnd himg hwa hwm hbb hegelu hwc0 hpb0 hwh hbh hgl hbl hg
      heexp0 heexp1 hfexp hscaleA hρ1 hwc hpos hcls hbc hWcls hbcls hb hln).Maps Ā Ē Ā' Ē' :=
  Maps.ofEq _ (((mPatch.comp hnd mBody).comp hnd mLN).comp hnd mHead)

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § 8. ViT-Tiny's block 0, closed at the emitted numerals
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 2000000 in
/-- ⭐⭐ **ViT-Tiny's block 0, end to end, at `vit_chain`'s numerals** — thirteen stages, two
    skips, twenty-six inequalities, all `norm_num`. `heads = 3`, `d_head = 64`, `mlpDim = 768`,
    `Np1 = 197` tokens; the profile is the measured checkpoint's, split by parameter kind
    (attention kernels `7/10`, MLP kernels `8/10`, biases `9/10`, LN γ `17/10`, LN β `6/10`).

    In: the patch embed's `(23.23, 5.633·10⁻⁴)`. Out: `(9.365·10¹⁹, 2.811·10²⁰)` — one block
    costs ~10¹⁸ of window, and twelve of them are what put the whole net at 10²¹⁸.

    ⚠ The device-LayerNorm envelopes `mln1`/`mln2` are hypotheses, as they must be: a device
    LN's mean and inverse-stddev have no IEEE specification, so their accuracy is supplied
    (the standing of `DeviceRsqrt`/`DeviceSigmoid`/`DeviceGelu` everywhere else in this cone).
    At the shipped profile they are `Maps.bnCapped` at `ε ≥ 10⁻⁵`, `emr = ei = 10⁻²`. -/
example (M : FloatModel) (hMu : M.u ≤ u32) (fgelu fexp : ℝ → ℝ)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ 1 / 100)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ (1 / 100) * Real.exp t)
    (hρ : smRho M.u (1 / 100) 197 < 1)
    (hscaleA : |(1 : ℝ) / Real.sqrt ((64 : ℕ) : ℝ)| ≤ 1 / 8)
    (ε : ℝ) (p : BlockParamsV (3 * 64) 768)
    (lnF : Vec (3 * 64) → Vec (3 * 64))
    (hb : BlockVBounded p (7 / 10) (8 / 10) (9 / 10) (17 / 10) (6 / 10))
    (hln : FloatBridgesTo (layerNormForward (3 * 64) ε 1 0) lnF)
    (mln1 : hln.Maps (2323 / 10 ^ 1) (5633 / 10 ^ 7) (1481 * 10 ^ 2) (2962 * 10 ^ 2))
    (mln2 : hln.Maps (9148 * 10 ^ 8) (1831 * 10 ^ 9) (5830 * 10 ^ 11) (1166 * 10 ^ 12)) :
    (floatBridgesTo_blockVFlat (Np1 := 197) M fgelu fexp ε p lnF (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hg
      (by norm_num) (by norm_num) hfexp hscaleA hρ hb hln).Maps
      (2323 / 10 ^ 1) (5633 / 10 ^ 7) (9365 * 10 ^ 16) (2811 * 10 ^ 17) := by
  have hsc : smCap M.u (1 / 100) 197 ≤ 2022 / 10 ^ 5 :=
    smCap_le M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6) (c := 2022 / 10 ^ 5)
      (by norm_num) hMu
      (smRho_le_of M (n := 197) (eexp := 1 / 100) (rb := 10012 / 10 ^ 6)
        (M.gamma_num (k := 198) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) (by norm_num)) (by norm_num) (by norm_num [u32])
  exact FloatBridgesTo.Maps.blockVFlat M fgelu fexp ε p lnF (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) hg
    (by norm_num) (by norm_num) hfexp hscaleA hρ hb hln
    (q := u32) (gd := 1157 / 10 ^ 8) (gm := 4590 / 10 ^ 8) (gattn := 1181 / 10 ^ 8)
    (sc := 2022 / 10 ^ 5)
    (A1 := 1481 * 10 ^ 2) (E1 := 2962 * 10 ^ 2)
    (A2 := 2518 * 10 ^ 2) (E2 := 5036 * 10 ^ 2)
    (A3 := 2519 * 10 ^ 2) (E3 := 5037 * 10 ^ 2)
    (A4 := 6805 * 10 ^ 6) (E4 := 1361 * 10 ^ 7)
    (A5 := 9147 * 10 ^ 8) (E5 := 1830 * 10 ^ 9)
    (A6 := 9148 * 10 ^ 8) (E6 := 1831 * 10 ^ 9)
    (A7 := 5830 * 10 ^ 11) (E7 := 1166 * 10 ^ 12)
    (A8 := 9912 * 10 ^ 11) (E8 := 1983 * 10 ^ 12)
    (A9 := 9913 * 10 ^ 11) (E9 := 1984 * 10 ^ 12)
    (A10 := 1523 * 10 ^ 14) (E10 := 3048 * 10 ^ 14)
    (A11 := 1524 * 10 ^ 14) (E11 := 4573 * 10 ^ 14)
    (A12 := 9364 * 10 ^ 16) (E12 := 2810 * 10 ^ 17)
    hMu (by norm_num) (by norm_num) (by norm_num)
    mln1 (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (by norm_num [u32]) (by norm_num [u32])
    (M.gamma_num (k := 3 * 64 + 2) (q := 1157 / 10 ^ 8) hMu (by norm_num [u32])
      (by norm_num [u32]))
    (M.gamma_num (k := 197 + 1) (q := 1181 / 10 ^ 8) hMu (by norm_num [u32])
      (by norm_num [u32]))
    hsc (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [u32]) (by norm_num [u32])
    mln2 (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (by norm_num [u32]) (by norm_num [u32])
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (M.gamma_num (k := 768 + 2) (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
    (by norm_num) (by norm_num)
    (by norm_num [u32]) (by norm_num [u32])

/-- ⭐ **The depth-`k` fold's sequence threading, exercised at `k = 2`.** The two per-block
    envelopes are hypotheses (block 0's arithmetic is closed above; what is under test here is
    that the fold hands block `i`'s OUTPUT window to block `i+1` as its input and comes out at
    `W k`). ⚠ The recursion is head-first, so `ps 0` is applied FIRST — the opposite association
    from `floatBridgesTo_convNextStageChK`, and the definition is what decides. -/
example (M : FloatModel) (fgelu fexp : ℝ → ℝ) (ε : ℝ)
    (lnF : Vec (3 * 64) → Vec (3 * 64))
    (wa wm bb gl bl egelu scaleA eexp : ℝ)
    (hn : 0 < 197) (hd : 0 < 3 * 64) (hdff : 0 < 768) (hnd : 0 < 197 * (3 * 64))
    (hwa : 0 ≤ wa) (hwm : 0 ≤ wm) (hbb : 0 ≤ bb) (hegelu : 0 ≤ egelu)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt ((64 : Nat) : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp 197 < 1)
    (hln : FloatBridgesTo (layerNormForward (3 * 64) ε 1 0) lnF)
    (ps : Fin 2 → BlockParamsV (3 * 64) 768)
    (hb : ∀ i, BlockVBounded (ps i) wa wm bb gl bl)
    (A0 E0 A1 E1 A2 E2 : ℝ)
    (h0 : (floatBridgesTo_blockVFlatC (Np1 := 197) M fgelu fexp ε (ps 0) lnF hn hd hdff
            hwa hwm hbb hegelu hg heexp0 heexp1 hfexp hscaleA hρ1 (hb 0) hln).Maps A0 E0 A1 E1)
    (h1 : (floatBridgesTo_blockVFlatC (Np1 := 197) M fgelu fexp ε (ps 1) lnF hn hd hdff
            hwa hwm hbb hegelu hg heexp0 heexp1 hfexp hscaleA hρ1 (hb 1) hln).Maps A1 E1 A2 E2) :
    (floatBridgesTo_vitBodyKVFlat (Np1 := 197) M fgelu fexp ε lnF hn hd hdff hwa hwm hbb hegelu
      hg heexp0 heexp1 hfexp hscaleA hρ1 hln 2 ps hb).Maps A0 E0 A2 E2 := by
  refine FloatBridgesTo.Maps.vitBodyKVFlat M fgelu fexp ε lnF hn hd hdff hnd hwa hwm hbb hegelu
    hg heexp0 heexp1 hfexp hscaleA hρ1 hln 2 ps hb
    (fun j => match j with | 0 => A0 | 1 => A1 | _ => A2)
    (fun j => match j with | 0 => E0 | 1 => E1 | _ => E2) ?_
  intro i
  fin_cases i
  · exact h0
  · exact h1

end Proofs
