import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StepTieB
import LeanMlir.Proofs.Foundation.ParamGradNodes
import LeanMlir.Proofs.Foundation.GradNodesBAt

/-! # MobileNetV2 — every parameter gradient node IS the loss's derivative in that parameter

`mnv2_net_tiedB` says each of the 210 parameter gradient nodes (158 emitted at the default
`convBias := false`) denotes its layer's parameter Jacobian contracted with the cotangent the
emitted backward chain threads to it, from a loss cotangent `g`; the `*_eq_vjp` lemmas say the
chain's segments are certified VJP backwards. `mnv2_net_lossGrad` composes them: for any loss `L`
of the logits whose gradient at the net's output is `g`, every node is `∂L/∂θ` of the WHOLE net with
that one parameter varied. `mnv2_net_lossGrad_smoothedCE` discharges `hL` for the label-smoothed
loss the artifacts ship.

**Scope.** One replica, at either precision: every conv and depthwise weight node is stated on
the renderers' switch (`convWeightGradBAt bf16 id …`, `depthwiseWeightGradBAt bf16 id …`, …;
`Foundation.GradNodesBAt`), so `bf16 := true` is the bf16 kind the `mobilenetv2*bf16` artifacts
emit — `mobilenetv2in_rmsdp64wxdols0eps0001bf16`, the book's ImageNet run, among them — read over ℝ
at the identity rounding (`Bf16Erasure`), and `bf16 := false` the f32 artifacts'. Classifier
dropout (`*do*`) is outside this statement. Sync-BN data parallelism is reached by composition: `mnv2_net_syncTiedB` says each all-reduced
gradient is this net's tied node at `N := R·N`, which this file's capstone makes the loss's
gradient at the global batch.

**How.** `ResNet50ParamGrad`'s shape:

* **Per stage** (at variable widths): the loss read at the output of the stem, the `t = 1` block,
  the stride-1 body, the stride-2 body and the head, pulled back one stage at a time by
  `ParamGradNodes`' pull-backs (relu6's `hasGradAt_relu6`, batch BN, the conv and depthwise
  input-VJPs) and this file's XLA-`SAME` strided depthwise `hasGradAt_depthwiseStridedXla`, so each
  internal activation's gradient is the chain's own cotangent. The stride-1 body bundle covers
  both the skip blocks and the two widenings: a skip block's body sees the loss `u ↦ Gn (u + v)`,
  whose gradient at the body output is still `dyOut` (`mnv2_resid_lossTiedB`).
* **Per net**: the loss read after each block (`mnv2Suf*`), pulled back through the seventeen
  certified block VJPs and the head's, and each `Φ` identified with the whole net at updated
  weights by a standalone `mnv2_factor_*` theorem.

**Hypotheses.** `MNV2PosB` (every BN `ε > 0`), `MNV2SmoothAtB` (all 35 relu6 sites off both kinks
at the real activations); for the smoothed loss every example's target summing to one and
`0 < nCls`.
-/

open Proofs Proofs.StableHLO

namespace Proofs.MobileNetV2TieB

open Proofs.BackLinks (bnInB bnInB_eq_bnBackB relu6MaskB cInB dInB gapInB reassocB rowB unrowB)
open Proofs.GradNodeB (hasGradAt_bnBatchLA hasGradAt_relu6 hasGradAt_conv hasGradAt_depthwise)
open scoped BigOperators

/-- `dStridedXlaInB` — the emitted XLA-`SAME` strided depthwise input-cotangent — is the batched
    strided depthwise VJP's backward, at any saved input. -/
theorem dStridedXlaInB_eq_batchMapBackward {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (dy : Vec (N * (c * h * w))) :
    dStridedXlaInB N W b dy
      = (batchMapHasVJP (depthwiseStride2FlatXla W b) (depthwiseStride2FlatXlaHasVJP W b)
          (depthwiseStride2FlatXla_differentiable W b)).backward x dy :=
  depthwiseStridedXlaBackBatched_faithful "" W b x (.operand "" dy)

/-- Back through a batched XLA-`SAME` strided depthwise: `dStridedXlaInB` (the stride-2 body's
    stage, beside `ParamGradNodes`' symmetric `hasGradAt_depthwiseStrided`). -/
theorem hasGradAt_depthwiseStridedXla {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) {G : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hG : HasGradAt G (batchMap N (depthwiseStride2FlatXla W b) x) dy) :
    HasGradAt (fun y => G (batchMap N (depthwiseStride2FlatXla W b) y)) x
      (dStridedXlaInB N W b dy) :=
  (hG.comp ((batchMap_differentiable _ (depthwiseStride2FlatXla_differentiable W b)) _)
    ((batchMapHasVJP _ (depthwiseStride2FlatXlaHasVJP W b)
      (depthwiseStride2FlatXla_differentiable W b)).toHasVJPAt _)).of_eq
    (dStridedXlaInB_eq_batchMapBackward (h := h) (w := w) W b _ _).symm

-- ════════════════════════════════════════════════════════════════
-- § The stem — XLA-`SAME` strided conv, batch BN, relu6
-- ════════════════════════════════════════════════════════════════

section Stem
variable {N h w ic oc : Nat}

/-- **Stem, every parameter node a loss derivative** — the four nodes `mnv2StemTiedB` ties, `Φ` the
    loss as a function of the stem's `(W, b, γ, β)`. -/
def mnv2StemLossTiedB (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic 3 3) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (bf16 : Bool) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (Φ : Kernel4 oc ic 3 3 → Vec oc → Vec oc → Vec oc → Vec 1) (dy : Vec (N * (oc * h * w))) :
    Prop :=
  let sc := batchMap N (flatConvStride2Xla Ws bs) x
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bs γs βs) (Kernel4.flatten Ws)
        (den (SHlo.convStridedXlaWeightGradBAt bf16 id xN bs x Ws
          (.operand cotN (mnv2StemCotC N h w Ws bs εs γs βs x dy)))))
  ∧ (HasGradAt (fun θ => Φ Ws θ γs βs) bs
        (den (SHlo.convStridedXlaBiasGradB (h := h) (w := w) Ws x bs
          (.operand cotN (mnv2StemCotC N h w Ws bs εs γs βs x dy)))))
  ∧ (HasGradAt (fun θ => Φ Ws bs θ βs) γs
        (den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc h w sc)
          (.operand cotN (reassocB N oc h w (mnv2StemCotN N h w Ws bs εs γs βs x dy))))))
  ∧ (HasGradAt (fun θ => Φ Ws bs γs θ) βs
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (mnv2StemCotN N h w Ws bs εs γs βs x dy))))))

theorem mnv2_stem_lossTiedB (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic 3 3) (bs : Vec oc)
    (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) (bf16 : Bool) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : MNV2StemSmoothAtB N h w Ws bs εs γs βs x)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (mnv2StemB N h w Ws bs εs γs βs x) dy)
    {Φ : Kernel4 oc ic 3 3 → Vec oc → Vec oc → Vec oc → Vec 1}
    (hΦ : ∀ W b γ β, Φ W b γ β = Gn (mnv2StemB N h w W b εs γ β x)) :
    mnv2StemLossTiedB xN cotN vN epsStr Ws bs εs γs βs bf16 x Φ dy := by
  rw [show Φ = fun W b γ β => Gn (mnv2StemB N h w W b εs γ β x) from
    funext fun W => funext fun b => funext fun γ => funext fun β => hΦ W b γ β]
  have hN := hasGradAt_relu6 _ hs hGn
  have hC := hasGradAt_bnBatchLA εs hεs γs βs _ hN
  exact ⟨GradNodeB.convStridedXlaWAt_hasGradAt bf16 xN cotN bs x Ws hC,
    GradNodeB.convStridedXlaB_hasGradAt cotN Ws x bs hC,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εs γs βs _ hN,
    GradNodeB.bnBeta_hasGradAt cotN εs γs βs _ hN⟩

end Stem

-- ════════════════════════════════════════════════════════════════
-- § The `t = 1` block (b1) — depthwise-BN-relu6, then the linear project
-- ════════════════════════════════════════════════════════════════

section NoExp
variable {N h w ic oc : Nat}

/-- **`t = 1` block, every parameter node a loss derivative** — the eight nodes `mnv2NoExpTiedB`
    ties. The project BN's γ/β read `dyOut` itself: nothing follows the linear bottleneck. -/
def mnv2NoExpLossTiedB (xN cotN vN epsStr : String) (p : IVWNoExp ic oc) (bf16 : Bool)
    (v : Vec (N * (ic * h * w))) (Φ : IVWNoExp ic oc → Vec 1) (dy : Vec (N * (oc * h * w))) :
    Prop :=
  let dc := batchMap N (depthwiseFlat p.dW p.db) v
  let dr := dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ v
  let pc := batchMap N (flatConv p.pW p.pb) dr
  (HasGradAt (fun θ => Φ { p with dW := Tensor3.unflatten θ }) (Tensor3.flatten p.dW)
        (den (SHlo.depthwiseWeightGradBAt bf16 id xN p.db v p.dW
          (.operand cotN (mnv2NoExpCotDc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with db := θ }) p.db
        (den (SHlo.depthwiseBiasGradB p.dW v p.db (.operand cotN (mnv2NoExpCotDc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with dγ := θ }) p.dγ
        (den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N ic h w dc)
          (.operand cotN (reassocB N ic h w (mnv2NoExpCotDn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with dβ := θ }) p.dβ
        (den (SHlo.bnBetaGradB (N := N) (oc := ic) (h := h) (w := w)
          (.operand cotN (reassocB N ic h w (mnv2NoExpCotDn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradBAt bf16 id xN p.pb dr p.pW
          (.operand cotN (mnv2NoExpCotPc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pb := θ }) p.pb
        (den (SHlo.convBiasGradB (h := h) (w := w) p.pW dr p.pb
          (.operand cotN (mnv2NoExpCotPc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pγ := θ }) p.pγ
        (den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pβ := θ }) p.pβ
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dy)))))

theorem mnv2_noexp_lossTiedB (xN cotN vN epsStr : String) (p : IVWNoExp ic oc) (bf16 : Bool)
    (hq : IVNoExpPos p) (v : Vec (N * (ic * h * w))) (hs : IVNoExpSmoothAtB N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (mnv2NoExpB N h w p v) dy) {Φ : IVWNoExp ic oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (mnv2NoExpB N h w p' v)) :
    mnv2NoExpLossTiedB xN cotN vN epsStr p bf16 v Φ dy := by
  rw [show Φ = fun p' => Gn (mnv2NoExpB N h w p' v) from funext hΦ]
  have hPn : HasGradAt Gn (bnBatchLA N oc h w p.pε p.pγ p.pβ (batchMap N (flatConv p.pW p.pb)
      (dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ v))) dy := hGn
  have hPc := hasGradAt_bnBatchLA p.pε hq.hp p.pγ p.pβ _ hPn
  have hDn := hasGradAt_relu6 _ hs.hd (hasGradAt_conv p.pW p.pb _ hPc)
  have hDc := hasGradAt_bnBatchLA p.dε hq.hd p.dγ p.dβ _ hDn
  exact ⟨GradNodeB.depthwiseWAt_hasGradAt bf16 xN cotN p.db v p.dW hDc,
    GradNodeB.depthwiseB_hasGradAt cotN p.dW v p.db hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.convWAt_hasGradAt bf16 xN cotN p.pb _ p.pW hPc,
    GradNodeB.convB_hasGradAt cotN p.pW _ p.pb hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.pε p.pγ p.pβ _ hPn,
    GradNodeB.bnBeta_hasGradAt cotN p.pε p.pγ p.pβ _ hPn⟩

end NoExp

-- ════════════════════════════════════════════════════════════════
-- § The stride-1 body — expand, depthwise, project; shared by the skip blocks and the widenings
--   `Gb` is the loss read at the BODY output. For a widening that is the block output; for a skip
--   block it is `u ↦ Gn (u + v)`, whose gradient there is still `dyOut` (`mnv2_resid_lossTiedB`).
-- ════════════════════════════════════════════════════════════════

section Body
variable {N h w ic mid oc : Nat}

/-- **Stride-1 body, every parameter node a loss derivative** — the twelve nodes
    `mnv2Stride1TiedB` ties, `Φ` the loss at the body output as a function of the weight record. -/
def mnv2Stride1LossTiedB (xN cotN vN epsStr : String) (p : IVW ic mid oc) (bf16 : Bool)
    (v : Vec (N * (ic * h * w))) (Φ : IVW ic mid oc → Vec 1) (dy : Vec (N * (oc * h * w))) :
    Prop :=
  let ec := batchMap N (flatConv p.eW p.eb) v
  let er := mnv2XE N h w p v
  let dc := batchMap N (depthwiseFlat p.dW p.db) er
  let dr := dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ er
  let pc := batchMap N (flatConv p.pW p.pb) dr
  (HasGradAt (fun θ => Φ { p with eW := Kernel4.unflatten θ }) (Kernel4.flatten p.eW)
        (den (SHlo.convWeightGradBAt bf16 id xN p.eb v p.eW (.operand cotN (mnv2CotEc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with eb := θ }) p.eb
        (den (SHlo.convBiasGradB (h := h) (w := w) p.eW v p.eb
          (.operand cotN (mnv2CotEc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with eγ := θ }) p.eγ
        (den (SHlo.bnGammaGradB vN epsStr p.eε (reassocB N mid h w ec)
          (.operand cotN (reassocB N mid h w (mnv2CotEn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with eβ := θ }) p.eβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (mnv2CotEn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with dW := Tensor3.unflatten θ }) (Tensor3.flatten p.dW)
        (den (SHlo.depthwiseWeightGradBAt bf16 id xN p.db er p.dW
          (.operand cotN (mnv2CotDc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with db := θ }) p.db
        (den (SHlo.depthwiseBiasGradB p.dW er p.db (.operand cotN (mnv2CotDc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with dγ := θ }) p.dγ
        (den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N mid h w dc)
          (.operand cotN (reassocB N mid h w (mnv2CotDn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with dβ := θ }) p.dβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (mnv2CotDn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradBAt bf16 id xN p.pb dr p.pW (.operand cotN (mnv2CotPc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pb := θ }) p.pb
        (den (SHlo.convBiasGradB (h := h) (w := w) p.pW dr p.pb
          (.operand cotN (mnv2CotPc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pγ := θ }) p.pγ
        (den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pβ := θ }) p.pβ
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dy)))))

/-- The stride-1 body bundle, from the loss `Gb` at the body output. A widening block (`b11`,
    `b17`) is this at `Gb := Gn`. -/
theorem mnv2_stride1_lossTiedB (xN cotN vN epsStr : String) (p : IVW ic mid oc) (bf16 : Bool)
    (hq : IVPos p)
    (v : Vec (N * (ic * h * w))) (hs : IVSmoothAtB N h w p v)
    {Gb : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGb : HasGradAt Gb (mnv2ExpOnlyB N h w p v) dy) {Φ : IVW ic mid oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gb (mnv2ExpOnlyB N h w p' v)) :
    mnv2Stride1LossTiedB xN cotN vN epsStr p bf16 v Φ dy := by
  rw [show Φ = fun p' => Gb (mnv2ExpOnlyB N h w p' v) from funext hΦ]
  have hPn : HasGradAt Gb (bnBatchLA N oc h w p.pε p.pγ p.pβ (batchMap N (flatConv p.pW p.pb)
      (dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ (mnv2XE N h w p v)))) dy := hGb
  have hPc := hasGradAt_bnBatchLA p.pε hq.hp p.pγ p.pβ _ hPn
  have hDn := hasGradAt_relu6 _ hs.hd (hasGradAt_conv p.pW p.pb _ hPc)
  have hDc := hasGradAt_bnBatchLA p.dε hq.hd p.dγ p.dβ _ hDn
  have hEn := hasGradAt_relu6 _ hs.he (hasGradAt_depthwise p.dW p.db _ hDc)
  have hEc := hasGradAt_bnBatchLA p.eε hq.he p.eγ p.eβ _ hEn
  exact ⟨GradNodeB.convWAt_hasGradAt bf16 xN cotN p.eb v p.eW hEc,
    GradNodeB.convB_hasGradAt cotN p.eW v p.eb hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.depthwiseWAt_hasGradAt bf16 xN cotN p.db _ p.dW hDc,
    GradNodeB.depthwiseB_hasGradAt cotN p.dW _ p.db hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.convWAt_hasGradAt bf16 xN cotN p.pb _ p.pW hPc,
    GradNodeB.convB_hasGradAt cotN p.pW _ p.pb hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.pε p.pγ p.pβ _ hPn,
    GradNodeB.bnBeta_hasGradAt cotN p.pε p.pγ p.pβ _ hPn⟩

/-- **A skip block's twelve nodes.** The identity skip is a constant once a body parameter varies,
    so the loss at the body output has gradient `dyOut` there and the body bundle applies. -/
theorem mnv2_resid_lossTiedB {c : Nat} (xN cotN vN epsStr : String) (p : IVW c mid c) (bf16 : Bool)
    (hq : IVPos p) (v : Vec (N * (c * h * w))) (hs : IVSmoothAtB N h w p v)
    {Gn : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hGn : HasGradAt Gn (mnv2ResidB N h w p v) dy) {Φ : IVW c mid c → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (mnv2ResidB N h w p' v)) :
    mnv2Stride1LossTiedB xN cotN vN epsStr p bf16 v Φ dy :=
  mnv2_stride1_lossTiedB xN cotN vN epsStr p bf16 hq v hs
    (GradNodeB.hasGradAt_addConst (mnv2ExpOnlyB N h w p v) v hGn) (fun p' => hΦ p')

end Body

-- ════════════════════════════════════════════════════════════════
-- § The stride-2 body (b2, b4, b7, b14) — expand at `2h × 2w`, the XLA-`SAME` strided depthwise
-- ════════════════════════════════════════════════════════════════

section SBody
variable {N h w ic mid oc : Nat}

/-- **Stride-2 block, every parameter node a loss derivative** — the twelve nodes
    `mnv2Stride2TiedB` ties; the depthwise nodes are the XLA-`SAME` strided ones. -/
def mnv2Stride2LossTiedB (xN cotN vN epsStr : String) (p : IVW ic mid oc) (bf16 : Bool)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (Φ : IVW ic mid oc → Vec 1)
    (dy : Vec (N * (oc * h * w))) : Prop :=
  let ec := batchMap N (flatConv p.eW p.eb) v
  let er := mnv2XES N h w p v
  let dc := batchMap N (depthwiseStride2FlatXla p.dW p.db) er
  let dr := dwbrBstrided N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ er
  let pc := batchMap N (flatConv p.pW p.pb) dr
  (HasGradAt (fun θ => Φ { p with eW := Kernel4.unflatten θ }) (Kernel4.flatten p.eW)
        (den (SHlo.convWeightGradBAt bf16 id xN p.eb v p.eW (.operand cotN (mnv2SCotEc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with eb := θ }) p.eb
        (den (SHlo.convBiasGradB (h := 2 * h) (w := 2 * w) p.eW v p.eb
          (.operand cotN (mnv2SCotEc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with eγ := θ }) p.eγ
        (den (SHlo.bnGammaGradB vN epsStr p.eε (reassocB N mid (2 * h) (2 * w) ec)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) (mnv2SCotEn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with eβ := θ }) p.eβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := 2 * h) (w := 2 * w)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) (mnv2SCotEn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with dW := Tensor3.unflatten θ }) (Tensor3.flatten p.dW)
        (den (SHlo.depthwiseStridedXlaWeightGradBAt bf16 id xN p.db er p.dW
          (.operand cotN (mnv2SCotDc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with db := θ }) p.db
        (den (SHlo.depthwiseStridedXlaBiasGradB (h := h) (w := w) p.dW er p.db
          (.operand cotN (mnv2SCotDc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with dγ := θ }) p.dγ
        (den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N mid h w dc)
          (.operand cotN (reassocB N mid h w (mnv2SCotDn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with dβ := θ }) p.dβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (mnv2SCotDn N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradBAt bf16 id xN p.pb dr p.pW (.operand cotN (mnv2SCotPc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pb := θ }) p.pb
        (den (SHlo.convBiasGradB (h := h) (w := w) p.pW dr p.pb
          (.operand cotN (mnv2SCotPc N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pγ := θ }) p.pγ
        (den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with pβ := θ }) p.pβ
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dy)))))

theorem mnv2_stride2_lossTiedB (xN cotN vN epsStr : String) (p : IVW ic mid oc) (bf16 : Bool)
    (hq : IVPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : IVStridedSmoothAtB N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (mnv2StridedB N h w p v) dy) {Φ : IVW ic mid oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (mnv2StridedB N h w p' v)) :
    mnv2Stride2LossTiedB xN cotN vN epsStr p bf16 v Φ dy := by
  rw [show Φ = fun p' => Gn (mnv2StridedB N h w p' v) from funext hΦ]
  have hPn : HasGradAt Gn (bnBatchLA N oc h w p.pε p.pγ p.pβ (batchMap N (flatConv p.pW p.pb)
      (dwbrBstrided N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ (mnv2XES N h w p v)))) dy := hGn
  have hPc := hasGradAt_bnBatchLA p.pε hq.hp p.pγ p.pβ _ hPn
  have hDn := hasGradAt_relu6 _ hs.hd (hasGradAt_conv p.pW p.pb _ hPc)
  have hDc := hasGradAt_bnBatchLA p.dε hq.hd p.dγ p.dβ _ hDn
  have hEn := hasGradAt_relu6 _ hs.he (hasGradAt_depthwiseStridedXla p.dW p.db _ hDc)
  have hEc := hasGradAt_bnBatchLA p.eε hq.he p.eγ p.eβ _ hEn
  exact ⟨GradNodeB.convWAt_hasGradAt bf16 (h := 2 * h) (w := 2 * w) xN cotN p.eb v p.eW hEc,
    GradNodeB.convB_hasGradAt (h := 2 * h) (w := 2 * w) cotN p.eW v p.eb hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.depthwiseStridedXlaWAt_hasGradAt bf16 xN cotN p.db _ p.dW hDc,
    GradNodeB.depthwiseStridedXlaB_hasGradAt cotN p.dW _ p.db hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.convWAt_hasGradAt bf16 xN cotN p.pb _ p.pW hPc,
    GradNodeB.convB_hasGradAt cotN p.pW _ p.pb hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.pε p.pγ p.pβ _ hPn,
    GradNodeB.bnBeta_hasGradAt cotN p.pε p.pγ p.pβ _ hPn⟩

end SBody

-- ════════════════════════════════════════════════════════════════
-- § The head — 1×1 conv-BN-relu6, GAP, dense
-- ════════════════════════════════════════════════════════════════

section Head
variable {N h w ic oc nCls : Nat}

/-- **Head, every parameter node a loss derivative** — the six nodes `mnv2HeadTiedB` ties, `Φ`
    the loss as a function of `(hW, hb, hγ, hβ, Wd, bd)`. -/
def mnv2HeadLossTiedB (xN cotN vN epsStr : String) (Wh : Kernel4 oc ic 1 1) (bh : Vec oc)
    (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool)
    (v : Vec (N * (ic * h * w)))
    (Φ : Kernel4 oc ic 1 1 → Vec oc → Vec oc → Vec oc → Mat oc nCls → Vec nCls → Vec 1)
    (g : Vec (N * nCls)) : Prop :=
  let hc := batchMap N (flatConv Wh bh) v
  let a := batchMap N (globalAvgPoolFlat oc h w) (cbrB N (h := h) (w := w) Wh bh εh γh βh v)
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bh γh βh Wd bd) (Kernel4.flatten Wh)
        (den (SHlo.convWeightGradBAt bf16 id xN bh v Wh
          (.operand cotN (mnv2HeadCotHc N h w Wh bh εh γh βh Wd v g)))))
  ∧ (HasGradAt (fun θ => Φ Wh θ γh βh Wd bd) bh
        (den (SHlo.convBiasGradB (h := h) (w := w) Wh v bh
          (.operand cotN (mnv2HeadCotHc N h w Wh bh εh γh βh Wd v g)))))
  ∧ (HasGradAt (fun θ => Φ Wh bh θ βh Wd bd) γh
        (den (SHlo.bnGammaGradB vN epsStr εh (reassocB N oc h w hc)
          (.operand cotN (reassocB N oc h w (mnv2HeadCotHn N h w Wh bh εh γh βh Wd v g))))))
  ∧ (HasGradAt (fun θ => Φ Wh bh γh θ Wd bd) βh
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (mnv2HeadCotHn N h w Wh bh εh γh βh Wd v g))))))
  ∧ (HasGradAt (fun θ => Φ Wh bh γh βh (Mat.unflatten θ) bd) (Mat.flatten Wd)
        (den (SHlo.denseWeightGradB (c := nCls) xN a (.operand cotN g))))
  ∧ (HasGradAt (fun θ => Φ Wh bh γh βh Wd θ) bd
        (den (SHlo.denseBiasGradB (N := N) (.operand cotN g))))

theorem mnv2_head_lossTiedB (xN cotN vN epsStr : String) (Wh : Kernel4 oc ic 1 1) (bh : Vec oc)
    (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (bf16 : Bool)
    (v : Vec (N * (ic * h * w))) (hs : MNV2HeadSmoothAtB N h w Wh bh εh γh βh v)
    {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (mnv2HeadB N h w Wh bh εh γh βh Wd bd v) g)
    {Φ : Kernel4 oc ic 1 1 → Vec oc → Vec oc → Vec oc → Mat oc nCls → Vec nCls → Vec 1}
    (hΦ : ∀ W b γ β Wd' bd', Φ W b γ β Wd' bd' = L (mnv2HeadB N h w W b εh γ β Wd' bd' v)) :
    mnv2HeadLossTiedB xN cotN vN epsStr Wh bh εh γh βh Wd bd bf16 v Φ g := by
  rw [show Φ = fun W b γ β Wd' bd' => L (mnv2HeadB N h w W b εh γ β Wd' bd' v) from
    funext fun W => funext fun b => funext fun γ => funext fun β => funext fun Wd' =>
      funext fun bd' => hΦ W b γ β Wd' bd']
  -- back through the classifier and the GAP, then the head's relu6 and BN
  have hA := HasGradAt.comp (f := batchMap N (dense Wd bd))
    (x := batchMap N (globalAvgPoolFlat oc h w) (cbrB N (h := h) (w := w) Wh bh εh γh βh v))
    hL ((batchMap_differentiable _ (dense_differentiable Wd bd)) _)
    ((batchMapHasVJP _ (denseHasVJP Wd bd) (dense_differentiable Wd bd)).toHasVJPAt _)
  have hR := HasGradAt.comp (f := batchMap N (globalAvgPoolFlat oc h w))
    (x := cbrB N (h := h) (w := w) Wh bh εh γh βh v) hA
    ((batchMap_differentiable _ (globalAvgPoolFlat_differentiable oc h w)) _)
    ((batchMapHasVJP _ (globalAvgPoolFlatHasVJP oc h w)
      (globalAvgPoolFlat_differentiable oc h w)).toHasVJPAt _)
  have hN := hasGradAt_relu6 _ hs hR
  have hC := hasGradAt_bnBatchLA εh hεh γh βh _ hN
  exact ⟨GradNodeB.convWAt_hasGradAt bf16 xN cotN bh v Wh hC,
    GradNodeB.convB_hasGradAt cotN Wh v bh hC,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εh γh βh _ hN,
    GradNodeB.bnBeta_hasGradAt cotN εh γh βh _ hN,
    GradNodeB.denseW_hasGradAt xN cotN _ Wd bd hL,
    GradNodeB.denseB_hasGradAt cotN Wd (fun _ => 0) _ bd hL⟩

end Head

-- ════════════════════════════════════════════════════════════════
-- § The whole net: the loss after each block, and the net with one block's weights varied
-- ════════════════════════════════════════════════════════════════

/-- Pull the loss gradient back through the `t = 1` block (`mnv2NoExpCotIn_eq_vjp`). -/
theorem mnv2NoExpB_hasGradAt_comp {N h w ic oc : Nat} (p : IVWNoExp ic oc) (hq : IVNoExpPos p)
    (v : Vec (N * (ic * h * w))) (hs : IVNoExpSmoothAtB N h w p v)
    {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (mnv2NoExpB N h w p v) dy) :
    HasGradAt (fun y => G (mnv2NoExpB N h w p y)) v (mnv2NoExpCotIn N h w p v dy) :=
  (HasGradAt.comp (f := mnv2NoExpB N h w p) (x := v) hG
    (((StableHLO.dwbrLayer N (h := h) (w := w) p.dW p.db p.dε hq.hd p.dγ p.dβ).comp
      (StableHLO.projLayer N p.pW p.pb p.pε hq.hp p.pγ p.pβ)).diff v ⟨hs.hd, trivial⟩)
    (mnv2NoExpBHasVJPAt N h w p hq v hs)).of_eq (mnv2NoExpCotIn_eq_vjp N h w p hq v dy hs).symm

/-- …through a widening block (`mnv2ExpOnlyCotIn_eq_vjp`). -/
theorem mnv2ExpOnlyB_hasGradAt_comp {N h w ic mid oc : Nat} (p : IVW ic mid oc) (hq : IVPos p)
    (v : Vec (N * (ic * h * w))) (hs : IVSmoothAtB N h w p v)
    {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (mnv2ExpOnlyB N h w p v) dy) :
    HasGradAt (fun y => G (mnv2ExpOnlyB N h w p y)) v (mnv2CotInBody N h w p v dy) :=
  (HasGradAt.comp (f := mnv2ExpOnlyB N h w p) (x := v) hG
    (StableHLO.mnv2BodyB_differentiableAt N p.eW p.eb p.eε hq.he p.eγ p.eβ
      p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd)
    (mnv2ExpOnlyBHasVJPAt N h w p hq v hs)).of_eq (mnv2ExpOnlyCotIn_eq_vjp N h w p hq v dy hs).symm

/-- …through a skip block (`mnv2ResidCotIn_eq_vjp`). -/
theorem mnv2ResidB_hasGradAt_comp {N h w c mid : Nat} (p : IVW c mid c) (hq : IVPos p)
    (v : Vec (N * (c * h * w))) (hs : IVSmoothAtB N h w p v)
    {G : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hG : HasGradAt G (mnv2ResidB N h w p v) dy) :
    HasGradAt (fun y => G (mnv2ResidB N h w p y)) v (mnv2ResidCotIn N h w p v dy) :=
  (HasGradAt.comp (f := mnv2ResidB N h w p) (x := v) hG
    (residual_differentiableAt (StableHLO.mnv2BodyB_differentiableAt N p.eW p.eb p.eε hq.he p.eγ p.eβ
      p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd))
    (mnv2ResidBHasVJPAt N h w p hq v hs)).of_eq (mnv2ResidCotIn_eq_vjp N h w p hq v dy hs).symm

/-- …and through a stride-2 block (`mnv2StridedCotIn_eq_vjp`). -/
theorem mnv2StridedB_hasGradAt_comp {N h w ic mid oc : Nat} (p : IVW ic mid oc) (hq : IVPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : IVStridedSmoothAtB N h w p v)
    {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (mnv2StridedB N h w p v) dy) :
    HasGradAt (fun y => G (mnv2StridedB N h w p y)) v (mnv2StridedCotIn N h w p v dy) :=
  (HasGradAt.comp (f := mnv2StridedB N h w p) (x := v) hG
    (StableHLO.mnv2DownBodyB_differentiableAt N p.eW p.eb p.eε hq.he p.eγ p.eβ
      p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd)
    (mnv2StridedBHasVJPAt N h w p hq v hs)).of_eq (mnv2StridedCotIn_eq_vjp N h w p hq v dy hs).symm

/-- The net after block `b17` — the head. -/
noncomputable def mnv2SufB17 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (320 * 7 * 7)) → Vec (N * nCls) :=
  mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb

/-- The net after block `b16`: block `b17`, then the rest. -/
noncomputable def mnv2SufB16 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (160 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv2SufB17 N w (mnv2ExpOnlyB N 7 7 w.b17 y)

/-- The net after block `b15`: block `b16`, then the rest. -/
noncomputable def mnv2SufB15 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (160 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv2SufB16 N w (mnv2ResidB N 7 7 w.b16 y)

/-- The net after block `b14`: block `b15`, then the rest. -/
noncomputable def mnv2SufB14 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (160 * 7 * 7)) → Vec (N * nCls) :=
  fun y => mnv2SufB15 N w (mnv2ResidB N 7 7 w.b15 y)

/-- The net after block `b13`: block `b14`, then the rest. -/
noncomputable def mnv2SufB13 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (96 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB14 N w (mnv2StridedB N 7 7 w.b14 y)

/-- The net after block `b12`: block `b13`, then the rest. -/
noncomputable def mnv2SufB12 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (96 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB13 N w (mnv2ResidB N 14 14 w.b13 y)

/-- The net after block `b11`: block `b12`, then the rest. -/
noncomputable def mnv2SufB11 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (96 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB12 N w (mnv2ResidB N 14 14 w.b12 y)

/-- The net after block `b10`: block `b11`, then the rest. -/
noncomputable def mnv2SufB10 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (64 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB11 N w (mnv2ExpOnlyB N 14 14 w.b11 y)

/-- The net after block `b9`: block `b10`, then the rest. -/
noncomputable def mnv2SufB9 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (64 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB10 N w (mnv2ResidB N 14 14 w.b10 y)

/-- The net after block `b8`: block `b9`, then the rest. -/
noncomputable def mnv2SufB8 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (64 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB9 N w (mnv2ResidB N 14 14 w.b9 y)

/-- The net after block `b7`: block `b8`, then the rest. -/
noncomputable def mnv2SufB7 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (64 * 14 * 14)) → Vec (N * nCls) :=
  fun y => mnv2SufB8 N w (mnv2ResidB N 14 14 w.b8 y)

/-- The net after block `b6`: block `b7`, then the rest. -/
noncomputable def mnv2SufB6 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (32 * 28 * 28)) → Vec (N * nCls) :=
  fun y => mnv2SufB7 N w (mnv2StridedB N 14 14 w.b7 y)

/-- The net after block `b5`: block `b6`, then the rest. -/
noncomputable def mnv2SufB5 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (32 * 28 * 28)) → Vec (N * nCls) :=
  fun y => mnv2SufB6 N w (mnv2ResidB N 28 28 w.b6 y)

/-- The net after block `b4`: block `b5`, then the rest. -/
noncomputable def mnv2SufB4 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (32 * 28 * 28)) → Vec (N * nCls) :=
  fun y => mnv2SufB5 N w (mnv2ResidB N 28 28 w.b5 y)

/-- The net after block `b3`: block `b4`, then the rest. -/
noncomputable def mnv2SufB3 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (24 * 56 * 56)) → Vec (N * nCls) :=
  fun y => mnv2SufB4 N w (mnv2StridedB N 28 28 w.b4 y)

/-- The net after block `b2`: block `b3`, then the rest. -/
noncomputable def mnv2SufB2 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (24 * 56 * 56)) → Vec (N * nCls) :=
  fun y => mnv2SufB3 N w (mnv2ResidB N 56 56 w.b3 y)

/-- The net after block `b1`: block `b2`, then the rest. -/
noncomputable def mnv2SufB1 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (16 * 112 * 112)) → Vec (N * nCls) :=
  fun y => mnv2SufB2 N w (mnv2StridedB N 56 56 w.b2 y)

/-- The net after the stem: block `b1`, then the rest. -/
noncomputable def mnv2SufStem (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (32 * 112 * 112)) → Vec (N * nCls) :=
  fun y => mnv2SufB1 N w (mnv2NoExpB N 112 112 w.b1 y)

/-- **The net with the stem's parameters varied** is the suffix after the stem at the varied stem. -/
theorem mnv2_factor_stem (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (W : Kernel4 32 3 3 3) (b γ β : Vec 32) :
    mobilenetv2ForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x
      = mnv2SufStem N w (mnv2StemB N 112 112 W b w.sε γ β x) := rfl

/-- **The net with block `b1`'s weights varied** is the suffix after `b1` at the varied block. -/
theorem mnv2_factor_b1 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVWNoExp 32 16) :
    mobilenetv2ForwardBFull N { w with b1 := p } x
      = mnv2SufB1 N w (mnv2NoExpB N 112 112 p (mnv2PreB0 N w x)) := by
  rw [mnv2PreB0_apply]; rfl

/-- **The net with block `b2`'s weights varied** is the suffix after `b2` at the varied block. -/
theorem mnv2_factor_b2 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 16 96 24) :
    mobilenetv2ForwardBFull N { w with b2 := p } x
      = mnv2SufB2 N w (mnv2StridedB N 56 56 p (mnv2PreB1 N w x)) := by
  rw [mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b3`'s weights varied** is the suffix after `b3` at the varied block. -/
theorem mnv2_factor_b3 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 24 144 24) :
    mobilenetv2ForwardBFull N { w with b3 := p } x
      = mnv2SufB3 N w (mnv2ResidB N 56 56 p (mnv2PreB2 N w x)) := by
  rw [mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b4`'s weights varied** is the suffix after `b4` at the varied block. -/
theorem mnv2_factor_b4 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 24 144 32) :
    mobilenetv2ForwardBFull N { w with b4 := p } x
      = mnv2SufB4 N w (mnv2StridedB N 28 28 p (mnv2PreB3 N w x)) := by
  rw [mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b5`'s weights varied** is the suffix after `b5` at the varied block. -/
theorem mnv2_factor_b5 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 32 192 32) :
    mobilenetv2ForwardBFull N { w with b5 := p } x
      = mnv2SufB5 N w (mnv2ResidB N 28 28 p (mnv2PreB4 N w x)) := by
  rw [mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b6`'s weights varied** is the suffix after `b6` at the varied block. -/
theorem mnv2_factor_b6 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 32 192 32) :
    mobilenetv2ForwardBFull N { w with b6 := p } x
      = mnv2SufB6 N w (mnv2ResidB N 28 28 p (mnv2PreB5 N w x)) := by
  rw [mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b7`'s weights varied** is the suffix after `b7` at the varied block. -/
theorem mnv2_factor_b7 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 32 192 64) :
    mobilenetv2ForwardBFull N { w with b7 := p } x
      = mnv2SufB7 N w (mnv2StridedB N 14 14 p (mnv2PreB6 N w x)) := by
  rw [mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b8`'s weights varied** is the suffix after `b8` at the varied block. -/
theorem mnv2_factor_b8 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 64 384 64) :
    mobilenetv2ForwardBFull N { w with b8 := p } x
      = mnv2SufB8 N w (mnv2ResidB N 14 14 p (mnv2PreB7 N w x)) := by
  rw [mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b9`'s weights varied** is the suffix after `b9` at the varied block. -/
theorem mnv2_factor_b9 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 64 384 64) :
    mobilenetv2ForwardBFull N { w with b9 := p } x
      = mnv2SufB9 N w (mnv2ResidB N 14 14 p (mnv2PreB8 N w x)) := by
  rw [mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b10`'s weights varied** is the suffix after `b10` at the varied block. -/
theorem mnv2_factor_b10 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 64 384 64) :
    mobilenetv2ForwardBFull N { w with b10 := p } x
      = mnv2SufB10 N w (mnv2ResidB N 14 14 p (mnv2PreB9 N w x)) := by
  rw [mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b11`'s weights varied** is the suffix after `b11` at the varied block. -/
theorem mnv2_factor_b11 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 64 384 96) :
    mobilenetv2ForwardBFull N { w with b11 := p } x
      = mnv2SufB11 N w (mnv2ExpOnlyB N 14 14 p (mnv2PreB10 N w x)) := by
  rw [mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b12`'s weights varied** is the suffix after `b12` at the varied block. -/
theorem mnv2_factor_b12 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 96 576 96) :
    mobilenetv2ForwardBFull N { w with b12 := p } x
      = mnv2SufB12 N w (mnv2ResidB N 14 14 p (mnv2PreB11 N w x)) := by
  rw [mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b13`'s weights varied** is the suffix after `b13` at the varied block. -/
theorem mnv2_factor_b13 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 96 576 96) :
    mobilenetv2ForwardBFull N { w with b13 := p } x
      = mnv2SufB13 N w (mnv2ResidB N 14 14 p (mnv2PreB12 N w x)) := by
  rw [mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b14`'s weights varied** is the suffix after `b14` at the varied block. -/
theorem mnv2_factor_b14 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 96 576 160) :
    mobilenetv2ForwardBFull N { w with b14 := p } x
      = mnv2SufB14 N w (mnv2StridedB N 7 7 p (mnv2PreB13 N w x)) := by
  rw [mnv2PreB13_apply, mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b15`'s weights varied** is the suffix after `b15` at the varied block. -/
theorem mnv2_factor_b15 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 160 960 160) :
    mobilenetv2ForwardBFull N { w with b15 := p } x
      = mnv2SufB15 N w (mnv2ResidB N 7 7 p (mnv2PreB14 N w x)) := by
  rw [mnv2PreB14_apply, mnv2PreB13_apply, mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b16`'s weights varied** is the suffix after `b16` at the varied block. -/
theorem mnv2_factor_b16 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 160 960 160) :
    mobilenetv2ForwardBFull N { w with b16 := p } x
      = mnv2SufB16 N w (mnv2ResidB N 7 7 p (mnv2PreB15 N w x)) := by
  rw [mnv2PreB15_apply, mnv2PreB14_apply, mnv2PreB13_apply, mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with block `b17`'s weights varied** is the suffix after `b17` at the varied block. -/
theorem mnv2_factor_b17 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (p : IVW 160 960 320) :
    mobilenetv2ForwardBFull N { w with b17 := p } x
      = mnv2SufB17 N w (mnv2ExpOnlyB N 7 7 p (mnv2PreB16 N w x)) := by
  rw [mnv2PreB16_apply, mnv2PreB15_apply, mnv2PreB14_apply, mnv2PreB13_apply, mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **The net with the head varied** is the head at the varied parameters. -/
theorem mnv2_factor_head (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (W : Kernel4 1280 320 1 1) (b γ β : Vec 1280) (Wd : Mat 1280 nCls) (bd : Vec nCls) :
    mobilenetv2ForwardBFull N { w with hW := W, hb := b, hγ := γ, hβ := β, fcW := Wd, fcb := bd } x
      = mnv2HeadB N 7 7 W b w.hε γ β Wd bd (mnv2PreB17 N w x) := by
  rw [mnv2PreB17_apply, mnv2PreB16_apply, mnv2PreB15_apply, mnv2PreB14_apply, mnv2PreB13_apply, mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]; rfl

/-- **Every MobileNetV2 parameter gradient node is the derivative of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 210 slots
    `mnv2_net_tiedB` ties, each at the cotangent the emitted chain threads to it, stated against `L`
    of `mobilenetv2ForwardBFull` with that one parameter varied. -/
def MNV2NetLossTiedB (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String) (w : MNV2BWeights nCls)
    (bf16 : Bool)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (L : Vec (N * nCls) → Vec 1) (g : Vec (N * nCls)) : Prop :=
    let dy17 := mnv2HeadCotBlk N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW (mnv2PreB17 N w x) g
    let dy16 := mnv2CotInBody N 7 7 w.b17 (mnv2PreB16 N w x) dy17
    let dy15 := mnv2ResidCotIn N 7 7 w.b16 (mnv2PreB15 N w x) dy16
    let dy14 := mnv2ResidCotIn N 7 7 w.b15 (mnv2PreB14 N w x) dy15
    let dy13 := mnv2StridedCotIn N 7 7 w.b14 (mnv2PreB13 N w x) dy14
    let dy12 := mnv2ResidCotIn N 14 14 w.b13 (mnv2PreB12 N w x) dy13
    let dy11 := mnv2ResidCotIn N 14 14 w.b12 (mnv2PreB11 N w x) dy12
    let dy10 := mnv2CotInBody N 14 14 w.b11 (mnv2PreB10 N w x) dy11
    let dy9 := mnv2ResidCotIn N 14 14 w.b10 (mnv2PreB9 N w x) dy10
    let dy8 := mnv2ResidCotIn N 14 14 w.b9 (mnv2PreB8 N w x) dy9
    let dy7 := mnv2ResidCotIn N 14 14 w.b8 (mnv2PreB7 N w x) dy8
    let dy6 := mnv2StridedCotIn N 14 14 w.b7 (mnv2PreB6 N w x) dy7
    let dy5 := mnv2ResidCotIn N 28 28 w.b6 (mnv2PreB5 N w x) dy6
    let dy4 := mnv2ResidCotIn N 28 28 w.b5 (mnv2PreB4 N w x) dy5
    let dy3 := mnv2StridedCotIn N 28 28 w.b4 (mnv2PreB3 N w x) dy4
    let dy2 := mnv2ResidCotIn N 56 56 w.b3 (mnv2PreB2 N w x) dy3
    let dy1 := mnv2StridedCotIn N 56 56 w.b2 (mnv2PreB1 N w x) dy2
    let cotStem := mnv2NoExpCotIn N 112 112 w.b1 (mnv2PreB0 N w x) dy1
    mnv2StemLossTiedB (N := N) (h := 112) (w := 112) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ bf16 x
      (fun W b γ β => L (mobilenetv2ForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x))
      cotStem
  ∧ mnv2NoExpLossTiedB (N := N) (h := 112) (w := 112) xN cotN vN epsStr w.b1 bf16 (mnv2PreB0 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b1 := p } x)) dy1
  ∧ mnv2Stride2LossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.b2 bf16 (mnv2PreB1 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b2 := p } x)) dy2
  ∧ mnv2Stride1LossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.b3 bf16 (mnv2PreB2 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b3 := p } x)) dy3
  ∧ mnv2Stride2LossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b4 bf16 (mnv2PreB3 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b4 := p } x)) dy4
  ∧ mnv2Stride1LossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b5 bf16 (mnv2PreB4 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b5 := p } x)) dy5
  ∧ mnv2Stride1LossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b6 bf16 (mnv2PreB5 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b6 := p } x)) dy6
  ∧ mnv2Stride2LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b7 bf16 (mnv2PreB6 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b7 := p } x)) dy7
  ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b8 bf16 (mnv2PreB7 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b8 := p } x)) dy8
  ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b9 bf16 (mnv2PreB8 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b9 := p } x)) dy9
  ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b10 bf16 (mnv2PreB9 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b10 := p } x)) dy10
  ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b11 bf16 (mnv2PreB10 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b11 := p } x)) dy11
  ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b12 bf16 (mnv2PreB11 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b12 := p } x)) dy12
  ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b13 bf16 (mnv2PreB12 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b13 := p } x)) dy13
  ∧ mnv2Stride2LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b14 bf16 (mnv2PreB13 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b14 := p } x)) dy14
  ∧ mnv2Stride1LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b15 bf16 (mnv2PreB14 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b15 := p } x)) dy15
  ∧ mnv2Stride1LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b16 bf16 (mnv2PreB15 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b16 := p } x)) dy16
  ∧ mnv2Stride1LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b17 bf16 (mnv2PreB16 N w x)
      (fun p => L (mobilenetv2ForwardBFull N { w with b17 := p } x)) dy17
  ∧ mnv2HeadLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16
      (mnv2PreB17 N w x)
      (fun W b γ β Wd bd => L (mobilenetv2ForwardBFull N
        { w with hW := W, hb := b, hγ := γ, hβ := β, fcW := Wd, fcb := bd } x)) g

/-- **Every MobileNetV2 parameter gradient node is the derivative of the loss in that
    parameter.** For any loss `L` of the logits with gradient `g` at the net's output, each of the
    210 slots `mnv2_net_tiedB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net,
    `mobilenetv2ForwardBFull` with that one parameter varied (a stem field, a block's weight record
    `w.bk := p` with one slot changed, or a head field).

    Hypotheses: every BN `ε` positive (`MNV2PosB`) and all 35 relu6 sites off both kinks at the real
    activations (`MNV2SmoothAtB`). The loss enters only through `hL`;
    `mnv2_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship.
    One replica; `bf16` selects the conv and depthwise weight nodes' kind (the module's Scope). -/
theorem mnv2_net_lossGrad (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : MNV2BWeights nCls) (hq : MNV2PosB w) (bf16 : Bool)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (hx : MNV2SmoothAtB N w x)
    {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (mobilenetv2ForwardBFull N w x) g) :
    MNV2NetLossTiedB N xN cotN vN epsStr w bf16 x L g := by
  unfold MNV2NetLossTiedB
  intro dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotStem
  have hL' : HasGradAt L (mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb (mnv2PreB17 N w x)) g :=
    hL.congr_point (by rw [mobilenetv2ForwardBFull_eq_chain, Function.comp_apply])
  have h17 : HasGradAt (fun y => L (mnv2SufB17 N w y)) (mnv2PreB17 N w x) dy17 :=
    (HasGradAt.comp (f := mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb)
      (x := mnv2PreB17 N w x) hL'
      (((batchMap_differentiable _ (dense_differentiable w.fcW w.fcb)) _).comp _
        (((batchMap_differentiable _ (globalAvgPoolFlat_differentiable 1280 7 7)) _).comp _
          (StableHLO.cbrB_differentiableAt N w.hW w.hb w.hε hq.h w.hγ w.hβ _ hx.head)))
      (mnv2HeadBHasVJPAt N 7 7 w.hW w.hb w.hε hq.h w.hγ w.hβ w.fcW w.fcb _ hx.head)).of_eq
      (mnv2HeadCotBlk_eq_vjp N 7 7 w.hW w.hb w.hε hq.h w.hγ w.hβ w.fcW w.fcb _ g hx.head).symm
  have h16 : HasGradAt (fun y => L (mnv2SufB16 N w y)) (mnv2PreB16 N w x) dy16 :=
    mnv2ExpOnlyB_hasGradAt_comp w.b17 hq.b17 _ hx.b17 (h17.congr_point (mnv2PreB17_apply N w x))
  have h15 : HasGradAt (fun y => L (mnv2SufB15 N w y)) (mnv2PreB15 N w x) dy15 :=
    mnv2ResidB_hasGradAt_comp w.b16 hq.b16 _ hx.b16 (h16.congr_point (mnv2PreB16_apply N w x))
  have h14 : HasGradAt (fun y => L (mnv2SufB14 N w y)) (mnv2PreB14 N w x) dy14 :=
    mnv2ResidB_hasGradAt_comp w.b15 hq.b15 _ hx.b15 (h15.congr_point (mnv2PreB15_apply N w x))
  have h13 : HasGradAt (fun y => L (mnv2SufB13 N w y)) (mnv2PreB13 N w x) dy13 :=
    mnv2StridedB_hasGradAt_comp w.b14 hq.b14 _ hx.b14 (h14.congr_point (mnv2PreB14_apply N w x))
  have h12 : HasGradAt (fun y => L (mnv2SufB12 N w y)) (mnv2PreB12 N w x) dy12 :=
    mnv2ResidB_hasGradAt_comp w.b13 hq.b13 _ hx.b13 (h13.congr_point (mnv2PreB13_apply N w x))
  have h11 : HasGradAt (fun y => L (mnv2SufB11 N w y)) (mnv2PreB11 N w x) dy11 :=
    mnv2ResidB_hasGradAt_comp w.b12 hq.b12 _ hx.b12 (h12.congr_point (mnv2PreB12_apply N w x))
  have h10 : HasGradAt (fun y => L (mnv2SufB10 N w y)) (mnv2PreB10 N w x) dy10 :=
    mnv2ExpOnlyB_hasGradAt_comp w.b11 hq.b11 _ hx.b11 (h11.congr_point (mnv2PreB11_apply N w x))
  have h9 : HasGradAt (fun y => L (mnv2SufB9 N w y)) (mnv2PreB9 N w x) dy9 :=
    mnv2ResidB_hasGradAt_comp w.b10 hq.b10 _ hx.b10 (h10.congr_point (mnv2PreB10_apply N w x))
  have h8 : HasGradAt (fun y => L (mnv2SufB8 N w y)) (mnv2PreB8 N w x) dy8 :=
    mnv2ResidB_hasGradAt_comp w.b9 hq.b9 _ hx.b9 (h9.congr_point (mnv2PreB9_apply N w x))
  have h7 : HasGradAt (fun y => L (mnv2SufB7 N w y)) (mnv2PreB7 N w x) dy7 :=
    mnv2ResidB_hasGradAt_comp w.b8 hq.b8 _ hx.b8 (h8.congr_point (mnv2PreB8_apply N w x))
  have h6 : HasGradAt (fun y => L (mnv2SufB6 N w y)) (mnv2PreB6 N w x) dy6 :=
    mnv2StridedB_hasGradAt_comp w.b7 hq.b7 _ hx.b7 (h7.congr_point (mnv2PreB7_apply N w x))
  have h5 : HasGradAt (fun y => L (mnv2SufB5 N w y)) (mnv2PreB5 N w x) dy5 :=
    mnv2ResidB_hasGradAt_comp w.b6 hq.b6 _ hx.b6 (h6.congr_point (mnv2PreB6_apply N w x))
  have h4 : HasGradAt (fun y => L (mnv2SufB4 N w y)) (mnv2PreB4 N w x) dy4 :=
    mnv2ResidB_hasGradAt_comp w.b5 hq.b5 _ hx.b5 (h5.congr_point (mnv2PreB5_apply N w x))
  have h3 : HasGradAt (fun y => L (mnv2SufB3 N w y)) (mnv2PreB3 N w x) dy3 :=
    mnv2StridedB_hasGradAt_comp w.b4 hq.b4 _ hx.b4 (h4.congr_point (mnv2PreB4_apply N w x))
  have h2 : HasGradAt (fun y => L (mnv2SufB2 N w y)) (mnv2PreB2 N w x) dy2 :=
    mnv2ResidB_hasGradAt_comp w.b3 hq.b3 _ hx.b3 (h3.congr_point (mnv2PreB3_apply N w x))
  have h1 : HasGradAt (fun y => L (mnv2SufB1 N w y)) (mnv2PreB1 N w x) dy1 :=
    mnv2StridedB_hasGradAt_comp w.b2 hq.b2 _ hx.b2 (h2.congr_point (mnv2PreB2_apply N w x))
  have h0 : HasGradAt (fun y => L (mnv2SufStem N w y)) (mnv2PreB0 N w x) cotStem :=
    mnv2NoExpB_hasGradAt_comp w.b1 hq.b1 _ hx.b1 (h1.congr_point (mnv2PreB1_apply N w x))
  refine ⟨mnv2_stem_lossTiedB xN cotN vN epsStr w.sW w.sb w.sε hq.s w.sγ w.sβ bf16 x hx.stem
      (h0.congr_point (mnv2PreB0_apply N w x)) (fun W b γ β => by rw [mnv2_factor_stem]), ?_⟩
  refine ⟨mnv2_noexp_lossTiedB xN cotN vN epsStr w.b1 bf16 hq.b1 _ hx.b1
      (h1.congr_point (mnv2PreB1_apply N w x)) (fun p => by rw [mnv2_factor_b1]), ?_⟩
  refine ⟨mnv2_stride2_lossTiedB xN cotN vN epsStr w.b2 bf16 hq.b2 _ hx.b2
      (h2.congr_point (mnv2PreB2_apply N w x)) (fun p => by rw [mnv2_factor_b2]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b3 bf16 hq.b3 _ hx.b3
      (h3.congr_point (mnv2PreB3_apply N w x)) (fun p => by rw [mnv2_factor_b3]), ?_⟩
  refine ⟨mnv2_stride2_lossTiedB xN cotN vN epsStr w.b4 bf16 hq.b4 _ hx.b4
      (h4.congr_point (mnv2PreB4_apply N w x)) (fun p => by rw [mnv2_factor_b4]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b5 bf16 hq.b5 _ hx.b5
      (h5.congr_point (mnv2PreB5_apply N w x)) (fun p => by rw [mnv2_factor_b5]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b6 bf16 hq.b6 _ hx.b6
      (h6.congr_point (mnv2PreB6_apply N w x)) (fun p => by rw [mnv2_factor_b6]), ?_⟩
  refine ⟨mnv2_stride2_lossTiedB xN cotN vN epsStr w.b7 bf16 hq.b7 _ hx.b7
      (h7.congr_point (mnv2PreB7_apply N w x)) (fun p => by rw [mnv2_factor_b7]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b8 bf16 hq.b8 _ hx.b8
      (h8.congr_point (mnv2PreB8_apply N w x)) (fun p => by rw [mnv2_factor_b8]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b9 bf16 hq.b9 _ hx.b9
      (h9.congr_point (mnv2PreB9_apply N w x)) (fun p => by rw [mnv2_factor_b9]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b10 bf16 hq.b10 _ hx.b10
      (h10.congr_point (mnv2PreB10_apply N w x)) (fun p => by rw [mnv2_factor_b10]), ?_⟩
  refine ⟨mnv2_stride1_lossTiedB xN cotN vN epsStr w.b11 bf16 hq.b11 _ hx.b11
      (h11.congr_point (mnv2PreB11_apply N w x)) (fun p => by rw [mnv2_factor_b11]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b12 bf16 hq.b12 _ hx.b12
      (h12.congr_point (mnv2PreB12_apply N w x)) (fun p => by rw [mnv2_factor_b12]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b13 bf16 hq.b13 _ hx.b13
      (h13.congr_point (mnv2PreB13_apply N w x)) (fun p => by rw [mnv2_factor_b13]), ?_⟩
  refine ⟨mnv2_stride2_lossTiedB xN cotN vN epsStr w.b14 bf16 hq.b14 _ hx.b14
      (h14.congr_point (mnv2PreB14_apply N w x)) (fun p => by rw [mnv2_factor_b14]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b15 bf16 hq.b15 _ hx.b15
      (h15.congr_point (mnv2PreB15_apply N w x)) (fun p => by rw [mnv2_factor_b15]), ?_⟩
  refine ⟨mnv2_resid_lossTiedB xN cotN vN epsStr w.b16 bf16 hq.b16 _ hx.b16
      (h16.congr_point (mnv2PreB16_apply N w x)) (fun p => by rw [mnv2_factor_b16]), ?_⟩
  refine ⟨mnv2_stride1_lossTiedB xN cotN vN epsStr w.b17 bf16 hq.b17 _ hx.b17
      (h17.congr_point (mnv2PreB17_apply N w x)) (fun p => by rw [mnv2_factor_b17]), ?_⟩
  exact mnv2_head_lossTiedB xN cotN vN epsStr w.hW w.hb w.hε hq.h w.hγ w.hβ w.fcW w.fcb bf16 _ hx.head hL'
    (fun W b γ β Wd bd => by rw [mnv2_factor_head])

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLoss`, `g` the six-op cotangent the render emits. -/
theorem mnv2_net_lossGrad_smoothedCE (N : Nat) {nCls : Nat} (hK : 0 < nCls)
    (xN cotN vN epsStr aStr negAK bStr logN ohN : String) (α B : ℝ) (w : MNV2BWeights nCls)
    (hq : MNV2PosB w) (bf16 : Bool)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (hx : MNV2SmoothAtB N w x) (t : Vec (N * (1 * nCls)))
    (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    MNV2NetLossTiedB N xN cotN vN epsStr w bf16 x (smoothedBatchLoss N nCls α B t)
      (unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (mobilenetv2ForwardBFull N w x)) t))) :=
  mnv2_net_lossGrad N xN cotN vN epsStr w hq bf16 x hx
    ⟨(smoothedBatchLoss_differentiable N nCls α B t) _,
      fun J => smoothedBatchLoss_grad N nCls hK α B aStr negAK bStr logN ohN t _ ht J⟩


/-- **The emitted MobileNetV2 step's gradient nodes ARE the loss's gradient, at one chain.** For
    each of the 210 parameter slots, at ONE cotangent chain (the tie's own, from `g`): the node
    denotes its layer's Jacobian against the chain cotangent (`mnv2_net_tiedB`), and any loss `L` of
    the logits with gradient `g` at the network's output of `mobilenetv2ForwardBFull` with that one
    slot varied is differentiable there with the node as its gradient (`mnv2_net_lossGrad`). The two
    theorems each state the chain; this one states it once, so an edit to either chain breaks its
    proof. -/
theorem mnv2_net_tied_lossGrad (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : MNV2BWeights nCls) (bf16 : Bool)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (g : Vec (N * nCls))
    (hq : MNV2PosB w) (hx : MNV2SmoothAtB N w x) {L : Vec (N * nCls) → Vec 1}
    (hL : HasGradAt L (mobilenetv2ForwardBFull N w x) g) :
    -- the backward chain: the head's own four nodes, then the seventeen certified block backwards
    let dy17 := mnv2HeadCotBlk N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW (mnv2PreB17 N w x) g
    let dy16 := mnv2CotInBody N 7 7 w.b17 (mnv2PreB16 N w x) dy17
    let dy15 := mnv2ResidCotIn N 7 7 w.b16 (mnv2PreB15 N w x) dy16
    let dy14 := mnv2ResidCotIn N 7 7 w.b15 (mnv2PreB14 N w x) dy15
    let dy13 := mnv2StridedCotIn N 7 7 w.b14 (mnv2PreB13 N w x) dy14
    let dy12 := mnv2ResidCotIn N 14 14 w.b13 (mnv2PreB12 N w x) dy13
    let dy11 := mnv2ResidCotIn N 14 14 w.b12 (mnv2PreB11 N w x) dy12
    let dy10 := mnv2CotInBody N 14 14 w.b11 (mnv2PreB10 N w x) dy11
    let dy9 := mnv2ResidCotIn N 14 14 w.b10 (mnv2PreB9 N w x) dy10
    let dy8 := mnv2ResidCotIn N 14 14 w.b9 (mnv2PreB8 N w x) dy9
    let dy7 := mnv2ResidCotIn N 14 14 w.b8 (mnv2PreB7 N w x) dy8
    let dy6 := mnv2StridedCotIn N 14 14 w.b7 (mnv2PreB6 N w x) dy7
    let dy5 := mnv2ResidCotIn N 28 28 w.b6 (mnv2PreB5 N w x) dy6
    let dy4 := mnv2ResidCotIn N 28 28 w.b5 (mnv2PreB4 N w x) dy5
    let dy3 := mnv2StridedCotIn N 28 28 w.b4 (mnv2PreB3 N w x) dy4
    let dy2 := mnv2ResidCotIn N 56 56 w.b3 (mnv2PreB2 N w x) dy3
    let dy1 := mnv2StridedCotIn N 56 56 w.b2 (mnv2PreB1 N w x) dy2
    let cotStem := mnv2NoExpCotIn N 112 112 w.b1 (mnv2PreB0 N w x) dy1
    (mnv2StemTiedB N 112 112 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ bf16 x cotStem
      ∧ mnv2StemLossTiedB (N := N) (h := 112) (w := 112) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ bf16 x
        (fun W b γ β => L (mobilenetv2ForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x))
        cotStem)
  ∧ (mnv2NoExpTiedB N 112 112 xN cotN vN epsStr w.b1 bf16 (mnv2PreB0 N w x) dy1
      ∧ mnv2NoExpLossTiedB (N := N) (h := 112) (w := 112) xN cotN vN epsStr w.b1 bf16 (mnv2PreB0 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b1 := p } x)) dy1)
  ∧ (mnv2Stride2TiedB N 56 56 xN cotN vN epsStr w.b2 bf16 (mnv2PreB1 N w x) dy2
      ∧ mnv2Stride2LossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.b2 bf16 (mnv2PreB1 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b2 := p } x)) dy2)
  ∧ (mnv2Stride1TiedB N 56 56 xN cotN vN epsStr w.b3 bf16 (mnv2PreB2 N w x) dy3
      ∧ mnv2Stride1LossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.b3 bf16 (mnv2PreB2 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b3 := p } x)) dy3)
  ∧ (mnv2Stride2TiedB N 28 28 xN cotN vN epsStr w.b4 bf16 (mnv2PreB3 N w x) dy4
      ∧ mnv2Stride2LossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b4 bf16 (mnv2PreB3 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b4 := p } x)) dy4)
  ∧ (mnv2Stride1TiedB N 28 28 xN cotN vN epsStr w.b5 bf16 (mnv2PreB4 N w x) dy5
      ∧ mnv2Stride1LossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b5 bf16 (mnv2PreB4 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b5 := p } x)) dy5)
  ∧ (mnv2Stride1TiedB N 28 28 xN cotN vN epsStr w.b6 bf16 (mnv2PreB5 N w x) dy6
      ∧ mnv2Stride1LossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b6 bf16 (mnv2PreB5 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b6 := p } x)) dy6)
  ∧ (mnv2Stride2TiedB N 14 14 xN cotN vN epsStr w.b7 bf16 (mnv2PreB6 N w x) dy7
      ∧ mnv2Stride2LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b7 bf16 (mnv2PreB6 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b7 := p } x)) dy7)
  ∧ (mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b8 bf16 (mnv2PreB7 N w x) dy8
      ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b8 bf16 (mnv2PreB7 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b8 := p } x)) dy8)
  ∧ (mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b9 bf16 (mnv2PreB8 N w x) dy9
      ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b9 bf16 (mnv2PreB8 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b9 := p } x)) dy9)
  ∧ (mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b10 bf16 (mnv2PreB9 N w x) dy10
      ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b10 bf16 (mnv2PreB9 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b10 := p } x)) dy10)
  ∧ (mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b11 bf16 (mnv2PreB10 N w x) dy11
      ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b11 bf16 (mnv2PreB10 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b11 := p } x)) dy11)
  ∧ (mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b12 bf16 (mnv2PreB11 N w x) dy12
      ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b12 bf16 (mnv2PreB11 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b12 := p } x)) dy12)
  ∧ (mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b13 bf16 (mnv2PreB12 N w x) dy13
      ∧ mnv2Stride1LossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.b13 bf16 (mnv2PreB12 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b13 := p } x)) dy13)
  ∧ (mnv2Stride2TiedB N 7 7 xN cotN vN epsStr w.b14 bf16 (mnv2PreB13 N w x) dy14
      ∧ mnv2Stride2LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b14 bf16 (mnv2PreB13 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b14 := p } x)) dy14)
  ∧ (mnv2Stride1TiedB N 7 7 xN cotN vN epsStr w.b15 bf16 (mnv2PreB14 N w x) dy15
      ∧ mnv2Stride1LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b15 bf16 (mnv2PreB14 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b15 := p } x)) dy15)
  ∧ (mnv2Stride1TiedB N 7 7 xN cotN vN epsStr w.b16 bf16 (mnv2PreB15 N w x) dy16
      ∧ mnv2Stride1LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b16 bf16 (mnv2PreB15 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b16 := p } x)) dy16)
  ∧ (mnv2Stride1TiedB N 7 7 xN cotN vN epsStr w.b17 bf16 (mnv2PreB16 N w x) dy17
      ∧ mnv2Stride1LossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.b17 bf16 (mnv2PreB16 N w x)
        (fun p => L (mobilenetv2ForwardBFull N { w with b17 := p } x)) dy17)
  ∧ (mnv2HeadTiedB N 7 7 xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16
        (mnv2PreB17 N w x) g
      ∧ mnv2HeadLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb bf16
        (mnv2PreB17 N w x)
        (fun W b γ β Wd bd => L (mobilenetv2ForwardBFull N
        { w with hW := W, hb := b, hγ := γ, hβ := β, fcW := Wd, fcb := bd } x)) g) := by
  intro dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotStem
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18⟩ :=
    mnv2_net_tiedB N xN cotN vN epsStr w bf16 x g
  have hl :=
    mnv2_net_lossGrad N xN cotN vN epsStr w hq bf16 x hx hL
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, l12⟩, ⟨t13, l13⟩, ⟨t14, l14⟩, ⟨t15, l15⟩, ⟨t16, l16⟩,
    ⟨t17, l17⟩, ⟨t18, l18⟩⟩

end Proofs.MobileNetV2TieB
