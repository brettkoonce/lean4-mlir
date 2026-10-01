import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTieG
import LeanMlir.Proofs.Foundation.ParamGradNodes

/-! # EfficientNet-B0 — every parameter gradient node IS the loss's derivative in that parameter

`efficientnet_net_tiedG` says each of the 262 parameter gradient nodes denotes its layer's
parameter Jacobian contracted with the cotangent the emitted backward chain threads to it, the
chain's top being the smoothed-loss cotangent and each block's cotangent its certified VJP's
backward. `enet_net_lossGrad` composes them: for any loss `L` of the logits whose gradient at the
net's output is `g`, every node is `∂L/∂θ` of the WHOLE net `efficientnetForwardBFull` with that
one parameter varied. `enet_net_lossGrad_smoothedCE` discharges `hL` for the label-smoothed loss
the artifacts ship.

**How.** `MobileNetV2ParamGrad`'s shape:

* **The tail every MBConv block shares** (depthwise BN → swish → squeeze-excite → 1×1 project →
  project BN, over `EnTail`): the loss read at each of its activations and its gradient, the tie's
  own cotangent (`enetTail_hasGradAt`). Every stage VJP is global (swish has no kink), and the
  tie's cotangents are spelled as the certified backwards (`bnBackB`, `swBackB`, `seInB`), so each
  step is one `HasGradAt.comp`.
* **The SE gate** (`enet_se_lossTiedB`): with the block input `dr` held fixed, the SE output is the
  gated product `dr ⊙ broadcast(σ(e2))` (`seB_eq_gateMul`), whose VJP in the gate is the emitted
  `gateCotB` (`seGateMulBHasVJP`). The loss read at the excite and reduce pre-activations then has
  the tie's `cotE2` and `cotE1` as its gradients, and the four SE dense nodes are its derivatives.
* **Per block kind** (at variable widths): the expand stage (stride 1 or at the input grid
  `2h × 2w`) or none, the depthwise, then the tail. The stride-1 bundle serves both the skip blocks
  and the two widenings (`b9`, `b16`): a skip block's body sees the loss `u ↦ Gn (u + v)`, whose
  gradient at the body output is still `dyOut` (`enet_resid_lossTiedG`).
* **Bias nodes.** B0 emits every conv and depthwise bias gradient with the BatchNorm β op
  (`ConvBBetaTiedB`); `GradNodeB.biasBeta_hasGradAt` makes it the bias gradient.
* **Per net**: the loss read after each block (`enetSuf*`), pulled back through the sixteen
  certified block VJPs and the head's, and each `Φ` identified with the whole net at updated
  weights by a standalone `enet_factor_*` theorem.

**Hypotheses.** `B0Weights.EpsPos` (every BN `ε > 0`), as in the tie; there is no smoothness
hypothesis. For the smoothed loss, every example's target sums to one and `0 < nCls`. Drop-path,
classifier dropout and the bf16 nodes are outside this statement, as they are outside the tie.
-/

open Proofs Proofs.StableHLO

namespace Proofs.EnetTieG

open Proofs.BackLinks (reassocB bnBackB swBackB sigBackB cInB dInB dStridedInB gapInB seInB
  gateCotB rowB unrowB seGateMulB seGateMulBHasVJP seGateMulB_differentiable)
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The SE gate — the block split at its gated product (`seGateMulB`, BackLinks)
-- ════════════════════════════════════════════════════════════════

/-- **The batched SE block, split at the gate.** `seB` is the gated product of the block input
    with the sigmoid of the excite dense's output, the squeeze path computed on the whole batch —
    the spelling of the tie's SE activations `s e1 z e2`. -/
theorem seB_eq_gateMul (N : Nat) {c h w r : Nat} (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c)
    (b₂ : Vec c) (x : Vec (N * (c * h * w))) :
    seB N (h := h) (w := w) W₁ b₁ W₂ b₂ x
      = seGateMulB N c h w x (sigmoid (N * c) (batchMap N (dense W₂ b₂)
          (swish (N * r) (batchMap N (dense W₁ b₁) (batchMap N (globalAvgPoolFlat c h w) x))))) := by
  funext J
  simp only [seB, seBlockFull, seBlock, elemwiseProduct, seGate, seGateMulB, batchMap,
    Function.comp_apply, broadcastFlat, sigmoid, swish, Equiv.symm_apply_apply,
    Equiv.apply_symm_apply, Prod.mk.eta]
  rfl

section SE
variable (N : Nat) {c h w r : Nat}

/-- The loss at the excite dense's output (the sigmoid's input), the SE input `dr` held fixed.
    `Gse` is the loss at the SE output. -/
noncomputable def enetSeGE2 (Gse : Vec (N * (c * h * w)) → Vec 1) (dr : Vec (N * (c * h * w))) :
    Vec (N * c) → Vec 1 :=
  fun e => Gse (seGateMulB N c h w dr (sigmoid (N * c) e))

/-- The loss at the reduce dense's output (the swish's input), `dr` held fixed. -/
noncomputable def enetSeGE1 (Gse : Vec (N * (c * h * w)) → Vec 1) (dr : Vec (N * (c * h * w)))
    (W₂ : Mat r c) (b₂ : Vec c) : Vec (N * r) → Vec 1 :=
  fun e => enetSeGE2 N Gse dr (batchMap N (dense W₂ b₂) (swish (N * r) e))

variable {N}

/-- **The SE gate's two cotangents are loss gradients**: from the gradient `cot` at the SE output,
    the loss at the excite pre-activation has gradient `σ'(e2) · gateCotB dr cot` (the tie's
    `cotE2`), and at the reduce pre-activation the tie's `cotE1`. -/
theorem enetSe_hasGradAt (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (dr : Vec (N * (c * h * w))) {Gse : Vec (N * (c * h * w)) → Vec 1}
    {cot : Vec (N * (c * h * w))} (hG : HasGradAt Gse (seB N (h := h) (w := w) W₁ b₁ W₂ b₂ dr) cot) :
    let e1 := batchMap N (dense W₁ b₁) (batchMap N (globalAvgPoolFlat c h w) dr)
    let e2 := batchMap N (dense W₂ b₂) (swish (N * r) e1)
    let cotE2 := sigBackB (N * c) e2 (gateCotB N c h w dr cot)
    HasGradAt (enetSeGE2 N Gse dr) e2 cotE2
      ∧ HasGradAt (enetSeGE1 N Gse dr W₂ b₂) e1 (swBackB (N * r) e1 (rowDenseBackFlat N r c W₂ cotE2)) := by
  intro e1 e2 cotE2
  have hS : HasGradAt Gse (seGateMulB N c h w dr (sigmoid (N * c) e2)) cot :=
    hG.congr_point (seB_eq_gateMul N W₁ b₁ W₂ b₂ dr)
  have hG2 : HasGradAt (fun s => Gse (seGateMulB N c h w dr s)) (sigmoid (N * c) e2)
      (gateCotB N c h w dr cot) :=
    HasGradAt.comp (f := seGateMulB N c h w dr) (x := sigmoid (N * c) e2) hS
      ((seGateMulB_differentiable N c h w dr) _) ((seGateMulBHasVJP N c h w dr).toHasVJPAt _)
  have hE2 : HasGradAt (enetSeGE2 N Gse dr) e2 cotE2 :=
    HasGradAt.comp (f := sigmoid (N * c)) (x := e2) hG2 ((sigmoid_differentiable _) _)
      ((sigmoidHasVJP _).toHasVJPAt _)
  have hZ : HasGradAt (fun z => enetSeGE2 N Gse dr (batchMap N (dense W₂ b₂) z))
      (swish (N * r) e1) (rowDenseBackFlat N r c W₂ cotE2) :=
    HasGradAt.comp (f := batchMap N (dense W₂ b₂)) (x := swish (N * r) e1) hE2
      ((batchMap_differentiable _ (dense_differentiable W₂ b₂)) _)
      ((batchMapHasVJP _ (denseHasVJP W₂ b₂) (dense_differentiable W₂ b₂)).toHasVJPAt _)
  exact ⟨hE2, GradNodeB.hasGradAt_swish e1 hZ⟩

/-- **SE, every parameter node a loss derivative** — the reduce and excite dense W/b nodes the tie
    states, `Ψ` the loss as a function of `(W₁, b₁, W₂, b₂)` with the SE input `dr` fixed. -/
def enetSeLossTiedB (xN cotN : String) (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (dr : Vec (N * (c * h * w))) (Ψ : Mat c r → Vec r → Mat r c → Vec c → Vec 1)
    (cot : Vec (N * (c * h * w))) : Prop :=
  let s  : Vec (N * c) := batchMap N (globalAvgPoolFlat c h w) dr
  let e1 : Vec (N * r) := batchMap N (dense W₁ b₁) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * c) := batchMap N (dense W₂ b₂) z
  let cotE2 : Vec (N * c) := sigBackB (N * c) e2 (gateCotB N c h w dr cot)
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 (rowDenseBackFlat N r c W₂ cotE2)
  (HasGradAt (fun θ => Ψ (Mat.unflatten θ) b₁ W₂ b₂) (Mat.flatten W₁)
        (den (SHlo.denseWeightGradB (c := r) xN s (.operand cotN cotE1))))
  ∧ (HasGradAt (fun θ => Ψ W₁ θ W₂ b₂) b₁
        (den (SHlo.denseBiasGradB (N := N) (.operand cotN cotE1))))
  ∧ (HasGradAt (fun θ => Ψ W₁ b₁ (Mat.unflatten θ) b₂) (Mat.flatten W₂)
        (den (SHlo.denseWeightGradB (c := c) xN z (.operand cotN cotE2))))
  ∧ (HasGradAt (fun θ => Ψ W₁ b₁ W₂ θ) b₂
        (den (SHlo.denseBiasGradB (N := N) (.operand cotN cotE2))))

theorem enet_se_lossTiedB (xN cotN : String) (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c)
    (b₂ : Vec c) (dr : Vec (N * (c * h * w))) {Gse : Vec (N * (c * h * w)) → Vec 1}
    {cot : Vec (N * (c * h * w))} (hG : HasGradAt Gse (seB N (h := h) (w := w) W₁ b₁ W₂ b₂ dr) cot)
    {Ψ : Mat c r → Vec r → Mat r c → Vec c → Vec 1}
    (hΨ : ∀ a b a' b', Ψ a b a' b' = Gse (seB N (h := h) (w := w) a b a' b' dr)) :
    enetSeLossTiedB xN cotN W₁ b₁ W₂ b₂ dr Ψ cot := by
  obtain ⟨h2, h1⟩ := enetSe_hasGradAt W₁ b₁ W₂ b₂ dr hG
  have hr1 : ∀ a b, Ψ a b W₂ b₂ = enetSeGE1 N Gse dr W₂ b₂
      (batchMap N (dense a b) (batchMap N (globalAvgPoolFlat c h w) dr)) := fun a b => by
    rw [hΨ, seB_eq_gateMul]; rfl
  have hr2 : ∀ a b, Ψ W₁ b₁ a b = enetSeGE2 N Gse dr
      (batchMap N (dense a b) (swish (N * r)
        (batchMap N (dense W₁ b₁) (batchMap N (globalAvgPoolFlat c h w) dr)))) := fun a b => by
    rw [hΨ, seB_eq_gateMul]; rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp only [hr1]; exact GradNodeB.denseW_hasGradAt xN cotN _ W₁ b₁ h1
  · simp only [hr1]; exact GradNodeB.denseB_hasGradAt cotN W₁ (fun _ => 0) _ b₁ h1
  · simp only [hr2]; exact GradNodeB.denseW_hasGradAt xN cotN _ W₂ b₂ h2
  · simp only [hr2]; exact GradNodeB.denseB_hasGradAt cotN W₂ (fun _ => 0) _ b₂ h2

end SE

-- ════════════════════════════════════════════════════════════════
-- § The tail every MBConv block shares — depthwise BN, swish, SE, project conv, project BN
--   `dc` is the depthwise conv's output; `Gb` is the loss read at the tail (= body) output.
-- ════════════════════════════════════════════════════════════════

section Tail
variable (N h w : Nat) {mid oc r : Nat}

/-- The depthwise BN's output. -/
noncomputable def enetTailDn (t : EnTail mid oc r) (dc : Vec (N * (mid * h * w))) :
    Vec (N * (mid * h * w)) :=
  bnBatchLA N mid h w t.dε t.dγ t.dβ dc

/-- The depthwise swish's output (the SE input). -/
noncomputable def enetTailDr (t : EnTail mid oc r) (dc : Vec (N * (mid * h * w))) :
    Vec (N * (mid * h * w)) :=
  swish (N * (mid * h * w)) (enetTailDn N h w t dc)

/-- The SE output. -/
noncomputable def enetTailSe (t : EnTail mid oc r) (dc : Vec (N * (mid * h * w))) :
    Vec (N * (mid * h * w)) :=
  seB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 (enetTailDr N h w t dc)

/-- The project conv's output. -/
noncomputable def enetTailPc (t : EnTail mid oc r) (dc : Vec (N * (mid * h * w))) :
    Vec (N * (oc * h * w)) :=
  batchMap N (flatConv t.pW t.pb) (enetTailSe N h w t dc)

/-- The tail's output, the project BN's. -/
noncomputable def enetTailB (t : EnTail mid oc r) (dc : Vec (N * (mid * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnBatchLA N oc h w t.pε t.pγ t.pβ (enetTailPc N h w t dc)

/-- The loss at the project conv's output. -/
noncomputable def enetTailGPc (Gb : Vec (N * (oc * h * w)) → Vec 1) (t : EnTail mid oc r) :
    Vec (N * (oc * h * w)) → Vec 1 :=
  fun z => Gb (bnBatchLA N oc h w t.pε t.pγ t.pβ z)

/-- The loss at the SE output. -/
noncomputable def enetTailGSe (Gb : Vec (N * (oc * h * w)) → Vec 1) (t : EnTail mid oc r) :
    Vec (N * (mid * h * w)) → Vec 1 :=
  fun u => enetTailGPc N h w Gb t (batchMap N (flatConv t.pW t.pb) u)

/-- The loss at the depthwise BN's output. -/
noncomputable def enetTailGDn (Gb : Vec (N * (oc * h * w)) → Vec 1) (t : EnTail mid oc r) :
    Vec (N * (mid * h * w)) → Vec 1 :=
  fun u => enetTailGSe N h w Gb t
    (seB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 (swish (N * (mid * h * w)) u))

/-- The loss at the depthwise conv's output. -/
noncomputable def enetTailGDc (Gb : Vec (N * (oc * h * w)) → Vec 1) (t : EnTail mid oc r) :
    Vec (N * (mid * h * w)) → Vec 1 :=
  fun z => enetTailGDn N h w Gb t (bnBatchLA N mid h w t.dε t.dγ t.dβ z)

/-- The tie's `cotPbn`: the cotangent at the project conv's output. -/
noncomputable def enetTailCotPbn (t : EnTail mid oc r) (hp : 0 < t.pε)
    (dc : Vec (N * (mid * h * w))) (dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnBackB N oc h w t.pε hp t.pγ t.pβ (enetTailPc N h w t dc) dy

/-- The tie's `cotSeOut`: the cotangent at the SE output. -/
noncomputable def enetTailCotSeOut (t : EnTail mid oc r) (hp : 0 < t.pε)
    (dc : Vec (N * (mid * h * w))) (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  cInB N t.pW t.pb (enetTailCotPbn N h w t hp dc dy)

/-- The tie's `cotDn`: the cotangent at the depthwise BN's output. -/
noncomputable def enetTailCotDn (t : EnTail mid oc r) (hp : 0 < t.pε)
    (dc : Vec (N * (mid * h * w))) (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w)) (enetTailDn N h w t dc)
    (seInB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 (enetTailDr N h w t dc)
      (enetTailCotSeOut N h w t hp dc dy))

/-- The tie's `cotDc`: the cotangent at the depthwise conv's output. -/
noncomputable def enetTailCotDc (t : EnTail mid oc r) (hd : 0 < t.dε) (hp : 0 < t.pε)
    (dc : Vec (N * (mid * h * w))) (dy : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnBackB N mid h w t.dε hd t.dγ t.dβ dc (enetTailCotDn N h w t hp dc dy)

variable {N h w}

/-- **The tail's cotangents are loss gradients**, each stage one `HasGradAt.comp` through its
    certified VJP. -/
theorem enetTail_hasGradAt (t : EnTail mid oc r) (hd : 0 < t.dε) (hp : 0 < t.pε)
    (dc : Vec (N * (mid * h * w))) {Gb : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGb : HasGradAt Gb (enetTailB N h w t dc) dy) :
    HasGradAt (enetTailGPc N h w Gb t) (enetTailPc N h w t dc) (enetTailCotPbn N h w t hp dc dy)
      ∧ HasGradAt (enetTailGSe N h w Gb t) (enetTailSe N h w t dc)
          (enetTailCotSeOut N h w t hp dc dy)
      ∧ HasGradAt (enetTailGDn N h w Gb t) (enetTailDn N h w t dc) (enetTailCotDn N h w t hp dc dy)
      ∧ HasGradAt (enetTailGDc N h w Gb t) dc (enetTailCotDc N h w t hd hp dc dy) := by
  have hPc := GradNodeB.hasGradAt_bnBackB t.pε hp t.pγ t.pβ _ hGb
  have hSe := GradNodeB.hasGradAt_conv (h := h) (w := w) t.pW t.pb _ hPc
  have hDr : HasGradAt (fun u => enetTailGSe N h w Gb t (seB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 u))
      (enetTailDr N h w t dc)
      (seInB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2 (enetTailDr N h w t dc)
        (enetTailCotSeOut N h w t hp dc dy)) :=
    HasGradAt.comp (f := seB N (h := h) (w := w) t.z1 t.zb1 t.z2 t.zb2) (x := enetTailDr N h w t dc)
      hSe ((seB_differentiable N t.z1 t.zb1 t.z2 t.zb2) _)
      ((seBHasVJP N t.z1 t.zb1 t.z2 t.zb2).toHasVJPAt _)
  have hDn := GradNodeB.hasGradAt_swish _ hDr
  exact ⟨hPc, hSe, hDn, GradNodeB.hasGradAt_bnBackB t.dε hd t.dγ t.dβ dc hDn⟩

end Tail

-- ════════════════════════════════════════════════════════════════
-- § The stride-1 expand block body — shared by the skip blocks and the widenings (b9, b16)
--   `Gb` is the loss read at the BODY output. For a widening that is the block output; for a skip
--   block it is `u ↦ Gn (u + v)`, whose gradient there is still `dyOut` (`enet_resid_lossTiedG`).
-- ════════════════════════════════════════════════════════════════

section Exp
variable (N h w : Nat) {ic mid oc r kh kw : Nat}

/-- The expand conv's output. -/
noncomputable def enetExpEc (p : MBW ic mid oc r kh kw) (xin : Vec (N * (ic * h * w))) :
    Vec (N * (mid * h * w)) :=
  batchMap N (flatConv p.eW p.eb) xin

/-- The expand swish's output (the depthwise input). -/
noncomputable def enetExpEr (p : MBW ic mid oc r kh kw) (xin : Vec (N * (ic * h * w))) :
    Vec (N * (mid * h * w)) :=
  swish (N * (mid * h * w)) (bnBatchLA N mid h w p.eε p.eγ p.eβ (enetExpEc N h w p xin))

/-- The depthwise conv's output. -/
noncomputable def enetExpDc (p : MBW ic mid oc r kh kw) (xin : Vec (N * (ic * h * w))) :
    Vec (N * (mid * h * w)) :=
  batchMap N (depthwiseFlat p.dW p.db) (enetExpEr N h w p xin)

/-- The loss at the expand BN's output. -/
noncomputable def enetExpGEn (Gb : Vec (N * (oc * h * w)) → Vec 1) (p : MBW ic mid oc r kh kw) :
    Vec (N * (mid * h * w)) → Vec 1 :=
  fun u => enetTailGDc N h w Gb p.toEnTail
    (batchMap N (depthwiseFlat p.dW p.db) (swish (N * (mid * h * w)) u))

/-- The loss at the expand conv's output. -/
noncomputable def enetExpGEc (Gb : Vec (N * (oc * h * w)) → Vec 1) (p : MBW ic mid oc r kh kw) :
    Vec (N * (mid * h * w)) → Vec 1 :=
  fun z => enetExpGEn N h w Gb p (bnBatchLA N mid h w p.eε p.eγ p.eβ z)

variable {N h w}

theorem enetExp_hasGradAt (p : MBW ic mid oc r kh kw) (hq : p.EpsPos)
    (xin : Vec (N * (ic * h * w))) {Gb : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGb : HasGradAt Gb (mbExpW N h w p xin) dy) :
    let cotDc := enetTailCotDc N h w p.toEnTail hq.d hq.p (enetExpDc N h w p xin) dy
    let cotEn := swBackB (N * (mid * h * w))
      (bnBatchLA N mid h w p.eε p.eγ p.eβ (enetExpEc N h w p xin)) (dInB N p.dW p.db cotDc)
    HasGradAt (enetExpGEn N h w Gb p)
        (bnBatchLA N mid h w p.eε p.eγ p.eβ (enetExpEc N h w p xin)) cotEn
      ∧ HasGradAt (enetExpGEc N h w Gb p) (enetExpEc N h w p xin)
          (bnBackB N mid h w p.eε hq.e p.eγ p.eβ (enetExpEc N h w p xin) cotEn) := by
  intro cotDc cotEn
  obtain ⟨-, -, -, hDc⟩ := enetTail_hasGradAt p.toEnTail hq.d hq.p (enetExpDc N h w p xin) hGb
  have hEr := GradNodeB.hasGradAt_depthwise (h := h) (w := w) p.dW p.db _ hDc
  have hEn := GradNodeB.hasGradAt_swish _ hEr
  exact ⟨hEn, GradNodeB.hasGradAt_bnBackB p.eε hq.e p.eγ p.eβ _ hEn⟩

/-- **Stride-1 expand body, every parameter node a loss derivative** — the thirteen nodes
    `enetExpTiedG` ties (its BN pairs split into γ and β), at the tie's forward activations and
    cotangents, `Φ` the loss at the body output as a function of the block's weight record. -/
def enetExpLossTiedG (xN vN epsStr cotN : String) (p : MBW ic mid oc r kh kw) (hq : p.EpsPos)
    (xin : Vec (N * (ic * h * w))) (Φ : MBW ic mid oc r kh kw → Vec 1)
    (dyOut : Vec (N * (oc * h * w))) : Prop :=
  -- forward activations
  let ec : Vec (N * (mid * h * w)) := batchMap N (flatConv p.eW p.eb) xin
  let en : Vec (N * (mid * h * w)) := bnBatchLA N mid h w p.eε p.eγ p.eβ ec
  let er : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) en
  let dc : Vec (N * (mid * h * w)) := batchMap N (depthwiseFlat p.dW p.db) er
  let dn : Vec (N * (mid * h * w)) := bnBatchLA N mid h w p.dε p.dγ p.dβ dc
  let dr : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) dn
  let se : Vec (N * (mid * h * w)) := seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv p.pW p.pb) se
  -- backward chain cotangents (composed from dyOut)
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w p.pε hq.p p.pγ p.pβ pc dyOut
  let cotSeOut : Vec (N * (mid * h * w)) := cInB N p.pW p.pb cotPbn
  let cotDxSe : Vec (N * (mid * h * w)) := seInB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 dr cotSeOut
  let cotDn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) dn cotDxSe
  let cotDc : Vec (N * (mid * h * w)) := bnBackB N mid h w p.dε hq.d p.dγ p.dβ dc cotDn
  let cotEr : Vec (N * (mid * h * w)) := dInB N p.dW p.db cotDc
  let cotEn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) en cotEr
  let cotEc : Vec (N * (mid * h * w)) := bnBackB N mid h w p.eε hq.e p.eγ p.eβ ec cotEn
  -- expand 1×1 conv (ic → mid), cot = cotEc
  (HasGradAt (fun θ => Φ { p with eW := Kernel4.unflatten θ }) (Kernel4.flatten p.eW)
        (den (SHlo.convWeightGradB xN p.eb xin p.eW (.operand cotN cotEc))))
  ∧ (HasGradAt (fun θ => Φ { p with eb := θ }) p.eb
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotEc)))))
  ∧ (HasGradAt (fun θ => Φ { p with eγ := θ }) p.eγ
        (den (SHlo.bnGammaGradB vN epsStr p.eε (reassocB N mid h w ec)
          (.operand cotN (reassocB N mid h w cotEn)))))
  ∧ (HasGradAt (fun θ => Φ { p with eβ := θ }) p.eβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotEn)))))
  -- depthwise (stride 1), cot = cotDc
  ∧ (HasGradAt (fun θ => Φ { p with dW := Tensor3.unflatten θ }) (Tensor3.flatten p.dW)
        (den (SHlo.depthwiseWeightGradB xN p.db er p.dW (.operand cotN cotDc))))
  ∧ (HasGradAt (fun θ => Φ { p with db := θ }) p.db
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotDc)))))
  ∧ (HasGradAt (fun θ => Φ { p with dγ := θ }) p.dγ
        (den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N mid h w dc)
          (.operand cotN (reassocB N mid h w cotDn)))))
  ∧ (HasGradAt (fun θ => Φ { p with dβ := θ }) p.dβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotDn)))))
  -- SE reduce dense W₁/b₁ and excite dense W₂/b₂
  ∧ enetSeLossTiedB xN cotN p.z1 p.zb1 p.z2 p.zb2 dr
      (fun a b a' b' => Φ { p with z1 := a, zb1 := b, z2 := a', zb2 := b' }) cotSeOut
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradB xN p.pb se p.pW (.operand cotN cotPbn))))
  ∧ (HasGradAt (fun θ => Φ { p with pb := θ }) p.pb
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotPbn)))))
  ∧ (HasGradAt (fun θ => Φ { p with pγ := θ }) p.pγ
        (den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with pβ := θ }) p.pβ
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dyOut)))))

/-- The stride-1 expand bundle, from the loss `Gb` at the body output. A widening block (`b9`,
    `b16`) is this at `Gb := Gn`. -/
theorem enet_exp_lossTiedG (xN vN epsStr cotN : String) (p : MBW ic mid oc r kh kw)
    (hq : p.EpsPos) (xin : Vec (N * (ic * h * w))) {Gb : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGb : HasGradAt Gb (mbExpW N h w p xin) dy)
    {Φ : MBW ic mid oc r kh kw → Vec 1} (hΦ : ∀ p', Φ p' = Gb (mbExpW N h w p' xin)) :
    enetExpLossTiedG xN vN epsStr cotN p hq xin Φ dy := by
  rw [show Φ = fun p' => Gb (mbExpW N h w p' xin) from funext hΦ]
  obtain ⟨hPc, hSe, hDn, hDc⟩ :=
    enetTail_hasGradAt p.toEnTail hq.d hq.p (enetExpDc N h w p xin) hGb
  obtain ⟨hEn, hEc⟩ := enetExp_hasGradAt p hq xin hGb
  have hSeB := enet_se_lossTiedB xN cotN p.z1 p.zb1 p.z2 p.zb2 _ hSe
    (Ψ := fun a b a' b' => Gb (mbExpW N h w { p with z1 := a, zb1 := b, z2 := a', zb2 := b' } xin))
    (fun _ _ _ _ => rfl)
  exact ⟨GradNodeB.convW_hasGradAt xN cotN p.eb xin p.eW hEc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConv p.eW θ y)
      (GradNodeB.flatConv_bias_split p.eW) xin p.eb hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.depthwiseW_hasGradAt xN cotN p.db _ p.dW hDc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => depthwiseFlat p.dW θ y)
      (GradNodeB.depthwiseFlat_bias_split p.dW) _ p.db hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN p.dε p.dγ p.dβ _ hDn,
    hSeB,
    GradNodeB.convW_hasGradAt xN cotN p.pb _ p.pW hPc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConv p.pW θ y)
      (GradNodeB.flatConv_bias_split p.pW) _ p.pb hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.pε p.pγ p.pβ _ hGb,
    GradNodeB.bnBeta_hasGradAt cotN p.pε p.pγ p.pβ _ hGb⟩

/-- **A skip block's thirteen nodes.** The identity skip is a constant once a body parameter
    varies, so the loss at the body output has gradient `dyOut` there and the body bundle
    applies. -/
theorem enet_resid_lossTiedG {c : Nat} (xN vN epsStr cotN : String) (p : MBW c mid c r kh kw)
    (hq : p.EpsPos) (v : Vec (N * (c * h * w))) {Gn : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hGn : HasGradAt Gn (mbResidW N h w p v) dy)
    {Φ : MBW c mid c r kh kw → Vec 1} (hΦ : ∀ p', Φ p' = Gn (mbResidW N h w p' v)) :
    enetExpLossTiedG xN vN epsStr cotN p hq v Φ dy :=
  enet_exp_lossTiedG xN vN epsStr cotN p hq v
    (GradNodeB.hasGradAt_addConst (mbExpW N h w p v) v hGn) (fun p' => hΦ p')

end Exp

-- ════════════════════════════════════════════════════════════════
-- § The stride-2 block (b2, b4, b6, b12) — expand at `2h × 2w`, the symmetric strided depthwise
-- ════════════════════════════════════════════════════════════════

section Strided
variable (N h w : Nat) {ic mid oc r kh kw : Nat}

/-- The expand conv's output, at the input grid `2h × 2w`. -/
noncomputable def enetStrEc (p : MBW ic mid oc r kh kw) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  batchMap N (flatConv p.eW p.eb) xin

/-- The expand swish's output (the strided depthwise's input). -/
noncomputable def enetStrEr (p : MBW ic mid oc r kh kw) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  swish (N * (mid * (2 * h) * (2 * w)))
    (bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ (enetStrEc N h w p xin))

/-- The strided depthwise conv's output. -/
noncomputable def enetStrDc (p : MBW ic mid oc r kh kw) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    Vec (N * (mid * h * w)) :=
  batchMap N (depthwiseStride2Flat p.dW p.db) (enetStrEr N h w p xin)

/-- The loss at the expand BN's output. -/
noncomputable def enetStrGEn (Gb : Vec (N * (oc * h * w)) → Vec 1) (p : MBW ic mid oc r kh kw) :
    Vec (N * (mid * (2 * h) * (2 * w))) → Vec 1 :=
  fun u => enetTailGDc N h w Gb p.toEnTail
    (batchMap N (depthwiseStride2Flat p.dW p.db) (swish (N * (mid * (2 * h) * (2 * w))) u))

/-- The loss at the expand conv's output. -/
noncomputable def enetStrGEc (Gb : Vec (N * (oc * h * w)) → Vec 1) (p : MBW ic mid oc r kh kw) :
    Vec (N * (mid * (2 * h) * (2 * w))) → Vec 1 :=
  fun z => enetStrGEn N h w Gb p (bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ z)

variable {N h w}

theorem enetStr_hasGradAt (p : MBW ic mid oc r kh kw) (hq : p.EpsPos)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) {Gb : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGb : HasGradAt Gb (mbStridedW N h w p xin) dy) :
    let cotDc := enetTailCotDc N h w p.toEnTail hq.d hq.p (enetStrDc N h w p xin) dy
    let cotEn := swBackB (N * (mid * (2 * h) * (2 * w)))
      (bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ (enetStrEc N h w p xin))
      (dStridedInB N p.dW p.db cotDc)
    HasGradAt (enetStrGEn N h w Gb p)
        (bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ (enetStrEc N h w p xin)) cotEn
      ∧ HasGradAt (enetStrGEc N h w Gb p) (enetStrEc N h w p xin)
          (bnBackB N mid (2 * h) (2 * w) p.eε hq.e p.eγ p.eβ (enetStrEc N h w p xin) cotEn) := by
  intro cotDc cotEn
  obtain ⟨-, -, -, hDc⟩ := enetTail_hasGradAt p.toEnTail hq.d hq.p (enetStrDc N h w p xin) hGb
  have hEr := GradNodeB.hasGradAt_depthwiseStrided (h := h) (w := w) p.dW p.db _ hDc
  have hEn := GradNodeB.hasGradAt_swish _ hEr
  exact ⟨hEn, GradNodeB.hasGradAt_bnBackB p.eε hq.e p.eγ p.eβ _ hEn⟩

/-- **Stride-2 block, every parameter node a loss derivative** — the thirteen nodes
    `enetStridedTiedG` ties; the depthwise nodes are the symmetric strided ones. -/
def enetStridedLossTiedG (xN vN epsStr cotN : String) (p : MBW ic mid oc r kh kw) (hq : p.EpsPos)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (Φ : MBW ic mid oc r kh kw → Vec 1)
    (dyOut : Vec (N * (oc * h * w))) : Prop :=
  -- forward activations (expand at 2h×2w, depthwise downsamples to h×w)
  let ec : Vec (N * (mid * (2 * h) * (2 * w))) := batchMap N (flatConv p.eW p.eb) xin
  let en : Vec (N * (mid * (2 * h) * (2 * w))) := bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ ec
  let er : Vec (N * (mid * (2 * h) * (2 * w))) := swish (N * (mid * (2 * h) * (2 * w))) en
  let dc : Vec (N * (mid * h * w)) := batchMap N (depthwiseStride2Flat p.dW p.db) er
  let dn : Vec (N * (mid * h * w)) := bnBatchLA N mid h w p.dε p.dγ p.dβ dc
  let dr : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) dn
  let se : Vec (N * (mid * h * w)) := seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv p.pW p.pb) se
  -- backward chain cotangents
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w p.pε hq.p p.pγ p.pβ pc dyOut
  let cotSeOut : Vec (N * (mid * h * w)) := cInB N p.pW p.pb cotPbn
  let cotDxSe : Vec (N * (mid * h * w)) := seInB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 dr cotSeOut
  let cotDn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) dn cotDxSe
  let cotDc : Vec (N * (mid * h * w)) := bnBackB N mid h w p.dε hq.d p.dγ p.dβ dc cotDn
  let cotEr : Vec (N * (mid * (2 * h) * (2 * w))) := dStridedInB N p.dW p.db cotDc
  let cotEn : Vec (N * (mid * (2 * h) * (2 * w))) := swBackB (N * (mid * (2 * h) * (2 * w))) en cotEr
  let cotEc : Vec (N * (mid * (2 * h) * (2 * w))) :=
    bnBackB N mid (2 * h) (2 * w) p.eε hq.e p.eγ p.eβ ec cotEn
  -- expand 1×1 conv (ic → mid, at 2h×2w), cot = cotEc
  (HasGradAt (fun θ => Φ { p with eW := Kernel4.unflatten θ }) (Kernel4.flatten p.eW)
        (den (SHlo.convWeightGradB xN p.eb xin p.eW (.operand cotN cotEc))))
  ∧ (HasGradAt (fun θ => Φ { p with eb := θ }) p.eb
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := 2 * h) (w := 2 * w)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEc)))))
  ∧ (HasGradAt (fun θ => Φ { p with eγ := θ }) p.eγ
        (den (SHlo.bnGammaGradB vN epsStr p.eε (reassocB N mid (2 * h) (2 * w) ec)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEn)))))
  ∧ (HasGradAt (fun θ => Φ { p with eβ := θ }) p.eβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := 2 * h) (w := 2 * w)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEn)))))
  -- strided depthwise (2h → h), cot = cotDc
  ∧ (HasGradAt (fun θ => Φ { p with dW := Tensor3.unflatten θ }) (Tensor3.flatten p.dW)
        (den (SHlo.depthwiseStridedWeightGradB xN p.db er p.dW (.operand cotN cotDc))))
  ∧ (HasGradAt (fun θ => Φ { p with db := θ }) p.db
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotDc)))))
  ∧ (HasGradAt (fun θ => Φ { p with dγ := θ }) p.dγ
        (den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N mid h w dc)
          (.operand cotN (reassocB N mid h w cotDn)))))
  ∧ (HasGradAt (fun θ => Φ { p with dβ := θ }) p.dβ
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w cotDn)))))
  -- SE reduce/excite dense
  ∧ enetSeLossTiedB xN cotN p.z1 p.zb1 p.z2 p.zb2 dr
      (fun a b a' b' => Φ { p with z1 := a, zb1 := b, z2 := a', zb2 := b' }) cotSeOut
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradB xN p.pb se p.pW (.operand cotN cotPbn))))
  ∧ (HasGradAt (fun θ => Φ { p with pb := θ }) p.pb
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotPbn)))))
  ∧ (HasGradAt (fun θ => Φ { p with pγ := θ }) p.pγ
        (den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with pβ := θ }) p.pβ
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dyOut)))))

theorem enet_strided_lossTiedG (xN vN epsStr cotN : String) (p : MBW ic mid oc r kh kw)
    (hq : p.EpsPos) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (mbStridedW N h w p xin) dy)
    {Φ : MBW ic mid oc r kh kw → Vec 1} (hΦ : ∀ p', Φ p' = Gn (mbStridedW N h w p' xin)) :
    enetStridedLossTiedG xN vN epsStr cotN p hq xin Φ dy := by
  rw [show Φ = fun p' => Gn (mbStridedW N h w p' xin) from funext hΦ]
  obtain ⟨hPc, hSe, hDn, hDc⟩ :=
    enetTail_hasGradAt p.toEnTail hq.d hq.p (enetStrDc N h w p xin) hGn
  obtain ⟨hEn, hEc⟩ := enetStr_hasGradAt p hq xin hGn
  have hSeB := enet_se_lossTiedB xN cotN p.z1 p.zb1 p.z2 p.zb2 _ hSe
    (Ψ := fun a b a' b' =>
      Gn (mbStridedW N h w { p with z1 := a, zb1 := b, z2 := a', zb2 := b' } xin))
    (fun _ _ _ _ => rfl)
  exact ⟨GradNodeB.convW_hasGradAt xN cotN p.eb xin p.eW hEc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConv p.eW θ y)
      (GradNodeB.flatConv_bias_split p.eW) xin p.eb hEc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.bnBeta_hasGradAt cotN p.eε p.eγ p.eβ _ hEn,
    GradNodeB.depthwiseStridedW_hasGradAt xN cotN p.db _ p.dW hDc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => depthwiseStride2Flat p.dW θ y)
      (GradNodeB.depthwiseStride2Flat_bias_split p.dW) _ p.db hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN p.dε p.dγ p.dβ _ hDn,
    hSeB,
    GradNodeB.convW_hasGradAt xN cotN p.pb _ p.pW hPc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConv p.pW θ y)
      (GradNodeB.flatConv_bias_split p.pW) _ p.pb hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.pε p.pγ p.pβ _ hGn,
    GradNodeB.bnBeta_hasGradAt cotN p.pε p.pγ p.pβ _ hGn⟩

end Strided

-- ════════════════════════════════════════════════════════════════
-- § The no-expand block (b1, `t = 1`) — depthwise on the block input, then the tail
-- ════════════════════════════════════════════════════════════════

section NoExp
variable {N h w ic oc r kh kw : Nat}

/-- **No-expand block, every parameter node a loss derivative** — the ten nodes `enetNoExpTiedG`
    ties. -/
def enetNoExpLossTiedG (xN vN epsStr cotN : String) (p : MBWNoExp ic oc r kh kw) (hq : p.EpsPos)
    (xin : Vec (N * (ic * h * w))) (Φ : MBWNoExp ic oc r kh kw → Vec 1)
    (dyOut : Vec (N * (oc * h * w))) : Prop :=
  -- forward activations (depthwise on the block input ic, no expand)
  let dc : Vec (N * (ic * h * w)) := batchMap N (depthwiseFlat p.dW p.db) xin
  let dn : Vec (N * (ic * h * w)) := bnBatchLA N ic h w p.dε p.dγ p.dβ dc
  let dr : Vec (N * (ic * h * w)) := swish (N * (ic * h * w)) dn
  let se : Vec (N * (ic * h * w)) := seB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv p.pW p.pb) se
  -- backward chain cotangents
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w p.pε hq.p p.pγ p.pβ pc dyOut
  let cotSeOut : Vec (N * (ic * h * w)) := cInB N p.pW p.pb cotPbn
  let cotDxSe : Vec (N * (ic * h * w)) := seInB N (h := h) (w := w) p.z1 p.zb1 p.z2 p.zb2 dr cotSeOut
  let cotDn : Vec (N * (ic * h * w)) := swBackB (N * (ic * h * w)) dn cotDxSe
  let cotDc : Vec (N * (ic * h * w)) := bnBackB N ic h w p.dε hq.d p.dγ p.dβ dc cotDn
  -- depthwise (stride 1, on ic), cot = cotDc
  (HasGradAt (fun θ => Φ { p with dW := Tensor3.unflatten θ }) (Tensor3.flatten p.dW)
        (den (SHlo.depthwiseWeightGradB xN p.db xin p.dW (.operand cotN cotDc))))
  ∧ (HasGradAt (fun θ => Φ { p with db := θ }) p.db
        (den (SHlo.bnBetaGradB (N := N) (oc := ic) (h := h) (w := w)
          (.operand cotN (reassocB N ic h w cotDc)))))
  ∧ (HasGradAt (fun θ => Φ { p with dγ := θ }) p.dγ
        (den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N ic h w dc)
          (.operand cotN (reassocB N ic h w cotDn)))))
  ∧ (HasGradAt (fun θ => Φ { p with dβ := θ }) p.dβ
        (den (SHlo.bnBetaGradB (N := N) (oc := ic) (h := h) (w := w)
          (.operand cotN (reassocB N ic h w cotDn)))))
  -- SE reduce/excite dense (ic → r → ic)
  ∧ enetSeLossTiedB xN cotN p.z1 p.zb1 p.z2 p.zb2 dr
      (fun a b a' b' => Φ { p with z1 := a, zb1 := b, z2 := a', zb2 := b' }) cotSeOut
  -- project 1×1 conv (ic → oc), cot = cotPbn
  ∧ (HasGradAt (fun θ => Φ { p with pW := Kernel4.unflatten θ }) (Kernel4.flatten p.pW)
        (den (SHlo.convWeightGradB xN p.pb se p.pW (.operand cotN cotPbn))))
  ∧ (HasGradAt (fun θ => Φ { p with pb := θ }) p.pb
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotPbn)))))
  ∧ (HasGradAt (fun θ => Φ { p with pγ := θ }) p.pγ
        (den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
          (.operand cotN (reassocB N oc h w dyOut)))))
  ∧ (HasGradAt (fun θ => Φ { p with pβ := θ }) p.pβ
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w dyOut)))))

theorem enet_noexp_lossTiedG (xN vN epsStr cotN : String) (p : MBWNoExp ic oc r kh kw)
    (hq : p.EpsPos) (xin : Vec (N * (ic * h * w))) {Gn : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGn : HasGradAt Gn (mbNoExpW N h w p xin) dy)
    {Φ : MBWNoExp ic oc r kh kw → Vec 1} (hΦ : ∀ p', Φ p' = Gn (mbNoExpW N h w p' xin)) :
    enetNoExpLossTiedG xN vN epsStr cotN p hq xin Φ dy := by
  rw [show Φ = fun p' => Gn (mbNoExpW N h w p' xin) from funext hΦ]
  obtain ⟨hPc, hSe, hDn, hDc⟩ := enetTail_hasGradAt p.toEnTail hq.d hq.p
    (batchMap N (depthwiseFlat (h := h) (w := w) p.dW p.db) xin) hGn
  have hSeB := enet_se_lossTiedB xN cotN p.z1 p.zb1 p.z2 p.zb2 _ hSe
    (Ψ := fun a b a' b' =>
      Gn (mbNoExpW N h w { p with z1 := a, zb1 := b, z2 := a', zb2 := b' } xin))
    (fun _ _ _ _ => rfl)
  exact ⟨GradNodeB.depthwiseW_hasGradAt xN cotN p.db xin p.dW hDc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => depthwiseFlat p.dW θ y)
      (GradNodeB.depthwiseFlat_bias_split p.dW) xin p.db hDc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.dε p.dγ p.dβ _ hDn,
    GradNodeB.bnBeta_hasGradAt cotN p.dε p.dγ p.dβ _ hDn,
    hSeB,
    GradNodeB.convW_hasGradAt xN cotN p.pb _ p.pW hPc,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConv p.pW θ y)
      (GradNodeB.flatConv_bias_split p.pW) _ p.pb hPc,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.pε p.pγ p.pβ _ hGn,
    GradNodeB.bnBeta_hasGradAt cotN p.pε p.pγ p.pβ _ hGn⟩

end NoExp

-- ════════════════════════════════════════════════════════════════
-- § The stem — XLA-`SAME` strided conv, batch BN, swish
-- ════════════════════════════════════════════════════════════════

section Stem
variable {N h w ic oc kHs kWs : Nat}

/-- **Stem, every parameter node a loss derivative** — the four nodes `enetStemTiedG` ties, `Φ`
    the loss as a function of the stem's `(W, b, γ, β)`. -/
def enetStemLossTiedG (xN vN epsStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (Φ : Kernel4 oc ic kHs kWs → Vec oc → Vec oc → Vec oc → Vec 1)
    (dyStem : Vec (N * (oc * h * w))) : Prop :=
  let stc : Vec (N * (oc * h * w)) := batchMap N (flatConvStride2Xla Ws bs) x
  let stn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εs γs βs stc
  let cotBnS : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) stn dyStem
  let cotStc : Vec (N * (oc * h * w)) := bnBackB N oc h w εs hεs γs βs stc cotBnS
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bs γs βs) (Kernel4.flatten Ws)
        (den (SHlo.convStridedXlaWeightGradB xN bs x Ws (.operand cotN cotStc))))
  ∧ (HasGradAt (fun θ => Φ Ws θ γs βs) bs
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotStc)))))
  ∧ (HasGradAt (fun θ => Φ Ws bs θ βs) γs
        (den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc h w stc)
          (.operand cotN (reassocB N oc h w cotBnS)))))
  ∧ (HasGradAt (fun θ => Φ Ws bs γs θ) βs
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotBnS)))))

theorem enet_stem_lossTiedG (xN vN epsStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (stemB N (h := h) (w := w) Ws bs εs γs βs x) dy)
    {Φ : Kernel4 oc ic kHs kWs → Vec oc → Vec oc → Vec oc → Vec 1}
    (hΦ : ∀ W b γ β, Φ W b γ β = Gn (stemB N (h := h) (w := w) W b εs γ β x)) :
    enetStemLossTiedG xN vN epsStr cotN εs hεs Ws bs γs βs x Φ dy := by
  rw [show Φ = fun W b γ β => Gn (stemB N (h := h) (w := w) W b εs γ β x) from
    funext fun W => funext fun b => funext fun γ => funext fun β => hΦ W b γ β]
  have hN := GradNodeB.hasGradAt_swish
    (bnBatchLA N oc h w εs γs βs (batchMap N (flatConvStride2Xla Ws bs) x)) hGn
  have hC := GradNodeB.hasGradAt_bnBackB εs hεs γs βs _ hN
  exact ⟨GradNodeB.convStridedXlaW_hasGradAt xN cotN bs x Ws hC,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConvStride2Xla Ws θ y)
      (GradNodeB.flatConvStride2Xla_bias_split Ws) x bs hC,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εs γs βs _ hN,
    GradNodeB.bnBeta_hasGradAt cotN εs γs βs _ hN⟩

end Stem

-- ════════════════════════════════════════════════════════════════
-- § The head — 1×1 conv, batch BN, swish, GAP, dense
-- ════════════════════════════════════════════════════════════════

section Head
variable {N h w c oc nC : Nat}

/-- **Head, every parameter node a loss derivative** — the six nodes `enetHeadTiedG` ties, at the
    loss cotangent `g`, `Φ` the loss as a function of `(hW, hb, hγ, hβ, Wfc, bfc)`. -/
def enetHeadLossTiedG (xN vN epsStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w)))
    (Φ : Kernel4 oc c 1 1 → Vec oc → Vec oc → Vec oc → Mat oc nC → Vec nC → Vec 1)
    (g : Vec (N * nC)) : Prop :=
  let hc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wh bh) xhead
  let hn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εh γh βh hc
  let hr : Vec (N * (oc * h * w)) := swish (N * (oc * h * w)) hn
  let a_gap : Vec (N * oc) := batchMap N (globalAvgPoolFlat oc h w) hr
  let cotGapIn : Vec (N * oc) := rowDenseBackFlat N oc nC Wfc g
  let cotHr : Vec (N * (oc * h * w)) := gapInB N oc h w cotGapIn
  let cotHsw : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) hn cotHr
  let cotHbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εh hεh γh βh hc cotHsw
  -- head 1×1 conv (c → oc), cot = cotHbn
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bh γh βh Wfc bfc) (Kernel4.flatten Wh)
        (den (SHlo.convWeightGradB xN bh xhead Wh (.operand cotN cotHbn))))
  ∧ (HasGradAt (fun θ => Φ Wh θ γh βh Wfc bfc) bh
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotHbn)))))
  ∧ (HasGradAt (fun θ => Φ Wh bh θ βh Wfc bfc) γh
        (den (SHlo.bnGammaGradB vN epsStr εh (reassocB N oc h w hc)
          (.operand cotN (reassocB N oc h w cotHsw)))))
  ∧ (HasGradAt (fun θ => Φ Wh bh γh θ Wfc bfc) βh
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w cotHsw)))))
  -- dense classifier (oc → nC), cot = g
  ∧ (HasGradAt (fun θ => Φ Wh bh γh βh (Mat.unflatten θ) bfc) (Mat.flatten Wfc)
        (den (SHlo.denseWeightGradB (c := nC) dN a_gap (.operand cotN g))))
  ∧ (HasGradAt (fun θ => Φ Wh bh γh βh Wfc θ) bfc
        (den (SHlo.denseBiasGradB (N := N) (.operand cotN g))))

theorem enet_head_lossTiedG (xN vN epsStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) {L : Vec (N * nC) → Vec 1} {g : Vec (N * nC)}
    (hL : HasGradAt L (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc xhead) g)
    {Φ : Kernel4 oc c 1 1 → Vec oc → Vec oc → Vec oc → Mat oc nC → Vec nC → Vec 1}
    (hΦ : ∀ W b γ β Wd bd, Φ W b γ β Wd bd = L (headFwdB N (h := h) (w := w) W b εh γ β Wd bd xhead)) :
    enetHeadLossTiedG xN vN epsStr cotN dN εh hεh Wh bh γh βh Wfc bfc xhead Φ g := by
  rw [show Φ = fun W b γ β Wd bd => L (headFwdB N (h := h) (w := w) W b εh γ β Wd bd xhead) from
    funext fun W => funext fun b => funext fun γ => funext fun β => funext fun Wd =>
      funext fun bd => hΦ W b γ β Wd bd]
  let hr := cbsB N (h := h) (w := w) Wh bh εh γh βh xhead
  have hA : HasGradAt (fun a => L (batchMap N (dense Wfc bfc) a))
      (batchMap N (globalAvgPoolFlat oc h w) hr) (rowDenseBackFlat N oc nC Wfc g) :=
    HasGradAt.comp (f := batchMap N (dense Wfc bfc)) (x := batchMap N (globalAvgPoolFlat oc h w) hr)
      hL ((batchMap_differentiable _ (dense_differentiable Wfc bfc)) _)
      ((batchMapHasVJP _ (denseHasVJP Wfc bfc) (dense_differentiable Wfc bfc)).toHasVJPAt _)
  have hR : HasGradAt (fun u => L (batchMap N (dense Wfc bfc) (batchMap N (globalAvgPoolFlat oc h w) u)))
      hr (gapInB N oc h w (rowDenseBackFlat N oc nC Wfc g)) :=
    HasGradAt.comp (f := batchMap N (globalAvgPoolFlat oc h w)) (x := hr) hA
      ((batchMap_differentiable _ (globalAvgPoolFlat_differentiable oc h w)) _)
      ((batchMapHasVJP _ (globalAvgPoolFlatHasVJP oc h w)
        (globalAvgPoolFlat_differentiable oc h w)).toHasVJPAt _)
  have hN := GradNodeB.hasGradAt_swish
    (bnBatchLA N oc h w εh γh βh (batchMap N (flatConv Wh bh) xhead)) hR
  have hC := GradNodeB.hasGradAt_bnBackB εh hεh γh βh _ hN
  exact ⟨GradNodeB.convW_hasGradAt xN cotN bh xhead Wh hC,
    GradNodeB.biasBeta_hasGradAt cotN (fun θ y => flatConv Wh θ y)
      (GradNodeB.flatConv_bias_split Wh) xhead bh hC,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN εh γh βh _ hN,
    GradNodeB.bnBeta_hasGradAt cotN εh γh βh _ hN,
    GradNodeB.denseW_hasGradAt dN cotN _ Wfc bfc hL,
    GradNodeB.denseB_hasGradAt cotN Wfc (fun _ => 0) _ bfc hL⟩

end Head

-- ════════════════════════════════════════════════════════════════
-- § The whole net: the prefix before each block, the loss after it, the net with one block varied
-- ════════════════════════════════════════════════════════════════

/-- Pull the loss gradient back through the `t = 1` block's certified VJP. -/
theorem enetNoExpW_hasGradAt_comp {N h w ic oc r kh kw : Nat} (p : MBWNoExp ic oc r kh kw)
    (hq : p.EpsPos) (v : Vec (N * (ic * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hG : HasGradAt G (mbNoExpW N h w p v) dy) :
    HasGradAt (fun y => G (mbNoExpW N h w p y)) v
      ((mbNoExpWHasVJP N h w p hq.d hq.p).backward v dy) :=
  HasGradAt.comp (f := mbNoExpW N h w p) (x := v) hG
    ((mbNoExpW_differentiable N h w p hq.d hq.p) v) ((mbNoExpWHasVJP N h w p hq.d hq.p).toHasVJPAt v)

/-- …through a stride-2 block. -/
theorem enetStridedW_hasGradAt_comp {N h w ic mid oc r kh kw : Nat} (p : MBW ic mid oc r kh kw)
    (hq : p.EpsPos) (v : Vec (N * (ic * (2 * h) * (2 * w)))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hG : HasGradAt G (mbStridedW N h w p v) dy) :
    HasGradAt (fun y => G (mbStridedW N h w p y)) v
      ((mbStridedWHasVJP N h w p hq.e hq.d hq.p).backward v dy) :=
  HasGradAt.comp (f := mbStridedW N h w p) (x := v) hG
    ((mbStridedW_differentiable N h w p hq.e hq.d hq.p) v)
    ((mbStridedWHasVJP N h w p hq.e hq.d hq.p).toHasVJPAt v)

/-- …through a skip block. -/
theorem enetResidW_hasGradAt_comp {N h w c mid r kh kw : Nat} (p : MBW c mid c r kh kw)
    (hq : p.EpsPos) (v : Vec (N * (c * h * w))) {G : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hG : HasGradAt G (mbResidW N h w p v) dy) :
    HasGradAt (fun y => G (mbResidW N h w p y)) v
      ((mbResidWHasVJP N h w p hq.e hq.d hq.p).backward v dy) :=
  HasGradAt.comp (f := mbResidW N h w p) (x := v) hG
    ((mbResidW_differentiable N h w p hq.e hq.d hq.p) v)
    ((mbResidWHasVJP N h w p hq.e hq.d hq.p).toHasVJPAt v)

/-- …and through a widening block. -/
theorem enetExpW_hasGradAt_comp {N h w ic mid oc r kh kw : Nat} (p : MBW ic mid oc r kh kw)
    (hq : p.EpsPos) (v : Vec (N * (ic * h * w))) {G : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hG : HasGradAt G (mbExpW N h w p v) dy) :
    HasGradAt (fun y => G (mbExpW N h w p y)) v
      ((mbExpWHasVJP N h w p hq.e hq.d hq.p).backward v dy) :=
  HasGradAt.comp (f := mbExpW N h w p) (x := v) hG
    ((mbExpW_differentiable N h w p hq.e hq.d hq.p) v)
    ((mbExpWHasVJP N h w p hq.e hq.d hq.p).toHasVJPAt v)

/-- The stem's output — block `b1`'s input (the tie's `a0`). -/
noncomputable def enetPreB0 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (32 * 112 * 112)) :=
  stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ

/-- Block `b1`'s output (the tie's `a1`). -/
noncomputable def enetPreB1 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (16 * 112 * 112)) :=
  mbNoExpW N 112 112 w.b1 ∘ enetPreB0 N w

/-- Block `b2`'s output (the tie's `a2`). -/
noncomputable def enetPreB2 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (24 * 56 * 56)) :=
  mbStridedW N 56 56 w.b2 ∘ enetPreB1 N w

/-- Block `b3`'s output (the tie's `a3`). -/
noncomputable def enetPreB3 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (24 * 56 * 56)) :=
  mbResidW N 56 56 w.b3 ∘ enetPreB2 N w

/-- Block `b4`'s output (the tie's `a4`). -/
noncomputable def enetPreB4 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (40 * 28 * 28)) :=
  mbStridedW N 28 28 w.b4 ∘ enetPreB3 N w

/-- Block `b5`'s output (the tie's `a5`). -/
noncomputable def enetPreB5 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (40 * 28 * 28)) :=
  mbResidW N 28 28 w.b5 ∘ enetPreB4 N w

/-- Block `b6`'s output (the tie's `a6`). -/
noncomputable def enetPreB6 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (80 * 14 * 14)) :=
  mbStridedW N 14 14 w.b6 ∘ enetPreB5 N w

/-- Block `b7`'s output (the tie's `a7`). -/
noncomputable def enetPreB7 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (80 * 14 * 14)) :=
  mbResidW N 14 14 w.b7 ∘ enetPreB6 N w

/-- Block `b8`'s output (the tie's `a8`). -/
noncomputable def enetPreB8 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (80 * 14 * 14)) :=
  mbResidW N 14 14 w.b8 ∘ enetPreB7 N w

/-- Block `b9`'s output (the tie's `a9`). -/
noncomputable def enetPreB9 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (112 * 14 * 14)) :=
  mbExpW N 14 14 w.b9 ∘ enetPreB8 N w

/-- Block `b10`'s output (the tie's `a10`). -/
noncomputable def enetPreB10 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (112 * 14 * 14)) :=
  mbResidW N 14 14 w.b10 ∘ enetPreB9 N w

/-- Block `b11`'s output (the tie's `a11`). -/
noncomputable def enetPreB11 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (112 * 14 * 14)) :=
  mbResidW N 14 14 w.b11 ∘ enetPreB10 N w

/-- Block `b12`'s output (the tie's `a12`). -/
noncomputable def enetPreB12 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 7 * 7)) :=
  mbStridedW N 7 7 w.b12 ∘ enetPreB11 N w

/-- Block `b13`'s output (the tie's `a13`). -/
noncomputable def enetPreB13 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 7 * 7)) :=
  mbResidW N 7 7 w.b13 ∘ enetPreB12 N w

/-- Block `b14`'s output (the tie's `a14`). -/
noncomputable def enetPreB14 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 7 * 7)) :=
  mbResidW N 7 7 w.b14 ∘ enetPreB13 N w

/-- Block `b15`'s output (the tie's `a15`). -/
noncomputable def enetPreB15 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (192 * 7 * 7)) :=
  mbResidW N 7 7 w.b15 ∘ enetPreB14 N w

/-- Block `b16`'s output (the tie's `a16`). -/
noncomputable def enetPreB16 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (3 * 224 * 224)) → Vec (N * (320 * 7 * 7)) :=
  mbExpW N 7 7 w.b16 ∘ enetPreB15 N w

theorem enetPreB0_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB0 N w x = stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ x := by
  rw [enetPreB0]

theorem enetPreB1_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB1 N w x = mbNoExpW N 112 112 w.b1 (enetPreB0 N w x) := by
  rw [enetPreB1, Function.comp_apply]

theorem enetPreB2_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB2 N w x = mbStridedW N 56 56 w.b2 (enetPreB1 N w x) := by
  rw [enetPreB2, Function.comp_apply]

theorem enetPreB3_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB3 N w x = mbResidW N 56 56 w.b3 (enetPreB2 N w x) := by
  rw [enetPreB3, Function.comp_apply]

theorem enetPreB4_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB4 N w x = mbStridedW N 28 28 w.b4 (enetPreB3 N w x) := by
  rw [enetPreB4, Function.comp_apply]

theorem enetPreB5_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB5 N w x = mbResidW N 28 28 w.b5 (enetPreB4 N w x) := by
  rw [enetPreB5, Function.comp_apply]

theorem enetPreB6_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB6 N w x = mbStridedW N 14 14 w.b6 (enetPreB5 N w x) := by
  rw [enetPreB6, Function.comp_apply]

theorem enetPreB7_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB7 N w x = mbResidW N 14 14 w.b7 (enetPreB6 N w x) := by
  rw [enetPreB7, Function.comp_apply]

theorem enetPreB8_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB8 N w x = mbResidW N 14 14 w.b8 (enetPreB7 N w x) := by
  rw [enetPreB8, Function.comp_apply]

theorem enetPreB9_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB9 N w x = mbExpW N 14 14 w.b9 (enetPreB8 N w x) := by
  rw [enetPreB9, Function.comp_apply]

theorem enetPreB10_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB10 N w x = mbResidW N 14 14 w.b10 (enetPreB9 N w x) := by
  rw [enetPreB10, Function.comp_apply]

theorem enetPreB11_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB11 N w x = mbResidW N 14 14 w.b11 (enetPreB10 N w x) := by
  rw [enetPreB11, Function.comp_apply]

theorem enetPreB12_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB12 N w x = mbStridedW N 7 7 w.b12 (enetPreB11 N w x) := by
  rw [enetPreB12, Function.comp_apply]

theorem enetPreB13_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB13 N w x = mbResidW N 7 7 w.b13 (enetPreB12 N w x) := by
  rw [enetPreB13, Function.comp_apply]

theorem enetPreB14_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB14 N w x = mbResidW N 7 7 w.b14 (enetPreB13 N w x) := by
  rw [enetPreB14, Function.comp_apply]

theorem enetPreB15_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB15 N w x = mbResidW N 7 7 w.b15 (enetPreB14 N w x) := by
  rw [enetPreB15, Function.comp_apply]

theorem enetPreB16_apply (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    enetPreB16 N w x = mbExpW N 7 7 w.b16 (enetPreB15 N w x) := by
  rw [enetPreB16, Function.comp_apply]

/-- The net after block `b16` — the head. -/
noncomputable def enetSufB16 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (320 * 7 * 7)) → Vec (N * nCls) :=
  headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb

/-- The net after block `b15`: block `b16`, then the rest. -/
noncomputable def enetSufB15 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (192 * 7 * 7)) → Vec (N * nCls) :=
  fun y => enetSufB16 N w (mbExpW N 7 7 w.b16 y)

/-- The net after block `b14`: block `b15`, then the rest. -/
noncomputable def enetSufB14 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (192 * 7 * 7)) → Vec (N * nCls) :=
  fun y => enetSufB15 N w (mbResidW N 7 7 w.b15 y)

/-- The net after block `b13`: block `b14`, then the rest. -/
noncomputable def enetSufB13 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (192 * 7 * 7)) → Vec (N * nCls) :=
  fun y => enetSufB14 N w (mbResidW N 7 7 w.b14 y)

/-- The net after block `b12`: block `b13`, then the rest. -/
noncomputable def enetSufB12 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (192 * 7 * 7)) → Vec (N * nCls) :=
  fun y => enetSufB13 N w (mbResidW N 7 7 w.b13 y)

/-- The net after block `b11`: block `b12`, then the rest. -/
noncomputable def enetSufB11 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (112 * 14 * 14)) → Vec (N * nCls) :=
  fun y => enetSufB12 N w (mbStridedW N 7 7 w.b12 y)

/-- The net after block `b10`: block `b11`, then the rest. -/
noncomputable def enetSufB10 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (112 * 14 * 14)) → Vec (N * nCls) :=
  fun y => enetSufB11 N w (mbResidW N 14 14 w.b11 y)

/-- The net after block `b9`: block `b10`, then the rest. -/
noncomputable def enetSufB9 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (112 * 14 * 14)) → Vec (N * nCls) :=
  fun y => enetSufB10 N w (mbResidW N 14 14 w.b10 y)

/-- The net after block `b8`: block `b9`, then the rest. -/
noncomputable def enetSufB8 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (80 * 14 * 14)) → Vec (N * nCls) :=
  fun y => enetSufB9 N w (mbExpW N 14 14 w.b9 y)

/-- The net after block `b7`: block `b8`, then the rest. -/
noncomputable def enetSufB7 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (80 * 14 * 14)) → Vec (N * nCls) :=
  fun y => enetSufB8 N w (mbResidW N 14 14 w.b8 y)

/-- The net after block `b6`: block `b7`, then the rest. -/
noncomputable def enetSufB6 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (80 * 14 * 14)) → Vec (N * nCls) :=
  fun y => enetSufB7 N w (mbResidW N 14 14 w.b7 y)

/-- The net after block `b5`: block `b6`, then the rest. -/
noncomputable def enetSufB5 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (40 * 28 * 28)) → Vec (N * nCls) :=
  fun y => enetSufB6 N w (mbStridedW N 14 14 w.b6 y)

/-- The net after block `b4`: block `b5`, then the rest. -/
noncomputable def enetSufB4 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (40 * 28 * 28)) → Vec (N * nCls) :=
  fun y => enetSufB5 N w (mbResidW N 28 28 w.b5 y)

/-- The net after block `b3`: block `b4`, then the rest. -/
noncomputable def enetSufB3 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (24 * 56 * 56)) → Vec (N * nCls) :=
  fun y => enetSufB4 N w (mbStridedW N 28 28 w.b4 y)

/-- The net after block `b2`: block `b3`, then the rest. -/
noncomputable def enetSufB2 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (24 * 56 * 56)) → Vec (N * nCls) :=
  fun y => enetSufB3 N w (mbResidW N 56 56 w.b3 y)

/-- The net after block `b1`: block `b2`, then the rest. -/
noncomputable def enetSufB1 (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (16 * 112 * 112)) → Vec (N * nCls) :=
  fun y => enetSufB2 N w (mbStridedW N 56 56 w.b2 y)

/-- The net after the stem: block `b1`, then the rest. -/
noncomputable def enetSufStem (N : Nat) {nCls : Nat} (w : B0Weights nCls) :
    Vec (N * (32 * 112 * 112)) → Vec (N * nCls) :=
  fun y => enetSufB1 N w (mbNoExpW N 112 112 w.b1 y)

/-- **The net with the stem's parameters varied** is the suffix after the stem at the varied stem. -/
theorem enet_factor_stem (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 32 3 3 3) (b γ β : Vec 32) :
    efficientnetForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x
      = enetSufStem N w (stemB N (h := 112) (w := 112) W b w.sε γ β x) := rfl

/-- **The net with block `b1`'s weights varied** is the suffix after `b1` at the varied block. -/
theorem enet_factor_b1 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBWNoExp 32 16 8 3 3) :
    efficientnetForwardBFull N { w with b1 := p } x
      = enetSufB1 N w (mbNoExpW N 112 112 p (enetPreB0 N w x)) := by
  rw [enetPreB0_apply]; rfl

/-- **The net with block `b2`'s weights varied** is the suffix after `b2` at the varied block. -/
theorem enet_factor_b2 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 16 96 24 4 3 3) :
    efficientnetForwardBFull N { w with b2 := p } x
      = enetSufB2 N w (mbStridedW N 56 56 p (enetPreB1 N w x)) := by
  rw [enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b3`'s weights varied** is the suffix after `b3` at the varied block. -/
theorem enet_factor_b3 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 24 144 24 6 3 3) :
    efficientnetForwardBFull N { w with b3 := p } x
      = enetSufB3 N w (mbResidW N 56 56 p (enetPreB2 N w x)) := by
  rw [enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b4`'s weights varied** is the suffix after `b4` at the varied block. -/
theorem enet_factor_b4 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 24 144 40 6 5 5) :
    efficientnetForwardBFull N { w with b4 := p } x
      = enetSufB4 N w (mbStridedW N 28 28 p (enetPreB3 N w x)) := by
  rw [enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b5`'s weights varied** is the suffix after `b5` at the varied block. -/
theorem enet_factor_b5 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 40 240 40 10 5 5) :
    efficientnetForwardBFull N { w with b5 := p } x
      = enetSufB5 N w (mbResidW N 28 28 p (enetPreB4 N w x)) := by
  rw [enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b6`'s weights varied** is the suffix after `b6` at the varied block. -/
theorem enet_factor_b6 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 40 240 80 10 3 3) :
    efficientnetForwardBFull N { w with b6 := p } x
      = enetSufB6 N w (mbStridedW N 14 14 p (enetPreB5 N w x)) := by
  rw [enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b7`'s weights varied** is the suffix after `b7` at the varied block. -/
theorem enet_factor_b7 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 80 480 80 20 3 3) :
    efficientnetForwardBFull N { w with b7 := p } x
      = enetSufB7 N w (mbResidW N 14 14 p (enetPreB6 N w x)) := by
  rw [enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b8`'s weights varied** is the suffix after `b8` at the varied block. -/
theorem enet_factor_b8 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 80 480 80 20 3 3) :
    efficientnetForwardBFull N { w with b8 := p } x
      = enetSufB8 N w (mbResidW N 14 14 p (enetPreB7 N w x)) := by
  rw [enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b9`'s weights varied** is the suffix after `b9` at the varied block. -/
theorem enet_factor_b9 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 80 480 112 20 5 5) :
    efficientnetForwardBFull N { w with b9 := p } x
      = enetSufB9 N w (mbExpW N 14 14 p (enetPreB8 N w x)) := by
  rw [enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b10`'s weights varied** is the suffix after `b10` at the varied block. -/
theorem enet_factor_b10 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 112 672 112 28 5 5) :
    efficientnetForwardBFull N { w with b10 := p } x
      = enetSufB10 N w (mbResidW N 14 14 p (enetPreB9 N w x)) := by
  rw [enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b11`'s weights varied** is the suffix after `b11` at the varied block. -/
theorem enet_factor_b11 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 112 672 112 28 5 5) :
    efficientnetForwardBFull N { w with b11 := p } x
      = enetSufB11 N w (mbResidW N 14 14 p (enetPreB10 N w x)) := by
  rw [enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b12`'s weights varied** is the suffix after `b12` at the varied block. -/
theorem enet_factor_b12 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 112 672 192 28 5 5) :
    efficientnetForwardBFull N { w with b12 := p } x
      = enetSufB12 N w (mbStridedW N 7 7 p (enetPreB11 N w x)) := by
  rw [enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b13`'s weights varied** is the suffix after `b13` at the varied block. -/
theorem enet_factor_b13 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 192 1152 192 48 5 5) :
    efficientnetForwardBFull N { w with b13 := p } x
      = enetSufB13 N w (mbResidW N 7 7 p (enetPreB12 N w x)) := by
  rw [enetPreB12_apply, enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b14`'s weights varied** is the suffix after `b14` at the varied block. -/
theorem enet_factor_b14 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 192 1152 192 48 5 5) :
    efficientnetForwardBFull N { w with b14 := p } x
      = enetSufB14 N w (mbResidW N 7 7 p (enetPreB13 N w x)) := by
  rw [enetPreB13_apply, enetPreB12_apply, enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b15`'s weights varied** is the suffix after `b15` at the varied block. -/
theorem enet_factor_b15 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 192 1152 192 48 5 5) :
    efficientnetForwardBFull N { w with b15 := p } x
      = enetSufB15 N w (mbResidW N 7 7 p (enetPreB14 N w x)) := by
  rw [enetPreB14_apply, enetPreB13_apply, enetPreB12_apply, enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with block `b16`'s weights varied** is the suffix after `b16` at the varied block. -/
theorem enet_factor_b16 (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (p : MBW 192 1152 320 48 3 3) :
    efficientnetForwardBFull N { w with b16 := p } x
      = enetSufB16 N w (mbExpW N 7 7 p (enetPreB15 N w x)) := by
  rw [enetPreB15_apply, enetPreB14_apply, enetPreB13_apply, enetPreB12_apply, enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **The net with the head varied** is the head at the varied parameters. -/
theorem enet_factor_head (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224)))
    (W : Kernel4 1280 320 1 1) (b γ β : Vec 1280) (Wd : Mat 1280 nCls) (bd : Vec nCls) :
    efficientnetForwardBFull N { w with hW := W, hb := b, hγ := γ, hβ := β, fcW := Wd, fcb := bd } x
      = headFwdB N (h := 7) (w := 7) W b w.hε γ β Wd bd (enetPreB16 N w x) := by
  rw [enetPreB16_apply, enetPreB15_apply, enetPreB14_apply, enetPreB13_apply, enetPreB12_apply, enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- The net's output is the head at block `b16`'s output. -/
theorem enet_forward_eq_head (N : Nat) {nCls : Nat} (w : B0Weights nCls) (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFull N w x
      = headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb (enetPreB16 N w x) := by
  rw [enetPreB16_apply, enetPreB15_apply, enetPreB14_apply, enetPreB13_apply, enetPreB12_apply, enetPreB11_apply, enetPreB10_apply, enetPreB9_apply, enetPreB8_apply, enetPreB7_apply, enetPreB6_apply, enetPreB5_apply, enetPreB4_apply, enetPreB3_apply, enetPreB2_apply, enetPreB1_apply, enetPreB0_apply]; rfl

/-- **Every EfficientNet-B0 parameter gradient node is the derivative of `L` in that parameter**,
    for a loss `L` of the logits and `g` the cotangent the chain starts from: the 262 nodes
    `efficientnet_net_tiedG` ties, each at the cotangent the certified block VJPs thread to it from
    `g`, stated against `L` of `efficientnetForwardBFull` with that one parameter varied. -/
def EnetNetLossTiedG (xN vN epsStr cotN dN : String) (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (hεw : w.EpsPos) (x : Vec (N * (3 * 224 * 224))) (L : Vec (N * nCls) → Vec 1) (g : Vec (N * nCls)) : Prop :=
    let dy16 := (headFwdBHasVJP N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb).backward
        (enetPreB16 N w x) g
    let dy15 := (mbExpWHasVJP N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p).backward (enetPreB15 N w x) dy16
    let dy14 := (mbResidWHasVJP N 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p).backward (enetPreB14 N w x) dy15
    let dy13 := (mbResidWHasVJP N 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p).backward (enetPreB13 N w x) dy14
    let dy12 := (mbResidWHasVJP N 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p).backward (enetPreB12 N w x) dy13
    let dy11 := (mbStridedWHasVJP N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p).backward (enetPreB11 N w x) dy12
    let dy10 := (mbResidWHasVJP N 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p).backward (enetPreB10 N w x) dy11
    let dy9 := (mbResidWHasVJP N 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p).backward (enetPreB9 N w x) dy10
    let dy8 := (mbExpWHasVJP N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p).backward (enetPreB8 N w x) dy9
    let dy7 := (mbResidWHasVJP N 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p).backward (enetPreB7 N w x) dy8
    let dy6 := (mbResidWHasVJP N 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p).backward (enetPreB6 N w x) dy7
    let dy5 := (mbStridedWHasVJP N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p).backward (enetPreB5 N w x) dy6
    let dy4 := (mbResidWHasVJP N 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p).backward (enetPreB4 N w x) dy5
    let dy3 := (mbStridedWHasVJP N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p).backward (enetPreB3 N w x) dy4
    let dy2 := (mbResidWHasVJP N 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p).backward (enetPreB2 N w x) dy3
    let dy1 := (mbStridedWHasVJP N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p).backward (enetPreB1 N w x) dy2
    let dy0 := (mbNoExpWHasVJP N 112 112 w.b1 hεw.b1.d hεw.b1.p).backward (enetPreB0 N w x) dy1
    enetStemLossTiedG (N := N) (h := 112) (w := 112) xN vN epsStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x
      (fun W b γ β => L (efficientnetForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x))
      dy0
  ∧ enetNoExpLossTiedG (N := N) (h := 112) (w := 112) xN vN epsStr cotN w.b1 hεw.b1 (enetPreB0 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b1 := p } x)) dy1
  ∧ enetStridedLossTiedG (N := N) (h := 56) (w := 56) xN vN epsStr cotN w.b2 hεw.b2 (enetPreB1 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b2 := p } x)) dy2
  ∧ enetExpLossTiedG (N := N) (h := 56) (w := 56) xN vN epsStr cotN w.b3 hεw.b3 (enetPreB2 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b3 := p } x)) dy3
  ∧ enetStridedLossTiedG (N := N) (h := 28) (w := 28) xN vN epsStr cotN w.b4 hεw.b4 (enetPreB3 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b4 := p } x)) dy4
  ∧ enetExpLossTiedG (N := N) (h := 28) (w := 28) xN vN epsStr cotN w.b5 hεw.b5 (enetPreB4 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b5 := p } x)) dy5
  ∧ enetStridedLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b6 hεw.b6 (enetPreB5 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b6 := p } x)) dy6
  ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b7 hεw.b7 (enetPreB6 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b7 := p } x)) dy7
  ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b8 hεw.b8 (enetPreB7 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b8 := p } x)) dy8
  ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b9 hεw.b9 (enetPreB8 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b9 := p } x)) dy9
  ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b10 hεw.b10 (enetPreB9 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b10 := p } x)) dy10
  ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b11 hεw.b11 (enetPreB10 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b11 := p } x)) dy11
  ∧ enetStridedLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b12 hεw.b12 (enetPreB11 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b12 := p } x)) dy12
  ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b13 hεw.b13 (enetPreB12 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b13 := p } x)) dy13
  ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b14 hεw.b14 (enetPreB13 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b14 := p } x)) dy14
  ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b15 hεw.b15 (enetPreB14 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b15 := p } x)) dy15
  ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b16 hεw.b16 (enetPreB15 N w x)
      (fun p => L (efficientnetForwardBFull N { w with b16 := p } x)) dy16
  ∧ enetHeadLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ
      w.fcW w.fcb (enetPreB16 N w x)
      (fun W b γ β Wd bd => L (efficientnetForwardBFull N
        { w with hW := W, hb := b, hγ := γ, hβ := β, fcW := Wd, fcb := bd } x)) g

/-- **Every EfficientNet-B0 parameter gradient node is the derivative of the loss in that
    parameter.** For any loss `L` of the logits with gradient `g` at the net's output, each of the
    262 nodes `efficientnet_net_tiedG` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net,
    `efficientnetForwardBFull` with that one parameter varied (a stem field, a block's weight
    record `w.bk := p` with one slot changed, or a head field).

    Hypothesis: every BN `ε` positive (`B0Weights.EpsPos`), as in the tie. The loss enters only
    through `hL`; `enet_net_lossGrad_smoothedCE` discharges it for the loss the artifacts ship. -/
theorem enet_net_lossGrad (xN vN epsStr cotN dN : String) (N : Nat) {nCls : Nat}
    (w : B0Weights nCls) (hεw : w.EpsPos) (x : Vec (N * (3 * 224 * 224))) {L : Vec (N * nCls) → Vec 1}
    {g : Vec (N * nCls)} (hL : HasGradAt L (efficientnetForwardBFull N w x) g) :
    EnetNetLossTiedG xN vN epsStr cotN dN N w hεw x L g := by
  unfold EnetNetLossTiedG
  intro dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dy0
  have hL' : HasGradAt L (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
      (enetPreB16 N w x)) g := hL.congr_point (enet_forward_eq_head N w x)
  have h16 : HasGradAt (fun y => L (enetSufB16 N w y)) (enetPreB16 N w x) dy16 :=
    HasGradAt.comp_global (f := headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb)
      (x := enetPreB16 N w x) hL'
      (headFwdB_differentiable N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb)
      (headFwdBHasVJP N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb)
  have h15 : HasGradAt (fun y => L (enetSufB15 N w y)) (enetPreB15 N w x) dy15 :=
    enetExpW_hasGradAt_comp w.b16 hεw.b16 _ (h16.congr_point (enetPreB16_apply N w x))
  have h14 : HasGradAt (fun y => L (enetSufB14 N w y)) (enetPreB14 N w x) dy14 :=
    enetResidW_hasGradAt_comp w.b15 hεw.b15 _ (h15.congr_point (enetPreB15_apply N w x))
  have h13 : HasGradAt (fun y => L (enetSufB13 N w y)) (enetPreB13 N w x) dy13 :=
    enetResidW_hasGradAt_comp w.b14 hεw.b14 _ (h14.congr_point (enetPreB14_apply N w x))
  have h12 : HasGradAt (fun y => L (enetSufB12 N w y)) (enetPreB12 N w x) dy12 :=
    enetResidW_hasGradAt_comp w.b13 hεw.b13 _ (h13.congr_point (enetPreB13_apply N w x))
  have h11 : HasGradAt (fun y => L (enetSufB11 N w y)) (enetPreB11 N w x) dy11 :=
    enetStridedW_hasGradAt_comp w.b12 hεw.b12 _ (h12.congr_point (enetPreB12_apply N w x))
  have h10 : HasGradAt (fun y => L (enetSufB10 N w y)) (enetPreB10 N w x) dy10 :=
    enetResidW_hasGradAt_comp w.b11 hεw.b11 _ (h11.congr_point (enetPreB11_apply N w x))
  have h9 : HasGradAt (fun y => L (enetSufB9 N w y)) (enetPreB9 N w x) dy9 :=
    enetResidW_hasGradAt_comp w.b10 hεw.b10 _ (h10.congr_point (enetPreB10_apply N w x))
  have h8 : HasGradAt (fun y => L (enetSufB8 N w y)) (enetPreB8 N w x) dy8 :=
    enetExpW_hasGradAt_comp w.b9 hεw.b9 _ (h9.congr_point (enetPreB9_apply N w x))
  have h7 : HasGradAt (fun y => L (enetSufB7 N w y)) (enetPreB7 N w x) dy7 :=
    enetResidW_hasGradAt_comp w.b8 hεw.b8 _ (h8.congr_point (enetPreB8_apply N w x))
  have h6 : HasGradAt (fun y => L (enetSufB6 N w y)) (enetPreB6 N w x) dy6 :=
    enetResidW_hasGradAt_comp w.b7 hεw.b7 _ (h7.congr_point (enetPreB7_apply N w x))
  have h5 : HasGradAt (fun y => L (enetSufB5 N w y)) (enetPreB5 N w x) dy5 :=
    enetStridedW_hasGradAt_comp w.b6 hεw.b6 _ (h6.congr_point (enetPreB6_apply N w x))
  have h4 : HasGradAt (fun y => L (enetSufB4 N w y)) (enetPreB4 N w x) dy4 :=
    enetResidW_hasGradAt_comp w.b5 hεw.b5 _ (h5.congr_point (enetPreB5_apply N w x))
  have h3 : HasGradAt (fun y => L (enetSufB3 N w y)) (enetPreB3 N w x) dy3 :=
    enetStridedW_hasGradAt_comp w.b4 hεw.b4 _ (h4.congr_point (enetPreB4_apply N w x))
  have h2 : HasGradAt (fun y => L (enetSufB2 N w y)) (enetPreB2 N w x) dy2 :=
    enetResidW_hasGradAt_comp w.b3 hεw.b3 _ (h3.congr_point (enetPreB3_apply N w x))
  have h1 : HasGradAt (fun y => L (enetSufB1 N w y)) (enetPreB1 N w x) dy1 :=
    enetStridedW_hasGradAt_comp w.b2 hεw.b2 _ (h2.congr_point (enetPreB2_apply N w x))
  have h0 : HasGradAt (fun y => L (enetSufStem N w y)) (enetPreB0 N w x) dy0 :=
    enetNoExpW_hasGradAt_comp w.b1 hεw.b1 _ (h1.congr_point (enetPreB1_apply N w x))
  refine ⟨enet_stem_lossTiedG (h := 112) (w := 112) xN vN epsStr cotN w.sε hεw.s w.sW w.sb w.sγ
      w.sβ x
      (h0.congr_point (enetPreB0_apply N w x)) (fun W b γ β => by rw [enet_factor_stem]), ?_⟩
  refine ⟨enet_noexp_lossTiedG xN vN epsStr cotN w.b1 hεw.b1 _
      (h1.congr_point (enetPreB1_apply N w x)) (fun p => by rw [enet_factor_b1]), ?_⟩
  refine ⟨enet_strided_lossTiedG xN vN epsStr cotN w.b2 hεw.b2 _
      (h2.congr_point (enetPreB2_apply N w x)) (fun p => by rw [enet_factor_b2]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b3 hεw.b3 _
      (h3.congr_point (enetPreB3_apply N w x)) (fun p => by rw [enet_factor_b3]), ?_⟩
  refine ⟨enet_strided_lossTiedG xN vN epsStr cotN w.b4 hεw.b4 _
      (h4.congr_point (enetPreB4_apply N w x)) (fun p => by rw [enet_factor_b4]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b5 hεw.b5 _
      (h5.congr_point (enetPreB5_apply N w x)) (fun p => by rw [enet_factor_b5]), ?_⟩
  refine ⟨enet_strided_lossTiedG xN vN epsStr cotN w.b6 hεw.b6 _
      (h6.congr_point (enetPreB6_apply N w x)) (fun p => by rw [enet_factor_b6]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b7 hεw.b7 _
      (h7.congr_point (enetPreB7_apply N w x)) (fun p => by rw [enet_factor_b7]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b8 hεw.b8 _
      (h8.congr_point (enetPreB8_apply N w x)) (fun p => by rw [enet_factor_b8]), ?_⟩
  refine ⟨enet_exp_lossTiedG xN vN epsStr cotN w.b9 hεw.b9 _
      (h9.congr_point (enetPreB9_apply N w x)) (fun p => by rw [enet_factor_b9]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b10 hεw.b10 _
      (h10.congr_point (enetPreB10_apply N w x)) (fun p => by rw [enet_factor_b10]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b11 hεw.b11 _
      (h11.congr_point (enetPreB11_apply N w x)) (fun p => by rw [enet_factor_b11]), ?_⟩
  refine ⟨enet_strided_lossTiedG xN vN epsStr cotN w.b12 hεw.b12 _
      (h12.congr_point (enetPreB12_apply N w x)) (fun p => by rw [enet_factor_b12]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b13 hεw.b13 _
      (h13.congr_point (enetPreB13_apply N w x)) (fun p => by rw [enet_factor_b13]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b14 hεw.b14 _
      (h14.congr_point (enetPreB14_apply N w x)) (fun p => by rw [enet_factor_b14]), ?_⟩
  refine ⟨enet_resid_lossTiedG xN vN epsStr cotN w.b15 hεw.b15 _
      (h15.congr_point (enetPreB15_apply N w x)) (fun p => by rw [enet_factor_b15]), ?_⟩
  refine ⟨enet_exp_lossTiedG xN vN epsStr cotN w.b16 hεw.b16 _
      (h16.congr_point (enetPreB16_apply N w x)) (fun p => by rw [enet_factor_b16]), ?_⟩
  exact enet_head_lossTiedG xN vN epsStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ w.fcW w.fcb _ hL'
    (fun W b γ β Wd bd => by rw [enet_factor_head])

/-- **The loss the artifacts ship**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLoss`, `g` the six-op cotangent the render emits — the tie's own
    `g`, whose logits `headFwdB … a16` are `efficientnetForwardBFull N w x`. -/
theorem enet_net_lossGrad_smoothedCE (xN vN epsStr cotN dN aStr negAK bStr logN ohN : String)
    (N : Nat) {nCls : Nat} (hK : 0 < nCls) (α B : ℝ) (w : B0Weights nCls) (hεw : w.EpsPos)
    (x : Vec (N * (3 * 224 * 224))) (t : Vec (N * (1 * nCls)))
    (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    EnetNetLossTiedG xN vN epsStr cotN dN N w hεw x (smoothedBatchLoss N nCls α B t)
      (unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (efficientnetForwardBFull N w x)) t))) :=
  enet_net_lossGrad xN vN epsStr cotN dN N w hεw x
    ⟨(smoothedBatchLoss_differentiable N nCls α B t) _,
      fun J => smoothedBatchLoss_grad N nCls hK α B aStr negAK bStr logN ohN t _ ht J⟩

/-- **The emitted EfficientNet-B0 step's gradient nodes ARE the loss's gradient, at one chain.**
    For each of the 262 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`efficientnet_net_tiedG`), and the batched smoothed loss of `efficientnetForwardBFull` with that
    one slot varied is differentiable there with the node as its gradient
    (`enet_net_lossGrad_smoothedCE`). The tie spells each block input as its own let; the proof
    rewrites the loss side's `enetPre*` into those lets (`enetPreB0_apply`, …) and the loss side's
    logits into the tie's (`enet_forward_eq_head`). -/
theorem enet_net_tied_lossGrad (xN vN epsStr cotN dN : String) (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (hεw : w.EpsPos)
    (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (x : Vec (N * (3 * 224 * 224))) (t : Vec (N * (1 * nCls)))
    (hK : 0 < nCls) (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    -- forward block inputs (the prefixes of efficientnetForwardBFull)
    let a0  : Vec (N * (32 * 112 * 112)) := stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ x
    let a1  : Vec (N * (16 * 112 * 112)) := mbNoExpW N 112 112 w.b1 a0
    let a2  : Vec (N * (24 * 56 * 56))   := mbStridedW N 56 56 w.b2 a1
    let a3  : Vec (N * (24 * 56 * 56))   := mbResidW N 56 56 w.b3 a2
    let a4  : Vec (N * (40 * 28 * 28))   := mbStridedW N 28 28 w.b4 a3
    let a5  : Vec (N * (40 * 28 * 28))   := mbResidW N 28 28 w.b5 a4
    let a6  : Vec (N * (80 * 14 * 14))   := mbStridedW N 14 14 w.b6 a5
    let a7  : Vec (N * (80 * 14 * 14))   := mbResidW N 14 14 w.b7 a6
    let a8  : Vec (N * (80 * 14 * 14))   := mbResidW N 14 14 w.b8 a7
    let a9  : Vec (N * (112 * 14 * 14))  := mbExpW N 14 14 w.b9 a8
    let a10 : Vec (N * (112 * 14 * 14))  := mbResidW N 14 14 w.b10 a9
    let a11 : Vec (N * (112 * 14 * 14))  := mbResidW N 14 14 w.b11 a10
    let a12 : Vec (N * (192 * 7 * 7))    := mbStridedW N 7 7 w.b12 a11
    let a13 : Vec (N * (192 * 7 * 7))    := mbResidW N 7 7 w.b13 a12
    let a14 : Vec (N * (192 * 7 * 7))    := mbResidW N 7 7 w.b14 a13
    let a15 : Vec (N * (192 * 7 * 7))    := mbResidW N 7 7 w.b15 a14
    let a16 : Vec (N * (320 * 7 * 7))    := mbExpW N 7 7 w.b16 a15
    -- loss cotangent + backward block-output cotangents (composed top-down by the block VJPs)
    let g    : Vec (N * nCls) :=
      Proofs.BackLinks.unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (Proofs.BackLinks.rowB N nCls
          (headFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb a16)) t))
    let dy16 : Vec (N * (320 * 7 * 7))   := (headFwdBHasVJP N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb).backward a16 g
    let dy15 : Vec (N * (192 * 7 * 7))   := (mbExpWHasVJP N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p).backward a15 dy16
    let dy14 : Vec (N * (192 * 7 * 7))   := (mbResidWHasVJP N 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p).backward a14 dy15
    let dy13 : Vec (N * (192 * 7 * 7))   := (mbResidWHasVJP N 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p).backward a13 dy14
    let dy12 : Vec (N * (192 * 7 * 7))   := (mbResidWHasVJP N 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p).backward a12 dy13
    let dy11 : Vec (N * (112 * 14 * 14)) := (mbStridedWHasVJP N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p).backward a11 dy12
    let dy10 : Vec (N * (112 * 14 * 14)) := (mbResidWHasVJP N 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p).backward a10 dy11
    let dy9  : Vec (N * (112 * 14 * 14)) := (mbResidWHasVJP N 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p).backward a9 dy10
    let dy8  : Vec (N * (80 * 14 * 14))  := (mbExpWHasVJP N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p).backward a8 dy9
    let dy7  : Vec (N * (80 * 14 * 14))  := (mbResidWHasVJP N 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p).backward a7 dy8
    let dy6  : Vec (N * (80 * 14 * 14))  := (mbResidWHasVJP N 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p).backward a6 dy7
    let dy5  : Vec (N * (40 * 28 * 28))  := (mbStridedWHasVJP N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p).backward a5 dy6
    let dy4  : Vec (N * (40 * 28 * 28))  := (mbResidWHasVJP N 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p).backward a4 dy5
    let dy3  : Vec (N * (24 * 56 * 56))  := (mbStridedWHasVJP N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p).backward a3 dy4
    let dy2  : Vec (N * (24 * 56 * 56))  := (mbResidWHasVJP N 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p).backward a2 dy3
    let dy1  : Vec (N * (16 * 112 * 112)) := (mbStridedWHasVJP N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p).backward a1 dy2
    let dy0  : Vec (N * (32 * 112 * 112)) := (mbNoExpWHasVJP N 112 112 w.b1 hεw.b1.d hεw.b1.p).backward a0 dy1
    let L := smoothedBatchLoss N nCls α B t
    -- every block + stem + head tied at its real input + threaded output cotangent
    (enetStemTiedG xN vN epsStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x dy0
      ∧ enetStemLossTiedG (N := N) (h := 112) (w := 112) xN vN epsStr cotN w.sε hεw.s w.sW w.sb w.sγ w.sβ x
        (fun W b γ β => L (efficientnetForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x))
        dy0)
  ∧ (enetNoExpTiedGAt xN vN epsStr cotN 112 112 w.b1 hεw.b1.d hεw.b1.p a0 dy1
      ∧ enetNoExpLossTiedG (N := N) (h := 112) (w := 112) xN vN epsStr cotN w.b1 hεw.b1 a0
        (fun p => L (efficientnetForwardBFull N { w with b1 := p } x)) dy1)
  ∧ (enetStridedTiedGAt xN vN epsStr cotN 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p a1 dy2
      ∧ enetStridedLossTiedG (N := N) (h := 56) (w := 56) xN vN epsStr cotN w.b2 hεw.b2 a1
        (fun p => L (efficientnetForwardBFull N { w with b2 := p } x)) dy2)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 56 56 w.b3 hεw.b3.e hεw.b3.d hεw.b3.p a2 dy3
      ∧ enetExpLossTiedG (N := N) (h := 56) (w := 56) xN vN epsStr cotN w.b3 hεw.b3 a2
        (fun p => L (efficientnetForwardBFull N { w with b3 := p } x)) dy3)
  ∧ (enetStridedTiedGAt xN vN epsStr cotN 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p a3 dy4
      ∧ enetStridedLossTiedG (N := N) (h := 28) (w := 28) xN vN epsStr cotN w.b4 hεw.b4 a3
        (fun p => L (efficientnetForwardBFull N { w with b4 := p } x)) dy4)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 28 28 w.b5 hεw.b5.e hεw.b5.d hεw.b5.p a4 dy5
      ∧ enetExpLossTiedG (N := N) (h := 28) (w := 28) xN vN epsStr cotN w.b5 hεw.b5 a4
        (fun p => L (efficientnetForwardBFull N { w with b5 := p } x)) dy5)
  ∧ (enetStridedTiedGAt xN vN epsStr cotN 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p a5 dy6
      ∧ enetStridedLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b6 hεw.b6 a5
        (fun p => L (efficientnetForwardBFull N { w with b6 := p } x)) dy6)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 14 14 w.b7 hεw.b7.e hεw.b7.d hεw.b7.p a6 dy7
      ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b7 hεw.b7 a6
        (fun p => L (efficientnetForwardBFull N { w with b7 := p } x)) dy7)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 14 14 w.b8 hεw.b8.e hεw.b8.d hεw.b8.p a7 dy8
      ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b8 hεw.b8 a7
        (fun p => L (efficientnetForwardBFull N { w with b8 := p } x)) dy8)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p a8 dy9
      ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b9 hεw.b9 a8
        (fun p => L (efficientnetForwardBFull N { w with b9 := p } x)) dy9)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 14 14 w.b10 hεw.b10.e hεw.b10.d hεw.b10.p a9 dy10
      ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b10 hεw.b10 a9
        (fun p => L (efficientnetForwardBFull N { w with b10 := p } x)) dy10)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 14 14 w.b11 hεw.b11.e hεw.b11.d hεw.b11.p a10 dy11
      ∧ enetExpLossTiedG (N := N) (h := 14) (w := 14) xN vN epsStr cotN w.b11 hεw.b11 a10
        (fun p => L (efficientnetForwardBFull N { w with b11 := p } x)) dy11)
  ∧ (enetStridedTiedGAt xN vN epsStr cotN 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p a11 dy12
      ∧ enetStridedLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b12 hεw.b12 a11
        (fun p => L (efficientnetForwardBFull N { w with b12 := p } x)) dy12)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 7 7 w.b13 hεw.b13.e hεw.b13.d hεw.b13.p a12 dy13
      ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b13 hεw.b13 a12
        (fun p => L (efficientnetForwardBFull N { w with b13 := p } x)) dy13)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 7 7 w.b14 hεw.b14.e hεw.b14.d hεw.b14.p a13 dy14
      ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b14 hεw.b14 a13
        (fun p => L (efficientnetForwardBFull N { w with b14 := p } x)) dy14)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 7 7 w.b15 hεw.b15.e hεw.b15.d hεw.b15.p a14 dy15
      ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b15 hεw.b15 a14
        (fun p => L (efficientnetForwardBFull N { w with b15 := p } x)) dy15)
  ∧ (enetExpTiedGAt xN vN epsStr cotN 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p a15 dy16
      ∧ enetExpLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN w.b16 hεw.b16 a15
        (fun p => L (efficientnetForwardBFull N { w with b16 := p } x)) dy16)
  ∧ (enetHeadTiedG xN vN epsStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ w.fcW w.fcb a16 g
      ∧ enetHeadLossTiedG (N := N) (h := 7) (w := 7) xN vN epsStr cotN dN w.hε hεw.h w.hW w.hb w.hγ w.hβ
        w.fcW w.fcb a16
        (fun W b γ β Wd bd => L (efficientnetForwardBFull N
        { w with hW := W, hb := b, hγ := γ, hβ := β, fcW := Wd, fcb := bd } x)) g) := by
  intro a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 g dy16 dy15 dy14 dy13 dy12 dy11
    dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 dy0 L
  have htie :=
    efficientnet_net_tiedG xN vN epsStr cotN dN N w hεw aStr negAK bStr logN ohN α B x t
  extract_lets at htie
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17⟩ := htie
  have hl :=
    enet_net_lossGrad_smoothedCE xN vN epsStr cotN dN aStr negAK bStr logN ohN N hK α B w hεw x t ht
  -- the loss side's activations and logits, in the tie's spelling
  have e0 : enetPreB0 N w x = a0 := by rw [enetPreB0_apply N w x]
  have e1 : enetPreB1 N w x = a1 := by rw [enetPreB1_apply N w x, e0]
  have e2 : enetPreB2 N w x = a2 := by rw [enetPreB2_apply N w x, e1]
  have e3 : enetPreB3 N w x = a3 := by rw [enetPreB3_apply N w x, e2]
  have e4 : enetPreB4 N w x = a4 := by rw [enetPreB4_apply N w x, e3]
  have e5 : enetPreB5 N w x = a5 := by rw [enetPreB5_apply N w x, e4]
  have e6 : enetPreB6 N w x = a6 := by rw [enetPreB6_apply N w x, e5]
  have e7 : enetPreB7 N w x = a7 := by rw [enetPreB7_apply N w x, e6]
  have e8 : enetPreB8 N w x = a8 := by rw [enetPreB8_apply N w x, e7]
  have e9 : enetPreB9 N w x = a9 := by rw [enetPreB9_apply N w x, e8]
  have e10 : enetPreB10 N w x = a10 := by rw [enetPreB10_apply N w x, e9]
  have e11 : enetPreB11 N w x = a11 := by rw [enetPreB11_apply N w x, e10]
  have e12 : enetPreB12 N w x = a12 := by rw [enetPreB12_apply N w x, e11]
  have e13 : enetPreB13 N w x = a13 := by rw [enetPreB13_apply N w x, e12]
  have e14 : enetPreB14 N w x = a14 := by rw [enetPreB14_apply N w x, e13]
  have e15 : enetPreB15 N w x = a15 := by rw [enetPreB15_apply N w x, e14]
  have e16 : enetPreB16 N w x = a16 := by rw [enetPreB16_apply N w x, e15]
  have eg : unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
      (rowB N nCls (efficientnetForwardBFull N w x)) t)) = g := by
    rw [enet_forward_eq_head, e16]
  unfold EnetNetLossTiedG at hl
  rw [eg, e16, e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0] at hl
  extract_lets at hl
  -- the loss def's abstracted proof arguments keep these from merging; match them stepwise
  rename_i d_dy15 d_dy14 d_dy13 d_dy12 d_dy11 d_dy10 d_dy9 d_dy8 d_dy7 d_dy6 d_dy5 d_dy4 d_dy3
    d_dy2 d_dy1 d_dy0
  have q_dy15 : d_dy15 = dy15 := rfl
  have q_dy14 : d_dy14 = dy14 := by simp only [d_dy14, q_dy15]; rfl
  have q_dy13 : d_dy13 = dy13 := by simp only [d_dy13, q_dy14]; rfl
  have q_dy12 : d_dy12 = dy12 := by simp only [d_dy12, q_dy13]; rfl
  have q_dy11 : d_dy11 = dy11 := by simp only [d_dy11, q_dy12]; rfl
  have q_dy10 : d_dy10 = dy10 := by simp only [d_dy10, q_dy11]; rfl
  have q_dy9 : d_dy9 = dy9 := by simp only [d_dy9, q_dy10]; rfl
  have q_dy8 : d_dy8 = dy8 := by simp only [d_dy8, q_dy9]; rfl
  have q_dy7 : d_dy7 = dy7 := by simp only [d_dy7, q_dy8]; rfl
  have q_dy6 : d_dy6 = dy6 := by simp only [d_dy6, q_dy7]; rfl
  have q_dy5 : d_dy5 = dy5 := by simp only [d_dy5, q_dy6]; rfl
  have q_dy4 : d_dy4 = dy4 := by simp only [d_dy4, q_dy5]; rfl
  have q_dy3 : d_dy3 = dy3 := by simp only [d_dy3, q_dy4]; rfl
  have q_dy2 : d_dy2 = dy2 := by simp only [d_dy2, q_dy3]; rfl
  have q_dy1 : d_dy1 = dy1 := by simp only [d_dy1, q_dy2]; rfl
  have q_dy0 : d_dy0 = dy0 := by simp only [d_dy0, q_dy1]; rfl
  rw [q_dy0, q_dy1, q_dy2, q_dy3, q_dy4, q_dy5, q_dy6, q_dy7, q_dy8, q_dy9, q_dy10, q_dy11, q_dy12,
    q_dy13, q_dy14, q_dy15] at hl
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, l12⟩, ⟨t13, l13⟩, ⟨t14, l14⟩, ⟨t15, l15⟩, ⟨t16, l16⟩,
    ⟨t17, l17⟩⟩

end Proofs.EnetTieG
