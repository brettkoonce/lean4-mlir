import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Foundation.ParamGradNodes
import LeanMlir.Proofs.Training.BatchSealKit

/-! # ResNet-34 — every parameter gradient node IS the loss's derivative in that parameter

`r34_net_tiedB` says each of the 146 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it; the
`*CotIn_eq_vjp` lemmas say the block-input cotangents are certified VJP backwards; and
`r34_lossCot_is_smoothedCE_grad` identifies the loss cotangent row by row. `r34_net_lossGrad`
composes them: at the same cotangents, every node is `∂L/∂θ` of the WHOLE net with that one
parameter varied, `L` the batched label-smoothed cross-entropy (`smoothedBatchLoss`) the trainer
minimises.

**How.** Three layers, each generic where it can be:

* **Per node kind** (`ParamGradNodes`): a node is `∂G/∂θ` whenever its cotangent is the gradient of
  `G` — the loss read at that op's output — there (`HasGradAt`).
* **Per block kind** (this file, at variable widths): the loss read at each internal activation of
  an identity / downsample block, the stem and the head (`r34IdG*`, `r34DownG*`, `r34StemG*`), with
  its gradient the chain's own cotangent (`r34IdGC1_hasGradAt`, …), pulled back one certified stage at a
  time (`HasGradAt.comp`). The bundles `r34IdLossTiedB` / `r34DownLossTiedB` / `r34StemLossTiedB` /
  `r34HeadLossTiedB` state all of a block's nodes against `Φ`, the loss as a function of that
  block's weight record.
* **Per net**: the loss read after each block (`r34Suf*`), its gradient pulled back through the
  sixteen certified block VJPs (`r34IdB_hasGradAt_comp`, `r34DownB_hasGradAt_comp`), and `Φ` identified with the
  whole net at updated weights (`r34_factor_*`) — each a standalone `rfl`; inside the capstone the
  same identity is a kernel deep recursion at the literal widths.

**Hypotheses.** `R34PosB` (every BN `ε > 0`), `R34LossSmoothAtB` (every relu off its kink and every
stem-pool window dead, or its maximum at one position up to cells reading identical input
patches, at the real activations), every example's target summing to one, `0 < nCls`.

**The stem pool's clause is stated for the parameters, not the image.** Real batches have stem-pool
windows whose positive maximum sits at two positions, because two cells read identical input
patches (flat image regions). There the net has no derivative in the image, so the input VJP's
`R34SmoothAtB` rejects them, but the tied cells are the same function of the stem's weights, so the
loss IS differentiable in the parameters. `R34LossSmoothAtB` allows exactly those ties
(`StemPoolTwinAt` at `StemConvTwin`), and the stem's parameter nodes are proved through the argmax
gather they reduce to (`r34StemPool_param_germ`). The probe script
scripts/probes/stem_pool_smooth_probe.py checks the stem's clauses on real batches.
-/

open Proofs Proofs.StableHLO

namespace Proofs.ResNet34TieB

open Proofs.BackLinks (bnInB bnInB_eq_bnBackB reluMaskB cInB reassocB rowB unrowB)
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The identity block: the loss at each internal activation, and its gradient there
--   `Gn` is the loss read at the block's OUTPUT (the rest of the net, then the loss). Each
--   `r34IdG*` is the same loss read one stage further in; each `r34IdG*_hasGradAt` says its gradient
--   there is the emitted chain's cotangent.
-- ════════════════════════════════════════════════════════════════

section IdBlock
variable (N h w : Nat) {c : Nat}

/-- The loss at the outer relu's input `a`. -/
noncomputable def r34IdGA (Gn : Vec (N * (c * h * w)) → Vec 1) : Vec (N * (c * h * w)) → Vec 1 :=
  fun u => Gn (relu (N * (c * h * w)) u)

/-- The loss at bn₂'s output (the skip `v` held fixed). -/
noncomputable def r34IdGN2 (Gn : Vec (N * (c * h * w)) → Vec 1) (v : Vec (N * (c * h * w))) :
    Vec (N * (c * h * w)) → Vec 1 :=
  fun u => r34IdGA N h w Gn (fun i => u i + v i)

/-- The loss at conv₂'s output. -/
noncomputable def r34IdGC2 (Gn : Vec (N * (c * h * w)) → Vec 1) (p : R34IdW c)
    (v : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) → Vec 1 :=
  fun z => r34IdGN2 N h w Gn v (bnBatchLA N c h w p.ε₂ p.γ₂ p.β₂ z)

/-- The loss at bn₁'s output. -/
noncomputable def r34IdGN1 (Gn : Vec (N * (c * h * w)) → Vec 1) (p : R34IdW c)
    (v : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) → Vec 1 :=
  fun u => r34IdGC2 N h w Gn p v (batchMap N (flatConv p.W₂ p.b₂) (relu (N * (c * h * w)) u))

/-- The loss at conv₁'s output. -/
noncomputable def r34IdGC1 (Gn : Vec (N * (c * h * w)) → Vec 1) (p : R34IdW c)
    (v : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) → Vec 1 :=
  fun z => r34IdGN1 N h w Gn p v (bnBatchLA N c h w p.ε₁ p.γ₁ p.β₁ z)

variable {N h w}

theorem r34IdGA_hasGradAt (p : R34IdW c) (v : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p v)
    {Gn : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hGn : HasGradAt Gn (r34IdB N h w p v) dy) :
    HasGradAt (r34IdGA N h w Gn)
      (residual (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
        cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v)
      (r34IdCotA N h w p v dy) :=
  HasGradAt.comp (f := relu (N * (c * h * w)))
    (x := residual (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v)
    hGn (relu_differentiableAt_of_smooth _ _ hs.hout) (reluHasVJPAt _ _ hs.hout)

theorem r34IdGN2_hasGradAt (p : R34IdW c) (v : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p v)
    {Gn : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hGn : HasGradAt Gn (r34IdB N h w p v) dy) :
    HasGradAt (r34IdGN2 N h w Gn v)
      (bnBatchLA N c h w p.ε₂ p.γ₂ p.β₂ (batchMap N (flatConv p.W₂ p.b₂)
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)))
      (r34IdCotA N h w p v dy) :=
  (r34IdGA_hasGradAt p v hs hGn).comp (f := fun u i => u i + v i) (differentiableAt_id.add_const v)
    (addConstHasVJPAt (fun u => u) v _ differentiableAt_id (identityHasVJPAt _ _))

theorem r34IdGC2_hasGradAt (p : R34IdW c) (hq : R34IdPos p) (v : Vec (N * (c * h * w)))
    (hs : R34IdSmoothAt N h w p v) {Gn : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hGn : HasGradAt Gn (r34IdB N h w p v) dy) :
    HasGradAt (r34IdGC2 N h w Gn p v)
      (batchMap N (flatConv p.W₂ p.b₂) (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))
      (r34IdCotC2 N h w p v dy) :=
  ((r34IdGN2_hasGradAt p v hs hGn).comp ((bnBatchLA_differentiable N c h w p.ε₂ hq.h2 p.γ₂ p.β₂) _)
    ((bnBatchLAHasVJP N c h w p.ε₂ hq.h2 p.γ₂ p.β₂).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N c h w p.ε₂ hq.h2 p.γ₂ p.β₂ _ _).symm

theorem r34IdGN1_hasGradAt (p : R34IdW c) (hq : R34IdPos p) (v : Vec (N * (c * h * w)))
    (hs : R34IdSmoothAt N h w p v) {Gn : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hGn : HasGradAt Gn (r34IdB N h w p v) dy) :
    HasGradAt (r34IdGN1 N h w Gn p v)
      (bnBatchLA N c h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) v))
      (r34IdCotN1 N h w p v dy) := by
  have hc := ((r34IdGC2_hasGradAt p hq v hs hGn).comp
    ((batchMap_differentiable _ (flatConv_differentiable p.W₂ p.b₂)) _)
    ((batchMapHasVJP _ (flatConvHasVJP p.W₂ p.b₂) (flatConv_differentiable p.W₂ p.b₂)).toHasVJPAt
      _)).of_eq (GradNodeB.cInB_eq_batchMapBackward (h := h) (w := w) p.W₂ p.b₂ _ _).symm
  exact HasGradAt.comp (f := relu (N * (c * h * w)))
    (x := bnBatchLA N c h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConv p.W₁ p.b₁) v))
    hc (relu_differentiableAt_of_smooth _ _ hs.hmid) (reluHasVJPAt _ _ hs.hmid)

theorem r34IdGC1_hasGradAt (p : R34IdW c) (hq : R34IdPos p) (v : Vec (N * (c * h * w)))
    (hs : R34IdSmoothAt N h w p v) {Gn : Vec (N * (c * h * w)) → Vec 1}
    {dy : Vec (N * (c * h * w))} (hGn : HasGradAt Gn (r34IdB N h w p v) dy) :
    HasGradAt (r34IdGC1 N h w Gn p v) (batchMap N (flatConv p.W₁ p.b₁) v)
      (r34IdCotC1 N h w p v dy) :=
  ((r34IdGN1_hasGradAt p hq v hs hGn).comp ((bnBatchLA_differentiable N c h w p.ε₁ hq.h1 p.γ₁ p.β₁) _)
    ((bnBatchLAHasVJP N c h w p.ε₁ hq.h1 p.γ₁ p.β₁).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N c h w p.ε₁ hq.h1 p.γ₁ p.β₁ _ _).symm

/-- **Identity block, every parameter node a loss derivative.** With `Gn` the loss read at the
    block's output and `Φ` the loss as a function of the block's weight record (`hΦ`), each of the
    eight nodes `r34IdTiedB` ties — at the same cotangents — is `∂Φ/∂slot` with that one slot
    varied. -/
def r34IdLossTiedB (xN cotN vN epsStr : String) (p : R34IdW c) (v : Vec (N * (c * h * w)))
    (Φ : R34IdW c → Vec 1) (dy : Vec (N * (c * h * w))) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let c1 := batchMap N (flatConv p.W₁ p.b₁) v
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  (∀ idx, den (SHlo.convWeightGradB xN p.b₁ v p.W₁ (.operand cotN (r34IdCotC1 N h w p v dy))) idx
      = pdiv (fun θ => Φ { p with W₁ := Kernel4.unflatten θ })
          (Kernel4.flatten p.W₁) idx 0)
  ∧ (∀ o, den (SHlo.convBiasGradB (h := h) (w := w) p.W₁ v p.b₁
        (.operand cotN (r34IdCotC1 N h w p v dy))) o
      = pdiv (fun θ => Φ { p with b₁ := θ }) p.b₁ o 0)
  ∧ (∀ k, den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N c h w c1)
        (.operand cotN (reassocB N c h w (r34IdCotN1 N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with γ₁ := θ }) p.γ₁ k 0)
  ∧ (∀ k, den (SHlo.bnBetaGradB (N := N) (oc := c) (h := h) (w := w)
        (.operand cotN (reassocB N c h w (r34IdCotN1 N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with β₁ := θ }) p.β₁ k 0)
  ∧ (∀ idx, den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN (r34IdCotC2 N h w p v dy))) idx
      = pdiv (fun θ => Φ { p with W₂ := Kernel4.unflatten θ })
          (Kernel4.flatten p.W₂) idx 0)
  ∧ (∀ o, den (SHlo.convBiasGradB (h := h) (w := w) p.W₂ r1 p.b₂
        (.operand cotN (r34IdCotC2 N h w p v dy))) o
      = pdiv (fun θ => Φ { p with b₂ := θ }) p.b₂ o 0)
  ∧ (∀ k, den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N c h w c2)
        (.operand cotN (reassocB N c h w (r34IdCotA N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with γ₂ := θ }) p.γ₂ k 0)
  ∧ (∀ k, den (SHlo.bnBetaGradB (N := N) (oc := c) (h := h) (w := w)
        (.operand cotN (reassocB N c h w (r34IdCotA N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with β₂ := θ }) p.β₂ k 0)

theorem r34_idblock_lossTiedB (xN cotN vN epsStr : String) (p : R34IdW c) (hq : R34IdPos p)
    (v : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p v)
    {Gn : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hGn : HasGradAt Gn (r34IdB N h w p v) dy) {Φ : R34IdW c → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r34IdB N h w p' v)) :
    r34IdLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r34IdB N h w p' v) from funext hΦ]
  have hC1 := r34IdGC1_hasGradAt p hq v hs hGn
  have hN1 := r34IdGN1_hasGradAt p hq v hs hGn
  have hC2 := r34IdGC2_hasGradAt p hq v hs hGn
  have hN2 := r34IdGN2_hasGradAt p v hs hGn
  exact ⟨fun idx => GradNodeB.convW_eq_pdiv xN cotN p.b₁ v p.W₁ hC1 idx,
    fun o => GradNodeB.convB_eq_pdiv cotN p.W₁ v p.b₁ hC1 o,
    fun k => GradNodeB.bnGamma_eq_pdiv vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1 k,
    fun k => GradNodeB.bnBeta_eq_pdiv cotN p.ε₁ p.γ₁ p.β₁ _ hN1 k,
    fun idx => GradNodeB.convW_eq_pdiv xN cotN p.b₂ _ p.W₂ hC2 idx,
    fun o => GradNodeB.convB_eq_pdiv cotN p.W₂ _ p.b₂ hC2 o,
    fun k => GradNodeB.bnGamma_eq_pdiv vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2 k,
    fun k => GradNodeB.bnBeta_eq_pdiv cotN p.ε₂ p.γ₂ p.β₂ _ hN2 k⟩

end IdBlock

-- ════════════════════════════════════════════════════════════════
-- § The downsample block — the same chain with a projected skip
--   `residualProj proj body = proj + body`, so a body parameter sees the projection branch as a
--   constant on the LEFT and a projection parameter sees the body branch on the right.
-- ════════════════════════════════════════════════════════════════

section DownBlock
variable (N h w : Nat) {ic oc : Nat}

/-- The loss at the downsample block's pre-relu sum. -/
noncomputable def r34DownGA (Gn : Vec (N * (oc * h * w)) → Vec 1) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun u => Gn (relu (N * (oc * h * w)) u)

/-- The projection branch's output, `bnₚ(convₚ v)`. -/
@[reducible] noncomputable def r34DownProjOut (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) :=
  projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp v

/-- The body branch's output, `bn₂(conv₂(relu(bn₁(conv₁ v))))`. -/
@[reducible] noncomputable def r34DownBodyOut (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
    (cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)

/-- The loss at bn₂'s output (projection branch fixed, on the left). -/
noncomputable def r34DownGN2 (Gn : Vec (N * (oc * h * w)) → Vec 1) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun u => r34DownGA N h w Gn (fun i => r34DownProjOut N h w p v i + u i)

noncomputable def r34DownGC2 (Gn : Vec (N * (oc * h * w)) → Vec 1) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun z => r34DownGN2 N h w Gn p v (bnBatchLA N oc h w p.ε₂ p.γ₂ p.β₂ z)

noncomputable def r34DownGN1 (Gn : Vec (N * (oc * h * w)) → Vec 1) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun u => r34DownGC2 N h w Gn p v (batchMap N (flatConv p.W₂ p.b₂) (relu (N * (oc * h * w)) u))

noncomputable def r34DownGC1 (Gn : Vec (N * (oc * h * w)) → Vec 1) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun z => r34DownGN1 N h w Gn p v (bnBatchLA N oc h w p.ε₁ p.γ₁ p.β₁ z)

/-- The loss at bnₚ's output (body branch fixed, on the right). -/
noncomputable def r34DownGNp (Gn : Vec (N * (oc * h * w)) → Vec 1) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun u => r34DownGA N h w Gn (fun i => u i + r34DownBodyOut N h w p v i)

noncomputable def r34DownGCp (Gn : Vec (N * (oc * h * w)) → Vec 1) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (oc * h * w)) → Vec 1 :=
  fun z => r34DownGNp N h w Gn p v (bnBatchLA N oc h w p.εp p.γp p.βp z)

variable {N h w}

theorem r34DownGA_hasGradAt (p : R34DownW ic oc) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R34DownSmoothAt N h w p v) {Gn : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGA N h w Gn) (r34DownPre N h w p v) (r34DownCotA N h w p v dy) :=
  HasGradAt.comp (f := relu (N * (oc * h * w))) (x := r34DownPre N h w p v)
    hGn (relu_differentiableAt_of_smooth _ _ hs.hout) (reluHasVJPAt _ _ hs.hout)

theorem r34DownGN2_hasGradAt (p : R34DownW ic oc) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R34DownSmoothAt N h w p v) {Gn : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGN2 N h w Gn p v) (r34DownBodyOut N h w p v) (r34DownCotA N h w p v dy) :=
  HasGradAt.comp (f := fun u i => r34DownProjOut N h w p v i + u i) (x := r34DownBodyOut N h w p v)
    (r34DownGA_hasGradAt p v hs hGn) (differentiableAt_id.const_add _)
    (constAddHasVJPAt _ (fun u => u) _ differentiableAt_id (identityHasVJPAt _ _))

theorem r34DownGNp_hasGradAt (p : R34DownW ic oc) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R34DownSmoothAt N h w p v) {Gn : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGNp N h w Gn p v) (r34DownProjOut N h w p v) (r34DownCotA N h w p v dy) :=
  HasGradAt.comp (f := fun u i => u i + r34DownBodyOut N h w p v i) (x := r34DownProjOut N h w p v)
    (r34DownGA_hasGradAt p v hs hGn) (differentiableAt_id.add_const _)
    (addConstHasVJPAt (fun u => u) _ _ differentiableAt_id (identityHasVJPAt _ _))

theorem r34DownGC2_hasGradAt (p : R34DownW ic oc) (hq : R34DownPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGC2 N h w Gn p v)
      (batchMap N (flatConv p.W₂ p.b₂) (cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))
      (r34DownCotC2 N h w p v dy) :=
  (HasGradAt.comp (f := bnBatchLA N oc h w p.ε₂ p.γ₂ p.β₂)
    (x := batchMap N (flatConv p.W₂ p.b₂)
      (cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))
    (r34DownGN2_hasGradAt p v hs hGn) ((bnBatchLA_differentiable N oc h w p.ε₂ hq.h2 p.γ₂ p.β₂) _)
    ((bnBatchLAHasVJP N oc h w p.ε₂ hq.h2 p.γ₂ p.β₂).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N oc h w p.ε₂ hq.h2 p.γ₂ p.β₂ _ _).symm

theorem r34DownGN1_hasGradAt (p : R34DownW ic oc) (hq : R34DownPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGN1 N h w Gn p v)
      (bnBatchLA N oc h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConvStride2 p.W₁ p.b₁) v))
      (r34DownCotN1 N h w p v dy) := by
  have hc := ((r34DownGC2_hasGradAt p hq v hs hGn).comp
    ((batchMap_differentiable _ (flatConv_differentiable p.W₂ p.b₂)) _)
    ((batchMapHasVJP _ (flatConvHasVJP p.W₂ p.b₂) (flatConv_differentiable p.W₂ p.b₂)).toHasVJPAt
      _)).of_eq (GradNodeB.cInB_eq_batchMapBackward (h := h) (w := w) p.W₂ p.b₂ _ _).symm
  exact HasGradAt.comp (f := relu (N * (oc * h * w)))
    (x := bnBatchLA N oc h w p.ε₁ p.γ₁ p.β₁ (batchMap N (flatConvStride2 p.W₁ p.b₁) v))
    hc (relu_differentiableAt_of_smooth _ _ hs.hmid) (reluHasVJPAt _ _ hs.hmid)

theorem r34DownGC1_hasGradAt (p : R34DownW ic oc) (hq : R34DownPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGC1 N h w Gn p v) (batchMap N (flatConvStride2 p.W₁ p.b₁) v)
      (r34DownCotC1 N h w p v dy) :=
  ((r34DownGN1_hasGradAt p hq v hs hGn).comp ((bnBatchLA_differentiable N oc h w p.ε₁ hq.h1 p.γ₁ p.β₁) _)
    ((bnBatchLAHasVJP N oc h w p.ε₁ hq.h1 p.γ₁ p.β₁).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N oc h w p.ε₁ hq.h1 p.γ₁ p.β₁ _ _).symm

theorem r34DownGCp_hasGradAt (p : R34DownW ic oc) (hq : R34DownPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34DownB N h w p v) dy) :
    HasGradAt (r34DownGCp N h w Gn p v) (batchMap N (flatConvStride2 p.Wp p.bp) v)
      (r34DownCotCp N h w p v dy) :=
  (HasGradAt.comp (f := bnBatchLA N oc h w p.εp p.γp p.βp)
    (x := batchMap N (flatConvStride2 p.Wp p.bp) v)
    (r34DownGNp_hasGradAt p v hs hGn) ((bnBatchLA_differentiable N oc h w p.εp hq.hp p.γp p.βp) _)
    ((bnBatchLAHasVJP N oc h w p.εp hq.hp p.γp p.βp).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N oc h w p.εp hq.hp p.γp p.βp _ _).symm

/-- **Downsample block, every parameter node a loss derivative** — the twelve nodes
    `r34DownTiedB` ties. -/
def r34DownLossTiedB (xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (Φ : R34DownW ic oc → Vec 1)
    (dy : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let c1 := batchMap N (flatConvStride2 p.W₁ p.b₁) v
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let cp := batchMap N (flatConvStride2 p.Wp p.bp) v
  (∀ idx, den (SHlo.convStridedWeightGradB xN p.b₁ v p.W₁
        (.operand cotN (r34DownCotC1 N h w p v dy))) idx
      = pdiv (fun θ => Φ { p with W₁ := Kernel4.unflatten θ })
          (Kernel4.flatten p.W₁) idx 0)
  ∧ (∀ o, den (SHlo.convStridedBiasGradB (h := h) (w := w) p.W₁ v p.b₁
        (.operand cotN (r34DownCotC1 N h w p v dy))) o
      = pdiv (fun θ => Φ { p with b₁ := θ }) p.b₁ o 0)
  ∧ (∀ k, den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N oc h w c1)
        (.operand cotN (reassocB N oc h w (r34DownCotN1 N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with γ₁ := θ }) p.γ₁ k 0)
  ∧ (∀ k, den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
        (.operand cotN (reassocB N oc h w (r34DownCotN1 N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with β₁ := θ }) p.β₁ k 0)
  ∧ (∀ idx, den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂
        (.operand cotN (r34DownCotC2 N h w p v dy))) idx
      = pdiv (fun θ => Φ { p with W₂ := Kernel4.unflatten θ })
          (Kernel4.flatten p.W₂) idx 0)
  ∧ (∀ o, den (SHlo.convBiasGradB (h := h) (w := w) p.W₂ r1 p.b₂
        (.operand cotN (r34DownCotC2 N h w p v dy))) o
      = pdiv (fun θ => Φ { p with b₂ := θ }) p.b₂ o 0)
  ∧ (∀ k, den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N oc h w c2)
        (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with γ₂ := θ }) p.γ₂ k 0)
  ∧ (∀ k, den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
        (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with β₂ := θ }) p.β₂ k 0)
  ∧ (∀ idx, den (SHlo.convStridedWeightGradB xN p.bp v p.Wp
        (.operand cotN (r34DownCotCp N h w p v dy))) idx
      = pdiv (fun θ => Φ { p with Wp := Kernel4.unflatten θ })
          (Kernel4.flatten p.Wp) idx 0)
  ∧ (∀ o, den (SHlo.convStridedBiasGradB (h := h) (w := w) p.Wp v p.bp
        (.operand cotN (r34DownCotCp N h w p v dy))) o
      = pdiv (fun θ => Φ { p with bp := θ }) p.bp o 0)
  ∧ (∀ k, den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N oc h w cp)
        (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with γp := θ }) p.γp k 0)
  ∧ (∀ k, den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
        (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy)))) k
      = pdiv (fun θ => Φ { p with βp := θ }) p.βp k 0)

theorem r34_downblock_lossTiedB (xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (hq : R34DownPos p) (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34DownB N h w p v) dy) {Φ : R34DownW ic oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r34DownB N h w p' v)) :
    r34DownLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r34DownB N h w p' v) from funext hΦ]
  have hC1 := r34DownGC1_hasGradAt p hq v hs hGn
  have hN1 := r34DownGN1_hasGradAt p hq v hs hGn
  have hC2 := r34DownGC2_hasGradAt p hq v hs hGn
  have hN2 := r34DownGN2_hasGradAt p v hs hGn
  have hCp := r34DownGCp_hasGradAt p hq v hs hGn
  have hNp := r34DownGNp_hasGradAt p v hs hGn
  exact ⟨fun idx => GradNodeB.convStridedW_eq_pdiv xN cotN p.b₁ v p.W₁ hC1 idx,
    fun o => GradNodeB.convStridedB_eq_pdiv cotN p.W₁ v p.b₁ hC1 o,
    fun k => GradNodeB.bnGamma_eq_pdiv vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1 k,
    fun k => GradNodeB.bnBeta_eq_pdiv cotN p.ε₁ p.γ₁ p.β₁ _ hN1 k,
    fun idx => GradNodeB.convW_eq_pdiv xN cotN p.b₂ _ p.W₂ hC2 idx,
    fun o => GradNodeB.convB_eq_pdiv cotN p.W₂ _ p.b₂ hC2 o,
    fun k => GradNodeB.bnGamma_eq_pdiv vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2 k,
    fun k => GradNodeB.bnBeta_eq_pdiv cotN p.ε₂ p.γ₂ p.β₂ _ hN2 k,
    fun idx => GradNodeB.convStridedW_eq_pdiv xN cotN p.bp v p.Wp hCp idx,
    fun o => GradNodeB.convStridedB_eq_pdiv cotN p.Wp v p.bp hCp o,
    fun k => GradNodeB.bnGamma_eq_pdiv vN epsStr cotN p.εp p.γp p.βp _ hNp k,
    fun k => GradNodeB.bnBeta_eq_pdiv cotN p.εp p.γp p.βp _ hNp k⟩

end DownBlock

-- ════════════════════════════════════════════════════════════════
-- § The stem (through the 3×3/s2 pool) and the head
-- ════════════════════════════════════════════════════════════════

section StemHead
variable (N h w : Nat) {ic oc : Nat}

/-- **Two cells of example `r`'s stem grid read identical input patches**: every kernel and bias
    give them the same conv output, in every channel. A flat image region does this, and at such
    a pair the stem pool can tie at a positive maximum. The tie is then the same at every stem
    weight (`r34StemZ_twin`), which is why the parameter gradient survives it. -/
def StemConvTwin (N h w : Nat) {ic : Nat} (oc : Nat)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (r : Fin N)
    (p q : Fin (2 * h) × Fin (2 * w)) : Prop :=
  ∀ (W : Kernel4 oc ic 7 7) (b : Vec oc) (ci : Fin oc),
    batchMap N (flatConvStride2 W b) x (finProdFinEquiv (r, finProdFinEquiv (finProdFinEquiv (ci, p.1), p.2)))
      = batchMap N (flatConvStride2 W b) x (finProdFinEquiv (r, finProdFinEquiv (finProdFinEquiv (ci, q.1), q.2)))

/-- Twinned stem cells stay equal through batch BN, at every stem parameter. -/
theorem r34StemZ_twin (N h w : Nat) {ic oc : Nat} (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (εs : ℝ) (W : Kernel4 oc ic 7 7) (b γ β : Vec oc) (r : Fin N) (ci : Fin oc)
    (p q : Fin (2 * h) × Fin (2 * w)) (hT : StemConvTwin N h w oc x r p q) :
    bnBatchLA N oc (2 * h) (2 * w) εs γ β (batchMap N (flatConvStride2 W b) x)
        (finProdFinEquiv (r, finProdFinEquiv (finProdFinEquiv (ci, p.1), p.2)))
      = bnBatchLA N oc (2 * h) (2 * w) εs γ β (batchMap N (flatConvStride2 W b) x)
        (finProdFinEquiv (r, finProdFinEquiv (finProdFinEquiv (ci, q.1), q.2))) :=
  BatchSeal.bnBatchLA_bcell_eq_of_eq εs γ β _ r ci p.1 q.1 p.2 q.2 (hT W b ci)

variable {N h w}

/-- **Along any family of stem parameters, the pooled stem is the argmax gather.** The family
    passes through the stem's own parameters at `θ₀`; there the pre-ReLU has no zero entry and
    every pool window is smooth up to input-patch twins. Then the pooled stem agrees near `θ₀`
    with the gather, whose argmax is fixed at `θ₀`. -/
theorem r34StemPool_param_germ {P : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hstem : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpool : StemPoolTwinAt N h w (StemConvTwin N h w oc x)
      (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x))
    (Wf : Vec P → Kernel4 oc ic 7 7) (bf γf βf : Vec P → Vec oc) (θ₀ : Vec P)
    (h0 : bnBatchLA N oc (2 * h) (2 * w) εs (γf θ₀) (βf θ₀) (batchMap N (flatConvStride2 (Wf θ₀) (bf θ₀)) x)
      = bnBatchLA N oc (2 * h) (2 * w) εs γs βs (batchMap N (flatConvStride2 Ws bs) x))
    (hc : ContinuousAt (fun θ => bnBatchLA N oc (2 * h) (2 * w) εs (γf θ) (βf θ)
      (batchMap N (flatConvStride2 (Wf θ) (bf θ)) x)) θ₀) :
    (fun θ => r34StemB N h w (Wf θ) (bf θ) εs (γf θ) (βf θ) x) =ᶠ[nhds θ₀]
      (fun θ k => relu _ (bnBatchLA N oc (2 * h) (2 * w) εs (γf θ) (βf θ)
          (batchMap N (flatConvStride2 (Wf θ) (bf θ)) x))
        (maxPool3s2LocalReindexB N oc h w (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) k)) := by
  have hg := stemPoolRelu_param_eventuallyEq
    (fun θ => bnBatchLA N oc (2 * h) (2 * w) εs (γf θ) (βf θ) (batchMap N (flatConvStride2 (Wf θ) (bf θ)) x))
    θ₀ hc (by rw [h0]; exact hstem) (StemConvTwin N h w oc x) (by rw [h0]; exact hpool)
    (fun θ r ci p q hT => r34StemZ_twin N h w x εs (Wf θ) (bf θ) (γf θ) (βf θ) r ci p q hT)
  rw [h0] at hg
  exact hg

/-- The stem's loss as the gather model sees it, at the BN output: the pool replaced by the fixed
    argmax gather `σ`. -/
noncomputable def r34StemGNg (σ : Fin (N * (oc * h * w)) → Fin (N * (oc * (2 * h) * (2 * w))))
    (Gn : Vec (N * (oc * h * w)) → Vec 1) : Vec (N * (oc * (2 * h) * (2 * w))) → Vec 1 :=
  fun u => Gn (fun k => relu _ u (σ k))

/-- …and at the conv output. -/
noncomputable def r34StemGCg (σ : Fin (N * (oc * h * w)) → Fin (N * (oc * (2 * h) * (2 * w))))
    (Gn : Vec (N * (oc * h * w)) → Vec 1) (εs : ℝ) (γs βs : Vec oc) :
    Vec (N * (oc * (2 * h) * (2 * w))) → Vec 1 :=
  fun z => r34StemGNg σ Gn (bnBatchLA N oc (2 * h) (2 * w) εs γs βs z)

/-- **The gather model's gradients are the render's stem cotangents.** No pool hypothesis: the
    gather is linear after the ReLU. The loss's gradient at the pooled stem output transfers
    because at the stem's own parameters the pool IS the gather (`hpt`). -/
theorem r34StemGCg_hasGradAt (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs)
    (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hstem : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpt : r34StemB N h w Ws bs εs γs βs x = fun k => relu _
      (bnBatchLA N oc (2 * h) (2 * w) εs γs βs (batchMap N (flatConvStride2 Ws bs) x))
      (maxPool3s2LocalReindexB N oc h w (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) k))
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34StemB N h w Ws bs εs γs βs x) dy) :
    HasGradAt (r34StemGNg (maxPool3s2LocalReindexB N oc h w
          (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x)) Gn)
        (bnBatchLA N oc (2 * h) (2 * w) εs γs βs (batchMap N (flatConvStride2 Ws bs) x))
        (r34StemCotN N h w Ws bs εs γs βs x dy)
      ∧ HasGradAt (r34StemGCg (maxPool3s2LocalReindexB N oc h w
          (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x)) Gn εs γs βs)
        (batchMap N (flatConvStride2 Ws bs) x) (r34StemCotC N h w Ws bs εs γs βs x dy) := by
  have hN : HasGradAt (r34StemGNg (maxPool3s2LocalReindexB N oc h w
        (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x)) Gn)
      (bnBatchLA N oc (2 * h) (2 * w) εs γs βs (batchMap N (flatConvStride2 Ws bs) x))
      (r34StemCotN N h w Ws bs εs γs βs x dy) :=
    (HasGradAt.comp (x := bnBatchLA N oc (2 * h) (2 * w) εs γs βs (batchMap N (flatConvStride2 Ws bs) x))
      (hGn.congr_point hpt) (gatherRelu_differentiableAt _ _ hstem)
      (gatherReluHasVJPAt _ _ hstem)).of_eq
      ((gatherReluHasVJPAt_stem_backward N _ hstem dy).trans
        (congrArg _ ((den_maxPool3s2BackB_eq_flatBackB "" _ (.operand "" dy)).symm.trans rfl)))
  exact ⟨hN, (hN.comp ((bnBatchLA_differentiable N oc (2 * h) (2 * w) εs hεs γs βs) _)
    ((bnBatchLAHasVJP N oc (2 * h) (2 * w) εs hεs γs βs).toHasVJPAt _)).of_eq
    (bnInB_eq_bnBackB N oc (2 * h) (2 * w) εs hεs γs βs _ _).symm⟩

/-- **Stem, every parameter node a loss derivative** — the four nodes `r34StemTiedB` ties, `Φ` the
    loss as a function of the stem's `(W, b, γ, β)`. -/
def r34StemLossTiedB (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (Φ : Kernel4 oc ic 7 7 → Vec oc → Vec oc → Vec oc → Vec 1) (dy : Vec (N * (oc * h * w))) :
    Prop :=
  let sc := batchMap N (flatConvStride2 Ws bs) x
  (∀ idx, den (SHlo.convStridedWeightGradB xN bs x Ws
        (.operand cotN (r34StemCotC N h w Ws bs εs γs βs x dy))) idx
      = pdiv (fun θ => Φ (Kernel4.unflatten θ) bs γs βs) (Kernel4.flatten Ws) idx 0)
  ∧ (∀ o, den (SHlo.convStridedBiasGradB (h := 2 * h) (w := 2 * w) Ws x bs
        (.operand cotN (r34StemCotC N h w Ws bs εs γs βs x dy))) o
      = pdiv (fun θ => Φ Ws θ γs βs) bs o 0)
  ∧ (∀ k, den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc (2 * h) (2 * w) sc)
        (.operand cotN (reassocB N oc (2 * h) (2 * w) (r34StemCotN N h w Ws bs εs γs βs x dy)))) k
      = pdiv (fun θ => Φ Ws bs θ βs) γs k 0)
  ∧ (∀ k, den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := 2 * h) (w := 2 * w)
        (.operand cotN (reassocB N oc (2 * h) (2 * w) (r34StemCotN N h w Ws bs εs γs βs x dy)))) k
      = pdiv (fun θ => Φ Ws bs γs θ) βs k 0)

theorem r34_stem_lossTiedB (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hstem : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpool : StemPoolTwinAt N h w (StemConvTwin N h w oc x)
      (cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x))
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34StemB N h w Ws bs εs γs βs x) dy)
    {Φ : Kernel4 oc ic 7 7 → Vec oc → Vec oc → Vec oc → Vec 1}
    (hΦ : ∀ W b γ β, Φ W b γ β = Gn (r34StemB N h w W b εs γ β x)) :
    r34StemLossTiedB xN cotN vN epsStr Ws bs εs γs βs x Φ dy := by
  rw [show Φ = fun W b γ β => Gn (r34StemB N h w W b εs γ β x) from
    funext fun W => funext fun b => funext fun γ => funext fun β => hΦ W b γ β]
  have hbn := bnBatchLA_continuous N oc (2 * h) (2 * w) εs hεs
  -- the four one-slot parameter families, each through the stem's own parameters
  have gW := r34StemPool_param_germ Ws bs εs γs βs x hstem hpool
    Kernel4.unflatten (fun _ => bs) (fun _ => γs) (fun _ => βs) (Kernel4.flatten Ws)
    (by rw [Kernel4.unflatten_flatten])
    ((hbn γs βs).continuousAt.comp (batchMap_param_differentiableAt
      (fun θ y => flatConvStride2 (Kernel4.unflatten θ) bs y) x _
      (fun y => (GradNodeB.flatConvStride2_weight_differentiable bs y) _)).continuousAt)
  have gb := r34StemPool_param_germ Ws bs εs γs βs x hstem hpool
    (fun _ => Ws) id (fun _ => γs) (fun _ => βs) bs rfl
    ((hbn γs βs).continuousAt.comp (batchMap_param_differentiableAt
      (fun θ y => flatConvStride2 Ws θ y) x _
      (fun y => (GradNodeB.flatConvStride2_bias_differentiable Ws y) _)).continuousAt)
  have hγc : Continuous (fun θ : Vec oc => bnBatchLA N oc (2 * h) (2 * w) εs θ βs
      (batchMap N (flatConvStride2 Ws bs) x)) := by
    -- `bnBatchLA` is the per-channel core read at a permuted cell (`bnBatchLA_apply_perm`, `rfl`)
    exact continuous_pi fun J => (continuous_apply (GradNodeB.bnLAPerm N oc (2 * h) (2 * w) J)).comp
      (GradNodeB.bnPerChannelFlat_gamma_differentiable oc (N * (2 * h * (2 * w))) εs βs
        (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w)
          (batchMap N (flatConvStride2 Ws bs) x)))).continuous
  have hβc : Continuous (fun θ : Vec oc => bnBatchLA N oc (2 * h) (2 * w) εs γs θ
      (batchMap N (flatConvStride2 Ws bs) x)) := by
    -- `bnBatchLA` is the per-channel core read at a permuted cell (`bnBatchLA_apply_perm`, `rfl`)
    exact continuous_pi fun J => (continuous_apply (GradNodeB.bnLAPerm N oc (2 * h) (2 * w) J)).comp
      (GradNodeB.bnPerChannelFlat_beta_differentiable oc (N * (2 * h * (2 * w))) εs γs
        (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w)
          (batchMap N (flatConvStride2 Ws bs) x)))).continuous
  have gγ := r34StemPool_param_germ Ws bs εs γs βs x hstem hpool
    (fun _ => Ws) (fun _ => bs) id (fun _ => βs) γs rfl hγc.continuousAt
  have gβ := r34StemPool_param_germ Ws bs εs γs βs x hstem hpool
    (fun _ => Ws) (fun _ => bs) (fun _ => γs) id βs rfl hβc.continuousAt
  obtain ⟨hN, hC⟩ := r34StemGCg_hasGradAt Ws bs εs hεs γs βs x hstem gb.self_of_nhds hGn
  exact ⟨fun idx => (GradNodeB.convStridedW_eq_pdiv (h := 2 * h) (w := 2 * w) xN cotN bs x Ws hC
      idx).trans (pdiv_congr_of_eventuallyEq (gW.fun_comp Gn).symm idx 0),
    fun o => (GradNodeB.convStridedB_eq_pdiv (h := 2 * h) (w := 2 * w) cotN Ws x bs hC o).trans
      (pdiv_congr_of_eventuallyEq (gb.fun_comp Gn).symm o 0),
    fun k => (GradNodeB.bnGamma_eq_pdiv vN epsStr cotN εs γs βs _ hN k).trans
      (pdiv_congr_of_eventuallyEq (gγ.fun_comp Gn).symm k 0),
    fun k => (GradNodeB.bnBeta_eq_pdiv cotN εs γs βs _ hN k).trans
      (pdiv_congr_of_eventuallyEq (gβ.fun_comp Gn).symm k 0)⟩

/-- **Head, both parameter nodes loss derivatives** — the classifier weight and bias nodes
    `r34HeadTiedB` ties, `Φ` the loss as a function of `(Wd, bd)`. -/
def r34HeadLossTiedB {c nCls : Nat} (xN cotN : String) (Wd : Mat c nCls) (bd : Vec nCls)
    (v : Vec (N * (c * h * w))) (Φ : Mat c nCls → Vec nCls → Vec 1) (g : Vec (N * nCls)) : Prop :=
  (∀ i j, den (SHlo.denseWeightGradB (c := nCls) xN (batchMap N (globalAvgPoolFlat c h w) v)
        (.operand cotN g)) (finProdFinEquiv (i, j))
      = pdiv (fun θ => Φ (Mat.unflatten θ) bd) (Mat.flatten Wd) (finProdFinEquiv (i, j)) 0)
  ∧ (∀ j, den (SHlo.denseBiasGradB (N := N) (.operand cotN g)) j
      = pdiv (fun θ => Φ Wd θ) bd j 0)

theorem r34_head_lossTiedB {c nCls : Nat} (xN cotN : String) (Wd : Mat c nCls) (bd : Vec nCls)
    (v : Vec (N * (c * h * w))) {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (r34HeadB N h w Wd bd v) g) {Φ : Mat c nCls → Vec nCls → Vec 1}
    (hΦ : ∀ W b, Φ W b = L (r34HeadB N h w W b v)) :
    r34HeadLossTiedB xN cotN Wd bd v Φ g := by
  rw [show Φ = fun W b => L (r34HeadB N h w W b v) from funext fun W => funext fun b => hΦ W b]
  exact ⟨fun i j => GradNodeB.denseW_eq_pdiv xN cotN _ Wd bd hL i j,
    fun j => GradNodeB.denseB_eq_pdiv cotN Wd (fun _ => 0) _ bd hL j⟩

end StemHead

-- ════════════════════════════════════════════════════════════════
-- § The whole net: every parameter node IS the batched smoothed loss's derivative
-- ════════════════════════════════════════════════════════════════

-- ════════════════════════════════════════════════════════════════
-- § The net after each block, and the net with one block's weights varied
-- ════════════════════════════════════════════════════════════════

/-- The net after block `e1` — the head. -/
noncomputable def r34SufE1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (512 * 7 * 7)) → Vec (N * nCls) :=
  r34HeadB N 7 7 w.Wd w.bd

/-- The net after block `e0`: block `e1`, then the rest. -/
noncomputable def r34SufE0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (512 * 7 * 7)) → Vec (N * nCls) :=
  fun y => r34SufE1 N w (r34IdB N 7 7 w.e1 y)

/-- The net after block `d4`: block `e0`, then the rest. -/
noncomputable def r34SufD4 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (512 * 7 * 7)) → Vec (N * nCls) :=
  fun y => r34SufE0 N w (r34IdB N 7 7 w.e0 y)

/-- The net after block `c4`: block `d4`, then the rest. -/
noncomputable def r34SufC4 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (256 * 14 * 14)) → Vec (N * nCls) :=
  fun y => r34SufD4 N w (r34DownB N 7 7 w.d4 y)

/-- The net after block `c3`: block `c4`, then the rest. -/
noncomputable def r34SufC3 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (256 * 14 * 14)) → Vec (N * nCls) :=
  fun y => r34SufC4 N w (r34IdB N 14 14 w.c4 y)

/-- The net after block `c2`: block `c3`, then the rest. -/
noncomputable def r34SufC2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (256 * 14 * 14)) → Vec (N * nCls) :=
  fun y => r34SufC3 N w (r34IdB N 14 14 w.c3 y)

/-- The net after block `c1`: block `c2`, then the rest. -/
noncomputable def r34SufC1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (256 * 14 * 14)) → Vec (N * nCls) :=
  fun y => r34SufC2 N w (r34IdB N 14 14 w.c2 y)

/-- The net after block `c0`: block `c1`, then the rest. -/
noncomputable def r34SufC0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (256 * 14 * 14)) → Vec (N * nCls) :=
  fun y => r34SufC1 N w (r34IdB N 14 14 w.c1 y)

/-- The net after block `d3`: block `c0`, then the rest. -/
noncomputable def r34SufD3 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (256 * 14 * 14)) → Vec (N * nCls) :=
  fun y => r34SufC0 N w (r34IdB N 14 14 w.c0 y)

/-- The net after block `b2`: block `d3`, then the rest. -/
noncomputable def r34SufB2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (128 * 28 * 28)) → Vec (N * nCls) :=
  fun y => r34SufD3 N w (r34DownB N 14 14 w.d3 y)

/-- The net after block `b1`: block `b2`, then the rest. -/
noncomputable def r34SufB1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (128 * 28 * 28)) → Vec (N * nCls) :=
  fun y => r34SufB2 N w (r34IdB N 28 28 w.b2 y)

/-- The net after block `b0`: block `b1`, then the rest. -/
noncomputable def r34SufB0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (128 * 28 * 28)) → Vec (N * nCls) :=
  fun y => r34SufB1 N w (r34IdB N 28 28 w.b1 y)

/-- The net after block `d2`: block `b0`, then the rest. -/
noncomputable def r34SufD2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (128 * 28 * 28)) → Vec (N * nCls) :=
  fun y => r34SufB0 N w (r34IdB N 28 28 w.b0 y)

/-- The net after block `a2`: block `d2`, then the rest. -/
noncomputable def r34SufA2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (64 * 56 * 56)) → Vec (N * nCls) :=
  fun y => r34SufD2 N w (r34DownB N 28 28 w.d2 y)

/-- The net after block `a1`: block `a2`, then the rest. -/
noncomputable def r34SufA1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (64 * 56 * 56)) → Vec (N * nCls) :=
  fun y => r34SufA2 N w (r34IdB N 56 56 w.a2 y)

/-- The net after block `a0`: block `a1`, then the rest. -/
noncomputable def r34SufA0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (64 * 56 * 56)) → Vec (N * nCls) :=
  fun y => r34SufA1 N w (r34IdB N 56 56 w.a1 y)

/-- The net after the stem: block `a0`, then the rest. -/
noncomputable def r34SufStem (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (64 * 56 * 56)) → Vec (N * nCls) :=
  fun y => r34SufA0 N w (r34IdB N 56 56 w.a0 y)

/-- **The net with the stem's parameters varied** is the suffix after the stem at the varied stem. -/
theorem r34_factor_stem (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (W : Kernel4 64 3 7 7) (b γ β : Vec 64) :
    resnet34ForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x
      = r34SufStem N w (r34StemB N 56 56 W b w.sε γ β x) := rfl

/-- **The net with block `a0`'s weights varied** is the suffix after `a0` at the varied block. -/
theorem r34_factor_a0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 64) :
    resnet34ForwardBFull N { w with a0 := p } x
      = r34SufA0 N w (r34IdB N 56 56 p (r34Pre0 N w x)) := by
  rw [r34Pre0_apply]; rfl

/-- **The net with block `a1`'s weights varied** is the suffix after `a1` at the varied block. -/
theorem r34_factor_a1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 64) :
    resnet34ForwardBFull N { w with a1 := p } x
      = r34SufA1 N w (r34IdB N 56 56 p (r34Pre1 N w x)) := by
  rw [r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `a2`'s weights varied** is the suffix after `a2` at the varied block. -/
theorem r34_factor_a2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 64) :
    resnet34ForwardBFull N { w with a2 := p } x
      = r34SufA2 N w (r34IdB N 56 56 p (r34Pre2 N w x)) := by
  rw [r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `d2`'s weights varied** is the suffix after `d2` at the varied block. -/
theorem r34_factor_d2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34DownW 64 128) :
    resnet34ForwardBFull N { w with d2 := p } x
      = r34SufD2 N w (r34DownB N 28 28 p (r34Pre3 N w x)) := by
  rw [r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `b0`'s weights varied** is the suffix after `b0` at the varied block. -/
theorem r34_factor_b0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 128) :
    resnet34ForwardBFull N { w with b0 := p } x
      = r34SufB0 N w (r34IdB N 28 28 p (r34Pre4 N w x)) := by
  rw [r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `b1`'s weights varied** is the suffix after `b1` at the varied block. -/
theorem r34_factor_b1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 128) :
    resnet34ForwardBFull N { w with b1 := p } x
      = r34SufB1 N w (r34IdB N 28 28 p (r34Pre5 N w x)) := by
  rw [r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `b2`'s weights varied** is the suffix after `b2` at the varied block. -/
theorem r34_factor_b2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 128) :
    resnet34ForwardBFull N { w with b2 := p } x
      = r34SufB2 N w (r34IdB N 28 28 p (r34Pre6 N w x)) := by
  rw [r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `d3`'s weights varied** is the suffix after `d3` at the varied block. -/
theorem r34_factor_d3 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34DownW 128 256) :
    resnet34ForwardBFull N { w with d3 := p } x
      = r34SufD3 N w (r34DownB N 14 14 p (r34Pre7 N w x)) := by
  rw [r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `c0`'s weights varied** is the suffix after `c0` at the varied block. -/
theorem r34_factor_c0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 256) :
    resnet34ForwardBFull N { w with c0 := p } x
      = r34SufC0 N w (r34IdB N 14 14 p (r34Pre8 N w x)) := by
  rw [r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `c1`'s weights varied** is the suffix after `c1` at the varied block. -/
theorem r34_factor_c1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 256) :
    resnet34ForwardBFull N { w with c1 := p } x
      = r34SufC1 N w (r34IdB N 14 14 p (r34Pre9 N w x)) := by
  rw [r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `c2`'s weights varied** is the suffix after `c2` at the varied block. -/
theorem r34_factor_c2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 256) :
    resnet34ForwardBFull N { w with c2 := p } x
      = r34SufC2 N w (r34IdB N 14 14 p (r34Pre10 N w x)) := by
  rw [r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `c3`'s weights varied** is the suffix after `c3` at the varied block. -/
theorem r34_factor_c3 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 256) :
    resnet34ForwardBFull N { w with c3 := p } x
      = r34SufC3 N w (r34IdB N 14 14 p (r34Pre11 N w x)) := by
  rw [r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `c4`'s weights varied** is the suffix after `c4` at the varied block. -/
theorem r34_factor_c4 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 256) :
    resnet34ForwardBFull N { w with c4 := p } x
      = r34SufC4 N w (r34IdB N 14 14 p (r34Pre12 N w x)) := by
  rw [r34Pre12_apply, r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `d4`'s weights varied** is the suffix after `d4` at the varied block. -/
theorem r34_factor_d4 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34DownW 256 512) :
    resnet34ForwardBFull N { w with d4 := p } x
      = r34SufD4 N w (r34DownB N 7 7 p (r34Pre13 N w x)) := by
  rw [r34Pre13_apply, r34Pre12_apply, r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `e0`'s weights varied** is the suffix after `e0` at the varied block. -/
theorem r34_factor_e0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 512) :
    resnet34ForwardBFull N { w with e0 := p } x
      = r34SufE0 N w (r34IdB N 7 7 p (r34Pre14 N w x)) := by
  rw [r34Pre14_apply, r34Pre13_apply, r34Pre12_apply, r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with block `e1`'s weights varied** is the suffix after `e1` at the varied block. -/
theorem r34_factor_e1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (p : R34IdW 512) :
    resnet34ForwardBFull N { w with e1 := p } x
      = r34SufE1 N w (r34IdB N 7 7 p (r34Pre15 N w x)) := by
  rw [r34Pre15_apply, r34Pre14_apply, r34Pre13_apply, r34Pre12_apply, r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

/-- **The net with the classifier varied** is the head at the varied classifier. -/
theorem r34_factor_head (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (W : Mat 512 nCls) (b : Vec nCls) :
    resnet34ForwardBFull N { w with Wd := W, bd := b } x = r34HeadB N 7 7 W b (r34Pre16 N w x) := by
  rw [r34Pre16_apply, r34Pre15_apply, r34Pre14_apply, r34Pre13_apply, r34Pre12_apply, r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]; rfl

section Net

/-- **The smooth-point bundle a loss gradient needs** — `R34SmoothAtB` with the stem pool's clause
    weakened to allow ties between cells that read identical input patches (`StemPoolTwinAt` at
    `StemConvTwin`). Those ties exclude an input VJP (the net has no derivative in the image there)
    but not a parameter gradient (the tied cells are the same function of the stem's weights), and
    real batches have them. Every relu clause is `R34SmoothAtB`'s. -/
structure R34LossSmoothAtB (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) : Prop where
  stem : R34StemSmoothAt N 56 56 w.sW w.sb w.sε w.sγ w.sβ x
  pool : StemPoolTwinAt N 56 56 (StemConvTwin N 56 56 64 x)
    (cbReluStridedB N (h := 2 * 56) (w := 2 * 56) w.sW w.sb w.sε w.sγ w.sβ x)
  a0 : R34IdSmoothAt N 56 56 w.a0 (r34Pre0 N w x)
  a1 : R34IdSmoothAt N 56 56 w.a1 (r34Pre1 N w x)
  a2 : R34IdSmoothAt N 56 56 w.a2 (r34Pre2 N w x)
  d2 : R34DownSmoothAt N 28 28 w.d2 (r34Pre3 N w x)
  b0 : R34IdSmoothAt N 28 28 w.b0 (r34Pre4 N w x)
  b1 : R34IdSmoothAt N 28 28 w.b1 (r34Pre5 N w x)
  b2 : R34IdSmoothAt N 28 28 w.b2 (r34Pre6 N w x)
  d3 : R34DownSmoothAt N 14 14 w.d3 (r34Pre7 N w x)
  c0 : R34IdSmoothAt N 14 14 w.c0 (r34Pre8 N w x)
  c1 : R34IdSmoothAt N 14 14 w.c1 (r34Pre9 N w x)
  c2 : R34IdSmoothAt N 14 14 w.c2 (r34Pre10 N w x)
  c3 : R34IdSmoothAt N 14 14 w.c3 (r34Pre11 N w x)
  c4 : R34IdSmoothAt N 14 14 w.c4 (r34Pre12 N w x)
  d4 : R34DownSmoothAt N 7 7 w.d4 (r34Pre13 N w x)
  e0 : R34IdSmoothAt N 7 7 w.e0 (r34Pre14 N w x)
  e1 : R34IdSmoothAt N 7 7 w.e1 (r34Pre15 N w x)

/-- The input-VJP bundle implies the loss-gradient one. -/
theorem r34LossSmoothAtB_of_smoothAtB {N nCls : Nat} {w : R34BWeights nCls}
    {x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))} (hx : R34SmoothAtB N w x) :
    R34LossSmoothAtB N w x :=
  ⟨hx.stem, stemPoolTwinAt_of_smoothAt _ _ _ _ hx.pool, hx.a0, hx.a1, hx.a2, hx.d2, hx.b0, hx.b1, hx.b2, hx.d3, hx.c0, hx.c1, hx.c2, hx.c3, hx.c4, hx.d4, hx.e0, hx.e1⟩


/-- Pull the loss gradient back through an identity block: the certified block VJP, read at the
    chain's own fan-in (`r34IdCotIn_eq_vjp`). -/
theorem r34IdB_hasGradAt_comp {N h w c : Nat} (p : R34IdW c) (hq : R34IdPos p) (v : Vec (N * (c * h * w)))
    (hs : R34IdSmoothAt N h w p v) {G : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hG : HasGradAt G (r34IdB N h w p v) dy) :
    HasGradAt (fun y => G (r34IdB N h w p y)) v (r34IdCotIn N h w p v dy) :=
  (HasGradAt.comp (f := r34IdB N h w p) (x := v) hG
    ((StableHLO.r34BasicBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂).diff v ⟨⟨hs.hmid, trivial⟩, hs.hout⟩)
    (r34IdBHasVJPAt N h w p hq v hs)).of_eq (r34IdCotIn_eq_vjp N h w p hq v dy hs).symm

/-- …and through a downsample block (`r34DownCotIn_eq_vjp`). -/
theorem r34DownB_hasGradAt_comp {N h w ic oc : Nat} (p : R34DownW ic oc) (hq : R34DownPos p)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (r34DownB N h w p v) dy) :
    HasGradAt (fun y => G (r34DownB N h w p y)) v (r34DownCotIn N h w p v dy) :=
  (HasGradAt.comp (f := r34DownB N h w p) (x := v) hG
    ((StableHLO.r34DownBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.Wp p.bp p.εp hq.hp p.γp p.βp).diff v
      ⟨⟨trivial, hs.hmid, trivial⟩, hs.hout⟩)
    (r34DownBHasVJPAt N h w p hq v hs)).of_eq (r34DownCotIn_eq_vjp N h w p hq v dy hs).symm

/-- **Every ResNet-34 parameter gradient node is the derivative of the batched smoothed loss in
    that parameter.** `r34_net_tiedB` threads the label-smoothed cotangent `g` down the emitted
    backward chain and ties each of the 146 parameter nodes to its layer's Jacobian at the
    cotangent reaching it. Here each node, at that same cotangent, is `∂L/∂θ` of the WHOLE net —
    `L` the batched label-smoothed cross-entropy `smoothedBatchLoss` of `resnet34ForwardBFull`
    with that one parameter varied (a stem field, a block's weight record `w.blk := p` with one
    slot changed, or the classifier).

    Hypotheses: every BN `ε` positive (`R34PosB`), every relu off its kink and every stem-pool
    window dead or tied only between cells reading identical input patches, at the real
    activations (`R34LossSmoothAtB`), every example's target summing to one, and at least one
    class. -/
theorem r34_net_lossGrad (N : Nat) {nCls : Nat} (hK : 0 < nCls) (xN cotN vN epsStr : String)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R34BWeights nCls) (hq : R34PosB w)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34LossSmoothAtB N w x)
    (t : Vec (N * (1 * nCls))) (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    let L := smoothedBatchLoss N nCls α B t
    let g : Vec (N * nCls) :=
      unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (resnet34ForwardBFull N w x)) t))
    let dyE1 := r34HeadCotBlk N 7 7 w.Wd w.bd (r34Pre16 N w x) g
    let dyE0 := r34IdCotIn N 7 7 w.e1 (r34Pre15 N w x) dyE1
    let dyD4 := r34IdCotIn N 7 7 w.e0 (r34Pre14 N w x) dyE0
    let dyC4 := r34DownCotIn N 7 7 w.d4 (r34Pre13 N w x) dyD4
    let dyC3 := r34IdCotIn N 14 14 w.c4 (r34Pre12 N w x) dyC4
    let dyC2 := r34IdCotIn N 14 14 w.c3 (r34Pre11 N w x) dyC3
    let dyC1 := r34IdCotIn N 14 14 w.c2 (r34Pre10 N w x) dyC2
    let dyC0 := r34IdCotIn N 14 14 w.c1 (r34Pre9 N w x) dyC1
    let dyD3 := r34IdCotIn N 14 14 w.c0 (r34Pre8 N w x) dyC0
    let dyB2 := r34DownCotIn N 14 14 w.d3 (r34Pre7 N w x) dyD3
    let dyB1 := r34IdCotIn N 28 28 w.b2 (r34Pre6 N w x) dyB2
    let dyB0 := r34IdCotIn N 28 28 w.b1 (r34Pre5 N w x) dyB1
    let dyD2 := r34IdCotIn N 28 28 w.b0 (r34Pre4 N w x) dyB0
    let dyA2 := r34DownCotIn N 28 28 w.d2 (r34Pre3 N w x) dyD2
    let dyA1 := r34IdCotIn N 56 56 w.a2 (r34Pre2 N w x) dyA2
    let dyA0 := r34IdCotIn N 56 56 w.a1 (r34Pre1 N w x) dyA1
    let cotPool := r34IdCotIn N 56 56 w.a0 (r34Pre0 N w x) dyA0
    r34StemLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x
      (fun W b γ β => L (resnet34ForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x))
      cotPool
  ∧ r34IdLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.a0 (r34Pre0 N w x)
      (fun p => L (resnet34ForwardBFull N { w with a0 := p } x)) dyA0
  ∧ r34IdLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.a1 (r34Pre1 N w x)
      (fun p => L (resnet34ForwardBFull N { w with a1 := p } x)) dyA1
  ∧ r34IdLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.a2 (r34Pre2 N w x)
      (fun p => L (resnet34ForwardBFull N { w with a2 := p } x)) dyA2
  ∧ r34DownLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.d2 (r34Pre3 N w x)
      (fun p => L (resnet34ForwardBFull N { w with d2 := p } x)) dyD2
  ∧ r34IdLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b0 (r34Pre4 N w x)
      (fun p => L (resnet34ForwardBFull N { w with b0 := p } x)) dyB0
  ∧ r34IdLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b1 (r34Pre5 N w x)
      (fun p => L (resnet34ForwardBFull N { w with b1 := p } x)) dyB1
  ∧ r34IdLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b2 (r34Pre6 N w x)
      (fun p => L (resnet34ForwardBFull N { w with b2 := p } x)) dyB2
  ∧ r34DownLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.d3 (r34Pre7 N w x)
      (fun p => L (resnet34ForwardBFull N { w with d3 := p } x)) dyD3
  ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c0 (r34Pre8 N w x)
      (fun p => L (resnet34ForwardBFull N { w with c0 := p } x)) dyC0
  ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c1 (r34Pre9 N w x)
      (fun p => L (resnet34ForwardBFull N { w with c1 := p } x)) dyC1
  ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c2 (r34Pre10 N w x)
      (fun p => L (resnet34ForwardBFull N { w with c2 := p } x)) dyC2
  ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c3 (r34Pre11 N w x)
      (fun p => L (resnet34ForwardBFull N { w with c3 := p } x)) dyC3
  ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c4 (r34Pre12 N w x)
      (fun p => L (resnet34ForwardBFull N { w with c4 := p } x)) dyC4
  ∧ r34DownLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.d4 (r34Pre13 N w x)
      (fun p => L (resnet34ForwardBFull N { w with d4 := p } x)) dyD4
  ∧ r34IdLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.e0 (r34Pre14 N w x)
      (fun p => L (resnet34ForwardBFull N { w with e0 := p } x)) dyE0
  ∧ r34IdLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.e1 (r34Pre15 N w x)
      (fun p => L (resnet34ForwardBFull N { w with e1 := p } x)) dyE1
  ∧ r34HeadLossTiedB (N := N) (h := 7) (w := 7) xN cotN w.Wd w.bd (r34Pre16 N w x)
      (fun W b => L (resnet34ForwardBFull N { w with Wd := W, bd := b } x)) g := by
  intro L g dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0 cotPool
  -- the loss at the logits, then the head
  have hL : HasGradAt L (resnet34ForwardBFull N w x) g :=
    ⟨(smoothedBatchLoss_differentiable N nCls α B t) _,
      fun J => smoothedBatchLoss_grad N nCls hK α B aStr negAK bStr logN ohN t _ ht J⟩
  have hL' : HasGradAt L (r34HeadB N 7 7 w.Wd w.bd (r34Pre16 N w x)) g :=
    hL.congr_point (by rw [resnet34ForwardBFull_eq_chain, Function.comp_apply])
  have hE1 : HasGradAt (fun y => L (r34SufE1 N w y)) (r34Pre16 N w x) dyE1 :=
    HasGradAt.comp (f := r34HeadB N 7 7 w.Wd w.bd) (x := r34Pre16 N w x) hL'
      (((batchMap_differentiable _ (dense_differentiable w.Wd w.bd)).comp
        (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 512 7 7))) _)
      ((r34HeadBHasVJP N 7 7 w.Wd w.bd).toHasVJPAt _)
  have hE0 : HasGradAt (fun y => L (r34SufE0 N w y)) (r34Pre15 N w x) dyE0 :=
    r34IdB_hasGradAt_comp w.e1 hq.e1 _ hx.e1 (hE1.congr_point (r34Pre16_apply N w x))
  have hD4 : HasGradAt (fun y => L (r34SufD4 N w y)) (r34Pre14 N w x) dyD4 :=
    r34IdB_hasGradAt_comp w.e0 hq.e0 _ hx.e0 (hE0.congr_point (r34Pre15_apply N w x))
  have hC4 : HasGradAt (fun y => L (r34SufC4 N w y)) (r34Pre13 N w x) dyC4 :=
    r34DownB_hasGradAt_comp w.d4 hq.d4 _ hx.d4 (hD4.congr_point (r34Pre14_apply N w x))
  have hC3 : HasGradAt (fun y => L (r34SufC3 N w y)) (r34Pre12 N w x) dyC3 :=
    r34IdB_hasGradAt_comp w.c4 hq.c4 _ hx.c4 (hC4.congr_point (r34Pre13_apply N w x))
  have hC2 : HasGradAt (fun y => L (r34SufC2 N w y)) (r34Pre11 N w x) dyC2 :=
    r34IdB_hasGradAt_comp w.c3 hq.c3 _ hx.c3 (hC3.congr_point (r34Pre12_apply N w x))
  have hC1 : HasGradAt (fun y => L (r34SufC1 N w y)) (r34Pre10 N w x) dyC1 :=
    r34IdB_hasGradAt_comp w.c2 hq.c2 _ hx.c2 (hC2.congr_point (r34Pre11_apply N w x))
  have hC0 : HasGradAt (fun y => L (r34SufC0 N w y)) (r34Pre9 N w x) dyC0 :=
    r34IdB_hasGradAt_comp w.c1 hq.c1 _ hx.c1 (hC1.congr_point (r34Pre10_apply N w x))
  have hD3 : HasGradAt (fun y => L (r34SufD3 N w y)) (r34Pre8 N w x) dyD3 :=
    r34IdB_hasGradAt_comp w.c0 hq.c0 _ hx.c0 (hC0.congr_point (r34Pre9_apply N w x))
  have hB2 : HasGradAt (fun y => L (r34SufB2 N w y)) (r34Pre7 N w x) dyB2 :=
    r34DownB_hasGradAt_comp w.d3 hq.d3 _ hx.d3 (hD3.congr_point (r34Pre8_apply N w x))
  have hB1 : HasGradAt (fun y => L (r34SufB1 N w y)) (r34Pre6 N w x) dyB1 :=
    r34IdB_hasGradAt_comp w.b2 hq.b2 _ hx.b2 (hB2.congr_point (r34Pre7_apply N w x))
  have hB0 : HasGradAt (fun y => L (r34SufB0 N w y)) (r34Pre5 N w x) dyB0 :=
    r34IdB_hasGradAt_comp w.b1 hq.b1 _ hx.b1 (hB1.congr_point (r34Pre6_apply N w x))
  have hD2 : HasGradAt (fun y => L (r34SufD2 N w y)) (r34Pre4 N w x) dyD2 :=
    r34IdB_hasGradAt_comp w.b0 hq.b0 _ hx.b0 (hB0.congr_point (r34Pre5_apply N w x))
  have hA2 : HasGradAt (fun y => L (r34SufA2 N w y)) (r34Pre3 N w x) dyA2 :=
    r34DownB_hasGradAt_comp w.d2 hq.d2 _ hx.d2 (hD2.congr_point (r34Pre4_apply N w x))
  have hA1 : HasGradAt (fun y => L (r34SufA1 N w y)) (r34Pre2 N w x) dyA1 :=
    r34IdB_hasGradAt_comp w.a2 hq.a2 _ hx.a2 (hA2.congr_point (r34Pre3_apply N w x))
  have hA0 : HasGradAt (fun y => L (r34SufA0 N w y)) (r34Pre1 N w x) dyA0 :=
    r34IdB_hasGradAt_comp w.a1 hq.a1 _ hx.a1 (hA1.congr_point (r34Pre2_apply N w x))
  have hPool : HasGradAt (fun y => L (r34SufStem N w y)) (r34Pre0 N w x) cotPool :=
    r34IdB_hasGradAt_comp w.a0 hq.a0 _ hx.a0 (hA0.congr_point (r34Pre1_apply N w x))
  have hStem := hPool.congr_point (r34Pre0_apply N w x)
  refine ⟨r34_stem_lossTiedB xN cotN vN epsStr
      w.sW w.sb w.sε hq.s w.sγ w.sβ x hx.stem hx.pool hStem
      (fun W b γ β => by rw [r34_factor_stem]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.a0 hq.a0 _ hx.a0
      (hA0.congr_point (r34Pre1_apply N w x)) (fun p => by rw [r34_factor_a0]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.a1 hq.a1 _ hx.a1
      (hA1.congr_point (r34Pre2_apply N w x)) (fun p => by rw [r34_factor_a1]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.a2 hq.a2 _ hx.a2
      (hA2.congr_point (r34Pre3_apply N w x)) (fun p => by rw [r34_factor_a2]), ?_⟩
  refine ⟨r34_downblock_lossTiedB xN cotN vN epsStr w.d2 hq.d2 _ hx.d2
      (hD2.congr_point (r34Pre4_apply N w x)) (fun p => by rw [r34_factor_d2]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.b0 hq.b0 _ hx.b0
      (hB0.congr_point (r34Pre5_apply N w x)) (fun p => by rw [r34_factor_b0]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.b1 hq.b1 _ hx.b1
      (hB1.congr_point (r34Pre6_apply N w x)) (fun p => by rw [r34_factor_b1]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.b2 hq.b2 _ hx.b2
      (hB2.congr_point (r34Pre7_apply N w x)) (fun p => by rw [r34_factor_b2]), ?_⟩
  refine ⟨r34_downblock_lossTiedB xN cotN vN epsStr w.d3 hq.d3 _ hx.d3
      (hD3.congr_point (r34Pre8_apply N w x)) (fun p => by rw [r34_factor_d3]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.c0 hq.c0 _ hx.c0
      (hC0.congr_point (r34Pre9_apply N w x)) (fun p => by rw [r34_factor_c0]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.c1 hq.c1 _ hx.c1
      (hC1.congr_point (r34Pre10_apply N w x)) (fun p => by rw [r34_factor_c1]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.c2 hq.c2 _ hx.c2
      (hC2.congr_point (r34Pre11_apply N w x)) (fun p => by rw [r34_factor_c2]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.c3 hq.c3 _ hx.c3
      (hC3.congr_point (r34Pre12_apply N w x)) (fun p => by rw [r34_factor_c3]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.c4 hq.c4 _ hx.c4
      (hC4.congr_point (r34Pre13_apply N w x)) (fun p => by rw [r34_factor_c4]), ?_⟩
  refine ⟨r34_downblock_lossTiedB xN cotN vN epsStr w.d4 hq.d4 _ hx.d4
      (hD4.congr_point (r34Pre14_apply N w x)) (fun p => by rw [r34_factor_d4]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.e0 hq.e0 _ hx.e0
      (hE0.congr_point (r34Pre15_apply N w x)) (fun p => by rw [r34_factor_e0]), ?_⟩
  refine ⟨r34_idblock_lossTiedB xN cotN vN epsStr w.e1 hq.e1 _ hx.e1
      (hE1.congr_point (r34Pre16_apply N w x)) (fun p => by rw [r34_factor_e1]), ?_⟩
  exact r34_head_lossTiedB xN cotN w.Wd w.bd (r34Pre16 N w x) hL'
    (fun W b => by rw [r34_factor_head])

end Net

end Proofs.ResNet34TieB
