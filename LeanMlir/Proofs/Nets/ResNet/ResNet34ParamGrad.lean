import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Foundation.ParamGradNodes
import LeanMlir.Proofs.Training.BatchSealKit

/-! # ResNet-34 — every parameter gradient node IS the loss's derivative in that parameter

`r34_net_tiedB` says each of the 146 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it; the
`*CotIn_eq_vjp` lemmas say the block-input cotangents are certified VJP backwards; and
`r34_lossCot_is_smoothedCE_grad` identifies the loss cotangent row by row. `r34_net_lossGrad`
composes them: at the same cotangents, the loss of the WHOLE net with that one parameter varied is
differentiable in it and every node is its gradient (`HasGradAt`), for any loss `L` of the logits
with gradient `g` there; `r34_net_lossGrad_smoothedCE` instantiates it at the batched
label-smoothed cross-entropy (`smoothedBatchLoss`) the renders emit.

**Scope.** Every node here is an f32 `*GradB` node on one replica. The bf16 nodes (`*GradBBf16`)
that `resnet34in_momdp64bf16`, the book's ImageNet run, emits are outside this statement. Sync-BN
data parallelism is reached by composition: `r34_net_syncTiedB` says each all-reduced gradient is
this net's tied node at `N := R·N`, which this file's capstone makes the loss's gradient at the
global batch.

**How.** Three layers, each generic where it can be:

* **Per node kind** (`ParamGradNodes`): a node is `∂G/∂θ` whenever its cotangent is the gradient of
  `G` — the loss read at that op's output — there (`HasGradAt`).
* **Per block kind** (this file, at variable widths): the loss read at the block output, pulled
  back one certified stage at a time by `ParamGradNodes`' pull-backs (`hasGradAt_relu`,
  `hasGradAt_bnBatchLA`, `hasGradAt_conv`, the skip's `hasGradAt_addConst`), so each internal
  activation's gradient is the chain's own cotangent. The stem goes through the argmax gather
  instead (`r34StemGCg_hasGradAt`). The bundles `r34IdLossTiedB` / `r34DownLossTiedB` /
  `r34StemLossTiedB` / `r34HeadLossTiedB` state all of a block's nodes against `Φ`, the loss as a
  function of that block's weight record.
* **Per net**: the loss read after each block (`r34Suf*`), its gradient pulled back through the
  sixteen certified block VJPs (`r34IdB_hasGradAt_comp`, `r34DownB_hasGradAt_comp`), and `Φ` identified with the
  whole net at updated weights (`r34_factor_*`) — each a standalone `rfl`; inside the capstone the
  same identity is a kernel deep recursion at the literal widths.

**Hypotheses.** `R34PosB` (every BN `ε > 0`), `R34LossSmoothAtB` (every relu off its kink and every
stem-pool window dead, or its maximum at one position up to cells reading identical input
patches, at the real activations), and `L`'s gradient `g` at the logits; the smoothed-CE corollary
discharges that from every example's target summing to one and `0 < nCls`.

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
open Proofs.GradNodeB (hasGradAt_bnBatchLA hasGradAt_relu hasGradAt_conv hasGradAt_addConst
  hasGradAt_constAdd)
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The identity block
--   `Gn` is the loss read at the block's OUTPUT (the rest of the net, then the loss). The block's
--   proof pulls its gradient back one stage at a time through `ParamGradNodes`' pull-backs, each
--   landing on the emitted chain's own cotangent.
-- ════════════════════════════════════════════════════════════════

section IdBlock
variable {N h w c : Nat}

/-- **Identity block, every parameter node a loss derivative.** With `Gn` the loss read at the
    block's output and `Φ` the loss as a function of the block's weight record (`hΦ`), each of the
    eight nodes `r34IdTiedB` ties — at the same cotangents — is `∂Φ/∂slot` with that one slot
    varied. -/
def r34IdLossTiedB (xN cotN vN epsStr : String) (p : R34IdW c) (v : Vec (N * (c * h * w)))
    (Φ : R34IdW c → Vec 1) (dy : Vec (N * (c * h * w))) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let c1 := batchMap N (flatConv p.W₁ p.b₁) v
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  (HasGradAt (fun θ => Φ { p with W₁ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₁)
        (den (SHlo.convWeightGradB xN p.b₁ v p.W₁ (.operand cotN (r34IdCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with b₁ := θ }) p.b₁
        (den (SHlo.convBiasGradB (h := h) (w := w) p.W₁ v p.b₁
          (.operand cotN (r34IdCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₁ := θ }) p.γ₁
        (den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N c h w c1)
          (.operand cotN (reassocB N c h w (r34IdCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₁ := θ }) p.β₁
        (den (SHlo.bnBetaGradB (N := N) (oc := c) (h := h) (w := w)
          (.operand cotN (reassocB N c h w (r34IdCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₂ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₂)
        (den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN (r34IdCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with b₂ := θ }) p.b₂
        (den (SHlo.convBiasGradB (h := h) (w := w) p.W₂ r1 p.b₂
          (.operand cotN (r34IdCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₂ := θ }) p.γ₂
        (den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N c h w c2)
          (.operand cotN (reassocB N c h w (r34IdCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₂ := θ }) p.β₂
        (den (SHlo.bnBetaGradB (N := N) (oc := c) (h := h) (w := w)
          (.operand cotN (reassocB N c h w (r34IdCotA N h w p v dy))))))

theorem r34_idblock_lossTiedB (xN cotN vN epsStr : String) (p : R34IdW c) (hq : R34IdPos p)
    (v : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p v)
    {Gn : Vec (N * (c * h * w)) → Vec 1} {dy : Vec (N * (c * h * w))}
    (hGn : HasGradAt Gn (r34IdB N h w p v) dy) {Φ : R34IdW c → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r34IdB N h w p' v)) :
    r34IdLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r34IdB N h w p' v) from funext hΦ]
  have hN2 := hasGradAt_addConst (bnBatchLA N c h w p.ε₂ p.γ₂ p.β₂ (batchMap N (flatConv p.W₂ p.b₂)
    (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))) v (hasGradAt_relu _ hs.hout hGn)
  have hC2 := hasGradAt_bnBatchLA p.ε₂ hq.h2 p.γ₂ p.β₂ _ hN2
  have hN1 := hasGradAt_relu _ hs.hmid (hasGradAt_conv p.W₂ p.b₂ _ hC2)
  have hC1 := hasGradAt_bnBatchLA p.ε₁ hq.h1 p.γ₁ p.β₁ _ hN1
  exact ⟨GradNodeB.convW_hasGradAt xN cotN p.b₁ v p.W₁ hC1,
    GradNodeB.convB_hasGradAt cotN p.W₁ v p.b₁ hC1,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.convW_hasGradAt xN cotN p.b₂ _ p.W₂ hC2,
    GradNodeB.convB_hasGradAt cotN p.W₂ _ p.b₂ hC2,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₂ p.γ₂ p.β₂ _ hN2⟩

end IdBlock

-- ════════════════════════════════════════════════════════════════
-- § The downsample block — the same chain with a projected skip
--   `residualProj proj body = proj + body`, so a body parameter sees the projection branch as a
--   constant on the LEFT and a projection parameter sees the body branch on the right.
-- ════════════════════════════════════════════════════════════════

section DownBlock
variable {N h w ic oc : Nat}

/-- **Downsample block, every parameter node a loss derivative** — the twelve nodes
    `r34DownTiedB` ties. -/
def r34DownLossTiedB (xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (Φ : R34DownW ic oc → Vec 1)
    (dy : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let c1 := batchMap N (flatConvStride2 p.W₁ p.b₁) v
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let cp := batchMap N (flatConvStride2 p.Wp p.bp) v
  (HasGradAt (fun θ => Φ { p with W₁ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₁)
        (den (SHlo.convStridedWeightGradB xN p.b₁ v p.W₁
          (.operand cotN (r34DownCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with b₁ := θ }) p.b₁
        (den (SHlo.convStridedBiasGradB (h := h) (w := w) p.W₁ v p.b₁
          (.operand cotN (r34DownCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₁ := θ }) p.γ₁
        (den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N oc h w c1)
          (.operand cotN (reassocB N oc h w (r34DownCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₁ := θ }) p.β₁
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r34DownCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₂ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₂)
        (den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂
          (.operand cotN (r34DownCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with b₂ := θ }) p.b₂
        (den (SHlo.convBiasGradB (h := h) (w := w) p.W₂ r1 p.b₂
          (.operand cotN (r34DownCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₂ := θ }) p.γ₂
        (den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N oc h w c2)
          (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₂ := θ }) p.β₂
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wp := Kernel4.unflatten θ }) (Kernel4.flatten p.Wp)
        (den (SHlo.convStridedWeightGradB xN p.bp v p.Wp
          (.operand cotN (r34DownCotCp N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with bp := θ }) p.bp
        (den (SHlo.convStridedBiasGradB (h := h) (w := w) p.Wp v p.bp
          (.operand cotN (r34DownCotCp N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γp := θ }) p.γp
        (den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N oc h w cp)
          (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with βp := θ }) p.βp
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r34DownCotA N h w p v dy))))))

theorem r34_downblock_lossTiedB (xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (hq : R34DownPos p) (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R34DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r34DownB N h w p v) dy) {Φ : R34DownW ic oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r34DownB N h w p' v)) :
    r34DownLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r34DownB N h w p' v) from funext hΦ]
  have hA := hasGradAt_relu (r34DownPre N h w p v) hs.hout hGn
  -- the body sees the projection branch as a constant on the left, the projection the body's on
  -- the right (`residualProj proj body = proj + body`)
  have hN2 := hasGradAt_constAdd (projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp v)
    (bnBatchLA N oc h w p.ε₂ p.γ₂ p.β₂ (batchMap N (flatConv p.W₂ p.b₂)
      (cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))) hA
  have hNp := hasGradAt_addConst
    (bnBatchLA N oc h w p.εp p.γp p.βp (batchMap N (flatConvStride2 p.Wp p.bp) v))
    (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
      (cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) hA
  have hC2 := hasGradAt_bnBatchLA p.ε₂ hq.h2 p.γ₂ p.β₂ _ hN2
  have hN1 := hasGradAt_relu _ hs.hmid (hasGradAt_conv p.W₂ p.b₂ _ hC2)
  have hC1 := hasGradAt_bnBatchLA p.ε₁ hq.h1 p.γ₁ p.β₁ _ hN1
  have hCp := hasGradAt_bnBatchLA p.εp hq.hp p.γp p.βp _ hNp
  exact ⟨GradNodeB.convStridedW_hasGradAt xN cotN p.b₁ v p.W₁ hC1,
    GradNodeB.convStridedB_hasGradAt cotN p.W₁ v p.b₁ hC1,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.convW_hasGradAt xN cotN p.b₂ _ p.W₂ hC2,
    GradNodeB.convB_hasGradAt cotN p.W₂ _ p.b₂ hC2,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.convStridedW_hasGradAt xN cotN p.bp v p.Wp hCp,
    GradNodeB.convStridedB_hasGradAt cotN p.Wp v p.bp hCp,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.εp p.γp p.βp _ hNp,
    GradNodeB.bnBeta_hasGradAt cotN p.εp p.γp p.βp _ hNp⟩

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
  (HasGradAt (fun θ => Φ (Kernel4.unflatten θ) bs γs βs) (Kernel4.flatten Ws)
        (den (SHlo.convStridedWeightGradB xN bs x Ws
          (.operand cotN (r34StemCotC N h w Ws bs εs γs βs x dy)))))
  ∧ (HasGradAt (fun θ => Φ Ws θ γs βs) bs
        (den (SHlo.convStridedBiasGradB (h := 2 * h) (w := 2 * w) Ws x bs
          (.operand cotN (r34StemCotC N h w Ws bs εs γs βs x dy)))))
  ∧ (HasGradAt (fun θ => Φ Ws bs θ βs) γs
        (den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc (2 * h) (2 * w) sc)
          (.operand cotN (reassocB N oc (2 * h) (2 * w) (r34StemCotN N h w Ws bs εs γs βs x dy))))))
  ∧ (HasGradAt (fun θ => Φ Ws bs γs θ) βs
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := 2 * h) (w := 2 * w)
          (.operand cotN (reassocB N oc (2 * h) (2 * w) (r34StemCotN N h w Ws bs εs γs βs x dy))))))

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
    -- `bnBatchLA` is the per-channel core read at the `bnLAPerm` cell, by `rfl`
    exact continuous_pi fun J => (continuous_apply (GradNodeB.bnLAPerm N oc (2 * h) (2 * w) J)).comp
      (GradNodeB.bnPerChannelFlat_gamma_differentiable oc (N * (2 * h * (2 * w))) εs βs
        (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w)
          (batchMap N (flatConvStride2 Ws bs) x)))).continuous
  have hβc : Continuous (fun θ : Vec oc => bnBatchLA N oc (2 * h) (2 * w) εs γs θ
      (batchMap N (flatConvStride2 Ws bs) x)) := by
    -- `bnBatchLA` is the per-channel core read at the `bnLAPerm` cell, by `rfl`
    exact continuous_pi fun J => (continuous_apply (GradNodeB.bnLAPerm N oc (2 * h) (2 * w) J)).comp
      (GradNodeB.bnPerChannelFlat_beta_differentiable oc (N * (2 * h * (2 * w))) εs γs
        (bnchwFwd N oc (2 * h) (2 * w) (reassocB N oc (2 * h) (2 * w)
          (batchMap N (flatConvStride2 Ws bs) x)))).continuous
  have gγ := r34StemPool_param_germ Ws bs εs γs βs x hstem hpool
    (fun _ => Ws) (fun _ => bs) id (fun _ => βs) γs rfl hγc.continuousAt
  have gβ := r34StemPool_param_germ Ws bs εs γs βs x hstem hpool
    (fun _ => Ws) (fun _ => bs) (fun _ => γs) id βs rfl hβc.continuousAt
  obtain ⟨hN, hC⟩ := r34StemGCg_hasGradAt Ws bs εs hεs γs βs x hstem gb.self_of_nhds hGn
  exact ⟨(GradNodeB.convStridedW_hasGradAt (h := 2 * h) (w := 2 * w) xN cotN bs x Ws
      hC).congr_of_eventuallyEq (gW.fun_comp Gn).symm,
    (GradNodeB.convStridedB_hasGradAt (h := 2 * h) (w := 2 * w) cotN Ws x bs
      hC).congr_of_eventuallyEq (gb.fun_comp Gn).symm,
    (GradNodeB.bnGamma_hasGradAt vN epsStr cotN εs γs βs _ hN).congr_of_eventuallyEq
      (gγ.fun_comp Gn).symm,
    (GradNodeB.bnBeta_hasGradAt cotN εs γs βs _ hN).congr_of_eventuallyEq
      (gβ.fun_comp Gn).symm⟩

/-- **Head, both parameter nodes loss derivatives** — the classifier weight and bias nodes
    `r34HeadTiedB` ties, `Φ` the loss as a function of `(Wd, bd)`. -/
def r34HeadLossTiedB {c nCls : Nat} (xN cotN : String) (Wd : Mat c nCls) (bd : Vec nCls)
    (v : Vec (N * (c * h * w))) (Φ : Mat c nCls → Vec nCls → Vec 1) (g : Vec (N * nCls)) : Prop :=
  (HasGradAt (fun θ => Φ (Mat.unflatten θ) bd) (Mat.flatten Wd)
        (den (SHlo.denseWeightGradB (c := nCls) xN (batchMap N (globalAvgPoolFlat c h w) v)
          (.operand cotN g))))
  ∧ (HasGradAt (fun θ => Φ Wd θ) bd
        (den (SHlo.denseBiasGradB (N := N) (.operand cotN g))))

theorem r34_head_lossTiedB {c nCls : Nat} (xN cotN : String) (Wd : Mat c nCls) (bd : Vec nCls)
    (v : Vec (N * (c * h * w))) {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (r34HeadB N h w Wd bd v) g) {Φ : Mat c nCls → Vec nCls → Vec 1}
    (hΦ : ∀ W b, Φ W b = L (r34HeadB N h w W b v)) :
    r34HeadLossTiedB xN cotN Wd bd v Φ g := by
  rw [show Φ = fun W b => L (r34HeadB N h w W b v) from funext fun W => funext fun b => hΦ W b]
  exact ⟨GradNodeB.denseW_hasGradAt xN cotN _ Wd bd hL,
    GradNodeB.denseB_hasGradAt cotN Wd (fun _ => 0) _ bd hL⟩

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

/-- **Every ResNet-34 parameter gradient node is the gradient of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 146 nodes
    `r34_net_tiedB` ties, each at the cotangent the emitted chain threads to it, stated against `L`
    of `resnet34ForwardBFull` with that one parameter varied (a stem field, a block's weight record
    `w.blk := p` with one slot changed, or the classifier). `r34_net_lossGrad` proves it whenever
    `g` is `L`'s gradient at the logits; `r34_net_lossGrad_smoothedCE` instantiates it at the loss
    the artifacts ship. -/
def R34NetLossTiedB (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String) (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (L : Vec (N * nCls) → Vec 1)
    (g : Vec (N * nCls)) : Prop :=
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
      (fun W b => L (resnet34ForwardBFull N { w with Wd := W, bd := b } x)) g

/-- **Every ResNet-34 parameter gradient node is the gradient of `L` in that parameter**, whenever
    `g` is `L`'s gradient at the logits.

    Hypotheses: every BN `ε` positive (`R34PosB`), every relu off its kink and every stem-pool
    window dead or tied only between cells reading identical input patches, at the real
    activations (`R34LossSmoothAtB`), and `hL`.
    The nodes are the f32 ones on one replica (the module's Scope). -/
theorem r34_net_lossGrad (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : R34BWeights nCls) (hq : R34PosB w)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34LossSmoothAtB N w x)
    {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (resnet34ForwardBFull N w x) g) :
    R34NetLossTiedB N xN cotN vN epsStr w x L g := by
  unfold R34NetLossTiedB
  intro dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0 cotPool
  -- the head
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

/-- **The artifacts' loss**: every node is the gradient of the batched label-smoothed
    cross-entropy `smoothedBatchLoss`, `g` the emitted loss cotangent, given every example's target
    summing to one and at least one class. -/
theorem r34_net_lossGrad_smoothedCE (N : Nat) {nCls : Nat} (hK : 0 < nCls)
    (xN cotN vN epsStr aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R34BWeights nCls)
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (hx : R34LossSmoothAtB N w x) (t : Vec (N * (1 * nCls)))
    (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    R34NetLossTiedB N xN cotN vN epsStr w x (smoothedBatchLoss N nCls α B t)
      (unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (resnet34ForwardBFull N w x)) t))) :=
  r34_net_lossGrad N xN cotN vN epsStr w hq x hx
    ⟨(smoothedBatchLoss_differentiable N nCls α B t) _,
      fun J => smoothedBatchLoss_grad N nCls hK α B aStr negAK bStr logN ohN t _ ht J⟩

/-- **The emitted ResNet-34 step's gradient nodes ARE the loss's gradient, at one chain.** For each
    of the 146 parameter slots, at ONE cotangent chain (the tie's own, from the emitted
    smoothed-loss cotangent `g`): the node denotes its layer's Jacobian against the chain cotangent
    (`r34_net_tiedB`), and the batched smoothed loss of `resnet34ForwardBFull` with that one slot
    varied is differentiable there with the node as its gradient (`r34_net_lossGrad_smoothedCE`).
    The two theorems each state the chain; this one states it once, so an edit to either chain
    breaks its proof. -/
theorem r34_net_tied_lossGrad (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (t : Vec (N * (1 * nCls)))
    (hK : 0 < nCls) (hq : R34PosB w) (hx : R34LossSmoothAtB N w x)
    (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    -- the label-smoothed loss cotangent at the real logits and a general target
    let g : Vec (N * nCls) :=
      unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (resnet34ForwardBFull N w x)) t))
    -- the backward chain: the certified head backward, then the certified block backwards
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
    let L := smoothedBatchLoss N nCls α B t
    (r34StemTiedB N 56 56 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool
      ∧ r34StemLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x
        (fun W b γ β => L (resnet34ForwardBFull N { w with sW := W, sb := b, sγ := γ, sβ := β } x))
        cotPool)
  ∧ (r34IdTiedB N 56 56 xN cotN vN epsStr w.a0 (r34Pre0 N w x) dyA0
      ∧ r34IdLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.a0 (r34Pre0 N w x)
        (fun p => L (resnet34ForwardBFull N { w with a0 := p } x)) dyA0)
  ∧ (r34IdTiedB N 56 56 xN cotN vN epsStr w.a1 (r34Pre1 N w x) dyA1
      ∧ r34IdLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.a1 (r34Pre1 N w x)
        (fun p => L (resnet34ForwardBFull N { w with a1 := p } x)) dyA1)
  ∧ (r34IdTiedB N 56 56 xN cotN vN epsStr w.a2 (r34Pre2 N w x) dyA2
      ∧ r34IdLossTiedB (N := N) (h := 56) (w := 56) xN cotN vN epsStr w.a2 (r34Pre2 N w x)
        (fun p => L (resnet34ForwardBFull N { w with a2 := p } x)) dyA2)
  ∧ (r34DownTiedB N 28 28 xN cotN vN epsStr w.d2 (r34Pre3 N w x) dyD2
      ∧ r34DownLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.d2 (r34Pre3 N w x)
        (fun p => L (resnet34ForwardBFull N { w with d2 := p } x)) dyD2)
  ∧ (r34IdTiedB N 28 28 xN cotN vN epsStr w.b0 (r34Pre4 N w x) dyB0
      ∧ r34IdLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b0 (r34Pre4 N w x)
        (fun p => L (resnet34ForwardBFull N { w with b0 := p } x)) dyB0)
  ∧ (r34IdTiedB N 28 28 xN cotN vN epsStr w.b1 (r34Pre5 N w x) dyB1
      ∧ r34IdLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b1 (r34Pre5 N w x)
        (fun p => L (resnet34ForwardBFull N { w with b1 := p } x)) dyB1)
  ∧ (r34IdTiedB N 28 28 xN cotN vN epsStr w.b2 (r34Pre6 N w x) dyB2
      ∧ r34IdLossTiedB (N := N) (h := 28) (w := 28) xN cotN vN epsStr w.b2 (r34Pre6 N w x)
        (fun p => L (resnet34ForwardBFull N { w with b2 := p } x)) dyB2)
  ∧ (r34DownTiedB N 14 14 xN cotN vN epsStr w.d3 (r34Pre7 N w x) dyD3
      ∧ r34DownLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.d3 (r34Pre7 N w x)
        (fun p => L (resnet34ForwardBFull N { w with d3 := p } x)) dyD3)
  ∧ (r34IdTiedB N 14 14 xN cotN vN epsStr w.c0 (r34Pre8 N w x) dyC0
      ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c0 (r34Pre8 N w x)
        (fun p => L (resnet34ForwardBFull N { w with c0 := p } x)) dyC0)
  ∧ (r34IdTiedB N 14 14 xN cotN vN epsStr w.c1 (r34Pre9 N w x) dyC1
      ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c1 (r34Pre9 N w x)
        (fun p => L (resnet34ForwardBFull N { w with c1 := p } x)) dyC1)
  ∧ (r34IdTiedB N 14 14 xN cotN vN epsStr w.c2 (r34Pre10 N w x) dyC2
      ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c2 (r34Pre10 N w x)
        (fun p => L (resnet34ForwardBFull N { w with c2 := p } x)) dyC2)
  ∧ (r34IdTiedB N 14 14 xN cotN vN epsStr w.c3 (r34Pre11 N w x) dyC3
      ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c3 (r34Pre11 N w x)
        (fun p => L (resnet34ForwardBFull N { w with c3 := p } x)) dyC3)
  ∧ (r34IdTiedB N 14 14 xN cotN vN epsStr w.c4 (r34Pre12 N w x) dyC4
      ∧ r34IdLossTiedB (N := N) (h := 14) (w := 14) xN cotN vN epsStr w.c4 (r34Pre12 N w x)
        (fun p => L (resnet34ForwardBFull N { w with c4 := p } x)) dyC4)
  ∧ (r34DownTiedB N 7 7 xN cotN vN epsStr w.d4 (r34Pre13 N w x) dyD4
      ∧ r34DownLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.d4 (r34Pre13 N w x)
        (fun p => L (resnet34ForwardBFull N { w with d4 := p } x)) dyD4)
  ∧ (r34IdTiedB N 7 7 xN cotN vN epsStr w.e0 (r34Pre14 N w x) dyE0
      ∧ r34IdLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.e0 (r34Pre14 N w x)
        (fun p => L (resnet34ForwardBFull N { w with e0 := p } x)) dyE0)
  ∧ (r34IdTiedB N 7 7 xN cotN vN epsStr w.e1 (r34Pre15 N w x) dyE1
      ∧ r34IdLossTiedB (N := N) (h := 7) (w := 7) xN cotN vN epsStr w.e1 (r34Pre15 N w x)
        (fun p => L (resnet34ForwardBFull N { w with e1 := p } x)) dyE1)
  ∧ (r34HeadTiedB N 7 7 xN cotN w.Wd w.bd (r34Pre16 N w x) g
      ∧ r34HeadLossTiedB (N := N) (h := 7) (w := 7) xN cotN w.Wd w.bd (r34Pre16 N w x)
        (fun W b => L (resnet34ForwardBFull N { w with Wd := W, bd := b } x)) g) := by
  intro g dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0 cotPool L
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17⟩ :=
    r34_net_tiedB N xN cotN vN epsStr aStr negAK bStr logN ohN α B w x t
  have hl :=
    r34_net_lossGrad_smoothedCE N hK xN cotN vN epsStr aStr negAK bStr logN ohN α B w hq x hx t ht
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, l12⟩, ⟨t13, l13⟩, ⟨t14, l14⟩, ⟨t15, l15⟩, ⟨t16, l16⟩,
    ⟨t17, l17⟩⟩

end Net

end Proofs.ResNet34TieB
