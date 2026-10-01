import LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34ParamGrad
import LeanMlir.Proofs.Foundation.BceBatchLoss

/-! # ResNet-50 — every parameter gradient node IS the loss's derivative in that parameter

`r50_net_tiedB` says each of the 161 parameter gradient nodes denotes its layer's parameter
Jacobian contracted with the cotangent the emitted backward chain threads to it, from a loss
cotangent `g` it takes as a binder; the `r50*CotIn_eq_vjp` lemmas say the block-input cotangents
are certified VJP backwards. `r50_net_lossGrad` composes them: for any loss `L` of the logits whose
gradient at the net's output is `g`, every node is `∂L/∂θ` of the WHOLE net with that one parameter
varied. The two losses the artifacts ship discharge `hL`:

* `r50_net_lossGrad_smoothedCE` — the `bce := false` artifacts: `L = smoothedBatchLoss`, `g` the
  six-op label-smoothed chain (`smoothedBatchLoss_grad`).
* `r50_net_lossGrad_bce` — the `bce := true` artifacts, `resnet50in160_lambaccdp8x64wxclipbce` among
  them: `L = bceBatchLoss`, the mean over `B×K`, `g` the three-op chain (`bceBatchLoss_grad`).

**Scope.** Every node here is an f32 `*GradB` node on one replica. The bf16 nodes (`*GradBBf16`)
that `resnet50in_momdp64bf16` and `resnet50in160_lambaccdp4x128wxclipbcebf16`, the book's ImageNet
runs, emit are outside this statement, and so are the drop-path forwards (`*drop*`). Sync-BN data
parallelism is reached by composition: `r50_net_syncTiedB` says each all-reduced gradient is this
net's tied node at `N := R·N`, which this file's capstone makes the loss's gradient at the global
batch.

**How.** `ResNet34ParamGrad`'s three layers, with R34's stem and head bundles reused verbatim (the
stem and head ARE R34's functions at R50's widths):

* **Per block kind** (at variable widths): the loss read at an identity / stride-1 projection /
  strided projection bottleneck's output, pulled back one certified stage at a time by
  `ParamGradNodes`' pull-backs, so each internal activation's gradient is the chain's own cotangent.
  The bundles `r50IdLossTiedB` / `r50ProjLossTiedB` / `r50DownLossTiedB` state the block's nodes
  against `Φ`, the loss as a function of the block's weight record — the same slots as
  `r50IdTiedB` / `r50ProjTiedB` / `r50DownTiedB`, no conv bias (`ResNet50RenderB` emits none).
* **Per net**: the loss read after each block (`r50Suf*`), its gradient pulled back through the
  sixteen certified bottleneck VJPs, and `Φ` identified with the whole net at updated weights
  (`r50_factor_*`), each a standalone theorem.

**Hypotheses.** `R50PosB` (every BN `ε > 0`), `R50LossSmoothAtB` (every relu off its kink and every
stem-pool window dead, or its maximum at one position up to cells reading identical input
patches, at the real activations); for the smoothed loss also every example's target summing to one and `0 < nCls`.
The BCE corollary takes no hypothesis on the target.

**The stem pool's clause is stated for the parameters, not the image.** Real batches have stem-pool
windows whose positive maximum sits at two positions, because two cells read identical input
patches (flat image regions). There the net has no derivative in the image, so the input VJP's
`R50SmoothAtB` rejects them, but the tied cells are the same function of the stem's weights, so the
loss IS differentiable in the parameters. `R50LossSmoothAtB` allows exactly those ties
(`StemPoolTwinAt` at `StemConvTwin`), and the stem's parameter nodes are proved through the argmax
gather they reduce to (`r34StemPool_param_germ`). The probe script
scripts/probes/stem_pool_smooth_probe.py checks the stem's clauses on real batches.
-/

open Proofs Proofs.StableHLO Proofs.ResNet34TieB

namespace Proofs.ResNet50TieB

open Proofs.BackLinks (bnInB bnInB_eq_bnBackB reluMaskB cInB cStridedInB reassocB rowB unrowB)
open Proofs.GradNodeB (hasGradAt_bnBatchLA hasGradAt_relu hasGradAt_conv hasGradAt_convStrided
  hasGradAt_addConst hasGradAt_constAdd)
open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The identity bottleneck
--   `Gn` is the loss read at the block's OUTPUT, pulled back one stage at a time through
--   `ParamGradNodes`' pull-backs.
-- ════════════════════════════════════════════════════════════════

section IdBlock
variable {N h w mid oc : Nat}

/-- **Identity bottleneck, every parameter node a loss derivative.** With `Gn` the loss read at
    the block's output and `Φ` the loss as a function of the block's weight record (`hΦ`), each of
    the nine nodes `r50IdTiedB` ties — at the same cotangents — is `∂Φ/∂slot` with that one slot
    varied. -/
def r50IdLossTiedB (xN cotN vN epsStr : String) (p : R50IdW mid oc) (v : Vec (N * (oc * h * w)))
    (Φ : R50IdW mid oc → Vec 1) (dy : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let r2 := cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap N (flatConv p.W₁ p.b₁) v
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let c3 := batchMap N (flatConv p.W₃ p.b₃) r2
  (HasGradAt (fun θ => Φ { p with W₁ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₁)
        (den (SHlo.convWeightGradB xN p.b₁ v p.W₁ (.operand cotN (r50IdCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₁ := θ }) p.γ₁
        (den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N mid h w c1)
          (.operand cotN (reassocB N mid h w (r50IdCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₁ := θ }) p.β₁
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (r50IdCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₂ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₂)
        (den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN (r50IdCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₂ := θ }) p.γ₂
        (den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N mid h w c2)
          (.operand cotN (reassocB N mid h w (r50IdCotN2 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₂ := θ }) p.β₂
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (r50IdCotN2 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₃ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₃)
        (den (SHlo.convWeightGradB xN p.b₃ r2 p.W₃ (.operand cotN (r50IdCotC3 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₃ := θ }) p.γ₃
        (den (SHlo.bnGammaGradB vN epsStr p.ε₃ (reassocB N oc h w c3)
          (.operand cotN (reassocB N oc h w (r50IdCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₃ := θ }) p.β₃
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r50IdCotA N h w p v dy))))))

theorem r50_idblock_lossTiedB (xN cotN vN epsStr : String) (p : R50IdW mid oc) (hq : R50IdPos p)
    (v : Vec (N * (oc * h * w))) (hs : R50IdSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r50IdB N h w p v) dy) {Φ : R50IdW mid oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r50IdB N h w p' v)) :
    r50IdLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r50IdB N h w p' v) from funext hΦ]
  have hN3 := hasGradAt_addConst (bnBatchLA N oc h w p.ε₃ p.γ₃ p.β₃ (batchMap N (flatConv p.W₃ p.b₃)
    (cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
      (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)))) v (hasGradAt_relu _ hs.hout hGn)
  have hC3 := hasGradAt_bnBatchLA p.ε₃ hq.h3 p.γ₃ p.β₃ _ hN3
  have hN2 := hasGradAt_relu _ hs.hm2 (hasGradAt_conv p.W₃ p.b₃ _ hC3)
  have hC2 := hasGradAt_bnBatchLA p.ε₂ hq.h2 p.γ₂ p.β₂ _ hN2
  have hN1 := hasGradAt_relu _ hs.hm1 (hasGradAt_conv p.W₂ p.b₂ _ hC2)
  have hC1 := hasGradAt_bnBatchLA p.ε₁ hq.h1 p.γ₁ p.β₁ _ hN1
  exact ⟨GradNodeB.convW_hasGradAt xN cotN p.b₁ v p.W₁ hC1,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.convW_hasGradAt xN cotN p.b₂ _ p.W₂ hC2,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.convW_hasGradAt xN cotN p.b₃ _ p.W₃ hC3,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₃ p.γ₃ p.β₃ _ hN3,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₃ p.γ₃ p.β₃ _ hN3⟩

end IdBlock

-- ════════════════════════════════════════════════════════════════
-- § The stride-1 projection bottleneck (stage 1 block 0)
--   `residualProj proj body = proj + body`: a body parameter sees the projection branch as a
--   constant on the LEFT, a projection parameter sees the body on the RIGHT.
-- ════════════════════════════════════════════════════════════════

section ProjBlock
variable {N h w ic mid oc : Nat}

/-- **Stride-1 projection bottleneck, every parameter node a loss derivative** — the twelve nodes
    `r50ProjTiedB` ties. -/
def r50ProjLossTiedB (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (v : Vec (N * (ic * h * w))) (Φ : R50ProjW ic mid oc → Vec 1) (dy : Vec (N * (oc * h * w))) :
    Prop :=
  let r1 := cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let r2 := cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap N (flatConv p.W₁ p.b₁) v
  let c2 := batchMap N (flatConv p.W₂ p.b₂) r1
  let c3 := batchMap N (flatConv p.W₃ p.b₃) r2
  let cp := batchMap N (flatConv p.Wp p.bp) v
  (HasGradAt (fun θ => Φ { p with W₁ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₁)
        (den (SHlo.convWeightGradB xN p.b₁ v p.W₁ (.operand cotN (r50ProjCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₁ := θ }) p.γ₁
        (den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N mid h w c1)
          (.operand cotN (reassocB N mid h w (r50ProjCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₁ := θ }) p.β₁
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (r50ProjCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₂ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₂)
        (den (SHlo.convWeightGradB xN p.b₂ r1 p.W₂ (.operand cotN (r50ProjCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₂ := θ }) p.γ₂
        (den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N mid h w c2)
          (.operand cotN (reassocB N mid h w (r50ProjCotN2 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₂ := θ }) p.β₂
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (r50ProjCotN2 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₃ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₃)
        (den (SHlo.convWeightGradB xN p.b₃ r2 p.W₃ (.operand cotN (r50ProjCotC3 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₃ := θ }) p.γ₃
        (den (SHlo.bnGammaGradB vN epsStr p.ε₃ (reassocB N oc h w c3)
          (.operand cotN (reassocB N oc h w (r50ProjCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₃ := θ }) p.β₃
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r50ProjCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wp := Kernel4.unflatten θ }) (Kernel4.flatten p.Wp)
        (den (SHlo.convWeightGradB xN p.bp v p.Wp (.operand cotN (r50ProjCotCp N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γp := θ }) p.γp
        (den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N oc h w cp)
          (.operand cotN (reassocB N oc h w (r50ProjCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with βp := θ }) p.βp
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r50ProjCotA N h w p v dy))))))

theorem r50_projblock_lossTiedB (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (v : Vec (N * (ic * h * w))) (hs : R50ProjSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r50ProjB N h w p v) dy) {Φ : R50ProjW ic mid oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r50ProjB N h w p' v)) :
    r50ProjLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r50ProjB N h w p' v) from funext hΦ]
  -- `residualProj proj body = proj + body`: a body parameter sees the projection branch as a
  -- constant on the left, a projection parameter the body on the right
  have hA := hasGradAt_relu (fun i => projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp v i
    + projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃
      (cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) i) hs.hout hGn
  have hN3 := hasGradAt_constAdd (projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp v)
    (bnBatchLA N oc h w p.ε₃ p.γ₃ p.β₃ (batchMap N (flatConv p.W₃ p.b₃)
      (cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)))) hA
  have hNp := hasGradAt_addConst (bnBatchLA N oc h w p.εp p.γp p.βp (batchMap N (flatConv p.Wp p.bp) v))
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃
      (cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))) hA
  have hC3 := hasGradAt_bnBatchLA p.ε₃ hq.h3 p.γ₃ p.β₃ _ hN3
  have hN2 := hasGradAt_relu _ hs.hm2 (hasGradAt_conv p.W₃ p.b₃ _ hC3)
  have hC2 := hasGradAt_bnBatchLA p.ε₂ hq.h2 p.γ₂ p.β₂ _ hN2
  have hN1 := hasGradAt_relu _ hs.hm1 (hasGradAt_conv p.W₂ p.b₂ _ hC2)
  have hC1 := hasGradAt_bnBatchLA p.ε₁ hq.h1 p.γ₁ p.β₁ _ hN1
  have hCp := hasGradAt_bnBatchLA p.εp hq.hp p.γp p.βp _ hNp
  exact ⟨GradNodeB.convW_hasGradAt xN cotN p.b₁ v p.W₁ hC1,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.convW_hasGradAt xN cotN p.b₂ _ p.W₂ hC2,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.convW_hasGradAt xN cotN p.b₃ _ p.W₃ hC3,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₃ p.γ₃ p.β₃ _ hN3,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₃ p.γ₃ p.β₃ _ hN3,
    GradNodeB.convW_hasGradAt xN cotN p.bp v p.Wp hCp,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.εp p.γp p.βp _ hNp,
    GradNodeB.bnBeta_hasGradAt cotN p.εp p.γp p.βp _ hNp⟩

end ProjBlock

-- ════════════════════════════════════════════════════════════════
-- § The strided projection bottleneck (stages 2/3/4 block 0)
--   v1.5: conv₁/bn₁/relu₁ run at `2h × 2w`, conv₂ and the skip are strided; the cotangent crosses
--   the strided conv₂ inside the block (`cStridedInB_eq_batchMapBackward`).
-- ════════════════════════════════════════════════════════════════

section DownBlock
variable {N h w ic mid oc : Nat}

/-- **Strided projection bottleneck, every parameter node a loss derivative** — the twelve nodes
    `r50DownTiedB` ties: `W₁` an ordinary conv node at `2h × 2w`, `W₂` and `Wp` strided. -/
def r50DownLossTiedB (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (Φ : R50ProjW ic mid oc → Vec 1)
    (dy : Vec (N * (oc * h * w))) : Prop :=
  let r1 := cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v
  let r2 := cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap N (flatConv p.W₁ p.b₁) v
  let c2 := batchMap N (flatConvStride2 p.W₂ p.b₂) r1
  let c3 := batchMap N (flatConv p.W₃ p.b₃) r2
  let cp := batchMap N (flatConvStride2 p.Wp p.bp) v
  (HasGradAt (fun θ => Φ { p with W₁ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₁)
        (den (SHlo.convWeightGradB xN p.b₁ v p.W₁ (.operand cotN (r50DownCotC1 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₁ := θ }) p.γ₁
        (den (SHlo.bnGammaGradB vN epsStr p.ε₁ (reassocB N mid (2 * h) (2 * w) c1)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) (r50DownCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₁ := θ }) p.β₁
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := 2 * h) (w := 2 * w)
          (.operand cotN (reassocB N mid (2 * h) (2 * w) (r50DownCotN1 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₂ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₂)
        (den (SHlo.convStridedWeightGradB xN p.b₂ r1 p.W₂
          (.operand cotN (r50DownCotC2 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₂ := θ }) p.γ₂
        (den (SHlo.bnGammaGradB vN epsStr p.ε₂ (reassocB N mid h w c2)
          (.operand cotN (reassocB N mid h w (r50DownCotN2 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₂ := θ }) p.β₂
        (den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
          (.operand cotN (reassocB N mid h w (r50DownCotN2 N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with W₃ := Kernel4.unflatten θ }) (Kernel4.flatten p.W₃)
        (den (SHlo.convWeightGradB xN p.b₃ r2 p.W₃ (.operand cotN (r50DownCotC3 N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γ₃ := θ }) p.γ₃
        (den (SHlo.bnGammaGradB vN epsStr p.ε₃ (reassocB N oc h w c3)
          (.operand cotN (reassocB N oc h w (r50DownCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with β₃ := θ }) p.β₃
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r50DownCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with Wp := Kernel4.unflatten θ }) (Kernel4.flatten p.Wp)
        (den (SHlo.convStridedWeightGradB xN p.bp v p.Wp
          (.operand cotN (r50DownCotCp N h w p v dy)))))
  ∧ (HasGradAt (fun θ => Φ { p with γp := θ }) p.γp
        (den (SHlo.bnGammaGradB vN epsStr p.εp (reassocB N oc h w cp)
          (.operand cotN (reassocB N oc h w (r50DownCotA N h w p v dy))))))
  ∧ (HasGradAt (fun θ => Φ { p with βp := θ }) p.βp
        (den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB N oc h w (r50DownCotA N h w p v dy))))))

theorem r50_downblock_lossTiedB (xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R50DownSmoothAt N h w p v)
    {Gn : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hGn : HasGradAt Gn (r50DownB N h w p v) dy) {Φ : R50ProjW ic mid oc → Vec 1}
    (hΦ : ∀ p', Φ p' = Gn (r50DownB N h w p' v)) :
    r50DownLossTiedB xN cotN vN epsStr p v Φ dy := by
  rw [show Φ = fun p' => Gn (r50DownB N h w p' v) from funext hΦ]
  have hA := hasGradAt_relu (fun i => projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp v i
    + projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃
      (cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) i) hs.hout hGn
  have hN3 := hasGradAt_constAdd (projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp v)
    (bnBatchLA N oc h w p.ε₃ p.γ₃ p.β₃ (batchMap N (flatConv p.W₃ p.b₃)
      (cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)))) hA
  have hNp := hasGradAt_addConst
    (bnBatchLA N oc h w p.εp p.γp p.βp (batchMap N (flatConvStride2 p.Wp p.bp) v))
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃
      (cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v))) hA
  have hC3 := hasGradAt_bnBatchLA p.ε₃ hq.h3 p.γ₃ p.β₃ _ hN3
  have hN2 := hasGradAt_relu _ hs.hm2 (hasGradAt_conv p.W₃ p.b₃ _ hC3)
  have hC2 := hasGradAt_bnBatchLA p.ε₂ hq.h2 p.γ₂ p.β₂ _ hN2
  have hN1 := hasGradAt_relu _ hs.hm1 (hasGradAt_convStrided p.W₂ p.b₂ _ hC2)
  have hC1 := hasGradAt_bnBatchLA p.ε₁ hq.h1 p.γ₁ p.β₁ _ hN1
  have hCp := hasGradAt_bnBatchLA p.εp hq.hp p.γp p.βp _ hNp
  exact ⟨GradNodeB.convW_hasGradAt (h := 2 * h) (w := 2 * w) xN cotN p.b₁ v p.W₁ hC1,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₁ p.γ₁ p.β₁ _ hN1,
    GradNodeB.convStridedW_hasGradAt xN cotN p.b₂ _ p.W₂ hC2,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₂ p.γ₂ p.β₂ _ hN2,
    GradNodeB.convW_hasGradAt xN cotN p.b₃ _ p.W₃ hC3,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.ε₃ p.γ₃ p.β₃ _ hN3,
    GradNodeB.bnBeta_hasGradAt cotN p.ε₃ p.γ₃ p.β₃ _ hN3,
    GradNodeB.convStridedW_hasGradAt xN cotN p.bp v p.Wp hCp,
    GradNodeB.bnGamma_hasGradAt vN epsStr cotN p.εp p.γp p.βp _ hNp,
    GradNodeB.bnBeta_hasGradAt cotN p.εp p.γp p.βp _ hNp⟩

end DownBlock

-- ════════════════════════════════════════════════════════════════
-- § The whole net: the loss after each block, and the net with one block's weights varied
-- ════════════════════════════════════════════════════════════════

/-- Pull the loss gradient back through an identity bottleneck: the certified block VJP, read at
    the chain's own fan-in (`r50IdCotIn_eq_vjp`). -/
theorem r50IdB_hasGradAt_comp {N h w mid oc : Nat} (p : R50IdW mid oc) (hq : R50IdPos p)
    (v : Vec (N * (oc * h * w))) (hs : R50IdSmoothAt N h w p v) {G : Vec (N * (oc * h * w)) → Vec 1}
    {dy : Vec (N * (oc * h * w))} (hG : HasGradAt G (r50IdB N h w p v) dy) :
    HasGradAt (fun y => G (r50IdB N h w p y)) v (r50IdCotIn N h w p v dy) :=
  (HasGradAt.comp (f := r50IdB N h w p) (x := v) hG
    ((StableHLO.r50BottleneckLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃).diff v
      ⟨⟨⟨hs.hm1, hs.hm2⟩, trivial⟩, hs.hout⟩)
    (r50IdBHasVJPAt N h w p hq v hs)).of_eq (r50IdCotIn_eq_vjp N h w p hq v dy hs).symm

/-- …through the stride-1 projection bottleneck (`r50ProjCotIn_eq_vjp`). -/
theorem r50ProjB_hasGradAt_comp {N h w ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (v : Vec (N * (ic * h * w))) (hs : R50ProjSmoothAt N h w p v)
    {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (r50ProjB N h w p v) dy) :
    HasGradAt (fun y => G (r50ProjB N h w p y)) v (r50ProjCotIn N h w p v dy) :=
  (HasGradAt.comp (f := r50ProjB N h w p) (x := v) hG
    ((StableHLO.r50ProjBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃
      p.Wp p.bp p.εp hq.hp p.γp p.βp).diff v ⟨⟨trivial, ⟨hs.hm1, hs.hm2⟩, trivial⟩, hs.hout⟩)
    (r50ProjBHasVJPAt N h w p hq v hs)).of_eq (r50ProjCotIn_eq_vjp N h w p hq v dy hs).symm

/-- …and through the strided projection bottleneck (`r50DownCotIn_eq_vjp`). -/
theorem r50DownB_hasGradAt_comp {N h w ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (v : Vec (N * (ic * (2 * h) * (2 * w)))) (hs : R50DownSmoothAt N h w p v)
    {G : Vec (N * (oc * h * w)) → Vec 1} {dy : Vec (N * (oc * h * w))}
    (hG : HasGradAt G (r50DownB N h w p v) dy) :
    HasGradAt (fun y => G (r50DownB N h w p y)) v (r50DownCotIn N h w p v dy) :=
  (HasGradAt.comp (f := r50DownB N h w p) (x := v) hG
    ((StableHLO.r50DownBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃
      p.Wp p.bp p.εp hq.hp p.γp p.βp).diff v ⟨⟨trivial, ⟨hs.hm1, hs.hm2⟩, trivial⟩, hs.hout⟩)
    (r50DownBHasVJPAt N h w p hq v hs)).of_eq (r50DownCotIn_eq_vjp N h w p hq v dy hs).symm

/-- The net after block `s4b2` — the head. -/
noncomputable def r50SufS4b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (2048 * q * q)) → Vec (N * nCls) :=
  r34HeadB N q q w.Wd w.bd

/-- The net after block `s4b1`: block `s4b2`, then the rest. -/
noncomputable def r50SufS4b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (2048 * q * q)) → Vec (N * nCls) :=
  fun y => r50SufS4b2 N q w (r50IdB N q q w.s4b2 y)

/-- The net after block `s4b0`: block `s4b1`, then the rest. -/
noncomputable def r50SufS4b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (2048 * q * q)) → Vec (N * nCls) :=
  fun y => r50SufS4b1 N q w (r50IdB N q q w.s4b1 y)

/-- The net after block `s3b5`: block `s4b0`, then the rest. -/
noncomputable def r50SufS3b5 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * nCls) :=
  fun y => r50SufS4b0 N q w (r50DownB N q q w.s4b0 y)

/-- The net after block `s3b4`: block `s3b5`, then the rest. -/
noncomputable def r50SufS3b4 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * nCls) :=
  fun y => r50SufS3b5 N q w (r50IdB N (2 * q) (2 * q) w.s3b5 y)

/-- The net after block `s3b3`: block `s3b4`, then the rest. -/
noncomputable def r50SufS3b3 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * nCls) :=
  fun y => r50SufS3b4 N q w (r50IdB N (2 * q) (2 * q) w.s3b4 y)

/-- The net after block `s3b2`: block `s3b3`, then the rest. -/
noncomputable def r50SufS3b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * nCls) :=
  fun y => r50SufS3b3 N q w (r50IdB N (2 * q) (2 * q) w.s3b3 y)

/-- The net after block `s3b1`: block `s3b2`, then the rest. -/
noncomputable def r50SufS3b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * nCls) :=
  fun y => r50SufS3b2 N q w (r50IdB N (2 * q) (2 * q) w.s3b2 y)

/-- The net after block `s3b0`: block `s3b1`, then the rest. -/
noncomputable def r50SufS3b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * nCls) :=
  fun y => r50SufS3b1 N q w (r50IdB N (2 * q) (2 * q) w.s3b1 y)

/-- The net after block `s2b3`: block `s3b0`, then the rest. -/
noncomputable def r50SufS2b3 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * nCls) :=
  fun y => r50SufS3b0 N q w (r50DownB N (2 * q) (2 * q) w.s3b0 y)

/-- The net after block `s2b2`: block `s2b3`, then the rest. -/
noncomputable def r50SufS2b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * nCls) :=
  fun y => r50SufS2b3 N q w (r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 y)

/-- The net after block `s2b1`: block `s2b2`, then the rest. -/
noncomputable def r50SufS2b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * nCls) :=
  fun y => r50SufS2b2 N q w (r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 y)

/-- The net after block `s2b0`: block `s2b1`, then the rest. -/
noncomputable def r50SufS2b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * nCls) :=
  fun y => r50SufS2b1 N q w (r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 y)

/-- The net after block `s1b2`: block `s2b0`, then the rest. -/
noncomputable def r50SufS1b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * nCls) :=
  fun y => r50SufS2b0 N q w (r50DownB N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 y)

/-- The net after block `s1b1`: block `s1b2`, then the rest. -/
noncomputable def r50SufS1b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * nCls) :=
  fun y => r50SufS1b2 N q w (r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 y)

/-- The net after block `s1b0`: block `s1b1`, then the rest. -/
noncomputable def r50SufS1b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * nCls) :=
  fun y => r50SufS1b1 N q w (r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 y)

/-- The net after the stem: block `s1b0`, then the rest. -/
noncomputable def r50SufStem (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * nCls) :=
  fun y => r50SufS1b0 N q w (r50ProjB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 y)

/-- **The net with the stem's parameters varied** is the suffix after the stem at the varied stem. -/
theorem r50_factor_stem (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (W : Kernel4 64 3 7 7) (b γ β : Vec 64) :
    resnet50ForwardBFull N q { w with sW := W, sb := b, sγ := γ, sβ := β } x
      = r50SufStem N q w (r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) W b w.sε γ β x) := rfl

/-- **The net with block `s1b0`'s weights varied** is the suffix after `s1b0` at the varied block. -/
theorem r50_factor_s1b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50ProjW 64 64 256) :
    resnet50ForwardBFull N q { w with s1b0 := p } x
      = r50SufS1b0 N q w (r50ProjB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) p (r50Pre0 N q w x)) := by
  rw [r50Pre0_apply]; rfl

/-- **The net with block `s1b1`'s weights varied** is the suffix after `s1b1` at the varied block. -/
theorem r50_factor_s1b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 64 256) :
    resnet50ForwardBFull N q { w with s1b1 := p } x
      = r50SufS1b1 N q w (r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) p (r50Pre1 N q w x)) := by
  rw [r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s1b2`'s weights varied** is the suffix after `s1b2` at the varied block. -/
theorem r50_factor_s1b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 64 256) :
    resnet50ForwardBFull N q { w with s1b2 := p } x
      = r50SufS1b2 N q w (r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) p (r50Pre2 N q w x)) := by
  rw [r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s2b0`'s weights varied** is the suffix after `s2b0` at the varied block. -/
theorem r50_factor_s2b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50ProjW 256 128 512) :
    resnet50ForwardBFull N q { w with s2b0 := p } x
      = r50SufS2b0 N q w (r50DownB N (2 * (2 * q)) (2 * (2 * q)) p (r50Pre3 N q w x)) := by
  rw [r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s2b1`'s weights varied** is the suffix after `s2b1` at the varied block. -/
theorem r50_factor_s2b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 128 512) :
    resnet50ForwardBFull N q { w with s2b1 := p } x
      = r50SufS2b1 N q w (r50IdB N (2 * (2 * q)) (2 * (2 * q)) p (r50Pre4 N q w x)) := by
  rw [r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s2b2`'s weights varied** is the suffix after `s2b2` at the varied block. -/
theorem r50_factor_s2b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 128 512) :
    resnet50ForwardBFull N q { w with s2b2 := p } x
      = r50SufS2b2 N q w (r50IdB N (2 * (2 * q)) (2 * (2 * q)) p (r50Pre5 N q w x)) := by
  rw [r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s2b3`'s weights varied** is the suffix after `s2b3` at the varied block. -/
theorem r50_factor_s2b3 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 128 512) :
    resnet50ForwardBFull N q { w with s2b3 := p } x
      = r50SufS2b3 N q w (r50IdB N (2 * (2 * q)) (2 * (2 * q)) p (r50Pre6 N q w x)) := by
  rw [r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s3b0`'s weights varied** is the suffix after `s3b0` at the varied block. -/
theorem r50_factor_s3b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50ProjW 512 256 1024) :
    resnet50ForwardBFull N q { w with s3b0 := p } x
      = r50SufS3b0 N q w (r50DownB N (2 * q) (2 * q) p (r50Pre7 N q w x)) := by
  rw [r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s3b1`'s weights varied** is the suffix after `s3b1` at the varied block. -/
theorem r50_factor_s3b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 256 1024) :
    resnet50ForwardBFull N q { w with s3b1 := p } x
      = r50SufS3b1 N q w (r50IdB N (2 * q) (2 * q) p (r50Pre8 N q w x)) := by
  rw [r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s3b2`'s weights varied** is the suffix after `s3b2` at the varied block. -/
theorem r50_factor_s3b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 256 1024) :
    resnet50ForwardBFull N q { w with s3b2 := p } x
      = r50SufS3b2 N q w (r50IdB N (2 * q) (2 * q) p (r50Pre9 N q w x)) := by
  rw [r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s3b3`'s weights varied** is the suffix after `s3b3` at the varied block. -/
theorem r50_factor_s3b3 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 256 1024) :
    resnet50ForwardBFull N q { w with s3b3 := p } x
      = r50SufS3b3 N q w (r50IdB N (2 * q) (2 * q) p (r50Pre10 N q w x)) := by
  rw [r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s3b4`'s weights varied** is the suffix after `s3b4` at the varied block. -/
theorem r50_factor_s3b4 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 256 1024) :
    resnet50ForwardBFull N q { w with s3b4 := p } x
      = r50SufS3b4 N q w (r50IdB N (2 * q) (2 * q) p (r50Pre11 N q w x)) := by
  rw [r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s3b5`'s weights varied** is the suffix after `s3b5` at the varied block. -/
theorem r50_factor_s3b5 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 256 1024) :
    resnet50ForwardBFull N q { w with s3b5 := p } x
      = r50SufS3b5 N q w (r50IdB N (2 * q) (2 * q) p (r50Pre12 N q w x)) := by
  rw [r50Pre12_apply, r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s4b0`'s weights varied** is the suffix after `s4b0` at the varied block. -/
theorem r50_factor_s4b0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50ProjW 1024 512 2048) :
    resnet50ForwardBFull N q { w with s4b0 := p } x
      = r50SufS4b0 N q w (r50DownB N q q p (r50Pre13 N q w x)) := by
  rw [r50Pre13_apply, r50Pre12_apply, r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s4b1`'s weights varied** is the suffix after `s4b1` at the varied block. -/
theorem r50_factor_s4b1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 512 2048) :
    resnet50ForwardBFull N q { w with s4b1 := p } x
      = r50SufS4b1 N q w (r50IdB N q q p (r50Pre14 N q w x)) := by
  rw [r50Pre14_apply, r50Pre13_apply, r50Pre12_apply, r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with block `s4b2`'s weights varied** is the suffix after `s4b2` at the varied block. -/
theorem r50_factor_s4b2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (p : R50IdW 512 2048) :
    resnet50ForwardBFull N q { w with s4b2 := p } x
      = r50SufS4b2 N q w (r50IdB N q q p (r50Pre15 N q w x)) := by
  rw [r50Pre15_apply, r50Pre14_apply, r50Pre13_apply, r50Pre12_apply, r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **The net with the classifier varied** is the head at the varied classifier. -/
theorem r50_factor_head (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (W : Mat 2048 nCls) (b : Vec nCls) :
    resnet50ForwardBFull N q { w with Wd := W, bd := b } x = r34HeadB N q q W b (r50Pre16 N q w x) := by
  rw [r50Pre16_apply, r50Pre15_apply, r50Pre14_apply, r50Pre13_apply, r50Pre12_apply, r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]; rfl

/-- **Every ResNet-50 parameter gradient node is the derivative of `L` in that parameter**, for a
    loss `L` of the logits and `g` the cotangent the chain starts from: the 161 nodes
    `r50_net_tiedB` ties, each at the cotangent the emitted chain threads to it, stated against `L`
    of `resnet50ForwardBFull` with that one parameter varied. `r50_net_lossGrad` proves it whenever
    `g` is `L`'s gradient at the logits; the two losses the artifacts ship instantiate it. -/
def R50NetLossTiedB (N q : Nat) {nCls : Nat} (xN cotN vN epsStr : String) (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) (L : Vec (N * nCls) → Vec 1) (g : Vec (N * nCls)) : Prop :=
    let dy16 := r34HeadCotBlk N q q w.Wd w.bd (r50Pre16 N q w x) g
    let dy15 := r50IdCotIn N q q w.s4b2 (r50Pre15 N q w x) dy16
    let dy14 := r50IdCotIn N q q w.s4b1 (r50Pre14 N q w x) dy15
    let dy13 := r50DownCotIn N q q w.s4b0 (r50Pre13 N q w x) dy14
    let dy12 := r50IdCotIn N (2 * q) (2 * q) w.s3b5 (r50Pre12 N q w x) dy13
    let dy11 := r50IdCotIn N (2 * q) (2 * q) w.s3b4 (r50Pre11 N q w x) dy12
    let dy10 := r50IdCotIn N (2 * q) (2 * q) w.s3b3 (r50Pre10 N q w x) dy11
    let dy9 := r50IdCotIn N (2 * q) (2 * q) w.s3b2 (r50Pre9 N q w x) dy10
    let dy8 := r50IdCotIn N (2 * q) (2 * q) w.s3b1 (r50Pre8 N q w x) dy9
    let dy7 := r50DownCotIn N (2 * q) (2 * q) w.s3b0 (r50Pre7 N q w x) dy8
    let dy6 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 N q w x) dy7
    let dy5 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 N q w x) dy6
    let dy4 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 N q w x) dy5
    let dy3 := r50DownCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 N q w x) dy4
    let dy2 := r50IdCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Pre2 N q w x) dy3
    let dy1 := r50IdCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Pre1 N q w x) dy2
    let cotPool := r50ProjCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Pre0 N q w x) dy1
    r34StemLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x
      (fun W b γ β => L (resnet50ForwardBFull N q { w with sW := W, sb := b, sγ := γ, sβ := β } x))
      cotPool
  ∧ r50ProjLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.s1b0 (r50Pre0 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s1b0 := p } x)) dy1
  ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.s1b1 (r50Pre1 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s1b1 := p } x)) dy2
  ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.s1b2 (r50Pre2 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s1b2 := p } x)) dy3
  ∧ r50DownLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b0 (r50Pre3 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s2b0 := p } x)) dy4
  ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b1 (r50Pre4 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s2b1 := p } x)) dy5
  ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b2 (r50Pre5 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s2b2 := p } x)) dy6
  ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b3 (r50Pre6 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s2b3 := p } x)) dy7
  ∧ r50DownLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b0 (r50Pre7 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s3b0 := p } x)) dy8
  ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b1 (r50Pre8 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s3b1 := p } x)) dy9
  ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b2 (r50Pre9 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s3b2 := p } x)) dy10
  ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b3 (r50Pre10 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s3b3 := p } x)) dy11
  ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b4 (r50Pre11 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s3b4 := p } x)) dy12
  ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b5 (r50Pre12 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s3b5 := p } x)) dy13
  ∧ r50DownLossTiedB (N := N) (h := q) (w := q) xN cotN vN epsStr w.s4b0 (r50Pre13 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s4b0 := p } x)) dy14
  ∧ r50IdLossTiedB (N := N) (h := q) (w := q) xN cotN vN epsStr w.s4b1 (r50Pre14 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s4b1 := p } x)) dy15
  ∧ r50IdLossTiedB (N := N) (h := q) (w := q) xN cotN vN epsStr w.s4b2 (r50Pre15 N q w x)
      (fun p => L (resnet50ForwardBFull N q { w with s4b2 := p } x)) dy16
  ∧ r34HeadLossTiedB (N := N) (h := q) (w := q) xN cotN w.Wd w.bd (r50Pre16 N q w x)
      (fun W b => L (resnet50ForwardBFull N q { w with Wd := W, bd := b } x)) g

/-- **The smooth-point bundle a loss gradient needs** — `R50SmoothAtB` with the stem pool's clause
    weakened to allow ties between cells that read identical input patches, as
    `ResNet34TieB.R34LossSmoothAtB` does for ResNet-34 (whose stem this is). -/
structure R50LossSmoothAtB (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) : Prop where
  stem : R34StemSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ x
  pool : StemPoolTwinAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (StemConvTwin N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) 64 x)
    (cbReluStridedB N (h := 2 * (2 * (2 * (2 * q)))) (w := 2 * (2 * (2 * (2 * q)))) w.sW w.sb w.sε w.sγ w.sβ x)
  s1b0 : R50ProjSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Pre0 N q w x)
  s1b1 : R50IdSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Pre1 N q w x)
  s1b2 : R50IdSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Pre2 N q w x)
  s2b0 : R50DownSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 N q w x)
  s2b1 : R50IdSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 N q w x)
  s2b2 : R50IdSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 N q w x)
  s2b3 : R50IdSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 N q w x)
  s3b0 : R50DownSmoothAt N (2 * q) (2 * q) w.s3b0 (r50Pre7 N q w x)
  s3b1 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b1 (r50Pre8 N q w x)
  s3b2 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b2 (r50Pre9 N q w x)
  s3b3 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b3 (r50Pre10 N q w x)
  s3b4 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b4 (r50Pre11 N q w x)
  s3b5 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b5 (r50Pre12 N q w x)
  s4b0 : R50DownSmoothAt N q q w.s4b0 (r50Pre13 N q w x)
  s4b1 : R50IdSmoothAt N q q w.s4b1 (r50Pre14 N q w x)
  s4b2 : R50IdSmoothAt N q q w.s4b2 (r50Pre15 N q w x)

/-- The input-VJP bundle implies the loss-gradient one. -/
theorem r50LossSmoothAtB_of_smoothAtB {N q nCls : Nat} {w : R50BWeights nCls}
    {x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))}
    (hx : R50SmoothAtB N q w x) : R50LossSmoothAtB N q w x :=
  ⟨hx.stem, stemPoolTwinAt_of_smoothAt _ _ _ _ hx.pool, hx.s1b0, hx.s1b1, hx.s1b2, hx.s2b0, hx.s2b1, hx.s2b2, hx.s2b3, hx.s3b0, hx.s3b1, hx.s3b2, hx.s3b3, hx.s3b4, hx.s3b5, hx.s4b0, hx.s4b1, hx.s4b2⟩

/-- **Every ResNet-50 parameter gradient node is the derivative of the loss in that parameter.**
    For any loss `L` of the logits with gradient `g` at the net's output, each of the 161 nodes
    `r50_net_tiedB` ties — at the same cotangent — is `∂L/∂θ` of the WHOLE net, `resnet50ForwardBFull`
    with that one parameter varied (a stem field, a block's weight record `w.blk := p` with one slot
    changed, or the classifier).

    Hypotheses: every BN `ε` positive (`R50PosB`), every relu off its kink and every stem-pool
    window dead or tied only between cells reading identical input patches, at the real
    activations (`R50LossSmoothAtB`). The loss enters only through `hL`;
    `r50_net_lossGrad_smoothedCE` and `r50_net_lossGrad_bce` discharge it for the two losses the
    artifacts ship.
    The nodes are the f32 ones on one replica (the module's Scope). -/
theorem r50_net_lossGrad (N q : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : R50BWeights nCls) (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) (hx : R50LossSmoothAtB N q w x)
    {L : Vec (N * nCls) → Vec 1} {g : Vec (N * nCls)}
    (hL : HasGradAt L (resnet50ForwardBFull N q w x) g) :
    R50NetLossTiedB N q xN cotN vN epsStr w x L g := by
  unfold R50NetLossTiedB
  intro dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotPool
  have hL' : HasGradAt L (r34HeadB N q q w.Wd w.bd (r50Pre16 N q w x)) g :=
    hL.congr_point (by rw [resnet50ForwardBFull_eq_chain, Function.comp_apply])
  have h16 : HasGradAt (fun y => L (r50SufS4b2 N q w y)) (r50Pre16 N q w x) dy16 :=
    HasGradAt.comp (f := r34HeadB N q q w.Wd w.bd) (x := r50Pre16 N q w x) hL'
      (((batchMap_differentiable _ (dense_differentiable w.Wd w.bd)).comp
        (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 2048 q q))) _)
      ((r34HeadBHasVJP N q q w.Wd w.bd).toHasVJPAt _)
  have h15 : HasGradAt (fun y => L (r50SufS4b1 N q w y)) (r50Pre15 N q w x) dy15 :=
    r50IdB_hasGradAt_comp w.s4b2 hp.s4b2 _ hx.s4b2 (h16.congr_point (r50Pre16_apply N q w x))
  have h14 : HasGradAt (fun y => L (r50SufS4b0 N q w y)) (r50Pre14 N q w x) dy14 :=
    r50IdB_hasGradAt_comp w.s4b1 hp.s4b1 _ hx.s4b1 (h15.congr_point (r50Pre15_apply N q w x))
  have h13 : HasGradAt (fun y => L (r50SufS3b5 N q w y)) (r50Pre13 N q w x) dy13 :=
    r50DownB_hasGradAt_comp w.s4b0 hp.s4b0 _ hx.s4b0 (h14.congr_point (r50Pre14_apply N q w x))
  have h12 : HasGradAt (fun y => L (r50SufS3b4 N q w y)) (r50Pre12 N q w x) dy12 :=
    r50IdB_hasGradAt_comp w.s3b5 hp.s3b5 _ hx.s3b5 (h13.congr_point (r50Pre13_apply N q w x))
  have h11 : HasGradAt (fun y => L (r50SufS3b3 N q w y)) (r50Pre11 N q w x) dy11 :=
    r50IdB_hasGradAt_comp w.s3b4 hp.s3b4 _ hx.s3b4 (h12.congr_point (r50Pre12_apply N q w x))
  have h10 : HasGradAt (fun y => L (r50SufS3b2 N q w y)) (r50Pre10 N q w x) dy10 :=
    r50IdB_hasGradAt_comp w.s3b3 hp.s3b3 _ hx.s3b3 (h11.congr_point (r50Pre11_apply N q w x))
  have h9 : HasGradAt (fun y => L (r50SufS3b1 N q w y)) (r50Pre9 N q w x) dy9 :=
    r50IdB_hasGradAt_comp w.s3b2 hp.s3b2 _ hx.s3b2 (h10.congr_point (r50Pre10_apply N q w x))
  have h8 : HasGradAt (fun y => L (r50SufS3b0 N q w y)) (r50Pre8 N q w x) dy8 :=
    r50IdB_hasGradAt_comp w.s3b1 hp.s3b1 _ hx.s3b1 (h9.congr_point (r50Pre9_apply N q w x))
  have h7 : HasGradAt (fun y => L (r50SufS2b3 N q w y)) (r50Pre7 N q w x) dy7 :=
    r50DownB_hasGradAt_comp w.s3b0 hp.s3b0 _ hx.s3b0 (h8.congr_point (r50Pre8_apply N q w x))
  have h6 : HasGradAt (fun y => L (r50SufS2b2 N q w y)) (r50Pre6 N q w x) dy6 :=
    r50IdB_hasGradAt_comp w.s2b3 hp.s2b3 _ hx.s2b3 (h7.congr_point (r50Pre7_apply N q w x))
  have h5 : HasGradAt (fun y => L (r50SufS2b1 N q w y)) (r50Pre5 N q w x) dy5 :=
    r50IdB_hasGradAt_comp w.s2b2 hp.s2b2 _ hx.s2b2 (h6.congr_point (r50Pre6_apply N q w x))
  have h4 : HasGradAt (fun y => L (r50SufS2b0 N q w y)) (r50Pre4 N q w x) dy4 :=
    r50IdB_hasGradAt_comp w.s2b1 hp.s2b1 _ hx.s2b1 (h5.congr_point (r50Pre5_apply N q w x))
  have h3 : HasGradAt (fun y => L (r50SufS1b2 N q w y)) (r50Pre3 N q w x) dy3 :=
    r50DownB_hasGradAt_comp w.s2b0 hp.s2b0 _ hx.s2b0 (h4.congr_point (r50Pre4_apply N q w x))
  have h2 : HasGradAt (fun y => L (r50SufS1b1 N q w y)) (r50Pre2 N q w x) dy2 :=
    r50IdB_hasGradAt_comp w.s1b2 hp.s1b2 _ hx.s1b2 (h3.congr_point (r50Pre3_apply N q w x))
  have h1 : HasGradAt (fun y => L (r50SufS1b0 N q w y)) (r50Pre1 N q w x) dy1 :=
    r50IdB_hasGradAt_comp w.s1b1 hp.s1b1 _ hx.s1b1 (h2.congr_point (r50Pre2_apply N q w x))
  have h0 : HasGradAt (fun y => L (r50SufStem N q w y)) (r50Pre0 N q w x) cotPool :=
    r50ProjB_hasGradAt_comp w.s1b0 hp.s1b0 _ hx.s1b0 (h1.congr_point (r50Pre1_apply N q w x))
  refine ⟨r34_stem_lossTiedB xN cotN vN epsStr
      w.sW w.sb w.sε hp.s w.sγ w.sβ x hx.stem hx.pool (h0.congr_point (r50Pre0_apply N q w x))
      (fun W b γ β => by rw [r50_factor_stem]), ?_⟩
  refine ⟨r50_projblock_lossTiedB xN cotN vN epsStr w.s1b0 hp.s1b0 _ hx.s1b0
      (h1.congr_point (r50Pre1_apply N q w x)) (fun p => by rw [r50_factor_s1b0]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s1b1 hp.s1b1 _ hx.s1b1
      (h2.congr_point (r50Pre2_apply N q w x)) (fun p => by rw [r50_factor_s1b1]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s1b2 hp.s1b2 _ hx.s1b2
      (h3.congr_point (r50Pre3_apply N q w x)) (fun p => by rw [r50_factor_s1b2]), ?_⟩
  refine ⟨r50_downblock_lossTiedB xN cotN vN epsStr w.s2b0 hp.s2b0 _ hx.s2b0
      (h4.congr_point (r50Pre4_apply N q w x)) (fun p => by rw [r50_factor_s2b0]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s2b1 hp.s2b1 _ hx.s2b1
      (h5.congr_point (r50Pre5_apply N q w x)) (fun p => by rw [r50_factor_s2b1]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s2b2 hp.s2b2 _ hx.s2b2
      (h6.congr_point (r50Pre6_apply N q w x)) (fun p => by rw [r50_factor_s2b2]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s2b3 hp.s2b3 _ hx.s2b3
      (h7.congr_point (r50Pre7_apply N q w x)) (fun p => by rw [r50_factor_s2b3]), ?_⟩
  refine ⟨r50_downblock_lossTiedB xN cotN vN epsStr w.s3b0 hp.s3b0 _ hx.s3b0
      (h8.congr_point (r50Pre8_apply N q w x)) (fun p => by rw [r50_factor_s3b0]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s3b1 hp.s3b1 _ hx.s3b1
      (h9.congr_point (r50Pre9_apply N q w x)) (fun p => by rw [r50_factor_s3b1]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s3b2 hp.s3b2 _ hx.s3b2
      (h10.congr_point (r50Pre10_apply N q w x)) (fun p => by rw [r50_factor_s3b2]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s3b3 hp.s3b3 _ hx.s3b3
      (h11.congr_point (r50Pre11_apply N q w x)) (fun p => by rw [r50_factor_s3b3]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s3b4 hp.s3b4 _ hx.s3b4
      (h12.congr_point (r50Pre12_apply N q w x)) (fun p => by rw [r50_factor_s3b4]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s3b5 hp.s3b5 _ hx.s3b5
      (h13.congr_point (r50Pre13_apply N q w x)) (fun p => by rw [r50_factor_s3b5]), ?_⟩
  refine ⟨r50_downblock_lossTiedB xN cotN vN epsStr w.s4b0 hp.s4b0 _ hx.s4b0
      (h14.congr_point (r50Pre14_apply N q w x)) (fun p => by rw [r50_factor_s4b0]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s4b1 hp.s4b1 _ hx.s4b1
      (h15.congr_point (r50Pre15_apply N q w x)) (fun p => by rw [r50_factor_s4b1]), ?_⟩
  refine ⟨r50_idblock_lossTiedB xN cotN vN epsStr w.s4b2 hp.s4b2 _ hx.s4b2
      (h16.congr_point (r50Pre16_apply N q w x)) (fun p => by rw [r50_factor_s4b2]), ?_⟩
  exact r34_head_lossTiedB xN cotN w.Wd w.bd (r50Pre16 N q w x) hL'
    (fun W b => by rw [r50_factor_head])

-- ════════════════════════════════════════════════════════════════
-- § The two losses the artifacts ship
-- ════════════════════════════════════════════════════════════════

/-- **The `bce := false` artifacts**: every node is the derivative of the batched label-smoothed
    cross-entropy `smoothedBatchLoss`, `g` the six-op cotangent the render emits. -/
theorem r50_net_lossGrad_smoothedCE (N q : Nat) {nCls : Nat} (hK : 0 < nCls)
    (xN cotN vN epsStr aStr negAK bStr logN ohN : String) (α B : ℝ) (w : R50BWeights nCls)
    (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (hx : R50LossSmoothAtB N q w x) (t : Vec (N * (1 * nCls)))
    (ht : ∀ n, ∑ k : Fin nCls, targetRow N nCls t n k = 1) :
    R50NetLossTiedB N q xN cotN vN epsStr w x (smoothedBatchLoss N nCls α B t)
      (unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (resnet50ForwardBFull N q w x)) t))) :=
  r50_net_lossGrad N q xN cotN vN epsStr w hp x hx
    ⟨(smoothedBatchLoss_differentiable N nCls α B t) _,
      fun J => smoothedBatchLoss_grad N nCls hK α B aStr negAK bStr logN ohN t _ ht J⟩

/-- **The `bce := true` artifacts** (`resnet50in160_lambaccdp8x64wxclipbce` among them): every node is
    the derivative of the batched BCE-with-logits `bceBatchLoss`, the mean over `B×K`, `g` the
    three-op cotangent at the committed divisor `N·K`. No hypothesis on the target. -/
theorem r50_net_lossGrad_bce (N q : Nat) {nCls : Nat}
    (xN cotN vN epsStr bStr logN ohN : String) (w : R50BWeights nCls) (hp : R50PosB w)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (hx : R50LossSmoothAtB N q w x) (t : Vec (N * (1 * nCls))) :
    R50NetLossTiedB N q xN cotN vN epsStr w x (bceBatchLoss N nCls t)
      (unrowB N nCls (den (bceLossCotGraph N nCls ((N : ℝ) * (nCls : ℝ)) bStr logN ohN
        (rowB N nCls (resnet50ForwardBFull N q w x)) t))) :=
  r50_net_lossGrad N q xN cotN vN epsStr w hp x hx
    ⟨(bceBatchLoss_differentiable N nCls t) _,
      fun J => bceBatchLoss_grad N nCls bStr logN ohN t _ J⟩


/-- **The emitted ResNet-50 step's gradient nodes ARE the loss's gradient, at one chain.** For each
    of the 161 parameter slots, at ONE cotangent chain (the tie's own, from `g`): the node denotes
    its layer's Jacobian against the chain cotangent (`r50_net_tiedB`), and any loss `L` of the
    logits with gradient `g` at the network's output of `resnet50ForwardBFull` with that one slot
    varied is differentiable there with the node as its gradient (`r50_net_lossGrad`). The two
    theorems each state the chain; this one states it once, so an edit to either chain breaks its
    proof. -/
theorem r50_net_tied_lossGrad (N q : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) (g : Vec (N * nCls))
    (hp : R50PosB w) (hx : R50LossSmoothAtB N q w x) {L : Vec (N * nCls) → Vec 1}
    (hL : HasGradAt L (resnet50ForwardBFull N q w x) g) :
    let dy16 := r34HeadCotBlk N q q w.Wd w.bd (r50Pre16 N q w x) g
    let dy15 := r50IdCotIn N q q w.s4b2 (r50Pre15 N q w x) dy16
    let dy14 := r50IdCotIn N q q w.s4b1 (r50Pre14 N q w x) dy15
    let dy13 := r50DownCotIn N q q w.s4b0 (r50Pre13 N q w x) dy14
    let dy12 := r50IdCotIn N (2 * q) (2 * q) w.s3b5 (r50Pre12 N q w x) dy13
    let dy11 := r50IdCotIn N (2 * q) (2 * q) w.s3b4 (r50Pre11 N q w x) dy12
    let dy10 := r50IdCotIn N (2 * q) (2 * q) w.s3b3 (r50Pre10 N q w x) dy11
    let dy9 := r50IdCotIn N (2 * q) (2 * q) w.s3b2 (r50Pre9 N q w x) dy10
    let dy8 := r50IdCotIn N (2 * q) (2 * q) w.s3b1 (r50Pre8 N q w x) dy9
    let dy7 := r50DownCotIn N (2 * q) (2 * q) w.s3b0 (r50Pre7 N q w x) dy8
    let dy6 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 N q w x) dy7
    let dy5 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 N q w x) dy6
    let dy4 := r50IdCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 N q w x) dy5
    let dy3 := r50DownCotIn N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 N q w x) dy4
    let dy2 := r50IdCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Pre2 N q w x) dy3
    let dy1 := r50IdCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Pre1 N q w x) dy2
    let cotPool := r50ProjCotIn N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Pre0 N q w x) dy1
    (r34StemTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool
      ∧ r34StemLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x
        (fun W b γ β => L (resnet50ForwardBFull N q { w with sW := W, sb := b, sγ := γ, sβ := β } x))
        cotPool)
  ∧ (r50ProjTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b0 (r50Pre0 N q w x) dy1
      ∧ r50ProjLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.s1b0 (r50Pre0 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s1b0 := p } x)) dy1)
  ∧ (r50IdTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b1 (r50Pre1 N q w x) dy2
      ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.s1b1 (r50Pre1 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s1b1 := p } x)) dy2)
  ∧ (r50IdTiedB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr w.s1b2 (r50Pre2 N q w x) dy3
      ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * (2 * q)))) (w := (2 * (2 * (2 * q)))) xN cotN vN epsStr w.s1b2 (r50Pre2 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s1b2 := p } x)) dy3)
  ∧ (r50DownTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b0 (r50Pre3 N q w x) dy4
      ∧ r50DownLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b0 (r50Pre3 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s2b0 := p } x)) dy4)
  ∧ (r50IdTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b1 (r50Pre4 N q w x) dy5
      ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b1 (r50Pre4 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s2b1 := p } x)) dy5)
  ∧ (r50IdTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b2 (r50Pre5 N q w x) dy6
      ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b2 (r50Pre5 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s2b2 := p } x)) dy6)
  ∧ (r50IdTiedB N (2 * (2 * q)) (2 * (2 * q)) xN cotN vN epsStr w.s2b3 (r50Pre6 N q w x) dy7
      ∧ r50IdLossTiedB (N := N) (h := (2 * (2 * q))) (w := (2 * (2 * q))) xN cotN vN epsStr w.s2b3 (r50Pre6 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s2b3 := p } x)) dy7)
  ∧ (r50DownTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b0 (r50Pre7 N q w x) dy8
      ∧ r50DownLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b0 (r50Pre7 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s3b0 := p } x)) dy8)
  ∧ (r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b1 (r50Pre8 N q w x) dy9
      ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b1 (r50Pre8 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s3b1 := p } x)) dy9)
  ∧ (r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b2 (r50Pre9 N q w x) dy10
      ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b2 (r50Pre9 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s3b2 := p } x)) dy10)
  ∧ (r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b3 (r50Pre10 N q w x) dy11
      ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b3 (r50Pre10 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s3b3 := p } x)) dy11)
  ∧ (r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b4 (r50Pre11 N q w x) dy12
      ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b4 (r50Pre11 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s3b4 := p } x)) dy12)
  ∧ (r50IdTiedB N (2 * q) (2 * q) xN cotN vN epsStr w.s3b5 (r50Pre12 N q w x) dy13
      ∧ r50IdLossTiedB (N := N) (h := (2 * q)) (w := (2 * q)) xN cotN vN epsStr w.s3b5 (r50Pre12 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s3b5 := p } x)) dy13)
  ∧ (r50DownTiedB N q q xN cotN vN epsStr w.s4b0 (r50Pre13 N q w x) dy14
      ∧ r50DownLossTiedB (N := N) (h := q) (w := q) xN cotN vN epsStr w.s4b0 (r50Pre13 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s4b0 := p } x)) dy14)
  ∧ (r50IdTiedB N q q xN cotN vN epsStr w.s4b1 (r50Pre14 N q w x) dy15
      ∧ r50IdLossTiedB (N := N) (h := q) (w := q) xN cotN vN epsStr w.s4b1 (r50Pre14 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s4b1 := p } x)) dy15)
  ∧ (r50IdTiedB N q q xN cotN vN epsStr w.s4b2 (r50Pre15 N q w x) dy16
      ∧ r50IdLossTiedB (N := N) (h := q) (w := q) xN cotN vN epsStr w.s4b2 (r50Pre15 N q w x)
        (fun p => L (resnet50ForwardBFull N q { w with s4b2 := p } x)) dy16)
  ∧ (r34HeadTiedB N q q xN cotN w.Wd w.bd (r50Pre16 N q w x) g
      ∧ r34HeadLossTiedB (N := N) (h := q) (w := q) xN cotN w.Wd w.bd (r50Pre16 N q w x)
        (fun W b => L (resnet50ForwardBFull N q { w with Wd := W, bd := b } x)) g) := by
  intro dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotPool
  obtain ⟨t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17⟩ :=
    r50_net_tiedB N q xN cotN vN epsStr w x g
  have hl :=
    r50_net_lossGrad N q xN cotN vN epsStr w hp x hx hL
  obtain ⟨l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17⟩ := hl
  exact ⟨⟨t0, l0⟩, ⟨t1, l1⟩, ⟨t2, l2⟩, ⟨t3, l3⟩, ⟨t4, l4⟩, ⟨t5, l5⟩, ⟨t6, l6⟩, ⟨t7, l7⟩, ⟨t8, l8⟩,
    ⟨t9, l9⟩, ⟨t10, l10⟩, ⟨t11, l11⟩, ⟨t12, l12⟩, ⟨t13, l13⟩, ⟨t14, l14⟩, ⟨t15, l15⟩, ⟨t16, l16⟩,
    ⟨t17, l17⟩⟩

end Proofs.ResNet50TieB
