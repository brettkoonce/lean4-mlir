import LeanMlir.Proofs.Foundation.CertifiedChain
import LeanMlir.Proofs.Foundation.ResNet50BackNet
import LeanMlir.Proofs.Foundation.ResNet34BackB0
import LeanMlir.Proofs.Architectures.EfficientNetBackB0
import LeanMlir.Proofs.Architectures.ConvNeXtBackB0
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Architectures.EfficientNetChainClose

/-! # The remaining four nets, folded — `CertLayer` instances for r34, mnv2, enet, convnext

`ResNet50BackNet.lean` folded R50 and `CertifiedChain.lean` made the machinery net-agnostic. This
file pays that off: every other conv net's block capstone becomes a `CertLayer`, so
`CertLayer.comp` / `CertLayer.chain` compose them into stages and trunks with **no new proof per
net and no new proof per depth**.

⭐ **The per-net work was making the blocks pluggable, not proving anything.** Each capstone took
its cotangent as `dy : Vec n` and wrapped it internally as `.operand "%dy" dy`, so a block could
only ever be the LAST thing in a graph. All four now take `ecot : SHlo n` — strictly more general,
since the old statement is this one at `ecot := .operand "%dy" dy`. `residualBackGraph` was
generalized the same way, which is what let mnv2/enet/convnext follow.

## ⭐ Two tiers of layer, and the split is the ACTIVATION

| net | block activation | `ok` | why |
|---|---|---|---|
| **enet** | swish (smooth) | `True` | global `HasVJP`, lifted by `.toHasVJPAt` |
| **convnext** | gelu (smooth) | `True` | global `HasVJP` |
| **r34** | relu (one kink) | 2 clauses | `_at`: mid-relu + outer post-residual relu |
| **mnv2** | relu6 (two kinks) | 2 clauses | `_at`: `≠ 0 ∧ ≠ 6` at expand and depthwise |
| *(r50)* | relu | 3 clauses | 3 convs ⇒ 2 interior relus + the outer one |

⚠ A globally-smooth net is **not** a weaker certificate — it is a stronger one: `ok = True` means
the backward graph denotes the VJP *everywhere*, with no side condition to discharge. The `_at`
nets carry conditions because relu genuinely has no derivative at 0, and `CertLayer.comp` conjoins
those at the right activations rather than quietly dropping them.

## What this does NOT do

Stages and trunks are `comp`/`chain` applications, so they are available for all five nets — but
only R50 has them written out (`ResNet50BackNet.lean`), because a trunk needs the net's block table
and resolution ladder spelled out and that is per-net bookkeeping, not proof. ⚠ ViT is **not**
here: its blocks are per-token `Mat`-shaped with a different backward vocabulary
(`transformerBlockBackGraph` and three MH variants), and `ViTBackB0` is the heaviest module in the
repo (~11 min, ~14 GB — memory `vit-backb0-ci-cost`). It is a separate sitting.

✅ **That sitting happened — `Foundation/ViTBackNet.lean` (2026-08-10).** Two corrections it
forced, both worth reading before trusting this file's framing:
* ViT was never the *least*-folded net; it was the only one with a concrete whole-net backward
  graph (stem and head included, at every depth). See `CertifiedChain.lean`'s correction block.
* The ~11 min / ~14 GB figure is a **2-core CI runner** number. On a workstation `ViTBackB0`
  rebuilds in seconds, so "budget the CI cost" was not the constraint it looked like, and it is
  not a reason to defer work on this module.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § EfficientNet — swish is smooth, so `ok = True`
-- ════════════════════════════════════════════════════════════════

/-- The batched EfficientNet MBConv residual block as a `CertLayer`. ⭐ Globally certified
    (`ok = True`): swish and sigmoid are smooth, so the block has a global `HasVJP` and the
    backward graph denotes it at **every** input. An endomorphism, so `chain` iterates it. -/
noncomputable def enetMBConvLayer (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp
  ok := fun _ => True
  diff := fun x _ => (mbResidFwdB_differentiable N (h := h) (w := w) We be εe hεe γe βe
    Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp) x
  vjp := fun x _ => (mbResidFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).toHasVJPAt x
  graph := fun x e => mbResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e
  faithful := fun x _ e => mbResidBlockBackBatchedGraph_faithful We be εe hεe γe βe
    Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x e

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt — gelu is smooth, so `ok = True`
-- ════════════════════════════════════════════════════════════════

/-- The per-example ConvNeXt residual block as a `CertLayer` (spatial-LN form). Globally
    certified. ⚠ ConvNeXt is per-example (batch-1): LayerNorm is separable across examples, so
    there is no batched machinery to carry — see `ConvNeXtBackB0`'s header. -/
noncomputable def cnxBlockLayer {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    CertLayer (c * h * w) (c * h * w) where
  fwd := convNextBlock Wdw bdw εn γn βn Wex bex Wpr bpr γls
  ok := fun _ => True
  diff := fun x _ =>
    (convNextBlock_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls) x
  vjp := fun x _ =>
    (convNextBlock_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls).toHasVJPAt x
  graph := fun x e => cnxResidBlockBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x e
  faithful := fun x _ e =>
    cnxResidBlockBackGraph_faithful Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls x e

/-- ⭐ The **channel-LN** ConvNeXt block as a `CertLayer` — the form the *shipped* net's stages are
    actually built from (`cnxResidBlockChBackGraph_faithful` is described in `ConvNeXtBackB0` as
    "the capstone the shipped net was missing"). This is the one to chain for a real ConvNeXt
    stage; `cnxBlockLayer` above is the spatial-LN sibling. -/
noncomputable def cnxBlockChLayer {c cExp h w kH kW : Nat}
    (p : CnxBlockParamsCh c cExp h w kH kW) (hε : 0 < p.εn) :
    CertLayer (c * h * w) (c * h * w) where
  fwd := cnxBlockChW p
  ok := fun _ => True
  diff := fun x _ => (cnxBlockChW_diff p hε) x
  vjp := fun x _ => (cnxBlockChW_has_vjp p hε).toHasVJPAt x
  graph := fun x e => cnxResidBlockChBackGraph p x e
  faithful := fun x _ e => cnxResidBlockChBackGraph_faithful p hε x e

-- ════════════════════════════════════════════════════════════════
-- § ResNet-34 — relu, so `_at` with TWO smoothness clauses
-- ════════════════════════════════════════════════════════════════

/-- The batched R34 identity basic block as a `CertLayer`. Two `ok` clauses: the body's mid-relu
    and the OUTER post-residual relu — the extra factor R34 has over the MBConv/inverted-residual
    blocks. An endomorphism, so `chain` iterates it into a stage tail. -/
noncomputable def r34BasicBlockLayer (N : Nat) {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := relu (N * (c * h * w)) ∘
    residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)
  ok := fun x =>
    (∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) ∧
    (∀ k, residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
            cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0)
  diff := by
    intro x hx
    exact (relu_differentiableAt_of_smooth _ _ hx.2).comp x
      ((r34BodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x hx.1).add
        differentiable_id.differentiableAt)
  vjp := fun x hx =>
    r34BasicBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x hx.1 hx.2
  graph := fun x e => r34BasicBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x e
  faithful := fun x hx e =>
    r34BasicBlockBackBatchedGraph_faithful W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x e hx.1 hx.2

/-- The batched R34 downsample basic block as a `CertLayer`. Halves resolution (hence the `2*h` in
    the input type), with a strided conv1 and a strided projection skip. -/
noncomputable def r34DownBlockLayer (N : Nat) {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) where
  fwd := relu (N * (oc * h * w)) ∘
    residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
      (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
       cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)
  ok := fun x =>
    (∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0) ∧
    (∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0)
  diff := by
    intro x hx
    exact (relu_differentiableAt_of_smooth _ _ hx.2).comp x
      (((projStridedB_differentiable N Wp bp εp hεp γp βp) x).add
        (r34DownBodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x hx.1))
  vjp := fun x hx =>
    r34DownBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      Wp bp εp hεp γp βp x hx.1 hx.2
  graph := fun x e => r34DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
    Wp bp εp γp βp x e
  faithful := fun x hx e =>
    r34DownBlockBackBatchedGraph_faithful W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      Wp bp εp hεp γp βp x e hx.1 hx.2

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV2 — relu6, so `_at` with TWO-SIDED clauses
-- ════════════════════════════════════════════════════════════════

/-- The batched MobileNetV2 inverted-residual block as a `CertLayer`.

    ⚠ Its `ok` clauses are **two-sided** (`≠ 0 ∧ ≠ 6`) because relu6 has a kink at each end — the
    difference `MobileNetV2BackB0` records against R34's one-sided relu. ⚠ And unlike R34/R50 there
    is **no outer relu**: mnv2's block output IS the residual add, so `ok` has two clauses covering
    the expand and depthwise stages and none for an output activation. -/
noncomputable def mnv2ResidBlockLayer (N : Nat) {c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := residual (projB N (h := h) (w := w) Wp bp εp γp βp ∘
                   dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘
                   cbrB N (h := h) (w := w) We be εe γe βe)
  ok := fun x =>
    (∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
          bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6) ∧
    (∀ k, bnBatchLA N mid h w εd γd βd
            (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
          bnBatchLA N mid h w εd γd βd
            (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6)
  diff := by
    intro x hx
    exact (mnv2BodyB_differentiableAt N We be εe hεe γe βe Wd bd εd hεd γd βd
      Wp bp εp hεp γp βp x hx.1 hx.2).add differentiable_id.differentiableAt
  vjp := fun x hx =>
    residual_has_vjp_at _ x
      (mnv2BodyB_differentiableAt N We be εe hεe γe βe Wd bd εd hεd γd βd
        Wp bp εp hεp γp βp x hx.1 hx.2)
      (mnv2BodyB_has_vjp_at N We be εe hεe γe βe Wd bd εd hεd γd βd
        Wp bp εp hεp γp βp x hx.1 hx.2)
  graph := fun x e => mnv2ResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd
    Wp bp εp γp βp x e
  faithful := fun x hx e =>
    mnv2ResidBlockBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
      Wp bp εp hεp γp βp x e hx.1 hx.2

-- ════════════════════════════════════════════════════════════════
-- § The payoff — every net folds with the SAME two combinators
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **A chain of MBConv blocks is certified, at any depth.** Immediate from
    `CertLayer.chain_faithful`; stated per net only to make the payoff visible. The same one-liner
    works for every layer in this file, which is what "the machinery is net-agnostic" means. -/
theorem enetChain_faithful {N c mid h w kHd kWd r : Nat}
    (Ls : List (CertLayer (N * (c * h * w)) (N * (c * h * w))))
    (x : Vec (N * (c * h * w))) (hx : (CertLayer.chain Ls).ok x)
    (e : SHlo (N * (c * h * w))) :
    den ((CertLayer.chain Ls).graph x e) = ((CertLayer.chain Ls).vjp x hx).backward (den e) :=
  CertLayer.chain_faithful Ls x hx e

/-- An R34 stage: the downsample block, then any number of identity blocks. The R34 peer of
    `r50StageDown`, and the same two `comp`/`chain` calls. -/
noncomputable def r34Stage {m n : Nat}
    (P : CertLayer m n) (tail : List (CertLayer n n)) : CertLayer m n :=
  P.comp (CertLayer.chain tail)

/-- **R34's block table, as a type-level check**: stages are 3/4/6/3 = one entry block plus
    2/3/5/2 identity blocks — the same counts as R50, since the two nets differ in block *form*
    (basic vs bottleneck), not in depth per stage. -/
noncomputable def r34Trunk_3463 {N c₀ c₁ c₂ c₃ c₄ h w : Nat}
    (P1 : CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w))))))
    (I1 : CertLayer (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w))))))
    (P2 : CertLayer (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₂ * (2*(2*h)) * (2*(2*w)))))
    (I2 : CertLayer (N * (c₂ * (2*(2*h)) * (2*(2*w)))) (N * (c₂ * (2*(2*h)) * (2*(2*w)))))
    (P3 : CertLayer (N * (c₂ * (2*(2*h)) * (2*(2*w)))) (N * (c₃ * (2*h) * (2*w))))
    (I3 : CertLayer (N * (c₃ * (2*h) * (2*w))) (N * (c₃ * (2*h) * (2*w))))
    (P4 : CertLayer (N * (c₃ * (2*h) * (2*w))) (N * (c₄ * h * w)))
    (I4 : CertLayer (N * (c₄ * h * w)) (N * (c₄ * h * w))) :
    CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w))))) (N * (c₄ * h * w)) :=
  (r34Stage P1 (List.replicate 2 I1)).comp
    ((r34Stage P2 (List.replicate 3 I2)).comp
      ((r34Stage P3 (List.replicate 5 I3)).comp (r34Stage P4 (List.replicate 2 I4))))

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ R34 WEIGHT WIRING — parameters TYPED BY THEIR BLOCK ROW
-- ════════════════════════════════════════════════════════════════

/-! The R34 peer of MNv4's `UibParams` and R50's `BottleneckParams`: a row type, and a parameter
record whose every width is a projection of the row, so a record disagreeing with its row cannot be
constructed. ⚠ Same caveat as the other two — typing pins **shape**, not **identity**: two rows of
the same shape have the same record type, so a stage's identity-block repeats stay interchangeable.
Closing that wants weights drawn from one indexed array in row order (a renderer concern). -/

/-- One R34 basic-block row: `ic → oc` at output resolution `h`, optionally strided. ⚠ A basic
    block has no bottleneck width — that single missing field is the whole difference from
    `BottleneckSpec`, and it is why R34 and R50 share stage depths but not block forms. -/
structure BasicSpec where
  ic : Nat
  oc : Nat
  h : Nat
  stride2 : Bool
deriving Repr, DecidableEq

/-- **One basic block's parameters, typed by its row.** Two 3×3 convs (extents left as binders) and
    a projection for the downsample form. -/
structure BasicParams (s : BasicSpec) (k1 k2 kp : Nat) where
  W1 : Kernel4 s.oc s.ic k1 k1
  b1 : Vec s.oc
  e1 : ℝ
  h1 : 0 < e1
  g1 : Vec s.oc
  bt1 : Vec s.oc
  W2 : Kernel4 s.oc s.oc k2 k2
  b2 : Vec s.oc
  e2 : ℝ
  h2 : 0 < e2
  g2 : Vec s.oc
  bt2 : Vec s.oc
  Wp : Kernel4 s.oc s.ic kp kp
  bp : Vec s.oc
  ep : ℝ
  hp : 0 < ep
  gp : Vec s.oc
  btp : Vec s.oc

/-- ⭐ **The R34 downsample basic block, built from its row.** Input resolution is `2 * s.h`; the
    row records the OUTPUT size, matching `UibSpec` and `BottleneckSpec`. -/
noncomputable def r34DownBlockOfRow (N : Nat) (s : BasicSpec) {k1 k2 kp : Nat}
    (p : BasicParams s k1 k2 kp) :
    CertLayer (N * (s.ic * (2 * s.h) * (2 * s.h))) (N * (s.oc * s.h * s.h)) :=
  r34DownBlockLayer (h := s.h) (w := s.h) N
    p.W1 p.b1 p.e1 p.h1 p.g1 p.bt1 p.W2 p.b2 p.e2 p.h2 p.g2 p.bt2
    p.Wp p.bp p.ep p.hp p.gp p.btp

/-- ⭐ **The R34 block table's stage entries**, the peer of `r50StageEntries`. ⚠ Stage 1's entry is
    an IDENTITY block (`64 → 64`, stride 1), which is exactly why R34 never needed R50's stride-1
    projection form (§8a) — the one structural difference between the two nets' stage-entry rows. -/
def r34StageEntries : List BasicSpec :=
  [ ⟨ 64,  64, 56, false⟩,   -- stage 1 block 0: identity, NOT a projection
    ⟨ 64, 128, 28, true ⟩,   -- stage 2 block 0
    ⟨128, 256, 14, true ⟩,   -- stage 3 block 0
    ⟨256, 512,  7, true ⟩ ]  -- stage 4 block 0

-- The 56→28→14→7 ladder and the channel doubling, as checks rather than prose.
#guard r34StageEntries.map (·.h) = [56, 28, 14, 7]
#guard r34StageEntries.map (·.stride2) = [false, true, true, true]
-- ⭐ Exactly one stage entry preserves channels — stage 1 — and it is the non-strided one. That is
-- the whole reason R50 needs a block form R34 does not.
#guard (r34StageEntries.filter (fun s => s.ic == s.oc)).length = 1
#guard r34StageEntries.all (fun s => s.stride2 || s.ic == s.oc)
-- R34 and R50 share stage depths (3/4/6/3) and the resolution ladder, differing only in block form.
#guard r34StageEntries.map (·.h) = r50StageEntries.map (·.h)

end Proofs.StableHLO
