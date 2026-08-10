import LeanMlir.Proofs.Foundation.ResNet50BackB0
import LeanMlir.Proofs.Foundation.CertifiedChain

/-! # R50's NET-level backward fold — the block capstones chained into stages and a trunk

`ResNet50BackB0.lean` proves the three *block* capstones. This file folds them: each block becomes
a `CertLayer` (`Foundation/CertifiedChain.lean`), and stages and the trunk are `CertLayer.comp` /
`CertLayer.chain` of those. Faithfulness at every level is then **`CertLayer.faithful` applied to
the composite** — no per-depth proof, and nothing re-derived.

## ⚠ WHAT "WHOLE-NET COMPOSED BACKWARD" MEANT BEFORE THIS

Measured 2026-08-10: **no net in the repo folded its blocks.** `ResNet34BackB0`,
`MobileNetV2BackB0`, `EfficientNetBackB0` and `ConvNeXtBackB0` all stop at a block capstone, so
§8's ✓ column in `planning/mnv4_verified.md` means *block* capstones for all five of the nets it
credits. R50 was already at parity when `ResNet50BackB0` landed; this file is the step past it, and
the machinery is deliberately net-agnostic so the other five can follow.

## What made it possible, and it was not a proof

The blocks had to become **pluggable**. `r50…BackBatchedGraph` originally took the incoming
cotangent as a `dy : Vec n` and wrapped it internally as `.operand "%dy" dy`, so nothing could be
fed upstream of a block — a block could only ever be the last thing in a graph. They now take
`ecot : SHlo n`, which is strictly more general (the old statement is this one at
`ecot := .operand "%dy" dy`) and is what lets `CertLayer.comp` nest them.

⭐ The composition proof itself is `CertLayer.comp`, written once. Every
`<body>BackBatchedGraph_faithful` in the repo is a hand-instance of it.

## The smoothness bookkeeping is the real content

R50 is relu throughout, so every block is `_at` — certified only where its pre-activations miss the
kinks. Composition conjoins those conditions **at the right activations**:
`ok x = L₁.ok x ∧ L₂.ok (L₁.fwd x)`. For a 16-block trunk that is a 48-clause hypothesis (3 relu
families per bottleneck) threaded through 16 successive activations, which is precisely the
bookkeeping nobody wants to write by hand and exactly what `comp` generates.

⚠ So `(r50Trunk …).ok x` is **not** a formality — it is the honest statement that the certificate
holds where the net is differentiable, and it deepens correctly rather than being assumed away.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The three block forms, as CertLayers
-- ════════════════════════════════════════════════════════════════

/-- The identity bottleneck as a `CertLayer`. ⭐ **An endomorphism** (`ic = oc`, resolution
    unchanged), which is what lets `CertLayer.chain` iterate it — a stage tail is n of these. -/
noncomputable def r50BottleneckLayer (N : Nat) {c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := relu (N * (c * h * w)) ∘
    residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
              cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)
  ok := fun x =>
    (∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) ∧
    (∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
        (batchMap N (flatConv W₂ b₂) (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) ∧
    (∀ k, residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
            cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
            cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0)
  diff := by
    intro x hx
    exact (relu_differentiableAt_of_smooth _ _ hx.2.2).comp x
      ((r50BodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x hx.1 hx.2.1).add differentiable_id.differentiableAt)
  vjp := fun x hx =>
    r50BottleneckB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      W₃ b₃ ε₃ hε₃ γ₃ β₃ x hx.1 hx.2.1 hx.2.2
  graph := fun x e => r50BottleneckBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
    W₃ b₃ ε₃ γ₃ β₃ x e
  faithful := fun x hx e =>
    r50BottleneckBackBatchedGraph_faithful W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      W₃ b₃ ε₃ hε₃ γ₃ β₃ x e hx.1 hx.2.1 hx.2.2

/-- ⭐ The **stride-1 projection** bottleneck as a `CertLayer` — R50 stage 1 block 0, the form with
    no R34 analogue. Changes channels, keeps resolution, so it is NOT an endomorphism and composes
    via `comp` rather than `chain`. -/
noncomputable def r50ProjBlockLayer (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) where
  fwd := relu (N * (oc * h * w)) ∘
    residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
      (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
       cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
       cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)
  ok := fun x =>
    (∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) ∧
    (∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
        (batchMap N (flatConv W₂ b₂) (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) ∧
    (∀ k, residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
             cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0)
  diff := by
    intro x hx
    exact (relu_differentiableAt_of_smooth _ _ hx.2.2).comp x
      (((projB_differentiable N Wp bp εp hεp γp βp) x).add
        (r50BodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x hx.1 hx.2.1))
  vjp := fun x hx =>
    r50ProjBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x hx.1 hx.2.1 hx.2.2
  graph := fun x e => r50ProjBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
    W₃ b₃ ε₃ γ₃ β₃ Wp bp εp γp βp x e
  faithful := fun x hx e =>
    r50ProjBlockBackBatchedGraph_faithful W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x e hx.1 hx.2.1 hx.2.2

/-- The **strided projection** bottleneck as a `CertLayer` — stages 2/3/4, block 0. Halves the
    resolution, which is why its input type carries `2*h`/`2*w`. ⚠ The stride is on the 3×3
    (`cbReluStridedB` at `W₂`), so `h_s1` is stated at the input resolution and `h_s2` at the
    output one. -/
noncomputable def r50DownBlockLayer (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) where
  fwd := relu (N * (oc * h * w)) ∘
    residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
      (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
       cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
       cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁)
  ok := fun x =>
    (∀ k, bnBatchLA N mid (2 * h) (2 * w) ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) ∧
    (∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
        (batchMap N (flatConvStride2 W₂ b₂)
          (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) ∧
    (∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
             cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0)
  diff := by
    intro x hx
    exact (relu_differentiableAt_of_smooth _ _ hx.2.2).comp x
      (((projStridedB_differentiable N Wp bp εp hεp γp βp) x).add
        (r50DownBodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x hx.1 hx.2.1))
  vjp := fun x hx =>
    r50DownBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x hx.1 hx.2.1 hx.2.2
  graph := fun x e => r50DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
    W₃ b₃ ε₃ γ₃ β₃ Wp bp εp γp βp x e
  faithful := fun x hx e =>
    r50DownBlockBackBatchedGraph_faithful W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
      W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x e hx.1 hx.2.1 hx.2.2

-- ════════════════════════════════════════════════════════════════
-- § Stages — a projection block followed by ANY number of identity blocks
-- ════════════════════════════════════════════════════════════════

/-- **R50 stage 1**: the stride-1 projection block, then its identity tail. Resolution is constant
    through this stage (56×56 at 224² input). -/
noncomputable def r50StageFirst {m n : Nat}
    (P : CertLayer m n) (tail : List (CertLayer n n)) : CertLayer m n :=
  P.comp (CertLayer.chain tail)

/-- **R50 stages 2/3/4**: the strided projection block, then its identity tail. Halves resolution
    at the projection block and holds it through the tail. -/
noncomputable def r50StageDown {m n : Nat}
    (P : CertLayer m n) (tail : List (CertLayer n n)) : CertLayer m n :=
  P.comp (CertLayer.chain tail)

/-- ⭐ **A stage's backward graph denotes its VJP, at any depth.** Immediate from
    `CertLayer.faithful` — which is the entire point of the fold: this needed no induction on the
    tail length and no new proof. -/
theorem r50Stage_faithful {m n : Nat} (P : CertLayer m n) (tail : List (CertLayer n n))
    (x : Vec m) (hx : (r50StageFirst P tail).ok x) (e : SHlo n) :
    den ((r50StageFirst P tail).graph x e)
      = ((r50StageFirst P tail).vjp x hx).backward (den e) :=
  (r50StageFirst P tail).faithful x hx e

-- ════════════════════════════════════════════════════════════════
-- § The trunk — all four stages
-- ════════════════════════════════════════════════════════════════

/-- **The R50 trunk**: stage 1 (stride-1) then three halving stages.

    ⚠ The resolutions are written as nested doublings (`2*(2*(2*h))`) rather than `8*h` on purpose:
    Nat multiplication is not definitionally associative in a variable, so `2*(2*(2*h))` and `8*h`
    are propositionally but NOT definitionally equal, and the stage types would fail to line up.
    `h`/`w` here are the FINAL resolution (7×7 for R50 at 224²). -/
noncomputable def r50Trunk {N c₀ c₁ c₂ c₃ c₄ h w : Nat}
    (S1 : CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w))))))
    (S2 : CertLayer (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₂ * (2*(2*h)) * (2*(2*w)))))
    (S3 : CertLayer (N * (c₂ * (2*(2*h)) * (2*(2*w)))) (N * (c₃ * (2*h) * (2*w))))
    (S4 : CertLayer (N * (c₃ * (2*h) * (2*w))) (N * (c₄ * h * w))) :
    CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w))))) (N * (c₄ * h * w)) :=
  S1.comp (S2.comp (S3.comp S4))

/-- ⭐⭐ **THE NET-LEVEL THEOREM.** The whole R50 trunk's backward graph denotes the trunk's VJP,
    wherever every constituent block is smooth.

    Sixteen bottleneck blocks — 48 relu smoothness families threaded through 16 successive
    activations — and the proof is `CertLayer.faithful`. That is what the fold buys: depth is free
    once composition is proven once. -/
theorem r50Trunk_faithful {N c₀ c₁ c₂ c₃ c₄ h w : Nat}
    (S1 : CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w))))))
    (S2 : CertLayer (N * (c₁ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                    (N * (c₂ * (2*(2*h)) * (2*(2*w)))))
    (S3 : CertLayer (N * (c₂ * (2*(2*h)) * (2*(2*w)))) (N * (c₃ * (2*h) * (2*w))))
    (S4 : CertLayer (N * (c₃ * (2*h) * (2*w))) (N * (c₄ * h * w)))
    (x : Vec (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w))))))
    (hx : (r50Trunk S1 S2 S3 S4).ok x) (e : SHlo (N * (c₄ * h * w))) :
    den ((r50Trunk S1 S2 S3 S4).graph x e)
      = ((r50Trunk S1 S2 S3 S4).vjp x hx).backward (den e) :=
  (r50Trunk S1 S2 S3 S4).faithful x hx e

/-- **R50's actual block table, as a type-level check.** Stages are 3/4/6/3 blocks = one projection
    block plus 2/3/5/2 identity blocks. If the arities or resolutions were wrong this would not
    elaborate. -/
noncomputable def r50Trunk_3463 {N c₀ c₁ c₂ c₃ c₄ h w : Nat}
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
  r50Trunk
    (r50StageFirst P1 (List.replicate 2 I1))
    (r50StageDown  P2 (List.replicate 3 I2))
    (r50StageDown  P3 (List.replicate 5 I3))
    (r50StageDown  P4 (List.replicate 2 I4))

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ WEIGHT WIRING — parameters TYPED BY THEIR BLOCK ROW
-- ════════════════════════════════════════════════════════════════

/-! ⛔ **The gap.** The block layers above take their weights as free arguments, so nothing stops a
caller pairing stage 2's widths with stage 3's kernels. `r50Trunk_3463` pins arities and
resolutions; it does not pin the weights.

⭐ **Fix, the same shape MNv4 uses**: a row type, and a parameter record whose every width is a
*projection of the row*. A record disagreeing with its row cannot be constructed.

⚠ **What it does not pin** (same caveat as MNv4's): two rows with identical shape have identical
record types, so their weights are interchangeable to the type checker. Typing pins **shape**, not
**identity**. In R50 that bites within a stage — a stage's identity blocks are all
`oc → oc/4 → oc`, so its 2/3/5/2 repeats are mutually swappable. Closing that wants the weights
drawn from one indexed array in row order, which is a renderer concern. -/

/-- One R50 bottleneck row: `ic → mid → oc` at output resolution `h`, optionally strided.
    ⚠ `mid` is explicit rather than `oc / 4` because the torchvision ratio is a fact about the
    block table, not about the block form — and baking a division into the type would make every
    width proof carry it. -/
structure BottleneckSpec where
  ic : Nat
  mid : Nat
  oc : Nat
  h : Nat
  stride2 : Bool
deriving Repr, DecidableEq

/-- **One bottleneck's parameters, typed by its row.** `kq`/`kd`/`kz`/`kp` are the four kernel
    extents (1×1, 3×3, 1×1, and the projection's), left as binders exactly as
    `Resnet50BlocksCertified` argues they should be — 1×1-vs-3×3 is an argument, not a literal. -/
structure BottleneckParams (s : BottleneckSpec) (kq kd kz kp : Nat) where
  W1 : Kernel4 s.mid s.ic kq kq
  b1 : Vec s.mid
  e1 : ℝ
  h1 : 0 < e1
  g1 : Vec s.mid
  bt1 : Vec s.mid
  W2 : Kernel4 s.mid s.mid kd kd
  b2 : Vec s.mid
  e2 : ℝ
  h2 : 0 < e2
  g2 : Vec s.mid
  bt2 : Vec s.mid
  W3 : Kernel4 s.oc s.mid kz kz
  b3 : Vec s.oc
  e3 : ℝ
  h3 : 0 < e3
  g3 : Vec s.oc
  bt3 : Vec s.oc
  Wp : Kernel4 s.oc s.ic kp kp
  bp : Vec s.oc
  ep : ℝ
  hp : 0 < ep
  gp : Vec s.oc
  btp : Vec s.oc

/-- ⭐ **The stride-1 projection bottleneck, built from its row** (R50 stage 1 block 0). Widths and
    resolution come from `s`; only the numeric values are free. -/
noncomputable def r50ProjBlockOfRow (N : Nat) (s : BottleneckSpec) {kq kd kz kp : Nat}
    (p : BottleneckParams s kq kd kz kp) :
    CertLayer (N * (s.ic * s.h * s.h)) (N * (s.oc * s.h * s.h)) :=
  r50ProjBlockLayer (h := s.h) (w := s.h) N
    p.W1 p.b1 p.e1 p.h1 p.g1 p.bt1 p.W2 p.b2 p.e2 p.h2 p.g2 p.bt2
    p.W3 p.b3 p.e3 p.h3 p.g3 p.bt3 p.Wp p.bp p.ep p.hp p.gp p.btp

/-- ⭐ **The strided projection bottleneck, built from its row** (stages 2/3/4, block 0). The input
    resolution is `2 * s.h` — the row records the OUTPUT size, as MNv4's `UibSpec` does. -/
noncomputable def r50DownBlockOfRow (N : Nat) (s : BottleneckSpec) {kq kd kz kp : Nat}
    (p : BottleneckParams s kq kd kz kp) :
    CertLayer (N * (s.ic * (2 * s.h) * (2 * s.h))) (N * (s.oc * s.h * s.h)) :=
  r50DownBlockLayer (h := s.h) (w := s.h) N
    p.W1 p.b1 p.e1 p.h1 p.g1 p.bt1 p.W2 p.b2 p.e2 p.h2 p.g2 p.bt2
    p.W3 p.b3 p.e3 p.h3 p.g3 p.bt3 p.Wp p.bp p.ep p.hp p.gp p.btp

/-- ⭐ **The R50 block table**, as data the builders can read — the peer of `mnv4Blocks`. Stage
    entry blocks only (the identity repeats are `2/3/5/2` of the stage's `oc → oc/4 → oc` shape).
    ⚠ `h` is each block's OUTPUT resolution at 224² input. -/
def r50StageEntries : List BottleneckSpec :=
  [ ⟨  64,  64,  256, 56, false⟩,   -- stage 1 block 0: stride-1 projection, the R34-less form
    ⟨ 256, 128,  512, 28, true ⟩,   -- stage 2 block 0
    ⟨ 512, 256, 1024, 14, true ⟩,   -- stage 3 block 0
    ⟨1024, 512, 2048,  7, true ⟩ ]  -- stage 4 block 0

-- The torchvision bottleneck ratio `mid = oc / 4`, and the 56→28→14→7 ladder, as checks on the
-- table rather than prose. A wrong width here is §3's silent class one level up.
#guard r50StageEntries.all (fun s => s.mid * 4 == s.oc)
#guard r50StageEntries.map (·.h) = [56, 28, 14, 7]
#guard r50StageEntries.map (·.stride2) = [false, true, true, true]
-- Exactly one stride-1 projection — the form §8a records as having no R34 analogue.
#guard (r50StageEntries.filter (fun s => !s.stride2)).length = 1
-- Every stage entry changes channels, which is why none of the four has an identity skip.
#guard r50StageEntries.all (fun s => s.ic != s.oc)

end Proofs.StableHLO
