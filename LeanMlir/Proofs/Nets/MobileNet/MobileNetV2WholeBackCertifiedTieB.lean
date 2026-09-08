import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2PaperWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains

/-! # ⭐⭐ `mnv2InputGradB` IS the certified whole-net MobileNetV2 gradient AT BATCH BATCH-NORM

`MobileNetV2PaperWholeBackCertifiedTie.lean` closed this for the PER-EXAMPLE seventeen-bottleneck
net — the reverse of `mobilenetv2ForwardPaper`, the forward the retired `MobileNetV2Render`
emitted. This file closes it for the net the shipped trainers run: `mobilenetv2ForwardB_full`, the
same `[t,c,n,s]` ladder at **`bnBatchLA`**, at a variable batch `N`. It is tier **T6** of
`planning/archive/proofs_tier_to_paper_nets.md` §4.2, alongside `ResNet34BackCertifiedTieB.lean`.

## ⭐⭐ The whole apex is reused, not rewritten

`mobilenetv2PaperPC_has_vjp_at` — the per-example file's twenty-one-stage chain — is generic in
every dimension and in every stage, so the batched net instantiates it directly: `stem` at
`mnv2StemB`, the seventeen bottlenecks at the batched block maps, and the head's three stages at
`cbrB` / `batchMap gap` / `batchMap dense`. `opaqueA0 … A17` come with it. So this file
defines **no apex and no prefix defs at all** — where ResNet-34's peer had to write both, because
its committed apex bundles the stem's pool into `stem` and its head into one stage.

## The three pieces

1. `mnv2StemBBack_eq_vjp_backward` and `cbrBBack_eq_vjp_backward` — the two concrete
   conv-BN-relu6 endpoints, one `rw` of a conv leaf tie and then `rfl` each. ⚠ The stem's is the
   XLA-`SAME` leaf (`flatConvStride2XlaBack`) and the head's the plain one; that is the one
   convention this net and ResNet-34 do not share.
2. `mnv2InputGradB_eq_mobilenetv2B_full_vjp` and `mnv2InputGradB_correct` — the tie, and its
   reading as `∑ pdiv … * dy`: the chain IS the Jacobian-transpose of the twenty-one-stage
   composition, at every batch size.
3. `mobilenetv2ForwardB_full_eq_slots` — the shape check: those twenty-one stages ARE
   `mobilenetv2ForwardB_full`, the forward `mobilenetv2FwdGraphB_full_faithful` (4.2b) says the
   typed graph denotes. Without it the tie would be a statement about variables.

⛔ **Why the blocks stay opaque, measured on ResNet-34's peer.** Instantiating a tie of this shape
at the concrete blocks is a *kernel* deterministic timeout: the block witnesses are `HasVJPAt` at
`opaqueA{k-1} … x` and a caller's are at `mnv2PreB{k-1} N w x`, which is seventeen defeq
checks between seventeen-deep nested applications spelled through different definition chains.
B0's file takes that step only because swish has no kink, so its witnesses are GLOBAL `HasVJP`
and carry no point at all. The shape check is what replaces it, and it is the same answer the
per-example file gave.

⚠ It stays a SMOOTH-POINT statement: relu6 is kinked on BOTH sides, so each of the 35 sites
carries `≠ 0 ∧ ≠ 6`. ⛔ MobileNetV2's two clauses per block are the expand relu6 and the
depthwise relu6, both INSIDE the body — not ResNet-34's mid-relu and post-residual outer relu.
Those are 4.2b's bundles, reused verbatim; this file adds no hypothesis of its own.

⛔ **What this does NOT reach.** Every gradient node in `mobilenetv2in_rmsdp64` is followed by an
all-reduce emitted as text outside the AST, so this is at the per-replica gradient (§4d). And it
is about the INPUT gradient; the parameter gradients are `MobileNetV2StepTieB.lean`'s tie (§4.2c).
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The two concrete conv-BN-relu6 endpoint ties
-- ════════════════════════════════════════════════════════════════

/-- **The STEM tie.** `batchMap (flatConvStride2XlaBack) ∘ bnBack ∘ reluMaskBack` IS `mnv2StemB`'s
    certified backward at a smooth point. One `rw` of the odd-kernel XLA-`SAME` strided leaf tie,
    then `rfl` — the stage's VJP is `vjp_comp_at`-built so its backward is already the
    composition, and a convolution's backward ignores its primal argument, so the row-wise
    `batchMap` lift matches at every saved input. -/
theorem mnv2StemBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : MNV2StemSmoothAtB N h w Ws bs ε γ β x) :
    (StableHLO.batchMap N (flatConvStride2XlaBack (h := h) (w := w) Ws)
        ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x)
        ∘ reluMaskBack (fun i =>
            0 < StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i ∧
            StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i < 6))
      = (mnv2StemB_has_vjp_at N h w Ws bs ε hε γ β x hs).backward := by
  rw [flatConvStride2XlaBack_eq_vjp_backward hkH hkW Ws bs (fun _ => 0)]
  rfl

/-- **The HEAD's conv-BN-relu6 tie.** `batchMap (convFlatBack) ∘ bnBack ∘ reluMaskBack` IS
    `cbrB`'s certified backward at a smooth point — the stride-1 peer of the stem's, at the plain
    (non-XLA) convolution leaf. ⚠ MobileNetV2's head is NOT hypothesis-free, unlike ResNet-34's:
    it puts this relu6 in front of the pool, so the net's 35th kink site is here. -/
theorem cbrBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Wh : Kernel4 oc ic kH kW) (bh : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (ic * h * w)))
    (hs : ∀ k, StableHLO.bnBatchLA N oc h w ε γ β
           (StableHLO.batchMap N (flatConv Wh bh) v) k ≠ 0 ∧
         StableHLO.bnBatchLA N oc h w ε γ β
           (StableHLO.batchMap N (flatConv Wh bh) v) k ≠ 6) :
    (StableHLO.batchMap N (convFlatBack (h := h) (w := w) Wh)
        ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConv Wh bh) v)
        ∘ reluMaskBack (fun i =>
            0 < StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConv Wh bh) v) i ∧
            StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConv Wh bh) v) i < 6))
      = (StableHLO.cbrB_has_vjp_at N Wh bh ε hε γ β v hs).backward := by
  rw [convFlatBack_eq_vjp_backward hkH hkW Wh bh (fun _ => 0)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE TIE — stem and head concrete, the seventeen bottlenecks opaque
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 800000 in
set_option maxHeartbeats 1000000 in
/-- ⭐⭐ **`mnv2InputGradB` IS the certified whole-net batch-BN MobileNetV2 gradient.** The
    committed backward chain, with its two BatchNorm and two relu6-mask slots filled by the
    certified per-op backwards and its seventeen bottlenecks left OPAQUE, equals the backward of
    `mobilenetv2PaperPC_has_vjp_at` at those twenty-one stages. `unfold`, three `rw`s, `rfl`. -/
theorem mnv2InputGradB_eq_mobilenetv2B_full_vjp (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wh : Kernel4 1280 320 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wfc : Mat 1280 nCls) (bfc : Vec nCls)
    (b1 : Vec (N * (32 * 112 * 112)) → Vec (N * (16 * 112 * 112)))
    (b2 : Vec (N * (16 * 112 * 112)) → Vec (N * (24 * 56 * 56)))
    (b3 : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4 : Vec (N * (24 * 56 * 56)) → Vec (N * (32 * 28 * 28)))
    (b5 : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b6 : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b7 : Vec (N * (32 * 28 * 28)) → Vec (N * (64 * 14 * 14)))
    (b8 : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b9 : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b10 : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b11 : Vec (N * (64 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b12 : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b13 : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b14 : Vec (N * (96 * 14 * 14)) → Vec (N * (160 * 7 * 7)))
    (b15 : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b16 : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b17 : Vec (N * (160 * 7 * 7)) → Vec (N * (320 * 7 * 7)))
    (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (h_stem : MNV2StemSmoothAtB N 112 112 Ws bs εs γs βs x)
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 (mnv2StemB N 112 112 Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ b1 (opaqueA0 (mnv2StemB N 112 112 Ws bs εs γs βs) x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA1 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA1 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA2 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA2 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA3 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA3 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA4 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA4 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA5 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA5 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA6 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA6 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA7 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA7 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA8 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA8 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA9 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA9 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA10 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA10 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA11 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA11 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA12 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA12 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA13 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA13 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA14 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA14 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA15 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA15 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA16 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA16 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (h_head : MNV2HeadSmoothAtB N 7 7 Wh bh εh γh βh (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) :
    mnv2InputGradB N Ws Wh Wfc
      ((bnBatchLA_has_vjp N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((bnBatchLA_has_vjp N 1280 7 7 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
      hb1.fst.backward
      hb2.fst.backward
      hb3.fst.backward
      hb4.fst.backward
      hb5.fst.backward
      hb6.fst.backward
      hb7.fst.backward
      hb8.fst.backward
      hb9.fst.backward
      hb10.fst.backward
      hb11.fst.backward
      hb12.fst.backward
      hb13.fst.backward
      hb14.fst.backward
      hb15.fst.backward
      hb16.fst.backward
      hb17.fst.backward
      (fun i =>
        0 < StableHLO.bnBatchLA N 32 112 112 εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i ∧
        StableHLO.bnBatchLA N 32 112 112 εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i < 6)
      (fun i =>
        0 < StableHLO.bnBatchLA N 1280 7 7 εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) i ∧
        StableHLO.bnBatchLA N 1280 7 7 εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) i < 6)
      = (mobilenetv2PaperPC_has_vjp_at (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17
          (StableHLO.cbrB N (h := 7) (w := 7) Wh bh εh γh βh) (StableHLO.batchMap N (globalAvgPoolFlat 1280 7 7)) (StableHLO.batchMap N (Proofs.dense Wfc bfc)) x
          ⟨mnv2StemB_has_vjp_at N 112 112 Ws bs εs hεs γs βs x h_stem,
            mnv2StemB_differentiableAt N 112 112 Ws bs εs hεs γs βs x h_stem⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17
          ⟨StableHLO.cbrB_has_vjp_at N Wh bh εh hεh γh βh _ h_head,
            StableHLO.cbrB_differentiableAt N Wh bh εh hεh γh βh _ h_head⟩
          ⟨(batchMap_has_vjp _ (globalAvgPoolFlat_has_vjp 1280 7 7)
              (globalAvgPoolFlat_differentiable 1280 7 7)).toHasVJPAt _,
            (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 1280 7 7)) _⟩
          ⟨(batchMap_has_vjp _ (dense_has_vjp Wfc bfc) (dense_differentiable Wfc bfc)).toHasVJPAt _,
            (batchMap_differentiable _ (dense_differentiable Wfc bfc)) _⟩).backward := by
  unfold mnv2InputGradB
  rw [mnv2StemBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbrBBack_eq_vjp_backward (by decide) (by decide) Wh bh εh hεh γh βh _ h_head,
      dense_transpose_eq_vjp_backward Wfc bfc (fun _ => 0)]
  rfl

set_option maxRecDepth 800000 in
set_option maxHeartbeats 1000000 in
/-- ⭐⭐ **The batched chain IS the `pdiv`-contracted Jacobian of the twenty-one-stage net** — at
    every batch size, every input, every loss cotangent and every input pixel. The tie above read
    through the apex's own `.correct`; `mobilenetv2ForwardB_full_eq_slots` below is what says
    those twenty-one stages are the committed forward. -/
theorem mnv2InputGradB_correct (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wh : Kernel4 1280 320 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wfc : Mat 1280 nCls) (bfc : Vec nCls)
    (b1 : Vec (N * (32 * 112 * 112)) → Vec (N * (16 * 112 * 112)))
    (b2 : Vec (N * (16 * 112 * 112)) → Vec (N * (24 * 56 * 56)))
    (b3 : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4 : Vec (N * (24 * 56 * 56)) → Vec (N * (32 * 28 * 28)))
    (b5 : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b6 : Vec (N * (32 * 28 * 28)) → Vec (N * (32 * 28 * 28)))
    (b7 : Vec (N * (32 * 28 * 28)) → Vec (N * (64 * 14 * 14)))
    (b8 : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b9 : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b10 : Vec (N * (64 * 14 * 14)) → Vec (N * (64 * 14 * 14)))
    (b11 : Vec (N * (64 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b12 : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b13 : Vec (N * (96 * 14 * 14)) → Vec (N * (96 * 14 * 14)))
    (b14 : Vec (N * (96 * 14 * 14)) → Vec (N * (160 * 7 * 7)))
    (b15 : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b16 : Vec (N * (160 * 7 * 7)) → Vec (N * (160 * 7 * 7)))
    (b17 : Vec (N * (160 * 7 * 7)) → Vec (N * (320 * 7 * 7)))
    (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (h_stem : MNV2StemSmoothAtB N 112 112 Ws bs εs γs βs x)
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 (mnv2StemB N 112 112 Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ b1 (opaqueA0 (mnv2StemB N 112 112 Ws bs εs γs βs) x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA1 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA1 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA2 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA2 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA3 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA3 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA4 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA4 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA5 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA5 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA6 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA6 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA7 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA7 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA8 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA8 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA9 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA9 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA10 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA10 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA11 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA11 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA12 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA12 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA13 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA13 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA14 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA14 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA15 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA15 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA16 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA16 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (h_head : MNV2HeadSmoothAtB N 7 7 Wh bh εh γh βh (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2InputGradB N Ws Wh Wfc
      ((bnBatchLA_has_vjp N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((bnBatchLA_has_vjp N 1280 7 7 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
      hb1.fst.backward
      hb2.fst.backward
      hb3.fst.backward
      hb4.fst.backward
      hb5.fst.backward
      hb6.fst.backward
      hb7.fst.backward
      hb8.fst.backward
      hb9.fst.backward
      hb10.fst.backward
      hb11.fst.backward
      hb12.fst.backward
      hb13.fst.backward
      hb14.fst.backward
      hb15.fst.backward
      hb16.fst.backward
      hb17.fst.backward
      (fun i =>
        0 < StableHLO.bnBatchLA N 32 112 112 εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i ∧
        StableHLO.bnBatchLA N 32 112 112 εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i < 6)
      (fun i =>
        0 < StableHLO.bnBatchLA N 1280 7 7 εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) i ∧
        StableHLO.bnBatchLA N 1280 7 7 εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) i < 6)
      dy i
      = ∑ j : Fin (N * nCls),
          pdiv (StableHLO.batchMap N (Proofs.dense Wfc bfc) ∘ StableHLO.batchMap N (globalAvgPoolFlat 1280 7 7) ∘ StableHLO.cbrB N (h := 7) (w := 7) Wh bh εh γh βh
          ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1
          ∘ mnv2StemB N 112 112 Ws bs εs γs βs) x i j * dy j := by
  rw [congrFun (mnv2InputGradB_eq_mobilenetv2B_full_vjp N Ws bs εs hεs γs βs Wh bh εh hεh γh βh
    Wfc bfc b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x h_stem hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 h_head) dy]
  exact (mobilenetv2PaperPC_has_vjp_at (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17
          (StableHLO.cbrB N (h := 7) (w := 7) Wh bh εh γh βh) (StableHLO.batchMap N (globalAvgPoolFlat 1280 7 7)) (StableHLO.batchMap N (Proofs.dense Wfc bfc)) x
          ⟨mnv2StemB_has_vjp_at N 112 112 Ws bs εs hεs γs βs x h_stem,
            mnv2StemB_differentiableAt N 112 112 Ws bs εs hεs γs βs x h_stem⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17
          ⟨StableHLO.cbrB_has_vjp_at N Wh bh εh hεh γh βh _ h_head,
            StableHLO.cbrB_differentiableAt N Wh bh εh hεh γh βh _ h_head⟩
          ⟨(batchMap_has_vjp _ (globalAvgPoolFlat_has_vjp 1280 7 7)
              (globalAvgPoolFlat_differentiable 1280 7 7)).toHasVJPAt _,
            (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 1280 7 7)) _⟩
          ⟨(batchMap_has_vjp _ (dense_has_vjp Wfc bfc) (dense_differentiable Wfc bfc)).toHasVJPAt _,
            (batchMap_differentiable _ (dense_differentiable Wfc bfc)) _⟩).correct dy i

/-- ⭐⭐ **The head group re-associated, proved once where the terms are VARIABLES.**
    `mnv2HeadB` is three stages and the apex takes them as three slots, so the shape check below
    meets `(dns ∘ gap ∘ head) ∘ trunk` on one side and `dns ∘ gap ∘ head ∘ trunk` on the other.
    They are definitionally equal — and ⛔ letting the kernel discover that on the CONCRETE
    twenty-one-stage net is a deterministic timeout, because whnf unfolds the `@[reducible]`
    block abbreviations to get there. Between variables it is `rfl` and costs nothing.
    `ResNet34BackCertifiedTie.lean`'s `chainComp₂_comp` is the same trick: prove the reduction
    where the terms are variables, then REWRITE. -/
private theorem comp3_assoc {m a b c n : Nat} (f : Vec c → Vec n) (g : Vec b → Vec c)
    (h : Vec a → Vec b) (k : Vec m → Vec a) : (f ∘ g ∘ h) ∘ k = f ∘ g ∘ h ∘ k := rfl

/-- ⭐⭐ **THE SHAPE CHECK — the twenty-one slots the tie is about ARE the committed forward.**
    `mobilenetv2ForwardB_full`, regrouped into exactly the twenty-one arguments
    `mobilenetv2PaperPC_has_vjp_at` takes: the XLA-`SAME` stem, `b1` the `t = 1` bottleneck,
    `b3/b5/b6/b8/b9/b10/b12/b13/b15/b16` the bodies under the identity skip, `b2/b4/b7/b14` the
    stride-2 downsamplers, `b11/b17` the stride-1 bodies whose channels change, and the head's
    three stages.

    ⛔ **This is the theorem that would have caught ResNet-34's wrong pool** (§3.10) — the tie
    keeps its blocks opaque, so its subject is a chain of VARIABLES and nothing in it says which
    net they are. It goes through `mobilenetv2ForwardB_full_eq_chain` (4.2b) for the depth-17 half
    and then unfolds the named prefixes and the head. -/
theorem mobilenetv2ForwardB_full_eq_slots (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mobilenetv2ForwardB_full N w x
      = (StableHLO.batchMap N (Proofs.dense w.fcW w.fcb)
          ∘ StableHLO.batchMap N (globalAvgPoolFlat 1280 7 7)
          ∘ StableHLO.cbrB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ
          ∘ mnv2ExpOnlyB N 7 7 w.b17
          ∘ mnv2ResidB N 7 7 w.b16
          ∘ mnv2ResidB N 7 7 w.b15
          ∘ mnv2StridedB N 7 7 w.b14
          ∘ mnv2ResidB N 14 14 w.b13
          ∘ mnv2ResidB N 14 14 w.b12
          ∘ mnv2ExpOnlyB N 14 14 w.b11
          ∘ mnv2ResidB N 14 14 w.b10
          ∘ mnv2ResidB N 14 14 w.b9
          ∘ mnv2ResidB N 14 14 w.b8
          ∘ mnv2StridedB N 14 14 w.b7
          ∘ mnv2ResidB N 28 28 w.b6
          ∘ mnv2ResidB N 28 28 w.b5
          ∘ mnv2StridedB N 28 28 w.b4
          ∘ mnv2ResidB N 56 56 w.b3
          ∘ mnv2StridedB N 56 56 w.b2
          ∘ mnv2NoExpB N 112 112 w.b1
          ∘ mnv2StemB N 112 112 w.sW w.sb w.sε w.sγ w.sβ) x := by
  rw [mobilenetv2ForwardB_full_eq_chain N w x]
  simp only [mnv2HeadB, mnv2PreB17, mnv2PreB16, mnv2PreB15, mnv2PreB14, mnv2PreB13, mnv2PreB12, mnv2PreB11, mnv2PreB10, mnv2PreB9, mnv2PreB8, mnv2PreB7, mnv2PreB6, mnv2PreB5, mnv2PreB4, mnv2PreB3, mnv2PreB2, mnv2PreB1, mnv2PreB0]
  rw [comp3_assoc]

end Proofs
