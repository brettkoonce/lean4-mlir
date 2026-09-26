import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains

/-! # `mnv2InputGradB` and the certified whole-net MobileNetV2 input VJP at batch BatchNorm

Ties the batched input-gradient chain `mnv2InputGradB` to a certified VJP for
`mobilenetv2ForwardBFull`, the `[t,c,n,s]` ladder at `bnBatchLA`, at a variable batch `N`: the
MobileNetV2 peer of `ResNet34BackCertifiedTieB.lean`.

## One apex, every stage opaque

`mobilenetv2PaperPCHasVJPAt` (below) is a twenty-one-stage chain generic in every dimension and in every stage, so the batched
net instantiates it directly: `stem` at `mnv2StemB`, the seventeen bottlenecks at the batched
block maps, and the head's three stages at `cbrB` / `batchMap gap` / `batchMap dense`.
`opaqueA0 … A17` are `OpaquePrefix.lean`'s. So this file defines **no prefix defs** —
where ResNet-34's peer had to, because its committed apex bundles the stem's pool into `stem` and
its head into one stage.

## The three pieces

1. `mnv2StemBBack_eq_vjp_backward` and `cbrBBack_eq_vjp_backward` — the two concrete
   conv-BN-relu6 endpoints, one `rw` of a conv leaf tie and then `rfl` each. The stem's is the
   XLA-`SAME` leaf (`flatConvStride2XlaBack`) and the head's the plain one; that is the one
   convention this net and ResNet-34 do not share.
2. `mnv2InputGradB_eq_mobilenetv2B_full_vjp` and `mnv2InputGradB_correct` — the tie, and its
   reading as `∑ pdiv … * dy`: with its seventeen block slots filled by certified block VJPs,
   the chain is the Jacobian-transpose of the twenty-one-stage composition, at every batch size.
3. `mobilenetv2ForwardBFull_eq_slots` — the shape check: those twenty-one stages ARE
   `mobilenetv2ForwardBFull`, the forward `mobilenetv2FwdGraphBFull_faithful` says the
   typed graph denotes. Without it the tie would be a statement about variables.

**Why the blocks stay opaque.** Instantiating a tie of this shape
at the concrete blocks is a *kernel* deterministic timeout: the block witnesses are `HasVJPAt` at
`opaqueA{k-1} … x` and a caller's are at `mnv2PreB{k-1} N w x`, which is seventeen defeq
checks between seventeen-deep nested applications spelled through different definition chains.
B0's file takes that step only because swish has no kink, so its witnesses are GLOBAL `HasVJP`
and carry no point at all. The shape check is what replaces it.

It is a smooth-point statement: relu6 is kinked on both sides, so each of the 35 sites carries
`≠ 0 ∧ ≠ 6`. MobileNetV2's two clauses per block are the expand relu6 and the depthwise relu6,
both inside the body — not ResNet-34's mid-relu and post-residual outer relu. Those are
`MobileNetV2FullBVJP`'s bundles; this file adds no hypothesis of its own.

**Scope.** One device's batch `N`: the data-parallel artifacts normalise over the global batch, so
this describes them only at `N := R·N`. It is about the input gradient; the parameter gradients
are `MobileNetV2StepTieB.lean`'s tie.
-/

namespace Proofs

-- ═══════════════════════════════════════════════════════════════
-- § The apex — a straight 21-stage chain, every stage opaque (generic in every dimension)
-- ═══════════════════════════════════════════════════════════════

/-- **The whole-network MobileNetV2 VJP at opaque stages.** `dns ∘ gap ∘ head ∘ b17 ∘ … ∘ b1 ∘ stem`. Twenty `vjpCompDiffAt`s and nothing else:
    MobileNetV2's skips live INSIDE the block maps and its strides inside the strided bodies, so
    there is no list of blocks and no separate downsample slot at any depth. Dimension-generic
    and parametric in every component. -/
noncomputable def mobilenetv2PaperPCHasVJPAt
    {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 : Nat}
    (stem : Vec s0 → Vec s1)
    (b1 : Vec s1 → Vec s2)
    (b2 : Vec s2 → Vec s3)
    (b3 : Vec s3 → Vec s4)
    (b4 : Vec s4 → Vec s5)
    (b5 : Vec s5 → Vec s6)
    (b6 : Vec s6 → Vec s7)
    (b7 : Vec s7 → Vec s8)
    (b8 : Vec s8 → Vec s9)
    (b9 : Vec s9 → Vec s10)
    (b10 : Vec s10 → Vec s11)
    (b11 : Vec s11 → Vec s12)
    (b12 : Vec s12 → Vec s13)
    (b13 : Vec s13 → Vec s14)
    (b14 : Vec s14 → Vec s15)
    (b15 : Vec s15 → Vec s16)
    (b16 : Vec s16 → Vec s17)
    (b17 : Vec s17 → Vec s18)
    (head : Vec s18 → Vec s19) (gap : Vec s19 → Vec s20) (dns : Vec s20 → Vec s21)
    (x : Vec s0)
    (hstem : HasVJPDiffAt stem x)
    (hb1 : HasVJPDiffAt b1 (opaqueA0 stem x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 stem b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 stem b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 stem b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 stem b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 stem b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 stem b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 stem b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 stem b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (hhead : HasVJPDiffAt head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (hgap : HasVJPDiffAt gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
    (hdns : HasVJPDiffAt dns (gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))))
    : HasVJPAt (dns ∘ gap ∘ head ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) x :=
  let p1 := vjpCompDiffAt stem b1 x hstem hb1
  let p2 := vjpCompDiffAt (b1 ∘ stem) b2 x p1 hb2
  let p3 := vjpCompDiffAt (b2 ∘ b1 ∘ stem) b3 x p2 hb3
  let p4 := vjpCompDiffAt (b3 ∘ b2 ∘ b1 ∘ stem) b4 x p3 hb4
  let p5 := vjpCompDiffAt (b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b5 x p4 hb5
  let p6 := vjpCompDiffAt (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b6 x p5 hb6
  let p7 := vjpCompDiffAt (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b7 x p6 hb7
  let p8 := vjpCompDiffAt (b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b8 x p7 hb8
  let p9 := vjpCompDiffAt (b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b9 x p8 hb9
  let p10 := vjpCompDiffAt (b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b10 x p9 hb10
  let p11 := vjpCompDiffAt (b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b11 x p10 hb11
  let p12 := vjpCompDiffAt (b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b12 x p11 hb12
  let p13 := vjpCompDiffAt (b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b13 x p12 hb13
  let p14 := vjpCompDiffAt (b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b14 x p13 hb14
  let p15 := vjpCompDiffAt (b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b15 x p14 hb15
  let p16 := vjpCompDiffAt (b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b16 x p15 hb16
  let p17 := vjpCompDiffAt (b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b17 x p16 hb17
  let p18 := vjpCompDiffAt (b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) head x p17 hhead
  let p19 := vjpCompDiffAt (head ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) gap x p18 hgap
  let p20 := vjpCompDiffAt (gap ∘ head ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) dns x p19 hdns
  p20.fst

/-- **The apex's backward, peeled** — each stage's backward in turn, head first. `rfl` over
    VARIABLE stages; the tie below instantiates it by `rw`, so the kernel never re-derives the
    concrete chain (the ResNet-34 apex's `r34BFullHasVJPAt_backward`). -/
theorem mobilenetv2PaperPCHasVJPAt_backward
    {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 : Nat}
    (stem : Vec s0 → Vec s1)
    (b1 : Vec s1 → Vec s2)
    (b2 : Vec s2 → Vec s3)
    (b3 : Vec s3 → Vec s4)
    (b4 : Vec s4 → Vec s5)
    (b5 : Vec s5 → Vec s6)
    (b6 : Vec s6 → Vec s7)
    (b7 : Vec s7 → Vec s8)
    (b8 : Vec s8 → Vec s9)
    (b9 : Vec s9 → Vec s10)
    (b10 : Vec s10 → Vec s11)
    (b11 : Vec s11 → Vec s12)
    (b12 : Vec s12 → Vec s13)
    (b13 : Vec s13 → Vec s14)
    (b14 : Vec s14 → Vec s15)
    (b15 : Vec s15 → Vec s16)
    (b16 : Vec s16 → Vec s17)
    (b17 : Vec s17 → Vec s18)
    (head : Vec s18 → Vec s19) (gap : Vec s19 → Vec s20) (dns : Vec s20 → Vec s21)
    (x : Vec s0)
    (hstem : HasVJPDiffAt stem x)
    (hb1 : HasVJPDiffAt b1 (opaqueA0 stem x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 stem b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 stem b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 stem b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 stem b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 stem b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 stem b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 stem b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 stem b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (hhead : HasVJPDiffAt head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (hgap : HasVJPDiffAt gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
    (hdns : HasVJPDiffAt dns (gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))))
    (dy : Vec s21) :
    (mobilenetv2PaperPCHasVJPAt stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 head gap dns x hstem hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hhead hgap hdns).backward dy
      =
      hstem.fst.backward
        (hb1.fst.backward
          (hb2.fst.backward
            (hb3.fst.backward
              (hb4.fst.backward
                (hb5.fst.backward
                  (hb6.fst.backward
                    (hb7.fst.backward
                      (hb8.fst.backward
                        (hb9.fst.backward
                          (hb10.fst.backward
                            (hb11.fst.backward
                              (hb12.fst.backward
                                (hb13.fst.backward
                                  (hb14.fst.backward
                                    (hb15.fst.backward
                                      (hb16.fst.backward
                                        (hb17.fst.backward
                                          (hhead.fst.backward
                                            (hgap.fst.backward
                                              (hdns.fst.backward dy)))))))))))))))))))) := rfl

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The two concrete conv-BN-relu6 endpoint ties
-- ════════════════════════════════════════════════════════════════

/-- **The STEM tie.** `batchMap (flatConvStride2XlaBack) ∘ bnBack ∘ reluMaskBack` is `mnv2StemB`'s
    certified backward at a smooth point. One `rw` of the odd-kernel XLA-`SAME` strided leaf tie,
    then `rfl` — the stage's VJP is `vjpCompAt`-built so its backward is already the
    composition, and a convolution's backward ignores its primal argument, so the row-wise
    `batchMap` lift matches at every saved input. -/
theorem mnv2StemBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : MNV2StemSmoothAtB N h w Ws bs ε γ β x) :
    (StableHLO.batchMap N (flatConvStride2XlaBack (h := h) (w := w) Ws)
        ∘ (bnBatchLAHasVJP N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x)
        ∘ reluMaskBack (fun i =>
            0 < StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i ∧
            StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i < 6))
      = (mnv2StemBHasVJPAt N h w Ws bs ε hε γ β x hs).backward := by
  rw [flatConvStride2XlaBack_eq_vjp_backward hkH hkW Ws bs (fun _ => 0)]
  rfl

/-- **The HEAD's conv-BN-relu6 tie.** `batchMap (convFlatBack) ∘ bnBack ∘ reluMaskBack` IS
    `cbrB`'s certified backward at a smooth point — the stride-1 peer of the stem's, at the plain
    (non-XLA) convolution leaf. MobileNetV2's head is not hypothesis-free, unlike ResNet-34's:
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
        ∘ (bnBatchLAHasVJP N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConv Wh bh) v)
        ∘ reluMaskBack (fun i =>
            0 < StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConv Wh bh) v) i ∧
            StableHLO.bnBatchLA N oc h w ε γ β
              (StableHLO.batchMap N (flatConv Wh bh) v) i < 6))
      = (StableHLO.cbrBHasVJPAt N Wh bh ε hε γ β v hs).backward := by
  rw [convFlatBack_eq_vjp_backward hkH hkW Wh bh (fun _ => 0)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE TIE — stem and head concrete, the seventeen bottlenecks opaque
-- ════════════════════════════════════════════════════════════════

/-- **`mnv2InputGradB` is the backward of the whole-net VJP at opaque blocks.** The committed
    backward chain, with its two BatchNorm and two relu6-mask slots filled by the certified per-op
    backwards and its seventeen block slots filled by the supplied block witnesses' backwards
    (`hb1 … hb17`, blocks `b1 … b17` left opaque), equals the backward of
    `mobilenetv2PaperPCHasVJPAt` at those twenty-one stages, at an input where the stem and head
    relu6 clauses (`h_stem`, `h_head`) hold. -/
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
    (hb1 : HasVJPDiffAt b1 (opaqueA0 (mnv2StemB N 112 112 Ws bs εs γs βs) x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA16 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (h_head : MNV2HeadSmoothAtB N 7 7 Wh bh εh γh βh (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) :
    mnv2InputGradB N Ws Wh Wfc
      ((bnBatchLAHasVJP N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((bnBatchLAHasVJP N 1280 7 7 εh hεh γh βh).backward
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
      = (mobilenetv2PaperPCHasVJPAt (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17
          (StableHLO.cbrB N (h := 7) (w := 7) Wh bh εh γh βh) (StableHLO.batchMap N (globalAvgPoolFlat 1280 7 7)) (StableHLO.batchMap N (Proofs.dense Wfc bfc)) x
          ⟨mnv2StemBHasVJPAt N 112 112 Ws bs εs hεs γs βs x h_stem,
            mnv2StemB_differentiableAt N 112 112 Ws bs εs hεs γs βs x h_stem⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17
          ⟨StableHLO.cbrBHasVJPAt N Wh bh εh hεh γh βh _ h_head,
            StableHLO.cbrB_differentiableAt N Wh bh εh hεh γh βh _ h_head⟩
          ⟨(batchMapHasVJP _ (globalAvgPoolFlatHasVJP 1280 7 7)
              (globalAvgPoolFlat_differentiable 1280 7 7)).toHasVJPAt _,
            (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 1280 7 7)) _⟩
          ⟨(batchMapHasVJP _ (denseHasVJP Wfc bfc) (dense_differentiable Wfc bfc)).toHasVJPAt _,
            (batchMap_differentiable _ (dense_differentiable Wfc bfc)) _⟩).backward := by
  unfold mnv2InputGradB
  rw [mnv2StemBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbrBBack_eq_vjp_backward (by decide) (by decide) Wh bh εh hεh γh βh _ h_head,
      dense_transpose_eq_vjp_backward Wfc bfc (fun _ => 0)]
  funext dy
  rw [mobilenetv2PaperPCHasVJPAt_backward]
  repeat rw [Function.comp_apply]
  rfl

/-- **The batched chain is the `pdiv`-contracted Jacobian of the twenty-one-stage composition** —
    at every batch size and loss cotangent, and at every input where the stem and head relu6
    clauses hold (`h_stem`, `h_head`): the chain, with its seventeen block slots filled by the
    supplied certified block VJPs at the running activations (`hb1 … hb17`), is the
    Jacobian-transpose of `dense ∘ gap ∘ cbrB ∘ b17 ∘ … ∘ b1 ∘ mnv2StemB`. The blocks are
    variables here; `mobilenetv2ForwardBFull_eq_slots` identifies the stages with the committed
    forward. -/
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
    (hb1 : HasVJPDiffAt b1 (opaqueA0 (mnv2StemB N 112 112 Ws bs εs γs βs) x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA16 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (h_head : MNV2HeadSmoothAtB N 7 7 Wh bh εh γh βh (opaqueA17 (mnv2StemB N 112 112 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2InputGradB N Ws Wh Wfc
      ((bnBatchLAHasVJP N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((bnBatchLAHasVJP N 1280 7 7 εh hεh γh βh).backward
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
  exact HasVJPAt.correct_of_backward_eq _ (mnv2InputGradB_eq_mobilenetv2B_full_vjp N Ws bs εs hεs γs βs Wh bh εh hεh γh βh
    Wfc bfc b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x h_stem hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 h_head) dy i

/-- **The head group re-associated, proved once where the terms are variables.**
    `mnv2HeadB` is three stages and the apex takes them as three slots, so the shape check below
    meets `(dns ∘ gap ∘ head) ∘ trunk` on one side and `dns ∘ gap ∘ head ∘ trunk` on the other.
    They are definitionally equal. Note: letting the kernel discover that on the concrete
    twenty-one-stage net is a deterministic timeout, because whnf unfolds the `@[reducible]`
    block abbreviations to get there; between variables it is `rfl`. -/
private theorem comp3_assoc {m a b c n : Nat} (f : Vec c → Vec n) (g : Vec b → Vec c)
    (h : Vec a → Vec b) (k : Vec m → Vec a) : (f ∘ g ∘ h) ∘ k = f ∘ g ∘ h ∘ k := rfl

/-- **The shape check — the twenty-one slots the tie is about are the committed forward.**
    `mobilenetv2ForwardBFull`, regrouped into exactly the twenty-one arguments
    `mobilenetv2PaperPCHasVJPAt` takes: the XLA-`SAME` stem, `b1` the `t = 1` bottleneck,
    `b3/b5/b6/b8/b9/b10/b12/b13/b15/b16` the bodies under the identity skip, `b2/b4/b7/b14` the
    stride-2 downsamplers, `b11/b17` the stride-1 bodies whose channels change, and the head's
    three stages.

    The tie keeps its blocks opaque, so its subject is a chain of variables and nothing in it
    says which net they are; this theorem does. It goes through `mobilenetv2ForwardBFull_eq_chain`
    for the depth-17 half and then unfolds the named prefixes and the head. -/
theorem mobilenetv2ForwardBFull_eq_slots (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mobilenetv2ForwardBFull N w x
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
  rw [mobilenetv2ForwardBFull_eq_chain N w x]
  simp only [mnv2HeadB, mnv2PreB17, mnv2PreB16, mnv2PreB15, mnv2PreB14, mnv2PreB13, mnv2PreB12, mnv2PreB11, mnv2PreB10, mnv2PreB9, mnv2PreB8, mnv2PreB7, mnv2PreB6, mnv2PreB5, mnv2PreB4, mnv2PreB3, mnv2PreB2, mnv2PreB1, mnv2PreB0]
  rw [comp3_assoc]

end Proofs
