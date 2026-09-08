import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2WholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullVJP

/-! # ⭐⭐ `mnv2PaperInputGrad` IS the certified whole-net PAPER MobileNetV2 gradient

`MobileNetV2WholeBackCertifiedTie.lean` closed this for the reduced ch7 net — six blocks, the
`[t,c,n,s]` table's shape but not its depth. This file closes it for the net
`mobilenetv2ForwardPaper` actually is: **all seventeen bottlenecks**, the T6 row of
`planning/archive/proofs_tier_to_paper_nets.md` §3.2(c).

Nothing here is new mathematics. The four endpoint leaf ties (stem, head, GAP, dense) are the
ones the 6-block file proved, reused verbatim at the paper widths; the seventeen blocks stay
**opaque**, entering as `PProd (HasVJPAt …) (DifferentiableAt …)` witnesses whose `.fst.backward`
is what the reverse chain's block slots are pinned to. So the composition is checked between
variables and costs nothing, and the file elaborates in seconds rather than the minutes a
concrete 17-block `rfl` would not survive at all.

## The one thing that had to change at depth 17

The 6-block statement spells each hypothesis's running activation as a nested application —
`HasVJPAt b5 (b4 (b3 (b2 (b1 (stem x)))))`. At seventeen blocks that is unreadable by block 5 and
quadratic in the writing, the same wall `MobileNetV2FullVJP.lean` hit and answered with its
`mnv2Pre1 … mnv2Pre17` prefix defs. The opaque peer of that answer is `opaqueA0 … A17` below:
one top-level `def` per slot, each the previous one with one more block applied. They are plain
`def`s, NOT `@[irreducible]` — the closing `rfl` has to see through them.

## What ties this to the committed net

`mobilenetv2ForwardPaper_eq_slots` at the bottom of this file is the shape check: the twenty-one
slots `mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp` instantiates the apex at ARE
`mobilenetv2ForwardPaper`, block for block. It leans on `mobilenetv2ForwardPaper_eq_chain`
(`MobileNetV2FullVJP.lean`) for the depth-17 half and then unfolds the named prefixes.

⛔ **It cannot be a one-step `rfl`, and that is worth knowing before trying.** At depth 19 the
kernel takes a deterministic timeout after three minutes, and putting `Function.comp_assoc` in
the `simp` set reproduces it exactly. The statement's head group is therefore parenthesised the
way `mnv2HeadW` associates, so the peeled goal closes with no associativity step at all.

That check is the discipline §3.10's ResNet-34 pool drift cost a month for: *"the same net as the
tie"* was prose in a docstring there, and the wrong pool survived it.

⚠ It stays a SMOOTH-POINT statement, as every `HasVJPAt` in this cone is: relu6 is kinked, so
the stem's and head's post-BN clamp windows (`≠ 0 ∧ ≠ 6`) and the seventeen blocks' own VJP
witnesses are hypotheses.
-/

namespace Proofs

-- The opaque running activations `opaqueA0 … opaqueA17` are `Foundation/OpaquePrefix.lean`'s.


-- ════════════════════════════════════════════════════════════════
-- § The apex — a straight 21-stage chain, every stage opaque
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network paper MobileNetV2 VJP.** `dns ∘ gap ∘ head ∘ b17 ∘ … ∘ b1 ∘ stem`, the
    17-block peer of `mobilenetv2PC_has_vjp_at`. Twenty `vjp_comp_diff_at`s and nothing else:
    MobileNetV2's skips live INSIDE the block maps and its strides inside the strided bodies, so
    there is no `ChainData` list and no separate downsample slot at any depth. Dimension-generic
    and parametric in every component. -/
noncomputable def mobilenetv2PaperPC_has_vjp_at
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
    (hstem : PProd (HasVJPAt stem x) (DifferentiableAt ℝ stem x))
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 stem x))
                 (DifferentiableAt ℝ b1 (opaqueA0 stem x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA1 stem b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA1 stem b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA2 stem b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA2 stem b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA3 stem b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA3 stem b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA4 stem b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA4 stem b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA5 stem b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA5 stem b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA6 stem b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA6 stem b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA7 stem b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA7 stem b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA8 stem b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA8 stem b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA9 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA9 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA10 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA10 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA11 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA11 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA12 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA12 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA13 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA13 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA14 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA14 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA15 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA15 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (hhead : PProd (HasVJPAt head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
                   (DifferentiableAt ℝ head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
    (hgap : PProd (HasVJPAt gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
                  (DifferentiableAt ℝ gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))))
    (hdns : PProd (HasVJPAt dns (gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))))
                  (DifferentiableAt ℝ dns (gap (head (opaqueA17 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))))
    : HasVJPAt (dns ∘ gap ∘ head ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) x :=
  let p1 := vjp_comp_diff_at stem b1 x hstem hb1
  let p2 := vjp_comp_diff_at (b1 ∘ stem) b2 x p1 hb2
  let p3 := vjp_comp_diff_at (b2 ∘ b1 ∘ stem) b3 x p2 hb3
  let p4 := vjp_comp_diff_at (b3 ∘ b2 ∘ b1 ∘ stem) b4 x p3 hb4
  let p5 := vjp_comp_diff_at (b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b5 x p4 hb5
  let p6 := vjp_comp_diff_at (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b6 x p5 hb6
  let p7 := vjp_comp_diff_at (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b7 x p6 hb7
  let p8 := vjp_comp_diff_at (b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b8 x p7 hb8
  let p9 := vjp_comp_diff_at (b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b9 x p8 hb9
  let p10 := vjp_comp_diff_at (b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b10 x p9 hb10
  let p11 := vjp_comp_diff_at (b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b11 x p10 hb11
  let p12 := vjp_comp_diff_at (b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b12 x p11 hb12
  let p13 := vjp_comp_diff_at (b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b13 x p12 hb13
  let p14 := vjp_comp_diff_at (b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b14 x p13 hb14
  let p15 := vjp_comp_diff_at (b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b15 x p14 hb15
  let p16 := vjp_comp_diff_at (b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b16 x p15 hb16
  let p17 := vjp_comp_diff_at (b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b17 x p16 hb17
  let p18 := vjp_comp_diff_at (b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) head x p17 hhead
  let p19 := vjp_comp_diff_at (head ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) gap x p18 hgap
  let p20 := vjp_comp_diff_at (gap ∘ head ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) dns x p19 hdns
  p20.fst


-- ════════════════════════════════════════════════════════════════
-- § The reverse chain, and the tie
-- ════════════════════════════════════════════════════════════════

/-- The whole paper MobileNetV2 input-gradient VJP at a smooth point — the **exact reverse of
    `mobilenetv2ForwardPaper`**. The stem/head/GAP/dense endpoints are concrete
    (`flatConvStride2XlaBack ∘ bnBs ∘ reluMaskBack` / `convFlatBack ∘ bnBh ∘ reluMaskBack` /
    `gapBack` / `dense (transposeᵀ) 0`); the seventeen bottleneck backwards are supplied, the
    skip blocks already wrapped by `Proofs.residual`. -/
noncomputable def mnv2PaperInputGrad
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 10)
    (bnBs : Vec (32 * 112 * 112) → Vec (32 * 112 * 112))
    (bnBh : Vec (1280 * 7 * 7) → Vec (1280 * 7 * 7))
    (b1B : Vec (16*112*112) → Vec (32*112*112))
    (b2B : Vec (24*56*56) → Vec (16*112*112))
    (b3B : Vec (24*56*56) → Vec (24*56*56))
    (b4B : Vec (32*28*28) → Vec (24*56*56))
    (b5B : Vec (32*28*28) → Vec (32*28*28))
    (b6B : Vec (32*28*28) → Vec (32*28*28))
    (b7B : Vec (64*14*14) → Vec (32*28*28))
    (b8B : Vec (64*14*14) → Vec (64*14*14))
    (b9B : Vec (64*14*14) → Vec (64*14*14))
    (b10B : Vec (64*14*14) → Vec (64*14*14))
    (b11B : Vec (96*14*14) → Vec (64*14*14))
    (b12B : Vec (96*14*14) → Vec (96*14*14))
    (b13B : Vec (96*14*14) → Vec (96*14*14))
    (b14B : Vec (160*7*7) → Vec (96*14*14))
    (b15B : Vec (160*7*7) → Vec (160*7*7))
    (b16B : Vec (160*7*7) → Vec (160*7*7))
    (b17B : Vec (320*7*7) → Vec (160*7*7))
    (m_stem : Fin (32 * 112 * 112) → Prop) [DecidablePred m_stem]
    (m_head : Fin (1280 * 7 * 7) → Prop) [DecidablePred m_head] :
    Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride2XlaBack (h := 112) (w := 112) Ws ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B ∘ b17B
  ∘ (convFlatBack (h := 7) (w := 7) Wh ∘ bnBh ∘ reluMaskBack m_head)
  ∘ gapBack 1280 7 7
  ∘ dense (Mat.transpose Wfc) (0 : Vec 1280)


set_option maxRecDepth 800000 in
set_option maxHeartbeats 4000000 in
/-- ⭐⭐ **`mnv2PaperInputGrad` IS the certified whole-net paper-MobileNetV2 gradient.** The
    seventeen bottlenecks are opaque; only the four concrete endpoints are rewritten, and the
    proof is `unfold`, three `rw`s, `rfl` — the 6-block proof at depth 17. -/
theorem mnv2PaperInputGrad_eq_mobilenetv2Paper_vjp
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (γs βs : Vec 32) (hεs : 0 < εs)
    (Wh : Kernel4 1280 320 1 1) (bh : Vec 1280) (εh : ℝ) (γh βh : Vec 1280) (hεh : 0 < εh)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    (b1 : Vec (32*112*112) → Vec (16*112*112))
    (b2 : Vec (16*112*112) → Vec (24*56*56))
    (b3 : Vec (24*56*56) → Vec (24*56*56))
    (b4 : Vec (24*56*56) → Vec (32*28*28))
    (b5 : Vec (32*28*28) → Vec (32*28*28))
    (b6 : Vec (32*28*28) → Vec (32*28*28))
    (b7 : Vec (32*28*28) → Vec (64*14*14))
    (b8 : Vec (64*14*14) → Vec (64*14*14))
    (b9 : Vec (64*14*14) → Vec (64*14*14))
    (b10 : Vec (64*14*14) → Vec (64*14*14))
    (b11 : Vec (64*14*14) → Vec (96*14*14))
    (b12 : Vec (96*14*14) → Vec (96*14*14))
    (b13 : Vec (96*14*14) → Vec (96*14*14))
    (b14 : Vec (96*14*14) → Vec (160*7*7))
    (b15 : Vec (160*7*7) → Vec (160*7*7))
    (b16 : Vec (160*7*7) → Vec (160*7*7))
    (b17 : Vec (160*7*7) → Vec (320*7*7))
    (x : Vec (3 * 224 * 224))
    (hstem_smooth : ∀ k,
      bnPerChannelTensor3 32 112 112 εs γs βs
        (flatConvStride2Xla (h := 112) (w := 112) Ws bs x) k ≠ 0 ∧
      bnPerChannelTensor3 32 112 112 εs γs βs
        (flatConvStride2Xla (h := 112) (w := 112) Ws bs x) k ≠ 6)
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs)  x))
                 (DifferentiableAt ℝ b1 (opaqueA0 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs)  x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA1 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA1 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA2 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA2 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA3 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA3 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA4 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA4 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA5 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA5 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA6 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA6 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA7 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA7 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA8 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA8 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA9 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA9 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA10 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA10 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA11 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA11 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA12 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA12 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA13 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA13 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA14 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA14 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA15 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA15 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA16 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA16 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (hhead_smooth : ∀ k,
      bnPerChannelTensor3 1280 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
        (opaqueA17 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) k ≠ 0 ∧
      bnPerChannelTensor3 1280 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
        (opaqueA17 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) k ≠ 6) :
    mnv2PaperInputGrad Ws Wh Wfc
      ((bnPerChannelTensor3_has_vjp 32 112 112 εs hεs γs βs).backward
        (flatConvStride2Xla (h := 112) (w := 112) Ws bs x))
      ((bnPerChannelTensor3_has_vjp 1280 7 7 εh hεh γh βh).backward
        (flatConv (h := 7) (w := 7) Wh bh (opaqueA17 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
      hb1.fst.backward hb2.fst.backward hb3.fst.backward hb4.fst.backward hb5.fst.backward hb6.fst.backward hb7.fst.backward hb8.fst.backward hb9.fst.backward hb10.fst.backward hb11.fst.backward hb12.fst.backward hb13.fst.backward hb14.fst.backward hb15.fst.backward hb16.fst.backward hb17.fst.backward
      (fun i => 0 < bnPerChannelTensor3 32 112 112 εs γs βs
                  (flatConvStride2Xla (h := 112) (w := 112) Ws bs x) i ∧
                bnPerChannelTensor3 32 112 112 εs γs βs
                  (flatConvStride2Xla (h := 112) (w := 112) Ws bs x) i < 6)
      (fun i => 0 < bnPerChannelTensor3 1280 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
                  (opaqueA17 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) i ∧
                bnPerChannelTensor3 1280 7 7 εh γh βh (flatConv (h := 7) (w := 7) Wh bh
                  (opaqueA17 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)) i < 6)
      = (mobilenetv2PaperPC_has_vjp_at
          (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs)
          b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17
          (relu6 (1280 * 7 * 7) ∘ bnPerChannelTensor3 1280 7 7 εh γh βh
            ∘ flatConv (h := 7) (w := 7) Wh bh)
          (globalAvgPoolFlat 1280 7 7) (dense Wfc bfc) x
          ⟨convStridedBnRelu6PC_has_vjp_at (ic := 3) (oc := 32) (h := 112) (w := 112)
              Ws bs εs γs βs hεs x hstem_smooth,
            convStridedBnRelu6PC_differentiableAt (ic := 3) (oc := 32) (h := 112) (w := 112)
              Ws bs εs γs βs hεs x hstem_smooth⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17
          ⟨convBnRelu6PC_has_vjp_at (ic := 320) (oc := 1280) (h := 7) (w := 7)
              Wh bh εh γh βh hεh _ hhead_smooth,
            convBnRelu6PC_differentiableAt (ic := 320) (oc := 1280) (h := 7) (w := 7)
              Wh bh εh γh βh hεh _ hhead_smooth⟩
          ⟨(globalAvgPoolFlat_has_vjp 1280 7 7).toHasVJPAt _,
            (globalAvgPoolFlat_differentiable 1280 7 7) _⟩
          ⟨(dense_has_vjp Wfc bfc).toHasVJPAt _, (dense_differentiable Wfc bfc) _⟩).backward := by
  unfold mnv2PaperInputGrad
  rw [convStridedBnRelu6PCBack_eq_vjp_backward (ic := 3) (oc := 32) (h := 112) (w := 112)
        (by decide) (by decide) Ws bs εs γs βs hεs x hstem_smooth,
      convBnRelu6PCBack_eq_vjp_backward (ic := 320) (oc := 1280) (h := 7) (w := 7)
        (by decide) (by decide) Wh bh εh γh βh hεh _ hhead_smooth,
      dense_transpose_eq_vjp_backward Wfc bfc
        (globalAvgPoolFlat 1280 7 7 ((relu6 (1280 * 7 * 7) ∘ bnPerChannelTensor3 1280 7 7 εh γh βh
            ∘ flatConv (h := 7) (w := 7) Wh bh)
          (opaqueA17 (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 εs γs βs
            ∘ flatConvStride2Xla (h := 112) (w := 112) Ws bs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE SHAPE CHECK — the 21 slots the tie is about ARE the committed forward
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The chain `mobilenetv2PaperPC_has_vjp_at` is instantiated at IS
    `mobilenetv2ForwardPaper`**, slot for slot: `b1` the t=1 bottleneck, `b3/b5/b6/b8/b9/b10/
    b12/b13/b15/b16` the bodies under the identity skip, `b2/b4/b7/b14` the stride-2
    downsamplers, `b11/b17` the stride-1 bodies whose channels change, and the two endpoints
    spelled as the render spells them.

    `mobilenetv2ForwardPaper_eq_chain` (`MobileNetV2FullVJP.lean`) does the depth-17 half by
    peeling one `mnv2Pre` layer at a time — a bare `rfl` exhausts the elaborator there — and this
    flattens the named prefixes into the twenty-one slots the tie names.

    ⛔ **This is the theorem that would have caught ResNet-34's wrong pool.** §3.10's drift lived
    a month because *"the same net as the tie"* was prose in a docstring. -/
theorem mobilenetv2ForwardPaper_eq_slots (w : MNV2PaperWeights) (x : Vec (3 * 224 * 224)) :
    mobilenetv2ForwardPaper w x
      = ((dense w.fcW w.fcb
            ∘ globalAvgPoolFlat 1280 7 7
            ∘ (relu6 (1280 * 7 * 7) ∘ bnPerChannelTensor3 1280 7 7 w.hε w.hγ w.hβ
                ∘ flatConv (h := 7) (w := 7) w.hW w.hb))
          ∘ ivExpOnlyW 7 7 w.b17
          ∘ ivResidW 7 7 w.b16
          ∘ ivResidW 7 7 w.b15
          ∘ ivStridedW 7 7 w.b14
          ∘ ivResidW 14 14 w.b13
          ∘ ivResidW 14 14 w.b12
          ∘ ivExpOnlyW 14 14 w.b11
          ∘ ivResidW 14 14 w.b10
          ∘ ivResidW 14 14 w.b9
          ∘ ivResidW 14 14 w.b8
          ∘ ivStridedW 14 14 w.b7
          ∘ ivResidW 28 28 w.b6
          ∘ ivResidW 28 28 w.b5
          ∘ ivStridedW 28 28 w.b4
          ∘ ivResidW 56 56 w.b3
          ∘ ivStridedW 56 56 w.b2
          ∘ ivNoExpW 112 112 w.b1
          ∘ (relu6 (32 * 112 * 112) ∘ bnPerChannelTensor3 32 112 112 w.sε w.sγ w.sβ
              ∘ flatConvStride2Xla (h := 112) (w := 112) w.sW w.sb)) x := by
  rw [mobilenetv2ForwardPaper_eq_chain w x]
  simp only [mnv2HeadW, mnv2Pre17, mnv2Pre16, mnv2Pre15, mnv2Pre14, mnv2Pre13, mnv2Pre12,
    mnv2Pre11, mnv2Pre10, mnv2Pre9, mnv2Pre8, mnv2Pre7, mnv2Pre6, mnv2Pre5, mnv2Pre4,
    mnv2Pre3, mnv2Pre2, mnv2Pre1, mnv2StemW]

end Proofs
