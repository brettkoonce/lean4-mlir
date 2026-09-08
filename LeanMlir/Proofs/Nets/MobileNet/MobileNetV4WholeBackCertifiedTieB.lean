import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBVJP
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Nets.ResNet.Resnet34BackCertifiedTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains

/-! # ⭐⭐ `mnv4InputGradB` IS the certified whole-net MobileNetV4-Conv-M gradient

Tier **T6** of `planning/archive/mnv4_proofs_tier.md` §Session 3 — the last tier this net can have, and
with it MobileNetV4-Conv-M is certified from its ℝ forward (T1) through its typed graph (T2), its
233-parameter train-step tie (T3) and now its whole-net input gradient. T4 and T5 are float
budgets and `planning/archive/float_budget_numbers.md` closed that thread by user decision on 2026-09-05.

⚠⚠ **No accuracy is quoted for this net.** Conv-M has no Imagenette run and no verified ImageNet
run; what pins the artifact to the reference's function is the 2026-09-07 tie pair (forward
`max |Δ| = 3.770e-06`, gradient 0 of 232 live parameters outside the reference's own
relu-discontinuity floor).

## The four pieces

1. `mnv4StemBBack_eq_vjp_backward` and `cbReluBBack_eq_vjp_backward` — the concrete endpoint stage
   ties, one `rw` of a conv leaf tie and then `rfl` each. ⚠ The stem's is the XLA-`SAME` leaf
   (`flatConvStride2XlaBack`) and the head's the plain one; MobileNetV4 carries both phases.
2. `mnv4B_full_has_vjp_at` — the generic **twenty-six-stage** apex
   `head ∘ hc2 ∘ hc1 ∘ b21 ∘ … ∘ b1 ∘ fused ∘ stem`, twenty-five `vjp_comp_diff_at`s and nothing
   else, with `opaqueA0 … A24` naming the running activations.
3. `mnv4InputGradB_eq_mnv4B_full_vjp` and `mnv4InputGradB_correct` — the tie, and its reading as
   `∑ pdiv … * dy`: the chain IS the Jacobian-transpose of the twenty-six-stage composition, at
   every batch size and both shipped class counts.
4. `mobilenetv4ForwardB_full_eq_slots` — the shape check: those twenty-six stages ARE
   `mobilenetv4ForwardB_full`, the forward `mnv4FwdGraphB_full_faithful` (T2) says the typed graph
   denotes. Without it the tie would be a statement about variables.

## ⭐⭐ What is MobileNetV4's own, and what is borrowed

ResNet-50's T6 was four declarations because its stem and head ARE ResNet-34's. MobileNetV4's stem
is EfficientNet-B0's (3×3/s2 at the XLA-`SAME` phase) with ResNet-34's activation, so the leaf
ties compose rather than transfer: `flatConvStride2XlaBack_eq_vjp_backward` is B0's/MobileNetV2's
and the relu mask is ResNet-34's, and the stage tie that joins them is new here. The head's two
1×1 conv-BN-relu stages are `cbReluB` — ResNet-34's stage vocabulary at a 1×1 kernel — and its
**GAP-and-dense tail is `r34HeadB` verbatim**, exactly as T3 reused `r34HeadCotBlk`/`r34HeadTiedB`.
So what this file adds is the two stage ties, the wider apex, and the tie itself.

⚠ **Twenty-six stages, not eighteen.** Conv-M's ladder is 21 UIB blocks plus a fused stage, and
its head has TWO convs before the pool where MobileNetV2's has one and ResNet-34's has none.

## ⛔⛔ The apex is composed at BLOCK granularity, not at `MobileNetV4FullB.lean`'s groups

`mobilenetv4ForwardB_full_has_vjp_at` (T1) is a chain of seven `vjp_comp_at`s over the five
resolution-group `CertLayer`s, because `Mnv4SmoothAt` wants one `.ok` per group. T6 wants the
finer chain — a tie whose opaque slots are groups would say nothing about which block is which —
so the stages here are the twenty-one blocks themselves, `mnv4Blk0 … mnv4Blk21`'s
(`MobileNetV4TiePoCB.lean`) stage functions. ⭐ Peeling `CertLayer.comp` to reach `.fwd` INSIDE a
group is cheap (`comp_fwd` is a generic `rfl` lemma and T2's five group-faithfulness proofs each
do it); what is not payable at MobileNetV4's literal resolutions is composing the compositions,
which is why there is no `mnv4NetLayer` and why the shape check below stops at the block.

⚠⚠ **Everything generic in its widths, instantiated only in the capstone.** Both stage ties bind
`{N ic oc h w kH kW}` and the apex binds `{s0 … s26}`; MobileNetV4's 224/112/56/28/14/7 are
literals, and stating a `den`- or width-indexed `rfl` at them lets it RUN — six blow-ups across
sessions 1–2 trace to that one cause. Instantiating a proven lemma is free.

⛔ **The blocks stay opaque and there is no `backward_unique` step**, for §4.2d's measured reason:
instantiating a tie of this shape at the concrete blocks is a KERNEL deterministic timeout when
the witnesses are `HasVJPAt` carrying a saved activation. B0 takes that step only because swish
has no kink. The shape check is what replaces it.

⚠ It stays a SMOOTH-POINT statement, and MobileNetV4's kink budget is **relu, not relu6**: one
clause per site, where MobileNetV2's `.selectMid` carries two. The stem's and the two head convs'
are bound here; the ~60 inside the blocks are `CertLayer.comp`'s and are never written down.

⛔ **What this does NOT reach.** Under `mnv4in_adamdp64*` every gradient is all-reduced by
`allReduceMeanF` (`DataParallelNode.lean`, §4d), so this is at the per-replica gradient. And it is
about the INPUT gradient; the 233 parameter gradients are `MobileNetV4TiePoCB.lean`'s tie (T3).
-/

open Proofs.StableHLO

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The stem's concrete conv-BN-relu endpoint tie
--
--   The head's stage tie, `cbReluBBack_eq_vjp_backward` (stride-1, applied twice at `%h1W` and
--   `%hW`), is `Resnet34BackCertifiedTieB.lean`'s — net-agnostic, beside its strided peer.
-- ════════════════════════════════════════════════════════════════

/-- **The STEM tie.** `batchMap (flatConvStride2XlaBack) ∘ bnBack ∘ reluMaskBack` IS `mnv4StemB`'s
    certified backward at a smooth point. One `rw` of the odd-kernel XLA-`SAME` strided leaf tie,
    then `rfl` — the stage's VJP is `vjp_comp_at`-built so its backward is already the
    composition, and a convolution's backward ignores its primal argument, so the row-wise
    `batchMap` lift matches at every saved input.

    ⚠ **Plain relu, XLA-`SAME` padding** — the two axes MobileNetV4's stem does not share with a
    neighbour. MobileNetV2's stem is this map with `relu6` (two mask clauses); ResNet-34's is this
    map with SYMMETRIC padding (`cbReluStridedBBack_eq_vjp_backward`); EfficientNet-B0's is this
    padding with `swish`. One token apart from each, and a different certificate from all three.

    ⛔⛔ **And this is one step past the artifact.** No render emits a gradient into `%x`, so
    MobileNetV4's committed backward ends at the stem conv's WEIGHT gradient, whose operand is the
    stem-BN cotangent `mnv4StemCotN` ties (T3). This lemma names the map that carries that
    cotangent the rest of the way to the image — B0's choice at the identical stem. -/
theorem mnv4StemBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : Mnv4StemSmoothAtB N h w Ws bs ε γ β x) :
    (StableHLO.batchMap N (flatConvStride2XlaBack (h := h) (w := w) Ws)
        ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x)
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc h w ε γ β
            (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i > 0))
      = (mnv4StemB_has_vjp_at N h w Ws bs ε hε γ β x hs).backward := by
  rw [flatConvStride2XlaBack_eq_vjp_backward hkH hkW Ws bs (fun _ => 0)]
  rfl

-- The opaque running activations `opaqueA0 … opaqueA24` are `Foundation/OpaquePrefix.lean`'s.

-- ════════════════════════════════════════════════════════════════
-- § The generic twenty-six-stage apex
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network MobileNetV4-Conv-M VJP.** The VJP of the twenty-six-stage chain
    `head ∘ hc2 ∘ hc1 ∘ b21 ∘ … ∘ b1 ∘ fused ∘ stem` — twenty-five `vjp_comp_diff_at`s and
    nothing else, dimension-generic and parametric in every component.

    ⚠ Pointwise (`HasVJPAt`), and necessarily: relu is kinked, so each stage's witness carries the
    activation it sees. `r34B_full_has_vjp_at` is the same construction at eighteen stages;
    MobileNetV4 needs its own because Conv-M's ladder is longer, not because anything differs. -/
noncomputable def mnv4B_full_has_vjp_at {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 : Nat}
    (stem : Vec s0 → Vec s1)
    (fused : Vec s1 → Vec s2)
    (b1 : Vec s2 → Vec s3)
    (b2 : Vec s3 → Vec s4)
    (b3 : Vec s4 → Vec s5)
    (b4 : Vec s5 → Vec s6)
    (b5 : Vec s6 → Vec s7)
    (b6 : Vec s7 → Vec s8)
    (b7 : Vec s8 → Vec s9)
    (b8 : Vec s9 → Vec s10)
    (b9 : Vec s10 → Vec s11)
    (b10 : Vec s11 → Vec s12)
    (b11 : Vec s12 → Vec s13)
    (b12 : Vec s13 → Vec s14)
    (b13 : Vec s14 → Vec s15)
    (b14 : Vec s15 → Vec s16)
    (b15 : Vec s16 → Vec s17)
    (b16 : Vec s17 → Vec s18)
    (b17 : Vec s18 → Vec s19)
    (b18 : Vec s19 → Vec s20)
    (b19 : Vec s20 → Vec s21)
    (b20 : Vec s21 → Vec s22)
    (b21 : Vec s22 → Vec s23)
    (hc1 : Vec s23 → Vec s24)
    (hc2 : Vec s24 → Vec s25)
    (head : Vec s25 → Vec s26)
    (x : Vec s0)
    (hstem : PProd (HasVJPAt stem (x))
                 (DifferentiableAt ℝ stem (x)))
    (hfused : PProd (HasVJPAt fused (opaqueA0 stem x))
                 (DifferentiableAt ℝ fused (opaqueA0 stem x)))
    (hb1 : PProd (HasVJPAt b1 (opaqueA1 stem fused x))
                 (DifferentiableAt ℝ b1 (opaqueA1 stem fused x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA2 stem fused b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA2 stem fused b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA3 stem fused b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA3 stem fused b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA4 stem fused b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA4 stem fused b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA5 stem fused b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA5 stem fused b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA6 stem fused b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA6 stem fused b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA7 stem fused b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA7 stem fused b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA8 stem fused b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA8 stem fused b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA9 stem fused b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA9 stem fused b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA10 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA10 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA11 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA11 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA12 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA12 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA13 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA13 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA14 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA14 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA15 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA15 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA16 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA16 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA17 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA17 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (hb18 : PProd (HasVJPAt b18 (opaqueA18 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
                 (DifferentiableAt ℝ b18 (opaqueA18 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
    (hb19 : PProd (HasVJPAt b19 (opaqueA19 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
                 (DifferentiableAt ℝ b19 (opaqueA19 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x)))
    (hb20 : PProd (HasVJPAt b20 (opaqueA20 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
                 (DifferentiableAt ℝ b20 (opaqueA20 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x)))
    (hb21 : PProd (HasVJPAt b21 (opaqueA21 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
                 (DifferentiableAt ℝ b21 (opaqueA21 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x)))
    (hhc1 : PProd (HasVJPAt hc1 (opaqueA22 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x))
                 (DifferentiableAt ℝ hc1 (opaqueA22 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)))
    (hhc2 : PProd (HasVJPAt hc2 (opaqueA23 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 x))
                 (DifferentiableAt ℝ hc2 (opaqueA23 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 x)))
    (hhead : PProd (HasVJPAt head (opaqueA24 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 hc2 x))
                 (DifferentiableAt ℝ head (opaqueA24 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 hc2 x)))
    : HasVJPAt (head ∘ hc2 ∘ hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) x :=
  let p1 := vjp_comp_diff_at (stem) fused x hstem hfused
  let p2 := vjp_comp_diff_at (fused ∘ stem) b1 x p1 hb1
  let p3 := vjp_comp_diff_at (b1 ∘ fused ∘ stem) b2 x p2 hb2
  let p4 := vjp_comp_diff_at (b2 ∘ b1 ∘ fused ∘ stem) b3 x p3 hb3
  let p5 := vjp_comp_diff_at (b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b4 x p4 hb4
  let p6 := vjp_comp_diff_at (b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b5 x p5 hb5
  let p7 := vjp_comp_diff_at (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b6 x p6 hb6
  let p8 := vjp_comp_diff_at (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b7 x p7 hb7
  let p9 := vjp_comp_diff_at (b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b8 x p8 hb8
  let p10 := vjp_comp_diff_at (b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b9 x p9 hb9
  let p11 := vjp_comp_diff_at (b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b10 x p10 hb10
  let p12 := vjp_comp_diff_at (b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b11 x p11 hb11
  let p13 := vjp_comp_diff_at (b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b12 x p12 hb12
  let p14 := vjp_comp_diff_at (b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b13 x p13 hb13
  let p15 := vjp_comp_diff_at (b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b14 x p14 hb14
  let p16 := vjp_comp_diff_at (b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b15 x p15 hb15
  let p17 := vjp_comp_diff_at (b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b16 x p16 hb16
  let p18 := vjp_comp_diff_at (b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b17 x p17 hb17
  let p19 := vjp_comp_diff_at (b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b18 x p18 hb18
  let p20 := vjp_comp_diff_at (b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b19 x p19 hb19
  let p21 := vjp_comp_diff_at (b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b20 x p20 hb20
  let p22 := vjp_comp_diff_at (b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b21 x p21 hb21
  let p23 := vjp_comp_diff_at (b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) hc1 x p22 hhc1
  let p24 := vjp_comp_diff_at (hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) hc2 x p23 hhc2
  let p25 := vjp_comp_diff_at (hc2 ∘ hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) head x p24 hhead
  p25.fst

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE TIE — stem and head concrete, the fused stage and 21 blocks opaque
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 800000 in
set_option maxHeartbeats 2000000 in
/-- ⭐⭐ **`mnv4InputGradB` IS the certified whole-net MobileNetV4-Conv-M gradient.** The committed
    backward chain, with its three BatchNorm and three relu-mask slots filled by the certified
    per-op backwards and its fused stage and twenty-one UIB blocks left OPAQUE, equals the
    backward of `mnv4B_full_has_vjp_at` at those twenty-six stages. `unfold`, three `rw`s, `rfl`.

    ⭐ `N` and `nCls` are both binders, so this covers the 10-class Imagenette artifacts and the
    1000-class `mnv4in` ones at every batch size — and the artifacts' `N` is the PER-REPLICA batch
    (`DataParallel.lean`, §4d). -/
theorem mnv4InputGradB_eq_mnv4B_full_vjp (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wh1 : Kernel4 960 256 1 1) (bh1 : Vec 960) (εh1 : ℝ) (hεh1 : 0 < εh1) (γh1 βh1 : Vec 960)
    (Wh : Kernel4 1280 960 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wd : Mat 1280 nCls) (bd : Vec nCls)
    (fused : Vec (N * (32 * 112 * 112)) → Vec (N * (48 * 56 * 56)))
    (b1 : Vec (N * (48 * 56 * 56)) → Vec (N * (80 * 28 * 28)))
    (b2 : Vec (N * (80 * 28 * 28)) → Vec (N * (80 * 28 * 28)))
    (b3 : Vec (N * (80 * 28 * 28)) → Vec (N * (160 * 14 * 14)))
    (b4 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b5 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b6 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b7 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b8 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b9 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b10 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b11 : Vec (N * (160 * 14 * 14)) → Vec (N * (256 * 7 * 7)))
    (b12 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b13 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b14 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b15 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b16 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b17 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b18 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b19 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b20 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b21 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (x : Vec (N * (3 * 224 * 224)))
    (h_stem : Mnv4StemSmoothAtB N 112 112 Ws bs εs γs βs x)
    (hfused : PProd (HasVJPAt fused (opaqueA0 (mnv4StemB N 112 112 Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ fused (opaqueA0 (mnv4StemB N 112 112 Ws bs εs γs βs) x)))
    (hb1 : PProd (HasVJPAt b1 (opaqueA1 (mnv4StemB N 112 112 Ws bs εs γs βs) fused x))
                 (DifferentiableAt ℝ b1 (opaqueA1 (mnv4StemB N 112 112 Ws bs εs γs βs) fused x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA2 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA2 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA3 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA3 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA4 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA4 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA5 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA5 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA6 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA6 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA7 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA7 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA8 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA8 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA9 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA9 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA10 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA10 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA11 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA11 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA12 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA12 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA13 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA13 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA14 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA14 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA15 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA15 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA16 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA16 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA17 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA17 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (hb18 : PProd (HasVJPAt b18 (opaqueA18 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
                 (DifferentiableAt ℝ b18 (opaqueA18 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
    (hb19 : PProd (HasVJPAt b19 (opaqueA19 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
                 (DifferentiableAt ℝ b19 (opaqueA19 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x)))
    (hb20 : PProd (HasVJPAt b20 (opaqueA20 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
                 (DifferentiableAt ℝ b20 (opaqueA20 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x)))
    (hb21 : PProd (HasVJPAt b21 (opaqueA21 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
                 (DifferentiableAt ℝ b21 (opaqueA21 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x)))
    (h_h1 : ∀ k, StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
      (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) k ≠ 0)
    (h_h2 : ∀ k, StableHLO.bnBatchLA N 1280 7 7 εh γh βh
      (StableHLO.batchMap N (flatConv Wh bh) (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x)) k ≠ 0) :
    mnv4InputGradB N Ws Wh1 Wh Wd
      ((bnBatchLA_has_vjp N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((bnBatchLA_has_vjp N 960 7 7 εh1 hεh1 γh1 βh1).backward
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)))
      ((bnBatchLA_has_vjp N 1280 7 7 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x)))
      hfused.fst.backward
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
      hb18.fst.backward
      hb19.fst.backward
      hb20.fst.backward
      hb21.fst.backward
      (fun i => StableHLO.bnBatchLA N 32 112 112 εs γs βs
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i > 0)
      (fun i => StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) i > 0)
      (fun i => StableHLO.bnBatchLA N 1280 7 7 εh γh βh
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x)) i > 0)
      = (mnv4B_full_has_vjp_at
          (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21
          (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1)
          (cbReluB N (h := 7) (w := 7) Wh bh εh γh βh)
          (r34HeadB N 7 7 Wd bd)
          x
          ⟨mnv4StemB_has_vjp_at N 112 112 Ws bs εs hεs γs βs x h_stem,
            mnv4StemB_differentiableAt N 112 112 Ws bs εs hεs γs βs x h_stem⟩
          hfused hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hb18 hb19 hb20 hb21
          ⟨cbReluB_has_vjp_at N Wh1 bh1 εh1 hεh1 γh1 βh1 _ h_h1,
            cbReluB_differentiableAt N Wh1 bh1 εh1 hεh1 γh1 βh1 _ h_h1⟩
          ⟨cbReluB_has_vjp_at N Wh bh εh hεh γh βh _ h_h2,
            cbReluB_differentiableAt N Wh bh εh hεh γh βh _ h_h2⟩
          ⟨(r34HeadB_has_vjp N 7 7 Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N 7 7 Wd bd) _⟩).backward := by
  unfold mnv4InputGradB
  rw [mnv4StemBBack_eq_vjp_backward (N := N) (h := 112) (w := 112)
        (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbReluBBack_eq_vjp_backward (N := N) (h := 7) (w := 7)
        (by decide) (by decide) Wh1 bh1 εh1 hεh1 γh1 βh1
        (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x) h_h1,
      cbReluBBack_eq_vjp_backward (N := N) (h := 7) (w := 7)
        (by decide) (by decide) Wh bh εh hεh γh βh
        (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x) h_h2,
      r34HeadBBack_eq_vjp_backward (N := N) (h := 7) (w := 7) Wd bd
        (opaqueA24 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) (cbReluB N (h := 7) (w := 7) Wh bh εh γh βh) x)]
  rfl

set_option maxRecDepth 800000 in
set_option maxHeartbeats 2000000 in
/-- ⭐⭐ **The chain IS the `pdiv`-contracted Jacobian of the twenty-six-stage net** — at every
    batch size, every class count, every input, every loss cotangent and every input pixel. The
    tie above read through the apex's own `.correct`; `mobilenetv4ForwardB_full_eq_slots` below is
    what says those twenty-six stages are the committed forward. -/
theorem mnv4InputGradB_correct (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wh1 : Kernel4 960 256 1 1) (bh1 : Vec 960) (εh1 : ℝ) (hεh1 : 0 < εh1) (γh1 βh1 : Vec 960)
    (Wh : Kernel4 1280 960 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wd : Mat 1280 nCls) (bd : Vec nCls)
    (fused : Vec (N * (32 * 112 * 112)) → Vec (N * (48 * 56 * 56)))
    (b1 : Vec (N * (48 * 56 * 56)) → Vec (N * (80 * 28 * 28)))
    (b2 : Vec (N * (80 * 28 * 28)) → Vec (N * (80 * 28 * 28)))
    (b3 : Vec (N * (80 * 28 * 28)) → Vec (N * (160 * 14 * 14)))
    (b4 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b5 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b6 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b7 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b8 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b9 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b10 : Vec (N * (160 * 14 * 14)) → Vec (N * (160 * 14 * 14)))
    (b11 : Vec (N * (160 * 14 * 14)) → Vec (N * (256 * 7 * 7)))
    (b12 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b13 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b14 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b15 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b16 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b17 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b18 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b19 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b20 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (b21 : Vec (N * (256 * 7 * 7)) → Vec (N * (256 * 7 * 7)))
    (x : Vec (N * (3 * 224 * 224)))
    (h_stem : Mnv4StemSmoothAtB N 112 112 Ws bs εs γs βs x)
    (hfused : PProd (HasVJPAt fused (opaqueA0 (mnv4StemB N 112 112 Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ fused (opaqueA0 (mnv4StemB N 112 112 Ws bs εs γs βs) x)))
    (hb1 : PProd (HasVJPAt b1 (opaqueA1 (mnv4StemB N 112 112 Ws bs εs γs βs) fused x))
                 (DifferentiableAt ℝ b1 (opaqueA1 (mnv4StemB N 112 112 Ws bs εs γs βs) fused x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA2 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA2 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA3 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA3 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA4 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA4 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA5 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA5 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA6 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA6 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA7 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA7 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA8 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA8 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA9 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA9 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA10 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA10 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA11 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA11 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA12 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA12 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA13 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA13 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA14 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA14 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA15 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA15 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA16 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA16 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (hb17 : PProd (HasVJPAt b17 (opaqueA17 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                 (DifferentiableAt ℝ b17 (opaqueA17 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (hb18 : PProd (HasVJPAt b18 (opaqueA18 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
                 (DifferentiableAt ℝ b18 (opaqueA18 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x)))
    (hb19 : PProd (HasVJPAt b19 (opaqueA19 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
                 (DifferentiableAt ℝ b19 (opaqueA19 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x)))
    (hb20 : PProd (HasVJPAt b20 (opaqueA20 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
                 (DifferentiableAt ℝ b20 (opaqueA20 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x)))
    (hb21 : PProd (HasVJPAt b21 (opaqueA21 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
                 (DifferentiableAt ℝ b21 (opaqueA21 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x)))
    (h_h1 : ∀ k, StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
      (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) k ≠ 0)
    (h_h2 : ∀ k, StableHLO.bnBatchLA N 1280 7 7 εh γh βh
      (StableHLO.batchMap N (flatConv Wh bh) (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x)) k ≠ 0)
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * 224 * 224))) :
    mnv4InputGradB N Ws Wh1 Wh Wd
      ((bnBatchLA_has_vjp N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((bnBatchLA_has_vjp N 960 7 7 εh1 hεh1 γh1 βh1).backward
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)))
      ((bnBatchLA_has_vjp N 1280 7 7 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x)))
      hfused.fst.backward
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
      hb18.fst.backward
      hb19.fst.backward
      hb20.fst.backward
      hb21.fst.backward
      (fun i => StableHLO.bnBatchLA N 32 112 112 εs γs βs
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) i > 0)
      (fun i => StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) i > 0)
      (fun i => StableHLO.bnBatchLA N 1280 7 7 εh γh βh
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x)) i > 0)
      dy i
      = ∑ j : Fin (N * nCls),
          pdiv (r34HeadB N 7 7 Wd bd
          ∘ cbReluB N (h := 7) (w := 7) Wh bh εh γh βh
          ∘ cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1
          ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1
          ∘ fused ∘ mnv4StemB N 112 112 Ws bs εs γs βs) x i j * dy j := by
  rw [congrFun (mnv4InputGradB_eq_mnv4B_full_vjp N Ws bs εs hεs γs βs Wh1 bh1 εh1 hεh1 γh1 βh1
    Wh bh εh hεh γh βh Wd bd fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x h_stem hfused
    hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hb18 hb19 hb20 hb21 h_h1 h_h2) dy]
  exact (mnv4B_full_has_vjp_at
          (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21
          (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1)
          (cbReluB N (h := 7) (w := 7) Wh bh εh γh βh)
          (r34HeadB N 7 7 Wd bd)
          x
          ⟨mnv4StemB_has_vjp_at N 112 112 Ws bs εs hεs γs βs x h_stem,
            mnv4StemB_differentiableAt N 112 112 Ws bs εs hεs γs βs x h_stem⟩
          hfused hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hb18 hb19 hb20 hb21
          ⟨cbReluB_has_vjp_at N Wh1 bh1 εh1 hεh1 γh1 βh1 _ h_h1,
            cbReluB_differentiableAt N Wh1 bh1 εh1 hεh1 γh1 βh1 _ h_h1⟩
          ⟨cbReluB_has_vjp_at N Wh bh εh hεh γh βh _ h_h2,
            cbReluB_differentiableAt N Wh bh εh hεh γh βh _ h_h2⟩
          ⟨(r34HeadB_has_vjp N 7 7 Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N 7 7 Wd bd) _⟩).correct dy i

-- The projection lemmas this file's shape check rewrites with — `CertLayer.comp_fwd_apply`, the
-- three layer `_fwd_apply`s, `r34HeadB_apply` and the per-group `mnv4Res*Layer_fwd_apply` — live
-- beside the things they project (`CertifiedChain.lean`, `MobileNetV4BackB0.lean`,
-- `ResNet34FullB.lean`, `MobileNetV4FullB.lean`). ⚠⚠ They exist because peeling one
-- `CertLayer.comp` to reach `.fwd` at MNv4's LITERAL resolutions is a kernel deterministic timeout
-- in every spelling that does the peel here (`rfl`, `simp only [.., Function.comp_apply]`,
-- Mathlib's `Function.comp_assoc`), and 2 s through a lemma proved between variables and APPLIED.
-- The head's stays here: it reads the tail through ResNet-34's `r34HeadB`, which is this file's
-- spelling (the apex is `r34B_full_has_vjp_at`'s), not `MobileNetV4FullB`'s.

/-- The **head**, as its two 1×1 conv-BN-relu stages and ResNet-34's GAP-and-dense tail. -/
theorem mnv4HeadStack_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (256 * 7 * 7))) :
    (mnv4HeadStack N w).fwd v
      = r34HeadB N 7 7 w.Wd w.bd
          (cbReluB N (h := 7) (w := 7) w.hW w.hb w.hE w.hg w.hbt
            (cbReluB N (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt v)) := by
  simp only [mnv4HeadStack, mnv4Head, CertLayer.comp_fwd_apply, mnv4ExpandLayer_fwd_apply,
    mnv4GapLayer_fwd_apply, mnv4DenseLayer_fwd_apply, r34HeadB_apply]

/-- The forward at GROUP granularity — `rfl`, because `mobilenetv4ForwardB_full` IS the nest of
    `mnv4Pre0 … mnv4Pre6`. Nothing is peeled here; the seven `.fwd`s stay folded. -/
theorem mnv4_fwd_eq_groups (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardB_full N w x
      = (mnv4HeadStack N w).fwd
          ((mnv4Res7bLayer N w).fwd
            ((mnv4Res7aLayer N w).fwd
              ((mnv4Res14bLayer N w).fwd
                ((mnv4Res14aLayer N w).fwd
                  ((mnv4Res28Layer N w).fwd
                    ((mnv4FusedStack N w).fwd
                      (mnv4StemB N 112 112 w.sW w.sb w.sE w.sg w.sbt x))))))) := rfl

/-- The twenty-six-stage composition, APPLIED — the apex's chain read as a nested application.
    Generic in every dimension and every stage, so it costs nothing here. -/
theorem mnv4Chain_apply {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 : Nat}
    (stem : Vec s0 → Vec s1)
    (fused : Vec s1 → Vec s2)
    (b1 : Vec s2 → Vec s3)
    (b2 : Vec s3 → Vec s4)
    (b3 : Vec s4 → Vec s5)
    (b4 : Vec s5 → Vec s6)
    (b5 : Vec s6 → Vec s7)
    (b6 : Vec s7 → Vec s8)
    (b7 : Vec s8 → Vec s9)
    (b8 : Vec s9 → Vec s10)
    (b9 : Vec s10 → Vec s11)
    (b10 : Vec s11 → Vec s12)
    (b11 : Vec s12 → Vec s13)
    (b12 : Vec s13 → Vec s14)
    (b13 : Vec s14 → Vec s15)
    (b14 : Vec s15 → Vec s16)
    (b15 : Vec s16 → Vec s17)
    (b16 : Vec s17 → Vec s18)
    (b17 : Vec s18 → Vec s19)
    (b18 : Vec s19 → Vec s20)
    (b19 : Vec s20 → Vec s21)
    (b20 : Vec s21 → Vec s22)
    (b21 : Vec s22 → Vec s23)
    (hc1 : Vec s23 → Vec s24)
    (hc2 : Vec s24 → Vec s25)
    (head : Vec s25 → Vec s26)
    (x : Vec s0) :
    (head ∘ hc2 ∘ hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) x
      = head (hc2 (hc1 (b21 (b20 (b19 (b18 (b17 (b16 (b15 (b14 (b13 (b12 (b11 (b10 (b9 (b8 (b7 (b6 (b5 (b4 (b3 (b2 (b1 (fused (stem (x)))))))))))))))))))))))))) := rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE SHAPE CHECK — the twenty-six slots ARE the committed forward
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 800000 in
/-- ⭐⭐ **The twenty-six slots the tie is about ARE `mobilenetv4ForwardB_full`.** The committed
    forward, regrouped into exactly the twenty-six arguments `mnv4B_full_has_vjp_at` takes: the
    XLA-`SAME` stem, the fused stage, the three pre-strided rows (1, 3, 11) as
    `mnv4PreStridedBodyOfRow`, the eighteen skip rows as `CertLayer.residual` of `mnv4BodyOfRow`
    at their own table rows, the head's two `cbReluB`s and ResNet-34's GAP-and-dense tail.

    ⛔ **This is the theorem that would have caught ResNet-34's wrong pool** (§3.10) — the tie
    keeps its blocks opaque, so its subject is a chain of VARIABLES and nothing in it says which
    net they are. ⭐ And it is what carries the block table into T6: every slot names its row, so
    rows 4/5/10, 12/18 and 15/19/20 — shape-identical, hence interchangeable to every type and
    `#guard` — are pinned here by the row constant, as T2's SSA names pin them in the graph.

    ⭐ Seven rewrites of lemmas proved above, and every one of them was proved either at group
    granularity (where nothing is peeled) or between variables. Nothing in this proof discharges
    a `CertLayer` peel at a literal resolution, which is why it takes two seconds. -/
theorem mobilenetv4ForwardB_full_eq_slots (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardB_full N w x
      = (r34HeadB N 7 7 w.Wd w.bd
          ∘ cbReluB N (h := 7) (w := 7) w.hW w.hb w.hE w.hg w.hbt
          ∘ cbReluB N (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row21 w.b21)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row20 w.b20)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row19 w.b19)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row18 w.b18)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row17 w.b17)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row16 w.b16)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row15 w.b15)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row14 w.b14)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row13 w.b13)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row12 w.b12)).fwd
          ∘ (mnv4PreStridedBodyOfRow N mnv4Row11 w.b11).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row10 w.b10)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row9 w.b9)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row8 w.b8)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row7 w.b7)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row6 w.b6)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row5 w.b5)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row4 w.b4)).fwd
          ∘ (mnv4PreStridedBodyOfRow N mnv4Row3 w.b3).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2)).fwd
          ∘ (mnv4PreStridedBodyOfRow N mnv4Row1 w.b1).fwd
          ∘ (mnv4FusedStack N w).fwd
          ∘ mnv4StemB N 112 112 w.sW w.sb w.sE w.sg w.sbt) x := by
  rw [mnv4Chain_apply, mnv4_fwd_eq_groups, mnv4HeadStack_fwd_apply, mnv4Res7bLayer_fwd_apply,
    mnv4Res7aLayer_fwd_apply, mnv4Res14bLayer_fwd_apply, mnv4Res14aLayer_fwd_apply,
    mnv4Res28Layer_fwd_apply]

end Proofs
