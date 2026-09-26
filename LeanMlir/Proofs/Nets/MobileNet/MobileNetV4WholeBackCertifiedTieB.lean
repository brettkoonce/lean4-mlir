import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBVJP
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains

/-! # `mnv4InputGradB` and the certified whole-net MobileNetV4-Conv-M input VJP

Ties the batched input-gradient chain `mnv4InputGradB` to a certified VJP for
`mobilenetv4ForwardBFull`: the MobileNetV4 peer of `MobileNetV2WholeBackCertifiedTieB.lean`.

No accuracy is quoted for this net. The statements are about timm's `mobilenetv4_conv_medium`.

## The four pieces

1. The concrete endpoint stage ties: the stem's (ResNet's symmetric strided conv-bn-relu tie),
   the first head conv's (`cbReluBBack_eq_vjp_backward`), `conv_head`'s with the pool in front of
   it (`mnv4Hc2BBack_eq_vjp_backward`), and the classifier's with its relabel
   (`mnv4ClsBBack_eq_vjp_backward`).
2. `mnv4BFullHasVJPAt` — the generic **twenty-six-stage** apex
   `cls ∘ hc2 ∘ hc1 ∘ b21 ∘ … ∘ b1 ∘ fused ∘ stem`, twenty-five `vjpCompDiffAt`s and nothing
   else, with `opaqueA0 … A24` naming the running activations. timm's head is three stages here:
   `hc1` (conv-bn-relu at 7×7), `hc2` (GAP, relabel to `[N, 960, 1, 1]`, `conv_head`-bn-relu) and
   `cls` (relabel to `[N, 1280]`, dense).
3. `mnv4InputGradB_eq_mnv4B_full_vjp` and `mnv4InputGradB_correct` — the tie, and its reading as
   `∑ pdiv … * dy`: with its fused-stage and block slots filled by certified VJPs, the chain is
   the Jacobian-transpose of the twenty-six-stage composition, at every batch size and class
   count.
4. `mobilenetv4ForwardBFull_eq_slots` — the shape check: those twenty-six stages ARE
   `mobilenetv4ForwardBFull`, the forward `mnv4FwdGraphBFull_faithful` says the typed graph
   denotes. Without it the tie would be a statement about variables.

## What is MobileNetV4's own, and what is borrowed

The stem is ResNet's symmetric strided conv-bn-relu, so its tie is ResNet's; the first head conv is
`cbReluB`. What is MobileNetV4's own is the pool stage in front of `conv_head` (GAP then a
`Fin.cast` relabel, whose certified backward along a bijection is the plain relabel,
`reindex_cast_backward`) and the relabel in front of the classifier.

## The apex is composed at block granularity, not at `MobileNetV4FullB.lean`'s groups

`mobilenetv4ForwardBFullHasVJPAt` is a chain of seven `vjpCompAt`s over the five
resolution-group `CertLayer`s, because `Mnv4SmoothAt` wants one `.ok` per group. This tie wants
the finer chain — a tie whose opaque slots are groups would say nothing about which block is which —
so the stages here are the twenty-one blocks themselves, `mnv4Blk0 … mnv4Blk21`'s
(`MobileNetV4StepTieB.lean`) stage functions. Peeling `CertLayer.comp` to reach `.fwd` inside a
group is cheap (`comp_fwd` is a generic `rfl` lemma and `MobileNetV4FullB`'s five
group-faithfulness proofs each do it); what is not payable at MobileNetV4's literal resolutions is composing the compositions,
which is why there is no `mnv4NetLayer` and why the shape check below stops at the block.

**Everything generic in its widths, instantiated only in the capstone.** Both stage ties bind
`{N ic oc h w kH kW}` and the apex binds `{s0 … s26}`; MobileNetV4's 224/112/56/28/14/7 are
literals, and stating a `den`- or width-indexed `rfl` at them lets it run into kernel timeouts.
Instantiating a proven lemma is cheap.

**The blocks stay opaque and there is no `backward_unique` step**, because instantiating a tie of this shape at the concrete blocks is a KERNEL deterministic timeout when
the witnesses are `HasVJPAt` carrying a saved activation. B0 takes that step only because swish
has no kink. The shape check is what replaces it.

It is a smooth-point statement, and MobileNetV4's kink budget is **relu, not relu6**: one
clause per site, where MobileNetV2's `.selectMid` carries two. The stem's and the two head convs'
are bound here (`conv_head`'s at the pooled `1×1`); the ones inside the fused stage and the blocks
are `CertLayer.comp`'s and are never written down.

**Scope.** One device's batch `N`: the data-parallel artifacts (`mnv4in_adamdp64*`) normalise
over the global batch, so this describes them only at `N := R·N`. It is about the input gradient;
the 233 parameter gradients are `MobileNetV4StepTieB.lean`'s tie.
-/

open Proofs.StableHLO

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The stem's concrete conv-BN-relu endpoint tie
--
--   The head's stage tie, `cbReluBBack_eq_vjp_backward` (stride-1, applied twice at `%h1W` and
--   `%hW`), is `ResNet34BackCertifiedTieB.lean`'s — net-agnostic, beside its strided peer.
-- ════════════════════════════════════════════════════════════════

/-- **The STEM tie.** `batchMap (flatConvStride2Back) ∘ bnBack ∘ reluMaskBack` IS `mnv4StemB`'s
    certified backward at a smooth point — the symmetric strided conv-bn-relu tie, since
    `mnv4StemB` is `cbReluStridedB` at the stem's widths.

    **This is one step past the artifact.** No render emits a gradient into `%x`, so
    MobileNetV4's committed backward ends at the stem conv's weight gradient, whose operand is the
    stem-BN cotangent `mnv4StemCotN` (`MobileNetV4StepTieB`). This lemma names the map that carries that
    cotangent the rest of the way to the image. -/
theorem mnv4StemBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : Mnv4StemSmoothAtB N h w Ws bs ε γ β x) :
    (StableHLO.batchMap N (flatConvStride2Back (h := h) (w := w) Ws)
        ∘ (bnBatchLAHasVJP N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x)
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc h w ε γ β
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0))
      = (mnv4StemBHasVJPAt N h w Ws bs ε hε γ β x hs).backward :=
  cbReluStridedBBack_eq_vjp_backward hkH hkW Ws bs ε hε γ β x hs

-- ════════════════════════════════════════════════════════════════
-- § The head's pool and classifier stages, and their ties
-- ════════════════════════════════════════════════════════════════

/-- A `Fin.cast` reindex's certified backward is the reverse relabel: along a bijection exactly
    one term of `reindexHasVJP`'s masked sum survives. -/
theorem reindex_cast_backward {n m : Nat} (h : n = m) (x : Vec n) (dy : Vec m) :
    (reindexHasVJP (Fin.cast h.symm)).backward x dy = fun i => dy (Fin.cast h i) := by
  funext i
  show ∑ k : Fin m, (if i = Fin.cast h.symm k then dy k else 0) = dy (Fin.cast h i)
  have hk : ∀ k : Fin m, i = Fin.cast h.symm k ↔ Fin.cast h i = k := fun k => by
    constructor
    · rintro rfl; exact Fin.ext rfl
    · rintro rfl; exact Fin.ext rfl
  simp only [hk, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

/-- **The pool stage** in front of `conv_head`: GAP, then the relabel to `[N, c, 1, 1]`. -/
@[reducible] noncomputable def mnv4PoolB (N h w : Nat) {c : Nat} :
    Vec (N * (c * h * w)) → Vec (N * (c * 1 * 1)) :=
  reindexCLM (Fin.cast (mnv4_pool11 N c).symm) ∘ StableHLO.batchMap N (globalAvgPoolFlat c h w)

theorem mnv4PoolB_differentiable (N h w : Nat) {c : Nat} :
    Differentiable ℝ (mnv4PoolB N h w (c := c)) :=
  (reindexCLM _).differentiable.comp
    (batchMap_differentiable _ (globalAvgPoolFlat_differentiable c h w))

noncomputable def mnv4PoolBHasVJP (N h w : Nat) {c : Nat} : HasVJP (mnv4PoolB N h w (c := c)) :=
  vjpComp _ _ (batchMap_differentiable _ (globalAvgPoolFlat_differentiable c h w))
    (reindexCLM _).differentiable
    (batchMapHasVJP _ (globalAvgPoolFlatHasVJP c h w) (globalAvgPoolFlat_differentiable c h w))
    (reindexHasVJP _)

/-- The pool stage's certified backward: the reverse relabel, then GAP-back. -/
theorem mnv4PoolB_backward (N h w : Nat) {c : Nat} (v : Vec (N * (c * h * w)))
    (d : Vec (N * (c * 1 * 1))) :
    (mnv4PoolBHasVJP N h w (c := c)).backward v d
      = StableHLO.batchMap N (gapBack c h w) (fun i => d (Fin.cast (mnv4_pool11 N c) i)) := by
  unfold mnv4PoolBHasVJP
  rw [vjpComp_backward, reindex_cast_backward]
  rfl

/-- **`conv_head`'s stage**: the pool, then conv-bn-relu at `1×1`, with its VJP at a point where
    that relu is away from its kink. -/
noncomputable def mnv4Hc2BHasVJPDiffAt (N h w : Nat) {mid oc : Nat}
    (W : Kernel4 oc mid 1 1) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (mid * h * w)))
    (hs : ∀ k, StableHLO.bnBatchLA N oc 1 1 ε γ β
      (StableHLO.batchMap N (flatConv W b) (mnv4PoolB N h w v)) k ≠ 0) :
    HasVJPDiffAt (cbReluB N (h := 1) (w := 1) W b ε γ β ∘ mnv4PoolB N h w) v :=
  vjpCompDiffAt (mnv4PoolB N h w) (cbReluB N (h := 1) (w := 1) W b ε γ β) v
    ⟨(mnv4PoolBHasVJP N h w).toHasVJPAt v, (mnv4PoolB_differentiable N h w) v⟩
    ⟨cbReluBHasVJPAt N W b ε hε γ β _ hs, cbReluB_differentiableAt N W b ε hε γ β _ hs⟩

/-- **`conv_head`'s stage tie**: GAP-back after the relabel, after conv-back ∘ BN-back ∘ relu mask
    at `1×1`, IS the stage's certified backward. -/
theorem mnv4Hc2BBack_eq_vjp_backward {N h w mid oc : Nat}
    (W : Kernel4 oc mid 1 1) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (mid * h * w)))
    (hs : ∀ k, StableHLO.bnBatchLA N oc 1 1 ε γ β
      (StableHLO.batchMap N (flatConv W b) (mnv4PoolB N h w v)) k ≠ 0) :
    ((StableHLO.batchMap N (gapBack mid h w)
        ∘ fun u i => u (Fin.cast (by rw [Nat.mul_one, Nat.mul_one]) i))
      ∘ (StableHLO.batchMap N (convFlatBack (h := 1) (w := 1) W)
        ∘ (bnBatchLAHasVJP N oc 1 1 ε hε γ β).backward
            (StableHLO.batchMap N (flatConv W b) (mnv4PoolB N h w v))
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc 1 1 ε γ β
            (StableHLO.batchMap N (flatConv W b) (mnv4PoolB N h w v)) i > 0)))
      = (mnv4Hc2BHasVJPDiffAt N h w W b ε hε γ β v hs).fst.backward := by
  rw [cbReluBBack_eq_vjp_backward (h := 1) (w := 1) (by decide) (by decide) W b ε hε γ β _ hs]
  funext dy
  rw [mnv4Hc2BHasVJPDiffAt, vjpCompDiffAt_fst_backward]
  show _ = (mnv4PoolBHasVJP N h w).backward v _
  rw [mnv4PoolB_backward]
  rfl

/-- **The classifier stage**: the relabel of `[N, oc, 1, 1]` to `[N, oc]`, then dense. -/
@[reducible] noncomputable def mnv4ClsB (N : Nat) {oc nCls : Nat} (Wd : Mat oc nCls)
    (bd : Vec nCls) : Vec (N * (oc * 1 * 1)) → Vec (N * nCls) :=
  StableHLO.batchMap N (dense Wd bd) ∘ reindexCLM (Fin.cast (mnv4_pool11 N oc))

theorem mnv4ClsB_differentiable (N : Nat) {oc nCls : Nat} (Wd : Mat oc nCls) (bd : Vec nCls) :
    Differentiable ℝ (mnv4ClsB N Wd bd) :=
  (batchMap_differentiable _ (dense_differentiable Wd bd)).comp (reindexCLM _).differentiable

noncomputable def mnv4ClsBHasVJP (N : Nat) {oc nCls : Nat} (Wd : Mat oc nCls) (bd : Vec nCls) :
    HasVJP (mnv4ClsB N Wd bd) :=
  vjpComp _ _ (reindexCLM _).differentiable (batchMap_differentiable _ (dense_differentiable Wd bd))
    (reindexHasVJP _) (batchMapHasVJP _ (denseHasVJP Wd bd) (dense_differentiable Wd bd))

/-- The classifier stage's certified backward: dense-back, then the reverse relabel. -/
theorem mnv4ClsB_backward (N : Nat) {oc nCls : Nat} (Wd : Mat oc nCls) (bd : Vec nCls)
    (x : Vec (N * (oc * 1 * 1))) (d : Vec (N * nCls)) :
    (mnv4ClsBHasVJP N Wd bd).backward x d
      = fun i => StableHLO.batchMap N ((denseHasVJP Wd bd).backward (fun _ => 0)) d
          (Fin.cast (mnv4_pool11 N oc).symm i) := by
  unfold mnv4ClsBHasVJP
  rw [vjpComp_backward, reindex_cast_backward]
  rfl

/-- **The classifier stage tie**: dense-back (`Wᵀ`), then the relabel, IS its certified backward. -/
theorem mnv4ClsBBack_eq_vjp_backward {N oc nCls : Nat} (Wd : Mat oc nCls) (bd : Vec nCls)
    (x : Vec (N * (oc * 1 * 1))) :
    ((fun u i => u (Fin.cast (by rw [Nat.mul_one, Nat.mul_one]) i))
      ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec oc)))
      = (mnv4ClsBHasVJP N Wd bd).backward x := by
  rw [dense_transpose_eq_vjp_backward Wd bd (fun _ => 0)]
  funext dy
  rw [mnv4ClsB_backward]
  rfl

-- The opaque running activations `opaqueA0 … opaqueA24` are `Foundation/OpaquePrefix.lean`'s.

-- ════════════════════════════════════════════════════════════════
-- § The generic twenty-six-stage apex
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network MobileNetV4-Conv-M VJP.** The VJP of the twenty-six-stage chain
    `head ∘ hc2 ∘ hc1 ∘ b21 ∘ … ∘ b1 ∘ fused ∘ stem` — twenty-five `vjpCompDiffAt`s and
    nothing else, dimension-generic and parametric in every component.

    Pointwise (`HasVJPAt`): relu is kinked, so each stage's witness carries the
    activation it sees. `r34BFullHasVJPAt` is the same construction at eighteen stages;
    MobileNetV4 needs its own because Conv-M's ladder is longer, not because anything differs. -/
noncomputable def mnv4BFullHasVJPAt {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 : Nat}
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
    (hstem : HasVJPDiffAt stem (x))
    (hfused : HasVJPDiffAt fused (opaqueA0 stem x))
    (hb1 : HasVJPDiffAt b1 (opaqueA1 stem fused x))
    (hb2 : HasVJPDiffAt b2 (opaqueA2 stem fused b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA3 stem fused b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA4 stem fused b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA5 stem fused b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA6 stem fused b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA7 stem fused b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA8 stem fused b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA9 stem fused b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA10 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA11 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA12 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA13 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA14 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA15 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA16 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA17 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (hb18 : HasVJPDiffAt b18 (opaqueA18 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (hb19 : HasVJPDiffAt b19 (opaqueA19 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
    (hb20 : HasVJPDiffAt b20 (opaqueA20 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
    (hb21 : HasVJPDiffAt b21 (opaqueA21 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
    (hhc1 : HasVJPDiffAt hc1 (opaqueA22 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x))
    (hhc2 : HasVJPDiffAt hc2 (opaqueA23 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 x))
    (hhead : HasVJPDiffAt head (opaqueA24 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 hc2 x))
    : HasVJPAt (head ∘ hc2 ∘ hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) x :=
  let p1 := vjpCompDiffAt (stem) fused x hstem hfused
  let p2 := vjpCompDiffAt (fused ∘ stem) b1 x p1 hb1
  let p3 := vjpCompDiffAt (b1 ∘ fused ∘ stem) b2 x p2 hb2
  let p4 := vjpCompDiffAt (b2 ∘ b1 ∘ fused ∘ stem) b3 x p3 hb3
  let p5 := vjpCompDiffAt (b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b4 x p4 hb4
  let p6 := vjpCompDiffAt (b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b5 x p5 hb5
  let p7 := vjpCompDiffAt (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b6 x p6 hb6
  let p8 := vjpCompDiffAt (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b7 x p7 hb7
  let p9 := vjpCompDiffAt (b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b8 x p8 hb8
  let p10 := vjpCompDiffAt (b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b9 x p9 hb9
  let p11 := vjpCompDiffAt (b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b10 x p10 hb10
  let p12 := vjpCompDiffAt (b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b11 x p11 hb11
  let p13 := vjpCompDiffAt (b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b12 x p12 hb12
  let p14 := vjpCompDiffAt (b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b13 x p13 hb13
  let p15 := vjpCompDiffAt (b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b14 x p14 hb14
  let p16 := vjpCompDiffAt (b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b15 x p15 hb15
  let p17 := vjpCompDiffAt (b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b16 x p16 hb16
  let p18 := vjpCompDiffAt (b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b17 x p17 hb17
  let p19 := vjpCompDiffAt (b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b18 x p18 hb18
  let p20 := vjpCompDiffAt (b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b19 x p19 hb19
  let p21 := vjpCompDiffAt (b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b20 x p20 hb20
  let p22 := vjpCompDiffAt (b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) b21 x p21 hb21
  let p23 := vjpCompDiffAt (b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) hc1 x p22 hhc1
  let p24 := vjpCompDiffAt (hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) hc2 x p23 hhc2
  let p25 := vjpCompDiffAt (hc2 ∘ hc1 ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ fused ∘ stem) head x p24 hhead
  p25.fst

/-- **The apex's backward, peeled** — each stage's backward in turn, head first. `rfl` over
    VARIABLE stages; the tie below instantiates it by `rw`, so the kernel never re-derives the
    concrete chain (the ResNet-34 apex's `r34BFullHasVJPAt_backward`). -/
theorem mnv4BFullHasVJPAt_backward
    {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 : Nat}
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
    (hstem : HasVJPDiffAt stem (x))
    (hfused : HasVJPDiffAt fused (opaqueA0 stem x))
    (hb1 : HasVJPDiffAt b1 (opaqueA1 stem fused x))
    (hb2 : HasVJPDiffAt b2 (opaqueA2 stem fused b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA3 stem fused b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA4 stem fused b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA5 stem fused b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA6 stem fused b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA7 stem fused b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA8 stem fused b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA9 stem fused b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA10 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA11 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA12 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA13 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA14 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA15 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA16 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA17 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (hb18 : HasVJPDiffAt b18 (opaqueA18 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (hb19 : HasVJPDiffAt b19 (opaqueA19 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
    (hb20 : HasVJPDiffAt b20 (opaqueA20 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
    (hb21 : HasVJPDiffAt b21 (opaqueA21 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
    (hhc1 : HasVJPDiffAt hc1 (opaqueA22 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x))
    (hhc2 : HasVJPDiffAt hc2 (opaqueA23 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 x))
    (hhead : HasVJPDiffAt head (opaqueA24 stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 hc2 x))
    (dy : Vec s26) :
    (mnv4BFullHasVJPAt stem fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 hc1 hc2 head x hstem hfused hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hb18 hb19 hb20 hb21 hhc1 hhc2 hhead).backward dy
      =
      hstem.fst.backward
        (hfused.fst.backward
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
                                            (hb18.fst.backward
                                              (hb19.fst.backward
                                                (hb20.fst.backward
                                                  (hb21.fst.backward
                                                    (hhc1.fst.backward
                                                      (hhc2.fst.backward
                                                        (hhead.fst.backward dy))))))))))))))))))))))))) := rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE TIE — stem and head concrete, the fused stage and 21 blocks opaque
-- ════════════════════════════════════════════════════════════════

/-- **`mnv4InputGradB` is the backward of the whole-net VJP at opaque blocks.** The committed
    backward chain, with its three BatchNorm and three relu-mask slots filled by the certified
    per-op backwards and its fused-stage and twenty-one block slots filled by the supplied
    witnesses' backwards (`hfused`, `hb1 … hb21`; the stages themselves left opaque), equals the
    backward of `mnv4BFullHasVJPAt` at those twenty-six stages, at an input where the stem and
    both head relu clauses (`h_stem`, `h_h1`, `h_h2`) hold.

    `N` and `nCls` are both binders, so this covers the 10-class Imagenette artifacts and the
    1000-class `mnv4in` ones at every batch size. On the data-parallel (sync-BN) artifacts,
    instantiate at the global batch `R·N` (see `MobileNetV4SyncB`). -/
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
    (hfused : HasVJPDiffAt fused (opaqueA0 (mnv4StemB N 112 112 Ws bs εs γs βs) x))
    (hb1 : HasVJPDiffAt b1 (opaqueA1 (mnv4StemB N 112 112 Ws bs εs γs βs) fused x))
    (hb2 : HasVJPDiffAt b2 (opaqueA2 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA3 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA4 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA5 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA6 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA7 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA8 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA9 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA10 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA11 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA12 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA13 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA14 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA15 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA16 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA17 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (hb18 : HasVJPDiffAt b18 (opaqueA18 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (hb19 : HasVJPDiffAt b19 (opaqueA19 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
    (hb20 : HasVJPDiffAt b20 (opaqueA20 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
    (hb21 : HasVJPDiffAt b21 (opaqueA21 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
    (h_h1 : ∀ k, StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
      (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) k ≠ 0)
    (h_h2 : ∀ k, StableHLO.bnBatchLA N 1280 1 1 εh γh βh
      (StableHLO.batchMap N (flatConv Wh bh) (mnv4PoolB N 7 7 (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x))) k ≠ 0) :
    mnv4InputGradB N Ws Wh1 Wh Wd
      ((bnBatchLAHasVJP N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x))
      ((bnBatchLAHasVJP N 960 7 7 εh1 hεh1 γh1 βh1).backward
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)))
      ((bnBatchLAHasVJP N 1280 1 1 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (mnv4PoolB N 7 7 (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x))))
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
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0)
      (fun i => StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) i > 0)
      (fun i => StableHLO.bnBatchLA N 1280 1 1 εh γh βh
        (StableHLO.batchMap N (flatConv Wh bh) (mnv4PoolB N 7 7 (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x))) i > 0)
      = (mnv4BFullHasVJPAt
          (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21
          (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1)
          (cbReluB N (h := 1) (w := 1) Wh bh εh γh βh ∘ mnv4PoolB N 7 7)
          (mnv4ClsB N Wd bd)
          x
          ⟨mnv4StemBHasVJPAt N 112 112 Ws bs εs hεs γs βs x h_stem,
            mnv4StemB_differentiableAt N 112 112 Ws bs εs hεs γs βs x h_stem⟩
          hfused hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hb18 hb19 hb20 hb21
          ⟨cbReluBHasVJPAt N Wh1 bh1 εh1 hεh1 γh1 βh1 _ h_h1,
            cbReluB_differentiableAt N Wh1 bh1 εh1 hεh1 γh1 βh1 _ h_h1⟩
          (mnv4Hc2BHasVJPDiffAt N 7 7 Wh bh εh hεh γh βh _ h_h2)
          ⟨(mnv4ClsBHasVJP N Wd bd).toHasVJPAt _,
            (mnv4ClsB_differentiable N Wd bd) _⟩).backward := by
  unfold mnv4InputGradB
  rw [mnv4StemBBack_eq_vjp_backward (N := N) (h := 112) (w := 112)
        (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbReluBBack_eq_vjp_backward (N := N) (h := 7) (w := 7)
        (by decide) (by decide) Wh1 bh1 εh1 hεh1 γh1 βh1
        (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x) h_h1,
      mnv4Hc2BBack_eq_vjp_backward (N := N) (h := 7) (w := 7) Wh bh εh hεh γh βh
        (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x) h_h2,
      mnv4ClsBBack_eq_vjp_backward (N := N) Wd bd
        (opaqueA24 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) (cbReluB N (h := 1) (w := 1) Wh bh εh γh βh ∘ mnv4PoolB N 7 7) x)]
  funext dy
  rw [mnv4BFullHasVJPAt_backward]
  repeat rw [Function.comp_apply]
  rfl

/-- **The chain is the `pdiv`-contracted Jacobian of the twenty-six-stage composition** — at
    every batch size, class count and loss cotangent, and at every input where the stem and both
    head relu clauses hold (`h_stem`, `h_h1`, `h_h2`): the chain, with its fused-stage and
    twenty-one block slots filled by the supplied certified VJPs at the running activations
    (`hfused`, `hb1 … hb21`), is the Jacobian-transpose of the composition of those stages. The
    stages are variables here; `mobilenetv4ForwardBFull_eq_slots` identifies them with the
    committed forward. -/
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
    (hfused : HasVJPDiffAt fused (opaqueA0 (mnv4StemB N 112 112 Ws bs εs γs βs) x))
    (hb1 : HasVJPDiffAt b1 (opaqueA1 (mnv4StemB N 112 112 Ws bs εs γs βs) fused x))
    (hb2 : HasVJPDiffAt b2 (opaqueA2 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA3 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA4 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA5 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA6 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA7 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA8 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA9 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA10 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA11 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA12 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA13 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA14 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA15 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA16 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (hb17 : HasVJPDiffAt b17 (opaqueA17 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (hb18 : HasVJPDiffAt b18 (opaqueA18 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 x))
    (hb19 : HasVJPDiffAt b19 (opaqueA19 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 x))
    (hb20 : HasVJPDiffAt b20 (opaqueA20 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 x))
    (hb21 : HasVJPDiffAt b21 (opaqueA21 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 x))
    (h_h1 : ∀ k, StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
      (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) k ≠ 0)
    (h_h2 : ∀ k, StableHLO.bnBatchLA N 1280 1 1 εh γh βh
      (StableHLO.batchMap N (flatConv Wh bh) (mnv4PoolB N 7 7 (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x))) k ≠ 0)
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * 224 * 224))) :
    mnv4InputGradB N Ws Wh1 Wh Wd
      ((bnBatchLAHasVJP N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x))
      ((bnBatchLAHasVJP N 960 7 7 εh1 hεh1 γh1 βh1).backward
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)))
      ((bnBatchLAHasVJP N 1280 1 1 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (mnv4PoolB N 7 7 (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x))))
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
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0)
      (fun i => StableHLO.bnBatchLA N 960 7 7 εh1 γh1 βh1
        (StableHLO.batchMap N (flatConv Wh1 bh1) (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x)) i > 0)
      (fun i => StableHLO.bnBatchLA N 1280 1 1 εh γh βh
        (StableHLO.batchMap N (flatConv Wh bh) (mnv4PoolB N 7 7 (opaqueA23 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 (cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1) x))) i > 0)
      dy i
      = ∑ j : Fin (N * nCls),
          pdiv (mnv4ClsB N Wd bd
          ∘ (cbReluB N (h := 1) (w := 1) Wh bh εh γh βh ∘ mnv4PoolB N 7 7)
          ∘ cbReluB N (h := 7) (w := 7) Wh1 bh1 εh1 γh1 βh1
          ∘ b21 ∘ b20 ∘ b19 ∘ b18 ∘ b17 ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1
          ∘ fused ∘ mnv4StemB N 112 112 Ws bs εs γs βs) x i j * dy j := by
  exact HasVJPAt.correct_of_backward_eq _ (mnv4InputGradB_eq_mnv4B_full_vjp N Ws bs εs hεs γs βs Wh1 bh1 εh1 hεh1 γh1 βh1
    Wh bh εh hεh γh βh Wd bd fused b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 x h_stem hfused
    hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hb17 hb18 hb19 hb20 hb21 h_h1 h_h2) dy i

-- The projection lemmas this file's shape check rewrites with — `CertLayer.comp_fwd_apply`, the
-- three layer `_fwd_apply`s, `r34HeadB_apply` and the per-group `mnv4Res*Layer_fwd_apply` — live
-- beside the things they project (`CertifiedChain.lean`, `MobileNetV4BackB0.lean`,
-- `ResNet34FullB.lean`, `MobileNetV4FullB.lean`). ⚠⚠ They exist because peeling one
-- `CertLayer.comp` to reach `.fwd` at MNv4's LITERAL resolutions is a kernel deterministic timeout
-- in every spelling that does the peel here (`rfl`, `simp only [.., Function.comp_apply]`,
-- Mathlib's `Function.comp_assoc`), and 2 s through a lemma proved between variables and APPLIED.
-- The head's stays here: it reads the tail through the tie's own stages (`mnv4PoolB`, `mnv4ClsB`), which are this file's
-- spelling (the apex is `r34BFullHasVJPAt`'s), not `MobileNetV4FullB`'s.

/-- The **head**, as the tie's three stages: `cn_960` conv-BN-relu, the pool with `conv_head`, the
    relabelled classifier. -/
theorem mnv4HeadStack_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (256 * 7 * 7))) :
    (mnv4HeadStack N w).fwd v
      = mnv4ClsB N w.Wd w.bd
          ((cbReluB N (h := 1) (w := 1) w.hW w.hb w.hE w.hg w.hbt ∘ mnv4PoolB N 7 7)
            (cbReluB N (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt v)) := by
  simp only [mnv4HeadStack, mnv4Head, CertLayer.comp_fwd_apply, cbReluLayer_fwd_apply,
    gapLayer_fwd_apply, denseLayer_fwd_apply, castLayer_fwd_apply]
  rfl

/-- The forward at GROUP granularity — `rfl`, because `mobilenetv4ForwardBFull` IS the nest of
    `mnv4Pre0 … mnv4Pre6`. Nothing is peeled here; the seven `.fwd`s stay folded. -/
theorem mnv4_fwd_eq_groups (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardBFull N w x
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

/-- **The twenty-six slots the tie is about are `mobilenetv4ForwardBFull`.** The committed
    forward, regrouped into exactly the twenty-six arguments `mnv4BFullHasVJPAt` takes: the
    symmetric stem, the fused stage, the three strided rows (1, 3, 11) as `mnv4StridedBodyOfRow`,
    the eighteen skip rows as `CertLayer.residual` of `mnv4BodyOfRow` at their own table rows, and
    the head's three stages (`cn_960`, the pool with `conv_head`, the relabelled classifier).

    The tie keeps its blocks opaque, so its subject is a chain of variables and nothing in it
    says which net they are; this theorem does. It also carries the block table into the tie:
    every slot names its row, so rows 4/5/7, 8/10, 12/18, 13/14 and 15/19/20 — shape-identical,
    hence interchangeable to every type and `#guard` — are pinned here by the row constant, as
    `mnv4FwdGraphBFull`'s SSA names pin them in the graph. -/
theorem mobilenetv4ForwardBFull_eq_slots (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardBFull N w x
      = (mnv4ClsB N w.Wd w.bd
          ∘ (cbReluB N (h := 1) (w := 1) w.hW w.hb w.hE w.hg w.hbt ∘ mnv4PoolB N 7 7)
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
          ∘ (mnv4StridedBodyOfRow N mnv4Row11 w.b11).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row10 w.b10)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row9 w.b9)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row8 w.b8)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row7 w.b7)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row6 w.b6)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row5 w.b5)).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row4 w.b4)).fwd
          ∘ (mnv4StridedBodyOfRow N mnv4Row3 w.b3).fwd
          ∘ (CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2)).fwd
          ∘ (mnv4StridedBodyOfRow N mnv4Row1 w.b1).fwd
          ∘ (mnv4FusedStack N w).fwd
          ∘ mnv4StemB N 112 112 w.sW w.sb w.sE w.sg w.sbt) x := by
  -- Seven rewrites of lemmas proved above, each proved at group granularity (nothing peeled) or
  -- between variables: no `CertLayer` peel is discharged at a literal resolution here.
  rw [mnv4Chain_apply, mnv4_fwd_eq_groups, mnv4HeadStack_fwd_apply, mnv4Res7bLayer_fwd_apply,
    mnv4Res7aLayer_fwd_apply, mnv4Res14bLayer_fwd_apply, mnv4Res14aLayer_fwd_apply,
    mnv4Res28Layer_fwd_apply]

end Proofs
