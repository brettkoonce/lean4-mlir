import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie
import LeanMlir.Proofs.Nets.ResNet.ResNetBackChains

/-! # `r34InputGradB` IS the certified whole-net ResNet-34 gradient AT BATCH BATCH-NORM

The input-gradient tie for the net the shipped trainers run: `resnet34ForwardBFull`, the
[3,4,6,3] ladder at **`bnBatchLA`**, at a variable batch `N`. The leaf ties it rests on are
`ConvBackCertifiedTie`'s.

Nothing here is new mathematics. Two endpoint stage ties (stem and head) and one pool tie, then
the sixteen basic blocks stay **opaque** — they enter as `HasVJPDiffAt` witnesses, and the reverse
chain's block slots are pinned to their `.backward` — so the composition is checked between
variables. The chain is peeled by `rw` with a lemma proved over variable stages, so the module
checks with no budget raised.

## The four pieces

1. `r34HeadBBack_eq_vjp_backward` and `ConvBackCertifiedTie`'s stage tie
   `cbReluStridedBBack_eq_vjp_backward` — the concrete endpoints (head and stem); the pool needs
   no lemma.
   The pool endpoint is **`rfl`**: `batchMapHasVJPAt` and `maxPool3s2FlatHasVJPAtVec` are built
   field by field rather than transported with `▸`, so `batchMapAux` of the leaf and the lift's
   `.backward` are the same term.
2. `r34BFullHasVJPAt` — the generic eighteen-stage apex `head ∘ b16 ∘ … ∘ b1 ∘ stem`,
   seventeen nested `vjpCompDiffAt`s and nothing else, over `opaqueA0 … A16`, one prefix `def`
   per running activation — and `r34BFullHasVJPAt_backward`, its backward peeled into the
   eighteen stage backwards, `rfl` at variable stages.
3. `r34InputGradB_eq_r34B_full_vjp` and `r34InputGradB_correct` — the tie, and its reading as
   `∑ pdiv … * dy`: the chain IS the Jacobian-transpose of the eighteen-stage composition.
4. `resnet34ForwardBFull_eq_slots` — the shape check: those eighteen stages ARE
   `resnet34ForwardBFull`, the forward `resnet34FwdGraphBFull_faithful` says the typed
   graph denotes. Without it the tie would be a statement about variables.

## Scope

A smooth-point statement: the tie assumes the stem relu clause (`h_stem : R34StemSmoothAt …`),
the stem pool's per-example no-tie (`h_pool : StemPoolSmoothAt …`) and `0 < εs`, and takes each of
the sixteen blocks as an opaque `HasVJPDiffAt` witness at its running activation (a basic block's
two relu clauses are the caller's, inside that witness). One device: the data-parallel step,
collectives included, is `ResNet34SyncStepTieB`'s. It is about the INPUT gradient; the parameter
gradients are `ResNet34StepTieB`'s tie.
-/

-- Build notes (two walls, both measured):
-- * A `rfl` straight at `resnet34ForwardBFullHasVJPAt` does not terminate — five minutes to
--   `(deterministic) timeout at isDefEq` at four million heartbeats. That is why the generic apex
--   `r34BFullHasVJPAt` exists.
-- * Instantiating the generic tie at the sixteen concrete blocks is a *kernel* deterministic
--   timeout at six minutes, with `HasVJPAt.backward_unique` or without it. So this file stops at
--   opaque blocks plus a shape check (as MobileNetV2's per-example tie does), not concrete blocks
--   then `backward_unique` (as B0's does). The difference is the kink: B0's generic tie takes
--   GLOBAL `HasVJP` witnesses, which carry no point, so instantiating them is free; r34's are
--   `HasVJPAt` at `opaqueA{k-1} … x`, and the witnesses a caller has are at `r34Pre{k-1} N w x` —
--   sixteen defeq checks between two sixteen-deep nested applications spelled through different
--   definition chains.
-- * Not a closing `rfl` in the tie itself: written as seventeen `let`s and closed that way, the tie
--   cost ~45 s and 8 GB under `maxRecDepth 800000`.

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The concrete endpoint ties
-- ════════════════════════════════════════════════════════════════

/-- **The HEAD tie.** `batchMap (gapBack) ∘ batchMap (dense Wᵀ 0)` IS `r34HeadB`'s certified
    backward. The head takes no hypothesis at all: GAP and dense are smooth and each is
    `batchMap` of a per-example op, so `r34HeadBHasVJP` is GLOBAL — the one place in this net
    where the certified backward comes with nothing attached. `gapBack` needs no rewrite; it is
    definitionally the global-average-pool VJP's backward. -/
theorem r34HeadBBack_eq_vjp_backward {N c nCls h w : Nat}
    (Wd : Mat c nCls) (bd : Vec nCls) (x : Vec (N * (c * h * w))) :
    (StableHLO.batchMap N (gapBack c h w)
      ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec c)))
      = (r34HeadBHasVJP N h w Wd bd).backward x := by
  rw [dense_transpose_eq_vjp_backward Wd bd (fun _ => 0)]
  rfl

/-- The stem's backward is the pool's, then the conv-BN-relu stage's. `rfl` at variable widths;
    the tie rewrites with it rather than leaving this step to its closing `rfl`, which reaches
    `maxRecDepth` at the net's numerals. -/
theorem r34StemBHasVJPAt_backward (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (hc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hrelu : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpool : StemPoolSmoothAt N h w
      (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x))
    (v : Vec (N * (oc * h * w))) :
    (r34StemBHasVJPAt N h w Ws bs εs hεs γs βs hc hh hw x hrelu hpool).backward v
      = (StableHLO.cbReluStridedBHasVJPAt N Ws bs εs hεs γs βs x hrelu).backward
          (maxPool3s2FlatBackB N oc h w
            (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) v) := rfl

-- The opaque running activations `opaqueA0 … opaqueA16` are `Foundation/OpaquePrefix.lean`'s.

-- ════════════════════════════════════════════════════════════════
-- § The apex — a straight 18-stage chain, every stage opaque
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network batched ResNet-34 VJP, every stage opaque.** `head ∘ b16 ∘ … ∘ b1 ∘ stem`,
    seventeen `vjpCompDiffAt`s and nothing else. ResNet-34's [3,4,6,3] ladder needs no
    list of blocks and no separate downsample slot: a downsample block is just a block of a
    different type, and the stem's pool lives INSIDE `stem`. Dimension-generic and parametric in
    every component, so the tie below is checked between variables.

    Pointwise (`HasVJPAt`), as every ResNet-34 statement is: relu is kinked, and this net has
    two relu sites per block. -/
noncomputable def r34BFullHasVJPAt {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 : Nat}
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
    (head : Vec s17 → Vec s18)
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
    (hhead : HasVJPDiffAt head (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    : HasVJPAt (head ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) x :=
  (vjpCompDiffAt (b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) head x
    (vjpCompDiffAt (b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b16 x
      (vjpCompDiffAt (b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b15 x
        (vjpCompDiffAt (b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b14 x
          (vjpCompDiffAt (b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b13 x
            (vjpCompDiffAt (b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b12 x
              (vjpCompDiffAt (b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b11 x
                (vjpCompDiffAt (b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b10 x
                  (vjpCompDiffAt (b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b9 x
                    (vjpCompDiffAt (b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b8 x
                      (vjpCompDiffAt (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b7 x
                        (vjpCompDiffAt (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b6 x
                          (vjpCompDiffAt (b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b5 x
                            (vjpCompDiffAt (b3 ∘ b2 ∘ b1 ∘ stem) b4 x
                              (vjpCompDiffAt (b2 ∘ b1 ∘ stem) b3 x
                                (vjpCompDiffAt (b1 ∘ stem) b2 x
                                  (vjpCompDiffAt stem b1 x
                                    hstem hb1)
                                  hb2)
                                hb3)
                              hb4)
                            hb5)
                          hb6)
                        hb7)
                      hb8)
                    hb9)
                  hb10)
                hb11)
              hb12)
            hb13)
          hb14)
        hb15)
      hb16)
    hhead).fst



/-- **The apex's backward, peeled** — the head's backward, then each block's, then the stem's.
    `rfl` over VARIABLE stages, where every witness sits at the same `opaqueA` point on both
    sides; the tie below instantiates it by `rw`, so the kernel never re-derives the chain. -/
theorem r34BFullHasVJPAt_backward {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 : Nat}
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
    (head : Vec s17 → Vec s18)
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
    (hhead : HasVJPDiffAt head (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
    (dy : Vec s18) :
    (r34BFullHasVJPAt stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 head x hstem hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hhead).backward dy
      = hstem.fst.backward
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
                                          (hhead.fst.backward dy))))))))))))))))) := rfl

-- ════════════════════════════════════════════════════════════════
-- § THE TIE — stem, pool and head concrete, the sixteen blocks opaque
-- ════════════════════════════════════════════════════════════════

/-- **`r34InputGradB` IS the certified whole-net batch-BN ResNet-34 gradient.** The committed
    backward chain, with its stem BatchNorm and relu-mask slots filled by the certified per-op
    backwards, its saved pool activation the stem's own, and its sixteen basic blocks left OPAQUE,
    equals the backward of `r34BFullHasVJPAt` at those eighteen stages. `unfold`, the two
    endpoint `rw`s, the apex and stem peels, then `rfl` between identical block backwards — the
    blocks are variables on both sides, so the kernel never looks inside one. -/
theorem r34InputGradB_eq_r34B_full_vjp (N : Nat) {nCls : Nat}
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 64)
    (Wd : Mat 512 nCls) (bd : Vec nCls)
    (b1 : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (b2 : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (b3 : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (b4 : Vec (N * (64 * 56 * 56)) → Vec (N * (128 * 28 * 28)))
    (b5 : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (b6 : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (b7 : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (b8 : Vec (N * (128 * 28 * 28)) → Vec (N * (256 * 14 * 14)))
    (b9 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b10 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b11 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b12 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b13 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b14 : Vec (N * (256 * 14 * 14)) → Vec (N * (512 * 7 * 7)))
    (b15 : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (b16 : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (h_stem : R34StemSmoothAt N 56 56 Ws bs εs γs βs x)
    (h_pool : StemPoolSmoothAt N 56 56
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) Ws bs εs γs βs x))
    (hb1 : HasVJPDiffAt b1 (opaqueA0 (r34StemB N 56 56 Ws bs εs γs βs) x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 (r34StemB N 56 56 Ws bs εs γs βs) b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)) :
    r34InputGradB N Ws Wd
      ((bnBatchLAHasVJP N 64 (2 * 56) (2 * 56) εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x))
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) Ws bs εs γs βs x)
      hb16.fst.backward
      hb15.fst.backward
      hb14.fst.backward
      hb13.fst.backward
      hb12.fst.backward
      hb11.fst.backward
      hb10.fst.backward
      hb9.fst.backward
      hb8.fst.backward
      hb7.fst.backward
      hb6.fst.backward
      hb5.fst.backward
      hb4.fst.backward
      hb3.fst.backward
      hb2.fst.backward
      hb1.fst.backward
      (fun i => StableHLO.bnBatchLA N 64 (2 * 56) (2 * 56) εs γs βs
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0)
      = (r34BFullHasVJPAt (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16
          (r34HeadB N 7 7 Wd bd) x
          ⟨r34StemBHasVJPAt N 56 56 Ws bs εs hεs γs βs
              (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool,
            r34StemB_differentiableAt N 56 56 Ws bs εs hεs γs βs
              (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
          ⟨(r34HeadBHasVJP N 7 7 Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N 7 7 Wd bd) _⟩).backward := by
  unfold r34InputGradB
  rw [cbReluStridedBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      r34HeadBBack_eq_vjp_backward Wd bd (opaqueA16 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)]
  funext dy
  rw [r34BFullHasVJPAt_backward, r34StemBHasVJPAt_backward]
  repeat rw [Function.comp_apply]
  rfl


/-- **The batched chain IS the `pdiv`-contracted Jacobian of the eighteen-stage net** — at every
    batch size, loss cotangent and input pixel, at any input `x` where the stem relu is off its
    kink (`h_stem`) and no stem-pool window ties (`h_pool`), for any block maps `b1 … b16` carrying
    `HasVJPDiffAt` witnesses at their running activations (`hb1 … hb16`), with `0 < εs`. The tie
    above read through the apex's own `.correct`. `resnet34ForwardBFull_eq_slots` below identifies
    those eighteen stages with the committed forward. -/
theorem r34InputGradB_correct (N : Nat) {nCls : Nat}
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 64)
    (Wd : Mat 512 nCls) (bd : Vec nCls)
    (b1 : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (b2 : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (b3 : Vec (N * (64 * 56 * 56)) → Vec (N * (64 * 56 * 56)))
    (b4 : Vec (N * (64 * 56 * 56)) → Vec (N * (128 * 28 * 28)))
    (b5 : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (b6 : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (b7 : Vec (N * (128 * 28 * 28)) → Vec (N * (128 * 28 * 28)))
    (b8 : Vec (N * (128 * 28 * 28)) → Vec (N * (256 * 14 * 14)))
    (b9 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b10 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b11 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b12 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b13 : Vec (N * (256 * 14 * 14)) → Vec (N * (256 * 14 * 14)))
    (b14 : Vec (N * (256 * 14 * 14)) → Vec (N * (512 * 7 * 7)))
    (b15 : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (b16 : Vec (N * (512 * 7 * 7)) → Vec (N * (512 * 7 * 7)))
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (h_stem : R34StemSmoothAt N 56 56 Ws bs εs γs βs x)
    (h_pool : StemPoolSmoothAt N 56 56
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) Ws bs εs γs βs x))
    (hb1 : HasVJPDiffAt b1 (opaqueA0 (r34StemB N 56 56 Ws bs εs γs βs) x))
    (hb2 : HasVJPDiffAt b2 (opaqueA1 (r34StemB N 56 56 Ws bs εs γs βs) b1 x))
    (hb3 : HasVJPDiffAt b3 (opaqueA2 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 x))
    (hb4 : HasVJPDiffAt b4 (opaqueA3 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 x))
    (hb5 : HasVJPDiffAt b5 (opaqueA4 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 x))
    (hb6 : HasVJPDiffAt b6 (opaqueA5 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
    (hb7 : HasVJPDiffAt b7 (opaqueA6 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
    (hb8 : HasVJPDiffAt b8 (opaqueA7 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
    (hb9 : HasVJPDiffAt b9 (opaqueA8 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
    (hb10 : HasVJPDiffAt b10 (opaqueA9 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
    (hb11 : HasVJPDiffAt b11 (opaqueA10 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
    (hb12 : HasVJPDiffAt b12 (opaqueA11 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
    (hb13 : HasVJPDiffAt b13 (opaqueA12 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
    (hb14 : HasVJPDiffAt b14 (opaqueA13 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
    (hb15 : HasVJPDiffAt b15 (opaqueA14 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
    (hb16 : HasVJPDiffAt b16 (opaqueA15 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34InputGradB N Ws Wd
      ((bnBatchLAHasVJP N 64 (2 * 56) (2 * 56) εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x))
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) Ws bs εs γs βs x)
      hb16.fst.backward
      hb15.fst.backward
      hb14.fst.backward
      hb13.fst.backward
      hb12.fst.backward
      hb11.fst.backward
      hb10.fst.backward
      hb9.fst.backward
      hb8.fst.backward
      hb7.fst.backward
      hb6.fst.backward
      hb5.fst.backward
      hb4.fst.backward
      hb3.fst.backward
      hb2.fst.backward
      hb1.fst.backward
      (fun i => StableHLO.bnBatchLA N 64 (2 * 56) (2 * 56) εs γs βs
        (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0)
      dy i
      = ∑ j : Fin (N * nCls),
          pdiv (r34HeadB N 7 7 Wd bd ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ r34StemB N 56 56 Ws bs εs γs βs) x i j * dy j := by
  exact HasVJPAt.correct_of_backward_eq _ (r34InputGradB_eq_r34B_full_vjp N Ws bs εs hεs γs βs Wd bd
    b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x h_stem h_pool hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16) dy i


/-- **THE SHAPE CHECK — the eighteen slots the tie is about ARE the committed forward.**
    `resnet34ForwardBFull`, regrouped into exactly the eighteen arguments `r34BFullHasVJPAt`
    takes: the stem (7×7/s2 conv-BN-relu and He et al.'s 3×3/s2 pool), the [3,4,6,3] ladder as
    sixteen basic blocks, and the GAP+dense head.

    The tie keeps its blocks OPAQUE — they enter as the VJP witnesses, so its subject is a chain of
    VARIABLES and nothing in it says which net they are. This theorem puts the stem pool on both
    sides of one statement the kernel checks. -/
theorem resnet34ForwardBFull_eq_slots (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    resnet34ForwardBFull N w x
      = (r34HeadB N 7 7 w.Wd w.bd
          ∘ r34IdB N 7 7 w.e1
          ∘ r34IdB N 7 7 w.e0
          ∘ r34DownB N 7 7 w.d4
          ∘ r34IdB N 14 14 w.c4
          ∘ r34IdB N 14 14 w.c3
          ∘ r34IdB N 14 14 w.c2
          ∘ r34IdB N 14 14 w.c1
          ∘ r34IdB N 14 14 w.c0
          ∘ r34DownB N 14 14 w.d3
          ∘ r34IdB N 28 28 w.b2
          ∘ r34IdB N 28 28 w.b1
          ∘ r34IdB N 28 28 w.b0
          ∘ r34DownB N 28 28 w.d2
          ∘ r34IdB N 56 56 w.a2
          ∘ r34IdB N 56 56 w.a1
          ∘ r34IdB N 56 56 w.a0
          ∘ r34StemB N 56 56 w.sW w.sb w.sε w.sγ w.sβ) x := by
  -- `resnet34ForwardBFull_eq_chain` for the depth-16 half, then unfold the named prefixes. (A 2×2
  -- pool's backward against this 3×3/s2 forward once went unnoticed for a month because the tie's
  -- subject is variables; this shape check is what catches that.)
  rw [resnet34ForwardBFull_eq_chain N w x]
  simp only [r34Pre16, r34Pre15, r34Pre14, r34Pre13, r34Pre12, r34Pre11, r34Pre10, r34Pre9, r34Pre8, r34Pre7, r34Pre6, r34Pre5, r34Pre4, r34Pre3, r34Pre2, r34Pre1, r34Pre0]

end Proofs
