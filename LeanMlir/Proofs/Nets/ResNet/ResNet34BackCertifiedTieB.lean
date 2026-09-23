import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTie
import LeanMlir.Proofs.Nets.ResNet.ResNetBackChains

/-! # ⭐⭐ `r34InputGradB` IS the certified whole-net ResNet-34 gradient AT BATCH BATCH-NORM

The per-example net's tie (retired 2026-09-19 with `ResNet34RenderPC.lean`; `ResNet34BackCertifiedTie.lean`
keeps its leaf ties) closed this for the forward the retired `ResNet34Render.lean` emitted. This file closes it
for the net the shipped trainers run: `resnet34ForwardB_full`, the [3,4,6,3] ladder at
**`bnBatchLA`**, at a variable batch `N`. It is tier **T6** of
`planning/archive/proofs_tier_to_paper_nets.md` §4.2, the last real statement in that section's port
(T4/T5 there are float budgets, and `planning/archive/float_budget_numbers.md` closed that thread).

Nothing here is new mathematics. Two endpoint stage ties (stem and head) and one pool tie, then
the sixteen basic blocks stay **opaque** — they enter as the `_at` VJP witnesses 4.1d's apex
already takes, and the reverse chain's block slots are pinned to their `.backward` — so the
composition is checked between variables. The chain is peeled by `rw` with a lemma proved over
variable stages (piece 2), so the whole module checks in a few seconds with no budget raised.

## The four pieces

1. `cbReluStridedBBack_eq_vjp_backward` / `r34HeadBBack_eq_vjp_backward` — the concrete
   endpoints; the stem is the first and the pool needs no lemma.
   ⭐ The pool endpoint is **`rfl`**, and that is the payoff of two earlier decisions: 4.1c built
   `batchMap_has_vjp_at` field by field rather than transporting it with `▸`, and
   `maxPool3s2Flat_has_vjp_at_vec` did the same one tier down, so `batchMapAux` of the leaf and
   the lift's `.backward` are the same term. It also closes the one seam §4.2a left open — that
   file could thread the pool backward only as the emitted `den`.
2. `r34B_full_has_vjp_at` — the generic eighteen-stage apex `head ∘ b16 ∘ … ∘ b1 ∘ stem`,
   seventeen nested `vjp_comp_diff_at`s and nothing else, over `opaqueA0 … A16`, one prefix `def`
   per running activation — and `r34B_full_has_vjp_at_backward`, its backward peeled into the
   eighteen stage backwards, `rfl` at variable stages. ⛔ Not a closing `rfl` in the tie itself:
   written as seventeen `let`s and closed that way, the tie cost ~45 s and 8 GB under
   `maxRecDepth 800000`.
3. `r34InputGradB_eq_r34B_full_vjp` and `r34InputGradB_correct` — the tie, and its reading as
   `∑ pdiv … * dy`: the chain IS the Jacobian-transpose of the eighteen-stage composition.
4. `resnet34ForwardB_full_eq_slots` — the shape check: those eighteen stages ARE
   `resnet34ForwardB_full`, the forward `resnet34FwdGraphB_full_faithful` (4.1b) says the typed
   graph denotes. Without it the tie would be a statement about variables.

## ⛔ Two walls, both measured, both worth not re-paying

**A `rfl` straight at 4.1d's `resnet34ForwardB_full_has_vjp_at` does not terminate** — five
minutes to `(deterministic) timeout at isDefEq` at four million heartbeats. That is §5's
elaboration trap and the reason the generic apex exists.

⛔ **And so does instantiating the generic tie at the sixteen concrete blocks** — a *kernel*
deterministic timeout at six minutes, with `HasVJPAt.backward_unique` or without it. So this
file stops where MobileNetV2's per-example T6 stops (opaque blocks plus a shape check) rather
than where B0's goes (concrete blocks, then `backward_unique`). ⭐ **The difference is not depth
and not the net: it is the KINK.** B0's generic tie takes GLOBAL `HasVJP` witnesses, which carry
no point, so instantiating them is free. r34's are `HasVJPAt` at `opaqueA{k-1} … x`, and the
witnesses a caller has are at `r34Pre{k-1} N w x` — sixteen defeq checks between two
sixteen-deep nested applications spelled through different definition chains.

⚠ It stays a SMOOTH-POINT statement, and r34 carries the heaviest hypothesis budget of the five
nets: two relu clauses per block (the body's mid-relu and the post-residual OUTER relu), the
stem's relu, and the stem pool's per-example no-tie condition. That is 4.1d's bundle list,
reused verbatim — this file adds no hypothesis of its own.

⛔ **What this does NOT reach.** One device: the data-parallel step, collectives included, is
`ResNet34SyncStepTieB.lean`'s. And it is about the INPUT gradient; the parameter gradients are
`ResNet34StepTieB.lean`'s tie (§4.2a).
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The concrete endpoint ties
-- ════════════════════════════════════════════════════════════════

/-- **The conv-BN-relu STAGE tie.** The hand-written
    `batchMap (flatConvStride2Back) ∘ bnBack ∘ reluMaskBack` IS `cbReluStridedB`'s certified
    backward at a smooth point. One `rw` of the odd-kernel strided leaf tie, then `rfl` — the
    stage's VJP is `vjp_comp_at`-built so its backward is already the composition, and the conv
    leaf's backward is input-independent (a convolution is linear), so the row-wise `batchMap`
    lift matches at every saved input.

    ⚠ SYMMETRIC padding (`flatConvStride2Back`), not the XLA-`SAME` phase B0's and MobileNetV2's
    stems take. Identical types, different certificates. -/
theorem cbReluStridedBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hrelu : ∀ k, StableHLO.bnBatchLA N oc h w ε γ β
      (StableHLO.batchMap N (flatConvStride2 Ws bs) x) k ≠ 0) :
    (StableHLO.batchMap N (flatConvStride2Back (h := h) (w := w) Ws)
        ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x)
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc h w ε γ β
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x) i > 0))
      = (StableHLO.cbReluStridedB_has_vjp_at N Ws bs ε hε γ β x hrelu).backward := by
  rw [flatConvStride2Back_eq_vjp_backward hkH hkW Ws bs (fun _ => 0)]
  rfl

/-- **The HEAD CONV tie.** `batchMap (convFlatBack) ∘ bnBack ∘ reluMaskBack` IS `cbReluB`'s
    certified backward at a smooth point — the stride-1, plain-convolution peer of the stem's.

    ⭐ The kernel extent is a binder, so a 1×1 conv-BN-relu is one instance of this stage and not
    a new one: MobileNetV4's two head convs (`%h1W` 256 → 960, `%hW` 960 → 1280) tie by applying
    this twice, and its head is therefore not hypothesis-free the way ResNet-34's is. -/
theorem cbReluBBack_eq_vjp_backward {N ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (ic * h * w)))
    (hs : ∀ k, StableHLO.bnBatchLA N oc h w ε γ β
           (StableHLO.batchMap N (flatConv W b) v) k ≠ 0) :
    (StableHLO.batchMap N (convFlatBack (h := h) (w := w) W)
        ∘ (bnBatchLA_has_vjp N oc h w ε hε γ β).backward
            (StableHLO.batchMap N (flatConv W b) v)
        ∘ reluMaskBack (fun i => StableHLO.bnBatchLA N oc h w ε γ β
            (StableHLO.batchMap N (flatConv W b) v) i > 0))
      = (StableHLO.cbReluB_has_vjp_at N W b ε hε γ β v hs).backward := by
  rw [convFlatBack_eq_vjp_backward hkH hkW W b (fun _ => 0)]
  rfl

/-- **The HEAD tie.** `batchMap (gapBack) ∘ batchMap (dense Wᵀ 0)` IS `r34HeadB`'s certified
    backward. ⭐ The head takes no hypothesis at all: GAP and dense are smooth and each is
    `batchMap` of a per-example op, so `r34HeadB_has_vjp` is GLOBAL — the one place in this net
    where the certified backward comes with nothing attached. `gapBack` needs no rewrite; it is
    definitionally the global-average-pool VJP's backward. -/
theorem r34HeadBBack_eq_vjp_backward {N c nCls h w : Nat}
    (Wd : Mat c nCls) (bd : Vec nCls) (x : Vec (N * (c * h * w))) :
    (StableHLO.batchMap N (gapBack c h w)
      ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec c)))
      = (r34HeadB_has_vjp N h w Wd bd).backward x := by
  rw [dense_transpose_eq_vjp_backward Wd bd (fun _ => 0)]
  rfl

/-- The stem's backward is the pool's, then the conv-BN-relu stage's. `rfl` at variable widths;
    the tie rewrites with it rather than leaving this step to its closing `rfl`, which reaches
    `maxRecDepth` at the net's numerals. -/
theorem r34StemB_has_vjp_at_backward (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (hc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hrelu : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpool : R34PoolSmoothAt N h w
      (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x))
    (v : Vec (N * (oc * h * w))) :
    (r34StemB_has_vjp_at N h w Ws bs εs hεs γs βs hc hh hw x hrelu hpool).backward v
      = (StableHLO.cbReluStridedB_has_vjp_at N Ws bs εs hεs γs βs x hrelu).backward
          (maxPool3s2FlatBackB N oc h w
            (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) v) := rfl

-- The opaque running activations `opaqueA0 … opaqueA16` are `Foundation/OpaquePrefix.lean`'s.

-- ════════════════════════════════════════════════════════════════
-- § The apex — a straight 18-stage chain, every stage opaque
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network batched ResNet-34 VJP, every stage opaque.** `head ∘ b16 ∘ … ∘ b1 ∘ stem`,
    seventeen `vjp_comp_diff_at`s and nothing else. ⭐ ResNet-34's [3,4,6,3] ladder needs no
    `ChainData` list and no separate downsample slot: a downsample block is just a block of a
    different type, and the stem's pool lives INSIDE `stem`. Dimension-generic and parametric in
    every component, so the tie below is checked between variables.

    ⚠ Pointwise (`HasVJPAt`), as every ResNet-34 statement is: relu is kinked, and this net has
    two relu sites per block. -/
noncomputable def r34B_full_has_vjp_at {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 : Nat}
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
    (hhead : PProd (HasVJPAt head (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                   (DifferentiableAt ℝ head (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    : HasVJPAt (head ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) x :=
  (vjp_comp_diff_at (b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) head x
    (vjp_comp_diff_at (b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b16 x
      (vjp_comp_diff_at (b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b15 x
        (vjp_comp_diff_at (b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b14 x
          (vjp_comp_diff_at (b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b13 x
            (vjp_comp_diff_at (b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b12 x
              (vjp_comp_diff_at (b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b11 x
                (vjp_comp_diff_at (b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b10 x
                  (vjp_comp_diff_at (b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b9 x
                    (vjp_comp_diff_at (b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b8 x
                      (vjp_comp_diff_at (b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b7 x
                        (vjp_comp_diff_at (b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b6 x
                          (vjp_comp_diff_at (b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) b5 x
                            (vjp_comp_diff_at (b3 ∘ b2 ∘ b1 ∘ stem) b4 x
                              (vjp_comp_diff_at (b2 ∘ b1 ∘ stem) b3 x
                                (vjp_comp_diff_at (b1 ∘ stem) b2 x
                                  (vjp_comp_diff_at stem b1 x
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
theorem r34B_full_has_vjp_at_backward {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 : Nat}
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
    (hhead : PProd (HasVJPAt head (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))
                   (DifferentiableAt ℝ head (opaqueA16 stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
    (dy : Vec s18) :
    (r34B_full_has_vjp_at stem b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 head x hstem hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16 hhead).backward dy
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
-- § ⭐⭐ THE TIE — stem, pool and head concrete, the sixteen blocks opaque
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **`r34InputGradB` IS the certified whole-net batch-BN ResNet-34 gradient.** The committed
    backward chain, with its stem BatchNorm and relu-mask slots filled by the certified per-op
    backwards, its saved pool activation the stem's own, and its sixteen basic blocks left OPAQUE,
    equals the backward of `r34B_full_has_vjp_at` at those eighteen stages. `unfold`, the two
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
    (h_pool : R34PoolSmoothAt N 56 56
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) Ws bs εs γs βs x))
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 (r34StemB N 56 56 Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ b1 (opaqueA0 (r34StemB N 56 56 Ws bs εs γs βs) x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA1 (r34StemB N 56 56 Ws bs εs γs βs) b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA1 (r34StemB N 56 56 Ws bs εs γs βs) b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA2 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA2 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA3 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA3 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA4 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA4 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA5 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA5 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA6 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA6 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA7 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA7 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA8 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA8 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA9 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA9 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA10 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA10 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA11 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA11 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA12 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA12 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA13 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA13 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA14 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA14 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA15 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA15 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))) :
    r34InputGradB N Ws Wd
      ((bnBatchLA_has_vjp N 64 (2 * 56) (2 * 56) εs hεs γs βs).backward
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
      = (r34B_full_has_vjp_at (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16
          (r34HeadB N 7 7 Wd bd) x
          ⟨r34StemB_has_vjp_at N 56 56 Ws bs εs hεs γs βs
              (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool,
            r34StemB_differentiableAt N 56 56 Ws bs εs hεs γs βs
              (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
          ⟨(r34HeadB_has_vjp N 7 7 Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N 7 7 Wd bd) _⟩).backward := by
  unfold r34InputGradB
  rw [cbReluStridedBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      r34HeadBBack_eq_vjp_backward Wd bd (opaqueA16 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)]
  funext dy
  rw [r34B_full_has_vjp_at_backward, r34StemB_has_vjp_at_backward]
  repeat rw [Function.comp_apply]
  rfl


/-- ⭐⭐ **The batched chain IS the `pdiv`-contracted Jacobian of the eighteen-stage net** — at
    every batch size, every input, every loss cotangent and every input pixel. The tie above read
    through the apex's own `.correct`. `resnet34ForwardB_full_eq_slots` below is what says those
    eighteen stages are the committed forward, so the two together are the T6 statement. -/
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
    (h_pool : R34PoolSmoothAt N 56 56
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) Ws bs εs γs βs x))
    (hb1 : PProd (HasVJPAt b1 (opaqueA0 (r34StemB N 56 56 Ws bs εs γs βs) x))
                 (DifferentiableAt ℝ b1 (opaqueA0 (r34StemB N 56 56 Ws bs εs γs βs) x)))
    (hb2 : PProd (HasVJPAt b2 (opaqueA1 (r34StemB N 56 56 Ws bs εs γs βs) b1 x))
                 (DifferentiableAt ℝ b2 (opaqueA1 (r34StemB N 56 56 Ws bs εs γs βs) b1 x)))
    (hb3 : PProd (HasVJPAt b3 (opaqueA2 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 x))
                 (DifferentiableAt ℝ b3 (opaqueA2 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 x)))
    (hb4 : PProd (HasVJPAt b4 (opaqueA3 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 x))
                 (DifferentiableAt ℝ b4 (opaqueA3 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 x)))
    (hb5 : PProd (HasVJPAt b5 (opaqueA4 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 x))
                 (DifferentiableAt ℝ b5 (opaqueA4 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 x)))
    (hb6 : PProd (HasVJPAt b6 (opaqueA5 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
                 (DifferentiableAt ℝ b6 (opaqueA5 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 x)))
    (hb7 : PProd (HasVJPAt b7 (opaqueA6 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
                 (DifferentiableAt ℝ b7 (opaqueA6 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x)))
    (hb8 : PProd (HasVJPAt b8 (opaqueA7 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
                 (DifferentiableAt ℝ b8 (opaqueA7 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x)))
    (hb9 : PProd (HasVJPAt b9 (opaqueA8 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
                 (DifferentiableAt ℝ b9 (opaqueA8 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x)))
    (hb10 : PProd (HasVJPAt b10 (opaqueA9 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
                 (DifferentiableAt ℝ b10 (opaqueA9 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x)))
    (hb11 : PProd (HasVJPAt b11 (opaqueA10 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
                 (DifferentiableAt ℝ b11 (opaqueA10 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x)))
    (hb12 : PProd (HasVJPAt b12 (opaqueA11 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
                 (DifferentiableAt ℝ b12 (opaqueA11 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x)))
    (hb13 : PProd (HasVJPAt b13 (opaqueA12 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
                 (DifferentiableAt ℝ b13 (opaqueA12 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x)))
    (hb14 : PProd (HasVJPAt b14 (opaqueA13 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
                 (DifferentiableAt ℝ b14 (opaqueA13 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x)))
    (hb15 : PProd (HasVJPAt b15 (opaqueA14 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
                 (DifferentiableAt ℝ b15 (opaqueA14 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x)))
    (hb16 : PProd (HasVJPAt b16 (opaqueA15 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
                 (DifferentiableAt ℝ b16 (opaqueA15 (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x)))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34InputGradB N Ws Wd
      ((bnBatchLA_has_vjp N 64 (2 * 56) (2 * 56) εs hεs γs βs).backward
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
  rw [congrFun (r34InputGradB_eq_r34B_full_vjp N Ws bs εs hεs γs βs Wd bd
    b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x h_stem h_pool hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16) dy]
  exact (r34B_full_has_vjp_at (r34StemB N 56 56 Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16
          (r34HeadB N 7 7 Wd bd) x
          ⟨r34StemB_has_vjp_at N 56 56 Ws bs εs hεs γs βs
              (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool,
            r34StemB_differentiableAt N 56 56 Ws bs εs hεs γs βs
              (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool⟩
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
          ⟨(r34HeadB_has_vjp N 7 7 Wd bd).toHasVJPAt _,
            (r34HeadB_differentiable N 7 7 Wd bd) _⟩).correct dy i


/-- ⭐⭐ **THE SHAPE CHECK — the eighteen slots the tie is about ARE the committed forward.**
    `resnet34ForwardB_full`, regrouped into exactly the eighteen arguments `r34B_full_has_vjp_at`
    takes: the stem (7×7/s2 conv-BN-relu and He et al.'s 3×3/s2 pool), the [3,4,6,3] ladder as
    sixteen basic blocks, and the GAP+dense head.

    ⛔ **This is the theorem that would have caught ResNet-34's wrong pool.** The tie keeps its
    blocks OPAQUE — they enter as the VJP witnesses, so its subject is a chain of VARIABLES and
    nothing in it says which net they are. §3.10's drift (the 2×2 pool's backward against a
    forward that pools 3×3/s2) lived a month for exactly that reason:
    *"the same net as the tie"* was prose in a docstring. Here the pool appears on both sides of
    one statement the kernel checks.

    It goes through `resnet34ForwardB_full_eq_chain` (4.1d) for the depth-16 half and then unfolds
    the named prefixes. -/
theorem resnet34ForwardB_full_eq_slots (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    resnet34ForwardB_full N w x
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
  rw [resnet34ForwardB_full_eq_chain N w x]
  simp only [r34Pre16, r34Pre15, r34Pre14, r34Pre13, r34Pre12, r34Pre11, r34Pre10, r34Pre9, r34Pre8, r34Pre7, r34Pre6, r34Pre5, r34Pre4, r34Pre3, r34Pre2, r34Pre1, r34Pre0]

end Proofs
