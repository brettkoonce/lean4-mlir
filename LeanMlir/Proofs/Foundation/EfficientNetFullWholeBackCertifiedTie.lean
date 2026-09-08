import LeanMlir.Proofs.Architectures.EfficientNetFullB0
import LeanMlir.Proofs.Foundation.OpaquePrefix
import LeanMlir.Proofs.Architectures.ConvNeXtBackCertifiedTie
import LeanMlir.Proofs.Foundation.EfficientNetWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.EfficientNetBackChains

/-! # ⭐⭐ `efficientnetInputGradB_full` IS the certified whole-net PAPER EfficientNet-B0 gradient

`EfficientNetWholeBackCertifiedTie.lean` closed this for the three-block representative. This
file closes it for the net `efficientnetForwardB_full` actually is: **all sixteen MBConv
blocks**, the T6 row of `planning/proofs_tier_to_paper_nets.md` §3.3(c) — and closes it one
step further than the representative's, against the concrete `efficientnetForwardB_full_has_vjp`
and, through `efficientnetForwardB_full_has_vjp_correct`, against the Jacobian of the committed
nested-application forward itself.

Nothing here is new mathematics. The two endpoint stage ties (`stemBBack_eq_vjp_backward`,
`headFwdBBack_eq_vjp_backward`) are the representative's, reused verbatim at the paper widths
(head 320→1280 at 7×7); the sixteen blocks stay **opaque** in the tie, entering as `HasVJP`
witnesses whose `.backward` is what the reverse chain's block slots are pinned to, so the
composition is checked between variables. As on B0's representative and unlike MobileNetV2's,
there is no smooth point: swish and the SE sigmoid are differentiable everywhere.

## The three pieces

1. `efficientnetB_full_has_vjp` — the generic eighteen-stage apex `head ∘ b16 ∘ … ∘ b1 ∘ stem`,
   seventeen `vjp_comp`s and nothing else, and `opaqueA0 … A16`, one prefix `def` per running
   activation (the MobileNetV2 paper file's answer to the quadratic writing of nested
   applications; plain `def`s so the closing `rfl` sees through them).
2. `efficientnetInputGradB_full_eq_efficientnetB_full_vjp` — the tie with the stem and head
   concrete and the sixteen blocks opaque: `unfold`, two `rw`s, `rfl`.
3. `efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp` — the tie instantiated at the
   sixteen concrete blocks (`mbNoExpW`, `mbStridedW`, `mbResidW`, `mbExpW` at `B0Weights`'s
   widths) and carried to `efficientnetForwardB_full_has_vjp` by `HasVJP.backward_unique`: two
   witnesses for one map have one backward, so the tactic-built whole-net witness never has to
   be unfolded. ⭐ That is what the representative's file could not do — it stopped at a
   `▸`-transported `_committed` witness that the kernel could not reduce through — and the
   difference is not depth but `backward_unique`, which was proved for ConvNeXt-T's tie.
   `efficientnetInputGradB_full_correct` then reads the result through
   `efficientnetForwardB_full_has_vjp_correct`, whose proof IS the shape check
   `efficientnetForwardB_full_eq_chain`: the hand-written chain is the `pdiv`-contracted
   Jacobian of `efficientnetForwardB_full` at every input, every cotangent and every pixel.

⭐ **General `N`.** `bnBatchLA_has_vjp` exists at every batch size, so the tie does; this is
the certified chain's property, not the float chain's (`EfficientNetBackFloatBudget.lean`'s
number is at `N = 1` for a reason that does not apply here).

⛔ **No backward number at sixteen blocks.** `b0_full_back_chain`'s certified window at the
shipped leaves is `9.112·10²⁶⁴⁸`; the float file this imports says why it is not written down.
-/

namespace Proofs

-- The opaque running activations `opaqueA0 … opaqueA16` are `Foundation/OpaquePrefix.lean`'s.

-- ════════════════════════════════════════════════════════════════
-- § The generic eighteen-stage apex
-- ════════════════════════════════════════════════════════════════

/-- **Whole-network paper EfficientNet-B0 VJP.** The VJP of the eighteen-stage chain
    `head ∘ b16 ∘ … ∘ b1 ∘ stem` — seventeen `vjp_comp`s, dimension-generic and parametric in
    every component, `HasVJP` everywhere (no smooth point: every activation on this path is
    differentiable on all of `ℝ`). The sixteen-block peer of `efficientnetB_has_vjp`. -/
noncomputable def efficientnetB_full_has_vjp {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 : Nat}
    (stem : Vec s0 → Vec s1) (b1 : Vec s1 → Vec s2) (b2 : Vec s2 → Vec s3) (b3 : Vec s3 → Vec s4) (b4 : Vec s4 → Vec s5) (b5 : Vec s5 → Vec s6) (b6 : Vec s6 → Vec s7) (b7 : Vec s7 → Vec s8) (b8 : Vec s8 → Vec s9) (b9 : Vec s9 → Vec s10) (b10 : Vec s10 → Vec s11) (b11 : Vec s11 → Vec s12) (b12 : Vec s12 → Vec s13) (b13 : Vec s13 → Vec s14) (b14 : Vec s14 → Vec s15) (b15 : Vec s15 → Vec s16) (b16 : Vec s16 → Vec s17)
    (head : Vec s17 → Vec s18)
    (dstem : Differentiable ℝ stem) (db1 : Differentiable ℝ b1) (db2 : Differentiable ℝ b2) (db3 : Differentiable ℝ b3) (db4 : Differentiable ℝ b4) (db5 : Differentiable ℝ b5) (db6 : Differentiable ℝ b6) (db7 : Differentiable ℝ b7) (db8 : Differentiable ℝ b8) (db9 : Differentiable ℝ b9) (db10 : Differentiable ℝ b10) (db11 : Differentiable ℝ b11) (db12 : Differentiable ℝ b12) (db13 : Differentiable ℝ b13) (db14 : Differentiable ℝ b14) (db15 : Differentiable ℝ b15) (db16 : Differentiable ℝ b16)
    (dhead : Differentiable ℝ head)
    (hstem : HasVJP stem) (hb1 : HasVJP b1) (hb2 : HasVJP b2) (hb3 : HasVJP b3) (hb4 : HasVJP b4) (hb5 : HasVJP b5) (hb6 : HasVJP b6) (hb7 : HasVJP b7) (hb8 : HasVJP b8) (hb9 : HasVJP b9) (hb10 : HasVJP b10) (hb11 : HasVJP b11) (hb12 : HasVJP b12) (hb13 : HasVJP b13) (hb14 : HasVJP b14) (hb15 : HasVJP b15) (hb16 : HasVJP b16)
    (hhead : HasVJP head) :
    HasVJP (head ∘ b16 ∘ b15 ∘ b14 ∘ b13 ∘ b12 ∘ b11 ∘ b10 ∘ b9 ∘ b8 ∘ b7 ∘ b6 ∘ b5 ∘ b4 ∘ b3 ∘ b2 ∘ b1 ∘ stem) :=
  let v1 := vjp_comp _ _ dstem db1 hstem hb1
  let f1 := db1.comp dstem
  let v2 := vjp_comp _ _ f1 db2 v1 hb2
  let f2 := db2.comp f1
  let v3 := vjp_comp _ _ f2 db3 v2 hb3
  let f3 := db3.comp f2
  let v4 := vjp_comp _ _ f3 db4 v3 hb4
  let f4 := db4.comp f3
  let v5 := vjp_comp _ _ f4 db5 v4 hb5
  let f5 := db5.comp f4
  let v6 := vjp_comp _ _ f5 db6 v5 hb6
  let f6 := db6.comp f5
  let v7 := vjp_comp _ _ f6 db7 v6 hb7
  let f7 := db7.comp f6
  let v8 := vjp_comp _ _ f7 db8 v7 hb8
  let f8 := db8.comp f7
  let v9 := vjp_comp _ _ f8 db9 v8 hb9
  let f9 := db9.comp f8
  let v10 := vjp_comp _ _ f9 db10 v9 hb10
  let f10 := db10.comp f9
  let v11 := vjp_comp _ _ f10 db11 v10 hb11
  let f11 := db11.comp f10
  let v12 := vjp_comp _ _ f11 db12 v11 hb12
  let f12 := db12.comp f11
  let v13 := vjp_comp _ _ f12 db13 v12 hb13
  let f13 := db13.comp f12
  let v14 := vjp_comp _ _ f13 db14 v13 hb14
  let f14 := db14.comp f13
  let v15 := vjp_comp _ _ f14 db15 v14 hb15
  let f15 := db15.comp f14
  let v16 := vjp_comp _ _ f15 db16 v15 hb16
  let f16 := db16.comp f15
  vjp_comp _ _ f16 dhead v16 hhead

-- ════════════════════════════════════════════════════════════════
-- § The tie — stem and head concrete, the sixteen blocks opaque
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 800000 in
set_option maxHeartbeats 4000000 in
/-- ⭐⭐ **THE TIE — `efficientnetInputGradB_full` IS the certified whole-net paper-B0 gradient.**
    The committed backward chain, with its stem/head BatchNorm and swish slots filled by the
    certified per-op backwards and its sixteen MBConv blocks left opaque, equals the backward
    of `efficientnetB_full_has_vjp` at those eighteen stages. `unfold`, two `rw`s, `rfl`. -/
theorem efficientnetInputGradB_full_eq_efficientnetB_full_vjp
    (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wh : Kernel4 1280 320 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wfc : Mat 1280 nCls) (bfc : Vec nCls)
    (b1 : Vec (N * (32 * 112 * 112)) → Vec (N * (16 * 112 * 112)))
    (b2 : Vec (N * (16 * 112 * 112)) → Vec (N * (24 * 56 * 56)))
    (b3 : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4 : Vec (N * (24 * 56 * 56)) → Vec (N * (40 * 28 * 28)))
    (b5 : Vec (N * (40 * 28 * 28)) → Vec (N * (40 * 28 * 28)))
    (b6 : Vec (N * (40 * 28 * 28)) → Vec (N * (80 * 14 * 14)))
    (b7 : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b8 : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b9 : Vec (N * (80 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b10 : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b11 : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b12 : Vec (N * (112 * 14 * 14)) → Vec (N * (192 * 7 * 7)))
    (b13 : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b14 : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b15 : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b16 : Vec (N * (192 * 7 * 7)) → Vec (N * (320 * 7 * 7)))
    (db1 : Differentiable ℝ b1) (db2 : Differentiable ℝ b2) (db3 : Differentiable ℝ b3) (db4 : Differentiable ℝ b4) (db5 : Differentiable ℝ b5) (db6 : Differentiable ℝ b6) (db7 : Differentiable ℝ b7) (db8 : Differentiable ℝ b8) (db9 : Differentiable ℝ b9) (db10 : Differentiable ℝ b10) (db11 : Differentiable ℝ b11) (db12 : Differentiable ℝ b12) (db13 : Differentiable ℝ b13) (db14 : Differentiable ℝ b14) (db15 : Differentiable ℝ b15) (db16 : Differentiable ℝ b16)
    (hb1 : HasVJP b1) (hb2 : HasVJP b2) (hb3 : HasVJP b3) (hb4 : HasVJP b4) (hb5 : HasVJP b5) (hb6 : HasVJP b6) (hb7 : HasVJP b7) (hb8 : HasVJP b8) (hb9 : HasVJP b9) (hb10 : HasVJP b10) (hb11 : HasVJP b11) (hb12 : HasVJP b12) (hb13 : HasVJP b13) (hb14 : HasVJP b14) (hb15 : HasVJP b15) (hb16 : HasVJP b16)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetInputGradB_full N Ws Wh Wfc
      ((bnBatchLA_has_vjp N 32 112 112 εs hεs γs βs).backward
        (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x))
      ((swish_has_vjp (N * (32 * 112 * 112))).backward
        (StableHLO.bnBatchLA N 32 112 112 εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x)))
      ((bnBatchLA_has_vjp N 1280 7 7 εh hεh γh βh).backward
        (StableHLO.batchMap N (flatConv Wh bh) (opaqueA16 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)))
      ((swish_has_vjp (N * (1280 * 7 * 7))).backward
        (StableHLO.bnBatchLA N 1280 7 7 εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) (opaqueA16 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x))))
      (hb1.backward (opaqueA0 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) x))
      (hb2.backward (opaqueA1 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 x))
      (hb3.backward (opaqueA2 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 x))
      (hb4.backward (opaqueA3 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 x))
      (hb5.backward (opaqueA4 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 x))
      (hb6.backward (opaqueA5 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
      (hb7.backward (opaqueA6 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
      (hb8.backward (opaqueA7 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
      (hb9.backward (opaqueA8 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
      (hb10.backward (opaqueA9 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
      (hb11.backward (opaqueA10 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
      (hb12.backward (opaqueA11 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
      (hb13.backward (opaqueA12 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
      (hb14.backward (opaqueA13 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 x))
      (hb15.backward (opaqueA14 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 x))
      (hb16.backward (opaqueA15 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 x))
      = (efficientnetB_full_has_vjp
          (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16
          (headFwdB N (h := 7) (w := 7) Wh bh εh γh βh Wfc bfc)
          (stemB_differentiable N (h := 112) (w := 112) Ws bs εs hεs γs βs)
          db1 db2 db3 db4 db5 db6 db7 db8 db9 db10 db11 db12 db13 db14 db15 db16
          (headFwdB_differentiable N (h := 7) (w := 7) Wh bh εh hεh γh βh Wfc bfc)
          (stemB_has_vjp N (h := 112) (w := 112) Ws bs εs hεs γs βs)
          hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
          (headFwdB_has_vjp N (h := 7) (w := 7) Wh bh εh hεh γh βh Wfc bfc)).backward x := by
  unfold efficientnetInputGradB_full
  rw [stemBBack_eq_vjp_backward (N := N) (h := 112) (w := 112) (by decide) (by decide)
        Ws bs εs hεs γs βs x,
      headFwdBBack_eq_vjp_backward (N := N) (h := 7) (w := 7) Wh bh εh hεh γh βh Wfc bfc
        (opaqueA16 (stemB N (h := 112) (w := 112) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 x)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The tie at the CONCRETE blocks, against `efficientnetForwardB_full_has_vjp`
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 800000 in
set_option maxHeartbeats 4000000 in
/-- ⭐⭐ **The chain, at the sixteen concrete MBConv blocks, IS `efficientnetForwardB_full_has_vjp`'s
    backward.** The tie above instantiated at `mbNoExpW`/`mbStridedW`/`mbResidW`/`mbExpW` at
    `B0Weights`'s widths, then `HasVJP.backward_unique` between the generic apex and the
    tactic-built whole-net witness — both are VJPs of the same eighteen-stage composition, so
    they have the same backward, and neither is unfolded. -/
theorem efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp (N : Nat) (w : B0Weights)
    (hsε : 0 < w.sε)
    (hb1d : 0 < w.b1.dε) (hb1p : 0 < w.b1.pε)
    (hb2e : 0 < w.b2.eε) (hb2d : 0 < w.b2.dε) (hb2p : 0 < w.b2.pε)
    (hb3e : 0 < w.b3.eε) (hb3d : 0 < w.b3.dε) (hb3p : 0 < w.b3.pε)
    (hb4e : 0 < w.b4.eε) (hb4d : 0 < w.b4.dε) (hb4p : 0 < w.b4.pε)
    (hb5e : 0 < w.b5.eε) (hb5d : 0 < w.b5.dε) (hb5p : 0 < w.b5.pε)
    (hb6e : 0 < w.b6.eε) (hb6d : 0 < w.b6.dε) (hb6p : 0 < w.b6.pε)
    (hb7e : 0 < w.b7.eε) (hb7d : 0 < w.b7.dε) (hb7p : 0 < w.b7.pε)
    (hb8e : 0 < w.b8.eε) (hb8d : 0 < w.b8.dε) (hb8p : 0 < w.b8.pε)
    (hb9e : 0 < w.b9.eε) (hb9d : 0 < w.b9.dε) (hb9p : 0 < w.b9.pε)
    (hb10e : 0 < w.b10.eε) (hb10d : 0 < w.b10.dε) (hb10p : 0 < w.b10.pε)
    (hb11e : 0 < w.b11.eε) (hb11d : 0 < w.b11.dε) (hb11p : 0 < w.b11.pε)
    (hb12e : 0 < w.b12.eε) (hb12d : 0 < w.b12.dε) (hb12p : 0 < w.b12.pε)
    (hb13e : 0 < w.b13.eε) (hb13d : 0 < w.b13.dε) (hb13p : 0 < w.b13.pε)
    (hb14e : 0 < w.b14.eε) (hb14d : 0 < w.b14.dε) (hb14p : 0 < w.b14.pε)
    (hb15e : 0 < w.b15.eε) (hb15d : 0 < w.b15.dε) (hb15p : 0 < w.b15.pε)
    (hb16e : 0 < w.b16.eε) (hb16d : 0 < w.b16.dε) (hb16p : 0 < w.b16.pε)
    (hhε : 0 < w.hε)
    (x : Vec (N * (3 * 224 * 224))) :
    efficientnetInputGradB_full N w.sW w.hW w.fcW
      ((bnBatchLA_has_vjp N 32 112 112 w.sε hsε w.sγ w.sβ).backward
        (StableHLO.batchMap N (flatConvStride2Xla w.sW w.sb) x))
      ((swish_has_vjp (N * (32 * 112 * 112))).backward
        (StableHLO.bnBatchLA N 32 112 112 w.sε w.sγ w.sβ
          (StableHLO.batchMap N (flatConvStride2Xla w.sW w.sb) x)))
      ((bnBatchLA_has_vjp N 1280 7 7 w.hε hhε w.hγ w.hβ).backward
        (StableHLO.batchMap N (flatConv w.hW w.hb) (opaqueA16 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) (mbExpW N 7 7 w.b16) x)))
      ((swish_has_vjp (N * (1280 * 7 * 7))).backward
        (StableHLO.bnBatchLA N 1280 7 7 w.hε w.hγ w.hβ
          (StableHLO.batchMap N (flatConv w.hW w.hb) (opaqueA16 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) (mbExpW N 7 7 w.b16) x))))
      ((mbNoExpW_has_vjp N 112 112 w.b1 hb1d hb1p).backward (opaqueA0 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) x))
      ((mbStridedW_has_vjp N 56 56 w.b2 hb2e hb2d hb2p).backward (opaqueA1 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) x))
      ((mbResidW_has_vjp N 56 56 w.b3 hb3e hb3d hb3p).backward (opaqueA2 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) x))
      ((mbStridedW_has_vjp N 28 28 w.b4 hb4e hb4d hb4p).backward (opaqueA3 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) x))
      ((mbResidW_has_vjp N 28 28 w.b5 hb5e hb5d hb5p).backward (opaqueA4 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) x))
      ((mbStridedW_has_vjp N 14 14 w.b6 hb6e hb6d hb6p).backward (opaqueA5 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) x))
      ((mbResidW_has_vjp N 14 14 w.b7 hb7e hb7d hb7p).backward (opaqueA6 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) x))
      ((mbResidW_has_vjp N 14 14 w.b8 hb8e hb8d hb8p).backward (opaqueA7 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) x))
      ((mbExpW_has_vjp N 14 14 w.b9 hb9e hb9d hb9p).backward (opaqueA8 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) x))
      ((mbResidW_has_vjp N 14 14 w.b10 hb10e hb10d hb10p).backward (opaqueA9 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) x))
      ((mbResidW_has_vjp N 14 14 w.b11 hb11e hb11d hb11p).backward (opaqueA10 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) x))
      ((mbStridedW_has_vjp N 7 7 w.b12 hb12e hb12d hb12p).backward (opaqueA11 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) x))
      ((mbResidW_has_vjp N 7 7 w.b13 hb13e hb13d hb13p).backward (opaqueA12 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) x))
      ((mbResidW_has_vjp N 7 7 w.b14 hb14e hb14d hb14p).backward (opaqueA13 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) x))
      ((mbResidW_has_vjp N 7 7 w.b15 hb15e hb15d hb15p).backward (opaqueA14 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) x))
      ((mbExpW_has_vjp N 7 7 w.b16 hb16e hb16d hb16p).backward (opaqueA15 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) x))
      = (efficientnetForwardB_full_has_vjp N w hsε hb1d hb1p hb2e hb2d hb2p hb3e hb3d hb3p hb4e hb4d hb4p hb5e hb5d hb5p hb6e hb6d hb6p hb7e hb7d hb7p hb8e hb8d hb8p hb9e hb9d hb9p hb10e hb10d hb10p hb11e hb11d hb11p hb12e hb12d hb12p hb13e hb13d hb13p hb14e hb14d hb14p hb15e hb15d hb15p hb16e hb16d hb16p hhε).backward x :=
  (efficientnetInputGradB_full_eq_efficientnetB_full_vjp N w.sW w.sb w.sε hsε w.sγ w.sβ
      w.hW w.hb w.hε hhε w.hγ w.hβ w.fcW w.fcb
      (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) (mbExpW N 7 7 w.b16)
      (mbNoExpW_differentiable N 112 112 w.b1 hb1d hb1p) (mbStridedW_differentiable N 56 56 w.b2 hb2e hb2d hb2p) (mbResidW_differentiable N 56 56 w.b3 hb3e hb3d hb3p) (mbStridedW_differentiable N 28 28 w.b4 hb4e hb4d hb4p) (mbResidW_differentiable N 28 28 w.b5 hb5e hb5d hb5p) (mbStridedW_differentiable N 14 14 w.b6 hb6e hb6d hb6p) (mbResidW_differentiable N 14 14 w.b7 hb7e hb7d hb7p) (mbResidW_differentiable N 14 14 w.b8 hb8e hb8d hb8p) (mbExpW_differentiable N 14 14 w.b9 hb9e hb9d hb9p) (mbResidW_differentiable N 14 14 w.b10 hb10e hb10d hb10p) (mbResidW_differentiable N 14 14 w.b11 hb11e hb11d hb11p) (mbStridedW_differentiable N 7 7 w.b12 hb12e hb12d hb12p) (mbResidW_differentiable N 7 7 w.b13 hb13e hb13d hb13p) (mbResidW_differentiable N 7 7 w.b14 hb14e hb14d hb14p) (mbResidW_differentiable N 7 7 w.b15 hb15e hb15d hb15p) (mbExpW_differentiable N 7 7 w.b16 hb16e hb16d hb16p)
      (mbNoExpW_has_vjp N 112 112 w.b1 hb1d hb1p) (mbStridedW_has_vjp N 56 56 w.b2 hb2e hb2d hb2p) (mbResidW_has_vjp N 56 56 w.b3 hb3e hb3d hb3p) (mbStridedW_has_vjp N 28 28 w.b4 hb4e hb4d hb4p) (mbResidW_has_vjp N 28 28 w.b5 hb5e hb5d hb5p) (mbStridedW_has_vjp N 14 14 w.b6 hb6e hb6d hb6p) (mbResidW_has_vjp N 14 14 w.b7 hb7e hb7d hb7p) (mbResidW_has_vjp N 14 14 w.b8 hb8e hb8d hb8p) (mbExpW_has_vjp N 14 14 w.b9 hb9e hb9d hb9p) (mbResidW_has_vjp N 14 14 w.b10 hb10e hb10d hb10p) (mbResidW_has_vjp N 14 14 w.b11 hb11e hb11d hb11p) (mbStridedW_has_vjp N 7 7 w.b12 hb12e hb12d hb12p) (mbResidW_has_vjp N 7 7 w.b13 hb13e hb13d hb13p) (mbResidW_has_vjp N 7 7 w.b14 hb14e hb14d hb14p) (mbResidW_has_vjp N 7 7 w.b15 hb15e hb15d hb15p) (mbExpW_has_vjp N 7 7 w.b16 hb16e hb16d hb16p) x).trans
    (funext fun dy => HasVJP.backward_unique _ _ x dy)

/-- ⭐⭐ **The hand-written sixteen-block chain IS the Jacobian-transpose of the committed
    `efficientnetForwardB_full`** — at every input, every loss cotangent and every input pixel.
    The tie above read through `efficientnetForwardB_full_has_vjp_correct`, whose proof is the
    shape check `efficientnetForwardB_full_eq_chain`: the nested-application forward the render
    denotes and the `∘`-chain the VJP was assembled on are one function. -/
theorem efficientnetInputGradB_full_correct (N : Nat) (w : B0Weights)
    (hsε : 0 < w.sε)
    (hb1d : 0 < w.b1.dε) (hb1p : 0 < w.b1.pε)
    (hb2e : 0 < w.b2.eε) (hb2d : 0 < w.b2.dε) (hb2p : 0 < w.b2.pε)
    (hb3e : 0 < w.b3.eε) (hb3d : 0 < w.b3.dε) (hb3p : 0 < w.b3.pε)
    (hb4e : 0 < w.b4.eε) (hb4d : 0 < w.b4.dε) (hb4p : 0 < w.b4.pε)
    (hb5e : 0 < w.b5.eε) (hb5d : 0 < w.b5.dε) (hb5p : 0 < w.b5.pε)
    (hb6e : 0 < w.b6.eε) (hb6d : 0 < w.b6.dε) (hb6p : 0 < w.b6.pε)
    (hb7e : 0 < w.b7.eε) (hb7d : 0 < w.b7.dε) (hb7p : 0 < w.b7.pε)
    (hb8e : 0 < w.b8.eε) (hb8d : 0 < w.b8.dε) (hb8p : 0 < w.b8.pε)
    (hb9e : 0 < w.b9.eε) (hb9d : 0 < w.b9.dε) (hb9p : 0 < w.b9.pε)
    (hb10e : 0 < w.b10.eε) (hb10d : 0 < w.b10.dε) (hb10p : 0 < w.b10.pε)
    (hb11e : 0 < w.b11.eε) (hb11d : 0 < w.b11.dε) (hb11p : 0 < w.b11.pε)
    (hb12e : 0 < w.b12.eε) (hb12d : 0 < w.b12.dε) (hb12p : 0 < w.b12.pε)
    (hb13e : 0 < w.b13.eε) (hb13d : 0 < w.b13.dε) (hb13p : 0 < w.b13.pε)
    (hb14e : 0 < w.b14.eε) (hb14d : 0 < w.b14.dε) (hb14p : 0 < w.b14.pε)
    (hb15e : 0 < w.b15.eε) (hb15d : 0 < w.b15.dε) (hb15p : 0 < w.b15.pε)
    (hb16e : 0 < w.b16.eε) (hb16d : 0 < w.b16.dε) (hb16p : 0 < w.b16.pε)
    (hhε : 0 < w.hε)
    (x : Vec (N * (3 * 224 * 224))) (dy : Vec (N * 10)) (i : Fin (N * (3 * 224 * 224))) :
    efficientnetInputGradB_full N w.sW w.hW w.fcW
      ((bnBatchLA_has_vjp N 32 112 112 w.sε hsε w.sγ w.sβ).backward
        (StableHLO.batchMap N (flatConvStride2Xla w.sW w.sb) x))
      ((swish_has_vjp (N * (32 * 112 * 112))).backward
        (StableHLO.bnBatchLA N 32 112 112 w.sε w.sγ w.sβ
          (StableHLO.batchMap N (flatConvStride2Xla w.sW w.sb) x)))
      ((bnBatchLA_has_vjp N 1280 7 7 w.hε hhε w.hγ w.hβ).backward
        (StableHLO.batchMap N (flatConv w.hW w.hb) (opaqueA16 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) (mbExpW N 7 7 w.b16) x)))
      ((swish_has_vjp (N * (1280 * 7 * 7))).backward
        (StableHLO.bnBatchLA N 1280 7 7 w.hε w.hγ w.hβ
          (StableHLO.batchMap N (flatConv w.hW w.hb) (opaqueA16 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) (mbExpW N 7 7 w.b16) x))))
      ((mbNoExpW_has_vjp N 112 112 w.b1 hb1d hb1p).backward (opaqueA0 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) x))
      ((mbStridedW_has_vjp N 56 56 w.b2 hb2e hb2d hb2p).backward (opaqueA1 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) x))
      ((mbResidW_has_vjp N 56 56 w.b3 hb3e hb3d hb3p).backward (opaqueA2 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) x))
      ((mbStridedW_has_vjp N 28 28 w.b4 hb4e hb4d hb4p).backward (opaqueA3 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) x))
      ((mbResidW_has_vjp N 28 28 w.b5 hb5e hb5d hb5p).backward (opaqueA4 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) x))
      ((mbStridedW_has_vjp N 14 14 w.b6 hb6e hb6d hb6p).backward (opaqueA5 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) x))
      ((mbResidW_has_vjp N 14 14 w.b7 hb7e hb7d hb7p).backward (opaqueA6 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) x))
      ((mbResidW_has_vjp N 14 14 w.b8 hb8e hb8d hb8p).backward (opaqueA7 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) x))
      ((mbExpW_has_vjp N 14 14 w.b9 hb9e hb9d hb9p).backward (opaqueA8 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) x))
      ((mbResidW_has_vjp N 14 14 w.b10 hb10e hb10d hb10p).backward (opaqueA9 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) x))
      ((mbResidW_has_vjp N 14 14 w.b11 hb11e hb11d hb11p).backward (opaqueA10 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) x))
      ((mbStridedW_has_vjp N 7 7 w.b12 hb12e hb12d hb12p).backward (opaqueA11 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) x))
      ((mbResidW_has_vjp N 7 7 w.b13 hb13e hb13d hb13p).backward (opaqueA12 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) x))
      ((mbResidW_has_vjp N 7 7 w.b14 hb14e hb14d hb14p).backward (opaqueA13 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) x))
      ((mbResidW_has_vjp N 7 7 w.b15 hb15e hb15d hb15p).backward (opaqueA14 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) x))
      ((mbExpW_has_vjp N 7 7 w.b16 hb16e hb16d hb16p).backward (opaqueA15 (stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) (mbNoExpW N 112 112 w.b1) (mbStridedW N 56 56 w.b2) (mbResidW N 56 56 w.b3) (mbStridedW N 28 28 w.b4) (mbResidW N 28 28 w.b5) (mbStridedW N 14 14 w.b6) (mbResidW N 14 14 w.b7) (mbResidW N 14 14 w.b8) (mbExpW N 14 14 w.b9) (mbResidW N 14 14 w.b10) (mbResidW N 14 14 w.b11) (mbStridedW N 7 7 w.b12) (mbResidW N 7 7 w.b13) (mbResidW N 7 7 w.b14) (mbResidW N 7 7 w.b15) x))
      dy i
      = ∑ j : Fin (N * 10), pdiv (efficientnetForwardB_full N w) x i j * dy j := by
  rw [congrFun (efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp N w
    hsε hb1d hb1p hb2e hb2d hb2p hb3e hb3d hb3p hb4e hb4d hb4p hb5e hb5d hb5p hb6e hb6d hb6p hb7e hb7d hb7p hb8e hb8d hb8p hb9e hb9d hb9p hb10e hb10d hb10p hb11e hb11d hb11p hb12e hb12d hb12p hb13e hb13d hb13p hb14e hb14d hb14p hb15e hb15d hb15p hb16e hb16d hb16p hhε x) dy]
  exact efficientnetForwardB_full_has_vjp_correct N w hsε hb1d hb1p hb2e hb2d hb2p hb3e hb3d hb3p hb4e hb4d hb4p hb5e hb5d hb5p hb6e hb6d hb6p hb7e hb7d hb7p hb8e hb8d hb8p hb9e hb9d hb9p hb10e hb10d hb10p hb11e hb11d hb11p hb12e hb12d hb12p hb13e hb13d hb13p hb14e hb14d hb14p hb15e hb15d hb15p hb16e hb16d hb16p hhε x dy i

end Proofs
