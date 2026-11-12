module

public import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StepTieB
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # MobileNetV2's whole-net input VJP at the classifier-dropout site

`MobileNetV2FullBVJP` certifies the input VJP of `mobilenetv2ForwardBFull`, the forward without
dropout. The `*do*` renders train `mobilenetv2ForwardBFullDoOpt N w (some m)`: the same trunk with
`dropout m` between the head's GAP and the dense. This file states the theorem for that function
at either value of the site:

* `mobilenetv2ForwardBFullDoOptHasVJPAt` — the certified input VJP at a smooth point;
* `mobilenetv2ForwardBFullDoOptHasVJPAt_correct` — its backward is the `pdiv`-contracted Jacobian
  of the masked forward in the image;
* `mobilenetv2ForwardBFullDoOpt_differentiableAt`.

The mask is held fixed; the derivative is in the image. At `none` the forward is
`mobilenetv2ForwardBFull` by `rfl`.

**How.** `MobileNetV2FullBVJP`'s chain of `vjpCompDiffAt`s over the seventeen bottlenecks,
unchanged (the trunk carries no site), closed by the head at the site: `mnv2HeadBDoOptHasVJPAt`,
the step tie's, and `mnv2HeadBDoOpt_differentiableAt` here.

**Hypotheses.** `MNV2PosB` and `MNV2SmoothAtB`, the drop-free theorem's own: the site is after the
head's relu6 and is linear, so it adds no kink clause and moves no activation a clause is stated
at.
-/

open Proofs Proofs.StableHLO

namespace Proofs.MobileNetV2TieB

open scoped BigOperators

/-- The head at the dropout site is differentiable where the plain head is. -/
theorem mnv2HeadBDoOpt_differentiableAt (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (cd : Option (Vec (N * oc))) (v : Vec (N * (ic * h * w)))
    (hs : MNV2HeadSmoothAtB N h w Wh bh εh γh βh v) :
    DifferentiableAt ℝ (mnv2HeadBDoOpt N h w Wh bh εh γh βh Wd bd cd) v :=
  ((batchMap_differentiable _ (dense_differentiable Wd bd)) _).comp v
    (((dropoutOpt_differentiable cd) _).comp v
      (((batchMap_differentiable _ (globalAvgPoolFlat_differentiable oc h w)) _).comp v
        (cbrB_differentiableAt N Wh bh εh hεh γh βh v hs)))

/-- The chain's VJP and differentiability at the site: the seventeen trunk joins of
    `mnv2ChainB`, then the head with its dropout site. -/
noncomputable def mnv2ChainBDo (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (hq : MNV2PosB w) (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (hx : MNV2SmoothAtB N w x) :
    HasVJPDiffAt (mnv2HeadBDoOpt N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘ mnv2PreB17 N w) x :=
  let p0 : HasVJPDiffAt (mnv2PreB0 N w) x :=
    ⟨mnv2StemBHasVJPAt N 112 112 w.sW w.sb w.sε hq.s w.sγ w.sβ x hx.stem,
      mnv2StemB_differentiableAt N 112 112 w.sW w.sb w.sε hq.s w.sγ w.sβ x hx.stem⟩
  let p1 : HasVJPDiffAt (mnv2PreB1 N w) x :=
    vjpCompDiffAt _ _ x p0 ⟨mnv2NoExpBHasVJPAt N 112 112 w.b1 hq.b1 _ hx.b1,
      mnv2NoExpB_differentiableAt N 112 112 w.b1 hq.b1 _ hx.b1⟩
  let p2 : HasVJPDiffAt (mnv2PreB2 N w) x :=
    vjpCompDiffAt _ _ x p1 ⟨mnv2StridedBHasVJPAt N 56 56 w.b2 hq.b2 _ hx.b2,
      mnv2StridedB_differentiableAt N 56 56 w.b2 hq.b2 _ hx.b2⟩
  let p3 : HasVJPDiffAt (mnv2PreB3 N w) x :=
    vjpCompDiffAt _ _ x p2 ⟨mnv2ResidBHasVJPAt N 56 56 w.b3 hq.b3 _ hx.b3,
      mnv2ResidB_differentiableAt N 56 56 w.b3 hq.b3 _ hx.b3⟩
  let p4 : HasVJPDiffAt (mnv2PreB4 N w) x :=
    vjpCompDiffAt _ _ x p3 ⟨mnv2StridedBHasVJPAt N 28 28 w.b4 hq.b4 _ hx.b4,
      mnv2StridedB_differentiableAt N 28 28 w.b4 hq.b4 _ hx.b4⟩
  let p5 : HasVJPDiffAt (mnv2PreB5 N w) x :=
    vjpCompDiffAt _ _ x p4 ⟨mnv2ResidBHasVJPAt N 28 28 w.b5 hq.b5 _ hx.b5,
      mnv2ResidB_differentiableAt N 28 28 w.b5 hq.b5 _ hx.b5⟩
  let p6 : HasVJPDiffAt (mnv2PreB6 N w) x :=
    vjpCompDiffAt _ _ x p5 ⟨mnv2ResidBHasVJPAt N 28 28 w.b6 hq.b6 _ hx.b6,
      mnv2ResidB_differentiableAt N 28 28 w.b6 hq.b6 _ hx.b6⟩
  let p7 : HasVJPDiffAt (mnv2PreB7 N w) x :=
    vjpCompDiffAt _ _ x p6 ⟨mnv2StridedBHasVJPAt N 14 14 w.b7 hq.b7 _ hx.b7,
      mnv2StridedB_differentiableAt N 14 14 w.b7 hq.b7 _ hx.b7⟩
  let p8 : HasVJPDiffAt (mnv2PreB8 N w) x :=
    vjpCompDiffAt _ _ x p7 ⟨mnv2ResidBHasVJPAt N 14 14 w.b8 hq.b8 _ hx.b8,
      mnv2ResidB_differentiableAt N 14 14 w.b8 hq.b8 _ hx.b8⟩
  let p9 : HasVJPDiffAt (mnv2PreB9 N w) x :=
    vjpCompDiffAt _ _ x p8 ⟨mnv2ResidBHasVJPAt N 14 14 w.b9 hq.b9 _ hx.b9,
      mnv2ResidB_differentiableAt N 14 14 w.b9 hq.b9 _ hx.b9⟩
  let p10 : HasVJPDiffAt (mnv2PreB10 N w) x :=
    vjpCompDiffAt _ _ x p9 ⟨mnv2ResidBHasVJPAt N 14 14 w.b10 hq.b10 _ hx.b10,
      mnv2ResidB_differentiableAt N 14 14 w.b10 hq.b10 _ hx.b10⟩
  let p11 : HasVJPDiffAt (mnv2PreB11 N w) x :=
    vjpCompDiffAt _ _ x p10 ⟨mnv2ExpOnlyBHasVJPAt N 14 14 w.b11 hq.b11 _ hx.b11,
      mnv2ExpOnlyB_differentiableAt N 14 14 w.b11 hq.b11 _ hx.b11⟩
  let p12 : HasVJPDiffAt (mnv2PreB12 N w) x :=
    vjpCompDiffAt _ _ x p11 ⟨mnv2ResidBHasVJPAt N 14 14 w.b12 hq.b12 _ hx.b12,
      mnv2ResidB_differentiableAt N 14 14 w.b12 hq.b12 _ hx.b12⟩
  let p13 : HasVJPDiffAt (mnv2PreB13 N w) x :=
    vjpCompDiffAt _ _ x p12 ⟨mnv2ResidBHasVJPAt N 14 14 w.b13 hq.b13 _ hx.b13,
      mnv2ResidB_differentiableAt N 14 14 w.b13 hq.b13 _ hx.b13⟩
  let p14 : HasVJPDiffAt (mnv2PreB14 N w) x :=
    vjpCompDiffAt _ _ x p13 ⟨mnv2StridedBHasVJPAt N 7 7 w.b14 hq.b14 _ hx.b14,
      mnv2StridedB_differentiableAt N 7 7 w.b14 hq.b14 _ hx.b14⟩
  let p15 : HasVJPDiffAt (mnv2PreB15 N w) x :=
    vjpCompDiffAt _ _ x p14 ⟨mnv2ResidBHasVJPAt N 7 7 w.b15 hq.b15 _ hx.b15,
      mnv2ResidB_differentiableAt N 7 7 w.b15 hq.b15 _ hx.b15⟩
  let p16 : HasVJPDiffAt (mnv2PreB16 N w) x :=
    vjpCompDiffAt _ _ x p15 ⟨mnv2ResidBHasVJPAt N 7 7 w.b16 hq.b16 _ hx.b16,
      mnv2ResidB_differentiableAt N 7 7 w.b16 hq.b16 _ hx.b16⟩
  let p17 : HasVJPDiffAt (mnv2PreB17 N w) x :=
    vjpCompDiffAt _ _ x p16 ⟨mnv2ExpOnlyBHasVJPAt N 7 7 w.b17 hq.b17 _ hx.b17,
      mnv2ExpOnlyB_differentiableAt N 7 7 w.b17 hq.b17 _ hx.b17⟩
  vjpCompDiffAt _ _ x p17
    ⟨mnv2HeadBDoOptHasVJPAt N 7 7 w.hW w.hb w.hε hq.h w.hγ w.hβ w.fcW w.fcb cd _ hx.head,
      mnv2HeadBDoOpt_differentiableAt N 7 7 w.hW w.hb w.hε hq.h w.hγ w.hβ w.fcW w.fcb cd _ hx.head⟩

/-- **MobileNetV2 with classifier dropout has a certified input-VJP at a smooth point**, at
    either value of the site. `none` is `mobilenetv2ForwardBFullHasVJPAt`'s function. -/
noncomputable def mobilenetv2ForwardBFullDoOptHasVJPAt (N : Nat) {nCls : Nat}
    (w : MNV2BWeights nCls) (hq : MNV2PosB w) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (hx : MNV2SmoothAtB N w x) :
    HasVJPAt (mobilenetv2ForwardBFullDoOpt N w cd) x :=
  (mnv2ChainBDo N w hq cd x hx).fst.congr
    (funext fun v => (mobilenetv2ForwardBFullDoOpt_eq_chain N w cd v).symm)

/-- **Public correctness theorem at the site**: the backward is the `pdiv`-contracted Jacobian
    of `mobilenetv2ForwardBFullDoOpt` in the image. -/
theorem mobilenetv2ForwardBFullDoOptHasVJPAt_correct (N : Nat) {nCls : Nat}
    (w : MNV2BWeights nCls) (hq : MNV2PosB w) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (hx : MNV2SmoothAtB N w x) (dy : Vec (N * nCls))
    (i : Fin (N * (3 * (2 * 112) * (2 * 112)))) :
    (mobilenetv2ForwardBFullDoOptHasVJPAt N w hq cd x hx).backward dy i =
      ∑ j : Fin (N * nCls), pdiv (mobilenetv2ForwardBFullDoOpt N w cd) x i j * dy j :=
  (mobilenetv2ForwardBFullDoOptHasVJPAt N w hq cd x hx).correct dy i

/-- The masked forward is differentiable in the image at every smooth point. -/
theorem mobilenetv2ForwardBFullDoOpt_differentiableAt (N : Nat) {nCls : Nat}
    (w : MNV2BWeights nCls) (hq : MNV2PosB w) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (hx : MNV2SmoothAtB N w x) :
    DifferentiableAt ℝ (mobilenetv2ForwardBFullDoOpt N w cd) x := by
  rw [show mobilenetv2ForwardBFullDoOpt N w cd
      = mnv2HeadBDoOpt N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘ mnv2PreB17 N w
    from funext (mobilenetv2ForwardBFullDoOpt_eq_chain N w cd)]
  exact (mnv2ChainBDo N w hq cd x hx).snd

end Proofs.MobileNetV2TieB
