module

public import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4ParamGrad
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # MobileNetV4-Conv-M's whole-net input VJP at the masks

`MobileNetV4FullBVJP` certifies the input VJP of `mobilenetv4ForwardBFull`, the forward with no
regulariser. The `drop` / `do` renders train a different function: stochastic depth scales each
skip row's branch and classifier dropout masks the dense's input. This file states the same
theorem for that function, `mobilenetv4ForwardBFullDropOpt N f w sd cd`, at every choice of the
eighteen stochastic-depth sites `sd` and the dropout site `cd`:

* `mobilenetv4ForwardBFullDropOptHasVJPAt` — the certified input VJP at a smooth point;
* `mobilenetv4ForwardBFullDropOptHasVJPAt_correct` — its backward is the `pdiv`-contracted
  Jacobian of the masked forward in the image;
* `mobilenetv4ForwardBFullDropOpt_differentiableAt`.

The masks are held fixed; the derivative is in the image. At `none` / `none` the forward is
`mobilenetv4ForwardBFull` (`mobilenetv4ForwardBFullDropOpt_none_none`).

**How.** `MobileNetV4FullBVJP`'s bottom-up chain of `vjpCompDiffAt`s, one join per table row
instead of one per resolution group, over the per-block prefixes `mnv4Blk{k}` the step tie
already uses. A skip row's layer is `mnv4SkipDropLayer` at its site, certified where its body is
(the site is linear); the head is `mnv4HeadStackDo`, certified where the plain head is.

**Hypothesis.** `Mnv4SmoothAtDrop`: every relu off its kink at the activation the masked forward
gives it. It is the bundle `mnv4_net_lossGrad` takes, and `Mnv4SmoothAt` at no sites
(`mnv4SmoothAtDrop_of_smoothAt`). No `0 < ε` hypothesis appears; those are in the weight record.

The rendered backward at the masks is not stated as one graph here. Each row's backward graph is
its layer's `.faithful`, and the step tie's chain threads them (`mnv4SkipCotIn_eq_vjp_drop`,
`mnv4HeadCotIn_eq_vjp_do`).
-/

open Proofs Proofs.StableHLO

namespace Proofs.Mnv4TieB

open scoped BigOperators

/-- The masked chain's VJP and its differentiability together, one `vjpCompDiffAt` per join: the
    stem, the fused stage, the twenty-one rows each at its site, the head at its site. -/
noncomputable def mnv4ChainBDrop (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (hx : Mnv4SmoothAtDrop N f w sd x) :
    HasVJPDiffAt (mobilenetv4ForwardBFullDropOpt N f w sd cd) x :=
  let p0 : HasVJPDiffAt (mnv4Pre0 N f w) x :=
    ⟨mnv4StemBHasVJPAt N (2 * (2 * (2 * (2 * f)))) (2 * (2 * (2 * (2 * f)))) w.sW w.sb w.sE w.hsE w.sg w.sbt x hx.stem,
      mnv4StemB_differentiableAt N (2 * (2 * (2 * (2 * f)))) (2 * (2 * (2 * (2 * f)))) w.sW w.sb w.sE w.hsE w.sg w.sbt x hx.stem⟩
  let q0 : HasVJPDiffAt (mnv4Blk0 N f w) x :=
    vjpCompDiffAt (mnv4Pre0 N f w) (mnv4FusedStack N f w).fwd x p0
      ⟨(mnv4FusedStack N f w).vjp _ hx.fused, (mnv4FusedStack N f w).diff _ hx.fused⟩
  let q1 : HasVJPDiffAt (mnv4Blk1 N f w) x :=
    vjpCompDiffAt (mnv4Blk0 N f w) (mnv4StridedBodyOfRow N (mnv4Row1 f) w.b1).fwd x q0
      ⟨(mnv4StridedBodyOfRow N (mnv4Row1 f) w.b1).vjp _ hx.b1,
        (mnv4StridedBodyOfRow N (mnv4Row1 f) w.b1).diff _ hx.b1⟩
  let q2 : HasVJPDiffAt (mnv4Blk2 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk1 N f w) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row2 f) w.b2) (mnv4Sd sd 0)).fwd x q1
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row2 f) w.b2) (mnv4Sd sd 0)).vjp _ ⟨hx.b2, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row2 f) w.b2) (mnv4Sd sd 0)).diff _ ⟨hx.b2, trivial⟩⟩
  let q3 : HasVJPDiffAt (mnv4Blk3 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk2 N f w sd) (mnv4StridedBodyOfRow N (mnv4Row3 f) w.b3).fwd x q2
      ⟨(mnv4StridedBodyOfRow N (mnv4Row3 f) w.b3).vjp _ hx.b3,
        (mnv4StridedBodyOfRow N (mnv4Row3 f) w.b3).diff _ hx.b3⟩
  let q4 : HasVJPDiffAt (mnv4Blk4 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk3 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row4 f) w.b4) (mnv4Sd sd 1)).fwd x q3
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row4 f) w.b4) (mnv4Sd sd 1)).vjp _ ⟨hx.b4, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row4 f) w.b4) (mnv4Sd sd 1)).diff _ ⟨hx.b4, trivial⟩⟩
  let q5 : HasVJPDiffAt (mnv4Blk5 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk4 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row5 f) w.b5) (mnv4Sd sd 2)).fwd x q4
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row5 f) w.b5) (mnv4Sd sd 2)).vjp _ ⟨hx.b5, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row5 f) w.b5) (mnv4Sd sd 2)).diff _ ⟨hx.b5, trivial⟩⟩
  let q6 : HasVJPDiffAt (mnv4Blk6 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk5 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row6 f) w.b6) (mnv4Sd sd 3)).fwd x q5
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row6 f) w.b6) (mnv4Sd sd 3)).vjp _ ⟨hx.b6, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row6 f) w.b6) (mnv4Sd sd 3)).diff _ ⟨hx.b6, trivial⟩⟩
  let q7 : HasVJPDiffAt (mnv4Blk7 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk6 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row7 f) w.b7) (mnv4Sd sd 4)).fwd x q6
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row7 f) w.b7) (mnv4Sd sd 4)).vjp _ ⟨hx.b7, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row7 f) w.b7) (mnv4Sd sd 4)).diff _ ⟨hx.b7, trivial⟩⟩
  let q8 : HasVJPDiffAt (mnv4Blk8 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk7 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row8 f) w.b8) (mnv4Sd sd 5)).fwd x q7
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row8 f) w.b8) (mnv4Sd sd 5)).vjp _ ⟨hx.b8, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row8 f) w.b8) (mnv4Sd sd 5)).diff _ ⟨hx.b8, trivial⟩⟩
  let q9 : HasVJPDiffAt (mnv4Blk9 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk8 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row9 f) w.b9) (mnv4Sd sd 6)).fwd x q8
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row9 f) w.b9) (mnv4Sd sd 6)).vjp _ ⟨hx.b9, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row9 f) w.b9) (mnv4Sd sd 6)).diff _ ⟨hx.b9, trivial⟩⟩
  let q10 : HasVJPDiffAt (mnv4Blk10 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk9 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row10 f) w.b10) (mnv4Sd sd 7)).fwd x q9
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row10 f) w.b10) (mnv4Sd sd 7)).vjp _ ⟨hx.b10, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row10 f) w.b10) (mnv4Sd sd 7)).diff _ ⟨hx.b10, trivial⟩⟩
  let q11 : HasVJPDiffAt (mnv4Blk11 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk10 N f w sd) (mnv4StridedBodyOfRow N (mnv4Row11 f) w.b11).fwd x q10
      ⟨(mnv4StridedBodyOfRow N (mnv4Row11 f) w.b11).vjp _ hx.b11,
        (mnv4StridedBodyOfRow N (mnv4Row11 f) w.b11).diff _ hx.b11⟩
  let q12 : HasVJPDiffAt (mnv4Blk12 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk11 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row12 f) w.b12) (mnv4Sd sd 8)).fwd x q11
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row12 f) w.b12) (mnv4Sd sd 8)).vjp _ ⟨hx.b12, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row12 f) w.b12) (mnv4Sd sd 8)).diff _ ⟨hx.b12, trivial⟩⟩
  let q13 : HasVJPDiffAt (mnv4Blk13 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk12 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row13 f) w.b13) (mnv4Sd sd 9)).fwd x q12
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row13 f) w.b13) (mnv4Sd sd 9)).vjp _ ⟨hx.b13, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row13 f) w.b13) (mnv4Sd sd 9)).diff _ ⟨hx.b13, trivial⟩⟩
  let q14 : HasVJPDiffAt (mnv4Blk14 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk13 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row14 f) w.b14) (mnv4Sd sd 10)).fwd x q13
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row14 f) w.b14) (mnv4Sd sd 10)).vjp _ ⟨hx.b14, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row14 f) w.b14) (mnv4Sd sd 10)).diff _ ⟨hx.b14, trivial⟩⟩
  let q15 : HasVJPDiffAt (mnv4Blk15 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk14 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row15 f) w.b15) (mnv4Sd sd 11)).fwd x q14
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row15 f) w.b15) (mnv4Sd sd 11)).vjp _ ⟨hx.b15, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row15 f) w.b15) (mnv4Sd sd 11)).diff _ ⟨hx.b15, trivial⟩⟩
  let q16 : HasVJPDiffAt (mnv4Blk16 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk15 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row16 f) w.b16) (mnv4Sd sd 12)).fwd x q15
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row16 f) w.b16) (mnv4Sd sd 12)).vjp _ ⟨hx.b16, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row16 f) w.b16) (mnv4Sd sd 12)).diff _ ⟨hx.b16, trivial⟩⟩
  let q17 : HasVJPDiffAt (mnv4Blk17 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk16 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row17 f) w.b17) (mnv4Sd sd 13)).fwd x q16
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row17 f) w.b17) (mnv4Sd sd 13)).vjp _ ⟨hx.b17, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row17 f) w.b17) (mnv4Sd sd 13)).diff _ ⟨hx.b17, trivial⟩⟩
  let q18 : HasVJPDiffAt (mnv4Blk18 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk17 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row18 f) w.b18) (mnv4Sd sd 14)).fwd x q17
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row18 f) w.b18) (mnv4Sd sd 14)).vjp _ ⟨hx.b18, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row18 f) w.b18) (mnv4Sd sd 14)).diff _ ⟨hx.b18, trivial⟩⟩
  let q19 : HasVJPDiffAt (mnv4Blk19 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk18 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row19 f) w.b19) (mnv4Sd sd 15)).fwd x q18
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row19 f) w.b19) (mnv4Sd sd 15)).vjp _ ⟨hx.b19, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row19 f) w.b19) (mnv4Sd sd 15)).diff _ ⟨hx.b19, trivial⟩⟩
  let q20 : HasVJPDiffAt (mnv4Blk20 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk19 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row20 f) w.b20) (mnv4Sd sd 16)).fwd x q19
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row20 f) w.b20) (mnv4Sd sd 16)).vjp _ ⟨hx.b20, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row20 f) w.b20) (mnv4Sd sd 16)).diff _ ⟨hx.b20, trivial⟩⟩
  let q21 : HasVJPDiffAt (mnv4Blk21 N f w sd) x :=
    vjpCompDiffAt (mnv4Blk20 N f w sd) (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row21 f) w.b21) (mnv4Sd sd 17)).fwd x q20
      ⟨(mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row21 f) w.b21) (mnv4Sd sd 17)).vjp _ ⟨hx.b21, trivial⟩,
        (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row21 f) w.b21) (mnv4Sd sd 17)).diff _ ⟨hx.b21, trivial⟩⟩
  -- the head with the dropout site is certified where the head is: the site and the dense add `True`
  have hh : (mnv4HeadStackDo N f w cd).ok (mnv4Blk21 N f w sd x) :=
    ⟨hx.head.1, hx.head.2.1, hx.head.2.2.1, hx.head.2.2.2.1, hx.head.2.2.2.2.1, trivial, trivial⟩
  vjpCompDiffAt (mnv4Blk21 N f w sd) (mnv4HeadStackDo N f w cd).fwd x q21
    ⟨(mnv4HeadStackDo N f w cd).vjp _ hh, (mnv4HeadStackDo N f w cd).diff _ hh⟩

/-- **MobileNetV4-Conv-M with stochastic depth and classifier dropout has a certified input-VJP
    at a smooth point**, at every choice of the masks. `none` / `none` is
    `mobilenetv4ForwardBFullHasVJPAt`'s function. -/
noncomputable def mobilenetv4ForwardBFullDropOptHasVJPAt (N f : Nat) {nCls : Nat}
    (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (hx : Mnv4SmoothAtDrop N f w sd x) :
    HasVJPAt (mobilenetv4ForwardBFullDropOpt N f w sd cd) x :=
  (mnv4ChainBDrop N f w sd cd x hx).fst

/-- The masked forward is differentiable in the image at every smooth point. -/
theorem mobilenetv4ForwardBFullDropOpt_differentiableAt (N f : Nat) {nCls : Nat}
    (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (hx : Mnv4SmoothAtDrop N f w sd x) :
    DifferentiableAt ℝ (mobilenetv4ForwardBFullDropOpt N f w sd cd) x :=
  (mnv4ChainBDrop N f w sd cd x hx).snd

/-- **Public correctness theorem at the masks**: the backward is the `pdiv`-contracted Jacobian
    of `mobilenetv4ForwardBFullDropOpt` in the image, at every batch size, class count and choice
    of sites. -/
theorem mobilenetv4ForwardBFullDropOptHasVJPAt_correct (N f : Nat) {nCls : Nat}
    (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (hx : Mnv4SmoothAtDrop N f w sd x)
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    (mobilenetv4ForwardBFullDropOptHasVJPAt N f w sd cd x hx).backward dy i
      = ∑ j : Fin (N * nCls), pdiv (mobilenetv4ForwardBFullDropOpt N f w sd cd) x i j * dy j :=
  (mobilenetv4ForwardBFullDropOptHasVJPAt N f w sd cd x hx).correct dy i

end Proofs.Mnv4TieB
