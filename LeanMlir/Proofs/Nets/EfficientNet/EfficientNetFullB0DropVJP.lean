module

public import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Drop
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # EfficientNet-B0's whole-net input VJP at the masks

`EfficientNetFullB0` certifies the input VJP of `efficientnetForwardBFull`, the forward with no
regulariser. The `drop` / `dropdo` renders train `efficientnetForwardBFullDrop N w sd cd`:
stochastic depth on the nine skip blocks' branches and classifier dropout between the pool and
the dense. This file states the theorem for that function at every choice of the sites:

* `efficientnetForwardBFullDropHasVJP` — the certified input VJP, global: swish and the SE gate
  have no corner and a site is linear, so there is no smooth-point hypothesis, as for the
  drop-free net;
* `efficientnetForwardBFullDropHasVJP_correct` — its backward is the `pdiv`-contracted Jacobian
  of `efficientnetForwardBFullDrop` in the image, at every input;
* `efficientnetForwardBFullDrop_differentiable`.

The masks are held fixed; the derivative is in the image. At `none` / `none` the forward is
`efficientnetForwardBFull` (`efficientnetForwardBFullDrop_none`).

**How.** `efficientnetForwardBFullHasVJP`'s chain of `vjpComp`s with the nine residual blocks at
`mbResidDropWHasVJP` and the head at `headDoFwdBHasVJP`, `EfficientNetFullB0Drop`'s block-level
VJPs. The only hypothesis is every BatchNorm's `ε > 0`.
-/

namespace Proofs

open scoped BigOperators

/-- **`efficientnetForwardBFullDrop` is the `∘`-chain the VJP is stated on.** Equation-lemma
    `rw` and seventeen `comp_apply` rewrites, as `efficientnetForwardBFull_eq_chain`. -/
theorem efficientnetForwardBFullDrop_eq_chain (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * 224 * 224))) :
    efficientnetForwardBFullDrop N w sd cd x =
      (headDoFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘
        mbExpW N 7 7 w.b16 ∘
        mbResidDropW N 7 7 w.b15 (sd.map (· 8)) ∘
        mbResidDropW N 7 7 w.b14 (sd.map (· 7)) ∘
        mbResidDropW N 7 7 w.b13 (sd.map (· 6)) ∘
        mbStridedW N 7 7 w.b12 ∘
        mbResidDropW N 14 14 w.b11 (sd.map (· 5)) ∘
        mbResidDropW N 14 14 w.b10 (sd.map (· 4)) ∘
        mbExpW N 14 14 w.b9 ∘
        mbResidDropW N 14 14 w.b8 (sd.map (· 3)) ∘
        mbResidDropW N 14 14 w.b7 (sd.map (· 2)) ∘
        mbStridedW N 14 14 w.b6 ∘
        mbResidDropW N 28 28 w.b5 (sd.map (· 1)) ∘
        mbStridedW N 28 28 w.b4 ∘
        mbResidDropW N 56 56 w.b3 (sd.map (· 0)) ∘
        mbStridedW N 56 56 w.b2 ∘
        mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) x := by
  rw [efficientnetForwardBFullDrop]
  rw [Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply, Function.comp_apply]

/-- The chain's VJP together with its differentiability. -/
noncomputable def efficientnetChainDrop (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (hεw : w.EpsPos) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280))) :
    HasVJPDiff
      (headDoFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘
        mbExpW N 7 7 w.b16 ∘
        mbResidDropW N 7 7 w.b15 (sd.map (· 8)) ∘
        mbResidDropW N 7 7 w.b14 (sd.map (· 7)) ∘
        mbResidDropW N 7 7 w.b13 (sd.map (· 6)) ∘
        mbStridedW N 7 7 w.b12 ∘
        mbResidDropW N 14 14 w.b11 (sd.map (· 5)) ∘
        mbResidDropW N 14 14 w.b10 (sd.map (· 4)) ∘
        mbExpW N 14 14 w.b9 ∘
        mbResidDropW N 14 14 w.b8 (sd.map (· 3)) ∘
        mbResidDropW N 14 14 w.b7 (sd.map (· 2)) ∘
        mbStridedW N 14 14 w.b6 ∘
        mbResidDropW N 28 28 w.b5 (sd.map (· 1)) ∘
        mbStridedW N 28 28 w.b4 ∘
        mbResidDropW N 56 56 w.b3 (sd.map (· 0)) ∘
        mbStridedW N 56 56 w.b2 ∘
        mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) := by
  have dS := stemB_differentiable N (h := 112) (w := 112) w.sW w.sb w.sε hεw.s w.sγ w.sβ
  have vS := stemBHasVJP N (h := 112) (w := 112) w.sW w.sb w.sε hεw.s w.sγ w.sβ
  have d1 := mbNoExpW_differentiable N 112 112 w.b1 hεw.b1.d hεw.b1.p
  have v1 := mbNoExpWHasVJP N 112 112 w.b1 hεw.b1.d hεw.b1.p
  have d2 := mbStridedW_differentiable N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p
  have v2 := mbStridedWHasVJP N 56 56 w.b2 hεw.b2.e hεw.b2.d hεw.b2.p
  have d3 := mbResidDropW_differentiable N 56 56 w.b3 (sd.map (· 0)) hεw.b3.e hεw.b3.d hεw.b3.p
  have v3 := mbResidDropWHasVJP N 56 56 w.b3 (sd.map (· 0)) hεw.b3.e hεw.b3.d hεw.b3.p
  have d4 := mbStridedW_differentiable N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p
  have v4 := mbStridedWHasVJP N 28 28 w.b4 hεw.b4.e hεw.b4.d hεw.b4.p
  have d5 := mbResidDropW_differentiable N 28 28 w.b5 (sd.map (· 1)) hεw.b5.e hεw.b5.d hεw.b5.p
  have v5 := mbResidDropWHasVJP N 28 28 w.b5 (sd.map (· 1)) hεw.b5.e hεw.b5.d hεw.b5.p
  have d6 := mbStridedW_differentiable N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p
  have v6 := mbStridedWHasVJP N 14 14 w.b6 hεw.b6.e hεw.b6.d hεw.b6.p
  have d7 := mbResidDropW_differentiable N 14 14 w.b7 (sd.map (· 2)) hεw.b7.e hεw.b7.d hεw.b7.p
  have v7 := mbResidDropWHasVJP N 14 14 w.b7 (sd.map (· 2)) hεw.b7.e hεw.b7.d hεw.b7.p
  have d8 := mbResidDropW_differentiable N 14 14 w.b8 (sd.map (· 3)) hεw.b8.e hεw.b8.d hεw.b8.p
  have v8 := mbResidDropWHasVJP N 14 14 w.b8 (sd.map (· 3)) hεw.b8.e hεw.b8.d hεw.b8.p
  have d9 := mbExpW_differentiable N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p
  have v9 := mbExpWHasVJP N 14 14 w.b9 hεw.b9.e hεw.b9.d hεw.b9.p
  have d10 := mbResidDropW_differentiable N 14 14 w.b10 (sd.map (· 4)) hεw.b10.e hεw.b10.d hεw.b10.p
  have v10 := mbResidDropWHasVJP N 14 14 w.b10 (sd.map (· 4)) hεw.b10.e hεw.b10.d hεw.b10.p
  have d11 := mbResidDropW_differentiable N 14 14 w.b11 (sd.map (· 5)) hεw.b11.e hεw.b11.d hεw.b11.p
  have v11 := mbResidDropWHasVJP N 14 14 w.b11 (sd.map (· 5)) hεw.b11.e hεw.b11.d hεw.b11.p
  have d12 := mbStridedW_differentiable N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p
  have v12 := mbStridedWHasVJP N 7 7 w.b12 hεw.b12.e hεw.b12.d hεw.b12.p
  have d13 := mbResidDropW_differentiable N 7 7 w.b13 (sd.map (· 6)) hεw.b13.e hεw.b13.d hεw.b13.p
  have v13 := mbResidDropWHasVJP N 7 7 w.b13 (sd.map (· 6)) hεw.b13.e hεw.b13.d hεw.b13.p
  have d14 := mbResidDropW_differentiable N 7 7 w.b14 (sd.map (· 7)) hεw.b14.e hεw.b14.d hεw.b14.p
  have v14 := mbResidDropWHasVJP N 7 7 w.b14 (sd.map (· 7)) hεw.b14.e hεw.b14.d hεw.b14.p
  have d15 := mbResidDropW_differentiable N 7 7 w.b15 (sd.map (· 8)) hεw.b15.e hεw.b15.d hεw.b15.p
  have v15 := mbResidDropWHasVJP N 7 7 w.b15 (sd.map (· 8)) hεw.b15.e hεw.b15.d hεw.b15.p
  have d16 := mbExpW_differentiable N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p
  have v16 := mbExpWHasVJP N 7 7 w.b16 hεw.b16.e hεw.b16.d hεw.b16.p
  have dH := headDoFwdB_differentiable N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb cd
  have vH := headDoFwdBHasVJP N (h := 7) (w := 7) w.hW w.hb w.hε hεw.h w.hγ w.hβ w.fcW w.fcb cd
  have e1 := vjpComp _ _ dS d1 vS v1;            have f1 := d1.comp dS
  have e2 := vjpComp _ _ f1 d2 e1 v2;            have f2 := d2.comp f1
  have e3 := vjpComp _ _ f2 d3 e2 v3;            have f3 := d3.comp f2
  have e4 := vjpComp _ _ f3 d4 e3 v4;            have f4 := d4.comp f3
  have e5 := vjpComp _ _ f4 d5 e4 v5;            have f5 := d5.comp f4
  have e6 := vjpComp _ _ f5 d6 e5 v6;            have f6 := d6.comp f5
  have e7 := vjpComp _ _ f6 d7 e6 v7;            have f7 := d7.comp f6
  have e8 := vjpComp _ _ f7 d8 e7 v8;            have f8 := d8.comp f7
  have e9 := vjpComp _ _ f8 d9 e8 v9;            have f9 := d9.comp f8
  have e10 := vjpComp _ _ f9 d10 e9 v10;            have f10 := d10.comp f9
  have e11 := vjpComp _ _ f10 d11 e10 v11;            have f11 := d11.comp f10
  have e12 := vjpComp _ _ f11 d12 e11 v12;            have f12 := d12.comp f11
  have e13 := vjpComp _ _ f12 d13 e12 v13;            have f13 := d13.comp f12
  have e14 := vjpComp _ _ f13 d14 e13 v14;            have f14 := d14.comp f13
  have e15 := vjpComp _ _ f14 d15 e14 v15;            have f15 := d15.comp f14
  have e16 := vjpComp _ _ f15 d16 e15 v16;            have f16 := d16.comp f15
  exact ⟨vjpComp _ _ f16 dH e16 vH, dH.comp f16⟩

/-- **EfficientNet-B0 with stochastic depth and classifier dropout has a certified input VJP**,
    at every choice of the masks and every input: the chain stem → sixteen MBConv blocks → head,
    the nine skip blocks at their drop sites and the head at its dropout site. Stated on the
    `∘`-composition, as `efficientnetForwardBFullHasVJP`. -/
noncomputable def efficientnetForwardBFullDropHasVJP (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (hεw : w.EpsPos) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280))) :
    HasVJP
      (headDoFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘
        mbExpW N 7 7 w.b16 ∘
        mbResidDropW N 7 7 w.b15 (sd.map (· 8)) ∘
        mbResidDropW N 7 7 w.b14 (sd.map (· 7)) ∘
        mbResidDropW N 7 7 w.b13 (sd.map (· 6)) ∘
        mbStridedW N 7 7 w.b12 ∘
        mbResidDropW N 14 14 w.b11 (sd.map (· 5)) ∘
        mbResidDropW N 14 14 w.b10 (sd.map (· 4)) ∘
        mbExpW N 14 14 w.b9 ∘
        mbResidDropW N 14 14 w.b8 (sd.map (· 3)) ∘
        mbResidDropW N 14 14 w.b7 (sd.map (· 2)) ∘
        mbStridedW N 14 14 w.b6 ∘
        mbResidDropW N 28 28 w.b5 (sd.map (· 1)) ∘
        mbStridedW N 28 28 w.b4 ∘
        mbResidDropW N 56 56 w.b3 (sd.map (· 0)) ∘
        mbStridedW N 56 56 w.b2 ∘
        mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ) :=
  (efficientnetChainDrop N w hεw sd cd).fst

/-- **Public correctness theorem at the masks**: the backward is the `pdiv`-contracted Jacobian
    of `efficientnetForwardBFullDrop` in the image, at every input. -/
theorem efficientnetForwardBFullDropHasVJP_correct (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (hεw : w.EpsPos) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * 224 * 224))) (dy : Vec (N * nCls)) (i : Fin (N * (3 * 224 * 224))) :
    (efficientnetForwardBFullDropHasVJP N w hεw sd cd).backward x dy i =
      ∑ j : Fin (N * nCls), pdiv (efficientnetForwardBFullDrop N w sd cd) x i j * dy j := by
  have h := (efficientnetForwardBFullDropHasVJP N w hεw sd cd).correct x dy i
  rwa [show efficientnetForwardBFullDrop N w sd cd =
        (headDoFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘
        mbExpW N 7 7 w.b16 ∘
        mbResidDropW N 7 7 w.b15 (sd.map (· 8)) ∘
        mbResidDropW N 7 7 w.b14 (sd.map (· 7)) ∘
        mbResidDropW N 7 7 w.b13 (sd.map (· 6)) ∘
        mbStridedW N 7 7 w.b12 ∘
        mbResidDropW N 14 14 w.b11 (sd.map (· 5)) ∘
        mbResidDropW N 14 14 w.b10 (sd.map (· 4)) ∘
        mbExpW N 14 14 w.b9 ∘
        mbResidDropW N 14 14 w.b8 (sd.map (· 3)) ∘
        mbResidDropW N 14 14 w.b7 (sd.map (· 2)) ∘
        mbStridedW N 14 14 w.b6 ∘
        mbResidDropW N 28 28 w.b5 (sd.map (· 1)) ∘
        mbStridedW N 28 28 w.b4 ∘
        mbResidDropW N 56 56 w.b3 (sd.map (· 0)) ∘
        mbStridedW N 56 56 w.b2 ∘
        mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ)
      from funext (efficientnetForwardBFullDrop_eq_chain N w sd cd)]

/-- The masked forward is differentiable in the image everywhere. -/
theorem efficientnetForwardBFullDrop_differentiable (N : Nat) {nCls : Nat} (w : B0Weights nCls)
    (hεw : w.EpsPos) (sd : Option (Fin 9 → Vec N)) (cd : Option (Vec (N * 1280))) :
    Differentiable ℝ (efficientnetForwardBFullDrop N w sd cd) := by
  rw [show efficientnetForwardBFullDrop N w sd cd =
        (headDoFwdB N (h := 7) (w := 7) w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb cd ∘
        mbExpW N 7 7 w.b16 ∘
        mbResidDropW N 7 7 w.b15 (sd.map (· 8)) ∘
        mbResidDropW N 7 7 w.b14 (sd.map (· 7)) ∘
        mbResidDropW N 7 7 w.b13 (sd.map (· 6)) ∘
        mbStridedW N 7 7 w.b12 ∘
        mbResidDropW N 14 14 w.b11 (sd.map (· 5)) ∘
        mbResidDropW N 14 14 w.b10 (sd.map (· 4)) ∘
        mbExpW N 14 14 w.b9 ∘
        mbResidDropW N 14 14 w.b8 (sd.map (· 3)) ∘
        mbResidDropW N 14 14 w.b7 (sd.map (· 2)) ∘
        mbStridedW N 14 14 w.b6 ∘
        mbResidDropW N 28 28 w.b5 (sd.map (· 1)) ∘
        mbStridedW N 28 28 w.b4 ∘
        mbResidDropW N 56 56 w.b3 (sd.map (· 0)) ∘
        mbStridedW N 56 56 w.b2 ∘
        mbNoExpW N 112 112 w.b1 ∘
        stemB N (h := 112) (w := 112) w.sW w.sb w.sε w.sγ w.sβ)
      from funext (efficientnetForwardBFullDrop_eq_chain N w sd cd)]
  exact (efficientnetChainDrop N w hεw sd cd).snd

end Proofs
