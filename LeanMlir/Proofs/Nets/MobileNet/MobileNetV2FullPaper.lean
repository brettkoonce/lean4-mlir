import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPC

/-! # MobileNetV2's bottleneck weight records

`IVW` (expand `ic→mid` 1×1, depthwise 3×3, project `mid→oc` 1×1, per-channel BN after each) and
`IVWNoExp` (the t=1 first bottleneck: no expand conv). They hold kernels, epsilons, gammas and
betas — nothing that knows which BatchNorm world reduces them — so the batch-BN net
(`MobileNetV2FullB.lean`, `MNV2BWeights`) and its step tie bind them. `IVPos` / `IVNoExpPos`
are their BN-epsilon positivity bundles.

Paper `[t,c,n,s]` spec (stem 3×3-s2 3→32; head 1×1 320→1280 → GAP → dense):
  (1, 16,1,1) (6, 24,2,2) (6, 32,3,2) (6, 64,4,2) (6, 96,3,1) (6,160,3,2) (6,320,1,1)
Per-block (ic→oc, mid=t·ic, spatial, kind):
  b1  32→16   mid32  @112 noExp(t=1)     b10 64→64   mid384 @14  resid
  b2  16→24   mid96  112→56 strided      b11 64→96   mid384 @14  exp(no-resid, s=1)
  b3  24→24   mid144 @56  resid          b12 96→96   mid576 @14  resid
  b4  24→32   mid144 56→28 strided       b13 96→96   mid576 @14  resid
  b5  32→32   mid192 @28  resid          b14 96→160  mid576 14→7 strided
  b6  32→32   mid192 @28  resid          b15 160→160 mid960 @7   resid
  b7  32→64   mid192 28→14 strided       b16 160→160 mid960 @7   resid
  b8  64→64   mid384 @14  resid          b17 160→320 mid960 @7   exp(no-resid, s=1)
  b9  64→64   mid384 @14  resid
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block weight bundles (the 17-block net has 158 block params + stem/head/fc)
-- ════════════════════════════════════════════════════════════════

/-- Weights of one MobileNetV2 bottleneck (expand `ic→mid` 1×1, depthwise 3×3,
    project `mid→oc` 1×1, per-channel BN after each). -/
structure IVW (ic mid oc : Nat) where
  eW : Kernel4 mid ic 1 1
  eb : Vec mid
  eε : ℝ
  eγ : Vec mid
  eβ : Vec mid
  dW : DepthwiseKernel mid 3 3
  db : Vec mid
  dε : ℝ
  dγ : Vec mid
  dβ : Vec mid
  pW : Kernel4 oc mid 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

/-- Weights of the t=1 first bottleneck (NO expand conv): depthwise 3×3 on `ic`,
    project `ic→oc` 1×1, per-channel BN after each. -/
structure IVWNoExp (ic oc : Nat) where
  dW : DepthwiseKernel ic 3 3
  db : Vec ic
  dε : ℝ
  dγ : Vec ic
  dβ : Vec ic
  pW : Kernel4 oc ic 1 1
  pb : Vec oc
  pε : ℝ
  pγ : Vec oc
  pβ : Vec oc

-- ════════════════════════════════════════════════════════════════
-- § Per-block BN-epsilon positivity bundles
-- ════════════════════════════════════════════════════════════════

/-- The three BN epsilons of a full bottleneck are positive. -/
structure IVPos {ic mid oc : Nat} (q : IVW ic mid oc) : Prop where
  he : 0 < q.eε
  hd : 0 < q.dε
  hp : 0 < q.pε

/-- The two BN epsilons of the t=1 (no-expand) bottleneck are positive. -/
structure IVNoExpPos {ic oc : Nat} (q : IVWNoExp ic oc) : Prop where
  hd : 0 < q.dε
  hp : 0 < q.pε

end Proofs
