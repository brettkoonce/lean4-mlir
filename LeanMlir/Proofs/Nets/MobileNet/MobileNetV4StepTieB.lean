import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FoldB
import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetStepTie

/-! # T3 §1a tie for MobileNetV4-Conv-M — every gradient node at its CHAIN cotangent

`MobileNetV4FoldB.lean` proves every parameter gradient node denotes the certified gradient
**for an arbitrary cotangent**. This file removes that freedom: each node is stated at the
cotangent the render's own backward chain delivers, driven by a loss cotangent `g` at the logits.

⚠⚠ **No accuracy is quoted for this net.** Conv-M has no Imagenette run and no verified ImageNet
run; the ties that pin these statements to the reference's function are the 2026-09-07 pair.

## ⭐ The UIB bottleneck is LINEAR, and that makes MNv4's chain shorter than ResNet's

`uibFwdSkipB` emits `addVB (project-BN out) (block input)` with **no activation after the add** and
none after the project's BatchNorm. So the block-output cotangent `dyOut` reaches the project BN's
γ and β *directly* — where ResNet-50's `r50IdCotA` must first pass through the post-residual relu's
mask, and where its skip branch carries the masked cotangent. Here the skip fan-in is
`addVB (body dx) dyOut`, unmasked.

⭐⭐ **And every block's `*CotIn_eq_vjp` is `mnv4BodyOfRow_faithful`, not a new derivation.** The
UIB bodies are `CertLayer`s, so `den (graph x e) = vjp.backward (den e)` is already a theorem one
tier down — the very fact 4.2a/4.2c/§3.5c re-derive per block for r34, mnv2 and R50. This file
composes certified VJPs; it does not re-prove them.

## ⚠⚠ Everything here is GENERIC IN THE ROW, and that is load-bearing

Every definition and theorem below takes a `UibSpec` binder `s` and reads its widths off it, so
`s.ic`, `s.h` and `s.ic * s.expand` are VARIABLES. That is not a convenience: MNv4's resolutions
are literals, and `MobileNetV4FullB.lean` records four separate kernel blow-ups caused by letting
`den` and width-indexed `rfl`s actually RUN at 224/112/56/28/14/7. Stated at a row binder they stay
stuck; the capstone then instantiates at the 21 concrete rows, which is application and is free.

## The chain, node for node from `uibBackSkipGradB`

| cotangent | at | feeds |
|---|---|---|
| `dyOut` | the block output — and the project BN's output, the add being linear | `%u{p}pg`, `%u{p}pbt` |
| `CotPc` | project conv's output (`bnBatchBack`) | `%u{p}pW` |
| `CotDn` | post-DW BN's output (`convBackBatched`, then the post-DW relu's `selectPos` mask) | `%u{p}dg`, `%u{p}dbt` |
| `CotDc` | post-DW conv's output | `%u{p}dW` |
| `CotEn` | expand BN's output (`depthwiseBackBatched`, then the expand relu's mask) | `%u{p}eg`, `%u{p}ebt` |
| `CotEc` | expand conv's output | `%u{p}eW` |
| `CotQn` | pre-DW BN's output | `%u{p}qg`, `%u{p}qbt` |
| `CotQc` | pre-DW conv's output | `%u{p}qW` |
| `CotIn` | the block input — `addVB (depthwiseBackBatched dQc) dyOut` | the previous block |

⛔ A cotangent one step off is a silently wrong gradient, not a type error: `%u{p}eg` reads the
cotangent at the expand BN's OUTPUT and `%u{p}eW` the one at the expand CONV's output, and both
have the same type.

⛔ **One replica.** Under `mnv4in_adamdp64*` every node named here feeds `allReduceMeanF`
(`DataParallelNode.lean`, §4d); this is the per-replica gradient.
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.ResNet34TieB Proofs.EnetTiePoC

namespace Proofs.Mnv4TieB

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The ExtraDW block's cotangent chain — 13 of Conv-M's 21 rows
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the project CONV's output — `dyOut` through the project BN's backward.
    ⭐ `dyOut` itself is the cotangent at the project BN's output: the bottleneck is linear, so
    nothing masks it. Feeds `%u{p}pW`. -/
noncomputable def mnv4CotPc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.oc * s.h * s.h)) :=
  bnInB N s.oc s.h s.h p.ez p.gz
    (batchMap N (flatConv p.Wz p.bz)
      ((mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd
        ((mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    dyOut

/-- Cotangent at the post-DW BN's output — the project conv's input-VJP, masked by the post-DW
    relu. Feeds `%u{p}dg` and `%u{p}dbt`. -/
noncomputable def mnv4CotDn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (bnBatchLA N (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
      (batchMap N (depthwiseFlat p.Wd p.bd)
        ((mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    (cInB N p.Wz p.bz (mnv4CotPc N s p xin dyOut))

/-- Cotangent at the post-DW CONV's output — through the post-DW BN's backward. Feeds `%u{p}dW`. -/
noncomputable def mnv4CotDc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnInB N (s.ic * s.expand) s.h s.h p.ed p.gd
    (batchMap N (depthwiseFlat p.Wd p.bd)
      ((mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
        ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (mnv4CotDn N s p xin dyOut)

/-- Cotangent at the expand BN's output — masked by the expand relu. Feeds `%u{p}eg`/`%u{p}ebt`.

    ⭐⭐ **This is where the chain DISPATCHES on the table**, exactly as `mnv4PostDWSlot` does and
    off the same row: with a post-depthwise the incoming cotangent is that depthwise's input-VJP;
    without one (`postDWk = 0`, the ConvNeXt-like and FFN rows) the project conv's input-VJP
    arrives here directly, because the render emits no post-DW nodes at all. One chain, three
    stride-1 profiles, and the `if` reduces at every concrete row. -/
noncomputable def mnv4CotEn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (bnBatchLA N (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be)
        ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (if s.postDWk = 0 then cInB N p.Wz p.bz (mnv4CotPc N s p xin dyOut)
     else dInB N p.Wd p.bd (mnv4CotDc N s p xin dyOut))

/-- Cotangent at the expand CONV's output — through the expand BN's backward. Feeds `%u{p}eW`. -/
noncomputable def mnv4CotEc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnInB N (s.ic * s.expand) s.h s.h p.ee p.ge
    (batchMap N (flatConv p.We p.be)
      ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
        xin))
    (mnv4CotEn N s p xin dyOut)

/-- Cotangent at the pre-DW BN's output — the expand conv's input-VJP, masked by the pre-DW relu.
    Feeds `%u{p}qg` and `%u{p}qbt`. -/
noncomputable def mnv4CotQn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.h * s.h))
    (bnBatchLA N s.ic s.h s.h p.eq_ p.gq p.bq2
      (batchMap N (depthwiseFlat p.Wq p.bq) xin))
    (cInB N p.We p.be (mnv4CotEc N s p xin dyOut))

/-- Cotangent at the pre-DW CONV's output — through the pre-DW BN's backward. Feeds `%u{p}qW`. -/
noncomputable def mnv4CotQc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  bnInB N s.ic s.h s.h p.eq_ p.gq
    (batchMap N (depthwiseFlat p.Wq p.bq) xin)
    (mnv4CotQn N s p xin dyOut)


-- ════════════════════════════════════════════════════════════════
-- § The PRE-STRIDED block's chain — rows 1, 3 and 11, and no skip
-- ════════════════════════════════════════════════════════════════

/-! ⚠⚠ **A near-copy of the stride-1 chain, and it has to be.** Only two things differ — the
leading depthwise is `depthwiseStride2Flat` rather than `depthwiseFlat`, and the block input sits
at `2h` — but those two changes run through every type in the chain, so the whole thing is
re-stated rather than instantiated. ⭐ Everything from the expand down is the same composition at
the reduced resolution; the stride is entirely consumed by the first op, which is what
`mnv4UibPreStridedBody` means one tier up.

⛔ **And there is no skip**: all three stride-2 rows change channels (`ic ≠ oc`), so the block IS
the body, `dx` is the strided depthwise's input-VJP alone, and there is no `addVB` fan-in. -/

/-- Cotangent at the project CONV's output — `dyOut` through the project BN's backward.
    ⭐ `dyOut` itself is the cotangent at the project BN's output: the bottleneck is linear, so
    nothing masks it. Feeds `%u{p}pW`. -/
noncomputable def mnv4SCotPc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.oc * s.h * s.h)) :=
  bnInB N s.oc s.h s.h p.ez p.gz
    (batchMap N (flatConv p.Wz p.bz)
      ((mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd
        ((mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    dyOut

/-- Cotangent at the post-DW BN's output — the project conv's input-VJP, masked by the post-DW
    relu. Feeds `%u{p}dg` and `%u{p}dbt`. -/
noncomputable def mnv4SCotDn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (bnBatchLA N (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
      (batchMap N (depthwiseFlat p.Wd p.bd)
        ((mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    (cInB N p.Wz p.bz (mnv4SCotPc N s p xin dyOut))

/-- Cotangent at the post-DW CONV's output — through the post-DW BN's backward. Feeds `%u{p}dW`. -/
noncomputable def mnv4SCotDc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnInB N (s.ic * s.expand) s.h s.h p.ed p.gd
    (batchMap N (depthwiseFlat p.Wd p.bd)
      ((mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
        ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (mnv4SCotDn N s p xin dyOut)

/-- Cotangent at the expand BN's output — masked by the expand relu. Feeds `%u{p}eg`/`%u{p}ebt`.

    ⚠ No dispatch here, unlike the stride-1 chain: all three of Conv-M's stride-2 rows have
    `postDWk > 0`, so the post-depthwise is always present on this path. -/
noncomputable def mnv4SCotEn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (bnBatchLA N (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be)
        ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (dInB N p.Wd p.bd (mnv4SCotDc N s p xin dyOut))

/-- Cotangent at the expand CONV's output — through the expand BN's backward. Feeds `%u{p}eW`. -/
noncomputable def mnv4SCotEc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnInB N (s.ic * s.expand) s.h s.h p.ee p.ge
    (batchMap N (flatConv p.We p.be)
      ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
        xin))
    (mnv4SCotEn N s p xin dyOut)

/-- Cotangent at the STRIDED pre-DW BN's output — the expand conv's input-VJP, masked by the pre-DW relu.
    Feeds `%u{p}qg` and `%u{p}qbt`. -/
noncomputable def mnv4SCotQn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.h * s.h))
    (bnBatchLA N s.ic s.h s.h p.eq_ p.gq p.bq2
      (batchMap N (depthwiseStride2Flat p.Wq p.bq) xin))
    (cInB N p.We p.be (mnv4SCotEc N s p xin dyOut))

/-- Cotangent at the STRIDED pre-DW conv's output — through the pre-DW BN's backward. Feeds `%u{p}qW`. -/
noncomputable def mnv4SCotQc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  bnInB N s.ic s.h s.h p.eq_ p.gq
    (batchMap N (depthwiseStride2Flat p.Wq p.bq) xin)
    (mnv4SCotQn N s p xin dyOut)

/-- **The pre-strided block's input cotangent** — the STRIDED depthwise's input-VJP, landing at
    `2h`. No fan-in: `ic ≠ oc`, so the block has no skip. -/
noncomputable def mnv4SBodyCotIn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * (2 * s.h) * (2 * s.h))) :=
  dStridedInB N p.Wq p.bq (mnv4SCotQc N s p xin dyOut)

/-- **The BODY's input cotangent** — what the render's `dx` carries before the skip fan-in.

    ⭐ Dispatches on `s.preDWk` the way `mnv4PreDWSlot` does: with a pre-depthwise the body's `dx`
    is that depthwise's input-VJP, without one it is the expand conv's. -/
noncomputable def mnv4BodyCotIn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  if s.preDWk = 0 then cInB N p.We p.be (mnv4CotEc N s p xin dyOut)
  else dInB N p.Wq p.bq (mnv4CotQc N s p xin dyOut)

/-- The skip fan-in itself: `body dx + dyOut`, at the block-input shape.

    ⚠ Split from `mnv4BodyCotIn` for the same reason `mnv4SkipGraphB` is split from the body graph
    builders: the add needs `s.oc` and `s.ic` to be the SAME type, which they are at every stride-1
    row and are not at a row binder. The body's cotangent is row-generic; the add is applied at the
    concrete row, where `s.oc = s.ic` is `rfl`. -/
noncomputable def mnv4SkipCotIn {N n : Nat} (bodyDx dyOut : Vec (N * n)) : Vec (N * n) :=
  fun i => bodyDx i + dyOut i

-- ════════════════════════════════════════════════════════════════
-- § The ExtraDW block, tied — all TWELVE parameter nodes
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **ExtraDW block, tied.** All twelve parameter nodes — four conv/depthwise weights and four
    BatchNorm γ/β pairs — denote the certified batched `Σ_n` gradient at the real forward
    activations and the real backward-chain cotangent driven by `dyOut`.

    ⚠ Each BatchNorm's γ/β reads the cotangent at THAT BatchNorm's output (`CotQn`, `CotEn`,
    `CotDn`, and `dyOut` itself for the project) while its conv reads the one at the conv's output
    (`CotQc`, `CotEc`, `CotDc`, `CotPc`). Off by one and the gradient is silently wrong — the two
    have the same type. ⭐ The project BN's pair reads `dyOut` UNMASKED: the bottleneck is linear.

    ⛔ There are no conv-bias conjuncts: `MobileNetV4RenderB` has no `convBias` flag, so those ops
    are never emitted and every slot here is exercised by the artifact. -/
def mnv4ExtraDWTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let dc := batchMap N (depthwiseFlat p.Wd p.bd) er
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotQn := mnv4CotQn N s p xin dyOut
  let cotQc := mnv4CotQc N s p xin dyOut
  let cotEn := mnv4CotEn N s p xin dyOut
  let cotEc := mnv4CotEc N s p xin dyOut
  let cotDn := mnv4CotDn N s p xin dyOut
  let cotDc := mnv4CotDc N s p xin dyOut
  let cotPc := mnv4CotPc N s p xin dyOut
  (∀ idx : Fin (s.ic * s.preDWk * s.preDWk),
      den (SHlo.depthwiseWeightGradB xN p.bq xin p.Wq (.operand cotN cotQc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.ic * s.h * s.h),
            pdiv (fun v' : Vec (s.ic * s.preDWk * s.preDWk) =>
                    depthwiseFlat (Tensor3.unflatten v') p.bq
                      (batchSlice N (s.ic * s.h * s.h) xin n))
                 (Tensor3.flatten p.Wq) idx j * batchSlice N (s.ic * s.h * s.h) cotQc n j)
  ∧
  (∀ k : Fin (s.ic),
      den (SHlo.bnGammaGradB vN epsStr p.eq_ (reassocB N (s.ic) s.h s.h qc)
            (.operand cotN (reassocB N (s.ic) s.h s.h cotQn))) k
        = ∑ j : Fin ((s.ic) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic) =>
                    bnPerChannelFlat (s.ic) (N * (s.h * s.h)) p.eq_ γ' p.bq2
                      (bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h qc)))
                 p.gq k j * bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h cotQn) j)
  ∧
  (∀ k : Fin (s.ic),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic) s.h s.h cotQn))) k
        = ∑ j : Fin ((s.ic) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic) =>
                    bnPerChannelFlat (s.ic) (N * (s.h * s.h)) p.eq_ p.gq β'
                      (bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h qc)))
                 p.bq2 k j * bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h cotQn) j)
  ∧
  (∀ idx : Fin ((s.ic * s.expand) * s.ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN cotEc)) idx
        = ∑ n : Fin N, ∑ j : Fin ((s.ic * s.expand) * s.h * s.h),
            pdiv (fun v' : Vec ((s.ic * s.expand) * s.ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.be
                      (Tensor3.unflatten (batchSlice N (s.ic * s.h * s.h) qr n))))
                 (Kernel4.flatten p.We) idx j
              * batchSlice N ((s.ic * s.expand) * s.h * s.h) cotEc n j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee γ' p.be2
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.ge k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee p.ge β'
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.be2 k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ idx : Fin ((s.ic * s.expand) * s.postDWk * s.postDWk),
      den (SHlo.depthwiseWeightGradB xN p.bd er p.Wd (.operand cotN cotDc)) idx
        = ∑ n : Fin N, ∑ j : Fin ((s.ic * s.expand) * s.h * s.h),
            pdiv (fun v' : Vec ((s.ic * s.expand) * s.postDWk * s.postDWk) =>
                    depthwiseFlat (Tensor3.unflatten v') p.bd
                      (batchSlice N ((s.ic * s.expand) * s.h * s.h) er n))
                 (Tensor3.flatten p.Wd) idx j
              * batchSlice N ((s.ic * s.expand) * s.h * s.h) cotDc n j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnGammaGradB vN epsStr p.ed (reassocB N (s.ic * s.expand) s.h s.h dc)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotDn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ed γ' p.bd2
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h dc)))
                 p.gd k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotDn) j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotDn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ed p.gd β'
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h dc)))
                 p.bd2 k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotDn) j)
  ∧
  (∀ idx : Fin (s.oc * (s.ic * s.expand) * 1 * 1),
      den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.oc * s.h * s.h),
            pdiv (fun v' : Vec (s.oc * (s.ic * s.expand) * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.bz
                      (Tensor3.unflatten (batchSlice N ((s.ic * s.expand) * s.h * s.h) dr n))))
                 (Kernel4.flatten p.Wz) idx j * batchSlice N (s.oc * s.h * s.h) cotPc n j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N (s.oc) s.h s.h pc)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez γ' p.bz2
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.gz k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez p.gz β'
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.bz2 k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)

/-- ⭐⭐ **And it holds** — twelve instantiations of `MobileNetV4FoldB`'s `∀ cot` fold with
    the freedom removed. Nothing here is new mathematics; what is new is that the cotangents are
    the chain's, not free. -/
theorem mnv4_extradw_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4ExtraDWTiedB N s xN cotN vN epsStr p xin dyOut := by
  unfold mnv4ExtraDWTiedB
  intro qr er dr qc ec dc pc cotQn cotQc cotEn cotEc cotDn cotDc cotPc
  exact ⟨fun idx => EnetPoCG.depthwiseWGradB_den xN cotN p.bq xin p.Wq cotQc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.eq_ p.gq p.bq2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.eq_ p.gq p.bq2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.be qr p.We cotEc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ee p.ge p.be2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ee p.ge p.be2 _ _ k,
    fun idx => EnetPoCG.depthwiseWGradB_den xN cotN p.bd er p.Wd cotDc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ed p.gd p.bd2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ed p.gd p.bd2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.bz dr p.Wz cotPc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ez p.gz p.bz2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ez p.gz p.bz2 _ _ k⟩


def mnv4ConvNeXtTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotQn := mnv4CotQn N s p xin dyOut
  let cotQc := mnv4CotQc N s p xin dyOut
  let cotEn := mnv4CotEn N s p xin dyOut
  let cotEc := mnv4CotEc N s p xin dyOut
  let cotPc := mnv4CotPc N s p xin dyOut
  (∀ idx : Fin (s.ic * s.preDWk * s.preDWk),
      den (SHlo.depthwiseWeightGradB xN p.bq xin p.Wq (.operand cotN cotQc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.ic * s.h * s.h),
            pdiv (fun v' : Vec (s.ic * s.preDWk * s.preDWk) =>
                    depthwiseFlat (Tensor3.unflatten v') p.bq
                      (batchSlice N (s.ic * s.h * s.h) xin n))
                 (Tensor3.flatten p.Wq) idx j * batchSlice N (s.ic * s.h * s.h) cotQc n j)
  ∧
  (∀ k : Fin (s.ic),
      den (SHlo.bnGammaGradB vN epsStr p.eq_ (reassocB N (s.ic) s.h s.h qc)
            (.operand cotN (reassocB N (s.ic) s.h s.h cotQn))) k
        = ∑ j : Fin ((s.ic) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic) =>
                    bnPerChannelFlat (s.ic) (N * (s.h * s.h)) p.eq_ γ' p.bq2
                      (bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h qc)))
                 p.gq k j * bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h cotQn) j)
  ∧
  (∀ k : Fin (s.ic),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic) s.h s.h cotQn))) k
        = ∑ j : Fin ((s.ic) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic) =>
                    bnPerChannelFlat (s.ic) (N * (s.h * s.h)) p.eq_ p.gq β'
                      (bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h qc)))
                 p.bq2 k j * bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h cotQn) j)
  ∧
  (∀ idx : Fin ((s.ic * s.expand) * s.ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN cotEc)) idx
        = ∑ n : Fin N, ∑ j : Fin ((s.ic * s.expand) * s.h * s.h),
            pdiv (fun v' : Vec ((s.ic * s.expand) * s.ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.be
                      (Tensor3.unflatten (batchSlice N (s.ic * s.h * s.h) qr n))))
                 (Kernel4.flatten p.We) idx j
              * batchSlice N ((s.ic * s.expand) * s.h * s.h) cotEc n j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee γ' p.be2
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.ge k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee p.ge β'
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.be2 k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ idx : Fin (s.oc * (s.ic * s.expand) * 1 * 1),
      den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.oc * s.h * s.h),
            pdiv (fun v' : Vec (s.oc * (s.ic * s.expand) * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.bz
                      (Tensor3.unflatten (batchSlice N ((s.ic * s.expand) * s.h * s.h) dr n))))
                 (Kernel4.flatten p.Wz) idx j * batchSlice N (s.oc * s.h * s.h) cotPc n j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N (s.oc) s.h s.h pc)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez γ' p.bz2
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.gz k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez p.gz β'
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.bz2 k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)

/-- ⭐⭐ **And it holds** — nine instantiations of the §1 fold at the chain's cotangents. -/
theorem mnv4_convnext_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4ConvNeXtTiedB N s xN cotN vN epsStr p xin dyOut := by
  unfold mnv4ConvNeXtTiedB
  intro qr er dr qc ec pc cotQn cotQc cotEn cotEc cotPc
  exact ⟨fun idx => EnetPoCG.depthwiseWGradB_den xN cotN p.bq xin p.Wq cotQc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.eq_ p.gq p.bq2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.eq_ p.gq p.bq2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.be qr p.We cotEc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ee p.ge p.be2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ee p.ge p.be2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.bz dr p.Wz cotPc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ez p.gz p.bz2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ez p.gz p.bz2 _ _ k⟩


def mnv4FfnTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let ec := batchMap N (flatConv p.We p.be) qr
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotEn := mnv4CotEn N s p xin dyOut
  let cotEc := mnv4CotEc N s p xin dyOut
  let cotPc := mnv4CotPc N s p xin dyOut
  (∀ idx : Fin ((s.ic * s.expand) * s.ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN cotEc)) idx
        = ∑ n : Fin N, ∑ j : Fin ((s.ic * s.expand) * s.h * s.h),
            pdiv (fun v' : Vec ((s.ic * s.expand) * s.ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.be
                      (Tensor3.unflatten (batchSlice N (s.ic * s.h * s.h) qr n))))
                 (Kernel4.flatten p.We) idx j
              * batchSlice N ((s.ic * s.expand) * s.h * s.h) cotEc n j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee γ' p.be2
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.ge k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee p.ge β'
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.be2 k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ idx : Fin (s.oc * (s.ic * s.expand) * 1 * 1),
      den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.oc * s.h * s.h),
            pdiv (fun v' : Vec (s.oc * (s.ic * s.expand) * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.bz
                      (Tensor3.unflatten (batchSlice N ((s.ic * s.expand) * s.h * s.h) dr n))))
                 (Kernel4.flatten p.Wz) idx j * batchSlice N (s.oc * s.h * s.h) cotPc n j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N (s.oc) s.h s.h pc)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez γ' p.bz2
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.gz k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez p.gz β'
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.bz2 k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)

/-- ⭐⭐ **And it holds** — six instantiations of the §1 fold at the chain's cotangents. -/
theorem mnv4_ffn_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4FfnTiedB N s xN cotN vN epsStr p xin dyOut := by
  unfold mnv4FfnTiedB
  intro qr er dr ec pc cotEn cotEc cotPc
  exact ⟨fun idx => ResNet34PoCB.convWGradB_den xN cotN p.be qr p.We cotEc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ee p.ge p.be2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ee p.ge p.be2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.bz dr p.Wz cotPc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ez p.gz p.bz2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ez p.gz p.bz2 _ _ k⟩


-- ════════════════════════════════════════════════════════════════
-- § The PRE-STRIDED block, tied — rows 1, 3 and 11
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **Pre-strided block, tied** — rows 1, 3 and 11. All twelve parameter nodes — four conv/depthwise weights and four
    BatchNorm γ/β pairs — denote the certified batched `Σ_n` gradient at the real forward
    activations and the real backward-chain cotangent driven by `dyOut`.

    ⚠ Each BatchNorm's γ/β reads the cotangent at THAT BatchNorm's output (`CotQn`, `CotEn`,
    `CotDn`, and `dyOut` itself for the project) while its conv reads the one at the conv's output
    (`CotQc`, `CotEc`, `CotDc`, `CotPc`). Off by one and the gradient is silently wrong — the two
    have the same type. ⭐ The project BN's pair reads `dyOut` UNMASKED: the bottleneck is linear.

    ⚠⚠ The leading node is `depthwiseStridedWeightGradB` — `depthwiseStride2Flat`, SYMMETRIC
    padding, reading its input at `2h`. Every other node is the stride-1 profile's at the reduced
    resolution, because the stride is consumed entirely by that first depthwise.

    ⛔ No skip and no `addVB`: `ic ≠ oc` at all three rows, so the block IS its body. -/
def mnv4PreStridedTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (mnv4ExpandLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseStride2Flat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let dc := batchMap N (depthwiseFlat p.Wd p.bd) er
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotQn := mnv4SCotQn N s p xin dyOut
  let cotQc := mnv4SCotQc N s p xin dyOut
  let cotEn := mnv4SCotEn N s p xin dyOut
  let cotEc := mnv4SCotEc N s p xin dyOut
  let cotDn := mnv4SCotDn N s p xin dyOut
  let cotDc := mnv4SCotDc N s p xin dyOut
  let cotPc := mnv4SCotPc N s p xin dyOut
  (∀ idx : Fin (s.ic * s.preDWk * s.preDWk),
      den (SHlo.depthwiseStridedWeightGradB xN p.bq xin p.Wq (.operand cotN cotQc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.ic * s.h * s.h),
            pdiv (fun v' : Vec (s.ic * s.preDWk * s.preDWk) =>
                    depthwiseStride2Flat (Tensor3.unflatten v') p.bq
                      (batchSlice N (s.ic * (2 * s.h) * (2 * s.h)) xin n))
                 (Tensor3.flatten p.Wq) idx j * batchSlice N (s.ic * s.h * s.h) cotQc n j)
  ∧
  (∀ k : Fin (s.ic),
      den (SHlo.bnGammaGradB vN epsStr p.eq_ (reassocB N (s.ic) s.h s.h qc)
            (.operand cotN (reassocB N (s.ic) s.h s.h cotQn))) k
        = ∑ j : Fin ((s.ic) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic) =>
                    bnPerChannelFlat (s.ic) (N * (s.h * s.h)) p.eq_ γ' p.bq2
                      (bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h qc)))
                 p.gq k j * bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h cotQn) j)
  ∧
  (∀ k : Fin (s.ic),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic) s.h s.h cotQn))) k
        = ∑ j : Fin ((s.ic) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic) =>
                    bnPerChannelFlat (s.ic) (N * (s.h * s.h)) p.eq_ p.gq β'
                      (bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h qc)))
                 p.bq2 k j * bnchwFwd N (s.ic) s.h s.h (reassocB N (s.ic) s.h s.h cotQn) j)
  ∧
  (∀ idx : Fin ((s.ic * s.expand) * s.ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.be qr p.We (.operand cotN cotEc)) idx
        = ∑ n : Fin N, ∑ j : Fin ((s.ic * s.expand) * s.h * s.h),
            pdiv (fun v' : Vec ((s.ic * s.expand) * s.ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.be
                      (Tensor3.unflatten (batchSlice N (s.ic * s.h * s.h) qr n))))
                 (Kernel4.flatten p.We) idx j
              * batchSlice N ((s.ic * s.expand) * s.h * s.h) cotEc n j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnGammaGradB vN epsStr p.ee (reassocB N (s.ic * s.expand) s.h s.h ec)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee γ' p.be2
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.ge k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotEn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ee p.ge β'
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h ec)))
                 p.be2 k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotEn) j)
  ∧
  (∀ idx : Fin ((s.ic * s.expand) * s.postDWk * s.postDWk),
      den (SHlo.depthwiseWeightGradB xN p.bd er p.Wd (.operand cotN cotDc)) idx
        = ∑ n : Fin N, ∑ j : Fin ((s.ic * s.expand) * s.h * s.h),
            pdiv (fun v' : Vec ((s.ic * s.expand) * s.postDWk * s.postDWk) =>
                    depthwiseFlat (Tensor3.unflatten v') p.bd
                      (batchSlice N ((s.ic * s.expand) * s.h * s.h) er n))
                 (Tensor3.flatten p.Wd) idx j
              * batchSlice N ((s.ic * s.expand) * s.h * s.h) cotDc n j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnGammaGradB vN epsStr p.ed (reassocB N (s.ic * s.expand) s.h s.h dc)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotDn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ed γ' p.bd2
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h dc)))
                 p.gd k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotDn) j)
  ∧
  (∀ k : Fin (s.ic * s.expand),
      den (SHlo.bnBetaGradB (N := N) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.ic * s.expand) s.h s.h cotDn))) k
        = ∑ j : Fin ((s.ic * s.expand) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.ic * s.expand) =>
                    bnPerChannelFlat (s.ic * s.expand) (N * (s.h * s.h)) p.ed p.gd β'
                      (bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h dc)))
                 p.bd2 k j * bnchwFwd N (s.ic * s.expand) s.h s.h (reassocB N (s.ic * s.expand) s.h s.h cotDn) j)
  ∧
  (∀ idx : Fin (s.oc * (s.ic * s.expand) * 1 * 1),
      den (SHlo.convWeightGradB xN p.bz dr p.Wz (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (s.oc * s.h * s.h),
            pdiv (fun v' : Vec (s.oc * (s.ic * s.expand) * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.bz
                      (Tensor3.unflatten (batchSlice N ((s.ic * s.expand) * s.h * s.h) dr n))))
                 (Kernel4.flatten p.Wz) idx j * batchSlice N (s.oc * s.h * s.h) cotPc n j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnGammaGradB vN epsStr p.ez (reassocB N (s.oc) s.h s.h pc)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun γ' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez γ' p.bz2
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.gz k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)
  ∧
  (∀ k : Fin (s.oc),
      den (SHlo.bnBetaGradB (N := N) (oc := s.oc) (h := s.h) (w := s.h)
            (.operand cotN (reassocB N (s.oc) s.h s.h dyOut))) k
        = ∑ j : Fin ((s.oc) * (N * (s.h * s.h))),
            pdiv (fun β' : Vec (s.oc) =>
                    bnPerChannelFlat (s.oc) (N * (s.h * s.h)) p.ez p.gz β'
                      (bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h pc)))
                 p.bz2 k j * bnchwFwd N (s.oc) s.h s.h (reassocB N (s.oc) s.h s.h dyOut) j)

/-- ⭐⭐ **And it holds** — twelve instantiations (the first at the STRIDED depthwise) of `MobileNetV4FoldB`'s `∀ cot` fold with
    the freedom removed. Nothing here is new mathematics; what is new is that the cotangents are
    the chain's, not free. -/
theorem mnv4_prestrided_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4PreStridedTiedB N s xN cotN vN epsStr p xin dyOut := by
  unfold mnv4PreStridedTiedB
  intro qr er dr qc ec dc pc cotQn cotQc cotEn cotEc cotDn cotDc cotPc
  exact ⟨fun idx => EnetPoCG.depthwiseStridedWGradB_den xN cotN p.bq xin p.Wq cotQc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.eq_ p.gq p.bq2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.eq_ p.gq p.bq2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.be qr p.We cotEc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ee p.ge p.be2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ee p.ge p.be2 _ _ k,
    fun idx => EnetPoCG.depthwiseWGradB_den xN cotN p.bd er p.Wd cotDc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ed p.gd p.bd2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ed p.gd p.bd2 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN p.bz dr p.Wz cotPc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.ez p.gz p.bz2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN p.ez p.gz p.bz2 _ _ k⟩

-- ════════════════════════════════════════════════════════════════
-- § The stem, the fused stage and the head — the three non-UIB stages
-- ════════════════════════════════════════════════════════════════

/-! ⚠ All three are GENERIC IN THEIR WIDTHS, for the reason `MobileNetV4FullB.lean`'s stem-graph
docstring records at length: pinning MNv4's literal resolutions here lets `den` and the
width-indexed `rfl`s actually run, and the kernel gives up. The capstone instantiates. -/

/-- Cotangent at the stem BN's output — the fused stage's `dx`, masked by the stem relu.
    Feeds `%sg` and `%sbt`. -/
noncomputable def mnv4StemCotN (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyStem : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (bnBatchLA N oc h w εs γs βs (batchMap N (flatConvStride2Xla Ws bs) x)) dyStem

/-- Cotangent at the stem CONV's output — through the stem BN's backward. Feeds `%sW`. -/
noncomputable def mnv4StemCotC (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyStem : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w εs γs (batchMap N (flatConvStride2Xla Ws bs) x)
    (mnv4StemCotN N h w Ws bs εs γs βs x dyStem)

/-- **Stem, tied.** Its three nodes at the chain's cotangents.

    ⛔⛔ **And the chain STOPS here.** There is no `convStridedXlaBackBatched` node: no render emits
    a gradient into `%x`, so the artifact's backward ends at this weight gradient. That is why the
    stem sits outside `MobileNetV4FullB.lean`'s `CertLayer` trunk, and it is B0's situation
    exactly. -/
def mnv4StemTiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) : Prop :=
  let sc := batchMap N (flatConvStride2Xla Ws bs) x
  let cotN' := mnv4StemCotN N h w Ws bs εs γs βs x dyStem
  let cotC := mnv4StemCotC N h w Ws bs εs γs βs x dyStem
  (∀ idx : Fin (oc * ic * kH * kW),
      den (SHlo.convStridedXlaWeightGradB xN bs x Ws (.operand cotN cotC)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                    flatConvStride2Xla (Kernel4.unflatten v') bs
                      (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                 (Kernel4.flatten Ws) idx j * batchSlice N (oc * h * w) cotC n j)
  ∧
  (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc h w sc)
            (.operand cotN (reassocB N oc h w cotN'))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εs γ' βs
                      (bnchwFwd N oc h w (reassocB N oc h w sc)))
                 γs k j * bnchwFwd N oc h w (reassocB N oc h w cotN') j)
  ∧
  (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotN'))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εs γs β'
                      (bnchwFwd N oc h w (reassocB N oc h w sc)))
                 βs k j * bnchwFwd N oc h w (reassocB N oc h w cotN') j)

theorem mnv4_stem_tiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) :
    mnv4StemTiedB N h w xN cotN vN epsStr Ws bs εs γs βs x dyStem := by
  unfold mnv4StemTiedB
  intro sc cotN' cotC
  exact ⟨fun idx => EnetPoCG.convStridedXlaWGradB_den xN cotN bs x Ws cotC idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN εs γs βs _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN εs γs βs _ _ k⟩

/-- Cotangent at the fused stage's project CONV output. ⭐ `dyF` reaches the project BN's γ/β
    unmasked — the fused stage ends in a BatchNorm with no activation. Feeds `%f0pW`. -/
noncomputable def mnv4FusedCotPc (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    -- ⭐ `βp` is taken and NOT read: the BatchNorm input-gradient does not depend on the shift,
    -- which `bnInB` records by not taking one. Kept in the signature so every cotangent in this
    -- chain has the same argument list as the tie bundle that consumes it.
    (γp _βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w εp γp
    (batchMap N (flatConv Wp bp) (fusedConvB N (h := h) (w := w) Wc bc εc γc βc xin)) dyF

/-- Cotangent at the fused BN's output — the project conv's input-VJP through **swish**'s
    backward. ⭐ No mask: swish is smooth, which is why this stage carries no kink hypothesis
    anywhere. Feeds `%f0cg` and `%f0cbt`. -/
noncomputable def mnv4FusedCotN (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w))
    (bnBatchLA N mid h w εc γc βc (batchMap N (flatConvStride2 Wc bc) xin))
    (cInB N Wp bp (mnv4FusedCotPc N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF))

/-- Cotangent at the fused CONV's output — through the fused BN's backward. Feeds `%f0cW`. -/
noncomputable def mnv4FusedCotC (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w εc γc (batchMap N (flatConvStride2 Wc bc) xin)
    (mnv4FusedCotN N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF)

/-- The fused stage's input cotangent — the SYMMETRIC strided conv's input-VJP, landing at `2h`. -/
noncomputable def mnv4FusedCotIn (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w))) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  cStridedInB N Wc bc (mnv4FusedCotC N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF)


/-- ⭐ **Fused stage, tied** — its six parameter nodes. ⚠ `%f0cW` is `convStridedWeightGradB`,
    SYMMETRIC padding, where the stem's is the XLA-`SAME` twin: two phases in one net. ⭐ And the
    fused BN's γ/β read the cotangent that came through SWISH's backward, not through a relu mask —
    this is the one stage in MNv4 with no kink anywhere. -/
def mnv4FusedTiedB (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xN cotN vN epsStr : String)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyF : Vec (N * (oc * h * w))) : Prop :=
  let sw := fusedConvB N (h := h) (w := w) Wc bc εc γc βc xin
  let fc := batchMap N (flatConvStride2 Wc bc) xin
  let pc := batchMap N (flatConv Wp bp) sw
  let cotN' := mnv4FusedCotN N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  let cotC := mnv4FusedCotC N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  let cotPc := mnv4FusedCotPc N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  (∀ idx : Fin (mid * ic * kH * kW),
      den (SHlo.convStridedWeightGradB xN bc xin Wc (.operand cotN cotC)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * ic * kH * kW) =>
                    flatConvStride2 (Kernel4.unflatten v') bc
                      (batchSlice N (ic * (2 * h) * (2 * w)) xin n))
                 (Kernel4.flatten Wc) idx j * batchSlice N (mid * h * w) cotC n j)
  ∧
  (∀ k : Fin mid,
      den (SHlo.bnGammaGradB vN epsStr εc (reassocB N mid h w fc)
            (.operand cotN (reassocB N mid h w cotN'))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun γ' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) εc γ' βc
                      (bnchwFwd N mid h w (reassocB N mid h w fc)))
                 γc k j * bnchwFwd N mid h w (reassocB N mid h w cotN') j)
  ∧
  (∀ k : Fin mid,
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N mid h w cotN'))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun β' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) εc γc β'
                      (bnchwFwd N mid h w (reassocB N mid h w fc)))
                 βc k j * bnchwFwd N mid h w (reassocB N mid h w cotN') j)
  ∧
  (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN bp sw Wp (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') bp
                      (Tensor3.unflatten (batchSlice N (mid * h * w) sw n))))
                 (Kernel4.flatten Wp) idx j * batchSlice N (oc * h * w) cotPc n j)
  ∧
  (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr εp (reassocB N oc h w pc)
            (.operand cotN (reassocB N oc h w dyF))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εp γ' βp
                      (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 γp k j * bnchwFwd N oc h w (reassocB N oc h w dyF) j)
  ∧
  (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w dyF))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εp γp β'
                      (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 βp k j * bnchwFwd N oc h w (reassocB N oc h w dyF) j)

theorem mnv4_fused_tiedB (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xN cotN vN epsStr : String)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyF : Vec (N * (oc * h * w))) :
    mnv4FusedTiedB N h w Wc bc εc γc βc Wp bp εp γp βp xN cotN vN epsStr xin dyF := by
  unfold mnv4FusedTiedB
  intro sw fc pc cotN' cotC cotPc
  exact ⟨fun idx => ResNet34PoCB.convStridedWGradB_den xN cotN bc xin Wc cotC idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN εc γc βc _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN εc γc βc _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN bp sw Wp cotPc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN εp γp βp _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN εp γp βp _ _ k⟩


/-! ⭐ The head's GAP-and-dense tail is ResNet-34's, reused: `r34HeadCotBlk` is its certified
input cotangent and `r34HeadTiedB` its two parameter nodes. What is MNv4's own is the pair of
1×1 conv-BN-relu stages in front of it — Conv-M's head has TWO convs where `mnv4Head` models
one. -/

/-- Cotangent at the SECOND head BN's output — the GAP/dense tail's input cotangent, masked by
    that stage's relu. Feeds `%hg` and `%hbt`. -/
noncomputable def mnv4HeadCotHn (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (bnBatchLA N oc h w ε2 γ2 β2
      (batchMap N (flatConv W2 b2) (cbReluB N (h := h) (w := w) W1 b1 ε1 γ1 β1 xin)))
    (r34HeadCotBlk N h w Wd bd
      (cbReluB N (h := h) (w := w) W2 b2 ε2 γ2 β2
        (cbReluB N (h := h) (w := w) W1 b1 ε1 γ1 β1 xin)) g)

/-- Cotangent at the second head CONV's output. Feeds `%hW`. -/
noncomputable def mnv4HeadCotHc (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w ε2 γ2
    (batchMap N (flatConv W2 b2) (cbReluB N (h := h) (w := w) W1 b1 ε1 γ1 β1 xin))
    (mnv4HeadCotHn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g)

/-- Cotangent at the FIRST head BN's output. Feeds `%h1g` and `%h1bt`. -/
noncomputable def mnv4HeadCotH1n (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w ε1 γ1 β1 (batchMap N (flatConv W1 b1) xin))
    (cInB N W2 b2 (mnv4HeadCotHc N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g))

/-- Cotangent at the first head CONV's output. Feeds `%h1W`. -/
noncomputable def mnv4HeadCotH1c (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w ε1 γ1 (batchMap N (flatConv W1 b1) xin)
    (mnv4HeadCotH1n N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g)

/-- **The head's input cotangent** — what block 21 receives as its `dyOut`. -/
noncomputable def mnv4HeadCotIn (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (c * h * w)) :=
  cInB N W1 b1 (mnv4HeadCotH1c N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g)

/-- ⭐ **Head, tied** — all EIGHT nodes: two conv weights, two BatchNorm γ/β pairs, and the
    classifier's weight and bias. ⭐ The last two are `r34HeadTiedB`, reused verbatim: MNv4's
    GAP-and-dense tail IS ResNet-34's at a different width. -/
def mnv4HeadTiedB (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Prop :=
  let r1 := cbReluB N (h := h) (w := w) W1 b1 ε1 γ1 β1 xin
  let r2 := cbReluB N (h := h) (w := w) W2 b2 ε2 γ2 β2 r1
  let c1 := batchMap N (flatConv W1 b1) xin
  let c2 := batchMap N (flatConv W2 b2) r1
  let cotH1n := mnv4HeadCotH1n N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  let cotH1c := mnv4HeadCotH1c N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  let cotHn := mnv4HeadCotHn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  let cotHc := mnv4HeadCotHc N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin g
  (∀ idx : Fin (mid * c * 1 * 1),
      den (SHlo.convWeightGradB xN b1 xin W1 (.operand cotN cotH1c)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * c * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') b1
                      (Tensor3.unflatten (batchSlice N (c * h * w) xin n))))
                 (Kernel4.flatten W1) idx j * batchSlice N (mid * h * w) cotH1c n j)
  ∧
  (∀ k : Fin mid,
      den (SHlo.bnGammaGradB vN epsStr ε1 (reassocB N mid h w c1)
            (.operand cotN (reassocB N mid h w cotH1n))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun γ' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) ε1 γ' β1
                      (bnchwFwd N mid h w (reassocB N mid h w c1)))
                 γ1 k j * bnchwFwd N mid h w (reassocB N mid h w cotH1n) j)
  ∧
  (∀ k : Fin mid,
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N mid h w cotH1n))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun β' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) ε1 γ1 β'
                      (bnchwFwd N mid h w (reassocB N mid h w c1)))
                 β1 k j * bnchwFwd N mid h w (reassocB N mid h w cotH1n) j)
  ∧
  (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN b2 r1 W2 (.operand cotN cotHc)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') b2
                      (Tensor3.unflatten (batchSlice N (mid * h * w) r1 n))))
                 (Kernel4.flatten W2) idx j * batchSlice N (oc * h * w) cotHc n j)
  ∧
  (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr ε2 (reassocB N oc h w c2)
            (.operand cotN (reassocB N oc h w cotHn))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) ε2 γ' β2
                      (bnchwFwd N oc h w (reassocB N oc h w c2)))
                 γ2 k j * bnchwFwd N oc h w (reassocB N oc h w cotHn) j)
  ∧
  (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotHn))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) ε2 γ2 β'
                      (bnchwFwd N oc h w (reassocB N oc h w c2)))
                 β2 k j * bnchwFwd N oc h w (reassocB N oc h w cotHn) j)
  ∧ ResNet34TieB.r34HeadTiedB N h w xN cotN Wd bd r2 g

theorem mnv4_head_tiedB (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) :
    mnv4HeadTiedB N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xN cotN vN epsStr xin g := by
  unfold mnv4HeadTiedB
  intro r1 r2 c1 c2 cotH1n cotH1c cotHn cotHc
  exact ⟨fun idx => ResNet34PoCB.convWGradB_den xN cotN b1 xin W1 cotH1c idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN ε1 γ1 β1 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN ε1 γ1 β1 _ _ k,
    fun idx => ResNet34PoCB.convWGradB_den xN cotN b2 r1 W2 cotHc idx,
    fun k => ResNet34PoCB.bnGammaGradB_den vN epsStr cotN ε2 γ2 β2 _ _ k,
    fun k => ResNet34PoCB.bnBetaGradB_den cotN ε2 γ2 β2 _ _ k,
    ResNet34TieB.r34_head_tiedB N h w xN cotN Wd bd r2 g⟩


-- ════════════════════════════════════════════════════════════════
-- § The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- The per-BLOCK forward prefixes: `mnv4Blk0` is the fused stage's output — block 1's input —
    and `mnv4Blk{k}` is the activation entering block `k+1`.

    ⚠ `MobileNetV4FullB.lean`'s `mnv4Pre0 … mnv4Pre6` are the RESOLUTION-GROUP prefixes, which is
    the granularity T1 and T2 need; the tie needs one per BLOCK, so these 22 name the finer chain.
    `mnv4Blk0` is definitionally `mnv4Pre1`. -/
noncomputable def mnv4Blk0 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (48 * 56 * 56)) := mnv4Pre1 N w x

noncomputable def mnv4Blk1 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (80 * 28 * 28)) :=
  (mnv4PreStridedBodyOfRow N mnv4Row1 w.b1).fwd (mnv4Blk0 N w x)

noncomputable def mnv4Blk2 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (80 * 28 * 28)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2)).fwd (mnv4Blk1 N w x)

noncomputable def mnv4Blk3 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (mnv4PreStridedBodyOfRow N mnv4Row3 w.b3).fwd (mnv4Blk2 N w x)

noncomputable def mnv4Blk4 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row4 w.b4)).fwd (mnv4Blk3 N w x)

noncomputable def mnv4Blk5 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row5 w.b5)).fwd (mnv4Blk4 N w x)

noncomputable def mnv4Blk6 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row6 w.b6)).fwd (mnv4Blk5 N w x)

noncomputable def mnv4Blk7 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row7 w.b7)).fwd (mnv4Blk6 N w x)

noncomputable def mnv4Blk8 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row8 w.b8)).fwd (mnv4Blk7 N w x)

noncomputable def mnv4Blk9 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row9 w.b9)).fwd (mnv4Blk8 N w x)

noncomputable def mnv4Blk10 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row10 w.b10)).fwd (mnv4Blk9 N w x)

noncomputable def mnv4Blk11 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (mnv4PreStridedBodyOfRow N mnv4Row11 w.b11).fwd (mnv4Blk10 N w x)

noncomputable def mnv4Blk12 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row12 w.b12)).fwd (mnv4Blk11 N w x)

noncomputable def mnv4Blk13 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row13 w.b13)).fwd (mnv4Blk12 N w x)

noncomputable def mnv4Blk14 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row14 w.b14)).fwd (mnv4Blk13 N w x)

noncomputable def mnv4Blk15 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row15 w.b15)).fwd (mnv4Blk14 N w x)

noncomputable def mnv4Blk16 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row16 w.b16)).fwd (mnv4Blk15 N w x)

noncomputable def mnv4Blk17 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row17 w.b17)).fwd (mnv4Blk16 N w x)

noncomputable def mnv4Blk18 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row18 w.b18)).fwd (mnv4Blk17 N w x)

noncomputable def mnv4Blk19 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row19 w.b19)).fwd (mnv4Blk18 N w x)

noncomputable def mnv4Blk20 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row20 w.b20)).fwd (mnv4Blk19 N w x)

noncomputable def mnv4Blk21 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row21 w.b21)).fwd (mnv4Blk20 N w x)

set_option maxHeartbeats 1600000 in
/-- ⭐⭐ **The whole batch-BN MobileNetV4-Conv-M train step, tied.** Threading the net's own forward
    prefixes as the block inputs and an arbitrary loss cotangent `g` down through the certified
    head backward, the 21 certified UIB block backwards and the fused stage, every parameter
    GRADIENT node of the net — stem 3, fused 6, thirteen ExtraDW-profile blocks × 12, four
    ConvNeXt-like × 9, four FFN × 6, head 8 — denotes the certified batched `Σ_n` gradient. That is
    **233**, the render's own census and `mnv4_fwd.mlir`'s signature minus `%x`. No free activation
    and no symbolic cotangent below the loss.

    ⭐⭐ **`g` IS A BINDER.** The loss chain is not part of this statement; `mnv4_lossCot_is_smoothedCE_grad`
    instantiates it at the label-smoothed softmax cotangent the artifacts actually emit.

    ⭐ **No smoothness hypothesis and no `0 < ε`**, and `N` and `nCls` are both binders. The folds
    are `∀ cot` statements at explicitly constructed cotangents; the kink and positivity conditions
    live one tier down, in the `CertLayer`s whose `.ok` `MobileNetV4FullBVJP.lean` binds. ⚠ MNv4
    ships a single resolution, so unlike ResNet-50 there is no `q`.

    ⛔ **One replica.** Under `mnv4in_adamdp64*` every node named here feeds `allReduceMeanF`
    (`DataParallelNode.lean`, §4d), and the AdamW tail sits downstream of all of them. -/
theorem mnv4_net_tiedB (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) (g : Vec (N * nCls)) :
  let dy21 := mnv4HeadCotIn N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
               w.Wd w.bd (mnv4Blk21 N w x) g
  let dy20 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row21 w.b21 (mnv4Blk20 N w x) dy21) dy21
  let dy19 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row20 w.b20 (mnv4Blk19 N w x) dy20) dy20
  let dy18 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row19 w.b19 (mnv4Blk18 N w x) dy19) dy19
  let dy17 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row18 w.b18 (mnv4Blk17 N w x) dy18) dy18
  let dy16 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row17 w.b17 (mnv4Blk16 N w x) dy17) dy17
  let dy15 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row16 w.b16 (mnv4Blk15 N w x) dy16) dy16
  let dy14 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row15 w.b15 (mnv4Blk14 N w x) dy15) dy15
  let dy13 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row14 w.b14 (mnv4Blk13 N w x) dy14) dy14
  let dy12 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row13 w.b13 (mnv4Blk12 N w x) dy13) dy13
  let dy11 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row12 w.b12 (mnv4Blk11 N w x) dy12) dy12
  let dy10 := mnv4SBodyCotIn N mnv4Row11 w.b11 (mnv4Blk10 N w x) dy11
  let dy9 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row10 w.b10 (mnv4Blk9 N w x) dy10) dy10
  let dy8 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row9 w.b9 (mnv4Blk8 N w x) dy9) dy9
  let dy7 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row8 w.b8 (mnv4Blk7 N w x) dy8) dy8
  let dy6 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row7 w.b7 (mnv4Blk6 N w x) dy7) dy7
  let dy5 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row6 w.b6 (mnv4Blk5 N w x) dy6) dy6
  let dy4 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row5 w.b5 (mnv4Blk4 N w x) dy5) dy5
  let dy3 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row4 w.b4 (mnv4Blk3 N w x) dy4) dy4
  let dy2 := mnv4SBodyCotIn N mnv4Row3 w.b3 (mnv4Blk2 N w x) dy3
  let dy1 := mnv4SkipCotIn (mnv4BodyCotIn N mnv4Row2 w.b2 (mnv4Blk1 N w x) dy2) dy2
  let dy0 := mnv4SBodyCotIn N mnv4Row1 w.b1 (mnv4Blk0 N w x) dy1
  let dyStem := mnv4FusedCotIn N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                 w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt (mnv4Pre0 N w x) dy0
  mnv4StemTiedB N 112 112 xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt x dyStem
  ∧ mnv4FusedTiedB N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.f0pg
      w.f0pbt xN cotN vN epsStr (mnv4Pre0 N w x) dy0
  ∧ mnv4PreStridedTiedB N mnv4Row1 xN cotN vN epsStr w.b1 (mnv4Blk0 N w x) dy1
  ∧ mnv4ExtraDWTiedB N mnv4Row2 xN cotN vN epsStr w.b2 (mnv4Blk1 N w x) dy2
  ∧ mnv4PreStridedTiedB N mnv4Row3 xN cotN vN epsStr w.b3 (mnv4Blk2 N w x) dy3
  ∧ mnv4ExtraDWTiedB N mnv4Row4 xN cotN vN epsStr w.b4 (mnv4Blk3 N w x) dy4
  ∧ mnv4ExtraDWTiedB N mnv4Row5 xN cotN vN epsStr w.b5 (mnv4Blk4 N w x) dy5
  ∧ mnv4ExtraDWTiedB N mnv4Row6 xN cotN vN epsStr w.b6 (mnv4Blk5 N w x) dy6
  ∧ mnv4ExtraDWTiedB N mnv4Row7 xN cotN vN epsStr w.b7 (mnv4Blk6 N w x) dy7
  ∧ mnv4ConvNeXtTiedB N mnv4Row8 xN cotN vN epsStr w.b8 (mnv4Blk7 N w x) dy8
  ∧ mnv4FfnTiedB N mnv4Row9 xN cotN vN epsStr w.b9 (mnv4Blk8 N w x) dy9
  ∧ mnv4ConvNeXtTiedB N mnv4Row10 xN cotN vN epsStr w.b10 (mnv4Blk9 N w x) dy10
  ∧ mnv4PreStridedTiedB N mnv4Row11 xN cotN vN epsStr w.b11 (mnv4Blk10 N w x) dy11
  ∧ mnv4ExtraDWTiedB N mnv4Row12 xN cotN vN epsStr w.b12 (mnv4Blk11 N w x) dy12
  ∧ mnv4ExtraDWTiedB N mnv4Row13 xN cotN vN epsStr w.b13 (mnv4Blk12 N w x) dy13
  ∧ mnv4ExtraDWTiedB N mnv4Row14 xN cotN vN epsStr w.b14 (mnv4Blk13 N w x) dy14
  ∧ mnv4FfnTiedB N mnv4Row15 xN cotN vN epsStr w.b15 (mnv4Blk14 N w x) dy15
  ∧ mnv4ConvNeXtTiedB N mnv4Row16 xN cotN vN epsStr w.b16 (mnv4Blk15 N w x) dy16
  ∧ mnv4ExtraDWTiedB N mnv4Row17 xN cotN vN epsStr w.b17 (mnv4Blk16 N w x) dy17
  ∧ mnv4ExtraDWTiedB N mnv4Row18 xN cotN vN epsStr w.b18 (mnv4Blk17 N w x) dy18
  ∧ mnv4FfnTiedB N mnv4Row19 xN cotN vN epsStr w.b19 (mnv4Blk18 N w x) dy19
  ∧ mnv4FfnTiedB N mnv4Row20 xN cotN vN epsStr w.b20 (mnv4Blk19 N w x) dy20
  ∧ mnv4ConvNeXtTiedB N mnv4Row21 xN cotN vN epsStr w.b21 (mnv4Blk20 N w x) dy21
  ∧ mnv4HeadTiedB N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
      w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 N w x) g := by
  intro dy21 dy20 dy19 dy18 dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3
    dy2 dy1 dy0 dyStem
  exact ⟨mnv4_stem_tiedB N 112 112 xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt x dyStem,
    mnv4_fused_tiedB N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.f0pg
      w.f0pbt xN cotN vN epsStr (mnv4Pre0 N w x) dy0,
    mnv4_prestrided_tiedB N mnv4Row1 xN cotN vN epsStr w.b1 (mnv4Blk0 N w x) dy1,
    mnv4_extradw_tiedB N mnv4Row2 xN cotN vN epsStr w.b2 (mnv4Blk1 N w x) dy2,
    mnv4_prestrided_tiedB N mnv4Row3 xN cotN vN epsStr w.b3 (mnv4Blk2 N w x) dy3,
    mnv4_extradw_tiedB N mnv4Row4 xN cotN vN epsStr w.b4 (mnv4Blk3 N w x) dy4,
    mnv4_extradw_tiedB N mnv4Row5 xN cotN vN epsStr w.b5 (mnv4Blk4 N w x) dy5,
    mnv4_extradw_tiedB N mnv4Row6 xN cotN vN epsStr w.b6 (mnv4Blk5 N w x) dy6,
    mnv4_extradw_tiedB N mnv4Row7 xN cotN vN epsStr w.b7 (mnv4Blk6 N w x) dy7,
    mnv4_convnext_tiedB N mnv4Row8 xN cotN vN epsStr w.b8 (mnv4Blk7 N w x) dy8,
    mnv4_ffn_tiedB N mnv4Row9 xN cotN vN epsStr w.b9 (mnv4Blk8 N w x) dy9,
    mnv4_convnext_tiedB N mnv4Row10 xN cotN vN epsStr w.b10 (mnv4Blk9 N w x) dy10,
    mnv4_prestrided_tiedB N mnv4Row11 xN cotN vN epsStr w.b11 (mnv4Blk10 N w x) dy11,
    mnv4_extradw_tiedB N mnv4Row12 xN cotN vN epsStr w.b12 (mnv4Blk11 N w x) dy12,
    mnv4_extradw_tiedB N mnv4Row13 xN cotN vN epsStr w.b13 (mnv4Blk12 N w x) dy13,
    mnv4_extradw_tiedB N mnv4Row14 xN cotN vN epsStr w.b14 (mnv4Blk13 N w x) dy14,
    mnv4_ffn_tiedB N mnv4Row15 xN cotN vN epsStr w.b15 (mnv4Blk14 N w x) dy15,
    mnv4_convnext_tiedB N mnv4Row16 xN cotN vN epsStr w.b16 (mnv4Blk15 N w x) dy16,
    mnv4_extradw_tiedB N mnv4Row17 xN cotN vN epsStr w.b17 (mnv4Blk16 N w x) dy17,
    mnv4_extradw_tiedB N mnv4Row18 xN cotN vN epsStr w.b18 (mnv4Blk17 N w x) dy18,
    mnv4_ffn_tiedB N mnv4Row19 xN cotN vN epsStr w.b19 (mnv4Blk18 N w x) dy19,
    mnv4_ffn_tiedB N mnv4Row20 xN cotN vN epsStr w.b20 (mnv4Blk19 N w x) dy20,
    mnv4_convnext_tiedB N mnv4Row21 xN cotN vN epsStr w.b21 (mnv4Blk20 N w x) dy21,
    mnv4_head_tiedB N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
      w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 N w x) g⟩


/-- ⭐ **And the loss cotangent `g` is instantiated: MNv4's is the label-smoothed softmax chain.**
    Row by row, the six-op chain `MobileNetV4RenderB` emits (`softmaxRow → subB → scaleB → addVB →
    shiftB → divConstB`, α = 0.1) is `(1/B)·∂/∂logits` of soft-target cross-entropy against the
    SMOOTHED target, at that example's real logits.

    ⚠ `softmaxRow` at `m := 1` — the `rowB`/`unrowB` spelling ResNet-34 and ResNet-50 use, NOT
    ConvNeXt's and ViT's `expe → softmaxDiv` at the plain `N·K` width. Read off the render's own
    lines rather than assumed: those two take different lemmas (`smoothedLossCotGraph` here,
    `smoothedLossCotGraphDiv` there) and nothing in the types tells them apart.

    ⭐ MNv4 ships ONE loss — there is no BCE twin to state, where ResNet-50 needed both. The only
    hypothesis is that the example's target sums to 1: a one-hot, or mixup's convex combination. -/
theorem mnv4_lossCot_is_smoothedCE_grad (N : Nat) {nCls : Nat} (hK : 0 < nCls)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) (t : Vec (N * (1 * nCls)))
    (n : Fin N) (j : Fin nCls)
    (ht : ∑ k : Fin nCls, Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1) k = 1) :
    den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
          (rowB N nCls (mobilenetv4ForwardB_full N w x)) t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec nCls => fun _ : Fin 1 =>
            softCE nCls (smoothTarget nCls α
              (Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1))) z')
          (Mat.unflatten (batchSlice N (1 * nCls)
            (rowB N nCls (mobilenetv4ForwardB_full N w x)) n) (0 : Fin 1)) j 0) / B :=
  smoothedLossCotGraph_row N nCls hK α B aStr negAK bStr logN ohN _ t n j ht

end Proofs.Mnv4TieB
