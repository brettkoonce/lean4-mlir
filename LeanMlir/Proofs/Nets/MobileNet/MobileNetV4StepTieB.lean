module

public import LeanMlir.Proofs.Foundation.GradNodesBAt
public import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBDrop
public import LeanMlir.Proofs.Foundation.DropSites
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The train-step tie for MobileNetV4-Conv-M — every gradient node at its chain cotangent

Every parameter gradient node MNv4's batched train step emits denotes the certified gradient
**for an arbitrary cotangent**, by a leaf lemma it shares with ResNet-34 or EfficientNet-B0 (the
table below). This file removes that freedom: each node is stated at the cotangent the render's own
backward chain delivers, driven by a loss cotangent `g` at the logits.

No accuracy is quoted for this net. The statements are about timm's `mobilenetv4_conv_medium`;
the gates that pin the artifacts to it are `scripts/parity/mnv4_timm_parity.py`,
`scripts/parity/mnv4_forward_tie.py` and `scripts/parity/grad_tie.py --net mnv4`.

## The UIB bottleneck is linear, and that makes MNv4's chain shorter than ResNet's

`uibFwdSkipB` emits `addVB (project-BN out) (block input)` with **no activation after the add** and
none after the project's BatchNorm. So the block-output cotangent `dyOut` reaches the project BN's
γ and β *directly* — where ResNet-50's `r50IdCotA` must first pass through the post-residual relu's
mask, and where its skip branch carries the masked cotangent. Here the skip fan-in is
`addVB (body dx) dyOut`, unmasked.

**And every block's `*CotIn_eq_vjp` is its block layer's `.faithful`, not a new derivation.** The
UIB bodies, the fused stage and the head are `CertLayer`s, so `den (graph x e) = vjp.backward
(den e)` is already a theorem one tier down — the fact the r34, mnv2 and R50 ties re-derive per
block. `mnv4BodyCotIn_eq_vjp`, `mnv4SBodyCotIn_eq_vjp`, `mnv4SkipCotIn_eq_vjp`,
`mnv4FusedCotIn_eq_vjp` and `mnv4HeadCotIn_eq_vjp` state it for each constructed cotangent the
capstone threads: each is the layer's `.faithful` plus a definitional match. The stem has none:
no render emits a gradient into `%x`.

## No new fp32 op-kind lemmas — MNv4's nine kinds are ResNet-34's and EfficientNet-B0's

| op kind | sites | certificate |
|---|---|---|
| `bnGammaGradB` / `bnBetaGradB` | 77 BN layers | `GradNodeB.bnGammaGradB_den` / `bnBetaGradB_den` |
| `convWeightGradB` | expands, projects, both head convs, the fused project | `GradNodeB.convWGradB_den` |
| `convStridedWeightGradB` (SYMMETRIC) | the stem's and the fused stage's 3×3/s2 | `GradNodeB.convStridedWGradB_den` |
| `depthwiseWeightGradB` | every stride-1 depthwise | `GradNodeB.depthwiseWGradB_den` |
| `depthwiseStridedWeightGradB` | rows 1, 3, 11's post-DW (timm's `dw_mid`) | `GradNodeB.depthwiseStridedWGradB_den` |
| `denseWeightGradB` / `denseBiasGradB` | the classifier | `GradNodeB.denseWGradB_den` / `denseBGradB_den` |

Every strided site pads symmetrically (timm), so the stem and the fused stage share one
certificate.

**MNv4 emits no conv bias gradient at all.** `MobileNetV4RenderB` has no `convBias` flag — every
bias is folded into its BatchNorm and bound to `%zb{c}` — so `convBiasGradB` and its strided peers
are never emitted and there is nothing to state. Same situation as ResNet-50.

**Precision is a flag on the statement.** The bf16 ImageNet artifacts
(`mnv4in_emaacc{,dp}8x128wxdowd005bf16`, `mnv4in_acc{,dp}8x128wxdropdowd01bf16`) emit the bf16
conv and depthwise weight-gradient kinds (`convWeightGradBBf16`, `depthwiseWeightGradBBf16`, …;
`Foundation.Bf16GradNodes`). Every conv and depthwise weight node below is stated on the renderers'
switch — `ConvWTiedBAt bf16` / `ConvStridedWTiedBAt bf16` / `DepthwiseWTiedBAt bf16` /
`DepthwiseStridedWTiedBAt bf16` (`Foundation.GradNodesBAt`): `bf16 := false` is the f32 node,
`bf16 := true` the bf16 one — and the capstone `mnv4_net_tiedB` takes `bf16`, so it reaches the
bf16 artifacts' gradient nodes read over ℝ exactly as it reads the f32 ones (`Bf16Erasure`: at the
identity rounding the bf16 kind denotes what its f32 peer does). BatchNorm's and the classifier's
nodes carry no flag because no render switches them.

## Everything here is generic in the row

Every definition and theorem below takes a `UibSpec` binder `s` and reads its widths off it, so
`s.ic`, `s.h` and `s.ic * s.expand` are variables. That is not a convenience: at the literal
sides 224/112/56/28/14/7, letting `den` and width-indexed `rfl`s run ends in kernel timeouts
(`MobileNetV4FullB.lean`). Stated at a row binder they stay stuck; the capstone then
instantiates at the 21 concrete rows — at the final side `f` (`mnv4Row{k} f`), so the one
statement covers the artifacts at 224×224 (`f = 7`) and the paper tier's `mnv4in256` ones (`f = 8`).

## The chain, node for node from `uibBackSkipGradB`

| cotangent | at | feeds |
|---|---|---|
| `dyOut` | the block output — and the project BN's output, the add being linear | `%u{p}pg`, `%u{p}pbt` |
| `CotPc` | project conv's output (`bnBatchBack`) | `%u{p}pW` |
| `CotDn` | post-DW BN's output (`convBackBatched`, then the post-DW relu's `selectPos` mask) | `%u{p}dg`, `%u{p}dbt` |
| `CotDc` | post-DW conv's output | `%u{p}dW` |
| `CotEn` | expand BN's output (`depthwiseBackBatched`, then the expand relu's mask) | `%u{p}eg`, `%u{p}ebt` |
| `CotEc` | expand conv's output | `%u{p}eW` |
| `CotQn` | pre-DW BN's output (`convBackBatched`; no mask — the pre-DW is BN only) | `%u{p}qg`, `%u{p}qbt` |
| `CotQc` | pre-DW conv's output | `%u{p}qW` |
| `CotIn` | the block input — `addVB (depthwiseBackBatched dQc) dyOut` | the previous block |

A cotangent one step off is a silently wrong gradient, not a type error: `%u{p}eg` reads the
cotangent at the expand BN's OUTPUT and `%u{p}eW` the one at the expand CONV's output, and both
have the same type.

**Data parallel.** In `mnv4in_adamdp64*` every node named here feeds `allReduceMeanF` and
BatchNorm is synchronised across replicas, so a replica's gradient node is not this file's node at
its own `N`. This file describes a data-parallel artifact only at `N := R·N`: the per-replica
statement is `MobileNetV4SyncTieB.mnv4_net_syncTiedB`, whose right-hand side is `mnv4_net_tiedB`'s
node at the global batch.
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.BackLinks

namespace Proofs.Mnv4TieB

open scoped BigOperators
open Proofs.GradNodeB (bnPairTiedB_holds convStridedWTiedBAt_holds convWTiedBAt_holds
  denseBTiedB_holds denseWTiedB_holds depthwiseStridedWTiedBAt_holds depthwiseWTiedBAt_holds)

-- ════════════════════════════════════════════════════════════════
-- § The ExtraDW block's cotangent chain — 13 of Conv-M's 21 rows
-- ════════════════════════════════════════════════════════════════

/-- Cotangent at the project CONV's output — `dyOut` through the project BN's backward.
    `dyOut` itself is the cotangent at the project BN's output: the bottleneck is linear, so
    nothing masks it. Feeds `%u{p}pW`. -/
noncomputable def mnv4CotPc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.oc * s.h * s.h)) :=
  bnInB N s.oc s.h s.h p.ez p.gz
    (batchMap N (flatConv p.Wz p.bz)
      ((mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd
        ((cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
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
        ((cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    (cInB N p.Wz p.bz (mnv4CotPc N s p xin dyOut))

/-- Cotangent at the post-DW CONV's output — through the post-DW BN's backward. Feeds `%u{p}dW`. -/
noncomputable def mnv4CotDc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnInB N (s.ic * s.expand) s.h s.h p.ed p.gd
    (batchMap N (depthwiseFlat p.Wd p.bd)
      ((cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
        ((mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (mnv4CotDn N s p xin dyOut)

/-- Cotangent at the expand BN's output — masked by the expand relu. Feeds `%u{p}eg`/`%u{p}ebt`.

    **This is where the chain dispatches on the table**, exactly as `mnv4PostDWSlot` does and
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

/-- Cotangent at the pre-DW BN's output — the expand conv's input-VJP, UNMASKED: the pre-DW is
    BN only (timm's `dw_start`). Feeds `%u{p}qg` and `%u{p}qbt`. -/
noncomputable def mnv4CotQn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  cInB N p.We p.be (mnv4CotEc N s p xin dyOut)

/-- Cotangent at the pre-DW CONV's output — through the pre-DW BN's backward. Feeds `%u{p}qW`. -/
noncomputable def mnv4CotQc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  bnInB N s.ic s.h s.h p.eq_ p.gq
    (batchMap N (depthwiseFlat p.Wq p.bq) xin)
    (mnv4CotQn N s p xin dyOut)


-- ════════════════════════════════════════════════════════════════
-- § The STRIDED block's chain — rows 1, 3 and 11, and no skip
-- ════════════════════════════════════════════════════════════════

/-! **A near-copy of the stride-1 chain, and it has to be.** The post-DW carries the stride
(timm's `dw_mid`), so the block input, the BN-only pre-DW and the expand sit at `2h` and the post-DW
and the project at `h` — a change that runs through every type in the chain.

**And there is no skip**: all three stride-2 rows change channels (`ic ≠ oc`), so the block IS
the body and there is no `addVB` fan-in. -/

/-- Cotangent at the project CONV's output. `dyOut` itself is the cotangent at the project BN's
    output: the bottleneck is linear. Feeds `%u{p}pW`. -/
noncomputable def mnv4SCotPc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.oc * s.h * s.h)) :=
  bnInB N s.oc s.h s.h p.ez p.gz
    (batchMap N (flatConv p.Wz p.bz)
      ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd
        ((cbReluLayer (h := 2 * s.h) (w := 2 * s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    dyOut

/-- Cotangent at the STRIDED post-DW BN's output — the project conv's input-VJP, masked by the
    post-DW relu. Feeds `%u{p}dg` and `%u{p}dbt`. -/
noncomputable def mnv4SCotDn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (bnBatchLA N (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
      (batchMap N (depthwiseStride2Flat p.Wd p.bd)
        ((cbReluLayer (h := 2 * s.h) (w := 2 * s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
            xin))))
    (cInB N p.Wz p.bz (mnv4SCotPc N s p xin dyOut))

/-- Cotangent at the STRIDED post-DW CONV's output — through its BN's backward. Feeds `%u{p}dW`. -/
noncomputable def mnv4SCotDc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnInB N (s.ic * s.expand) s.h s.h p.ed p.gd
    (batchMap N (depthwiseStride2Flat p.Wd p.bd)
      ((cbReluLayer (h := 2 * s.h) (w := 2 * s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd
        ((mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (mnv4SCotDn N s p xin dyOut)

/-- Cotangent at the expand BN's output, at `2h` — the strided post-DW's input-VJP, masked by the
    expand relu. Feeds `%u{p}eg`/`%u{p}ebt`. -/
noncomputable def mnv4SCotEn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * (2 * s.h) * (2 * s.h))) :=
  reluMaskB (N * (s.ic * s.expand * (2 * s.h) * (2 * s.h)))
    (bnBatchLA N (s.ic * s.expand) (2 * s.h) (2 * s.h) p.ee p.ge p.be2
      (batchMap N (flatConv p.We p.be)
        ((mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
          xin)))
    (dStridedInB N p.Wd p.bd (mnv4SCotDc N s p xin dyOut))

/-- Cotangent at the expand CONV's output, at `2h`. Feeds `%u{p}eW`. -/
noncomputable def mnv4SCotEc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.expand * (2 * s.h) * (2 * s.h))) :=
  bnInB N (s.ic * s.expand) (2 * s.h) (2 * s.h) p.ee p.ge
    (batchMap N (flatConv p.We p.be)
      ((mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd
        xin))
    (mnv4SCotEn N s p xin dyOut)

/-- Cotangent at the pre-DW BN's output, at `2h` — the expand conv's input-VJP, unmasked (the
    pre-DW is BN only). Feeds `%u{p}qg` and `%u{p}qbt`. -/
noncomputable def mnv4SCotQn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * (2 * s.h) * (2 * s.h))) :=
  cInB N p.We p.be (mnv4SCotEc N s p xin dyOut)

/-- Cotangent at the pre-DW CONV's output, at `2h`. Feeds `%u{p}qW`. -/
noncomputable def mnv4SCotQc (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * (2 * s.h) * (2 * s.h))) :=
  bnInB N s.ic (2 * s.h) (2 * s.h) p.eq_ p.gq
    (batchMap N (depthwiseFlat p.Wq p.bq) xin)
    (mnv4SCotQn N s p xin dyOut)

/-- **The strided block's input cotangent**, at `2h`: the pre-DW's input-VJP when there is one, the
    expand conv's otherwise. No fan-in: `ic ≠ oc`, so the block has no skip. -/
noncomputable def mnv4SBodyCotIn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * (2 * s.h) * (2 * s.h))) :=
  if s.preDWk = 0 then cInB N p.We p.be (mnv4SCotEc N s p xin dyOut)
  else dInB N p.Wq p.bq (mnv4SCotQc N s p xin dyOut)

/-- **The BODY's input cotangent** — what the render's `dx` carries before the skip fan-in.

    Dispatches on `s.preDWk` the way `mnv4PreDWSlot` does: with a pre-depthwise the body's `dx`
    is that depthwise's input-VJP, without one it is the expand conv's. -/
noncomputable def mnv4BodyCotIn (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Vec (N * (s.ic * s.h * s.h)) :=
  if s.preDWk = 0 then cInB N p.We p.be (mnv4CotEc N s p xin dyOut)
  else dInB N p.Wq p.bq (mnv4CotQc N s p xin dyOut)

/-- The skip fan-in itself: `body dx + dyOut`, at the block-input shape.

    Split from `mnv4BodyCotIn` for the same reason `mnv4SkipGraphB` is split from the body graph
    builders: the add needs `s.oc` and `s.ic` to be the SAME type, which they are at every stride-1
    row and are not at a row binder. The body's cotangent is row-generic; the add is applied at the
    concrete row, where `s.oc = s.ic` is `rfl`. -/
noncomputable def mnv4SkipCotIn {N n : Nat} (bodyDx dyOut : Vec (N * n)) : Vec (N * n) :=
  fun i => bodyDx i + dyOut i

-- ════════════════════════════════════════════════════════════════
-- § The ExtraDW block, tied — all TWELVE parameter nodes
-- ════════════════════════════════════════════════════════════════

/-- **ExtraDW block, tied.** All twelve parameter nodes — four conv/depthwise weights and four
    BatchNorm γ/β pairs — denote the certified batched `Σ_n` gradient at the real forward
    activations and the real backward-chain cotangent driven by `dyOut`.

    Each BatchNorm's γ/β reads the cotangent at THAT BatchNorm's output (`CotQn`, `CotEn`,
    `CotDn`, and `dyOut` itself for the project) while its conv reads the one at the conv's output
    (`CotQc`, `CotEc`, `CotDc`, `CotPc`). Off by one and the gradient is silently wrong — the two
    have the same type. The project BN's pair reads `dyOut` unmasked: the bottleneck is linear.

    There are no conv-bias conjuncts: `MobileNetV4RenderB` has no `convBias` flag, so those ops
    are never emitted and every slot here is exercised by the artifact. -/
def mnv4ExtraDWTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
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
  GradNodeB.DepthwiseWTiedBAt bf16 N s.h s.h xN cotN p.bq xin p.Wq cotQc
  ∧
  GradNodeB.BnPairTiedB N s.ic s.h s.h vN epsStr cotN p.eq_ p.gq p.bq2
      (reassocB N s.ic s.h s.h qc) (reassocB N s.ic s.h s.h cotQn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.be qr p.We cotEc
  ∧
  GradNodeB.BnPairTiedB N (s.ic * s.expand) s.h s.h vN epsStr cotN p.ee p.ge p.be2
      (reassocB N (s.ic * s.expand) s.h s.h ec) (reassocB N (s.ic * s.expand) s.h s.h cotEn)
  ∧
  GradNodeB.DepthwiseWTiedBAt bf16 N s.h s.h xN cotN p.bd er p.Wd cotDc
  ∧
  GradNodeB.BnPairTiedB N (s.ic * s.expand) s.h s.h vN epsStr cotN p.ed p.gd p.bd2
      (reassocB N (s.ic * s.expand) s.h s.h dc) (reassocB N (s.ic * s.expand) s.h s.h cotDn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.bz dr p.Wz cotPc
  ∧
  GradNodeB.BnPairTiedB N s.oc s.h s.h vN epsStr cotN p.ez p.gz p.bz2
      (reassocB N s.oc s.h s.h pc) (reassocB N s.oc s.h s.h dyOut)

/-- **And it holds** — twelve instantiations of the shared `∀ cot` leaf folds with
    the freedom removed: the cotangents are the chain's, not free. -/
theorem mnv4_extradw_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4ExtraDWTiedB N s xN cotN vN epsStr p bf16 xin dyOut := by
  unfold mnv4ExtraDWTiedB
  exact ⟨depthwiseWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds,
    depthwiseWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds⟩


def mnv4ConvNeXtTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotQn := mnv4CotQn N s p xin dyOut
  let cotQc := mnv4CotQc N s p xin dyOut
  let cotEn := mnv4CotEn N s p xin dyOut
  let cotEc := mnv4CotEc N s p xin dyOut
  let cotPc := mnv4CotPc N s p xin dyOut
  GradNodeB.DepthwiseWTiedBAt bf16 N s.h s.h xN cotN p.bq xin p.Wq cotQc
  ∧
  GradNodeB.BnPairTiedB N s.ic s.h s.h vN epsStr cotN p.eq_ p.gq p.bq2
      (reassocB N s.ic s.h s.h qc) (reassocB N s.ic s.h s.h cotQn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.be qr p.We cotEc
  ∧
  GradNodeB.BnPairTiedB N (s.ic * s.expand) s.h s.h vN epsStr cotN p.ee p.ge p.be2
      (reassocB N (s.ic * s.expand) s.h s.h ec) (reassocB N (s.ic * s.expand) s.h s.h cotEn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.bz dr p.Wz cotPc
  ∧
  GradNodeB.BnPairTiedB N s.oc s.h s.h vN epsStr cotN p.ez p.gz p.bz2
      (reassocB N s.oc s.h s.h pc) (reassocB N s.oc s.h s.h dyOut)

/-- **And it holds** — nine instantiations of the shared `∀ cot` leaf folds at the chain's cotangents. -/
theorem mnv4_convnext_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4ConvNeXtTiedB N s xN cotN vN epsStr p bf16 xin dyOut := by
  unfold mnv4ConvNeXtTiedB
  exact ⟨depthwiseWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds,
    convWTiedBAt_holds bf16, bnPairTiedB_holds⟩


def mnv4FfnTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let ec := batchMap N (flatConv p.We p.be) qr
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotEn := mnv4CotEn N s p xin dyOut
  let cotEc := mnv4CotEc N s p xin dyOut
  let cotPc := mnv4CotPc N s p xin dyOut
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.be qr p.We cotEc
  ∧
  GradNodeB.BnPairTiedB N (s.ic * s.expand) s.h s.h vN epsStr cotN p.ee p.ge p.be2
      (reassocB N (s.ic * s.expand) s.h s.h ec) (reassocB N (s.ic * s.expand) s.h s.h cotEn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.bz dr p.Wz cotPc
  ∧
  GradNodeB.BnPairTiedB N s.oc s.h s.h vN epsStr cotN p.ez p.gz p.bz2
      (reassocB N s.oc s.h s.h pc) (reassocB N s.oc s.h s.h dyOut)

/-- **And it holds** — six instantiations of the shared `∀ cot` leaf folds at the chain's cotangents. -/
theorem mnv4_ffn_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4FfnTiedB N s xN cotN vN epsStr p bf16 xin dyOut := by
  unfold mnv4FfnTiedB
  exact ⟨convWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds⟩


-- ════════════════════════════════════════════════════════════════
-- § The STRIDED block, tied — rows 1, 3 and 11
-- ════════════════════════════════════════════════════════════════

/-- **Strided block, tied** — rows 1, 3 and 11. All twelve parameter nodes denote the certified
    batched `Σ_n` gradient at the real forward activations and the backward-chain cotangent driven
    by `dyOut`.

    The strided node is the post-DW's `depthwiseStridedWeightGradB` (timm's `dw_mid`,
    symmetric), reading its input at `2h`; the pre-DW and the expand nodes read theirs at `2h`, the
    project's at `h`.

    No skip and no `addVB`: `ic ≠ oc` at all three rows, so the block IS its body. -/
def mnv4StridedTiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h))))
    (dyOut : Vec (N * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2).fwd xin
  let er := (cbReluLayer (h := 2 * s.h) (w := 2 * s.h) N p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd er
  let qc := batchMap N (depthwiseFlat p.Wq p.bq) xin
  let ec := batchMap N (flatConv p.We p.be) qr
  let dc := batchMap N (depthwiseStride2Flat p.Wd p.bd) er
  let pc := batchMap N (flatConv p.Wz p.bz) dr
  let cotQn := mnv4SCotQn N s p xin dyOut
  let cotQc := mnv4SCotQc N s p xin dyOut
  let cotEn := mnv4SCotEn N s p xin dyOut
  let cotEc := mnv4SCotEc N s p xin dyOut
  let cotDn := mnv4SCotDn N s p xin dyOut
  let cotDc := mnv4SCotDc N s p xin dyOut
  let cotPc := mnv4SCotPc N s p xin dyOut
  GradNodeB.DepthwiseWTiedBAt bf16 N (2 * s.h) (2 * s.h) xN cotN p.bq xin p.Wq cotQc
  ∧
  GradNodeB.BnPairTiedB N s.ic (2 * s.h) (2 * s.h) vN epsStr cotN p.eq_ p.gq p.bq2
      (reassocB N s.ic (2 * s.h) (2 * s.h) qc) (reassocB N s.ic (2 * s.h) (2 * s.h) cotQn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N (2 * s.h) (2 * s.h) xN cotN p.be qr p.We cotEc
  ∧
  GradNodeB.BnPairTiedB N (s.ic * s.expand) (2 * s.h) (2 * s.h) vN epsStr cotN p.ee p.ge p.be2
      (reassocB N (s.ic * s.expand) (2 * s.h) (2 * s.h) ec)
      (reassocB N (s.ic * s.expand) (2 * s.h) (2 * s.h) cotEn)
  ∧
  GradNodeB.DepthwiseStridedWTiedBAt bf16 N s.h s.h xN cotN p.bd er p.Wd cotDc
  ∧
  GradNodeB.BnPairTiedB N (s.ic * s.expand) s.h s.h vN epsStr cotN p.ed p.gd p.bd2
      (reassocB N (s.ic * s.expand) s.h s.h dc) (reassocB N (s.ic * s.expand) s.h s.h cotDn)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N s.h s.h xN cotN p.bz dr p.Wz cotPc
  ∧
  GradNodeB.BnPairTiedB N s.oc s.h s.h vN epsStr cotN p.ez p.gz p.bz2
      (reassocB N s.oc s.h s.h pc) (reassocB N s.oc s.h s.h dyOut)

/-- **And it holds** — twelve instantiations (one at the strided post-DW) of the shared
    `∀ cot` leaf folds with the freedom removed. -/
theorem mnv4_strided_tiedB (N : Nat) (s : UibSpec) (xN cotN vN epsStr : String) (p : UibParams s)
    (bf16 : Bool) (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h))))
    (dyOut : Vec (N * (s.oc * s.h * s.h))) :
    mnv4StridedTiedB N s xN cotN vN epsStr p bf16 xin dyOut := by
  unfold mnv4StridedTiedB
  exact ⟨depthwiseWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds,
    depthwiseStridedWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds⟩

-- ════════════════════════════════════════════════════════════════
-- § The stem, the fused stage and the head — the three non-UIB stages
-- ════════════════════════════════════════════════════════════════

/-! All three are generic in their widths, for the reason `MobileNetV4FullB.lean`'s stem-graph
docstring records: pinning literal resolutions here lets `den` and the
width-indexed `rfl`s actually run, and the kernel gives up. The capstone instantiates. -/

/-- Cotangent at the stem BN's output — the fused stage's `dx`, masked by the stem relu.
    Feeds `%sg` and `%sbt`. -/
noncomputable def mnv4StemCotN (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyStem : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (bnBatchLA N oc h w εs γs βs (batchMap N (flatConvStride2 Ws bs) x)) dyStem

/-- Cotangent at the stem CONV's output — through the stem BN's backward. Feeds `%sW`. -/
noncomputable def mnv4StemCotC (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyStem : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w εs γs (batchMap N (flatConvStride2 Ws bs) x)
    (mnv4StemCotN N h w Ws bs εs γs βs x dyStem)

/-- **Stem, tied.** Its three nodes at the chain's cotangents.

    **And the chain stops here.** No render emits a gradient into `%x`, so the artifact's
    backward ends at this weight gradient. That is why the
    stem sits outside `MobileNetV4FullB.lean`'s `CertLayer` trunk, and it is B0's situation
    exactly. -/
def mnv4StemTiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (bf16 : Bool)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) : Prop :=
  let sc := batchMap N (flatConvStride2 Ws bs) x
  let cotN' := mnv4StemCotN N h w Ws bs εs γs βs x dyStem
  let cotC := mnv4StemCotC N h w Ws bs εs γs βs x dyStem
  GradNodeB.ConvStridedWTiedBAt bf16 N h w xN cotN bs x Ws cotC
  ∧
  GradNodeB.BnPairTiedB N oc h w vN epsStr cotN εs γs βs (reassocB N oc h w sc)
      (reassocB N oc h w cotN')

theorem mnv4_stem_tiedB (N h w : Nat) {ic oc kH kW : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (bf16 : Bool)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) :
    mnv4StemTiedB N h w xN cotN vN epsStr Ws bs εs γs βs bf16 x dyStem := by
  unfold mnv4StemTiedB
  exact ⟨convStridedWTiedBAt_holds bf16, bnPairTiedB_holds⟩

/-- Cotangent at the fused stage's project CONV output. `dyF` reaches the project BN's γ/β
    unmasked — the fused stage ends in a BatchNorm with no activation. Feeds `%f0pW`. -/
noncomputable def mnv4FusedCotPc (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    -- `βp` is taken and NOT read: the BatchNorm input-gradient does not depend on the shift,
    -- which `bnInB` records by not taking one. Kept in the signature so every cotangent in this
    -- chain has the same argument list as the tie bundle that consumes it.
    (γp _βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w εp γp
    (batchMap N (flatConv Wp bp) (cbReluStridedB N (h := h) (w := w) Wc bc εc γc βc xin)) dyF

/-- Cotangent at the fused BN's output — the project conv's input-VJP, masked by the fused relu.
    Feeds `%f0cg` and `%f0cbt`. -/
noncomputable def mnv4FusedCotN (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w))) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
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


/-- **Fused stage, tied** — its six parameter nodes. `%f0cW` is `convStridedWeightGradB`,
    symmetric padding, like the stem's. -/
def mnv4FusedTiedB (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xN cotN vN epsStr : String) (bf16 : Bool)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyF : Vec (N * (oc * h * w))) : Prop :=
  let sw := cbReluStridedB N (h := h) (w := w) Wc bc εc γc βc xin
  let fc := batchMap N (flatConvStride2 Wc bc) xin
  let pc := batchMap N (flatConv Wp bp) sw
  let cotN' := mnv4FusedCotN N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  let cotC := mnv4FusedCotC N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  let cotPc := mnv4FusedCotPc N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
  GradNodeB.ConvStridedWTiedBAt bf16 N h w xN cotN bc xin Wc cotC
  ∧
  GradNodeB.BnPairTiedB N mid h w vN epsStr cotN εc γc βc (reassocB N mid h w fc)
      (reassocB N mid h w cotN')
  ∧
  GradNodeB.ConvWTiedBAt bf16 N h w xN cotN bp sw Wp cotPc
  ∧
  GradNodeB.BnPairTiedB N oc h w vN epsStr cotN εp γp βp (reassocB N oc h w pc)
      (reassocB N oc h w dyF)

theorem mnv4_fused_tiedB (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xN cotN vN epsStr : String) (bf16 : Bool)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyF : Vec (N * (oc * h * w))) :
    mnv4FusedTiedB N h w Wc bc εc γc βc Wp bp εp γp βp xN cotN vN epsStr bf16 xin dyF := by
  unfold mnv4FusedTiedB
  exact ⟨convStridedWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds⟩


/-! The head, timm's order: `cn_960` conv-bn-relu at `h×w`, GAP, `conv_head` conv-bn-relu on the
pooled `[N, mid, 1, 1]`, dense. The two `1×1` relabellings (`mnv4_pool11`) are no-ops in the render
and `Fin.cast` reads here. -/

/-- The pooled `[N, c]` read at `[N, c, 1, 1]` and back — `castLayer`'s forward and backward. -/
noncomputable def mnv4To11 {N c : Nat} (v : Vec (N * c)) : Vec (N * (c * 1 * 1)) :=
  fun i => v (Fin.cast (mnv4_pool11 N c).symm i)
noncomputable def mnv4From11 {N c : Nat} (v : Vec (N * (c * 1 * 1))) : Vec (N * c) :=
  fun i => v (Fin.cast (mnv4_pool11 N c) i)

/-- The pooled head features at `[N, mid, 1, 1]`: GAP of the first head stage, relabelled. -/
noncomputable def mnv4HeadPool (N h w : Nat) {c mid : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (xin : Vec (N * (c * h * w))) : Vec (N * (mid * 1 * 1)) :=
  mnv4To11 (batchMap N (globalAvgPoolFlat mid h w) (cbReluB N (h := h) (w := w) W1 b1 ε1 γ1 β1 xin))

/-- The dense input: `conv_head`-bn-relu of the pooled features, relabelled to `[N, oc]`. -/
noncomputable def mnv4HeadFeat (N h w : Nat) {c mid oc : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (xin : Vec (N * (c * h * w))) : Vec (N * oc) :=
  mnv4From11 (cbReluB N (h := 1) (w := 1) W2 b2 ε2 γ2 β2 (mnv4HeadPool N h w W1 b1 ε1 γ1 β1 xin))

/-- Cotangent at `conv_head`'s BN output (`[N, oc, 1, 1]`) — the dense input-VJP (`rowDenseBackFlat`,
    the `den` of the render's `denseRowBack`), relabelled,
    masked by that stage's relu. Feeds `%hg` and `%hbt`. -/
noncomputable def mnv4HeadCotHn (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    -- `bd` is taken and NOT read: the dense input-VJP does not see the bias. Kept so every
    -- cotangent in the head chain has the tie bundle's argument list.
    (Wd : Mat oc nCls) (_bd : Vec nCls) (cd : Option (Vec (N * oc)))
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (oc * 1 * 1)) :=
  reluMaskB (N * (oc * 1 * 1))
    (bnBatchLA N oc 1 1 ε2 γ2 β2
      (batchMap N (flatConv W2 b2) (mnv4HeadPool N h w W1 b1 ε1 γ1 β1 xin)))
    (mnv4To11 (dropoutOpt cd (rowDenseBackFlat N oc nCls Wd g)))

/-- Cotangent at `conv_head`'s CONV output. Feeds `%hW`. -/
noncomputable def mnv4HeadCotHc (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (cd : Option (Vec (N * oc)))
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (oc * 1 * 1)) :=
  bnInB N oc 1 1 ε2 γ2
    (batchMap N (flatConv W2 b2) (mnv4HeadPool N h w W1 b1 ε1 γ1 β1 xin))
    (mnv4HeadCotHn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g)

/-- Cotangent at the FIRST head BN's output — `conv_head`'s input-VJP, relabelled, spread by the
    GAP backward, masked by the first relu. Feeds `%h1g` and `%h1bt`. -/
noncomputable def mnv4HeadCotH1n (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (cd : Option (Vec (N * oc)))
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (bnBatchLA N mid h w ε1 γ1 β1 (batchMap N (flatConv W1 b1) xin))
    (gapInB N mid h w
      (mnv4From11 (cInB N W2 b2 (mnv4HeadCotHc N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g))))

/-- Cotangent at the first head CONV's output. Feeds `%h1W`. -/
noncomputable def mnv4HeadCotH1c (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (cd : Option (Vec (N * oc)))
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (mid * h * w)) :=
  bnInB N mid h w ε1 γ1 (batchMap N (flatConv W1 b1) xin)
    (mnv4HeadCotH1n N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g)

/-- **The head's input cotangent** — what block 21 receives as its `dyOut`. -/
noncomputable def mnv4HeadCotIn (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (cd : Option (Vec (N * oc)))
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Vec (N * (c * h * w)) :=
  cInB N W1 b1 (mnv4HeadCotH1c N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g)

/-- **Head, tied** — all EIGHT nodes: two conv weights, two BatchNorm γ/β pairs, and the
    classifier's weight and bias. `conv_head`'s nodes (`%hW`, `%hg`, `%hbt`) read the POOLED
    features at `h = w = 1`; the classifier reads `conv_head`'s relu output. -/
def mnv4HeadTiedB (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String) (bf16 : Bool)
    (cd : Option (Vec (N * oc))) (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) : Prop :=
  let pool := mnv4HeadPool N h w W1 b1 ε1 γ1 β1 xin
  let feat := mnv4HeadFeat N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 xin
  let c1 := batchMap N (flatConv W1 b1) xin
  let c2 := batchMap N (flatConv W2 b2) pool
  let cotH1n := mnv4HeadCotH1n N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g
  let cotH1c := mnv4HeadCotH1c N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g
  let cotHn := mnv4HeadCotHn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g
  let cotHc := mnv4HeadCotHc N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g
  GradNodeB.ConvWTiedBAt bf16 N h w xN cotN b1 xin W1 cotH1c
  ∧
  GradNodeB.BnPairTiedB N mid h w vN epsStr cotN ε1 γ1 β1 (reassocB N mid h w c1)
      (reassocB N mid h w cotH1n)
  ∧
  GradNodeB.ConvWTiedBAt bf16 N 1 1 xN cotN b2 pool W2 cotHc
  ∧
  GradNodeB.BnPairTiedB N oc 1 1 vN epsStr cotN ε2 γ2 β2 (reassocB N oc 1 1 c2)
      (reassocB N oc 1 1 cotHn)
  ∧ GradNodeB.DenseWTiedB N xN cotN (dropoutOpt cd feat) Wd bd g
  ∧ GradNodeB.DenseBTiedB N cotN Wd (fun _ => 0) bd g

theorem mnv4_head_tiedB (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String) (bf16 : Bool)
    (cd : Option (Vec (N * oc))) (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls)) :
    mnv4HeadTiedB N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xN cotN vN epsStr bf16 cd xin g := by
  unfold mnv4HeadTiedB
  exact ⟨convWTiedBAt_holds bf16, bnPairTiedB_holds, convWTiedBAt_holds bf16, bnPairTiedB_holds,
    denseWTiedB_holds, denseBTiedB_holds⟩


-- ════════════════════════════════════════════════════════════════
-- § Every block's input cotangent is its certified VJP's backward
-- ════════════════════════════════════════════════════════════════

/-! The capstone below threads `dy21 … dyStem` through `mnv4HeadCotIn`, `mnv4BodyCotIn` +
`mnv4SkipCotIn`, `mnv4SBodyCotIn` and `mnv4FusedCotIn`. Each is stated here to be the backward
of the certified `CertLayer` VJP it stands for, at a block input where that layer is certified
(its `.ok`). Every proof is the layer's `.faithful` at `e := .operand "" dyOut`
(`CertLayer.backward_eq_den_graph`), then a definitional match with the constructed chain. -/

/-- **A stride-1 UIB body's input cotangent is its certified VJP's backward.** The chain the
    render emits (`mnv4BodyCotIn`, through `mnv4CotQc … mnv4CotPc`) is
    `(mnv4BodyOfRow N s p).vjp`'s backward at the block input, for every row: the four
    `preDWk`/`postDWk` cases (ExtraDW, ConvNeXt-like, FFN, and the IB shape no Conv-M row uses)
    each reduce to the body layer's `.faithful`. -/
theorem mnv4BodyCotIn_eq_vjp (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) (dyOut : Vec (N * (s.oc * s.h * s.h)))
    (hok : (mnv4BodyOfRow N s p).ok xin) :
    mnv4BodyCotIn N s p xin dyOut = ((mnv4BodyOfRow N s p).vjp xin hok).backward dyOut := by
  rw [CertLayer.backward_eq_den_graph]
  by_cases hq : s.preDWk = 0 <;> by_cases hd : s.postDWk = 0 <;>
    simp only [mnv4BodyCotIn, mnv4CotQc, mnv4CotQn, mnv4CotEc, mnv4CotEn, mnv4CotDc, mnv4CotDn,
      mnv4CotPc, mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot, hq, hd, ↓reduceIte,
      CertLayer.comp, CertLayer.id', mnv4DWBnLayer, mnv4DWReluLayer, cbReluLayer, projLayer] <;>
    rfl

/-- **A strided UIB body's input cotangent is its certified VJP's backward** — rows 1, 3, 11,
    at `2h`. -/
theorem mnv4SBodyCotIn_eq_vjp (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) (dyOut : Vec (N * (s.oc * s.h * s.h)))
    (hok : (mnv4StridedBodyOfRow N s p).ok xin) :
    mnv4SBodyCotIn N s p xin dyOut = ((mnv4StridedBodyOfRow N s p).vjp xin hok).backward dyOut := by
  rw [CertLayer.backward_eq_den_graph]
  by_cases hq : s.preDWk = 0 <;>
    simp only [mnv4SBodyCotIn, mnv4SCotQc, mnv4SCotQn, mnv4SCotEc, mnv4SCotEn, mnv4SCotDc,
      mnv4SCotDn, mnv4SCotPc, mnv4StridedBodyOfRow, mnv4UibStridedBody, mnv4PreDWSlot, hq,
      ↓reduceIte, CertLayer.comp, CertLayer.id', mnv4DWBnLayer, mnv4DWReluStridedLayer,
      cbReluLayer, projLayer] <;>
    rfl


/-- **The skip fan-in is the residual layer's backward**: `body dx + dyOut`, with `body dx` the
    body's certified backward, is `CertLayer.residual`'s. Generic in the body, so at a stride-1 row
    it composes with `mnv4BodyCotIn_eq_vjp` (the add needs `s.oc = s.ic`, which is `rfl` at every
    concrete row and not at a row binder). -/
theorem mnv4SkipCotIn_eq_vjp {N n : Nat} (L : CertLayer (N * n) (N * n)) (x : Vec (N * n))
    (hok : L.ok x) (dyOut : Vec (N * n)) :
    mnv4SkipCotIn ((L.vjp x hok).backward dyOut) dyOut
      = ((CertLayer.residual L).vjp x hok).backward dyOut := by
  rw [CertLayer.backward_eq_den_graph (CertLayer.residual L) x hok, CertLayer.residual_graph,
    den_addV, ← CertLayer.backward_eq_den_graph L x hok, den_operand]
  rfl

/-- **The fused stage's input cotangent is its certified VJP's backward.** -/
theorem mnv4FusedCotIn_eq_vjp (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (hεc : 0 < εc) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc)
    (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyF : Vec (N * (oc * h * w)))
    (hok : (mnv4FusedStage N (cbReluStridedLayer (h := h) (w := w) N Wc bc εc hεc γc βc)
      (projLayer (h := h) (w := w) N Wp bp εp hεp γp βp)).ok xin) :
    mnv4FusedCotIn N h w Wc bc εc γc βc Wp bp εp γp βp xin dyF
      = ((mnv4FusedStage N (cbReluStridedLayer (h := h) (w := w) N Wc bc εc hεc γc βc)
          (projLayer (h := h) (w := w) N Wp bp εp hεp γp βp)).vjp xin hok).backward dyF := by
  rw [CertLayer.backward_eq_den_graph]
  rfl


-- `den` pushed through the head's stage graphs, generic in the incoming graph `e`, so the head's
-- two `castIdx` relabellings (`h ▸ e`, which `rfl` cannot see through) can be rewritten by
-- `den_castIdx` rather than unfolded.
private theorem den_cbReluBack {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (γ β : Vec oc) (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (cbReluBackBatchedGraph W b ε γ β x e)
      = cInB N W b (bnInB N oc h w ε γ (batchMap N (flatConv W b) x)
          (reluMaskB (N * (oc * h * w)) (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x))
            (den e))) := rfl

private theorem den_gapBack {N c h w : Nat} (e : SHlo (N * c)) :
    den (SHlo.gapBackBatched (N := N) (c := c) (h := h) (w := w) e) = gapInB N c h w (den e) := rfl

private theorem den_denseRowBack {N a nC : Nat} (W : Mat a nC) (e : SHlo (N * nC)) :
    den (SHlo.denseRowBack (N := N) (a := a) (c := nC) "%Wd" W e)
      = rowDenseBackFlat N a nC W (den e) := rfl

/-- **The classifier-dropout site as a certified layer**: `dropoutOpt cd`, certified everywhere,
    its backward graph the `dropoutB` node at the same mask (nothing at `none`). -/
noncomputable def dropoutOptLayer (N : Nat) {n : Nat} (cd : Option (Vec (N * n))) :
    CertLayer (N * n) (N * n) where
  fwd := dropoutOpt cd
  ok := fun _ => True
  diff := fun x _ => (dropoutOpt_differentiable cd) x
  vjp := fun x _ => (dropoutOptHasVJP cd).toHasVJPAt x
  graph := fun _ e => dropoutOptG "%do" cd e
  faithful := fun x _ e => by rw [den_dropoutOptG]; rfl

/-- **The head's input cotangent — block 21's `dyOut` — is its certified VJP's backward**, at
    any loss cotangent `g`. -/
theorem mnv4HeadCotIn_eq_vjp (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls))
    (hok : (mnv4Head N (cbReluLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1)
      (gapLayer N (c := mid) (h := h) (w := w))
      (cbReluLayer (h := 1) (w := 1) N W2 b2 ε2 hε2 γ2 β2) (denseLayer N Wd bd)).ok xin) :
    mnv4HeadCotIn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd none xin g
      = ((mnv4Head N (cbReluLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1)
          (gapLayer N (c := mid) (h := h) (w := w))
          (cbReluLayer (h := 1) (w := 1) N W2 b2 ε2 hε2 γ2 β2) (denseLayer N Wd bd)).vjp
            xin hok).backward g := by
  rw [CertLayer.backward_eq_den_graph]
  simp only [mnv4Head, CertLayer.comp, castLayer, gapLayer, denseLayer, cbReluLayer,
    den_cbReluBack, den_gapBack, den_castIdx, den_denseRowBack, den_operand, reindexCLM_apply]
  rfl


/-- **The same at the classifier-dropout site**: with `dropoutOpt cd` between `conv_head`'s
    output and the dense, the chain at `cd` is the backward of the head whose classifier is the
    dropout layer followed by the dense. -/
theorem mnv4HeadCotIn_eq_vjp_do (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (cd : Option (Vec (N * oc)))
    (xin : Vec (N * (c * h * w))) (g : Vec (N * nCls))
    (hok : (mnv4Head N (cbReluLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1)
      (gapLayer N (c := mid) (h := h) (w := w))
      (cbReluLayer (h := 1) (w := 1) N W2 b2 ε2 hε2 γ2 β2) ((dropoutOptLayer N cd).comp (denseLayer N Wd bd))).ok xin) :
    mnv4HeadCotIn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd cd xin g
      = ((mnv4Head N (cbReluLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1)
          (gapLayer N (c := mid) (h := h) (w := w))
          (cbReluLayer (h := 1) (w := 1) N W2 b2 ε2 hε2 γ2 β2) ((dropoutOptLayer N cd).comp (denseLayer N Wd bd))).vjp
            xin hok).backward g := by
  rw [CertLayer.backward_eq_den_graph]
  simp only [mnv4Head, CertLayer.comp, castLayer, gapLayer, denseLayer, cbReluLayer, dropoutOptLayer, den_dropoutOptG,
    den_cbReluBack, den_gapBack, den_castIdx, den_denseRowBack, den_operand, reindexCLM_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The stochastic-depth sites — one per skip row
-- ════════════════════════════════════════════════════════════════

/-- Site `k`'s per-example scale — of the eighteen skip rows in table order (rows 2, 4–10, 12–21;
    the renderer's `mnv4DropSites`, `MobileNetV4FullBDrop`'s table) — when the render carries
    stochastic depth. Named, so no statement spells the `Option.map`. -/
def mnv4Sd {N : Nat} (sd : Option (Fin 18 → Vec N)) (k : Fin 18) : Option (Vec N) :=
  sd.map fun m => m k

@[simp] theorem mnv4Sd_none {N : Nat} (k : Fin 18) :
    mnv4Sd (none : Option (Fin 18 → Vec N)) k = none := rfl

@[simp] theorem mnv4Sd_some {N : Nat} (m : Fin 18 → Vec N) (k : Fin 18) :
    mnv4Sd (some m) k = some (m k) := rfl

/-- **A stochastic-depth site as a certified layer**: `dropPathOpt N n s`, certified everywhere,
    its backward graph the `dropPathB` node at the same per-example scale (nothing at `none`). -/
noncomputable def dropPathOptLayer (N n : Nat) (s : Option (Vec N)) : CertLayer (N * n) (N * n) where
  fwd := dropPathOpt N n s
  ok := fun _ => True
  diff := fun x _ => (dropPathOpt_differentiable N n s) x
  vjp := fun x _ => (dropPathOptHasVJP N n s).toHasVJPAt x
  graph := fun _ e => dropPathOptG "%dp" s e
  faithful := fun x _ e => by rw [den_dropPathOptG]; rfl

/-- **A skip row with its stochastic-depth site**, certified: the body, then the site on the
    BRANCH, then the identity skip — `x ↦ dropPathOpt s (L.fwd x) + x`. At `none` its forward is
    `CertLayer.residual L`'s, at `some s` `MobileNetV4FullBDrop`'s `mnv4SkipDrop L.fwd s`, both
    `rfl`; it is certified exactly where the body is. -/
noncomputable def mnv4SkipDropLayer {N n : Nat} (L : CertLayer (N * n) (N * n)) (s : Option (Vec N)) :
    CertLayer (N * n) (N * n) :=
  CertLayer.residual (L.comp (dropPathOptLayer N n s))

theorem mnv4SkipDropLayer_none {N n : Nat} (L : CertLayer (N * n) (N * n)) :
    (mnv4SkipDropLayer L none).fwd = (CertLayer.residual L).fwd := rfl

theorem mnv4SkipDropLayer_some {N n : Nat} (L : CertLayer (N * n) (N * n)) (s : Vec N) :
    (mnv4SkipDropLayer L (some s)).fwd = mnv4SkipDrop L.fwd s := rfl

theorem mnv4SkipDropLayer_ok {N n : Nat} (L : CertLayer (N * n) (N * n)) (s : Option (Vec N))
    (x : Vec (N * n)) : (mnv4SkipDropLayer L s).ok x ↔ L.ok x :=
  ⟨fun h => h.1, fun h => ⟨h, trivial⟩⟩

/-- **The skip fan-in through a drop site is the dropped skip row's backward**: the body reads
    the block-output cotangent THROUGH the site (`dropPathOpt s dyOut` — the site's VJP is
    itself, `dropPathOpt_vjp_is_self`), the skip reads it raw, and the sum is
    `mnv4SkipDropLayer`'s certified backward. `mnv4SkipCotIn_eq_vjp` is this at `none`. -/
theorem mnv4SkipCotIn_eq_vjp_drop {N n : Nat} (L : CertLayer (N * n) (N * n)) (s : Option (Vec N))
    (x : Vec (N * n)) (hok : L.ok x) (dyOut : Vec (N * n)) :
    mnv4SkipCotIn ((L.vjp x hok).backward (dropPathOpt N n s dyOut)) dyOut
      = ((mnv4SkipDropLayer L s).vjp x ⟨hok, trivial⟩).backward dyOut :=
  mnv4SkipCotIn_eq_vjp (L.comp (dropPathOptLayer N n s)) x ⟨hok, trivial⟩ dyOut

-- ════════════════════════════════════════════════════════════════
-- § The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- The per-BLOCK forward prefixes: `mnv4Blk0` is the fused stage's output — block 1's input —
    and `mnv4Blk{k}` is the activation entering block `k+1`.

    `MobileNetV4FullB.lean`'s `mnv4Pre0 … mnv4Pre6` are the resolution-group prefixes, which is
    the granularity the whole-net forward, graph and VJP need; the tie needs one per BLOCK, so these 22 name the finer chain.
    `mnv4Blk0` is definitionally `mnv4Pre1`. From `mnv4Blk2` on they carry the stochastic-depth
    sites `sd`: skip row `k` is `mnv4SkipDropLayer` at its site `mnv4Sd sd ·` (`none` the
    drop-free forward, where `mnv4Blk{k}` meets the group prefixes — `mnv4Pre6_eq_blk`). -/
noncomputable def mnv4Blk0 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (48 * (2 * (2 * (2 * f))) * (2 * (2 * (2 * f))))) := mnv4Pre1 N f w x

noncomputable def mnv4Blk1 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (80 * (2 * (2 * f)) * (2 * (2 * f)))) :=
  (mnv4StridedBodyOfRow N (mnv4Row1 f) w.b1).fwd (mnv4Blk0 N f w x)

noncomputable def mnv4Blk2 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (80 * (2 * (2 * f)) * (2 * (2 * f)))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row2 f) w.b2) (mnv4Sd sd 0)).fwd (mnv4Blk1 N f w x)

noncomputable def mnv4Blk3 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4StridedBodyOfRow N (mnv4Row3 f) w.b3).fwd (mnv4Blk2 N f w sd x)

noncomputable def mnv4Blk4 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row4 f) w.b4) (mnv4Sd sd 1)).fwd (mnv4Blk3 N f w sd x)

noncomputable def mnv4Blk5 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row5 f) w.b5) (mnv4Sd sd 2)).fwd (mnv4Blk4 N f w sd x)

noncomputable def mnv4Blk6 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row6 f) w.b6) (mnv4Sd sd 3)).fwd (mnv4Blk5 N f w sd x)

noncomputable def mnv4Blk7 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row7 f) w.b7) (mnv4Sd sd 4)).fwd (mnv4Blk6 N f w sd x)

noncomputable def mnv4Blk8 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row8 f) w.b8) (mnv4Sd sd 5)).fwd (mnv4Blk7 N f w sd x)

noncomputable def mnv4Blk9 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row9 f) w.b9) (mnv4Sd sd 6)).fwd (mnv4Blk8 N f w sd x)

noncomputable def mnv4Blk10 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (160 * (2 * f) * (2 * f))) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row10 f) w.b10) (mnv4Sd sd 7)).fwd (mnv4Blk9 N f w sd x)

noncomputable def mnv4Blk11 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4StridedBodyOfRow N (mnv4Row11 f) w.b11).fwd (mnv4Blk10 N f w sd x)

noncomputable def mnv4Blk12 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row12 f) w.b12) (mnv4Sd sd 8)).fwd (mnv4Blk11 N f w sd x)

noncomputable def mnv4Blk13 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row13 f) w.b13) (mnv4Sd sd 9)).fwd (mnv4Blk12 N f w sd x)

noncomputable def mnv4Blk14 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row14 f) w.b14) (mnv4Sd sd 10)).fwd (mnv4Blk13 N f w sd x)

noncomputable def mnv4Blk15 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row15 f) w.b15) (mnv4Sd sd 11)).fwd (mnv4Blk14 N f w sd x)

noncomputable def mnv4Blk16 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row16 f) w.b16) (mnv4Sd sd 12)).fwd (mnv4Blk15 N f w sd x)

noncomputable def mnv4Blk17 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row17 f) w.b17) (mnv4Sd sd 13)).fwd (mnv4Blk16 N f w sd x)

noncomputable def mnv4Blk18 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row18 f) w.b18) (mnv4Sd sd 14)).fwd (mnv4Blk17 N f w sd x)

noncomputable def mnv4Blk19 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row19 f) w.b19) (mnv4Sd sd 15)).fwd (mnv4Blk18 N f w sd x)

noncomputable def mnv4Blk20 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row20 f) w.b20) (mnv4Sd sd 16)).fwd (mnv4Blk19 N f w sd x)

noncomputable def mnv4Blk21 (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (sd : Option (Fin 18 → Vec N))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) : Vec (N * (256 * f * f)) :=
  (mnv4SkipDropLayer (mnv4BodyOfRow N (mnv4Row21 f) w.b21) (mnv4Sd sd 17)).fwd (mnv4Blk20 N f w sd x)

/-- **The whole batch-BN MobileNetV4-Conv-M train step, tied.** Threading the net's own forward
    prefixes as the block inputs and an arbitrary loss cotangent `g` down through the head, the 21
    UIB blocks and the fused stage — each `dy` is that layer's certified VJP backward wherever the
    layer is certified (`mnv4HeadCotIn_eq_vjp`, `mnv4BodyCotIn_eq_vjp` with
    `mnv4SkipCotIn_eq_vjp`, `mnv4SBodyCotIn_eq_vjp`, `mnv4FusedCotIn_eq_vjp`) — every parameter
    GRADIENT node of the net — stem 3, fused 6, thirteen ExtraDW blocks × 12, four
    ConvNeXt-like × 9, four FFN × 6, head 8 — denotes the certified batched `Σ_n` gradient. That is
    **233**, the render's own census and `mnv4_fwd.mlir`'s signature minus `%x`. No free activation
    and no symbolic cotangent below the loss.

    **`g` is a binder.** The loss chain is not part of this statement; `mnv4_lossCot_is_smoothedCE_grad`
    instantiates it at the label-smoothed softmax cotangent the artifacts actually emit.

    **No smoothness hypothesis and no `0 < ε`**, and `N` and `nCls` are both binders. The folds
    are `∀ cot` statements at explicitly constructed cotangents; the kink and positivity conditions
    live in the `CertLayer`s whose `.ok` `MobileNetV4FullBVJP.lean` binds. The final side `f`
    is a binder, as ResNet-50's `q`: `f = 7` is the artifacts at 224×224 (`mnv4in_adamdp64`, …), `f = 8`
    the paper tier's `mnv4in256_acc{,dp}8x128wxdropdowd01bf16`, a drop artifact. `cd` is the
    classifier dropout as an optional site: `none` is the drop-free artifacts' chain, `some m`
    the `do` renders', where the dense weight node reads the dropped activation and the
    cotangent handed down to `conv_head` passes through the mask. `sd` is stochastic depth,
    likewise optional: the eighteen skip rows' per-example scales (site `k` the `k`-th skip row in
    table order — rows 2, 4–10, 12–21 — read off as `mnv4Sd sd k`), `none` the chain without
    drop-path and `some s` the `drop` renders' — the paper tier's — where the block prefixes are
    the dropped forward's (`mnv4SkipDropLayer`), a skip row's body nodes are tied at the
    cotangent THROUGH its site (`dropPathOpt N n (mnv4Sd sd k) dy`) and the skip fan-in adds the
    raw `dy` (`mnv4SkipCotIn_eq_vjp_drop`).
    One device, at either precision:
    `bf16` selects the conv and depthwise weight nodes' kind — `false` the f32 artifacts', `true`
    the bf16 ones', read over ℝ (`Bf16Erasure`); the right-hand side is the same certified
    gradient at either value.

    On the data-parallel (sync-BN) artifacts this theorem applies at `N := R·N`: the per-replica
    statement is `MobileNetV4SyncTieB.mnv4_net_syncTiedB`, whose right-hand sides are this
    theorem's nodes at the global batch. The AdamW update that follows is not stated here. -/
theorem mnv4_net_tiedB (N f : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (g : Vec (N * nCls)) :
  let dy21 := mnv4HeadCotIn N f f w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
               w.Wd w.bd cd (mnv4Blk21 N f w sd x) g
  let dy20 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row21 f) w.b21 (mnv4Blk20 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 17) dy21)) dy21
  let dy19 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row20 f) w.b20 (mnv4Blk19 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 16) dy20)) dy20
  let dy18 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row19 f) w.b19 (mnv4Blk18 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 15) dy19)) dy19
  let dy17 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row18 f) w.b18 (mnv4Blk17 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 14) dy18)) dy18
  let dy16 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row17 f) w.b17 (mnv4Blk16 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 13) dy17)) dy17
  let dy15 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row16 f) w.b16 (mnv4Blk15 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 12) dy16)) dy16
  let dy14 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row15 f) w.b15 (mnv4Blk14 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 11) dy15)) dy15
  let dy13 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row14 f) w.b14 (mnv4Blk13 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 10) dy14)) dy14
  let dy12 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row13 f) w.b13 (mnv4Blk12 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 9) dy13)) dy13
  let dy11 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row12 f) w.b12 (mnv4Blk11 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 8) dy12)) dy12
  let dy10 := mnv4SBodyCotIn N (mnv4Row11 f) w.b11 (mnv4Blk10 N f w sd x) dy11
  let dy9 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row10 f) w.b10 (mnv4Blk9 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 7) dy10)) dy10
  let dy8 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row9 f) w.b9 (mnv4Blk8 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 6) dy9)) dy9
  let dy7 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row8 f) w.b8 (mnv4Blk7 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 5) dy8)) dy8
  let dy6 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row7 f) w.b7 (mnv4Blk6 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 4) dy7)) dy7
  let dy5 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row6 f) w.b6 (mnv4Blk5 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 3) dy6)) dy6
  let dy4 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row5 f) w.b5 (mnv4Blk4 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 2) dy5)) dy5
  let dy3 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row4 f) w.b4 (mnv4Blk3 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 1) dy4)) dy4
  let dy2 := mnv4SBodyCotIn N (mnv4Row3 f) w.b3 (mnv4Blk2 N f w sd x) dy3
  let dy1 := mnv4SkipCotIn (mnv4BodyCotIn N (mnv4Row2 f) w.b2 (mnv4Blk1 N f w x) (dropPathOpt N (80 * (2 * (2 * f)) * (2 * (2 * f))) (mnv4Sd sd 0) dy2)) dy2
  let dy0 := mnv4SBodyCotIn N (mnv4Row1 f) w.b1 (mnv4Blk0 N f w x) dy1
  let dyStem := mnv4FusedCotIn N (2 * (2 * (2 * f))) (2 * (2 * (2 * f))) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                 w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt (mnv4Pre0 N f w x) dy0
  mnv4StemTiedB N (2 * (2 * (2 * (2 * f)))) (2 * (2 * (2 * (2 * f)))) xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt bf16 x dyStem
  ∧ mnv4FusedTiedB N (2 * (2 * (2 * f))) (2 * (2 * (2 * f))) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.f0pg
      w.f0pbt xN cotN vN epsStr bf16 (mnv4Pre0 N f w x) dy0
  ∧ mnv4StridedTiedB N (mnv4Row1 f) xN cotN vN epsStr w.b1 bf16 (mnv4Blk0 N f w x) dy1
  ∧ mnv4ExtraDWTiedB N (mnv4Row2 f) xN cotN vN epsStr w.b2 bf16 (mnv4Blk1 N f w x) (dropPathOpt N (80 * (2 * (2 * f)) * (2 * (2 * f))) (mnv4Sd sd 0) dy2)
  ∧ mnv4StridedTiedB N (mnv4Row3 f) xN cotN vN epsStr w.b3 bf16 (mnv4Blk2 N f w sd x) dy3
  ∧ mnv4ExtraDWTiedB N (mnv4Row4 f) xN cotN vN epsStr w.b4 bf16 (mnv4Blk3 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 1) dy4)
  ∧ mnv4ExtraDWTiedB N (mnv4Row5 f) xN cotN vN epsStr w.b5 bf16 (mnv4Blk4 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 2) dy5)
  ∧ mnv4ExtraDWTiedB N (mnv4Row6 f) xN cotN vN epsStr w.b6 bf16 (mnv4Blk5 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 3) dy6)
  ∧ mnv4ExtraDWTiedB N (mnv4Row7 f) xN cotN vN epsStr w.b7 bf16 (mnv4Blk6 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 4) dy7)
  ∧ mnv4ConvNeXtTiedB N (mnv4Row8 f) xN cotN vN epsStr w.b8 bf16 (mnv4Blk7 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 5) dy8)
  ∧ mnv4FfnTiedB N (mnv4Row9 f) xN cotN vN epsStr w.b9 bf16 (mnv4Blk8 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 6) dy9)
  ∧ mnv4ConvNeXtTiedB N (mnv4Row10 f) xN cotN vN epsStr w.b10 bf16 (mnv4Blk9 N f w sd x) (dropPathOpt N (160 * (2 * f) * (2 * f)) (mnv4Sd sd 7) dy10)
  ∧ mnv4StridedTiedB N (mnv4Row11 f) xN cotN vN epsStr w.b11 bf16 (mnv4Blk10 N f w sd x) dy11
  ∧ mnv4ExtraDWTiedB N (mnv4Row12 f) xN cotN vN epsStr w.b12 bf16 (mnv4Blk11 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 8) dy12)
  ∧ mnv4ExtraDWTiedB N (mnv4Row13 f) xN cotN vN epsStr w.b13 bf16 (mnv4Blk12 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 9) dy13)
  ∧ mnv4ExtraDWTiedB N (mnv4Row14 f) xN cotN vN epsStr w.b14 bf16 (mnv4Blk13 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 10) dy14)
  ∧ mnv4FfnTiedB N (mnv4Row15 f) xN cotN vN epsStr w.b15 bf16 (mnv4Blk14 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 11) dy15)
  ∧ mnv4ConvNeXtTiedB N (mnv4Row16 f) xN cotN vN epsStr w.b16 bf16 (mnv4Blk15 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 12) dy16)
  ∧ mnv4ExtraDWTiedB N (mnv4Row17 f) xN cotN vN epsStr w.b17 bf16 (mnv4Blk16 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 13) dy17)
  ∧ mnv4ExtraDWTiedB N (mnv4Row18 f) xN cotN vN epsStr w.b18 bf16 (mnv4Blk17 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 14) dy18)
  ∧ mnv4FfnTiedB N (mnv4Row19 f) xN cotN vN epsStr w.b19 bf16 (mnv4Blk18 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 15) dy19)
  ∧ mnv4FfnTiedB N (mnv4Row20 f) xN cotN vN epsStr w.b20 bf16 (mnv4Blk19 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 16) dy20)
  ∧ mnv4ConvNeXtTiedB N (mnv4Row21 f) xN cotN vN epsStr w.b21 bf16 (mnv4Blk20 N f w sd x) (dropPathOpt N (256 * f * f) (mnv4Sd sd 17) dy21)
  ∧ mnv4HeadTiedB N f f w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
      w.Wd w.bd xN cotN vN epsStr bf16 cd (mnv4Blk21 N f w sd x) g := by
  intro dy21 dy20 dy19 dy18 dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3
    dy2 dy1 dy0 dyStem
  exact ⟨mnv4_stem_tiedB N (2 * (2 * (2 * (2 * f)))) (2 * (2 * (2 * (2 * f)))) xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt bf16 x dyStem,
    mnv4_fused_tiedB N (2 * (2 * (2 * f))) (2 * (2 * (2 * f))) w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE w.f0pg
      w.f0pbt xN cotN vN epsStr bf16 (mnv4Pre0 N f w x) dy0,
    mnv4_strided_tiedB N (mnv4Row1 f) xN cotN vN epsStr w.b1 bf16 (mnv4Blk0 N f w x) dy1,
    mnv4_extradw_tiedB N (mnv4Row2 f) xN cotN vN epsStr w.b2 bf16 (mnv4Blk1 N f w x) _,
    mnv4_strided_tiedB N (mnv4Row3 f) xN cotN vN epsStr w.b3 bf16 (mnv4Blk2 N f w sd x) dy3,
    mnv4_extradw_tiedB N (mnv4Row4 f) xN cotN vN epsStr w.b4 bf16 (mnv4Blk3 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row5 f) xN cotN vN epsStr w.b5 bf16 (mnv4Blk4 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row6 f) xN cotN vN epsStr w.b6 bf16 (mnv4Blk5 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row7 f) xN cotN vN epsStr w.b7 bf16 (mnv4Blk6 N f w sd x) _,
    mnv4_convnext_tiedB N (mnv4Row8 f) xN cotN vN epsStr w.b8 bf16 (mnv4Blk7 N f w sd x) _,
    mnv4_ffn_tiedB N (mnv4Row9 f) xN cotN vN epsStr w.b9 bf16 (mnv4Blk8 N f w sd x) _,
    mnv4_convnext_tiedB N (mnv4Row10 f) xN cotN vN epsStr w.b10 bf16 (mnv4Blk9 N f w sd x) _,
    mnv4_strided_tiedB N (mnv4Row11 f) xN cotN vN epsStr w.b11 bf16 (mnv4Blk10 N f w sd x) dy11,
    mnv4_extradw_tiedB N (mnv4Row12 f) xN cotN vN epsStr w.b12 bf16 (mnv4Blk11 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row13 f) xN cotN vN epsStr w.b13 bf16 (mnv4Blk12 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row14 f) xN cotN vN epsStr w.b14 bf16 (mnv4Blk13 N f w sd x) _,
    mnv4_ffn_tiedB N (mnv4Row15 f) xN cotN vN epsStr w.b15 bf16 (mnv4Blk14 N f w sd x) _,
    mnv4_convnext_tiedB N (mnv4Row16 f) xN cotN vN epsStr w.b16 bf16 (mnv4Blk15 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row17 f) xN cotN vN epsStr w.b17 bf16 (mnv4Blk16 N f w sd x) _,
    mnv4_extradw_tiedB N (mnv4Row18 f) xN cotN vN epsStr w.b18 bf16 (mnv4Blk17 N f w sd x) _,
    mnv4_ffn_tiedB N (mnv4Row19 f) xN cotN vN epsStr w.b19 bf16 (mnv4Blk18 N f w sd x) _,
    mnv4_ffn_tiedB N (mnv4Row20 f) xN cotN vN epsStr w.b20 bf16 (mnv4Blk19 N f w sd x) _,
    mnv4_convnext_tiedB N (mnv4Row21 f) xN cotN vN epsStr w.b21 bf16 (mnv4Blk20 N f w sd x) _,
    mnv4_head_tiedB N f f w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
      w.Wd w.bd xN cotN vN epsStr bf16 cd (mnv4Blk21 N f w sd x) g⟩


/-- **The head with the classifier-dropout site**: `mnv4HeadStack` with `dropoutOptLayer` between
    `conv_head`'s output and the dense — the certified layer `mnv4HeadCotIn_eq_vjp_do` reads the
    head chain against. -/
noncomputable def mnv4HeadStackDo (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (cd : Option (Vec (N * 1280))) : CertLayer (N * (256 * f * f)) (N * nCls) :=
  mnv4Head N (cbReluLayer (h := f) (w := f) N w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt)
    (gapLayer N (c := 960) (h := f) (w := f))
    (cbReluLayer (h := 1) (w := 1) N w.hW w.hb w.hE w.hhE w.hg w.hbt)
    ((dropoutOptLayer N cd).comp (denseLayer N w.Wd w.bd))

/-- **The forward the loss reads, at the classifier-dropout site**: the head with the site
    (`mnv4HeadStackDo`) on the trunk — `mobilenetv4ForwardBFull` at `none`,
    `mobilenetv4ForwardBFullDo` at `some m`, both `rfl`. -/
noncomputable def mobilenetv4ForwardBFullDoOpt (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    Vec (N * nCls) :=
  (mnv4HeadStackDo N f w cd).fwd (mnv4Pre6 N f w x)

theorem mobilenetv4ForwardBFullDoOpt_none (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    mobilenetv4ForwardBFullDoOpt N f w none = mobilenetv4ForwardBFull N f w := rfl

theorem mobilenetv4ForwardBFullDoOpt_some (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (m : Vec (N * 1280)) :
    mobilenetv4ForwardBFullDoOpt N f w (some m) = mobilenetv4ForwardBFullDo N f w m := rfl

/-- **The forward the loss reads, at both regularisers' sites**: the head with the
    classifier-dropout site on the per-block chain with the eighteen stochastic-depth sites.
    `none`/`none` is `mobilenetv4ForwardBFull`, `none`/`cd` is `mobilenetv4ForwardBFullDoOpt`
    (`mobilenetv4ForwardBFullDropOpt_none`), and `some s`/`some m` is `MobileNetV4FullBDrop`'s
    `mobilenetv4ForwardBFullDrop` (`mobilenetv4ForwardBFullDropOpt_some`), the forward the typed
    graph `mnv4FwdGraphBFullDrop` denotes. -/
noncomputable def mobilenetv4ForwardBFullDropOpt (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    Vec (N * nCls) :=
  (mnv4HeadStackDo N f w cd).fwd (mnv4Blk21 N f w sd x)

/-- The group prefixes meet the block prefixes at every group boundary, at the drop-free sites. -/
theorem mnv4Pre2_eq_blk (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mnv4Pre2 N f w x = mnv4Blk2 N f w none x := by
  rw [mnv4Pre2, mnv4Res28Layer, CertLayer.comp_fwd_apply]; rfl

theorem mnv4Pre3_eq_blk (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mnv4Pre3 N f w x = mnv4Blk6 N f w none x := by
  rw [mnv4Pre3, mnv4Res14aLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, mnv4Pre2_eq_blk]; rfl

theorem mnv4Pre4_eq_blk (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mnv4Pre4 N f w x = mnv4Blk10 N f w none x := by
  rw [mnv4Pre4, mnv4Res14bLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, mnv4Pre3_eq_blk]; rfl

theorem mnv4Pre5_eq_blk (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mnv4Pre5 N f w x = mnv4Blk15 N f w none x := by
  rw [mnv4Pre5, mnv4Res7aLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply, mnv4Pre4_eq_blk]; rfl

theorem mnv4Pre6_eq_blk (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mnv4Pre6 N f w x = mnv4Blk21 N f w none x := by
  rw [mnv4Pre6, mnv4Res7bLayer, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply, CertLayer.comp_fwd_apply,
    mnv4Pre5_eq_blk]; rfl

/-- **Without stochastic depth the forward is the classifier-dropout one** — so at `none`/`none`
    it is `mobilenetv4ForwardBFull` (`mobilenetv4ForwardBFullDoOpt_none`). -/
theorem mobilenetv4ForwardBFullDropOpt_none (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (cd : Option (Vec (N * 1280))) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mobilenetv4ForwardBFullDropOpt N f w none cd x = mobilenetv4ForwardBFullDoOpt N f w cd x := by
  rw [mobilenetv4ForwardBFullDropOpt, mobilenetv4ForwardBFullDoOpt, mnv4Pre6_eq_blk]

/-- **With neither regulariser the forward is `mobilenetv4ForwardBFull`**, the drop-free
    artifacts' — so every tie stated at `mobilenetv4ForwardBFullDropOpt … none none` is the
    statement on the plain net. -/
theorem mobilenetv4ForwardBFullDropOpt_none_none (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mobilenetv4ForwardBFullDropOpt N f w none none x = mobilenetv4ForwardBFull N f w x := by
  rw [mobilenetv4ForwardBFullDropOpt_none, mobilenetv4ForwardBFullDoOpt_none]

/-- **At rendered masks the forward is `MobileNetV4FullBDrop`'s**, the one its typed graph
    denotes (`mnv4FwdGraphBFullDrop_faithful`). -/
theorem mobilenetv4ForwardBFullDropOpt_some (N f : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (s : Fin 18 → Vec N) (m : Vec (N * 1280)) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) :
    mobilenetv4ForwardBFullDropOpt N f w (some s) (some m) x
      = mobilenetv4ForwardBFullDrop N f w s m x := by
  -- one `rfl` per resolution group and one for the head, each at a variable input
  have h2 : mnv4Blk2 N f w (some s) x = mnv4Res28Drop N f w s (mnv4Pre1 N f w x) := rfl
  have h6 : mnv4Blk6 N f w (some s) x
      = mnv4Res14aDrop N f w s (mnv4Blk2 N f w (some s) x) := rfl
  have h10 : mnv4Blk10 N f w (some s) x
      = mnv4Res14bDrop N f w s (mnv4Blk6 N f w (some s) x) := rfl
  have h15 : mnv4Blk15 N f w (some s) x
      = mnv4Res7aDrop N f w s (mnv4Blk10 N f w (some s) x) := rfl
  have h21 : mnv4Blk21 N f w (some s) x
      = mnv4Res7bDrop N f w s (mnv4Blk15 N f w (some s) x) := rfl
  have hh : ∀ v : Vec (N * (256 * f * f)), (mnv4HeadStackDo N f w (some m)).fwd v
      = batchMap N (dense w.Wd w.bd) (dropout m
          (mnv4HeadFeatB N f f w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt v)) :=
    fun _ => rfl
  rw [mobilenetv4ForwardBFullDropOpt, mobilenetv4ForwardBFullDrop, hh, h21, h15, h10, h6, h2]

/-- **And the loss cotangent `g` is instantiated: MNv4's is the label-smoothed softmax chain.**
    Row by row, the six-op chain `MobileNetV4RenderB` emits (`softmaxRow → subB → scaleB → addVB →
    shiftB → divConstB`, α = 0.1) is `(1/B)·∂/∂logits` of soft-target cross-entropy against the
    SMOOTHED target, at that example's real logits.

    `softmaxRow` at `m := 1` — the `rowB`/`unrowB` spelling ResNet-34 and ResNet-50 use, NOT
    ConvNeXt's and ViT's `expe → softmaxDiv` at the plain `N·K` width. Read off the render's own
    lines rather than assumed: those two take different lemmas (`smoothedLossCotGraph` here,
    `smoothedLossCotGraphDiv` there) and nothing in the types tells them apart.

    MNv4 ships one loss — there is no BCE twin to state, where ResNet-50 needed both. The only
    hypothesis is that the example's target sums to 1: a one-hot, or mixup's convex combination. -/
theorem mnv4_lossCot_is_smoothedCE_grad (N f : Nat) {nCls : Nat}
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : Mnv4BWeights nCls)
    (sd : Option (Fin 18 → Vec N)) (cd : Option (Vec (N * 1280)))
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * f))))) * (2 * (2 * (2 * (2 * (2 * f)))))))) (t : Vec (N * (1 * nCls)))
    (n : Fin N) (j : Fin nCls)
    (ht : ∑ k : Fin nCls, Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1) k = 1) :
    den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
          (rowB N nCls (mobilenetv4ForwardBFullDropOpt N f w sd cd x)) t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec nCls => fun _ : Fin 1 =>
            softCE nCls (smoothTarget nCls α
              (Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1))) z')
          (Mat.unflatten (batchSlice N (1 * nCls)
            (rowB N nCls (mobilenetv4ForwardBFullDropOpt N f w sd cd x)) n) (0 : Fin 1)) j 0) / B :=
  smoothedLossCotGraph_row N nCls α B aStr negAK bStr logN ohN _ t n j ht

end Proofs.Mnv4TieB
