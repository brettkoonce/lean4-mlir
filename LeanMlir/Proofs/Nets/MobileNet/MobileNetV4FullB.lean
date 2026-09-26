import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4BackB0
import LeanMlir.Proofs.Foundation.HeadLayers

/-! # MobileNetV4-Conv-M at batch BatchNorm — the whole net's forward and graph

`MobileNetV4BackB0.lean` is the block and stage level — every UIB family, the stride-2 form, the
fused stage, the head, the table-driven `k = 0` dispatch and the row-typed `UibParams`. This file is
the net level: a net-level ℝ forward at the 21-row Conv-M table (`mobilenetv4ForwardBFull`), the
typed StableHLO graph over it at `mnv4FwdChainB`'s own tokens (`mnv4FwdGraphBFull`), and their
faithfulness (`mnv4FwdGraphBFull_faithful`). `ResNet50FullB.lean` is the file this one mirrors.

**This is timm's `mobilenetv4_conv_medium`** (1.0.28, the pinned spec): the post-DW carries each
downsample's stride, the pre-DW is BN only, stage 0 is relu, the head pools before `conv_head`,
the stem pads symmetrically. The artifacts are pinned to that function by three gates:
`scripts/parity/mnv4_timm_parity.py` (the JAX reference = timm on shared weights),
`scripts/parity/mnv4_forward_tie.py` (the render = the JAX reference) and
`scripts/parity/grad_tie.py --net mnv4` (the render's backward = `jax.grad` of it).

No accuracy is quoted for this net.

## The trunk is five `CertLayer` groups

Each resolution group — rows 1–2, 3–6, 7–10, 11–15, 16–21 — is assembled with `CertLayer.comp`
and `CertLayer.residual` directly, as are the fused stage and the head, and inside a group:

* `.fwd` **is** that group's forward — no second definition to keep in step;
* `.ok` **is** its smoothness hypothesis, conjoined at exactly the right activations by `comp`
  rather than written out (the net has 38 relu clauses, counted by `MobileNetV4FullBSeal`'s
  `#guard`; none of them is written here);
* `.vjp` **is** its `HasVJPAt` (`MobileNetV4FullBVJP.lean` chains them); and
* `.faithful` **is** its backward-graph faithfulness.

The groups are composed by a prefix chain (`mnv4Pre0` … `mnv4Pre6`), not by one more
`CertLayer.comp`; the section before `mnv4Pre0` says why. `Mnv4SmoothAt` therefore binds one
`.ok` per group plus the stem's clause (eight fields), and no `0 < ε` hypothesis, because those
live inside the weight records.

**The stem sits outside the chain**, as EfficientNet-B0's does. `CertLayer` demands a backward
graph, and no render emits a gradient into `%x`: the artifact's backward ends at the stem conv's
weight gradient. So `mnv4StemB` is a plain function here (`cbReluStridedB` at the stem's widths),
its VJP is `cbReluStridedBHasVJPAt`, and the net-level VJP composes the two with `vjpCompAt`.

## Conventions this net runs at

| | |
|---|---|
| depth | 21 UIB blocks + the fused stage; 13 ExtraDW / 4 ConvNeXt-like / 4 FFN, and **no IB** |
| ladder | 224 →(stem s2) 112 →(fused s2) 56 →(blk1) 28 →(blk3) 14 →(blk11) 7 → GAP |
| channels | 32 → 48 → 80 → 160 → 256, head 256 → 960 → 1280 |
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N·h·w`) at all 77 sites |
| activation | **relu**, not relu6 (MobileNetV2 sits one file over and uses relu6), after every expand, post-DW, the stem, stage 0 and both head convs; the pre-DW and the project are BN only |
| padding | SYMMETRIC `(k-1)/2` at every strided site — the stem, stage 0, the three strided post-DWs (timm) |
| stride | the three stride-2 rows (1, 3, 11) stride their POST-DW (timm's `dw_mid`); the pre-DW and the expand run at the input resolution |
| head | 1×1 256 → 960 conv-bn-relu at 7×7, GAP, then `conv_head` 960 → 1280 conv-bn-relu on the pooled `[N, 960, 1, 1]` (its BN over the batch alone), then dense |
| census | **233** parameter slots at `nCls = 10` (8,447,322 scalars; 9,715,512 at 1000), bias-free by construction |
| artifacts | `mnv4_fwd`, `mnv4in_fwd`, and the f32 224×224 train steps (`mnv4_adam_train_step`, `mnv4in_adam64`, `mnv4in_adamdp64`). Not this graph: the `bf16` train steps, the frozen-statistics evals (`mnv4{,in}_fwd_eval`), the 256×256 eval `mnv4in_fwd_eval_s256`, and the classifier-dropout variants `mnv4in_emaacc{,dp}8x128wxdowd005bf16` (a `%do` operand) |

`N` stays a binder throughout, as at r34/R50. On the data-parallel artifacts the render's `N` is
the per-replica batch and BatchNorm is synchronised across replicas; `MobileNetV4SyncB.lean` is
this file's twin for them: replica `r`'s forward graph denotes shard `r` of
`mobilenetv4ForwardBFull (R * N)`, this file's forward at the global batch. Unlike R50 there is
no resolution binder `q`: the statements are at 224×224.

**Rows 4/5/7, 8/10, 12/18, 13/14 and 15/19/20 are shape-identical** (same `ic, oc, expand,
preDWk, postDWk, h`), so their `UibParams` records have the same type and swapping their weights
typechecks. Typing pins shape, not identity; what pins identity is the SSA names the forward graph
writes (`%u4qW` vs `%u10qW`), which is why the graph reads
its names from `s.p` off the table rather than taking them as arguments.

Checked against the committed bytes: `verified_mlir/mnv4_fwd.mlir`'s signature is **234
arguments = `%x` + 233 parameters**, and every name this file writes appears there.
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The block table, one row per constant
-- ════════════════════════════════════════════════════════════════

/-! These are `abbrev`s, and the rows are named rather than indexed. `UibParams (mnv4Blocks[3]!)`
in a type would force `whnf` through `List.get!` at every use; a named reducible constant reduces
to its projections directly, which is what lets `CertLayer.comp` line up `2 * 28` with `56` across
a stride join without a single transport.

The `#guard` below is the whole safety of that move: these 21 constants are pinned to
`mnv4Blocks` — the ONE table `mnv4FwdChainB`, the backward, the parameter signature and the BN stat
list all fold over — so a typo here is a build failure rather than a proof about a different net. -/

abbrev mnv4Row1 : UibSpec := ⟨"1",  48,  80, 4, 3, 5, 28, true⟩  -- ExtraDW, stride 2
abbrev mnv4Row2 : UibSpec := ⟨"2",  80,  80, 2, 3, 3, 28, false⟩  -- ExtraDW
abbrev mnv4Row3 : UibSpec := ⟨"3",  80, 160, 6, 3, 5, 14, true⟩  -- ExtraDW, stride 2
abbrev mnv4Row4 : UibSpec := ⟨"4", 160, 160, 4, 3, 3, 14, false⟩  -- ExtraDW
abbrev mnv4Row5 : UibSpec := ⟨"5", 160, 160, 4, 3, 3, 14, false⟩  -- ExtraDW
abbrev mnv4Row6 : UibSpec := ⟨"6", 160, 160, 4, 3, 5, 14, false⟩  -- ExtraDW
abbrev mnv4Row7 : UibSpec := ⟨"7", 160, 160, 4, 3, 3, 14, false⟩  -- ExtraDW
abbrev mnv4Row8 : UibSpec := ⟨"8", 160, 160, 4, 3, 0, 14, false⟩  -- ConvNeXt-like
abbrev mnv4Row9 : UibSpec := ⟨"9", 160, 160, 2, 0, 0, 14, false⟩  -- FFN
abbrev mnv4Row10 : UibSpec := ⟨"10", 160, 160, 4, 3, 0, 14, false⟩  -- ConvNeXt-like
abbrev mnv4Row11 : UibSpec := ⟨"11", 160, 256, 6, 5, 5,  7, true⟩  -- ExtraDW, stride 2
abbrev mnv4Row12 : UibSpec := ⟨"12", 256, 256, 4, 5, 5,  7, false⟩  -- ExtraDW
abbrev mnv4Row13 : UibSpec := ⟨"13", 256, 256, 4, 3, 5,  7, false⟩  -- ExtraDW
abbrev mnv4Row14 : UibSpec := ⟨"14", 256, 256, 4, 3, 5,  7, false⟩  -- ExtraDW
abbrev mnv4Row15 : UibSpec := ⟨"15", 256, 256, 4, 0, 0,  7, false⟩  -- FFN
abbrev mnv4Row16 : UibSpec := ⟨"16", 256, 256, 4, 3, 0,  7, false⟩  -- ConvNeXt-like
abbrev mnv4Row17 : UibSpec := ⟨"17", 256, 256, 2, 3, 5,  7, false⟩  -- ExtraDW
abbrev mnv4Row18 : UibSpec := ⟨"18", 256, 256, 4, 5, 5,  7, false⟩  -- ExtraDW
abbrev mnv4Row19 : UibSpec := ⟨"19", 256, 256, 4, 0, 0,  7, false⟩  -- FFN
abbrev mnv4Row20 : UibSpec := ⟨"20", 256, 256, 4, 0, 0,  7, false⟩  -- FFN
abbrev mnv4Row21 : UibSpec := ⟨"21", 256, 256, 2, 5, 0,  7, false⟩  -- ConvNeXt-like

-- ⭐⭐ The 21 constants ARE `mnv4Blocks`, in order. Nothing below can be about a different net.
#guard mnv4Blocks = [
  mnv4Row1, mnv4Row2, mnv4Row3, mnv4Row4, mnv4Row5, mnv4Row6, mnv4Row7, mnv4Row8, mnv4Row9,
  mnv4Row10, mnv4Row11, mnv4Row12, mnv4Row13, mnv4Row14, mnv4Row15, mnv4Row16, mnv4Row17,
  mnv4Row18, mnv4Row19, mnv4Row20, mnv4Row21
  ]

-- The families these rows denote, recomputed here rather than restated: 13 / 4 / 4, no IB.
#guard (mnv4Blocks.filter (fun s => s.family == .extraDW)).length = 13
#guard (mnv4Blocks.filter (fun s => s.family == .ffn)).length = 4
#guard (mnv4Blocks.filter (fun s => s.family == .convNeXtLike)).length = 4
#guard mnv4Blocks.all (fun s => s.family != .ib)
-- The three stride-2 rows; each has a post-DW to carry the stride and a pre-DW in its slot.
#guard (mnv4Blocks.filter (fun s => s.stride2)).map (·.p) = ["1", "3", "11"]
#guard (mnv4Blocks.filter (fun s => s.stride2)).all (fun s => s.preDWk != 0 && s.postDWk != 0)

-- ════════════════════════════════════════════════════════════════
-- § The weights, typed by their table rows
-- ════════════════════════════════════════════════════════════════

/-- **Every MobileNetV4-Conv-M parameter**, generic in the class count so one statement covers the
    10-class Imagenette artifacts and the 1000-class `mnv4in` ones.

    The 21 block fields are `UibParams mnv4Row{k}` — a record whose every width is a *projection
    of its row*, so a record that disagrees with its row **cannot be constructed** and the forward
    below needs no side conditions on widths. That is strictly stronger than ResNet-50's
    `R50IdW`/`R50ProjW`, which are typed by loose `{mid oc}` binders. It still does not pin
    identity between shape-identical rows (4/5/7, 8/10, 12/18, 13/14, 15/19/20) — see the header.

    The `0 < ε` obligations live INSIDE the records (`UibParams`'s `hq he hd hz`), so the stem,
    the fused stage and the head carry theirs as fields too. R50 keeps a separate `R50IdPos`
    bundle; matching `UibParams` here means the whole-net VJP binds no epsilon hypotheses at all.

    Every conv is bias-free — both renders bake `convBias := false` and bind each bias to the
    `%zb{c}` zero the prelude declares — but the records still carry a `b` slot because the stage
    vocabulary takes one. Those fields are `∀`-quantified over; `bias = 0` is one instance. Field
    names are the render's own SSA prefixes, so a reader can match a parameter to its emitted name
    without a table. -/
structure Mnv4BWeights (nCls : Nat) where
  /-- stem `%sW`/`%sg`/`%sbt`: 3×3/s2, symmetric, 3 → 32, 224 → 112. -/
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sE : ℝ
  hsE : 0 < sE
  sg : Vec 32
  sbt : Vec 32
  /-- fused stage `%f0cW`: 3×3/s2 symmetric, 32 → 128, 112 → 56, then relu. -/
  f0cW : Kernel4 128 32 3 3
  f0cb : Vec 128
  f0cE : ℝ
  hf0cE : 0 < f0cE
  f0cg : Vec 128
  f0cbt : Vec 128
  /-- fused stage `%f0pW`: the 1×1 project, 128 → 48, no activation. -/
  f0pW : Kernel4 48 128 1 1
  f0pb : Vec 48
  f0pE : ℝ
  hf0pE : 0 < f0pE
  f0pg : Vec 48
  f0pbt : Vec 48
  /-- block 1: `%u1*`, 48 → 80, expand 4, dw 3/5, at 28×28. -/
  b1 : UibParams mnv4Row1
  /-- block 2: `%u2*`, 80 → 80, expand 2, dw 3/3, at 28×28. -/
  b2 : UibParams mnv4Row2
  /-- block 3: `%u3*`, 80 → 160, expand 6, dw 3/5, at 14×14. -/
  b3 : UibParams mnv4Row3
  /-- block 4: `%u4*`, 160 → 160, expand 4, dw 3/3, at 14×14. -/
  b4 : UibParams mnv4Row4
  /-- block 5: `%u5*`, 160 → 160, expand 4, dw 3/3, at 14×14. -/
  b5 : UibParams mnv4Row5
  /-- block 6: `%u6*`, 160 → 160, expand 4, dw 3/5, at 14×14. -/
  b6 : UibParams mnv4Row6
  /-- block 7: `%u7*`, 160 → 160, expand 4, dw 3/3, at 14×14. -/
  b7 : UibParams mnv4Row7
  /-- block 8: `%u8*`, 160 → 160, expand 4, dw 3/0, at 14×14. -/
  b8 : UibParams mnv4Row8
  /-- block 9: `%u9*`, 160 → 160, expand 2, dw 0/0, at 14×14. -/
  b9 : UibParams mnv4Row9
  /-- block 10: `%u10*`, 160 → 160, expand 4, dw 3/0, at 14×14. -/
  b10 : UibParams mnv4Row10
  /-- block 11: `%u11*`, 160 → 256, expand 6, dw 5/5, at 7×7. -/
  b11 : UibParams mnv4Row11
  /-- block 12: `%u12*`, 256 → 256, expand 4, dw 5/5, at 7×7. -/
  b12 : UibParams mnv4Row12
  /-- block 13: `%u13*`, 256 → 256, expand 4, dw 3/5, at 7×7. -/
  b13 : UibParams mnv4Row13
  /-- block 14: `%u14*`, 256 → 256, expand 4, dw 3/5, at 7×7. -/
  b14 : UibParams mnv4Row14
  /-- block 15: `%u15*`, 256 → 256, expand 4, dw 0/0, at 7×7. -/
  b15 : UibParams mnv4Row15
  /-- block 16: `%u16*`, 256 → 256, expand 4, dw 3/0, at 7×7. -/
  b16 : UibParams mnv4Row16
  /-- block 17: `%u17*`, 256 → 256, expand 2, dw 3/5, at 7×7. -/
  b17 : UibParams mnv4Row17
  /-- block 18: `%u18*`, 256 → 256, expand 4, dw 5/5, at 7×7. -/
  b18 : UibParams mnv4Row18
  /-- block 19: `%u19*`, 256 → 256, expand 4, dw 0/0, at 7×7. -/
  b19 : UibParams mnv4Row19
  /-- block 20: `%u20*`, 256 → 256, expand 4, dw 0/0, at 7×7. -/
  b20 : UibParams mnv4Row20
  /-- block 21: `%u21*`, 256 → 256, expand 2, dw 5/0, at 7×7. -/
  b21 : UibParams mnv4Row21
  /-- head conv 1 `%h1W`: 1×1, 256 → 960, at 7×7. -/
  h1W : Kernel4 960 256 1 1
  h1b : Vec 960
  h1E : ℝ
  hh1E : 0 < h1E
  h1g : Vec 960
  h1bt : Vec 960
  /-- head conv 2 `%hW` (timm's `conv_head`): 1×1, 960 → 1280, on the POOLED features. -/
  hW : Kernel4 1280 960 1 1
  hb : Vec 1280
  hE : ℝ
  hhE : 0 < hE
  hg : Vec 1280
  hbt : Vec 1280
  /-- classifier `%Wd`/`%bd`, after `conv_head`. -/
  Wd : Mat 1280 nCls
  bd : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § The stem — outside the chain, and it cannot be otherwise
-- ════════════════════════════════════════════════════════════════

/-- MNv4's stem forward: 3×3/s2 conv, SYMMETRIC padding (timm's `conv_stem`) → batch BN → relu.
    The XLA-`SAME` phase (`flatConvStride2Xla`, pads (0,1) at 224) also gives 112×112, so only a
    forward on shared weights tells the two apart. -/
@[reducible] noncomputable def mnv4StemB (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w εs γs βs ∘
    batchMap N (flatConvStride2 Ws bs)

-- ════════════════════════════════════════════════════════════════
-- § The trunk, as five certified resolution groups
-- ════════════════════════════════════════════════════════════════

/-- **The fused stage (stage 0)**: 3×3/s2 symmetric conv-bn-relu 32 → 128 at 112 → 56, then the
    1×1 project 128 → 48 (timm's `EdgeResidual`). -/
noncomputable def mnv4FusedStack (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (32 * 112 * 112)) (N * (48 * 56 * 56)) :=
  mnv4FusedStage N
    (cbReluStridedLayer (h := 56) (w := 56) N w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt)
    (projLayer (h := 56) (w := 56) N w.f0pW w.f0pb w.f0pE w.hf0pE w.f0pg w.f0pbt)

/-- **The head**, timm's order: 1×1 256 → 960 conv-bn-relu at 7×7, GAP, `conv_head` 1×1
    960 → 1280 conv-bn-relu on the pooled features (its BN over the batch alone), classifier. -/
noncomputable def mnv4HeadStack (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (256 * 7 * 7)) (N * nCls) :=
  mnv4Head N (cbReluLayer (h := 7) (w := 7) N w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt)
    (gapLayer N (c := 960) (h := 7) (w := 7))
    (cbReluLayer (h := 1) (w := 1) N w.hW w.hb w.hE w.hhE w.hg w.hbt)
    (denseLayer N w.Wd w.bd)

/-! **The trunk is built in groups, and that is a proof-engineering requirement.** One 24-stage
`CertLayer` elaborates fine — it is the graph faithfulness proof over it that does not: the
whole-net rewrite chain produces a term whose kernel check ends in a deterministic timeout. Split
at the net's own resolution boundaries, each group's proof is small, and the whole-net theorem is
six rewrites over them. The grouping is the ladder a reader already knows — 56, 28, 14, 7 — so
it costs nothing in readability and buys a bounded proof. -/

/-- Trunk group **Res28** — rows 1–2: the 56→28 reduction and the block that follows it. -/
noncomputable def mnv4Res28Layer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (48 * 56 * 56)) (N * (80 * 28 * 28)) :=
  (mnv4StridedBodyOfRow N mnv4Row1 w.b1).comp
      (CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2))

/-- Trunk group **Res14a** — rows 3–6: the 28→14 reduction, then three ExtraDW blocks. -/
noncomputable def mnv4Res14aLayer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (80 * 28 * 28)) (N * (160 * 14 * 14)) :=
  (mnv4StridedBodyOfRow N mnv4Row3 w.b3).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row4 w.b4)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row5 w.b5)).comp
      (CertLayer.residual (mnv4BodyOfRow N mnv4Row6 w.b6))))

/-- Trunk group **Res14b** — rows 7–10 at 14×14: ExtraDW, ConvNeXt, FFN, ConvNeXt — three families in four blocks. -/
noncomputable def mnv4Res14bLayer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (160 * 14 * 14)) (N * (160 * 14 * 14)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row7 w.b7)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row8 w.b8)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row9 w.b9)).comp
      (CertLayer.residual (mnv4BodyOfRow N mnv4Row10 w.b10))))

/-- Trunk group **Res7a** — rows 11–15: the last reduction (14→7), then four blocks at 7×7. -/
noncomputable def mnv4Res7aLayer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (160 * 14 * 14)) (N * (256 * 7 * 7)) :=
  (mnv4StridedBodyOfRow N mnv4Row11 w.b11).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row12 w.b12)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row13 w.b13)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row14 w.b14)).comp
      (CertLayer.residual (mnv4BodyOfRow N mnv4Row15 w.b15)))))

/-- Trunk group **Res7b** — rows 16–21 at 7×7: the net's tail. -/
noncomputable def mnv4Res7bLayer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (256 * 7 * 7)) (N * (256 * 7 * 7)) :=
  (CertLayer.residual (mnv4BodyOfRow N mnv4Row16 w.b16)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row17 w.b17)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row18 w.b18)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row19 w.b19)).comp
      ((CertLayer.residual (mnv4BodyOfRow N mnv4Row20 w.b20)).comp
      (CertLayer.residual (mnv4BodyOfRow N mnv4Row21 w.b21))))))

/-! **The seven groups are composed by a prefix chain, not by one more `CertLayer.comp`.**

A `mnv4NetLayer := fused.comp (res28.comp (… .comp head))` elaborates fine and reads beautifully.
But every downstream statement then has to peel `CertLayer.comp` to get at `.fwd`, and at MNv4's
LITERAL resolutions that peel is fatal: `(L₁.comp L₂).fwd = L₂.fwd ∘ L₁.fwd` is `rfl`, yet
discharging it at these instances — by `rfl`, by `simp only [CertLayer.comp_fwd]`, inside the
graph-faithfulness capstone or in a standalone lemma — ends in a `(kernel) deterministic timeout`.
The groups' own five-stage `comp` chains are fine; it is composing the
compositions, under something that can start unfolding, that is not.

So the top level is seven named prefixes and the forward is their nest — ResNet-50's shape at
seven stages instead of eighteen. What that costs is the hypothesis bundle: `Mnv4SmoothAt` binds
one `.ok` per group plus the stem's clause (eight fields) rather than one for the whole trunk.
Each group's `.ok` is still the conjunction `CertLayer.comp` assembled from its blocks'
conditions at their own activations, so the 38 relu clauses are never written down one by one,
and no `0 < ε` hypothesis appears at all. R50's two apex bundles carry 35 fields.

More generally, a net whose resolutions are literals cannot afford the proof idioms a net with a
resolution binder can: ResNet-50's `q` keeps `den` stuck, MNv4's 224/112/56/28/14/7 let it run.
The stem graph builder's genericity, the prefix chain and the capstone's `rw` in place of
`simp only` all trace to that. -/

/-- Prefix 0: the stem's output. -/
@[reducible] noncomputable def mnv4Pre0 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (32 * 112 * 112)) :=
  mnv4StemB N 112 112 w.sW w.sb w.sE w.sg w.sbt x

/-- Prefix 1: through the fused stage, at 56×56. -/
@[reducible] noncomputable def mnv4Pre1 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (48 * 56 * 56)) :=
  (mnv4FusedStack N w).fwd (mnv4Pre0 N w x)

/-- Prefix 2: through rows 1–2, at 28×28. -/
@[reducible] noncomputable def mnv4Pre2 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (80 * 28 * 28)) :=
  (mnv4Res28Layer N w).fwd (mnv4Pre1 N w x)

/-- Prefix 3: through rows 3–6, at 14×14. -/
@[reducible] noncomputable def mnv4Pre3 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (mnv4Res14aLayer N w).fwd (mnv4Pre2 N w x)

/-- Prefix 4: through rows 7–10, still at 14×14. -/
@[reducible] noncomputable def mnv4Pre4 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (160 * 14 * 14)) :=
  (mnv4Res14bLayer N w).fwd (mnv4Pre3 N w x)

/-- Prefix 5: through rows 11–15, at 7×7. -/
@[reducible] noncomputable def mnv4Pre5 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (mnv4Res7aLayer N w).fwd (mnv4Pre4 N w x)

/-- Prefix 6: through rows 16–21 — the whole trunk below the head. -/
@[reducible] noncomputable def mnv4Pre6 (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * (256 * 7 * 7)) :=
  (mnv4Res7bLayer N w).fwd (mnv4Pre5 N w x)

/-- **The full batch-BN MobileNetV4-Conv-M**, `N*(3*224*224) → N*nCls`.

    The stem, the fused stage, the five resolution groups, the head. Every block inside those
    groups is `mnv4BodyOfRow` at its own row, so the `k = 0` dispatch is READ from `mnv4Blocks`
    rather than chosen here — the property `MobileNetV4BackB0.lean`'s dispatch section exists to
    establish, now carried to the net. The eighteen skips are `CertLayer.residual`, which
    typechecks with no transport because `s.oc` and `s.ic` reduce to the same literal at every
    stride-1 row (guarded there). -/
noncomputable def mobilenetv4ForwardBFull (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * nCls) :=
  (mnv4HeadStack N w).fwd (mnv4Pre6 N w x)

-- ⭐ The stem's own arithmetic, checked rather than asserted: its input is the 224-px image and
-- its output is what the fused stage reads. A wrong nest depth is well-typed at a variable.
#guard 2 * 112 == 224
#guard 2 * 56 == 112
#guard 2 * 28 == 56
#guard 2 * 14 == 28
#guard 2 * 7 == 14

-- ⭐ `mobilenetv4ForwardBFull` really does bind at the literal 224-px image type.
example (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardBFull N w x = mobilenetv4ForwardBFull N w x := rfl

-- ════════════════════════════════════════════════════════════════
-- § T2 — the typed forward graph, at `mnv4FwdChainB`'s own tokens
--   `.batchOp` of `.convStrided` (stem, fused) / `.conv` / `.depthwise` / `.depthwiseStrided` /
--   `.relu` / `.gap` / `.dense`, `.bnBatchF` for the batch-coupled norm, `.addVB` for the
--   residual add, and `castIdx` for the head's two `1×1` relabellings (no text).
--
--   ⚠ Every SSA name is read off the ROW (`s.p`), never taken as an argument. That is what pins
--   identity between shape-identical rows: rows 4, 5 and 10 have the same `UibParams` type, so
--   only `%u4qW` vs `%u5qW` vs `%u10qW` tells them apart, and here those come from the table.
--
--   ⚠ Bias operands are `%zb{c}`, the shared zero constant every bias folds into its BatchNorm
--   and binds to. `MobileNetV4RenderB` has no `convBias` flag at all, so this is the only name
--   this net emits.
-- ════════════════════════════════════════════════════════════════

/-- Stem graph: 3×3/s2 symmetric conv → batch BN → relu.

    **Generic in the widths, and that is an elaboration requirement, not style.**
    Pinning `ic := 3, oc := 32, h := 112` here makes the conv's `den_batchOp` `rfl` a claim
    about concrete 150528- and 401408-element tensors, and the KERNEL tries to reduce it: the
    lemma takes over a minute and then fails with `(kernel) deterministic timeout`. Proven at
    binders it takes two seconds, and applying it at the net's literals is free — instantiating a
    proven lemma, not proving one. `mnv2StemGraphB` and `r50StemGraphB` are generic for the same
    reason, which is easy to read as a stylistic habit and is not. -/
def mnv4StemGraphB (epsStr : String) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .batchOp (N := N) (.relu (n := oc * h * w))
    (.bnBatchF "%sg" "%sbt" epsStr εs γs βs
      (.batchOp (N := N) (.convStrided (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) e))

theorem mnv4StemGraphB_faithful (epsStr : String) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mnv4StemGraphB epsStr N h w Ws bs εs γs βs e)
      = mnv4StemB N h w Ws bs εs γs βs (den e) := by
  unfold mnv4StemGraphB
  simp only [mnv4StemB, ↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, denOp,
    den_bnBatchF, Function.comp_apply]

/-- Fused stage graph: 3×3/s2 symmetric conv → BN → relu → 1×1 project → BN. No skip.
    Generic in the widths, for the reason `mnv4StemGraphB` records. -/
def mnv4FusedGraphB (epsStr : String) (N h w : Nat) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF "%f0pg" "%f0pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) "%f0pW" s!"%zb{oc}" Wp bp)
      (.batchOp (N := N) (.relu (n := mid * h * w))
        (.bnBatchF "%f0cg" "%f0cbt" epsStr εc γc βc
          (.batchOp (N := N) (.convStrided (h := h) (w := w) "%f0cW" s!"%zb{mid}" Wc bc) e))))

theorem mnv4FusedGraphB_faithful (epsStr : String) (N h w : Nat) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (hεc : 0 < εc) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mnv4FusedGraphB epsStr N h w Wc bc εc γc βc Wp bp εp γp βp e)
      = (mnv4FusedStage N (cbReluStridedLayer (h := h) (w := w) N Wc bc εc hεc γc βc)
          (projLayer (h := h) (w := w) N Wp bp εp hεp γp βp)).fwd (den e) := by
  simp only [mnv4FusedGraphB, mnv4FusedStage, cbReluStridedLayer,
    projLayer, CertLayer.comp_fwd, cbReluStridedB, projB,
    ↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, denOp,
    den_bnBatchF, Function.comp_apply]

/-- **ExtraDW body graph** — both depthwises present, 13 of Conv-M's 21 rows (and all three
    stride-2 ones, whose own builder is below). The BODY only: the identity skip is `.addVB`'d on
    at the call site, which is what keeps the whole-net graph LINEAR instead of duplicating each
    skip's entire input subtree. -/
def mnv4ExtraDWBodyGraphB (epsStr : String) (N : Nat) (s : UibSpec) (p : UibParams s)
    (e : SHlo (N * (s.ic * s.h * s.h))) : SHlo (N * (s.oc * s.h * s.h)) :=
  .bnBatchF s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr p.ez p.gz p.bz2
    (.batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (.bnBatchF s!"%u{s.p}dg" s!"%u{s.p}dbt" epsStr p.ed p.gd p.bd2
          (.batchOp (N := N) (.depthwise (c := s.ic * s.expand) (h := s.h) (w := s.h)
              s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.Wd p.bd)
            (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
              (.bnBatchF s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr p.ee p.ge p.be2
                (.batchOp (N := N)
                  (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
                    s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
                  (.bnBatchF s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr p.eq_ p.gq p.bq2
                    (.batchOp (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
                        s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq)
                      e)))))))))

/-- The ExtraDW body graph denotes the row-typed body's forward — **generic in the row**, so one
    theorem serves all thirteen. The two hypotheses are exactly the dispatch conditions
    `mnv4PreDWSlot`/`mnv4PostDWSlot` branch on, discharged by `decide` at each concrete row. -/
theorem mnv4ExtraDWBodyGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hq : s.preDWk ≠ 0) (hd : s.postDWk ≠ 0)
    (e : SHlo (N * (s.ic * s.h * s.h))) :
    den (mnv4ExtraDWBodyGraphB epsStr N s p e) = (mnv4BodyOfRow N s p).fwd (den e) := by
  simp only [mnv4ExtraDWBodyGraphB, mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot,
    ite_eq_right hq, ite_eq_right hd, mnv4DWBnLayer, mnv4DWReluLayer, cbReluLayer, projLayer,
    CertLayer.comp_fwd, projB, cbReluB, dwbB, dwbReluB, ↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, denOp,
    den_bnBatchF, Function.comp_apply]

/-- **ConvNeXt-like body graph** — pre-DW only, `postDWk = 0`, four of Conv-M's rows (8, 10, 16,
    21). The absent depthwise emits no tokens, exactly as `mnv4PostDWSlot` inserts `id'`: the
    `UibParams` record still carries a degenerate `DepthwiseKernel _ 0 0` in that slot and this
    graph simply does not read it. A token stated for an absent depthwise would be a `den` of a
    node the artifact does not have. -/
def mnv4ConvNeXtBodyGraphB (epsStr : String) (N : Nat) (s : UibSpec) (p : UibParams s)
    (e : SHlo (N * (s.ic * s.h * s.h))) : SHlo (N * (s.oc * s.h * s.h)) :=
  .bnBatchF s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr p.ez p.gz p.bz2
    (.batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (.bnBatchF s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr p.ee p.ge p.be2
          (.batchOp (N := N) (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
              s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
            (.bnBatchF s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr p.eq_ p.gq p.bq2
              (.batchOp (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
                  s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq)
                e))))))

theorem mnv4ConvNeXtBodyGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hq : s.preDWk ≠ 0) (hd : s.postDWk = 0)
    (e : SHlo (N * (s.ic * s.h * s.h))) :
    den (mnv4ConvNeXtBodyGraphB epsStr N s p e) = (mnv4BodyOfRow N s p).fwd (den e) := by
  simp only [mnv4ConvNeXtBodyGraphB, mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot,
    ite_eq_right hq, ite_eq_left hd, mnv4DWBnLayer, cbReluLayer, projLayer, CertLayer.id'_fwd,
    CertLayer.comp_fwd, projB, cbReluB, dwbB, ↓den_batchOp_relu_eq_reluF, reluF_faithful,
    den_batchOp, denOp, den_bnBatchF, Function.comp_apply]

/-- **FFN body graph** — neither depthwise, four of Conv-M's rows (9, 15, 19, 20): expand,
    project, and nothing else. Both slots are `id'`. -/
def mnv4FfnBodyGraphB (epsStr : String) (N : Nat) (s : UibSpec) (p : UibParams s)
    (e : SHlo (N * (s.ic * s.h * s.h))) : SHlo (N * (s.oc * s.h * s.h)) :=
  .bnBatchF s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr p.ez p.gz p.bz2
    (.batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (.bnBatchF s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr p.ee p.ge p.be2
          (.batchOp (N := N) (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := s.h) (w := s.h)
              s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
            e))))

theorem mnv4FfnBodyGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hq : s.preDWk = 0) (hd : s.postDWk = 0)
    (e : SHlo (N * (s.ic * s.h * s.h))) :
    den (mnv4FfnBodyGraphB epsStr N s p e) = (mnv4BodyOfRow N s p).fwd (den e) := by
  simp only [mnv4FfnBodyGraphB, mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot,
    ite_eq_left hq, ite_eq_left hd, cbReluLayer, projLayer, CertLayer.id'_fwd, CertLayer.comp_fwd,
    projB, cbReluB, ↓den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp, denOp,
    den_bnBatchF, Function.comp_apply]

/-- **Strided block graph** — rows 1, 3 and 11, all ExtraDW. timm strides `dw_mid`: the BN-only
    pre-DW and the expand run at the input resolution `2h`, the post-DW (`.depthwiseStrided`,
    symmetric) takes it to `h`, the project runs at `h`. No skip: `ic ≠ oc` at all three, so the
    block IS the body and there is no `.addVB`. -/
def mnv4StridedGraphB (epsStr : String) (N : Nat) (s : UibSpec) (p : UibParams s)
    (e : SHlo (N * (s.ic * (2 * s.h) * (2 * s.h)))) : SHlo (N * (s.oc * s.h * s.h)) :=
  .bnBatchF s!"%u{s.p}pg" s!"%u{s.p}pbt" epsStr p.ez p.gz p.bz2
    (.batchOp (N := N) (.conv (ic := s.ic * s.expand) (oc := s.oc) (h := s.h) (w := s.h)
        s!"%u{s.p}pW" s!"%zb{s.oc}" p.Wz p.bz)
      (.batchOp (N := N) (.relu (n := s.ic * s.expand * s.h * s.h))
        (.bnBatchF s!"%u{s.p}dg" s!"%u{s.p}dbt" epsStr p.ed p.gd p.bd2
          (.batchOp (N := N) (.depthwiseStrided (c := s.ic * s.expand) (h := s.h) (w := s.h)
              s!"%u{s.p}dW" s!"%zb{s.ic * s.expand}" p.Wd p.bd)
            (.batchOp (N := N) (.relu (n := s.ic * s.expand * (2 * s.h) * (2 * s.h)))
              (.bnBatchF s!"%u{s.p}eg" s!"%u{s.p}ebt" epsStr p.ee p.ge p.be2
                (.batchOp (N := N)
                  (.conv (ic := s.ic) (oc := s.ic * s.expand) (h := 2 * s.h) (w := 2 * s.h)
                    s!"%u{s.p}eW" s!"%zb{s.ic * s.expand}" p.We p.be)
                  (.bnBatchF s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr p.eq_ p.gq p.bq2
                    (.batchOp (N := N) (.depthwise (c := s.ic) (h := 2 * s.h) (w := 2 * s.h)
                        s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq)
                      e)))))))))

theorem mnv4StridedGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hq : s.preDWk ≠ 0)
    (e : SHlo (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    den (mnv4StridedGraphB epsStr N s p e) = (mnv4StridedBodyOfRow N s p).fwd (den e) := by
  simp only [mnv4StridedGraphB, mnv4StridedBodyOfRow, mnv4UibStridedBody, mnv4PreDWSlot,
    ite_eq_right hq, mnv4DWBnLayer, mnv4DWReluStridedLayer, cbReluLayer, projLayer,
    CertLayer.comp_fwd, projB, cbReluB, dwbB, dwbReluBstrided, ↓den_batchOp_relu_eq_reluF,
    reluF_faithful, den_batchOp, denOp,
    den_bnBatchF, Function.comp_apply]

/-- **One skip row's graph: its body's, plus the identity skip.** Trivial as a definition; it
    exists as a barrier to unfolding.

    **Why it is a named combinator and not an inline `.addVB`.** The residual add needs the
    block's input subtree twice, and MNv4 has eighteen of them. Written inline — or behind a `let`
    in the whole-net graph, which `simp only [mnv4FwdGraphBFull]` zeta-expands — the term doubles at
    every skip the moment anything unfolds it, and the whole-net faithfulness proof ends in a
    kernel deterministic timeout.

    Kept folded, with `mnv4SkipGraphB_faithful` rewriting `den (mnv4SkipGraphB body e)` in ONE
    step, `den e` occurs once and the whole-net term stays linear in the depth. R50 never met this:
    its `r50IdGraphB` takes `e` as a binder and duplicates it inside the builder, which has the
    same effect for the same reason. -/
def mnv4SkipGraphB {N n : Nat} (body : SHlo (N * n) → SHlo (N * n)) (e : SHlo (N * n)) :
    SHlo (N * n) :=
  .addVB (body e) e

/-- A skip row denotes `residual` of whatever its body denotes — generic in both, so one
    theorem covers all eighteen and the body's own faithfulness lemma is the only input. -/
theorem mnv4SkipGraphB_faithful {N n : Nat} (body : SHlo (N * n) → SHlo (N * n))
    (f : Vec (N * n) → Vec (N * n))
    (hb : ∀ e' : SHlo (N * n), den (body e') = f (den e')) (e : SHlo (N * n)) :
    den (mnv4SkipGraphB body e) = Proofs.residual f (den e) := by
  simp only [mnv4SkipGraphB, den_addVB, hb, Proofs.residual]

/-- **Head graph**, timm's order: 1×1 conv-BN-relu, GAP, `conv_head` 1×1 conv-BN-relu on the
    pooled features, dense — with the two `castIdx` relabellings `mnv4Head` carries, which emit no
    text. Generic in the widths, for the reason `mnv4StemGraphB` records. -/
def mnv4HeadGraphB (epsStr : String) (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (e : SHlo (N * (c * h * w))) : SHlo (N * nCls) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (castIdx (mnv4_pool11 N oc).symm
      (.batchOp (N := N) (.relu (n := oc * 1 * 1))
        (.bnBatchF "%hg" "%hbt" epsStr ε2 γ2 β2
          (.batchOp (N := N) (.conv (h := 1) (w := 1) "%hW" s!"%zb{oc}" W2 b2)
            (castIdx (mnv4_pool11 N mid)
              (.batchOp (N := N) (.gap (c := mid) (h := h) (w := w))
                (.batchOp (N := N) (.relu (n := mid * h * w))
                  (.bnBatchF "%h1g" "%h1bt" epsStr ε1 γ1 β1
                    (.batchOp (N := N) (.conv (h := h) (w := w) "%h1W" s!"%zb{mid}" W1 b1)
                      e)))))))))

theorem mnv4HeadGraphB_faithful (epsStr : String) (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : SHlo (N * (c * h * w))) :
    den (mnv4HeadGraphB epsStr N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd e)
      = (mnv4Head N (cbReluLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1)
          (gapLayer N (c := mid) (h := h) (w := w))
          (cbReluLayer (h := 1) (w := 1) N W2 b2 ε2 hε2 γ2 β2)
          (denseLayer N Wd bd)).fwd (den e) := by
  simp only [mnv4HeadGraphB, mnv4Head, cbReluLayer, gapLayer, castLayer,
    denseLayer, CertLayer.comp_fwd, cbReluB, ↓den_batchOp_relu_eq_reluF, reluF_faithful,
    den_castIdx, reindexCLM_apply, den_batchOp, denOp, den_bnBatchF, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole graph + faithfulness (T2)
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28**'s graph — rows 1–2: the 56→28 reduction and the block that follows it. -/
def mnv4Res28GraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (48 * 56 * 56))) : SHlo (N * (80 * 28 * 28)) :=
  mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row2 w.b2)
      (mnv4StridedGraphB epsStr N mnv4Row1 w.b1
      (e))

theorem mnv4Res28GraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (48 * 56 * 56))) :
    den (mnv4Res28GraphB N epsStr w e) = (mnv4Res28Layer N w).fwd (den e) := by
  simp only [mnv4Res28GraphB, mnv4Res28Layer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4StridedGraphB_faithful epsStr N mnv4Row1 w.b1 (by decide),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row2 w.b2) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row2 w.b2 (by decide) (by decide)),]

/-- Trunk group **Res14a**'s graph — rows 3–6: the 28→14 reduction, then three ExtraDW blocks. -/
def mnv4Res14aGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (80 * 28 * 28))) : SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row6 w.b6)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row5 w.b5)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row4 w.b4)
      (mnv4StridedGraphB epsStr N mnv4Row3 w.b3
      (e))))

theorem mnv4Res14aGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (80 * 28 * 28))) :
    den (mnv4Res14aGraphB N epsStr w e) = (mnv4Res14aLayer N w).fwd (den e) := by
  simp only [mnv4Res14aGraphB, mnv4Res14aLayer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4StridedGraphB_faithful epsStr N mnv4Row3 w.b3 (by decide),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row4 w.b4) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row4 w.b4 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row5 w.b5) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row5 w.b5 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row6 w.b6) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row6 w.b6 (by decide) (by decide)),]

/-- Trunk group **Res14b**'s graph — rows 7–10 at 14×14: ExtraDW, ConvNeXt, FFN, ConvNeXt — three families in four blocks. -/
def mnv4Res14bGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (160 * 14 * 14))) : SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipGraphB (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row10 w.b10)
      (mnv4SkipGraphB (mnv4FfnBodyGraphB epsStr N mnv4Row9 w.b9)
      (mnv4SkipGraphB (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row8 w.b8)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row7 w.b7)
      (e))))

theorem mnv4Res14bGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (160 * 14 * 14))) :
    den (mnv4Res14bGraphB N epsStr w e) = (mnv4Res14bLayer N w).fwd (den e) := by
  simp only [mnv4Res14bGraphB, mnv4Res14bLayer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row7 w.b7) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row7 w.b7 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row8 w.b8) _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row8 w.b8 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4FfnBodyGraphB epsStr N mnv4Row9 w.b9) _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row9 w.b9 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row10 w.b10) _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row10 w.b10 (by decide) (by decide)),]

/-- Trunk group **Res7a**'s graph — rows 11–15: the last reduction (14→7), then four blocks at 7×7. -/
def mnv4Res7aGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (160 * 14 * 14))) : SHlo (N * (256 * 7 * 7)) :=
  mnv4SkipGraphB (mnv4FfnBodyGraphB epsStr N mnv4Row15 w.b15)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row14 w.b14)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row13 w.b13)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row12 w.b12)
      (mnv4StridedGraphB epsStr N mnv4Row11 w.b11
      (e)))))

theorem mnv4Res7aGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (160 * 14 * 14))) :
    den (mnv4Res7aGraphB N epsStr w e) = (mnv4Res7aLayer N w).fwd (den e) := by
  simp only [mnv4Res7aGraphB, mnv4Res7aLayer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4StridedGraphB_faithful epsStr N mnv4Row11 w.b11 (by decide),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row12 w.b12) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row12 w.b12 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row13 w.b13) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row13 w.b13 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row14 w.b14) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row14 w.b14 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4FfnBodyGraphB epsStr N mnv4Row15 w.b15) _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row15 w.b15 (by decide) (by decide)),]

/-- Trunk group **Res7b**'s graph — rows 16–21 at 7×7: the net's tail. -/
def mnv4Res7bGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (256 * 7 * 7))) : SHlo (N * (256 * 7 * 7)) :=
  mnv4SkipGraphB (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row21 w.b21)
      (mnv4SkipGraphB (mnv4FfnBodyGraphB epsStr N mnv4Row20 w.b20)
      (mnv4SkipGraphB (mnv4FfnBodyGraphB epsStr N mnv4Row19 w.b19)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row18 w.b18)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row17 w.b17)
      (mnv4SkipGraphB (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row16 w.b16)
      (e))))))

theorem mnv4Res7bGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (256 * 7 * 7))) :
    den (mnv4Res7bGraphB N epsStr w e) = (mnv4Res7bLayer N w).fwd (den e) := by
  simp only [mnv4Res7bGraphB, mnv4Res7bLayer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4SkipGraphB_faithful (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row16 w.b16) _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row16 w.b16 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row17 w.b17) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row17 w.b17 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row18 w.b18) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row18 w.b18 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4FfnBodyGraphB epsStr N mnv4Row19 w.b19) _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row19 w.b19 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4FfnBodyGraphB epsStr N mnv4Row20 w.b20) _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row20 w.b20 (by decide) (by decide)),
    mnv4SkipGraphB_faithful (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row21 w.b21) _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row21 w.b21 (by decide) (by decide)),]

/-- The fused stage's graph faithfulness, restated at `mnv4FusedStack` itself.

    This corollary exists so the whole-net proof never has to unfold `mnv4FusedStack`. It looks
    redundant and is not: at MNv4's literal resolutions, letting anything unfold far enough for
    `den` to start recursing turns the kernel's check into an evaluation of the whole graph, which
    is the failure the group split above already had to work around once. Seven rewrites all of
    the shape `den <subgraph> = <subLayer>.fwd (den ·)` keep every stage opaque. -/
theorem mnv4FusedStack_graph_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (32 * 112 * 112))) :
    den (mnv4FusedGraphB epsStr N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
          w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt e)
      = (mnv4FusedStack N w).fwd (den e) :=
  mnv4FusedGraphB_faithful epsStr N 56 56 w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt
    w.f0pW w.f0pb w.f0pE w.hf0pE w.f0pg w.f0pbt e

/-- The head's graph faithfulness, restated at `mnv4HeadStack` itself. Same reason. -/
theorem mnv4HeadStack_graph_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (256 * 7 * 7))) :
    den (mnv4HeadGraphB epsStr N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt
          w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd e)
      = (mnv4HeadStack N w).fwd (den e) :=
  mnv4HeadGraphB_faithful epsStr N 7 7 w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt
    w.hW w.hb w.hE w.hhE w.hg w.hbt w.Wd w.bd e

/-- The stem's, likewise, at the net's own widths. -/
theorem mnv4StemB_graph_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (3 * 224 * 224))) :
    den (mnv4StemGraphB epsStr N 112 112 w.sW w.sb w.sE w.sg w.sbt e)
      = mnv4StemB N 112 112 w.sW w.sb w.sE w.sg w.sbt (den e) :=
  mnv4StemGraphB_faithful epsStr N 112 112 w.sW w.sb w.sE w.sg w.sbt e

/-- **The full batch-BN MobileNetV4-Conv-M forward graph**, at `mnv4FwdChainB`'s own tokens
    and its own SSA names, so the typed graph diffs against `mnv4_fwd.mlir` and `mnv4in_fwd.mlir`
    name for name (the artifacts it covers are listed in the module's conventions table). Checked
    against the committed bytes: all 247 names this writes appear in `mnv4_fwd.mlir`, and between
    them they cover all 233 of its declared parameters.

    The eighteen skip rows go through `mnv4SkipGraphB`, which is what keeps this term LINEAR in
    the depth — see that combinator's docstring for the failure mode it exists to prevent. -/
def mnv4FwdGraphBFull (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (3 * 224 * 224))) : SHlo (N * nCls) :=
  mnv4HeadGraphB epsStr N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt
    w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd
    (mnv4Res7bGraphB N epsStr w
      (mnv4Res7aGraphB N epsStr w
        (mnv4Res14bGraphB N epsStr w
          (mnv4Res14aGraphB N epsStr w
            (mnv4Res28GraphB N epsStr w
              (mnv4FusedGraphB epsStr N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt
                (mnv4StemGraphB epsStr N 112 112 w.sW w.sb w.sE w.sg w.sbt e)))))))

/-- **The MobileNetV4-Conv-M forward graph at batch BatchNorm denotes the whole-net forward.**
    Seven rewrites — the stem, the fused stage, the five resolution groups and the head — each of
    which was itself proved one block at a time.

    Each group's proof discharges its blocks' dispatch hypotheses by `decide` at the concrete
    row, so what selects ExtraDW / ConvNeXt / FFN is the TABLE, not this file. A row wired to the
    wrong builder fails to elaborate rather than proving something about a different net. -/
theorem mnv4FwdGraphBFull_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (3 * 224 * 224))) :
    den (mnv4FwdGraphBFull N epsStr w e) = mobilenetv4ForwardBFull N w (den e) := by
  -- ⚠⚠ `rw`, NOT `simp only`, and the difference is not cosmetic: the `simp only` spelling of
  -- this same chain elaborates for ~9 minutes and then dies in the KERNEL with a deterministic
  -- timeout. `simp only` traverses and rebuilds the whole term at each step, and at MNv4's literal
  -- resolutions that is enough for `den` to start unfolding into the graph itself. Outside-in
  -- `rw` never forms those terms.
  unfold mnv4FwdGraphBFull mobilenetv4ForwardBFull
    mnv4Pre6 mnv4Pre5 mnv4Pre4 mnv4Pre3 mnv4Pre2 mnv4Pre1 mnv4Pre0
  rw [mnv4HeadStack_graph_faithful, mnv4Res7bGraphB_faithful, mnv4Res7aGraphB_faithful,
      mnv4Res14bGraphB_faithful, mnv4Res14aGraphB_faithful, mnv4Res28GraphB_faithful,
      mnv4FusedStack_graph_faithful, mnv4StemB_graph_faithful]

-- ════════════════════════════════════════════════════════════════
-- § Each resolution group, expanded into its own table rows — proved WHERE THE TERMS ARE
--   VARIABLES (`CertLayer.comp_fwd_apply`), which is what makes the whole-net shape check in
--   `MobileNetV4WholeBackCertifiedTieB.lean` a 2-second `rw` chain rather than a kernel timeout.
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28** — rows 1–2, at 56 → 28 — as its own blocks, each at its table row (`mnv4Row1`, `mnv4Row2`). -/
theorem mnv4Res28Layer_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (48 * 56 * 56))) :
    (mnv4Res28Layer N w).fwd v
      = (CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2)).fwd
          ((mnv4StridedBodyOfRow N mnv4Row1 w.b1).fwd
          (v)) := by
  simp only [mnv4Res28Layer, CertLayer.comp_fwd_apply]

/-- Trunk group **Res14a** — rows 3–6, at 28 → 14 — as its own blocks, each at its table row (`mnv4Row3`, `mnv4Row4`, `mnv4Row5`, `mnv4Row6`). -/
theorem mnv4Res14aLayer_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (80 * 28 * 28))) :
    (mnv4Res14aLayer N w).fwd v
      = (CertLayer.residual (mnv4BodyOfRow N mnv4Row6 w.b6)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row5 w.b5)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row4 w.b4)).fwd
          ((mnv4StridedBodyOfRow N mnv4Row3 w.b3).fwd
          (v)))) := by
  simp only [mnv4Res14aLayer, CertLayer.comp_fwd_apply]

/-- Trunk group **Res14b** — rows 7–10, at 14×14 — as its own blocks, each at its table row (`mnv4Row7`, `mnv4Row8`, `mnv4Row9`, `mnv4Row10`). -/
theorem mnv4Res14bLayer_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (160 * 14 * 14))) :
    (mnv4Res14bLayer N w).fwd v
      = (CertLayer.residual (mnv4BodyOfRow N mnv4Row10 w.b10)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row9 w.b9)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row8 w.b8)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row7 w.b7)).fwd
          (v)))) := by
  simp only [mnv4Res14bLayer, CertLayer.comp_fwd_apply]

/-- Trunk group **Res7a** — rows 11–15, at 14 → 7 — as its own blocks, each at its table row (`mnv4Row11`, `mnv4Row12`, `mnv4Row13`, `mnv4Row14`, `mnv4Row15`). -/
theorem mnv4Res7aLayer_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (160 * 14 * 14))) :
    (mnv4Res7aLayer N w).fwd v
      = (CertLayer.residual (mnv4BodyOfRow N mnv4Row15 w.b15)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row14 w.b14)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row13 w.b13)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row12 w.b12)).fwd
          ((mnv4StridedBodyOfRow N mnv4Row11 w.b11).fwd
          (v))))) := by
  simp only [mnv4Res7aLayer, CertLayer.comp_fwd_apply]

/-- Trunk group **Res7b** — rows 16–21, at 7×7 — as its own blocks, each at its table row (`mnv4Row16`, `mnv4Row17`, `mnv4Row18`, `mnv4Row19`, `mnv4Row20`, `mnv4Row21`). -/
theorem mnv4Res7bLayer_fwd_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (256 * 7 * 7))) :
    (mnv4Res7bLayer N w).fwd v
      = (CertLayer.residual (mnv4BodyOfRow N mnv4Row21 w.b21)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row20 w.b20)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row19 w.b19)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row18 w.b18)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row17 w.b17)).fwd
          ((CertLayer.residual (mnv4BodyOfRow N mnv4Row16 w.b16)).fwd
          (v)))))) := by
  simp only [mnv4Res7bLayer, CertLayer.comp_fwd_apply]

end StableHLO

end Proofs
