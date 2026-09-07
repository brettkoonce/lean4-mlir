import LeanMlir.Proofs.Foundation.MobileNetV4BackB0

/-! # MobileNetV4-Conv-M at TRUE BATCH-NORM — the whole net's forward and graph (T1-forward, T2)

MobileNetV4 was the last net in `planning/proofs_tier_to_paper_nets.md` §2's table with nothing at
the net level. `MobileNetV4BackB0.lean` is complete at the BLOCK and STAGE level — every UIB
family, both stride-2 forms, the fused stage, the head, the table-driven `k = 0` dispatch and the
row-typed `UibParams` — and this file is the tier above: a net-level ℝ forward at the 21-row
Conv-M table, and the typed StableHLO graph over it at `mnv4FwdChainB`'s own tokens.
`planning/mnv4_proofs_tier.md` is the plan; ResNet-50 closed the same two tiers on 2026-09-06 and
`ResNet50FullB.lean` is the file this one mirrors.

⚠⚠ **NO ACCURACY IS QUOTED FOR THIS NET.** Conv-M has no Imagenette run and no verified ImageNet
run; `RESULTS.md`'s 84.58% belongs to the SUPERSEDED Conv-S table. What the artifacts under this
tier *are* pinned to is the reference's function: the forward tie measures `max |Δ| = 3.770e-06`
against `jax/.lake/build/generated_mobilenet_v4.py` on shared weights and the gradient tie puts 0
of 232 live parameters outside the reference's own fp32 noise floor (both re-run at the Conv-M
table on 2026-09-07, `planning/mnv4_convm_ties_todo.md`). That is what makes the tiers below
statements about MobileNetV4 rather than about a net.

## ⭐⭐ The trunk is FIVE `CertLayer` groups — and the reason it is not ONE is the finding here

ResNet-50's T1 needed sixteen `r50Pre_k` prefix definitions and a hand-written bottom-up `have`
chain for its apex, because its `CertLayer` trunk predated the tie files' needs. MNv4 has no such
legacy, so each resolution group — the fused stage, rows 1–2, 3–6, 7–10, 11–15, 16–21, the head —
is assembled with `CertLayer.comp` and `CertLayer.residual` directly, and inside a group:

* `.fwd` **is** that group's forward — no second definition to keep in step;
* `.ok` **is** its smoothness hypothesis, conjoined at exactly the right activations by `comp`
  rather than written out (~60 relu clauses across the net, none of them written here);
* `.vjp` **is** its `HasVJPAt` (`MobileNetV4FullBVJP.lean` chains seven of them); and
* `.faithful` **is** its BACKWARD-graph faithfulness, for free.

⛔⛔ **But composing the groups into ONE `CertLayer` does not work, and this cost a day to
establish, so it is recorded rather than re-discovered.** `fused.comp (res28.comp (… .comp head))`
elaborates fine and reads beautifully. Every later statement then has to peel `CertLayer.comp` to
reach `.fwd`, and at MNv4's LITERAL resolutions that peel is fatal: `(L₁.comp L₂).fwd =
L₂.fwd ∘ L₁.fwd` is `rfl`, and discharging it at these instances — by `rfl`, by
`simp only [CertLayer.comp_fwd]`, inside the T2 capstone or in a standalone lemma — costs ten
minutes of elaboration and then a `(kernel) deterministic timeout`. Every one of those four
spellings was measured. ⚠ The groups' own five-stage `comp` chains are completely fine; it is
composing the compositions, under something that can start unfolding `den`, that is not.

⭐ So the top level is seven named prefixes (`mnv4Pre0` … `mnv4Pre6`) and the forward is their
nest. What it costs is the hypothesis bundle: `Mnv4SmoothAt` binds one `.ok` per group, eight
fields rather than two. What it keeps is everything that mattered — R50's apex binds 33, and MNv4
binds no `0 < ε` hypothesis at all, because those live inside the weight records.

▶ **The general lesson, and it is not MNv4-specific:** a net whose resolutions are LITERALS cannot
afford the proof idioms a net with a resolution BINDER can. ResNet-50's `q` keeps `den` stuck;
MNv4's 224/112/56/28/14/7 let it run. Three separate blow-ups in this file trace to exactly that —
this one, the graph builders that had to be made generic in their widths, and the whole-net
capstone that had to become `rw` instead of `simp only`.

⚠⚠ **The stem sits OUTSIDE the chain, and this is EfficientNet-B0's situation exactly.**
`CertLayer` demands a backward graph, and **no render emits a gradient into `%x`** — there is no
`convStridedXlaBackBatched` token, because the artifact's backward ends at the stem conv's WEIGHT
gradient. B0's `enetTrunk` takes its stem as a parameter for the same reason. So `mnv4StemB` is a
plain function here, its VJP is `bnReluStage_has_vjp_at` at `flatConvStride2Xla`, and the net-level
VJP composes the two with `vjp_comp_at`.

## Conventions this net runs at

| | |
|---|---|
| depth | 21 UIB blocks + the fused stage; 13 ExtraDW / 4 ConvNeXt-like / 4 FFN, and **no IB** |
| ladder | 224 →(stem s2) 112 →(fused s2) 56 →(blk1) 28 →(blk3) 14 →(blk11) 7 → GAP |
| channels | 32 → 48 → 80 → 160 → 256, head 256 → 960 → 1280 |
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N·h·w`) at all 77 sites |
| activation | **relu**, not relu6 (MobileNetV2 sits one file over and uses relu6); the fused stage alone is **swish** |
| padding | ⚠ TWO phases in one net: the stem is XLA-`SAME` (`flatConvStride2Xla`), the fused stage and all three strided depthwises are SYMMETRIC. Both correct; `scripts/convention_audit.py` reads them. Do not tidy one to match the other. |
| stride | all three stride-2 UIB rows (1, 3, 11) are PRE-strided; Conv-M has no post-strided row at all |
| census | **233** parameter slots at `nCls = 10` (8,447,322 scalars; 9,715,512 at 1000), bias-free by construction |
| artifacts | `mnv4_fwd`, `mnv4_fwd_eval`, `mnv4_adam_train_step`, and the five `mnv4in*` ImageNet twins |

⚠ `N` stays a binder throughout, as at r34/R50: this tier carries no batch numeral, and the
artifacts' `N` is the PER-REPLICA batch (`DataParallel.lean`, §4d). Unlike R50 there is no `q`
binder — MNv4 ships one resolution.

⚠ **Rows 4/5/10, 12/18 and 15/19/20 are shape-identical**, so their `UibParams` records have the
same TYPE and swapping their weights typechecks. Typing pins shape, not identity; what pins
identity is the SSA NAMES the T2 graph writes (`%u4qW` vs `%u10qW`), which is why the graph reads
its names from `s.p` off the table rather than taking them as arguments.

✅ Checked against the committed bytes: `verified_mlir/mnv4_fwd.mlir`'s signature is **234
arguments = `%x` + 233 parameters**, and every name this file writes appears there.
-/

namespace Proofs

open scoped BigOperators

-- ⚠ Both R50 tier files raise this. Here it is the 24-stage chain: the whole-net faithfulness
-- proof rewrites through every stage, and the kernel's check of the resulting term does not fit
-- the 200000 default. A `(kernel) deterministic timeout` on a `simp only` is what that looks like.
set_option maxHeartbeats 2000000

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § `CertLayer`'s forward projections, as `simp` lemmas
-- ════════════════════════════════════════════════════════════════

/-! ⚠⚠ **Without these this file takes THREE MINUTES to elaborate; with them, seconds.** The
reason is worth recording. `simp only [CertLayer.comp]` rewrites `L₁.comp L₂` to the full
structure literal — `fwd`, `ok`, `diff`, `vjp`, `graph` AND `faithful` — and only then projects
`.fwd` out of it. Over a 24-stage chain that builds an enormous intermediate term whose bulk is
PROOFS the goal never mentions. Projecting `.fwd` in one step never forms it.

⛔ These are `CertLayer` API lemmas and belong in `Foundation/CertifiedChain.lean`, not here. They
are parked in this leaf deliberately: `CertifiedChain.lean` is imported by `BackNetFolds.lean`,
which most of the corpus sits downstream of, and the root-file-lemma rule says to park in the leaf
with a note and move such lemmas as a BATCH. ▶ Move them when `CertifiedChain.lean` is next
touched for another reason. -/

@[simp] theorem CertLayer.comp_fwd {m n p : Nat} (L₁ : CertLayer m n) (L₂ : CertLayer n p) :
    (L₁.comp L₂).fwd = L₂.fwd ∘ L₁.fwd := rfl

@[simp] theorem CertLayer.residual_fwd {n : Nat} (L : CertLayer n n) :
    (CertLayer.residual L).fwd = Proofs.residual L.fwd := rfl

@[simp] theorem CertLayer.id'_fwd (n : Nat) : (CertLayer.id' n).fwd = fun y => y := rfl

-- ════════════════════════════════════════════════════════════════
-- § The block table, one row per constant
-- ════════════════════════════════════════════════════════════════

/-! ⚠ These are `abbrev`s, and the rows are NAMED rather than indexed. `UibParams (mnv4Blocks[3]!)`
in a type would force `whnf` through `List.get!` at every use; a named reducible constant reduces
to its projections directly, which is what lets `CertLayer.comp` line up `2 * 28` with `56` across
a stride join without a single transport.

⭐ The `#guard` below is the whole safety of that move: these 21 constants are pinned to
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
-- ⚠ All three stride-2 rows are PRE-strided, so `mnv4UibPostStridedBody` has no consumer in this
-- file. That arm stays certified and unexercised; a green build is not coverage of it.
#guard (mnv4Blocks.filter (fun s => s.stride2)).map (·.p) = ["1", "3", "11"]
#guard (mnv4Blocks.filter (fun s => s.stride2)).all (fun s => s.preDWk != 0)

-- ════════════════════════════════════════════════════════════════
-- § The weights, typed by their table rows
-- ════════════════════════════════════════════════════════════════

/-- **Every MobileNetV4-Conv-M parameter**, generic in the class count so one statement covers the
    10-class Imagenette artifacts and the 1000-class `mnv4in` ones.

    ⭐ The 21 block fields are `UibParams mnv4Row{k}` — a record whose every width is a *projection
    of its row*, so a record that disagrees with its row **cannot be constructed** and the forward
    below needs no side conditions on widths. That is strictly stronger than ResNet-50's
    `R50IdW`/`R50ProjW`, which are typed by loose `{mid oc}` binders. ⚠ It still does not pin
    IDENTITY between shape-identical rows (4/5/10, 12/18, 15/19/20) — see the header.

    ⭐ The `0 < ε` obligations live INSIDE the records (`UibParams`'s `hq he hd hz`), so the stem,
    the fused stage and the head carry theirs as fields too. R50 keeps a separate `R50IdPos`
    bundle; matching `UibParams` here means the whole-net VJP binds no epsilon hypotheses at all.

    ⚠ Every conv is bias-free — both renders bake `convBias := false` and bind each bias to the
    `%zb{c}` zero the prelude declares — but the records still carry a `b` slot because the stage
    vocabulary takes one. Those fields are `∀`-quantified over; `bias = 0` is one instance. Field
    names are the render's own SSA prefixes, so a reader can match a parameter to its emitted name
    without a table. -/
structure Mnv4BWeights (nCls : Nat) where
  /-- stem `%sW`/`%sg`/`%sbt`: 3×3/s2 at the **XLA-`SAME`** phase, 3 → 32, 224 → 112. -/
  sW : Kernel4 32 3 3 3
  sb : Vec 32
  sE : ℝ
  hsE : 0 < sE
  sg : Vec 32
  sbt : Vec 32
  /-- fused stage `%f0cW`: 3×3/s2 **symmetric**, 32 → 128, 112 → 56, then swish. -/
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
  /-- head conv 2 `%hW`: 1×1, 960 → 1280. ⚠ Conv-M's head has TWO convs; `mnv4Head` models one. -/
  hW : Kernel4 1280 960 1 1
  hb : Vec 1280
  hE : ℝ
  hhE : 0 < hE
  hg : Vec 1280
  hbt : Vec 1280
  /-- classifier `%Wd`/`%bd`, after GAP(7×7). -/
  Wd : Mat 1280 nCls
  bd : Vec nCls

-- ════════════════════════════════════════════════════════════════
-- § The stem — outside the chain, and it cannot be otherwise
-- ════════════════════════════════════════════════════════════════

/-- MNv4's stem forward: 3×3/s2 conv at the **XLA-`SAME`** phase → batch BN → **relu**.

    ⚠⚠ This is the ONE XLA-padded site in the net, and the reason `.convStridedXla` exists at all:
    XLA `'SAME'` on a 3×3/s2 at 224 pads **(0,1)**, not (1,1). Both give 112×112, so no shape
    check, `#guard`, op count or arity audit can see the difference — the forward tie is the only
    thing that can, and it measured 6.16e-2 with the symmetric token against 1.79e-6 with the
    reference patched to match (`planning/mnv4_verified.md` §3b). Every OTHER stride-2 site in this
    net is genuinely symmetric.

    ⚠ Plain relu, and it is `relu6` one file over in `MobileNetV2FullB.lean` at the same XLA
    padding — the two stems differ in exactly one token. -/
@[reducible] noncomputable def mnv4StemB (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w εs γs βs ∘
    batchMap N (flatConvStride2Xla Ws bs)

-- ════════════════════════════════════════════════════════════════
-- § The trunk, as five certified resolution groups
-- ════════════════════════════════════════════════════════════════

/-- **The fused stage (stage 0)**: 3×3/s2 SYMMETRIC conv-bn-**swish** 32 → 128 at 112 → 56, then
    the 1×1 project 128 → 48. ⭐ The only globally-certified stage in the net — swish has no kink,
    so `ok = True` and this stage discharges nothing. -/
noncomputable def mnv4FusedStack (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (32 * 112 * 112)) (N * (48 * 56 * 56)) :=
  mnv4FusedStage N
    (mnv4FusedConvLayer (h := 56) (w := 56) N w.f0cW w.f0cb w.f0cE w.hf0cE w.f0cg w.f0cbt)
    (mnv4ProjectLayer (h := 56) (w := 56) N w.f0pW w.f0pb w.f0pE w.hf0pE w.f0pg w.f0pbt)

/-- **The head**: 1×1 256 → 960 conv-bn-relu, 1×1 960 → 1280 conv-bn-relu, GAP(7×7), classifier.

    ⚠ `mnv4Head` models ONE conv stage and Conv-M's render emits **two** (`%h1W` then `%hW`), so
    the first is composed on the outside as a second `mnv4ExpandLayer` — conv-bn-relu is
    conv-bn-relu and the kernel extent is a binder, so 1×1 is an argument. ⭐ GAP and dense are
    both globally certified and both tie by `rfl`; only the two relus carry a condition. -/
noncomputable def mnv4HeadStack (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (256 * 7 * 7)) (N * nCls) :=
  (mnv4ExpandLayer (h := 7) (w := 7) N w.h1W w.h1b w.h1E w.hh1E w.h1g w.h1bt).comp
    (mnv4Head N (mnv4ExpandLayer (h := 7) (w := 7) N w.hW w.hb w.hE w.hhE w.hg w.hbt)
      (mnv4GapLayer N (c := 1280) (h := 7) (w := 7)) (mnv4DenseLayer N w.Wd w.bd))

/-! ⚠⚠ **The trunk is built in GROUPS, and that is a proof-engineering requirement.** One 24-stage
`CertLayer` elaborates fine — it is the T2 faithfulness proof over it that does not: the whole-net
rewrite chain produces a term whose KERNEL check exceeds any reasonable budget (measured: the
elaboration succeeds after ~9 minutes and the kernel then reports a deterministic timeout). Split
at the net's own resolution boundaries, each group's proof is small, and the whole-net theorem is
six rewrites over them. ⭐ The grouping is the ladder a reader already knows — 56, 28, 14, 7 — so
it costs nothing in readability and buys a bounded proof. -/

/-- Trunk group **Res28** — rows 1–2: the 56→28 reduction and the block that follows it. -/
noncomputable def mnv4Res28Layer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (48 * 56 * 56)) (N * (80 * 28 * 28)) :=
  (mnv4PreStridedBodyOfRow N mnv4Row1 w.b1).comp
      (CertLayer.residual (mnv4BodyOfRow N mnv4Row2 w.b2))

/-- Trunk group **Res14a** — rows 3–6: the 28→14 reduction, then three ExtraDW blocks. -/
noncomputable def mnv4Res14aLayer (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) :
    CertLayer (N * (80 * 28 * 28)) (N * (160 * 14 * 14)) :=
  (mnv4PreStridedBodyOfRow N mnv4Row3 w.b3).comp
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
  (mnv4PreStridedBodyOfRow N mnv4Row11 w.b11).comp
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

/-! ⛔⛔ **The seven groups are composed by a PREFIX CHAIN, not by one more `CertLayer.comp`, and
this is the single hardest thing this file learned.**

A `mnv4NetLayer := fused.comp (res28.comp (… .comp head))` elaborates fine and reads beautifully.
But every downstream statement then has to peel `CertLayer.comp` to get at `.fwd`, and at MNv4's
LITERAL resolutions that peel is fatal: `(L₁.comp L₂).fwd = L₂.fwd ∘ L₁.fwd` is `rfl`, yet
discharging it at these instances — by `rfl`, by `simp only [CertLayer.comp_fwd]`, inside the T2
capstone or in a standalone lemma — costs ten minutes of elaboration and then a `(kernel)
deterministic timeout`. ⚠ The groups' OWN five-stage `comp` chains are fine; it is composing the
compositions, under something that can start unfolding, that is not.

⭐ So the top level is seven named prefixes and the forward is their nest — ResNet-50's shape at
seven stages instead of eighteen. What that costs is the hypothesis bundle: `Mnv4SmoothAt` binds
one `.ok` per group (seven) rather than one for the whole trunk. What it keeps is everything that
mattered — each group's `.ok` is still the conjunction `CertLayer.comp` assembled from its blocks'
conditions at their own activations, so ~60 relu clauses are still never written down, and no
`0 < ε` hypothesis appears at all. R50's apex binds 33. -/

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

/-- **T1's forward half: the full batch-BN MobileNetV4-Conv-M**, `N*(3*224*224) → N*nCls`.

    The stem, the fused stage, the five resolution groups, the head. Every block inside those
    groups is `mnv4BodyOfRow` at its own row, so the `k = 0` dispatch is READ from `mnv4Blocks`
    rather than chosen here — the property `MobileNetV4BackB0.lean`'s dispatch section exists to
    establish, now carried to the net. The eighteen skips are `CertLayer.residual`, which
    typechecks with no transport because `s.oc` and `s.ic` reduce to the same literal at every
    stride-1 row (guarded there). -/
noncomputable def mobilenetv4ForwardB_full (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Vec (N * nCls) :=
  (mnv4HeadStack N w).fwd (mnv4Pre6 N w x)

-- ⭐ The stem's own arithmetic, checked rather than asserted: its input is the 224-px image and
-- its output is what the fused stage reads. A wrong nest depth is well-typed at a variable.
#guard 2 * 112 == 224
#guard 2 * 56 == 112
#guard 2 * 28 == 56
#guard 2 * 14 == 28
#guard 2 * 7 == 14

-- ⭐ `mobilenetv4ForwardB_full` really does bind at the literal 224-px image type.
example (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardB_full N w x = mobilenetv4ForwardB_full N w x := rfl

-- ════════════════════════════════════════════════════════════════
-- § T2 — the typed forward graph, at `mnv4FwdChainB`'s own tokens
--   `.batchOp` of `.convStridedXla` (stem) / `.convStrided` (fused) / `.conv` / `.depthwise` /
--   `.depthwiseStrided` / `.relu` / `.swish` / `.gap` / `.dense`, `.bnBatchF` for the
--   batch-coupled norm, and `.addVB` for the residual add.
--
--   ⚠ Every SSA name is read off the ROW (`s.p`), never taken as an argument. That is what pins
--   identity between shape-identical rows: rows 4, 5 and 10 have the same `UibParams` type, so
--   only `%u4qW` vs `%u5qW` vs `%u10qW` tells them apart, and here those come from the table.
--
--   ⚠ Bias operands are `%zb{c}`, the shared zero constant every bias folds into its BatchNorm
--   and binds to. `MobileNetV4RenderB` has no `convBias` flag at all, so this is the only name
--   this net emits.
-- ════════════════════════════════════════════════════════════════

/-- Stem graph: 3×3/s2 XLA-`SAME` conv → batch BN → relu.

    ⚠⚠ **GENERIC in the widths, and that is a correctness-of-elaboration requirement, not style.**
    Pinning `ic := 3, oc := 32, h := 112` here makes `den_batchOp_convStridedXla`'s `rfl` a claim
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
      (.batchOp (N := N) (.convStridedXla (h := h) (w := w) "%sW" s!"%zb{oc}" Ws bs) e))

theorem mnv4StemGraphB_faithful (epsStr : String) (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mnv4StemGraphB epsStr N h w Ws bs εs γs βs e)
      = mnv4StemB N h w Ws bs εs γs βs (den e) := by
  simp only [mnv4StemGraphB, mnv4StemB, den_batchOp_relu_eq_reluF, reluF_faithful,
    den_batchOp_convStridedXla, den_bnBatchF, Function.comp_apply]

/-- Fused stage graph: 3×3/s2 SYMMETRIC conv → BN → **swish** → 1×1 project → BN. No skip.
    ⚠ Generic in the widths, for the reason `mnv4StemGraphB` records. -/
def mnv4FusedGraphB (epsStr : String) (N h w : Nat) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) : SHlo (N * (oc * h * w)) :=
  .bnBatchF "%f0pg" "%f0pbt" epsStr εp γp βp
    (.batchOp (N := N) (.conv (h := h) (w := w) "%f0pW" s!"%zb{oc}" Wp bp)
      (.batchOp (N := N) (.swish (n := mid * h * w))
        (.bnBatchF "%f0cg" "%f0cbt" epsStr εc γc βc
          (.batchOp (N := N) (.convStrided (h := h) (w := w) "%f0cW" s!"%zb{mid}" Wc bc) e))))

theorem mnv4FusedGraphB_faithful (epsStr : String) (N h w : Nat) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (hεc : 0 < εc) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (e : SHlo (N * (ic * (2 * h) * (2 * w)))) :
    den (mnv4FusedGraphB epsStr N h w Wc bc εc γc βc Wp bp εp γp βp e)
      = (mnv4FusedStage N (mnv4FusedConvLayer (h := h) (w := w) N Wc bc εc hεc γc βc)
          (mnv4ProjectLayer (h := h) (w := w) N Wp bp εp hεp γp βp)).fwd (den e) := by
  simp only [mnv4FusedGraphB, mnv4FusedStage, mnv4FusedConvLayer,
    mnv4ProjectLayer, CertLayer.comp_fwd, fusedConvB, projB,
    den_batchOp_swish_eq_swishF, swishF_faithful, den_batchOp_conv, den_batchOp_convStrided,
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
                  (.batchOp (N := N) (.relu (n := s.ic * s.h * s.h))
                    (.bnBatchF s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr p.eq_ p.gq p.bq2
                      (.batchOp (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
                          s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq)
                        e))))))))))

/-- ⭐ The ExtraDW body graph denotes the row-typed body's forward — **generic in the row**, so one
    theorem serves all thirteen. The two hypotheses are exactly the dispatch conditions
    `mnv4PreDWSlot`/`mnv4PostDWSlot` branch on, discharged by `decide` at each concrete row. -/
theorem mnv4ExtraDWBodyGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hq : s.preDWk ≠ 0) (hd : s.postDWk ≠ 0)
    (e : SHlo (N * (s.ic * s.h * s.h))) :
    den (mnv4ExtraDWBodyGraphB epsStr N s p e) = (mnv4BodyOfRow N s p).fwd (den e) := by
  simp only [mnv4ExtraDWBodyGraphB, mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot,
    if_neg hq, if_neg hd, mnv4DWReluLayer, mnv4ExpandLayer, mnv4ProjectLayer, CertLayer.comp_fwd,
    projB, cbReluB, dwbReluB, den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv,
    den_batchOp_depthwise, den_bnBatchF, Function.comp_apply]

/-- **ConvNeXt-like body graph** — pre-DW only, `postDWk = 0`, four of Conv-M's rows (8, 10, 16,
    21). ⛔ The absent depthwise emits NO tokens, exactly as `mnv4PostDWSlot` inserts `id'`: the
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
            (.batchOp (N := N) (.relu (n := s.ic * s.h * s.h))
              (.bnBatchF s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr p.eq_ p.gq p.bq2
                (.batchOp (N := N) (.depthwise (c := s.ic) (h := s.h) (w := s.h)
                    s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq)
                  e)))))))

theorem mnv4ConvNeXtBodyGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hq : s.preDWk ≠ 0) (hd : s.postDWk = 0)
    (e : SHlo (N * (s.ic * s.h * s.h))) :
    den (mnv4ConvNeXtBodyGraphB epsStr N s p e) = (mnv4BodyOfRow N s p).fwd (den e) := by
  simp only [mnv4ConvNeXtBodyGraphB, mnv4BodyOfRow, mnv4UibBody, mnv4PreDWSlot, mnv4PostDWSlot,
    if_neg hq, if_pos hd, mnv4DWReluLayer, mnv4ExpandLayer, mnv4ProjectLayer, CertLayer.id'_fwd,
    CertLayer.comp_fwd, projB, cbReluB, dwbReluB, den_batchOp_relu_eq_reluF, reluF_faithful,
    den_batchOp_conv, den_batchOp_depthwise, den_bnBatchF, Function.comp_apply]

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
    if_pos hq, if_pos hd, mnv4ExpandLayer, mnv4ProjectLayer, CertLayer.id'_fwd, CertLayer.comp_fwd,
    projB, cbReluB, den_batchOp_relu_eq_reluF, reluF_faithful, den_batchOp_conv,
    den_bnBatchF, Function.comp_apply]

/-- **Pre-strided block graph** — rows 1, 3 and 11, the only stride-2 rows Conv-M has, and all
    three PRE-strided. The leading depthwise carries the stride (`.depthwiseStrided`, SYMMETRIC
    padding), so everything after it runs at the reduced `h`. ⚠ No skip: `ic ≠ oc` at all three,
    so the block IS the body and there is no `.addVB`. -/
def mnv4PreStridedGraphB (epsStr : String) (N : Nat) (s : UibSpec) (p : UibParams s)
    (e : SHlo (N * (s.ic * (2 * s.h) * (2 * s.h)))) : SHlo (N * (s.oc * s.h * s.h)) :=
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
                  (.batchOp (N := N) (.relu (n := s.ic * s.h * s.h))
                    (.bnBatchF s!"%u{s.p}qg" s!"%u{s.p}qbt" epsStr p.eq_ p.gq p.bq2
                      (.batchOp (N := N) (.depthwiseStrided (c := s.ic) (h := s.h) (w := s.h)
                          s!"%u{s.p}qW" s!"%zb{s.ic}" p.Wq p.bq)
                        e))))))))))

theorem mnv4PreStridedGraphB_faithful (epsStr : String) (N : Nat) (s : UibSpec)
    (p : UibParams s) (hd : s.postDWk ≠ 0)
    (e : SHlo (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    den (mnv4PreStridedGraphB epsStr N s p e) = (mnv4PreStridedBodyOfRow N s p).fwd (den e) := by
  simp only [mnv4PreStridedGraphB, mnv4PreStridedBodyOfRow, mnv4UibPreStridedBody, mnv4PostDWSlot,
    if_neg hd, mnv4DWReluLayer, mnv4DWReluStridedLayer, mnv4ExpandLayer, mnv4ProjectLayer,
    CertLayer.comp_fwd, projB, cbReluB, dwbReluB, dwbReluBstrided, den_batchOp_relu_eq_reluF,
    reluF_faithful, den_batchOp_conv, den_batchOp_depthwise, den_batchOp_depthwiseStrided,
    den_bnBatchF, Function.comp_apply]

/-- ⭐⭐ **One skip row's graph: its body's, plus the identity skip.** Trivial as a definition and
    load-bearing as a barrier.

    ⚠⚠ **This is why it is a named combinator and not an inline `.addVB`.** The residual add needs
    the block's input subtree TWICE, and MNv4 has eighteen of them. Written inline — or hidden
    behind a `let` in the whole-net graph, which is what this file did first — the term doubles at
    every skip the moment anything unfolds it, and `simp only [mnv4FwdGraphB_full]` ZETA-EXPANDS
    lets, so the `let` form bought nothing at all: the whole-net faithfulness proof elaborated and
    then died in the KERNEL with a deterministic timeout.

    ⭐ Kept folded, with `mnv4SkipGraphB_faithful` rewriting `den (mnv4SkipGraphB body e)` in ONE
    step, `den e` occurs once and the whole-net term stays linear in the depth. R50 never met this:
    its `r50IdGraphB` takes `e` as a binder and duplicates it inside the builder, which has the
    same effect for the same reason. -/
def mnv4SkipGraphB {N n : Nat} (body : SHlo (N * n) → SHlo (N * n)) (e : SHlo (N * n)) :
    SHlo (N * n) :=
  .addVB (body e) e

/-- ⭐ A skip row denotes `residual` of whatever its body denotes — generic in both, so one
    theorem covers all eighteen and the body's own faithfulness lemma is the only input. -/
theorem mnv4SkipGraphB_faithful {N n : Nat} (body : SHlo (N * n) → SHlo (N * n))
    (f : Vec (N * n) → Vec (N * n))
    (hb : ∀ e' : SHlo (N * n), den (body e') = f (den e')) (e : SHlo (N * n)) :
    den (mnv4SkipGraphB body e) = Proofs.residual f (den e) := by
  simp only [mnv4SkipGraphB, den_addVB, hb, Proofs.residual]

/-- **Head graph**: 1×1 conv-BN-relu, a SECOND 1×1 conv-BN-relu, GAP, dense — Conv-M's head has
    two convs where `mnv4Head` models one. ⚠ Generic in the widths, for the reason
    `mnv4StemGraphB` records. -/
def mnv4HeadGraphB (epsStr : String) (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (e : SHlo (N * (c * h * w))) : SHlo (N * nCls) :=
  .batchOp (N := N) (.dense "%Wd" "%bd" Wd bd)
    (.batchOp (N := N) (.gap (c := oc) (h := h) (w := w))
      (.batchOp (N := N) (.relu (n := oc * h * w))
        (.bnBatchF "%hg" "%hbt" epsStr ε2 γ2 β2
          (.batchOp (N := N) (.conv (h := h) (w := w) "%hW" s!"%zb{oc}" W2 b2)
            (.batchOp (N := N) (.relu (n := mid * h * w))
              (.bnBatchF "%h1g" "%h1bt" epsStr ε1 γ1 β1
                (.batchOp (N := N) (.conv (h := h) (w := w) "%h1W" s!"%zb{mid}" W1 b1)
                  e)))))))

theorem mnv4HeadGraphB_faithful (epsStr : String) (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (hε1 : 0 < ε1) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (hε2 : 0 < ε2) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (e : SHlo (N * (c * h * w))) :
    den (mnv4HeadGraphB epsStr N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd e)
      = ((mnv4ExpandLayer (h := h) (w := w) N W1 b1 ε1 hε1 γ1 β1).comp
          (mnv4Head N (mnv4ExpandLayer (h := h) (w := w) N W2 b2 ε2 hε2 γ2 β2)
            (mnv4GapLayer N (c := oc) (h := h) (w := w)) (mnv4DenseLayer N Wd bd))).fwd (den e) := by
  simp only [mnv4HeadGraphB, mnv4Head, mnv4ExpandLayer, mnv4GapLayer,
    mnv4DenseLayer, CertLayer.comp_fwd, cbReluB, den_batchOp_relu_eq_reluF, reluF_faithful,
    den_batchOp_conv, den_batchOp_gap, den_batchOp_dense, den_bnBatchF, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole graph + faithfulness (T2)
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28**'s graph — rows 1–2: the 56→28 reduction and the block that follows it. -/
def mnv4Res28GraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (48 * 56 * 56))) : SHlo (N * (80 * 28 * 28)) :=
  mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row2 w.b2)
      (mnv4PreStridedGraphB epsStr N mnv4Row1 w.b1
      (e))

theorem mnv4Res28GraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (48 * 56 * 56))) :
    den (mnv4Res28GraphB N epsStr w e) = (mnv4Res28Layer N w).fwd (den e) := by
  simp only [mnv4Res28GraphB, mnv4Res28Layer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4PreStridedGraphB_faithful epsStr N mnv4Row1 w.b1 (by decide),
    mnv4SkipGraphB_faithful (mnv4ExtraDWBodyGraphB epsStr N mnv4Row2 w.b2) _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row2 w.b2 (by decide) (by decide)),]

/-- Trunk group **Res14a**'s graph — rows 3–6: the 28→14 reduction, then three ExtraDW blocks. -/
def mnv4Res14aGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (e : SHlo (N * (80 * 28 * 28))) : SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row6 w.b6)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row5 w.b5)
      (mnv4SkipGraphB (mnv4ExtraDWBodyGraphB epsStr N mnv4Row4 w.b4)
      (mnv4PreStridedGraphB epsStr N mnv4Row3 w.b3
      (e))))

theorem mnv4Res14aGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (80 * 28 * 28))) :
    den (mnv4Res14aGraphB N epsStr w e) = (mnv4Res14aLayer N w).fwd (den e) := by
  simp only [mnv4Res14aGraphB, mnv4Res14aLayer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4PreStridedGraphB_faithful epsStr N mnv4Row3 w.b3 (by decide),
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
      (mnv4PreStridedGraphB epsStr N mnv4Row11 w.b11
      (e)))))

theorem mnv4Res7aGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (160 * 14 * 14))) :
    den (mnv4Res7aGraphB N epsStr w e) = (mnv4Res7aLayer N w).fwd (den e) := by
  simp only [mnv4Res7aGraphB, mnv4Res7aLayer, CertLayer.comp_fwd, CertLayer.residual_fwd,
    Function.comp_apply,
    mnv4PreStridedGraphB_faithful epsStr N mnv4Row11 w.b11 (by decide),
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

    ⚠ This corollary exists so the whole-net proof never has to UNFOLD `mnv4FusedStack`. It looks
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

/-- ⭐⭐ **The full batch-BN MobileNetV4-Conv-M forward graph**, at `mnv4FwdChainB`'s own tokens
    and its own SSA names, so the typed graph diffs against `mnv4_fwd.mlir` and its five ImageNet
    twins name for name. ✅ Checked against the committed bytes: all 247 names this writes appear
    in that file, and between them they cover all 233 of its declared parameters.

    ⚠ The eighteen skip rows go through `mnv4SkipGraphB`, which is what keeps this term LINEAR in
    the depth — see that combinator's docstring for the failure mode it exists to prevent. -/
def mnv4FwdGraphB_full (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
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

/-- ⭐⭐ **T2 for MobileNetV4-Conv-M at batch BatchNorm**: the typed graph denotes the whole-net
    forward. Seven rewrites — the stem, the fused stage, the five resolution groups and the head —
    each of which was itself proved one block at a time. The first graph-level tier this net has
    ever had.

    ⚠ Each group's proof discharges its blocks' dispatch hypotheses by `decide` at the concrete
    row, so what selects ExtraDW / ConvNeXt / FFN is the TABLE, not this file. A row wired to the
    wrong builder fails to elaborate rather than proving something about a different net. -/
theorem mnv4FwdGraphB_full_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (e : SHlo (N * (3 * 224 * 224))) :
    den (mnv4FwdGraphB_full N epsStr w e) = mobilenetv4ForwardB_full N w (den e) := by
  -- ⚠⚠ `rw`, NOT `simp only`, and the difference is not cosmetic: the `simp only` spelling of
  -- this same chain elaborates for ~9 minutes and then dies in the KERNEL with a deterministic
  -- timeout. `simp only` traverses and rebuilds the whole term at each step, and at MNv4's literal
  -- resolutions that is enough for `den` to start unfolding into the graph itself. Outside-in
  -- `rw` never forms those terms.
  unfold mnv4FwdGraphB_full mobilenetv4ForwardB_full
    mnv4Pre6 mnv4Pre5 mnv4Pre4 mnv4Pre3 mnv4Pre2 mnv4Pre1 mnv4Pre0
  rw [mnv4HeadStack_graph_faithful, mnv4Res7bGraphB_faithful, mnv4Res7aGraphB_faithful,
      mnv4Res14bGraphB_faithful, mnv4Res14aGraphB_faithful, mnv4Res28GraphB_faithful,
      mnv4FusedStack_graph_faithful, mnv4StemB_graph_faithful]

end StableHLO

end Proofs
