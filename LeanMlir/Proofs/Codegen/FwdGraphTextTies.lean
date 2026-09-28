import LeanMlir.Proofs.Codegen.ResNet50RenderB
import LeanMlir.Proofs.Nets.ResNet.ResNet50FullB
import LeanMlir.Proofs.Codegen.MobileNetV2RenderB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaperEval
import LeanMlir.Proofs.Codegen.MobileNetV4RenderB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBEval
import LeanMlir.Proofs.Codegen.EfficientNetRender.Basic
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0Drop

/-! # FwdGraphTextTies — the rendered forward blocks are `pretty` of the typed block graphs

A net's graph-faithfulness theorem (`resnet34FwdGraphBFull_faithful`, …) says a typed graph
denotes the proven forward. The artifact in `verified_mlir/` is written by the renderer's forward chain
(`r34FwdChainB`, …), a separate definition that emits one block at a time. This module ties the
two by TEXT: for every block kind, the renderer's block emitter and `pretty` of the typed block graph —
the block input an `.operand` leaf named as the emitter's input, every weight zero (`pretty` reads
names and shapes only) — print the same bytes from the same `EmitS` start state.

**Why per block.** The whole-net graph cannot be printed and compared: `pretty` shares nothing,
and a residual graph repeats its block-input subtree in both `addVB` operands, so the prefix is
re-emitted at every skip (≈2¹⁶ copies of the stem for ResNet-34). Per block the input is a leaf, so
repeating it emits nothing — exactly what the chain does when it names the input twice. What stays
outside the guards is the chain's glue: it calls these emitters in the typed graph's nesting
order, with the typed graph's prefixes and shapes, each block reading the previous block's output name.

**Scope.** f32, `convBias := false`, one replica (`sync := false`) — the configuration the typed
graphs describe. The bf16 renders swap in `…Bf16` constructors (`Bf16Fold`, `Bf16GradNodes`);
sync-BN renders swap the BN site (`SyncBnSites`, the `*SyncB` twins). Covered: ResNet-34,
ResNet-50, MobileNetV2, MobileNetV4-Conv-M and EfficientNet-B0 — every block kind, stem and head,
each checked by `#guard` at batch 2 on one concrete shape (for MobileNetV4, every row of the
21-row table), MobileNetV2's per-example inference forward (`MobileNetV2FullPaperEval`), the
MobileNetV2, MobileNetV4 and EfficientNet-B0 heads with classifier dropout (`cd := true`),
EfficientNet-B0's residual block with its stochastic-depth site (`sd := true`), and
MobileNetV4's inference forward (`.eval`, frozen-statistics BN) at both input
sizes its evals are rendered at. Not covered: ConvNeXt-T and ViT, whose typed graphs are per-example, with their
own constructors.

A `#guard` failing here means the emitted text and the proven graph drifted: the graph's operand
order, a name, a constructor or a shape differs from what the renderer writes. Fix the side that is
wrong; the artifact is the ground truth unless it is the bug. -/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO.FwdGraphTextTies

/-- The state every comparison starts from: a nonzero SSA counter and an empty shape table. -/
def s0 : EmitS := (7, [])

/-- Run an emitter from `s0` and keep the text. -/
def textOf {α : Type} (m : StateM EmitS α) (code : α → String) : String :=
  code (Id.run (m.run' s0))

/-- `pretty` of a graph from `s0`. -/
def prettyText (B : Nat) {k : Nat} (g : SHlo k) : String :=
  textOf (pretty B g) (·.1)

/-- A zero-valued operand leaf: the block input. -/
def leaf (nm : String) (n : Nat) : SHlo n := .operand nm (fun _ => 0)

-- ════════════════════════════════════════════════════════════════
-- § ResNet-34 — `r34FwdChainB` vs `resnet34FwdGraphBFull`
-- ════════════════════════════════════════════════════════════════

def r34IdW0 (c : Nat) : R34IdW c :=
  ⟨fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0⟩

def r34DownW0 (ic oc : Nat) : R34DownW ic oc :=
  ⟨fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0⟩

-- Stem: `r34StemFwdB` = `pretty (r34StemGraphB …)` on `%x`.
#guard textOf (r34StemFwdB 2 "1.0e-05" false) (·.code) ==
  prettyText 2 (r34StemGraphB "1.0e-05" 2 56 56 (ic := 3) (oc := 64)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%x" _))

-- Identity block (`s1b0`, 64 ch at 56²).
#guard textOf (idFwdB 2 64 56 "1.0e-05" "s1b0" "%in" false) (·.code) ==
  prettyText 2 (r34IdGraphB "s1b0" "1.0e-05" 2 56 56 (r34IdW0 64) (leaf "%in" _))

-- Downsample block (`d2`, 64 → 128, 56² → 28²).
#guard textOf (downFwdB 2 64 128 28 "1.0e-05" "d2" "%in" false) (·.code) ==
  prettyText 2 (r34DownGraphB "d2" "1.0e-05" 2 28 28 (r34DownW0 64 128) (leaf "%in" _))

-- Head: GAP(7²) → dense(512 → 10).
#guard textOf (r34HeadFwdB 2 10 "%in") (·.1) ==
  prettyText 2 (r34HeadGraphB 2 7 7 (c := 512) (nCls := 10) (fun _ _ => 0) (fun _ => 0)
    (leaf "%in" _))

-- ════════════════════════════════════════════════════════════════
-- § ResNet-50 — `r50FwdChainB` vs `resnet50FwdGraphBFull` (q = 7)
-- ════════════════════════════════════════════════════════════════

def r50IdW0 (mid oc : Nat) : R50IdW mid oc :=
  ⟨fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0⟩

def r50ProjW0 (ic mid oc : Nat) : R50ProjW ic mid oc :=
  ⟨fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0⟩

-- Stem at q = 7: 224 → 112 → 56.
#guard textOf (r50StemFwdB 2 7 "1.0e-05") (·.code) ==
  prettyText 2 (r50StemGraphB "1.0e-05" 2 56 56 (ic := 3) (oc := 64)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%x" _))

-- Identity bottleneck (`s2b1`, 512 → 128 → 512 at 28²).
#guard textOf (bnkIdFwdB 2 128 512 28 "1.0e-05" "s2b1" "%in") (·.code) ==
  prettyText 2 (r50IdGraphB "s2b1" "1.0e-05" 2 28 28 (r50IdW0 128 512) (leaf "%in" _))

-- Stride-1 projection bottleneck (`s1b0`, 64 → 64 → 256 at 56²).
#guard textOf (bnkProjFwdB 2 64 64 256 56 "1.0e-05" "s1b0" "%in") (·.code) ==
  prettyText 2 (r50ProjGraphB "s1b0" "1.0e-05" 2 56 56 (r50ProjW0 64 64 256) (leaf "%in" _))

-- Strided projection bottleneck (`s3b0`, 512 → 256 → 1024, 28² → 14²).
#guard textOf (bnkStridedFwdB 2 512 256 1024 14 "1.0e-05" "s3b0" "%in") (·.code) ==
  prettyText 2 (r50DownGraphB "s3b0" "1.0e-05" 2 14 14 (r50ProjW0 512 256 1024) (leaf "%in" _))

-- Head: GAP(7²) → dense(2048 → 10) — ResNet-34's head graph at c = 2048.
#guard textOf (r50HeadFwdB 2 7 10 "%in") (·.1) ==
  prettyText 2 (r34HeadGraphB 2 7 7 (c := 2048) (nCls := 10) (fun _ _ => 0) (fun _ => 0)
    (leaf "%in" _))

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV2 — `mnv2FwdChainB` vs `mobilenetv2FwdGraphBFull`
-- ════════════════════════════════════════════════════════════════

def mnv2IVW0 (ic mid oc : Nat) : IVW ic mid oc :=
  ⟨fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0⟩

def mnv2NoExpW0 (ic oc : Nat) : IVWNoExp ic oc :=
  ⟨fun _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, 0, fun _ => 0, fun _ => 0⟩

-- Stem: 3×3/s2 SAME, 224 → 112.
#guard textOf (mnv2StemFwdB 2 "1.0e-03" false) (·.code) ==
  prettyText 2 (mnv2StemGraphB "1.0e-03" 2 112 112 (ic := 3) (oc := 32) (kH := 3) (kW := 3)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%x" _))

-- `t = 1` block (b1, 32 → 16 at 112²).
#guard textOf (irFwdNoExpB 2 32 16 112 "1.0e-03" "1" "%in" false) (·.code) ==
  prettyText 2 (mnv2NoExpGraphB "1" "1.0e-03" 2 112 112 (mnv2NoExpW0 32 16) (leaf "%in" _))

-- Strided block (b2, 16 → 96 → 24, 112² → 56²).
#guard textOf (irFwdStridedB 2 16 96 24 56 "1.0e-03" "2" "%in" false) (·.code) ==
  prettyText 2 (mnv2StridedGraphB "2" "1.0e-03" 2 56 56 (mnv2IVW0 16 96 24) (leaf "%in" _))

-- Skip block (b3, 24 → 144 → 24 at 56²).
#guard textOf (irFwdSkipB 2 24 144 24 56 "1.0e-03" "3" "%in" false) (·.code) ==
  prettyText 2 (mnv2ResidGraphB "3" "1.0e-03" 2 56 56 (mnv2IVW0 24 144 24) (leaf "%in" _))

-- Expand, no skip (b11, 64 → 384 → 96 at 14²).
#guard textOf (irFwdNoSkipB 2 64 384 96 14 "1.0e-03" "11" "%in" false) (·.code) ==
  prettyText 2 (mnv2ExpOnlyGraphB "11" "1.0e-03" 2 14 14 (mnv2IVW0 64 384 96) (leaf "%in" _))

-- Head: 1×1 (320 → 1280) → BN → relu6 → GAP(7²) → dense(1280 → 10).
#guard textOf (mnv2HeadFwdB 2 10 "1.0e-03" "%in" false) (·.code) ==
  prettyText 2 (mnv2HeadGraphB "1.0e-03" 2 7 7 (ic := 320) (oc := 1280) (nCls := 10)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (leaf "%in" _))

-- Head with classifier dropout (`cd := true`, the `%do` train steps): `dropoutB` between GAP and dense.
#guard textOf (mnv2HeadFwdB 2 10 "1.0e-03" "%in" false (cd := true)) (·.code) ==
  prettyText 2 (mnv2HeadGraphBDo "1.0e-03" doName 2 7 7 (ic := 320) (oc := 1280) (nCls := 10)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ => 0) (leaf "%in" _))

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV2 at inference — `mnv2FwdChain`'s pieces vs `mobilenetv2FwdGraphPaperEval`'s
--   The eval chain is per-example constructors printed at batch `B`; the guards compare them at
--   `B = 2` with `convBias := false`, the committed artifacts' setting.
-- ════════════════════════════════════════════════════════════════

/-- Zero inference weights for an expand-bearing bottleneck. -/
def ivwEval0 (ic mid oc : Nat) : IVWEval ic mid oc :=
  ⟨fun _ _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0,
   fun _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩

/-- Zero inference weights for the `t = 1` bottleneck. -/
def ivwNoExpEval0 (ic oc : Nat) : IVWNoExpEval ic oc :=
  ⟨fun _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0,
   fun _ _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩

-- Stem: 3×3/s2 XLA-SAME (3 → 32, 224 → 112) → BN → relu6.
#guard textOf (mnv2EvalStemFwd 2 "1.0e-03" false) (·.1) ==
  prettyText 2 (mnv2StemGraphPaperEval "1.0e-03" 0 (ic := 3) (oc := 32) (h := 112) (w := 112)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0)
    (leaf "%x" _))

-- `t = 1` (b1: 32 → 16 at 112²).
#guard textOf (irFwdNoExp 2 32 16 112 "1.0e-03" "1" "%in" false) (·.code) ==
  prettyText 2 (ivNoExpGraphEvalW "1" "1.0e-03" 112 112 0 (ivwNoExpEval0 32 16) (leaf "%in" _))

-- Strided (b2: 16 → 96 → 24, 112² → 56²).
#guard textOf (irFwdStrided 2 16 96 24 56 "1.0e-03" "2" "%in" false) (·.code) ==
  prettyText 2 (ivStridedGraphEvalW "2" "1.0e-03" 56 56 0 (ivwEval0 16 96 24) (leaf "%in" _))

-- Skip (b3: 24 → 144 → 24 at 56²).
#guard textOf (irFwd 2 24 144 24 56 "1.0e-03" "3" "%in" false) (·.code) ==
  prettyText 2 (ivResidGraphEvalW "3" "1.0e-03" 56 56 0 (ivwEval0 24 144 24) (leaf "%in" _))

-- Widening, no skip (b11: 64 → 384 → 96 at 14²).
#guard textOf (irFwdNoSkip 2 64 384 96 14 "1.0e-03" "11" "%in" false) (·.code) ==
  prettyText 2 (ivExpOnlyGraphEvalW "11" "1.0e-03" 14 14 0 (ivwEval0 64 384 96) (leaf "%in" _))

-- Head: 1×1 (320 → 1280) → BN → relu6 → GAP(7²) → dense(→ 10).
#guard textOf (mnv2EvalHeadFwd 2 10 "1.0e-03" false "%in") (·.1) ==
  prettyText 2 (mnv2HeadGraphPaperEval "1.0e-03" 0 (c := 320) (oc := 1280) (h := 7) (w := 7)
    (nC := 10) (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (leaf "%in" _))

-- The batched artifact is the per-example graph at each example: its only reduction is the GAP
-- over `[2, 3]` — nothing in the eval render reduces across the batch axis.
#guard
  let t := mnv2FwdEvalFaithfulV 2 10 "1.0e-03"
  (t.splitOn "across dimensions = [").length == 2 &&
    (t.splitOn "across dimensions = [2, 3]").length == 2

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV4-Conv-M — `mnv4FwdChainB` vs `mnv4FwdGraphBFull`
-- ════════════════════════════════════════════════════════════════

/-- Zero weights for one table row; the BN ε's are `1` so the record's positivity fields hold. -/
def mnv4UibW0 (s : UibSpec) : UibParams s :=
  ⟨DWSlot.ofParams ⟨fun _ _ _ => 0, fun _ => 0, 1, one_pos, fun _ => 0, fun _ => 0⟩,
   fun _ _ _ _ => 0, fun _ => 0, 1, one_pos, fun _ => 0, fun _ => 0,
   DWSlot.ofParams ⟨fun _ _ _ => 0, fun _ => 0, 1, one_pos, fun _ => 0, fun _ => 0⟩,
   fun _ _ _ _ => 0, fun _ => 0, 1, one_pos, fun _ => 0, fun _ => 0⟩

/-- `pretty` of the typed graph for one table row, dispatched as `mnv4FwdGraphBFull` builds it: a
    stride-2 row is `mnv4StridedGraphB` (the post-DW carries the stride; all three of Conv-M's have
    a pre-DW); a stride-1 row is its family's body plus the identity skip (`mnv4SkipGraphB`, spelled
    out here because the leaf is width-polymorphic and a skip row has `ic = oc` only numerically).
    A row with no typed graph — an IB block, or a strided row without a pre-DW — prints `""`, so a
    table that grew one fails. -/
def mnv4RowGraphText (B : Nat) (s : UibSpec) : String :=
  if s.stride2 then
    if s.preDWk > 0 then
      prettyText B (mnv4StridedGraphB "1.0e-03" B s (mnv4UibW0 s) (leaf "%in" _))
    else ""
  else
    let body : SHlo (B * (s.oc * s.h * s.h)) :=
      if s.preDWk > 0 ∧ s.postDWk > 0 then
        mnv4ExtraDWBodyGraphB "1.0e-03" B s (mnv4UibW0 s) (leaf "%in" _)
      else if s.preDWk > 0 then
        mnv4ConvNeXtBodyGraphB "1.0e-03" B s (mnv4UibW0 s) (leaf "%in" _)
      else
        mnv4FfnBodyGraphB "1.0e-03" B s (mnv4UibW0 s) (leaf "%in" _)
    if s.postDWk > 0 ∧ s.preDWk = 0 then "" else
    prettyText B (.addVB body (leaf "%in" _))

-- Stem: 3×3/s2 symmetric, 224 → 112.
#guard textOf (mnv4StemFwdB 2 "1.0e-03") (·.code) ==
  prettyText 2 (mnv4StemGraphB "1.0e-03" 2 112 112 (ic := 3) (oc := 32) (kH := 3) (kW := 3)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%x" _))

-- Fused stage 0: 3×3/s2 conv (32 → 128) → BN → relu → 1×1 project (→ 48) → BN, 112² → 56².
#guard textOf (fusedMbConvFwdStridedB 2 32 48 4 3 56 .train "1.0e-03" "0" "%in") (·.code) ==
  prettyText 2 (mnv4FusedGraphB "1.0e-03" 2 56 56 (ic := 32) (mid := 128) (oc := 48) (kH := 3) (kW := 3)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- All 21 table rows: the render's dispatch vs the typed graph's, row by row.
#guard mnv4Blocks.length == 21 && mnv4Blocks.all fun s =>
  textOf (uibFwdDispatch 2 s .train "1.0e-03" "%in") (·.code) == mnv4RowGraphText 2 s

-- Head: 1×1 (256 → 960) → BN → relu → GAP(7²) → 1×1 (→ 1280) → BN → relu → dense(→ 10).
#guard textOf (mnv4HeadFwdB 2 10 "1.0e-03" "%in") (·.code) ==
  prettyText 2 (mnv4HeadGraphB "1.0e-03" 2 7 7 (c := 256) (mid := 960) (oc := 1280) (nCls := 10)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (leaf "%in" _))

-- Head with classifier dropout (`cd := true`, the `%do` train steps): the same head with `dropoutB`
-- on the dense's input.
#guard textOf (mnv4HeadFwdB 2 10 "1.0e-03" "%in" (cd := true)) (·.code) ==
  prettyText 2 (mnv4HeadGraphBDo "1.0e-03" doName 2 7 7 (c := 256) (mid := 960) (oc := 1280)
    (nCls := 10)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- ════════════════════════════════════════════════════════════════
-- § MobileNetV4-Conv-M at inference — `mnv4FwdChainB … .eval` vs `mnv4FwdGraphBFullEval`
--   at both committed input sizes: `f = 7` (224, `mnv4{,in}_fwd_eval`) and `f = 8` (256,
--   `mnv4in_fwd_eval_s256`), each row at the side `mnv4FwdChainB` scales it to.
-- ════════════════════════════════════════════════════════════════

/-- A zero depthwise slot at inference, at any extent. -/
def mnv4DWEval0 {c : Nat} : (k : Nat) → Mnv4DWEvalSlot c k
  | 0 => PUnit.unit
  | _ + 1 => ⟨fun _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩

/-- Zero inference weights for one table row. -/
def mnv4UibEvalW0 (s : UibSpec) : UibEvalParams s :=
  ⟨mnv4DWEval0 _, fun _ _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0,
   mnv4DWEval0 _, fun _ _ _ _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩

/-- `pretty` of one row's inference graph at side `h`: the strided block, or the body plus the
    identity skip. -/
def mnv4RowGraphTextEval (B : Nat) (s : UibSpec) (h : Nat) : String :=
  if s.stride2 then
    prettyText B (mnv4StridedGraphBEval "1.0e-03" B h 0 s (mnv4UibEvalW0 s) (leaf "%in" _))
  else
    prettyText B (.addVB (mnv4BodyGraphBEval "1.0e-03" B h 0 s (mnv4UibEvalW0 s) (leaf "%in" _))
      (leaf "%in" _))

-- Stem: 3×3/s2 symmetric, 32f → 16f.
#guard [7, 8].all fun f => textOf (mnv4StemFwdB 2 "1.0e-03" .eval (f := f)) (·.code) ==
  prettyText 2 (mnv4StemGraphBEval "1.0e-03" 2 (16 * f) 0 (ic := 3) (oc := 32) (kH := 3) (kW := 3)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (leaf "%x" _))

-- Fused stage 0: 16f → 8f.
#guard [7, 8].all fun f =>
  textOf (fusedMbConvFwdStridedB 2 32 48 4 3 (8 * f) .eval "1.0e-03" "0" "%in") (·.code) ==
  prettyText 2 (mnv4FusedGraphBEval "1.0e-03" 2 (8 * f) 0 (ic := 32) (mid := 128) (oc := 48)
    (kH := 3) (kW := 3)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- All 21 rows, at the side the chain scales each to.
#guard mnv4Blocks.length == 21 && [7, 8].all fun f => mnv4Blocks.all fun s =>
  let s' := if f == 7 then s else { s with h := s.h * f / 7 }
  textOf (uibFwdDispatch 2 s' .eval "1.0e-03" "%in") (·.code) == mnv4RowGraphTextEval 2 s s'.h

-- The graph's ladder is the chain's: rows at 4f / 2f / f, i.e. the table's 28 / 14 / 7 scaled.
#guard [7, 8].all fun f => mnv4Blocks.all fun s =>
  (if f == 7 then s.h else s.h * f / 7) == s.h / 7 * f

-- Head: 1×1 (256 → 960) → BN → relu at f², GAP, 1×1 (→ 1280) → BN → relu, dense (→ 10).
#guard [7, 8].all fun f => textOf (mnv4HeadFwdB 2 10 "1.0e-03" "%in" .eval (f := f)) (·.code) ==
  prettyText 2 (mnv4HeadGraphBEval "1.0e-03" 2 f 0 (c := 256) (mid := 960) (oc := 1280) (nCls := 10)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (leaf "%in" _))

-- ════════════════════════════════════════════════════════════════
-- § EfficientNet-B0 — `enetFwdChain` vs `efficientnetFwdGraphBFull`
-- ════════════════════════════════════════════════════════════════

-- Stem: 3×3/s2 XLA-SAME (3 → 32, 224 → 112) → BN → swish.
#guard textOf (enetStemFwdB 2 .train "1.0e-03" false) (·.code) ==
  prettyText 2 (stemGraphB "1.0e-03" (N := 2) (ic := 3) (oc := 32) (h := 112) (w := 112)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%x" _))

-- MBConv1, no expand (b1: 32 → 16 at 112², 3×3, SE r = 8).
#guard textOf (eFwdNoExp 2 32 16 112 3 8 .train "1.0e-03" "b1" "%in" false) (·.code) ==
  prettyText 2 (mbNoExpGraphB "b1" "1.0e-03" (N := 2) (ic := 32) (oc := 16) (h := 112) (w := 112)
    (kHd := 3) (kWd := 3) (r := 8)
    (fun _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- Strided MBConv6 (b4: 24 → 144 → 40, 56² → 28², 5×5, SE r = 6).
#guard textOf (eFwdStrided 2 24 144 40 28 5 6 .train "1.0e-03" "b4" "%in" false) (·.code) ==
  prettyText 2 (mbStridedGraphB "b4" "1.0e-03" (N := 2) (ic := 24) (mid := 144) (oc := 40) (h := 28)
    (w := 28) (kHd := 5) (kWd := 5) (r := 6)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- Residual MBConv6 (b3: 24 → 144 → 24 at 56², 3×3, SE r = 6), no drop site.
#guard textOf (eFwd 2 24 144 24 56 3 6 .train "1.0e-03" "b3" "%in" false) (·.code) ==
  prettyText 2 (mbResidGraphB "b3" "1.0e-03" (N := 2) (c := 24) (mid := 144) (h := 56) (w := 56)
    (kHd := 3) (kWd := 3) (r := 6)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- Residual MBConv6 with its stochastic-depth site (`sd := true`, b3 = block index 2): `dropPathB`
-- on the branch, then the skip add (`EfficientNetFullB0Drop`).
#guard textOf (eFwd 2 24 144 24 56 3 6 .train "1.0e-03" "b3" "%in" false (some 2)) (·.code) ==
  prettyText 2 (mbResidDropGraphB "b3" "1.0e-03" (dpName 2) (N := 2) (c := 24) (mid := 144) (h := 56)
    (w := 56) (kHd := 3) (kWd := 3) (r := 6)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (some fun _ => 0) (leaf "%in" _))

-- Expand, no skip (b9: 80 → 480 → 112 at 14², 5×5, SE r = 20).
#guard textOf (eFwdNoSkip 2 80 480 112 14 5 20 .train "1.0e-03" "b9" "%in" false) (·.code) ==
  prettyText 2 (mbExpGraphB "b9" "1.0e-03" (N := 2) (ic := 80) (mid := 480) (oc := 112) (h := 14)
    (w := 14) (kHd := 5) (kWd := 5) (r := 20)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0)
    (fun _ _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (leaf "%in" _))

-- Head: 1×1 (320 → 1280) → BN → swish → GAP(7²) → dense(1280 → 10). The chain's classifier dropout
-- sits between the GAP and the dense and is off in the typed graphs' configuration (`cd := false`).
#guard textOf (do
    let hd ← enetHeadFwdB 2 10 .train "1.0e-03" "%in" false
    let (c, _) ← pretty 2 (.batchOp (N := 2) (.dense "%Wd" "%bd" (fun _ _ => 0 : Mat 1280 10) (fun _ => 0))
      (.operand hd.gap (fun _ => 0 : Vec (2 * 1280))))
    pure (hd.code ++ c)) id ==
  prettyText 2 (headGraphB "1.0e-03" (N := 2) (c := 320) (oc := 1280) (h := 7) (w := 7) (nC := 10)
    (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (leaf "%in" _))

-- Head with classifier dropout (`cd := true`, the `%do` renders): `dropoutB` between the GAP and
-- the dense, as `enetFwdChain` emits it.
#guard textOf (do
    let hd ← enetHeadFwdB 2 10 .train "1.0e-03" "%in" false
    let (cDo, nCin) ← pretty 2 (.dropoutB (N := 2) (n := 1280) doName (fun _ => 0 : Vec (2 * 1280))
      (.operand hd.gap (fun _ => 0 : Vec (2 * 1280))))
    let (c, _) ← pretty 2 (.batchOp (N := 2) (.dense "%Wd" "%bd" (fun _ _ => 0 : Mat 1280 10) (fun _ => 0))
      (.operand nCin (fun _ => 0 : Vec (2 * 1280))))
    pure (hd.code ++ cDo ++ c)) id ==
  prettyText 2 (headGraphBDo "1.0e-03" doName (N := 2) (c := 320) (oc := 1280) (h := 7) (w := 7)
    (nC := 10) (fun _ _ _ _ => 0) (fun _ => 0) 0 (fun _ => 0) (fun _ => 0) (fun _ _ => 0) (fun _ => 0)
    (some fun _ => 0) (leaf "%in" _))

end Proofs.StableHLO.FwdGraphTextTies
