import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FoldPaperG
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB

/-! # MobileNetV2's T3 §1a TIE at TRUE BATCH-NORM — the un-fused, batched whole-net thread

`MobileNetV2FoldPaperG.lean` (4b.4) makes every parameter GRADIENT node of the batched
MobileNetV2 train step `den`-faithful for an arbitrary cotangent. This file removes the
"arbitrary": each cotangent is pinned to the one the emitted backward chain delivers, so the whole
train step is `den`-composed forward → loss → backward with no free activation and no symbolic
cotangent. With 4.2b it completes MobileNetV2's T3, and it is `ResNet34StepTieB.lean`'s peer.

⭐⭐ **The block cotangents are NOT derived here.** 4.2b's `mnv2{ExpOnly,Resid,Strided,NoExp}B_has_vjp_at`
ARE the certified block backwards, and `mnv2{Body,DownBody,ResidBlock}BackBatchedGraph_faithful`
(`MobileNetV2BackB0.lean`) already prove the emitted backward subgraphs denote exactly them. The
four `*CotIn_eq_vjp` lemmas below are those statements in this file's vocabulary, and they are what
make the cross-block thread a composition of certified VJPs rather than a re-derivation.

⭐ **One parameter-tie bundle covers twelve of the seventeen blocks.** A skip block and a stride-1
widening have the SAME parameter cotangents — the identity skip changes only the `dx` handed to the
previous block, which is why `MobileNetV2RenderB`'s `irBackStride1GradB` is one function with a
`skip` flag rather than two near-copies. So `mnv2Stride1TiedB` is stated once and instantiated at
`b3`, `b5`, `b6`, `b8`–`b13`, `b15`, `b16` (skip) and `b11`, `b17` (no skip).

⭐ **The loss cotangent is the LABEL-SMOOTHED one, at a general target**, shared with ResNet-34:
`Foundation/SmoothedLossCot.lean`. `MobileNetV2RenderB` composes it from the same six kit ops
(`softmaxRow → subB → scaleB → addVB → shiftB → divConstB`) with α at 0.1 and the target arriving
as the graph input `%onehot` — a soft vector under mixup or cutmix.

⭐ **`N` is a binder.** The artifacts at 32 (`mobilenetv2_adam_train_step`) or 64
(`mobilenetv2in_rmsdp64`) are instances. T3 carries no numerals.

⛔ **The all-reduce, since 4d piece 2 (2026-09-07).** In `mobilenetv2in_rmsdp64` each `*GradB`
node feeds `allReduceMeanF` — the collective as an AST node whose `den` is the replica MEAN of the
per-replica gradient nodes; until then `emitGradAllReduce`, emitted text and a declared carve-out
outside the `SHlo` AST. Every statement below is at the PER-REPLICA gradient node;
`DataParallelNode.lean` composes it with the mean and the tail.

⚠ **`bnInB` and `bnInB_eq_bnBackB` are ResNet-34's, imported rather than copied.** They are the
batched BatchNorm input-cotangent written as the `den` of the emitted backward op, and its identity
with the certified `bnBatchLA` VJP — both net-agnostic, and they happen to live in the file that
first needed them. What MobileNetV2 adds is the TWO-SIDED relu6 mask (`relu6MaskB`, where r34
threads the one-sided `reluMaskB`), the depthwise input-VJP and its XLA-`SAME` strided peer.

## The parameter census is 158, and this file states 210 slots

`MobileNetV2RenderB` defaults to `convBias := false`: every conv, depthwise and project bias is
folded into the BatchNorm that follows it and bound to `zeroBiasPrelude`'s shared zero constant. So
`mobilenetv2_adam_train_step.mlir` carries **158** updated parameter tensors — stem 3 + b1 6 +
16 blocks × 9 + head 3 + dense 2 — and 210 is the census at `convBias := true`. The bias conjuncts
below are kept (one delegation each, and they cover the flag) and are about ops the committed
artifacts do not emit. Nothing here weakens a theorem: every fold is `∀`-quantified over op
instances, and `bias = 0` is one of them.

## Conventions, stated because nothing here checks them

| | |
|---|---|
| BatchNorm | **batch** (`bnBatchLA`, reduce `[0,2,3]`, width `N·h·w`), 52 sites |
| padding | **XLA-`SAME`** at all five stride-2 sites (the stem conv, the four strided depthwises) |
| activation | relu6, TWO kinks per site (`≠ 0` and `≠ 6`), 35 sites, none after a `project` |
| optimizer form | the RAW gradient (`*GradB`); this net's renders emit no fused `θ − lr·g` at all |
| loss | label-smoothed softmax-CE at a general target, batch-meaned |
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.MobileNetV2TieB

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB bnBackB cInB dInB gapInB)
open Proofs.ResNet34TieB (bnInB bnInB_eq_bnBackB unrowB rowB)

-- ════════════════════════════════════════════════════════════════
-- § The two chain helpers MobileNetV2 adds
-- ════════════════════════════════════════════════════════════════

/-- **The relu6 backward mask** — `den (.selectMidB _ pre e) = fun i => if 0 < pre i ∧ pre i < 6
    then e i else 0`. TWO-sided, where ResNet-34's `reluMaskB` tests `pre i > 0` only. MobileNetV2
    applies it at 35 sites: two per expand-bearing block, one in `b1`, one at the stem, one in the
    head. -/
noncomputable def relu6MaskB (n : Nat) (pre dy : Vec n) : Vec n :=
  fun i => if 0 < pre i ∧ pre i < 6 then dy i else 0

/-- **Batched XLA-`SAME` STRIDED depthwise input-VJP** (= `den depthwiseStridedXlaBackBatched`;
    upsamples `h → 2h`). ⚠ NOT `EnetTiePoC.dStridedInB`, which is the SYMMETRIC
    `depthwiseStride2Flat` — B0's strided depthwise and MobileNetV2's have identical types and
    different certificates, and this is the one place that distinction is recorded on the backward
    side. -/
noncomputable def dStridedXlaInB (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (dy : Vec (N * (c * h * w))) : Vec (N * (c * (2 * h) * (2 * w))) :=
  batchMap N (fun d => (depthwiseStride2FlatXla_has_vjp W b).backward (fun _ => 0) d) dy

-- ════════════════════════════════════════════════════════════════
-- § The `t = 1` block (b1) — `projB ∘ dwbrB`, no expand and no skip
-- ════════════════════════════════════════════════════════════════

/-! `MobileNetV2RenderB.irBackNoExpGradB`, node for node, from the block-output cotangent `dyOut`:

```
%dpc = bnBatchBack(pg, pc)   %ddr = convBackBatched(pW)   %ddm = selectMidB(dn)
%ddn = bnBatchBack(dg, dc)   %dxb = depthwiseBackBatched(dW)
```

with `dW,db ← %ddn`, `dg,dbt ← %ddm`, `pW,pb ← %dpc`, `pg,pbt ← %dy`. -/

/-- Cotangent at the project conv's output. Feeds `pW`/`pb`. -/
noncomputable def mnv2NoExpCotPc (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.pε p.pγ
    (batchMap N (flatConv p.pW p.pb)
      (dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ xin))
    dyOut

/-- Cotangent at the depthwise BN's output — the project conv's input-VJP masked by the depthwise
    relu6. Feeds `dg`/`dbt`. -/
noncomputable def mnv2NoExpCotDn (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * h * w)) :=
  relu6MaskB (N * (ic * h * w))
    (bnBatchLA N ic h w p.dε p.dγ p.dβ (batchMap N (depthwiseFlat p.dW p.db) xin))
    (cInB N p.pW p.pb (mnv2NoExpCotPc N h w p xin dyOut))

/-- Cotangent at the depthwise conv's output. Feeds `dW`/`db`. -/
noncomputable def mnv2NoExpCotDc (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * h * w)) :=
  bnInB N ic h w p.dε p.dγ (batchMap N (depthwiseFlat p.dW p.db) xin)
    (mnv2NoExpCotDn N h w p xin dyOut)

/-- **The block-INPUT cotangent**: the depthwise backward, directly — no expand conv and no skip. -/
noncomputable def mnv2NoExpCotIn (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * h * w)) :=
  dInB N p.dW p.db (mnv2NoExpCotDc N h w p xin dyOut)

/-- The `t = 1` block's backward graph: the two stage graphs chained at their forward activations.
    ⚠ It lives here rather than in `MobileNetV2BackB0.lean` because `mnv2NoExpB` is a wrapper of
    `MobileNetV2FullB.lean`'s, one tier above that file's vocabulary. -/
noncomputable def mnv2NoExpBackGraph {N ic oc h w : Nat} (p : IVWNoExp ic oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  dwbrBackBatchedGraph p.dW p.db p.dε p.dγ p.dβ x
    (projBackBatchedGraph p.pW p.pb p.pε p.pγ p.pβ
      (dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ x) e)

theorem mnv2NoExpBackGraph_faithful {N ic oc h w : Nat} (p : IVWNoExp ic oc) (hq : IVNoExpPos p)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (hs : IVNoExpSmoothAtB N h w p x) :
    den (mnv2NoExpBackGraph p x e) = (mnv2NoExpB_has_vjp_at N h w p hq x hs).backward (den e) := by
  rw [mnv2NoExpBackGraph, dwbrBackBatchedGraph_faithful (hε := hq.hd) (h_smooth := hs.hd),
      projBackBatchedGraph_faithful (hε := hq.hp)]
  simp only [mnv2NoExpB_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt, Function.comp_apply]

/-- ⭐⭐ **The emitted `t = 1` chain IS the certified block VJP's backward.** -/
theorem mnv2NoExpCotIn_eq_vjp (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hq : IVNoExpPos p) (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w)))
    (hs : IVNoExpSmoothAtB N h w p xin) (cotN : String) :
    mnv2NoExpCotIn N h w p xin dyOut = (mnv2NoExpB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := mnv2NoExpBackGraph_faithful p hq xin (.operand cotN dyOut) hs
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [← h]
  rfl


-- ════════════════════════════════════════════════════════════════
-- § The stride-1 inverted-residual body — shared by the ten skip blocks and the two widenings
-- ════════════════════════════════════════════════════════════════

/-! `MobileNetV2RenderB.irBackStride1GradB`, node for node, from the block-output cotangent `dyOut`:

```
%dpc = bnBatchBack(pg, pc)   %ddr = convBackBatched(pW)         %ddm = selectMidB(dn)
%ddn = bnBatchBack(dg, dc)   %der = depthwiseBackBatched(dW)    %dem = selectMidB(en)
%den = bnBatchBack(eg, ec)   %dxb = convBackBatched(eW)
%dx  = if skip then addVB(%dxb, %dy) else %dxb
```

with `eW,eb ← %den`, `eg,ebt ← %dem`, `dW,db ← %ddn`, `dg,dbt ← %ddm`, `pW,pb ← %dpc`,
`pg,pbt ← %dy`. ⭐ The `skip` flag touches ONLY `%dx`, so all twelve parameter cotangents below are
shared between the skip blocks and the two stage-first widenings (`b11`, `b17`). -/

/-- The expand stage's output — the depthwise's input. -/
@[reducible] noncomputable def mnv2XE (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : Vec (N * (mid * h * w)) :=
  cbrB N (h := h) (w := w) p.eW p.eb p.eε p.eγ p.eβ xin

/-- Cotangent at the project conv's output. Feeds `pW`/`pb`. -/
noncomputable def mnv2CotPc (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.pε p.pγ
    (batchMap N (flatConv p.pW p.pb)
      (dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ (mnv2XE N h w p xin)))
    dyOut

/-- Cotangent at the depthwise BN's output — the project conv's input-VJP masked by the depthwise
    relu6. Feeds `dg`/`dbt`. -/
noncomputable def mnv2CotDn (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  relu6MaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.dε p.dγ p.dβ
      (batchMap N (depthwiseFlat p.dW p.db) (mnv2XE N h w p xin)))
    (cInB N p.pW p.pb (mnv2CotPc N h w p xin dyOut))

/-- Cotangent at the depthwise conv's output. Feeds `dW`/`db`. -/
noncomputable def mnv2CotDc (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.dε p.dγ (batchMap N (depthwiseFlat p.dW p.db) (mnv2XE N h w p xin))
    (mnv2CotDn N h w p xin dyOut)

/-- Cotangent at the expand BN's output — the depthwise's input-VJP masked by the expand relu6.
    Feeds `eg`/`ebt`. -/
noncomputable def mnv2CotEn (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  relu6MaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.eε p.eγ p.eβ (batchMap N (flatConv p.eW p.eb) xin))
    (dInB N p.dW p.db (mnv2CotDc N h w p xin dyOut))

/-- Cotangent at the expand conv's output. Feeds `eW`/`eb`. -/
noncomputable def mnv2CotEc (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.eε p.eγ (batchMap N (flatConv p.eW p.eb) xin)
    (mnv2CotEn N h w p xin dyOut)

/-- **The BODY's input cotangent** — the expand conv's backward. This is the whole block-input
    cotangent for a widening (`b11`, `b17`); a skip block adds `dyOut` to it. -/
noncomputable def mnv2CotInBody (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * h * w)) :=
  cInB N p.eW p.eb (mnv2CotEc N h w p xin dyOut)

/-- ⭐⭐ **The emitted stride-1 chain IS the certified body VJP's backward** — the widening blocks'
    `_eq_vjp`, straight from `mnv2BodyBackBatchedGraph_faithful`. -/
theorem mnv2ExpOnlyCotIn_eq_vjp (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc) (hq : IVPos p)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w)))
    (hs : IVSmoothAtB N h w p xin) (cotN : String) :
    mnv2CotInBody N h w p xin dyOut
      = (mnv2ExpOnlyB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := mnv2BodyBackBatchedGraph_faithful (N := N) p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ xin (.operand cotN dyOut)
    hs.he hs.hd
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [mnv2ExpOnlyB_has_vjp_at, ← h]
  rfl

/-- **The SKIP block's input cotangent**: the body branch plus the identity skip, the `addVB`
    fan-in the render emits. -/
noncomputable def mnv2ResidCotIn (N h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (xin dyOut : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  fun i => mnv2CotInBody N h w p xin dyOut i + dyOut i

/-- ⭐⭐ **The emitted residual fan-in IS the certified skip-block VJP's backward**, from
    `mnv2ResidBlockBackBatchedGraph_faithful`. ⭐ No `add_comm` is needed here, unlike ResNet-34's
    downsample block: the render emits `addVB(body, %dy)` and `residualBackGraph` builds the fan-in
    in the same order. -/
theorem mnv2ResidCotIn_eq_vjp (N h w : Nat) {c mid : Nat} (p : IVW c mid c) (hq : IVPos p)
    (xin dyOut : Vec (N * (c * h * w))) (hs : IVSmoothAtB N h w p xin) (cotN : String) :
    mnv2ResidCotIn N h w p xin dyOut
      = (mnv2ResidB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := mnv2ResidBlockBackBatchedGraph_faithful (N := N) p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ xin (.operand cotN dyOut)
    hs.he hs.hd
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  -- ⚠ `rw [← h]` cannot close this one: `mnv2ResidB_has_vjp_at` unfolds to `residual_has_vjp_at`
  -- at `mnv2ExpOnlyB`, while the graph lemma states it at that abbreviation's own unfolding. The
  -- two are definitionally equal but not syntactically, so the step goes through `Eq.trans`.
  refine Eq.trans ?_ h
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The stride-2 downsampling block (b2, b4, b7, b14)
-- ════════════════════════════════════════════════════════════════

/-! `MobileNetV2RenderB.irBackStridedGradB`: the stride-1 chain with the depthwise replaced by its
XLA-`SAME` strided peer, so the expand half runs at the `2h × 2w` input grid and `%der` upsamples.
There is no skip, so `%dx` is `%dxb` directly. -/

/-- The expand stage's output at the pre-downsample grid — the strided depthwise's input. -/
@[reducible] noncomputable def mnv2XES (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  cbrB N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε p.eγ p.eβ xin

/-- Cotangent at the project conv's output. Feeds `pW`/`pb`. -/
noncomputable def mnv2SCotPc (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w p.pε p.pγ
    (batchMap N (flatConv p.pW p.pb)
      (dwbrBstrided N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ (mnv2XES N h w p xin)))
    dyOut

/-- Cotangent at the strided depthwise BN's output. Feeds `dg`/`dbt`. -/
noncomputable def mnv2SCotDn (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  relu6MaskB (N * (mid * h * w))
    (bnBatchLA N mid h w p.dε p.dγ p.dβ
      (batchMap N (depthwiseStride2FlatXla p.dW p.db) (mnv2XES N h w p xin)))
    (cInB N p.pW p.pb (mnv2SCotPc N h w p xin dyOut))

/-- Cotangent at the strided depthwise conv's output. Feeds `dW`/`db`. -/
noncomputable def mnv2SCotDc (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * h * w)) :=
  bnInB N mid h w p.dε p.dγ
    (batchMap N (depthwiseStride2FlatXla p.dW p.db) (mnv2XES N h w p xin))
    (mnv2SCotDn N h w p xin dyOut)

/-- Cotangent at the expand BN's output, at the `2h × 2w` grid — the STRIDED depthwise's input-VJP
    (which upsamples) masked by the expand relu6. Feeds `eg`/`ebt`. -/
noncomputable def mnv2SCotEn (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  relu6MaskB (N * (mid * (2 * h) * (2 * w)))
    (bnBatchLA N mid (2 * h) (2 * w) p.eε p.eγ p.eβ (batchMap N (flatConv p.eW p.eb) xin))
    (dStridedXlaInB N p.dW p.db (mnv2SCotDc N h w p xin dyOut))

/-- Cotangent at the expand conv's output. Feeds `eW`/`eb`. -/
noncomputable def mnv2SCotEc (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnInB N mid (2 * h) (2 * w) p.eε p.eγ (batchMap N (flatConv p.eW p.eb) xin)
    (mnv2SCotEn N h w p xin dyOut)

/-- **The block-INPUT cotangent**: the expand conv's backward. No skip — the downsample changes
    both spatial and channels. -/
noncomputable def mnv2StridedCotIn (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) :
    Vec (N * (ic * (2 * h) * (2 * w))) :=
  cInB N p.eW p.eb (mnv2SCotEc N h w p xin dyOut)

/-- ⭐⭐ **The emitted stride-2 chain IS the certified downsample-body VJP's backward.** -/
theorem mnv2StridedCotIn_eq_vjp (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc) (hq : IVPos p)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w)))
    (hs : IVStridedSmoothAtB N h w p xin) (cotN : String) :
    mnv2StridedCotIn N h w p xin dyOut
      = (mnv2StridedB_has_vjp_at N h w p hq xin hs).backward dyOut := by
  have h := mnv2DownBodyBackBatchedGraph_faithful (N := N) p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ xin (.operand cotN dyOut)
    hs.he hs.hd
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [mnv2StridedB_has_vjp_at, ← h]
  rfl


-- ════════════════════════════════════════════════════════════════
-- § The stem — the render's chain from block 1's input cotangent
-- ════════════════════════════════════════════════════════════════

/-! `MobileNetV2RenderB`, after the seventeen block backwards:

```
%dsm = selectMidB(stn)   %dsn = bnBatchBack(sg, stc)
```

with `sW,sb ← %dsn` and `sg,sbt ← %dsm`. ⭐ There is NO conv-back past `%x`, and no pool: the stem
is one XLA-`SAME` strided conv, its BatchNorm and one relu6. -/

/-- Cotangent at the stem BN's output — the stem relu6's mask. Feeds `sg`/`sbt`. -/
noncomputable def mnv2StemCotN (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (cotStem : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  relu6MaskB (N * (oc * h * w))
    (bnBatchLA N oc h w εs γs βs (batchMap N (flatConvStride2Xla Ws bs) x)) cotStem

/-- Cotangent at the stem conv's output. Feeds `sW`/`sb`. -/
noncomputable def mnv2StemCotC (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (cotStem : Vec (N * (oc * h * w))) :
    Vec (N * (oc * h * w)) :=
  bnInB N oc h w εs γs (batchMap N (flatConvStride2Xla Ws bs) x)
    (mnv2StemCotN N h w Ws bs εs γs βs x cotStem)

-- ════════════════════════════════════════════════════════════════
-- § The head — 1x1 conv-BN-relu6, GAP, dense; then the loss cotangent
-- ════════════════════════════════════════════════════════════════

/-! `MobileNetV2RenderB`, from the loss cotangent `g`:

```
%dgi = denseRowBack(Wd)   %dgp = gapBackBatched   %dhm = selectMidB(hn)
%dhn = bnBatchBack(hg, hc)   %dhx = convBackBatched(hW)
```

with `Wd,bd ← %dy`, `hW,hb ← %dhn`, `hg,hbt ← %dhm`, and `%dhx` the cotangent handed to `b17`.
⚠ Unlike ResNet-34's, this head is NOT `batchMap` of a smooth per-example map — MobileNetV2 puts a
1x1 conv-BN-relu6 in front of the pool — so the chain is spelled here as `den`s, exactly as
`EfficientNetStepTie.enetHeadTied` spells B0's identically-shaped head. -/

/-- Cotangent at the GAP output — the classifier's input-VJP. -/
noncomputable def mnv2HeadCotGapIn (N : Nat) {oc nCls : Nat} (Wd : Mat oc nCls)
    (g : Vec (N * nCls)) : Vec (N * oc) :=
  rowDenseBackFlat N oc nCls Wd g

/-- Cotangent at the head relu6's output — the GAP backward. -/
noncomputable def mnv2HeadCotHr (N h w : Nat) {oc nCls : Nat} (Wd : Mat oc nCls)
    (g : Vec (N * nCls)) : Vec (N * (oc * h * w)) :=
  gapInB N oc h w (mnv2HeadCotGapIn N Wd g)

/-- Cotangent at the head BN's output — the head relu6's mask. Feeds `hg`/`hbt`. -/
noncomputable def mnv2HeadCotHn (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1)
    (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls)
    (xin : Vec (N * (ic * h * w))) (g : Vec (N * nCls)) : Vec (N * (oc * h * w)) :=
  relu6MaskB (N * (oc * h * w))
    (bnBatchLA N oc h w εh γh βh (batchMap N (flatConv Wh bh) xin))
    (mnv2HeadCotHr N h w Wd g)

/-- Cotangent at the head conv's output. Feeds `hW`/`hb`. -/
noncomputable def mnv2HeadCotHc (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1)
    (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls)
    (xin : Vec (N * (ic * h * w))) (g : Vec (N * nCls)) : Vec (N * (oc * h * w)) :=
  bnInB N oc h w εh γh (batchMap N (flatConv Wh bh) xin)
    (mnv2HeadCotHn N h w Wh bh εh γh βh Wd xin g)

/-- **The cotangent the head hands to `b17`** — the head conv's backward. -/
noncomputable def mnv2HeadCotBlk (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1)
    (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls)
    (xin : Vec (N * (ic * h * w))) (g : Vec (N * nCls)) : Vec (N * (ic * h * w)) :=
  cInB N Wh bh (mnv2HeadCotHc N h w Wh bh εh γh βh Wd xin g)


-- ════════════════════════════════════════════════════════════════
-- § The per-block-type tie bundles — every parameter node at its chain cotangent
-- ════════════════════════════════════════════════════════════════

/-! Each conjunct is `MobileNetV2FoldPaperG`'s `∀ cot` fold instantiated at the cotangent the
render's chain delivers, so nothing here is a new proof: the bundles are the §1 fold with the
freedom removed. `reassocB` bridges the conv/relu6 index `N·(c·h·w)` to the BatchNorm parameter
ops' `N·(c·(h·w))`.

⚠ The BIAS conjuncts are about `conv{,StridedXla}BiasGradB` and `depthwise{,StridedXla}BiasGradB`,
which the committed artifacts do NOT emit — `MobileNetV2RenderB` runs `convBias := false` and binds
every bias operand to `zeroBiasPrelude`'s zero constant. They are kept because they cost one
delegation each and they cover the flag. -/

/-- **Stem, tied.** The 3x3/s2 XLA-`SAME` conv's weight and bias and its BatchNorm's γ/β, at the
    cotangent that reaches the stem through block 1's input fan-in. ⚠ `convStridedXla*`, not r34's
    symmetric `convStrided*`: identical types, identical emitted shapes, different certificates. -/
def mnv2StemTiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (cotStem : Vec (N * (oc * h * w))) : Prop :=
  let sc := batchMap N (flatConvStride2Xla Ws bs) x
  let cotN' := mnv2StemCotN N h w Ws bs εs γs βs x cotStem
  let cotC := mnv2StemCotC N h w Ws bs εs γs βs x cotStem
  (∀ idx : Fin (oc * ic * 3 * 3),
      den (SHlo.convStridedXlaWeightGradB xN bs x Ws (.operand cotN cotC)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 3 * 3) =>
                    flatConvStride2Xla (Kernel4.unflatten v') bs
                      (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                 (Kernel4.flatten Ws) idx j * batchSlice N (oc * h * w) cotC n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convStridedXlaBiasGradB (h := h) (w := w) Ws x bs (.operand cotN cotC)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    flatConvStride2Xla Ws b' (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                 bs o j * batchSlice N (oc * h * w) cotC n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr εs (reassocB N oc h w sc)
            (.operand cotN (reassocB N oc h w cotN'))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εs γ' βs (bnchwFwd N oc h w (reassocB N oc h w sc)))
                 γs k j * bnchwFwd N oc h w (reassocB N oc h w cotN') j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotN'))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εs γs β' (bnchwFwd N oc h w (reassocB N oc h w sc)))
                 βs k j * bnchwFwd N oc h w (reassocB N oc h w cotN') j)

theorem mnv2_stem_tiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic 3 3) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (cotStem : Vec (N * (oc * h * w))) :
    mnv2StemTiedB N h w xN cotN vN epsStr Ws bs εs γs βs x cotStem := by
  unfold mnv2StemTiedB
  intro sc cotN' cotC
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoCG.convStridedXlaWGradB_den xN cotN bs x Ws cotC idx
  · intro o;   exact Mnv2PaperPoCG.convStridedXlaBGradB_den cotN Ws x bs cotC o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN εs γs βs
                 (reassocB N oc h w sc) (reassocB N oc h w cotN') k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN εs γs βs
                 (bnchwFwd N oc h w (reassocB N oc h w sc)) (reassocB N oc h w cotN') k

/-- **`t = 1` block (b1), tied.** All eight parameter nodes — the stride-1 depthwise's weight and
    bias and its BatchNorm's γ/β, then the project 1x1's weight and bias and its BatchNorm's γ/β.
    ⭐ The project BatchNorm's γ/β read `dyOut` itself: the linear bottleneck has no activation
    after `project`, so the block-output cotangent IS that BatchNorm's output cotangent. -/
def mnv2NoExpTiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String) (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  let dc := batchMap N (depthwiseFlat p.dW p.db) xin
  let dr := dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ xin
  let pc := batchMap N (flatConv p.pW p.pb) dr
  let cotPc := mnv2NoExpCotPc N h w p xin dyOut
  let cotDn := mnv2NoExpCotDn N h w p xin dyOut
  let cotDc := mnv2NoExpCotDc N h w p xin dyOut
  (∀ idx : Fin (ic * 3 * 3),
      den (SHlo.depthwiseWeightGradB xN p.db xin p.dW (.operand cotN cotDc)) idx
        = ∑ n : Fin N, ∑ j : Fin (ic * h * w),
            pdiv (fun v' : Vec (ic * 3 * 3) =>
                    Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') p.db
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 (Tensor3.flatten p.dW) idx j * batchSlice N (ic * h * w) cotDc n j)
  ∧ (∀ o : Fin ic,
      den (SHlo.depthwiseBiasGradB p.dW xin p.db (.operand cotN cotDc)) o
        = ∑ n : Fin N, ∑ j : Fin (ic * h * w),
            pdiv (fun b' : Vec ic =>
                    Tensor3.flatten (depthwiseConv2d p.dW b'
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 p.db o j * batchSlice N (ic * h * w) cotDc n j)
  ∧ (∀ k : Fin ic,
      den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N ic h w dc)
            (.operand cotN (reassocB N ic h w cotDn))) k
        = ∑ j : Fin (ic * (N * (h * w))),
            pdiv (fun γ' : Vec ic =>
                    bnPerChannelFlat ic (N * (h * w)) p.dε γ' p.dβ (bnchwFwd N ic h w (reassocB N ic h w dc)))
                 p.dγ k j * bnchwFwd N ic h w (reassocB N ic h w cotDn) j)
  ∧ (∀ k : Fin ic,
      den (SHlo.bnBetaGradB (N := N) (oc := ic) (h := h) (w := w)
            (.operand cotN (reassocB N ic h w cotDn))) k
        = ∑ j : Fin (ic * (N * (h * w))),
            pdiv (fun β' : Vec ic =>
                    bnPerChannelFlat ic (N * (h * w)) p.dε p.dγ β' (bnchwFwd N ic h w (reassocB N ic h w dc)))
                 p.dβ k j * bnchwFwd N ic h w (reassocB N ic h w cotDn) j)
  ∧ (∀ idx : Fin (oc * ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.pb dr p.pW (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.pb
                      (Tensor3.unflatten (batchSlice N (ic * h * w) dr n))))
                 (Kernel4.flatten p.pW) idx j * batchSlice N (oc * h * w) cotPc n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convBiasGradB (h := h) (w := w) p.pW dr p.pb (.operand cotN cotPc)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    Tensor3.flatten (conv2d p.pW b'
                      (Tensor3.unflatten (batchSlice N (ic * h * w) dr n))))
                 p.pb o j * batchSlice N (oc * h * w) cotPc n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
            (.operand cotN (reassocB N oc h w dyOut))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.pε γ' p.pβ (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 p.pγ k j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w dyOut))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.pε p.pγ β' (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 p.pβ k j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)

theorem mnv2_noexp_tiedB (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (p : IVWNoExp ic oc) (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    mnv2NoExpTiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold mnv2NoExpTiedB
  intro dc dr pc cotPc cotDn cotDc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoCG.depthwiseWGradB_den xN cotN p.db xin p.dW cotDc idx
  · intro o;   exact Mnv2PaperPoCG.depthwiseBGradB_den cotN p.dW xin p.db cotDc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.dε p.dγ p.dβ
                 (reassocB N ic h w dc) (reassocB N ic h w cotDn) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.dε p.dγ p.dβ
                 (bnchwFwd N ic h w (reassocB N ic h w dc)) (reassocB N ic h w cotDn) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.pb dr p.pW cotPc idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.pW dr p.pb cotPc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.pε p.pγ p.pβ
                 (reassocB N oc h w pc) (reassocB N oc h w dyOut) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.pε p.pγ p.pβ
                 (bnchwFwd N oc h w (reassocB N oc h w pc)) (reassocB N oc h w dyOut) k


/-- **Stride-1 inverted-residual block, tied — all twelve parameter nodes.** ⭐ ONE statement for
    twelve of the seventeen blocks: the ten identity-skip ones and the two stage-first widenings
    (`b11`, `b17`). A skip changes only the `dx` handed to the previous block, never a parameter
    cotangent, which is why `MobileNetV2RenderB.irBackStride1GradB` is one function with a flag. -/
def mnv2Stride1TiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String)
    (p : IVW ic mid oc) (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) : Prop :=
  let ec := batchMap N (flatConv p.eW p.eb) xin
  let er := mnv2XE N h w p xin
  let dc := batchMap N (depthwiseFlat p.dW p.db) er
  let dr := dwbrB N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ er
  let pc := batchMap N (flatConv p.pW p.pb) dr
  let cotPc := mnv2CotPc N h w p xin dyOut
  let cotDn := mnv2CotDn N h w p xin dyOut
  let cotDc := mnv2CotDc N h w p xin dyOut
  let cotEn := mnv2CotEn N h w p xin dyOut
  let cotEc := mnv2CotEc N h w p xin dyOut
  -- expand 1x1 (ic → mid), cot = cotEc
  (∀ idx : Fin (mid * ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.eb xin p.eW (.operand cotN cotEc)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.eb
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 (Kernel4.flatten p.eW) idx j * batchSlice N (mid * h * w) cotEc n j)
  ∧ (∀ o : Fin mid,
      den (SHlo.convBiasGradB (h := h) (w := w) p.eW xin p.eb (.operand cotN cotEc)) o
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun b' : Vec mid =>
                    Tensor3.flatten (conv2d p.eW b'
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 p.eb o j * batchSlice N (mid * h * w) cotEc n j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnGammaGradB vN epsStr p.eε (reassocB N mid h w ec)
            (.operand cotN (reassocB N mid h w cotEn))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun γ' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) p.eε γ' p.eβ (bnchwFwd N mid h w (reassocB N mid h w ec)))
                 p.eγ k j * bnchwFwd N mid h w (reassocB N mid h w cotEn) j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N mid h w cotEn))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun β' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) p.eε p.eγ β' (bnchwFwd N mid h w (reassocB N mid h w ec)))
                 p.eβ k j * bnchwFwd N mid h w (reassocB N mid h w cotEn) j)
  -- depthwise 3x3 stride-1 (mid), cot = cotDc
  ∧ (∀ idx : Fin (mid * 3 * 3),
      den (SHlo.depthwiseWeightGradB xN p.db er p.dW (.operand cotN cotDc)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * 3 * 3) =>
                    Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') p.db
                      (Tensor3.unflatten (batchSlice N (mid * h * w) er n))))
                 (Tensor3.flatten p.dW) idx j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ o : Fin mid,
      den (SHlo.depthwiseBiasGradB p.dW er p.db (.operand cotN cotDc)) o
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun b' : Vec mid =>
                    Tensor3.flatten (depthwiseConv2d p.dW b'
                      (Tensor3.unflatten (batchSlice N (mid * h * w) er n))))
                 p.db o j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N mid h w dc)
            (.operand cotN (reassocB N mid h w cotDn))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun γ' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) p.dε γ' p.dβ (bnchwFwd N mid h w (reassocB N mid h w dc)))
                 p.dγ k j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N mid h w cotDn))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun β' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) p.dε p.dγ β' (bnchwFwd N mid h w (reassocB N mid h w dc)))
                 p.dβ k j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  -- project 1x1 (mid → oc), cot = cotPc; its BN reads dyOut itself (no activation after project)
  ∧ (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN p.pb dr p.pW (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.pb
                      (Tensor3.unflatten (batchSlice N (mid * h * w) dr n))))
                 (Kernel4.flatten p.pW) idx j * batchSlice N (oc * h * w) cotPc n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convBiasGradB (h := h) (w := w) p.pW dr p.pb (.operand cotN cotPc)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    Tensor3.flatten (conv2d p.pW b'
                      (Tensor3.unflatten (batchSlice N (mid * h * w) dr n))))
                 p.pb o j * batchSlice N (oc * h * w) cotPc n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
            (.operand cotN (reassocB N oc h w dyOut))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.pε γ' p.pβ (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 p.pγ k j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w dyOut))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.pε p.pγ β' (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 p.pβ k j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)

theorem mnv2_stride1_tiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String)
    (p : IVW ic mid oc) (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) :
    mnv2Stride1TiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold mnv2Stride1TiedB
  intro ec er dc dr pc cotPc cotDn cotDc cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.eb xin p.eW cotEc idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.eW xin p.eb cotEc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.eε p.eγ p.eβ
                 (reassocB N mid h w ec) (reassocB N mid h w cotEn) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.eε p.eγ p.eβ
                 (bnchwFwd N mid h w (reassocB N mid h w ec)) (reassocB N mid h w cotEn) k
  · intro idx; exact EnetPoCG.depthwiseWGradB_den xN cotN p.db er p.dW cotDc idx
  · intro o;   exact Mnv2PaperPoCG.depthwiseBGradB_den cotN p.dW er p.db cotDc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.dε p.dγ p.dβ
                 (reassocB N mid h w dc) (reassocB N mid h w cotDn) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.dε p.dγ p.dβ
                 (bnchwFwd N mid h w (reassocB N mid h w dc)) (reassocB N mid h w cotDn) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.pb dr p.pW cotPc idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.pW dr p.pb cotPc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.pε p.pγ p.pβ
                 (reassocB N oc h w pc) (reassocB N oc h w dyOut) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.pε p.pγ p.pβ
                 (bnchwFwd N oc h w (reassocB N oc h w pc)) (reassocB N oc h w dyOut) k


/-- **Stride-2 downsampling block, tied — all twelve parameter nodes** (`b2`, `b4`, `b7`, `b14`).
    Identical to the stride-1 profile except that the expand half runs at the `2h x 2w` input grid
    and the depthwise is the XLA-`SAME` strided one. ⚠ `depthwiseStridedXla*GradB`, NOT B0's
    symmetric `depthwiseStrided*GradB`: the two have identical types and identical emitted shapes,
    and only the certificate says which correlation the weight gradient runs. -/
def mnv2Stride2TiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String)
    (p : IVW ic mid oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyOut : Vec (N * (oc * h * w))) : Prop :=
  let ec := batchMap N (flatConv p.eW p.eb) xin
  let er := mnv2XES N h w p xin
  let dc := batchMap N (depthwiseStride2FlatXla p.dW p.db) er
  let dr := dwbrBstrided N (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ er
  let pc := batchMap N (flatConv p.pW p.pb) dr
  let cotPc := mnv2SCotPc N h w p xin dyOut
  let cotDn := mnv2SCotDn N h w p xin dyOut
  let cotDc := mnv2SCotDc N h w p xin dyOut
  let cotEn := mnv2SCotEn N h w p xin dyOut
  let cotEc := mnv2SCotEc N h w p xin dyOut
  -- expand 1x1 (ic → mid) at the pre-downsample grid, cot = cotEc
  (∀ idx : Fin (mid * ic * 1 * 1),
      den (SHlo.convWeightGradB xN p.eb xin p.eW (.operand cotN cotEc)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * (2 * h) * (2 * w)),
            pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.eb
                      (Tensor3.unflatten (batchSlice N (ic * (2 * h) * (2 * w)) xin n))))
                 (Kernel4.flatten p.eW) idx j * batchSlice N (mid * (2 * h) * (2 * w)) cotEc n j)
  ∧ (∀ o : Fin mid,
      den (SHlo.convBiasGradB (h := 2 * h) (w := 2 * w) p.eW xin p.eb (.operand cotN cotEc)) o
        = ∑ n : Fin N, ∑ j : Fin (mid * (2 * h) * (2 * w)),
            pdiv (fun b' : Vec mid =>
                    Tensor3.flatten (conv2d p.eW b'
                      (Tensor3.unflatten (batchSlice N (ic * (2 * h) * (2 * w)) xin n))))
                 p.eb o j * batchSlice N (mid * (2 * h) * (2 * w)) cotEc n j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnGammaGradB vN epsStr p.eε (reassocB N mid (2 * h) (2 * w) ec)
            (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEn))) k
        = ∑ j : Fin (mid * (N * ((2 * h) * (2 * w)))),
            pdiv (fun γ' : Vec mid =>
                    bnPerChannelFlat mid (N * ((2 * h) * (2 * w))) p.eε γ' p.eβ
                      (bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) ec)))
                 p.eγ k j * bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) cotEn) j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := 2 * h) (w := 2 * w)
            (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEn))) k
        = ∑ j : Fin (mid * (N * ((2 * h) * (2 * w)))),
            pdiv (fun β' : Vec mid =>
                    bnPerChannelFlat mid (N * ((2 * h) * (2 * w))) p.eε p.eγ β'
                      (bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) ec)))
                 p.eβ k j * bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) cotEn) j)
  -- XLA-SAME strided depthwise 3x3/s2 (mid), cot = cotDc
  ∧ (∀ idx : Fin (mid * 3 * 3),
      den (SHlo.depthwiseStridedXlaWeightGradB xN p.db er p.dW (.operand cotN cotDc)) idx
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun v' : Vec (mid * 3 * 3) =>
                    depthwiseStride2FlatXla (Tensor3.unflatten v') p.db
                      (batchSlice N (mid * (2 * h) * (2 * w)) er n))
                 (Tensor3.flatten p.dW) idx j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ o : Fin mid,
      den (SHlo.depthwiseStridedXlaBiasGradB (h := h) (w := w) p.dW er p.db
            (.operand cotN cotDc)) o
        = ∑ n : Fin N, ∑ j : Fin (mid * h * w),
            pdiv (fun b' : Vec mid =>
                    depthwiseStride2FlatXla p.dW b'
                      (batchSlice N (mid * (2 * h) * (2 * w)) er n))
                 p.db o j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnGammaGradB vN epsStr p.dε (reassocB N mid h w dc)
            (.operand cotN (reassocB N mid h w cotDn))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun γ' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) p.dε γ' p.dβ (bnchwFwd N mid h w (reassocB N mid h w dc)))
                 p.dγ k j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  ∧ (∀ k : Fin mid,
      den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w)
            (.operand cotN (reassocB N mid h w cotDn))) k
        = ∑ j : Fin (mid * (N * (h * w))),
            pdiv (fun β' : Vec mid =>
                    bnPerChannelFlat mid (N * (h * w)) p.dε p.dγ β' (bnchwFwd N mid h w (reassocB N mid h w dc)))
                 p.dβ k j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  -- project 1x1 (mid → oc), cot = cotPc; its BN reads dyOut itself
  ∧ (∀ idx : Fin (oc * mid * 1 * 1),
      den (SHlo.convWeightGradB xN p.pb dr p.pW (.operand cotN cotPc)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') p.pb
                      (Tensor3.unflatten (batchSlice N (mid * h * w) dr n))))
                 (Kernel4.flatten p.pW) idx j * batchSlice N (oc * h * w) cotPc n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convBiasGradB (h := h) (w := w) p.pW dr p.pb (.operand cotN cotPc)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    Tensor3.flatten (conv2d p.pW b'
                      (Tensor3.unflatten (batchSlice N (mid * h * w) dr n))))
                 p.pb o j * batchSlice N (oc * h * w) cotPc n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr p.pε (reassocB N oc h w pc)
            (.operand cotN (reassocB N oc h w dyOut))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.pε γ' p.pβ (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 p.pγ k j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w dyOut))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) p.pε p.pγ β' (bnchwFwd N oc h w (reassocB N oc h w pc)))
                 p.pβ k j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)

theorem mnv2_stride2_tiedB (N h w : Nat) {ic mid oc : Nat} (xN cotN vN epsStr : String)
    (p : IVW ic mid oc) (xin : Vec (N * (ic * (2 * h) * (2 * w))))
    (dyOut : Vec (N * (oc * h * w))) :
    mnv2Stride2TiedB N h w xN cotN vN epsStr p xin dyOut := by
  unfold mnv2Stride2TiedB
  intro ec er dc dr pc cotPc cotDn cotDc cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.eb xin p.eW cotEc idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.eW xin p.eb cotEc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.eε p.eγ p.eβ
                 (reassocB N mid (2 * h) (2 * w) ec) (reassocB N mid (2 * h) (2 * w) cotEn) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.eε p.eγ p.eβ
                 (bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) ec))
                 (reassocB N mid (2 * h) (2 * w) cotEn) k
  · intro idx; exact Mnv2PaperPoCG.depthwiseStridedXlaWGradB_den xN cotN p.db er p.dW cotDc idx
  · intro o;   exact Mnv2PaperPoCG.depthwiseStridedXlaBGradB_den cotN p.dW er p.db cotDc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.dε p.dγ p.dβ
                 (reassocB N mid h w dc) (reassocB N mid h w cotDn) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.dε p.dγ p.dβ
                 (bnchwFwd N mid h w (reassocB N mid h w dc)) (reassocB N mid h w cotDn) k
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.pb dr p.pW cotPc idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.pW dr p.pb cotPc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN p.pε p.pγ p.pβ
                 (reassocB N oc h w pc) (reassocB N oc h w dyOut) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN p.pε p.pγ p.pβ
                 (bnchwFwd N oc h w (reassocB N oc h w pc)) (reassocB N oc h w dyOut) k

/-- **Head, tied.** The 1x1 conv's weight and bias, its BatchNorm's γ/β, and the classifier's
    weight and bias, at the loss cotangent `g` and the chain it drives. ⚠ The dense-bias conjunct's
    Jacobian witness carries a zero activation: `dense`'s derivative in `b` is the identity whatever
    `x` is, so the statement is `x`-free (the shape `EfficientNetStepTie`'s bias conjuncts take). -/
def mnv2HeadTiedB (N h w : Nat) {ic oc nCls : Nat} (xN cotN vN epsStr : String)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (ic * h * w))) (g : Vec (N * nCls)) : Prop :=
  let hc := batchMap N (flatConv Wh bh) xin
  let hr := cbrB N (h := h) (w := w) Wh bh εh γh βh xin
  let a := batchMap N (globalAvgPoolFlat oc h w) hr
  let cotHn := mnv2HeadCotHn N h w Wh bh εh γh βh Wd xin g
  let cotHc := mnv2HeadCotHc N h w Wh bh εh γh βh Wd xin g
  (∀ idx : Fin (oc * ic * 1 * 1),
      den (SHlo.convWeightGradB xN bh xin Wh (.operand cotN cotHc)) idx
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                    Tensor3.flatten (conv2d (Kernel4.unflatten v') bh
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 (Kernel4.flatten Wh) idx j * batchSlice N (oc * h * w) cotHc n j)
  ∧ (∀ o : Fin oc,
      den (SHlo.convBiasGradB (h := h) (w := w) Wh xin bh (.operand cotN cotHc)) o
        = ∑ n : Fin N, ∑ j : Fin (oc * h * w),
            pdiv (fun b' : Vec oc =>
                    Tensor3.flatten (conv2d Wh b'
                      (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                 bh o j * batchSlice N (oc * h * w) cotHc n j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnGammaGradB vN epsStr εh (reassocB N oc h w hc)
            (.operand cotN (reassocB N oc h w cotHn))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun γ' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εh γ' βh (bnchwFwd N oc h w (reassocB N oc h w hc)))
                 γh k j * bnchwFwd N oc h w (reassocB N oc h w cotHn) j)
  ∧ (∀ k : Fin oc,
      den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w cotHn))) k
        = ∑ j : Fin (oc * (N * (h * w))),
            pdiv (fun β' : Vec oc =>
                    bnPerChannelFlat oc (N * (h * w)) εh γh β' (bnchwFwd N oc h w (reassocB N oc h w hc)))
                 βh k j * bnchwFwd N oc h w (reassocB N oc h w cotHn) j)
  ∧ (∀ (i : Fin oc) (j : Fin nCls),
      den (SHlo.denseWeightGradB (c := nCls) xN a (.operand cotN g)) (finProdFinEquiv (i, j))
        = ∑ n : Fin N, ∑ k : Fin nCls,
            pdiv (fun v : Vec (oc * nCls) => dense (Mat.unflatten v) bd (batchSlice N oc a n))
                 (Mat.flatten Wd) (finProdFinEquiv (i, j)) k * batchSlice N nCls g n k)
  ∧ (∀ j : Fin nCls,
      den (SHlo.denseBiasGradB (N := N) (.operand cotN g)) j
        = ∑ n : Fin N, ∑ k : Fin nCls,
            pdiv (fun b' : Vec nCls => dense Wd b' (fun _ => 0)) bd j k
              * batchSlice N nCls g n k)

theorem mnv2_head_tiedB (N h w : Nat) {ic oc nCls : Nat} (xN cotN vN epsStr : String)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls)
    (xin : Vec (N * (ic * h * w))) (g : Vec (N * nCls)) :
    mnv2HeadTiedB N h w xN cotN vN epsStr Wh bh εh γh βh Wd bd xin g := by
  unfold mnv2HeadTiedB
  intro hc hr a cotHn cotHc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN bh xin Wh cotHc idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN Wh xin bh cotHc o
  · intro k;   exact ResNet34PoCB.bnGammaGradB_den vN epsStr cotN εh γh βh
                 (reassocB N oc h w hc) (reassocB N oc h w cotHn) k
  · intro k;   exact ResNet34PoCB.bnBetaGradB_den cotN εh γh βh
                 (bnchwFwd N oc h w (reassocB N oc h w hc)) (reassocB N oc h w cotHn) k
  · intro i j; exact ResNet34PoCB.denseWGradB_den xN cotN a Wd bd g i j
  · intro j;   exact EnetPoCG.denseBGradB_den cotN Wd (fun _ => 0) bd g j


-- ════════════════════════════════════════════════════════════════
-- § The whole-net capstone
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 1600000 in
/-- ⭐⭐ **The whole batch-BN MobileNetV2 train step, tied.** Threading
    `mobilenetv2ForwardB_full`'s own prefixes as the block inputs and the label-smoothed loss
    cotangent down through the head chain and the seventeen certified block backwards, every
    parameter GRADIENT node of the net — stem 4, `b1` 8, sixteen blocks x 12, head 4, dense 2 —
    denotes the certified batched `Σ_n` gradient. No free activation and no symbolic cotangent.
    With 4.2b this is MobileNetV2's T3 complete at batch BatchNorm.

    ⭐ **`N` is a binder and there is no smoothness hypothesis.** The folds are `∀ cot` statements
    instantiated at explicitly-constructed cotangents, so the capstone needs neither `0 < ε` nor a
    relu6-kink condition. Those enter only in the four `*CotIn_eq_vjp` lemmas, which say the
    constructed chain IS the certified whole-net backward — the two halves of the tie, kept apart
    because they have different hypotheses.

    ⚠ Of the 210 conjunct slots, the committed artifacts exercise **158**: `MobileNetV2RenderB`
    runs `convBias := false`, so the 52 bias nodes are not emitted (each bias is folded into the
    BatchNorm after it and bound to `zeroBiasPrelude`'s zero constant).

    ⛔ One replica. In `mobilenetv2in_rmsdp64` every gradient node feeds `allReduceMeanF`, an AST
    node since 4d piece 2; `DataParallelNode.lean` composes the per-replica statement with it. -/
theorem mnv2_net_tiedB (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (t : Vec (N * (1 * nCls))) :
    -- the label-smoothed loss cotangent at the real logits and a general target
    let g : Vec (N * nCls) :=
      unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (mobilenetv2ForwardB_full N w x)) t))
    -- the backward chain: the head's own four nodes, then the seventeen certified block backwards
    let dy17 := mnv2HeadCotBlk N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW (mnv2PreB17 N w x) g
    let dy16 := mnv2CotInBody N 7 7 w.b17 (mnv2PreB16 N w x) dy17
    let dy15 := mnv2ResidCotIn N 7 7 w.b16 (mnv2PreB15 N w x) dy16
    let dy14 := mnv2ResidCotIn N 7 7 w.b15 (mnv2PreB14 N w x) dy15
    let dy13 := mnv2StridedCotIn N 7 7 w.b14 (mnv2PreB13 N w x) dy14
    let dy12 := mnv2ResidCotIn N 14 14 w.b13 (mnv2PreB12 N w x) dy13
    let dy11 := mnv2ResidCotIn N 14 14 w.b12 (mnv2PreB11 N w x) dy12
    let dy10 := mnv2CotInBody N 14 14 w.b11 (mnv2PreB10 N w x) dy11
    let dy9 := mnv2ResidCotIn N 14 14 w.b10 (mnv2PreB9 N w x) dy10
    let dy8 := mnv2ResidCotIn N 14 14 w.b9 (mnv2PreB8 N w x) dy9
    let dy7 := mnv2ResidCotIn N 14 14 w.b8 (mnv2PreB7 N w x) dy8
    let dy6 := mnv2StridedCotIn N 14 14 w.b7 (mnv2PreB6 N w x) dy7
    let dy5 := mnv2ResidCotIn N 28 28 w.b6 (mnv2PreB5 N w x) dy6
    let dy4 := mnv2ResidCotIn N 28 28 w.b5 (mnv2PreB4 N w x) dy5
    let dy3 := mnv2StridedCotIn N 28 28 w.b4 (mnv2PreB3 N w x) dy4
    let dy2 := mnv2ResidCotIn N 56 56 w.b3 (mnv2PreB2 N w x) dy3
    let dy1 := mnv2StridedCotIn N 56 56 w.b2 (mnv2PreB1 N w x) dy2
    let cotStem := mnv2NoExpCotIn N 112 112 w.b1 (mnv2PreB0 N w x) dy1
    mnv2StemTiedB N 112 112 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotStem
  ∧ mnv2NoExpTiedB N 112 112 xN cotN vN epsStr w.b1 (mnv2PreB0 N w x) dy1
  ∧ mnv2Stride2TiedB N 56 56 xN cotN vN epsStr w.b2 (mnv2PreB1 N w x) dy2
  ∧ mnv2Stride1TiedB N 56 56 xN cotN vN epsStr w.b3 (mnv2PreB2 N w x) dy3
  ∧ mnv2Stride2TiedB N 28 28 xN cotN vN epsStr w.b4 (mnv2PreB3 N w x) dy4
  ∧ mnv2Stride1TiedB N 28 28 xN cotN vN epsStr w.b5 (mnv2PreB4 N w x) dy5
  ∧ mnv2Stride1TiedB N 28 28 xN cotN vN epsStr w.b6 (mnv2PreB5 N w x) dy6
  ∧ mnv2Stride2TiedB N 14 14 xN cotN vN epsStr w.b7 (mnv2PreB6 N w x) dy7
  ∧ mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b8 (mnv2PreB7 N w x) dy8
  ∧ mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b9 (mnv2PreB8 N w x) dy9
  ∧ mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b10 (mnv2PreB9 N w x) dy10
  ∧ mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b11 (mnv2PreB10 N w x) dy11
  ∧ mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b12 (mnv2PreB11 N w x) dy12
  ∧ mnv2Stride1TiedB N 14 14 xN cotN vN epsStr w.b13 (mnv2PreB12 N w x) dy13
  ∧ mnv2Stride2TiedB N 7 7 xN cotN vN epsStr w.b14 (mnv2PreB13 N w x) dy14
  ∧ mnv2Stride1TiedB N 7 7 xN cotN vN epsStr w.b15 (mnv2PreB14 N w x) dy15
  ∧ mnv2Stride1TiedB N 7 7 xN cotN vN epsStr w.b16 (mnv2PreB15 N w x) dy16
  ∧ mnv2Stride1TiedB N 7 7 xN cotN vN epsStr w.b17 (mnv2PreB16 N w x) dy17
  ∧ mnv2HeadTiedB N 7 7 xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
      (mnv2PreB17 N w x) g := by
  intro g dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotStem
  exact ⟨mnv2_stem_tiedB N 112 112 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotStem,
    mnv2_noexp_tiedB N 112 112 xN cotN vN epsStr w.b1 (mnv2PreB0 N w x) dy1,
    mnv2_stride2_tiedB N 56 56 xN cotN vN epsStr w.b2 (mnv2PreB1 N w x) dy2,
    mnv2_stride1_tiedB N 56 56 xN cotN vN epsStr w.b3 (mnv2PreB2 N w x) dy3,
    mnv2_stride2_tiedB N 28 28 xN cotN vN epsStr w.b4 (mnv2PreB3 N w x) dy4,
    mnv2_stride1_tiedB N 28 28 xN cotN vN epsStr w.b5 (mnv2PreB4 N w x) dy5,
    mnv2_stride1_tiedB N 28 28 xN cotN vN epsStr w.b6 (mnv2PreB5 N w x) dy6,
    mnv2_stride2_tiedB N 14 14 xN cotN vN epsStr w.b7 (mnv2PreB6 N w x) dy7,
    mnv2_stride1_tiedB N 14 14 xN cotN vN epsStr w.b8 (mnv2PreB7 N w x) dy8,
    mnv2_stride1_tiedB N 14 14 xN cotN vN epsStr w.b9 (mnv2PreB8 N w x) dy9,
    mnv2_stride1_tiedB N 14 14 xN cotN vN epsStr w.b10 (mnv2PreB9 N w x) dy10,
    mnv2_stride1_tiedB N 14 14 xN cotN vN epsStr w.b11 (mnv2PreB10 N w x) dy11,
    mnv2_stride1_tiedB N 14 14 xN cotN vN epsStr w.b12 (mnv2PreB11 N w x) dy12,
    mnv2_stride1_tiedB N 14 14 xN cotN vN epsStr w.b13 (mnv2PreB12 N w x) dy13,
    mnv2_stride2_tiedB N 7 7 xN cotN vN epsStr w.b14 (mnv2PreB13 N w x) dy14,
    mnv2_stride1_tiedB N 7 7 xN cotN vN epsStr w.b15 (mnv2PreB14 N w x) dy15,
    mnv2_stride1_tiedB N 7 7 xN cotN vN epsStr w.b16 (mnv2PreB15 N w x) dy16,
    mnv2_stride1_tiedB N 7 7 xN cotN vN epsStr w.b17 (mnv2PreB16 N w x) dy17,
    mnv2_head_tiedB N 7 7 xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb
      (mnv2PreB17 N w x) g⟩


/-- ⭐ **And the cotangent the capstone threads is the smoothed loss's gradient.** Row by row: the
    `g` above is, at example `n` and class `j`, `(1/B)·∂/∂logits` of soft-target cross-entropy
    against the SMOOTHED target `(1−α)·t + α/K`, at that example's real logits. The only hypothesis
    is that the example's target sums to 1 — a one-hot, or mixup's convex combination of two.
    Together with the capstone this closes the top of the chain: every parameter node denotes the
    certified gradient at the cotangent of the loss the trainer actually minimises. Shared with
    ResNet-34 through `Foundation/SmoothedLossCot.lean`, at a general target. -/
theorem mnv2_lossCot_is_smoothedCE_grad (N : Nat) {nCls : Nat} (hK : 0 < nCls)
    (aStr negAK bStr logN ohN : String) (α B : ℝ) (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) (t : Vec (N * (1 * nCls)))
    (n : Fin N) (j : Fin nCls)
    (ht : ∑ k : Fin nCls, Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1) k = 1) :
    den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
          (rowB N nCls (mobilenetv2ForwardB_full N w x)) t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec nCls => fun _ : Fin 1 =>
            softCE nCls (smoothTarget nCls α
              (Mat.unflatten (batchSlice N (1 * nCls) t n) (0 : Fin 1))) z')
          (Mat.unflatten (batchSlice N (1 * nCls)
            (rowB N nCls (mobilenetv2ForwardB_full N w x)) n) (0 : Fin 1)) j 0) / B :=
  smoothedLossCotGraph_row N nCls hK α B aStr negAK bStr logN ohN _ t n j ht

end Proofs.MobileNetV2TieB
