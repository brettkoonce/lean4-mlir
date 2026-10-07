import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB
import LeanMlir.Proofs.Codegen.StableHLO.Pretty

/-! # MobileNetV4-Conv-M with stochastic depth and classifier dropout — forward, graph, faithfulness

`MobileNetV4FullB` states the net and its classifier-dropout form (`mnv4FwdGraphBFullDo`). The
paper-tier train steps `mnv4in_acc{,dp}8x128wxdropdowd01bf16` also carry **stochastic depth**
(`sd`, the `%dp<k>` inputs): a per-example scale `dropPath` on the residual BRANCH, after the
project BN and before the skip add, at each of the eighteen skip rows. Site `k` is the `k`-th
skip row in table order (the renderer's `mnv4DropSites`):

| site | 0 | 1–7 | 8–17 |
|---|---|---|---|
| rows | 2 | 4–10 | 12–21 |

The three strided rows (1, 3, 11) have no skip and no site. This file states that forward with
both regularisers, as the renderer's `uibFwdSkipB … (drop := some k)` and `mnv4HeadFwdB …
(cd := true)` emit them, and proves the typed graph denotes it:

* `mnv4FwdGraphBFullDrop_faithful` — the graph denotes `mobilenetv4ForwardBFullDrop`, at every
  pair of masks;
* `mobilenetv4ForwardBFullDrop_sdOnes` — at the all-ones drop masks the forward is the dropout
  forward `mobilenetv4ForwardBFullDo`; `mobilenetv4ForwardBFullDrop_ones` — at all-ones masks
  everywhere it is `mobilenetv4ForwardBFull`, exactly (the keep probability is folded into the
  mask, `Training/DropPath`).

Those two artifacts are bf16, and the graph takes the renderers' `bf16` flag through the same
switch as `MobileNetV4FullB` (`StableHLO.PrecisionSwitch`), so `mnv4FwdGraphBFullDrop … true` is
their forward read over ℝ (`Bf16Erasure`) and `false` its f32 twin — as `mnv4FwdGraphBFullDo` is
for the classifier-dropout ones. The train steps' backward through the drop sites is outside this
statement, as it is outside `MobileNetV4StepTieB`.

**Why a separate file, and why `rw`.** `MobileNetV4FullB`'s group graphs are private and their
whole-net proof is the one whose `simp only` spelling dies in the kernel, so the drop-carrying
groups are new definitions here rather than edits there. Every group and whole-net step below is
an outside-in `rw` with a faithfulness lemma of the shape `den <subgraph> = <sub>.fwd (den ·)`,
so `den` never meets a literal-width term it could start evaluating.

## References

- Huang et al. 2016, *Deep Networks with Stochastic Depth*. <https://arxiv.org/abs/1603.09382>
- Qin et al. 2024, *MobileNetV4: Universal Models for the Mobile Ecosystem*. <https://arxiv.org/abs/2404.10518>
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § One skip row with its drop site
-- ════════════════════════════════════════════════════════════════

/-- **A skip row with its stochastic-depth site**: the body's output scaled per example by `s`,
    then the identity skip — the reference's `x + _drop_branch(body(x))`. -/
noncomputable def mnv4SkipDrop {N n : Nat} (f : Vec (N * n) → Vec (N * n)) (s : Vec N) :
    Vec (N * n) → Vec (N * n) :=
  Proofs.residual (dropPath N n s ∘ f)

/-- At the all-ones scale the drop-carrying skip row is the plain one. -/
theorem mnv4SkipDrop_ones {N n : Nat} (f : Vec (N * n) → Vec (N * n)) :
    mnv4SkipDrop f (fun _ => 1) = Proofs.residual f := by
  have h : dropPath N n (fun _ => 1) ∘ f = f := funext fun x => dropPath_ones_id N n (f x)
  rw [mnv4SkipDrop, h]

/-- **A skip row's graph with its drop site**: `mnv4SkipGraphB` with a `dropPathB` on the body's
    output, mask input `mN` (the renderer's is `dpName k`). Like `mnv4SkipGraphB`, a named
    combinator so the block input occurs once in the whole-net term. -/
def mnv4SkipDropGraphB {N n : Nat} (mN : String) (s : Vec N)
    (body : SHlo (N * n) → SHlo (N * n)) (e : SHlo (N * n)) : SHlo (N * n) :=
  .addVB (.dropPathB mN s (body e)) e

/-- A drop-carrying skip row denotes `mnv4SkipDrop` of whatever its body denotes — generic in
    both, so one theorem covers all eighteen. -/
theorem mnv4SkipDropGraphB_faithful {N n : Nat} (mN : String) (s : Vec N)
    (body : SHlo (N * n) → SHlo (N * n)) (f : Vec (N * n) → Vec (N * n))
    (hb : ∀ e' : SHlo (N * n), den (body e') = f (den e')) (e : SHlo (N * n)) :
    den (mnv4SkipDropGraphB mN s body e) = mnv4SkipDrop f s (den e) := by
  rw [mnv4SkipDropGraphB, den_addVB, den_dropPathB, hb]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The five resolution groups with their drop sites
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28** with its drop site: row 2 reads site 0. -/
noncomputable def mnv4Res28Drop (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Fin 18 → Vec N) (v : Vec (N * (48 * 56 * 56))) : Vec (N * (80 * 28 * 28)) :=
  mnv4SkipDrop (mnv4BodyOfRow N mnv4Row2 w.b2).fwd (sd 0)
    ((mnv4StridedBodyOfRow N mnv4Row1 w.b1).fwd v)

/-- Trunk group **Res14a** with its drop sites: rows 4–6 read sites 1–3. -/
noncomputable def mnv4Res14aDrop (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Fin 18 → Vec N) (v : Vec (N * (80 * 28 * 28))) : Vec (N * (160 * 14 * 14)) :=
  mnv4SkipDrop (mnv4BodyOfRow N mnv4Row6 w.b6).fwd (sd 3)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row5 w.b5).fwd (sd 2)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row4 w.b4).fwd (sd 1)
    ((mnv4StridedBodyOfRow N mnv4Row3 w.b3).fwd v)))

/-- Trunk group **Res14b** with its drop sites: rows 7–10 read sites 4–7. -/
noncomputable def mnv4Res14bDrop (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Fin 18 → Vec N) (v : Vec (N * (160 * 14 * 14))) : Vec (N * (160 * 14 * 14)) :=
  mnv4SkipDrop (mnv4BodyOfRow N mnv4Row10 w.b10).fwd (sd 7)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row9 w.b9).fwd (sd 6)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row8 w.b8).fwd (sd 5)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row7 w.b7).fwd (sd 4) v)))

/-- Trunk group **Res7a** with its drop sites: rows 12–15 read sites 8–11. -/
noncomputable def mnv4Res7aDrop (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Fin 18 → Vec N) (v : Vec (N * (160 * 14 * 14))) : Vec (N * (256 * 7 * 7)) :=
  mnv4SkipDrop (mnv4BodyOfRow N mnv4Row15 w.b15).fwd (sd 11)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row14 w.b14).fwd (sd 10)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row13 w.b13).fwd (sd 9)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row12 w.b12).fwd (sd 8)
    ((mnv4StridedBodyOfRow N mnv4Row11 w.b11).fwd v))))

/-- Trunk group **Res7b** with its drop sites: rows 16–21 read sites 12–17. -/
noncomputable def mnv4Res7bDrop (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Fin 18 → Vec N) (v : Vec (N * (256 * 7 * 7))) : Vec (N * (256 * 7 * 7)) :=
  mnv4SkipDrop (mnv4BodyOfRow N mnv4Row21 w.b21).fwd (sd 17)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row20 w.b20).fwd (sd 16)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row19 w.b19).fwd (sd 15)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row18 w.b18).fwd (sd 14)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row17 w.b17).fwd (sd 13)
    (mnv4SkipDrop (mnv4BodyOfRow N mnv4Row16 w.b16).fwd (sd 12) v)))))

/-! At the all-ones masks each group is its `CertLayer`'s forward. Stated against the
`*_fwd_apply` expansions, where the terms are variables. -/

theorem mnv4Res28Drop_ones (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (48 * 56 * 56))) :
    mnv4Res28Drop N w (fun _ _ => 1) v = (mnv4Res28Layer N w).fwd v := by
  rw [mnv4Res28Layer_fwd_apply, mnv4Res28Drop]
  simp only [mnv4SkipDrop_ones, CertLayer.residual_fwd]

theorem mnv4Res14aDrop_ones (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (80 * 28 * 28))) :
    mnv4Res14aDrop N w (fun _ _ => 1) v = (mnv4Res14aLayer N w).fwd v := by
  rw [mnv4Res14aLayer_fwd_apply, mnv4Res14aDrop]
  simp only [mnv4SkipDrop_ones, CertLayer.residual_fwd]

theorem mnv4Res14bDrop_ones (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (160 * 14 * 14))) :
    mnv4Res14bDrop N w (fun _ _ => 1) v = (mnv4Res14bLayer N w).fwd v := by
  rw [mnv4Res14bLayer_fwd_apply, mnv4Res14bDrop]
  simp only [mnv4SkipDrop_ones, CertLayer.residual_fwd]

theorem mnv4Res7aDrop_ones (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (160 * 14 * 14))) :
    mnv4Res7aDrop N w (fun _ _ => 1) v = (mnv4Res7aLayer N w).fwd v := by
  rw [mnv4Res7aLayer_fwd_apply, mnv4Res7aDrop]
  simp only [mnv4SkipDrop_ones, CertLayer.residual_fwd]

theorem mnv4Res7bDrop_ones (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (256 * 7 * 7))) :
    mnv4Res7bDrop N w (fun _ _ => 1) v = (mnv4Res7bLayer N w).fwd v := by
  rw [mnv4Res7bLayer_fwd_apply, mnv4Res7bDrop]
  simp only [mnv4SkipDrop_ones, CertLayer.residual_fwd]

-- ════════════════════════════════════════════════════════════════
-- § The whole net with both regularisers
-- ════════════════════════════════════════════════════════════════

/-- **MobileNetV4-Conv-M with stochastic depth and classifier dropout**:
    `mobilenetv4ForwardBFullDo` with `sd`'s eighteen per-example scales on the skip rows'
    branches (site `k` = the `k`-th skip row, the module's table) and `m` before the classifier. -/
noncomputable def mobilenetv4ForwardBFullDrop (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (sd : Fin 18 → Vec N) (m : Vec (N * 1280)) (x : Vec (N * (3 * 224 * 224))) :
    Vec (N * nCls) :=
  batchMap N (dense w.Wd w.bd) (dropout m
    (mnv4HeadFeatB N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
      (mnv4Res7bDrop N w sd (mnv4Res7aDrop N w sd (mnv4Res14bDrop N w sd
        (mnv4Res14aDrop N w sd (mnv4Res28Drop N w sd (mnv4Pre1 N w x))))))))

/-- At the all-ones drop masks the forward is the classifier-dropout forward. -/
theorem mobilenetv4ForwardBFullDrop_sdOnes (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (m : Vec (N * 1280)) (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardBFullDrop N w (fun _ _ => 1) m x = mobilenetv4ForwardBFullDo N w m x := by
  unfold mobilenetv4ForwardBFullDrop mobilenetv4ForwardBFullDo
    mnv4Pre6 mnv4Pre5 mnv4Pre4 mnv4Pre3 mnv4Pre2
  rw [mnv4Res28Drop_ones, mnv4Res14aDrop_ones, mnv4Res14bDrop_ones, mnv4Res7aDrop_ones,
    mnv4Res7bDrop_ones]

/-- **At the all-ones masks the forward is `mobilenetv4ForwardBFull`**, exactly. -/
theorem mobilenetv4ForwardBFullDrop_ones (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) :
    mobilenetv4ForwardBFullDrop N w (fun _ _ => 1) (fun _ => 1) x
      = mobilenetv4ForwardBFull N w x := by
  rw [mobilenetv4ForwardBFullDrop_sdOnes, mobilenetv4ForwardBFullDo_ones]

-- ════════════════════════════════════════════════════════════════
-- § The group graphs with their drop sites + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- Trunk group **Res28**'s graph with its drop site. -/
def mnv4Res28DropGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (48 * 56 * 56))) :
    SHlo (N * (80 * 28 * 28)) :=
  mnv4SkipDropGraphB (dpName 0) (sd 0) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row2 w.b2 bf16)
    (mnv4StridedGraphB epsStr N mnv4Row1 w.b1 bf16 e)

theorem mnv4Res28DropGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (48 * 56 * 56))) :
    den (mnv4Res28DropGraphB N epsStr w bf16 sd e) = mnv4Res28Drop N w sd (den e) := by
  rw [mnv4Res28DropGraphB,
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row2 w.b2 bf16 (by decide) (by decide)),
    mnv4StridedGraphB_faithful epsStr N mnv4Row1 w.b1 bf16 (by decide), mnv4Res28Drop]

/-- Trunk group **Res14a**'s graph with its drop sites. -/
def mnv4Res14aDropGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (80 * 28 * 28))) :
    SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipDropGraphB (dpName 3) (sd 3) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row6 w.b6 bf16)
    (mnv4SkipDropGraphB (dpName 2) (sd 2) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row5 w.b5 bf16)
    (mnv4SkipDropGraphB (dpName 1) (sd 1) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row4 w.b4 bf16)
    (mnv4StridedGraphB epsStr N mnv4Row3 w.b3 bf16 e)))

theorem mnv4Res14aDropGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (80 * 28 * 28))) :
    den (mnv4Res14aDropGraphB N epsStr w bf16 sd e) = mnv4Res14aDrop N w sd (den e) := by
  rw [mnv4Res14aDropGraphB,
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row6 w.b6 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row5 w.b5 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row4 w.b4 bf16 (by decide) (by decide)),
    mnv4StridedGraphB_faithful epsStr N mnv4Row3 w.b3 bf16 (by decide), mnv4Res14aDrop]

/-- Trunk group **Res14b**'s graph with its drop sites. -/
def mnv4Res14bDropGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (160 * 14 * 14))) :
    SHlo (N * (160 * 14 * 14)) :=
  mnv4SkipDropGraphB (dpName 7) (sd 7) (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row10 w.b10 bf16)
    (mnv4SkipDropGraphB (dpName 6) (sd 6) (mnv4FfnBodyGraphB epsStr N mnv4Row9 w.b9 bf16)
    (mnv4SkipDropGraphB (dpName 5) (sd 5) (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row8 w.b8 bf16)
    (mnv4SkipDropGraphB (dpName 4) (sd 4) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row7 w.b7 bf16) e)))

theorem mnv4Res14bDropGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (160 * 14 * 14))) :
    den (mnv4Res14bDropGraphB N epsStr w bf16 sd e) = mnv4Res14bDrop N w sd (den e) := by
  rw [mnv4Res14bDropGraphB,
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row10 w.b10 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row9 w.b9 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row8 w.b8 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row7 w.b7 bf16 (by decide) (by decide)),
    mnv4Res14bDrop]

/-- Trunk group **Res7a**'s graph with its drop sites. -/
def mnv4Res7aDropGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (160 * 14 * 14))) :
    SHlo (N * (256 * 7 * 7)) :=
  mnv4SkipDropGraphB (dpName 11) (sd 11) (mnv4FfnBodyGraphB epsStr N mnv4Row15 w.b15 bf16)
    (mnv4SkipDropGraphB (dpName 10) (sd 10) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row14 w.b14 bf16)
    (mnv4SkipDropGraphB (dpName 9) (sd 9) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row13 w.b13 bf16)
    (mnv4SkipDropGraphB (dpName 8) (sd 8) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row12 w.b12 bf16)
    (mnv4StridedGraphB epsStr N mnv4Row11 w.b11 bf16 e))))

theorem mnv4Res7aDropGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (160 * 14 * 14))) :
    den (mnv4Res7aDropGraphB N epsStr w bf16 sd e) = mnv4Res7aDrop N w sd (den e) := by
  rw [mnv4Res7aDropGraphB,
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row15 w.b15 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row14 w.b14 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row13 w.b13 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row12 w.b12 bf16 (by decide) (by decide)),
    mnv4StridedGraphB_faithful epsStr N mnv4Row11 w.b11 bf16 (by decide), mnv4Res7aDrop]

/-- Trunk group **Res7b**'s graph with its drop sites. -/
def mnv4Res7bDropGraphB (N : Nat) (epsStr : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (256 * 7 * 7))) : SHlo (N * (256 * 7 * 7)) :=
  mnv4SkipDropGraphB (dpName 17) (sd 17) (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row21 w.b21 bf16)
    (mnv4SkipDropGraphB (dpName 16) (sd 16) (mnv4FfnBodyGraphB epsStr N mnv4Row20 w.b20 bf16)
    (mnv4SkipDropGraphB (dpName 15) (sd 15) (mnv4FfnBodyGraphB epsStr N mnv4Row19 w.b19 bf16)
    (mnv4SkipDropGraphB (dpName 14) (sd 14) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row18 w.b18 bf16)
    (mnv4SkipDropGraphB (dpName 13) (sd 13) (mnv4ExtraDWBodyGraphB epsStr N mnv4Row17 w.b17 bf16)
    (mnv4SkipDropGraphB (dpName 12) (sd 12) (mnv4ConvNeXtBodyGraphB epsStr N mnv4Row16 w.b16 bf16)
      e)))))

theorem mnv4Res7bDropGraphB_faithful (N : Nat) (epsStr : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Fin 18 → Vec N) (e : SHlo (N * (256 * 7 * 7))) :
    den (mnv4Res7bDropGraphB N epsStr w bf16 sd e) = mnv4Res7bDrop N w sd (den e) := by
  rw [mnv4Res7bDropGraphB,
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row21 w.b21 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row20 w.b20 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4FfnBodyGraphB_faithful epsStr N mnv4Row19 w.b19 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row18 w.b18 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ExtraDWBodyGraphB_faithful epsStr N mnv4Row17 w.b17 bf16 (by decide) (by decide)),
    mnv4SkipDropGraphB_faithful _ _ _ _
      (mnv4ConvNeXtBodyGraphB_faithful epsStr N mnv4Row16 w.b16 bf16 (by decide) (by decide)),
    mnv4Res7bDrop]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net graph + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- **The MobileNetV4-Conv-M forward graph with stochastic depth and classifier dropout** — the
    typed form of the forward half of `mnv4in_acc{,dp}8x128wxdropdowd01bf16` (at `bf16 := true`;
    `false` is its f32 twin): each skip row's
    drop site reads `%dp<k>` at its site index `k`, the classifier dropout the input `mName` (the
    render's is `doName`). -/
def mnv4FwdGraphBFullDrop (N : Nat) (epsStr mName : String) {nCls : Nat} (w : Mnv4BWeights nCls)
    (bf16 : Bool) (sd : Fin 18 → Vec N) (m : Vec (N * 1280)) (e : SHlo (N * (3 * 224 * 224))) :
    SHlo (N * nCls) :=
  mnv4HeadGraphBDo epsStr mName N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt
    w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd bf16 m
    (mnv4Res7bDropGraphB N epsStr w bf16 sd
      (mnv4Res7aDropGraphB N epsStr w bf16 sd
        (mnv4Res14bDropGraphB N epsStr w bf16 sd
          (mnv4Res14aDropGraphB N epsStr w bf16 sd
            (mnv4Res28DropGraphB N epsStr w bf16 sd
              (mnv4FusedGraphB epsStr N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt bf16
                (mnv4StemGraphB epsStr N 112 112 w.sW w.sb w.sE w.sg w.sbt bf16 e)))))))

/-- **The graph denotes the forward, at every pair of masks** — eight outside-in rewrites, as
    `mnv4FwdGraphBFullDo_faithful`. -/
theorem mnv4FwdGraphBFullDrop_faithful (N : Nat) (epsStr mName : String) {nCls : Nat}
    (w : Mnv4BWeights nCls) (bf16 : Bool) (sd : Fin 18 → Vec N) (m : Vec (N * 1280))
    (e : SHlo (N * (3 * 224 * 224))) :
    den (mnv4FwdGraphBFullDrop N epsStr mName w bf16 sd m e)
      = mobilenetv4ForwardBFullDrop N w sd m (den e) := by
  unfold mnv4FwdGraphBFullDrop mobilenetv4ForwardBFullDrop mnv4Pre1 mnv4Pre0
  rw [mnv4HeadDo_graph_faithful, mnv4Res7bDropGraphB_faithful, mnv4Res7aDropGraphB_faithful,
      mnv4Res14bDropGraphB_faithful, mnv4Res14aDropGraphB_faithful, mnv4Res28DropGraphB_faithful,
      mnv4FusedStack_graph_faithful, mnv4StemB_graph_faithful]

end StableHLO

end Proofs
