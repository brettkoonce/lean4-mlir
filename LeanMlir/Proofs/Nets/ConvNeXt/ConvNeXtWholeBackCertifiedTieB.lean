import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.BatchMapVJPAt

/-! # `convnextInputGradB` is the certified whole-net ConvNeXt-T gradient at a batch

`ConvNeXtWholeBackCertifiedTie.lean` proves for one image that `convnextInputGrad`, the reverse of
`convNextForwardTCh`, is the certified gradient at that image. The batched step tie
(`ConvNeXtStepTieGB.lean`) states every activation as `StableHLO.batchMap B` of the per-example
prefix and every cotangent as `batchMapAux B` of the per-example chain, because channel LayerNorm
is per-example and no ConvNeXt op couples examples. This file proves the same at that index: the
twelve-stage batched chain `convnextInputGradB` (`ConvNeXtBackChains.lean`), every slot the
per-example slot lifted at the batched saved activation, is the certified gradient of
`batchMap B convNextForwardTCh` at every batch `x`, for every `B` and every class count `nC`.

The forward is `convNextForwardTCh`: ConvNeXt-T (`[3,3,9,3]` at `96→192→384→768`), without
drop-path, in exact real arithmetic. ConvNeXt is smooth everywhere, so every stage has a global
`HasVJP` and its batched witness is `batchMapHasVJPAt` over `HasVJP.toHasVJPAt` at each row. The
hypotheses are the 23 LayerNorm positivities the per-example tie carries and no smooth-point
condition.

1. `cnxSavedB0 … cnxSavedB10` — the eleven batched stage inputs, as reducible functions of the
   batch saved stage by stage (`batchMap B stage ∘ cnxSavedB_{k-1} B w`), not as `batchMap B` of
   the composed per-example prefix: the two agree only up to `batchMap_comp`, not `rfl`, and the
   apex's `vjpCompDiffAt` produces the former. Each is also its apex level's inner map, so the
   stage witness above it sits at `cnxSavedB_k B w x` on the nose.
2. The twelve batched stage witnesses `cnx*BAt`, each `batchMapHasVJPAt` over the per-example
   `HasVJP` at each row, at the dimension spellings the per-example tie normalised (`cnxDn1`,
   `cnxLNh`, `cnxSavedA0`).
3. The batched leaf ties. GAP's is `rfl` (its per-example tie is); the others are
   `batchMapAux_eq_batchMapHasVJPAt` (or its `batchMap` form for the linear leaves) over the
   per-example leaf tie at each row. The channel-LN and downsample leaves are proved at variable
   dims (`cnxChanLNBackB_eq_vjp`, `cnxDownBackB_eq_vjp`) and instantiated by term.
4. `convNextForwardTChBHasVJPAt` — the twelve-stage apex, eleven `vjpCompDiffAt`s over the
   batched stage witnesses, level `k`'s inner map named `cnxSavedB_k B w` — and
   `convnextInputGradB_eq_convNextForwardTChB_vjp`, the tie: twelve leaf rewrites, then the eleven
   levels peeled by `rw [vjpCompDiffAt_fst_backward]`.
5. `convNextForwardTChB_eq_chain` — `batchMap B` of the per-example twelve-factor chain is the
   twelve batched stages, by `batchMap_comp` — and
   `convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp`, the tie carried to the global witness
   `batchMapHasVJP _ (convNextForwardTChHasVJP …)` through `HasVJPAt.backward_unique_of_eq`, plus
   the `∑ pdiv` reading on `convNextForwardTCh` itself (`convnextInputGradB_correct`).
6. `convnextImagenetInputGradB_eq_vjp` — the same statement at `nC = 1000`, `B` a binder.
-/

-- Proof-shape notes:
-- * Channel-LN and downsample leaves are proved at variable dims and instantiated by term: when
--   they closed by `rfl`, the literal `96 56 56` recursed past `maxRecDepth 100000` on the
--   numerals (the batched form of `ConvNeXtWholeBackCertifiedTie.lean`'s "two spellings of one
--   numeral" rule).
-- * Neither the apex nor the tie may leave the kernel a definitional step across the chain. A
--   witness point spelled as the composed chain applied to `x` (what `_` elaborates to), or a peel
--   by `simp only` (the peel lemma is `rfl`, so simp records no step), makes the kernel unfold
--   saved activations against the chain underneath the witnesses' `.backward`s. Spelled that way
--   this module took ~18 min on Lean 4.32.2 and did not check on 4.34.0 (kernel timeout). As
--   written it checks in seconds.

namespace Proofs

open scoped BigOperators

-- ═════════════════════════════════════════════════
-- § The batched saved activations — stage by stage
-- ═════════════════════════════════════════════════

/-- The batched stem-conv output — the stem LayerNorm's saved input at every example. -/
noncomputable abbrev cnxSavedB0 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (96 * 56 * 56)) :=
  StableHLO.batchMap B (cnxSavedA0 w)

/-- Stage 1's batched saved input. -/
noncomputable abbrev cnxSavedB1 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (96 * 56 * 56)) :=
  StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ) ∘ cnxSavedB0 B w

/-- Downsample 1's batched saved input. -/
noncomputable abbrev cnxSavedB2 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (96 * 56 * 56)) :=
  StableHLO.batchMap B (convNextStageChK 3 w.s1) ∘ cnxSavedB1 B w

/-- Stage 2's batched saved input. -/
noncomputable abbrev cnxSavedB3 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (192 * 28 * 28)) :=
  StableHLO.batchMap B (cnxDn1 w) ∘ cnxSavedB2 B w

/-- Downsample 2's batched saved input. -/
noncomputable abbrev cnxSavedB4 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (192 * 28 * 28)) :=
  StableHLO.batchMap B (convNextStageChK 3 w.s2) ∘ cnxSavedB3 B w

/-- Stage 3's batched saved input. -/
noncomputable abbrev cnxSavedB5 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (384 * 14 * 14)) :=
  StableHLO.batchMap B (cnxDn2 w) ∘ cnxSavedB4 B w

/-- Downsample 3's batched saved input. -/
noncomputable abbrev cnxSavedB6 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (384 * 14 * 14)) :=
  StableHLO.batchMap B (convNextStageChK 9 w.s3) ∘ cnxSavedB5 B w

/-- Stage 4's batched saved input. -/
noncomputable abbrev cnxSavedB7 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (768 * 7 * 7)) :=
  StableHLO.batchMap B (cnxDn3 w) ∘ cnxSavedB6 B w

/-- GAP's batched saved input. -/
noncomputable abbrev cnxSavedB8 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * (768 * 7 * 7)) :=
  StableHLO.batchMap B (convNextStageChK 3 w.s4) ∘ cnxSavedB7 B w

/-- The head LayerNorm's batched saved input. -/
noncomputable abbrev cnxSavedB9 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * 768) :=
  StableHLO.batchMap B (globalAvgPoolFlat 768 7 7) ∘ cnxSavedB8 B w

/-- The classifier's batched saved input. -/
noncomputable abbrev cnxSavedB10 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    Vec (B * (3 * 224 * 224)) → Vec (B * 768) :=
  StableHLO.batchMap B (cnxLNh w) ∘ cnxSavedB9 B w

-- ═════════════════════════════════════════════════
-- § The twelve batched stage witnesses
-- ═════════════════════════════════════════════════

/-- The batched stem-conv witness at `x`. -/
noncomputable def cnxStemBAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : HasVJPAt (StableHLO.batchMap B (cnxSavedA0 w)) x :=
  batchMapHasVJPAt _ x (fun _ => (cnxV0 w).toHasVJPAt _) (fun _ => (cnxSavedA0_differentiable w).differentiableAt)

/-- The batched channel-LayerNorm witness at `v`, at any `c h w`. -/
noncomputable def cnxChanLNBAt (B c h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (v : Vec (B * (c * h * w))) :
    HasVJPAt (StableHLO.batchMap B (chanLNTensor3 c h w ε γ β)) v :=
  batchMapHasVJPAt _ v
    (fun _ => (chanLNTensor3HasVJP c h w ε γ β hε).toHasVJPAt _)
    (fun _ => (chanLNTensor3_differentiable c h w ε γ β hε).differentiableAt)

/-- The batched stem-LayerNorm witness at `v`. -/
noncomputable def cnxStemLNBAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε)
    (v : Vec (B * (96 * 56 * 56))) :
    HasVJPAt (StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)) v :=
  cnxChanLNBAt B 96 56 56 w.sε hsε w.sγ w.sβ v

/-- The batched depth-`k` stage witness at `v` — one definition for all four stages. -/
noncomputable def cnxStageBAt (B : Nat) {c cExp h w kH kW : Nat} (k : Nat)
    (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW) (hε : ∀ i, 0 < (ps i).εn)
    (v : Vec (B * (c * h * w))) :
    HasVJPAt (StableHLO.batchMap B (convNextStageChK k ps)) v :=
  batchMapHasVJPAt _ v
    (fun _ => (convNextStageChKHasVJP k ps hε).toHasVJPAt _)
    (fun _ => (convNextStageChK_differentiable k ps hε).differentiableAt)

/-- The batched downsample witness at `v`, at any resolution and channel pair. -/
noncomputable def cnxDownBAt (B h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) (v : Vec (B * (cin * (2 * h) * (2 * w)))) :
    HasVJPAt (StableHLO.batchMap B (cnxDownChW h w p)) v :=
  batchMapHasVJPAt _ v (fun _ => (cnxDownChWHasVJP h w p hε).toHasVJPAt _)
    (fun _ => (cnxDownChW_differentiable h w p hε).differentiableAt)

/-- The batched downsample-1 witness at `v`, at the chain's dimension spelling (`cnxDn1`). -/
noncomputable def cnxDn1BAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd1 : 0 < w.d1.ε)
    (v : Vec (B * (96 * 56 * 56))) : HasVJPAt (StableHLO.batchMap B (cnxDn1 w)) v :=
  cnxDownBAt B 28 28 w.d1 hd1 v

/-- The batched downsample-2 witness at `v`, at the chain's dimension spelling (`cnxDn2`). -/
noncomputable def cnxDn2BAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε)
    (v : Vec (B * (192 * 28 * 28))) : HasVJPAt (StableHLO.batchMap B (cnxDn2 w)) v :=
  cnxDownBAt B 14 14 w.d2 hd2 v

/-- The batched downsample-3 witness at `v`, at the chain's dimension spelling (`cnxDn3`). -/
noncomputable def cnxDn3BAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε)
    (v : Vec (B * (384 * 14 * 14))) : HasVJPAt (StableHLO.batchMap B (cnxDn3 w)) v :=
  cnxDownBAt B 7 7 w.d3 hd3 v

/-- The batched GAP witness at `v`. -/
noncomputable def cnxGapBAt (B : Nat) (v : Vec (B * (768 * 7 * 7))) :
    HasVJPAt (StableHLO.batchMap B (globalAvgPoolFlat 768 7 7)) v :=
  batchMapHasVJPAt _ v (fun _ => (globalAvgPoolFlatHasVJP 768 7 7).toHasVJPAt _)
    (fun _ => (globalAvgPoolFlat_differentiable 768 7 7).differentiableAt)

/-- The batched head-LayerNorm witness at `v`. -/
noncomputable def cnxLNhBAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε)
    (v : Vec (B * 768)) : HasVJPAt (StableHLO.batchMap B (cnxLNh w)) v :=
  batchMapHasVJPAt _ v (fun _ => (cnxLNhVjp w hhε).toHasVJPAt _)
    (fun _ => (cnxLNh_differentiable w hhε).differentiableAt)

/-- The batched classifier witness at `v`. -/
noncomputable def cnxDenseBAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (v : Vec (B * 768)) :
    HasVJPAt (StableHLO.batchMap B (dense w.Wd w.bd)) v :=
  batchMapHasVJPAt _ v (fun _ => (denseHasVJP w.Wd w.bd).toHasVJPAt _)
    (fun _ => (dense_differentiable w.Wd w.bd).differentiableAt)

-- ═════════════════════════════════════════════════
-- § The twelve batched leaf ties
-- ═════════════════════════════════════════════════

/-- **The batched stem tie.** `batchMap B` of the reversed-kernel conv at the zero-extended 4×4
    kernel IS the lift's backward: `flatConvStride4Back_padOdd_eq_vjp_backward` at one example's
    row. `padOdd` is required here as in the per-example tie: `w.sW` is 4×4. -/
theorem cnxStemBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) :
    StableHLO.batchMap B (flatConvStride4Back (h := 56) (w := 56) (padOdd w.sW))
      = (cnxStemBAt B w x).backward :=
  batchMap_eq_batchMapHasVJPAt _ _ x _ _ fun _ =>
    flatConvStride4Back_padOdd_eq_vjp_backward (h := 56) (w := 56) (by norm_num) (by norm_num)
      w.sW w.sb _

/-- **The batched channel-LayerNorm tie**, at any `c h w` — `chanLNTensor3Back_eq_chanLN_vjp` at
    each row. Note: stated at variable dims because an `rfl` proof at the literal `96 56 56`
    recursed past `maxRecDepth 100000` on the numerals; at variables it is ViT's
    `vitLNBackB_eq_vjp`. The stem instance below is a term. -/
theorem cnxChanLNBackB_eq_vjp (B c h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (v : Vec (B * (c * h * w))) :
    StableHLO.batchMapAux B (chanLNTensor3Back c h w ε γ) v
      = (cnxChanLNBAt B c h w ε hε γ β v).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ v _ _ fun _ =>
    chanLNTensor3Back_eq_chanLN_vjp (β := β) ε hε γ _

/-- **The batched stem-LayerNorm tie** — `cnxChanLNBackB_eq_vjp` at `96 56 56`. -/
theorem cnxStemLNBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε)
    (v : Vec (B * (96 * 56 * 56))) :
    StableHLO.batchMapAux B (chanLNTensor3Back 96 56 56 w.sε w.sγ) v
      = (cnxStemLNBAt B w hsε v).backward :=
  cnxChanLNBackB_eq_vjp B 96 56 56 w.sε hsε w.sγ w.sβ v

/-- **The batched stage tie** — `cnxStageChKBack_eq_vjp` at one example's row, for every depth. -/
theorem cnxStageBackB_eq_vjp (B : Nat) {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kHd kWd) (hε : ∀ i, 0 < (ps i).εn)
    (v : Vec (B * (c * h * w))) :
    StableHLO.batchMapAux B (cnxStageChKBack k ps) v = (cnxStageBAt B k ps hε v).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ v _ _ fun _ => cnxStageChKBack_eq_vjp hkHd hkWd k ps hε _

/-- **The batched downsample tie**, at any resolution — `cnxDownChBack_eq_vjp` at one example's
    row. Generic for the same reason as `cnxChanLNBackB_eq_vjp`; the three instances below are
    terms at the chain's dimension spellings (`cnxDn1 … cnxDn3`), which is the per-example tie's
    `cnxDn1Back_eq_vjp … cnxDn3Back_eq_vjp` one batch index over. `padOdd` is required: `p.W` is
    2×2. -/
theorem cnxDownBackB_eq_vjp (B h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) (v : Vec (B * (cin * (2 * h) * (2 * w)))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := h) (w := w) (padOdd p.W)
        (chanLNTensor3Back cin (2 * h) (2 * w) p.ε p.γ u)) v
      = (cnxDownBAt B h w p hε v).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ v _ _ fun _ => cnxDownChBack_eq_vjp (h := h) (w := w) p hε _

/-- **The batched downsample-1 tie** — `cnxDownBackB_eq_vjp` at `28 28`. -/
theorem cnxDn1BackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd1 : 0 < w.d1.ε)
    (v : Vec (B * (96 * 56 * 56))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
        (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ u)) v
      = (cnxDn1BAt B w hd1 v).backward :=
  cnxDownBackB_eq_vjp B 28 28 w.d1 hd1 v

/-- **The batched downsample-2 tie** — `cnxDownBackB_eq_vjp` at `14 14`. -/
theorem cnxDn2BackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε)
    (v : Vec (B * (192 * 28 * 28))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
        (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ u)) v
      = (cnxDn2BAt B w hd2 v).backward :=
  cnxDownBackB_eq_vjp B 14 14 w.d2 hd2 v

/-- **The batched downsample-3 tie** — `cnxDownBackB_eq_vjp` at `7 7`. -/
theorem cnxDn3BackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε)
    (v : Vec (B * (384 * 14 * 14))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
        (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ u)) v
      = (cnxDn3BAt B w hd3 v).backward :=
  cnxDownBackB_eq_vjp B 7 7 w.d3 hd3 v

/-- **The batched GAP tie is `rfl`**, as the per-example `gapBack_eq_vjp_backward` is. -/
theorem cnxGapBackB_eq_vjp (B : Nat) (v : Vec (B * (768 * 7 * 7))) :
    StableHLO.batchMap B (gapBack 768 7 7) = (cnxGapBAt B v).backward := rfl

/-- **The batched head-LayerNorm tie** — `cnxLNhBack_eq_vjp` at one example's row. -/
theorem cnxLNhBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε)
    (v : Vec (B * 768)) :
    StableHLO.batchMapAux B (rowLNVecFlatBack 1 768 w.hε w.hγ) v
      = (cnxLNhBAt B w hhε v).backward :=
  batchMapAux_eq_batchMapHasVJPAt _ _ v _ _ fun _ => cnxLNhBack_eq_vjp w hhε _

/-- **The batched classifier tie** — `dense_transpose_eq_vjp_backward` at one example's row (the
    head is linear, so the saved `v` is free). -/
theorem cnxDenseBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (v : Vec (B * 768)) :
    StableHLO.batchMap B (dense (Mat.transpose w.Wd) (0 : Vec 768))
      = (cnxDenseBAt B w v).backward :=
  batchMap_eq_batchMapHasVJPAt _ _ v _ _ fun r =>
    dense_transpose_eq_vjp_backward w.Wd w.bd (Mat.unflatten v r)

-- ═════════════════════════════════════════════════
-- § The batched apex, the tie, and the shape check
-- ═════════════════════════════════════════════════

/-- **The batched whole-net witness**, twelve batched stages composed by `vjpCompDiffAt`, each
    at the batched saved activation the chain uses (`cnxSavedB0 … cnxSavedB10`). Level `k`'s inner
    map is named `cnxSavedB_k B w` rather than left to the unifier, which would fill it with the
    composed chain and put every witness at the chain applied to `x`. -/
noncomputable def convNextForwardTChBHasVJPAt (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (B * (3 * 224 * 224))) :
    HasVJPAt
      (StableHLO.batchMap B (dense w.Wd w.bd)
        ∘ StableHLO.batchMap B (cnxLNh w)
        ∘ StableHLO.batchMap B (globalAvgPoolFlat 768 7 7)
        ∘ StableHLO.batchMap B (convNextStageChK 3 w.s4)
        ∘ StableHLO.batchMap B (cnxDn3 w)
        ∘ StableHLO.batchMap B (convNextStageChK 9 w.s3)
        ∘ StableHLO.batchMap B (cnxDn2 w)
        ∘ StableHLO.batchMap B (convNextStageChK 3 w.s2)
        ∘ StableHLO.batchMap B (cnxDn1 w)
        ∘ StableHLO.batchMap B (convNextStageChK 3 w.s1)
        ∘ StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)
        ∘ StableHLO.batchMap B (cnxSavedA0 w)) x :=
  (vjpCompDiffAt (cnxSavedB10 B w) (StableHLO.batchMap B (dense w.Wd w.bd)) x
    (vjpCompDiffAt (cnxSavedB9 B w) (StableHLO.batchMap B (cnxLNh w)) x
      (vjpCompDiffAt (cnxSavedB8 B w) (StableHLO.batchMap B (globalAvgPoolFlat 768 7 7)) x
        (vjpCompDiffAt (cnxSavedB7 B w) (StableHLO.batchMap B (convNextStageChK 3 w.s4)) x
          (vjpCompDiffAt (cnxSavedB6 B w) (StableHLO.batchMap B (cnxDn3 w)) x
            (vjpCompDiffAt (cnxSavedB5 B w) (StableHLO.batchMap B (convNextStageChK 9 w.s3)) x
              (vjpCompDiffAt (cnxSavedB4 B w) (StableHLO.batchMap B (cnxDn2 w)) x
                (vjpCompDiffAt (cnxSavedB3 B w)
                    (StableHLO.batchMap B (convNextStageChK 3 w.s2)) x
                  (vjpCompDiffAt (cnxSavedB2 B w) (StableHLO.batchMap B (cnxDn1 w)) x
                    (vjpCompDiffAt (cnxSavedB1 B w)
                        (StableHLO.batchMap B (convNextStageChK 3 w.s1)) x
                      (vjpCompDiffAt (cnxSavedB0 B w)
                        (StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)) x
                        ⟨cnxStemBAt B w x,
                         batchMap_differentiableAt _ x (fun _ => (cnxSavedA0_differentiable w).differentiableAt)⟩
                        ⟨cnxStemLNBAt B w hsε (cnxSavedB0 B w x),
                         batchMap_differentiableAt _ _ (fun _ =>
                           (chanLNTensor3_differentiable 96 56 56 w.sε w.sγ w.sβ hsε).differentiableAt)⟩)
                      ⟨cnxStageBAt B 3 w.s1 h1 (cnxSavedB1 B w x),
                       batchMap_differentiableAt _ _ (fun _ =>
                         (convNextStageChK_differentiable 3 w.s1 h1).differentiableAt)⟩)
                    ⟨cnxDn1BAt B w hd1 (cnxSavedB2 B w x),
                     batchMap_differentiableAt _ _ (fun _ => (cnxDn1_differentiable w hd1).differentiableAt)⟩)
                  ⟨cnxStageBAt B 3 w.s2 h2 (cnxSavedB3 B w x),
                   batchMap_differentiableAt _ _ (fun _ =>
                     (convNextStageChK_differentiable 3 w.s2 h2).differentiableAt)⟩)
                ⟨cnxDn2BAt B w hd2 (cnxSavedB4 B w x),
                 batchMap_differentiableAt _ _ (fun _ => (cnxDn2_differentiable w hd2).differentiableAt)⟩)
              ⟨cnxStageBAt B 9 w.s3 h3 (cnxSavedB5 B w x),
               batchMap_differentiableAt _ _ (fun _ =>
                 (convNextStageChK_differentiable 9 w.s3 h3).differentiableAt)⟩)
            ⟨cnxDn3BAt B w hd3 (cnxSavedB6 B w x),
             batchMap_differentiableAt _ _ (fun _ => (cnxDn3_differentiable w hd3).differentiableAt)⟩)
          ⟨cnxStageBAt B 3 w.s4 h4 (cnxSavedB7 B w x),
           batchMap_differentiableAt _ _ (fun _ =>
             (convNextStageChK_differentiable 3 w.s4 h4).differentiableAt)⟩)
        ⟨cnxGapBAt B (cnxSavedB8 B w x),
         batchMap_differentiableAt _ _ (fun _ =>
           (globalAvgPoolFlat_differentiable 768 7 7).differentiableAt)⟩)
      ⟨cnxLNhBAt B w hhε (cnxSavedB9 B w x),
       batchMap_differentiableAt _ _ (fun _ => (cnxLNh_differentiable w hhε).differentiableAt)⟩)
    ⟨cnxDenseBAt B w (cnxSavedB10 B w x),
     batchMap_differentiableAt _ _ (fun _ => (dense_differentiable w.Wd w.bd).differentiableAt)⟩).fst

/-- **The batched tie.** `convnextInputGradB` with every slot the per-example slot at the
    batched saved activation is the backward of `convNextForwardTChBHasVJPAt`, at every `B`, `nC`
    and batch `x`, under the 23 LayerNorm positivities. -/
theorem convnextInputGradB_eq_convNextForwardTChB_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (B * (3 * 224 * 224))) :
    convnextInputGradB B w.Wd (padOdd w.sW)
        (chanLNTensor3Back 96 56 56 w.sε w.sγ) (cnxSavedB0 B w x)
        (rowLNVecFlatBack 1 768 w.hε w.hγ) (cnxSavedB9 B w x)
        (cnxStageChKBack 3 w.s1) (cnxSavedB1 B w x)
        (fun u => cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
          (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ u)) (cnxSavedB2 B w x)
        (cnxStageChKBack 3 w.s2) (cnxSavedB3 B w x)
        (fun u => cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
          (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ u)) (cnxSavedB4 B w x)
        (cnxStageChKBack 9 w.s3) (cnxSavedB5 B w x)
        (fun u => cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
          (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ u)) (cnxSavedB6 B w x)
        (cnxStageChKBack 3 w.s4) (cnxSavedB7 B w x)
      = (convNextForwardTChBHasVJPAt B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x).backward := by
  -- Twelve leaf rewrites, the chain's eleven `∘`s applied, then the eleven composition levels
  -- peeled by `vjpCompDiffAt_fst_backward`; every step a `rw`, so the kernel replays rewrites.
  unfold convnextInputGradB
  rw [cnxStemBackB_eq_vjp B w x,
      cnxStemLNBackB_eq_vjp B w hsε (cnxSavedB0 B w x),
      cnxStageBackB_eq_vjp B (by norm_num) (by norm_num) 3 w.s1 h1 (cnxSavedB1 B w x),
      cnxDn1BackB_eq_vjp B w hd1 (cnxSavedB2 B w x),
      cnxStageBackB_eq_vjp B (by norm_num) (by norm_num) 3 w.s2 h2 (cnxSavedB3 B w x),
      cnxDn2BackB_eq_vjp B w hd2 (cnxSavedB4 B w x),
      cnxStageBackB_eq_vjp B (by norm_num) (by norm_num) 9 w.s3 h3 (cnxSavedB5 B w x),
      cnxDn3BackB_eq_vjp B w hd3 (cnxSavedB6 B w x),
      cnxStageBackB_eq_vjp B (by norm_num) (by norm_num) 3 w.s4 h4 (cnxSavedB7 B w x),
      cnxGapBackB_eq_vjp B (cnxSavedB8 B w x),
      cnxLNhBackB_eq_vjp B w hhε (cnxSavedB9 B w x),
      cnxDenseBackB_eq_vjp B w (cnxSavedB10 B w x)]
  funext dy
  repeat rw [Function.comp_apply]
  rw [convNextForwardTChBHasVJPAt]
  repeat rw [vjpCompDiffAt_fst_backward]

/-- **The shape check.** `batchMap B` of the per-example twelve-factor chain — the function
    `convNextForwardTChHasVJP` is stated on — IS the twelve batched stages the apex is stated
    on: `batchMap_comp` eleven times, and the normalised spellings (`cnxLNh`, `cnxDn1 … cnxDn3`,
    `cnxSavedA0`) unfold to the chain's. -/
theorem convNextForwardTChB_eq_chain (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) :
    StableHLO.batchMap B
      (dense w.Wd w.bd ∘
        rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageChK 3 w.s4 ∘
        cnxDownChW 7 7 w.d3 ∘
        convNextStageChK 9 w.s3 ∘
        cnxDownChW 14 14 w.d2 ∘
        convNextStageChK 3 w.s2 ∘
        cnxDownChW 28 28 w.d1 ∘
        convNextStageChK 3 w.s1 ∘
        chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb)
      = StableHLO.batchMap B (dense w.Wd w.bd)
        ∘ StableHLO.batchMap B (cnxLNh w)
        ∘ StableHLO.batchMap B (globalAvgPoolFlat 768 7 7)
        ∘ StableHLO.batchMap B (convNextStageChK 3 w.s4)
        ∘ StableHLO.batchMap B (cnxDn3 w)
        ∘ StableHLO.batchMap B (convNextStageChK 9 w.s3)
        ∘ StableHLO.batchMap B (cnxDn2 w)
        ∘ StableHLO.batchMap B (convNextStageChK 3 w.s2)
        ∘ StableHLO.batchMap B (cnxDn1 w)
        ∘ StableHLO.batchMap B (convNextStageChK 3 w.s1)
        ∘ StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)
        ∘ StableHLO.batchMap B (cnxSavedA0 w) := by
  simp only [batchMap_comp]
  rfl

/-- **The batched apex at the global witness.** `convnextInputGradB` is
    `(batchMapHasVJP _ (convNextForwardTChHasVJP …) …).backward x` — the certified gradient of
    the drop-free per-example ConvNeXt-T forward lifted over `B` examples, for every `B`, every
    `nC` and every batch `x`. The hypotheses are the 23 LayerNorm positivities. Carried from the
    chain-shaped apex by `HasVJPAt.backward_unique_of_eq` along `convNextForwardTChB_eq_chain`. -/
theorem convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp (B : Nat) {nC : Nat}
    (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (B * (3 * 224 * 224))) :
    convnextInputGradB B w.Wd (padOdd w.sW)
        (chanLNTensor3Back 96 56 56 w.sε w.sγ) (cnxSavedB0 B w x)
        (rowLNVecFlatBack 1 768 w.hε w.hγ) (cnxSavedB9 B w x)
        (cnxStageChKBack 3 w.s1) (cnxSavedB1 B w x)
        (fun u => cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
          (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ u)) (cnxSavedB2 B w x)
        (cnxStageChKBack 3 w.s2) (cnxSavedB3 B w x)
        (fun u => cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
          (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ u)) (cnxSavedB4 B w x)
        (cnxStageChKBack 9 w.s3) (cnxSavedB5 B w x)
        (fun u => cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
          (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ u)) (cnxSavedB6 B w x)
        (cnxStageChKBack 3 w.s4) (cnxSavedB7 B w x)
      = (batchMapHasVJP (N := B) _
          (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
          (convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)).backward x := by
  funext dy
  rw [convnextInputGradB_eq_convNextForwardTChB_vjp B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x]
  exact HasVJPAt.backward_unique_of_eq (convNextForwardTChB_eq_chain B w).symm
    (convNextForwardTChBHasVJPAt B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x)
    ((batchMapHasVJP (N := B) _
        (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
        (convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)).toHasVJPAt x) dy

/-- **The batched apex, read as the Jacobian of the committed forward.** `convnextInputGradB` is
    the `pdiv`-contracted Jacobian transpose of `batchMap B (convNextForwardTCh w)` — the
    nested-application forward the graph faithfulness `convNextFwdGraphTCh_faithful` is about —
    at EVERY batch and EVERY cotangent, through `convNextForwardTCh_eq_chain`. -/
theorem convnextInputGradB_correct (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (B * (3 * 224 * 224))) (dy : Vec (B * nC)) (i : Fin (B * (3 * 224 * 224))) :
    convnextInputGradB B w.Wd (padOdd w.sW)
        (chanLNTensor3Back 96 56 56 w.sε w.sγ) (cnxSavedB0 B w x)
        (rowLNVecFlatBack 1 768 w.hε w.hγ) (cnxSavedB9 B w x)
        (cnxStageChKBack 3 w.s1) (cnxSavedB1 B w x)
        (fun u => cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
          (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ u)) (cnxSavedB2 B w x)
        (cnxStageChKBack 3 w.s2) (cnxSavedB3 B w x)
        (fun u => cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
          (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ u)) (cnxSavedB4 B w x)
        (cnxStageChKBack 9 w.s3) (cnxSavedB5 B w x)
        (fun u => cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
          (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ u)) (cnxSavedB6 B w x)
        (cnxStageChKBack 3 w.s4) (cnxSavedB7 B w x) dy i
      = ∑ j : Fin (B * nC),
          pdiv (StableHLO.batchMap B (convNextForwardTCh w)) x i j * dy j := by
  rw [convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x,
      (batchMapHasVJP (N := B) _
        (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
        (convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)).correct x dy i,
      show convNextForwardTCh w =
        (dense w.Wd w.bd ∘
          rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
          globalAvgPoolFlat 768 7 7 ∘
          convNextStageChK 3 w.s4 ∘
          cnxDownChW 7 7 w.d3 ∘
          convNextStageChK 9 w.s3 ∘
          cnxDownChW 14 14 w.d2 ∘
          convNextStageChK 3 w.s2 ∘
          cnxDownChW 28 28 w.d1 ∘
          convNextStageChK 3 w.s1 ∘
          chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
          flatConvStride4 (h := 56) (w := 56) w.sW w.sb)
        from funext (convNextForwardTCh_eq_chain w)]

-- ═════════════════════════════════════════════════
-- § The production capstone — the ImageNet class count, `B` a binder
-- ═════════════════════════════════════════════════

/-- **ConvNeXt-T's batched whole-net backward tie at the ImageNet head.**
    `convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp` at `nC = 1000`, the class count of the
    `convnextin_*` (ConvNeXt-T) artifacts, at a variable batch `B` (64 per device in those
    artifacts). The dims are ConvNeXt-T's (`3×224²`, `[3,3,9,3]` at `96→192→384→768`). The forward
    is the drop-free `convNextForwardTCh` in exact arithmetic: the drop-path (`*drop*`) artifacts
    and ConvNeXt-S/B (`convnextsin_*`, `convnextbin_*`) compute other functions and are not
    covered. The batched peers of the other nets are `r34InputGradB_eq_r34B_full_vjp`,
    `mnv2InputGradB_eq_mobilenetv2B_full_vjp` and `vitTinyInputGradB_eq_vitTiny_vjp`. -/
theorem convnextImagenetInputGradB_eq_vjp (B : Nat) (w : CnxTWeightsCh 1000)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (B * (3 * 224 * 224))) :
    convnextInputGradB B w.Wd (padOdd w.sW)
        (chanLNTensor3Back 96 56 56 w.sε w.sγ) (cnxSavedB0 B w x)
        (rowLNVecFlatBack 1 768 w.hε w.hγ) (cnxSavedB9 B w x)
        (cnxStageChKBack 3 w.s1) (cnxSavedB1 B w x)
        (fun u => cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
          (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ u)) (cnxSavedB2 B w x)
        (cnxStageChKBack 3 w.s2) (cnxSavedB3 B w x)
        (fun u => cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
          (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ u)) (cnxSavedB4 B w x)
        (cnxStageChKBack 9 w.s3) (cnxSavedB5 B w x)
        (fun u => cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
          (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ u)) (cnxSavedB6 B w x)
        (cnxStageChKBack 3 w.s4) (cnxSavedB7 B w x)
      = (batchMapHasVJP (N := B) _
          (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
          (convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)).backward x :=
  convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x

end Proofs
