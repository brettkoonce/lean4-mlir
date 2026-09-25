import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTie
import LeanMlir.Proofs.Foundation.BatchMapVJPAt

/-! # ⭐⭐ `convnextInputGradB` IS the certified whole-net ConvNeXt-T gradient AT A BATCH

`ConvNeXtWholeBackCertifiedTie.lean` closed T6 for ONE image: `convnextInputGrad`, the reverse of
`convNextForwardTCh`, IS the certified gradient at that image. Every shipped ConvNeXt artifact
runs a batch — `convnext_adam_train_step` and the `convnextin_*` / `convnextsin_*` /
`convnextbin_*` families — and its batched T3 tie (`ConvNeXtStepTieGB.lean`) states every
activation as `StableHLO.batchMap B` of the per-example prefix and every cotangent as
`batchMapAux B` of the per-example chain, because LayerNorm is per-example and no ConvNeXt op
couples examples. This file closes T6 at that index: the twelve-stage batched chain
`convnextInputGradB` (`ConvNeXtBackChains.lean`), every slot the per-example slot lifted at the
batched saved activation, IS the certified gradient of `batchMap B convNextForwardTCh` at every
batch `x`, for every `B` and every class count `nC`. ConvNeXt was the last net without a batched
whole-net tie (`planning/archive/renderer_convergence.md` leg 3: its ImageNet artifacts never had
a batched fold); with this the `*InputGradB_eq_*_vjp` family covers all seven nets.

Nothing here is new mathematics, and it is ViT's batched tie (`ViTWholeBackCertifiedTieB.lean`)
one architecture over: ConvNeXt is smooth everywhere, so every stage has a GLOBAL `HasVJP` and its
batched witness is `batchMapHasVJPAt` over `HasVJP.toHasVJPAt` at each row — no smooth-point
hypothesis anywhere, only the 23 LayerNorm positivities the per-example tie already carries.

1. `cnxSavedB0 … cnxSavedB10` — the eleven batched stage inputs, saved STAGE BY STAGE
   (`batchMap B stage (cnxSavedB_{k-1} …)`), not as `batchMap B` of the composed per-example
   prefix: the two agree only up to `batchMap_comp`, not `rfl`, and the apex's `vjpCompDiffAt`
   produces the former.
2. The twelve batched stage witnesses `cnx*B_at`, each `batchMapHasVJPAt` over the per-example
   `HasVJP` at each row — at the dimension spellings the per-example tie normalised (`cnxDn1`,
   `cnxLNh`, `cnxSavedA0`: `ConvNeXtWholeBackCertifiedTie.lean`'s "two spellings of one numeral"
   rule holds one batch index over).
3. The batched leaf ties. GAP's is `rfl` (its per-example tie is); the others are `funext` to one
   example, one rewrite of the per-example leaf tie at that example's row, then `rfl` —
   `batchMapAux`'s slice and the lift's `.backward` row are the same term. ⚠ The channel-LN and
   downsample leaves are proven at VARIABLE dims (`cnxChanLNBackB_eq_vjp`, `cnxDownBackB_eq_vjp`)
   and instantiated by term: at the literal `96 56 56` the same `rfl` recurses past
   `maxRecDepth 100000` on the numerals, the batched form of the per-example tie's "two spellings
   of one numeral" rule.
4. `convNextForwardTChBHasVJPAt` — the twelve-stage apex, eleven `vjpCompDiffAt`s over the
   batched stage witnesses at the batched saved activations — and
   `convnextInputGradB_eq_convNextForwardTChB_vjp`, the tie: twelve leaf rewrites, then the eleven
   levels peeled by `vjpCompDiffAt_fst_backward` (a twelve-level `rfl` times out at 10⁶
   heartbeats: it looks for the unfolding through the concrete witnesses first).
5. `convNextForwardTChB_eq_chain` — the shape check: `batchMap B` of the per-example twelve-factor
   chain IS the twelve batched stages, by `batchMap_comp` eleven times — and
   `convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp`, the tie carried to the committed GLOBAL
   witness `batchMapHasVJP _ (convNextForwardTChHasVJP …)` through
   `HasVJPAt.backward_unique_of_eq`, plus the `∑ pdiv` reading on `convNextForwardTCh` itself.
6. `convnextImagenetInputGradB_eq_vjp` — the same statement at `nC = 1000`, the class count of
   every `convnextin_*` artifact, `B` a binder.
-/

namespace Proofs

open scoped BigOperators

-- The 4×4-stem and 2×2-downsample leaves and the twelve-stage `rfl` recurse deeper than the
-- default allows, exactly as the per-example tie's leaves do (`ConvNeXtWholeBackCertifiedTie.lean`).
set_option maxRecDepth 100000

-- ═════════════════════════════════════════════════
-- § The batched saved activations — stage by stage
-- ═════════════════════════════════════════════════

/-- The batched stem-conv output — the stem LayerNorm's saved input at every example. -/
noncomputable def cnxSavedB0 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (96 * 56 * 56)) :=
  StableHLO.batchMap B (cnxSavedA0 w) x

/-- Stage 1's batched saved input. -/
noncomputable def cnxSavedB1 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (96 * 56 * 56)) :=
  StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ) (cnxSavedB0 B w x)

/-- Downsample 1's batched saved input. -/
noncomputable def cnxSavedB2 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (96 * 56 * 56)) :=
  StableHLO.batchMap B (convNextStageChK 3 w.s1) (cnxSavedB1 B w x)

/-- Stage 2's batched saved input. -/
noncomputable def cnxSavedB3 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (192 * 28 * 28)) :=
  StableHLO.batchMap B (cnxDn1 w) (cnxSavedB2 B w x)

/-- Downsample 2's batched saved input. -/
noncomputable def cnxSavedB4 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (192 * 28 * 28)) :=
  StableHLO.batchMap B (convNextStageChK 3 w.s2) (cnxSavedB3 B w x)

/-- Stage 3's batched saved input. -/
noncomputable def cnxSavedB5 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (384 * 14 * 14)) :=
  StableHLO.batchMap B (cnxDn2 w) (cnxSavedB4 B w x)

/-- Downsample 3's batched saved input. -/
noncomputable def cnxSavedB6 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (384 * 14 * 14)) :=
  StableHLO.batchMap B (convNextStageChK 9 w.s3) (cnxSavedB5 B w x)

/-- Stage 4's batched saved input. -/
noncomputable def cnxSavedB7 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (768 * 7 * 7)) :=
  StableHLO.batchMap B (cnxDn3 w) (cnxSavedB6 B w x)

/-- GAP's batched saved input. -/
noncomputable def cnxSavedB8 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * (768 * 7 * 7)) :=
  StableHLO.batchMap B (convNextStageChK 3 w.s4) (cnxSavedB7 B w x)

/-- The head LayerNorm's batched saved input. -/
noncomputable def cnxSavedB9 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * 768) :=
  StableHLO.batchMap B (globalAvgPoolFlat 768 7 7) (cnxSavedB8 B w x)

/-- The classifier's batched saved input. -/
noncomputable def cnxSavedB10 (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : Vec (B * 768) :=
  StableHLO.batchMap B (cnxLNh w) (cnxSavedB9 B w x)

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
    row. ⛔ `padOdd` is load-bearing here exactly as in the per-example tie: `w.sW` is 4×4. -/
theorem cnxStemBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) :
    StableHLO.batchMap B (flatConvStride4Back (h := 56) (w := 56) (padOdd w.sW))
      = (cnxStemBAt B w x).backward := by
  funext dy idx
  show flatConvStride4Back (h := 56) (w := 56) (padOdd w.sW)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [flatConvStride4Back_padOdd_eq_vjp_backward (h := 56) (w := 56) (by norm_num) (by norm_num)
        w.sW w.sb (Mat.unflatten x (finProdFinEquiv.symm idx).1)]
  rfl

/-- **The batched channel-LayerNorm tie**, at any `c h w` — `chanLNTensor3Back_eq_chanLN_vjp` at
    one example's row. ⚠ Generic on purpose: stated at the literal `96 56 56` the closing `rfl`
    recurses past `maxRecDepth 100000` on the numerals; at variables it is ViT's `vitLNBackB_eq_vjp`
    and closes at once. The stem instance below is a term. -/
theorem cnxChanLNBackB_eq_vjp (B c h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (v : Vec (B * (c * h * w))) :
    StableHLO.batchMapAux B (chanLNTensor3Back c h w ε γ) v
      = (cnxChanLNBAt B c h w ε hε γ β v).backward := by
  funext dy idx
  show chanLNTensor3Back c h w ε γ (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [chanLNTensor3Back_eq_chanLN_vjp (β := β) ε hε γ]
  rfl

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
    StableHLO.batchMapAux B (cnxStageChKBack k ps) v = (cnxStageBAt B k ps hε v).backward := by
  funext dy idx
  show cnxStageChKBack k ps (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [cnxStageChKBack_eq_vjp hkHd hkWd k ps hε]
  rfl

/-- **The batched downsample tie**, at any resolution — `cnxDownChBack_eq_vjp` at one example's
    row. Generic for the same reason as `cnxChanLNBackB_eq_vjp`; the three instances below are
    terms at the chain's dimension spellings (`cnxDn1 … cnxDn3`), which is the per-example tie's
    `cnxDn1Back_eq_vjp … cnxDn3Back_eq_vjp` one batch index over. ⛔ `padOdd` is load-bearing:
    `p.W` is 2×2. -/
theorem cnxDownBackB_eq_vjp (B h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) (v : Vec (B * (cin * (2 * h) * (2 * w)))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := h) (w := w) (padOdd p.W)
        (chanLNTensor3Back cin (2 * h) (2 * w) p.ε p.γ u)) v
      = (cnxDownBAt B h w p hε v).backward := by
  funext dy idx
  show cnxDownBack (h := h) (w := w) (padOdd p.W)
      (chanLNTensor3Back cin (2 * h) (2 * w) p.ε p.γ (Mat.unflatten v (finProdFinEquiv.symm idx).1))
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [cnxDownChBack_eq_vjp (h := h) (w := w) p hε]
  rfl

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
      = (cnxLNhBAt B w hhε v).backward := by
  funext dy idx
  show rowLNVecFlatBack 1 768 w.hε w.hγ (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [cnxLNhBack_eq_vjp w hhε]
  rfl

/-- **The batched classifier tie** — `dense_transpose_eq_vjp_backward` at one example's row (the
    head is linear, so the saved `v` is free). -/
theorem cnxDenseBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (v : Vec (B * 768)) :
    StableHLO.batchMap B (dense (Mat.transpose w.Wd) (0 : Vec 768))
      = (cnxDenseBAt B w v).backward := by
  funext dy idx
  show dense (Mat.transpose w.Wd) (0 : Vec 768)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [dense_transpose_eq_vjp_backward w.Wd w.bd (Mat.unflatten v (finProdFinEquiv.symm idx).1)]
  rfl

-- ═════════════════════════════════════════════════
-- § The batched apex, the tie, and the shape check
-- ═════════════════════════════════════════════════


/-- Truncated apex: 3 level(s) above the stem pair. -/
noncomputable def truncApex (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn)
    (x : Vec (B * (3 * 224 * 224))) :=
  vjpCompDiffAt _ (StableHLO.batchMap B (convNextStageChK 3 w.s2)) x
    (vjpCompDiffAt _ (StableHLO.batchMap B (cnxDn1 w)) x
    (vjpCompDiffAt _ (StableHLO.batchMap B (convNextStageChK 3 w.s1)) x
    (vjpCompDiffAt (StableHLO.batchMap B (cnxSavedA0 w))
    (StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)) x
    ⟨cnxStemBAt B w x,
     batchMap_differentiableAt _ x (fun _ => (cnxSavedA0_differentiable w).differentiableAt)⟩
    ⟨cnxStemLNBAt B w hsε (cnxSavedB0 B w x),
     batchMap_differentiableAt _ _ (fun _ =>
       (chanLNTensor3_differentiable 96 56 56 w.sε w.sγ w.sβ hsε).differentiableAt)⟩)
    (⟨cnxStageBAt B 3 w.s1 h1 (cnxSavedB1 B w x),
     batchMap_differentiableAt _ _ (fun _ => (convNextStageChK_differentiable 3 w.s1 h1).differentiableAt)⟩))
    (⟨cnxDn1BAt B w hd1 (cnxSavedB2 B w x),
     batchMap_differentiableAt _ _ (fun _ => (cnxDn1_differentiable w hd1).differentiableAt)⟩))
    (⟨cnxStageBAt B 3 w.s2 h2 (cnxSavedB3 B w x),
     batchMap_differentiableAt _ _ (fun _ => (convNextStageChK_differentiable 3 w.s2 h2).differentiableAt)⟩)

end Proofs
