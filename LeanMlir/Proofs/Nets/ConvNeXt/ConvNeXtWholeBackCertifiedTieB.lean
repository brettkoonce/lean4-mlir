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
batched witness is `batchMap_has_vjp_at` over `HasVJP.toHasVJPAt` at each row — no smooth-point
hypothesis anywhere, only the 23 LayerNorm positivities the per-example tie already carries.

1. `cnxSavedB0 … cnxSavedB10` — the eleven batched stage inputs, saved STAGE BY STAGE
   (`batchMap B stage (cnxSavedB_{k-1} …)`), not as `batchMap B` of the composed per-example
   prefix: the two agree only up to `batchMap_comp`, not `rfl`, and the apex's `vjp_comp_diff_at`
   produces the former.
2. The twelve batched stage witnesses `cnx*B_at`, each `batchMap_has_vjp_at` over the per-example
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
4. `convNextForwardTChB_has_vjp_at` — the twelve-stage apex, eleven `vjp_comp_diff_at`s over the
   batched stage witnesses at the batched saved activations — and
   `convnextInputGradB_eq_convNextForwardTChB_vjp`, the tie: twelve leaf rewrites, then the eleven
   levels peeled by `vjp_comp_diff_at_fst_backward` (a twelve-level `rfl` times out at 10⁶
   heartbeats: it looks for the unfolding through the concrete witnesses first).
5. `convNextForwardTChB_eq_chain` — the shape check: `batchMap B` of the per-example twelve-factor
   chain IS the twelve batched stages, by `batchMap_comp` eleven times — and
   `convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp`, the tie carried to the committed GLOBAL
   witness `batchMap_has_vjp _ (convNextForwardTCh_has_vjp …)` through
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
noncomputable def cnxStemB_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
    (x : Vec (B * (3 * 224 * 224))) : HasVJPAt (StableHLO.batchMap B (cnxSavedA0 w)) x :=
  batchMap_has_vjp_at _ x (fun _ => (cnxV0 w).toHasVJPAt _) (fun _ => (cnxD0 w).differentiableAt)

/-- The batched channel-LayerNorm witness at `v`, at any `c h w`. -/
noncomputable def cnxChanLNB_at (B c h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (v : Vec (B * (c * h * w))) :
    HasVJPAt (StableHLO.batchMap B (chanLNTensor3 c h w ε γ β)) v :=
  batchMap_has_vjp_at _ v
    (fun _ => (chanLNTensor3_has_vjp c h w ε γ β hε).toHasVJPAt _)
    (fun _ => (chanLNTensor3_diff c h w ε γ β hε).differentiableAt)

/-- The batched stem-LayerNorm witness at `v`. -/
noncomputable def cnxStemLNB_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε)
    (v : Vec (B * (96 * 56 * 56))) :
    HasVJPAt (StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)) v :=
  cnxChanLNB_at B 96 56 56 w.sε hsε w.sγ w.sβ v

/-- The batched depth-`k` stage witness at `v` — one definition for all four stages. -/
noncomputable def cnxStageB_at (B : Nat) {c cExp h w kH kW : Nat} (k : Nat)
    (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW) (hε : ∀ i, 0 < (ps i).εn)
    (v : Vec (B * (c * h * w))) :
    HasVJPAt (StableHLO.batchMap B (convNextStageChK k ps)) v :=
  batchMap_has_vjp_at _ v
    (fun _ => (convNextStageChK_has_vjp k ps hε).toHasVJPAt _)
    (fun _ => (convNextStageChK_diff k ps hε).differentiableAt)

/-- The batched downsample witness at `v`, at any resolution and channel pair. -/
noncomputable def cnxDownB_at (B h w : Nat) {cin cout : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) (v : Vec (B * (cin * (2 * h) * (2 * w)))) :
    HasVJPAt (StableHLO.batchMap B (cnxDownChW h w p)) v :=
  batchMap_has_vjp_at _ v (fun _ => (cnxDownChW_has_vjp h w p hε).toHasVJPAt _)
    (fun _ => (cnxDownChW_diff h w p hε).differentiableAt)

/-- The batched downsample-1 witness at `v`, at the chain's dimension spelling (`cnxDn1`). -/
noncomputable def cnxDn1B_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd1 : 0 < w.d1.ε)
    (v : Vec (B * (96 * 56 * 56))) : HasVJPAt (StableHLO.batchMap B (cnxDn1 w)) v :=
  cnxDownB_at B 28 28 w.d1 hd1 v

/-- The batched downsample-2 witness at `v`, at the chain's dimension spelling (`cnxDn2`). -/
noncomputable def cnxDn2B_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε)
    (v : Vec (B * (192 * 28 * 28))) : HasVJPAt (StableHLO.batchMap B (cnxDn2 w)) v :=
  cnxDownB_at B 14 14 w.d2 hd2 v

/-- The batched downsample-3 witness at `v`, at the chain's dimension spelling (`cnxDn3`). -/
noncomputable def cnxDn3B_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε)
    (v : Vec (B * (384 * 14 * 14))) : HasVJPAt (StableHLO.batchMap B (cnxDn3 w)) v :=
  cnxDownB_at B 7 7 w.d3 hd3 v

/-- The batched GAP witness at `v`. -/
noncomputable def cnxGapB_at (B : Nat) (v : Vec (B * (768 * 7 * 7))) :
    HasVJPAt (StableHLO.batchMap B (globalAvgPoolFlat 768 7 7)) v :=
  batchMap_has_vjp_at _ v (fun _ => (globalAvgPoolFlat_has_vjp 768 7 7).toHasVJPAt _)
    (fun _ => (globalAvgPoolFlat_differentiable 768 7 7).differentiableAt)

/-- The batched head-LayerNorm witness at `v`. -/
noncomputable def cnxLNhB_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε)
    (v : Vec (B * 768)) : HasVJPAt (StableHLO.batchMap B (cnxLNh w)) v :=
  batchMap_has_vjp_at _ v (fun _ => (cnxLNhVjp w hhε).toHasVJPAt _)
    (fun _ => (cnxLNhDiff w hhε).differentiableAt)

/-- The batched classifier witness at `v`. -/
noncomputable def cnxDenseB_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (v : Vec (B * 768)) :
    HasVJPAt (StableHLO.batchMap B (dense w.Wd w.bd)) v :=
  batchMap_has_vjp_at _ v (fun _ => (dense_has_vjp w.Wd w.bd).toHasVJPAt _)
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
      = (cnxStemB_at B w x).backward := by
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
      = (cnxChanLNB_at B c h w ε hε γ β v).backward := by
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
      = (cnxStemLNB_at B w hsε v).backward :=
  cnxChanLNBackB_eq_vjp B 96 56 56 w.sε hsε w.sγ w.sβ v

/-- **The batched stage tie** — `cnxStageChKBack_eq_vjp` at one example's row, for every depth. -/
theorem cnxStageBackB_eq_vjp (B : Nat) {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd)
    (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kHd kWd) (hε : ∀ i, 0 < (ps i).εn)
    (v : Vec (B * (c * h * w))) :
    StableHLO.batchMapAux B (cnxStageChKBack k ps) v = (cnxStageB_at B k ps hε v).backward := by
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
      = (cnxDownB_at B h w p hε v).backward := by
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
      = (cnxDn1B_at B w hd1 v).backward :=
  cnxDownBackB_eq_vjp B 28 28 w.d1 hd1 v

/-- **The batched downsample-2 tie** — `cnxDownBackB_eq_vjp` at `14 14`. -/
theorem cnxDn2BackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε)
    (v : Vec (B * (192 * 28 * 28))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
        (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ u)) v
      = (cnxDn2B_at B w hd2 v).backward :=
  cnxDownBackB_eq_vjp B 14 14 w.d2 hd2 v

/-- **The batched downsample-3 tie** — `cnxDownBackB_eq_vjp` at `7 7`. -/
theorem cnxDn3BackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε)
    (v : Vec (B * (384 * 14 * 14))) :
    StableHLO.batchMapAux B (fun u => cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
        (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ u)) v
      = (cnxDn3B_at B w hd3 v).backward :=
  cnxDownBackB_eq_vjp B 7 7 w.d3 hd3 v

/-- **The batched GAP tie is `rfl`**, as the per-example `gapBack_eq_vjp_backward` is. -/
theorem cnxGapBackB_eq_vjp (B : Nat) (v : Vec (B * (768 * 7 * 7))) :
    StableHLO.batchMap B (gapBack 768 7 7) = (cnxGapB_at B v).backward := rfl

/-- **The batched head-LayerNorm tie** — `cnxLNhBack_eq_vjp` at one example's row. -/
theorem cnxLNhBackB_eq_vjp (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε)
    (v : Vec (B * 768)) :
    StableHLO.batchMapAux B (rowLNVecFlatBack 1 768 w.hε w.hγ) v
      = (cnxLNhB_at B w hhε v).backward := by
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
      = (cnxDenseB_at B w v).backward := by
  funext dy idx
  show dense (Mat.transpose w.Wd) (0 : Vec 768)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [dense_transpose_eq_vjp_backward w.Wd w.bd (Mat.unflatten v (finProdFinEquiv.symm idx).1)]
  rfl

-- ═════════════════════════════════════════════════
-- § The batched apex, the tie, and the shape check
-- ═════════════════════════════════════════════════

/-- **The batched whole-net witness**, twelve batched stages composed by `vjp_comp_diff_at`, each
    at the batched saved activation the chain uses (`cnxSavedB0 … cnxSavedB10`). -/
noncomputable def convNextForwardTChB_has_vjp_at (B : Nat) {nC : Nat} (w : CnxTWeightsCh nC)
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
  (vjp_comp_diff_at _ (StableHLO.batchMap B (dense w.Wd w.bd)) x
    (vjp_comp_diff_at _ (StableHLO.batchMap B (cnxLNh w)) x
      (vjp_comp_diff_at _ (StableHLO.batchMap B (globalAvgPoolFlat 768 7 7)) x
        (vjp_comp_diff_at _ (StableHLO.batchMap B (convNextStageChK 3 w.s4)) x
          (vjp_comp_diff_at _ (StableHLO.batchMap B (cnxDn3 w)) x
            (vjp_comp_diff_at _ (StableHLO.batchMap B (convNextStageChK 9 w.s3)) x
              (vjp_comp_diff_at _ (StableHLO.batchMap B (cnxDn2 w)) x
                (vjp_comp_diff_at _ (StableHLO.batchMap B (convNextStageChK 3 w.s2)) x
                  (vjp_comp_diff_at _ (StableHLO.batchMap B (cnxDn1 w)) x
                    (vjp_comp_diff_at _ (StableHLO.batchMap B (convNextStageChK 3 w.s1)) x
                      (vjp_comp_diff_at (StableHLO.batchMap B (cnxSavedA0 w))
                        (StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)) x
                        ⟨cnxStemB_at B w x,
                         batchMap_differentiableAt _ x (fun _ => (cnxD0 w).differentiableAt)⟩
                        ⟨cnxStemLNB_at B w hsε (cnxSavedB0 B w x),
                         batchMap_differentiableAt _ _ (fun _ =>
                           (chanLNTensor3_diff 96 56 56 w.sε w.sγ w.sβ hsε).differentiableAt)⟩)
                      ⟨cnxStageB_at B 3 w.s1 h1 (cnxSavedB1 B w x),
                       batchMap_differentiableAt _ _ (fun _ =>
                         (convNextStageChK_diff 3 w.s1 h1).differentiableAt)⟩)
                    ⟨cnxDn1B_at B w hd1 (cnxSavedB2 B w x),
                     batchMap_differentiableAt _ _ (fun _ => (cnxDn1Diff w hd1).differentiableAt)⟩)
                  ⟨cnxStageB_at B 3 w.s2 h2 (cnxSavedB3 B w x),
                   batchMap_differentiableAt _ _ (fun _ =>
                     (convNextStageChK_diff 3 w.s2 h2).differentiableAt)⟩)
                ⟨cnxDn2B_at B w hd2 (cnxSavedB4 B w x),
                 batchMap_differentiableAt _ _ (fun _ => (cnxDn2Diff w hd2).differentiableAt)⟩)
              ⟨cnxStageB_at B 9 w.s3 h3 (cnxSavedB5 B w x),
               batchMap_differentiableAt _ _ (fun _ =>
                 (convNextStageChK_diff 9 w.s3 h3).differentiableAt)⟩)
            ⟨cnxDn3B_at B w hd3 (cnxSavedB6 B w x),
             batchMap_differentiableAt _ _ (fun _ => (cnxDn3Diff w hd3).differentiableAt)⟩)
          ⟨cnxStageB_at B 3 w.s4 h4 (cnxSavedB7 B w x),
           batchMap_differentiableAt _ _ (fun _ =>
             (convNextStageChK_diff 3 w.s4 h4).differentiableAt)⟩)
        ⟨cnxGapB_at B (cnxSavedB8 B w x),
         batchMap_differentiableAt _ _ (fun _ =>
           (globalAvgPoolFlat_differentiable 768 7 7).differentiableAt)⟩)
      ⟨cnxLNhB_at B w hhε (cnxSavedB9 B w x),
       batchMap_differentiableAt _ _ (fun _ => (cnxLNhDiff w hhε).differentiableAt)⟩)
    ⟨cnxDenseB_at B w (cnxSavedB10 B w x),
     batchMap_differentiableAt _ _ (fun _ => (dense_differentiable w.Wd w.bd).differentiableAt)⟩).fst

/-- One `vjp_comp_diff_at` level's backward, unfolded: the composite runs `g`'s backward, then
    `f`'s. Definitional, stated so that a chain of eleven levels peels by `simp only` rather than
    by a `rfl` that has to find the same unfolding through twelve concrete witnesses. -/
theorem vjp_comp_diff_at_fst_backward {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : PProd (HasVJPAt f x) (DifferentiableAt ℝ f x))
    (hg : PProd (HasVJPAt g (f x)) (DifferentiableAt ℝ g (f x))) (dy : Vec p) :
    (vjp_comp_diff_at f g x hf hg).fst.backward dy = hf.fst.backward (hg.fst.backward dy) := rfl

set_option maxHeartbeats 1000000 in
/-- ⭐⭐ **THE BATCHED TIE.** `convnextInputGradB` with every slot the per-example slot at the
    batched saved activation IS the batched apex's backward. Twelve leaf rewrites, then the eleven
    composition levels peeled by `vjp_comp_diff_at_fst_backward`. -/
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
      = (convNextForwardTChB_has_vjp_at B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x).backward := by
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
  simp only [Function.comp_apply, convNextForwardTChB_has_vjp_at, vjp_comp_diff_at_fst_backward]

/-- **The shape check.** `batchMap B` of the per-example twelve-factor chain — the function
    `convNextForwardTCh_has_vjp` is stated on — IS the twelve batched stages the apex is stated
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

/-- ⭐⭐ **THE APEX, at the committed batched witness.** `convnextInputGradB` IS
    `(batchMap_has_vjp _ (convNextForwardTCh_has_vjp …) …).backward x` — the certified gradient of
    the per-example net lifted whole over `B` examples, for every `B`, every `nC` and every batch
    `x`. Carried from the chain-shaped apex by `HasVJPAt.backward_unique_of_eq` along the shape
    check. Only the 23 LayerNorm positivities. -/
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
      = (batchMap_has_vjp (N := B) _
          (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
          (convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)).backward x := by
  funext dy
  rw [convnextInputGradB_eq_convNextForwardTChB_vjp B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x]
  exact HasVJPAt.backward_unique_of_eq (convNextForwardTChB_eq_chain B w).symm
    (convNextForwardTChB_has_vjp_at B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x)
    ((batchMap_has_vjp (N := B) _
        (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
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
      (batchMap_has_vjp (N := B) _
        (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
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

/-- ⭐⭐ **ConvNeXt-T's BATCHED whole-net backward tie at the ImageNet head — tier T6 at the paper
    net and the shipped index.** `convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp` at
    `nC = 1000`, the class count of every `convnextin_*` / `convnextsin_*` / `convnextbin_*`
    artifact, at a variable batch `B` — 64 or 128 per device in those runs, and neither number
    appears here. The dims are the paper's (`3×224²`, `[3,3,9,3]` at `96→192→384→768`), so this
    is the whole statement at the artifact and not an instance of it. ConvNeXt's entry in the
    batched T6 column beside `r34InputGradB_eq_r34B_full_vjp`,
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
      = (batchMap_has_vjp (N := B) _
          (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
          (convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)).backward x :=
  convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp B w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x

end Proofs
