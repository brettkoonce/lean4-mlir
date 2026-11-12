module

public import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtParamGrad
public import LeanMlir.Proofs.Architectures.EvenKernelConvBack
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # ConvNeXt-T's whole-net input VJP at the stochastic-depth masks

`ConvNeXtFullT` certifies the input VJP of `convNextForwardTCh`, one example, no drop site, and
`ConvNeXtWholeBackCertifiedTieB` lifts it to a batch. The `drop` renders train `cnxNetB gf N ε w
sd`: the batched net with example `n` running block `k` at its own entry of site `k`'s mask
(`exampleSite (cnxSd sd k) n`). This file states the theorem for that function at every choice
of the eighteen sites:

* `cnxNetBHasVJPAt` — the certified input VJP at every input;
* `cnxNetBHasVJPAt_correct` — its backward is the `pdiv`-contracted Jacobian of `cnxNetB` in the
  image;
* `cnxNetB_differentiableAt`.

The masks are held fixed; the derivative is in the image. At `sd := none` the forward is
`batchMap N (convNextForwardTCh (w.toCh ε))` (`cnxNetB_eq_convNextForwardTCh`).

**How.** One `vjpCompDiffAt` per stage over the loss-gradient file's prefixes (`cnxPreS`,
`cnxPreB1`, …): the stem and the three downsamples are `batchMap` lifts of their per-example
VJPs, each block is the INDEXED lift `batchMapIdxHasVJPAt` of `CnxTieBlk.fwdODHasVJP` at the
example's own site, and the head is the `batchMap` lift of `cnxHeadHasVJP`.

**Hypothesis.** Only `0 < ε`, the net's one LayerNorm epsilon: GELU, LayerNorm and the
convolutions are smooth and a site is linear, so there is no smooth-point condition, as for the
drop-free net.
-/

open Proofs Proofs.StableHLO

namespace Proofs.CnxTieGB

open scoped BigOperators
open Proofs.CnxTie (cnxStemFwdO CnxTieWeights)

/-- The stem per example, patchify conv then channel-LayerNorm, has a VJP. -/
noncomputable def cnxStemFwdOHasVJP {c h w : Nat} (ε : ℝ) (hε : 0 < ε) (Wst : Kernel4 c 3 4 4)
    (bst psng psnbt : Vec c) : HasVJP (cnxStemFwdO (h := h) (w := w) ε Wst bst psng psnbt) :=
  vjpComp (flatConvStride4 (h := h) (w := w) Wst bst) (chanLNTensor3 c h w ε psng psnbt)
    (flatConvStride4_differentiable (h := h) (w := w) Wst bst)
    (chanLNTensor3_differentiable c h w ε psng psnbt hε)
    (flatConvStride4HasVJP (h := h) (w := w) Wst bst)
    (chanLNTensor3HasVJP c h w ε psng psnbt hε)

theorem cnxStemFwdO_differentiable {c h w : Nat} (ε : ℝ) (hε : 0 < ε) (Wst : Kernel4 c 3 4 4)
    (bst psng psnbt : Vec c) :
    Differentiable ℝ (cnxStemFwdO (h := h) (w := w) ε Wst bst psng psnbt) :=
  (chanLNTensor3_differentiable c h w ε psng psnbt hε).comp
    (flatConvStride4_differentiable (h := h) (w := w) Wst bst)

/-- The masked chain's VJP and its differentiability together, one `vjpCompDiffAt` per stage:
    the stem, the eighteen blocks each at its site, the three downsamples, the head. -/
noncomputable def cnxNetBChain (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    HasVJPDiffAt (batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc) ∘ cnxPreB18 gf N ε w sd) x :=
  let p0 : HasVJPDiffAt (cnxPreS N ε w) x :=
    ⟨batchMapHasVJPAt (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ) x
        (fun _ => (cnxStemFwdOHasVJP (h := 56) (w := 56) ε hε w.sW w.sb w.sγ w.sβ).toHasVJPAt _)
        (fun _ => (cnxStemFwdO_differentiable (h := 56) (w := 56) ε hε w.sW w.sb w.sγ w.sβ) _),
      (batchMap_differentiable (N := N) (cnxStemFwdO (h := 56) (w := 56) ε w.sW w.sb w.sγ w.sβ)
        (cnxStemFwdO_differentiable (h := 56) (w := 56) ε hε w.sW w.sb w.sγ w.sβ)) x⟩
  let p1 : HasVJPDiffAt (cnxPreB1 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreS N ε w) (batchMapIdx N (fun n => w.b1.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 0) n))) x p0
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b1.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 0) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b1 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b1 _ _)⟩
  let p2 : HasVJPDiffAt (cnxPreB2 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB1 gf N ε w sd) (batchMapIdx N (fun n => w.b2.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 1) n))) x p1
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b2.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 1) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b2 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b2 _ _)⟩
  let p3 : HasVJPDiffAt (cnxPreB3 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB2 gf N ε w sd) (batchMapIdx N (fun n => w.b3.fwdOD gf (h := 56) (w := 56) ε (exampleSite (cnxSd sd 2) n))) x p2
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b3.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 2) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b3 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b3 _ _)⟩
  let p4 : HasVJPDiffAt (cnxPreD0 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB3 gf N ε w sd) (batchMap N (w.d0.fwdO (h := 28) (w := 28) ε)) x p3
      ⟨batchMapHasVJPAt (cnxDownChW 28 28 ⟨ε, w.d0.G, w.d0.T, w.d0.W, w.d0.B⟩) _
          (fun _ => (cnxDownChWHasVJP 28 28 ⟨ε, w.d0.G, w.d0.T, w.d0.W, w.d0.B⟩ hε).toHasVJPAt _)
          (fun _ => (cnxDownChW_differentiable 28 28 ⟨ε, w.d0.G, w.d0.T, w.d0.W, w.d0.B⟩ hε) _),
        (batchMap_differentiable _ (cnxDownChW_differentiable 28 28 ⟨ε, w.d0.G, w.d0.T, w.d0.W, w.d0.B⟩ hε)) _⟩
  let p5 : HasVJPDiffAt (cnxPreB4 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreD0 gf N ε w sd) (batchMapIdx N (fun n => w.b4.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 3) n))) x p4
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b4.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 3) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b4 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b4 _ _)⟩
  let p6 : HasVJPDiffAt (cnxPreB5 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB4 gf N ε w sd) (batchMapIdx N (fun n => w.b5.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 4) n))) x p5
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b5.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 4) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b5 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b5 _ _)⟩
  let p7 : HasVJPDiffAt (cnxPreB6 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB5 gf N ε w sd) (batchMapIdx N (fun n => w.b6.fwdOD gf (h := 28) (w := 28) ε (exampleSite (cnxSd sd 5) n))) x p6
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b6.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 5) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b6 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b6 _ _)⟩
  let p8 : HasVJPDiffAt (cnxPreD1 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB6 gf N ε w sd) (batchMap N (w.d1.fwdO (h := 14) (w := 14) ε)) x p7
      ⟨batchMapHasVJPAt (cnxDownChW 14 14 ⟨ε, w.d1.G, w.d1.T, w.d1.W, w.d1.B⟩) _
          (fun _ => (cnxDownChWHasVJP 14 14 ⟨ε, w.d1.G, w.d1.T, w.d1.W, w.d1.B⟩ hε).toHasVJPAt _)
          (fun _ => (cnxDownChW_differentiable 14 14 ⟨ε, w.d1.G, w.d1.T, w.d1.W, w.d1.B⟩ hε) _),
        (batchMap_differentiable _ (cnxDownChW_differentiable 14 14 ⟨ε, w.d1.G, w.d1.T, w.d1.W, w.d1.B⟩ hε)) _⟩
  let p9 : HasVJPDiffAt (cnxPreB7 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreD1 gf N ε w sd) (batchMapIdx N (fun n => w.b7.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 6) n))) x p8
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b7.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 6) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b7 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b7 _ _)⟩
  let p10 : HasVJPDiffAt (cnxPreB8 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB7 gf N ε w sd) (batchMapIdx N (fun n => w.b8.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 7) n))) x p9
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b8.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 7) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b8 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b8 _ _)⟩
  let p11 : HasVJPDiffAt (cnxPreB9 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB8 gf N ε w sd) (batchMapIdx N (fun n => w.b9.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 8) n))) x p10
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b9.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 8) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b9 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b9 _ _)⟩
  let p12 : HasVJPDiffAt (cnxPreB10 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB9 gf N ε w sd) (batchMapIdx N (fun n => w.b10.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 9) n))) x p11
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b10.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 9) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b10 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b10 _ _)⟩
  let p13 : HasVJPDiffAt (cnxPreB11 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB10 gf N ε w sd) (batchMapIdx N (fun n => w.b11.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 10) n))) x p12
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b11.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 10) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b11 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b11 _ _)⟩
  let p14 : HasVJPDiffAt (cnxPreB12 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB11 gf N ε w sd) (batchMapIdx N (fun n => w.b12.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 11) n))) x p13
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b12.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 11) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b12 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b12 _ _)⟩
  let p15 : HasVJPDiffAt (cnxPreB13 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB12 gf N ε w sd) (batchMapIdx N (fun n => w.b13.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 12) n))) x p14
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b13.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 12) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b13 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b13 _ _)⟩
  let p16 : HasVJPDiffAt (cnxPreB14 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB13 gf N ε w sd) (batchMapIdx N (fun n => w.b14.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 13) n))) x p15
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b14.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 13) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b14 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b14 _ _)⟩
  let p17 : HasVJPDiffAt (cnxPreB15 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB14 gf N ε w sd) (batchMapIdx N (fun n => w.b15.fwdOD gf (h := 14) (w := 14) ε (exampleSite (cnxSd sd 14) n))) x p16
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b15.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 14) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b15 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b15 _ _)⟩
  let p18 : HasVJPDiffAt (cnxPreD2 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB15 gf N ε w sd) (batchMap N (w.d2.fwdO (h := 7) (w := 7) ε)) x p17
      ⟨batchMapHasVJPAt (cnxDownChW 7 7 ⟨ε, w.d2.G, w.d2.T, w.d2.W, w.d2.B⟩) _
          (fun _ => (cnxDownChWHasVJP 7 7 ⟨ε, w.d2.G, w.d2.T, w.d2.W, w.d2.B⟩ hε).toHasVJPAt _)
          (fun _ => (cnxDownChW_differentiable 7 7 ⟨ε, w.d2.G, w.d2.T, w.d2.W, w.d2.B⟩ hε) _),
        (batchMap_differentiable _ (cnxDownChW_differentiable 7 7 ⟨ε, w.d2.G, w.d2.T, w.d2.W, w.d2.B⟩ hε)) _⟩
  let p19 : HasVJPDiffAt (cnxPreB16 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreD2 gf N ε w sd) (batchMapIdx N (fun n => w.b16.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 15) n))) x p18
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b16.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 15) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b16 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b16 _ _)⟩
  let p20 : HasVJPDiffAt (cnxPreB17 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB16 gf N ε w sd) (batchMapIdx N (fun n => w.b17.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 16) n))) x p19
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b17.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 16) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b17 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b17 _ _)⟩
  let p21 : HasVJPDiffAt (cnxPreB18 gf N ε w sd) x :=
    vjpCompDiffAt (cnxPreB17 gf N ε w sd) (batchMapIdx N (fun n => w.b18.fwdOD gf (h := 7) (w := 7) ε (exampleSite (cnxSd sd 17) n))) x p20
      ⟨batchMapIdxHasVJPAt _ _ (fun n => (w.b18.fwdODHasVJP gf ε hε (exampleSite (cnxSd sd 17) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable ε hε w.b18 _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable ε hε w.b18 _ _)⟩
  vjpCompDiffAt (cnxPreB18 gf N ε w sd) (batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc)) x p21
    ⟨batchMapHasVJPAt (dense w.Wfc w.bfc ∘ rowLNVecFlat 1 768 ε w.hG w.hT ∘ globalAvgPoolFlat 768 7 7) _
        (fun _ => (cnxHeadHasVJP 7 7 ε hε w.hG w.hT w.Wfc w.bfc).toHasVJPAt _)
        (fun _ => ((dense_differentiable w.Wfc w.bfc).comp
          ((rowLNVecFlat_differentiable 1 768 ε w.hG w.hT hε).comp
            (globalAvgPoolFlat_differentiable 768 7 7))) _),
      (batchMap_differentiable _ ((dense_differentiable w.Wfc w.bfc).comp
          ((rowLNVecFlat_differentiable 1 768 ε w.hG w.hT hε).comp
            (globalAvgPoolFlat_differentiable 768 7 7)))) _⟩

/-- **ConvNeXt-T with stochastic depth has a certified input VJP**, at every choice of the
    eighteen masks and every input, at batch `N`. `sd := none` is the batched drop-free net. -/
noncomputable def cnxNetBHasVJPAt (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    HasVJPAt (cnxNetB gf N ε w sd) x :=
  (cnxNetBChain gf N ε hε w sd x).fst.congr
    (funext fun v => (cnx_forward_eq_head N ε w sd v).symm)

/-- **Public correctness theorem at the masks**: the backward is the `pdiv`-contracted Jacobian
    of `cnxNetB` in the image. -/
theorem cnxNetBHasVJPAt_correct {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224)))
    (dy : Vec (N * nC)) (i : Fin (N * (3 * 224 * 224))) :
    (cnxNetBHasVJPAt gf N ε hε w sd x).backward dy i =
      ∑ j : Fin (N * nC), pdiv (cnxNetB gf N ε w sd) x i j * dy j :=
  (cnxNetBHasVJPAt gf N ε hε w sd x).correct dy i

/-- The masked forward is differentiable in the image everywhere. -/
theorem cnxNetB_differentiableAt {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : CnxTieWeights nC) (sd : Option (Fin 18 → Vec N)) (x : Vec (N * (3 * 224 * 224))) :
    DifferentiableAt ℝ (cnxNetB gf N ε w sd) x := by
  rw [show cnxNetB gf N ε w sd
      = batchMap N (cnxHeadO 7 7 ε w.hG w.hT w.Wfc w.bfc) ∘ cnxPreB18 gf N ε w sd
    from funext (cnx_forward_eq_head N ε w sd)]
  exact (cnxNetBChain gf N ε hε w sd x).snd

end Proofs.CnxTieGB
