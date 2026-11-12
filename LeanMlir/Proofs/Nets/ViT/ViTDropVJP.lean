module

public import LeanMlir.Proofs.Nets.ViT.ViTParamGrad
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # ViT-Tiny's whole-net input VJP at the stochastic-depth masks

`vitForwardKVHasVJP` certifies the input VJP of the ViT forward for one example with no drop
site, and `ViTWholeBackCertifiedTieB` lifts it to a batch. The `drop` renders train
`vitNetB gf N ε w sd`: the batched net with example `n` running block `k` at its own entries of
that block's two sites, one after attention and one after the MLP
(`exampleSite (vitSdA sd k) n`, `exampleSite (vitSdM sd k) n`). This file states the theorem for
that function at every choice of the twenty-four masks:

* `vitNetBHasVJPAt` — the certified input VJP at every input;
* `vitNetBHasVJPAt_correct` — its backward is the `pdiv`-contracted Jacobian of `vitNetB` in the
  image;
* `vitNetB_differentiableAt`.

The masks are held fixed; the derivative is in the image. At `sd := none` the forward is
`batchMap N` of `vitForwardKV` at the twelve blocks (`vitNetB_eq_vitForwardKV`).

**How.** One `vjpCompDiffAt` per stage over the loss-gradient file's prefixes (`vitPreE`,
`vitPreB1`, …): the patch embedding is the `batchMap` lift of `patchEmbedFlatHasVJP`, each block
the INDEXED lift `batchMapIdxHasVJPAt` of `BlockParamsV.fwdODHasVJP` at the example's own two
sites, and the head the `batchMap` lift of `vitHeadHasVJP`.

**Hypothesis.** Only `0 < ε`, the net's one LayerNorm epsilon: attention, GELU and LayerNorm are
smooth and a site is linear, so there is no smooth-point condition, as for the drop-free net.
-/

open Proofs Proofs.StableHLO

namespace Proofs.ViTTieGB

open scoped BigOperators
open Proofs.ViTTie (ViTTieWeights)

/-- The batched patch embedding's VJP and differentiability at `x`, at variable sizes: stated
    between binders so the net-level chain instantiates it at ViT-Tiny's literals without
    unfolding the embedding there. -/
noncomputable def vitEmbedBDiffAt (B ic H W patchSize N D : Nat)
    (W_conv : Kernel4 D ic patchSize patchSize) (b_conv cls_token : Vec D)
    (pos_embed : Mat (N + 1) D) (x : Vec (B * (ic * H * W))) :
    HasVJPDiffAt (batchMap B
      (patchEmbedFlat ic H W patchSize N D W_conv b_conv cls_token pos_embed)) x :=
  ⟨batchMapHasVJPAt _ x
      (fun _ => (patchEmbedFlatHasVJP ic H W patchSize N D W_conv b_conv cls_token
        pos_embed).toHasVJPAt _)
      (fun _ => (patchEmbedFlat_differentiable ic H W patchSize N D W_conv b_conv cls_token
        pos_embed).differentiableAt),
    batchMap_differentiableAt _ x (fun _ => (patchEmbedFlat_differentiable ic H W patchSize N D
      W_conv b_conv cls_token pos_embed).differentiableAt)⟩

/-- The masked chain's VJP and its differentiability together, one `vjpCompDiffAt` per stage:
    the patch embedding, the twelve blocks each at its two sites, the head. -/
noncomputable def vitNetBChain (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N))
    (x : Vec (N * (3 * 224 * 224))) :
    HasVJPDiffAt (batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls) ∘ vitPreB12 gf N ε w sd) x :=
  let p0 : HasVJPDiffAt (vitPreE N w) x :=
    vitEmbedBDiffAt N 3 224 224 16 196 192 w.Wc w.bc w.cls w.pos x
  let p1 : HasVJPDiffAt (vitPreB1 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreE N w)
      (batchMapIdx N (fun n => w.b1.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 0) n) (exampleSite (vitSdM sd 0) n))) x p0
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b1.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 0) n)
            (exampleSite (vitSdM sd 0) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b1 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b1 _ _ _)⟩
  let p2 : HasVJPDiffAt (vitPreB2 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB1 gf N ε w sd)
      (batchMapIdx N (fun n => w.b2.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 1) n) (exampleSite (vitSdM sd 1) n))) x p1
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b2.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 1) n)
            (exampleSite (vitSdM sd 1) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b2 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b2 _ _ _)⟩
  let p3 : HasVJPDiffAt (vitPreB3 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB2 gf N ε w sd)
      (batchMapIdx N (fun n => w.b3.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 2) n) (exampleSite (vitSdM sd 2) n))) x p2
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b3.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 2) n)
            (exampleSite (vitSdM sd 2) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b3 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b3 _ _ _)⟩
  let p4 : HasVJPDiffAt (vitPreB4 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB3 gf N ε w sd)
      (batchMapIdx N (fun n => w.b4.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 3) n) (exampleSite (vitSdM sd 3) n))) x p3
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b4.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 3) n)
            (exampleSite (vitSdM sd 3) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b4 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b4 _ _ _)⟩
  let p5 : HasVJPDiffAt (vitPreB5 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB4 gf N ε w sd)
      (batchMapIdx N (fun n => w.b5.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 4) n) (exampleSite (vitSdM sd 4) n))) x p4
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b5.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 4) n)
            (exampleSite (vitSdM sd 4) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b5 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b5 _ _ _)⟩
  let p6 : HasVJPDiffAt (vitPreB6 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB5 gf N ε w sd)
      (batchMapIdx N (fun n => w.b6.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 5) n) (exampleSite (vitSdM sd 5) n))) x p5
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b6.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 5) n)
            (exampleSite (vitSdM sd 5) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b6 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b6 _ _ _)⟩
  let p7 : HasVJPDiffAt (vitPreB7 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB6 gf N ε w sd)
      (batchMapIdx N (fun n => w.b7.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 6) n) (exampleSite (vitSdM sd 6) n))) x p6
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b7.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 6) n)
            (exampleSite (vitSdM sd 6) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b7 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b7 _ _ _)⟩
  let p8 : HasVJPDiffAt (vitPreB8 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB7 gf N ε w sd)
      (batchMapIdx N (fun n => w.b8.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 7) n) (exampleSite (vitSdM sd 7) n))) x p7
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b8.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 7) n)
            (exampleSite (vitSdM sd 7) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b8 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b8 _ _ _)⟩
  let p9 : HasVJPDiffAt (vitPreB9 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB8 gf N ε w sd)
      (batchMapIdx N (fun n => w.b9.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 8) n) (exampleSite (vitSdM sd 8) n))) x p8
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b9.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 8) n)
            (exampleSite (vitSdM sd 8) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b9 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b9 _ _ _)⟩
  let p10 : HasVJPDiffAt (vitPreB10 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB9 gf N ε w sd)
      (batchMapIdx N (fun n => w.b10.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 9) n) (exampleSite (vitSdM sd 9) n))) x p9
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b10.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 9) n)
            (exampleSite (vitSdM sd 9) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b10 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b10 _ _ _)⟩
  let p11 : HasVJPDiffAt (vitPreB11 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB10 gf N ε w sd)
      (batchMapIdx N (fun n => w.b11.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 10) n) (exampleSite (vitSdM sd 10) n))) x p10
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b11.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 10) n)
            (exampleSite (vitSdM sd 10) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b11 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b11 _ _ _)⟩
  let p12 : HasVJPDiffAt (vitPreB12 gf N ε w sd) x :=
    vjpCompDiffAt (vitPreB11 gf N ε w sd)
      (batchMapIdx N (fun n => w.b12.fwdOD gf (Np1 := 197) (heads := 3) (d := 64) ε
        (exampleSite (vitSdA sd 11) n) (exampleSite (vitSdM sd 11) n))) x p11
      ⟨batchMapIdxHasVJPAt _ _
          (fun n => (w.b12.fwdODHasVJP gf (Np1 := 197) (heads := 3) (d := 64) ε hε (exampleSite (vitSdA sd 11) n)
            (exampleSite (vitSdM sd 11) n)).toHasVJPAt _)
          (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b12 _ _ _),
        batchMapIdx_differentiableAt _ _ (fun _ => fwdOD_differentiable (Np1 := 197) (heads := 3) (d := 64) ε hε w.b12 _ _ _)⟩
  vjpCompDiffAt (vitPreB12 gf N ε w sd) (batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls)) x p12
    ⟨batchMapHasVJPAt _ _
        (fun _ => (vitHeadHasVJP 196 192 nC ε hε w.γF w.βF w.Wcls w.bcls).toHasVJPAt _)
        (fun _ => ((classifierFlat_differentiable 196 192 nC w.Wcls w.bcls).comp
          (layerNormVec_per_token_flat_differentiable (196 + 1) 192 ε w.γF w.βF hε)) _),
      (batchMap_differentiable _ ((classifierFlat_differentiable 196 192 nC w.Wcls w.bcls).comp
          (layerNormVec_per_token_flat_differentiable (196 + 1) 192 ε w.γF w.βF hε))) _⟩

/-- **ViT-Tiny with stochastic depth has a certified input VJP**, at every choice of the masks
    and every input, at batch `N`. `sd := none` is the batched drop-free net. -/
noncomputable def vitNetBHasVJPAt (gf : GeluForm) (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N))
    (x : Vec (N * (3 * 224 * 224))) :
    HasVJPAt (vitNetB gf N ε w sd) x :=
  (vitNetBChain gf N ε hε w sd x).fst.congr
    (funext fun v => (vit_forward_eq_head N ε w sd v).symm)

/-- **Public correctness theorem at the masks**: the backward is the `pdiv`-contracted Jacobian
    of `vitNetB` in the image. -/
theorem vitNetBHasVJPAt_correct {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N))
    (x : Vec (N * (3 * 224 * 224))) (dy : Vec (N * nC)) (i : Fin (N * (3 * 224 * 224))) :
    (vitNetBHasVJPAt gf N ε hε w sd x).backward dy i =
      ∑ j : Fin (N * nC), pdiv (vitNetB gf N ε w sd) x i j * dy j :=
  (vitNetBHasVJPAt gf N ε hε w sd x).correct dy i

/-- The masked forward is differentiable in the image everywhere. -/
theorem vitNetB_differentiableAt {gf : GeluForm} (N : Nat) {nC : Nat} (ε : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec N × Vec N))
    (x : Vec (N * (3 * 224 * 224))) :
    DifferentiableAt ℝ (vitNetB gf N ε w sd) x := by
  rw [show vitNetB gf N ε w sd
      = batchMap N (vitHeadO ε w.γF w.βF w.Wcls w.bcls) ∘ vitPreB12 gf N ε w sd
    from funext (vit_forward_eq_head N ε w sd)]
  exact (vitNetBChain gf N ε hε w sd x).snd

end Proofs.ViTTieGB
