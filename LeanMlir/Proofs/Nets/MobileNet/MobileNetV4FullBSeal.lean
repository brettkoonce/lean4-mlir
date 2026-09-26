import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBVJP
import LeanMlir.Proofs.Training.BatchSealKit
import LeanMlir.Proofs.Training.JacobianSeal

/-!
# MobileNetV4-Conv-M's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`MobileNetV4FullBVJP.lean` proves
`mobilenetv4ForwardBFullHasVJPAt`: the whole-net VJP at any `(w, x)` satisfying **eight clause
bundles** — the stem's relu, the fused stage's, one per resolution group and the head's, **38**
relu sites in all (the `#guard` below counts them off the block table). That statement is
pointwise, so it could in principle be vacuous. This file exhibits a `(w, x)` that discharges
every clause with genuinely nonzero weights, shows the forward is not constant there, and seals
the Jacobian nonzero — hence, through
[`Training/JacobianSeal.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Training/JacobianSeal.lean),
the proven backward is not the zero map at that point.

The net is timm's `mobilenetv4_conv_medium`: relu throughout,
BN-only pre-DWs, the stride on each downsample's post-DW, and a head that pools before
`conv_head`, whose BatchNorm therefore normalises the pooled features over the batch alone.

## All 38 clauses are weight-only — and generic in the table row

Every kink in this net is a relu sitting directly on a `bnBatchLA` output (`Mnv4StemSmoothAtB`,
`cbReluLayer.ok`, `cbReluStridedLayer.ok`, `mnv4DWReluLayer.ok`, `mnv4DWReluStridedLayer.ok`);
`projLayer.ok`, `mnv4DWBnLayer.ok`, `castLayer.ok` and `CertLayer.id'.ok` are `True`.
`bnBatchLA_pos` bounds a BatchNorm output above `β − |γ|√(N·h·w) > 0` at EVERY input, so with
`γ = 1`, `β = 160`, `ε = 1` the whole bundle is discharged without the activation ever being read
— including `conv_head`'s, whose BatchNorm sees `N·1·1 = 2` cells.

## The carrier

The input is per-(example, channel) constant (`sealX t = t • rayV`), so it is an `EDiff` from the
start: example 0 is example 1 plus `t` on channel 0. Every stage on the carrier's path is a
centre-tap conv or depthwise then a BatchNorm, and inside the margin window every relu is the
identity, so each stage multiplies the carrier by one positive `istd`. Seventeen BatchNorms sit
on the path: the stem's, the fused stage's two, four in each of rows 1, 3 and 11 (the only rows
that change channels, hence the only ones without a skip) and the head's two. The pool in front
of `conv_head` keeps the carrier (a per-channel shift survives an average), and the two relabels
are index bijections. The readout along the ray is therefore `t · Rr t`, with `Rr` the product of
the seventeen `istd`s: continuous and positive, and no BatchNorm variance derivative is taken.

Note: every collapse, every clause bundle and every block lemma here is proved at variable shapes
and instantiated at the witness's numerals afterwards, never proved at them.
-/

namespace Proofs
namespace Mnv4FullBSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs BatchSeal StableHLO Proofs.BatchSeal

-- ⭐ **The clause count, read off `mnv4Blocks` rather than asserted.** One relu per expand conv
-- and per present post-DW in each of the 21 UIB rows, plus the stem's, the fused stage's and the
-- head's two; the pre-DW (timm's `dw_start`) and the project convs are BN only.
-- `sealUib_ok` / `sealUibStrided_ok` discharge all of them in two lemmas.
#guard 1 + 1 + (StableHLO.mnv4Blocks.map (fun s =>
    1 + (if s.postDWk = 0 then 0 else 1))).sum + 2 = 38

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural weights
--   ⭐ One `γ = 1`, one `ε = 1`, and `β = 160` at every BatchNorm except the projections'
--   (`β = 0`, no relu follows them), so a zeroed body is the constant `0` and its block is the
--   exact identity. ⭐⭐ ONE record, `sealP`, with the four kernels as arguments: the carrier's
--   rows pass centre taps, the eighteen skipped rows pass zeros, and every lemma below is proved
--   once over `sealP`.
-- ════════════════════════════════════════════════════════════════
noncomputable def sealP (s : UibSpec)
    (Wq : DepthwiseKernel s.ic s.preDWk s.preDWk)
    (We : Kernel4 (s.ic * s.expand) s.ic 1 1)
    (Wd : DepthwiseKernel (s.ic * s.expand) s.postDWk s.postDWk)
    (Wz : Kernel4 s.oc (s.ic * s.expand) 1 1) : UibParams s where
  Wq := Wq
  bq := kv s.ic 0
  eq_ := 1
  hq := one_pos
  gq := kv s.ic 1
  bq2 := kv s.ic 160
  We := We
  be := kv (s.ic * s.expand) 0
  ee := 1
  he := one_pos
  ge := kv (s.ic * s.expand) 1
  be2 := kv (s.ic * s.expand) 160
  Wd := Wd
  bd := kv (s.ic * s.expand) 0
  ed := 1
  hd := one_pos
  gd := kv (s.ic * s.expand) 1
  bd2 := kv (s.ic * s.expand) 160
  Wz := Wz
  bz := kv s.oc 0
  ez := 1
  hz := one_pos
  gz := kv s.oc 1
  bz2 := kv s.oc 0

/-- a carrier row: every kernel a centre tap. -/
noncomputable def sealCT (s : UibSpec) : UibParams s :=
  sealP s (ctDW s.ic s.preDWk s.preDWk 1) (ctK (s.ic * s.expand) s.ic 1 1 1)
    (ctDW (s.ic * s.expand) s.postDWk s.postDWk 1) (ctK s.oc (s.ic * s.expand) 1 1 1)

/-- a skipped row: every kernel zero, so the body is the constant `0`. -/
noncomputable def sealZ (s : UibSpec) : UibParams s :=
  sealP s (dzk s.ic s.preDWk s.preDWk) (zk (s.ic * s.expand) s.ic 1 1)
    (dzk (s.ic * s.expand) s.postDWk s.postDWk) (zk s.oc (s.ic * s.expand) 1 1)

noncomputable def sealW (nCls : Nat) : Mnv4BWeights nCls where
  sW := ctK 32 3 3 3 1
  sb := kv 32 0
  sE := 1
  hsE := one_pos
  sg := kv 32 1
  sbt := kv 32 160
  f0cW := ctK 128 32 3 3 1
  f0cb := kv 128 0
  f0cE := 1
  hf0cE := one_pos
  f0cg := kv 128 1
  f0cbt := kv 128 160
  f0pW := ctK 48 128 1 1 1
  f0pb := kv 48 0
  f0pE := 1
  hf0pE := one_pos
  f0pg := kv 48 1
  f0pbt := kv 48 0
  b1 := sealCT mnv4Row1
  b2 := sealZ mnv4Row2
  b3 := sealCT mnv4Row3
  b4 := sealZ mnv4Row4
  b5 := sealZ mnv4Row5
  b6 := sealZ mnv4Row6
  b7 := sealZ mnv4Row7
  b8 := sealZ mnv4Row8
  b9 := sealZ mnv4Row9
  b10 := sealZ mnv4Row10
  b11 := sealCT mnv4Row11
  b12 := sealZ mnv4Row12
  b13 := sealZ mnv4Row13
  b14 := sealZ mnv4Row14
  b15 := sealZ mnv4Row15
  b16 := sealZ mnv4Row16
  b17 := sealZ mnv4Row17
  b18 := sealZ mnv4Row18
  b19 := sealZ mnv4Row19
  b20 := sealZ mnv4Row20
  b21 := sealZ mnv4Row21
  h1W := ctK 960 256 1 1 1
  h1b := kv 960 0
  h1E := 1
  hh1E := one_pos
  h1g := kv 960 1
  h1bt := kv 960 160
  hW := ctK 1280 960 1 1 1
  hb := kv 1280 0
  hE := 1
  hhE := one_pos
  hg := kv 1280 1
  hbt := kv 1280 160
  Wd := fun i j => if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0
  bd := kv nCls 0

-- ════════════════════════════════════════════════════════════════
-- § 2. The margin — weight-only, at every input
--   ⭐⭐ `bnBatchLA_pos` puts a BatchNorm output above `β − |γ|√(N·h·w) > 0` at EVERY input, so
--   every relu in this net is off its kink without the activation ever being read.
-- ════════════════════════════════════════════════════════════════
abbrev Mg (N h w : Nat) : Prop := ((N * (h * w) : ℕ) : ℝ) < 25600

theorem bpos (N oc h w : Nat) (hm : Mg N h w) (v : Vec (N * (oc * h * w))) :
    ∀ k, 0 < StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160) v k :=
  bnBatchLA_pos 1 one_pos (kv oc 1) (kv oc 160) 1 160 (fun _ => rfl) (fun _ => rfl)
    (margin160 _ hm) v

theorem bne (N oc h w : Nat) (hm : Mg N h w) (v : Vec (N * (oc * h * w))) :
    ∀ k, StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160) v k ≠ 0 :=
  fun k => (bpos N oc h w hm v k).ne'

-- ════════════════════════════════════════════════════════════════
-- § 3. Stage collapses, at VARIABLE shapes
-- ════════════════════════════════════════════════════════════════
theorem cbReluB_eq {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * h * w))) :
    StableHLO.cbReluB N (h := h) (w := w) W b 1 (kv oc 1) (kv oc 160) x
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConv W b) x) :=
  relu_id_of_pos (bpos N oc h w hm _)

theorem cbReluStridedB_eq {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    StableHLO.cbReluStridedB N (h := h) (w := w) W b 1 (kv oc 1) (kv oc 160) x
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConvStride2 W b) x) :=
  relu_id_of_pos (bpos N oc h w hm _)

theorem dwbReluBstrided_eq {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (hm : Mg N h w) (x : Vec (N * (c * (2 * h) * (2 * w)))) :
    StableHLO.dwbReluBstrided N (h := h) (w := w) W b 1 (kv c 1) (kv c 160) x
      = StableHLO.bnBatchLA N c h w 1 (kv c 1) (kv c 160)
          (StableHLO.batchMap N (depthwiseStride2Flat W b) x) :=
  relu_id_of_pos (bpos N c h w hm _)

theorem mnv4StemB_eq {N h w ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    mnv4StemB N h w Ws bs 1 (kv oc 1) (kv oc 160) x
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConvStride2 Ws bs) x) :=
  relu_id_of_pos (bpos N oc h w hm _)

-- ════════════════════════════════════════════════════════════════
-- § 4. The block collapses — one lemma per FORM, generic in the table row
-- ════════════════════════════════════════════════════════════════
/-- **A zeroed UIB body is the constant `0`, whatever its slots are.** The project conv's kernel
    is zero, so `projB_zero_const` closes the block without any stage inside it being analysed. -/
theorem sealZBody_eq (N : Nat) (s : UibSpec) (hn : 0 < N * (s.h * s.h))
    (v : Vec (N * (s.ic * s.h * s.h))) :
    (mnv4BodyOfRow N s (sealZ s)).fwd v = fun _ => (0 : ℝ) := by
  simp only [mnv4BodyOfRow, mnv4UibBody, CertLayer.comp_fwd_apply]
  exact projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 0 (fun _ => rfl) _

/-- and so a skipped row is the exact IDENTITY, not ResNet's `a ↦ a + 1`. -/
theorem resid_id {n : Nat} (L : CertLayer n n) (v : Vec n) (hL : L.fwd v = fun _ => (0 : ℝ)) :
    (CertLayer.residual L).fwd v = v := by
  funext k
  show L.fwd v k + v k = v k
  rw [hL]
  ring

/-- **A carrier row (all three are strided ExtraDW) collapses to four BatchNorms**: the BN-only
    pre-DW and the expand at `2h`, the strided post-DW, the project. -/
theorem sealCTStrided_eq (N : Nat) (s : UibSpec) (hq : s.preDWk ≠ 0)
    (hmd : Mg N s.h s.h) (hme : Mg N (2 * s.h) (2 * s.h))
    (v : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    (mnv4StridedBodyOfRow N s (sealCT s)).fwd v
      = StableHLO.bnBatchLA N s.oc s.h s.h 1 (kv s.oc 1) (kv s.oc 0)
          (StableHLO.batchMap N
            (flatConv (h := s.h) (w := s.h) (ctK s.oc (s.ic * s.expand) 1 1 1) (kv s.oc 0))
          (StableHLO.bnBatchLA N (s.ic * s.expand) s.h s.h 1 (kv (s.ic * s.expand) 1)
              (kv (s.ic * s.expand) 160)
          (StableHLO.batchMap N
            (depthwiseStride2Flat (h := s.h) (w := s.h)
              (ctDW (s.ic * s.expand) s.postDWk s.postDWk 1) (kv (s.ic * s.expand) 0))
          (StableHLO.bnBatchLA N (s.ic * s.expand) (2 * s.h) (2 * s.h) 1
              (kv (s.ic * s.expand) 1) (kv (s.ic * s.expand) 160)
          (StableHLO.batchMap N
            (flatConv (h := 2 * s.h) (w := 2 * s.h) (ctK (s.ic * s.expand) s.ic 1 1 1)
              (kv (s.ic * s.expand) 0))
          (StableHLO.bnBatchLA N s.ic (2 * s.h) (2 * s.h) 1 (kv s.ic 1) (kv s.ic 160)
            (StableHLO.batchMap N
              (depthwiseFlat (h := 2 * s.h) (w := 2 * s.h) (ctDW s.ic s.preDWk s.preDWk 1)
                (kv s.ic 0)) v))))))) := by
  simp only [mnv4StridedBodyOfRow, mnv4UibStridedBody, CertLayer.comp_fwd_apply,
    mnv4PreDWSlot, hq, ↓reduceIte]
  show projB N (h := s.h) (w := s.h) (ctK s.oc (s.ic * s.expand) 1 1 1) (kv s.oc 0) 1
        (kv s.oc 1) (kv s.oc 0)
      (StableHLO.dwbReluBstrided N (h := s.h) (w := s.h)
          (ctDW (s.ic * s.expand) s.postDWk s.postDWk 1) (kv (s.ic * s.expand) 0) 1
          (kv (s.ic * s.expand) 1) (kv (s.ic * s.expand) 160)
        (StableHLO.cbReluB N (h := 2 * s.h) (w := 2 * s.h) (ctK (s.ic * s.expand) s.ic 1 1 1)
            (kv (s.ic * s.expand) 0) 1 (kv (s.ic * s.expand) 1) (kv (s.ic * s.expand) 160)
          (dwbB N (h := 2 * s.h) (w := 2 * s.h)
            (ctDW s.ic s.preDWk s.preDWk 1) (kv s.ic 0) 1 (kv s.ic 1) (kv s.ic 160) v))) = _
  rw [cbReluB_eq _ _ hme, dwbReluBstrided_eq _ _ hmd]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 5. The clause bundle — ⭐⭐ weight-only, and proved GENERICALLY IN THE ROW
-- ════════════════════════════════════════════════════════════════
/-- **Every clause of a stride-1 UIB body, at every input.** The pre-DW slot contributes `True`
    either way (`id'` or the BN-only depthwise); the expand's and the post-DW's relus sit on
    BatchNorm outputs, so the discharge never looks at the activation. -/
theorem sealUib_ok (N : Nat) (s : UibSpec) (hm : Mg N s.h s.h)
    (Wq : DepthwiseKernel s.ic s.preDWk s.preDWk)
    (We : Kernel4 (s.ic * s.expand) s.ic 1 1)
    (Wd : DepthwiseKernel (s.ic * s.expand) s.postDWk s.postDWk)
    (Wz : Kernel4 s.oc (s.ic * s.expand) 1 1) (x : Vec (N * (s.ic * s.h * s.h))) :
    (mnv4BodyOfRow N s (sealP s Wq We Wd Wz)).ok x := by
  refine ⟨?_, ?_, ?_, trivial⟩
  · show (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk _ _ _ _ _ _).ok _
    unfold mnv4PreDWSlot
    by_cases hk : s.preDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; trivial
  · exact bne N (s.ic * s.expand) s.h s.h hm _
  · show (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk _ _ _ _ _ _).ok _
    unfold mnv4PostDWSlot
    by_cases hk : s.postDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; exact bne N (s.ic * s.expand) s.h s.h hm _

/-- the strided peer: the pre-DW slot and the expand at `2h`, the post-DW carrying the stride. -/
theorem sealUibStrided_ok (N : Nat) (s : UibSpec) (hm : Mg N s.h s.h)
    (hme : Mg N (2 * s.h) (2 * s.h))
    (Wq : DepthwiseKernel s.ic s.preDWk s.preDWk)
    (We : Kernel4 (s.ic * s.expand) s.ic 1 1)
    (Wd : DepthwiseKernel (s.ic * s.expand) s.postDWk s.postDWk)
    (Wz : Kernel4 s.oc (s.ic * s.expand) 1 1)
    (x : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    (mnv4StridedBodyOfRow N s (sealP s Wq We Wd Wz)).ok x := by
  refine ⟨?_, ?_, bne N (s.ic * s.expand) s.h s.h hm _, trivial⟩
  · show (mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk _ _ _ _ _ _).ok _
    unfold mnv4PreDWSlot
    by_cases hk : s.preDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; trivial
  · exact bne N (s.ic * s.expand) (2 * s.h) (2 * s.h) hme _

-- ════════════════════════════════════════════════════════════════
-- § 6. The ray — a grid-constant base
-- ════════════════════════════════════════════════════════════════
noncomputable def sealV : Vec (2 * (3 * (2 * 112) * (2 * 112))) := rayV (2 * 112) (2 * 112)

noncomputable def sealX (t : ℝ) : Vec (2 * (3 * (2 * 112) * (2 * 112))) := t • sealV

theorem sealX_zero_add (t : ℝ) : sealX 0 + t • sealV = sealX t := by
  rw [sealX, sealX, zero_smul, zero_add]

/-- the witness input is per-(example, channel) CONSTANT: `t` on example 0's channel 0, zero
    everywhere else. -/
theorem bUnif_sealX (t : ℝ) :
    BUnif (h := 2 * 112) (w := 2 * 112)
      (fun n ci => if n.val = 0 ∧ ci.val = 0 then t else 0) (sealX t) := by
  intro n ci i j
  show bcell (t • rayV (2 * 112) (2 * 112)) n ci i j = _
  rw [bcell_smul, rayV, bcell_bfrom]
  by_cases h : n.val = 0 ∧ ci.val = 0 <;> simp [h]

/-- so it is a carrier from the start: `t` on channel 0. -/
theorem ed_sealX (t : ℝ) :
    EDiff (h := 2 * 112) (w := 2 * 112) (fun ci : Fin 3 => if ci.val = 0 then t else 0) (sealX t) :=
  eDiff_of_bUnif _ _ _ (bUnif_sealX t) (fun ci => by
    by_cases h : ci.val = 0 <;> simp [h])

-- ════════════════════════════════════════════════════════════════
-- § 7. The eight clause bundles, and the whole-net VJP at the witness
-- ════════════════════════════════════════════════════════════════
theorem sc_stem (nCls : Nat) (t : ℝ) :
    Mnv4StemSmoothAtB 2 112 112 (sealW nCls).sW (sealW nCls).sb (sealW nCls).sE
      (sealW nCls).sg (sealW nCls).sbt (sealX t) :=
  bne 2 32 112 112 (by norm_num) _

theorem sc_fused (nCls : Nat) (t : ℝ) :
    (mnv4FusedStack 2 (sealW nCls)).ok (mnv4Pre0 2 (sealW nCls) (sealX t)) :=
  ⟨bne 2 128 56 56 (by norm_num) _, trivial⟩

theorem sc28 (nCls : Nat) (t : ℝ) :
    (mnv4Res28Layer 2 (sealW nCls)).ok (mnv4Pre1 2 (sealW nCls) (sealX t)) :=
  ⟨sealUibStrided_ok 2 mnv4Row1 (by norm_num) (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row2 (by norm_num) _ _ _ _ _⟩

theorem sc14a (nCls : Nat) (t : ℝ) :
    (mnv4Res14aLayer 2 (sealW nCls)).ok (mnv4Pre2 2 (sealW nCls) (sealX t)) :=
  ⟨sealUibStrided_ok 2 mnv4Row3 (by norm_num) (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row4 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row5 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row6 (by norm_num) _ _ _ _ _⟩

theorem sc14b (nCls : Nat) (t : ℝ) :
    (mnv4Res14bLayer 2 (sealW nCls)).ok (mnv4Pre3 2 (sealW nCls) (sealX t)) :=
  ⟨sealUib_ok 2 mnv4Row7 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row8 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row9 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row10 (by norm_num) _ _ _ _ _⟩

theorem sc7a (nCls : Nat) (t : ℝ) :
    (mnv4Res7aLayer 2 (sealW nCls)).ok (mnv4Pre4 2 (sealW nCls) (sealX t)) :=
  ⟨sealUibStrided_ok 2 mnv4Row11 (by norm_num) (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row12 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row13 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row14 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row15 (by norm_num) _ _ _ _ _⟩

theorem sc7b (nCls : Nat) (t : ℝ) :
    (mnv4Res7bLayer 2 (sealW nCls)).ok (mnv4Pre5 2 (sealW nCls) (sealX t)) :=
  ⟨sealUib_ok 2 mnv4Row16 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row17 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row18 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row19 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row20 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row21 (by norm_num) _ _ _ _ _⟩

/-- the head's two relus — `cn_960`'s at 7×7 and `conv_head`'s on the pooled `1×1` (its BatchNorm
    over the two examples alone); GAP, the relabels and dense are `True`. -/
theorem sc_head (nCls : Nat) (t : ℝ) :
    (mnv4HeadStack 2 (sealW nCls)).ok (mnv4Pre6 2 (sealW nCls) (sealX t)) :=
  ⟨bne 2 960 7 7 (by norm_num) _, trivial, trivial, bne 2 1280 1 1 (by norm_num) _, trivial,
    trivial⟩

/-- The eight group conditions above, as the apex's one smoothness hypothesis. -/
theorem seal_smooth (nCls : Nat) (t : ℝ) : Mnv4SmoothAt 2 (sealW nCls) (sealX t) :=
  ⟨sc_stem nCls t, sc_fused nCls t, sc28 nCls t, sc14a nCls t, sc14b nCls t, sc7a nCls t,
    sc7b nCls t, sc_head nCls t⟩

/-- **The whole-net VJP at the witness** — all eight bundles discharged. -/
noncomputable def sealVJP (nCls : Nat) (t : ℝ) :
    HasVJPAt (mobilenetv4ForwardBFull 2 (sealW nCls)) (sealX t) :=
  mobilenetv4ForwardBFullHasVJPAt 2 (sealW nCls) (sealX t) (seal_smooth nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 8. The pool and the relabels carry the carrier
-- ════════════════════════════════════════════════════════════════
/-- Reading a `[N, c]` vector relabelled to `[N, c, 1, 1]` at `(n, ci, 0, 0)` reads `(n, ci)`. -/
theorem bcell_to11 {N c : Nat} (u : Vec (N * c)) (n : Fin N) (ci : Fin c) (i j : Fin 1) :
    bcell (fun k => u (Fin.cast (mnv4_pool11 N c).symm k)) n ci i j = Mat.unflatten u n ci := by
  simp only [bcell, Mat.unflatten, Tensor3.unflatten]
  congr 1
  apply Fin.ext
  simp only [Fin.val_cast, finProdFinEquiv_apply_val, Fin.val_eq_zero i, Fin.val_eq_zero j,
    Nat.mul_one, Nat.one_mul, Nat.zero_add]

/-- and back: the `[N, c]` relabel of a `[N, c, 1, 1]` vector reads its `(n, ci, 0, 0)` cell. -/
theorem unflatten_from11 {N c : Nat} (v : Vec (N * (c * 1 * 1))) (n : Fin N) (ci : Fin c) :
    Mat.unflatten (fun k => v (Fin.cast (mnv4_pool11 N c) k)) n ci = bcell v n ci 0 0 := by
  simp only [bcell, Mat.unflatten, Tensor3.unflatten]
  congr 1
  apply Fin.ext
  simp only [Fin.val_cast, finProdFinEquiv_apply_val, Fin.val_zero, Nat.mul_one, Nat.one_mul,
    Nat.zero_add]

/-- The pool is continuous — for `Rr_continuous`'s `fun_prop`, which reads it through `Pl`. -/
@[fun_prop]
theorem globalAvgPoolFlat_continuous (c h w : Nat) : Continuous (globalAvgPoolFlat c h w) :=
  (globalAvgPoolFlat_differentiable c h w).continuous

/-- **GAP, then the relabel to `[N, c, 1, 1]`, keeps the carrier**: a per-channel shift of every
    cell shifts the average by the same amount. -/
theorem eDiff_gapTo11 {c h w : Nat} (hh : 0 < h) (hw : 0 < w) (δ : Fin c → ℝ)
    (v : Vec (2 * (c * h * w))) (hv : EDiff δ v) :
    EDiff (h := 1) (w := 1) δ
      (fun k => StableHLO.batchMap 2 (globalAvgPoolFlat c h w) v
        (Fin.cast (mnv4_pool11 2 c).symm k)) := by
  intro ci i j
  rw [bcell_to11, bcell_to11, row_batchMap, row_batchMap]
  exact globalAvgPool_shift hh hw _ _ (δ ci) ci (fun i j => hv ci i j)

/-- **The relabelled classifier reads the carrier off channel `c₀`.** -/
theorem cls_diff {c nCls : Nat} (c₀ : Fin c) (hc₀ : c₀.val = 0) (j : Fin nCls)
    (Wd : Mat c nCls) (bd : Vec nCls)
    (hWd : ∀ ci, Wd ci j = if ci.val = 0 then (1 : ℝ) else 0) (hbd : bd j = 0)
    (v : Vec (2 * (c * 1 * 1))) (δ : Fin c → ℝ) (hv : EDiff (h := 1) (w := 1) δ v) :
    StableHLO.batchMap 2 (dense Wd bd) (fun k => v (Fin.cast (mnv4_pool11 2 c) k))
        (finProdFinEquiv ((0 : Fin 2), j))
      - StableHLO.batchMap 2 (dense Wd bd) (fun k => v (Fin.cast (mnv4_pool11 2 c) k))
        (finProdFinEquiv ((1 : Fin 2), j))
      = δ c₀ := by
  have e : ∀ n : Fin 2, StableHLO.batchMap 2 (dense Wd bd)
      (fun k => v (Fin.cast (mnv4_pool11 2 c) k)) (finProdFinEquiv (n, j))
      = dense Wd bd (fun ci => bcell v n ci 0 0) j := by
    intro n
    have hr := congrFun (row_batchMap (dense Wd bd) (fun k => v (Fin.cast (mnv4_pool11 2 c) k)) n) j
    rw [show Mat.unflatten (StableHLO.batchMap 2 (dense Wd bd)
          (fun k => v (Fin.cast (mnv4_pool11 2 c) k))) n j
        = StableHLO.batchMap 2 (dense Wd bd) (fun k => v (Fin.cast (mnv4_pool11 2 c) k))
          (finProdFinEquiv (n, j)) from rfl] at hr
    rw [hr]
    congr 1
    funext ci
    exact unflatten_from11 v n ci
  rw [e 0, e 1]
  simp only [dense, hWd, hbd, add_zero]
  rw [← Finset.sum_sub_distrib]
  rw [Finset.sum_congr rfl (fun ci _ => show
      bcell v 0 ci 0 0 * (if ci.val = 0 then (1 : ℝ) else 0)
        - bcell v 1 ci 0 0 * (if ci.val = 0 then (1 : ℝ) else 0)
      = δ ci * (if ci.val = 0 then (1 : ℝ) else 0) from by rw [hv ci 0 0]; ring)]
  refine (Finset.sum_eq_single_of_mem c₀ (Finset.mem_univ _) ?_).trans ?_
  · intro ci _ hci
    have hc : ci.val ≠ 0 := fun hz => hci (Fin.ext (hz.trans hc₀.symm))
    simp [hc]
  · simp [hc₀]

-- ════════════════════════════════════════════════════════════════
-- § 9. The seventeen activations on the carrier's path
--   ⚠ Each is written in terms of the PREVIOUS one, never in terms of `mnv4Pre_k`: the continuity
--   proof then unfolds straight back to `sealX`, and `pc0`–`pc6` say which prefix each one IS.
-- ════════════════════════════════════════════════════════════════

noncomputable def Zs (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.batchMap 2 (flatConvStride2 (h := 112) (w := 112) (ctK 32 3 3 3 1) (kv 32 0))
    (sealX t)

noncomputable def As (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.bnBatchLA 2 32 112 112 1 (kv 32 1) (kv 32 160) (Zs t)

noncomputable def Zf (t : ℝ) : Vec (2 * (128 * 56 * 56)) :=
  StableHLO.batchMap 2 (flatConvStride2 (h := 56) (w := 56) (ctK 128 32 3 3 1) (kv 128 0))
    (As t)

noncomputable def Af (t : ℝ) : Vec (2 * (128 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 128 56 56 1 (kv 128 1) (kv 128 160) (Zf t)

noncomputable def Z1p (t : ℝ) : Vec (2 * (48 * 56 * 56)) :=
  StableHLO.batchMap 2 (flatConv (h := 56) (w := 56) (ctK 48 128 1 1 1) (kv 48 0)) (Af t)

noncomputable def A1p (t : ℝ) : Vec (2 * (48 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 48 56 56 1 (kv 48 1) (kv 48 0) (Z1p t)

noncomputable def Zaq (t : ℝ) : Vec (2 * (48 * 56 * 56)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 56) (w := 56) (ctDW 48 3 3 1) (kv 48 0)) (A1p t)

noncomputable def Aaq (t : ℝ) : Vec (2 * (48 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 48 56 56 1 (kv 48 1) (kv 48 160) (Zaq t)

noncomputable def Zae (t : ℝ) : Vec (2 * (192 * 56 * 56)) :=
  StableHLO.batchMap 2 (flatConv (h := 56) (w := 56) (ctK 192 48 1 1 1) (kv 192 0)) (Aaq t)

noncomputable def Aae (t : ℝ) : Vec (2 * (192 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 192 56 56 1 (kv 192 1) (kv 192 160) (Zae t)

noncomputable def Zad (t : ℝ) : Vec (2 * (192 * 28 * 28)) :=
  StableHLO.batchMap 2 (depthwiseStride2Flat (h := 28) (w := 28) (ctDW 192 5 5 1) (kv 192 0))
    (Aae t)

noncomputable def Aad (t : ℝ) : Vec (2 * (192 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 192 28 28 1 (kv 192 1) (kv 192 160) (Zad t)

noncomputable def Zaz (t : ℝ) : Vec (2 * (80 * 28 * 28)) :=
  StableHLO.batchMap 2 (flatConv (h := 28) (w := 28) (ctK 80 192 1 1 1) (kv 80 0)) (Aad t)

noncomputable def Aaz (t : ℝ) : Vec (2 * (80 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 80 28 28 1 (kv 80 1) (kv 80 0) (Zaz t)

noncomputable def Zbq (t : ℝ) : Vec (2 * (80 * 28 * 28)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 28) (w := 28) (ctDW 80 3 3 1) (kv 80 0)) (Aaz t)

noncomputable def Abq (t : ℝ) : Vec (2 * (80 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 80 28 28 1 (kv 80 1) (kv 80 160) (Zbq t)

noncomputable def Zbe (t : ℝ) : Vec (2 * (480 * 28 * 28)) :=
  StableHLO.batchMap 2 (flatConv (h := 28) (w := 28) (ctK 480 80 1 1 1) (kv 480 0)) (Abq t)

noncomputable def Abe (t : ℝ) : Vec (2 * (480 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 480 28 28 1 (kv 480 1) (kv 480 160) (Zbe t)

noncomputable def Zbd (t : ℝ) : Vec (2 * (480 * 14 * 14)) :=
  StableHLO.batchMap 2 (depthwiseStride2Flat (h := 14) (w := 14) (ctDW 480 5 5 1) (kv 480 0))
    (Abe t)

noncomputable def Abd (t : ℝ) : Vec (2 * (480 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 480 14 14 1 (kv 480 1) (kv 480 160) (Zbd t)

noncomputable def Zbz (t : ℝ) : Vec (2 * (160 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 160 480 1 1 1) (kv 160 0)) (Abd t)

noncomputable def Abz (t : ℝ) : Vec (2 * (160 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 160 14 14 1 (kv 160 1) (kv 160 0) (Zbz t)

noncomputable def Zcq (t : ℝ) : Vec (2 * (160 * 14 * 14)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 14) (w := 14) (ctDW 160 5 5 1) (kv 160 0)) (Abz t)

noncomputable def Acq (t : ℝ) : Vec (2 * (160 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 160 14 14 1 (kv 160 1) (kv 160 160) (Zcq t)

noncomputable def Zce (t : ℝ) : Vec (2 * (960 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 960 160 1 1 1) (kv 960 0)) (Acq t)

noncomputable def Ace (t : ℝ) : Vec (2 * (960 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 960 14 14 1 (kv 960 1) (kv 960 160) (Zce t)

noncomputable def Zcd (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (depthwiseStride2Flat (h := 7) (w := 7) (ctDW 960 5 5 1) (kv 960 0))
    (Ace t)

noncomputable def Acd (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 1) (kv 960 160) (Zcd t)

noncomputable def Zcz (t : ℝ) : Vec (2 * (256 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 256 960 1 1 1) (kv 256 0)) (Acd t)

noncomputable def Acz (t : ℝ) : Vec (2 * (256 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 256 7 7 1 (kv 256 1) (kv 256 0) (Zcz t)

noncomputable def Zh1 (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 960 256 1 1 1) (kv 960 0)) (Acz t)

noncomputable def Ah1 (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 1) (kv 960 160) (Zh1 t)

/-- the pooled head features at `[2, 960, 1, 1]`: GAP of the first head stage, relabelled. -/
noncomputable def Pl (t : ℝ) : Vec (2 * (960 * 1 * 1)) :=
  fun k => StableHLO.batchMap 2 (globalAvgPoolFlat 960 7 7) (Ah1 t) (Fin.cast (mnv4_pool11 2 960).symm k)

noncomputable def Zh2 (t : ℝ) : Vec (2 * (1280 * 1 * 1)) :=
  StableHLO.batchMap 2 (flatConv (h := 1) (w := 1) (ctK 1280 960 1 1 1) (kv 1280 0)) (Pl t)

noncomputable def Ah2 (t : ℝ) : Vec (2 * (1280 * 1 * 1)) :=
  StableHLO.bnBatchLA 2 1280 1 1 1 (kv 1280 1) (kv 1280 160) (Zh2 t)

-- ════════════════════════════════════════════════════════════════
-- § 10. The carrier, stage by stage
--   ⭐ A centre-tap conv collapses `δ` to `s · δ 0` at every output channel; a centre-tap
--   DEPTHWISE cannot broadcast, so it scales `δ` channel by channel. Only `δ 0` is ever read.
-- ════════════════════════════════════════════════════════════════
/-- one carrier BatchNorm's whole contribution: `γ · istd` at `γ = 1`. -/
noncomputable def rf (n : Nat) (z : Vec n) : ℝ := bnIstd n z 1

theorem rf_pos (n : Nat) (z : Vec n) : 0 < rf n z := bnIstd_pos _ 1 one_pos

/-- the carrier at the input: `t` on channel 0. -/
noncomputable def dx (t : ℝ) : Fin 3 → ℝ := fun ci => if ci.val = 0 then t else 0

noncomputable def ds (t : ℝ) : Fin 32 → ℝ :=
  fun o => dx t 0 * rf (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Zs t) o)

noncomputable def df (t : ℝ) : Fin 128 → ℝ :=
  fun o => ds t 0 * rf (2 * (56 * 56)) (bnRowLA 2 128 56 56 (Zf t) o)

noncomputable def d1p (t : ℝ) : Fin 48 → ℝ :=
  fun o => df t 0 * rf (2 * (56 * 56)) (bnRowLA 2 48 56 56 (Z1p t) o)

noncomputable def daq (t : ℝ) : Fin 48 → ℝ :=
  fun ch => d1p t ch * rf (2 * (56 * 56)) (bnRowLA 2 48 56 56 (Zaq t) ch)

noncomputable def dae (t : ℝ) : Fin 192 → ℝ :=
  fun o => daq t 0 * rf (2 * (56 * 56)) (bnRowLA 2 192 56 56 (Zae t) o)

noncomputable def dad (t : ℝ) : Fin 192 → ℝ :=
  fun ch => dae t ch * rf (2 * (28 * 28)) (bnRowLA 2 192 28 28 (Zad t) ch)

noncomputable def daz (t : ℝ) : Fin 80 → ℝ :=
  fun o => dad t 0 * rf (2 * (28 * 28)) (bnRowLA 2 80 28 28 (Zaz t) o)

noncomputable def dbq (t : ℝ) : Fin 80 → ℝ :=
  fun ch => daz t ch * rf (2 * (28 * 28)) (bnRowLA 2 80 28 28 (Zbq t) ch)

noncomputable def dbe (t : ℝ) : Fin 480 → ℝ :=
  fun o => dbq t 0 * rf (2 * (28 * 28)) (bnRowLA 2 480 28 28 (Zbe t) o)

noncomputable def dbd (t : ℝ) : Fin 480 → ℝ :=
  fun ch => dbe t ch * rf (2 * (14 * 14)) (bnRowLA 2 480 14 14 (Zbd t) ch)

noncomputable def dbz (t : ℝ) : Fin 160 → ℝ :=
  fun o => dbd t 0 * rf (2 * (14 * 14)) (bnRowLA 2 160 14 14 (Zbz t) o)

noncomputable def dcq (t : ℝ) : Fin 160 → ℝ :=
  fun ch => dbz t ch * rf (2 * (14 * 14)) (bnRowLA 2 160 14 14 (Zcq t) ch)

noncomputable def dce (t : ℝ) : Fin 960 → ℝ :=
  fun o => dcq t 0 * rf (2 * (14 * 14)) (bnRowLA 2 960 14 14 (Zce t) o)

noncomputable def dcd (t : ℝ) : Fin 960 → ℝ :=
  fun ch => dce t ch * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zcd t) ch)

noncomputable def dcz (t : ℝ) : Fin 256 → ℝ :=
  fun o => dcd t 0 * rf (2 * (7 * 7)) (bnRowLA 2 256 7 7 (Zcz t) o)

noncomputable def dh1 (t : ℝ) : Fin 960 → ℝ :=
  fun o => dcz t 0 * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zh1 t) o)

noncomputable def dh2 (t : ℝ) : Fin 1280 → ℝ :=
  fun o => dh1 t 0 * rf (2 * (1 * 1)) (bnRowLA 2 1280 1 1 (Zh2 t) o)

theorem eds (t : ℝ) : EDiff (ds t) (As t) :=
  eDiff_convS2Bn (h := 112) (w := 112) (0 : Fin 3) rfl (by norm_num) (by norm_num) 1 160 (Zs t)
    (ed_sealX t) rfl (fun ci => by simp only [ds, rf, dx, Fin.val_zero, ite_true]; ring)

theorem edf (t : ℝ) : EDiff (df t) (Af t) :=
  eDiff_convS2Bn (h := 56) (w := 56) (0 : Fin 32) rfl (by norm_num) (by norm_num) 1 160 (Zf t)
    (eds t) rfl (fun ci => by simp only [df, rf]; ring)

theorem ed1p (t : ℝ) : EDiff (d1p t) (A1p t) :=
  eDiff_convBn (h := 56) (w := 56) (0 : Fin 128) rfl (by norm_num) (by norm_num) 1 0 (Z1p t)
    (edf t) rfl (fun ci => by simp only [d1p, rf]; ring)

theorem edaq (t : ℝ) : EDiff (daq t) (Aaq t) :=
  eDiff_dwBn (h := 56) (w := 56) (by norm_num) (by norm_num) 1 160 (Zaq t) (ed1p t) rfl
    (fun ci => by simp only [daq, rf]; ring)

theorem edae (t : ℝ) : EDiff (dae t) (Aae t) :=
  eDiff_convBn (h := 56) (w := 56) (0 : Fin 48) rfl (by norm_num) (by norm_num) 1 160 (Zae t)
    (edaq t) rfl (fun ci => by simp only [dae, rf]; ring)

theorem edad (t : ℝ) : EDiff (dad t) (Aad t) :=
  eDiff_dwS2Bn (h := 28) (w := 28) (by norm_num) (by norm_num) 1 160 (Zad t) (edae t) rfl
    (fun ci => by simp only [dad, rf]; ring)

theorem edaz (t : ℝ) : EDiff (daz t) (Aaz t) :=
  eDiff_convBn (h := 28) (w := 28) (0 : Fin 192) rfl (by norm_num) (by norm_num) 1 0 (Zaz t)
    (edad t) rfl (fun ci => by simp only [daz, rf]; ring)

theorem edbq (t : ℝ) : EDiff (dbq t) (Abq t) :=
  eDiff_dwBn (h := 28) (w := 28) (by norm_num) (by norm_num) 1 160 (Zbq t) (edaz t) rfl
    (fun ci => by simp only [dbq, rf]; ring)

theorem edbe (t : ℝ) : EDiff (dbe t) (Abe t) :=
  eDiff_convBn (h := 28) (w := 28) (0 : Fin 80) rfl (by norm_num) (by norm_num) 1 160 (Zbe t)
    (edbq t) rfl (fun ci => by simp only [dbe, rf]; ring)

theorem edbd (t : ℝ) : EDiff (dbd t) (Abd t) :=
  eDiff_dwS2Bn (h := 14) (w := 14) (by norm_num) (by norm_num) 1 160 (Zbd t) (edbe t) rfl
    (fun ci => by simp only [dbd, rf]; ring)

theorem edbz (t : ℝ) : EDiff (dbz t) (Abz t) :=
  eDiff_convBn (h := 14) (w := 14) (0 : Fin 480) rfl (by norm_num) (by norm_num) 1 0 (Zbz t)
    (edbd t) rfl (fun ci => by simp only [dbz, rf]; ring)

theorem edcq (t : ℝ) : EDiff (dcq t) (Acq t) :=
  eDiff_dwBn (h := 14) (w := 14) (by norm_num) (by norm_num) 1 160 (Zcq t) (edbz t) rfl
    (fun ci => by simp only [dcq, rf]; ring)

theorem edce (t : ℝ) : EDiff (dce t) (Ace t) :=
  eDiff_convBn (h := 14) (w := 14) (0 : Fin 160) rfl (by norm_num) (by norm_num) 1 160 (Zce t)
    (edcq t) rfl (fun ci => by simp only [dce, rf]; ring)

theorem edcd (t : ℝ) : EDiff (dcd t) (Acd t) :=
  eDiff_dwS2Bn (h := 7) (w := 7) (by norm_num) (by norm_num) 1 160 (Zcd t) (edce t) rfl
    (fun ci => by simp only [dcd, rf]; ring)

theorem edcz (t : ℝ) : EDiff (dcz t) (Acz t) :=
  eDiff_convBn (h := 7) (w := 7) (0 : Fin 960) rfl (by norm_num) (by norm_num) 1 0 (Zcz t)
    (edcd t) rfl (fun ci => by simp only [dcz, rf]; ring)

theorem edh1 (t : ℝ) : EDiff (dh1 t) (Ah1 t) :=
  eDiff_convBn (h := 7) (w := 7) (0 : Fin 256) rfl (by norm_num) (by norm_num) 1 160 (Zh1 t)
    (edcz t) rfl (fun ci => by simp only [dh1, rf]; ring)

/-- the pool and its relabel keep the carrier. -/
theorem eDiff_pl (t : ℝ) : EDiff (h := 1) (w := 1) (dh1 t) (Pl t) :=
  eDiff_gapTo11 (by norm_num) (by norm_num) (dh1 t) (Ah1 t) (edh1 t)

theorem edh2 (t : ℝ) : EDiff (dh2 t) (Ah2 t) :=
  eDiff_convBn (h := 1) (w := 1) (0 : Fin 960) rfl (by norm_num) (by norm_num) 1 160 (Zh2 t)
    (eDiff_pl t) rfl (fun ci => by simp only [dh2, rf]; ring)

-- ════════════════════════════════════════════════════════════════
-- § 11. `Rr` — the seventeen carrier BatchNorm factors
--   ⚠ No BatchNorm VARIANCE derivative is ever taken: `Rr` enters only as a continuous factor.
-- ════════════════════════════════════════════════════════════════
noncomputable def Rr (t : ℝ) : ℝ :=
  rf (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Zs t) 0) *
    (rf (2 * (56 * 56)) (bnRowLA 2 128 56 56 (Zf t) 0) *
    (rf (2 * (56 * 56)) (bnRowLA 2 48 56 56 (Z1p t) 0) *
    (rf (2 * (56 * 56)) (bnRowLA 2 48 56 56 (Zaq t) 0) *
    (rf (2 * (56 * 56)) (bnRowLA 2 192 56 56 (Zae t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 192 28 28 (Zad t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 80 28 28 (Zaz t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 80 28 28 (Zbq t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 480 28 28 (Zbe t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 480 14 14 (Zbd t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 160 14 14 (Zbz t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 160 14 14 (Zcq t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 960 14 14 (Zce t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zcd t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 256 7 7 (Zcz t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zh1 t) 0) *
    (rf (2 * (1 * 1)) (bnRowLA 2 1280 1 1 (Zh2 t) 0)))))))))))))))))

theorem Rr_pos (t : ℝ) : 0 < Rr t := by
  unfold Rr
  -- ⚠ one explicit factor per carrier BatchNorm, not `repeat' apply mul_pos`: `rf` would be
  -- split inside and leave goals `rf_pos` cannot close.
  exact (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (rf_pos _ _)))))))))))))))))

/-- `Rr` is continuous: `fun_prop` composes the carrier's seventeen activations (each written off
    the one before, back to `sealX`) once they are unfolded. Every BN has `ε = 1`. -/
theorem Rr_continuous : Continuous Rr := by
  unfold Rr rf
  repeat
    (first
      | unfold Zs | unfold As | unfold Zf | unfold Af | unfold Z1p | unfold A1p | unfold Zaq
      | unfold Aaq | unfold Zae | unfold Aae | unfold Zad | unfold Aad | unfold Zaz | unfold Aaz
      | unfold Zbq | unfold Abq | unfold Zbe | unfold Abe | unfold Zbd | unfold Abd | unfold Zbz
      | unfold Abz | unfold Zcq | unfold Acq | unfold Zce | unfold Ace | unfold Zcd | unfold Acd
      | unfold Zcz | unfold Acz | unfold Zh1 | unfold Ah1 | unfold Pl | unfold Zh2 | unfold Ah2)
  unfold sealX
  fun_prop (disch := exact one_pos)

-- ════════════════════════════════════════════════════════════════
-- § 12. The collapsed trunk — which activation each prefix IS
--   ⭐ The eighteen skipped rows are the EXACT identity (`sealZBody_eq` + `resid_id`): with the
--   project BatchNorm's `β = 0` a zeroed body is the constant `0`.
--   ⚠ Each step is restated at `(sealW nCls).b_k` before rewriting — `rw` does not see through a
--   structure projection, and `sealZ mnv4Row_k` is only DEFEQ to it.
-- ════════════════════════════════════════════════════════════════
/-- the stem's relu is the identity here, so `mnv4Pre0` IS the stem BatchNorm. -/
theorem pc0 (nCls : Nat) (t : ℝ) : mnv4Pre0 2 (sealW nCls) (sealX t) = As t :=
  mnv4StemB_eq (ctK 32 3 3 3 1) (kv 32 0) (by norm_num) (sealX t)

theorem pc1 (nCls : Nat) (t : ℝ) : mnv4Pre1 2 (sealW nCls) (sealX t) = A1p t := by
  show (mnv4FusedStack 2 (sealW nCls)).fwd (mnv4Pre0 2 (sealW nCls) (sealX t)) = _
  rw [pc0]
  simp only [mnv4FusedStack, mnv4FusedStage, CertLayer.comp_fwd_apply]
  show projB 2 (h := 56) (w := 56) (ctK 48 128 1 1 1) (kv 48 0) 1 (kv 48 1) (kv 48 0)
      (StableHLO.cbReluStridedB 2 (h := 56) (w := 56) (ctK 128 32 3 3 1) (kv 128 0) 1
        (kv 128 1) (kv 128 160) (As t)) = _
  rw [cbReluStridedB_eq _ _ (by norm_num)]
  rfl

theorem pc2 (nCls : Nat) (t : ℝ) : mnv4Pre2 2 (sealW nCls) (sealX t) = Aaz t := by
  have hs : (mnv4StridedBodyOfRow 2 mnv4Row1 (sealW nCls).b1).fwd (A1p t) = Aaz t :=
    sealCTStrided_eq 2 mnv4Row1 (by norm_num) (by norm_num) (by norm_num) (A1p t)
  have e2 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row2 (sealW nCls).b2)).fwd (Aaz t)
      = Aaz t := resid_id _ _ (sealZBody_eq 2 mnv4Row2 (by norm_num) _)
  show (mnv4Res28Layer 2 (sealW nCls)).fwd (mnv4Pre1 2 (sealW nCls) (sealX t)) = _
  rw [mnv4Res28Layer_fwd_apply, pc1, hs, e2]

theorem pc3 (nCls : Nat) (t : ℝ) : mnv4Pre3 2 (sealW nCls) (sealX t) = Abz t := by
  have hs : (mnv4StridedBodyOfRow 2 mnv4Row3 (sealW nCls).b3).fwd (Aaz t) = Abz t :=
    sealCTStrided_eq 2 mnv4Row3 (by norm_num) (by norm_num) (by norm_num) (Aaz t)
  have e4 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row4 (sealW nCls).b4)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row4 (by norm_num) _)
  have e5 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row5 (sealW nCls).b5)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row5 (by norm_num) _)
  have e6 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row6 (sealW nCls).b6)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row6 (by norm_num) _)
  show (mnv4Res14aLayer 2 (sealW nCls)).fwd (mnv4Pre2 2 (sealW nCls) (sealX t)) = _
  rw [mnv4Res14aLayer_fwd_apply, pc2, hs, e4, e5, e6]

theorem pc4 (nCls : Nat) (t : ℝ) : mnv4Pre4 2 (sealW nCls) (sealX t) = Abz t := by
  have e7 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row7 (sealW nCls).b7)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row7 (by norm_num) _)
  have e8 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row8 (sealW nCls).b8)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row8 (by norm_num) _)
  have e9 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row9 (sealW nCls).b9)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row9 (by norm_num) _)
  have e10 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row10 (sealW nCls).b10)).fwd (Abz t)
      = Abz t := resid_id _ _ (sealZBody_eq 2 mnv4Row10 (by norm_num) _)
  show (mnv4Res14bLayer 2 (sealW nCls)).fwd (mnv4Pre3 2 (sealW nCls) (sealX t)) = _
  rw [mnv4Res14bLayer_fwd_apply, pc3, e7, e8, e9, e10]

theorem pc5 (nCls : Nat) (t : ℝ) : mnv4Pre5 2 (sealW nCls) (sealX t) = Acz t := by
  have hs : (mnv4StridedBodyOfRow 2 mnv4Row11 (sealW nCls).b11).fwd (Abz t) = Acz t :=
    sealCTStrided_eq 2 mnv4Row11 (by norm_num) (by norm_num) (by norm_num) (Abz t)
  have e12 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row12 (sealW nCls).b12)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row12 (by norm_num) _)
  have e13 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row13 (sealW nCls).b13)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row13 (by norm_num) _)
  have e14 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row14 (sealW nCls).b14)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row14 (by norm_num) _)
  have e15 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row15 (sealW nCls).b15)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row15 (by norm_num) _)
  show (mnv4Res7aLayer 2 (sealW nCls)).fwd (mnv4Pre4 2 (sealW nCls) (sealX t)) = _
  rw [mnv4Res7aLayer_fwd_apply, pc4, hs, e12, e13, e14, e15]

theorem pc6 (nCls : Nat) (t : ℝ) : mnv4Pre6 2 (sealW nCls) (sealX t) = Acz t := by
  have e16 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row16 (sealW nCls).b16)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row16 (by norm_num) _)
  have e17 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row17 (sealW nCls).b17)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row17 (by norm_num) _)
  have e18 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row18 (sealW nCls).b18)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row18 (by norm_num) _)
  have e19 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row19 (sealW nCls).b19)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row19 (by norm_num) _)
  have e20 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row20 (sealW nCls).b20)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row20 (by norm_num) _)
  have e21 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row21 (sealW nCls).b21)).fwd (Acz t)
      = Acz t := resid_id _ _ (sealZBody_eq 2 mnv4Row21 (by norm_num) _)
  show (mnv4Res7bLayer 2 (sealW nCls)).fwd (mnv4Pre5 2 (sealW nCls) (sealX t)) = _
  rw [mnv4Res7bLayer_fwd_apply, pc5, e16, e17, e18, e19, e20, e21]

-- ════════════════════════════════════════════════════════════════
-- § 13. The head reads the carrier off channel 0
-- ════════════════════════════════════════════════════════════════
theorem sealW_Wd (nCls : Nat) :
    (sealW nCls).Wd = fun (i : Fin 1280) (j : Fin nCls) =>
      if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0 := rfl

/-- the head, peeled: `cn_960` conv-bn-relu, GAP and its relabel, `conv_head` conv-bn-relu at
    `1×1`, the relabel back, the classifier. Proved at variables and applied at the net's
    literals. -/
theorem headStack_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (256 * 7 * 7))) :
    (mnv4HeadStack N w).fwd v
      = StableHLO.batchMap N (dense w.Wd w.bd)
          (fun k => StableHLO.cbReluB N (h := 1) (w := 1) w.hW w.hb w.hE w.hg w.hbt
            (fun j => StableHLO.batchMap N (globalAvgPoolFlat 960 7 7)
              (StableHLO.cbReluB N (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt v)
              (Fin.cast (mnv4_pool11 N 960).symm j))
            (Fin.cast (mnv4_pool11 N 1280) k)) := by
  simp only [mnv4HeadStack, mnv4Head, CertLayer.comp_fwd_apply, cbReluLayer_fwd_apply,
    gapLayer_fwd_apply, denseLayer_fwd_apply, castLayer_fwd_apply]

theorem head_eq_dense (nCls : Nat) (t : ℝ) :
    mobilenetv4ForwardBFull 2 (sealW nCls) (sealX t)
      = StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
          (fun k => Ah2 t (Fin.cast (mnv4_pool11 2 1280) k)) := by
  show (mnv4HeadStack 2 (sealW nCls)).fwd (mnv4Pre6 2 (sealW nCls) (sealX t)) = _
  rw [headStack_apply, pc6]
  show StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
      (fun k => StableHLO.cbReluB 2 (h := 1) (w := 1) (ctK 1280 960 1 1 1) (kv 1280 0) 1
          (kv 1280 1) (kv 1280 160)
        (fun j => StableHLO.batchMap 2 (globalAvgPoolFlat 960 7 7)
          (StableHLO.cbReluB 2 (h := 7) (w := 7) (ctK 960 256 1 1 1) (kv 960 0) 1 (kv 960 1)
            (kv 960 160) (Acz t))
          (Fin.cast (mnv4_pool11 2 960).symm j))
        (Fin.cast (mnv4_pool11 2 1280) k)) = _
  rw [cbReluB_eq (ctK 960 256 1 1 1) (kv 960 0) (by norm_num) (Acz t),
    cbReluB_eq (ctK 1280 960 1 1 1) (kv 1280 0) (by norm_num) _]
  rfl

/-- **The class-0 difference between the two examples, along the ray**: `t` times the
    seventeen BatchNorm factors on the carrier's path. -/
theorem gd_ray (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    mobilenetv4ForwardBFull 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - mobilenetv4ForwardBFull 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = t * Rr t := by
  rw [head_eq_dense, cls_diff (0 : Fin 1280) rfl ⟨0, hn⟩ _ _ (fun ci => by rw [sealW_Wd]; simp) rfl
    (Ah2 t) (dh2 t) (edh2 t)]
  simp only [dh2, dh1, dcz, dcd, dce, dcq, dbz, dbd, dbe, dbq, daz, dad, dae, daq, d1p, df, ds, dx, Rr]
  norm_num
  ring

-- ════════════════════════════════════════════════════════════════
-- § 14. The derivative along the ray, and the seal
-- ════════════════════════════════════════════════════════════════
theorem hasDerivAt_gd : HasDerivAt (fun t : ℝ => t * Rr t) (Rr 0) 0 :=
  hasDerivAt_mul_self_zero Rr_continuous.continuousAt

theorem seal_differentiableAt (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (mobilenetv4ForwardBFull 2 (sealW nCls)) (sealX t) :=
  mobilenetv4ForwardBFull_differentiableAt 2 (sealW nCls) (sealX t) (seal_smooth nCls t)

/-- **Level 2 — the witness is non-degenerate**: the full-width batch-BN MobileNetV4-Conv-M at
    the structural weights is NOT constant in its input. -/
theorem sealX_nonconstant (nCls : Nat) (hn : 0 < nCls) :
    mobilenetv4ForwardBFull 2 (sealW nCls) (sealX 1)
      ≠ mobilenetv4ForwardBFull 2 (sealW nCls) (sealX 0) :=
  ne_of_ray_readout _ sealX _ _ (gd_ray nCls hn) (by simpa using (Rr_pos 1).ne')

/-- **Level 3 — the whole-net Jacobian is nonzero at the witness.** -/
theorem sealX_jacobian_nonzero (nCls : Nat) (hn : 0 < nCls) :
    fderiv ℝ (mobilenetv4ForwardBFull 2 (sealW nCls)) (sealX 0) ≠ 0 :=
  fderiv_ne_zero_of_ray_readout _ sealX sealV sealX_zero_add _ _ (gd_ray nCls hn)
    (seal_differentiableAt nCls 0) (Rr_pos 0).ne' hasDerivAt_gd

/-- **The seal**: the proven whole-network backward of the full-width, batch-BatchNorm,
    21-block, 224×224 MobileNetV4-Conv-M — `mobilenetv4ForwardBFull`, the training-BatchNorm
    forward of the MobileNetV4 train steps (the `mnv4in_emaacc*` ones add classifier dropout),
    here at `N = 2` — is **not the zero map** at the witness. -/
theorem sealX_backward_nontrivial (nCls : Nat) (hn : 0 < nCls) :
    ∃ (j₀ : Fin (2 * nCls)) (i₀ : Fin (2 * (3 * (2 * 112) * (2 * 112)))),
      (sealVJP nCls 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (sealVJP nCls 0).backward_nontrivial_of_fderiv_ne (sealX_jacobian_nonzero nCls hn)

end Mnv4FullBSeal
end Proofs
