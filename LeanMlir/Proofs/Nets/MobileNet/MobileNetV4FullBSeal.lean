import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBSeal

/-!
# MobileNetV4-Conv-M's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`planning/full_width_seals.md` §4.4, the last of four. `MobileNetV4FullBVJP.lean` proves
`mobilenetv4ForwardB_full_has_vjp_at`: the whole-net VJP at any `(w, x)` satisfying
**eight clause bundles** — the stem's relu, the fused stage's (vacuous), one per resolution group
and the head's, **54** relu sites in all (the `#guard` below counts them off the block table). That statement is pointwise, so it could in
principle be vacuous. This file exhibits a `(w, x)` that discharges every clause with genuinely
nonzero weights, shows the forward is not constant there, and seals the Jacobian nonzero — hence,
through [`Training/JacobianSeal.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Training/JacobianSeal.lean), the proven backward is not the zero map at that
point. MobileNetV4 had no witness at any depth before this; with it, every kinked net in the book
is sealed on the forward its artifacts run.

## ⭐⭐ All 54 clauses are weight-only — and generic in the table row

Every kink in this net is a relu sitting directly on a `bnBatchLA` output, in four spellings
(`Mnv4StemSmoothAtB`, `cbReluLayer.ok`, `mnv4DWReluLayer.ok`, `mnv4DWReluStridedLayer.ok`);
`projLayer.ok`, `mnv4FusedConvLayer.ok` and `CertLayer.id'.ok` are `True`. `bnBatchLA_pos` bounds a
BatchNorm output above `β − |γ|√(N·h·w) > 0` at EVERY input, so with `γ = 1`, `β = 160`, `ε = 1`
the whole bundle is discharged without the activation ever being read. ⭐ Better still, the
discharge is proved **generically in the `UibSpec` row** (`sealUib_ok`, `sealUibStrided_ok`): the
`k = 0` depthwise slots contribute `True` and the rest are the same BatchNorm fact, so 21 blocks
cost two lemmas.

## ⭐⭐ The swish is the one thing that is not MobileNetV2 again

`planning/full_width_seals.md` §4.4 called the clause bundle the package's remaining uncertainty
and the `CertLayer` peel its real work. Both turned out cheap — `CertLayer.comp_fwd_apply` and the
group `*_fwd_apply` lemmas in `MobileNetV4FullB.lean` already peel at variables. What was missed is
the **fused stage's swish**, and it changes the witness:

* the other three seals ride a positionally-injective **ramp**, which a stem max-pool's no-tie
  needs. A carrier crosses their relus because relu is the *identity* inside the margin window;
* swish is the identity on no window at all. A gap between the two examples comes out of it as
  `swish(a) − swish(b)`, which is not a multiple of `a − b` and — worse — is not even constant over
  the grid unless `a` and `b` are;
* ⭐ so this witness's base is **grid-constant**, not a ramp: `sealX t = t • rayV`. Then every
  activation down to the fused BatchNorm is one value per (example, channel) (`BUnif`), that
  BatchNorm puts the two values symmetrically about `β` (`bnBatchLA_pair`), and the swish's two
  outputs differ by `swishGap β u` — a function of their half-gap `u` alone. `EDiff` takes over
  from there and every remaining stage is multiplicative. MobileNetV4 has no pool, so nothing
  wanted the ramp.

The readout along the ray is therefore `swishGap 160 (uF t 0) · Rr t`, **not** `t · Rr t`, and the
seal closes with `hasDerivAt_mul_of_zero` rather than `hasDerivAt_mul_self_zero`. ⚠ No BatchNorm
variance derivative is taken anywhere: the two pre-swish factors enter through `t · Q0 t` and the
fifteen after it through a continuous `Rr`; the swish's slope at `β` is the one honest derivative
in the chain, and it is positive for every `β ≥ 0`.

## The carrier threads seventeen BatchNorms

The stem's, the fused stage's two, four in each of rows 1, 3 and 11 — the only rows that change
channels, hence the only ones without a skip — and the head's two. ⛔ Counted from the net, not
from prose: the other eighteen rows are `CertLayer.residual`, so the carrier goes round their
bodies. With the project BatchNorm's `β = 0` a zeroed body is the constant `0`, so those rows are
the EXACT identity and two of the seven groups collapse to nothing at all.

⚠⚠ Every collapse, every clause bundle and every block lemma here is proved at **variable** shapes
and instantiated at the witness's numerals afterwards, never proved at them
(`planning/full_width_seals.md` §3.5). ⚠ And a lemma whose spatial dims are only reachable through
a `Vec (… (2*h) …)` argument needs them passed explicitly — `2 * ?h =?= 56` is nonlinear, and
`BUnif_convS2` without `(h := 56) (w := 56)` is a `maxHeartbeats` timeout in `isDefEq`.
-/

namespace Proofs
namespace Mnv4FullBSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs BatchSeal StableHLO R34FullBSeal

-- ⭐ **The clause count, read off `mnv4Blocks` rather than asserted.** One relu per present
-- depthwise slot and one per expand conv in each of the 21 UIB rows, plus the stem's and the
-- head's two; the fused stage is swish and contributes none, and no project conv has an
-- activation. `sealUib_ok` / `sealUibStrided_ok` discharge all of them in two lemmas.
#guard 1 + (StableHLO.mnv4Blocks.map (fun s =>
    (if s.preDWk = 0 then 0 else 1) + 1 + (if s.postDWk = 0 then 0 else 1))).sum + 2 = 54

set_option maxHeartbeats 1000000

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural weights
--   ⭐ One `γ = 1`, one `ε = 1` and `β = 160` at every BatchNorm a relu follows; `β = 0` at the
--   projections, which none follows, so a zeroed body is the constant `0` and its block is the
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
--   ⚠⚠ Proved at variable `N, ic, oc, h, w, kH, kW` and instantiated at the witness's numerals
--   afterwards, never proved at them (`planning/full_width_seals.md` §3.5).
-- ════════════════════════════════════════════════════════════════
theorem cbReluB_eq {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * h * w))) :
    StableHLO.cbReluB N (h := h) (w := w) W b 1 (kv oc 1) (kv oc 160) x
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConv W b) x) :=
  relu_id_of_pos (bpos N oc h w hm _)

theorem dwbReluB_eq {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (hm : Mg N h w) (x : Vec (N * (c * h * w))) :
    StableHLO.dwbReluB N (h := h) (w := w) W b 1 (kv c 1) (kv c 160) x
      = StableHLO.bnBatchLA N c h w 1 (kv c 1) (kv c 160)
          (StableHLO.batchMap N (depthwiseFlat W b) x) :=
  relu_id_of_pos (bpos N c h w hm _)

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
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) :=
  relu_id_of_pos (bpos N oc h w hm _)

-- ════════════════════════════════════════════════════════════════
-- § 4. The block collapses — one lemma per FORM, generic in the table row
-- ════════════════════════════════════════════════════════════════
/-- ⭐ **A zeroed UIB body is the constant `0`, whatever its slots are.** The project conv's kernel
    is zero, so `projB_zero_const` closes the block without any stage inside it being analysed —
    the `k = 0` dispatch never has to be read. -/
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

/-- ⭐ **A carrier row (all three are pre-strided ExtraDW) collapses to four BatchNorms.** -/
theorem sealCTStrided_eq (N : Nat) (s : UibSpec) (hd : s.postDWk ≠ 0)
    (hmd : Mg N s.h s.h) (v : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    (mnv4PreStridedBodyOfRow N s (sealCT s)).fwd v
      = StableHLO.bnBatchLA N s.oc s.h s.h 1 (kv s.oc 1) (kv s.oc 0)
          (StableHLO.batchMap N
            (flatConv (h := s.h) (w := s.h) (ctK s.oc (s.ic * s.expand) 1 1 1) (kv s.oc 0))
          (StableHLO.bnBatchLA N (s.ic * s.expand) s.h s.h 1 (kv (s.ic * s.expand) 1)
              (kv (s.ic * s.expand) 160)
          (StableHLO.batchMap N
            (depthwiseFlat (h := s.h) (w := s.h)
              (ctDW (s.ic * s.expand) s.postDWk s.postDWk 1) (kv (s.ic * s.expand) 0))
          (StableHLO.bnBatchLA N (s.ic * s.expand) s.h s.h 1 (kv (s.ic * s.expand) 1)
              (kv (s.ic * s.expand) 160)
          (StableHLO.batchMap N
            (flatConv (h := s.h) (w := s.h) (ctK (s.ic * s.expand) s.ic 1 1 1)
              (kv (s.ic * s.expand) 0))
          (StableHLO.bnBatchLA N s.ic s.h s.h 1 (kv s.ic 1) (kv s.ic 160)
            (StableHLO.batchMap N
              (depthwiseStride2Flat (h := s.h) (w := s.h) (ctDW s.ic s.preDWk s.preDWk 1)
                (kv s.ic 0)) v))))))) := by
  simp only [mnv4PreStridedBodyOfRow, mnv4UibPreStridedBody, CertLayer.comp_fwd_apply,
    mnv4PostDWSlot, hd, ↓reduceIte]
  show projB N (h := s.h) (w := s.h) (ctK s.oc (s.ic * s.expand) 1 1 1) (kv s.oc 0) 1
        (kv s.oc 1) (kv s.oc 0)
      (StableHLO.dwbReluB N (h := s.h) (w := s.h)
          (ctDW (s.ic * s.expand) s.postDWk s.postDWk 1) (kv (s.ic * s.expand) 0) 1
          (kv (s.ic * s.expand) 1) (kv (s.ic * s.expand) 160)
        (StableHLO.cbReluB N (h := s.h) (w := s.h) (ctK (s.ic * s.expand) s.ic 1 1 1)
            (kv (s.ic * s.expand) 0) 1 (kv (s.ic * s.expand) 1) (kv (s.ic * s.expand) 160)
          (StableHLO.dwbReluBstrided N (h := s.h) (w := s.h)
            (ctDW s.ic s.preDWk s.preDWk 1) (kv s.ic 0) 1 (kv s.ic 1) (kv s.ic 160) v))) = _
  rw [dwbReluBstrided_eq _ _ hmd, cbReluB_eq _ _ hmd, dwbReluB_eq _ _ hmd]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 5. The clause bundle — ⭐⭐ weight-only, and proved GENERICALLY IN THE ROW
-- ════════════════════════════════════════════════════════════════
theorem comp_ok {m n p : Nat} (L₁ : CertLayer m n) (L₂ : CertLayer n p) (v : Vec m)
    (h₁ : L₁.ok v) (h₂ : L₂.ok (L₁.fwd v)) : (L₁.comp L₂).ok v := ⟨h₁, h₂⟩

/-- ⭐ **Every clause of a UIB body, at every input.** The `k = 0` slots contribute `True` and the
    rest are relus on BatchNorm outputs, so the discharge never looks at the activation and holds
    for the centre-tap rows and the zeroed rows alike. -/
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
    · simp only [hk, ↓reduceIte]; exact bne N s.ic s.h s.h hm _
  · exact bne N (s.ic * s.expand) s.h s.h hm _
  · show (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk _ _ _ _ _ _).ok _
    unfold mnv4PostDWSlot
    by_cases hk : s.postDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; exact bne N (s.ic * s.expand) s.h s.h hm _

/-- the pre-strided peer: the leading depthwise carries the stride, so it is not a slot. -/
theorem sealUibStrided_ok (N : Nat) (s : UibSpec) (hm : Mg N s.h s.h)
    (Wq : DepthwiseKernel s.ic s.preDWk s.preDWk)
    (We : Kernel4 (s.ic * s.expand) s.ic 1 1)
    (Wd : DepthwiseKernel (s.ic * s.expand) s.postDWk s.postDWk)
    (Wz : Kernel4 s.oc (s.ic * s.expand) 1 1)
    (x : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) :
    (mnv4PreStridedBodyOfRow N s (sealP s Wq We Wd Wz)).ok x := by
  refine ⟨bne N s.ic s.h s.h hm _, ?_, ?_, trivial⟩
  · exact bne N (s.ic * s.expand) s.h s.h hm _
  · show (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk _ _ _ _ _ _).ok _
    unfold mnv4PostDWSlot
    by_cases hk : s.postDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; exact bne N (s.ic * s.expand) s.h s.h hm _

-- ════════════════════════════════════════════════════════════════
-- § 6. The ray
--   ⛔⛔ **NOT the kit's `rayX`.** Every other seal rides a positionally-injective RAMP, which a
--   max-pool's no-tie needs. MobileNetV4's fused stage is **swish**, and swish is not the identity
--   on any window — so a carrier can only cross it if the two examples' slabs are CONSTANT over
--   the grid, where `bnBatchLA` puts them symmetrically about `β` and the gap becomes a function
--   of itself. A ramp base does not survive the fused stage at all; this base is the whole reason
--   §8 exists. MobileNetV4 has no pool, so nothing wants the ramp.
-- ════════════════════════════════════════════════════════════════
noncomputable def sealV : Vec (2 * (3 * (2 * 112) * (2 * 112))) := rayV (2 * 112) (2 * 112)

noncomputable def sealX (t : ℝ) : Vec (2 * (3 * (2 * 112) * (2 * 112))) := t • sealV

theorem sealX_zero_add (t : ℝ) : sealX 0 + t • sealV = sealX t := by
  rw [sealX, sealX, zero_smul, zero_add]

theorem sealX_continuous : Continuous sealX :=
  continuous_id.smul continuous_const

/-- ⭐ the witness input is per-(example, channel) CONSTANT: `t` on example 0's channel 0, zero
    everywhere else. -/
theorem BUnif_sealX (t : ℝ) :
    BUnif (h := 2 * 112) (w := 2 * 112)
      (fun n ci => if n.val = 0 ∧ ci.val = 0 then t else 0) (sealX t) := by
  intro n ci i j
  show bcell (t • rayV (2 * 112) (2 * 112)) n ci i j = _
  rw [bcell_smul, rayV, bcell_bfrom]
  by_cases h : n.val = 0 ∧ ci.val = 0 <;> simp [h]

-- ════════════════════════════════════════════════════════════════
-- § 7. The eight clause bundles, and the whole-net VJP at the witness
-- ════════════════════════════════════════════════════════════════
theorem scStem (nCls : Nat) (t : ℝ) :
    Mnv4StemSmoothAtB 2 112 112 (sealW nCls).sW (sealW nCls).sb (sealW nCls).sE
      (sealW nCls).sg (sealW nCls).sbt (sealX t) :=
  bne 2 32 112 112 (by norm_num) _

theorem scFused (nCls : Nat) (t : ℝ) :
    (mnv4FusedStack 2 (sealW nCls)).ok (mnv4Pre0 2 (sealW nCls) (sealX t)) :=
  ⟨trivial, trivial⟩

theorem sc28 (nCls : Nat) (t : ℝ) :
    (mnv4Res28Layer 2 (sealW nCls)).ok (mnv4Pre1 2 (sealW nCls) (sealX t)) :=
  ⟨sealUibStrided_ok 2 mnv4Row1 (by norm_num) _ _ _ _ _,
   sealUib_ok 2 mnv4Row2 (by norm_num) _ _ _ _ _⟩

theorem sc14a (nCls : Nat) (t : ℝ) :
    (mnv4Res14aLayer 2 (sealW nCls)).ok (mnv4Pre2 2 (sealW nCls) (sealX t)) :=
  ⟨sealUibStrided_ok 2 mnv4Row3 (by norm_num) _ _ _ _ _,
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
  ⟨sealUibStrided_ok 2 mnv4Row11 (by norm_num) _ _ _ _ _,
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

theorem scHead (nCls : Nat) (t : ℝ) :
    (mnv4HeadStack 2 (sealW nCls)).ok (mnv4Pre6 2 (sealW nCls) (sealX t)) :=
  ⟨bne 2 960 7 7 (by norm_num) _, bne 2 1280 7 7 (by norm_num) _, trivial, trivial⟩

/-- ⭐⭐ **The whole-net VJP at the witness** — all eight bundles discharged. -/
noncomputable def sealVJP (nCls : Nat) (t : ℝ) :
    HasVJPAt (mobilenetv4ForwardB_full 2 (sealW nCls)) (sealX t) :=
  mobilenetv4ForwardB_full_has_vjp_at 2 (sealW nCls) (sealX t)
    ⟨scStem nCls t, scFused nCls t, sc28 nCls t, sc14a nCls t, sc14b nCls t, sc7a nCls t,
      sc7b nCls t, scHead nCls t⟩

-- ════════════════════════════════════════════════════════════════
-- § 8. ⭐⭐ Across the SWISH — the one stage a carrier cannot cross by the identity
--   Up to the fused stage's BatchNorm the witness is tracked as a PAIR of per-channel constants
--   (`BUnif`), not as a difference. That buys the one fact `EDiff` cannot: batch BN on a
--   grid-constant slab puts the two examples symmetrically about `β`, so their swish outputs
--   differ by `swishGap β u` — a function of their half-gap `u` alone. From there the carrier is
--   an `EDiff` again and every remaining stage is multiplicative.
-- ════════════════════════════════════════════════════════════════
noncomputable def Zs (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.batchMap 2 (flatConvStride2Xla (h := 112) (w := 112) (ctK 32 3 3 3 1) (kv 32 0))
    (sealX t)

noncomputable def As (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.bnBatchLA 2 32 112 112 1 (kv 32 1) (kv 32 160) (Zs t)

/-- the stem BatchNorm's `istd`, channel by channel. -/
noncomputable def iS (t : ℝ) (o : Fin 32) : ℝ :=
  bnIstd (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Zs t) o) 1

noncomputable def aS (t : ℝ) : Fin 2 → Fin 32 → ℝ :=
  fun n o => 160 + (if n.val = 0 then t / 2 else -(t / 2)) * iS t o

theorem bu_Zs (t : ℝ) :
    BUnif (h := 112) (w := 112) (fun n _ => if n.val = 0 then t else 0) (Zs t) := by
  refine BUnif_convS2Xla (0 : Fin 3) rfl (by norm_num) (by norm_num) 1 (kv 32 0)
    (fun n ci => if n.val = 0 ∧ ci.val = 0 then t else 0)
    (fun n _ => if n.val = 0 then t else 0) (sealX t) (BUnif_sealX t) ?_
  intro n o
  simp only [kv_apply, Fin.val_zero, and_true, one_mul, zero_add]

/-- ⭐ the stem BatchNorm puts the two examples at `160 ± t/2 · istd` — symmetrically about `β`,
    which is the fact the swish needs and a difference alone cannot give. -/
theorem bu_As (t : ℝ) : BUnif (h := 112) (w := 112) (aS t) (As t) := by
  refine bnBatchLA_pair (by norm_num) 1 (kv 32 1) (kv 32 160) _ (Zs t) (bu_Zs t) (aS t) ?_
  intro n o
  have h0 : (if (0 : Fin 2).val = 0 then t else 0) = t := by norm_num
  have h1 : (if (1 : Fin 2).val = 0 then t else 0) = (0 : ℝ) := by norm_num
  simp only [aS, iS, kv_apply, one_mul, h0, h1]
  split_ifs with h
  · ring
  · ring

/-- the stem's relu is the identity here, so `mnv4Pre0` IS the stem BatchNorm. -/
theorem pc0 (nCls : Nat) (t : ℝ) : mnv4Pre0 2 (sealW nCls) (sealX t) = As t :=
  mnv4StemB_eq (ctK 32 3 3 3 1) (kv 32 0) (by norm_num) (sealX t)

noncomputable def Zf (t : ℝ) : Vec (2 * (128 * 56 * 56)) :=
  StableHLO.batchMap 2 (flatConvStride2 (h := 56) (w := 56) (ctK 128 32 3 3 1) (kv 128 0)) (As t)

noncomputable def Af (t : ℝ) : Vec (2 * (128 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 128 56 56 1 (kv 128 1) (kv 128 160) (Zf t)

/-- the fused conv's BatchNorm `istd`, channel by channel. -/
noncomputable def iF (t : ℝ) (o : Fin 128) : ℝ :=
  bnIstd (2 * (56 * 56)) (bnRowLA 2 128 56 56 (Zf t) o) 1

/-- ⭐ **the half-gap at the swish's input** — `t` times the two pre-swish BatchNorm factors. -/
noncomputable def uF (t : ℝ) (o : Fin 128) : ℝ := t / 2 * iS t 0 * iF t o

noncomputable def aF (t : ℝ) : Fin 2 → Fin 128 → ℝ :=
  fun n o => 160 + (if n.val = 0 then uF t o else -(uF t o))

theorem bu_Zf (t : ℝ) :
    BUnif (h := 56) (w := 56)
      (fun n _ => 160 + (if n.val = 0 then t / 2 else -(t / 2)) * iS t 0) (Zf t) := by
  refine BUnif_convS2 (h := 56) (w := 56) (0 : Fin 32) rfl (by norm_num) (by norm_num) 1
    (kv 128 0) (aS t)
    (fun n _ => 160 + (if n.val = 0 then t / 2 else -(t / 2)) * iS t 0) (As t) (bu_As t) ?_
  intro n o
  simp only [aS, kv_apply, one_mul, zero_add]

theorem bu_Af (t : ℝ) : BUnif (h := 56) (w := 56) (aF t) (Af t) := by
  refine bnBatchLA_pair (by norm_num) 1 (kv 128 1) (kv 128 160) _ (Zf t) (bu_Zf t) (aF t) ?_
  intro n o
  have h0 : (160 : ℝ) + (if (0 : Fin 2).val = 0 then t / 2 else -(t / 2)) * iS t 0
      = 160 + t / 2 * iS t 0 := by norm_num
  have h1 : (160 : ℝ) + (if (1 : Fin 2).val = 0 then t / 2 else -(t / 2)) * iS t 0
      = 160 - t / 2 * iS t 0 := by norm_num; ring
  simp only [aF, uF, iF, kv_apply, one_mul, h0, h1]
  split_ifs with h
  · ring
  · ring

/-- the fused stage's swish output. -/
noncomputable def Sw (t : ℝ) : Vec (2 * (128 * 56 * 56)) := swish (2 * (128 * 56 * 56)) (Af t)

/-- ⭐⭐ **the carrier crosses the swish.** The two examples straddle `β = 160`, so their outputs
    differ by `swishGap 160` of the half-gap — and from here on the carrier is an `EDiff` and
    every stage multiplies it. -/
theorem ed_Sw (t : ℝ) : EDiff (fun o => swishGap 160 (uF t o)) (Sw t) := by
  refine EDiff_of_BUnif (fun n o => swishScalar (aF t n o)) _ (Sw t) ?_ ?_
  · exact BUnif_map swishScalar (aF t) _ (Af t) (bu_Af t) (fun _ _ => rfl)
  · intro o
    simp only [swishGap, aF, Fin.isValue]
    norm_num
    rw [sub_eq_add_neg]

-- ════════════════════════════════════════════════════════════════
-- § 9. The fifteen post-swish activations on the carrier's path
--   ⚠ Each is written in terms of the PREVIOUS one, never in terms of `mnv4Pre_k`: §11's
--   continuity chain then runs straight off `sealX_continuous`, and `pc0`–`pc6` say which prefix
--   each one IS. ⚠ Spatial sizes stay in the net's own nest.
-- ════════════════════════════════════════════════════════════════
noncomputable def Z1p (t : ℝ) : Vec (2 * (48 * 56 * 56)) :=
  StableHLO.batchMap 2 (flatConv (h := 56) (w := 56) (ctK 48 128 1 1 1) (kv 48 0))
    (Sw t)

noncomputable def A1p (t : ℝ) : Vec (2 * (48 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 48 56 56 1 (kv 48 1) (kv 48 0) (Z1p t)

noncomputable def Zaq (t : ℝ) : Vec (2 * (48 * 28 * 28)) :=
  StableHLO.batchMap 2 (depthwiseStride2Flat (h := 28) (w := 28) (ctDW 48 3 3 1) (kv 48 0))
    (A1p t)

noncomputable def Aaq (t : ℝ) : Vec (2 * (48 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 48 28 28 1 (kv 48 1) (kv 48 160) (Zaq t)

noncomputable def Zae (t : ℝ) : Vec (2 * (192 * 28 * 28)) :=
  StableHLO.batchMap 2 (flatConv (h := 28) (w := 28) (ctK 192 48 1 1 1) (kv 192 0))
    (Aaq t)

noncomputable def Aae (t : ℝ) : Vec (2 * (192 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 192 28 28 1 (kv 192 1) (kv 192 160) (Zae t)

noncomputable def Zad (t : ℝ) : Vec (2 * (192 * 28 * 28)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 28) (w := 28) (ctDW 192 5 5 1) (kv 192 0))
    (Aae t)

noncomputable def Aad (t : ℝ) : Vec (2 * (192 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 192 28 28 1 (kv 192 1) (kv 192 160) (Zad t)

noncomputable def Zaz (t : ℝ) : Vec (2 * (80 * 28 * 28)) :=
  StableHLO.batchMap 2 (flatConv (h := 28) (w := 28) (ctK 80 192 1 1 1) (kv 80 0))
    (Aad t)

noncomputable def Aaz (t : ℝ) : Vec (2 * (80 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 80 28 28 1 (kv 80 1) (kv 80 0) (Zaz t)

noncomputable def Zbq (t : ℝ) : Vec (2 * (80 * 14 * 14)) :=
  StableHLO.batchMap 2 (depthwiseStride2Flat (h := 14) (w := 14) (ctDW 80 3 3 1) (kv 80 0))
    (Aaz t)

noncomputable def Abq (t : ℝ) : Vec (2 * (80 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 80 14 14 1 (kv 80 1) (kv 80 160) (Zbq t)

noncomputable def Zbe (t : ℝ) : Vec (2 * (480 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 480 80 1 1 1) (kv 480 0))
    (Abq t)

noncomputable def Abe (t : ℝ) : Vec (2 * (480 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 480 14 14 1 (kv 480 1) (kv 480 160) (Zbe t)

noncomputable def Zbd (t : ℝ) : Vec (2 * (480 * 14 * 14)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 14) (w := 14) (ctDW 480 5 5 1) (kv 480 0))
    (Abe t)

noncomputable def Abd (t : ℝ) : Vec (2 * (480 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 480 14 14 1 (kv 480 1) (kv 480 160) (Zbd t)

noncomputable def Zbz (t : ℝ) : Vec (2 * (160 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 160 480 1 1 1) (kv 160 0))
    (Abd t)

noncomputable def Abz (t : ℝ) : Vec (2 * (160 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 160 14 14 1 (kv 160 1) (kv 160 0) (Zbz t)

noncomputable def Zcq (t : ℝ) : Vec (2 * (160 * 7 * 7)) :=
  StableHLO.batchMap 2 (depthwiseStride2Flat (h := 7) (w := 7) (ctDW 160 5 5 1) (kv 160 0))
    (Abz t)

noncomputable def Acq (t : ℝ) : Vec (2 * (160 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 160 7 7 1 (kv 160 1) (kv 160 160) (Zcq t)

noncomputable def Zce (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 960 160 1 1 1) (kv 960 0))
    (Acq t)

noncomputable def Ace (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 1) (kv 960 160) (Zce t)

noncomputable def Zcd (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 7) (w := 7) (ctDW 960 5 5 1) (kv 960 0))
    (Ace t)

noncomputable def Acd (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 1) (kv 960 160) (Zcd t)

noncomputable def Zcz (t : ℝ) : Vec (2 * (256 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 256 960 1 1 1) (kv 256 0))
    (Acd t)

noncomputable def Acz (t : ℝ) : Vec (2 * (256 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 256 7 7 1 (kv 256 1) (kv 256 0) (Zcz t)

noncomputable def Zh1 (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 960 256 1 1 1) (kv 960 0))
    (Acz t)

noncomputable def Ah1 (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 1) (kv 960 160) (Zh1 t)

noncomputable def Zh2 (t : ℝ) : Vec (2 * (1280 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 1280 960 1 1 1) (kv 1280 0))
    (Ah1 t)

noncomputable def Ah2 (t : ℝ) : Vec (2 * (1280 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 1280 7 7 1 (kv 1280 1) (kv 1280 160) (Zh2 t)

-- ════════════════════════════════════════════════════════════════
-- § 10. The carrier, stage by stage
--   ⭐ A centre-tap conv collapses `δ` to `s · δ 0` at every output channel; a centre-tap
--   DEPTHWISE cannot broadcast, so it scales `δ` channel by channel. Only `δ 0` is ever read.
-- ════════════════════════════════════════════════════════════════
/-- one carrier BatchNorm's whole contribution: `γ · istd` at `γ = 1`. -/
noncomputable def rf (n : Nat) (z : Vec n) : ℝ := bnIstd n z 1

theorem rf_pos (n : Nat) (z : Vec n) : 0 < rf n z := bnIstd_pos _ 1 one_pos

theorem rf_cont (n : Nat) (k : Fin n) : Continuous (fun z : Vec n => rf n z) :=
  bnIstd_cont 1 one_pos k

theorem rfac_cont (oc h w : Nat) (k : Fin (2 * (h * w))) (c : Fin oc)
    (Z : ℝ → Vec (2 * (oc * h * w))) (hZ : Continuous Z) :
    Continuous (fun t => rf (2 * (h * w)) (bnRowLA 2 oc h w (Z t) c)) :=
  (rf_cont _ k).comp ((bnRowLA_continuous 2 oc h w c).comp hZ)

/-- the carrier at the swish's output — ⭐⭐ the one step of the whole chain that is NOT a
    multiple of the step before it. -/
noncomputable def dSw (t : ℝ) : Fin 128 → ℝ := fun o => swishGap 160 (uF t o)

noncomputable def d1p (t : ℝ) : Fin 48 → ℝ :=
  fun o => dSw t 0 * rf (2 * (56 * 56)) (bnRowLA 2 48 56 56 (Z1p t) o)

noncomputable def daq (t : ℝ) : Fin 48 → ℝ :=
  fun ch => d1p t ch * rf (2 * (28 * 28)) (bnRowLA 2 48 28 28 (Zaq t) ch)

noncomputable def dae (t : ℝ) : Fin 192 → ℝ :=
  fun o => daq t 0 * rf (2 * (28 * 28)) (bnRowLA 2 192 28 28 (Zae t) o)

noncomputable def dad (t : ℝ) : Fin 192 → ℝ :=
  fun ch => dae t ch * rf (2 * (28 * 28)) (bnRowLA 2 192 28 28 (Zad t) ch)

noncomputable def daz (t : ℝ) : Fin 80 → ℝ :=
  fun o => dad t 0 * rf (2 * (28 * 28)) (bnRowLA 2 80 28 28 (Zaz t) o)

noncomputable def dbq (t : ℝ) : Fin 80 → ℝ :=
  fun ch => daz t ch * rf (2 * (14 * 14)) (bnRowLA 2 80 14 14 (Zbq t) ch)

noncomputable def dbe (t : ℝ) : Fin 480 → ℝ :=
  fun o => dbq t 0 * rf (2 * (14 * 14)) (bnRowLA 2 480 14 14 (Zbe t) o)

noncomputable def dbd (t : ℝ) : Fin 480 → ℝ :=
  fun ch => dbe t ch * rf (2 * (14 * 14)) (bnRowLA 2 480 14 14 (Zbd t) ch)

noncomputable def dbz (t : ℝ) : Fin 160 → ℝ :=
  fun o => dbd t 0 * rf (2 * (14 * 14)) (bnRowLA 2 160 14 14 (Zbz t) o)

noncomputable def dcq (t : ℝ) : Fin 160 → ℝ :=
  fun ch => dbz t ch * rf (2 * (7 * 7)) (bnRowLA 2 160 7 7 (Zcq t) ch)

noncomputable def dce (t : ℝ) : Fin 960 → ℝ :=
  fun o => dcq t 0 * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zce t) o)

noncomputable def dcd (t : ℝ) : Fin 960 → ℝ :=
  fun ch => dce t ch * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zcd t) ch)

noncomputable def dcz (t : ℝ) : Fin 256 → ℝ :=
  fun o => dcd t 0 * rf (2 * (7 * 7)) (bnRowLA 2 256 7 7 (Zcz t) o)

noncomputable def dh1 (t : ℝ) : Fin 960 → ℝ :=
  fun o => dcz t 0 * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zh1 t) o)

noncomputable def dh2 (t : ℝ) : Fin 1280 → ℝ :=
  fun o => dh1 t 0 * rf (2 * (7 * 7)) (bnRowLA 2 1280 7 7 (Zh2 t) o)

theorem ed1p (t : ℝ) : EDiff (d1p t) (A1p t) := by
  refine EDiff_bn 48 56 56 1 (kv 48 1) (kv 48 0) (fun _ => 1 * dSw t 0) (d1p t) (Z1p t) ?_ ?_
  · exact EDiff_conv (h := 56) (w := 56) (0 : Fin 128) rfl (by norm_num) (by norm_num) 1
      (kv 48 0) (dSw t) _ (Sw t) (ed_Sw t) (fun o => rfl)
  · intro ci
    simp only [d1p, rf, kv_apply]
    ring

theorem edaq (t : ℝ) : EDiff (daq t) (Aaq t) := by
  refine EDiff_bn 48 28 28 1 (kv 48 1) (kv 48 160) (fun ch => 1 * d1p t ch) (daq t) (Zaq t) ?_ ?_
  · exact EDiff_dwS2 (h := 28) (w := 28) (by norm_num) (by norm_num) 1
      (kv 48 0) (d1p t) _ (A1p t) (ed1p t) (fun ch => rfl)
  · intro ci
    simp only [daq, rf, kv_apply]
    ring

theorem edae (t : ℝ) : EDiff (dae t) (Aae t) := by
  refine EDiff_bn 192 28 28 1 (kv 192 1) (kv 192 160) (fun _ => 1 * daq t 0) (dae t) (Zae t) ?_ ?_
  · exact EDiff_conv (h := 28) (w := 28) (0 : Fin 48) rfl (by norm_num) (by norm_num) 1
      (kv 192 0) (daq t) _ (Aaq t) (edaq t) (fun o => rfl)
  · intro ci
    simp only [dae, rf, kv_apply]
    ring

theorem edad (t : ℝ) : EDiff (dad t) (Aad t) := by
  refine EDiff_bn 192 28 28 1 (kv 192 1) (kv 192 160) (fun ch => 1 * dae t ch) (dad t) (Zad t) ?_ ?_
  · exact EDiff_dw (h := 28) (w := 28) (by norm_num) (by norm_num) 1
      (kv 192 0) (dae t) _ (Aae t) (edae t) (fun ch => rfl)
  · intro ci
    simp only [dad, rf, kv_apply]
    ring

theorem edaz (t : ℝ) : EDiff (daz t) (Aaz t) := by
  refine EDiff_bn 80 28 28 1 (kv 80 1) (kv 80 0) (fun _ => 1 * dad t 0) (daz t) (Zaz t) ?_ ?_
  · exact EDiff_conv (h := 28) (w := 28) (0 : Fin 192) rfl (by norm_num) (by norm_num) 1
      (kv 80 0) (dad t) _ (Aad t) (edad t) (fun o => rfl)
  · intro ci
    simp only [daz, rf, kv_apply]
    ring

theorem edbq (t : ℝ) : EDiff (dbq t) (Abq t) := by
  refine EDiff_bn 80 14 14 1 (kv 80 1) (kv 80 160) (fun ch => 1 * daz t ch) (dbq t) (Zbq t) ?_ ?_
  · exact EDiff_dwS2 (h := 14) (w := 14) (by norm_num) (by norm_num) 1
      (kv 80 0) (daz t) _ (Aaz t) (edaz t) (fun ch => rfl)
  · intro ci
    simp only [dbq, rf, kv_apply]
    ring

theorem edbe (t : ℝ) : EDiff (dbe t) (Abe t) := by
  refine EDiff_bn 480 14 14 1 (kv 480 1) (kv 480 160) (fun _ => 1 * dbq t 0) (dbe t) (Zbe t) ?_ ?_
  · exact EDiff_conv (h := 14) (w := 14) (0 : Fin 80) rfl (by norm_num) (by norm_num) 1
      (kv 480 0) (dbq t) _ (Abq t) (edbq t) (fun o => rfl)
  · intro ci
    simp only [dbe, rf, kv_apply]
    ring

theorem edbd (t : ℝ) : EDiff (dbd t) (Abd t) := by
  refine EDiff_bn 480 14 14 1 (kv 480 1) (kv 480 160) (fun ch => 1 * dbe t ch) (dbd t) (Zbd t) ?_ ?_
  · exact EDiff_dw (h := 14) (w := 14) (by norm_num) (by norm_num) 1
      (kv 480 0) (dbe t) _ (Abe t) (edbe t) (fun ch => rfl)
  · intro ci
    simp only [dbd, rf, kv_apply]
    ring

theorem edbz (t : ℝ) : EDiff (dbz t) (Abz t) := by
  refine EDiff_bn 160 14 14 1 (kv 160 1) (kv 160 0) (fun _ => 1 * dbd t 0) (dbz t) (Zbz t) ?_ ?_
  · exact EDiff_conv (h := 14) (w := 14) (0 : Fin 480) rfl (by norm_num) (by norm_num) 1
      (kv 160 0) (dbd t) _ (Abd t) (edbd t) (fun o => rfl)
  · intro ci
    simp only [dbz, rf, kv_apply]
    ring

theorem edcq (t : ℝ) : EDiff (dcq t) (Acq t) := by
  refine EDiff_bn 160 7 7 1 (kv 160 1) (kv 160 160) (fun ch => 1 * dbz t ch) (dcq t) (Zcq t) ?_ ?_
  · exact EDiff_dwS2 (h := 7) (w := 7) (by norm_num) (by norm_num) 1
      (kv 160 0) (dbz t) _ (Abz t) (edbz t) (fun ch => rfl)
  · intro ci
    simp only [dcq, rf, kv_apply]
    ring

theorem edce (t : ℝ) : EDiff (dce t) (Ace t) := by
  refine EDiff_bn 960 7 7 1 (kv 960 1) (kv 960 160) (fun _ => 1 * dcq t 0) (dce t) (Zce t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 160) rfl (by norm_num) (by norm_num) 1
      (kv 960 0) (dcq t) _ (Acq t) (edcq t) (fun o => rfl)
  · intro ci
    simp only [dce, rf, kv_apply]
    ring

theorem edcd (t : ℝ) : EDiff (dcd t) (Acd t) := by
  refine EDiff_bn 960 7 7 1 (kv 960 1) (kv 960 160) (fun ch => 1 * dce t ch) (dcd t) (Zcd t) ?_ ?_
  · exact EDiff_dw (h := 7) (w := 7) (by norm_num) (by norm_num) 1
      (kv 960 0) (dce t) _ (Ace t) (edce t) (fun ch => rfl)
  · intro ci
    simp only [dcd, rf, kv_apply]
    ring

theorem edcz (t : ℝ) : EDiff (dcz t) (Acz t) := by
  refine EDiff_bn 256 7 7 1 (kv 256 1) (kv 256 0) (fun _ => 1 * dcd t 0) (dcz t) (Zcz t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 960) rfl (by norm_num) (by norm_num) 1
      (kv 256 0) (dcd t) _ (Acd t) (edcd t) (fun o => rfl)
  · intro ci
    simp only [dcz, rf, kv_apply]
    ring

theorem edh1 (t : ℝ) : EDiff (dh1 t) (Ah1 t) := by
  refine EDiff_bn 960 7 7 1 (kv 960 1) (kv 960 160) (fun _ => 1 * dcz t 0) (dh1 t) (Zh1 t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 256) rfl (by norm_num) (by norm_num) 1
      (kv 960 0) (dcz t) _ (Acz t) (edcz t) (fun o => rfl)
  · intro ci
    simp only [dh1, rf, kv_apply]
    ring

theorem edh2 (t : ℝ) : EDiff (dh2 t) (Ah2 t) := by
  refine EDiff_bn 1280 7 7 1 (kv 1280 1) (kv 1280 160) (fun _ => 1 * dh1 t 0) (dh2 t) (Zh2 t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 960) rfl (by norm_num) (by norm_num) 1
      (kv 1280 0) (dh1 t) _ (Ah1 t) (edh1 t) (fun o => rfl)
  · intro ci
    simp only [dh2, rf, kv_apply]
    ring

-- ════════════════════════════════════════════════════════════════
-- § 11. Continuity along the ray — what `Rr` needs and nothing more
-- ════════════════════════════════════════════════════════════════
theorem Zs_continuous : Continuous Zs :=
  (batchMap_continuous _ (flatConvStride2Xla_differentiable _ _).continuous).comp sealX_continuous

theorem As_continuous : Continuous As :=
  (bnBatchLA_differentiable 2 32 112 112 1 one_pos _ _).continuous.comp Zs_continuous

theorem Zf_continuous : Continuous Zf :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp As_continuous

theorem Af_continuous : Continuous Af :=
  (bnBatchLA_differentiable 2 128 56 56 1 one_pos _ _).continuous.comp Zf_continuous

theorem Sw_continuous : Continuous Sw := (swish_diff _).continuous.comp Af_continuous

theorem Z1p_continuous : Continuous Z1p :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Sw_continuous

theorem A1p_continuous : Continuous A1p :=
  (bnBatchLA_differentiable 2 48 56 56 1 one_pos _ _).continuous.comp Z1p_continuous

theorem Zaq_continuous : Continuous Zaq :=
  (batchMap_continuous _ (depthwiseStride2Flat_differentiable _ _).continuous).comp A1p_continuous

theorem Aaq_continuous : Continuous Aaq :=
  (bnBatchLA_differentiable 2 48 28 28 1 one_pos _ _).continuous.comp Zaq_continuous

theorem Zae_continuous : Continuous Zae :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Aaq_continuous

theorem Aae_continuous : Continuous Aae :=
  (bnBatchLA_differentiable 2 192 28 28 1 one_pos _ _).continuous.comp Zae_continuous

theorem Zad_continuous : Continuous Zad :=
  (batchMap_continuous _ (depthwiseFlat_differentiable _ _).continuous).comp Aae_continuous

theorem Aad_continuous : Continuous Aad :=
  (bnBatchLA_differentiable 2 192 28 28 1 one_pos _ _).continuous.comp Zad_continuous

theorem Zaz_continuous : Continuous Zaz :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Aad_continuous

theorem Aaz_continuous : Continuous Aaz :=
  (bnBatchLA_differentiable 2 80 28 28 1 one_pos _ _).continuous.comp Zaz_continuous

theorem Zbq_continuous : Continuous Zbq :=
  (batchMap_continuous _ (depthwiseStride2Flat_differentiable _ _).continuous).comp Aaz_continuous

theorem Abq_continuous : Continuous Abq :=
  (bnBatchLA_differentiable 2 80 14 14 1 one_pos _ _).continuous.comp Zbq_continuous

theorem Zbe_continuous : Continuous Zbe :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Abq_continuous

theorem Abe_continuous : Continuous Abe :=
  (bnBatchLA_differentiable 2 480 14 14 1 one_pos _ _).continuous.comp Zbe_continuous

theorem Zbd_continuous : Continuous Zbd :=
  (batchMap_continuous _ (depthwiseFlat_differentiable _ _).continuous).comp Abe_continuous

theorem Abd_continuous : Continuous Abd :=
  (bnBatchLA_differentiable 2 480 14 14 1 one_pos _ _).continuous.comp Zbd_continuous

theorem Zbz_continuous : Continuous Zbz :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Abd_continuous

theorem Abz_continuous : Continuous Abz :=
  (bnBatchLA_differentiable 2 160 14 14 1 one_pos _ _).continuous.comp Zbz_continuous

theorem Zcq_continuous : Continuous Zcq :=
  (batchMap_continuous _ (depthwiseStride2Flat_differentiable _ _).continuous).comp Abz_continuous

theorem Acq_continuous : Continuous Acq :=
  (bnBatchLA_differentiable 2 160 7 7 1 one_pos _ _).continuous.comp Zcq_continuous

theorem Zce_continuous : Continuous Zce :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Acq_continuous

theorem Ace_continuous : Continuous Ace :=
  (bnBatchLA_differentiable 2 960 7 7 1 one_pos _ _).continuous.comp Zce_continuous

theorem Zcd_continuous : Continuous Zcd :=
  (batchMap_continuous _ (depthwiseFlat_differentiable _ _).continuous).comp Ace_continuous

theorem Acd_continuous : Continuous Acd :=
  (bnBatchLA_differentiable 2 960 7 7 1 one_pos _ _).continuous.comp Zcd_continuous

theorem Zcz_continuous : Continuous Zcz :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Acd_continuous

theorem Acz_continuous : Continuous Acz :=
  (bnBatchLA_differentiable 2 256 7 7 1 one_pos _ _).continuous.comp Zcz_continuous

theorem Zh1_continuous : Continuous Zh1 :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Acz_continuous

theorem Ah1_continuous : Continuous Ah1 :=
  (bnBatchLA_differentiable 2 960 7 7 1 one_pos _ _).continuous.comp Zh1_continuous

theorem Zh2_continuous : Continuous Zh2 :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp Ah1_continuous

theorem Ah2_continuous : Continuous Ah2 :=
  (bnBatchLA_differentiable 2 1280 7 7 1 one_pos _ _).continuous.comp Zh2_continuous

-- ════════════════════════════════════════════════════════════════
-- § 12. `Rr` — the fifteen post-swish BatchNorm factors
--   ⭐ Seventeen BatchNorms sit on MobileNetV4's carrier: the stem's, the fused stage's two, four
--   in each of rows 1, 3 and 11 — the only rows that change channels and so the only ones without
--   a skip — and the head's two. Two of them are inside `uF`, before the swish; these are the
--   other fifteen. ⚠ No BatchNorm VARIANCE derivative is ever taken: `Rr` enters only as a
--   continuous factor.
-- ════════════════════════════════════════════════════════════════
noncomputable def Rr (t : ℝ) : ℝ :=
  rf (2 * (56 * 56)) (bnRowLA 2 48 56 56 (Z1p t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 48 28 28 (Zaq t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 192 28 28 (Zae t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 192 28 28 (Zad t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 80 28 28 (Zaz t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 80 14 14 (Zbq t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 480 14 14 (Zbe t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 480 14 14 (Zbd t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 160 14 14 (Zbz t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 160 7 7 (Zcq t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zce t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zcd t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 256 7 7 (Zcz t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Zh1 t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 1280 7 7 (Zh2 t) 0)))))))))))))))

theorem Rr_pos (t : ℝ) : 0 < Rr t := by
  unfold Rr
  -- ⚠ one explicit factor per carrier BatchNorm, not `repeat' apply mul_pos`: `rf` would be
  -- split inside and leave goals `rf_pos` cannot close.
  exact mul_pos (rf_pos _ _)
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
    (rf_pos _ _))))))))))))))

theorem Rr_continuous : Continuous Rr := by
  unfold Rr
  exact (rfac_cont 48 56 56 ⟨0, by norm_num⟩ 0 _ Z1p_continuous).mul
    ((rfac_cont 48 28 28 ⟨0, by norm_num⟩ 0 _ Zaq_continuous).mul
    ((rfac_cont 192 28 28 ⟨0, by norm_num⟩ 0 _ Zae_continuous).mul
    ((rfac_cont 192 28 28 ⟨0, by norm_num⟩ 0 _ Zad_continuous).mul
    ((rfac_cont 80 28 28 ⟨0, by norm_num⟩ 0 _ Zaz_continuous).mul
    ((rfac_cont 80 14 14 ⟨0, by norm_num⟩ 0 _ Zbq_continuous).mul
    ((rfac_cont 480 14 14 ⟨0, by norm_num⟩ 0 _ Zbe_continuous).mul
    ((rfac_cont 480 14 14 ⟨0, by norm_num⟩ 0 _ Zbd_continuous).mul
    ((rfac_cont 160 14 14 ⟨0, by norm_num⟩ 0 _ Zbz_continuous).mul
    ((rfac_cont 160 7 7 ⟨0, by norm_num⟩ 0 _ Zcq_continuous).mul
    ((rfac_cont 960 7 7 ⟨0, by norm_num⟩ 0 _ Zce_continuous).mul
    ((rfac_cont 960 7 7 ⟨0, by norm_num⟩ 0 _ Zcd_continuous).mul
    ((rfac_cont 256 7 7 ⟨0, by norm_num⟩ 0 _ Zcz_continuous).mul
    ((rfac_cont 960 7 7 ⟨0, by norm_num⟩ 0 _ Zh1_continuous).mul
    (rfac_cont 1280 7 7 ⟨0, by norm_num⟩ 0 _ Zh2_continuous))))))))))))))



-- ════════════════════════════════════════════════════════════════
-- § 13. The collapsed trunk — which activation each prefix IS
--   ⭐ The eighteen skipped rows are the EXACT identity (`sealZBody_eq` + `resid_id`), not
--   ResNet's `a ↦ a + 1`: with the project BatchNorm's `β = 0` a zeroed body is the constant `0`.
--   So two of the seven groups collapse to nothing at all and three to one row each.
--   ⚠ Each step is restated at `(sealW nCls).b_k` before rewriting — `rw` does not see through a
--   structure projection, and `sealZ mnv4Row_k` is only DEFEQ to it.
-- ════════════════════════════════════════════════════════════════
theorem pc1 (nCls : Nat) (t : ℝ) : mnv4Pre1 2 (sealW nCls) (sealX t) = A1p t := by
  show (mnv4FusedStack 2 (sealW nCls)).fwd (mnv4Pre0 2 (sealW nCls) (sealX t)) = _
  rw [pc0]
  simp only [mnv4FusedStack, mnv4FusedStage, CertLayer.comp_fwd_apply]
  rfl

theorem pc2 (nCls : Nat) (t : ℝ) : mnv4Pre2 2 (sealW nCls) (sealX t) = Aaz t := by
  have hs : (mnv4PreStridedBodyOfRow 2 mnv4Row1 (sealW nCls).b1).fwd (A1p t) = Aaz t :=
    sealCTStrided_eq 2 mnv4Row1 (by norm_num) (by norm_num) (A1p t)
  have e2 : (CertLayer.residual (mnv4BodyOfRow 2 mnv4Row2 (sealW nCls).b2)).fwd (Aaz t)
      = Aaz t := resid_id _ _ (sealZBody_eq 2 mnv4Row2 (by norm_num) _)
  show (mnv4Res28Layer 2 (sealW nCls)).fwd (mnv4Pre1 2 (sealW nCls) (sealX t)) = _
  rw [mnv4Res28Layer_fwd_apply, pc1, hs, e2]

theorem pc3 (nCls : Nat) (t : ℝ) : mnv4Pre3 2 (sealW nCls) (sealX t) = Abz t := by
  have hs : (mnv4PreStridedBodyOfRow 2 mnv4Row3 (sealW nCls).b3).fwd (Aaz t) = Abz t :=
    sealCTStrided_eq 2 mnv4Row3 (by norm_num) (by norm_num) (Aaz t)
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
  have hs : (mnv4PreStridedBodyOfRow 2 mnv4Row11 (sealW nCls).b11).fwd (Abz t) = Acz t :=
    sealCTStrided_eq 2 mnv4Row11 (by norm_num) (by norm_num) (Abz t)
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
-- § 14. The head reads the carrier off channel 0
-- ════════════════════════════════════════════════════════════════
theorem sealW_Wd (nCls : Nat) :
    (sealW nCls).Wd = fun (i : Fin 1280) (j : Fin nCls) =>
      if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0 := rfl

/-- the head, peeled: two 1×1 conv-bn-relus, then GAP and the classifier. ⚠ Proved at variables
    (`CertLayer.comp_fwd_apply`) and applied at the net's literals — peeling a `CertLayer.comp`
    here is a kernel timeout, which is why `MobileNetV4FullB.lean` keeps the group peels generic
    too. -/
theorem headStack_apply (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (v : Vec (N * (256 * 7 * 7))) :
    (mnv4HeadStack N w).fwd v
      = StableHLO.batchMap N (dense w.Wd w.bd)
          (StableHLO.batchMap N (globalAvgPoolFlat 1280 7 7)
            (StableHLO.cbReluB N (h := 7) (w := 7) w.hW w.hb w.hE w.hg w.hbt
              (StableHLO.cbReluB N (h := 7) (w := 7) w.h1W w.h1b w.h1E w.h1g w.h1bt v))) := by
  simp only [mnv4HeadStack, mnv4Head, CertLayer.comp_fwd_apply, cbReluLayer_fwd_apply,
    mnv4GapLayer_fwd_apply, mnv4DenseLayer_fwd_apply]

theorem headA (nCls : Nat) (t : ℝ) :
    mobilenetv4ForwardB_full 2 (sealW nCls) (sealX t)
      = StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
          (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7) (Ah2 t)) := by
  show (mnv4HeadStack 2 (sealW nCls)).fwd (mnv4Pre6 2 (sealW nCls) (sealX t)) = _
  rw [headStack_apply, pc6]
  show StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
      (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7)
        (StableHLO.cbReluB 2 (h := 7) (w := 7) (ctK 1280 960 1 1 1) (kv 1280 0) 1 (kv 1280 1)
            (kv 1280 160)
          (StableHLO.cbReluB 2 (h := 7) (w := 7) (ctK 960 256 1 1 1) (kv 960 0) 1 (kv 960 1)
            (kv 960 160) (Acz t)))) = _
  rw [cbReluB_eq (ctK 960 256 1 1 1) (kv 960 0) (by norm_num) (Acz t),
    cbReluB_eq (ctK 1280 960 1 1 1) (kv 1280 0) (by norm_num) _]
  rfl

theorem head_diff (nCls : Nat) (hn : 0 < nCls) (v : Vec (2 * (1280 * 7 * 7)))
    (δ : Fin 1280 → ℝ) (hv : EDiff δ v) :
    StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
        (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7) v)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
        (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7) v)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = δ 0 :=
  head_diff_ct (by norm_num) (by norm_num) (0 : Fin 1280) rfl ⟨0, hn⟩ _ _
    (fun ci => by rw [sealW_Wd]; simp) rfl v δ hv

/-- ⭐⭐ **The class-0 difference between the two examples, along the ray**: the swish's gap times
    the fifteen BatchNorm factors below it. ⚠ NOT `t · Rr t` — the swish is not affine, and that
    is exactly what `swishGap` carries. -/
theorem gd_ray (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    mobilenetv4ForwardB_full 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - mobilenetv4ForwardB_full 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = swishGap 160 (uF t 0) * Rr t := by
  rw [headA, head_diff nCls hn _ (dh2 t) (edh2 t)]
  simp only [dh2, dh1, dcz, dcd, dce, dcq, dbz, dbd, dbe, dbq, daz, dad, dae, daq, d1p, dSw, Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 15. The derivative along the ray, and the seal
--   ⭐ `g t = swishGap 160 (uF t 0) · Rr t`, with `uF t 0 = t · Q0 t`. Three factors, each
--   nonzero: the swish's slope at `β`, the two pre-swish BatchNorm factors, and the fifteen
--   after it. ⚠ No BatchNorm variance derivative is taken — `Q0` and `Rr` enter only as
--   continuous factors, and the swish's slope is the one honest derivative in the chain.
-- ════════════════════════════════════════════════════════════════
/-- the two pre-swish BatchNorm factors, as `uF t 0 = t · Q0 t`. -/
noncomputable def Q0 (t : ℝ) : ℝ := iS t 0 * iF t 0 / 2

theorem uF_eq (t : ℝ) : uF t 0 = t * Q0 t := by
  simp only [uF, Q0]
  ring

theorem Q0_pos (t : ℝ) : 0 < Q0 t := by
  have h1 : 0 < iS t 0 := bnIstd_pos _ 1 one_pos
  have h2 : 0 < iF t 0 := bnIstd_pos _ 1 one_pos
  simp only [Q0]
  exact div_pos (mul_pos h1 h2) two_pos

theorem Q0_continuous : Continuous Q0 := by
  have h1 : Continuous fun t => iS t 0 :=
    rfac_cont 32 112 112 ⟨0, by norm_num⟩ 0 _ Zs_continuous
  have h2 : Continuous fun t => iF t 0 :=
    rfac_cont 128 56 56 ⟨0, by norm_num⟩ 0 _ Zf_continuous
  exact (h1.mul h2).div_const 2

theorem hasDerivAt_uF : HasDerivAt (fun t : ℝ => uF t 0) (Q0 0) 0 := by
  have h : (fun t : ℝ => uF t 0) = fun t : ℝ => t * Q0 t := funext uF_eq
  rw [h]
  exact hasDerivAt_mul_self_zero Q0_continuous.continuousAt

theorem uF_zero : uF 0 0 = 0 := by simp [uF]

/-- ⭐⭐ the swish's contribution: `swishGap` is differentiable at `0` with slope `2 · swish' β`,
    and that is the whole reason this net's readout is not `t · Rr t`. -/
theorem hasDerivAt_gsw :
    HasDerivAt (fun t : ℝ => swishGap 160 (uF t 0)) (2 * swishD 160 * Q0 0) 0 := by
  have hg : HasDerivAt (swishGap 160) (2 * swishD 160) (uF 0 0) := by
    rw [uF_zero]; exact hasDerivAt_swishGap 160
  have h2 := HasDerivAt.comp (0 : ℝ) hg hasDerivAt_uF
  exact h2

theorem gd_zero : swishGap 160 (uF 0 0) = 0 := by rw [uF_zero, swishGap_zero]

theorem hasDerivAt_gd :
    HasDerivAt (fun t : ℝ => swishGap 160 (uF t 0) * Rr t)
      (2 * swishD 160 * Q0 0 * Rr 0) 0 :=
  hasDerivAt_mul_of_zero hasDerivAt_gsw gd_zero Rr_continuous.continuousAt

theorem gd_slope_ne : 2 * swishD 160 * Q0 0 * Rr 0 ≠ 0 := by
  have h1 : 0 < swishD 160 := swishD_pos (by norm_num)
  have h2 := Q0_pos 0
  have h3 := Rr_pos 0
  have : 0 < 2 * swishD 160 * Q0 0 * Rr 0 :=
    mul_pos (mul_pos (mul_pos two_pos h1) h2) h3
  exact this.ne'

/-- at `t = 1` the readout is strictly positive: the half-gap is inside `(0, 160]` because
    `ε = 1` caps every `istd` at `1`, and the swish is strictly increasing there. -/
theorem gd_one_pos : 0 < swishGap 160 (uF 1 0) * Rr 1 := by
  have h1 : 0 < iS 1 0 := bnIstd_pos _ 1 one_pos
  have h2 : 0 < iF 1 0 := bnIstd_pos _ 1 one_pos
  have h3 : iS 1 0 ≤ 1 := bnIstd_le_one _
  have h4 : iF 1 0 ≤ 1 := bnIstd_le_one _
  have hu : 0 < uF 1 0 := by
    simp only [uF]
    exact mul_pos (mul_pos (by norm_num) h1) h2
  have hub : uF 1 0 ≤ 160 := by
    have : uF 1 0 = 1 / 2 * iS 1 0 * iF 1 0 := by simp [uF]
    nlinarith
  exact mul_pos (swishGap_pos hu hub) (Rr_pos 1)

theorem sealDiffAt (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (mobilenetv4ForwardB_full 2 (sealW nCls)) (sealX t) := by
  have d0 : DifferentiableAt ℝ (mnv4Pre0 2 (sealW nCls)) (sealX t) :=
    mnv4StemB_differentiableAt 2 112 112 _ _ _ one_pos _ _ (sealX t) (scStem nCls t)
  have d1 : DifferentiableAt ℝ (mnv4Pre1 2 (sealW nCls)) (sealX t) :=
    ((mnv4FusedStack 2 (sealW nCls)).diff _ (scFused nCls t)).comp (sealX t) d0
  have d2 : DifferentiableAt ℝ (mnv4Pre2 2 (sealW nCls)) (sealX t) :=
    ((mnv4Res28Layer 2 (sealW nCls)).diff _ (sc28 nCls t)).comp (sealX t) d1
  have d3 : DifferentiableAt ℝ (mnv4Pre3 2 (sealW nCls)) (sealX t) :=
    ((mnv4Res14aLayer 2 (sealW nCls)).diff _ (sc14a nCls t)).comp (sealX t) d2
  have d4 : DifferentiableAt ℝ (mnv4Pre4 2 (sealW nCls)) (sealX t) :=
    ((mnv4Res14bLayer 2 (sealW nCls)).diff _ (sc14b nCls t)).comp (sealX t) d3
  have d5 : DifferentiableAt ℝ (mnv4Pre5 2 (sealW nCls)) (sealX t) :=
    ((mnv4Res7aLayer 2 (sealW nCls)).diff _ (sc7a nCls t)).comp (sealX t) d4
  have d6 : DifferentiableAt ℝ (mnv4Pre6 2 (sealW nCls)) (sealX t) :=
    ((mnv4Res7bLayer 2 (sealW nCls)).diff _ (sc7b nCls t)).comp (sealX t) d5
  exact ((mnv4HeadStack 2 (sealW nCls)).diff _ (scHead nCls t)).comp (sealX t) d6

/-- ⭐⭐ **Level 2 — the witness is non-degenerate**: the full-width batch-BN MobileNetV4-Conv-M at
    the structural weights is NOT constant in its input. -/
theorem sealX_nonconstant (nCls : Nat) (hn : 0 < nCls) :
    mobilenetv4ForwardB_full 2 (sealW nCls) (sealX 1)
      ≠ mobilenetv4ForwardB_full 2 (sealW nCls) (sealX 0) := by
  intro heq
  have h1 := gd_ray nCls hn 1
  have h0 := gd_ray nCls hn 0
  rw [heq] at h1
  have hz : swishGap 160 (uF 1 0) * Rr 1 = swishGap 160 (uF 0 0) * Rr 0 := by rw [← h1, ← h0]
  rw [gd_zero, zero_mul] at hz
  linarith [gd_one_pos]

/-- ⭐⭐ **Level 3 — the whole-net Jacobian is nonzero at the witness.** -/
theorem sealX_jacobian_nonzero (nCls : Nat) (hn : 0 < nCls) :
    fderiv ℝ (mobilenetv4ForwardB_full 2 (sealW nCls)) (sealX 0) ≠ 0 := by
  refine fderiv_ne_zero_of_ray sealV (sealDiffAt nCls 0)
    (fun y => y (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - y (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))) (by fun_prop)
    gd_slope_ne ?_
  have heq : (fun t : ℝ => mobilenetv4ForwardB_full 2 (sealW nCls) (sealX 0 + t • sealV)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - mobilenetv4ForwardB_full 2 (sealW nCls) (sealX 0 + t • sealV)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls))))
      = fun t : ℝ => swishGap 160 (uF t 0) * Rr t := by
    funext t
    rw [sealX_zero_add]
    exact gd_ray nCls hn t
  rw [heq]
  exact hasDerivAt_gd

/-- ⭐⭐ **The seal**: the proven whole-network backward of the full-width, batch-BatchNorm,
    21-block, 224×224 MobileNetV4-Conv-M — `mobilenetv4ForwardB_full`, the forward every
    MobileNetV4 artifact runs — is **not the zero map** at the witness. -/
theorem sealX_backward_nontrivial (nCls : Nat) (hn : 0 < nCls) :
    ∃ (j₀ : Fin (2 * nCls)) (i₀ : Fin (2 * (3 * (2 * 112) * (2 * 112)))),
      (sealVJP nCls 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (sealVJP nCls 0).backward_nontrivial_of_fderiv_ne (sealX_jacobian_nonzero nCls hn)

end Mnv4FullBSeal
end Proofs
