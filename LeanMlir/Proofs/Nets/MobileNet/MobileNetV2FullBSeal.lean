import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBSeal

/-!
# MobileNetV2's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`planning/full_width_seals.md` §4.3. `MobileNetV2FullBVJP.lean` proves
`mobilenetv2ForwardB_full_has_vjp_at`: the whole-net VJP at any `(w, x)` satisfying **19 clause
bundles** — the stem's relu6, one per bottleneck, and the head's — covering 35 relu6 sites, each a
two-sided window `≠ 0 ∧ ≠ 6`. A conditional theorem of that shape says nothing unless its
hypotheses are jointly satisfiable at a point with a nonzero Jacobian, and until now that was
exhibited only on a per-example, two-block, 2-channel proxy, deleted when this file landed.
This file exhibits it on `mobilenetv2ForwardB_full` itself: the seventeen bottlenecks of the
`[t,c,n,s]` table, 32→1280 channels, XLA-`SAME` stride-2 padding, **batch** BatchNorm, at 224×224.

## The witness

Weights are *structural*, not trained (`planning/full_width_seals.md` §7):

* `ε = 1` and `γ = 1/64` at all 52 BatchNorms; `β = 3` wherever a relu6 follows, `β = 0` at the
  eleven linear-bottleneck projections, which no activation follows;
* the ten **residual** bottlenecks have all three kernels zeroed. A zero kernel gives a constant
  channel, batch BN of a constant channel is `β`, and the block's last `β` is `0` — so a residual
  block is the **exact identity** (`sealResB_eq`), simpler than ResNet's `a ↦ a + 1`;
* the seven **channel-changing** blocks (`b1`, `b2`, `b4`, `b7`, `b11`, `b14`, `b17`), the stem and
  the head carry the signal: every 1×1 expand and project is a centre-tap broadcast
  (`BatchSeal.ctK`), every 3×3 depthwise a centre-tap identity (`BatchSeal.ctDW`);
* the head reads channel 0 into class 0, so `0 < nCls` is the only constraint on the class count.

`N = 2`, and the input is the shared ray `sealX t = rayX … t`.

## ⭐⭐ Every one of the 35 clauses is weight-only

Better than ResNet-34, whose post-residual relu still needed `0 ≤ activation`. Two facts compose:
every relu6 in this net sits **directly on a BatchNorm output**, and `BatchSeal.bnBatchLA_window`
bounds a BN output inside `(0, 6)` at *every* input once `|γ|·√(N·h·w) < β = 3`; and the linear
bottleneck has **no relu after the residual add**, so there is no post-residual clause at all. The
margin holds at every site because the widest is `2·112² = 25 088` and `√25 088 / 64 < 2.48`
(`BatchSeal.margin192`). Consequence: no nonnegativity layer, no positional injectivity, and — with
no max-pool anywhere in this net — no no-tie argument.

## ⭐ The carrier threads twenty-two BatchNorms

MobileNetV2's channel-changing blocks have **no skip** — the body *is* the block — so unlike
ResNet's carrier, which saw only the projection of each downsample, this one crosses every BN
inside them: `1 (stem) + 2 (b1) + 3 × 6 + 1 (head)`. The ten residual blocks pass it through
untouched. `BatchSeal.EDiff` is the invariant, `BatchSeal.bnBatchLA_exdiff` the step that survives
per-channel batch BN, and ⭐ `BatchSeal.EDiff_dw` is the one genuinely new shape: a depthwise
cannot broadcast, so where a centre-tap conv collapses the carrier to `fun _ => s · δ 0` at every
output channel, a centre-tap depthwise scales the whole function `δ` channel by channel. The
class-0 output difference between the two examples is `t · Rr t` with `Rr` a 22-fold product of
`1/64 · istd`, continuous and positive, so `g'(0) = Rr 0 ≠ 0`.

⚠⚠ Every collapse below is stated at **variable** `N, h, w, ic, mid, oc` and instantiated at the
witness's numerals afterwards, never proved at them (`planning/full_width_seals.md` §3.5).
-/

namespace Proofs
namespace Mnv2FullBSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs BatchSeal R34FullBSeal

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural weights
--   ⭐ One `γ = 1/64` and one `ε = 1` at all 52 BatchNorms. `β = 3` at every site a relu6
--   follows and `β = 0` at the eleven projections, which none follows — so a zeroed residual
--   body is the constant `0` and its block is the exact identity.
-- ════════════════════════════════════════════════════════════════

noncomputable def sealIVW (ic mid oc : Nat) : IVW ic mid oc where
  eW := ctK mid ic 1 1 1
  eb := kv mid 0
  eε := 1
  eγ := kv mid (1 / 64)
  eβ := kv mid 3
  dW := ctDW mid 3 3 1
  db := kv mid 0
  dε := 1
  dγ := kv mid (1 / 64)
  dβ := kv mid 3
  pW := ctK oc mid 1 1 1
  pb := kv oc 0
  pε := 1
  pγ := kv oc (1 / 64)
  pβ := kv oc 0

noncomputable def sealResW (c mid : Nat) : IVW c mid c where
  eW := zk mid c 1 1
  eb := kv mid 0
  eε := 1
  eγ := kv mid (1 / 64)
  eβ := kv mid 3
  dW := dzk mid 3 3
  db := kv mid 0
  dε := 1
  dγ := kv mid (1 / 64)
  dβ := kv mid 3
  pW := zk c mid 1 1
  pb := kv c 0
  pε := 1
  pγ := kv c (1 / 64)
  pβ := kv c 0

noncomputable def sealNoExpW (ic oc : Nat) : IVWNoExp ic oc where
  dW := ctDW ic 3 3 1
  db := kv ic 0
  dε := 1
  dγ := kv ic (1 / 64)
  dβ := kv ic 3
  pW := ctK oc ic 1 1 1
  pb := kv oc 0
  pε := 1
  pγ := kv oc (1 / 64)
  pβ := kv oc 0

noncomputable def sealW (nCls : Nat) : MNV2BWeights nCls where
  sW := ctK 32 3 3 3 1
  sb := kv 32 0
  sε := 1
  sγ := kv 32 (1 / 64)
  sβ := kv 32 3
  b1 := sealNoExpW 32 16
  b2 := sealIVW 16 96 24
  b3 := sealResW 24 144
  b4 := sealIVW 24 144 32
  b5 := sealResW 32 192
  b6 := sealResW 32 192
  b7 := sealIVW 32 192 64
  b8 := sealResW 64 384
  b9 := sealResW 64 384
  b10 := sealResW 64 384
  b11 := sealIVW 64 384 96
  b12 := sealResW 96 576
  b13 := sealResW 96 576
  b14 := sealIVW 96 576 160
  b15 := sealResW 160 960
  b16 := sealResW 160 960
  b17 := sealIVW 160 960 320
  hW := ctK 1280 320 1 1 1
  hb := kv 1280 0
  hε := 1
  hγ := kv 1280 (1 / 64)
  hβ := kv 1280 3
  fcW := fun i j => if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0
  fcb := kv nCls 0

-- ════════════════════════════════════════════════════════════════
-- § 2. The relu6 window — weight-only, at every input
--   `Mg N h w` is this net's margin check. ⭐⭐ `bnBatchLA_window` bounds a BatchNorm output
--   inside `(0, 6)` at EVERY input once it holds, which is why all 35 clauses below are
--   discharged without ever reading the activation.
-- ════════════════════════════════════════════════════════════════
abbrev Mg (N h w : Nat) : Prop :=
  |(1 / 64 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 3

theorem win (N oc h w : Nat) (hm : Mg N h w) (v : Vec (N * (oc * h * w))) :
    ∀ k, 0 < StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 3) v k ∧
         StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 3) v k < 6 :=
  bnBatchLA_window 1 one_pos (kv oc (1 / 64)) (kv oc 3) (1 / 64) 3 (fun _ => rfl) (fun _ => rfl)
    hm (by have := hm; linarith) v

theorem win6 (N oc h w : Nat) (hm : Mg N h w) (v : Vec (N * (oc * h * w))) :
    ∀ k, StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 3) v k ≠ 0 ∧
         StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 3) v k ≠ 6 :=
  fun k => ⟨(win N oc h w hm v k).1.ne', (win N oc h w hm v k).2.ne⟩

-- ════════════════════════════════════════════════════════════════
-- § 3. Stage collapses, at VARIABLE shapes
--   ⚠⚠ Every collapse here and in §4–§5 is proved at variable `N, h, w, ic, mid, oc` and
--   instantiated at the witness's numerals afterwards, never proved at them: instantiating a
--   proved lemma is substitution, while a numeral-shaped defeq kills the kernel
--   (`planning/full_width_seals.md` §3.5).
-- ════════════════════════════════════════════════════════════════
theorem cbrB_eq {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * h * w))) :
    StableHLO.cbrB N (h := h) (w := w) W b 1 (kv oc (1 / 64)) (kv oc 3) x
      = StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 3)
          (StableHLO.batchMap N (flatConv W b) x) :=
  relu6_id_window _ _ (win N oc h w hm _)

theorem dwbrB_eq {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (hm : Mg N h w) (x : Vec (N * (c * h * w))) :
    StableHLO.dwbrB N (h := h) (w := w) W b 1 (kv c (1 / 64)) (kv c 3) x
      = StableHLO.bnBatchLA N c h w 1 (kv c (1 / 64)) (kv c 3)
          (StableHLO.batchMap N (depthwiseFlat W b) x) :=
  relu6_id_window _ _ (win N c h w hm _)

theorem dwbrBstrided_eq {N c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (hm : Mg N h w) (x : Vec (N * (c * (2 * h) * (2 * w)))) :
    StableHLO.dwbrBstrided N (h := h) (w := w) W b 1 (kv c (1 / 64)) (kv c 3) x
      = StableHLO.bnBatchLA N c h w 1 (kv c (1 / 64)) (kv c 3)
          (StableHLO.batchMap N (depthwiseStride2FlatXla W b) x) :=
  relu6_id_window _ _ (win N c h w hm _)

theorem mnv2StemB_eq {N h w ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    mnv2StemB N h w Ws bs 1 (kv oc (1 / 64)) (kv oc 3) x
      = StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 3)
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) :=
  relu6_id_window _ _ (win N oc h w hm _)

-- ════════════════════════════════════════════════════════════════
-- § 4. The residual blocks are the exact identity
-- ════════════════════════════════════════════════════════════════
theorem sealResBody (N h w c mid : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w))) :
    mnv2ExpOnlyB N h w (sealResW c mid) v = fun _ => (0 : ℝ) :=
  projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 0 (fun _ => rfl) _

theorem sealResB_eq (N h w c mid : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w))) :
    mnv2ResidB N h w (sealResW c mid) v = v := by
  funext k
  rw [mnv2ResidB, residual_apply, congrFun (sealResBody N h w c mid hn v) k]
  ring
-- ════════════════════════════════════════════════════════════════
-- § 5. The three channel-changing block collapses
-- ════════════════════════════════════════════════════════════════
theorem sealExpB_eq (N h w ic mid oc : Nat) (hm : Mg N h w) (v : Vec (N * (ic * h * w))) :
    mnv2ExpOnlyB N h w (sealIVW ic mid oc) v
      = StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 0)
          (StableHLO.batchMap N (flatConv (h := h) (w := w) (ctK oc mid 1 1 1) (kv oc 0))
            (StableHLO.bnBatchLA N mid h w 1 (kv mid (1 / 64)) (kv mid 3)
              (StableHLO.batchMap N
                (depthwiseFlat (h := h) (w := w) (ctDW mid 3 3 1) (kv mid 0))
                (StableHLO.bnBatchLA N mid h w 1 (kv mid (1 / 64)) (kv mid 3)
                  (StableHLO.batchMap N
                    (flatConv (h := h) (w := w) (ctK mid ic 1 1 1) (kv mid 0)) v))))) := by
  show projB N (h := h) (w := w) (ctK oc mid 1 1 1) (kv oc 0) 1 (kv oc (1 / 64)) (kv oc 0)
      (StableHLO.dwbrB N (h := h) (w := w) (ctDW mid 3 3 1) (kv mid 0) 1 (kv mid (1 / 64))
        (kv mid 3)
        (StableHLO.cbrB N (h := h) (w := w) (ctK mid ic 1 1 1) (kv mid 0) 1 (kv mid (1 / 64))
          (kv mid 3) v)) = _
  rw [cbrB_eq _ _ hm, dwbrB_eq _ _ hm]
  rfl

theorem sealStridedB_eq (N h w ic mid oc : Nat) (hme : Mg N (2 * h) (2 * w)) (hmd : Mg N h w)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    mnv2StridedB N h w (sealIVW ic mid oc) v
      = StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 0)
          (StableHLO.batchMap N (flatConv (h := h) (w := w) (ctK oc mid 1 1 1) (kv oc 0))
            (StableHLO.bnBatchLA N mid h w 1 (kv mid (1 / 64)) (kv mid 3)
              (StableHLO.batchMap N
                (depthwiseStride2FlatXla (h := h) (w := w) (ctDW mid 3 3 1) (kv mid 0))
                (StableHLO.bnBatchLA N mid (2 * h) (2 * w) 1 (kv mid (1 / 64)) (kv mid 3)
                  (StableHLO.batchMap N
                    (flatConv (h := 2 * h) (w := 2 * w) (ctK mid ic 1 1 1) (kv mid 0))
                    v))))) := by
  show projB N (h := h) (w := w) (ctK oc mid 1 1 1) (kv oc 0) 1 (kv oc (1 / 64)) (kv oc 0)
      (StableHLO.dwbrBstrided N (h := h) (w := w) (ctDW mid 3 3 1) (kv mid 0) 1 (kv mid (1 / 64))
        (kv mid 3)
        (StableHLO.cbrB N (h := 2 * h) (w := 2 * w) (ctK mid ic 1 1 1) (kv mid 0) 1
          (kv mid (1 / 64)) (kv mid 3) v)) = _
  rw [cbrB_eq _ _ hme, dwbrBstrided_eq _ _ hmd]
  rfl

theorem sealNoExpB_eq (N h w ic oc : Nat) (hm : Mg N h w) (v : Vec (N * (ic * h * w))) :
    mnv2NoExpB N h w (sealNoExpW ic oc) v
      = StableHLO.bnBatchLA N oc h w 1 (kv oc (1 / 64)) (kv oc 0)
          (StableHLO.batchMap N (flatConv (h := h) (w := w) (ctK oc ic 1 1 1) (kv oc 0))
            (StableHLO.bnBatchLA N ic h w 1 (kv ic (1 / 64)) (kv ic 3)
              (StableHLO.batchMap N
                (depthwiseFlat (h := h) (w := w) (ctDW ic 3 3 1) (kv ic 0)) v))) := by
  show projB N (h := h) (w := w) (ctK oc ic 1 1 1) (kv oc 0) 1 (kv oc (1 / 64)) (kv oc 0)
      (StableHLO.dwbrB N (h := h) (w := w) (ctDW ic 3 3 1) (kv ic 0) 1 (kv ic (1 / 64))
        (kv ic 3) v) = _
  rw [dwbrB_eq _ _ hm]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 6. Continuity of every block kind
--   relu6 is continuous (`relu6_continuous`), so these need no smoothness hypothesis — they
--   are for the ray argument's `Rr`, not for the VJP.
-- ════════════════════════════════════════════════════════════════
theorem cbrB_continuous (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Continuous (StableHLO.cbrB N (h := h) (w := w) W b ε γ β) :=
  (relu6_continuous _).comp (projB_continuous N W b ε hε γ β)

theorem dwbB_continuous (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    Continuous (StableHLO.bnBatchLA N c h w ε γ β ∘
      StableHLO.batchMap N (depthwiseFlat (h := h) (w := w) W b)) :=
  (bnBatchLA_differentiable N c h w ε hε γ β).continuous.comp
    (batchMap_continuous _ (depthwiseFlat_differentiable W b).continuous)

theorem dwbrB_continuous (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    Continuous (StableHLO.dwbrB N (h := h) (w := w) W b ε γ β) :=
  (relu6_continuous _).comp (dwbB_continuous N W b ε hε γ β)

theorem dwbrBstrided_continuous (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    Continuous (StableHLO.dwbrBstrided N (h := h) (w := w) W b ε γ β) :=
  (relu6_continuous _).comp
    ((bnBatchLA_differentiable N c h w ε hε γ β).continuous.comp
      (batchMap_continuous _ (depthwiseStride2FlatXla_differentiable W b).continuous))

theorem mnv2StemB_continuous (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
    (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) :
    Continuous (mnv2StemB N h w Ws bs εs γs βs) :=
  (relu6_continuous _).comp
    ((bnBatchLA_differentiable N oc h w εs hεs γs βs).continuous.comp
      (batchMap_continuous _ (flatConvStride2Xla_differentiable Ws bs).continuous))

theorem mnv2NoExpB_continuous (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hd : 0 < p.dε) (hp : 0 < p.pε) : Continuous (mnv2NoExpB N h w p) :=
  (projB_continuous N p.pW p.pb p.pε hp p.pγ p.pβ).comp
    (dwbrB_continuous N p.dW p.db p.dε hd p.dγ p.dβ)

theorem mnv2ExpOnlyB_continuous (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Continuous (mnv2ExpOnlyB N h w p) :=
  (projB_continuous N p.pW p.pb p.pε hp p.pγ p.pβ).comp
    ((dwbrB_continuous N p.dW p.db p.dε hd p.dγ p.dβ).comp
      (cbrB_continuous N p.eW p.eb p.eε he p.eγ p.eβ))

theorem mnv2ResidB_continuous (N h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Continuous (mnv2ResidB N h w p) :=
  residual_continuous _ (mnv2ExpOnlyB_continuous N h w p he hd hp)

theorem mnv2StridedB_continuous (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (he : 0 < p.eε) (hd : 0 < p.dε) (hp : 0 < p.pε) : Continuous (mnv2StridedB N h w p) :=
  (projB_continuous N p.pW p.pb p.pε hp p.pγ p.pβ).comp
    ((dwbrBstrided_continuous N p.dW p.db p.dε hd p.dγ p.dβ).comp
      (cbrB_continuous N (h := 2 * h) (w := 2 * w) p.eW p.eb p.eε he p.eγ p.eβ))

-- ════════════════════════════════════════════════════════════════
-- § 7. The witness input and the ray
-- ════════════════════════════════════════════════════════════════

/-- The witness input: the shared ray at 224×224, `N = 2`. Both examples carry the same ramp in
    channel 0, and `sealV` adds `t` to all of example 0's channel 0. ⭐ Every clause of this net is
    weight-only, so the ramp is doing no work beyond keeping one witness shape across the four
    sealed nets. -/
noncomputable def sealX (t : ℝ) : Vec (2 * (3 * (2 * 112) * (2 * 112))) :=
  rayX (2 * 112) (2 * 112) t

/-- Its direction — all of example 0's channel 0. -/
noncomputable def sealV : Vec (2 * (3 * (2 * 112) * (2 * 112))) := rayV (2 * 112) (2 * 112)

theorem sealX_zero_add (t : ℝ) : sealX 0 + t • sealV = sealX t := rayX_zero_add _ _ t

theorem EDiff_sealX (t : ℝ) : EDiff (fun ci => if ci.val = 0 then t else 0) (sealX t) :=
  EDiff_rayX _ _ t

theorem sealX_continuous : Continuous sealX := rayX_continuous _ _

-- ════════════════════════════════════════════════════════════════
-- § 8. The scalar BatchNorm factor
--   ⭐ `rf` is the whole contribution of one carrier BatchNorm: `γ · istd` at `γ = 1/64`.
--   `Rr` (§13) is a 22-fold product of these and nothing else.
-- ════════════════════════════════════════════════════════════════
noncomputable def rf (n : Nat) (z : Vec n) : ℝ := 1 / 64 * bnIstd n z 1

theorem rf_pos (n : Nat) (z : Vec n) : 0 < rf n z :=
  mul_pos (by norm_num) (bnIstd_pos _ 1 one_pos)

theorem rf_cont (n : Nat) (k : Fin n) : Continuous (fun z : Vec n => rf n z) :=
  continuous_const.mul (bnIstd_cont 1 one_pos k)

theorem rfac_cont (oc h w : Nat) (k : Fin (2 * (h * w))) (c : Fin oc)
    (Z : ℝ → Vec (2 * (oc * h * w))) (hZ : Continuous Z) :
    Continuous (fun t => rf (2 * (h * w)) (bnRowLA 2 oc h w (Z t) c)) :=
  (rf_cont _ k).comp ((bnRowLA_continuous 2 oc h w c).comp hZ)
-- ════════════════════════════════════════════════════════════════
-- § 9. Positivity and the 19 clause bundles, generically
-- ════════════════════════════════════════════════════════════════
theorem sealIVPos (ic mid oc : Nat) : IVPos (sealIVW ic mid oc) := ⟨one_pos, one_pos, one_pos⟩

theorem sealResPos (c mid : Nat) : IVPos (sealResW c mid) := ⟨one_pos, one_pos, one_pos⟩

theorem sealNoExpPos (ic oc : Nat) : IVNoExpPos (sealNoExpW ic oc) := ⟨one_pos, one_pos⟩

theorem sealIVSmooth (N h w ic mid oc : Nat) (hm : Mg N h w) (v : Vec (N * (ic * h * w))) :
    IVSmoothAtB N h w (sealIVW ic mid oc) v where
  he := win6 N mid h w hm _
  hd := win6 N mid h w hm _

theorem sealResSmooth (N h w c mid : Nat) (hm : Mg N h w) (v : Vec (N * (c * h * w))) :
    IVSmoothAtB N h w (sealResW c mid) v where
  he := win6 N mid h w hm _
  hd := win6 N mid h w hm _

theorem sealStridedSmooth (N h w ic mid oc : Nat) (hme : Mg N (2 * h) (2 * w)) (hmd : Mg N h w)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IVStridedSmoothAtB N h w (sealIVW ic mid oc) v where
  he := win6 N mid (2 * h) (2 * w) hme _
  hd := win6 N mid h w hmd _

theorem sealNoExpSmooth (N h w ic oc : Nat) (hm : Mg N h w) (v : Vec (N * (ic * h * w))) :
    IVNoExpSmoothAtB N h w (sealNoExpW ic oc) v where
  hd := win6 N ic h w hm _

theorem sealStemSmooth (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (hm : Mg N h w) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    MNV2StemSmoothAtB N h w Ws bs 1 (kv oc (1 / 64)) (kv oc 3) x :=
  win6 N oc h w hm _

theorem sealHeadSmooth (N h w : Nat) {ic oc : Nat} (Wh : Kernel4 oc ic 1 1) (bh : Vec oc)
    (hm : Mg N h w) (v : Vec (N * (ic * h * w))) :
    MNV2HeadSmoothAtB N h w Wh bh 1 (kv oc (1 / 64)) (kv oc 3) v :=
  win6 N oc h w hm _
-- ════════════════════════════════════════════════════════════════
-- § 10. The 19 bundles at the witness, and the whole-net VJP
--   ⭐⭐ Every one of these is weight-only: not one reads `sealX t`, and none of them mentions
--   `t` at all beyond carrying it through the activation's type.
-- ════════════════════════════════════════════════════════════════
theorem scStem (nCls : Nat) (t : ℝ) :
    MNV2StemSmoothAtB 2 112 112 (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε
      (sealW nCls).sγ (sealW nCls).sβ (sealX t) :=
  sealStemSmooth 2 112 112 _ _ (margin192 _ (by norm_num)) (sealX t)

theorem sc1 (nCls : Nat) (t : ℝ) :
    IVNoExpSmoothAtB 2 112 112 (sealW nCls).b1 (mnv2PreB0 2 (sealW nCls) (sealX t)) :=
  sealNoExpSmooth 2 112 112 32 16 (margin192 _ (by norm_num)) _

theorem sc2 (nCls : Nat) (t : ℝ) :
    IVStridedSmoothAtB 2 56 56 (sealW nCls).b2 (mnv2PreB1 2 (sealW nCls) (sealX t)) :=
  sealStridedSmooth 2 56 56 16 96 24 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem sc3 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 56 56 (sealW nCls).b3 (mnv2PreB2 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 56 56 24 144 (margin192 _ (by norm_num)) _

theorem sc4 (nCls : Nat) (t : ℝ) :
    IVStridedSmoothAtB 2 28 28 (sealW nCls).b4 (mnv2PreB3 2 (sealW nCls) (sealX t)) :=
  sealStridedSmooth 2 28 28 24 144 32 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem sc5 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 28 28 (sealW nCls).b5 (mnv2PreB4 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 28 28 32 192 (margin192 _ (by norm_num)) _

theorem sc6 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 28 28 (sealW nCls).b6 (mnv2PreB5 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 28 28 32 192 (margin192 _ (by norm_num)) _

theorem sc7 (nCls : Nat) (t : ℝ) :
    IVStridedSmoothAtB 2 14 14 (sealW nCls).b7 (mnv2PreB6 2 (sealW nCls) (sealX t)) :=
  sealStridedSmooth 2 14 14 32 192 64 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem sc8 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 14 14 (sealW nCls).b8 (mnv2PreB7 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 14 14 64 384 (margin192 _ (by norm_num)) _

theorem sc9 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 14 14 (sealW nCls).b9 (mnv2PreB8 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 14 14 64 384 (margin192 _ (by norm_num)) _

theorem sc10 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 14 14 (sealW nCls).b10 (mnv2PreB9 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 14 14 64 384 (margin192 _ (by norm_num)) _

theorem sc11 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 14 14 (sealW nCls).b11 (mnv2PreB10 2 (sealW nCls) (sealX t)) :=
  sealIVSmooth 2 14 14 64 384 96 (margin192 _ (by norm_num)) _

theorem sc12 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 14 14 (sealW nCls).b12 (mnv2PreB11 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 14 14 96 576 (margin192 _ (by norm_num)) _

theorem sc13 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 14 14 (sealW nCls).b13 (mnv2PreB12 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 14 14 96 576 (margin192 _ (by norm_num)) _

theorem sc14 (nCls : Nat) (t : ℝ) :
    IVStridedSmoothAtB 2 7 7 (sealW nCls).b14 (mnv2PreB13 2 (sealW nCls) (sealX t)) :=
  sealStridedSmooth 2 7 7 96 576 160 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem sc15 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 7 7 (sealW nCls).b15 (mnv2PreB14 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 7 7 160 960 (margin192 _ (by norm_num)) _

theorem sc16 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 7 7 (sealW nCls).b16 (mnv2PreB15 2 (sealW nCls) (sealX t)) :=
  sealResSmooth 2 7 7 160 960 (margin192 _ (by norm_num)) _

theorem sc17 (nCls : Nat) (t : ℝ) :
    IVSmoothAtB 2 7 7 (sealW nCls).b17 (mnv2PreB16 2 (sealW nCls) (sealX t)) :=
  sealIVSmooth 2 7 7 160 960 320 (margin192 _ (by norm_num)) _

theorem scHead (nCls : Nat) (t : ℝ) :
    MNV2HeadSmoothAtB 2 7 7 (sealW nCls).hW (sealW nCls).hb (sealW nCls).hε (sealW nCls).hγ
      (sealW nCls).hβ (mnv2PreB17 2 (sealW nCls) (sealX t)) :=
  sealHeadSmooth 2 7 7 _ _ (margin192 _ (by norm_num)) _

/-- ⭐⭐ **The whole-net VJP at the witness** — all 19 bundles discharged, on
    `mobilenetv2ForwardB_full` itself (through `mobilenetv2ForwardB_full_eq_chain`). -/
noncomputable def sealVJP (nCls : Nat) (t : ℝ) :
    HasVJPAt (mobilenetv2ForwardB_full 2 (sealW nCls)) (sealX t) := by
  rw [show mobilenetv2ForwardB_full 2 (sealW nCls)
      = mnv2HeadB 2 7 7 (sealW nCls).hW (sealW nCls).hb (sealW nCls).hε (sealW nCls).hγ
          (sealW nCls).hβ (sealW nCls).fcW (sealW nCls).fcb ∘ mnv2PreB17 2 (sealW nCls)
      from funext (mobilenetv2ForwardB_full_eq_chain 2 (sealW nCls))]
  exact mobilenetv2ForwardB_full_has_vjp_at 2 (sealW nCls) one_pos one_pos
    (sealNoExpPos 32 16) (sealIVPos 16 96 24) (sealResPos 24 144) (sealIVPos 24 144 32) (sealResPos 32 192) (sealResPos 32 192) (sealIVPos 32 192 64) (sealResPos 64 384) (sealResPos 64 384) (sealResPos 64 384) (sealIVPos 64 384 96) (sealResPos 96 576) (sealResPos 96 576) (sealIVPos 96 576 160) (sealResPos 160 960) (sealResPos 160 960) (sealIVPos 160 960 320)
    (sealX t) (scStem nCls t)
    (sc1 nCls t) (sc2 nCls t) (sc3 nCls t) (sc4 nCls t) (sc5 nCls t) (sc6 nCls t) (sc7 nCls t) (sc8 nCls t) (sc9 nCls t) (sc10 nCls t) (sc11 nCls t) (sc12 nCls t) (sc13 nCls t) (sc14 nCls t) (sc15 nCls t) (sc16 nCls t) (sc17 nCls t)
    (scHead nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 11. The 22 pre-BatchNorm activations on the carrier's path
--   ⚠ Spatial sizes are written in the net's own `2 * h` nest, never as the collapsed numeral —
--   `ResNet50FullB.lean`'s header records why.
-- ════════════════════════════════════════════════════════════════
noncomputable def Zs (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.batchMap 2 (flatConvStride2Xla (h := 112) (w := 112) (ctK 32 3 3 3 1) (kv 32 0))
    (sealX t)

noncomputable def Z1d (nCls : Nat) (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 112) (w := 112) (ctDW 32 3 3 1) (kv 32 0))
    (mnv2PreB0 2 (sealW nCls) (sealX t))

noncomputable def A1d (nCls : Nat) (t : ℝ) : Vec (2 * (32 * 112 * 112)) :=
  StableHLO.bnBatchLA 2 32 112 112 1 (kv 32 (1 / 64)) (kv 32 3) (Z1d nCls t)

noncomputable def Z1p (nCls : Nat) (t : ℝ) : Vec (2 * (16 * 112 * 112)) :=
  StableHLO.batchMap 2 (flatConv (h := 112) (w := 112) (ctK 16 32 1 1 1) (kv 16 0)) (A1d nCls t)

noncomputable def Z2e (nCls : Nat) (t : ℝ) : Vec (2 * (96 * (2 * 56) * (2 * 56))) :=
  StableHLO.batchMap 2 (flatConv (h := (2 * 56)) (w := (2 * 56)) (ctK 96 16 1 1 1) (kv 96 0))
    (mnv2PreB1 2 (sealW nCls) (sealX t))

noncomputable def A2e (nCls : Nat) (t : ℝ) : Vec (2 * (96 * (2 * 56) * (2 * 56))) :=
  StableHLO.bnBatchLA 2 96 (2 * 56) (2 * 56) 1 (kv 96 (1 / 64)) (kv 96 3) (Z2e nCls t)

noncomputable def Z2d (nCls : Nat) (t : ℝ) : Vec (2 * (96 * 56 * 56)) :=
  StableHLO.batchMap 2 (depthwiseStride2FlatXla (h := 56) (w := 56) (ctDW 96 3 3 1) (kv 96 0)) (A2e nCls t)

noncomputable def A2d (nCls : Nat) (t : ℝ) : Vec (2 * (96 * 56 * 56)) :=
  StableHLO.bnBatchLA 2 96 56 56 1 (kv 96 (1 / 64)) (kv 96 3) (Z2d nCls t)

noncomputable def Z2p (nCls : Nat) (t : ℝ) : Vec (2 * (24 * 56 * 56)) :=
  StableHLO.batchMap 2 (flatConv (h := 56) (w := 56) (ctK 24 96 1 1 1) (kv 24 0)) (A2d nCls t)

noncomputable def Z4e (nCls : Nat) (t : ℝ) : Vec (2 * (144 * (2 * 28) * (2 * 28))) :=
  StableHLO.batchMap 2 (flatConv (h := (2 * 28)) (w := (2 * 28)) (ctK 144 24 1 1 1) (kv 144 0))
    (mnv2PreB3 2 (sealW nCls) (sealX t))

noncomputable def A4e (nCls : Nat) (t : ℝ) : Vec (2 * (144 * (2 * 28) * (2 * 28))) :=
  StableHLO.bnBatchLA 2 144 (2 * 28) (2 * 28) 1 (kv 144 (1 / 64)) (kv 144 3) (Z4e nCls t)

noncomputable def Z4d (nCls : Nat) (t : ℝ) : Vec (2 * (144 * 28 * 28)) :=
  StableHLO.batchMap 2 (depthwiseStride2FlatXla (h := 28) (w := 28) (ctDW 144 3 3 1) (kv 144 0)) (A4e nCls t)

noncomputable def A4d (nCls : Nat) (t : ℝ) : Vec (2 * (144 * 28 * 28)) :=
  StableHLO.bnBatchLA 2 144 28 28 1 (kv 144 (1 / 64)) (kv 144 3) (Z4d nCls t)

noncomputable def Z4p (nCls : Nat) (t : ℝ) : Vec (2 * (32 * 28 * 28)) :=
  StableHLO.batchMap 2 (flatConv (h := 28) (w := 28) (ctK 32 144 1 1 1) (kv 32 0)) (A4d nCls t)

noncomputable def Z7e (nCls : Nat) (t : ℝ) : Vec (2 * (192 * (2 * 14) * (2 * 14))) :=
  StableHLO.batchMap 2 (flatConv (h := (2 * 14)) (w := (2 * 14)) (ctK 192 32 1 1 1) (kv 192 0))
    (mnv2PreB6 2 (sealW nCls) (sealX t))

noncomputable def A7e (nCls : Nat) (t : ℝ) : Vec (2 * (192 * (2 * 14) * (2 * 14))) :=
  StableHLO.bnBatchLA 2 192 (2 * 14) (2 * 14) 1 (kv 192 (1 / 64)) (kv 192 3) (Z7e nCls t)

noncomputable def Z7d (nCls : Nat) (t : ℝ) : Vec (2 * (192 * 14 * 14)) :=
  StableHLO.batchMap 2 (depthwiseStride2FlatXla (h := 14) (w := 14) (ctDW 192 3 3 1) (kv 192 0)) (A7e nCls t)

noncomputable def A7d (nCls : Nat) (t : ℝ) : Vec (2 * (192 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 192 14 14 1 (kv 192 (1 / 64)) (kv 192 3) (Z7d nCls t)

noncomputable def Z7p (nCls : Nat) (t : ℝ) : Vec (2 * (64 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 64 192 1 1 1) (kv 64 0)) (A7d nCls t)

noncomputable def Z11e (nCls : Nat) (t : ℝ) : Vec (2 * (384 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 384 64 1 1 1) (kv 384 0))
    (mnv2PreB10 2 (sealW nCls) (sealX t))

noncomputable def A11e (nCls : Nat) (t : ℝ) : Vec (2 * (384 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 384 14 14 1 (kv 384 (1 / 64)) (kv 384 3) (Z11e nCls t)

noncomputable def Z11d (nCls : Nat) (t : ℝ) : Vec (2 * (384 * 14 * 14)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 14) (w := 14) (ctDW 384 3 3 1) (kv 384 0)) (A11e nCls t)

noncomputable def A11d (nCls : Nat) (t : ℝ) : Vec (2 * (384 * 14 * 14)) :=
  StableHLO.bnBatchLA 2 384 14 14 1 (kv 384 (1 / 64)) (kv 384 3) (Z11d nCls t)

noncomputable def Z11p (nCls : Nat) (t : ℝ) : Vec (2 * (96 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConv (h := 14) (w := 14) (ctK 96 384 1 1 1) (kv 96 0)) (A11d nCls t)

noncomputable def Z14e (nCls : Nat) (t : ℝ) : Vec (2 * (576 * (2 * 7) * (2 * 7))) :=
  StableHLO.batchMap 2 (flatConv (h := (2 * 7)) (w := (2 * 7)) (ctK 576 96 1 1 1) (kv 576 0))
    (mnv2PreB13 2 (sealW nCls) (sealX t))

noncomputable def A14e (nCls : Nat) (t : ℝ) : Vec (2 * (576 * (2 * 7) * (2 * 7))) :=
  StableHLO.bnBatchLA 2 576 (2 * 7) (2 * 7) 1 (kv 576 (1 / 64)) (kv 576 3) (Z14e nCls t)

noncomputable def Z14d (nCls : Nat) (t : ℝ) : Vec (2 * (576 * 7 * 7)) :=
  StableHLO.batchMap 2 (depthwiseStride2FlatXla (h := 7) (w := 7) (ctDW 576 3 3 1) (kv 576 0)) (A14e nCls t)

noncomputable def A14d (nCls : Nat) (t : ℝ) : Vec (2 * (576 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 576 7 7 1 (kv 576 (1 / 64)) (kv 576 3) (Z14d nCls t)

noncomputable def Z14p (nCls : Nat) (t : ℝ) : Vec (2 * (160 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 160 576 1 1 1) (kv 160 0)) (A14d nCls t)

noncomputable def Z17e (nCls : Nat) (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 960 160 1 1 1) (kv 960 0))
    (mnv2PreB16 2 (sealW nCls) (sealX t))

noncomputable def A17e (nCls : Nat) (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 (1 / 64)) (kv 960 3) (Z17e nCls t)

noncomputable def Z17d (nCls : Nat) (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.batchMap 2 (depthwiseFlat (h := 7) (w := 7) (ctDW 960 3 3 1) (kv 960 0)) (A17e nCls t)

noncomputable def A17d (nCls : Nat) (t : ℝ) : Vec (2 * (960 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 960 7 7 1 (kv 960 (1 / 64)) (kv 960 3) (Z17d nCls t)

noncomputable def Z17p (nCls : Nat) (t : ℝ) : Vec (2 * (320 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 320 960 1 1 1) (kv 320 0)) (A17d nCls t)

noncomputable def Zh (nCls : Nat) (t : ℝ) : Vec (2 * (1280 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConv (h := 7) (w := 7) (ctK 1280 320 1 1 1) (kv 1280 0))
    (mnv2PreB17 2 (sealW nCls) (sealX t))

noncomputable def Ah (nCls : Nat) (t : ℝ) : Vec (2 * (1280 * 7 * 7)) :=
  StableHLO.bnBatchLA 2 1280 7 7 1 (kv 1280 (1 / 64)) (kv 1280 3) (Zh nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 12. The collapsed trunk, block by block
-- ════════════════════════════════════════════════════════════════

theorem pc0 (nCls : Nat) (t : ℝ) :
    mnv2PreB0 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 32 112 112 1 (kv 32 (1 / 64)) (kv 32 3) (Zs t) := by
  rw [mnv2PreB0_apply]
  exact mnv2StemB_eq _ _ (margin192 _ (by norm_num)) (sealX t)

theorem pc1 (nCls : Nat) (t : ℝ) :
    mnv2PreB1 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 16 112 112 1 (kv 16 (1 / 64)) (kv 16 0) (Z1p nCls t) := by
  rw [mnv2PreB1_apply]
  exact sealNoExpB_eq 2 112 112 32 16 (margin192 _ (by norm_num)) _

theorem pc2 (nCls : Nat) (t : ℝ) :
    mnv2PreB2 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 24 56 56 1 (kv 24 (1 / 64)) (kv 24 0) (Z2p nCls t) := by
  rw [mnv2PreB2_apply]
  exact sealStridedB_eq 2 56 56 16 96 24 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem pc3 (nCls : Nat) (t : ℝ) :
    mnv2PreB3 2 (sealW nCls) (sealX t) = mnv2PreB2 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB3_apply]
  exact sealResB_eq 2 56 56 24 144 (by norm_num) _

theorem pc4 (nCls : Nat) (t : ℝ) :
    mnv2PreB4 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 32 28 28 1 (kv 32 (1 / 64)) (kv 32 0) (Z4p nCls t) := by
  rw [mnv2PreB4_apply]
  exact sealStridedB_eq 2 28 28 24 144 32 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem pc5 (nCls : Nat) (t : ℝ) :
    mnv2PreB5 2 (sealW nCls) (sealX t) = mnv2PreB4 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB5_apply]
  exact sealResB_eq 2 28 28 32 192 (by norm_num) _

theorem pc6 (nCls : Nat) (t : ℝ) :
    mnv2PreB6 2 (sealW nCls) (sealX t) = mnv2PreB5 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB6_apply]
  exact sealResB_eq 2 28 28 32 192 (by norm_num) _

theorem pc7 (nCls : Nat) (t : ℝ) :
    mnv2PreB7 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 64 14 14 1 (kv 64 (1 / 64)) (kv 64 0) (Z7p nCls t) := by
  rw [mnv2PreB7_apply]
  exact sealStridedB_eq 2 14 14 32 192 64 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem pc8 (nCls : Nat) (t : ℝ) :
    mnv2PreB8 2 (sealW nCls) (sealX t) = mnv2PreB7 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB8_apply]
  exact sealResB_eq 2 14 14 64 384 (by norm_num) _

theorem pc9 (nCls : Nat) (t : ℝ) :
    mnv2PreB9 2 (sealW nCls) (sealX t) = mnv2PreB8 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB9_apply]
  exact sealResB_eq 2 14 14 64 384 (by norm_num) _

theorem pc10 (nCls : Nat) (t : ℝ) :
    mnv2PreB10 2 (sealW nCls) (sealX t) = mnv2PreB9 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB10_apply]
  exact sealResB_eq 2 14 14 64 384 (by norm_num) _

theorem pc11 (nCls : Nat) (t : ℝ) :
    mnv2PreB11 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 96 14 14 1 (kv 96 (1 / 64)) (kv 96 0) (Z11p nCls t) := by
  rw [mnv2PreB11_apply]
  exact sealExpB_eq 2 14 14 64 384 96 (margin192 _ (by norm_num)) _

theorem pc12 (nCls : Nat) (t : ℝ) :
    mnv2PreB12 2 (sealW nCls) (sealX t) = mnv2PreB11 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB12_apply]
  exact sealResB_eq 2 14 14 96 576 (by norm_num) _

theorem pc13 (nCls : Nat) (t : ℝ) :
    mnv2PreB13 2 (sealW nCls) (sealX t) = mnv2PreB12 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB13_apply]
  exact sealResB_eq 2 14 14 96 576 (by norm_num) _

theorem pc14 (nCls : Nat) (t : ℝ) :
    mnv2PreB14 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 160 7 7 1 (kv 160 (1 / 64)) (kv 160 0) (Z14p nCls t) := by
  rw [mnv2PreB14_apply]
  exact sealStridedB_eq 2 7 7 96 576 160 (margin192 _ (by norm_num)) (margin192 _ (by norm_num)) _

theorem pc15 (nCls : Nat) (t : ℝ) :
    mnv2PreB15 2 (sealW nCls) (sealX t) = mnv2PreB14 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB15_apply]
  exact sealResB_eq 2 7 7 160 960 (by norm_num) _

theorem pc16 (nCls : Nat) (t : ℝ) :
    mnv2PreB16 2 (sealW nCls) (sealX t) = mnv2PreB15 2 (sealW nCls) (sealX t) := by
  rw [mnv2PreB16_apply]
  exact sealResB_eq 2 7 7 160 960 (by norm_num) _

theorem pc17 (nCls : Nat) (t : ℝ) :
    mnv2PreB17 2 (sealW nCls) (sealX t)
      = StableHLO.bnBatchLA 2 320 7 7 1 (kv 320 (1 / 64)) (kv 320 0) (Z17p nCls t) := by
  rw [mnv2PreB17_apply]
  exact sealExpB_eq 2 7 7 160 960 320 (margin192 _ (by norm_num)) _

-- ════════════════════════════════════════════════════════════════
-- § 13. The carrier: 22 `EDiff` steps from the ray to the head
--   ⭐ `EDiff_dw` is the shape ResNet never needed: a depthwise cannot broadcast, so it scales
--   the carrier channel by channel where a centre-tap conv collapses it to `fun _ => s · δ 0`.
-- ════════════════════════════════════════════════════════════════

noncomputable def dS (t : ℝ) : Fin 32 → ℝ :=
  fun ci => t * rf (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Zs t) ci)

noncomputable def d1d (nCls : Nat) (t : ℝ) : Fin 32 → ℝ :=
  fun ci => dS t ci * rf (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Z1d nCls t) ci)

noncomputable def d1p (nCls : Nat) (t : ℝ) : Fin 16 → ℝ :=
  fun ci => d1d nCls t 0 * rf (2 * (112 * 112)) (bnRowLA 2 16 112 112 (Z1p nCls t) ci)

noncomputable def d2e (nCls : Nat) (t : ℝ) : Fin 96 → ℝ :=
  fun ci => d1p nCls t 0 * rf (2 * ((2 * 56) * (2 * 56))) (bnRowLA 2 96 (2 * 56) (2 * 56) (Z2e nCls t) ci)

noncomputable def d2d (nCls : Nat) (t : ℝ) : Fin 96 → ℝ :=
  fun ci => d2e nCls t ci * rf (2 * (56 * 56)) (bnRowLA 2 96 56 56 (Z2d nCls t) ci)

noncomputable def d2p (nCls : Nat) (t : ℝ) : Fin 24 → ℝ :=
  fun ci => d2d nCls t 0 * rf (2 * (56 * 56)) (bnRowLA 2 24 56 56 (Z2p nCls t) ci)

noncomputable def d4e (nCls : Nat) (t : ℝ) : Fin 144 → ℝ :=
  fun ci => d2p nCls t 0 * rf (2 * ((2 * 28) * (2 * 28))) (bnRowLA 2 144 (2 * 28) (2 * 28) (Z4e nCls t) ci)

noncomputable def d4d (nCls : Nat) (t : ℝ) : Fin 144 → ℝ :=
  fun ci => d4e nCls t ci * rf (2 * (28 * 28)) (bnRowLA 2 144 28 28 (Z4d nCls t) ci)

noncomputable def d4p (nCls : Nat) (t : ℝ) : Fin 32 → ℝ :=
  fun ci => d4d nCls t 0 * rf (2 * (28 * 28)) (bnRowLA 2 32 28 28 (Z4p nCls t) ci)

noncomputable def d7e (nCls : Nat) (t : ℝ) : Fin 192 → ℝ :=
  fun ci => d4p nCls t 0 * rf (2 * ((2 * 14) * (2 * 14))) (bnRowLA 2 192 (2 * 14) (2 * 14) (Z7e nCls t) ci)

noncomputable def d7d (nCls : Nat) (t : ℝ) : Fin 192 → ℝ :=
  fun ci => d7e nCls t ci * rf (2 * (14 * 14)) (bnRowLA 2 192 14 14 (Z7d nCls t) ci)

noncomputable def d7p (nCls : Nat) (t : ℝ) : Fin 64 → ℝ :=
  fun ci => d7d nCls t 0 * rf (2 * (14 * 14)) (bnRowLA 2 64 14 14 (Z7p nCls t) ci)

noncomputable def d11e (nCls : Nat) (t : ℝ) : Fin 384 → ℝ :=
  fun ci => d7p nCls t 0 * rf (2 * (14 * 14)) (bnRowLA 2 384 14 14 (Z11e nCls t) ci)

noncomputable def d11d (nCls : Nat) (t : ℝ) : Fin 384 → ℝ :=
  fun ci => d11e nCls t ci * rf (2 * (14 * 14)) (bnRowLA 2 384 14 14 (Z11d nCls t) ci)

noncomputable def d11p (nCls : Nat) (t : ℝ) : Fin 96 → ℝ :=
  fun ci => d11d nCls t 0 * rf (2 * (14 * 14)) (bnRowLA 2 96 14 14 (Z11p nCls t) ci)

noncomputable def d14e (nCls : Nat) (t : ℝ) : Fin 576 → ℝ :=
  fun ci => d11p nCls t 0 * rf (2 * ((2 * 7) * (2 * 7))) (bnRowLA 2 576 (2 * 7) (2 * 7) (Z14e nCls t) ci)

noncomputable def d14d (nCls : Nat) (t : ℝ) : Fin 576 → ℝ :=
  fun ci => d14e nCls t ci * rf (2 * (7 * 7)) (bnRowLA 2 576 7 7 (Z14d nCls t) ci)

noncomputable def d14p (nCls : Nat) (t : ℝ) : Fin 160 → ℝ :=
  fun ci => d14d nCls t 0 * rf (2 * (7 * 7)) (bnRowLA 2 160 7 7 (Z14p nCls t) ci)

noncomputable def d17e (nCls : Nat) (t : ℝ) : Fin 960 → ℝ :=
  fun ci => d14p nCls t 0 * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Z17e nCls t) ci)

noncomputable def d17d (nCls : Nat) (t : ℝ) : Fin 960 → ℝ :=
  fun ci => d17e nCls t ci * rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Z17d nCls t) ci)

noncomputable def d17p (nCls : Nat) (t : ℝ) : Fin 320 → ℝ :=
  fun ci => d17d nCls t 0 * rf (2 * (7 * 7)) (bnRowLA 2 320 7 7 (Z17p nCls t) ci)

noncomputable def dH (nCls : Nat) (t : ℝ) : Fin 1280 → ℝ :=
  fun ci => d17p nCls t 0 * rf (2 * (7 * 7)) (bnRowLA 2 1280 7 7 (Zh nCls t) ci)

theorem ed0 (nCls : Nat) (t : ℝ) : EDiff (dS t) (mnv2PreB0 2 (sealW nCls) (sealX t)) := by
  rw [pc0]
  refine EDiff_bn 32 112 112 1 (kv 32 (1 / 64)) (kv 32 3) (fun _ => 1 * t) (dS t) (Zs t) ?_ ?_
  · exact EDiff_convS2Xla (h := 112) (w := 112) (0 : Fin 3) rfl (by norm_num) (by norm_num) 1
      (kv 32 0) _ _ (sealX t) (EDiff_sealX t) (fun o => by norm_num)
  · intro ci
    simp only [dS, rf, kv_apply]
    ring

theorem ed1d (nCls : Nat) (t : ℝ) : EDiff (d1d nCls t) (A1d nCls t) := by
  refine EDiff_bn 32 112 112 1 (kv 32 (1 / 64)) (kv 32 3) (fun ch => 1 * dS t ch) (d1d nCls t)
    (Z1d nCls t) ?_ ?_
  · exact EDiff_dw (by norm_num) (by norm_num) 1 (kv 32 0) (dS t) _
      (mnv2PreB0 2 (sealW nCls) (sealX t)) (ed0 nCls t) (fun ch => rfl)
  · intro ci
    simp only [d1d, rf, kv_apply]
    ring

theorem ed1 (nCls : Nat) (t : ℝ) : EDiff (d1p nCls t) (mnv2PreB1 2 (sealW nCls) (sealX t)) := by
  rw [pc1]
  refine EDiff_bn 16 112 112 1 (kv 16 (1 / 64)) (kv 16 0) (fun _ => 1 * d1d nCls t 0)
    (d1p nCls t) (Z1p nCls t) ?_ ?_
  · exact EDiff_conv (h := 112) (w := 112) (0 : Fin 32) rfl (by norm_num) (by norm_num) 1
      (kv 16 0) (d1d nCls t) _ (A1d nCls t) (ed1d nCls t) (fun o => rfl)
  · intro ci
    simp only [d1p, rf, kv_apply]
    ring

theorem ed2e (nCls : Nat) (t : ℝ) : EDiff (d2e nCls t) (A2e nCls t) := by
  refine EDiff_bn 96 (2 * 56) (2 * 56) 1 (kv 96 (1 / 64)) (kv 96 3) (fun _ => 1 * d1p nCls t 0)
    (d2e nCls t) (Z2e nCls t) ?_ ?_
  · exact EDiff_conv (h := (2 * 56)) (w := (2 * 56)) (0 : Fin 16) rfl (by norm_num) (by norm_num) 1
      (kv 96 0) (d1p nCls t) _ (mnv2PreB1 2 (sealW nCls) (sealX t)) (ed1 nCls t) (fun o => rfl)
  · intro ci
    simp only [d2e, rf, kv_apply]
    ring

theorem ed2d (nCls : Nat) (t : ℝ) : EDiff (d2d nCls t) (A2d nCls t) := by
  refine EDiff_bn 96 56 56 1 (kv 96 (1 / 64)) (kv 96 3) (fun ch => 1 * d2e nCls t ch)
    (d2d nCls t) (Z2d nCls t) ?_ ?_
  · exact EDiff_dwS2Xla (by norm_num) (by norm_num) 1 (kv 96 0) (d2e nCls t) _
      (A2e nCls t) (ed2e nCls t) (fun ch => rfl)
  · intro ci
    simp only [d2d, rf, kv_apply]
    ring

theorem ed2 (nCls : Nat) (t : ℝ) :
    EDiff (d2p nCls t) (mnv2PreB2 2 (sealW nCls) (sealX t)) := by
  rw [pc2]
  refine EDiff_bn 24 56 56 1 (kv 24 (1 / 64)) (kv 24 0) (fun _ => 1 * d2d nCls t 0)
    (d2p nCls t) (Z2p nCls t) ?_ ?_
  · exact EDiff_conv (h := 56) (w := 56) (0 : Fin 96) rfl (by norm_num) (by norm_num) 1
      (kv 24 0) (d2d nCls t) _ (A2d nCls t) (ed2d nCls t) (fun o => rfl)
  · intro ci
    simp only [d2p, rf, kv_apply]
    ring

theorem ed3 (nCls : Nat) (t : ℝ) :
    EDiff (d2p nCls t) (mnv2PreB3 2 (sealW nCls) (sealX t)) := by
  rw [pc3]
  exact ed2 nCls t

theorem ed4e (nCls : Nat) (t : ℝ) : EDiff (d4e nCls t) (A4e nCls t) := by
  refine EDiff_bn 144 (2 * 28) (2 * 28) 1 (kv 144 (1 / 64)) (kv 144 3) (fun _ => 1 * d2p nCls t 0)
    (d4e nCls t) (Z4e nCls t) ?_ ?_
  · exact EDiff_conv (h := (2 * 28)) (w := (2 * 28)) (0 : Fin 24) rfl (by norm_num) (by norm_num) 1
      (kv 144 0) (d2p nCls t) _ (mnv2PreB3 2 (sealW nCls) (sealX t)) (ed3 nCls t) (fun o => rfl)
  · intro ci
    simp only [d4e, rf, kv_apply]
    ring

theorem ed4d (nCls : Nat) (t : ℝ) : EDiff (d4d nCls t) (A4d nCls t) := by
  refine EDiff_bn 144 28 28 1 (kv 144 (1 / 64)) (kv 144 3) (fun ch => 1 * d4e nCls t ch)
    (d4d nCls t) (Z4d nCls t) ?_ ?_
  · exact EDiff_dwS2Xla (by norm_num) (by norm_num) 1 (kv 144 0) (d4e nCls t) _
      (A4e nCls t) (ed4e nCls t) (fun ch => rfl)
  · intro ci
    simp only [d4d, rf, kv_apply]
    ring

theorem ed4 (nCls : Nat) (t : ℝ) :
    EDiff (d4p nCls t) (mnv2PreB4 2 (sealW nCls) (sealX t)) := by
  rw [pc4]
  refine EDiff_bn 32 28 28 1 (kv 32 (1 / 64)) (kv 32 0) (fun _ => 1 * d4d nCls t 0)
    (d4p nCls t) (Z4p nCls t) ?_ ?_
  · exact EDiff_conv (h := 28) (w := 28) (0 : Fin 144) rfl (by norm_num) (by norm_num) 1
      (kv 32 0) (d4d nCls t) _ (A4d nCls t) (ed4d nCls t) (fun o => rfl)
  · intro ci
    simp only [d4p, rf, kv_apply]
    ring

theorem ed5 (nCls : Nat) (t : ℝ) :
    EDiff (d4p nCls t) (mnv2PreB5 2 (sealW nCls) (sealX t)) := by
  rw [pc5]
  exact ed4 nCls t

theorem ed6 (nCls : Nat) (t : ℝ) :
    EDiff (d4p nCls t) (mnv2PreB6 2 (sealW nCls) (sealX t)) := by
  rw [pc6]
  exact ed5 nCls t

theorem ed7e (nCls : Nat) (t : ℝ) : EDiff (d7e nCls t) (A7e nCls t) := by
  refine EDiff_bn 192 (2 * 14) (2 * 14) 1 (kv 192 (1 / 64)) (kv 192 3) (fun _ => 1 * d4p nCls t 0)
    (d7e nCls t) (Z7e nCls t) ?_ ?_
  · exact EDiff_conv (h := (2 * 14)) (w := (2 * 14)) (0 : Fin 32) rfl (by norm_num) (by norm_num) 1
      (kv 192 0) (d4p nCls t) _ (mnv2PreB6 2 (sealW nCls) (sealX t)) (ed6 nCls t) (fun o => rfl)
  · intro ci
    simp only [d7e, rf, kv_apply]
    ring

theorem ed7d (nCls : Nat) (t : ℝ) : EDiff (d7d nCls t) (A7d nCls t) := by
  refine EDiff_bn 192 14 14 1 (kv 192 (1 / 64)) (kv 192 3) (fun ch => 1 * d7e nCls t ch)
    (d7d nCls t) (Z7d nCls t) ?_ ?_
  · exact EDiff_dwS2Xla (by norm_num) (by norm_num) 1 (kv 192 0) (d7e nCls t) _
      (A7e nCls t) (ed7e nCls t) (fun ch => rfl)
  · intro ci
    simp only [d7d, rf, kv_apply]
    ring

theorem ed7 (nCls : Nat) (t : ℝ) :
    EDiff (d7p nCls t) (mnv2PreB7 2 (sealW nCls) (sealX t)) := by
  rw [pc7]
  refine EDiff_bn 64 14 14 1 (kv 64 (1 / 64)) (kv 64 0) (fun _ => 1 * d7d nCls t 0)
    (d7p nCls t) (Z7p nCls t) ?_ ?_
  · exact EDiff_conv (h := 14) (w := 14) (0 : Fin 192) rfl (by norm_num) (by norm_num) 1
      (kv 64 0) (d7d nCls t) _ (A7d nCls t) (ed7d nCls t) (fun o => rfl)
  · intro ci
    simp only [d7p, rf, kv_apply]
    ring

theorem ed8 (nCls : Nat) (t : ℝ) :
    EDiff (d7p nCls t) (mnv2PreB8 2 (sealW nCls) (sealX t)) := by
  rw [pc8]
  exact ed7 nCls t

theorem ed9 (nCls : Nat) (t : ℝ) :
    EDiff (d7p nCls t) (mnv2PreB9 2 (sealW nCls) (sealX t)) := by
  rw [pc9]
  exact ed8 nCls t

theorem ed10 (nCls : Nat) (t : ℝ) :
    EDiff (d7p nCls t) (mnv2PreB10 2 (sealW nCls) (sealX t)) := by
  rw [pc10]
  exact ed9 nCls t

theorem ed11e (nCls : Nat) (t : ℝ) : EDiff (d11e nCls t) (A11e nCls t) := by
  refine EDiff_bn 384 14 14 1 (kv 384 (1 / 64)) (kv 384 3) (fun _ => 1 * d7p nCls t 0)
    (d11e nCls t) (Z11e nCls t) ?_ ?_
  · exact EDiff_conv (h := 14) (w := 14) (0 : Fin 64) rfl (by norm_num) (by norm_num) 1
      (kv 384 0) (d7p nCls t) _ (mnv2PreB10 2 (sealW nCls) (sealX t)) (ed10 nCls t) (fun o => rfl)
  · intro ci
    simp only [d11e, rf, kv_apply]
    ring

theorem ed11d (nCls : Nat) (t : ℝ) : EDiff (d11d nCls t) (A11d nCls t) := by
  refine EDiff_bn 384 14 14 1 (kv 384 (1 / 64)) (kv 384 3) (fun ch => 1 * d11e nCls t ch)
    (d11d nCls t) (Z11d nCls t) ?_ ?_
  · exact EDiff_dw (by norm_num) (by norm_num) 1 (kv 384 0) (d11e nCls t) _
      (A11e nCls t) (ed11e nCls t) (fun ch => rfl)
  · intro ci
    simp only [d11d, rf, kv_apply]
    ring

theorem ed11 (nCls : Nat) (t : ℝ) :
    EDiff (d11p nCls t) (mnv2PreB11 2 (sealW nCls) (sealX t)) := by
  rw [pc11]
  refine EDiff_bn 96 14 14 1 (kv 96 (1 / 64)) (kv 96 0) (fun _ => 1 * d11d nCls t 0)
    (d11p nCls t) (Z11p nCls t) ?_ ?_
  · exact EDiff_conv (h := 14) (w := 14) (0 : Fin 384) rfl (by norm_num) (by norm_num) 1
      (kv 96 0) (d11d nCls t) _ (A11d nCls t) (ed11d nCls t) (fun o => rfl)
  · intro ci
    simp only [d11p, rf, kv_apply]
    ring

theorem ed12 (nCls : Nat) (t : ℝ) :
    EDiff (d11p nCls t) (mnv2PreB12 2 (sealW nCls) (sealX t)) := by
  rw [pc12]
  exact ed11 nCls t

theorem ed13 (nCls : Nat) (t : ℝ) :
    EDiff (d11p nCls t) (mnv2PreB13 2 (sealW nCls) (sealX t)) := by
  rw [pc13]
  exact ed12 nCls t

theorem ed14e (nCls : Nat) (t : ℝ) : EDiff (d14e nCls t) (A14e nCls t) := by
  refine EDiff_bn 576 (2 * 7) (2 * 7) 1 (kv 576 (1 / 64)) (kv 576 3) (fun _ => 1 * d11p nCls t 0)
    (d14e nCls t) (Z14e nCls t) ?_ ?_
  · exact EDiff_conv (h := (2 * 7)) (w := (2 * 7)) (0 : Fin 96) rfl (by norm_num) (by norm_num) 1
      (kv 576 0) (d11p nCls t) _ (mnv2PreB13 2 (sealW nCls) (sealX t)) (ed13 nCls t) (fun o => rfl)
  · intro ci
    simp only [d14e, rf, kv_apply]
    ring

theorem ed14d (nCls : Nat) (t : ℝ) : EDiff (d14d nCls t) (A14d nCls t) := by
  refine EDiff_bn 576 7 7 1 (kv 576 (1 / 64)) (kv 576 3) (fun ch => 1 * d14e nCls t ch)
    (d14d nCls t) (Z14d nCls t) ?_ ?_
  · exact EDiff_dwS2Xla (by norm_num) (by norm_num) 1 (kv 576 0) (d14e nCls t) _
      (A14e nCls t) (ed14e nCls t) (fun ch => rfl)
  · intro ci
    simp only [d14d, rf, kv_apply]
    ring

theorem ed14 (nCls : Nat) (t : ℝ) :
    EDiff (d14p nCls t) (mnv2PreB14 2 (sealW nCls) (sealX t)) := by
  rw [pc14]
  refine EDiff_bn 160 7 7 1 (kv 160 (1 / 64)) (kv 160 0) (fun _ => 1 * d14d nCls t 0)
    (d14p nCls t) (Z14p nCls t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 576) rfl (by norm_num) (by norm_num) 1
      (kv 160 0) (d14d nCls t) _ (A14d nCls t) (ed14d nCls t) (fun o => rfl)
  · intro ci
    simp only [d14p, rf, kv_apply]
    ring

theorem ed15 (nCls : Nat) (t : ℝ) :
    EDiff (d14p nCls t) (mnv2PreB15 2 (sealW nCls) (sealX t)) := by
  rw [pc15]
  exact ed14 nCls t

theorem ed16 (nCls : Nat) (t : ℝ) :
    EDiff (d14p nCls t) (mnv2PreB16 2 (sealW nCls) (sealX t)) := by
  rw [pc16]
  exact ed15 nCls t

theorem ed17e (nCls : Nat) (t : ℝ) : EDiff (d17e nCls t) (A17e nCls t) := by
  refine EDiff_bn 960 7 7 1 (kv 960 (1 / 64)) (kv 960 3) (fun _ => 1 * d14p nCls t 0)
    (d17e nCls t) (Z17e nCls t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 160) rfl (by norm_num) (by norm_num) 1
      (kv 960 0) (d14p nCls t) _ (mnv2PreB16 2 (sealW nCls) (sealX t)) (ed16 nCls t) (fun o => rfl)
  · intro ci
    simp only [d17e, rf, kv_apply]
    ring

theorem ed17d (nCls : Nat) (t : ℝ) : EDiff (d17d nCls t) (A17d nCls t) := by
  refine EDiff_bn 960 7 7 1 (kv 960 (1 / 64)) (kv 960 3) (fun ch => 1 * d17e nCls t ch)
    (d17d nCls t) (Z17d nCls t) ?_ ?_
  · exact EDiff_dw (by norm_num) (by norm_num) 1 (kv 960 0) (d17e nCls t) _
      (A17e nCls t) (ed17e nCls t) (fun ch => rfl)
  · intro ci
    simp only [d17d, rf, kv_apply]
    ring

theorem ed17 (nCls : Nat) (t : ℝ) :
    EDiff (d17p nCls t) (mnv2PreB17 2 (sealW nCls) (sealX t)) := by
  rw [pc17]
  refine EDiff_bn 320 7 7 1 (kv 320 (1 / 64)) (kv 320 0) (fun _ => 1 * d17d nCls t 0)
    (d17p nCls t) (Z17p nCls t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 960) rfl (by norm_num) (by norm_num) 1
      (kv 320 0) (d17d nCls t) _ (A17d nCls t) (ed17d nCls t) (fun o => rfl)
  · intro ci
    simp only [d17p, rf, kv_apply]
    ring

theorem edH (nCls : Nat) (t : ℝ) : EDiff (dH nCls t) (Ah nCls t) := by
  refine EDiff_bn 1280 7 7 1 (kv 1280 (1 / 64)) (kv 1280 3) (fun _ => 1 * d17p nCls t 0)
    (dH nCls t) (Zh nCls t) ?_ ?_
  · exact EDiff_conv (h := 7) (w := 7) (0 : Fin 320) rfl (by norm_num) (by norm_num) 1
      (kv 1280 0) (d17p nCls t) _ (mnv2PreB17 2 (sealW nCls) (sealX t)) (ed17 nCls t)
      (fun o => rfl)
  · intro ci
    simp only [dH, rf, kv_apply]
    ring

-- ════════════════════════════════════════════════════════════════
-- § 14. Continuity of the trunk and of every carrier activation
-- ════════════════════════════════════════════════════════════════

theorem cn0 (nCls : Nat) : Continuous (mnv2PreB0 2 (sealW nCls)) :=
  mnv2StemB_continuous 2 112 112 _ _ _ one_pos _ _

theorem cn1 (nCls : Nat) : Continuous (mnv2PreB1 2 (sealW nCls)) :=
  (mnv2NoExpB_continuous 2 112 112 (sealW nCls).b1 one_pos one_pos).comp (cn0 nCls)

theorem cn2 (nCls : Nat) : Continuous (mnv2PreB2 2 (sealW nCls)) :=
  (mnv2StridedB_continuous 2 56 56 (sealW nCls).b2 one_pos one_pos one_pos).comp (cn1 nCls)

theorem cn3 (nCls : Nat) : Continuous (mnv2PreB3 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 56 56 (sealW nCls).b3 one_pos one_pos one_pos).comp (cn2 nCls)

theorem cn4 (nCls : Nat) : Continuous (mnv2PreB4 2 (sealW nCls)) :=
  (mnv2StridedB_continuous 2 28 28 (sealW nCls).b4 one_pos one_pos one_pos).comp (cn3 nCls)

theorem cn5 (nCls : Nat) : Continuous (mnv2PreB5 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 28 28 (sealW nCls).b5 one_pos one_pos one_pos).comp (cn4 nCls)

theorem cn6 (nCls : Nat) : Continuous (mnv2PreB6 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 28 28 (sealW nCls).b6 one_pos one_pos one_pos).comp (cn5 nCls)

theorem cn7 (nCls : Nat) : Continuous (mnv2PreB7 2 (sealW nCls)) :=
  (mnv2StridedB_continuous 2 14 14 (sealW nCls).b7 one_pos one_pos one_pos).comp (cn6 nCls)

theorem cn8 (nCls : Nat) : Continuous (mnv2PreB8 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 14 14 (sealW nCls).b8 one_pos one_pos one_pos).comp (cn7 nCls)

theorem cn9 (nCls : Nat) : Continuous (mnv2PreB9 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 14 14 (sealW nCls).b9 one_pos one_pos one_pos).comp (cn8 nCls)

theorem cn10 (nCls : Nat) : Continuous (mnv2PreB10 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 14 14 (sealW nCls).b10 one_pos one_pos one_pos).comp (cn9 nCls)

theorem cn11 (nCls : Nat) : Continuous (mnv2PreB11 2 (sealW nCls)) :=
  (mnv2ExpOnlyB_continuous 2 14 14 (sealW nCls).b11 one_pos one_pos one_pos).comp (cn10 nCls)

theorem cn12 (nCls : Nat) : Continuous (mnv2PreB12 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 14 14 (sealW nCls).b12 one_pos one_pos one_pos).comp (cn11 nCls)

theorem cn13 (nCls : Nat) : Continuous (mnv2PreB13 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 14 14 (sealW nCls).b13 one_pos one_pos one_pos).comp (cn12 nCls)

theorem cn14 (nCls : Nat) : Continuous (mnv2PreB14 2 (sealW nCls)) :=
  (mnv2StridedB_continuous 2 7 7 (sealW nCls).b14 one_pos one_pos one_pos).comp (cn13 nCls)

theorem cn15 (nCls : Nat) : Continuous (mnv2PreB15 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 7 7 (sealW nCls).b15 one_pos one_pos one_pos).comp (cn14 nCls)

theorem cn16 (nCls : Nat) : Continuous (mnv2PreB16 2 (sealW nCls)) :=
  (mnv2ResidB_continuous 2 7 7 (sealW nCls).b16 one_pos one_pos one_pos).comp (cn15 nCls)

theorem cn17 (nCls : Nat) : Continuous (mnv2PreB17 2 (sealW nCls)) :=
  (mnv2ExpOnlyB_continuous 2 7 7 (sealW nCls).b17 one_pos one_pos one_pos).comp (cn16 nCls)

theorem Zs_continuous : Continuous Zs :=
  (batchMap_continuous _ (flatConvStride2Xla_differentiable _ _).continuous).comp sealX_continuous

theorem Z1d_continuous (nCls : Nat) : Continuous (Z1d nCls) :=
  (batchMap_continuous _ (depthwiseFlat_differentiable _ _).continuous).comp
    ((cn0 nCls).comp sealX_continuous)

theorem Z1p_continuous (nCls : Nat) : Continuous (Z1p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 32 112 112 1 one_pos _ _).continuous.comp (Z1d_continuous nCls))

theorem Z2e_continuous (nCls : Nat) : Continuous (Z2e nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn1 nCls).comp sealX_continuous)

theorem Z2d_continuous (nCls : Nat) : Continuous (Z2d nCls) :=
  (batchMap_continuous _ (depthwiseStride2FlatXla_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 96 (2 * 56) (2 * 56) 1 one_pos _ _).continuous.comp (Z2e_continuous nCls))

theorem Z2p_continuous (nCls : Nat) : Continuous (Z2p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 96 56 56 1 one_pos _ _).continuous.comp (Z2d_continuous nCls))

theorem Z4e_continuous (nCls : Nat) : Continuous (Z4e nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn3 nCls).comp sealX_continuous)

theorem Z4d_continuous (nCls : Nat) : Continuous (Z4d nCls) :=
  (batchMap_continuous _ (depthwiseStride2FlatXla_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 144 (2 * 28) (2 * 28) 1 one_pos _ _).continuous.comp (Z4e_continuous nCls))

theorem Z4p_continuous (nCls : Nat) : Continuous (Z4p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 144 28 28 1 one_pos _ _).continuous.comp (Z4d_continuous nCls))

theorem Z7e_continuous (nCls : Nat) : Continuous (Z7e nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn6 nCls).comp sealX_continuous)

theorem Z7d_continuous (nCls : Nat) : Continuous (Z7d nCls) :=
  (batchMap_continuous _ (depthwiseStride2FlatXla_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 192 (2 * 14) (2 * 14) 1 one_pos _ _).continuous.comp (Z7e_continuous nCls))

theorem Z7p_continuous (nCls : Nat) : Continuous (Z7p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 192 14 14 1 one_pos _ _).continuous.comp (Z7d_continuous nCls))

theorem Z11e_continuous (nCls : Nat) : Continuous (Z11e nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn10 nCls).comp sealX_continuous)

theorem Z11d_continuous (nCls : Nat) : Continuous (Z11d nCls) :=
  (batchMap_continuous _ (depthwiseFlat_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 384 14 14 1 one_pos _ _).continuous.comp (Z11e_continuous nCls))

theorem Z11p_continuous (nCls : Nat) : Continuous (Z11p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 384 14 14 1 one_pos _ _).continuous.comp (Z11d_continuous nCls))

theorem Z14e_continuous (nCls : Nat) : Continuous (Z14e nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn13 nCls).comp sealX_continuous)

theorem Z14d_continuous (nCls : Nat) : Continuous (Z14d nCls) :=
  (batchMap_continuous _ (depthwiseStride2FlatXla_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 576 (2 * 7) (2 * 7) 1 one_pos _ _).continuous.comp (Z14e_continuous nCls))

theorem Z14p_continuous (nCls : Nat) : Continuous (Z14p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 576 7 7 1 one_pos _ _).continuous.comp (Z14d_continuous nCls))

theorem Z17e_continuous (nCls : Nat) : Continuous (Z17e nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn16 nCls).comp sealX_continuous)

theorem Z17d_continuous (nCls : Nat) : Continuous (Z17d nCls) :=
  (batchMap_continuous _ (depthwiseFlat_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 960 7 7 1 one_pos _ _).continuous.comp (Z17e_continuous nCls))

theorem Z17p_continuous (nCls : Nat) : Continuous (Z17p nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((bnBatchLA_differentiable 2 960 7 7 1 one_pos _ _).continuous.comp (Z17d_continuous nCls))

theorem Zh_continuous (nCls : Nat) : Continuous (Zh nCls) :=
  (batchMap_continuous _ (flatConv_differentiable _ _).continuous).comp
    ((cn17 nCls).comp sealX_continuous)

-- ════════════════════════════════════════════════════════════════
-- § 15. The nonlinear factor `Rr`
-- ════════════════════════════════════════════════════════════════
/-- ⭐⭐ **The positive, continuous nonlinear factor.** MobileNetV2's channel-changing blocks have
    no skip, so the carrier threads every BatchNorm inside them: the stem, both of `b1`'s, three
    each in `b2`, `b4`, `b7`, `b11`, `b14`, `b17`, and the head's. ⚠ No BatchNorm *variance*
    derivative is ever taken — `Rr` enters only through `t * Rr t`, whose derivative at `0` is
    `Rr 0` for any `Rr` continuous there. -/
noncomputable def Rr (nCls : Nat) (t : ℝ) : ℝ :=
  rf (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Zs t) 0) *
    (rf (2 * (112 * 112)) (bnRowLA 2 32 112 112 (Z1d nCls t) 0) *
    (rf (2 * (112 * 112)) (bnRowLA 2 16 112 112 (Z1p nCls t) 0) *
    (rf (2 * ((2 * 56) * (2 * 56))) (bnRowLA 2 96 (2 * 56) (2 * 56) (Z2e nCls t) 0) *
    (rf (2 * (56 * 56)) (bnRowLA 2 96 56 56 (Z2d nCls t) 0) *
    (rf (2 * (56 * 56)) (bnRowLA 2 24 56 56 (Z2p nCls t) 0) *
    (rf (2 * ((2 * 28) * (2 * 28))) (bnRowLA 2 144 (2 * 28) (2 * 28) (Z4e nCls t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 144 28 28 (Z4d nCls t) 0) *
    (rf (2 * (28 * 28)) (bnRowLA 2 32 28 28 (Z4p nCls t) 0) *
    (rf (2 * ((2 * 14) * (2 * 14))) (bnRowLA 2 192 (2 * 14) (2 * 14) (Z7e nCls t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 192 14 14 (Z7d nCls t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 64 14 14 (Z7p nCls t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 384 14 14 (Z11e nCls t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 384 14 14 (Z11d nCls t) 0) *
    (rf (2 * (14 * 14)) (bnRowLA 2 96 14 14 (Z11p nCls t) 0) *
    (rf (2 * ((2 * 7) * (2 * 7))) (bnRowLA 2 576 (2 * 7) (2 * 7) (Z14e nCls t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 576 7 7 (Z14d nCls t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 160 7 7 (Z14p nCls t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Z17e nCls t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 960 7 7 (Z17d nCls t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 320 7 7 (Z17p nCls t) 0) *
    (rf (2 * (7 * 7)) (bnRowLA 2 1280 7 7 (Zh nCls t) 0))))))))))))))))))))))

theorem Rr_pos (nCls : Nat) (t : ℝ) : 0 < Rr nCls t := by
  unfold Rr
  -- ⚠ not `repeat' apply mul_pos`: `rf` is itself a product, so `mul_pos` splits inside it and
  -- leaves `0 < 1/64` goals `rf_pos` cannot close. One factor per carrier BatchNorm, explicitly.
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
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
    (rf_pos _ _)))))))))))))))))))))

theorem Rr_continuous (nCls : Nat) : Continuous (Rr nCls) := by
  unfold Rr
  exact (rfac_cont 32 112 112 ⟨0, by norm_num⟩ 0 _ Zs_continuous).mul
    ((rfac_cont 32 112 112 ⟨0, by norm_num⟩ 0 _ (Z1d_continuous nCls)).mul
    ((rfac_cont 16 112 112 ⟨0, by norm_num⟩ 0 _ (Z1p_continuous nCls)).mul
    ((rfac_cont 96 (2 * 56) (2 * 56) ⟨0, by norm_num⟩ 0 _ (Z2e_continuous nCls)).mul
    ((rfac_cont 96 56 56 ⟨0, by norm_num⟩ 0 _ (Z2d_continuous nCls)).mul
    ((rfac_cont 24 56 56 ⟨0, by norm_num⟩ 0 _ (Z2p_continuous nCls)).mul
    ((rfac_cont 144 (2 * 28) (2 * 28) ⟨0, by norm_num⟩ 0 _ (Z4e_continuous nCls)).mul
    ((rfac_cont 144 28 28 ⟨0, by norm_num⟩ 0 _ (Z4d_continuous nCls)).mul
    ((rfac_cont 32 28 28 ⟨0, by norm_num⟩ 0 _ (Z4p_continuous nCls)).mul
    ((rfac_cont 192 (2 * 14) (2 * 14) ⟨0, by norm_num⟩ 0 _ (Z7e_continuous nCls)).mul
    ((rfac_cont 192 14 14 ⟨0, by norm_num⟩ 0 _ (Z7d_continuous nCls)).mul
    ((rfac_cont 64 14 14 ⟨0, by norm_num⟩ 0 _ (Z7p_continuous nCls)).mul
    ((rfac_cont 384 14 14 ⟨0, by norm_num⟩ 0 _ (Z11e_continuous nCls)).mul
    ((rfac_cont 384 14 14 ⟨0, by norm_num⟩ 0 _ (Z11d_continuous nCls)).mul
    ((rfac_cont 96 14 14 ⟨0, by norm_num⟩ 0 _ (Z11p_continuous nCls)).mul
    ((rfac_cont 576 (2 * 7) (2 * 7) ⟨0, by norm_num⟩ 0 _ (Z14e_continuous nCls)).mul
    ((rfac_cont 576 7 7 ⟨0, by norm_num⟩ 0 _ (Z14d_continuous nCls)).mul
    ((rfac_cont 160 7 7 ⟨0, by norm_num⟩ 0 _ (Z14p_continuous nCls)).mul
    ((rfac_cont 960 7 7 ⟨0, by norm_num⟩ 0 _ (Z17e_continuous nCls)).mul
    ((rfac_cont 960 7 7 ⟨0, by norm_num⟩ 0 _ (Z17d_continuous nCls)).mul
    ((rfac_cont 320 7 7 ⟨0, by norm_num⟩ 0 _ (Z17p_continuous nCls)).mul
    ((rfac_cont 1280 7 7 ⟨0, by norm_num⟩ 0 _ (Zh_continuous nCls)))))))))))))))))))))))

-- ════════════════════════════════════════════════════════════════
-- § 16. The head reads the carrier off channel 0
-- ════════════════════════════════════════════════════════════════
theorem sealW_fcW (nCls : Nat) :
    (sealW nCls).fcW = fun (i : Fin 1280) (j : Fin nCls) =>
      if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0 := rfl

theorem headA (nCls : Nat) (t : ℝ) :
    mnv2HeadB 2 7 7 (sealW nCls).hW (sealW nCls).hb (sealW nCls).hε (sealW nCls).hγ
        (sealW nCls).hβ (sealW nCls).fcW (sealW nCls).fcb (mnv2PreB17 2 (sealW nCls) (sealX t))
      = StableHLO.batchMap 2 (dense (sealW nCls).fcW (sealW nCls).fcb)
          (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7) (Ah nCls t)) := by
  show StableHLO.batchMap 2 (dense _ _) (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7)
      (StableHLO.cbrB 2 (h := 7) (w := 7) (ctK 1280 320 1 1 1) (kv 1280 0) 1 (kv 1280 (1 / 64))
        (kv 1280 3) (mnv2PreB17 2 (sealW nCls) (sealX t)))) = _
  rw [cbrB_eq _ _ (margin192 _ (by norm_num))]
  rfl

theorem head_diff (nCls : Nat) (hn : 0 < nCls) (v : Vec (2 * (1280 * 7 * 7)))
    (δ : Fin 1280 → ℝ) (hv : EDiff δ v) :
    StableHLO.batchMap 2 (dense (sealW nCls).fcW (sealW nCls).fcb)
        (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7) v)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - StableHLO.batchMap 2 (dense (sealW nCls).fcW (sealW nCls).fcb)
        (StableHLO.batchMap 2 (globalAvgPoolFlat 1280 7 7) v)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = δ 0 :=
  head_diff_ct (by norm_num) (by norm_num) (0 : Fin 1280) rfl ⟨0, hn⟩ _ _
    (fun ci => by rw [sealW_fcW]; simp) rfl v δ hv

/-- ⭐⭐ **The class-0 difference between the two examples, along the ray, is `t · Rr t`.** -/
theorem gd_ray (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    mobilenetv2ForwardB_full 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - mobilenetv2ForwardB_full 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = t * Rr nCls t := by
  rw [mobilenetv2ForwardB_full_eq_chain, Function.comp_apply, headA,
    head_diff nCls hn _ (dH nCls t) (edH nCls t)]
  simp only [dH, d17p, d17d, d17e, d14p, d14d, d14e, d11p, d11d, d11e, d7p, d7d, d7e, d4p, d4d, d4e, d2p, d2d, d2e, d1p, d1d, dS, Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 17. The seal
-- ════════════════════════════════════════════════════════════════
theorem sealDiffAt (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (mobilenetv2ForwardB_full 2 (sealW nCls)) (sealX t) := by
  rw [show mobilenetv2ForwardB_full 2 (sealW nCls)
      = mnv2HeadB 2 7 7 (sealW nCls).hW (sealW nCls).hb (sealW nCls).hε (sealW nCls).hγ
          (sealW nCls).hβ (sealW nCls).fcW (sealW nCls).fcb ∘ mnv2PreB17 2 (sealW nCls)
      from funext (mobilenetv2ForwardB_full_eq_chain 2 (sealW nCls))]
  have f0 : DifferentiableAt ℝ (mnv2PreB0 2 (sealW nCls)) (sealX t) :=
    mnv2StemB_differentiableAt 2 112 112 _ _ _ one_pos _ _ (sealX t) (scStem nCls t)
  have f1 : DifferentiableAt ℝ (mnv2PreB1 2 (sealW nCls)) (sealX t) :=
    (mnv2NoExpB_differentiableAt 2 112 112 (sealW nCls).b1 (sealNoExpPos 32 16) _ (sc1 nCls t)).comp (sealX t) f0
  have f2 : DifferentiableAt ℝ (mnv2PreB2 2 (sealW nCls)) (sealX t) :=
    (mnv2StridedB_differentiableAt 2 56 56 (sealW nCls).b2 (sealIVPos 16 96 24) _ (sc2 nCls t)).comp (sealX t) f1
  have f3 : DifferentiableAt ℝ (mnv2PreB3 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 56 56 (sealW nCls).b3 (sealResPos 24 144) _ (sc3 nCls t)).comp (sealX t) f2
  have f4 : DifferentiableAt ℝ (mnv2PreB4 2 (sealW nCls)) (sealX t) :=
    (mnv2StridedB_differentiableAt 2 28 28 (sealW nCls).b4 (sealIVPos 24 144 32) _ (sc4 nCls t)).comp (sealX t) f3
  have f5 : DifferentiableAt ℝ (mnv2PreB5 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 28 28 (sealW nCls).b5 (sealResPos 32 192) _ (sc5 nCls t)).comp (sealX t) f4
  have f6 : DifferentiableAt ℝ (mnv2PreB6 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 28 28 (sealW nCls).b6 (sealResPos 32 192) _ (sc6 nCls t)).comp (sealX t) f5
  have f7 : DifferentiableAt ℝ (mnv2PreB7 2 (sealW nCls)) (sealX t) :=
    (mnv2StridedB_differentiableAt 2 14 14 (sealW nCls).b7 (sealIVPos 32 192 64) _ (sc7 nCls t)).comp (sealX t) f6
  have f8 : DifferentiableAt ℝ (mnv2PreB8 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 14 14 (sealW nCls).b8 (sealResPos 64 384) _ (sc8 nCls t)).comp (sealX t) f7
  have f9 : DifferentiableAt ℝ (mnv2PreB9 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 14 14 (sealW nCls).b9 (sealResPos 64 384) _ (sc9 nCls t)).comp (sealX t) f8
  have f10 : DifferentiableAt ℝ (mnv2PreB10 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 14 14 (sealW nCls).b10 (sealResPos 64 384) _ (sc10 nCls t)).comp (sealX t) f9
  have f11 : DifferentiableAt ℝ (mnv2PreB11 2 (sealW nCls)) (sealX t) :=
    (mnv2ExpOnlyB_differentiableAt 2 14 14 (sealW nCls).b11 (sealIVPos 64 384 96) _ (sc11 nCls t)).comp (sealX t) f10
  have f12 : DifferentiableAt ℝ (mnv2PreB12 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 14 14 (sealW nCls).b12 (sealResPos 96 576) _ (sc12 nCls t)).comp (sealX t) f11
  have f13 : DifferentiableAt ℝ (mnv2PreB13 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 14 14 (sealW nCls).b13 (sealResPos 96 576) _ (sc13 nCls t)).comp (sealX t) f12
  have f14 : DifferentiableAt ℝ (mnv2PreB14 2 (sealW nCls)) (sealX t) :=
    (mnv2StridedB_differentiableAt 2 7 7 (sealW nCls).b14 (sealIVPos 96 576 160) _ (sc14 nCls t)).comp (sealX t) f13
  have f15 : DifferentiableAt ℝ (mnv2PreB15 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 7 7 (sealW nCls).b15 (sealResPos 160 960) _ (sc15 nCls t)).comp (sealX t) f14
  have f16 : DifferentiableAt ℝ (mnv2PreB16 2 (sealW nCls)) (sealX t) :=
    (mnv2ResidB_differentiableAt 2 7 7 (sealW nCls).b16 (sealResPos 160 960) _ (sc16 nCls t)).comp (sealX t) f15
  have f17 : DifferentiableAt ℝ (mnv2PreB17 2 (sealW nCls)) (sealX t) :=
    (mnv2ExpOnlyB_differentiableAt 2 7 7 (sealW nCls).b17 (sealIVPos 160 960 320) _ (sc17 nCls t)).comp (sealX t) f16
  exact (mnv2HeadB_differentiableAt 2 7 7 _ _ _ one_pos _ _ _ _ _ (scHead nCls t)).comp
    (sealX t) f17

/-- ⭐⭐ **Level 2 — the witness is non-degenerate**: the full-width batch-BN MobileNetV2 at the
    structural weights is NOT constant in its input. -/
theorem sealX_nonconstant (nCls : Nat) (hn : 0 < nCls) :
    mobilenetv2ForwardB_full 2 (sealW nCls) (sealX 1)
      ≠ mobilenetv2ForwardB_full 2 (sealW nCls) (sealX 0) := by
  intro heq
  have h1 := gd_ray nCls hn 1
  have h0 := gd_ray nCls hn 0
  rw [heq] at h1
  have hz : (1 : ℝ) * Rr nCls 1 = 0 * Rr nCls 0 := by rw [← h1, ← h0]
  rw [one_mul, zero_mul] at hz
  linarith [Rr_pos nCls 1]

/-- ⭐⭐ **Level 3 — the whole-net Jacobian is nonzero at the witness.** -/
theorem sealX_jacobian_nonzero (nCls : Nat) (hn : 0 < nCls) :
    fderiv ℝ (mobilenetv2ForwardB_full 2 (sealW nCls)) (sealX 0) ≠ 0 := by
  refine fderiv_ne_zero_of_ray sealV (sealDiffAt nCls 0)
    (fun y => y (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - y (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))) (by fun_prop)
    (Rr_pos nCls 0).ne' ?_
  have heq : (fun t : ℝ => mobilenetv2ForwardB_full 2 (sealW nCls) (sealX 0 + t • sealV)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - mobilenetv2ForwardB_full 2 (sealW nCls) (sealX 0 + t • sealV)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls))))
      = fun t : ℝ => t * Rr nCls t := by
    funext t
    rw [sealX_zero_add]
    exact gd_ray nCls hn t
  rw [heq]
  exact hasDerivAt_mul_self_zero (Rr_continuous nCls).continuousAt

/-- ⭐⭐ **The seal**: the proven whole-network backward of the full-width, batch-BatchNorm,
    seventeen-bottleneck, 224×224 MobileNetV2 — `mobilenetv2ForwardB_full`, the forward every
    MobileNetV2 artifact runs — is **not the zero map** at the witness. -/
theorem sealX_backward_nontrivial (nCls : Nat) (hn : 0 < nCls) :
    ∃ (j₀ : Fin (2 * nCls)) (i₀ : Fin (2 * (3 * (2 * 112) * (2 * 112)))),
      (sealVJP nCls 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (sealVJP nCls 0).backward_nontrivial_of_fderiv_ne (sealX_jacobian_nonzero nCls hn)

end Mnv2FullBSeal
end Proofs
