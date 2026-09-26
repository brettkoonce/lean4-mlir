import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP
import LeanMlir.Proofs.Training.BatchSealKit
import LeanMlir.Proofs.Training.JacobianSeal

/-!
# ResNet-34's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`ResNet34FullBVJP.lean` proves
`resnet34ForwardBFullHasVJPAt`: the whole-net VJP at any `(w, x)` that satisfies 32 relu
clauses, a stem clause and the stem pool's no-tie. A *conditional* theorem of that shape says
nothing unless its hypotheses are jointly satisfiable at a point with a nonzero Jacobian. This
file exhibits such a point on `resnet34ForwardBFull` itself: 64→512 channels, `[3,4,6,3]`
blocks, the 7×7/s2 stem, the 3×3/s2 pool, **batch** BatchNorm, at `224×224`.

## The witness

Weights are *structural*, not trained (a trained-weight witness would be a numeric fact about
millions of floats):

* every residual body is zeroed, so it is the constant `β₂ = 1` whatever `γ₂` is
  (`BatchSeal.bnBatchLA_const`: a constant channel has variance 0, so batch BN returns `β`). On a
  nonnegative activation the identity block is therefore the affine shift `a ↦ a + 1`
  (`sealIdB_eq`) — and the shift is batch-uniform, hence invisible to the carrier;
* every channel-changing conv (the stem, the three 1×1/s2 projections) is a **centre-tap
  broadcast** (`BatchSeal.ctK`): every output channel is a copy of input channel 0 read through the
  kernel's centre tap. Centre tap, not a general kernel, because `conv2d` zero-pads: a conv of a
  constant is not constant at the border, but the centre tap is in range at every cell. And a
  *broadcast*, not a diagonal on channel 0: a kernel feeding only output channel 0 leaves
  the stem's other 63 channels constant, and a constant channel ties every 3×3 window of the pool
  — `StemPoolSmoothAt` quantifies over channels, so the tap has to reach all of them;
* `γ = 1`, `β = 160` at the stem and the projections, `ε = 1` everywhere. `√(N·h·w) ≤ √25088 < 160`
  at every one of those four BN widths, so `BatchSeal.bnBatchLA_pos` puts **every** relu strictly
  off its kink at **every** input — the net needs no eventually-argument for its relus, and the
  only genuine kink left is the stem pool;
* the head reads channel 0 into class 0 (`Wd 0 0 = 1`, `bd = 0`), so `0 < nCls` is the only
  constraint on the class count.

`N = 2` and the input is `sealX t = sealBase + t • sealV`: both examples carry the same strictly
decreasing ramp in channel 0, and `sealV` adds `t` to **all of example 0's channel 0**.

## Why the carrier is a batch difference

`bnBatchLA` normalizes each channel over all `N·h·w` cells, so a within-example channel-difference
carrier is exactly what a channel's own mean subtracts, and at `N = 1` this net is constant in its
input. The carrier here is `EDiff`: example 0's slab is example 1's plus a per-channel constant.
Batch BN keeps it and scales it by `γ_c · istd_c` (`BatchSeal.bnBatchLA_exdiff`) because the two
examples share one mean and one `istd`; the centre-tap convs copy channel 0's offset to every
channel; the pool shifts with it (`maxPool3s2_shift`, for every `t`, no argmax argument);
the zeroed bodies add a batch-uniform constant and so are transparent. The class-0 output
difference between the two examples is therefore `t · R t` with `R` a product of the four
carrier-path `istd`s, continuous and positive, so `g'(0) = R 0 ≠ 0` and no BN-variance derivative
is ever taken.
-/

namespace Proofs
namespace R34FullBSeal
open Proofs.BatchSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs BatchSeal

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural weights
-- ════════════════════════════════════════════════════════════════

/-- The structural identity-block weights: both convs zeroed, every BN `(ε, γ, β) = (1, 1, 1)`.
    The body is then the constant `1` and the block is `a ↦ a + 1` on a nonnegative activation. -/
noncomputable def sealIdW (c : Nat) : R34IdW c where
  W₁ := zk c c 3 3
  b₁ := kv c 0
  ε₁ := 1
  γ₁ := kv c 1
  β₁ := kv c 1
  W₂ := zk c c 3 3
  b₂ := kv c 0
  ε₂ := 1
  γ₂ := kv c 1
  β₂ := kv c 1

/-- The structural downsample weights: zeroed body, centre-tap 1×1/s2 projection, `β_p = 160`
    (the margin that keeps the post-residual relu off its kink at every input). -/
noncomputable def sealDnW (ic oc : Nat) : R34DownW ic oc where
  W₁ := zk oc ic 3 3
  b₁ := kv oc 0
  ε₁ := 1
  γ₁ := kv oc 1
  β₁ := kv oc 1
  W₂ := zk oc oc 3 3
  b₂ := kv oc 0
  ε₂ := 1
  γ₂ := kv oc 1
  β₂ := kv oc 1
  Wp := ctK oc ic 1 1 1
  bp := kv oc 0
  εp := 1
  γp := kv oc 1
  βp := kv oc 160

/-- **The witness weights**, generic in the class count. -/
noncomputable def sealW (nCls : Nat) : R34BWeights nCls where
  sW := ctK 64 3 7 7 1
  sb := kv 64 0
  sε := 1
  sγ := kv 64 1
  sβ := kv 64 160
  a0 := sealIdW 64
  a1 := sealIdW 64
  a2 := sealIdW 64
  d2 := sealDnW 64 128
  b0 := sealIdW 128
  b1 := sealIdW 128
  b2 := sealIdW 128
  d3 := sealDnW 128 256
  c0 := sealIdW 256
  c1 := sealIdW 256
  c2 := sealIdW 256
  c3 := sealIdW 256
  c4 := sealIdW 256
  d4 := sealDnW 256 512
  e0 := sealIdW 512
  e1 := sealIdW 512
  Wd := fun i j => if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0
  bd := kv nCls 0

-- ════════════════════════════════════════════════════════════════
-- § 2. What the structural blocks are
-- ════════════════════════════════════════════════════════════════
-- ⚠⚠ **Every collapse lemma below is stated at VARIABLE `N, h, w, c` and instantiated at the
-- witness's numerals afterwards, never proved at them.** `relu_id_of_pos` applied directly to, say,
-- `cbReluStridedB 2 (h := 2*56) …` leaves the KERNEL a defeq between two numeral-shaped
-- compositions and it dies ("deep recursion" at `oc = 64, h = 112`, and a timeout already at
-- `h = 16`). At variables the same proof is instant, and instantiating a proved lemma is
-- substitution — no defeq at all. Same lesson as `maxPool3s2`'s header records for `rfl`.

/-- The identity block's body is the constant `1`, at every input. -/
theorem seal_id_body (N h w c : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w))) :
    (projB N (h := h) (w := w) (sealIdW c).W₂ (sealIdW c).b₂ (sealIdW c).ε₂
        (sealIdW c).γ₂ (sealIdW c).β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) (sealIdW c).W₁ (sealIdW c).b₁ (sealIdW c).ε₁
        (sealIdW c).γ₁ (sealIdW c).β₁) v = fun _ => (1 : ℝ) :=
  projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1 (fun _ => rfl) _

/-- The downsample block's body is the constant `1`, at every input. -/
theorem seal_dn_body (N h w ic oc : Nat) (hn : 0 < N * (h * w))
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    (projB N (h := h) (w := w) (sealDnW ic oc).W₂ (sealDnW ic oc).b₂ (sealDnW ic oc).ε₂
        (sealDnW ic oc).γ₂ (sealDnW ic oc).β₂ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) (sealDnW ic oc).W₁ (sealDnW ic oc).b₁
        (sealDnW ic oc).ε₁ (sealDnW ic oc).γ₁ (sealDnW ic oc).β₁) v = fun _ => (1 : ℝ) :=
  projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1 (fun _ => rfl) _

/-- **The structural identity block is the shift `a ↦ a + 1`** on a nonnegative activation: the
    body is the constant `1` and the post-residual relu is off (`1 + a ≥ 1 > 0`). -/
theorem sealIdB_eq (N h w c : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w)))
    (hv : ∀ k, 0 ≤ v k) :
    r34IdB N h w (sealIdW c) v = fun k => v k + 1 := by
  have hbody := seal_id_body N h w c hn v
  have hres : ∀ k, residual
      (projB N (h := h) (w := w) (sealIdW c).W₂ (sealIdW c).b₂ (sealIdW c).ε₂
          (sealIdW c).γ₂ (sealIdW c).β₂ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealIdW c).W₁ (sealIdW c).b₁ (sealIdW c).ε₁
          (sealIdW c).γ₁ (sealIdW c).β₁) v k = v k + 1 := by
    intro k
    rw [residual_apply, hbody]
    ring
  funext k
  show relu (N * (c * h * w)) (residual _ v) k = v k + 1
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [hv i]), hres k]

/-- **The structural downsample is its projection plus one**: the body is the constant `1` and
    the post-residual relu is off (`proj > 0`). -/
theorem sealDnB_eq (N h w ic oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    r34DownB N h w (sealDnW ic oc) v = fun k => sealProj N h w ic oc v k + 1 := by
  have hbody := seal_dn_body N h w ic oc hn v
  have hres : ∀ k, residualProj
      (StableHLO.projStridedB N (h := h) (w := w) (sealDnW ic oc).Wp (sealDnW ic oc).bp
        (sealDnW ic oc).εp (sealDnW ic oc).γp (sealDnW ic oc).βp)
      (projB N (h := h) (w := w) (sealDnW ic oc).W₂ (sealDnW ic oc).b₂
          (sealDnW ic oc).ε₂ (sealDnW ic oc).γ₂ (sealDnW ic oc).β₂ ∘
        StableHLO.cbReluStridedB N (h := h) (w := w) (sealDnW ic oc).W₁ (sealDnW ic oc).b₁
          (sealDnW ic oc).ε₁ (sealDnW ic oc).γ₁ (sealDnW ic oc).β₁) v k
      = sealProj N h w ic oc v k + 1 := by
    intro k
    show sealProj N h w ic oc v k + _ = _
    rw [hbody]
  funext k
  show relu (N * (oc * h * w)) (residualProj _ _ v) k = sealProj N h w ic oc v k + 1
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [sealProj_pos N h w ic oc hm v i]),
    hres k]

/-- **The stem with its relu removed**: pool ∘ bn ∘ strided conv. The pool stays — it is the
    net's only remaining kink, and the carrier crosses it by `maxPool3s2_shift`. -/
theorem r34StemB_eq {N h w ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hp : ∀ k, 0 < StableHLO.bnBatchLA N oc (2 * h) (2 * w) εs γs βs
      (StableHLO.batchMap N (flatConvStride2 Ws bs) x) k) :
    r34StemB N h w Ws bs εs γs βs x
      = StableHLO.batchMap N (maxPool3s2Flat oc h w)
          (StableHLO.bnBatchLA N oc (2 * h) (2 * w) εs γs βs
            (StableHLO.batchMap N (flatConvStride2 Ws bs) x)) := by
  show StableHLO.batchMap N (maxPool3s2Flat oc h w)
      (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) = _
  rw [cbReluStridedB_eq Ws bs εs γs βs x hp]

-- ════════════════════════════════════════════════════════════════
-- § 3. Every running activation is nonnegative
-- ════════════════════════════════════════════════════════════════

/-- A block output is a relu, hence nonnegative — whatever the weights. -/
theorem r34IdB_nonneg (N h w c : Nat) (p : R34IdW c) (v : Vec (N * (c * h * w)))
    (k : Fin (N * (c * h * w))) : 0 ≤ r34IdB N h w p v k := relu_nonneg _ _ k

theorem r34DownB_nonneg (N h w ic oc : Nat) (p : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (k : Fin (N * (oc * h * w))) :
    0 ≤ r34DownB N h w p v k := relu_nonneg _ _ k

/-- The stem's pool preserves the relu's nonnegativity. -/
theorem r34StemB_nonneg (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (k : Fin (N * (oc * h * w))) :
    0 ≤ r34StemB N h w Ws bs εs γs βs x k := by
  show 0 ≤ StableHLO.batchMap N (maxPool3s2Flat oc h w)
    (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x) k
  refine forall_flat_of_cell (P := fun z => 0 ≤ z) ?_ k
  intro n ci i j
  rw [bcell_pool]
  refine maxPool3s2_nonneg _ (fun ci' r s => ?_) _ _ _
  rw [bcell_eq_laIdx]
  exact relu_nonneg _ _ _

-- ════════════════════════════════════════════════════════════════
-- § 4. The clause bundles at the structural weights
-- ════════════════════════════════════════════════════════════════

/-- Both `ε`s of a structural identity block are positive. -/
theorem seal_id_pos (c : Nat) : R34IdPos (sealIdW c) := ⟨one_pos, one_pos⟩

/-- All three `ε`s of a structural downsample are positive. -/
theorem seal_dn_pos (ic oc : Nat) : R34DownPos (sealDnW ic oc) := ⟨one_pos, one_pos, one_pos⟩

/-- **The identity block's two relu clauses**: the mid-relu sees the constant `β₁ = 1`
    (weight-only), the outer one sees `1 + activation > 0`. -/
theorem seal_id_smooth (N h w c : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w)))
    (hv : ∀ k, 0 ≤ v k) : R34IdSmoothAt N h w (sealIdW c) v where
  hmid := by
    intro k
    show StableHLO.bnBatchLA N c h w 1 (kv c 1) (kv c 1)
      (StableHLO.batchMap N (flatConv (zk c c 3 3) (kv c 0)) v) k ≠ 0
    rw [batchMap_flatConv_zero (zk c c 3 3) (kv c 0) (fun _ _ _ _ => rfl) (fun _ => rfl),
      bnBatchLA_const hn 1 (kv c 1) (kv c 1) 1 0 (fun _ => rfl) k]
    norm_num
  hout := by
    intro k
    show _ + v k ≠ 0
    rw [congrFun (seal_id_body N h w c hn v) k]
    intro hc
    linarith [hv k]

/-- **The downsample's two relu clauses** — both weight-only: the mid-relu sees `β₁ = 1`, the
    outer one `proj + 1 > 0` with `proj > 0` by the margin. No hypothesis on the activation. -/
theorem seal_dn_smooth (N h w ic oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : R34DownSmoothAt N h w (sealDnW ic oc) v where
  hmid := by
    intro k
    show StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 1)
      (StableHLO.batchMap N (flatConvStride2 (zk oc ic 3 3) (kv oc 0)) v) k ≠ 0
    rw [batchMap_flatConvStride2_zero (zk oc ic 3 3) (kv oc 0) (fun _ _ _ _ => rfl)
        (fun _ => rfl),
      bnBatchLA_const hn 1 (kv oc 1) (kv oc 1) 1 0 (fun _ => rfl) k]
    norm_num
  hout := by
    intro k
    show sealProj N h w ic oc v k + _ ≠ 0
    rw [congrFun (seal_dn_body N h w ic oc hn v) k]
    intro hc
    linarith [sealProj_pos N h w ic oc hm v k]

/-- The stem's relu clause — weight-only, from the `β = 160` margin. -/
theorem seal_stem_smooth (N h w ic oc : Nat) (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (hm : |(1 : ℝ)| * Real.sqrt ((N * ((2 * h) * (2 * w)) : ℕ) : ℝ) < 160)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    R34StemSmoothAt N h w Ws bs 1 (kv oc 1) (kv oc 160) x :=
  fun k => (bnBatchLA_pos 1 one_pos (kv oc 1) (kv oc 160) 1 160 (fun _ => rfl) (fun _ => rfl)
    hm (StableHLO.batchMap N (flatConvStride2 Ws bs) x) k).ne'

-- ════════════════════════════════════════════════════════════════
-- § 5. The witness input and the carrier, at this net's spelling
--   Everything here is `BatchSeal`'s, instantiated: the ray (`rayX`), the carrier (`EDiff` and its
--   per-op steps) and the stem's centre-tap conv with the pool's no-tie (`ctConv*`) are shared with
--   ResNet-50, which runs the same 7×7/s2 stem at a different spatial nest.
-- ════════════════════════════════════════════════════════════════

/-- The witness input: the shared ray at 224×224. -/
noncomputable def sealX (t : ℝ) : Vec (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  rayX (2 * (2 * 56)) (2 * (2 * 56)) t

/-- Its direction — all of example 0's channel 0. -/
noncomputable def sealV : Vec (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  rayV (2 * (2 * 56)) (2 * (2 * 56))

theorem sealX_zero_add (t : ℝ) : sealX 0 + t • sealV = sealX t := by
  rw [sealX, sealX, sealV]
  exact rayX_zero_add _ _ t

theorem eDiff_sealX (t : ℝ) : EDiff (fun ci => if ci.val = 0 then t else 0) (sealX t) := by
  rw [sealX]
  exact eDiff_rayX _ _ t

/-- The stem's centre-tap conv output — the pre-BN activation on the carrier's path. -/
noncomputable def Zs (t : ℝ) : Vec (2 * (64 * (2 * 56) * (2 * 56))) := ctConv 64 7 7 56 56 t

theorem margin_stem : |(1 : ℝ)| * Real.sqrt ((2 * ((2 * 56) * (2 * 56)) : ℕ) : ℝ) < 160 :=
  margin160 _ (by norm_num)

theorem margin28 : |(1 : ℝ)| * Real.sqrt ((2 * (28 * 28) : ℕ) : ℝ) < 160 :=
  margin160 _ (by norm_num)

theorem margin14 : |(1 : ℝ)| * Real.sqrt ((2 * (14 * 14) : ℕ) : ℝ) < 160 :=
  margin160 _ (by norm_num)

theorem margin7 : |(1 : ℝ)| * Real.sqrt ((2 * (7 * 7) : ℕ) : ℝ) < 160 :=
  margin160 _ (by norm_num)

/-- The stem BN is strictly positive at every point of the ray. -/
theorem Zs_bn_pos (t : ℝ) (k : Fin (2 * (64 * (2 * 56) * (2 * 56)))) :
    0 < StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t) k := by
  rw [Zs]
  exact ctConv_bn_pos 64 7 7 56 56 margin_stem t k

/-- The stem pool has no tie at the witness — the ramp is positionally injective and BN is
    injective within a channel. -/
theorem seal_pool_smooth (t : ℝ) :
    StemPoolSmoothAt 2 56 56
      (StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t)) := by
  rw [Zs]
  exact ctConv_pool_smooth 64 7 7 56 56 (by norm_num) (by norm_num) t

-- ════════════════════════════════════════════════════════════════
-- § 6. The running activations: nonnegative, and collapsed
--   ⚠ Each `nn`/`pc` below INSTANTIATES a §3/§4 lemma proved at variable shapes. Proving any of
--   them at these numerals directly is what kills the kernel (see §2's banner).
-- ════════════════════════════════════════════════════════════════

/-- The stem's output is nonnegative (a pool of a relu). -/
theorem nn0 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre0 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre0_apply]
  exact r34StemB_nonneg 2 56 56 _ _ _ _ _ (sealX t) k

/-- The stem, collapsed: its relu is off, so it is pool ∘ bn ∘ centre-tap conv. -/
theorem pc0 (nCls : Nat) (t : ℝ) :
    r34Pre0 2 (sealW nCls) (sealX t)
      = StableHLO.batchMap 2 (maxPool3s2Flat 64 56 56)
          (StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t)) := by
  rw [r34Pre0_apply]
  exact r34StemB_eq _ _ _ _ _ (sealX t) (fun k => Zs_bn_pos t k)

theorem nn1 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre1 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre1_apply]
  exact r34IdB_nonneg 2 56 56 64 _ _ k

theorem nn2 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre2 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre2_apply]
  exact r34IdB_nonneg 2 56 56 64 _ _ k

theorem nn4 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre4 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre4_apply]
  exact r34DownB_nonneg 2 28 28 64 128 _ _ k

theorem nn5 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre5 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre5_apply]
  exact r34IdB_nonneg 2 28 28 128 _ _ k

theorem nn6 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre6 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre6_apply]
  exact r34IdB_nonneg 2 28 28 128 _ _ k

theorem nn8 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre8 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre8_apply]
  exact r34DownB_nonneg 2 14 14 128 256 _ _ k

theorem nn9 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre9 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre9_apply]
  exact r34IdB_nonneg 2 14 14 256 _ _ k

theorem nn10 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre10 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre10_apply]
  exact r34IdB_nonneg 2 14 14 256 _ _ k

theorem nn11 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre11 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre11_apply]
  exact r34IdB_nonneg 2 14 14 256 _ _ k

theorem nn12 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre12 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre12_apply]
  exact r34IdB_nonneg 2 14 14 256 _ _ k

theorem nn14 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre14 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre14_apply]
  exact r34DownB_nonneg 2 7 7 256 512 _ _ k

theorem nn15 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre15 2 (sealW nCls) (sealX t) k := by
  intro k
  rw [r34Pre15_apply]
  exact r34IdB_nonneg 2 7 7 512 _ _ k

theorem pc1 (nCls : Nat) (t : ℝ) :
    r34Pre1 2 (sealW nCls) (sealX t) = fun k => r34Pre0 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre1_apply]
  exact sealIdB_eq 2 56 56 64 (by norm_num) _ (nn0 nCls t)

theorem pc2 (nCls : Nat) (t : ℝ) :
    r34Pre2 2 (sealW nCls) (sealX t) = fun k => r34Pre1 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre2_apply]
  exact sealIdB_eq 2 56 56 64 (by norm_num) _ (nn1 nCls t)

theorem pc3 (nCls : Nat) (t : ℝ) :
    r34Pre3 2 (sealW nCls) (sealX t) = fun k => r34Pre2 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre3_apply]
  exact sealIdB_eq 2 56 56 64 (by norm_num) _ (nn2 nCls t)

theorem pc4 (nCls : Nat) (t : ℝ) :
    r34Pre4 2 (sealW nCls) (sealX t)
      = fun k => sealProj 2 28 28 64 128 (r34Pre3 2 (sealW nCls) (sealX t)) k + 1 := by
  rw [r34Pre4_apply]
  exact sealDnB_eq 2 28 28 64 128 (by norm_num) margin28 _

theorem pc5 (nCls : Nat) (t : ℝ) :
    r34Pre5 2 (sealW nCls) (sealX t) = fun k => r34Pre4 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre5_apply]
  exact sealIdB_eq 2 28 28 128 (by norm_num) _ (nn4 nCls t)

theorem pc6 (nCls : Nat) (t : ℝ) :
    r34Pre6 2 (sealW nCls) (sealX t) = fun k => r34Pre5 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre6_apply]
  exact sealIdB_eq 2 28 28 128 (by norm_num) _ (nn5 nCls t)

theorem pc7 (nCls : Nat) (t : ℝ) :
    r34Pre7 2 (sealW nCls) (sealX t) = fun k => r34Pre6 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre7_apply]
  exact sealIdB_eq 2 28 28 128 (by norm_num) _ (nn6 nCls t)

theorem pc8 (nCls : Nat) (t : ℝ) :
    r34Pre8 2 (sealW nCls) (sealX t)
      = fun k => sealProj 2 14 14 128 256 (r34Pre7 2 (sealW nCls) (sealX t)) k + 1 := by
  rw [r34Pre8_apply]
  exact sealDnB_eq 2 14 14 128 256 (by norm_num) margin14 _

theorem pc9 (nCls : Nat) (t : ℝ) :
    r34Pre9 2 (sealW nCls) (sealX t) = fun k => r34Pre8 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre9_apply]
  exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn8 nCls t)

theorem pc10 (nCls : Nat) (t : ℝ) :
    r34Pre10 2 (sealW nCls) (sealX t) = fun k => r34Pre9 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre10_apply]
  exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn9 nCls t)

theorem pc11 (nCls : Nat) (t : ℝ) :
    r34Pre11 2 (sealW nCls) (sealX t) = fun k => r34Pre10 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre11_apply]
  exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn10 nCls t)

theorem pc12 (nCls : Nat) (t : ℝ) :
    r34Pre12 2 (sealW nCls) (sealX t) = fun k => r34Pre11 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre12_apply]
  exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn11 nCls t)

theorem pc13 (nCls : Nat) (t : ℝ) :
    r34Pre13 2 (sealW nCls) (sealX t) = fun k => r34Pre12 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre13_apply]
  exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn12 nCls t)

theorem pc14 (nCls : Nat) (t : ℝ) :
    r34Pre14 2 (sealW nCls) (sealX t)
      = fun k => sealProj 2 7 7 256 512 (r34Pre13 2 (sealW nCls) (sealX t)) k + 1 := by
  rw [r34Pre14_apply]
  exact sealDnB_eq 2 7 7 256 512 (by norm_num) margin7 _

theorem pc15 (nCls : Nat) (t : ℝ) :
    r34Pre15 2 (sealW nCls) (sealX t) = fun k => r34Pre14 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre15_apply]
  exact sealIdB_eq 2 7 7 512 (by norm_num) _ (nn14 nCls t)

theorem pc16 (nCls : Nat) (t : ℝ) :
    r34Pre16 2 (sealW nCls) (sealX t) = fun k => r34Pre15 2 (sealW nCls) (sealX t) k + 1 := by
  rw [r34Pre16_apply]
  exact sealIdB_eq 2 7 7 512 (by norm_num) _ (nn15 nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 7. The whole-net VJP at the witness
-- ════════════════════════════════════════════════════════════════

/-- The stem's relu is off at the witness, so the pool's no-tie condition can be stated on the
    BN output (`seal_pool_smooth`). -/
theorem stem_relu_off (nCls : Nat) (t : ℝ) :
    StableHLO.cbReluStridedB 2 (h := 2 * 56) (w := 2 * 56) (sealW nCls).sW (sealW nCls).sb
        (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX t)
      = StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t) :=
  cbReluStridedB_eq _ _ _ _ _ (sealX t) (fun k => Zs_bn_pos t k)

theorem seal_pool_clause (nCls : Nat) (t : ℝ) :
    StemPoolSmoothAt 2 56 56 (StableHLO.cbReluStridedB 2 (h := 2 * 56) (w := 2 * 56)
      (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ
        (sealX t)) := by
  rw [stem_relu_off]
  exact seal_pool_smooth t

theorem seal_stem_clause (nCls : Nat) (t : ℝ) :
    R34StemSmoothAt 2 56 56 (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ
      (sealW nCls).sβ (sealX t) :=
  seal_stem_smooth 2 56 56 3 64 _ _ margin_stem (sealX t)

theorem sc_a0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 56 56 (sealW nCls).a0 (r34Pre0 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 56 56 64 (by norm_num) _ (nn0 nCls t)

theorem sc_a1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 56 56 (sealW nCls).a1 (r34Pre1 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 56 56 64 (by norm_num) _ (nn1 nCls t)

theorem sc_a2 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 56 56 (sealW nCls).a2 (r34Pre2 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 56 56 64 (by norm_num) _ (nn2 nCls t)

theorem sc_d2 (nCls : Nat) (t : ℝ) :
    R34DownSmoothAt 2 28 28 (sealW nCls).d2 (r34Pre3 2 (sealW nCls) (sealX t)) :=
  seal_dn_smooth 2 28 28 64 128 (by norm_num) margin28 _

theorem sc_b0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 28 28 (sealW nCls).b0 (r34Pre4 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 28 28 128 (by norm_num) _ (nn4 nCls t)

theorem sc_b1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 28 28 (sealW nCls).b1 (r34Pre5 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 28 28 128 (by norm_num) _ (nn5 nCls t)

theorem sc_b2 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 28 28 (sealW nCls).b2 (r34Pre6 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 28 28 128 (by norm_num) _ (nn6 nCls t)

theorem sc_d3 (nCls : Nat) (t : ℝ) :
    R34DownSmoothAt 2 14 14 (sealW nCls).d3 (r34Pre7 2 (sealW nCls) (sealX t)) :=
  seal_dn_smooth 2 14 14 128 256 (by norm_num) margin14 _

theorem sc_c0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c0 (r34Pre8 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 14 14 256 (by norm_num) _ (nn8 nCls t)

theorem sc_c1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c1 (r34Pre9 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 14 14 256 (by norm_num) _ (nn9 nCls t)

theorem sc_c2 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c2 (r34Pre10 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 14 14 256 (by norm_num) _ (nn10 nCls t)

theorem sc_c3 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c3 (r34Pre11 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 14 14 256 (by norm_num) _ (nn11 nCls t)

theorem sc_c4 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c4 (r34Pre12 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 14 14 256 (by norm_num) _ (nn12 nCls t)

theorem sc_d4 (nCls : Nat) (t : ℝ) :
    R34DownSmoothAt 2 7 7 (sealW nCls).d4 (r34Pre13 2 (sealW nCls) (sealX t)) :=
  seal_dn_smooth 2 7 7 256 512 (by norm_num) margin7 _

theorem sc_e0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 7 7 (sealW nCls).e0 (r34Pre14 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 7 7 512 (by norm_num) _ (nn14 nCls t)

theorem sc_e1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 7 7 (sealW nCls).e1 (r34Pre15 2 (sealW nCls) (sealX t)) :=
  seal_id_smooth 2 7 7 512 (by norm_num) _ (nn15 nCls t)

/-- Every BN `ε` of the witness is `1`. -/
theorem seal_pos (nCls : Nat) : R34PosB (sealW nCls) :=
  ⟨one_pos, seal_id_pos 64, seal_id_pos 64, seal_id_pos 64, seal_dn_pos 64 128,
    seal_id_pos 128, seal_id_pos 128, seal_id_pos 128, seal_dn_pos 128 256,
    seal_id_pos 256, seal_id_pos 256, seal_id_pos 256, seal_id_pos 256, seal_id_pos 256,
    seal_dn_pos 256 512, seal_id_pos 512, seal_id_pos 512⟩

/-- The stem clause, the pool's no-tie and all 32 relu clauses at `(sealW nCls, sealX t)`. -/
theorem seal_smooth (nCls : Nat) (t : ℝ) : R34SmoothAtB 2 (sealW nCls) (sealX t) :=
  ⟨seal_stem_clause nCls t, seal_pool_clause nCls t,
    sc_a0 nCls t, sc_a1 nCls t, sc_a2 nCls t, sc_d2 nCls t,
    sc_b0 nCls t, sc_b1 nCls t, sc_b2 nCls t, sc_d3 nCls t,
    sc_c0 nCls t, sc_c1 nCls t, sc_c2 nCls t, sc_c3 nCls t, sc_c4 nCls t,
    sc_d4 nCls t, sc_e0 nCls t, sc_e1 nCls t⟩

/-- **The whole-net VJP at the witness** — every one of the 32 relu clauses, the stem clause
    and the pool's no-tie discharged at `(sealW nCls, sealX t)`, on `resnet34ForwardBFull`
    itself (transported through `resnet34ForwardBFull_eq_chain`). -/
noncomputable def sealVJP (nCls : Nat) (t : ℝ) :
    HasVJPAt (resnet34ForwardBFull 2 (sealW nCls)) (sealX t) := by
  rw [show resnet34ForwardBFull 2 (sealW nCls)
      = r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd ∘ r34Pre16 2 (sealW nCls)
      from funext (resnet34ForwardBFull_eq_chain 2 (sealW nCls))]
  exact resnet34ForwardBFullHasVJPAt 2 (sealW nCls) (seal_pos nCls) (sealX t)
    (seal_smooth nCls t)

/-- The net is differentiable at the witness — `fderiv_ne_zero_of_ray`'s first hypothesis. -/
theorem seal_differentiableAt (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (resnet34ForwardBFull 2 (sealW nCls)) (sealX t) :=
  resnet34ForwardBFull_differentiableAt 2 (sealW nCls) (seal_pos nCls) (sealX t)
    (seal_smooth nCls t)

-- ════════════════════════════════════════════════════════════════
-- § 8. The carrier along the ray
--   Four BN sites lie on the carrier's path (the stem and the three projections); the thirteen
--   identity blocks contribute a batch-uniform `+1` each, which the carrier does not see. So
--   `EDiff` takes only four distinct values down the whole trunk.
-- ════════════════════════════════════════════════════════════════

/-- The three projections' pre-BN activations, on the carrier's path. -/
noncomputable def Zp2 (nCls : Nat) (t : ℝ) : Vec (2 * (128 * 28 * 28)) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 128 64 1 1 1) (kv 128 0))
    (r34Pre3 2 (sealW nCls) (sealX t))

noncomputable def Zp3 (nCls : Nat) (t : ℝ) : Vec (2 * (256 * 14 * 14)) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 256 128 1 1 1) (kv 256 0))
    (r34Pre7 2 (sealW nCls) (sealX t))

noncomputable def Zp4 (nCls : Nat) (t : ℝ) : Vec (2 * (512 * 7 * 7)) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 512 256 1 1 1) (kv 512 0))
    (r34Pre13 2 (sealW nCls) (sealX t))

/-- The carrier after the stem BN: `t` scaled by that BN's `istd`. -/
noncomputable def dS (t : ℝ) : Fin 64 → ℝ :=
  fun ci => t * bnIstd (2 * ((2 * 56) * (2 * 56))) (bnRowLA 2 64 (2 * 56) (2 * 56) (Zs t) ci) 1

/-- The carrier after `d2`'s projection BN. -/
noncomputable def dP2 (nCls : Nat) (t : ℝ) : Fin 128 → ℝ :=
  fun ci => dS t 0 * bnIstd (2 * (28 * 28)) (bnRowLA 2 128 28 28 (Zp2 nCls t) ci) 1

/-- The carrier after `d3`'s projection BN. -/
noncomputable def dP3 (nCls : Nat) (t : ℝ) : Fin 256 → ℝ :=
  fun ci => dP2 nCls t 0 * bnIstd (2 * (14 * 14)) (bnRowLA 2 256 14 14 (Zp3 nCls t) ci) 1

/-- The carrier after `d4`'s projection BN — the one the head reads. -/
noncomputable def dP4 (nCls : Nat) (t : ℝ) : Fin 512 → ℝ :=
  fun ci => dP3 nCls t 0 * bnIstd (2 * (7 * 7)) (bnRowLA 2 512 7 7 (Zp4 nCls t) ci) 1

/-- The carrier at the stem's output: the centre-tap conv copies channel 0's `t` to every
    channel, the BN scales it by `istd`, and the pool carries it through unchanged. -/
theorem ed0 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre0 2 (sealW nCls) (sealX t)) := by
  rw [pc0]
  refine eDiff_pool 64 56 56 (dS t) _ ?_
  refine eDiff_bn 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (fun _ => t) (dS t) (Zs t) ?_ ?_
  · exact eDiff_convS2 (0 : Fin 3) rfl (by norm_num) (by norm_num) 1 (kv 64 0) _ (fun _ => t)
      (sealX t) (eDiff_sealX t) (fun o => by norm_num)
  · intro ci
    simp only [dS, kv_apply]
    ring

theorem ed1 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre1 2 (sealW nCls) (sealX t)) := by
  rw [pc1]
  exact eDiff_shift _ _ 1 (ed0 nCls t)

theorem ed2 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre2 2 (sealW nCls) (sealX t)) := by
  rw [pc2]
  exact eDiff_shift _ _ 1 (ed1 nCls t)

theorem ed3 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre3 2 (sealW nCls) (sealX t)) := by
  rw [pc3]
  exact eDiff_shift _ _ 1 (ed2 nCls t)

theorem ed4 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre4 2 (sealW nCls) (sealX t)) := by
  rw [pc4]
  refine eDiff_shift _ _ 1 ?_
  rw [sealProj_apply]
  exact eDiff_convS2Bn (h := 28) (w := 28) (0 : Fin 64) rfl (by norm_num) (by norm_num) 1 160
    (Zp2 nCls t) (ed3 nCls t) rfl (fun ci => by simp only [dP2]; ring)

theorem ed5 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre5 2 (sealW nCls) (sealX t)) := by
  rw [pc5]
  exact eDiff_shift _ _ 1 (ed4 nCls t)

theorem ed6 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre6 2 (sealW nCls) (sealX t)) := by
  rw [pc6]
  exact eDiff_shift _ _ 1 (ed5 nCls t)

theorem ed7 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre7 2 (sealW nCls) (sealX t)) := by
  rw [pc7]
  exact eDiff_shift _ _ 1 (ed6 nCls t)

theorem ed8 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre8 2 (sealW nCls) (sealX t)) := by
  rw [pc8]
  refine eDiff_shift _ _ 1 ?_
  rw [sealProj_apply]
  exact eDiff_convS2Bn (h := 14) (w := 14) (0 : Fin 128) rfl (by norm_num) (by norm_num) 1 160
    (Zp3 nCls t) (ed7 nCls t) rfl (fun ci => by simp only [dP3]; ring)

theorem ed9 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre9 2 (sealW nCls) (sealX t)) := by
  rw [pc9]
  exact eDiff_shift _ _ 1 (ed8 nCls t)

theorem ed10 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre10 2 (sealW nCls) (sealX t)) := by
  rw [pc10]
  exact eDiff_shift _ _ 1 (ed9 nCls t)

theorem ed11 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre11 2 (sealW nCls) (sealX t)) := by
  rw [pc11]
  exact eDiff_shift _ _ 1 (ed10 nCls t)

theorem ed12 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre12 2 (sealW nCls) (sealX t)) := by
  rw [pc12]
  exact eDiff_shift _ _ 1 (ed11 nCls t)

theorem ed13 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre13 2 (sealW nCls) (sealX t)) := by
  rw [pc13]
  exact eDiff_shift _ _ 1 (ed12 nCls t)

theorem ed14 (nCls : Nat) (t : ℝ) : EDiff (dP4 nCls t) (r34Pre14 2 (sealW nCls) (sealX t)) := by
  rw [pc14]
  refine eDiff_shift _ _ 1 ?_
  rw [sealProj_apply]
  exact eDiff_convS2Bn (h := 7) (w := 7) (0 : Fin 256) rfl (by norm_num) (by norm_num) 1 160
    (Zp4 nCls t) (ed13 nCls t) rfl (fun ci => by simp only [dP4]; ring)

theorem ed15 (nCls : Nat) (t : ℝ) : EDiff (dP4 nCls t) (r34Pre15 2 (sealW nCls) (sealX t)) := by
  rw [pc15]
  exact eDiff_shift _ _ 1 (ed14 nCls t)

theorem ed16 (nCls : Nat) (t : ℝ) : EDiff (dP4 nCls t) (r34Pre16 2 (sealW nCls) (sealX t)) := by
  rw [pc16]
  exact eDiff_shift _ _ 1 (ed15 nCls t)


-- ════════════════════════════════════════════════════════════════
-- § 9. The head reads the carrier off channel 0
-- ════════════════════════════════════════════════════════════════

theorem sealW_Wd (nCls : Nat) :
    (sealW nCls).Wd = fun (i : Fin 512) (j : Fin nCls) =>
      if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0 := rfl

/-- The head reads channel 0 into class 0 — `BatchSeal.head_diff_ct` at this net's widths. -/
theorem head_diff (nCls : Nat) (hn : 0 < nCls) (v : Vec (2 * (512 * 7 * 7))) (δ : Fin 512 → ℝ)
    (hv : EDiff δ v) :
    r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = δ 0 :=
  head_diff_ct (by norm_num) (by norm_num) (0 : Fin 512) rfl ⟨0, hn⟩ _ _
    (fun ci => by rw [sealW_Wd]; simp) rfl v δ hv

-- ════════════════════════════════════════════════════════════════
-- § 10. The output difference along the ray is `t · R t`
-- ════════════════════════════════════════════════════════════════

/-- **The positive, continuous nonlinear factor**: one `istd` per BN on the carrier's path —
    the stem's and the three projections'. No `γ` appears because every carrier-path `γ` is `1`,
    and no BN-*variance* derivative is ever taken: `R` enters only through the factor `t · R t`,
    whose derivative at `0` is `R 0` for any `R` continuous there. -/
noncomputable def Rr (nCls : Nat) (t : ℝ) : ℝ :=
  bnIstd (2 * ((2 * 56) * (2 * 56))) (bnRowLA 2 64 (2 * 56) (2 * 56) (Zs t) 0) 1
  * (bnIstd (2 * (28 * 28)) (bnRowLA 2 128 28 28 (Zp2 nCls t) 0) 1
     * (bnIstd (2 * (14 * 14)) (bnRowLA 2 256 14 14 (Zp3 nCls t) 0) 1
        * bnIstd (2 * (7 * 7)) (bnRowLA 2 512 7 7 (Zp4 nCls t) 0) 1))

theorem Rr_pos (nCls : Nat) (t : ℝ) : 0 < Rr nCls t :=
  mul_pos (bnIstd_pos _ 1 one_pos)
    (mul_pos (bnIstd_pos _ 1 one_pos)
      (mul_pos (bnIstd_pos _ 1 one_pos) (bnIstd_pos _ 1 one_pos)))

/-- **The class-0 difference between the two examples, along the ray, is `t · R t`.** The
    carrier vanishes at the base (both examples carry the same ramp), so the product rule's cross
    terms all carry a factor `t`. -/
theorem gd_ray (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    resnet34ForwardBFull 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - resnet34ForwardBFull 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = t * Rr nCls t := by
  rw [resnet34ForwardBFull_eq_chain, Function.comp_apply,
    head_diff nCls hn _ (dP4 nCls t) (ed16 nCls t)]
  simp only [dP4, dP3, dP2, dS, Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 11. `R` is continuous (every block is, `relu` and the pool included)
-- ════════════════════════════════════════════════════════════════

theorem sealX_continuous : Continuous sealX :=
  continuous_const.add (continuous_id.smul continuous_const)

theorem r34StemB_continuous (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) : Continuous (r34StemB N h w Ws bs εs γs βs) := by
  unfold r34StemB StableHLO.cbReluStridedB
  fun_prop (disch := assumption)

/-- `R` is continuous: every block is (`relu`, the pool and the residual adds included), and
    `fun_prop` composes them through the fourteen-block prefix once the chain is unfolded to its
    atoms. Every BN on the witness has `ε = 1`. -/
theorem Rr_continuous (nCls : Nat) : Continuous (Rr nCls) := by
  unfold Rr Zs Zp2 Zp3 Zp4 ctConv sealX r34Pre13 r34Pre12 r34Pre11 r34Pre10 r34Pre9 r34Pre8
    r34Pre7 r34Pre6 r34Pre5 r34Pre4 r34Pre3 r34Pre2 r34Pre1 r34Pre0
  unfold r34IdB r34DownB r34StemB projB StableHLO.cbReluB StableHLO.cbReluStridedB
    StableHLO.projStridedB
  fun_prop (disch := exact one_pos)

-- ════════════════════════════════════════════════════════════════
-- § 12. The seal
-- ════════════════════════════════════════════════════════════════

/-- **Level 2 — the witness is non-degenerate**: the full-width batch-BN ResNet-34 at the
    structural weights is NOT constant in its input. Straight from the ray: the class-0 difference
    between the two examples is `R 1 > 0` at `t = 1` and `0` at the base. -/
theorem sealX_nonconstant (nCls : Nat) (hn : 0 < nCls) :
    resnet34ForwardBFull 2 (sealW nCls) (sealX 1)
      ≠ resnet34ForwardBFull 2 (sealW nCls) (sealX 0) :=
  ne_of_ray_readout _ sealX _ _ (gd_ray nCls hn) (by simpa using (Rr_pos nCls 1).ne')

/-- **Level 3 — the whole-net Jacobian is nonzero at the witness.** `fderiv_ne_zero_of_ray` at
    the readout "example 0's class 0 minus example 1's class 0": along the ray it is `t · R t` with
    `R` continuous and `R 0 > 0`, so its derivative at `0` is `R 0 ≠ 0`. -/
theorem sealX_jacobian_nonzero (nCls : Nat) (hn : 0 < nCls) :
    fderiv ℝ (resnet34ForwardBFull 2 (sealW nCls)) (sealX 0) ≠ 0 :=
  fderiv_ne_zero_of_ray_readout _ sealX sealV sealX_zero_add _ _ (gd_ray nCls hn)
    (seal_differentiableAt nCls 0) (Rr_pos nCls 0).ne'
    (hasDerivAt_mul_self_zero (Rr_continuous nCls).continuousAt)

/-- **The seal**: the proven whole-network backward of the **full-width, batch-BatchNorm,
    `[3,4,6,3]`, 224×224** ResNet-34 — `resnet34ForwardBFull`, the forward the ImageNet artifacts
    run — is **not the zero map** at the witness. The conditional apex
    `resnet34ForwardBFullHasVJPAt` is therefore not vacuous, and it is not vacuous on the net
    itself rather than on a 2-channel proxy of it. -/
theorem sealX_backward_nontrivial (nCls : Nat) (hn : 0 < nCls) :
    ∃ (j₀ : Fin (2 * nCls)) (i₀ : Fin (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))),
      (sealVJP nCls 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (sealVJP nCls 0).backward_nontrivial_of_fderiv_ne (sealX_jacobian_nonzero nCls hn)

end R34FullBSeal
end Proofs
