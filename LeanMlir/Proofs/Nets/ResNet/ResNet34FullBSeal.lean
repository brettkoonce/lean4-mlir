import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP
import LeanMlir.Proofs.Training.BatchSealKit
import LeanMlir.Proofs.Training.JacobianSeal

/-!
# ResNet-34's non-degeneracy seal, on the full-width batched net (levels 2 and 3)

`planning/full_width_seals.md` §4.1. `ResNet34FullBVJP.lean` proves
`resnet34ForwardB_full_has_vjp_at`: the whole-net VJP at any `(w, x)` that satisfies 32 relu
clauses, a stem clause and the stem pool's no-tie. A *conditional* theorem of that shape says
nothing unless its hypotheses are jointly satisfiable at a point with a nonzero Jacobian — and
until now that was exhibited only on a 2-channel per-example proxy (`ResNet34Live*`, retired with
this file). This file exhibits it on `resnet34ForwardB_full` itself: 64→512 channels, `[3,4,6,3]`
blocks, the 7×7/s2 stem, the 3×3/s2 pool, **batch** BatchNorm, at `224×224`.

## The witness

Weights are *structural*, not trained (a trained-weight witness is a numeric fact about millions of
floats, which is what the training runs evidence — `planning/full_width_seals.md` §7):

* every residual body is zeroed, so it is the constant `β₂ = 1` whatever `γ₂` is
  (`BatchSeal.bnBatchLA_const`: a constant channel has variance 0, so batch BN returns `β`). On a
  nonnegative activation the identity block is therefore the affine shift `a ↦ a + 1`
  (`sealIdB_eq`) — and the shift is batch-uniform, hence invisible to the carrier;
* every channel-changing conv (the stem, the three 1×1/s2 projections) is a **centre-tap
  broadcast** (`BatchSeal.ctK`): every output channel is a copy of input channel 0 read through the
  kernel's centre tap. ⚠ Centre tap, not a general kernel, because `conv2d` zero-pads: a conv of a
  constant is not constant at the border, but the centre tap is in range at every cell. ⚠⚠ And a
  *broadcast*, not the plan's diagonal-on-channel-0: a kernel feeding only output channel 0 leaves
  the stem's other 63 channels constant, and a constant channel ties every 3×3 window of the pool
  — `R34PoolSmoothAt` quantifies over channels, so the tap has to reach all of them;
* `γ = 1`, `β = 160` at the stem and the projections, `ε = 1` everywhere. `√(N·h·w) ≤ √25088 < 160`
  at every one of those four BN widths, so `BatchSeal.bnBatchLA_pos` puts **every** relu strictly
  off its kink at **every** input — the net needs no eventually-argument for its relus, and the
  only genuine kink left is the stem pool;
* the head reads channel 0 into class 0 (`Wd 0 0 = 1`, `bd = 0`), so `0 < nCls` is the only
  constraint on the class count.

`N = 2` and the input is `sealX t = sealBase + t • sealV`: both examples carry the same strictly
decreasing ramp in channel 0, and `sealV` adds `t` to **all of example 0's channel 0**.

## Why the carrier is a batch difference (§3.2 of the plan)

⭐⭐ `bnBatchLA` normalizes each channel over all `N·h·w` cells, so the proxies' channel-difference
carrier is exactly what a channel's own mean subtracts, and at `N = 1` this net is constant in its
input. The carrier here is `EDiff`: example 0's slab is example 1's plus a per-channel constant.
Batch BN keeps it and scales it by `γ_c · istd_c` (`BatchSeal.bnBatchLA_exdiff`) because the two
examples share one mean and one `istd`; the centre-tap convs copy channel 0's offset to every
channel; the pool shifts with it (`BatchSeal.maxPool3s2_shift`, for every `t`, no argmax argument);
the zeroed bodies add a batch-uniform constant and so are transparent. The class-0 output
difference between the two examples is therefore `t · R t` with `R` a product of the four
carrier-path `istd`s, continuous and positive, so `g'(0) = R 0 ≠ 0` and no BN-variance derivative
is ever taken.
-/

namespace Proofs
namespace R34FullBSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs BatchSeal

-- ════════════════════════════════════════════════════════════════
-- § 1. The structural weights
-- ════════════════════════════════════════════════════════════════

/-- A channel-constant BN parameter. -/
noncomputable def kv (c : Nat) (x : ℝ) : Vec c := fun _ => x

@[simp] theorem kv_apply (c : Nat) (x : ℝ) (i : Fin c) : kv c x i = x := rfl

/-- The zero kernel (every residual body). -/
noncomputable def zk (oc ic kH kW : Nat) : Kernel4 oc ic kH kW := fun _ _ _ _ => 0

@[simp] theorem zk_apply (oc ic kH kW : Nat) (o : Fin oc) (c : Fin ic) (kh : Fin kH)
    (kw : Fin kW) : zk oc ic kH kW o c kh kw = 0 := rfl

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
-- § 2. The margin: `|γ|·√(N·h·w) < β` at all four carrier BNs
-- ════════════════════════════════════════════════════════════════

/-- `1 · √n < 160` whenever `n < 25600` — the margin check, in the shape
    `BatchSeal.bnBatchLA_pos` consumes. All four carrier BN widths (`2·112²  = 25088`, `2·28²`,
    `2·14²`, `2·7²`) clear it. -/
theorem margin160 (n : ℕ) (h : (n : ℝ) < 25600) :
    |(1 : ℝ)| * Real.sqrt ((n : ℕ) : ℝ) < 160 := by
  rw [abs_one, one_mul]
  exact sqrt_lt_param n 160 (by norm_num) (by nlinarith)

-- ════════════════════════════════════════════════════════════════
-- § 3. What the structural blocks are
-- ════════════════════════════════════════════════════════════════
-- ⚠⚠ **Every collapse lemma below is stated at VARIABLE `N, h, w, c` and instantiated at the
-- witness's numerals afterwards, never proved at them.** `relu_id_of_pos` applied directly to, say,
-- `cbReluStridedB 2 (h := 2*56) …` leaves the KERNEL a defeq between two numeral-shaped
-- compositions and it dies ("deep recursion" at `oc = 64, h = 112`, and a timeout already at
-- `h = 16`). At variables the same proof is instant, and instantiating a proved lemma is
-- substitution — no defeq at all. Same lesson as `maxPool3s2`'s header records for `rfl`.

/-- **A zeroed final conv makes a body the constant `β₂`** — `projB` at a zero kernel is
    `bnBatchLA` of the constant `0`, which is `β₂` (variance 0). Used for both block kinds. -/
theorem projB_zero_const {N ic oc h w kH kW : Nat} (hn : 0 < N * (h * w))
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0)
    (ε : ℝ) (γ β : Vec oc) (bb : ℝ) (hβ : ∀ ci, β ci = bb) (u : Vec (N * (ic * h * w))) :
    projB N (h := h) (w := w) W b ε γ β u = fun _ => bb := by
  funext k
  show StableHLO.bnBatchLA N oc h w ε γ β (StableHLO.batchMap N (flatConv W b) u) k = bb
  rw [batchMap_flatConv_zero W b hW hb]
  exact bnBatchLA_const hn ε γ β bb 0 hβ k

/-- The identity block's body is the constant `1`, at every input. -/
theorem sealIdBody (N h w c : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w))) :
    (projB N (h := h) (w := w) (sealIdW c).W₂ (sealIdW c).b₂ (sealIdW c).ε₂
        (sealIdW c).γ₂ (sealIdW c).β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) (sealIdW c).W₁ (sealIdW c).b₁ (sealIdW c).ε₁
        (sealIdW c).γ₁ (sealIdW c).β₁) v = fun _ => (1 : ℝ) :=
  projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1 (fun _ => rfl) _

/-- The downsample block's body is the constant `1`, at every input. -/
theorem sealDnBody (N h w ic oc : Nat) (hn : 0 < N * (h * w))
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    (projB N (h := h) (w := w) (sealDnW ic oc).W₂ (sealDnW ic oc).b₂ (sealDnW ic oc).ε₂
        (sealDnW ic oc).γ₂ (sealDnW ic oc).β₂ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) (sealDnW ic oc).W₁ (sealDnW ic oc).b₁
        (sealDnW ic oc).ε₁ (sealDnW ic oc).γ₁ (sealDnW ic oc).β₁) v = fun _ => (1 : ℝ) :=
  projB_zero_const hn _ _ (fun _ _ _ _ => rfl) (fun _ => rfl) _ _ _ 1 (fun _ => rfl) _

/-- ⭐ **The structural identity block is the shift `a ↦ a + 1`** on a nonnegative activation: the
    body is the constant `1` and the post-residual relu is off (`1 + a ≥ 1 > 0`). -/
theorem sealIdB_eq (N h w c : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w)))
    (hv : ∀ k, 0 ≤ v k) :
    r34IdB N h w (sealIdW c) v = fun k => v k + 1 := by
  have hbody := sealIdBody N h w c hn v
  have hres : ∀ k, residual
      (projB N (h := h) (w := w) (sealIdW c).W₂ (sealIdW c).b₂ (sealIdW c).ε₂
          (sealIdW c).γ₂ (sealIdW c).β₂ ∘
        StableHLO.cbReluB N (h := h) (w := w) (sealIdW c).W₁ (sealIdW c).b₁ (sealIdW c).ε₁
          (sealIdW c).γ₁ (sealIdW c).β₁) v k = v k + 1 := by
    intro k
    show _ + v k = v k + 1
    rw [hbody]
    ring
  funext k
  show relu (N * (c * h * w)) (residual _ v) k = v k + 1
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [hv i]), hres k]

/-- The structural downsample's collapsed form: its centre-tap projection. -/
noncomputable def sealProj (N h w ic oc : Nat) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  StableHLO.projStridedB N (h := h) (w := w) (ctK oc ic 1 1 1) (kv oc 0) 1 (kv oc 1) (kv oc 160)

/-- The projection is strictly positive at every input (the `β = 160` margin). -/
theorem sealProj_pos (N h w ic oc : Nat)
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) (k : Fin (N * (oc * h * w))) :
    0 < sealProj N h w ic oc v k :=
  bnBatchLA_pos 1 one_pos (kv oc 1) (kv oc 160) 1 160 (fun _ => rfl) (fun _ => rfl) hm _ k

/-- ⭐ **The structural downsample is its projection plus one**: the body is the constant `1` and
    the post-residual relu is off (`proj > 0`). -/
theorem sealDnB_eq (N h w ic oc : Nat) (hn : 0 < N * (h * w))
    (hm : |(1 : ℝ)| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 160)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    r34DownB N h w (sealDnW ic oc) v = fun k => sealProj N h w ic oc v k + 1 := by
  have hbody := sealDnBody N h w ic oc hn v
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

/-- A strided conv-bn-relu stage whose BN is everywhere positive has no relu left. -/
theorem cbReluStridedB_eq {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (γ β : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hp : ∀ k, 0 < StableHLO.bnBatchLA N oc h w ε γ β
      (StableHLO.batchMap N (flatConvStride2 W b) x) k) :
    StableHLO.cbReluStridedB N (h := h) (w := w) W b ε γ β x
      = StableHLO.bnBatchLA N oc h w ε γ β (StableHLO.batchMap N (flatConvStride2 W b) x) :=
  relu_id_of_pos hp

/-- ⭐ **The stem with its relu removed**: pool ∘ bn ∘ strided conv. The pool stays — it is the
    net's only remaining kink, and the carrier crosses it by `BatchSeal.maxPool3s2_shift`. -/
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
-- § 4. Every running activation is nonnegative
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
-- § 5. The clause bundles at the structural weights
-- ════════════════════════════════════════════════════════════════

/-- Both `ε`s of a structural identity block are positive. -/
theorem sealIdPos (c : Nat) : R34IdPos (sealIdW c) := ⟨one_pos, one_pos⟩

/-- All three `ε`s of a structural downsample are positive. -/
theorem sealDnPos (ic oc : Nat) : R34DownPos (sealDnW ic oc) := ⟨one_pos, one_pos, one_pos⟩

/-- ⭐ **The identity block's two relu clauses**: the mid-relu sees the constant `β₁ = 1`
    (weight-only), the outer one sees `1 + activation > 0`. -/
theorem sealIdSmooth (N h w c : Nat) (hn : 0 < N * (h * w)) (v : Vec (N * (c * h * w)))
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
    rw [congrFun (sealIdBody N h w c hn v) k]
    intro hc
    linarith [hv k]

/-- ⭐ **The downsample's two relu clauses** — both weight-only: the mid-relu sees `β₁ = 1`, the
    outer one `proj + 1 > 0` with `proj > 0` by the margin. No hypothesis on the activation. -/
theorem sealDnSmooth (N h w ic oc : Nat) (hn : 0 < N * (h * w))
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
    rw [congrFun (sealDnBody N h w ic oc hn v) k]
    intro hc
    linarith [sealProj_pos N h w ic oc hm v k]

/-- The stem's relu clause — weight-only, from the `β = 160` margin. -/
theorem sealStemSmooth (N h w ic oc : Nat) (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (hm : |(1 : ℝ)| * Real.sqrt ((N * ((2 * h) * (2 * w)) : ℕ) : ℝ) < 160)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    R34StemSmoothAt N h w Ws bs 1 (kv oc 1) (kv oc 160) x :=
  fun k => (bnBatchLA_pos 1 one_pos (kv oc 1) (kv oc 160) 1 160 (fun _ => rfl) (fun _ => rfl)
    hm (StableHLO.batchMap N (flatConvStride2 Ws bs) x) k).ne'

-- ════════════════════════════════════════════════════════════════
-- § 6. The witness input: a ramp in channel 0, perturbed on example 0
-- ════════════════════════════════════════════════════════════════

/-- The base slab: channel 0 carries the strictly decreasing ramp `−(i·224 + j)` — positionally
    injective, which is the stem pool's no-tie condition — and the other two channels are zero
    (the centre-tap stem reads only channel 0). Both examples carry the same slab, so the carrier
    vanishes at `t = 0`. -/
noncomputable def rampT : Tensor3 3 (2 * (2 * 56)) (2 * (2 * 56)) :=
  fun ci i j => if ci.val = 0 then -((i.val : ℝ) * 224 + (j.val : ℝ)) else 0

/-- The base point. -/
noncomputable def sealBase : Vec (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  bfrom (fun _ => rampT)

/-- The perturbation: **all** of example 0's channel 0. Uniform over the spatial grid, so it
    survives the pool for every `t` (`BatchSeal.maxPool3s2_shift`) with no argmax argument. -/
noncomputable def sealV : Vec (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  bfrom (fun n ci _ _ => if n.val = 0 ∧ ci.val = 0 then (1 : ℝ) else 0)

/-- The ray. -/
noncomputable def sealX (t : ℝ) : Vec (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) :=
  sealBase + t • sealV

theorem bcell_sealX (t : ℝ) (n : Fin 2) (ci : Fin 3) (i j : Fin (2 * (2 * 56))) :
    bcell (sealX t) n ci i j
      = (if ci.val = 0 then -((i.val : ℝ) * 224 + (j.val : ℝ)) else 0)
        + t * (if n.val = 0 ∧ ci.val = 0 then (1 : ℝ) else 0) := by
  rw [sealX, bcell_add, bcell_smul, sealBase, sealV, bcell_bfrom, bcell_bfrom]
  rfl

theorem sealX_zero_add (t : ℝ) : sealX 0 + t • sealV = sealX t := by
  rw [sealX, sealX, zero_smul, add_zero]

-- ════════════════════════════════════════════════════════════════
-- § 7. `EDiff` — the batch carrier, and what each op does to it
-- ════════════════════════════════════════════════════════════════

/-- **The carrier**: example 0's slab is example 1's plus the per-channel constant `δ`. ⭐⭐ The
    replacement for the proxies' channel difference, which per-channel batch BN annihilates. -/
def EDiff {c h w : Nat} (δ : Fin c → ℝ) (v : Vec (2 * (c * h * w))) : Prop :=
  ∀ (ci : Fin c) (i : Fin h) (j : Fin w),
    bcell v 0 ci i j = bcell v 1 ci i j + δ ci

/-- The ray's carrier: `t` in channel 0, nothing elsewhere. -/
theorem EDiff_sealX (t : ℝ) : EDiff (fun ci => if ci.val = 0 then t else 0) (sealX t) := by
  intro ci i j
  rw [bcell_sealX, bcell_sealX]
  by_cases h : ci.val = 0 <;> simp [h]

/-- A batch-uniform shift (a zeroed residual body) is transparent to the carrier. -/
theorem EDiff_shift {c h w : Nat} (δ : Fin c → ℝ) (v : Vec (2 * (c * h * w))) (s : ℝ)
    (hv : EDiff δ v) : EDiff δ (fun k => v k + s) := by
  intro ci i j
  rw [bcell_shift, bcell_shift, hv ci i j]
  ring

/-- ⭐⭐ **Batch BN scales the carrier by `γ_c · istd_c`** — the two examples share the channel's
    mean and `istd`, so centring keeps their difference (`BatchSeal.bnBatchLA_exdiff`). -/
theorem EDiff_bn (oc h w : Nat) (ε : ℝ) (γ β : Vec oc) (δ δ' : Fin oc → ℝ)
    (v : Vec (2 * (oc * h * w))) (hv : EDiff δ v)
    (hδ : ∀ ci, δ' ci = γ ci * δ ci * bnIstd (2 * (h * w)) (bnRowLA 2 oc h w v ci) ε) :
    EDiff δ' (StableHLO.bnBatchLA 2 oc h w ε γ β v) := by
  intro ci i j
  have hx := bnBatchLA_exdiff (N := 2) ε γ β v 0 1 ci i j
  have hd : bcell v 0 ci i j - bcell v 1 ci i j = δ ci := by rw [hv ci i j]; ring
  rw [hd] at hx
  rw [hδ ci]
  linarith

/-- A centre-tap strided conv copies channel 0's offset to **every** output channel (and is
    transparent to the zero padding, because only the centre tap is nonzero). -/
theorem EDiff_convS2 {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc) (δ : Fin ic → ℝ) (δ' : Fin oc → ℝ)
    (v : Vec (2 * (ic * (2 * h) * (2 * w)))) (hv : EDiff δ v) (hδ : ∀ o, δ' o = s * δ c₀) :
    EDiff δ' (StableHLO.batchMap 2 (flatConvStride2 (h := h) (w := w) (ctK oc ic kH kW s) b) v) := by
  intro o i j
  rw [hδ o, bcell_convS2_ctK c₀ hc₀ hkH hkW s b v 0 o i j,
    bcell_convS2_ctK c₀ hc₀ hkH hkW s b v 1 o i j, hv c₀ _ _]
  ring

/-- The 3×3/s2 pool keeps the carrier, at every `t`. -/
theorem EDiff_pool (c h w : Nat) (δ : Fin c → ℝ) (v : Vec (2 * (c * (2 * h) * (2 * w))))
    (hv : EDiff δ v) : EDiff δ (StableHLO.batchMap 2 (maxPool3s2Flat c h w) v) := by
  intro ci i j
  rw [bcell_pool, bcell_pool]
  exact maxPool3s2_shift (bcell v 0) (bcell v 1) (δ ci) ci (fun r s => hv ci r s) i j

-- ════════════════════════════════════════════════════════════════
-- § 8. The stem's pre-BN activation, and the pool's no-tie
-- ════════════════════════════════════════════════════════════════

/-- The stem's centre-tap conv output — the pre-BN activation on the carrier's path. -/
noncomputable def Zs (t : ℝ) : Vec (2 * (64 * (2 * 56) * (2 * 56))) :=
  StableHLO.batchMap 2 (flatConvStride2 (ctK 64 3 7 7 1) (kv 64 0)) (sealX t)

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
    0 < StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t) k :=
  bnBatchLA_pos 1 one_pos (kv 64 1) (kv 64 160) 1 160 (fun _ => rfl) (fun _ => rfl)
    margin_stem (Zs t) k

/-- ⭐ **The pre-BN stem activation is positionally injective** within each example and channel:
    the centre tap decimates the ramp, and example 0's uniform `+t` shifts every position alike. -/
theorem Zs_inj (t : ℝ) (n : Fin 2) (o : Fin 64) (r r' s s' : Fin (2 * 56))
    (heq : bcell (Zs t) n o r s = bcell (Zs t) n o r' s') : r = r' ∧ s = s' := by
  rw [Zs, bcell_convS2_ctK (0 : Fin 3) rfl (by norm_num) (by norm_num) 1 (kv 64 0) (sealX t) n o r s,
    bcell_convS2_ctK (0 : Fin 3) rfl (by norm_num) (by norm_num) 1 (kv 64 0) (sealX t) n o r' s',
    bcell_sealX, bcell_sealX] at heq
  simp only [kv_apply, Fin.val_zero, one_mul, zero_add, ite_true] at heq
  have hnat : 2 * r.val * 224 + 2 * s.val = 2 * r'.val * 224 + 2 * s'.val := by
    have hr : ((2 * r.val : ℕ) : ℝ) * 224 + ((2 * s.val : ℕ) : ℝ)
        = ((2 * r'.val : ℕ) : ℝ) * 224 + ((2 * s'.val : ℕ) : ℝ) := by
      push_cast at heq ⊢
      linarith
    have := hr
    push_cast at this
    have h2 : ((2 * r.val * 224 + 2 * s.val : ℕ) : ℝ) = ((2 * r'.val * 224 + 2 * s'.val : ℕ) : ℝ) := by
      push_cast
      linarith
    exact_mod_cast h2
  have hs := s.isLt
  have hs' := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

/-- ⭐ **The stem pool has no tie** at the witness: BN is injective within a channel
    (`BatchSeal.bnBatchLA_cell_inj`) and the pre-BN activation is positionally injective. -/
theorem sealPoolSmooth (t : ℝ) :
    R34PoolSmoothAt 2 56 56
      (StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t)) := by
  intro n
  refine maxPool3s2Smooth_of_injective _ (fun o r r' s s' heq => ?_)
  exact Zs_inj t n o r r' s s'
    (bnBatchLA_cell_inj 1 one_pos (kv 64 1) (kv 64 160) (Zs t) n o (by norm_num) r r' s s' heq)

-- ════════════════════════════════════════════════════════════════
-- § 9. The running activations: nonnegative, and collapsed
--   ⚠ Each `nn`/`pc` below INSTANTIATES a §3/§4 lemma proved at variable shapes. Proving any of
--   them at these numerals directly is what kills the kernel (see §3's banner).
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
-- § 10. The whole-net VJP at the witness
-- ════════════════════════════════════════════════════════════════

/-- The stem's relu is off at the witness, so the pool's no-tie condition can be stated on the
    BN output (`sealPoolSmooth`). -/
theorem stem_relu_off (nCls : Nat) (t : ℝ) :
    StableHLO.cbReluStridedB 2 (h := 2 * 56) (w := 2 * 56) (sealW nCls).sW (sealW nCls).sb
        (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ (sealX t)
      = StableHLO.bnBatchLA 2 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (Zs t) :=
  cbReluStridedB_eq _ _ _ _ _ (sealX t) (fun k => Zs_bn_pos t k)

theorem sealPoolClause (nCls : Nat) (t : ℝ) :
    R34PoolSmoothAt 2 56 56 (StableHLO.cbReluStridedB 2 (h := 2 * 56) (w := 2 * 56)
      (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ (sealW nCls).sβ
        (sealX t)) := by
  rw [stem_relu_off]
  exact sealPoolSmooth t

theorem sealStemClause (nCls : Nat) (t : ℝ) :
    R34StemSmoothAt 2 56 56 (sealW nCls).sW (sealW nCls).sb (sealW nCls).sε (sealW nCls).sγ
      (sealW nCls).sβ (sealX t) :=
  sealStemSmooth 2 56 56 3 64 _ _ margin_stem (sealX t)

theorem sc_a0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 56 56 (sealW nCls).a0 (r34Pre0 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 56 56 64 (by norm_num) _ (nn0 nCls t)

theorem sc_a1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 56 56 (sealW nCls).a1 (r34Pre1 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 56 56 64 (by norm_num) _ (nn1 nCls t)

theorem sc_a2 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 56 56 (sealW nCls).a2 (r34Pre2 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 56 56 64 (by norm_num) _ (nn2 nCls t)

theorem sc_d2 (nCls : Nat) (t : ℝ) :
    R34DownSmoothAt 2 28 28 (sealW nCls).d2 (r34Pre3 2 (sealW nCls) (sealX t)) :=
  sealDnSmooth 2 28 28 64 128 (by norm_num) margin28 _

theorem sc_b0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 28 28 (sealW nCls).b0 (r34Pre4 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 28 28 128 (by norm_num) _ (nn4 nCls t)

theorem sc_b1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 28 28 (sealW nCls).b1 (r34Pre5 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 28 28 128 (by norm_num) _ (nn5 nCls t)

theorem sc_b2 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 28 28 (sealW nCls).b2 (r34Pre6 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 28 28 128 (by norm_num) _ (nn6 nCls t)

theorem sc_d3 (nCls : Nat) (t : ℝ) :
    R34DownSmoothAt 2 14 14 (sealW nCls).d3 (r34Pre7 2 (sealW nCls) (sealX t)) :=
  sealDnSmooth 2 14 14 128 256 (by norm_num) margin14 _

theorem sc_c0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c0 (r34Pre8 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 14 14 256 (by norm_num) _ (nn8 nCls t)

theorem sc_c1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c1 (r34Pre9 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 14 14 256 (by norm_num) _ (nn9 nCls t)

theorem sc_c2 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c2 (r34Pre10 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 14 14 256 (by norm_num) _ (nn10 nCls t)

theorem sc_c3 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c3 (r34Pre11 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 14 14 256 (by norm_num) _ (nn11 nCls t)

theorem sc_c4 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 14 14 (sealW nCls).c4 (r34Pre12 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 14 14 256 (by norm_num) _ (nn12 nCls t)

theorem sc_d4 (nCls : Nat) (t : ℝ) :
    R34DownSmoothAt 2 7 7 (sealW nCls).d4 (r34Pre13 2 (sealW nCls) (sealX t)) :=
  sealDnSmooth 2 7 7 256 512 (by norm_num) margin7 _

theorem sc_e0 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 7 7 (sealW nCls).e0 (r34Pre14 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 7 7 512 (by norm_num) _ (nn14 nCls t)

theorem sc_e1 (nCls : Nat) (t : ℝ) :
    R34IdSmoothAt 2 7 7 (sealW nCls).e1 (r34Pre15 2 (sealW nCls) (sealX t)) :=
  sealIdSmooth 2 7 7 512 (by norm_num) _ (nn15 nCls t)

/-- ⭐⭐ **The whole-net VJP at the witness** — every one of the 32 relu clauses, the stem clause
    and the pool's no-tie discharged at `(sealW nCls, sealX t)`, on `resnet34ForwardB_full`
    itself (transported through `resnet34ForwardB_full_eq_chain`). -/
noncomputable def sealVJP (nCls : Nat) (t : ℝ) :
    HasVJPAt (resnet34ForwardB_full 2 (sealW nCls)) (sealX t) := by
  rw [show resnet34ForwardB_full 2 (sealW nCls)
      = r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd ∘ r34Pre16 2 (sealW nCls)
      from funext (resnet34ForwardB_full_eq_chain 2 (sealW nCls))]
  exact resnet34ForwardB_full_has_vjp_at 2 (sealW nCls) one_pos
    (sealIdPos 64) (sealIdPos 64) (sealIdPos 64) (sealDnPos 64 128)
    (sealIdPos 128) (sealIdPos 128) (sealIdPos 128) (sealDnPos 128 256)
    (sealIdPos 256) (sealIdPos 256) (sealIdPos 256) (sealIdPos 256) (sealIdPos 256)
    (sealDnPos 256 512) (sealIdPos 512) (sealIdPos 512)
    (sealX t) (sealStemClause nCls t) (sealPoolClause nCls t)
    (sc_a0 nCls t) (sc_a1 nCls t) (sc_a2 nCls t) (sc_d2 nCls t)
    (sc_b0 nCls t) (sc_b1 nCls t) (sc_b2 nCls t) (sc_d3 nCls t)
    (sc_c0 nCls t) (sc_c1 nCls t) (sc_c2 nCls t) (sc_c3 nCls t) (sc_c4 nCls t)
    (sc_d4 nCls t) (sc_e0 nCls t) (sc_e1 nCls t)

/-- The net is differentiable at the witness — `fderiv_ne_zero_of_ray`'s first hypothesis. Built
    block by block, as the apex's own proof builds its chain. -/
theorem sealDiffAt (nCls : Nat) (t : ℝ) :
    DifferentiableAt ℝ (resnet34ForwardB_full 2 (sealW nCls)) (sealX t) := by
  rw [show resnet34ForwardB_full 2 (sealW nCls)
      = r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd ∘ r34Pre16 2 (sealW nCls)
      from funext (resnet34ForwardB_full_eq_chain 2 (sealW nCls))]
  have f0 : DifferentiableAt ℝ (r34Pre0 2 (sealW nCls)) (sealX t) :=
    r34StemB_differentiableAt 2 56 56 _ _ _ one_pos _ _ (by norm_num) (by norm_num)
      (by norm_num) (sealX t) (sealStemClause nCls t) (sealPoolClause nCls t)
  have f1 : DifferentiableAt ℝ (r34Pre1 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 56 56 (sealW nCls).a0 (sealIdPos 64) _ (sc_a0 nCls t)).comp (sealX t) f0
  have f2 : DifferentiableAt ℝ (r34Pre2 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 56 56 (sealW nCls).a1 (sealIdPos 64) _ (sc_a1 nCls t)).comp (sealX t) f1
  have f3 : DifferentiableAt ℝ (r34Pre3 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 56 56 (sealW nCls).a2 (sealIdPos 64) _ (sc_a2 nCls t)).comp (sealX t) f2
  have f4 : DifferentiableAt ℝ (r34Pre4 2 (sealW nCls)) (sealX t) :=
    (r34DownB_differentiableAt 2 28 28 (sealW nCls).d2 (sealDnPos 64 128) _ (sc_d2 nCls t)).comp (sealX t) f3
  have f5 : DifferentiableAt ℝ (r34Pre5 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 28 28 (sealW nCls).b0 (sealIdPos 128) _ (sc_b0 nCls t)).comp (sealX t) f4
  have f6 : DifferentiableAt ℝ (r34Pre6 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 28 28 (sealW nCls).b1 (sealIdPos 128) _ (sc_b1 nCls t)).comp (sealX t) f5
  have f7 : DifferentiableAt ℝ (r34Pre7 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 28 28 (sealW nCls).b2 (sealIdPos 128) _ (sc_b2 nCls t)).comp (sealX t) f6
  have f8 : DifferentiableAt ℝ (r34Pre8 2 (sealW nCls)) (sealX t) :=
    (r34DownB_differentiableAt 2 14 14 (sealW nCls).d3 (sealDnPos 128 256) _ (sc_d3 nCls t)).comp (sealX t) f7
  have f9 : DifferentiableAt ℝ (r34Pre9 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 14 14 (sealW nCls).c0 (sealIdPos 256) _ (sc_c0 nCls t)).comp (sealX t) f8
  have f10 : DifferentiableAt ℝ (r34Pre10 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 14 14 (sealW nCls).c1 (sealIdPos 256) _ (sc_c1 nCls t)).comp (sealX t) f9
  have f11 : DifferentiableAt ℝ (r34Pre11 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 14 14 (sealW nCls).c2 (sealIdPos 256) _ (sc_c2 nCls t)).comp (sealX t) f10
  have f12 : DifferentiableAt ℝ (r34Pre12 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 14 14 (sealW nCls).c3 (sealIdPos 256) _ (sc_c3 nCls t)).comp (sealX t) f11
  have f13 : DifferentiableAt ℝ (r34Pre13 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 14 14 (sealW nCls).c4 (sealIdPos 256) _ (sc_c4 nCls t)).comp (sealX t) f12
  have f14 : DifferentiableAt ℝ (r34Pre14 2 (sealW nCls)) (sealX t) :=
    (r34DownB_differentiableAt 2 7 7 (sealW nCls).d4 (sealDnPos 256 512) _ (sc_d4 nCls t)).comp (sealX t) f13
  have f15 : DifferentiableAt ℝ (r34Pre15 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 7 7 (sealW nCls).e0 (sealIdPos 512) _ (sc_e0 nCls t)).comp (sealX t) f14
  have f16 : DifferentiableAt ℝ (r34Pre16 2 (sealW nCls)) (sealX t) :=
    (r34IdB_differentiableAt 2 7 7 (sealW nCls).e1 (sealIdPos 512) _ (sc_e1 nCls t)).comp (sealX t) f15
  exact (r34HeadB_differentiable 2 7 7 (sealW nCls).Wd (sealW nCls).bd _).comp (sealX t) f16

-- ════════════════════════════════════════════════════════════════
-- § 11. The carrier along the ray
--   Four BN sites lie on the carrier's path (the stem and the three projections); the thirteen
--   identity blocks contribute a batch-uniform `+1` each, which the carrier does not see. So
--   `EDiff` takes only four distinct values down the whole trunk.
-- ════════════════════════════════════════════════════════════════

/-- `sealProj`, unfolded — bn of the centre-tap strided conv. -/
theorem sealProj_apply (N h w ic oc : Nat) (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    sealProj N h w ic oc v
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConvStride2 (ctK oc ic 1 1 1) (kv oc 0)) v) := rfl

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

/-- ⭐ The carrier at the stem's output: the centre-tap conv copies channel 0's `t` to every
    channel, the BN scales it by `istd`, and the pool carries it through unchanged. -/
theorem ed0 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre0 2 (sealW nCls) (sealX t)) := by
  rw [pc0]
  refine EDiff_pool 64 56 56 (dS t) _ ?_
  refine EDiff_bn 64 (2 * 56) (2 * 56) 1 (kv 64 1) (kv 64 160) (fun _ => t) (dS t) (Zs t) ?_ ?_
  · exact EDiff_convS2 (0 : Fin 3) rfl (by norm_num) (by norm_num) 1 (kv 64 0) _ (fun _ => t)
      (sealX t) (EDiff_sealX t) (fun o => by norm_num)
  · intro ci
    simp only [dS, kv_apply]
    ring

theorem ed1 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre1 2 (sealW nCls) (sealX t)) := by
  rw [pc1]
  exact EDiff_shift _ _ 1 (ed0 nCls t)

theorem ed2 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre2 2 (sealW nCls) (sealX t)) := by
  rw [pc2]
  exact EDiff_shift _ _ 1 (ed1 nCls t)

theorem ed3 (nCls : Nat) (t : ℝ) : EDiff (dS t) (r34Pre3 2 (sealW nCls) (sealX t)) := by
  rw [pc3]
  exact EDiff_shift _ _ 1 (ed2 nCls t)

theorem ed4 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre4 2 (sealW nCls) (sealX t)) := by
  rw [pc4]
  refine EDiff_shift _ _ 1 ?_
  rw [sealProj_apply]
  refine EDiff_bn 128 28 28 1 (kv 128 1) (kv 128 160) (fun _ => 1 * dS t 0) (dP2 nCls t)
    (Zp2 nCls t) ?_ ?_
  · exact EDiff_convS2 (h := 28) (w := 28) (0 : Fin 64) rfl (by norm_num) (by norm_num) 1
      (kv 128 0) (dS t)
      _ (r34Pre3 2 (sealW nCls) (sealX t)) (ed3 nCls t) (fun o => rfl)
  · intro ci
    simp only [dP2, kv_apply]
    ring

theorem ed5 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre5 2 (sealW nCls) (sealX t)) := by
  rw [pc5]
  exact EDiff_shift _ _ 1 (ed4 nCls t)

theorem ed6 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre6 2 (sealW nCls) (sealX t)) := by
  rw [pc6]
  exact EDiff_shift _ _ 1 (ed5 nCls t)

theorem ed7 (nCls : Nat) (t : ℝ) : EDiff (dP2 nCls t) (r34Pre7 2 (sealW nCls) (sealX t)) := by
  rw [pc7]
  exact EDiff_shift _ _ 1 (ed6 nCls t)

theorem ed8 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre8 2 (sealW nCls) (sealX t)) := by
  rw [pc8]
  refine EDiff_shift _ _ 1 ?_
  rw [sealProj_apply]
  refine EDiff_bn 256 14 14 1 (kv 256 1) (kv 256 160) (fun _ => 1 * dP2 nCls t 0) (dP3 nCls t)
    (Zp3 nCls t) ?_ ?_
  · exact EDiff_convS2 (h := 14) (w := 14) (0 : Fin 128) rfl (by norm_num) (by norm_num) 1
      (kv 256 0) (dP2 nCls t)
      _ (r34Pre7 2 (sealW nCls) (sealX t)) (ed7 nCls t) (fun o => rfl)
  · intro ci
    simp only [dP3, kv_apply]
    ring

theorem ed9 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre9 2 (sealW nCls) (sealX t)) := by
  rw [pc9]
  exact EDiff_shift _ _ 1 (ed8 nCls t)

theorem ed10 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre10 2 (sealW nCls) (sealX t)) := by
  rw [pc10]
  exact EDiff_shift _ _ 1 (ed9 nCls t)

theorem ed11 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre11 2 (sealW nCls) (sealX t)) := by
  rw [pc11]
  exact EDiff_shift _ _ 1 (ed10 nCls t)

theorem ed12 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre12 2 (sealW nCls) (sealX t)) := by
  rw [pc12]
  exact EDiff_shift _ _ 1 (ed11 nCls t)

theorem ed13 (nCls : Nat) (t : ℝ) : EDiff (dP3 nCls t) (r34Pre13 2 (sealW nCls) (sealX t)) := by
  rw [pc13]
  exact EDiff_shift _ _ 1 (ed12 nCls t)

theorem ed14 (nCls : Nat) (t : ℝ) : EDiff (dP4 nCls t) (r34Pre14 2 (sealW nCls) (sealX t)) := by
  rw [pc14]
  refine EDiff_shift _ _ 1 ?_
  rw [sealProj_apply]
  refine EDiff_bn 512 7 7 1 (kv 512 1) (kv 512 160) (fun _ => 1 * dP3 nCls t 0) (dP4 nCls t)
    (Zp4 nCls t) ?_ ?_
  · exact EDiff_convS2 (h := 7) (w := 7) (0 : Fin 256) rfl (by norm_num) (by norm_num) 1
      (kv 512 0) (dP3 nCls t)
      _ (r34Pre13 2 (sealW nCls) (sealX t)) (ed13 nCls t) (fun o => rfl)
  · intro ci
    simp only [dP4, kv_apply]
    ring

theorem ed15 (nCls : Nat) (t : ℝ) : EDiff (dP4 nCls t) (r34Pre15 2 (sealW nCls) (sealX t)) := by
  rw [pc15]
  exact EDiff_shift _ _ 1 (ed14 nCls t)

theorem ed16 (nCls : Nat) (t : ℝ) : EDiff (dP4 nCls t) (r34Pre16 2 (sealW nCls) (sealX t)) := by
  rw [pc16]
  exact EDiff_shift _ _ 1 (ed15 nCls t)


-- ════════════════════════════════════════════════════════════════
-- § 12. The head reads the carrier off channel 0
-- ════════════════════════════════════════════════════════════════

theorem sealW_Wd (nCls : Nat) :
    (sealW nCls).Wd = fun (i : Fin 512) (j : Fin nCls) =>
      if i.val = 0 ∧ j.val = 0 then (1 : ℝ) else 0 := rfl

/-- ⭐ **GAP and the dense head deliver the carrier to class 0**: GAP of a uniformly shifted
    channel is shifted by the same constant, and `Wd` reads channel 0 into class 0. -/
theorem head_diff (nCls : Nat) (hn : 0 < nCls) (v : Vec (2 * (512 * 7 * 7))) (δ : Fin 512 → ℝ)
    (hv : EDiff δ v) :
    r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = δ 0 := by
  have hrow : ∀ n : Fin 2, Mat.unflatten (r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v) n
      = dense (sealW nCls).Wd (sealW nCls).bd (globalAvgPool (bcell v n)) := by
    intro n
    show Mat.unflatten (StableHLO.batchMap 2 (dense (sealW nCls).Wd (sealW nCls).bd)
      (StableHLO.batchMap 2 (globalAvgPoolFlat 512 7 7) v)) n = _
    rw [row_batchMap, row_batchMap]
    rfl
  have e0 : r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v
      (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = dense (sealW nCls).Wd (sealW nCls).bd (globalAvgPool (bcell v 0)) ⟨0, hn⟩ :=
    congrFun (hrow 0) ⟨0, hn⟩
  have e1 : r34HeadB 2 7 7 (sealW nCls).Wd (sealW nCls).bd v
      (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = dense (sealW nCls).Wd (sealW nCls).bd (globalAvgPool (bcell v 1)) ⟨0, hn⟩ :=
    congrFun (hrow 1) ⟨0, hn⟩
  have hgap : ∀ ci : Fin 512,
      globalAvgPool (bcell v 0) ci = globalAvgPool (bcell v 1) ci + δ ci :=
    fun ci => globalAvgPool_shift (by norm_num) (by norm_num) _ _ (δ ci) ci (fun i j => hv ci i j)
  have hWd : ∀ ci : Fin 512, (sealW nCls).Wd ci ⟨0, hn⟩ = if ci.val = 0 then (1 : ℝ) else 0 := by
    intro ci
    rw [sealW_Wd]
    simp
  have hbd : (sealW nCls).bd (⟨0, hn⟩ : Fin nCls) = 0 := rfl
  rw [e0, e1]
  simp only [dense, hWd, hbd, add_zero]
  rw [← Finset.sum_sub_distrib]
  rw [Finset.sum_congr rfl (fun ci _ => show
      globalAvgPool (bcell v 0) ci * (if ci.val = 0 then (1 : ℝ) else 0)
        - globalAvgPool (bcell v 1) ci * (if ci.val = 0 then (1 : ℝ) else 0)
      = δ ci * (if ci.val = 0 then (1 : ℝ) else 0) from by rw [hgap ci]; ring)]
  refine (Finset.sum_eq_single_of_mem (0 : Fin 512) (Finset.mem_univ _) ?_).trans ?_
  · intro ci _ hci
    have hc : ci.val ≠ 0 := fun h => hci (Fin.ext h)
    simp [hc]
  · simp

-- ════════════════════════════════════════════════════════════════
-- § 13. The output difference along the ray is `t · R t`
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The positive, continuous nonlinear factor**: one `istd` per BN on the carrier's path —
    the stem's and the three projections'. ⚠ No `γ` appears because every carrier-path `γ` is `1`,
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

/-- ⭐⭐ **The class-0 difference between the two examples, along the ray, is `t · R t`.** The
    carrier vanishes at the base (both examples carry the same ramp), so the product rule's cross
    terms all carry a factor `t`. -/
theorem gd_ray (nCls : Nat) (hn : 0 < nCls) (t : ℝ) :
    resnet34ForwardB_full 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - resnet34ForwardB_full 2 (sealW nCls) (sealX t)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      = t * Rr nCls t := by
  rw [resnet34ForwardB_full_eq_chain, Function.comp_apply,
    head_diff nCls hn _ (dP4 nCls t) (ed16 nCls t)]
  simp only [dP4, dP3, dP2, dS, Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 14. `R` is continuous (every block is, `relu` and the pool included)
-- ════════════════════════════════════════════════════════════════

theorem sealX_continuous : Continuous sealX :=
  continuous_const.add (continuous_id.smul continuous_const)

theorem projB_continuous (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Continuous (projB N (h := h) (w := w) W b ε γ β) :=
  (bnBatchLA_differentiable N oc h w ε hε γ β).continuous.comp
    (batchMap_continuous _ (flatConv_differentiable W b).continuous)

theorem cbReluB_continuous (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Continuous (StableHLO.cbReluB N (h := h) (w := w) W b ε γ β) :=
  (relu_continuous _).comp (projB_continuous N W b ε hε γ β)

theorem projStridedB_continuous (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Continuous (StableHLO.projStridedB N (h := h) (w := w) W b ε γ β) :=
  (bnBatchLA_differentiable N oc h w ε hε γ β).continuous.comp
    (batchMap_continuous _ (flatConvStride2_differentiable W b).continuous)

theorem cbReluStridedB_continuous (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Continuous (StableHLO.cbReluStridedB N (h := h) (w := w) W b ε γ β) :=
  (relu_continuous _).comp (projStridedB_continuous N W b ε hε γ β)

theorem r34IdB_continuous (N h w c : Nat) (p : R34IdW c) (h1 : 0 < p.ε₁) (h2 : 0 < p.ε₂) :
    Continuous (r34IdB N h w p) :=
  (relu_continuous _).comp (residual_continuous _
    ((projB_continuous N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ h2 p.γ₂ p.β₂).comp
      (cbReluB_continuous N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ h1 p.γ₁ p.β₁)))

theorem r34DownB_continuous (N h w ic oc : Nat) (p : R34DownW ic oc) (h1 : 0 < p.ε₁)
    (h2 : 0 < p.ε₂) (hp : 0 < p.εp) : Continuous (r34DownB N h w p) :=
  (relu_continuous _).comp (residualProj_continuous _ _
    (projStridedB_continuous N (h := h) (w := w) p.Wp p.bp p.εp hp p.γp p.βp)
    ((projB_continuous N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ h2 p.γ₂ p.β₂).comp
      (cbReluStridedB_continuous N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ h1 p.γ₁ p.β₁)))

theorem r34StemB_continuous (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) : Continuous (r34StemB N h w Ws bs εs γs βs) :=
  (batchMap_continuous _ (maxPool3s2Flat_continuous oc h w)).comp
    (cbReluStridedB_continuous N (h := 2 * h) (w := 2 * w) Ws bs εs hεs γs βs)

theorem cn0 (nCls : Nat) : Continuous (r34Pre0 2 (sealW nCls)) :=
  r34StemB_continuous 2 56 56 _ _ _ one_pos _ _

theorem cn1 (nCls : Nat) : Continuous (r34Pre1 2 (sealW nCls)) :=
  (r34IdB_continuous 2 56 56 64 (sealW nCls).a0 one_pos one_pos).comp (cn0 nCls)

theorem cn2 (nCls : Nat) : Continuous (r34Pre2 2 (sealW nCls)) :=
  (r34IdB_continuous 2 56 56 64 (sealW nCls).a1 one_pos one_pos).comp (cn1 nCls)

theorem cn3 (nCls : Nat) : Continuous (r34Pre3 2 (sealW nCls)) :=
  (r34IdB_continuous 2 56 56 64 (sealW nCls).a2 one_pos one_pos).comp (cn2 nCls)

theorem cn4 (nCls : Nat) : Continuous (r34Pre4 2 (sealW nCls)) :=
  (r34DownB_continuous 2 28 28 64 128 (sealW nCls).d2 one_pos one_pos one_pos).comp (cn3 nCls)

theorem cn5 (nCls : Nat) : Continuous (r34Pre5 2 (sealW nCls)) :=
  (r34IdB_continuous 2 28 28 128 (sealW nCls).b0 one_pos one_pos).comp (cn4 nCls)

theorem cn6 (nCls : Nat) : Continuous (r34Pre6 2 (sealW nCls)) :=
  (r34IdB_continuous 2 28 28 128 (sealW nCls).b1 one_pos one_pos).comp (cn5 nCls)

theorem cn7 (nCls : Nat) : Continuous (r34Pre7 2 (sealW nCls)) :=
  (r34IdB_continuous 2 28 28 128 (sealW nCls).b2 one_pos one_pos).comp (cn6 nCls)

theorem cn8 (nCls : Nat) : Continuous (r34Pre8 2 (sealW nCls)) :=
  (r34DownB_continuous 2 14 14 128 256 (sealW nCls).d3 one_pos one_pos one_pos).comp (cn7 nCls)

theorem cn9 (nCls : Nat) : Continuous (r34Pre9 2 (sealW nCls)) :=
  (r34IdB_continuous 2 14 14 256 (sealW nCls).c0 one_pos one_pos).comp (cn8 nCls)

theorem cn10 (nCls : Nat) : Continuous (r34Pre10 2 (sealW nCls)) :=
  (r34IdB_continuous 2 14 14 256 (sealW nCls).c1 one_pos one_pos).comp (cn9 nCls)

theorem cn11 (nCls : Nat) : Continuous (r34Pre11 2 (sealW nCls)) :=
  (r34IdB_continuous 2 14 14 256 (sealW nCls).c2 one_pos one_pos).comp (cn10 nCls)

theorem cn12 (nCls : Nat) : Continuous (r34Pre12 2 (sealW nCls)) :=
  (r34IdB_continuous 2 14 14 256 (sealW nCls).c3 one_pos one_pos).comp (cn11 nCls)

theorem cn13 (nCls : Nat) : Continuous (r34Pre13 2 (sealW nCls)) :=
  (r34IdB_continuous 2 14 14 256 (sealW nCls).c4 one_pos one_pos).comp (cn12 nCls)

theorem Zs_continuous : Continuous Zs :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp sealX_continuous

theorem Zp2_continuous (nCls : Nat) : Continuous (Zp2 nCls) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    ((cn3 nCls).comp sealX_continuous)

theorem Zp3_continuous (nCls : Nat) : Continuous (Zp3 nCls) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    ((cn7 nCls).comp sealX_continuous)

theorem Zp4_continuous (nCls : Nat) : Continuous (Zp4 nCls) :=
  (batchMap_continuous _ (flatConvStride2_differentiable _ _).continuous).comp
    ((cn13 nCls).comp sealX_continuous)

theorem Rr_continuous (nCls : Nat) : Continuous (Rr nCls) := by
  have c1 : Continuous (fun t : ℝ =>
      bnIstd (2 * ((2 * 56) * (2 * 56))) (bnRowLA 2 64 (2 * 56) (2 * 56) (Zs t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by norm_num⟩ : Fin (2 * ((2 * 56) * (2 * 56))))).comp
      ((bnRowLA_continuous 2 64 (2 * 56) (2 * 56) 0).comp Zs_continuous)
  have c2 : Continuous (fun t : ℝ =>
      bnIstd (2 * (28 * 28)) (bnRowLA 2 128 28 28 (Zp2 nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by norm_num⟩ : Fin (2 * (28 * 28)))).comp
      ((bnRowLA_continuous 2 128 28 28 0).comp (Zp2_continuous nCls))
  have c3 : Continuous (fun t : ℝ =>
      bnIstd (2 * (14 * 14)) (bnRowLA 2 256 14 14 (Zp3 nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by norm_num⟩ : Fin (2 * (14 * 14)))).comp
      ((bnRowLA_continuous 2 256 14 14 0).comp (Zp3_continuous nCls))
  have c4 : Continuous (fun t : ℝ =>
      bnIstd (2 * (7 * 7)) (bnRowLA 2 512 7 7 (Zp4 nCls t) 0) 1) :=
    (bnIstd_cont 1 one_pos (⟨0, by norm_num⟩ : Fin (2 * (7 * 7)))).comp
      ((bnRowLA_continuous 2 512 7 7 0).comp (Zp4_continuous nCls))
  exact c1.mul (c2.mul (c3.mul c4))

-- ════════════════════════════════════════════════════════════════
-- § 15. The seal
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **Level 2 — the witness is non-degenerate**: the full-width batch-BN ResNet-34 at the
    structural weights is NOT constant in its input. Straight from the ray: the class-0 difference
    between the two examples is `R 1 > 0` at `t = 1` and `0` at the base. -/
theorem sealX_nonconstant (nCls : Nat) (hn : 0 < nCls) :
    resnet34ForwardB_full 2 (sealW nCls) (sealX 1)
      ≠ resnet34ForwardB_full 2 (sealW nCls) (sealX 0) := by
  intro heq
  have h1 := gd_ray nCls hn 1
  have h0 := gd_ray nCls hn 0
  rw [heq] at h1
  have hz : (1 : ℝ) * Rr nCls 1 = 0 * Rr nCls 0 := by rw [← h1, ← h0]
  rw [one_mul, zero_mul] at hz
  linarith [Rr_pos nCls 1]

/-- ⭐⭐ **Level 3 — the whole-net Jacobian is nonzero at the witness.** `fderiv_ne_zero_of_ray` at
    the readout "example 0's class 0 minus example 1's class 0": along the ray it is `t · R t` with
    `R` continuous and `R 0 > 0`, so its derivative at `0` is `R 0 ≠ 0`. -/
theorem sealX_jacobian_nonzero (nCls : Nat) (hn : 0 < nCls) :
    fderiv ℝ (resnet34ForwardB_full 2 (sealW nCls)) (sealX 0) ≠ 0 := by
  refine fderiv_ne_zero_of_ray sealV (sealDiffAt nCls 0)
    (fun y => y (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - y (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls)))) (by fun_prop)
    (Rr_pos nCls 0).ne' ?_
  have heq : (fun t : ℝ => resnet34ForwardB_full 2 (sealW nCls) (sealX 0 + t • sealV)
        (finProdFinEquiv ((0 : Fin 2), (⟨0, hn⟩ : Fin nCls)))
      - resnet34ForwardB_full 2 (sealW nCls) (sealX 0 + t • sealV)
        (finProdFinEquiv ((1 : Fin 2), (⟨0, hn⟩ : Fin nCls))))
      = fun t : ℝ => t * Rr nCls t := by
    funext t
    rw [sealX_zero_add]
    exact gd_ray nCls hn t
  rw [heq]
  exact hasDerivAt_mul_self_zero (Rr_continuous nCls).continuousAt

/-- ⭐⭐ **The seal**: the proven whole-network backward of the **full-width, batch-BatchNorm,
    `[3,4,6,3]`, 224×224** ResNet-34 — `resnet34ForwardB_full`, the forward the ImageNet artifacts
    run — is **not the zero map** at the witness. The conditional apex
    `resnet34ForwardB_full_has_vjp_at` is therefore not vacuous, and it is not vacuous on the net
    itself rather than on a 2-channel proxy of it. -/
theorem sealX_backward_nontrivial (nCls : Nat) (hn : 0 < nCls) :
    ∃ (j₀ : Fin (2 * nCls)) (i₀ : Fin (2 * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))),
      (sealVJP nCls 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (sealVJP nCls 0).backward_nontrivial_of_fderiv_ne (sealX_jacobian_nonzero nCls hn)

end R34FullBSeal
end Proofs
