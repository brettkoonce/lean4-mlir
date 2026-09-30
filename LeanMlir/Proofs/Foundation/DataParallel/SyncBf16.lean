import LeanMlir.Proofs.Foundation.DataParallel.Sync
import LeanMlir.Proofs.Float.RndP

/-! # Data parallelism at bf16 — every node shards exactly but the conv weight gradient

`DataParallel.Sync` states P3/P4 at the f32 nodes. The bf16 data-parallel renders ResNet-34
and ResNet-50 train from (`resnet34in_momdp64bf16`, `resnet50in_momdp64bf16`,
`resnet50in160_lambaccdp8x64wxclipbcebf16`) swap six of those nodes for bf16 kinds, and each
has its own `den`: operands rounded going in, the result rounded once on the way out. This file
is what those six kinds do under sharding.

| kind | under sharding | lemma |
|---|---|---|
| `convBf16`, `convStridedBf16` (forward, `.batchOp`) | exact | `den_convBf16_shard`, `den_convStridedBf16_shard` |
| `convBackBatchedBf16`, `convStridedBackBatchedBf16` | exact | `den_convBackBatchedBf16_shard`, `den_convStridedBackBatchedBf16_shard` |
| `convWeightGradBBf16`, `convStridedWeightGradBBf16`, all-reduced | rounded per replica | `den_allReduceMeanF_convWeightGradBBf16_sub_global` and its strided peer |

The first four round per element of a per-example map, so replica `r`'s value is `batchShard r`
of the same node at batch `R·N`, exactly as at f32.

**The weight gradients are not.** The emitted weight-gradient convolution contracts the batch
in one op and stores bf16 once, so the node's `den` is `rnd (Σ_n …)` with the rounding outside
the batch sum (`den_convWeightGradBBf16_eq_rnd`). On `R` replicas each rounds its OWN partial
sum `S_r` before the f32 all-reduce; on one device the global sum `Σ_r S_r` is rounded once. So
the collective is `(1/R)·Σ_r rnd S_r` where the batch-`R·N` node is `rnd (Σ_r S_r)`, and
`den_allReduceMeanF_convWeightGradBBf16_sub_global` states the difference exactly. At
`rnd := id` the two agree (`…_shard_id`), which is the f32 statement.

## The divisor step at bf16

A DP render divides its loss by the per-replica batch, so its cotangents are `R ×` the shard of
the global ones (`DataParallel.Sync`, "The `1/R`"). At f32 linearity carries that factor
through every backward; at bf16 it has to pass through `rnd` as well. The `*_smul` lemmas below
take that as a hypothesis, `∀ x, rnd (s * x) = s * rnd x`, and `rndP_zpow_mul` proves it for
the repo's rounding model at every integer power of two — so for bf16 (`rndP 7`) at `s = R` and
`s = 1/R` whenever `R` is a power of two, the ImageNet runs' `R = 4` among them.

## What is NOT claimed

Note: No whole-net statement: there is no single-device bf16 chain for either net to tie a twin to,
here or in the f32 tier. These are the per-node cases a bf16 twin would walk its chain with.
Note: `rndP` has an unbounded exponent (`RndP.lean`): bf16 overflow and subnormals are
outside it, where scaling by 4 can move a value across the format's boundary.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The forward convs — exact
-- ════════════════════════════════════════════════════════════════

/-- **A per-example node on replica `r` is shard `r` of the same node at batch `R·N`.** Stated
    node to node, because at bf16 there is no ℝ-level forward to name on the right. -/
theorem den_batchOp_shard_node {R N a b : Nat} (op : BatchableOp a b) (t : String)
    (e : Fin R → SHlo (N * a)) (X : Vec ((R * N) * a))
    (he : ∀ r, den (e r) = batchShard R N a X r) (r : Fin R) :
    den (.batchOp (N := N) op (e r))
      = batchShard R N b (den (.batchOp (N := R * N) op (.operand t X))) r := by
  rw [den_batchOp, den_batchOp, den_operand, he, batchShard_batchMap]

/-- **The bf16 forward conv shards exactly** — every 1×1 and 3×3 in the ResNet bf16 renders. -/
theorem den_convBf16_shard {R N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wN bN t : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : Fin R → SHlo (N * (ic * h * w)))
    (X : Vec ((R * N) * (ic * h * w))) (he : ∀ r, den (e r) = batchShard R N (ic * h * w) X r)
    (r : Fin R) :
    den (.batchOp (N := N) (.convBf16 (h := h) (w := w) rnd wN bN W b) (e r))
      = batchShard R N (oc * h * w)
          (den (.batchOp (N := R * N) (.convBf16 (h := h) (w := w) rnd wN bN W b)
            (.operand t X))) r :=
  den_batchOp_shard_node _ t e X he r

/-- **…and so does the bf16 symmetric stride-2 conv** — the 7×7 stem and the downsamples. -/
theorem den_convStridedBf16_shard {R N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wN bN t : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : Fin R → SHlo (N * (ic * (2 * h) * (2 * w))))
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (he : ∀ r, den (e r) = batchShard R N (ic * (2 * h) * (2 * w)) X r) (r : Fin R) :
    den (.batchOp (N := N) (.convStridedBf16 (h := h) (w := w) rnd wN bN W b) (e r))
      = batchShard R N (oc * h * w)
          (den (.batchOp (N := R * N) (.convStridedBf16 (h := h) (w := w) rnd wN bN W b)
            (.operand t X))) r :=
  den_batchOp_shard_node _ t e X he r

-- ════════════════════════════════════════════════════════════════
-- § The input-VJPs — exact
-- ════════════════════════════════════════════════════════════════

/-- **The bf16 conv input-VJP shards exactly.** Its rounding is per element of a per-example
    map, so it never sees another replica's rows. -/
theorem den_convBackBatchedBf16_shard {R N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (wN t : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (dy : Fin R → SHlo (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    den (.convBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b (dy r))
      = batchShard R N (ic * h * w)
          (den (.convBackBatchedBf16 (N := R * N) (ic := ic) (h := h) (w := w) rnd wN W b
            (.operand t DY))) r := by
  simp only [denStep]
  rw [hdy, batchShard_batchMap]

/-- **…and so does the bf16 strided conv input-VJP.** -/
theorem den_convStridedBackBatchedBf16_shard {R N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ)
    (wN t : String) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (dy : Fin R → SHlo (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    den (.convStridedBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b (dy r))
      = batchShard R N (ic * (2 * h) * (2 * w))
          (den (.convStridedBackBatchedBf16 (N := R * N) (ic := ic) (h := h) (w := w) rnd wN W b
            (.operand t DY))) r := by
  simp only [denStep]
  rw [hdy, batchShard_batchMap]

-- ════════════════════════════════════════════════════════════════
-- § The weight gradients — rounded per replica
-- ════════════════════════════════════════════════════════════════

/-- **The bf16 weight-gradient node is the f32 node at rounded operands, rounded once.** -/
theorem den_convWeightGradBBf16_eq_rnd {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ)
    (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (.convWeightGradBBf16 rnd xN b x W e) idx
      = rnd (den (.convWeightGradB xN b (fun i => rnd (x i)) W
          (.operand cotN (fun i => rnd (den e i)))) idx) := rfl

/-- **…and the strided one.** -/
theorem den_convStridedWeightGradBBf16_eq_rnd {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ)
    (xN cotN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (e : SHlo (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (.convStridedWeightGradBBf16 rnd xN b x W e) idx
      = rnd (den (.convStridedWeightGradB xN b (fun i => rnd (x i)) W
          (.operand cotN (fun i => rnd (den e i)))) idx) := rfl

/-- **Replica `r`'s partial sum**: the f32 conv weight-gradient node on shard `r` of the rounded
    global operands. The collective and the batch-`R·N` node are both built from these; they
    differ only in where `rnd` is applied. -/
noncomputable def convWGradShardSum {R N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (xN cotN : String)
    (b : Vec oc) (X : Vec ((R * N) * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (DY : Vec ((R * N) * (oc * h * w))) (r : Fin R) (idx : Fin (oc * ic * kH * kW)) : ℝ :=
  den (.convWeightGradB xN b (batchShard R N (ic * h * w) (fun i => rnd (X i)) r) W
    (.operand cotN (batchShard R N (oc * h * w) (fun i => rnd (DY i)) r))) idx

/-- **The strided peer of `convWGradShardSum`.** -/
noncomputable def convStridedWGradShardSum {R N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ)
    (xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (DY : Vec ((R * N) * (oc * h * w))) (r : Fin R)
    (idx : Fin (oc * ic * kH * kW)) : ℝ :=
  den (.convStridedWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) (fun i => rnd (X i)) r)
    W (.operand cotN (batchShard R N (oc * h * w) (fun i => rnd (DY i)) r))) idx

/-- **The all-reduced bf16 weight gradient is the mean of the replicas' ROUNDED partial
    sums.** Each replica on its shard, at the shard-`r` block of the global cotangent. -/
theorem den_allReduceMeanF_convWeightGradBBf16_shard {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (rnd : ℝ → ℝ) (t xN cotN : String) (ds : List Nat) (b : Vec oc)
    (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convWeightGradBBf16 rnd xN b (batchShard R N (ic * h * w) X r) W (dy r))) idx
      = (1 / (R : ℝ)) * ∑ r : Fin R, rnd (convWGradShardSum rnd xN cotN b X W DY r idx) := by
  simp only [den_allReduceMeanF]
  congr 1
  apply Finset.sum_congr rfl; intro r _
  rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN, hdy]
  rfl

/-- **The batch-`R·N` bf16 weight gradient rounds the SUM of the same partial sums, once.** -/
theorem den_convWeightGradBBf16_global_split {N ic oc h w kH kW : Nat} (R : Nat) (rnd : ℝ → ℝ)
    (xN cotN : String) (b : Vec oc) (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (.convWeightGradBBf16 (N := R * N) rnd xN b X W (.operand cotN DY)) idx
      = rnd (∑ r : Fin R, convWGradShardSum rnd xN cotN b X W DY r idx) := by
  rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN]
  congr 1
  simp only [convWGradShardSum, denStep, denStepApp]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  -- `rw`, not `simp`: the `x` slice sits inside `conv2dWeightGradHasVJP b x`, whose TYPE
  -- depends on it (as in `den_allReduceMeanF_convWeightGradB_shard`).
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **The one difference, exactly.** The all-reduced bf16 weight gradient minus `1/R` of the
    batch-`R·N` bf16 node is `1/R` of (sum of the rounded partial sums − the rounded sum). -/
theorem den_allReduceMeanF_convWeightGradBBf16_sub_global {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (rnd : ℝ → ℝ) (t xN cotN : String) (ds : List Nat) (b : Vec oc)
    (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convWeightGradBBf16 rnd xN b (batchShard R N (ic * h * w) X r) W (dy r))) idx
      - (1 / (R : ℝ)) * den (.convWeightGradBBf16 (N := R * N) rnd xN b X W (.operand cotN DY)) idx
      = (1 / (R : ℝ)) * ((∑ r : Fin R, rnd (convWGradShardSum rnd xN cotN b X W DY r idx))
          - rnd (∑ r : Fin R, convWGradShardSum rnd xN cotN b X W DY r idx)) := by
  rw [den_allReduceMeanF_convWeightGradBBf16_shard R hR rnd t xN cotN ds b W X DY dy hdy,
      den_convWeightGradBBf16_global_split R rnd xN cotN b W X DY, mul_sub]

/-- **At `rnd := id` the difference vanishes** — the collective is `1/R` of the batch-`R·N`
    node, which is the f32 statement `den_allReduceMeanF_convWeightGradB_shard`. -/
theorem den_allReduceMeanF_convWeightGradBBf16_shard_id {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec oc)
    (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convWeightGradBBf16 (fun x => x) xN b (batchShard R N (ic * h * w) X r) W
            (dy r))) idx
      = (1 / (R : ℝ)) * den (.convWeightGradBBf16 (N := R * N) (fun x => x) xN b X W
          (.operand cotN DY)) idx := by
  rw [den_allReduceMeanF_convWeightGradBBf16_shard R hR _ t xN cotN ds b W X DY dy hdy,
      den_convWeightGradBBf16_global_split R _ xN cotN b W X DY]

/-- **Strided: the all-reduced bf16 weight gradient is the mean of the rounded partial sums.** -/
theorem den_allReduceMeanF_convStridedWeightGradBBf16_shard {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (rnd : ℝ → ℝ) (t xN cotN : String) (ds : List Nat) (b : Vec oc)
    (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (DY : Vec ((R * N) * (oc * h * w))) (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convStridedWeightGradBBf16 rnd xN b
            (batchShard R N (ic * (2 * h) * (2 * w)) X r) W (dy r))) idx
      = (1 / (R : ℝ)) * ∑ r : Fin R, rnd (convStridedWGradShardSum rnd xN cotN b X W DY r idx) := by
  simp only [den_allReduceMeanF]
  congr 1
  apply Finset.sum_congr rfl; intro r _
  rw [den_convStridedWeightGradBBf16_eq_rnd rnd xN cotN, hdy]
  rfl

/-- **Strided: the batch-`R·N` node rounds the sum of the same partial sums, once.** -/
theorem den_convStridedWeightGradBBf16_global_split {N ic oc h w kH kW : Nat} (R : Nat)
    (rnd : ℝ → ℝ) (xN cotN : String) (b : Vec oc) (W : Kernel4 oc ic kH kW)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (oc * h * w)))
    (idx : Fin (oc * ic * kH * kW)) :
    den (.convStridedWeightGradBBf16 (N := R * N) rnd xN b X W (.operand cotN DY)) idx
      = rnd (∑ r : Fin R, convStridedWGradShardSum rnd xN cotN b X W DY r idx) := by
  rw [den_convStridedWeightGradBBf16_eq_rnd rnd xN cotN]
  congr 1
  simp only [convStridedWGradShardSum, denStep, denStepApp]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **Strided: the one difference, exactly.** -/
theorem den_allReduceMeanF_convStridedWeightGradBBf16_sub_global {N ic oc h w kH kW : Nat}
    (R : Nat) (hR : 0 < R) (rnd : ℝ → ℝ) (t xN cotN : String) (ds : List Nat) (b : Vec oc)
    (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (DY : Vec ((R * N) * (oc * h * w))) (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convStridedWeightGradBBf16 rnd xN b
            (batchShard R N (ic * (2 * h) * (2 * w)) X r) W (dy r))) idx
      - (1 / (R : ℝ)) * den (.convStridedWeightGradBBf16 (N := R * N) rnd xN b X W
          (.operand cotN DY)) idx
      = (1 / (R : ℝ)) * ((∑ r : Fin R, rnd (convStridedWGradShardSum rnd xN cotN b X W DY r idx))
          - rnd (∑ r : Fin R, convStridedWGradShardSum rnd xN cotN b X W DY r idx)) := by
  rw [den_allReduceMeanF_convStridedWeightGradBBf16_shard R hR rnd t xN cotN ds b W X DY dy hdy,
      den_convStridedWeightGradBBf16_global_split R rnd xN cotN b W X DY, mul_sub]

-- ════════════════════════════════════════════════════════════════
-- § The divisor step at bf16 — scaling through `rnd`
-- ════════════════════════════════════════════════════════════════

/-- **`Int.log b` shifts by `z` under scaling by `b^z`**, for any base `b > 1` and any integer
    exponent. -/
theorem int_log_zpow_mul {b : ℕ} (hb : 1 < b) (z : ℤ) {r : ℝ} (hr : 0 < r) :
    Int.log b ((b : ℝ) ^ z * r) = z + Int.log b r := by
  have hb0 : (0 : ℝ) < b := by exact_mod_cast zero_lt_one.trans hb
  have hbz : (0 : ℝ) < (b : ℝ) ^ z := zpow_pos hb0 z
  have hzr : 0 < (b : ℝ) ^ z * r := mul_pos hbz hr
  have hpow : ∀ y : ℤ, (b : ℝ) ^ (z + y) = (b : ℝ) ^ z * (b : ℝ) ^ y :=
    fun y => zpow_add₀ hb0.ne' z y
  apply le_antisymm
  · -- log (b^z r) < z + log r + 1, since b^z·r < b^(z + log r + 1)
    have hlt : (b : ℝ) ^ z * r < (b : ℝ) ^ (z + (Int.log b r + 1)) := by
      rw [hpow]
      exact mul_lt_mul_of_pos_left (Int.lt_zpow_succ_log_self hb r) hbz
    have := (Int.lt_zpow_iff_log_lt hb hzr).mp hlt
    omega
  · -- b^(z + log r) ≤ b^z·r
    apply (Int.zpow_le_iff_le_log hb hzr).mp
    rw [hpow]
    exact mul_le_mul_of_nonneg_left (Int.zpow_log_le_self hb hr) hbz.le

/-- **`Int.log` reads `|x|`, so the shift holds for either sign.** -/
theorem int_log_abs_zpow_mul {b : ℕ} (hb : 1 < b) (z : ℤ) {x : ℝ} (hx : x ≠ 0) :
    Int.log b |(b : ℝ) ^ z * x| = z + Int.log b |x| := by
  have hb0 : (0 : ℝ) < b := by exact_mod_cast zero_lt_one.trans hb
  rw [abs_mul, abs_of_pos (zpow_pos hb0 z)]
  exact int_log_zpow_mul hb z (abs_pos.mpr hx)

/-- **The repo's rounding model commutes with scaling by a power of two** — the grid at
    `2^z·x` is the grid at `x` scaled by `2^z`, because the exponent is unbounded. The
    exponent `z` is an integer, so this covers the multiplier `R = 2^k` and the divisor
    `1/R = 2^(-k)` alike. -/
theorem rndP_zpow_mul (p : ℕ) (z : ℤ) (x : ℝ) :
    rndP p ((2 : ℝ) ^ z * x) = (2 : ℝ) ^ z * rndP p x := by
  rcases eq_or_ne x 0 with hx | hx
  · simp [hx]
  have h2z : (2 : ℝ) ^ z ≠ 0 := zpow_ne_zero z two_ne_zero
  have hzx : (2 : ℝ) ^ z * x ≠ 0 := mul_ne_zero h2z hx
  have hlog : Int.log 2 |(2 : ℝ) ^ z * x| = z + Int.log 2 |x| := by
    have := int_log_abs_zpow_mul (b := 2) (by norm_num) z hx
    simpa using this
  unfold rndP
  rw [ite_eq_right hzx, ite_eq_right hx, hlog]
  have hs : (2 : ℝ) ^ (z + Int.log 2 |x| - (p : ℤ))
      = (2 : ℝ) ^ z * (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ)) := by
    rw [show z + Int.log 2 |x| - (p : ℤ) = z + (Int.log 2 |x| - (p : ℤ)) by ring,
        zpow_add₀ two_ne_zero]
  rw [hs, mul_div_mul_left _ _ h2z]
  ring

/-- **The bf16 conv input-VJP scales with its cotangent** when `rnd` commutes with the scale. -/
theorem convBackBatchedBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (wN t : String) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (DY : Vec (N * (oc * h * w))) :
    den (.convBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b
        (.operand t (fun i => s * DY i)))
      = fun i => s * den (.convBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b
          (.operand t DY)) i := by
  funext idx
  simp only [denStepApp, batchMap, hrnd, HasVJP.backward_smul]

/-- **…and the strided one.** -/
theorem convStridedBackBatchedBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (wN t : String) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (DY : Vec (N * (oc * h * w))) :
    den (.convStridedBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b
        (.operand t (fun i => s * DY i)))
      = fun i => s * den (.convStridedBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN
          W b (.operand t DY)) i := by
  funext idx
  simp only [denStepApp, batchMap, hrnd, HasVJP.backward_smul]

/-- **The bf16 conv weight gradient scales with its cotangent** when `rnd` commutes with it. -/
theorem convWeightGradBBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (xN t : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (DY : Vec (N * (oc * h * w)))
    (idx : Fin (oc * ic * kH * kW)) :
    den (.convWeightGradBBf16 rnd xN b x W (.operand t (fun i => s * DY i))) idx
      = s * den (.convWeightGradBBf16 rnd xN b x W (.operand t DY)) idx := by
  simp only [denStep, denStepApp]
  rw [← hrnd, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl; intro n _
  rw [show (fun j => rnd (batchSlice N (oc * h * w) (fun i => s * DY i) n j))
        = fun j => s * rnd (batchSlice N (oc * h * w) DY n j) from funext fun j => hrnd _,
      HasVJP.backward_smul]

/-- **…and the strided one.** -/
theorem convStridedWeightGradBBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (xN t : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (DY : Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (.convStridedWeightGradBBf16 rnd xN b x W (.operand t (fun i => s * DY i))) idx
      = s * den (.convStridedWeightGradBBf16 rnd xN b x W (.operand t DY)) idx := by
  simp only [denStep, denStepApp]
  rw [← hrnd, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl; intro n _
  rw [show (fun j => rnd (batchSlice N (oc * h * w) (fun i => s * DY i) n j))
        = fun j => s * rnd (batchSlice N (oc * h * w) DY n j) from funext fun j => hrnd _,
      HasVJP.backward_smul]

end Proofs
