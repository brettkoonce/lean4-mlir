import LeanMlir.Proofs.Foundation.DataParallelSync
import LeanMlir.Proofs.Float.Binary32Instance

/-! # Data parallelism at bf16 — every node shards exactly but the conv weight gradient

`DataParallelSync.lean` states P3/P4 at the f32 nodes. The bf16 data-parallel renders ResNet-34
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

⭐⭐ **The weight gradients are not.** The emitted weight-gradient convolution contracts the batch
in one op and stores bf16 once, so the node's `den` is `rnd (Σ_n …)` with the rounding outside
the batch sum (`den_convWeightGradBBf16_eq_rnd`). On `R` replicas each rounds its OWN partial
sum `S_r` before the f32 all-reduce; on one device the global sum `Σ_r S_r` is rounded once. So
the collective is `(1/R)·Σ_r rnd S_r` where the batch-`R·N` node is `rnd (Σ_r S_r)`, and
`den_allReduceMeanF_convWeightGradBBf16_sub_global` states the difference exactly. At
`rnd := id` the two agree (`…_shard_id`), which is the f32 statement.

## The divisor step at bf16

A DP render divides its loss by the per-replica batch, so its cotangents are `R ×` the shard of
the global ones (`DataParallelSync.lean`, "The `1/R`"). At f32 linearity carries that factor
through every backward; at bf16 it has to pass through `rnd` as well. The `*_smul` lemmas below
take that as a hypothesis, `∀ x, rnd (s * x) = s * rnd x`, and `rndP_two_pow_mul` proves it for
the repo's rounding model at every power of two — so for bf16 (`rndP 7`) at `R = 4`
(`rndP_mul_four`), the replica count of every ImageNet run.

## What is NOT claimed

⚠ No whole-net statement: there is no single-device bf16 chain for either net to tie a twin to,
here or in the f32 tier. These are the per-node cases a bf16 twin would walk its chain with.
⚠ `rndP` has an unbounded exponent (`Binary32Instance.lean`): bf16 overflow and subnormals are
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
  simp only [den]
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
  simp only [den]
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

/-- ⭐ **The all-reduced bf16 weight gradient is the mean of the replicas' ROUNDED partial
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

/-- ⭐ **The batch-`R·N` bf16 weight gradient rounds the SUM of the same partial sums, once.** -/
theorem den_convWeightGradBBf16_global_split {N ic oc h w kH kW : Nat} (R : Nat) (rnd : ℝ → ℝ)
    (xN cotN : String) (b : Vec oc) (W : Kernel4 oc ic kH kW) (X : Vec ((R * N) * (ic * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (.convWeightGradBBf16 (N := R * N) rnd xN b X W (.operand cotN DY)) idx
      = rnd (∑ r : Fin R, convWGradShardSum rnd xN cotN b X W DY r idx) := by
  rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN]
  congr 1
  simp only [convWGradShardSum, den]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  -- ⚠ `rw`, not `simp`: the `x` slice sits inside `conv2d_weight_grad_has_vjp b x`, whose TYPE
  -- depends on it (as in `den_allReduceMeanF_convWeightGradB_shard`).
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- ⭐⭐ **The one difference, exactly.** The all-reduced bf16 weight gradient minus `1/R` of the
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

/-- ⭐ **Strided: the all-reduced bf16 weight gradient is the mean of the rounded partial sums.** -/
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

/-- ⭐ **Strided: the batch-`R·N` node rounds the sum of the same partial sums, once.** -/
theorem den_convStridedWeightGradBBf16_global_split {N ic oc h w kH kW : Nat} (R : Nat)
    (rnd : ℝ → ℝ) (xN cotN : String) (b : Vec oc) (W : Kernel4 oc ic kH kW)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (oc * h * w)))
    (idx : Fin (oc * ic * kH * kW)) :
    den (.convStridedWeightGradBBf16 (N := R * N) rnd xN b X W (.operand cotN DY)) idx
      = rnd (∑ r : Fin R, convStridedWGradShardSum rnd xN cotN b X W DY r idx) := by
  rw [den_convStridedWeightGradBBf16_eq_rnd rnd xN cotN]
  congr 1
  simp only [convStridedWGradShardSum, den]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- ⭐⭐ **Strided: the one difference, exactly.** -/
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

/-- **`Int.log 2` shifts by `k` under scaling by `2^k`.** -/
theorem int_log_two_pow_mul (k : ℕ) {y : ℝ} (hy : 0 < y) :
    Int.log 2 ((2 : ℝ) ^ k * y) = (k : ℤ) + Int.log 2 y := by
  have hb : 1 < (2 : ℕ) := by norm_num
  have h2k : (0 : ℝ) < (2 : ℝ) ^ k := by positivity
  have hky : 0 < (2 : ℝ) ^ k * y := mul_pos h2k hy
  have hpow : ∀ z : ℤ, ((2 : ℕ) : ℝ) ^ ((k : ℤ) + z) = (2 : ℝ) ^ k * ((2 : ℕ) : ℝ) ^ z := by
    intro z
    rw [zpow_add₀ (by norm_num), zpow_natCast]
    norm_num
  apply le_antisymm
  · -- log (2^k y) < k + log y + 1, since 2^k·y < 2^(k + log y + 1)
    have hlt : (2 : ℝ) ^ k * y < ((2 : ℕ) : ℝ) ^ ((k : ℤ) + (Int.log 2 y + 1)) := by
      rw [hpow]
      exact mul_lt_mul_of_pos_left (by exact_mod_cast Int.lt_zpow_succ_log_self hb y) h2k
    have := (Int.lt_zpow_iff_log_lt hb hky).mp hlt
    omega
  · -- 2^(k + log y) ≤ 2^k·y
    apply (Int.zpow_le_iff_le_log hb hky).mp
    rw [hpow]
    exact mul_le_mul_of_nonneg_left (by exact_mod_cast Int.zpow_log_le_self hb hy) h2k.le

/-- **`Int.log` reads `|x|`, so the shift holds for either sign.** -/
theorem int_log_abs_two_pow_mul (k : ℕ) {x : ℝ} (hx : x ≠ 0) :
    Int.log 2 |(2 : ℝ) ^ k * x| = (k : ℤ) + Int.log 2 |x| := by
  rw [abs_mul, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ k)]
  exact int_log_two_pow_mul k (abs_pos.mpr hx)

/-- ⭐ **The repo's rounding model commutes with scaling by a power of two** — the grid at
    `2^k·x` is the grid at `x` scaled by `2^k`, because the exponent is unbounded. -/
theorem rndP_two_pow_mul (p k : ℕ) (x : ℝ) :
    rndP p ((2 : ℝ) ^ k * x) = (2 : ℝ) ^ k * rndP p x := by
  rcases eq_or_ne x 0 with hx | hx
  · simp [hx]
  have hkx : (2 : ℝ) ^ k * x ≠ 0 := mul_ne_zero (by positivity) hx
  unfold rndP
  rw [ite_eq_right hkx, ite_eq_right hx, int_log_abs_two_pow_mul k hx]
  have hs : (2 : ℝ) ^ ((k : ℤ) + Int.log 2 |x| - (p : ℤ))
      = (2 : ℝ) ^ k * (2 : ℝ) ^ (Int.log 2 |x| - (p : ℤ)) := by
    rw [show (k : ℤ) + Int.log 2 |x| - (p : ℤ) = (k : ℤ) + (Int.log 2 |x| - (p : ℤ)) by ring,
        zpow_add₀ (by norm_num), zpow_natCast]
  have h2k : (2 : ℝ) ^ k ≠ 0 := by positivity
  rw [hs, mul_div_mul_left _ _ h2k]
  ring

/-- **At `R = 4`, the replica count of every ImageNet run.** -/
theorem rndP_mul_four (p : ℕ) (x : ℝ) : rndP p (4 * x) = 4 * rndP p x := by
  have := rndP_two_pow_mul p 2 x
  norm_num at this
  exact this

/-- **The bf16 conv input-VJP scales with its cotangent** when `rnd` commutes with the scale. -/
theorem convBackBatchedBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (wN t : String) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (DY : Vec (N * (oc * h * w))) :
    den (.convBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b
        (.operand t (fun i => s * DY i)))
      = fun i => s * den (.convBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b
          (.operand t DY)) i := by
  funext idx
  simp only [den, batchMap, hrnd, HasVJP.backward_smul]

/-- **…and the strided one.** -/
theorem convStridedBackBatchedBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (wN t : String) (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (DY : Vec (N * (oc * h * w))) :
    den (.convStridedBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN W b
        (.operand t (fun i => s * DY i)))
      = fun i => s * den (.convStridedBackBatchedBf16 (N := N) (ic := ic) (h := h) (w := w) rnd wN
          W b (.operand t DY)) i := by
  funext idx
  simp only [den, batchMap, hrnd, HasVJP.backward_smul]

/-- **The bf16 conv weight gradient scales with its cotangent** when `rnd` commutes with it. -/
theorem convWeightGradBBf16_smul {N ic oc h w kH kW : Nat} (rnd : ℝ → ℝ) (s : ℝ)
    (hrnd : ∀ x, rnd (s * x) = s * rnd x) (xN t : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (DY : Vec (N * (oc * h * w)))
    (idx : Fin (oc * ic * kH * kW)) :
    den (.convWeightGradBBf16 rnd xN b x W (.operand t (fun i => s * DY i))) idx
      = s * den (.convWeightGradBBf16 rnd xN b x W (.operand t DY)) idx := by
  simp only [den]
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
  simp only [den]
  rw [← hrnd, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl; intro n _
  rw [show (fun j => rnd (batchSlice N (oc * h * w) (fun i => s * DY i) n j))
        = fun j => s * rnd (batchSlice N (oc * h * w) DY n j) from funext fun j => hrnd _,
      HasVJP.backward_smul]

end Proofs
