import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.MaxPool3s2
import LeanMlir.Proofs.Foundation.BatchedStageLayers

/-!
# The batch-BatchNorm seal kit — non-degeneracy machinery for the full-width nets

`planning/full_width_seals.md` §3. The level-2/3 witnesses ([`Training/JacobianSeal.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Training/JacobianSeal.lean)'s
`backward_nontrivial_of_fderiv_ne`) have until now been exhibited on **2-channel per-example
proxies**, whose BatchNorm is `bnForward` over one activation. The nets the ImageNet artifacts
actually run normalize with `StableHLO.bnBatchLA` — `bnBatchTensor4` at the network's left-assoc
index, i.e. `bnPerChannelFlat oc (N·h·w)`: **each channel over all its batch-and-spatial cells.**
This file is the machinery for sealing those, shared by the four kinked full-width nets
(ResNet-34/50, MobileNetV2/V4).

## ⭐⭐ Why the carrier has to change, and what that costs

A proxy seal carries a **channel** difference: `channel 0 = channel 1 + δ` at every position. Under
per-channel batch BN that carrier dies — a channel-uniform offset is exactly what the channel's own
mean subtracts — and at `N = 1` the structural net is constant in its input outright. So the
carrier here is an **example** difference (`EDiff` in the per-net files): example 0's slab is
example 1's slab plus `δ`, per channel. Batch BN keeps it and scales it by `γ_c · istd_c`
(`bnBatchLA_exdiff`), because the two examples share one mean and one `istd`. That forces `N = 2`
and makes the witness exercise the one op that couples examples.

The cost is that every fact now has to be read at a cell `(n, c, i, j)` of a left-assoc flat index.
§1 pays that once: `bcell` is the per-example `Tensor3` view, `bnRowLA` is the row BN normalizes,
and `bnBatchLA_bcell` is the bridge. Everything after it is the usual BN algebra on one row.

## What is here

* §0 `bcell` / `bfrom` — the `[N,C,H,W]` cell view, and `batchMap`'s action on it.
* §1 the index bridge: `laIdx`, `bnRowLA`, `bnBatchLA_bcell`, and `bnBatchLA_pointwise` (the
  workhorse: a channel-independent property of every cell of a `bnBatchLA` output reduces to the
  scalar `bnForward` on one row, with no index decomposition at the call site).
* §2 the BN consequences the seals need: constant channel ↦ `β`, the `|bn − β| ≤ |γ|√n` margin
  (`bnForward_abs_sub_le`, in `BatchNorm` with the other op-level BN facts; hence positivity, hence relu off its kink **at every input**; hence, two-sided, the relu6
  window `bnBatchLA_window` / `bnBatchLA_smooth6`), the example-difference identity, and
  within-example injectivity (the stem pool's no-tie).
* §3 the centre-tap kernel `ctK` and its conv value: the one weight shape that carries a signal
  through a channel-changing conv while staying transparent to a uniform offset. ⚠ Only the
  **centre** tap is nonzero, which is what makes it padding-proof — a conv of a constant is not
  constant near a zero-padded border, but a centre tap is always in range.
* §3b the XLA-`SAME` peers, which keep the **odd** spatial positions where the symmetric ops keep
  the even ones (`decimateOdd_unflatten`, `flatConvStride2Xla_ctK`, `bcell_convS2Xla_ctK`).
* §3c the centre-tap **depthwise** kernel `ctDW`. ⭐ A depthwise cannot broadcast, so where `ctK`
  collapses the carrier to one value at every output channel, `ctDW` scales it channel by channel.
* The op-level facts the ray argument leans on live with their ops, not here: the 3×3/s2 pool
  shifts with a uniform offset and keeps nonnegativity (`maxPool3s2_shift`, `maxPool3s2_nonneg`,
  `MaxPool3s2`), GAP shifts likewise (`globalAvgPool_shift`, `CNN`), and `relu`, `residual`, the
  pool, `batchMap` and `bnIstd` are continuous (`MLP`, `Residual`, `MaxPool3s2`, `Batched`,
  `BatchNorm`). §5 keeps `bnRowLA_continuous`, which is about this file's own reindex.
* §11 the SYMMETRIC strided depthwise (`EDiff_dwS2`), MobileNetV4's; §3c's is the XLA one.
* §12–§14 what MobileNetV4's **swish** needs, and nothing else does. ⭐⭐ `EDiff` carries only the
  gap between the two examples, which is all a relu-in-the-window or a centre-tap conv reads. A
  stage that is smooth but NOT affine changes that gap by an amount depending on the values
  themselves, so the carrier has to know them: `BUnif` says each example's slab is constant over
  the grid, one value per channel. Every op in these nets preserves that, and `bnBatchLA` then
  puts the two values symmetrically about `β` (`bnBatchLA_pair`) — so the swish's two outputs, and
  hence their gap `swishGap`, are functions of the gap alone. `EDiff_of_BUnif` hands the carrier
  back to §8 on the far side.
-/

namespace Proofs
namespace BatchSeal

open scoped BigOperators
open Finset

-- ════════════════════════════════════════════════════════════════
-- § 0. The `[N,C,H,W]` cell view
-- ════════════════════════════════════════════════════════════════

/-- **Example `n`'s `[C,H,W]` slab** of a batched activation `Vec (N·(c·h·w))`. Every per-cell
    statement in a batched seal is about this. -/
noncomputable def bcell {N c h w : Nat} (v : Vec (N * (c * h * w))) (n : Fin N) : Tensor3 c h w :=
  Tensor3.unflatten (Mat.unflatten v n)

/-- Assemble a batched activation from per-example slabs — the inverse of `bcell`, used to write
    the witness input down. -/
noncomputable def bfrom {N c h w : Nat} (f : Fin N → Tensor3 c h w) : Vec (N * (c * h * w)) :=
  Mat.flatten (fun n => Tensor3.flatten (f n))

theorem bcell_bfrom {N c h w : Nat} (f : Fin N → Tensor3 c h w) (n : Fin N) :
    bcell (bfrom f) n = f n := by
  have h := congrFun (Mat.unflatten_flatten (fun n => Tensor3.flatten (f n))) n
  simp only [bcell, bfrom]
  rw [h, Tensor3.unflatten_flatten]

theorem bcell_add {N c h w : Nat} (u v : Vec (N * (c * h * w))) (n : Fin N)
    (ci : Fin c) (i : Fin h) (j : Fin w) :
    bcell (u + v) n ci i j = bcell u n ci i j + bcell v n ci i j := rfl

theorem bcell_smul {N c h w : Nat} (t : ℝ) (v : Vec (N * (c * h * w))) (n : Fin N)
    (ci : Fin c) (i : Fin h) (j : Fin w) :
    bcell (t • v) n ci i j = t * bcell v n ci i j := rfl

/-- Adding a batch-uniform vector shifts every example's slab the same way — the shape a zeroed
    residual body contributes, and the reason it is transparent to the example difference. -/
theorem bcell_shift {N c h w : Nat} (v : Vec (N * (c * h * w))) (s : ℝ) (n : Fin N)
    (ci : Fin c) (i : Fin h) (j : Fin w) :
    bcell (fun k => v k + s) n ci i j = bcell v n ci i j + s := rfl

/-- **`batchMap` acts slab by slab.** -/
theorem bcell_batchMap {N a oc h' w' : Nat} (f : Vec a → Vec (oc * h' * w'))
    (x : Vec (N * a)) (n : Fin N) :
    bcell (StableHLO.batchMap N f x) n = Tensor3.unflatten (f (Mat.unflatten x n)) := by
  simp only [bcell]
  congr 1
  exact StableHLO.batchSlice_batchMap f x n

/-- The batched 3×3/s2 pool, slab by slab. -/
theorem bcell_pool {N c h w : Nat} (x : Vec (N * (c * (2 * h) * (2 * w)))) (n : Fin N) :
    bcell (StableHLO.batchMap N (maxPool3s2Flat c h w) x) n = maxPool3s2 (bcell x n) := by
  rw [bcell_batchMap]
  simp only [maxPool3s2Flat, Tensor3.unflatten_flatten, bcell]

/-- `batchMap` at a per-example output that is read as a plain `Vec` (the head's GAP/dense). -/
theorem row_batchMap {N a b : Nat} (f : Vec a → Vec b) (x : Vec (N * a)) (n : Fin N) :
    Mat.unflatten (StableHLO.batchMap N f x) n = f (Mat.unflatten x n) :=
  StableHLO.batchSlice_batchMap f x n

-- ════════════════════════════════════════════════════════════════
-- § 1. The index bridge — `bnBatchLA` at one cell
-- ════════════════════════════════════════════════════════════════

/-- The network's left-assoc `[N,C,H,W]` flat index. -/
noncomputable def laIdx (N oc h w : Nat) (n : Fin N) (c : Fin oc) (i : Fin h) (j : Fin w) :
    Fin (N * (oc * h * w)) :=
  finProdFinEquiv (n, finProdFinEquiv (finProdFinEquiv (c, i), j))

theorem bcell_eq_laIdx {N oc h w : Nat} (v : Vec (N * (oc * h * w))) (n : Fin N) (c : Fin oc)
    (i : Fin h) (j : Fin w) :
    bcell v n c i j = v (laIdx N oc h w n c i j) := rfl

/-- ⭐ **The one arithmetic fact**: `((c,i),j)` and `(c,(i,j))` are the same offset, so the
    `mul_assoc` cast that defines `bnBatchLA` sends the network cell to the `bnBatchTensor4` cell.
    `finProdFinEquiv` is row-major, so both sides are `j + w·i + h·w·c + oc·h·w·n`. -/
theorem laIdx_cast (N oc h w : Nat) (n : Fin N) (c : Fin oc) (i : Fin h) (j : Fin w) :
    Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) (laIdx N oc h w n c i j)
      = finProdFinEquiv (n, finProdFinEquiv (c, finProdFinEquiv (i, j))) := by
  apply Fin.ext
  simp only [laIdx, Fin.val_cast, finProdFinEquiv_apply_val]
  ring

/-- **The row batch BN normalizes**: channel `c`'s `N·h·w` cells, over the whole batch. -/
noncomputable def bnRowLA (N oc h w : Nat) (v : Vec (N * (oc * h * w))) (c : Fin oc) :
    Vec (N * (h * w)) :=
  Mat.unflatten (bnchwFwd N oc h w
    (v ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm)) c

/-- The row, read at `(n, (i, j))`, is the cell `(n, c, i, j)`. -/
theorem bnRowLA_apply {N oc h w : Nat} (v : Vec (N * (oc * h * w))) (c : Fin oc) (n : Fin N)
    (i : Fin h) (j : Fin w) :
    bnRowLA N oc h w v c (finProdFinEquiv (n, finProdFinEquiv (i, j))) = bcell v n c i j := by
  simp only [bnRowLA, Mat.unflatten, bnchwFwd, bnchwFwdIdx, Equiv.symm_apply_apply,
    Function.comp_apply, bcell_eq_laIdx]
  congr 1
  apply Fin.ext
  simp only [laIdx, Fin.val_cast, finProdFinEquiv_apply_val]
  ring

/-- ⭐⭐ **The bridge**: a cell of a `bnBatchLA` output is the scalar `bnForward` of that cell's
    channel row, at that cell's position in the row. Everything else in this file is BN algebra on
    one row. -/
theorem bnBatchLA_bcell (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) (v : Vec (N * (oc * h * w)))
    (n : Fin N) (c : Fin oc) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.bnBatchLA N oc h w ε γ β v) n c i j
      = bnForward (N * (h * w)) ε (γ c) (β c) (bnRowLA N oc h w v c)
          (finProdFinEquiv (n, finProdFinEquiv (i, j))) := by
  rw [bcell_eq_laIdx]
  show bnBatchTensor4 N oc h w ε γ β
      (v ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm)
      (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)) (laIdx N oc h w n c i j)) = _
  rw [laIdx_cast]
  show bnPerChannelFlat oc (N * (h * w)) ε γ β
      (bnchwFwd N oc h w (v ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm))
      (bnchwBackIdx N oc h w (finProdFinEquiv (n, finProdFinEquiv (c, finProdFinEquiv (i, j))))) = _
  rw [show bnchwBackIdx N oc h w (finProdFinEquiv (n, finProdFinEquiv (c, finProdFinEquiv (i, j))))
        = finProdFinEquiv (c, finProdFinEquiv (n, finProdFinEquiv (i, j))) by
      simp only [bnchwBackIdx, Equiv.symm_apply_apply]]
  simp only [bnPerChannelFlat, Mat.flatten, bnPerChannelMat, Equiv.symm_apply_apply, bnRowLA]

/-- ⭐ **The workhorse.** A property of every cell of a `bnBatchLA` output, reduced to the scalar
    `bnForward` on each channel's row — no index decomposition at the call site. Every clause of
    the shape "this BN output is off the kink / inside a window" goes through here. -/
theorem bnBatchLA_pointwise {N oc h w : Nat} (ε : ℝ) (γ β : Vec oc)
    (v : Vec (N * (oc * h * w))) (P : ℝ → Prop)
    (hP : ∀ (c : Fin oc) (q : Fin (N * (h * w))),
            P (bnForward (N * (h * w)) ε (γ c) (β c) (bnRowLA N oc h w v c) q)) :
    ∀ k, P (StableHLO.bnBatchLA N oc h w ε γ β v k) := by
  intro k
  obtain ⟨⟨n, r⟩, rfl⟩ := finProdFinEquiv.surjective k
  obtain ⟨⟨ch, jj⟩, rfl⟩ := finProdFinEquiv.surjective r
  obtain ⟨⟨c, ii⟩, rfl⟩ := finProdFinEquiv.surjective ch
  have key : StableHLO.bnBatchLA N oc h w ε γ β v
      (finProdFinEquiv (n, finProdFinEquiv (finProdFinEquiv (c, ii), jj)))
      = bnForward (N * (h * w)) ε (γ c) (β c) (bnRowLA N oc h w v c)
          (finProdFinEquiv (n, finProdFinEquiv (ii, jj))) :=
    bnBatchLA_bcell N oc h w ε γ β v n c ii jj
  rw [key]
  exact hP c _

-- ════════════════════════════════════════════════════════════════
-- § 2. What BN does to the structural weights
-- ════════════════════════════════════════════════════════════════

/-- A constant activation has a constant row. -/
theorem bnRowLA_const {N oc h w : Nat} (c₀ : ℝ) (ci : Fin oc) :
    bnRowLA N oc h w (fun _ => c₀) ci = fun _ => c₀ := by funext _; rfl

/-- **A constant channel normalizes to `β`** (variance 0, `xhat = 0`): a zeroed residual body is
    the constant `β₂`, whatever `γ₂` is. The batched peer of `bnForward_const`. -/
theorem bnBatchLA_const {N oc h w : Nat} (hn : 0 < N * (h * w)) (ε : ℝ) (γ β : Vec oc) (b c₀ : ℝ)
    (hβ : ∀ ci, β ci = b) :
    ∀ k, StableHLO.bnBatchLA N oc h w ε γ β (fun _ => c₀) k = b := by
  refine bnBatchLA_pointwise ε γ β _ (· = b) ?_
  intro ci q
  rw [bnRowLA_const, congrFun (bnForward_const hn ε (γ ci) (β ci) c₀) q, hβ]

/-- **The batched margin**: with `γ`, `β` channel-constant, every cell is within `|g|√(N·h·w)`
    of `b`. -/
theorem bnBatchLA_abs_sub_le {N oc h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (g b : ℝ)
    (hγ : ∀ ci, γ ci = g) (hβ : ∀ ci, β ci = b) (v : Vec (N * (oc * h * w))) :
    ∀ k, |StableHLO.bnBatchLA N oc h w ε γ β v k - b| ≤ |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) := by
  refine bnBatchLA_pointwise ε γ β v
    (fun z => |z - b| ≤ |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ)) ?_
  intro ci q
  rw [hγ ci, hβ ci]
  exact bnForward_abs_sub_le ε g b hε _ q

/-- **Positivity from the margin** — `|g|√(N·h·w) < b` makes the whole BN output strictly
    positive, at every input. The relu clause of every conv-bn-relu stage in the witness. -/
theorem bnBatchLA_pos {N oc h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (g b : ℝ)
    (hγ : ∀ ci, γ ci = g) (hβ : ∀ ci, β ci = b)
    (hm : |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < b) (v : Vec (N * (oc * h * w))) :
    ∀ k, 0 < StableHLO.bnBatchLA N oc h w ε γ β v k := by
  intro k
  have h := bnBatchLA_abs_sub_le ε hε γ β g b hγ hβ v k
  have h2 := abs_le.mp h
  linarith [h2.1]

/-- **The relu6 window from the margin** — the two-sided twin of `bnBatchLA_pos`. With `β = b`
    strictly inside `(|g|√(N·h·w), 6 − |g|√(N·h·w))` the whole BN output sits strictly inside
    `(0, 6)`, **at every input**, so `relu6_id_window` collapses the stage that follows it.
    ⭐ `b = 3` centres the window and makes both hypotheses the single check `|g|√(N·h·w) < 3`. -/
theorem bnBatchLA_window {N oc h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (g b : ℝ)
    (hγ : ∀ ci, γ ci = g) (hβ : ∀ ci, β ci = b)
    (hlo : |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < b)
    (hhi : b + |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 6) (v : Vec (N * (oc * h * w))) :
    ∀ k, 0 < StableHLO.bnBatchLA N oc h w ε γ β v k ∧
         StableHLO.bnBatchLA N oc h w ε γ β v k < 6 := by
  intro k
  have h := abs_le.mp (bnBatchLA_abs_sub_le ε hε γ β g b hγ hβ v k)
  exact ⟨by linarith [h.1], by linarith [h.2]⟩

/-- ⭐⭐ **Both relu6 clauses at once**, in the `≠ 0 ∧ ≠ 6` shape every MobileNet smoothness bundle
    is stated in. Since the bound is input-independent, this discharges a relu6 clause **without
    reading the activation** — which is why a relu6 net's whole clause bundle is weight-only. -/
theorem bnBatchLA_smooth6 {N oc h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (g b : ℝ)
    (hγ : ∀ ci, γ ci = g) (hβ : ∀ ci, β ci = b)
    (hlo : |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < b)
    (hhi : b + |g| * Real.sqrt ((N * (h * w) : ℕ) : ℝ) < 6) (v : Vec (N * (oc * h * w))) :
    ∀ k, StableHLO.bnBatchLA N oc h w ε γ β v k ≠ 0 ∧
         StableHLO.bnBatchLA N oc h w ε γ β v k ≠ 6 := fun k =>
  ⟨(bnBatchLA_window ε hε γ β g b hγ hβ hlo hhi v k).1.ne',
   (bnBatchLA_window ε hε γ β g b hγ hβ hlo hhi v k).2.ne⟩

/-- ⭐⭐ **The carrier step.** Two examples of one channel share the channel's mean and `istd`, so
    batch BN keeps their difference and multiplies it by `γ_c · istd_c`. This is what a
    channel-difference carrier cannot do under per-channel batch BN, and it is why the witness is
    at `N = 2`. -/
theorem bnBatchLA_exdiff {N oc h w : Nat} (ε : ℝ) (γ β : Vec oc) (v : Vec (N * (oc * h * w)))
    (n₀ n₁ : Fin N) (c : Fin oc) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.bnBatchLA N oc h w ε γ β v) n₀ c i j
        - bcell (StableHLO.bnBatchLA N oc h w ε γ β v) n₁ c i j
      = γ c * (bcell v n₀ c i j - bcell v n₁ c i j)
          * bnIstd (N * (h * w)) (bnRowLA N oc h w v c) ε := by
  rw [bnBatchLA_bcell, bnBatchLA_bcell, bnForward_chan_diff_γ, bnRowLA_apply, bnRowLA_apply]

/-- **Batch BN is injective within one example and channel** (`γ_c ≠ 0`): it is the strictly
    monotone affine map `γ_c·istd_c·(· − μ_c) + β_c` there. The stem pool's no-tie discharge —
    equal pooled cells force equal pre-BN cells, and the witness's ramp is positionally injective. -/
theorem bnBatchLA_cell_inj {N oc h w : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (oc * h * w))) (n : Fin N) (c : Fin oc) (hγ : γ c ≠ 0)
    (i j : Fin h) (i' j' : Fin w)
    (heq : bcell (StableHLO.bnBatchLA N oc h w ε γ β v) n c i i'
      = bcell (StableHLO.bnBatchLA N oc h w ε γ β v) n c j j') :
    bcell v n c i i' = bcell v n c j j' := by
  have key : γ c * (bcell v n c i i' - bcell v n c j j')
      * bnIstd (N * (h * w)) (bnRowLA N oc h w v c) ε = 0 := by
    rw [← bnRowLA_apply v c n i i', ← bnRowLA_apply v c n j j',
      ← bnForward_chan_diff_γ (ε := ε) (β := β c),
      ← bnBatchLA_bcell N oc h w ε γ β v n c i i', ← bnBatchLA_bcell N oc h w ε γ β v n c j j',
      heq, sub_self]
  have hist := (bnIstd_pos (bnRowLA N oc h w v c) ε hε).ne'
  rcases mul_eq_zero.mp key with h | h
  · rcases mul_eq_zero.mp h with h' | h'
    · exact absurd h' hγ
    · linarith
  · exact absurd h hist

-- ════════════════════════════════════════════════════════════════
-- § 3. The centre-tap kernel
-- ════════════════════════════════════════════════════════════════

/-- **The centre-tap broadcast kernel.** Every output channel reads input channel `0` through the
    kernel's *centre* tap, scaled by `s`; every other tap is zero.

    ⚠ Only the centre tap, and that is the point: `conv2d` pads with zeros, so a conv of a constant
    is **not** constant near the border — but the centre tap `kh = (kH−1)/2` reads position `hi`
    itself, which is in range at every output cell. So this kernel is transparent to a uniform
    offset at every position, which a multi-tap kernel is not. -/
noncomputable def ctK (oc ic kH kW : Nat) (s : ℝ) : Kernel4 oc ic kH kW :=
  fun _o i kh kw =>
    if i.val = 0 ∧ kh.val = (kH - 1) / 2 ∧ kw.val = (kW - 1) / 2 then s else 0

/-- **The centre-tap conv value**: `b o + s · (input channel 0 at the same position)`, at every
    output channel. ⚠ `c₀` is passed in (rather than built from `0 < ic`) so that the carrier's
    channel index is the same *term* at every use site — `ring` needs those `istd`s to be one atom. -/
theorem conv2d_ctK {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec oc) (x : Tensor3 ic h w) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d (ctK oc ic kH kW s) b x o hi wi = b o + s * x c₀ hi wi := by
  have hpH : (kH - 1) / 2 < kH := by omega
  have hpW : (kW - 1) / 2 < kW := by omega
  have hzero : ∀ (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
      ¬(c.val = 0 ∧ kh.val = (kH - 1) / 2 ∧ kw.val = (kW - 1) / 2) →
      ctK oc ic kH kW s o c kh kw = 0 := fun c kh kw hne => ite_eq_right hne
  unfold conv2d
  congr 1
  refine (Finset.sum_eq_single_of_mem c₀ (Finset.mem_univ _) ?_).trans ?_
  · intro c _ hc
    refine Finset.sum_eq_zero (fun kh _ => Finset.sum_eq_zero (fun kw _ => ?_))
    rw [hzero c kh kw (fun hh => hc (Fin.ext (hh.1.trans hc₀.symm))), zero_mul]
  refine (Finset.sum_eq_single_of_mem (⟨(kH - 1) / 2, hpH⟩ : Fin kH) (Finset.mem_univ _) ?_).trans ?_
  · intro kh _ hkh
    refine Finset.sum_eq_zero (fun kw _ => ?_)
    rw [hzero _ kh kw (fun hh => hkh (Fin.ext hh.2.1)), zero_mul]
  refine (Finset.sum_eq_single_of_mem (⟨(kW - 1) / 2, hpW⟩ : Fin kW) (Finset.mem_univ _) ?_).trans ?_
  · intro kw _ hkw
    rw [hzero _ _ kw (fun hh => hkw (Fin.ext hh.2.2)), zero_mul]
  rw [show ctK oc ic kH kW s o c₀ (⟨(kH - 1) / 2, hpH⟩ : Fin kH)
        (⟨(kW - 1) / 2, hpW⟩ : Fin kW) = s from ite_eq_left ⟨hc₀, rfl, rfl⟩]
  congr 1
  dsimp only
  split
  · refine congrArg₂ (x c₀) ?_ ?_ <;> (apply Fin.ext; simp only []; omega)
  · rename_i hcond
    exact absurd (by
      have := hi.isLt; have := wi.isLt
      refine ⟨?_, ?_, ?_, ?_⟩ <;> omega) hcond

/-- The flat centre-tap conv, in cell coordinates. -/
theorem flatConv_ctK {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec oc) (v : Vec (ic * h * w)) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (flatConv (h := h) (w := w) (ctK oc ic kH kW s) b v) o hi wi
      = b o + s * Tensor3.unflatten v c₀ hi wi := by
  simp only [flatConv, Tensor3.unflatten_flatten]
  exact conv2d_ctK c₀ hc₀ hkH hkW s b _ o hi wi

/-- Decimation reads position `(2i, 2j)`, channel for channel (the generic peer of the retired
    2-channel `decimate_unflatten`). -/
theorem decimate_unflatten (oc h w : Nat) (z : Vec (oc * (2 * h) * (2 * w))) (c : Fin oc)
    (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (decimateFlat oc h w z) c hi wi
      = (Tensor3.unflatten z : Tensor3 oc (2 * h) (2 * w)) c
          ⟨2 * hi.val, by have := hi.isLt; omega⟩ ⟨2 * wi.val, by have := wi.isLt; omega⟩ := by
  simp only [Tensor3.unflatten, decimateFlat, decimateIdx, Equiv.symm_apply_apply]

/-- The strided centre-tap conv: `b o + s · (input channel 0 at the even position)`. -/
theorem flatConvStride2_ctK {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec oc) (v : Vec (ic * (2 * h) * (2 * w))) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (flatConvStride2 (h := h) (w := w) (ctK oc ic kH kW s) b v) o hi wi
      = b o + s * (Tensor3.unflatten v : Tensor3 ic (2 * h) (2 * w)) c₀
          ⟨2 * hi.val, by have := hi.isLt; omega⟩ ⟨2 * wi.val, by have := wi.isLt; omega⟩ := by
  simp only [flatConvStride2, Function.comp_apply]
  rw [decimate_unflatten, flatConv_ctK c₀ hc₀ hkH hkW]

/-- The batched centre-tap conv, in cell coordinates — the carrier's conv step at stride 1
    (ResNet-50's stage-1 projection is the one site that needs it). -/
theorem bcell_conv_ctK {N ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (n : Fin N) (o : Fin oc) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.batchMap N (flatConv (h := h) (w := w) (ctK oc ic kH kW s) b) x) n o i j
      = b o + s * bcell x n c₀ i j := by
  rw [bcell_batchMap]
  exact flatConv_ctK c₀ hc₀ hkH hkW s b _ o i j

/-- The batched strided centre-tap conv, in cell coordinates — the carrier's conv step. -/
theorem bcell_convS2_ctK {N ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (n : Fin N) (o : Fin oc) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.batchMap N (flatConvStride2 (h := h) (w := w) (ctK oc ic kH kW s) b) x) n o i j
      = b o + s * bcell x n c₀
          ⟨2 * i.val, by have := i.isLt; omega⟩ ⟨2 * j.val, by have := j.isLt; omega⟩ := by
  rw [bcell_batchMap]
  exact flatConvStride2_ctK c₀ hc₀ hkH hkW s b _ o i j

/-- A zeroed conv (zero kernel, zero bias) sends everything to the constant `0` — the residual
    bodies. `flatConv_eq_zero` lifted across the batch. -/
theorem batchMap_flatConv_zero {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0) (x : Vec (N * (ic * h * w))) :
    StableHLO.batchMap N (flatConv (h := h) (w := w) W b) x = fun _ => (0 : ℝ) := by
  funext k
  show flatConv (h := h) (w := w) W b (fun i => x _) _ = 0
  rw [flatConv_eq_zero W b hW hb]

/-- A zeroed STRIDED conv sends everything to the constant `0` (decimation of a constant). -/
theorem batchMap_flatConvStride2_zero {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    StableHLO.batchMap N (flatConvStride2 (h := h) (w := w) W b) x = fun _ => (0 : ℝ) := by
  funext k
  show flatConvStride2 (h := h) (w := w) W b (fun i => x _) _ = 0
  simp only [flatConvStride2, Function.comp_apply]
  rw [flatConv_eq_zero W b hW hb]
  rfl

-- ═════════════════════════════════════════════════════════════
-- § 3b. The XLA-`SAME` peers — odd decimation
--   ⚠ At an even input a stride-2 XLA-`SAME` conv pads asymmetrically and so keeps the **odd**
--   positions, where the symmetric `flatConvStride2` keeps the even ones. MobileNetV2's stem and
--   all four of its strided depthwises are this family; ResNet's are not. A centre tap does not
--   care which phase survives, so these are §3's proofs with one index changed.
-- ═════════════════════════════════════════════════════════════

/-- Odd decimation reads position `(2i+1, 2j+1)`, channel for channel — the peer of
    `decimate_unflatten`. -/
theorem decimateOdd_unflatten (oc h w : Nat) (z : Vec (oc * (2 * h) * (2 * w))) (c : Fin oc)
    (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (decimateOddFlat oc h w z) c hi wi
      = (Tensor3.unflatten z : Tensor3 oc (2 * h) (2 * w)) c
          ⟨2 * hi.val + 1, by have := hi.isLt; omega⟩
          ⟨2 * wi.val + 1, by have := wi.isLt; omega⟩ := by
  simp only [Tensor3.unflatten, decimateOddFlat, decimateOddIdx, Equiv.symm_apply_apply]

/-- The strided XLA-`SAME` centre-tap conv: `b o + s · (input channel 0 at the ODD position)`.
    `flatConvStride2Xla` is `decimateOddFlat ∘ flatConv`, so this is `flatConvStride2_ctK` with
    `decimateOdd_unflatten` in place of `decimate_unflatten`. -/
theorem flatConvStride2Xla_ctK {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec oc) (v : Vec (ic * (2 * h) * (2 * w))) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (flatConvStride2Xla (h := h) (w := w) (ctK oc ic kH kW s) b v) o hi wi
      = b o + s * (Tensor3.unflatten v : Tensor3 ic (2 * h) (2 * w)) c₀
          ⟨2 * hi.val + 1, by have := hi.isLt; omega⟩
          ⟨2 * wi.val + 1, by have := wi.isLt; omega⟩ := by
  simp only [flatConvStride2Xla, Function.comp_apply]
  rw [decimateOdd_unflatten, flatConv_ctK c₀ hc₀ hkH hkW]

/-- The batched strided XLA-`SAME` centre-tap conv, in cell coordinates — MobileNetV2's stem. -/
theorem bcell_convS2Xla_ctK {N ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (n : Fin N) (o : Fin oc) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.batchMap N
        (flatConvStride2Xla (h := h) (w := w) (ctK oc ic kH kW s) b) x) n o i j
      = b o + s * bcell x n c₀
          ⟨2 * i.val + 1, by have := i.isLt; omega⟩
          ⟨2 * j.val + 1, by have := j.isLt; omega⟩ := by
  rw [bcell_batchMap]
  exact flatConvStride2Xla_ctK c₀ hc₀ hkH hkW s b _ o i j

-- ═════════════════════════════════════════════════════════════
-- § 3c. The centre-tap DEPTHWISE kernel
--   ⭐ A depthwise conv cannot broadcast — it reads only its own channel — so its centre tap is
--   the identity on every channel at once, and the carrier's per-channel `δ` survives unchanged
--   rather than collapsing to `fun _ => s · δ 0`. `depthwiseConv2d`'s padding guard is `conv2d`'s,
--   so the value proof is `conv2d_ctK`'s with the channel sum deleted.
-- ═════════════════════════════════════════════════════════════

/-- **The centre-tap depthwise kernel**: every channel reads itself through the kernel's centre
    tap, scaled by `s`. At `s = 1`, `b = 0` it is the identity. The depthwise peer of `ctK`. -/
noncomputable def ctDW (c kH kW : Nat) (s : ℝ) : DepthwiseKernel c kH kW :=
  fun _ch kh kw => if kh.val = (kH - 1) / 2 ∧ kw.val = (kW - 1) / 2 then s else 0

/-- **The centre-tap depthwise value**: `b ch + s · (the same channel at the same position)`. -/
theorem depthwise2d_ctDW {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec c) (x : Tensor3 c h w) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    depthwiseConv2d (ctDW c kH kW s) b x ch hi wi = b ch + s * x ch hi wi := by
  have hpH : (kH - 1) / 2 < kH := by omega
  have hpW : (kW - 1) / 2 < kW := by omega
  have hzero : ∀ (kh : Fin kH) (kw : Fin kW),
      ¬(kh.val = (kH - 1) / 2 ∧ kw.val = (kW - 1) / 2) →
      ctDW c kH kW s ch kh kw = 0 := fun kh kw hne => ite_eq_right hne
  unfold depthwiseConv2d
  congr 1
  refine (Finset.sum_eq_single_of_mem (⟨(kH - 1) / 2, hpH⟩ : Fin kH) (Finset.mem_univ _) ?_).trans ?_
  · intro kh _ hkh
    refine Finset.sum_eq_zero (fun kw _ => ?_)
    rw [hzero kh kw (fun hh => hkh (Fin.ext hh.1)), zero_mul]
  refine (Finset.sum_eq_single_of_mem (⟨(kW - 1) / 2, hpW⟩ : Fin kW) (Finset.mem_univ _) ?_).trans ?_
  · intro kw _ hkw
    rw [hzero _ kw (fun hh => hkw (Fin.ext hh.2)), zero_mul]
  rw [show ctDW c kH kW s ch (⟨(kH - 1) / 2, hpH⟩ : Fin kH) (⟨(kW - 1) / 2, hpW⟩ : Fin kW) = s from
      ite_eq_left ⟨rfl, rfl⟩]
  congr 1
  dsimp only
  split
  · refine congrArg₂ (x ch) ?_ ?_ <;> (apply Fin.ext; simp only []; omega)
  · rename_i hcond
    exact absurd (by
      have := hi.isLt; have := wi.isLt
      refine ⟨?_, ?_, ?_, ?_⟩ <;> omega) hcond

/-- The flat centre-tap depthwise, in cell coordinates. -/
theorem depthwiseFlat_ctDW {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec c) (v : Vec (c * h * w)) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (depthwiseFlat (h := h) (w := w) (ctDW c kH kW s) b v) ch hi wi
      = b ch + s * Tensor3.unflatten v ch hi wi := by
  simp only [depthwiseFlat, Tensor3.unflatten_flatten]
  exact depthwise2d_ctDW hkH hkW s b _ ch hi wi

/-- The batched centre-tap depthwise — the carrier's depthwise step at stride 1. -/
theorem bcell_dw_ctDW {N c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec c)
    (x : Vec (N * (c * h * w))) (n : Fin N) (ch : Fin c) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.batchMap N (depthwiseFlat (h := h) (w := w) (ctDW c kH kW s) b) x) n ch i j
      = b ch + s * bcell x n ch i j := by
  rw [bcell_batchMap]
  exact depthwiseFlat_ctDW hkH hkW s b _ ch i j

/-- The strided XLA-`SAME` centre-tap depthwise, at the odd position. -/
theorem depthwiseStride2FlatXla_ctDW {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec c) (v : Vec (c * (2 * h) * (2 * w))) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (depthwiseStride2FlatXla (h := h) (w := w) (ctDW c kH kW s) b v) ch hi wi
      = b ch + s * (Tensor3.unflatten v : Tensor3 c (2 * h) (2 * w)) ch
          ⟨2 * hi.val + 1, by have := hi.isLt; omega⟩
          ⟨2 * wi.val + 1, by have := wi.isLt; omega⟩ := by
  simp only [depthwiseStride2FlatXla, Function.comp_apply]
  rw [decimateOdd_unflatten, depthwiseFlat_ctDW hkH hkW]

/-- The batched strided XLA-`SAME` centre-tap depthwise — MobileNetV2's four downsampling blocks. -/
theorem bcell_dwS2Xla_ctDW {N c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (n : Fin N) (ch : Fin c) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.batchMap N
        (depthwiseStride2FlatXla (h := h) (w := w) (ctDW c kH kW s) b) x) n ch i j
      = b ch + s * bcell x n ch
          ⟨2 * i.val + 1, by have := i.isLt; omega⟩
          ⟨2 * j.val + 1, by have := j.isLt; omega⟩ := by
  rw [bcell_batchMap]
  exact depthwiseStride2FlatXla_ctDW hkH hkW s b _ ch i j

/-- **Cellwise ⇒ flatwise.** A property of every cell `(n, c, i, j)` holds at every flat index —
    the bridge from the `bcell` view back to the `∀ k` shape every clause is stated in. -/
theorem forall_flat_of_cell {N c h w : Nat} {v : Vec (N * (c * h * w))} {P : ℝ → Prop}
    (hc : ∀ (n : Fin N) (ci : Fin c) (i : Fin h) (j : Fin w), P (bcell v n ci i j)) :
    ∀ k, P (v k) := by
  intro k
  obtain ⟨⟨n, r⟩, rfl⟩ := finProdFinEquiv.surjective k
  obtain ⟨⟨ch, jj⟩, rfl⟩ := finProdFinEquiv.surjective r
  obtain ⟨⟨ci, ii⟩, rfl⟩ := finProdFinEquiv.surjective ch
  exact hc n ci ii jj

-- ════════════════════════════════════════════════════════════════
-- § 5. `bnRowLA` is continuous (the other continuity, BN-bound and pool-shift facts the ray
--   argument uses are in their op files: `BatchNorm`, `MLP`, `Residual`, `MaxPool3s2`, `CNN`,
--   `Batched`)
-- ════════════════════════════════════════════════════════════════

/-- `bnRowLA` is continuous in the activation — it is a reindex. -/
@[fun_prop]
theorem bnRowLA_continuous (N oc h w : Nat) (c : Fin oc) :
    Continuous (fun v : Vec (N * (oc * h * w)) => bnRowLA N oc h w v c) := by
  refine continuous_pi (fun q => ?_)
  exact continuous_apply _

-- ════════════════════════════════════════════════════════════════
-- § 6. Channel-constant parameters and zeroed kernels
-- ════════════════════════════════════════════════════════════════

/-- A channel-constant BN parameter. Every structural witness's `γ` and `β` are one of these,
    which is what lets `bnBatchLA_pointwise`'s property be channel-independent. -/
noncomputable def kv (c : Nat) (x : ℝ) : Vec c := fun _ => x

@[simp] theorem kv_apply (c : Nat) (x : ℝ) (i : Fin c) : kv c x i = x := rfl

/-- The zero kernel — every residual body. -/
noncomputable def zk (oc ic kH kW : Nat) : Kernel4 oc ic kH kW := fun _ _ _ _ => 0

@[simp] theorem zk_apply (oc ic kH kW : Nat) (o : Fin oc) (c : Fin ic) (kh : Fin kH)
    (kw : Fin kW) : zk oc ic kH kW o c kh kw = 0 := rfl

/-- The zero **depthwise** kernel — an inverted-residual net's zeroed bodies. -/
noncomputable def dzk (c kH kW : Nat) : DepthwiseKernel c kH kW := fun _ _ _ => 0

@[simp] theorem dzk_apply (c kH kW : Nat) (ch : Fin c) (kh : Fin kH) (kw : Fin kW) :
    dzk c kH kW ch kh kw = 0 := rfl

/-- `1 · √n < 160` whenever `n < 25600` — the margin `bnBatchLA_pos` consumes, at `γ = 1`,
    `β = 160`. Every BN width of a 224×224 ResNet witness clears it (the widest is the stem's
    `2·112² = 25088`). -/
theorem margin160 (n : ℕ) (h : (n : ℝ) < 25600) :
    |(1 : ℝ)| * Real.sqrt ((n : ℕ) : ℝ) < 160 := by
  rw [abs_one, one_mul]
  exact sqrt_lt_param n 160 (by norm_num) (by nlinarith)

/-- `|1/64|·√n < 3` whenever `n < 36864 = (3·64)²` — the relu6 margin `bnBatchLA_smooth6` consumes,
    at `γ = 1/64`, `β = 3`. It clears both of that lemma's hypotheses at once, `β = 3` being the
    centre of `(0, 6)`. Every relu6 BN width of a 224×224 MobileNetV2 witness fits: the widest is
    `2·112² = 25088`, shared by the stem, b1's depthwise and b2's expand. -/
theorem margin192 (n : ℕ) (h : (n : ℝ) < 36864) :
    |(1 / 64 : ℝ)| * Real.sqrt ((n : ℕ) : ℝ) < 3 := by
  have hs : Real.sqrt ((n : ℕ) : ℝ) < 192 := sqrt_lt_param n 192 (by norm_num) (by nlinarith)
  rw [abs_of_pos (by norm_num : (0 : ℝ) < 1 / 64)]
  linarith

-- ════════════════════════════════════════════════════════════════
-- § 7. The ray: a ramp in channel 0, perturbed on example 0
-- ════════════════════════════════════════════════════════════════

/-- The base slab: channel 0 carries the strictly decreasing ramp `−(i·W + j)` — positionally
    injective, which is a stem pool's no-tie condition — and the other two channels are zero (a
    centre-tap stem reads only channel 0). Both examples carry the same slab, so the carrier
    vanishes at `t = 0`. -/
noncomputable def rayRamp (H W : Nat) : Tensor3 3 H W :=
  fun ci i j => if ci.val = 0 then -((i.val : ℝ) * (W : ℝ) + (j.val : ℝ)) else 0

noncomputable def rayBase (H W : Nat) : Vec (2 * (3 * H * W)) := bfrom (fun _ => rayRamp H W)

/-- The perturbation: **all** of example 0's channel 0. Uniform over the spatial grid, so it
    survives a max-pool for every `t` (`maxPool3s2_shift`) with no argmax argument. -/
noncomputable def rayV (H W : Nat) : Vec (2 * (3 * H * W)) :=
  bfrom (fun n ci _ _ => if n.val = 0 ∧ ci.val = 0 then (1 : ℝ) else 0)

/-- The witness input, as a ray through the base. -/
noncomputable def rayX (H W : Nat) (t : ℝ) : Vec (2 * (3 * H * W)) :=
  rayBase H W + t • rayV H W

theorem bcell_rayX (H W : Nat) (t : ℝ) (n : Fin 2) (ci : Fin 3) (i : Fin H) (j : Fin W) :
    bcell (rayX H W t) n ci i j
      = (if ci.val = 0 then -((i.val : ℝ) * (W : ℝ) + (j.val : ℝ)) else 0)
        + t * (if n.val = 0 ∧ ci.val = 0 then (1 : ℝ) else 0) := by
  rw [rayX, bcell_add, bcell_smul, rayBase, rayV, bcell_bfrom, bcell_bfrom]
  rfl

theorem rayX_zero_add (H W : Nat) (t : ℝ) : rayX H W 0 + t • rayV H W = rayX H W t := by
  rw [rayX, rayX, zero_smul, add_zero]

/-- The ray is continuous in its parameter. -/
@[fun_prop]
theorem rayX_continuous (H W : Nat) : Continuous (rayX H W) :=
  continuous_const.add (continuous_id.smul continuous_const)

-- ════════════════════════════════════════════════════════════════
-- § 8. `EDiff` — the batch carrier, and what each op does to it
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The carrier**: example 0's slab is example 1's plus the per-channel constant `δ`. The
    replacement for a channel difference, which per-channel batch BN annihilates.

    ⭐ `δ` is a FUNCTION of the channel, not one scalar, and that is what makes it cheap: BN
    multiplies channel `c`'s offset by `γ_c · istd_c` with no need to prove the channels share an
    `istd`, a centre-tap conv collapses the whole function to `fun _ => s · δ 0`, and only `δ 0` is
    ever read (by the head, and at each channel-changing conv). -/
def EDiff {c h w : Nat} (δ : Fin c → ℝ) (v : Vec (2 * (c * h * w))) : Prop :=
  ∀ (ci : Fin c) (i : Fin h) (j : Fin w),
    bcell v 0 ci i j = bcell v 1 ci i j + δ ci

/-- The ray's carrier: `t` in channel 0, nothing elsewhere. -/
theorem EDiff_rayX (H W : Nat) (t : ℝ) :
    EDiff (fun ci => if ci.val = 0 then t else 0) (rayX H W t) := by
  intro ci i j
  rw [bcell_rayX, bcell_rayX]
  by_cases h : ci.val = 0 <;> simp [h]

/-- A batch-uniform shift (a zeroed residual body) is transparent to the carrier. -/
theorem EDiff_shift {c h w : Nat} (δ : Fin c → ℝ) (v : Vec (2 * (c * h * w))) (s : ℝ)
    (hv : EDiff δ v) : EDiff δ (fun k => v k + s) := by
  intro ci i j
  rw [bcell_shift, bcell_shift, hv ci i j]
  ring

/-- ⭐⭐ **Batch BN scales the carrier by `γ_c · istd_c`** — the two examples share the channel's
    mean and `istd`, so centring keeps their difference (`bnBatchLA_exdiff`). -/
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

/-- A centre-tap conv copies channel 0's offset to **every** output channel, and is transparent to
    the zero padding because only the centre tap is nonzero. -/
theorem EDiff_conv {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc) (δ : Fin ic → ℝ) (δ' : Fin oc → ℝ)
    (v : Vec (2 * (ic * h * w))) (hv : EDiff δ v) (hδ : ∀ o, δ' o = s * δ c₀) :
    EDiff δ' (StableHLO.batchMap 2 (flatConv (h := h) (w := w) (ctK oc ic kH kW s) b) v) := by
  intro o i j
  rw [hδ o, bcell_conv_ctK c₀ hc₀ hkH hkW s b v 0 o i j,
    bcell_conv_ctK c₀ hc₀ hkH hkW s b v 1 o i j, hv c₀ _ _]
  ring

/-- The strided peer of `EDiff_conv`. -/
theorem EDiff_convS2 {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc) (δ : Fin ic → ℝ) (δ' : Fin oc → ℝ)
    (v : Vec (2 * (ic * (2 * h) * (2 * w)))) (hv : EDiff δ v) (hδ : ∀ o, δ' o = s * δ c₀) :
    EDiff δ'
      (StableHLO.batchMap 2 (flatConvStride2 (h := h) (w := w) (ctK oc ic kH kW s) b) v) := by
  intro o i j
  rw [hδ o, bcell_convS2_ctK c₀ hc₀ hkH hkW s b v 0 o i j,
    bcell_convS2_ctK c₀ hc₀ hkH hkW s b v 1 o i j, hv c₀ _ _]
  ring

/-- The XLA-`SAME` strided peer of `EDiff_conv` — MobileNetV2's stem. -/
theorem EDiff_convS2Xla {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc) (δ : Fin ic → ℝ) (δ' : Fin oc → ℝ)
    (v : Vec (2 * (ic * (2 * h) * (2 * w)))) (hv : EDiff δ v) (hδ : ∀ o, δ' o = s * δ c₀) :
    EDiff δ'
      (StableHLO.batchMap 2 (flatConvStride2Xla (h := h) (w := w) (ctK oc ic kH kW s) b) v) := by
  intro o i j
  rw [hδ o, bcell_convS2Xla_ctK c₀ hc₀ hkH hkW s b v 0 o i j,
    bcell_convS2Xla_ctK c₀ hc₀ hkH hkW s b v 1 o i j, hv c₀ _ _]
  ring

/-- ⭐ **A centre-tap depthwise scales the carrier channel by channel.** Unlike `EDiff_conv`, which
    collapses δ to the single value `s * δ c₀` at every output channel, a depthwise reads only its
    own channel, so the whole function δ survives, scaled. -/
theorem EDiff_dw {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec c)
    (δ δ' : Fin c → ℝ) (v : Vec (2 * (c * h * w))) (hv : EDiff δ v)
    (hδ : ∀ ch, δ' ch = s * δ ch) :
    EDiff δ' (StableHLO.batchMap 2 (depthwiseFlat (h := h) (w := w) (ctDW c kH kW s) b) v) := by
  intro ch i j
  rw [hδ ch, bcell_dw_ctDW hkH hkW s b v 0 ch i j, bcell_dw_ctDW hkH hkW s b v 1 ch i j,
    hv ch i j]
  ring

/-- The strided XLA-`SAME` peer of `EDiff_dw`. The carrier is spatially uniform, so decimation —
    whichever phase it keeps — is transparent to it. -/
theorem EDiff_dwS2Xla {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec c)
    (δ δ' : Fin c → ℝ) (v : Vec (2 * (c * (2 * h) * (2 * w)))) (hv : EDiff δ v)
    (hδ : ∀ ch, δ' ch = s * δ ch) :
    EDiff δ'
      (StableHLO.batchMap 2 (depthwiseStride2FlatXla (h := h) (w := w) (ctDW c kH kW s) b) v) := by
  intro ch i j
  rw [hδ ch, bcell_dwS2Xla_ctDW hkH hkW s b v 0 ch i j, bcell_dwS2Xla_ctDW hkH hkW s b v 1 ch i j,
    hv ch _ _]
  ring

/-- The 3×3/s2 pool keeps the carrier, at every `t`. -/
theorem EDiff_pool (c h w : Nat) (δ : Fin c → ℝ) (v : Vec (2 * (c * (2 * h) * (2 * w))))
    (hv : EDiff δ v) : EDiff δ (StableHLO.batchMap 2 (maxPool3s2Flat c h w) v) := by
  intro ci i j
  rw [bcell_pool, bcell_pool]
  exact maxPool3s2_shift (bcell v 0) (bcell v 1) (δ ci) ci (fun r s => hv ci r s) i j

-- ════════════════════════════════════════════════════════════════
-- § 9. The stem's centre-tap conv on the ray, and the pool's no-tie
--   `oc`/`kH`/`kW` are binders: ResNet-34 and ResNet-50 share this stem (64 channels, 7×7/s2)
--   at different spatial nests, and both instantiate these four facts.
-- ════════════════════════════════════════════════════════════════

/-- `W·a + b` determines `a` and `b` when `b < W` (division with remainder). -/
theorem divmod_inj {W a b a' b' : ℕ} (hb : b < W) (hb' : b' < W)
    (h : W * a + b = W * a' + b') : a = a' ∧ b = b' := by
  have hW : 0 < W := by omega
  obtain ⟨h1, h2⟩ := (Nat.div_mod_unique hW).2 ⟨add_comm b (W * a), hb⟩
  obtain ⟨h3, h4⟩ := (Nat.div_mod_unique hW).2 ⟨add_comm b' (W * a'), hb'⟩
  rw [h] at h1 h2
  exact ⟨h1.symm.trans h3, h2.symm.trans h4⟩

/-- The witness's strided centre-tap stem conv — the pre-BN activation on the carrier's path. -/
noncomputable def ctConv (oc kH kW h w : Nat) (t : ℝ) : Vec (2 * (oc * (2 * h) * (2 * w))) :=
  StableHLO.batchMap 2 (flatConvStride2 (h := 2 * h) (w := 2 * w) (ctK oc 3 kH kW 1) (kv oc 0))
    (rayX (2 * (2 * h)) (2 * (2 * w)) t)

/-- The stem BN is strictly positive at every point of the ray (the `β = 160` margin). -/
theorem ctConv_bn_pos (oc kH kW h w : Nat)
    (hm : |(1 : ℝ)| * Real.sqrt ((2 * ((2 * h) * (2 * w)) : ℕ) : ℝ) < 160) (t : ℝ)
    (k : Fin (2 * (oc * (2 * h) * (2 * w)))) :
    0 < StableHLO.bnBatchLA 2 oc (2 * h) (2 * w) 1 (kv oc 1) (kv oc 160) (ctConv oc kH kW h w t) k :=
  bnBatchLA_pos 1 one_pos (kv oc 1) (kv oc 160) 1 160 (fun _ => rfl) (fun _ => rfl) hm _ k

/-- ⭐ **The pre-BN stem activation is positionally injective** within each example and channel:
    the centre tap decimates the ramp, and example 0's uniform `+t` shifts every position alike. -/
theorem ctConv_inj (oc kH kW h w : Nat) (hkH : 0 < kH) (hkW : 0 < kW) (t : ℝ) (n : Fin 2)
    (o : Fin oc) (r r' : Fin (2 * h)) (s s' : Fin (2 * w))
    (heq : bcell (ctConv oc kH kW h w t) n o r s = bcell (ctConv oc kH kW h w t) n o r' s') :
    r = r' ∧ s = s' := by
  rw [ctConv,
    bcell_convS2_ctK (0 : Fin 3) rfl hkH hkW 1 (kv oc 0) (rayX (2 * (2 * h)) (2 * (2 * w)) t) n o r s,
    bcell_convS2_ctK (0 : Fin 3) rfl hkH hkW 1 (kv oc 0) (rayX (2 * (2 * h)) (2 * (2 * w)) t) n o r' s',
    bcell_rayX, bcell_rayX] at heq
  simp only [kv_apply, Fin.val_zero, one_mul, zero_add, ite_true] at heq
  -- the ramp value determines the position: `(2r)·W + 2s` with `2s < W = 2·(2w)`
  have hR : ((2 * r.val * (2 * (2 * w)) + 2 * s.val : ℕ) : ℝ)
      = ((2 * r'.val * (2 * (2 * w)) + 2 * s'.val : ℕ) : ℝ) := by
    push_cast at heq ⊢
    linarith
  have hnat : 2 * r.val * (2 * (2 * w)) + 2 * s.val
      = 2 * r'.val * (2 * (2 * w)) + 2 * s'.val := by exact_mod_cast hR
  have hcomm : (2 * (2 * w)) * (2 * r.val) + 2 * s.val
      = (2 * (2 * w)) * (2 * r'.val) + 2 * s'.val := by
    rw [Nat.mul_comm (2 * (2 * w)) (2 * r.val), Nat.mul_comm (2 * (2 * w)) (2 * r'.val)]
    exact hnat
  have hs := s.isLt
  have hs' := s'.isLt
  obtain ⟨h1, h2⟩ := divmod_inj (W := 2 * (2 * w)) (by omega) (by omega) hcomm
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

/-- ⭐ **The stem pool has no tie** at the witness: BN is injective within a channel
    (`bnBatchLA_cell_inj`) and the pre-BN activation is positionally injective. Stated in the
    `∀ example, MaxPool3s2Smooth (slab)` shape the nets' `*PoolSmoothAt` unfolds to. -/
theorem ctConv_pool_smooth (oc kH kW h w : Nat) (hkH : 0 < kH) (hkW : 0 < kW) (t : ℝ) :
    ∀ n : Fin 2, MaxPool3s2Smooth
      (bcell (StableHLO.bnBatchLA 2 oc (2 * h) (2 * w) 1 (kv oc 1) (kv oc 160)
        (ctConv oc kH kW h w t)) n) := by
  intro n
  refine maxPool3s2Smooth_of_injective _ (fun o r r' s s' heq => ?_)
  exact ctConv_inj oc kH kW h w hkH hkW t n o r r' s s'
    (bnBatchLA_cell_inj 1 one_pos (kv oc 1) (kv oc 160) (ctConv oc kH kW h w t) n o
      (by simp only [kv_apply]; norm_num) r r' s s' heq)

-- ════════════════════════════════════════════════════════════════
-- § 10. The head reads the carrier off channel 0
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **GAP and the dense head deliver the carrier to one class**: GAP of a uniformly shifted
    channel is shifted by the same constant, and a `Wd` that reads channel `c₀` into class `j`
    turns the per-channel carrier into `δ c₀`. Stated on the `batchMap`s a `*HeadB` unfolds to, so
    every net's head instantiates it. -/
theorem head_diff_ct {c h w nCls : Nat} (hh : 0 < h) (hw : 0 < w) (c₀ : Fin c) (hc₀ : c₀.val = 0)
    (j : Fin nCls) (Wd : Mat c nCls) (bd : Vec nCls)
    (hWd : ∀ ci, Wd ci j = if ci.val = 0 then (1 : ℝ) else 0) (hbd : bd j = 0)
    (v : Vec (2 * (c * h * w))) (δ : Fin c → ℝ) (hv : EDiff δ v) :
    StableHLO.batchMap 2 (dense Wd bd) (StableHLO.batchMap 2 (globalAvgPoolFlat c h w) v)
        (finProdFinEquiv ((0 : Fin 2), j))
      - StableHLO.batchMap 2 (dense Wd bd) (StableHLO.batchMap 2 (globalAvgPoolFlat c h w) v)
        (finProdFinEquiv ((1 : Fin 2), j))
      = δ c₀ := by
  have hrow : ∀ n : Fin 2,
      Mat.unflatten (StableHLO.batchMap 2 (dense Wd bd)
        (StableHLO.batchMap 2 (globalAvgPoolFlat c h w) v)) n
      = dense Wd bd (globalAvgPool (bcell v n)) := by
    intro n
    rw [row_batchMap, row_batchMap]
    rfl
  have e0 : StableHLO.batchMap 2 (dense Wd bd) (StableHLO.batchMap 2 (globalAvgPoolFlat c h w) v)
      (finProdFinEquiv ((0 : Fin 2), j)) = dense Wd bd (globalAvgPool (bcell v 0)) j :=
    congrFun (hrow 0) j
  have e1 : StableHLO.batchMap 2 (dense Wd bd) (StableHLO.batchMap 2 (globalAvgPoolFlat c h w) v)
      (finProdFinEquiv ((1 : Fin 2), j)) = dense Wd bd (globalAvgPool (bcell v 1)) j :=
    congrFun (hrow 1) j
  have hgap : ∀ ci : Fin c,
      globalAvgPool (bcell v 0) ci = globalAvgPool (bcell v 1) ci + δ ci :=
    fun ci => globalAvgPool_shift hh hw _ _ (δ ci) ci (fun i j => hv ci i j)
  rw [e0, e1]
  simp only [dense, hWd, hbd, add_zero]
  rw [← Finset.sum_sub_distrib]
  rw [Finset.sum_congr rfl (fun ci _ => show
      globalAvgPool (bcell v 0) ci * (if ci.val = 0 then (1 : ℝ) else 0)
        - globalAvgPool (bcell v 1) ci * (if ci.val = 0 then (1 : ℝ) else 0)
      = δ ci * (if ci.val = 0 then (1 : ℝ) else 0) from by rw [hgap ci]; ring)]
  refine (Finset.sum_eq_single_of_mem c₀ (Finset.mem_univ _) ?_).trans ?_
  · intro ci _ hci
    have hc : ci.val ≠ 0 := fun hz => hci (Fin.ext (hz.trans hc₀.symm))
    simp [hc]
  · simp [hc₀]

-- ════════════════════════════════════════════════════════════════
-- § 11. The SYMMETRIC strided depthwise — `EDiff_dwS2Xla`'s peer
--   ⚠ MobileNetV2's strided depthwises are XLA-`SAME` (odd decimation); MobileNetV4's are
--   symmetric (even). One lemma per padding token, as at the strided convs.
-- ════════════════════════════════════════════════════════════════
theorem depthwiseStride2Flat_ctDW {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW)
    (s : ℝ) (b : Vec c) (v : Vec (c * (2 * h) * (2 * w))) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten (depthwiseStride2Flat (h := h) (w := w) (ctDW c kH kW s) b v) ch hi wi
      = b ch + s * (Tensor3.unflatten v : Tensor3 c (2 * h) (2 * w)) ch
          ⟨2 * hi.val, by have := hi.isLt; omega⟩
          ⟨2 * wi.val, by have := wi.isLt; omega⟩ := by
  simp only [depthwiseStride2Flat, Function.comp_apply]
  rw [decimate_unflatten, depthwiseFlat_ctDW hkH hkW]

theorem bcell_dwS2_ctDW {N c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (n : Fin N) (ch : Fin c) (i : Fin h) (j : Fin w) :
    bcell (StableHLO.batchMap N
        (depthwiseStride2Flat (h := h) (w := w) (ctDW c kH kW s) b) x) n ch i j
      = b ch + s * bcell x n ch
          ⟨2 * i.val, by have := i.isLt; omega⟩ ⟨2 * j.val, by have := j.isLt; omega⟩ := by
  rw [bcell_batchMap]
  exact depthwiseStride2Flat_ctDW hkH hkW s b _ ch i j

theorem EDiff_dwS2 {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec c)
    (δ δ' : Fin c → ℝ) (v : Vec (2 * (c * (2 * h) * (2 * w)))) (hv : EDiff δ v)
    (hδ : ∀ ch, δ' ch = s * δ ch) :
    EDiff δ'
      (StableHLO.batchMap 2 (depthwiseStride2Flat (h := h) (w := w) (ctDW c kH kW s) b) v) := by
  intro ch i j
  rw [hδ ch, bcell_dwS2_ctDW hkH hkW s b v 0 ch i j, bcell_dwS2_ctDW hkH hkW s b v 1 ch i j,
    hv ch _ _]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 12. `BUnif` — the carrier that also carries the VALUES
--   ⭐⭐ `EDiff` tracks only the gap between the two examples, which is all a relu-in-the-window
--   or a centre-tap conv needs. A stage that is smooth but NOT affine — MobileNetV4's swish —
--   changes the gap by an amount that depends on the values themselves, so the carrier has to
--   know them. `BUnif` does: each example's slab is CONSTANT over the grid, one value per
--   channel, which every op in these nets preserves and `bnBatchLA` makes symmetric about `β`.
-- ════════════════════════════════════════════════════════════════
def BUnif {c h w : Nat} (a : Fin 2 → Fin c → ℝ) (v : Vec (2 * (c * h * w))) : Prop :=
  ∀ (n : Fin 2) (ci : Fin c) (i : Fin h) (j : Fin w), bcell v n ci i j = a n ci

-- ════════════════════════════════════════════════════════════════
-- § One carrier stage: a centre-tap op at scale 1, bias 0, then its BatchNorm at ε = 1
--   The per-net seals walk the carrier through ~45 of these; each is the op's `EDiff`, fed to
--   `EDiff_bn` with `γ = kv γ0`. `Z` is the BN input as the net spells it (`hZ` by `rfl`).
-- ════════════════════════════════════════════════════════════════

/-- A centre-tap 1×1 (or `kH×kW`) conv, then its BatchNorm: the carrier is channel `c₀`'s offset
    times `γ0 · istd`, at every output channel. -/
theorem EDiff_convBn {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0) (hkH : 0 < kH)
    (hkW : 0 < kW) (γ0 β0 : ℝ) (Z : Vec (2 * (oc * h * w))) {δ : Fin ic → ℝ}
    {δ' : Fin oc → ℝ} {v : Vec (2 * (ic * h * w))} (hv : EDiff δ v)
    (hZ : Z = StableHLO.batchMap 2 (flatConv (h := h) (w := w) (ctK oc ic kH kW 1) (kv oc 0)) v)
    (hδ : ∀ ci, δ' ci = γ0 * δ c₀ * bnIstd (2 * (h * w)) (bnRowLA 2 oc h w Z ci) 1) :
    EDiff δ' (StableHLO.bnBatchLA 2 oc h w 1 (kv oc γ0) (kv oc β0) Z) := by
  subst hZ
  refine EDiff_bn oc h w 1 (kv oc γ0) (kv oc β0) (fun _ => 1 * δ c₀) δ' _
    (EDiff_conv c₀ hc₀ hkH hkW 1 (kv oc 0) δ _ v hv (fun _ => rfl)) (fun ci => ?_)
  rw [hδ ci, kv_apply]
  ring

/-- The strided peer of `EDiff_convBn`. -/
theorem EDiff_convS2Bn {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0) (hkH : 0 < kH)
    (hkW : 0 < kW) (γ0 β0 : ℝ) (Z : Vec (2 * (oc * h * w))) {δ : Fin ic → ℝ}
    {δ' : Fin oc → ℝ} {v : Vec (2 * (ic * (2 * h) * (2 * w)))} (hv : EDiff δ v)
    (hZ : Z = StableHLO.batchMap 2
      (flatConvStride2 (h := h) (w := w) (ctK oc ic kH kW 1) (kv oc 0)) v)
    (hδ : ∀ ci, δ' ci = γ0 * δ c₀ * bnIstd (2 * (h * w)) (bnRowLA 2 oc h w Z ci) 1) :
    EDiff δ' (StableHLO.bnBatchLA 2 oc h w 1 (kv oc γ0) (kv oc β0) Z) := by
  subst hZ
  refine EDiff_bn oc h w 1 (kv oc γ0) (kv oc β0) (fun _ => 1 * δ c₀) δ' _
    (EDiff_convS2 c₀ hc₀ hkH hkW 1 (kv oc 0) δ _ v hv (fun _ => rfl)) (fun ci => ?_)
  rw [hδ ci, kv_apply]
  ring

/-- A centre-tap depthwise conv, then its BatchNorm: channel by channel, `δ ci · γ0 · istd`. -/
theorem EDiff_dwBn {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (γ0 β0 : ℝ)
    (Z : Vec (2 * (c * h * w))) {δ δ' : Fin c → ℝ} {v : Vec (2 * (c * h * w))} (hv : EDiff δ v)
    (hZ : Z = StableHLO.batchMap 2 (depthwiseFlat (h := h) (w := w) (ctDW c kH kW 1) (kv c 0)) v)
    (hδ : ∀ ci, δ' ci = γ0 * δ ci * bnIstd (2 * (h * w)) (bnRowLA 2 c h w Z ci) 1) :
    EDiff δ' (StableHLO.bnBatchLA 2 c h w 1 (kv c γ0) (kv c β0) Z) := by
  subst hZ
  refine EDiff_bn c h w 1 (kv c γ0) (kv c β0) (fun ch => 1 * δ ch) δ' _
    (EDiff_dw hkH hkW 1 (kv c 0) δ _ v hv (fun _ => rfl)) (fun ci => ?_)
  rw [hδ ci, kv_apply]
  ring

/-- The strided peer of `EDiff_dwBn`. -/
theorem EDiff_dwS2Bn {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (γ0 β0 : ℝ)
    (Z : Vec (2 * (c * h * w))) {δ δ' : Fin c → ℝ} {v : Vec (2 * (c * (2 * h) * (2 * w)))}
    (hv : EDiff δ v)
    (hZ : Z = StableHLO.batchMap 2
      (depthwiseStride2Flat (h := h) (w := w) (ctDW c kH kW 1) (kv c 0)) v)
    (hδ : ∀ ci, δ' ci = γ0 * δ ci * bnIstd (2 * (h * w)) (bnRowLA 2 c h w Z ci) 1) :
    EDiff δ' (StableHLO.bnBatchLA 2 c h w 1 (kv c γ0) (kv c β0) Z) := by
  subst hZ
  refine EDiff_bn c h w 1 (kv c γ0) (kv c β0) (fun ch => 1 * δ ch) δ' _
    (EDiff_dwS2 hkH hkW 1 (kv c 0) δ _ v hv (fun _ => rfl)) (fun ci => ?_)
  rw [hδ ci, kv_apply]
  ring

/-- The XLA-`SAME` strided peer of `EDiff_dwBn`. -/
theorem EDiff_dwS2XlaBn {c h w kH kW : Nat} (hkH : 0 < kH) (hkW : 0 < kW) (γ0 β0 : ℝ)
    (Z : Vec (2 * (c * h * w))) {δ δ' : Fin c → ℝ} {v : Vec (2 * (c * (2 * h) * (2 * w)))}
    (hv : EDiff δ v)
    (hZ : Z = StableHLO.batchMap 2
      (depthwiseStride2FlatXla (h := h) (w := w) (ctDW c kH kW 1) (kv c 0)) v)
    (hδ : ∀ ci, δ' ci = γ0 * δ ci * bnIstd (2 * (h * w)) (bnRowLA 2 c h w Z ci) 1) :
    EDiff δ' (StableHLO.bnBatchLA 2 c h w 1 (kv c γ0) (kv c β0) Z) := by
  subst hZ
  refine EDiff_bn c h w 1 (kv c γ0) (kv c β0) (fun ch => 1 * δ ch) δ' _
    (EDiff_dwS2Xla hkH hkW 1 (kv c 0) δ _ v hv (fun _ => rfl)) (fun ci => ?_)
  rw [hδ ci, kv_apply]
  ring

theorem EDiff_of_BUnif {c h w : Nat} (a : Fin 2 → Fin c → ℝ) (δ : Fin c → ℝ)
    (v : Vec (2 * (c * h * w))) (hv : BUnif (h := h) (w := w) a v)
    (hδ : ∀ ci, δ ci = a 0 ci - a 1 ci) : EDiff δ v := by
  intro ci i j
  rw [hv 0 ci i j, hv 1 ci i j, hδ ci]
  ring

theorem BUnif_convS2Xla {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc) (a : Fin 2 → Fin ic → ℝ)
    (a' : Fin 2 → Fin oc → ℝ) (v : Vec (2 * (ic * (2 * h) * (2 * w))))
    (hv : BUnif (h := 2 * h) (w := 2 * w) a v) (ha : ∀ n o, a' n o = b o + s * a n c₀) :
    BUnif (h := h) (w := w) a'
      (StableHLO.batchMap 2 (flatConvStride2Xla (h := h) (w := w) (ctK oc ic kH kW s) b) v) := by
  intro n o i j
  rw [bcell_convS2Xla_ctK c₀ hc₀ hkH hkW s b v n o i j, hv n c₀ _ _, ha n o]

theorem BUnif_convS2 {ic oc h w kH kW : Nat} (c₀ : Fin ic) (hc₀ : c₀.val = 0)
    (hkH : 0 < kH) (hkW : 0 < kW) (s : ℝ) (b : Vec oc) (a : Fin 2 → Fin ic → ℝ)
    (a' : Fin 2 → Fin oc → ℝ) (v : Vec (2 * (ic * (2 * h) * (2 * w))))
    (hv : BUnif (h := 2 * h) (w := 2 * w) a v) (ha : ∀ n o, a' n o = b o + s * a n c₀) :
    BUnif (h := h) (w := w) a'
      (StableHLO.batchMap 2 (flatConvStride2 (h := h) (w := w) (ctK oc ic kH kW s) b) v) := by
  intro n o i j
  rw [bcell_convS2_ctK c₀ hc₀ hkH hkW s b v n o i j, hv n c₀ _ _, ha n o]

/-- a pointwise activation preserves `BUnif`, value by value. -/
theorem BUnif_map {c h w : Nat} (f : ℝ → ℝ) (a a' : Fin 2 → Fin c → ℝ)
    (v : Vec (2 * (c * h * w))) (hv : BUnif (h := h) (w := w) a v)
    (ha : ∀ n ci, a' n ci = f (a n ci)) :
    BUnif (h := h) (w := w) a' (fun k => f (v k)) := by
  intro n ci i j
  rw [ha n ci, ← hv n ci i j]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 13. Batch BN on a `BUnif` slab: ⭐⭐ the two values straddle `β`
--   With both slabs grid-constant the channel's mean is their midpoint, so the outputs are
--   `β ± γ·(gap/2)·istd`. THAT is what lets a non-affine activation be crossed: its two outputs
--   are a function of the gap alone.
-- ════════════════════════════════════════════════════════════════
theorem bnBatchLA_pair {oc h w : Nat} (hhw : 0 < h * w) (ε : ℝ) (γ β : Vec oc)
    (a : Fin 2 → Fin oc → ℝ) (v : Vec (2 * (oc * h * w)))
    (hv : BUnif (h := h) (w := w) a v) (a' : Fin 2 → Fin oc → ℝ)
    (ha : ∀ n ci, a' n ci = β ci + γ ci * ((a n ci - (a 0 ci + a 1 ci) / 2) *
            bnIstd (2 * (h * w)) (bnRowLA 2 oc h w v ci) ε)) :
    BUnif (h := h) (w := w) a' (StableHLO.bnBatchLA 2 oc h w ε γ β v) := by
  intro n ci i j
  have hz : ∀ (nn : Fin 2) (q : Fin (h * w)),
      bnRowLA 2 oc h w v ci (finProdFinEquiv (nn, q)) = a nn ci := by
    intro nn q
    obtain ⟨⟨ii, jj⟩, rfl⟩ := finProdFinEquiv.surjective q
    rw [bnRowLA_apply]
    exact hv nn ci ii jj
  rw [bnBatchLA_bcell, ha n ci]
  simp only [bnForward, bnXhat, bnMean_pair (h * w) hhw (fun nn => a nn ci) _ hz, hz n]
  ring

-- ════════════════════════════════════════════════════════════════
-- § 14. `swish` — the one activation no window makes the identity
--   Used only by MobileNetV4, whose fused stage is swish where every other stage in every other
--   net is relu or relu6. Its scalar facts (`hasDerivAt_swishScalar`, `swishScalarDeriv_pos`,
--   `swishScalar_lt`) are in `Architectures/LayerNorm`, beside `swishScalar`.
-- ════════════════════════════════════════════════════════════════
/-- the two examples' swish outputs, as a function of half their gap. -/
noncomputable def swishGap (β u : ℝ) : ℝ := swishScalar (β + u) - swishScalar (β - u)

@[simp] theorem swishGap_zero (β : ℝ) : swishGap β 0 = 0 := by
  simp only [swishGap, add_zero, sub_zero, sub_self]

theorem hasDerivAt_swishGap (β : ℝ) : HasDerivAt (swishGap β) (2 * swishScalarDeriv β) 0 := by
  have hp : HasDerivAt (fun u : ℝ => swishScalar (β + u)) (swishScalarDeriv β) 0 := by
    have h1 : HasDerivAt (fun u : ℝ => β + u) 1 0 := (hasDerivAt_id (0:ℝ)).const_add β
    have h2 := HasDerivAt.comp (0:ℝ) (hasDerivAt_swishScalar (β + 0)) h1
    rw [add_zero, mul_one] at h2
    exact h2
  have hm : HasDerivAt (fun u : ℝ => swishScalar (β - u)) (-swishScalarDeriv β) 0 := by
    have h1 : HasDerivAt (fun u : ℝ => β - u) (-1) 0 := (hasDerivAt_id (0:ℝ)).const_sub β
    have h2 := HasDerivAt.comp (0:ℝ) (hasDerivAt_swishScalar (β - 0)) h1
    rw [sub_zero, mul_neg, mul_one] at h2
    exact h2
  have h3 : HasDerivAt (fun u : ℝ => swishScalar (β + u) - swishScalar (β - u))
      (swishScalarDeriv β - -swishScalarDeriv β) 0 := hp.sub hm
  rw [sub_neg_eq_add, ← two_mul] at h3
  exact h3

theorem swishGap_pos {β u : ℝ} (hu : 0 < u) (hub : u ≤ β) : 0 < swishGap β u := by
  have h1 : (0:ℝ) ≤ β - u := by linarith
  have h2 : β - u < β + u := by linarith
  have := swishScalar_lt h1 h2
  simp only [swishGap]
  linarith

end BatchSeal
namespace R34FullBSeal

/-! ### Stage facts first needed by ResNet-34's seal, shared by every conv-net seal

The zero-kernel collapse, the relu-free strided stage, the centre-tap projection witness
(`sealProj`). Stage continuity needs no lemma here: the op files tag their continuity facts
`@[fun_prop]`, and each seal's `Rr_continuous` is one `fun_prop`. The namespace is ResNet-34's, kept so that every
citation keeps its name. -/

open scoped BigOperators
open BatchSeal

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

/-- A strided conv-bn-relu stage whose BN is everywhere positive has no relu left. -/
theorem cbReluStridedB_eq {N ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (γ β : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hp : ∀ k, 0 < StableHLO.bnBatchLA N oc h w ε γ β
      (StableHLO.batchMap N (flatConvStride2 W b) x) k) :
    StableHLO.cbReluStridedB N (h := h) (w := w) W b ε γ β x
      = StableHLO.bnBatchLA N oc h w ε γ β (StableHLO.batchMap N (flatConvStride2 W b) x) :=
  relu_id_of_pos hp

/-- `sealProj`, unfolded — bn of the centre-tap strided conv. -/
theorem sealProj_apply (N h w ic oc : Nat) (v : Vec (N * (ic * (2 * h) * (2 * w)))) :
    sealProj N h w ic oc v
      = StableHLO.bnBatchLA N oc h w 1 (kv oc 1) (kv oc 160)
          (StableHLO.batchMap N (flatConvStride2 (ctK oc ic 1 1 1) (kv oc 0)) v) := rfl

end R34FullBSeal

end Proofs
