import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.MaxPool3s2
import LeanMlir.Proofs.Nets.ResNet.ResNet34

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
  (hence positivity, hence relu off its kink **at every input**), the example-difference identity,
  and within-example injectivity (the stem pool's no-tie).
* §3 the centre-tap kernel `ctK` and its conv value: the one weight shape that carries a signal
  through a channel-changing conv while staying transparent to a uniform offset. ⚠ Only the
  **centre** tap is nonzero, which is what makes it padding-proof — a conv of a constant is not
  constant near a zero-padded border, but a centre tap is always in range.
* §4 the 3×3/s2 pool: it shifts with a uniform offset (`maxPool3s2_shift`, no argmax argument), and
  it preserves nonnegativity.
* §5 continuity odds and ends for the ray argument (`relu`, `bnIstd`).
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

/-- The flattened slab is the `Mat` row — the form every per-example op consumes. -/
theorem flatten_bcell {N c h w : Nat} (v : Vec (N * (c * h * w))) (n : Fin N) :
    Tensor3.flatten (bcell v n) = Mat.unflatten v n := by
  simp only [bcell, Tensor3.flatten_unflatten]

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

/-- **BN acts on coordinate differences by `γ·istd`** — the exact identity that propagates the
    carrier undamped through every BN, and the reason no BN-variance derivative is ever taken:
    the difference is `γ · (difference) · istd` with `istd` evaluated at the *same* activation. -/
theorem bnForward_chan_diff_γ {n : Nat} (ε γ β : ℝ) (z : Vec n) (k₀ k₁ : Fin n) :
    bnForward n ε γ β z k₀ - bnForward n ε γ β z k₁ = γ * (z k₀ - z k₁) * bnIstd n z ε := by
  simp only [bnForward, bnXhat]; ring

/-- **The two-sided BN margin** `|bn − β| ≤ |γ|·√n`, with no mean/variance computation
    (`bnXhat_sq_le`). `bnForward_lb`'s symmetric form; what makes a large `β` keep a relu off its
    kink at **every** input, so the structural net needs no eventually-argument for its relus. -/
theorem bnForward_abs_sub_le {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (v : Vec n) (k : Fin n) :
    |bnForward n ε γ β v k - β| ≤ |γ| * Real.sqrt (n : ℝ) := by
  have habs : |bnXhat n ε v k| ≤ Real.sqrt (n : ℝ) := Real.abs_le_sqrt (bnXhat_sq_le ε hε v k)
  have he : bnForward n ε γ β v k - β = γ * bnXhat n ε v k := by simp only [bnForward]; ring
  rw [he, abs_mul]
  exact mul_le_mul_of_nonneg_left habs (abs_nonneg γ)

/-- `√n < β` from `n < β²` — the margin check at each of the witness's BN widths. -/
theorem sqrt_lt_param (n : ℕ) (β : ℝ) (hβ : 0 ≤ β) (h : (n : ℝ) < β ^ 2) :
    Real.sqrt (n : ℝ) < β :=
  (Real.sqrt_lt n.cast_nonneg hβ).2 h

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
-- § 4. The stem pool
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The 3×3/s2 pool shifts with a uniform offset.** If one slab's channel is another's plus
    the constant `δ`, so are their pooled values — `max` of a uniformly shifted family
    (`Finset.apply_sup'_eq_sup'_comp` at `(· + δ)`). No argmax or eventually-argument: this holds at
    every point of the ray, which is what lets the carrier cross the only real kink in the net. -/
theorem maxPool3s2_shift {c h w : Nat} (x y : Tensor3 c (2 * h) (2 * w)) (δ : ℝ) (ci : Fin c)
    (hxy : ∀ r s, x ci r s = y ci r s + δ) (hi : Fin h) (wi : Fin w) :
    maxPool3s2 x ci hi wi = maxPool3s2 y ci hi wi + δ := by
  have hg : ∀ p q : ℝ, (p ⊔ q) + δ = (p + δ) ⊔ (q + δ) := fun p q => (max_add_add_right p q δ).symm
  simp only [maxPool3s2, hxy]
  exact (Finset.apply_sup'_eq_sup'_comp Finset.univ_nonempty (fun z : ℝ => z + δ) hg).symm

/-- The pool keeps a nonnegative slab nonnegative (it selects a window cell). -/
theorem maxPool3s2_nonneg {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (hx : ∀ ci r s, 0 ≤ x ci r s) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    0 ≤ maxPool3s2 x ci hi wi :=
  le_trans (hx _ _ _) (le_maxPool3s2 x ci hi wi (0, 0))

-- ════════════════════════════════════════════════════════════════
-- § 4b. The head
-- ════════════════════════════════════════════════════════════════

/-- **GAP of a uniformly shifted channel is shifted by the same constant.** -/
theorem globalAvgPool_shift {c h w : Nat} (hh : 0 < h) (hw : 0 < w) (x y : Tensor3 c h w) (δ : ℝ)
    (ci : Fin c) (hxy : ∀ i j, x ci i j = y ci i j + δ) :
    globalAvgPool x ci = globalAvgPool y ci + δ := by
  have hh' : ((h : ℝ)) ≠ 0 := Nat.cast_ne_zero.mpr hh.ne'
  have hw' : ((w : ℝ)) ≠ 0 := Nat.cast_ne_zero.mpr hw.ne'
  simp only [globalAvgPool, hxy, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp

-- ════════════════════════════════════════════════════════════════
-- § 5. Continuity odds and ends (the ray argument)
-- ════════════════════════════════════════════════════════════════

/-- `relu` is continuous everywhere — it is `max · 0`; only its *derivative* has a kink. -/
theorem relu_continuous (n : Nat) : Continuous (relu n) := by
  refine continuous_pi (fun k => ?_)
  have he : (fun x : Vec n => relu n x k) = fun x : Vec n => max (x k) 0 := by
    funext x
    simp only [relu]
    split_ifs with h
    · exact (max_eq_left h.le).symm
    · exact (max_eq_right (not_lt.mp h)).symm
  rw [he]
  exact (continuous_apply k).max continuous_const

/-- A residual branch is continuous when its body is. -/
theorem residual_continuous {n : Nat} (F : Vec n → Vec n) (hF : Continuous F) :
    Continuous (residual F) :=
  continuous_pi (fun k => ((continuous_apply k).comp hF).add (continuous_apply k))

/-- A projected residual is continuous when both branches are. -/
theorem residualProj_continuous {m n : Nat} (P F : Vec m → Vec n) (hP : Continuous P)
    (hF : Continuous F) : Continuous (residualProj P F) :=
  continuous_pi (fun k => ((continuous_apply k).comp hP).add ((continuous_apply k).comp hF))

/-- `bnIstd` is continuous in the activation (`ε > 0`). -/
theorem bnIstd_cont {n : Nat} (ε : ℝ) (hε : 0 < ε) (k : Fin n) :
    Continuous (fun v : Vec n => bnIstd n v ε) :=
  (continuous_apply k).comp (bnIstdBroadcast_diff n ε hε).continuous

/-- `bnRowLA` is continuous in the activation — it is a reindex. -/
theorem bnRowLA_continuous (N oc h w : Nat) (c : Fin oc) :
    Continuous (fun v : Vec (N * (oc * h * w)) => bnRowLA N oc h w v c) := by
  refine continuous_pi (fun q => ?_)
  exact continuous_apply _

/-- `maxPool3s2Flat` is continuous (a `sup'` of coordinates). -/
theorem maxPool3s2Flat_continuous (c h w : Nat) : Continuous (maxPool3s2Flat c h w) := by
  refine continuous_pi (fun k => ?_)
  show Continuous (fun v => Tensor3.flatten (maxPool3s2 (Tensor3.unflatten v)) k)
  simp only [Tensor3.flatten, maxPool3s2, Tensor3.unflatten]
  exact Continuous.finset_sup'_apply Finset.univ_nonempty (fun ab _ => continuous_apply _)

/-- `batchMap` of a continuous per-example op is continuous. -/
theorem batchMap_continuous {N a b : Nat} (f : Vec a → Vec b) (hf : Continuous f) :
    Continuous (StableHLO.batchMap N f) := by
  refine continuous_pi (fun k => ?_)
  exact ((continuous_apply _).comp hf).comp (continuous_pi (fun _ => continuous_apply _))

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

/-- `1 · √n < 160` whenever `n < 25600` — the margin `bnBatchLA_pos` consumes, at `γ = 1`,
    `β = 160`. Every BN width of a 224×224 ResNet witness clears it (the widest is the stem's
    `2·112² = 25088`). -/
theorem margin160 (n : ℕ) (h : (n : ℝ) < 25600) :
    |(1 : ℝ)| * Real.sqrt ((n : ℕ) : ℝ) < 160 := by
  rw [abs_one, one_mul]
  exact sqrt_lt_param n 160 (by norm_num) (by nlinarith)

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

end BatchSeal
end Proofs
