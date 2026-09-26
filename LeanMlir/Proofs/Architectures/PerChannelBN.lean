import LeanMlir.Proofs.Architectures.BatchNorm

/-! # Per-channel BatchNorm — the block-diagonal VJP

Chapters 4–5 used a per-example **global** BatchNorm: one scalar `(γ, β)` over the
whole `oc·h·w` activation (LayerNorm-shaped). Real ResNet wants **per-channel** BN:
normalize each channel-slice independently with its *own* `(γ_c, β_c)`, `γ/β : Vec oc`.

Because each channel is independent, the whole Jacobian is **block-diagonal** across
the channel axis — the genuinely-new piece. We get it for free by generalizing the
existing `rowwiseHasVJPMat` (Tensor.lean, multi-head attention) from a *single*
per-row map to a **per-row family** `g : Fin m → (Vec n → Vec p)`: viewing the
activation as `Mat oc (h·w)` (row = channel), per-channel BN is exactly
`fun A => fun c => bnForward (h·w) ε (γ c) (β c) (A c)`. Its VJP runs each channel's
`bnHasVJP` on that channel's cotangent slice; the cross-channel blocks vanish.

The file also holds inference BN (frozen statistics), batch BN on the `[N,C,H,W]` layout
(`bnBatchTensor4`, chapter 7) and the sync-BN op at supplied statistics (`bnSyncTensor4`, its
γ and input gradients) — the forms `StableHLO`'s `den` reads. Sharding that layout across
replicas (`batchShard` and the shard = global identities) is `DataParallelSync`'s.

Everything closes under `[propext, Classical.choice, Quot.sound]`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Per-row independence (a per-row family generalizing `rowwise`)
-- ════════════════════════════════════════════════════════════════

/-- **Row-wise lifting of a per-row `HasVJP` family.** Each row `r` gets its own map
    `g r` (with its own VJP); the matrix backward runs `(g r).backward` on row `r`'s
    cotangent. The per-row peer of `rowwiseHasVJPMat`. -/
noncomputable def rowwisePerRowHasVJPMat {m n p : Nat} (g : Fin m → (Vec n → Vec p))
    (hg : ∀ r, HasVJP (g r)) (hg_diff : ∀ r, Differentiable ℝ (g r)) :
    HasVJPMat (fun A : Mat m n => fun r => g r (A r)) where
  backward := fun A dY => fun r c => (hg r).backward (A r) (dY r) c
  correct := by
    intro A dY i j
    simp_rw [pdivMat_rowIndep_perRow_at g A (fun r => hg_diff r (A r)), ite_mul, zero_mul,
      Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    exact (hg i).correct (A i) (dY i) j

/-- **A per-row family flattens to a differentiable `Vec → Vec` map.** The
    `Differentiable` witness `vjpCompAt` / the network composition needs to thread a
    per-channel BN through a block. -/
theorem rowwisePerRow_flat_differentiable {m n p : Nat} (g : Fin m → (Vec n → Vec p))
    (h_g_diff : ∀ r, Differentiable ℝ (g r)) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten ((fun A : Mat m n => fun r => g r (A r)) (Mat.unflatten v))) := by
  unfold Mat.flatten Mat.unflatten; fun_prop

-- ════════════════════════════════════════════════════════════════
-- § Per-channel BatchNorm
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel BatchNorm** (matrix view): BN each row (= channel-slice of `m = h·w`
    spatial cells) with its own `(γ_c, β_c)`. The real-ResNet BN that `bnForward`'s
    global scalar version approximates. -/
noncomputable def bnPerChannelMat (oc m : Nat) (ε : ℝ) (γ β : Vec oc) :
    Mat oc m → Mat oc m :=
  fun A => fun c => bnForward m ε (γ c) (β c) (A c)

/-- **Per-channel BN VJP (block-diagonal).** Each channel runs its own `bnHasVJP`;
    the cross-channel Jacobian blocks vanish (`pdivMat_rowIndep_perRow_at`). -/
noncomputable def bnPerChannelMatHasVJP (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJPMat (bnPerChannelMat oc m ε γ β) :=
  rowwisePerRowHasVJPMat (fun c => bnForward m ε (γ c) (β c))
    (fun c => bnHasVJP m ε (γ c) (β c) hε)
    (fun c => bnForward_differentiable m ε (γ c) (β c) hε)

/-- Per-channel BN as a flat-vector op `Vec (oc·m) → Vec (oc·m)` (row-major, channel
    `c` = the `m`-wide slab at flat positions `finProdFinEquiv (c, ·)`). -/
noncomputable def bnPerChannelFlat (oc m : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (oc * m) → Vec (oc * m) :=
  fun v => Mat.flatten (bnPerChannelMat oc m ε γ β (Mat.unflatten v))

/-- **Per-channel BN flat VJP** — the block-diagonal matrix VJP bridged to `Vec`. -/
noncomputable def bnPerChannelFlatHasVJP (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (bnPerChannelFlat oc m ε γ β) :=
  HasVJPMat.toHasVJP (bnPerChannelMatHasVJP oc m ε hε γ β)

/-- **Per-channel BN is differentiable everywhere** (`ε > 0`). The composition witness. -/
theorem bnPerChannelFlat_differentiable (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (bnPerChannelFlat oc m ε γ β) :=
  rowwisePerRow_flat_differentiable (fun c => bnForward m ε (γ c) (β c))
    (fun c => bnForward_differentiable m ε (γ c) (β c) hε)

/-- **Per-channel BN VJP correctness** (ℝ-headline): the flat backward equals the
    `pdiv`-contracted (block-diagonal) Jacobian of per-channel BN. -/
theorem bnPerChannelFlatHasVJP_correct (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v dy : Vec (oc * m)) (i : Fin (oc * m)) :
    (bnPerChannelFlatHasVJP oc m ε hε γ β).backward v dy i =
      ∑ j : Fin (oc * m), pdiv (bnPerChannelFlat oc m ε γ β) v i j * dy j :=
  (bnPerChannelFlatHasVJP oc m ε hε γ β).correct v dy i

-- ════════════════════════════════════════════════════════════════
-- § Renderable closed-form backward (the per-channel consolidated gradient)
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel consolidated BN input-gradient** — the renderable closed form: run
    the per-example three-term `bnGradInput` on each channel-slice (`m = h·w` spatial
    cells), reusing that channel's `γ_c`. This is exactly what a `bnPerChannelBack` SHlo
    op / `renderLNBack`-per-channel emits; the abstract `bnPerChannelFlatHasVJP.backward`
    is the spec it must match. -/
noncomputable def bnPerChannelGradInput (oc m : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (oc * m)) : Vec (oc * m) :=
  fun idx =>
    bnGradInput m ε (γ (finProdFinEquiv.symm idx).1)
      (Mat.unflatten x (finProdFinEquiv.symm idx).1)
      (Mat.unflatten dy (finProdFinEquiv.symm idx).1)
      (finProdFinEquiv.symm idx).2

/-- **Renderable backward is faithful** (ℝ-headline): the per-channel consolidated
    gradient equals the `pdiv`-contracted Jacobian of per-channel BN, under `0 < ε`.
    Each channel reduces to the per-example `bn_input_grad_correct`. The licence to
    render per-channel BN's backward as the three-term formula per channel. -/
theorem bnPerChannelGradInput_correct (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (oc * m)) (i : Fin (oc * m)) :
    bnPerChannelGradInput oc m ε γ x dy i =
      ∑ j : Fin (oc * m), pdiv (bnPerChannelFlat oc m ε γ β) x i j * dy j := by
  rw [← bnPerChannelFlatHasVJP_correct oc m ε hε γ β]
  show bnGradInput m ε (γ (finProdFinEquiv.symm i).1)
        (Mat.unflatten x (finProdFinEquiv.symm i).1) (Mat.unflatten dy (finProdFinEquiv.symm i).1)
        (finProdFinEquiv.symm i).2
      = (bnHasVJP m ε (γ (finProdFinEquiv.symm i).1) (β (finProdFinEquiv.symm i).1) hε).backward
          (Mat.unflatten x (finProdFinEquiv.symm i).1) (Mat.unflatten dy (finProdFinEquiv.symm i).1)
          (finProdFinEquiv.symm i).2
  rw [(bnHasVJP m ε (γ (finProdFinEquiv.symm i).1) (β (finProdFinEquiv.symm i).1) hε).correct,
      ← bn_input_grad_correct m ε (γ (finProdFinEquiv.symm i).1) (β (finProdFinEquiv.symm i).1) hε]

-- ════════════════════════════════════════════════════════════════
-- § Layout bridge: Tensor3 `(oc*h)*w`  ↔  Mat-split `oc*(h*w)`  (B9 entry)
-- ════════════════════════════════════════════════════════════════

/-! The network carries its activations in the **Tensor3** flat layout `(oc*h)*w`
(`flatConv` etc.), but `bnPerChannelFlat` is defined on the **Mat-split** layout
`oc*(h*w)` (row `c` = the `h·w` spatial cells of channel `c`). The two `Vec`s have
the same size; they differ only in how `finProdFinEquiv` associates the product. So
the bridge is a pure **re-association reindex** (a permutation of coordinates) — a
`reindexCLM` whose VJP is the scatter `pdiv_reindex` gives, exactly like
`decimateFlat`. Conjugating `bnPerChannelFlat` by this bridge yields per-channel BN
acting on the network's Tensor3 activations, with its VJP for free via `vjpComp`. -/

/-- Re-association index `Fin (oc*(h*w)) → Fin (oc*h*w)`: a Mat-split flat index
    `(c, s)` with `s ↔ (hi, wi)` maps to the Tensor3 flat index `((c, hi), wi)`. A
    pure product re-association — no arithmetic. -/
noncomputable def reassocFwdIdx (oc h w : Nat) (mIdx : Fin (oc * (h * w))) :
    Fin (oc * h * w) :=
  let cs := finProdFinEquiv.symm mIdx        -- (Fin oc, Fin (h*w))
  let hw := finProdFinEquiv.symm cs.2        -- (Fin h, Fin w)
  finProdFinEquiv (finProdFinEquiv (cs.1, hw.1), hw.2)

/-- Re-association index `Fin (oc*h*w) → Fin (oc*(h*w))`: the inverse direction,
    Tensor3 `((c, hi), wi)` ↦ Mat-split `(c, (hi, wi))`. -/
noncomputable def reassocBackIdx (oc h w : Nat) (t : Fin (oc * h * w)) :
    Fin (oc * (h * w)) :=
  let chw := finProdFinEquiv.symm t          -- (Fin (oc*h), Fin w)
  let ch := finProdFinEquiv.symm chw.1       -- (Fin oc, Fin h)
  finProdFinEquiv (ch.1, finProdFinEquiv (ch.2, chw.2))

/-- The two re-association indices are mutual inverses — the bridge is a genuine
    relabeling (so conjugating by it really *is* per-channel BN, just in Tensor3
    coordinates). Pure `finProdFinEquiv` round-trip. -/
theorem reassocFwdIdx_reassocBackIdx (oc h w : Nat) (t : Fin (oc * h * w)) :
    reassocFwdIdx oc h w (reassocBackIdx oc h w t) = t := by
  unfold reassocFwdIdx reassocBackIdx
  simp only [Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]

theorem reassocBackIdx_reassocFwdIdx (oc h w : Nat) (mIdx : Fin (oc * (h * w))) :
    reassocBackIdx oc h w (reassocFwdIdx oc h w mIdx) = mIdx := by
  unfold reassocFwdIdx reassocBackIdx
  simp only [Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **Tensor3 → Mat-split** reindex: read the `((c,hi),wi)` cell at Mat position
    `(c, (hi,wi))`. A `reindexCLM`, hence continuous-linear / differentiable. -/
noncomputable def reassocFwd (oc h w : Nat) :
    Vec (oc * h * w) → Vec (oc * (h * w)) :=
  fun y k => y (reassocFwdIdx oc h w k)

/-- **Mat-split → Tensor3** reindex (the inverse relabeling). -/
noncomputable def reassocBack (oc h w : Nat) :
    Vec (oc * (h * w)) → Vec (oc * h * w) :=
  fun y k => y (reassocBackIdx oc h w k)

-- ════════════════════════════════════════════════════════════════
-- § Inference-time BN: the same affine map, with the statistics frozen
-- ════════════════════════════════════════════════════════════════

/-! Training BN computes μ/σ² from the activation it is normalizing; inference BN consumes
**frozen** statistics (the driver's EMA'd running mean/var) and is therefore a plain affine
map — pointwise in the activation, with no reduction at all. That is exactly why an eval
forward built on it is class-batch-independent: an example's logits do not depend on which
other examples share its batch.

The chain below mirrors `bnForward → bnPerChannelMat → bnPerChannelFlat → bnPerChannelTensor3`
one-for-one, so the eval op drops into the same layout bridge as the training op. -/

/-- **Inference BN on one channel's `m` activations**: `yᵢ = γ · (xᵢ − μ) · (var + ε)^(−1/2) + β`,
    with `μ`/`var` supplied rather than computed from `x`. The `bnForward` peer — note it takes
    `x` pointwise, where `bnForward` reduces over all of `x` to get its own μ/σ². -/
noncomputable def bnEvalForward (m : Nat) (ε γ β μ v : ℝ) (x : Vec m) : Vec m :=
  fun i => γ * ((x i - μ) * (1 / Real.sqrt (v + ε))) + β

/-- **Per-channel inference BN (Mat layout)** — row `c` gets channel `c`'s frozen stats. -/
noncomputable def bnPerChannelEvalMat (oc m : Nat) (ε : ℝ) (γ β μ v : Vec oc) :
    Mat oc m → Mat oc m :=
  fun A => fun c => bnEvalForward m ε (γ c) (β c) (μ c) (v c) (A c)

/-- **Per-channel inference BN (flat layout)** — the `bnPerChannelFlat` peer. -/
noncomputable def bnPerChannelEvalFlat (oc m : Nat) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (oc * m) → Vec (oc * m) :=
  fun x => Mat.flatten (bnPerChannelEvalMat oc m ε γ β μ v (Mat.unflatten x))

/-- **Per-channel inference BN (Tensor3 layout)** — the `bnPerChannelTensor3` peer, through
    the same `reassoc` bridge. This is what `SHlo.bnPerChannelEvalF` denotes. -/
noncomputable def bnPerChannelEvalTensor3 (oc h w : Nat) (ε : ℝ) (γ β μ v : Vec oc) :
    Vec (oc * h * w) → Vec (oc * h * w) :=
  reassocBack oc h w ∘ (bnPerChannelEvalFlat oc (h * w) ε γ β μ v) ∘ reassocFwd oc h w

/-- **Frozen-stats BN at a channel's own statistics is the training BN.**

    Hand `bnEvalForward` the mean and the second moment of `x` itself and it reproduces
    `bnForward` exactly, the variance arriving through `bnVar_eq_bnMeanSq_sub_sq`.

    This is the `R = 1` anchor of synchronised BatchNorm, and the reason the sync render is
    a drop-in: **a graph that normalises with handed-in statistics denotes the same function
    as one that computes them, whenever the handed-in ones are the right ones.** At `R = 1`
    every `allReduceMeanF` threads its operand, so the sync forward collapses to exactly this
    and the single-device artifacts need not move. -/
theorem bnEvalForward_at_own_stats (n : Nat) (hn : n ≠ 0) (ε γ β : ℝ) (x : Vec n) :
    bnEvalForward n ε γ β (bnMean n x) (bnMeanSq n x - bnMean n x * bnMean n x) x
      = bnForward n ε γ β x := by
  rw [← bnVar_eq_bnMeanSq_sub_sq n hn x]
  funext i
  unfold bnEvalForward bnForward bnXhat bnIstd
  ring

/-- **Inference BN is differentiable everywhere** — it is affine in `x`, so unlike the
    training BN this needs no `0 < ε` hypothesis: `ε` only enters the constant scale factor. -/
theorem bnEvalForward_differentiable (m : Nat) (ε γ β μ v : ℝ) :
    Differentiable ℝ (bnEvalForward m ε γ β μ v) := by
  unfold bnEvalForward; fun_prop

/-- The rendered **per-channel γ gradient**: `dγ_c = Σ_{s} dy_(c,s) · x̂_(c,s)` (the
    `reduce` over batch/spatial of `dy·x̂` that the `bnGammaSgd` op emits).
    `x̂` is recomputed from the saved BN input `v` (the conv output). Lives here (not
    `PerChannelBNGrad`) so the `bnGammaSgd` `SHlo` op's `den` can reference it. -/
noncomputable def bnPerChannelGradGamma (oc m : Nat) (ε : ℝ) (v dy : Vec (oc * m)) : Vec oc :=
  fun c => ∑ s : Fin m, dy (finProdFinEquiv (c, s)) * bnXhat m ε (Mat.unflatten v c) s

/-- The rendered **per-channel β gradient**: `dβ_c = Σ_{s} dy_(c,s)`. -/
noncomputable def bnPerChannelGradBeta (oc m : Nat) (dy : Vec (oc * m)) : Vec oc :=
  fun c => ∑ s : Fin m, dy (finProdFinEquiv (c, s))

theorem reassocFwd_differentiable (oc h w : Nat) :
    Differentiable ℝ (reassocFwd oc h w) :=
  (reindexCLM (reassocFwdIdx oc h w)).differentiable

theorem reassocBack_differentiable (oc h w : Nat) :
    Differentiable ℝ (reassocBack oc h w) :=
  (reindexCLM (reassocBackIdx oc h w)).differentiable

/-- VJP of the forward reindex — the scatter `pdiv_reindex` gives. Mirrors
    `decimateFlatHasVJP`. -/
noncomputable def reassocFwdHasVJP (oc h w : Nat) :
    HasVJP (reassocFwd oc h w) :=
  reindexVJP (reassocFwdIdx oc h w)

noncomputable def reassocBackHasVJP (oc h w : Nat) :
    HasVJP (reassocBack oc h w) :=
  reindexVJP (reassocBackIdx oc h w)

/-- The bridge is a permutation, so each reindex's VJP backward is just the *inverse*
    reindex (the single matching delta survives the scatter). These two collapse the
    `vjpComp` backwards into a clean closed form for `bnPerChannelTensor3`. -/
theorem reassocBackHasVJP_backward_eq (oc h w : Nat) (v : Vec (oc * (h * w)))
    (dy : Vec (oc * h * w)) :
    (reassocBackHasVJP oc h w).backward v dy = reassocFwd oc h w dy :=
  reindexVJP_backward_of_inv _ _ (reassocBackIdx_reassocFwdIdx oc h w)
    (reassocFwdIdx_reassocBackIdx oc h w) v dy

theorem reassocFwdHasVJP_backward_eq (oc h w : Nat) (v : Vec (oc * h * w))
    (dy : Vec (oc * (h * w))) :
    (reassocFwdHasVJP oc h w).backward v dy = reassocBack oc h w dy :=
  reindexVJP_backward_of_inv _ _ (reassocFwdIdx_reassocBackIdx oc h w)
    (reassocBackIdx_reassocFwdIdx oc h w) v dy

-- ════════════════════════════════════════════════════════════════
-- § Per-channel BN on the network's Tensor3 layout (the plug-in op)
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel BatchNorm on the Tensor3 `(oc*h)*w` activation layout.** Conjugate
    the Mat-split `bnPerChannelFlat` by the layout bridge: relabel to Mat-split,
    normalize each channel over its `h·w` spatial cells, relabel back. This is what
    `SHlo.bnPerChannelF` denotes. -/
noncomputable def bnPerChannelTensor3 (oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (oc * h * w) → Vec (oc * h * w) :=
  reassocBack oc h w ∘ (bnPerChannelFlat oc (h * w) ε γ β) ∘ reassocFwd oc h w

@[fun_prop]
theorem bnPerChannelTensor3_differentiable (oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (bnPerChannelTensor3 oc h w ε γ β) := by
  unfold bnPerChannelTensor3
  exact (reassocBack_differentiable oc h w).comp
    ((bnPerChannelFlat_differentiable oc (h * w) ε hε γ β).comp
      (reassocFwd_differentiable oc h w))

/-- **Per-channel BN (Tensor3 layout) VJP** — block-diagonal across channels, lifted
    through the layout bridge by `vjpComp` (twice). -/
noncomputable def bnPerChannelTensor3HasVJP (oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (bnPerChannelTensor3 oc h w ε γ β) :=
  let inner : Vec (oc * h * w) → Vec (oc * (h * w)) :=
    bnPerChannelFlat oc (h * w) ε γ β ∘ reassocFwd oc h w
  let inner_diff : Differentiable ℝ inner :=
    (bnPerChannelFlat_differentiable oc (h * w) ε hε γ β).comp (reassocFwd_differentiable oc h w)
  let inner_vjp : HasVJP inner :=
    vjpComp (reassocFwd oc h w) (bnPerChannelFlat oc (h * w) ε γ β)
      (reassocFwd_differentiable oc h w) (bnPerChannelFlat_differentiable oc (h * w) ε hε γ β)
      (reassocFwdHasVJP oc h w) (bnPerChannelFlatHasVJP oc (h * w) ε hε γ β)
  show HasVJP (reassocBack oc h w ∘ inner) from
  vjpComp inner (reassocBack oc h w) inner_diff (reassocBack_differentiable oc h w)
    inner_vjp (reassocBackHasVJP oc h w)

/-- **Per-channel BN (Tensor3 layout) VJP correctness** (ℝ-headline): the backward
    equals the `pdiv`-contracted (block-diagonal) Jacobian of per-channel BN on the
    network's activation layout. The licence to wire per-channel BN into ResNet-34. -/
theorem bnPerChannelTensor3HasVJP_correct (oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (oc * h * w)) (i : Fin (oc * h * w)) :
    (bnPerChannelTensor3HasVJP oc h w ε hε γ β).backward x dy i =
      ∑ j : Fin (oc * h * w), pdiv (bnPerChannelTensor3 oc h w ε γ β) x i j * dy j :=
  (bnPerChannelTensor3HasVJP oc h w ε hε γ β).correct x dy i

/-- The composed `vjpComp` backward collapses (the bridge reindexes are permutations):
    per-channel BN's Tensor3 backward is the Mat-split block-diagonal backward,
    conjugated by the layout bridge. -/
theorem bnPerChannelTensor3HasVJP_backward_eq (oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (oc * h * w)) :
    (bnPerChannelTensor3HasVJP oc h w ε hε γ β).backward x dy =
      reassocBack oc h w
        ((bnPerChannelFlatHasVJP oc (h * w) ε hε γ β).backward
          (reassocFwd oc h w x) (reassocFwd oc h w dy)) := by
  unfold bnPerChannelTensor3HasVJP
  rw [vjpComp_backward, vjpComp_backward, reassocBackHasVJP_backward_eq,
    reassocFwdHasVJP_backward_eq]

-- ════════════════════════════════════════════════════════════════
-- § Renderable closed-form backward in the Tensor3 layout (B8b's `den` target)
-- ════════════════════════════════════════════════════════════════

/-- **Renderable per-channel BN backward on the Tensor3 `(oc*h)*w` layout** — relabel
    to Mat-split, run the per-channel consolidated three-term `bnPerChannelGradInput`,
    relabel back. This is exactly what the `bnPerChannelBack` SHlo op emits (per-channel
    `renderLNBack`, reducing over the spatial axis); its faithfulness spec is below. -/
noncomputable def bnPerChannelTensor3GradInput (oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (oc * h * w)) : Vec (oc * h * w) :=
  reassocBack oc h w
    (bnPerChannelGradInput oc (h * w) ε γ (reassocFwd oc h w x) (reassocFwd oc h w dy))

/-- **Renderable Tensor3 backward is faithful** (ℝ-headline): equals the
    `pdiv`-contracted (block-diagonal) Jacobian of per-channel BN on the network's
    activation layout, under `0 < ε`. The licence to render per-channel BN's backward
    in ResNet-34. -/
theorem bnPerChannelTensor3GradInput_correct (oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (oc * h * w)) (i : Fin (oc * h * w)) :
    bnPerChannelTensor3GradInput oc h w ε γ x dy i =
      ∑ j : Fin (oc * h * w), pdiv (bnPerChannelTensor3 oc h w ε γ β) x i j * dy j := by
  rw [← bnPerChannelTensor3HasVJP_correct oc h w ε hε γ β,
      bnPerChannelTensor3HasVJP_backward_eq oc h w ε hε γ β]
  show bnPerChannelGradInput oc (h * w) ε γ (reassocFwd oc h w x) (reassocFwd oc h w dy)
        (reassocBackIdx oc h w i)
      = (bnPerChannelFlatHasVJP oc (h * w) ε hε γ β).backward
          (reassocFwd oc h w x) (reassocFwd oc h w dy) (reassocBackIdx oc h w i)
  rw [bnPerChannelGradInput_correct oc (h * w) ε hε γ β,
      bnPerChannelFlatHasVJP_correct oc (h * w) ε hε γ β]

-- ════════════════════════════════════════════════════════════════
-- § Chapter 7 (EfficientNet) — BATCH norm per channel on the [N,C,H,W] layout
--
-- EfficientNet uses batch-norm, not the per-example instance-norm above. The KEY
-- observation: batch-norm normalizes each channel over its `N·H·W` (batch+spatial)
-- elements — which is EXACTLY `bnPerChannelFlat oc m` with `m = N·(h·w)`, the SAME
-- proven block-diagonal VJP, just with the per-channel group enlarged from the
-- spatial cells of one example to the whole batch. So no new BN math is needed; the
-- only new content is the layout bridge `[N,C,H,W] ↔ [C, N·H·W]` (move the channel
-- axis to the front — a TRANSPOSE-reindex, vs the instance-norm bridge's pure
-- re-association). Mirrors `reassocFwd/Back` + `bnPerChannelTensor3` exactly.
-- ════════════════════════════════════════════════════════════════

/-- Transpose-reindex `Fin (oc*(N*(h*w))) → Fin (N*(oc*(h*w)))`: a per-channel Mat
    index `(c, (n, s))` maps to the network `[N,C,H,W]` flat index `(n, (c, s))` —
    swap the batch and channel axes (`s ↔ (hi,wi)` carried along). A permutation. -/
noncomputable def bnchwFwdIdx (N oc h w : Nat) (mIdx : Fin (oc * (N * (h * w)))) :
    Fin (N * (oc * (h * w))) :=
  let cs := finProdFinEquiv.symm mIdx        -- (Fin oc, Fin (N*(h*w)))
  let ns := finProdFinEquiv.symm cs.2        -- (Fin N, Fin (h*w))
  finProdFinEquiv (ns.1, finProdFinEquiv (cs.1, ns.2))

/-- Transpose-reindex `Fin (N*(oc*(h*w))) → Fin (oc*(N*(h*w)))`: the inverse,
    network `(n, (c, s))` ↦ per-channel Mat `(c, (n, s))`. -/
noncomputable def bnchwBackIdx (N oc h w : Nat) (t : Fin (N * (oc * (h * w)))) :
    Fin (oc * (N * (h * w))) :=
  let nr := finProdFinEquiv.symm t           -- (Fin N, Fin (oc*(h*w)))
  let cs := finProdFinEquiv.symm nr.2        -- (Fin oc, Fin (h*w))
  finProdFinEquiv (cs.1, finProdFinEquiv (nr.1, cs.2))

theorem bnchwFwdIdx_bnchwBackIdx (N oc h w : Nat) (t : Fin (N * (oc * (h * w)))) :
    bnchwFwdIdx N oc h w (bnchwBackIdx N oc h w t) = t := by
  unfold bnchwFwdIdx bnchwBackIdx
  simp only [Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]

theorem bnchwBackIdx_bnchwFwdIdx (N oc h w : Nat) (mIdx : Fin (oc * (N * (h * w)))) :
    bnchwBackIdx N oc h w (bnchwFwdIdx N oc h w mIdx) = mIdx := by
  unfold bnchwFwdIdx bnchwBackIdx
  simp only [Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]

/-- **[N,C,H,W] → [C,N·H·W]** reindex (gather the network cell at the Mat position). -/
noncomputable def bnchwFwd (N oc h w : Nat) :
    Vec (N * (oc * (h * w))) → Vec (oc * (N * (h * w))) :=
  fun y k => y (bnchwFwdIdx N oc h w k)

theorem bnchwFwd_apply (N oc h w : Nat) (y : Vec (N * (oc * (h * w))))
    (k : Fin (oc * (N * (h * w)))) : bnchwFwd N oc h w y k = y (bnchwFwdIdx N oc h w k) := rfl

/-- **[C,N·H·W] → [N,C,H,W]** reindex (the inverse relabeling). -/
noncomputable def bnchwBack (N oc h w : Nat) :
    Vec (oc * (N * (h * w))) → Vec (N * (oc * (h * w))) :=
  fun y k => y (bnchwBackIdx N oc h w k)

theorem bnchwFwd_differentiable (N oc h w : Nat) :
    Differentiable ℝ (bnchwFwd N oc h w) :=
  (reindexCLM (bnchwFwdIdx N oc h w)).differentiable

theorem bnchwBack_differentiable (N oc h w : Nat) :
    Differentiable ℝ (bnchwBack N oc h w) :=
  (reindexCLM (bnchwBackIdx N oc h w)).differentiable

noncomputable def bnchwFwdHasVJP (N oc h w : Nat) :
    HasVJP (bnchwFwd N oc h w) :=
  reindexVJP (bnchwFwdIdx N oc h w)

noncomputable def bnchwBackHasVJP (N oc h w : Nat) :
    HasVJP (bnchwBack N oc h w) :=
  reindexVJP (bnchwBackIdx N oc h w)

theorem bnchwBackHasVJP_backward_eq (N oc h w : Nat) (v : Vec (oc * (N * (h * w))))
    (dy : Vec (N * (oc * (h * w)))) :
    (bnchwBackHasVJP N oc h w).backward v dy = bnchwFwd N oc h w dy :=
  reindexVJP_backward_of_inv _ _ (bnchwBackIdx_bnchwFwdIdx N oc h w)
    (bnchwFwdIdx_bnchwBackIdx N oc h w) v dy

theorem bnchwFwdHasVJP_backward_eq (N oc h w : Nat) (v : Vec (N * (oc * (h * w))))
    (dy : Vec (oc * (N * (h * w)))) :
    (bnchwFwdHasVJP N oc h w).backward v dy = bnchwBack N oc h w dy :=
  reindexVJP_backward_of_inv _ _ (bnchwFwdIdx_bnchwBackIdx N oc h w)
    (bnchwBackIdx_bnchwFwdIdx N oc h w) v dy

/-- **Batch-norm per channel on the network's `[N,C,H,W]` layout.** Conjugate the
    Mat-split `bnPerChannelFlat` (with `m = N·h·w`, the whole batch's cells per channel)
    by the transpose bridge: relabel `[N,C,H,W] → [C, N·H·W]`, normalize each channel
    over ALL its batch+spatial cells, relabel back. The EfficientNet normalization. -/
noncomputable def bnBatchTensor4 (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (oc * (h * w))) → Vec (N * (oc * (h * w))) :=
  bnchwBack N oc h w ∘ (bnPerChannelFlat oc (N * (h * w)) ε γ β) ∘ bnchwFwd N oc h w

/-- **Synchronised batch-norm on the `[N,C,H,W]` layout — statistics handed in.**

    `bnBatchTensor4`'s peer, conjugated by the same `[N,C,H,W] → [C, N·H·W]` bridge, but the
    per-channel normalisation reads `μ` and the second moment `m2` from its arguments instead
    of reducing `x` for them. Under data parallelism those arguments are the ALL-REDUCED
    global statistics — which is how one replica normalises over a batch it cannot see.

    It is stated at `μ` and the second moment `m2`, and forms the variance as `m2 − μ²`. The
    emitted sync forward exchanges `μ` and `σ²` (Chan's parallel variance, `bnVar_shard_chan`),
    and `SHlo.bnSyncF`'s `den` supplies `m2 := σ² + μ²`, so `m2 − μ²` is that `σ²` in ℝ. -/
noncomputable def bnSyncTensor4 (N oc h w : Nat) (ε : ℝ) (γ β μ m2 : Vec oc) :
    Vec (N * (oc * (h * w))) → Vec (N * (oc * (h * w))) :=
  bnchwBack N oc h w ∘
    (bnPerChannelEvalFlat oc (N * (h * w)) ε γ β μ (fun c => m2 c - μ c * μ c)) ∘
    bnchwFwd N oc h w

/-- **Frozen-stats per-channel BN is POINTWISE**, with the channel read off the index: the only
    thing the Mat round-trip does is decide *which* channel's `γ β μ v` a cell gets. -/
theorem bnPerChannelEvalFlat_apply (oc m : Nat) (ε : ℝ) (γ β μ v : Vec oc)
    (z : Vec (oc * m)) (idx : Fin (oc * m)) :
    bnPerChannelEvalFlat oc m ε γ β μ v z idx
      = γ (finProdFinEquiv.symm idx).1
          * ((z idx - μ (finProdFinEquiv.symm idx).1)
             * (1 / Real.sqrt (v (finProdFinEquiv.symm idx).1 + ε)))
        + β (finProdFinEquiv.symm idx).1 := by
  unfold bnPerChannelEvalFlat bnPerChannelEvalMat bnEvalForward Mat.flatten Mat.unflatten
  simp only [Prod.mk.eta, Equiv.apply_symm_apply]

/-- **`R = 1`: the sync forward at the batch's own statistics is `bnBatchTensor4`.**

    At `R = 1` every `allReduceMeanF` threads its operand, so the sync graph hands in exactly
    the statistics the batch would have computed, and this says that graph denotes
    `bnBatchTensor4`, the per-batch forward. -/
theorem bnSyncTensor4_at_own_stats (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (ε : ℝ) (γ β : Vec oc) (x : Vec (N * (oc * (h * w)))) :
    bnSyncTensor4 N oc h w ε γ β
        (fun c => bnMean   (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
        (fun c => bnMeanSq (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c)) x
      = bnBatchTensor4 N oc h w ε γ β x := by
  unfold bnSyncTensor4 bnBatchTensor4 bnPerChannelEvalFlat bnPerChannelFlat
  simp only [Function.comp_apply]
  congr 2
  funext c
  exact bnEvalForward_at_own_stats _ hm _ _ _ _

theorem bnBatchTensor4_differentiable (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (bnBatchTensor4 N oc h w ε γ β) := by
  unfold bnBatchTensor4
  exact (bnchwBack_differentiable N oc h w).comp
    ((bnPerChannelFlat_differentiable oc (N * (h * w)) ε hε γ β).comp
      (bnchwFwd_differentiable N oc h w))

/-- **Batch-norm (network layout) VJP** — block-diagonal across channels (now coupling
    the whole batch within each channel), lifted through the transpose bridge. -/
noncomputable def bnBatchTensor4HasVJP (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (bnBatchTensor4 N oc h w ε γ β) :=
  let inner : Vec (N * (oc * (h * w))) → Vec (oc * (N * (h * w))) :=
    bnPerChannelFlat oc (N * (h * w)) ε γ β ∘ bnchwFwd N oc h w
  let inner_diff : Differentiable ℝ inner :=
    (bnPerChannelFlat_differentiable oc (N * (h * w)) ε hε γ β).comp (bnchwFwd_differentiable N oc h w)
  let inner_vjp : HasVJP inner :=
    vjpComp (bnchwFwd N oc h w) (bnPerChannelFlat oc (N * (h * w)) ε γ β)
      (bnchwFwd_differentiable N oc h w) (bnPerChannelFlat_differentiable oc (N * (h * w)) ε hε γ β)
      (bnchwFwdHasVJP N oc h w) (bnPerChannelFlatHasVJP oc (N * (h * w)) ε hε γ β)
  show HasVJP (bnchwBack N oc h w ∘ inner) from
  vjpComp inner (bnchwBack N oc h w) inner_diff (bnchwBack_differentiable N oc h w)
    inner_vjp (bnchwBackHasVJP N oc h w)

theorem bnBatchTensor4HasVJP_correct (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) (i : Fin (N * (oc * (h * w)))) :
    (bnBatchTensor4HasVJP N oc h w ε hε γ β).backward x dy i =
      ∑ j : Fin (N * (oc * (h * w))), pdiv (bnBatchTensor4 N oc h w ε γ β) x i j * dy j :=
  (bnBatchTensor4HasVJP N oc h w ε hε γ β).correct x dy i

theorem bnBatchTensor4HasVJP_backward_eq (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) :
    (bnBatchTensor4HasVJP N oc h w ε hε γ β).backward x dy =
      bnchwBack N oc h w
        ((bnPerChannelFlatHasVJP oc (N * (h * w)) ε hε γ β).backward
          (bnchwFwd N oc h w x) (bnchwFwd N oc h w dy)) := by
  unfold bnBatchTensor4HasVJP
  rw [vjpComp_backward, vjpComp_backward, bnchwBackHasVJP_backward_eq,
    bnchwFwdHasVJP_backward_eq]

/-- **Renderable batch-norm backward on the `[N,C,H,W]` layout** — relabel to the
    per-channel Mat, run the consolidated three-term `bnPerChannelGradInput` over the
    whole batch (`m = N·h·w`), relabel back. Exactly what the batched batch-norm backward
    StableHLO fragment emits (reduce over `[0,2,3]` per channel). -/
noncomputable def bnBatchTensor4GradInput (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) : Vec (N * (oc * (h * w))) :=
  bnchwBack N oc h w
    (bnPerChannelGradInput oc (N * (h * w)) ε γ (bnchwFwd N oc h w x) (bnchwFwd N oc h w dy))

/-- **Per-channel SYNC backward (flat layout)** — `bnPerChannelGradInput`'s peer, with each
    channel's `μ`, `E[x²]` and two reduction means supplied rather than reduced out of `x`/`dy`.
    Under data parallelism those are the all-reduced global ones. -/
noncomputable def bnSyncPerChannelGradInput (oc m : Nat) (ε : ℝ) (γ μ m2 mdy mdyx : Vec oc)
    (x dy : Vec (oc * m)) : Vec (oc * m) :=
  fun idx =>
    bnSyncGradInput m ε (γ (finProdFinEquiv.symm idx).1) (μ (finProdFinEquiv.symm idx).1)
      (m2 (finProdFinEquiv.symm idx).1) (mdy (finProdFinEquiv.symm idx).1)
      (mdyx (finProdFinEquiv.symm idx).1)
      (Mat.unflatten x (finProdFinEquiv.symm idx).1)
      (Mat.unflatten dy (finProdFinEquiv.symm idx).1)
      (finProdFinEquiv.symm idx).2

/-- **The per-channel γ gradient with `x̂` at handed-in statistics** —
    `bnPerChannelGradGamma`'s peer. Under sync-BN the forward normalised with the all-reduced
    global statistics, so `∂L/∂γ_c = Σ dy·x̂` must use the SAME `x̂`; `bnPerChannelGradGamma`
    rebuilds it from the shard's own statistics (`bnXhat`), which is a different function once
    `R > 1`. β's gradient reads no statistic and needs no peer. -/
noncomputable def bnSyncPerChannelGradGamma (oc m : Nat) (ε : ℝ) (μ m2 : Vec oc)
    (v dy : Vec (oc * m)) : Vec oc :=
  fun c => ∑ s : Fin m, Mat.unflatten dy c s * bnSyncXhat m ε (μ c) (m2 c) (Mat.unflatten v c) s

/-- **`R = 1`: the sync γ gradient at its own statistics IS `bnPerChannelGradGamma`.** The
    γ-gradient anchor beside `bnSyncTensor4_at_own_stats`: a single-device sync render's γ node
    denotes what today's `bnGammaGradB` denotes. -/
theorem bnSyncPerChannelGradGamma_at_own_stats (oc m : Nat) (hm : m ≠ 0) (ε : ℝ)
    (v dy : Vec (oc * m)) :
    bnSyncPerChannelGradGamma oc m ε
        (fun c => bnMean m (Mat.unflatten v c)) (fun c => bnMeanSq m (Mat.unflatten v c)) v dy
      = bnPerChannelGradGamma oc m ε v dy := by
  funext c
  simp only [bnSyncPerChannelGradGamma, bnPerChannelGradGamma, bnSyncXhat_at_own_stats m hm,
             Mat.unflatten]

/-- **The sync batch-norm input-VJP on `[N,C,H,W]`** — `bnBatchTensor4GradInput`'s peer,
    through the same `bnchwFwd`/`bnchwBack` bridge. What a replica emits for its shard of the
    backward, given the four all-reduced per-channel statistic vectors. -/
noncomputable def bnSyncTensor4GradInput (N oc h w : Nat) (ε : ℝ) (γ μ m2 mdy mdyx : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) : Vec (N * (oc * (h * w))) :=
  bnchwBack N oc h w
    (bnSyncPerChannelGradInput oc (N * (h * w)) ε γ μ m2 mdy mdyx
      (bnchwFwd N oc h w x) (bnchwFwd N oc h w dy))

/-- **`R = 1`: the sync backward at its own statistics is `bnBatchTensor4GradInput`.**

    The `[N,C,H,W]` lift of `bnSyncGradInput_at_own_stats`, and the backward half of the
    drop-in claim: a single-device sync render computes the same gradient as the per-batch
    `bnBatchTensor4GradInput`. -/
theorem bnSyncTensor4GradInput_at_own_stats (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (ε : ℝ) (γ : Vec oc) (x dy : Vec (N * (oc * (h * w)))) :
    bnSyncTensor4GradInput N oc h w ε γ
        (fun c => bnMean   (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
        (fun c => bnMeanSq (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
        (fun c => bnMean (N*(h*w))
          (fun k => γ c * Mat.unflatten (bnchwFwd N oc h w dy) c k))
        (fun c => bnMean (N*(h*w))
          (fun k => bnXhat (N*(h*w)) ε (Mat.unflatten (bnchwFwd N oc h w x) c) k
                      * (γ c * Mat.unflatten (bnchwFwd N oc h w dy) c k)))
        x dy
      = bnBatchTensor4GradInput N oc h w ε γ x dy := by
  unfold bnSyncTensor4GradInput bnBatchTensor4GradInput
  congr 1
  funext idx
  exact congrFun (bnSyncGradInput_at_own_stats _ hm _ _ _ _) _

/-- **Renderable batch-norm backward is faithful** (ℝ-headline): equals the
    `pdiv`-contracted (block-diagonal-across-channels, batch-coupled) Jacobian of
    batch-norm on the network's `[N,C,H,W]` layout, under `0 < ε`. The licence to render
    EfficientNet's batch-norm backward as the per-channel three-term formula over the batch. -/
theorem bnBatchTensor4GradInput_correct (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) (i : Fin (N * (oc * (h * w)))) :
    bnBatchTensor4GradInput N oc h w ε γ x dy i =
      ∑ j : Fin (N * (oc * (h * w))), pdiv (bnBatchTensor4 N oc h w ε γ β) x i j * dy j := by
  rw [← bnBatchTensor4HasVJP_correct N oc h w ε hε γ β,
      bnBatchTensor4HasVJP_backward_eq N oc h w ε hε γ β]
  show bnPerChannelGradInput oc (N * (h * w)) ε γ (bnchwFwd N oc h w x) (bnchwFwd N oc h w dy)
        (bnchwBackIdx N oc h w i)
      = (bnPerChannelFlatHasVJP oc (N * (h * w)) ε hε γ β).backward
          (bnchwFwd N oc h w x) (bnchwFwd N oc h w dy) (bnchwBackIdx N oc h w i)
  rw [bnPerChannelGradInput_correct oc (N * (h * w)) ε hε γ β,
      bnPerChannelFlatHasVJP_correct oc (N * (h * w)) ε hε γ β]

end Proofs
