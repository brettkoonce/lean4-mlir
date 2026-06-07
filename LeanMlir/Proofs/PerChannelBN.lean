import LeanMlir.Proofs.BatchNorm

/-! # Per-channel BatchNorm (Chapter 6 Milestone B8) — the block-diagonal VJP

Chapters 5–6 used a per-example **global** BatchNorm: one scalar `(γ, β)` over the
whole `oc·h·w` activation (LayerNorm-shaped). Real ResNet wants **per-channel** BN:
normalize each channel-slice independently with its *own* `(γ_c, β_c)`, `γ/β : Vec oc`.

Because each channel is independent, the whole Jacobian is **block-diagonal** across
the channel axis — the genuinely-new piece. We get it for free by generalizing the
existing `rowwise_has_vjp_mat` (Tensor.lean, multi-head attention) from a *single*
per-row map to a **per-row family** `g : Fin m → (Vec n → Vec p)`: viewing the
activation as `Mat oc (h·w)` (row = channel), per-channel BN is exactly
`fun A => fun c => bnForward (h·w) ε (γ c) (β c) (A c)`. Its VJP runs each channel's
`bn_has_vjp` on that channel's cotangent slice; the cross-channel blocks vanish.

Everything closes under `[propext, Classical.choice, Quot.sound]`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Per-row independence (a per-row family generalizing `rowwise`)
-- ════════════════════════════════════════════════════════════════

/-- **Block-diagonal Jacobian of a per-row family.** Applying a *different* map `g r`
    to each row `r` of a matrix keeps the matrix Jacobian block-diagonal across the
    row axis: output row `k` depends only on input row `k` (via `g k`). The per-row
    generalization of `pdivMat_rowIndep` (which fixes one `g` for all rows). -/
theorem pdivMat_rowIndep_perRow {m n p : Nat} (g : Fin m → (Vec n → Vec p))
    (h_g_diff : ∀ r, Differentiable ℝ (g r))
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g r (M r)) A i j k l =
    if i = k then pdiv (g k) (A k) j l else 0 := by
  unfold pdivMat pdiv
  set F : Vec (m * n) → Vec (m * p) :=
    fun v => Mat.flatten ((fun M : Mat m n => fun r => g r (M r)) (Mat.unflatten v))
    with hF
  set rowProj : Fin m → (Vec (m * n) →L[ℝ] Vec n) := fun k' =>
    reindexCLM (fun j' : Fin n => finProdFinEquiv (k', j'))
  have h_coord : ∀ (k' : Fin m) (l' : Fin p),
      (fun v : Vec (m * n) => F v (finProdFinEquiv (k', l'))) =
      (fun w : Vec n => g k' w l') ∘ (rowProj k') := by
    intro k' l'
    funext v
    show Mat.flatten ((fun M : Mat m n => fun r => g r (M r)) (Mat.unflatten v))
        (finProdFinEquiv (k', l')) = g k' ((rowProj k') v) l'
    unfold Mat.flatten
    simp only [Equiv.symm_apply_apply]
    show g k' (Mat.unflatten v k') l' = g k' ((rowProj k') v) l'
    rfl
  have h_g_l : ∀ (r : Fin m) (l' : Fin p) (w : Vec n),
      DifferentiableAt ℝ (fun w => g r w l') w :=
    fun r l' w => differentiableAt_pi.mp (h_g_diff r w) l'
  have h_coord_diff : ∀ (k' : Fin m) (l' : Fin p) (v : Vec (m * n)),
      DifferentiableAt ℝ (fun v' : Vec (m * n) => F v' (finProdFinEquiv (k', l'))) v := by
    intro k' l' v
    rw [h_coord k' l']
    exact (h_g_l k' l' _).comp v (rowProj k').differentiableAt
  have h_F_diff : DifferentiableAt ℝ F (Mat.flatten A) := by
    rw [(differentiableAt_pi : DifferentiableAt ℝ F (Mat.flatten A) ↔ _)]
    intro idx
    have h_idx : finProdFinEquiv (finProdFinEquiv.symm idx) = idx :=
      Equiv.apply_symm_apply _ _
    have h_idx' : idx = finProdFinEquiv
        ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2) := by
      conv_lhs => rw [← h_idx]
    rw [h_idx']
    exact h_coord_diff _ _ (Mat.flatten A)
  have h_swap :
      fderiv ℝ F (Mat.flatten A) (basisVec (finProdFinEquiv (i, j))) (finProdFinEquiv (k, l)) =
      fderiv ℝ (fun v : Vec (m * n) => F v (finProdFinEquiv (k, l))) (Mat.flatten A)
        (basisVec (finProdFinEquiv (i, j))) := by
    rw [fderiv_apply h_F_diff (finProdFinEquiv (k, l))]
    rfl
  rw [h_swap]
  rw [h_coord k l]
  rw [fderiv_comp _ (h_g_l k l _) (rowProj k).differentiableAt]
  rw [(rowProj k).fderiv]
  have h_row_A : (rowProj k) (Mat.flatten A) = A k := by
    funext j'
    show Mat.flatten A (finProdFinEquiv (k, j')) = A k j'
    show A (finProdFinEquiv.symm (finProdFinEquiv (k, j'))).1
            (finProdFinEquiv.symm (finProdFinEquiv (k, j'))).2 = A k j'
    simp
  rw [h_row_A]
  rw [fderiv_apply (h_g_diff k _) l]
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply]
  by_cases hik : i = k
  · subst hik
    rw [if_pos rfl]
    have h_basis : (rowProj i) (basisVec (finProdFinEquiv (i, j))) = basisVec j := by
      funext j'
      show basisVec (finProdFinEquiv (i, j)) (finProdFinEquiv (i, j')) = basisVec j j'
      simp only [basisVec_apply]
      by_cases hjj : j' = j
      · subst hjj; simp
      · rw [if_neg hjj, if_neg ?_]
        intro heq
        apply hjj
        exact (Prod.mk.inj (finProdFinEquiv.injective heq.symm)).2.symm
    rw [h_basis]
  · rw [if_neg hik]
    have h_basis : (rowProj k) (basisVec (finProdFinEquiv (i, j))) = (0 : Vec n) := by
      funext j'
      show basisVec (finProdFinEquiv (i, j)) (finProdFinEquiv (k, j')) = (0 : ℝ)
      simp only [basisVec_apply]
      rw [if_neg]
      intro heq
      apply hik
      exact (Prod.mk.inj (finProdFinEquiv.injective heq)).1.symm
    rw [h_basis]
    simp

/-- **Row-wise lifting of a per-row `HasVJP` family.** Each row `r` gets its own map
    `g r` (with its own VJP); the matrix backward runs `(g r).backward` on row `r`'s
    cotangent. The per-row peer of `rowwise_has_vjp_mat`. -/
noncomputable def rowwisePerRow_has_vjp_mat {m n p : Nat} (g : Fin m → (Vec n → Vec p))
    (hg : ∀ r, HasVJP (g r)) (hg_diff : ∀ r, Differentiable ℝ (g r)) :
    HasVJPMat (fun A : Mat m n => fun r => g r (A r)) where
  backward := fun A dY => fun r c => (hg r).backward (A r) (dY r) c
  correct := by
    intro A dY i j
    simp_rw [pdivMat_rowIndep_perRow g hg_diff]
    have h : ∀ k : Fin m,
        (∑ l : Fin p, (if i = k then pdiv (g k) (A k) j l else 0) * dY k l) =
        if i = k then ∑ l : Fin p, pdiv (g k) (A k) j l * dY k l else 0 := by
      intro k
      by_cases hik : i = k
      · simp [hik]
      · simp [hik]
    simp_rw [h]
    rw [Finset.sum_ite_eq Finset.univ i
        (fun k => ∑ l : Fin p, pdiv (g k) (A k) j l * dY k l)]
    simp only [Finset.mem_univ, if_true]
    exact (hg i).correct (A i) (dY i) j

/-- **A per-row family flattens to a differentiable `Vec → Vec` map.** The
    `Differentiable` witness `vjp_comp_at` / the network composition needs to thread a
    per-channel BN through a block. -/
theorem rowwisePerRow_flat_differentiable {m n p : Nat} (g : Fin m → (Vec n → Vec p))
    (h_g_diff : ∀ r, Differentiable ℝ (g r)) :
    Differentiable ℝ (fun v : Vec (m * n) =>
      Mat.flatten ((fun A : Mat m n => fun r => g r (A r)) (Mat.unflatten v))) := by
  set F : Vec (m * n) → Vec (m * p) :=
    fun v => Mat.flatten ((fun A : Mat m n => fun r => g r (A r)) (Mat.unflatten v)) with hF
  set rowProj : Fin m → (Vec (m * n) →L[ℝ] Vec n) := fun k' =>
    reindexCLM (fun j' : Fin n => finProdFinEquiv (k', j'))
  have h_coord : ∀ (k' : Fin m) (l' : Fin p),
      (fun v : Vec (m * n) => F v (finProdFinEquiv (k', l'))) =
      (fun w : Vec n => g k' w l') ∘ (rowProj k') := by
    intro k' l'
    funext v
    show Mat.flatten ((fun A : Mat m n => fun r => g r (A r)) (Mat.unflatten v))
        (finProdFinEquiv (k', l')) = g k' ((rowProj k') v) l'
    unfold Mat.flatten
    simp only [Equiv.symm_apply_apply]
    show g k' (Mat.unflatten v k') l' = g k' ((rowProj k') v) l'
    rfl
  have h_g_l : ∀ (r : Fin m) (l' : Fin p) (w : Vec n),
      DifferentiableAt ℝ (fun w => g r w l') w :=
    fun r l' w => differentiableAt_pi.mp (h_g_diff r w) l'
  intro v
  rw [(differentiableAt_pi : DifferentiableAt ℝ F v ↔ _)]
  intro idx
  have h_idx' : idx = finProdFinEquiv
      ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2) := by
    conv_lhs => rw [← Equiv.apply_symm_apply finProdFinEquiv idx]
  rw [h_idx', h_coord _ _]
  exact (h_g_l _ _ _).comp v (rowProj _).differentiableAt

-- ════════════════════════════════════════════════════════════════
-- § Per-channel BatchNorm
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel BatchNorm** (matrix view): BN each row (= channel-slice of `m = h·w`
    spatial cells) with its own `(γ_c, β_c)`. The real-ResNet BN that `bnForward`'s
    global scalar version approximates. -/
noncomputable def bnPerChannelMat (oc m : Nat) (ε : ℝ) (γ β : Vec oc) :
    Mat oc m → Mat oc m :=
  fun A => fun c => bnForward m ε (γ c) (β c) (A c)

/-- **Per-channel BN VJP (block-diagonal).** Each channel runs its own `bn_has_vjp`;
    the cross-channel Jacobian blocks vanish (`pdivMat_rowIndep_perRow`). -/
noncomputable def bnPerChannelMat_has_vjp (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJPMat (bnPerChannelMat oc m ε γ β) :=
  rowwisePerRow_has_vjp_mat (fun c => bnForward m ε (γ c) (β c))
    (fun c => bn_has_vjp m ε (γ c) (β c) hε)
    (fun c => bnForward_differentiable m ε (γ c) (β c) hε)

/-- Per-channel BN as a flat-vector op `Vec (oc·m) → Vec (oc·m)` (row-major, channel
    `c` = the `m`-wide slab at flat positions `finProdFinEquiv (c, ·)`). -/
noncomputable def bnPerChannelFlat (oc m : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (oc * m) → Vec (oc * m) :=
  fun v => Mat.flatten (bnPerChannelMat oc m ε γ β (Mat.unflatten v))

/-- **Per-channel BN flat VJP** — the block-diagonal matrix VJP bridged to `Vec`. -/
noncomputable def bnPerChannelFlat_has_vjp (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (bnPerChannelFlat oc m ε γ β) :=
  hasVJPMat_to_hasVJP (bnPerChannelMat_has_vjp oc m ε hε γ β)

/-- **Per-channel BN is differentiable everywhere** (`ε > 0`). The composition witness. -/
theorem bnPerChannelFlat_differentiable (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (bnPerChannelFlat oc m ε γ β) :=
  rowwisePerRow_flat_differentiable (fun c => bnForward m ε (γ c) (β c))
    (fun c => bnForward_differentiable m ε (γ c) (β c) hε)

/-- **Per-channel BN VJP correctness** (ℝ-headline): the flat backward equals the
    `pdiv`-contracted (block-diagonal) Jacobian of per-channel BN. -/
theorem bnPerChannelFlat_has_vjp_correct (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v dy : Vec (oc * m)) (i : Fin (oc * m)) :
    (bnPerChannelFlat_has_vjp oc m ε hε γ β).backward v dy i =
      ∑ j : Fin (oc * m), pdiv (bnPerChannelFlat oc m ε γ β) v i j * dy j :=
  (bnPerChannelFlat_has_vjp oc m ε hε γ β).correct v dy i

-- ════════════════════════════════════════════════════════════════
-- § Renderable closed-form backward (the per-channel consolidated gradient)
-- ════════════════════════════════════════════════════════════════

/-- **Per-channel consolidated BN input-gradient** — the renderable closed form: run
    the per-example three-term `bn_grad_input` on each channel-slice (`m = h·w` spatial
    cells), reusing that channel's `γ_c`. This is exactly what a `bnPerChannelBack` SHlo
    op / `renderLNBack`-per-channel emits; the abstract `bnPerChannelFlat_has_vjp.backward`
    is the spec it must match. -/
noncomputable def bnPerChannel_grad_input (oc m : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (oc * m)) : Vec (oc * m) :=
  fun idx =>
    bn_grad_input m ε (γ (finProdFinEquiv.symm idx).1)
      (Mat.unflatten x (finProdFinEquiv.symm idx).1)
      (Mat.unflatten dy (finProdFinEquiv.symm idx).1)
      (finProdFinEquiv.symm idx).2

/-- **Renderable backward is faithful** (ℝ-headline): the per-channel consolidated
    gradient equals the `pdiv`-contracted Jacobian of per-channel BN, under `0 < ε`.
    Each channel reduces to the per-example `bn_input_grad_correct`. The licence to
    render per-channel BN's backward as the three-term formula per channel. -/
theorem bnPerChannel_grad_input_correct (oc m : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (oc * m)) (i : Fin (oc * m)) :
    bnPerChannel_grad_input oc m ε γ x dy i =
      ∑ j : Fin (oc * m), pdiv (bnPerChannelFlat oc m ε γ β) x i j * dy j := by
  rw [← bnPerChannelFlat_has_vjp_correct oc m ε hε γ β]
  show bn_grad_input m ε (γ (finProdFinEquiv.symm i).1)
        (Mat.unflatten x (finProdFinEquiv.symm i).1) (Mat.unflatten dy (finProdFinEquiv.symm i).1)
        (finProdFinEquiv.symm i).2
      = (bn_has_vjp m ε (γ (finProdFinEquiv.symm i).1) (β (finProdFinEquiv.symm i).1) hε).backward
          (Mat.unflatten x (finProdFinEquiv.symm i).1) (Mat.unflatten dy (finProdFinEquiv.symm i).1)
          (finProdFinEquiv.symm i).2
  rw [(bn_has_vjp m ε (γ (finProdFinEquiv.symm i).1) (β (finProdFinEquiv.symm i).1) hε).correct,
      ← bn_input_grad_correct m ε (γ (finProdFinEquiv.symm i).1) (β (finProdFinEquiv.symm i).1) hε]

end Proofs
