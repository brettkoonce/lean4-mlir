import LeanMlir.Proofs.ViTFwdGraph

/-! # Closing the ViT render — the parameter-gradient close (ch10 Item C)

`planning/vit_close.md` Item C, applied to the representative 2-block / 1-head ViT
(`vitForward2`, the proven whole-net VJP `vitForward2_has_vjp`; graph `vitFwdGraph`).
Generic in the cotangent `dy` the backward chain delivers at each site's output
(pinning that cotangent to the actual attention chain is the optional Item D), batch-1 —
everything in a ViT is per-example separable (the EfficientNet contrast).

| family (render SSA)                  | forward fn                 | certified by |
|--------------------------------------|----------------------------|--------------|
| Wq/Wk/Wv/Wo, Wfc1/Wfc2 + biases      | per-token dense (rowwise)  | `vit_render_rowdense{W,b}_certified` (**new family**): `dW = Σ_tokens xᵣ ⊗ dyᵣ`, `db = Σ_tokens dyᵣ` — the M2 outer-product bridge row-lifted |
| classifier `Wcls`/`bcls`             | dense on the CLS row       | M2 `weight/bias_grad_bridge` (**reuse** — single-vector dense) |
| LN γ/β ×5 sites (scalar, per-token)  | rowwise `layerNormForward` | `vit_render_rowln{gamma,beta}_certified` (**new**): the ConvNeXtClose `Vec 1` embedding row-lifted — `dγ = Σ_{tokens} Σ_D dy·x̂`, `dβ = Σ Σ dy` (affine in the params ⇒ no `0 < ε`) |
| `pos_embed`                          | additive (`patchEmbed_flat`) | `vit_render_pos_certified`: the pos-Jacobian is the identity ⇒ `dPos = dy` |
| `cls_token`                          | row-0 scatter (`patchEmbed_flat`) | `vit_render_cls_certified`: masked-gather Jacobian ⇒ `dCls = dy` row-0 slice |
| attention internals (softmax, scale) | —                          | no parameters |

Two genuinely-new bridge families (everything else is reuse or a reindex):

* **Per-token dense W/b** — every row of `[N,a]` through the same `W : [a,c]` (+ `b`).
  The W-Jacobian of the flattened rowwise dense is block-sparse —
  `∂y_(r,k)/∂W_(i,j) = X_(r,i)·δ_(k,j)` — so the rendered per-token outer-product reduce
  `dW_(i,j) = Σ_r X_(r,i)·dY_(r,j)` (one `dot_general` contracting the token axis) is the
  certified contraction. Covers Wq/Wk/Wv/Wo, Wfc1/Wfc2 and their biases at every block.
* **Row-lifted scalar-LN γ/β** — the ConvNeXtClose `Vec 1` embedding generalized from one
  LN site over `Vec n` to N token rows: as a function of `γ' : Vec 1` the rowwise LN is
  affine, `γ' ↦ fun (r,k) => x̂_r(k)·γ'(0) + β`, so the Jacobian is the flattened per-row
  x̂ and the rendered whole-tensor reduce `dγ = Σ_r Σ_k dY_(r,k)·x̂_r(k) = Σ_r bn_grad_gamma (X r) (dY r)`
  is certified. Likewise `dβ = Σ_r Σ_k dY_(r,k)`.

The classifier head (`dense` on the CLS vector) is VERBATIM M2 `weight/bias_grad_bridge`
reuse at `[D, nClasses]`. The patch-embed conv `Wp`/`bp` close over `patchEmbed_flat`
is in § E. 3-axiom clean by construction.
-/

namespace Proofs

open scoped BigOperators

/-- Sum over a flat product index = double sum over the factors (row-major). -/
private lemma sum_fin_prod {M : Type*} [AddCommMonoid M] (m n : Nat)
    (f : Fin (m * n) → M) :
    (∑ idx : Fin (m * n), f idx)
      = ∑ r : Fin m, ∑ k : Fin n, f (finProdFinEquiv (r, k)) := by
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)) f,
      Fintype.sum_prod_type]

-- ════════════════════════════════════════════════════════════════
-- § A. Per-token dense W/b — the row-lifted M2 family (genuinely new)
--
-- `fun v => Mat.flatten (fun r => dense (Mat.unflatten v) b (X r))` is, in `v`, a sum of
-- constant×reindex summands per output coordinate (the CifarBnClose recipe at the flat
-- `[N·c]` output index): `y_(r,k) = Σ_i X_(r,i)·v_(i,k) + b_k`.
-- ════════════════════════════════════════════════════════════════

/-- **Jacobian of the per-token dense w.r.t. the (flattened) shared weight** —
    `∂y_(r,k)/∂W_(i,j') = X_(r,i)·δ_(k,j')`. The row-lift of `pdiv_dense_W`. -/
theorem pdiv_rowDense_W {N a c : Nat} (bb : Vec c) (X : Mat N a) (W : Mat a c)
    (i : Fin a) (j' : Fin c) (idx : Fin (N * c)) :
    pdiv (fun v : Vec (a * c) =>
            Mat.flatten (fun r => dense (Mat.unflatten v) bb (X r)))
         (Mat.flatten W) (finProdFinEquiv (i, j')) idx =
      if j' = (finProdFinEquiv.symm idx).2
        then X (finProdFinEquiv.symm idx).1 i else 0 := by
  -- Step 1: explicit Vec (a*c) → Vec (N*c) normal form.
  rw [show (fun v : Vec (a * c) =>
              Mat.flatten (fun r => dense (Mat.unflatten v) bb (X r))) =
        (fun v : Vec (a * c) => fun o : Fin (N * c) =>
          (∑ i' : Fin a, X (finProdFinEquiv.symm o).1 i' *
              v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) +
            bb (finProdFinEquiv.symm o).2) from by
      funext v o; unfold dense Mat.unflatten Mat.flatten; rfl]
  -- Step 2: split (sum) + (constant bias): pdiv_add + pdiv_const.
  rw [show (fun v : Vec (a * c) => fun o : Fin (N * c) =>
              (∑ i' : Fin a, X (finProdFinEquiv.symm o).1 i' *
                  v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) +
                bb (finProdFinEquiv.symm o).2) =
        (fun v o =>
          (fun w : Vec (a * c) => fun o' : Fin (N * c) =>
              ∑ i' : Fin a, X (finProdFinEquiv.symm o').1 i' *
                w (finProdFinEquiv (i', (finProdFinEquiv.symm o').2))) v o +
          (fun _ : Vec (a * c) =>
              fun o' : Fin (N * c) => bb (finProdFinEquiv.symm o').2) v o) from rfl]
  have h_summand_diff : ∀ i' ∈ (Finset.univ : Finset (Fin a)),
      DifferentiableAt ℝ
        (fun (v : Vec (a * c)) (o : Fin (N * c)) =>
          X (finProdFinEquiv.symm o).1 i' *
            v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) (Mat.flatten W) := by
    intro i' _
    have h_const : DifferentiableAt ℝ
        (fun (_ : Vec (a * c)) (o : Fin (N * c)) =>
          X (finProdFinEquiv.symm o).1 i') (Mat.flatten W) :=
      differentiableAt_const _
    have h_reindex : DifferentiableAt ℝ
        (fun (w : Vec (a * c)) (o : Fin (N * c)) =>
          w (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) (Mat.flatten W) :=
      (reindexCLM (fun o : Fin (N * c) =>
        finProdFinEquiv (i', (finProdFinEquiv.symm o).2))).differentiableAt
    exact h_const.mul h_reindex
  have h_sum_diff : DifferentiableAt ℝ
      (fun (w : Vec (a * c)) (o : Fin (N * c)) =>
        ∑ i' : Fin a, X (finProdFinEquiv.symm o).1 i' *
          w (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) (Mat.flatten W) := by
    have h_eq : (fun (w : Vec (a * c)) (o : Fin (N * c)) =>
                  ∑ i' : Fin a, X (finProdFinEquiv.symm o).1 i' *
                    w (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) =
                (fun w : Vec (a * c) => ∑ i' : Fin a,
                  fun o : Fin (N * c) => X (finProdFinEquiv.symm o).1 i' *
                    w (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) := by
      funext w o; rw [Finset.sum_apply]
    rw [h_eq]
    exact DifferentiableAt.fun_sum (fun i' _ => h_summand_diff i' (Finset.mem_univ i'))
  have h_const_diff : DifferentiableAt ℝ
      (fun _ : Vec (a * c) =>
        fun o' : Fin (N * c) => bb (finProdFinEquiv.symm o').2) (Mat.flatten W) :=
    differentiableAt_const _
  rw [pdiv_add _ _ _ h_sum_diff h_const_diff, pdiv_const, add_zero]
  -- Step 3: pdiv through the Fin a sum.
  rw [pdiv_finset_sum (Finset.univ : Finset (Fin a))
      (fun i' v o => X (finProdFinEquiv.symm o).1 i' *
        v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2)))
      (Mat.flatten W) h_summand_diff (finProdFinEquiv (i, j')) idx]
  -- Step 4: each summand is (const) × (reindex); pdiv_mul + pdiv_const + pdiv_reindex.
  have hterm : ∀ i' : Fin a,
      pdiv (fun v : Vec (a * c) => fun o : Fin (N * c) =>
              X (finProdFinEquiv.symm o).1 i' *
                v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2)))
           (Mat.flatten W) (finProdFinEquiv (i, j')) idx =
      if i = i' ∧ j' = (finProdFinEquiv.symm idx).2
        then X (finProdFinEquiv.symm idx).1 i else 0 := by
    intro i'
    rw [show (fun v : Vec (a * c) => fun o : Fin (N * c) =>
                X (finProdFinEquiv.symm o).1 i' *
                  v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) =
          (fun v o =>
            (fun (_ : Vec (a * c)) (o' : Fin (N * c)) =>
              X (finProdFinEquiv.symm o').1 i') v o *
            (fun (w : Vec (a * c)) (o' : Fin (N * c)) =>
              w (finProdFinEquiv (i', (finProdFinEquiv.symm o').2))) v o) from rfl]
    have h_const_inner : DifferentiableAt ℝ
        (fun (_ : Vec (a * c)) (o' : Fin (N * c)) =>
          X (finProdFinEquiv.symm o').1 i') (Mat.flatten W) :=
      differentiableAt_const _
    have h_reindex_inner : DifferentiableAt ℝ
        (fun (w : Vec (a * c)) (o' : Fin (N * c)) =>
          w (finProdFinEquiv (i', (finProdFinEquiv.symm o').2))) (Mat.flatten W) :=
      (reindexCLM (fun o' : Fin (N * c) =>
        finProdFinEquiv (i', (finProdFinEquiv.symm o').2))).differentiableAt
    rw [pdiv_mul _ _ _ h_const_inner h_reindex_inner,
        pdiv_const,
        pdiv_reindex (fun o' : Fin (N * c) =>
          finProdFinEquiv (i', (finProdFinEquiv.symm o').2))
          (Mat.flatten W) (finProdFinEquiv (i, j')) idx]
    rw [zero_mul, zero_add]
    by_cases hij : finProdFinEquiv (i, j') =
        finProdFinEquiv (i', (finProdFinEquiv.symm idx).2)
    · have hpair : (i, j') = (i', (finProdFinEquiv.symm idx).2) :=
        finProdFinEquiv.injective hij
      have hi : i = i' := congrArg Prod.fst hpair
      have hj : j' = (finProdFinEquiv.symm idx).2 := congrArg Prod.snd hpair
      rw [if_pos hij, if_pos ⟨hi, hj⟩, mul_one, ← hi]
    · rw [if_neg hij, mul_zero, if_neg]
      intro ⟨hi, hj⟩
      exact hij (by rw [hi, hj])
  simp_rw [hterm]
  -- Step 5: collapse the Fin a sum at i' = i.
  rw [Finset.sum_eq_single i
      (fun i' _ hne => by
        rw [if_neg]; intro ⟨hi, _⟩; exact hne hi.symm)
      (fun h => absurd (Finset.mem_univ i) h)]
  by_cases hj : j' = (finProdFinEquiv.symm idx).2
  · rw [if_pos ⟨rfl, hj⟩, if_pos hj]
  · rw [if_neg (fun hc => hj hc.2), if_neg hj]

/-- The rendered **per-token dense weight gradient**: the token-axis-contracted
    outer product `dW_(i,j) = Σ_r X_(r,i)·dY_(r,j)` (one `dot_general` contracting
    the token axis — the row-lift of `dense_weight_grad = x ⊗ dy`). -/
noncomputable def rowDense_weight_grad {N a c : Nat} (X : Mat N a) (dY : Mat N c) :
    Mat a c :=
  fun i j => ∑ r : Fin N, X r i * dY r j

/-- The rendered **per-token dense bias gradient**: the token-axis reduce
    `db_j = Σ_r dY_(r,j)`. -/
noncomputable def rowDense_bias_grad {N c : Nat} (dY : Mat N c) : Vec c :=
  fun j => ∑ r : Fin N, dY r j

/-- **Per-token dense W-gradient bridge.** The rendered token-contracted outer
    product equals the certified Jacobian of the rowwise dense (as a function of
    the flattened shared `W`) contracted with the cotangent. -/
theorem vit_rowDenseW_grad_bridge {N a c : Nat} (bb : Vec c) (X : Mat N a)
    (W : Mat a c) (dy : Vec (N * c)) (i : Fin a) (j : Fin c) :
    rowDense_weight_grad X (Mat.unflatten dy) i j
      = ∑ o : Fin (N * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bb (X r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o * dy o := by
  simp_rw [pdiv_rowDense_W]
  rw [sum_fin_prod N c]
  unfold rowDense_weight_grad Mat.unflatten
  apply Finset.sum_congr rfl
  intro r _
  rw [Finset.sum_eq_single j
      (fun k _ hne => by
        rw [Equiv.symm_apply_apply, if_neg (by simpa using (Ne.symm hne)), zero_mul])
      (fun h => absurd (Finset.mem_univ j) h)]
  rw [Equiv.symm_apply_apply, if_pos rfl]

/-- **Per-token dense b-gradient bridge.** The rendered token-axis reduce equals
    the certified rowwise-dense ∂/∂b contraction. -/
theorem vit_rowDenseb_grad_bridge {N a c : Nat} (W : Mat a c) (X : Mat N a)
    (bb : Vec c) (dy : Vec (N * c)) (i : Fin c) :
    rowDense_bias_grad (Mat.unflatten dy) i
      = ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r)))
               bb i o * dy o := by
  -- Jacobian: ∂y_(r,k)/∂b_i = δ_(k,i) — constant + gather.
  have hpdiv : ∀ o : Fin (N * c),
      pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) bb i o =
        if i = (finProdFinEquiv.symm o).2 then 1 else 0 := by
    intro o
    rw [show (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) =
          (fun b' : Vec c => fun o' : Fin (N * c) =>
            (fun (_ : Vec c) (o'' : Fin (N * c)) =>
              ∑ i' : Fin a, X (finProdFinEquiv.symm o'').1 i' *
                W i' (finProdFinEquiv.symm o'').2) b' o' +
            (fun (w : Vec c) (o'' : Fin (N * c)) =>
              w ((fun o''' : Fin (N * c) => (finProdFinEquiv.symm o''').2) o'')) b' o') from by
        funext b' o'; unfold dense Mat.flatten; rfl]
    have h_const : DifferentiableAt ℝ
        (fun (_ : Vec c) (o'' : Fin (N * c)) =>
          ∑ i' : Fin a, X (finProdFinEquiv.symm o'').1 i' *
            W i' (finProdFinEquiv.symm o'').2) bb := differentiableAt_const _
    have h_gather : DifferentiableAt ℝ
        (fun (w : Vec c) (o'' : Fin (N * c)) =>
          w ((fun o''' : Fin (N * c) => (finProdFinEquiv.symm o''').2) o'')) bb :=
      (reindexCLM (fun o''' : Fin (N * c) => (finProdFinEquiv.symm o''').2)).differentiableAt
    rw [pdiv_add _ _ _ h_const h_gather, pdiv_const, zero_add,
        pdiv_reindex (fun o''' : Fin (N * c) => (finProdFinEquiv.symm o''').2) bb i o]
  simp_rw [hpdiv]
  rw [sum_fin_prod N c]
  unfold rowDense_bias_grad Mat.unflatten
  apply Finset.sum_congr rfl
  intro r _
  rw [Finset.sum_eq_single i
      (fun k _ hne => by
        rw [Equiv.symm_apply_apply, if_neg (Ne.symm hne), zero_mul])
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [Equiv.symm_apply_apply, if_pos rfl, one_mul]

/-- **Per-token dense W output, certified.** `Wⁿ = W − lr·(Σ_tokens xᵣ ⊗ dyᵣ)` denotes
    `W − lr·(certified ∂(rowwise dense)/∂W · cotangent)`. Covers Wq/Wk/Wv/Wo and
    Wfc1/Wfc2 at every block of the representative ViT (each at its own `[a,c]`). -/
theorem vit_render_rowdenseW_certified {N a c : Nat} (bb : Vec c) (X : Mat N a)
    (W : Mat a c) (dy : Vec (N * c)) (lr : ℝ) (i : Fin a) (j : Fin c) :
    W i j - lr * rowDense_weight_grad X (Mat.unflatten dy) i j
      = W i j - lr * ∑ o : Fin (N * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bb (X r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o * dy o := by
  rw [vit_rowDenseW_grad_bridge]

/-- **Per-token dense b output, certified.** `bⁿ = b − lr·(Σ_tokens dyᵣ)` denotes the
    certified rowwise-dense ∂/∂b contraction. Covers all six per-block biases. -/
theorem vit_render_rowdenseb_certified {N a c : Nat} (W : Mat a c) (X : Mat N a)
    (bb : Vec c) (dy : Vec (N * c)) (lr : ℝ) (i : Fin c) :
    bb i - lr * rowDense_bias_grad (Mat.unflatten dy) i
      = bb i - lr * ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r)))
               bb i o * dy o := by
  rw [vit_rowDenseb_grad_bridge W X bb dy i]

-- ════════════════════════════════════════════════════════════════
-- § B. Row-lifted scalar-LN γ/β — the ConvNeXtClose Vec-1 embedding over N token rows
--
-- As a function of `γ' : Vec 1`, the rowwise LN is affine: `γ' ↦ fun (r,k) => x̂_r(k)·γ'(0) + β`
-- (per-row x̂, flattened). Same recipe as `pdiv_layerNorm_gamma`, abstracted over the
-- coefficient vector so one proof serves every site shape. Affine in the params ⇒ no `0 < ε`.
-- ════════════════════════════════════════════════════════════════

/-- Generic scalar-affine Jacobian: `∂(XH_k·γ(0) + C_k)/∂γ = XH_k`. The
    `pdiv_layerNorm_gamma` recipe with the coefficient vector abstracted. -/
private theorem pdiv_scalarAffine_gamma {m : Nat} (XH C : Vec m) (γ : Vec 1)
    (i : Fin 1) (j : Fin m) :
    pdiv (fun y : Vec 1 => fun k => XH k * y ((fun _ : Fin m => (0 : Fin 1)) k) + C k)
      γ i j = XH j := by
  have hmul_diff : DifferentiableAt ℝ
      (fun y : Vec 1 => fun k : Fin m => XH k * y ((fun _ : Fin m => (0 : Fin 1)) k)) γ :=
    (differentiableAt_const XH).mul
      ((reindexCLM (fun _ : Fin m => (0 : Fin 1))).differentiableAt)
  have hconst_diff : DifferentiableAt ℝ (fun _ : Vec 1 => C) γ := differentiableAt_const _
  rw [show (fun (y : Vec 1) (k : Fin m) => XH k * y ((fun _ : Fin m => (0 : Fin 1)) k) + C k)
        = (fun y k => (fun y' k' => XH k' * y' ((fun _ : Fin m => (0 : Fin 1)) k')) y k
                      + (fun _ k' => C k') y k) from rfl,
      pdiv_add _ _ _ hmul_diff hconst_diff,
      pdiv_const C γ i j, add_zero]
  have hconstXH_diff : DifferentiableAt ℝ (fun _ : Vec 1 => XH) γ := differentiableAt_const _
  have hgather_diff : DifferentiableAt ℝ
      (fun y : Vec 1 => fun k : Fin m => y ((fun _ : Fin m => (0 : Fin 1)) k)) γ :=
    (reindexCLM (fun _ : Fin m => (0 : Fin 1))).differentiableAt
  rw [show (fun (y : Vec 1) (k : Fin m) => XH k * y ((fun _ : Fin m => (0 : Fin 1)) k))
        = (fun y k => (fun _ k' => XH k') y k
                      * (fun y' k' => y' ((fun _ : Fin m => (0 : Fin 1)) k')) y k) from rfl,
      pdiv_mul _ _ _ hconstXH_diff hgather_diff,
      pdiv_const XH γ i j,
      pdiv_reindex (fun _ : Fin m => (0 : Fin 1)) γ i j]
  have hi0 : i = 0 := Fin.eq_zero i
  subst hi0
  simp

/-- Generic scalar-shift Jacobian: `∂(C_k + β(0))/∂β = 1`. The
    `pdiv_layerNorm_beta` recipe, coefficient-abstracted. -/
private theorem pdiv_scalarAffine_beta {m : Nat} (C : Vec m) (β : Vec 1)
    (i : Fin 1) (j : Fin m) :
    pdiv (fun y : Vec 1 => fun k => C k + y ((fun _ : Fin m => (0 : Fin 1)) k)) β i j
      = 1 := by
  have hconst_diff : DifferentiableAt ℝ (fun _ : Vec 1 => C) β := differentiableAt_const _
  have hgather_diff : DifferentiableAt ℝ
      (fun y : Vec 1 => fun k : Fin m => y ((fun _ : Fin m => (0 : Fin 1)) k)) β :=
    (reindexCLM (fun _ : Fin m => (0 : Fin 1))).differentiableAt
  rw [show (fun (y : Vec 1) (k : Fin m) => C k + y ((fun _ : Fin m => (0 : Fin 1)) k))
        = (fun y k => (fun _ k' => C k') y k
                      + (fun y' k' => y' ((fun _ : Fin m => (0 : Fin 1)) k')) y k) from rfl,
      pdiv_add _ _ _ hconst_diff hgather_diff,
      pdiv_const C β i j, zero_add,
      pdiv_reindex (fun _ : Fin m => (0 : Fin 1)) β i j]
  have hi0 : i = 0 := Fin.eq_zero i
  subst hi0
  simp

/-- Rowwise scalar-LN as a function of γ (β, X fixed), affinely: the coefficient
    at flat index `(r,k)` is the row-r normalized value `x̂_r(k)`. -/
private theorem rowLN_gamma_affine (N D : Nat) (ε β : ℝ) (X : Mat N D) :
    (fun γ' : Vec 1 => Mat.flatten (fun r => layerNormForward D ε (γ' 0) β (X r)))
      = fun y idx =>
          bnXhat D ε (X (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 *
            y ((fun _ : Fin (N * D) => (0 : Fin 1)) idx) + β := by
  funext y idx
  simp only [Mat.flatten, layerNormForward, bnForward]
  ring

/-- Rowwise scalar-LN as a function of β (γ, X fixed). -/
private theorem rowLN_beta_affine (N D : Nat) (ε γ : ℝ) (X : Mat N D) :
    (fun β' : Vec 1 => Mat.flatten (fun r => layerNormForward D ε γ (β' 0) (X r)))
      = fun y idx =>
          γ * bnXhat D ε (X (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 +
            y ((fun _ : Fin (N * D) => (0 : Fin 1)) idx) := by
  funext y idx
  simp only [Mat.flatten, layerNormForward, bnForward]

/-- **Jacobian of the rowwise scalar-LN w.r.t. γ** — `∂y_(r,k)/∂γ = x̂_r(k)` (dense:
    the shared scalar γ scales every token's every channel). -/
theorem pdiv_rowLN_gamma (N D : Nat) (ε β : ℝ) (X : Mat N D) (γ : Vec 1)
    (i : Fin 1) (idx : Fin (N * D)) :
    pdiv (fun γ' : Vec 1 =>
            Mat.flatten (fun r => layerNormForward D ε (γ' 0) β (X r))) γ i idx
      = bnXhat D ε (X (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 := by
  rw [rowLN_gamma_affine]
  exact pdiv_scalarAffine_gamma
    (fun o => bnXhat D ε (X (finProdFinEquiv.symm o).1) (finProdFinEquiv.symm o).2)
    (fun _ => β) γ i idx

/-- **Jacobian of the rowwise scalar-LN w.r.t. β** — `∂y_(r,k)/∂β = 1`. -/
theorem pdiv_rowLN_beta (N D : Nat) (ε γ : ℝ) (X : Mat N D) (β : Vec 1)
    (i : Fin 1) (idx : Fin (N * D)) :
    pdiv (fun β' : Vec 1 =>
            Mat.flatten (fun r => layerNormForward D ε γ (β' 0) (X r))) β i idx
      = 1 := by
  rw [rowLN_beta_affine]
  exact pdiv_scalarAffine_beta
    (fun o => γ * bnXhat D ε (X (finProdFinEquiv.symm o).1) (finProdFinEquiv.symm o).2)
    β i idx

/-- The rendered **rowwise-LN γ gradient**: the whole-tensor reduce
    `dγ = Σ_r Σ_k dY_(r,k)·x̂_r(k)` — per-row `bn_grad_gamma`, summed over tokens. -/
noncomputable def rowLN_grad_gamma (N D : Nat) (ε : ℝ) (X dY : Mat N D) : ℝ :=
  ∑ r : Fin N, bn_grad_gamma D ε (X r) (dY r)

/-- The rendered **rowwise-LN β gradient**: `dβ = Σ_r Σ_k dY_(r,k)`. -/
noncomputable def rowLN_grad_beta (N D : Nat) (dY : Mat N D) : ℝ :=
  ∑ r : Fin N, bn_grad_beta D (dY r)

/-- **Rowwise scalar-LN γ-gradient bridge.** The rendered whole-tensor reduce
    equals the certified rowwise-LN ∂/∂γ contraction. -/
theorem vit_rowlnGamma_grad_bridge (N D : Nat) (ε β : ℝ) (γ : Vec 1) (X : Mat N D)
    (dy : Vec (N * D)) :
    rowLN_grad_gamma N D ε X (Mat.unflatten dy)
      = ∑ idx : Fin (N * D),
          pdiv (fun γ' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε (γ' 0) β (X r)))
            γ 0 idx * dy idx := by
  simp_rw [pdiv_rowLN_gamma]
  rw [sum_fin_prod N D]
  unfold rowLN_grad_gamma bn_grad_gamma Mat.unflatten
  apply Finset.sum_congr rfl
  intro r _
  apply Finset.sum_congr rfl
  intro k _
  rw [Equiv.symm_apply_apply]
  dsimp only
  ring

/-- **Rowwise scalar-LN β-gradient bridge.** The rendered whole-tensor reduce
    `Σ_r Σ_k dY_(r,k)` equals the certified rowwise-LN ∂/∂β contraction. -/
theorem vit_rowlnBeta_grad_bridge (N D : Nat) (ε γ : ℝ) (β : Vec 1) (X : Mat N D)
    (dy : Vec (N * D)) :
    rowLN_grad_beta N D (Mat.unflatten dy)
      = ∑ idx : Fin (N * D),
          pdiv (fun β' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε γ (β' 0) (X r)))
            β 0 idx * dy idx := by
  simp_rw [pdiv_rowLN_beta, one_mul]
  rw [sum_fin_prod N D]
  unfold rowLN_grad_beta bn_grad_beta Mat.unflatten
  rfl

/-- **Rowwise scalar-LN γ output, certified.** `γⁿ = γ − lr·(Σ_tokens Σ_D dy·x̂)`
    denotes the certified rowwise-LN ∂/∂γ contraction. Covers all five LN sites
    of the representative ViT (LN1/LN2 per block + the final LN). -/
theorem vit_render_rowlngamma_certified (N D : Nat) (ε β : ℝ) (γ : Vec 1)
    (X : Mat N D) (dy : Vec (N * D)) (lr : ℝ) :
    γ 0 - lr * rowLN_grad_gamma N D ε X (Mat.unflatten dy)
      = γ 0 - lr * ∑ idx : Fin (N * D),
          pdiv (fun γ' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε (γ' 0) β (X r)))
            γ 0 idx * dy idx := by
  rw [vit_rowlnGamma_grad_bridge N D ε β γ X dy]

/-- **Rowwise scalar-LN β output, certified.** -/
theorem vit_render_rowlnbeta_certified (N D : Nat) (ε γ : ℝ) (β : Vec 1)
    (X : Mat N D) (dy : Vec (N * D)) (lr : ℝ) :
    β 0 - lr * rowLN_grad_beta N D (Mat.unflatten dy)
      = β 0 - lr * ∑ idx : Fin (N * D),
          pdiv (fun β' : Vec 1 =>
                  Mat.flatten (fun r => layerNormForward D ε γ (β' 0) (X r)))
            β 0 idx * dy idx := by
  rw [vit_rowlnBeta_grad_bridge N D ε γ β X dy]

-- ════════════════════════════════════════════════════════════════
-- § C. pos_embed + cls_token — the two embed-parameter reindex closes
--
-- Both live on `patchEmbed_flat` directly: as a function of the (flattened)
-- position embedding the output is `p + const` (identity Jacobian ⇒ dPos = dy);
-- as a function of the CLS token it is a row-0 masked gather
-- (⇒ dCls = the row-0 slice of the cotangent — exactly `clsSliceF`'s shape).
-- ════════════════════════════════════════════════════════════════

/-- Identity-plus-constant Jacobian: `∂(p_k + C_k)/∂p_i = δ_(i,k)`. -/
private theorem pdiv_id_add_const {m : Nat} (C : Vec m) (x : Vec m) (i j : Fin m) :
    pdiv (fun p : Vec m => fun k => p k + C k) x i j = if i = j then 1 else 0 := by
  have h_id : DifferentiableAt ℝ (fun p : Vec m => p) x := differentiableAt_id
  have h_const : DifferentiableAt ℝ (fun _ : Vec m => C) x := differentiableAt_const _
  rw [show (fun p : Vec m => fun k => p k + C k)
        = (fun p k => (fun w : Vec m => w) p k + (fun _ : Vec m => C) p k) from rfl,
      pdiv_add _ _ _ h_id h_const, pdiv_const, add_zero, pdiv_id]

/-- Masked-gather-plus-constant Jacobian:
    `∂(mask_k·cl_(σ k) + C_k)/∂cl_i = mask_k·δ_(i,σ k)`. -/
private theorem pdiv_maskGather_add_const {m D : Nat} (mask : Vec m) (σ : Fin m → Fin D)
    (C : Vec m) (x : Vec D) (i : Fin D) (j : Fin m) :
    pdiv (fun cl : Vec D => fun k => mask k * cl (σ k) + C k) x i j
      = mask j * (if i = σ j then 1 else 0) := by
  have h_mask : DifferentiableAt ℝ (fun _ : Vec D => mask) x := differentiableAt_const _
  have h_gather : DifferentiableAt ℝ (fun (w : Vec D) (k : Fin m) => w (σ k)) x :=
    (reindexCLM σ).differentiableAt
  have h_mul : DifferentiableAt ℝ (fun (w : Vec D) (k : Fin m) => mask k * w (σ k)) x :=
    h_mask.mul h_gather
  have h_const : DifferentiableAt ℝ (fun _ : Vec D => C) x := differentiableAt_const _
  rw [show (fun cl : Vec D => fun k => mask k * cl (σ k) + C k)
        = (fun cl k =>
            (fun (w : Vec D) (k' : Fin m) =>
              (fun _ : Vec D => mask) w k' * (fun (w' : Vec D) (k'' : Fin m) => w' (σ k'')) w k') cl k +
            (fun _ : Vec D => C) cl k) from rfl,
      pdiv_add _ _ _ h_mul h_const, pdiv_const, add_zero,
      pdiv_mul _ _ _ h_mask h_gather, pdiv_const,
      pdiv_reindex σ x i j]
  ring

/-- **Jacobian of `patchEmbed_flat` w.r.t. the (flattened) position embedding** —
    the identity: pos is broadcast-added to every token. -/
theorem pdiv_patchEmbed_pos {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (i j : Fin ((N + 1) * D)) :
    pdiv (fun p : Vec ((N + 1) * D) =>
            patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
      (Mat.flatten pos) i j = if i = j then 1 else 0 := by
  rw [show (fun p : Vec ((N + 1) * D) =>
              patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
        = (fun p : Vec ((N + 1) * D) => fun idx =>
            p idx +
            patchEmbed_flat ic H W P N D Wc bc cls (fun _ _ => (0 : ℝ)) img idx) from by
      funext p idx
      unfold patchEmbed_flat
      simp only [Mat.unflatten, Prod.mk.eta, Equiv.apply_symm_apply, zero_add]]
  exact pdiv_id_add_const _ (Mat.flatten pos) i j

/-- **pos-embed output, certified.** The pos Jacobian is the identity, so the
    rendered `dPos = dy` (the cotangent itself, batch-summed by the batched
    render) is the certified contraction. -/
theorem vit_render_pos_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin ((N + 1) * D)) :
    Mat.flatten pos i - lr * dy i
      = Mat.flatten pos i - lr * ∑ j : Fin ((N + 1) * D),
          pdiv (fun p : Vec ((N + 1) * D) =>
                  patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
            (Mat.flatten pos) i j * dy j := by
  simp_rw [pdiv_patchEmbed_pos]
  rw [Finset.sum_eq_single i
      (fun j _ hne => by rw [if_neg (Ne.symm (Ne.symm hne).symm), zero_mul])
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [if_pos rfl, one_mul]

/-- The rendered **CLS-token gradient**: the row-0 slice of the patch-embed
    output cotangent (`clsSliceF`'s shape, applied to the embed cotangent). -/
noncomputable def cls_token_grad {N D : Nat} (dy : Vec ((N + 1) * D)) : Vec D :=
  fun i => dy (finProdFinEquiv ((0 : Fin (N + 1)), i))

/-- **Jacobian of `patchEmbed_flat` w.r.t. the CLS token** — the row-0 masked
    gather: `∂y_(n,k)/∂cls_i = [n = 0]·δ_(i,k)`. -/
theorem pdiv_patchEmbed_cls {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (i : Fin D) (j : Fin ((N + 1) * D)) :
    pdiv (fun cl : Vec D =>
            patchEmbed_flat ic H W P N D Wc bc cl pos img) cls i j
      = (if (finProdFinEquiv.symm j).1.val = 0 then (1 : ℝ) else 0) *
          (if i = (finProdFinEquiv.symm j).2 then 1 else 0) := by
  rw [show (fun cl : Vec D => patchEmbed_flat ic H W P N D Wc bc cl pos img)
        = (fun cl : Vec D => fun idx : Fin ((N + 1) * D) =>
            (fun o : Fin ((N + 1) * D) =>
              if (finProdFinEquiv.symm o).1.val = 0 then (1 : ℝ) else 0) idx *
              cl ((fun o : Fin ((N + 1) * D) => (finProdFinEquiv.symm o).2) idx) +
            patchEmbed_flat ic H W P N D Wc bc (fun _ => (0 : ℝ)) pos img idx) from by
      funext cl idx
      unfold patchEmbed_flat
      by_cases h : (finProdFinEquiv.symm idx).1.val = 0
      · simp only [h, if_true]
        ring
      · simp only [h, if_false]
        ring]
  exact pdiv_maskGather_add_const _ _ _ cls i j

/-- **CLS-token output, certified.** `clsⁿ = cls − lr·(row-0 slice of the embed
    cotangent)` denotes the certified ∂(patchEmbed)/∂cls contraction. -/
theorem vit_render_cls_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin D) :
    cls i - lr * cls_token_grad dy i
      = cls i - lr * ∑ j : Fin ((N + 1) * D),
          pdiv (fun cl : Vec D =>
                  patchEmbed_flat ic H W P N D Wc bc cl pos img) cls i j * dy j := by
  simp_rw [pdiv_patchEmbed_cls]
  unfold cls_token_grad
  congr 1
  rw [sum_fin_prod (N + 1) D]
  rw [Finset.sum_eq_single (0 : Fin (N + 1))
      (fun n _ hne => by
        apply Finset.sum_eq_zero
        intro k _
        rw [Equiv.symm_apply_apply]
        dsimp only
        rw [if_neg (by simpa using (Fin.val_ne_of_ne hne)), zero_mul, zero_mul])
      (fun h => absurd (Finset.mem_univ _) h)]
  rw [Finset.sum_eq_single i
      (fun k _ hne => by
        rw [Equiv.symm_apply_apply]
        dsimp only
        rw [if_neg (Ne.symm hne), mul_zero, zero_mul])
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [Equiv.symm_apply_apply]
  dsimp only
  simp

-- The classifier head (`dense Wcls bcls` on the CLS vector) is covered VERBATIM by the
-- existing M2 `weight_grad_bridge`/`bias_grad_bridge` (`dense_weight_grad_correct`/
-- `dense_bias_grad_correct`) at the `[D, nClasses]` shape — single-vector dense, nothing
-- to row-lift. Softmax and the 1/√d scale carry no parameters. With the per-token dense
-- W/b family (§ A: Wq/Wk/Wv/Wo, Wfc1/Wfc2 + biases), the row-lifted scalar-LN γ/β (§ B:
-- all five sites), and pos/cls (§ C), every parameter family of the representative ViT
-- train step except the patch-projection conv `Wp`/`bp` (§ E, follow-up) is certified
-- `θ − lr·(certified Jacobian · cotangent)`.

end Proofs
