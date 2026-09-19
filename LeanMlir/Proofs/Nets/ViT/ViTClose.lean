import LeanMlir.Proofs.Nets.ViT.ViTFwdGraph

/-! # Closing the ViT render — the parameter-gradient close (ch10 Item C)

`planning/archive/vit_close.md` Item C: the per-parameter gradient bridges `ViTFold` delegates to.
Generic in the cotangent `dy` the backward chain delivers at each site's output
(pinning that cotangent to the actual attention chain is the optional Item D), batch-1 —
everything in a ViT is per-example separable (the EfficientNet contrast).

| family (render SSA)                  | forward fn                 | certified by |
|--------------------------------------|----------------------------|--------------|
| Wq/Wk/Wv/Wo, Wfc1/Wfc2 + biases      | per-token dense (rowwise)  | `vit_render_rowdense{W,b}_certified` (**new family**): `dW = Σ_tokens xᵣ ⊗ dyᵣ`, `db = Σ_tokens dyᵣ` — the M2 outer-product bridge row-lifted |
| classifier `Wcls`/`bcls`             | dense on the CLS row       | M2 `weight/bias_grad_bridge` (**reuse** — single-vector dense) |
| LN γ/β (vector, per-token)           | rowwise vector LayerNorm   | `vit_vecln{Gamma,Beta}_grad_bridge` (`ViTVecLN`) |
| `pos_embed`                          | additive (`patchEmbed_flat`) | `vit_render_pos_certified`: the pos-Jacobian is the identity ⇒ `dPos = dy` |
| `cls_token`                          | row-0 scatter (`patchEmbed_flat`) | `vit_render_cls_certified`: masked-gather Jacobian ⇒ `dCls = dy` row-0 slice |
| patch conv `Wp`/`bp`                 | stride-P conv (`patchEmbed_flat`) | `vit_render_patch{W,b}_certified`: kernel-linear w/ constant guarded reads ⇒ `dWp = Σ_p read·dy_(p+1)`, `dbp = Σ_p dy_(p+1)` (CLS row excluded) |
| attention internals (softmax, scale) | —                          | no parameters |

One genuinely-new bridge family (everything else is reuse or a reindex):

* **Per-token dense W/b** — every row of `[N,a]` through the same `W : [a,c]` (+ `b`).
  The W-Jacobian of the flattened rowwise dense is block-sparse —
  `∂y_(r,k)/∂W_(i,j) = X_(r,i)·δ_(k,j)` — so the rendered per-token outer-product reduce
  `dW_(i,j) = Σ_r X_(r,i)·dY_(r,j)` (one `dot_general` contracting the token axis) is the
  certified contraction. Covers Wq/Wk/Wv/Wo, Wfc1/Wfc2 and their biases at every block.

The classifier head (`dense` on the CLS vector) is VERBATIM M2 `weight/bias_grad_bridge`
reuse at `[D, nClasses]`. The patch-embed conv `Wp`/`bp` (§ E) closes over
`patchEmbed_flat` directly — the kernel is the VARIABLE and the pad-guarded image reads
are CONSTANT coefficients (the mirror of the input-grad case), so the forward is affine in
the kernel and `pdiv_of_affine` applies with the CLS row masked out. 3-axiom clean by
construction.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § A. Per-token dense W/b — the row-lifted M2 family (genuinely new)
--
-- `fun v => Mat.flatten (fun r => dense (Mat.unflatten v) b (X r))` is, in `v`, linear plus
-- the constant bias — `y_(r,k) = Σ_i X_(r,i)·v_(i,k) + b_k` at the flat `[N·c]` output
-- index — so `pdiv_of_affine` reads the Jacobian off the basis vector.
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
  rw [show (fun v : Vec (a * c) => Mat.flatten (fun r => dense (Mat.unflatten v) bb (X r))) =
      fun v => (fun o : Fin (N * c) => ∑ i' : Fin a, X (finProdFinEquiv.symm o).1 i' *
          v (finProdFinEquiv (i', (finProdFinEquiv.symm o).2))) +
        fun o => bb (finProdFinEquiv.symm o).2 from rfl,
    pdiv_of_affine _ _ (fun _ _ => by funext; simp [mul_add, Finset.sum_add_distrib])
      (fun _ _ => by funext; simp [Finset.mul_sum, mul_left_comm])]
  obtain ⟨⟨r, k⟩, rfl⟩ := finProdFinEquiv.surjective idx
  rcases eq_or_ne j' k with rfl | h
  · simp [Prod.ext_iff]
  · simp [Prod.ext_iff, h, Ne.symm h]

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
  rw [sum_finProdFinEquiv (m := N) (n := c)]
  simp [rowDense_weight_grad, Mat.unflatten]

/-- **Per-token dense b-gradient bridge.** The rendered token-axis reduce equals
    the certified rowwise-dense ∂/∂b contraction. -/
theorem vit_rowDenseb_grad_bridge {N a c : Nat} (W : Mat a c) (X : Mat N a)
    (bb : Vec c) (dy : Vec (N * c)) (i : Fin c) :
    rowDense_bias_grad (Mat.unflatten dy) i
      = ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r)))
               bb i o * dy o := by
  -- Jacobian: ∂y_(r,k)/∂b_i = δ_(k,i) — gather + constant.
  have hpdiv : ∀ o : Fin (N * c),
      pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) bb i o =
        if i = (finProdFinEquiv.symm o).2 then 1 else 0 := by
    intro o
    rw [show (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) =
          fun b' => (fun o' : Fin (N * c) => b' (finProdFinEquiv.symm o').2) +
            fun o' => ∑ i' : Fin a, X (finProdFinEquiv.symm o').1 i' *
              W i' (finProdFinEquiv.symm o').2 from by
        funext b' o'; unfold dense Mat.flatten; exact add_comm _ _,
      pdiv_of_affine _ _ (fun _ _ => rfl) (fun _ _ => rfl)]
    simp only [basisVec_apply, @eq_comm _ _ i]
  simp_rw [hpdiv]
  rw [sum_finProdFinEquiv (m := N) (n := c)]
  simp [rowDense_bias_grad, Mat.unflatten]

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
-- § C. pos_embed + cls_token — the two embed-parameter reindex closes
--
-- Both live on `patchEmbed_flat` directly: as a function of the (flattened)
-- position embedding the output is `p + const` (identity Jacobian ⇒ dPos = dy);
-- as a function of the CLS token it is a row-0 masked gather
-- (⇒ dCls = the row-0 slice of the cotangent — exactly `clsSliceF`'s shape).
-- ════════════════════════════════════════════════════════════════

/-- Identity-plus-constant Jacobian: `∂(p_k + C_k)/∂p_i = δ_(i,k)`. -/
theorem pdiv_id_add_const {m : Nat} (C : Vec m) (x : Vec m) (i j : Fin m) :
    pdiv (fun p : Vec m => fun k => p k + C k) x i j = if i = j then 1 else 0 := by
  rw [show (fun p : Vec m => fun k => p k + C k) = fun p => p + C from rfl,
    pdiv_of_affine _ _ (fun _ _ => rfl) (fun _ _ => rfl)]
  simp only [basisVec_apply, @eq_comm _ j i]

/-- Masked-gather-plus-constant Jacobian:
    `∂(mask_k·cl_(σ k) + C_k)/∂cl_i = mask_k·δ_(i,σ k)`. -/
theorem pdiv_maskGather_add_const {m D : Nat} (mask : Vec m) (σ : Fin m → Fin D)
    (C : Vec m) (x : Vec D) (i : Fin D) (j : Fin m) :
    pdiv (fun cl : Vec D => fun k => mask k * cl (σ k) + C k) x i j
      = mask j * (if i = σ j then 1 else 0) := by
  rw [show (fun cl : Vec D => fun k => mask k * cl (σ k) + C k)
      = fun cl => (fun k => mask k * cl (σ k)) + C from rfl,
    pdiv_of_affine _ _ (fun _ _ => by funext; simp [mul_add])
      (fun _ _ => by funext; simp [mul_left_comm])]
  simp only [basisVec_apply, @eq_comm _ (σ j) i]

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
  simp [pdiv_patchEmbed_pos]

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
      · simp only [h, ite_true]
        ring
      · simp only [h, ite_false]
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
  rw [sum_finProdFinEquiv (m := N + 1) (n := D)]
  simp [cls_token_grad]

-- ════════════════════════════════════════════════════════════════
-- § E. Patch-projection conv Wp/bp — the embed kernel close
--
-- The KEY structural fact: as a function of the (flattened) kernel, `patchEmbed_flat`
-- is linear with CONSTANT coefficients — the pad-guarded image reads sit in the
-- coefficient, not the variable (the mirror of the input-grad case, where the
-- pad-eval calculus was needed). So `pdiv_of_affine` reads the Jacobian off the basis
-- vector, as in §A, with the CLS row's coefficient zero.
-- ════════════════════════════════════════════════════════════════

/-- The pad-guarded patch read of `patchEmbed_flat`, named: input pixel
    `(c, h'·P + kh, w'·P + kw)` of patch `p` (row-major patch grid of width
    `W/P`), zero out of range. Constant in the kernel. -/
noncomputable def patchRead (ic H W P : Nat) (img : Vec (ic * H * W))
    (c : Fin ic) (kh kw : Fin P) (p : Nat) : ℝ :=
  let W' := W / P
  let h' := p / W'
  let w' := p % W'
  let hh := h' * P + kh.val
  let ww := w' * P + kw.val
  if hpad : hh < H ∧ ww < W then
    img (finProdFinEquiv (finProdFinEquiv (c, ⟨hh, hpad.1⟩), ⟨ww, hpad.2⟩))
  else 0

/-- **Jacobian of `patchEmbed_flat` w.r.t. the (flattened) patch kernel** —
    `∂y_(n,dd)/∂W_(d,c,kh,kw) = [n ≠ 0]·δ_(dd,d)·read(c,kh,kw, patch n−1)`. -/
theorem pdiv_patchEmbed_W {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W))
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) (idx : Fin ((N + 1) * D)) :
    pdiv (fun v : Vec (D * ic * P * P) =>
            patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
      (Kernel4.flatten Wc)
      (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) idx
      = if (finProdFinEquiv.symm idx).2 = d then
          (if (finProdFinEquiv.symm idx).1.val = 0 then 0
           else patchRead ic H W P img c kh kw ((finProdFinEquiv.symm idx).1.val - 1))
        else 0 := by
  -- Linear in the kernel (the pad-guarded reads are constant coefficients, zero on the CLS
  -- row) plus the kernel-free part, so `pdiv_of_affine` reads the entry off the basis vector.
  rw [show (fun v : Vec (D * ic * P * P) =>
              patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img) =
        fun v => (fun o : Fin ((N + 1) * D) =>
          if (finProdFinEquiv.symm o).1.val = 0 then 0 else
            ∑ c' : Fin ic, ∑ kh' : Fin P, ∑ kw' : Fin P,
              v (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv
                  ((finProdFinEquiv.symm o).2, c'), kh'), kw')) *
                patchRead ic H W P img c' kh' kw' ((finProdFinEquiv.symm o).1.val - 1)) +
          patchEmbed_flat ic H W P N D (fun _ _ _ _ => 0) bc cls pos img from by
      funext v o
      by_cases h : (finProdFinEquiv.symm o).1.val = 0 <;>
        simp only [patchEmbed_flat, patchRead, Kernel4.unflatten, Pi.add_apply, h, ite_true,
          ite_false, zero_mul, Finset.sum_const_zero, add_zero, zero_add]
      ring,
    pdiv_of_affine _ _
      (fun _ _ => by
        funext; simp only [Pi.add_apply]; split_ifs <;> simp [add_mul, Finset.sum_add_distrib])
      (fun _ _ => by
        funext; simp only [Pi.smul_apply, smul_eq_mul]
        split_ifs <;> simp [Finset.mul_sum, mul_assoc])]
  obtain ⟨⟨n, dd⟩, rfl⟩ := finProdFinEquiv.surjective idx
  rcases eq_or_ne dd d with rfl | h
  · simp [Prod.ext_iff, ite_and]
  · simp [Prod.ext_iff, h]

/-- The rendered **patch-kernel gradient**: for each tap `(d,c,kh,kw)`, the
    patch-grid reduce `Σ_p read(c,kh,kw,p)·dy_(p+1,d)` — the "dilate dy /
    valid conv" weight grad, with the CLS row (token 0) excluded. -/
noncomputable def patchEmbed_weight_grad (ic H W P N D : Nat)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) : Kernel4 D ic P P :=
  fun d c kh kw =>
    ∑ n : Fin N, patchRead ic H W P img c kh kw n.val *
      dy (finProdFinEquiv (n.succ, d))

/-- The rendered **patch bias gradient**: `db_d = Σ_p dy_(p+1,d)` (the CLS row
    excluded — token 0 carries no conv bias). -/
noncomputable def patchEmbed_bias_grad (N D : Nat) (dy : Vec ((N + 1) * D)) : Vec D :=
  fun d => ∑ n : Fin N, dy (finProdFinEquiv (n.succ, d))

/-- **Patch-kernel gradient bridge.** The rendered patch-grid reduce equals the
    certified ∂(patchEmbed)/∂W contraction. -/
theorem vit_patchW_grad_bridge {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D))
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    patchEmbed_weight_grad ic H W P N D img dy d c kh kw
      = ∑ o : Fin ((N + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
            (Kernel4.flatten Wc)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o
            * dy o := by
  simp_rw [pdiv_patchEmbed_W]
  rw [sum_finProdFinEquiv (m := N + 1) (n := D)]
  simp [patchEmbed_weight_grad, Fin.sum_univ_succ]

/-- **Patch-kernel output, certified.** `Wpⁿ = Wp − lr·(patch-grid reduce)`
    denotes the certified ∂(patchEmbed)/∂Wp contraction. -/
theorem vit_render_patchW_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) (lr : ℝ)
    (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    Wc d c kh kw - lr * patchEmbed_weight_grad ic H W P N D img dy d c kh kw
      = Wc d c kh kw - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
            (Kernel4.flatten Wc)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o
            * dy o := by
  rw [vit_patchW_grad_bridge Wc bc cls pos img dy d c kh kw]

/-- **Jacobian of `patchEmbed_flat` w.r.t. the patch bias** — the row-masked
    gather `∂y_(n,k)/∂bc_i = [n ≠ 0]·δ_(i,k)` (token 0 is the CLS row). -/
theorem pdiv_patchEmbed_b {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (i : Fin D) (j : Fin ((N + 1) * D)) :
    pdiv (fun b' : Vec D =>
            patchEmbed_flat ic H W P N D Wc b' cls pos img) bc i j
      = (if (finProdFinEquiv.symm j).1.val = 0 then (0 : ℝ) else 1) *
          (if i = (finProdFinEquiv.symm j).2 then 1 else 0) := by
  rw [show (fun b' : Vec D => patchEmbed_flat ic H W P N D Wc b' cls pos img)
        = (fun b' : Vec D => fun o : Fin ((N + 1) * D) =>
            (fun o' : Fin ((N + 1) * D) =>
              if (finProdFinEquiv.symm o').1.val = 0 then (0 : ℝ) else 1) o *
              b' ((fun o' : Fin ((N + 1) * D) => (finProdFinEquiv.symm o').2) o) +
            patchEmbed_flat ic H W P N D Wc (fun _ => (0 : ℝ)) cls pos img o) from by
      funext b' o
      unfold patchEmbed_flat
      by_cases h : (finProdFinEquiv.symm o).1.val = 0
      · simp only [h, ite_true]
        ring
      · simp only [h, ite_false]
        ring]
  exact pdiv_maskGather_add_const _ _ _ bc i j

/-- **Patch bias gradient bridge.** The rendered CLS-row-excluded reduce equals
    the certified ∂(patchEmbed)/∂bc contraction. -/
theorem vit_patchb_grad_bridge {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) (i : Fin D) :
    patchEmbed_bias_grad N D dy i
      = ∑ o : Fin ((N + 1) * D),
          pdiv (fun b' : Vec D =>
                  patchEmbed_flat ic H W P N D Wc b' cls pos img) bc i o * dy o := by
  simp_rw [pdiv_patchEmbed_b]
  rw [sum_finProdFinEquiv (m := N + 1) (n := D)]
  simp [patchEmbed_bias_grad, Fin.sum_univ_succ]

/-- **Patch bias output, certified.** -/
theorem vit_render_patchb_certified {ic H W P N D : Nat}
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (img : Vec (ic * H * W)) (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin D) :
    bc i - lr * patchEmbed_bias_grad N D dy i
      = bc i - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun b' : Vec D =>
                  patchEmbed_flat ic H W P N D Wc b' cls pos img) bc i o * dy o := by
  rw [vit_patchb_grad_bridge Wc bc cls pos img dy i]

-- The classifier head (`dense Wcls bcls` on the CLS vector) is covered VERBATIM by the
-- existing M2 `weight_grad_bridge`/`bias_grad_bridge` (`dense_weight_grad_correct`/
-- `dense_bias_grad_correct`) at the `[D, nClasses]` shape — single-vector dense, nothing
-- to row-lift. Softmax and the 1/√d scale carry no parameters. With the per-token dense
-- W/b family (§ A), the row-lifted scalar-LN γ/β (§ B), pos/cls (§ C), and the patch
-- conv Wp/bp (§ E), EVERY parameter family of the representative ViT train step is
-- certified `θ − lr·(certified Jacobian · cotangent)`.

end Proofs
