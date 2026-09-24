import LeanMlir.Proofs.Foundation.Batched

/-! # `batchMap` at a POINT — the pointwise peer of `batchMap_has_vjp`

`EfficientNetChainClose.lean` lifts a batch-separable op's VJP to the whole batch in the GLOBAL
form: `batchMap_has_vjp` takes `HasVJP f` and `Differentiable ℝ f`. EfficientNet never needed
anything weaker — swish is smooth everywhere and B0's stem has no pooling — so the pointwise peer
was never written.

ResNet-34 needs it. Its stem is `batchMap N (maxPool3s2Flat c h w) ∘ cbReluStridedB`, and a
max-pool has no derivative at a tie: `maxPool3s2Flat_has_vjp_at_vec` is `_at` by nature. Without
the lift below, r34's whole-net VJP at batch BN cannot be assembled — the one thing standing
between `ResNet34FullB.lean` and T1.

⭐ **The weakening is exactly as narrow as it looks.** `pdivMat_rowIndep` requires
`Differentiable ℝ g`, and its docstring explains why (a non-differentiable coordinate makes
`fderiv` junk and breaks the per-row decomposition) — but every use of that hypothesis is at a
ROW of the matrix it is stated about. So it weakens to `∀ r, DifferentiableAt ℝ g (A r)`, which
is how `Tensor.lean`'s `pdivMat_rowIndep_perRow_at` states it; `pdivMat_rowIndep_at` below is that
lemma with the same map on every row.

⚠ The r34 stem's instance lives with r34's VJP, not here — `maxPool3s2Flat_has_vjp_at_vec` is
in the `Float` tier and this is a `Foundation` file. It is two lines there:
`batchMap_has_vjp_at _ v (fun r => maxPool3s2Flat_has_vjp_at_vec (Mat.unflatten v r) (hs r))
(fun r => maxPool3s2Flat_differentiableAt_vec (Mat.unflatten v r) (hs r) hc hh hw)`, with no
glue between the two, which is what says the lemma below has the right shape.
-/

namespace Proofs

open scoped BigOperators

/-- **Row-wise Jacobian decomposition at a point.** `pdivMat_rowIndep_perRow_at` (`Tensor.lean`)
    with one map `g` for every row: global differentiability of `g` weakened to differentiability
    at each ROW of `A`. -/
theorem pdivMat_rowIndep_at {m n p : Nat} (g : Vec n → Vec p)
    (A : Mat m n) (h_g_diff : ∀ r : Fin m, DifferentiableAt ℝ g (A r))
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 := by
  rw [pdivMat_rowIndep_perRow_at (fun _ => g) A h_g_diff]
  split_ifs with h <;> simp [h]

-- ════════════════════════════════════════════════════════════════
-- § `batchMap` at a point
-- ════════════════════════════════════════════════════════════════

/-- **`batchMap N f` is differentiable at `v`** when `f` is differentiable at each of `v`'s rows.
    The pointwise peer of `batchMap_differentiable`. -/
theorem batchMap_differentiableAt {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a))
    (hf : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r)) :
    DifferentiableAt ℝ (StableHLO.batchMap N f) v := by
  unfold Mat.unflatten at hf
  unfold StableHLO.batchMap
  fun_prop (disch := assumption)

/-- **`batchMap`'s Jacobian is block-diagonal across the batch, at a point.** `pdivMat_rowIndep_at`
    read through `batchMap_eq_rowwiseFlat`: entry `(idx, jdx)` vanishes unless the two indices name
    the same example, and is `f`'s own entry on that example's row otherwise. -/
theorem pdiv_batchMap_at {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r))
    (idx : Fin (N * a)) (jdx : Fin (N * b)) :
    pdiv (StableHLO.batchMap N f) v idx jdx =
      if (finProdFinEquiv.symm idx).1 = (finProdFinEquiv.symm jdx).1 then
        pdiv f (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2 (finProdFinEquiv.symm jdx).2
      else 0 := by
  have h := pdivMat_rowIndep_at f (Mat.unflatten v) hf_diff
      (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
      (finProdFinEquiv.symm jdx).1 (finProdFinEquiv.symm jdx).2
  unfold pdivMat at h
  simp only [Mat.flatten_unflatten, Prod.mk.eta, Equiv.apply_symm_apply] at h
  exact h

/-- ⭐ **`batchMap N f`'s VJP at a point** — the pointwise peer of `batchMap_has_vjp`, and the lift
    a batch-separable op with a KINK needs. The backward reshapes to `[N, ·]` and runs each
    example's own `_at` backward on its own row, exactly as the global one runs `f.backward`
    row-wise.

    ⚠ Unlike `batchMap_has_vjp` this is built field by field rather than transported along
    `batchMap_eq_rowwiseFlat` with `▸`: an `Eq.mpr` blocks `.backward` from reducing, which a
    whole-net certified-backward tie later needs. -/
noncomputable def batchMap_has_vjp_at {N a b : Nat} (f : Vec a → Vec b) (v : Vec (N * a))
    (hf : ∀ r : Fin N, HasVJPAt f (Mat.unflatten v r))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r)) :
    HasVJPAt (StableHLO.batchMap N f) v where
  backward := fun dy idx =>
    (hf (finProdFinEquiv.symm idx).1).backward
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2
  correct := by
    intro dy idx
    simp only [sum_finProdFinEquiv, pdiv_batchMap_at f v hf_diff, Equiv.symm_apply_apply, ite_mul,
      zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq, Finset.mem_univ,
      ite_true]
    exact (hf _).correct _ _

/-- **A batched backward tie from the per-example one.** If `g` is each row's certified backward,
    `batchMapAux N g v` IS the lifted witness's backward: both read example `r`'s row of `v` and
    `dy`, so the two agree index by index once the rows do. -/
theorem batchMapAux_eq_batchMap_has_vjp_at {N a b : Nat} (f : Vec a → Vec b)
    (g : Vec a → Vec b → Vec a) (v : Vec (N * a))
    (hf : ∀ r : Fin N, HasVJPAt f (Mat.unflatten v r))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r))
    (hg : ∀ r : Fin N, g (Mat.unflatten v r) = (hf r).backward) :
    StableHLO.batchMapAux N g v = (batchMap_has_vjp_at f v hf hf_diff).backward := by
  funext dy idx
  exact congrFun (congrFun (hg _) _) _

/-- The linear-backward form of `batchMapAux_eq_batchMap_has_vjp_at`: a backward that ignores the
    saved input lifts by `batchMap`. -/
theorem batchMap_eq_batchMap_has_vjp_at {N a b : Nat} (f : Vec a → Vec b) (g : Vec b → Vec a)
    (v : Vec (N * a)) (hf : ∀ r : Fin N, HasVJPAt f (Mat.unflatten v r))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ f (Mat.unflatten v r))
    (hg : ∀ r : Fin N, g = (hf r).backward) :
    StableHLO.batchMap N g = (batchMap_has_vjp_at f v hf hf_diff).backward := by
  funext dy idx
  exact congrFun (congrFun (hg _) _) _

-- ════════════════════════════════════════════════════════════════
-- § `batchMap` distributes over composition
-- ════════════════════════════════════════════════════════════════

/-- **`batchMap B (g ∘ f) = batchMap B g ∘ batchMap B f`.** Both sides read example `p.1`'s slice
    of the input and run `g ∘ f` on it; peeling the inner lift off at one example is
    `batchSlice_batchMap`. The two spellings are NOT `rfl` — they agree only up to
    `finProdFinEquiv.symm_apply_apply` — which is why every batched whole-net chain saves its
    activations stage by stage (`vitSavedBodyB`, `cnxSavedB1 … cnxSavedB10`) and its shape check
    goes through this lemma. Shared by the ViT and ConvNeXt batched ties. -/
theorem batchMap_comp (B : Nat) {a b c : Nat} (f : Vec a → Vec b) (g : Vec b → Vec c) :
    StableHLO.batchMap B (g ∘ f) = StableHLO.batchMap B g ∘ StableHLO.batchMap B f := by
  funext x idx
  show g (f (StableHLO.batchSlice B a x (finProdFinEquiv.symm idx).1)) (finProdFinEquiv.symm idx).2
    = g (StableHLO.batchSlice B b (StableHLO.batchMap B f x) (finProdFinEquiv.symm idx).1)
        (finProdFinEquiv.symm idx).2
  rw [StableHLO.batchSlice_batchMap]

-- ════════════════════════════════════════════════════════════════
-- § `batchMap` is differentiable, and its VJP is block-diagonal (the per-example VJP, batched)
-- ════════════════════════════════════════════════════════════════

/-- **`batchMap N f` is the flattened row-wise application of `f`.** Reading the output at flat index
    `idx` (decoding to example `m`, coord `c`) gives `f (row m of the input) c` on both sides — the
    `Mat.flatten`/`unflatten` row-major convention is exactly `batchMap`'s `finProdFinEquiv` split. -/
theorem batchMap_eq_rowwiseFlat {N a b : Nat} (f : Vec a → Vec b) :
    StableHLO.batchMap N f
      = fun v : Vec (N * a) => Mat.flatten ((fun A : Mat N a => fun r => f (A r)) (Mat.unflatten v)) := by
  funext v idx
  rfl

/-- **`batchMap N f` is differentiable** when `f` is — it is `f` applied independently per example. -/
@[fun_prop]
theorem batchMap_differentiable {N a b : Nat} (f : Vec a → Vec b) (hf : Differentiable ℝ f) :
    Differentiable ℝ (StableHLO.batchMap N f) := by
  unfold StableHLO.batchMap; fun_prop

/-- **`batchMap N f` VJP — block-diagonal (the genuinely-new lemma).** A batch-separable op's VJP
    applies `f`'s proven VJP independently per example. The backward, like the forward, reshapes to
    `[N, ·]` and runs `f.backward` row-wise. Reuses `rowwise_has_vjp_mat` + `hasVJPMat_to_hasVJP`. This
    is `seBlockFull_has_vjp` / the conv-depthwise-dense VJPs "lifted by batchMap" to the whole batch. -/
noncomputable def batchMap_has_vjp {N a b : Nat} (f : Vec a → Vec b)
    (hf : HasVJP f) (hf_diff : Differentiable ℝ f) :
    HasVJP (StableHLO.batchMap N f) :=
  (batchMap_eq_rowwiseFlat f).symm ▸ hasVJPMat_to_hasVJP (rowwise_has_vjp_mat hf hf_diff)

-- ════════════════════════════════════════════════════════════════
-- § True batch-norm `bnBatchLA` VJP — the proven `bnBatchTensor4`, reindex-conjugated
-- ════════════════════════════════════════════════════════════════

/-- **Reindex VJP at `reindexCLM`** — `reindexVJP`, with the backward spelled as the masked
    sum the IR's scatter ops denote (`reindexVJP_backward`), so the batched graph ties close by
    `rfl`. -/
noncomputable def reindex_has_vjp {a b : Nat} (σ : Fin b → Fin a) :
    HasVJP (reindexCLM σ) where
  backward := fun _v dy => fun i => ∑ k : Fin b, (if i = σ k then dy k else 0)
  correct v dy i :=
    (congrFun (reindexVJP_backward σ v dy) i).symm.trans ((reindexVJP σ).correct v dy i)

/-- **`bnBatchLA` is the proven `bnBatchTensor4`, conjugated by the `mul_assoc` reindex.** Both reindex
    maps are `reindexCLM (Fin.cast …)`; the middle is the genuinely batch-coupled true batch-norm. -/
theorem bnBatchLA_eq_comp (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    StableHLO.bnBatchLA N oc h w ε γ β
      = (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))) ∘
          bnBatchTensor4 N oc h w ε γ β ∘
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm)) := by
  rfl

@[fun_prop]
theorem bnBatchLA_differentiable (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (StableHLO.bnBatchLA N oc h w ε γ β) := by
  rw [bnBatchLA_eq_comp]
  exact (reindexCLM _).differentiable.comp
    ((bnBatchTensor4_differentiable N oc h w ε hε γ β).comp (reindexCLM _).differentiable)

@[fun_prop]
theorem bnBatchLA_continuous (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Continuous (StableHLO.bnBatchLA N oc h w ε γ β) :=
  (bnBatchLA_differentiable N oc h w ε hε γ β).continuous

/-- **True batch-norm VJP at the network's flat index.** `bnBatchLA`'s backward is the proven
    `bnBatchTensor4` VJP (batch-coupled — NOT a `batchMap`), conjugated by the reindex isos. -/
noncomputable def bnBatchLA_has_vjp (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (StableHLO.bnBatchLA N oc h w ε γ β) := by
  rw [bnBatchLA_eq_comp]
  exact vjp_comp _ _
    ((bnBatchTensor4_differentiable N oc h w ε hε γ β).comp (reindexCLM _).differentiable)
    (reindexCLM _).differentiable
    (vjp_comp _ _ (reindexCLM _).differentiable (bnBatchTensor4_differentiable N oc h w ε hε γ β)
      (reindex_has_vjp _) (bnBatchTensor4_has_vjp N oc h w ε hε γ β))
    (reindex_has_vjp _)

end Proofs
