import LeanMlir.Proofs.Foundation.ParamGrad
import LeanMlir.Proofs.Foundation.BatchMapVJPAt
import LeanMlir.Proofs.Foundation.DropSites

/-! # The indexed lift — a different per-example map at every example

`batchMap N f` runs ONE map on every example, and every lemma the whole-net ties of the
batch-separable nets (ViT, ConvNeXt) thread is about that shape: `batchSlice_batchMap`,
`batchMapHasVJPAt`, `HasGradAt.param_batchMap_through`, `batchShard_batchMap`. A drop-path site
breaks it: example `n`'s block scales its branch by example `n`'s own mask entry, so the block is a
different map at every example. `batchMapIdx N f` takes the family `f : Fin N → Vec a → Vec b`,
and this file restates each of those lemmas for it.

`batchMap N f` is `batchMapIdx N (fun _ => f)` and `batchMapAux N f aux` is
`batchMapAuxIdx N (fun _ => f) aux`, both by `rfl` (`batchMap_eq_batchMapIdx`,
`batchMapAux_eq_batchMapAuxIdx`), so a statement made at the indexed lift holds at the uniform
one with no extra step.

* `batchSlice_batchMapIdx` / `batchSlice_batchMapAuxIdx` — example `n` of the lift is `f n` at
  example `n`.
* `batchMapIdxHasVJPAt` / `batchMapIdxHasVJP` — the Jacobian is block-diagonal, with `f n`'s own
  block at example `n` (`pdiv_batchMapIdx_at`, from `pdivMat_rowIndep_perRow_at`, which already
  allows a different map on every row); `batchMapAuxIdx_eq_batchMapIdxHasVJPAt` ties a batched
  backward chain to it.
* `HasGradAt.param_batchMapIdx_through` — a parameter op shared by every example, between an
  indexed prefix and an indexed suffix: the batched node is the parameter's loss gradient.
* `batchShard_batchMapIdx` / `_batchMapAuxIdx` — shard `r`'s family is the global one at
  `finProdFinEquiv (r, ·)`.

**One example's site.** What the family varies by is a drop site read at one example:
`dropScalarOpt s` scales a vector by `s`'s scalar, or is `id` at `none`, and `batchSlice_dropPathOpt`
says the batched site (`dropPathOpt`, `Foundation.DropSites`) is it at every example's own mask
entry. `siteResHasVJP` is the residual with a site on its branch, `v ↦ v + s ⊙ br v`: its backward
is `dy + br.back (s ⊙ dy)` by definition — the skip reads the raw cotangent, the branch the dropped
one (`eBack`'s rule, `dropPath_vjp_is_self`).
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

/-- **Per-example block-apply, a different map at every example.** Example `n` of the batch (the
    `finProdFinEquiv` block `{(n, ·)}`) is mapped by `f n`. -/
noncomputable def batchMapIdx (N : Nat) {a b : Nat} (f : Fin N → Vec a → Vec b) :
    Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f p.1 (fun i : Fin a => x (finProdFinEquiv (p.1, i))) p.2

/-- **`batchMapAux` with a different map at every example**: example `n` is handed its own slice of
    `aux` and mapped by `f n`. -/
noncomputable def batchMapAuxIdx (N : Nat) {s a b : Nat} (f : Fin N → Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) : Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f p.1 (batchSlice N s aux p.1) (batchSlice N a x p.1) p.2

/-- The uniform lift is the indexed one at a constant family. -/
theorem batchMap_eq_batchMapIdx (N : Nat) {a b : Nat} (f : Vec a → Vec b) :
    batchMap N f = batchMapIdx N (fun _ => f) := rfl

/-- …and so is the uniform auxiliary lift. -/
theorem batchMapAux_eq_batchMapAuxIdx (N : Nat) {s a b : Nat} (f : Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) :
    batchMapAux N f aux = batchMapAuxIdx N (fun _ => f) aux := rfl

/-- The auxiliary lift is the indexed one with each example's slice of `aux` applied. -/
theorem batchMapAuxIdx_eq_batchMapIdx (N : Nat) {s a b : Nat} (f : Fin N → Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) :
    batchMapAuxIdx N f aux = batchMapIdx N (fun n => f n (batchSlice N s aux n)) := rfl

/-- `batchSlice` of a `batchMapIdx` is example `n`'s map at the slice. -/
theorem batchSlice_batchMapIdx {N a b : Nat} (f : Fin N → Vec a → Vec b) (x : Vec (N * a))
    (n : Fin N) :
    batchSlice N b (batchMapIdx N f x) n = f n (batchSlice N a x n) := by
  funext i
  simp only [batchSlice, batchMapIdx, Equiv.symm_apply_apply]
  rfl

/-- `batchSlice` of a `batchMapAuxIdx` is example `n`'s map at the two slices. -/
theorem batchSlice_batchMapAuxIdx {N s a b : Nat} (f : Fin N → Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) (x : Vec (N * a)) (n : Fin N) :
    batchSlice N b (batchMapAuxIdx N f aux x) n
      = f n (batchSlice N s aux n) (batchSlice N a x n) := by
  funext i
  simp [batchSlice, batchMapAuxIdx]

/-- **`batchMapIdx` distributes over composition**, example by example. -/
theorem batchMapIdx_comp (B : Nat) {a b c : Nat} (f : Fin B → Vec a → Vec b)
    (g : Fin B → Vec b → Vec c) :
    batchMapIdx B (fun n => g n ∘ f n) = batchMapIdx B g ∘ batchMapIdx B f := by
  funext x idx
  show g (finProdFinEquiv.symm idx).1 (f (finProdFinEquiv.symm idx).1
      (batchSlice B a x (finProdFinEquiv.symm idx).1)) (finProdFinEquiv.symm idx).2
    = g (finProdFinEquiv.symm idx).1
        (batchSlice B b (batchMapIdx B f x) (finProdFinEquiv.symm idx).1)
        (finProdFinEquiv.symm idx).2
  rw [batchSlice_batchMapIdx]

end StableHLO

open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Differentiability and the VJP — block-diagonal, `f n`'s block at example `n`
-- ════════════════════════════════════════════════════════════════

/-- `batchMapIdx N f` is the flattened row-wise application of the family. -/
theorem batchMapIdx_eq_rowwiseFlat {N a b : Nat} (f : Fin N → Vec a → Vec b) :
    batchMapIdx N f
      = fun v : Vec (N * a) => Mat.flatten (fun r => f r (Mat.unflatten v r)) := by
  funext v idx
  rfl

/-- **`batchMapIdx N f` is differentiable at `v`** when each `f r` is at row `r`. -/
theorem batchMapIdx_differentiableAt {N a b : Nat} (f : Fin N → Vec a → Vec b) (v : Vec (N * a))
    (hf : ∀ r : Fin N, DifferentiableAt ℝ (f r) (Mat.unflatten v r)) :
    DifferentiableAt ℝ (batchMapIdx N f) v := by
  rw [differentiableAt_pi]
  intro idx
  have hrow : DifferentiableAt ℝ (fun x : Vec (N * a) => Mat.unflatten x (finProdFinEquiv.symm idx).1) v :=
    (reindexCLM fun j => finProdFinEquiv ((finProdFinEquiv.symm idx).1, j)).differentiableAt
  exact differentiableAt_pi.1 ((hf _).comp v hrow) (finProdFinEquiv.symm idx).2

/-- `batchMapIdx N f` is differentiable when every `f n` is. -/
theorem batchMapIdx_differentiable {N a b : Nat} (f : Fin N → Vec a → Vec b)
    (hf : ∀ n, Differentiable ℝ (f n)) : Differentiable ℝ (batchMapIdx N f) :=
  fun v => batchMapIdx_differentiableAt f v fun r => hf r _

/-- **`batchMapIdx`'s Jacobian is block-diagonal across the batch, at a point**: entry
    `(idx, jdx)` vanishes unless the two indices name the same example, and is that example's own
    map's entry otherwise. -/
theorem pdiv_batchMapIdx_at {N a b : Nat} (f : Fin N → Vec a → Vec b) (v : Vec (N * a))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ (f r) (Mat.unflatten v r))
    (idx : Fin (N * a)) (jdx : Fin (N * b)) :
    pdiv (batchMapIdx N f) v idx jdx =
      if (finProdFinEquiv.symm idx).1 = (finProdFinEquiv.symm jdx).1 then
        pdiv (f (finProdFinEquiv.symm jdx).1) (Mat.unflatten v (finProdFinEquiv.symm jdx).1)
          (finProdFinEquiv.symm idx).2 (finProdFinEquiv.symm jdx).2
      else 0 := by
  have h := pdivMat_rowIndep_perRow_at f (Mat.unflatten v) hf_diff
      (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
      (finProdFinEquiv.symm jdx).1 (finProdFinEquiv.symm jdx).2
  unfold pdivMat at h
  simp only [Mat.flatten_unflatten, Prod.mk.eta, Equiv.apply_symm_apply] at h
  exact h

/-- **`batchMapIdx N f`'s VJP at a point**: each example's row runs its own map's backward.
    Built field by field (as `batchMapHasVJPAt`) so `.backward` reduces. -/
noncomputable def batchMapIdxHasVJPAt {N a b : Nat} (f : Fin N → Vec a → Vec b) (v : Vec (N * a))
    (hf : ∀ r : Fin N, HasVJPAt (f r) (Mat.unflatten v r))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ (f r) (Mat.unflatten v r)) :
    HasVJPAt (batchMapIdx N f) v where
  backward := fun dy idx =>
    (hf (finProdFinEquiv.symm idx).1).backward
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2
  correct := by
    intro dy idx
    simp only [sum_finProdFinEquiv, pdiv_batchMapIdx_at f v hf_diff, Equiv.symm_apply_apply,
      ite_mul, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq,
      Finset.mem_univ, ite_true]
    exact (hf _).correct _ _

/-- The global VJP of `batchMapIdx N f`, every `f n` globally certified. -/
noncomputable def batchMapIdxHasVJP {N a b : Nat} (f : Fin N → Vec a → Vec b)
    (hf : ∀ n, HasVJP (f n)) (hf_diff : ∀ n, Differentiable ℝ (f n)) :
    HasVJP (batchMapIdx N f) where
  backward v := (batchMapIdxHasVJPAt f v (fun r => (hf r).toHasVJPAt _)
    (fun r => hf_diff r _)).backward
  correct v := (batchMapIdxHasVJPAt f v (fun r => (hf r).toHasVJPAt _)
    (fun r => hf_diff r _)).correct

/-- **A batched backward tie from the per-example ones, at an indexed family.** If `g n` is
    example `n`'s certified backward, `batchMapAuxIdx N g v` IS the lifted witness's backward. -/
theorem batchMapAuxIdx_eq_batchMapIdxHasVJPAt {N a b : Nat} (f : Fin N → Vec a → Vec b)
    (g : Fin N → Vec a → Vec b → Vec a) (v : Vec (N * a))
    (hf : ∀ r : Fin N, HasVJPAt (f r) (Mat.unflatten v r))
    (hf_diff : ∀ r : Fin N, DifferentiableAt ℝ (f r) (Mat.unflatten v r))
    (hg : ∀ r : Fin N, g r (Mat.unflatten v r) = (hf r).backward) :
    batchMapAuxIdx N g v = (batchMapIdxHasVJPAt f v hf hf_diff).backward := by
  funext dy idx
  exact congrFun (congrFun (hg _) _) _

-- ════════════════════════════════════════════════════════════════
-- § The loss gradient in a parameter shared by every example
-- ════════════════════════════════════════════════════════════════

/-- The batched parameterised op `θ ↦ batchMapIdx N (fun n => per n θ) r` is differentiable when
    each example's map is differentiable in the parameter. -/
theorem batchMapIdx_param_differentiableAt {P N a q : Nat} (per : Fin N → Vec P → Vec a → Vec q)
    (r : Vec (N * a)) (θ : Vec P) (hper : ∀ n y, DifferentiableAt ℝ (fun θ' => per n θ' y) θ) :
    DifferentiableAt ℝ (fun θ' => batchMapIdx N (fun n => per n θ') r) θ := by
  refine differentiableAt_pi.2 fun J => ?_
  exact differentiableAt_pi.1 (hper _ _) _

/-- **`HasGradAt.param_batchMap` at an indexed family**: the Jacobian split by example, example `n`
    differentiated through its own map. -/
theorem HasGradAt.param_batchMapIdx {P N a q : Nat} {G : Vec (N * q) → Vec 1}
    (per : Fin N → Vec P → Vec a → Vec q) (r : Vec (N * a)) {θ : Vec P} {dy : Vec (N * q)}
    (hG : HasGradAt G (batchMapIdx N (fun n => per n θ) r) dy)
    (hper : ∀ n y, DifferentiableAt ℝ (fun θ' => per n θ' y) θ) :
    HasGradAt (fun θ' => G (batchMapIdx N (fun n => per n θ') r)) θ
      (fun i => ∑ n : Fin N, ∑ j : Fin q,
          pdiv (fun θ' => per n θ' (batchSlice N a r n)) θ i j * batchSlice N q dy n j) := by
  have hP := hG.param (layer := fun θ' => batchMapIdx N (fun n => per n θ') r)
    (batchMapIdx_param_differentiableAt per r θ hper)
  refine ⟨hP.differentiableAt, fun i => ?_⟩
  rw [hP.pdiv_eq i]
  rw [← finProdFinEquiv.sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun n _ => Finset.sum_congr rfl fun j _ => ?_
  congr 1
  rw [pdiv_eq_fderiv_coord (batchMapIdx_param_differentiableAt per r θ hper),
    pdiv_eq_fderiv_coord (hper _ _)]
  simp only [batchMapIdx, Equiv.symm_apply_apply]
  rfl

/-- **`HasGradAt.param_batchMap_through` at an indexed family.** Example `n` runs
    `y ↦ post n y (per θ (pre n y))`: the shared parameterised op between example `n`'s own prefix
    and suffix (a drop scale on either side is example `n`'s mask entry). If, per example, the loss
    `⟨post n y ·, dy⟩` has gradient `cot n y dy` at the op's output, the batched node
    `Σ_n Σ_j ∂per/∂θ · cotₙ` — at any saved activation `A` and cotangent `COT` whose slices are
    `pre n yₙ` and `cot n yₙ dyₙ` — is the gradient in `θ` of the whole batched loss. -/
theorem HasGradAt.param_batchMapIdx_through {P N a b m q : Nat} {G : Vec (N * q) → Vec 1}
    (pre : Fin N → Vec a → Vec b) (per : Vec P → Vec b → Vec m)
    (post : Fin N → Vec a → Vec m → Vec q) (cot : Fin N → Vec a → Vec q → Vec m)
    (X : Vec (N * a)) {θ : Vec P} {dY : Vec (N * q)}
    (hG : HasGradAt G (batchMapIdx N (fun n y => post n y (per θ (pre n y))) X) dY)
    (hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ)
    (hpost : ∀ n y, Differentiable ℝ (post n y))
    (hcot : ∀ n y dy, HasGradAt (fun u => linLoss dy (post n y u)) (per θ (pre n y)) (cot n y dy))
    (A : Vec (N * b)) (COT : Vec (N * m))
    (hA : ∀ n, batchSlice N b A n = pre n (batchSlice N a X n))
    (hC : ∀ n, batchSlice N m COT n = cot n (batchSlice N a X n) (batchSlice N q dY n)) :
    HasGradAt (fun θ' => G (batchMapIdx N (fun n y => post n y (per θ' (pre n y))) X)) θ
      (fun i => ∑ n : Fin N, ∑ j : Fin m,
        pdiv (fun θ' => per θ' (batchSlice N b A n)) θ i j * batchSlice N m COT n j) := by
  have hP := hG.param_batchMapIdx (fun n θ' y => post n y (per θ' (pre n y))) X
    (fun n y => (hpost n y _).comp θ (hper (pre n y)))
  refine ⟨hP.differentiableAt, fun i => ?_⟩
  rw [hP.pdiv_eq i]
  refine Finset.sum_congr rfl fun n _ => ?_
  rw [hA, hC, ← ((hcot _ _ _).param
    (layer := fun θ' => per θ' (pre n (batchSlice N a X n))) (hper _)).2 i]
  exact (((hasGradAt_linLoss _ _).param
    (layer := fun θ' => post n (batchSlice N a X n) (per θ' (pre n (batchSlice N a X n))))
    ((hpost _ _ _).comp θ (hper _))).2 i).symm

-- ════════════════════════════════════════════════════════════════
-- § One example's drop site
-- ════════════════════════════════════════════════════════════════

/-- One entry through a drop site that may be absent: `none` passes it, `some a` scales it. -/
noncomputable def siteScale : Option ℝ → ℝ → ℝ
  | none => fun x => x
  | some a => fun x => a * x

@[simp] theorem siteScale_none (x : ℝ) : siteScale none x = x := rfl

@[simp] theorem siteScale_some (a x : ℝ) : siteScale (some a) x = a * x := rfl

/-- **One example's drop scale at a site that may be absent**, entry by entry: `none` is the
    identity, `some a` scales every entry by `a` — example `n`'s reading of `dropPathOpt`
    (`batchSlice_dropPathOpt`). Pointwise, so it reads the same on a flat vector and on a row of
    its matrix. -/
noncomputable def dropScalarOpt {k : Nat} (s : Option ℝ) (v : Vec k) : Vec k :=
  fun i => siteScale s (v i)

@[simp] theorem dropScalarOpt_none {k : Nat} : dropScalarOpt (k := k) none = id := rfl

@[simp] theorem dropScalarOpt_some {k : Nat} (a : ℝ) :
    dropScalarOpt (k := k) (some a) = fun v i => a * v i := rfl

theorem dropScalarOpt_differentiable {k : Nat} (s : Option ℝ) :
    Differentiable ℝ (dropScalarOpt (k := k) s) := by
  cases s with
  | none => exact differentiable_id
  | some a => exact layerScale_differentiable (fun _ => a)

/-- The site's VJP is the site itself, stated as the `backward` field (as `dropPathOptHasVJP`) so
    it unfolds at a symbolic site. -/
noncomputable def dropScalarOptHasVJP {k : Nat} (s : Option ℝ) : HasVJP (dropScalarOpt (k := k) s) where
  backward := fun _ dy => dropScalarOpt s dy
  correct := by
    intro x dy i
    cases s with
    | none => exact (identityHasVJP k).correct x dy i
    | some a => exact (layerScaleHasVJP (fun _ => a)).correct x dy i

/-- The site is linear in what flows through it. -/
theorem dropScalarOpt_smul {k : Nat} (s : Option ℝ) : IsHomog (dropScalarOpt (k := k) s) := by
  intro c v
  cases s with
  | none => rfl
  | some a =>
    funext i
    show a * (c * v i) = c * (a * v i)
    ring

/-- **Example `n`'s site** of a per-example mask that may be absent: its entry at `n`, or `none`.
    Named, not spelled `sd.map fun v => v n` at every use: a statement that spells the lambda
    twice gets two hygienic binders, and `extract_lets` then stops merging the two `let` chains
    (planning/droppath_tie.md §2). -/
def exampleSite {N : Nat} (sd : Option (Vec N)) (n : Fin N) : Option ℝ :=
  sd.map fun v => v n

@[simp] theorem exampleSite_none {N : Nat} (n : Fin N) : exampleSite (none : Option (Vec N)) n = none :=
  rfl

@[simp] theorem exampleSite_some {N : Nat} (s : Vec N) (n : Fin N) :
    exampleSite (some s) n = some (s n) := rfl

/-- **The batched site, read at one example**, is that example's scalar site at its mask entry. -/
theorem batchSlice_dropPathOpt {N k : Nat} (sd : Option (Vec N)) (x : Vec (N * k)) (n : Fin N) :
    batchSlice N k (dropPathOpt N k sd x) n
      = dropScalarOpt (exampleSite sd n) (batchSlice N k x n) := by
  cases sd with
  | none => rfl
  | some s =>
    funext i
    simp only [batchSlice, dropPathOpt_some, dropPath_apply, exampleSite_some, dropScalarOpt_some,
      Equiv.symm_apply_apply]

/-- **A residual with a drop site on its branch**, `v ↦ v + s ⊙ br v`: the skip's backward is the
    raw cotangent, the branch's is `br`'s at the dropped one (`siteResHasVJP_backward`, `rfl`). -/
noncomputable def siteResHasVJP {k : Nat} (s : Option ℝ) (br : Vec k → Vec k)
    (hd : Differentiable ℝ br) (hv : HasVJP br) :
    HasVJP (fun v i => v i + dropScalarOpt s (br v) i) :=
  biPathHasVJP (fun v => v) (dropScalarOpt s ∘ br) differentiable_id
    ((dropScalarOpt_differentiable s).comp hd) (identityHasVJP k)
    (vjpComp br (dropScalarOpt s) hd (dropScalarOpt_differentiable s) hv (dropScalarOptHasVJP s))

theorem siteResHasVJP_backward {k : Nat} (s : Option ℝ) (br : Vec k → Vec k)
    (hd : Differentiable ℝ br) (hv : HasVJP br) (x dy : Vec k) :
    (siteResHasVJP s br hd hv).backward x dy = fun i => dy i + hv.backward x (dropScalarOpt s dy) i :=
  rfl

theorem siteRes_differentiable {k : Nat} (s : Option ℝ) (br : Vec k → Vec k)
    (hd : Differentiable ℝ br) :
    Differentiable ℝ (fun v i => v i + dropScalarOpt s (br v) i) :=
  differentiable_id.add ((dropScalarOpt_differentiable s).comp hd)

-- ════════════════════════════════════════════════════════════════
-- § Sharding and homogeneity
-- ════════════════════════════════════════════════════════════════

namespace StableHLO

/-- **`batchMapIdx` commutes with sharding**: shard `r`'s family is the global family at the
    global indices of its examples. -/
theorem batchShard_batchMapIdx {R N a b : Nat} (f : Fin (R * N) → Vec a → Vec b)
    (X : Vec ((R * N) * a)) (r : Fin R) :
    batchShard R N b (batchMapIdx (R * N) f X) r
      = batchMapIdx N (fun n => f (finProdFinEquiv (r, n))) (batchShard R N a X r) := by
  funext idx
  simp only [batchShard, batchMapIdx, Equiv.symm_apply_apply]

/-- …and so does `batchMapAuxIdx`. -/
theorem batchShard_batchMapAuxIdx {R N s a b : Nat} (f : Fin (R * N) → Vec s → Vec a → Vec b)
    (aux : Vec ((R * N) * s)) (X : Vec ((R * N) * a)) (r : Fin R) :
    batchShard R N b (batchMapAuxIdx (R * N) f aux X) r
      = batchMapAuxIdx N (fun n => f (finProdFinEquiv (r, n)))
          (batchShard R N s aux r) (batchShard R N a X r) := by
  funext idx
  simp only [batchShard, batchMapAuxIdx, Equiv.symm_apply_apply, batchSlice_batchShard]

end StableHLO

/-- An indexed lift of homogeneous maps is homogeneous. -/
theorem batchMapIdx_smul {N a b : Nat} (f : Fin N → Vec a → Vec b) (hf : ∀ n, IsHomog (f n)) :
    IsHomog (batchMapIdx N f) := fun s X => by
  funext idx
  exact congrFun (hf _ s _) _

/-- …and so is an indexed auxiliary lift, homogeneous in its last argument. -/
theorem batchMapAuxIdx_smul {N t a b : Nat} (f : Fin N → Vec t → Vec a → Vec b)
    (hf : ∀ n x, IsHomog (f n x)) (aux : Vec (N * t)) : IsHomog (batchMapAuxIdx N f aux) :=
  fun s X => by
    funext idx
    exact congrFun (hf _ _ s _) _

end Proofs
