import LeanMlir.Proofs.Architectures.PerChannelBN

/-! # The batched index — per-example lifts and true batch-norm on `Vec (N·a)`

A batch of `N` examples is laid out row-major `[N, a]`: example `n` is the `finProdFinEquiv` block
`{(n, ·)}`. `batchMap N f` lifts a per-example map across the batch, `batchMapAux` lifts one that
also reads each example's own saved value, and `batchSlice N a v n` reads example `n` back.
`bnBatchLA` is the one batch-coupled op: true batch-norm (`bnBatchTensor4`) at the network's
left-associated `N·(c·h·w)` index. None of this mentions the IR; `StableHLO`'s batched nodes
denote these. The namespace is `StableHLO` for history, kept so every citation keeps its name.
-/

namespace Proofs
namespace StableHLO

/-- **Per-example block-apply.** Lift a per-example map `f : Vec a → Vec b` to a
    batch of `N` examples laid out row-major `[N, a] ↦ [N, b]` (the network's
    `[N,C,H,W]`-style flattening): example `n` occupies the `finProdFinEquiv`
    block `{(n, ·)}`. Every spatial/channel op in EfficientNet is batch-separable
    and lifts this way; only true batch-norm (`bnBatchTensor4`) couples the batch. -/
noncomputable def batchMap (N : Nat) {a b : Nat} (f : Vec a → Vec b) :
    Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f (fun i : Fin a => x (finProdFinEquiv (p.1, i))) p.2

/-- The `n`-th example's slice of a batch laid out row-major `[N, a]`. A shared
    weight's batched gradient is the sum over `n` of the per-example gradient on
    `batchSlice n` — the form the batched param-SGD dens take (so a batched parameter fold
    closes via the per-example cert + sum-linearity). -/
def batchSlice (N a : Nat) (v : Vec (N * a)) (n : Fin N) : Vec a :=
  fun i => v (finProdFinEquiv (n, i))

/-- `batchSlice` of a `batchMap` is the lifted function at the slice — the lemma that peels a
    per-example lift back off at one example. -/
theorem batchSlice_batchMap {N a b : Nat} (f : Vec a → Vec b) (x : Vec (N * a))
    (n : Fin N) :
    batchSlice N b (batchMap N f x) n = f (batchSlice N a x n) :=
  congrFun (Mat.unflatten_flatten fun n => f (batchSlice N a x n)) n

/-- **Per-example block-apply with per-example AUXILIARY data.** `batchMap` lifts one *fixed*
    function across the batch; this lifts a family indexed by each example's own saved value —
    example `n` is handed `batchSlice n aux`, not the whole `aux` and not example 0's.

    Every batched backward that recomputes from a saved forward activation has this shape, and
    that is exactly why such ops cannot be `BatchableOp` descriptors: a descriptor's
    `batchMap N (denOp op)` would apply ONE example's saved value to all `N`. Cf. `swishBackB`,
    `sigmoidBackB`, `selectPosB` (pointwise, so they take the whole-batch `x` directly) and
    `seBackBatched` (which inlines this shape). -/
noncomputable def batchMapAux (N : Nat) {s a b : Nat} (f : Vec s → Vec a → Vec b)
    (aux : Vec (N * s)) : Vec (N * a) → Vec (N * b) :=
  fun x idx =>
    let p := finProdFinEquiv.symm idx
    f (batchSlice N s aux p.1) (batchSlice N a x p.1) p.2

/-- **True batch-norm at the network's left-assoc `[N,C,H,W]` flat index.** The
    proven `bnBatchTensor4` (typed at `N·(oc·(h·w))`) conjugated by the `mul_assoc`
    reindex so it slots into the `N·(oc·h·w)` batched composition (where conv/etc.
    produce `oc·h·w = (oc·h)·w`). Reindex only — the function IS `bnBatchTensor4`. -/
noncomputable def bnBatchLA (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (oc * h * w)) → Vec (N * (oc * h * w)) :=
  fun v =>
    (fun y => y ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))
      (bnBatchTensor4 N oc h w ε γ β
        (v ∘ Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm))

/-- **Pointwise maps are `batchMap`-free.** Lifting an elementwise map across `N` examples IS the
    elementwise map at the batched index `N·n`. This is why moving the pointwise nodes onto
    descriptors was denotation-preserving, and it is the half of that claim the artifact cannot
    witness: the render is value-independent, so a descriptor with the wrong `den` emits the same
    bytes. Cf. `swishBackB`/`sigmoidBackB`, which are NOT descriptors precisely because their
    backward is not of this shape — it reads a per-example saved activation. -/
theorem batchMap_pointwise {N n : Nat} (g : ℝ → ℝ) (v : Vec (N * n)) :
    batchMap N (fun (x : Vec n) i => g (x i)) v = fun idx => g (v idx) :=
  congrArg (fun w idx => g (w idx)) (Mat.flatten_unflatten v)

end StableHLO

/-- `batchMap` of a continuous per-example op is continuous. -/
@[fun_prop]
theorem batchMap_continuous {N a b : Nat} (f : Vec a → Vec b) (hf : Continuous f) :
    Continuous (StableHLO.batchMap N f) := by
  refine continuous_pi (fun k => ?_)
  exact ((continuous_apply _).comp hf).comp (continuous_pi (fun _ => continuous_apply _))

end Proofs
