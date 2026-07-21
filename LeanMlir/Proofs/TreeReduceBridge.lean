import LeanMlir.Proofs.Float.FloatBridge

/-!
# Balanced-tree reduction bound (P2 — the tree-reduction quarantine)

`FloatModel.dot_close` / `sum_close` bound a length-`n` reduction by the
**association-independent** Higham factor `((1+u)^(n+1) − 1)·Σ|·|` — sound for
*every* evaluation order, but paying `n·u` because the worst (sequential) order
threads the seed through `n` additions.  Real hardware does not sum
sequentially: IREE lowers a StableHLO `reduce` (and the contraction of a
`dot_general`) as a **balanced reduction tree**, so each summand passes through
only `⌈log₂ n⌉` additions, not `n`.

This file proves the tree bound as pure ℝ arithmetic on the SAME `FloatModel`:
for any binary evaluation tree `t`,

    |evalTree t − (exact sum of leaves)| ≤ ((1+u)^(depth t) − 1)·(Σ |leaves|)

(`tree_close`), and its leaf-rounded twin `((1+u)^(depth t + 1) − 1)·Σ|·|`
(`dot_tree_close`, the dot version — round each product, then tree-sum).  The
bound is a function of the tree's *depth*, and holds for EVERY tree shape — it is
unconditional, zero project axioms, closure `[propext, Classical.choice,
Quot.sound]`.

The `n·u → log₂n·u` improvement then rests on ONE quarantined, empirically
validated fact — supplied as a named hypothesis at the application site, exactly
like `esig`/`egelu`, never an axiom:

    the deployed reduce of `n` elements evaluates as some tree `t` of
    depth ≤ ⌈log₂ n⌉      (`t.depth ≤ Nat.clog 2 n`)

`tree_close_of_depth` is the ready-to-apply form taking that `d = ⌈log₂ n⌉`
directly.  Validated by `scripts/kernel_faithfulness_probe.py`, which measures
real gfx1100 kernels sitting 20–10⁴× *inside* even the sequential bound.

Numerically (probe §10–§12 with the tree factor): fan-in 6272 ⇒ `n+1` = 6273
falls to `⌈log₂6272⌉` = 13; ConvNeXt's n = 301056 LN reduction falls to 19 —
turning the two "impossible" faces and every conv/dense fresh budget by a factor
`~n/log₂n`.
-/

namespace Proofs

/-- A binary evaluation tree over real leaves — the shape of a reduction's
    summation order.  `node l r` sums the two subtree results with one rounded
    add; a balanced tree over `n` leaves has `depth = ⌈log₂ n⌉`. -/
inductive SumTree where
  | leaf : ℝ → SumTree
  | node : SumTree → SumTree → SumTree

namespace SumTree

/-- The exact real value of the tree — associativity of ℝ makes this the plain
    sum of the leaves, independent of the tree shape. -/
def eval : SumTree → ℝ
  | leaf x => x
  | node l r => l.eval + r.eval

/-- `Σ |leaves|` — the magnitude the Higham bound multiplies. -/
def sumAbs : SumTree → ℝ
  | leaf x => |x|
  | node l r => l.sumAbs + r.sumAbs

/-- Tree depth: `leaf` = 0, `node` = 1 + max of children.  A summand at depth
    `d` passes through `d` rounded additions. -/
def depth : SumTree → ℕ
  | leaf _ => 0
  | node l r => 1 + max l.depth r.depth

/-- Number of leaves (= number of summands). -/
def size : SumTree → ℕ
  | leaf _ => 1
  | node l r => l.size + r.size

theorem one_le_size (t : SumTree) : 1 ≤ t.size := by
  induction t with
  | leaf x => simp [size]
  | node l r ihl ihr => simp only [size]; omega

theorem sumAbs_nonneg (t : SumTree) : 0 ≤ t.sumAbs := by
  induction t with
  | leaf x => exact abs_nonneg x
  | node l r ihl ihr => simp only [sumAbs]; linarith

theorem eval_abs_le (t : SumTree) : |t.eval| ≤ t.sumAbs := by
  induction t with
  | leaf x => simp [eval, sumAbs]
  | node l r ihl ihr =>
    simp only [eval, sumAbs]
    calc |l.eval + r.eval| ≤ |l.eval| + |r.eval| := abs_add_le _ _
      _ ≤ l.sumAbs + r.sumAbs := by linarith

end SumTree

namespace FloatModel

variable (M : FloatModel)

/-- Float evaluation of a `SumTree` with a per-leaf map `leafF` (identity for an
    exact leaf, `M.rnd` for a rounded product) and one rounded `add` at each
    node.  This models the deployed balanced reduction. -/
noncomputable def evalTreeWith (leafF : ℝ → ℝ) : SumTree → ℝ
  | .leaf x => leafF x
  | .node l r => M.add (evalTreeWith leafF l) (evalTreeWith leafF r)

/-- **The tree Higham bound, general form.**  If every leaf is evaluated within
    the budget `((1+u)^c − 1)·|x|` (c additional roundings per leaf), the whole
    balanced-tree reduction is within `((1+u)^(depth+c) − 1)·Σ|leaves|` —
    depth-LINEAR in the number of additions each summand actually sees, not the
    element count.  Same telescoping algebra as `dot_close`'s `step_bound`, over
    a tree instead of a left fold. -/
theorem tree_close_gen (leafF : ℝ → ℝ) (c : ℕ)
    (hleaf : ∀ x : ℝ, |leafF x - x| ≤ ((1 + M.u) ^ c - 1) * |x|)
    (t : SumTree) :
    |M.evalTreeWith leafF t - t.eval|
      ≤ ((1 + M.u) ^ (t.depth + c) - 1) * t.sumAbs := by
  have hp1 : (1 : ℝ) ≤ 1 + M.u := by have := M.u_nonneg; linarith
  have hu0 := M.u_nonneg
  induction t with
  | leaf x =>
    simpa [evalTreeWith, SumTree.eval, SumTree.depth, SumTree.sumAbs] using hleaf x
  | node l r ihl ihr =>
    -- unfold the node
    have hev : M.evalTreeWith leafF (SumTree.node l r)
        = M.rnd (M.evalTreeWith leafF l + M.evalTreeWith leafF r) := rfl
    set FL := M.evalTreeWith leafF l with hFLdef
    set FR := M.evalTreeWith leafF r with hFRdef
    have hAL := l.sumAbs_nonneg
    have hAR := r.sumAbs_nonneg
    have hSL := l.eval_abs_le
    have hSR := r.eval_abs_le
    -- magnitude of each float subtree: |FL| ≤ (1+u)^(dL+c)·AL
    have hFLm : |FL| ≤ (1 + M.u) ^ (l.depth + c) * l.sumAbs := by
      have h1 : |FL| ≤ |FL - l.eval| + |l.eval| := by
        simpa using abs_sub_le FL l.eval 0
      have : |FL| ≤ ((1 + M.u) ^ (l.depth + c) - 1) * l.sumAbs + l.sumAbs := by
        linarith [ihl, hSL]
      calc |FL| ≤ ((1 + M.u) ^ (l.depth + c) - 1) * l.sumAbs + l.sumAbs := this
        _ = (1 + M.u) ^ (l.depth + c) * l.sumAbs := by ring
    have hFRm : |FR| ≤ (1 + M.u) ^ (r.depth + c) * r.sumAbs := by
      have h1 : |FR| ≤ |FR - r.eval| + |r.eval| := by
        simpa using abs_sub_le FR r.eval 0
      have : |FR| ≤ ((1 + M.u) ^ (r.depth + c) - 1) * r.sumAbs + r.sumAbs := by
        linarith [ihr, hSR]
      calc |FR| ≤ ((1 + M.u) ^ (r.depth + c) - 1) * r.sumAbs + r.sumAbs := this
        _ = (1 + M.u) ^ (r.depth + c) * r.sumAbs := by ring
    -- one rounded add
    have hsum : |FL + FR|
        ≤ (1 + M.u) ^ (l.depth + c) * l.sumAbs
          + (1 + M.u) ^ (r.depth + c) * r.sumAbs :=
      (abs_add_le _ _).trans (by linarith [hFLm, hFRm])
    have humul : M.u * |FL + FR|
        ≤ M.u * ((1 + M.u) ^ (l.depth + c) * l.sumAbs)
          + M.u * ((1 + M.u) ^ (r.depth + c) * r.sumAbs) := by
      calc M.u * |FL + FR|
          ≤ M.u * ((1 + M.u) ^ (l.depth + c) * l.sumAbs
              + (1 + M.u) ^ (r.depth + c) * r.sumAbs) :=
            mul_le_mul_of_nonneg_left hsum hu0
        _ = M.u * ((1 + M.u) ^ (l.depth + c) * l.sumAbs)
              + M.u * ((1 + M.u) ^ (r.depth + c) * r.sumAbs) := by ring
    have hadd : |M.rnd (FL + FR) - (l.eval + r.eval)|
        ≤ M.u * |FL + FR| + (|FL - l.eval| + |FR - r.eval|) := by
      have t1 : |M.rnd (FL + FR) - (l.eval + r.eval)|
          ≤ |M.rnd (FL + FR) - (FL + FR)| + |(FL + FR) - (l.eval + r.eval)| :=
        abs_sub_le _ _ _
      have t2 : |(FL + FR) - (l.eval + r.eval)|
          ≤ |FL - l.eval| + |FR - r.eval| := by
        have he : (FL + FR) - (l.eval + r.eval)
            = (FL - l.eval) + (FR - r.eval) := by ring
        rw [he]; exact abs_add_le _ _
      have t3 : |M.rnd (FL + FR) - (FL + FR)| ≤ M.u * |FL + FR| := M.err _
      linarith
    -- collect to (p^(dL+c+1)−1)AL + (p^(dR+c+1)−1)AR
    have e1 : (1 + M.u) ^ (l.depth + c + 1) - 1
        = M.u * (1 + M.u) ^ (l.depth + c) + ((1 + M.u) ^ (l.depth + c) - 1) := by
      rw [pow_succ]; ring
    have e2 : (1 + M.u) ^ (r.depth + c + 1) - 1
        = M.u * (1 + M.u) ^ (r.depth + c) + ((1 + M.u) ^ (r.depth + c) - 1) := by
      rw [pow_succ]; ring
    have key : |M.evalTreeWith leafF (SumTree.node l r) - (l.eval + r.eval)|
        ≤ ((1 + M.u) ^ (l.depth + c + 1) - 1) * l.sumAbs
          + ((1 + M.u) ^ (r.depth + c + 1) - 1) * r.sumAbs := by
      rw [hev]
      have expand :
          ((1 + M.u) ^ (l.depth + c + 1) - 1) * l.sumAbs
            + ((1 + M.u) ^ (r.depth + c + 1) - 1) * r.sumAbs
          = (M.u * ((1 + M.u) ^ (l.depth + c) * l.sumAbs)
              + ((1 + M.u) ^ (l.depth + c) - 1) * l.sumAbs)
            + (M.u * ((1 + M.u) ^ (r.depth + c) * r.sumAbs)
              + ((1 + M.u) ^ (r.depth + c) - 1) * r.sumAbs) := by
        rw [e1, e2]; ring
      rw [expand]
      linarith [hadd, humul, ihl, ihr]
    -- monotone in depth: dL+c+1, dR+c+1 ≤ depth(node)+c = 1+max+c
    have mL : (1 + M.u) ^ (l.depth + c + 1)
        ≤ (1 + M.u) ^ ((SumTree.node l r).depth + c) :=
      pow_le_pow_right₀ hp1 (by simp only [SumTree.depth]; omega)
    have mR : (1 + M.u) ^ (r.depth + c + 1)
        ≤ (1 + M.u) ^ ((SumTree.node l r).depth + c) :=
      pow_le_pow_right₀ hp1 (by simp only [SumTree.depth]; omega)
    have hsa : (SumTree.node l r).sumAbs = l.sumAbs + r.sumAbs := rfl
    have fin :
        ((1 + M.u) ^ (l.depth + c + 1) - 1) * l.sumAbs
          + ((1 + M.u) ^ (r.depth + c + 1) - 1) * r.sumAbs
        ≤ ((1 + M.u) ^ ((SumTree.node l r).depth + c) - 1)
            * (SumTree.node l r).sumAbs := by
      rw [hsa]
      have c1 : ((1 + M.u) ^ (l.depth + c + 1) - 1) * l.sumAbs
          ≤ ((1 + M.u) ^ ((SumTree.node l r).depth + c) - 1) * l.sumAbs :=
        mul_le_mul_of_nonneg_right (by linarith [mL]) hAL
      have c2 : ((1 + M.u) ^ (r.depth + c + 1) - 1) * r.sumAbs
          ≤ ((1 + M.u) ^ ((SumTree.node l r).depth + c) - 1) * r.sumAbs :=
        mul_le_mul_of_nonneg_right (by linarith [mR]) hAR
      nlinarith [c1, c2]
    -- `(node l r).eval` reduces to `l.eval + r.eval`
    show |M.evalTreeWith leafF (SumTree.node l r) - (l.eval + r.eval)| ≤ _
    linarith [key, fin]

/-- **Pure reduction tree** (exact leaves — the `reduce`/BN-mean/var/gap/softmax
    case): `|evalTree t − Σ leaves| ≤ ((1+u)^(depth t) − 1)·Σ|leaves|`.  Contrast
    `sum_close`'s `(1+u)^(n+1)`: a balanced `t` has `depth = ⌈log₂ n⌉ ≪ n`. -/
theorem tree_close (t : SumTree) :
    |M.evalTreeWith id t - t.eval| ≤ ((1 + M.u) ^ t.depth - 1) * t.sumAbs := by
  have h := M.tree_close_gen id 0 (fun x => by simp) t
  simpa using h

/-- **Dot / weighted-reduction tree** (one leaf rounding for the product, then
    the balanced add tree): `((1+u)^(depth t + 1) − 1)·Σ|leaves|`.  With leaves
    the exact products `xᵢyᵢ`, `evalTreeWith M.rnd t` is the deployed
    `dot_general` contraction; contrast `dot_close`'s `(1+u)^(n+1)`. -/
theorem dot_tree_close (t : SumTree) :
    |M.evalTreeWith M.rnd t - t.eval|
      ≤ ((1 + M.u) ^ (t.depth + 1) - 1) * t.sumAbs := by
  have hleaf : ∀ x : ℝ, |M.rnd x - x| ≤ ((1 + M.u) ^ 1 - 1) * |x| := by
    intro x; have := M.err x; simpa using this
  exact M.tree_close_gen M.rnd 1 hleaf t

/-- **Ready-to-apply quarantine form.**  If the deployed reduce evaluates as a
    tree of depth ≤ `d` (the trusted balance fact — supply `d = Nat.clog 2 n` at
    the site, provenance `kernel_faithfulness_probe`), the reduction error is
    within `((1+u)^d − 1)·Σ|·|`.  Turning `n·u` into `log₂n·u` is exactly
    `d = ⌈log₂ n⌉` here. -/
theorem tree_close_of_depth (t : SumTree) (d : ℕ) (hd : t.depth ≤ d) :
    |M.evalTreeWith id t - t.eval| ≤ ((1 + M.u) ^ d - 1) * t.sumAbs := by
  refine (M.tree_close t).trans ?_
  have hp1 : (1 : ℝ) ≤ 1 + M.u := by have := M.u_nonneg; linarith
  have hmono : (1 + M.u) ^ t.depth ≤ (1 + M.u) ^ d := pow_le_pow_right₀ hp1 hd
  have hsa := t.sumAbs_nonneg
  nlinarith [hmono, hsa]

/-- The tree bound never loses to the sequential `sum_close`: any tree over `n`
    leaves has `depth ≤ n`, so `tree_close`'s factor is ≤ `sum_close`'s
    `(1+u)^(depth) ≤ (1+u)^(n) < (1+u)^(n+1)`.  (The strict win is the balanced
    `depth = ⌈log₂ n⌉`; this only records "never worse".) -/
theorem depth_le_size (t : SumTree) : t.depth ≤ t.size := by
  induction t with
  | leaf x => simp [SumTree.depth, SumTree.size]
  | node l r ihl ihr =>
    have hl : 1 ≤ l.size := l.one_le_size
    have hr : 1 ≤ r.size := r.one_le_size
    simp only [SumTree.depth, SumTree.size]
    omega

end FloatModel

end Proofs
