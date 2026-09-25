import LeanMlir.Proofs.Foundation.Tensor

/-!
# Residual Connections — Gradient Accumulation

The first chapter where backprop has to **accumulate** gradients from
multiple paths into the same input. So far every layer has been a
straight-line composition (chain rule), but residual blocks introduce
fan-out: one input feeds two paths whose outputs are added.

The math is trivial — it's the *pattern* that matters. Once you see
"two backwards add", you'll see it everywhere: residuals, attention,
SE blocks, multi-head outputs, anywhere a tensor is consumed by more
than one downstream op.

This file builds on the proved foundations in `Tensor.lean`:
- `biPath f g` and `biPathHasVJP` (additive fan-in, proved)
- `identityHasVJP` (identity VJP, proved)
- `pdiv_add` and `pdiv_id` (calculus facts, proved from Mathlib's `fderiv`)

With those in hand, the residual definitions are one-liners — no sorry's.

1. Defines `residual f x = f x + x` via `biPath f id` and its VJP.
2. Defines `residualProj proj f x = proj x + f x` via `biPath proj f`
   and its VJP.
3. Comments on how this matches the ResNet skip connection in the
   MLIR (`MlirCodegen.lean` residual block emission).
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Residual block: y = f(x) + x
-- ════════════════════════════════════════════════════════════════

/-- A basic residual block: output = sub-network output + identity.

    `residual f x = f(x) + x`

    The "skip connection" lets gradients flow directly from output back
    to input without going through `f`. This is why ResNets train: even
    if `f` has near-zero gradients (vanishing), the identity path keeps
    the signal alive. -/
noncomputable def residual {n : Nat} (f : Vec n → Vec n) : Vec n → Vec n :=
  biPath f (fun x => x)

/-- **Residual VJP**: `dx = f.back(x, dy) + dy`.

    The skip's contribution is just `dy` (identity backward). The block's
    contribution is `f.back(x, dy)`. They add. This is **why** ResNets
    are easier to train: the gradient floor is `dy` itself, so it can
    never get smaller than the loss gradient at this layer.

    MLIR (`MlirCodegen.lean`, the residual-block case of the backward walk):
      The "skip grad" is added to the first convBn of the block — exactly
      `f.back(x, dy) + dy_skip`, where `dy_skip = dy` here.

    Proof: immediate from `biPathHasVJP` and `identityHasVJP`,
    both proved in `Tensor.lean`. -/
noncomputable def residualHasVJP {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f) :
    HasVJP (residual f) :=
  biPathHasVJP f (fun x => x) hf_diff differentiable_id hf (identityHasVJP n)

-- ════════════════════════════════════════════════════════════════
-- § Projected residual: y = proj(x) + f(x)
-- ════════════════════════════════════════════════════════════════

/-- Projected residual block: when input and output shapes don't match
    (e.g. when stride > 1 downsamples), the skip is not identity but a
    1×1 projection conv.

    `residualProj proj f x = proj(x) + f(x)`

    Both paths now have nontrivial backwards. The gradient still adds at
    the input — neither path is privileged. -/
noncomputable def residualProj {m n : Nat}
    (proj f : Vec m → Vec n) : Vec m → Vec n :=
  biPath proj f

/-- **Projected residual VJP**: `dx = proj.back(x, dy) + f.back(x, dy)`.

    Both backwards run on the same `dy` and their results sum at `x`.
    This is the truly general "fan-out → backward fan-in" pattern.

    MLIR: ResNets with stride > 1 use this — see `emitConvBnBackward`
    where the projection's VJP is emitted alongside the main block's,
    and both gradients accumulate into the same incoming-grad SSA.

    Proof: immediate from `biPathHasVJP`, proved in `Tensor.lean`. -/
noncomputable def residualProjHasVJP {m n : Nat}
    (proj f : Vec m → Vec n)
    (hproj_diff : Differentiable ℝ proj) (hf_diff : Differentiable ℝ f)
    (hproj : HasVJP proj) (hf : HasVJP f) :
    HasVJP (residualProj proj f) :=
  biPathHasVJP proj f hproj_diff hf_diff hproj hf

-- ════════════════════════════════════════════════════════════════
-- § The pattern, in plain English
-- ════════════════════════════════════════════════════════════════

/-! ## Why this matters beyond ResNets

The fan-out/backward-add pattern is the **structural building block** for
every modern architecture:

  • **ResNets** — `y = f(x) + x` (this file).
  • **DenseNets** — `y = concat(f(x), x)`. The concat splits dy and each
    half goes back through its respective path. Same pattern, different
    glue (split instead of add).
  • **Squeeze-and-Excitation** — `y = x · gate(x)`. The product rule
    introduces a different kind of bi-path: the gate's gradient gets
    `x ⊙ dy` and the main path's gradient gets `gate(x) ⊙ dy`. See
    `SE.lean` for that derivation.
  • **Multi-head attention** — concatenated heads. Same structure.
  • **Two-tower models** — independent encoders → joint loss. Even more
    extreme fan-out.

If you understand `biPathHasVJP`, you understand backprop through any
DAG. Composition (chain rule) handles the "linear" part; bi-path handles
the joins. Together they're enough for any computation graph.
-/

/-- **Public correctness theorem for `residualHasVJP`**: skip-connection
backward equals the `pdiv`-contracted Jacobian of `f + id`. -/
theorem residualHasVJP_correct {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (residualHasVJP f hf_diff hf).backward x dy i =
    ∑ j : Fin n, pdiv (residual f) x i j * dy j :=
  (residualHasVJP f hf_diff hf).correct x dy i

/-- **Public correctness theorem for `residualProjHasVJP`**: same as
`residualHasVJP_correct` but for the projected variant where the skip
isn't identity. -/
theorem residualProjHasVJP_correct {m n : Nat}
    (proj f : Vec m → Vec n)
    (hproj_diff : Differentiable ℝ proj) (hf_diff : Differentiable ℝ f)
    (hproj : HasVJP proj) (hf : HasVJP f)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (residualProjHasVJP proj f hproj_diff hf_diff hproj hf).backward x dy i =
    ∑ j : Fin n, pdiv (residualProj proj f) x i j * dy j :=
  (residualProjHasVJP proj f hproj_diff hf_diff hproj hf).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Smooth-point variants (for the CNN/ResNet composition)
-- ════════════════════════════════════════════════════════════════

/-! ResNet residual bodies contain ReLU, which is only `DifferentiableAt`
at smooth points (see `MLP.lean`), not `Differentiable` globally. So the
end-to-end CNN VJP (`cnnHasVJPAt`, future) must chain through the
pointwise `HasVJPAt` framework — `vjpCompAt` + the witnesses below —
exactly as `mlpHasVJPAt` does for the MLP. These mirror the everywhere
versions above, at a fixed `x`; `pdiv_add` is already stated at-point so
the proof is the everywhere one with `intro x` dropped. -/

/-- **Additive fan-in at a point** — smooth-point analog of `biPathHasVJP`. -/
noncomputable def biPathHasVJPAt {m n : Nat}
    (f g : Vec m → Vec n) (x : Vec m)
    (hf_diff : DifferentiableAt ℝ f x) (hg_diff : DifferentiableAt ℝ g x)
    (hf : HasVJPAt f x) (hg : HasVJPAt g x) :
    HasVJPAt (biPath f g) x where
  backward dy i := hf.backward dy i + hg.backward dy i
  correct := by
    intro dy i
    rw [hf.correct, hg.correct, ← Finset.sum_add_distrib]
    congr 1; ext j; rw [pdiv_add _ _ _ hf_diff hg_diff]; ring

/-- **Residual VJP at a point**: `dx = f.back(dy) + dy`. The skip
    (identity) is differentiable everywhere, so only `f` needs the
    smooth-point hypothesis. -/
noncomputable def residualHasVJPAt {n : Nat}
    (f : Vec n → Vec n) (x : Vec n)
    (hf_diff : DifferentiableAt ℝ f x) (hf : HasVJPAt f x) :
    HasVJPAt (residual f) x :=
  biPathHasVJPAt f (fun x => x) x
    hf_diff differentiable_id.differentiableAt
    hf ((identityHasVJP n).toHasVJPAt x)

theorem residual_apply {n : Nat} (f : Vec n → Vec n) (v : Vec n) (k : Fin n) :
    residual f v k = f v k + v k := rfl

theorem residual_differentiableAt {n : Nat} {f : Vec n → Vec n} {x : Vec n}
    (hf : DifferentiableAt ℝ f x) : DifferentiableAt ℝ (residual f) x :=
  hf.add differentiable_id.differentiableAt

/-- **Projected residual VJP at a point**: both paths carry smooth-point
    hypotheses (the 1×1 stride-2 projection is linear, but stated at-point
    for uniformity with the composition). -/
noncomputable def residualProjHasVJPAt {m n : Nat}
    (proj f : Vec m → Vec n) (x : Vec m)
    (hproj_diff : DifferentiableAt ℝ proj x) (hf_diff : DifferentiableAt ℝ f x)
    (hproj : HasVJPAt proj x) (hf : HasVJPAt f x) :
    HasVJPAt (residualProj proj f) x :=
  biPathHasVJPAt proj f x hproj_diff hf_diff hproj hf

/-- A residual branch is continuous when its body is. -/
@[fun_prop]
theorem residual_continuous {n : Nat} (F : Vec n → Vec n) (hF : Continuous F) :
    Continuous (residual F) :=
  continuous_pi (fun k => ((continuous_apply k).comp hF).add (continuous_apply k))

/-- A projected residual is continuous when both branches are. -/
@[fun_prop]
theorem residualProj_continuous {m n : Nat} (P F : Vec m → Vec n) (hP : Continuous P)
    (hF : Continuous F) : Continuous (residualProj P F) :=
  continuous_pi (fun k => ((continuous_apply k).comp hP).add ((continuous_apply k).comp hF))

end Proofs
