import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Architectures.Residual
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Architectures.LayerNorm

/-!
# Squeeze-and-Excitation — the "main × gate" pattern

The most interesting layer in this stack, structurally. It's the first
op where the output is the **product** of two functions of the same input:

    y = x ⊙ gate(x)

The "main path" is the input itself. The "gate" is a small subnetwork
that computes a per-channel scaling factor by squeezing spatial info
through a bottleneck. Each channel's output is multiplied by its own
gate value, so SE acts as **learned channel attention**.

## Why this matters for backprop

This is where we hit the **product rule** for the first time. Both
factors of `x ⊙ gate(x)` depend on `x`, so the gradient at `x` has
contributions from both:

  - Through the main path: the gate itself acts as a "stop-gradient
    multiplier" — the gradient flowing back through the main path is
    just `gate(x) ⊙ dy`.

  - Through the gate path: the input flows back through the entire gate
    sub-network, and the cotangent it sees is `x ⊙ dy` (not just `dy`,
    because the gate is multiplying the main path).

The two contributions add at `x`. This is the same fan-in pattern as
residual blocks (`Residual.lean`), but now driven by **multiplication**
rather than addition — and that changes which cotangents each path sees.

## Foreshadowing

The exact same structure shows up in Transformer attention:

    out = softmax(QKᵀ/√d) · V
        = attention_weights ⊙ V_with_some_extra_steps

The output is a product of "attention weights" (a function of Q and K)
and `V`. Backprop through attention is just the product rule applied
twice (once for each factor) plus the chain rule through softmax. SE
is the simplest non-trivial instance of this pattern; if you understand
it, attention is downhill.

## What this file provides

All foundational definitions and proofs live in `Tensor.lean`:
  - `elemwiseProduct f g` — pointwise product of two vector functions
  - `elemwiseProductHasVJP` — the bi-cotangent VJP (proved, no sorry)
  - `identityHasVJP` — identity backward is passthrough (proved)
  - `pdiv_mul` — product rule for partial derivatives (theorem)

This file specializes to the SE pattern: `f = identity`, `g = gate`.
`seBlockHasVJP` takes the gate abstractly (it only needs `HasVJP gate`).
The file then defines the concrete gate `seGate` (GAP → dense → swish →
dense → sigmoid → broadcast) with its VJP `seGateHasVJP`, the pieces it
needs (`sigmoidHasVJP`, `broadcastFlatHasVJP`), and the full block
`seBlockFull` / `seBlockFullHasVJP`.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § SE block: identity × gate
-- ════════════════════════════════════════════════════════════════

/-- An SE-style block, parameterized by an arbitrary "gate" sub-network.

    `seBlock gate x = x ⊙ gate(x)`

    The gate can be anything — for the actual SE we use
    `gate = sigmoid ∘ dense_exp ∘ swish ∘ dense_red ∘ globalAvgPool`,
    but the VJP derivation doesn't care about the gate's internals.
    All we need is `HasVJP gate`. -/
noncomputable def seBlock {n : Nat} (gate : Vec n → Vec n) : Vec n → Vec n :=
  elemwiseProduct (fun x => x) gate

/-- **SE block VJP** — direct application of the elemwise product formula.

    With `f = identity` (so `f(x) = x` and `f.back(x, dy) = dy`), the
    general formula simplifies to:

      back_SE(x, dy) = (gate(x) ⊙ dy)              -- main path: id backward
                     + gate.back(x, x ⊙ dy)        -- gate path

    First term: gradient flows back through the "main path" as
    `gate(x) ⊙ dy` — each channel scaled by its gate value.

    Second term: gradient flows back through the "gate sub-network",
    which sees `x ⊙ dy` as its cotangent (not just `dy`!). Inside the
    gate, `globalAvgPool` will broadcast this back over spatial dims,
    `dense_red` and `dense_exp` will do their usual VJPs, etc.

    The MLIR emits exactly this two-path backward: `gate(x) * dy` plus
    the gate's own backward chain with cotangent `x * dy`.

    **No sorry** — this delegates to `elemwiseProductHasVJP` and
    `identityHasVJP`, both proved in `Tensor.lean`. -/
noncomputable def seBlockHasVJP {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate) :
    HasVJP (seBlock gate) :=
  elemwiseProductHasVJP (fun x => x) gate
    differentiable_id hg_diff (identityHasVJP n) hg

-- ════════════════════════════════════════════════════════════════
-- § Sketching the concrete SE gate (the proved one is `seGate`, at the end of this file)
-- ════════════════════════════════════════════════════════════════

/-! ## What's actually inside `gate`

For the MobileNetV3 SE block (`MlirCodegen.emitSEBlock`), the gate is:

  1. **Squeeze**: Global average pool over (H, W) -> (B, C)
       `g[c] = (1/(H*W)) sum_{h,w} x[c, h, w]`

  2. **Reduce**: Dense `(C -> C/4)` (or similar bottleneck)
       `r = W_red * g + b_red`

  3. **Activation**: Swish `r ⊙ sigma(r)` (or ReLU in V3)

  4. **Expand**: Dense `(C/4 -> C)` back to per-channel
       `e = W_exp * sigma_swish(r) + b_exp`

  5. **Sigmoid gate** (or h-sigmoid in V3): `sigma(e)` — squashes each
     channel's "importance score" into [0, 1]

  6. **Broadcast** back to `(C, H, W)` so it can multiply the main path

So the `gate` is actually a *Vec-shaped* function that takes the spatial
input, summarizes it via GAP, runs it through a tiny FC network, and
broadcasts the per-channel result back to spatial.

`seGate` at the end of this file builds `gate` as the composition

    gate = broadcast ∘ sigmoid ∘ dense_exp ∘ swish ∘ dense_red ∘ globalAvgPool

and `seGateHasVJP` assembles its VJP with `vjpComp` from
`globalAvgPoolFlatHasVJP` (CNN.lean), `denseHasVJP`, `swishHasVJP`
(LayerNorm.lean), `sigmoidHasVJP` and `broadcastFlatHasVJP` (this file;
broadcast is the adjoint of GAP up to the `1/(h·w)` factor).

## Why this generalizes

Replace "gate" with "attention weights" and SE becomes the core of
self-attention:

    out = (sequence) ⊙ (per-token attention weights)

The structural pattern is identical: a main tensor multiplied by a
side-computed scalar (or vector) per element. The bi-cotangent rule
(`elemwiseProductHasVJP`) is the right tool for both. SE is the
on-ramp; once you've internalized this VJP shape, attention falls out
of the same theorem.
-/

/-- **Public correctness theorem for `seBlockHasVJP`**: the SE-block
backward (input × gate Jacobian via the product rule) equals the
`pdiv`-contracted Jacobian of `seBlock gate`. -/
theorem seBlockHasVJP_correct {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (seBlockHasVJP gate hg_diff hg).backward x dy i =
    ∑ j : Fin n, pdiv (seBlock gate) x i j * dy j :=
  (seBlockHasVJP gate hg_diff hg).correct x dy i

open Finset BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Sigmoid activation (smooth, logistic)
-- ════════════════════════════════════════════════════════════════

noncomputable def sigmoidScalar (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

noncomputable def sigmoid (n : Nat) (x : Vec n) : Vec n :=
  fun i => sigmoidScalar (x i)

noncomputable def sigmoidScalarDeriv (x : ℝ) : ℝ :=
  deriv sigmoidScalar x

/-- `sigmoidScalar` is Mathlib's logistic function `Real.sigmoid`. -/
theorem sigmoidScalar_eq_sigmoid : sigmoidScalar = Real.sigmoid := by
  funext x; simp [sigmoidScalar, Real.sigmoid]

/-- The closed form σ' = σ·(1 − σ). `sigmoidHasVJP`'s backward is stated with `deriv`; the
    emitted `sigmoidBack` text computes `σ(x)·(1 − σ(x))` — this is the equation between them,
    so it stays pinned in the axiom audit although no Lean proof consumes it. -/
theorem sigmoidScalarDeriv_eq (x : ℝ) :
    sigmoidScalarDeriv x = sigmoidScalar x * (1 - sigmoidScalar x) := by
  simp [sigmoidScalarDeriv, sigmoidScalar_eq_sigmoid, Real.deriv_sigmoid]

@[fun_prop]
lemma sigmoidScalar_differentiable : Differentiable ℝ sigmoidScalar := by
  rw [sigmoidScalar_eq_sigmoid]; exact differentiable_sigmoid

lemma sigmoid_differentiable (D : Nat) : Differentiable ℝ (sigmoid D) := by
  unfold sigmoid; fun_prop

theorem pdiv_sigmoid (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (sigmoid n) x i j =
    if i = j then sigmoidScalarDeriv (x i) else 0 :=
  pdiv_elementwise sigmoidScalar x (fun _ => sigmoidScalar_differentiable _) i j

noncomputable def sigmoidHasVJP (n : Nat) : HasVJP (sigmoid n) where
  backward := fun x dy i => dy i * sigmoidScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_sigmoid, mul_comm]

theorem sigmoidHasVJP_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (sigmoidHasVJP n).backward x dy i =
    ∑ j : Fin n, pdiv (sigmoid n) x i j * dy j :=
  (sigmoidHasVJP n).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Broadcast: per-channel scalar → spatial (adjoint of GAP)
-- ════════════════════════════════════════════════════════════════

/-- **Broadcast a per-channel vector back to spatial layout.**
    `broadcastFlat c h w v idx = v (flatChannel c h w idx)` — every spatial
    cell of channel `k` receives `v k`. This is the reindex map along
    `flatChannel`, i.e. the adjoint of `globalAvgPoolFlat` (up to the
    1/(h·w) scale). `Vec c → Vec (c*h*w)`. -/
noncomputable def broadcastFlat (c h w : Nat) : Vec c → Vec (c * h * w) :=
  fun v => fun idx => v (flatChannel c h w idx)

theorem broadcastFlat_differentiable (c h w : Nat) :
    Differentiable ℝ (broadcastFlat c h w) :=
  (reindexCLM (flatChannel c h w)).differentiable

/-- **Broadcast VJP** — linear reindex; backward sums each channel's
    spatial cotangents (the adjoint of broadcast = sum-over-spatial). -/
noncomputable def broadcastFlatHasVJP (c h w : Nat) :
    HasVJP (broadcastFlat c h w) where
  backward := fun _v dy => fun k =>
    ∑ idx : Fin (c * h * w), (if flatChannel c h w idx = k then dy idx else 0)
  correct := by
    intro v dy k
    show (∑ idx : Fin (c * h * w),
            (if flatChannel c h w idx = k then dy idx else 0)) =
      ∑ j : Fin (c * h * w), pdiv (broadcastFlat c h w) v k j * dy j
    have hpd : ∀ j : Fin (c * h * w),
        pdiv (broadcastFlat c h w) v k j =
          if k = flatChannel c h w j then 1 else 0 := by
      intro j
      exact pdiv_reindex (flatChannel c h w) v k j
    simp only [hpd, ite_mul, one_mul, zero_mul, @eq_comm _ k]

-- ════════════════════════════════════════════════════════════════
-- § SE gate: squeeze → reduce(swish) → expand → sigmoid → broadcast
-- ════════════════════════════════════════════════════════════════

/-- **The squeeze-excite gate.** Maps `Vec (c*h*w) → Vec (c*h*w)`:
      broadcast ∘ sigmoid ∘ dense(W₂,b₂) ∘ swish ∘ dense(W₁,b₁) ∘ GAP
    Squeeze (GAP `c·h·w → c`), reduce (dense `c → r`), swish, expand
    (dense `r → c`), sigmoid gate, broadcast back to spatial.  Every
    stage is smooth everywhere (swish/sigmoid smooth, dense/GAP/broadcast
    linear-affine), so the gate is differentiable everywhere and has a
    global `HasVJP`. -/
noncomputable def seGate {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  broadcastFlat c h w ∘ sigmoid c ∘ dense W₂ b₂ ∘ swish r ∘
    dense W₁ b₁ ∘ globalAvgPoolFlat c h w

@[fun_prop]
theorem seGate_differentiable {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Differentiable ℝ (seGate (h := h) (w := w) W₁ b₁ W₂ b₂) := by
  unfold seGate broadcastFlat sigmoid swish; fun_prop

noncomputable def seGateHasVJP {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    HasVJP (seGate (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  vjpComp _ (broadcastFlat c h w)
    ((sigmoid_differentiable c).comp
      ((dense_differentiable W₂ b₂).comp
        ((swish_differentiable r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w)))))
    (broadcastFlat_differentiable c h w)
    (vjpComp _ (sigmoid c)
      ((dense_differentiable W₂ b₂).comp
        ((swish_differentiable r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w))))
      (sigmoid_differentiable c)
      (vjpComp _ (dense W₂ b₂)
        ((swish_differentiable r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w)))
        (dense_differentiable W₂ b₂)
        (vjpComp _ (swish r)
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w))
          (swish_differentiable r)
          (vjpComp _ (dense W₁ b₁)
            (globalAvgPoolFlat_differentiable c h w)
            (dense_differentiable W₁ b₁)
            (globalAvgPoolFlatHasVJP c h w)
            (denseHasVJP W₁ b₁))
          (swishHasVJP r))
        (denseHasVJP W₂ b₂))
      (sigmoidHasVJP c))
    (broadcastFlatHasVJP c h w)

/-- **The full SE block** with the concrete gate: `x ⊙ seGate(x)`. -/
noncomputable def seBlockFull {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  seBlock (seGate (h := h) (w := w) W₁ b₁ W₂ b₂)

noncomputable def seBlockFullHasVJP {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    HasVJP (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  seBlockHasVJP (seGate (h := h) (w := w) W₁ b₁ W₂ b₂)
    (seGate_differentiable W₁ b₁ W₂ b₂)
    (seGateHasVJP W₁ b₁ W₂ b₂)

@[fun_prop]
theorem seBlockFull_differentiable {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Differentiable ℝ (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂) := by
  unfold seBlockFull seBlock; fun_prop

end Proofs
