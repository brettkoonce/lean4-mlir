import LeanMlir.Proofs.Foundation.Tensor

/-! # Plain SGD and Nesterov-momentum steps over ℝ — the optimizer peers of `AdamStep`

The ℝ reference for the `cifar8_{sgd,mom}_train_step` family (handoff §2i). Coordinatewise over
`Vec`, mirroring the emitted StableHLO op-for-op so the faithfulness theorems in `StableHLO.lean`
are structural matches (`rfl`), exactly as `AdamStep.lean` is for the AdamW triple.

**Why these did not exist.** The kit already had a `*Sgd` op family, but those **fuse** the gradient
and the update and bake `lr` as a *literal* — so there was nothing to hand a scheduled optimizer, the
same shape of blocker §2a found for Adam ("the fusion `θ − lr·g`, not Adam, was the actual blocker").
The renders that need these take `%lr` as a runtime `tensor<f32>` argument so one graph serves a
whole LR schedule.

**Claim ceiling.** Like `AdamStep`, the verified target here is **faithfulness** — the rendered
update denotes these functions — *not* a loss-decrease bound. Plain SGD does have descent results in
this repo (`Proofs/Training/SgdDescent*`), but they are about a specific net's loss under smoothness
hypotheses; nothing below claims Nesterov descends. Say "the momentum render is certified", never
"momentum is proven to descend".

The velocity convention matches the emitter it replaces (`tests/TestCifar8AdamTrain.emitMomentum`):
the **`v` slot carries the velocity** and the `m` slot is an untouched passthrough, so the packed
`[θ|m|v]` signature is shared verbatim with the AdamW render. -/

namespace Proofs

variable {n : Nat}

/-- **Plain SGD parameter update**, coordinatewise: `θ' = θ − lr·g`.

    `lr` is an ℝ here and a `tensor<f32>` function argument in the render — the op carries both, the
    name for emission and the value for denotation, exactly as `adamWParamF` does. -/
def sgdParam (lr : ℝ) (θ g : Vec n) : Vec n :=
  fun i => θ i - lr * g i

/-- **Nesterov velocity update**: `v' = μ·v + g`.

    Note the gradient enters undamped (coefficient 1, not `1 − μ`) — that is the convention the
    emitter uses, and it is the one PyTorch's `SGD(momentum=μ, dampening=0)` implements. -/
def momVNext (μ : ℝ) (v g : Vec n) : Vec n :=
  fun i => μ * v i + g i

/-- **Nesterov parameter update**: `θ' = θ − lr·(g + μ·v')`, where `v'` is the *updated* velocity.

    The `g + μ·v'` look-ahead is what makes this Nesterov rather than heavy-ball; plain momentum
    would step by `v'` alone (see `momParam_heavyBall_diff`). -/
def momParam (μ lr : ℝ) (θ v g : Vec n) : Vec n :=
  fun i => θ i - lr * (g i + μ * momVNext μ v g i)

/-- One Nesterov step: the new parameter together with the new velocity — the pair the rendered
    train step returns per parameter (`m` rides through unchanged). -/
def momStep (μ lr : ℝ) (θ v g : Vec n) : Vec n × Vec n :=
  (momParam μ lr θ v g, momVNext μ v g)

/-- **`μ = 0` collapses Nesterov to plain SGD.** The bridge between the two renders, and a cheap
    check that the momentum formula has no stray term: at zero momentum `v' = g` and the look-ahead
    `g + 0·v'` is just `g`. -/
theorem momParam_mu_zero (lr : ℝ) (θ v g : Vec n) :
    momParam 0 lr θ v g = sgdParam lr θ g := by
  funext i; simp [momParam, sgdParam, momVNext]

/-- **`lr = 0` freezes the parameter** — the determinism probe's precondition, stated so a render
    can be checked to move nothing at zero learning rate while still updating the velocity. -/
theorem sgdParam_lr_zero (θ g : Vec n) : sgdParam 0 θ g = θ := by
  funext i; simp [sgdParam]

/-- **Nesterov is heavy-ball plus one extra `μ·v'` of look-ahead.** Stated to pin which variant is
    rendered: a heavy-ball emitter would step by `lr·v'`, and this is the exact difference. -/
theorem momParam_heavyBall_diff (μ lr : ℝ) (θ v g : Vec n) (i : Fin n) :
    momParam μ lr θ v g i = (θ i - lr * momVNext μ v g i) - lr * (μ * momVNext μ v g i - μ * v i) := by
  simp [momParam, momVNext]; ring

/-- **Velocity is affine in the gradient** — the property the shard/DP gates rely on when they use
    a moment slot as a gradient proxy: at `v = 0`, `v' = g` exactly, so averaging velocities across
    replicas is averaging gradients. (`adamMNextF`'s analogue is `m' = (1−β₁)·g`.) -/
theorem momVNext_v_zero (μ : ℝ) (g : Vec n) :
    momVNext μ (fun _ => 0) g = g := by
  funext i; simp [momVNext]

end Proofs
