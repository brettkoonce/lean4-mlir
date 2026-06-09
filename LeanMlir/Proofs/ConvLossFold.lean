import LeanMlir.Proofs.MobileNetV2Close

/-! # The cotangent pass — folding the certified per-layer Jacobian to `∂loss/∂θ`

The Item C/D closes certify each conv/depthwise param output as `θ − lr·(certified ∂(layer)/∂θ · c)`
for the cotangent `c` the backward chain delivers at that layer's output. What ties that to genuine
gradient descent on the **loss** is the *fold*: the single gradient of the whole loss wrt `θ` equals
exactly that certified Jacobian contracted with `∂loss/∂(layer output)`. This is the conv/depthwise
analogue of `mlp_hidden_total_loss_grad` (MlpTrainStep.lean) — the `pdiv_comp` chain rule applied to
`loss-of-θ = (downstream loss) ∘ (the θ-weight-map)`, differentiable at a smooth point.

The inner factor `pdiv G (layer output) k 0 = ∂loss/∂(layer output)_k` **IS** the cotangent the
backward chain delivers at the layer output — abstractly. The Item D `*Cot*` defs render that exact
cotangent concretely (per-op faithfulness composed through the downstream). So composing this fold
with Item D's pin closes the loop: `θⁿ = θ − lr·∂loss/∂θ`, for any conv/depthwise param, at a smooth
point. The smooth-point hypothesis is bundled honestly as "the downstream loss `G` is differentiable
at the layer output" (the relu6/BN smoothness bundle, exactly as `mobilenetv2_has_vjp_at` carries it).

These folds are **program-wide** — generic in the downstream `G`, so one theorem covers every conv (in
CNN / CIFAR / MobileNetV2 / ResNet-34) and one covers every depthwise (MobileNetV2 / EfficientNet …).
3-axiom clean.
-/

namespace Proofs

open scoped BigOperators

/-- **Conv-layer total-loss fold.** The single gradient of the whole loss wrt a conv kernel `W`
    (flattened) equals the certified `∂conv/∂W` contracted with `∂loss/∂(conv output)` — the chain
    rule (`pdiv_comp`) on `G ∘ (conv weight-map)`, at any point where the downstream loss `G` is
    differentiable at the conv output (the smoothness bundle). The conv analogue of
    `mlp_hidden_total_loss_grad`; generic in `G`, so it covers every conv layer. The inner factor
    `pdiv G (conv output) k 0` is the cotangent the backward chain delivers there (Item C/D's `c`). -/
theorem conv_total_loss_grad_fold {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic kH kW)
    (G : Vec (oc * h * w) → Vec 1)
    (hG : DifferentiableAt ℝ G (Tensor3.flatten (conv2d W b x)))
    (idx : Fin (oc * ic * kH * kW)) :
    pdiv (fun v : Vec (oc * ic * kH * kW) =>
            G (Tensor3.flatten (conv2d (Kernel4.unflatten v) b x))) (Kernel4.flatten W) idx 0
      = ∑ k : Fin (oc * h * w),
          pdiv (fun v : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)) (Kernel4.flatten W) idx k
            * pdiv G (Tensor3.flatten (conv2d W b x)) k 0 := by
  rw [show (fun v : Vec (oc * ic * kH * kW) =>
              G (Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)))
        = G ∘ (fun v : Vec (oc * ic * kH * kW) =>
                Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)) from rfl,
      pdiv_comp _ G (Kernel4.flatten W) ((conv2d_weight_differentiable b x).differentiableAt)
        (by simp only [Kernel4.unflatten_flatten]; exact hG) idx 0]
  simp only [Kernel4.unflatten_flatten]

/-- **Conv bias total-loss fold.** Same fold for the conv bias `b` (the bias-map is `b ↦ flatten(conv
    W b x)`, differentiable everywhere). -/
theorem conv_bias_total_loss_grad_fold {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b : Vec oc)
    (G : Vec (oc * h * w) → Vec 1)
    (hG : DifferentiableAt ℝ G (Tensor3.flatten (conv2d W b x))) (o : Fin oc) :
    pdiv (fun b' : Vec oc => G (Tensor3.flatten (conv2d W b' x))) b o 0
      = ∑ k : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o k
            * pdiv G (Tensor3.flatten (conv2d W b x)) k 0 := by
  rw [show (fun b' : Vec oc => G (Tensor3.flatten (conv2d W b' x)))
        = G ∘ (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) from rfl,
      pdiv_comp _ G b ((conv2d_bias_differentiable W x).differentiableAt) hG o 0]

/-- **Depthwise total-loss fold.** The depthwise analogue: the single loss gradient wrt the depthwise
    kernel `W` (flattened) equals the certified `∂(depthwiseConv2d)/∂W` contracted with
    `∂loss/∂(depthwise output)`, via `pdiv_comp` on `G ∘ (depthwise weight-map)` at a smooth point. -/
theorem depthwise_total_loss_grad_fold {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW)
    (G : Vec (c * h * w) → Vec 1)
    (hG : DifferentiableAt ℝ G (Tensor3.flatten (depthwiseConv2d W b x)))
    (idx : Fin (c * kH * kW)) :
    pdiv (fun v : Vec (c * kH * kW) =>
            G (Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v) b x))) (Tensor3.flatten W) idx 0
      = ∑ k : Fin (c * h * w),
          pdiv (fun v : Vec (c * kH * kW) =>
                  Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v) b x)) (Tensor3.flatten W) idx k
            * pdiv G (Tensor3.flatten (depthwiseConv2d W b x)) k 0 := by
  rw [show (fun v : Vec (c * kH * kW) =>
              G (Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v) b x)))
        = G ∘ (fun v : Vec (c * kH * kW) =>
                Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v) b x)) from rfl,
      pdiv_comp _ G (Tensor3.flatten W) ((depthwise_weight_differentiable b x).differentiableAt)
        (by simp only [Tensor3.unflatten_flatten]; exact hG) idx 0]
  simp only [Tensor3.unflatten_flatten]

/-- **Depthwise bias total-loss fold.** -/
theorem depthwise_bias_total_loss_grad_fold {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c)
    (G : Vec (c * h * w) → Vec 1)
    (hG : DifferentiableAt ℝ G (Tensor3.flatten (depthwiseConv2d W b x))) (o : Fin c) :
    pdiv (fun b' : Vec c => G (Tensor3.flatten (depthwiseConv2d W b' x))) b o 0
      = ∑ k : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b o k
            * pdiv G (Tensor3.flatten (depthwiseConv2d W b x)) k 0 := by
  rw [show (fun b' : Vec c => G (Tensor3.flatten (depthwiseConv2d W b' x)))
        = G ∘ (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) from rfl,
      pdiv_comp _ G b ((depthwise_bias_differentiable W x).differentiableAt) hG o 0]

end Proofs
