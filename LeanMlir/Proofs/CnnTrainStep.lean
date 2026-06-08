import LeanMlir.Proofs.CNN

/-! # M3 — the CNN train step: convolution parameter-gradient bridges

The MNIST CNN (`conv → relu → maxpool → conv → relu → maxpool → dense → … → dense`)
train step has two kinds of parameters: the dense classifier head (whose grads reuse
M2's `weight_grad_bridge`/`bias_grad_bridge`) and the **convolution kernels/biases**,
whose gradient is a *correlation*, not an outer product. This file supplies the conv
analogue of the dense bridges.

As in M2, the cotangent the backward chain delivers at each conv layer's output flows
through a backward graph — here the Tensor3-level `IR.Back3` (`convBackDenote`,
`maxPoolBackDenote`, with `IR.denote_subst3` the chain rule), exactly as the MLP used
`IR.Back` (`mlpCotOut0`/`mlpCotOut1`). Given that cotangent `c`, the conv kernel and
bias gradients (the transpose-trick `conv2d_weight_grad`/`conv2d_bias_grad`) are the
certified Jacobian of `conv2d` — as a function of the flattened kernel / of the bias —
contracted with `c`. Both bridges are the `.correct` field of the proven conv
parameter VJPs (`conv2d_weight_grad_has_vjp`/`conv2d_bias_grad_has_vjp`).

Together with M2's dense bridges and the `Back3` cotangent chain, this covers every
parameter of the CNN train step. (The SGD wrapping `θ − lr·∇` is identical to the
linear/MLP case.)
-/

namespace Proofs

/-- **Conv weight-gradient bridge.** At any cotangent `c` at the conv layer's output
    (and any kernel point `v = Kernel4.flatten W`), the emitted conv kernel gradient
    equals the certified Jacobian of `conv2d` viewed as a function of the flattened
    kernel, contracted with `c`. The convolution analogue of `IR.weight_grad_bridge`;
    it is the `.correct` field of `conv2d_weight_grad_has_vjp`. -/
theorem conv_weight_grad_bridge {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w)
    (v : Vec (oc * ic * kH * kW)) (c : Vec (oc * h * w))
    (idx : Fin (oc * ic * kH * kW)) :
    (conv2d_weight_grad_has_vjp b x).backward v c idx
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               v idx j * c j :=
  (conv2d_weight_grad_has_vjp b x).correct v c idx

/-- **Conv bias-gradient bridge.** Likewise the conv bias gradient (`db[o] = Σ
    spatial c`) is the certified Jacobian of `conv2d` wrt the bias, contracted with
    `c` — the `.correct` field of `conv2d_bias_grad_has_vjp`. -/
theorem conv_bias_grad_bridge {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc * h * w)) (o : Fin oc) :
    (conv2d_bias_grad_has_vjp W x).backward b c o
      = ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  (conv2d_bias_grad_has_vjp W x).correct b c o

end Proofs
