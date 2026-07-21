import LeanMlir.Proofs.MobileNetV2Close

/-! # Closing the ResNet-34 render — the parameter-gradient close (a FREE close)

`planning/mobilenetv2_close.md` Item C, applied to ResNet-34 (`tests/TestResnet34Train.lean`,
146 params). Unlike MobileNetV2 — whose close needed a genuinely-new depthwise bridge family —
**every ResNet-34 parameter family is already certified by an existing generic bridge**. ResNet-34
uses only regular convolutions (3×3 and the 7×7 stem), per-channel BN, plain relu, maxpool, residual
add, GAP and dense; there is no depthwise, no relu6, and the maxpool/relu/add/GAP carry no parameters.

| family (render SSA)                              | forward fn         | certified by                                      |
|--------------------------------------------------|--------------------|---------------------------------------------------|
| 3×3 stride-1 conv W/b (id `W1`/`W2`, down `W2`)  | `conv2d`           | `cnn_render_conv{W,b}_certified` (M3, **reuse**)  |
| 3×3 stride-2 conv W/b (down `W1`, projection `Wp`)| `flatConvStride2`  | `mnv2_render_stem_conv{W,b}_certified` (**reuse**)|
| **7×7 stride-2 stem** conv W/b (`sW`/`sb`)        | `flatConvStride2`  | `mnv2_render_stem_conv{W,b}_certified` (**reuse**, kH=kW=7) |
| per-channel BN γ/β (every `g*`/`bt*`)            | `bnPerChannelFlat` | `cifar_bn_render_{gamma,beta}_certified` (**reuse**)|
| dense `Wd`/`bd`                                  | matmul / +bias     | `weight_grad_bridge` / `bias_grad_bridge` (M2, **reuse**)|
| maxpool / relu / residual add / GAP             | —                  | no parameters                                     |

So this file adds **no new VJP**. Its value is the audit gate: it pins the generic strided/regular
conv bridges to ResNet-34's exact kernel sizes — confirming the **7×7 stem** and the **3×3 strided
projection** (the two shapes no prior net exercised through these bridges) really are covered. Each
theorem is the generic certified denotation specialized to the kernel size; `#print axioms` stays
`[propext, Classical.choice, Quot.sound]` by inheritance. The strided-conv W/b bridges this reuses
are exactly the ones built for MobileNetV2's stem (`MobileNetV2Close.lean`); the per-channel BN γ/β
and the dense bridges are verbatim reuse (no kernel to pin), documented above. See
`planning/render_close_handoff.md` §"Validation recipe".
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 7×7 strided stem conv (`sW`/`sb`) — the only 7×7 conv in any net so far
-- ════════════════════════════════════════════════════════════════

/-- **Stem conv weight output, certified (7×7 stride-2).** `sWⁿ = sW − lr·(strided transpose-trick
    grad)` denotes `sW − lr·(certified ∂(flatConvStride2)/∂sW · cotangent)`, the generic strided
    weight bridge at `kH=kW=7`. -/
theorem r34_render_stem_convW_certified {ic oc h w : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * 7 * 7)) (dy : Vec (oc * h * w)) (lr : ℝ) (i : Fin (oc * ic * 7 * 7)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * 7 * 7) => flatConvStride2 (Kernel4.unflatten v') b x)
            v i j * dy j :=
  mnv2_render_stem_convW_certified b x v dy lr i

/-- **Stem conv bias output, certified (7×7 stride-2).** -/
theorem r34_render_stem_convb_certified {ic oc h w : Nat}
    (W : Kernel4 oc ic 7 7) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * dy j :=
  mnv2_render_stem_convb_certified W x b dy lr o

-- ════════════════════════════════════════════════════════════════
-- § 3×3 stride-1 conv (identity-block `W1`/`W2`, downsample `W2`)
-- ════════════════════════════════════════════════════════════════

/-- **Block conv weight output, certified (3×3 stride-1).** The regular-conv weight bridge at
    `kH=kW=3`; covers every stride-1 conv of the identity blocks and the downsample `W2`. -/
theorem r34_render_blockConvW_certified {ic oc h w : Nat}
    (b : Vec oc) (x : Tensor3 ic h w)
    (v : Vec (oc * ic * 3 * 3)) (c : Vec (oc * h * w)) (lr : ℝ) (idx : Fin (oc * ic * 3 * 3)) :
    v idx - lr * (conv2d_weight_grad_has_vjp b x).backward v c idx
      = v idx - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * 3 * 3) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v idx j * c j :=
  cnn_render_convW_certified b x v c lr idx

/-- **Block conv bias output, certified (3×3 stride-1).** -/
theorem r34_render_blockConvb_certified {ic oc h w : Nat}
    (W : Kernel4 oc ic 3 3) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (conv2d_bias_grad_has_vjp W x).backward b c o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  cnn_render_convb_certified W x b c lr o

-- ════════════════════════════════════════════════════════════════
-- § 3×3 stride-2 conv (downsample `W1` + projection `Wp`)
-- ════════════════════════════════════════════════════════════════

/-- **Downsample conv weight output, certified (3×3 stride-2).** The generic strided weight bridge
    at `kH=kW=3`; covers the downsample blocks' `W1` and the 3×3 strided projection skip `Wp`. -/
theorem r34_render_downConvW_certified {ic oc h w : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * 3 * 3)) (dy : Vec (oc * h * w)) (lr : ℝ) (i : Fin (oc * ic * 3 * 3)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * 3 * 3) => flatConvStride2 (Kernel4.unflatten v') b x)
            v i j * dy j :=
  mnv2_render_stem_convW_certified b x v dy lr i

/-- **Downsample conv bias output, certified (3×3 stride-2).** -/
theorem r34_render_downConvb_certified {ic oc h w : Nat}
    (W : Kernel4 oc ic 3 3) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * dy j :=
  mnv2_render_stem_convb_certified W x b dy lr o

-- The per-channel BN γ/β (`cifar_bn_render_{gamma,beta}_certified`) and the dense W/b
-- (`weight_grad_bridge`/`bias_grad_bridge`) families are covered VERBATIM by the existing generic
-- bridges at the ResNet-34 shapes — no kernel size to pin, so no r34-named restatement is added
-- (they are already audited under their own names). Together with the six conv theorems above, every
-- ResNet-34 train-step parameter output is certified `θ − lr·(certified Jacobian · the cotangent)`.

end Proofs
