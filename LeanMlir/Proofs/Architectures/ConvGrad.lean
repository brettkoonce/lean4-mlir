import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.ConvIndex

/-! # Convolution parameter-gradient bridges (the MNIST CNN train step's conv parameters)

The MNIST CNN (`conv → relu → conv → relu → maxpool → dense → relu → dense → relu → dense`)
train step has two kinds of parameters: the dense classifier head (whose grads reuse
`IR.weight_grad_bridge`/`IR.bias_grad_bridge`) and the **convolution kernels/biases**,
whose gradient is a *correlation*, not an outer product. This file supplies the conv
analogue of the dense bridges.

As for the MLP, the cotangent the backward chain delivers at each conv layer's output flows
through a backward graph — here the Tensor3-level `IR.Back3` (`convBackDenote`,
`maxPoolBackDenote`, with `IR.denote_subst3` the chain rule), exactly as the MLP used
`IR.Back` (`mlpCotOut0`/`mlpCotOut1`). Given that cotangent `c`, the conv kernel and
bias gradients (the transpose-trick `conv2dWeightGrad`/`conv2dBiasGrad`) are the
certified Jacobian of `conv2d` — as a function of the flattened kernel / of the bias —
contracted with `c`. Both bridges are the `.correct` field of the proven conv
parameter VJPs (`conv2dWeightGradHasVJP`/`conv2dBiasGradHasVJP`). The statements are
about those witnesses' backwards; no rendered text appears in them. The link from a
rendered conv gradient op to these witnesses is that op's `den` theorem (see `CnnFold`).

With the dense bridges in `IR` and the `Back3` cotangent chain, this gives a bridge for
every parameter of the CNN train step. (The SGD wrapping `θ − lr·∇` is identical to the
linear/MLP case.)

The next section adds the same bridges for the depthwise bias and the stride-2 conv kernel and
bias, which the MobileNetV2, ResNet-34 and ConvNeXt fold ties use.

The rest are the conv's Jacobians in closed form and its drifts, read off the same certified
VJPs: the kernel map (`conv2d_weight_pdiv`, the drift `conv2d_kernel_drift_sum`), the input map
(the kernel tap `convTap`, `conv2d_input_pdiv3`, the locality bound `convTap_out_l1` and the drift
`conv2d_input_l1_drift`) and the bias map (`conv2d_bias_pdiv`, `conv2d_flat_bias_drift_sum`), with
the weight and bias gradients as a flat dot and sum (`convWeightGrad_eq_dot`,
`convBiasGrad_eq_sum`). The MNIST CNN descent rungs (`SgdDescent.Cnn`) consume them.
-/

namespace Proofs

/-- **Conv weight-gradient bridge.** At any cotangent `c` at the conv layer's output
    (and any kernel point `v = Kernel4.flatten W`), the backward of
    `conv2dWeightGradHasVJP` (the transpose-trick kernel gradient) equals the certified Jacobian of `conv2d` viewed as a function of the flattened
    kernel, contracted with `c`. The convolution analogue of `IR.weight_grad_bridge`;
    it is the `.correct` field of `conv2dWeightGradHasVJP`. -/
theorem conv_weight_grad_bridge {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w)
    (v : Vec (oc * ic * kH * kW)) (c : Vec (oc * h * w))
    (idx : Fin (oc * ic * kH * kW)) :
    (conv2dWeightGradHasVJP b x).backward v c idx
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               v idx j * c j :=
  (conv2dWeightGradHasVJP b x).correct v c idx

/-- **Conv bias-gradient bridge.** Likewise the conv bias gradient (`db[o] = Σ
    spatial c`) is the certified Jacobian of `conv2d` wrt the bias, contracted with
    `c` — the `.correct` field of `conv2dBiasGradHasVJP`. -/
theorem conv_bias_grad_bridge {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc * h * w)) (o : Fin oc) :
    (conv2dBiasGradHasVJP W x).backward b c o
      = ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  (conv2dBiasGradHasVJP W x).correct b c o

-- ════════════════════════════════════════════════════════════════
-- § Conv SGD steps — the SGD form of the conv bridges
--
-- The CNN train step (`cnnTrainStepText` — conv→relu→conv→relu→maxpool→dense→relu→
-- dense→relu→dense) emits, per conv layer, `%dWᵢ = convWGrad` (the transpose-trick
-- kernel gradient) and `%Wᵢn = Wᵢ − lr·%dWᵢ`. The two theorems below are stated on the
-- functions, not on emitted text: `θ − lr·(conv backward at c)` equals `θ − lr·(certified
-- conv Jacobian · c)`. The `SgdNodes` conv nodes carry them to the emitted ops. Generic in the cotangent `c`, so one theorem covers both conv
-- layers (W₁, W₂); the dense layers use `weight_grad_bridge`/`bias_grad_bridge`.
-- ════════════════════════════════════════════════════════════════

/-- **Conv weight SGD step — SGD form of `conv_weight_grad_bridge`.** At the flattened
    kernel `v`, `v − lr·(conv2dWeightGradHasVJP backward at c)` equals
    `v − lr·(certified ∂conv/∂kernel · c)`. A rewrite by the bridge; no rendered text
    appears in the statement. -/
theorem conv_weight_sgd_certified {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w)
    (v : Vec (oc * ic * kH * kW)) (c : Vec (oc * h * w)) (lr : ℝ)
    (idx : Fin (oc * ic * kH * kW)) :
    v idx - lr * (conv2dWeightGradHasVJP b x).backward v c idx
      = v idx - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               v idx j * c j := by
  rw [conv_weight_grad_bridge b x v c idx]

/-- **Conv bias SGD step — SGD form of `conv_bias_grad_bridge`.** Likewise
    `b − lr·(conv2dBiasGradHasVJP backward at c)` equals
    `b − lr·(certified ∂conv/∂bias · c)`. -/
theorem conv_bias_sgd_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (conv2dBiasGradHasVJP W x).backward b c o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j := by
  rw [conv_bias_grad_bridge W x b c o]

-- ════════════════════════════════════════════════════════════════
-- § Depthwise (stride-1) bias and stride-2 conv parameter bridges
--
-- The same `.correct`-field bridges for two more conv shapes: the depthwise bias
-- (`depthwiseBiasGradHasVJP`, `Depthwise.lean`, the spatial reduce `db[c] = Σ dy`) and the
-- stride-2 conv kernel and bias (`flatConvStride2WeightGradHasVJP` /
-- `flatConvStride2BiasGradHasVJP`, `StridedConv.lean`). `SgdNodes` builds the stride-1
-- depthwise op nodes on the bias bridge (`SgdNode.depthwiseB_den`) and the strided conv nodes on the
-- two conv bridges (`SgdNode.convStrided{W,B}_den`). The cotangent is a binder here too.
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise bias output, certified.** Likewise `bⁿ = b − lr·(spatial reduce)` denotes
    `b − lr·(certified ∂(depthwiseConv2d)/∂b · cotangent)`. -/
theorem depthwise_bias_sgd_certified {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ) (cc : Fin c) :
    b cc - lr * (depthwiseBiasGradHasVJP W x).backward b dy cc
      = b cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j := by
  rw [(depthwiseBiasGradHasVJP W x).correct]

/-- **Stem conv weight output, certified.** `sWⁿ = sW − lr·(strided transpose-trick grad)` denotes
    `sW − lr·(certified ∂(flatConvStride2)/∂sW · cotangent)`, via `flatConvStride2WeightGradHasVJP`
    (the ch6 strided conv weight VJP). -/
theorem convStride2_weight_sgd_certified {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (lr : ℝ)
    (i : Fin (oc * ic * kH * kW)) :
    v i - lr * (flatConvStride2WeightGradHasVJP b x).backward v dy i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride2 (Kernel4.unflatten v') b x)
            v i j * dy j := by
  rw [flatConvStride2WeightGradHasVJP_correct]

/-- **Stem conv bias output, certified.** `sbⁿ = sb − lr·(spatial reduce)` denotes
    `sb − lr·(certified ∂(flatConvStride2)/∂sb · cotangent)`. -/
theorem convStride2_bias_sgd_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2BiasGradHasVJP W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * dy j := by
  rw [(flatConvStride2BiasGradHasVJP W x).correct]

-- ════════════════════════════════════════════════════════════════
-- § Conv kernel drift: the output moves by the kernel perturbation's slab
-- ════════════════════════════════════════════════════════════════

theorem conv2d_kernel_sub {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) (v e : Vec (oc * ic * kH * kW))
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi =
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi := by
  simp only [conv2d_eq_convPad, unflatten_k4Idx, Pi.add_apply, add_sub_add_left_eq_sub,
    ← Finset.sum_sub_distrib, add_mul, add_sub_cancel_left]

/-- **Per-entry conv drift, slab-refined**: a kernel perturbation moves the
    output entry `(o, hi, wi)` by at most `a` times the `ℓ1` mass of the
    channel-`o` slab (each output reads only its own slab). -/
theorem conv2d_kernel_drift {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW))
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi| ≤
      a * ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        |e (k4Idx o c kh kw)| := by
  rw [conv2d_kernel_sub]
  calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi|
      ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW,
          e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW,
          e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi| :=
        Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi| :=
        Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)| * a := by
        refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.sum_le_sum fun kw _ => ?_
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left
          (abs_convPad_le x ha hx c kh kw hi wi) (abs_nonneg _)
    _ = (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)|) * a := by
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun c _ => ?_
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun kh _ => ?_
        rw [Finset.sum_mul]
    _ = a * ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)| := mul_comm _ _

/-- Per-entry conv drift against the TOTAL `ℓ1` mass — the form the relu
    margins consume. -/
theorem conv2d_kernel_drift_total {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW))
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi| ≤ a * ∑ idx, |e idx| :=
  le_trans (conv2d_kernel_drift b x ha hx v e o hi wi)
    (mul_le_mul_of_nonneg_left (sum_abs_kernel_slab_le e o) ha)

/-- **`ℓ1` conv drift**: summed over all output entries, the drift is at
    most `(h·w)·a·‖e‖₁` — the spatial multiplicity `h·w` is the price of
    weight sharing (each kernel entry touches every spatial position). -/
theorem conv2d_kernel_drift_sum {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW)) :
    ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
          conv2d (Kernel4.unflatten v) b x o hi wi| ≤
      ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) := by
  refine (Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun hi _ =>
    Finset.sum_le_sum fun wi _ => conv2d_kernel_drift b x ha hx v e o hi wi).trans_eq ?_
  rw [sum_abs_k4]
  simp [Finset.mul_sum, mul_assoc]

-- ════════════════════════════════════════════════════════════════
-- § The conv weight-map Jacobian: closed form, point-free, ℓ1 row mass
-- ════════════════════════════════════════════════════════════════

/-- **Closed form of the conv weight-map `pdiv`** — extracted from the
    certified VJP (`conv2dWeightGradHasVJP`) by contracting its
    `.correct` field against a basis vector. Kernel entry `(o,cc,kh,kw)`
    touches output `(co,hi,wi)` iff `co = o`, with coefficient the padded
    input read `convPad`. NB the right-hand side does not mention `v`:
    the weight map is affine, so its Jacobian is point-free — this is
    what lets the gradient difference along a step segment collapse to
    the head drift alone. -/
theorem conv2d_weight_pdiv {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) (v : Vec (oc * ic * kH * kW))
    (o : Fin oc) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW)
    (co : Fin oc) (hi : Fin h) (wi : Fin w) :
    pdiv (fun v' : Vec (oc * ic * kH * kW) =>
        Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
      (k4Idx o cc kh kw) (t3Idx co hi wi)
      = if co = o then convPad kH kW x cc kh kw hi wi else 0 := by
  have hb := conv_weight_grad_bridge b x v (basisVec (t3Idx co hi wi))
    (k4Idx o cc kh kw)
  have hsum : ∑ j : Fin (oc * h * w),
      pdiv (fun v' : Vec (oc * ic * kH * kW) =>
          Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
        (k4Idx o cc kh kw) j * basisVec (t3Idx co hi wi) j
      = pdiv (fun v' : Vec (oc * ic * kH * kW) =>
          Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
        (k4Idx o cc kh kw) (t3Idx co hi wi) := by
    simp
  rw [← hsum, ← hb]
  -- evaluate the transpose-trick backward at the basis vector
  simp only [conv2dWeightGradHasVJP, k4Idx, Equiv.symm_apply_apply,
    basisVec_apply, convPad]
  simp only [t3Idx_def]
  simp [ite_and, @eq_comm _ o co]

-- ════════════════════════════════════════════════════════════════
-- § Conv gradient windows: the conv weight grad is a spatial
--   correlation (a dot), the bias grad a spatial sum — the forms the float
--   dot and sum round.
-- ════════════════════════════════════════════════════════════════

/-- The padded-input window for a fixed kernel slot, flattened over the
    `(hi, wi)` spatial grid — the left operand of the conv weight-grad dot. -/
noncomputable def convPadWin {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) : Vec (h * w) :=
  fun s => convPad kH kW x cc kh kw (finProdFinEquiv.symm s).1
    (finProdFinEquiv.symm s).2

/-- The cotangent slab for a fixed output channel, flattened over `(hi, wi)`. -/
noncomputable def cotWin {oc h w : Nat} (cot : Tensor3 oc h w) (o : Fin oc) :
    Vec (h * w) :=
  fun s => cot o (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2

@[simp] theorem convPadWin_apply {ic h w : Nat} (kH kW : Nat)
    (x : Tensor3 ic h w) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW)
    (hi : Fin h) (wi : Fin w) :
    convPadWin kH kW x cc kh kw (finProdFinEquiv (hi, wi)) =
      convPad kH kW x cc kh kw hi wi := by
  simp [convPadWin, Equiv.symm_apply_apply]

@[simp] theorem cotWin_apply {oc h w : Nat} (cot : Tensor3 oc h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    cotWin cot o (finProdFinEquiv (hi, wi)) = cot o hi wi := by
  simp [cotWin, Equiv.symm_apply_apply]

/-- **The conv weight gradient is the spatial dot** `Σ_{hi,wi} convPad · cot`
    (the contraction `conv2d_weight_pdiv` certifies as `∂L/∂W_{o,cc,kh,kw}`),
    re-expressed as a flat `Fin (h·w)` dot of the padded-input window against
    the cotangent slab — the form the float dot rounds. -/
theorem convWeightGrad_eq_dot {ic oc h w kH kW : Nat} (x : Tensor3 ic h w)
    (cot : Tensor3 oc h w) (o : Fin oc) (cc : Fin ic) (kh : Fin kH)
    (kw : Fin kW) :
    ∑ s, convPadWin kH kW x cc kh kw s * cotWin cot o s =
      ∑ hi : Fin h, ∑ wi : Fin w,
        convPad kH kW x cc kh kw hi wi * cot o hi wi := by
  rw [sum_finProdFinEquiv (fun s => convPadWin kH kW x cc kh kw s * cotWin cot o s)]
  refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
  rw [convPadWin_apply, cotWin_apply]

/-- The conv bias gradient is the spatial sum `Σ_{hi,wi} cot`. -/
theorem convBiasGrad_eq_sum {oc h w : Nat} (cot : Tensor3 oc h w) (o : Fin oc) :
    ∑ s, cotWin cot o s = ∑ hi : Fin h, ∑ wi : Fin w, cot o hi wi := by
  rw [sum_finProdFinEquiv (fun s => cotWin cot o s)]
  refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
  rw [cotWin_apply]

-- ════════════════════════════════════════════════════════════════
-- § Conv as a function of its INPUT: the tap Jacobian and its masses
--
-- The conv1 rung crosses conv2 as a function of its input. Conv is
-- LINEAR in its input; the Jacobian entry pairing input `(ci,hi,wi)`
-- with output `(co,ho,wo)` is a single kernel tap (`convTap`, the
-- input-side peer of `convPad`), extracted from the certified input-VJP
-- (`conv2dHasVJP3`) by contracting `.correct` against a basis
-- cotangent — point-free, exactly like `conv2d_weight_pdiv`. Each
-- input entry feeds at most `oc·kH·kW` outputs (`convTap_out_l1`): the
-- `ℓ1` operator factor of a conv crossing is `(channels)·kH·kW·w₂ᶜ`,
-- NOT a spatial count — locality is what keeps the conv1 constant
-- usable at trained magnitudes.
-- ════════════════════════════════════════════════════════════════

/-- Swap the two index pairs of a quadruple sum. -/
private theorem sum_swap_pair_pair {α β γ δ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] [Fintype δ]
    (f : α → β → γ → δ → ℝ) :
    ∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, f a b c d =
      ∑ c : γ, ∑ d : δ, ∑ a : α, ∑ b : β, f a b c d :=
  calc ∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, f a b c d
      = ∑ c : γ, ∑ a : α, ∑ b : β, ∑ d : δ, f a b c d :=
        Finset.sum_comm_cycle (f := fun a b c => ∑ d : δ, f a b c d)
    _ = ∑ c : γ, ∑ a : α, ∑ d : δ, ∑ b : β, f a b c d :=
        Finset.sum_congr rfl fun _c _ =>
          Finset.sum_congr rfl fun _a _ => Finset.sum_comm
    _ = ∑ c : γ, ∑ d : δ, ∑ a : α, ∑ b : β, f a b c d :=
        Finset.sum_congr rfl fun _c _ => Finset.sum_comm

/-- The kernel tap that multiplies input entry `(ci,hi,wi)` in output
    entry `(co,ho,wo)` — the input-side Jacobian entry of `conv2d`.
    Depends on the kernel only, never the input (conv is linear in its
    input). Deliberately let-free, like `convPad`. -/
noncomputable def convTap {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) : ℝ :=
  if hpad : ho.val ≤ hi.val + (kH - 1) / 2 ∧
      hi.val + (kH - 1) / 2 - ho.val < kH ∧
      wo.val ≤ wi.val + (kW - 1) / 2 ∧
      wi.val + (kW - 1) / 2 - wo.val < kW then
    W co ci ⟨hi.val + (kH - 1) / 2 - ho.val, hpad.2.1⟩
            ⟨wi.val + (kW - 1) / 2 - wo.val, hpad.2.2.2⟩
  else 0

/-- A single conv tap is bounded by the kernel magnitude (out-of-pad taps are
    zero) — the per-entry bound the conv-2 backward `dot_perturbed_close` uses. -/
theorem convTap_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {w' : ℝ} (hw' : 0 ≤ w') (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    |convTap W ci hi wi co ho wo| ≤ w' := by
  unfold convTap
  split_ifs with hpad
  · exact hW _ _ _ _
  · simpa using hw'

/-- **The tap as a kernel-offset indicator sum**: `|convTap|` is the sum
    over kernel offsets `(kh,kw)` of `|W co ci kh kw|` pinned to the
    unique offset aligning input `(hi,wi)` with output `(ho,wo)`. The
    workhorse for both mass bounds: summing it over OUTPUTS pins
    `(ho,wo)` per offset, summing it over INPUTS pins `(hi,wi)`. -/
theorem abs_convTap_expand {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    |convTap W ci hi wi co ho wo| =
      ∑ kh : Fin kH, ∑ kw : Fin kW,
        if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
            kw.val + wo.val = wi.val + (kW - 1) / 2
          then |W co ci kh kw| else 0 := by
  unfold convTap
  split_ifs with hpad
  · have e1 : ∀ kh : Fin kH, kh.val + ho.val = hi.val + (kH - 1) / 2 ↔ kh = ⟨_, hpad.2.1⟩ :=
      fun kh => by simp only [Fin.ext_iff]; omega
    have e2 : ∀ kw : Fin kW, kw.val + wo.val = wi.val + (kW - 1) / 2 ↔ kw = ⟨_, hpad.2.2.2⟩ :=
      fun kw => by simp only [Fin.ext_iff]; omega
    simp only [e1, e2, ite_and, Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq',
      Finset.mem_univ, ite_true]
  · rw [abs_zero, eq_comm]
    exact Finset.sum_eq_zero fun kh _ => Finset.sum_eq_zero fun kw _ =>
      ite_eq_right fun hcon => hpad (by omega)

/-- Output-side tap mass: one input entry feeds at most `oc·kH·kW`
    outputs, each through a tap bounded by `wK` — the `ℓ1→ℓ1` operator
    factor of a conv crossing as a function of its input. -/
theorem convTap_out_l1 {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    {wK : ℝ} (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w) :
    ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |convTap W ci hi wi co ho wo| ≤
      ((oc * kH * kW : ℕ) : ℝ) * wK := by
  calc ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |convTap W ci hi wi co ho wo|
      ≤ ∑ _co : Fin oc, ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK := by
        refine Finset.sum_le_sum fun co _ => ?_
        calc ∑ ho : Fin h, ∑ wo : Fin w, |convTap W ci hi wi co ho wo|
            = ∑ ho : Fin h, ∑ wo : Fin w, ∑ kh : Fin kH, ∑ kw : Fin kW,
                (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = wi.val + (kW - 1) / 2
                  then |W co ci kh kw| else 0) := by
              refine Finset.sum_congr rfl fun ho _ =>
                Finset.sum_congr rfl fun wo _ => ?_
              exact abs_convTap_expand W ci hi wi co ho wo
          _ = ∑ kh : Fin kH, ∑ kw : Fin kW, ∑ ho : Fin h, ∑ wo : Fin w,
                (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = wi.val + (kW - 1) / 2
                  then |W co ci kh kw| else 0) := by
              exact sum_swap_pair_pair _
          _ ≤ ∑ kh : Fin kH, ∑ kw : Fin kW, |W co ci kh kw| := by
              refine Finset.sum_le_sum fun kh _ =>
                Finset.sum_le_sum fun kw _ => ?_
              rw [← Fintype.sum_prod_type', ← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
              refine mul_le_of_le_one_left (abs_nonneg _)
                (Nat.cast_le_one.mpr (Finset.card_le_one.mpr ?_))
              simp only [Finset.mem_filter, Finset.mem_univ, true_and]
              exact fun p hp q hq => Prod.ext (Fin.ext (by omega)) (Fin.ext (by omega))
          _ ≤ ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK :=
              Finset.sum_le_sum fun kh _ => Finset.sum_le_sum fun kw _ =>
                hW co ci kh kw
    _ = ((oc * kH * kW : ℕ) : ℝ) * wK := by simp [mul_assoc]

/-- **Closed form of the conv input-map `pdiv3`** — extracted from the
    certified input-VJP (`conv2dHasVJP3`) by contracting its
    `.correct` field against a basis cotangent. Point-free in `x`:
    conv is linear in its input. -/
theorem conv2d_input_pdiv3 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    pdiv3 (conv2d W b) x ci hi wi co ho wo =
      convTap W ci hi wi co ho wo := by
  have hb := (conv2dHasVJP3 W b).correct x
    (fun co' ho' wo' =>
      if co' = co ∧ ho' = ho ∧ wo' = wo then (1:ℝ) else 0) ci hi wi
  have hsum : ∑ co' : Fin oc, ∑ ho' : Fin h, ∑ wo' : Fin w,
      pdiv3 (conv2d W b) x ci hi wi co' ho' wo' *
        (if co' = co ∧ ho' = ho ∧ wo' = wo then (1:ℝ) else 0) =
      pdiv3 (conv2d W b) x ci hi wi co ho wo := by
    simp [ite_and, mul_ite]
  rw [← hsum, ← hb]
  -- evaluate the explicit input-gradient formula at the basis cotangent
  simp only [conv2dHasVJP3, conv2dInputGradFormula]
  rw [Fintype.sum_eq_single co fun co' hne => Finset.sum_eq_zero fun ho' _ =>
      Finset.sum_eq_zero fun wo' _ => by simp [hne],
    Fintype.sum_eq_single ho fun ho' hne => Finset.sum_eq_zero fun wo' _ => by simp [hne],
    Fintype.sum_eq_single wo fun wo' hne => by simp [hne]]
  simp only [and_self, ite_true, mul_one]
  rfl

/-- Flat-coordinate form of `conv2d_input_pdiv3` — the shape the chain
    rule through `flatConv W₂ b₂` consumes. -/
theorem conv2d_flat_input_pdiv {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (y : Vec (ic * h * w))
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    pdiv (fun u : Vec (ic * h * w) =>
        Tensor3.flatten (conv2d W b (Tensor3.unflatten u))) y
      (t3Idx ci hi wi) (t3Idx co ho wo) =
      convTap W ci hi wi co ho wo := by
  have h1 : pdiv (fun u : Vec (ic * h * w) =>
      Tensor3.flatten (conv2d W b (Tensor3.unflatten u))) y
      (t3Idx ci hi wi) (t3Idx co ho wo) =
      pdiv3 (conv2d W b) (Tensor3.unflatten y) ci hi wi co ho wo := by
    unfold pdiv3
    rw [Tensor3.flatten_unflatten]
  rw [h1, conv2d_input_pdiv3]

/-- The real conv-2 backward `∑ convTap·c2R` is magnitude-bounded by the tap
    ℓ∞-mass `(c·(2h)·(2w))·w₂` times the cotangent bound `CP` — the (loose,
    uniform) bound on the real conv-1 cotangent. -/
theorem convTap_back_abs_le {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (c2R : Tensor3 c (2*h) (2*w))
    {w₂ CP : ℝ} (hw₂ : 0 ≤ w₂)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hc2R : ∀ co ho wo, |c2R co ho wo| ≤ CP)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo * c2R co ho wo| ≤
      ((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * CP) := by
  have hbound : (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
      |convTap W₂ ci hi wi co ho wo * c2R co ho wo|) ≤
      ((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * CP) := by
    calc (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
            |convTap W₂ ci hi wi co ho wo * c2R co ho wo|)
        ≤ ∑ _co : Fin c, ∑ _ho : Fin (2*h), ∑ _wo : Fin (2*w), w₂ * CP := by
          refine Finset.sum_le_sum fun co _ => Finset.sum_le_sum fun ho _ =>
            Finset.sum_le_sum fun wo _ => ?_
          rw [abs_mul]
          exact mul_le_mul (convTap_abs_le hw₂ hW₂ ci hi wi co ho wo)
            (hc2R co ho wo) (abs_nonneg _) hw₂
      _ = ((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * CP) := by simp [mul_assoc]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine (Finset.sum_le_sum fun co _ => Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine (Finset.sum_le_sum fun co _ => Finset.sum_le_sum fun ho _ =>
    Finset.abs_sum_le_sum_abs _ _).trans hbound

-- ════════════════════════════════════════════════════════════════
-- § Conv input drift: per-entry (ℓ∞) and total (ℓ1), locality factors
-- ════════════════════════════════════════════════════════════════

/-- Padded reads move no more than the input entries. -/
private theorem abs_convPad_sub_le {ic h w kH kW : Nat} (x x' : Tensor3 ic h w)
    {δ : ℝ} (hδ : 0 ≤ δ) (hclose : ∀ c i j, |x' c i j - x c i j| ≤ δ)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW x' c kh kw hi wi - convPad kH kW x c kh kw hi wi| ≤
      δ := by
  unfold convPad
  split_ifs with hcond
  · exact hclose _ _ _
  · simpa using hδ

/-- The conv output difference under an input perturbation, exactly: the kernel taps
    contract the padded-input differences — `conv2d` is linear in its input (the input-side
    peer of `conv2d_kernel_sub` / `conv2d_bias_sub`). -/
private theorem conv2d_input_sub {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x x' : Tensor3 ic h w) (o : Fin oc) (ho : Fin h) (wo : Fin w) :
    conv2d W b x' o ho wo - conv2d W b x o ho wo =
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * (convPad kH kW x' c kh kw ho wo - convPad kH kW x c kh kw ho wo) := by
  simp only [conv2d_eq_convPad, add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib, mul_sub]

/-- **Per-entry conv input drift**: each output reads `ic·kH·kW` padded
    inputs through taps bounded by `wK`. -/
theorem conv2d_input_entry_drift {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x x' : Tensor3 ic h w)
    {wK δ : ℝ} (hwK : 0 ≤ wK) (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK)
    (hδ : 0 ≤ δ) (hclose : ∀ c i j, |x' c i j - x c i j| ≤ δ)
    (o : Fin oc) (ho : Fin h) (wo : Fin w) :
    |conv2d W b x' o ho wo - conv2d W b x o ho wo| ≤
      ((ic * kH * kW : ℕ) : ℝ) * (wK * δ) := by
  rw [conv2d_input_sub]
  calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * (convPad kH kW x' c kh kw ho wo -
          convPad kH kW x c kh kw ho wo)|
      ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)| :=
        Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)| :=
        Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK * δ := by
        refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.sum_le_sum fun kw _ => ?_
        rw [abs_mul]
        exact mul_le_mul (hW o c kh kw)
          (abs_convPad_sub_le x x' hδ hclose c kh kw ho wo)
          (abs_nonneg _) hwK
    _ = ((ic * kH * kW : ℕ) : ℝ) * (wK * δ) := by simp [mul_assoc]

/-- The padded-read drift as a position-pinned indicator sum — the
    input-side peer of `abs_convTap_expand`, for the `ℓ1` bound. -/
private theorem abs_convPad_sub_expand {ic h w kH kW : Nat} (x x' : Tensor3 ic h w)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (ho : Fin h) (wo : Fin w) :
    |convPad kH kW x' c kh kw ho wo - convPad kH kW x c kh kw ho wo| =
      ∑ i : Fin h, ∑ j : Fin w,
        if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
            kw.val + wo.val = j.val + (kW - 1) / 2
          then |x' c i j - x c i j| else 0 := by
  unfold convPad
  split_ifs with hpad
  · have e1 : ∀ i : Fin h, kh.val + ho.val = i.val + (kH - 1) / 2 ↔ i = ⟨_, hpad.2.1⟩ :=
      fun i => by simp only [Fin.ext_iff]; omega
    have e2 : ∀ j : Fin w, kw.val + wo.val = j.val + (kW - 1) / 2 ↔ j = ⟨_, hpad.2.2.2⟩ :=
      fun j => by simp only [Fin.ext_iff]; omega
    simp only [e1, e2, ite_and, Finset.sum_ite_irrel, Finset.sum_const_zero, Finset.sum_ite_eq',
      Finset.mem_univ, ite_true]
  · rw [sub_zero, abs_zero, eq_comm]
    exact Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun j _ =>
      ite_eq_right fun hcon => hpad (by omega)

/-- **`ℓ1` conv input drift**: each input entry feeds at most `oc·kH·kW`
    outputs, so the total output drift is at most `oc·kH·kW·wK` times
    the total input drift — locality, not a spatial count. -/
theorem conv2d_input_l1_drift {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x x' : Tensor3 ic h w)
    {wK : ℝ} (hwK : 0 ≤ wK) (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK) :
    ∑ o : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |conv2d W b x' o ho wo - conv2d W b x o ho wo| ≤
      ((oc * kH * kW : ℕ) : ℝ) *
        (wK * ∑ c : Fin ic, ∑ i : Fin h, ∑ j : Fin w,
          |x' c i j - x c i j|) := by
  have hentry : ∀ (o : Fin oc) (ho : Fin h) (wo : Fin w),
      |conv2d W b x' o ho wo - conv2d W b x o ho wo| ≤
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        wK * |convPad kH kW x' c kh kw ho wo -
          convPad kH kW x c kh kw ho wo| := by
    intro o ho wo
    rw [conv2d_input_sub]
    calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)|
        ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW,
            W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo)| :=
          Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            |W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo)| :=
          Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
            Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| := by
          refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum
            fun kh _ => Finset.sum_le_sum fun kw _ => ?_
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_right (hW o c kh kw) (abs_nonneg _)
  calc ∑ o : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |conv2d W b x' o ho wo - conv2d W b x o ho wo|
      ≤ ∑ o : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
          ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| :=
        Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun ho _ =>
          Finset.sum_le_sum fun wo _ => hentry o ho wo
    _ = ∑ o : Fin oc, ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          ∑ ho : Fin h, ∑ wo : Fin w,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| := by
        refine Finset.sum_congr rfl fun o _ => ?_
        exact Finset.sum_comm_cycle.trans
          (Finset.sum_congr rfl fun c _ => sum_swap_pair_pair _)
    _ ≤ ∑ _o : Fin oc, ∑ c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW,
          wK * (∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j|) := by
        refine Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun c _ =>
          Finset.sum_le_sum fun kh _ => Finset.sum_le_sum fun kw _ => ?_
        simp only [← Finset.mul_sum]
        refine mul_le_mul_of_nonneg_left ?_ hwK
        calc ∑ ho : Fin h, ∑ wo : Fin w,
              |convPad kH kW x' c kh kw ho wo -
                convPad kH kW x c kh kw ho wo|
            = ∑ ho : Fin h, ∑ wo : Fin w, ∑ i : Fin h, ∑ j : Fin w,
                (if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = j.val + (kW - 1) / 2
                  then |x' c i j - x c i j| else 0) := by
              refine Finset.sum_congr rfl fun ho _ =>
                Finset.sum_congr rfl fun wo _ => ?_
              exact abs_convPad_sub_expand x x' c kh kw ho wo
          _ = ∑ i : Fin h, ∑ j : Fin w, ∑ ho : Fin h, ∑ wo : Fin w,
                (if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = j.val + (kW - 1) / 2
                  then |x' c i j - x c i j| else 0) := by
              exact sum_swap_pair_pair _
          _ ≤ ∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j| := by
              refine Finset.sum_le_sum fun i _ =>
                Finset.sum_le_sum fun j _ => ?_
              rw [← Fintype.sum_prod_type', ← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
              refine mul_le_of_le_one_left (abs_nonneg _)
                (Nat.cast_le_one.mpr (Finset.card_le_one.mpr ?_))
              simp only [Finset.mem_filter, Finset.mem_univ, true_and]
              exact fun p hp q hq => Prod.ext (Fin.ext (by omega)) (Fin.ext (by omega))
    _ = ((oc * kH * kW : ℕ) : ℝ) *
          (wK * ∑ c : Fin ic, ∑ i : Fin h, ∑ j : Fin w,
            |x' c i j - x c i j|) := by
        simp [← Finset.mul_sum, mul_assoc, mul_left_comm]

-- ════════════════════════════════════════════════════════════════
-- § The conv bias: affine difference, drift, and the Kronecker Jacobian
--
-- The bias rungs. `conv2d` is affine in its bias with the SIMPLEST
-- possible Jacobian: output `(co,hi,wi)` reads bias entry `co` with
-- coefficient 1 — a Kronecker channel indicator, point-free. The
-- per-entry drift is exactly `|e o|` (no input bound `a`, no kernel
-- mass); only the `ℓ1` drift picks up the spatial multiplicity `h·w`
-- (one bias entry feeds a whole channel — sharing, exactly as for the
-- kernel). Everything downstream of the conv is the kernel-rung
-- argument verbatim.
-- ════════════════════════════════════════════════════════════════

/-- The conv output difference under a bias perturbation, exactly:
    output `(o,hi,wi)` moves by `e o` — `conv2d` is affine in the bias. -/
theorem conv2d_bias_sub {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (x : Tensor3 ic h w) (b e : Vec oc)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W (b + e) x o hi wi - conv2d W b x o hi wi = e o := by
  rw [conv2d_eq_convPad, conv2d_eq_convPad, Pi.add_apply]
  ring

/-- Per-entry conv drift under a bias perturbation: the perturbation's
    own entry — no `a` factor, no kernel mass. Flat-index form. -/
theorem conv2d_flat_bias_drift_total {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b e : Vec oc)
    (k : Fin (oc * h * w)) :
    |Tensor3.flatten (conv2d W (b + e) x) k -
      Tensor3.flatten (conv2d W b x) k| ≤ ∑ idx, |e idx| := by
  obtain ⟨o, hi, wi, rfl⟩ := t3Idx_surj k
  rw [flatten_t3Idx, flatten_t3Idx, conv2d_bias_sub]
  exact Finset.single_le_sum (f := fun idx => |e idx|)
    (fun idx _ => abs_nonneg _) (Finset.mem_univ o)

/-- **`ℓ1` conv bias drift**: summed over all output entries, at most
    `(h·w)·‖e‖₁` — one bias entry feeds every spatial position of its
    channel. -/
theorem conv2d_flat_bias_drift_sum {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b e : Vec oc) :
    ∑ k, |Tensor3.flatten (conv2d W (b + e) x) k -
        Tensor3.flatten (conv2d W b x) k| ≤
      ((h * w : ℕ) : ℝ) * ∑ idx, |e idx| := by
  rw [sum_t3 (fun k : Fin (oc * h * w) =>
    |Tensor3.flatten (conv2d W (b + e) x) k -
      Tensor3.flatten (conv2d W b x) k|)]
  refine le_of_eq ?_
  calc ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |Tensor3.flatten (conv2d W (b + e) x) (t3Idx o hi wi) -
          Tensor3.flatten (conv2d W b x) (t3Idx o hi wi)|
      = ∑ o : Fin oc, ∑ _hi : Fin h, ∑ _wi : Fin w, |e o| := by
        refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [flatten_t3Idx, flatten_t3Idx, conv2d_bias_sub]
    _ = ((h * w : ℕ) : ℝ) * ∑ idx, |e idx| := by simp [Finset.mul_sum, mul_assoc]

/-- **Closed form of the conv bias-map `pdiv`** — extracted from the
    certified VJP (`conv2dBiasGradHasVJP`) by contracting its
    `.correct` field against a basis vector, exactly as
    `conv2d_weight_pdiv`. Bias entry `o` touches output `(co,hi,wi)`
    iff `co = o`, with coefficient 1 — the Kronecker channel indicator.
    Point-free (the bias map is affine), so along a step segment only
    the head gradient moves. -/
theorem conv2d_bias_pdiv {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (x : Tensor3 ic h w) (b : Vec oc) (o : Fin oc)
    (co : Fin oc) (hi : Fin h) (wi : Fin w) :
    pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b
      o (t3Idx co hi wi)
      = if co = o then (1:ℝ) else 0 := by
  have hb := conv_bias_grad_bridge W x b (basisVec (t3Idx co hi wi)) o
  have hsum : ∑ j : Fin (oc * h * w),
      pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j *
        basisVec (t3Idx co hi wi) j
      = pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o
          (t3Idx co hi wi) := by
    simp
  rw [← hsum, ← hb]
  -- evaluate the spatial-sum backward at the basis vector
  simp only [conv2dBiasGradHasVJP, basisVec_apply]
  simp only [t3Idx_def]
  simp [ite_and, @eq_comm _ o co]

end Proofs
