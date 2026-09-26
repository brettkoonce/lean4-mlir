import LeanMlir.Proofs.Architectures.ConvIndex
import LeanMlir.Proofs.Float.FloatBridge

/-! # The conv forward in floating point — conv as a weight-shared dense layer, and its rounding budget

The 2D conv read as a dense layer over the zero-padded window (`convPad`, `k4Idx`, `w3Idx`,
`convWindow`), the kernel drift that makes it Lipschitz in the weights, and the float conv:
`convF` / `flatConvF` and the `flatConvF_close` budget (used by `floatClose_flatConv` and
`SgdDescentCnn`).
The whole MNIST-CNN forward budget built from them is in `SgdDescentCnn`.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Conv-kernel drift: a dense layer with weight sharing
-- ════════════════════════════════════════════════════════════════

/-- `conv2d` through `convPad`: bias plus the kernel-linear form. -/
theorem conv2d_eq_convPad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi =
      b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * convPad kH kW x c kh kw hi wi := rfl

/-- Padded reads are bounded by the input bound (out-of-bounds reads are
    zero). -/
theorem abs_convPad_le {ic h w kH kW : Nat} (x : Tensor3 ic h w) {a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ c i j, |x c i j| ≤ a)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW x c kh kw hi wi| ≤ a := by
  unfold convPad
  split_ifs with h
  · exact hx _ _ _
  · simpa using ha

/-- Flat index of a `Kernel4` entry (the suite's row-major layout). -/
def k4Idx {oc ic kH kW : Nat} (o : Fin oc) (c : Fin ic)
    (kh : Fin kH) (kw : Fin kW) : Fin (oc * ic * kH * kW) :=
  finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, c), kh), kw)

/-- `k4Idx` reads back through `Kernel4.unflatten`. -/
theorem unflatten_k4Idx {oc ic kH kW : Nat} (v : Vec (oc * ic * kH * kW))
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Kernel4.unflatten v o c kh kw = v (k4Idx o c kh kw) := rfl

/-- `Kernel4.flatten` reads off at a `k4Idx` — the forward peer of
    `unflatten_k4Idx`, lifting a per-entry kernel bound to the flattened vector. -/
theorem flatten_k4Idx {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Kernel4.flatten W (k4Idx o c kh kw) = W o c kh kw := by
  simp only [Kernel4.flatten, k4Idx, Equiv.symm_apply_apply]

/-- Every flat kernel index is a `k4Idx` — lets the abstract `∀ idx` gradient
    accuracy be discharged per `(o,cc,kh,kw)` by `cnn_conv2_grad_close`. -/
theorem k4Idx_surj {oc ic kH kW : Nat} (idx : Fin (oc * ic * kH * kW)) :
    ∃ (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
      idx = k4Idx o c kh kw := by
  refine ⟨(finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1).1,
    (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1).2,
    (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2,
    (finProdFinEquiv.symm idx).2, ?_⟩
  simp only [k4Idx, Prod.mk.eta, Equiv.apply_symm_apply]

/-- The output-channel slabs tile the kernel: summing the slab masses over
    the output channels recovers the total `ℓ1` mass. -/
theorem sum_abs_k4 {oc ic kH kW : Nat} (e : Vec (oc * ic * kH * kW)) :
    ∑ idx, |e idx| =
      ∑ o : Fin oc, ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        |e (k4Idx o c kh kw)| := by
  simp only [sum_finProdFinEquiv]; rfl

/-- The `ℓ1` mass of one output-channel slab is at most the total `ℓ1`
    mass — the conv analogue of a dense column being part of the flat
    parameter vector. -/
theorem sum_abs_kernel_slab_le {oc ic kH kW : Nat}
    (e : Vec (oc * ic * kH * kW)) (o : Fin oc) :
    ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, |e (k4Idx o c kh kw)| ≤
      ∑ idx, |e idx| := by
  rw [sum_abs_k4 e]
  exact Finset.single_le_sum (f := fun o => ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
    |e (k4Idx o c kh kw)|) (fun _ _ => by positivity) (Finset.mem_univ o)

-- ════════════════════════════════════════════════════════════════
-- § Conv forward rounding budget (planning §1b-A): conv = dense at the
--   conv fan-in, so the float conv close IS `dense_close` on the
--   per-output-coordinate flattened window.
-- ════════════════════════════════════════════════════════════════

/-- Flat index of a conv *window* slot `(c, kh, kw)` — `k4Idx` without the
    output channel (row-major, fan-in `ic·kH·kW`). -/
def w3Idx {ic kH kW : Nat} (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Fin (ic * kH * kW) :=
  finProdFinEquiv (finProdFinEquiv (c, kh), kw)

/-- The triple conv-window sum collapses to one flat sum over the fan-in —
    the conv analogue of `dot` being a single-index sum (mirrors `sum_abs_k4`,
    one fewer axis). -/
theorem sum_w3 {ic kH kW : Nat} (g : Fin (ic * kH * kW) → ℝ) :
    ∑ idx, g idx =
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, g (w3Idx c kh kw) :=
  sum_finProdFinEquiv₃ g

/-- The per-output-coordinate conv *window* as a flat `Vec` over the fan-in:
    the (padded) input reads that the kernel slab dots against. -/
noncomputable def convWindow {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (hi : Fin h) (wi : Fin w) : Vec (ic * kH * kW) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    convPad kH kW x q.1 q.2 p.2 hi wi

/-- The kernel as a `Mat (ic·kH·kW) oc` — column `o` is the flattened slab. -/
noncomputable def convKernelMat {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : Mat (ic * kH * kW) oc :=
  fun idx o =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    W o q.1 q.2 p.2

@[simp] theorem convWindow_w3 {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (hi : Fin h) (wi : Fin w) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    convWindow kH kW x hi wi (w3Idx c kh kw) = convPad kH kW x c kh kw hi wi := by
  simp [convWindow, w3Idx, Equiv.symm_apply_apply]

@[simp] theorem convKernelMat_w3 {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    convKernelMat W (w3Idx c kh kw) o = W o c kh kw := by
  simp [convKernelMat, w3Idx, Equiv.symm_apply_apply]

/-- **conv2d is a dense layer at the conv fan-in** — `conv = dense-with-sharing`
    made exact: each output coordinate is `Proofs.dense` of the kernel slab
    against the flattened window. The structural fact that lets the float conv
    budget reuse `dense_close`. -/
theorem conv2d_eq_dense {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi =
      Proofs.dense (convKernelMat W) b (convWindow kH kW x hi wi) o := by
  rw [conv2d_eq_convPad]
  show b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      W o c kh kw * convPad kH kW x c kh kw hi wi
    = (∑ idx, convWindow kH kW x hi wi idx * convKernelMat W idx o) + b o
  rw [sum_w3 (fun idx => convWindow kH kW x hi wi idx * convKernelMat W idx o),
      add_comm]
  refine congrArg (· + b o) ?_
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun kh _ =>
    Finset.sum_congr rfl fun kw _ => ?_
  rw [convWindow_w3, convKernelMat_w3]; ring

/-- Padded reads of inputs within `e` stay within `e` (the read is either a
    coordinate, diff `≤ e`, or `0`, diff `0`). -/
theorem convPad_close {ic h w kH kW : Nat} (xt xa : Tensor3 ic h w) {e : ℝ}
    (he : 0 ≤ e) (hx : ∀ c i j, |xt c i j - xa c i j| ≤ e)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW xt c kh kw hi wi - convPad kH kW xa c kh kw hi wi| ≤ e := by
  unfold convPad
  split_ifs with h
  · exact hx _ _ _
  · simpa using he

/-- **The float conv layer** — `M.dense` of the kernel slab against the
    flattened window, per output coordinate. The float peer of `conv2d`
    (every product/accumulate/bias-add rounded), in the dense form. -/
noncomputable def FloatModel.convF {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w) :
    Tensor3 oc h w :=
  fun o hi wi => M.dense (convKernelMat W) b (convWindow kH kW x hi wi) o

/-- **Conv forward rounding budget (Item A).** The rounded conv at a float input
    within `e` of the real activation is within the conv-fan-in `denseErr` of the
    real conv — `dense_close` at the flattened window. The compounded Higham
    factor rides the fan-in `ic·kH·kW` (the dense column length here), exactly as
    the planning doc calls for. -/
theorem FloatModel.convF_close {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (xt xa : Tensor3 ic h w) {e : ℝ}
    (he : 0 ≤ e) (hx : ∀ c i j, |xt c i j - xa c i j| ≤ e)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |M.convF W b xt o hi wi - conv2d W b xa o hi wi| ≤
      M.denseErr (convKernelMat W) b (convWindow kH kW xa hi wi) e o := by
  rw [conv2d_eq_dense, FloatModel.convF]
  refine M.dense_close (convKernelMat W) b (convWindow kH kW xt hi wi)
    (convWindow kH kW xa hi wi) e he ?_ o
  intro idx
  simp only [convWindow]
  exact convPad_close xt xa he hx _ _ _ hi wi

/-- Kernel-slab entries inherit the uniform kernel magnitude bound. -/
theorem convKernelMat_abs_le {oc ic kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {w' : ℝ} (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    (i : Fin (ic * kH * kW)) (j : Fin oc) : |convKernelMat W i j| ≤ w' := by
  simp only [convKernelMat]; exact hW _ _ _ _

/-- Window reads inherit the uniform input magnitude bound (padding reads 0). -/
theorem convWindow_abs_le {ic h w kH kW : Nat} {x : Tensor3 ic h w} {a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ c i j, |x c i j| ≤ a) (hi : Fin h) (wi : Fin w)
    (idx : Fin (ic * kH * kW)) : |convWindow kH kW x hi wi idx| ≤ a := by
  simp only [convWindow]; exact abs_convPad_le x ha hx _ _ _ hi wi

/-- **Conv output magnitude bound** = `dense_abs_le` at the fan-in: conv is a
    dense layer, so `|conv2dⱼ| ≤ layerAct (ic·kH·kW) w β a`. -/
theorem conv2d_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {b : Vec oc} {x : Tensor3 ic h w} {w' β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hx : ∀ c i j, |x c i j| ≤ a) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d W b x o hi wi| ≤ FloatModel.layerAct (ic * kH * kW) w' β a := by
  rw [conv2d_eq_dense]
  exact FloatModel.dense_abs_le ha (fun i j => convKernelMat_abs_le hW i j) hb
    (fun idx => convWindow_abs_le ha hx hi wi idx) o

-- ════════════════════════════════════════════════════════════════
-- § Vec-space float conv: the form the MNIST-CNN forward composes
-- ════════════════════════════════════════════════════════════════

/-- **Vec-space float conv** — the float peer of `flatConv`
    (`flatten ∘ conv2d ∘ unflatten`), with the rounded `convF` inside. -/
noncomputable def FloatModel.flatConvF {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  fun v => Tensor3.flatten (M.convF W b (Tensor3.unflatten v))

/-- **Vec-space conv forward budget, uniform.** The rounded `flatConvF` at a
    float input within `e` of the real activation is within the conv-fan-in
    `layerBudget` of the real `flatConv` — every output coordinate, one closed
    form. The conv layer threads exactly like a dense layer at fan-in
    `ic·kH·kW`. -/
theorem FloatModel.flatConvF_close {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (vt va : Vec (ic * h * w))
    {w' β a e : ℝ} (hw' : 0 ≤ w') (ha : 0 ≤ a) (he : 0 ≤ e)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hva : ∀ k, |va k| ≤ a) (hvte : ∀ k, |vt k - va k| ≤ e)
    (k : Fin (oc * h * w)) :
    |M.flatConvF W b vt k - flatConv W b va k| ≤
      FloatModel.layerBudget M.u (ic * kH * kW) w' β a e := by
  have huf_e : ∀ c i j,
      |Tensor3.unflatten vt c i j - Tensor3.unflatten va c i j| ≤ e := by
    intro c i j; simp only [Tensor3.unflatten]; exact hvte _
  have huf_a : ∀ c i j, |Tensor3.unflatten va c i j| ≤ a := by
    intro c i j; simp only [Tensor3.unflatten]; exact hva _
  simp only [FloatModel.flatConvF, flatConv, Tensor3.flatten]
  refine (M.convF_close W b (Tensor3.unflatten vt) (Tensor3.unflatten va)
    he huf_e _ _ _).trans ?_
  exact M.denseErr_le_uniform hw' he (fun i j => convKernelMat_abs_le hW i j) hb
    (fun idx => convWindow_abs_le ha huf_a _ _ idx) _

/-- Vec-space conv magnitude bound (the activation-norm pass-through). -/
theorem flatConv_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {b : Vec oc} {v : Vec (ic * h * w)} {w' β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hv : ∀ k, |v k| ≤ a) (k : Fin (oc * h * w)) :
    |flatConv W b v k| ≤ FloatModel.layerAct (ic * kH * kW) w' β a := by
  have huf : ∀ c i j, |Tensor3.unflatten v c i j| ≤ a := by
    intro c i j; simp only [Tensor3.unflatten]; exact hv _
  simp only [flatConv, Tensor3.flatten]
  exact conv2d_abs_le ha hW hb huf _ _ _

end Proofs
