import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.FloatComposeBridge

/-!
# ℝ→Float32 bridge: depthwise convolution (the one new conv lemma for EfficientNet)

EfficientNet's MBConv depthwise stage is the only forward op in the enet line whose
float bound isn't already a wrap of an existing closeness. Structurally it is *easier*
than a full conv: per output `(ch, hi, wi)` it is

  `depthwiseConv2d W b x ch hi wi = b ch + Σ_{kh,kw} W ch kh kw · pad(x ch …)`

— a dot over the `kH·kW` window with **no channel sum** (fan-in `kH·kW`, not `ic·kH·kW`).
That is the depthwise efficiency advantage, and it carries to the bound: the Higham
γ-factor rides `kH·kW`.

We get it for free from the existing conv scaffolding. The padded read in the depthwise
forward is *definitionally* `convPad kH kW x ch kh kw hi wi` (same SAME-padding `dite`),
so each output channel is a single-output `Proofs.dense` over the `kH·kW`-flattened
window (`depthwiseConv2d_eq_dense`), and `dense_close` / `denseErr_le_uniform` deliver
the budget exactly as they do for the regular conv (`flatConvF_close`). Then
`floatClose_depthwise` is the `FloatClose` wrap, the depthwise peer of
`floatClose_flatConv`.
-/

namespace Proofs

open Finset

-- ════════════════════════════════════════════════════════════════
-- § Depthwise window / kernel as a single-output dense at fan-in kH·kW
-- ════════════════════════════════════════════════════════════════

/-- The per-output-coordinate depthwise *window* as a flat `Vec` over the `kH·kW`
    fan-in: channel `ch`'s padded reads that its filter dots against. The depthwise
    analogue of `convWindow`, one channel only (no `ic` axis). -/
noncomputable def dwWindow {c h w : Nat} (kH kW : Nat) (x : Tensor3 c h w)
    (ch : Fin c) (hi : Fin h) (wi : Fin w) : Vec (kH * kW) :=
  fun idx => convPad kH kW x ch
    (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 hi wi

/-- Channel `ch`'s depthwise filter as a single-column `Mat (kH·kW) 1`. -/
noncomputable def dwKernelMat {c kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (ch : Fin c) : Mat (kH * kW) 1 :=
  fun idx _ => W ch (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2

@[simp] theorem dwWindow_k {c h w : Nat} (kH kW : Nat) (x : Tensor3 c h w)
    (ch : Fin c) (hi : Fin h) (wi : Fin w) (kh : Fin kH) (kw : Fin kW) :
    dwWindow kH kW x ch hi wi (finProdFinEquiv (kh, kw)) = convPad kH kW x ch kh kw hi wi := by
  simp [dwWindow, Equiv.symm_apply_apply]

@[simp] theorem dwKernelMat_k {c kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (ch : Fin c) (kh : Fin kH) (kw : Fin kW) (j : Fin 1) :
    dwKernelMat W ch (finProdFinEquiv (kh, kw)) j = W ch kh kw := by
  simp [dwKernelMat, Equiv.symm_apply_apply]

/-- **Depthwise conv2d is a single-output dense at fan-in `kH·kW`** — each output
    channel `ch` is `Proofs.dense` of its flattened filter against its window. The
    structural fact that lets the float depthwise budget reuse `dense_close`. -/
theorem depthwiseConv2d_eq_dense {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Tensor3 c h w)
    (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    depthwiseConv2d W b x ch hi wi =
      Proofs.dense (dwKernelMat W ch) (fun _ => b ch) (dwWindow kH kW x ch hi wi) 0 := by
  show b ch + ∑ kh : Fin kH, ∑ kw : Fin kW, W ch kh kw * convPad kH kW x ch kh kw hi wi
    = (∑ idx, dwWindow kH kW x ch hi wi idx * dwKernelMat W ch idx 0) + b ch
  rw [sum_s2 (fun idx => dwWindow kH kW x ch hi wi idx * dwKernelMat W ch idx 0), add_comm]
  refine congrArg (· + b ch) ?_
  refine Finset.sum_congr rfl fun kh _ => Finset.sum_congr rfl fun kw _ => ?_
  rw [dwWindow_k, dwKernelMat_k]; ring

/-- Depthwise filter entries inherit the uniform kernel magnitude bound. -/
theorem dwKernelMat_abs_le {c kH kW : Nat} {W : DepthwiseKernel c kH kW} {w' : ℝ}
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (ch : Fin c)
    (i : Fin (kH * kW)) (j : Fin 1) : |dwKernelMat W ch i j| ≤ w' := by
  simp only [dwKernelMat]; exact hW _ _ _

/-- Depthwise window reads inherit the uniform input magnitude bound (padding reads 0). -/
theorem dwWindow_abs_le {c h w kH kW : Nat} {x : Tensor3 c h w} {a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ ch i j, |x ch i j| ≤ a) (ch : Fin c) (hi : Fin h) (wi : Fin w)
    (idx : Fin (kH * kW)) : |dwWindow kH kW x ch hi wi idx| ≤ a := by
  simp only [dwWindow]; exact abs_convPad_le x ha hx _ _ _ hi wi

/-- **Depthwise output magnitude bound** = `dense_abs_le` at fan-in `kH·kW`. -/
theorem depthwiseConv2d_abs_le {c h w kH kW : Nat} {W : DepthwiseKernel c kH kW}
    {b : Vec c} {x : Tensor3 c h w} {w' β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ β)
    (hx : ∀ ch i j, |x ch i j| ≤ a) (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    |depthwiseConv2d W b x ch hi wi| ≤ FloatModel.layerAct (kH * kW) w' β a := by
  rw [depthwiseConv2d_eq_dense]
  exact FloatModel.dense_abs_le ha (fun i j => dwKernelMat_abs_le hW ch i j)
    (fun _ => hb ch) (fun idx => dwWindow_abs_le ha hx ch hi wi idx) 0

-- ════════════════════════════════════════════════════════════════
-- § The float depthwise conv + its rounding budget
-- ════════════════════════════════════════════════════════════════

/-- **The float depthwise conv** — `M.dense` of the single-column filter against the
    flattened window, per output coordinate. The float peer of `depthwiseConv2d`. -/
noncomputable def FloatModel.depthwiseConv2dF {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (x : Tensor3 c h w) : Tensor3 c h w :=
  fun ch hi wi => M.dense (dwKernelMat W ch) (fun _ => b ch) (dwWindow kH kW x ch hi wi) 0

/-- **Depthwise conv forward rounding budget.** The rounded depthwise conv at a float
    input within `e` of the real activation is within the `kH·kW`-fan-in `denseErr` of
    the real depthwise conv — `dense_close` at the flattened window. -/
theorem FloatModel.depthwiseConv2dF_close {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (xt xa : Tensor3 c h w) {e : ℝ}
    (he : 0 ≤ e) (hx : ∀ ch i j, |xt ch i j - xa ch i j| ≤ e)
    (ch : Fin c) (hi : Fin h) (wi : Fin w) :
    |M.depthwiseConv2dF W b xt ch hi wi - depthwiseConv2d W b xa ch hi wi| ≤
      M.denseErr (dwKernelMat W ch) (fun _ => b ch) (dwWindow kH kW xa ch hi wi) e 0 := by
  rw [depthwiseConv2d_eq_dense]
  simp only [FloatModel.depthwiseConv2dF]
  refine M.dense_close (dwKernelMat W ch) (fun _ => b ch) (dwWindow kH kW xt ch hi wi)
    (dwWindow kH kW xa ch hi wi) e he ?_ 0
  intro idx
  simp only [dwWindow]
  exact convPad_close xt xa he hx _ _ _ hi wi

-- ════════════════════════════════════════════════════════════════
-- § Vec-space float depthwise conv (the form the MBConv composes)
-- ════════════════════════════════════════════════════════════════

/-- **Vec-space float depthwise conv** — the float peer of `depthwiseFlat`
    (`flatten ∘ depthwiseConv2d ∘ unflatten`), with the rounded `depthwiseConv2dF`. -/
noncomputable def FloatModel.depthwiseFlatF {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (M.depthwiseConv2dF W b (Tensor3.unflatten v))

/-- **Vec-space depthwise forward budget, uniform.** The rounded `depthwiseFlatF` at a
    float input within `e` of the real activation is within the `kH·kW`-fan-in
    `layerBudget` of the real `depthwiseFlat`, every output coordinate. -/
theorem FloatModel.depthwiseFlatF_close {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (vt va : Vec (c * h * w))
    {w' β a e : ℝ} (hw' : 0 ≤ w') (ha : 0 ≤ a) (he : 0 ≤ e)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ β)
    (hva : ∀ k, |va k| ≤ a) (hvte : ∀ k, |vt k - va k| ≤ e)
    (k : Fin (c * h * w)) :
    |M.depthwiseFlatF W b vt k - depthwiseFlat W b va k| ≤
      FloatModel.layerBudget M.u (kH * kW) w' β a e := by
  have huf_e : ∀ ch i j,
      |Tensor3.unflatten vt ch i j - Tensor3.unflatten va ch i j| ≤ e := by
    intro ch i j; simp only [Tensor3.unflatten]; exact hvte _
  have huf_a : ∀ ch i j, |Tensor3.unflatten va ch i j| ≤ a := by
    intro ch i j; simp only [Tensor3.unflatten]; exact hva _
  simp only [FloatModel.depthwiseFlatF, depthwiseFlat, Tensor3.flatten]
  refine (M.depthwiseConv2dF_close W b (Tensor3.unflatten vt) (Tensor3.unflatten va)
    he huf_e _ _ _).trans ?_
  exact M.denseErr_le_uniform hw' he (fun i j => dwKernelMat_abs_le hW _ i j)
    (fun _ => hb _) (fun idx => dwWindow_abs_le ha huf_a _ _ _ idx) 0

/-- Vec-space depthwise magnitude bound (the activation-norm pass-through). -/
theorem depthwiseFlat_abs_le {c h w kH kW : Nat} {W : DepthwiseKernel c kH kW}
    {b : Vec c} {v : Vec (c * h * w)} {w' β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ β)
    (hv : ∀ k, |v k| ≤ a) (k : Fin (c * h * w)) :
    |depthwiseFlat W b v k| ≤ FloatModel.layerAct (kH * kW) w' β a := by
  have huf : ∀ ch i j, |Tensor3.unflatten v ch i j| ≤ a := by
    intro ch i j; simp only [Tensor3.unflatten]; exact hv _
  simp only [depthwiseFlat, Tensor3.flatten]
  exact depthwiseConv2d_abs_le ha hW hb huf _ _ _

-- ════════════════════════════════════════════════════════════════
-- § FloatClose instance: the depthwise peer of floatClose_flatConv
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise convolution is `FloatClose`** with modulus the depthwise-fan-in
    `layerBudget` (fan-in `kH·kW`, no channel sum — the depthwise efficiency carries
    into the budget). Real output ≤ `layerAct`; float output ≤ that + the fresh-input
    rounding `layerBudget(e=0)`. The depthwise peer of `floatClose_flatConv`; the
    MBConv depthwise stage folds through `.comp` like any other conv. -/
theorem floatClose_depthwise {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A) (hn : 0 < c * h * w)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ β) :
    FloatClose A
      (FloatModel.layerAct (kH * kW) w' β A + FloatModel.layerBudget M.u (kH * kW) w' β A 0)
      (depthwiseFlat (h := h) (w := w) W b) (M.depthwiseFlatF (h := h) (w := w) W b)
      (fun e => FloatModel.layerBudget M.u (kH * kW) w' β A e) := by
  have hLB0 : 0 ≤ FloatModel.layerBudget M.u (kH * kW) w' β A 0 :=
    FloatModel.layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  refine ⟨fun v hv k => ?_, fun vt va e hva hvt hd k => ?_⟩
  · have hreal := depthwiseFlat_abs_le hA hW hb hv k
    have hround : |M.depthwiseFlatF W b v k - depthwiseFlat W b v k|
        ≤ FloatModel.layerBudget M.u (kH * kW) w' β A 0 :=
      M.depthwiseFlatF_close W b v v hw' hA le_rfl hW hb hv (fun k => by simp) k
    refine ⟨hreal.trans (le_add_of_nonneg_right hLB0), ?_⟩
    calc |M.depthwiseFlatF W b v k|
        ≤ |M.depthwiseFlatF W b v k - depthwiseFlat W b v k| + |depthwiseFlat W b v k| := by
          simpa using abs_sub_le (M.depthwiseFlatF W b v k) (depthwiseFlat W b v k) 0
      _ ≤ FloatModel.layerBudget M.u (kH * kW) w' β A 0
            + FloatModel.layerAct (kH * kW) w' β A := add_le_add hround hreal
      _ = FloatModel.layerAct (kH * kW) w' β A
            + FloatModel.layerBudget M.u (kH * kW) w' β A 0 := by ring
  · have he : 0 ≤ e := (abs_nonneg _).trans (hd ⟨0, hn⟩)
    exact M.depthwiseFlatF_close W b vt va hw' hA he hW hb hva hd k

/-- Depthwise conv float-bridges (output magnitude `layerAct + layerBudget` at fan-in
    `kH·kW`) — the depthwise stage of the MBConv whole-net fold. -/
theorem floatBridges_depthwise {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) {w' β : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hn : 0 < c * h * w)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ β) :
    FloatBridges (depthwiseFlat (h := h) (w := w) W b) :=
  fun _A hA => ⟨_, _, _,
    add_nonneg (FloatModel.layerAct_nonneg hw' hβ hA)
      (FloatModel.layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl),
    floatClose_depthwise M W b hw' hβ hA hn hW hb⟩

end Proofs
