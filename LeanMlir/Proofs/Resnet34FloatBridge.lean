import LeanMlir.Proofs.SgdDescentCnn
import LeanMlir.Proofs.Foundation.StridedConv
import LeanMlir.Proofs.BnFloatBridge
import LeanMlir.Proofs.Foundation.PerChannelBN

/-!
# ℝ→Float32 bridge: the ResNet-34 structural ops

Extending the float bridge from CIFAR/BN toward ResNet-34. After the `rsqrt`
keystone (`BnFloatBridge.lean`), no new *numerical* primitives remain — the r34
ops are reuses or thin wrappers:

* **residual skip** `relu(F(x) + skip(x))` — needs a two-operand `add_close`
  (the additive peer of `mul_close`); `reluAdd_close` is the post-skip output.
* **strided conv** — `flatConvStride2 = decimateFlat ∘ flatConv`, so the float
  closeness is `flatConvF_close` at the decimated coordinate.
* **per-channel BN** — `bnPerChannelMat` is `bnForward` applied per channel-row,
  so `bnPerChannelFlat_close_of` maps `bnForward_close_of` over channels.
* **global-avg-pool** — a per-channel mean, so `gapFlat_close` reduces to
  `bnMean_close` on the channel slice (`sum_s2` flattens the spatial double sum).

The remaining work for a whole-net `r34_float_close` is the (large, mechanical)
per-block + whole-net composition threading the inherited error through the 16
residual blocks — no new numerical content.
-/

namespace Proofs

namespace FloatModel

variable (M : FloatModel)

-- ════════════════════════════════════════════════════════════════
-- § Residual additive fan-in
-- ════════════════════════════════════════════════════════════════

/-- **Rounded addition with inherited operand errors.** `fl(xt ⊕ yt)` is within
    `u·(|x| + ex + |y| + ey) + (ex + ey)` of the exact `x + y`, given
    `|xt − x| ≤ ex` and `|yt − y| ≤ ey`. The additive peer of `mul_close`; the
    residual fan-in's float budget. -/
theorem add_close {xt x yt y ex ey : ℝ} (hx : |xt - x| ≤ ex) (hy : |yt - y| ≤ ey) :
    |M.add xt yt - (x + y)| ≤ M.u * (|x| + ex + |y| + ey) + (ex + ey) := by
  have hu := M.u_nonneg
  have hxt : |xt| ≤ |x| + ex := by
    have h := abs_sub_le xt x 0; simp only [sub_zero] at h; linarith
  have hyt : |yt| ≤ |y| + ey := by
    have h := abs_sub_le yt y 0; simp only [sub_zero] at h; linarith
  have h1 : |M.add xt yt - (xt + yt)| ≤ M.u * |xt + yt| := M.err _
  have h2 : |(xt + yt) - (x + y)| ≤ ex + ey := by
    have he : (xt + yt) - (x + y) = (xt - x) + (yt - y) := by ring
    rw [he]; exact (abs_add_le _ _).trans (add_le_add hx hy)
  have hsum : |xt + yt| ≤ |x| + ex + (|y| + ey) :=
    (abs_add_le _ _).trans (add_le_add hxt hyt)
  calc |M.add xt yt - (x + y)|
      ≤ |M.add xt yt - (xt + yt)| + |(xt + yt) - (x + y)| := abs_sub_le _ _ _
    _ ≤ M.u * |xt + yt| + (ex + ey) := add_le_add h1 h2
    _ ≤ M.u * (|x| + ex + |y| + ey) + (ex + ey) := by gcongr; linarith [hsum]

/-- **Residual block output (post-skip ReLU).** With the two branches `bt`/`st`
    within `eb`/`es` of the real `b`/`s` (magnitudes `≤ A`/`B`), the rounded
    `relu(fl(bt ⊕ st))` is within the `add_close` budget of `relu(b + s)` per
    coordinate (ReLU is exact in float and 1-Lipschitz). The float closeness of
    `relu(F(x) + skip(x))`. -/
theorem reluAdd_close {n : Nat} {bt b st s : Vec n} {eb es A B : ℝ}
    (hb : ∀ i, |bt i - b i| ≤ eb) (hs : ∀ i, |st i - s i| ≤ es)
    (hB : ∀ i, |b i| ≤ A) (hS : ∀ i, |s i| ≤ B) (i : Fin n) :
    |relu n (fun j => M.add (bt j) (st j)) i - relu n (fun j => b j + s j) i| ≤
      M.u * (A + eb + B + es) + (eb + es) := by
  have hadd : ∀ j, |M.add (bt j) (st j) - (b j + s j)| ≤ M.u * (A + eb + B + es) + (eb + es) :=
    fun j => by
      refine (M.add_close (hb j) (hs j)).trans ?_
      have hu := M.u_nonneg
      gcongr <;> [exact hB j; exact hS j]
  exact relu_close (fun j => M.add (bt j) (st j)) (fun j => b j + s j)
    (M.u * (A + eb + B + es) + (eb + es)) hadd i

-- ════════════════════════════════════════════════════════════════
-- § Strided convolution  (flatConvStride2 = decimate ∘ flatConv)
-- ════════════════════════════════════════════════════════════════

/-- The float stride-2 conv: decimate the float stride-1 conv (the float peer of
    `flatConvStride2 = decimateFlat ∘ flatConv`). -/
noncomputable def flatConvStride2F {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  decimateFlat oc h w ∘ (M.flatConvF (h := 2 * h) (w := 2 * w) W b)

/-- **Stride-2 conv forward budget.** Decimation only selects output coordinates,
    so the strided-conv closeness is `flatConvF_close` evaluated at the decimated
    coordinate — the same conv-fan-in `layerBudget`. -/
theorem flatConvStride2F_close {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (vt va : Vec (ic * (2 * h) * (2 * w)))
    {w' β a e : ℝ} (hw' : 0 ≤ w') (ha : 0 ≤ a) (he : 0 ≤ e)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hva : ∀ k, |va k| ≤ a) (hvte : ∀ k, |vt k - va k| ≤ e)
    (k : Fin (oc * h * w)) :
    |M.flatConvStride2F W b vt k - flatConvStride2 W b va k| ≤
      FloatModel.layerBudget M.u (ic * kH * kW) w' β a e := by
  simp only [FloatModel.flatConvStride2F, flatConvStride2, Function.comp, decimateFlat]
  exact M.flatConvF_close (h := 2 * h) (w := 2 * w) W b vt va hw' ha he hW hb hva hvte
    (decimateIdx oc h w k)

-- ════════════════════════════════════════════════════════════════
-- § Per-channel BatchNorm  (bnForward applied per channel-row)
-- ════════════════════════════════════════════════════════════════

/-- The float per-channel BN: `bnForwardF` per channel-row, with the per-channel
    mean `fμ c` and inverse-stddev `fistdv c`. The float peer of
    `bnPerChannelFlat` (= `bnForward` per row). -/
noncomputable def bnPerChannelFlatF {oc m : Nat} (M : FloatModel)
    (γ β fμ fistdv : Vec oc) (v : Vec (oc * m)) : Vec (oc * m) :=
  Mat.flatten (fun c => M.bnForwardF (γ c) (β c) (fμ c) (fistdv c) (Mat.unflatten v c))

/-- **Per-channel BN forward closeness.** Each channel-row runs `bnForward`, so
    the float per-channel BN is within `bnNormBudget` of `bnPerChannelFlat` per
    entry — `bnForward_close_of` mapped over channels (uniform per-channel mean/
    istd errors and magnitude bounds). -/
theorem bnPerChannelFlat_close_of {oc m : Nat} (M : FloatModel)
    {ε emean eistd D S G Bbnd : ℝ} (γ β fμ fistdv : Vec oc) (v : Vec (oc * m))
    (hmean : ∀ c, |fμ c - bnMean m (Mat.unflatten v c)| ≤ emean)
    (histd : ∀ c, |fistdv c - bnIstd m (Mat.unflatten v c) ε| ≤ eistd)
    (hD : ∀ c j, |Mat.unflatten v c j - bnMean m (Mat.unflatten v c)| ≤ D)
    (hSabs : ∀ c, |bnIstd m (Mat.unflatten v c) ε| ≤ S)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd) (k : Fin (oc * m)) :
    |M.bnPerChannelFlatF γ β fμ fistdv v k - bnPerChannelFlat oc m ε γ β v k| ≤
      bnNormBudget M.u D S G Bbnd emean eistd := by
  simp only [FloatModel.bnPerChannelFlatF, bnPerChannelFlat, bnPerChannelMat, Mat.flatten]
  exact M.bnForward_close_of (ε := ε) (Mat.unflatten v ((finProdFinEquiv.symm k).1))
    ((finProdFinEquiv.symm k).2) (hmean _) (histd _) (hD _ _) (hSabs _) (hγ _) (hβ _)

-- ════════════════════════════════════════════════════════════════
-- § Global average pool  (a per-channel mean)
-- ════════════════════════════════════════════════════════════════

/-- The float global-average-pool: per channel, the float mean of the channel's
    `h·w` spatial slice (rounded sum, rounded `/(h·w)`). The float peer of
    `globalAvgPoolFlat`. -/
noncomputable def gapFlatF {c h w : Nat} (M : FloatModel) (v : Vec (c * h * w)) : Vec c :=
  fun ci => M.div (M.sum (fun s : Fin (h * w) =>
    Tensor3.unflatten v ci (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2)) ((h * w : ℕ) : ℝ)

/-- **Global-average-pool closeness.** GAP is the per-channel spatial mean, so the
    float GAP is within the `bnMean_close` budget of `globalAvgPoolFlat` per
    channel (`sum_s2` flattens the spatial double sum to the `Fin (h·w)` slice the
    mean rounds). -/
theorem gapFlat_close {c h w : Nat} (M : FloatModel) (v : Vec (c * h * w)) {A : ℝ}
    (hhw : 0 < h * w) (hA : ∀ ci hi wi, |Tensor3.unflatten v ci hi wi| ≤ A) (ci : Fin c) :
    |M.gapFlatF v ci - globalAvgPoolFlat c h w v ci| ≤
      M.u * ((1 + M.u) ^ (h * w + 1) * A) + ((1 + M.u) ^ (h * w + 1) - 1) * A := by
  set cs : Vec (h * w) := fun s =>
    Tensor3.unflatten v ci (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2 with hcs
  have hgap : globalAvgPoolFlat c h w v ci = bnMean (h * w) cs := by
    simp only [globalAvgPoolFlat, globalAvgPool, bnMean]
    congr 1
    · rw [sum_s2 cs]
      refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
      simp only [hcs, Equiv.symm_apply_apply]
    · push_cast; ring
  rw [FloatModel.gapFlatF, ← hcs, hgap]
  exact M.bnMean_close cs hhw (fun i => by rw [hcs]; exact hA _ _ _)

end FloatModel

/-- **GAP as a per-channel `bnMean`.** `globalAvgPoolFlat c h w v ci` is the mean of
    channel `ci`'s spatial slice — `bnMean (h·w)` of the `Fin (h·w)`-indexed gather.
    The reduction `gapFlat_close` performs inline, exposed so the float-bridge
    magnitude/input-shift bounds (`bnMean_abs_le` / `bnMean_input_close`) apply. -/
theorem globalAvgPoolFlat_eq_bnMean {c h w : Nat} (v : Vec (c * h * w)) (ci : Fin c) :
    globalAvgPoolFlat c h w v ci
      = bnMean (h * w) (fun s => Tensor3.unflatten v ci
          (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2) := by
  simp only [globalAvgPoolFlat, globalAvgPool, bnMean]
  congr 1
  · rw [sum_s2 _]
    refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
    simp only [Equiv.symm_apply_apply]
  · push_cast; ring

end Proofs
