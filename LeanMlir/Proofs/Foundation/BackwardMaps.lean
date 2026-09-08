import LeanMlir.Proofs.Foundation.IR
import LeanMlir.Proofs.Foundation.ResNet34
import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.MaxPool3s2

/-! # The per-op ℝ backward maps — what the certified backward ties are stated about

Every whole-net certified backward tie (`Resnet34BackCertifiedTie`, `MobileNetV2BackCertifiedTie`,
`ConvNeXtBackCertifiedTie`, …) says that a hand-composed chain of per-op backward maps on the
cotangent equals the certified VJP `.backward` of the committed forward. This file is the
per-op vocabulary those chains are written in: the ReLU sign mask, the diagonal scale of a smooth
activation, the per-row lifts, the reversed-kernel conv and depthwise backwards, the two
zero-upsampling scatters behind every strided conv, and the accumulating scatter of the 3×3/s2
stem pool. Each is a plain `noncomputable def` on `Vec`, so a tie's closing `rfl` can unfold it.

Until 2026-09-08 these lived inside the `Proofs/Float/*FloatBridge` files, next to their float
twins and the `FloatClose` budgets — the tier whose numbers were found vacuous and deleted
(`planning/archive/float_second_pass.md`). The definitions that survived are the ones the ties consume;
the leaf ties that are one `rfl` from a certified VJP (`decimateBack_eq_vjp`,
`maxPool3s2FlatBack_eq_vjp_backward`) moved with them. The composite leaf ties that need a proof
(`convFlatBack_eq_vjp_backward`, `depthwiseFlatBack_eq_vjp_backward`, …) stay in the per-op tie
files that always held them.

Net-level chains (`r34InputGradB`, `mnv2InputGrad`, `vitInputGradK`, …) are NOT here: each lives
beside its own tie. The ConvNeXt/ViT channel-LayerNorm backward is in `ChannelLNBack.lean`, which
imports this file. -/

namespace Proofs

open Proofs.IR

-- ════════════════════════════════════════════════════════════════
-- § Pointwise backwards: the ReLU sign mask and the diagonal scale
-- ════════════════════════════════════════════════════════════════

/-- ReLU backward (the rendered `selectPos`): keep `dy i` where the saved pre-activation was
    positive (`cond i`), else 0. The mask `cond` is fixed — the smooth-point sign pattern, which
    is what the ties' nonzero-kink hypotheses pin. -/
noncomputable def reluMaskBack {n : Nat} (cond : Fin n → Prop) [DecidablePred cond]
    (dy : Vec n) : Vec n :=
  fun i => if cond i then dy i else 0

/-- Smooth-activation backward (the rendered `emitActBack`/`scale`): multiply the cotangent
    pointwise by the **saved derivative** `s = act'(preact)`. GELU, Swish/SiLU and sigmoid all have
    a diagonal Jacobian, so their backward is this single `multiply` at a fixed vector `s`. -/
noncomputable def diagBack {n : Nat} (s : Vec n) (dy : Vec n) : Vec n := fun i => s i * dy i

-- ════════════════════════════════════════════════════════════════
-- § Per-row lifts: a per-token map on the flattened `Mat n d ≅ Vec (n·d)`
-- ════════════════════════════════════════════════════════════════

/-- Apply a per-token map `f : Vec d → Vec d` to every row, on the flattened `Vec (n·d)`
    (`Mat.unflatten` → per-row `f` → `Mat.flatten`). The whole-sequence form of a per-token op
    (LayerNorm, the MLP sub-block), so it can compose with a cross-token attention. -/
noncomputable def perRowFlat (n d : Nat) (f : Vec d → Vec d) : Vec (n * d) → Vec (n * d) :=
  fun v => Mat.flatten (fun i => f (Mat.unflatten v i))

/-- `perRowFlat` reads coordinatewise as the per-row map at `(row, col) = finProdFinEquiv.symm idx`. -/
theorem perRowFlat_apply {n d : Nat} (f : Vec d → Vec d) (v : Vec (n * d)) (idx : Fin (n * d)) :
    perRowFlat n d f v idx
      = f (Mat.unflatten v (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 := rfl

/-- **Per-token-input-aware flat lift.** Each row `r` gets its OWN per-token map `g r`, rather
    than the single shared `f` of `perRowFlat`. The flat analogue of `rowwise` (`Tensor.lean`):
    the seam a BACKWARD needs, because a per-token op's input-VJP depends on that token's saved
    activation (LayerNorm-back threads the saved input, GELU-back the saved pre-activation), so
    one shared map cannot carry it. `perRowFlat f` is the special case `g = fun _ => f`
    (`perRowFlatPR_const`). Read block-diagonally it is also the per-block lift — block `hd`
    gets `g hd` — which is the per-head form of `perRowFlat` (multi-head attention: heads =
    blocks) and the per-channel form the BN / channel-LN backwards use. -/
noncomputable def perRowFlatPR (n d : Nat) (g : Fin n → (Vec d → Vec d)) :
    Vec (n * d) → Vec (n * d) :=
  fun v => Mat.flatten (fun i => g i (Mat.unflatten v i))

/-- `perRowFlatPR` reads coordinatewise as row `r`'s own map at `(row, col)`. -/
theorem perRowFlatPR_apply {n d : Nat} (g : Fin n → (Vec d → Vec d))
    (v : Vec (n * d)) (idx : Fin (n * d)) :
    perRowFlatPR n d g v idx
      = g (finProdFinEquiv.symm idx).1 (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2 := rfl

/-- A `perRowFlatPR` over `g = fun _ => f` is the plain `perRowFlat f`. -/
theorem perRowFlatPR_const {n d : Nat} (f : Vec d → Vec d) :
    perRowFlatPR n d (fun _ => f) = perRowFlat n d f := rfl

/-- **Composition of per-row families fuses** — `(perRowFlatPR g) ∘ (perRowFlatPR g')` is
    `perRowFlatPR (fun r => g r ∘ g' r)` (each row is independent, so the two per-row maps
    compose row-by-row). The flat reflection of `rowwise`'s `vjpMat_comp`. -/
theorem perRowFlatPR_comp {n d : Nat} (g g' : Fin n → (Vec d → Vec d)) :
    perRowFlatPR n d g ∘ perRowFlatPR n d g' = perRowFlatPR n d (fun r => g r ∘ g' r) := by
  funext v
  simp only [Function.comp, perRowFlatPR, Mat.unflatten_flatten]

-- ════════════════════════════════════════════════════════════════
-- § The 2×2 max-pool backward (a lookup) and the conv input-VJP
-- ════════════════════════════════════════════════════════════════

/-- **MaxPool backward in flat `Vec` space** — `maxPoolBackDenote x` crossing the flatten
    boundary (`Vec (c·h·w) → Vec (c·(2h)·(2w))`): scatter the pooled cotangent back to each
    window's arg-max input cell, 0 elsewhere. The saved input `x` fixes the arg-max map (the
    smooth-point assumption). The backward of `maxPoolFlat c h w`; tied to the certified VJP by
    `maxPoolFlatBack_eq_vjp_backward`. ⛔ The 3×3/s2 stem pool's backward is
    `maxPool3s2FlatBack`, a different function of the same type. -/
noncomputable def maxPoolFlatBack {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
    Vec (c * h * w) → Vec (c * (2*h) * (2*w)) :=
  fun dy => Tensor3.flatten (maxPoolBackDenote x (Tensor3.unflatten dy))

/-- **Conv backward in flat `Vec` space** — `dx = convBackDenote W dy`. The emitted
    `convolution(dy, reverse(transpose(W)))` denotes a forward `conv2d (reverseSwap W) 0`, which
    in flat space is `flatConv (reverseSwap W) 0`. The backward of `flatConv W b`
    (`Vec (oc·h·w) → Vec (ic·h·w)`); `convFlatBack_eq_vjp_backward` ties it to the certified VJP
    at odd kernels, and `EvenKernelConvBack.lean` says why an even kernel is not its own adjoint. -/
noncomputable def convFlatBack {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Vec (oc * h * w) → Vec (ic * h * w) :=
  flatConv (h := h) (w := w) (reverseSwap W) (fun _ => 0)

-- ════════════════════════════════════════════════════════════════
-- § The two decimation scatters (the zero-upsampling every strided conv reverses through)
-- ════════════════════════════════════════════════════════════════

/-- **Decimation backward (zero-upsampling scatter)** — the certified `decimateFlat` VJP: route
    `dy k` to the even position `decimateIdx k`, 0 elsewhere. `Vec (oc·h·w) → Vec (oc·2h·2w)`.
    The StableHLO `lhs_dilation=[2,2]` of a transposed conv. -/
noncomputable def decimateBack (oc h w : Nat) (dy : Vec (oc * h * w)) :
    Vec (oc * (2 * h) * (2 * w)) :=
  fun idx => ∑ k : Fin (oc * h * w), (if idx = decimateIdx oc h w k then (1 : ℝ) else 0) * dy k

/-- `decimateBack` is exactly the certified `decimateFlat` VJP backward (faithful by definition —
    the VJP backward ignores its primal argument). -/
theorem decimateBack_eq_vjp (oc h w : Nat) (v : Vec (oc * (2 * h) * (2 * w)))
    (dy : Vec (oc * h * w)) :
    decimateBack oc h w dy = (decimateFlat_has_vjp oc h w).backward v dy := rfl

/-- **`decimateOddIdx` is injective** — distinct output cells land at distinct ODD spatial
    positions. Same proof as `decimateIdx_injective` (`ResNet34`): the `2·v+1` doublings are
    injective (`omega`), then peel the `finProdFinEquiv`s. -/
theorem decimateOddIdx_injective (oc h w : Nat) :
    Function.Injective (decimateOddIdx oc h w) := by
  intro k₁ k₂ heq
  simp only [decimateOddIdx] at heq
  obtain ⟨hA, hB⟩ := Prod.mk.inj (finProdFinEquiv.injective heq)
  have hp2 : (finProdFinEquiv.symm k₁).2 = (finProdFinEquiv.symm k₂).2 := by
    have : 2 * (finProdFinEquiv.symm k₁).2.val + 1 = 2 * (finProdFinEquiv.symm k₂).2.val + 1 :=
      Fin.mk.inj_iff.mp hB
    exact Fin.ext (by omega)
  obtain ⟨hA1, hA2⟩ := Prod.mk.inj (finProdFinEquiv.injective hA)
  have hq2 : (finProdFinEquiv.symm (finProdFinEquiv.symm k₁).1).2
           = (finProdFinEquiv.symm (finProdFinEquiv.symm k₂).1).2 := by
    have : 2 * (finProdFinEquiv.symm (finProdFinEquiv.symm k₁).1).2.val + 1
         = 2 * (finProdFinEquiv.symm (finProdFinEquiv.symm k₂).1).2.val + 1 :=
      Fin.mk.inj_iff.mp hA2
    exact Fin.ext (by omega)
  have hq : finProdFinEquiv.symm (finProdFinEquiv.symm k₁).1
          = finProdFinEquiv.symm (finProdFinEquiv.symm k₂).1 := Prod.ext hA1 hq2
  have hp1 : (finProdFinEquiv.symm k₁).1 = (finProdFinEquiv.symm k₂).1 :=
    finProdFinEquiv.symm.injective hq
  exact finProdFinEquiv.symm.injective (Prod.ext hp1 hp2)

/-- **Odd-decimation backward (zero-upsampling scatter at the odd positions)** — the certified
    `decimateOddFlat` VJP: route `dy k` to the odd position `decimateOddIdx k`, 0 elsewhere.
    `Vec (oc·h·w) → Vec (oc·2h·2w)`. The odd-position sibling of `decimateBack`; the map the
    emitted `[p+1, p-1]` transposed-conv pad of an XLA-`SAME` stride-2 conv denotes. -/
noncomputable def decimateOddBack (oc h w : Nat) (dy : Vec (oc * h * w)) :
    Vec (oc * (2 * h) * (2 * w)) :=
  fun idx => ∑ k : Fin (oc * h * w), (if idx = decimateOddIdx oc h w k then (1 : ℝ) else 0) * dy k

/-- `decimateOddBack` is exactly the certified `decimateOddFlat` VJP backward (by definition). -/
theorem decimateOddBack_eq_vjp (oc h w : Nat) (v : Vec (oc * (2 * h) * (2 * w)))
    (dy : Vec (oc * h * w)) :
    decimateOddBack oc h w dy = (decimateOddFlat_has_vjp oc h w).backward v dy := rfl

-- ════════════════════════════════════════════════════════════════
-- § The strided conv backwards: a scatter, then the reversed-kernel conv
-- ════════════════════════════════════════════════════════════════

/-- **Stride-2 conv backward in flat `Vec` space** — the input-VJP of
    `flatConvStride2 W b = decimateFlat ∘ flatConv`: zero-upsample the cotangent (`decimateBack`),
    then run the reversed-kernel conv (`convFlatBack`). `Vec (oc·h·w) → Vec (ic·2h·2w)`. The
    symmetric-pad (ResNet) stem and down-blocks; tied by `flatConvStride2Back_eq_vjp_backward`. -/
noncomputable def flatConvStride2Back {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  convFlatBack (h := 2 * h) (w := 2 * w) W ∘ decimateBack oc h w

/-- **XLA-`SAME` stride-2 conv backward in flat `Vec` space** — the input-VJP of
    `flatConvStride2Xla W b = decimateOddFlat ∘ flatConv` (`StridedConv.lean`): scatter the
    cotangent onto the ODD positions (`decimateOddBack`), then run the reversed-kernel conv.
    The odd-phase peer of `flatConvStride2Back` — the TF-origin stems of EfficientNet-B0,
    MobileNetV2 and MobileNetV4; tied by `flatConvStride2XlaBack_eq_vjp_backward`. -/
noncomputable def flatConvStride2XlaBack {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  convFlatBack (h := 2 * h) (w := 2 * w) W ∘ decimateOddBack oc h w

/-- **Stride-4 patchify conv backward in flat `Vec` space** — the input-VJP of
    `flatConvStride4 W b = decimateFlat ∘ decimateOddFlat ∘ flatConv`: zero-upsample the cotangent
    twice (`decimateBack` then `decimateOddBack`), then run the reversed-kernel conv.
    `Vec (oc·h·w) → Vec (ic·4h·4w)`. The ConvNeXt 4×4/s4 stem's backward. -/
noncomputable def flatConvStride4Back {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Vec (oc * h * w) → Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) :=
  convFlatBack (h := 2 * (2 * h)) (w := 2 * (2 * w)) W
    ∘ decimateOddBack oc (2 * h) (2 * w)
    ∘ decimateBack oc h w

-- ════════════════════════════════════════════════════════════════
-- § The depthwise backwards: the spatially-reversed kernel, then the same two scatters
-- ════════════════════════════════════════════════════════════════

/-- **Spatial reversal of a depthwise kernel** — reverse both spatial axes (`kRev k = kH−1−k`),
    keeping the channel axis (depthwise has no cross-channel mixing, so no transpose, unlike the
    regular conv's `reverseSwap`). The kernel the codegen feeds to the backward depthwise
    `stablehlo.convolution` (`reverse [2,3]`, `feature_group_count = c`). -/
noncomputable def dwReverse {c kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    DepthwiseKernel c kH kW :=
  fun ch kh kw => W ch (Proofs.IR.kRev kh) (Proofs.IR.kRev kw)

/-- **Depthwise conv backward in flat `Vec` space** — the emitted reversed-kernel depthwise
    convolution denotes a forward `depthwiseConv2d (dwReverse W) 0`, which in flat space is
    `depthwiseFlat (dwReverse W) 0`. The backward of `depthwiseFlat W b`
    (`Vec (c·h·w) → Vec (c·h·w)`, channels preserved); tied by `depthwiseFlatBack_eq_vjp_backward`
    through `depthwiseConv2d_dwReverse_eq_input_grad_formula`. -/
noncomputable def depthwiseFlatBack {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    Vec (c * h * w) → Vec (c * h * w) :=
  depthwiseFlat (h := h) (w := w) (dwReverse W) (fun _ => 0)

/-- **Stride-2 depthwise conv backward in flat `Vec` space** — the input-VJP of
    `depthwiseStride2Flat W b = decimateFlat ∘ depthwiseFlat`: zero-upsample the cotangent
    (`decimateBack`, channels preserved), then the reversed-kernel depthwise conv.
    `Vec (c·h·w) → Vec (c·2h·2w)`. The depthwise twin of `flatConvStride2Back`. -/
noncomputable def depthwiseStride2FlatBack {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    Vec (c * h * w) → Vec (c * (2 * h) * (2 * w)) :=
  depthwiseFlatBack (h := 2 * h) (w := 2 * w) W ∘ decimateBack c h w

/-- **XLA-`SAME` stride-2 depthwise conv backward in flat `Vec` space** — the input-VJP of
    `depthwiseStride2FlatXla W b = decimateOddFlat ∘ depthwiseFlat` (`Depthwise.lean`): scatter
    the cotangent onto the ODD positions, then the reversed-kernel depthwise conv. The odd-phase
    peer of `depthwiseStride2FlatBack` (MobileNetV2's four strided depthwises, B0's strided
    MBConvs); tied by `depthwiseStride2FlatXlaBack_eq_vjp_backward`. -/
noncomputable def depthwiseStride2FlatXlaBack {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) :
    Vec (c * h * w) → Vec (c * (2 * h) * (2 * w)) :=
  depthwiseFlatBack (h := 2 * h) (w := 2 * w) W ∘ decimateOddBack c h w

-- ════════════════════════════════════════════════════════════════
-- § The global-average-pool backward (broadcast ÷ h·w)
-- ════════════════════════════════════════════════════════════════

/-- **Global-average-pool backward** — the certified GAP VJP: route `dy(channel)` to every spatial
    cell of that channel, divided by `h·w`. `Vec c → Vec (c·h·w)`. The head endpoint of every conv
    net's backward chain (`r34InputGrad`, `mnv2InputGrad`, `efficientnetInputGradB`, …); the
    emitted `SHlo.gapBack` denotes `globalAvgPoolFlat_has_vjp`'s backward, which is this map. -/
noncomputable def gapBack (c h w : Nat) (dy : Vec c) : Vec (c * h * w) :=
  fun idx => dy (flatChannel c h w idx) / ((h : ℝ) * (w : ℝ))

-- ════════════════════════════════════════════════════════════════
-- § The 3×3/s2 stem-pool backward: an ACCUMULATING scatter, and its VJP at a `Vec` point
-- ════════════════════════════════════════════════════════════════

/-- Row-major re-indexing of a `Fin (c*h*w)` sum as a triple sum — the shape every
    `maxPool3s2` statement is written in. -/
theorem sum_flat3 {c h w : Nat} (g : Fin (c*h*w) → ℝ) :
    ∑ k : Fin (c*h*w), g k
      = ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          g (finProdFinEquiv (finProdFinEquiv (co, ho), wo)) := by
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin (c*h) × Fin w ≃ Fin (c*h*w)) g,
      Fintype.sum_prod_type,
      ← Equiv.sum_comp (finProdFinEquiv : Fin c × Fin h ≃ Fin (c*h))
        (fun a => ∑ b : Fin w, g (finProdFinEquiv (a, b))),
      Fintype.sum_prod_type]

/-- **3×3/s2 max-pool backward in flat `Vec` space** — the accumulating scatter: each input cell
    collects `dy` from every output whose 3×3 window selects it. ⛔ `maxPool2`'s windows TILE, so
    `maxPoolFlatBack` is a lookup; 3×3/s2 windows OVERLAP, so an input cell can be the argmax of
    up to four outputs and this is a reduction. Spelled as the masked sum the kernel performs,
    which is `maxPool3s2_has_vjp_at3`'s backward reindexed (`maxPool3s2FlatBack_eq_vjp_backward`).
    Found 2026-08 because `r34InputGrad` had been written as the reverse of the 2×2 pool while
    its docstring claimed the committed forward. -/
noncomputable def maxPool3s2FlatBack {c h w : Nat} (x : Tensor3 c (2*h) (2*w)) :
    Vec (c*h*w) → Vec (c*(2*h)*(2*w)) :=
  fun dy idx => ∑ k : Fin (c*h*w), (if maxPool3s2LocalReindex x k = idx then dy k else 0)

/-- **3×3/s2 pool input-VJP leaf tie (smooth point).** `maxPool3s2FlatBack x` IS the certified
    pool input-VJP `(maxPool3s2Flat_has_vjp_at x h_smooth).backward`: the certified backward is the
    triple sum `∑_{co,ho,wo} [σ(co,ho,wo) = idx]·dy(co,ho,wo)`, and this is that sum re-indexed
    row-major (`sum_flat3`). The 3×3/s2 peer of `maxPoolFlatBack_eq_vjp_backward`. -/
theorem maxPool3s2FlatBack_eq_vjp_backward {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (h_smooth : MaxPool3s2Smooth x) :
    maxPool3s2FlatBack x = (maxPool3s2Flat_has_vjp_at x h_smooth).backward := by
  funext dy idx
  show (∑ k : Fin (c*h*w), (if maxPool3s2LocalReindex x k = idx then dy k else 0)) = _
  rw [sum_flat3 (fun k => if maxPool3s2LocalReindex x k = idx then dy k else 0)]
  have hidx : finProdFinEquiv
      (finProdFinEquiv ((finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1,
        (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2),
        (finProdFinEquiv.symm idx).2) = idx := by
    rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
  simp only [maxPool3s2Flat_has_vjp_at, hasVJPAt3_to_hasVJPAt, maxPool3s2_has_vjp_at3,
    Tensor3.unflatten]
  refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl fun ho _ =>
    Finset.sum_congr rfl fun wo _ => ?_
  rw [hidx]
  split <;> simp

/-- ⭐ **The pool VJP at a `Vec` point, with its backward DEFINITIONALLY `maxPool3s2FlatBack`.**
    `maxPool3s2Flat_has_vjp_at` is stated at `Tensor3.flatten x`, and a whole-net chain needs it at
    the stem's `Vec` output. ⛔ Transporting with `▸`/`rwa` would work for the TYPE and leave a
    `backward` field behind an `Eq.mpr` that will not reduce. Building the structure field-by-field
    instead keeps `backward` the leaf itself, which is what lets the whole-net ties
    (`Resnet34BackCertifiedTie`, `ResNet34FullBVJP`'s batched pool) close by `rfl` at this stage
    rather than by a rewrite. -/
noncomputable def maxPool3s2Flat_has_vjp_at_vec {c h w : Nat} (v : Vec (c * (2*h) * (2*w)))
    (h_smooth : MaxPool3s2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))) :
    HasVJPAt (maxPool3s2Flat c h w) v where
  backward := maxPool3s2FlatBack (Tensor3.unflatten v)
  correct := by
    intro dy i
    have hc := (maxPool3s2Flat_has_vjp_at (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))
      h_smooth).correct dy i
    rw [← maxPool3s2FlatBack_eq_vjp_backward _ h_smooth] at hc
    rwa [Tensor3.flatten_unflatten] at hc

/-- The `Vec`-point differentiability companion of `maxPool3s2Flat_has_vjp_at_vec`. -/
theorem maxPool3s2Flat_differentiableAt_vec {c h w : Nat} (v : Vec (c * (2*h) * (2*w)))
    (h_smooth : MaxPool3s2Smooth (Tensor3.unflatten v : Tensor3 c (2*h) (2*w)))
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    DifferentiableAt ℝ (maxPool3s2Flat c h w) v := by
  have h := maxPool3s2Flat_differentiableAt (Tensor3.unflatten v : Tensor3 c (2*h) (2*w))
    h_smooth hc hh hw
  rwa [Tensor3.flatten_unflatten] at h

-- ═════════════════════════════════════════════════
-- § The MLP chain — the three-layer input-gradient backward the PGD apps run
-- ═════════════════════════════════════════════════

/-- The 3-layer MLP input-gradient VJP at a smooth point: `dy ↦ Wᵀ₀·(mask₁ ⊙ Wᵀ₁·(mask₂ ⊙
    Wᵀ₂·dy))`. The certified backward of `dense W₂ ∘ relu ∘ dense W₁ ∘ relu ∘ dense W₀`
    (input gradient), the ReLU kinks read off the fixed sign masks `c₁`/`c₂`. -/
noncomputable def mlpInputGrad {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (c₁ : Fin d₁ → Prop) [DecidablePred c₁] (c₂ : Fin d₂ → Prop) [DecidablePred c₂] :
    Vec d₃ → Vec d₀ :=
  dense (Mat.transpose W₀) 0 ∘ reluMaskBack c₁ ∘ dense (Mat.transpose W₁) 0
    ∘ reluMaskBack c₂ ∘ dense (Mat.transpose W₂) 0

end Proofs
