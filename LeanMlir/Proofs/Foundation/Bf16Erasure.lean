import LeanMlir.Proofs.Codegen.StableHLO.Basic
import LeanMlir.Proofs.Codegen.StableHLO.PrecisionSwitch
import LeanMlir.Proofs.Foundation.ParamGradNodes

/-! # Erasure — every batched bf16 kind is its f32 peer at the identity rounding

`flatConvFBf16_id` (`StableHLO.Basic`) says of the per-example CIFAR conv that the bf16 op
"adds ROUNDING and nothing else — no reassociation, no dropped bias, no moved padding". This file
says it of the twenty-five batched kinds the ImageNet renders emit: for each, `den` (or `denOp`)
at `rnd := id` is the f32 constructor's `den`, named against its own peer. The renderers pass
`zrnd = id` (`StableHLO.Pretty`), so these are the equalities the rendered bf16 ASTs satisfy,
and they are what lets a whole-net statement at the f32 nodes be restated at the bf16 artifact
(planning/bf16_tie.md §3).

Each lemma names its own peer. The symmetric and XLA-`SAME` strided kinds have identical types
and emitted shapes and differ only in `denOp` (`Basic.lean`, the `convStridedXla` arm), so a
mismatched pairing fails to prove — which is the check.

| group | kinds | proof |
|---|---|---|
| forward conv / depthwise (7) | `convBf16`, `convStridedBf16`, `convStridedXlaBf16`, `convStride4Bf16`, `depthwiseBf16`, `depthwiseStridedBf16`, `depthwiseStridedXlaBf16` | the bias sits outside the store in the bf16 `den` and inside the conv in the f32 one: `*_bias_split` |
| forward dense / patch (2) | `denseRowBf16`, `patchEmbedBf16` | `rowBiasFlat` after the store vs `dense`'s bias; `patchEmbedFlatBf16 id` is `patchEmbedFlat` |
| `denseRowBackBf16`, `matmulFBBf16` | | `rfl` |
| dgrad (5) | `convBackBatchedBf16`, `convStridedBackBatchedBf16`, `depthwiseBackBatchedBf16`, `depthwiseStridedBackBatchedBf16`, `depthwiseStridedXlaBackBatchedBf16` | `rfl` |
| wgrad (9) | the `Bf16GradNodes` table | `rfl` |

Two bias splits the suite did not have (`flatConvStride2`, `depthwiseStride2FlatXla`) are proved
here beside their siblings' pattern from `ParamGradNodes`.

The last section states the same thing of the renderers' switches (`StableHLO.PrecisionSwitch`):
`denOp (.convAt bf16 id …) = denOp (.conv …)` for either `bf16`, one lemma per switch. A typed
forward graph built on the switches (`r34IdGraphB`, …) is then faithful at either precision by the
f32 proof with these rewrites in front of it — `simp only […, denOp_convAt_id]` before `denOp`,
since on a symbolic `bf16` the `denOp` equations are stuck on the `if` and simp would otherwise
unfold it to a stuck `match`.

Nothing here says how large the rounding is; `rnd` is a binder everywhere else and `id` here.
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.GradNodeB

namespace Proofs.Bf16Fold

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The two missing bias splits
-- ════════════════════════════════════════════════════════════════

private theorem flatChannel_decimateIdx (c h w : Nat) (k : Fin (c * h * w)) :
    flatChannel c (2 * h) (2 * w) (decimateIdx c h w k) = flatChannel c h w k := by
  simp [flatChannel, decimateIdx]

private theorem flatChannel_decimateOddIdx (c h w : Nat) (k : Fin (c * h * w)) :
    flatChannel c (2 * h) (2 * w) (decimateOddIdx c h w k) = flatChannel c h w k := by
  simp [flatChannel, decimateOddIdx]

/-- A symmetric strided conv's bias is a channel broadcast: decimation keeps channels. -/
theorem flatConvStride2_bias_split {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (θ : Vec oc) (y : Vec (ic * (2 * h) * (2 * w))) :
    (flatConvStride2 W θ y : Vec (oc * h * w))
      = fun k => flatConvStride2 W 0 y k + broadcastFlat oc h w θ k := by
  funext k
  simp only [flatConvStride2, Function.comp_apply, decimateFlat]
  rw [flatConv_bias_split W θ y]
  simp only [broadcastFlat, flatChannel_decimateIdx]

/-- An XLA-`SAME` strided depthwise conv's bias is a channel broadcast. -/
theorem depthwiseStride2FlatXla_bias_split {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (θ : Vec c) (y : Vec (c * (2 * h) * (2 * w))) :
    (depthwiseStride2FlatXla W θ y : Vec (c * h * w))
      = fun k => depthwiseStride2FlatXla W 0 y k + broadcastFlat c h w θ k := by
  funext k
  simp only [depthwiseStride2FlatXla, Function.comp_apply, decimateOddFlat]
  rw [depthwiseFlat_bias_split W θ y]
  simp only [broadcastFlat, flatChannel_decimateOddIdx]

-- ════════════════════════════════════════════════════════════════
-- § Forward `BatchableOp` kinds — `denOp` at `id`
-- ════════════════════════════════════════════════════════════════

/-- `.batchOp` lifts a `denOp` equality: the batched node at either precision. -/
theorem den_batchOp_congr {N a b : Nat} {op op' : BatchableOp a b} (h : denOp op = denOp op')
    (e : SHlo (N * a)) :
    den (.batchOp (N := N) op e) = den (.batchOp (N := N) op' e) := by
  simp only [den_batchOp, h]

theorem convBf16_id {ic oc h w kH kW : Nat} (wN bN : String) (W : Kernel4 oc ic kH kW)
    (bias : Vec oc) :
    denOp (.convBf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.conv (h := h) (w := w) wN bN W bias) := by
  funext x i
  show flatConv W 0 x i + Tensor3.flatten (fun o _ _ => bias o) i = flatConv W bias x i
  rw [flatConv_bias_split W bias x]
  rfl

theorem convStridedBf16_id {ic oc h w kH kW : Nat} (wN bN : String) (W : Kernel4 oc ic kH kW)
    (bias : Vec oc) :
    denOp (.convStridedBf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.convStrided (h := h) (w := w) wN bN W bias) := by
  funext x i
  show flatConvStride2 W 0 x i + Tensor3.flatten (fun o _ _ => bias o) i
      = flatConvStride2 W bias x i
  rw [flatConvStride2_bias_split W bias x]
  rfl

theorem convStridedXlaBf16_id {ic oc h w kH kW : Nat} (wN bN : String) (W : Kernel4 oc ic kH kW)
    (bias : Vec oc) :
    denOp (.convStridedXlaBf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.convStridedXla (h := h) (w := w) wN bN W bias) := by
  funext x i
  show flatConvStride2Xla W 0 x i + Tensor3.flatten (fun o _ _ => bias o) i
      = flatConvStride2Xla W bias x i
  rw [flatConvStride2Xla_bias_split W bias x]
  rfl

theorem convStride4Bf16_id {ic oc h w kH kW : Nat} (wN bN : String) (W : Kernel4 oc ic kH kW)
    (bias : Vec oc) :
    denOp (.convStride4Bf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.convStride4 (h := h) (w := w) wN bN W bias) := by
  funext x i
  show flatConvStride4 W 0 x i + Tensor3.flatten (fun o _ _ => bias o) i
      = flatConvStride4 W bias x i
  rw [flatConvStride4_bias_split W bias x]
  rfl

theorem depthwiseBf16_id {c h w kH kW : Nat} (wN bN : String) (W : DepthwiseKernel c kH kW)
    (bias : Vec c) :
    denOp (.depthwiseBf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.depthwise (h := h) (w := w) wN bN W bias) := by
  funext x i
  show depthwiseFlat W 0 x i + Tensor3.flatten (fun cc _ _ => bias cc) i
      = depthwiseFlat W bias x i
  rw [depthwiseFlat_bias_split W bias x]
  rfl

theorem depthwiseStridedBf16_id {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    denOp (.depthwiseStridedBf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.depthwiseStrided (h := h) (w := w) wN bN W bias) := by
  funext x i
  show depthwiseStride2Flat W 0 x i + Tensor3.flatten (fun cc _ _ => bias cc) i
      = depthwiseStride2Flat W bias x i
  rw [depthwiseStride2Flat_bias_split W bias x]
  rfl

theorem depthwiseStridedXlaBf16_id {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    denOp (.depthwiseStridedXlaBf16 (h := h) (w := w) id wN bN W bias)
      = denOp (.depthwiseStridedXla (h := h) (w := w) wN bN W bias) := by
  funext x i
  show depthwiseStride2FlatXla W 0 x i + Tensor3.flatten (fun cc _ _ => bias cc) i
      = depthwiseStride2FlatXla W bias x i
  rw [depthwiseStride2FlatXla_bias_split W bias x]
  rfl

/-- The row-dense forward: the bf16 store sits before the bias (`rowBiasFlat` after it), the f32
    `dense` carries its bias inside; at `id` the two are the same affine map. -/
theorem denseRowBf16_id {N a c : Nat} (wN bN : String) (W : Mat a c) (b : Vec c) :
    denOp (.denseRowBf16 (N := N) id wN bN W b) = denOp (.denseRow (N := N) wN bN W b) := by
  funext x k
  show rowBiasFlat N c b (fun i => rowDenseFlat N a c W (fun _ => 0) x i) k
      = rowDenseFlat N a c W b x k
  simp only [rowBiasFlat, rowDenseFlat, Mat.flatten, Mat.unflatten, Equiv.symm_apply_apply, dense]
  ring

theorem patchEmbedBf16_id {ic H W P N D : Nat} (wN bN cN pN : String)
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D) :
    denOp (.patchEmbedBf16 (H := H) (W := W) id wN bN cN pN Wc bc cls pos)
      = denOp (.patchEmbed (H := H) (W := W) wN bN cN pN Wc bc cls pos) := rfl

theorem denseRowBackBf16_id {rows a c : Nat} (wN : String) (W : Mat a c) :
    denOp (.denseRowBackBf16 (rows := rows) id wN W) = denOp (.denseRowBack (rows := rows) wN W) :=
  rfl

-- ════════════════════════════════════════════════════════════════
-- § `SHlo` kinds — `den` at `id`
-- ════════════════════════════════════════════════════════════════

theorem matmulFBBf16_id {N m k n : Nat} (a : SHlo (N * (m * k))) (b : SHlo (N * (k * n))) :
    den (.matmulFBBf16 id a b) = den (.matmulFB a b) := rfl

theorem convBackBatchedBf16_id {N ic oc h w kH kW : Nat} (wN : String) (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (e : SHlo (N * (oc * h * w))) :
    den (.convBackBatchedBf16 (ic := ic) (h := h) (w := w) id wN W b e)
      = den (.convBackBatched (ic := ic) (h := h) (w := w) wN W b e) := rfl

theorem convStridedBackBatchedBf16_id {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (N * (oc * h * w))) :
    den (.convStridedBackBatchedBf16 (ic := ic) (h := h) (w := w) id wN W b e)
      = den (.convStridedBackBatched (ic := ic) (h := h) (w := w) wN W b e) := rfl

theorem depthwiseBackBatchedBf16_id {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (N * (c * h * w))) :
    den (.depthwiseBackBatchedBf16 (h := h) (w := w) id wN W b e)
      = den (.depthwiseBackBatched (h := h) (w := w) wN W b e) := rfl

theorem depthwiseStridedBackBatchedBf16_id {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (N * (c * h * w))) :
    den (.depthwiseStridedBackBatchedBf16 (h := h) (w := w) id wN W b e)
      = den (.depthwiseStridedBackBatched (h := h) (w := w) wN W b e) := rfl

theorem depthwiseStridedXlaBackBatchedBf16_id {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (N * (c * h * w))) :
    den (.depthwiseStridedXlaBackBatchedBf16 (h := h) (w := w) id wN W b e)
      = den (.depthwiseStridedXlaBackBatched (h := h) (w := w) wN W b e) := rfl

theorem convWeightGradBBf16_id {N ic oc h w kH kW : Nat} (xN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (e : SHlo (N * (oc * h * w))) :
    den (.convWeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.convWeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem convStridedWeightGradBBf16_id {N ic oc h w kH kW : Nat} (xN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (.convStridedWeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.convStridedWeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem convStridedXlaWeightGradBBf16_id {N ic oc h w kH kW : Nat} (xN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (.convStridedXlaWeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.convStridedXlaWeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem convStride4WeightGradBBf16_id {N ic oc h w kH kW : Nat} (xN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (.convStride4WeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.convStride4WeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem depthwiseWeightGradBBf16_id {N c h w kH kW : Nat} (xN : String) (b : Vec c)
    (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (e : SHlo (N * (c * h * w))) :
    den (.depthwiseWeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.depthwiseWeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem depthwiseStridedWeightGradBBf16_id {N c h w kH kW : Nat} (xN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (e : SHlo (N * (c * h * w))) :
    den (.depthwiseStridedWeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.depthwiseStridedWeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem depthwiseStridedXlaWeightGradBBf16_id {N c h w kH kW : Nat} (xN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (e : SHlo (N * (c * h * w))) :
    den (.depthwiseStridedXlaWeightGradBBf16 (h := h) (w := w) id xN b x W e)
      = den (.depthwiseStridedXlaWeightGradB (h := h) (w := w) xN b x W e) := rfl

theorem rowDenseWeightGradBBf16_id {N tk a c : Nat} (xN : String) (x : Vec (N * (tk * a)))
    (e : SHlo (N * (tk * c))) :
    den (.rowDenseWeightGradBBf16 (a := a) id xN x e) = den (.rowDenseWeightGradB (a := a) xN x e) :=
  rfl

theorem patchEmbedWeightGradBBf16_id {N ic H W P tk D : Nat} (xN : String)
    (x : Vec (N * (ic * H * W))) (e : SHlo (N * ((tk + 1) * D))) :
    den (.patchEmbedWeightGradBBf16 (P := P) id xN x e)
      = den (.patchEmbedWeightGradB (P := P) xN x e) := rfl

-- ════════════════════════════════════════════════════════════════
-- § The renderers' switches at `id` — either `bf16`, the f32 peer
--   `cases bf16`: `false` is the f32 branch by reduction, `true` the erasure above.
-- ════════════════════════════════════════════════════════════════

theorem denOp_convAt_id (bf16 : Bool) {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    denOp (BatchableOp.convAt bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.conv (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact convBf16_id wN bN W b

theorem denOp_convStridedAt_id (bf16 : Bool) {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    denOp (BatchableOp.convStridedAt bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.convStrided (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact convStridedBf16_id wN bN W b

theorem denOp_convStridedXlaAt_id (bf16 : Bool) {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    denOp (BatchableOp.convStridedXlaAt bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.convStridedXla (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact convStridedXlaBf16_id wN bN W b

theorem denOp_convStride4At_id (bf16 : Bool) {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    denOp (BatchableOp.convStride4At bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.convStride4 (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact convStride4Bf16_id wN bN W b

theorem denOp_depthwiseAt_id (bf16 : Bool) {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    denOp (BatchableOp.depthwiseAt bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.depthwise (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact depthwiseBf16_id wN bN W b

theorem denOp_depthwiseStridedAt_id (bf16 : Bool) {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    denOp (BatchableOp.depthwiseStridedAt bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.depthwiseStrided (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact depthwiseStridedBf16_id wN bN W b

theorem denOp_depthwiseStridedXlaAt_id (bf16 : Bool) {c h w kH kW : Nat} (wN bN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    denOp (BatchableOp.depthwiseStridedXlaAt bf16 (h := h) (w := w) id wN bN W b)
      = denOp (.depthwiseStridedXla (h := h) (w := w) wN bN W b) := by
  cases bf16
  · rfl
  · exact depthwiseStridedXlaBf16_id wN bN W b

theorem denOp_denseRowAt_id (bf16 : Bool) {N a c : Nat} (wN bN : String) (W : Mat a c)
    (b : Vec c) :
    denOp (BatchableOp.denseRowAt bf16 (N := N) id wN bN W b)
      = denOp (.denseRow (N := N) wN bN W b) := by
  cases bf16
  · rfl
  · exact denseRowBf16_id wN bN W b

theorem denOp_denseRowBackAt_id (bf16 : Bool) {rows a c : Nat} (wN : String) (W : Mat a c) :
    denOp (BatchableOp.denseRowBackAt bf16 (rows := rows) id wN W)
      = denOp (.denseRowBack (rows := rows) wN W) := by
  cases bf16
  · rfl
  · exact denseRowBackBf16_id wN W

theorem den_flatConvFAt_id (bf16 : Bool) {ic oc h w kH kW : Nat} (wN bN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (ic * h * w)) :
    den (SHlo.flatConvFAt bf16 (h := h) (w := w) id wN bN W b e)
      = den (.flatConvF (h := h) (w := w) wN bN W b e) := by
  cases bf16
  · rfl
  · exact flatConvFBf16_id wN bN W b e

theorem den_matmulFBAt_id (bf16 : Bool) {N m k n : Nat} (a : SHlo (N * (m * k)))
    (b : SHlo (N * (k * n))) :
    den (SHlo.matmulFBAt bf16 id a b) = den (.matmulFB a b) := by
  cases bf16
  · rfl
  · exact matmulFBBf16_id a b

theorem den_convBackBatchedAt_id (bf16 : Bool) {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convBackBatchedAt bf16 (ic := ic) (h := h) (w := w) id wN W b e)
      = den (.convBackBatched (ic := ic) (h := h) (w := w) wN W b e) := by
  cases bf16
  · rfl
  · exact convBackBatchedBf16_id wN W b e

theorem den_convStridedBackBatchedAt_id (bf16 : Bool) {N ic oc h w kH kW : Nat} (wN : String)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convStridedBackBatchedAt bf16 (ic := ic) (h := h) (w := w) id wN W b e)
      = den (.convStridedBackBatched (ic := ic) (h := h) (w := w) wN W b e) := by
  cases bf16
  · rfl
  · exact convStridedBackBatchedBf16_id wN W b e

theorem den_depthwiseBackBatchedAt_id (bf16 : Bool) {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseBackBatchedAt bf16 (h := h) (w := w) id wN W b e)
      = den (.depthwiseBackBatched (h := h) (w := w) wN W b e) := by
  cases bf16
  · rfl
  · exact depthwiseBackBatchedBf16_id wN W b e

theorem den_depthwiseStridedBackBatchedAt_id (bf16 : Bool) {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedBackBatchedAt bf16 (h := h) (w := w) id wN W b e)
      = den (.depthwiseStridedBackBatched (h := h) (w := w) wN W b e) := by
  cases bf16
  · rfl
  · exact depthwiseStridedBackBatchedBf16_id wN W b e

theorem den_depthwiseStridedXlaBackBatchedAt_id (bf16 : Bool) {N c h w kH kW : Nat}
    (wN : String) (W : DepthwiseKernel c kH kW) (b : Vec c) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedXlaBackBatchedAt bf16 (h := h) (w := w) id wN W b e)
      = den (.depthwiseStridedXlaBackBatched (h := h) (w := w) wN W b e) := by
  cases bf16
  · rfl
  · exact depthwiseStridedXlaBackBatchedBf16_id wN W b e

theorem den_convWeightGradBAt_id (bf16 : Bool) {N ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convWeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.convWeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact convWeightGradBBf16_id xN b x W e

theorem den_convStridedWeightGradBAt_id (bf16 : Bool) {N ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convStridedWeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.convStridedWeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact convStridedWeightGradBBf16_id xN b x W e

theorem den_convStridedXlaWeightGradBAt_id (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (xN : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convStridedXlaWeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.convStridedXlaWeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact convStridedXlaWeightGradBBf16_id xN b x W e

theorem den_convStride4WeightGradBAt_id (bf16 : Bool) {N ic oc h w kH kW : Nat} (xN : String)
    (b : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (W : Kernel4 oc ic kH kW)
    (e : SHlo (N * (oc * h * w))) :
    den (SHlo.convStride4WeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.convStride4WeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact convStride4WeightGradBBf16_id xN b x W e

theorem den_depthwiseWeightGradBAt_id (bf16 : Bool) {N c h w kH kW : Nat} (xN : String)
    (b : Vec c) (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseWeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.depthwiseWeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact depthwiseWeightGradBBf16_id xN b x W e

theorem den_depthwiseStridedWeightGradBAt_id (bf16 : Bool) {N c h w kH kW : Nat} (xN : String)
    (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedWeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.depthwiseStridedWeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact depthwiseStridedWeightGradBBf16_id xN b x W e

theorem den_depthwiseStridedXlaWeightGradBAt_id (bf16 : Bool) {N c h w kH kW : Nat}
    (xN : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedXlaWeightGradBAt bf16 (h := h) (w := w) id xN b x W e)
      = den (.depthwiseStridedXlaWeightGradB (h := h) (w := w) xN b x W e) := by
  cases bf16
  · rfl
  · exact depthwiseStridedXlaWeightGradBBf16_id xN b x W e

theorem den_rowDenseWeightGradBAt_id (bf16 : Bool) {N tk a c : Nat} (xN : String)
    (x : Vec (N * (tk * a))) (e : SHlo (N * (tk * c))) :
    den (SHlo.rowDenseWeightGradBAt bf16 (a := a) id xN x e)
      = den (.rowDenseWeightGradB (a := a) xN x e) := by
  cases bf16
  · rfl
  · exact rowDenseWeightGradBBf16_id xN x e

end Proofs.Bf16Fold
