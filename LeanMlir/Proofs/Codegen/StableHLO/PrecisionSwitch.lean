import LeanMlir.Proofs.Codegen.StableHLO.Basic

/-! # Precision-switched constructors — `XAt bf16 rnd …` is `XBf16 rnd …` or `X …`

Every bf16 op is its f32 peer with one extra leading argument, the rounding `rnd`. `XAt bf16 rnd …`
is the `if bf16 then .XBf16 rnd … else .X …` choice, written once per constructor instead of at
every call site. `pretty` evaluates it, so the emitted text is exactly the chosen branch's.

A leaf on `Basic`: the renderers (`RenderKit` and the `*Render*` modules) and the typed forward
graphs in `Nets/` (`r34IdGraphB`, …) both choose the constructor here, so a render and the graph
its faithfulness theorem is about cannot pick different kinds for the same flag. The renderers
pass `zrnd = id` (`Pretty.lean`), the graphs pass `id`, and `Foundation/Bf16Erasure.lean` states
each switch at `id` equal to its f32 peer (`denOp_convAt_id`, `den_convBackBatchedAt_id`, …). -/

namespace Proofs.StableHLO
/-- `conv`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*h*w) (oc*h*w) :=
  if bf16 then .convBf16 rnd wName bName W bias else .conv wName bName W bias

/-- `convBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convBackBatchedAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    SHlo (N * (oc * h * w)) → SHlo (N * (ic * h * w)) :=
  if bf16 then .convBackBatchedBf16 rnd wName W b else .convBackBatched wName W b

/-- `convStride4`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convStride4At (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*(2*(2*h))*(2*(2*w))) (oc*h*w) :=
  if bf16 then .convStride4Bf16 rnd wName bName W bias else .convStride4 wName bName W bias

/-- `convStride4WeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStride4WeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic*(2*(2*h))*(2*(2*w)))))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc*h*w)) → SHlo (oc*ic*kH*kW) :=
  if bf16 then .convStride4WeightGradBBf16 rnd xName b x W else .convStride4WeightGradB xName b x W

/-- `convStrided`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convStridedAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*(2*h)*(2*w)) (oc*h*w) :=
  if bf16 then .convStridedBf16 rnd wName bName W bias else .convStrided wName bName W bias

/-- `convStridedBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStridedBackBatchedAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    SHlo (N * (oc * h * w)) → SHlo (N * (ic * (2 * h) * (2 * w))) :=
  if bf16 then .convStridedBackBatchedBf16 rnd wName W b else .convStridedBackBatched wName W b

/-- `convStridedWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStridedWeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW) :=
  if bf16 then .convStridedWeightGradBBf16 rnd xName b x W else .convStridedWeightGradB xName b x W

/-- `convStridedXla`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convStridedXlaAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*(2*h)*(2*w)) (oc*h*w) :=
  if bf16 then .convStridedXlaBf16 rnd wName bName W bias else .convStridedXla wName bName W bias

/-- `convStridedXlaWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStridedXlaWeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW) :=
  if bf16 then .convStridedXlaWeightGradBBf16 rnd xName b x W
  else .convStridedXlaWeightGradB xName b x W

/-- `convWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convWeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic * h * w)))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW) :=
  if bf16 then .convWeightGradBBf16 rnd xName b x W else .convWeightGradB xName b x W

/-- `denseRow`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.denseRowAt (bf16 : Bool) {N a c : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Mat a c) (b : Vec c) :
    BatchableOp (N*a) (N*c) :=
  if bf16 then .denseRowBf16 rnd wName bName W b else .denseRow wName bName W b

/-- `denseRowBack`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.denseRowBackAt (bf16 : Bool) {rows a c : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : Mat a c) :
    BatchableOp (rows*c) (rows*a) :=
  if bf16 then .denseRowBackBf16 rnd wName W else .denseRowBack wName W

/-- `depthwise`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.depthwiseAt (bf16 : Bool) {c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    BatchableOp (c*h*w) (c*h*w) :=
  if bf16 then .depthwiseBf16 rnd wName bName W bias else .depthwise wName bName W bias

/-- `depthwiseBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseBackBatchedAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : DepthwiseKernel c kH kW) (b : Vec c) :
    SHlo (N * (c * h * w)) → SHlo (N * (c * h * w)) :=
  if bf16 then .depthwiseBackBatchedBf16 rnd wName W b else .depthwiseBackBatched wName W b

/-- `depthwiseStrided`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.depthwiseStridedAt (bf16 : Bool) {c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    BatchableOp (c*(2*h)*(2*w)) (c*h*w) :=
  if bf16 then .depthwiseStridedBf16 rnd wName bName W bias
  else .depthwiseStrided wName bName W bias

/-- `depthwiseStridedBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedBackBatchedAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : DepthwiseKernel c kH kW) (b : Vec c) :
    SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w))) :=
  if bf16 then .depthwiseStridedBackBatchedBf16 rnd wName W b
  else .depthwiseStridedBackBatched wName W b

/-- `depthwiseStridedWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedWeightGradBAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) :
    SHlo (N * (c * h * w)) → SHlo (c * kH * kW) :=
  if bf16 then .depthwiseStridedWeightGradBBf16 rnd xName b x W
  else .depthwiseStridedWeightGradB xName b x W

/-- `depthwiseStridedXla`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.depthwiseStridedXlaAt (bf16 : Bool) {c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    BatchableOp (c*(2*h)*(2*w)) (c*h*w) :=
  if bf16 then .depthwiseStridedXlaBf16 rnd wName bName W bias
  else .depthwiseStridedXla wName bName W bias

/-- `depthwiseStridedXlaBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedXlaBackBatchedAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : DepthwiseKernel c kH kW) (b : Vec c) :
    SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w))) :=
  if bf16 then .depthwiseStridedXlaBackBatchedBf16 rnd wName W b
  else .depthwiseStridedXlaBackBatched wName W b

/-- `depthwiseStridedXlaWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedXlaWeightGradBAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) :
    SHlo (N * (c * h * w)) → SHlo (c * kH * kW) :=
  if bf16 then .depthwiseStridedXlaWeightGradBBf16 rnd xName b x W
  else .depthwiseStridedXlaWeightGradB xName b x W

/-- `depthwiseWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseWeightGradBAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec c) (x : Vec (N * (c * h * w)))
    (W : DepthwiseKernel c kH kW) :
    SHlo (N * (c * h * w)) → SHlo (c * kH * kW) :=
  if bf16 then .depthwiseWeightGradBBf16 rnd xName b x W else .depthwiseWeightGradB xName b x W

/-- `flatConvF`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.flatConvFAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    SHlo (ic*h*w) → SHlo (oc*h*w) :=
  if bf16 then .flatConvFBf16 rnd wName bName W b else .flatConvF wName bName W b

/-- `matmulFB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.matmulFBAt (bf16 : Bool) {N m k n : Nat}
    (rnd : ℝ → ℝ) :
    SHlo (N*(m*k)) → SHlo (N*(k*n)) → SHlo (N*(m*n)) :=
  if bf16 then .matmulFBBf16 rnd  else .matmulFB

/-- `rowDenseWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.rowDenseWeightGradBAt (bf16 : Bool) {N tk a c : Nat}
    (rnd : ℝ → ℝ) (xName : String) (x : Vec (N*(tk*a))) :
    SHlo (N*(tk*c)) → SHlo (a*c) :=
  if bf16 then .rowDenseWeightGradBBf16 rnd xName x else .rowDenseWeightGradB xName x

end Proofs.StableHLO
